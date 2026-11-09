/*
 * ds_engine.c — generation on top of the DeepSeek-V4.1 forward pass.
 *
 * Shape of a run, identical to the GLM engine's so the server sees one
 * contract:
 *
 *   prefill  the prompt in chunks of DS_CHUNK positions; the last chunk's
 *            logits already predict the token after the last prompt token
 *   decode   one forward per token, pick the next token (argmax by default,
 *            sampled when the checkpoint or the request asks for it), stop on
 *            EOS or budget
 *
 * Everything expensive here is the forward pass, so this file is only
 * bookkeeping: chunking, interrupts, the sink callback, timing and the
 * /metrics counters the server scrapes mid-stream.
 */
#include "ds_engine.h"

#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "log.h"
#include "mem.h"      /* cwick_gib, cwick_process_footprint_bytes */
#include "metrics.h"
#include "sample.h"
#include "util.h"

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1e3 + (double)tv.tv_usec / 1e3;
}

static bool ds_stopped(const struct ds_ctx *cx) {
    if (cx->stop && *cx->stop) return true;
    if (cx->abort_poll && cx->abort_poll(cx->abort_poll_user)) return true;
    return false;
}

static bool ds_is_eos(const cwick_model_spec *spec, uint32_t tok) {
    for (int i = 0; i < spec->n_eos; i++)
        if ((int64_t)tok == spec->eos_ids[i]) return true;
    return (int64_t)tok == spec->eos_token_id;
}

/* The decode loop picks through sample.c, not an argmax: the checkpoint's
 * generation_config.json (and the request) decide whether that means argmax
 * (temperature <= 0) or a draw. */

/* How the progress line should label the current step: plain "decode" until
 * the server's segment machine has classified a token, then
 * "decode(thinking)" / "decode(content)" / "decode(tools)". */
static const char *ds_phase_label(const struct ds_ctx *cx, char *buf, size_t cap) {
    if (cx && cx->decode_phase)
        snprintf(buf, cap, "decode(%s)", cx->decode_phase);
    else
        snprintf(buf, cap, "decode");
    return buf;
}

/* The draw seed. CWICK_SEED pins it (same seed + same logits -> same tokens);
 * without it every run differs, which is what a request that sent no seed
 * should get. Cached: one getenv per process, not per step. */
static uint64_t ds_sampler_seed(void) {
    static uint64_t cached = 0;
    static bool read = false;
    if (!read) {
        read = true;
        const char *e = getenv("CWICK_SEED");
        if (e && *e) cached = strtoull(e, NULL, 0);
    }
    return cached;
}

/* How often a decode progress line is due. Same knob the GLM path uses, so
 * one environment variable tunes both backends. */
static double ds_progress_min_ms(void) {
    static double cached = -1.0;
    if (cached < 0.0) {
        const char *e = getenv("CWICK_PROGRESS_MIN_MS");
        double v = e ? atof(e) : 0.0;
        cached = v > 0.0 ? v : 5000.0;
    }
    return cached;
}

int ds_engine_open(ds_engine *de, const char *root, int64_t capacity,
                   cwick_error *err) {
    memset(de, 0, sizeof(*de));
    de->capacity = capacity > 0 ? capacity : 4096;
    if (ds_model_open(&de->m, root, (int)de->capacity, err) != 0) return -1;
    de->logits = cwick_xmalloc(sizeof(float) * (size_t)de->m.cfg.vocab_size);
    de->pos = 0;
    de->opened = 1;
    cwick_sampler_cfg_default(&de->smp);   /* greedy until told otherwise */
    /* 0 is a legal budget (no thinking at all), so the disabled state has to
     * be an explicit -1. `de` comes from cwick_xmalloc with no memset, and a
     * zeroed ctx would read as "close the think block before the first
     * token" -- which is exactly what an unarmed CLI run did before this was
     * initialised. Same convention as cwick_engine_new's. */
    de->ctx.think_budget = -1;
    de->ctx.think_open_id = -1;
    de->ctx.think_close_id = -1;
    de->ctx.think_open_at_start = false;
    return 0;
}

void ds_engine_close(ds_engine *de) {
    if (!de || !de->opened) return;
    free(de->logits);
    de->logits = NULL;
    ds_model_close(&de->m);
    de->opened = 0;
}

cwick_status ds_engine_generate(ds_engine *de, const uint32_t *prompt,
                                int64_t prompt_len, int64_t max_new,
                                bool keep_prompt, cwick_token_sink sink,
                                void *user, cwick_gen_result *result,
                                cwick_error *err) {
    cwick_error_clear(err);
    if (!de || !de->opened) {
        cwick_error_set(err, CWICK_ERR_STATE, "ds engine is not open");
        return CWICK_ERR_STATE;
    }
    const struct ds_ctx *cx = &de->ctx;
    const cwick_model_spec *spec = cx ? cx->spec : NULL;
    const char *tag = (cx && cx->tag) ? cx->tag : "gen";
    ds_config *c = &de->m.cfg;

    /* A new turn starts before the segment machine has classified anything;
     * without this the first progress lines of this request would inherit the
     * phase the previous one ended in. */
    de->ctx.decode_phase = NULL;
    de->next_checkpoint =
        de->checkpoint_block > 0 ? de->checkpoint_block : 0;

    /* de->pos is the restored prefix length when the server replayed one, so
     * the tail plus the budget is measured from there -- otherwise a 100K
     * prefix plus a 2K tail would pass a check made only of the tail.
     *
     * `max_new` is a ceiling, not a demand, so a budget that overruns the
     * window is clamped rather than rejected (see cwick_gen_window): the
     * model stops at EOS long before either number, and a 400 for "you asked
     * for more than fits" turned ordinary requests into errors. */
    int64_t total_new = max_new;   /* 0 = capacity ceiling: no denominator */
    {
        const int64_t want = max_new;
        cwick_status wst = cwick_gen_window(de->pos, prompt_len, &max_new,
                                            de->capacity, err);
        if (wst != CWICK_OK) return wst;
        if (max_new != want) {
            total_new = 0;
            CWICK_INFO(tag, "max_tokens %lld clamped to %lld: prompt %lld at "
                       "position %lld leaves %lld of %lld positions",
                       (long long)want, (long long)max_new,
                       (long long)prompt_len, (long long)de->pos,
                       (long long)max_new, (long long)de->capacity);
        }
    }

    int32_t *ids = cwick_xmalloc((size_t)(prompt_len > 0 ? prompt_len : 1) *
                                 sizeof(int32_t));
    for (int64_t i = 0; i < prompt_len; i++) ids[i] = (int32_t)prompt[i];

    /* ---- prefill: chunked, the last chunk's logits predict token 1 ---- */
    double prefill_ms = 0.0;
    int64_t done = 0;
    ds_prefill_progress_begin(prompt_len, c->n_layers);
    while (done < prompt_len) {
        if (cx && ds_stopped(cx)) {
            ds_prefill_progress_end();
            free(ids);
            cwick_error_set(err, CWICK_ERR_STATE, "interrupted during prefill");
            return CWICK_ERR_STATE;
        }
        int n = (int)(prompt_len - done);
        if (n > DS_CHUNK) n = DS_CHUNK;
        double t0 = now_ms();
        if (ds_model_forward(&de->m, ids + done, n, de->pos, de->logits,
                             err) != 0) {
            ds_prefill_progress_end();
            free(ids);
            return CWICK_ERR_STATE;
        }
        prefill_ms += now_ms() - t0;
        de->pos += n;
        done += n;
        ds_prefill_progress_done(done);
        /* Long prompts leave a reusable head behind every block: at ~1.4
         * tok/s a 30K-token prefill is hours, and losing it to a restart is
         * the difference between a slow first turn and an unusable one. */
        if (de->on_prefill_checkpoint && de->checkpoint_block > 0 &&
            de->pos >= de->next_checkpoint) {
            de->on_prefill_checkpoint(de->on_prefill_checkpoint_user, de->pos);
            de->next_checkpoint = de->pos + de->checkpoint_block;
        }
        /* A long prefill fills the pin cache too, and it is minutes of work
         * per chunk: check the plan here as well as in the decode loop. */
        if (de->m.stream)
            cwick_stream_steer_memory(de->m.stream, de->m.rss_target, done,
                                      "mem");
    }
    if (prompt_len == 0) {
        /* Nothing to condition on: the caller still gets a first token, so
         * seed the stream the way a 1-token prompt would. */
        double t0 = now_ms();
        if (ds_model_forward(&de->m, ids, 1, de->pos, de->logits, err) != 0) {
            ds_prefill_progress_end();
            free(ids);
            return CWICK_ERR_STATE;
        }
        prefill_ms += now_ms() - t0;
        de->pos += 1;
    }
    ds_prefill_progress_end();

    /* The state now equals "prompt prefilled", which is exactly what a prefix
     * snapshot must capture. One decode step later it has moved on, so this is
     * the only point where it can be taken. */
    if (de->on_prefill_done)
        de->on_prefill_done(de->on_prefill_done_user, de->pos);

    int64_t collected = keep_prompt ? prompt_len : 0;
    uint32_t *tokens = cwick_xmalloc((size_t)(collected + max_new + 1) *
                                     sizeof(uint32_t));
    if (keep_prompt && prompt_len > 0)
        memcpy(tokens, prompt, (size_t)prompt_len * sizeof(uint32_t));

    int64_t gen = 0, decode_steps = 0;
    bool hit_eos = false, interrupted = false, sink_stopped = false;
    bool loop_stopped = false;
    double decode_t0 = now_ms();
    double last_rep = decode_t0;   /* wall clock of the last progress line */
    int64_t steps_accounted = 0, gen_accounted = 0;

    /* Thinking budget: the ceiling this turn runs under, clamped to what the
     * output budget can honour, plus the block state it acts on. Same policy
     * and same reserve as the GLM path (engine.c's L4 block) because the two
     * backends must cap a 高 request identically: a ceiling at or above
     * max_new can never fire, and the turn it exists to save is exactly the
     * one that dies at the length limit. A live request measured what happens
     * without this -- 16000 thinking tokens against a configured 8192, still
     * going, holding the only generation slot. */
    const int64_t think_open_id = cx ? cx->think_open_id : -1;
    const int64_t think_close_id = cx ? cx->think_close_id : -1;
    int64_t think_budget = cx ? cx->think_budget : -1;
    if (think_budget >= max_new) {
        int64_t room = max_new - CWICK_THINK_RESERVE;
        if (room < 0) room = max_new / 2;
        if (cx && cx->verbose)
            CWICK_VERB(tag, "thinking budget %lld clamped to %lld: %lld tokens "
                       "of output leave no room to answer",
                       (long long)(cx->think_budget), (long long)room,
                       (long long)max_new);
        think_budget = room;
    }
    bool think_open = cx ? cx->think_open_at_start : false;
    bool think_closed = false;
    bool think_budget_hit = false;   /* we closed the think block ourselves */

    /* One sampler per generation: the repetition counts belong to this
     * sequence. Same config the GLM path got, copied in by the owner. */
    cwick_sampler smp;
    cwick_sampler_init(&smp, &de->smp, c->vocab_size, ds_sampler_seed());
    if (smp.counts)
        for (int64_t i = 0; i < prompt_len; i++) cwick_sampler_prime(&smp, prompt[i]);

    for (int64_t k = 0; k < max_new; k++) {
        if (cx && ds_stopped(cx)) { interrupted = true; break; }
        /* Age the expert pin set once per decode step, the same cadence the
         * GLM path uses: heat is per-pass, so a topic shift frees slots for
         * the experts the new conversation actually reaches. */
        if (de->m.stream) {
            cwick_stream_expert_decay(de->m.stream);
            /* The steering valve this loop never had. Without it the pin cache
             * fills to its cap and stays there: the only trim in the tree was
             * engine.c's, behind a GLM-only guard, so a DeepSeek decode ran
             * 20+ GiB over the plan and macOS answered by compressing the pin
             * cache itself -- the cache stayed 99.99% full and 75% of the
             * lookups still "hit", but every hit was a decompression fault and
             * the step time tripled. */
            cwick_stream_steer_memory(de->m.stream, de->m.rss_target, k, "mem");
        }
        /* Thinking budget. Reaching the ceiling is not a reason to stop the
         * turn -- it is a reason to stop *thinking*. This loop writes the
         * close tag itself, once, into both the output and the context: the
         * sink sees it (so the wire carries a closed think block and the
         * server's segment machine reclassifies the rest as content) and the
         * forward pass consumes it (so the model, which has just written its
         * own delimiter, answers instead of continuing the thought). Both
         * have to happen in the same step, and only the decode loop can feed
         * a token it did not sample. Mirrors engine.c. */
        if (think_budget >= 0 && !think_closed) {
            cwick_think_action act =
                cwick_think_plan(gen, think_budget, think_open, false);
            if (act != CWICK_THINK_NONE) {
                uint32_t inj[2];
                int n_inj = 0;
                if (act == CWICK_THINK_OPEN_CLOSE)
                    inj[n_inj++] = (uint32_t)think_open_id;
                inj[n_inj++] = (uint32_t)think_close_id;
                bool aborted = false;
                int written = 0;
                for (int t = 0; t < n_inj && gen < max_new; t++) {
                    uint32_t ct = inj[t];
                    tokens[collected + gen++] = ct;
                    written++;
                    cwick_sampler_accept(&smp, ct);   /* it is now history */
                    if ((int64_t)ct == think_open_id) think_open = true;
                    if ((int64_t)ct == think_close_id) think_closed = true;
                    if (sink && !sink(user, ct, gen - 1)) {
                        sink_stopped = true;
                        aborted = true;
                        break;
                    }
                    if (k + 1 < max_new) {
                        ids[0] = (int32_t)ct;
                        if (ds_model_forward(&de->m, ids, 1, de->pos, de->logits,
                                             err) != 0) {
                            cwick_sampler_free(&smp);
                            free(tokens);
                            free(ids);
                            return CWICK_ERR_STATE;
                        }
                        de->pos++;
                        decode_steps++;
                    }
                }
                if (written) {
                    think_budget_hit = true;
                    CWICK_INFO(tag, "thinking budget: %lld tokens reached — "
                              "wrote %s so the turn answers instead of "
                              "thinking", (long long)think_budget,
                              act == CWICK_THINK_OPEN_CLOSE ? "</think>"
                                                            : "</think>");
                }
                if (aborted) break;
                if (gen >= max_new) break;   /* an injected pair may have used it */
                continue;   /* sample the first content token next */
            }
        }
        uint32_t next = cwick_sampler_pick(&smp, de->logits, c->vocab_size);
        if (spec && ds_is_eos(spec, next)) { hit_eos = true; break; }
        tokens[collected + gen++] = next;
        cwick_sampler_accept(&smp, next);
        /* Keep the budget's view of the block honest: the model closes its own
         * thought in the normal case, and that must silence the ceiling. */
        if ((int64_t)next == think_open_id) think_open = true;
        if ((int64_t)next == think_close_id) think_closed = true;
        /* Loop guard: a repetition attractor would otherwise run to max_new
         * (one request burned its whole 5282-token budget on a repeated
         * label). Same rule the GLM path uses. */
        int lp = 0;
        int64_t lr = 0;
        if (cwick_sampler_looped(&smp, &lp, &lr)) {
            gen--;   /* the token that closed the loop is framing, not content */
            CWICK_INFO(tag, "loop guard: period-%d block repeated %lld times "
                      "(%lld tokens) — stopping after %lld generated; "
                      "CWICK_LOOP_DETECT=0 disables", lp,
                      (long long)(lr / (lp > 0 ? lp : 1) + 1), (long long)lr,
                      (long long)gen);
            loop_stopped = true;
            break;
        }
        if (sink && !sink(user, next, gen - 1)) { sink_stopped = true; break; }

        double step_t0 = now_ms();
        bool did_forward = false;
        if (k + 1 < max_new) {
            ids[0] = (int32_t)next;
            if (ds_model_forward(&de->m, ids, 1, de->pos, de->logits,
                                 err) != 0) {
                cwick_sampler_free(&smp);
                free(tokens);
                free(ids);
                return CWICK_ERR_STATE;
            }
            de->pos++;
            decode_steps++;
            did_forward = true;
        }
        double step_ms = now_ms() - step_t0;
        char phase_lab[32];
        ds_phase_label(cx, phase_lab, sizeof(phase_lab));
        if (cx && cx->verbose && did_forward) {
            if (total_new > 0)
                CWICK_VERB(tag, "%s forward %lld/%lld %.0f ms (%.3f tok/s)",
                           phase_lab, (long long)decode_steps,
                           (long long)(total_new - 1), step_ms,
                           step_ms > 0 ? 1000.0 / step_ms : 0.0);
            else
                CWICK_VERB(tag, "%s forward %lld %.0f ms (%.3f tok/s, context "
                           "cap)", phase_lab, (long long)decode_steps, step_ms,
                           step_ms > 0 ? 1000.0 / step_ms : 0.0);
        }
        /* A long answer is minutes of otherwise silent decode. One line per
         * throttle window, labelled with the phase the server says we are in,
         * so the operator can tell "still thinking" from "writing the answer"
         * from "emit a tool call" without waiting for the request to end. */
        if (did_forward && (now_ms() - last_rep) >= ds_progress_min_ms()) {
            double el = now_ms() - decode_t0;
            double per = decode_steps > 0 ? el / (double)decode_steps : 0.0;
            last_rep = now_ms();
            if (total_new > 0) {
                double eta = per > 0.0
                                 ? (double)(total_new - 1 - decode_steps) * per
                                 : 0.0;
                CWICK_INFO(tag, "%s %lld/%lld tokens (%.1f%%) — %.0f ms/step, "
                           "%.3f tok/s, eta %.0fs", phase_lab, (long long)gen,
                           (long long)total_new,
                           100.0 * (double)gen / (double)total_new,
                           per, per > 0.0 ? 1000.0 / per : 0.0, eta / 1000.0);
            } else {
                /* A window-derived budget is a ceiling, not a length. */
                CWICK_INFO(tag, "%s %lld tokens (context cap) — %.0f ms/step, "
                           "%.3f tok/s", phase_lab, (long long)gen, per,
                           per > 0.0 ? 1000.0 / per : 0.0);
            }
        }
        if (did_forward) {
            cwick_metrics_observe_decode_step(step_ms / 1000.0);
            cwick_metrics_observe_iteration_tokens(1);
            if ((decode_steps & 15) == 0) {
                cwick_metrics_add_decode_steps(decode_steps - steps_accounted);
                steps_accounted = decode_steps;
                if (gen > gen_accounted) {
                    cwick_metrics_add_generation_tokens(gen - gen_accounted);
                    gen_accounted = gen;
                }
            }
        }
    }
    double decode_ms = now_ms() - decode_t0;

    if (decode_steps > steps_accounted)
        cwick_metrics_add_decode_steps(decode_steps - steps_accounted);
    if (gen > gen_accounted)
        cwick_metrics_add_generation_tokens(gen - gen_accounted);

    {
        char pf_tps[32], dc_tps[32];
        if (prefill_ms > 0.0)
            snprintf(pf_tps, sizeof(pf_tps), "%.3f",
                     (double)prompt_len * 1000.0 / prefill_ms);
        else
            snprintf(pf_tps, sizeof(pf_tps), "n/a");
        if (decode_steps > 0 && decode_ms > 0.0)
            snprintf(dc_tps, sizeof(dc_tps), "%.3f",
                     (double)decode_steps * 1000.0 / decode_ms);
        else
            snprintf(dc_tps, sizeof(dc_tps), "n/a");
        /* Same fields as the GLM line. Without them a slow step gave no way to
         * tell an SSD miss storm from memory compression after the fact --
         * exactly the question that came up when a live request measured
         * 6.6 s/step and the log had nothing but the step time. */
        uint64_t bs = 0, rc = 0, ch = 0, cm = 0;
        if (de->m.stream) cwick_stream_stats(de->m.stream, &bs, &rc, &ch, &cm);
        CWICK_INFO(tag, "prompt=%lld gen=%lld finish=%s prefill=%.0fms/%s "
                   "tok/s decode=%.0fms/%lld steps %s tok/s "
                   "cache=%.1f%%(%llu/%llu) streamed=%.2fGiB mem=%.2fGiB",
                   (long long)prompt_len, (long long)gen,
                   interrupted ? "interrupted"
                               : loop_stopped ? "loop"
                               : ((hit_eos || sink_stopped) ? "stop" : "length"),
                   prefill_ms, pf_tps, decode_ms, (long long)decode_steps,
                   dc_tps,
                   (ch + cm) ? 100.0 * (double)ch / (double)(ch + cm) : 0.0,
                   (unsigned long long)ch, (unsigned long long)cm,
                   cwick_gib(bs), cwick_gib(cwick_process_footprint_bytes()));
    }

    result->tokens = tokens;
    result->n_tokens = collected + gen;
    result->stop_reason_length =
        !hit_eos && !interrupted && !sink_stopped && !loop_stopped &&
        gen == max_new;
    result->prefill_ms = prefill_ms;
    result->decode_ms = decode_ms;
    result->decode_steps = decode_steps;
    result->interrupted = interrupted;
    result->sink_stopped = sink_stopped;
    result->loop_stopped = loop_stopped;
    result->think_budget_hit = think_budget_hit;
    result->think_budget_applied = think_budget;
    cwick_sampler_free(&smp);
    free(ids);
    if (interrupted) {
        cwick_error_set(err, CWICK_ERR_STATE, "generation interrupted");
        return CWICK_ERR_STATE;
    }
    return CWICK_OK;
}
