/*
 * mm_engine.c — generation on top of the MiniMax-M3.1-Flash forward pass.
 *
 * The shape of a run is the DeepSeek backend's, deliberately: prefill in
 * chunks, then one forward per token, with the same sampler, the same
 * thinking-budget policy, the same loop guard and the same progress lines.
 * What differs is only what the model requires -- a forward pass that cannot
 * be interrupted mid-chunk without losing an hour of expert reads is the main
 * reason the chunk size is defined in mm.h rather than here.
 */
#include "mm_engine.h"

#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "log.h"
#include "mem.h"
#include "metrics.h"
#include "sample.h"
#include "util.h"

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1e3 + (double)tv.tv_usec / 1e3;
}

static bool mm_stopped(const struct mm_ctx *cx) {
    if (cx->stop && *cx->stop) return true;
    if (cx->abort_poll && cx->abort_poll(cx->abort_poll_user)) return true;
    return false;
}

static bool mm_is_eos(const cwick_model_spec *spec, uint32_t tok) {
    for (int i = 0; i < spec->n_eos; i++)
        if ((int64_t)tok == spec->eos_ids[i]) return true;
    return (int64_t)tok == spec->eos_token_id;
}

static const char *mm_phase_label(const struct mm_ctx *cx, char *buf, size_t cap) {
    if (cx && cx->decode_phase)
        snprintf(buf, cap, "decode(%s)", cx->decode_phase);
    else
        snprintf(buf, cap, "decode");
    return buf;
}

static uint64_t mm_sampler_seed(void) {
    static uint64_t cached = 0;
    static bool read = false;
    if (!read) {
        read = true;
        const char *e = getenv("CWICK_SEED");
        if (e && *e) cached = strtoull(e, NULL, 0);
    }
    return cached;
}

static double mm_progress_min_ms(void) {
    static double cached = -1.0;
    if (cached < 0.0) {
        const char *e = getenv("CWICK_PROGRESS_MIN_MS");
        double v = e ? atof(e) : 0.0;
        cached = v > 0.0 ? v : 5000.0;
    }
    return cached;
}

int mm_engine_open(mm_engine *me, const char *root, int64_t capacity,
                   cwick_error *err) {
    memset(me, 0, sizeof(*me));
    me->capacity = capacity > 0 ? capacity : 4096;
    if (mm_model_open(&me->m, root, (int)me->capacity, err) != 0) return -1;
    me->logits = cwick_xmalloc(sizeof(float) * (size_t)me->m.cfg.vocab_size);
    me->pos = 0;
    me->opened = 1;
    cwick_sampler_cfg_default(&me->smp);
    /* 0 is a legal budget ("do not think"), so disabled has to be -1. */
    me->ctx.think_budget = -1;
    me->ctx.think_open_id = -1;
    me->ctx.think_close_id = -1;
    me->ctx.think_open_at_start = false;
    return 0;
}

void mm_engine_close(mm_engine *me) {
    if (!me || !me->opened) return;
    free(me->logits);
    me->logits = NULL;
    free(me->spec_logits);
    free(me->spec_cand);
    free(me->spec_draft);
    free(me->spec_ids);
    me->spec_logits = me->spec_cand = NULL;
    me->spec_draft = NULL;
    me->spec_ids = NULL;
    me->spec_n = 0;
    mm_model_close(&me->m);
    me->opened = 0;
}

/* ------------------------------------------------------------------ */
/* speculative drafting                                                */
/* ------------------------------------------------------------------ */

/* Candidate continuations taken from the token stream itself: look for the most
 * recent earlier occurrence of the trailing k-gram and copy what followed it.
 *
 * This is deliberately a *zero-weight* drafter. What it exists to measure is
 * the other half of speculation -- what it costs to verify a block, and whether
 * the expert-union penalty of running 8 positions through a memory-bound MoE
 * leaves anything on the table. Those numbers decide whether a real drafter is
 * worth porting; they do not change with the quality of the drafts, and a
 * drafter that needs no weights gets them today.
 *
 * k walks down from 4 to 1, so a phrase repeated at a shorter distance is still
 * found; the scan is backwards because the nearest occurrence is the one whose
 * continuation is most likely to repeat. */
int mm_draft_lookup(const uint32_t *h, int64_t n, int nmax,
                    uint32_t *out) {
    if (!h || n < 2 || nmax <= 0) return 0;
    for (int k = 4; k >= 1; k--) {
        if (n < (int64_t)k + 1) continue;
        for (int64_t j = n - 2; j >= k - 1; j--) {
            int hit = 1;
            for (int i = 0; i < k; i++)
                if (h[j - i] != h[n - 1 - i]) { hit = 0; break; }
            if (!hit) continue;
            int nd = 0;
            for (int64_t p = j + 1; p < n && nd < nmax; p++) out[nd++] = h[p];
            if (nd > 0) return nd;
        }
    }
    return 0;
}

cwick_status mm_engine_generate(mm_engine *me, const uint32_t *prompt,
                                int64_t prompt_len, int64_t max_new,
                                bool keep_prompt, cwick_token_sink sink,
                                void *user, cwick_gen_result *result,
                                cwick_error *err) {
    cwick_error_clear(err);
    if (!me || !me->opened) {
        cwick_error_set(err, CWICK_ERR_STATE, "mm engine is not open");
        return CWICK_ERR_STATE;
    }
    const struct mm_ctx *cx = &me->ctx;
    const cwick_model_spec *spec = cx ? cx->spec : NULL;
    const char *tag = (cx && cx->tag) ? cx->tag : "gen";
    mm_config *c = &me->m.cfg;

    me->ctx.decode_phase = NULL;
    me->next_checkpoint = me->checkpoint_block > 0 ? me->checkpoint_block : 0;

    /* max_new is a ceiling, not a demand. A client that sends
     * max_tokens=32768 against a 32768-position server with an 8K prompt is
     * asking for "as much as fits", and the model stops at EOS long before
     * the ceiling -- so clamping is the honest answer and a 400 was not. Only
     * a prompt that does not fit at all is a client error (cwick_gen_window),
     * because its fix (shorter prompt / larger --context) is a different one
     * from "lower max_tokens", and the old message conflated the two. */
    int64_t total_new = max_new;   /* 0 = capacity ceiling: no denominator */
    {
        const int64_t want = max_new;
        cwick_status wst = cwick_gen_window(me->pos, prompt_len, &max_new,
                                            me->capacity, err);
        if (wst != CWICK_OK) return wst;
        if (max_new != want) {
            total_new = 0;
            CWICK_INFO(tag, "max_tokens %lld clamped to %lld: prompt %lld at "
                       "position %lld leaves %lld of %lld positions",
                       (long long)want, (long long)max_new,
                       (long long)prompt_len, (long long)me->pos,
                       (long long)max_new, (long long)me->capacity);
        }
    }

    /* Sized for the block, not for the prompt: a normal step only ever writes
     * ids[0], but a verified block writes MM_SPEC_MAX slots regardless of how
     * long the prompt is -- and with a prefix-cache replay the prompt handed to
     * this call is a single token, which used to make this a 4-byte buffer. */
    int64_t ids_n = prompt_len > MM_SPEC_MAX ? prompt_len : MM_SPEC_MAX + 1;
    int32_t *ids = cwick_xmalloc((size_t)ids_n * sizeof(int32_t));
    for (int64_t i = 0; i < prompt_len; i++) ids[i] = (int32_t)prompt[i];

    /* ---- prefill ---- */
    double prefill_ms = 0.0;
    int64_t done = 0;
    mm_prefill_progress_begin(prompt_len, c->n_layers);
    while (done < prompt_len) {
        if (cx && mm_stopped(cx)) {
            mm_prefill_progress_end();
            free(ids);
            cwick_error_set(err, CWICK_ERR_STATE, "interrupted during prefill");
            return CWICK_ERR_STATE;
        }
        int n = (int)(prompt_len - done);
        if (n > MM_CHUNK) n = MM_CHUNK;
        double t0 = now_ms();
        if (mm_model_forward(&me->m, ids + done, n, me->pos, me->logits,
                             err) != 0) {
            mm_prefill_progress_end();
            free(ids);
            return CWICK_ERR_STATE;
        }
        prefill_ms += now_ms() - t0;
        me->pos += n;
        done += n;
        mm_prefill_progress_done(done);
        if (me->on_prefill_checkpoint && me->checkpoint_block > 0 &&
            me->pos >= me->next_checkpoint) {
            me->on_prefill_checkpoint(me->on_prefill_checkpoint_user, me->pos);
            me->next_checkpoint = me->pos + me->checkpoint_block;
        }
    }
    if (prompt_len == 0) {
        double t0 = now_ms();
        if (mm_model_forward(&me->m, ids, 1, me->pos, me->logits, err) != 0) {
            mm_prefill_progress_end();
            free(ids);
            return CWICK_ERR_STATE;
        }
        prefill_ms += now_ms() - t0;
        me->pos += 1;
    }
    mm_prefill_progress_end();

    if (me->on_prefill_done) me->on_prefill_done(me->on_prefill_done_user, me->pos);

    int64_t collected = keep_prompt ? prompt_len : 0;
    uint32_t *tokens = cwick_xmalloc((size_t)(collected + max_new + 1) *
                                     sizeof(uint32_t));
    if (keep_prompt && prompt_len > 0)
        memcpy(tokens, prompt, (size_t)prompt_len * sizeof(uint32_t));

    int64_t gen = 0, decode_steps = 0;
    bool hit_eos = false, interrupted = false, sink_stopped = false;
    bool loop_stopped = false;
    double decode_t0 = now_ms();
    double last_rep = decode_t0;
    int64_t steps_accounted = 0, gen_accounted = 0;

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
    bool think_closed = cx ? cx->think_closed_at_start : false;
    bool think_budget_hit = false;

    cwick_sampler smp;
    cwick_sampler_init(&smp, &me->smp, c->vocab_size, mm_sampler_seed());
    if (smp.counts)
        for (int64_t i = 0; i < prompt_len; i++) cwick_sampler_prime(&smp, prompt[i]);

    /* Speculative block verification.
     *
     * MM_SPEC=N is the block size: N-1 candidates are verified per forward, so
     * one forward can commit up to N tokens (the one already sampled plus the
     * candidates that survive). Off by default, because whether it wins is a
     * property of the model and the cache, not of the code: on a memory-bound
     * MoE the block's expert *union* decides the cost, and that is measured,
     * not assumed (see the summary line's `spec=` counters).
     *
     * Greedy only. With a drafter that proposes one fixed token, exact
     * speculative sampling has to accept with probability p(token) and, on
     * rejection, draw from the residual distribution -- the sampler does not
     * expose the filtered distribution, so doing this for temperature > 0 would
     * change the output distribution silently. Greedy acceptance is an equality
     * test, which is exact by construction. */
    static int spec_want = -1;
    if (spec_want < 0) {
        const char *e = getenv("MM_SPEC");
        spec_want = (e && *e) ? atoi(e) : 0;
        if (spec_want < 2 || spec_want > MM_SPEC_MAX) spec_want = 0;
    }
    bool spec_on = spec_want > 0 && cwick_sampler_cfg_is_greedy(&smp.cfg);
    if (spec_on && me->spec_n != spec_want) {
        int64_t V = c->vocab_size;
        free(me->spec_logits);
        free(me->spec_cand);
        free(me->spec_draft);
        free(me->spec_ids);
        me->spec_logits = cwick_xmalloc(sizeof(float) * (size_t)spec_want *
                                        (size_t)V);
        me->spec_cand = cwick_xmalloc(sizeof(float) * (size_t)V);
        me->spec_draft = cwick_xmalloc(sizeof(uint32_t) * (size_t)spec_want);
        me->spec_ids = cwick_xmalloc(sizeof(int32_t) * (size_t)spec_want);
        me->spec_n = spec_want;
    }
    int64_t spec_tokens = 0;   /* tokens committed by verification */

    for (int64_t k = 0; k < max_new; k++) {
        if (cx && mm_stopped(cx)) { interrupted = true; break; }
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
                    cwick_sampler_accept(&smp, ct);
                    if ((int64_t)ct == think_open_id) think_open = true;
                    if ((int64_t)ct == think_close_id) think_closed = true;
                    if (sink && !sink(user, ct, gen - 1)) {
                        sink_stopped = true;
                        aborted = true;
                        break;
                    }
                    if (k + 1 < max_new) {
                        ids[0] = (int32_t)ct;
                        if (mm_model_forward(&me->m, ids, 1, me->pos, me->logits,
                                             err) != 0) {
                            cwick_sampler_free(&smp);
                            free(tokens);
                            free(ids);
                            return CWICK_ERR_STATE;
                        }
                        me->pos++;
                        decode_steps++;
                    }
                }
                if (written) {
                    think_budget_hit = true;
                    CWICK_INFO(tag, "thinking budget: %lld tokens reached — "
                              "wrote </think> so the turn answers instead of "
                              "thinking", (long long)think_budget);
                }
                if (aborted) break;
                if (gen >= max_new) break;
                continue;
            }
        }
        uint32_t next = cwick_sampler_pick(&smp, me->logits, c->vocab_size);
        if (spec && mm_is_eos(spec, next)) { hit_eos = true; break; }
        tokens[collected + gen++] = next;
        cwick_sampler_accept(&smp, next);
        if ((int64_t)next == think_open_id) think_open = true;
        if ((int64_t)next == think_close_id) think_closed = true;
        int lp = 0;
        int64_t lr = 0;
        if (cwick_sampler_looped(&smp, &lp, &lr)) {
            gen--;
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
            int nd = spec_on ? mm_draft_lookup(tokens, collected + gen,
                                               spec_want - 1, me->spec_draft) : 0;
            int acc = 1;          /* positions this forward commits: `next` */
            if (nd > 0) {
                int nb = nd + 1;
                for (int i = 0; i < nd; i++) ids[i + 1] = (int32_t)me->spec_draft[i];
                if (mm_model_forward_multi(&me->m, ids, nb, me->pos,
                                           me->spec_logits, err) != 0) {
                    cwick_sampler_free(&smp);
                    free(tokens);
                    free(ids);
                    return CWICK_ERR_STATE;
                }
                /* Candidate i is judged by the distribution of the position
                 * before it, which is logits_all[i]. The pick runs on a copy
                 * because it is allowed to modify the logits it is handed
                 * (penalties, the loop breaker), and the copy that survives is
                 * the one the next iteration samples from. */
                const int64_t V = c->vocab_size;
                int kept = 0;
                for (int i = 0; i < nd; i++) {
                    const float *src = me->spec_logits + (size_t)i * (size_t)V;
                    memcpy(me->spec_cand, src, sizeof(float) * (size_t)V);
                    uint32_t got = cwick_sampler_pick(&smp, me->spec_cand, V);
                    memcpy(me->logits, src, sizeof(float) * (size_t)V);
                    if (got != me->spec_draft[i]) break;
                    acc++;
                    kept++;
                }
                if (kept == nd)
                    memcpy(me->logits, me->spec_logits + (size_t)nd * (size_t)V,
                           sizeof(float) * (size_t)V);
                me->spec_proposed += nd;
                me->spec_accepted += acc - 1;
                /* The accepted candidates are real tokens now: they go through
                 * the same accept/stop path as a sampled one, or the loop guard,
                 * the stop set and the thinking markers would disagree with the
                 * context. */
                int emitted = 0;
                for (int i = 0; i < kept; i++) {
                    uint32_t t = me->spec_draft[i];
                    if (spec && mm_is_eos(spec, t)) { hit_eos = true; break; }
                    if (gen >= max_new) break;
                    tokens[collected + gen++] = t;
                    cwick_sampler_accept(&smp, t);
                    if ((int64_t)t == think_open_id) think_open = true;
                    if ((int64_t)t == think_close_id) think_closed = true;
                    emitted++;
                    if (sink && !sink(user, t, gen - 1)) {
                        sink_stopped = true;
                        break;
                    }
                }
                /* Only what was actually emitted is committed state; a block
                 * that ran into the token ceiling leaves the rest of its KV
                 * behind `pos`, where the next block overwrites it. */
                me->pos += 1 + emitted;
                spec_tokens += emitted;
            } else {
                if (mm_model_forward(&me->m, ids, 1, me->pos, me->logits,
                                     err) != 0) {
                    cwick_sampler_free(&smp);
                    free(tokens);
                    free(ids);
                    return CWICK_ERR_STATE;
                }
                me->pos++;
            }
            decode_steps++;
            did_forward = true;
            if (hit_eos || sink_stopped || gen >= max_new) break;
        }
        double step_ms = now_ms() - step_t0;
        char phase_lab[32];
        mm_phase_label(cx, phase_lab, sizeof(phase_lab));
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
        if (did_forward && (now_ms() - last_rep) >= mm_progress_min_ms()) {
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
                /* A budget that came from the window is a ceiling, not a
                 * length: "1/24453 (0.0%)" reads as stalled, and the real
                 * answer stops at EOS long before the ceiling. */
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
        char pf_tps[32], dc_tps[32], spec_txt[56];
        if (prefill_ms > 0.0)
            snprintf(pf_tps, sizeof(pf_tps), "%.3f",
                     (double)prompt_len * 1000.0 / prefill_ms);
        else
            snprintf(pf_tps, sizeof(pf_tps), "n/a");
        /* The rate is per *token*, not per forward: with verification on, one
         * forward can commit several tokens, and reporting forwards/second
         * would understate the engine by exactly the thing being measured. */
        int64_t rate_tokens = decode_steps + spec_tokens;
        if (rate_tokens > 0 && decode_ms > 0.0)
            snprintf(dc_tps, sizeof(dc_tps), "%.3f",
                     (double)rate_tokens * 1000.0 / decode_ms);
        else
            snprintf(dc_tps, sizeof(dc_tps), "n/a");
        spec_txt[0] = '\0';
        if (spec_tokens > 0)
            snprintf(spec_txt, sizeof(spec_txt), " spec=%lld/%lld",
                     (long long)me->spec_accepted,
                     (long long)me->spec_proposed);
        uint64_t bs = 0, rc = 0, ch = 0, cm = 0;
        if (me->m.stream) cwick_stream_stats(me->m.stream, &bs, &rc, &ch, &cm);
        CWICK_INFO(tag, "prompt=%lld gen=%lld finish=%s prefill=%.0fms/%s "
                   "tok/s decode=%.0fms/%lld steps %s tok/s "
                   "cache=%.1f%%(%llu/%llu) streamed=%.2fGiB mem=%.2fGiB%s",
                   (long long)prompt_len, (long long)gen,
                   interrupted ? "interrupted"
                               : loop_stopped ? "loop"
                               : ((hit_eos || sink_stopped) ? "stop" : "length"),
                   prefill_ms, pf_tps, decode_ms, (long long)decode_steps,
                   dc_tps,
                   (ch + cm) ? 100.0 * (double)ch / (double)(ch + cm) : 0.0,
                   (unsigned long long)ch, (unsigned long long)cm,
                   cwick_gib(bs), cwick_gib(cwick_process_footprint_bytes()),
                   spec_txt);
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
