/*
 * ik_engine.c — generation for the Inkling-2-Mini backend.
 *
 * Deliberately plain: prefill in chunks, then one token per forward. The
 * checkpoint is fully resident, so there is no expert cache to prime and no
 * drafter worth running; the only things that carry over from the other
 * backends are the ones a caller can observe — the stop set, the thinking
 * budget, the loop guard, the prefix-cache hooks and the metrics counters.
 */
#include "ik_engine.h"

#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "log.h"
#include "mem.h"
#include "metrics.h"
#include "mm.h"
#include "util.h"
#include "sample.h"

static double ik_now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1e3 + (double)tv.tv_usec / 1e3;
}

static bool ik_stopped(const struct ik_ctx *cx) {
    if (cx && cx->stop && *cx->stop) return true;
    if (cx && cx->abort_poll && cx->abort_poll(cx->abort_poll_user)) return true;
    return false;
}

static bool ik_is_eos(const cwick_model_spec *spec, uint32_t tok) {
    if (!spec) return false;
    for (int i = 0; i < spec->n_eos; i++)
        if ((int64_t)tok == spec->eos_ids[i]) return true;
    return (int64_t)tok == spec->eos_token_id;
}

static const char *ik_phase_label(const struct ik_ctx *cx, char *buf,
                                  size_t cap) {
    if (cx && cx->decode_phase)
        snprintf(buf, cap, "decode(%s)", cx->decode_phase);
    else
        snprintf(buf, cap, "decode");
    return buf;
}

int ik_engine_open(ik_engine *ie, const char *root, int64_t capacity,
                   cwick_error *err) {
    memset(ie, 0, sizeof(*ie));
    if (ik_model_open(&ie->m, root, capacity, err) != 0) return -1;
    ie->logits = (float *)cwick_xmalloc((size_t)ie->m.c.vocab * sizeof(float));
    if (!ie->logits) {
        ik_model_close(&ie->m);
        cwick_error_set(err, CWICK_ERR_MEMORY, "logit buffer allocation failed");
        return -1;
    }
    ie->capacity = ie->m.cap;
    /* This checkpoint has no <think> marker: -1 means "no enforced budget".
     * Zero here would make the engine force a close marker on step one. */
    ie->ctx.think_budget = -1;
    ie->opened = true;
    return 0;
}

static void ik_images_clear(ik_engine *ie) {
    if (ie->img_owned) {
        if (ie->img_locked)
            cwick_mem_unlock_range(ie->img_owned,
                                   (size_t)ie->m.img_tokens * ie->m.c.hidden * sizeof(float));
        free(ie->img_owned);
    }
    ie->img_owned = NULL;
    ie->img_locked = false;
    ie->m.img_embeds = NULL;
    ie->m.img_tokens = 0;
    ie->m.img_used = 0;
}

void ik_engine_close(ik_engine *ie) {
    if (ie) ik_images_clear(ie);
    if (!ie->opened) return;
    ik_model_close(&ie->m);
    free(ie->logits);
    ie->logits = NULL;
    ie->opened = false;
}

void ik_engine_reset(ik_engine *ie) {
    if (!ie->opened) return;
    ik_images_clear(ie);
    ik_model_reset(&ie->m);
    ie->pos = 0;
}

cwick_status ik_engine_set_images(ik_engine *ie, const float *embeds,
                                  int64_t n_tokens, const int64_t *image_tokens,
                                  size_t n_images, cwick_error *err) {
    if (!ie) return CWICK_ERR_ARGUMENT;
    ik_images_clear(ie);
    if (n_images > 16) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "at most 16 images are supported");
        return CWICK_ERR_ARGUMENT;
    }
    int64_t sum = 0;
    for (size_t i = 0; i < n_images; i++) {
        if (!image_tokens || image_tokens[i] <= 0 || image_tokens[i] > 4096 ||
            sum > IK_MAX_IMAGE_TOKENS - image_tokens[i]) {
            cwick_error_set(err, CWICK_ERR_ARGUMENT,
                            "image token spans exceed %d tokens", IK_MAX_IMAGE_TOKENS);
            return CWICK_ERR_ARGUMENT;
        }
        sum += image_tokens[i];
    }
    if (sum != n_tokens || !embeds || n_tokens <= 0) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "image token spans do not match embeddings");
        return CWICK_ERR_ARGUMENT;
    }
    size_t n = (size_t)n_tokens * (size_t)ie->m.c.hidden;
    ie->img_owned = (float *)malloc(n * sizeof(float));
    if (!ie->img_owned) {
        cwick_error_set(err, CWICK_ERR_MEMORY, "image embedding copy failed");
        return CWICK_ERR_MEMORY;
    }
    memcpy(ie->img_owned, embeds, n * sizeof(float));
    ie->m.img_embeds = ie->img_owned;
    ie->m.img_tokens = n_tokens;
    if (cwick_memory_noswap())
        ie->img_locked = cwick_mem_lock_range(ie->img_owned, n * sizeof(float),
                                              "Inkling image embeddings");
    return CWICK_OK;
}

int ik_engine_vision_embed(ik_engine *ie, const uint8_t *data, size_t len,
                           float **embeds, int64_t *n_tokens, int64_t *rows,
                           int64_t *cols, cwick_error *err) {
    if (!ie || !ie->opened) {
        cwick_error_set(err, CWICK_ERR_STATE, "Inkling-2-Mini engine is not open");
        return -1;
    }
    return ik_vision_embed(&ie->m, data, len, embeds, n_tokens, rows, cols, err);
}

cwick_status ik_engine_generate(ik_engine *ie, const uint32_t *prompt,
                                int64_t prompt_len, int64_t max_new,
                                bool keep_prompt, cwick_token_sink sink,
                                void *user, cwick_gen_result *result,
                                cwick_error *err) {
    cwick_error_clear(err);
    if (!ie || !ie->opened) {
        cwick_error_set(err, CWICK_ERR_STATE, "Inkling-2-Mini engine is not open");
        return CWICK_ERR_STATE;
    }
    const struct ik_ctx *cx = &ie->ctx;
    const cwick_model_spec *spec = cx ? cx->spec : NULL;
    const char *tag = (cx && cx->tag) ? cx->tag : "gen";
    const int64_t V = ie->m.c.unpadded_vocab > 0 ? ie->m.c.unpadded_vocab : ie->m.c.vocab;

    ie->ctx.decode_phase = NULL;
    ie->next_checkpoint = ie->checkpoint_block > 0 ? ie->checkpoint_block : 0;
    ie->m.img_used = 0;

    /* max_new is a ceiling, not a demand */
    {
        const int64_t want = max_new;
        cwick_status wst = cwick_gen_window(ie->pos, prompt_len, &max_new,
                                            ie->capacity, err);
        if (wst != CWICK_OK) return wst;
        if (max_new != want)
            CWICK_INFO(tag, "max_tokens %lld clamped to %lld: prompt %lld at "
                       "position %lld leaves %lld of %lld positions",
                       (long long)want, (long long)max_new,
                       (long long)prompt_len, (long long)ie->pos,
                       (long long)max_new, (long long)ie->capacity);
    }

    int32_t *ids = (int32_t *)cwick_xmalloc(
        (size_t)(prompt_len > 1 ? prompt_len : 1) * sizeof(int32_t));
    for (int64_t i = 0; i < prompt_len; i++) ids[i] = (int32_t)prompt[i];

    /* ---- prefill ---- */
    double prefill_ms = 0.0;
    int64_t done = 0;
    mm_prefill_progress_begin(prompt_len, ie->m.c.n_layers);
    while (done < prompt_len) {
        if (ik_stopped(cx)) {
            mm_prefill_progress_end();
            free(ids);
            cwick_error_set(err, CWICK_ERR_STATE, "interrupted during prefill");
            return CWICK_ERR_STATE;
        }
        int n = (int)(prompt_len - done);
        if (n > IK_CHUNK) n = IK_CHUNK;
        double t0 = ik_now_ms();
        if (ik_model_forward(&ie->m, ids + done, n, ie->pos, ie->logits,
                             err) != 0) {
            mm_prefill_progress_end();
            free(ids);
            return CWICK_ERR_STATE;
        }
        prefill_ms += ik_now_ms() - t0;
        ie->pos += n;
        done += n;
        mm_prefill_progress_done(done);
        if (ie->on_prefill_checkpoint && ie->checkpoint_block > 0 &&
            ie->pos >= ie->next_checkpoint) {
            ie->on_prefill_checkpoint(ie->on_prefill_checkpoint_user, ie->pos);
            ie->next_checkpoint = ie->pos + ie->checkpoint_block;
        }
    }
    if (prompt_len == 0) {
        ids[0] = (int32_t)(spec && spec->pad_token_id >= 0
                               ? spec->pad_token_id : 199999);
        double t0 = ik_now_ms();
        if (ik_model_forward(&ie->m, ids, 1, ie->pos, ie->logits, err) != 0) {
            mm_prefill_progress_end();
            free(ids);
            return CWICK_ERR_STATE;
        }
        prefill_ms += ik_now_ms() - t0;
        ie->pos += 1;
    }
    mm_prefill_progress_end();
    if (ie->m.img_tokens > 0 && ie->m.img_used != ie->m.img_tokens) {
        free(ids);
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "image placeholders consumed %lld of %lld vectors",
                        (long long)ie->m.img_used, (long long)ie->m.img_tokens);
        return CWICK_ERR_ARGUMENT;
    }
    if (ie->on_prefill_done) ie->on_prefill_done(ie->on_prefill_done_user, ie->pos);

    int64_t collected = keep_prompt ? prompt_len : 0;
    uint32_t *tokens = (uint32_t *)cwick_xmalloc(
        (size_t)(collected + max_new + 1) * sizeof(uint32_t));
    if (keep_prompt && prompt_len > 0)
        memcpy(tokens, prompt, (size_t)prompt_len * sizeof(uint32_t));

    int64_t gen = 0, decode_steps = 0;
    bool hit_eos = false, interrupted = false, sink_stopped = false;
    bool loop_stopped = false, think_budget_hit = false;

    const int64_t think_open_id = cx ? cx->think_open_id : -1;
    const int64_t think_close_id = cx ? cx->think_close_id : -1;
    int64_t think_budget = cx ? cx->think_budget : -1;
    if (think_budget >= max_new) {
        int64_t room = max_new - CWICK_THINK_RESERVE;
        if (room < 0) room = max_new / 2;
        if (cx && cx->verbose)
            CWICK_VERB(tag, "thinking budget %lld clamped to %lld", 
                       (long long)think_budget, (long long)room);
        think_budget = room;
    }
    bool think_open = cx ? cx->think_open_at_start : false;
    bool think_closed = cx ? cx->think_closed_at_start : false;

    cwick_sampler smp;
    uint64_t seed = 0;
    const char *seed_env = getenv("CWICK_SEED");
    if (seed_env && *seed_env) seed = strtoull(seed_env, NULL, 0);
    cwick_sampler_init(&smp, &ie->smp, V, seed);
    if (smp.counts)
        for (int64_t i = 0; i < prompt_len; i++) cwick_sampler_prime(&smp, prompt[i]);

    double decode_t0 = ik_now_ms();
    int64_t steps_accounted = 0, gen_accounted = 0;

    for (int64_t k = 0; k < max_new; k++) {
        if (ik_stopped(cx)) { interrupted = true; break; }

        /* ---- thinking budget: close the block so the turn answers ---- */
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
                        if (ik_model_forward(&ie->m, ids, 1, ie->pos,
                                             ie->logits, err) != 0) {
                            cwick_sampler_free(&smp);
                            free(tokens); free(ids);
                            return CWICK_ERR_STATE;
                        }
                        ie->pos++;
                        decode_steps++;
                    }
                }
                if (written) {
                    think_budget_hit = true;
                    CWICK_INFO(tag, "thinking budget: %lld tokens reached — "
                               "closed the block so the turn answers",
                               (long long)think_budget);
                }
                if (aborted) break;
                if (gen >= max_new) break;
                continue;
            }
        }

        uint32_t next = cwick_sampler_pick(&smp, ie->logits, V);
        if (spec && ik_is_eos(spec, next)) { hit_eos = true; break; }
        tokens[collected + gen++] = next;
        cwick_sampler_accept(&smp, next);
        if ((int64_t)next == think_open_id) think_open = true;
        if ((int64_t)next == think_close_id) think_closed = true;
        int lp = 0;
        int64_t lr = 0;
        if (cwick_sampler_looped(&smp, &lp, &lr)) {
            gen--;
            CWICK_INFO(tag, "loop guard: period-%d block repeated %lld times — "
                       "stopping after %lld generated", lp,
                       (long long)(lr / (lp > 0 ? lp : 1) + 1), (long long)gen);
            loop_stopped = true;
            break;
        }
        if (sink && !sink(user, next, gen - 1)) { sink_stopped = true; break; }
        if (k + 1 >= max_new) break;

        if (ie->pos + 1 > ie->capacity) break;
        ids[0] = (int32_t)next;
        if (ik_model_forward(&ie->m, ids, 1, ie->pos, ie->logits, err) != 0) {
            cwick_sampler_free(&smp);
            free(tokens); free(ids);
            return CWICK_ERR_STATE;
        }
        ie->pos++;
        decode_steps++;
    }

    double decode_ms = ik_now_ms() - decode_t0;
    {
        char ph[48];
        double tps = decode_ms > 0.0 ? (double)decode_steps * 1000.0 / decode_ms : 0.0;
        CWICK_INFO(tag, "%s: %lld tokens in %.0f ms (%.2f tok/s), prefill %.0f ms"
                   " (%.2f tok/s) — %lld generated", ik_phase_label(cx, ph, sizeof(ph)),
                   (long long)decode_steps, decode_ms, tps, prefill_ms,
                   prefill_ms > 0.0 ? (double)prompt_len * 1000.0 / prefill_ms : 0.0,
                   (long long)gen);
    }
    cwick_prof_report(tag, decode_ms);

    result->stop_reason_length =
        !hit_eos && !interrupted && !sink_stopped && !loop_stopped && gen == max_new;
    result->prefill_ms = prefill_ms;
    result->decode_ms = decode_ms;
    result->decode_steps = decode_steps;
    result->interrupted = interrupted;
    result->sink_stopped = sink_stopped;
    result->loop_stopped = loop_stopped;
    result->think_budget_hit = think_budget_hit;
    result->think_budget_applied = think_budget;
    if (decode_steps > steps_accounted)
        cwick_metrics_add_decode_steps(decode_steps - steps_accounted);
    if (gen > gen_accounted) cwick_metrics_add_generation_tokens(gen - gen_accounted);
    ie->ctx.decode_phase = NULL;
    result->tokens = tokens;
    result->n_tokens = collected + gen;
    cwick_sampler_free(&smp);
    free(ids);
    if (interrupted) {
        cwick_error_set(err, CWICK_ERR_STATE, "interrupted");
        return CWICK_ERR_STATE;
    }
    return CWICK_OK;
}
