/*
 * mm_engine.h — the MiniMax-M3.1-Flash engine backend.
 *
 * Deliberately field-for-field the same contract as ds_engine.h: the owning
 * cwick_engine copies its sampler config, its interrupt sources, its log tag,
 * the spec's stop set and the thinking budget into `ctx` before every turn,
 * and the prefix cache drives it through the same two hooks. The server, the
 * CLI and the metrics registry cannot tell the three backends apart, which is
 * the point -- adding a model must not add a second generation policy.
 */
#ifndef CWICK_MM_ENGINE_H
#define CWICK_MM_ENGINE_H

#include <stdint.h>
#include <stddef.h>

#include "cwick.h"
#include "mm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mm_engine {
    mm_model m;
    float *logits;          /* [vocab] the last forward's output */
    int64_t pos;            /* next absolute position to write */
    int64_t capacity;       /* positions the caches were sized for */
    int opened;
    cwick_sampler_cfg smp;

    void (*on_prefill_done)(void *user, int64_t pos);
    void *on_prefill_done_user;
    void (*on_prefill_checkpoint)(void *user, int64_t pos);
    void *on_prefill_checkpoint_user;
    int64_t checkpoint_block;
    int64_t next_checkpoint;

    /* Same fields, same meanings, same reasons as struct ds_ctx. */
    struct mm_ctx {
        volatile int *stop;
        bool (*abort_poll)(void *user);
        void *abort_poll_user;
        bool verbose;
        const char *tag;
        const cwick_model_spec *spec;
        const char *decode_phase;
        int64_t think_budget;
        int64_t think_open_id, think_close_id;
        bool think_open_at_start;
        /* The prompt itself already ended with the close marker. This
         * checkpoint's template writes `</mm:think>` after the assistant
         * header when the turn is not meant to think, so with a ceiling of 0
         * the engine used to inject a *second*, empty `<mm:think></mm:think>`
         * on top of it: two wasted decode steps and a malformed sequence in
         * the model's context. */
        bool think_closed_at_start;
    } ctx;

    /* Speculative decoding (see mm_engine.c): the tokens the draft model
     * accepted last round, and the counters the summary line reports. */
    int64_t spec_proposed, spec_accepted;
    /* Block buffers, allocated on first use because the vocabulary is only
     * known once the checkpoint is open. `spec_cand` exists because the pick
     * may modify the logits it is given (penalties, loop breaker) and the one
     * the next iteration samples from has to stay pristine. */
    float *spec_logits;     /* [spec_n][vocab] */
    float *spec_cand;       /* [vocab] */
    uint32_t *spec_draft;   /* [spec_n] */
    int32_t *spec_ids;      /* [spec_n] */
    int spec_n;             /* 0 = speculation off */
} mm_engine;

int mm_engine_open(mm_engine *me, const char *root, int64_t capacity,
                   cwick_error *err);
void mm_engine_close(mm_engine *me);

cwick_status mm_engine_generate(mm_engine *me, const uint32_t *prompt,
                                int64_t prompt_len, int64_t max_new,
                                bool keep_prompt, cwick_token_sink sink,
                                void *user, cwick_gen_result *result,
                                cwick_error *err);

#ifdef __cplusplus
}
#endif

#endif /* CWICK_MM_ENGINE_H */
