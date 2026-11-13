/*
 * q35_engine.h — the Qwen3.5 engine backend.
 *
 * Field-for-field the same contract as ds_engine.h and mm_engine.h: the owning
 * cwick_engine copies its sampler config, interrupt sources, log tag, stop set
 * and thinking budget into `ctx` before every turn, and the prefix cache drives
 * it through the same two hooks. The server, the CLI and the metrics registry
 * cannot tell the backends apart, which is the point -- a fourth model must not
 * add a second generation policy.
 */
#ifndef CWICK_Q35_ENGINE_H
#define CWICK_Q35_ENGINE_H

#include <stdint.h>
#include <stddef.h>

#include "cwick.h"
#include "q35.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct q35_engine {
    q35_model m;
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

    struct q35_ctx {
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
        bool think_closed_at_start;
    } ctx;
} q35_engine;

int q35_engine_open(q35_engine *qe, const char *root, int64_t capacity,
                    cwick_error *err);
void q35_engine_close(q35_engine *qe);

cwick_status q35_engine_generate(q35_engine *qe, const uint32_t *prompt,
                                 int64_t prompt_len, int64_t max_new,
                                 bool keep_prompt, cwick_token_sink sink,
                                 void *user, cwick_gen_result *result,
                                 cwick_error *err);

/* Reset the position to 0 and clear every cache. Used by the prefix-cache
 * replay path, where the prompt handed over is the suffix after a snapshot. */
void q35_engine_reset(q35_engine *qe);

#ifdef __cplusplus
}
#endif

#endif /* CWICK_Q35_ENGINE_H */
