/*
 * ds_engine.h — the DeepSeek-V4.1 engine backend.
 *
 * The GLM engine in engine.c owns a cwick_spine, MLA/KDA caches and an
 * indexer; this one owns a ds_model and nothing else. Same public entry
 * points, same greedy contract, same result struct — so the server, the CLI
 * and the metrics registry cannot tell the difference.
 */
#ifndef CWICK_DS_ENGINE_H
#define CWICK_DS_ENGINE_H

#include <stdint.h>
#include <stddef.h>

#include "cwick.h"
#include "ds_model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds_engine {
    ds_model m;
    float *logits;          /* [vocab] the last forward's output */
    int64_t pos;            /* next absolute position to write */
    int64_t capacity;       /* positions the caches were sized for */
    int opened;
    /* How the decode loop turns logits into a token. Copied in from the owning
     * cwick_engine (which reads it from the checkpoint's generation_config.json
     * and from the request), so the two backends sample identically. */
    cwick_sampler_cfg smp;
    /* Fired once, right after the prompt is prefilled and before the first
     * decode step, with the position the snapshot would cover. The owning
     * cwick_engine installs it so the prefix cache can snapshot this backend
     * at the one moment its state equals "prompt prefilled" -- the same point
     * the GLM path captures at the end of its prefill. */
    void (*on_prefill_done)(void *user, int64_t pos);
    void *on_prefill_done_user;
    /* Mid-prefill checkpoints, every `checkpoint_block` positions (0 = off).
     * These do not consume the final snapshot: each one is a separate state,
     * which is what lets a later prompt that shares only the head of a long
     * one still replay that head. */
    void (*on_prefill_checkpoint)(void *user, int64_t pos);
    void *on_prefill_checkpoint_user;
    int64_t checkpoint_block;
    int64_t next_checkpoint;
    /* Interrupt sources, log tag and the spec's stop set, copied in by the
     * owning engine before every generation. */
    struct ds_ctx {
        volatile int *stop;
        bool (*abort_poll)(void *user);
        void *abort_poll_user;
        bool verbose;
        const char *tag;
        const cwick_model_spec *spec;
        /* Which part of the answer is being decoded right now: "thinking",
         * "content", "tools", or NULL before the server could classify the
         * first token. Pushed in by cwick_engine_set_decode_phase from the
         * same state machine that splits the stream, so the label on a
         * progress line is the label on the wire. */
        const char *decode_phase;
        /* Thinking budget, copied in by the owning engine exactly like the
         * fields above. The GLM path enforces its ceiling inside its own
         * decode loop; this backend runs a separate loop, so without a copy
         * here the server's "thinking budget N tokens" line would be true of
         * the configuration and false of the behaviour -- which is how a
         * 高 (8192) request once generated 16000 thinking tokens and kept
         * going. `budget` < 0 disables the mechanism, same as the engine's. */
        int64_t think_budget;
        int64_t think_open_id, think_close_id;
        bool think_open_at_start;   /* the template already appended the tag */
    } ctx;
} ds_engine;

/* Open the checkpoint at `root`, sizing the caches for `capacity` positions.
 * Returns 0 on success; on failure fills err and returns nonzero. */
int ds_engine_open(ds_engine *de, const char *root, int64_t capacity,
                   cwick_error *err);
void ds_engine_close(ds_engine *de);

/* Greedy generation over `prompt`. Mirrors cwick_engine_generate's
 * contract: the prefill's leftover logits already predict the token after
 * the last prompt token, so a max_new=N run performs N-1 decode forwards.
 * `sink` (optional) sees every token as it is produced. */
cwick_status ds_engine_generate(ds_engine *de, const uint32_t *prompt,
                                int64_t prompt_len, int64_t max_new,
                                bool keep_prompt, cwick_token_sink sink,
                                void *user, cwick_gen_result *result,
                                cwick_error *err);

#ifdef __cplusplus
}
#endif

#endif /* CWICK_DS_ENGINE_H */
