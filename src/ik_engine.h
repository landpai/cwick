/*
 * ik_engine.h — the Inkling-2-Mini engine backend.
 *
 * Same contract as the other backends: `ctx` carries the interrupt sources, log
 * tag, stop set and thinking budget the owner copies in before every turn, and
 * the prefix cache drives the position through the same two hooks. There is no
 * speculative path here — the checkpoint ships an MTP head this build does not
 * use, and on a resident model the drafter's extra weight traffic costs more
 * than the tokens it wins.
 *
 * Images arrive as embeddings, not as tokens: the tokenizer emits one
 * placeholder per merged patch, the vision tower turns the picture into that
 * many vectors, and the engine splices them in at the placeholders.
 */
#ifndef CWICK_IK_ENGINE_H
#define CWICK_IK_ENGINE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "cwick.h"
#include "ik.h"

#define IK_MAX_IMAGE_TOKENS 16384

typedef struct ik_engine {
    ik_model m;
    float *logits;        /* [vocab] */
    float *img_owned;     /* engine-owned copy of the assembled image vectors */
    bool img_locked;
    int64_t pos;
    int64_t capacity;
    bool opened;
    cwick_sampler_cfg smp;

    void (*on_prefill_done)(void *user, int64_t pos);
    void *on_prefill_done_user;
    void (*on_prefill_checkpoint)(void *user, int64_t pos);
    void *on_prefill_checkpoint_user;
    int64_t checkpoint_block;
    int64_t next_checkpoint;

    struct ik_ctx {
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
} ik_engine;

int ik_engine_open(ik_engine *ie, const char *root, int64_t capacity,
                   cwick_error *err);
void ik_engine_close(ik_engine *ie);
void ik_engine_reset(ik_engine *ie);

/* Hand the engine the vision tower's output for the next turn. `embeds` and
 * per-image token counts are borrowed for the duration of the call. */
cwick_status ik_engine_set_images(ik_engine *ie, const float *embeds,
                                  int64_t n_tokens, const int64_t *image_tokens,
                                  size_t n_images, cwick_error *err);

/* Decode and run the vision tower on one prompt image. */
int ik_engine_vision_embed(ik_engine *ie, const uint8_t *data, size_t len,
                           float **embeds, int64_t *n_tokens, int64_t *rows,
                           int64_t *cols, cwick_error *err);

cwick_status ik_engine_generate(ik_engine *ie, const uint32_t *prompt,
                                int64_t prompt_len, int64_t max_new,
                                bool keep_prompt, cwick_token_sink sink,
                                void *user, cwick_gen_result *result,
                                cwick_error *err);

#endif /* CWICK_IK_ENGINE_H */
