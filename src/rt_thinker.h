/* rt_thinker.h — the listening head of the realtime model: a 64-layer
 * Qwen3-VL text decoder with the audio tower's projected frames injected at
 * the <|audio|> positions.
 *
 * Weights are read straight from the checkpoint's shards (mmap, FP8 E4M3 with
 * one BF16 scale per output row) and never copied: a 32 GiB decoder has no
 * room for a second representation. The GEMMs run on the GPU through the
 * per-row-scale kernels in metal_objc.m, with a CPU path that produces the
 * same numbers when Metal is unavailable.
 *
 * The audio tower lives in rt_ear.c; this file only consumes its output. */
#ifndef CWICK_RT_THINKER_H
#define CWICK_RT_THINKER_H

#include <stdbool.h>
#include <stdint.h>

#include "cwick.h"

typedef enum {
    RT_BACKEND_AUTO = 0,
    RT_BACKEND_METAL,
    RT_BACKEND_CPU,
} rt_backend;

typedef struct rt_thinker rt_thinker;

/* Opens `<thinker_dir>` (the subdirectory holding config.json and the
 * shards). `max_seq` bounds the KV cache; 0 picks a default. */
rt_thinker *rt_thinker_open(const char *thinker_dir, rt_backend backend,
                            int max_seq, cwick_error *err);
void rt_thinker_close(rt_thinker *t);

const cwick_tokenizer *rt_thinker_tokenizer(const rt_thinker *t);
int rt_thinker_hidden(const rt_thinker *t);
int rt_thinker_layers(const rt_thinker *t);
int rt_thinker_vocab(const rt_thinker *t);
int rt_thinker_pos(const rt_thinker *t);
int rt_thinker_capacity(const rt_thinker *t);
bool rt_thinker_on_gpu(const rt_thinker *t);

/* Token id -> input embedding (f32, H values). */
int rt_thinker_embed_row(rt_thinker *t, int32_t id, float *out);

/* Drops the KV cache and rewinds to position 0. */
void rt_thinker_reset(rt_thinker *t);
/* Keeps the first `pos` positions, dropping everything after (barge-in). */
void rt_thinker_truncate(rt_thinker *t, int pos);

/* Runs `n` new positions whose input embeddings are given row-major
 * (n * H floats). Positions are allocated at the current end of the cache.
 * After the call, rt_thinker_logits() holds the last row's logits. */
int rt_thinker_forward(rt_thinker *t, const float *embeds, int n);

/* Logits of the last forwarded position (vocab floats, f32). The buffer is
 * reused by the next call. */
const float *rt_thinker_logits(rt_thinker *t);

#endif /* CWICK_RT_THINKER_H */
