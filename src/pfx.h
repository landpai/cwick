/* pfx.h — prefix cache: snapshot and restore the inference state of a token
 * prefix, so a prompt that repeats (every turn of an agent session sends the
 * same 30K-token system prompt plus tool declarations) is prefilled once and
 * replayed from disk afterwards.
 *
 * Why this exists: on the released checkpoint a 29,798-token prefill costs
 * ~2.8 hours, and it is the *same* 29,798 tokens on every turn. The state of
 * that prefix is ~0.9 GB and reads back in ~0.3 s, so the cache is worth four
 * orders of magnitude.
 *
 * What has to be saved is the whole cross-token state, not just the KV cache:
 *
 *   keys/values    one row per position per sparse-attention layer
 *   kda_rec        the DeltaNet recurrent accumulator [L][heads][dim][dim]
 *   kda_conv       the DeltaNet conv1d window (last 3 tokens per layer)
 *   indexer pools  the DSA compressed rows + the open pool's tail
 *   pos            where the next token goes
 *
 * The mHC state is deliberately absent: it is *intra-token* (layer 0 expands
 * the embedding, each later layer mixes the previous layer's output for the
 * same token), so it never crosses a token boundary and nothing about it
 * depends on history. That is what makes this checkpoint resumable at all:
 * the recurrent parts above are small and snapshot whole, which is a different
 * thing from being able to *rewind* them by dropping tokens -- that they
 * cannot do.
 *
 * Storage is one file per prefix under a cache directory, named by a 128-bit
 * hash of the token sequence. A hit is confirmed by comparing the stored
 * tokens, so a hash collision costs a memcmp and never corrupts a generation.
 */
#ifndef CWICK_PFX_H
#define CWICK_PFX_H

#include <stdbool.h>
#include <stdint.h>

#include "cwick.h"

/* The serializable state of one prefix. All buffers are owned by the state
 * and freed by cwick_prefix_state_free. `idx_pool` is allocated at pool_cap
 * rows but only the first idx_npools[s] rows of slot s are meaningful. */
typedef struct cwick_prefix_state {
    /* Which backend captured this. 0 = the mHC/KDA/DSA checkpoint described
     * below; 1 = DeepSeek-V4.1, 2 = MiniMax-M3.1, 3 = Qwen3.5, whose
     * cross-token state is a different set of buffers and travels as an opaque
     * `blob` (built by ds_state_pack / mm_state_pack / q35_prefix_pack). The
     * file format carries the same flag, so one cache directory can hold all
     * four kinds without any of them being able to misread another. */
    int kind;
    void *blob;           /* kind 1..3: backend-owned snapshot bytes */
    size_t blob_len;

    int64_t pos;          /* prefix length in tokens */
    int64_t nslots;       /* layers that own a KV slot (DSA/MLA) */
    int64_t lat_stride;   /* floats per position in keys (kv_lora, or k+v) */
    int64_t v_stride;     /* floats per position in values; 0 in latent mode */
    int64_t L;            /* layers */
    int64_t lin_heads, lin_dim;   /* DeltaNet recurrent dims */
    int64_t kpool, hd;    /* DSA indexer: positions per pool, head dim */
    int64_t pool_cap;
    int64_t H;            /* hidden size (anchors are H-wide) */
    int64_t max_anchors;
    int64_t n_anchors;    /* legacy residual-anchor path; 0 on mHC checkpoints */

    float *keys;          /* [nslots][pos][lat_stride] */
    float *values;        /* [nslots][pos][v_stride], NULL in latent mode */
    float *kda_rec;       /* [L][lin_heads][lin_dim][lin_dim] */
    float *kda_conv;      /* [L][3][lin_heads*lin_dim][3] */
    float *idx_pool;      /* [nslots][pool_cap][hd] */
    float *idx_tail;      /* [nslots][2][kpool][hd] */
    int64_t *idx_npools;  /* [nslots] */
    float *anchors;       /* [max_anchors][H] */
} cwick_prefix_state;

void cwick_prefix_state_free(cwick_prefix_state *s);

/* Allocate every buffer for a state of these dimensions (used by engine.c
 * when capturing, and by the loader when reading one back). */
bool cwick_prefix_state_alloc_dims(cwick_prefix_state *s, int64_t pos,
                                  int64_t nslots, int64_t lat_stride,
                                  int64_t v_stride, int64_t L, int64_t lin_heads,
                                  int64_t lin_dim, int64_t kpool, int64_t hd,
                                  int64_t pool_cap, int64_t H, int64_t max_anchors);

/* Copy the first `pos` positions of the engine's live state into `out`.
 * `pos` must be <= eng->pos. Everything the next token needs to continue is
 * copied, so the engine is free to be reset (or the process to exit) after. */
cwick_status cwick_engine_capture_prefix(cwick_engine *eng, int64_t pos,
                                       cwick_prefix_state *out, cwick_error *err);

/* Put a captured state back. Sets eng->pos = s->pos; the caller then prefills
 * only the tokens after the prefix. */
cwick_status cwick_engine_apply_prefix(cwick_engine *eng, const cwick_prefix_state *s,
                                     cwick_error *err);

/* ------------------------------------------------------------------ */
/* On-disk cache                                                       */
/* ------------------------------------------------------------------ */
typedef struct cwick_pfx_cache cwick_pfx_cache;

/* `dir` is created if missing; `cap_bytes` bounds the total size of the
 * cache (oldest-first eviction). Returns NULL without touching anything if
 * the directory cannot be used -- a cache is an optimisation, and a serve
 * that cannot write one must still serve. */
cwick_pfx_cache *cwick_pfx_open(const char *dir, uint64_t cap_bytes);
void cwick_pfx_close(cwick_pfx_cache *c);

/* Length of the longest cached prefix that matches ids[0..P), or 0 for a
 * miss. Cheap: one pass of two rolling hashes over the prompt, then a
 * comparison per cached entry. */
int64_t cwick_pfx_lookup(cwick_pfx_cache *c, const uint32_t *ids, int64_t n);

/* Read the cached prefix of exactly length `P` back into `out`. The stored
 * tokens are compared against ids before anything is returned, so a hit is
 * exact rather than probabilistic. */
cwick_status cwick_pfx_load(cwick_pfx_cache *c, const uint32_t *ids, int64_t P,
                          cwick_prefix_state *out, cwick_error *err);

/* Write `s` (the state of ids[0..P)) to the cache. */
cwick_status cwick_pfx_store(cwick_pfx_cache *c, const uint32_t *ids, int64_t P,
                           const cwick_prefix_state *s, cwick_error *err);

/* Diagnostics for the startup banner and the access log. */
void cwick_pfx_stats(const cwick_pfx_cache *c, int64_t *entries, uint64_t *bytes);

/* Delete every snapshot in the cache. Call it when the resident checkpoint
 * changes: an entry is only valid for the geometry it was captured under. */
void cwick_pfx_clear(cwick_pfx_cache *c);

#endif /* CWICK_PFX_H */
