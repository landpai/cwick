/*
 * ds_ops.h — the pure-math half of the DeepSeek port.
 *
 * Nothing here knows about weights or files: every function takes plain float
 * buffers, so each one can be checked against an independent implementation
 * without loading a checkpoint. The model layer (ds_model.c) sits on top.
 *
 * Two rules the reference implementation is strict about, and so is this file:
 *
 *   1. Norms use rms_norm_eps = 1e-20 (not a usual 1e-6) and are computed in
 *      fp32 even when the stream is bf16.
 *   2. Several tensors are quantized and immediately dequantized before they
 *      are used (the window KV, the compressed KV, the indexer keys and
 *      queries, the attention output). Those round trips are lossy and are
 *      part of the model, not an optimization, so they happen here.
 */
#ifndef CWICK_DS_OPS_H
#define CWICK_DS_OPS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* rotary                                                               */
/* ------------------------------------------------------------------ */

/* Per-dimension frequencies for one RoPE configuration. The checkpoint uses
 * two: theta=10000 with YaRN off (the pure sliding-window layers) and
 * theta=160000 with YaRN on (everything that also reads compressed KV). */
typedef struct {
    int npairs;          /* rope_head_dim / 2 */
    float freq[64];      /* radians per position, per pair */
} ds_freqs;

/* base=theta, factor=YaRN factor, original=original_max_position_embeddings
 * (0 disables YaRN), beta_fast/beta_slow as in the config. */
void ds_freqs_init(ds_freqs *f, int rope_dim, double base, double factor,
                   int original, int beta_fast, int beta_slow);

/* Rotate the last 2*npairs elements of x in place, adjacent pairs as complex
 * numbers. inverse conjugates the rotation. */
void ds_rope(float *x, const ds_freqs *f, int64_t pos, int inverse);

/* ------------------------------------------------------------------ */
/* norms and activations                                                */
/* ------------------------------------------------------------------ */

/* y = w * x * rsqrt(mean(x^2) + eps), all in fp32. w == NULL means ones. */
void ds_rmsnorm(float *y, const float *x, const float *w, int n, float eps);

/* The same, with a bf16 weight tensor (what the checkpoint stores) and the
 * result rounded to bf16 the way the reference's `.to(dtype)` does. */
void ds_rmsnorm_bf16w(float *y, const float *x, const uint16_t *w, int n, float eps);

/* Round to nearest even, bfloat16. */
float ds_bf16_round(float v);

float ds_silu(float x);
void ds_softmax(float *x, int n);

/* ------------------------------------------------------------------ */
/* hyper-connections                                                    */
/* ------------------------------------------------------------------ */

/* One projection of the flattened stream plus Sinkhorn, giving the three
 * coefficient sets for one sublayer.
 *   x    [hc*d]  the flattened residual stream
 *   fn   [mix, hc*d] fp32   scale [3]   base [mix]   mix = (2+hc)*hc
 *   pre  [hc], post [hc], comb [hc*hc]  (row-major, comb[j*hc+k])
 * eps is hc_eps (1e-6), iters is hc_sinkhorn_iters (20). */
void ds_hc_mixes(const float *x, const float *fn, const float *scale,
                 const float *base, int hc, int d, int iters, float eps,
                 float *pre, float *post, float *comb);

/* Collapse the hc copies into one sublayer input: y = sum_j pre[j]*x[j*d..]. */
void ds_hc_pre(float *y, const float *x, const float *pre, int hc, int d);

/* Expand back out and mix the residual in through `comb`:
 *   y[j] = post[j] * x + sum_i comb[i*hc + j] * res[i]
 * `comb` is a mixing matrix over the residual copies: the sum runs along its
 * first index and the output copy is its second. Sinkhorn makes it doubly
 * stochastic, so it is close to (but not) a plain residual add. */
void ds_hc_post(float *y, const float *x, const float *res, const float *post,
                const float *comb, int hc, int d);

/* ------------------------------------------------------------------ */
/* quantize / dequantize round trips (the reference's inplace=True paths)  */
/* ------------------------------------------------------------------ */

/* act_quant(x, 32, "ue8m0", e8m0, True): per-32 power-of-two scale, e4m3
 * payload, written back as floats. Used for the window KV. */
void ds_qdq_fp8_e8m0(float *x, int n);

/* fp4_act_quant(x, 32, True) with the default e8m0 scale: per-32 power-of-two
 * scale, e2m1 payload. Used for the indexer keys and queries. */
void ds_qdq_fp4_e8m0(float *x, int n);

/* fp4_act_quant(x, block, True, scale_dtype=e4m3): per-`block` e4m3 scale
 * (a rounded value, not a power of two), e2m1 payload. Used for the compressed
 * KV with block=16. An all-zero block still gets a nonzero scale. */
void ds_qdq_fp4_e4m3(float *x, int n, int block);

/* ------------------------------------------------------------------ */
/* selection                                                           */
/* ------------------------------------------------------------------ */

/* Keep the k best (score, index) pairs. The chosen indices are written to
 * `out` in ascending order and the count is returned (min(k, n)).
 * Ties keep the lower index. `heap` and `heap_idx` need room for k entries. */
int ds_topk_scores(const float *sc, int n, int k, int *out,
                   float *heap, int *heap_idx);

/* ------------------------------------------------------------------ */
/* fp4 rounding                                                         */
/* ------------------------------------------------------------------ */

/* Round to the nearest e2m1 value, ties to the even index. |v| <= 6 assumed. */
float ds_fp4_round(float v);

#ifdef __cplusplus
}
#endif

#endif /* CWICK_DS_OPS_H */
