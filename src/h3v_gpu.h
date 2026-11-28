/* h3v_gpu.h — Metal path for the video transformer and the visual VAE.
 *
 * The two compute stages have different checkpoint dtypes: the DiT stores BF16
 * weights and the VAE stores F32. Both are exposed as `out = x @ W^T` with the
 * activation and result in FP32, so one small set of kernels covers them. The
 * attention kernel uses the same tiled flash form for the DiT's 128-wide heads
 * and the decoder's 64-wide heads.
 *
 * Every entry point returns 0 when the shape or the policy does not suit the
 * GPU, and the caller then takes its existing CPU path. Nothing here is
 * required for a clip to be produced.
 */
#ifndef CWICK_H3V_GPU_H
#define CWICK_H3V_GPU_H

#include <stddef.h>
#include <stdint.h>

/* Brought up on first use. 0 means the CPU path stays in charge. */
int h3v_gpu_ready(void);
void h3v_gpu_shutdown(void);

/* Combined policy: honours the --backend metal selection and
 * CWICK_H3V_GPU=0 (an A/B escape hatch), then brings the device up lazily.
 * The result is cached after the first call. */
int h3v_gpu_on(void);

/* Register a whole weight mapping. Projections whose bytes fall inside a
 * registered mapping reuse its no-copy buffer instead of building a new view
 * (and faulting its pages) per projection. */
void h3v_gpu_adopt(const void *base, size_t len);

/* c[r][cols] = a[r][k] @ w[cols][k]^T with row strides lda / ldc.
 * `w` is BF16 for the DiT (gemm_bf16) and F32 for the VAE (gemm_f32).
 * Both stage the activation into the kernel's tile memory; the products
 * accumulate in FP32. Return 1 when the GPU wrote the result. */
int h3v_gpu_gemm_bf16(const float *a, const void *w, float *c,
                      int rows, int k, int cols, int lda, int ldc);
int h3v_gpu_gemm_f32(const float *a, const float *w, float *c,
                     int rows, int k, int cols, int lda, int ldc);
/* C += A @ W^T. Used when several convolution taps write one output tile;
 * the plain entry point above overwrites it. */
int h3v_gpu_gemm_f32_accum(const float *a, const float *w, float *c,
                           int rows, int k, int cols, int lda, int ldc);

typedef struct {
    const float *a;
    const void *w;
    float *c;
    int rows, k, cols, lda, ldc;
} h3v_gpu_bf16_item;

/* Independent BF16 GEMMs in one command buffer. Every item must use the same
 * BF16 pipeline; activation/result strides remain per-item. */
int h3v_gpu_gemm_bf16_batch(const h3v_gpu_bf16_item *items, int n);

/* Fused QKV projection from the release layout: weight rows are grouped per
 * head as [q | k | v] and `part` selects one of the three output planes.
 * `wrows` is the weight's full row count (3 * cols); `ldc` is the plane pitch. */
int h3v_gpu_gemm_qkv(const float *a, const void *w, float *c,
                     int rows, int k, int cols, int wrows, int lda, int ldc,
                     int part, int hd);
int h3v_gpu_gemm_qkv3(const float *a, const void *w, float *q, float *k,
                      float *v, int rows, int inner, int cols, int wrows,
                      int lda, int ldc, int hd);
int h3v_gpu_swiglu(float *x, int64_t n, int64_t half, int legacy);
int h3v_gpu_qk_prep(float *q, float *k, const float *wq, const float *wk,
                    const float *cos, const float *sin, int seq, int ld,
                    int heads, int hd, int rope_on, int rope_mode);

/* Non-causal attention over one packed [seq][heads*hd] sequence, per head.
 * Returns 1 when the GPU wrote all heads. */
int h3v_gpu_attention(const float *q, const float *k, const float *v, float *o,
                      int seq, int ld, int heads, int hd);

/* Causal single-token Inkling attention with GQA, relative bias and sink.
 * K/V use absolute rows for full attention and a ring for window > 0. */
int h3v_gpu_inkling_decode_attn(const float *q, const float *k, const float *v,
                                const float *r, const float *rel,
                                const float *sink, float *o, int pos,
                                int cache_rows, int ld, int nh, int kvh,
                                int d, int dr, int extent, int window);

/* Causal prefill tile with the same relative bias, sink and ring semantics. */
int h3v_gpu_inkling_prefill_attn(const float *q, const float *k, const float *v,
                                 const float *r, const float *rel,
                                 const float *sink, float *o, int qstart,
                                 int qrows, int cache_rows, int ld, int nh,
                                 int kvh, int d, int dr, int extent, int window);

/* Exposed for the shader-probe test, which compiles every kernel in the
 * production source rather than trusting a kernel-count constant. */
const char *h3v_gpu_shader_source(void);

#endif
