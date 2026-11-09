/* tensor.c — Apple Silicon math core.
 *
 *   - dense linear layers run as fp32 GEMMs (Accelerate/BNNS on arm64);
 *     BF16 and row-int8 weights are dequantized into an execution arena;
 *   - RMSNorm, softmax and SiTU use a fixed operation order (no fast-math
 *     approximation), so a given input always produces the same output;
 *   - an optional NEON MXFP4 GEMV kernel (mxfp4_gemv_rows) for the
 *     routed-expert path.
 */
#include "cwick.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define CWICK_GEMV_NEON 1
#endif

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
/* portable fallback */
static void cblas_sgemm_(int rows, int k, int cols, const float *a,
                         const float *b, float *c) {
    for (int r = 0; r < rows; r++) {
        for (int n = 0; n < cols; n++) {
            float acc = 0.0f;
            for (int i = 0; i < k; i++) acc += a[r * k + i] * b[n * k + i];
            c[r * cols + n] = acc;
        }
    }
}
#define cblas_sgemm(RM, TA, TB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    cblas_sgemm_((int)(M), (int)(K), (int)(N), (const float *)(A), \
                 (const float *)(B), (float *)(C))
#endif

#include "util.h"

/* ------------------------------------------------------------------ */
/* Dense linear: out[rows][k] = sum_j in[rows][j] * W[k][j]
 * W is row-major [k, j] (torch linear layout).                        */
/* ------------------------------------------------------------------ */

void cwick_gemm_f32(float *out, int64_t rows, int64_t k, int64_t cols,
                   const float *in, const float *w) {
    if (rows <= 0 || k <= 0 || cols <= 0) return;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)rows, (int)cols, (int)k,
                1.0f, in, (int)k, w, (int)k, 0.0f, out, (int)cols);
}

/* ------------------------------------------------------------------ */
/* Streaming (tile-based) dequantizing GEMM / GEMV                     */
/*                                                                     */
/* The released GLM-5.3-Flash checkpoint (306 GiB) is far larger than  */
/* any laptop's RAM, so no kernel may materialize a whole weight       */
/* matrix in fp32. The numbers that matter:                            */
/*     lm_head [154880, 4096] BF16  -> 2.54 GiB if materialized        */
/*     q_proj  [  8192, 4096] BF16  ->  134 MiB per call, 3x per KDA   */
/*     expert  [  2048, 4096] F8    ->   33 MiB per matrix             */
/* Every kernel below dequantizes into a small fixed tile and          */
/* accumulates, so extra memory is O(tile) no matter how large the     */
/* tensor is. That is what keeps the fp32 compute path inside a        */
/* bounded memory budget while the weights stream off the SSD.         */
/* ------------------------------------------------------------------ */
#define CWICK_DEQ_TILE 128 /* columns of W dequantized per pass */

#if defined(__aarch64__) && !defined(CWICK_DISABLE_NEON_DEQ)
#include <arm_neon.h>
static void deq_bf16_tile(float *dst, const uint16_t *src, int64_t n) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t h = vld1q_u16(src + i);
        vst1q_f32(dst + i, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h), 16)));
        vst1q_f32(dst + i + 4, vreinterpretq_f32_u32(vshll_high_n_u16(h, 16)));
    }
    for (; i < n; i++) dst[i] = cwick_bf16_to_f32(src[i]);
}
#else
static void deq_bf16_tile(float *dst, const uint16_t *src, int64_t n) {
    for (int64_t i = 0; i < n; i++) dst[i] = cwick_bf16_to_f32(src[i]);
}
#endif

/* FP8 E4M3 has only 256 codes: a flat LUT turns the inner loop into a
 * load and a multiply (no branches, no exponent math). */
static float g_fp8_lut[256];
static int g_fp8_lut_ready = 0;
static void fp8_lut_init(void) {
    if (g_fp8_lut_ready) return;
    for (int i = 0; i < 256; i++) g_fp8_lut[i] = cwick_fp8_e4m3_to_f32((uint8_t)i);
    g_fp8_lut_ready = 1;
}

/* ---- parallel decode GEMV ----------------------------------------
 *
 * The decode path (rows == 1) is where a 306 GiB checkpoint spends its
 * life: ~1400 matrix-vector products per token, every one of them reading
 * the whole weight matrix off the spine. Two things were wrong with the
 * original loops and both are fixed here:
 *
 *   1. They were scalar with a `double` accumulator. On Apple Silicon a
 *      scalar fp32 FMA retires 1 MAC/cycle while NEON retires 4, so the
 *      kernels were running at a quarter of one core's arithmetic rate.
 *   2. They ran on the caller's thread. 11 of 12 performance cores sat
 *      idle through every one of those 1400 calls.
 *
 * Each kernel below is split into a `_range` function that owns output rows
 * [c0, c1) and is dispatched through cwick_pool_for(). Slices are disjoint so
 * no reduction or atomic is needed.
 * ------------------------------------------------------------------ */

#if defined(__aarch64__) && defined(__ARM_NEON)
/* bf16 -> fp32 is a 16-bit left shift of the bit pattern, so widening with
 * vshll_n_u16 and reinterpreting does it in one instruction per 4 lanes. */
#define CWICK_BF16_VEC 1
static inline float32x4_t bf16_lo(uint16x8_t h) {
    return vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h), 16));
}
static inline float32x4_t bf16_hi(uint16x8_t h) {
    return vreinterpretq_f32_u32(vshll_high_n_u16(h, 16));
}
#endif

typedef struct {
    const void *w;
    const float *in;
    float *out;
    int64_t k, cols;
} gemv_arg;

typedef struct {
    const int8_t *w;
    const uint16_t *scales;
    const float *in;
    float *out;
    int64_t k, cols;
} gemv_i8_arg;

static void gemv_bf16_range(void *argp, int64_t c0, int64_t c1) {
    gemv_arg *a = (gemv_arg *)argp;
    const uint16_t *w = (const uint16_t *)a->w;
    const float *in = a->in;
    float *out = a->out;
    const int64_t k = a->k;
    for (int64_t c = c0; c < c1; c++) {
        const uint16_t *row = w + (size_t)c * (size_t)k;
#if defined(CWICK_BF16_VEC)
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
        int64_t j = 0;
        for (; j + 16 <= k; j += 16) {
            uint16x8_t h0 = vld1q_u16(row + j);
            uint16x8_t h1 = vld1q_u16(row + j + 8);
            a0 = vfmaq_f32(a0, bf16_lo(h0), vld1q_f32(in + j + 0));
            a1 = vfmaq_f32(a1, bf16_hi(h0), vld1q_f32(in + j + 4));
            a2 = vfmaq_f32(a2, bf16_lo(h1), vld1q_f32(in + j + 8));
            a3 = vfmaq_f32(a3, bf16_hi(h1), vld1q_f32(in + j + 12));
        }
        float32x4_t s = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
        float acc = vgetq_lane_f32(s, 0) + vgetq_lane_f32(s, 1) +
                    vgetq_lane_f32(s, 2) + vgetq_lane_f32(s, 3);
        for (; j < k; j++) acc += cwick_bf16_to_f32(row[j]) * in[j];
        out[c] = acc;
#else
        double acc = 0.0;
        for (int64_t j = 0; j < k; j++)
            acc += (double)cwick_bf16_to_f32(row[j]) * (double)in[j];
        out[c] = (float)acc;
#endif
    }
}

/* out[c] = sum_j x[j] * W[c][j]; W is [cols, k] row-major BF16.
 * Extra memory: none. Rows [c0,c1) are produced by one worker. */
void cwick_gemv_bf16(float *out, int64_t k, int64_t cols, const float *in,
                    const void *w_bf16) {
    if (k <= 0 || cols <= 0) return;
    gemv_arg a;
    a.w = w_bf16; a.in = in; a.out = out; a.k = k; a.cols = cols;
    cwick_pool_init();
    cwick_pool_for(cols, &a, gemv_bf16_range);
}

static void gemv_i8_range(void *argp, int64_t c0, int64_t c1) {
    gemv_i8_arg *a = (gemv_i8_arg *)argp;
    const int8_t *w = a->w;
    const uint16_t *scales = a->scales;
    const float *in = a->in;
    float *out = a->out;
    const int64_t k = a->k;
    for (int64_t c = c0; c < c1; c++) {
        const int8_t *row = w + (size_t)c * (size_t)k;
        float s = scales ? cwick_bf16_to_f32(scales[c]) : 1.0f;
#if defined(CWICK_BF16_VEC)
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
        int64_t j = 0;
        for (; j + 16 <= k; j += 16) {
            int8x16_t q = vld1q_s8(row + j);
            int16x8_t lo = vmovl_s8(vget_low_s8(q));
            int16x8_t hi = vmovl_high_s8(q);
            a0 = vfmaq_f32(a0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),
                           vld1q_f32(in + j + 0));
            a1 = vfmaq_f32(a1, vcvtq_f32_s32(vmovl_high_s16(lo)),
                           vld1q_f32(in + j + 4));
            a2 = vfmaq_f32(a2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),
                           vld1q_f32(in + j + 8));
            a3 = vfmaq_f32(a3, vcvtq_f32_s32(vmovl_high_s16(hi)),
                           vld1q_f32(in + j + 12));
        }
        float32x4_t t = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
        float acc = vgetq_lane_f32(t, 0) + vgetq_lane_f32(t, 1) +
                    vgetq_lane_f32(t, 2) + vgetq_lane_f32(t, 3);
        for (; j < k; j++) acc += (float)row[j] * in[j];
        out[c] = acc * s;
#else
        double acc = 0.0;
        for (int64_t j = 0; j < k; j++) acc += (double)row[j] * (double)in[j];
        out[c] = (float)(acc * (double)s);
#endif
    }
}

/* Row-int8 GEMV: one fp16 scale per output row, applied at the end so the
 * inner loop stays integer -> fp32 convert + FMA. */
void cwick_gemv_int8(float *out, int64_t k, int64_t cols, const float *in,
                    const int8_t *w, const uint16_t *scales) {
    if (k <= 0 || cols <= 0) return;
    gemv_i8_arg a;
    a.w = w; a.scales = scales; a.in = in; a.out = out; a.k = k; a.cols = cols;
    cwick_pool_init();
    cwick_pool_for(cols, &a, gemv_i8_range);
}

/* ---- batched (prefill) GEMM --------------------------------------- */

typedef struct {
    const uint16_t *w;
    const float *in;
    float *out;
    int64_t rows, k, cols;
    int64_t tile;
} gemm_bf16_arg;

/* The range is expressed in TILES, not columns: a worker that started in the
 * middle of a tile would emit the wrong output columns into `out`. */
static void gemm_bf16_range(void *argp, int64_t t0, int64_t t1) {
    gemm_bf16_arg *a = (gemm_bf16_arg *)argp;
    const int64_t tile = a->tile;
    int64_t c0 = t0 * tile, c1 = t1 * tile;
    if (c1 > a->cols) c1 = a->cols;
    if (c0 >= c1) return;
    float *arena = cwick_xmalloc((size_t)tile * (size_t)a->k * sizeof(float));
    if (!arena) return;
    for (int64_t base = c0; base < c1; base += tile) {
        int64_t n = c1 - base < tile ? c1 - base : tile;
        deq_bf16_tile(arena, a->w + (size_t)base * (size_t)a->k, n * a->k);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)a->rows, (int)n, (int)a->k,
                    1.0f, a->in, (int)a->k, arena, (int)a->k,
                    0.0f, a->out + base, (int)a->cols);
    }
    free(arena);
}

void cwick_gemm_bf16(float *out, int64_t rows, int64_t k, int64_t cols,
                    const float *in, const void *w_bf16) {
    if (rows <= 0 || k <= 0 || cols <= 0) return;
    if (rows == 1) { cwick_gemv_bf16(out, k, cols, in, w_bf16); return; }
    /* Tile over output rows of W: each worker holds CWICK_DEQ_TILE rows only,
     * so extra memory is O(threads * tile * k), never O(cols * k). The tile
     * deliberately stays small — lm_head is [154880, 4096], and growing the
     * tile to cut dispatch count would make each worker's arena 64 MiB. */
    int64_t tile = CWICK_DEQ_TILE;
    if (tile > cols) tile = cols;
    int64_t ntiles = (cols + tile - 1) / tile;
    gemm_bf16_arg a;
    a.w = (const uint16_t *)w_bf16; a.in = in; a.out = out;
    a.rows = rows; a.k = k; a.cols = cols; a.tile = tile;
    cwick_pool_init();
    cwick_pool_for(ntiles, &a, gemm_bf16_range);
}

typedef struct {
    const int8_t *w;
    const uint16_t *scales;
    const float *in;
    float *out;
    int64_t rows, k, cols;
    int64_t tile;
} gemm_i8_arg;

/* Dequantize a tile of rows into fp32, then one sgemm per tile. The old code
 * called sgemm once per output row with N=1, which is the worst possible
 * shape for Accelerate (a strided dot product per call). */
static void gemm_i8_range(void *argp, int64_t t0, int64_t t1) {
    gemm_i8_arg *a = (gemm_i8_arg *)argp;
    const int64_t tile = a->tile;
    int64_t c0 = t0 * tile, c1 = t1 * tile;
    if (c1 > a->cols) c1 = a->cols;
    if (c0 >= c1) return;
    float *arena = cwick_xmalloc((size_t)tile * (size_t)a->k * sizeof(float));
    if (!arena) return;
    for (int64_t base = c0; base < c1; base += tile) {
        int64_t n = c1 - base < tile ? c1 - base : tile;
        for (int64_t i = 0; i < n; i++) {
            const int8_t *row = a->w + (size_t)(base + i) * (size_t)a->k;
            float s = a->scales ? cwick_bf16_to_f32(a->scales[base + i]) : 1.0f;
            float *dst = arena + (size_t)i * (size_t)a->k;
            for (int64_t j = 0; j < a->k; j++) dst[j] = (float)row[j] * s;
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)a->rows, (int)n, (int)a->k,
                    1.0f, a->in, (int)a->k, arena, (int)a->k,
                    0.0f, a->out + base, (int)a->cols);
    }
    free(arena);
}

void cwick_gemm_int8(float *out, int64_t rows, int64_t k, int64_t cols,
                    const float *in, const int8_t *w, const uint16_t *scales) {
    if (rows <= 0 || k <= 0 || cols <= 0) return;
    if (rows == 1) { cwick_gemv_int8(out, k, cols, in, w, scales); return; }
    int64_t tile = CWICK_DEQ_TILE;
    if (tile > cols) tile = cols;
    int64_t ntiles = (cols + tile - 1) / tile;
    gemm_i8_arg a;
    a.w = w; a.scales = scales; a.in = in; a.out = out;
    a.rows = rows; a.k = k; a.cols = cols; a.tile = tile;
    cwick_pool_init();
    cwick_pool_for(ntiles, &a, gemm_i8_range);
}

/* FP8 E4M3 -> f32 (OCP MX / IEEE-style layout: 1 sign, 4 exponent bits with
 * BIAS 7, 3 mantissa bits; subnormal step 2^-9).
 *
 * Careful: E4M3 has NO infinity and its exponent field 15 is NOT reserved.
 *   0x7E = 448.0  (the largest finite value: (1 + 6/8) * 2^(15-7))
 *   0x7F / 0xFF = NaN  (exp == 15 AND mantissa == 7)
 * Treating every exp==15 byte as NaN/inf (as an earlier revision did) turns
 * roughly 1% of every quantized weight matrix -- exactly the largest-magnitude
 * weights, one per 128x128 block by construction -- into NaN and poisons the
 * whole forward pass. */
float cwick_fp8_e4m3_to_f32(uint8_t b) {
    int sign = (b >> 7) & 1;
    int exp = (b >> 3) & 0xF;
    int man = b & 0x7;
    float v;
    if (exp == 0) {
        /* subnormal: mantissa * 2^-9 (largest subnormal = 0.013671875) */
        v = (float)man * 0x1p-9f;
    } else if (exp == 15 && man == 7) {
        v = 0.0f / 0.0f; /* the only non-finite patterns in E4M3 */
    } else {
        v = (1.0f + (float)man / 8.0f) * ldexpf(1.0f, exp - 7);
    }
    return sign ? -v : v;
}

/* FP8 GEMV: out[c] = sum_j x[j] * W[c][j] * inv_scale[block(c, j)].
 * Decoding one 128-column tile at a time keeps the working set at 512 B
 * instead of cols * k floats, so an expert matrix costs no allocation at
 * all and the dequantized bytes never reach RAM. */
typedef struct {
    const uint8_t *w;
    const float *scale_inv;
    const float *in;
    float *out;
    int64_t k, cols, ncb;
} gemv_fp8_arg;

/* Declared here because the per-row worker below taps them. */
static uint64_t g_fp8_calls, g_fp8_ns;
static uint64_t g_fp8_kernel_ns, g_fp8_kernel_cnt;
static int g_fp8_stats = -1;

/* One 128-weight block: sum_j dequant(W[j]) * in[j], WITHOUT the block scale
 * (the caller applies that to the sum). The scalar form is the fallback for
 * non-ARM builds and the reference the NEON form is checked against. */
static float fp8_block_dot_scalar(const uint8_t *row, const float *in, int64_t n) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    int64_t j = 0;
    for (; j + 8 <= n; j += 8) {
        a0 += g_fp8_lut[row[j + 0]] * in[j + 0];
        a1 += g_fp8_lut[row[j + 1]] * in[j + 1];
        a2 += g_fp8_lut[row[j + 2]] * in[j + 2];
        a3 += g_fp8_lut[row[j + 3]] * in[j + 3];
        a4 += g_fp8_lut[row[j + 4]] * in[j + 4];
        a5 += g_fp8_lut[row[j + 5]] * in[j + 5];
        a6 += g_fp8_lut[row[j + 6]] * in[j + 6];
        a7 += g_fp8_lut[row[j + 7]] * in[j + 7];
    }
    float s = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
    for (; j < n; j++) s += g_fp8_lut[row[j]] * in[j];
    return s;
}

#ifdef CWICK_GEMV_NEON

/* Eight FP8 E4M3 bytes -> eight bf16 bit patterns, plus a mask marking the
 * lanes whose exponent field is zero.
 *
 * Why this beats the LUT: a byte-indexed gather costs THREE L1 loads per
 * weight (the byte, the LUT slot, the input element) and NEON has no gather,
 * so the scalar loop is pinned at ~1.5 cycles per weight no matter how it is
 * scheduled. bf16 and E4M3 are the same shape -- sign, exponent, mantissa,
 * biased so the mantissa sits directly under the exponent -- so the
 * conversion is a shift plus a bias add, which does vectorize: one ld1 for
 * sixteen weights.
 *
 * Concretely, b = s eeee mmm widened to 16 bits and shifted left by 4 lands
 * the fields at s@11 e@10..7 m@6..4. That is bf16's mantissa position
 * already (m@6..4) with the exponent four bits too low, so adding
 * 120 << 7 -- bf16 bias 127 minus E4M3 bias 7 -- finishes it. Only the sign
 * needs its own shift, from bit 11 up to bit 15.
 *
 * Exponent zero is the one case the identity cannot express: E4M3 spends it
 * on zero and the subnormals (mantissa * 2^-9), which have no counterpart in
 * a format whose smallest normal is 2^-126. Those lanes get flagged and the
 * caller redoes the block with the LUT. In a real expert matrix they are
 * 0.011% of the bytes -- the per-block scale keeps weights far inside the
 * normal range -- so the flag costs two instructions per eight weights and
 * the branch it guards is never taken. */
static inline void fp8x8_to_bf16(uint8x8_t v, uint16x8_t *bf) {
    /* vshll_n_u8 does the widen and the shift in one instruction: widening
     * and then shifting the u16 halves costs two. */
    uint16x8_t y = vshll_n_u8(v, 4);
    uint16x8_t e = vaddq_u16(vandq_u16(y, vdupq_n_u16(0x07F0)), vdupq_n_u16(0x3C00));
    uint16x8_t s = vshlq_n_u16(vandq_u16(y, vdupq_n_u16(0x0800)), 4);
    *bf = vorrq_u16(e, s);
    /* The zero/subnormal case is NOT detected here any more. Flagging it
     * per eight weights cost four instructions (mask, compare, mask, compare)
     * plus an OR to fold the flags together -- five of the thirty-three the
     * inner loop issues per sixteen weights. The caller now keeps a running
     * bytewise MINIMUM of the exponent fields instead, which is two
     * instructions per sixteen and catches the same cases: a byte has
     * exponent zero exactly when (b & 0x78) == 0, so the minimum over the
     * block is zero exactly when some byte needs the slow path. */
}

/* bf16 -> fp32 is just the 16 bits placed at the top of the word, which
 * vshll (shift-left-long) does while widening. */
static inline float32x4_t bf16x4_to_f32(uint16x4_t h) {
    return vreinterpretq_f32_u32(vshll_n_u16(h, 16));
}

static float fp8_block_dot_neon(const uint8_t *row, const float *in, int64_t n) {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
    uint8x16_t zacc = vdupq_n_u8(0xFF);   /* running min of the exponent fields */
    int64_t j = 0;
    /* Four independent accumulators: NEON's fp32 FMA has a 4-cycle latency,
     * so one accumulator would cap the loop at a quarter of the issue rate. */
    for (; j + 16 <= n; j += 16) {
        uint8x16_t v = vld1q_u8(row + j);
        uint16x8_t blo, bhi;
        fp8x8_to_bf16(vget_low_u8(v),  &blo);
        fp8x8_to_bf16(vget_high_u8(v), &bhi);
        zacc = vminq_u8(zacc, vandq_u8(v, vdupq_n_u8(0x78)));
        a0 = vfmaq_f32(a0, bf16x4_to_f32(vget_low_u16(blo)),  vld1q_f32(in + j + 0));
        a1 = vfmaq_f32(a1, bf16x4_to_f32(vget_high_u16(blo)), vld1q_f32(in + j + 4));
        a2 = vfmaq_f32(a2, bf16x4_to_f32(vget_low_u16(bhi)),  vld1q_f32(in + j + 8));
        a3 = vfmaq_f32(a3, bf16x4_to_f32(vget_high_u16(bhi)), vld1q_f32(in + j + 12));
    }
    /* Zero unless some byte in the block had exponent field zero. */
    if (vminvq_u8(zacc) == 0)
        return fp8_block_dot_scalar(row, in, n); /* zero / subnormal fp8 */
    float r = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; j < n; j++) r += g_fp8_lut[row[j]] * in[j]; /* n not a multiple of 16 */
    return r;
}
#endif /* CWICK_GEMV_NEON */

/* CWICK_GEMV_SCALAR=1 selects the scalar reference; it exists so the two can
 * be timed and diffed against each other. */
static int g_fp8_neon = -1;

static float fp8_block_dot(const uint8_t *row, const float *in, int64_t n) {
#ifdef CWICK_GEMV_NEON
    if (g_fp8_neon != 0) return fp8_block_dot_neon(row, in, n);
#endif
    return fp8_block_dot_scalar(row, in, n);
}

static void gemv_fp8_range(void *argp, int64_t c0, int64_t c1) {
    gemv_fp8_arg *a = (gemv_fp8_arg *)argp;
    uint64_t t0 = g_fp8_stats ? cwick_prof_ns() : 0;
    const int64_t k = a->k, ncb = a->ncb;
    const uint8_t *w = a->w;
    const float *scale_inv = a->scale_inv;
    const float *in = a->in;
    float *out = a->out;
    const bool aligned = (k % CWICK_ST_BLOCK) == 0;

    for (int64_t c = c0; c < c1; c++) {
        const uint8_t *row = w + (size_t)c * (size_t)k;
        const float *srow = scale_inv + (size_t)(c / CWICK_ST_BLOCK) * (size_t)ncb;
        double acc = 0.0;
        if (aligned) {
            /* The 128-element block carries ONE scale, so it is applied to
             * the block sum instead of to every weight: one multiply per 128
             * instead of 128.
             *
             * Block sums stay fp32 (so the rounding error does not grow with
             * k); only the k/128-way outer accumulation is double, as
             * before. */
            for (int64_t b = 0; b < k; b += CWICK_ST_BLOCK)
                acc += (double)fp8_block_dot(row + b, in + b, CWICK_ST_BLOCK) *
                       (double)srow[b / CWICK_ST_BLOCK];
        } else {
            /* ragged tail: fall back to a per-element scale lookup */
            for (int64_t j = 0; j < k; j++)
                acc += (double)g_fp8_lut[row[j]] * (double)in[j] *
                       (double)srow[j / CWICK_ST_BLOCK];
        }
        out[c] = (float)acc;
    }
    if (g_fp8_stats) {
        __sync_fetch_and_add(&g_fp8_kernel_ns, cwick_prof_ns() - t0);
        __sync_fetch_and_add(&g_fp8_kernel_cnt, 1);
    }
}

/* CWICK_GEMV_STATS=1: report the real per-call cost of the expert GEMV at
 * exit, split into the part the worker threads spend inside the kernel and
 * the part the caller spends dispatching and joining them. Worth keeping:
 * a standalone benchmark of this kernel measures ~150 us/call while the same
 * kernel inside a real decode run measures ~400 us/call, and the split is
 * what tells those two apart -- thread count, QoS, address spread and
 * allocation were all ruled out before it was added. */
static void fp8_stats_dump(void) {
    if (g_fp8_stats != 1 || !g_fp8_calls) return;
    double per_call = (double)g_fp8_ns / (double)g_fp8_calls / 1e3;
    double per_worker = g_fp8_kernel_cnt
                        ? (double)g_fp8_kernel_ns / (double)g_fp8_kernel_cnt / 1e3
                        : 0.0;
    fprintf(stderr,
            "fp8gemv: %llu calls, %.1f us/call (%.1f us in the kernel, "
            "%.1f us dispatch/join), %.2f GMAC/s, pool %d thread(s)\n",
            (unsigned long long)g_fp8_calls, per_call, per_worker,
            per_call - per_worker,
            (double)4096 * 2048 * (double)g_fp8_calls / ((double)g_fp8_ns / 1e9) / 1e9,
            cwick_pool_threads());
}

/* Several input rows against the same weight matrix.
 *
 * Prefill feeds a batch of tokens through the model, and three or four of them
 * usually route to the same expert. Calling the GEMV once per token reads the
 * whole 8 MiB matrix back out of DRAM once per token. Putting the token loop
 * inside the column loop instead keeps ONE column of the matrix -- k bytes,
 * 4 KiB at this checkpoint's k=4096 -- in L1 across the whole token group, so
 * the matrix crosses DRAM exactly once however many tokens share it. Nothing
 * has to be unrolled or re-blocked for this; it is pure loop order. */
typedef struct {
    const uint8_t *w;
    const float *scale_inv;
    const float *in;   /* [rows][k] */
    float *out;        /* [rows][cols] */
    int64_t k, cols, ncb, rows;
} gemv_fp8_rows_arg;

static void gemv_fp8_rows_range(void *argp, int64_t c0, int64_t c1) {
    gemv_fp8_rows_arg *a = (gemv_fp8_rows_arg *)argp;
    const int64_t k = a->k, ncb = a->ncb, rows = a->rows, cols = a->cols;
    const uint8_t *w = a->w;
    const bool aligned = (k % CWICK_ST_BLOCK) == 0;

    for (int64_t c = c0; c < c1; c++) {
        const uint8_t *row = w + (size_t)c * (size_t)k;
        const float *srow = a->scale_inv + (size_t)(c / CWICK_ST_BLOCK) * (size_t)ncb;
        for (int64_t s = 0; s < rows; s++) {
            const float *in = a->in + (size_t)s * (size_t)k;
            double acc = 0.0;
            if (aligned) {
                for (int64_t b = 0; b < k; b += CWICK_ST_BLOCK)
                    acc += (double)fp8_block_dot(row + b, in + b, CWICK_ST_BLOCK) *
                           (double)srow[b / CWICK_ST_BLOCK];
            } else {
                for (int64_t j = 0; j < k; j++)
                    acc += (double)g_fp8_lut[row[j]] * (double)in[j] *
                           (double)srow[j / CWICK_ST_BLOCK];
            }
            a->out[(size_t)s * (size_t)cols + c] = (float)acc;
        }
    }
}

/* Opt-in: send the batched prefill GEMV to the GPU.
 *
 * The CPU path below is a fork-join over `cols` with NEON dot products, and
 * it measured 1.20 ms per 8 MiB expert matrix during a real prefill -- against
 * 0.188 ms for the same matrix in decode, which runs on the GPU. The Metal
 * kernel already takes `n_rows`/`stride` (its grid is 2D and it was written
 * for exactly this shape), so nothing new is needed on the GPU side; only this
 * call was never wired up.
 *
 * It is opt-in because the GPU can only bind a blob that lives in the
 * GPU-visible arena, and prefill's staging buffers were ordinary malloc on
 * purpose (see expert_view_into in moe.c, where arena memory once turned a
 * 43 s prefill into one that never finished). CWICK_PREFILL_METAL=1 switches
 * those allocations to the arena as well, so the two halves have to agree --
 * and if they do not, cwick_metal_gemv simply refuses and we fall through to
 * the CPU path, which is why this is safe to try unconditionally.
 *
 * MEASURED: off by default because it loses. Called this way it is one
 * commit-and-wait per matrix, which costs 0.97 ms against the CPU's 0.93 ms --
 * no win at all. The batched form in moe.c does win 23% on compute but makes
 * the SSD about twice as slow, so that loses too; see the table there. */
static int g_prefill_metal = -1;
static bool g_prefill_metal_susp = false;   /* a batched group fell back */
static bool prefill_metal_wanted(void) {
    if (g_prefill_metal < 0)
        g_prefill_metal = cwick_env_true("CWICK_PREFILL_METAL") ? 1 : 0;
    return g_prefill_metal == 1 && !g_prefill_metal_susp;
}

/* Called by the batched prefill group when Metal declines the whole group.
 * Without this the fallback would re-enter cwick_metal_gemv once per matrix --
 * one commit-and-wait each, which is exactly the cost the batch exists to
 * avoid -- so a failed batch would be slower than never having tried. */
void cwick_prefill_metal_suspend(bool on) { g_prefill_metal_susp = on; }

/* Whether the GPU path was actually taken, and how many times it declined.
 * Without this the A/B is unreadable: cwick_metal_gemv returns 0 for reasons
 * the caller cannot see (blob outside the arena, k not a multiple of 128), and
 * a silent fall-through to the CPU looks exactly like "the GPU is not faster".
 */
static uint64_t g_pf_metal_ok, g_pf_metal_miss;
static void prefill_metal_dump(void) {
    if (!g_pf_metal_ok && !g_pf_metal_miss) return;
    fprintf(stderr, "prefill gemv: %llu/%llu on the GPU, %llu fell back to CPU\n",
            (unsigned long long)g_pf_metal_ok,
            (unsigned long long)(g_pf_metal_ok + g_pf_metal_miss),
            (unsigned long long)g_pf_metal_miss);
}

void cwick_gemv_fp8_rows(float *out, int64_t rows, int64_t k, int64_t cols,
                        const float *in, const void *w_fp8,
                        const float *scale_inv, int64_t ncb) {
    if (rows <= 0 || k <= 0 || cols <= 0 || !scale_inv || ncb <= 0) return;
    if (prefill_metal_wanted()) {
        /* `in` is [rows][k] row-major, so the row stride is exactly k. */
        int ok = cwick_metal_gemv(w_fp8, scale_inv, in, out, k, cols, ncb,
                                 rows, k);
        if (ok) g_pf_metal_ok++;
        else    g_pf_metal_miss++;
        /* Register on the first call either way -- counting only the failures
         * is how a run where every dispatch succeeded ends up silent, which
         * reads exactly like "the GPU path was never reached". */
        if (g_pf_metal_ok + g_pf_metal_miss == 1) atexit(prefill_metal_dump);
        if (ok) return;
    }
    fp8_lut_init();
    if (g_fp8_neon < 0)
        g_fp8_neon = cwick_env_true("CWICK_GEMV_SCALAR") ? 0 : 1;
    gemv_fp8_rows_arg a;
    a.w = (const uint8_t *)w_fp8; a.scale_inv = scale_inv; a.in = in; a.out = out;
    a.k = k; a.cols = cols; a.ncb = ncb; a.rows = rows;
    cwick_pool_init();
    cwick_pool_for(cols, &a, gemv_fp8_rows_range);
}

void cwick_gemv_fp8(float *out, int64_t k, int64_t cols, const float *in,
                   const void *w_fp8, const float *scale_inv, int64_t ncb) {
    if (k <= 0 || cols <= 0 || !scale_inv || ncb <= 0) return;
    fp8_lut_init();
    if (g_fp8_neon < 0)
        g_fp8_neon = cwick_env_true("CWICK_GEMV_SCALAR") ? 0 : 1;
    if (g_fp8_stats < 0) {
        g_fp8_stats = cwick_env_true("CWICK_GEMV_STATS") ? 1 : 0;
        if (g_fp8_stats) atexit(fp8_stats_dump);
    }
    uint64_t tfp8 = g_fp8_stats ? cwick_prof_ns() : 0;
    gemv_fp8_arg a;
    a.w = (const uint8_t *)w_fp8; a.scale_inv = scale_inv; a.in = in;
    a.out = out; a.k = k; a.cols = cols; a.ncb = ncb;
    cwick_pool_init();
    cwick_pool_for(cols, &a, gemv_fp8_range);
    if (g_fp8_stats) { g_fp8_ns += cwick_prof_ns() - tfp8; g_fp8_calls++; }
}

/* Output rows [c0, c1) of one matrix, with no dispatch of its own.
 *
 * This is what lets a whole routed expert group go out as two dispatches
 * instead of twenty-four. A decode step calls the expert GEMV ~1000 times,
 * and each call used to be its own fork-join — waking twelve workers and
 * then the caller, which costs more than the 280 us kernel it was starting.
 * The caller owns the parallelism across experts and rows; this is the leaf. */
void cwick_gemv_fp8_sub(float *out, int64_t k, int64_t cols, int64_t c0, int64_t c1,
                       const float *in, const void *w_fp8, const float *scale_inv,
                       int64_t ncb) {
    if (k <= 0 || cols <= 0 || !scale_inv || ncb <= 0 || c1 <= c0) return;
    fp8_lut_init();
    if (g_fp8_neon < 0)
        g_fp8_neon = cwick_env_true("CWICK_GEMV_SCALAR") ? 0 : 1;
    gemv_fp8_arg a;
    a.w = (const uint8_t *)w_fp8; a.scale_inv = scale_inv; a.in = in;
    a.out = out; a.k = k; a.cols = cols; a.ncb = ncb;
    uint64_t t0 = cwick_prof_ns();
    gemv_fp8_range(&a, c0, c1);
    if (g_fp8_stats) { g_fp8_ns += cwick_prof_ns() - t0; g_fp8_calls++; }
}

void cwick_gemm_fp8(float *out, int64_t rows, int64_t k, int64_t cols,
                   const float *in, const void *w_fp8, const float *scale_inv,
                   int64_t nrb, int64_t ncb) {
    if (rows <= 0 || k <= 0 || cols <= 0) return;
    if (rows == 1) {
        cwick_gemv_fp8(out, k, cols, in, w_fp8, scale_inv, ncb);
        return;
    }
    (void)nrb;
    const uint8_t *w = (const uint8_t *)w_fp8;
    int64_t tile = CWICK_DEQ_TILE;
    if (tile > cols) tile = cols;
    float *arena = cwick_xmalloc((size_t)tile * (size_t)k * sizeof(float));
    for (int64_t c0 = 0; c0 < cols; c0 += tile) {
        int64_t n = cols - c0 < tile ? cols - c0 : tile;
        for (int64_t i = 0; i < n; i++) {
            const uint8_t *row = w + (size_t)(c0 + i) * (size_t)k;
            float *dst = arena + (size_t)i * (size_t)k;
            int64_t rb = (c0 + i) / CWICK_ST_BLOCK;
            const float *srow = scale_inv + rb * ncb;
            for (int64_t j = 0; j < k; j++)
                dst[j] = cwick_fp8_e4m3_to_f32(row[j]) * srow[j / CWICK_ST_BLOCK];
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)rows, (int)n, (int)k,
                    1.0f, in, (int)k, arena, (int)k, 0.0f, out + c0, (int)cols);
    }
    free(arena);
}

/* Linear with a tensor that may be BF16, f32, row-int8, or FP8 E4M3. */
void cwick_linear(float *out, const float *in, int64_t rows, int64_t cols_in,
                 int64_t cols_out, const cwick_tensor *t) {
    if (t->kind == CWICK_WEIGHT_ROW_I8_F16) {
        cwick_gemm_int8(out, rows, cols_in, cols_out, in,
                       (const int8_t *)t->data, (const uint16_t *)t->scales);
    } else if (t->kind == CWICK_WEIGHT_F32) {
        cwick_gemm_f32(out, rows, cols_in, cols_out, in, (const float *)t->data);
    } else if (t->kind == CWICK_WEIGHT_FP8_E4M3) {
        int64_t nrb = t->rows / CWICK_ST_BLOCK;
        int64_t ncb = t->cols / CWICK_ST_BLOCK;
        cwick_gemm_fp8(out, rows, cols_in, cols_out, in, t->data,
                      (const float *)t->scales, nrb, ncb);
    } else if (t->kind == CWICK_WEIGHT_BF16) {
        if (rows == 1 && cwick_metal_linear_bf16(t->data, in, out,
                                                 cols_in, cols_out))
            return;
        cwick_gemm_bf16(out, rows, cols_in, cols_out, in, t->data);
    } else {
        cwick_gemm_bf16(out, rows, cols_in, cols_out, in, t->data);
    }
}

float cwick_tensor_value(const cwick_tensor *t, int64_t index) {
    if (!t || !t->data || index < 0 || index >= t->rows * t->cols) return 0.0f;
    if (t->kind == CWICK_WEIGHT_BF16)
        return cwick_bf16_to_f32(((const uint16_t *)t->data)[index]);
    if (t->kind == CWICK_WEIGHT_F32)
        return ((const float *)t->data)[index];
    return 0.0f;
}

/* ------------------------------------------------------------------ */
/* Normalization / activations                                         */
/* ------------------------------------------------------------------ */

void cwick_rms_norm(float *out, const float *in, int64_t n,
                   const float *weight, float eps) {
    double ss = 0.0;
    for (int64_t i = 0; i < n; i++) ss += (double)in[i] * (double)in[i];
    float rms = (float)sqrt(ss / (double)n + (double)eps);
    float inv = 1.0f / rms;
    if (weight) {
        for (int64_t i = 0; i < n; i++) out[i] = in[i] * inv * weight[i];
    } else {
        for (int64_t i = 0; i < n; i++) out[i] = in[i] * inv;
    }
}

void cwick_softmax(float *scores, int64_t n) {
    float mx = scores[0];
    for (int64_t i = 1; i < n; i++) if (scores[i] > mx) mx = scores[i];
    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) {
        scores[i] = expf(scores[i] - mx);
        sum += scores[i];
    }
    float inv = (float)(1.0 / sum);
    for (int64_t i = 0; i < n; i++) scores[i] *= inv;
}

void cwick_silu_vec(float *x, int64_t n) {
    for (int64_t i = 0; i < n; i++) x[i] = x[i] * (1.0f / (1.0f + expf(-x[i])));
}

/* GLM-5.3-Flash SwiGLU with clamp (reference: vLLM SiluAndMulWithClamp,
 * activation.py) — the released config sets hidden_act="silu" and
 * swiglu_limit=10.0, so:
 *     gate = clamp(x[..., :d], max=limit)
 *     up   = clamp(x[..., d:], min=-limit, max=limit)
 *     out  = gate * sigmoid(gate) * up
 * Note the asymmetry: the gate is clamped from above only (it feeds the
 * sigmoid), while `up` is clamped symmetrically. Getting this backwards
 * changes logits, so it mirrors the reference exactly. */
void cwick_swiglu_clamp(float *out, const float *gate, const float *up,
                       int64_t n, float limit) {
    for (int64_t i = 0; i < n; i++) {
        float g = gate[i];
        if (g > limit) g = limit;
        float u = up[i];
        if (u > limit) u = limit;
        else if (u < -limit) u = -limit;
        out[i] = g * (1.0f / (1.0f + expf(-g))) * u;
    }
}

void cwick_situ(float *out, const float *gate, const float *up, int64_t n,
               float beta, float linear_beta) {
    for (int64_t i = 0; i < n; i++) {
        float g = gate[i];
        float bounded_gate = tanhf(g / beta) * beta * (1.0f / (1.0f + expf(-g)));
        float bounded_up = tanhf(up[i] / linear_beta) * linear_beta;
        out[i] = bounded_gate * bounded_up;
    }
}

/* ------------------------------------------------------------------ */
/* MXFP4 expert GEMV                                                */
/*                                                                     */
/* Packed layout: rows x (cols/2) bytes, two e2m1 nibbles per byte     */
/* (LOW first); scales: rows x (cols/32) e8m0 bytes. Output is fp32.   */
/* A portable scalar reference and the arm64 NEON fused kernel share   */
/* one accumulation order; the unit test cross-checks them against a   */
/* hand-computed spec, not merely against each other.                  */
/* ------------------------------------------------------------------ */
static float e2m1_to_f32(int nib) {
    int sign = (nib >> 3) & 1;
    int exp = (nib >> 1) & 3;
    int man = nib & 1;
    float v;
    if (exp == 0) {
        /* subnormal: mantissa is 0 or 2^-1 */
        v = man ? 0.5f : 0.0f;
    } else {
        /* normal: (1 + man * 2^-1) * 2^(exp - 1) */
        v = (1.0f + man * 0.5f) * ldexpf(1.0f, exp - 1);
    }
    return sign ? -v : v;
}

static float e8m0_to_f32(int byte) {
    /* e8m0: pure 8-bit exponent (no sign, no mantissa), bias 127 */
    return ldexpf(1.0f, byte - 127);
}

/* y[r] = sum_j x[j] * W[r][j]; W is [rows, cols/2] packed + [rows, cols/32]
 * scales. Scalar reference (exact dequant, same per-group order). */
void mxfp4_gemv_rows_ref(const uint8_t *p, const uint8_t *s,
                         const float *x, float *y,
                         int row0, int row1, int cols) {
    const int groups = cols / 32;
    for (int r = row0; r < row1; r++) {
        const uint8_t *pp = p + (size_t)r * (cols / 2);
        const uint8_t *sp = s + (size_t)r * groups;
        double acc = 0.0;
        for (int g = 0; g < groups; g++) {
            float scale = e8m0_to_f32(sp[g]);
            for (int i = 0; i < 16; i++) {
                uint8_t b = pp[g * 16 + i];
                double v0 = (double)(e2m1_to_f32(b & 0xF) * scale);
                double v1 = (double)(e2m1_to_f32(b >> 4) * scale);
                acc += v0 * (double)x[g * 32 + 2 * i] +
                       v1 * (double)x[g * 32 + 2 * i + 1];
            }
        }
        y[r] = (float)acc;
    }
}

#if defined(__aarch64__) && !defined(CWICK_DISABLE_NEON_MXFP4)
#include <arm_neon.h>

/* fp32 top-16 bits of each e2m1 value; low 16 bits are always zero for the
 * e2m1 set, so multiplying by an e8m0 scale is (scale-127) added to the
 * exponent field, i.e. ((s-127)<<7) added to the top-16 halfword. */
static const uint16_t mxfp4_base16[16] = {
    0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0,
    0x8000, 0xBF00, 0xBF80, 0xBFC0, 0xC000, 0xC040, 0xC080, 0xC0C0,
};
static const uint16_t mxfp4_mask16[16] = {
    0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
    0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
};

typedef struct { uint8_t lo[16]; uint8_t hi[16]; } mxfp4_scale_row;
static mxfp4_scale_row mxfp4_scale_lut[256];
static int mxfp4_lut_ready = 0;

static void mxfp4_lut_init(void) {
    if (mxfp4_lut_ready) return;
    for (int s = 0; s < 256; s++) {
        uint16_t delta = (uint16_t)(((int)s - 127) * 128);
        for (int n = 0; n < 16; n++) {
            uint16_t bits = (uint16_t)(mxfp4_base16[n] +
                                       (uint16_t)(delta & mxfp4_mask16[n]));
            mxfp4_scale_lut[s].lo[n] = (uint8_t)bits;
            mxfp4_scale_lut[s].hi[n] = (uint8_t)(bits >> 8);
        }
    }
    mxfp4_lut_ready = 1;
}

static inline float mxfp4_reduce8(float32x4_t a0, float32x4_t a1,
                                  float32x4_t a2, float32x4_t a3,
                                  float32x4_t a4, float32x4_t a5,
                                  float32x4_t a6, float32x4_t a7) {
    float32x4_t b0 = vaddq_f32(a0, a1), b1 = vaddq_f32(a2, a3);
    float32x4_t b2 = vaddq_f32(a4, a5), b3 = vaddq_f32(a6, a7);
    float32x4_t c0 = vaddq_f32(b0, b1), c1 = vaddq_f32(b2, b3);
    return vaddvq_f32(vaddq_f32(c0, c1));
}

/* y[rows] = x[cols] * W^T — fused kernel, byte-LUT form. */
void mxfp4_gemv_rows(const uint8_t *restrict p, const uint8_t *restrict s,
                     const float *restrict x, float *restrict y,
                     int row0, int row1, int cols) {
    const int cp = cols / 2, groups = cols / 32;
    const uint8x16_t M0F = vdupq_n_u8(0x0F);
    mxfp4_lut_init();
    for (int r = row0; r < row1; r++) {
        const uint8_t *pp = p + (size_t)r * cp;
        const uint8_t *sp = s + (size_t)r * groups;
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
        float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
        float32x4_t a4 = vdupq_n_f32(0), a5 = vdupq_n_f32(0);
        float32x4_t a6 = vdupq_n_f32(0), a7 = vdupq_n_f32(0);
        for (int g = 0; g < groups; g++) {
            const mxfp4_scale_row *row = &mxfp4_scale_lut[sp[g]];
            uint8x16_t TABL = vld1q_u8(row->lo);
            uint8x16_t TABH = vld1q_u8(row->hi);
            uint8x16_t P = vld1q_u8(pp + (size_t)g * 16);
            uint8x16_t lo = vandq_u8(P, M0F);
            uint8x16_t hi = vshrq_n_u8(P, 4);
            uint8x16_t n0 = vzip1q_u8(lo, hi); /* logical elements 0..15 */
            uint8x16_t n1 = vzip2q_u8(lo, hi); /* elements 16..31 */
            const float *xp = x + (size_t)g * 32;
            {
                uint8x16_t bl = vqtbl1q_u8(TABL, n0);
                uint8x16_t bh = vqtbl1q_u8(TABH, n0);
                uint16x8_t h0 = vreinterpretq_u16_u8(vzip1q_u8(bl, bh));
                uint16x8_t h1 = vreinterpretq_u16_u8(vzip2q_u8(bl, bh));
                a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h0), 16)), vld1q_f32(xp));
                a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_high_n_u16(h0, 16)), vld1q_f32(xp + 4));
                a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h1), 16)), vld1q_f32(xp + 8));
                a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_high_n_u16(h1, 16)), vld1q_f32(xp + 12));
            }
            {
                uint8x16_t bl = vqtbl1q_u8(TABL, n1);
                uint8x16_t bh = vqtbl1q_u8(TABH, n1);
                uint16x8_t h0 = vreinterpretq_u16_u8(vzip1q_u8(bl, bh));
                uint16x8_t h1 = vreinterpretq_u16_u8(vzip2q_u8(bl, bh));
                a4 = vfmaq_f32(a4, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h0), 16)), vld1q_f32(xp + 16));
                a5 = vfmaq_f32(a5, vreinterpretq_f32_u32(vshll_high_n_u16(h0, 16)), vld1q_f32(xp + 20));
                a6 = vfmaq_f32(a6, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h1), 16)), vld1q_f32(xp + 24));
                a7 = vfmaq_f32(a7, vreinterpretq_f32_u32(vshll_high_n_u16(h1, 16)), vld1q_f32(xp + 28));
            }
        }
        y[r] = mxfp4_reduce8(a0, a1, a2, a3, a4, a5, a6, a7);
    }
}
#else
void mxfp4_gemv_rows(const uint8_t *restrict p, const uint8_t *restrict s,
                     const float *restrict x, float *restrict y,
                     int row0, int row1, int cols) {
    mxfp4_gemv_rows_ref(p, s, x, y, row0, row1, cols);
}
#endif
