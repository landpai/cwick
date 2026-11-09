/*
 * mm_model.c — MiniMax-M3.1-Flash: config, weights and the forward pass.
 *
 * See mm.h for the storage layouts; tools/mm_probe prints them for any layer.
 *
 * Two things about this port are worth knowing before reading it:
 *
 * 1. The routed experts are 216 GiB of packed FP4 and are read through the
 *    same pin cache the other backends use. A decode step touches 4 of 128
 *    per layer, so the working set is 6.8 GiB per step -- far more than the
 *    cache can hold, which is why the budget, not the hit rate, decides the
 *    token rate.
 *
 * 2. Prefill is layer-major over a chunk of MM_CHUNK positions. Running one
 *    position at a time would read the whole model per token; a chunk reads
 *    each layer's weights once for the whole chunk. The cost is that a chunk
 *    reaches the union of every position's routing choice, so the MoE half
 *    routes every position first and then walks the union.
 */
#include "mm.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <sys/time.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "cwick.h"
#include "ds.h"
#include "json.h"
#include "log.h"
#include "mem.h"
#include "metrics.h"
#include "util.h"

/* ------------------------------------------------------------------ */
/* scalar formats                                                      */
/* ------------------------------------------------------------------ */

static double mm_now_ms(void);
static double g_mm_now(void);

static inline uint16_t f_to_bf16(float f) {
    union { uint32_t u; float f; } v;
    v.f = f;
    uint32_t r = ((v.u >> 16) & 1) + 0x7FFF;   /* round to nearest even */
    return (uint16_t)((v.u + r) >> 16);
}

/* e2m1 nibble -> the high byte of the equivalent fp16. The low byte is zero
 * for every entry, so a shift by eight *is* the fp16 bit pattern -- which is
 * what reduces the NEON kernel below to a table lookup plus a widening shift.
 *
 *   index  bits   value      fp16
 *     0    0000    0.0       0x0000
 *     1    0001    0.5       0x3800
 *     2    0010    1.0       0x3C00
 *     3    0011    1.5       0x3E00
 *     4    0100    2.0       0x4000
 *     5    0101    3.0       0x4200
 *     6    0110    4.0       0x4400
 *     7    0111    6.0       0x4600
 *   8..15 are the same with the sign bit set. */
static const uint8_t MM_FP4_HI[16] = {
    0x00, 0x38, 0x3C, 0x3E, 0x40, 0x42, 0x44, 0x46,
    0x80, 0xB8, 0xBC, 0xBE, 0xC0, 0xC2, 0xC4, 0xC6
};

/* Unused on aarch64, where the table lookup below runs as NEON; kept as the
 * scalar reference the NEON path is checked against. */
__attribute__((unused))
static inline float fp4_to_f(uint8_t nib) {
    uint16_t h = (uint16_t)MM_FP4_HI[nib & 0x0F] << 8;
    uint32_t u = (uint32_t)h << 16;            /* fp16 -> fp32 exponent trick */
    uint32_t sign = u & 0x80000000u;
    uint32_t exp = (u >> 23) & 0xFF;
    uint32_t man = u & 0x7FFFFF;
    if (exp == 0) {                            /* fp16 subnormal -> fp32 normal */
        if (man == 0) { u = sign; } else {
            int shift = __builtin_clz(man) - 8;
            u = sign | ((uint32_t)(127 - 15 - shift) << 23) |
                ((man << (shift + 1)) & 0x7FFFFF);
        }
    } else {
        u = sign | ((exp - 127 + 15) << 23) | man;
    }
    union { uint32_t u; float f; } v;
    v.u = u;
    return v.f;
}

/* ------------------------------------------------------------------ */
/* GEMV over the two weight formats                                    */
/* ------------------------------------------------------------------ */

/* e4m3 block-scale table. ds_e4m3_to_f32() is a function call that first
 * asks whether the shared table is built; the expert GEMV asks it once per
 * 16-element block, i.e. 384 times per row and ~1.2 million times per matrix,
 * which measured as the difference between 17 GB/s and the memory ceiling.
 * Same values, one indexed load. */
static float g_mm_e4m3[256];
/* Where a decode step's milliseconds actually go, in two unambiguous
 * quantities: time inside the FP4 GEMV and time inside the layer's read-ahead.
 * The phase counters around them are useful for the shape of a step but they
 * nest, and nested scopes are how "experts 800 ms / io 887 ms / compute 39 ms"
 * ended up summing to more than the step. These are totals, printed once per
 * step under MM_PROF=1. */
static double g_mm_t_gemv, g_mm_t_io;
static int64_t g_mm_n_gemv;
static int64_t g_mm_n_gemm, g_mm_n_rows, g_mm_n_uni;
static int64_t g_mm_site[4];   /* 0 gate, 1 up, 2 down, 3 fallback */
static float g_mm_pow2[256];
static void mm_build_e4m3(void) {
    for (int i = 0; i < 256; i++) {
        g_mm_e4m3[i] = ds_e4m3_to_f32((uint8_t)i);
        g_mm_pow2[i] = ldexpf(1.0f, i - 127);
    }
}

typedef struct {
    const uint8_t *w, *s;
    const float *ev, *od;
    float inv_g;
    int64_t K, N;
    float *out;
} mm_fp4_arg;

#if !defined(__aarch64__)
static void mm_fp4_range(void *argp, int64_t n0, int64_t n1) {
    mm_fp4_arg *g = (mm_fp4_arg *)argp;
    const int64_t nb = g->K / 16;
    for (int64_t n = n0; n < n1; n++) {
        const uint8_t *row = g->w + n * (g->K / 2);
        const uint8_t *rs = g->s + n * nb;
        float acc = 0.0f;
        for (int64_t b = 0; b < nb; b++) {
            const float *e = g->ev + b * 8, *o = g->od + b * 8;
            float part = 0.0f;
            for (int j = 0; j < 8; j++) {
                uint8_t by = row[b * 8 + j];
                part += fp4_to_f(by & 0x0F) * e[j];
                part += fp4_to_f(by >> 4) * o[j];
            }
            acc += part * (g_mm_e4m3[rs[b]] * g->inv_g);
        }
        g->out[n] = acc;
    }
}
#endif

#if defined(__aarch64__)
static void mm_fp4_range_neon(void *argp, int64_t n0, int64_t n1) {
    mm_fp4_arg *g = (mm_fp4_arg *)argp;
    const int64_t nb = g->K / 16;
    const float *ev = g->ev, *od = g->od;
    const float inv_g = g->inv_g;
    const uint8x16_t tbl = vld1q_u8(MM_FP4_HI);
    const uint8x8_t mask = vdup_n_u8(0x0F);

    for (int64_t n = n0; n < n1; n++) {
        const uint8_t *row = g->w + n * (g->K / 2);
        const uint8_t *rs = g->s + n * nb;
        /* One vector accumulator for the whole row, not a scalar per block.
         * The block scale has to be applied before the 16 values are summed,
         * but summing them horizontally in scalar each time (the first version)
         * put an ~11-cycle reduction and a scalar multiply on a serial chain
         * every 16 elements: measured 16 GB/s against the 91 GB/s the bf16
         * GEMV next door reaches with the same pool. Folding the scaled block
         * into a vector accumulator leaves one reduction per row. */
        float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
        for (int64_t b = 0; b < nb; b++) {
            uint8x8_t v = vld1_u8(row + b * 8);
            /* low nibbles carry elements 2j, high nibbles elements 2j+1 */
            uint8x8_t lo_b = vqtbl1_u8(tbl, vand_u8(v, mask));
            uint8x8_t hi_b = vqtbl1_u8(tbl, vshr_n_u8(v, 4));
            uint16x8_t lo = vshll_n_u8(lo_b, 8);
            uint16x8_t hi = vshll_n_u8(hi_b, 8);

            float32x4_t s0 = vmulq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(lo))),
                vld1q_f32(ev + b * 8));
            float32x4_t s1 = vmulq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(lo))),
                vld1q_f32(ev + b * 8 + 4));
            s0 = vfmaq_f32(s0,
                vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(hi))),
                vld1q_f32(od + b * 8));
            s1 = vfmaq_f32(s1,
                vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(hi))),
                vld1q_f32(od + b * 8 + 4));
            /* Two accumulators, not one: the single one made every block wait
             * for the previous block's FMA, and the latency of that chain was
             * the whole difference between 78 and 138 GMAC/s (the grouped
             * kernel, which has the split by construction, measures the
             * latter). Block scales are per block, so both are scaled by the
             * same factor -- this changes the summation order, not the value
             * beyond f32 rounding. */
            acc0 = vfmaq_n_f32(acc0, s0, g_mm_e4m3[rs[b]] * inv_g);
            acc1 = vfmaq_n_f32(acc1, s1, g_mm_e4m3[rs[b]] * inv_g);
        }
        g->out[n] = vaddvq_f32(vaddq_f32(acc0, acc1));
    }
}
#endif

/* ------------------------------------------------------------------ */
/* FP8 dense GEMV / GEMM (e4m3 payload, e8m0 scale per 32 K, per row)   */
/* ------------------------------------------------------------------ */

/* 2^(s-127) for every e8m0 byte (built with the e4m3 table): a power of two
 * is exact in f32, which is why it can be multiplied in without changing the
 * value. */
/* Decode one e4m3 row into f32 with the scale applied.
 *
 * e4m3 -> f16 is a bit shuffle (sign stays, exponent +8 in a 5-bit field,
 * mantissa <<7), then f16 -> f32 is one instruction; a 256-entry f16 table
 * would be four vqtbl4 lookups plus selects per 8 bytes, so the arithmetic
 * wins. Subnormals (exponent field 0) come out as 2^-7*(1+m/8) instead of
 * 2^-6*(m/8): an error bounded by 2x on weights that are at most 3.5e-5 of
 * their block's maximum, i.e. below the noise of the block scale itself. */
static void mm_fp8_row_f32(float *dst, const uint8_t *row, const uint8_t *rs,
                           int64_t K) {
    const int64_t nb = K / 32;
    for (int64_t b = 0; b < nb; b++) {
        const uint8_t *src = row + b * 32;
        float *o = dst + b * 32;
        float sc = g_mm_pow2[rs[b]];
        for (int j = 0; j < 32; j += 8) {
            uint8x8_t v = vld1_u8(src + j);
            uint16x8_t h = vmovl_u8(v);
            uint16x8_t sgn = vshlq_n_u16(
                vandq_u16(h, vdupq_n_u16(0x0080)), 8);
            uint16x8_t mag = vshlq_n_u16(
                vandq_u16(h, vdupq_n_u16(0x007F)), 7);
            mag = vaddq_u16(mag, vdupq_n_u16(0x2000));   /* +8 exponent */
            h = vorrq_u16(mag, sgn);
            float32x4_t f0 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(h)));
            float32x4_t f1 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(h)));
            vst1q_f32(o + j, vmulq_n_f32(f0, sc));
            vst1q_f32(o + j + 4, vmulq_n_f32(f1, sc));
        }
    }
}

typedef struct {
    const uint8_t *w, *s;
    const float *x;      /* [n][K] */
    float *out;          /* [n][N] */
    int64_t K, N, n, nb; /* nb = K/32, the scale-block count */
} mm_fp8_arg;

/* Rows [c0,c1) of the weight, each decoded once and then dotted with all n
 * activations. That is what keeps prefill affordable: a 128-position chunk
 * would otherwise re-read (and re-decode) the whole matrix per position. */
#if defined(__aarch64__)
/* Decode 8 packed e4m3 values straight into two f32 vectors and fold them into
 * `acc`. This is the whole difference between this kernel losing and winning:
 * decoding a row into scratch and dotting it afterwards moves 2x the data and
 * measured 250 ms for the projections against bf16's 142 ms. */

/* Debugging aid, off unless MM_TRACE_HASH is set: a cheap order-sensitive
 * checksum of the residual stream, printed after each phase of each layer.
 * Two runs of the same prompt must print the same sequence; comparing them
 * localises non-determinism to one phase of one layer instead of to "the
 * model". Not a hash in any strong sense -- it only has to move when any lane
 * of any position moves. */
static int mm_trace_on(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("MM_TRACE_HASH") != NULL;
    return cached;
}
static void mm_trace(const char *what, int L, const float *v, size_t n) {
    if (!mm_trace_on()) return;
    uint64_t h = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < n; i++) {
        uint32_t b; memcpy(&b, &v[i], 4);
        h = (h ^ (b + 0x9E3779B9u + (h << 6) + (h >> 2))) * 0x2545F4914F6CDD1Dull;
    }
    fprintf(stderr, "trace L%02d %-12s %016llx\n", L, what,
            (unsigned long long)(h >> 32));
}

/* There is one reduction order that every *verify-sized* call has to use, and
 * it is the order the single-activation kernel below uses: within each
 * 32-element block the two 8-element groups go into two chains, the chains are
 * added, and the block's power-of-two scale multiplies that partial sum before
 * it joins the row accumulator.
 *
 * Why this matters: during block verification the model scores several
 * positions at once, and a candidate survives only if that score agrees with
 * what a step-at-a-time run would have produced -- otherwise speculation
 * changes the *answer*, not just how long it took. Reassociating 6144 products
 * moves the last bit or two, and over 60 layers that flips a near-tied argmax.
 * So every call with a small `n` runs that one kernel once per activation
 * instead of a batched one with a different tree. See MM_FP8_EXACT_N below for
 * where the line is drawn and what it costs. */
#define MM_FP8_FMA(acc, src, xv) do {                                       \
        uint8x8_t _v = vld1_u8(src);                                        \
        uint16x8_t _h = vmovl_u8(_v);                                       \
        uint16x8_t _sg = vshlq_n_u16(vandq_u16(_h, vdupq_n_u16(0x0080)), 8);\
        uint16x8_t _mg = vshlq_n_u16(vandq_u16(_h, vdupq_n_u16(0x007F)), 7);\
        _h = vorrq_u16(vaddq_u16(_mg, vdupq_n_u16(0x2000)), _sg);           \
        acc = vfmaq_f32(acc,                                                 \
                        vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(_h))),\
                        vld1q_f32(xv));                                      \
        acc = vfmaq_f32(acc,                                                 \
                       vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(_h))),\
                       vld1q_f32(xv + 4));                                   \
    } while (0)
#endif

/* Per-thread scratch for the tiled path: four decoded weight rows. Thread-local
 * rather than on the stack because K reaches 12288 (the dense MLP down
 * projection), i.e. 192 KB, and the pool's threads have the default 512 KB. */
static _Thread_local float *g_fp8_rows;
static _Thread_local size_t g_fp8_rows_cap;

static float *mm_fp8_rows(size_t need) {
    if (g_fp8_rows_cap < need) {
        free(g_fp8_rows);
        g_fp8_rows = (float *)cwick_xmalloc(sizeof(float) * need);
        g_fp8_rows_cap = need;
    }
    return g_fp8_rows;
}

/* One row against one activation, no tiling. Used for the rows and the
 * activations that do not fill a block of four. */
static void mm_fp8_rows_plain(const mm_fp8_arg *g, int64_t c0, int64_t c1,
                              int64_t t0, int64_t t1) {
    const int64_t K = g->K, N = g->N, nb = g->nb;
    float *rowf = mm_fp8_rows((size_t)K);
    for (int64_t c = c0; c < c1; c++) {
        mm_fp8_row_f32(rowf, g->w + (size_t)c * (size_t)K,
                       g->s + (size_t)c * nb, K);
        for (int64_t t = t0; t < t1; t++) {
            const float *xv = g->x + (size_t)t * (size_t)K;
            float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
            int64_t j = 0;
            for (; j + 8 <= K; j += 8) {
                a0 = vfmaq_f32(a0, vld1q_f32(rowf + j), vld1q_f32(xv + j));
                a1 = vfmaq_f32(a1, vld1q_f32(rowf + j + 4),
                               vld1q_f32(xv + j + 4));
            }
            float acc = vaddvq_f32(vaddq_f32(a0, a1));
            for (; j < K; j++) acc += rowf[j] * xv[j];
            g->out[(size_t)t * (size_t)N + c] = acc;
        }
    }
}

static void mm_fp8_range(void *argp, int64_t c0, int64_t c1) {
    mm_fp8_arg *g = (mm_fp8_arg *)argp;
    const int64_t K = g->K, n = g->n, N = g->N, nb = g->nb;

#if defined(__aarch64__)
    /* The exact path. A verify-sized call -- one decode step, or the handful
     * of speculative positions being scored together -- runs this kernel once
     * per activation rather than the row-tiled one below, even though the
     * tiled kernel is 2.5x faster at n=8. The reason is the reduction order,
     * not the speed: a candidate is accepted if the model's own argmax agrees
     * with it, and for that to mean anything the batched pass has to produce
     * the same bits as the step it replaces. MM_FP8_EXACT_N is the largest `n`
     * that pays for that guarantee; anything bigger is a prefill chunk, whose
     * outputs are never compared against a reference.
     *
     * It is also the faster path below four activations: the tiling decodes
     * four rows into scratch and re-reads each of them per activation, which
     * measured 2.4 ms at n=2 against 1.4 ms for doing this twice. */
    static int64_t exact_n = -1;
    if (exact_n < 0) {
        const char *e = getenv("MM_FP8_EXACT_N");
        exact_n = (e && *e) ? atoll(e) : 8;
        if (exact_n < 1) exact_n = 1;
    }
    if (n <= exact_n) {
        for (int64_t t = 0; t < n; t++) {
            const float *xv = g->x + (size_t)t * (size_t)K;
            float *ov = g->out + (size_t)t * (size_t)N;
            for (int64_t c = c0; c < c1; c++) {
                const uint8_t *row = g->w + (size_t)c * (size_t)K;
                const uint8_t *rs = g->s + (size_t)c * nb;
                float32x4_t accv = vdupq_n_f32(0.0f);
                for (int64_t b = 0; b < nb; b++) {
                    const uint8_t *src = row + b * 32;
                    const float *xb = xv + b * 32;
                    float32x4_t b0 = vdupq_n_f32(0.0f);
                    float32x4_t b1 = vdupq_n_f32(0.0f);
                    MM_FP8_FMA(b0, src + 0, xb + 0);
                    MM_FP8_FMA(b1, src + 8, xb + 8);
                    MM_FP8_FMA(b0, src + 16, xb + 16);
                    MM_FP8_FMA(b1, src + 24, xb + 24);
                    accv = vfmaq_n_f32(accv, vaddq_f32(b0, b1),
                                       g_mm_pow2[rs[b]]);
                }
                ov[c] = vaddvq_f32(accv);
            }
        }
        return;
    }
#endif

    /* Prefill. Four weight rows by four activations, with the weights decoded
     * once per row block.
     *
     * The untiled loop read the whole activation matrix once per weight row:
     * for an 8192x6144 matrix and a 302-token chunk that is 60 GB of L2
     * traffic for 15 G MAC, which is why it stalled at 96 GMAC/s while the
     * kernels around it run at 130+. Here each activation vector is loaded
     * once and used by four rows, and each weight vector once for four
     * activations, so both traffic and issue count drop by four. Sixteen
     * accumulators plus the eight live loads fit in the 32 NEON registers. */
    int64_t c = c0;
    for (; c + 4 <= c1; c += 4) {
        float *rows = mm_fp8_rows((size_t)(4 * K));
        float *r0 = rows, *r1 = rows + K, *r2 = rows + 2 * K, *r3 = rows + 3 * K;
        mm_fp8_row_f32(r0, g->w + (size_t)c * (size_t)K,
                       g->s + (size_t)c * nb, K);
        mm_fp8_row_f32(r1, g->w + (size_t)(c + 1) * (size_t)K,
                       g->s + (size_t)(c + 1) * nb, K);
        mm_fp8_row_f32(r2, g->w + (size_t)(c + 2) * (size_t)K,
                       g->s + (size_t)(c + 2) * nb, K);
        mm_fp8_row_f32(r3, g->w + (size_t)(c + 3) * (size_t)K,
                       g->s + (size_t)(c + 3) * nb, K);

        int64_t t = 0;
        for (; t + 4 <= n; t += 4) {
            const float *x0 = g->x + (size_t)(t + 0) * (size_t)K;
            const float *x1 = g->x + (size_t)(t + 1) * (size_t)K;
            const float *x2 = g->x + (size_t)(t + 2) * (size_t)K;
            const float *x3 = g->x + (size_t)(t + 3) * (size_t)K;
            float32x4_t a00 = vdupq_n_f32(0.0f), a01 = a00, a02 = a00, a03 = a00;
            float32x4_t a10 = a00, a11 = a00, a12 = a00, a13 = a00;
            float32x4_t a20 = a00, a21 = a00, a22 = a00, a23 = a00;
            float32x4_t a30 = a00, a31 = a00, a32 = a00, a33 = a00;
            int64_t j = 0;
            for (; j + 4 <= K; j += 4) {
                float32x4_t w0 = vld1q_f32(r0 + j);
                float32x4_t w1 = vld1q_f32(r1 + j);
                float32x4_t w2 = vld1q_f32(r2 + j);
                float32x4_t w3 = vld1q_f32(r3 + j);
                float32x4_t b0 = vld1q_f32(x0 + j);
                float32x4_t b1 = vld1q_f32(x1 + j);
                float32x4_t b2 = vld1q_f32(x2 + j);
                float32x4_t b3 = vld1q_f32(x3 + j);
                a00 = vfmaq_f32(a00, w0, b0);
                a01 = vfmaq_f32(a01, w0, b1);
                a02 = vfmaq_f32(a02, w0, b2);
                a03 = vfmaq_f32(a03, w0, b3);
                a10 = vfmaq_f32(a10, w1, b0);
                a11 = vfmaq_f32(a11, w1, b1);
                a12 = vfmaq_f32(a12, w1, b2);
                a13 = vfmaq_f32(a13, w1, b3);
                a20 = vfmaq_f32(a20, w2, b0);
                a21 = vfmaq_f32(a21, w2, b1);
                a22 = vfmaq_f32(a22, w2, b2);
                a23 = vfmaq_f32(a23, w2, b3);
                a30 = vfmaq_f32(a30, w3, b0);
                a31 = vfmaq_f32(a31, w3, b1);
                a32 = vfmaq_f32(a32, w3, b2);
                a33 = vfmaq_f32(a33, w3, b3);
            }
            float k0 = 0, k1 = 0, k2 = 0, k3 = 0;   /* K % 4 leftovers */
            for (; j < K; j++) {
                k0 += r0[j] * x0[j];
                k1 += r1[j] * x0[j];
                k2 += r2[j] * x0[j];
                k3 += r3[j] * x0[j];
            }
            g->out[(size_t)(t + 0) * (size_t)N + c + 0] = vaddvq_f32(a00)+ k0;
            g->out[(size_t)(t + 0) * (size_t)N + c + 1] = vaddvq_f32(a10)+ k1;
            g->out[(size_t)(t + 0) * (size_t)N + c + 2] = vaddvq_f32(a20)+ k2;
            g->out[(size_t)(t + 0) * (size_t)N + c + 3] = vaddvq_f32(a30)+ k3;
            g->out[(size_t)(t + 1) * (size_t)N + c + 0] = vaddvq_f32(a01);
            g->out[(size_t)(t + 1) * (size_t)N + c + 1] = vaddvq_f32(a11);
            g->out[(size_t)(t + 1) * (size_t)N + c + 2] = vaddvq_f32(a21);
            g->out[(size_t)(t + 1) * (size_t)N + c + 3] = vaddvq_f32(a31);
            g->out[(size_t)(t + 2) * (size_t)N + c + 0] = vaddvq_f32(a02);
            g->out[(size_t)(t + 2) * (size_t)N + c + 1] = vaddvq_f32(a12);
            g->out[(size_t)(t + 2) * (size_t)N + c + 2] = vaddvq_f32(a22);
            g->out[(size_t)(t + 2) * (size_t)N + c + 3] = vaddvq_f32(a32);
            g->out[(size_t)(t + 3) * (size_t)N + c + 0] = vaddvq_f32(a03);
            g->out[(size_t)(t + 3) * (size_t)N + c + 1] = vaddvq_f32(a13);
            g->out[(size_t)(t + 3) * (size_t)N + c + 2] = vaddvq_f32(a23);
            g->out[(size_t)(t + 3) * (size_t)N + c + 3] = vaddvq_f32(a33);
        }
        if (t < n) {
            /* Up to three activations left; the decoded rows are already here,
             * so just run them one at a time, in the canonical order. */
            for (int64_t tt = t; tt < n; tt++) {
                const float *xv = g->x + (size_t)tt * (size_t)K;
                const float *rr[4] = { r0, r1, r2, r3 };
                for (int r = 0; r < 4; r++) {
                    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0;
                    int64_t j = 0;
                    for (; j + 8 <= K; j += 8) {
                        s0 = vfmaq_f32(s0, vld1q_f32(rr[r] + j),
                                       vld1q_f32(xv + j));
                        s1 = vfmaq_f32(s1, vld1q_f32(rr[r] + j + 4),
                                       vld1q_f32(xv + j + 4));
                    }
                    float acc = vaddvq_f32(vaddq_f32(s0, s1));
                    for (; j < K; j++) acc += rr[r][j] * xv[j];
                    g->out[(size_t)tt * (size_t)N + c + r] = acc;
                }
            }
        }
    }
    if (c < c1) mm_fp8_rows_plain(g, c, c1, 0, n);
}

/* Where the FP8 dense time goes, by shape. `proj` in the profile covers five
 * matrices of very different width, and their sum is several times what the
 * same kernels cost standalone -- this table says which one is responsible. */
typedef struct { int64_t N, K, n; int64_t calls; double ms; } mm_fp8_slot;
static mm_fp8_slot g_fp8_stats[16];
static int g_fp8_stats_n;

static void mm_fp8_stat(int64_t N, int64_t K, int64_t n, double ms) {
    if (getenv("MM_PROF") == NULL) return;
    for (int i = 0; i < g_fp8_stats_n; i++)
        if (g_fp8_stats[i].N == N && g_fp8_stats[i].K == K &&
            g_fp8_stats[i].n == n) {
            g_fp8_stats[i].calls++;
            g_fp8_stats[i].ms += ms;
            return;
        }
    if (g_fp8_stats_n >= 16) return;
    mm_fp8_slot *e = &g_fp8_stats[g_fp8_stats_n++];
    e->N = N; e->K = K; e->n = n; e->calls = 1; e->ms = ms;
}

static void mm_fp8_stats_print(void) {
    for (int i = 0; i < g_fp8_stats_n; i++) {
        mm_fp8_slot *e = &g_fp8_stats[i];
        double mac = (double)e->N * (double)e->K * (double)e->n / 1e9;
        fprintf(stderr, "  fp8 [%lldx%lld] n=%lld: %lld calls, %.0f ms, "
                "%.0f GMAC/s\n", (long long)e->N, (long long)e->K,
                (long long)e->n, (long long)e->calls, e->ms,
                mac * (double)e->calls / (e->ms * 1e-3));
    }
    g_fp8_stats_n = 0;
}

/* out[t][c] = sum_j W[c][j] * x[t][j], W = e4m3 [N,K] + e8m0 [N,K/32]. */
static void mm_fp8_gemm(float *out, int64_t n, const uint8_t *w,
                        const uint8_t *s, const float *x, int64_t N, int64_t K) {
    if (n <= 0 || N <= 0 || K <= 0 || K > 16384 || !w || !s) return;
    mm_fp8_arg a = { w, s, x, out, K, N, n, K / 32 };
    double t0 = g_mm_now();
    cwick_pool_for(N, &a, mm_fp8_range);
    mm_fp8_stat(N, K, n, g_mm_now() - t0);
}



/* One NVFP4 matrix against one activation. `ev`/`od` are scratch of K/2
 * floats each: x de-interleaved into its even and odd lanes, because a packed
 * byte holds element 2j in its low nibble and 2j+1 in its high one. */
static void mm_fp4_gemv(float *out, const uint8_t *w, const uint8_t *s,
                        float inv_g, const float *x, int64_t N, int64_t K,
                        float *ev, float *od) {
    double _t0 = g_mm_now();
    for (int64_t j = 0; j < K / 2; j++) {
        ev[j] = x[2 * j];
        od[j] = x[2 * j + 1];
    }
    mm_fp4_arg a = { w, s, ev, od, inv_g, K, N, out };
#if defined(__aarch64__)
    cwick_pool_for(N, &a, mm_fp4_range_neon);
#else
    cwick_pool_for(N, &a, mm_fp4_range);
#endif
    g_mm_t_gemv += g_mm_now() - _t0;
    g_mm_n_gemv++;
}

/* One expert matrix against n activations ([n][K], contiguous).
 *
 * Prefill routes a whole chunk before walking the union, so a single expert is
 * usually asked for several positions at once -- and the per-row GEMV read its
 * 10.6 MB matrix once per position, i.e. 4x for a 128-position chunk and 16x
 * for a 512 one. Here the weight block is decoded once and dotted with every
 * activation, so the matrix is streamed once for the group.
 */
typedef struct {
    const uint8_t *w, *s;
    const float *x;      /* [n][K] */
    float *out;          /* [n][N] */
    float inv_g;
    int64_t K, N, n;
} mm_fp4_gemm_arg;

/* Decode the 8 packed bytes of block b into four f32 vectors: `l0/l1` hold the
 * even elements (2j) and `h0/h1` the odd ones (2j+1). */
#define MM_FP4_DECODE(row, b)                                                \
    uint8x8_t _v = vld1_u8((row) + (b) * 8);                                 \
    uint8x8_t _lb = vqtbl1_u8(tbl, vand_u8(_v, mask));                       \
    uint8x8_t _hb = vqtbl1_u8(tbl, vshr_n_u8(_v, 4));                        \
    uint16x8_t _lo = vshll_n_u8(_lb, 8);                                     \
    uint16x8_t _hi = vshll_n_u8(_hb, 8);                                     \
    float32x4_t l0 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(_lo)));  \
    float32x4_t l1 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(_lo))); \
    float32x4_t h0 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(_hi)));  \
    float32x4_t h1 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(_hi)))

/* Fold one block into activation T's two accumulators, scaled by the block's
 * e4m3 scale. Two accumulators, not a horizontal sum: the reduction happens
 * once per row at the end, exactly as in the single-row kernel.
 *
 * The low-lane and high-lane products are combined with a fused multiply-add,
 * not a multiply and an add. This is not a micro-optimisation: it is what makes
 * the grouped kernel return the *same bits* as the single-row one. The per-row
 * kernel computes `fma(l*x, h, o)` (one rounding for the pair); doing
 * `mul + mul + add` rounds each product separately, which perturbs the last
 * bit or two of every partial sum. Over 60 layers that is enough to flip a
 * near-tied argmax, and it is why block verification used to disagree with a
 * step-at-a-time run. Everything else here already matches the single-row
 * kernel -- same lane mapping, same two accumulators, same block order, same
 * final reduction -- so this one instruction is the whole difference. */
#define MM_FP4_ACC(A0, A1, T)                                                \
    do {                                                                     \
        const float *_ev = g->x + (size_t)(T) * (size_t)K + (size_t)bb * 8;  \
        const float *_od = _ev + half;                                       \
        const float32x4_t _e0 = vld1q_f32(_ev), _e1 = vld1q_f32(_ev + 4);    \
        const float32x4_t _o0 = vld1q_f32(_od), _o1 = vld1q_f32(_od + 4);    \
        A0 = vfmaq_n_f32(A0, vfmaq_f32(vmulq_f32(l0, _e0), h0, _o0), sc);     \
        A1 = vfmaq_n_f32(A1, vfmaq_f32(vmulq_f32(l1, _e1), h1, _o1), sc);     \
    } while (0)

static void mm_fp4_gemm_range(void *argp, int64_t n0, int64_t n1) {
    mm_fp4_gemm_arg *g = (mm_fp4_gemm_arg *)argp;
    const int64_t nb = g->K / 16, K = g->K, N = g->N, n = g->n;
    const int64_t half = K / 2;
    const float inv_g = g->inv_g;
    const uint8x16_t tbl = vld1q_u8(MM_FP4_HI);
    const uint8x8_t mask = vdup_n_u8(0x0F);

    for (int64_t c = n0; c < n1; c++) {
        const uint8_t *row = g->w + (size_t)c * (size_t)(K / 2);
        const uint8_t *rs = g->s + (size_t)c * (size_t)nb;
        int64_t t = 0;

        /* Four activations at a time: eight vector accumulators plus the four
         * decoded weight vectors fit in the 32 NEON registers, so the inner
         * loop does one decode for four dot products and touches memory only
         * for the activations. The first version of this kernel kept its
         * accumulators in a stack array and did a horizontal reduction per
         * block per activation -- which is why batching measured *slower*
         * than four single-row GEMVs (212 s vs 199 s on a 302-token prompt).
         * Eight activations was tried too and is worse (125 vs 143 GMAC/s at
         * n=8): sixteen accumulators plus the decode spills. */
        for (; t + 4 <= n; t += 4) {
            float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
            float32x4_t b0 = a0, b1 = a1, c0 = a0, c1 = a1;
            float32x4_t d0 = a0, d1 = a1;
            for (int64_t bb = 0; bb < nb; bb++) {
                MM_FP4_DECODE(row, bb);
                const float sc = g_mm_e4m3[rs[bb]] * inv_g;
                MM_FP4_ACC(a0, a1, t + 0);
                MM_FP4_ACC(b0, b1, t + 1);
                MM_FP4_ACC(c0, c1, t + 2);
                MM_FP4_ACC(d0, d1, t + 3);
            }
            g->out[(size_t)(t + 0) * (size_t)N + c] = vaddvq_f32(vaddq_f32(a0, a1));
            g->out[(size_t)(t + 1) * (size_t)N + c] = vaddvq_f32(vaddq_f32(b0, b1));
            g->out[(size_t)(t + 2) * (size_t)N + c] = vaddvq_f32(vaddq_f32(c0, c1));
            g->out[(size_t)(t + 3) * (size_t)N + c] = vaddvq_f32(vaddq_f32(d0, d1));
        }
        /* Pairs: an expert asked for two positions is the common case in a
         * short chunk, and the one-at-a-time tail measured 0.84x the single-row
         * GEMV, i.e. worse than not batching at all. */
        if (t + 2 <= n) {
            float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
            float32x4_t b0 = a0, b1 = a1;
            for (int64_t bb = 0; bb < nb; bb++) {
                MM_FP4_DECODE(row, bb);
                const float sc = g_mm_e4m3[rs[bb]] * inv_g;
                MM_FP4_ACC(a0, a1, t + 0);
                MM_FP4_ACC(b0, b1, t + 1);
            }
            g->out[(size_t)(t + 0) * (size_t)N + c] = vaddvq_f32(vaddq_f32(a0, a1));
            g->out[(size_t)(t + 1) * (size_t)N + c] = vaddvq_f32(vaddq_f32(b0, b1));
            t += 2;
        }
        /* Tail: at most three activations, one at a time. */
        for (; t < n; t++) {
            float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
            for (int64_t bb = 0; bb < nb; bb++) {
                MM_FP4_DECODE(row, bb);
                const float sc = g_mm_e4m3[rs[bb]] * inv_g;
                MM_FP4_ACC(a0, a1, t);
            }
            g->out[(size_t)t * (size_t)N + c] = vaddvq_f32(vaddq_f32(a0, a1));
        }
    }
}

#undef MM_FP4_ACC
#undef MM_FP4_DECODE

/* x[2j] -> ev[j], x[2j+1] -> od[j]: the layout mm_fp4_gemm expects. */
static void mm_deinterleave(float *ev, const float *x, int64_t K) {
    float *od = ev + K / 2;
    for (int64_t j = 0; j < K / 2; j++) {
        ev[j] = x[2 * j];
        od[j] = x[2 * j + 1];
    }
}

static void mm_fp4_gemm(float *out, int64_t n, const uint8_t *w, const uint8_t *s,
                        float inv_g, const float *x, int64_t N, int64_t K) {
    if (n <= 0 || N <= 0 || K <= 0 || n > MM_CHUNK * 4) return;
    mm_fp4_gemm_arg a = { w, s, x, out, inv_g, K, N, n };
    g_mm_n_gemm++;
    g_mm_n_rows += n;
    cwick_pool_for(N, &a, mm_fp4_gemm_range);
}

/* ------------------------------------------------------------------ */
/* norms, rope, activation                                             */
/* ------------------------------------------------------------------ */

/* The checkpoint sets `use_gemma_norm`, so the multiplier is (1 + w) rather
 * than w: a norm weight near zero then means "pass the stream through". */
static void mm_rmsnorm_gemma(float *y, const float *x, const uint16_t *w,
                             int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (int i = 0; i < n; i++)
        y[i] = x[i] * scale * (1.0f + cwick_bf16_to_f32(w[i]));
}

/* Per-head RMSNorm: one [head_dim] weight vector shared by every head. */
static void mm_head_norm(float *x, const uint16_t *w, int n_heads,
                         int head_dim, float eps) {
    for (int h = 0; h < n_heads; h++)
        mm_rmsnorm_gemma(x + (size_t)h * head_dim, x + (size_t)h * head_dim,
                         w, head_dim, eps);
}

/* RoPE over the first `rotary_dim` lanes, half-split: the partner of lane i
 * is lane i + rotary_dim/2. The remaining lanes pass through untouched. */
static void mm_rope(float *x, int rotary_dim, const float *cos,
                    const float *sin) {
    const int half = rotary_dim / 2;
    for (int i = 0; i < half; i++) {
        float a = x[i], b = x[i + half];
        x[i] = a * cos[i] - b * sin[i];
        x[i + half] = b * cos[i] + a * sin[i];
    }
}

static inline float mm_sigmoid(float v) { return 1.0f / (1.0f + expf(-v)); }

/* SwiGLU-OAI: clamp(gate * sigmoid(alpha*gate), +-limit) * (up + 1). The cap
 * stops one large gate from dominating the sum; the +1 is part of the
 * activation, not a bias. */
static inline float mm_swigluoai(float gate, float up, float alpha,
                                 float limit) {
    float g = gate * mm_sigmoid(alpha * gate);
    if (g > limit) g = limit;
    else if (g < -limit) g = -limit;
    return g * (up + 1.0f);
}

/* Keep the k largest of n. Indices come back ascending; `heap`/`heap_idx` are
 * scratch of k entries. Lower index wins a tie, which keeps the block choice
 * stable when a whole run of blocks scores the same. */
static int mm_topk(const float *sc, int n, int k, int *out, float *heap,
                   int *heap_idx) {
    if (k <= 0) return 0;
    int m = 0;
    for (int i = 0; i < n; i++) {
        if (m < k) {
            heap[m] = sc[i]; heap_idx[m] = i; m++;
            int c = m - 1;
            while (c > 0 && heap[c] < heap[(c - 1) / 2]) {
                float t = heap[c]; heap[c] = heap[(c - 1) / 2];
                heap[(c - 1) / 2] = t;
                int ti = heap_idx[c]; heap_idx[c] = heap_idx[(c - 1) / 2];
                heap_idx[(c - 1) / 2] = ti;
                c = (c - 1) / 2;
            }
        } else if (sc[i] > heap[0]) {
            heap[0] = sc[i]; heap_idx[0] = i;
            int p = 0;
            for (;;) {
                int l = 2 * p + 1, r = l + 1, s = p;
                if (l < k && heap[l] < heap[s]) s = l;
                if (r < k && heap[r] < heap[s]) s = r;
                if (s == p) break;
                float t = heap[p]; heap[p] = heap[s]; heap[s] = t;
                int ti = heap_idx[p]; heap_idx[p] = heap_idx[s];
                heap_idx[s] = ti;
                p = s;
            }
        }
    }
    for (int i = 0; i < m; i++) out[i] = heap_idx[i];
    for (int i = 1; i < m; i++) {
        int v = out[i], j = i - 1;
        while (j >= 0 && out[j] > v) { out[j + 1] = out[j]; j--; }
        out[j + 1] = v;
    }
    return m;
}

/* ------------------------------------------------------------------ */
/* config probing                                                      */
/* ------------------------------------------------------------------ */

static double jnum(json_value *v, double dflt) {
    return v ? cwick_json_num(v, dflt) : dflt;
}

int mm_model_probe(const char *root) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/config.json", root);
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    char buf[16384];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = 0;
    fclose(f);
    return strstr(buf, "minimax_m3_vl") != NULL ||
           strstr(buf, "MiniMaxM3Sparse") != NULL;
}

/* ------------------------------------------------------------------ */
/* weight binding                                                      */
/* ------------------------------------------------------------------ */

static const cwick_st_tensor *mm_t(mm_model *m, const char *fmt, ...) {
    char name[320];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof(name), fmt, ap);
    va_end(ap);
    for (size_t i = 0; i < m->saf.tensor_count; i++)
        if (strcmp(m->saf.tensors[i].name, name) == 0) return &m->saf.tensors[i];
    return NULL;
}

static int mm_bind(mm_qw *q, const char *tag,
                   const cwick_st_tensor *w, const cwick_st_tensor *s) {
    (void)tag;
    if (!w) { CWICK_INFO("mm", "missing %s", tag); return -1; }
    q->w = (const uint8_t *)w->data;
    q->rows = w->rows;
    q->cols = w->cols;
    q->s = s ? (const uint8_t *)s->data : NULL;
    q->w16 = NULL;
    if (!s || s->rows != w->rows || s->cols != w->cols / 32) {
        CWICK_INFO("mm", "%s: scale %lldx%lld does not match %lldx%lld",
                   w->name, s ? (long long)s->rows : -1,
                   s ? (long long)s->cols : -1,
                   (long long)w->rows, (long long)w->cols);
        return -1;
    }
    return 0;
}

/* Every FP8 matrix in the model, walked once to size the widened copy and
 * once to fill it. Collected in one place so the two passes cannot drift. */
typedef struct { mm_qw *q; } mm_qw_ref;

/* Widen one FP8 matrix to BF16 with its e8m0 scale folded in. The product of
 * an e4m3 value (four significant bits) and a power of two is exact in bf16
 * (eight significant bits), so this loses nothing -- and it turns the inner
 * loop's table lookup into a shift. */
/* ------------------------------------------------------------------ */
/* open                                                                */
/* ------------------------------------------------------------------ */

/* Budget for the expert pin cache.
 *
 * Same shape as the DeepSeek backend's, because it was measured the hard way
 * there: 80% of host RAM is the plan, and every GiB the process spends on
 * something else has to come off the cache or the plan is a lie. The three
 * terms that are not the cache here are
 *
 *   dense   the bf16 copy of every FP8 matrix, 19.7 GiB at load time -- the
 *           only copy, so it cannot be reclaimed
 *   KV      the k/v and indexer caches, sized by --context: 4.2 GiB at 32K
 *           but 129 GiB at 1M, which is why this backend's default context is
 *           small and why the number is computed rather than assumed
 *   rest    Metal arenas and driver accounting (none yet on this path, but
 *           the term is kept so the plan does not change shape when a GPU
 *           kernel lands), scratch, the allocator, and the page-cache share
 *           of embed_tokens/lm_head, which the kernel may or may not keep
 *
 * Getting this wrong does not fail loudly. The first version of the DeepSeek
 * budget under-counted the non-cache part by 24 GiB, the overshoot went
 * straight to the compressor, and the same model ran three times slower at an
 * unchanged cache hit rate.
 */
static size_t mm_expert_cache_budget(const mm_model *m, size_t dense_bytes,
                                     uint64_t *target_out) {
    const uint64_t GiB = 1024ull * 1024ull * 1024ull;
    const mm_config *c = &m->cfg;
    uint64_t host = cwick_host_memory_bytes();
    uint64_t target = host ? host / 10 * 8 : 0;
    if (target_out) *target_out = target;

    const char *e = getenv("CWICK_EXPERT_CACHE_GIB");
    if (e && *e) {
        double gib = atof(e);
        if (gib <= 0.0) return 0;
        return (size_t)(gib * 1024.0 * 1024.0 * 1024.0);
    }
    if (host == 0) return (size_t)32 * GiB;

    uint64_t kv = mm_kv_bytes(m, c->max_seq_len);
    /* Unlike the DeepSeek path there is no Metal arena here: every GEMV runs
     * on the CPU pool and the weights it reads are the pin cache itself. The
     * reserve for GPU-visible memory was costing 16 GiB of expert cache --
     * 27% more capacity, which is measured directly in the read-ahead below
     * (2.5 GiB/step at 20 GiB of cache). Keep a fixed 6 GiB for the scratch
     * buffers and allocator slack -- measured footprint lands ~93 GiB against
     * the 102.4 GiB ceiling, which is where the steering valve wants it. */
    uint64_t gpu = 6 * GiB;
    uint64_t reserve = 4 * GiB + kv + gpu;
    uint64_t fixed = (uint64_t)dense_bytes + reserve;
    if (target < fixed + 4 * GiB) target = fixed + 4 * GiB;

    /* The pin cache does not want all of RAM, and this is the one place where
     * the plan above is deliberately not followed.
     *
     * Past a certain footprint the compressor starts taking pages, and those
     * pages are not free to get back: a step that touches one pays a
     * decompression on its critical path, and the phases that never read the
     * cache at all (projections, attention) slow down with it. Measured on
     * MiniMax-M3.1-Flash, 23 decode steps, same prompt:
     *
     *     cache 14 GiB = 1.39 tok/s      30 GiB = 1.58
     *     cache 18 GiB = 1.58            44 GiB = 1.16
     *     cache 22 GiB = 1.62            63 GiB = 1.08   (the 80% plan)
     *
     * So the plateau is 18-30 GiB and a bigger cache is a loss even though it
     * does stream fewer bytes: at 63 GiB the hit rate is 78% and the step is
     * still 40% longer. `host / 6` lands at 21.3 GiB on a 128 GiB machine,
     * which is the middle of the plateau, and it scales down on smaller ones
     * instead of pretending they have room. */
    uint64_t cap = host / 6;
    if (target > fixed + cap) target = fixed + cap;
    if (target < fixed + 4 * GiB) target = fixed + 4 * GiB;
    return (size_t)(target - fixed);
}

int mm_model_open(mm_model *m, const char *root, int max_seq_len,
                  cwick_error *err) {
    memset(m, 0, sizeof(*m));
    mm_build_e4m3();

    char path[4096];
    snprintf(path, sizeof(path), "%s/config.json", root);
    char *text = NULL;
    size_t tlen = 0;
    if (cwick_read_file(path, &text, &tlen) != 0) {
        cwick_error_set(err, CWICK_ERR_IO, "mm: cannot read %s", path);
        return -1;
    }
    json_value *cf = cwick_json_parse(text, tlen, err);
    free(text);
    if (!cf) return -1;

    mm_config *c = &m->cfg;
    json_value *tc = cwick_json_get(cf, "text_config");
    if (!tc || tc->kind != JSON_OBJECT) tc = cf;

    c->dim = (int)jnum(cwick_json_get(tc, "hidden_size"), 6144);
    c->n_layers = (int)jnum(cwick_json_get(tc, "num_hidden_layers"), 60);
    c->n_heads = (int)jnum(cwick_json_get(tc, "num_attention_heads"), 64);
    c->n_kv_heads = (int)jnum(cwick_json_get(tc, "num_key_value_heads"), 4);
    c->head_dim = (int)jnum(cwick_json_get(tc, "head_dim"), 128);
    c->vocab_size = (int)jnum(cwick_json_get(tc, "vocab_size"), 200064);
    c->inter_dim = (int)jnum(cwick_json_get(tc, "intermediate_size"), 3072);
    c->dense_inter_dim =
        (int)jnum(cwick_json_get(tc, "dense_intermediate_size"), 12288);
    c->n_routed = (int)jnum(cwick_json_get(tc, "num_local_experts"), 128);
    c->n_activated = (int)jnum(cwick_json_get(tc, "num_experts_per_tok"), 4);
    c->rms_norm_eps = (float)jnum(cwick_json_get(tc, "rms_norm_eps"), 1e-6);
    c->rope_theta = jnum(cwick_json_get(tc, "rope_theta"), 1e7);
    c->swiglu_alpha = (float)jnum(cwick_json_get(tc, "swiglu_alpha"), 1.702);
    c->swiglu_limit = (float)jnum(cwick_json_get(tc, "swiglu_limit"), 7.0);
    c->route_scale = (float)jnum(cwick_json_get(tc, "routed_scaling_factor"), 2.0);
    c->norm_topk_prob = 1;

    double rdim = jnum(cwick_json_get(tc, "rotary_dim"), 0.0);
    double pfac = jnum(cwick_json_get(tc, "partial_rotary_factor"), 0.5);
    c->rotary_dim = rdim > 0 ? (int)rdim : (int)(c->head_dim * pfac);
    if (c->rotary_dim > c->head_dim) c->rotary_dim = c->head_dim;
    if ((c->rotary_dim & 1) || c->rotary_dim <= 0)
        c->rotary_dim = c->head_dim;

    c->first_k_dense = 0;
    json_value *fr = cwick_json_get(tc, "moe_layer_freq");
    if (fr && fr->kind == JSON_ARRAY)
        for (size_t i = 0; i < fr->count && (int)i < c->n_layers; i++)
            if (cwick_json_num(cwick_json_array_at(fr, i), 1.0) == 0.0)
                c->first_k_dense = (int)i + 1;

    c->index_dim = 128; c->index_n_heads = 4;
    c->topk_blocks = 16; c->block_size = 128; c->local_blocks = 1;
    json_value *sa = cwick_json_get(tc, "sparse_attention_config");
    if (sa && sa->kind == JSON_OBJECT) {
        c->index_dim = (int)jnum(cwick_json_get(sa, "sparse_index_dim"), 128);
        c->index_n_heads =
            (int)jnum(cwick_json_get(sa, "sparse_num_index_heads"), 4);
        c->topk_blocks = (int)jnum(cwick_json_get(sa, "sparse_topk_blocks"), 16);
        c->block_size = (int)jnum(cwick_json_get(sa, "sparse_block_size"), 128);
        c->local_blocks = (int)jnum(cwick_json_get(sa, "sparse_local_block"), 1);
    }
    if (c->block_size <= 0) c->block_size = 128;

    c->max_seq_len = max_seq_len > 0 ? max_seq_len : 8192;

    CWICK_INFO("init", "mm config dim=%d layers=%d heads=%d kv=%d hd=%d "
               "rope=%d experts=%d top-%d dense=%d index=%dx%d top%dblk/%d",
               c->dim, c->n_layers, c->n_heads, c->n_kv_heads, c->head_dim,
               c->rotary_dim, c->n_routed, c->n_activated, c->first_k_dense,
               c->index_n_heads, c->index_dim, c->topk_blocks, c->block_size);

    cwick_json_free(cf);

    cwick_error_clear(err);
    if (cwick_safetensors_open(root, &m->saf, err) != CWICK_OK) return -1;

    m->embed = (const uint16_t *)mm_t(m,
        "language_model.model.embed_tokens.weight")->data;
    m->head = (const uint16_t *)mm_t(m,
        "language_model.lm_head.weight")->data;
    m->final_norm = (const uint16_t *)mm_t(m,
        "language_model.model.norm.weight")->data;
    if (!m->embed || !m->head || !m->final_norm) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "mm: missing embed/head/norm");
        return -1;
    }

    for (int L = 0; L < c->n_layers; L++) {
        mm_layer *y = &m->layers[L];
        y->is_dense = L < c->first_k_dense;
#define NORM(field, suffix) do {                                            \
            const cwick_st_tensor *_t = mm_t(                              \
                m, "language_model.model.layers.%d." suffix, L);           \
            y->field = _t ? (const uint16_t *)_t->data : NULL;             \
        } while (0)
        NORM(in_norm, "input_layernorm.weight");
        NORM(post_norm, "post_attention_layernorm.weight");
        NORM(q_norm, "self_attn.q_norm.weight");
        NORM(k_norm, "self_attn.k_norm.weight");
        NORM(idx_q_norm, "self_attn.index_q_norm.weight");
        NORM(idx_k_norm, "self_attn.index_k_norm.weight");
#undef NORM
#define QW(field, nm)                                                       \
        do {                                                                \
            const cwick_st_tensor *_w = mm_t(                              \
                m, "language_model.model.layers.%d." nm ".weight", L);     \
            const cwick_st_tensor *_s = mm_t(                              \
                m, "language_model.model.layers.%d." nm ".weight_scale_inv", L); \
            if (mm_bind(&y->field, nm, _w, _s)) return -1;                 \
        } while (0)
        QW(wq, "self_attn.q_proj");
        QW(wk, "self_attn.k_proj");
        QW(wv, "self_attn.v_proj");
        QW(wo, "self_attn.o_proj");
        QW(idx_q, "self_attn.index_q_proj");
        QW(idx_k, "self_attn.index_k_proj");

        if (y->is_dense) {
            QW(dg, "mlp.gate_proj");
            QW(du, "mlp.up_proj");
            QW(dd, "mlp.down_proj");
            continue;
        }
        QW(sw1, "block_sparse_moe.shared_experts.gate_proj");
        QW(sw2, "block_sparse_moe.shared_experts.down_proj");
        QW(sw3, "block_sparse_moe.shared_experts.up_proj");

        const cwick_st_tensor *gw = mm_t(m,
            "language_model.model.layers.%d.block_sparse_moe.gate.weight", L);
        const cwick_st_tensor *gb = mm_t(m,
            "language_model.model.layers.%d.block_sparse_moe."
            "e_score_correction_bias", L);
        if (!gw || !gb) {
            cwick_error_set(err, CWICK_ERR_CONTRACT,
                            "mm: layer %d has no router", L);
            return -1;
        }
        y->gate_w = (const float *)gw->data;
        y->gate_bias = (const float *)gb->data;

        y->experts = (mm_expert *)cwick_xmalloc(sizeof(mm_expert) *
                                                (size_t)c->n_routed);
        memset(y->experts, 0, sizeof(mm_expert) * (size_t)c->n_routed);
        static const char *mat[3] = { "w1", "w2", "w3" };
        for (int e = 0; e < c->n_routed; e++) {
            mm_expert *x = &y->experts[e];
            for (int k = 0; k < 3; k++) {
                const cwick_st_tensor *tw = mm_t(m,
                    "language_model.model.layers.%d.block_sparse_moe."
                    "experts.%d.%s.weight_packed", L, e, mat[k]);
                const cwick_st_tensor *ts = mm_t(m,
                    "language_model.model.layers.%d.block_sparse_moe."
                    "experts.%d.%s.weight_scale", L, e, mat[k]);
                const cwick_st_tensor *tg = mm_t(m,
                    "language_model.model.layers.%d.block_sparse_moe."
                    "experts.%d.%s.weight_global_scale", L, e, mat[k]);
                if (!tw || !ts || !tg) {
                    cwick_error_set(err, CWICK_ERR_CONTRACT,
                                    "mm: layer %d expert %d %s is missing",
                                    L, e, mat[k]);
                    return -1;
                }
                x->tw[k] = *tw;
                x->ts[k] = *ts;
                x->p[k] = (const uint8_t *)tw->data;
                x->sc[k] = (const uint8_t *)ts->data;
                float gs = ((const float *)tg->data)[0];
                /* The probe settles the direction: as a multiplier these bytes
                 * give weights around 1e6, as a divisor they land where a
                 * 6144-wide projection belongs. */
                x->inv_g[k] = gs != 0.0f ? 1.0f / gs : 1.0f;
            }
        }
    }
#undef QW

    /* ---- no widening ----
     *
     * The FP8 matrices are used as they are stored: e4m3 payload plus one e8m0
     * scale per 32 K elements, decoded inside the GEMV (mm_fp8_gemm). Widening
     * them to bf16 at load time used to cost a 19.67 GiB anonymous copy and
     * 21.1 GB of DRAM traffic per token; the FP8 path moves 10.6 GB and is
     * computed, not bandwidth-bound, so the copy bought nothing.
     *
     * What it does cost is residency: the FP8 weights are now file-backed
     * pages, so the memory plan has to leave room for them or every step turns
     * into an SSD read of the model. `fp8_total` is that reservation.
     */
    size_t fp8_total = 0, fp8_scales = 0;
    for (int L = 0; L < c->n_layers && L < MM_MAX_LAYERS; L++) {
        mm_layer *y = &m->layers[L];
        mm_qw *all[] = { &y->wq, &y->wk, &y->wv, &y->wo, &y->idx_q, &y->idx_k,
                         &y->sw1, &y->sw2, &y->sw3, &y->dg, &y->du, &y->dd };
        for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++)
            if (all[i]->rows > 0) {
                fp8_total += (size_t)(all[i]->rows * all[i]->cols);
                fp8_scales += (size_t)(all[i]->rows * (all[i]->cols / 32));
            }
    }

    /* Kept as mapped pages on purpose. Copying them into anonymous memory
     * makes them resident (footprint 82 -> 92 GiB) but the 10 GiB comes out of
     * the page cache the expert reads live in, and the projections did not get
     * faster: prefill went 89 -> 95 s. The mapped pages were already resident;
     * what looks like a residency problem below is something else. */
    m->dense16_used = fp8_total;
    CWICK_INFO("init", "mm dense weights stay in fp8: %.2f GiB as mapped pages "
               "(+%.2f GiB of block scales)", cwick_gib(fp8_total),
               cwick_gib(fp8_scales));

    /* ---- expert pin cache ----
     *
     * The routed experts are 174 GiB and a decode step reaches 228 of them
     * (57 layers x top-4), so this cache is the only thing standing between
     * the forward pass and 6.8 GiB of SSD traffic per token. It is also the
     * only line of the memory plan that can be given back, which is why the
     * budget is derived from what the process actually spends elsewhere
     * rather than from a flat fraction of RAM.
     *
     * A failure here costs speed, not correctness: without it every expert
     * read goes through the shard mapping. Say so out loud, because the only
     * other symptom is a "0/N hit" line in the metrics after the fact. */
    {
        cwick_stream_options so;
        memset(&so, 0, sizeof(so));
        so.arena_bytes = 32u << 20;    /* one expert matrix is 9-19 MiB */
        /* F_NOCACHE, but off by default, and that is the opposite of the
         * DeepSeek backend's choice for a reason that is worth writing down
         * because it is the whole token rate here.
         *
         * This model reads 6.9 GiB of routed experts per token. The pin cache
         * holds 28% of the bank, so ~5 GiB of every step is a miss, and a
         * miss has to come from RAM to be affordable: from RAM it costs
         * ~50 ms a step, from the SSD ~1.2 s. With F_NOCACHE every miss went
         * to the SSD and decode measured 0.4 tok/s; the same bytes served
         * from the page cache are what made the earlier mmap-only runs look
         * fast. Clean file-backed pages also stay out of phys_footprint, so
         * the VM can drop them under pressure without the steering valve
         * mistaking them for an overshoot. CWICK_MM_NOCACHE=1 restores the
         * old behaviour for a machine with less RAM than this one. */
        {
            const char *nc = getenv("CWICK_MM_NOCACHE");
            so.nocache = nc && *nc && *nc != '0';
        }
        uint64_t ceiling = 0;
        so.expert_cache_bytes = mm_expert_cache_budget(m, m->dense16_used,
                                                       &ceiling);
        cwick_metrics_set_memory_ceiling(ceiling);
        m->rss_target = ceiling;
        cwick_error serr;
        cwick_error_clear(&serr);
        if (cwick_stream_open(&m->saf, &so, &m->stream, &serr) != CWICK_OK) {
            CWICK_INFO("init", "mm expert cache unavailable (%s); experts are "
                       "read through the shard mapping",
                       serr.message[0] ? serr.message : "stream open failed");
            m->stream = NULL;
        } else {
            /* MiniMax keeps expert blobs in plain memory (every GEMV here runs
             * on the CPU pool), and that is the path where a fixed-size blob
             * arena pays: it removes ~600 malloc/munmap pairs per step. The
             * other two backends hand these blobs to the Metal arena, whose
             * allocator is already built for 8 MiB units, so they keep it. */
            cwick_stream_enable_blob_pool(m->stream);
            uint64_t bank = 0;
            for (int L = c->first_k_dense; L < c->n_layers; L++)
                for (int e = 0; e < c->n_routed; e++)
                    for (int k = 0; k < 3; k++)
                        bank += m->layers[L].experts[e].tw[k].data_bytes +
                                m->layers[L].experts[e].ts[k].data_bytes;
            CWICK_INFO("init", "mm expert pin cache %.1f GiB (dense %.1f GiB + "
                       "KV %.1f GiB + GPU/scratch reserve; ceiling %.1f GiB = "
                       "80%% of host RAM; %.1f%% of the %.1f GiB expert bank)",
                       (double)so.expert_cache_bytes / (1024.0 * 1024.0 * 1024.0),
                       (double)m->dense16_used / (1024.0 * 1024.0 * 1024.0),
                       (double)mm_kv_bytes(m, c->max_seq_len) /
                           (1024.0 * 1024.0 * 1024.0),
                       (double)ceiling / (1024.0 * 1024.0 * 1024.0),
                       bank ? 100.0 * (double)so.expert_cache_bytes / (double)bank
                            : 0.0,
                       (double)bank / (1024.0 * 1024.0 * 1024.0));
        }
    }

    /* ---- runtime buffers ---- */
    const int DIM = c->dim, NH = c->n_heads, NKV = c->n_kv_heads;
    const int HD = c->head_dim, QD = NH * HD, KVD = NKV * HD;
    const int64_t S = c->max_seq_len;
    size_t kvsz = (size_t)c->n_layers * (size_t)S * (size_t)(KVD * 2);
    size_t idsz = (size_t)c->n_layers * (size_t)S * (size_t)c->index_dim;
    m->kcache = (uint16_t *)cwick_xmalloc(kvsz * sizeof(uint16_t));
    m->vcache = (uint16_t *)cwick_xmalloc(kvsz * sizeof(uint16_t));
    m->idxcache = (uint16_t *)cwick_xmalloc(idsz * sizeof(uint16_t));

    const int MAXSEL = (c->topk_blocks + c->local_blocks + 2) * c->block_size;
    m->max_sel = MAXSEL;
    /* The three dense layers widen to dense_inter_dim (12288), four times the
     * routed experts' 3072, and prefill stages one row per position: sizing
     * these to inter_dim wrote 1.5M floats past the end of a 393K one on the
     * first 128-position chunk, which surfaced as a SIGSEGV in an unrelated
     * expert GEMV two layers later. */
    const int FF = c->dense_inter_dim > c->inter_dim ? c->dense_inter_dim
                                                     : c->inter_dim;
#define A(p, cnt) ((p) = (float *)cwick_xmalloc(sizeof(float) * (size_t)(cnt)))
    A(m->h, (size_t)MM_CHUNK * DIM);
    A(m->h2, DIM);
    A(m->sub, (size_t)MM_CHUNK * DIM);
    A(m->subl, (size_t)MM_CHUNK * DIM);
    A(m->ch_q, (size_t)MM_CHUNK * QD);
    A(m->ch_k, (size_t)MM_CHUNK * KVD);
    A(m->ch_v, (size_t)MM_CHUNK * KVD);
    A(m->ch_o, (size_t)MM_CHUNK * QD);
    A(m->ch_iq, (size_t)MM_CHUNK * c->index_n_heads * c->index_dim);
    A(m->ch_ik, (size_t)MM_CHUNK * c->index_dim);
    A(m->att, (size_t)NH * MAXSEL);
    A(m->scores, S);
    A(m->bscore, S / c->block_size + 4);
    A(m->gsm, (size_t)MM_CHUNK * FF);
    A(m->usm, (size_t)MM_CHUNK * FF);
    A(m->router, c->n_routed);
    A(m->moe_y, DIM);
    A(m->ch_contrib, (size_t)MM_CHUNK * c->n_activated * DIM);
    {
        const size_t NP = (size_t)MM_CHUNK * (size_t)c->n_activated;
        A(m->ex_x, NP * DIM);
        A(m->ex_g, NP * c->inter_dim);
        A(m->ex_u, NP * c->inter_dim);
        A(m->ex_y, NP * DIM);
    }
#undef A
    m->sel = (int *)cwick_xmalloc(sizeof(int) * (size_t)c->n_activated);
    m->selw = (float *)cwick_xmalloc(sizeof(float) * (size_t)c->n_activated);
    m->heap = (float *)cwick_xmalloc(sizeof(float) *
                                     (size_t)(c->topk_blocks + 4));
    m->heap_idx = (int *)cwick_xmalloc(sizeof(int) * (size_t)(c->topk_blocks + 4));
    m->blk_out = (int *)cwick_xmalloc(sizeof(int) * (size_t)(c->topk_blocks + 4));
    m->ch_sel = (int *)cwick_xmalloc(sizeof(int) * (size_t)MM_CHUNK *
                                     c->n_activated);
    m->ch_selw = (float *)cwick_xmalloc(sizeof(float) * (size_t)MM_CHUNK *
                                        c->n_activated);
    m->ch_uni = (int *)cwick_xmalloc(sizeof(int) * (size_t)c->n_routed);
    m->ch_mark = (uint8_t *)cwick_xmalloc((size_t)c->n_routed);
    m->ch_rows = (int *)cwick_xmalloc(sizeof(int) * (size_t)MM_CHUNK *
                                      c->n_activated);
    m->evbuf = (float *)cwick_xmalloc(sizeof(float) * (size_t)(c->dim + 16));
    m->odbuf = (float *)cwick_xmalloc(sizeof(float) * (size_t)(c->dim + 16));
    m->selmask = (uint8_t *)cwick_xmalloc((size_t)(S / c->block_size + 4));

    /* ---- rope tables ---- */
    const int half = c->rotary_dim / 2;
    m->rope_cos = (float *)cwick_xmalloc(sizeof(float) * (size_t)S * half);
    m->rope_sin = (float *)cwick_xmalloc(sizeof(float) * (size_t)S * half);
    for (int64_t p = 0; p < S; p++)
        for (int i = 0; i < half; i++) {
            double inv = pow(c->rope_theta, (double)(2 * i) / (double)c->rotary_dim);
            double ang = (double)p / inv;
            m->rope_cos[p * half + i] = (float)cos(ang);
            m->rope_sin[p * half + i] = (float)sin(ang);
        }

    m->loaded = 1;
    m->pos = 0;
    CWICK_INFO("init", "mm ready: %zu tensors, KV %llu MiB for %lld positions",
               m->saf.tensor_count,
               (unsigned long long)((kvsz * 2 + idsz) * 2 >> 20),
               (long long)S);
    return 0;
}

void mm_model_close(mm_model *m) {
    if (!m || !m->loaded) return;
    for (int L = 0; L < m->cfg.n_layers && L < MM_MAX_LAYERS; L++)
        free(m->layers[L].experts);
    if (m->stream) { cwick_stream_close(m->stream); m->stream = NULL; }
    free(m->pin_base);
    free(m->dense16);
    free(m->kcache); free(m->vcache); free(m->idxcache);
    free(m->h); free(m->h2); free(m->sub); free(m->subl);
    free(m->ch_q); free(m->ch_k); free(m->ch_v); free(m->ch_o);
    free(m->ch_iq); free(m->ch_ik);
    free(m->att); free(m->scores); free(m->bscore);
    free(m->gsm); free(m->usm); free(m->router); free(m->moe_y);
    free(m->ch_contrib);
    free(m->sel); free(m->selw); free(m->heap); free(m->heap_idx);
    free(m->blk_out); free(m->ch_sel); free(m->ch_selw); free(m->ch_uni);
    free(m->ch_mark); free(m->ch_rows); free(m->evbuf); free(m->odbuf);
    free(m->ex_x); free(m->ex_g); free(m->ex_u); free(m->ex_y);
    free(m->selmask);
    free(m->rope_cos); free(m->rope_sin);
    cwick_safetensors_close(&m->saf);
    m->loaded = 0;
}

uint64_t mm_kv_bytes(const mm_model *m, int max_seq) {
    const mm_config *c = &m->cfg;
    return (uint64_t)c->n_layers *
           ((uint64_t)c->n_kv_heads * c->head_dim * 2 +
            (uint64_t)c->index_dim) *
           (uint64_t)max_seq * sizeof(uint16_t);
}

void mm_model_reset(mm_model *m) { if (m) m->pos = 0; }

/* ------------------------------------------------------------------ */
/* expert reads                                                        */
/* ------------------------------------------------------------------ */

/* Packed bytes and e4m3 scales for (layer, expert, matrix), lookup only.
 *
 * The read itself happens in mm_expert_prefetch_layer, one layer ahead of the
 * compute; this function exists so the hot loop cannot accidentally do I/O in
 * the middle of a GEMV. The two come back through `scale_out` rather than as
 * `packed + data_bytes` because that adjacency only holds inside the pin
 * cache -- in the mapped shards the two are separate tensors, which is how
 * this port first produced NaN. */

/* Debugging aid gated by MM_TRACE_W: hash the bytes a GEMV is about to read.
 * The same expert matrix has to hash the same way in every run and every
 * position; a difference here means the stream handed out bytes that are not
 * the tensor, which the profile cannot see because it only counts them. */
static int mm_trace_w_on(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("MM_TRACE_W") != NULL;
    return cached;
}
static void mm_trace_w(int L, int e, int mat, const uint8_t *d, size_t dn,
                       const uint8_t *sc, size_t sn) {
    if (!mm_trace_w_on()) return;
    uint64_t h = 0x9E3779B97F4A7C15ull;
    const uint8_t *src[2] = { d, sc };
    size_t len[2] = { dn, sn };
    for (int which = 0; which < 2; which++)
        for (size_t i = 0; i < len[which]; i++) {
            h = (h ^ (src[which][i] + 0x9E3779B9u + (h << 6) + (h >> 2))) *
                0x2545F4914F6CDD1Dull;
        }
    fprintf(stderr, "W L%02d e%03d m%d %016llx\n", L, e, mat,
            (unsigned long long)(h >> 32));
}

static const uint8_t *mm_expert_lookup(mm_model *m, int L, int e, int mat,
                                       const uint8_t **scale_out) {
    mm_expert *x = &m->layers[L].experts[e];
    if (m->stream) {
        size_t bytes = 0;
        const void *b = cwick_stream_expert_cached(m->stream, L, e, mat, &bytes);
        if (b) {
            *scale_out = (const uint8_t *)b + x->tw[mat].data_bytes;
            return (const uint8_t *)b;
        }
    }
    *scale_out = x->sc[mat];
    return x->p[mat];
}

/* ------------------------------------------------------------------ */
/* expert reads: one layer's misses, in parallel                        */
/* ------------------------------------------------------------------ */

typedef struct {
    cwick_stream *s;
    const cwick_st_tensor *t, *st;
    uint8_t *dst;
    size_t tbytes;
    int64_t layer, expert, matrix;
    int ok;
} mm_read_task;

static void mm_read_range(void *argp, int64_t i0, int64_t i1) {
    mm_read_task *v = (mm_read_task *)argp;
    for (int64_t i = i0; i < i1; i++) {
        cwick_error e;
        cwick_error_clear(&e);
        if (cwick_stream_read(v[i].s, v[i].t, v[i].dst, &e) != CWICK_OK) {
            v[i].ok = 0;
            continue;
        }
        if (v[i].st && cwick_stream_read(v[i].s, v[i].st,
                                        v[i].dst + v[i].tbytes, &e) != CWICK_OK)
            v[i].ok = 0;
    }
}

/* Spread `n` reads over the pool.
 *
 * `cwick_pool_for` divides work by CWICK_POOL_MIN_WORK, which is about the
 * cost of a *row*, so a decode layer's three or four misses fall below it and
 * are run inline -- one pread after another on the calling thread. The whole
 * point of the read phase is the opposite: one 10 MiB read at a time sustains
 * 6.6 GB/s on this device and four in flight sustain 10.2, which is where the
 * same-shard ceiling sits. So the dispatch is by item count, not by row count
 * (pool.c makes the same argument for the same reason). MM_READ_PAR=0 keeps
 * the old serial path as an A/B control. */
static void mm_read_dispatch(mm_read_task *t, int n) {
    static int par = -1;
    if (par < 0) {
        const char *e = getenv("MM_READ_PAR");
        par = (e && *e && atoi(e) == 0) ? 0 : 1;
    }
    if (n <= 0) return;
    if (!par || n == 1) mm_read_range(t, 0, n);
    else cwick_pool_for_items(n, t, mm_read_range);
}

/* Bring one layer's expert matrices into the pin cache with the reads spread
 * over the pool.
 *
 * This is the difference between "the expert phase" and "waiting for the SSD":
 * a decode step misses ~1 GiB here, and one 10 MiB pread at a time sustains
 * ~1.7 GB/s on this machine while several in flight reach 9 GB/s. Reserve on
 * this thread (the cache index is not thread-safe), read on the pool, then
 * compute -- and because nothing is read during compute any more, the blob a
 * GEMV is handed can no longer be evicted from under it, which is what the
 * first version of this port crashed on. */
/* ------------------------------------------------------------------ */
/* read-ahead: the first half here, the second half on helper threads  */
/* ------------------------------------------------------------------ */
/* The device sustains ~4 GB/s and every core is idle while a layer's reads
 * are outstanding, so the two phases are serialised for no reason: 96 ms of a
 * 250 ms token is spent waiting for the SSD with nothing else to do.
 *
 * Read the first half on the pool as before, hand the second half to a small
 * group of helper threads, compute the first half while they run, and join
 * before the
 * first expert that lives in the second half. The helpers do not go through
 * the pool -- they read directly, so they run *alongside* the GEMVs instead of
 * queueing behind them. */

typedef struct {
    mm_read_task *t;
    int n, next;
    pthread_mutex_t mtx;
    mm_model *m;
} mm_async_ctx;

static void *mm_async_worker(void *p) {
    mm_async_ctx *c = (mm_async_ctx *)p;
    for (;;) {
        pthread_mutex_lock(&c->mtx);
        int i = c->next++;
        pthread_mutex_unlock(&c->mtx);
        if (i >= c->n) break;
        mm_read_range(c->t, i, i + 1);
    }
    return NULL;
}

static mm_async_ctx g_async;
static pthread_t g_async_th[4];
static int g_async_nth;
static int g_async_active;

/* The helper-thread read-ahead used to be prefill-only, on the argument that a
 * decode layer has four experts so the split has nothing to split. That is
 * backwards: a decode layer's io is about three times its compute, so the read
 * is the bigger half and it runs with every core idle.
 *
 * Measured over 24 steady-state steps (MM_PROF means, same prompt, alternating
 * runs): io 388 -> 200 ms, experts 123 -> 282 ms, step 664 -> 638 ms. The wait
 * lands inside the expert phase because the join is there, so the honest
 * reading is "188 ms of waiting moved, 26 ms of it disappeared". On the first
 * few steps of a session, where the cache is cold and io is 426 ms, the same
 * change is worth 277 ms a step (713 -> 436). Small in steady state, large on
 * a cold start, never a loss -- so it is on by default and
 * MM_IO_OVERLAP_DECODE=0 restores the serialised read. Values do not change:
 * the join still happens before the first expert that lives in the second
 * half. */
static int mm_io_overlap_decode(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("MM_IO_OVERLAP_DECODE");
        cached = (e && *e && *e == '0') ? 0 : 1;
    }
    return cached;
}

static int mm_io_overlap_on(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("MM_IO_OVERLAP");
        cached = (e && *e && *e == '0') ? 0 : 1;
    }
    return cached;
}

/* Bookkeeping for a batch of reads: a blob whose read failed must never be
 * served as a cache hit, and the byte counters drive the profile. */
static void mm_read_account(mm_model *m, mm_read_task *t, int n) {
    for (int i = 0; i < n; i++) {
        if (!t[i].ok) {
            cwick_stream_expert_abort(m->stream, t[i].layer, t[i].expert,
                                     t[i].matrix);
            continue;
        }
        /* The blob is complete now, so it becomes an eviction candidate
         * again. Until this call the pin cache treats it as reserved-and-
         * unfilled and will not take its memory away for another admission --
         * which is what it used to do while the reads for this very layer were
         * still queued. */
        cwick_stream_expert_settle(m->stream, t[i].layer, t[i].expert,
                                   t[i].matrix);
        m->n_expert_reads++;
        m->n_expert_bytes += t[i].tbytes +
                             (t[i].st ? t[i].st->data_bytes : 0);
    }
}

static void mm_read_start(mm_model *m, mm_read_task *t, int n) {
    if (n <= 0) return;
    g_async.t = t;
    g_async.n = n;
    g_async.next = 0;
    g_async.m = m;
    pthread_mutex_init(&g_async.mtx, NULL);
    const int want = n < 4 ? n : 4;
    g_async_nth = 0;
    for (int i = 0; i < want; i++) {
        if (pthread_create(&g_async_th[i], NULL, mm_async_worker, &g_async))
            break;
        g_async_nth++;
    }
    if (g_async_nth == 0) {          /* could not spawn: do it here instead */
        mm_read_range(g_async.t, 0, g_async.n);
        mm_read_account(m, g_async.t, g_async.n);
        pthread_mutex_destroy(&g_async.mtx);
        return;
    }
    g_async_active = 1;
}

static void mm_read_join(void) {
    if (!g_async_active) return;
    for (int i = 0; i < g_async_nth; i++) pthread_join(g_async_th[i], NULL);
    pthread_mutex_destroy(&g_async.mtx);
    g_async_active = 0;
    mm_read_account(g_async.m, g_async.t, g_async.n);
}

/* Lay out the read tasks for one layer, in the order the caller will consume
 * them, then read the first `split` of them here and (unless asked not to) the
 * rest on helper threads. `*umid_out` is the union index at which the caller
 * must join -- -1 when everything was read synchronously. */
static mm_read_task g_rd_tasks[128 * 3];

static void mm_expert_prefetch_layer(mm_model *m, int L, int *uni,
                                     int nun, int *umid_out, bool overlap) {
    *umid_out = -1;
    if (!m->stream || nun <= 0) return;
    double _ti = g_mm_now();
    enum { MM_MAX_TASKS = 128 * 3 };
    mm_read_task *tasks = g_rd_tasks;
    const int umid = nun / 2;
    int split = -1;
    int n = 0;
    for (int u = 0; u < nun && n < MM_MAX_TASKS; u++) {
        if (u == umid) split = n;
        int e = uni[u];
        if (e < 0 || e >= m->cfg.n_routed) continue;
        mm_expert *x = &m->layers[L].experts[e];
        for (int k = 0; k < 3 && n < MM_MAX_TASKS; k++) {
            size_t have = 0;
            if (cwick_stream_expert_cached(m->stream, L, e, k, &have)) continue;
            size_t tb = x->tw[k].data_bytes;
            size_t sb = x->ts[k].data_bytes;
            uint8_t *dst = cwick_stream_expert_reserve(m->stream, L, e, k,
                                                      tb + sb);
            if (!dst) continue;      /* not admitted: read through the mapping */
            tasks[n].s = m->stream;
            tasks[n].t = &x->tw[k];
            tasks[n].st = sb ? &x->ts[k] : NULL;
            tasks[n].dst = dst;
            tasks[n].tbytes = tb;
            tasks[n].layer = L;
            tasks[n].expert = e;
            tasks[n].matrix = k;
            tasks[n].ok = 1;
            n++;
        }
    }
    if (split < 0 || split > n) split = n;
    if (n == 0) return;
    /* Reading them in file-offset order was tried and is slightly *worse*
     * (3.65 GB/s against 3.98): a 10 MiB pread is already large enough to be
     * efficient, and twelve sequential streams beat one sweep here. */
    if (split > 0) {
        mm_read_dispatch(tasks, split);
        mm_read_account(m, tasks, split);
    }
    if (split < n) {
        /* Prefill only. Decode was tried on the same machinery -- read every
         * miss on the helpers, compute the resident experts first, join at the
         * boundary -- and it is inert: `overlap` is `n > 1`, because a decode
         * layer has four experts and one position, so the split has nothing to
         * split and the four helper threads are slower than the pool. Decode's
         * io problem needs a cache answer, not an overlap one. */
        if (overlap && mm_io_overlap_on()) {
            mm_read_start(m, tasks + split, n - split);
            *umid_out = umid;
        } else if (n - split == 1) {
            mm_read_range(tasks + split, 0, 1);
            mm_read_account(m, tasks + split, n - split);
        } else {
            mm_read_dispatch(tasks + split, n - split);
            mm_read_account(m, tasks + split, n - split);
        }
    }
    g_mm_t_io += g_mm_now() - _ti;
}

/* ------------------------------------------------------------------ */
/* routing                                                            */
/* ------------------------------------------------------------------ */

/* sigmoid(logits) for the weights, sigmoid(logits) + bias for the choice,
 * then top-k, renormalise and scale. The asymmetry is the point: the bias
 * exists to spread the load across experts, not to change how much a chosen
 * expert contributes. */
typedef struct {
    const float *w, *x;
    float *out;
    int64_t dim;
} mm_route_arg;

static void mm_route_range(void *argp, int64_t i0, int64_t i1) {
    mm_route_arg *g = (mm_route_arg *)argp;
    const int64_t D = g->dim;
    for (int64_t i = i0; i < i1; i++) {
        const float *r = g->w + i * D;
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        int64_t j = 0;
        for (; j + 8 <= D; j += 8) {
            a0 = vfmaq_f32(a0, vld1q_f32(r + j), vld1q_f32(g->x + j));
            a1 = vfmaq_f32(a1, vld1q_f32(r + j + 4), vld1q_f32(g->x + j + 4));
        }
        float acc = vaddvq_f32(vaddq_f32(a0, a1));
        for (; j < D; j++) acc += r[j] * g->x[j];
        g->out[i] = acc;
    }
}

static void mm_route(const mm_model *m, const mm_layer *y, const float *x,
                     int *sel, float *selw) {
    const mm_config *c = &m->cfg;
    float *lg = m->router;
    /* 128 rows of 6144 F32 per layer, 57 layers: ~180 MB and 45 M MAC per
     * token. Serially that measured 33 ms of every step -- a sixth of the
     * per-token compute, spent on the smallest matrix in the model. */
    mm_route_arg a = { y->gate_w, x, lg, c->dim };
    cwick_pool_for(c->n_routed, &a, mm_route_range);
    for (int i = 0; i < c->n_routed; i++) m->scores[i] =
        mm_sigmoid(lg[i]) + y->gate_bias[i];
    int k = mm_topk(m->scores, c->n_routed, c->n_activated, sel, m->heap,
                    m->heap_idx);
    double sum = 0.0;
    for (int i = 0; i < k; i++) {
        selw[i] = mm_sigmoid(lg[sel[i]]);
        sum += selw[i];
    }
    if (c->norm_topk_prob && sum > 1e-20) {
        for (int i = 0; i < k; i++) selw[i] /= (float)sum;
    }
    for (int i = 0; i < k; i++) selw[i] *= c->route_scale;
    for (; k < c->n_activated; k++) { sel[k] = 0; selw[k] = 0.0f; }
}

/* ------------------------------------------------------------------ */
/* indexer: which blocks this position may look at                     */
/* ------------------------------------------------------------------ */

/* Fill m->blk_out with the blocks query `p` may attend to and return the
 * number of positions selected. Blocks are scored by the largest per-position
 * index score inside them (`sparse_score_type: max`), the most recent block is
 * always kept, and the selection is clipped to positions already written. */
static int mm_select_blocks(mm_model *m, int64_t p) {
    const mm_config *c = &m->cfg;
    const int BS = c->block_size;
    const int64_t limit = p + 1;                 /* positions written so far */
    const int64_t nfull = limit / BS;            /* complete blocks */
    const int64_t nblk = (limit + BS - 1) / BS;

    memset(m->bscore, 0, sizeof(float) * (size_t)(nblk + 1));
    for (int64_t b = 0; b < nblk; b++) {
        int64_t s0 = b * BS;
        int64_t s1 = s0 + BS;
        if (s1 > limit) s1 = limit;
        if (s0 >= s1) continue;
        float best = -1e30f;
        for (int64_t s = s0; s < s1; s++) best = fmaxf(best, m->scores[s]);
        m->bscore[b] = best;
    }
    int nb = mm_topk(m->bscore, (int)nblk, c->topk_blocks, m->blk_out,
                     m->heap, m->heap_idx);

    /* the most recent block, always */
    bool have_last = false;
    for (int i = 0; i < nb; i++)
        if (m->blk_out[i] == (int)(limit / BS)) have_last = true;
    if (!have_last && nb < c->topk_blocks + 4)
        m->blk_out[nb++] = (int)(limit / BS);
    (void)nfull;
    return nb;
}

/* Per-position index scores for query `iq` against keys 0..limit-1. */
static void mm_index_scores(mm_model *m, const float *iq, int64_t limit) {
    const mm_config *c = &m->cfg;
    const int IH = c->index_n_heads, ID = c->index_dim;
    for (int64_t s = 0; s < limit; s++) {
        const uint16_t *k = m->idxcache +
            ((size_t)m->cur_layer * (size_t)c->max_seq_len + (size_t)s) * ID;
        float tot = 0.0f;
        for (int h = 0; h < IH; h++) {
            const float *qh = iq + h * ID;
            float d = 0.0f;
            for (int i = 0; i < ID; i++)
                d += qh[i] * cwick_bf16_to_f32(k[i]);
            if (d > 0.0f) tot += d;              /* ReLU, then sum over heads */
        }
        m->scores[s] = tot;
    }
}

/* ------------------------------------------------------------------ */
/* forward                                                             */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* attention kernels                                                    */
/* ------------------------------------------------------------------ */

/* bf16 is the upper half of an f32: widening is one shift and is exact.
 * The scalar per-element version of these two loops was the single largest
 * item in a long-prompt prefill (398 ms of an 830 ms per-token step at 270
 * positions), because it converts and multiplies one element per iteration
 * with no ILP. */
#if defined(__aarch64__)
static inline float32x4_t mm_bf16x4(uint16x4_t h) {
    return vreinterpretq_f32_u32(vshll_n_u16(h, 16));
}

static inline float mm_dot_bf16(const uint16_t *k, const float *q, int HD) {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
    int i = 0;
    for (; i + 16 <= HD; i += 16) {
        uint16x8_t u0 = vld1q_u16(k + i);
        uint16x8_t u1 = vld1q_u16(k + i + 8);
        a0 = vfmaq_f32(a0, mm_bf16x4(vget_low_u16(u0)), vld1q_f32(q + i));
        a1 = vfmaq_f32(a1, mm_bf16x4(vget_high_u16(u0)), vld1q_f32(q + i + 4));
        a2 = vfmaq_f32(a2, mm_bf16x4(vget_low_u16(u1)), vld1q_f32(q + i + 8));
        a3 = vfmaq_f32(a3, mm_bf16x4(vget_high_u16(u1)), vld1q_f32(q + i + 12));
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; i < HD; i++) s += q[i] * cwick_bf16_to_f32(k[i]);
    return s;
}

static inline void mm_axpy_bf16(float *oh, const uint16_t *v, float w, int HD) {
    const float32x4_t wv = vdupq_n_f32(w);
    int i = 0;
    for (; i + 8 <= HD; i += 8) {
        uint16x8_t u = vld1q_u16(v + i);
        float32x4_t o0 = vld1q_f32(oh + i);
        float32x4_t o1 = vld1q_f32(oh + i + 4);
        o0 = vfmaq_f32(o0, wv, mm_bf16x4(vget_low_u16(u)));
        o1 = vfmaq_f32(o1, wv, mm_bf16x4(vget_high_u16(u)));
        vst1q_f32(oh + i, o0);
        vst1q_f32(oh + i + 4, o1);
    }
    for (; i < HD; i++) oh[i] += w * cwick_bf16_to_f32(v[i]);
}
#else
static inline float mm_dot_bf16(const uint16_t *k, const float *q, int HD) {
    float s = 0.0f;
    for (int i = 0; i < HD; i++) s += q[i] * cwick_bf16_to_f32(k[i]);
    return s;
}

static inline void mm_axpy_bf16(float *oh, const uint16_t *v, float w, int HD) {
    for (int i = 0; i < HD; i++) oh[i] += w * cwick_bf16_to_f32(v[i]);
}
#endif

static int mm_forward_chunk(mm_model *m, const int32_t *ids, int n,
                            int64_t start, float *logits, cwick_error *err) {
    (void)err;
    /* Phase clock (MM_PROF=1). The token rate is set by whichever of these
     * dominates, and the candidates differ by an order of magnitude, so the
     * only useful answer is a measurement. Cumulative; printed every 16
     * calls so a decode run reports once per sixteen steps. */
    static double prof[10];
    static int prof_on = -1;
    if (prof_on < 0) {
        const char *e = getenv("MM_PROF");
        prof_on = (e && *e && atoi(e)) ? 1 : 0;
    }
    double t_prof = prof_on ? mm_now_ms() : 0.0;
#define PROF_SLOT(i) do { if (prof_on) { double _t = mm_now_ms(); \
        prof[i] += _t - t_prof; t_prof = _t; } } while (0)
    const mm_config *c = &m->cfg;
    const int DIM = c->dim, NH = c->n_heads, NKV = c->n_kv_heads;
    const int HD = c->head_dim, QD = NH * HD, KVD = NKV * HD;
    const int ID = c->index_dim, IH = c->index_n_heads;
    const int64_t S = c->max_seq_len;
    const float qscale = 1.0f / sqrtf((float)HD);

    float *h = m->h, *sub = m->sub, *subl = m->subl;

    /* Finding the first non-finite value is the whole job when a port is wrong
     * somewhere in sixty layers, so the check is built in: MM_NAN_AT=1 prints
     * the stage and the layer where the stream first goes bad. */
    static int nan_check = -1;
    if (nan_check < 0) {
        const char *e = getenv("MM_NAN_AT");
        nan_check = (e && *e && atoi(e)) ? 1 : 0;
    }
#define MM_NAN(stage)                                                        \
    do {                                                                     \
        if (nan_check) {                                                     \
            for (int _t = 0; _t < n; _t++)                                    \
                for (int _i = 0; _i < DIM; _i++) {                            \
                    float _v = h[(size_t)_t * DIM + _i];                      \
                    if (!isfinite(_v)) {                                      \
                        fprintf(stderr, "mm: non-finite after %s at layer %d"  \
                                " pos %d dim %d\n", stage, L, _t, _i);        \
                        nan_check = 0;                                        \
                        goto mm_nan_done;                                     \
                    }                                                         \
                }                                                             \
        }                                                                     \
    } while (0)

    for (int t = 0; t < n; t++) {
        const uint16_t *e = m->embed + (size_t)ids[t] * DIM;
        float *ht = h + (size_t)t * DIM;
        for (int i = 0; i < DIM; i++) ht[i] = cwick_bf16_to_f32(e[i]);
    }
    {
        for (int t = 0; t < n; t++)
            for (int i = 0; i < DIM; i++)
                if (!isfinite(h[(size_t)t * DIM + i]) && nan_check) {
                    fprintf(stderr, "mm: non-finite in the embedding\n");
                    nan_check = 0;
                }
    }

    PROF_SLOT(0);   /* embedding */
    for (int L = 0; L < c->n_layers; L++) {
        mm_layer *y = &m->layers[L];
        m->cur_layer = L;

        for (int t = 0; t < n; t++)
            mm_rmsnorm_gemma(sub + (size_t)t * DIM, h + (size_t)t * DIM,
                             y->in_norm, DIM, c->rms_norm_eps);

        /* The five projections of a layer are independent, and on a decode
         * step each is a GEMV, so they go out as one Metal command buffer
         * instead of five commit-and-wait round trips. `_batch_end` returns 0
         * if any item could not be served and nothing was written, which
         * makes the CPU block below the fallback rather than a duplicate. */
        {
            mm_fp8_gemm(m->ch_q, n, y->wq.w, y->wq.s, sub, QD, DIM);
            mm_fp8_gemm(m->ch_k, n, y->wk.w, y->wk.s, sub, KVD, DIM);
            mm_fp8_gemm(m->ch_v, n, y->wv.w, y->wv.s, sub, KVD, DIM);
            mm_fp8_gemm(m->ch_iq, n, y->idx_q.w, y->idx_q.s, sub, IH * ID, DIM);
            mm_fp8_gemm(m->ch_ik, n, y->idx_k.w, y->idx_k.s, sub, ID, DIM);
        }
        PROF_SLOT(1);   /* q/k/v/index projections */
        mm_trace("proj", L, m->ch_q, (size_t)n * (size_t)QD);

        for (int t = 0; t < n; t++) {
            int64_t p = start + t;
            const float *cs = m->rope_cos + (size_t)p * (c->rotary_dim / 2);
            const float *sn = m->rope_sin + (size_t)p * (c->rotary_dim / 2);

            float *q = m->ch_q + (size_t)t * QD;
            mm_head_norm(q, y->q_norm, NH, HD, c->rms_norm_eps);
            for (int hh = 0; hh < NH; hh++)
                mm_rope(q + hh * HD, c->rotary_dim, cs, sn);

            float *kv = m->ch_k + (size_t)t * KVD;
            mm_head_norm(kv, y->k_norm, NKV, HD, c->rms_norm_eps);
            for (int hh = 0; hh < NKV; hh++)
                mm_rope(kv + hh * HD, c->rotary_dim, cs, sn);

            /* write the caches */
            uint16_t *kd = m->kcache + ((size_t)L * S + (size_t)p) * KVD;
            uint16_t *vd = m->vcache + ((size_t)L * S + (size_t)p) * KVD;
            for (int i = 0; i < KVD; i++) {
                kd[i] = f_to_bf16(kv[i]);
                vd[i] = f_to_bf16((m->ch_v + (size_t)t * KVD)[i]);
            }

            float *iq = m->ch_iq + (size_t)t * (IH * ID);
            float *ik = m->ch_ik + (size_t)t * ID;
            mm_head_norm(iq, y->idx_q_norm, IH, ID, c->rms_norm_eps);
            mm_rmsnorm_gemma(ik, ik, y->idx_k_norm, ID, c->rms_norm_eps);
            uint16_t *id = m->idxcache + ((size_t)L * S + (size_t)p) * ID;
            for (int i = 0; i < ID; i++) id[i] = f_to_bf16(ik[i]);
        }

        PROF_SLOT(2);   /* norms, rope, cache writes */
        /* ---- attention ---- */
        for (int t = 0; t < n; t++) {
            int64_t p = start + t;
            const int64_t limit = p + 1;
            float *iq = m->ch_iq + (size_t)t * (IH * ID);
            mm_index_scores(m, iq, limit);
            int nb = mm_select_blocks(m, p);

            int ns = 0;
            int64_t selpos[4096];
            for (int b = 0; b < nb && ns < 4096; b++) {
                int64_t s0 = (int64_t)m->blk_out[b] * c->block_size;
                int64_t s1 = s0 + c->block_size;
                if (s1 > limit) s1 = limit;
                for (int64_t s = s0; s < s1 && ns < 4096; s++) selpos[ns++] = s;
            }

            const float *q = m->ch_q + (size_t)t * QD;
            float *out = m->ch_o + (size_t)t * QD;
            for (int hh = 0; hh < NH; hh++) {
                const float *qh = q + hh * HD;
                float *sc = m->att + (size_t)hh * m->max_sel;
                float best = -1e30f;
                const size_t kvh = (size_t)(hh / (NH / NKV)) * HD;
                for (int i = 0; i < ns; i++) {
                    const uint16_t *k = m->kcache +
                        ((size_t)L * S + (size_t)selpos[i]) * KVD + kvh;
                    float d = mm_dot_bf16(k, qh, HD) * qscale;
                    sc[i] = d;
                    if (d > best) best = d;
                }
                float sum = 0.0f;
                for (int i = 0; i < ns; i++) {
                    sc[i] = expf(sc[i] - best);
                    sum += sc[i];
                }
                float *oh = out + hh * HD;
                for (int i = 0; i < ns; i++) sc[i] /= sum;
                if ((HD & 31) == 0) {
                    /* Blocked over the head dimension so the accumulator stays
                     * in registers instead of being read and written per key.
                     *
                     * Measured: 16 -> 15 ms/token, i.e. this was NOT the
                     * bandwidth problem it looks like (64 heads x 302 keys x
                     * 1 KB of read-modify-write per layer). The accumulator is
                     * 128 floats and lives in L1, so the stores were cheap;
                     * what is left is the arithmetic. Kept because it is
                     * strictly less traffic for the same result, but do not
                     * expect more than this from it -- the remaining attention
                     * time is 64 query heads each re-reading the KV head they
                     * share under GQA, which needs a restructure, not a block. */
                    for (int blk = 0; blk < HD; blk += 32) {
                        float32x4_t o0 = vdupq_n_f32(0.0f), o1 = o0, o2 = o0;
                        float32x4_t o3 = o0, o4 = o0, o5 = o0, o6 = o0, o7 = o0;
                        for (int i = 0; i < ns; i++) {
                            const uint16_t *v = m->vcache +
                                ((size_t)L * S + (size_t)selpos[i]) * KVD +
                                kvh + blk;
                            const float32x4_t wv = vdupq_n_f32(sc[i]);
                            o0 = vfmaq_f32(o0, wv, mm_bf16x4(vld1_u16(v + 0)));
                            o1 = vfmaq_f32(o1, wv, mm_bf16x4(vld1_u16(v + 4)));
                            o2 = vfmaq_f32(o2, wv, mm_bf16x4(vld1_u16(v + 8)));
                            o3 = vfmaq_f32(o3, wv, mm_bf16x4(vld1_u16(v + 12)));
                            o4 = vfmaq_f32(o4, wv, mm_bf16x4(vld1_u16(v + 16)));
                            o5 = vfmaq_f32(o5, wv, mm_bf16x4(vld1_u16(v + 20)));
                            o6 = vfmaq_f32(o6, wv, mm_bf16x4(vld1_u16(v + 24)));
                            o7 = vfmaq_f32(o7, wv, mm_bf16x4(vld1_u16(v + 28)));
                        }
                        vst1q_f32(oh + blk + 0, o0);
                        vst1q_f32(oh + blk + 4, o1);
                        vst1q_f32(oh + blk + 8, o2);
                        vst1q_f32(oh + blk + 12, o3);
                        vst1q_f32(oh + blk + 16, o4);
                        vst1q_f32(oh + blk + 20, o5);
                        vst1q_f32(oh + blk + 24, o6);
                        vst1q_f32(oh + blk + 28, o7);
                    }
                } else {
                    for (int i2 = 0; i2 < HD; i2++) oh[i2] = 0.0f;
                    for (int i = 0; i < ns; i++) {
                        const uint16_t *v = m->vcache +
                            ((size_t)L * S + (size_t)selpos[i]) * KVD + kvh;
                        mm_axpy_bf16(oh, v, sc[i], HD);
                    }
                }
            }
        }

        mm_trace("attn", L, m->ch_o, (size_t)n * (size_t)QD);
        PROF_SLOT(3);   /* indexer + attention */
        MM_NAN("attn-in");
        mm_fp8_gemm(subl, n, y->wo.w, y->wo.s, m->ch_o, DIM, QD);
        for (int t = 0; t < n; t++) {
            float *ht = h + (size_t)t * DIM;
            const float *sl = subl + (size_t)t * DIM;
            for (int i = 0; i < DIM; i++) ht[i] += sl[i];
        }
        mm_trace("wo", L, h, (size_t)n * (size_t)DIM);
        MM_NAN("attn-residual");

        /* ---- ffn ---- */
        for (int t = 0; t < n; t++)
            mm_rmsnorm_gemma(sub + (size_t)t * DIM, h + (size_t)t * DIM,
                             y->post_norm, DIM, c->rms_norm_eps);

        if (y->is_dense) {
            const int DI = c->dense_inter_dim;
            /* Batched for the same reason as the shared expert below: a chunk
             * is 302 positions, and a per-position GEMV reads the same 75 MB
             * matrix 302 times. */
            mm_fp8_gemm(m->gsm, n, y->dg.w, y->dg.s, sub, DI, DIM);
            mm_fp8_gemm(m->usm, n, y->du.w, y->du.s, sub, DI, DIM);
            for (size_t i = 0; i < (size_t)n * (size_t)DI; i++)
                m->gsm[i] = mm_swigluoai(m->gsm[i], m->usm[i],
                                         c->swiglu_alpha, c->swiglu_limit);
            mm_fp8_gemm(subl, n, y->dd.w, y->dd.s, m->gsm, DIM, DI);
            for (size_t i = 0; i < (size_t)n * (size_t)DIM; i++) h[i] += subl[i];
            /* Without this the three dense layers' MLP work was billed to the
             * *next* layer's projections, which made `proj` look like the
             * second largest item in a prefill. */
            mm_trace("denseout", L, h, (size_t)n * (size_t)DIM);
            PROF_SLOT(9);
            continue;
        }

        PROF_SLOT(4);   /* wo + residual */
        /* route every position, then walk the union of their choices */
        int nun = 0;
        memset(m->ch_mark, 0, (size_t)c->n_routed);
        for (int t = 0; t < n; t++) {
            mm_route(m, y, sub + (size_t)t * DIM,
                     m->ch_sel + (size_t)t * c->n_activated,
                     m->ch_selw + (size_t)t * c->n_activated);
            for (int k = 0; k < c->n_activated; k++) {
                int e = m->ch_sel[(size_t)t * c->n_activated + k];
                if (!m->ch_mark[e]) { m->ch_mark[e] = 1; m->ch_uni[nun++] = e; }
            }
        }

        for (int t = 0; t < n; t++)
            memset(m->ch_contrib + (size_t)t * c->n_activated * DIM, 0,
                   sizeof(float) * (size_t)c->n_activated * DIM);
        PROF_SLOT(5);   /* router */
        g_mm_n_uni += nun;
        int umid = -1;
        mm_expert_prefetch_layer(m, L, m->ch_uni, nun, &umid,
                                n > 1 || mm_io_overlap_decode());
        PROF_SLOT(8);   /* expert reads (SSD waiting) */

        for (int ui = 0; ui < nun; ui++) {
            /* The second half of this layer's reads has been running on the
             * helper threads since the prefetch returned; wait for it exactly
             * when the first expert that lives in it comes up. */
            if (ui == umid) mm_read_join();
            int e = m->ch_uni[ui];
            float i1 = m->layers[L].experts[e].inv_g[0];
            float i2 = m->layers[L].experts[e].inv_g[1];
            float i3 = m->layers[L].experts[e].inv_g[2];

            int nr = 0;
            for (int t = 0; t < n; t++)
                for (int k = 0; k < c->n_activated; k++)
                    if (m->ch_sel[(size_t)t * c->n_activated + k] == e)
                        m->ch_rows[nr++] = t * c->n_activated + k;
            if (nr == 0) continue;

            /* One live blob at a time, and that is a correctness rule, not a
             * style one: reading a miss into the pin cache makes room by
             * evicting other slots, which frees their blobs. Holding the gate
             * pointer across the read of `up` is therefore a use-after-free
             * -- it produced a SIGSEGV inside the NEON GEMV the first time a
             * 128-position chunk routed to more experts than the cache could
             * hold. Order matters for kernels instead: gate and up feed the
             * activation, so both are computed before down is fetched. */
            const int INTER = c->inter_dim;
            /* Grouped by default now that the kernel keeps one vector
             * accumulator per activation: it measured 1.5x (two positions) to
             * 1.85x (four) the cost of running the single-row kernel once per
             * position, because one decode of the weight block now serves
             * several activations. MM_PREFILL_GEMM=0 reverts to per-position. */
            static int pf_gemm = -1;
            if (pf_gemm < 0) {
                const char *e = getenv("MM_PREFILL_GEMM");
                pf_gemm = (e && *e && *e == '0') ? 0 : 1;
            }
            if (nr == 1 || !pf_gemm) {
                /* Decode: one position, one expert. The scalar GEMV is the
                 * right kernel here -- it keeps one vector accumulator per row
                 * instead of a horizontal add per block. */
                for (int rr = 0; rr < nr; rr++) {
                int tk = m->ch_rows[rr];
                const float *x = sub + (size_t)(tk / c->n_activated) * DIM;
                const uint8_t *sp = NULL;
                const uint8_t *p = mm_expert_lookup(m, L, e, 0, &sp);
                mm_trace_w(L, e, 0, p, m->layers[L].experts[e].tw[0].data_bytes,sp, m->layers[L].experts[e].ts[0].data_bytes);
                g_mm_site[3]++;
                mm_fp4_gemv(m->gsm, p, sp, i1, x, INTER, DIM, m->evbuf,
                            m->odbuf);
                p = mm_expert_lookup(m, L, e, 2, &sp);
                mm_trace_w(L, e, 2, p, m->layers[L].experts[e].tw[2].data_bytes,sp, m->layers[L].experts[e].ts[2].data_bytes);
                mm_fp4_gemv(m->usm, p, sp, i3, x, INTER, DIM, m->evbuf,
                            m->odbuf);
                for (int i = 0; i < INTER; i++)
                    m->gsm[i] = mm_swigluoai(m->gsm[i], m->usm[i],
                                             c->swiglu_alpha, c->swiglu_limit);
                p = mm_expert_lookup(m, L, e, 1, &sp);
                mm_trace_w(L, e, 1, p, m->layers[L].experts[e].tw[1].data_bytes,sp, m->layers[L].experts[e].ts[1].data_bytes);
                mm_fp4_gemv(m->moe_y, p, sp, i2, m->gsm, DIM, INTER,
                            m->evbuf, m->odbuf);
                float *dst = m->ch_contrib + (size_t)tk * DIM;
                for (int i = 0; i < DIM; i++) dst[i] = m->moe_y[i];
                }
                continue;
            }
            mm_trace("route", L, m->ch_selw, (size_t)n * (size_t)c->n_activated);
            /* Prefill: gather this expert's positions and read its matrix
             * once for all of them. */
            for (int r = 0; r < nr; r++)
                mm_deinterleave(m->ex_x + (size_t)r * DIM,
                                sub + (size_t)(m->ch_rows[r] / c->n_activated) * DIM,
                                DIM);
            {
                const uint8_t *sp = NULL;
                const uint8_t *p = mm_expert_lookup(m, L, e, 0, &sp);
                mm_trace_w(L, e, 0, p, m->layers[L].experts[e].tw[0].data_bytes, sp, m->layers[L].experts[e].ts[0].data_bytes);
                g_mm_site[0]++;
                mm_fp4_gemm(m->ex_g, nr, p, sp, i1, m->ex_x, INTER, DIM);
                if (getenv("MM_DEBUG_GEMM")) {
                    static int once = 0;
                    if (once < 4 && nr > 1) {
                        once++;
                        float *ref = (float *)cwick_xmalloc(
                            sizeof(float) * (size_t)INTER);
                        const float *xr = sub +
                            (size_t)(m->ch_rows[0] / c->n_activated) * DIM;
                        mm_fp4_gemv(ref, p, sp, i1, xr, INTER, DIM,
                                    m->evbuf, m->odbuf);
                        double worst = 0.0;
                        for (int i = 0; i < INTER; i++) {
                            double d = fabs((double)ref[i] - (double)m->ex_g[i]);
                            if (d > worst) worst = d;
                        }
                        fprintf(stderr, "mm gemm-vs-gemv L%d nr=%d n=%d: worst "
                                "%.6g (ref %.6g gemm %.6g)\n", L, nr, n, worst,
                                (double)ref[0], (double)m->ex_g[0]);
                        free(ref);
                    }
                }
                p = mm_expert_lookup(m, L, e, 2, &sp);
                mm_trace_w(L, e, 2, p, m->layers[L].experts[e].tw[2].data_bytes, sp, m->layers[L].experts[e].ts[2].data_bytes);
                g_mm_site[1]++;
                mm_fp4_gemm(m->ex_u, nr, p, sp, i3, m->ex_x, INTER, DIM);
            }
            for (size_t i = 0; i < (size_t)nr * INTER; i++)
                m->ex_g[i] = mm_swigluoai(m->ex_g[i], m->ex_u[i],
                                          c->swiglu_alpha, c->swiglu_limit);
            {
                /* The down matrix consumes the SwiGLU output, which is a plain
                 * vector: de-interleave it in place over ex_u (its up-projection
                 * content is already folded in). */
                for (int r = 0; r < nr; r++)
                    mm_deinterleave(m->ex_u + (size_t)r * INTER,
                                    m->ex_g + (size_t)r * INTER, INTER);
                const uint8_t *sp = NULL;
                const uint8_t *p = mm_expert_lookup(m, L, e, 1, &sp);
                mm_trace_w(L, e, 1, p, m->layers[L].experts[e].tw[1].data_bytes, sp, m->layers[L].experts[e].ts[1].data_bytes);
                g_mm_site[2]++;
                mm_fp4_gemm(m->ex_y, nr, p, sp, i2, m->ex_u, DIM, INTER);
            }
            for (int r = 0; r < nr; r++)
                memcpy(m->ch_contrib + (size_t)m->ch_rows[r] * DIM,
                       m->ex_y + (size_t)r * DIM, sizeof(float) * (size_t)DIM);
            if (getenv("MM_DEBUG_GEMM")) {
                static int once2 = 0;
                if (once2 < 3) {
                    once2++;
                    /* Recompute row 0 end to end with the scalar kernels. */
                    float *gref = (float *)cwick_xmalloc(sizeof(float) * (size_t)INTER);
                    float *uref = (float *)cwick_xmalloc(sizeof(float) * (size_t)INTER);
                    float *yref = (float *)cwick_xmalloc(sizeof(float) * (size_t)DIM);
                    const uint8_t *sq = NULL;
                    const float *xr = sub +
                        (size_t)(m->ch_rows[0] / c->n_activated) * DIM;
                    const uint8_t *pg = mm_expert_lookup(m, L, e, 0, &sq);
                    mm_trace_w(L, e, 0, pg, m->layers[L].experts[e].tw[0].data_bytes,sq, m->layers[L].experts[e].ts[0].data_bytes);
                    mm_fp4_gemv(gref, pg, sq, i1, xr, INTER, DIM, m->evbuf, m->odbuf);
                    pg = mm_expert_lookup(m, L, e, 2, &sq);
                    mm_trace_w(L, e, 2, pg, m->layers[L].experts[e].tw[2].data_bytes,sq, m->layers[L].experts[e].ts[2].data_bytes);
                    mm_fp4_gemv(uref, pg, sq, i3, xr, INTER, DIM, m->evbuf, m->odbuf);
                    for (int i = 0; i < INTER; i++)
                        gref[i] = mm_swigluoai(gref[i], uref[i],
                                               c->swiglu_alpha, c->swiglu_limit);
                    pg = mm_expert_lookup(m, L, e, 1, &sq);
                    mm_trace_w(L, e, 1, pg, m->layers[L].experts[e].tw[1].data_bytes,sq, m->layers[L].experts[e].ts[1].data_bytes);
                    mm_fp4_gemv(yref, pg, sq, i2, gref, DIM, INTER, m->evbuf, m->odbuf);
                    double wg = 0.0, wu = 0.0, wy = 0.0;
                    const float *cg = m->ch_contrib + (size_t)m->ch_rows[0] * DIM;
                    for (int i = 0; i < INTER; i++) {
                        double d = fabs((double)gref[i] - 0.0);
                        (void)d;
                    }
                    for (int i = 0; i < DIM; i++) {
                        double d = fabs((double)yref[i] - (double)cg[i]);
                        if (d > wy) wy = d;
                    }
                    fprintf(stderr, "mm down L%d nr=%d n=%d: |y_ref - contrib| "
                            "worst %.6g (ref %.6g contrib %.6g) g %.6g u %.6g\n",
                            L, nr, n, wy, (double)yref[0], (double)cg[0],
                            (double)gref[0], (double)uref[0]);
                    (void)wg; (void)wu;
                    free(gref); free(uref); free(yref);
                }
            }
        }

        PROF_SLOT(6);   /* routed experts */
        /* Shared expert over the whole chunk, as one GEMM.
         *
         * It used to run per position, i.e. 17,214 one-row GEMVs per chunk per
         * matrix, and that showed up as the single largest compute item in a
         * prefill: 9.7 s for sw1+sw3 and another ~5 s for sw2, all at 67
         * GMAC/s, where the same shapes as a chunk GEMM run at 233. The reason
         * is the same one that put the projections on a tiled kernel: a one-row
         * GEMV re-reads the whole weight for a single dot product, so it is
         * bound by bytes, and at n=302 those bytes are read once and used 302
         * times. */
        {
            const int INTER = c->inter_dim;
            mm_fp8_gemm(m->gsm, n, y->sw1.w, y->sw1.s, sub, INTER, DIM);
            mm_fp8_gemm(m->usm, n, y->sw3.w, y->sw3.s, sub, INTER, DIM);
            for (size_t i = 0; i < (size_t)n * (size_t)INTER; i++)
                m->gsm[i] = mm_swigluoai(m->gsm[i], m->usm[i],
                                         c->swiglu_alpha, c->swiglu_limit);
            mm_fp8_gemm(m->ex_y, n, y->sw2.w, y->sw2.s, m->gsm, DIM, INTER);
            for (int t = 0; t < n; t++) {
                float *ht = h + (size_t)t * DIM;
                const float *src = m->ex_y + (size_t)t * DIM;
                for (int i = 0; i < DIM; i++) ht[i] += src[i];
            }
        }

        for (int t = 0; t < n; t++) {
            float *ht = h + (size_t)t * DIM;
            for (int k = 0; k < c->n_activated; k++) {
                float w = m->ch_selw[(size_t)t * c->n_activated + k];
                const float *src = m->ch_contrib +
                    ((size_t)t * c->n_activated + k) * DIM;
                if (w == 0.0f) continue;
                for (int i = 0; i < DIM; i++) ht[i] += w * src[i];
            }
        }
        mm_read_join();   /* no-op unless a helper batch is still running */
        MM_NAN("ffn");
        /* Layer-granular progress. One line per window instead of one per
         * 2.5-minute chunk: `_done` only fires at chunk boundaries, and a
         * prompt that fits in a single chunk would otherwise print nothing at
         * all until it was already over. Decode never arms this (active=false),
         * so an n==1 step costs one branch. */
        mm_trace("out", L, h, (size_t)n * (size_t)DIM);
        if (n > 1) mm_prefill_progress_layer(start, n, L + 1, c->n_layers);
    }
mm_nan_done: ;
    if (m->spec_out) {
        /* Block verification: every position needs its own distribution. The
         * head is the most expensive matrix read in the model (vocab*dim bf16,
         * 2.4 GiB), so this is why a verified block is not free even when the
         * expert bytes are already spent. */
        for (int t = 0; t < n; t++) {
            mm_rmsnorm_gemma(m->h2, h + (size_t)t * DIM, m->final_norm, DIM,
                             c->rms_norm_eps);
            cwick_gemv_bf16(m->spec_out + (size_t)t * (size_t)c->vocab_size,
                            DIM, c->vocab_size, m->h2, m->head);
        }
        m->spec_out = NULL;
    } else {
        /* ---- final norm and the head, last position only ---- */
        const float *hlast = h + (size_t)(n - 1) * DIM;
        mm_rmsnorm_gemma(m->h2, hlast, m->final_norm, DIM, c->rms_norm_eps);
        if (nan_check) {
            for (int i = 0; i < DIM; i++)
                if (!isfinite(m->h2[i])) {
                    fprintf(stderr, "mm: non-finite in the final norm\n");
                    break;
                }
        }
        PROF_SLOT(7);   /* shared expert + combine */
        cwick_gemv_bf16(logits, DIM, c->vocab_size, m->h2, m->head);
    }
    if (prof_on) {
        /* Decode steps only: a prefill call is one big batch and would swamp
         * the numbers this exists to separate. */
        static uint64_t last_bytes;
        static uint64_t last_hits, last_miss;
        uint64_t bs = 0, rc = 0, ch = 0, cm = 0;
        if (m->stream && n == 1) cwick_stream_stats(m->stream, &bs, &rc, &ch, &cm);
        if (n >= 1) {
            double tot = 0.0;
            for (int i = 0; i < 10; i++) tot += prof[i];
            /* A prefill chunk is n positions in one call, so its counters are
             * divided by n: the interesting comparison is ms per token, since
             * that is what a decode step is. */
            const double d = (double)(n > 0 ? n : 1);
            fprintf(stderr, "mm prof%s %.0f ms%s: proj %.0f  prep %.0f  attn %.0f  "
                    "wo %.0f  route %.0f  experts %.0f  io %.0f  dense+share %.0f  "
                    "[gemv %.0f ms in %lld calls, read-ahead %.0f ms] "
                    "(ssd %.2f GiB, hit %.0f%%)\n",
                    n == 1 ? "" : " (prefill, per token)",
                    (tot + (mm_now_ms() - t_prof)) / d,
                    n == 1 ? "" : "", prof[1] / d, prof[2] / d, prof[3] / d,
                    prof[4] / d, prof[5] / d, prof[6] / d, prof[8] / d,
                    prof[7] / d + prof[9] / d,
                    g_mm_t_gemv / d, (long long)g_mm_n_gemv, g_mm_t_io / d,
                    cwick_gib((double)(bs - last_bytes)),
                    (ch + cm) > (last_hits + last_miss)
                        ? 100.0 * (double)(ch - last_hits) /
                              (double)((ch + cm) - (last_hits + last_miss))
                        : 0.0);
            last_bytes = bs; last_hits = ch; last_miss = cm;
            if (g_mm_n_gemm)
                fprintf(stderr, "mm groups: gate %lld up %lld down %lld "
                        "fallback %lld | %lld gemm calls (%.2f rows each), "
                        "%lld union entries, %lld single-row gemv\n",
                        (long long)g_mm_site[0], (long long)g_mm_site[1],
                        (long long)g_mm_site[2], (long long)g_mm_site[3],
                        (long long)g_mm_n_gemm,
                        (double)g_mm_n_rows / (double)g_mm_n_gemm,
                        (long long)g_mm_n_uni, (long long)g_mm_n_gemv);
            mm_fp8_stats_print();
            g_mm_t_gemv = 0; g_mm_t_io = 0; g_mm_n_gemv = 0;
            g_mm_n_gemm = 0; g_mm_n_rows = 0; g_mm_n_uni = 0;
            for (int i = 0; i < 4; i++) g_mm_site[i] = 0;
        }
        for (int i = 0; i < 10; i++) prof[i] = 0.0;
    }
    return 0;
#undef MM_NAN
}

int mm_model_forward(mm_model *m, const int32_t *ids, int n, int64_t start,
                     float *logits, cwick_error *err) {
    if (!m || !m->loaded) {
        cwick_error_set(err, CWICK_ERR_STATE, "mm: model is not open");
        return -1;
    }
    int done = 0;
    /* Debug lever: MM_CHUNK_CAP=1 turns off every n>1 path, which is how the
     * batched kernels are separated from the per-position ones. */
    static int cap = -2;
    if (cap == -2) {
        const char *e = getenv("MM_CHUNK_CAP");
        cap = (e && *e) ? atoi(e) : 0;
        if (cap <= 0 || cap > MM_CHUNK) cap = MM_CHUNK;
    }
    while (done < n) {
        int k = n - done;
        if (k > cap) k = cap;
        if (mm_forward_chunk(m, ids + done, k, start + done, logits, err) != 0)
            return -1;
        done += k;
        /* The pin cache is the only part of this backend's memory that can be
         * given back, so the check lives here rather than in the decode loop:
         * prefill fills the cache too, and it does so for minutes at a time.
         * Without it the cache grows to its cap and stays there, and an
         * overshoot lands in the compressor -- which keeps the hit rate but
         * turns every hit into a decompression fault. */
        if (m->stream) {
            cwick_stream_expert_decay(m->stream);
            cwick_stream_steer_memory(m->stream, m->rss_target, start + done - 1,
                                      "mem");
        }
    }
    m->pos = start + n;
    return 0;
}

/* Same contract as mm_model_forward, except that every position of the chunk
 * produces a distribution: `logits_all` is [n][vocab]. A block of candidates is
 * only verifiable if each position's own head output is available -- the
 * distribution that decides whether candidate i survives is the one the model
 * produced for position i-1, and re-running the block once per candidate would
 * cost exactly what speculation is supposed to save.
 *
 * The chunking here mirrors mm_model_forward, but the head is emitted per
 * chunk slice, so a block larger than MM_CHUNK still lands in the right place.
 * Callers use small blocks (<= 8), so this is one chunk in practice. */
int mm_model_forward_multi(mm_model *m, const int32_t *ids, int n, int64_t start,
                           float *logits_all, cwick_error *err) {
    if (!m || !m->loaded) {
        cwick_error_set(err, CWICK_ERR_STATE, "mm: model is not open");
        return -1;
    }
    if (n <= 0) return 0;
    int64_t V = m->cfg.vocab_size;
    int done = 0;
    while (done < n) {
        int k = n - done;
        if (k > MM_CHUNK) k = MM_CHUNK;
        m->spec_out = logits_all + (size_t)done * (size_t)V;
        int rc = mm_forward_chunk(m, ids + done, k, start + done,
                                  m->spec_out + (size_t)(k - 1) * (size_t)V, err);
        m->spec_out = NULL;
        if (rc != 0) return -1;
        done += k;
        if (m->stream) {
            cwick_stream_expert_decay(m->stream);
            cwick_stream_steer_memory(m->stream, m->rss_target, start + done - 1,
                                      "mem");
        }
    }
    m->pos = start + n;
    return 0;
}

/* ------------------------------------------------------------------ */
/* prefix snapshots                                                    */
/* ------------------------------------------------------------------ */

size_t mm_state_bytes(const mm_model *m, int64_t pos) {
    return mm_kv_bytes(m, (int)pos);
}
int mm_state_pack(const mm_model *m, int64_t pos, void *buf, size_t cap) {
    const mm_config *c = &m->cfg;
    size_t need = mm_state_bytes(m, pos);
    if (cap < need) return -1;
    uint8_t *o = (uint8_t *)buf;
    size_t kvsz = (size_t)pos * (size_t)(c->n_kv_heads * c->head_dim) *
                  sizeof(uint16_t);
    for (int L = 0; L < c->n_layers; L++) {
        memcpy(o, m->kcache + (size_t)L * c->max_seq_len *
                  (c->n_kv_heads * c->head_dim), kvsz);
        o += kvsz;
        memcpy(o, m->vcache + (size_t)L * c->max_seq_len *
                  (c->n_kv_heads * c->head_dim), kvsz);
        o += kvsz;
        size_t isz = (size_t)pos * c->index_dim * sizeof(uint16_t);
        memcpy(o, m->idxcache + (size_t)L * c->max_seq_len * c->index_dim, isz);
        o += isz;
    }
    return 0;
}
int mm_state_apply(mm_model *m, const void *buf, size_t len, int64_t *pos_out) {
    const mm_config *c = &m->cfg;
    /* The blob is layer-major and holds exactly the `pos` positions that were
     * captured, so `pos` is recovered from its length -- requiring the full
     * window instead (as this did) made every snapshot from a 248-token prompt
     * fail a check against 8160 MiB and the prefix cache never once applied.
     * That is the difference between a 126 s prefill and a 60 ms read on every
     * later turn. */
    size_t per_pos = (size_t)c->n_layers *
                     ((size_t)c->n_kv_heads * c->head_dim * 2 +
                      (size_t)c->index_dim) * sizeof(uint16_t);
    if (per_pos == 0 || len == 0 || len % per_pos != 0) return -1;
    int64_t pos = (int64_t)(len / per_pos);
    if (pos <= 0 || pos > c->max_seq_len) return -1;

    const uint8_t *i = (const uint8_t *)buf;
    size_t kvsz = (size_t)pos * (size_t)(c->n_kv_heads * c->head_dim) *
                  sizeof(uint16_t);
    size_t isz = (size_t)pos * (size_t)c->index_dim * sizeof(uint16_t);
    for (int L = 0; L < c->n_layers; L++) {
        memcpy(m->kcache + (size_t)L * c->max_seq_len *
               (c->n_kv_heads * c->head_dim), i, kvsz);
        i += kvsz;
        memcpy(m->vcache + (size_t)L * c->max_seq_len *
               (c->n_kv_heads * c->head_dim), i, kvsz);
        i += kvsz;
        memcpy(m->idxcache + (size_t)L * c->max_seq_len * c->index_dim, i, isz);
        i += isz;
    }
    m->pos = pos;
    *pos_out = pos;
    return 0;
}

/* Prefill progress. A 233 GiB checkpoint that routes every position reaches
 * most of the expert bank, so a 200-token prefill reads tens of GiB and takes
 * minutes: without a line here a stuck-looking request is indistinguishable
 * from a working one. One line per window, same knob as the decode loop. */
static struct {
    int64_t total, done, layers;
    int64_t layer;      /* 1-based layer of the chunk in flight (0 = none) */
    int64_t base;       /* absolute position of this prompt's first token */
    bool have_base;     /* base is only known once the first chunk starts */
    bool active;        /* inside a prefill: never print during decode */
    double t0, last;
} g_mm_pf;

/* CWICK_PROGRESS_MIN_MS, shared with the decode loop and the GLM path. */
static double mm_pf_window_ms(void) {
    static double cached = -1.0;
    if (cached < 0.0) {
        const char *e = getenv("CWICK_PROGRESS_MIN_MS");
        double v = e ? atof(e) : 0.0;
        cached = (v > 0.0) ? v : 5000.0;
    }
    return cached;
}

static double g_mm_now(void) { return mm_now_ms(); }
static double mm_now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1e3 + (double)tv.tv_usec / 1e3;
}

void mm_prefill_progress_begin(int64_t total_tokens, int layers) {
    g_mm_pf.total = total_tokens;
    g_mm_pf.done = 0;
    g_mm_pf.layers = layers;
    g_mm_pf.layer = 0;
    g_mm_pf.have_base = false;
    g_mm_pf.active = true;
    g_mm_pf.t0 = g_mm_pf.last = mm_now_ms();
    /* Also publish it: the log line only exists if this backend was asked to
     * print one, while a dashboard scraping /v1/metrics during a two-minute
     * prefill needs the gauge to move, or it draws 0/0 and looks hung. */
    cwick_metrics_set_prefill(0, total_tokens, 0, layers);
}

void mm_prefill_progress_done(int64_t done_tokens) {
    g_mm_pf.done = done_tokens;
    /* The layer counter belongs to the chunk that just finished, not to the
     * checkpoint: passing `layers` for both published "layer 60/60" from the
     * first second onward, which is what made /v1/metrics look finished while
     * a 30-minute prefill was still running. */
    cwick_metrics_set_prefill(done_tokens, g_mm_pf.total,
                              g_mm_pf.layer > 0 ? g_mm_pf.layer : 0,
                              g_mm_pf.layers);
    /* One line at each chunk boundary, plus the final one which is the only
     * summary a single-chunk prompt ever gets. The intra-chunk lines come from
     * mm_prefill_progress_layer(). */
    double now = mm_now_ms();
    if (done_tokens != g_mm_pf.total &&
        (now - g_mm_pf.last) < mm_pf_window_ms())
        return;
    g_mm_pf.last = now;
    double el = (now - g_mm_pf.t0) / 1000.0;
    double rate = done_tokens > 0 ? (double)done_tokens / (el > 0 ? el : 1) : 0.0;
    double eta = rate > 0 ? (double)(g_mm_pf.total - done_tokens) / rate : 0.0;
    CWICK_INFO("prefill", "%lld/%lld tokens (%.1f%%) — %.1f tok/s, "
               "eta %.0fs", (long long)done_tokens, (long long)g_mm_pf.total,
               g_mm_pf.total > 0
                   ? 100.0 * (double)done_tokens / (double)g_mm_pf.total : 0.0,
               rate, eta);
}

/* Called once per layer inside a prefill chunk.
 *
 * `mm_prefill_progress_done` only fires at chunk boundaries, and MM_CHUNK is
 * 512 positions that take ~2.5 minutes on this machine -- so a request used to
 * print one line every 170 s, and a prompt that fits in a single chunk printed
 * nothing at all until it was over. Layer granularity is what the GLM path
 * already had (`maybe_progress_at` with a layer counter): the token count
 * cannot advance inside a chunk, but the layer can, and that is the difference
 * between "working" and "hung" to whoever is watching.
 *
 * `start` is the absolute position of the chunk's first token; `n` its size.
 * The token count reported is the one from `start` (no invented precision),
 * while rate and eta use the fraction of the chunk that is done, so the ETA
 * counts down between chunk boundaries instead of stepping. */
void mm_prefill_progress_layer(int64_t start, int64_t n, int layer,
                               int nlayer) {
    if (!g_mm_pf.active || nlayer <= 0) return;
    if (!g_mm_pf.have_base) {          /* first chunk of this prompt */
        g_mm_pf.base = start;
        g_mm_pf.have_base = true;
    }
    g_mm_pf.layer = layer;
    double now = mm_now_ms();
    const bool every_nth = (layer % 8) == 0 || layer == nlayer;
    if (!every_nth && (now - g_mm_pf.last) < mm_pf_window_ms()) return;
    g_mm_pf.last = now;
    /* Progress *inside* the chunk. A prompt that fits in one chunk -- 201 of
     * the 512 positions here -- has `done == 0` for its entire duration, so
     * reporting only completed tokens says "0/201 (0.0%)" for two minutes and
     * then jumps to 100%. The layers of the chunk in flight are real work, so
     * they are counted: layer/nlayer of this chunk's tokens are done. The `~`
     * marks the estimate; at the last layer it becomes exact, and `_done`
     * prints the exact number at the boundary. */
    const int64_t done = start - g_mm_pf.base;
    const int64_t est = done + (n * layer) / nlayer;
    const bool exact = layer >= nlayer;
    cwick_metrics_set_prefill(est, g_mm_pf.total, layer, nlayer);
    double el = (now - g_mm_pf.t0) / 1000.0;
    double rate = el > 0.0 ? (double)est / el : 0.0;
    double eta = rate > 0.0 ? ((double)g_mm_pf.total - (double)est) / rate : 0.0;
    if (eta < 0.0) eta = 0.0;
    CWICK_INFO("prefill", "%s%lld/%lld tokens (%.1f%%), layer %d/%d — %.1f tok/s, "
               "eta %.0fs", exact ? "" : "~", (long long)est,
               (long long)g_mm_pf.total,
               g_mm_pf.total > 0
                   ? 100.0 * (double)est / (double)g_mm_pf.total : 0.0,
               layer, nlayer, rate, eta);
}

void mm_prefill_progress_end(void) {
    g_mm_pf.active = false;
    g_mm_pf.layer = 0;
    g_mm_pf.have_base = false;
    cwick_metrics_set_prefill(0, 0, 0, 0);
}

/* ------------------------------------------------------------------ */
/* kernel micro-benchmarks (tools/mm_run --bench)                      */
/* ------------------------------------------------------------------ */

double mm_bench_fp4(int64_t n, int64_t k, int iters, double *gmacs_out) {
    if (n <= 0 || k <= 0 || iters <= 0) return 0.0;
    const size_t wb = (size_t)(n * k / 2);
    const size_t sb = (size_t)(n * (k / 16));
    uint8_t *w = (uint8_t *)cwick_xmalloc(wb);
    uint8_t *s = (uint8_t *)cwick_xmalloc(sb);
    float *x = (float *)cwick_xmalloc(sizeof(float) * (size_t)k);
    float *ev = (float *)cwick_xmalloc(sizeof(float) * (size_t)(k / 2 + 8));
    float *od = (float *)cwick_xmalloc(sizeof(float) * (size_t)(k / 2 + 8));
    float *out = (float *)cwick_xmalloc(sizeof(float) * (size_t)n);
    /* Bytes with a plausible mix of nibbles; the kernel does not branch on
     * values, so any pattern measures the same path. */
    for (size_t i = 0; i < wb; i++) w[i] = (uint8_t)(i * 37u + 11u);
    for (size_t i = 0; i < sb; i++) s[i] = 0x3C;   /* e4m3 1.0 */
    for (int64_t i = 0; i < k; i++) x[i] = (float)((i % 17) - 8) * 0.0625f;
    double t0 = mm_now_ms();
    for (int it = 0; it < iters; it++)
        mm_fp4_gemv(out, w, s, 1.0f / 8192.0f, x, n, k, ev, od);
    double ms = (mm_now_ms() - t0) / (double)iters;
    if (gmacs_out)
        *gmacs_out = ms > 0 ? (double)n * (double)k / (ms * 1e6) : 0.0;
    free(w); free(s); free(x); free(ev); free(od); free(out);
    return ms;
}

/* Time the FP8 dense GEMM on a real shape, so "is it the kernel or is it the
 * pages" has an answer that does not depend on a 100-second model run. */
void mm_bench_fp8(int64_t N, int64_t K, int64_t n, int iters) {
    mm_build_e4m3();
    uint8_t *w = (uint8_t *)cwick_xmalloc((size_t)(N * K));
    uint8_t *s = (uint8_t *)cwick_xmalloc((size_t)(N * (K / 32)));
    float *x = (float *)cwick_xmalloc(sizeof(float) * (size_t)(K * n));
    float *o = (float *)cwick_xmalloc(sizeof(float) * (size_t)(N * n));
    for (int64_t i = 0; i < N * K; i++) w[i] = (uint8_t)((i * 37u + 11u) & 0x7F);
    for (int64_t i = 0; i < N * (K / 32); i++) s[i] = 0x7B;
    for (int64_t i = 0; i < K * n; i++) x[i] = (float)((i % 17) - 8) * 0.03125f;
    mm_fp8_gemm(o, n, w, s, x, N, K);          /* warm */

    /* Independent check: a naive triple loop over a small shape, including
     * tails on all three axes (N=30 rows, n=5 activations, K=96 = 3 blocks). */
    {
        const int64_t cN = 30, cK = 96, cn = 5;
        uint8_t *cw = (uint8_t *)cwick_xmalloc((size_t)(cN * cK));
        uint8_t *cs = (uint8_t *)cwick_xmalloc((size_t)(cN * (cK / 32)));
        float *cx = (float *)cwick_xmalloc(sizeof(float) * (size_t)(cK * cn));
        float *co = (float *)cwick_xmalloc(sizeof(float) * (size_t)(cN * cn));
        for (int64_t i = 0; i < cN * cK; i++) cw[i] = (uint8_t)((i * 91u + 7u) & 0x7F);
        for (int64_t i = 0; i < cN * (cK / 32); i++) cs[i] = (uint8_t)(120 + (i % 9));
        for (int64_t i = 0; i < cK * cn; i++) cx[i] = (float)((i % 13) - 6) * 0.125f;
        mm_fp8_gemm(co, cn, cw, cs, cx, cN, cK);
        double worst = 0.0, mag = 0.0;
        for (int64_t t = 0; t < cn; t++)
            for (int64_t c = 0; c < cN; c++) {
                double ref = 0.0;
                for (int64_t j = 0; j < cK; j++)
                    ref += (double)ds_e4m3_to_f32(cw[c * cK + j]) *
                           ldexpf(1.0, (int)cs[c * (cK / 32) + j / 32] - 127) *
                           (double)cx[t * cK + j];
                double d = fabs(ref - (double)co[t * cN + c]);
                if (d > worst) worst = d;
                if (fabs(ref) > mag) mag = fabs(ref);
            }
        fprintf(stderr, "fp8 gemm check [%lldx%lld n=%lld]: worst |err| %.3g of "
                "max |value| %.3g = %.2g relative\n", (long long)cN,
                (long long)cK, (long long)cn, worst, mag, worst / mag);
        free(cw); free(cs); free(cx); free(co);
    }
    /* The one check that actually constrains speculation: every position of a
     * block has to come out bit-identical to what it would have got on its own.
     * Synthetic rows with a full mantissa, because small multiples of a power
     * of two round the same way in any order and would pass either way. */
    {
        const int64_t cn = n > 32 ? 32 : (n > 0 ? n : 1);
        float *cx = (float *)cwick_xmalloc(sizeof(float) * (size_t)K * cn);
        float *co = (float *)cwick_xmalloc(sizeof(float) * (size_t)N * cn);
        float *ones = (float *)cwick_xmalloc(sizeof(float) * (size_t)N);
        uint64_t st = 0x123456789ABCDEFull;
        for (int64_t i = 0; i < K * cn; i++) {
            st ^= st >> 12; st *= 0x2545F4914F6CDD1Dull; st ^= st >> 27;
            uint32_t m = (uint32_t)(st >> 40) | 0x3F800000u;   /* [1,2) */
            float v; memcpy(&v, &m, sizeof v);
            cx[i] = (i & 1) ? -v : v;
        }
        /* Scales that are not the same for every block, so a kernel that
         * applies them in the wrong place cannot agree by accident. */
        for (int64_t i = 0; i < N * (K / 32); i++) w[i] = (uint8_t)((i * 91u + 7u) & 0x7F);
        for (int64_t i = 0; i < N * (K / 32); i++) s[i] = (uint8_t)(120 + (i % 9));
        mm_fp8_gemm(co, cn, w, s, cx, N, K);
        int64_t neq = 0;
        double worst = 0.0;
        for (int64_t t = 0; t < cn; t++) {
            mm_fp8_gemm(ones, 1, w, s, cx + (size_t)t * (size_t)K, N, K);
            for (int64_t c = 0; c < N; c++) {
                if (ones[c] != co[(size_t)t * (size_t)N + c]) neq++;
                double d = fabs((double)ones[c] - (double)co[(size_t)t * (size_t)N + c]);
                if (d > worst) worst = d;
            }
        }
        fprintf(stderr, "fp8 exact check [%lldx%lld] n=%lld: %lld of %lld "
                "values differ, worst |diff| %.3g -- %s\n",
                (long long)N, (long long)K, (long long)cn, (long long)neq,
                (long long)(N * cn), worst,
                neq == 0 ? "a block is bit-identical to step-at-a-time"
                         : "NOT bit-identical: block verification would "
                           "change the answer");
        free(cx); free(co); free(ones);
    }
    double t0 = mm_now_ms();
    for (int it = 0; it < iters; it++) mm_fp8_gemm(o, n, w, s, x, N, K);
    double ms = (mm_now_ms() - t0) / iters;
    double mac = (double)N * (double)K * (double)n / 1e9;
    fprintf(stderr, "fp8 gemm [%lldx%lld] n=%lld: %.1f ms  %.0f GMAC/s  "
            "%.1f GB/s of weights  (%.2f GiB weights, %.0f MiB activations)\n",
            (long long)N, (long long)K, (long long)n, ms, mac / (ms * 1e-3),
            (double)(N * K) / (ms * 1e6),
            (double)(N * K) / (1024.0 * 1024 * 1024),
            (double)(K * n * 4) / (1024.0 * 1024));
    free(w); free(s); free(x); free(o);
}

/* Cross-check the grouped kernel against the single-row one on synthetic data.
 * Shapes and packing are identical, so any difference is arithmetic. */
void mm_bench_gemm_check(int64_t N, int64_t K) {
    mm_build_e4m3();
    uint8_t *w = (uint8_t *)cwick_xmalloc((size_t)(N * K / 2));
    uint8_t *sc = (uint8_t *)cwick_xmalloc((size_t)(N * (K / 16)));
    float *x = (float *)cwick_xmalloc(sizeof(float) * (size_t)K * 2);
    float *ev = (float *)cwick_xmalloc(sizeof(float) * (size_t)(K / 2 + 8));
    float *od = (float *)cwick_xmalloc(sizeof(float) * (size_t)(K / 2 + 8));
    float *o1 = (float *)cwick_xmalloc(sizeof(float) * (size_t)2 * (size_t)N);
    float *o2 = (float *)cwick_xmalloc(sizeof(float) * (size_t)2 * (size_t)N);
    for (int64_t i = 0; i < N * K / 2; i++) w[i] = (uint8_t)(i * 37u + 11u);
    for (int64_t i = 0; i < N * (K / 16); i++) sc[i] = 0x3C;
    /* Values with a full mantissa, not small multiples of a power of two.
     * The earlier pattern was exact in every intermediate step, so it could
     * not see the difference between a fused and a split product hidden in
     * this comparison -- which is exactly the difference that made a whole
     * block disagree with a step-at-a-time run on the real checkpoint. */
    {
        uint64_t s = 0x9E3779B97F4A7C15ull;
        for (int64_t i = 0; i < 2 * K; i++) {
            s ^= s >> 12; s *= 0x2545F4914F6CDD1Dull; s ^= s >> 27;
            uint32_t m = (uint32_t)(s >> 40) | 0x3F800000u;   /* [1,2) */
            float v;
            memcpy(&v, &m, sizeof v);
            x[i] = (i & 1) ? -v : v;
        }
    }
    mm_fp4_gemv(o1, w, sc, 1.0f / 8192.0f, x, N, K, ev, od);
    float *xi = (float *)cwick_xmalloc(sizeof(float) * (size_t)K * 8);
    for (int64_t t = 0; t < 8; t++)
        mm_deinterleave(xi + t * K, x + (t % 2) * K, K);
    mm_fp4_gemm(o2, 2, w, sc, 1.0f / 8192.0f, xi, N, K);
    double worst = 0.0;
    int64_t neq = 0;
    for (int64_t c = 0; c < N; c++) {
        double d = fabs((double)o1[c] - (double)o2[c]);
        if (d > worst) worst = d;
        if (o1[c] != o2[c]) neq++;
    }
    fprintf(stderr, "gemm check [%lldx%lld]: worst |diff| %.6g, %lld of %lld "
            "rows not bit-identical (gemv %.9g, gemm %.9g)\n",
            (long long)N, (long long)K, worst, (long long)neq, (long long)N,
            (double)o1[0], (double)o2[0]);
    fprintf(stderr, "  %s\n", neq == 0
            ? "bit-identical: a grouped block reproduces the per-row kernel "
              "exactly, so block verification cannot disagree with decoding"
            : "NOT bit-identical: the grouped kernel rounds differently from "
              "the per-row one");

    /* Throughput at the group sizes prefill actually sees: an expert is asked
     * for ~4 positions of a 128-position chunk. `n x gemv` is the baseline the
     * grouped kernel has to beat, because that is what it replaces. */
    for (int64_t nn = 1; nn <= 8; nn *= 2) {
        const int it = 3;
        double t0 = mm_now_ms();
        for (int it2 = 0; it2 < it; it2++)
            for (int64_t t = 0; t < nn; t++)
                mm_fp4_gemv(o1 + t * N, w, sc, 1.0f / 8192.0f, x + t * K, N, K,
                            ev, od);
        double tg = (mm_now_ms() - t0) / it;
        t0 = mm_now_ms();
        for (int it2 = 0; it2 < it; it2++)
            mm_fp4_gemm(o2, nn, w, sc, 1.0f / 8192.0f, xi, N, K);
        double tm = (mm_now_ms() - t0) / it;
        double mac = (double)N * (double)K * (double)nn / 1e9;
        fprintf(stderr, "  n=%lld: %lld gemv %.1f ms (%.0f GMAC/s) | gemm %.1f ms "
                "(%.0f GMAC/s) = %.2fx\n", (long long)nn, (long long)nn, tg,
                mac / (tg * 1e-3), tm, mac / (tm * 1e-3), tg / tm);
    }
    free(w); free(sc); free(x); free(xi); free(ev); free(od); free(o1); free(o2);
}

double mm_bench_bf16(int64_t n, int64_t k, int iters, double *gmacs_out) {
    if (n <= 0 || k <= 0 || iters <= 0) return 0.0;
    uint16_t *w = (uint16_t *)cwick_xmalloc(sizeof(uint16_t) *
                                            (size_t)(n * k));
    float *x = (float *)cwick_xmalloc(sizeof(float) * (size_t)k);
    float *out = (float *)cwick_xmalloc(sizeof(float) * (size_t)n);
    for (int64_t i = 0; i < n * k; i++) w[i] = f_to_bf16(0.0005f * (float)(i % 7));
    for (int64_t i = 0; i < k; i++) x[i] = (float)((i % 13) - 6) * 0.0625f;
    double t0 = mm_now_ms();
    for (int it = 0; it < iters; it++) cwick_gemv_bf16(out, k, n, x, w);
    double ms = (mm_now_ms() - t0) / (double)iters;
    if (gmacs_out)
        *gmacs_out = ms > 0 ? (double)n * (double)k / (ms * 1e6) : 0.0;
    free(w); free(x); free(out);
    return ms;
}
