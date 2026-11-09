/*
 * ds_quant.c — the numeric core of the DeepSeek port: format conversions,
 * the activation quantizer, and GEMVs over quantized weights.
 *
 * Every routine here mirrors the reference implementation exactly, including
 * where it is lossy: activations are quantized to e4m3 with a per-32 power of
 * two scale before they ever touch a weight, and the compressed KV / indexer
 * keys are quantized and immediately dequantized back. Skipping either step
 * changes the numbers, not just the speed.
 */
#include "ds.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "cwick.h"   /* cwick_pool_for: the GEMV row split goes through it */
#include "util.h"    /* cwick_env_true: CWICK_DS_EXACT */

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

/* ------------------------------------------------------------------ */
/* scalar conversions                                                   */
/* ------------------------------------------------------------------ */

/* e2m1: sign in bit 3, then 2 magnitude bits and a half bit.
 *   0 .5 1 1.5 2 3 4 6 / 0 -.5 -1 -1.5 -2 -3 -4 -6 */
static const float FP4_E2M1[16] = {
    0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
    -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f,
};

/* The same tables as flat arrays, because the GEMV inner loops convert one
 * value per MAC: a load from a 1 KiB table that stays in L1 beats a branchy
 * decode by roughly an order of magnitude, and these loops run ~1e9 times per
 * token. Filled once at first use. */
float DS_FP4_LUT[16];
float DS_E4M3_LUT[256];

/* One byte of a packed FP4 row carries two e2m1 values. Both come out of a
 * single 8-byte load instead of two gathers into the 16-entry table: the
 * inner loop runs ~1e9 times per token, and two independent L1 gathers per
 * multiply-add was the whole reason a 12-core machine sat at 0.45 MAC per
 * cycle. */
typedef struct { float lo, hi; } ds_fp4_pair;
static ds_fp4_pair DS_FP4_PAIR[256];

static int luts_ready = 0;

static void build_luts(void) {
    if (luts_ready) return;
    for (int i = 0; i < 16; i++) DS_FP4_LUT[i] = FP4_E2M1[i];
    for (int b = 0; b < 256; b++) {
        DS_FP4_PAIR[b].lo = FP4_E2M1[b & 0x0F];
        DS_FP4_PAIR[b].hi = FP4_E2M1[b >> 4];
    }
    for (int b = 0; b < 256; b++) {
        uint32_t sign = (uint32_t)(b >> 7) & 1u;
        uint32_t e = ((uint32_t)b >> 3) & 0xFu;
        uint32_t m = (uint32_t)b & 0x7u;
        float v;
        if (e == 0) {
            v = ldexpf((float)m, -9);
        } else if (e == 15 && m == 7) {
            v = 0.0f / 0.0f;
        } else {
            v = ldexpf((float)(8u + m), (int)e - 10);
        }
        DS_E4M3_LUT[b] = sign ? -v : v;
    }
    luts_ready = 1;
}

float ds_fp4_to_f32(uint8_t nibble) { build_luts(); return DS_FP4_LUT[nibble & 0x0F]; }

float ds_e8m0_to_f32(uint8_t b) {
    /* 2^(b-127), built the way the reference's fast_pow2 does: reinterpret
     * (b) << 23. That makes byte 0 read as +0.0 rather than 2^-127, which is
     * what a raw e8m0 decode would say. The two can only differ for byte 0,
     * and byte 0 means the block's amax was below 2^-127*448, i.e. every
     * value in it is zero - so the product is zero either way. Measured on
     * the released checkpoint: scale bytes run 0x72..0x7b, never 0x00.
     * 0xFF would be NaN, which no trained scale ever is; returning a huge
     * finite value keeps a corrupt scale visible instead of silent. */
    if (b == 0xFF) return 0.0f / 0.0f;
    union { uint32_t u; float f; } c;
    c.u = (uint32_t)b << 23;
    return c.f;
}

float ds_e4m3_to_f32(uint8_t b) { build_luts(); return DS_E4M3_LUT[b]; }

float ds_bf16_to_f32(uint16_t b) {
    union { uint32_t u; float f; } c;
    c.u = (uint32_t)b << 16;
    return c.f;
}

/* Round-to-nearest-even into e4m3fn. The caller guarantees |v| <= 448. */
uint8_t ds_f32_to_e4m3(float v) {
    if (v == 0.0f) return 0;
    union { float f; uint32_t u; } c;
    c.f = v;
    uint32_t sign = c.u >> 31;
    int32_t exp = (int32_t)((c.u >> 23) & 0xFFu) - 127;
    uint32_t m24 = (1u << 23) | (c.u & 0x7FFFFFu);

    if (exp >= -6) {
        uint32_t keep = m24 >> 20;               /* implicit 1 + 3 explicit */
        uint32_t rem = m24 & 0xFFFFFu;
        const uint32_t half = 0x80000u;
        if (rem > half || (rem == half && (keep & 1u))) keep++;
        int32_t e = exp;
        if (keep == 16u) { keep >>= 1; e++; }    /* carried out of the mantissa */
        if (e > 8) { keep = 14u; e = 8; }        /* saturate: caller clamps first */
        return (uint8_t)((sign << 7) | ((uint32_t)(e + 7) << 3) | (keep & 7u));
    }
    /* subnormal grid: 2^-9 */
    if (exp < -10) return (uint8_t)(sign << 7);
    int32_t sh = 14 - exp;                       /* 14..24 */
    uint32_t mm = m24 >> sh;
    uint32_t rem = m24 & ((1u << sh) - 1u);
    uint32_t half = 1u << (sh - 1);
    if (rem > half || (rem == half && (mm & 1u))) mm++;
    return (uint8_t)((sign << 7) | (mm & 7u));
}

/* ceil(log2(x)) from the bit pattern, exactly like the reference's
 * fast_log2_ceil: exponent plus "was there anything below the leading bit". */
static int32_t log2_ceil_bits(float x) {
    union { float f; uint32_t u; } c;
    c.f = x;
    int32_t e = (int32_t)((c.u >> 23) & 0xFFu) - 127;
    return e + ((c.u & 0x7FFFFFu) != 0 ? 1 : 0);
}

uint8_t ds_f32_to_e8m0_exponent(float amax) {
    return (uint8_t)(log2_ceil_bits(amax * (1.0f / 448.0f)) + 127);
}

uint8_t ds_act_quant_block(const float *x, uint8_t *out_fp8) {
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        float a = x[i] < 0 ? -x[i] : x[i];
        if (a > amax) amax = a;
    }
    if (amax < 1e-4f) amax = 1e-4f;                 /* keep a zero block finite */
    int32_t e = log2_ceil_bits(amax * (1.0f / 448.0f));
    float s = ldexpf(1.0f, e);
    for (int i = 0; i < 32; i++) {
        float q = x[i] / s;
        if (q > 448.0f) q = 448.0f;
        if (q < -448.0f) q = -448.0f;
        out_fp8[i] = ds_f32_to_e4m3(q);
    }
    return (uint8_t)(e + 127);
}

/* ------------------------------------------------------------------ */
/* GEMVs                                                               */
/* ------------------------------------------------------------------ */

void ds_fp4_row(const uint8_t *packed, const uint8_t *scale, int64_t K, float *out) {
    int64_t nb = K / 32;
    for (int64_t b = 0; b < nb; b++) {
        float sb = ds_e8m0_to_f32(scale[b]);
        const uint8_t *p = packed + b * 16;
        for (int j = 0; j < 16; j++) {
            out[b * 32 + 2 * j] = ds_fp4_to_f32(p[j] & 0x0Fu) * sb;
            out[b * 32 + 2 * j + 1] = ds_fp4_to_f32(p[j] >> 4) * sb;
        }
    }
}

/* ------------------------------------------------------------------ */
/* quantized GEMV, split across the worker pool                        */
/*                                                                     */
/* These two are the hot loops of the port: 384 routed experts x 3      */
/* matrices, six routing slots per token, so one decode step runs       */
/* thousands of them. Written as plain loops they used exactly one      */
/* core -- sampling the server showed the other 11 performance cores    */
/* parked in __psynch_cvwait for an entire token.                       */
/*                                                                      */
/* The split is per OUTPUT ROW, and that is the only split that is      */
/* free: every row accumulates from zero over its own row of weights,   */
/* so a worker owning rows [n0,n1) runs the identical float sequence    */
/* the serial loop did. Bit-for-bit agreement with the independent      */
/* oracle is what makes this safe -- any reduction across rows would    */
/* not be.                                                              */
/* ------------------------------------------------------------------ */

/* Below this many rows the fork-join handshake costs more than the work;
 * these are the small projections (the head, the short latent paths). */
#define DS_GEMV_PARALLEL_MIN 128

/* Widest activation any of these GEMVs sees: wo_b takes the block-diagonal
 * output projection, K = o_groups * o_lora_rank = 8192. The dequantized copy
 * lives on the caller's stack (32 KiB) and is shared by every output row, so
 * a K-element table lookup happens once per GEMV instead of once per row.
 * Larger K than this falls back to a heap buffer rather than to a slow path:
 * the fast kernel is the only kernel. */
#define DS_ACT_MAX 8192

/* How many output rows the inner loop walks at once. Four gives each core four
 * independent accumulation chains, which is exactly what it takes to fill the
 * four-cycle fp32 add latency; wider groups start to spill the row pointers and
 * lose the activation block out of registers. DS_GEMV_ROWS=1 forces the plain
 * one-row loop, for measuring the effect or for a machine where four streams
 * thrash the cache. */
static int ds_gemv_rows = -1;

static int ds_gemv_row_group(void) {
    if (ds_gemv_rows < 0) {
        const char *s = getenv("DS_GEMV_ROWS");
        int v = s ? atoi(s) : 4;
        ds_gemv_rows = (v == 1) ? 1 : 4;
    }
    return ds_gemv_rows;
}

typedef struct {
    const uint8_t *a, *a_scale;
    const uint8_t *w, *w_scale;
    float *out;
    const float *af;     /* a[] dequantized once */
    const float *asf;    /* a_scale[] dequantized once */
    int64_t K, nb;
} ds_gemv_arg;

/* Fill `af` (K floats) and `asf` (K/32 floats) from the quantized activation.
 * Nothing is approximated: the values are the same table entries the inner
 * loop used to fetch per row. */
static void ds_gemv_dequant_act(const uint8_t *a, const uint8_t *a_scale,
                                int64_t K, float *af, float *asf) {
    const float *E4 = DS_E4M3_LUT;
    for (int64_t k = 0; k < K; k++) af[k] = E4[a[k]];
    int64_t nb = K / 32;
    for (int64_t b = 0; b < nb; b++) asf[b] = ds_e8m0_to_f32(a_scale[b]);
}

/* One output row, summed in the reference's order: 32 products per block into
 * `part`, then the two block scales, then into `acc`. */
static float ds_fp4_dot(const ds_fp4_pair *pair, const uint8_t *row,
                        const uint8_t *rs, const float *af, const float *asf,
                        int64_t nb) {
    float acc = 0.0f;
    for (int64_t b = 0; b < nb; b++) {
        const uint8_t *p = row + b * 16;
        const float *av = af + b * 32;
        float part = 0.0f;
        for (int j = 0; j < 16; j++) {
            ds_fp4_pair wv = pair[p[j]];
            part += wv.lo * av[2 * j];
            part += wv.hi * av[2 * j + 1];
        }
        acc += (part * asf[b]) * ds_e8m0_to_f32(rs[b]);
    }
    return acc;
}

/* Four rows at once. Each row keeps its own serial accumulation, so the
 * arithmetic is unchanged bit for bit -- what changes is that four independent
 * dependency chains now run side by side. A single row is 32 dependent fp32
 * adds per block, and an add retires in one cycle but only produces its result
 * four cycles later, so one row alone spends three quarters of every cycle
 * waiting. Four rows fill those slots. */
#define DS_FP4_STEP(R, P)                     \
    do {                                      \
        ds_fp4_pair wv = pair[(R)[b * 16 + j]]; \
        (P) += wv.lo * av[2 * j];             \
        (P) += wv.hi * av[2 * j + 1];         \
    } while (0)

static void ds_fp4_gemv_range(void *argp, int64_t n0, int64_t n1) {
    ds_gemv_arg *g = (ds_gemv_arg *)argp;
    const ds_fp4_pair *pair = DS_FP4_PAIR;
    const int64_t K = g->K, nb = g->nb, half = K / 2;
    const float *af = g->af, *asf = g->asf;
    int64_t n = n0;

    if (ds_gemv_row_group() == 1) {
        for (; n < n1; n++)
            g->out[n] = ds_fp4_dot(pair, g->w + n * half,
                                   g->w_scale + n * nb, af, asf, nb);
        return;
    }

    for (; n + 3 < n1; n += 4) {
        const uint8_t *r0 = g->w + (n + 0) * half;
        const uint8_t *r1 = g->w + (n + 1) * half;
        const uint8_t *r2 = g->w + (n + 2) * half;
        const uint8_t *r3 = g->w + (n + 3) * half;
        const uint8_t *s0 = g->w_scale + (n + 0) * nb;
        const uint8_t *s1 = g->w_scale + (n + 1) * nb;
        const uint8_t *s2 = g->w_scale + (n + 2) * nb;
        const uint8_t *s3 = g->w_scale + (n + 3) * nb;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int64_t b = 0; b < nb; b++) {
            const float *av = af + b * 32;
            float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
            for (int j = 0; j < 16; j++) {
                DS_FP4_STEP(r0, p0);
                DS_FP4_STEP(r1, p1);
                DS_FP4_STEP(r2, p2);
                DS_FP4_STEP(r3, p3);
            }
            a0 += (p0 * asf[b]) * ds_e8m0_to_f32(s0[b]);
            a1 += (p1 * asf[b]) * ds_e8m0_to_f32(s1[b]);
            a2 += (p2 * asf[b]) * ds_e8m0_to_f32(s2[b]);
            a3 += (p3 * asf[b]) * ds_e8m0_to_f32(s3[b]);
        }
        g->out[n + 0] = a0;
        g->out[n + 1] = a1;
        g->out[n + 2] = a2;
        g->out[n + 3] = a3;
    }
    for (; n < n1; n++)
        g->out[n] = ds_fp4_dot(pair, g->w + n * half, g->w_scale + n * nb,
                               af, asf, nb);
}

/* Dequantize the activation into `stack` when it fits, the heap otherwise. */
static float *ds_act_buffer(float *stack, int64_t n, int64_t cap) {
    if (n <= cap) return stack;
    return malloc(sizeof(float) * (size_t)n);
}

/* CWICK_DS_EXACT=1 pins the reference-order scalar kernels: one accumulator per
 * row, K additions in sequence. The vector paths keep every product exact (an
 * e2m1 weight has one mantissa bit, an e4m3 activation three, so a product
 * needs at most four) and still accumulate in fp32; what changes is that four
 * partial sums are combined at the end instead of one long chain. That is worth
 * 2-4x in bandwidth, and it is not bit-identical, so the layer oracle runs with
 * this on. */
static int ds_exact_mode(void) {
    static int cached = -1;
    if (cached < 0) cached = cwick_env_true("CWICK_DS_EXACT") ? 1 : 0;
    return cached;
}

#if defined(__aarch64__)

/* e2m1 -> fp16 through one 16-entry table of HIGH bytes. All sixteen values
 * have a zero low byte in fp16 (0, .5, 1, 1.5, 2, 3, 4, 6 and their negatives
 * sit on exponent boundaries), so a nibble becomes a half with one gather plus
 * one shift -- no per-value arithmetic, no second gather, and the sign rides in
 * the top bit of the byte. The multiply stays exact; see ds_exact_mode(). */
static const uint8_t DS_FP4_HI[16] = {
    0x00, 0x38, 0x3C, 0x3E, 0x40, 0x42, 0x44, 0x46,
    0x80, 0xB8, 0xBC, 0xBE, 0xC0, 0xC2, 0xC4, 0xC6,
};

/* The activation is shared by every output row, so it is de-interleaved once
 * per call: nibble 2j of a byte multiplies av[2j] and nibble 2j+1 multiplies
 * av[2j+1], so splitting the two lets each half run as its own 16-lane group
 * with no interleaving in the inner loop. */
typedef struct {
    const uint8_t *w, *w_scale;
    float *out;
    const float *asf;
    const float *af_even, *af_odd;   /* [K/2] each */
    int64_t K, nb;
} ds_fp4_arg_neon;

static void ds_fp4_gemv_range_neon(void *argp, int64_t n0, int64_t n1) {
    ds_fp4_arg_neon *g = (ds_fp4_arg_neon *)argp;
    const int64_t nb = g->nb, half = g->K / 2;
    const float *asf = g->asf, *ev = g->af_even, *od = g->af_odd;
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    const uint8x16_t tbl = vld1q_u8(DS_FP4_HI);

/* One 32-element block of one row: 16 packed bytes -> four fp32 lane sums ->
 * the block's contribution. `part` is a plain float; the caller decides the
 * order it is folded into the row accumulator, which is what keeps this
 * bit-identical to the scalar kernel. */
#define DS_FP4_BLOCK(VV, EE, OO, PART)                                        \
    do {                                                                      \
        uint8x16_t lo_b = vqtbl1q_u8(tbl, vandq_u8((VV), mask));              \
        uint8x16_t hi_b = vqtbl1q_u8(tbl, vshrq_n_u8((VV), 4));               \
                                                                              \
        /* low bytes are zero by construction: shifting the high byte up      \
         * eight bits *is* the fp16 bit pattern */                            \
        uint16x8_t lo_a = vshll_n_u8(vget_low_u8(lo_b), 8);                   \
        uint16x8_t lo_c = vshll_n_u8(vget_high_u8(lo_b), 8);                  \
        uint16x8_t hi_a = vshll_n_u8(vget_low_u8(hi_b), 8);                   \
        uint16x8_t hi_c = vshll_n_u8(vget_high_u8(hi_b), 8);                  \
                                                                              \
        float32x4_t s0 = vmulq_f32(                                           \
            vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(lo_a))),           \
            vld1q_f32(EE));                                                   \
        float32x4_t s1 = vmulq_f32(                                           \
            vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(lo_a))),          \
            vld1q_f32((EE) + 4));                                             \
        float32x4_t s2 = vmulq_f32(                                           \
            vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(lo_c))),           \
            vld1q_f32((EE) + 8));                                             \
        float32x4_t s3 = vmulq_f32(                                           \
            vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(lo_c))),          \
            vld1q_f32((EE) + 12));                                            \
        s0 = vfmaq_f32(s0, vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(hi_a))),\
                       vld1q_f32(OO));                                        \
        s1 = vfmaq_f32(s1, vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(hi_a))),\
                       vld1q_f32((OO) + 4));                                  \
        s2 = vfmaq_f32(s2, vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(hi_c))),\
                       vld1q_f32((OO) + 8));                                  \
        s3 = vfmaq_f32(s3, vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(hi_c))),\
                       vld1q_f32((OO) + 12));                                 \
                                                                              \
        (PART) = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3))); \
    } while (0)

    for (int64_t n = n0; n < n1; n++) {
        const uint8_t *row = g->w + n * half;
        const uint8_t *rs = g->w_scale + n * nb;
        float acc = 0.0f;
        for (int64_t b = 0; b < nb; b++) {
            const float *e = ev + b * 16, *o = od + b * 16;
            float part;
            DS_FP4_BLOCK(vld1q_u8(row + b * 16), e, o, part);
            acc += (part * asf[b]) * ds_e8m0_to_f32(rs[b]);
        }
        g->out[n] = acc;
    }
#undef DS_FP4_BLOCK
}
#endif /* __aarch64__ */

/* ------------------------------------------------------------------ */
/* splitting a GEMV between the CPU pool and the GPU                    */
/*                                                                      */
/* A decode step reads ~18 GB of weights and every byte of it is read    */
/* exactly once, so the step's floor is "bytes / achievable bandwidth".  */
/* The CPU pool reaches ~48 GB/s on the e2m1 experts and ~120 GB/s on    */
/* the bf16 projections; the GPU reaches ~200 GB/s on the same memory    */
/* because it is the same memory. They are not exclusive: rows are       */
/* independent, so both can work on one matrix at once and the reads     */
/* simply come from two engines.                                         */
/*                                                                      */
/* The split cannot change the arithmetic. Each output row is a serial   */
/* chain of block sums that the CPU kernel defines, and a row is written */
/* by exactly one engine, so the result is bit-identical to the          */
/* all-CPU path (ds_layertest checks exactly that). What the split does   */
/* change is which engine is idle while the other is talking to DRAM.    */
/* ------------------------------------------------------------------ */

/* Share of a projection's output rows handed to the GPU (CWICK_DS_GPU, 0..1).
 * Zero, or anything unparsable, keeps everything on the CPU pool -- which is
 * also what `CWICK_DS_EXACT=1` forces, so the layer oracle never depends on
 * the GPU being present.
 *
 * Default 0, and that is a measured decision, not caution. One matrix per
 * command buffer reaches ~80 GB/s on the dense bf16 shapes while the CPU pool
 * reaches 100-270 GB/s on the same bytes (tools/ds_gemvbench --gemv shapes),
 * because a submit+wait costs ~0.2 ms whether the matrix is 8 MB or 314. The
 * GPU only reaches ~198 GB/s when a dozen matrices share one command buffer,
 * which the current one-matrix-per-call interface cannot express. End to end
 * that made decode 763 ms/step against 461 all-CPU; a 16 MB size gate brought
 * it back to 518, still a loss. Keep the kernels -- the bf16 one is correct
 * and fast when batched -- and revisit if the submission is ever grouped. */
static double ds_gpu_frac(void) {
    static double cached = -1.0;
    if (cached >= 0.0) return cached;
    cached = 0.0;
    const char *e = getenv("CWICK_DS_GPU");
    if (e && *e) {
        double v = atof(e);
        cached = (v > 0.0 && v <= 1.0) ? v : 0.0;
    }
    return cached;
}

/* One pending GPU job per thread. A GEMV submits at most one command buffer,
 * and the engine never runs two GEMVs at once, so a single slot is enough --
 * with the guard below so a re-entrant call degrades to the CPU instead of
 * waiting on somebody else's buffer. */
static __thread void *t_ds_gpu_job = NULL;

/* Rows are handed to the GPU only for matrices large enough to pay for the
 * round trip. A submit+wait costs ~0.2 ms whether the matrix is 8 MB or 314,
 * and a decode step issues ~1400 GEMVs: at that price a split of the small
 * ones (the eight block-diagonal wo_a groups are 8 MB each) costs more than
 * the arithmetic it saves, which is exactly what the first end-to-end A/B
 * measured -- 763 ms/step against 461 all-CPU. Above ~16 MB the trip is a few
 * percent of the matrix's own time. */
static int64_t ds_gpu_min_bytes(void) {
    static int64_t cached = -1;
    if (cached < 0) {
        cached = 16 << 20;
        const char *e = getenv("CWICK_DS_GPU_MIN_MB");
        if (e && *e) {
            double v = atof(e);
            if (v >= 0.0) cached = (int64_t)(v * 1048576.0);
        }
    }
    return cached;
}

/* Submit rows [cpu_rows, N) to the GPU and return how many rows the caller
 * must compute itself. Returns N (nothing submitted) when the GPU declines. */
static int64_t ds_gpu_submit(int kind, const void *w, const void *ws,
                             int64_t N, int64_t K, const float *af,
                             const float *asf, float *out) {
    if (t_ds_gpu_job) return N;
    double f = ds_gpu_frac();
    if (f <= 0.0) return N;
    if (!cwick_metal_ds_ready()) return N;
    int64_t bytes = (kind == 0) ? N * (K / 2) : N * K * 2;
    if (bytes < ds_gpu_min_bytes()) return N;
    int64_t gpu_rows = (int64_t)((double)N * f);
    if (gpu_rows < 128) return N;
    if (gpu_rows > N) gpu_rows = N;
    int64_t cpu_rows = N - gpu_rows;
    void *b = cwick_metal_ds_batch_begin(0, kind);
    if (!b) return N;
    if (!cwick_metal_ds_batch_gemv(b, w, ws, N, K, cpu_rows, N, af, asf, out))
        return N;
    if (!cwick_metal_ds_batch_commit(b)) return N;
    t_ds_gpu_job = b;
    return cpu_rows;
}

/* Wait for the submitted rows and fold them into `out`. Must be called before
 * the caller returns, or the next GEMV will refuse to use the GPU. */
static void ds_gpu_finish(void) {
    if (!t_ds_gpu_job) return;
    cwick_metal_ds_batch_wait(t_ds_gpu_job);
    t_ds_gpu_job = NULL;
}

/* CWICK_DS_GPU_VERIFY=1 re-runs the rows the GPU owned on the CPU and reports
 * the first one that disagrees, with its magnitude. The split is supposed to
 * be bit-identical, so any difference at all is a bug -- and one that is
 * otherwise invisible, because a wrong activation does not crash, it just
 * routes the next token to a different expert set. Cheap enough to leave on
 * while developing (~2x the expert GEMV cost, so not for timing runs). */
static int ds_gpu_verify(void) {
    static int cached = -1;
    if (cached < 0) cached = cwick_env_true("CWICK_DS_GPU_VERIFY") ? 1 : 0;
    return cached;
}

static void ds_gpu_report(const char *what, const float *got, const float *want,
                          int64_t row0, int64_t n, int64_t k) {
    int64_t bad = 0, first = -1;
    float worst = 0.0f, wg = 0.0f, ww = 0.0f;
    for (int64_t r = row0; r < n; r++) {
        float a = got[r], b = want[r];
        if (a == b) continue;
        float d = fabsf(a - b), m = fabsf(b);
        float rel = m > 0.0f ? d / m : d;
        if (a != a || b != b) rel = 1.0e30f;   /* NaN */
        if (bad == 0 || rel > worst) { worst = rel; wg = a; ww = b; }
        if (first < 0) first = r;
        bad++;
    }
    /* One line per distinct shape: a decode step runs ~1400 GEMVs and only a
     * handful of shapes, so an undeduped report is 1400 lines of the same
     * thing and the one that matters scrolls away. */
    static struct { int64_t n, k; int kind; int64_t calls, rows, bad;
                    float worst, wg, ww; } seen[32];
    static int nseen = 0;
    if (!bad) return;
    int i = 0;
    for (; i < nseen; i++)
        if (seen[i].n == n && seen[i].k == k &&
            seen[i].kind == (what[0] == 'f' ? 0 : 1)) break;
    if (i == nseen) {
        if (nseen >= 32) return;
        memset(&seen[i], 0, sizeof(seen[i]));
        seen[i].n = n; seen[i].k = k;
        seen[i].kind = (what[0] == 'f' ? 0 : 1);
        nseen++;
    }
    seen[i].calls++;
    seen[i].rows += n - row0;
    seen[i].bad += bad;
    if (bad > 0 && worst > seen[i].worst) { seen[i].worst = worst; seen[i].wg = wg; seen[i].ww = ww; }
    (void)first;
    fprintf(stderr,
            "dsgpu-verify: %s n=%lld k=%lld row0=%lld: %lld/%lld differ "
            "(shape total %lld/%lld in %lld calls), worst rel %.3e "
            "(got %.9g want %.9g)\n",
            what, (long long)n, (long long)k, (long long)row0,
            (long long)bad, (long long)(n - row0),
            (long long)seen[i].bad, (long long)seen[i].rows,
            (long long)seen[i].calls, (double)seen[i].worst,
            (double)seen[i].wg, (double)seen[i].ww);
}

void ds_fp4_gemv(const uint8_t *a, const uint8_t *a_scale,
                 const uint8_t *w, const uint8_t *w_scale,
                 int64_t N, int64_t K, float *out) {
    build_luts();
    if (N <= 0 || K <= 0) return;
    float af_stack[DS_ACT_MAX], as_stack[DS_ACT_MAX / 32];
    float *af = ds_act_buffer(af_stack, K, DS_ACT_MAX);
    float *asf = ds_act_buffer(as_stack, K / 32, DS_ACT_MAX / 32);
    if (!af || !asf) { free(af); free(asf); return; }
    ds_gemv_dequant_act(a, a_scale, K, af, asf);

#if defined(__aarch64__)
    if (!ds_exact_mode()) {
        float ev_stack[DS_ACT_MAX / 2], od_stack[DS_ACT_MAX / 2];
        float *ev = ds_act_buffer(ev_stack, K / 2, DS_ACT_MAX / 2);
        float *od = ds_act_buffer(od_stack, K / 2, DS_ACT_MAX / 2);
        if (ev && od) {
            for (int64_t j = 0; j < K / 2; j++) {
                ev[j] = af[2 * j];
                od[j] = af[2 * j + 1];
            }
            ds_fp4_arg_neon g = { w, w_scale, out, asf, ev, od, K, K / 32 };
            /* Submit first, then run the CPU share: the two engines overlap
             * for the whole duration of this matrix instead of taking turns. */
            int64_t cpu_rows = ds_gpu_submit(0, w, w_scale, N, K, af, asf, out);
            cwick_pool_for(cpu_rows, &g, ds_fp4_gemv_range_neon);
            ds_gpu_finish();
            if (cpu_rows < N && ds_gpu_verify()) {
                float *ref = (float *)malloc((size_t)N * sizeof(float));
                if (ref) {
                    ds_fp4_arg_neon gv = g;
                    gv.out = ref;
                    ds_fp4_gemv_range_neon(&gv, cpu_rows, N);
                    ds_gpu_report("fp4", out, ref, cpu_rows, N, K);
                    free(ref);
                }
            }
            free(ev == ev_stack ? NULL : ev);
            free(od == od_stack ? NULL : od);
            free(af == af_stack ? NULL : af);
            free(asf == as_stack ? NULL : asf);
            return;
        }
        free(ev == ev_stack ? NULL : ev);
        free(od == od_stack ? NULL : od);
    }
#else
    (void)ds_exact_mode;
#endif

    ds_gemv_arg g = { a, a_scale, w, w_scale, out, af, asf, K, K / 32 };
    if (N < DS_GEMV_PARALLEL_MIN) {
        ds_fp4_gemv_range(&g, 0, N);
    } else {
        cwick_pool_init();
        cwick_pool_for(N, &g, ds_fp4_gemv_range);
    }
    free(af == af_stack ? NULL : af);
    free(asf == as_stack ? NULL : asf);
}

static float ds_fp8_dot(const float *E4, const uint8_t *row, const uint8_t *rs,
                        const float *af, const float *asf, int64_t nb) {
    float acc = 0.0f;
    for (int64_t b = 0; b < nb; b++) {
        const uint8_t *p = row + b * 32;
        const float *av = af + b * 32;
        float part = 0.0f;
        for (int j = 0; j < 32; j++)
            part += E4[p[j]] * av[j];
        acc += (part * asf[b]) * ds_e8m0_to_f32(rs[b]);
    }
    return acc;
}

/* Same four-row scheme as the FP4 kernel. The weight scale here is shared by
 * 32 consecutive rows rather than owned per row, so each row in the group
 * derives its own base from n/32 -- a group of four can straddle two of them. */
#define DS_FP8_STEP(R, P)                     \
    do {                                      \
        (P) += E4[(R)[b * 32 + j]] * av[j];   \
    } while (0)

static void ds_fp8_gemv_range(void *argp, int64_t n0, int64_t n1) {
    ds_gemv_arg *g = (ds_gemv_arg *)argp;
    const float *E4 = DS_E4M3_LUT;
    const int64_t K = g->K, nb = g->nb;
    const float *af = g->af, *asf = g->asf;
    int64_t n = n0;

    if (ds_gemv_row_group() == 1) {
        for (; n < n1; n++)
            g->out[n] = ds_fp8_dot(E4, g->w + n * K,
                                   g->w_scale + (n / 32) * nb, af, asf, nb);
        return;
    }

    for (; n + 3 < n1; n += 4) {
        const uint8_t *r0 = g->w + (n + 0) * K;
        const uint8_t *r1 = g->w + (n + 1) * K;
        const uint8_t *r2 = g->w + (n + 2) * K;
        const uint8_t *r3 = g->w + (n + 3) * K;
        const uint8_t *s0 = g->w_scale + ((n + 0) / 32) * nb;
        const uint8_t *s1 = g->w_scale + ((n + 1) / 32) * nb;
        const uint8_t *s2 = g->w_scale + ((n + 2) / 32) * nb;
        const uint8_t *s3 = g->w_scale + ((n + 3) / 32) * nb;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int64_t b = 0; b < nb; b++) {
            const float *av = af + b * 32;
            float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
            for (int j = 0; j < 32; j++) {
                DS_FP8_STEP(r0, p0);
                DS_FP8_STEP(r1, p1);
                DS_FP8_STEP(r2, p2);
                DS_FP8_STEP(r3, p3);
            }
            a0 += (p0 * asf[b]) * ds_e8m0_to_f32(s0[b]);
            a1 += (p1 * asf[b]) * ds_e8m0_to_f32(s1[b]);
            a2 += (p2 * asf[b]) * ds_e8m0_to_f32(s2[b]);
            a3 += (p3 * asf[b]) * ds_e8m0_to_f32(s3[b]);
        }
        g->out[n + 0] = a0;
        g->out[n + 1] = a1;
        g->out[n + 2] = a2;
        g->out[n + 3] = a3;
    }
    for (; n < n1; n++)
        g->out[n] = ds_fp8_dot(E4, g->w + n * K,
                               g->w_scale + (n / 32) * nb, af, asf, nb);
}

void ds_fp8_gemv(const uint8_t *a, const uint8_t *a_scale,
                 const uint8_t *w, const uint8_t *w_scale,
                 int64_t N, int64_t K, float *out) {
    build_luts();
    if (N <= 0 || K <= 0) return;
    float af_stack[DS_ACT_MAX], as_stack[DS_ACT_MAX / 32];
    float *af = ds_act_buffer(af_stack, K, DS_ACT_MAX);
    float *asf = ds_act_buffer(as_stack, K / 32, DS_ACT_MAX / 32);
    if (!af || !asf) { free(af); free(asf); return; }
    ds_gemv_dequant_act(a, a_scale, K, af, asf);
    ds_gemv_arg g = { a, a_scale, w, w_scale, out, af, asf, K, K / 32 };
    if (N < DS_GEMV_PARALLEL_MIN) {
        ds_fp8_gemv_range(&g, 0, N);
    } else {
        cwick_pool_init();
        cwick_pool_for(N, &g, ds_fp8_gemv_range);
    }
    free(af == af_stack ? NULL : af);
    free(asf == as_stack ? NULL : asf);
}

/* ------------------------------------------------------------------ */
/* dense weights held as bf16                                           */
/*                                                                      */
/* The attention projections, the shared expert and the indexer are read */
/* end to end on every token: ~5.4 GB per step, three times the routed   */
/* experts' arithmetic for a tenth of the FLOPs. Widening them to bf16   */
/* once at load is lossless (e4m3 has 3 mantissa bits, bf16 has 8) and    */
/* turns the inner loop into a shift-and-fma: measured 119 GB/s against   */
/* 38 GB/s for the fp8 table lookup.                                     */
/* ------------------------------------------------------------------ */

void ds_fp8_to_bf16_row(const uint8_t *src, uint16_t *dst, int64_t n) {
    build_luts();
    const float *E4 = DS_E4M3_LUT;
    for (int64_t i = 0; i < n; i++) {
        union { float f; uint32_t u; } c;
        c.f = E4[src[i]];
        dst[i] = (uint16_t)(c.u >> 16);   /* exact: the low 16 bits are zero */
    }
}

typedef struct {
    const uint16_t *w16;
    const uint8_t *w_scale;
    float *out;
    const float *af, *asf;
    int64_t K, nb;
} ds_qbf16_arg;

/* One output row: block sums of 32 over bf16 weights, then the two block
 * scales -- the reference's order. The block sum itself is four partial lanes
 * (see ds_exact_mode for what that costs in bits). */
static void ds_qbf16_gemv_range_neon(void *argp, int64_t n0, int64_t n1) {
    ds_qbf16_arg *g = (ds_qbf16_arg *)argp;
    const int64_t K = g->K, nb = g->nb;
    const float *asf = g->asf;
    const int64_t k32 = K & ~(int64_t)31;

    for (int64_t n = n0; n < n1; n++) {
        const uint16_t *row = g->w16 + n * K;
        const uint8_t *rs = g->w_scale + (n / 32) * nb;
        float acc = 0.0f;
        int64_t b = 0;
        for (; b < k32 / 32; b++) {
            const uint16_t *wp = row + b * 32;
            const float *av = g->af + b * 32;
            float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0, s2 = s0, s3 = s0;
            for (int q = 0; q < 32; q += 16) {
                uint16x8_t w0 = vld1q_u16(wp + q);
                uint16x8_t w1 = vld1q_u16(wp + q + 8);
                float32x4_t f0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w0), 16));
                float32x4_t f1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(w0), 16));
                float32x4_t f2 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w1), 16));
                float32x4_t f3 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(w1), 16));
                s0 = vfmaq_f32(s0, f0, vld1q_f32(av + q));
                s1 = vfmaq_f32(s1, f1, vld1q_f32(av + q + 4));
                s2 = vfmaq_f32(s2, f2, vld1q_f32(av + q + 8));
                s3 = vfmaq_f32(s3, f3, vld1q_f32(av + q + 12));
            }
            float part = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
            acc += (part * asf[b]) * ds_e8m0_to_f32(rs[b]);
        }
        g->out[n] = acc;
    }
}

/* Reference order: one accumulator, the 32 products of a block summed in
 * sequence. Used when CWICK_DS_EXACT=1 and on targets without NEON. */
static void ds_qbf16_gemv_range_serial(void *argp, int64_t n0, int64_t n1) {
    ds_qbf16_arg *g = (ds_qbf16_arg *)argp;
    const int64_t K = g->K, nb = g->nb;
    for (int64_t n = n0; n < n1; n++) {
        const uint16_t *row = g->w16 + n * K;
        const uint8_t *rs = g->w_scale + (n / 32) * nb;
        float acc = 0.0f;
        for (int64_t b = 0; b < K / 32; b++) {
            float part = 0.0f;
            for (int j = 0; j < 32; j++)
                part += ds_bf16_to_f32(row[b * 32 + j]) * g->af[b * 32 + j];
            acc += (part * g->asf[b]) * ds_e8m0_to_f32(rs[b]);
        }
        g->out[n] = acc;
    }
}

void ds_qbf16_gemv(const uint16_t *w16, const uint8_t *w_scale,
                   const uint8_t *a, const uint8_t *a_scale,
                   int64_t N, int64_t K, float *out) {
    build_luts();
    if (N <= 0 || K <= 0) return;
    float af_stack[DS_ACT_MAX], as_stack[DS_ACT_MAX / 32];
    float *af = ds_act_buffer(af_stack, K, DS_ACT_MAX);
    float *asf = ds_act_buffer(as_stack, K / 32, DS_ACT_MAX / 32);
    if (!af || !asf) { free(af); free(asf); return; }
    ds_gemv_dequant_act(a, a_scale, K, af, asf);
    ds_qbf16_arg g = { w16, w_scale, out, af, asf, K, K / 32 };
#if defined(__aarch64__)
    if (!ds_exact_mode()) {
        int64_t cpu_rows = ds_gpu_submit(1, w16, w_scale, N, K, af, asf, out);
        cwick_pool_for(cpu_rows, &g, ds_qbf16_gemv_range_neon);
        ds_gpu_finish();
        if (cpu_rows < N && ds_gpu_verify()) {
            float *ref = (float *)malloc((size_t)N * sizeof(float));
            if (ref) {
                ds_qbf16_arg gv = g;
                gv.out = ref;
                ds_qbf16_gemv_range_neon(&gv, cpu_rows, N);
                ds_gpu_report("bf16", out, ref, cpu_rows, N, K);
                free(ref);
            }
        }
        free(af == af_stack ? NULL : af);
        free(asf == as_stack ? NULL : asf);
        return;
    }
#endif
    if (N < DS_GEMV_PARALLEL_MIN) {
        ds_qbf16_gemv_range_serial(&g, 0, N);
    } else {
        cwick_pool_init();
        cwick_pool_for(N, &g, ds_qbf16_gemv_range_serial);
    }
    free(af == af_stack ? NULL : af);
    free(asf == as_stack ? NULL : asf);
}

/* Row split for the bf16/f32 GEMVs. Both accumulate one output row at a time
 * over K in order, so a worker owning rows [n0,n1) reproduces the serial
 * arithmetic exactly -- the same argument the quantized kernels rely on. */
typedef struct {
    const uint16_t *w16;
    const float *w32;
    const float *x, *bias;
    float *out;
    int64_t N, K;
} ds_plain_gemv_arg;

static void ds_f32_gemv_range(void *argp, int64_t n0, int64_t n1) {
    ds_plain_gemv_arg *g = (ds_plain_gemv_arg *)argp;
    const int64_t K = g->K;
    const float *x = g->x;
    for (int64_t n = n0; n < n1; n++) {
        const float *row = g->w32 + n * K;
        float acc = g->bias ? g->bias[n] : 0.0f;
        for (int64_t k = 0; k < K; k++) acc += row[k] * x[k];
        g->out[n] = acc;
    }
}

#if !defined(__aarch64__)
/* The scalar bf16 row: one accumulator, K additions in order. Kept as the
 * portable fallback for a target without NEON. */
static void ds_bf16_gemv_range_serial(void *argp, int64_t n0, int64_t n1) {
    ds_plain_gemv_arg *g = (ds_plain_gemv_arg *)argp;
    const int64_t K = g->K;
    const float *x = g->x;
    for (int64_t n = n0; n < n1; n++) {
        const uint16_t *row = g->w16 + n * K;
        float acc = g->bias ? g->bias[n] : 0.0f;
        for (int64_t k = 0; k < K; k++)
            acc += ds_bf16_to_f32(row[k]) * x[k];
        g->out[n] = acc;
    }
}
#endif

#if defined(__aarch64__)

/* bf16 weights are read as bf16 and widened with a shift: a bf16 value is the
 * top half of the fp32 it decodes to, so `(uint32)w << 16` reinterpreted as
 * float is the value itself -- no table, no conversion instruction. That is
 * what makes four lanes as cheap as one here, and why the vocabulary head and
 * the dense projections want their weights in bf16 rather than fp8.
 *
 * The accumulators are four partial sums, so a row's additions happen in a
 * different order than the serial loop's. Every product is still exact in fp32
 * (a bf16 weight has 8 mantissa bits, an e4m3 activation 3), so the difference
 * is the fp32 rounding of the sum, not a lower-precision multiply. */
static inline float ds_hsum4(float32x4_t a) { return vaddvq_f32(a); }

static void ds_bf16_gemv_range_neon(void *argp, int64_t n0, int64_t n1) {
    ds_plain_gemv_arg *g = (ds_plain_gemv_arg *)argp;
    const int64_t K = g->K;
    const int64_t k16 = K & ~(int64_t)15;
    const float *x = g->x;

    for (int64_t n = n0; n < n1; n++) {
        const uint16_t *row = g->w16 + n * K;
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
        int64_t k = 0;
        for (; k < k16; k += 16) {
            uint16x8_t w0 = vld1q_u16(row + k);
            uint16x8_t w1 = vld1q_u16(row + k + 8);
            float32x4_t f0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w0), 16));
            float32x4_t f1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(w0), 16));
            float32x4_t f2 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w1), 16));
            float32x4_t f3 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(w1), 16));
            a0 = vfmaq_f32(a0, f0, vld1q_f32(x + k));
            a1 = vfmaq_f32(a1, f1, vld1q_f32(x + k + 4));
            a2 = vfmaq_f32(a2, f2, vld1q_f32(x + k + 8));
            a3 = vfmaq_f32(a3, f3, vld1q_f32(x + k + 12));
        }
        float acc = ds_hsum4(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
        if (g->bias) acc += g->bias[n];
        for (; k < K; k++) acc += ds_bf16_to_f32(row[k]) * x[k];
        g->out[n] = acc;
    }
}
#endif /* __aarch64__ */

/* The vocabulary head is 129280 x 5120: 1.3 GiB of bf16 read on every decode
 * step, and as a single serial loop it was 448 ms of a 1050 ms step -- 40% of
 * the step on one core while eleven sat idle. Rows are independent, so the
 * split is free; the four-lane inner loop is what makes the read bandwidth,
 * not the add latency, the limit. */
void ds_bf16_gemv(const uint16_t *w, const float *x, const float *bias,
                  int64_t N, int64_t K, float *out) {
    ds_plain_gemv_arg g = { w, NULL, x, bias, out, N, K };
#if defined(__aarch64__)
    cwick_pool_for(N, &g, ds_bf16_gemv_range_neon);
#else
    if (N < DS_GEMV_PARALLEL_MIN) {
        ds_bf16_gemv_range_serial(&g, 0, N);
    } else {
        cwick_pool_init();
        cwick_pool_for(N, &g, ds_bf16_gemv_range_serial);
    }
#endif
}

void ds_f32_gemv(const float *w, const float *x, const float *bias,
                 int64_t N, int64_t K, float *out) {
    ds_plain_gemv_arg g = { NULL, w, x, bias, out, N, K };
    if (N < DS_GEMV_PARALLEL_MIN) {
        ds_f32_gemv_range(&g, 0, N);
    } else {
        cwick_pool_init();
        cwick_pool_for(N, &g, ds_f32_gemv_range);
    }
}
