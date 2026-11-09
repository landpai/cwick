/*
 * ds_gemvbench.c -- how fast the quantized GEMV kernels actually run.
 *
 * The forward pass mixes three things that all look like "slow": SSD reads,
 * page-cache misses and the multiply-add loops. Timing a whole token cannot
 * tell them apart, and an A/B of the whole model is at the mercy of whatever
 * the page cache happened to hold. This bench isolates the arithmetic: it
 * synthesises weights of the shape the model uses, calls one kernel in a loop,
 * and reports ms/call, GB/s and GMAC/s.
 *
 * No checkpoint and no files: only src/ds_quant.c is needed, so a kernel
 * change can be timed in seconds instead of by loading 475 GiB of experts.
 *
 * Usage
 *   ds_gemvbench                                   # fp4 gate/up + down
 *   ds_gemvbench --gemv fp4|fp8|bf16|head|check
 *                 [--reps N] [--threads N] [--distinct N] [--rows 4|8|16]
 *   ds_gemvbench [N] [K] [iters] [rotate]          # the original form
 *
 * Shapes default to the checkpoint's (dim 5120, moe_inter 2304, head vocab
 * 129280). Threads come from CWICK_THREADS and rows from DS_GEMV_ROWS, so a
 * run here measures exactly the kernel the model would use.
 *
 * What each mode answers
 *   fp4     routed experts: gate/up (2304x5120) and down (5120x2304)
 *   fp8     the dense weights as stored: e4m3 with 32x32 block scales
 *   bf16    the same weights widened to bf16 at load -- the faster path
 *   head    the 129280x5120 bf16 output projection, the priciest single call
 *   check   recompute a few rows in the reference's order and report the gap
 *   compare the original old-vs-new baseline for the fp4/fp8 kernels
 *
 * `--distinct N` keeps N different matrices in rotation. That flag is not a
 * detail: one 6 MiB matrix stays in the last-level cache and reports a
 * bandwidth no decode step ever sees, so a single-copy measurement flatters
 * the kernel by roughly the size of the cache.
 *
 * Compare only runs taken back to back. The same invocation measured 0.075,
 * 0.111 and 0.171 ms/call on this machine ten minutes apart -- thermal state,
 * not the kernel. The first run after an idle period is the fastest, so an A/B
 * that opens with arm A and later runs arm B will invent a regression.
 */
#include <math.h>
#include <stdint.h>
#include <time.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cwick.h"
#include "ds.h"

/* Declared in metal.c (kept out of the public header on purpose: only the
 * backend itself brings Metal up). */
int cwick_metal_init(void);

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static double now_ms(void) { return now_s() * 1e3; }

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", n); exit(1); }
    return p;
}

/* xorshift: the byte values only have to look like checkpoint bytes, and
 * rand() over 1.3 GiB is slower than the measurement it feeds. */
static void fill_random(void *p, size_t n) {
    uint8_t *b = (uint8_t *)p;
    uint64_t s = 0x243F6A8885A308D3ull;
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        b[i] = (uint8_t)(s >> 24);
    }
}

/* ------------------------------------------------------------------ */
/* the two kernels as they looked before the pair table and the        */
/* dequantized activation: two table gathers per multiply-add, and the  */
/* activation re-read from the quantized bytes for every output row.    */
/* Kept as the baseline `--gemv compare` measures the current ones      */
/* against.                                                            */
/* ------------------------------------------------------------------ */

static void old_fp4_gemv(const uint8_t *a, const uint8_t *a_scale,
                         const uint8_t *w, const uint8_t *w_scale,
                         int64_t N, int64_t K, float *out) {
    const float *F4 = DS_FP4_LUT, *E4 = DS_E4M3_LUT;
    int64_t nb = K / 32;
    for (int64_t n = 0; n < N; n++) {
        const uint8_t *row = w + n * (K / 2);
        const uint8_t *rs = w_scale + n * nb;
        float acc = 0.0f;
        for (int64_t b = 0; b < nb; b++) {
            const uint8_t *p = row + b * 16;
            float part = 0.0f;
            for (int j = 0; j < 16; j++) {
                part += F4[p[j] & 0x0F] * E4[a[b * 32 + 2 * j]];
                part += F4[p[j] >> 4] * E4[a[b * 32 + 2 * j + 1]];
            }
            acc += (part * ds_e8m0_to_f32(a_scale[b])) *
                   ds_e8m0_to_f32(rs[b]);
        }
        out[n] = acc;
    }
}

static void old_fp8_gemv(const uint8_t *a, const uint8_t *a_scale,
                         const uint8_t *w, const uint8_t *w_scale,
                         int64_t N, int64_t K, float *out) {
    const float *E4 = DS_E4M3_LUT;
    int64_t nb = K / 32;
    for (int64_t n = 0; n < N; n++) {
        const uint8_t *row = w + n * K;
        const uint8_t *rs = w_scale + (n / 32) * nb;
        float acc = 0.0f;
        for (int64_t b = 0; b < nb; b++) {
            const uint8_t *p = row + b * 32;
            const uint8_t *q = a + b * 32;
            float part = 0.0f;
            for (int j = 0; j < 32; j++) part += E4[p[j]] * E4[q[j]];
            acc += (part * ds_e8m0_to_f32(a_scale[b])) *
                   ds_e8m0_to_f32(rs[b]);
        }
        out[n] = acc;
    }
}

/* ------------------------------------------------------------------ */
/* timing                                                              */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *name;
    int64_t N, K;          /* GEMV shape */
    int kind;              /* 0 fp4, 1 fp8, 2 bf16 (fp8 widened at load) */
    size_t w_bytes;        /* bytes touched per call */
    int64_t macs;          /* multiply-accumulates per call */
} shape;

static void bench_q(const shape *s, int reps, int distinct) {
    if (distinct < 1) distinct = 1;
    size_t scale_bytes = (size_t)(s->N * (s->K / 32));
    size_t stride = s->w_bytes;
    uint8_t *a = (uint8_t *)xmalloc((size_t)s->K);
    uint8_t *as = (uint8_t *)xmalloc((size_t)(s->K / 32));
    float *out = (float *)xmalloc(sizeof(float) * (size_t)s->N);
    uint8_t *w = NULL, *ws = NULL;
    uint16_t *w16 = NULL;

    if (s->kind == 2) {
        /* widened: the bytes a call reads are the bf16 copy, twice the fp8 */
        stride = (size_t)(s->N * s->K) * sizeof(uint16_t);
        w16 = (uint16_t *)xmalloc(stride * (size_t)distinct);
        fill_random(w16, stride * (size_t)distinct);
    } else {
        w = (uint8_t *)xmalloc(stride * (size_t)distinct);
        fill_random(w, stride * (size_t)distinct);
        /* 0x7F and 0xFF are e4m3 NaN patterns; a checkpoint never stores one,
         * and a NaN-heavy inner loop measures a compiler-dependent thing */
        if (s->kind == 1)
            for (size_t i = 0; i < stride * (size_t)distinct; i++)
                if ((w[i] & 0x7F) == 0x7F) w[i] = 0;
    }
    /* scales land in the range the released checkpoint uses (0x72..0x7b) */
    ws = (uint8_t *)xmalloc(scale_bytes * (size_t)distinct);
    for (size_t i = 0; i < scale_bytes * (size_t)distinct; i++)
        ws[i] = (uint8_t)(0x72 + (i % 10));

    fill_random(a, (size_t)s->K);
    for (int64_t b = 0; b < s->K / 32; b++) as[b] = 0x7b;

    printf("%s N=%lld K=%lld weights %.2f MiB x %d = %.2f GiB  threads %d  rows %s\n",
           s->name, (long long)s->N, (long long)s->K,
           (double)stride / 1048576.0, distinct,
           (double)(stride * (size_t)distinct) / 1073741824.0,
           cwick_pool_threads(),
           getenv("DS_GEMV_ROWS") ? getenv("DS_GEMV_ROWS") : "4");

    for (int d = 0; d < distinct; d++) {          /* warm-up, all copies */
        const uint8_t *wp = w ? w + (size_t)d * stride : NULL;
        if (s->kind == 0)
            ds_fp4_gemv(a, as, wp, ws + (size_t)d * scale_bytes, s->N, s->K, out);
        else if (s->kind == 1)
            ds_fp8_gemv(a, as, wp, ws + (size_t)d * scale_bytes, s->N, s->K, out);
        else
            ds_qbf16_gemv(w16 + (size_t)d * (size_t)(s->N * s->K), ws,
                          a, as, s->N, s->K, out);
    }

    double t0 = now_ms();
    for (int r = 0; r < reps; r++) {
        int d = r % distinct;
        const uint8_t *wp = w ? w + (size_t)d * stride : NULL;
        if (s->kind == 0)
            ds_fp4_gemv(a, as, wp, ws + (size_t)d * scale_bytes, s->N, s->K, out);
        else if (s->kind == 1)
            ds_fp8_gemv(a, as, wp, ws + (size_t)d * scale_bytes, s->N, s->K, out);
        else
            ds_qbf16_gemv(w16 + (size_t)d * (size_t)(s->N * s->K), ws,
                          a, as, s->N, s->K, out);
    }
    double ms = (now_ms() - t0) / (double)reps;

    printf("  %.3f ms/call  %.1f GB/s  %.2f GMAC/s\n", ms,
           (double)stride / (ms * 1e6), (double)s->macs / (ms * 1e6));
    free(w); free(w16); free(ws); free(out); free(a); free(as);
}

static void bench_bf16_head(int reps) {
    const int64_t N = 129280, K = 5120;
    size_t bytes = sizeof(uint16_t) * (size_t)(N * K);
    uint16_t *w = (uint16_t *)xmalloc(bytes);
    float *x = (float *)xmalloc(sizeof(float) * (size_t)K);
    float *out = (float *)xmalloc(sizeof(float) * (size_t)N);
    fill_random(w, bytes);
    /* random bf16 patterns decode to arbitrary floats; the multiply does not
     * care, but a NaN would turn the loop into a denormal-slow measurement */
    for (int64_t i = 0; i < K; i++) x[i] = 0.5f;

    printf("bf16 head N=%lld K=%lld weights %.1f MiB  threads %d\n",
           (long long)N, (long long)K, (double)bytes / 1048576.0,
           cwick_pool_threads());
    ds_bf16_gemv(w, x, NULL, N, K, out);
    double t0 = now_ms();
    for (int r = 0; r < reps; r++) ds_bf16_gemv(w, x, NULL, N, K, out);
    double ms = (now_ms() - t0) / (double)reps;
    printf("  %.3f ms/call  %.1f GB/s  %.2f GMAC/s\n", ms,
           (double)bytes / (ms * 1e6), (double)(N * K) / (ms * 1e6));
    free(w); free(x); free(out);
}

/* ------------------------------------------------------------------ */
/* numeric check                                                        */
/*                                                                      */
/* The kernels reorder nothing about the products -- an e2m1 weight     */
/* times an e4m3 activation needs four mantissa bits and lands exactly  */
/* in fp32 -- but they do sum four partial lanes instead of one chain,   */
/* and the dense weights are widened from fp8 to bf16 at load. This      */
/* recomputes a handful of rows in the reference's order from the        */
/* dequantized weights and reports how far the kernels moved.            */
/* ------------------------------------------------------------------ */

static double rel_err(const float *got, const float *ref, int64_t n, double *rms) {
    double worst = 0.0, acc = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double d = fabs((double)got[i] - (double)ref[i]);
        double scale = fabs((double)ref[i]);
        double e = scale > 1e-30 ? d / scale : d;
        if (e > worst) worst = e;
        acc += d * d;
    }
    if (rms) *rms = sqrt(acc / (double)n);
    return worst;
}

static void check_kernels(void) {
    const int64_t N = 64, K = 512;
    const int64_t nb = K / 32, half = K / 2;

    float *x = (float *)xmalloc(sizeof(float) * (size_t)K);
    for (int64_t i = 0; i < K; i++)
        x[i] = (float)(((i * 37) % 61) - 30) * 0.37f;   /* mixed magnitudes */
    uint8_t *aq = (uint8_t *)xmalloc((size_t)K);
    uint8_t *as = (uint8_t *)xmalloc((size_t)nb);
    for (int64_t b = 0; b < nb; b++) as[b] = ds_act_quant_block(x + b * 32, aq + b * 32);

    /* the activation dequantized the way the kernels do it: raw value plus the
     * per-block scale, applied to the block sum */
    float *af = (float *)xmalloc(sizeof(float) * (size_t)K);
    float *asf = (float *)xmalloc(sizeof(float) * (size_t)nb);
    for (int64_t k = 0; k < K; k++) af[k] = ds_e4m3_to_f32(aq[k]);
    for (int64_t b = 0; b < nb; b++) asf[b] = ds_e8m0_to_f32(as[b]);

    /* ---- fp4 experts ---- */
    {
        uint8_t *w = (uint8_t *)xmalloc((size_t)(N * half));
        uint8_t *ws = (uint8_t *)xmalloc((size_t)(N * nb));
        fill_random(w, (size_t)(N * half));
        for (int64_t i = 0; i < N * nb; i++) ws[i] = (uint8_t)(0x72 + (i % 10));
        float *got = (float *)xmalloc(sizeof(float) * (size_t)N);
        float *ref = (float *)xmalloc(sizeof(float) * (size_t)N);
        for (int64_t n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int64_t b = 0; b < nb; b++) {
                const uint8_t *p = w + n * half + b * 16;
                float part = 0.0f;
                for (int j = 0; j < 16; j++) {
                    part += ds_fp4_to_f32(p[j] & 0x0Fu) * af[b * 32 + 2 * j];
                    part += ds_fp4_to_f32(p[j] >> 4) * af[b * 32 + 2 * j + 1];
                }
                acc += (part * asf[b]) * ds_e8m0_to_f32(ws[n * nb + b]);
            }
            ref[n] = acc;
        }
        ds_fp4_gemv(aq, as, w, ws, N, K, got);
        double rms = 0.0;
        printf("fp4  vs reference order: worst rel %.3e, rms abs %.3e\n",
               rel_err(got, ref, N, &rms), rms);
        free(w); free(ws); free(got); free(ref);
    }

    /* ---- fp8 dense, plain and widened ---- */
    {
        uint8_t *w = (uint8_t *)xmalloc((size_t)(N * K));
        uint8_t *ws = (uint8_t *)xmalloc((size_t)((N / 32) * nb));
        uint16_t *w16 = (uint16_t *)xmalloc(sizeof(uint16_t) * (size_t)(N * K));
        fill_random(w, (size_t)(N * K));
        for (int64_t i = 0; i < N * K; i++)
            if ((w[i] & 0x7F) == 0x7F) w[i] = 0;
        for (int64_t i = 0; i < (N / 32) * nb; i++) ws[i] = (uint8_t)(0x72 + (i % 10));
        ds_fp8_to_bf16_row(w, w16, N * K);

        float *got8 = (float *)xmalloc(sizeof(float) * (size_t)N);
        float *got16 = (float *)xmalloc(sizeof(float) * (size_t)N);
        float *ref = (float *)xmalloc(sizeof(float) * (size_t)N);
        for (int64_t n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int64_t b = 0; b < nb; b++) {
                float part = 0.0f;
                for (int j = 0; j < 32; j++)
                    part += ds_e4m3_to_f32(w[n * K + b * 32 + j]) * af[b * 32 + j];
                acc += (part * asf[b]) * ds_e8m0_to_f32(ws[(n / 32) * nb + b]);
            }
            ref[n] = acc;
        }
        ds_fp8_gemv(aq, as, w, ws, N, K, got8);
        ds_qbf16_gemv(w16, ws, aq, as, N, K, got16);
        double rms = 0.0;
        printf("fp8  scalar vs reference order: worst rel %.3e, rms abs %.3e\n",
               rel_err(got8, ref, N, &rms), rms);
        printf("bf16 widened vs reference order: worst rel %.3e, rms abs %.3e\n",
               rel_err(got16, ref, N, &rms), rms);
        printf("bf16 widened vs fp8 scalar:     worst rel %.3e\n",
               rel_err(got16, got8, N, NULL));
        free(w); free(ws); free(w16); free(got8); free(got16); free(ref);
    }

    free(x); free(aq); free(as); free(af); free(asf);
}

/* ------------------------------------------------------------------ */
/* GPU split check                                                      */
/*                                                                      */
/* The GPU path exists to run part of a matrix's output rows while the  */
/* CPU pool runs the rest, which is only sound if a row computed on the */
/* GPU is bit-identical to the same row computed on the CPU. This runs  */
/* one shape with everything on the GPU (CWICK_DS_GPU=1) and compares   */
/* against the reference summation, so a broken tree shows up as a      */
/* number rather than as a garbled generation three minutes later.      */
/* ------------------------------------------------------------------ */
static void gpu_check(void) {
    /* Must be set before the first GEMV: the share is read once. */
    setenv("CWICK_DS_GPU", "1", 1);
    if (!cwick_metal_init()) {
        printf("gpu: metal init failed (CPU path only)\n");
        return;
    }
    printf("gpu: ds batch %s\n", cwick_metal_ds_ready() ? "ready" : "UNAVAILABLE");
    if (!cwick_metal_ds_ready()) return;
    /* Hold the GPU clock up: the DeepSeek step alternates a GPU batch with
     * milliseconds of CPU work, which is exactly the pattern that lets the
     * power manager downclock between dispatches. */
    printf("gpu: keepalive %d\n", cwick_metal_keepalive_start());

    const int64_t N = 2304, K = 5120;    /* the routed gate/up shape */
    const int64_t nb = K / 32, half = K / 2;

    float *x = (float *)xmalloc(sizeof(float) * (size_t)K);
    for (int64_t i = 0; i < K; i++)
        x[i] = (float)(((i * 37) % 61) - 30) * 0.37f;
    uint8_t *aq = (uint8_t *)xmalloc((size_t)K);
    uint8_t *as = (uint8_t *)xmalloc((size_t)nb);
    for (int64_t b = 0; b < nb; b++) as[b] = ds_act_quant_block(x + b * 32, aq + b * 32);
    float *af = (float *)xmalloc(sizeof(float) * (size_t)K);
    float *asf = (float *)xmalloc(sizeof(float) * (size_t)nb);
    for (int64_t k = 0; k < K; k++) af[k] = ds_e4m3_to_f32(aq[k]);
    for (int64_t b = 0; b < nb; b++) asf[b] = ds_e8m0_to_f32(as[b]);

    /* ---- fp4 ---- */
    {
        uint8_t *w = (uint8_t *)xmalloc((size_t)(N * half));
        uint8_t *ws = (uint8_t *)xmalloc((size_t)(N * nb));
        fill_random(w, (size_t)(N * half));
        for (int64_t i = 0; i < N * nb; i++) ws[i] = (uint8_t)(0x72 + (i % 10));
        float *got = (float *)xmalloc(sizeof(float) * (size_t)N);
        float *ref = (float *)xmalloc(sizeof(float) * (size_t)N);
        for (int64_t n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int64_t b = 0; b < nb; b++) {
                const uint8_t *p = w + n * half + b * 16;
                float part = 0.0f;
                for (int j = 0; j < 16; j++) {
                    part += ds_fp4_to_f32(p[j] & 0x0Fu) * af[b * 32 + 2 * j];
                    part += ds_fp4_to_f32(p[j] >> 4) * af[b * 32 + 2 * j + 1];
                }
                acc += (part * asf[b]) * ds_e8m0_to_f32(ws[n * nb + b]);
            }
            ref[n] = acc;
        }
        ds_fp4_gemv(aq, as, w, ws, N, K, got);
        printf("fp4  gpu rows vs reference: worst rel %.3e\n",
               rel_err(got, ref, N, NULL));
        /* A second call on the same slot, with `got` cleared first: it proves
         * the staging offsets and the "rows the GPU owns" are reset between
         * calls rather than accumulating. */
        memset(got, 0, sizeof(float) * (size_t)N);
        ds_fp4_gemv(aq, as, w, ws, N, K, got);
        printf("fp4  gpu rows, 2nd call:      worst rel %.3e\n",
               rel_err(got, ref, N, NULL));
        double t0 = now_s();
        for (int i = 0; i < 20; i++) ds_fp4_gemv(aq, as, w, ws, N, K, got);
        double dt = (now_s() - t0) / 20.0;
        printf("fp4  gpu+pool split:          %.3f ms/call  %.1f GB/s\n",
               dt * 1e3, (double)(N * half) / dt / 1e9);

        /* One command buffer carrying twelve matrices, which is the shape a
         * decode layer should submit: the round trip is paid once for the
         * whole group instead of once per matrix. */
        float *outs[12];
        for (int i = 0; i < 12; i++) outs[i] = (float *)xmalloc(sizeof(float) * (size_t)N);
        t0 = now_s();
        for (int it = 0; it < 5; it++) {
            void *b = cwick_metal_ds_batch_begin(0, 0);
            for (int i = 0; i < 12; i++)
                cwick_metal_ds_batch_gemv(b, w, ws, N, K, 0, N, af, asf, outs[i]);
            cwick_metal_ds_batch_commit(b);
            cwick_metal_ds_batch_wait(b);
        }
        dt = (now_s() - t0) / 5.0;
        printf("fp4  gpu only, 12 per batch:  %.3f ms/batch  %.1f GB/s  "
               "worst rel %.3e\n", dt * 1e3,
               (double)(12 * N * half) / dt / 1e9, rel_err(outs[0], ref, N, NULL));
        for (int i = 0; i < 12; i++) free(outs[i]);
        free(w); free(ws); free(got); free(ref);
    }

    /* ---- bf16 ---- */
    {
        uint8_t *w = (uint8_t *)xmalloc((size_t)(N * K));
        uint8_t *ws = (uint8_t *)xmalloc((size_t)((N / 32) * nb));
        uint16_t *w16 = (uint16_t *)xmalloc(sizeof(uint16_t) * (size_t)(N * K));
        fill_random(w, (size_t)(N * K));
        for (int64_t i = 0; i < N * K; i++)
            if ((w[i] & 0x7F) == 0x7F) w[i] = 0;
        for (int64_t i = 0; i < (N / 32) * nb; i++) ws[i] = (uint8_t)(0x72 + (i % 10));
        ds_fp8_to_bf16_row(w, w16, N * K);
        float *got = (float *)xmalloc(sizeof(float) * (size_t)N);
        float *ref = (float *)xmalloc(sizeof(float) * (size_t)N);
        for (int64_t n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int64_t b = 0; b < nb; b++) {
                float part = 0.0f;
                for (int j = 0; j < 32; j++)
                    part += ds_e4m3_to_f32(w[n * K + b * 32 + j]) * af[b * 32 + j];
                acc += (part * asf[b]) * ds_e8m0_to_f32(ws[(n / 32) * nb + b]);
            }
            ref[n] = acc;
        }
        ds_qbf16_gemv(w16, ws, aq, as, N, K, got);
        printf("bf16 gpu rows vs reference: worst rel %.3e\n",
               rel_err(got, ref, N, NULL));
        /* Same 12-per-buffer shape as the fp4 case, on the dense projection
         * that dominates a decode step. */
        {
            float *outs[12];
            for (int i = 0; i < 12; i++)
                outs[i] = (float *)xmalloc(sizeof(float) * (size_t)N);
            double t0 = now_s();
            for (int it = 0; it < 5; it++) {
                void *b = cwick_metal_ds_batch_begin(0, 1);
                for (int i = 0; i < 12; i++)
                    cwick_metal_ds_batch_gemv(b, w16, ws, N, K, 0, N, af, asf,
                                              outs[i]);
                cwick_metal_ds_batch_commit(b);
                cwick_metal_ds_batch_wait(b);
            }
            double dt = (now_s() - t0) / 5.0;
            printf("bf16 gpu only, 12 per batch: %.3f ms/batch  %.1f GB/s  "
                   "worst rel %.3e\n", dt * 1e3,
                   (double)(12 * N * K * 2) / dt / 1e9,
                   rel_err(outs[0], ref, N, NULL));
            for (int i = 0; i < 12; i++) free(outs[i]);
        }
        free(w); free(ws); free(w16); free(got); free(ref);
    }
    free(x); free(aq); free(as); free(af); free(asf);
}

/* ------------------------------------------------------------------ */
/* the dense shapes a decode step actually issues                       */
/*                                                                      */
/* A step is ~600 bf16 projections, not one big matrix, and what the    */
/* GPU is worth on them depends entirely on whether its round trip is    */
/* small next to the matrix. This measures exactly that: the production  */
/* entry point, the real shapes, and whatever CWICK_DS_GPU says. Run it  */
/* twice -- CWICK_DS_GPU=0 and =1 -- and compare per shape.              */
/* ------------------------------------------------------------------ */

static void bench_shapes(int reps) {
    static const struct { const char *name; int64_t N, K; } sh[] = {
        { "q          ",  1280, 5120 },
        { "kv         ",   512, 5120 },
        { "wo_a (x8)  ",  1024, 4096 },
        { "wo_b       ",  5120, 8192 },
        { "shared w2  ",  5120, 2304 },
        { "shared w1/3",  2304, 5120 },
        { "indexer    ", 32768, 1280 },
        { "compressor ",  4096, 1280 },
        { "engram     ", 25600, 6144 },
        { "head       ",  1024, 5120 },
    };
    const int nsh = (int)(sizeof(sh) / sizeof(sh[0]));
    printf("dense shapes, %d reps each, CWICK_DS_GPU=%s\n", reps,
           getenv("CWICK_DS_GPU") ? getenv("CWICK_DS_GPU") : "(default)");
    printf("%-12s %8s %8s %10s %10s\n", "shape", "n", "k", "ms", "GB/s");
    for (int s = 0; s < nsh; s++) {
        int64_t N = sh[s].N, K = sh[s].K, nb = K / 32;
        uint16_t *w16 = (uint16_t *)xmalloc(sizeof(uint16_t) * (size_t)(N * K));
        uint8_t *ws = (uint8_t *)xmalloc((size_t)((N / 32) * nb));
        uint8_t *aq = (uint8_t *)xmalloc((size_t)K);
        uint8_t *asc = (uint8_t *)xmalloc((size_t)nb);
        float *out = (float *)xmalloc(sizeof(float) * (size_t)N);
        float *x = (float *)xmalloc(sizeof(float) * (size_t)K);
        for (int64_t i = 0; i < K; i++)
            x[i] = (float)(((i * 37) % 61) - 30) * 0.37f;
        for (int64_t b = 0; b < nb; b++)
            asc[b] = ds_act_quant_block(x + b * 32, aq + b * 32);
        uint8_t *w8 = (uint8_t *)xmalloc((size_t)(N * K));
        fill_random(w8, (size_t)(N * K));
        for (int64_t i = 0; i < N * K; i++)
            if ((w8[i] & 0x7F) == 0x7F) w8[i] = 0;
        ds_fp8_to_bf16_row(w8, w16, N * K);
        for (int64_t i = 0; i < (N / 32) * nb; i++) ws[i] = (uint8_t)(0x72 + (i % 10));
        ds_qbf16_gemv(w16, ws, aq, asc, N, K, out);      /* warm-up + GPU init */
        double t0 = now_s();
        for (int it = 0; it < reps; it++) ds_qbf16_gemv(w16, ws, aq, asc, N, K, out);
        double dt = (now_s() - t0) / (double)reps;
        printf("%-12s %8lld %8lld %10.3f %10.1f\n", sh[s].name,
               (long long)N, (long long)K, dt * 1e3,
               (double)(N * K * 2) / dt / 1e9);
        free(w16); free(ws); free(w8); free(aq); free(asc); free(out); free(x);
    }
}

/* ------------------------------------------------------------------ */
/* the original form: old against new, at a shape of the caller's      */
/* choosing, rotating through `rotate` weight copies                   */
/* ------------------------------------------------------------------ */

static int bench_compare(int64_t N, int64_t K, int iters, int rotate) {
    int64_t nb = K / 32;
    int64_t copies = rotate > 0 ? rotate : 1;
    size_t w4_bytes = (size_t)(N * K / 2) * (size_t)copies;
    size_t w4s_bytes = (size_t)(N * nb) * (size_t)copies;
    size_t w8_bytes = (size_t)(N * K) * (size_t)copies;
    size_t w8s_bytes = (size_t)((N / 32) * nb) * (size_t)copies;
    uint8_t *a = calloc((size_t)K, 1);
    uint8_t *as = calloc((size_t)nb, 1);
    uint8_t *w4 = calloc(w4_bytes, 1);
    uint8_t *w4s = calloc(w4s_bytes, 1);
    uint8_t *w8 = calloc(w8_bytes, 1);
    uint8_t *w8s = calloc(w8s_bytes, 1);
    float *out = calloc((size_t)N, sizeof(float));
    if (!a || !as || !w4 || !w4s || !w8 || !w8s || !out) return 1;

    srand(1234);
    for (int64_t i = 0; i < K; i++) a[i] = (uint8_t)(rand() & 0x7E);
    for (int64_t i = 0; i < nb; i++) as[i] = (uint8_t)(0x7A + (rand() & 3));
    for (size_t i = 0; i < w4_bytes; i++) w4[i] = (uint8_t)(rand() & 0xFF);
    for (size_t i = 0; i < w4s_bytes; i++) w4s[i] = (uint8_t)(0x74 + (rand() & 0xF));
    for (size_t i = 0; i < w8_bytes; i++) w8[i] = (uint8_t)(rand() & 0x7E);
    for (size_t i = 0; i < w8s_bytes; i++) w8s[i] = (uint8_t)(0x74 + (rand() & 0xF));

    double macs = (double)N * (double)K;
    /* Fill the conversion tables before any kernel runs: the old reference
     * loop below reads them directly and would otherwise measure zeros. */
    (void)ds_fp4_to_f32(0);
    struct { const char *name; void (*fn)(const uint8_t *, const uint8_t *,
                                          const uint8_t *, const uint8_t *,
                                          int64_t, int64_t, float *); } cases[] = {
        { "fp4 old", old_fp4_gemv },
        { "fp4 new", ds_fp4_gemv },
        { "fp8 old", old_fp8_gemv },
        { "fp8 new", ds_fp8_gemv },
    };

    printf("shape [%lld, %lld], %d iterations, %.1f MMAC each, %d weight copies\n",
           (long long)N, (long long)K, iters, macs / 1e6, rotate ? rotate : 1);
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        int fp4 = strncmp(cases[c].name, "fp4", 3) == 0;
        const uint8_t *w = fp4 ? w4 : w8;
        const uint8_t *ws = fp4 ? w4s : w8s;
        cases[c].fn(a, as, w, ws, N, K, out);      /* warm-up */
        double t0 = now_s();
        for (int it = 0; it < iters; it++) {
            /* rotating copies all hold the same bytes, so the result stays
             * comparable; only the cache footprint changes */
            int r = rotate > 0 ? (it % rotate) : 0;
            cases[c].fn(a, as, w + (size_t)r * (fp4 ? N * K / 2 : N * K),
                        ws + (size_t)r * (fp4 ? N * nb : (N / 32) * nb),
                        N, K, out);
        }
        double dt = now_s() - t0;
        printf("%-8s %8.2f ms/call  %7.2f GMAC/s  out[0]=%.6f\n",
               cases[c].name, dt * 1000.0 / iters,
               macs * (double)iters / dt / 1e9, (double)out[0]);
    }
    free(a); free(as); free(w4); free(w4s); free(w8); free(w8s); free(out);
    return 0;
}

int main(int argc, char **argv) {
    const char *which = "fp4";
    int reps = 20;
    int threads = 0;
    int distinct = 32;
    int flags = 0;

    for (int i = 1; i < argc; i++)
        if (argv[i][0] == '-' && argv[i][1] == '-') { flags = 1; break; }

    if (!flags) {
        /* the original positional form */
        int64_t N = argc > 1 ? atol(argv[1]) : 2304;
        int64_t K = argc > 2 ? atol(argv[2]) : 5120;
        int iters = argc > 3 ? atoi(argv[3]) : 20;
        int rotate = argc > 4 ? atoi(argv[4]) : 0;
        cwick_pool_init();
        return bench_compare(N, K, iters, rotate);
    }

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gemv") && i + 1 < argc) which = argv[++i];
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--distinct") && i + 1 < argc) distinct = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rows") && i + 1 < argc) {
            char buf[16];
            snprintf(buf, sizeof(buf), "%d", atoi(argv[++i]));
            setenv("DS_GEMV_ROWS", buf, 1);
        } else {
            fprintf(stderr, "unknown argument '%s'\n", argv[i]);
            return 2;
        }
    }
    if (threads > 0) {
        char buf[16];
        snprintf(buf, sizeof(buf), "%d", threads);
        setenv("CWICK_THREADS", buf, 1);
    }
    cwick_pool_init();

    if (!strcmp(which, "check")) {
        check_kernels();
        return 0;
    }
    if (!strcmp(which, "gpucheck")) {
        gpu_check();
        return 0;
    }
    if (!strcmp(which, "shapes")) {
        /* The GPU needs the same init the engine does; gpu_check does it for
         * its own mode, so do it here too rather than depending on the caller. */
        if (cwick_metal_init()) cwick_metal_keepalive_start();
        bench_shapes(reps);
        return 0;
    }
    if (!strcmp(which, "head")) {
        bench_bf16_head(reps);
        return 0;
    }
    if (!strcmp(which, "compare")) {
        return bench_compare(2304, 5120, reps, distinct > 1 ? distinct : 0);
    }
    if (!strcmp(which, "fp4")) {
        /* gate/up: moe_inter rows over dim columns */
        shape s = { "fp4 gate/up", 2304, 5120, 0, (size_t)2304 * 5120 / 2,
                    (int64_t)2304 * 5120 };
        bench_q(&s, reps, distinct);
        /* down: dim rows over moe_inter columns */
        shape d = { "fp4 down", 5120, 2304, 0, (size_t)5120 * 2304 / 2,
                    (int64_t)5120 * 2304 };
        bench_q(&d, reps, distinct);
        return 0;
    }
    if (!strcmp(which, "fp8")) {
        shape s = { "fp8", 2304, 5120, 1, (size_t)2304 * 5120,
                    (int64_t)2304 * 5120 };
        bench_q(&s, reps, distinct);
        return 0;
    }
    if (!strcmp(which, "bf16")) {
        /* the same weights widened at load, read as bf16 instead of fp8 */
        shape s = { "bf16 widened", 2304, 5120, 2,
                    (size_t)2304 * 5120 * sizeof(uint16_t),
                    (int64_t)2304 * 5120 };
        bench_q(&s, reps, distinct);
        return 0;
    }
    fprintf(stderr, "unknown gemv '%s' (want fp4|fp8|bf16|head|check|compare)\n",
            which);
    return 2;
}
