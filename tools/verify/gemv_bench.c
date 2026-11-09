/* microbench: bf16 / int8 / fp8 decode GEMV - scalar vs NEON vs pooled.
 * Usage: ./gemv_bench [threads]  (0 = force the serial inline path)
 */
#include "cwick.h"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void ref_bf16_rows(const uint16_t *w, const float *x, float *out,
                          int64_t k, int64_t c0, int64_t c1) {
    for (int64_t c = c0; c < c1; c++) {
        const uint16_t *row = w + c * k;
        double acc = 0;
        for (int64_t j = 0; j < k; j++)
            acc += (double)cwick_bf16_to_f32(row[j]) * (double)x[j];
        out[c] = (float)acc;
    }
}

static uint16_t f2bf(float v) {
    uint32_t u; memcpy(&u, &v, 4); return (uint16_t)(u >> 16);
}

int main(int argc, char **argv) {
    const char *th = (argc > 1) ? argv[1] : "12";
    setenv("CWICK_THREADS", th, 1);
    printf("CWICK_THREADS=%s\n", th);
    cwick_pool_init();
    printf("pool threads: %d\n", cwick_pool_threads());

    int64_t cols = 8192, k = 4096;
    int iters = 10;

    float *x = malloc(k * sizeof(float));
    float *out = malloc(cols * sizeof(float));
    float *ref = malloc(cols * sizeof(float));
    for (int64_t i = 0; i < k; i++) x[i] = (float)((i % 17) - 8) * 0.03125f;

    uint16_t *wbf = malloc(cols * k * sizeof(uint16_t));
    for (int64_t i = 0; i < cols * k; i++)
        wbf[i] = f2bf((float)((i % 251) - 125) * 0.0078125f);

    ref_bf16_rows(wbf, x, ref, k, 0, cols);

    double t0 = now();
    for (int it = 0; it < iters; it++) cwick_gemv_bf16(out, k, cols, x, wbf);
    double tp = now() - t0;
    double maxerr = 0;
    for (int64_t c = 0; c < cols; c++) {
        double d = fabs(ref[c] - out[c]) / (fabs(ref[c]) + 1e-6);
        if (d > maxerr) maxerr = d;
    }
    printf("bf16  [8192,4096] %8.2f ms/call  rel.err=%.2e\n", tp / iters * 1e3, maxerr);

    /* scalar double reference cost (the pre-optimization kernel) */
    t0 = now();
    for (int it = 0; it < iters; it++) ref_bf16_rows(wbf, x, out, k, 0, cols);
    double tsc = now() - t0;
    printf("bf16  scalar-double reference %8.2f ms/call   -> NEON+pool speedup %.2fx\n",
           tsc / iters * 1e3, tsc / tp);

    int8_t *wi8 = malloc(cols * k);
    uint16_t *sc = malloc(cols * sizeof(uint16_t));
    for (int64_t i = 0; i < cols * k; i++) wi8[i] = (int8_t)((i % 97) - 48);
    for (int64_t c = 0; c < cols; c++) sc[c] = f2bf(0.0078125f * (1.0f + (c % 5)));
    t0 = now();
    for (int it = 0; it < iters; it++) cwick_gemv_int8(out, k, cols, x, wi8, sc);
    tp = now() - t0;
    maxerr = 0;
    for (int64_t c = 0; c < cols; c++) {
        double r = 0;
        for (int64_t j = 0; j < k; j++) r += (double)wi8[c * k + j] * (double)x[j];
        r *= cwick_bf16_to_f32(sc[c]);
        double d = fabs(r - out[c]) / (fabs(r) + 1e-6);
        if (d > maxerr) maxerr = d;
    }
    printf("int8  [8192,4096] %8.2f ms/call  rel.err=%.2e\n", tp / iters * 1e3, maxerr);

    int64_t ncb = k / 128;
    uint8_t *wf8 = malloc(cols * k);
    float *sinv = malloc((cols / 128) * ncb * sizeof(float));
    for (int64_t i = 0; i < cols * k; i++) wf8[i] = (uint8_t)(i % 200 + 20);
    for (int64_t i = 0; i < (cols / 128) * ncb; i++) sinv[i] = 0.5f + (i % 7) * 0.01f;
    t0 = now();
    for (int it = 0; it < iters; it++) cwick_gemv_fp8(out, k, cols, x, wf8, sinv, ncb);
    tp = now() - t0;
    printf("fp8   [8192,4096] %8.2f ms/call\n", tp / iters * 1e3);

    cwick_pool_shutdown();
    return 0;
}
