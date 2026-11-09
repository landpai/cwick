/* microbench: BF16 batched GEMV — does one command buffer carrying N
 * dispatches beat N command buffers carrying one each, for the resident
 * [8192,4096] BF16 projections the attention path issues?
 *
 * This is the question behind cwick_metal_bf16_batch_*: the GPU beats the CPU
 * pool ~2.4x on this shape (311 us vs 745 us) but a commit+wait round trip
 * costs hundreds of us, so batching only pays if Metal pipelines the
 * dispatches inside one command buffer.
 *
 *   ./mbf16batch [ndispatch] [reps] [idle_ms]
 */
#include "cwick.h"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 3;      /* dispatches per group */
    int R = argc > 2 ? atoi(argv[2]) : 50;     /* groups */
    int idle_ms = argc > 3 ? atoi(argv[3]) : 0;
    const int64_t k = 4096, cols = 8192;       /* one KDA q/k/v projection */

    if (cwick_metal_force(1) != 0) {
        fprintf(stderr, "metal unavailable\n");
        return 1;
    }
    /* The serial path is opt-in; the bench exercises both halves of it. */
    setenv("CWICK_METAL_BF16", "1", 1);

    size_t wbytes = (size_t)cols * k * sizeof(uint16_t);
    uint16_t *w = malloc(wbytes);
    float *x = malloc((size_t)k * sizeof(float));
    float *out = malloc((size_t)cols * sizeof(float));
    if (!w || !x || !out) { fprintf(stderr, "oom\n"); return 1; }
    for (size_t i = 0; i < wbytes / 2; i++) w[i] = (uint16_t)(i * 2654435761u >> 13);
    for (int64_t i = 0; i < k; i++) x[i] = (float)((i % 17) - 8) * 0.05f;

    /* warm-up: one of each so neither side pays first-touch cost */
    if (!cwick_metal_linear_bf16(w, x, out, k, cols)) {
        fprintf(stderr, "bf16 serial path refused (not eligible?)\n");
        return 1;
    }
    {
        void *bb = cwick_metal_bf16_batch_begin();
        if (!bb || !cwick_metal_bf16_batch_add(bb, w, x, out, k, cols) ||
            !cwick_metal_bf16_batch_end(bb)) {
            fprintf(stderr, "bf16 batch path refused\n");
            return 1;
        }
    }

    /* Optionally hold the GPU clock up through the idle gaps, exactly like
     * the decode loop does (CWICK_KEEPALIVE=0 there, default on here). */
    int ka = getenv("BENCH_KA") ? atoi(getenv("BENCH_KA")) : 1;
    if (ka) cwick_metal_keepalive_start();

    struct timespec its;
    its.tv_sec = (time_t)(idle_ms / 1000.0);
    its.tv_nsec = (long)((idle_ms - its.tv_sec * 1000.0) * 1e6);
    double t0 = now();
    for (int r = 0; r < R; r++) {
        for (int i = 0; i < N; i++)
            if (!cwick_metal_linear_bf16(w, x, out, k, cols)) {
                fprintf(stderr, "serial failed\n"); return 1;
            }
        if (idle_ms > 0) nanosleep(&its, NULL);
    }
    double tsep = (now() - t0) / (double)R;

    t0 = now();
    for (int r = 0; r < R; r++) {
        void *bb = cwick_metal_bf16_batch_begin();
        if (!bb) { fprintf(stderr, "begin failed\n"); return 1; }
        for (int i = 0; i < N; i++)
            if (!cwick_metal_bf16_batch_add(bb, w, x, out, k, cols)) {
                fprintf(stderr, "add failed\n"); return 1;
            }
        if (!cwick_metal_bf16_batch_end(bb)) {
            fprintf(stderr, "end failed\n"); return 1;
        }
        if (idle_ms > 0) {
            struct timespec ts;
            ts.tv_sec = (time_t)(idle_ms / 1000.0);
            ts.tv_nsec = (long)((idle_ms - ts.tv_sec * 1000.0) * 1e6);
            nanosleep(&ts, NULL);
        }
    }
    double tbat = (now() - t0) / (double)R;
    if (ka) cwick_metal_keepalive_stop();

    printf("N=%d reps=%d idle=%dms  ([%lld,%lld] BF16)\n", N, R, idle_ms,
           (long long)cols, (long long)k);
    printf("  %d separate command buffers : %8.1f us/group  (%7.1f us/dispatch)\n",
           N, tsep * 1e6, tsep * 1e6 / N);
    printf("  1 command buffer, %2d dispatches: %8.1f us/group  (%7.1f us/dispatch)\n",
           N, tbat * 1e6, tbat * 1e6 / N);
    printf("  speedup: %.2fx\n", tsep / tbat);
    /* 45 layers x 3 projections (KDA) or 2 (MLA) per decode step */
    printf("  at 124 batched items/step: separate %.1f ms, batched %.1f ms\n",
           tsep * 1e6 / N * 124 / 1000.0, tbat * 1e6 / N * 124 / 1000.0);
    free(w); free(x); free(out);
    return 0;
}
