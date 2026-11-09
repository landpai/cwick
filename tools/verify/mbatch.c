/* microbench: does one command buffer carrying N dispatches beat N command
 * buffers carrying one dispatch each?
 *
 * This is the whole question behind the batched expert GEMV. A single
 * [2048,4096] FP8 matrix costs ~105 us of GPU work and ~300-700 us of
 * host-side round trip, so if Metal pipelines dispatches inside one command
 * buffer the batch is a large win; if it serialises them with the same
 * latency it is not, and the honest answer is to leave the code alone.
 *
 *   ./mbatch [ndispatch] [reps]
 */
#include "cwick.h"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static float *ka_x, *ka_out, *ka_sinv;
static pthread_t ka_tid;

static volatile int g_ka_run = 0;

/* Keepalive thread: dispatch a small GEMV every 500 us so the GPU (and the
 * memory subsystem) has real work during the idle gaps. Used to test whether
 * the idle-gap penalty measured below is power management that a keepalive
 * can defeat, or something a keepalive cannot touch. */
static void *ka_thread(void *arg) {
    (void)arg;
    const char *kb = getenv("CWICK_MB_KA");
    int reps = kb ? atoi(kb) : 4;         /* small GEMVs per tick */
    while (g_ka_run) {
        for (int i = 0; i < reps && g_ka_run; i++) {
            void *b = cwick_metal_batch_begin();
            if (!b) break;
            /* [128, 4096] FP8 = 512 KiB, arena-backed */
            if (!cwick_metal_batch_gemv(b, (uint8_t *)arg, ka_sinv, ka_x,
                                       ka_out, 4096, 128, 32, 1, 128))
                break;
            cwick_metal_batch_end(b);
        }
        usleep(500);
    }
    return NULL;
}

int main(int argc, char **argv) {
    int N  = argc > 1 ? atoi(argv[1]) : 16;
    int R  = argc > 2 ? atoi(argv[2]) : 20;
    int cols = argc > 3 ? atoi(argv[3]) : 2048;
    int k = 4096, ncb = k / 128;

    cwick_metal_force(1);

    /* One arena-backed weight matrix, reused by every dispatch: that is what
     * the real decode loop does when a layer's experts are already resident.
     * CWICK_MB_SPREAD=GiB allocates a much larger buffer and gives each
     * dispatch its own 8 MiB slice, which is what a cache MISS looks like. */
    size_t wbytes = (size_t)cols * k;
    size_t sbytes = (size_t)(cols / 128) * ncb * sizeof(float);
    const char *spread_env = getenv("CWICK_MB_SPREAD");
    size_t spread = spread_env ? (size_t)(atof(spread_env) * 1073741824.0) : 0;
    if (spread < wbytes) spread = wbytes;
    uint8_t *wbase = cwick_shmem_alloc(spread);
    uint8_t *w = wbase;
    float *sinv = cwick_shmem_alloc(sbytes);
    if (!wbase || !sinv) { fprintf(stderr, "shmem alloc failed\n"); return 1; }
    if (getenv("CWICK_MB_RANDOM")) {
        /* real FP8 weights are random-ish; i%251 is a regular pattern that
         * may interact differently with the LUT bank conflicts */
        uint64_t rs = 0x9E3779B97F4A7C15ull;
        for (size_t i = 0; i < wbytes; i++) {
            rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
            w[i] = (uint8_t)(rs >> 56);
        }
    } else {
        for (size_t i = 0; i < wbytes; i++) w[i] = (uint8_t)(i % 251);
    }
    for (size_t i = 0; i < (size_t)(cols / 128) * ncb; i++) sinv[i] = 0.5f;

    float *x   = cwick_xmalloc((size_t)k * sizeof(float));
    float **outs = cwick_xmalloc((size_t)N * sizeof(float *));
    for (int i = 0; i < k; i++) x[i] = (float)((i % 9) - 4) * 0.125f;
    for (int i = 0; i < N; i++) {
        outs[i] = cwick_xmalloc((size_t)cols * sizeof(float));
        memset(outs[i], 0, (size_t)cols * sizeof(float));
    }

    /* keepalive scratch: a small arena-backed matrix the keepalive dispatches */
    ka_x = cwick_xmalloc((size_t)4096 * sizeof(float));
    ka_out = cwick_xmalloc((size_t)128 * sizeof(float));
    ka_sinv = cwick_shmem_alloc((size_t)(128 / 128) * 32 * sizeof(float));
    for (int i = 0; i < 4096; i++) ka_x[i] = 0.25f;
    for (int i = 0; i < 32; i++) ka_sinv[i] = 0.5f;
    uint8_t *ka_w = (uint8_t *)wbase;   /* first 512 KiB of the spread buffer */
    (void)ka_w;

    /* warm up both paths */
    for (int i = 0; i < N; i++)
        cwick_metal_gemv(w, sinv, x, outs[i], k, cols, ncb, 1, cols);
    void *b = cwick_metal_batch_begin();
    for (int i = 0; i < N; i++)
        cwick_metal_batch_gemv(b, w, sinv, x, outs[i], k, cols, ncb, 1, cols);
    cwick_metal_batch_end(b);

    /* ---- N separate command buffers ---- */
    double t0 = now();
    for (int r = 0; r < R; r++)
        for (int i = 0; i < N; i++)
            if (!cwick_metal_gemv(w, sinv, x, outs[i], k, cols, ncb, 1, cols)) {
                fprintf(stderr, "unbatched gemv failed\n"); return 1;
            }
    double tsep = (now() - t0) / (double)R;

    /* ---- one command buffer, N dispatches ---- */
    const char *fresh_env = getenv("CWICK_MB_FRESH");
    int fresh = fresh_env != NULL;
    const char *idle_env = getenv("CWICK_MB_IDLE");   /* ms of CPU idle between batches */
    double idle_ms = idle_env ? atof(idle_env) : 0.0;
    int ka_on = getenv("CWICK_MB_KA") != NULL;
    if (ka_on && idle_ms > 0) {
        g_ka_run = 1;
        pthread_create(&ka_tid, NULL, ka_thread, wbase);
    }
    double t_fresh = 0.0;
    t0 = now();
    for (int r = 0; r < R; r++) {
        void *bb = cwick_metal_batch_begin();
        if (!bb) { fprintf(stderr, "batch_begin failed\n"); return 1; }
        for (int i = 0; i < N; i++) {
            /* slice depends on the rep too, so a spread buffer is read cold
             * rather than re-reading the same L2-resident 128 MiB */
            size_t slice = ((size_t)r * (size_t)N + (size_t)i) % (spread / wbytes);
            uint8_t *wi = wbase + slice * wbytes;
            if (fresh) {
                /* mimic the stream reader: DMA-write the blob, then hand the
                 * same bytes to the GPU. Times the write separately. */
                double tf = now();
                memcpy(wi, w, wbytes);
                t_fresh += now() - tf;
            }
            if (!cwick_metal_batch_gemv(bb, wi, sinv, x, outs[i], k, cols, ncb, 1, cols)) {
                fprintf(stderr, "batch_gemv failed\n"); return 1;
            }
        }
        if (!cwick_metal_batch_end(bb)) { fprintf(stderr, "batch_end failed\n"); return 1; }
        if (idle_ms > 0) {
            /* Simulate the real decode loop: the GPU finishes a layer's batch,
             * then the CPU spends milliseconds streaming and running attention
             * before the next batch. If the GPU downclocks while idle, the
             * per-dispatch time will climb toward the real run's 112 us. */
            struct timespec ts;
            ts.tv_sec = (time_t)(idle_ms / 1000.0);
            ts.tv_nsec = (long)((idle_ms - ts.tv_sec * 1000.0) * 1e6);
            nanosleep(&ts, NULL);
        }
    }
    double tbat = (now() - t0) / (double)R;
    if (ka_on) { g_ka_run = 0; pthread_join(ka_tid, NULL); }

    printf("N=%d reps=%d\n", N, R);
    printf("  %d separate command buffers : %8.1f us/group  (%7.1f us/dispatch)\n",
           N, tsep * 1e6, tsep * 1e6 / N);
    printf("  1 command buffer, %2d dispatches: %8.1f us/group  (%7.1f us/dispatch)\n",
           N, tbat * 1e6, tbat * 1e6 / N);
    printf("  speedup: %.2fx\n", tsep / tbat);
    printf("  at 1080 dispatches/step: separate %.0f ms, batched %.0f ms\n",
           tsep * 1e6 / N * 1080 / 1000.0, tbat * 1e6 / N * 1080 / 1000.0);
    return 0;
}
