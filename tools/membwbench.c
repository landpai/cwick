/* membwbench — what does this host's memory bus actually deliver?
 *
 * The decode ceiling for a dense checkpoint is bytes-per-step divided by
 * read bandwidth, so every argument about "is 40 tok/s possible" rests on one
 * number. This tool measures it three ways, because the three answers differ
 * by a factor:
 *
 *   word   4-byte scalar loads, one accumulator        — the naive loop
 *   wide   16-byte loads (uint32x4), no conversion     — the transport
 *   bf16   16-byte loads converted to f32 and summed   — what a BF16 GEMV
 *                                                        inner loop asks for
 *
 * Each thread walks its own contiguous slice. Every variant is run for
 * 1..16 threads on the same buffer, so a flat line means the bus is
 * saturated and a line that keeps climbing means the previous variant was
 * issuing-bound, not bandwidth-bound.
 *
 *   cc -O2 -o tools/membwbench tools/membwbench.c -lpthread
 *   ./tools/membwbench [gib] [repeats]
 */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

typedef enum { K_WORD, K_WIDE, K_BF16, K_COUNT } kind_t;
static const char *KIND_NAME[K_COUNT] = {"word", "wide", "bf16"};

#if defined(__ARM_NEON)
/* BF16 -> f32 is a 16-bit shift into the upper half of each lane. */
static inline float32x4_t bf16x8_to_f32x4_lo(uint16x8_t h) {
    return vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h), 16));
}
static inline float32x4_t bf16x8_to_f32x4_hi(uint16x8_t h) {
    return vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h), 16));
}
#endif

typedef struct {
    const uint8_t *base;
    size_t bytes;
    kind_t kind;
    double sink;      /* keeps the compiler from deleting the loop */
} job_t;

static volatile double g_sink;   /* consumed after each run, never printed */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void *worker(void *arg) {
    job_t *j = (job_t *)arg;
    double acc = 0.0;
    size_t n = j->bytes;
    const uint8_t *p = j->base;

    switch (j->kind) {
    case K_WORD: {
        const uint32_t *w = (const uint32_t *)p;
        uint32_t a = 0;
        for (size_t i = 0; i < n / 4; i++) a += w[i];
        acc = (double)a;
        break;
    }
    case K_WIDE: {
#if defined(__ARM_NEON)
        uint32x4_t a = vdupq_n_u32(0);
        size_t i = 0;
        for (; i + 64 <= n; i += 64) {
            uint32x4_t v0 = vld1q_u32((const uint32_t *)(p + i));
            uint32x4_t v1 = vld1q_u32((const uint32_t *)(p + i + 16));
            uint32x4_t v2 = vld1q_u32((const uint32_t *)(p + i + 32));
            uint32x4_t v3 = vld1q_u32((const uint32_t *)(p + i + 48));
            a = vaddq_u32(a, vaddq_u32(vaddq_u32(v0, v1), vaddq_u32(v2, v3)));
        }
        acc = (double)vaddvq_u32(a);
#else
        const uint32_t *w = (const uint32_t *)p;
        uint32_t a = 0;
        for (size_t i = 0; i < n / 4; i++) a += w[i];
        acc = (double)a;
#endif
        break;
    }
    case K_BF16: {
#if defined(__ARM_NEON)
        /* BF16 is the upper half of an f32: widen to u32 shifted left 16 and
         * reinterpret. That is exactly what a BF16 GEMV does with every weight
         * it reads, so this variant includes the conversion cost the real
         * kernel pays. */
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        size_t i = 0;
        for (; i + 64 <= n; i += 64) {
            uint16x8_t h0 = vld1q_u16((const uint16_t *)(p + i));
            uint16x8_t h1 = vld1q_u16((const uint16_t *)(p + i + 16));
            uint16x8_t h2 = vld1q_u16((const uint16_t *)(p + i + 32));
            uint16x8_t h3 = vld1q_u16((const uint16_t *)(p + i + 48));
            a0 = vaddq_f32(a0, bf16x8_to_f32x4_lo(h0));
            a0 = vaddq_f32(a0, bf16x8_to_f32x4_hi(h0));
            a0 = vaddq_f32(a0, bf16x8_to_f32x4_lo(h1));
            a0 = vaddq_f32(a0, bf16x8_to_f32x4_hi(h1));
            a1 = vaddq_f32(a1, bf16x8_to_f32x4_lo(h2));
            a1 = vaddq_f32(a1, bf16x8_to_f32x4_hi(h2));
            a1 = vaddq_f32(a1, bf16x8_to_f32x4_lo(h3));
            a1 = vaddq_f32(a1, bf16x8_to_f32x4_hi(h3));
        }
        float32x4_t s = vaddq_f32(a0, a1);
        acc = (double)(vgetq_lane_f32(s, 0) + vgetq_lane_f32(s, 1) +
                       vgetq_lane_f32(s, 2) + vgetq_lane_f32(s, 3));
#else
        const uint16_t *w = (const uint16_t *)p;
        uint32_t a = 0;
        for (size_t i = 0; i < n / 2; i++) a += w[i];
        acc = (double)a;
#endif
        break;
    }
    default:
        break;
    }
    j->sink = acc;
    return NULL;
}

int main(int argc, char **argv) {
    size_t gib = argc > 1 ? (size_t)strtoul(argv[1], NULL, 10) : 8;
    int repeats = argc > 2 ? atoi(argv[2]) : 3;
    if (gib < 1) gib = 1;
    size_t bytes = gib << 30;

    uint8_t *buf = (uint8_t *)malloc(bytes);
    if (!buf) {
        fprintf(stderr, "cannot allocate %zu GiB\n", gib);
        return 1;
    }
    /* Materialise the pages: an untouched anonymous mapping reads from the
     * shared zero page and would report an absurd bandwidth. */
    memset(buf, 0x3f, bytes);

    printf("%zu GiB buffer, %d repeat(s), %ld CPUs\n",
           gib, repeats, (long)sysconf(_SC_NPROCESSORS_ONLN));
    printf("%-6s %-6s %10s %14s\n", "kernel", "thr", "GB/s", "GiB/thread");

    const int thread_counts[] = {1, 2, 4, 6, 8, 10, 12, 16};
    for (kind_t k = 0; k < K_COUNT; k++) {
        double best = 0.0;
        for (size_t t = 0; t < sizeof(thread_counts) / sizeof(thread_counts[0]); t++) {
            int nthr = thread_counts[t];
            double rate = 0.0;
            for (int r = 0; r < repeats; r++) {
                pthread_t th[16];
                job_t job[16];
                size_t chunk = bytes / (size_t)nthr;
                double t0 = now_s();
                for (int i = 0; i < nthr; i++) {
                    job[i].base = buf + chunk * (size_t)i;
                    job[i].bytes = chunk;
                    job[i].kind = k;
                    job[i].sink = 0.0;
                    pthread_create(&th[i], NULL, worker, &job[i]);
                }
                for (int i = 0; i < nthr; i++) {
                    pthread_join(th[i], NULL);
                    g_sink += job[i].sink;
                }
                double dt = now_s() - t0;
                double gbs = ((double)bytes / 1e9) / dt;
                if (gbs > rate) rate = gbs;
            }
            if (rate > best) best = rate;
            printf("%-6s %-6d %10.1f %14.2f\n", KIND_NAME[k], nthr, rate,
                   (double)bytes / 1e9 / (double)nthr);
        }
        printf("%-6s peak %10.1f GB/s\n\n", KIND_NAME[k], best);
    }
    free(buf);
    return 0;
}
