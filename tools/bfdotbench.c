/* bfdotbench — does this machine actually execute a bf16 dot product faster
 * than it executes f32 multiplies?
 *
 * The FP4 expert kernel spends four f32 FMAs per 16 weights, and the plan for
 * the next factor of two is to replace them with bf16 dot products (8 MAC per
 * instruction instead of 4). That only pays if the hardware runs `bfdot` at a
 * higher MAC rate than `fma`; on some ARM implementations it runs at the same
 * rate, or is split, which would make the rewrite pointless. So: measure both
 * kernels, same length, same shape of loop, one thread.
 *
 *   make tools/bfdotbench && ./tools/bfdotbench [K] [reps]
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1e3 + (double)tv.tv_usec / 1e3;
}

int main(int argc, char **argv) {
    int64_t K = argc > 1 ? atoll(argv[1]) : 6144;
    int reps = argc > 2 ? atoi(argv[2]) : 200000;
    float *a = malloc(sizeof(float) * (size_t)K);
    float *b = malloc(sizeof(float) * (size_t)K);
    for (int64_t i = 0; i < K; i++) {
        a[i] = (float)((i % 13) - 6) * 0.125f;
        b[i] = (float)((i % 7) - 3) * 0.25f;
    }

#if !defined(__aarch64__)
    printf("aarch64 only\n");
    return 1;
#else
    float sink = 0.0f;

    /* ---- f32 FMA, four accumulators ---- */
    double t0 = now_ms();
    for (int r = 0; r < reps; r++) {
        float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0, s2 = s0, s3 = s0;
        int64_t i = 0;
        for (; i + 16 <= K; i += 16) {
            s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
            s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
            s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
            s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
        }
        sink += vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    }
    double tf32 = (now_ms() - t0) / reps;

    /* ---- bf16 dot product, four accumulators ----
     * Same number of multiplies per iteration, half the instructions: one
     * `bfdot` consumes eight bf16 pairs where two `fma` consume eight floats. */
    bfloat16_t *ab = (bfloat16_t *)malloc(sizeof(bfloat16_t) * (size_t)K);
    bfloat16_t *bb = (bfloat16_t *)malloc(sizeof(bfloat16_t) * (size_t)K);
    for (int64_t i = 0; i < K; i++) {
        ab[i] = vcvth_bf16_f32(a[i]);
        bb[i] = vcvth_bf16_f32(b[i]);
    }
    t0 = now_ms();
    for (int r = 0; r < reps; r++) {
        float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0, s2 = s0, s3 = s0;
        int64_t i = 0;
        for (; i + 32 <= K; i += 32) {
            s0 = vbfdotq_f32(s0, vld1q_bf16(ab + i), vld1q_bf16(bb + i));
            s1 = vbfdotq_f32(s1, vld1q_bf16(ab + i + 8), vld1q_bf16(bb + i + 8));
            s2 = vbfdotq_f32(s2, vld1q_bf16(ab + i + 16), vld1q_bf16(bb + i + 16));
            s3 = vbfdotq_f32(s3, vld1q_bf16(ab + i + 24), vld1q_bf16(bb + i + 24));
        }
        sink += vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    }
    double tbf = (now_ms() - t0) / reps;

    /* ---- f16 widening multiply-accumulate, as a reference point ----
     * `vfmlalq_low_f16` does 4 MAC per instruction on half precision; if that
     * is faster than bfdot, fp16 is the better target despite the narrower
     * exponent. */
    float16_t *ah = (float16_t *)malloc(sizeof(float16_t) * (size_t)K);
    float16_t *bh = (float16_t *)malloc(sizeof(float16_t) * (size_t)K);
    for (int64_t i = 0; i < K; i++) {
        ah[i] = (float16_t)a[i];
        bh[i] = (float16_t)b[i];
    }
    t0 = now_ms();
    for (int r = 0; r < reps; r++) {
        float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0, s2 = s0, s3 = s0;
        int64_t i = 0;
        for (; i + 32 <= K; i += 32) {
            s0 = vfmlalq_low_f16(s0, vld1q_f16(ah + i), vld1q_f16(bh + i));
            s1 = vfmlalq_high_f16(s1, vld1q_f16(ah + i), vld1q_f16(bh + i));
            s2 = vfmlalq_low_f16(s2, vld1q_f16(ah + i + 8), vld1q_f16(bh + i + 8));
            s3 = vfmlalq_high_f16(s3, vld1q_f16(ah + i + 8), vld1q_f16(bh + i + 8));
        }
        sink += vaddvq_f32(vaddq_f32(s0, s1)) +
                vaddvq_f32(vaddq_f32(s2, s3));
    }
    double tf16 = (now_ms() - t0) / reps;

    const double mac = (double)K;
    printf("K=%lld reps=%d  (one thread; GMAC/s and ns per K=%lld)\n",
           (long long)K, reps, (long long)K);
    printf("  f32  fma      %7.4f ms  %6.1f GMAC/s\n", tf32, mac / (tf32 * 1e6));
    printf("  bf16 bfdot    %7.4f ms  %6.1f GMAC/s   %.2fx f32\n", tbf,
           mac / (tbf * 1e6), tf32 / tbf);
    printf("  f16  fmlal    %7.4f ms  %6.1f GMAC/s   %.2fx f32\n", tf16,
           mac / (tf16 * 1e6), tf32 / tf16);
    printf("(sink %.3f)\n", sink);
    free(a); free(b); free(ab); free(bb); free(ah); free(bh);
    return 0;
#endif
}
