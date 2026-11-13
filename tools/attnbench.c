/* attnbench — what does this machine do with a (queries x keys) score pass and
 * a (keys x dims) output pass, at the shape qwen3.5 actually runs?
 *
 * The prefill profile says attention is the only term that grows with context,
 * and that it runs at ~1.5 GMAC/s per thread while the same thread does 16.8
 * GMAC/s on a plain f32 dot (bfdotbench). A factor of ten needs an explanation
 * before anything is rewritten, so this measures the two loops in the shapes
 * they could take:
 *
 *   A  score pass as shipped: eight queries per key row, one dot() call per
 *      query, result written into a row `keys` floats long
 *   B  score pass, four queries per key row with the widened key held in
 *      registers and every query's four accumulators live across the row
 *   C  score pass through cblas_sgemm on an f32 copy of the keys
 *   D  output pass as shipped: 1 KiB accumulator loaded and stored per key
 *   E  output pass, four queries x 16 dims with the accumulator in registers
 *      for a whole key run
 *   F  output pass through cblas_sgemm
 *
 * One thread, so the numbers are comparable to bfdotbench; multiply by the
 * pool size for a wall estimate. Every variant computes the same MAC count,
 * which is what makes the ratios meaningful.
 *
 *   make tools/attnbench && ./tools/attnbench [keys] [reps]
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <Accelerate/Accelerate.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#define NQ   512      /* queries in a prefill block */
#define HD   256      /* head_dim */
#define KVH  4        /* kv heads */
#define HEADS 16      /* query heads */
#define HDIM (KVH * HD)

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1e3 + (double)tv.tv_usec / 1e3;
}

static inline float bf1(const uint16_t *p) {
    uint32_t u = (uint32_t)p[0] << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

#if defined(__aarch64__)
static inline float32x4_t bf16x4(uint16x4_t h) {
    return vreinterpretq_f32_u32(vshll_n_u16(h, 16));
}

/* The shipped dot: four accumulators over 16 dims per iteration. */
static float dot_ship(const float *q, const uint16_t *k, int n) {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint16x8_t u0 = vld1q_u16(k + i);
        uint16x8_t u1 = vld1q_u16(k + i + 8);
        a0 = vfmaq_f32(a0, bf16x4(vget_low_u16(u0)), vld1q_f32(q + i));
        a1 = vfmaq_f32(a1, bf16x4(vget_high_u16(u0)), vld1q_f32(q + i + 4));
        a2 = vfmaq_f32(a2, bf16x4(vget_low_u16(u1)), vld1q_f32(q + i + 8));
        a3 = vfmaq_f32(a3, bf16x4(vget_high_u16(u1)), vld1q_f32(q + i + 12));
    }
    return vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
}

/* B: four queries per key row. The key is widened once and used four times,
 * and the 16 accumulators stay live for the whole row -- the same summation
 * order per query as dot_ship, so the answer is bit-identical. */
static inline void dots4(const float *const q[4], const uint16_t *k, int n,
                         float out[4]) {
    float32x4_t a[4][4];
    for (int j = 0; j < 4; j++)
        for (int t = 0; t < 4; t++) a[j][t] = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint16x8_t u0 = vld1q_u16(k + i);
        uint16x8_t u1 = vld1q_u16(k + i + 8);
        float32x4_t k0 = bf16x4(vget_low_u16(u0));
        float32x4_t k1 = bf16x4(vget_high_u16(u0));
        float32x4_t k2 = bf16x4(vget_low_u16(u1));
        float32x4_t k3 = bf16x4(vget_high_u16(u1));
        for (int j = 0; j < 4; j++) {
            const float *qj = q[j];
            a[j][0] = vfmaq_f32(a[j][0], k0, vld1q_f32(qj + i));
            a[j][1] = vfmaq_f32(a[j][1], k1, vld1q_f32(qj + i + 4));
            a[j][2] = vfmaq_f32(a[j][2], k2, vld1q_f32(qj + i + 8));
            a[j][3] = vfmaq_f32(a[j][3], k3, vld1q_f32(qj + i + 12));
        }
    }
    for (int j = 0; j < 4; j++)
        out[j] = vaddvq_f32(vaddq_f32(vaddq_f32(a[j][0], a[j][1]),
                                      vaddq_f32(a[j][2], a[j][3])));
}

/* E: output pass with the accumulator in registers. Four queries x 16 dims =
 * 16 live vectors, and the key run is *inside* the dim strip, so the strip is
 * read once per key for all four queries instead of the accumulator being
 * loaded and stored per key. */
static inline void av_strip(float *const o[4], const uint16_t *const v, int kv,
                            int kvh, int d0, int64_t keys, const float *w) {
    float32x4_t s[4][4];
    for (int j = 0; j < 4; j++)
        for (int t = 0; t < 4; t++) s[j][t] = vdupq_n_f32(0.0f);
    for (int64_t i = 0; i < keys; i++) {
        uint16x8_t u0 = vld1q_u16(v + i * kv + (size_t)kvh * 256 + d0);
        uint16x8_t u1 = vld1q_u16(v + i * kv + (size_t)kvh * 256 + d0 + 8);
        float32x4_t v0 = bf16x4(vget_low_u16(u0)), v1 = bf16x4(vget_high_u16(u0));
        float32x4_t v2 = bf16x4(vget_low_u16(u1)), v3 = bf16x4(vget_high_u16(u1));
        for (int j = 0; j < 4; j++) {
            const float32x4_t wv = vdupq_n_f32(w[i]);
            s[j][0] = vfmaq_f32(s[j][0], wv, v0);
            s[j][1] = vfmaq_f32(s[j][1], wv, v1);
            s[j][2] = vfmaq_f32(s[j][2], wv, v2);
            s[j][3] = vfmaq_f32(s[j][3], wv, v3);
        }
    }
    for (int j = 0; j < 4; j++)
        for (int t = 0; t < 4; t++) vst1q_f32(o[j] + d0 + t * 4, s[j][t]);
}
#endif

int main(int argc, char **argv) {
    int64_t keys = argc > 1 ? atoll(argv[1]) : 11000;
    int reps = argc > 2 ? atoi(argv[2]) : 3;
    const int64_t kv = HDIM;

    uint16_t *K = calloc((size_t)keys * (size_t)kv, sizeof(uint16_t));
    float *Q = calloc((size_t)NQ * (size_t)kv, sizeof(float));
    float *sc = calloc((size_t)keys, sizeof(float));
    float *O = calloc((size_t)NQ * (size_t)kv, sizeof(float));
    for (int64_t i = 0; i < keys * kv; i++) {
        float x = (float)((i * 37) % 101 - 50) * 0.02f;
        uint32_t u;
        memcpy(&u, &x, 4);
        K[i] = (uint16_t)(u >> 16);
    }
    for (size_t i = 0; i < (size_t)NQ * (size_t)kv; i++)
        Q[i] = (float)((i * 17) % 67 - 33) * 0.03f;

    /* One layer's work: 16 query heads, 512 queries, `keys` keys, 256 dims. */
    const double mac = (double)NQ * (double)HEADS * (double)keys * (double)HD;
    double sink = 0.0;
    printf("keys=%lld  queries=%d  heads=%d  head_dim=%d  (one thread)\n",
           (long long)keys, NQ, HEADS, HD);
    printf("each pass: %.2f GMAC\n", mac / 1e9);

#if !defined(__aarch64__)
    printf("aarch64 only\n");
    return 1;
#else
    /* ---- A: shipped score pass ---- */
    double t0 = now_ms();
    for (int r = 0; r < reps; r++) {
        for (int h = 0; h < HEADS; h++) {
            const int kvh = h / 4;
            for (int q0 = 0; q0 < NQ; q0 += 8)
                for (int64_t s = 0; s < keys; s++) {
                    const uint16_t *krow = K + (size_t)s * (size_t)kv +
                                           (size_t)kvh * HD;
                    float acc = 0.0f;
                    for (int i = 0; i < 8; i++)
                        acc += dot_ship(Q + (size_t)(q0 + i) * (size_t)kv +
                                            (size_t)h * HD, krow, HD);
                    sc[s] += acc * 0.0625f;
                }
        }
        sink += sc[0];
    }
    double tA = (now_ms() - t0) / reps;

    /* ---- B: four queries per key row, key in registers ---- */
    t0 = now_ms();
    for (int r = 0; r < reps; r++) {
        for (int h = 0; h < HEADS; h++) {
            const int kvh = h / 4;
            for (int q0 = 0; q0 < NQ; q0 += 4) {
                const float *qp[4];
                for (int i = 0; i < 4; i++)
                    qp[i] = Q + (size_t)(q0 + i) * (size_t)kv + (size_t)h * HD;
                float four[4];
                for (int64_t s = 0; s < keys; s++) {
                    const uint16_t *krow = K + (size_t)s * (size_t)kv +
                                           (size_t)kvh * HD;
                    dots4(qp, krow, HD, four);
                    sc[s] += (four[0] + four[1] + four[2] + four[3]) * 0.0625f;
                }
            }
        }
        sink += sc[0];
    }
    double tB = (now_ms() - t0) / reps;

    /* ---- C: cblas score pass, one head at a time ---- */
    float *Kf = malloc((size_t)keys * (size_t)kv * sizeof(float));
    for (int64_t i = 0; i < keys * kv; i++) Kf[i] = bf1(K + i);
    float *S = malloc((size_t)NQ * (size_t)keys * sizeof(float));
    t0 = now_ms();
    for (int r = 0; r < reps; r++) {
        for (int h = 0; h < HEADS; h++) {
            const int kvh = h / 4;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        (int)NQ, (int)keys, HD, 0.0625f,
                        Q + (size_t)h * HD, (int)kv,
                        Kf + (size_t)kvh * HD, (int)kv,
                        0.0f, S, (int)keys);
        }
        sink += S[0];
    }
    double tC = (now_ms() - t0) / reps;

    /* ---- D: shipped output pass ---- */
    t0 = now_ms();
    for (int r = 0; r < reps; r++) {
        for (int h = 0; h < HEADS; h++) {
            const int kvh = h / 4;
            for (int q0 = 0; q0 < NQ; q0 += 8)
                for (int64_t s = 0; s < keys; s++) {
                    const uint16_t *vrow = K + (size_t)s * (size_t)kv +
                                           (size_t)kvh * HD;
                    const float w = sc[s];
                    for (int i = 0; i < 8; i++) {
                        float *o = O + (size_t)(q0 + i) * (size_t)kv +
                                   (size_t)h * HD;
                        for (int d = 0; d < HD; d++) o[d] += w * bf1(vrow + d);
                    }
                }
        }
        sink += O[0];
    }
    double tD = (now_ms() - t0) / reps;

    /* ---- E: output pass, accumulator in registers ---- */
    t0 = now_ms();
    for (int r = 0; r < reps; r++) {
        for (int h = 0; h < HEADS; h++) {
            const int kvh = h / 4;
            for (int q0 = 0; q0 < NQ; q0 += 4) {
                float *op[4];
                for (int i = 0; i < 4; i++)
                    op[i] = O + (size_t)(q0 + i) * (size_t)kv + (size_t)h * HD;
                for (int d0 = 0; d0 < HD; d0 += 16)
                    av_strip(op, K, (int)kv, kvh, d0, keys, sc);
            }
        }
        sink += O[0];
    }
    double tE = (now_ms() - t0) / reps;

    /* ---- F: cblas output pass ---- */
    float *Vf = Kf;   /* same data, same layout */
    t0 = now_ms();
    for (int r = 0; r < reps; r++) {
        for (int h = 0; h < HEADS; h++) {
            const int kvh = h / 4;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        (int)NQ, HD, (int)keys, 1.0f,
                        S, (int)keys, Vf + (size_t)kvh * HD, (int)kv,
                        0.0f, O + (size_t)h * HD, (int)kv);
        }
        sink += O[0];
    }
    double tF = (now_ms() - t0) / reps;

    printf("  score pass\n");
    printf("    A shipped 8q/row      %8.1f ms  %6.2f GMAC/s\n", tA,
           mac / (tA * 1e6));
    printf("    B 4q, key in regs     %8.1f ms  %6.2f GMAC/s  %.2fx A\n", tB,
           mac / (tB * 1e6), tA / tB);
    printf("    C cblas sgemm         %8.1f ms  %6.2f GMAC/s  %.2fx A\n", tC,
           mac / (tC * 1e6), tA / tC);
    printf("  output pass\n");
    printf("    D shipped per-key     %8.1f ms  %6.2f GMAC/s\n", tD,
           mac / (tD * 1e6));
    printf("    E 4q, reg accumulator %8.1f ms  %6.2f GMAC/s  %.2fx D\n", tE,
           mac / (tE * 1e6), tD / tE);
    printf("    F cblas sgemm         %8.1f ms  %6.2f GMAC/s  %.2fx D\n", tF,
           mac / (tF * 1e6), tD / tF);
    printf("(sink %.3f)\n", sink);
    return 0;
#endif
}
