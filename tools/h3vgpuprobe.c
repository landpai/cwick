/* h3vgpuprobe — pure C check of the video Metal kernels.
 *
 * Compiles the production shader and compares each kernel with the CPU loop
 * on shaped inputs that mirror the DiT and the VAE. No checkpoint needed. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cwick.h"
#include "h3v_gpu.h"
#include "util.h"

static float *f32(int64_t n) { return (float *)malloc((size_t)n * sizeof(float)); }
static uint16_t *bf(int64_t n) { return (uint16_t *)malloc((size_t)n * sizeof(uint16_t)); }

static void cpu_gemm(float *out, const float *x, const float *w,
                     int rows, int k, int cols, int lda, int ldc) {
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            double acc = 0.0;
            for (int t = 0; t < k; t++)
                acc += (double)x[(size_t)r * lda + t] * w[(size_t)c * k + t];
            out[(size_t)r * ldc + c] = (float)acc;
        }
}

static void cpu_attn(float *out, const float *q, const float *k, const float *v,
                     int seq, int ld, int heads, int hd) {
    float scale = 1.0f / sqrtf((float)hd);
    for (int h = 0; h < heads; h++)
        for (int r = 0; r < seq; r++) {
            float mu = -INFINITY;
            for (int j = 0; j < seq; j++) {
                float s = 0.0f;
                for (int d = 0; d < hd; d++)
                    s += q[(size_t)r * ld + h * hd + d] * k[(size_t)j * ld + h * hd + d];
                if (s * scale > mu) mu = s * scale;
            }
            double den = 0.0;
            for (int j = 0; j < seq; j++) {
                float s = 0.0f;
                for (int d = 0; d < hd; d++)
                    s += q[(size_t)r * ld + h * hd + d] * k[(size_t)j * ld + h * hd + d];
                den += exp(s * scale - mu);
            }
            for (int d = 0; d < hd; d++) {
                double acc = 0.0;
                for (int j = 0; j < seq; j++) {
                    float s = 0.0f;
                    for (int t = 0; t < hd; t++)
                        s += q[(size_t)r * ld + h * hd + t] * k[(size_t)j * ld + h * hd + t];
                    acc += exp(s * scale - mu) * v[(size_t)j * ld + h * hd + d];
                }
                out[(size_t)r * ld + h * hd + d] = (float)(acc / den);
            }
        }
}

static double worst_diff(const float *a, const float *b, int64_t n) {
    double w = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > w) w = d;
    }
    return w;
}

static void fill(float *p, int64_t n) {
    for (int64_t i = 0; i < n; i++) p[i] = sinf((float)i * 0.37f) * 0.4f;
}

int main(int argc, char **argv) {
    if (!cwick_metal_backend_active()) { printf("no metal backend\n"); return 2; }
    if (!h3v_gpu_ready()) { printf("gpu init failed\n"); return 1; }
    printf("gpu ready, shader=%zu chars\n", strlen(h3v_gpu_shader_source()));
    int rc = 0;

    /* DiT-shaped projection: rows=128, k=256, cols=192 (tiles 64x64x16). */
    {
        int rows = 128, k = 256, cols = 192;
        float *x = f32((size_t)rows * k), *w = f32((size_t)cols * k);
        float *want = f32((size_t)rows * cols), *got = f32((size_t)rows * cols);
        fill(x, (size_t)rows * k); fill(w, (size_t)cols * k);
        cpu_gemm(want, x, w, rows, k, cols, k, cols);
        if (!h3v_gpu_gemm_f32(x, w, got, rows, k, cols, k, cols)) {
            printf("gemm_f32: dispatch failed\n"); rc = 1;
        } else {
            double d = worst_diff(want, got, (size_t)rows * cols);
            printf("gemm_f32   worst=%.6g\n", d);
            if (d > 0.02) { printf("  FAIL\n"); rc = 1; }
        }
        /* BF16 path: quantise the same weights to the checkpoint dtype. */
        uint16_t *wb = bf((size_t)cols * k);
        for (int i = 0; i < cols * k; i++) {
            uint32_t u; memcpy(&u, &w[i], 4);
            wb[i] = (uint16_t)(u >> 16);
        }
        float *wbf = f32((size_t)cols * k);
        for (int i = 0; i < cols * k; i++) wbf[i] = cwick_bf16_fast(wb[i]);
        cpu_gemm(want, x, wbf, rows, k, cols, k, cols);
        if (!h3v_gpu_gemm_bf16(x, wb, got, rows, k, cols, k, cols)) {
            printf("gemm_bf16: dispatch failed\n"); rc = 1;
        } else {
            double d = worst_diff(want, got, (size_t)rows * cols);
            printf("gemm_bf16  worst=%.6g\n", d);
            if (d > 0.05) { printf("  FAIL\n"); rc = 1; }
        }
        free(x); free(w); free(want); free(got); free(wb); free(wbf);
    }

    /* Release-layout QKV: weight rows are [q|k|v] per head. The kernel must
     * select one plane per dispatch, not three contiguous thirds. */
    {
        int rows = 64, k = 128, hd = 16, heads = 3;
        int cols = heads * hd, wrows = heads * 3 * hd;
        float *x = f32((size_t)rows * k), *w = f32((size_t)wrows * k);
        float *wp = f32((size_t)cols * k);
        float *want = f32((size_t)rows * cols), *got = f32((size_t)rows * cols);
        fill(x, (size_t)rows * k); fill(w, (size_t)wrows * k);
        uint16_t *wb = bf((size_t)wrows * k);
        for (int i = 0; i < wrows * k; i++) {
            uint32_t u; memcpy(&u, &w[i], 4);
            wb[i] = (uint16_t)(u >> 16);
        }
        for (int p = 0; p < 3; p++) {
            for (int h = 0; h < heads; h++)
                for (int d = 0; d < hd; d++)
                    for (int j = 0; j < k; j++)
                        wp[(h * hd + d) * k + j] = cwick_bf16_fast(wb[(h * 3 * hd + p * hd + d) * k + j]);
            cpu_gemm(want, x, wp, rows, k, cols, k, cols);
            if (!h3v_gpu_gemm_qkv(x, wb, got, rows, k, cols, wrows, k, cols, p, hd)) {
                printf("gemm_qkv part %d: dispatch failed\n", p); rc = 1;
            } else {
                double d = worst_diff(want, got, (size_t)rows * cols);
                printf("gemm_qkv%d   worst=%.6g\n", p, d);
                if (d > 0.05) { printf("  FAIL\n"); rc = 1; }
            }
        }
        free(x); free(w); free(wp); free(want); free(got); free(wb);
    }

    /* Strided operand: the SwiGLU leaves its result in the first half of a
     * 2*k-wide row. lda must be carried, not assumed equal to k. */
    {
        int rows = 96, k = 128, cols = 96, lda = 2 * k;
        float *x = f32((size_t)rows * lda), *w = f32((size_t)cols * k);
        float *want = f32((size_t)rows * cols), *got = f32((size_t)rows * cols);
        fill(x, (size_t)rows * lda); fill(w, (size_t)cols * k);
        cpu_gemm(want, x, w, rows, k, cols, lda, cols);
        if (!h3v_gpu_gemm_f32(x, w, got, rows, k, cols, lda, cols)) {
            printf("gemm_f32 stride: dispatch failed\n"); rc = 1;
        } else {
            double d = worst_diff(want, got, (size_t)rows * cols);
            printf("gemm_f32 stride worst=%.6g\n", d);
            if (d > 0.02) { printf("  FAIL\n"); rc = 1; }
        }
        free(x); free(w); free(want); free(got);
    }

    /* Attention, both head widths. seq is not a multiple of the 64-row tile
     * so the tail path is exercised too. */
    for (int hd = 64; hd <= 128; hd *= 2) {
        int seq = 80, heads = hd == 128 ? 8 : 4, ld = heads * hd;
        float *q = f32((size_t)seq * ld), *k = f32((size_t)seq * ld);
        float *v = f32((size_t)seq * ld), *want = f32((size_t)seq * ld);
        float *got = f32((size_t)seq * ld);
        /* Keep scores small so BF16 staging of Q/K remains faithful. */
        for (int i = 0; i < seq * ld; i++) {
            q[i] = sinf((float)i * 0.11f) * 0.1f;
            k[i] = cosf((float)i * 0.13f) * 0.1f;
            v[i] = sinf((float)i * 0.07f) * 0.5f;
        }
        cpu_attn(want, q, k, v, seq, ld, heads, hd);
        if (!h3v_gpu_attention(q, k, v, got, seq, ld, heads, hd)) {
            printf("attn%d: dispatch failed\n", hd); rc = 1;
        } else {
            double d = worst_diff(want, got, (size_t)seq * ld);
            printf("attn%-3d    worst=%.6g\n", hd, d);
            if (d > 0.02) {
                int64_t wi = 0;
                for (int64_t i = 1; i < (int64_t)seq * ld; i++)
                    if (fabs((double)want[i] - got[i]) > fabs((double)want[wi] - got[wi])) wi = i;
                printf("  worst at row %lld col %lld (head %lld): want=%.6f got=%.6f\n",
                       (long long)(wi / ld), (long long)(wi % ld),
                       (long long)((wi % ld) / hd), want[wi], got[wi]);
                for (int d2 = 0; d2 < hd && d2 < 8; d2++)
                    printf("  row0 h0 c%d: want=%.6f got=%.6f\n", d2,
                           want[d2], got[d2]);
                rc = 1;
            }
        }
        free(q); free(k); free(v); free(want); free(got);
    }

    h3v_gpu_shutdown();
    printf(rc ? "FAIL\n" : "OK\n");
    (void)argc; (void)argv;
    return rc;
}
