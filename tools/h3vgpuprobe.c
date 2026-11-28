/* h3vgpuprobe — pure C check of the video Metal kernels.
 *
 * Compiles the production shader and compares each kernel with the CPU loop
 * on shaped inputs that mirror the DiT and the VAE. No checkpoint needed. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

static void cpu_qk_prep(float *x, int seq, int ld, int heads, int hd,
                        const float *w, const float *cos, const float *sin) {
    for (int r = 0; r < seq; r++)
        for (int h = 0; h < heads; h++) {
            float *v = x + (size_t)r * ld + h * hd;
            double ss = 0.0;
            for (int d = 0; d < hd; d++) ss += (double)v[d] * v[d];
            float inv = 1.0f / sqrtf((float)(ss / hd) + 1e-5f);
            for (int d = 0; d < hd; d++) v[d] *= inv * w[d];
            for (int p = 0; p < 48; p++) {
                float a = v[p], b = v[p + 48];
                float c = cos[(size_t)r * 96 + p], s = sin[(size_t)r * 96 + p];
                v[p] = a * c - b * s;
                v[p + 48] = a * s + b * c;
            }
        }
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

    /* Convolution taps share one output. The plain F32 entry must replace the
     * tile; the accum entry must add to it, matching the CPU beta=1.0 path. */
    {
        int rows = 96, k = 64, cols = 128;
        float *x = f32((size_t)rows * k), *w = f32((size_t)cols * k);
        float *want = f32((size_t)rows * cols), *got = f32((size_t)rows * cols);
        fill(x, (size_t)rows * k); fill(w, (size_t)cols * k);
        cpu_gemm(want, x, w, rows, k, cols, k, cols);
        for (int i = 0; i < rows * cols; i++) want[i] *= 2.0f;
        for (int i = 0; i < rows * cols; i++) got[i] = 7.0f;
        if (!h3v_gpu_gemm_f32(x, w, got, rows, k, cols, k, cols) ||
            !h3v_gpu_gemm_f32_accum(x, w, got, rows, k, cols, k, cols)) {
            printf("gemm_f32 accum: dispatch failed\n"); rc = 1;
        } else {
            double d = worst_diff(want, got, (size_t)rows * cols);
            printf("gemm_f32 accum worst=%.6g\n", d);
            if (d > 0.02) { printf("  FAIL\n"); rc = 1; }
        }
        free(x); free(w); free(want); free(got);
    }

    /* Small activation tiles: routed experts usually contribute only a few
     * rows, so the 16-row kernel must match the same BF16 reference. */
    {
        int rows = 13, k = 256, cols = 192;
        float *x = f32((size_t)rows * k), *w = f32((size_t)cols * k);
        float *want = f32((size_t)rows * cols), *got = f32((size_t)rows * cols);
        fill(x, (size_t)rows * k); fill(w, (size_t)cols * k);
        uint16_t *wb = bf((size_t)cols * k);
        for (int i = 0; i < cols * k; i++) {
            uint32_t u; memcpy(&u, &w[i], 4);
            wb[i] = (uint16_t)(u >> 16);
        }
        for (int i = 0; i < cols * k; i++) w[i] = cwick_bf16_fast(wb[i]);
        cpu_gemm(want, x, w, rows, k, cols, k, cols);
        if (!h3v_gpu_gemm_bf16(x, wb, got, rows, k, cols, k, cols)) {
            printf("gemm_bf16_small: dispatch failed\n"); rc = 1;
        } else {
            double d = worst_diff(want, got, (size_t)rows * cols);
            printf("gemm_bf16_small worst=%.6g\n", d);
            if (d > 0.05) { printf("  FAIL\n"); rc = 1; }
        }
        free(x); free(w); free(want); free(got); free(wb);
    }

    /* Mixed small/large items in one concurrent batch. */
    {
        int rows[2] = {13, 65}, k = 256, cols = 192;
        h3v_gpu_bf16_item items[2];
        float *x[2], *w = f32((size_t)cols * k), *want[2], *got[2];
        uint16_t *wb = bf((size_t)cols * k);
        fill(w, (size_t)cols * k);
        for (int i = 0; i < cols * k; i++) {
            uint32_t u; memcpy(&u, &w[i], 4);
            wb[i] = (uint16_t)(u >> 16);
            w[i] = cwick_bf16_fast(wb[i]);
        }
        for (int j = 0; j < 2; j++) {
            x[j] = f32((size_t)rows[j] * k);
            want[j] = f32((size_t)rows[j] * cols);
            got[j] = f32((size_t)rows[j] * cols);
            fill(x[j], (size_t)rows[j] * k);
            cpu_gemm(want[j], x[j], w, rows[j], k, cols, k, cols);
            items[j] = (h3v_gpu_bf16_item){x[j], wb, got[j], rows[j], k, cols, k, cols};
        }
        if (!h3v_gpu_gemm_bf16_batch(items, 2)) {
            printf("gemm_bf16_batch: dispatch failed\n"); rc = 1;
        } else {
            double d = 0;
            for (int j = 0; j < 2; j++) {
                double dj = worst_diff(want[j], got[j], (size_t)rows[j] * cols);
                if (dj > d) d = dj;
            }
            printf("gemm_bf16_batch worst=%.6g\n", d);
            if (d > 0.05) { printf("  FAIL\n"); rc = 1; }
        }
        for (int j = 0; j < 2; j++) { free(x[j]); free(want[j]); free(got[j]); }
        free(w); free(wb);
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
        float *gotk = f32((size_t)rows * cols), *gotv = f32((size_t)rows * cols);
        float *outs[3] = {got, gotk, gotv};
        if (!h3v_gpu_gemm_qkv3(x, wb, got, gotk, gotv, rows, k, cols, wrows,
                               k, cols, hd)) {
            printf("gemm_qkv3: dispatch failed\n"); rc = 1;
        } else {
            double d = 0;
            for (int p = 0; p < 3; p++) {
                for (int h = 0; h < heads; h++)
                    for (int dd = 0; dd < hd; dd++)
                        for (int j = 0; j < k; j++)
                            wp[(h * hd + dd) * k + j] = cwick_bf16_fast(wb[(h * 3 * hd + p * hd + dd) * k + j]);
                cpu_gemm(want, x, wp, rows, k, cols, k, cols);
                double dp = worst_diff(want, outs[p], (size_t)rows * cols);
                if (dp > d) d = dp;
            }
            printf("gemm_qkv3  worst=%.6g\n", d);
            if (d > 0.05) { printf("  FAIL\n"); rc = 1; }
        }
        free(gotk); free(gotv);
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

    {
        int seq = 17, heads = 8, hd = 128, ld = heads * hd;
        size_t n = (size_t)seq * ld;
        float *q = f32(n), *k = f32(n), *wq = f32(hd), *wk = f32(hd);
        float *cos = f32((size_t)seq * 96), *sin = f32((size_t)seq * 96);
        float *wantq = f32(n), *wantk = f32(n);
        fill(q, n); fill(k, n); fill(wq, hd); fill(wk, hd);
        for (int i = 0; i < hd; i++) { wq[i] = 0.5f + fabsf(wq[i]); wk[i] = 0.5f + fabsf(wk[i]); }
        for (size_t i = 0; i < (size_t)seq * 96; i++) {
            cos[i] = cosf((float)i * 0.03f); sin[i] = sinf((float)i * 0.03f);
        }
        memcpy(wantq, q, n * sizeof(float)); memcpy(wantk, k, n * sizeof(float));
        cpu_qk_prep(wantq, seq, ld, heads, hd, wq, cos, sin);
        cpu_qk_prep(wantk, seq, ld, heads, hd, wk, cos, sin);
        if (!h3v_gpu_qk_prep(q, k, wq, wk, cos, sin, seq, ld, heads, hd, 1, 0)) {
            printf("qk_prep: dispatch failed\n"); rc = 1;
        } else {
            double d = worst_diff(wantq, q, n);
            double dk = worst_diff(wantk, k, n);
            if (dk > d) d = dk;
            printf("qk_prep    worst=%.6g\n", d);
            if (d > 0.001) { printf("  FAIL\n"); rc = 1; }
        }
        free(q); free(k); free(wq); free(wk); free(cos); free(sin);
        free(wantq); free(wantk);
    }

    {
        int half = 512, n = 500;
        float *x = f32((size_t)2 * half), *want = f32((size_t)2 * half);
        fill(x, (size_t)2 * half);
        memcpy(want, x, (size_t)2 * half * sizeof(float));
        for (int i = 0; i < n; i++) {
            float g = x[i], v = x[half + i];
            want[i] = v * (g / (1.0f + expf(-g)));
        }
        if (!h3v_gpu_swiglu(x, n, half, 0)) {
            printf("swiglu: dispatch failed\n"); rc = 1;
        } else {
            double d = worst_diff(want, x, (size_t)2 * half);
            printf("swiglu     worst=%.6g\n", d);
            if (d > 0.001) { printf("  FAIL\n"); rc = 1; }
        }
        free(x); free(want);
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

    if (getenv("H3VGPU_BENCH")) {
        int seq = 4096, heads = 8, hd = 128, ld = heads * hd;
        size_t n = (size_t)seq * ld;
        float *q = f32(n), *k = f32(n), *v = f32(n), *o = f32(n);
        for (size_t i = 0; i < n; i++) {
            q[i] = sinf((float)i * 0.011f) * 0.1f;
            k[i] = cosf((float)i * 0.013f) * 0.1f;
            v[i] = sinf((float)i * 0.007f) * 0.5f;
        }
        h3v_gpu_attention(q, k, v, o, seq, ld, heads, hd);
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < 3; i++)
            h3v_gpu_attention(q, k, v, o, seq, ld, heads, hd);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = ((double)(t1.tv_sec - t0.tv_sec) * 1000.0 +
                     (double)(t1.tv_nsec - t0.tv_nsec) / 1e6) / 3.0;
        printf("attn_bench seq=%d heads=%d hd=%d %.3f ms\n", seq, heads, hd, ms);
        const char *dump = getenv("H3VGPU_ATTN_DUMP");
        if (dump && *dump) {
            FILE *f = fopen(dump, "wb");
            if (f) { fwrite(o, sizeof(float), n, f); fclose(f); }
        }
        free(q); free(k); free(v); free(o);
    }

    if (getenv("H3VGPU_GEMM_BENCH")) {
        int rows = 4096, k = 5376, cols = 5376;
        size_t xn = (size_t)rows * k, wn = (size_t)cols * k;
        float *x = f32(xn), *o = f32(xn);
        uint16_t *w = bf(wn);
        for (size_t i = 0; i < xn; i++) x[i] = sinf((float)i * 0.013f) * 0.1f;
        for (size_t i = 0; i < wn; i++) {
            uint32_t u; float f = sinf((float)i * 0.017f) * 0.1f;
            memcpy(&u, &f, 4); w[i] = (uint16_t)(u >> 16);
        }
        h3v_gpu_gemm_bf16(x, w, o, rows, k, cols, k, cols);
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < 3; i++)
            h3v_gpu_gemm_bf16(x, w, o, rows, k, cols, k, cols);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = ((double)(t1.tv_sec - t0.tv_sec) * 1000.0 +
                     (double)(t1.tv_nsec - t0.tv_nsec) / 1e6) / 3.0;
        printf("gemm_bench rows=%d k=%d cols=%d %.3f ms\n", rows, k, cols, ms);
        free(x); free(o); free(w);
    }

    h3v_gpu_shutdown();
    printf(rc ? "FAIL\n" : "OK\n");
    (void)argc; (void)argv;
    return rc;
}
