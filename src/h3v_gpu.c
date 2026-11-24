/* h3v_gpu.c — CPU side of the video Metal backend.
 *
 * The object-C file owns the device, the queues and the pipelines; this file
 * owns the policy. A shape that the kernels cannot serve exactly is rejected
 * here so the caller's CPU path stays byte-for-byte in control, and a device
 * that cannot come up disables the path permanently for the process.
 */
#include "h3v_gpu.h"

#include <stdlib.h>

#include "cwick.h"
#include "log.h"
#include "util.h"

/* Implemented in h3v_gpu.m. */
int h3vm_init(void);
void h3vm_shutdown(void);
void h3vm_adopt(const void *base, size_t len);
int h3vm_gemm_bf16(const float *a, const void *w, float *c,
                   int rows, int k, int cols, int lda, int ldc);
int h3vm_gemm_bf16_batch(const h3v_gpu_bf16_item *items, int n);
int h3vm_gemm_bf16_qkv(const float *a, const void *w, float *c,
                       int rows, int k, int cols, int wrows, int lda, int ldc,
                       int part, int hd);
int h3vm_gemm_f32(const float *a, const float *w, float *c,
                  int rows, int k, int cols, int lda, int ldc);
int h3vm_attention(const float *q, const float *k, const float *v, float *o,
                   int seq, int ld, int heads, int hd);
int h3vm_inkling_decode_attn(const float *q, const float *k, const float *v,
                             const float *r, const float *rel, const float *sink,
                             float *o, int pos, int cache_rows, int ld,
                             int nh, int kvh, int d, int dr, int extent,
                             int window);
int h3vm_inkling_prefill_attn(const float *q, const float *k, const float *v,
                              const float *r, const float *rel, const float *sink,
                              float *o, int qstart, int qrows, int cache_rows,
                              int ld, int nh, int kvh, int d, int dr,
                              int extent, int window);

static int g_state = -1;   /* -1 untried, 0 unavailable, 1 ready */

int h3v_gpu_ready(void) {
    if (g_state < 0) g_state = h3vm_init() ? 1 : 0;
    return g_state;
}

void h3v_gpu_shutdown(void) {
    if (g_state == 1) h3vm_shutdown();
    g_state = -1;
}

int h3v_gpu_on(void) {
    static int cached = -1;
    if (cached >= 0) return cached;
    if (!cwick_metal_backend_active()) {
        cached = 0;
    } else if (getenv("CWICK_H3V_GPU") && !cwick_env_true("CWICK_H3V_GPU")) {
        cached = 0;
    } else {
        cached = h3v_gpu_ready() ? 1 : 0;
    }
    if (cached) CWICK_INFO("init", "video Metal backend up (DiT + VAE)");
    return cached;
}

void h3v_gpu_adopt(const void *base, size_t len) {
    if (!base || !len || !h3v_gpu_ready()) return;
    h3vm_adopt(base, len);
}

/* The tile loaders take a 16-wide K block, so an inner dimension that is not
 * a multiple of 16 would read past the row. Row strides must be large enough
 * for the activation and at least the output width for the result. */
static int gemm_shape_ok(int rows, int k, int cols, int lda, int ldc) {
    if (rows <= 0 || k <= 0 || cols <= 0) return 0;
    if (k % 16 || lda < k || ldc < cols) return 0;
    if (rows > 0x7FFFFFFFl || k > 0x7FFFFFFFl || cols > 0x7FFFFFFFl) return 0;
    return 1;
}

int h3v_gpu_gemm_bf16(const float *a, const void *w, float *c,
                      int rows, int k, int cols, int lda, int ldc) {
    if (g_state != 1 || !a || !w || !c) return 0;
    if (!gemm_shape_ok(rows, k, cols, lda, ldc)) return 0;
    return h3vm_gemm_bf16(a, w, c, rows, k, cols, lda, ldc);
}

int h3v_gpu_gemm_bf16_batch(const h3v_gpu_bf16_item *items, int n) {
    if (g_state != 1 || !items || n <= 0 || n > 64) return 0;
    for (int i = 0; i < n; i++) {
        if (!items[i].a || !items[i].w || !items[i].c ||
            !gemm_shape_ok(items[i].rows, items[i].k, items[i].cols,
                           items[i].lda, items[i].ldc)) return 0;
    }
    return h3vm_gemm_bf16_batch(items, n);
}

int h3v_gpu_gemm_f32(const float *a, const float *w, float *c,
                     int rows, int k, int cols, int lda, int ldc) {
    if (g_state != 1 || !a || !w || !c) return 0;
    if (!gemm_shape_ok(rows, k, cols, lda, ldc)) return 0;
    return h3vm_gemm_f32(a, w, c, rows, k, cols, lda, ldc);
}

int h3v_gpu_gemm_qkv(const float *a, const void *w, float *c,
                     int rows, int k, int cols, int wrows, int lda, int ldc,
                     int part, int hd) {
    if (g_state != 1 || !a || !w || !c) return 0;
    if (!gemm_shape_ok(rows, k, cols, lda, ldc)) return 0;
    if (part < 0 || part > 2 || hd <= 0 || hd % 16 || cols % hd ||
        wrows < (cols / hd) * 3 * hd) return 0;
    return h3vm_gemm_bf16_qkv(a, w, c, rows, k, cols, wrows, lda, ldc, part, hd);
}

int h3v_gpu_attention(const float *q, const float *k, const float *v, float *o,
                      int seq, int ld, int heads, int hd) {
    if (g_state != 1 || !q || !k || !v || !o) return 0;
    if (seq <= 0 || heads <= 0) return 0;
    /* One fragment lane covers eight channels; the kernels stage 64 or 128
     * channels per head and a different width would read the wrong columns. */
    if (hd != 64 && hd != 128) return 0;
    if ((long)ld < (long)heads * hd) return 0;
    return h3vm_attention(q, k, v, o, seq, ld, heads, hd);
}

int h3v_gpu_inkling_decode_attn(const float *q, const float *k, const float *v,
                                const float *r, const float *rel,
                                const float *sink, float *o, int pos,
                                int cache_rows, int ld, int nh, int kvh,
                                int d, int dr, int extent, int window) {
    if (g_state != 1 || !q || !k || !v || !r || !rel || !sink || !o) return 0;
    if (pos < 0 || cache_rows <= 0 || nh <= 0 || kvh <= 0 || nh % kvh ||
        d != 128 || dr <= 0 || extent <= 0 || window < 0 ||
        ld < kvh * d) return 0;
    return h3vm_inkling_decode_attn(q, k, v, r, rel, sink, o, pos, cache_rows,
                                    ld, nh, kvh, d, dr, extent, window);
}

int h3v_gpu_inkling_prefill_attn(const float *q, const float *k, const float *v,
                                 const float *r, const float *rel,
                                 const float *sink, float *o, int qstart,
                                 int qrows, int cache_rows, int ld, int nh,
                                 int kvh, int d, int dr, int extent, int window) {
    if (g_state != 1 || !q || !k || !v || !r || !rel || !sink || !o) return 0;
    if (qstart < 0 || qrows <= 0 || qrows > 64 || cache_rows <= 0 ||
        nh <= 0 || kvh <= 0 || nh % kvh || d != 128 || dr <= 0 ||
        extent <= 0 || window < 0 || ld < kvh * d) return 0;
    return h3vm_inkling_prefill_attn(q, k, v, r, rel, sink, o, qstart, qrows,
                                     cache_rows, ld, nh, kvh, d, dr,
                                     extent, window);
}
