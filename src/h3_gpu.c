/* h3_gpu.c — GPU front end for the image transformer.
 *
 * The transformer's projections are `out = in @ W^T` with W stored BF16 and the
 * activation FP32. On the CPU that costs a dequantise pass into an f32 arena
 * before every GEMM: 66 GB of weights per sampling step turn into 66 GB read
 * plus 132 GB written and 132 GB read again, and a step measures 12 GB/s of
 * *weight* throughput.
 *
 * This unit hands the same product to the GPU with the checkpoint's BF16 bytes
 * untouched: the kernel stages a BF16 tile of the activation and a BF16 tile of
 * the weight straight from the mapping, so a step reads each weight once. The
 * activation loses mantissa bits in that staging step, which the sampler
 * tolerates the same way the released pipeline does — it runs BF16 end to end.
 *
 * Everything here is best-effort: any failure (no device, shader compile,
 * pipeline creation) permanently disables the path and the caller keeps using
 * the CPU, because image generation must not depend on the GPU coming up.
 */
#include "h3_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Implemented in h3_gpu.m, which owns the device, the queue and the pipelines. */
int h3m_init(void);
void h3m_shutdown(void);
int h3m_adopt(const void *base, size_t len);
int h3m_gemm(const float *a, const void *w, float *c,
             int rows, int k, int cols, int lda, int ldc);
int h3m_attention(const float *q, const float *k, const float *v, float *o,
                  int seq, int ld, int heads);
const char *h3m_shader_source(void);

static int g_state = -1;   /* -1 untried, 0 unavailable, 1 ready */

int h3_gpu_ready(void) {
    if (g_state < 0) g_state = h3m_init() ? 1 : 0;
    return g_state;
}

void h3_gpu_shutdown(void) {
    if (g_state == 1) h3m_shutdown();
    g_state = -1;
}

/* Best effort: a mapping that cannot be adopted simply keeps taking the
 * per-call path. */
void h3_gpu_adopt(const void *base, size_t len) {
    if (!base || !len) return;
    if (!h3_gpu_ready()) return;
    h3m_adopt(base, len);
}

/* One projection. `lda`/`ldc` are element strides, which lets a caller read a
 * per-head slice of a wider activation without copying it. */
int h3_gpu_gemm(const float *a, const void *w, float *c,
                int rows, int k, int cols, int lda, int ldc) {
    if (g_state != 1 || !a || !w || !c) return 0;
    if (rows <= 0 || k <= 0 || cols <= 0) return 0;
    /* The tile staging reads a whole 16-wide k block, so a projection whose
     * inner dimension is not a multiple of that has to stay on the CPU. */
    if (k % 16) return 0;
    return h3m_gemm(a, w, c, rows, k, cols, lda, ldc);
}

/* Full attention over one packed sequence. `ld` is the row stride of the
 * [seq][heads*head_dim] layout, which is what makes a per-head slice readable
 * without copying it.
 *
 * The kernel's tile geometry is fixed at 128 channels per head, so a model
 * whose head width is anything else stays on the CPU. The check is an equality
 * rather than a lower bound: a wider head would pass a bound and then read the
 * wrong columns. */
int h3_gpu_attention(const float *q, const float *k, const float *v, float *o,
                     int seq, int ld, int heads) {
    if (g_state != 1 || !q || !k || !v || !o) return 0;
    if (seq <= 0 || heads <= 0) return 0;
    if ((long)ld != (long)heads * 128) return 0;
    return h3m_attention(q, k, v, o, seq, ld, heads);
}
