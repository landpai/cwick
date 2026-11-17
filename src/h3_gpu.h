/* h3_gpu.h — Metal path for the image transformer's projections. */
#ifndef CWICK_H3_GPU_H
#define CWICK_H3_GPU_H

#include <stddef.h>
#include <stdint.h>

/* Brought up on first use; 0 means the CPU path stays in charge. */
int h3_gpu_ready(void);
void h3_gpu_shutdown(void);

/* Register a whole weight mapping. Projections whose bytes fall inside a
 * registered mapping reuse the one view taken here instead of building a new
 * one per call; a step runs several hundred projections over 66 GB, and each
 * unregistered call faults the pages of its own tensor and keeps the handle
 * for the life of the process. */
void h3_gpu_adopt(const void *base, size_t len);

/* c[rows][cols] = a[rows][k] @ w[cols][k]^T, w BF16.
 * `lda` / `ldc` are element strides. Returns 1 if the GPU produced the
 * result, 0 if the caller must fall back to the CPU. */
int h3_gpu_gemm(const float *a, const void *w, float *c,
                int rows, int k, int cols, int lda, int ldc);

/* Non-causal attention over one packed [seq][ld] sequence, per head. */
int h3_gpu_attention(const float *q, const float *k, const float *v, float *o,
                     int seq, int ld, int heads);

#endif
