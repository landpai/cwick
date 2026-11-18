/* rt_dsp.h — numeric kernels shared by the speech stages.
 *
 * The talker's three checkpoints (LM, flow, vocoder) are all f32 state dicts
 * read through rt_torch, and all of them are stacks of small linear layers and
 * 1-D convolutions. Activations are time-major [T][C] here, which is the
 * transpose of the torch layout but makes every convolution tap one GEMM.
 *
 * Convolutions are stored [k][groups][cin_g][cout_g] so a tap is contiguous,
 * and weight_norm is folded at load: the vocoder ships g and v separately and
 * the effective kernel is g * v / ||v|| per output channel. */
#ifndef CWICK_RT_DSP_H
#define CWICK_RT_DSP_H

#include <stdint.h>

#include "cwick.h"
#include "rt_torch.h"

typedef struct {
    float *w; /* [k][groups][cin_g][cout_g] */
    float *b; /* [cout], NULL when the layer has no bias */
    int cin, cout, groups, k, dil, stride, padl, padr;
} rt_conv;

/* `name` is the module prefix ("conv_pre", "ups.0", …). Both the plain
 * `name.weight` and the weight_norm pair `name.parametrizations.weight.*` are
 * accepted. Returns -1 when the tensor is missing or has an unexpected rank. */
int rt_conv_load(rt_conv *c, const rt_pt *pt, const char *name, int groups,
                 int stride, int dil, int padl, int padr);
void rt_conv_free(rt_conv *c);
int rt_conv_len(const rt_conv *c, int T);
/* x [T][cin] -> y [rt_conv_len(c,T)][cout]. */
void rt_conv_apply(const rt_conv *c, const float *x, float *y, int T);

typedef struct {
    const float *w; /* [out][in] */
    const float *b; /* [out] or NULL */
    float *copy;    /* owned widening buffer, NULL when w/b point into pt */
    int in, out;
} rt_lin;

int rt_lin_load(rt_lin *l, const rt_pt *pt, const char *name, int bias);
void rt_lin_free(rt_lin *l);
/* x [m][in] -> y [m][out]. */
void rt_lin_apply(const rt_lin *l, const float *x, float *y, int m);
/* y[n] = w[n][k] * x[k], split over output rows unless RT_GEMV_THREADS=0. */
void rt_gemv(const float *w, const float *x, float *y, int k, int n);

/* The vocoder's STFT pair: n_fft 16, hop 4, periodic Hann, center=True with
 * reflect padding. `x` is N mono samples (N a multiple of 4); `y` is
 * [N/4+1][18], real bins 0..8 then imaginary bins 0..8, the channel order the
 * source downsamplers expect. */
int rt_stft16(const float *x, int N, float *y);
/* Inverse of the above: spec [TT][18] -> (TT-1)*4 samples, window-envelope
 * normalised and centre-cropped exactly as torch.istft does. */
void rt_istft16(const float *spec, int TT, float *out);

/* Deterministic normal/uniform stream. The reference seeds torch and slices a
 * pre-generated noise buffer; neither the seed nor the buffer is recoverable
 * from the checkpoint, so the speech stages use their own reproducible one. */
typedef struct { uint64_t s; } rt_rng;
void rt_rng_seed(rt_rng *r, uint64_t seed);
float rt_rng_uniform(rt_rng *r);
float rt_rng_normal(rt_rng *r);

void rt_silu(float *v, int n);
void rt_gelu_tanh(float *v, int n);
void rt_leaky(float *v, int n, float slope);
void rt_mish(float *v, int n);
/* LayerNorm without affine parameters, over the last axis of [m][n]. */
void rt_ln(float *dst, const float *src, int m, int n, float eps);

#endif /* CWICK_RT_DSP_H */