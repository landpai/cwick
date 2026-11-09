/*
 * ds_ops.c — see ds_ops.h. Every routine is a direct transcription of the
 * reference implementation's arithmetic; where the two could plausibly differ
 * (rounding direction, tie handling, the order of a normalization) the comment
 * says which choice was made and why.
 */
#include "ds_ops.h"
#include "ds.h" /* ds_e4m3_to_f32 / ds_f32_to_e4m3 for the compressed-KV scale */

#include <math.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* rotary                                                               */
/* ------------------------------------------------------------------ */

void ds_freqs_init(ds_freqs *f, int rope_dim, double base, double factor,
                   int original, int beta_fast, int beta_slow) {
    f->npairs = rope_dim / 2;
    double inv_log_base = 1.0 / log(base);
    for (int i = 0; i < f->npairs; i++) {
        double fr = pow(base, -(2.0 * i) / rope_dim); /* 1 / base^(2i/dim) */
        if (original > 0) {
            /* the dim whose wavelength completes `rotations` turns over the
             * training context; between beta_slow and beta_fast the frequency
             * is faded from its full value down to full/factor */
            double cd_fast = rope_dim * log((double)original / (beta_fast * 2.0 * M_PI)) *
                             inv_log_base / 2.0;
            double cd_slow = rope_dim * log((double)original / (beta_slow * 2.0 * M_PI)) *
                             inv_log_base / 2.0;
            int low = (int)floor(cd_fast);
            int high = (int)ceil(cd_slow);
            if (low < 0) low = 0;
            if (high > rope_dim - 1) high = rope_dim - 1;
            int span = (high - low) > 0 ? (high - low) : 1;
            double ramp = (i - low) / (double)span;
            if (ramp < 0.0) ramp = 0.0;
            if (ramp > 1.0) ramp = 1.0;
            double smooth = 1.0 - ramp;
            fr = fr / factor * (1.0 - smooth) + fr * smooth;
        }
        f->freq[i] = (float)fr;
    }
}

/* Rotate the first 2*npairs elements of x in place, adjacent pairs as complex
 * numbers (the reference passes the tail slice x[..., -rope_head_dim:], so the
 * caller hands us the start of that slice). inverse conjugates the rotation. */
void ds_rope(float *x, const ds_freqs *f, int64_t pos, int inverse) {
    for (int i = 0; i < f->npairs; i++) {
        double ang = (double)f->freq[i] * (double)pos;
        float c = (float)cos(ang), s = (float)sin(ang);
        if (inverse) s = -s;
        float a = x[2 * i], b = x[2 * i + 1];
        x[2 * i]     = a * c - b * s;
        x[2 * i + 1] = a * s + b * c;
    }
}

/* ------------------------------------------------------------------ */
/* norms and activations                                                */
/* ------------------------------------------------------------------ */

void ds_rmsnorm(float *y, const float *x, const float *w, int n, float eps) {
    double acc = 0.0;
    for (int i = 0; i < n; i++) acc += (double)x[i] * (double)x[i];
    float rsqrt_ = (float)(1.0 / sqrt(acc / n + eps));
    for (int i = 0; i < n; i++) y[i] = x[i] * rsqrt_ * (w ? w[i] : 1.0f);
}

/* The checkpoint stores every norm weight as bf16, and the reference's
 * RMSNorm returns `(self.weight * x).to(dtype)` with dtype == bf16 -- so the
 * product is formed in fp32 and then rounded, in that order. Multiplying by
 * the weight after the rsqrt (not before) keeps the rounding identical. */
void ds_rmsnorm_bf16w(float *y, const float *x, const uint16_t *w, int n, float eps) {
    double acc = 0.0;
    for (int i = 0; i < n; i++) acc += (double)x[i] * (double)x[i];
    float rsqrt_ = (float)(1.0 / sqrt(acc / n + eps));
    if (!w) {
        for (int i = 0; i < n; i++) y[i] = ds_bf16_round(x[i] * rsqrt_);
        return;
    }
    for (int i = 0; i < n; i++) {
        float wi = ds_bf16_to_f32(w[i]);
        y[i] = ds_bf16_round(wi * (x[i] * rsqrt_));
    }
}

float ds_bf16_round(float v) {
    union { float f; uint32_t u; } c;
    c.f = v;
    uint32_t lsb = (c.u >> 16) & 1u;
    c.u += 0x7FFFu + lsb;        /* round to nearest, ties to even */
    c.u &= 0xFFFF0000u;          /* then drop the 16 bits bf16 cannot hold */
    return c.f;
}

float ds_silu(float x) { return x / (1.0f + expf(-x)); }

void ds_softmax(float *x, int n) {
    float m = x[0];
    for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
    double s = 0.0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - m); s += x[i]; }
    float inv = (float)(1.0 / s);
    for (int i = 0; i < n; i++) x[i] *= inv;
}

/* ------------------------------------------------------------------ */
/* hyper-connections                                                    */
/* ------------------------------------------------------------------ */

void ds_hc_mixes(const float *x, const float *fn, const float *scale,
                 const float *base, int hc, int d, int iters, float eps,
                 float *pre, float *post, float *comb) {
    int mix = (2 + hc) * hc;
    int hd = hc * d;

    /* one statistic over the whole flattened stream, per token */
    double ms = 0.0;
    for (int i = 0; i < hd; i++) ms += (double)x[i] * (double)x[i];
    float rsqrt_ = (float)(1.0 / sqrt(ms / hd + 1e-20)); /* norm_eps, not hc_eps */

    float m[64];
    for (int j = 0; j < mix; j++) {
        double acc = 0.0;
        const float *row = fn + (size_t)j * hd;
        for (int i = 0; i < hd; i++) acc += (double)row[i] * (double)x[i];
        m[j] = (float)acc * rsqrt_;
    }

    for (int j = 0; j < hc; j++)
        pre[j] = 1.0f / (1.0f + expf(-(m[j] * scale[0] + base[j]))) + eps;
    for (int j = 0; j < hc; j++)
        post[j] = 2.0f / (1.0f + expf(-(m[j + hc] * scale[1] + base[j + hc])));

    for (int j = 0; j < hc; j++)
        for (int k = 0; k < hc; k++)
            comb[j * hc + k] = m[j * hc + k + 2 * hc] * scale[2] + base[j * hc + k + 2 * hc];

    /* Sinkhorn: row softmax + eps, then alternating row / column normalization.
     * The first column pass uses the sums of the softmaxed matrix, and every
     * later pass divides by (sum + eps), which is what keeps a dead row from
     * producing NaN. */
    for (int j = 0; j < hc; j++) {
        float mx = comb[j * hc];
        for (int k = 1; k < hc; k++) if (comb[j * hc + k] > mx) mx = comb[j * hc + k];
        float s = 0.0f;
        for (int k = 0; k < hc; k++) {
            comb[j * hc + k] = expf(comb[j * hc + k] - mx);
            s += comb[j * hc + k];
        }
        for (int k = 0; k < hc; k++) comb[j * hc + k] = comb[j * hc + k] / s + eps;
    }
    for (int k = 0; k < hc; k++) {
        float s = 0.0f;
        for (int j = 0; j < hc; j++) s += comb[j * hc + k];
        for (int j = 0; j < hc; j++) comb[j * hc + k] /= (s + eps);
    }
    for (int it = 1; it < iters; it++) {
        for (int j = 0; j < hc; j++) {
            float s = 0.0f;
            for (int k = 0; k < hc; k++) s += comb[j * hc + k];
            for (int k = 0; k < hc; k++) comb[j * hc + k] /= (s + eps);
        }
        for (int k = 0; k < hc; k++) {
            float s = 0.0f;
            for (int j = 0; j < hc; j++) s += comb[j * hc + k];
            for (int j = 0; j < hc; j++) comb[j * hc + k] /= (s + eps);
        }
    }
}

void ds_hc_pre(float *y, const float *x, const float *pre, int hc, int d) {
    for (int i = 0; i < d; i++) {
        double acc = 0.0;
        for (int j = 0; j < hc; j++) acc += (double)pre[j] * x[(size_t)j * d + i];
        y[i] = (float)acc;
    }
}

/* Expand back out and mix the residual in through comb:
 *   y[j] = post[j] * x + sum_i comb[i*hc + j] * res[i]
 * i.e. comb is a mixing matrix over the residual copies, summed along its
 * FIRST index, with the output copy selected by its second. Sinkhorn makes it
 * doubly stochastic, so a comb that were ignored would leave y ~= post*x + res;
 * the mixing is what the matrix is for. Two independent readings of the
 * reference agree on this shape: `sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)`
 * broadcasts comb's second axis against residual's hc axis, and a second
 * implementation of the same checkpoint addresses it as [dst, src]. */
void ds_hc_post(float *y, const float *x, const float *res, const float *post,
                const float *comb, int hc, int d) {
    for (int j = 0; j < hc; j++) {
        float *yj = y + (size_t)j * d;
        for (int i = 0; i < d; i++) yj[i] = post[j] * x[i];
    }
    for (int i = 0; i < hc; i++) {
        const float *ri = res + (size_t)i * d;
        for (int j = 0; j < hc; j++) {
            float c = comb[i * hc + j];
            if (c == 0.0f) continue;
            float *yj = y + (size_t)j * d;
            for (int t = 0; t < d; t++) yj[t] += c * ri[t];
        }
    }
}

/* ------------------------------------------------------------------ */
/* quantize / dequantize round trips                                    */
/* ------------------------------------------------------------------ */

float ds_fp4_round(float v) {
    /* e2m1 grid: 0 .5 1 1.5 2 3 4 6. Boundaries at .25 .75 1.25 1.75 2.5 3.5 5,
     * ties to the even index (0, 1, 1, 2, 2, 4, 4 as the rounded results). */
    float a = v < 0 ? -v : v;
    float r;
    if (a <= 0.25f) r = 0.0f;
    else if (a <= 0.75f) r = 1.0f;
    else if (a <= 1.25f) r = 1.0f;
    else if (a <= 1.75f) r = 2.0f;
    else if (a <= 2.5f) r = 2.0f;
    else if (a <= 3.5f) r = 4.0f;
    else if (a <= 5.0f) r = 4.0f;
    else r = 6.0f;
    return v < 0 ? -r : r;
}

static int log2_ceil_bits_(float x) {
    union { float f; uint32_t u; } c;
    c.f = x;
    int32_t e = (int32_t)((c.u >> 23) & 0xFFu) - 127;
    return e + ((c.u & 0x7FFFFFu) != 0 ? 1 : 0);
}

void ds_qdq_fp8_e8m0(float *x, int n) {
    for (int b = 0; b < n; b += 32) {
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) {
            float a = x[b + i] < 0 ? -x[b + i] : x[b + i];
            if (a > amax) amax = a;
        }
        if (amax < 1e-4f) amax = 1e-4f;
        int e = log2_ceil_bits_(amax * (1.0f / 448.0f));
        float s = ldexpf(1.0f, e);
        for (int i = 0; i < 32; i++) {
            float q = x[b + i] / s;
            if (q > 448.0f) q = 448.0f;
            if (q < -448.0f) q = -448.0f;
            x[b + i] = ds_e4m3_to_f32(ds_f32_to_e4m3(q)) * s;
        }
    }
}

void ds_qdq_fp4_e8m0(float *x, int n) {
    for (int b = 0; b < n; b += 32) {
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) {
            float a = x[b + i] < 0 ? -x[b + i] : x[b + i];
            if (a > amax) amax = a;
        }
        /* an all-zero block keeps a nonzero scale: 6 * 2^-126 */
        if (amax < 6.0f * ldexpf(1.0f, -126)) amax = 6.0f * ldexpf(1.0f, -126);
        int e = log2_ceil_bits_(amax * (1.0f / 6.0f));
        float s = ldexpf(1.0f, e);
        for (int i = 0; i < 32; i++) {
            float q = x[b + i] / s;
            if (q > 6.0f) q = 6.0f;
            if (q < -6.0f) q = -6.0f;
            x[b + i] = ds_fp4_round(q) * s;
        }
    }
}

void ds_qdq_fp4_e4m3(float *x, int n, int block) {
    for (int b = 0; b < n; b += block) {
        float amax = 0.0f;
        for (int i = 0; i < block; i++) {
            float a = x[b + i] < 0 ? -x[b + i] : x[b + i];
            if (a > amax) amax = a;
        }
        /* training's compressed KV: keep even an all-zero group's scale
         * nonzero, at 6 * 2^-9 */
        if (amax < 6.0f * ldexpf(1.0f, -9)) amax = 6.0f * ldexpf(1.0f, -9);
        /* the scale is a rounded e4m3 value here, not a power of two */
        float s = ds_e4m3_to_f32(ds_f32_to_e4m3(amax * (1.0f / 6.0f)));
        for (int i = 0; i < block; i++) {
            float q = x[b + i] / s;
            if (q > 6.0f) q = 6.0f;
            if (q < -6.0f) q = -6.0f;
            x[b + i] = ds_fp4_round(q) * s;
        }
    }
}
/* ---- top-k by score, ties keeping the lower index ---------------------- */

static void ds_heap_sift(float *h, int *hi, int size, int i) {
    for (;;) {
        int l = 2 * i + 1, r = l + 1, m = i;
        if (l < size && h[l] < h[m]) m = l;
        if (r < size && h[r] < h[m]) m = r;
        if (m == i) break;
        float tf = h[i]; h[i] = h[m]; h[m] = tf;
        int ti = hi[i]; hi[i] = hi[m]; hi[m] = ti;
        i = m;
    }
}

/* Keep the k best (score, index) pairs in a min-heap; write the chosen
 * indices to `out` in ascending order and return how many there were. */
int ds_topk_scores(const float *sc, int n, int k, int *out,
                   float *heap, int *heap_idx) {
    int size = 0;
    for (int t = 0; t < n; t++) {
        float v = sc[t];
        if (size == k) {
            if (v <= heap[0]) continue;
            heap[0] = v; heap_idx[0] = t;
            ds_heap_sift(heap, heap_idx, size, 0);
        } else {
            int i = size++;
            heap[i] = v; heap_idx[i] = t;
            while (i > 0) {
                int p = (i - 1) / 2;
                if (heap[p] <= heap[i]) break;
                float tf = heap[p]; heap[p] = heap[i]; heap[i] = tf;
                int ti = heap_idx[p]; heap_idx[p] = heap_idx[i]; heap_idx[i] = ti;
                i = p;
            }
        }
    }
    for (int i = 0; i < size; i++) out[i] = heap_idx[i];
    for (int i = 1; i < size; i++) {           /* insertion sort: k <= 512 */
        int v = out[i], j = i - 1;
        while (j >= 0 && out[j] > v) { out[j + 1] = out[j]; j--; }
        out[j + 1] = v;
    }
    return size;
}

