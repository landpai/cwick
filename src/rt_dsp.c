/* rt_dsp.c — see rt_dsp.h. Everything here is plain f32 arithmetic on
 * Accelerate GEMMs; no GPU is required, which keeps the speech stages testable
 * on a machine without Metal. */
#include "rt_dsp.h"

#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

/* ------------------------------------------------------------ convolution --- */

int rt_conv_load(rt_conv *c, const rt_pt *pt, const char *name, int groups,
                 int stride, int dil, int padl, int padr) {
    char nm[256];
    memset(c, 0, sizeof(*c));
    snprintf(nm, sizeof(nm), "%s.parametrizations.weight.original1", name);
    const rt_tensor *v = rt_pt_get(pt, nm);
    const rt_tensor *g = NULL;
    if (v) {
        snprintf(nm, sizeof(nm), "%s.parametrizations.weight.original0", name);
        g = rt_pt_get(pt, nm);
        if (!g) return -1;
    } else {
        snprintf(nm, sizeof(nm), "%s.weight", name);
        v = rt_pt_get(pt, nm);
    }
    if (!v || v->ndim != 3) return -1;
    int cout = (int)v->shape[0], cing = (int)v->shape[1], k = (int)v->shape[2];
    if (cing * groups <= 0 || cout % groups) return -1;
    int coutg = cout / groups;
    c->cin = cing * groups;
    c->cout = cout;
    c->groups = groups;
    c->k = k;
    c->dil = dil;
    c->stride = stride;
    c->padl = padl;
    c->padr = padr;
    c->w = (float *)cwick_xmalloc((size_t)k * groups * cing * coutg * sizeof(float));

    /* weight_norm: w[o] = g[o] * v[o] / ||v[o]||, the norm taken over the
     * (in, kernel) axes of each output channel. */
    for (int o = 0; o < cout; o++) {
        float scale = 1.0f;
        if (g) {
            double ss = 0.0;
            for (int64_t i = 0; i < (int64_t)cing * k; i++) {
                double x = rt_tensor_f32(v, (int64_t)o * cing * k + i);
                ss += x * x;
            }
            float gn = rt_tensor_f32(g, o);
            scale = (float)(gn / (sqrt(ss) + 1e-12));
        }
        int grp = o / coutg, ol = o % coutg;
        for (int j = 0; j < k; j++)
            for (int i = 0; i < cing; i++) {
                float x = rt_tensor_f32(v, ((int64_t)o * cing + i) * k + j);
                c->w[(((size_t)j * groups + grp) * cing + i) * coutg + ol] = x * scale;
            }
    }

    snprintf(nm, sizeof(nm), "%s.bias", name);
    const rt_tensor *b = rt_pt_get(pt, nm);
    if (b) {
        if (b->numel != cout) return -1;
        c->b = (float *)cwick_xmalloc((size_t)cout * sizeof(float));
        for (int o = 0; o < cout; o++) c->b[o] = rt_tensor_f32(b, o);
    }
    return 0;
}

void rt_conv_free(rt_conv *c) {
    if (!c) return;
    free(c->w);
    free(c->b);
    memset(c, 0, sizeof(*c));
}

int rt_conv_len(const rt_conv *c, int T) {
    int span = (c->k - 1) * c->dil + 1;
    int tp = T + c->padl + c->padr;
    if (tp < span) return 0;
    return (tp - span) / c->stride + 1;
}

void rt_conv_apply(const rt_conv *c, const float *x, float *y, int T) {
    int tout = rt_conv_len(c, T);
    if (tout <= 0) return;
    int cin = c->cin, cout = c->cout, groups = c->groups;
    int cing = cin / groups, coutg = cout / groups;
    int tp = T + c->padl + c->padr;
    const float *xp = x;
    float *padded = NULL;
    if (c->padl || c->padr) {
        padded = (float *)cwick_xmalloc((size_t)tp * cin * sizeof(float));
        memset(padded, 0, (size_t)tp * cin * sizeof(float));
        memcpy(padded + (size_t)c->padl * cin, x, (size_t)T * cin * sizeof(float));
        xp = padded;
    }
    /* One GEMM per (group, tap): output row t reads input row t*stride+j*dil,
     * so the input leading dimension carries the stride. */
    for (int g = 0; g < groups; g++) {
        for (int j = 0; j < c->k; j++) {
            const float *a = xp + (size_t)j * c->dil * cin + (size_t)g * cing;
            const float *b = c->w + (((size_t)j * groups + g) * cing) * coutg;
            float *out = y + (size_t)g * coutg;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, tout, coutg,
                        cing, 1.0f, a, cin * c->stride, b, coutg,
                        j == 0 ? 0.0f : 1.0f, out, cout);
        }
    }
    free(padded);
    if (c->b)
        for (int t = 0; t < tout; t++)
            for (int o = 0; o < cout; o++) y[(size_t)t * cout + o] += c->b[o];
}

/* ----------------------------------------------------------------- linear --- */

int rt_lin_load(rt_lin *l, const rt_pt *pt, const char *name, int bias) {
    char nm[256];
    memset(l, 0, sizeof(*l));
    snprintf(nm, sizeof(nm), "%s.weight", name);
    const rt_tensor *w = rt_pt_get(pt, nm);
    if (!w || w->ndim != 2) return -1;
    const rt_tensor *b = NULL;
    if (bias) {
        snprintf(nm, sizeof(nm), "%s.bias", name);
        b = rt_pt_get(pt, nm);
        if (!b || b->numel != w->shape[0]) return -1;
    }
    l->out = (int)w->shape[0];
    l->in = (int)w->shape[1];
    if (w->dtype == RT_DT_F32 && (!b || b->dtype == RT_DT_F32)) {
        l->w = (const float *)w->data;
        l->b = b ? (const float *)b->data : NULL;
        return 0;
    }
    size_t nw = (size_t)l->out * l->in, nb = b ? (size_t)l->out : 0;
    l->copy = (float *)cwick_xmalloc((nw + nb) * sizeof(float));
    for (size_t i = 0; i < nw; i++) l->copy[i] = rt_tensor_f32(w, (int64_t)i);
    l->w = l->copy;
    if (b) {
        for (size_t i = 0; i < nb; i++) l->copy[nw + i] = rt_tensor_f32(b, (int64_t)i);
        l->b = l->copy + nw;
    }
    return 0;
}

void rt_lin_free(rt_lin *l) {
    if (!l) return;
    free(l->copy);
    memset(l, 0, sizeof(*l));
}

/* Output-row-parallel sgemv. A decode step is pure weight bandwidth: the 0.5B
 * speech LM touches ~2 GB per token, which one core reads at ~110 GB/s while
 * the rest of the memory bus sits idle. Splitting the rows leaves every dot
 * product intact -- each slice is the same sgemv on a contiguous row block --
 * so the only thing that changes is how many cores fetch weights. */
typedef struct { const float *w, *x; float *y; int k; } rt_gemv_job;

static void rt_gemv_slice(void *arg, int64_t lo, int64_t hi) {
    const rt_gemv_job *j = (const rt_gemv_job *)arg;
    cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)(hi - lo), j->k, 1.0f,
                j->w + (size_t)lo * j->k, j->k, j->x, 1, 0.0f, j->y + lo, 1);
}

void rt_gemv(const float *w, const float *x, float *y, int k, int n) {
    static int par = -1;
    if (par < 0) {
        const char *e = getenv("RT_GEMV_THREADS");
        par = (e && atoi(e) == 0) ? 0 : 1;
        if (par) cwick_pool_init();
    }
    if (par) {
        rt_gemv_job j = { w, x, y, k };
        cwick_pool_for(n, &j, rt_gemv_slice);
        return;
    }
    cblas_sgemv(CblasRowMajor, CblasNoTrans, n, k, 1.0f, w, k, x, 1, 0.0f, y, 1);
}

void rt_lin_apply(const rt_lin *l, const float *x, float *y, int m) {
    /* Every speech-stage projection comes through here, so this is the one
     * place the GPU has to be offered the work. m=1 is the talker's decode
     * step, where the product is pure weight bandwidth and one CPU core
     * leaves most of the machine's memory bus idle. */
    if (m > 0 && cwick_metal_rt_gpu() &&
        cwick_metal_gemm_f32(l->w, x, y, l->in, l->out, m)) {
        if (l->b)
            for (int t = 0; t < m; t++)
                for (int o = 0; o < l->out; o++) y[(size_t)t * l->out + o] += l->b[o];
        return;
    }
    if (m == 1) {
        /* sgemm at m=1 goes through Accelerate's dispatch_apply thread pool,
         * and the hand-off costs several times the product itself. sgemv
         * stays on this thread and already reads weights at ~110 GB/s. */
        rt_gemv(l->w, x, y, l->in, l->out);
    } else {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, l->out, l->in, 1.0f,
                    x, l->in, l->w, l->in, 0.0f, y, l->out);
    }
    if (l->b)
        for (int t = 0; t < m; t++)
            for (int o = 0; o < l->out; o++) y[(size_t)t * l->out + o] += l->b[o];
}

/* ------------------------------------------------------------------- stft --- */

#define RT_NFFT 16
#define RT_HOP 4
#define RT_NBINS 9

static void hann16(float *w) {
    for (int n = 0; n < RT_NFFT; n++)
        w[n] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * n / RT_NFFT);
}

static void dft_tables(float *cs, float *sn) {
    for (int k = 0; k < RT_NBINS; k++)
        for (int n = 0; n < RT_NFFT; n++) {
            float a = 2.0f * (float)M_PI * k * n / RT_NFFT;
            cs[k * RT_NFFT + n] = cosf(a);
            sn[k * RT_NFFT + n] = sinf(a);
        }
}

int rt_stft16(const float *x, int N, float *y) {
    if (N < 0 || N % RT_HOP) return -1;
    int tt = N / RT_HOP + 1;
    float w[RT_NFFT], cs[RT_NBINS * RT_NFFT], sn[RT_NBINS * RT_NFFT];
    hann16(w);
    dft_tables(cs, sn);
    int np = N + RT_NFFT;
    float *xp = (float *)cwick_xmalloc((size_t)np * sizeof(float));
    /* center=True, pad_mode reflect: the edge sample is not repeated. */
    for (int i = 0; i < RT_NFFT / 2; i++) xp[i] = x[RT_NFFT / 2 - i];
    memcpy(xp + RT_NFFT / 2, x, (size_t)N * sizeof(float));
    for (int i = 0; i < RT_NFFT / 2; i++) xp[N + RT_NFFT / 2 + i] = x[N - 2 - i];
    float frame[RT_NFFT];
    for (int t = 0; t < tt; t++) {
        for (int n = 0; n < RT_NFFT; n++)
            frame[n] = xp[(size_t)t * RT_HOP + n] * w[n];
        for (int k = 0; k < RT_NBINS; k++) {
            float re = 0.0f, im = 0.0f;
            for (int n = 0; n < RT_NFFT; n++) {
                re += frame[n] * cs[k * RT_NFFT + n];
                im -= frame[n] * sn[k * RT_NFFT + n];
            }
            y[(size_t)t * 18 + k] = re;
            y[(size_t)t * 18 + RT_NBINS + k] = im;
        }
    }
    free(xp);
    return tt;
}

void rt_istft16(const float *spec, int TT, float *out) {
    int N = (TT - 1) * RT_HOP;
    float w[RT_NFFT], cs[RT_NBINS * RT_NFFT], sn[RT_NBINS * RT_NFFT];
    hann16(w);
    dft_tables(cs, sn);
    size_t ny = (size_t)N + RT_NFFT;
    float *acc = (float *)cwick_xmalloc(ny * sizeof(float));
    float *env = (float *)cwick_xmalloc(ny * sizeof(float));
    memset(acc, 0, ny * sizeof(float));
    memset(env, 0, ny * sizeof(float));
    float frame[RT_NFFT];
    for (int t = 0; t < TT; t++) {
        const float *s = spec + (size_t)t * 18;
        for (int n = 0; n < RT_NFFT; n++) {
            float v = s[0] * cs[n] + s[RT_NBINS - 1] * cs[(RT_NBINS - 1) * RT_NFFT + n];
            for (int k = 1; k < RT_NBINS - 1; k++)
                v += 2.0f * (s[k] * cs[k * RT_NFFT + n] -
                             s[RT_NBINS + k] * sn[k * RT_NFFT + n]);
            frame[n] = v / RT_NFFT * w[n];
        }
        for (int n = 0; n < RT_NFFT; n++) {
            acc[(size_t)t * RT_HOP + n] += frame[n];
            env[(size_t)t * RT_HOP + n] += w[n] * w[n];
        }
    }
    for (int i = 0; i < N; i++) {
        float e = env[(size_t)i + RT_NFFT / 2];
        out[i] = acc[(size_t)i + RT_NFFT / 2] / (e > 1e-11f ? e : 1e-11f);
    }
    free(acc);
    free(env);
}

/* -------------------------------------------------------------------- rng --- */

void rt_rng_seed(rt_rng *r, uint64_t seed) {
    r->s = seed ? seed : 0x9e3779b97f4a7c15ull;
}

static uint64_t rng_next(rt_rng *r) {
    uint64_t z = (r->s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

float rt_rng_uniform(rt_rng *r) {
    return (float)((rng_next(r) >> 11) * (1.0 / 9007199254740992.0));
}

float rt_rng_normal(rt_rng *r) {
    float u1 = rt_rng_uniform(r), u2 = rt_rng_uniform(r);
    if (u1 < 1e-12f) u1 = 1e-12f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

/* ------------------------------------------------------------ activations --- */

void rt_silu(float *v, int n) {
    for (int i = 0; i < n; i++) v[i] = v[i] / (1.0f + expf(-v[i]));
}

void rt_gelu_tanh(float *v, int n) {
    const float c = 0.7978845608028654f; /* sqrt(2/pi) */
    /* The DiT calls this on [T][2048] once per layer per euler step, so the
     * scalar tanhf it replaces was the single largest term in the flow's
     * wall clock. vForce is SIMD and accurate to under an ulp. */
    if (n <= 0) return;
    float *a = (float *)cwick_xmalloc((size_t)n * sizeof(float));
    float k = 0.044715f;
    vDSP_vsq(v, 1, a, 1, (vDSP_Length)n);          /* x^2 */
    vDSP_vsmul(a, 1, &k, a, 1, (vDSP_Length)n);    /* 0.044715 x^2 */
    float one = 1.0f;
    vDSP_vsadd(a, 1, &one, a, 1, (vDSP_Length)n);  /* 1 + 0.044715 x^2 */
    vDSP_vmul(a, 1, v, 1, a, 1, (vDSP_Length)n);   /* x + 0.044715 x^3 */
    vDSP_vsmul(a, 1, &c, a, 1, (vDSP_Length)n);
    int nn = n;
    vvtanhf(a, a, &nn);
    vDSP_vsadd(a, 1, &one, a, 1, (vDSP_Length)n);
    vDSP_vmul(a, 1, v, 1, a, 1, (vDSP_Length)n);
    float half = 0.5f;
    vDSP_vsmul(a, 1, &half, v, 1, (vDSP_Length)n);
    free(a);
}

void rt_leaky(float *v, int n, float slope) {
    for (int i = 0; i < n; i++) if (v[i] < 0.0f) v[i] *= slope;
}

void rt_mish(float *v, int n) {
    if (n <= 0) return;
    /* x * tanh(log1p(exp(x))). Letting exp overflow to +inf is deliberate:
     * log(inf)=inf and tanh(inf)=1, which is the x>20 branch's answer anyway,
     * so the whole row goes through vForce without a per-element test. */
    float *a = (float *)cwick_xmalloc((size_t)n * sizeof(float));
    int nn = n;
    vvexpf(a, v, &nn);
    float one = 1.0f;
    vDSP_vsadd(a, 1, &one, a, 1, (vDSP_Length)n);
    vvlogf(a, a, &nn);
    vvtanhf(a, a, &nn);
    vDSP_vmul(a, 1, v, 1, v, 1, (vDSP_Length)n);
    free(a);
}

void rt_ln(float *dst, const float *src, int m, int n, float eps) {
    if (m <= 0 || n <= 0) return;
    for (int r = 0; r < m; r++) {
        const float *s = src + (size_t)r * n;
        float *d = dst + (size_t)r * n;
        /* Two passes, like the reference: centre first, then measure the
         * spread of the centred row. Deriving the variance from E[x^2]-E[x]^2
         * in one pass would cancel away the precision when the mean is large. */
        float mean = 0.0f, ms = 0.0f;
        vDSP_meanv(s, 1, &mean, (vDSP_Length)n);
        float neg = -mean;
        vDSP_vsadd(s, 1, &neg, d, 1, (vDSP_Length)n);
        vDSP_measqv(d, 1, &ms, (vDSP_Length)n);
        float inv = 1.0f / sqrtf(ms + eps);
        vDSP_vsmul(d, 1, &inv, d, 1, (vDSP_Length)n);
    }
}