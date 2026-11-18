/* rt_hift.c — causal HiFT vocoder, forward pass only.
 *
 * Every shape and padding rule below was read out of hift.pt and the stage
 * definitions it was trained with, not assumed: the source downsamplers are
 * k30/s15, k6/s3 and k1, the upsample stages are nearest x8/x5/x3 followed by
 * a left-padded stride-1 conv, and the f0 predictor runs in double because its
 * ELU stack is sensitive to rounding.
 *
 * weight_norm is folded once at load (rt_dsp), so the inner loops are plain
 * GEMMs over time-major activations. */
#include "rt_hift.h"

#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "log.h"
#include "rt_dsp.h"
#include "rt_torch.h"
#include "util.h"

#define HT_TAG "hift"
#define HT_SR 24000
#define HT_HOP 480 /* prod(upsample_rates) * istft hop = 120 * 4 */

typedef struct {
    rt_conv c1[3], c2[3];
    const float *a1[3], *a2[3]; /* Snake alpha, read straight from the map */
    int ch;
} ht_res;

/* The f0 predictor is the one stage the reference runs in float64. */
typedef struct {
    double *w; /* [k][cin][cout] */
    double *b;
    int cin, cout, k, padl, padr;
} ht_conv64;

struct rt_hift {
    rt_pt *pt;
    rt_conv conv_pre, ups[3], sdown[3], conv_post;
    ht_res sres[3], res[9];
    rt_lin llinear;
    ht_conv64 f0c[5];
    double f0w[512], f0b;
};

static int load64(ht_conv64 *c, const rt_pt *pt, const char *name, int padl,
                  int padr) {
    char nm[256];
    memset(c, 0, sizeof(*c));
    snprintf(nm, sizeof(nm), "%s.parametrizations.weight.original1", name);
    const rt_tensor *v = rt_pt_get(pt, nm);
    snprintf(nm, sizeof(nm), "%s.parametrizations.weight.original0", name);
    const rt_tensor *g = rt_pt_get(pt, nm);
    if (!v || !g || v->ndim != 3) return -1;
    int cout = (int)v->shape[0], cin = (int)v->shape[1], k = (int)v->shape[2];
    c->cin = cin;
    c->cout = cout;
    c->k = k;
    c->padl = padl;
    c->padr = padr;
    c->w = (double *)cwick_xmalloc((size_t)k * cin * cout * sizeof(double));
    for (int o = 0; o < cout; o++) {
        double ss = 0.0;
        for (int64_t i = 0; i < (int64_t)cin * k; i++) {
            double x = rt_tensor_f32(v, (int64_t)o * cin * k + i);
            ss += x * x;
        }
        double scale = rt_tensor_f32(g, o) / (sqrt(ss) + 1e-12);
        for (int j = 0; j < k; j++)
            for (int i = 0; i < cin; i++)
                c->w[((size_t)j * cin + i) * cout + o] =
                    rt_tensor_f32(v, ((int64_t)o * cin + i) * k + j) * scale;
    }
    snprintf(nm, sizeof(nm), "%s.bias", name);
    const rt_tensor *b = rt_pt_get(pt, nm);
    if (!b || b->numel != cout) return -1;
    c->b = (double *)cwick_xmalloc((size_t)cout * sizeof(double));
    for (int o = 0; o < cout; o++) c->b[o] = rt_tensor_f32(b, o);
    return 0;
}

static void conv64_apply(const ht_conv64 *c, const double *x, double *y, int T) {
    int span = c->k;
    int tp = T + c->padl + c->padr;
    int tout = (tp - span) + 1;
    const double *xp = x;
    double *padded = NULL;
    if (c->padl || c->padr) {
        padded = (double *)cwick_xmalloc((size_t)tp * c->cin * sizeof(double));
        memset(padded, 0, (size_t)tp * c->cin * sizeof(double));
        memcpy(padded + (size_t)c->padl * c->cin, x,
               (size_t)T * c->cin * sizeof(double));
        xp = padded;
    }
    for (int j = 0; j < c->k; j++)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, tout, c->cout,
                    c->cin, 1.0, xp + (size_t)j * c->cin, c->cin,
                    c->w + (size_t)j * c->cin * c->cout, c->cout,
                    j == 0 ? 0.0 : 1.0, y, c->cout);
    free(padded);
    for (int t = 0; t < tout; t++)
        for (int o = 0; o < c->cout; o++) y[(size_t)t * c->cout + o] += c->b[o];
}

static void elu64(double *v, size_t n) {
    for (size_t i = 0; i < n; i++) if (v[i] < 0.0) v[i] = exp(v[i]) - 1.0;
}

/* ---------------------------------------------------------------- loading --- */

static int res_load(ht_res *r, const rt_pt *pt, const char *fmt, int idx, int ch,
                    int k) {
    char name[192];
    r->ch = ch;
    static const int dil[3] = {1, 3, 5};
    for (int j = 0; j < 3; j++) {
        /* CausalConv1d left padding: d*(k-1) for odd k. */
        snprintf(name, sizeof(name), fmt, idx, "convs1", j);
        if (rt_conv_load(&r->c1[j], pt, name, 1, 1, dil[j], dil[j] * (k - 1), 0))
            return -1;
        snprintf(name, sizeof(name), fmt, idx, "convs2", j);
        if (rt_conv_load(&r->c2[j], pt, name, 1, 1, 1, k - 1, 0)) return -1;
        snprintf(name, sizeof(name), fmt, idx, "activations1", j);
        char an[224];
        snprintf(an, sizeof(an), "%s.alpha", name);
        const rt_tensor *a = rt_pt_get(pt, an);
        if (!a || a->numel != ch) return -1;
        r->a1[j] = (const float *)a->data;
        snprintf(name, sizeof(name), fmt, idx, "activations2", j);
        snprintf(an, sizeof(an), "%s.alpha", name);
        a = rt_pt_get(pt, an);
        if (!a || a->numel != ch) return -1;
        r->a2[j] = (const float *)a->data;
    }
    return 0;
}

static void res_free(ht_res *r) {
    for (int j = 0; j < 3; j++) {
        rt_conv_free(&r->c1[j]);
        rt_conv_free(&r->c2[j]);
    }
}

rt_hift *rt_hift_open(const char *path, cwick_error *err) {
    rt_hift *h = (rt_hift *)cwick_xmalloc(sizeof(*h));
    memset(h, 0, sizeof(*h));
    if (rt_pt_open(&h->pt, path)) {
        cwick_error_set(err, CWICK_ERR_IO, "cannot read %s", path);
        free(h);
        return NULL;
    }
    int bad = 0;
    bad |= rt_conv_load(&h->conv_pre, h->pt, "conv_pre", 1, 1, 1, 0, 4);
    static const int uk[3] = {16, 11, 7};
    for (int i = 0; i < 3; i++) {
        char nm[64];
        snprintf(nm, sizeof(nm), "ups.%d", i);
        bad |= rt_conv_load(&h->ups[i], h->pt, nm, 1, 1, 1, uk[i] - 1, 0);
    }
    /* downsample_rates = [1] + reversed(upsample_rates)[:-1] = [1,3,5],
     * cumprod [1,3,15], walked backwards: k=u*2, stride=u, left pad u-1. */
    static const int sk[3] = {30, 6, 1}, ss[3] = {15, 3, 1};
    for (int i = 0; i < 3; i++) {
        char nm[64];
        snprintf(nm, sizeof(nm), "source_downs.%d", i);
        bad |= rt_conv_load(&h->sdown[i], h->pt, nm, 1, ss[i], 1,
                            ss[i] > 1 ? ss[i] - 1 : 0, 0);
        if (sk[i] != h->sdown[i].k && !bad) bad = 1;
    }
    static const int srk[3] = {7, 7, 11};
    for (int i = 0; i < 3; i++)
        bad |= res_load(&h->sres[i], h->pt, "source_resblocks.%d.%s.%d", i,
                        512 >> (i + 1), srk[i]);
    static const int rk[3] = {3, 7, 11};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            bad |= res_load(&h->res[i * 3 + j], h->pt, "resblocks.%d.%s.%d",
                            i * 3 + j, 512 >> (i + 1), rk[j]);
    bad |= rt_conv_load(&h->conv_post, h->pt, "conv_post", 1, 1, 1, 6, 0);
    bad |= rt_lin_load(&h->llinear, h->pt, "m_source.l_linear", 1);
    static const int f0k[5] = {4, 3, 3, 3, 3};
    for (int i = 0; i < 5; i++) {
        char nm[96];
        snprintf(nm, sizeof(nm), "f0_predictor.condnet.%d", i * 2);
        /* condnet[0] looks right (pad 3); the rest look left (pad 2). */
        bad |= load64(&h->f0c[i], h->pt, nm, i == 0 ? 0 : f0k[i] - 1,
                      i == 0 ? 3 : 0);
    }
    const rt_tensor *cw = rt_pt_get(h->pt, "f0_predictor.classifier.weight");
    const rt_tensor *cb = rt_pt_get(h->pt, "f0_predictor.classifier.bias");
    if (!cw || !cb || cw->numel != 512 || cb->numel != 1) bad = 1;
    if (!bad) {
        for (int i = 0; i < 512; i++) h->f0w[i] = rt_tensor_f32(cw, i);
        h->f0b = rt_tensor_f32(cb, 0);
    }
    if (bad) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "%s: unexpected vocoder layout", path);
        rt_hift_close(h);
        return NULL;
    }
    return h;
}

void rt_hift_close(rt_hift *h) {
    if (!h) return;
    rt_conv_free(&h->conv_pre);
    rt_conv_free(&h->conv_post);
    for (int i = 0; i < 3; i++) {
        rt_conv_free(&h->ups[i]);
        rt_conv_free(&h->sdown[i]);
        res_free(&h->sres[i]);
    }
    for (int i = 0; i < 9; i++) res_free(&h->res[i]);
    rt_lin_free(&h->llinear);
    for (int i = 0; i < 5; i++) {
        free(h->f0c[i].w);
        free(h->f0c[i].b);
    }
    if (h->pt) rt_pt_close(h->pt);
    free(h);
}

/* ------------------------------------------------------------------ source --- */

static void f0_predict(rt_hift *h, const float *mel, int T, double *f0) {
    double *x = (double *)cwick_xmalloc((size_t)T * 80 * sizeof(double));
    for (int64_t i = 0; i < (int64_t)T * 80; i++) x[i] = mel[i];
    double *a = (double *)cwick_xmalloc((size_t)T * 512 * sizeof(double));
    double *b = (double *)cwick_xmalloc((size_t)T * 512 * sizeof(double));
    conv64_apply(&h->f0c[0], x, a, T);
    elu64(a, (size_t)T * 512);
    for (int j = 1; j < 5; j++) {
        conv64_apply(&h->f0c[j], a, b, T);
        elu64(b, (size_t)T * 512);
        double *t = a; a = b; b = t;
    }
    for (int t = 0; t < T; t++) {
        double acc = h->f0b;
        const double *row = a + (size_t)t * 512;
        for (int i = 0; i < 512; i++) acc += row[i] * h->f0w[i];
        f0[t] = fabs(acc);
    }
    free(x);
    free(a);
    free(b);
}

/* Harmonic + noise excitation at 24 kHz. The phase is accumulated on the
 * 25 fps frame grid and held across each 480-sample block, which is what the
 * causal sine generator's down/up interpolate pair amounts to. */
static void source_signal(rt_hift *h, const double *f0, int T, float *s) {
    float *rb = (float *)cwick_xmalloc((size_t)T * 9 * sizeof(float));
    float *ph = (float *)cwick_xmalloc((size_t)T * 9 * sizeof(float));
    for (int i = 0; i < T; i++)
        for (int k = 0; k < 9; k++)
            rb[(size_t)i * 9 + k] =
                fmodf((float)(f0[i] * (k + 1) / HT_SR), 1.0f);
    float acc[9] = {0};
    for (int i = 0; i < T; i++) {
        for (int k = 0; k < 9; k++) {
            /* linear downsample by 480, align_corners=False: the sample lands
             * halfway between the last of block i and the first of block i+1. */
            float v = i + 1 < T ? 0.5f * (rb[(size_t)i * 9 + k] +
                                          rb[(size_t)(i + 1) * 9 + k])
                                : rb[(size_t)i * 9 + k];
            acc[k] += v;
            ph[(size_t)i * 9 + k] = acc[k] * 2.0f * (float)M_PI;
        }
    }
    rt_rng rng;
    rt_rng_seed(&rng, 0x5eed1234ull);
    const float *lw = h->llinear.w;
    float lb = h->llinear.b ? h->llinear.b[0] : 0.0f;
    for (int i = 0; i < T; i++) {
        float uv = f0[i] > 10.0 ? 1.0f : 0.0f;
        float amp = uv * 0.003f + (1.0f - uv) * (0.1f / 3.0f);
        float sn[9];
        for (int k = 0; k < 9; k++)
            sn[k] = sinf(ph[(size_t)i * 9 + k]) * 0.1f * uv;
        for (int j = 0; j < HT_HOP; j++) {
            float dot = lb;
            for (int k = 0; k < 9; k++)
                dot += lw[k] * (sn[k] + amp * rt_rng_uniform(&rng));
            s[(size_t)i * HT_HOP + j] = tanhf(dot);
        }
    }
    free(rb);
    free(ph);
}

/* ----------------------------------------------------------------- resblock --- */

static void snake(float *v, const float *alpha, int T, int ch) {
    for (int t = 0; t < T; t++)
        for (int c = 0; c < ch; c++) {
            float a = alpha[c], x = v[(size_t)t * ch + c];
            float s = sinf(a * x);
            v[(size_t)t * ch + c] = x + s * s / (a + 1e-9f);
        }
}

/* out = in + sum_j conv2(snake(conv1(snake(in)))), the residual accumulated
 * in place exactly as the reference does. t1/t2 are [T][ch] scratch. */
static void res_run(const ht_res *r, const float *in, float *out, float *t1,
                    float *t2, int T) {
    int ch = r->ch;
    if (out != in) memcpy(out, in, (size_t)T * ch * sizeof(float));
    for (int j = 0; j < 3; j++) {
        memcpy(t1, out, (size_t)T * ch * sizeof(float));
        snake(t1, r->a1[j], T, ch);
        rt_conv_apply(&r->c1[j], t1, t2, T);
        snake(t2, r->a2[j], T, ch);
        rt_conv_apply(&r->c2[j], t2, t1, T);
        for (int64_t i = 0; i < (int64_t)T * ch; i++) out[i] += t1[i];
    }
}

static void upsample_nearest(const float *x, int T, int ch, int u, float *y) {
    for (int t = 0; t < T; t++)
        for (int j = 0; j < u; j++)
            memcpy(y + ((size_t)t * u + j) * ch, x + (size_t)t * ch,
                   (size_t)ch * sizeof(float));
}

/* ------------------------------------------------------------------- synth --- */

int rt_hift_synth(rt_hift *h, const float *mel, int T, float **pcm) {
    if (!h || T <= 0) return -1;
    int N = T * HT_HOP;
    double *f0 = (double *)cwick_xmalloc((size_t)T * sizeof(double));
    f0_predict(h, mel, T, f0);
    float *s = (float *)cwick_xmalloc((size_t)N * sizeof(float));
    source_signal(h, f0, T, s);
    free(f0);
    int TT = N / 4 + 1;
    float *spec = (float *)cwick_xmalloc((size_t)TT * 18 * sizeof(float));
    rt_stft16(s, N, spec);
    free(s);

    static const int up[3] = {8, 5, 3};
    static const int ch[3] = {256, 128, 64};
    int tin = T, cin = 80;
    float *x = (float *)cwick_xmalloc((size_t)T * 512 * sizeof(float));
    rt_conv_apply(&h->conv_pre, mel, x, T);
    cin = 512;

    for (int i = 0; i < 3; i++) {
        int tu = tin * up[i];
        int tout = tu + (i == 2 ? 1 : 0);
        rt_leaky(x, tin * cin, 0.1f);
        float *xu = (float *)cwick_xmalloc((size_t)tu * cin * sizeof(float));
        upsample_nearest(x, tin, cin, up[i], xu);
        free(x);
        /* The last stage prepends one reflected sample so the STFT frame grid
         * and the upsampled mel grid line up. */
        float *xn = (float *)cwick_xmalloc((size_t)tout * ch[i] * sizeof(float));
        float *dst = i == 2 ? xn + ch[i] : xn;
        rt_conv_apply(&h->ups[i], xu, dst, tu);
        if (i == 2) memcpy(xn, xn + 2 * ch[i], (size_t)ch[i] * sizeof(float));

        int slen = rt_conv_len(&h->sdown[i], TT);
        if (slen != tout) {
            CWICK_INFO(HT_TAG, "stage %d: source %d vs mel %d frames", i, slen, tout);
            free(xu); free(xn); free(spec); free(x);
            return -1;
        }
        float *si = xu; /* xu is dead after the conv and is large enough */
        rt_conv_apply(&h->sdown[i], spec, si, TT);
        float *t1 = (float *)cwick_xmalloc((size_t)tout * ch[i] * sizeof(float));
        float *t2 = (float *)cwick_xmalloc((size_t)tout * ch[i] * sizeof(float));
        res_run(&h->sres[i], si, si, t1, t2, tout);
        for (int64_t j = 0; j < (int64_t)tout * ch[i]; j++) xn[j] += si[j];
        free(xu);

        float *acc = (float *)cwick_xmalloc((size_t)tout * ch[i] * sizeof(float));
        float *tmp = (float *)cwick_xmalloc((size_t)tout * ch[i] * sizeof(float));
        for (int j = 0; j < 3; j++) {
            const ht_res *r = &h->res[i * 3 + j];
            float *out = j == 0 ? acc : tmp;
            res_run(r, xn, out, t1, t2, tout);
            if (j)
                for (int64_t q = 0; q < (int64_t)tout * ch[i]; q++) acc[q] += tmp[q];
        }
        for (int64_t j = 0; j < (int64_t)tout * ch[i]; j++) acc[j] /= 3.0f;
        free(xn);
        free(tmp);
        free(t1);
        free(t2);
        x = acc;
        tin = tout;
        cin = ch[i];
    }
    free(spec);

    rt_leaky(x, tin * cin, 0.01f);
    float *post = (float *)cwick_xmalloc((size_t)tin * 18 * sizeof(float));
    rt_conv_apply(&h->conv_post, x, post, tin);
    free(x);
    float *ri = (float *)cwick_xmalloc((size_t)tin * 18 * sizeof(float));
    for (int t = 0; t < tin; t++) {
        for (int k = 0; k < 9; k++) {
            float m = expf(post[(size_t)t * 18 + k]);
            if (m > 1e2f) m = 1e2f;
            float p = sinf(post[(size_t)t * 18 + 9 + k]);
            ri[(size_t)t * 18 + k] = m * cosf(p);
            ri[(size_t)t * 18 + 9 + k] = m * sinf(p);
        }
    }
    free(post);
    float *out = (float *)cwick_xmalloc((size_t)N * sizeof(float));
    rt_istft16(ri, tin, out);
    free(ri);
    for (int i = 0; i < N; i++) {
        if (out[i] > 0.99f) out[i] = 0.99f;
        if (out[i] < -0.99f) out[i] = -0.99f;
    }
    *pcm = out;
    return N;
}