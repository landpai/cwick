/* h3v_audio.c — the audio VAE's BigVGAN decoder.
 *
 * The encoder is a DAC-style strided stack and is only used on the training
 * path. Decoding is:
 *
 *   32-channel latent --dec_in_proj (1x1)--> 2048 channels
 *                     --conv_pre (k=7)-->    1024 channels
 *                     --7 x [transposed-conv upsample, 3 AMP blocks]-->
 *                     --snakebeta, conv_post (k=7)--> mono waveform
 *
 * The upsample rates are [5,5,2,2,2,2,2]; their product is 800 samples per
 * latent token, which at 32 kHz is the 40 Hz token rate the model card quotes.
 * Channel width halves at every stage, 1024 down to 8.
 *
 * Every convolution in the checkpoint is weight-normalised: the shards carry
 * `weight_g` (one magnitude per output channel) and `weight_v` (the direction),
 * so the usable kernel is g * v / ||v||. Both are baked into a plain kernel
 * once per decode, laid out as [tap][out][in] so that each tap is a contiguous
 * sgemm operand and a 7-tap 2048x1024 conv becomes seven gemms.
 *
 * The activations are alias-free SnakeBeta: upsample 2x through a 12-tap
 * Kaiser windowed-sinc, apply x + sin^2(alpha*x)/beta, downsample 2x through
 * the matching low-pass. Alpha and beta are stored in log space and the two
 * filters ship in the checkpoint, so they are read rather than recomputed —
 * the reference build and this one then agree bit for bit on the window.
 *
 * Stereo: the model card states the VAE runs the same decoder on the left and
 * right channels independently and recombines afterwards, so the 32-channel
 * latent sequence is read as (frame, channel) pairs, de-interleaved, decoded
 * twice and interleaved again.
 *
 * Comments describe mechanism and measured numbers only.
 */
#include "h3v_internal.h"

#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "log.h"
#include "util.h"

/* <Accelerate/Accelerate.h> pulls in <complex.h>, whose `I` macro would
 * rewrite the `I` member of au_conv (input channels) into `_Complex`. The
 * decoder does no complex arithmetic, so dropping the macro is safe and keeps
 * the conv struct readable. */
#undef I

#define AU_STAGES      7      /* upsample_rates [5,5,2,2,2,2,2]               */
#define AU_BLOCKS      3      /* resblock_kernel_sizes 3, 7, 11              */
#define AU_DILS        3      /* resblock_dilation_sizes 1, 3, 5             */
#define AU_PRE_CH      1024   /* upsample_initial_channel                    */
#define AU_LATENT_CH   32
#define AU_FILTER      12     /* alias-free up/down kernel                   */
#define AU_UP_PAD      5      /* replicate padding inside UpSample1d         */
#define AU_HOP         800    /* product of the upsample rates               */

static const int AU_UP_RATE[AU_STAGES] = {5, 5, 2, 2, 2, 2, 2};
static const int AU_UP_KERN[AU_STAGES] = {9, 9, 4, 4, 4, 4, 4};
static const int AU_RES_KERN[AU_BLOCKS] = {3, 7, 11};
static const int AU_RES_DIL[AU_DILS] = {1, 3, 5};

static const float AU_LAT_MEAN[AU_LATENT_CH] = {
    -0.020211687488382354f, 0.3876466479950502f, -0.04398279799186767f,
    -0.28591514936373f, 0.08179686214561671f, -0.35782641352446604f,
    0.040623809960919084f, -0.01552534501956604f, -0.223362481667332f,
    0.1821006842509091f, 0.2941778783780663f, -0.07901167601970885f,
    -0.056815072777201f, -0.3699028221860095f, -0.31616315591624855f,
    0.5905951377425391f, -0.052139568068853864f, 0.013673160263486295f,
    -0.03691647864630577f, 0.09732660653298163f, -0.3394662328788498f,
    -0.30685677538541667f, -0.24504598907458763f, -0.034698524462007344f,
    0.02868032184767538f, -0.21217779266454084f, -0.1678263169941987f,
    0.3221287889040614f, -0.1223055851554907f, 0.4356604928128464f,
    -0.0502599202236253f, 0.3979258376211797f
};

static const float AU_LAT_STD[AU_LATENT_CH] = {
    1.6895524230479284f, 2.76263727217653f, 1.7945344281264435f,
    1.6801681847309828f, 1.6390226546605453f, 2.7788298348882177f,
    1.7659090095747236f, 1.6199757612137327f, 2.6336525640336896f,
    1.8539356672817833f, 2.5056497896915633f, 1.811019237886178f,
    1.9579657790720237f, 1.6685498243529284f, 1.4922469314453364f,
    3.298670198067373f, 1.9491804496832168f, 1.8720003270431442f,
    1.8334080103291832f, 1.6488070416529093f, 1.6176957696319716f,
    1.9131449234774398f, 1.5695245398428617f, 1.6943659940415912f,
    1.8318420762504692f, 1.5540637421583379f, 1.9344930328968526f,
    1.599198216109855f, 1.718045989838149f, 1.6307219190837705f,
    1.8661226051202384f, 1.5613768203168363f
};

static float *au_f32(int64_t n) {
    return (float *)cwick_xmalloc((size_t)n * sizeof(float));
}

/* A weight-normalised conv with the kernel transposed to [tap][out][in]. */
typedef struct {
    float *w;
    float *b;             /* NULL when the layer has no bias */
    int64_t O, I, K;
    int stride, dil, pad;
} au_conv;

static bool au_get(const cwick_safetensors *s, const char *name,
                   cwick_st_tensor *t, cwick_error *err, bool required) {
    if (cwick_safetensors_get(s, name, t, err) == CWICK_OK) return true;
    return !required;
}

static bool au_conv_build(const cwick_safetensors *s, au_conv *c,
                          const char *prefix, int64_t I, int64_t K, int stride,
                          int dil, int pad, cwick_error *err) {
    char n[192];
    cwick_st_tensor g, v, b;
    memset(c, 0, sizeof(*c));

    snprintf(n, sizeof(n), "%s.weight_g", prefix);
    memset(&g, 0, sizeof(g));
    au_get(s, n, &g, err, false);   /* leaves g zeroed when the layer has none */
    if (!g.data) {
        /* `dec_in_proj` is the only convolution in the audio decoder that was
         * saved without weight normalisation, so a plain `.weight` is already
         * the kernel and no g vector has to be scaled in. */
        snprintf(n, sizeof(n), "%s.weight", prefix);
        if (!au_get(s, n, &v, err, true)) return false;
        if (v.dtype != CWICK_ST_F32) {
            cwick_error_set(err, CWICK_ERR_PARSE, "%s is not float32", prefix);
            return false;
        }
        if (v.rows <= 0 || v.cols != I * K) {
            cwick_error_set(err, CWICK_ERR_PARSE,
                            "%s kernel is %lldx%lld, expected %lld wide", prefix,
                            (long long)v.rows, (long long)v.cols, (long long)I * K);
            return false;
        }
        c->O = v.rows;
        c->I = I;
        c->K = K;
        c->stride = stride;
        c->dil = dil;
        c->pad = pad;
        c->w = au_f32(K * c->O * I);
        const float *plane = (const float *)v.data;
        for (int64_t o = 0; o < c->O; o++)
            for (int64_t i = 0; i < I; i++)
                for (int64_t k = 0; k < K; k++)
                    c->w[(k * c->O + o) * I + i] = plane[o * I * K + i * K + k];
        goto bias;
    }
    snprintf(n, sizeof(n), "%s.weight_v", prefix);
    if (!au_get(s, n, &v, err, true)) return false;
    if (v.dtype != CWICK_ST_F32 || g.dtype != CWICK_ST_F32) {
        cwick_error_set(err, CWICK_ERR_PARSE, "%s is not float32", prefix);
        return false;
    }
    if (v.rows <= 0 || v.cols != I * K) {
        cwick_error_set(err, CWICK_ERR_PARSE,
                        "%s kernel is %lldx%lld, expected %lldx%lld", prefix,
                        (long long)v.rows, (long long)v.cols, (long long)I * K,
                        (long long)v.rows);
        return false;
    }
    c->O = v.rows;
    c->I = I;
    c->K = K;
    c->stride = stride;
    c->dil = dil;
    c->pad = pad;

    /* w[tap][o][i] = g[o] * v[o][i][tap] / ||v[o]|| */
    c->w = au_f32(K * c->O * I);
    const float *gv = (const float *)g.data;
    const float *vv = (const float *)v.data;
    for (int64_t o = 0; o < c->O; o++) {
        const float *row = vv + o * I * K;
        double ss = 0.0;
        for (int64_t q = 0; q < I * K; q++) ss += (double)row[q] * row[q];
        float norm = (float)sqrt(ss);
        if (norm < 1e-12f) norm = 1e-12f;
        const float scale = gv[o] / norm;
        for (int64_t i = 0; i < I; i++)
            for (int64_t k = 0; k < K; k++)
                c->w[(k * c->O + o) * I + i] = row[i * K + k] * scale;
    }

bias:
    snprintf(n, sizeof(n), "%s.bias", prefix);
    memset(&b, 0, sizeof(b));
    if (au_get(s, n, &b, err, false) && b.data && b.dtype == CWICK_ST_F32) {
        c->b = au_f32(c->O);
        memcpy(c->b, b.data, (size_t)c->O * sizeof(float));
    }
    return true;
}

static void au_conv_free(au_conv *c) {
    free(c->w);
    free(c->b);
    c->w = c->b = NULL;
}

/* The upsample stages are ConvTranspose1d, whose weight is stored [I][O][K]
 * (input channel first) and weight-normalised over dim 0 -- one magnitude per
 * *input* channel, the norm taken across (O,K). That is the transpose of the
 * regular conv layout au_conv_build reads, so it needs its own loader. The
 * kernel is still emitted as w[tap][o][i] for au_convt_run. */
static bool au_convt_build(const cwick_safetensors *s, au_conv *c,
                           const char *prefix, int64_t I, int64_t O, int64_t K,
                           int stride, int pad, cwick_error *err) {
    char n[192];
    cwick_st_tensor g, v, b;
    memset(c, 0, sizeof(*c));
    c->O = O; c->I = I; c->K = K;
    c->stride = stride; c->dil = 1; c->pad = pad;

    snprintf(n, sizeof(n), "%s.weight_g", prefix);
    memset(&g, 0, sizeof(g));
    au_get(s, n, &g, err, false);
    snprintf(n, sizeof(n), "%s.weight_v", prefix);
    if (!g.data) snprintf(n, sizeof(n), "%s.weight", prefix);
    memset(&v, 0, sizeof(v));
    if (!au_get(s, n, &v, err, true)) return false;
    if (v.dtype != CWICK_ST_F32 || (g.data && g.dtype != CWICK_ST_F32)) {
        cwick_error_set(err, CWICK_ERR_PARSE, "%s is not float32", prefix);
        return false;
    }
    if (v.rows != I || v.cols != O * K) {
        cwick_error_set(err, CWICK_ERR_PARSE,
                        "%s kernel is %lldx%lld, expected %lldx%lld", prefix,
                        (long long)v.rows, (long long)v.cols,
                        (long long)I, (long long)(O * K));
        return false;
    }
    c->w = au_f32(K * O * I);
    const float *vv = (const float *)v.data;
    const float *gv = g.data ? (const float *)g.data : NULL;
    for (int64_t i = 0; i < I; i++) {
        const float *row = vv + i * O * K;
        float scale = 1.0f;
        if (gv) {
            double ss = 0.0;
            for (int64_t q = 0; q < O * K; q++) ss += (double)row[q] * row[q];
            float norm = (float)sqrt(ss);
            if (norm < 1e-12f) norm = 1e-12f;
            scale = gv[i] / norm;
        }
        for (int64_t o = 0; o < O; o++)
            for (int64_t k = 0; k < K; k++)
                c->w[(k * O + o) * I + i] = row[o * K + k] * scale;
    }
    snprintf(n, sizeof(n), "%s.bias", prefix);
    memset(&b, 0, sizeof(b));
    if (au_get(s, n, &b, err, false) && b.data && b.dtype == CWICK_ST_F32) {
        c->b = au_f32(O);
        memcpy(c->b, b.data, (size_t)O * sizeof(float));
    }
    return true;
}

/* y[O][T] = conv1d(x[I][T]) with zero padding. */
static void au_conv1d_run(const au_conv *c, const float *x, int64_t T, float *y) {
    const int64_t O = c->O, I = c->I, K = c->K, pad = c->pad;
    const int64_t xp_len = T + 2 * pad;
    float *xp = au_f32(I * xp_len);
    memset(xp, 0, (size_t)I * xp_len * sizeof(float));
    for (int64_t i = 0; i < I; i++)
        memcpy(xp + i * xp_len + pad, x + i * T, (size_t)T * sizeof(float));

    memset(y, 0, (size_t)O * T * sizeof(float));
    for (int64_t k = 0; k < K; k++) {
        const float *wk = c->w + (k * O) * I;
        const float *src = xp + (int64_t)k * c->dil;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)O, (int)T,
                    (int)I, 1.0f, wk, (int)I, src, (int)xp_len, 1.0f, y, (int)T);
    }
    free(xp);
    if (c->b)
        for (int64_t o = 0; o < O; o++) {
            float *row = y + o * T;
            const float b = c->b[o];
            for (int64_t t = 0; t < T; t++) row[t] += b;
        }
}

/* y = conv_transpose1d(x); length is (T-1)*stride + K - 2*pad. */
static void au_convt_run(const au_conv *c, const float *x, int64_t T, float *y,
                         int64_t *out_len) {
    const int64_t O = c->O, I = c->I, K = c->K;
    const int64_t L = (T - 1) * c->stride + K - 2 * c->pad;
    memset(y, 0, (size_t)O * L * sizeof(float));
    float *tmp = au_f32(O * T);
    for (int64_t k = 0; k < K; k++) {
        const float *wk = c->w + (k * O) * I;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)O, (int)T,
                    (int)I, 1.0f, wk, (int)I, x, (int)T, 0.0f, tmp, (int)T);
        const int64_t off = k - c->pad;
        for (int64_t o = 0; o < O; o++) {
            const float *s = tmp + o * T;
            float *d = y + o * L;
            /* padding crops the scatter: t*stride+k-pad falls outside [0,L)
             * for the first/last taps and must be dropped, not written. */
            for (int64_t t = 0; t < T; t++) {
                int64_t pos = t * c->stride + off;
                if (pos >= 0 && pos < L) d[pos] += s[t];
            }
        }
    }
    free(tmp);
    if (c->b)
        for (int64_t o = 0; o < O; o++) {
            float *row = y + o * L;
            const float b = c->b[o];
            for (int64_t t = 0; t < L; t++) row[t] += b;
        }
    *out_len = L;
}

/* Upsample1d(2, 12) then SnakeBeta then DownSample1d(2, 12). */
static void au_act(const float *x, int64_t C, int64_t T, const float *alpha,
                   const float *beta, const float *up, const float *dn,
                   float *y) {
    const int64_t padded = T + 2 * AU_UP_PAD;
    float *xp = au_f32(C * padded);
    for (int64_t c = 0; c < C; c++) {
        const float *src = x + c * T;
        float *dst = xp + c * padded;
        for (int64_t t = 0; t < padded; t++) {
            int64_t st = t - AU_UP_PAD;
            if (st < 0) st = 0;
            if (st >= T) st = T - 1;
            dst[t] = src[st];
        }
    }
    const int64_t raw = (padded - 1) * 2 + AU_FILTER;
    float *big = au_f32(C * raw);
    memset(big, 0, (size_t)C * raw * sizeof(float));
    for (int64_t k = 0; k < AU_FILTER; k++) {
        const float fk = up[k] * 2.0f;
        for (int64_t c = 0; c < C; c++) {
            const float *src = xp + c * padded;
            float *dst = big + c * raw;
            for (int64_t t = 0; t < padded; t++) dst[t * 2 + k] += fk * src[t];
        }
    }
    free(xp);
    /* UpSample1d drops pad*2 + (k - 2)/2 samples from each end. */
    const int64_t trim = AU_UP_PAD * 2 + (AU_FILTER - 2) / 2;
    const int64_t up_len = 2 * T;
    float *mid = au_f32(C * up_len);
    for (int64_t c = 0; c < C; c++)
        memcpy(mid + c * up_len, big + c * raw + trim,
               (size_t)up_len * sizeof(float));
    free(big);

    for (int64_t c = 0; c < C; c++) {
        const float a = expf(alpha[c]), b = expf(beta[c]);
        float *row = mid + c * up_len;
        for (int64_t t = 0; t < up_len; t++) {
            const float s = sinf(a * row[t]);
            row[t] += s * s / (b + 1e-9f);
        }
    }

    /* LowPassFilter1d: replicate pad 5 left / 6 right, then stride 2. */
    const int64_t pl = AU_FILTER / 2 - 1;
    for (int64_t c = 0; c < C; c++) {
        const float *src = mid + c * up_len;
        float *dst = y + c * T;
        for (int64_t t = 0; t < T; t++) {
            float acc = 0.0f;
            const int64_t base = t * 2 - pl;
            for (int64_t k = 0; k < AU_FILTER; k++) {
                int64_t idx = base + k;
                if (idx < 0) idx = 0;
                if (idx >= up_len) idx = up_len - 1;
                acc += dn[k] * src[idx];
            }
            dst[t] = acc;
        }
    }
    free(mid);
}

/* One AMP block: three (snake, dilated conv, snake, 1x1-kernel conv) pairs,
 * all summed back into the residual. */
typedef struct {
    au_conv c1[AU_DILS];
    au_conv c2[AU_DILS];
    float *alpha[AU_DILS * 2], *beta[AU_DILS * 2];
    float *up[AU_DILS * 2], *dn[AU_DILS * 2];
    int64_t ch;
} au_amp;

static bool au_filter(const cwick_safetensors *s, const char *prefix, int idx,
                      const char *suffix, float *dst, cwick_error *err) {
    char n[192];
    cwick_st_tensor t;
    snprintf(n, sizeof(n), "%s.activations.%d.%s", prefix, idx, suffix);
    if (!au_get(s, n, &t, err, true)) return false;
    memcpy(dst, t.data, (size_t)AU_FILTER * sizeof(float));
    return true;
}

static bool au_amp_build(const cwick_safetensors *s, au_amp *b, const char *prefix,
                         int64_t ch, int64_t kern, cwick_error *err) {
    char n[192];
    cwick_st_tensor t;
    memset(b, 0, sizeof(*b));
    b->ch = ch;
    for (int d = 0; d < AU_DILS; d++) {
        const int dil = AU_RES_DIL[d];
        snprintf(n, sizeof(n), "%s.convs1.%d", prefix, d);
        if (!au_conv_build(s, &b->c1[d], n, ch, kern, 1, dil,
                           (int)((kern * dil - dil) / 2), err))
            return false;
        snprintf(n, sizeof(n), "%s.convs2.%d", prefix, d);
        if (!au_conv_build(s, &b->c2[d], n, ch, kern, 1, 1, (int)((kern - 1) / 2),
                           err))
            return false;
        for (int a = 0; a < 2; a++) {
            const int idx = d * 2 + a;
            snprintf(n, sizeof(n), "%s.activations.%d.act.alpha", prefix, idx);
            if (!au_get(s, n, &t, err, true)) return false;
            b->alpha[idx] = au_f32(ch);
            memcpy(b->alpha[idx], t.data, (size_t)ch * sizeof(float));
            snprintf(n, sizeof(n), "%s.activations.%d.act.beta", prefix, idx);
            if (!au_get(s, n, &t, err, true)) return false;
            b->beta[idx] = au_f32(ch);
            memcpy(b->beta[idx], t.data, (size_t)ch * sizeof(float));
            b->up[idx] = au_f32(AU_FILTER);
            b->dn[idx] = au_f32(AU_FILTER);
            if (!au_filter(s, prefix, idx, "upsample.filter", b->up[idx], err))
                return false;
            if (!au_filter(s, prefix, idx, "downsample.lowpass.filter", b->dn[idx],
                           err))
                return false;
        }
    }
    return true;
}

static void au_amp_free(au_amp *b) {
    for (int d = 0; d < AU_DILS; d++) {
        au_conv_free(&b->c1[d]);
        au_conv_free(&b->c2[d]);
        for (int a = 0; a < 2; a++) {
            const int idx = d * 2 + a;
            free(b->alpha[idx]);
            free(b->beta[idx]);
            free(b->up[idx]);
            free(b->dn[idx]);
        }
    }
}

static void au_amp_run(const au_amp *b, const float *x, int64_t T, float *out,
                       float *scratch) {
    const int64_t ch = b->ch;
    const int64_t n = ch * T;
    memcpy(out, x, (size_t)n * sizeof(float));
    float *tmp = scratch;          /* ch*T */
    float *res = scratch + n;      /* ch*T */
    for (int d = 0; d < AU_DILS; d++) {
        au_act(out, ch, T, b->alpha[d * 2], b->beta[d * 2], b->up[d * 2],
               b->dn[d * 2], tmp);
        au_conv1d_run(&b->c1[d], tmp, T, res);
        au_act(res, ch, T, b->alpha[d * 2 + 1], b->beta[d * 2 + 1],
               b->up[d * 2 + 1], b->dn[d * 2 + 1], tmp);
        memcpy(res, out, (size_t)n * sizeof(float));
        au_conv1d_run(&b->c2[d], tmp, T, out);
        for (int64_t i = 0; i < n; i++) out[i] += res[i];
    }
}

bool h3v_audio_amp_selftest(void) {
    enum { C = 2, T = 4 };
    au_amp b;
    memset(&b, 0, sizeof(b));
    b.ch = C;
    for (int d = 0; d < AU_DILS; d++) {
        b.c1[d].w = au_f32(3 * C * C);
        b.c2[d].w = au_f32(3 * C * C);
        b.c1[d].O = b.c1[d].I = C;
        b.c2[d].O = b.c2[d].I = C;
        b.c1[d].K = b.c2[d].K = 3;
        b.c1[d].dil = AU_RES_DIL[d];
        b.c2[d].dil = 1;
        b.c1[d].pad = (3 * AU_RES_DIL[d] - AU_RES_DIL[d]) / 2;
        b.c2[d].pad = 1;
        for (int i = 0; i < 3 * C * C; i++) b.c1[d].w[i] = 0.05f;
        for (int a = 0; a < 2; a++) {
            const int idx = d * 2 + a;
            b.alpha[idx] = au_f32(C);
            b.beta[idx] = au_f32(C);
            b.up[idx] = au_f32(AU_FILTER);
            b.dn[idx] = au_f32(AU_FILTER);
            for (int k = 0; k < AU_FILTER; k++)
                b.up[idx][k] = b.dn[idx][k] = 1.0f / AU_FILTER;
        }
    }
    float x[C * T], out[C * T], scratch[2 * C * T];
    for (int i = 0; i < C * T; i++) x[i] = 0.1f * (float)(i + 1);
    au_amp_run(&b, x, T, out, scratch);
    bool ok = true;
    for (int i = 0; i < C * T; i++)
        if (fabsf(out[i] - x[i]) > 1e-6f) ok = false;
    au_amp_free(&b);
    return ok;
}

/* Decode one channel: z is [32][T], out receives T*800 mono samples. */
static bool au_decode_mono(const cwick_safetensors *s, const float *z, int64_t T,
                           float *out, int64_t *out_len, cwick_error *err) {
    au_conv c;
    float *xcur = NULL;
    bool ok = false;

    /* 32 -> 2048, then 2048 -> 1024. The two are kept separate because only
     * conv_pre carries the 7-tap context. */
    if (!au_conv_build(s, &c, "dec_in_proj", AU_LATENT_CH, 1, 1, 1, 0, err))
        return false;
    float *h2k = au_f32(2048 * T);
    au_conv1d_run(&c, z, T, h2k);
    au_conv_free(&c);

    if (!au_conv_build(s, &c, "decoder.conv_pre", 2048, 7, 1, 1, 3, err)) {
        free(h2k);
        return false;
    }
    int64_t ch = c.O, len = T;
    xcur = au_f32(ch * len);
    au_conv1d_run(&c, h2k, len, xcur);
    au_conv_free(&c);
    free(h2k);

    for (int i = 0; i < AU_STAGES; i++) {
        char n[192];
        au_conv up;
        const int64_t in_ch = AU_PRE_CH >> i;
        const int64_t out_ch = AU_PRE_CH >> (i + 1);
        snprintf(n, sizeof(n), "decoder.ups.%d.0", i);
        if (!au_convt_build(s, &up, n, in_ch, out_ch, AU_UP_KERN[i],
                            AU_UP_RATE[i], (AU_UP_KERN[i] - AU_UP_RATE[i]) / 2, err))
            goto done;
        /* pad=(K-rate)/2 makes the transposed-conv output exactly len*rate. */
        float *nxt = au_f32(out_ch * len * AU_UP_RATE[i]);
        int64_t nlen = 0;
        au_convt_run(&up, xcur, len, nxt, &nlen);
        au_conv_free(&up);
        free(xcur);
        xcur = nxt;
        len = nlen;
        ch = out_ch;

        float *acc = au_f32(ch * len);
        float *blk = au_f32(ch * len);
        float *scratch = au_f32(2 * ch * len);
        memset(acc, 0, (size_t)ch * len * sizeof(float));
        for (int j = 0; j < AU_BLOCKS; j++) {
            au_amp amp;
            snprintf(n, sizeof(n), "decoder.resblocks.%d", i * AU_BLOCKS + j);
            if (!au_amp_build(s, &amp, n, ch, AU_RES_KERN[j], err)) {
                au_amp_free(&amp);
                free(acc); free(blk); free(scratch);
                goto done;
            }
            au_amp_run(&amp, xcur, len, blk, scratch);
            au_amp_free(&amp);
            for (int64_t q = 0; q < ch * len; q++) acc[q] += blk[q];
        }
        for (int64_t q = 0; q < ch * len; q++) acc[q] /= (float)AU_BLOCKS;
        free(blk);
        free(scratch);
        free(xcur);
        xcur = acc;
    }

    {
        float *alpha = au_f32(ch), *beta = au_f32(ch);
        float *up = au_f32(AU_FILTER), *dn = au_f32(AU_FILTER);
        float *act = au_f32(ch * len);
        cwick_st_tensor t;
        bool post_ok = false;

        if (au_get(s, "decoder.activation_post.act.alpha", &t, err, true)) {
            memcpy(alpha, t.data, (size_t)ch * sizeof(float));
            if (au_get(s, "decoder.activation_post.act.beta", &t, err, true)) {
                memcpy(beta, t.data, (size_t)ch * sizeof(float));
                if (au_get(s, "decoder.activation_post.upsample.filter", &t, err, true)) {
                    memcpy(up, t.data, (size_t)AU_FILTER * sizeof(float));
                    if (au_get(s, "decoder.activation_post.downsample.lowpass.filter",
                               &t, err, true)) {
                        memcpy(dn, t.data, (size_t)AU_FILTER * sizeof(float));
                        post_ok = true;
                    }
                }
            }
        }
        if (post_ok) {
            au_act(xcur, ch, len, alpha, beta, up, dn, act);
            /* conv_post has no bias in this checkpoint (use_bias_at_final is
             * false), which au_conv_build accepts. */
            if (au_conv_build(s, &c, "decoder.conv_post", ch, 7, 1, 1, 3, err)) {
                au_conv1d_run(&c, act, len, out);
                au_conv_free(&c);
                for (int64_t i = 0; i < len; i++) {
                    if (out[i] < -1.0f) out[i] = -1.0f;
                    if (out[i] > 1.0f) out[i] = 1.0f;
                }
                *out_len = len;
                ok = true;
            }
        }
        free(alpha); free(beta); free(up); free(dn); free(act);
    }

done:
    free(xcur);
    return ok;
}

static void au_lat_denorm(float *z, int64_t t) {
    for (int64_t c = 0; c < AU_LATENT_CH; c++)
        for (int64_t i = 0; i < t; i++)
            z[c * t + i] = z[c * t + i] * AU_LAT_STD[c] + AU_LAT_MEAN[c];
}

bool h3v_audio_latnorm_selftest(void) {
    float z[AU_LATENT_CH * 2] = {0};
    z[5 * 2] = 1.0f;
    z[15 * 2 + 1] = -1.0f;
    au_lat_denorm(z, 2);
    return fabsf(z[5 * 2] - (AU_LAT_STD[5] + AU_LAT_MEAN[5])) < 1e-6f &&
           fabsf(z[15 * 2 + 1] - (-AU_LAT_STD[15] + AU_LAT_MEAN[15])) < 1e-6f;
}

float *h3v_audio_decode(const h3v_model *m, const float *z, int64_t t,
                        int64_t *samples, h3v_progress_fn cb, void *cbctx,
                        cwick_error *err) {
    const cwick_safetensors *s = h3v_shards_avae(m);
    if (!s) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "audio VAE shards are not open");
        return NULL;
    }
    if (t < 2 || (t & 1)) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "audio latent length must be an even row count");
        return NULL;
    }
    const int64_t frames = t / 2;
    const int64_t cap = frames * AU_HOP + AU_HOP;
    float *zd = au_f32(AU_LATENT_CH * t);
    if (!zd) return NULL;
    memcpy(zd, z, (size_t)AU_LATENT_CH * t * sizeof(float));
    au_lat_denorm(zd, t);
    float *left = au_f32(AU_LATENT_CH * frames);
    float *right = au_f32(AU_LATENT_CH * frames);
    for (int64_t c = 0; c < AU_LATENT_CH; c++)
        for (int64_t i = 0; i < frames; i++) {
            left[c * frames + i] = zd[c * t + 2 * i];
            right[c * frames + i] = zd[c * t + 2 * i + 1];
        }
    free(zd);

    int64_t llen = 0, rlen = 0;
    float *lbuf = au_f32(cap);
    float *rbuf = au_f32(cap);
    bool ok = au_decode_mono(s, left, frames, lbuf, &llen, err);
    if (ok) {
        if (cb) cb(cbctx, H3V_PHASE_AUDIO, 0, 0, 0.5);
        ok = au_decode_mono(s, right, frames, rbuf, &rlen, err);
    }
    free(left);
    free(right);
    if (!ok) {
        free(lbuf);
        free(rbuf);
        return NULL;
    }
    const int64_t n = llen < rlen ? llen : rlen;
    float *out = au_f32(n * 2);
    for (int64_t i = 0; i < n; i++) {
        out[2 * i] = lbuf[i];
        out[2 * i + 1] = rbuf[i];
    }
    free(lbuf);
    free(rbuf);
    *samples = n;
    if (cb) cb(cbctx, H3V_PHASE_AUDIO, 0, 0, 1.0);
    return out;
}
