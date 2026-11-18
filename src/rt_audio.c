/* rt_audio.c — the thinker's ear, in one pass: Whisper log-mel, the Qwen3-ASR
 * audio tower and the stacked SwiGLU projector.
 *
 * Every constant here was read out of the checkpoint's own audio/config.json
 * rather than assumed: 128 mel bins, three stride-2 3x3 convs (480 channels),
 * conv_out onto 896, 18 post-LayerNorm encoder layers with 14 heads and block
 * local (non causal) attention, then stack 2 -> RMSNorm -> 4096 -> SwiGLU ->
 * RMSNorm -> 5120.
 *
 * Three details decide whether the result is audio or noise, and they are
 * spelled out where they happen:
 *   - the STFT is centred with reflect padding and its last frame is dropped,
 *     so `frames == samples / 160` exactly; a clip padded to a multiple of
 *     2560 samples therefore has no partially valid mel frame;
 *   - the tower's positional encoding is 13 long and is added per 100-frame
 *     chunk after conv_out, never to the layer inputs;
 *   - a window shorter than 3000 frames is zero padded *before* the convs, and
 *     the padded chunks are dropped again right after conv_out.
 *
 * The arithmetic runs on Accelerate. The tower is ~200 M parameters and a
 * 3-second turn costs a few tens of milliseconds there, so it does not warrant
 * a Metal path of its own. */
#include "rt_audio.h"

#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"
#include "log.h"
#include "util.h"

#define EA_TAG "ear"

#define MEL_BINS 128
#define N_FFT 400
#define HOP 160
#define N_FREQ (N_FFT / 2 + 1) /* 201 */
#define CHUNK 100              /* n_window * 2 mel frames per conv chunk */
#define CHUNK_POST 13          /* frames left per chunk after three convs */
#define WIN_FRAMES 3000        /* the tower's per-call window */

typedef struct {
    float *w; /* [out][in] */
    float *b;
    int64_t out, in;
} ea_lin;

typedef struct {
    ea_lin q, k, v, o, fc1, fc2;
    float *ln1w, *ln1b, *ln2w, *ln2b;
} ea_layer;

struct rt_ear {
    cwick_safetensors saf;

    int mel, d, ffn, nl, heads, hd, nwin, nwin_infer, stack, hid, dout;
    /* conv frontend, weights kept in PyTorch layout [out][in][3][3] */
    float *c1w, *c1b, *c2w, *c2b, *c3w, *c3b;
    int c1o, c2o, c3o;
    ea_lin conv_out;
    ea_layer *lay;
    float *lnpw, *lnpb, *pos;
    float *prew, *lin1w, *midw, *lin2w;

    float *melmat;      /* [201][128] slaney filterbank */
    float *dftc, *dfts; /* [201][400] DFT basis */

    /* scratch for one window */
    int rows;      /* capacity in tower frames */
    int block;     /* attention block length in tower frames */
    float *wmel;   /* [128][nchunks*100] zero-padded window */
    float *col;    /* im2col staging */
    float *cv1, *cv2, *cv3;
    float *stage;  /* [rows][d]: tower states, normed in place */
    float *packed; /* alias of stage, the layout the layers walk */
    float *xn, *q, *k, *v, *att, *ao, *ff;
    float *gq, *gk; /* gathered [rows][hd] for one head */
    float *scores;
};

static void *xcalloc(size_t n, size_t sz) {
    void *p = cwick_xmalloc(n * sz);
    memset(p, 0, n * sz);
    return p;
}

static const cwick_st_tensor *fetch(rt_ear *e, const char *name) {
    cwick_st_tensor *slot = (cwick_st_tensor *)cwick_xmalloc(sizeof(*slot));
    cwick_error err;
    memset(&err, 0, sizeof(err));
    if (cwick_safetensors_get(&e->saf, name, slot, &err) != CWICK_OK) {
        CWICK_INFO(EA_TAG, "missing tensor %s", name);
        free(slot);
        return NULL;
    }
    return slot;
}

/* Widen a whole tensor to f32 once. The tower is walked every turn and its
 * largest matrix is 3584x896, so a per-call widen would cost more than the
 * multiply it feeds. */
static float *widen(const cwick_st_tensor *st) {
    int64_t n = st->rows * st->cols;
    float *out = (float *)xcalloc((size_t)n, sizeof(float));
    if (st->dtype == CWICK_ST_BF16) {
        const uint16_t *p = (const uint16_t *)st->data;
        for (int64_t i = 0; i < n; i++) out[i] = cwick_bf16_fast(p[i]);
    } else if (st->dtype == CWICK_ST_F32) {
        memcpy(out, st->data, sizeof(float) * (size_t)n);
    } else {
        CWICK_INFO(EA_TAG, "%s: unexpected dtype %d", st->name, (int)st->dtype);
    }
    return out;
}

static int bind_lin(rt_ear *e, ea_lin *dst, const char *name) {
    const cwick_st_tensor *w = fetch(e, name);
    if (!w) return -1;
    dst->w = widen(w);
    dst->in = w->cols;
    dst->out = w->rows;
    return 0;
}

static float *bind_vec(rt_ear *e, const char *name) {
    const cwick_st_tensor *v = fetch(e, name);
    if (!v) return NULL;
    return widen(v);
}

/* ------------------------------------------------------------ frontend --- */

static double hertz_to_mel(double f) {
    const double f_sp = 200.0 / 3.0;
    const double min_log_hz = 1000.0;
    const double min_log_mel = min_log_hz / f_sp;
    const double logstep = log(6.4) / 27.0;
    if (f >= min_log_hz) return min_log_mel + log(f / min_log_hz) / logstep;
    return f / f_sp;
}

static double mel_to_hertz(double m) {
    const double f_sp = 200.0 / 3.0;
    const double min_log_hz = 1000.0;
    const double min_log_mel = min_log_hz / f_sp;
    const double logstep = log(6.4) / 27.0;
    if (m >= min_log_mel) return min_log_hz * exp(logstep * (m - min_log_mel));
    return f_sp * m;
}

/* Slaney-normalised triangular filterbank, [N_FREQ][128] -- the table the
 * reference feature extractor rebuilds on every call. */
static float *build_melmat(void) {
    const int nf = MEL_BINS;
    double edges[130];
    double lo = hertz_to_mel(0.0), hi = hertz_to_mel(8000.0);
    for (int i = 0; i < nf + 2; i++)
        edges[i] = mel_to_hertz(lo + (hi - lo) * (double)i / (double)(nf + 1));
    float *m = (float *)xcalloc((size_t)N_FREQ * nf, sizeof(float));
    for (int f = 0; f < N_FREQ; f++) {
        double hz = 8000.0 * (double)f / (double)(N_FREQ - 1);
        for (int j = 0; j < nf; j++) {
            double lower = (hz - edges[j]) / (edges[j + 1] - edges[j]);
            double upper = (edges[j + 2] - hz) / (edges[j + 2] - edges[j + 1]);
            double v = lower < upper ? lower : upper;
            if (v < 0.0) v = 0.0;
            v *= 2.0 / (edges[j + 2] - edges[j]); /* unit area per filter */
            m[(int64_t)f * nf + j] = (float)v;
        }
    }
    return m;
}

/* Reflection about the first/last sample: the padding the reference STFT
 * uses, which never repeats the edge sample. */
static float reflect_at(const float *pcm, int n, int i) {
    while (i < 0 || i >= n) {
        if (i < 0) i = -i;
        if (i >= n) i = 2 * (n - 1) - i;
    }
    return pcm[i];
}

int rt_ear_logmel(const rt_ear *e, const float *pcm, int samples, float *out) {
    if (samples < N_FFT) return -1;
    const int frames = samples / HOP;
    if (frames <= 0) return -1;
    /* Periodic Hann: the 401-point window minus its last sample. */
    float win[N_FFT];
    for (int i = 0; i < N_FFT; i++)
        win[i] = (float)(0.5 - 0.5 * cos(2.0 * M_PI * (double)i / (double)N_FFT));

    float *frame = (float *)xcalloc((size_t)N_FFT * frames, sizeof(float));
    for (int t = 0; t < frames; t++) {
        int base = t * HOP - N_FFT / 2;
        for (int i = 0; i < N_FFT; i++)
            frame[(int64_t)i * frames + t] =
                reflect_at(pcm, samples, base + i) * win[i];
    }
    /* Power spectrum as two real GEMMs against the DFT basis. 201 x 400 x T is
     * small enough that a hand-written transform would not beat BLAS. */
    float *re = (float *)xcalloc((size_t)N_FREQ * frames, sizeof(float));
    float *pw = (float *)xcalloc((size_t)N_FREQ * frames, sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N_FREQ, frames, N_FFT, 1.0f,
                e->dftc, N_FFT, frame, frames, 0.0f, re, frames);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N_FREQ, frames, N_FFT, 1.0f,
                e->dfts, N_FFT, frame, frames, 0.0f, pw, frames);
    for (int64_t i = 0; i < (int64_t)N_FREQ * frames; i++)
        pw[i] = re[i] * re[i] + pw[i] * pw[i];
    free(re);
    free(frame);

    /* mel[m][t] = sum_k M[k][m] * power[k][t] */
    float *mel = (float *)xcalloc((size_t)MEL_BINS * frames, sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, MEL_BINS, frames, N_FREQ, 1.0f,
                e->melmat, MEL_BINS, pw, frames, 0.0f, mel, frames);
    free(pw);

    double mx = -1e30;
    for (int64_t i = 0; i < (int64_t)MEL_BINS * frames; i++) {
        float v = mel[i] < 1e-10f ? 1e-10f : mel[i];
        float l = log10f(v);
        mel[i] = l;
        if (l > mx) mx = l;
    }
    const float floor_db = (float)mx - 8.0f;
    for (int64_t i = 0; i < (int64_t)MEL_BINS * frames; i++) {
        float v = mel[i] > floor_db ? mel[i] : floor_db;
        mel[i] = (v + 4.0f) / 4.0f;
    }
    memcpy(out, mel, sizeof(float) * (size_t)MEL_BINS * frames);
    free(mel);
    return frames;
}

/* --------------------------------------------------------------- tower --- */

static void gelu(float *p, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        float x = p[i];
        p[i] = 0.5f * x * (1.0f + erff(x * 0.70710678f));
    }
}

/* One 3x3/stride-2/pad-1 conv over a (mel, time) grid, as im2col + GEMM.
 *
 * The input is [cin][h][pitch] and only the first `wd` columns of each row are
 * read: the audio window is one long mel-major strip sliced into chunks, so
 * consecutive chunks are `CHUNK` columns apart inside the same row, not a
 * contiguous block of floats. `pitch` carries that row stride. */
static void conv2d(const float *w, const float *b, int cin, int cout, const float *in,
                   int h, int wd, int pitch, float *out, float *col, int *ho, int *wo) {
    const int oh = (h - 1) / 2 + 1, ow = (wd - 1) / 2 + 1;
    const int kk = cin * 9;
    const int spatial = oh * ow;
    for (int t = 0; t < kk; t++) {
        int ci = t / 9, ki = (t % 9) / 3, kj = t % 3;
        const float *src = in + (int64_t)ci * h * pitch;
        float *row = col + (int64_t)t * spatial;
        for (int oi = 0; oi < oh; oi++) {
            int hi = oi * 2 - 1 + ki;
            if (hi < 0 || hi >= h) {
                memset(row + (int64_t)oi * ow, 0, sizeof(float) * (size_t)ow);
                continue;
            }
            for (int oj = 0; oj < ow; oj++) {
                int wj = oj * 2 - 1 + kj;
                row[(int64_t)oi * ow + oj] =
                    (wj < 0 || wj >= wd) ? 0.0f : src[(int64_t)hi * pitch + wj];
            }
        }
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, cout, spatial, kk, 1.0f, w,
                kk, col, spatial, 0.0f, out, spatial);
    if (b)
        for (int c = 0; c < cout; c++) {
            float bv = b[c];
            float *dst = out + (int64_t)c * spatial;
            for (int i = 0; i < spatial; i++) dst[i] += bv;
        }
    *ho = oh;
    *wo = ow;
}

static void layernorm(float *dst, const float *src, const float *w, const float *b,
                      int n) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)src[i];
    /* LayerNorm centres the variance; the projector's RMSNorm does not. */
    float mean = (float)(ss / n);
    double sq = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)src[i] - mean;
        sq += d * d;
    }
    float r = 1.0f / sqrtf((float)(sq / n) + 1e-5f);
    for (int i = 0; i < n; i++) dst[i] = ((float)((double)src[i] - mean)) * r * w[i] + b[i];
}

static void rmsnorm_row(float *dst, const float *src, const float *w, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)src[i] * src[i];
    float r = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) dst[i] = src[i] * r * w[i];
}

/* y[m][n] = x[m][k] @ W[n][k]^T + b */
static void lin_apply(const ea_lin *L, const float *x, float *y, int m) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, (int)L->out, (int)L->in, 1.0f,
                x, (int)L->in, L->w, (int)L->in, 0.0f, y, (int)L->out);
    if (L->b)
        for (int r = 0; r < m; r++) {
            float *row = y + (int64_t)r * L->out;
            for (int64_t i = 0; i < L->out; i++) row[i] += L->b[i];
        }
}

/* Frames left after three stride-2 convs of a chunk. */
static int post_cnn(int len) {
    for (int i = 0; i < 3; i++) len = len > 0 ? (len - 1) / 2 + 1 : 0;
    return len;
}

/* One window of mel [128][pitch] (columns beyond `frames` already zero padded
 * to a whole number of chunks) -> tower states [rows][d], invalid tail frames
 * dropped chunk by chunk exactly as the reference does before the
 * transformer. */
static int tower_forward(rt_ear *e, const float *mel, int pitch, int chunks, int frames,
                         float *out) {
    const int d = e->d, hd = e->hd, nh = e->heads;
    int rows = 0;
    for (int c = 0; c < chunks; c++) {
        const float *chunk = mel + c * CHUNK;
        int ho, wo;
        conv2d(e->c1w, e->c1b, 1, e->c1o, chunk, MEL_BINS, CHUNK, pitch, e->cv1, e->col, &ho,
               &wo);
        gelu(e->cv1, (int64_t)e->c1o * ho * wo);
        conv2d(e->c2w, e->c2b, e->c1o, e->c2o, e->cv1, ho, wo, wo, e->cv2, e->col, &ho, &wo);
        gelu(e->cv2, (int64_t)e->c2o * ho * wo);
        conv2d(e->c3w, e->c3b, e->c2o, e->c3o, e->cv2, ho, wo, wo, e->cv3, e->col, &ho, &wo);
        gelu(e->cv3, (int64_t)e->c3o * ho * wo);
        /* [C][F][T] -> [T][C*F]: conv_out reads channels, then mel bins. */
        const int F = ho, T = wo, C = e->c3o;
        float *x = e->col;
        for (int t = 0; t < T; t++)
            for (int c2 = 0; c2 < C; c2++)
                for (int f = 0; f < F; f++)
                    x[((int64_t)t * C + c2) * F + f] =
                        e->cv3[((int64_t)c2 * F + f) * T + t];
        float *y = e->packed + (int64_t)rows * d;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, d, C * F, 1.0f, x, C * F,
                    e->conv_out.w, C * F, 0.0f, y, d);
        for (int t = 0; t < T; t++) {
            float *row = y + (int64_t)t * d;
            for (int i = 0; i < d; i++) row[i] += e->pos[(int64_t)t * d + i];
        }
        const char *dd = getenv("RT_DUMP_DIR");
        if (dd && c == 0) {
            char pp[1200];
            snprintf(pp, sizeof(pp), "%s/ear_conv.f32", dd);
            FILE *f = fopen(pp, "wb");
            if (f) {
                fwrite(y, sizeof(float), (size_t)T * d, f);
                fclose(f);
            }
        }
        /* Only the frames backed by real audio reach the transformer: the last
         * chunk of a clip is usually short, and its zero-padded tail must not
         * become tokens. */
        int valid = frames - c * CHUNK;
        if (valid > CHUNK) valid = CHUNK;
        int keep = post_cnn(valid);
        if (keep > T) keep = T;
        rows += keep;
    }
    if (rows <= 0) return 0;

    const int block = e->block;
    for (int l = 0; l < e->nl; l++) {
        ea_layer *L = &e->lay[l];
        for (int r = 0; r < rows; r++)
            layernorm(e->xn + (int64_t)r * d, e->packed + (int64_t)r * d, L->ln1w, L->ln1b, d);
        if (l == 0) {
            const char *dd = getenv("RT_DUMP_DIR");
            if (dd) {
                char pp[1200];
                snprintf(pp, sizeof(pp), "%s/ear_ln1.f32", dd);
                FILE *f = fopen(pp, "wb");
                if (f) { fwrite(e->xn, sizeof(float), (size_t)rows * d, f); fclose(f); }
            }
        }
        lin_apply(&L->q, e->xn, e->q, rows);
        lin_apply(&L->k, e->xn, e->k, rows);
        lin_apply(&L->v, e->xn, e->v, rows);
        const float scale = 1.0f / sqrtf((float)hd);
        for (int s = 0; s < rows; s += block) {
            int n = rows - s < block ? rows - s : block;
            for (int h = 0; h < nh; h++) {
                for (int i = 0; i < n; i++) {
                    memcpy(e->gq + (int64_t)i * hd, e->q + (int64_t)(s + i) * d + h * hd,
                           sizeof(float) * hd);
                    memcpy(e->gk + (int64_t)i * hd, e->k + (int64_t)(s + i) * d + h * hd,
                           sizeof(float) * hd);
                }
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, n, hd, scale, e->gq,
                            hd, e->gk, hd, 0.0f, e->scores, n);
                for (int i = 0; i < n; i++) {
                    float *row = e->scores + (int64_t)i * n;
                    float mx = -INFINITY;
                    for (int j = 0; j < n; j++)
                        if (row[j] > mx) mx = row[j];
                    float sum = 0.0f;
                    for (int j = 0; j < n; j++) {
                        row[j] = expf(row[j] - mx);
                        sum += row[j];
                    }
                    float inv = 1.0f / sum;
                    for (int j = 0; j < n; j++) row[j] *= inv;
                }
                for (int i = 0; i < n; i++)
                    memcpy(e->gq + (int64_t)i * hd, e->v + (int64_t)(s + i) * d + h * hd,
                           sizeof(float) * hd);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, hd, n, 1.0f,
                            e->scores, n, e->gq, hd, 0.0f, e->gk, hd);
                for (int i = 0; i < n; i++)
                    memcpy(e->att + (int64_t)(s + i) * d + h * hd, e->gk + (int64_t)i * hd,
                           sizeof(float) * hd);
            }
        }
        if (l == 0) {
            const char *dd = getenv("RT_DUMP_DIR");
            if (dd) {
                char pp[1200];
                snprintf(pp, sizeof(pp), "%s/ear_att.f32", dd);
                FILE *f = fopen(pp, "wb");
                if (f) { fwrite(e->att, sizeof(float), (size_t)rows * d, f); fclose(f); }
            }
        }
        lin_apply(&L->o, e->att, e->ao, rows);
        for (int64_t i = 0; i < (int64_t)rows * d; i++) e->packed[i] += e->ao[i];

        for (int r = 0; r < rows; r++)
            layernorm(e->xn + (int64_t)r * d, e->packed + (int64_t)r * d, L->ln2w, L->ln2b, d);
        lin_apply(&L->fc1, e->xn, e->ff, rows);
        gelu(e->ff, (int64_t)rows * e->ffn);
        lin_apply(&L->fc2, e->ff, e->ao, rows);
        for (int64_t i = 0; i < (int64_t)rows * d; i++) e->packed[i] += e->ao[i];
        const char *ld = getenv("RT_DUMP_LAYERS");
        if (ld) {
            char pp[1200];
            snprintf(pp, sizeof(pp), "%s/ear_l%d.f32", ld, l);
            FILE *f = fopen(pp, "wb");
            if (f) {
                fwrite(e->packed, sizeof(float), (size_t)rows * d, f);
                fclose(f);
            }
        }
    }
    if (getenv("RT_DUMP_DIR")) {
        char pp[1200];
        snprintf(pp, sizeof(pp), "%s/ear_pre_lnpost.f32", getenv("RT_DUMP_DIR"));
        FILE *f = fopen(pp, "wb");
        if (f) {
            fwrite(e->packed, sizeof(float), (size_t)rows * d, f);
            fclose(f);
        }
    }
    for (int r = 0; r < rows; r++)
        layernorm(out + (int64_t)r * d, e->packed + (int64_t)r * d, e->lnpw, e->lnpb, d);
    return rows;
}

/* ----------------------------------------------------------- projector --- */

/* stack 2 -> RMSNorm(1792) -> 4096 -> SwiGLU -> RMSNorm(2048) -> 5120. */
static int projector(rt_ear *e, const float *h, int rows, float *out) {
    const int din = e->d * e->stack;
    int t = rows / e->stack;
    const int dh = e->hid / 2;
    float *tmp = (float *)xcalloc((size_t)t * din, sizeof(float));
    float *w = (float *)xcalloc((size_t)t * e->hid, sizeof(float));
    float *g = (float *)xcalloc((size_t)t * dh, sizeof(float));
    for (int r = 0; r < t; r++)
        rmsnorm_row(tmp + (int64_t)r * din, h + (int64_t)r * din, e->prew, din, 1e-6f);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, t, e->hid, din, 1.0f, tmp, din,
                e->lin1w, din, 0.0f, w, e->hid);
    for (int r = 0; r < t; r++) {
        const float *row = w + (int64_t)r * e->hid;
        float *dst = g + (int64_t)r * dh;
        for (int i = 0; i < dh; i++) {
            float a = row[i], b = row[dh + i];
            dst[i] = a * (b / (1.0f + expf(-b)));
        }
        rmsnorm_row(dst, dst, e->midw, dh, 1e-6f);
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, t, e->dout, dh, 1.0f, g, dh,
                e->lin2w, dh, 0.0f, out, e->dout);
    free(g);
    free(w);
    free(tmp);
    return t;
}

int rt_ear_token_count(const rt_ear *e, int frames) {
    (void)e;
    if (frames <= 0) return 0;
    return (frames + 15) / 16;
}

int rt_ear_frames_for_samples(int samples) { return samples / HOP; }

int rt_ear_embed(rt_ear *e, const float *pcm, int samples, float **out, int *tokens) {
    const int d = e->d;
    int frames = rt_ear_frames_for_samples(samples);
    if (frames <= 0) return -1;
    float *mel = (float *)xcalloc((size_t)MEL_BINS * frames, sizeof(float));
    if (rt_ear_logmel(e, pcm, samples, mel) <= 0) {
        free(mel);
        return -1;
    }
    int cap = rt_ear_token_count(e, frames) + 8;
    float *vec = (float *)xcalloc((size_t)cap * e->dout, sizeof(float));
    int produced = 0;
    for (int off = 0; off < frames; off += WIN_FRAMES) {
        int len = frames - off < WIN_FRAMES ? frames - off : WIN_FRAMES;
        int chunks = (len + CHUNK - 1) / CHUNK;
        int win_mel = chunks * CHUNK;
        /* The window is a mel-major strip [128][win_mel]; each chunk is then a
         * column slice at c*CHUNK inside every row, which is why the convs take
         * an explicit row stride instead of a compact block. The frames past the
         * clip stay normalised zeros, so the padded tail of the last chunk can
         * never become audio tokens. */
        memset(e->wmel, 0, sizeof(float) * (size_t)MEL_BINS * win_mel);
        for (int f = 0; f < MEL_BINS; f++)
            memcpy(e->wmel + (int64_t)f * win_mel, mel + (int64_t)f * frames + off,
                   sizeof(float) * (size_t)len);
        int rows = tower_forward(e, e->wmel, win_mel, chunks, len, e->stage);
        if (rows <= 0) continue;
        /* RT_DUMP_DIR keeps the tower output so it can be bisected against an
         * independent implementation instead of only the final embeddings. */
        const char *dd = getenv("RT_DUMP_DIR");
        if (dd) {
            char p[1200];
            snprintf(p, sizeof(p), "%s/ear_tower.f32", dd);
            FILE *f = fopen(p, "wb");
            if (f) {
                fwrite(e->stage, sizeof(float), (size_t)rows * d, f);
                fclose(f);
            }
        }
        int even = rows + (rows & 1);
        if (even > rows) memset(e->stage + (int64_t)rows * d, 0, sizeof(float) * d);
        int t = projector(e, e->stage, even, vec + (int64_t)produced * e->dout);
        int cap_tok = (len + 15) / 16;
        if (t > cap_tok) t = cap_tok;
        produced += t;
    }
    free(mel);
    *out = (float *)cwick_xmalloc(sizeof(float) * (size_t)produced * e->dout);
    memcpy(*out, vec, sizeof(float) * (size_t)produced * e->dout);
    free(vec);
    *tokens = produced;
    return 0;
}

int rt_ear_hidden(const rt_ear *e) { return e->dout; }
int rt_ear_frame_tokens(const rt_ear *e) { return e->block; }

/* ---------------------------------------------------------------- wav --- */

/* Minimal RIFF reader: 16-bit PCM, mono or interleaved stereo (mixed down).
 * The chunk list is walked instead of assuming a 44-byte header, because
 * files carrying a LIST/INFO block push `data` further out. */
int rt_wav_load(const char *path, float **pcm, int *rate) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    unsigned char h[12];
    if (fread(h, 1, 12, f) != 12 || memcmp(h, "RIFF", 4) || memcmp(h + 8, "WAVE", 4)) {
        fclose(f);
        return -1;
    }
    int ch = 1, bits = 16, got_rate = 0;
    unsigned char *data = NULL;
    size_t dsz = 0;
    for (;;) {
        unsigned char ck[8];
        if (fread(ck, 1, 8, f) != 8) break;
        unsigned int sz = (unsigned int)ck[4] | ((unsigned int)ck[5] << 8) |
                          ((unsigned int)ck[6] << 16) | ((unsigned int)ck[7] << 24);
        if (!memcmp(ck, "fmt ", 4)) {
            unsigned char fmt[16];
            size_t want = sz < sizeof(fmt) ? sz : sizeof(fmt);
            if (fread(fmt, 1, want, f) != want) break;
            if (sz > want) fseek(f, (long)(sz - want), SEEK_CUR);
            ch = fmt[2] | (fmt[3] << 8);
            got_rate = fmt[4] | (fmt[5] << 8) | (fmt[6] << 16) | (fmt[7] << 24);
            bits = fmt[14] | (fmt[15] << 8);
        } else if (!memcmp(ck, "data", 4)) {
            data = (unsigned char *)malloc(sz ? sz : 1);
            dsz = fread(data, 1, sz, f);
            break;
        } else {
            fseek(f, (long)sz + (long)(sz & 1), SEEK_CUR);
        }
    }
    fclose(f);
    if (!data || bits != 16 || ch < 1) {
        free(data);
        return -1;
    }
    size_t n = dsz / 2 / (size_t)ch;
    float *out = (float *)cwick_xmalloc(sizeof(float) * (n ? n : 1));
    const short *s = (const short *)data;
    for (size_t i = 0; i < n; i++) {
        int acc = 0;
        for (int c = 0; c < ch; c++) acc += s[i * (size_t)ch + (size_t)c];
        out[i] = (float)acc / (32768.0f * (float)ch);
    }
    free(data);
    *pcm = out;
    if (rate) *rate = got_rate;
    return (int)n;
}

/* --------------------------------------------------------------- open --- */

static int cfg_i(json_value *o, const char *k, int dflt) {
    json_value *v = o ? cwick_json_get(o, k) : NULL;
    return v ? (int)cwick_json_int(v, dflt) : dflt;
}

rt_ear *rt_ear_open(const char *thinker_dir, cwick_error *err) {
    char path[1200];
    cwick_path_join(path, sizeof(path), thinker_dir, "audio/config.json");
    char *text = NULL;
    size_t len = 0;
    if (cwick_read_file(path, &text, &len) != 0) {
        if (err) cwick_error_set(err, CWICK_ERR_IO, "cannot read %s", path);
        return NULL;
    }
    json_value *root = cwick_json_parse(text, len, err);
    free(text);
    if (!root) return NULL;
    json_value *ac = cwick_json_get(root, "audio_config");
    json_value *tc = cwick_json_get(root, "text_config");

    rt_ear *e = (rt_ear *)xcalloc(1, sizeof(*e));
    e->d = cfg_i(ac, "d_model", 896);
    e->ffn = cfg_i(ac, "encoder_ffn_dim", 3584);
    e->nl = cfg_i(ac, "encoder_layers", 18);
    e->heads = cfg_i(ac, "encoder_attention_heads", 14);
    e->hd = e->d / e->heads;
    e->mel = cfg_i(ac, "num_mel_bins", MEL_BINS);
    e->nwin = cfg_i(ac, "n_window", 50);
    e->nwin_infer = cfg_i(ac, "n_window_infer", 800);
    e->stack = cfg_i(root, "stack_factor", 2);
    e->hid = cfg_i(root, "hidden_size", 4096);
    e->dout = tc ? cfg_i(tc, "hidden_size", 5120) : 5120;
    e->c1o = cfg_i(ac, "downsample_hidden_size", 480);
    e->c2o = e->c3o = e->c1o;
    cwick_json_free(root);
    if (e->mel != MEL_BINS) {
        CWICK_INFO(EA_TAG, "mel bins %d is not the 128 the frontend implements", e->mel);
        free(e);
        return NULL;
    }

    char adir[1200];
    cwick_path_join(adir, sizeof(adir), thinker_dir, "audio");
    if (cwick_safetensors_open(adir, &e->saf, err) != CWICK_OK) {
        free(e);
        return NULL;
    }

    const cwick_st_tensor *c1 = fetch(e, "audio_tower.conv2d1.weight");
    const cwick_st_tensor *c2 = fetch(e, "audio_tower.conv2d2.weight");
    const cwick_st_tensor *c3 = fetch(e, "audio_tower.conv2d3.weight");
    if (!c1 || !c2 || !c3) {
        rt_ear_close(e);
        return NULL;
    }
    e->c1w = widen(c1);
    e->c2w = widen(c2);
    e->c3w = widen(c3);
    e->c1b = bind_vec(e, "audio_tower.conv2d1.bias");
    e->c2b = bind_vec(e, "audio_tower.conv2d2.bias");
    e->c3b = bind_vec(e, "audio_tower.conv2d3.bias");
    e->lnpw = bind_vec(e, "audio_tower.ln_post.weight");
    e->lnpb = bind_vec(e, "audio_tower.ln_post.bias");
    if (bind_lin(e, &e->conv_out, "audio_tower.conv_out.weight") || !e->lnpw || !e->lnpb) {
        rt_ear_close(e);
        return NULL;
    }

    /* Sinusoidal table: log(10000)/(channels/2 - 1) spacing, sin then cos. */
    e->pos = (float *)xcalloc((size_t)CHUNK_POST * e->d, sizeof(float));
    {
        const int half = e->d / 2;
        double inc = log(10000.0) / (double)(half - 1);
        double *inv = (double *)cwick_xmalloc(sizeof(double) * (size_t)half);
        for (int i = 0; i < half; i++) inv[i] = exp(-inc * (double)i);
        for (int t = 0; t < CHUNK_POST; t++)
            for (int i = 0; i < half; i++) {
                double a = (double)t * inv[i];
                e->pos[(int64_t)t * e->d + i] = (float)sin(a);
                e->pos[(int64_t)t * e->d + half + i] = (float)cos(a);
            }
        free(inv);
    }

    e->lay = (ea_layer *)xcalloc((size_t)e->nl, sizeof(ea_layer));
    for (int l = 0; l < e->nl; l++) {
        char n[256];
        ea_layer *L = &e->lay[l];
        struct { ea_lin *dst; const char *sfx; } lins[] = {
            {&L->q, "self_attn.q_proj"},  {&L->k, "self_attn.k_proj"},
            {&L->v, "self_attn.v_proj"},  {&L->o, "self_attn.out_proj"},
            {&L->fc1, "fc1"},             {&L->fc2, "fc2"},
        };
        for (int i = 0; i < 6; i++) {
            snprintf(n, sizeof(n), "audio_tower.layers.%d.%s.weight", l, lins[i].sfx);
            if (bind_lin(e, lins[i].dst, n)) {
                rt_ear_close(e);
                return NULL;
            }
            snprintf(n, sizeof(n), "audio_tower.layers.%d.%s.bias", l, lins[i].sfx);
            lins[i].dst->b = bind_vec(e, n);
        }
        struct { const char *sfx; float **dst; } norms[] = {
            {"self_attn_layer_norm.weight", &L->ln1w},
            {"self_attn_layer_norm.bias", &L->ln1b},
            {"final_layer_norm.weight", &L->ln2w},
            {"final_layer_norm.bias", &L->ln2b},
        };
        for (int i = 0; i < 4; i++) {
            snprintf(n, sizeof(n), "audio_tower.layers.%d.%s", l, norms[i].sfx);
            *norms[i].dst = bind_vec(e, n);
            if (!*norms[i].dst) {
                rt_ear_close(e);
                return NULL;
            }
        }
    }

    /* Projector: ln_pre spans both stacked rows, so its weight is 2*d long. */
    e->prew = bind_vec(e, "multi_modal_projector.ln_pre.weight");
    e->midw = bind_vec(e, "multi_modal_projector.ln_mid.weight");
    ea_lin l1, l2;
    memset(&l1, 0, sizeof(l1));
    memset(&l2, 0, sizeof(l2));
    if (!e->prew || !e->midw ||
        bind_lin(e, &l1, "multi_modal_projector.linear_1.weight") ||
        bind_lin(e, &l2, "multi_modal_projector.linear_2.weight")) {
        rt_ear_close(e);
        return NULL;
    }
    e->lin1w = l1.w;
    e->lin2w = l2.w;

    e->melmat = build_melmat();
    e->dftc = (float *)xcalloc((size_t)N_FREQ * N_FFT, sizeof(float));
    e->dfts = (float *)xcalloc((size_t)N_FREQ * N_FFT, sizeof(float));
    for (int k = 0; k < N_FREQ; k++)
        for (int n = 0; n < N_FFT; n++) {
            double a = -2.0 * M_PI * (double)k * (double)n / (double)N_FFT;
            e->dftc[(int64_t)k * N_FFT + n] = (float)cos(a);
            e->dfts[(int64_t)k * N_FFT + n] = (float)sin(a);
        }

    /* Scratch sized for one full window. */
    e->block = CHUNK_POST * (e->nwin_infer / (e->nwin * 2));
    if (e->block < 1) e->block = CHUNK_POST;
    const int chunks = WIN_FRAMES / CHUNK;
    e->rows = chunks * CHUNK_POST + CHUNK_POST;
    const int d = e->d;
    e->wmel = (float *)xcalloc((size_t)MEL_BINS * chunks * CHUNK, sizeof(float));
    e->col = (float *)xcalloc((size_t)e->c2o * 9 * 800, sizeof(float));
    e->cv1 = (float *)xcalloc((size_t)e->c1o * 64 * 50, sizeof(float));
    e->cv2 = (float *)xcalloc((size_t)e->c2o * 32 * 25, sizeof(float));
    e->cv3 = (float *)xcalloc((size_t)e->c3o * 16 * 13, sizeof(float));
    e->stage = (float *)xcalloc((size_t)e->rows * d, sizeof(float));
    e->packed = e->stage; /* tower_forward writes straight into the stage */
    e->xn = (float *)xcalloc((size_t)e->rows * d, sizeof(float));
    e->q = (float *)xcalloc((size_t)e->rows * d, sizeof(float));
    e->k = (float *)xcalloc((size_t)e->rows * d, sizeof(float));
    e->v = (float *)xcalloc((size_t)e->rows * d, sizeof(float));
    e->att = (float *)xcalloc((size_t)e->rows * d, sizeof(float));
    e->ao = (float *)xcalloc((size_t)e->rows * d, sizeof(float));
    e->ff = (float *)xcalloc((size_t)e->rows * e->ffn, sizeof(float));
    e->gq = (float *)xcalloc((size_t)e->rows * e->hd, sizeof(float));
    e->gk = (float *)xcalloc((size_t)e->rows * e->hd, sizeof(float));
    e->scores = (float *)xcalloc((size_t)e->rows * e->rows, sizeof(float));

    CWICK_INFO(EA_TAG,
               "ear ready: %d layers, d=%d, ffn=%d, heads=%d/%d, %d-frame chunks, "
               "%d-frame blocks, stack=%d, projector %d->%d->%d",
               e->nl, e->d, e->ffn, e->heads, e->hd, CHUNK, e->block, e->stack,
               e->d * e->stack, e->hid, e->dout);
    return e;
}

void rt_ear_close(rt_ear *e) {
    if (!e) return;
    cwick_safetensors_close(&e->saf);
    free(e->c1w); free(e->c1b); free(e->c2w); free(e->c2b); free(e->c3w); free(e->c3b);
    free(e->conv_out.w);
    free(e->lnpw); free(e->lnpb); free(e->pos);
    free(e->prew); free(e->lin1w); free(e->midw); free(e->lin2w);
    free(e->melmat); free(e->dftc); free(e->dfts);
    free(e->wmel); free(e->col); free(e->cv1); free(e->cv2); free(e->cv3);
    free(e->stage); free(e->xn); free(e->q); free(e->k); free(e->v);
    free(e->att); free(e->ao); free(e->ff); free(e->gq); free(e->gk); free(e->scores);
    for (int l = 0; l < e->nl; l++) {
        ea_layer *L = &e->lay[l];
        free(L->q.w); free(L->q.b); free(L->k.w); free(L->k.b);
        free(L->v.w); free(L->v.b); free(L->o.w); free(L->o.b);
        free(L->fc1.w); free(L->fc1.b); free(L->fc2.w); free(L->fc2.b);
        free(L->ln1w); free(L->ln1b); free(L->ln2w); free(L->ln2b);
    }
    free(e->lay);
    free(e);
}
