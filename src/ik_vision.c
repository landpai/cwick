/*
 * ik_vision.c — the Inkling-2-Mini vision tower.
 *
 * The tower is a plain pre-LN ViT with three unusual traits, and all three
 * matter for the output to be right:
 *
 *   - one class token leads every image, and it carries its own learned
 *     position (the patch position table is not consulted for it);
 *   - the learned 32x32 position table is bilinearly resampled to the image's
 *     own patch grid at align_corners=false, i.e. half-pixel centres;
 *   - positions are 1-based and patches are emitted in merge-block order, so
 *     the merger can simply reshape groups of four into one language token.
 *
 * The single-image path is all this backend needs: the tokenizer expands one
 * placeholder per merged patch and the language model splices the embeddings
 * in, so images never share an attention segment.
 */
#include "ik.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <Accelerate/Accelerate.h>
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>

#include "h3v_gpu.h"
#include "log.h"
#include "util.h"

static void ikv_layernorm(float *out, const float *in, const float *w,
                          const float *b, int64_t n, float eps) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) sum += in[i];
    float mean = (float)(sum / (double)n);
    double var = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double d = (double)in[i] - mean;
        var += d * d;
    }
    var /= (double)n;
    float inv = 1.0f / sqrtf((float)var + eps);
    for (int64_t i = 0; i < n; i++)
        out[i] = ((in[i] - mean) * inv) * w[i] + b[i];
    ik_round_bf16(out, n);
}

/* ACT2FN["gelu"] is the exact erf form, not the tanh approximation. */
static void ikv_gelu(float *x, int64_t n) {
    const float k = 0.7071067811865476f;
    for (int64_t i = 0; i < n; i++)
        x[i] = 0.5f * x[i] * (1.0f + erff(x[i] * k));
    ik_round_bf16(x, n);
}

int ik_vision_open(ik_model *m, cwick_error *err) {
    if (!m->vis || !m->vis->loaded) {
        cwick_error_set(err, CWICK_ERR_STATE,
                        "Inkling-2-Mini has no vision tower loaded");
        return -1;
    }
    return 0;
}

void ik_vision_close(ik_model *m) {
    if (!m->vis) return;
    free(m->vis->layers);
    free(m->vis);
    m->vis = NULL;
}

/* ------------------------------------------------------------------ */
/* preprocessing                                                       */
/* ------------------------------------------------------------------ */

float ik_vision_pixel(float raw, int channel) {
    (void)channel;
    float value = (raw - 0.5f) / 0.5f;
    return cwick_bf16_to_f32(cwick_f32_to_bf16(value));
}

/* Bilinear taps into the 32x32 table for one axis, align_corners=false. */
static void ikv_axis(float at, int64_t size, int64_t side, int64_t taps[2],
                     float wts[2]) {
    float src = ((float)at + 0.5f) * (float)side / (float)size - 0.5f;
    float fl = floorf(src);
    int64_t t0 = (int64_t)fl;
    for (int k = 0; k < 2; k++) {
        int64_t t = t0 + k;
        if (t < 0) t = 0;
        if (t > side - 1) t = side - 1;
        taps[k] = t;
        float d = fabsf(src - (float)(t0 + k));
        wts[k] = 1.0f - d;
        if (wts[k] < 0.0f) wts[k] = 0.0f;
    }
    float sum = wts[0] + wts[1];
    if (sum > 0.0f) { wts[0] /= sum; wts[1] /= sum; }
}

/* ------------------------------------------------------------------ */
/* forward                                                             */
/* ------------------------------------------------------------------ */

typedef struct {
    int64_t n;        /* sequence length: 1 class token + patches */
    int64_t rows, cols;  /* language-token grid */
    int64_t ph, pw;   /* patch grid */
    float *x;         /* [n][v_hidden] */
    float *q, *k, *v, *att, *proj;
    float *scores;    /* [n][n] */
    float *mlp, *norm;
    float *pos;       /* [n][v_hidden] additive position */
} ikv;

static void ikv_free(ikv *w) {
    free(w->x); free(w->q); free(w->k); free(w->v); free(w->att);
    free(w->proj); free(w->scores); free(w->mlp); free(w->norm); free(w->pos);
    memset(w, 0, sizeof(*w));
}

static int ikv_alloc(ikv *w, int64_t n, int64_t H, int64_t inter) {
    memset(w, 0, sizeof(*w));
    w->n = n;
    w->x = (float *)calloc((size_t)(n * H), sizeof(float));
    w->q = (float *)calloc((size_t)(n * H), sizeof(float));
    w->k = (float *)calloc((size_t)(n * H), sizeof(float));
    w->v = (float *)calloc((size_t)(n * H), sizeof(float));
    w->att = (float *)calloc((size_t)(n * H), sizeof(float));
    w->proj = (float *)calloc((size_t)(n * H), sizeof(float));
    w->mlp = (float *)calloc((size_t)(n * inter), sizeof(float));
    w->norm = (float *)calloc((size_t)(n * H), sizeof(float));
    w->pos = (float *)calloc((size_t)(n * H), sizeof(float));
    w->scores = (float *)calloc((size_t)(n * n), sizeof(float));
    if (!w->x || !w->q || !w->k || !w->v || !w->att || !w->proj || !w->mlp ||
        !w->norm || !w->pos || !w->scores) {
        ikv_free(w);
        return -1;
    }
    return 0;
}

static void ikv_rope(float *q, float *k, const int64_t *pos_h,
                     const int64_t *pos_w, int64_t n, int64_t heads,
                     int64_t dim, float base) {
    const int64_t half = dim / 2;         /* spatial dim per axis */
    const int64_t nfreq = half / 2;       /* rotated pairs per axis */
    float *buf = (float *)malloc((size_t)dim * sizeof(float));
    if (!buf) return;
    for (int64_t t = 0; t < n; t++) {
        for (int64_t h = 0; h < heads; h++) {
            float *xq = q + (t * heads + h) * dim;
            float *xk = k + (t * heads + h) * dim;
            for (int pass = 0; pass < 2; pass++) {
                float *x = pass ? xk : xq;
                for (int64_t i = 0; i < nfreq; i++) {
                    /* the frequency list is [width freqs..., height freqs...] */
                    float ang = powf(base, -(float)(2 * i) / (float)half);
                    float aw = ang * (float)pos_w[t];
                    float ah = ang * (float)pos_h[t];
                    float cw = cosf(aw), sw = sinf(aw);
                    float ch = cosf(ah), sh = sinf(ah);
                    float x0 = x[2 * i], x1 = x[2 * i + 1];
                    buf[2 * i] = x0 * cw - x1 * sw;
                    buf[2 * i + 1] = x1 * cw + x0 * sw;
                    int64_t o = 2 * (nfreq + i);
                    float y0 = x[o], y1 = x[o + 1];
                    buf[o] = y0 * ch - y1 * sh;
                    buf[o + 1] = y1 * ch + y0 * sh;
                }
                memcpy(x, buf, (size_t)dim * sizeof(float));
            }
        }
    }
    free(buf);
}

/* Row-wise softmax with a plain max-subtract; `vvexpf` keeps the quadratic
 * exponent count off the critical path. */
static void ikv_softmax_rows(float *s, int64_t rows, int64_t cols) {
    for (int64_t r = 0; r < rows; r++) {
        float *row = s + r * cols;
        float mx = row[0];
        vDSP_maxv(row, 1, &mx, (vDSP_Length)cols);
        float neg = -mx;
        vDSP_vsadd(row, 1, &neg, row, 1, (vDSP_Length)cols);
        int32_t cnt = (int32_t)cols;
        vvexpf(row, row, &cnt);
        float sum = 0.0f;
        vDSP_sve(row, 1, &sum, (vDSP_Length)cols);
        if (!(sum > 0.0f)) sum = 1.0f;
        float inv = 1.0f / sum;
        vDSP_vsmul(row, 1, &inv, row, 1, (vDSP_Length)cols);
    }
}

typedef struct {
    const void *w;
    const float *x;
    float *out;
    int64_t k, cols;
} ikv_linear_arg;

static void ikv_linear(ik_model *m, int64_t n, const ikv_linear_arg *a) {
    const char *policy = getenv("CWICK_IK_GPU");
    if (n > 1 && cwick_metal_backend_active() &&
        (!policy || strcmp(policy, "0") != 0) && h3v_gpu_on() && a->k % 16 == 0 &&
        h3v_gpu_gemm_bf16(a->x, a->w, a->out, (int)n, (int)a->k,
                          (int)a->cols, (int)a->k, (int)a->cols)) {
        m->gpu_calls++;
        return;
    }
    if (n == 1) cwick_gemv_bf16(a->out, a->k, a->cols, a->x, a->w);
    else cwick_gemm_bf16(a->out, n, a->k, a->cols, a->x, a->w);
}

int ik_vision_embed_rgb(ik_model *m, const float *rgb, int64_t height,
                        int64_t width, float **embeds, int64_t *n_tokens,
                        int64_t *rows_out, int64_t *cols_out, cwick_error *err) {
    if (!m->vis || !m->vis->loaded) {
        cwick_error_set(err, CWICK_ERR_STATE,
                        "Inkling-2-Mini has no vision tower loaded");
        return -1;
    }
    if (!rgb || height <= 0 || width <= 0) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "empty image");
        return -1;
    }
    if (height > 8192 || width > 8192 ||
        height > 16777216 / width) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "image exceeds 8192 px per side or 16777216 pixels");
        return -1;
    }
    const ik_cfg *c = &m->c;
    const ik_vision *vis = m->vis;
    const int64_t VH = c->v_hidden, VL = c->v_layers, HV = c->v_heads;
    const int64_t VD = VH / HV, VI = c->v_inter, VP = c->v_proj;
    const int64_t grid = c->v_grid;
    const int64_t merge = c->v_merge, sub = merge * merge;
    const int64_t block = c->v_patch * merge;
    const int64_t rows = (height + block - 1) / block;
    const int64_t cols = (width + block - 1) / block;
    const int64_t ph = rows * merge, pw = cols * merge;
    const int64_t nblock = rows * cols;
    const int64_t npat = nblock * sub;
    const int64_t n = npat + 1;
    const int64_t k_in = 3 * c->v_patch * c->v_patch;
    if (npat > 4096) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "image exceeds 4096 vision patches");
        return -1;
    }

    ikv w;
    if (ikv_alloc(&w, n, VH, VI) != 0) {
        cwick_error_set(err, CWICK_ERR_MEMORY,
                        "vision workspace for %lld patches failed",
                        (long long)npat);
        return -1;
    }
    w.rows = rows; w.cols = cols; w.ph = ph; w.pw = pw;

    /* ---- patchify: one 1200-vector per 20x20 patch, merge-block order ---- */
    float *pat = (float *)calloc((size_t)(npat * k_in), sizeof(float));
    if (!pat) {
        ikv_free(&w);
        cwick_error_set(err, CWICK_ERR_MEMORY, "patch buffer failed");
        return -1;
    }
    const int64_t P = c->v_patch;
    for (int64_t br = 0; br < rows; br++) {
        for (int64_t bc = 0; bc < cols; bc++) {
            for (int64_t ir = 0; ir < merge; ir++) {
                for (int64_t ic = 0; ic < merge; ic++) {
                    int64_t p = ((br * cols + bc) * sub) + ir * merge + ic;
                    float *vec = pat + p * k_in;
                    for (int64_t y = 0; y < P; y++) {
                        int64_t sy = br * block + ir * P + y;
                        for (int64_t x = 0; x < P; x++) {
                            int64_t sx = bc * block + ic * P + x;
                            float *dst = vec + (y * P + x) * 3;
                            if (sy < height && sx < width) {
                                const float *src = rgb + (sy * width + sx) * 3;
                                for (int ch = 0; ch < 3; ch++)
                                    dst[ch] = ik_vision_pixel(src[ch], ch);
                            } else {
                                for (int ch = 0; ch < 3; ch++)
                                    dst[ch] = ik_vision_pixel(-1.0f / 255.0f, ch);
                            }
                        }
                    }
                }
            }
        }
    }
    float *pemb = (float *)calloc((size_t)(npat * VH), sizeof(float));
    if (!pemb) {
        free(pat);
        ikv_free(&w);
        cwick_error_set(err, CWICK_ERR_MEMORY, "patch embedding failed");
        return -1;
    }
    ikv_linear_arg patch = {vis->patch_embed, pat, pemb, k_in, VH};
    ikv_linear(m, npat, &patch);
    ik_round_bf16(pemb, npat * VH);
    free(pat);

    /* ---- additive positions: class token first, then the resampled table ---- */
    for (int64_t i = 0; i < VH; i++)
        w.pos[i] = vis->class_embed[i] + vis->cls_pos[i];
    int64_t *ph_ids = (int64_t *)calloc((size_t)n, sizeof(int64_t));
    int64_t *pw_ids = (int64_t *)calloc((size_t)n, sizeof(int64_t));
    if (!ph_ids || !pw_ids) {
        free(pemb); free(ph_ids); free(pw_ids);
        ikv_free(&w);
        cwick_error_set(err, CWICK_ERR_MEMORY, "position ids failed");
        return -1;
    }
    for (int64_t p = 0; p < npat; p++) {
        int64_t block_idx = p / sub;
        int64_t br = block_idx / cols, bc = block_idx % cols;
        int64_t ir = (p / merge) % merge, ic = p % merge;
        int64_t row = br * merge + ir, col = bc * merge + ic;
        ph_ids[p + 1] = row + 1;
        pw_ids[p + 1] = col + 1;
        int64_t th[2], tw[2];
        float wh[2], ww[2];
        ikv_axis(row, ph, grid, th, wh);
        ikv_axis(col, pw, grid, tw, ww);
        float *dst = w.pos + (p + 1) * VH;
        for (int a = 0; a < 2; a++)
            for (int b = 0; b < 2; b++) {
                const float *src = vis->pos_embed + (th[a] * grid + tw[b]) * VH;
                float k = wh[a] * ww[b];
                for (int64_t i = 0; i < VH; i++) dst[i] += k * src[i];
            }
    }

    /* ---- embeddings, then the pre-norm ---- */
    for (int64_t i = 0; i < VH; i++) w.x[i] = w.pos[i];
    for (int64_t p = 0; p < npat; p++)
        for (int64_t i = 0; i < VH; i++)
            w.x[(p + 1) * VH + i] = pemb[p * VH + i] + w.pos[(p + 1) * VH + i];
    free(pemb);
    ik_round_bf16(w.x, n * VH);
    for (int64_t t = 0; t < n; t++)
        ikv_layernorm(w.x + t * VH, w.x + t * VH,
                      vis->pre_ln_w, vis->pre_ln_b, VH, c->v_ln_eps);

    /* ---- encoder ---- */
    const float scale = 1.0f / sqrtf((float)VD);
    for (int64_t l = 0; l < VL; l++) {
        const ik_vlayer *vl = &vis->layers[l];
        for (int64_t t = 0; t < n; t++)
            ikv_layernorm(w.norm + t * VH, w.x + t * VH, vl->ln1_w, vl->ln1_b,
                          VH, c->v_ln_eps);
        ikv_linear_arg qkv[3] = {
            {vl->q, w.norm, w.q, VH, VH},
            {vl->k, w.norm, w.k, VH, VH},
            {vl->v, w.norm, w.v, VH, VH},
        };
        for (int i = 0; i < 3; i++) {
            ikv_linear(m, n, &qkv[i]);
            ik_round_bf16(i == 0 ? w.q : (i == 1 ? w.k : w.v), n * VH);
        }
        for (int64_t t = 0; t < n; t++) {
            for (int64_t i = 0; i < VH; i++) {
                w.q[t * VH + i] += vl->q_b[i];
                w.k[t * VH + i] += vl->k_b[i];
                w.v[t * VH + i] += vl->v_b[i];
            }
        }
        ik_round_bf16(w.q, n * VH);
        ik_round_bf16(w.k, n * VH);
        ik_round_bf16(w.v, n * VH);
        ikv_rope(w.q, w.k, ph_ids, pw_ids, n, HV, VD, 10000.0f);
        ik_round_bf16(w.q, n * VH);
        ik_round_bf16(w.k, n * VH);

        /* full (non-causal) attention inside the single image segment */
        for (int64_t h = 0; h < HV; h++) {
            const float *qh = w.q + h * VD;
            const float *kh = w.k + h * VD;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        (int)n, (int)n, (int)VD, scale, qh, (int)VH, kh,
                        (int)VH, 0.0f, w.scores, (int)n);
            ik_round_bf16(w.scores, n * n);
            ikv_softmax_rows(w.scores, n, n);
            ik_round_bf16(w.scores, n * n);
            const float *vh = w.v + h * VD;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        (int)n, (int)VD, (int)n, 1.0f, w.scores, (int)n, vh,
                        (int)VH, 0.0f, w.att + h * VD, (int)VH);
        }
        ik_round_bf16(w.att, n * VH);
        ikv_linear_arg out = {vl->o, w.att, w.proj, VH, VH};
        ikv_linear(m, n, &out);
        ik_round_bf16(w.proj, n * VH);
        for (int64_t t = 0; t < n; t++)
            for (int64_t i = 0; i < VH; i++)
                w.x[t * VH + i] += vl->ls1[i] * (w.proj[t * VH + i] + vl->o_b[i]);
        ik_round_bf16(w.x, n * VH);

        for (int64_t t = 0; t < n; t++)
            ikv_layernorm(w.norm + t * VH, w.x + t * VH, vl->ln2_w, vl->ln2_b,
                          VH, c->v_ln_eps);
        ikv_linear_arg fc1 = {vl->fc1, w.norm, w.mlp, VH, VI};
        ikv_linear(m, n, &fc1);
        ik_round_bf16(w.mlp, n * VI);
        for (int64_t t = 0; t < n; t++)
            for (int64_t i = 0; i < VI; i++) w.mlp[t * VI + i] += vl->fc1_b[i];
        ik_round_bf16(w.mlp, n * VI);
        ikv_gelu(w.mlp, n * VI);
        ikv_linear_arg fc2 = {vl->fc2, w.mlp, w.proj, VI, VH};
        ikv_linear(m, n, &fc2);
        ik_round_bf16(w.proj, n * VH);
        for (int64_t t = 0; t < n; t++)
            for (int64_t i = 0; i < VH; i++)
                w.x[t * VH + i] += vl->ls2[i] * (w.proj[t * VH + i] + vl->fc2_b[i]);
        ik_round_bf16(w.x, n * VH);
    }

    /* ---- merger: four patches per language token, then 4096 -> 2304 ---- */
    float *merged = (float *)calloc((size_t)(nblock * sub * VH), sizeof(float));
    float *hid = (float *)calloc((size_t)(nblock * VP), sizeof(float));
    float *out = (float *)calloc((size_t)(nblock * c->hidden), sizeof(float));
    if (!merged || !hid || !out) {
        free(merged); free(hid); free(out);
        free(ph_ids); free(pw_ids);
        ikv_free(&w);
        cwick_error_set(err, CWICK_ERR_MEMORY, "merger buffer failed");
        return -1;
    }
    for (int64_t b = 0; b < nblock; b++)
        for (int64_t s = 0; s < sub; s++) {
            int64_t p = b * sub + s;
            memcpy(merged + (b * sub + s) * VH, w.x + (p + 1) * VH,
                   (size_t)VH * sizeof(float));
        }
    ikv_linear_arg proj_in = {vis->proj_in, merged, hid, sub * VH, VP};
    ikv_linear_arg proj_out = {vis->proj_out, hid, out, VP, c->hidden};
    ikv_linear(m, nblock, &proj_in);
    ik_round_bf16(hid, nblock * VP);
    ikv_gelu(hid, nblock * VP);
    ikv_linear(m, nblock, &proj_out);
    ik_round_bf16(out, nblock * c->hidden);

    free(merged); free(hid);
    free(ph_ids); free(pw_ids);
    ikv_free(&w);
    *embeds = out;
    *n_tokens = nblock;
    if (rows_out) *rows_out = rows;
    if (cols_out) *cols_out = cols;
    return 0;
}

static int ikv_decode_rgb(const uint8_t *data, size_t len, float **rgb,
                          int64_t *width_out, int64_t *height_out,
                          cwick_error *err) {
    if (!data || len < 16 || len > 32 * 1024 * 1024) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "image must be between 16 bytes and 32 MiB");
        return -1;
    }
    const UInt8 *bytes = data;
    uint8_t *jpeg_tail = NULL;
    if (data[0] == 0xff && data[1] == 0xd8 &&
        !(data[len - 2] == 0xff && data[len - 1] == 0xd9)) {
        jpeg_tail = (uint8_t *)cwick_xmalloc(len + 2);
        memcpy(jpeg_tail, data, len);
        jpeg_tail[len] = 0xff;
        jpeg_tail[len + 1] = 0xd9;
        bytes = jpeg_tail;
    }
    CFDataRef src_data = CFDataCreate(kCFAllocatorDefault, bytes, (CFIndex)len +
                                      (jpeg_tail ? 2 : 0));
    free(jpeg_tail);
    CGImageSourceRef src = src_data ? CGImageSourceCreateWithData(src_data, NULL) : NULL;
    if (src_data) CFRelease(src_data);
    CGImageRef img = src ? CGImageSourceCreateImageAtIndex(src, 0, NULL) : NULL;
    if (src) CFRelease(src);
    if (!img) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "image decode failed");
        return -1;
    }
    int64_t width = CGImageGetWidth(img), height = CGImageGetHeight(img);
    if (width <= 0 || height <= 0 || width > 8192 || height > 8192 ||
        height > 16777216 / width) {
        CGImageRelease(img);
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "image exceeds 8192 px per side or 16777216 pixels");
        return -1;
    }
    size_t n = (size_t)(width * height);
    uint8_t *pixels = (uint8_t *)cwick_xmalloc(n * 4);
    CGImageAlphaInfo alpha = CGImageGetAlphaInfo(img);
    bool transparent = alpha != kCGImageAlphaNone && alpha != kCGImageAlphaNoneSkipLast;
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(pixels, (size_t)width, (size_t)height, 8,
                                             (size_t)width * 4, cs,
                                             transparent ? kCGImageAlphaPremultipliedLast
                                                         : kCGImageAlphaNoneSkipLast);
    CGColorSpaceRelease(cs);
    if (!ctx) {
        CGImageRelease(img); free(pixels);
        cwick_error_set(err, CWICK_ERR_STATE, "image decode context failed");
        return -1;
    }
    CGContextDrawImage(ctx, CGRectMake(0, 0, (CGFloat)width, (CGFloat)height), img);
    CGContextRelease(ctx);
    CGImageRelease(img);
    float *out = (float *)cwick_xmalloc(n * 3 * sizeof(float));
    for (int64_t y = 0; y < height; y++) {
        for (int64_t x = 0; x < width; x++) {
            size_t i = (size_t)(y * width + x);
            unsigned bg = ((x / 8 + y / 8) & 1) ? 153 : 204;
            unsigned a = transparent ? pixels[i * 4 + 3] : 255;
            for (int ch = 0; ch < 3; ch++) {
                unsigned p = pixels[i * 4 + ch];
                unsigned straight = a ? (p * 255 + a / 2) / a : 0;
                if (straight > 255) straight = 255;
                unsigned value = transparent
                    ? (straight * a + bg * (255 - a) + 127) / 255 : p;
                out[i * 3 + ch] = value / 255.0f;
            }
        }
    }
    free(pixels);
    *rgb = out;
    *width_out = width;
    *height_out = height;
    return 0;
}

int ik_vision_embed(ik_model *m, const uint8_t *data, size_t len,
                    float **embeds, int64_t *n_tokens, int64_t *rows,
                    int64_t *cols, cwick_error *err) {
    if (!m || !m->vis || !m->vis->loaded) {
        cwick_error_set(err, CWICK_ERR_STATE,
                        "Inkling-2-Mini has no vision tower loaded");
        return -1;
    }
    float *rgb = NULL;
    int64_t width = 0, height = 0;
    if (ikv_decode_rgb(data, len, &rgb, &width, &height, err) != 0) return -1;
    int rc = ik_vision_embed_rgb(m, rgb, height, width, embeds, n_tokens, rows,
                                 cols, err);
    free(rgb);
    return rc;
}
