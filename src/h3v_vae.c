/* h3v_vae.c — the visual VAE's ViT decoder.
 *
 * The encoder side of H3-VisualVAE is a causal 3D CNN and is only used on the
 * training path; generation needs the decoder alone. That decoder is not a CNN
 * at all: every latent cell becomes one token, 36 transformer blocks run full
 * self-attention over the whole clip, and each token's 3072 outputs expand to a
 * 3x4x16x16 pixel block. Compression is 16x spatially and 4x temporally, which
 * is where the latent grid's shape comes from.
 *
 * Token layout and the block's exact shape are taken from the shipped
 * config.json and weight shapes:
 *
 *   in   24 channels per latent cell (no patchify: patch size 1 on the way in)
 *   embed 24 -> 2048, then 4 learned register tokens and one zero cls token
 *         are appended, all five with a zero position id
 *   rope  3-axis, 8 frequencies per axis over 48 of the 64 head dims, ids
 *         normalised to [-1, 1] per axis
 *   block rmsnorm -> qkv (fused, per-head [q|k|v]) -> rmsnorm on q and k ->
 *         rope -> full attention -> out_proj -> gated residual, then
 *         rmsnorm -> SwiGLU(8192 gate, 8192 value) -> gated residual
 *   out   layernorm -> 2048 -> 3072 = 3 channels x 4 frames x 16 x 16
 *
 * The residual gates (`scale1`, `scale2`) are learned per-channel vectors, not
 * scalars, and start at zero — a block that skips them is a block that outputs
 * nothing.
 *
 * Full attention over a 10-second 768p clip is ~240k tokens; 240k^2 is not a
 * shape this machine can hold. The checkpoint documents 256-pixel tiles with
 * 64-pixel overlap for exactly this reason, so the latent grid is walked in
 * tiles here too and neighbouring tiles are cross-faded over the overlap.
 * Tiling costs a little redundant work (the overlap is decoded twice) and buys
 * a bounded attention matrix.
 *
 * Comments describe mechanism and measured numbers only.
 */
#include "h3v_gpu.h"
#include "h3v_internal.h"

#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "log.h"
#include "util.h"

#define VL_LAYERS      36
#define VL_DIM         2048
#define VL_HEADS       32
#define VL_HEAD_DIM    64
#define VL_QKV         (3 * VL_DIM)     /* 6144, per head [q|k|v] */
#define VL_FFN         8192             /* gated: w1 is 16384 wide */
#define VL_ROPE_FREQ   8
#define VL_ROPE_AXES   3
#define VL_ROPE_DIM    (VL_ROPE_FREQ * VL_ROPE_AXES)  /* 24 angles, tiled to 48 */
#define VL_ROPE_APPLY  48
#define VL_THETA       100.0
#define VL_REGISTERS   4
#define VL_SUFFIX      (1 + VL_REGISTERS)
#define VL_PATCH_DIM   (3 * H3V_VAE_FRAMES * H3V_VAE_PIXEL * H3V_VAE_PIXEL) /* 3072 */
#define VL_OPS_TILE    128              /* query rows per attention pass */

/* Per-channel latent statistics, copied verbatim from the checkpoint's
 * vae/config.json (`latents_mean`, `latents_std`). The transformer is trained
 * in this normalised space, so every latent that crosses the DiT/VAE boundary
 * must be converted:
 *
 *   decoder input : z = x * std + mean      (normalised -> raw)
 *   encoder output: x = (z - mean) / std    (raw -> normalised)
 *
 * Dropping either step leaves the decoder reading a latent that is shifted by
 * mean/std and scaled by 1/std per channel (std spans 0.45-3.28, mean spans
 * -1.37-1.07). The 36-block ViT then decodes a picture that has nothing to do
 * with the sampled clip: low-frequency mush that muxes into a valid mp4. */
static const float VL_LAT_MEAN[H3V_VAE_LATENT_CH] = {
    0.858090341091156f, -0.9606591463088989f, 1.0661640167236328f,
    -0.5090325474739075f, -0.2727581858634949f, -1.3675414323806763f,
    -0.2553254961967468f, -0.26907554268836975f, -0.5376840829849243f,
    -0.0464097298681736f, 0.6657370328903198f, 0.19690127670764923f,
    -0.5460608005523682f, -0.4035342037677765f, -0.23683024942874908f,
    0.25928452610969543f, -0.30133944749832153f, 0.211341992020607f,
    -1.1206848621368408f, 0.3581933379173279f, -0.04225143790245056f,
    0.2604829967021942f, 0.22864092886447906f, 0.7056031823158264f};
static const float VL_LAT_STD[H3V_VAE_LATENT_CH] = {
    1.2223774194717407f, 1.2767263650894165f, 1.6831774711608887f,
    1.7549455165863037f, 1.5636216402053833f, 2.194143533706665f,
    0.9653137922286987f, 1.0569885969161987f, 0.841948926448822f,
    0.7729952931404114f, 1.8955937623977661f, 0.946841835975647f,
    0.7996809482574463f, 0.44988900423049927f, 0.7197399735450745f,
    0.6936293244361877f, 2.961095094680786f, 2.7694199085235596f,
    3.0496184825897217f, 2.1088054180145264f, 3.276226282119751f,
    3.1627357006073f, 2.2816812992095947f, 2.6127843856811523f};

/* [C][N] channel-major, in place: normalised -> raw. */
static void vl_lat_denorm(float *lat, int64_t plane) {
    for (int64_t c = 0; c < H3V_VAE_LATENT_CH; c++) {
        float *p = lat + (size_t)c * (size_t)plane;
        const float m = VL_LAT_MEAN[c], s = VL_LAT_STD[c];
        for (int64_t i = 0; i < plane; i++) p[i] = p[i] * s + m;
    }
}

/* [C][N] channel-major, in place: raw -> normalised. */
static void vl_lat_norm(float *lat, int64_t plane) {
    for (int64_t c = 0; c < H3V_VAE_LATENT_CH; c++) {
        float *p = lat + (size_t)c * (size_t)plane;
        const float m = VL_LAT_MEAN[c], s = VL_LAT_STD[c];
        for (int64_t i = 0; i < plane; i++) p[i] = (p[i] - m) / s;
    }
}

static float *vl_f32(int64_t n) {
    return (float *)cwick_xmalloc((size_t)n * sizeof(float));
}

/* A tensor's float payload plus its [rows, cols] view. Every tensor in this
 * checkpoint is F32, so there is no dequantisation path here. */
typedef struct {
    const float *p;
    int64_t rows, cols;
} vl_mat;

static bool vl_get(const cwick_safetensors *s, const char *name, vl_mat *m,
                   cwick_error *err) {
    cwick_st_tensor t;
    if (cwick_safetensors_get(s, name, &t, err) != CWICK_OK) return false;
    if (t.dtype != CWICK_ST_F32) {
        cwick_error_set(err, CWICK_ERR_PARSE,
                        "video VAE tensor %s is not float32", name);
        return false;
    }
    m->p = (const float *)t.data;
    m->rows = t.rows;
    m->cols = t.cols;
    return true;
}

/* out[n][rows] = x[n][cols] @ W^T + bias */
static void vl_linear(float *out, const float *x, const vl_mat *w,
                      const float *bias, int64_t n) {
    /* Every VAE weight is F32, so the F32 tensor-core kernel can take the
     * mapping verbatim. Thin products stay on the CPU. */
    if (n >= 32 && (int64_t)n * (int64_t)w->cols * w->rows >= (int64_t)8 * 1024 * 1024 &&
        h3v_gpu_on() &&
        h3v_gpu_gemm_f32(x, w->p, out, (int)n, (int)w->cols, (int)w->rows,
                         (int)w->cols, (int)w->rows)) {
        /* bias below */
    } else {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)n, (int)w->rows,
                    (int)w->cols, 1.0f, x, (int)w->cols, w->p, (int)w->cols, 0.0f,
                    out, (int)w->rows);
    }
    if (!bias) return;
    for (int64_t i = 0; i < n; i++) {
        float *o = out + i * w->rows;
        for (int64_t j = 0; j < w->rows; j++) o[j] += bias[j];
    }
}

static void vl_rmsnorm(float *out, const float *x, const float *w, int64_t dim,
                       float eps) {
    double ss = 0.0;
    for (int64_t i = 0; i < dim; i++) ss += (double)x[i] * x[i];
    float inv = 1.0f / sqrtf((float)(ss / (double)dim) + eps);
    if (w) {
        for (int64_t i = 0; i < dim; i++) out[i] = x[i] * inv * w[i];
    } else {
        for (int64_t i = 0; i < dim; i++) out[i] = x[i] * inv;
    }
}

static void vl_layernorm(float *out, const float *x, const float *w,
                         const float *b, int64_t dim, float eps) {
    double mean = 0.0;
    for (int64_t i = 0; i < dim; i++) mean += (double)x[i];
    mean /= (double)dim;
    double var = 0.0;
    for (int64_t i = 0; i < dim; i++) {
        double d = (double)x[i] - mean;
        var += d * d;
    }
    var /= (double)dim;
    float inv = 1.0f / sqrtf((float)var + eps);
    float m = (float)mean;
    for (int64_t i = 0; i < dim; i++) out[i] = (x[i] - m) * inv * w[i] + b[i];
}

/* Fold post_quant_conv (24x24 pointwise) and x_embedder (2048x24) into one
 * 2048x24 matrix. Both are linear and nothing sits between them, so this is
 * exact and saves a pass over the latent for every tile. */
static void vl_fold_embed(const cwick_safetensors *s, float **w_out,
                          float **b_out, cwick_error *err) {
    vl_mat pq_w, pq_b, xe_w, xe_b;
    memset(&pq_b, 0, sizeof(pq_b));
    memset(&xe_b, 0, sizeof(xe_b));
    if (!vl_get(s, "post_quant_conv.weight", &pq_w, err)) return;
    vl_get(s, "post_quant_conv.bias", &pq_b, err);
    if (!vl_get(s, "decoder.x_embedder.weight", &xe_w, err)) return;
    vl_get(s, "decoder.x_embedder.bias", &xe_b, err);

    const int64_t C = H3V_VAE_LATENT_CH, D = VL_DIM;
    float *w = vl_f32(D * C);
    /* w[o][in] = sum_c xemb[o][c] * pqc[c][in] */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)D, (int)C,
                (int)C, 1.0f, xe_w.p, (int)C, pq_w.p, (int)C, 0.0f, w, (int)C);
    float *b = vl_f32(D);
    for (int64_t o = 0; o < D; o++) {
        double acc = xe_b.p ? xe_b.p[o] : 0.0;
        for (int64_t c = 0; c < C; c++)
            acc += (double)xe_w.p[o * C + c] * (pq_b.p ? pq_b.p[c] : 0.0);
        b[o] = (float)acc;
    }
    *w_out = w;
    *b_out = b;
}

/* 3-axis rotary tables for one tile. Ids are the checkpoint's
 * "length_normalized" ones: 2*(i+0.5)/n - 1 per axis, so a tile of any size
 * spans the same [-1, 1] range and the decoder sees the same positions a
 * whole-clip decode would give it. */
static void vl_rope(int64_t lt, int64_t th, int64_t tw, int64_t npat, float *cos,
                    float *sin) {
    const int64_t F = VL_ROPE_FREQ;
    float inv[VL_ROPE_FREQ];
    for (int64_t j = 0; j < F; j++)
        inv[j] = (float)pow(VL_THETA, -(double)j / (double)F);

    for (int64_t t = 0; t < lt; t++)
        for (int64_t y = 0; y < th; y++)
            for (int64_t x = 0; x < tw; x++) {
                int64_t r = (t * th + y) * tw + x;
                float id[3];
                id[0] = 2.0f * ((float)t + 0.5f) / (float)lt - 1.0f;
                id[1] = 2.0f * ((float)y + 0.5f) / (float)th - 1.0f;
                id[2] = 2.0f * ((float)x + 0.5f) / (float)tw - 1.0f;
                for (int64_t a = 0; a < VL_ROPE_AXES; a++)
                    for (int64_t j = 0; j < F; j++) {
                        float ang = (float)(2.0 * M_PI) * id[a] * inv[j];
                        cos[r * VL_ROPE_DIM + a * F + j] = cosf(ang);
                        sin[r * VL_ROPE_DIM + a * F + j] = sinf(ang);
                    }
            }
    /* Register and cls rows sit at the origin: angle 0, cos 1, sin 0. */
    for (int64_t r = npat; r < npat + VL_SUFFIX; r++)
        for (int64_t i = 0; i < VL_ROPE_DIM; i++) {
            cos[r * VL_ROPE_DIM + i] = 1.0f;
            sin[r * VL_ROPE_DIM + i] = 0.0f;
        }
}

/* The checkpoint applies the rotation as rotate_half over the 48 used dims,
 * with the 24 angles tiled twice — so dim i and dim i+24 share angle i. */
static void vl_rope_apply(float *x, int64_t seq, int64_t heads, int64_t head_dim,
                          const float *cos, const float *sin) {
    const int64_t half = VL_ROPE_DIM;   /* 24 rotated pairs */
    for (int64_t s = 0; s < seq; s++)
        for (int64_t h = 0; h < heads; h++) {
            float *v = x + (s * heads + h) * head_dim;
            const float *c = cos + s * VL_ROPE_DIM;
            const float *n = sin + s * VL_ROPE_DIM;
            for (int64_t i = 0; i < half; i++) {
                float a = v[i], b = v[i + half];
                v[i] = a * c[i] - b * n[i];
                v[i + half] = a * n[i] + b * c[i];
            }
        }
}

/* One tile through the 36-block decoder. Writes the tile's pixels into `dst`
 * as frames x (th*16) x (tw*16) x 3, frame-major. */
static bool vl_decode_tile(const cwick_safetensors *s, const float *weff,
                           const float *beff, const float *lat, int64_t lt,
                           int64_t lh, int64_t lw, int64_t y0, int64_t th,
                           int64_t x0, int64_t tw, unsigned char *dst,
                           cwick_error *err) {
    const int64_t D = VL_DIM, H = VL_HEADS, HD = VL_HEAD_DIM;
    const int64_t npat = lt * th * tw;
    const int64_t seq = npat + VL_SUFFIX;
    const int64_t frames = lt * H3V_VAE_FRAMES;
    (void)frames;
    const int64_t ph = th * H3V_VAE_PIXEL, pw = tw * H3V_VAE_PIXEL;
    bool ok = false;

    float *x = vl_f32(seq * D);
    float *xn = vl_f32(seq * D);
    float *qkv = vl_f32(seq * VL_QKV);
    float *att = vl_f32(seq * D);
    float *tmp = vl_f32(seq * VL_FFN * 2);
    float *cos = vl_f32(seq * VL_ROPE_DIM);
    float *sin = vl_f32(seq * VL_ROPE_DIM);
    float *qh = vl_f32(seq * HD), *kh = vl_f32(seq * HD), *vh = vl_f32(seq * HD);
    float *prod = vl_f32(seq * HD);
    float *out = NULL;

    /* ---- embed ---------------------------------------------------- */
    for (int64_t t = 0; t < lt; t++)
        for (int64_t y = 0; y < th; y++)
            for (int64_t xx = 0; xx < tw; xx++) {
                int64_t r = (t * th + y) * tw + xx;
                /* The latent is [C][T][H][W]; gather the 24 channels of this
                 * cell before the folded 24 -> 2048 projection. */
                float vec[H3V_VAE_LATENT_CH];
                for (int64_t c = 0; c < H3V_VAE_LATENT_CH; c++)
                    vec[c] = lat[((c * lt + t) * lh + (y0 + y)) * lw + (x0 + xx)];
                float *o = x + r * D;
                for (int64_t d = 0; d < D; d++) {
                    float acc = beff[d];
                    const float *w = weff + d * H3V_VAE_LATENT_CH;
                    for (int64_t c = 0; c < H3V_VAE_LATENT_CH; c++)
                        acc += w[c] * vec[c];
                    o[d] = acc;
                }
            }
    {
        vl_mat reg;
        if (!vl_get(s, "decoder.register_tokens", &reg, err)) goto done;
        memcpy(x + npat * D, reg.p, (size_t)(VL_REGISTERS * D) * sizeof(float));
    }
    memset(x + (npat + VL_REGISTERS) * D, 0, (size_t)D * sizeof(float));

    vl_rope(lt, th, tw, npat, cos, sin);

    /* ---- 36 blocks ------------------------------------------------ */
    char name[128];
    for (int64_t L = 0; L < VL_LAYERS; L++) {
        vl_mat qw, qb, ow, ob, f1w, f1b, f2w, f2b, n1, n2, s1, s2;
        snprintf(name, sizeof(name), "decoder.transformer_blocks.%lld.attn.to_qkv.weight",
                 (long long)L);
        if (!vl_get(s, name, &qw, err)) goto done;
        snprintf(name, sizeof(name), "decoder.transformer_blocks.%lld.attn.to_qkv.bias",
                 (long long)L);
        if (!vl_get(s, name, &qb, err)) goto done;
        snprintf(name, sizeof(name), "decoder.transformer_blocks.%lld.attn.to_out.weight",
                 (long long)L);
        if (!vl_get(s, name, &ow, err)) goto done;
        snprintf(name, sizeof(name), "decoder.transformer_blocks.%lld.attn.to_out.bias",
                 (long long)L);
        if (!vl_get(s, name, &ob, err)) goto done;
        snprintf(name, sizeof(name), "decoder.transformer_blocks.%lld.ff.w1.weight",
                 (long long)L);
        if (!vl_get(s, name, &f1w, err)) goto done;
        snprintf(name, sizeof(name), "decoder.transformer_blocks.%lld.ff.w1.bias",
                 (long long)L);
        if (!vl_get(s, name, &f1b, err)) goto done;
        snprintf(name, sizeof(name), "decoder.transformer_blocks.%lld.ff.w2.weight",
                 (long long)L);
        if (!vl_get(s, name, &f2w, err)) goto done;
        snprintf(name, sizeof(name), "decoder.transformer_blocks.%lld.ff.w2.bias",
                 (long long)L);
        if (!vl_get(s, name, &f2b, err)) goto done;
        snprintf(name, sizeof(name), "decoder.transformer_blocks.%lld.norm1.weight",
                 (long long)L);
        if (!vl_get(s, name, &n1, err)) goto done;
        snprintf(name, sizeof(name), "decoder.transformer_blocks.%lld.norm2.weight",
                 (long long)L);
        if (!vl_get(s, name, &n2, err)) goto done;
        snprintf(name, sizeof(name), "decoder.transformer_blocks.%lld.scale1",
                 (long long)L);
        if (!vl_get(s, name, &s1, err)) goto done;
        snprintf(name, sizeof(name), "decoder.transformer_blocks.%lld.scale2",
                 (long long)L);
        if (!vl_get(s, name, &s2, err)) goto done;

        for (int64_t r = 0; r < seq; r++)
            vl_rmsnorm(xn + r * D, x + r * D, n1.p, D, 1e-5f);
        vl_linear(qkv, xn, &qw, qb.p, seq);

        /* Per head: q | k | v, each 64 wide, contiguous inside the row. */
        for (int64_t h = 0; h < H; h++) {
            for (int64_t r = 0; r < seq; r++) {
                const float *src = qkv + r * VL_QKV + h * 3 * HD;
                memcpy(qh + r * HD, src, (size_t)HD * sizeof(float));
                memcpy(kh + r * HD, src + HD, (size_t)HD * sizeof(float));
                memcpy(vh + r * HD, src + 2 * HD, (size_t)HD * sizeof(float));
                /* qk_norm is a non-affine RMSNorm: no weight to apply. */
                vl_rmsnorm(qh + r * HD, qh + r * HD, NULL, HD, 1e-5f);
                vl_rmsnorm(kh + r * HD, kh + r * HD, NULL, HD, 1e-5f);
            }
            vl_rope_apply(qh, seq, 1, HD, cos, sin);
            vl_rope_apply(kh, seq, 1, HD, cos, sin);

            const float scale = 1.0f / sqrtf((float)HD);
            /* The per-head buffers are already [seq][64] with no head stride,
             * so one dispatch serves this head. */
            if (h3v_gpu_on() &&
                h3v_gpu_attention(qh, kh, vh, prod, (int)seq, (int)HD, 1, (int)HD)) {
                for (int64_t i = 0; i < seq; i++)
                    memcpy(att + i * D + h * HD, prod + i * HD,
                           (size_t)HD * sizeof(float));
                continue;
            }
            for (int64_t q0 = 0; q0 < seq; q0 += VL_OPS_TILE) {
                int64_t qn = seq - q0 < VL_OPS_TILE ? seq - q0 : VL_OPS_TILE;
                float *sc = vl_f32(qn * seq);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)qn,
                            (int)seq, (int)HD, scale, qh + q0 * HD, (int)HD, kh,
                            (int)HD, 0.0f, sc, (int)seq);
                for (int64_t i = 0; i < qn; i++) {
                    float *row = sc + i * seq;
                    float mx = row[0];
                    for (int64_t j = 1; j < seq; j++)
                        if (row[j] > mx) mx = row[j];
                    float sum = 0.0f;
                    for (int64_t j = 0; j < seq; j++) {
                        row[j] = expf(row[j] - mx);
                        sum += row[j];
                    }
                    float inv = 1.0f / sum;
                    for (int64_t j = 0; j < seq; j++) row[j] *= inv;
                }
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)qn,
                            (int)HD, (int)seq, 1.0f, sc, (int)seq, vh, (int)HD,
                            0.0f, prod, (int)HD);
                for (int64_t i = 0; i < qn; i++)
                    memcpy(att + (q0 + i) * D + h * HD, prod + i * HD,
                           (size_t)HD * sizeof(float));
                free(sc);
            }
        }
        vl_linear(xn, att, &ow, ob.p, seq);
        for (int64_t r = 0; r < seq; r++) {
            float *xr = x + r * D;
            const float *ar = xn + r * D;
            for (int64_t i = 0; i < D; i++) xr[i] += s1.p[i] * ar[i];
        }

        for (int64_t r = 0; r < seq; r++)
            vl_rmsnorm(xn + r * D, x + r * D, n2.p, D, 1e-5f);
        vl_linear(tmp, xn, &f1w, f1b.p, seq);
        /* w1 emits [gate | value] as two halves; act the gate, multiply the
         * value, then project back down. */
        for (int64_t r = 0; r < seq; r++) {
            float *g = tmp + r * VL_FFN * 2;
            float *v = g + VL_FFN;
            for (int64_t i = 0; i < VL_FFN; i++)
                g[i] = (g[i] / (1.0f + expf(-g[i]))) * v[i];
        }
        {
            /* fc2 reads the value half of a 2*F-wide row. The stride is passed
             * straight through, so no compaction copy is needed. */
            if (!h3v_gpu_on() ||
                !h3v_gpu_gemm_f32(tmp, f2w.p, att, (int)seq, (int)VL_FFN, (int)D,
                                  (int)(VL_FFN * 2), (int)D))
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)seq, (int)D,
                            (int)VL_FFN, 1.0f, tmp, (int)(VL_FFN * 2), f2w.p,
                            (int)VL_FFN, 0.0f, att, (int)D);
            for (int64_t r = 0; r < seq; r++) {
                float *xr = x + r * D;
                const float *fr = att + r * D;
                for (int64_t i = 0; i < D; i++)
                    xr[i] += s2.p[i] * (fr[i] + f2b.p[i]);
            }
        }
    }

    /* ---- output projection --------------------------------------- */
    {
        vl_mat nw, nb, pw, pb;
        if (!vl_get(s, "decoder.norm_out.weight", &nw, err)) goto done;
        if (!vl_get(s, "decoder.norm_out.bias", &nb, err)) goto done;
        if (!vl_get(s, "decoder.proj_out.weight", &pw, err)) goto done;
        if (!vl_get(s, "decoder.proj_out.bias", &pb, err)) goto done;
        for (int64_t r = 0; r < seq; r++)
            vl_layernorm(xn + r * D, x + r * D, nw.p, nb.p, D, 1e-5f);
        out = vl_f32(npat * VL_PATCH_DIM);
        vl_linear(out, xn, &pw, pb.p, npat);
    }

    /* Each token expands to one 3x4x16x16 block: channel-major inside the
     * token, then frame, then row, then column. */
    for (int64_t t = 0; t < lt; t++)
        for (int64_t y = 0; y < th; y++)
            for (int64_t xx = 0; xx < tw; xx++) {
                int64_t r = (t * th + y) * tw + xx;
                const float *tok = out + r * VL_PATCH_DIM;
                for (int64_t c = 0; c < 3; c++)
                    for (int64_t ft = 0; ft < H3V_VAE_FRAMES; ft++) {
                        int64_t f = t * H3V_VAE_FRAMES + ft;
                        unsigned char *fp = dst + ((size_t)f * ph + y * H3V_VAE_PIXEL) * pw * 3;
                        for (int64_t py = 0; py < H3V_VAE_PIXEL; py++)
                            for (int64_t px = 0; px < H3V_VAE_PIXEL; px++) {
                                int64_t idx = ((c * H3V_VAE_FRAMES + ft) * H3V_VAE_PIXEL + py)
                                                  * H3V_VAE_PIXEL + px;
                                float v = tok[idx];
                                /* imagenet de-normalisation, then clamp */
                                static const float mean[3] = {0.485f, 0.456f, 0.406f};
                                static const float stdv[3] = {0.229f, 0.224f, 0.225f};
                                v = v * stdv[c] + mean[c];
                                if (v < 0.0f) v = 0.0f;
                                if (v > 1.0f) v = 1.0f;
                                fp[(py * pw + (xx * H3V_VAE_PIXEL + px)) * 3 + c] =
                                    (unsigned char)(v * 255.0f + 0.5f);
                            }
                    }
            }
    ok = true;

done:
    free(x); free(xn); free(qkv); free(att); free(tmp);
    free(cos); free(sin); free(qh); free(kh); free(vh); free(prod); free(out);
    return ok;
}

/* Tile plan along one axis, following the released decoder's split_tiles:
 * every intermediate tile keeps the full tile size and the remaining length is
 * distributed into the overlaps. A short tail tile would change the local RoPE
 * grid and make adjacent tiles disagree about the same spatial positions. */
static int vl_plan(int64_t len, int64_t *starts, int64_t *sizes,
                   int64_t *overlaps, int cap) {
    int64_t tile = H3V_VAE_TILE_CELLS, ov = H3V_VAE_TILE_OVERLAP;
    { const char *e = getenv("CWICK_H3V_VAE_TILE");
      if (e && *e) { long v = strtol(e, NULL, 10); if (v >= 4 && v <= 128) {
          tile = v; ov = v / 4; if (ov < 1) ov = 1; } } }
    if (len <= tile) {
        starts[0] = 0;
        sizes[0] = len;
        overlaps[0] = 0;
        return 1;
    }
    int n = (int)((len + tile - 1) / tile);
    int64_t remaining = 0;
    for (;;) {
        remaining = tile * n - ov * (n - 1) - len;
        if (remaining >= 0) break;
        n++;
        if (n >= cap) return 0;
    }
    for (int i = 0; i < n - 1; i++) overlaps[i] = ov;
    for (int64_t i = 0; i < remaining; i++) overlaps[i % (n - 1)]++;
    starts[0] = 0;
    for (int i = 0; i < n; i++) {
        sizes[i] = tile;
        if (i > 0) starts[i] = starts[i - 1] + tile - overlaps[i - 1];
    }
    return n;
}

bool h3v_vae_plan_selftest(void) {
    int64_t starts[16], sizes[16], overlaps[16];
    int n = vl_plan(64, starts, sizes, overlaps, 16);
    if (n != 5) return false;
    for (int i = 0; i < n; i++)
        if (starts[i] != i * 12 || sizes[i] != 16 || (i < n - 1 && overlaps[i] != 4))
            return false;
    n = vl_plan(48, starts, sizes, overlaps, 16);
    if (n != 4 || overlaps[0] != 6 || overlaps[1] != 5 || overlaps[2] != 5)
        return false;
    for (int i = 0; i < n; i++)
        if (sizes[i] != 16 || starts[i] != (int64_t[]){0, 10, 21, 32}[i])
            return false;
    return true;
}

/* ---- temporal chunk geometry -------------------------------------------
 * One training clip is 17 pixel frames = 5 latent steps. Because 17 is not a
 * multiple of the 4-frame latent stride, the encoder drops 3 tail latents per
 * chunk. A correct decode re-decodes the 2-latent overlap and linearly blends
 * the shared 5 frames. Decoding all steps in one pass leaves a hard cut at
 * every chunk boundary (measured jump MAE 21.34 against a 4.52 median on a
 * 544x960 clip). The rope grid is length-normalised, so chunk decoding also
 * restores the local 7-latent rope the decoder was trained with.
 */
#define VL_T_CHUNK   5   /* tokens_chunk_size                    */
#define VL_T_OVERLAP 2   /* token_overlap  = (-3) % 5            */
#define VL_T_DROP    3   /* token_drop                           */
#define VL_T_PREPAD  3   /* frame_pre_padding = (-17) % 4        */
#define VL_T_XFADE   5   /* frame_overlap = 2*4 - 3              */

/* Copy latent steps [start, start+n) from a [C][lt][lh][lw] tensor into a
 * fresh [C][n][lh][lw]. Steps past the end repeat the last latent, which is
 * how a partial tail chunk is padded. */
static float *vl_slice_lat(const float *lat, int64_t lt, int64_t lh, int64_t lw,
                           int64_t start, int64_t n) {
    const int64_t plane = lh * lw;
    float *out = vl_f32((size_t)H3V_VAE_LATENT_CH * (size_t)n * (size_t)plane);
    for (int64_t c = 0; c < H3V_VAE_LATENT_CH; c++) {
        const float *sc = lat + (size_t)c * (size_t)lt * (size_t)plane;
        float *dc = out + (size_t)c * (size_t)n * (size_t)plane;
        for (int64_t t = 0; t < n; t++) {
            int64_t s = start + t;
            if (s < 0) s = 0;
            if (s >= lt) s = lt - 1;
            memcpy(dc + (size_t)t * (size_t)plane,
                   sc + (size_t)s * (size_t)plane, (size_t)plane * sizeof(float));
        }
    }
    return out;
}

/* Decode one temporal chunk's latent steps through every spatial tile and
 * cross-fade the tiles into `rgb` (frames x H x W x 3, frame-major). Factored
 * out of the entry point so the whole-clip and chunked paths share the exact
 * same spatial stitch. */
static bool vl_decode_spatial(const cwick_safetensors *s, const float *weff,
                              const float *beff, const float *lat, int64_t lt,
                              int64_t lh, int64_t lw, unsigned char *rgb,
                              cwick_error *err, h3v_progress_fn cb, void *cbctx,
                              int *done_tiles, int total_tiles) {
    const int64_t frames = lt * H3V_VAE_FRAMES;
    const int64_t H = lh * H3V_VAE_PIXEL, W = lw * H3V_VAE_PIXEL;
    int64_t ys[512], yl[512], yov[512], xs[512], xl[512], xov[512];
    const int ny = vl_plan(lh, ys, yl, yov, 512);
    const int nx = vl_plan(lw, xs, xl, xov, 512);
    if (ny <= 0 || nx <= 0) return false;

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            const int64_t th = yl[i], tw = xl[j];
            const int64_t ph = th * H3V_VAE_PIXEL, pw = tw * H3V_VAE_PIXEL;
            unsigned char *tile =
                (unsigned char *)cwick_xmalloc((size_t)frames * ph * pw * 3);
            if (!vl_decode_tile(s, weff, beff, lat, lt, lh, lw, ys[i], th,
                                xs[j], tw, tile, err)) {
                free(tile);
                return false;
            }
            const int64_t gy0 = ys[i] * H3V_VAE_PIXEL;
            const int64_t gx0 = xs[j] * H3V_VAE_PIXEL;
            for (int64_t f = 0; f < frames; f++)
                for (int64_t y = 0; y < ph; y++) {
                    const int64_t gy = gy0 + y;
                    const int64_t ov_y = i > 0 ? yov[i - 1] * H3V_VAE_PIXEL : 0;
                    float ay = (ov_y > 0 && y < ov_y) ? (float)y / (float)ov_y : 1.0f;
                    for (int64_t x = 0; x < pw; x++) {
                        const int64_t gx = gx0 + x;
                        const int64_t ov_x = j > 0 ? xov[j - 1] * H3V_VAE_PIXEL : 0;
                        float ax = (ov_x > 0 && x < ov_x) ? (float)x / (float)ov_x : 1.0f;
                        float a = (ay < 1.0f && ax < 1.0f) ? 0.5f * (ay + ax)
                                  : (ay < 1.0f ? ay : ax);
                        unsigned char *dst =
                            rgb + ((size_t)f * H + gy) * W * 3 + gx * 3;
                        const unsigned char *src =
                            tile + ((size_t)f * ph + y) * pw * 3 + x * 3;
                        if (a >= 1.0f) {
                            memcpy(dst, src, 3);
                        } else {
                            for (int c = 0; c < 3; c++)
                                dst[c] = (unsigned char)((float)dst[c] * (1.0f - a) +
                                                         (float)src[c] * a + 0.5f);
                        }
                    }
                }
            free(tile);
            (*done_tiles)++;
            if (cb) cb(cbctx, H3V_PHASE_VIDEO, 0, 0,
                       (double)*done_tiles / (double)total_tiles);
        }
    }
    return true;
}

unsigned char *h3v_vae_decode(const h3v_model *m, const float *lat, int64_t lt,
                              int64_t lh, int64_t lw, h3v_progress_fn cb,
                              void *cbctx, cwick_error *err) {
    const cwick_safetensors *s = h3v_shards_vae(m);
    if (!s) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "video VAE shards are not open");
        return NULL;
    }
    if (lt <= 0 || lh <= 0 || lw <= 0) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "empty latent grid");
        return NULL;
    }

    float *weff = NULL, *beff = NULL;
    vl_fold_embed(s, &weff, &beff, err);
    if (!weff) return NULL;

    const int64_t H = lh * H3V_VAE_PIXEL, W = lw * H3V_VAE_PIXEL;
    /* `lat` arrives normalised from the DiT. The decoder (post_quant_conv
     * folded into x_embedder) is trained on the raw VAE latent, so denormalise
     * once up front: the conversion is per cell, which keeps every tile's view
     * identical to a whole-clip decode. */
    const int64_t lat_plane = lt * lh * lw;
    float *lat_dn = vl_f32((size_t)H3V_VAE_LATENT_CH * (size_t)lat_plane);
    memcpy(lat_dn, lat, (size_t)H3V_VAE_LATENT_CH * (size_t)lat_plane *
                         sizeof(float));
    vl_lat_denorm(lat_dn, lat_plane);

    /* A single latent step has no chunk boundary to blend, and the reference
     * frame round trip needs the whole-clip path. Keep it, and keep an explicit
     * A/B switch so the two decoders can be measured on the same latent. */
    bool chunked = lt > VL_T_CHUNK;
    { const char *e = getenv("CWICK_H3V_TEMPORAL_XFADE");
      if (e && *e && strcmp(e, "0") == 0) chunked = false; }

    int64_t ys[512], yl[512], yov[512], xs[512], xl[512], xov[512];
    const int ny = vl_plan(lh, ys, yl, yov, 512);
    const int nx = vl_plan(lw, xs, xl, xov, 512);
    if (ny <= 0 || nx <= 0) {
        free(weff); free(beff); free(lat_dn);
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "video VAE tile plan is too large");
        return NULL;
    }
    int done_tiles = 0;

    if (!chunked) {
        const int64_t frames = lt * H3V_VAE_FRAMES;
        unsigned char *rgb = (unsigned char *)cwick_xmalloc((size_t)frames * H * W * 3);
        memset(rgb, 0, (size_t)frames * H * W * 3);
        const bool ok = vl_decode_spatial(s, weff, beff, lat_dn, lt, lh, lw, rgb,
                                          err, cb, cbctx, &done_tiles, ny * nx);
        free(weff);
        free(beff);
        free(lat_dn);
        if (!ok) { free(rgb); return NULL; }
        return rgb;
    }

    /* Chunked decode: 5 latent steps plus a 2-step overlap per clip. The
     * overlap makes adjacent chunks predict the same two latent steps, so the
     * shared frames can be blended instead of hard-cut. */
    const int64_t num_tokens = lt + VL_T_DROP;
    int64_t pad = ((-num_tokens) % VL_T_CHUNK + VL_T_CHUNK) % VL_T_CHUNK;
    int64_t num_chunks = (num_tokens + pad) / VL_T_CHUNK - 1;
    if (num_chunks < 1) num_chunks = 1;
    const int64_t clt = VL_T_CHUNK + VL_T_OVERLAP;
    const int64_t cframes = clt * H3V_VAE_FRAMES;
    /* 17*num_chunks + 5 legal frames; the mux keeps the first 17*num_chunks. */
    const int64_t out_frames = num_chunks * H3V_CLIP_FRAMES + VL_T_XFADE;
    const size_t fbytes = (size_t)H * W * 3;

    float *lat_pad = vl_slice_lat(lat_dn, lt, lh, lw, 0, lt + pad);
    unsigned char *rgb = (unsigned char *)cwick_xmalloc((size_t)out_frames * fbytes);
    memset(rgb, 0, (size_t)out_frames * fbytes);
    float *ov = NULL;                 /* previous chunk's shared 5 frames */
    bool ok = true;

    for (int64_t ci = 0; ci < num_chunks && ok; ci++) {
        const int64_t start = ci * VL_T_CHUNK;
        float *clat = vl_slice_lat(lat_pad, lt + pad, lh, lw, start, clt);
        unsigned char *chunk_rgb =
            (unsigned char *)cwick_xmalloc((size_t)cframes * fbytes);
        memset(chunk_rgb, 0, (size_t)cframes * fbytes);
        if (!vl_decode_spatial(s, weff, beff, clat, clt, lh, lw, chunk_rgb, err,
                               cb, cbctx, &done_tiles, num_chunks * ny * nx)) {
            ok = false;
        } else {
            /* A decoded chunk holds clt*4 frames. The first latent covers only
             * one pixel frame, so VL_T_PREPAD leading frames are dropped; the
             * main 17 come from [3,20) and the shared tail 5 from [23,28). */
            const unsigned char *main = chunk_rgb + (size_t)VL_T_PREPAD * fbytes;
            const unsigned char *tail =
                chunk_rgb + (size_t)(VL_T_CHUNK * H3V_VAE_FRAMES + VL_T_PREPAD) * fbytes;
            unsigned char *dst = rgb + (size_t)ci * H3V_CLIP_FRAMES * fbytes;
            for (int64_t f = 0; f < H3V_CLIP_FRAMES; f++) {
                const unsigned char *m = main + (size_t)f * fbytes;
                unsigned char *d = dst + (size_t)f * fbytes;
                if (ov && f < VL_T_XFADE) {
                    const float w = (float)f / (float)VL_T_XFADE;
                    const float iw = 1.0f - w;
                    for (size_t b = 0; b < fbytes; b++)
                        d[b] = (unsigned char)(ov[(size_t)f * fbytes + b] * iw +
                                               (float)m[b] * w + 0.5f);
                } else {
                    memcpy(d, m, fbytes);
                }
            }
            if (!ov) ov = vl_f32((size_t)VL_T_XFADE * fbytes);
            for (int64_t f = 0; f < VL_T_XFADE; f++)
                for (size_t b = 0; b < fbytes; b++)
                    ov[(size_t)f * fbytes + b] = (float)tail[(size_t)f * fbytes + b];
        }
        free(chunk_rgb);
        free(clat);
    }

    if (ok && ov) {
        unsigned char *last = rgb + (size_t)num_chunks * H3V_CLIP_FRAMES * fbytes;
        for (int64_t f = 0; f < VL_T_XFADE; f++)
            for (size_t b = 0; b < fbytes; b++)
                last[(size_t)f * fbytes + b] =
                    (unsigned char)(ov[(size_t)f * fbytes + b] + 0.5f);
    }

    free(ov);
    free(lat_pad);
    free(weff);
    free(beff);
    free(lat_dn);
    if (!ok) { free(rgb); return NULL; }
    return rgb;
}

/* ====================================================================== */
/* Visual VAE encoder — one reference frame -> one latent step.           */
/*                                                                        */
/* EncoderFCN3D is a causal 3D CNN. For a single frame the temporal axis   */
/* collapses: causal padding puts (KT-1) zeros in front, so only the last  */
/* temporal kernel slice contributes and the output depth is always 1.    */
/* Every 3D conv therefore reduces to a 2D conv with kernel               */
/* w[:, :, KT-1, :, :]. Spatial pad is reflect for res/conv blocks and a  */
/* (0,1,0,1) lower-only zero pad for the strided downsamplers.            */
/*                                                                        */
/* Layout from the shipped config: ch=128, ch_mult=[1,2,2,4,4,8], two     */
/* res blocks per level, space_down=[2,2,2,2,1,1], GroupNorm(32) eps 1e-6 */
/* + SiLU, then norm_out -> conv_out(48) -> quant_conv(48); the first 24   */
/* channels are the latent mean (eps=0 for a conditioning frame).         */
/* The encoder stages can be replayed with tools/h3vencprobe (pure C) against
 * a fixed RGB input; the latent is what the conditioning frame uses.         */
/* ====================================================================== */
#undef I   /* <Accelerate> pulls in <complex.h>, whose I macro collides. */

typedef struct {
    float *p;                    /* [c][h][w], row-major */
    int64_t c, h, w;
} ec_vol;

/* out[o][oh][ow] = sum_{i,kh,kw} w[o][i,kh,kw]*in[i][oh*s+kh][ow*s+kw].
 * k is the spatial kernel (3 for convs, 1 for shortcuts). When k>1 the
 * input is padded by 1: reflect on both edges, or lower-only zeros when
 * `lower` is set (Downsample3D's (0,1,0,1) pad). */
static void ec_conv2d(float *out, const ec_vol *in, const float *w,
                      const float *bias, int64_t och, int64_t k, int64_t s,
                      bool lower) {
    const int64_t ich = in->c, ih = in->h, iw = in->w;
    int64_t p0 = 0, p1 = 0;
    if (k > 1) { if (lower) { p0 = 0; p1 = 1; } else { p0 = 1; p1 = 1; } }
    const int64_t ph = ih + p0 + p1, pw = iw + p0 + p1;
    const int64_t oh = (ph - k) / s + 1, ow = (pw - k) / s + 1;

    float *pad = vl_f32((size_t)ich * ph * pw);
    for (int64_t c = 0; c < ich; c++) {
        const float *src = in->p + c * ih * iw;
        float *dst = pad + c * ph * pw;
        for (int64_t y = 0; y < ph; y++) {
            int64_t sy = y - p0;
            if (sy < 0) sy = lower ? 0 : -sy;
            else if (sy >= ih) sy = lower ? ih : 2 * ih - 2 - sy;
            for (int64_t x = 0; x < pw; x++) {
                int64_t sx = x - p0;
                if (sx < 0) sx = lower ? 0 : -sx;
                else if (sx >= iw) sx = lower ? iw : 2 * iw - 2 - sx;
                /* lower pad is zero; reflect uses the clamped index. */
                dst[y * pw + x] = (lower && (y >= ph - p1 || x >= pw - p1))
                                  ? 0.0f : src[sy * iw + sx];
            }
        }
    }

    const int64_t khw = k * k;
    float *col = vl_f32((size_t)ich * khw * oh * ow);
    for (int64_t i = 0; i < ich; i++) {
        const float *plane = pad + i * ph * pw;
        for (int64_t t = 0; t < khw; t++) {
            int64_t dy = t / k, dx = t % k;
            float *crow = col + (i * khw + t) * oh * ow;
            for (int64_t y = 0; y < oh; y++)
                for (int64_t x = 0; x < ow; x++)
                    crow[y * ow + x] = plane[(y * s + dy) * pw + (x * s + dx)];
        }
    }
    free(pad);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)och,
                (int)(oh * ow), (int)(ich * khw), 1.0f, w, (int)(ich * khw),
                col, (int)(oh * ow), 0.0f, out, (int)(oh * ow));
    free(col);
    if (bias)
        for (int64_t o = 0; o < och; o++)
            for (int64_t j = 0; j < oh * ow; j++) out[o * oh * ow + j] += bias[o];
}

/* GroupNorm(32) over each frame, eps=1e-6, affine, then SiLU. */
static void ec_gn_silu(float *out, const ec_vol *in, const float *weight,
                       const float *bias) {
    const int64_t C = in->c, S = in->h * in->w, G = 32, CPG = C / G;
    for (int64_t g = 0; g < G; g++) {
        const float *x = in->p + g * CPG * S;
        float mean = 0.0f, m2 = 0.0f;
        for (int64_t i = 0; i < CPG * S; i++) { mean += x[i]; m2 += x[i] * x[i]; }
        mean /= (float)(CPG * S);
        float var = m2 / (float)(CPG * S) - mean * mean;
        float inv = 1.0f / sqrtf(var + 1e-6f);
        for (int64_t i = 0; i < CPG; i++) {
            float sc = weight[g * CPG + i], sh = bias[g * CPG + i];
            float *o = out + (g * CPG + i) * S;
            for (int64_t j = 0; j < S; j++) {
                float v = (x[i * S + j] - mean) * inv * sc + sh;
                o[j] = v / (1.0f + expf(-v));
            }
        }
    }
}

/* Keep only the last temporal slice of a 3D conv weight. src is [O, I*KT*KH*KW];
 * result is a row-major [O, I*KH*KW] 2D kernel. */
static float *ec_kernel_last_t(const float *src, int64_t O, int64_t I,
                               int64_t KT, int64_t KH, int64_t KW) {
    float *k = vl_f32((size_t)O * I * KH * KW);
    for (int64_t o = 0; o < O; o++)
        for (int64_t i = 0; i < I; i++)
            for (int64_t y = 0; y < KH; y++)
                for (int64_t x = 0; x < KW; x++)
                    k[(o * I + i) * KH * KW + y * KW + x] =
                        src[(o * I + i) * KT * KH * KW + (KT - 1) * KH * KW
                            + y * KW + x];
    return k;
}

/* Debug: dump an intermediate volume to a directory when EC_DUMP is set.
 * Never active in the product path; used only to align against a reference. */
static void ec_dump(const char *tag, const float *p, int64_t n) {
    const char *d = getenv("CWICK_H3V_ENC_DUMP");
    if (!d || !*d) return;
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.f32", d, tag);
    FILE *f = fopen(path, "wb");
    if (f) { fwrite(p, sizeof(float), (size_t)n, f); fclose(f); }
}

static bool ec_get(const cwick_safetensors *s, const char *name,
                   const float **p, cwick_error *err) {
    cwick_st_tensor t;
    if (cwick_safetensors_get(s, name, &t, err) != CWICK_OK) return false;
    if (t.dtype != CWICK_ST_F32) {
        cwick_error_set(err, CWICK_ERR_PARSE, "encoder tensor %s is not float32", name);
        return false;
    }
    *p = (const float *)t.data;
    return true;
}

/* One ResnetBlock3D. Temporary buffers are allocated here and freed before
 * return; `out` is caller-owned and large enough for O2 * h * w. */
static bool ec_resblock(const cwick_safetensors *s, const char *prefix,
                        const ec_vol *x, ec_vol *out, cwick_error *err) {
    char n[192];
    cwick_st_tensor t;
    bool ok = false;
    float *n1 = NULL, *c1 = NULL, *n2 = NULL, *h2 = NULL, *sx = NULL;
    float *k1 = NULL, *k2 = NULL, *ks = NULL;

    int64_t C = x->c, H = x->h, W = x->w, S = H * W;

    snprintf(n, sizeof(n), "%s.norm1.weight", prefix);
    const float *nw1; if (!ec_get(s, n, &nw1, err)) goto done;
    snprintf(n, sizeof(n), "%s.norm1.bias", prefix);
    const float *nb1; if (!ec_get(s, n, &nb1, err)) goto done;
    n1 = vl_f32((size_t)C * S);
    ec_gn_silu(n1, x, nw1, nb1);

    snprintf(n, sizeof(n), "%s.conv1.weight", prefix);
    if (cwick_safetensors_get(s, n, &t, err) != CWICK_OK) goto done;
    int64_t O1 = t.rows;
    k1 = ec_kernel_last_t((const float *)t.data, O1, C, 3, 3, 3);
    snprintf(n, sizeof(n), "%s.conv1.bias", prefix);
    const float *b1; if (!ec_get(s, n, &b1, err)) goto done;
    c1 = vl_f32((size_t)O1 * S);
    { ec_vol v = { n1, C, H, W }; ec_conv2d(c1, &v, k1, b1, O1, 3, 1, false); }

    snprintf(n, sizeof(n), "%s.norm2.weight", prefix);
    const float *nw2; if (!ec_get(s, n, &nw2, err)) goto done;
    snprintf(n, sizeof(n), "%s.norm2.bias", prefix);
    const float *nb2; if (!ec_get(s, n, &nb2, err)) goto done;
    n2 = vl_f32((size_t)O1 * S);
    { ec_vol v = { c1, O1, H, W }; ec_gn_silu(n2, &v, nw2, nb2); }

    snprintf(n, sizeof(n), "%s.conv2.weight", prefix);
    if (cwick_safetensors_get(s, n, &t, err) != CWICK_OK) goto done;
    int64_t O2 = t.rows;
    k2 = ec_kernel_last_t((const float *)t.data, O2, O1, 3, 3, 3);
    snprintf(n, sizeof(n), "%s.conv2.bias", prefix);
    const float *b2; if (!ec_get(s, n, &b2, err)) goto done;
    h2 = vl_f32((size_t)O2 * S);
    { ec_vol v = { n2, O1, H, W }; ec_conv2d(h2, &v, k2, b2, O2, 3, 1, false); }

    if (O2 != C) {
        snprintf(n, sizeof(n), "%s.nin_shortcut.weight", prefix);
        if (cwick_safetensors_get(s, n, &t, err) != CWICK_OK) goto done;
        ks = ec_kernel_last_t((const float *)t.data, O2, C, 1, 1, 1);
        snprintf(n, sizeof(n), "%s.nin_shortcut.bias", prefix);
        const float *bs; if (!ec_get(s, n, &bs, err)) goto done;
        sx = vl_f32((size_t)O2 * S);
        ec_conv2d(sx, x, ks, bs, O2, 1, 1, false);
        for (int64_t i = 0; i < O2 * S; i++) out->p[i] = sx[i] + h2[i];
    } else {
        memcpy(out->p, x->p, (size_t)C * S * sizeof(float));
        for (int64_t i = 0; i < O2 * S; i++) out->p[i] += h2[i];
    }
    out->c = O2; out->h = H; out->w = W;
    ok = true;

done:
    free(n1); free(c1); free(n2); free(h2); free(sx);
    free(k1); free(k2); free(ks);
    return ok;
}

float *h3v_vae_encode(const h3v_model *m, const unsigned char *rgb, int width,
                      int height, cwick_error *err) {
    const cwick_safetensors *s = h3v_shards_vae(m);
    if (!s) { cwick_error_set(err, CWICK_ERR_ARGUMENT, "visual VAE not open"); return NULL; }
    if (!rgb || width <= 0 || height <= 0 ||
        width % H3V_VAE_PIXEL || height % H3V_VAE_PIXEL) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "encoder needs RGB24 with 16-aligned size");
        return NULL;
    }
    const int64_t lh = height / H3V_VAE_PIXEL, lw = width / H3V_VAE_PIXEL;
    static const float MEAN[3] = { 0.485f, 0.456f, 0.406f };
    static const float STD[3]  = { 0.229f, 0.224f, 0.225f };
    float *in = NULL, *v128 = NULL, *a = NULL, *b = NULL, *z = NULL, *mom = NULL;
    float *q = NULL, *nv = NULL;
    cwick_st_tensor t;
    bool ok = false;

    in = vl_f32((size_t)3 * height * width);
    for (int64_t c = 0; c < 3; c++)
        for (int64_t y = 0; y < height; y++)
            for (int64_t x = 0; x < width; x++)
                in[c * height * width + y * width + x] =
                    ((float)rgb[(y * width + x) * 3 + c] / 255.0f - MEAN[c]) / STD[c];
    ec_vol cur = { in, 3, height, width };
    ec_dump("input_norm", in, (size_t)3 * height * width);

    /* conv_in: 3 -> 128, k3, stride 1, reflect. */
    if (cwick_safetensors_get(s, "encoder.conv_in.weight", &t, err) != CWICK_OK)
        goto done;
    {
        float *k0 = ec_kernel_last_t((const float *)t.data, t.rows, 3, 3, 3, 3);
        ec_dump("conv_in_kernel", k0, (size_t)t.rows * 3 * 3 * 3);
        const float *b0; if (!ec_get(s, "encoder.conv_in.bias", &b0, err)) goto done;
        v128 = vl_f32((size_t)128 * height * width);
        ec_conv2d(v128, &cur, k0, b0, 128, 3, 1, false);
        free(k0);
        ec_dump("conv_in", v128, (size_t)128 * height * width);
        cur.p = v128; cur.c = 128;
    }

    /* Two ping-pong buffers for the residual blocks; each block writes into
     * the one that is not the current input. */
    const int64_t cap = 1024;
    a = vl_f32((size_t)cap * height * width);
    b = vl_f32((size_t)cap * height * width);
    const int SDOWN[6] = { 2, 2, 2, 2, 1, 1 };
    for (int L = 0; L < 6; L++) {
        for (int blk = 0; blk < 2; blk++) {
            char pre[96];
            snprintf(pre, sizeof(pre), "encoder.down.%d.block.%d", L, blk);
            float *dstp = (cur.p == a) ? b : a;
            ec_vol dst = { dstp, cur.c, cur.h, cur.w };
            if (!ec_resblock(s, pre, &cur, &dst, err)) goto done;
            cur.p = dst.p; cur.c = dst.c; cur.h = dst.h; cur.w = dst.w;
            { char tag[64]; snprintf(tag, sizeof(tag), "down%d.block%d", L, blk);
              ec_dump(tag, cur.p, (size_t)cur.c * cur.h * cur.w); }
        }
        if (SDOWN[L] > 1) {
            char dn[96];
            snprintf(dn, sizeof(dn), "encoder.down.%d.downsample.conv.weight", L);
            if (cwick_safetensors_get(s, dn, &t, err) != CWICK_OK) goto done;
            int64_t O = t.rows, I = cur.c;
            float *kd = ec_kernel_last_t((const float *)t.data, O, I, 3, 3, 3);
            snprintf(dn, sizeof(dn), "encoder.down.%d.downsample.conv.bias", L);
            const float *bd; if (!ec_get(s, dn, &bd, err)) goto done;
            int64_t nh = cur.h / SDOWN[L], nw = cur.w / SDOWN[L];
            nv = vl_f32((size_t)O * nh * nw);
            { ec_vol src = { cur.p, I, cur.h, cur.w };
              ec_conv2d(nv, &src, kd, bd, O, 3, SDOWN[L], true); }
            free(kd);
            cur.p = nv; cur.c = O; cur.h = nh; cur.w = nw;
            { char tag[64]; snprintf(tag, sizeof(tag), "down%d.sample", L);
              ec_dump(tag, cur.p, (size_t)cur.c * cur.h * cur.w); }
            nv = NULL;   /* ownership moved to cur */
        }
    }

    if (cur.h != lh || cur.w != lw) {
        cwick_error_set(err, CWICK_ERR_PARSE,
                        "encoder latent %lldx%lld != expected %lldx%lld",
                        (long long)cur.h, (long long)cur.w,
                        (long long)lh, (long long)lw);
        goto done;
    }

    const float *nw, *nb;
    if (!ec_get(s, "encoder.norm_out.weight", &nw, err)) goto done;
    if (!ec_get(s, "encoder.norm_out.bias", &nb, err)) goto done;
    ec_gn_silu(a, &cur, nw, nb);
    ec_dump("norm_out", a, (size_t)cur.c * cur.h * cur.w);
    { ec_vol v = { a, cur.c, cur.h, cur.w };
      if (cwick_safetensors_get(s, "encoder.conv_out.weight", &t, err) != CWICK_OK)
          goto done;
      int64_t O = t.rows;
      float *ko = ec_kernel_last_t((const float *)t.data, O, cur.c, 3, 3, 3);
      const float *bo; if (!ec_get(s, "encoder.conv_out.bias", &bo, err)) goto done;
      mom = vl_f32((size_t)O * cur.h * cur.w);
      ec_conv2d(mom, &v, ko, bo, O, 3, 1, false);
      free(ko);
      ec_dump("conv_out", mom, (size_t)O * cur.h * cur.w);

      cwick_st_tensor qw;
      if (cwick_safetensors_get(s, "quant_conv.weight", &qw, err) != CWICK_OK)
          goto done;
      float *kq = ec_kernel_last_t((const float *)qw.data, qw.rows, O, 1, 1, 1);
      const float *bq; if (!ec_get(s, "quant_conv.bias", &bq, err)) goto done;
      q = vl_f32((size_t)qw.rows * cur.h * cur.w);
      { ec_vol mv = { mom, O, cur.h, cur.w };
        ec_conv2d(q, &mv, kq, bq, qw.rows, 1, 1, false); }
      free(kq);
      ec_dump("quant_conv", q, (size_t)qw.rows * cur.h * cur.w);

      z = vl_f32((size_t)H3V_VAE_LATENT_CH * cur.h * cur.w);
      memcpy(z, q, (size_t)H3V_VAE_LATENT_CH * cur.h * cur.w * sizeof(float));
      /* quant_conv emits the raw latent mean (the second 24 channels are the
       * log-variance, dropped for a deterministic conditioning frame). The DiT
       * consumes normalised latents, so convert here rather than at every use
       * site: the conditioning frame and the sampled clip then share one
       * space. */
      vl_lat_norm(z, cur.h * cur.w);
    }
    ok = true;

done:
    free(in); free(v128); free(a); free(b); free(mom); free(q);
    if (ok) {
        CWICK_INFO("video", "reference image encoded: %dx%d RGB -> 1x%d latent "
                   "(%dx%d, %.2f MiB)", width, height, H3V_VAE_LATENT_CH,
                   (int)lw, (int)lh,
                   (double)H3V_VAE_LATENT_CH * lh * lw * sizeof(float) / 1048576.0);
        return z;
    }
    free(z);
    return NULL;
}

/* Pure-C regression check for the latent-scale conversion. A mixed-sign buffer
 * is normalised then denormalised and must return to its original values; a
 * zero buffer must land exactly on the per-channel mean, which catches a table
 * applied with the wrong stride or the two directions swapped. */
bool h3v_latnorm_selftest(void) {
    const int64_t plane = 7;
    const size_t n = (size_t)H3V_VAE_LATENT_CH * (size_t)plane;
    float *a = vl_f32((int64_t)n);
    float *b = vl_f32((int64_t)n);
    bool ok = true;
    for (int64_t c = 0; c < H3V_VAE_LATENT_CH; c++)
        for (int64_t i = 0; i < plane; i++)
            a[(size_t)c * (size_t)plane + i] =
                0.37f * (float)((int)((c * 3 + i * 5) % 11) - 5);
    memcpy(b, a, n * sizeof(float));
    vl_lat_norm(b, plane);
    vl_lat_denorm(b, plane);
    for (size_t i = 0; i < n; i++)
        if (fabsf(a[i] - b[i]) > 2e-5f) { ok = false; break; }

    memset(a, 0, n * sizeof(float));
    vl_lat_denorm(a, plane);
    for (int64_t c = 0; c < H3V_VAE_LATENT_CH && ok; c++)
        for (int64_t i = 0; i < plane; i++)
            if (fabsf(a[(size_t)c * (size_t)plane + i] - VL_LAT_MEAN[c]) > 1e-6f) {
                ok = false;
                break;
            }
    free(a);
    free(b);
    return ok;
}
