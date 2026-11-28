/* h3v.c — MiniMax-H3 video + stereo-audio pipeline.
 *
 * Layout of one generation:
 *
 *   prompt --H3-Encoder(Qwen3-VL)--> text hidden [n_text, 5120]
 *          --condition_proj-->        condition  [n_text, 5376]
 *   latent [24, T, H, W] --pack 2x2 spatial--> video rows [T*H/2*W/2, 96]
 *   audio  [32, T_a]     --audio_patch_proj--> audio rows [T_a, 32]
 *          --50 DiT blocks (joint self-attention, 3-axis RoPE)-->
 *          --final_layer--> video velocity (96) + audio velocity (32)
 *   velocity --flow-matching scheduler, shift 12.0--> clean latent
 *   latent --36-block ViT decoder, 4 frames x 16x16 per token--> RGB frames
 *   frames + audio --h3v_mp4_*--> mp4 bytes
 *
 * The transformer is 62 GiB of BF16; every sampling step re-reads it, so the
 * pipeline is opened once and generation is serialised behind the model lock.
 * Comments describe mechanism and measured numbers only.
 */
#include "h3v.h"

#include <Accelerate/Accelerate.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "cwick.h"
#include "h3.h"
#include "h3v_gpu.h"
#include "h3v_internal.h"
#include "json.h"
#include "log.h"
#include "mem.h"
#include "metrics.h"
#include "util.h"

/* ---- transformer shape (FL2VA/Ref2VA transformer/config.json) -------- */
#define VD_LAYERS        50
#define VD_HIDDEN        5376
#define VD_HEADS         56
#define VD_HEAD_DIM      128
#define VD_QKV           (VD_HEADS * VD_HEAD_DIM)   /* 7168  */
#define VD_FFN           14336
#define VD_TIME_DIM      2688
#define VD_TIME_HIDDEN   5376
#define VD_TEXT_DIM      5120
#define VD_LATENT        24
#define VD_AUDIO         32
#define VD_PATCH         2                            /* latent cells per row axis */
#define VD_PATCH_DIM     (VD_LATENT * VD_PATCH * VD_PATCH)   /* 96 */
#define VD_ADALN_ROWS    (18 * VD_HIDDEN)             /* 96768 */
#define VD_FINAL_ADALN   (2 * VD_HIDDEN)              /* 10752 */
#define VD_ROPE_FREQ     16
#define VD_ROPE_DIM      (2 * 3 * VD_ROPE_FREQ)       /* 96 of 128 */
#define VD_DEQ_TILE      512

/* ---- ViT video decoder shape (video_vae/config.json) ---------------- */
#define VV_LAYERS        36
#define VV_HIDDEN        2048
#define VV_HEADS         32
#define VV_HEAD_DIM      64
#define VV_QKV           (3 * VV_HIDDEN)              /* fused qkv */
#define VV_FFN           8192
#define VV_REGISTERS     4
#define VV_ROPE_FREQ     8
#define VV_ROPE_DIM      (2 * VV_ROPE_FREQ)           /* 16 of 64 */
#define VV_ROPE_THETA    100.0f
#define VV_PATCH         16                          /* pixels per latent cell */
#define VV_PATCH_T       4                           /* frames per latent cell */
#define VV_OUT_DIM       (3 * VV_PATCH_T * VV_PATCH * VV_PATCH)  /* 3072 */

/* ---- helpers -------------------------------------------------------- */
static double h3v_ms(void) { return (double)cwick_prof_ns() / 1e6; }

static float *h3v_f32(int64_t n) {
    return (float *)cwick_xmalloc((size_t)n * sizeof(float));
}

typedef struct {
    const void *p;
    int64_t rows, cols;
    bool f32;
} h3v_mat;

static bool h3v_mat_get(const cwick_safetensors *saf, const char *name,
                        h3v_mat *out, cwick_error *err) {
    cwick_st_tensor t;
    if (cwick_safetensors_get(saf, name, &t, err) != CWICK_OK) return false;
    out->p = t.data;
    out->rows = t.rows;
    out->cols = t.cols;
    out->f32 = t.dtype == CWICK_ST_F32;
    return true;
}

/* Materialise a vector (norm weight / adaln row) out of BF16 or F32. */
static float *h3v_vec(const h3v_mat *m) {
    int64_t n = m->rows * m->cols;
    float *v = h3v_f32(n);
    if (m->f32) {
        memcpy(v, m->p, (size_t)n * sizeof(float));
    } else {
        const uint16_t *s = (const uint16_t *)m->p;
        for (int64_t i = 0; i < n; i++) v[i] = cwick_bf16_fast(s[i]);
    }
    return v;
}

static void h3v_gemm_cpu(float *out, const float *x, int64_t xstride,
                        const h3v_mat *w, int64_t n) {
    const int64_t rows = w->rows, cols = w->cols;
    if (w->f32) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)n, (int)rows,
                    (int)cols, 1.0f, x, (int)xstride, (const float *)w->p,
                    (int)cols, 0.0f, out, (int)rows);
        return;
    }
    float *tile = h3v_f32(VD_DEQ_TILE * cols);
    for (int64_t r0 = 0; r0 < rows; r0 += VD_DEQ_TILE) {
        int64_t rn = rows - r0 < VD_DEQ_TILE ? rows - r0 : VD_DEQ_TILE;
        const uint16_t *src = (const uint16_t *)w->p + r0 * cols;
        for (int64_t i = 0; i < rn * cols; i++) tile[i] = cwick_bf16_fast(src[i]);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)n, (int)rn,
                    (int)cols, 1.0f, x, (int)xstride, tile, (int)cols, 0.0f,
                    out + r0, (int)rows);
    }
    free(tile);
}

/* out[n][rows] = x[n][cols] @ W^T. Weights are BF16 for the transformer, so
 * they are dequantised a tile at a time: the full 21504x5376 qkv block would
 * be 231 MiB of float32 for a GEMM that reads each row once.
 *
 * `xstride` exists because the SwiGLU leaves its result in the first half of a
 * two-half row; compacting it would be an extra pass over 200 MiB per block. */
static void h3v_gemm_s(float *out, const float *x, int64_t xstride,
                       const h3v_mat *w, int64_t n) {
    const int64_t rows = w->rows, cols = w->cols;
    /* The projection is out[n][rows] = x[n][cols] @ W^T, with strides carried
     * through so a SwiGLU half row needs no copy. Thin products (timestep and
     * modulation, n=1) spend more on a dispatch than the tile buys back. */
    if (n >= 32 && (int64_t)n * (int64_t)cols * rows >= (int64_t)8 * 1024 * 1024 &&
        h3v_gpu_on()) {
        int ok = w->f32
            ? h3v_gpu_gemm_f32(x, w->p, out, (int)n, (int)cols, (int)rows,
                               (int)xstride, (int)rows)
            : h3v_gpu_gemm_bf16(x, w->p, out, (int)n, (int)cols, (int)rows,
                                (int)xstride, (int)rows);
        if (ok) return;
    }
    h3v_gemm_cpu(out, x, xstride, w, n);
}

static void h3v_gemm(float *out, const float *x, const h3v_mat *w, int64_t n) {
    h3v_gemm_s(out, x, w->cols, w, n);
}

static void h3v_add_bias(float *out, const float *bias, int64_t rows, int64_t n) {
    if (!bias) return;
    for (int64_t i = 0; i < n; i++) {
        float *o = out + i * rows;
        for (int64_t j = 0; j < rows; j++) o[j] += bias[j];
    }
}

static void h3v_rmsnorm(float *out, const float *x, const float *w,
                        int64_t dim, float eps) {
    double ss = 0.0;
    for (int64_t i = 0; i < dim; i++) ss += (double)x[i] * x[i];
    float inv = 1.0f / sqrtf((float)(ss / (double)dim) + eps);
    for (int64_t i = 0; i < dim; i++) out[i] = x[i] * inv * (w ? w[i] : 1.0f);
}

static void h3v_silu_inplace(float *x, int64_t n) {
    for (int64_t i = 0; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}

/* One-shot A/B lever: the shipped transformer shards use the release
 * layout (per-head QKV, gate/value halves in that order). Set this to 1
 * only to reproduce the old, mis-decoded behaviour side by side. */
static int h3v_legacy_layout(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("CWICK_H3V_LEGACY_LAYOUT");
        v = (e && *e && strcmp(e, "1") == 0) ? 1 : 0;
    }
    return v;
}

static int h3v_gpu_scalar(void) {
    const char *e = getenv("CWICK_H3V_GPU_SCALAR");
    return !(e && *e && strcmp(e, "0") == 0);
}

static int h3v_gpu_qk(void) {
    const char *e = getenv("CWICK_H3V_GPU_QK");
    return h3v_gpu_scalar() && !(e && *e && strcmp(e, "0") == 0);
}

static int h3v_gpu_swiglu_on(void) {
    const char *e = getenv("CWICK_H3V_GPU_SWIGLU");
    return h3v_gpu_scalar() && e && *e && strcmp(e, "1") == 0;
}

/* The fused projection stores each head as [q | k | v] back to back, while
 * attention needs q, k and v in three contiguous planes. Scatter once per
 * projection instead of rewriting the 231 MiB weight tile per layer. */
static void h3v_split_qkv(float *q, float *k, float *v, const float *src,
                          int64_t rows, int64_t heads, int64_t hd) {
    const int64_t W = heads * 3 * hd, Q = heads * hd;
    for (int64_t r = 0; r < rows; r++) {
        const float *s = src + r * W;
        float *dq = q + r * Q, *dk = k + r * Q, *dv = v + r * Q;
        for (int64_t h = 0; h < heads; h++) {
            const float *sh = s + h * 3 * hd;
            memcpy(dq + h * hd, sh, (size_t)hd * sizeof(float));
            memcpy(dk + h * hd, sh + hd, (size_t)hd * sizeof(float));
            memcpy(dv + h * hd, sh + 2 * hd, (size_t)hd * sizeof(float));
        }
    }
}

/* Fused projection straight into three contiguous planes. The GPU maps the
 * per-head weight rows in the kernel, so a 768p step does not pay for a
 * second pass over a multi-gigabyte qkv output. */
static void h3v_gemm_qkv(float *q, float *k, float *v, const float *x,
                         const h3v_mat *w, int64_t n, int64_t heads, int64_t hd) {
    const int64_t cols = heads * hd;
    const int use_qkv3 = !(getenv("CWICK_H3V_QKV3") &&
                           strcmp(getenv("CWICK_H3V_QKV3"), "0") == 0);
    if (!w->f32 && n >= 32 && use_qkv3 &&
        (int64_t)n * (int64_t)w->cols * w->rows >= (int64_t)8 * 1024 * 1024 &&
        h3v_gpu_on()) {
        if (h3v_gpu_gemm_qkv3(x, w->p, q, k, v, (int)n, (int)w->cols,
                              (int)cols, (int)w->rows, (int)w->cols,
                              (int)cols, (int)hd)) return;
    }
    if (!w->f32 && n >= 32 && h3v_gpu_on()) {
        int ok = 1;
        float *planes[3] = {q, k, v};
        for (int p = 0; p < 3; p++) {
            if (!h3v_gpu_gemm_qkv(x, w->p, planes[p], (int)n, (int)w->cols,
                                  (int)cols, (int)w->rows, (int)w->cols,
                                  (int)cols, p, (int)hd)) { ok = 0; break; }
        }
        if (ok) return;
    }
    float *raw = h3v_f32((size_t)n * (size_t)w->rows);
    h3v_gemm_s(raw, x, w->cols, w, n);
    h3v_split_qkv(q, k, v, raw, n, heads, hd);
    free(raw);
}

/* Gated SwiGLU over a 2*F-wide row: the release layout emits [gate | value],
 * and fc2 consumes the value half in place. */
static void h3v_swiglu(float *x, int64_t n, int64_t half) {
    const bool legacy = h3v_legacy_layout() != 0;
    for (int64_t i = 0; i < n; i++) {
        float g = legacy ? x[half + i] : x[i];
        float v = legacy ? x[i] : x[half + i];
        x[i] = v * (g / (1.0f + expf(-g)));
    }
}

/* ---- model ---------------------------------------------------------- */
struct h3v_model {
    char *root;
    char *variant;              /* default variant name, e.g. "FL2VA" */
    char **variants;            /* available variant names found under root */
    char **variant_dits;        /* parallel transformer dirs for each variant */
    int nvariants;
    char *te_dir;
    char *tok_dir;
    char *dit_dir;              /* default transformer shards, opened per generation */
    cwick_safetensors dit;
    cwick_safetensors vae;
    cwick_safetensors avae;
    int dit_open, vae_open, avae_open;
    const void *vae_map;
    size_t vae_len;
    /* Flow-matching noise-scale shift, read from the checkpoint's
     * model_index.json `_minimax_h3.sigma_shift_scales` (video 12.0, audio
     * 3.0). Video previously used one hardcoded 12.0 for both streams. */
    float video_shift, audio_shift;
    pthread_mutex_t lock;
    bool ready;
};

static char *h3v_join(const char *a, const char *b) {
    size_t n = strlen(a) + strlen(b) + 2;
    char *s = (char *)cwick_xmalloc(n);
    cwick_path_join(s, n, a, b);
    return s;
}

/* Decide, from this machine's RAM and the memory policy, whether a mapped
 * weight stage should be faulted in and wired or left as an SSD-backed lazy
 * mapping. A wired stage cannot be compressed or paged, which is what the
 * 30-step sampling loop needs: it re-reads the whole transformer every step,
 * and a compressed page costs a fault inside waitUntilCompleted.
 *
 * The wire default is on when the whole stage fits under the ceiling, and
 * off when it does not (never wire an over-budget stage — that only pushes
 * something else into the compressor). `CWICK_H3V_WIRE=0/1` overrides either
 * way so the tradeoff can be measured without a rebuild. */
static void h3v_apply_residency(cwick_safetensors *saf, const char *name) {
    if (!saf || !saf->map_count) return;
    cwick_model_fit fit;
    cwick_mem_fit_model(saf->total_bytes, 0, 0, &fit);
    const char *ov = getenv("CWICK_H3V_WIRE");
    int wire = fit.wire ? 1 : 0;
    if (ov && *ov) wire = atoi(ov) ? 1 : 0;
    uint64_t got = cwick_safetensors_make_resident(saf, NULL, wire);
    cwick_metrics_set_model_residency(saf->total_bytes, got, fit.target_bytes, wire);
    CWICK_INFO("mem", "video %s: %.2f GiB weights, %.2f GiB ceiling (%.0f%% "
               "of %.2f GiB host) — %s %.2f GiB (%s); compressed/paged: %s",
               name, cwick_gib(saf->total_bytes), cwick_gib(fit.target_bytes),
               fit.target_bytes && fit.host_bytes
                   ? (double)fit.target_bytes / (double)fit.host_bytes * 100.0
                   : 0.0,
               cwick_gib(fit.host_bytes),
               wire ? "wire resident" : "stream lazily",
               cwick_gib(got),
               fit.fits ? "fits under budget" : "over budget, SSD-streamed",
               cwick_memory_noswap() ? "no" : "allowed (--allow-swap)");
}

/* Read the checkpoint's per-stream noise-scale shift. The shipped
 * model_index.json declares video=12.0 and audio=3.0 under
 * `_minimax_h3.sigma_shift_scales`; using one value for both streams changes
 * the audio Euler step by 4x. A missing or malformed key falls back to the
 * published defaults, and two env knobs allow an A/B without a rebuild. */
static void h3v_read_shifts(h3v_model *m, const char *base) {
    m->video_shift = 12.0f;
    m->audio_shift = 3.0f;
    char *p = h3v_join(base, "model_index.json");
    char *txt = NULL;
    size_t len = 0;
    if (cwick_read_file(p, &txt, &len) == 0 && txt) {
        cwick_error e;
        cwick_error_clear(&e);
        json_value *root = cwick_json_parse(txt, len, &e);
        if (root) {
            json_value *h3 = cwick_json_get(root, "_minimax_h3");
            json_value *sc = h3 ? cwick_json_get(h3, "sigma_shift_scales") : NULL;
            if (sc && sc->kind == JSON_OBJECT) {
                json_value *v = cwick_json_get(sc, "video");
                json_value *a = cwick_json_get(sc, "audio");
                if (v && v->kind == JSON_NUMBER) m->video_shift = (float)cwick_json_num(v, 12.0);
                if (a && a->kind == JSON_NUMBER) m->audio_shift = (float)cwick_json_num(a, 3.0);
            }
            cwick_json_free(root);
        }
        free(txt);
    }
    free(p);
    const char *ev = getenv("CWICK_H3V_VIDEO_SHIFT");
    const char *ea = getenv("CWICK_H3V_AUDIO_SHIFT");
    if (ev && *ev) {
        double d = atof(ev);
        if (d > 0.0 && d < 1000.0) m->video_shift = (float)d;
    }
    if (ea && *ea) {
        double d = atof(ea);
        if (d > 0.0 && d < 1000.0) m->audio_shift = (float)d;
    }
}

/* The two variants keep the visual VAE in different places: the FL2VA and
 * Ref2VA downloads ship its Python source next to a `source/` subdirectory
 * that holds the weights, while the flattened pipeline has a `vae/` at the
 * top. A directory only counts if a weight file is actually inside it. */
static bool h3v_has_weights(const char *dir) {
    if (!dir) return false;
    static const char *names[3] = {"model.safetensors.index.json",
                                   "diffusion_pytorch_model.safetensors.index.json",
                                   "model.safetensors"};
    for (int i = 0; i < 3; i++) {
        char *p = h3v_join(dir, names[i]);
        struct stat sb;
        bool ok = (stat(p, &sb) == 0 && S_ISREG(sb.st_mode));
        free(p);
        if (ok) return true;
    }
    return false;
}

static char *h3v_pick_dir(const char *base, const char *root, const char *rel,
                          const char *rel_src, const char *top) {
    char *cand = rel ? h3v_join(base, rel) : NULL;
    if (cand && h3v_has_weights(cand)) return cand;
    free(cand);
    cand = rel_src ? h3v_join(base, rel_src) : NULL;
    if (cand && h3v_has_weights(cand)) return cand;
    free(cand);
    cand = h3v_join(root, top);
    if (h3v_has_weights(cand)) return cand;
    free(cand);
    return NULL;
}

/* A variant is usable only when its transformer/ actually holds shards. The
 * FL2VA and Ref2VA downloads ship a full pipeline each, but a partial copy
 * (transformer missing) must not be advertised as selectable. */
static bool h3v_variant_usable(const char *root, const char *v) {
    char *base = h3v_join(root, v);
    char *dit = h3v_join(base, "transformer");
    bool ok = h3v_has_weights(dit);
    free(dit); free(base);
    return ok;
}

/* Case-insensitive variant name match. API callers send "ref2va" or "Ref2VA";
 * the canonical spelling is the directory name on disk. */
bool h3v_variant_name_eq(const char *a, const char *b) {
    if (!a || !b) return false;
    while (*a && *b) {
        char ca = *a, cb = *b;
        if (ca >= 'A' && ca <= 'Z') ca = (char)(ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'Z') cb = (char)(cb - 'A' + 'a');
        if (ca != cb) return false;
        a++; b++;
    }
    return *a == 0 && *b == 0;
}

/* Resolve a caller-supplied variant name against the variants this model
 * actually found. Returns the canonical name, or NULL when unknown. */
const char *h3v_model_resolve_variant(const h3v_model *m, const char *want) {
    if (!m || !want || !*want) return m && m->nvariants ? m->variants[0] : NULL;
    for (int i = 0; i < m->nvariants; i++)
        if (h3v_variant_name_eq(m->variants[i], want)) return m->variants[i];
    return NULL;
}

h3v_model *h3v_model_open(const char *root, cwick_error *err) {
    if (!root || !*root) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "video model root is required");
        return NULL;
    }
    /* The /v1/videos path is the omni-reference workflow: it takes a prompt
     * plus optional reference images and must use the Ref2VA transformer.
     * Only that partition is discovered or loaded here. A first/last-frame
     * transformer lives in older downloads but is not a video endpoint
     * variant, so it is never probed, offered or mapped. A flattened copy of
     * the reference pipeline is accepted as well. The decoders and text
     * encoder live at the root and stay mapped once for the process. */
    static const char *candidates[1] = {"Ref2VA"};
    char *vlist[1] = {NULL};
    char *dlist[1] = {NULL};
    int nv = 0;
    for (int i = 0; i < 1; i++) {
        if (!h3v_variant_usable(root, candidates[i])) continue;
        char *base = h3v_join(root, candidates[i]);
        vlist[nv] = cwick_strdup(candidates[i]);
        dlist[nv] = h3v_join(base, "transformer");
        nv++;
        free(base);
    }
    /* Flattened copy: files live directly under root. */
    if (nv == 0) {
        char *dit0 = h3v_join(root, "transformer");
        if (h3v_has_weights(dit0)) {
            vlist[0] = cwick_strdup("FLAT");
            dlist[0] = dit0;
            nv = 1;
        } else {
            free(dit0);
        }
    }
    if (nv == 0) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "%s has no usable transformer/ (expected a MiniMax-H3 "
                        "Ref2VA download)", root);
        return NULL;
    }

    /* Ref2VA is the only video variant; the environment override is kept for
     * a flat checkpoint that reports itself as "FLAT" and nothing else. */
    int def = 0;
    const char *ev = getenv("CWICK_H3V_VARIANT");
    if (ev && *ev) {
        for (int i = 0; i < nv; i++)
            if (h3v_variant_name_eq(vlist[i], ev)) { def = i; break; }
    }

    const char *base_v = vlist[def];
    char *base = h3v_join(root, base_v);
    if (h3v_variant_name_eq(base_v, "FLAT")) { free(base); base = cwick_strdup(root); }

    h3v_model *m = (h3v_model *)cwick_xmalloc(sizeof(*m));
    memset(m, 0, sizeof(*m));
    m->root = cwick_strdup(root);
    m->variants = (char **)cwick_xmalloc(sizeof(char *) * (size_t)(nv + 1));
    m->variant_dits = (char **)cwick_xmalloc(sizeof(char *) * (size_t)(nv + 1));
    for (int i = 0; i < nv; i++) {
        m->variants[i] = vlist[i];
        m->variant_dits[i] = dlist[i];
        vlist[i] = NULL; dlist[i] = NULL;
    }
    m->variants[nv] = NULL;
    m->variant_dits[nv] = NULL;
    m->nvariants = nv;
    m->variant = cwick_strdup(base_v);
    m->dit_dir = cwick_strdup(m->variant_dits[def]);
    pthread_mutex_init(&m->lock, NULL);
    h3v_read_shifts(m, base);

    char *vae = h3v_pick_dir(base, root, "video_vae", "video_vae/source", "vae");
    char *avae = h3v_pick_dir(base, root, "audio_vae", NULL, "audio_vae");
    m->te_dir = h3v_join(base, "text_encoder");
    m->tok_dir = h3v_join(base, "tokenizer");
    if (!vae || !avae) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "no VAE weights under %s: looked in %s/video_vae, "
                        "video_vae/source and vae/", base, base);
        free(vae); free(avae); free(base);
        h3v_model_close(m);
        return NULL;
    }
    /* The transformer is 62 GiB and the H3-Encoder is another 62 GiB. They
     * cannot be resident at the same time on a 128 GiB machine, so the
     * transformer is opened for the sampling loop and closed again before the
     * next request's text pass. The two VAEs stay open: together they are
     * 10.3 GiB and re-mapping them per request would cost more than the decode. */
    if (cwick_safetensors_open(vae, &m->vae, err) != CWICK_OK) goto fail;
    m->vae_open = 1;
    if (cwick_safetensors_open(avae, &m->avae, err) != CWICK_OK) goto fail;
    m->avae_open = 1;
    /* The decoders stay resident for the process lifetime, so fault them in
     * and wire them once here (unless the policy says the machine cannot
     * afford it). Unwiring happens in safetensors_close on model switch. */
    h3v_apply_residency(&m->vae, "video VAE");
    h3v_apply_residency(&m->avae, "audio VAE");
    /* The VAEs stay mapped for the process, so their GPU views are adopted
     * once here rather than on each decode. */
    if (h3v_gpu_on()) {
        for (size_t i = 0; i < m->vae.map_count; i++)
            h3v_gpu_adopt(m->vae.maps[i], m->vae.map_lens[i]);
    }

    free(vae); free(avae); free(base);
    m->ready = true;
    CWICK_INFO("init", "video model ready: %s (default variant %s, %d available)",
               root, m->variant, nv);
    return m;

fail:
    free(vae); free(avae); free(base);
    h3v_model_close(m);
    return NULL;
}

void h3v_model_close(h3v_model *m) {
    if (!m) return;
    if (m->dit_open) cwick_safetensors_close(&m->dit);
    if (m->vae_open) cwick_safetensors_close(&m->vae);
    if (m->avae_open) cwick_safetensors_close(&m->avae);
    pthread_mutex_destroy(&m->lock);
    free(m->root); free(m->variant); free(m->te_dir); free(m->tok_dir);
    free(m->dit_dir);
    if (m->variants) {
        for (int i = 0; i < m->nvariants; i++) {
            free(m->variants[i]);
            free(m->variant_dits[i]);
        }
    }
    free(m->variants); free(m->variant_dits);
    free(m);
}

const char *h3v_model_dir(const h3v_model *m) { return m ? m->root : NULL; }
bool h3v_model_ready(const h3v_model *m) { return m && m->ready; }

/* The decoders live in their own translation units and only need the shard
 * handle, so the handle crosses the boundary instead of the whole model. */
const cwick_safetensors *h3v_shards_vae(const h3v_model *m) {
    return m && m->vae_open ? &m->vae : NULL;
}
const cwick_safetensors *h3v_shards_avae(const h3v_model *m) {
    return m && m->avae_open ? &m->avae : NULL;
}
const char *h3v_model_variant(const h3v_model *m) {
    return m && m->variant ? m->variant : "";
}
int h3v_model_variant_count(const h3v_model *m) { return m ? m->nvariants : 0; }
const char *h3v_model_variant_name(const h3v_model *m, int i) {
    return m && i >= 0 && i < m->nvariants ? m->variants[i] : NULL;
}
void h3v_mp4_free(unsigned char *p) { free(p); }

int64_t h3v_frames_for_seconds(int seconds) {
    int s = seconds < H3V_MIN_SECONDS ? H3V_MIN_SECONDS
          : (seconds > H3V_MAX_SECONDS ? H3V_MAX_SECONDS : seconds);
    /* 17n+5 is the frame-count rule the checkpoint trains on; a request is
     * rounded up to whole 24 fps frames and then to the next legal count. */
    int64_t want = (int64_t)s * H3V_FPS;
    int64_t clips = (want + H3V_CLIP_FRAMES - 1) / H3V_CLIP_FRAMES;
    if (clips < 1) clips = 1;
    return clips * H3V_CLIP_FRAMES;
}

/* ---- 3-axis rotary tables for the DiT ------------------------------- */
static void h3v_rope_tables(int64_t ntext, int64_t nvideo, int64_t naudio,
                            int64_t nref, int64_t lt, int64_t lh, int64_t lw,
                            int64_t rlh, int64_t rlw,
                            float *cos, float *sin) {
    const int64_t F = VD_ROPE_FREQ, RD = VD_ROPE_DIM;
    int64_t seq = ntext + nref + naudio + nvideo;
    /* One image reference occupies one integer slot on the shared rotary
     * clock; the generated audio and video start where it ends. */
    const int64_t origin = ntext + (nref > 0 ? 1 : 0);
    float inv[VD_ROPE_FREQ];
    for (int64_t j = 0; j < F; j++)
        inv[j] = powf(1.0f / 10000.0f, (float)(2 * j) / (float)(2 * F));

    float *pos = h3v_f32(seq * 3);
    /* How far the rotary clock advances for one latent frame. The VAE folds
     * 17 pixel frames into 5 latent ones, so a clip's first latent frame only
     * covers one pixel frame while the rest cover four: the pattern
     * (1,4,4,4,4) scaled by 5/3 units per pixel frame, or 6.667 units for a
     * full latent. A uniform one-unit-per-frame clock compresses a 4 s clip
     * onto 1/7 of the range the checkpoint was trained on, which leaves the
     * model unable to tell how far apart two frames sit and turns every
     * generated frame into an independent sample. CWICK_H3V_POS=uniform keeps
     * the old linear clock for A/B. */
    static const int fpl[5] = {1, 4, 4, 4, 4};
    const float fscale = 5.0f / 3.0f;
    const char *ep = getenv("CWICK_H3V_POS");
    const int uniform = ep && *ep && strcmp(ep, "uniform") == 0;
    float *tgrid = h3v_f32(lt > 0 ? lt : 1);
    {
        float acc = 0.0f;
        for (int64_t t = 0; t < lt; t++) {
            tgrid[t] = uniform ? (float)t : acc;
            acc += fscale * (float)fpl[t % 5];
        }
    }
    /* Text rows sit on the time axis only. */
    for (int64_t s = 0; s < ntext; s++) {
        pos[s * 3 + 0] = (float)s;
        pos[s * 3 + 1] = 0.0f;
        pos[s * 3 + 2] = 0.0f;
    }
    /* Conditioning keyframe rows carry the first frame's spatial grid but a
     * time coordinate pinned to where the text rows end. */
    if (nref > 0) {
        const int64_t gh0 = rlh / VD_PATCH, gw0 = rlw / VD_PATCH;
        const float a0 = sqrtf((float)rlh * (float)rlw);
        const float rh = (float)rlh / a0, rw = (float)rlw / a0;
        const float lh0 = (1.0f - rh) * 0.5f, lw0 = (1.0f - rw) * 0.5f;
        for (int64_t r = 0; r < nref; r++) {
            const int64_t y = r / gw0, x = r % gw0;
            pos[(ntext + r) * 3 + 0] = (float)ntext;
            pos[(ntext + r) * 3 + 1] = (lh0 + (float)y * rh / (float)gh0) * 32.0f;
            pos[(ntext + r) * 3 + 2] = (lw0 + (float)x * rw / (float)gw0) * 32.0f;
        }
    }
    /* Audio rows share the video clock: one rotary unit per audio latent at
     * 40 latents/s, which is exactly 24 fps * 5/3. Rows are channel-major so
     * the time coordinate repeats once per channel. */
    if (naudio > 0) {
        const int64_t achan = naudio >= 2 ? 2 : 1;
        const int64_t alut = naudio / achan;
        const float geo_sqrt = sqrtf((float)lh * (float)lw);
        const float wratio = (float)lw / geo_sqrt;
        const float wleft = (1.0f - wratio) * 0.5f;
        const float w0 = wleft * 32.0f;
        const float w1 = (wleft + wratio * (1.0f - 1.0f / (float)(lw / VD_PATCH))) * 32.0f;
        for (int64_t s = 0; s < naudio; s++) {
            const int64_t c = s / alut, u = s % alut;
            pos[(ntext + nref + s) * 3 + 0] = (float)origin + (float)u;
            pos[(ntext + nref + s) * 3 + 1] = 0.0f;
            pos[(ntext + nref + s) * 3 + 2] = c == 0 ? w0 : w1;
        }
    }
    /* Video rows: (t, h, w). The spatial axes are aspect-normalised the way
     * the reference pipeline builds them: the longer side spans [0, 32) and
     * the shorter one is centred inside that range, one coordinate per patch,
     * right endpoint excluded. */
    {
        const int64_t gh = lh / VD_PATCH, gw = lw / VD_PATCH;
        const int64_t base = ntext + nref + naudio;
        const float sqrt_area = sqrtf((float)lh * (float)lw);
        const float ratio_h = (float)lh / sqrt_area, ratio_w = (float)lw / sqrt_area;
        const float left_h = (1.0f - ratio_h) * 0.5f, left_w = (1.0f - ratio_w) * 0.5f;
        const float step_h = ratio_h / (float)gh, step_w = ratio_w / (float)gw;
        for (int64_t t = 0; t < lt; t++)
            for (int64_t y = 0; y < gh; y++)
                for (int64_t x = 0; x < gw; x++) {
                    int64_t r = base + t * gh * gw + y * gw + x;
                    pos[r * 3 + 0] = (float)origin + tgrid[t];
                    pos[r * 3 + 1] = (left_h + (float)y * step_h) * 32.0f;
                    pos[r * 3 + 2] = (left_w + (float)x * step_w) * 32.0f;
                }
    }
    free(tgrid);
    for (int64_t s = 0; s < seq; s++) {
        for (int64_t j = 0; j < F; j++) {
            cos[s * RD + j] = cosf(pos[s * 3 + 0] * inv[j]);
            cos[s * RD + F + j] = cosf(pos[s * 3 + 1] * inv[j]);
            cos[s * RD + 2 * F + j] = cosf(pos[s * 3 + 2] * inv[j]);
            sin[s * RD + j] = sinf(pos[s * 3 + 0] * inv[j]);
            sin[s * RD + F + j] = sinf(pos[s * 3 + 1] * inv[j]);
            sin[s * RD + 2 * F + j] = sinf(pos[s * 3 + 2] * inv[j]);
        }
    }
    free(pos);
}

/* Rotary application over the 96 of 128 head dims that the 3-axis table
 * covers. The table carries 48 distinct angles (16 per axis, one per pair),
 * so there are only two ways to consume them:
 *   pair k = (2k, 2k+1)  — the interleaved convention the 3-axis video
 *                          transformers use, dims 0..31 time, 32..63 height,
 *                          64..95 width;
 *   pair k = (k, k+48)   — the half-split convention Flux/SD3 use, which needs
 *                          each angle duplicated twice in the table.
 * This table has no duplicates, so the second reading rotates unrelated
 * channels against each other and destroys the positional signal while
 * leaving every magnitude intact. CWICK_H3V_ROPE=half selects it. */
static void h3v_rope_apply(float *x, int64_t seq, int64_t heads, int64_t head_dim,
                           const float *cos, const float *sin, int interleaved) {
    const int64_t RD = VD_ROPE_DIM, half = RD / 2;
    for (int64_t s = 0; s < seq; s++)
        for (int64_t h = 0; h < heads; h++) {
            float *v = x + (s * heads + h) * head_dim;
            const float *c = cos + s * RD, *n = sin + s * RD;
            if (interleaved) {
                for (int64_t k = 0; k < RD; k++) {
                    float a = v[2 * k], b = v[2 * k + 1];
                    v[2 * k] = a * c[k] - b * n[k];
                    v[2 * k + 1] = a * n[k] + b * c[k];
                }
            } else {
                for (int64_t i = 0; i < half; i++) {
                    float a = v[i], b = v[i + half];
                    v[i] = a * c[i] - b * n[i];
                    v[i + half] = a * n[i] + b * c[i];
                }
            }
        }
}

/* Full self-attention over `seq` rows with Q, K and V in separate buffers.
 * `stride` is the row pitch of all three (heads * head_dim for the packed
 * layout this file uses). A query tile at a time keeps the score matrix at
 * 128 x seq floats instead of seq^2. */
static void h3v_attention(float *out, const float *q, const float *k,
                          const float *v, int64_t seq, int64_t heads,
                          int64_t hd, int64_t stride) {
    const int64_t TILE = 128;
    const float scale = 1.0f / sqrtf((float)hd);
    /* The joint self-attention is non-causal and the packed rows already put
     * each head at h*hd, which is the kernel's expected layout. */
    if (h3v_gpu_on() &&
        h3v_gpu_attention(q, k, v, out, (int)seq, (int)stride, (int)heads, (int)hd))
        return;
    for (int64_t q0 = 0; q0 < seq; q0 += TILE) {
        int64_t qn = seq - q0 < TILE ? seq - q0 : TILE;
        float *sc = h3v_f32(qn * seq);
        for (int64_t h = 0; h < heads; h++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)qn, (int)seq,
                        (int)hd, scale, q + q0 * stride + h * hd, (int)stride,
                        k + h * hd, (int)stride, 0.0f, sc, (int)seq);
            for (int64_t i = 0; i < qn; i++) {
                float *row = sc + i * seq;
                float mx = row[0];
                for (int64_t j = 1; j < seq; j++) if (row[j] > mx) mx = row[j];
                float sum = 0.0f;
                for (int64_t j = 0; j < seq; j++) {
                    row[j] = expf(row[j] - mx);
                    sum += row[j];
                }
                float inv = 1.0f / sum;
                for (int64_t j = 0; j < seq; j++) row[j] *= inv;
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)qn, (int)hd,
                        (int)seq, 1.0f, sc, (int)seq, v + h * hd, (int)stride,
                        0.0f, out + q0 * stride + h * hd, (int)stride);
        }
        free(sc);
    }
}

/* qk RMSNorm over every head, then 3-axis rotary. In place.
 * The table holds 48 angles (16 per axis) and rotate_half pairs channel i with
 * channel i+48, exactly as the image line does: there the table is stored
 * duplicated so both halves of a pair read the same angle, which is what
 * indexing the same entry twice reproduces here. CWICK_H3V_ROPE=int tests the
 * interleaved pairing (2k, 2k+1) instead. */
static int h3v_rope_mode(void) {
    static int mode = -1;
    if (mode < 0) {
        const char *em = getenv("CWICK_H3V_ROPE");
        mode = (em && *em && strcmp(em, "int") == 0) ? 1 : 0;
    }
    return mode;
}

/* Per-head RMSNorm, optionally followed by the 3-axis rotary. The refiner
 * blocks run the norm but no rotary: the checkpoint carries no rotary state
 * for the text stream, and rotating it there shifts the prompt rows against a
 * table the stack never saw them in. */
static void h3v_qk_prep(float *x, int64_t seq, const float *wq,
                        const float *cos, const float *sin, int rope_on) {
    for (int64_t r = 0; r < seq; r++)
        for (int64_t h = 0; h < VD_HEADS; h++)
            h3v_rmsnorm(x + r * VD_QKV + h * VD_HEAD_DIM,
                        x + r * VD_QKV + h * VD_HEAD_DIM, wq, VD_HEAD_DIM, 1e-5f);
    if (rope_on)
        h3v_rope_apply(x, seq, VD_HEADS, VD_HEAD_DIM, cos, sin, h3v_rope_mode());
}

/* The 5120 -> 5376 text projection is followed by two refiner blocks that run
 * on the prompt rows alone, before those rows join the audio and video ones.
 * They are ordinary pre-norm blocks: no adaln (the checkpoint carries no
 * modulation weights for them), and no output gate. */
static bool h3v_refine(h3v_model *m, float *x, int64_t ntext, const float *cos,
                       const float *sin, cwick_error *err) {
    const int64_t D = VD_HIDDEN, QKV = VD_QKV, F = VD_FFN;
    float *tmp = h3v_f32(ntext * D);
    float *qkv = h3v_legacy_layout() ? h3v_f32(ntext * 3 * QKV) : NULL;
    float *q = h3v_f32(ntext * QKV), *k = h3v_f32(ntext * QKV);
    float *v = h3v_f32(ntext * QKV), *att = h3v_f32(ntext * QKV);
    float *o = h3v_f32(ntext * D);
    float *hid = h3v_f32(ntext * 2 * F);
    bool ok = false;
    char name[192];

    for (int64_t L = 0; L < 2; L++) {
        h3v_mat qw, ow, qn, kn, f1, f2, n1, n2;
        const char *suf = "token_refiner.blocks.%lld.";
        snprintf(name, sizeof(name), "token_refiner.blocks.%lld.attn.qkv_proj.weight", (long long)L);
        if (!h3v_mat_get(&m->dit, name, &qw, err)) goto done;
        snprintf(name, sizeof(name), "token_refiner.blocks.%lld.attn.out_proj.weight", (long long)L);
        if (!h3v_mat_get(&m->dit, name, &ow, err)) goto done;
        snprintf(name, sizeof(name), "token_refiner.blocks.%lld.attn.q_norm.weight", (long long)L);
        if (!h3v_mat_get(&m->dit, name, &qn, err)) goto done;
        snprintf(name, sizeof(name), "token_refiner.blocks.%lld.attn.k_norm.weight", (long long)L);
        if (!h3v_mat_get(&m->dit, name, &kn, err)) goto done;
        snprintf(name, sizeof(name), "token_refiner.blocks.%lld.mlp.fc1.weight", (long long)L);
        if (!h3v_mat_get(&m->dit, name, &f1, err)) goto done;
        snprintf(name, sizeof(name), "token_refiner.blocks.%lld.mlp.fc2.weight", (long long)L);
        if (!h3v_mat_get(&m->dit, name, &f2, err)) goto done;
        snprintf(name, sizeof(name), "token_refiner.blocks.%lld.norm1.weight", (long long)L);
        if (!h3v_mat_get(&m->dit, name, &n1, err)) goto done;
        snprintf(name, sizeof(name), "token_refiner.blocks.%lld.norm2.weight", (long long)L);
        if (!h3v_mat_get(&m->dit, name, &n2, err)) goto done;
        (void)suf;

        float *w1 = h3v_vec(&n1), *w2 = h3v_vec(&n2);
        float *wq = h3v_vec(&qn), *wk = h3v_vec(&kn);

        for (int64_t r = 0; r < ntext; r++)
            h3v_rmsnorm(tmp + r * D, x + r * D, w1, D, 1e-5f);
        if (h3v_legacy_layout()) {
            h3v_gemm(qkv, tmp, &qw, ntext);
            for (int64_t r = 0; r < ntext; r++) {
                const float *src = qkv + r * 3 * QKV;
                memcpy(q + r * QKV, src, (size_t)QKV * sizeof(float));
                memcpy(k + r * QKV, src + QKV, (size_t)QKV * sizeof(float));
                memcpy(v + r * QKV, src + 2 * QKV, (size_t)QKV * sizeof(float));
            }
        } else {
            h3v_gemm_qkv(q, k, v, tmp, &qw, ntext, VD_HEADS, VD_HEAD_DIM);
        }
        h3v_qk_prep(q, ntext, wq, cos, sin, 0);
        h3v_qk_prep(k, ntext, wk, cos, sin, 0);
        h3v_attention(att, q, k, v, ntext, VD_HEADS, VD_HEAD_DIM, QKV);
        h3v_gemm(o, att, &ow, ntext);
        for (int64_t i = 0; i < ntext * D; i++) x[i] += o[i];

        for (int64_t r = 0; r < ntext; r++)
            h3v_rmsnorm(tmp + r * D, x + r * D, w2, D, 1e-5f);
        h3v_gemm(hid, tmp, &f1, ntext);
        for (int64_t r = 0; r < ntext; r++)
            h3v_swiglu(hid + r * 2 * F, F, F);
        h3v_gemm_s(o, hid, 2 * F, &f2, ntext);
        for (int64_t i = 0; i < ntext * D; i++) x[i] += o[i];

        free(w1); free(w2); free(wq); free(wk);
    }
    {
        h3v_mat fn;
        if (!h3v_mat_get(&m->dit, "token_refiner.final_norm.weight", &fn, err))
            goto done;
        float *w = h3v_vec(&fn);
        for (int64_t r = 0; r < ntext; r++)
            h3v_rmsnorm(x + r * D, x + r * D, w, D, 1e-5f);
        free(w);
    }
    ok = true;

done:
    free(tmp); free(qkv); free(q); free(k); free(v); free(att); free(o);
    free(hid);
    return ok;
}


/* ---- 3D latent packing: [C][T][H][W] -> [T*H/2*W/2][C*2*2] ---------- */
static void h3v_pack3d(float *dst, const float *lat, int64_t lt, int64_t lh,
                       int64_t lw) {
    const int64_t pw = lw / VD_PATCH, ph = lh / VD_PATCH;
    const int64_t per_frame = ph * pw;
    for (int64_t t = 0; t < lt; t++) {
        for (int64_t r = 0; r < per_frame; r++) {
            const int64_t py = r / pw, px = r % pw;
            int64_t row = t * per_frame + r;
            for (int64_t c = 0; c < VD_LATENT; c++) {
                float *d = dst + row * VD_PATCH_DIM + c * 4;
                for (int64_t dy = 0; dy < VD_PATCH; dy++)
                    for (int64_t dx = 0; dx < VD_PATCH; dx++)
                        d[dy * VD_PATCH + dx] =
                            lat[((c * lt + t) * lh + py * VD_PATCH + dy) * lw
                                + px * VD_PATCH + dx];
            }
        }
    }
}

/* Inverse of h3v_pack3d, used after the final layer. */
static void h3v_unpack3d(float *lat, const float *rows, int64_t lt, int64_t lh,
                         int64_t lw) {
    const int64_t pw = lw / VD_PATCH, ph = lh / VD_PATCH;
    const int64_t per_frame = ph * pw;
    for (int64_t t = 0; t < lt; t++)
        for (int64_t r = 0; r < per_frame; r++) {
            const int64_t py = r / pw, px = r % pw;
            int64_t row = t * per_frame + r;
            for (int64_t c = 0; c < VD_LATENT; c++) {
                const float *s = rows + row * VD_PATCH_DIM + c * 4;
                for (int64_t dy = 0; dy < VD_PATCH; dy++)
                    for (int64_t dx = 0; dx < VD_PATCH; dx++)
                        lat[((c * lt + t) * lh + py * VD_PATCH + dy) * lw
                            + px * VD_PATCH + dx] = s[dy * VD_PATCH + dx];
            }
        }
}

/* ---- one DiT step ----------------------------------------------------
 * The packed sequence is [text | audio | video]: text rows carry the prompt,
 * audio rows carry the 32-channel audio latent and video rows carry the 96-wide
 * video patch. One joint self-attention sees all three so the soundtrack and
 * the picture stay in sync; the final layer splits the velocity back apart.
 */
/* 256-wide sinusoid -> linear -> silu -> linear. The sampler hands this a
 * timestep in [0,1] and cos leads sin (flip_sin_to_cos). */
static void h3v_temb(h3v_model *m, float t, float *out, cwick_error *err) {
    h3v_mat tin, tin_b, tout, tout_b;
    if (!h3v_mat_get(&m->dit, "time_embedder.proj_in.weight", &tin, err)) return;
    h3v_mat_get(&m->dit, "time_embedder.proj_in.bias", &tin_b, err);
    if (!h3v_mat_get(&m->dit, "time_embedder.proj_out.weight", &tout, err)) return;
    h3v_mat_get(&m->dit, "time_embedder.proj_out.bias", &tout_b, err);
    float tb[256];
    for (int64_t i = 0; i < 128; i++) {
        float freq = powf(1.0f / 10000.0f, (float)i / 128.0f);
        float v = t * freq;
        tb[i] = cosf(v);
        tb[128 + i] = sinf(v);
    }
    float *hid = h3v_f32(VD_TIME_HIDDEN);
    h3v_gemm(hid, tb, &tin, 1);
    if (tin_b.p) {
        float *b = h3v_vec(&tin_b);
        h3v_add_bias(hid, b, VD_TIME_HIDDEN, 1);
        free(b);
    }
    h3v_silu_inplace(hid, VD_TIME_HIDDEN);
    h3v_gemm(out, hid, &tout, 1);
    if (tout_b.p) {
        float *b = h3v_vec(&tout_b);
        h3v_add_bias(out, b, VD_TIME_DIM, 1);
        free(b);
    }
    free(hid);
}

static bool h3v_adaln_precompute(h3v_model *m, const float *ts, const float *ats,
                                int steps, bool has_audio, bool has_condition,
                                float **out_mod, float **out_final,
                                cwick_error *err) {
    const int nts = 1 + (has_audio ? 1 : 0) + (has_condition ? 1 : 0);
    const int64_t rows = (int64_t)steps * nts;
    float *tin = h3v_f32(rows * VD_TIME_DIM);
    float *mod = h3v_f32((int64_t)VD_LAYERS * rows * VD_ADALN_ROWS);
    float *fmod = h3v_f32(rows * VD_FINAL_ADALN);
    if (!tin || !mod || !fmod) goto fail;
    for (int s = 0; s < steps; s++) {
        float tv[3] = {ts[s], has_audio ? ats[s] : 0.0f,
                       has_condition ? 0.999f : 0.0f};
        for (int ti = 0; ti < nts; ti++) {
            float *temb = h3v_f32(VD_TIME_DIM);
            float *dst = tin + ((int64_t)s * nts + ti) * VD_TIME_DIM;
            if (!temb) goto fail;
            h3v_temb(m, tv[ti], temb, err);
            if (err->message[0]) { free(temb); goto fail; }
            memcpy(dst, temb, (size_t)VD_TIME_DIM * sizeof(float));
            h3v_silu_inplace(dst, VD_TIME_DIM);
            free(temb);
        }
    }
    for (int64_t L = 0; L < VD_LAYERS; L++) {
        char name[192];
        h3v_mat adw, adb;
        snprintf(name, sizeof(name), "blocks.%lld.adaln_proj.linear.weight",
                 (long long)L);
        if (!h3v_mat_get(&m->dit, name, &adw, err)) goto fail;
        snprintf(name, sizeof(name), "blocks.%lld.adaln_proj.linear.bias",
                 (long long)L);
        if (!h3v_mat_get(&m->dit, name, &adb, err)) goto fail;
        float *am = h3v_f32(rows * VD_ADALN_ROWS);
        float *bias = h3v_vec(&adb);
        if (!am || !bias) { free(am); free(bias); goto fail; }
        h3v_gemm_cpu(am, tin, VD_TIME_DIM, &adw, rows);
        h3v_add_bias(am, bias, VD_ADALN_ROWS, rows);
        memcpy(mod + L * rows * VD_ADALN_ROWS, am,
               (size_t)rows * VD_ADALN_ROWS * sizeof(float));
        free(am); free(bias);
    }
    {
        h3v_mat adw, adb;
        if (!h3v_mat_get(&m->dit, "final_layer.adaln_proj.linear.weight",
                         &adw, err) ||
            !h3v_mat_get(&m->dit, "final_layer.adaln_proj.linear.bias",
                         &adb, err)) goto fail;
        float *bias = h3v_vec(&adb);
        if (!bias) goto fail;
        h3v_gemm_cpu(fmod, tin, VD_TIME_DIM, &adw, rows);
        h3v_add_bias(fmod, bias, VD_FINAL_ADALN, rows);
        free(bias);
    }
    free(tin);
    *out_mod = mod;
    *out_final = fmod;
    return true;
fail:
    free(tin); free(mod); free(fmod);
    return false;
}

static bool h3v_dit_step(h3v_model *m, const float *cond, int64_t ntext,
                         const float *ttags,
                         const float *vrows, int64_t nvideo,
                         const float *crows, int64_t nref,
                         const float *arows, int64_t naudio,
                         const float *mod_all, const float *final_mod_all,
                         const float *cos, const float *sin,
                         int64_t step, int steps,
                         h3v_progress_fn cb, void *cbctx,
                         float *vout, float *aout, cwick_error *err) {
    const int64_t D = VD_HIDDEN, H = VD_HEADS, HD = VD_HEAD_DIM;
    /* Which adaLN modality row each stream uses. Role order here is text,
     * audio, video; the checkpoint numbers its rows 0 = video, 1 = text,
     * 2 = audio, so the table index is 1/2/0 and a row group is
     * `modality * 6 + parameter`. CWICK_H3V_MODMAP overrides it for A/B only. */
    int mm[3] = {1, 2, 0};
    {
        const char *em = getenv("CWICK_H3V_MODMAP");
        if (em && strlen(em) == 3) {
            for (int i = 0; i < 3; i++) {
                int v = em[i] - '0';
                if (v < 0 || v > 2) { i = 3; break; }
                mm[i] = v;
            }
        }
    }
    const int64_t QKV = VD_QKV, F = VD_FFN;
    const bool has_condition = crows && nref > 0;
    const int nts = 1 + (naudio > 0) + (has_condition ? 1 : 0);
    const int condition_ti = naudio > 0 ? 2 : 1;
    const int64_t nref_use = has_condition ? nref : 0;
    const int64_t seq = ntext + nref_use + naudio + nvideo;
    const int64_t t_video0 = ntext + nref_use + naudio;
    /* Text and generated video use the video schedule, generated audio uses
     * the audio schedule, and conditioning rows stay at their anchor. */
#define H3V_TI(r) ((has_condition && (r) >= ntext && (r) < ntext + nref_use) ? condition_ti : \
                  ((r) >= ntext + nref_use && (r) < t_video0 ? 1 : 0))
/* Vision rows inside the text stream are tagged video by the presentation,
 * so they take the video AdaLN slot even though they sit before the generated
 * rows. A text-only presentation has every tag at 1 and keeps the old map. */
#define H3V_ROLE(r) ((r) < ntext ? (ttags && ttags[r] < 0.5f ? 2 : 0) : \
                     ((r) < ntext + nref_use ? 2 : ((r) < t_video0 ? 1 : 2)))
#define H3V_MODROW(r, p) ((H3V_TI(r) * 3 + mm[H3V_ROLE(r)]) * 6 + (p))

    float *x = h3v_f32(seq * D);
    float *tmp = h3v_f32(seq * D);
    float *qkv = h3v_legacy_layout() ? h3v_f32(seq * 3 * QKV) : NULL;
    float *attn = h3v_f32(seq * QKV);
    float *kbuf = h3v_f32(seq * QKV);
    float *vbuf = h3v_f32(seq * QKV);
    float *prod = h3v_f32(seq * QKV);
    /* fc1 is 2x wide: [gate | value] per row. */
    float *gate = h3v_f32(seq * 2 * F);
    bool ok = false;

    h3v_mat cp, cp_b, vp, vp_b, ap, ap_b;
    if (!h3v_mat_get(&m->dit, "condition_proj.weight", &cp, err)) goto done;
    h3v_mat_get(&m->dit, "condition_proj.bias", &cp_b, err);
    if (!h3v_mat_get(&m->dit, "video_patch_proj.weight", &vp, err)) goto done;
    h3v_mat_get(&m->dit, "video_patch_proj.bias", &vp_b, err);
    if (!h3v_mat_get(&m->dit, "audio_patch_proj.weight", &ap, err)) goto done;
    h3v_mat_get(&m->dit, "audio_patch_proj.bias", &ap_b, err);
    /* Text: project 5120 -> 5376 once, refine the prompt rows on their own,
     * then treat them as the first rows of the joint sequence. */
    if (getenv("CWICK_H3V_DUMPCOND")) {
        const char *dc = getenv("CWICK_H3V_DUMPCOND");
        FILE *f = (*dc) ? fopen(dc, "wb") : NULL;
        if (f) {
            fwrite(cond, sizeof(float), (size_t)ntext * VD_TEXT_DIM, f);
            fclose(f);
            CWICK_INFO("video", "dumped %lldx%d text conditioning to %s",
                       (long long)ntext, VD_TEXT_DIM, dc);
        }
    }
    h3v_gemm(x, cond, &cp, ntext);
    if (cp_b.p) {
        float *b = h3v_vec(&cp_b);
        h3v_add_bias(x, b, D, ntext);
        free(b);
    }
    if (!h3v_refine(m, x, ntext, cos, sin, err)) goto done;
    /* Conditioning keyframe rows: a nearly clean copy of the reference latent,
     * at its own noise level, in front of the target audio/video rows. */
    if (nref_use > 0) {
        h3v_gemm(x + ntext * D, crows, &vp, nref_use);
        if (vp_b.p) {
            float *b = h3v_vec(&vp_b);
            h3v_add_bias(x + ntext * D, b, D, nref_use);
            free(b);
        }
    }
    /* Video patch rows. */
    h3v_gemm(x + t_video0 * D, vrows, &vp, nvideo);
    if (vp_b.p) {
        float *b = h3v_vec(&vp_b);
        h3v_add_bias(x + t_video0 * D, b, D, nvideo);
        free(b);
    }
    /* Audio patch rows sit after the condition rows, before target video. */
    h3v_gemm(x + (ntext + nref_use) * D, arows, &ap, naudio);
    if (ap_b.p) {
        float *b = h3v_vec(&ap_b);
        h3v_add_bias(x + (ntext + nref_use) * D, b, D, naudio);
        free(b);
    }
    /* Per-segment magnitudes. If the text rows end up an order of magnitude
     * smaller than the video rows the joint attention effectively ignores the
     * prompt, which looks exactly like an unconditional model: a velocity field
     * that is large, directionless and identical for every prompt. */
    if (getenv("CWICK_H3V_NORMS")) {
        double ct = 0.0, nt = 0.0, na = 0.0, nv = 0.0;
        for (int64_t i = 0; i < (int64_t)ntext * 5120; i++)
            ct += (double)cond[i] * (double)cond[i];
        for (int64_t i = 0; i < ntext * D; i++)
            nt += (double)x[i] * (double)x[i];
        for (int64_t i = 0; i < naudio * D; i++)
            na += (double)x[(ntext + nref_use) * D + i] * x[(ntext + nref_use) * D + i];
        for (int64_t i = 0; i < nvideo * D; i++)
            nv += (double)x[t_video0 * D + i] * (double)x[t_video0 * D + i];
        CWICK_INFO("video", "NORMS rms/row: cond %.4f, text %.4f, audio %.4f, "
                   "video %.4f (rows text %lld audio %lld video %lld)",
                   sqrt(ct / (double)((int64_t)ntext * 5120)),
                   sqrt(nt / (double)(ntext * D)),
                   sqrt(na / (double)(naudio * D)),
                   sqrt(nv / (double)(nvideo * D)),
                   (long long)ntext, (long long)naudio, (long long)nvideo);
    }

    const int block_stats = getenv("CWICK_H3V_BLOCKSTATS") != NULL;

    char name[192];
    for (int64_t L = 0; L < VD_LAYERS; L++) {
        const float *mod = mod_all +
            ((L * steps + step) * nts) * VD_ADALN_ROWS;
        h3v_mat n1, n2, qw, ow, qn, kn, f1, f2;
        snprintf(name, sizeof(name), "blocks.%lld.norm1.weight", (long long)L);
        if (!h3v_mat_get(&m->dit, name, &n1, err)) goto done;
        snprintf(name, sizeof(name), "blocks.%lld.norm2.weight", (long long)L);
        if (!h3v_mat_get(&m->dit, name, &n2, err)) goto done;
        snprintf(name, sizeof(name), "blocks.%lld.attn.qkv_proj.weight", (long long)L);
        if (!h3v_mat_get(&m->dit, name, &qw, err)) goto done;
        snprintf(name, sizeof(name), "blocks.%lld.attn.out_proj.weight", (long long)L);
        if (!h3v_mat_get(&m->dit, name, &ow, err)) goto done;
        snprintf(name, sizeof(name), "blocks.%lld.attn.q_norm.weight", (long long)L);
        if (!h3v_mat_get(&m->dit, name, &qn, err)) goto done;
        snprintf(name, sizeof(name), "blocks.%lld.attn.k_norm.weight", (long long)L);
        if (!h3v_mat_get(&m->dit, name, &kn, err)) goto done;
        snprintf(name, sizeof(name), "blocks.%lld.mlp.fc1.weight", (long long)L);
        if (!h3v_mat_get(&m->dit, name, &f1, err)) goto done;
        snprintf(name, sizeof(name), "blocks.%lld.mlp.fc2.weight", (long long)L);
        if (!h3v_mat_get(&m->dit, name, &f2, err)) goto done;

        float *w1 = h3v_vec(&n1), *w2 = h3v_vec(&n2);
        float *wq = h3v_vec(&qn), *wk = h3v_vec(&kn);

        /* norm1 -> modulate -> attention -> gate1 -> residual */
        for (int64_t r = 0; r < seq; r++) {
            const float *sh = mod + H3V_MODROW(r, 0) * D;
            const float *sc = mod + H3V_MODROW(r, 1) * D;
            h3v_rmsnorm(tmp + r * D, x + r * D, w1, D, 1e-5f);
            for (int64_t i = 0; i < D; i++)
                tmp[r * D + i] = tmp[r * D + i] * (1.0f + sc[i]) + sh[i];
        }
        if (h3v_legacy_layout()) {
            h3v_gemm(qkv, tmp, &qw, seq);
            for (int64_t r = 0; r < seq; r++) {
                const float *src = qkv + r * 3 * QKV;
                memcpy(attn + r * QKV, src, (size_t)QKV * sizeof(float));
                memcpy(kbuf + r * QKV, src + QKV, (size_t)QKV * sizeof(float));
                memcpy(vbuf + r * QKV, src + 2 * QKV, (size_t)QKV * sizeof(float));
            }
        } else {
            h3v_gemm_qkv(attn, kbuf, vbuf, tmp, &qw, seq, H, HD);
        }
        if (!h3v_gpu_on() || !h3v_gpu_qk() ||
            !h3v_gpu_qk_prep(attn, kbuf, wq, wk, cos, sin, (int)seq,
                             (int)QKV, (int)H, (int)HD, 1, h3v_rope_mode())) {
            h3v_qk_prep(attn, seq, wq, cos, sin, 1);
            h3v_qk_prep(kbuf, seq, wk, cos, sin, 1);
        }
        h3v_attention(prod, attn, kbuf, vbuf, seq, H, HD, QKV);
        {
            /* out_proj and the attention gate */
            float *ao = h3v_f32(seq * D);
            h3v_gemm(ao, prod, &ow, seq);
            if (block_stats && (L < 3 || L % 8 == 0 || L + 1 == VD_LAYERS)) {
                double a2 = 0, t2 = 0, q2 = 0, x2 = 0, ss = 0, sh2 = 0, g2 = 0;
                for (int64_t i = 0; i < seq * (int64_t)D; i++) {
                    a2 += (double)ao[i] * ao[i];
                    t2 += (double)tmp[i] * tmp[i];
                    x2 += (double)x[i] * x[i];
                }
                for (int64_t i = 0; i < seq * (int64_t)QKV; i++) q2 += (double)prod[i] * prod[i];
            for (int64_t i = 0; i < D; i++) {
                double s0 = mod[H3V_MODROW(t_video0, 0) * D + i];
                double s1 = mod[H3V_MODROW(t_video0, 1) * D + i];
                double s2v = mod[H3V_MODROW(t_video0, 2) * D + i];
                    sh2 += s0 * s0; ss += s1 * s1; g2 += s2v * s2v;
                }
                const double n1 = (double)seq * D, n2 = (double)seq * QKV, nd = (double)D;
                double xt = 0, xa = 0, xv = 0, gt = 0, gm = 0;
                double scm = 0, shm = 0, gam = 0, gmm = 0, sc2 = 0, gm2 = 0;
                for (int64_t i = 0; i < ntext * D; i++)
                    xt += (double)x[i] * x[i];
                for (int64_t i = 0; i < naudio * D; i++)
                    xa += (double)x[ntext * D + i] * x[ntext * D + i];
                for (int64_t i = 0; i < nvideo * D; i++)
                    xv += (double)x[t_video0 * D + i] * x[t_video0 * D + i];
                for (int64_t i = 0; i < D; i++) {
                    double a = mod[H3V_MODROW(ntext, 2) * D + i];
                    double b = mod[H3V_MODROW(t_video0, 5) * D + i];
                    double s0 = mod[H3V_MODROW(t_video0, 0) * D + i];
                    double s1 = mod[H3V_MODROW(t_video0, 1) * D + i];
                    double s4 = mod[H3V_MODROW(t_video0, 4) * D + i];
                    gt += a * a; gm += b * b;
                    gam += a; gmm += b; scm += s1; shm += s0;
                    sc2 += s1 * s1; gm2 += s4 * s4;
                }
                CWICK_INFO("video", "BLOCK %2lld rms: x %.3f [t %.3f a %.3f v %.3f] "
                           "normmod %.3f qkv %.3f attnout %.3f | video-mod sc %.3f sh %.3f "
                           "gate %.3f mlpgate %.3f | txtgate %.3f",
                           (long long)L, sqrt(x2 / n1),
                           sqrt(xt / ((double)ntext * D)), sqrt(xa / ((double)naudio * D)),
                           sqrt(xv / ((double)nvideo * D)),
                           sqrt(t2 / n1), sqrt(q2 / n2), sqrt(a2 / n1),
                           sqrt(ss / nd), sqrt(sh2 / nd), sqrt(g2 / nd),
                           sqrt(gm / nd), sqrt(gt / nd));
                CWICK_INFO("video", "MODMEAN %2lld: sc %.3f+/-%.3f  sh %.3f  "
                           "gate %.3f  mlpsc %.3f+/-%.3f  mlpgate %.3f",
                           (long long)L, scm / nd, sqrt(sc2 / nd - (scm / nd) * (scm / nd)),
                           shm / nd, gam / nd, gmm / nd,
                           sqrt(gm2 / nd - (gmm / nd) * (gmm / nd)), gm / nd);
            }
            for (int64_t r = 0; r < seq; r++) {
                const float *g = mod + H3V_MODROW(r, 2) * D;
                for (int64_t i = 0; i < D; i++)
                    x[r * D + i] += g[i] * ao[r * D + i];
            }
            free(ao);
        }

        /* norm2 -> modulate -> SwiGLU -> gate2 -> residual */
        for (int64_t r = 0; r < seq; r++) {
            const float *sh = mod + H3V_MODROW(r, 3) * D;
            const float *sc = mod + H3V_MODROW(r, 4) * D;
            h3v_rmsnorm(tmp + r * D, x + r * D, w2, D, 1e-5f);
            for (int64_t i = 0; i < D; i++)
                tmp[r * D + i] = tmp[r * D + i] * (1.0f + sc[i]) + sh[i];
        }
        /* fc1 emits [gate | value] over 2*F columns; the SwiGLU keeps its
         * result in the first half of each row, so fc2 reads with stride 2*F. */
        h3v_gemm(gate, tmp, &f1, seq);
        {
            if (!h3v_gpu_on() || !h3v_gpu_swiglu_on() ||
                !h3v_gpu_swiglu(gate, seq * F, F, h3v_legacy_layout())) {
                for (int64_t r = 0; r < seq; r++)
                    h3v_swiglu(gate + r * 2 * F, F, F);
            }
            float *mo = h3v_f32(seq * D);
            h3v_gemm_s(mo, gate, 2 * F, &f2, seq);
            {
                const char *ems = getenv("CWICK_H3V_MLPSCALE");
                if (ems && *ems) {
                    float k = (float)atof(ems);
                    for (int64_t i = 0; i < seq * D; i++) mo[i] *= k;
                }
            }
            if (block_stats && (L < 3 || L % 8 == 0 || L + 1 == VD_LAYERS)) {
                double tm = 0, mm2 = 0;
                for (int64_t r = t_video0; r < seq; r++) {
                    for (int64_t i = 0; i < D; i++) {
                        tm += (double)tmp[r * D + i] * tmp[r * D + i];
                        mm2 += (double)mo[r * D + i] * mo[r * D + i];
                    }
                }
                CWICK_INFO("video", "MLP %2lld video: normmod %.3f mo %.3f", (long long)L,
                           sqrt(tm / ((double)nvideo * D)), sqrt(mm2 / ((double)nvideo * D)));
            }
            for (int64_t r = 0; r < seq; r++) {
                const float *g = mod + H3V_MODROW(r, 5) * D;
                for (int64_t i = 0; i < D; i++)
                    x[r * D + i] += g[i] * mo[r * D + i];
            }
            free(mo);
        }
        free(w1); free(w2); free(wq); free(wk);
        /* A single sampling step costs minutes at 1280x720. Reporting only at
         * step boundaries leaves the client's progress bar frozen at 10%,
         * which is indistinguishable from a hang. */
        if (cb)
            cb(cbctx, H3V_PHASE_DIT, (int)step, steps,
               ((double)step + (double)(L + 1) / (double)VD_LAYERS) /
               (double)steps);
        if (step == 0 && L == 0)
            CWICK_INFO("video", "dit block 0 done over %lld rows", (long long)seq);
    }

    /* final_layer: norm -> modulate -> split into video and audio velocity. */
    {
        h3v_mat fn, vw, vwb, aw, awb;
        if (!h3v_mat_get(&m->dit, "final_layer.norm.weight", &fn, err)) goto done;
        if (!h3v_mat_get(&m->dit, "final_layer.video_out.weight", &vw, err)) goto done;
        if (!h3v_mat_get(&m->dit, "final_layer.video_out.bias", &vwb, err)) goto done;
        if (!h3v_mat_get(&m->dit, "final_layer.audio_out.weight", &aw, err)) goto done;
        if (!h3v_mat_get(&m->dit, "final_layer.audio_out.bias", &awb, err)) goto done;

        float *fw = h3v_vec(&fn);
        const float *fmod = final_mod_all + (size_t)step * nts * VD_FINAL_ADALN;
        float *fv = h3v_f32(seq * D);
        /* Final norm_out has one shift/scale pair per timestep, indexed per
         * row (not per modality). */
        for (int64_t r = 0; r < seq; r++) {
            const float *p = fmod + (size_t)H3V_TI(r) * VD_FINAL_ADALN;
            const float *sh = p, *sc = p + D;
            h3v_rmsnorm(fv + r * D, x + r * D, fw, D, 1e-5f);
            for (int64_t i = 0; i < D; i++)
                fv[r * D + i] = fv[r * D + i] * (1.0f + sc[i]) + sh[i];
        }
        float *vo = h3v_f32(seq * VD_PATCH_DIM);
        float *ao = h3v_f32(seq * VD_AUDIO);
        h3v_gemm(vo, fv, &vw, seq);
        {
            float *b = h3v_vec(&vwb);
            h3v_add_bias(vo, b, VD_PATCH_DIM, seq);
            free(b);
        }
        h3v_gemm(ao, fv, &aw, seq);
        {
            float *b = h3v_vec(&awb);
            h3v_add_bias(ao, b, VD_AUDIO, seq);
            free(b);
        }
        /* Does the velocity still depend on the input row, or has the stream
         * collapsed onto one vector that every row emits? The row-mean versus
         * the row-deviation answers that in one line: a deviation far below
         * the mean means the network is writing a constant field, which the
         * sampler then turns into noise because the constant is unrelated to
         * the noise it is asked to remove. */
        if (getenv("CWICK_H3V_FINALSTATS")) {
            double xm = 0.0, xd = 0.0, vm = 0.0, vd = 0.0, shr = 0.0, scr = 0.0;
            for (int64_t i = 0; i < (int64_t)nvideo * D; i++)
                xm += (double)x[t_video0 * D + i];
            for (int64_t i = 0; i < (int64_t)nvideo * VD_PATCH_DIM; i++)
                vm += (double)vo[t_video0 * VD_PATCH_DIM + i];
            for (int64_t i = 0; i < D; i++) {
                shr += (double)fmod[i] * fmod[i];
                scr += (double)fmod[D + i] * fmod[D + i];
            }
            const double nx = (double)(nvideo * D), nv2 = (double)(nvideo * VD_PATCH_DIM);
            const double mx = xm / nx, mv = vm / nv2;
            for (int64_t r = 0; r < nvideo; r++) {
                for (int64_t i = 0; i < D; i++) {
                    double d = (double)x[(t_video0 + r) * D + i] - mx;
                    xd += d * d;
                }
                for (int64_t i = 0; i < VD_PATCH_DIM; i++) {
                    double d = (double)vo[(t_video0 + r) * VD_PATCH_DIM + i] - mv;
                    vd += d * d;
                }
            }
            CWICK_INFO("video", "FINALSTATS rms sh %.3f sc %.3f ; x mean-of-rows "
                       "%.5f row-dev rms %.4f ; v mean-of-rows %.5f row-dev rms %.4f",
                       sqrt(shr / (double)D), sqrt(scr / (double)D), mx,
                       sqrt(xd / nx), mv, sqrt(vd / nv2));
        }
        memcpy(vout, vo + t_video0 * VD_PATCH_DIM,
               (size_t)nvideo * VD_PATCH_DIM * sizeof(float));
        memcpy(aout, ao + (ntext + nref_use) * VD_AUDIO,
               (size_t)naudio * VD_AUDIO * sizeof(float));
        free(fw); free(fv); free(vo); free(ao);
    }
    ok = true;

done:
    free(x); free(tmp); free(qkv); free(attn); free(kbuf); free(vbuf);
    free(prod); free(gate);
    return ok;
}

/* ---- sampling --------------------------------------------------------- */

/* Deterministic standard-normal draws. Box-Muller on a splitmix64 stream: the
 * same seed gives the same clip on every machine, which is what makes a
 * side-by-side A/B of two builds meaningful. */
static uint64_t h3v_rand(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void h3v_noise(float *dst, int64_t n, uint64_t *s) {
    const double inv = 1.0 / 9007199254740992.0;
    for (int64_t i = 0; i < n; i += 2) {
        double u1 = (double)(h3v_rand(s) >> 11) * inv;
        double u2 = (double)(h3v_rand(s) >> 11) * inv;
        if (u1 < 1e-12) u1 = 1e-12;
        double r = sqrt(-2.0 * log(u1));
        double a = 6.283185307179586 * u2;
        dst[i] = (float)(r * cos(a));
        if (i + 1 < n) dst[i + 1] = (float)(r * sin(a));
    }
}

/* Flow-matching sigma grid. The checkpoint is trained with a 12.0 shift on the
 * rectified-flow schedule; the sampler walks it from 1 to 0 in `steps` even
 * increments and integrates with Euler in velocity form. */
static void h3v_sigmas(int steps, float shift, float *sg) {
    for (int i = 0; i <= steps; i++) {
        double t = 1.0 - (double)i / (double)steps;
        sg[i] = (float)((shift * t) / (1.0 + (shift - 1.0) * t));
    }
    sg[steps] = 0.0f;
}

/* ---- denoise quality signal (pure C, no extra model reads) ----------- */

/* Correlation floor below which a generated latent step is flagged as noise.
 * Provisional until a known-good clip is measured on this machine; overridable
 * so the threshold can be calibrated without a rebuild. */
#define H3V_CORR_FLOOR 0.35

/* Mean lag-1 horizontal autocorrelation and standard deviation of one latent
 * time step. The video latent is [C][T][H][W] — the same convention
 * h3v_pack3d()/h3v_unpack3d() and the VAE decoder use — so channel c at time t
 * starts at (c*lt + t)*lh*lw, NOT at t*VD_LATENT*lh*lw. Gathering the wrong
 * plane used to report a "correlation" measured across two unrelated channels
 * and made the noise floor meaningless. A decoded natural latent is smooth so
 * neighbouring cells agree; white Gaussian noise is uncorrelated and sits near
 * zero. This is the cheapest signal that separates "denoised" from "still
 * noise" without decoding a single frame. */
static double h3v_lat_step_stats(const float *lat, int64_t lt, int64_t t,
                                 int64_t lh, int64_t lw, double *std_out) {
    const int64_t n = (int64_t)VD_LATENT * lh * lw;
    double sum = 0.0, sum2 = 0.0;
    for (int64_t c = 0; c < VD_LATENT; c++) {
        const float *s = lat + (size_t)(c * lt + t) * (size_t)lh * (size_t)lw;
        for (int64_t i = 0; i < lh * lw; i++) {
            double v = s[i];
            sum += v; sum2 += v * v;
        }
    }
    double mean = sum / (double)n;
    double var = sum2 / (double)n - mean * mean;
    double sd = var > 0.0 ? sqrt(var) : 0.0;
    if (std_out) *std_out = sd;
    if (lw < 2 || sd <= 0.0) return 0.0;

    double num = 0.0, ax = 0.0, ay = 0.0;
    for (int64_t c = 0; c < VD_LATENT; c++) {
        const float *s = lat + (size_t)(c * lt + t) * (size_t)lh * (size_t)lw;
        for (int64_t row = 0; row < lh; row++) {
            const float *p = s + row * lw;
            for (int64_t w = 0; w + 1 < lw; w++) {
                double a = (double)p[w] - mean, b = (double)p[w + 1] - mean;
                num += a * b; ax += a * a; ay += b * b;
            }
        }
    }
    double den = sqrt(ax * ay);
    return den > 0.0 ? num / den : 0.0;
}

/* Synthetic check for the [C][T][H][W] convention. Every channel at t=0 holds
 * the same smooth horizontal ramp (correlation close to 1), while all other
 * time steps hold alternating values (correlation close to -1). A reader with
 * the wrong time/channel stride reports the noise planes instead. */
bool h3v_layout_selftest(void) {
    const int64_t lt = 3, lh = 4, lw = 8;
    const int64_t plane = lh * lw;
    float *lat = h3v_f32((size_t)VD_LATENT * (size_t)lt * (size_t)plane);
    /* Deterministic alternating fill: -1,+1,-1,+1,... Row boundaries repeat
     * a sign, so with lw=8 the lag-1 correlation is -(lw-2)/lw ~ -0.75. */
    for (int64_t i = 0; i < (int64_t)VD_LATENT * lt * plane; i++)
        lat[i] = (i & 1) ? 1.0f : -1.0f;
    for (int64_t c = 0; c < VD_LATENT; c++) {
        float *p = lat + (size_t)c * (size_t)lt * (size_t)plane;
        for (int64_t y = 0; y < lh; y++)
            for (int64_t x = 0; x < lw; x++)
                p[y * lw + x] = 4.0f + 0.01f * (float)x;
    }
    double sd = 0.0;
    double c0 = h3v_lat_step_stats(lat, lt, 0, lh, lw, &sd);
    double c1 = h3v_lat_step_stats(lat, lt, 1, lh, lw, NULL);
    /* An 8-cell linear ramp has lag-1 corr 0.88; the alternating fill has
     * -0.75. The old [T][C][H][W] stride at t=1 read channel 8's ramp and
     * would report +0.88 instead of a negative value. */
    bool ok = c0 > 0.8 && c1 < -0.6;
    /* pack/unpack must be exact inverses on the same buffer layout. */
    float *rows = h3v_f32((size_t)lt * (lh / VD_PATCH) * (lw / VD_PATCH) *
                          VD_PATCH_DIM);
    float *back = h3v_f32((size_t)VD_LATENT * (size_t)lt * (size_t)plane);
    h3v_pack3d(rows, lat, lt, lh, lw);
    h3v_unpack3d(back, rows, lt, lh, lw);
    for (int64_t i = 0; i < (int64_t)VD_LATENT * lt * plane; i++)
        if (back[i] != lat[i]) { ok = false; break; }
    free(lat); free(rows); free(back);
    return ok;
}

bool h3v_variant_selftest(void) {
    bool ok = h3v_variant_name_eq("FL2VA", "fl2va") &&
              h3v_variant_name_eq("Ref2VA", "ref2va") &&
              h3v_variant_name_eq("Ref2VA", "REF2VA") &&
              h3v_variant_name_eq("FLAT", "flat");
    ok = ok && !h3v_variant_name_eq("FL2VA", "Ref2VA") &&
               !h3v_variant_name_eq("FL2VA", "fl2va2") &&
               !h3v_variant_name_eq(NULL, "FL2VA") &&
               !h3v_variant_name_eq("FL2VA", "");
    return ok;
}

/* ---- debug latent cache (CWICK_H3V_CACHE) ---------------------------- */

/* Six header words pin the latent to the clip it came from, so a cache left
 * over from a different size or duration is ignored rather than decoded. */
static bool h3v_cache_load(int64_t nvideo, int64_t naudio, int64_t lt,
                           int64_t lh, int64_t lw, float *vlat, float *vrows,
                           float *arows) {
    const char *dir = getenv("CWICK_H3V_CACHE");
    if (!dir || !*dir) return false;
    char p[1152];
    int64_t want[6] = {nvideo, naudio, lt, lh, lw, VD_PATCH_DIM};
    int64_t got[6];

    snprintf(p, sizeof(p), "%s/vlat.f32", dir);
    FILE *f = fopen(p, "rb");
    if (!f) return false;
    if (fread(got, sizeof(got), 1, f) != 1 || memcmp(got, want, sizeof(want)) != 0) {
        fclose(f);
        return false;
    }
    size_t vn = (size_t)(H3V_LATENT_CHANNELS * lt * lh * lw);
    bool ok = fread(vlat, sizeof(float), vn, f) == vn;
    fclose(f);
    if (!ok) return false;

    snprintf(p, sizeof(p), "%s/vrows.f32", dir);
    f = fopen(p, "rb");
    if (!f) return false;
    ok = fread(vrows, sizeof(float), (size_t)(nvideo * VD_PATCH_DIM), f) ==
         (size_t)(nvideo * VD_PATCH_DIM);
    fclose(f);
    if (!ok) return false;

    snprintf(p, sizeof(p), "%s/arows.f32", dir);
    f = fopen(p, "rb");
    if (!f) return false;
    ok = fread(arows, sizeof(float), (size_t)(naudio * VD_AUDIO), f) ==
         (size_t)(naudio * VD_AUDIO);
    fclose(f);
    return ok;
}

static void h3v_cache_save(int64_t nvideo, int64_t naudio, int64_t lt,
                           int64_t lh, int64_t lw, const float *vlat,
                           const float *vrows, const float *arows) {
    const char *dir = getenv("CWICK_H3V_CACHE");
    if (!dir || !*dir) return;
    char p[1152];
    int64_t head[6] = {nvideo, naudio, lt, lh, lw, VD_PATCH_DIM};
    snprintf(p, sizeof(p), "%s/vlat.f32", dir);
    FILE *f = fopen(p, "wb");
    if (!f) return;
    fwrite(head, sizeof(head), 1, f);
    fwrite(vlat, sizeof(float), (size_t)(H3V_LATENT_CHANNELS * lt * lh * lw), f);
    fclose(f);
    snprintf(p, sizeof(p), "%s/vrows.f32", dir);
    if ((f = fopen(p, "wb"))) {
        fwrite(vrows, sizeof(float), (size_t)(nvideo * VD_PATCH_DIM), f);
        fclose(f);
    }
    snprintf(p, sizeof(p), "%s/arows.f32", dir);
    if ((f = fopen(p, "wb"))) {
        fwrite(arows, sizeof(float), (size_t)(naudio * VD_AUDIO), f);
        fclose(f);
    }
    CWICK_INFO("video", "latent cached under %s", dir);
}

/* ---- generation ------------------------------------------------------ */

void h3v_ref_encode_size(int src_w, int src_h, int edge, int *w, int *h) {
    if (!w || !h || src_w <= 0 || src_h <= 0) return;
    if (edge < H3V_CANVAS_MULTIPLE || edge > H3V_REF_EDGE_MAX) return;
    double s = (double)edge / (double)(src_w < src_h ? src_w : src_h);
    int rw = (int)(src_w * s / H3V_CANVAS_MULTIPLE + 0.5) * H3V_CANVAS_MULTIPLE;
    int rh = (int)(src_h * s / H3V_CANVAS_MULTIPLE + 0.5) * H3V_CANVAS_MULTIPLE;
    if (rw < H3V_CANVAS_MULTIPLE) rw = H3V_CANVAS_MULTIPLE;
    if (rh < H3V_CANVAS_MULTIPLE) rh = H3V_CANVAS_MULTIPLE;
    *w = rw; *h = rh;
}

/* Bilinear-resize an RGB24 image to the render canvas. A conditioning frame
 * must match the latent grid exactly, so a plain nearest/bilinear resample is
 * enough — no letterboxing: the frame is stretched to the requested aspect. */
static unsigned char *h3v_resize_rgb(const unsigned char *src, int sw, int sh,
                                     int dw, int dh) {
    unsigned char *dst = (unsigned char *)cwick_xmalloc((size_t)dw * dh * 3);
    for (int y = 0; y < dh; y++) {
        float fy = (float)(y + 0.5f) * (float)sh / (float)dh - 0.5f;
        int y0 = (int)floorf(fy), y1 = y0 + 1;
        float wy = fy - y0;
        if (y0 < 0) { y0 = 0; wy = 0; }
        if (y1 >= sh) { y1 = sh - 1; }
        for (int x = 0; x < dw; x++) {
            float fx = (float)(x + 0.5f) * (float)sw / (float)dw - 0.5f;
            int x0 = (int)floorf(fx), x1 = x0 + 1;
            float wx = fx - x0;
            if (x0 < 0) { x0 = 0; wx = 0; }
            if (x1 >= sw) { x1 = sw - 1; }
            for (int c = 0; c < 3; c++) {
                float v =
                    ((1 - wx) * (1 - wy) * src[(y0 * sw + x0) * 3 + c] +
                     (wx) * (1 - wy) * src[(y0 * sw + x1) * 3 + c] +
                     (1 - wx) * (wy) * src[(y1 * sw + x0) * 3 + c] +
                     (wx) * (wy) * src[(y1 * sw + x1) * 3 + c]);
                int b = (int)(v + 0.5f);
            dst[(y * dw + x) * 3 + c] = (unsigned char)(b < 0 ? 0 : b > 255 ? 255 : b);
            }
        }
    }
    return dst;
}

static bool h3v_gen_impl(h3v_model *m, const char *prompt, const char *variant,
                         const unsigned char *ref_rgb, int ref_w, int ref_h,
                         int w, int h, int seconds, uint64_t seed,
                         int steps_override,
                         h3v_progress_fn cb, void *cbctx,
                         unsigned char **out, size_t *out_len, h3v_timing *tim,
                         cwick_error *err) {
    if (!h3v_model_ready(m)) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "video model not ready");
        return false;
    }
    /* Resolve and pin the requested transformer before taking the lock. Each
     * job opens its own DiT shards, so two variants never map at once. */
    const char *resolved = h3v_model_resolve_variant(m, variant);
    if (!resolved) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "unknown video variant '%s'", variant ? variant : "");
        return false;
    }
    int vi = 0;
    for (; vi < m->nvariants; vi++)
        if (h3v_variant_name_eq(m->variants[vi], resolved)) break;
    const char *chosen_dit = m->variant_dits[vi];
    if (!prompt || !*prompt) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "prompt is required");
        return false;
    }
    if (w < H3V_MIN_SIZE || h < H3V_MIN_SIZE ||
        w > H3V_MAX_SIZE || h > H3V_MAX_SIZE) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "size out of range");
        return false;
    }
    if (out) *out = NULL;
    if (out_len) *out_len = 0;

    h3v_timing t;
    memset(&t, 0, sizeof(t));
    if (tim) *tim = t;
    double t0 = h3v_ms();

    /* 32 is the smallest multiple that satisfies both the 16x VAE ratio and
     * the 2x patch, so every axis divides exactly. */
    int cw = (w + H3V_CANVAS_MULTIPLE - 1) / H3V_CANVAS_MULTIPLE * H3V_CANVAS_MULTIPLE;
    int ch = (h + H3V_CANVAS_MULTIPLE - 1) / H3V_CANVAS_MULTIPLE * H3V_CANVAS_MULTIPLE;
    if (cw > H3V_MAX_SIZE) cw = H3V_MAX_SIZE;
    if (ch > H3V_MAX_SIZE) ch = H3V_MAX_SIZE;
    const int64_t lw = cw / H3V_VAE_RATIO, lh = ch / H3V_VAE_RATIO;
    const int64_t frames = h3v_frames_for_seconds(seconds);
    /* 17 frames per clip, 5 latent steps per clip. The decoder emits four
     * frames per latent step, so 5 steps produce 20 and the mux keeps the 17
     * the checkpoint's frame rule asks for. */
    const int64_t clips = frames / H3V_CLIP_FRAMES;
    const int64_t lt = clips * H3V_CLIP_LATENT;
    const int64_t nvideo = lt * (lh / VD_PATCH) * (lw / VD_PATCH);
    /* Audio runs at 40 tokens/s (32 kHz / 800 hop), stereo interleaved. */
    const int64_t aframes = (frames * 5 + 2) / 3;
    const int64_t naudio = aframes * 2;

    /* Sampling budget. `steps_override` is the per-request value carried by
     * the video job (a short smoke run asks for 2-4 steps); 0 means no override
     * and falls back to the process-wide knob. Both are clamped to
     * H3V_SAMPLE_STEPS: the sigma/timestep scratch arrays below are sized for
     * exactly that many entries, so a larger value used to write past them. */
    int steps = H3V_SAMPLE_STEPS;
    {
        const char *es = getenv("CWICK_H3V_STEPS");
        if (es && *es) {
            int v = atoi(es);
            if (v >= 1 && v <= H3V_SAMPLE_STEPS) steps = v;
        }
    }
    if (steps_override >= 1 && steps_override <= H3V_SAMPLE_STEPS)
        steps = steps_override;
    t.frames = frames;
    t.width = cw; t.height = ch;
    t.steps = steps;
    t.latent_t = lt; t.latent_h = lh; t.latent_w = lw;
    t.audio_samples = aframes * 800;
    t.video_tokens = nvideo;
    t.audio_tokens = naudio;

    pthread_mutex_lock(&m->lock);
    float *cond = NULL, *vlat = NULL, *vrows = NULL, *arows = NULL;
    float *vvel = NULL, *avel = NULL;
    float *mod_all = NULL, *final_mod_all = NULL;
    float *rope_cos = NULL, *rope_sin = NULL;
    float *crows = NULL;
    /* The reference frame belongs in its own conditioning rows, noise-augmented
     * to t=0.999 and held there for every step, while the generated frames
     * (including frame 0) denoise from pure noise. Writing the keyframe into
     * the target latent's t=0 instead pins one frame but starves the sampling
     * loop of the conditioning it should attend to on every layer: measured
     * 256^2 / 4 steps, same seed, the target-t0 path gave a blurred face
     * while independent rows kept the full cream-coat figure sharp. Default
     * to the correct layout; CWICK_H3V_CONDROWS=0 restores the old one for A/B. */
    int condrows = 1;
    {
        const char *ec = getenv("CWICK_H3V_CONDROWS");
        condrows = !(ec && *ec && strcmp(ec, "0") == 0);
    }
    float *ref_lat = NULL, *ref_packed = NULL;
    unsigned char *ref_rgb_scaled = NULL;
    float *ref_rgb_vision = NULL, *text_tags = NULL;
    int64_t ref_rows = 0;   /* packed reference rows (0 = text-to-video) */
    int64_t rlh = 0, rlw = 0;  /* reference latent grid, independent of canvas */
    bool ok = false;

    /* A reference image is also part of the text presentation: the model
     * reads `Picture 1: <vision block>` before the prompt, and those rows
     * carry the video modality tag. Without it the DiT sees the picture only
     * through its latent rows, which is measurably weaker. The old bare-prompt
     * presentation stays available for A/B. */
    int ref_present = 1;
    {
        const char *ep = getenv("CWICK_H3V_REF_PRESENT");
        ref_present = !(ep && *ep && strcmp(ep, "0") == 0);
    }

    /* 1. text: the H3-Encoder is opened for this call only, then unmapped, so
     * it never competes with the 62 GiB transformer held by the sampling loop. */
    double a = h3v_ms();
    int64_t ntext = 0, ntok = 0;
    if (cb) cb(cbctx, H3V_PHASE_TEXT, 0, 0, 0.0);
    if (ref_present && ref_rgb && ref_w > 0 && ref_h > 0) {
        const size_t px = (size_t)ref_w * (size_t)ref_h;
        ref_rgb_vision = h3v_f32(px * 3);
        for (size_t i = 0; i < px; i++) {
            ref_rgb_vision[i * 3 + 0] = (float)ref_rgb[i * 3 + 0] / 255.0f;
            ref_rgb_vision[i * 3 + 1] = (float)ref_rgb[i * 3 + 1] / 255.0f;
            ref_rgb_vision[i * 3 + 2] = (float)ref_rgb[i * 3 + 2] / 255.0f;
        }
        if (!h3_text_cond_ref_dir(m->te_dir, m->tok_dir, prompt,
                                  ref_rgb_vision, ref_w, ref_h,
                                  &cond, &ntext, &text_tags, err))
            goto done;
    } else if (!h3_text_cond_dir(m->te_dir, m->tok_dir, prompt,
                                 &cond, &ntext, &ntok, err)) {
        goto done;
    }
    if (!ntok) ntok = ntext;
    t.text_encode_ms = h3v_ms() - a;
    t.text_tokens = ntok;
    if (cb) cb(cbctx, H3V_PHASE_TEXT, 0, 0, 1.0);
    CWICK_INFO("video", "phase 1/6 text: %lld tokens encoded in %.1f ms",
               (long long)ntok, t.text_encode_ms);

    /* 2. pack: video latent [C][T][H][W] -> 2x2 packed rows, audio latent in
     * row-major [t][32] so the transformer sees one row per audio token. */
    a = h3v_ms();
    if (cb) cb(cbctx, H3V_PHASE_PACK, 0, 0, 0.0);
    {
        double vn = (double)H3V_LATENT_CHANNELS * (double)lt * (double)lh * (double)lw;
        double an = (double)naudio * (double)VD_AUDIO;
        CWICK_INFO("video", "phase 2/6 pack: %lld video rows + %lld audio rows "
                   "(%.1f MiB latent), canvas %dx%d, %lld frames",
                   (long long)nvideo, (long long)naudio,
                   (vn + an) * sizeof(float) / (1024.0 * 1024.0),
                   cw, ch, (long long)frames);
        vlat = h3v_f32((int64_t)vn);
        vrows = h3v_f32(nvideo * VD_PATCH_DIM);
        arows = h3v_f32(naudio * VD_AUDIO);
        uint64_t rng = seed ? seed : (uint64_t)cwick_prof_ns();
        h3v_noise(vlat, (int64_t)vn, &rng);
        h3v_noise(arows, naudio * VD_AUDIO, &rng);
        if (ref_rgb && ref_w > 0 && ref_h > 0) {
            int rp_w = cw, rp_h = ch;
            /* Ref2VA encodes an image reference at its own resolution; it
             * does not bind the generated canvas. The checkpoint's default is
             * a 2048 short edge, so CWICK_H3V_REF_EDGE=2048 reproduces the
             * released model. The range is H3V_REF_EDGE_MAX, not the canvas
             * cap; with an unset knob the canvas remains the fallback. */
            const char *ere = getenv("CWICK_H3V_REF_EDGE");
            int edge = ere && *ere ? atoi(ere) : 0;
            if (condrows) h3v_ref_encode_size(ref_w, ref_h, edge, &rp_w, &rp_h);
            rlh = rp_h / H3V_VAE_RATIO;
            rlw = rp_w / H3V_VAE_RATIO;
            {
                double e0 = h3v_ms();
                ref_rgb_scaled = h3v_resize_rgb(ref_rgb, ref_w, ref_h, rp_w, rp_h);
                ref_lat = h3v_vae_encode(m, ref_rgb_scaled, rp_w, rp_h, err);
                t.encode_ms = h3v_ms() - e0;
            }
            if (!ref_lat) goto done;
            /* Independent conditioning rows carry the reference frame at a
             * fixed noise level; `CWICK_H3V_REF_NOISE_AUG` only applies to the
             * legacy target-t0 layout that `CWICK_H3V_CONDROWS=0` restores. */
            float ref_aug = 0.0f;
            {
                const char *er = getenv("CWICK_H3V_REF_NOISE_AUG");
                if (er && *er) {
                    double d = atof(er);
                    if (d > 0.0 && d < 1.0) ref_aug = (float)d;
                }
            }
            /* The encoder returns one contiguous [C][H][W] plane set, while
             * vlat is [C][T][H][W]: channel c starts at c*lt*lh*lw. Copying
             * nref floats contiguously wrote the 24 reference channels into
             * channel 0's time axis instead of putting all channels at t=0.
             * Build one augmented frame in encoder layout, then scatter it to
             * vlat t=0 and pack that same frame into the transformer rows. */
            {
                const bool same_geo = (rlh == lh && rlw == lw);
                int64_t plane = rlh * rlw;
                float *ref0 = h3v_f32((int64_t)VD_LATENT * plane);
                for (int64_t c = 0; c < VD_LATENT; c++) {
                    const float *src = ref_lat + c * plane;
                    float *d = ref0 + c * plane;
                    float *dst = same_geo ? vlat + (size_t)(c * lt) * (size_t)(lh * lw) : NULL;
                    for (int64_t i = 0; i < plane; i++) {
                        /* With independent conditioning rows the keyframe is
                         * never written into the generated latent: it lives in
                         * its own rows at noise level ~1, while target frame 0
                         * stays pure noise like every other target frame. */
                        float v = src[i];
                        if (dst && !condrows && ref_aug > 0.0f)
                            v = ref_aug * src[i] + (1.0f - ref_aug) * dst[i];
                        d[i] = v;
                        if (dst && !condrows) dst[i] = v;
                    }
                }
                ref_rows = (rlh / VD_PATCH) * (rlw / VD_PATCH);
                ref_packed = h3v_f32(ref_rows * VD_PATCH_DIM);
                h3v_pack3d(ref_packed, ref0, 1, rlh, rlw);
                if (condrows) {
                    uint64_t crng = seed ? seed ^ 0x9e3779b97f4a7c15ULL
                                         : (uint64_t)cwick_prof_ns();
                    crows = h3v_f32(ref_rows * VD_PATCH_DIM);
                    for (int64_t i = 0; i < ref_rows * VD_PATCH_DIM; i++) {
                        float nz = (float)h3v_rand(&crng) / (float)UINT64_MAX * 2.0f - 1.0f;
                        crows[i] = 0.999f * ref_packed[i] + 0.001f * nz;
                    }
                }
                free(ref0);
            }
            if (!condrows && rlh == lh && rlw == lw)
                memcpy(vrows, ref_packed,
                       (size_t)ref_rows * VD_PATCH_DIM * sizeof(float));
            CWICK_INFO("video", "reference frame: %dx%d RGB encoded (%lld rows, "
                       "noise aug %.2f, %s)",
                       ref_w, ref_h, (long long)ref_rows, ref_aug,
                       condrows ? "independent condition rows" : "latent step 0");
        }
        h3v_pack3d(vrows, vlat, lt, lh, lw);
    }
    t.pack_ms = h3v_ms() - a;
    if (cb) cb(cbctx, H3V_PHASE_PACK, 0, 0, 1.0);

    /* 3. sample: the transformer shards are mapped here and released before
     * the decoders run, which keeps the peak under one 62 GiB model.
     *
     * CWICK_H3V_CACHE=<dir> is a decoder-iteration knob: when that directory
     * holds a matching `vlat.f32`/`arows.f32` pair this phase is skipped and the
     * saved latent is decoded instead. A full sampling pass is ~2.5 minutes per
     * step on the CPU, so without it a one-line change in the video or audio
     * decoder cannot be measured. It never changes what the decoders receive. */
    a = h3v_ms();
    if (!ref_lat && h3v_cache_load(nvideo, naudio, lt, lh, lw, vlat, vrows, arows)) {
        CWICK_INFO("video", "phase 3/6 dit skipped: latent restored from cache");
        t.cached = 1;
        goto decode;
    }
    CWICK_INFO("video", "variant %s: opening DiT shards %s",
               resolved, chosen_dit);
    if (cwick_safetensors_open(chosen_dit, &m->dit, err) != CWICK_OK) goto done;
    m->dit_open = 1;
    /* One no-copy GPU view per shard. Without this the kernel would fault and
     * wrap every projection's pages separately on each of the 50 blocks. */
    if (h3v_gpu_on()) {
        for (size_t i = 0; i < m->dit.map_count; i++)
            h3v_gpu_adopt(m->dit.maps[i], m->dit.map_lens[i]);
    }
    /* The transformer must stay wired through the sampling loop when it fits:
     * every step reads all 62 GiB, and a compressed page faults inside the GPU
     * wait. Fault it in once here; safetensors_close unwires it after the last
     * step. If it does not fit under the ceiling, leave it as a lazy mapping
     * and let the kernel stream it from SSD on demand. */
    h3v_apply_residency(&m->dit, "DiT transformer");
    {
        float sg[H3V_SAMPLE_STEPS + 1], asg[H3V_SAMPLE_STEPS + 1];
        float ts[H3V_SAMPLE_STEPS], ats[H3V_SAMPLE_STEPS];
        h3v_sigmas(steps, m->video_shift, sg);
        h3v_sigmas(steps, m->audio_shift, asg);
        /* Timestep convention. The working image line uses t = 1 - sigma, and
         * the original video sampler did too. A one-step probe at sigma=0.9
         * appeared to favour t=sigma (corr 0.068 -> 0.290 vs -> 0.025), but it
         * used a forced dt=0.9 giant step whose std exploded well past the clean
         * anchor — an out-of-distribution overshoot, not a convention signal.
         * The decisive check was two full 8-step runs at matched size/seed:
         * t=1-sigma ended at corr 0.360 (ok), t=sigma at 0.288 (suspect). The
         * empirical verdict vetoes the change, so the default stays t=1-sigma.
         * CWICK_H3V_TS=sigma keeps the alternative available for a larger
         * (>=4 MP) A/B, where the checkpoint is actually in distribution. */
        int ts_is_sigma = 0;
        {
            const char *ets = getenv("CWICK_H3V_TS");
            if (ets && *ets) {
                if (strcmp(ets, "1minus") == 0 || strcmp(ets, "legacy") == 0)
                    ts_is_sigma = 0;
                else if (strcmp(ets, "sigma") == 0)
                    ts_is_sigma = 1;
            }
        }
        for (int s = 0; s < steps; s++) {
            ts[s] = ts_is_sigma ? sg[s] : 1.0f - sg[s];
            ats[s] = ts_is_sigma ? asg[s] : 1.0f - asg[s];
        }
        /* Decisive one-step velocity test (pure C, probe-only). A 2-step run
         * from pure noise is too coarse to say whether the DiT convention is
         * right — both shifts end at noise. So CWICK_H3V_DIAG_MIX=<m> in (0,1)
         * replaces the generated rows with a KNOWN mix `m*noise + (1-m)*clean`
         * (clean = the encoded reference frame replicated over time) and sets
         * the step-0 timestep to the sigma level that mix represents. One
         * Euler step then has a knowable answer: if the velocity convention is
         * correct, spatial corr of the generated step must RISE toward the
         * clean anchor (~0.77); if it falls or stays at noise, the convention
         * (timestep, layout or sign) is wrong. CWICK_H3V_DIAG_TS overrides only
         * the conditioning timestep so the two conventions can be A/B'd at the
         * same sigma. Off by default. */
        float diag_mix = 0.0f;
        {
            const char *ed = getenv("CWICK_H3V_DIAG_MIX");
            if (ed && *ed) {
                float v = (float)atof(ed);
                if (v > 0.0f && v < 1.0f) diag_mix = v;
            }
        }
        if (diag_mix > 0.0f && ref_rows > 0) {
            int64_t rp = ref_rows * VD_PATCH_DIM;
            for (int64_t i = rp; i < nvideo * VD_PATCH_DIM; i++)
                vrows[i] = diag_mix * vrows[i] +
                           (1.0f - diag_mix) * ref_packed[i % rp];
            /* The mix sits at noise level sigma=m, which in this checkpoint's
             * convention is the conditioning time t = 1 - m, the same value
             * the scheduler hands the transformer at that noise level.
             * CWICK_H3V_DIAG_TS overrides ONLY the conditioning timestep.
             */
            ts[0] = 1.0f - diag_mix;
            {
                const char *et = getenv("CWICK_H3V_DIAG_TS");
                if (et && *et) {
                    float v = (float)atof(et);
                    if (v >= 0.0f && v <= 1.0f) ts[0] = v;
                }
            }
            sg[0] = diag_mix;
            /* A one-step probe at dt=mix is a 0.9-wide Euler step; that is
             * pathological for judging the velocity field because even a
             * correct direction overshoots. CWICK_H3V_DIAG_DT shrinks the step
             * so the local improvement (corr up, std toward the clean anchor)
             * can be observed on its own. Default keeps the old behaviour. */
            sg[1] = 0.0f;
            {
                const char *edt = getenv("CWICK_H3V_DIAG_DT");
                if (edt && *edt) {
                    float v = (float)atof(edt);
                    if (v > 0.0f && v <= diag_mix) sg[1] = diag_mix - v;
                }
            }
            h3v_unpack3d(vlat, vrows, lt, lh, lw);
            double sd = 0.0;
            double c0 = h3v_lat_step_stats(vlat, lt, 1, lh, lw, &sd);
            CWICK_INFO("video", "DIAG mix %.2f: starting gen corr %.3f "
                       "std %.3f, forced ts0 %.3f",
                       diag_mix, c0, sd, ts[0]);
        }
        /* Opt-in convergence trace: print the generated step's spatial corr
         * after every Euler update so a 2-step probe shows whether the latent
         * is moving toward the clean latent or staying at noise level. Off by
         * default; no effect on the sampled rows. */
        int trace_corr = getenv("CWICK_H3V_CORR_TRACE") != NULL || diag_mix > 0.0f;
        /* Opt-in per-step magnitudes. `spatial_corr` only says how smooth the
         * latent is; it cannot tell a wrong direction from a velocity that is
         * near zero. Printing rms(x), rms(v) and rms(dt*v) separates those two
         * cases, which is the difference between "flip the sign" and "the
         * output head is not producing a usable field". Off by default. */
        int step_stats = getenv("CWICK_H3V_STEPSTATS") != NULL;
        const int64_t seq = ntext + (condrows ? ref_rows : 0) + naudio + nvideo;
        rope_cos = h3v_f32(seq * VD_ROPE_DIM);
        rope_sin = h3v_f32(seq * VD_ROPE_DIM);
        if (!rope_cos || !rope_sin) goto done;
        h3v_rope_tables(ntext, nvideo, naudio, condrows ? ref_rows : 0,
                        lt, lh, lw, rlh > 0 ? rlh : lh,
                        rlw > 0 ? rlw : lw, rope_cos, rope_sin);
        double adaln_ms = h3v_ms();
        if (!h3v_adaln_precompute(m, ts, ats, steps, naudio > 0,
                                  condrows && ref_rows > 0,
                                  &mod_all, &final_mod_all, err))
            goto done;
        CWICK_INFO("video", "AdaLN precomputed for %d steps in %.1f ms",
                   steps, h3v_ms() - adaln_ms);
        vvel = h3v_f32(nvideo * VD_PATCH_DIM);
        avel = h3v_f32(naudio * VD_AUDIO);
        CWICK_INFO("video", "phase 3/6 dit: %d steps, seq %lld "
                   "(text %lld + audio %lld + video %lld), shift video %.1f audio %.1f",
                   steps, (long long)(ntext + naudio + nvideo),
                   (long long)ntext, (long long)naudio, (long long)nvideo,
                   m->video_shift, m->audio_shift);
        for (int s = 0; s < steps; s++) {
            double step_ms = h3v_ms();
            if (cb) cb(cbctx, H3V_PHASE_DIT, s, steps,
                       (double)s / (double)steps);
            if (!h3v_dit_step(m, cond, ntext, text_tags, vrows, nvideo,
                              condrows ? crows : NULL, condrows ? ref_rows : 0,
                              arows, naudio, mod_all, final_mod_all,
                              rope_cos, rope_sin,
                              s, steps, cb, cbctx,
                              vvel, avel, err))
                goto done;
            /* Network-probe: with a known mix `x = m*noise + (1-m)*x0` the
             * meaning of the output can be measured instead of assumed. This
             * checkpoint predicts a data-ward velocity, so the reference
             * direction is `x0 - x = m * (x0 - noise)`: cos(v,vel) close to
             * +1 is a correct field, -1 means the sign is inverted and 0 means
             * the field has nothing to do with the denoising direction.
             * cos(v,x0) and cos(v,x) split "predicts x0" from "predicts eps".
             * Needs CWICK_H3V_DIAG_MIX so that x0 is known for every row. */
            int netprobe = getenv("CWICK_H3V_NETPROBE") != NULL;
            if (netprobe && diag_mix > 0.0f && ref_rows > 0) {
                const int64_t rp = ref_rows * VD_PATCH_DIM;
                double vv = 0.0, ddx = 0.0, ex = 0.0, xx = 0.0;
                double cd = 0.0, ce = 0.0, cx = 0.0;
                for (int64_t i = 0; i < nvideo * VD_PATCH_DIM; i++) {
                    double x = vrows[i], x0 = ref_packed[i % rp];
                    double d = (x0 - x) / (double)diag_mix;
                    double v = vvel[i];
                    vv += v * v; ddx += d * d;
                    cd += v * d; ce += v * x0; cx += v * x;
                    ex += x0 * x0; xx += x * x;
                }
                double nv = sqrt(vv), nd = sqrt(ddx);
                double ne = sqrt(ex), nx = sqrt(xx);
                /* Dumping the first step's velocity lets two prompts be compared
                 * row by row: identical (or near-identical) fields mean the text
                 * conditioning never reached the transformer. */
                const char *dv = getenv("CWICK_H3V_DUMPV");
                if (dv && *dv && s == 0) {
                    FILE *fv = fopen(dv, "wb");
                    if (fv) {
                        fwrite(vvel, sizeof(float),
                               (size_t)nvideo * VD_PATCH_DIM, fv);
                        fclose(fv);
                        CWICK_INFO("video", "dumped first-step velocity to %s", dv);
                    }
                }
                CWICK_INFO("video", "NETPROBE: |v| %.4f |(x0-x)/m| %.4f |x0| %.4f "
                           "|x| %.4f ; cos(v,vel) %.3f cos(v,x0) %.3f cos(v,x) %.3f "
                           ", best scale for v=(x0-x)/m: %.3f",
                           nv, nd, ne, nx,
                           nv > 0 && nd > 0 ? cd / (nv * nd) : 0.0,
                           nv > 0 && ne > 0 ? ce / (nv * ne) : 0.0,
                           nv > 0 && nx > 0 ? cx / (nv * nx) : 0.0,
                           nd > 0 ? cd / (nd * nd) : 0.0);
            }
            float dt = sg[s] - sg[s + 1];
            float adt = asg[s] - asg[s + 1];
            /* Direction stayed `+` (the image line's sign). A controlled probe
             * with a known mix (0.6*noise + 0.4*clean) and a SMALL step says
             * the other sign moves toward the clean anchor: minus gave corr
             * 0.216->0.289 / std 0.699->0.886 at dt=0.2, plus gave
             * 0.216->0.146 / 0.699->0.786. But a same-size, same-step end-to-end
             * A/B (256x320, 4 steps, seed 42) went the other way: 0.277 minus
             * vs 0.300 plus. Four steps are dominated by the 0.8-wide final
             * Euler step, where overshoot masks direction, so the end-to-end
             * test cannot settle it. The default therefore keeps the measured
             * winner; CWICK_H3V_SIGN=minus exposes the other direction for a
             * 30-step A/B, where the mid-schedule steps that the probe favours
             * actually decide the result. */
            int vdir = 1;
            {
                const char *es = getenv("CWICK_H3V_SIGN");
                if (es && *es && strcmp(es, "minus") == 0) vdir = -1;
            }
            if (step_stats) {
                double x2 = 0.0, v2 = 0.0, d2 = 0.0;
                const double n = (double)(nvideo * VD_PATCH_DIM);
                for (int64_t i = 0; i < nvideo * VD_PATCH_DIM; i++) {
                    double a = vrows[i], b = vvel[i];
                    double d = (double)vdir * (double)dt * b;
                    x2 += a * a; v2 += b * b; d2 += d * d;
                }
                CWICK_INFO("video", "  stepstats %d/%d: rms x %.4f, rms v %.4f, "
                           "rms dx %.4f (dt %.4f, ts %.4f)",
                           s + 1, steps, sqrt(x2 / n), sqrt(v2 / n),
                           sqrt(d2 / n), dt, ts[s]);
            }
            for (int64_t i = 0; i < nvideo * VD_PATCH_DIM; i++)
                vrows[i] += (float)vdir * dt * vvel[i];
            for (int64_t i = 0; i < naudio * VD_AUDIO; i++)
                arows[i] += (float)vdir * adt * avel[i];
            /* With the old target-t0 layout the keyframe lived in the first
             * generated rows, so it had to be restored after each Euler step.
             * With independent conditioning rows the keyframe is in `crows`
             * (held there by the caller) and the target frame 0 is a normal
             * denoising frame — restoring it here would pin it to a clean
             * frame at every noise level and contradict the schedule. */
            if (ref_rows > 0 && !condrows && rlh == lh && rlw == lw)
                memcpy(vrows, ref_packed,
                       (size_t)ref_rows * VD_PATCH_DIM * sizeof(float));
            if (trace_corr && lt >= 2 && lw >= 2 && lh >= 1) {
                h3v_unpack3d(vlat, vrows, lt, lh, lw);
                double sd = 0.0;
                double c = h3v_lat_step_stats(vlat, lt, ref_lat ? 1 : 0, lh, lw, &sd);
                CWICK_INFO("video", "  corr trace step %d/%d: gen corr %.3f "
                           "std %.3f", s + 1, steps, c, sd);
            }
            CWICK_INFO("video", "  dit step %d/%d: %.1f ms (elapsed %.1f ms)",
                       s + 1, steps, h3v_ms() - step_ms, h3v_ms() - a);
        }
    }
    t.dit_ms = h3v_ms() - a;
    if (cb) cb(cbctx, H3V_PHASE_DIT, steps, steps, 1.0);
    if (m->dit_open) { cwick_safetensors_close(&m->dit); m->dit_open = 0; }
    h3v_unpack3d(vlat, vrows, lt, lh, lw);
    h3v_cache_save(nvideo, naudio, lt, lh, lw, vlat, vrows, arows);
    CWICK_INFO("video", "phase 3/6 dit done: %.1f ms total, transformer unmapped",
               t.dit_ms);

decode:
    /* Quality signal: measure the first generated latent step (step 1 for i2v,
     * whose step 0 is held clean) against the clean conditioning step. A value
     * near zero means the sampler left white noise, which muxes into a valid
     * mp4 that looks like static — the exact failure this flag is for. Placed
     * after the label so the cache path (which skips the DiT) is covered too. */
    if (lt >= 2 && lw >= 2 && lh >= 1) {
        double gen_std = 0.0, ref_std = 0.0;
        int64_t gen_t = ref_lat ? 1 : 0;
        t.spatial_corr = h3v_lat_step_stats(vlat, lt, gen_t, lh, lw, &gen_std);
        if (ref_lat) {
            /* The clean step is step 0 of vlat, identical to ref_lat's content. */
            t.clean_corr = h3v_lat_step_stats(vlat, lt, 0, lh, lw, &ref_std);
            t.snr_db = (ref_std > 0.0 && gen_std > 0.0)
                       ? 20.0 * log10(gen_std / ref_std) : 0.0;
        }
        double floor = H3V_CORR_FLOOR;
        {
            const char *ef = getenv("CWICK_H3V_CORR_FLOOR");
            if (ef && *ef) {
                double v = atof(ef);
                if (v > 0.0 && v < 1.0) floor = v;
            }
        }
        t.suspect_noise = t.spatial_corr < floor ? 1 : 0;
        CWICK_INFO("video", "denoise check: generated-step spatial corr %.3f "
                   "(floor %.2f), gen/ref std %.2f dB => %s",
                   t.spatial_corr, floor, t.snr_db,
                   t.suspect_noise ? "SUSPECT NOISE" : "ok");
    }

    /* 4. video VAE: 36-block ViT decoder, tiled with a cross-fade. */
    a = h3v_ms();
    if (cb) cb(cbctx, H3V_PHASE_VIDEO, 0, 0, 0.0);
    unsigned char *rgb = h3v_vae_decode(m, vlat, lt, lh, lw, cb, cbctx, err);
    if (!rgb) goto done;
    t.vae_ms = h3v_ms() - a;
    if (cb) cb(cbctx, H3V_PHASE_VIDEO, 0, 0, 1.0);
    CWICK_INFO("video", "phase 4/6 video_vae: %lldx%lld decoded in %.1f ms",
               (long long)cw, (long long)ch, t.vae_ms);

    /* 5. audio VAE: the transformer packs audio rows channel-major — every
     * latent of the left channel, then every latent of the right one — while
     * the decoder wants [32][2*T] with the two channels interleaved per step.
     * Row `s` is therefore stereo channel `s / T` at latent `s % T`. */
    a = h3v_ms();
    if (cb) cb(cbctx, H3V_PHASE_AUDIO, 0, 0, 0.0);
    {
        float *az = h3v_f32(VD_AUDIO * naudio);
        const int64_t alut = naudio / 2;
        for (int64_t s = 0; s < naudio; s++) {
            const int64_t sc = alut > 0 ? s / alut : 0;
            const int64_t u = alut > 0 ? s % alut : s;
            for (int64_t c = 0; c < VD_AUDIO; c++)
                az[c * naudio + 2 * u + sc] = arows[s * VD_AUDIO + c];
        }
        int64_t samples = 0;
        float *pcm = h3v_audio_decode(m, az, naudio, &samples, cb, cbctx, err);
        free(az);
        if (!pcm) { free(rgb); goto done; }
        t.audio_vae_ms = h3v_ms() - a;
        t.audio_samples = samples;
        if (cb) cb(cbctx, H3V_PHASE_AUDIO, 0, 0, 1.0);
        CWICK_INFO("video", "phase 5/6 audio_vae: %lld samples @32kHz in %.1f ms",
                   (long long)samples, t.audio_vae_ms);

        /* 6. mux: H.264 video + AAC audio into one mp4. */
        a = h3v_ms();
        if (cb) cb(cbctx, H3V_PHASE_MUX, 0, 0, 0.0);
        *out = h3v_mp4_encode(rgb, frames, cw, ch, pcm, samples, out_len, err);
        t.mux_ms = h3v_ms() - a;
        free(rgb);
        free(pcm);
        if (!*out) goto done;
        if (cb) cb(cbctx, H3V_PHASE_MUX, 0, 0, 1.0);
        CWICK_INFO("video", "phase 6/6 mux: %zu bytes mp4 in %.1f ms",
                   *out_len, t.mux_ms);
    }
    ok = true;

done:
    if (m->dit_open) { cwick_safetensors_close(&m->dit); m->dit_open = 0; }
    free(cond);
    free(vlat); free(vrows); free(arows);
    free(vvel); free(avel);
    free(mod_all); free(final_mod_all);
    free(rope_cos); free(rope_sin);
    free(ref_rgb_scaled); free(ref_lat); free(ref_packed); free(crows);
    free(ref_rgb_vision);
    if (text_tags) h3_text_cond_free(text_tags);
    pthread_mutex_unlock(&m->lock);
    if (!ok) {
        if (out) { free(*out); *out = NULL; }
        if (out_len) *out_len = 0;
        return false;
    }
    t.total_ms = h3v_ms() - t0;
    if (tim) *tim = t;
    return true;
}

bool h3v_generate(h3v_model *m, const char *prompt, int w, int h, int seconds,
                  uint64_t seed, h3v_progress_fn cb, void *cbctx,
                  unsigned char **out, size_t *out_len, h3v_timing *tim,
                  cwick_error *err) {
    return h3v_gen_impl(m, prompt, NULL, NULL, 0, 0, w, h, seconds, seed,
                        0, cb, cbctx, out, out_len, tim, err);
}

bool h3v_generate_i2v(h3v_model *m, const char *prompt,
                      const unsigned char *ref_rgb, int ref_w, int ref_h,
                      int w, int h, int seconds, uint64_t seed,
                      h3v_progress_fn cb, void *cbctx,
                      unsigned char **out, size_t *out_len, h3v_timing *tim,
                      cwick_error *err) {
    return h3v_gen_impl(m, prompt, NULL, ref_rgb, ref_w, ref_h, w, h, seconds,
                        seed, 0, cb, cbctx, out, out_len, tim, err);
}

bool h3v_generate_variant(h3v_model *m, const char *variant, const char *prompt,
                          const unsigned char *ref_rgb, int ref_w, int ref_h,
                          int w, int h, int seconds, uint64_t seed,
                          h3v_progress_fn cb, void *cbctx,
                          unsigned char **out, size_t *out_len,
                          h3v_timing *tim, cwick_error *err) {
    return h3v_gen_impl(m, prompt, variant, ref_rgb, ref_w, ref_h, w, h,
                        seconds, seed, 0, cb, cbctx, out, out_len, tim, err);
}

bool h3v_generate_variant_steps(h3v_model *m, const char *variant,
                                const char *prompt,
                                const unsigned char *ref_rgb, int ref_w, int ref_h,
                                int w, int h, int seconds, uint64_t seed, int steps,
                                h3v_progress_fn cb, void *cbctx,
                                unsigned char **out, size_t *out_len,
                                h3v_timing *tim, cwick_error *err) {
    return h3v_gen_impl(m, prompt, variant, ref_rgb, ref_w, ref_h, w, h,
                        seconds, seed, steps, cb, cbctx, out, out_len, tim, err);
}
