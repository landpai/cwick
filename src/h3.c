/* h3.c — MiniMax-H3-Image inference engine.
 *
 * The checkpoint is a three-stage image pipeline: a Qwen3-VL conditioner, a
 * 50-layer diffusion transformer sampled in six steps, and a ViT VAE decoder.
 * This file owns the weights, the forward passes and the PNG bytes; the HTTP
 * layer only supplies a prompt (plus optional reference PNGs) and consumes a
 * PNG back.
 *
 * Weights are mmap'd from the shipped safetensors shards. Nothing is copied
 * into a private arena, so a generation's memory profile is the same whether
 * it was invoked from the CLI or the API. A single generation holds one
 * mutex for its whole lifetime: at ~126 GiB of resident weights there is no
 * room for a second pipeline on a 128 GiB machine.
 *
 * Comments describe mechanism and measured numbers only.
 */
#include "h3.h"
#include "h3_gpu.h"

#include <Accelerate/Accelerate.h>
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

#include "cwick.h"
#include "json.h"
#include "log.h"
#include "util.h"

/* ------------------------------------------------------------------ */
/* Constants taken from the checkpoint configs                         */
/* ------------------------------------------------------------------ */
#define TE_LAYERS        50      /* hidden_states[50] is the conditioning */
#define TE_HIDDEN        5120
#define TE_HEADS         64
#define TE_KV_HEADS      8
#define TE_HEAD_DIM      128
#define TE_INTERMEDIATE  25600
#define TE_ROPE_THETA    5000000.0f

#define DIT_LAYERS       50
#define DIT_HIDDEN       5376
#define DIT_HEADS        56
#define DIT_HEAD_DIM     128
#define DIT_FFN          14336
#define DIT_QKV          (DIT_HEADS * DIT_HEAD_DIM)   /* 7168 */
#define DIT_TIME_DIM     2688
#define DIT_TIME_HIDDEN  5376
#define DIT_PATCH_DIM    (H3_LATENT_CHANNELS * H3_PATCH * H3_PATCH)
#define H3_ATTN_ROWS     256      /* score rows materialised per GEMM pair */
/* W rows dequantised per tile once the activation stack is tall enough to
 * amortise the fp32 arena. Measured on the DiT shapes at 4103 tokens: 1.07
 * TFLOPS at the 128-row default, 1.71 at 512. */
#define H3_DEQ_TILE      512
#define DIT_ROPE_FREQ    16
#define DIT_ROPE_DIM     (2 * 3 * DIT_ROPE_FREQ)      /* 96 rotated of 128 */
#define DIT_MODALITIES   3        /* (timestep, modality) table stride */
#define DIT_REFINER      2
#define DIT_ADALN_ROWS   (DIT_MODALITIES * 6 * DIT_HIDDEN)

#define VAE_HIDDEN       1024
#define VAE_HEADS        16
#define VAE_HEAD_DIM     64
#define VAE_FFN          4096
#define VAE_LAYERS       24
#define VAE_REGISTERS    4
#define VAE_ROPE_FREQ    8        /* 2 axes x 8, tiled twice -> 32 */
#define VAE_ROPE_DIM     (4 * VAE_ROPE_FREQ)
#define VAE_ROPE_THETA   100.0f

#define VAE_LATENTS      24
#define VAE_SICALS       /* spatial ratio is H3_VAE_RATIO */

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */
static double h3_ms(void) { return (double)cwick_prof_ns() / 1e6; }

static float h3_clamp(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static float *h3_f32(int64_t n) {
    return (float *)cwick_xmalloc((size_t)n * sizeof(float));
}

static void h3_zero(float *p, int64_t n) {
    if (n > 0) memset(p, 0, (size_t)n * sizeof(float));
}

/* Stage-by-stage dump hook. `CWICK_H3_DUMP=<dir>` writes raw little-endian f32
 * planes so every intermediate can be diffed against the reference
 * implementation tensor by tensor. Inert unless the variable is set. */
static bool h3_dump_on(void) {
    const char *dir = getenv("CWICK_H3_DUMP");
    return dir && *dir;
}

static void h3_dump(const char *name, const float *p, int64_t n) {
    const char *dir = getenv("CWICK_H3_DUMP");
    if (!dir || !*dir || !p || n <= 0) return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.f32", dir, name);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(p, sizeof(float), (size_t)n, f);
    fclose(f);
}

/* Per-block dumps for bisecting a deep stack. Off unless requested: a 50-layer
 * walk writes 50 files, which is only wanted while chasing a divergence. */
static bool h3_dump_blocks(void) {
    const char *v = getenv("CWICK_H3_DUMP_BLOCKS");
    return v && *v;
}

/* Token ids and the two tag vectors are integers, so they get their own writer
 * rather than being pushed through the f32 hook. */
static void h3_dump_i64(const char *name, const int64_t *p, int64_t n) {
    const char *dir = getenv("CWICK_H3_DUMP");
    if (!dir || !*dir || !p || n <= 0) return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.i64", dir, name);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(p, sizeof(int64_t), (size_t)n, f);
    fclose(f);
}

/* A weight matrix. `rows` x `cols` follows the checkpoint's own shape; all
 * projections consume it as torch does: out = in @ W^T. `f32` distinguishes
 * the few FP32 tensors in this otherwise BF16 checkpoint. */
typedef struct {
    const void *p;
    int64_t rows, cols;
    bool f32;
} h3_mat;

static bool h3_mat_get(const cwick_safetensors *saf, const char *name,
                       h3_mat *out, cwick_error *err) {
    cwick_st_tensor t;
    if (cwick_safetensors_get(saf, name, &t, err) != CWICK_OK) return false;
    out->p = t.data;
    out->rows = t.rows;
    out->cols = t.cols;
    out->f32 = t.dtype == CWICK_ST_F32;
    return true;
}

/* The checkpoint is BF16 for every tensor except the few modules pinned to
 * float32, so a norm weight or a bias that is read straight out of the mapping
 * is not a float and cannot be dereferenced as one. Materialise vectors. */
static float *h3_vec(const h3_mat *m) {
    int64_t n = m->rows * m->cols;
    float *v = h3_f32(n);
    if (m->f32) {
        memcpy(v, m->p, (size_t)n * sizeof(float));
    } else {
        const uint16_t *s = (const uint16_t *)m->p;
        for (int64_t i = 0; i < n; i++) v[i] = cwick_bf16_fast(s[i]);
    }
    return v;
}

/* ------------------------------------------------------------------ */
/* Model                                                               */
/* ------------------------------------------------------------------ */
static int h3_gpu_on(void);

struct h3_model {
    char *root;
    cwick_safetensors te;         /* text_encoder/  (model.safetensors.index.json) */
    cwick_safetensors dit;        /* transformer/   (model.safetensors.index.json) */
    int te_open, dit_open;
    /* The VAE and the fused PDD heads ship as single shards with no index. */
    const void *vae_map;
    size_t vae_len;
    const void *pdd_map;
    size_t pdd_len;
    cwick_tokenizer *tok;
    pthread_mutex_t lock;
    bool ready;
};

static char *h3_join(const char *a, const char *b) {
    size_t n = strlen(a) + strlen(b) + 2;
    char *s = (char *)cwick_xmalloc(n);
    cwick_path_join(s, n, a, b);
    return s;
}

/* Locate one flat F32 tensor in a standalone safetensors file. The result
 * aliases the mapping rather than a copy: the decoder's weights are read-only,
 * and the tiled decode walks all 403M of them once per tile. Copying cost a
 * full 1.6 GiB sweep per tile, 121 of them, and nothing was ever written. */
static bool h3_standalone_f32(const void *map, size_t len, const char *name,
                              const float **out, int64_t *n, cwick_error *err) {
    if (len < 16) {
        cwick_error_set(err, CWICK_ERR_IO, "standalone safetensors too small");
        return false;
    }
    uint64_t hlen = 0;
    memcpy(&hlen, map, 8);
    if (hlen > len - 8) {
        cwick_error_set(err, CWICK_ERR_IO, "standalone safetensors header out of range");
        return false;
    }
    const char *hdr = (const char *)map + 8;
    json_value *root = cwick_json_parse(hdr, (size_t)hlen, err);
    if (!root) return false;
    json_value *v = cwick_json_get(root, name);
    if (!v || v->kind != JSON_OBJECT) {
        cwick_json_free(root);
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "tensor %s not found", name);
        return false;
    }
    json_value *dt = cwick_json_get(v, "dtype");
    json_value *sh = cwick_json_get(v, "shape");
    json_value *of = cwick_json_get(v, "data_offsets");
    if (!dt || !sh || !of || sh->kind != JSON_ARRAY || of->kind != JSON_ARRAY ||
        strcmp(cwick_json_str(dt), "F32") != 0 || sh->count < 1 ||
        of->count < 2) {
        cwick_json_free(root);
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "tensor %s has unexpected dtype/shape", name);
        return false;
    }
    int64_t count = 1;
    for (size_t i = 0; i < sh->count; i++)
        count *= cwick_json_int(sh->items[i], 1);
    int64_t b0 = cwick_json_int(of->items[0], 0);
    int64_t b1 = cwick_json_int(of->items[1], 0);
    if (b0 < 0 || b1 < b0 || (uint64_t)b1 > len - 8 - hlen) {
        cwick_json_free(root);
        cwick_error_set(err, CWICK_ERR_IO, "tensor %s offsets out of range", name);
        return false;
    }
    const char *base = (const char *)map + 8 + hlen + b0;
    if ((size_t)(b1 - b0) != (size_t)count * sizeof(float) ||
        ((uintptr_t)base & 3u) != 0) {
        cwick_json_free(root);
        cwick_error_set(err, CWICK_ERR_IO, "tensor %s is not a flat aligned F32 block", name);
        return false;
    }
    cwick_json_free(root);
    *out = (const float *)base;
    *n = count;
    return true;
}
h3_model *h3_model_open(const char *root, cwick_error *err) {
    h3_model *m = (h3_model *)cwick_xmalloc(sizeof(*m));
    memset(m, 0, sizeof(*m));
    m->root = cwick_strdup(root);
    pthread_mutex_init(&m->lock, NULL);

    char *te = h3_join(root, "text_encoder");
    /* The diffusers layout is the one the port is verified against: its
     * projections are already split (to_q/to_k/to_v) and its SwiGLU stores
     * `[value; gate]`, so no load-time reordering is needed. The native
     * checkpoint keeps a per-head-interleaved fused qkv. */
    char *dit = h3_join(root, "transformer_diffusers");
    char *vae = h3_join(root, "vae_diffusers/diffusion_pytorch_model.safetensors");
    char *pdd = h3_join(root, "pdd_fused_heads.safetensors");

    if (cwick_safetensors_open(te, &m->te, err) != CWICK_OK) goto fail;
    m->te_open = 1;
    if (cwick_safetensors_open(dit, &m->dit, err) != CWICK_OK) goto fail;
    m->dit_open = 1;

    /* The transformer is re-read in full on every sampling step, so a reclaimed
     * weight page costs an SSD round trip inside the sampling loop. The text
     * encoder is read once per request and is deliberately left reclaimable:
     * wiring both checkpoints would ask for 124 GiB on a 128 GiB machine. */
    if (cwick_env_true("CWICK_H3_WIRE")) {
        uint64_t t0 = cwick_prof_ns();
        uint64_t got = cwick_safetensors_make_resident(&m->dit, NULL, 1);
        double ms = (double)(cwick_prof_ns() - t0) / 1e6;
        CWICK_INFO("init", "h3 transformer weights resident: %.2f GiB %s — "
                   "%.0f ms (%.1f GB/s)",
                   (double)got / 1073741824.0,
                   m->dit.pinned_wired ? "faulted in and wired"
                                       : "faulted in (not wired)",
                   ms, ms > 0.0 ? (double)got / (ms * 1e6) : 0.0);
    }

    /* Hand the GPU one view per transformer shard up front. A sampling step
     * runs a few hundred projections over the same 61 GiB; without this each
     * one faults its own pages and keeps its own handle, and step time grows
     * with the number of steps already taken. */
    if (h3_gpu_on()) {
        for (size_t i = 0; i < m->dit.map_count; i++)
            h3_gpu_adopt(m->dit.maps[i], m->dit.map_lens[i]);
    }

    size_t n = 0;
    m->vae_map = cwick_mmap_file(vae, &m->vae_len);
    if (!m->vae_map) {
        cwick_error_set(err, CWICK_ERR_IO, "cannot map %s", vae);
        goto fail;
    }
    m->pdd_map = cwick_mmap_file(pdd, &m->pdd_len);
    if (!m->pdd_map) {
        cwick_error_set(err, CWICK_ERR_IO, "cannot map %s", pdd);
        goto fail;
    }
    (void)n;

    cwick_model_spec spec;
    memset(&spec, 0, sizeof(spec));
    snprintf(spec.model_type, sizeof(spec.model_type), "%s", "qwen3_vl");
    char *tokdir = h3_join(root, "tokenizer");
    if (cwick_tokenizer_load(tokdir, &spec, &m->tok, err) != CWICK_OK) goto fail;

    free(te); free(dit); free(vae); free(pdd); free(tokdir);
    m->ready = true;
    return m;

fail:
    free(te); free(dit); free(vae); free(pdd);
    h3_model_close(m);
    return NULL;
}

void h3_model_close(h3_model *m) {
    if (!m) return;
    if (m->te_open) cwick_safetensors_close(&m->te);
    if (m->dit_open) cwick_safetensors_close(&m->dit);
    if (m->vae_map) cwick_munmap(m->vae_map, m->vae_len);
    if (m->pdd_map) cwick_munmap(m->pdd_map, m->pdd_len);
    if (m->tok) cwick_tokenizer_free(m->tok);
    pthread_mutex_destroy(&m->lock);
    free(m->root);
    free(m);
}

const char *h3_model_dir(const h3_model *m) { return m ? m->root : NULL; }
bool h3_model_ready(const h3_model *m) { return m && m->ready; }

/* ------------------------------------------------------------------ */
/* PNG: ImageIO is a C API, so no Objective-C translation unit is needed */
/* ------------------------------------------------------------------ */
void h3_png_free(unsigned char *png) { free(png); }

unsigned char *h3_png_encode(const float *rgb, int w, int h,
                                    size_t *out_len, cwick_error *err) {
    if (w <= 0 || h <= 0) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "bad PNG dimensions");
        return NULL;
    }
    size_t n = (size_t)w * (size_t)h;
    /* Quartz does not accept 24-bpp RGB contexts; store RGBA and let PNG
     * drop the fully opaque alpha channel on the way out. */
    unsigned char *bytes = (unsigned char *)cwick_xmalloc(n * 4);
    for (size_t i = 0; i < n; i++) {
        for (int c = 0; c < 3; c++)
            bytes[i * 4 + c] = (unsigned char)(h3_clamp(rgb[i * 3 + c], 0.0f, 1.0f) * 255.0f);
        bytes[i * 4 + 3] = 255;
    }

    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(bytes, (size_t)w, (size_t)h, 8,
                                             (size_t)w * 4, cs,
                                             kCGImageAlphaNoneSkipLast);
    CGColorSpaceRelease(cs);
    if (!ctx) {
        free(bytes);
        cwick_error_set(err, CWICK_ERR_STATE, "CGBitmapContextCreate failed");
        return NULL;
    }
    CGImageRef img = CGBitmapContextCreateImage(ctx);
    CGContextRelease(ctx);
    if (!img) {
        free(bytes);
        cwick_error_set(err, CWICK_ERR_STATE, "CGBitmapContextCreateImage failed");
        return NULL;
    }

    CFMutableDataRef data = CFDataCreateMutable(kCFAllocatorDefault, 0);
    CGImageDestinationRef dst =
        CGImageDestinationCreateWithData(data, CFSTR("public.png"), 1, NULL);
    if (!dst) {
        CGImageRelease(img); CFRelease(data); free(bytes);
        cwick_error_set(err, CWICK_ERR_STATE, "CGImageDestinationCreateWithData failed");
        return NULL;
    }
    CGImageDestinationAddImage(dst, img, NULL);
    bool ok = CGImageDestinationFinalize(dst);
    CGImageRelease(img);
    CFRelease(dst);
    if (!ok || CFDataGetLength(data) <= 0) {
        CFRelease(data); free(bytes);
        cwick_error_set(err, CWICK_ERR_STATE, "PNG encode failed");
        return NULL;
    }
    size_t len = (size_t)CFDataGetLength(data);
    unsigned char *out = (unsigned char *)cwick_xmalloc(len);
    memcpy(out, CFDataGetBytePtr(data), len);
    CFRelease(data);
    free(bytes);
    *out_len = len;
    return out;
}

bool h3_png_decode(const unsigned char *png, size_t len, float **rgb,
                   int *w, int *h, cwick_error *err) {
    if (!png || len < 16) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "empty PNG");
        return false;
    }
    CFDataRef data = CFDataCreate(kCFAllocatorDefault, (const UInt8 *)png, (CFIndex)len);
    CGImageSourceRef src = CGImageSourceCreateWithData(data, NULL);
    CFRelease(data);
    if (!src) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "PNG decode failed");
        return false;
    }
    CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, NULL);
    CFRelease(src);
    if (!img) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "PNG has no image");
        return false;
    }
    int iw = (int)CGImageGetWidth(img), ih = (int)CGImageGetHeight(img);
    if (iw <= 0 || ih <= 0) {
        CGImageRelease(img);
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "bad PNG dimensions");
        return false;
    }
    size_t n = (size_t)iw * (size_t)ih;
    unsigned char *bytes = (unsigned char *)cwick_xmalloc(n * 4);
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(bytes, (size_t)iw, (size_t)ih, 8,
                                             (size_t)iw * 4, cs,
                                             kCGImageAlphaNoneSkipLast);
    CGColorSpaceRelease(cs);
    if (!ctx) {
        CGImageRelease(img); free(bytes);
        cwick_error_set(err, CWICK_ERR_STATE, "decode context failed");
        return false;
    }
    CGRect rect = CGRectMake(0, 0, (CGFloat)iw, (CGFloat)ih);
    CGContextDrawImage(ctx, rect, img);
    CGContextRelease(ctx);
    CGImageRelease(img);

    float *out = h3_f32((int64_t)n * 3);
    for (size_t i = 0; i < n; i++)
        for (int c = 0; c < 3; c++) out[i * 3 + c] = bytes[i * 4 + c] / 255.0f;
    free(bytes);
    *rgb = out; *w = iw; *h = ih;
    return true;
}

/* ------------------------------------------------------------------ */
/* Numerics                                                            */
/* ------------------------------------------------------------------ */
/* Opt-in phase timer (`CWICK_H3_PROF=1`). The byte and MAC counters separate a
 * step limited by weight traffic from one limited by arithmetic: the first is
 * flat in the sequence length, the second is not. */
static double h3_prof_p[6];
static double h3_prof_bytes, h3_prof_macs;
static bool h3_prof_on(void) {
    static int cached = -1;
    if (cached < 0) cached = cwick_env_true("CWICK_H3_PROF") ? 1 : 0;
    return cached != 0;
}

/* The GPU path is the whole reason the image pipeline has a backend switch:
 * a sampling step moves 66 GB of BF16 weights, and on the CPU that traffic is
 * tripled by the dequantise-to-f32 arena it has to go through first. Off unless
 * the Metal backend is the selected one, and overridable for A/B runs. */
static int h3_gpu_on(void) {
    static int cached = -1;
    if (cached < 0) {
        if (!cwick_metal_backend_active()) {
            cached = 0;
        } else if (getenv("CWICK_H3_GPU") && !cwick_env_true("CWICK_H3_GPU")) {
            cached = 0;
        } else {
            cached = h3_gpu_ready() ? 1 : 0;
        }
    }
    return cached;
}

static void h3_linear(float *out, const float *x, h3_mat w, const float *bias,
                      int64_t k, int64_t n) {
    /* out[n, w.rows] = x[n, k] @ w[k, w.rows]^T */
    if (h3_prof_on()) {
        h3_prof_bytes += (double)w.rows * (double)k * (w.f32 ? 4.0 : 2.0);
        h3_prof_macs += (double)n * (double)k * (double)w.rows;
    }
    /* A projection thinner than one output tile spends more on the dispatch
     * and the page walk than the tile buys back, and that is exactly what the
     * per-timestep modulation and the time embedding are. */
    if (!w.f32 && h3_gpu_on() && n >= 64 && n * k * w.rows >= (int64_t)8 * 1024 * 1024 &&
        h3_gpu_gemm(x, w.p, out, (int)n, (int)k, (int)w.rows, (int)k, (int)w.rows)) {
        if (bias) {
            for (int64_t r = 0; r < n; r++)
                for (int64_t c = 0; c < w.rows; c++) out[r * w.rows + c] += bias[c];
        }
        return;
    }
    if (w.f32) cwick_gemm_f32(out, n, k, w.rows, x, (const float *)w.p);
    else if (n >= 1024) cwick_gemm_bf16_tile(out, n, k, w.rows, x, w.p, H3_DEQ_TILE);
    else       cwick_gemm_bf16(out, n, k, w.rows, x, w.p);
    if (bias) {
        for (int64_t r = 0; r < n; r++)
            for (int64_t c = 0; c < w.rows; c++) out[r * w.rows + c] += bias[c];
    }
}

static void h3_rmsnorm(float *out, const float *x, const float *w,
                       int64_t d, float eps) {
    float ss = 0.0f;
    for (int64_t i = 0; i < d; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / (float)d + eps);
    for (int64_t i = 0; i < d; i++) out[i] = x[i] * scale * w[i];
}

/* Per-head RMSNorm used by the text encoder, the DiT and the VAE decoder.
 * `x` is [tokens, heads, hd]; every (token, head) pair is normalised on its
 * own, so the walk has to advance through both axes. */
static void h3_rmsnorm_heads(float *x, const float *w, int64_t tokens,
                             int64_t heads, int64_t hd, float eps) {
    int64_t hdall = heads * hd;
    for (int64_t s = 0; s < tokens; s++) {
        for (int64_t h = 0; h < heads; h++) {
            float *v = x + s * hdall + h * hd;
            float ss = 0.0f;
            for (int64_t i = 0; i < hd; i++) ss += v[i] * v[i];
            float sc = 1.0f / sqrtf(ss / (float)hd + eps);
            for (int64_t i = 0; i < hd; i++) v[i] = v[i] * sc * w[i];
        }
    }
}

static void h3_silu(float *x, int64_t n) {
    for (int64_t i = 0; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}

/* Causal or full self-attention over one sequence, as two GEMMs per head.
 * q,k,v are [seq, heads, head_dim], so one head is a strided view of stride
 * `heads * head_dim` and cblas_sgemm reads it in place — no transposes.
 *
 * The hand-written triple loop this replaces measured 4.2 GFLOPS on the DiT
 * shapes (12.0 s of every 20.8 s step at seq 263) because it streamed a whole
 * key row per score. The same shapes through cblas_sgemm run at 1.5 TFLOPS,
 * which is the difference between a 4 MP canvas being tractable or not. */
static int h3_gpu_on(void);

static void h3_attention(float *out, const float *q, const float *k,
                         const float *v, int64_t seq, int64_t heads,
                         int64_t hd, bool causal, float *scratch) {
    /* The streaming GPU kernel is non-causal; the token refiner's causal use
     * stays on the CPU, where the sequence is short anyway. */
    if (!causal && h3_gpu_on() &&
        h3_gpu_attention(q, k, v, out, (int)seq, (int)(heads * hd), (int)heads))
        return;
    const int64_t ld = heads * hd;
    const float scale = 1.0f / sqrtf((float)hd);
    for (int64_t h = 0; h < heads; h++) {
        const float *qh = q + h * hd, *kh = k + h * hd, *vh = v + h * hd;
        float *oh = out + h * hd;
        for (int64_t r0 = 0; r0 < seq; r0 += H3_ATTN_ROWS) {
            int64_t rn = seq - r0 < H3_ATTN_ROWS ? seq - r0 : H3_ATTN_ROWS;
            /* A causal block never reads a key past its own last row. */
            int64_t cols = causal ? r0 + rn : seq;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        (int)rn, (int)cols, (int)hd,
                        scale, qh + r0 * ld, (int)ld, kh, (int)ld,
                        0.0f, scratch, (int)cols);
            float *sums = scratch + rn * cols;   /* one normaliser per row */
            for (int64_t i = 0; i < rn; i++) {
                float *row = scratch + i * cols;
                int64_t valid = causal ? (r0 + i + 1) : seq;
                int n = (int)valid;
                float mx = -INFINITY;
                vDSP_maxv(row, 1, &mx, (vDSP_Length)n);
                /* Subtract the row max before exp: a raw score above 88
                 * overflows fp32 and turns the row into NaN. */
                float neg = -mx;
                vDSP_vsadd(row, 1, &neg, row, 1, (vDSP_Length)n);
                vvexpf(row, row, &n);
                float sum = 0.0f;
                vDSP_sve(row, 1, &sum, (vDSP_Length)valid);
                sums[i] = sum > 0.0f ? 1.0f / sum : 0.0f;
                /* A causal block computes a prefix of its column range; the
                 * rest must contribute nothing to the value product. */
                for (int64_t j = valid; j < cols; j++) row[j] = 0.0f;
            }
            /* Normalising the probabilities would be a full pass over the
             * score block; folding 1/sum into the 128-wide output instead
             * costs 32x less memory traffic for the same result. */
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        (int)rn, (int)hd, (int)cols,
                        1.0f, scratch, (int)cols, vh, (int)ld,
                        0.0f, oh + r0 * ld, (int)ld);
            for (int64_t i = 0; i < rn; i++) {
                float *oi = oh + (r0 + i) * ld;
                vDSP_vsmul(oi, 1, &sums[i], oi, 1, (vDSP_Length)hd);
            }
        }
    }
}

/* rotate_half on the leading rotary_dim channels of every head. */
static void h3_rope(float *x, const float *cos, const float *sin,
                    int64_t seq, int64_t heads, int64_t hd, int64_t rd) {
    int64_t half = rd / 2;
    for (int64_t s = 0; s < seq; s++) {
        for (int64_t h = 0; h < heads; h++) {
            float *v = x + s * heads * hd + h * hd;
            for (int64_t i = 0; i < half; i++) {
                float a = v[i], b = v[i + half];
                v[i] = a * cos[s * rd + i] - b * sin[s * rd + i];
                v[i + half] = b * cos[s * rd + i + half] + a * sin[s * rd + i + half];
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Qwen3-VL vision tower                                               */
/*                                                                     */
/* 27 pre-norm blocks over 16-px patches of one frame, 2D axial RoPE,  */
/* and a 2x2 patch merger that projects to the conditioner's 5120-wide */
/* hidden size. Three extra mergers at blocks 8, 16 and 24 produce the */
/* deepstack features the conditioner adds back at its first three     */
/* layers. A still image occupies both temporal taps of the patch-     */
/* embedding conv3d, so the two taps fold into a single 16x16 kernel.  */
/* ------------------------------------------------------------------ */
#define VIS_DEPTH   27
#define VIS_HIDDEN  1152
#define VIS_HEADS   16
#define VIS_HEAD_DIM 72
#define VIS_INTER   4304
#define VIS_PATCH   16
#define VIS_MERGE   2
#define VIS_POS_SIDE 48          /* sqrt(num_position_embeddings = 2304) */
#define VIS_ROPE_N  18           /* head_dim / 4 octaves per axis */
#define VIS_MERGE_DIM (VIS_HIDDEN * VIS_MERGE * VIS_MERGE)   /* 4608 */
#define VIS_OUT     5120
#define VIS_DS      3

typedef struct {
    h3_mat qkv, qkvb, proj, projb;
    h3_mat fc1, fc1b, fc2, fc2b;
    h3_mat n1w, n1b, n2w, n2b;
} h3_vis_block;

typedef struct {
    h3_mat norm_w, norm_b, fc1, fc1b, fc2, fc2b;
} h3_vis_merger;

typedef struct {
    float *merged;      /* [gh*gw/4][VIS_OUT], in spatial-merge-block order */
    float *ds[VIS_DS];  /* one per deepstack tap, same shape */
    int64_t gh, gw;
} h3_vis_out;

/* LayerNorm over the last axis of an [n][d] block. */
static void h3_layernorm(float *out, const float *x, const float *w,
                         const float *b, int64_t n, int64_t d, float eps) {
    for (int64_t i = 0; i < n; i++) {
        const float *s = x + i * d;
        float *o = out + i * d;
        float mean = 0.0f;
        for (int64_t j = 0; j < d; j++) mean += s[j];
        mean /= (float)d;
        float var = 0.0f;
        for (int64_t j = 0; j < d; j++) {
            const float t = s[j] - mean;
            var += t * t;
        }
        const float inv = 1.0f / sqrtf(var / (float)d + eps);
        for (int64_t j = 0; j < d; j++)
            o[j] = (s[j] - mean) * inv * w[j] + b[j];
    }
}

static float h3_gelu_erf(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678f));
}

/* The tower's MLP activation is `gelu_pytorch_tanh`; the mergers use the
 * exact erf form, so the two are not interchangeable. */
static float h3_gelu_tanh(float x) {
    const float c = 0.7978845608f;      /* sqrt(2/pi) */
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

/* Patch index -> (row, col), in the spatial-merge-block order the processor
 * emits and `get_vision_position_ids` assumes. */
static void h3_vis_patch_rc(int64_t p, int64_t gw, int64_t *row, int64_t *col) {
    const int64_t blocks_w = gw / VIS_MERGE;
    const int64_t in_col = p % VIS_MERGE;
    const int64_t in_row = (p / VIS_MERGE) % VIS_MERGE;
    const int64_t block_col = (p / (VIS_MERGE * VIS_MERGE)) % blocks_w;
    const int64_t block_row = p / (VIS_MERGE * VIS_MERGE * blocks_w);
    *row = block_row * VIS_MERGE + in_row;
    *col = block_col * VIS_MERGE + in_col;
}

#define VIS_GET(var, ...) do {                                                \
    char n_[192];                                                             \
    snprintf(n_, sizeof(n_), __VA_ARGS__);                                    \
    if (!h3_mat_get(&m->te, n_, &(var), err)) return false;                   \
} while (0)

static bool h3_vis_load(h3_model *m, h3_vis_block *blk, h3_vis_merger *mg,
                        h3_vis_merger *ds, h3_mat *pe, h3_mat *pe_b,
                        h3_mat *pos, cwick_error *err) {
    for (int64_t L = 0; L < VIS_DEPTH; L++) {
        h3_vis_block *b = &blk[L];
        const long long l = (long long)L;
        VIS_GET(b->qkv, "model.visual.blocks.%lld.attn.qkv.weight", l);
        VIS_GET(b->qkvb, "model.visual.blocks.%lld.attn.qkv.bias", l);
        VIS_GET(b->proj, "model.visual.blocks.%lld.attn.proj.weight", l);
        VIS_GET(b->projb, "model.visual.blocks.%lld.attn.proj.bias", l);
        VIS_GET(b->fc1, "model.visual.blocks.%lld.mlp.linear_fc1.weight", l);
        VIS_GET(b->fc1b, "model.visual.blocks.%lld.mlp.linear_fc1.bias", l);
        VIS_GET(b->fc2, "model.visual.blocks.%lld.mlp.linear_fc2.weight", l);
        VIS_GET(b->fc2b, "model.visual.blocks.%lld.mlp.linear_fc2.bias", l);
        VIS_GET(b->n1w, "model.visual.blocks.%lld.norm1.weight", l);
        VIS_GET(b->n1b, "model.visual.blocks.%lld.norm1.bias", l);
        VIS_GET(b->n2w, "model.visual.blocks.%lld.norm2.weight", l);
        VIS_GET(b->n2b, "model.visual.blocks.%lld.norm2.bias", l);
    }
    VIS_GET(mg->norm_w, "model.visual.merger.norm.weight");
    VIS_GET(mg->norm_b, "model.visual.merger.norm.bias");
    VIS_GET(mg->fc1, "model.visual.merger.linear_fc1.weight");
    VIS_GET(mg->fc1b, "model.visual.merger.linear_fc1.bias");
    VIS_GET(mg->fc2, "model.visual.merger.linear_fc2.weight");
    VIS_GET(mg->fc2b, "model.visual.merger.linear_fc2.bias");
    for (int64_t d = 0; d < VIS_DS; d++) {
        h3_vis_merger *t = &ds[d];
        const long long i = (long long)d;
        VIS_GET(t->norm_w, "model.visual.deepstack_merger_list.%lld.norm.weight", i);
        VIS_GET(t->norm_b, "model.visual.deepstack_merger_list.%lld.norm.bias", i);
        VIS_GET(t->fc1, "model.visual.deepstack_merger_list.%lld.linear_fc1.weight", i);
        VIS_GET(t->fc1b, "model.visual.deepstack_merger_list.%lld.linear_fc1.bias", i);
        VIS_GET(t->fc2, "model.visual.deepstack_merger_list.%lld.linear_fc2.weight", i);
        VIS_GET(t->fc2b, "model.visual.deepstack_merger_list.%lld.linear_fc2.bias", i);
    }
    VIS_GET(*pe, "model.visual.patch_embed.proj.weight");
    VIS_GET(*pe_b, "model.visual.patch_embed.proj.bias");
    VIS_GET(*pos, "model.visual.pos_embed.weight");
    return true;
}
#undef VIS_GET

/* `x` is `n` rows of the merged 4608-wide vector. The image merger normalises
 * each 1152-wide token before grouping them, so it passes `pre_normed`; the
 * deepstack mergers normalise the grouped vector themselves. */
static void h3_vis_merger_run(const h3_vis_merger *t, const float *x, int64_t n,
                              bool pre_normed, float *out) {
    float *h = h3_f32(n * VIS_MERGE_DIM);
    if (pre_normed) {
        memcpy(h, x, (size_t)(n * VIS_MERGE_DIM) * sizeof(float));
    } else {
        float *w = h3_vec(&t->norm_w), *b = h3_vec(&t->norm_b);
        h3_layernorm(h, x, w, b, n, VIS_MERGE_DIM, 1e-6f);
        free(w); free(b);
    }
    float *mid = h3_f32(n * VIS_MERGE_DIM);
    h3_linear(mid, h, t->fc1, NULL, VIS_MERGE_DIM, n);
    {
        float *b = h3_vec(&t->fc1b);
        for (int64_t i = 0; i < n; i++)
            for (int64_t c = 0; c < VIS_MERGE_DIM; c++)
                mid[i * VIS_MERGE_DIM + c] += b[c];
        free(b);
    }
    for (int64_t i = 0; i < n * VIS_MERGE_DIM; i++) mid[i] = h3_gelu_erf(mid[i]);
    h3_linear(out, mid, t->fc2, NULL, VIS_MERGE_DIM, n);
    {
        float *b = h3_vec(&t->fc2b);
        for (int64_t i = 0; i < n; i++)
            for (int64_t c = 0; c < VIS_OUT; c++) out[i * VIS_OUT + c] += b[c];
        free(b);
    }
    free(h);
    free(mid);
}

static bool h3_vis_run(h3_model *m, const float *rgb, int64_t W, int64_t H,
                       h3_vis_out *vo, cwick_error *err) {
    const int64_t gh = H / VIS_PATCH, gw = W / VIS_PATCH;
    const int64_t n = gh * gw;
    if (n <= 0 || H % VIS_PATCH || W % VIS_PATCH || gw % VIS_MERGE) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "vision grid %lldx%lld is not whole patches",
                        (long long)gw, (long long)gh);
        return false;
    }
    const int64_t nmerge = n / (VIS_MERGE * VIS_MERGE);
    vo->merged = h3_f32(nmerge * VIS_OUT);
    for (int64_t d = 0; d < VIS_DS; d++) vo->ds[d] = h3_f32(nmerge * VIS_OUT);

    h3_vis_block *blk = cwick_xmalloc(sizeof(*blk) * VIS_DEPTH);
    h3_vis_merger mg, ds[VIS_DS];
    h3_mat pe, pe_b, pos;
    memset(blk, 0, sizeof(*blk) * VIS_DEPTH);
    memset(&mg, 0, sizeof(mg));
    memset(ds, 0, sizeof(ds));
    if (!h3_vis_load(m, blk, &mg, ds, &pe, &pe_b, &pos, err)) {
        free(blk);
        free(vo->merged);
        for (int64_t d = 0; d < VIS_DS; d++) free(vo->ds[d]);
        vo->merged = NULL;
        for (int64_t d = 0; d < VIS_DS; d++) vo->ds[d] = NULL;
        return false;
    }
    float *q = NULL, *k = NULL, *v = NULL, *att = NULL, *nrm = NULL;
    float *po = NULL, *g1 = NULL, *x = NULL, *scores = NULL;
    float *cos = NULL, *sin = NULL, *fold = NULL, *a = NULL;
    float *tab = NULL;

    /* Patch embedding. `kernel = stride = (2, 16, 16)` over `[C][T][H][W]`, and
     * both temporal taps hold the same frame, so the two taps add up into one
     * 16x16 kernel over 768 input values. */
    const int64_t kdim = 3 * VIS_PATCH * VIS_PATCH;
    fold = h3_f32((int64_t)VIS_HIDDEN * kdim);
    {
        const uint16_t *w = (const uint16_t *)pe.p;
        const int64_t per = 3 * 2 * VIS_PATCH * VIS_PATCH;
        for (int64_t o = 0; o < VIS_HIDDEN; o++)
            for (int64_t c = 0; c < 3; c++)
                for (int64_t y = 0; y < VIS_PATCH; y++)
                    for (int64_t xx = 0; xx < VIS_PATCH; xx++) {
                        const int64_t kk = (c * VIS_PATCH + y) * VIS_PATCH + xx;
                        const int64_t base = o * per + (c * 2) * (VIS_PATCH * VIS_PATCH)
                                             + y * VIS_PATCH + xx;
                        fold[o * kdim + kk] =
                            cwick_bf16_fast(w[base]) +
                            cwick_bf16_fast(w[base + VIS_PATCH * VIS_PATCH]);
                    }
    }
    a = h3_f32(n * kdim);
    for (int64_t p = 0; p < n; p++) {
        int64_t row = 0, col = 0;
        h3_vis_patch_rc(p, gw, &row, &col);
        float *d = a + p * kdim;
        for (int64_t y = 0; y < VIS_PATCH; y++)
            for (int64_t xx = 0; xx < VIS_PATCH; xx++) {
                const float *s = rgb + (((row * VIS_PATCH + y) * W) +
                                        col * VIS_PATCH + xx) * 3;
                for (int64_t c = 0; c < 3; c++)
                    /* The processor normalises with mean = std = 0.5. */
                    d[(c * VIS_PATCH + y) * VIS_PATCH + xx] = s[c] * 2.0f - 1.0f;
            }
    }
    x = h3_f32(n * VIS_HIDDEN);
    {
        h3_mat wm = {fold, VIS_HIDDEN, kdim, true};
        h3_linear(x, a, wm, NULL, kdim, n);
        float *bb = h3_vec(&pe_b);
        for (int64_t i = 0; i < n; i++)
            for (int64_t c = 0; c < VIS_HIDDEN; c++) x[i * VIS_HIDDEN + c] += bb[c];
        free(bb);
    }
    free(a); a = NULL;
    free(fold); fold = NULL;
    h3_dump("c_vis_pe", x, n * VIS_HIDDEN);

    /* Learned square position table, resampled bilinearly with
     * `align_corners=True` and clamped taps. */
    tab = h3_vec(&pos);
    for (int64_t r = 0; r < gh; r++) {
        const float src_h = (float)r * (float)(VIS_POS_SIDE - 1) /
                            (float)(gh > 1 ? gh - 1 : 1);
        int64_t h0 = (int64_t)floorf(src_h);
        const float fh = src_h - (float)h0;
        int64_t h1 = h0 + 1;
        if (h0 > VIS_POS_SIDE - 1) h0 = VIS_POS_SIDE - 1;
        if (h1 > VIS_POS_SIDE - 1) h1 = VIS_POS_SIDE - 1;
        for (int64_t c = 0; c < gw; c++) {
            const float src_w = (float)c * (float)(VIS_POS_SIDE - 1) /
                                (float)(gw > 1 ? gw - 1 : 1);
            int64_t w0 = (int64_t)floorf(src_w);
            const float fw = src_w - (float)w0;
            int64_t w1 = w0 + 1;
            if (w0 > VIS_POS_SIDE - 1) w0 = VIS_POS_SIDE - 1;
            if (w1 > VIS_POS_SIDE - 1) w1 = VIS_POS_SIDE - 1;
            const int64_t br = r / VIS_MERGE, ir = r % VIS_MERGE;
            const int64_t bc = c / VIS_MERGE, ic = c % VIS_MERGE;
            const int64_t p = ((br * (gw / VIS_MERGE) + bc) * VIS_MERGE + ir) *
                              VIS_MERGE + ic;
            float *dst = x + p * VIS_HIDDEN;
            const float wh[2] = {1.0f - fh, fh};
            const float ww[2] = {1.0f - fw, fw};
            const int64_t hi[2] = {h0, h1}, wi[2] = {w0, w1};
            for (int64_t i = 0; i < 2; i++)
                for (int64_t j = 0; j < 2; j++) {
                    const float wt = wh[i] * ww[j];
                    if (wt == 0.0f) continue;
                    const float *s = tab + (hi[i] * VIS_POS_SIDE + wi[j]) * VIS_HIDDEN;
                    for (int64_t d = 0; d < VIS_HIDDEN; d++) dst[d] += wt * s[d];
                }
        }
    }
    free(tab); tab = NULL;
    h3_dump("c_vis_x0", x, n * VIS_HIDDEN);

    /* 2D axial RoPE: 18 octaves per axis, concatenated into 36 values and
     * repeated, which is the whole 72-channel head. */
    cos = h3_f32(n * VIS_HEAD_DIM);
    sin = h3_f32(n * VIS_HEAD_DIM);
    {
        float inv[VIS_ROPE_N];
        for (int64_t j = 0; j < VIS_ROPE_N; j++)
            inv[j] = powf(1.0f / 10000.0f, (float)(2 * j) / 36.0f);
        for (int64_t p = 0; p < n; p++) {
            int64_t row = 0, col = 0;
            h3_vis_patch_rc(p, gw, &row, &col);
            for (int64_t j = 0; j < VIS_ROPE_N; j++) {
                const float ch = cosf((float)row * inv[j]);
                const float sh = sinf((float)row * inv[j]);
                const float cw = cosf((float)col * inv[j]);
                const float sw = sinf((float)col * inv[j]);
                float *cp = cos + p * VIS_HEAD_DIM;
                float *sp = sin + p * VIS_HEAD_DIM;
                cp[j] = ch;
                cp[VIS_ROPE_N + j] = cw;
                cp[VIS_HEAD_DIM / 2 + j] = ch;
                cp[VIS_HEAD_DIM / 2 + VIS_ROPE_N + j] = cw;
                sp[j] = sh;
                sp[VIS_ROPE_N + j] = sw;
                sp[VIS_HEAD_DIM / 2 + j] = sh;
                sp[VIS_HEAD_DIM / 2 + VIS_ROPE_N + j] = sw;
            }
        }
    }

    q = h3_f32(n * VIS_HIDDEN);
    k = h3_f32(n * VIS_HIDDEN);
    v = h3_f32(n * VIS_HIDDEN);
    att = h3_f32(n * VIS_HIDDEN);
    nrm = h3_f32(n * VIS_HIDDEN);
    po = h3_f32(n * VIS_HIDDEN);
    g1 = h3_f32(n * VIS_INTER);
    float *qkv = h3_f32(n * 3 * VIS_HIDDEN);
    scores = h3_f32(H3_ATTN_ROWS * (n + 1));

    for (int64_t L = 0; L < VIS_DEPTH; L++) {
        const h3_vis_block *b = &blk[L];
        {
            float *w = h3_vec(&b->n1w), *bb = h3_vec(&b->n1b);
            h3_layernorm(nrm, x, w, bb, n, VIS_HIDDEN, 1e-6f);
            free(w); free(bb);
        }
        h3_linear(qkv, nrm, b->qkv, NULL, VIS_HIDDEN, n);
        {
            float *bb = h3_vec(&b->qkvb);
            for (int64_t i = 0; i < n; i++)
                for (int64_t c = 0; c < 3 * VIS_HIDDEN; c++)
                    qkv[i * 3 * VIS_HIDDEN + c] += bb[c];
            free(bb);
        }
        for (int64_t i = 0; i < n; i++) {
            const float *s = qkv + i * 3 * VIS_HIDDEN;
            memcpy(q + i * VIS_HIDDEN, s, (size_t)VIS_HIDDEN * 4);
            memcpy(k + i * VIS_HIDDEN, s + VIS_HIDDEN, (size_t)VIS_HIDDEN * 4);
            memcpy(v + i * VIS_HIDDEN, s + 2 * VIS_HIDDEN, (size_t)VIS_HIDDEN * 4);
        }
        h3_rope(q, cos, sin, n, VIS_HEADS, VIS_HEAD_DIM, VIS_HEAD_DIM);
        h3_rope(k, cos, sin, n, VIS_HEADS, VIS_HEAD_DIM, VIS_HEAD_DIM);
        h3_attention(att, q, k, v, n, VIS_HEADS, VIS_HEAD_DIM, false, scores);
        h3_linear(po, att, b->proj, NULL, VIS_HIDDEN, n);
        {
            float *bb = h3_vec(&b->projb);
            for (int64_t i = 0; i < n; i++)
                for (int64_t c = 0; c < VIS_HIDDEN; c++)
                    x[i * VIS_HIDDEN + c] += po[i * VIS_HIDDEN + c] + bb[c];
            free(bb);
        }

        {
            float *w = h3_vec(&b->n2w), *bb = h3_vec(&b->n2b);
            h3_layernorm(nrm, x, w, bb, n, VIS_HIDDEN, 1e-6f);
            free(w); free(bb);
        }
        h3_linear(g1, nrm, b->fc1, NULL, VIS_HIDDEN, n);
        {
            float *bb = h3_vec(&b->fc1b);
            for (int64_t i = 0; i < n; i++)
                for (int64_t c = 0; c < VIS_INTER; c++) g1[i * VIS_INTER + c] += bb[c];
            free(bb);
        }
        for (int64_t i = 0; i < n * VIS_INTER; i++) g1[i] = h3_gelu_tanh(g1[i]);
        h3_linear(po, g1, b->fc2, NULL, VIS_INTER, n);
        {
            float *bb = h3_vec(&b->fc2b);
            for (int64_t i = 0; i < n; i++)
                for (int64_t c = 0; c < VIS_HIDDEN; c++)
                    x[i * VIS_HIDDEN + c] += po[i * VIS_HIDDEN + c] + bb[c];
            free(bb);
        }

        if (L == 0) h3_dump("c_vis_blk0", x, n * VIS_HIDDEN);
        for (int64_t d = 0; d < VIS_DS; d++)
            if (L == 8 + 8 * d)
                h3_vis_merger_run(&ds[d], x, nmerge, false, vo->ds[d]);
    }
    free(q); q = NULL;
    free(k); k = NULL;
    free(v); v = NULL;
    free(att); att = NULL;
    free(g1); g1 = NULL;
    free(qkv); qkv = NULL;
    free(cos); cos = NULL;
    free(sin); sin = NULL;
    h3_dump("c_vis_last", x, n * VIS_HIDDEN);

    /* The image merger normalises each 1152-wide token first; four consecutive
     * tokens are one merge block, so the normalised stack already has the
     * 4608-wide rows its projection wants. */
    {
        float *w = h3_vec(&mg.norm_w), *bb = h3_vec(&mg.norm_b);
        h3_layernorm(nrm, x, w, bb, n, VIS_HIDDEN, 1e-6f);
        free(w); free(bb);
    }
    h3_vis_merger_run(&mg, nrm, nmerge, true, vo->merged);

    h3_dump("c_vis_merged", vo->merged, nmerge * VIS_OUT);
    for (int64_t d = 0; d < VIS_DS; d++) {
        char nm[32];
        snprintf(nm, sizeof(nm), "c_vis_ds%lld", (long long)d);
        h3_dump(nm, vo->ds[d], nmerge * VIS_OUT);
    }
    free(x);
    free(nrm);
    free(po);
    free(scores);
    free(blk);
    vo->gh = gh;
    vo->gw = gw;
    return true;
}

/* ------------------------------------------------------------------ */
/* Qwen3-VL language conditioner                                       */
/* ------------------------------------------------------------------ */
typedef struct {
    float *buf;       /* seq x TE_HIDDEN */
    float *hs50;      /* the layer the DiT consumes */
    float *tags;      /* per row: 1 text, 0 vision (edit path only) */
    int64_t seq;
} h3_text;

/* Vision conditioning for the conditioner. `types` marks the rows the vision
 * tower owns (`1`) as opposed to text (`0`); `vis` holds their tower output in
 * row order and `ds` the three deepstack taps, which the reference adds back
 * into the vision rows after blocks 0, 1 and 2. `grids` carries `gh, gw` per
 * vision block, consumed in row order, and only drives the rotary layout. */
typedef struct {
    const int32_t *types;
    const float *tags;    /* the transformer's per-row modality tag */
    const float *vis;
    const float *ds[VIS_DS];
    const int64_t *grids;
    int64_t nvis, nblocks;
} h3_cond;

static bool h3_text_encode(h3_model *m, const uint32_t *ids, int64_t n,
                           const h3_cond *cond, h3_text *out, cwick_error *err) {
    const int64_t D = TE_HIDDEN, NH = TE_HEADS, NKV = TE_KV_HEADS;
    const int64_t HD = TE_HEAD_DIM, F = TE_INTERMEDIATE;
    if (n <= 0 || n > 32768) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "prompt token count out of range");
        return false;
    }
    h3_mat embedding;
    if (!h3_mat_get(&m->te, "model.language_model.embed_tokens.weight", &embedding, err))
        return false;

    float *x = h3_f32(n * D), *tmp = h3_f32(n * D);
    int64_t nvis = 0;
    for (int64_t i = 0; i < n; i++) {
        if (cond && cond->types[i]) {
            if (nvis >= cond->nvis) {
                free(x); free(tmp);
                cwick_error_set(err, CWICK_ERR_ARGUMENT,
                                "vision rows outnumber the tower output");
                return false;
            }
            memcpy(x + i * D, cond->vis + nvis * D, (size_t)D * sizeof(float));
            nvis++;
            continue;
        }
        if (ids[i] >= (uint64_t)embedding.rows) {
            free(x); free(tmp);
            cwick_error_set(err, CWICK_ERR_ARGUMENT, "token id out of range");
            return false;
        }
        if (embedding.f32) {
            const float *row = (const float *)embedding.p + ids[i] * D;
            memcpy(x + i * D, row, (size_t)D * sizeof(float));
        } else {
            const uint16_t *row = (const uint16_t *)embedding.p + ids[i] * D;
            for (int64_t j = 0; j < D; j++) x[i * D + j] = cwick_bf16_fast(row[j]);
        }
    }
    if (cond && nvis != cond->nvis) {
        free(x); free(tmp);
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "vision rows outnumber the tower output");
        return false;
    }

    h3_dump("c_te_emb", x, n * D);
    /* Qwen3-VL's 3-axis mrope runs the rotary table over `position_ids` of
     * shape [3][seq]: the T axis fills all 64 half-slots, then the H and W
     * axes overwrite every third slot starting at 1 and 2 over the first 60,
     * which is 24 T + 20 H + 20 W. A text row carries the same position on all
     * three axes, so for a text-only presentation the layout collapses to
     * plain 1-D RoPE; a vision row does not.
     *
     * The table is duplicated so `rotate_half` can pair it. */
    float *cos = h3_f32(n * HD), *sin = h3_f32(n * HD);
    {
        int64_t *pos = cwick_xmalloc((size_t)3 * n * sizeof(int64_t));
        if (!cond) {
            for (int64_t i = 0; i < n; i++) pos[i] = pos[n + i] = pos[2 * n + i] = i;
        } else {
            int64_t cp = 0, i = 0, blk = 0;
            while (i < n) {
                const int32_t t = cond->types[i];
                int64_t j = i;
                while (j < n && cond->types[j] == t) j++;
                if (t == 0) {
                    for (int64_t k = i; k < j; k++)
                        pos[k] = pos[n + k] = pos[2 * n + k] = cp + (k - i);
                    cp += j - i;
                } else {
                    /* The text encoder lays the merged grid out raster-wise
                     * even though the tower emits merge-block order, so the
                     * two orders disagree on every row but the first. */
                    if (blk >= cond->nblocks) {
                        free(pos); free(x); free(tmp);
                        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                                        "more vision blocks than grids");
                        return false;
                    }
                    const int64_t gh = cond->grids[2 * blk], gw = cond->grids[2 * blk + 1];
                    blk++;
                    const int64_t lw = gw / VIS_MERGE;
                    for (int64_t k = i; k < j; k++) {
                        const int64_t r = k - i;
                        pos[k] = cp;
                        pos[n + k] = cp + r / lw;
                        pos[2 * n + k] = cp + r % lw;
                    }
                    cp += (gh > gw ? gh : gw) / VIS_MERGE;
                }
                i = j;
            }
        }
        h3_dump_i64("c_te_pos", pos, 3 * n);
        for (int64_t s = 0; s < n; s++) {
            for (int64_t j = 0; j < HD / 2; j++) {
                const float freq = powf(1.0f / TE_ROPE_THETA, (float)(2 * j) / (float)HD);
                const int64_t axis = j >= 60 ? 0 : j % 3;
                const float ang = (float)pos[axis * n + s] * freq;
                cos[s * HD + j] = cosf(ang);
                cos[s * HD + j + HD / 2] = cosf(ang);
                sin[s * HD + j] = sinf(ang);
                sin[s * HD + j + HD / 2] = sinf(ang);
            }
        }
        free(pos);
    }
    h3_dump("c_te_cos", cos, n * HD);

    float *q = h3_f32(n * NH * HD);
    float *k = h3_f32(n * NKV * HD);
    float *v = h3_f32(n * NKV * HD);
    float *att = h3_f32(n * NH * HD);
    float *proj = h3_f32(n * D);
    float *gate = h3_f32(n * F), *up = h3_f32(n * F);
    float *scores = h3_f32(H3_ATTN_ROWS * (n + 1));

    char name[256];
    for (int64_t L = 0; L < TE_LAYERS; L++) {
        h3_mat n1, n2, qn, kn, wq, wk, wv, wo, wg, wu, wd;
        snprintf(name, sizeof(name), "model.language_model.layers.%lld.input_layernorm.weight", (long long)L);
        if (!h3_mat_get(&m->te, name, &n1, err)) goto fail;
        snprintf(name, sizeof(name), "model.language_model.layers.%lld.self_attn.q_norm.weight", (long long)L);
        if (!h3_mat_get(&m->te, name, &qn, err)) goto fail;
        snprintf(name, sizeof(name), "model.language_model.layers.%lld.self_attn.k_norm.weight", (long long)L);
        if (!h3_mat_get(&m->te, name, &kn, err)) goto fail;
        snprintf(name, sizeof(name), "model.language_model.layers.%lld.self_attn.q_proj.weight", (long long)L);
        if (!h3_mat_get(&m->te, name, &wq, err)) goto fail;
        snprintf(name, sizeof(name), "model.language_model.layers.%lld.self_attn.k_proj.weight", (long long)L);
        if (!h3_mat_get(&m->te, name, &wk, err)) goto fail;
        snprintf(name, sizeof(name), "model.language_model.layers.%lld.self_attn.v_proj.weight", (long long)L);
        if (!h3_mat_get(&m->te, name, &wv, err)) goto fail;
        snprintf(name, sizeof(name), "model.language_model.layers.%lld.self_attn.o_proj.weight", (long long)L);
        if (!h3_mat_get(&m->te, name, &wo, err)) goto fail;

        float *wn1 = h3_vec(&n1), *wqn = h3_vec(&qn), *wkn = h3_vec(&kn);
        for (int64_t s = 0; s < n; s++)
            h3_rmsnorm(tmp + s * D, x + s * D, wn1, D, 1e-6f);
        if (L == 0) h3_dump("c_te_norm", tmp, n * D);

        h3_linear(q, tmp, wq, NULL, D, n);
        h3_linear(k, tmp, wk, NULL, D, n);
        h3_linear(v, tmp, wv, NULL, D, n);
        if (L == 0) { h3_dump("c_te_qraw", q, n * NH * HD); h3_dump("c_wqn", wqn, HD); }

        /* Qwen3-VL applies q/k norm per head on the head dimension. */
        h3_rmsnorm_heads(q, wqn, n, NH, HD, 1e-6f);
        h3_rmsnorm_heads(k, wkn, n, NKV, HD, 1e-6f);
        free(wn1); free(wqn); free(wkn);
        h3_rope(q, cos, sin, n, NH, HD, HD);
        h3_rope(k, cos, sin, n, NKV, HD, HD);
        if (L == 0) { h3_dump("c_te_q", q, n * NH * HD); h3_dump("c_te_k", k, n * NKV * HD); }

        /* GQA: expand each KV head to the 8 query heads that share it. */
        {
            int64_t rep = NH / NKV;
            float *ke = h3_f32(n * NH * HD), *ve = h3_f32(n * NH * HD);
            for (int64_t s = 0; s < n; s++)
                for (int64_t h = 0; h < NH; h++) {
                    memcpy(ke + (s * NH + h) * HD, k + (s * NKV + h / rep) * HD, HD * sizeof(float));
                    memcpy(ve + (s * NH + h) * HD, v + (s * NKV + h / rep) * HD, HD * sizeof(float));
                }
            h3_attention(att, q, ke, ve, n, NH, HD, true, scores);
            free(ke); free(ve);
        }
        if (L == 0) h3_dump("c_te_att", att, n * NH * HD);
        h3_linear(proj, att, wo, NULL, NH * HD, n);
        for (int64_t i = 0; i < n * D; i++) x[i] += proj[i];
        if (L == 0) h3_dump("c_te_attn", x, n * D);

        snprintf(name, sizeof(name), "model.language_model.layers.%lld.post_attention_layernorm.weight", (long long)L);
        if (!h3_mat_get(&m->te, name, &n2, err)) goto fail;
        snprintf(name, sizeof(name), "model.language_model.layers.%lld.mlp.gate_proj.weight", (long long)L);
        if (!h3_mat_get(&m->te, name, &wg, err)) goto fail;
        snprintf(name, sizeof(name), "model.language_model.layers.%lld.mlp.up_proj.weight", (long long)L);
        if (!h3_mat_get(&m->te, name, &wu, err)) goto fail;
        snprintf(name, sizeof(name), "model.language_model.layers.%lld.mlp.down_proj.weight", (long long)L);
        if (!h3_mat_get(&m->te, name, &wd, err)) goto fail;

        float *wn2 = h3_vec(&n2);
        for (int64_t s = 0; s < n; s++)
            h3_rmsnorm(tmp + s * D, x + s * D, wn2, D, 1e-6f);
        free(wn2);
        h3_linear(gate, tmp, wg, NULL, D, n);
        h3_linear(up, tmp, wu, NULL, D, n);
        h3_silu(gate, n * F);
        for (int64_t i = 0; i < n * F; i++) gate[i] *= up[i];
        h3_linear(proj, gate, wd, NULL, F, n);
        for (int64_t i = 0; i < n * D; i++) x[i] += proj[i];
        if (L == 0) h3_dump("c_te0", x, n * D);
        /* The three deepstack taps go back into the vision rows only, after
         * blocks 0, 1 and 2 — not into the text rows that follow them. The
         * reference snapshots `hidden_states[k]` before the add, so the dumps
         * bracket it rather than landing on it. */
        if (cond && L < VIS_DS) {
            char nm[24];
            snprintf(nm, sizeof(nm), "c_te_pre%lld", (long long)L);
            h3_dump(nm, x, n * D);
            int64_t k = 0;
            for (int64_t i = 0; i < n; i++)
                if (cond->types[i]) {
                    const float *s = cond->ds[L] + k * D;
                    float *d = x + i * D;
                    for (int64_t c = 0; c < D; c++) d[c] += s[c];
                    k++;
                }
            snprintf(nm, sizeof(nm), "c_te_ds%lld", (long long)L);
            h3_dump(nm, x, n * D);
        }
        if (h3_dump_blocks()) {
            snprintf(name, sizeof(name), "c_te%02lld", (long long)L);
            h3_dump(name, x, n * D);
        }
    }

    /* `hidden_states[50]` is the residual stream after 50 blocks: pre-norm,
     * never the post-norm final state, which is why the stack stops here. */
    float *hs50 = h3_f32(n * D);
    memcpy(hs50, x, (size_t)n * D * sizeof(float));
    h3_dump("c_hs50", hs50, n * D);

    float *tags = h3_f32(n);
    for (int64_t i = 0; i < n; i++)
        tags[i] = (cond && cond->tags) ? cond->tags[i]
                                       : ((cond && cond->types[i]) ? 0.0f : 1.0f);

    free(tmp); free(q); free(k); free(v); free(att);
    free(proj); free(gate); free(up); free(scores); free(cos); free(sin);
    out->buf = x; out->hs50 = hs50; out->tags = tags; out->seq = n;
    return true;

fail:
    free(x); free(tmp); free(q); free(k); free(v); free(att);
    free(proj); free(gate); free(up); free(scores); free(cos); free(sin);
    return false;
}

static void h3_text_free(h3_text *t) {
    free(t->buf); free(t->hs50); free(t->tags);
    t->buf = t->hs50 = t->tags = NULL;
}

/* ------------------------------------------------------------------ */
/* DiT: one packed sequence, full self-attention, no cross-attention.  */
/*                                                                     */
/* Layout is [text rows | image rows]. The per-row AdaLN index is      */
/* `timestep_index * 3 + token_tag`, and the table holds one row of    */
/* six modulation vectors per (timestep, modality) pair.              */
/* ------------------------------------------------------------------ */
typedef struct {
    float *img;        /* num_image_rows x DIT_PATCH_DIM, frame-major then row-major */
    int64_t rows;
} h3_latent_rows;

/* A reference image's latent grid and how many packed rows it occupies. */
typedef struct {
    int64_t lw, lh, rows;
} h3_refgeo;

/* Reference rows are not clean: they enter the packed sequence noised to this
 * level and stay there for every step, which is the level the checkpoint was
 * conditioned on. */
#define H3_REF_NOISE_AUG 0.999f

/* [C][lh][lw] -> packed rows, the inverse of the unpatchify at the end of a
 * generation. Row r holds one 2x2 latent patch of frame cell (r / pw, r % pw),
 * laid out as `[c][dy][dx]`. */
static void h3_pack_rows(float *dst, const float *lat, int lw, int lh) {
    const int64_t pw = lw / H3_PATCH, ph = lh / H3_PATCH;
    for (int64_t r = 0; r < ph * pw; r++) {
        const int64_t py = r / pw, px = r % pw;
        for (int64_t c = 0; c < H3_LATENT_CHANNELS; c++) {
            float *d = dst + r * DIT_PATCH_DIM + c * 4;
            for (int dy = 0; dy < H3_PATCH; dy++)
                for (int dx = 0; dx < H3_PATCH; dx++)
                    d[dy * H3_PATCH + dx] =
                        lat[(c * lh + py * H3_PATCH + dy) * lw + px * H3_PATCH + dx];
        }
    }
}

/* One latent frame's rotary grid: each axis centred on the unit interval and
 * scaled by 32, so a square canvas spans [0, 32). */
static void h3_rope_frame(float *pos, int64_t start, int64_t rows, float t,
                          int lw, int lh) {
    const int64_t gh = lh / H3_PATCH, gw = lw / H3_PATCH;
    const float sqrt_area = sqrtf((float)lh * (float)lw);
    const float ratio_h = (float)lh / sqrt_area, ratio_w = (float)lw / sqrt_area;
    const float left_h = (1.0f - ratio_h) * 0.5f, left_w = (1.0f - ratio_w) * 0.5f;
    const float step_h = ratio_h / (float)gh, step_w = ratio_w / (float)gw;
    for (int64_t r = 0; r < rows; r++) {
        const int64_t y = r / gw, x = r % gw;
        const int64_t s = start + r;
        pos[s * 3 + 0] = t;
        pos[s * 3 + 1] = (left_h + (float)y * step_h) * 32.0f;
        pos[s * 3 + 2] = (left_w + (float)x * step_w) * 32.0f;
    }
}

/* 3-axis rotary table. Frequencies are shared by the (t, h, w) axes and
 * concatenated, then duplicated: 96 of the 128 head channels rotate.
 * Text rows sit at (t = s, h = 0, w = 0). Every image is one frame on the
 * rotary clock, which starts where the text rows end: a reference takes one
 * integer slot each and the generated image the slot after the last one. */
static void h3_dit_rope_tables(int64_t ntext, const h3_refgeo *ref, int nrefblk,
                               int64_t nrefrows, int64_t ngen, int lw, int lh,
                               float *cos, float *sin) {
    const int64_t F = DIT_ROPE_FREQ, RD = DIT_ROPE_DIM;
    int64_t seq = ntext + nrefrows + ngen;
    float inv[DIT_ROPE_FREQ];
    for (int64_t j = 0; j < F; j++)
        inv[j] = powf(1.0f / 10000.0f, (float)(2 * j) / (float)(2 * F));

    float *pos = h3_f32(seq * 3);   /* t, h, w per row */
    for (int64_t s = 0; s < ntext; s++) {
        pos[s * 3 + 0] = (float)s;
        pos[s * 3 + 1] = 0.0f;
        pos[s * 3 + 2] = 0.0f;
    }
    int64_t cursor = ntext;
    float rt = (float)ntext;
    for (int b = 0; b < nrefblk; b++) {
        h3_rope_frame(pos, cursor, ref[b].rows, rt, ref[b].lw, ref[b].lh);
        cursor += ref[b].rows;
        rt += 1.0f;
    }
    h3_rope_frame(pos, cursor, ngen, rt, lw, lh);

    for (int64_t s = 0; s < seq; s++) {
        for (int64_t j = 0; j < F; j++) {
            float ft = pos[s * 3 + 0] * inv[j];
            float fh = pos[s * 3 + 1] * inv[j];
            float fw = pos[s * 3 + 2] * inv[j];
            cos[s * RD + j] = cosf(ft);
            cos[s * RD + F + j] = cosf(fh);
            cos[s * RD + 2 * F + j] = cosf(fw);
            sin[s * RD + j] = sinf(ft);
            sin[s * RD + F + j] = sinf(fh);
            sin[s * RD + 2 * F + j] = sinf(fw);
        }
        for (int64_t k = 0; k < RD; k++) {
            cos[s * RD + RD / 2 + k] = cos[s * RD + k];
            sin[s * RD + RD / 2 + k] = sin[s * RD + k];
        }
    }
    free(pos);
}

/* SwiGLU as diffusers stores it: the first half is the value, the second is
 * the gate, and the product is `value * silu(gate)`.
 *
 * Written as five vector passes rather than one scalar loop because at 4096
 * rows this touches 59 million elements per block: the scalar form measured
 * 3.9 s per sampling step, and the exponential is the only part that cannot be
 * folded into the multiply. The product moves into `val` first so the
 * denominator can be built in place over `gate`. */
static void h3_swiglu(float *val, float *gate, int64_t n) {
    if (n <= 0) return;
    if (n > 2147483647LL) {
        for (int64_t i = 0; i < n; i++)
            val[i] = val[i] * (gate[i] / (1.0f + expf(-gate[i])));
        return;
    }
    vDSP_Length len = (vDSP_Length)n;
    const float one = 1.0f;
    int ni = (int)n;
    vDSP_vmul(val, 1, gate, 1, val, 1, len);        /* val = value * gate */
    vDSP_vneg(gate, 1, gate, 1, len);               /* gate = -gate        */
    vvexpf(gate, gate, &ni);                        /* gate = exp(-gate)   */
    vDSP_vsadd(gate, 1, &one, gate, 1, len);        /* gate = 1 + exp(...) */
    vDSP_vdiv(gate, 1, val, 1, val, 1, len);        /* val / denominator   */
}

/* One attention block over the packed sequence (or over the text stream
 * alone, which is what the token refiner runs). `rope` may be NULL. */
static bool h3_dit_attn_block(const cwick_safetensors *saf, const char *prefix,
                              float *x, int64_t seq, bool rope_on,
                              const float *cos, const float *sin, float *scratch,
                              cwick_error *err) {
    char name[256];
    h3_mat n1, n2, qw, kw, vw, ow, nq, nk, w1, w2;
    snprintf(name, sizeof(name), "%snorm1.weight", prefix);
    if (!h3_mat_get(saf, name, &n1, err)) return false;
    snprintf(name, sizeof(name), "%snorm2.weight", prefix);
    if (!h3_mat_get(saf, name, &n2, err)) return false;
    snprintf(name, sizeof(name), "%sattn.to_q.weight", prefix);
    if (!h3_mat_get(saf, name, &qw, err)) return false;
    snprintf(name, sizeof(name), "%sattn.to_k.weight", prefix);
    if (!h3_mat_get(saf, name, &kw, err)) return false;
    snprintf(name, sizeof(name), "%sattn.to_v.weight", prefix);
    if (!h3_mat_get(saf, name, &vw, err)) return false;
    snprintf(name, sizeof(name), "%sattn.to_out.0.weight", prefix);
    if (!h3_mat_get(saf, name, &ow, err)) return false;
    snprintf(name, sizeof(name), "%sattn.norm_q.weight", prefix);
    if (!h3_mat_get(saf, name, &nq, err)) return false;
    snprintf(name, sizeof(name), "%sattn.norm_k.weight", prefix);
    if (!h3_mat_get(saf, name, &nk, err)) return false;
    snprintf(name, sizeof(name), "%sff.net.0.proj.weight", prefix);
    if (!h3_mat_get(saf, name, &w1, err)) return false;
    snprintf(name, sizeof(name), "%sff.net.2.weight", prefix);
    if (!h3_mat_get(saf, name, &w2, err)) return false;

    const int64_t D = DIT_HIDDEN, NH = DIT_HEADS, HD = DIT_HEAD_DIM;
    const int64_t K = DIT_QKV, FF = DIT_FFN;
    float *norm = h3_f32(seq * D), *q = h3_f32(seq * K);
    float *k = h3_f32(seq * K), *v = h3_f32(seq * K), *att = h3_f32(seq * K);
    float *po = h3_f32(seq * D), *g1 = h3_f32(seq * FF), *g2 = h3_f32(seq * FF);
    float *wn1 = h3_vec(&n1), *wn2 = h3_vec(&n2);
    float *wq = h3_vec(&nq), *wk = h3_vec(&nk);

    for (int64_t s = 0; s < seq; s++)
        h3_rmsnorm(norm + s * D, x + s * D, wn1, D, 1e-5f);
    h3_linear(q, norm, qw, NULL, D, seq);
    h3_linear(k, norm, kw, NULL, D, seq);
    h3_linear(v, norm, vw, NULL, D, seq);
    h3_rmsnorm_heads(q, wq, seq, NH, HD, 1e-5f);
    h3_rmsnorm_heads(k, wk, seq, NH, HD, 1e-5f);
    if (rope_on) {
        h3_rope(q, cos, sin, seq, NH, HD, DIT_ROPE_DIM);
        h3_rope(k, cos, sin, seq, NH, HD, DIT_ROPE_DIM);
    }
    h3_attention(att, q, k, v, seq, NH, HD, false, scratch);
    h3_linear(po, att, ow, NULL, K, seq);
    for (int64_t i = 0; i < seq * D; i++) x[i] += po[i];

    for (int64_t s = 0; s < seq; s++)
        h3_rmsnorm(norm + s * D, x + s * D, wn2, D, 1e-5f);
    {
        h3_mat wv = w1, wg = w1;
        wv.rows = FF;
        wg.rows = FF;
        wg.p = (const char *)w1.p + (size_t)FF * D * (w1.f32 ? 4 : 2);
        h3_linear(g1, norm, wv, NULL, D, seq);
        h3_linear(g2, norm, wg, NULL, D, seq);
    }
    h3_swiglu(g1, g2, seq * FF);
    h3_linear(po, g1, w2, NULL, FF, seq);
    for (int64_t i = 0; i < seq * D; i++) x[i] += po[i];

    free(norm); free(q); free(k); free(v); free(att);
    free(po); free(g1); free(g2);
    free(wn1); free(wn2); free(wq); free(wk);
    return true;
}

/* Opt-in phase timer: see the declaration above the numerics section. */
#define H3_P(ph) do { if (h3_prof_on()) h3_prof_p[ph] -= h3_ms(); } while (0)
#define H3_PE(ph) do { if (h3_prof_on()) h3_prof_p[ph] += h3_ms(); } while (0)

static bool h3_dit_step(h3_model *m, const h3_text *txt, const h3_latent_rows *img,
                        const h3_refgeo *ref, int nrefblk, int64_t nrefrows,
                        int lw, int lh, int step, float timestep,
                        float **out_vel, cwick_error *err) {
    const int64_t D = DIT_HIDDEN, NH = DIT_HEADS, HD = DIT_HEAD_DIM;
    const int64_t K = DIT_QKV, F = DIT_FFN;
    int64_t ntext = txt->seq, nimg = img->rows, seq = ntext + nimg;

    /* Distinct noise levels in the packed sequence, ordered as
     * `unique(sorted=True)` orders them, which is what the AdaLN row of a
     * sequence position addresses: `timestep_index * 3 + token_tag`. A
     * text2image request has one level; an edit adds the reference level and
     * the text rows keep the generated one. */
    int nts = 1, gen_ti = 0, ref_ti = 0;
    float tv[2] = {timestep, 0.0f};
    if (nrefrows > 0) {
        if (H3_REF_NOISE_AUG == timestep) {
            nts = 1;
        } else if (H3_REF_NOISE_AUG < timestep) {
            nts = 2; tv[0] = H3_REF_NOISE_AUG; tv[1] = timestep;
            ref_ti = 0; gen_ti = 1;
        } else {
            nts = 2; tv[0] = timestep; tv[1] = H3_REF_NOISE_AUG;
            gen_ti = 0; ref_ti = 1;
        }
    }
#define H3_ROW_TI(s) ((s) >= ntext && (s) < ntext + nrefrows ? ref_ti : gen_ti)
    /* Text rows are modality 1, image rows 0 — except that a reference's
     * vision block is image-tagged even though it lives in the text stream,
     * which is what the conditioner's per-row tag records. */
#define H3_ROW_TAG(s) ((s) < ntext ? (txt->tags ? (int64_t)txt->tags[s] : 1) : 0)

    if (h3_prof_on()) {
        memset(h3_prof_p, 0, sizeof(h3_prof_p));
        h3_prof_bytes = 0.0;
        h3_prof_macs = 0.0;
    }
    double prof_t0 = h3_ms();
    float *x = h3_f32(seq * D);
    float *ctx = h3_f32(ntext * D), *imge = h3_f32(nimg * D);
    float *tb = h3_f32(256), *t1 = h3_f32(DIT_TIME_HIDDEN);
    float *temb = h3_f32((int64_t)nts * DIT_TIME_DIM);
    float *adaln = h3_f32((int64_t)nts * DIT_ADALN_ROWS);
    float *final_mod = h3_f32((int64_t)nts * 2 * D);
    float *scores = h3_f32(H3_ATTN_ROWS * (seq + 1));
    float *cos = h3_f32(seq * DIT_ROPE_DIM), *sin = h3_f32(seq * DIT_ROPE_DIM);
    float *vel = NULL;
    h3_mat proj_in, proj_in_b, cond, cond_b, tlin1, tlin1_b, tlin2, tlin2_b;
    h3_mat no, nol, nolb;
    int64_t nimg_out = nimg;

    if (!h3_mat_get(&m->dit, "proj_in.weight", &proj_in, err)) goto fail;
    if (!h3_mat_get(&m->dit, "proj_in.bias", &proj_in_b, err)) goto fail;
    if (!h3_mat_get(&m->dit, "context_embedder.weight", &cond, err)) goto fail;
    if (!h3_mat_get(&m->dit, "context_embedder.bias", &cond_b, err)) goto fail;
    if (!h3_mat_get(&m->dit, "time_embedder.linear_1.weight", &tlin1, err)) goto fail;
    if (!h3_mat_get(&m->dit, "time_embedder.linear_1.bias", &tlin1_b, err)) goto fail;
    if (!h3_mat_get(&m->dit, "time_embedder.linear_2.weight", &tlin2, err)) goto fail;
    if (!h3_mat_get(&m->dit, "time_embedder.linear_2.bias", &tlin2_b, err)) goto fail;
    if (!h3_mat_get(&m->dit, "norm_out.norm.weight", &no, err)) goto fail;
    if (!h3_mat_get(&m->dit, "norm_out.linear.weight", &nol, err)) goto fail;
    if (!h3_mat_get(&m->dit, "norm_out.linear.bias", &nolb, err)) goto fail;

    /* Text stream: project, refine, then scatter into the packed buffer. */
    {
        float *cb = h3_vec(&cond_b);
        h3_linear(ctx, txt->hs50, cond, cb, TE_HIDDEN, ntext);
        free(cb);
    }
    {
        char prefix[128];
        for (int64_t R = 0; R < DIT_REFINER; R++) {
            snprintf(prefix, sizeof(prefix), "token_refiner.refiner_blocks.%lld.", (long long)R);
            if (!h3_dit_attn_block(&m->dit, prefix, ctx, ntext, false, NULL, NULL,
                                   scores, err)) goto fail;
        }
        h3_mat fn;
        if (!h3_mat_get(&m->dit, "token_refiner.final_norm.weight", &fn, err)) goto fail;
        float *wfn = h3_vec(&fn);
        for (int64_t s = 0; s < ntext; s++)
            h3_rmsnorm(ctx + s * D, ctx + s * D, wfn, D, 1e-5f);
        free(wfn);
    }
    if (step == 0) h3_dump("c_refiner", ctx, ntext * D);
    {
        float *pb = h3_vec(&proj_in_b);
        h3_linear(imge, img->img, proj_in, pb, DIT_PATCH_DIM, nimg);
        free(pb);
    }
    memcpy(x, ctx, (size_t)ntext * D * sizeof(float));
    memcpy(x + ntext * D, imge, (size_t)nimg * D * sizeof(float));

    /* Timestep embedding: 256-wide sinusoid, `flip_sin_to_cos` with
     * `downscale_freq_shift = 0`, then the two-layer MLP at float32. One row
     * per distinct level. */
    for (int ti = 0; ti < nts; ti++) {
        for (int64_t i = 0; i < 128; i++) {
            float freq = powf(1.0f / 10000.0f, (float)i / 128.0f);
            float v = tv[ti] * freq;
            tb[i] = cosf(v);
            tb[128 + i] = sinf(v);
        }
        h3_linear(t1, tb, tlin1, (const float *)tlin1_b.p, 256, 1);
        h3_silu(t1, DIT_TIME_HIDDEN);
        h3_linear(temb + ti * DIT_TIME_DIM, t1, tlin2, (const float *)tlin2_b.p,
                  DIT_TIME_HIDDEN, 1);
    }
    free(t1);
    t1 = NULL;
    if (step == 0) h3_dump("c_temb", temb, (int64_t)nts * DIT_TIME_DIM);

    h3_dit_rope_tables(ntext, ref, nrefblk, nrefrows, nimg - nrefrows, lw, lh, cos, sin);
    if (step == 0) h3_dump("c_packed", x, seq * D);

    char name[256];
    for (int64_t L = 0; L < DIT_LAYERS; L++) {
        h3_mat adw, adb, n1, n2, qw, kw, vw, ow, nq, nk, w1, w2;
        snprintf(name, sizeof(name), "transformer_blocks.%lld.adaln_proj.linear.weight", (long long)L);
        if (!h3_mat_get(&m->dit, name, &adw, err)) goto fail;
        snprintf(name, sizeof(name), "transformer_blocks.%lld.adaln_proj.linear.bias", (long long)L);
        if (!h3_mat_get(&m->dit, name, &adb, err)) goto fail;
        snprintf(name, sizeof(name), "transformer_blocks.%lld.norm1.weight", (long long)L);
        if (!h3_mat_get(&m->dit, name, &n1, err)) goto fail;
        snprintf(name, sizeof(name), "transformer_blocks.%lld.norm2.weight", (long long)L);
        if (!h3_mat_get(&m->dit, name, &n2, err)) goto fail;
        snprintf(name, sizeof(name), "transformer_blocks.%lld.attn.to_q.weight", (long long)L);
        if (!h3_mat_get(&m->dit, name, &qw, err)) goto fail;
        snprintf(name, sizeof(name), "transformer_blocks.%lld.attn.to_k.weight", (long long)L);
        if (!h3_mat_get(&m->dit, name, &kw, err)) goto fail;
        snprintf(name, sizeof(name), "transformer_blocks.%lld.attn.to_v.weight", (long long)L);
        if (!h3_mat_get(&m->dit, name, &vw, err)) goto fail;
        snprintf(name, sizeof(name), "transformer_blocks.%lld.attn.to_out.0.weight", (long long)L);
        if (!h3_mat_get(&m->dit, name, &ow, err)) goto fail;
        snprintf(name, sizeof(name), "transformer_blocks.%lld.attn.norm_q.weight", (long long)L);
        if (!h3_mat_get(&m->dit, name, &nq, err)) goto fail;
        snprintf(name, sizeof(name), "transformer_blocks.%lld.attn.norm_k.weight", (long long)L);
        if (!h3_mat_get(&m->dit, name, &nk, err)) goto fail;
        snprintf(name, sizeof(name), "transformer_blocks.%lld.ff.net.0.proj.weight", (long long)L);
        if (!h3_mat_get(&m->dit, name, &w1, err)) goto fail;
        snprintf(name, sizeof(name), "transformer_blocks.%lld.ff.net.2.weight", (long long)L);
        if (!h3_mat_get(&m->dit, name, &w2, err)) goto fail;

        /* Six modulation vectors per (timestep, modality) row. */
        for (int ti = 0; ti < nts; ti++) {
            float *sil = h3_f32(DIT_TIME_DIM);
            float *b = h3_vec(&adb);
            memcpy(sil, temb + ti * DIT_TIME_DIM, (size_t)DIT_TIME_DIM * sizeof(float));
            h3_silu(sil, DIT_TIME_DIM);
            H3_P(3);
            h3_linear(adaln + ti * DIT_ADALN_ROWS, sil, adw, b, DIT_TIME_DIM, 1);
            H3_PE(3);
            free(sil); free(b);
        }
        float *wn1 = h3_vec(&n1), *wn2 = h3_vec(&n2);
        float *wq = h3_vec(&nq), *wk = h3_vec(&nk);

        float *norm = h3_f32(seq * D), *q = h3_f32((seq + 8) * K);
        float *kk = h3_f32((seq + 8) * K), *vv = h3_f32((seq + 8) * K);
        float *att = h3_f32((seq + 8) * K);
        float *po = h3_f32(seq * D), *g1 = h3_f32(seq * F), *g2 = h3_f32(seq * F);

        for (int64_t s = 0; s < seq; s++) {
            int64_t tag = H3_ROW_TAG(s);
            const float *base = adaln + (H3_ROW_TI(s) * 3 + tag) * 6 * D;
            const float *sh = base, *sc = base + D, *ga = base + 2 * D;
            h3_rmsnorm(norm + s * D, x + s * D, wn1, D, 1e-5f);
            for (int64_t d = 0; d < D; d++)
                norm[s * D + d] = norm[s * D + d] * (1.0f + sc[d]) + sh[d];
            (void)ga;
        }
        H3_P(0);
        h3_linear(q, norm, qw, NULL, D, seq);
        h3_linear(kk, norm, kw, NULL, D, seq);
        h3_linear(vv, norm, vw, NULL, D, seq);
        H3_PE(0);
        h3_rmsnorm_heads(q, wq, seq, NH, HD, 1e-5f);
        h3_rmsnorm_heads(kk, wk, seq, NH, HD, 1e-5f);
        H3_P(1);
        h3_rope(q, cos, sin, seq, NH, HD, DIT_ROPE_DIM);
        h3_rope(kk, cos, sin, seq, NH, HD, DIT_ROPE_DIM);
        h3_attention(att, q, kk, vv, seq, NH, HD, false, scores);
        H3_PE(1);
        H3_P(0);
        h3_linear(po, att, ow, NULL, K, seq);
        H3_PE(0);
        for (int64_t s = 0; s < seq; s++) {
            int64_t tag = H3_ROW_TAG(s);
            const float *gate = adaln + (H3_ROW_TI(s) * 3 + tag) * 6 * D + 2 * D;
            for (int64_t d = 0; d < D; d++) x[s * D + d] += po[s * D + d] * gate[d];
        }

        for (int64_t s = 0; s < seq; s++) {
            int64_t tag = H3_ROW_TAG(s);
            const float *base = adaln + (H3_ROW_TI(s) * 3 + tag) * 6 * D + 3 * D;
            const float *sh = base, *sc = base + D;
            h3_rmsnorm(norm + s * D, x + s * D, wn2, D, 1e-5f);
            for (int64_t d = 0; d < D; d++)
                norm[s * D + d] = norm[s * D + d] * (1.0f + sc[d]) + sh[d];
        }
        H3_P(0);
        {
            h3_mat wv = w1, wg = w1;
            wv.rows = F; wg.rows = F;
            wg.p = (const char *)w1.p + (size_t)F * D * (w1.f32 ? 4 : 2);
            h3_linear(g1, norm, wv, NULL, D, seq);
            h3_linear(g2, norm, wg, NULL, D, seq);
        }
        H3_PE(0);
        H3_P(2);
        h3_swiglu(g1, g2, seq * F);
        H3_PE(2);
        H3_P(0);
        h3_linear(po, g1, w2, NULL, F, seq);
        H3_PE(0);
        for (int64_t s = 0; s < seq; s++) {
            int64_t tag = H3_ROW_TAG(s);
            const float *gate = adaln + (H3_ROW_TI(s) * 3 + tag) * 6 * D + 5 * D;
            for (int64_t d = 0; d < D; d++) x[s * D + d] += po[s * D + d] * gate[d];
        }

        free(norm); free(q); free(kk); free(vv); free(att);
        free(po); free(g1); free(g2);
        free(wn1); free(wn2); free(wq); free(wk);
        if (step == 0 && L == 0) h3_dump("c_block0", x, seq * D);
        if (step == 0 && L == DIT_LAYERS - 1) h3_dump("c_blocklast", x, seq * D);
    }

    /* Output head of this sampling step. `norm_out` modulates per timestep
     * (not per modality) and selects by `timestep_indices`, then every row is
     * projected; the caller keeps only the generated rows. */
    for (int ti = 0; ti < nts; ti++) {
        float *sil = h3_f32(DIT_TIME_DIM);
        float *b = h3_vec(&nolb);
        memcpy(sil, temb + ti * DIT_TIME_DIM, (size_t)DIT_TIME_DIM * sizeof(float));
        h3_silu(sil, DIT_TIME_DIM);
        h3_linear(final_mod + ti * 2 * D, sil, nol, b, DIT_TIME_DIM, 1);
        free(sil); free(b);
    }
    {
        h3_mat hw, hb;
        char nm[128];
        snprintf(nm, sizeof(nm), "pdd_proj_out.%d.weight", step);
        if (!h3_mat_get(&m->dit, nm, &hw, err)) goto fail;
        snprintf(nm, sizeof(nm), "pdd_proj_out.%d.bias", step);
        if (!h3_mat_get(&m->dit, nm, &hb, err)) goto fail;

        float *hn = h3_f32(nimg * D);
        float *wno = h3_vec(&no), *bh = h3_vec(&hb);
        for (int64_t s = 0; s < nimg; s++) {
            const float *row = x + (ntext + s) * D;
            const float *fm = final_mod + H3_ROW_TI(ntext + s) * 2 * D;
            h3_rmsnorm(hn + s * D, row, wno, D, 1e-5f);
            for (int64_t d = 0; d < D; d++)
                hn[s * D + d] = hn[s * D + d] * (1.0f + fm[D + d]) + fm[d];
        }
        vel = h3_f32(nimg_out * DIT_PATCH_DIM);
        h3_linear(vel, hn, hw, bh, D, nimg);
        free(wno); free(bh);
        free(hn);
    }
    if (step == 0) h3_dump("c_vel", vel, nimg_out * DIT_PATCH_DIM);

    if (h3_prof_on()) {
        double wall = h3_ms() - prof_t0;
        double acc = h3_prof_p[0] + h3_prof_p[1] + h3_prof_p[2] + h3_prof_p[3];
        fprintf(stderr,
                "h3 prof step %d (seq %lld): gemm %.1f attn %.1f swiglu %.1f "
                "adaln %.1f other %.1f wall %.1f ms | weights %.1f GB %.1f GB/s, "
                "%.1f GMAC %.1f GMAC/s\n",
                step, (long long)seq, h3_prof_p[0], h3_prof_p[1], h3_prof_p[2],
                h3_prof_p[3], wall - acc, wall,
                h3_prof_bytes / 1e9, h3_prof_bytes / 1e9 / (wall / 1e3),
                h3_prof_macs / 1e9, h3_prof_macs / 1e9 / (wall / 1e3));
    }
    free(x); free(ctx); free(imge); free(tb); free(temb);
    free(adaln); free(final_mod); free(scores); free(cos); free(sin);
    *out_vel = vel;
    return true;

fail:
    free(x); free(ctx); free(imge); free(tb); free(t1); free(temb);
    free(adaln); free(final_mod); free(scores); free(cos); free(sin);
    free(vel);
    return false;
}
#undef H3_ROW_TI
#undef H3_ROW_TAG
/* ------------------------------------------------------------------ */
/* VAE encoder                                                         */
/*                                                                     */
/* The encoder is the video model's causal 3D CNN run over a single     */
/* frame. Temporal padding of `kernel_t - 1` zero frames followed by a  */
/* three-tap kernel leaves the frame on the last kernel slice, so every */
/* conv3d here is a 2D convolution reading `weight[..., 2, ky, kx]` —   */
/* slice 0 for the 1x1 shortcuts, whose kernel_t is one. Four spatial   */
/* downsamples produce the 16x latent grid the decoder consumes.        */
/* ------------------------------------------------------------------ */
#define H3_CONV_ROWS 8
#define H3_ENC_LEVELS 6
#define H3_ENC_BLOCKS 2

/* Tile geometry, shared by the encoder and the decoder. `_split_tiles` counts
 * tiles until their overlaps can span the length, then hands the slack out one
 * cell at a time so every boundary stays cell-aligned. `H3_MAX_TILES` covers a
 * 6272 px side, the largest canvas the API accepts. */
#define VAE_TILE_CELLS   16      /* 256 px at this VAE's 16x ratio */
#define VAE_TILE_OVERLAP 4       /* 64 px minimum */
#define H3_MAX_TILES      64

static int64_t h3_vae_split(int64_t len, int64_t *starts, int64_t *overlaps) {
    if (VAE_TILE_CELLS >= len) {
        starts[0] = 0;
        return 1;
    }
    int64_t n = (len + VAE_TILE_CELLS - 1) / VAE_TILE_CELLS;
    while (VAE_TILE_CELLS * n - VAE_TILE_OVERLAP * (n - 1) - len < 0) n++;
    for (int64_t i = 0; i < n - 1; i++) overlaps[i] = VAE_TILE_OVERLAP;
    int64_t remaining = VAE_TILE_CELLS * n - VAE_TILE_OVERLAP * (n - 1) - len;
    for (int64_t i = 0; i < remaining; i++) overlaps[i % (n - 1)] += 1;
    starts[0] = 0;
    for (int64_t i = 1; i < n; i++)
        starts[i] = starts[i - 1] + VAE_TILE_CELLS - overlaps[i - 1];
    return n;
}

/* One 2D convolution, tap-major. The checkpoint stores `[cout][cin][kt]
 * [ky][kx]` with the taps interleaved inside each `(cout, cin)` pair, which no
 * BLAS call can read directly; transposing once per image costs one pass over
 * 239 MB of weights and turns every tap into a plain GEMM. */
typedef struct {
    int64_t cin, cout, side, ntaps;
    float *w;                 /* [ntaps][cin][cout] */
    const float *b;           /* [cout] */
} h3_conv_w;

static void h3_conv_prep(h3_conv_w *cv, const float *w, const float *b,
                         int64_t cout, int64_t cin, int64_t side) {
    const int64_t ntaps = side * side;
    cv->cin = cin; cv->cout = cout; cv->side = side; cv->ntaps = ntaps;
    cv->b = b;
    cv->w = h3_f32(ntaps * cin * cout);
    const int64_t kt = side == 3 ? 3 : 1;
    const int64_t slice = side == 3 ? 2 : 0;
    for (int64_t t = 0; t < ntaps; t++) {
        const int64_t ky = t / side, kx = t % side;
        for (int64_t ci = 0; ci < cin; ci++) {
            float *dst = cv->w + (t * cin + ci) * cout;
            const float *src = w + ((ci * kt + slice) * ntaps) + ky * side + kx;
            for (int64_t co = 0; co < cout; co++)
                dst[co] = src[(size_t)co * cin * kt * ntaps];
        }
    }
}

static void h3_conv_free(h3_conv_w *cv) { free(cv->w); cv->w = NULL; }

/* 3x3 (or 1x1) convolution with reflect padding, [H][W][C] fp32 in and out.
 * Each tap is one GEMM over `H3_CONV_ROWS` rows: eight rows of 128 channels is
 * 4 MB of gather buffer, small enough for the 1024-px levels where a
 * four-dimensional im2col buffer would be 4.7 GB. */
static void h3_conv2d(float *out, const float *in, const h3_conv_w *cv,
                      int64_t W, int64_t H) {
    const int64_t cin = cv->cin, cout = cv->cout;
    const int64_t padw = cv->side / 2;
    const int64_t PW = W + 2 * padw;
    float *pad = NULL;
    const float *src = in;
    int64_t stride = W * cin;
    if (padw) {
        pad = h3_f32((H + 2) * PW * cin);
        for (int64_t y = 0; y < H + 2; y++) {
            int64_t sy = y - 1;
            if (sy < 0) sy = 1;
            if (sy > H - 1) sy = H - 2;
            for (int64_t x = 0; x < PW; x++) {
                int64_t sx = x - 1;
                if (sx < 0) sx = 1;
                if (sx > W - 1) sx = W - 2;
                memcpy(pad + (y * PW + x) * cin, in + (sy * W + sx) * cin,
                       (size_t)cin * sizeof(float));
            }
        }
        src = pad;
        stride = PW * cin;
    }

    for (int64_t i = 0; i < H * W; i++)
        for (int64_t c = 0; c < cout; c++) out[i * cout + c] = cv->b[c];

    float *gather = h3_f32(H3_CONV_ROWS * W * cin);
    for (int64_t t = 0; t < cv->ntaps; t++) {
        const int64_t ky = t / cv->side, kx = t % cv->side;
        const float *B = cv->w + (size_t)t * cin * cout;
        for (int64_t y0 = 0; y0 < H; y0 += H3_CONV_ROWS) {
            const int64_t rn = H - y0 < H3_CONV_ROWS ? H - y0 : H3_CONV_ROWS;
            for (int64_t y = 0; y < rn; y++)
                memcpy(gather + y * W * cin,
                       src + (y0 + y + ky) * stride + kx * cin,
                       (size_t)(W * cin) * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        (int)(rn * W), (int)cout, (int)cin,
                        1.0f, gather, (int)cin, B, (int)cout,
                        1.0f, out + y0 * W * cout, (int)cout);
        }
    }
    free(gather);
    free(pad);
}

/* Stride-2 3x3 convolution. The release pads one pixel on the bottom and right
 * only — reflect, so the last row is duplicated from the second-to-last — and
 * the convolution itself carries no padding, which makes the output exactly
 * `ceil(size / 2)`. */
static void h3_conv2d_s2(float *out, const float *in, const h3_conv_w *cv,
                         int64_t W, int64_t H, int64_t *oW, int64_t *oH) {
    const int64_t cin = cv->cin, cout = cv->cout;
    const int64_t PW = W + 1, PH = H + 1;
    float *pad = h3_f32(PH * PW * cin);
    for (int64_t y = 0; y < PH; y++) {
        const int64_t sy = y < H ? y : H - 2;
        for (int64_t x = 0; x < PW; x++) {
            const int64_t sx = x < W ? x : W - 2;
            memcpy(pad + (y * PW + x) * cin, in + (sy * W + sx) * cin,
                   (size_t)cin * sizeof(float));
        }
    }
    const int64_t OW = (PW - 3) / 2 + 1, OH = (PH - 3) / 2 + 1;

    for (int64_t i = 0; i < OH * OW; i++)
        for (int64_t c = 0; c < cout; c++) out[i * cout + c] = cv->b[c];

    float *gather = h3_f32(H3_CONV_ROWS * OW * cin);
    for (int64_t t = 0; t < 9; t++) {
        const int64_t ky = t / 3, kx = t % 3;
        const float *B = cv->w + (size_t)t * cin * cout;
        for (int64_t y0 = 0; y0 < OH; y0 += H3_CONV_ROWS) {
            const int64_t rn = OH - y0 < H3_CONV_ROWS ? OH - y0 : H3_CONV_ROWS;
            for (int64_t y = 0; y < rn; y++) {
                const float *srcrow =
                    pad + ((y0 + y) * 2 + ky) * PW * cin + kx * cin;
                float *dst = gather + y * OW * cin;
                for (int64_t x = 0; x < OW; x++)
                    memcpy(dst + x * cin, srcrow + x * 2 * cin,
                           (size_t)cin * sizeof(float));
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        (int)(rn * OW), (int)cout, (int)cin,
                        1.0f, gather, (int)cin, B, (int)cout,
                        1.0f, out + y0 * OW * cout, (int)cout);
        }
    }
    free(gather);
    free(pad);
    *oW = OW;
    *oH = OH;
}

/* GroupNorm with the temporal axis already collapsed: statistics over
 * `channels / groups` channels and every pixel of the single frame. */
static void h3_gnorm(float *x, const float *w, const float *b,
                     int64_t n, int64_t C, int64_t groups, float eps) {
    const int64_t gs = C / groups;
    for (int64_t g = 0; g < groups; g++) {
        double sum = 0.0, sq = 0.0;
        for (int64_t i = 0; i < n; i++) {
            const float *p = x + i * C + g * gs;
            for (int64_t c = 0; c < gs; c++) {
                sum += p[c];
                sq += (double)p[c] * (double)p[c];
            }
        }
        const double cnt = (double)n * (double)gs;
        const double mu = sum / cnt;
        double var = sq / cnt - mu * mu;
        if (var < 0.0) var = 0.0;
        const float fm = (float)mu;
        const float fi = (float)(1.0 / sqrt(var + (double)eps));
        for (int64_t i = 0; i < n; i++) {
            float *p = x + i * C + g * gs;
            for (int64_t c = 0; c < gs; c++) {
                const int64_t ch = g * gs + c;
                p[c] = (p[c] - fm) * fi * w[ch] + b[ch];
            }
        }
    }
}

typedef struct {
    const float *n1w, *n1b, *n2w, *n2b;
    h3_conv_w c1, c2, sc;
    bool has_sc;
} h3_enc_res;

typedef struct {
    h3_enc_res res[H3_ENC_BLOCKS];
    h3_conv_w down;
    bool has_down;
} h3_enc_level;

typedef struct {
    h3_conv_w in, out, quant;
    h3_enc_level lv[H3_ENC_LEVELS];
    const float *nw, *nb;
    int64_t ch[H3_ENC_LEVELS];
} h3_enc_w;

#define ET(var, name) do {                                                    \
    int64_t n_ = 0;                                                           \
    if (!h3_standalone_f32(m->vae_map, m->vae_len, name, &(var), &n_, err))    \
        return false;                                                         \
} while (0)

static bool h3_enc_res_load(h3_model *m, h3_enc_res *r, const char *pfx,
                            int64_t cin, int64_t cout, bool shortcut,
                            cwick_error *err) {
    char nm[256];
    const float *w = NULL, *b = NULL;
    memset(r, 0, sizeof(*r));
    snprintf(nm, sizeof(nm), "%s.norm1.weight", pfx); ET(r->n1w, nm);
    snprintf(nm, sizeof(nm), "%s.norm1.bias", pfx);   ET(r->n1b, nm);
    snprintf(nm, sizeof(nm), "%s.norm2.weight", pfx); ET(r->n2w, nm);
    snprintf(nm, sizeof(nm), "%s.norm2.bias", pfx);   ET(r->n2b, nm);
    snprintf(nm, sizeof(nm), "%s.conv1.weight", pfx); ET(w, nm);
    snprintf(nm, sizeof(nm), "%s.conv1.bias", pfx);   ET(b, nm);
    h3_conv_prep(&r->c1, w, b, cout, cin, 3);
    snprintf(nm, sizeof(nm), "%s.conv2.weight", pfx); ET(w, nm);
    snprintf(nm, sizeof(nm), "%s.conv2.bias", pfx);   ET(b, nm);
    h3_conv_prep(&r->c2, w, b, cout, cout, 3);
    r->has_sc = shortcut;
    if (shortcut) {
        snprintf(nm, sizeof(nm), "%s.conv_shortcut.weight", pfx); ET(w, nm);
        snprintf(nm, sizeof(nm), "%s.conv_shortcut.bias", pfx);   ET(b, nm);
        h3_conv_prep(&r->sc, w, b, cout, cin, 1);
    }
    return true;
}

/* The encoder's 116 tensors, resolved once per image. */
static bool h3_enc_load(h3_model *m, h3_enc_w *e, cwick_error *err) {
    static const int64_t CH[H3_ENC_LEVELS] = {128, 256, 256, 512, 512, 1024};
    static const int64_t DST[H3_ENC_LEVELS] = {2, 2, 2, 2, 1, 1};
    char nm[256];
    const float *w = NULL, *b = NULL;
    memset(e, 0, sizeof(*e));
    for (int64_t i = 0; i < H3_ENC_LEVELS; i++) e->ch[i] = CH[i];

    ET(w, "encoder.conv_in.weight");
    ET(b, "encoder.conv_in.bias");
    h3_conv_prep(&e->in, w, b, CH[0], 3, 3);
    ET(w, "encoder.conv_out.weight");
    ET(b, "encoder.conv_out.bias");
    h3_conv_prep(&e->out, w, b, 2 * VAE_LATENTS, CH[5], 3);
    ET(w, "quant_conv.weight");
    ET(b, "quant_conv.bias");
    h3_conv_prep(&e->quant, w, b, 2 * VAE_LATENTS, 2 * VAE_LATENTS, 1);
    ET(e->nw, "encoder.norm_out.weight");
    ET(e->nb, "encoder.norm_out.bias");

    for (int64_t L = 0; L < H3_ENC_LEVELS; L++) {
        const int64_t cin = L == 0 ? CH[0] : CH[L - 1];
        for (int64_t b2 = 0; b2 < H3_ENC_BLOCKS; b2++) {
            snprintf(nm, sizeof(nm), "encoder.down_blocks.%lld.resnets.%lld",
                     (long long)L, (long long)b2);
            const bool sc = (b2 == 0 && cin != CH[L]);
            if (!h3_enc_res_load(m, &e->lv[L].res[b2], nm,
                                 b2 == 0 ? cin : CH[L], CH[L], sc, err))
                return false;
        }
        e->lv[L].has_down = DST[L] > 1;
        if (e->lv[L].has_down) {
            snprintf(nm, sizeof(nm), "encoder.down_blocks.%lld.downsamplers.0.conv.weight",
                     (long long)L);
            ET(w, nm);
            snprintf(nm, sizeof(nm), "encoder.down_blocks.%lld.downsamplers.0.conv.bias",
                     (long long)L);
            ET(b, nm);
            h3_conv_prep(&e->lv[L].down, w, b, CH[L], CH[L], 3);
        }
    }
    return true;
}

static void h3_enc_free(h3_enc_w *e) {
    h3_conv_free(&e->in);
    h3_conv_free(&e->out);
    h3_conv_free(&e->quant);
    for (int64_t L = 0; L < H3_ENC_LEVELS; L++) {
        for (int64_t b2 = 0; b2 < H3_ENC_BLOCKS; b2++) {
            h3_conv_free(&e->lv[L].res[b2].c1);
            h3_conv_free(&e->lv[L].res[b2].c2);
            if (e->lv[L].res[b2].has_sc) h3_conv_free(&e->lv[L].res[b2].sc);
        }
        if (e->lv[L].has_down) h3_conv_free(&e->lv[L].down);
    }
}
#undef ET

/* One tile through the encoder. `px` is [H][W][3] already normalized the way
 * the VAE expects; `mom` receives [2*latents][H/16][W/16] channel-major.
 *
 * Three buffers hold the activation: `cur`, the residual copy `res`, and the
 * scratch `wrk`. A tile is at most 256 px, so the widest buffer is
 * 256 x 256 x 128 floats. */
static void h3_enc_tile(const h3_enc_w *e, const float *px, int64_t W, int64_t H,
                        float *mom) {
    const int64_t cap = 256 * 256 * 128;
    float *ba = h3_f32(cap), *bb = h3_f32(cap), *bc = h3_f32(cap);
    float *cur = ba, *res = bb, *wrk = bc;
    int64_t cw = W, chh = H;

    h3_conv2d(cur, px, &e->in, W, H);

    for (int64_t L = 0; L < H3_ENC_LEVELS; L++) {
        for (int64_t b = 0; b < H3_ENC_BLOCKS; b++) {
            const h3_enc_res *r = &e->lv[L].res[b];
            const int64_t cin = b == 0 ? (L == 0 ? e->ch[0] : e->ch[L - 1])
                                       : e->ch[L];
            const int64_t cout = e->ch[L];
            if (r->has_sc) {
                h3_conv2d(res, cur, &r->sc, cw, chh);
            } else {
                memcpy(res, cur, (size_t)(cw * chh * cin) * sizeof(float));
            }
            memcpy(wrk, cur, (size_t)(cw * chh * cin) * sizeof(float));
            h3_gnorm(wrk, r->n1w, r->n1b, cw * chh, cin, 32, 1e-6f);
            h3_silu(wrk, cw * chh * cin);
            h3_conv2d(cur, wrk, &r->c1, cw, chh);
            h3_gnorm(cur, r->n2w, r->n2b, cw * chh, cout, 32, 1e-6f);
            h3_silu(cur, cw * chh * cout);
            h3_conv2d(wrk, cur, &r->c2, cw, chh);
            for (int64_t i = 0; i < cw * chh * cout; i++) cur[i] = res[i] + wrk[i];
        }
        if (e->lv[L].has_down) {
            int64_t ow = 0, oh = 0;
            h3_conv2d_s2(wrk, cur, &e->lv[L].down, cw, chh, &ow, &oh);
            float *t = cur; cur = wrk; wrk = t;
            cw = ow; chh = oh;
        }
    }

    h3_gnorm(cur, e->nw, e->nb, cw * chh, e->ch[H3_ENC_LEVELS - 1], 32, 1e-6f);
    h3_silu(cur, cw * chh * e->ch[H3_ENC_LEVELS - 1]);
    h3_conv2d(wrk, cur, &e->out, cw, chh);
    h3_conv2d(res, wrk, &e->quant, cw, chh);
    for (int64_t c = 0; c < 2 * VAE_LATENTS; c++)
        for (int64_t i = 0; i < cw * chh; i++)
            mom[(size_t)c * cw * chh + i] =
                res[(size_t)i * 2 * VAE_LATENTS + c];

    free(ba);
    free(bb);
    free(bc);
}

/* Cross-fade the last `e` lines of `prev` into the first `e` of `cur`,
 * channel-major [C][h][w], with the decoder's `1 - i/e` / `i/e` weights. As in
 * `h3_vae_blend`, the weight belongs to the offset inside the band: the
 * horizontal branch carries `i` down to the pixel, since blending the whole
 * strip once per offset leaves only the last offset's weight on it. */
static void h3_enc_blend(const float *prev, float *cur, int64_t e,
                         int64_t C, int64_t w, int64_t h, bool vertical) {
    if (e <= 0) return;
    if (e > h) e = h;
    if (e > w) e = w;
    for (int64_t c = 0; c < C; c++) {
        const float *pa = prev + (size_t)c * h * w;
        float *pb = cur + (size_t)c * h * w;
        for (int64_t i = 0; i < e; i++) {
            const float wa = 1.0f - (float)i / (float)e;
            const float wb = (float)i / (float)e;
            if (vertical) {
                const float *s = pa + (h - e + i) * w;
                float *d = pb + i * w;
                for (int64_t x = 0; x < w; x++) d[x] = s[x] * wa + d[x] * wb;
            } else {
                for (int64_t y = 0; y < h; y++) {
                    const float *s = pa + y * w + (w - e + i);
                    float *d = pb + y * w + i;
                    *d = *s * wa + *d * wb;
                }
            }
        }
    }
}

/* Round-trip through float16, which is what the released recipe stores the
 * posterior sample in before cropping it to whole patches. */
static float h3_round_f16(float v) {
    _Float16 h = (_Float16)v;
    return (float)h;
}

/* Encode one reference image. `rgb` is [H][W][3] in [0,1] at the size the VAE
 * reads — both sides whole VAE cells, which the caller guarantees. `noise` is
 * one draw of the latent shape. Returns `[24][lh][lw]` channel-major,
 * normalized by `latents_mean` / `latents_std`.
 *
 * The encoder tiles like the decoder, and the tiles are stitched in latent
 * space: a reference is at most 2048 px on its long side, so the whole tile
 * grid of a 128x128 latent is 64 tiles of 16x16x48, which is cheap to hold. */
static bool h3_vae_encode(h3_model *m, const float *rgb, int64_t W, int64_t H,
                          const float *noise, float **out_lat, cwick_error *err) {
    static const float pmean[3] = {0.485f, 0.456f, 0.406f};
    static const float pstd[3] = {0.229f, 0.224f, 0.225f};
    if (W % H3_VAE_RATIO || H % H3_VAE_RATIO || W < 256 || H < 256) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "reference image %lldx%lld is not whole VAE cells",
                        (long long)W, (long long)H);
        return false;
    }
    h3_enc_w e;
    if (!h3_enc_load(m, &e, err)) return false;

    const int64_t lw = W / H3_VAE_RATIO, lh = H / H3_VAE_RATIO;
    const int64_t cells = VAE_TILE_CELLS;         /* 16 cells = 256 px */
    int64_t sh[H3_MAX_TILES], hov[H3_MAX_TILES];
    int64_t sw[H3_MAX_TILES], wov[H3_MAX_TILES];
    const int64_t nth = h3_vae_split(lh, sh, hov);
    const int64_t ntw = h3_vae_split(lw, sw, wov);
    const int64_t ncell = cells * cells * 2 * VAE_LATENTS;
    float *mom = h3_f32((int64_t)2 * VAE_LATENTS * lw * lh);
    float *tiles = h3_f32(nth * ntw * ncell);
    float *px = h3_f32(256 * 256 * 3);

    for (int64_t i = 0; i < nth; i++) {
        for (int64_t j = 0; j < ntw; j++) {
            const int64_t px0 = sw[j] * H3_VAE_RATIO, py0 = sh[i] * H3_VAE_RATIO;
            const int64_t tw = cells * H3_VAE_RATIO, th = cells * H3_VAE_RATIO;
            for (int64_t y = 0; y < th; y++)
                for (int64_t x = 0; x < tw; x++) {
                    const float *s = rgb + ((py0 + y) * W + px0 + x) * 3;
                    float *d = px + (y * tw + x) * 3;
                    for (int64_t c = 0; c < 3; c++)
                        d[c] = (s[c] - pmean[c]) / pstd[c];
                }
            h3_enc_tile(&e, px, tw, th, tiles + (i * ntw + j) * ncell);
        }
    }
    h3_enc_free(&e);

    /* Stitch, in the released order: cross-fade against the raw tile above,
     * then against the raw tile to the left, then drop the far edges. */
    for (int64_t i = 0; i < nth; i++) {
        for (int64_t j = 0; j < ntw; j++) {
            float *cur = h3_f32(ncell);
            memcpy(cur, tiles + (i * ntw + j) * ncell,
                   (size_t)ncell * sizeof(float));
            if (i > 0)
                h3_enc_blend(tiles + ((i - 1) * ntw + j) * ncell, cur,
                             hov[i - 1], 2 * VAE_LATENTS, cells, cells, true);
            if (j > 0)
                h3_enc_blend(tiles + (i * ntw + j - 1) * ncell, cur,
                             wov[j - 1], 2 * VAE_LATENTS, cells, cells, false);
            const int64_t ow = j < ntw - 1 ? cells - wov[j] : cells;
            const int64_t oh = i < nth - 1 ? cells - hov[i] : cells;
            for (int64_t c = 0; c < 2 * VAE_LATENTS; c++)
                for (int64_t y = 0; y < oh; y++)
                    memcpy(mom + ((size_t)c * lh + sh[i] + y) * lw + sw[j],
                           cur + ((size_t)c * cells + y) * cells,
                           (size_t)ow * sizeof(float));
            free(cur);
        }
    }
    free(tiles);
    free(px);
    h3_dump("c_enc_mom", mom, (int64_t)2 * VAE_LATENTS * lw * lh);

    /* Diagonal Gaussian: sample, round to float16, crop to whole patches, then
     * normalize. `mean` and `logvar` are the two halves of `mom`. */
    float *lat = h3_f32((int64_t)VAE_LATENTS * lw * lh);
    static const float lmean[24] = {
        0.858090341091156f, -0.9606591463088989f, 1.0661640167236328f,
        -0.5090325474739075f, -0.2727581858634949f, -1.3675414323806763f,
        -0.2553254961967468f, -0.2690755426883697f, -0.5376840829849243f,
        -0.0464097298681736f, 0.6657370328903198f, 0.196901276707649f,
        -0.5460608005523682f, -0.4035342037677765f, -0.2368302494287491f,
        0.2592845261096954f, -0.3013394474983215f, 0.2113419920206070f,
        -1.120684862136841f, 0.3581933379173279f, -0.0422514379024505f,
        0.2604829967021942f, 0.2286409288644790f, 0.7056031823158264f};
    static const float lstd[24] = {
        1.2223774194717407f, 1.2767263650863647f, 1.6831774711608887f,
        1.7549455165863037f, 1.5636216402053833f, 2.194143533706665f,
        0.9653137922286987f, 1.0569885969161987f, 0.841948926448822f,
        0.7729952931404114f, 1.8955937623977661f, 0.946841835975647f,
        0.7996809482574463f, 0.449889004230497f, 0.7197399735450743f,
        0.6936293244361877f, 2.961095094680786f, 2.769419908523559f,
        3.049618482589722f, 2.1088054180145264f, 3.276226282119751f,
        3.1627357006073f, 2.2816812992095947f, 2.6127843856811523f};
    for (int64_t c = 0; c < VAE_LATENTS; c++) {
        for (int64_t i = 0; i < lw * lh; i++) {
            const float mu = mom[(size_t)c * lw * lh + i];
            float lv = mom[(size_t)(c + VAE_LATENTS) * lw * lh + i];
            if (lv < -30.0f) lv = -30.0f;
            if (lv > 20.0f) lv = 20.0f;
            const float sd = expf(0.5f * lv);
            const float s = h3_round_f16(mu + sd * noise[(size_t)c * lw * lh + i]);
            lat[(size_t)c * lw * lh + i] = (s - lmean[c]) / lstd[c];
        }
    }
    free(mom);
    h3_dump("c_enc_latent", lat, (int64_t)VAE_LATENTS * lw * lh);
    *out_lat = lat;
    return true;
}

/* ------------------------------------------------------------------ */
/* VAE decoder: a 24-layer 2D ViT over latent pixels, not a CNN stack.  */
/* Every latent cell becomes one token; four learned registers and one  */
/* zero token are appended at position 0 and dropped before the patch    */
/* projection.                                                          */
/* ------------------------------------------------------------------ */
/* One tile of the ViT decoder. `zt` is a contiguous [lw*lh, VAE_LATENTS] slab
 * of already denormalised, post-quant-convolved latents; `rgb` receives
 * lw*H3_VAE_RATIO by lh*H3_VAE_RATIO interleaved pixels. `mark` enables the
 * stage dumps for one tile only, since a run decodes many. */
typedef struct {
    h3_mat pi, pib, po_w, pob, no, nob, regs;
} h3_vae_stem;

typedef struct {
    h3_mat n1, n2, qw, qb, kw, kb, vw, vb, ow, ob, w1, b1, w2, b2, s1, s2;
} h3_vae_block;

typedef struct {
    h3_vae_stem stem;
    h3_vae_block blk[VAE_LAYERS];
} h3_vae_weights;

/* Resolve all 391 decoder tensors once per image. Every one of them aliases
 * the mapping, so a tiled decode re-reads the same pages from the file cache
 * instead of copying 1.6 GiB per tile; the header is also parsed once rather
 * than 24 times per tile. */
#define VAE_MAT(var, name, r, c) do {                                         \
    const float *vp_ = NULL; int64_t vn_ = 0;                                 \
    if (!h3_standalone_f32(m->vae_map, m->vae_len, name, &vp_, &vn_, err) ||   \
        vn_ != (r) * (c)) {                                                    \
        cwick_error_set(err, CWICK_ERR_STATE, "bad VAE tensor %s", name);       \
        return false;                                                          \
    }                                                                          \
    (var).p = vp_; (var).rows = (r); (var).cols = (c); (var).f32 = true;        \
} while (0)

static bool h3_vae_weights_load(h3_model *m, h3_vae_weights *w, cwick_error *err) {
    const int64_t D = VAE_HIDDEN, FF = VAE_FFN;
    w->stem = (h3_vae_stem){0};
    memset(w->blk, 0, sizeof(w->blk));
    VAE_MAT(w->stem.pi, "decoder.proj_in.weight", D, VAE_LATENTS);
    VAE_MAT(w->stem.pib, "decoder.proj_in.bias", D, 1);
    VAE_MAT(w->stem.po_w, "decoder.proj_out.weight", 3 * H3_VAE_RATIO * H3_VAE_RATIO, D);
    VAE_MAT(w->stem.pob, "decoder.proj_out.bias", 3 * H3_VAE_RATIO * H3_VAE_RATIO, 1);
    VAE_MAT(w->stem.no, "decoder.norm_out.weight", D, 1);
    VAE_MAT(w->stem.nob, "decoder.norm_out.bias", D, 1);
    VAE_MAT(w->stem.regs, "decoder.register_tokens", VAE_REGISTERS, D);
    static const char *keys[16] = {
        "norm1.weight", "norm2.weight",
        "attn.to_q.weight", "attn.to_q.bias", "attn.to_k.weight", "attn.to_k.bias",
        "attn.to_v.weight", "attn.to_v.bias", "attn.to_out.0.weight", "attn.to_out.0.bias",
        "ff.net.0.proj.weight", "ff.net.0.proj.bias", "ff.net.2.weight", "ff.net.2.bias",
        "scale1", "scale2"};
    static const int64_t rows[16] = {
        D, D, D, D, D, D, D, D, D, D, 2 * FF, 2 * FF, D, D, D, D};
    static const int64_t cols[16] = {
        1, 1, D, 1, D, 1, D, 1, D, 1, D, 1, FF, 1, 1, 1};
    for (int64_t L = 0; L < VAE_LAYERS; L++) {
        h3_mat *mats[16] = {
            &w->blk[L].n1, &w->blk[L].n2, &w->blk[L].qw, &w->blk[L].qb,
            &w->blk[L].kw, &w->blk[L].kb, &w->blk[L].vw, &w->blk[L].vb,
            &w->blk[L].ow, &w->blk[L].ob, &w->blk[L].w1, &w->blk[L].b1,
            &w->blk[L].w2, &w->blk[L].b2, &w->blk[L].s1, &w->blk[L].s2};
        char nm[192];
        for (int ki = 0; ki < 16; ki++) {
            snprintf(nm, sizeof(nm), "decoder.transformer_blocks.%lld.%s",
                     (long long)L, keys[ki]);
            h3_mat *d = mats[ki];
            d->rows = rows[ki]; d->cols = cols[ki]; d->f32 = true;
            const float *vp = NULL; int64_t vn = 0;
            if (!h3_standalone_f32(m->vae_map, m->vae_len, nm, &vp, &vn, err) ||
                vn != rows[ki] * cols[ki]) {
                cwick_error_set(err, CWICK_ERR_STATE, "bad VAE tensor %s", nm);
                return false;
            }
            d->p = vp;
        }
    }
    return true;
}
#undef VAE_MAT

static void h3_vae_decode_tile(const h3_vae_weights *w, const float *zt,
                               int lw, int lh, float *rgb, bool mark) {
    const int64_t D = VAE_HIDDEN, NH = VAE_HEADS, HD = VAE_HEAD_DIM;
    const int64_t FF = VAE_FFN, RD = VAE_ROPE_DIM;
    int64_t np = (int64_t)lw * lh;
    int64_t seq = np + VAE_REGISTERS + 1;

    const float *z = zt;

    float *x = h3_f32(seq * D), *norm = h3_f32(seq * D);
    float *q = h3_f32(seq * D), *k = h3_f32(seq * D), *v = h3_f32(seq * D);
    float *att = h3_f32(seq * D), *po = h3_f32(seq * D);
    float *g1 = h3_f32(seq * FF), *g2 = h3_f32(seq * FF);
    float *scores = h3_f32(H3_ATTN_ROWS * (seq + 1));
    float *cos = h3_f32(seq * RD), *sin = h3_f32(seq * RD);

    h3_linear(x, z, w->stem.pi, (const float *)w->stem.pib.p, VAE_LATENTS, np);
    if (mark) h3_dump("c_vae_proj_in", x, np * D);
    memcpy(x + np * D, w->stem.regs.p, (size_t)VAE_REGISTERS * D * sizeof(float));
    h3_zero(x + (np + VAE_REGISTERS) * D, D);

    /* 2-axis rotary over the latent grid: coordinates in [-1, 1), scaled by
     * 2*pi, theta 100, eight frequencies per axis, tiled twice. */
    {
        /* `1 / theta ** arange(0, 1, 2 * axes / dim)` with dim = 2 * axes *
         * freq: eight frequencies per axis, spaced by theta^(1/8). */
        float inv[VAE_ROPE_FREQ];
        for (int64_t j = 0; j < VAE_ROPE_FREQ; j++)
            inv[j] = powf(1.0f / VAE_ROPE_THETA, (float)j / (float)VAE_ROPE_FREQ);
        for (int64_t y = 0; y < lh; y++) {
            for (int64_t xx = 0; xx < lw; xx++) {
                int64_t s = y * lw + xx;
                float py = 2.0f * ((float)y + 0.5f) / (float)lh - 1.0f;
                float px = 2.0f * ((float)xx + 0.5f) / (float)lw - 1.0f;
                for (int64_t j = 0; j < VAE_ROPE_FREQ; j++) {
                    float ah = 2.0f * (float)M_PI * py * inv[j];
                    float aw = 2.0f * (float)M_PI * px * inv[j];
                    cos[s * RD + j] = cosf(ah);
                    cos[s * RD + VAE_ROPE_FREQ + j] = cosf(aw);
                    sin[s * RD + j] = sinf(ah);
                    sin[s * RD + VAE_ROPE_FREQ + j] = sinf(aw);
                }
                for (int64_t j = 0; j < 2 * VAE_ROPE_FREQ; j++) {
                    cos[s * RD + 2 * VAE_ROPE_FREQ + j] = cos[s * RD + j];
                    sin[s * RD + 2 * VAE_ROPE_FREQ + j] = sin[s * RD + j];
                }
            }
        }
        for (int64_t s = np; s < seq; s++) {
            for (int64_t j = 0; j < RD; j++) { cos[s * RD + j] = 1.0f; sin[s * RD + j] = 0.0f; }
        }
    }

    for (int64_t L = 0; L < VAE_LAYERS; L++) {
        const h3_vae_block *b = &w->blk[L];
        for (int64_t s = 0; s < seq; s++)
            h3_rmsnorm(norm + s * D, x + s * D, (const float *)b->n1.p, D, 1e-5f);
        h3_linear(q, norm, b->qw, (const float *)b->qb.p, D, seq);
        h3_linear(k, norm, b->kw, (const float *)b->kb.p, D, seq);
        h3_linear(v, norm, b->vw, (const float *)b->vb.p, D, seq);
        {
            float *one = h3_f32(HD);
            for (int64_t d = 0; d < HD; d++) one[d] = 1.0f;
            h3_rmsnorm_heads(q, one, seq, NH, HD, 1e-5f);
            h3_rmsnorm_heads(k, one, seq, NH, HD, 1e-5f);
            free(one);
        }
        h3_rope(q, cos, sin, seq, NH, HD, RD);
        h3_rope(k, cos, sin, seq, NH, HD, RD);
        h3_attention(att, q, k, v, seq, NH, HD, false, scores);
        h3_linear(po, att, b->ow, (const float *)b->ob.p, D, seq);
        for (int64_t s = 0; s < seq; s++)
            for (int64_t d = 0; d < D; d++)
                x[s * D + d] += po[s * D + d] * ((const float *)b->s1.p)[d];

        for (int64_t s = 0; s < seq; s++)
            h3_rmsnorm(norm + s * D, x + s * D, (const float *)b->n2.p, D, 1e-5f);
        {
            /* `ff.net.0.proj` holds [value; gate] stacked on the output axis,
             * so the gate half is the same pointer shifted by FF rows. */
            h3_mat wv = b->w1, wg = b->w1;
            wv.rows = FF; wg.rows = FF;
            wg.p = (const char *)b->w1.p + (size_t)FF * D * 4;
            h3_linear(g1, norm, wv, (const float *)b->b1.p, D, seq);
            h3_linear(g2, norm, wg, (const float *)b->b1.p + FF, D, seq);
        }
        h3_swiglu(g1, g2, seq * FF);
        h3_linear(po, g1, b->w2, (const float *)b->b2.p, FF, seq);
        for (int64_t s = 0; s < seq; s++)
            for (int64_t d = 0; d < D; d++)
                x[s * D + d] += po[s * D + d] * ((const float *)b->s2.p)[d];

        if (mark && L == 0) h3_dump("c_vae_block0", x, seq * D);
        if (mark && L == VAE_LAYERS - 1) h3_dump("c_vae_blocklast", x, seq * D);
    }

    /* LayerNorm, then the patch projection: [np, 3*16*16] -> an image. */
    {
        float *px = h3_f32(np * 3 * H3_VAE_RATIO * H3_VAE_RATIO);
        for (int64_t s = 0; s < np; s++) {
            float mu = 0.0f, va = 0.0f;
            for (int64_t d = 0; d < D; d++) mu += x[s * D + d];
            mu /= (float)D;
            for (int64_t d = 0; d < D; d++) { float e = x[s * D + d] - mu; va += e * e; }
            va /= (float)D;
            float inv = 1.0f / sqrtf(va + 1e-5f);
            for (int64_t d = 0; d < D; d++)
                norm[s * D + d] = (x[s * D + d] - mu) * inv * ((const float *)w->stem.no.p)[d]
                                + ((const float *)w->stem.nob.p)[d];
        }
        h3_linear(px, norm, w->stem.po_w, (const float *)w->stem.pob.p, D, np);
        if (mark) h3_dump("c_vae_proj_out", px, np * 3 * H3_VAE_RATIO * H3_VAE_RATIO);
        const float pm[3] = {0.485f, 0.456f, 0.406f}, ps[3] = {0.229f, 0.224f, 0.225f};
        int ow = lw * H3_VAE_RATIO;
        for (int64_t y = 0; y < lh; y++)
            for (int64_t xx = 0; xx < lw; xx++) {
                int64_t s = y * lw + xx;
                for (int dy = 0; dy < H3_VAE_RATIO; dy++)
                    for (int dx = 0; dx < H3_VAE_RATIO; dx++)
                        for (int c = 0; c < 3; c++) {
                            int64_t pi2 = ((((int64_t)y * H3_VAE_RATIO + dy) * ow) +
                                           xx * H3_VAE_RATIO + dx) * 3 + c;
                            int64_t li = s * (3 * H3_VAE_RATIO * H3_VAE_RATIO) +
                                         c * H3_VAE_RATIO * H3_VAE_RATIO + dy * H3_VAE_RATIO + dx;
                            float val = px[li] * ps[c] + pm[c];
                            /* Clamped here so the cross-fade and the resize
                             * never see an out-of-range sample, but not
                             * quantised: the released recipe blends float
                             * tiles and truncates to 8 bits once at the end,
                             * and rounding before the blend costs up to two
                             * levels across an overlap band. */
                            rgb[pi2] = h3_clamp(val, 0.0f, 1.0f);
                        }
            }
        free(px);
    }

    free(x); free(norm); free(q); free(k); free(v);
    free(att); free(po); free(g1); free(g2); free(scores); free(cos); free(sin);
}

/* Tiled decode. The released decoder ships with spatial tiling on — 256 px
 * tiles with a 64 px minimum overlap, linearly cross-faded — and the released
 * images are the tiled ones. Decoding a whole 128x128 latent in one pass puts
 * the ViT far outside the 16x16 context it was released for and returns a
 * per-cell modulation that is plainly visible; the tiled decode is both the
 * faithful result and about eight times less work, since attention within a
 * tile is 16x16 rather than 128x128.
 *
 * Geometry follows the released `_split_tiles`: tiles are counted until their
 * overlaps can span the length, then the slack is handed out one cell at a
 * time so every boundary stays cell-aligned. */
/* Cross-fade the last `e` lines of `prev` into the first `e` lines of `cur`.
 * Same weights as the released decoder: `1 - i/e` and `i/e`. The weight
 * belongs to the offset inside the band, so the horizontal branch has to
 * carry it down to the pixel: blending the whole strip once per offset leaves
 * only the last offset's weight applied, which turned the cross-fade into a
 * hard cut on the left tile's side of the seam. */
static void h3_vae_blend(const float *prev, float *cur, int64_t e,
                         int64_t w, int64_t h, bool vertical) {
    if (e <= 0) return;
    if (e > h) e = h;
    if (e > w) e = w;
    const float inv = 1.0f / (float)e;
    for (int64_t i = 0; i < e; i++) {
        const float wa = 1.0f - (float)i * inv;
        const float wb = (float)i * inv;
        if (vertical) {
            const float *a = prev + (h - e + i) * w * 3;
            float *b = cur + i * w * 3;
            for (int64_t x = 0; x < w * 3; x++) b[x] = a[x] * wa + b[x] * wb;
        } else {
            for (int64_t y = 0; y < h; y++) {
                const float *a = prev + (y * w + (w - e + i)) * 3;
                float *b = cur + (y * w + i) * 3;
                for (int64_t c = 0; c < 3; c++) b[c] = a[c] * wa + b[c] * wb;
            }
        }
    }
}

static bool h3_vae_decode(h3_model *m, const float *latent, int lw, int lh,
                          float **rgb, cwick_error *err) {
    static const float mean[24] = {
        0.858090341091156, -0.9606591463088989, 1.0661640167236328,
       -0.5090325474739075, -0.2727581858634949, -1.3675414323806763,
       -0.2553254961967468, -0.26907554268836975, -0.5376840829849243,
       -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
       -0.5460608005523682, -0.4035342037677765, -0.23683024942874908,
        0.25928452610969543, -0.30133944749832153, 0.211341992020607,
       -1.1206848621368408, 0.3581933379173279, -0.04225143790245056,
        0.2604829967021942, 0.22864092886447906, 0.7056031823158264};
    static const float stdv[24] = {
        1.2223774194717407, 1.2767263650894165, 1.6831774711608887,
        1.7549455165863037, 1.5636216402053833, 2.194143533706665,
        0.9653137922286987, 1.0569885969161987, 0.841948926448822,
        0.7729952931404114, 1.8955937623977661, 0.946841835975647,
        0.7996809482574463, 0.44988900423049927, 0.7197399735450743,
        0.6936293244361877, 2.961095094680786, 2.7694199085235596,
        3.0496184825897217, 2.1088054180145264, 3.276226282119751,
        3.1627357006073, 2.2816812992095947, 2.6127843856811523};

    int64_t np = (int64_t)lw * lh;
    /* `latent` arrives channel-major from the unpatchify step, while the
     * decoder consumes one token per latent cell. Denormalise and apply the
     * 1x1 post-quant convolution once, before any tile is cut: both are
     * per-cell, so tiling afterwards is exact. */
    float *z = h3_f32(np * VAE_LATENTS);
    for (int64_t i = 0; i < np; i++)
        for (int64_t c = 0; c < VAE_LATENTS; c++)
            z[i * VAE_LATENTS + c] = latent[c * np + i] * stdv[c] + mean[c];
    {
        const float *w = NULL, *b = NULL;
        int64_t nw = 0, nb = 0;
        bool have = h3_standalone_f32(m->vae_map, m->vae_len, "post_quant_conv.weight", &w, &nw, err) &&
                    h3_standalone_f32(m->vae_map, m->vae_len, "post_quant_conv.bias", &b, &nb, err) &&
                    nw == VAE_LATENTS * VAE_LATENTS && nb == VAE_LATENTS;
        if (!have) {
            free(z);
            cwick_error_set(err, CWICK_ERR_STATE, "unexpected post_quant_conv shape");
            return false;
        }
        for (int64_t i = 0; i < np; i++) {
            float v[VAE_LATENTS];
            for (int64_t c = 0; c < VAE_LATENTS; c++) {
                float acc = b[c];
                for (int64_t j = 0; j < VAE_LATENTS; j++)
                    acc += w[c * VAE_LATENTS + j] * z[i * VAE_LATENTS + j];
                v[c] = acc;
            }
            memcpy(z + i * VAE_LATENTS, v, sizeof(v));
        }
    }

    h3_vae_weights wts;
    if (!h3_vae_weights_load(m, &wts, err)) {
        free(z);
        return false;
    }

    /* A short axis is one tile wide; otherwise every tile is the full 16 cells,
     * because the split distributes its slack so the last tile ends flush. */
    int64_t cells_w = lw < VAE_TILE_CELLS ? lw : VAE_TILE_CELLS;
    int64_t cells_h = lh < VAE_TILE_CELLS ? lh : VAE_TILE_CELLS;
    int64_t max_tiles = 512;
    int64_t *sh = cwick_xmalloc((size_t)max_tiles * sizeof(int64_t));
    int64_t *sw = cwick_xmalloc((size_t)max_tiles * sizeof(int64_t));
    int64_t *hov = cwick_xmalloc((size_t)max_tiles * sizeof(int64_t));
    int64_t *wov = cwick_xmalloc((size_t)max_tiles * sizeof(int64_t));
    int64_t nth = h3_vae_split(lh, sh, hov);
    int64_t ntw = h3_vae_split(lw, sw, wov);
    if (h3_prof_on()) {
        fprintf(stderr, "cwick h3: vae %lldx%lld tiles:",
                (long long)nth, (long long)ntw);
        for (int64_t t = 0; t < ntw; t++)
            fprintf(stderr, " %lld+%lld", (long long)(sw[t] * H3_VAE_RATIO),
                    (long long)(wov[t] * H3_VAE_RATIO));
        fprintf(stderr, "\n");
    }
    if (nth > max_tiles || ntw > max_tiles) {
        free(sh); free(sw); free(hov); free(wov); free(z);
        cwick_error_set(err, CWICK_ERR_STATE, "vae tile count out of range");
        return false;
    }

    const int64_t tw = cells_w * H3_VAE_RATIO;             /* tile px */
    const int64_t th = cells_h * H3_VAE_RATIO;
    const int64_t ow = (int64_t)lw * H3_VAE_RATIO;
    const int64_t oh = (int64_t)lh * H3_VAE_RATIO;

    float *out = h3_f32(ow * oh * 3);
    float *raw[2] = {h3_f32(ntw * tw * th * 3), h3_f32(ntw * tw * th * 3)};
    float *cur = h3_f32(tw * th * 3);
    float *gather = h3_f32(cells_w * cells_h * VAE_LATENTS);

    for (int64_t i = 0; i < nth; i++) {
        float *row_raw = raw[i & 1];
        for (int64_t j = 0; j < ntw; j++) {
            for (int64_t y = 0; y < cells_h; y++)
                for (int64_t x = 0; x < cells_w; x++)
                    for (int64_t c = 0; c < VAE_LATENTS; c++)
                        gather[(y * cells_w + x) * VAE_LATENTS + c] =
                            z[((sh[i] + y) * lw + sw[j] + x) * VAE_LATENTS + c];
            float *tile = row_raw + j * tw * th * 3;
            h3_vae_decode_tile(&wts, gather, (int)cells_w, (int)cells_h,
                               tile, i == 0 && j == 0);
            if (h3_prof_on() && j == 0) {
                struct rusage ru;
                getrusage(RUSAGE_SELF, &ru);
                fprintf(stderr, "cwick h3: vae row %lld/%lld, %lld tiles, peak rss %lld MB\n",
                        (long long)i, (long long)nth, (long long)ntw,
                        (long long)(ru.ru_maxrss / (1024 * 1024)));
            }
        }

        /* A row of raw tiles before stitching. Diffing these through the
         * released `_stitch_tiles` is what located the blend bug; keeping the
         * hook makes the same comparison repeatable after a change. */
        if (h3_dump_on()) {
            char nm2[64];
            snprintf(nm2, sizeof(nm2), "c_rawrow%02lld", (long long)i);
            h3_dump(nm2, row_raw, ntw * tw * th * 3);
        }
        /* Stitch this row: the released decoder cross-fades each tile against
         * the raw tile above and the raw tile to its left, then drops the far
         * edges so neighbours abut without counting a region twice. */        for (int64_t j = 0; j < ntw; j++) {
            const float *src = row_raw + j * tw * th * 3;
            memcpy(cur, src, (size_t)(tw * th * 3) * sizeof(float));
            if (i > 0)
                h3_vae_blend(raw[(i - 1) & 1] + j * tw * th * 3, cur,
                             hov[i - 1] * H3_VAE_RATIO, tw, th, true);
            if (j > 0)
                h3_vae_blend(row_raw + (j - 1) * tw * th * 3, cur,
                             wov[j - 1] * H3_VAE_RATIO, tw, th, false);
            int64_t copy_w = tw - (j < ntw - 1 ? wov[j] * H3_VAE_RATIO : 0);
            int64_t copy_h = th - (i < nth - 1 ? hov[i] * H3_VAE_RATIO : 0);
            int64_t x0 = sw[j] * H3_VAE_RATIO, y0 = sh[i] * H3_VAE_RATIO;
            if (x0 + copy_w > ow) copy_w = ow - x0;
            if (y0 + copy_h > oh) copy_h = oh - y0;
            if (copy_w <= 0 || copy_h <= 0) continue;
            for (int64_t y = 0; y < copy_h; y++)
                memcpy(out + ((y0 + y) * ow + x0) * 3, cur + y * tw * 3,
                       (size_t)(copy_w * 3) * sizeof(float));
        }
    }

    free(raw[0]); free(raw[1]); free(cur); free(gather);
    free(sh); free(sw); free(hov); free(wov); free(z);
    *rgb = out;
    return true;
}

/* ------------------------------------------------------------------ */
/* Generation                                                          */
/* ------------------------------------------------------------------ */
static uint64_t h3_rand(uint64_t *s) {
    *s += 0x9e3779b97f4a7c15ULL;
    uint64_t z = *s;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/* The sampler starts from a standard normal draw, not a uniform one. Box-Muller
 * over SplitMix64 keeps the bytes identical on every machine for a given seed.
 * The state is passed in so a request can keep drawing off one stream: the
 * generated rows first, then one reference block at a time, which is the order
 * the reference implementation draws them in. */
static void h3_noise_at(float *dst, int64_t n, uint64_t *state) {
    for (int64_t i = 0; i < n; i += 2) {
        double u1 = ((double)(h3_rand(state) >> 11) + 0.5) / 9007199254740992.0;
        double u2 = (double)(h3_rand(state) >> 11) / 9007199254740992.0;
        double r = sqrt(-2.0 * log(u1));
        double th = 6.283185307179586 * u2;
        dst[i] = (float)(r * cos(th));
        if (i + 1 < n) dst[i + 1] = (float)(r * sin(th));
    }
}

static uint64_t h3_noise_seed(uint64_t seed) {
    return seed ? seed : 0x9e3779b97f4a7c15ULL;
}

static void h3_noise(float *dst, int64_t n, uint64_t seed) {
    uint64_t s = h3_noise_seed(seed);
    h3_noise_at(dst, n, &s);
}

static const double H3_SIGMAS[H3_STEPS + 1] = {
    1.0, 0.9675236940383911, 0.8875485062599182,
    0.75, 0.5327739119529724, 0.23200887441635132, 0.0
};

/* The canvas a request is generated on. Below `area` pixels the canvas is
 * scaled up by `sqrt(area / (w * h))` at the same aspect ratio, then both sides
 * round to the nearest canvas multiple. Measured: at 256x256 the checkpoint
 * returns textured noise and the reference implementation returns the same
 * textured noise, which is what the released pipeline's 4 MP floor exists to
 * avoid. */
static int32_t h3_min_area(void) {
    const char *v = getenv("CWICK_H3_MIN_AREA");
    if (v && *v) {
        long n = strtol(v, NULL, 10);
        if (n >= 1024 && n <= 16777216) return (int32_t)n;
    }
    return H3_MIN_AREA;
}

static void h3_canvas_size(int want_w, int want_h, int *cw, int *ch) {
    double area = (double)want_w * (double)want_h;
    double scale = area > 0.0 ? sqrt((double)h3_min_area() / area) : 1.0;
    if (!(scale > 1.0)) scale = 1.0;
    int w = (int)(lround((double)want_w * scale / H3_CANVAS_MULTIPLE) * H3_CANVAS_MULTIPLE);
    int h = (int)(lround((double)want_h * scale / H3_CANVAS_MULTIPLE) * H3_CANVAS_MULTIPLE);
    if (w < H3_CANVAS_MULTIPLE) w = H3_CANVAS_MULTIPLE;
    if (h < H3_CANVAS_MULTIPLE) h = H3_CANVAS_MULTIPLE;
    *cw = w;
    *ch = h;
}

/* Separable Lanczos-3 resize. The canvas a request is generated on is much
 * larger than the requested size, so the filter matters: nearest neighbour
 * turns a 2048x2048 canvas into a visibly aliased 512x512. */
static double h3_lanczos(double x) {
    if (x < 0.0) x = -x;
    if (x < 1e-9) return 1.0;
    if (x >= 3.0) return 0.0;
    const double pi = 3.141592653589793;
    return (3.0 * sin(pi * x) * sin(pi * x / 3.0)) / (pi * pi * x * x);
}

static void h3_resize_lanczos(const float *src, int sw, int sh,
                              float *dst, int dw, int dh) {
    const int TAPS = 3;
    double scale_x = (double)sw / (double)dw;
    double scale_y = (double)sh / (double)dh;
    double fx = scale_x < 1.0 ? 1.0 : scale_x;
    double fy = scale_y < 1.0 ? 1.0 : scale_y;
    int tx = (int)ceil(fx * TAPS), ty = (int)ceil(fy * TAPS);

    float *rows = h3_f32((int64_t)sh * dw * 3);
    for (int y = 0; y < sh; y++) {
        for (int x = 0; x < dw; x++) {
            double centre = ((double)x + 0.5) * scale_x - 0.5;
            int lo = (int)floor(centre) - tx + 1;
            int hi = (int)floor(centre) + tx;
            double acc[3] = {0.0, 0.0, 0.0}, sum = 0.0;
            for (int i = lo; i <= hi; i++) {
                int si = i < 0 ? 0 : (i >= sw ? sw - 1 : i);
                double wgt = h3_lanczos(((double)i - centre) / fx);
                if (wgt == 0.0) continue;
                const float *p = src + ((int64_t)y * sw + si) * 3;
                acc[0] += wgt * p[0]; acc[1] += wgt * p[1]; acc[2] += wgt * p[2];
                sum += wgt;
            }
            float *o = rows + ((int64_t)y * dw + x) * 3;
            if (sum != 0.0) {
                o[0] = (float)(acc[0] / sum);
                o[1] = (float)(acc[1] / sum);
                o[2] = (float)(acc[2] / sum);
            } else {
                o[0] = o[1] = o[2] = 0.0f;
            }
        }
    }

    for (int y = 0; y < dh; y++) {
        double centre = ((double)y + 0.5) * scale_y - 0.5;
        int lo = (int)floor(centre) - ty + 1;
        int hi = (int)floor(centre) + ty;
        for (int x = 0; x < dw; x++) {
            double acc[3] = {0.0, 0.0, 0.0}, sum = 0.0;
            for (int i = lo; i <= hi; i++) {
                int si = i < 0 ? 0 : (i >= sh ? sh - 1 : i);
                double wgt = h3_lanczos(((double)i - centre) / fy);
                if (wgt == 0.0) continue;
                const float *p = rows + ((int64_t)si * dw + x) * 3;
                acc[0] += wgt * p[0]; acc[1] += wgt * p[1]; acc[2] += wgt * p[2];
                sum += wgt;
            }
            float *o = dst + ((int64_t)y * dw + x) * 3;
            if (sum != 0.0) {
                o[0] = (float)(acc[0] / sum);
                o[1] = (float)(acc[1] / sum);
                o[2] = (float)(acc[2] / sum);
            } else {
                o[0] = o[1] = o[2] = 0.0f;
            }
        }
    }
    free(rows);
}

/* `bound_short_edge`: a reference keeps its own size when its short edge is
 * already within [1024, 2048]; otherwise it is scaled, keeping its ratio, so the
 * short edge lands on the nearer bound and both sides round to 32. A reference
 * is never resized to the size of the generated image. */
static void h3_bound_edge(int w, int h, int lo, int hi, int *ow, int *oh) {
    const int m = H3_CANVAS_MULTIPLE;
    const int se = w < h ? w : h;
    if (se >= lo && se <= hi) {
        /* Inside its bounds a reference keeps its own size, which the packed
         * layout cannot take unless both sides divide by 32 twice. The
         * published pipeline raises there instead; snapping a side by at most
         * 16 pixels is the difference between a usable request and a hard
         * failure on a camera image. */
        *ow = (int)lround((double)w / m) * m;
        *oh = (int)lround((double)h / m) * m;
        if (*ow < m) *ow = m;
        if (*oh < m) *oh = m;
        return;
    }
    const double s = (double)(se < lo ? lo : hi) / (double)se;
    int nw = (int)lround((double)w * s / m) * m;
    int nh = (int)lround((double)h * s / m) * m;
    if (nw < m) nw = m;
    if (nh < m) nh = m;
    *ow = nw;
    *oh = nh;
}

/* `bound_short_edge`: a reference keeps its own size when its short edge is
 * already within [1024, 2048]; otherwise it is scaled, keeping its ratio, so the
 * short edge lands on the nearer bound and both sides round to 32. A reference
 * is never resized to the size of the generated image. */
static void h3_ref_size(int w, int h, int *ow, int *oh) {
    h3_bound_edge(w, h, 1024, 2048, ow, oh);
}

/* The tower reads its own copy, whose short edge is pinned to exactly 1024 and
 * which is resized from the original rather than from the VAE copy. */
static void h3_vision_size(int w, int h, int *ow, int *oh) {
    h3_bound_edge(w, h, 1024, 1024, ow, oh);
}

/* Decodes one reference and resizes it to `tw` x `th`; callers pick the target
 * with whichever of the two resize rules their consumer needs. */

/* ------------------------------------------------------------------ */
/* Edit presentation                                                   */
/*                                                                     */
/* [Picture i: <|vision_start|> <|image_pad|> x (gh*gw/4)              */
/*  <|vision_end|>] for every reference in order, then the prompt       */
/* verbatim. The label carries no angle brackets here, unlike the       */
/* `<Picture i>` the prompt refers to it by. Two tag vectors come out   */
/* of this: `types` is the conditioner's own (`1` on the pads only, so  */
/* the delimiters join the surrounding text run and the 3-axis rotary   */
/* pieces one run per block) and `tags` is the transformer's (`0` over  */
/* the whole block, delimiters included).                              */
/* ------------------------------------------------------------------ */
typedef struct {
    uint32_t *ids;
    int32_t *types;
    float *tags;
    float *vis;
    float *ds[VIS_DS];
    int64_t *grids;
    int64_t n, nvis, nblocks;
} h3_present;

static void h3_present_free(h3_present *p) {
    free(p->ids); free(p->types); free(p->tags); free(p->vis);
    free(p->ds[0]); free(p->ds[1]); free(p->ds[2]); free(p->grids);
    memset(p, 0, sizeof(*p));
}

static bool h3_present_push(h3_present *p, int64_t *cap, uint32_t id, int32_t type,
                            float tag) {
    if (p->n == *cap) {
        int64_t nc = *cap ? *cap * 2 : 256;
        uint32_t *ni = cwick_xmalloc((size_t)nc * sizeof(uint32_t));
        int32_t *nt = cwick_xmalloc((size_t)nc * sizeof(int32_t));
        float *ng = cwick_xmalloc((size_t)nc * sizeof(float));
        if (p->n) {
            memcpy(ni, p->ids, (size_t)p->n * sizeof(uint32_t));
            memcpy(nt, p->types, (size_t)p->n * sizeof(int32_t));
            memcpy(ng, p->tags, (size_t)p->n * sizeof(float));
        }
        free(p->ids); free(p->types); free(p->tags);
        p->ids = ni; p->types = nt; p->tags = ng;
        *cap = nc;
    }
    p->ids[p->n] = id;
    p->types[p->n] = type;
    p->tags[p->n] = tag;
    p->n++;
    return true;
}

/* Encodes `text` and appends it with a fixed tag and type. */
static bool h3_present_text(h3_model *m, h3_present *p, int64_t *cap, const char *text,
                            int32_t type, float tag, cwick_error *err) {
    uint32_t *ids = NULL;
    int64_t n = 0;
    if (cwick_tokenizer_encode(m->tok, text, &ids, &n, err) != CWICK_OK) return false;
    for (int64_t i = 0; i < n; i++) h3_present_push(p, cap, ids[i], type, tag);
    free(ids);
    return true;
}

/* Builds the whole presentation and runs the tower on every reference, so the
 * `vis` and `ds` planes come back already stacked in pad order. */
static bool h3_present_build(h3_model *m, const char *prompt,
                            const float *const *ref, const int *rw, const int *rh,
                            int n_ref, h3_present *p, cwick_error *err) {
    int64_t cap = 0;
    memset(p, 0, sizeof(*p));
    for (int r = 0; r < n_ref; r++) {
        h3_vis_out vo;
        memset(&vo, 0, sizeof(vo));
        if (!h3_vis_run(m, ref[r], rw[r], rh[r], &vo, err)) {
            h3_present_free(p);
            return false;
        }
        const int64_t rows = (int64_t)vo.gh * vo.gw / (VIS_MERGE * VIS_MERGE);
        const int64_t have = p->nvis;
        p->vis = cwick_xrealloc(p->vis, (size_t)(have + rows) * TE_HIDDEN * sizeof(float));
        memcpy(p->vis + have * TE_HIDDEN, vo.merged,
               (size_t)(rows * TE_HIDDEN) * sizeof(float));
        for (int64_t d = 0; d < VIS_DS; d++) {
            p->ds[d] = cwick_xrealloc(p->ds[d],
                                      (size_t)(have + rows) * TE_HIDDEN * sizeof(float));
            memcpy(p->ds[d] + have * TE_HIDDEN, vo.ds[d],
                   (size_t)(rows * TE_HIDDEN) * sizeof(float));
        }
        p->grids = cwick_xrealloc(p->grids, (size_t)(p->nblocks + 1) * 2 * sizeof(int64_t));
        p->grids[2 * p->nblocks] = vo.gh;
        p->grids[2 * p->nblocks + 1] = vo.gw;
        p->nblocks++;
        p->nvis += rows;
        free(vo.merged);
        for (int64_t d = 0; d < VIS_DS; d++) free(vo.ds[d]);

        char label[48];
        snprintf(label, sizeof(label), "Picture %d: <|vision_start|>", r + 1);
        /* The label and the `<|vision_start|>` are text to the conditioner but
         * already part of the vision block to the transformer. */
        if (!h3_present_text(m, p, &cap, label, 0, 1.0f, err)) {
            h3_present_free(p);
            return false;
        }
        p->tags[p->n - 1] = 0.0f;

        if (!h3_present_text(m, p, &cap, "<|image_pad|>", 1, 0.0f, err)) {
            h3_present_free(p);
            return false;
        }
        const uint32_t pad = p->ids[p->n - 1];
        p->n--;
        for (int64_t i = 0; i < rows; i++) h3_present_push(p, &cap, pad, 1, 0.0f);
        if (!h3_present_text(m, p, &cap, "<|vision_end|>", 0, 0.0f, err)) {
            h3_present_free(p);
            return false;
        }
    }
    if (!h3_present_text(m, p, &cap, prompt, 0, 1.0f, err)) {
        h3_present_free(p);
        return false;
    }
    return true;
}

bool h3_generate(h3_model *m, const char *prompt, int w, int h, uint64_t seed,
                 const unsigned char **ref_png, const size_t *ref_len, int n_ref,
                 unsigned char **out, size_t *out_len, h3_timing *tim,
                 cwick_error *err) {
    if (!h3_model_ready(m)) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "image model not ready");
        return false;
    }
    /* Vision-tower-only run: one PNG through the tower, so `patch_embed`,
     * `x0`, `blk0`, `last_hidden`, `merged` and the three deepstack taps can be
     * differenced against a reference pass without a generation. */
    {
        const char *ep = getenv("CWICK_H3_VIS_PNG");
        if (ep && *ep) {
            float *rgb = NULL;
            int rw = 0, rh = 0;
            FILE *f = fopen(ep, "rb");
            unsigned char *buf = NULL;
            size_t blen = 0;
            if (f) {
                fseek(f, 0, SEEK_END);
                blen = (size_t)ftell(f);
                fseek(f, 0, SEEK_SET);
                buf = cwick_xmalloc(blen);
                if (fread(buf, 1, blen, f) != blen) blen = 0;
                fclose(f);
            }
            if (!blen || !h3_png_decode(buf, blen, &rgb, &rw, &rh, err)) {
                free(buf); free(rgb);
                cwick_error_set(err, CWICK_ERR_ARGUMENT, "cannot read %s", ep);
                return false;
            }
            free(buf);
            h3_vis_out vo;
            memset(&vo, 0, sizeof(vo));
            bool ok = h3_vis_run(m, rgb, rw, rh, &vo, err);
            free(rgb);
            free(vo.merged);
            for (int64_t d = 0; d < VIS_DS; d++) free(vo.ds[d]);
            if (!ok) return false;
            float *blank = h3_f32(3);
            h3_zero(blank, 3);
            *out = h3_png_encode(blank, 1, 1, out_len, err);
            free(blank);
            return *out != NULL;
        }
    }
    /* Encoder-only run: one PNG through the reference path, which can be
     * differenced against a reference implementation without a generation. */
    {
        const char *ep = getenv("CWICK_H3_ENC_PNG");
        if (ep && *ep) {
            float *rgb = NULL;
            int rw = 0, rh = 0;
            FILE *f = fopen(ep, "rb");
            unsigned char *buf = NULL;
            size_t blen = 0;
            if (f) {
                fseek(f, 0, SEEK_END);
                blen = (size_t)ftell(f);
                fseek(f, 0, SEEK_SET);
                buf = cwick_xmalloc(blen);
                if (fread(buf, 1, blen, f) != blen) blen = 0;
                fclose(f);
            }
            if (!blen || !h3_png_decode(buf, blen, &rgb, &rw, &rh, err)) {
                free(buf); free(rgb);
                cwick_error_set(err, CWICK_ERR_ARGUMENT, "cannot read %s", ep);
                return false;
            }
            free(buf);
            int tw = 0, th = 0;
            h3_ref_size(rw, rh, &tw, &th);
            float *fit = rgb;
            if (tw != rw || th != rh) {
                fit = h3_f32((int64_t)tw * th * 3);
                h3_resize_lanczos(rgb, rw, rh, fit, tw, th);
                free(rgb);
            }
            const int64_t lw = tw / H3_VAE_RATIO, lh = th / H3_VAE_RATIO;
            float *noise = h3_f32((int64_t)VAE_LATENTS * lw * lh);
            const char *nf = getenv("CWICK_H3_ENC_NOISE");
            bool got = false;
            if (nf && *nf) {
                FILE *g = fopen(nf, "rb");
                if (g) {
                    got = fread(noise, sizeof(float),
                                (size_t)(VAE_LATENTS * lw * lh), g) ==
                          (size_t)(VAE_LATENTS * lw * lh);
                    fclose(g);
                }
            }
            if (!got) h3_noise(noise, (int64_t)VAE_LATENTS * lw * lh, seed);
            float *lat = NULL;
            bool ok = h3_vae_encode(m, fit, tw, th, noise, &lat, err);
            free(fit);
            free(noise);
            if (!ok) return false;
            float *blank = h3_f32(3);
            h3_zero(blank, 3);
            *out = h3_png_encode(blank, 1, 1, out_len, err);
            free(blank);
            free(lat);
            return *out != NULL;
        }
    }
    if (!prompt || !*prompt) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "prompt is required");
        return false;
    }
    if (w < H3_MIN_SIZE || h < H3_MIN_SIZE || w > H3_MAX_SIZE || h > H3_MAX_SIZE) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "size out of range");
        return false;
    }
    if (n_ref < 0 || n_ref > H3_MAX_REFERENCES || (n_ref > 0 && (!ref_png || !ref_len))) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "too many reference images");
        return false;
    }

    /* Without an HTTP layer there is no way to hand a reference to the CLI, so
     * a single file can stand in for one. Purely a debugging entry. */
    unsigned char *dbg_png = NULL;
    const unsigned char *dbg_ref[1];
    size_t dbg_ref_len[1];
    {
        const char *rp = getenv("CWICK_H3_EDIT_PNG");
        if (n_ref == 0 && rp && *rp) {
            FILE *f = fopen(rp, "rb");
            if (!f) {
                cwick_error_set(err, CWICK_ERR_ARGUMENT, "cannot read %s", rp);
                return false;
            }
            fseek(f, 0, SEEK_END);
            dbg_ref_len[0] = (size_t)ftell(f);
            fseek(f, 0, SEEK_SET);
            dbg_png = cwick_xmalloc(dbg_ref_len[0]);
            if (fread(dbg_png, 1, dbg_ref_len[0], f) != dbg_ref_len[0]) {
                fclose(f); free(dbg_png);
                cwick_error_set(err, CWICK_ERR_ARGUMENT, "cannot read %s", rp);
                return false;
            }
            fclose(f);
            dbg_ref[0] = dbg_png;
            ref_png = dbg_ref;
            ref_len = dbg_ref_len;
            n_ref = 1;
        }
    }

    h3_timing t;
    memset(&t, 0, sizeof(t));
    if (tim) *tim = t;
    double t0 = h3_ms();

    pthread_mutex_lock(&m->lock);

    uint32_t *ids = NULL;
    int64_t nids = 0;
    h3_present pres;
    memset(&pres, 0, sizeof(pres));
    float **ref_vae = NULL, **ref_vis = NULL;
    int *ref_vw = NULL, *ref_vh = NULL, *ref_aw = NULL, *ref_ah = NULL;
    double a = h3_ms();
    if (n_ref > 0) {
        ref_vae = cwick_xmalloc((size_t)n_ref * sizeof(float *));
        ref_vis = cwick_xmalloc((size_t)n_ref * sizeof(float *));
        ref_vw = cwick_xmalloc((size_t)n_ref * sizeof(int));
        ref_vh = cwick_xmalloc((size_t)n_ref * sizeof(int));
        ref_aw = cwick_xmalloc((size_t)n_ref * sizeof(int));
        ref_ah = cwick_xmalloc((size_t)n_ref * sizeof(int));
        memset(ref_vae, 0, (size_t)n_ref * sizeof(float *));
        memset(ref_vis, 0, (size_t)n_ref * sizeof(float *));
        for (int r = 0; r < n_ref; r++) {
            float *native = NULL;
            int nw = 0, nh = 0;
            if (!h3_png_decode(ref_png[r], ref_len[r], &native, &nw, &nh, err)) {
                cwick_error_set(err, CWICK_ERR_ARGUMENT, "reference %d is not a PNG", r + 1);
                goto unlock_fail;
            }
            /* Both consumers resize from the original, never from each other,
             * so each gets its own buffer even when the size is unchanged. */
            int tw = 0, th = 0;
            h3_ref_size(nw, nh, &tw, &th);
            ref_aw[r] = tw;
            ref_ah[r] = th;
            ref_vae[r] = h3_f32((int64_t)tw * th * 3);
            if (tw == nw && th == nh)
                memcpy(ref_vae[r], native, (size_t)tw * th * 3 * sizeof(float));
            else
                h3_resize_lanczos(native, nw, nh, ref_vae[r], tw, th);
            h3_vision_size(nw, nh, &tw, &th);
            ref_vw[r] = tw;
            ref_vh[r] = th;
            ref_vis[r] = h3_f32((int64_t)tw * th * 3);
            if (tw == nw && th == nh)
                memcpy(ref_vis[r], native, (size_t)tw * th * 3 * sizeof(float));
            else
                h3_resize_lanczos(native, nw, nh, ref_vis[r], tw, th);
            free(native);
        }
        if (!h3_present_build(m, prompt, (const float *const *)ref_vis, ref_vw, ref_vh,
                              n_ref, &pres, err))
            goto unlock_fail;
        ids = pres.ids;
        nids = pres.n;
    } else if (cwick_tokenizer_encode(m->tok, prompt, &ids, &nids, err) != CWICK_OK) {
        goto unlock_fail;
    }
    t.tokenize_ms = h3_ms() - a;
    t.text_tokens = nids;
    CWICK_INFO("image", "phase 1/6 tokenize: %lld token%s in %.1f ms",
               (long long)nids, nids == 1 ? "" : "s", t.tokenize_ms);
    if (n_ref > 0) {
        int64_t *lay = cwick_xmalloc((size_t)nids * sizeof(int64_t));
        for (int64_t i = 0; i < nids; i++) lay[i] = pres.ids[i];
        h3_dump_i64("c_te_ids", lay, nids);
        for (int64_t i = 0; i < nids; i++) lay[i] = pres.types[i];
        h3_dump_i64("c_te_types", lay, nids);
        for (int64_t i = 0; i < nids; i++) lay[i] = (int64_t)pres.tags[i];
        h3_dump_i64("c_te_tags", lay, nids);
        free(lay);
    }

    a = h3_ms();
    h3_text txt;
    memset(&txt, 0, sizeof(txt));
    {
        h3_cond cond;
        memset(&cond, 0, sizeof(cond));
        cond.types = pres.types;
        cond.tags = pres.tags;
        cond.vis = pres.vis;
        cond.ds[0] = pres.ds[0];
        cond.ds[1] = pres.ds[1];
        cond.ds[2] = pres.ds[2];
        cond.grids = pres.grids;
        cond.nvis = pres.nvis;
        cond.nblocks = pres.nblocks;
        if (!h3_text_encode(m, ids, nids, n_ref > 0 ? &cond : NULL, &txt, err)) {
            h3_present_free(&pres);
            goto unlock_fail;
        }
    }
    t.text_encode_ms = h3_ms() - a;
    {
        int64_t nv = pres.nvis;   /* read before h3_present_free below */
        CWICK_INFO("image", "phase 2/6 text-encode: %.1f ms, %lld vision token(s)",
                   t.text_encode_ms, (long long)nv);
    }
    /* The presentation owns the id array it produced. */
    if (n_ref > 0) { h3_present_free(&pres); ids = NULL; }
    free(ids);

    /* Text-only run: the conditioner can be diffed against a reference without
     * paying for the six transformer steps and the decoder. */
    {
        const char *tonly = getenv("CWICK_H3_TEXT_ONLY");
        if (tonly && *tonly) {
            float *blank = h3_f32(3);
            h3_zero(blank, 3);
            *out = h3_png_encode(blank, 1, 1, out_len, err);
            free(blank);
            h3_text_free(&txt);
            t.total_ms = h3_ms() - t0;
            if (tim) *tim = t;
            free(dbg_png);
            pthread_mutex_unlock(&m->lock);
            return *out != NULL;
        }
    }

    /* A reference conditioner's output can be fed in verbatim, which separates
     * an error in this stage from one in the transformer. */
    {
        const char *ef = getenv("CWICK_H3_EMBEDS");
        if (ef && *ef) {
            FILE *f = fopen(ef, "rb");
            size_t want = (size_t)(nids * TE_HIDDEN);
            bool ok = false;
            if (f) {
                ok = fread(txt.hs50, sizeof(float), want, f) == want;
                fclose(f);
            }
            if (!ok) {
                h3_text_free(&txt);
                pthread_mutex_unlock(&m->lock);
                cwick_error_set(err, CWICK_ERR_ARGUMENT,
                                "CWICK_H3_EMBEDS does not hold %lld floats",
                                (long long)want);
                return false;
            }
        }
    }

    /* Canvas: the checkpoint is distilled on canvases of about 4 to 20 MP and
     * degrades into texture when asked for a small one, so a request below
     * `H3_MIN_AREA` is generated at the same aspect ratio on a canvas of that
     * area and resized back after decoding. Both sides round to the nearest 32
     * so the result survives the 16x VAE ratio and the 2x patch. */
    int cw, ch;
    h3_canvas_size(w, h, &cw, &ch);
    int lw = cw / H3_VAE_RATIO, lh = ch / H3_VAE_RATIO;
    int ph = lh / H3_PATCH, pw = lw / H3_PATCH;
    int64_t nrows = (int64_t)ph * pw;

    a = h3_ms();
    h3_latent_rows img;
    img.rows = nrows;
    img.img = h3_f32(nrows * DIT_PATCH_DIM);
    /* One generator serves the whole request: the generated rows draw first,
     * then one reference block each, in packed order. */
    uint64_t rng = h3_noise_seed(seed);
    {
        /* A reference implementation's initial noise can be fed in verbatim,
         * which is what makes a bit-level diff of the forward pass possible. */
        const char *nf = getenv("CWICK_H3_NOISE");
        bool loaded = false;
        if (nf && *nf) {
            FILE *f = fopen(nf, "rb");
            if (f) {
                size_t got = fread(img.img, sizeof(float), (size_t)(nrows * DIT_PATCH_DIM), f);
                fclose(f);
                loaded = got == (size_t)(nrows * DIT_PATCH_DIM);
            }
            if (!loaded) {
                free(img.img); h3_text_free(&txt);
                pthread_mutex_unlock(&m->lock);
                cwick_error_set(err, CWICK_ERR_ARGUMENT,
                                "CWICK_H3_NOISE does not hold %lld floats",
                                (long long)(nrows * DIT_PATCH_DIM));
                return false;
            }
        }
        if (!loaded) h3_noise_at(img.img, nrows * DIT_PATCH_DIM, &rng);
    }
    h3_dump("c_noise", img.img, nrows * DIT_PATCH_DIM);

    /* Reference rows ride in front of the generated ones, noised to the
     * conditioning level and held there for the whole sampling loop. */
    h3_refgeo rgeo[H3_MAX_REFERENCES];
    int nrefblk = 0;
    int64_t nrefrows = 0;
    if (n_ref > 0) {
        float *rows = NULL;
        float *refnoise = NULL;
        for (int r = 0; r < n_ref; r++) {
            const int rlw = ref_aw[r] / H3_VAE_RATIO, rlh = ref_ah[r] / H3_VAE_RATIO;
            const int64_t rn = (int64_t)(rlw / H3_PATCH) * (rlh / H3_PATCH);
            float *noise = h3_f32((int64_t)VAE_LATENTS * rlw * rlh);
            h3_noise_at(noise, (int64_t)VAE_LATENTS * rlw * rlh, &rng);
            float *lat = NULL;
            /* A reference latent can be fed in verbatim so the transformer can
             * be differenced without paying for the encoder. */
            const char *rl = getenv("CWICK_H3_REF_LATENT");
            bool injected = false;
            if (rl && *rl && r == 0) {
                FILE *f = fopen(rl, "rb");
                if (f) {
                    lat = h3_f32((int64_t)VAE_LATENTS * rlw * rlh);
                    injected = fread(lat, sizeof(float),
                                     (size_t)(VAE_LATENTS * rlw * rlh), f) ==
                               (size_t)(VAE_LATENTS * rlw * rlh);
                    fclose(f);
                }
            }
            if (!injected) {
                free(lat);
                if (!h3_vae_encode(m, ref_vae[r], ref_aw[r], ref_ah[r], noise, &lat,
                                   err)) {
                    free(noise); free(rows); free(refnoise);
                    free(img.img); h3_text_free(&txt);
                    goto unlock_fail;
                }
            }
            refnoise = cwick_xrealloc(refnoise,
                                      (size_t)(nrefrows + rn) * DIT_PATCH_DIM * sizeof(float));
            h3_pack_rows(refnoise + nrefrows * DIT_PATCH_DIM, noise, rlw, rlh);
            /* `scale_noise` with a tensor timestep mixes in float32, which is
             * where the checkpoint's conditioning level lives. */
            const float t0 = H3_REF_NOISE_AUG;
            for (int64_t i = 0; i < (int64_t)VAE_LATENTS * rlw * rlh; i++)
                lat[i] = t0 * lat[i] + (1.0f - t0) * noise[i];
            free(noise);
            rows = cwick_xrealloc(rows, (size_t)(nrefrows + rn) * DIT_PATCH_DIM * sizeof(float));
            h3_pack_rows(rows + nrefrows * DIT_PATCH_DIM, lat, rlw, rlh);
            free(lat);
            rgeo[nrefblk].lw = rlw;
            rgeo[nrefblk].lh = rlh;
            rgeo[nrefblk].rows = rn;
            nrefblk++;
            nrefrows += rn;
        }
        h3_dump("c_ref_rows", rows, nrefrows * DIT_PATCH_DIM);
        h3_dump("c_ref_noise", refnoise, nrefrows * DIT_PATCH_DIM);
        free(refnoise);
        float *all = h3_f32((nrefrows + nrows) * DIT_PATCH_DIM);
        memcpy(all, rows, (size_t)nrefrows * DIT_PATCH_DIM * sizeof(float));
        memcpy(all + nrefrows * DIT_PATCH_DIM, img.img,
               (size_t)nrows * DIT_PATCH_DIM * sizeof(float));
        free(rows);
        free(img.img);
        img.img = all;
        img.rows = nrefrows + nrows;
    }
    t.pack_ms = h3_ms() - a;
    t.image_tokens = nrows;
    t.canvas_w = cw; t.canvas_h = ch;
    t.output_w = w;  t.output_h = h;
    CWICK_INFO("image", "phase 3/6 pack: %lld image rows on %lldx%lld canvas "
               "(%d reference block%s), %.1f ms",
               (long long)nrows, (long long)cw, (long long)ch, nrefblk,
               nrefblk == 1 ? "" : "s", t.pack_ms);

    a = h3_ms();
    /* The sampler stores `t = float32(1 - sigma)` and recovers the sigma it
     * steps on as `1 - t`, so below 0.5 a grid value can move by an ulp. Keep
     * the round trip rather than the published sigma. */
    float ts[H3_STEPS], sg[H3_STEPS + 1];
    for (int s = 0; s < H3_STEPS; s++) ts[s] = (float)(1.0 - H3_SIGMAS[s]);
    for (int s = 0; s < H3_STEPS; s++) sg[s] = 1.0f - ts[s];
    sg[H3_STEPS] = 0.0f;
    /* A finished latent can be fed in to exercise the decoder on its own. */
    const char *lf = getenv("CWICK_H3_LATENT");
    bool lat_given = false;
    if (lf && *lf) {
        FILE *f = fopen(lf, "rb");
        if (f) {
            size_t want = (size_t)(nrows * DIT_PATCH_DIM);
            lat_given = fread(img.img + nrefrows * DIT_PATCH_DIM, sizeof(float), want, f) == want;
            fclose(f);
        }
    }
    CWICK_INFO("image", "phase 4/6 dit: %d steps on %lld packed rows "
               "(seq %lld), canvas %dx%d",
               lat_given ? 0 : H3_STEPS,
               (long long)img.rows, (long long)(img.rows + txt.seq),
               cw, ch);
    for (int s = 0; s < H3_STEPS && !lat_given; s++) {
        double step_ms = h3_ms();
        float *vel = NULL;
        if (!h3_dit_step(m, &txt, &img, rgeo, nrefblk, nrefrows, lw, lh, s, ts[s],
                         &vel, err)) {
            free(img.img); h3_text_free(&txt);
            pthread_mutex_unlock(&m->lock);
            return false;
        }
        /* Euler in velocity form: x <- x + (sigma - sigma_next) * v. Only the
         * generated rows move; the reference rows are conditioning, not state. */
        float dt = sg[s] - sg[s + 1];
        for (int64_t i = nrefrows * DIT_PATCH_DIM; i < img.rows * DIT_PATCH_DIM; i++)
            img.img[i] += dt * vel[i];
        free(vel);
        CWICK_INFO("image", "  dit step %d/%d: %.1f ms (elapsed %.1f ms)",
                   s + 1, H3_STEPS, h3_ms() - step_ms, h3_ms() - a);
    }
    t.dit_ms = h3_ms() - a;
    CWICK_INFO("image", "phase 4/6 dit done: %.1f ms total", t.dit_ms);
    h3_dump("c_latents_rows", img.img + nrefrows * DIT_PATCH_DIM, nrows * DIT_PATCH_DIM);

    /* Unpatchify [ph, pw, C, 2, 2] -> [C, lh, lw]. */
    float *latent = h3_f32(nrows * DIT_PATCH_DIM);
    for (int64_t r = 0; r < nrows; r++) {
        int64_t py = r / pw, px = r % pw;
        for (int64_t c = 0; c < H3_LATENT_CHANNELS; c++) {
            const float *src = img.img + (nrefrows + r) * DIT_PATCH_DIM + c * 4;
            for (int dy = 0; dy < H3_PATCH; dy++)
                for (int dx = 0; dx < H3_PATCH; dx++)
                    latent[(c * lh + py * H3_PATCH + dy) * lw + px * H3_PATCH + dx] = src[dy * H3_PATCH + dx];
        }
    }
    free(img.img);
    h3_text_free(&txt);
    h3_dump("c_latent", latent, nrows * DIT_PATCH_DIM);

    a = h3_ms();
    float *rgb = NULL;
    if (!h3_vae_decode(m, latent, lw, lh, &rgb, err)) {
        free(latent);
        pthread_mutex_unlock(&m->lock);
        return false;
    }
    t.vae_ms = h3_ms() - a;
    CWICK_INFO("image", "phase 5/6 vae: %lldx%lld decoded in %.1f ms",
               (long long)cw, (long long)ch, t.vae_ms);
    h3_dump("c_rgb", rgb, (int64_t)cw * ch * 3);
    free(latent);

    /* Resize back to the requested size. The canvas is at least 4 MP, so a
     * 512x512 request is a 4x reduction and the filter is visible. */
    int ow = w, oh = h;
    float *fit = rgb;
    if (ow != cw || oh != ch) {
        fit = h3_f32((int64_t)ow * oh * 3);
        h3_resize_lanczos(rgb, cw, ch, fit, ow, oh);
        free(rgb);
    }

    a = h3_ms();
    *out = h3_png_encode(fit, ow, oh, out_len, err);
    t.png_ms = h3_ms() - a;
    CWICK_INFO("image", "phase 6/6 png: %dx%d output in %.1f ms",
               ow, oh, t.png_ms);
    free(fit);
    for (int r = 0; r < n_ref; r++) { free(ref_vae[r]); free(ref_vis[r]); }
    free(ref_vae); free(ref_vis); free(ref_vw); free(ref_vh);
    free(ref_aw); free(ref_ah);
    free(dbg_png);
    pthread_mutex_unlock(&m->lock);
    if (!*out) return false;

    t.total_ms = h3_ms() - t0;
    if (tim) *tim = t;
    return true;

unlock_fail:
    for (int r = 0; r < n_ref; r++) { free(ref_vae[r]); free(ref_vis[r]); }
    free(ref_vae); free(ref_vis); free(ref_vw); free(ref_vh);
    free(ref_aw); free(ref_ah);
    free(dbg_png);
    pthread_mutex_unlock(&m->lock);
    return false;
}
