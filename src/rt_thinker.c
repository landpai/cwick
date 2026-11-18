/* rt_thinker.c — the 64-layer decoder of the realtime model, forward pass only.
 *
 * Layout, all measured from the checkpoint rather than assumed:
 *   q_proj [8192, 5120]  k/v_proj [1024, 5120]  o_proj [5120, 8192]
 *   gate/up [25600, 5120]  down [5120, 25600]   lm_head [151936, 5120] bf16
 * Every projection is FP8 E4M3 with one BF16 scale per output row, so the
 * arithmetic is `y[n] = scale[n] * dot(w[n], x)` -- the scale comes out of the
 * dot product and the kernels need no block bookkeeping.
 *
 * Weights stay in the mapped shards. Nothing is dequantized ahead of time:
 * a 32 GiB decoder would need 64 GiB in BF16 and the machine has to keep the
 * speech stages resident as well.
 *
 * Two math paths, same numbers: the GPU kernels in metal_objc.m when Metal is
 * up, and Accelerate (dequantize a row block, then sgemm) when it is not. The
 * CPU path exists so tests and CI do not depend on a GPU. */
#include "rt_thinker.h"

#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds.h" /* DS_E4M3_LUT / ds_e4m3_to_f32: the same FP8 decode the rest of cwick uses */
#include "json.h"
#include "log.h"
#include "util.h"

#define RT_TAG "thinker"
#ifndef RT_PREFILL_CHUNK
#define RT_PREFILL_CHUNK 64
#endif

int cwick_metal_init(void);
int cwick_metal_gemv_fp8_row(const void *w, const float *scale, const float *x,
                             float *out, uint32_t k, uint32_t n);
int cwick_metal_gemm_fp8_row(const void *w, const float *scale, const float *x,
                             float *out, uint32_t k, uint32_t n, uint32_t m);
int cwick_metal_gemv_bf16(const void *w_bf16, const float *x, float *out,
                          uint32_t k, uint32_t cols);

/* ------------------------------------------------------------ weights --- */

typedef struct {
    const uint8_t *w8;  /* FP8 E4M3 [out][in] */
    const float *sc;    /* per-output-row scale, f32 [out] */
    const uint16_t *w16;/* BF16 [out][in] when the projection is not quantized */
    int64_t out, in;
} rt_lin;

typedef struct {
    rt_lin q, k, v, o, gate, up, down;
    const float *in_ln, *post_ln, *q_norm, *k_norm;
} rt_layer;

struct rt_thinker {
    cwick_safetensors saf;
    cwick_tokenizer *tok;
    int H, NQ, NKV, HD, INTER, NL, VOCAB, max_seq, pos;
    float eps, theta;
    rt_layer *lay;
    const uint16_t *embed;
    const uint16_t *lmhead;
    const float *final_norm; /* model.norm.weight, applied before lm_head */
    float *rc, *rs;      /* rope cos/sin [max_seq][HD/2] */
    float *kc, *vc;      /* [NL][max_seq][NKV*HD] */
    float *logits;

    /* activation scratch, sized for max_prefill rows */
    int cap;
    float *x, *xn, *hq, *hk, *hv, *att, *ao, *ga, *gu, *ac, *tmp;
    float *gkv, *gq, *scores; /* attention gathers and the score matrix */

    int gpu;             /* Metal kernels are usable */
    int head_group;      /* NQ / NKV */
    /* staging for the CPU dequant path */
    float *dw;
    int64_t dw_rows;
};

static void *xmalloc0(size_t n) {
    void *p = cwick_xmalloc(n);
    memset(p, 0, n);
    return p;
}

static const cwick_st_tensor *fetch(rt_thinker *t, const char *name) {
    cwick_st_tensor *slot = (cwick_st_tensor *)cwick_xmalloc(sizeof(*slot));
    cwick_error err;
    memset(&err, 0, sizeof(err));
    if (cwick_safetensors_get(&t->saf, name, slot, &err) != CWICK_OK) {
        CWICK_INFO(RT_TAG, "missing tensor %s", name);
        free(slot);
        return NULL;
    }
    return slot;
}

/* BF16 [n] -> f32 [n]; the norm and scale vectors are tiny and read on every
 * step, so they are widened once at load. */
static float *widen_bf16(const cwick_st_tensor *st) {
    int64_t n = st->rows * st->cols;
    float *out = (float *)cwick_xmalloc(sizeof(float) * (size_t)n);
    const uint16_t *p = (const uint16_t *)st->data;
    for (int64_t i = 0; i < n; i++) out[i] = cwick_bf16_fast(p[i]);
    return out;
}

/* Binds one projection, widening its scale vector. Returns 0 when the tensor
 * or its scale is absent (a hard error for the caller). */
static int bind_lin(rt_thinker *t, rt_lin *dst, const char *fmt, int layer) {
    char name[256];
    snprintf(name, sizeof(name), fmt, layer);
    const cwick_st_tensor *w = fetch(t, name);
    if (!w) return -1;
    memset(dst, 0, sizeof(*dst));
    dst->in = w->cols;
    dst->out = w->rows;
    if (w->dtype == CWICK_ST_F8_E4M3) {
        char sn[300];
        snprintf(sn, sizeof(sn), "%s_scale", name);
        const cwick_st_tensor *s = fetch(t, sn);
        if (!s) {
            CWICK_INFO(RT_TAG, "missing scale for %s", name);
            return -1;
        }
        dst->w8 = (const uint8_t *)w->data;
        dst->sc = widen_bf16(s);
    } else if (w->dtype == CWICK_ST_BF16) {
        dst->w16 = (const uint16_t *)w->data;
    } else {
        CWICK_INFO(RT_TAG, "%s: unsupported dtype %d", name, (int)w->dtype);
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------- kernels --- */

static void rmsnorm(float *dst, const float *src, const float *w, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)src[i] * src[i];
    float r = 1.0f / sqrtf((float)(ss / n) + eps);
    if (w)
        for (int i = 0; i < n; i++) dst[i] = src[i] * r * w[i];
    else
        for (int i = 0; i < n; i++) dst[i] = src[i] * r;
}

/* Row-wise norm over an [m][n] block. The single-row rmsnorm above is used on
 * one head at a time, and calling it once per layer block leaves rows 1..m-1
 * untouched -- which silently projected zeros for every prefill row but the
 * first, while a one-token decode looked perfect. */
static void rmsnorm_rows(float *dst, const float *src, const float *w, int m, int n,
                         float eps) {
    for (int r = 0; r < m; r++)
        rmsnorm(dst + (int64_t)r * n, src + (int64_t)r * n, w, n, eps);
}

static void rmsnorm_inplace(float *v, const float *w, int n, float eps) {
    rmsnorm(v, v, w, n, eps);
}

static void silu_mul(float *dst, const float *g, const float *u, int n) {
    for (int i = 0; i < n; i++) {
        float x = g[i];
        dst[i] = (x / (1.0f + expf(-x))) * u[i];
    }
}

/* RoPE, half-split form: out = x*cos + rotate_half(x)*sin. */
static void rope_apply(float *head, const float *c, const float *s, int half) {
    for (int i = 0; i < half; i++) {
        float x1 = head[i], x2 = head[i + half];
        head[i] = x1 * c[i] - x2 * s[i];
        head[i + half] = x2 * c[i] + x1 * s[i];
    }
}

/* out[m][n] = x[m][in] @ w[n][in]^T, one row block of weights at a time. */
static void lin_cpu(const rt_thinker *t, const rt_lin *L, const float *x,
                    float *y, int m) {
    const int64_t n = L->out, k = L->in;
    if (L->w8) {
        const int64_t block = t->dw_rows;
        for (int64_t b = 0; b < n; b += block) {
            int64_t nb = n - b < block ? n - b : block;
            for (int64_t r = 0; r < nb; r++) {
                const uint8_t *src = L->w8 + (b + r) * k;
                float s = L->sc[b + r];
                float *dst = t->dw + r * k;
                for (int64_t c = 0; c < k; c++) dst[c] = DS_E4M3_LUT[src[c]] * s;
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, (int)nb,
                        (int)k, 1.0f, x, (int)k, t->dw, (int)k, 0.0f, y + b,
                        (int)n);
        }
        return;
    }
    /* BF16: widen a row block exactly the same way. */
    const int64_t block = t->dw_rows;
    for (int64_t b = 0; b < n; b += block) {
        int64_t nb = n - b < block ? n - b : block;
        for (int64_t r = 0; r < nb; r++) {
            const uint16_t *src = L->w16 + (b + r) * k;
            float *dst = t->dw + r * k;
            for (int64_t c = 0; c < k; c++) dst[c] = cwick_bf16_fast(src[c]);
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, (int)nb, (int)k,
                    1.0f, x, (int)k, t->dw, (int)k, 0.0f, y + b, (int)n);
    }
}

static void lin_apply(rt_thinker *t, const rt_lin *L, const float *x, float *y,
                      int m) {
    if (t->gpu && L->w8 &&
        cwick_metal_gemm_fp8_row(L->w8, L->sc, x, y, (uint32_t)L->in,
                                 (uint32_t)L->out, (uint32_t)m))
        return;
    lin_cpu(t, L, x, y, m);
}

/* lm_head is BF16 and much larger than any single projection; the GPU path
 * (already in cwick for the other checkpoints) reads it in place. */
static void lmhead_apply(rt_thinker *t, const float *x, float *logits) {
    if (t->gpu && cwick_metal_gemv_bf16(t->lmhead, x, logits, (uint32_t)t->H,
                                        (uint32_t)t->VOCAB))
        return;
    for (int v = 0; v < t->VOCAB; v++) {
        const uint16_t *row = t->lmhead + (int64_t)v * t->H;
        float acc = 0.0f;
        for (int i = 0; i < t->H; i++) acc += cwick_bf16_fast(row[i]) * x[i];
        logits[v] = acc;
    }
}

/* ----------------------------------------------------------- attention --- */

/* Causal attention over the KV cache for `m` new rows starting at `p0`.
 * q is [m][NQ*HD], out is [m][NQ*HD]. Heads sharing a KV head are batched
 * into one GEMM, so a layer costs 2*NKV GEMMs rather than 2*NQ. */
static void attention(rt_thinker *t, int layer, const float *q, float *out,
                      int m, int p0) {
    const int HD = t->HD, G = t->head_group, NKV = t->NKV;
    const int L = p0 + m;
    float *kc = t->kc + (int64_t)layer * t->max_seq * (NKV * HD);
    float *vc = t->vc + (int64_t)layer * t->max_seq * (NKV * HD);
    const float scale = 1.0f / sqrtf((float)HD);

    for (int hkv = 0; hkv < NKV; hkv++) {
        /* K/V rows for this head, gathered contiguous: [L][HD]. */
        float *kg = t->gkv;
        float *vg = t->gkv + (int64_t)L * HD;
        for (int i = 0; i < L; i++) {
            const float *ksrc = kc + ((int64_t)i * NKV + hkv) * HD;
            const float *vsrc = vc + ((int64_t)i * NKV + hkv) * HD;
            memcpy(kg + (int64_t)i * HD, ksrc, sizeof(float) * HD);
            memcpy(vg + (int64_t)i * HD, vsrc, sizeof(float) * HD);
        }
        /* Q rows of the G heads that share this KV head: [m*G][HD]. */
        float *qg = t->gq;
        for (int i = 0; i < m; i++)
            for (int g = 0; g < G; g++) {
                int h = hkv * G + g;
                memcpy(qg + ((int64_t)i * G + g) * HD,
                       q + (int64_t)i * (t->NQ * HD) + h * HD, sizeof(float) * HD);
            }
        float *S = t->scores;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m * G, L, HD,
                    scale, qg, HD, kg, HD, 0.0f, S, L);
        /* Causal mask + softmax. Row r of S belongs to (i = r/G). */
        for (int r = 0; r < m * G; r++) {
            int i = r / G;
            int limit = p0 + i + 1;
            float *row = S + (int64_t)r * L;
            float mx = -INFINITY;
            for (int j = 0; j < limit; j++)
                if (row[j] > mx) mx = row[j];
            float sum = 0.0f;
            for (int j = 0; j < limit; j++) {
                float e = expf(row[j] - mx);
                row[j] = e;
                sum += e;
            }
            float inv = 1.0f / sum;
            for (int j = 0; j < limit; j++) row[j] *= inv;
            /* The masked-out tail still holds raw dot products, and the S@V
             * GEMM spans all L columns -- leaving it in mixes future positions
             * into every row of a prefill batch. Decode never shows this: with
             * m == 1 the limit always covers the whole row. */
            for (int j = limit; j < L; j++) row[j] = 0.0f;
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m * G, HD, L,
                    1.0f, S, L, vg, HD, 0.0f, qg, HD);
        for (int i = 0; i < m; i++)
            for (int g = 0; g < G; g++) {
                int h = hkv * G + g;
                memcpy(out + (int64_t)i * (t->NQ * HD) + h * HD,
                       qg + ((int64_t)i * G + g) * HD, sizeof(float) * HD);
            }
    }
}

/* ------------------------------------------------------------ forward --- */

/* RT_DUMP_DIR=<dir> writes the first layer's intermediates as raw f32. The
 * only way to localise a wrong number in a 64-layer stack is to compare one
 * layer against an independent implementation, and that needs the tensors. */
static void dump(const char *name, const float *p, int64_t n) {
    static const char *dir;
    static int once;
    if (!once) {
        dir = getenv("RT_DUMP_DIR");
        once = 1;
    }
    if (!dir || !p) return;
    char path[1200];
    snprintf(path, sizeof(path), "%s/%s.f32", dir, name);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(p, sizeof(float), (size_t)n, f);
    fclose(f);
}

static int forward_chunk(rt_thinker *t, const float *emb, int m) {
    const int H = t->H, NQ = t->NQ, NKV = t->NKV, HD = t->HD, INTER = t->INTER;
    const int p0 = t->pos;
    if (p0 + m > t->max_seq) {
        CWICK_INFO(RT_TAG, "context full (%d + %d > %d)", p0, m, t->max_seq);
        return -1;
    }
    memcpy(t->x, emb, sizeof(float) * (size_t)m * H);
    dump("x0", t->x, (int64_t)m * H);

    for (int l = 0; l < t->NL; l++) {
        rt_layer *L = &t->lay[l];
        rmsnorm_rows(t->xn, t->x, L->in_ln, m, H, t->eps);
        if (l == 0) dump("xn", t->xn, (int64_t)m * H);
        lin_apply(t, &L->q, t->xn, t->hq, m);
        lin_apply(t, &L->k, t->xn, t->hk, m);
        lin_apply(t, &L->v, t->xn, t->hv, m);
        if (l == 0) {
            dump("q_raw", t->hq, (int64_t)m * NQ * HD);
            dump("k_raw", t->hk, (int64_t)m * NKV * HD);
            dump("v_raw", t->hv, (int64_t)m * NKV * HD);
        }

        for (int i = 0; i < m; i++) {
            const float *c = t->rc + (int64_t)(p0 + i) * (HD / 2);
            const float *s = t->rs + (int64_t)(p0 + i) * (HD / 2);
            for (int h = 0; h < NQ; h++) {
                float *qh = t->hq + (int64_t)i * (NQ * HD) + h * HD;
                rmsnorm_inplace(qh, L->q_norm, HD, t->eps);
                rope_apply(qh, c, s, HD / 2);
            }
            for (int h = 0; h < NKV; h++) {
                float *kh = t->hk + (int64_t)i * (NKV * HD) + h * HD;
                rmsnorm_inplace(kh, L->k_norm, HD, t->eps);
                rope_apply(kh, c, s, HD / 2);
            }
        }
        /* K/V into the cache: the cache row holds all KV heads contiguously,
         * exactly the layout t->hk / t->hv already have. */
        memcpy(t->kc + ((int64_t)l * t->max_seq + p0) * (NKV * HD), t->hk,
               sizeof(float) * (size_t)m * NKV * HD);
        memcpy(t->vc + ((int64_t)l * t->max_seq + p0) * (NKV * HD), t->hv,
               sizeof(float) * (size_t)m * NKV * HD);

        attention(t, l, t->hq, t->att, m, p0);
        if (l == 0) dump("att", t->att, (int64_t)m * NQ * HD);

        lin_apply(t, &L->o, t->att, t->ao, m);
        for (int64_t i = 0; i < (int64_t)m * H; i++) t->x[i] += t->ao[i];
        if (l == 0) dump("xatt", t->x, (int64_t)m * H);

        rmsnorm_rows(t->xn, t->x, L->post_ln, m, H, t->eps);
        lin_apply(t, &L->gate, t->xn, t->ga, m);
        lin_apply(t, &L->up, t->xn, t->gu, m);
        if (l == 0) {
            dump("ga", t->ga, (int64_t)m * INTER);
            dump("gu", t->gu, (int64_t)m * INTER);
        }
        silu_mul(t->ac, t->ga, t->gu, m * INTER);
        lin_apply(t, &L->down, t->ac, t->tmp, m);
        for (int64_t i = 0; i < (int64_t)m * H; i++) t->x[i] += t->tmp[i];
        if (l == 0) dump("x1", t->x, (int64_t)m * H);
    }

    t->pos += m;
    return 0;
}

/* --------------------------------------------------------------- api --- */

static float cfg_num(json_value *o, const char *key, float dflt) {
    json_value *v = cwick_json_get(o, key);
    return v ? (float)cwick_json_num(v, dflt) : dflt;
}

static int cfg_int(json_value *o, const char *key, int dflt) {
    json_value *v = cwick_json_get(o, key);
    return v ? (int)cwick_json_int(v, dflt) : dflt;
}

rt_thinker *rt_thinker_open(const char *dir, rt_backend backend, int max_seq,
                            cwick_error *err) {
    char path[1200];
    cwick_path_join(path, sizeof(path), dir, "config.json");
    char *text = NULL;
    size_t len = 0;
    if (cwick_read_file(path, &text, &len) != 0) {
        if (err) cwick_error_set(err, CWICK_ERR_IO, "cannot read %s", path);
        return NULL;
    }
    json_value *root = cwick_json_parse(text, len, err);
    free(text);
    if (!root) return NULL;
    json_value *tc = cwick_json_get(root, "text_config");
    if (!tc) {
        cwick_json_free(root);
        if (err) cwick_error_set(err, CWICK_ERR_PARSE, "%s: no text_config", path);
        return NULL;
    }

    rt_thinker *t = (rt_thinker *)xmalloc0(sizeof(*t));
    t->H = cfg_int(tc, "hidden_size", 5120);
    t->NQ = cfg_int(tc, "num_attention_heads", 64);
    t->NKV = cfg_int(tc, "num_key_value_heads", 8);
    t->HD = cfg_int(tc, "head_dim", 128);
    t->INTER = cfg_int(tc, "intermediate_size", 25600);
    t->NL = cfg_int(tc, "num_hidden_layers", 64);
    t->VOCAB = cfg_int(tc, "vocab_size", 151936);
    t->eps = cfg_num(tc, "rms_norm_eps", 1e-6f);
    t->theta = cfg_num(tc, "rope_theta", 5e6f);
    json_value *rp = cwick_json_get(tc, "rope_parameters");
    if (rp) t->theta = cfg_num(rp, "rope_theta", t->theta);
    cwick_json_free(root);

    t->head_group = t->NQ / t->NKV;
    t->max_seq = max_seq > 0 ? max_seq : 4096;
    t->cap = RT_PREFILL_CHUNK;
    /* DS_E4M3_LUT is filled on first use; the inner loops read it directly,
     * so it has to be materialised before the first GEMV, not during it. */
    (void)ds_e4m3_to_f32(0);

    if (backend != RT_BACKEND_CPU) {
        t->gpu = cwick_metal_init() ? 1 : 0;
        if (backend == RT_BACKEND_METAL && !t->gpu) {
            CWICK_INFO(RT_TAG, "Metal unavailable; decoder runs on the CPU");
        }
    }

    if (cwick_safetensors_open(dir, &t->saf, err) != CWICK_OK) {
        free(t);
        return NULL;
    }

    /* This checkpoint is not a file read once: every token walks all 64 layers,
     * so the sequential hint installed at mmap time lets the kernel drop pages
     * behind the read pointer and the next turn faults them back from SSD.
     * Resident by default: a live conversation must not pay a multi-second
     * weight fault mid-turn. RT_THINKER_RESIDENT=0 opts back into lazy mmap. */
    const char *res = getenv("RT_THINKER_RESIDENT");
    if (!res || atoi(res) != 0) {
        double r0 = (double)cwick_prof_ns() / 1e9;
        uint64_t got = cwick_safetensors_make_resident(&t->saf, "language_model.", 1);
        CWICK_INFO(RT_TAG, "decoder weights resident: %.2f GiB in %.2f s (%s)",
                   (double)got / 1073741824.0, (double)cwick_prof_ns() / 1e9 - r0,
                   t->saf.pinned_wired ? "wired" : "page cache");
    }

    const cwick_st_tensor *emb = fetch(t, "language_model.model.embed_tokens.weight");
    const cwick_st_tensor *lmh = fetch(t, "language_model.lm_head.weight");
    if (!emb || !lmh) {
        CWICK_INFO(RT_TAG, "checkpoint is missing the embedding or lm_head");
        free(t);
        return NULL;
    }
    t->embed = (const uint16_t *)emb->data;
    t->lmhead = (const uint16_t *)lmh->data;
    const cwick_st_tensor *fnorm = fetch(t, "language_model.model.norm.weight");
    if (!fnorm) {
        CWICK_INFO(RT_TAG, "checkpoint is missing the final norm");
        free(t);
        return NULL;
    }
    t->final_norm = widen_bf16(fnorm);

    t->lay = (rt_layer *)xmalloc0(sizeof(rt_layer) * (size_t)t->NL);
    char name[256];
    for (int l = 0; l < t->NL; l++) {
        rt_layer *L = &t->lay[l];
        if (bind_lin(t, &L->q, "language_model.model.layers.%d.self_attn.q_proj.weight", l) ||
            bind_lin(t, &L->k, "language_model.model.layers.%d.self_attn.k_proj.weight", l) ||
            bind_lin(t, &L->v, "language_model.model.layers.%d.self_attn.v_proj.weight", l) ||
            bind_lin(t, &L->o, "language_model.model.layers.%d.self_attn.o_proj.weight", l) ||
            bind_lin(t, &L->gate, "language_model.model.layers.%d.mlp.gate_proj.weight", l) ||
            bind_lin(t, &L->up, "language_model.model.layers.%d.mlp.up_proj.weight", l) ||
            bind_lin(t, &L->down, "language_model.model.layers.%d.mlp.down_proj.weight", l)) {
            CWICK_INFO(RT_TAG, "layer %d: projection binding failed", l);
            rt_thinker_close(t);
            return NULL;
        }
        struct { const char *sfx; const float **dst; } norms[] = {
            {"input_layernorm.weight", &L->in_ln},
            {"post_attention_layernorm.weight", &L->post_ln},
            {"self_attn.q_norm.weight", &L->q_norm},
            {"self_attn.k_norm.weight", &L->k_norm},
        };
        for (int i = 0; i < 4; i++) {
            snprintf(name, sizeof(name), "language_model.model.layers.%d.%s", l,
                     norms[i].sfx);
            const cwick_st_tensor *s = fetch(t, name);
            if (!s) {
                rt_thinker_close(t);
                return NULL;
            }
            *(float **)norms[i].dst = widen_bf16(s);
        }
    }

    /* RoPE tables: inv_freq[i] = theta^(-2i/HD), duplicated over the two
     * halves the rotate-half form expects. */
    t->rc = (float *)xmalloc0(sizeof(float) * (size_t)t->max_seq * (t->HD / 2));
    t->rs = (float *)xmalloc0(sizeof(float) * (size_t)t->max_seq * (t->HD / 2));
    double *inv = (double *)cwick_xmalloc(sizeof(double) * (size_t)(t->HD / 2));
    for (int i = 0; i < t->HD / 2; i++)
        inv[i] = 1.0 / pow((double)t->theta, (double)(2 * i) / (double)t->HD);
    for (int p = 0; p < t->max_seq; p++)
        for (int i = 0; i < t->HD / 2; i++) {
            double a = (double)p * inv[i];
            t->rc[(int64_t)p * (t->HD / 2) + i] = (float)cos(a);
            t->rs[(int64_t)p * (t->HD / 2) + i] = (float)sin(a);
        }
    free(inv);

    int64_t kvw = (int64_t)t->NKV * t->HD;
    t->kc = (float *)xmalloc0(sizeof(float) * (size_t)t->NL * t->max_seq * kvw);
    t->vc = (float *)xmalloc0(sizeof(float) * (size_t)t->NL * t->max_seq * kvw);

    t->x = (float *)xmalloc0(sizeof(float) * (size_t)t->cap * t->H);
    t->xn = (float *)xmalloc0(sizeof(float) * (size_t)t->cap * t->H);
    t->ao = (float *)xmalloc0(sizeof(float) * (size_t)t->cap * t->H);
    t->att = (float *)xmalloc0(sizeof(float) * (size_t)t->cap * t->NQ * t->HD);
    t->hq = (float *)xmalloc0(sizeof(float) * (size_t)t->cap * t->NQ * t->HD);
    t->hk = (float *)xmalloc0(sizeof(float) * (size_t)t->cap * t->NKV * t->HD);
    t->hv = (float *)xmalloc0(sizeof(float) * (size_t)t->cap * t->NKV * t->HD);
    t->ga = (float *)xmalloc0(sizeof(float) * (size_t)t->cap * t->INTER);
    t->gu = (float *)xmalloc0(sizeof(float) * (size_t)t->cap * t->INTER);
    t->ac = (float *)xmalloc0(sizeof(float) * (size_t)t->cap * t->INTER);
    t->logits = (float *)xmalloc0(sizeof(float) * (size_t)t->VOCAB);
    /* Attention scratch: one gather of K and V for the head being processed
     * ([2][max_seq][HD]), the gathered Q rows of the heads sharing that KV
     * head ([cap*G][HD]), and the score matrix ([cap*G][max_seq]). */
    t->tmp = (float *)xmalloc0(sizeof(float) * (size_t)t->cap * t->H);
    t->gkv = (float *)xmalloc0(sizeof(float) * 2 * (size_t)t->max_seq * t->HD);
    t->gq = (float *)xmalloc0(sizeof(float) * (size_t)t->cap * t->NQ * t->HD);
    t->scores = (float *)xmalloc0(sizeof(float) * (size_t)t->cap * t->head_group *
                                 (size_t)t->max_seq);

    /* CPU dequant block: keep the working set at a few MiB. */
    t->dw_rows = 256;
    t->dw = (float *)cwick_xmalloc(sizeof(float) * (size_t)t->dw_rows * t->INTER);

    t->tok = NULL;
    CWICK_INFO(RT_TAG,
               "decoder ready: %d layers, H=%d, heads=%d/%d, inter=%d, vocab=%d, "
               "ctx=%d, %s",
               t->NL, t->H, t->NQ, t->NKV, t->INTER, t->VOCAB, t->max_seq,
               t->gpu ? "GPU kernels" : "CPU kernels");
    return t;
}

void rt_thinker_close(rt_thinker *t) {
    if (!t) return;
    cwick_safetensors_close(&t->saf);
    free(t->logits);
    free((void *)t->final_norm);
    free(t->scores);
    free(t->gq);
    free(t->gkv);
    free(t->tmp);
    free(t->ac);
    free(t->gu);
    free(t->ga);
    free(t->hv);
    free(t->hk);
    free(t->hq);
    free(t->att);
    free(t->ao);
    free(t->xn);
    free(t->x);
    free(t->kc);
    free(t->vc);
    free(t->rc);
    free(t->rs);
    free(t->dw);
    for (int l = 0; l < t->NL; l++) {
        rt_layer *L = &t->lay[l];
        free((void *)L->in_ln);
        free((void *)L->post_ln);
        free((void *)L->q_norm);
        free((void *)L->k_norm);
        free((void *)L->q.sc);
        free((void *)L->k.sc);
        free((void *)L->v.sc);
        free((void *)L->o.sc);
        free((void *)L->gate.sc);
        free((void *)L->up.sc);
        free((void *)L->down.sc);
    }
    free(t->lay);
    free(t);
}

const cwick_tokenizer *rt_thinker_tokenizer(const rt_thinker *t) { return t->tok; }
int rt_thinker_hidden(const rt_thinker *t) { return t->H; }
int rt_thinker_layers(const rt_thinker *t) { return t->NL; }
int rt_thinker_vocab(const rt_thinker *t) { return t->VOCAB; }
int rt_thinker_pos(const rt_thinker *t) { return t->pos; }
int rt_thinker_capacity(const rt_thinker *t) { return t->max_seq; }
bool rt_thinker_on_gpu(const rt_thinker *t) { return t->gpu != 0; }

int rt_thinker_embed_row(rt_thinker *t, int32_t id, float *out) {
    if (id < 0 || id >= t->VOCAB) return -1;
    const uint16_t *row = t->embed + (int64_t)id * t->H;
    for (int i = 0; i < t->H; i++) out[i] = cwick_bf16_fast(row[i]);
    return 0;
}

void rt_thinker_reset(rt_thinker *t) { t->pos = 0; }

void rt_thinker_truncate(rt_thinker *t, int pos) {
    if (pos < 0) pos = 0;
    if (pos < t->pos) t->pos = pos;
}

int rt_thinker_forward(rt_thinker *t, const float *embeds, int n) {
    if (n <= 0) return -1;
    int done = 0, last = 0;
    while (done < n) {
        int m = n - done;
        if (m > t->cap) m = t->cap;
        if (forward_chunk(t, embeds + (int64_t)done * t->H, m) != 0) return -1;
        done += m;
        last = m;
    }
    /* The head only ever needs the newest position's logits, and they are
     * taken after the decoder's own final norm -- skipping it turns the
     * unnormalised residual into logits, which samples as noise. */
    float *lastrow = t->x + (int64_t)(last - 1) * t->H;
    dump("xlast", lastrow, t->H);
    rmsnorm(t->xn, lastrow, t->final_norm, t->H, t->eps);
    dump("xnfinal", t->xn, t->H);
    lmhead_apply(t, t->xn, t->logits);
    dump("logits", t->logits, t->VOCAB);
    return 0;
}

const float *rt_thinker_logits(rt_thinker *t) { return t->logits; }
