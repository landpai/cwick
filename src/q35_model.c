/*
 * q35_model.c — Qwen3.5 hybrid attention: config, weights and the forward pass.
 *
 * See q35.h for the geometry and for what the two layer kinds are. Three
 * things about this port are worth knowing before reading it.
 *
 * 1. Prefill is layer-major over a block of Q35_CHUNK positions. Every
 *    projection runs once for the whole block, because a projection run per
 *    position would re-read 18 GiB of weights per token. The delta-rule
 *    recurrence cannot be batched that way -- it is a scan by definition -- so
 *    it walks the block position by position. It is still parallel, just not
 *    over positions: each value head owns its own state matrix and reads no
 *    other, so the work is split by head (`lin_rec_heads`). The attention
 *    phase is split the same way, over (head, position) pairs (`attn_items`).
 *    Both were serial once, and together they were 58% of a block's time on a
 *    single core -- the numbers are in DECISIONS §52.
 *
 * 2. The state of a DeltaNet layer is not a cache of keys and values, it is a
 *    [128, 128] matrix of f32 per value head, updated in place. That matrix is
 *    stored transposed -- [value_dim][key_dim] rather than [key_dim][value_dim]
 *    -- because all three of its uses (S^T k, the rank-one update, S^T q) then
 *    read it with a contiguous inner loop.
 *
 * 3. Two pieces of the checkpoint are never read: the vision tower and the
 *    text shards' extra tensors. The bytes stay on disk; q35_model_open simply
 *    does not bind them. The MTP draft head is bound, from its own file (see
 *    mtp_open).
 */
#include "q35.h"
#include "q35_gpu.h"
#include "log.h"
#include "mem.h"

#include <math.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "json.h"
#include "log.h"
#include "mem.h"
/* The prefill reporter is shared with the MiniMax path (begin / done / end).
 * Layer granularity is what makes a 72-second block look alive instead of
 * hung, and it is where cwick:prefill_layer gets its number. */
#include "mm.h"
#include "util.h"

/* ------------------------------------------------------------------ */
/* small helpers                                                       */
/* ------------------------------------------------------------------ */

static inline float bf(const uint16_t *p) { return cwick_bf16_to_f32(*p); }

static inline float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

/* softplus with the overflow guard transformers uses: for large x the answer
 * is x, and expf(88) is already inf. */
static inline float softplus_f(float x) {
    if (x > 20.0f) return x;
    return log1pf(expf(x));
}

/* ---- phase timers (CWICK_Q35_PROF=1) ---------------------------------
 *
 * A prefill block is a mix of three things that scale differently: pooled
 * GEMMs (thread-count dependent), the delta-rule scan (serial by definition)
 * and the attention loop (serial over keys). Without a split, "prefill is
 * slow" has no address -- and the first guess (the GEMMs) was wrong: at 12
 * threads and at 1 thread this checkpoint's prefill took the same time. */
static int q35_prof_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("CWICK_Q35_PROF");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v;
}

static double q35_prof_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static double prof_gemm_ms, prof_rec_ms, prof_attn_ms, prof_act_ms;

/* Attention has three parts with different fixes, so "attention is 6 seconds"
 * is not actionable. These are written from inside the pool, hence atomics;
 * one add per tile per phase is noise next to a tile that costs ~100 us. */
static _Atomic double prof_qk_ms, prof_sm_ms, prof_av_ms;

/* out[i] = in[i] * rsqrt(mean(in^2) + eps) * (1 + w[i]). The RMS is taken over
 * the whole vector, which for this checkpoint means one head.
 *
 * The `1 + w` is the one thing here that differs from every other backend in
 * this tree, and it is not cosmetic: the checkpoint's norm weights sit around
 * zero (layer 0's input norm averages 0.033, layer 1's post norm -0.11), not
 * around one. Read as a plain multiplier those weights scale the residual
 * stream down to 3% per layer -- the model then emits an endless run of spaces
 * and newlines, which is exactly what the first run of this port did. The
 * table settled it: every norm in the file averages near zero, so the weights
 * are offsets from one, the same convention as MiniMax's `use_gemma_norm`. */
static void rms_norm_w(float *out, const float *in, int n, const float *w,
                       float eps) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) acc += in[i] * in[i];
    float rstd = 1.0f / sqrtf(acc / (float)n + eps);
    if (w) {
        for (int i = 0; i < n; i++) out[i] = in[i] * rstd * (1.0f + w[i]);
    } else {
        for (int i = 0; i < n; i++) out[i] = in[i] * rstd;
    }
}

/* L2 normalisation of one head, in place. The epsilon is applied to the norm
 * before the divide, which is what the reference implementation does --
 * clamping the squared norm instead shifts every near-zero head. */
static void l2_norm_inplace(float *v, int n, float eps) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) acc += v[i] * v[i];
    float inv = 1.0f / fmaxf(sqrtf(acc), eps);
    for (int i = 0; i < n; i++) v[i] *= inv;
}

/* Partial rotation, Llama style. The rotated slice splits into two halves that
 * share one frequency per index: index i pairs with index i + pairs, and
 *
 *     out[i]        = v[i]        * cos - v[i + pairs] * sin
 *     out[i + pairs] = v[i + pairs]* cos + v[i]        * sin
 *
 * This is `rotate_half` from the reference implementation, not the adjacent-
 * pair form: with the pairs (2i, 2i+1) the two halves would turn against a
 * different set of frequencies and every position's rotation would be wrong,
 * which is exactly the state this port spent its first runs in. */
static void rope_apply(float *v, const float *cos_p, const float *sin_p,
                       int rotary) {
    int pairs = rotary / 2;
    for (int i = 0; i < pairs; i++) {
        float c = cos_p[i], s = sin_p[i];
        float a = v[i], b = v[i + pairs];
        v[i] = a * c - b * s;
        v[i + pairs] = b * c + a * s;
    }
}

/* ------------------------------------------------------------------ */
/* config                                                              */
/* ------------------------------------------------------------------ */

static int64_t cfg_int(json_value *o, const char *k, int64_t dflt) {
    json_value *v = cwick_json_get(o, k);
    if (!v) return dflt;
    if (v->kind == JSON_NUMBER) return v->integer;
    return dflt;
}

static double cfg_num(json_value *o, const char *k, double dflt) {
    json_value *v = cwick_json_get(o, k);
    if (v && v->kind == JSON_NUMBER) return v->number;
    return dflt;
}

static void *xcalloc(size_t n, size_t sz) {
    void *p = cwick_xmalloc(n * sz);
    memset(p, 0, n * sz);
    return p;
}

static uint8_t *f32_from_bf16(const uint16_t *src, int64_t n) {
    float *d = cwick_xmalloc((size_t)n * sizeof(float));
    for (int64_t i = 0; i < n; i++) d[i] = cwick_bf16_to_f32(src[i]);
    return (uint8_t *)d;
}

static int read_config(q35_config *c, const char *root, cwick_error *err) {
    char path[1200];
    snprintf(path, sizeof(path), "%s/config.json", root);
    char *text = NULL;
    size_t len = 0;
    if (cwick_read_file(path, &text, &len) != 0) {
        cwick_error_set(err, CWICK_ERR_IO, "cannot read %s", path);
        return -1;
    }
    cwick_error jerr;
    cwick_error_clear(&jerr);
    json_value *root_v = cwick_json_parse(text, len, &jerr);
    free(text);
    if (!root_v) {
        cwick_error_set(err, CWICK_ERR_PARSE, "config.json is not JSON: %s",
                        cwick_error_message(&jerr));
        return -1;
    }
    json_value *tc = cwick_json_get(root_v, "text_config");
    json_value *o = (tc && tc->kind == JSON_OBJECT) ? tc : root_v;

    c->dim = (int)cfg_int(o, "hidden_size", 0);
    c->n_layers = (int)cfg_int(o, "num_hidden_layers", 0);
    c->n_heads = (int)cfg_int(o, "num_attention_heads", 0);
    c->n_kv_heads = (int)cfg_int(o, "num_key_value_heads", 0);
    c->head_dim = (int)cfg_int(o, "head_dim", 0);
    c->inter_dim = (int)cfg_int(o, "intermediate_size", 0);
    c->vocab_size = (int)cfg_int(o, "vocab_size", 0);
    c->lin_k_heads = (int)cfg_int(o, "linear_num_key_heads", 0);
    c->lin_v_heads = (int)cfg_int(o, "linear_num_value_heads", 0);
    c->lin_k_dim = (int)cfg_int(o, "linear_key_head_dim", 0);
    c->lin_v_dim = (int)cfg_int(o, "linear_value_head_dim", 0);
    c->conv_kernel = (int)cfg_int(o, "linear_conv_kernel_dim", 4);
    c->max_seq_len = (int)cfg_int(o, "max_position_embeddings", 32768);
    c->rms_eps = (float)cfg_num(o, "rms_norm_eps", 1e-6);
    c->l2_eps = 1e-6f;

    if (c->head_dim <= 0 && c->n_heads > 0) c->head_dim = c->dim / c->n_heads;

    json_value *rp = cwick_json_get(o, "rope_parameters");
    if (rp && rp->kind == JSON_OBJECT) {
        c->rope_theta = cfg_num(rp, "rope_theta", 1e7);
        double prf = cfg_num(rp, "partial_rotary_factor", 1.0);
        c->rotary_dim = (int)((double)c->head_dim * prf);
    } else {
        c->rope_theta = 1e7;
        c->rotary_dim = c->head_dim;
    }
    if (c->rotary_dim <= 0 || c->rotary_dim > c->head_dim)
        c->rotary_dim = c->head_dim;
    if (c->rotary_dim % 2) c->rotary_dim--;

    if (c->dim <= 0 || c->n_layers <= 0 || c->n_layers > Q35_MAX_LAYERS ||
        c->n_heads <= 0 || c->n_kv_heads <= 0 || c->vocab_size <= 0 ||
        c->inter_dim <= 0) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                        "qwen3_5 config is missing geometry (dim=%d layers=%d "
                        "heads=%d kv=%d vocab=%d inter=%d)",
                        c->dim, c->n_layers, c->n_heads, c->n_kv_heads,
                        c->vocab_size, c->inter_dim);
        cwick_json_free(root_v);
        return -1;
    }
    if (c->lin_k_heads <= 0 || c->lin_v_heads <= 0 || c->lin_k_dim <= 0 ||
        c->lin_v_dim <= 0) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                        "qwen3_5 config has no linear-attention geometry: this "
                        "build implements the hybrid (DeltaNet + GQA) release");
        cwick_json_free(root_v);
        return -1;
    }
    if (c->lin_v_heads != c->lin_k_heads * Q35_LIN_KV_GROUP) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                        "qwen3_5: %d value heads is not %d key heads x %d; the "
                        "grouped value-head layout this build assumes does not "
                        "apply", c->lin_v_heads, c->lin_k_heads,
                        Q35_LIN_KV_GROUP);
        cwick_json_free(root_v);
        return -1;
    }

    /* layer_types names the mixer of every layer; `full_attention_interval`
     * describes the same thing less directly (every Nth layer is attention).
     * Prefer the explicit list and fall back to the interval. */
    c->is_linear = xcalloc((size_t)c->n_layers, 1);
    json_value *lt = cwick_json_get(o, "layer_types");
    if (lt && lt->kind == JSON_ARRAY && (int)lt->count == c->n_layers) {
        for (int i = 0; i < c->n_layers; i++) {
            const char *s = cwick_json_str(cwick_json_array_at(lt, i));
            c->is_linear[i] = (s && strstr(s, "linear") != NULL) ? 1 : 0;
        }
    } else {
        int64_t interval = cfg_int(o, "full_attention_interval", 4);
        if (interval < 1) interval = 4;
        for (int i = 0; i < c->n_layers; i++)
            c->is_linear[i] = ((i + 1) % interval) ? 1 : 0;
    }
    c->n_lin_layers = 0;
    for (int i = 0; i < c->n_layers; i++)
        if (c->is_linear[i]) c->n_lin_layers++;
    c->n_full_layers = c->n_layers - c->n_lin_layers;

    /* `attn_output_gate` decides whether q_proj is one head-width wide or two;
     * the tensor shape check in bind_layer settles it a second time. */
    json_value *ag = cwick_json_get(o, "attn_output_gate");
    c->attn_gate = (ag && ag->kind == JSON_BOOL) ? (ag->boolean ? 1 : 0) : 1;

    cwick_json_free(root_v);
    return 0;
}

int q35_linear_layers(const q35_config *c) {
    if (!c) return 0;
    return c->n_lin_layers;
}

/* ------------------------------------------------------------------ */
/* weights                                                             */
/* ------------------------------------------------------------------ */

static const uint16_t *bind_mat(const q35_model *m, const char *name,
                                int64_t rows, int64_t cols, int required,
                                cwick_error *err) {
    cwick_error_clear(err);
    cwick_st_tensor t;
    if (cwick_safetensors_get(&m->saf, name, &t, err) != CWICK_OK) {
        if (!required) {
            cwick_error_clear(err);
            return NULL;
        }
        return NULL;
    }
    if (t.dtype != CWICK_ST_BF16) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                        "%s is not BF16 (dtype %d); this build runs the BF16 "
                        "release only", name, (int)t.dtype);
        return NULL;
    }
    if (t.rows != rows || t.cols != cols) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                        "%s is [%lld, %lld], expected [%lld, %lld]",
                        name, (long long)t.rows, (long long)t.cols,
                        (long long)rows, (long long)cols);
        return NULL;
    }
    return (const uint16_t *)t.data;
}

/* A 1-D tensor, widened to f32. */
static float *bind_vec(const q35_model *m, const char *name, int64_t n,
                       cwick_error *err) {
    cwick_error_clear(err);
    cwick_st_tensor t;
    if (cwick_safetensors_get(&m->saf, name, &t, err) != CWICK_OK) return NULL;
    if (t.dtype != CWICK_ST_BF16) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "%s is not BF16", name);
        return NULL;
    }
    if (t.rows * t.cols != n) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                        "%s holds %lld values, expected %lld", name,
                        (long long)(t.rows * t.cols), (long long)n);
        return NULL;
    }
    return (float *)f32_from_bf16((const uint16_t *)t.data, n);
}

/* The depthwise convolution's kernel, transposed to [tap][channel] and widened
 * to f32. The on-disk layout is [channel, 1, tap], so the forward pass would
 * otherwise read one f32 per channel with a stride of `tap` inside its
 * innermost loop. */
static float *bind_conv(const q35_model *m, const char *name, int qkv_dim,
                        int taps, cwick_error *err) {
    cwick_error_clear(err);
    cwick_st_tensor t;
    if (cwick_safetensors_get(&m->saf, name, &t, err) != CWICK_OK) return NULL;
    if (t.dtype != CWICK_ST_BF16) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "%s is not BF16", name);
        return NULL;
    }
    if (t.rows != qkv_dim || t.cols != taps) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                        "%s is [%lld, %lld], expected [%d, %d]", name,
                        (long long)t.rows, (long long)t.cols, qkv_dim, taps);
        return NULL;
    }
    const uint16_t *src = (const uint16_t *)t.data;
    float *w = cwick_xmalloc((size_t)taps * (size_t)qkv_dim * sizeof(float));
    for (int c = 0; c < qkv_dim; c++)
        for (int k = 0; k < taps; k++)
            w[(size_t)k * qkv_dim + c] =
                cwick_bf16_to_f32(src[(size_t)c * taps + k]);
    return w;
}

static int bind_layer(q35_model *m, int l, cwick_error *err) {
    const q35_config *c = &m->cfg;
    q35_layer *L = &m->layers[l];
    L->is_linear = c->is_linear[l];
    char name[256];

#define BIND(field, fmt, r, cc)                                               \
    do {                                                                      \
        snprintf(name, sizeof(name), "model.language_model.layers.%d." fmt,    \
                 l);                                                          \
        L->field = bind_mat(m, name, (r), (cc), 1, err);                      \
        if (!L->field) return -1;                                             \
    } while (0)

    snprintf(name, sizeof(name),
             "model.language_model.layers.%d.input_layernorm.weight", l);
    L->in_norm = bind_vec(m, name, c->dim, err);
    if (!L->in_norm) return -1;
    snprintf(name, sizeof(name),
             "model.language_model.layers.%d.post_attention_layernorm.weight", l);
    L->post_norm = bind_vec(m, name, c->dim, err);
    if (!L->post_norm) return -1;

    BIND(w_gate, "mlp.gate_proj.weight", c->inter_dim, c->dim);
    BIND(w_up, "mlp.up_proj.weight", c->inter_dim, c->dim);
    BIND(w_down, "mlp.down_proj.weight", c->dim, c->inter_dim);

    if (L->is_linear) {
        int key_dim = c->lin_k_heads * c->lin_k_dim;
        int val_dim = c->lin_v_heads * c->lin_v_dim;
        int qkv_dim = 2 * key_dim + val_dim;

        BIND(lin.w_qkv, "linear_attn.in_proj_qkv.weight", qkv_dim, c->dim);
        BIND(lin.w_z, "linear_attn.in_proj_z.weight", val_dim, c->dim);
        BIND(lin.w_a, "linear_attn.in_proj_a.weight", c->lin_v_heads, c->dim);
        BIND(lin.w_b, "linear_attn.in_proj_b.weight", c->lin_v_heads, c->dim);
        BIND(lin.w_o, "linear_attn.out_proj.weight", c->dim, val_dim);

        snprintf(name, sizeof(name),
                 "model.language_model.layers.%d.linear_attn.conv1d.weight", l);
        L->lin.conv = bind_conv(m, name, qkv_dim, c->conv_kernel, err);
        if (!L->lin.conv) return -1;

        snprintf(name, sizeof(name),
                 "model.language_model.layers.%d.linear_attn.A_log", l);
        L->lin.a_log = bind_vec(m, name, c->lin_v_heads, err);
        if (!L->lin.a_log) return -1;
        snprintf(name, sizeof(name),
                 "model.language_model.layers.%d.linear_attn.dt_bias", l);
        L->lin.dt_bias = bind_vec(m, name, c->lin_v_heads, err);
        if (!L->lin.dt_bias) return -1;
        snprintf(name, sizeof(name),
                 "model.language_model.layers.%d.linear_attn.norm.weight", l);
        L->lin.norm = bind_vec(m, name, c->lin_v_dim, err);
        if (!L->lin.norm) return -1;
    } else {
        int qwidth = c->n_heads * c->head_dim * (1 + c->attn_gate);
        int kv = c->n_kv_heads * c->head_dim;
        BIND(attn.wq, "self_attn.q_proj.weight", qwidth, c->dim);
        BIND(attn.wk, "self_attn.k_proj.weight", kv, c->dim);
        BIND(attn.wv, "self_attn.v_proj.weight", kv, c->dim);
        BIND(attn.wo, "self_attn.o_proj.weight", c->dim, c->n_heads * c->head_dim);

        snprintf(name, sizeof(name),
                 "model.language_model.layers.%d.self_attn.q_norm.weight", l);
        L->attn.q_norm = bind_vec(m, name, c->head_dim, err);
        if (!L->attn.q_norm) return -1;
        snprintf(name, sizeof(name),
                 "model.language_model.layers.%d.self_attn.k_norm.weight", l);
        L->attn.k_norm = bind_vec(m, name, c->head_dim, err);
        if (!L->attn.k_norm) return -1;
    }
#undef BIND
    return 0;
}

/* ------------------------------------------------------------------ */
/* open / close                                                        */
/* ------------------------------------------------------------------ */

static void build_rope(q35_model *m) {
    const q35_config *c = &m->cfg;
    int pairs = c->rotary_dim / 2;
    m->rope_cos = cwick_xmalloc((size_t)c->max_seq_len * (size_t)pairs *
                                sizeof(float));
    m->rope_sin = cwick_xmalloc((size_t)c->max_seq_len * (size_t)pairs *
                                sizeof(float));
    double *inv = cwick_xmalloc((size_t)pairs * sizeof(double));
    for (int i = 0; i < pairs; i++)
        inv[i] = 1.0 / pow(c->rope_theta, (double)(2 * i) / (double)c->rotary_dim);
    for (int p = 0; p < c->max_seq_len; p++) {
        for (int i = 0; i < pairs; i++) {
            double a = (double)p * inv[i];
            m->rope_cos[(size_t)p * pairs + i] = (float)cos(a);
            m->rope_sin[(size_t)p * pairs + i] = (float)sin(a);
        }
    }
    free(inv);
}

/* ---- state geometry, shared by the allocator and the wiring pass ------- */

static size_t q35_kv_row(const q35_config *c) {
    return (size_t)c->n_kv_heads * (size_t)c->head_dim;
}

static size_t q35_kv_layer_bytes(const q35_config *c) {
    return (size_t)c->max_seq_len * q35_kv_row(c) * sizeof(float);
}

/* Row conversions between the cache (f32, what the GEMMs read) and the prefix
 * snapshot (bf16, half the bytes). Defined with the attention code below. */
static void q35_widen_rows(float *dst, const uint16_t *src, size_t n);
static void q35_narrow_rows(uint16_t *dst, const float *src, size_t n);

static size_t q35_lin_key_dim(const q35_config *c) {
    return (size_t)c->lin_k_heads * (size_t)c->lin_k_dim;
}

static size_t q35_lin_val_dim(const q35_config *c) {
    return (size_t)c->lin_v_heads * (size_t)c->lin_v_dim;
}

/* One layer's recurrence state: [value heads][value dim][key dim] of f32.
 *
 * Note what this is *not*: value_dim x key_dim over the concatenated
 * dimensions. Every value head owns its own [128, 128] matrix and reads only
 * its own key head, so the total is heads x 128 x 128 -- 2 MiB per layer.
 * Allocating the product of the two concatenated widths instead (16x more)
 * still "works", because the head stride is what the forward pass uses, but it
 * pins 800 MiB of never-touched memory for the life of the process. */
static size_t q35_ssm_layer_bytes(const q35_config *c) {
    return (size_t)c->lin_v_heads * (size_t)c->lin_k_dim *
           (size_t)c->lin_v_dim * sizeof(float);
}

static size_t q35_conv_layer_bytes(const q35_config *c) {
    int taps = c->conv_kernel > 0 ? c->conv_kernel : 4;
    size_t qkv = 2 * q35_lin_key_dim(c) + q35_lin_val_dim(c);
    return (size_t)(taps - 1) * qkv * sizeof(float);
}

/* Bytes the forward pass touches that are not the checkpoint: KV, the DeltaNet
 * recurrence state, the chunk-shaped scratch, and the score matrix of one
 * attention tile. Also what the residency decision has to budget for, so it is
 * computed from the config alone. */
static uint64_t q35_state_bytes(const q35_config *c) {
    uint64_t kv = (uint64_t)c->n_full_layers * 2 * q35_kv_layer_bytes(c);
    uint64_t ssm = (uint64_t)c->n_lin_layers * q35_ssm_layer_bytes(c) *
                   Q35_STATE_SLOTS;
    uint64_t conv = (uint64_t)c->n_lin_layers * q35_conv_layer_bytes(c) *
                    Q35_STATE_SLOTS;
    uint64_t scratch = (uint64_t)Q35_CHUNK * (uint64_t)c->dim * 6 * sizeof(float);
    uint64_t scores = (uint64_t)Q35_ATTN_Q * (uint64_t)c->max_seq_len * sizeof(float);
    return kv + ssm + conv + scratch + scores;
}

/* Wire the runtime state as well as the weights. Unlike the weights this is
 * anonymous memory: the VM would rather compress it than page it, and a
 * compressed KV page has to be inflated before the next attention call reads
 * it -- a stall inside the step instead of at the edges. */
static uint64_t q35_lock_state(q35_model *m) {
    const q35_config *c = &m->cfg;
    uint64_t bytes = 0;
    for (int l = 0; l < c->n_layers; l++) {
        if (m->kcache && m->kcache[l]) {
            size_t n = q35_kv_layer_bytes(c);
            if (cwick_mem_lock_range(m->kcache[l], n, "q35 KV keys")) bytes += n;
        }
        if (m->vcache && m->vcache[l]) {
            size_t n = q35_kv_layer_bytes(c);
            if (cwick_mem_lock_range(m->vcache[l], n, "q35 KV values")) bytes += n;
        }
        if (m->ssm_state && m->ssm_state[l]) {
            size_t n = q35_ssm_layer_bytes(c) * Q35_STATE_SLOTS;
            if (cwick_mem_lock_range(m->ssm_state[l], n, "q35 DeltaNet state"))
                bytes += n;
        }
        if (m->conv_state && m->conv_state[l]) {
            size_t n = q35_conv_layer_bytes(c) * Q35_STATE_SLOTS;
            if (cwick_mem_lock_range(m->conv_state[l], n, "q35 conv state"))
                bytes += n;
        }
    }
    return bytes;
}

/* ---- prefix snapshot -------------------------------------------------
 *
 * Same contract as ds_state_pack / mm_state_pack: one opaque blob, layout
 * owned by this file, and the prefix cache stores it with kind 3 so a
 * snapshot from another backend can never be applied here. */

static size_t q35_prefix_fixed_bytes(const q35_config *c) {
    return (size_t)c->n_lin_layers *
           (q35_conv_layer_bytes(c) + q35_ssm_layer_bytes(c));
}

/* Bytes per position in the snapshot. bf16 rows, even though the live cache is
 * f32: the file is read and written far less often than the cache is, and half
 * the size means twice as many snapshots under the same budget. */
static size_t q35_prefix_per_pos(const q35_config *c) {
    return (size_t)c->n_full_layers * 2 * q35_kv_row(c) * sizeof(uint16_t);
}

size_t q35_prefix_bytes(const q35_model *m, int64_t pos) {
    const q35_config *c = &m->cfg;
    if (pos < 0) pos = 0;
    if (pos > c->max_seq_len) pos = c->max_seq_len;
    return q35_prefix_fixed_bytes(c) + (size_t)pos * q35_prefix_per_pos(c);
}

int q35_prefix_pack(const q35_model *m, int64_t pos, void *buf, size_t cap) {
    if (!m || !buf || pos < 0) return -1;
    const q35_config *c = &m->cfg;
    if (pos > c->max_seq_len) return -1;
    if (cap < q35_prefix_bytes(m, pos)) return -1;
    const size_t conv = q35_conv_layer_bytes(c);
    const size_t ssm = q35_ssm_layer_bytes(c);
    const size_t row = q35_kv_row(c);
    uint8_t *o = (uint8_t *)buf;
    for (int l = 0; l < c->n_layers; l++) {
        if (!c->is_linear[l]) continue;
        if (!m->conv_state || !m->conv_state[l] || !m->ssm_state ||
            !m->ssm_state[l])
            return -1;
        memcpy(o, m->conv_state[l], conv);
        o += conv;
        memcpy(o, m->ssm_state[l], ssm);
        o += ssm;
    }
    for (int l = 0; l < c->n_layers; l++) {
        if (c->is_linear[l]) continue;
        if (!m->kcache || !m->kcache[l] || !m->vcache || !m->vcache[l]) return -1;
        for (int which = 0; which < 2; which++) {
            const float *src = which ? m->vcache[l] : m->kcache[l];
            q35_narrow_rows((uint16_t *)o, src, (size_t)pos * row);
            o += (size_t)pos * row * sizeof(uint16_t);
        }
    }
    return 0;
}

int q35_prefix_apply(q35_model *m, const void *buf, size_t len, int64_t want,
                     int64_t *pos_out) {
    if (!m || !buf || !pos_out) return -1;
    const q35_config *c = &m->cfg;
    const size_t fixed = q35_prefix_fixed_bytes(c);
    const size_t per = q35_prefix_per_pos(c);
    if (per == 0 || len < fixed) return -1;
    const size_t span = len - fixed;
    if (span % per) return -1;
    const int64_t have = (int64_t)(span / per);
    if (have <= 0 || have > c->max_seq_len) return -1;
    /* `want` is the caller's truncation (a whole-prompt hit re-runs the last
     * position, because the snapshot carries no hidden state for lm_head).
     * The snapshot was packed with `have` rows, so the source stride follows
     * `have` while only `want` rows are written. */
    int64_t pos = have;
    if (want > 0 && want < pos) pos = want;
    const size_t conv = q35_conv_layer_bytes(c);
    const size_t ssm = q35_ssm_layer_bytes(c);
    const size_t row = q35_kv_row(c);
    const size_t src_rows = (size_t)have * row * sizeof(uint16_t);
    const uint8_t *i = (const uint8_t *)buf;
    for (int l = 0; l < c->n_layers; l++) {
        if (!c->is_linear[l]) continue;
        if (!m->conv_state || !m->conv_state[l] || !m->ssm_state ||
            !m->ssm_state[l])
            return -1;
        memcpy(m->conv_state[l], i, conv);
        i += conv;
        memcpy(m->ssm_state[l], i, ssm);
        i += ssm;
    }
    for (int l = 0; l < c->n_layers; l++) {
        if (c->is_linear[l]) continue;
        if (!m->kcache || !m->kcache[l] || !m->vcache || !m->vcache[l]) return -1;
        for (int which = 0; which < 2; which++) {
            float *dst = which ? m->vcache[l] : m->kcache[l];
            q35_widen_rows(dst, (const uint16_t *)i, (size_t)pos * row);
            i += src_rows;
        }
    }
    *pos_out = pos;
    return 0;
}

int q35_resident_wanted(uint64_t weights, uint64_t state, uint64_t host,
                        uint64_t ceiling) {
    if (weights == 0 || host == 0) return 0;
    uint64_t need = weights + state;
    /* Half of RAM, not all of it: the weights are wired for the life of the
     * process and cannot be reclaimed, so a checkpoint that needs more than
     * half the machine has no room left for the KV of a long context, for the
     * other resident models this server can switch between, or for the OS. */
    if (need > host / 2) return 0;
    /* 4 GiB of slack under the ceiling, so the resident set never lands close
     * enough to the line that the next allocation triggers the compressor. */
    if (ceiling && need + (4ull << 30) > ceiling) return 0;
    return 1;
}

static int q35_env_flag(const char *name, int dflt) {
    const char *e = getenv(name);
    if (!e || !*e) return dflt;
    if (*e == '0' || *e == 'n' || *e == 'N' || *e == 'f' || *e == 'F') return 0;
    return 1;
}

/* ---- the MTP draft head ---------------------------------------------- *
 *
 * `model-mtp.safetensors` is a single-file checkpoint and is not named in
 * `model.safetensors.index.json`, so cwick_safetensors_open cannot reach it:
 * that loader is driven by an index of shards. The header format is the same
 * one every safetensors file uses -- an 8-byte little-endian header length
 * followed by JSON -- so this maps the file and resolves the fifteen tensors
 * by name directly.
 *
 * The head is optional. A checkpoint without it simply drafts by prompt lookup,
 * which costs nothing and returns little; there is no reason to refuse to serve
 * a model over a missing drafter. */

typedef struct {
    const char *name;
    int dims;               /* 1 or 2 */
    int64_t rows, cols;
} mtp_spec;

static const void *mtp_tensor(const json_value *hdr, const void *base,
                              size_t len, size_t hlen, const char *name,
                              int64_t *rows, int64_t *cols, cwick_error *err) {
    json_value *t = cwick_json_get((json_value *)hdr, name);
    if (!t) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                        "model-mtp.safetensors has no %s", name);
        return NULL;
    }
    const char *dt = cwick_json_str(cwick_json_get(t, "dtype"));
    if (!dt || strcmp(dt, "BF16") != 0) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "%s is %s, not BF16", name,
                        dt ? dt : "(no dtype)");
        return NULL;
    }
    json_value *sh = cwick_json_get(t, "shape");
    json_value *of = cwick_json_get(t, "data_offsets");
    if (!sh || !of || cwick_json_array_len(of) != 2 ||
        cwick_json_array_len(sh) < 1 || cwick_json_array_len(sh) > 2) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "%s has a malformed header",
                        name);
        return NULL;
    }
    const int64_t d0 = cwick_json_int(cwick_json_array_at(sh, 0), 0);
    const int64_t d1 = cwick_json_array_len(sh) > 1
                           ? cwick_json_int(cwick_json_array_at(sh, 1), 0)
                           : 0;
    const int64_t lo = cwick_json_int(cwick_json_array_at(of, 0), -1);
    const int64_t hi = cwick_json_int(cwick_json_array_at(of, 1), -1);
    const int64_t n_el = d0 * (d1 > 0 ? d1 : 1);
    if (d0 <= 0 || lo < 0 || hi <= lo || hi - lo != n_el * 2 ||
        (uint64_t)(8 + hlen) + (uint64_t)hi > (uint64_t)len) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                        "%s claims [%lld, %lld] at %lld..%lld, which does not "
                        "fit the file", name, (long long)d0, (long long)d1,
                        (long long)lo, (long long)hi);
        return NULL;
    }
    *rows = d0;
    *cols = d1;
    return (const uint8_t *)base + 8 + hlen + lo;
}

static int mtp_open(q35_model *m, const char *root, cwick_error *err) {
    if (!q35_env_flag("CWICK_Q35_MTP", 1)) return 0;
    char path[1152];
    cwick_path_join(path, sizeof(path), root, "model-mtp.safetensors");
    if (!cwick_file_exists(path)) {
        CWICK_INFO("init", "qwen3_5: no model-mtp.safetensors — drafts come "
                   "from prompt lookup only");
        return 0;
    }
    size_t len = 0;
    const void *base = cwick_mmap_file(path, &len);
    if (!base) {
        cwick_error_set(err, CWICK_ERR_IO, "cannot map %s", path);
        return -1;
    }
    uint64_t hlen = 0;
    if (len < 9) {
        cwick_error_set(err, CWICK_ERR_PARSE, "%s is too short", path);
        goto bad;
    }
    memcpy(&hlen, base, sizeof(hlen));
    if (hlen == 0 || hlen > len - 8) {
        cwick_error_set(err, CWICK_ERR_PARSE, "%s has a %llu-byte header in a "
                        "%zu-byte file", path, (unsigned long long)hlen, len);
        goto bad;
    }
    json_value *hdr = NULL;
    {
        /* The parser wants a NUL-terminated buffer; the mapping is not one. */
        char *text = cwick_xmalloc((size_t)hlen + 1);
        memcpy(text, (const uint8_t *)base + 8, (size_t)hlen);
        text[hlen] = '\0';
        hdr = cwick_json_parse(text, (size_t)hlen, err);
        free(text);
        if (!hdr) goto bad;
    }

    const q35_config *c = &m->cfg;
    const int64_t D = c->dim;
    const int64_t kvw = (int64_t)c->n_kv_heads * c->head_dim;
    const int64_t qwidth = (int64_t)c->n_heads * c->head_dim *
                           (1 + c->attn_gate);
    const int64_t qdim = (int64_t)c->n_heads * c->head_dim;
    const size_t sh = (size_t)hlen;
    int ok = 1;

#define MTP_MAT(field, nm, r, cc)                                             \
    do {                                                                      \
        int64_t rr_ = 0, cc_ = 0;                                             \
        m->mtp.field = (const uint16_t *)mtp_tensor(hdr, base, len, sh, nm,    \
                                                    &rr_, &cc_, err);         \
        if (!m->mtp.field) { ok = 0; break; }                                 \
        if (rr_ != (r) || cc_ != (cc)) {                                      \
            cwick_error_set(err, CWICK_ERR_CONTRACT,                          \
                            "%s is [%lld, %lld], expected [%lld, %lld]", nm,  \
                            (long long)rr_, (long long)cc_, (long long)(r),   \
                            (long long)(cc));                                 \
            ok = 0;                                                           \
            break;                                                            \
        }                                                                     \
    } while (0)

#define MTP_VEC(field, nm, n)                                                 \
    do {                                                                      \
        int64_t rr_ = 0, cc_ = 0;                                             \
        const void *p_ = mtp_tensor(hdr, base, len, sh, nm, &rr_, &cc_, err);  \
        if (!p_) { ok = 0; break; }                                            \
        if (cc_ != 0 || rr_ != (n)) {                                          \
            cwick_error_set(err, CWICK_ERR_CONTRACT,                          \
                            "%s holds %lld values, expected %lld", nm,        \
                            (long long)(cc_ ? cc_ : rr_), (long long)(n));    \
            ok = 0;                                                           \
            break;                                                            \
        }                                                                     \
        m->mtp.field = (float *)f32_from_bf16((const uint16_t *)p_,           \
                                                (int64_t)(n));                \
    } while (0)

    do {
        MTP_MAT(fc, "mtp.fc.weight", D, 2 * D);
        MTP_MAT(wq, "mtp.layers.0.self_attn.q_proj.weight", qwidth, D);
        MTP_MAT(wk, "mtp.layers.0.self_attn.k_proj.weight", kvw, D);
        MTP_MAT(wv, "mtp.layers.0.self_attn.v_proj.weight", kvw, D);
        MTP_MAT(wo, "mtp.layers.0.self_attn.o_proj.weight", D, qdim);
        MTP_MAT(gate, "mtp.layers.0.mlp.gate_proj.weight", c->inter_dim, D);
        MTP_MAT(up, "mtp.layers.0.mlp.up_proj.weight", c->inter_dim, D);
        MTP_MAT(down, "mtp.layers.0.mlp.down_proj.weight", D, c->inter_dim);
        MTP_VEC(pre_norm_emb, "mtp.pre_fc_norm_embedding.weight", D);
        MTP_VEC(pre_norm_hid, "mtp.pre_fc_norm_hidden.weight", D);
        MTP_VEC(in_norm, "mtp.layers.0.input_layernorm.weight", D);
        MTP_VEC(post_norm, "mtp.layers.0.post_attention_layernorm.weight", D);
        MTP_VEC(out_norm, "mtp.norm.weight", D);
        MTP_VEC(q_norm, "mtp.layers.0.self_attn.q_norm.weight", c->head_dim);
        MTP_VEC(k_norm, "mtp.layers.0.self_attn.k_norm.weight", c->head_dim);
    } while (0);
#undef MTP_MAT
#undef MTP_VEC
    cwick_json_free(hdr);
    if (!ok) goto bad;

    m->mtp_map = (void *)base;
    m->mtp_map_len = len;
    /* A window, not the whole context: an agent turn is 31K tokens and this
     * cache would be 1 GiB at that length, paid out of the same budget as the
     * text model's KV. Nothing in a one-token draft needs 128K of *its own*
     * keys -- the fusion term already carries the whole prefix through h_t. */
    {
        const char *e = getenv("CWICK_Q35_MTP_WIN");
        int64_t w = e && *e ? atoll(e) : 512;
        if (w < (int64_t)Q35_GK + 1) w = (int64_t)Q35_GK + 1;
        if (w > (int64_t)c->max_seq_len) w = (int64_t)c->max_seq_len;
        m->mtp_win = (int)w;
    }
    m->mtp_kcache = xcalloc((size_t)m->mtp_win, (size_t)kvw * sizeof(float));
    m->mtp_vcache = xcalloc((size_t)m->mtp_win, (size_t)kvw * sizeof(float));
    m->mtp_n = 0;
    m->mtp_lo = 0;
    m->mtp_base = 0;
    m->mtp_ok = 1;
    CWICK_INFO("init", "qwen3_5: MTP draft head bound — %.0f MiB of weights, "
               "%d-row window (%.0f MiB); drafts chain through the shared "
               "vocabulary head",
               (double)(2 * D * D + qwidth * D + 2 * kvw * D + D * qdim +
                        2 * c->inter_dim * D + D * c->inter_dim) * 2.0 /
                   1048576.0,
               m->mtp_win,
               2.0 * (double)m->mtp_win * (double)kvw * 4.0 / 1048576.0);
    return 1;

bad:
    free(m->mtp.pre_norm_emb); free(m->mtp.pre_norm_hid);
    free(m->mtp.in_norm); free(m->mtp.post_norm); free(m->mtp.out_norm);
    free(m->mtp.q_norm); free(m->mtp.k_norm);
    memset(&m->mtp, 0, sizeof(m->mtp));
    cwick_munmap(base, len);
    return -1;
}

int q35_model_open(q35_model *m, const char *root, int max_seq,
                   cwick_error *err) {
    memset(m, 0, sizeof(*m));
    if (read_config(&m->cfg, root, err) != 0) return -1;
    if (max_seq > 0 && max_seq < m->cfg.max_seq_len)
        m->cfg.max_seq_len = max_seq;

    if (cwick_safetensors_open(root, &m->saf, err) != CWICK_OK) return -1;

    const q35_config *c = &m->cfg;
    char name[256];
    snprintf(name, sizeof(name), "model.language_model.embed_tokens.weight");
    m->embed = bind_mat(m, name, c->vocab_size, c->dim, 1, err);
    if (!m->embed) goto fail;
    snprintf(name, sizeof(name), "lm_head.weight");
    m->head = bind_mat(m, name, c->vocab_size, c->dim, 1, err);
    if (!m->head) goto fail;
    snprintf(name, sizeof(name), "model.language_model.norm.weight");
    m->final_norm = bind_vec(m, name, c->dim, err);
    if (!m->final_norm) goto fail;

    for (int l = 0; l < c->n_layers; l++)
        if (bind_layer(m, l, err) != 0) goto fail;

    /* ---- full residency ----
     *
     * Nothing in this checkpoint streams: every weight is read by every
     * position, so the only question is whether the bytes live in core or wait
     * on the SSD. Left lazy, the first forward pass takes one minor fault per
     * page over all of it -- ~1.2 M faults for 18 GiB, which is 1-2 s of
     * plumbing charged to the first prefill of the first request, plus whatever
     * the VM decides to do with pages it thinks belong to the buffer cache.
     *
     * Doing it here instead makes the cost a startup cost, parallel and paid
     * once, and lets the pages be wired so the compressor cannot take them
     * back mid-decode. Both halves matter: reading 18 GiB per token is the
     * whole cost of a decode step, and a decompression stall lands inside it. */
    {
        uint64_t weights = cwick_safetensors_bytes_under(&m->saf, NULL);
        uint64_t state = q35_state_bytes(c);
        uint64_t host = cwick_host_memory_bytes();
        double frac = CWICK_MEM_DEFAULT_FRACTION;
        uint64_t limit = 0;
        cwick_memory_policy(&frac, &limit);
        uint64_t ceiling = host ? (uint64_t)((double)host * frac) : 0;
        if (limit && (!ceiling || limit < ceiling)) ceiling = limit;

        int want = q35_env_flag("CWICK_Q35_RESIDENT",
                                q35_resident_wanted(weights, state, host,
                                                    ceiling));
        if (want) {
            uint64_t t0 = cwick_prof_ns();
            uint64_t got = cwick_safetensors_make_resident(&m->saf, NULL,
                                                          cwick_memory_noswap());
            double ms = (double)(cwick_prof_ns() - t0) / 1e6;
            if (got) {
                m->resident = 1;
                m->resident_bytes = got;
                CWICK_INFO("init", "qwen3_5 weights resident: %.2f GiB %s — "
                           "%.0f ms (%.1f GB/s); state budget %.2f GiB",
                           cwick_gib(got),
                           m->saf.pinned_wired ? "faulted in and wired"
                                               : "faulted in (not wired)",
                           ms, ms > 0.0 ? (double)got / (ms * 1e6) : 0.0,
                           cwick_gib(state));
            } else {
                CWICK_INFO("init", "qwen3_5 residency: nothing to fault in "
                           "(%.2f GiB of checkpoint bytes reported)",
                           cwick_gib(weights));
            }
        } else {
            CWICK_INFO("init", "qwen3_5 residency: left to the page cache — "
                       "%.2f GiB of weights + %.2f GiB of state against %.2f "
                       "GiB host / %.2f GiB ceiling", cwick_gib(weights),
                       cwick_gib(state), cwick_gib(host), cwick_gib(ceiling));
        }
    }

    build_rope(m);

    /* ---- caches ---- */
    m->kcache = xcalloc((size_t)c->n_layers, sizeof(float *));
    m->vcache = xcalloc((size_t)c->n_layers, sizeof(float *));
    m->conv_state = xcalloc((size_t)c->n_layers, sizeof(float *));
    m->ssm_state = xcalloc((size_t)c->n_layers, sizeof(float *));
    {
        for (int l = 0; l < c->n_layers; l++) {
            if (c->is_linear[l]) {
                m->conv_state[l] = xcalloc(Q35_STATE_SLOTS,
                                           q35_conv_layer_bytes(c));
                m->ssm_state[l] = xcalloc(Q35_STATE_SLOTS,
                                          q35_ssm_layer_bytes(c));
            } else {
                m->kcache[l] = xcalloc(1, q35_kv_layer_bytes(c));
                m->vcache[l] = xcalloc(1, q35_kv_layer_bytes(c));
            }
        }
    }

    /* ---- scratch ---- */
    {
        size_t chunk = Q35_CHUNK;
        int64_t key_dim = (int64_t)c->lin_k_heads * c->lin_k_dim;
        int64_t val_dim = (int64_t)c->lin_v_heads * c->lin_v_dim;
        int64_t qkv_dim = 2 * key_dim + val_dim;
        int64_t qwidth = (int64_t)c->n_heads * c->head_dim * (1 + c->attn_gate);
        int64_t mixw = qkv_dim > qwidth ? qkv_dim : qwidth;
        int64_t gatew = val_dim > c->inter_dim ? val_dim : c->inter_dim;
        size_t f = sizeof(float);
        m->x = cwick_xmalloc(chunk * (size_t)c->dim * f);
        m->xn = cwick_xmalloc(chunk * (size_t)c->dim * f);
        m->mix = cwick_xmalloc(chunk * (size_t)mixw * f);
        m->gate = cwick_xmalloc(chunk * (size_t)gatew * f);
        m->up = cwick_xmalloc(chunk * (size_t)gatew * f);
        m->down = cwick_xmalloc(chunk * (size_t)c->dim * f);
        m->att = cwick_xmalloc(chunk * (size_t)c->n_heads * (size_t)c->head_dim * f);
        m->qb = cwick_xmalloc(chunk * (size_t)c->n_heads * (size_t)c->head_dim * f);
        m->kb = cwick_xmalloc(chunk * (size_t)c->n_kv_heads * (size_t)c->head_dim * f);
        m->vb = cwick_xmalloc(chunk * (size_t)c->n_kv_heads * (size_t)c->head_dim * f);
        m->lin_o = cwick_xmalloc(chunk * (size_t)val_dim * f);
        m->gdec = cwick_xmalloc(chunk * (size_t)c->lin_v_heads * f);
        m->beta = cwick_xmalloc(chunk * (size_t)c->lin_v_heads * f);
        m->logits = cwick_xmalloc((size_t)c->vocab_size * f);
        /* The score matrix of a query tile spans every key that tile can see,
         * so this one scratch buffer grows with the sequence: 64 MiB at 128K,
         * against the 6 GiB of f32 cache it reads. */
        m->attn_s_cap = (size_t)Q35_ATTN_Q * (size_t)c->max_seq_len;
        m->attn_s = cwick_xmalloc(m->attn_s_cap * f);
        m->dec_out = cwick_xmalloc((size_t)Q35_DEC_SLICES *
                                   (size_t)c->n_heads * (size_t)c->head_dim *
                                   f);
    }

    /* Wire the runtime state too, but only when the weights themselves were
     * wired: the two decisions answer the same question (does this model own
     * its memory or does it rent it), and half a guarantee is not one. With
     * --allow-swap the user asked for the renting kind, so the state stays
     * reclaimable as well. */
    if (m->resident && m->saf.pinned_wired) m->state_locked_bytes = q35_lock_state(m);

    m->loaded = 1;
    CWICK_INFO("init", "qwen3_5 ready: dim=%d layers=%d (%d linear + %d full) "
               "heads=%d/%d head_dim=%d rotary=%d vocab=%d context=%d "
               "resident=%s %.2f GiB (state %.2f GiB)",
               c->dim, c->n_layers, c->n_lin_layers, c->n_full_layers,
               c->n_heads, c->n_kv_heads, c->head_dim, c->rotary_dim,
               c->vocab_size, c->max_seq_len,
               !m->resident ? "no"
                            : (m->saf.pinned_wired ? "yes, wired" : "yes, faults in"),
               cwick_gib(m->resident_bytes),
               cwick_gib(m->state_locked_bytes));

    /* The draft head is an optimisation, so a checkpoint that ships a broken
     * one still serves: it just drafts by prompt lookup instead. It is bound
     * before the GPU comes up, because the GPU adopts its caches. */
    if (mtp_open(m, root, err) < 0) {
        CWICK_INFO("init", "qwen3_5: MTP draft head rejected (%s); drafts fall "
                   "back to prompt lookup", err ? err->message : "unknown");
        cwick_error_clear(err);
    }

    /* Best effort: hand one-position steps to the GPU when the caller asked
     * for it. A refusal leaves the CPU path exactly as it was. */
    if (q35_gpu_available()) q35_gpu_open(m);
    return 0;

fail:
    q35_model_close(m);
    return -1;
}

void q35_model_close(q35_model *m) {
    if (!m) return;
    /* The GPU views are no-copy wrappers over these buffers, so they have to
     * go before the memory they point at does. */
    q35_gpu_close();
    const q35_config *c = &m->cfg;
    for (int l = 0; l < c->n_layers && l < Q35_MAX_LAYERS; l++) {
        free(m->layers[l].in_norm);
        free(m->layers[l].post_norm);
        free(m->layers[l].lin.a_log);
        free(m->layers[l].lin.dt_bias);
        free(m->layers[l].lin.norm);
        free(m->layers[l].lin.conv);
        free(m->layers[l].attn.q_norm);
        free(m->layers[l].attn.k_norm);
        /* Unlock before free: munlock of a range that is about to be returned
         * to the allocator is what keeps the next model from inheriting a
         * wired-page count it cannot account for. */
        if (m->state_locked_bytes) {
            if (m->kcache && m->kcache[l])
                cwick_mem_unlock_range(m->kcache[l], q35_kv_layer_bytes(c));
            if (m->vcache && m->vcache[l])
                cwick_mem_unlock_range(m->vcache[l], q35_kv_layer_bytes(c));
            if (m->conv_state && m->conv_state[l])
                cwick_mem_unlock_range(m->conv_state[l],
                                       q35_conv_layer_bytes(c) * Q35_STATE_SLOTS);
            if (m->ssm_state && m->ssm_state[l])
                cwick_mem_unlock_range(m->ssm_state[l],
                                       q35_ssm_layer_bytes(c) * Q35_STATE_SLOTS);
        }
        if (m->kcache) free(m->kcache[l]);
        if (m->vcache) free(m->vcache[l]);
        if (m->conv_state) free(m->conv_state[l]);
        if (m->ssm_state) free(m->ssm_state[l]);
    }
    m->state_locked_bytes = 0;
    free(m->kcache);
    free(m->vcache);
    free(m->conv_state);
    free(m->ssm_state);
    free(m->final_norm);
    free(m->rope_cos);
    free(m->rope_sin);
    free(m->cfg.is_linear);
    free(m->x); free(m->xn); free(m->mix); free(m->gate); free(m->up);
    free(m->down); free(m->att); free(m->qb); free(m->kb); free(m->vb);
    free(m->lin_o); free(m->gdec); free(m->beta);
    free(m->logits);
    free(m->attn_s); free(m->dec_out);
    free(m->mtp.pre_norm_emb); free(m->mtp.pre_norm_hid);
    free(m->mtp.in_norm); free(m->mtp.post_norm); free(m->mtp.out_norm);
    free(m->mtp.q_norm); free(m->mtp.k_norm);
    memset(&m->mtp, 0, sizeof(m->mtp));
    free(m->mtp_kcache); free(m->mtp_vcache);
    m->mtp_kcache = m->mtp_vcache = NULL;
    m->mtp_ok = 0;
    m->mtp_n = m->mtp_lo = m->mtp_base = 0;
    m->mtp_win = 0;
    if (m->mtp_map) cwick_munmap(m->mtp_map, m->mtp_map_len);
    m->mtp_map = NULL;
    m->mtp_map_len = 0;
    m->cfg.is_linear = NULL;
    if (m->saf.root) cwick_safetensors_close(&m->saf);
    m->loaded = 0;
}

uint64_t q35_cache_bytes(const q35_model *m) {
    if (!m) return 0;
    const q35_config *c = &m->cfg;
    uint64_t kv = (uint64_t)c->n_full_layers * 2 *
                  (uint64_t)c->n_kv_heads * (uint64_t)c->head_dim *
                  (uint64_t)c->max_seq_len * sizeof(float);
    uint64_t ssm = (uint64_t)c->n_lin_layers * (uint64_t)c->lin_v_heads *
                   (uint64_t)c->lin_k_dim * (uint64_t)c->lin_v_dim * 4;
    uint64_t mtp = m->mtp_ok
                       ? 2 * (uint64_t)m->mtp_win *
                             (uint64_t)c->n_kv_heads * (uint64_t)c->head_dim *
                             sizeof(float)
                       : 0;
    return kv + ssm + mtp;
}

/* ------------------------------------------------------------------ */
/* MTP drafter, the parts that are not GPU work                         */
/* ------------------------------------------------------------------ */

int q35_mtp_available(const q35_model *m) {
    return m && m->mtp_ok && q35_gpu_available();
}

/* ------------------------------------------------------------------ */
/* forward                                                             */
/* ------------------------------------------------------------------ */

static inline uint16_t f2bf(float f) {
    union { uint32_t u; float f; } v;
    v.f = f;
    uint32_t r = ((v.u >> 16) & 1) + 0x7FFF;
    return (uint16_t)((v.u + r) >> 16);
}


/* ---- Gated DeltaNet recurrence, one value head per work item ---------- */

typedef struct {
    q35_model *m;
    const q35_layer *L;
    int l, n;
    int nh, head_k, head_v, key_dim, val_dim, qkv_dim;
} lin_rec_arg;

/* Value heads [vh0, vh1), each walking the whole block.
 *
 * The state S is [value][key] -- transposed on purpose, so all three of its
 * uses read contiguously -- and belongs to exactly one head, which is what
 * makes this parallel: the scan is sequential over positions, but nothing a
 * head writes is read by another. */
static void lin_rec_heads(void *argp, int64_t vh0, int64_t vh1) {
    lin_rec_arg *a = (lin_rec_arg *)argp;
    q35_model *m = a->m;
    const q35_layer *L = a->L;
    const int l = a->l, n = a->n;
    const int nh = a->nh, head_k = a->head_k, head_v = a->head_v;
    const int key_dim = a->key_dim, val_dim = a->val_dim, qkv_dim = a->qkv_dim;
    const float rms_eps = m->cfg.rms_eps;

    static int gdn_dbg = -1;
    if (gdn_dbg < 0) {
        const char *e = getenv("CWICK_Q35_DEBUG");
        gdn_dbg = (e && *e && *e != '0') ? 1 : 0;
    }
    /* One transient vector per worker, not per head: head_v is 128 here, and a
     * shared buffer would be the one thing two threads could collide on. */
    float *tmp = cwick_xmalloc((size_t)head_v * sizeof(float));

    for (int64_t vh = vh0; vh < vh1; vh++) {
        const int kh = (int)(vh / Q35_LIN_KV_GROUP);
        const float a_log = L->lin.a_log[vh];
        const float dt_bias = L->lin.dt_bias[vh];
        float *S = m->ssm_state[l] + (size_t)vh * (size_t)head_v *
                                    (size_t)head_k;   /* [value][key] */
        for (int t = 0; t < n; t++) {
            const float *row = m->up + (size_t)t * qkv_dim;
            const float *qh = row + (size_t)kh * head_k;
            const float *kk = row + key_dim + (size_t)kh * head_k;
            const float *vrow = row + 2 * key_dim + (size_t)vh * head_v;
            const float araw = m->gdec[(size_t)t * nh + vh];
            const float g = -expf(a_log) * softplus_f(araw + dt_bias);
            const float decay = expf(g);
            const float bet = 1.0f / (1.0f + expf(-m->beta[(size_t)t * nh + vh]));
            if (gdn_dbg && l == 0 && vh == 0 && t == 0) {
                float ks = 0.0f, qs = 0.0f;
                for (int i = 0; i < head_k; i++) {
                    ks += kk[i] * kk[i];
                    qs += qh[i] * qh[i];
                }
                CWICK_INFO("q35dbg", "  gdn t=%d decay=%.5f beta=%.4f a=%.4f "
                           "a_log=%.3f dt_bias=%.3f |k|=%.4f |q|=%.4f",
                           t, decay, bet, araw, a_log, dt_bias,
                           sqrtf(ks), sqrtf(qs));
            }
            for (int j = 0; j < head_v; j++) {
                float *sj = S + (size_t)j * head_k;
                /* Decay first, then read: the correction term is
                 * (v - (decayed S)^T k), not (v - S^T k). The reference does
                 * the multiplication before the read, and doing it after
                 * leaves every position's delta scaled by the wrong decay. */
                float dot = 0.0f;
                for (int i = 0; i < head_k; i++) {
                    sj[i] *= decay;
                    dot += sj[i] * kk[i];
                }
                float delta = bet * (vrow[j] - dot);
                float acc = 0.0f;
                for (int i = 0; i < head_k; i++) {
                    sj[i] += kk[i] * delta;
                    acc += sj[i] * qh[i];
                }
                tmp[j] = acc;
            }
            /* RMSNormGated: normalise the head, scale by the learned weight,
             * then multiply by silu of the gate projection. Unlike every other
             * norm in this file the weight here is a plain multiplier -- it is
             * initialised at 1, not at 0, so it must NOT get the `1 +`. */
            float ss = 0.0f;
            for (int j = 0; j < head_v; j++) ss += tmp[j] * tmp[j];
            float rstd = 1.0f / sqrtf(ss / (float)head_v + rms_eps);
            const float *zrow = m->gate + (size_t)t * val_dim +
                                (size_t)vh * head_v;
            float *orow = m->lin_o + (size_t)t * val_dim +
                          (size_t)vh * head_v;
            for (int j = 0; j < head_v; j++)
                orow[j] = tmp[j] * rstd * L->lin.norm[j] * silu_f(zrow[j]);
        }
    }
    free(tmp);
}

/* One Gated DeltaNet layer's mixer, ending in the out_proj result stored in
 * m->down ([n][dim]).
 *
 * The three projections are one GEMM each over the whole block; the recurrence
 * then walks positions because it is a scan. Nothing here is per-token-shaped
 * on purpose -- see the note at the top of the file. */
static void mixer_linear(q35_model *m, int l, int n) {
    const q35_config *c = &m->cfg;
    const q35_layer *L = &m->layers[l];
    const int D = c->dim;
    const int key_dim = c->lin_k_heads * c->lin_k_dim;
    const int val_dim = c->lin_v_heads * c->lin_v_dim;
    const int qkv_dim = 2 * key_dim + val_dim;
    const int nh = c->lin_v_heads;
    const int taps = c->conv_kernel;
    const int head_k = c->lin_k_dim;
    const int head_v = c->lin_v_dim;

    double t_gemm = q35_prof_on() ? q35_prof_ms() : 0.0;
    cwick_gemm_bf16(m->mix, n, D, qkv_dim, m->xn, L->lin.w_qkv);

    cwick_gemm_bf16(m->gate, n, D, val_dim, m->xn, L->lin.w_z);

    float *av = m->gdec;   /* [n][nh], the raw decay input */
    float *bv = m->beta;   /* [n][nh], the raw write input */
    cwick_gemm_bf16(av, n, D, nh, m->xn, L->lin.w_a);
    cwick_gemm_bf16(bv, n, D, nh, m->xn, L->lin.w_b);
    if (q35_prof_on()) prof_gemm_ms += q35_prof_ms() - t_gemm;
    double t_rec = q35_prof_on() ? q35_prof_ms() : 0.0;

    /* ---- depthwise causal convolution, then SiLU ---- */
    float *st = m->conv_state[l];   /* [taps-1][qkv_dim] */
    if (n >= taps - 1) {
        for (int t = 0; t < n; t++) {
            float *y = m->up + (size_t)t * qkv_dim;
            for (int ch = 0; ch < qkv_dim; ch++) {
                float acc = 0.0f;
                for (int k = 0; k < taps; k++) {
                    /* `src` is an absolute position inside the block, so it
                     * indexes m->mix directly -- adding it to a per-position
                     * base would read position t+src instead of src. */
                    int src = t + k - (taps - 1);
                    float v = src >= 0
                                  ? m->mix[(size_t)src * qkv_dim + ch]
                                  : st[(size_t)(src + taps - 1) * qkv_dim + ch];
                    acc += L->lin.conv[(size_t)k * qkv_dim + ch] * v;
                }
                y[ch] = silu_f(acc);
            }
        }
        for (int j = 0; j < taps - 1; j++)
            memcpy(st + (size_t)j * qkv_dim,
                   m->mix + (size_t)(n - (taps - 1) + j) * qkv_dim,
                   (size_t)qkv_dim * sizeof(float));
    } else {
        /* A decode step: fewer new positions than the window is wide. The
         * output must be computed from the OLD window -- shifting the state
         * down first would feed the first tap its own about-to-be-updated
         * value and turn the convolution into a different one. */
        for (int t = 0; t < n; t++) {
            float *y = m->up + (size_t)t * qkv_dim;
            for (int ch = 0; ch < qkv_dim; ch++) {
                float acc = 0.0f;
                for (int k = 0; k < taps; k++) {
                    int src = t + k - (taps - 1);
                    float v = src >= 0
                                  ? m->mix[(size_t)src * qkv_dim + ch]
                                  : st[(size_t)(src + taps - 1) * qkv_dim + ch];
                    acc += L->lin.conv[(size_t)k * qkv_dim + ch] * v;
                }
                y[ch] = silu_f(acc);
            }
        }
        int keep = taps - 1 - n;
        for (int j = 0; j < keep; j++)
            memcpy(st + (size_t)j * qkv_dim, st + (size_t)(j + n) * qkv_dim,
                   (size_t)qkv_dim * sizeof(float));
        for (int t = 0; t < n; t++)
            memcpy(st + (size_t)(keep + t) * qkv_dim,
                   m->mix + (size_t)t * qkv_dim, (size_t)qkv_dim * sizeof(float));
    }

    /* ---- q/k L2 normalisation, per key head ---- */
    /* The reference also scales q by head_dim^-0.5 right after the norm. It
     * cancels out at the gated norm below, but it is part of the defined
     * computation and costs nothing, so it stays. */
    const float qscale = 1.0f / sqrtf((float)head_k);
    for (int t = 0; t < n; t++) {
        float *q = m->up + (size_t)t * qkv_dim;
        for (int h = 0; h < c->lin_k_heads; h++) {
            float *qh = q + (size_t)h * head_k;
            l2_norm_inplace(qh, head_k, c->l2_eps);
            for (int i = 0; i < head_k; i++) qh[i] *= qscale;
            l2_norm_inplace(q + key_dim + (size_t)h * head_k, head_k,
                            c->l2_eps);
        }
    }

    /* ---- the delta rule ----
     *
     * One state matrix per value head, and the heads never read each other, so
     * the scan parallelises over heads even though it cannot parallelise over
     * positions. That is the shape the pool wants: 32 independent jobs, each
     * long enough to be worth waking a core for. Measured on a 176-position
     * block this phase was 35% of prefill and ran on one core; splitting it by
     * head is what makes a block's cost track core count instead of clock
     * speed. */
    lin_rec_arg ra;
    ra.m = m;
    ra.L = L;
    ra.l = l;
    ra.n = n;
    ra.nh = nh;
    ra.head_k = head_k;
    ra.head_v = head_v;
    ra.key_dim = key_dim;
    ra.val_dim = val_dim;
    ra.qkv_dim = qkv_dim;
    cwick_pool_for_items(nh, &ra, lin_rec_heads);

    if (q35_prof_on()) prof_rec_ms += q35_prof_ms() - t_rec;
    double t_out = q35_prof_on() ? q35_prof_ms() : 0.0;
    cwick_gemm_bf16(m->down, n, val_dim, D, m->lin_o, L->lin.w_o);
    if (q35_prof_on()) prof_gemm_ms += q35_prof_ms() - t_out;
}

/* ---- attention: two matrix multiplies, on the engine's own GEMM path ----
 *
 * A score pass and an output pass are both dense GEMMs:
 *
 *     S[nq x keys] = Q[nq x HD] · K[keys x HD]^T
 *     O[nq x HD]   = S[nq x keys] · V[keys x HD]
 *
 * and this tree already knows how to run one: every other projection in the
 * model goes through Accelerate. Hand-writing these two, per query and per key,
 * is what made attention the only term that grows with context. Measured at the
 * real shape and the same MAC count (tools/attnbench, one thread): a tiled bf16
 * dot runs at 22 GMAC/s where cblas runs at 750, and the output pass at 19 vs
 * 477. Reordering the hand kernel (four queries per key row, accumulator in
 * registers) buys 1.4x and 2.3x -- real, and 20x short of the GEMM path.
 *
 * What cblas costs is f32 operands. The cache stays bf16 -- half the memory,
 * and the prefix snapshots are written in that form -- so one block widens the
 * visible rows once per layer into a scratch copy. At 30K positions that is
 * 60 MB read and 120 MB written per layer, about 4 ms, against the ~800 ms the
 * two passes used to take.
 */
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
/* Accelerate brings in <complex.h>, whose `I` is the imaginary unit; this file
 * uses `I` for the MLP width in every layer function. */
#ifdef I
#undef I
#endif
#define Q35_HAVE_CBLAS 1
#else
#define Q35_HAVE_CBLAS 0
#endif

/* ---- decode: one position, every head ----
 *
 * The cache row of a position holds all four kv heads side by side, so a row is
 * 4 KiB that sixteen dots can share (four kv slices x four query heads in each
 * group). A matrix-vector call per head instead reads 1 KiB out of every 4 KiB
 * and ran at 13 GB/s -- 52 ms per step at an 11K context, ten times the hand
 * kernel it replaced. Walking rows and reusing them costs 4 ms.
 */
static inline float q35_dot_f32(const float *a, const float *b, int n) {
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

static inline void q35_axpy_f32(float *o, const float *v, float w, int n) {
    const float32x4_t wv = vdupq_n_f32(w);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(o + i, vfmaq_f32(vld1q_f32(o + i), wv, vld1q_f32(v + i)));
        vst1q_f32(o + i + 4,
                  vfmaq_f32(vld1q_f32(o + i + 4), wv, vld1q_f32(v + i + 4)));
    }
    for (; i < n; i++) o[i] += w * v[i];
}

typedef struct {
    const float *q;      /* [n_heads][HD] */
    const float *k, *v;  /* [keys][kv] */
    float *s;            /* [n_heads][keys] */
    float *o;            /* [n_heads][HD], the attention output rows */
    int64_t keys, kv;
    int HD, kvh_n, group;
    float scale;
} dec_arg;

static void dec_scores(void *argp, int64_t i0, int64_t i1) {
    dec_arg *a = (dec_arg *)argp;
    for (int64_t t = i0; t < i1; t++) {
        const float *row = a->k + t * a->kv;
        for (int g = 0; g < a->kvh_n; g++) {
            const float *ks = row + (size_t)g * a->HD;
            for (int j = 0; j < a->group; j++) {
                const int h = g * a->group + j;
                a->s[(size_t)h * a->keys + t] =
                    q35_dot_f32(a->q + (size_t)h * a->HD, ks, a->HD) * a->scale;
            }
        }
    }
}

/* The output pass reads the same rows as the score pass, so it wants the same
 * shape: a work item is a slice of *rows*, not a kv head. A per-head item reads
 * 1 KiB of every 4 KiB and only four of them exist, which is 13 GB/s -- 79 ms of
 * a 202 ms step at 31K. Slicing by rows reads whole rows and fills the pool;
 * each slice accumulates its own copy of the output, and the copies are summed
 * at the end (16 heads x 256 floats per slice, so the reduction is free). */
typedef struct {
    const float *v;
    const float *s;
    float *buf;          /* [nslice][n_heads][HD] */
    int64_t keys, kv;
    int HD, nh, group, nslice;
} dec_av_arg;

static void dec_out_slices(void *argp, int64_t s0, int64_t s1) {
    dec_av_arg *a = (dec_av_arg *)argp;
    for (int64_t sidx = s0; sidx < s1; sidx++) {
        const int64_t lo = a->keys * sidx / a->nslice;
        const int64_t hi = a->keys * (sidx + 1) / a->nslice;
        float *o = a->buf + (size_t)sidx * (size_t)a->nh * (size_t)a->HD;
        memset(o, 0, (size_t)a->nh * (size_t)a->HD * sizeof(float));
        for (int64_t t = lo; t < hi; t++) {
            const float *row = a->v + t * a->kv;
            for (int h = 0; h < a->nh; h++) {
                const float w = a->s[(size_t)h * a->keys + t];
                if (w == 0.0f) continue;
                q35_axpy_f32(o + (size_t)h * a->HD,
                             row + (size_t)(h / a->group) * a->HD, w, a->HD);
            }
        }
    }
}

typedef struct {
    float *s;
    int64_t keys;
    int n;
} dec_sm_arg;

static void dec_softmax(void *argp, int64_t i0, int64_t i1) {
    dec_sm_arg *a = (dec_sm_arg *)argp;
    for (int64_t h = i0; h < i1; h++)
        cwick_softmax(a->s + (size_t)h * a->keys, a->keys);
}

/* S is row-major [rows][keys]; each row is softmaxed over the keys at or before
 * its own position, and the tail is set to -1e30 so exp() underflows to exactly
 * zero. That is the same arithmetic as skipping the tail, and it lets the GEMM
 * compute a rectangle instead of a triangle. */
typedef struct {
    float *s;
    int64_t keys, pos, t0;
} softmax_arg;

static void softmax_items(void *argp, int64_t i0, int64_t i1) {
    softmax_arg *a = (softmax_arg *)argp;
    for (int64_t i = i0; i < i1; i++) {
        const int64_t lim = (int64_t)a->pos + a->t0 + i + 1;
        float *row = a->s + (size_t)i * (size_t)a->keys;
        for (int64_t s = lim; s < a->keys; s++) row[s] = -1e30f;
        cwick_softmax(row, a->keys);
    }
}

/* bf16 row -> f32. bf16 is the top half of an f32, so this is a shift, not a
 * conversion. */
/* bf16 row -> f32. bf16 is the top half of an f32, so this is a shift, not a
 * conversion. */
static void q35_widen_rows(float *dst, const uint16_t *src, size_t n) {
    size_t i = 0;
#if defined(__aarch64__)
    for (; i + 8 <= n; i += 8) {
        uint16x8_t u = vld1q_u16(src + i);
        vst1q_f32(dst + i,
                  vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(u), 16)));
        vst1q_f32(dst + i + 4,
                  vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(u), 16)));
    }
#endif
    for (; i < n; i++) dst[i] = bf(src + i);
}

/* f32 row -> bf16, round to nearest even, for the prefix snapshot. */
static void q35_narrow_rows(uint16_t *dst, const float *src, size_t n) {
    size_t i = 0;
#if defined(__aarch64__)
    const uint32x4_t bias = vdupq_n_u32(0x7fffu);
    const uint32x4_t lsb = vdupq_n_u32(1);
    for (; i + 4 <= n; i += 4) {
        uint32x4_t u = vreinterpretq_u32_f32(vld1q_f32(src + i));
        /* The rounding addend is 0x7fff plus the low bit of the result, which
         * is what "round half to even" means in this representation. */
        uint32x4_t r = vaddq_u32(u, vaddq_u32(bias, vandq_u32(vshrq_n_u32(u, 16),
                                                             lsb)));
        r = vreinterpretq_u32_f32(vreinterpretq_f32_u32(r));
        vst1_u16(dst + i,
                 vmovn_u32(vshrq_n_u32(r, 16)));
    }
#endif
    for (; i < n; i++) dst[i] = f2bf(src[i]);
}

#if Q35_HAVE_CBLAS
#define Q35_MM(m, ncol, kk, alpha, a, lda, b, ldb, btrans, beta, c_, ldc)     \
    cblas_sgemm(CblasRowMajor, CblasNoTrans,                                   \
                (btrans) ? CblasTrans : CblasNoTrans, (int)(m), (int)(ncol),   \
                (int)(kk), (alpha), (a), (int)(lda), (b), (int)(ldb), (beta),  \
                (c_), (int)(ldc))
#else
/* Fallback for a build without Accelerate: same shape, obviously slow, and not
 * the path this engine ships on. */
static void q35_mm(int64_t m, int64_t ncol, int64_t kk, float alpha,
                   const float *a, int64_t lda, const float *b, int64_t ldb,
                   int btrans, float beta, float *c_, int64_t ldc) {
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < ncol; j++) {
            float acc = 0.0f;
            for (int64_t t = 0; t < kk; t++)
                acc += a[(size_t)i * (size_t)lda + (size_t)t] *
                       (btrans ? b[(size_t)j * (size_t)ldb + (size_t)t]
                               : b[(size_t)t * (size_t)ldb + (size_t)j]);
            float *cp = &c_[(size_t)i * (size_t)ldc + (size_t)j];
            *cp = beta ? beta * *cp + alpha * acc : alpha * acc;
        }
}
#define Q35_MM(m, ncol, kk, alpha, a, lda, b, ldb, btrans, beta, c_, ldc)     \
    q35_mm((m), (ncol), (kk), (alpha), (a), (lda), (b), (ldb), (btrans),       \
           (beta), (c_), (ldc))
#endif

/* One full-attention layer's mixer: GQA with a per-head Q/K norm, a partial
 * rotation, and an output gate taken from the second half of q_proj. */
static void mixer_attn(q35_model *m, int l, int n, int64_t pos) {
    const q35_config *c = &m->cfg;
    const q35_layer *L = &m->layers[l];
    const int D = c->dim;
    const int H = c->n_heads, KV = c->n_kv_heads, HD = c->head_dim;
    const int qwidth = H * HD * (1 + c->attn_gate);
    const int kv = KV * HD;
    const int qdim = H * HD;
    const int group = H / KV;
    int pairs = c->rotary_dim / 2;

    double t_gemm = q35_prof_on() ? q35_prof_ms() : 0.0;
    cwick_gemm_bf16(m->mix, n, D, qwidth, m->xn, L->attn.wq);
    cwick_gemm_bf16(m->kb, n, D, kv, m->xn, L->attn.wk);
    cwick_gemm_bf16(m->vb, n, D, kv, m->xn, L->attn.wv);
    if (q35_prof_on()) prof_gemm_ms += q35_prof_ms() - t_gemm;

    float *kc = m->kcache[l];
    float *vc = m->vcache[l];

    for (int t = 0; t < n; t++) {
        const int64_t p = pos + t;
        const float *cos_p = m->rope_cos + (size_t)p * pairs;
        const float *sin_p = m->rope_sin + (size_t)p * pairs;
        float *qg = m->mix + (size_t)t * qwidth;
        /* q_proj is laid out per head: [q(HD) | gate(HD)] repeated n_heads
         * times, because the reference views it as (..., n_heads, 2*HD) and
         * chunks that last axis. Reading the whole first half as q instead
         * would mix one head's gate into the next head's query. */
        for (int h = 0; h < H; h++) {
            float *qh = qg + (size_t)h * HD * 2;
            rms_norm_w(qh, qh, HD, L->attn.q_norm, c->rms_eps);
            rope_apply(qh, cos_p, sin_p, c->rotary_dim);
            memcpy(m->qb + (size_t)t * qdim + (size_t)h * HD, qh,
                   (size_t)HD * sizeof(float));
        }
        float *kh = m->kb + (size_t)t * kv;
        for (int h = 0; h < KV; h++) {
            float *row = kh + (size_t)h * HD;
            rms_norm_w(row, row, HD, L->attn.k_norm, c->rms_eps);
            rope_apply(row, cos_p, sin_p, c->rotary_dim);
        }
        memcpy(kc + (size_t)p * kv, kh, (size_t)kv * sizeof(float));
        memcpy(vc + (size_t)p * kv, m->vb + (size_t)t * kv,
               (size_t)kv * sizeof(float));
    }

    double t_attn = q35_prof_on() ? q35_prof_ms() : 0.0;
    const float scale = 1.0f / sqrtf((float)HD);
    if (n == 1) {
        /* Decode. Every head reads the same one position, so this is the one
         * case where a tile is the wrong shape: see dec_scores. */
        dec_arg da;
        dec_arg *a = &da;
        a->q = m->qb;
        a->k = kc;
        a->v = vc;
        a->s = m->attn_s;
        a->o = m->att;
        a->keys = (int64_t)pos + 1;
        a->kv = kv;
        a->HD = HD;
        a->kvh_n = KV;
        a->group = group;
        a->scale = scale;
        double tp0 = q35_prof_on() ? q35_prof_ms() : 0.0;
        cwick_pool_for_items(a->keys, a, dec_scores);
        double tp1 = q35_prof_on() ? q35_prof_ms() : 0.0;
        dec_sm_arg sa;
        sa.s = m->attn_s;
        sa.keys = a->keys;
        sa.n = H;
        cwick_pool_for_items(H, &sa, dec_softmax);
        double tp2 = q35_prof_on() ? q35_prof_ms() : 0.0;
        const int nslice = Q35_DEC_SLICES;
        dec_av_arg va;
        va.v = vc;
        va.s = m->attn_s;
        va.buf = m->dec_out;
        va.keys = a->keys;
        va.kv = kv;
        va.HD = HD;
        va.nh = H;
        va.group = group;
        va.nslice = nslice;
        cwick_pool_for_items(nslice, &va, dec_out_slices);
        for (int h = 0; h < H; h++) {
            float *dst = m->att + (size_t)h * HD;
            memcpy(dst, m->dec_out + (size_t)h * HD,
                   (size_t)HD * sizeof(float));
            for (int sl = 1; sl < nslice; sl++)
                q35_axpy_f32(dst, m->dec_out +
                                      ((size_t)sl * (size_t)H + (size_t)h) * HD,
                             1.0f, HD);
        }
        if (tp0 > 0.0) {
            double tp3 = q35_prof_ms();
            atomic_fetch_add_explicit(&prof_qk_ms, tp1 - tp0,
                                      memory_order_relaxed);
            atomic_fetch_add_explicit(&prof_sm_ms, tp2 - tp1,
                                      memory_order_relaxed);
            atomic_fetch_add_explicit(&prof_av_ms, tp3 - tp2,
                                      memory_order_relaxed);
        }
        goto gate;
    }
    for (int h = 0; h < H; h++) {
        const int kvh = h / group;
        for (int t0 = 0; t0 < n; t0 += Q35_ATTN_Q) {
            const int nq = n - t0 < Q35_ATTN_Q ? n - t0 : Q35_ATTN_Q;
            /* The tile's queries can see everything up to the last of them;
             * the rectangle below that diagonal is masked in softmax_items. */
            const int64_t kk = (int64_t)pos + t0 + nq;
            float *S = m->attn_s;
            const float *q = m->qb + (size_t)t0 * (size_t)qdim + (size_t)h * HD;
            float *o = m->att + (size_t)t0 * (size_t)qdim + (size_t)h * HD;
            double tp0 = q35_prof_on() ? q35_prof_ms() : 0.0;
            Q35_MM(nq, kk, HD, scale, q, qdim,
                   kc + (size_t)kvh * HD, kv, 1, 0.0f, S, kk);
            double tp1 = q35_prof_on() ? q35_prof_ms() : 0.0;
            softmax_arg sa;
            sa.s = S;
            sa.keys = kk;
            sa.pos = pos;
            sa.t0 = t0;
            cwick_pool_for_items(nq, &sa, softmax_items);
            double tp2 = q35_prof_on() ? q35_prof_ms() : 0.0;
            Q35_MM(nq, HD, kk, 1.0f, S, kk, vc + (size_t)kvh * HD, kv, 0,
                   0.0f, o, qdim);
            if (tp0 > 0.0) {
                double tp3 = q35_prof_ms();
                atomic_fetch_add_explicit(&prof_qk_ms, tp1 - tp0,
                                          memory_order_relaxed);
                atomic_fetch_add_explicit(&prof_sm_ms, tp2 - tp1,
                                          memory_order_relaxed);
                atomic_fetch_add_explicit(&prof_av_ms, tp3 - tp2,
                                          memory_order_relaxed);
            }
        }
    }

gate:
    if (c->attn_gate) {
        for (int t = 0; t < n; t++) {
            const float *qg = m->mix + (size_t)t * qwidth;
            float *o = m->att + (size_t)t * qdim;
            for (int h = 0; h < H; h++) {
                const float *g = qg + (size_t)h * HD * 2 + HD;
                float *oh = o + (size_t)h * HD;
                for (int i = 0; i < HD; i++)
                    oh[i] *= 1.0f / (1.0f + expf(-g[i]));
            }
        }
    }

    if (q35_prof_on()) prof_attn_ms += q35_prof_ms() - t_attn;
    double t_out = q35_prof_on() ? q35_prof_ms() : 0.0;
    cwick_gemm_bf16(m->up, n, qdim, D, m->att, L->attn.wo);
    memcpy(m->down, m->up, (size_t)n * (size_t)D * sizeof(float));
    if (q35_prof_on()) prof_gemm_ms += q35_prof_ms() - t_out;
}

static void mlp_forward(q35_model *m, int l, int n) {
    const q35_config *c = &m->cfg;
    const q35_layer *L = &m->layers[l];
    const int D = c->dim, I = c->inter_dim;
    double t0 = q35_prof_on() ? q35_prof_ms() : 0.0;
    cwick_gemm_bf16(m->gate, n, D, I, m->xn, L->w_gate);
    cwick_gemm_bf16(m->up, n, D, I, m->xn, L->w_up);
    if (q35_prof_on()) {
        double t = q35_prof_ms();
        prof_gemm_ms += t - t0;
        t0 = t;
    }
    for (int t = 0; t < n; t++) {
        float *g = m->gate + (size_t)t * I;
        const float *u = m->up + (size_t)t * I;
        for (int i = 0; i < I; i++) g[i] = silu_f(g[i]) * u[i];
    }
    if (q35_prof_on()) {
        double t = q35_prof_ms();
        prof_act_ms += t - t0;
        t0 = t;
    }
    cwick_gemm_bf16(m->down, n, I, D, m->gate, L->w_down);
    if (q35_prof_on()) prof_gemm_ms += q35_prof_ms() - t0;
}

static int forward_chunk(q35_model *m, const int32_t *ids, int n, int64_t pos,
                         float *logits, cwick_error *err) {
    const q35_config *c = &m->cfg;
    const int D = c->dim;

    /* CWICK_Q35_DEBUG=1 prints one line per layer: the mean, the absolute
     * maximum and the first few values of the residual stream after the mixer
     * and after the MLP. Reading a wrong architecture off a generated string
     * is guesswork; reading where the stream stops resembling one is not. */
    static int dbg = -1;
    if (dbg < 0) {
        const char *e = getenv("CWICK_Q35_DEBUG");
        dbg = (e && *e && *e != '0') ? 1 : 0;
    }

    if (dbg) {
        char idbuf[256];
        size_t o = 0;
        for (int t = 0; t < n && o + 12 < sizeof(idbuf); t++)
            o += (size_t)snprintf(idbuf + o, sizeof(idbuf) - o, "%s%d",
                                  t ? "," : "", ids[t]);
        CWICK_INFO("q35dbg", "chunk n=%d pos=%lld ids=[%s]", n,
                   (long long)pos, idbuf);
    }

    for (int t = 0; t < n; t++) {
        int32_t id = ids[t];
        if (id < 0 || id >= c->vocab_size) {
            cwick_error_set(err, CWICK_ERR_CONTRACT,
                            "token id %d is outside the vocabulary (%d)", id,
                            c->vocab_size);
            return -1;
        }
        const uint16_t *row = m->embed + (size_t)id * (size_t)D;
        float *dst = m->x + (size_t)t * D;
        for (int j = 0; j < D; j++) dst[j] = bf(row + j);
    }

    for (int l = 0; l < c->n_layers; l++) {
        const q35_layer *L = &m->layers[l];
        for (int t = 0; t < n; t++)
            rms_norm_w(m->xn + (size_t)t * D, m->x + (size_t)t * D, D,
                       L->in_norm, c->rms_eps);

        if (L->is_linear) mixer_linear(m, l, n);
        else mixer_attn(m, l, n, pos);

        for (int t = 0; t < n; t++) {
            float *x = m->x + (size_t)t * D;
            const float *d = m->down + (size_t)t * D;
            for (int j = 0; j < D; j++) x[j] += d[j];
        }
        if (dbg && l < 4) {
            const float *x = m->x + (size_t)(n - 1) * D;
            double s = 0.0;
            float amax = 0.0f;
            for (int j = 0; j < D; j++) { s += x[j]; if (fabsf(x[j]) > amax) amax = fabsf(x[j]); }
            CWICK_INFO("q35dbg", "L%d %s post-mixer: mean=%.5f absmax=%.4f "
                       "x[0..3]=%.4f %.4f %.4f %.4f",
                       l, L->is_linear ? "lin " : "full", s / D, amax,
                       x[0], x[1], x[2], x[3]);
        }
        for (int t = 0; t < n; t++)
            rms_norm_w(m->xn + (size_t)t * D, m->x + (size_t)t * D, D,
                       L->post_norm, c->rms_eps);
        mlp_forward(m, l, n);
        for (int t = 0; t < n; t++) {
            float *x = m->x + (size_t)t * D;
            const float *d = m->down + (size_t)t * D;
            for (int j = 0; j < D; j++) x[j] += d[j];
        }
        /* One call per layer: a 512-position block takes ~7 s at the start of
         * a prompt and ~72 s at 30K, so reporting only at block boundaries
         * left the live gauges saying "0 / idle" for over a minute at a time. */
        mm_prefill_progress_layer(pos, n, l + 1, c->n_layers);

        if (dbg && l < 4) {
            const float *x = m->x + (size_t)(n - 1) * D;
            double s = 0.0;
            float amax = 0.0f;
            for (int j = 0; j < D; j++) { s += x[j]; if (fabsf(x[j]) > amax) amax = fabsf(x[j]); }
            CWICK_INFO("q35dbg", "L%d %s post-mlp  : mean=%.5f absmax=%.4f "
                       "x[0..3]=%.4f %.4f %.4f %.4f",
                       l, L->is_linear ? "lin " : "full", s / D, amax,
                       x[0], x[1], x[2], x[3]);
        }
    }

    rms_norm_w(m->xn, m->x + (size_t)(n - 1) * D, D, m->final_norm, c->rms_eps);
    cwick_gemv_bf16(logits, D, c->vocab_size, m->xn, m->head);

    if (q35_prof_on()) {
        double tot = prof_gemm_ms + prof_rec_ms + prof_attn_ms + prof_act_ms;
        double qk = atomic_load_explicit(&prof_qk_ms, memory_order_relaxed);
        double sm = atomic_load_explicit(&prof_sm_ms, memory_order_relaxed);
        double av = atomic_load_explicit(&prof_av_ms, memory_order_relaxed);
        double inner = qk + sm + av;
        CWICK_INFO("q35prof", "chunk n=%d total %.0f ms — gemm %.0f (%.0f%%), "
                   "delta-rule %.0f (%.0f%%), attention %.0f (%.0f%%), "
                   "activation %.0f (%.0f%%)",
                   n, tot, prof_gemm_ms,
                   tot > 0 ? 100.0 * prof_gemm_ms / tot : 0.0, prof_rec_ms,
                   tot > 0 ? 100.0 * prof_rec_ms / tot : 0.0, prof_attn_ms,
                   tot > 0 ? 100.0 * prof_attn_ms / tot : 0.0, prof_act_ms,
                   tot > 0 ? 100.0 * prof_act_ms / tot : 0.0);
        if (inner > 0.0)
            CWICK_INFO("q35prof", "  attention split (thread-time): qk %.0f "
                       "(%.0f%%), softmax %.0f (%.0f%%), av %.0f (%.0f%%)",
                       qk, 100.0 * qk / inner, sm, 100.0 * sm / inner,
                       av, 100.0 * av / inner);
        prof_gemm_ms = prof_rec_ms = prof_attn_ms = prof_act_ms = 0.0;
        atomic_store_explicit(&prof_qk_ms, 0.0, memory_order_relaxed);
        atomic_store_explicit(&prof_sm_ms, 0.0, memory_order_relaxed);
        atomic_store_explicit(&prof_av_ms, 0.0, memory_order_relaxed);
    }
    if (dbg) {
        int best[5] = {-1, -1, -1, -1, -1};
        float bestv[5] = {-1e30f, -1e30f, -1e30f, -1e30f, -1e30f};
        for (int v = 0; v < c->vocab_size; v++) {
            float x = logits[v];
            if (x <= bestv[4]) continue;
            int j = 4;
            while (j > 0 && bestv[j - 1] < x) {
                bestv[j] = bestv[j - 1];
                best[j] = best[j - 1];
                j--;
            }
            bestv[j] = x;
            best[j] = v;
        }
        CWICK_INFO("q35dbg", "logits top5: %d(%.3f) %d(%.3f) %d(%.3f) "
                   "%d(%.3f) %d(%.3f)", best[0], bestv[0], best[1], bestv[1],
                   best[2], bestv[2], best[3], bestv[3], best[4], bestv[4]);
    }
    return 0;
}

/* An exact f32 copy of the state a step mutates.
 *
 * q35_prefix_pack carries the same set, but it narrows the KV rows to bf16 to
 * keep the prefix cache small. Right for a cache, wrong for a check: a bf16 KV
 * row is a different KV row, worth about 5e-3 at the logits -- the same order
 * as the difference the check exists to detect, which is how a harness ends up
 * reporting its own rounding as an arithmetic bug. So the diagnostic gets its
 * own lossless copy.
 *
 * `rows` is how many positions survive the step; the step cannot write at or
 * beyond that, so nothing there needs saving. Layers are layer-major, conv
 * then ssm for the recurrence layers and keys then values for the others. */
static size_t state_exact_bytes(const q35_model *m, int64_t rows) {
    const q35_config *c = &m->cfg;
    if (rows < 0) rows = 0;
    size_t b = (size_t)c->n_lin_layers *
               (q35_conv_layer_bytes(c) + q35_ssm_layer_bytes(c));
    b += (size_t)c->n_full_layers * 2 * (size_t)rows * q35_kv_row(c) *
         sizeof(float);
    return b;
}

static int state_exact_copy(const q35_model *m, int64_t rows, void *buf,
                            int save) {
    const q35_config *c = &m->cfg;
    const size_t conv = q35_conv_layer_bytes(c);
    const size_t ssm = q35_ssm_layer_bytes(c);
    const size_t kv = (size_t)rows * q35_kv_row(c) * sizeof(float);
    uint8_t *o = (uint8_t *)buf;
    for (int l = 0; l < c->n_layers; l++) {
        if (!c->is_linear[l]) continue;
        if (!m->conv_state || !m->conv_state[l] || !m->ssm_state ||
            !m->ssm_state[l])
            return -1;
        for (int which = 0; which < 2; which++) {
            void *p = which ? (void *)m->ssm_state[l] : (void *)m->conv_state[l];
            const size_t b = which ? ssm : conv;
            if (save) memcpy(o, p, b); else memcpy(p, o, b);
            o += b;
        }
    }
    for (int l = 0; l < c->n_layers; l++) {
        if (c->is_linear[l]) continue;
        if (!m->kcache || !m->kcache[l] || !m->vcache || !m->vcache[l])
            return -1;
        for (int which = 0; which < 2; which++) {
            void *p = which ? (void *)m->vcache[l] : (void *)m->kcache[l];
            if (save) memcpy(o, p, kv); else memcpy(p, o, kv);
            o += kv;
        }
    }
    return 0;
}

/* One decode position on the GPU, optionally checked against the CPU path.
 *
 * The check has to restore the recurrence state between the two runs, because
 * both advance it in place: a DeltaNet layer's state is not a cache that can
 * be truncated, it is a matrix that the step mutates. The save/restore below
 * is exactly that -- the conv window and the ssm state of the 24 linear layers
 * plus the KV rows of the other 8 -- so the second run starts from the same
 * bytes the first run did and the two logits vectors are comparable. Without
 * it the second run would simply be a different step.
 *
 * Enabling this costs one extra CPU step and one ~50 MB snapshot per decode
 * position, so it is a diagnostic (CWICK_Q35_GPU_CHECK=1), not a mode. */
static int gpu_step_checked(q35_model *m, int32_t id, int64_t pos,
                            float *logits, cwick_error *err) {
    static int check = -1;
    if (check < 0) {
        const char *e = getenv("CWICK_Q35_GPU_CHECK");
        check = (e && *e && *e != '0') ? 1 : 0;
    }
    uint8_t *snap = NULL;
    size_t snap_len = 0;
    if (check) {
        snap_len = state_exact_bytes(m, pos + 1);
        snap = cwick_xmalloc(snap_len);
        if (state_exact_copy(m, pos + 1, snap, 1) < 0) {
            free(snap);
            snap = NULL;
        }
    }
    int r = q35_gpu_step(m, id, pos, logits);
    if (r <= 0) {
        if (snap) free(snap);
        return r;   /* 0: declined before writing, -1: failed mid-step */
    }
    if (!snap) return 1;

    float *gpu_logits = cwick_xmalloc((size_t)m->cfg.vocab_size * sizeof(float));
    memcpy(gpu_logits, logits, (size_t)m->cfg.vocab_size * sizeof(float));
    if (state_exact_copy(m, pos + 1, snap, 0) < 0) {
        CWICK_INFO("q35gpu", "state restore failed; the check is skipped");
        free(snap);
        free(gpu_logits);
        return 1;
    }
    free(snap);
    if (forward_chunk(m, &id, 1, pos, logits, err) != 0) {
        free(gpu_logits);
        return 1;   /* the GPU already produced a usable answer */
    }
    double maxd = 0.0, sum = 0.0, mag = 0.0;
    int amax_gpu = 0, amax_cpu = 0;
    for (int v = 0; v < m->cfg.vocab_size; v++) {
        double d = fabs((double)gpu_logits[v] - (double)logits[v]);
        if (d > maxd) maxd = d;
        sum += d;
        mag += fabs((double)gpu_logits[v]);
        if (gpu_logits[v] > gpu_logits[amax_gpu]) amax_gpu = v;
        if (logits[v] > logits[amax_cpu]) amax_cpu = v;
    }
    CWICK_INFO("q35gpu", "check pos=%lld: max|gpu-cpu|=%.5f mean=%.6f "
               "mean|logit|=%.4f argmax gpu=%d cpu=%d %s",
               (long long)pos, maxd, sum / m->cfg.vocab_size,
               mag / m->cfg.vocab_size, amax_gpu, amax_cpu,
               amax_gpu == amax_cpu ? "agree" : "DIFFER");
    free(gpu_logits);
    return 1;
}

/* Fingerprint the recurrent state of the first `nl` layers: the DeltaNet
 * state, the convolution window and the KV cache. All three are host arrays
 * (the GPU adopts them without a copy), so a mismatch can be read directly --
 * which is the only way to tell "row i was computed from the wrong state"
 * apart from "row i was computed wrongly". Only used in check mode.
 *
 * Only the first `rows` KV positions are compared: everything past them is
 * still zero in both runs, so carrying it would cost gigabytes at a long
 * context without adding a single bit of evidence. */
static size_t state_blob_kv(const q35_config *c, int64_t rows) {
    return (size_t)rows * c->n_kv_heads * c->head_dim;
}

static size_t state_blob_bytes(const q35_model *m, int nl, int64_t rows) {
    const q35_config *c = &m->cfg;
    const size_t ssm = (size_t)c->lin_v_heads * c->lin_k_dim * c->lin_v_dim;
    const int taps = c->conv_kernel > 0 ? c->conv_kernel : 4;
    const size_t qkv = (size_t)(2 * c->lin_k_heads * c->lin_k_dim +
                                c->lin_v_heads * c->lin_v_dim);
    const size_t conv = (size_t)(taps - 1) * qkv;
    const size_t kv = state_blob_kv(c, rows);
    size_t at = 0;
    for (int l = 0; l < nl && l < c->n_layers; l++)
        at += m->layers[l].is_linear ? ssm + conv : 2 * kv;
    return at;
}

static size_t state_blob(const q35_model *m, int nl, int64_t rows, float *dst) {
    const q35_config *c = &m->cfg;
    const size_t ssm = (size_t)c->lin_v_heads * c->lin_k_dim * c->lin_v_dim;
    const int taps = c->conv_kernel > 0 ? c->conv_kernel : 4;
    const size_t qkv = (size_t)(2 * c->lin_k_heads * c->lin_k_dim +
                                c->lin_v_heads * c->lin_v_dim);
    const size_t conv = (size_t)(taps - 1) * qkv;
    const size_t kv = state_blob_kv(c, rows);
    size_t at = 0;
    for (int l = 0; l < nl && l < c->n_layers; l++) {
        if (m->layers[l].is_linear) {
            memcpy(dst + at, m->ssm_state[l], ssm * sizeof(float)); at += ssm;
            memcpy(dst + at, m->conv_state[l], conv * sizeof(float)); at += conv;
        } else {
            memcpy(dst + at, m->kcache[l], kv * sizeof(float)); at += kv;
            memcpy(dst + at, m->vcache[l], kv * sizeof(float)); at += kv;
        }
    }
    return at;
}

/* The k-row twin of gpu_step_checked. Two things differ, both because a row
 * is now part of a chain rather than a whole step: the snapshot has to cover
 * all k positions, and every row's logits are compared -- a row that read the
 * recurrence one position early still produces a plausible vector, and only a
 * per-row comparison shows it. */
static int gpu_rows_checked(q35_model *m, const int32_t *ids, int n,
                            int64_t pos, float *all, cwick_error *err) {
    static int check = -1;
    if (check < 0) {
        const char *e = getenv("CWICK_Q35_GPU_CHECK");
        check = (e && *e && *e != '0') ? (*e == '2' ? 2 : 1) : 0;
    }
    const size_t V = (size_t)m->cfg.vocab_size;
    uint8_t *snap = NULL;
    if (check) {
        snap = cwick_xmalloc(state_exact_bytes(m, pos + n));
        if (state_exact_copy(m, pos + n, snap, 1) < 0) {
            free(snap);
            snap = NULL;
        }
    }
    int r = q35_gpu_step_n(m, ids, pos, n, all);
    if (r <= 0) {
        free(snap);
        return r;
    }
    if (!snap) return 1;

    /* The state as the k-row step left it, to be compared against the state
     * the k single steps leave behind. A multi-row pass writes each row into
     * its own slot, so the live slot still holds the *pre*-step state until it
     * is committed -- commit the most pessimistic choice (keep row 0 only) and
     * the comparison below tests the same thing the rejected-draft path does. */
    int nl = m->cfg.n_layers;
    {
        const char *e = getenv("CWICK_Q35_GPU_MAXL");
        int v = e && *e ? atoi(e) : 0;
        if (v > 0 && v < nl) nl = v;
    }
    q35_gpu_commit_rows(0);
    const int64_t rows = pos + 1;
    const size_t st_len = state_blob_bytes(m, nl, rows);
    float *st_a = cwick_xmalloc(st_len * sizeof(float));
    float *st_b = cwick_xmalloc(st_len * sizeof(float));
    state_blob(m, nl, rows, st_a);

    float *gpu_all = cwick_xmalloc(V * (size_t)n * sizeof(float));
    memcpy(gpu_all, all, V * (size_t)n * sizeof(float));
    if (state_exact_copy(m, pos + n, snap, 0) < 0) {
        CWICK_INFO("q35gpu", "state restore failed; the check is skipped");
        free(snap);
        free(st_a);
        free(st_b);
        free(gpu_all);
        return 1;
    }
    free(snap);
    float *c = cwick_xmalloc(V * sizeof(float));
    for (int i = 0; i < n; i++) {
        /* Mode 1 is the sharp test: the reference is the same GPU path on one
         * row at a time, so the only variable left is "n rows at once". The
         * tiled kernel accumulates in the same lane order as the plain one,
         * so these should agree to the last bit. Mode 2 is the CPU pool, which
         * additionally measures how far the two backends drift (0.005 is
         * normal, and a repeated-token context amplifies it). */
        if (check == 1) {
            if (q35_gpu_step(m, ids[i], pos + i, c) <= 0) break;
        } else if (forward_chunk(m, ids + i, 1, pos + i, c, err) != 0) {
            break;
        }
        /* st_a is the state the verify kept after committing row 0, so the one
         * thing it can be compared against is the reference's *first* step --
         * comparing it against the state after all n would only measure the n-1
         * rows that were deliberately thrown away. */
        if (i == 0) state_blob(m, nl, rows, st_b);
        double maxd = 0.0, sum = 0.0;
        int amax_gpu = 0, amax_cpu = 0;
        for (size_t v = 0; v < V; v++) {
            double d = fabs((double)gpu_all[(size_t)i * V + v] - (double)c[v]);
            if (d > maxd) maxd = d;
            sum += d;
            if (gpu_all[(size_t)i * V + v] > gpu_all[(size_t)i * V + amax_gpu])
                amax_gpu = (int)v;
            if (c[v] > c[amax_cpu]) amax_cpu = (int)v;
        }
        CWICK_INFO("q35gpu", "check row %d/%d pos=%lld: max|gpu-cpu|=%.5f "
                   "mean=%.6f argmax gpu=%d cpu=%d %s", i + 1, n,
                   (long long)(pos + i), maxd, sum / (double)V,
                   amax_gpu, amax_cpu,
                   amax_gpu == amax_cpu ? "agree" : "DIFFER");
    }
    {
        const q35_config *c2 = &m->cfg;
        const size_t ssm = (size_t)c2->lin_v_heads * c2->lin_k_dim *
                           c2->lin_v_dim;
        const int taps2 = c2->conv_kernel > 0 ? c2->conv_kernel : 4;
        const size_t qkv2 = (size_t)(2 * c2->lin_k_heads * c2->lin_k_dim +
                                     c2->lin_v_heads * c2->lin_v_dim);
        const size_t conv = (size_t)(taps2 - 1) * qkv2;
        const size_t kv = state_blob_kv(c2, rows);
        size_t at = 0;
        for (int l = 0; l < nl && l < c2->n_layers; l++) {
            size_t parts[2];
            int np = 0;
            if (m->layers[l].is_linear) { parts[np++] = ssm; parts[np++] = conv; }
            else { parts[np++] = kv; parts[np++] = kv; }
            for (int q = 0; q < np; q++) {
                double maxd = 0.0;
                size_t worst = 0, nd = 0;
                for (size_t i = at; i < at + parts[q]; i++) {
                    double d = fabs((double)st_a[i] - (double)st_b[i]);
                    if (d > 1e-6) nd++;
                    if (d > maxd) { maxd = d; worst = i - at; }
                }
                CWICK_INFO("q35gpu", "  layer %d %-4s: max=%.6g at %zu, "
                           "%zu of %zu differ", l,
                           m->layers[l].is_linear ? (q ? "conv" : "ssm")
                                                  : (q ? "vc" : "kc"),
                           maxd, worst, nd, parts[q]);
                if (maxd > 1e-4) {
                    CWICK_INFO("q35gpu", "    at worst: k-row=%.6g single=%.6g "
                               "ratio=%.4g", st_a[at + worst],
                               st_b[at + worst],
                               st_b[at + worst] != 0.0
                                   ? (double)st_a[at + worst] /
                                         (double)st_b[at + worst]
                                   : 0.0);
                    CWICK_INFO("q35gpu", "    first 4: %.6g/%.6g %.6g/%.6g "
                               "%.6g/%.6g %.6g/%.6g",
                               st_a[at], st_b[at], st_a[at + 1], st_b[at + 1],
                               st_a[at + 2], st_b[at + 2], st_a[at + 3],
                               st_b[at + 3]);
                }
                at += parts[q];
            }
        }
    }
    free(st_a);
    free(st_b);
    free(c);
    free(gpu_all);
    return 1;
}

/* `n` consecutive positions, all of their logits. This is the verify entry
 * point a speculative step needs: k drafted tokens in, k distributions out,
 * from one weight pass. */
static int forward_rows(q35_model *m, const int32_t *ids, int n, int64_t pos,
                        float *all, int autokeep, cwick_error *err) {
    if (!m || !m->loaded) {
        cwick_error_set(err, CWICK_ERR_STATE, "qwen3_5 model is not open");
        return -1;
    }
    if (n <= 0 || !all) {
        cwick_error_set(err, CWICK_ERR_STATE, "empty verify batch");
        return -1;
    }
    if (pos + n > m->cfg.max_seq_len) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                        "position %lld + %d exceeds the %d-position context",
                        (long long)pos, n, m->cfg.max_seq_len);
        return -1;
    }
    if (n == 1) return q35_model_forward(m, ids, 1, pos, all, err);
    if (q35_gpu_available() && q35_gpu_supports_n(n)) {
        int g = gpu_rows_checked(m, ids, n, pos, all, err);
        if (g > 0) {
            /* Keeping every row is the default; a caller that decides after
             * seeing the logits calls q35_model_keep_rows instead, which is
             * why this is a choice and not something this function can do. */
            if (autokeep) q35_gpu_commit_rows(n - 1);
            return 0;
        }
        if (g < 0) {
            cwick_error_set(err, CWICK_ERR_STATE,
                            "q35 GPU verify of %d rows failed at position %lld",
                            n, (long long)pos);
            return -1;
        }
    }
    /* No k-row kernel: n single steps, which is correct and slower. */
    const size_t V = (size_t)m->cfg.vocab_size;
    for (int i = 0; i < n; i++)
        if (q35_model_forward(m, ids + i, 1, pos + i, all + (size_t)i * V,
                              err) != 0)
            return -1;
    return 0;
}

int q35_model_forward_all(q35_model *m, const int32_t *ids, int n, int64_t pos,
                          float *all, cwick_error *err) {
    return forward_rows(m, ids, n, pos, all, 1, err);
}

int q35_model_verify_rows(q35_model *m, const int32_t *ids, int n, int64_t pos,
                          float *all, cwick_error *err) {
    return forward_rows(m, ids, n, pos, all, 0, err);
}

int q35_model_keep_rows(q35_model *m, int j) {
    if (!m || !m->loaded || j < 0) return -1;
    if (!q35_gpu_available()) return -1;
    return q35_gpu_commit_rows(j) ? 0 : -1;
}

int q35_model_forward(q35_model *m, const int32_t *ids, int n, int64_t pos,
                      float *logits, cwick_error *err) {    if (!m || !m->loaded) {
        cwick_error_set(err, CWICK_ERR_STATE, "qwen3_5 model is not open");
        return -1;
    }
    if (n <= 0) {
        cwick_error_set(err, CWICK_ERR_STATE, "empty forward");
        return -1;
    }
    if (pos + n > m->cfg.max_seq_len) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                        "position %lld + %d exceeds the %d-position context",
                        (long long)pos, n, m->cfg.max_seq_len);
        return -1;
    }
    /* The chunk size bounds every scratch buffer in the model, so a longer run
     * is split rather than refused. Only the last chunk yields logits, which is
     * the only one the caller ever reads. */
    /* One position is a decode step, which is the only shape the GPU forward
     * implements -- and the only shape that pays: a per-projection GPU path
     * loses to 128 round trips per step, while a whole step in one command
     * buffer is 2.9x the CPU pool. Prefill stays on the CPU, where it already
     * runs at 71-79 tok/s. */
    if (n == 1) {
        int g = gpu_step_checked(m, ids[0], pos, logits, err);
        if (g > 0) return 0;
        if (g < 0) {
            /* Half a step ran on the GPU, so the recurrence state no longer
             * matches anything. Failing the request is the only honest
             * answer; the next one rebuilds the state from scratch. */
            cwick_error_set(err, CWICK_ERR_STATE,
                            "q35 GPU decode step failed at position %lld",
                            (long long)pos);
            return -1;
        }
    }
    int done = 0;
    while (done < n) {
        int take = n - done;
        if (take > Q35_CHUNK) take = Q35_CHUNK;
        bool last = (done + take == n);
        if (forward_chunk(m, ids + done, take, pos + done,
                          last ? logits : m->logits, err) != 0)
            return -1;
        done += take;
    }
    return 0;
}
