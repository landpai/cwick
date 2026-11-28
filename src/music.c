/* Native MiniMax-Music3 inference. Weights remain in read-only mappings; the
 * only derived weight data are per-channel weight-norm scale vectors. */
#include "music.h"

#include <Accelerate/Accelerate.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include "h3v_gpu.h"
#include "json.h"
#include "log.h"
#include "mem.h"
#include "rt_torch.h"
#include "sample.h"
#include "util.h"

#undef I

#define MUSIC_LM_LAYERS 36
#define MUSIC_LM_HIDDEN 4096
#define MUSIC_LM_HEADS 32
#define MUSIC_LM_KV_HEADS 8
#define MUSIC_LM_HEAD_DIM 128
#define MUSIC_LM_INTER 12288
#define MUSIC_LM_VOCAB 200000
#define MUSIC_LM_PREFILL 256

#define MUSIC_DEPTH_LAYERS 4
#define MUSIC_DEPTH_HEADS 16
#define MUSIC_DEPTH_INTER 6144
#define MUSIC_DEPTH_CODEBOOKS 7
#define MUSIC_DEPTH_VOCAB 1024

#define MUSIC_COND_LAYERS 8
#define MUSIC_COND_DIM 2048
#define MUSIC_LATENT_CH 128
#define MUSIC_DIT_LAYERS 36
#define MUSIC_DIT_WIDTH 2048
#define MUSIC_DIT_HEADS 32
#define MUSIC_DIT_HEAD_DIM 64
#define MUSIC_DIT_FF 8192
#define MUSIC_DIT_ROTARY 32
#define MUSIC_DIT_IN (2 * MUSIC_LATENT_CH + MUSIC_COND_DIM)

#define MUSIC_CHUNK_FRAMES 200
#define MUSIC_SINGLE_CHUNK_FRAMES 256
#define MUSIC_CHUNK_HOP 100
#define MUSIC_OVERLAP_LATENT 172
#define MUSIC_CARRY_LATENT 344
#define MUSIC_CROP_LEFT 86
#define MUSIC_CROP_RIGHT 258
#define MUSIC_LATENT_HOP 512
#define MUSIC_VAE_RATE 44100

#define MUSIC_AUDIO_END 151670
#define MUSIC_AUDIO_CFG 151654
#define MUSIC_AUDIO_OFFSET 151675
#define MUSIC_SEMANTIC_VOCAB 16384
#define MUSIC_AUDIO_HEAD_GAP (MUSIC_AUDIO_OFFSET - MUSIC_AUDIO_END)
#define MUSIC_AUDIO_HEAD_ROWS (MUSIC_AUDIO_OFFSET + MUSIC_SEMANTIC_VOCAB - MUSIC_AUDIO_END)
#define MUSIC_AR_CFG 1.5f
#define MUSIC_AR_TOPK 50
#define MUSIC_RS_TAPS 32

static int music_ar_min_frames(int max_frames) {
    return (max_frames * MUSIC_AR_MIN_PERCENT + 99) / 100;
}

static bool music_early_end_enabled(void) {
    const char *e = getenv("CWICK_MUSIC_ALLOW_EARLY_END");
    return e && *e && strcmp(e, "0") != 0;
}

static uint32_t music_audio_head_token(uint32_t row) {
    return MUSIC_AUDIO_END + row;
}

static float *music_batch_second(float *batch, int64_t frames) {
    return batch + frames * MUSIC_LATENT_CH;
}

/* The checkpoint's flow time runs 0 -> 1, with 0 = pure noise and 1 = data.
 * The reference scheduler is a FlowMatchEulerDiscreteScheduler configured with
 * invert_sigmas, so the sigma grid starts at 0 and the transformer is fed that
 * same value as its timestep. Running the grid backwards feeds t=1 (data) to a
 * pure-noise latent, which is what turned every job into noise. */
static double music_flow_time(int step, int steps) {
    if (getenv("CWICK_MUSIC_FLOW_LEGACY")) return 1.0 - (double)step / steps;
    return (double)step / steps;
}

static float music_flow_step_size(int steps) {
    if (getenv("CWICK_MUSIC_FLOW_LEGACY")) return -1.0f / (float)steps;
    return 1.0f / (float)steps;
}

static double music_ms(void) { return (double)cwick_prof_ns() / 1e6; }

static bool music_audio_valid(const float *audio, int64_t count,
                              const char *stage, cwick_error *err);

static bool music_check_cancel(const music_request *req, cwick_error *err) {
    if (req && req->cancel && atomic_load_explicit(req->cancel, memory_order_acquire)) {
        cwick_error_set(err, CWICK_ERR_STATE, "music generation cancelled");
        return false;
    }
    return true;
}

static void music_progress(const music_request *req, const char *phase,
                           int64_t done, int64_t total) {
    if (!req || !req->progress) return;
    double frac = total > 0 ? (double)done / (double)total : 0.0;
    if (frac < 0.0) frac = 0.0;
    if (frac > 1.0) frac = 1.0;
    req->progress(req->progress_ctx, phase, done, total, frac);
}

static float *music_f32(int64_t n) {
    if (n < 0 || (uint64_t)n > SIZE_MAX / sizeof(float)) return NULL;
    return (float *)cwick_xmalloc((size_t)n * sizeof(float));
}

static float *music_calloc_f32(int64_t n) {
    float *p = music_f32(n);
    if (p) memset(p, 0, (size_t)n * sizeof(float));
    return p;
}

typedef struct {
    const void *p;
    int64_t rows, cols;
    bool f32;
} music_mat;

typedef struct {
    music_mat q, k, v, o, gate, up, down;
    float *input_norm, *post_norm, *q_norm, *k_norm;
} music_lm_layer;

typedef struct {
    music_lm_layer layer[MUSIC_LM_LAYERS];
    music_mat embed, head;
    float *norm;
} music_lm;

typedef struct {
    music_mat q, k, v, o, gate, up, down;
    float *input_norm, *post_norm;
} music_depth_layer;

typedef struct {
    music_depth_layer layer[MUSIC_DEPTH_LAYERS];
    music_mat audio_embeddings, projection, pos_embedding, norm;
    music_mat audio_heads[MUSIC_DEPTH_CODEBOOKS];
    float *norm_weight;
} music_depth;

typedef struct {
    float *layer_logits;
    float layer_scale;
    const float *proj_w, *proj_b;
} music_condition;

typedef struct {
    music_mat q, k, v, o, ff_in, ff_out;
    const float *norm1_w, *norm1_b, *norm2_w, *norm2_b;
    const float *ff_in_b, *ff_out_b;
} music_dit_layer;

typedef struct {
    music_dit_layer layer[MUSIC_DIT_LAYERS];
    music_mat preprocess, proj_in, proj_out, postprocess;
    const float *time_proj, *time1_w, *time1_b, *time2_w, *time2_b;
} music_dit;

/* A convolution still reads the checkpoint's [out,in,k] (or [in,out,k]) bytes.
 * Weight normalisation is a per-channel scale, not a converted weight matrix. */
typedef struct {
    const float *v, *g, *b;
    float *scale;
    float *kernel;              /* transposed-conv tap-major inference layout */
    int I, O, K, stride, dil, pad;
    bool transposed;
} music_conv;

typedef struct {
    music_conv c1, c2;
    const float *alpha1, *alpha2;
} music_resunit;

typedef struct {
    music_conv up;
    music_resunit unit[3];
    const float *alpha;
} music_vocoder_block;

typedef struct {
    music_conv in_proj, conv_in, conv_out;
    music_vocoder_block block[4];
    const float *alpha_out;
} music_vocoder;

typedef struct {
    music_conv input;
    music_conv unit[4][6];
    music_conv down[MUSIC_DEPTH_LAYERS];
    music_conv final;
    music_conv mean, logs;
    const float *unit_alpha[MUSIC_DEPTH_LAYERS][6];
    const float *down_alpha[MUSIC_DEPTH_LAYERS];
    const float *final_alpha;
} music_vae_encoder;

struct music_model {
    char *root;
    pthread_mutex_t lock;
    bool ready;
    double load_ms;
    cwick_safetensors lm_saf, depth_saf, cond_saf, dit_saf, voc_saf;
    bool lm_open, depth_open, cond_open, dit_open, voc_open;
    rt_pt *dav;
    bool *dav_locked;
    int dav_locked_count;
    bool gpu_adopted;
    cwick_tokenizer *tok;
    music_lm lm;
    music_depth depth;
    music_condition condition;
    music_dit dit;
    music_vocoder vocoder;
    music_vae_encoder encoder;
};

static char *music_join(const char *a, const char *b) {
    size_t n = strlen(a) + strlen(b) + 2;
    char *s = (char *)cwick_xmalloc(n);
    cwick_path_join(s, n, a, b);
    return s;
}

static bool music_json_int_file(const char *path, const char *key, int64_t want,
                                cwick_error *err) {
    char *text = NULL;
    size_t len = 0;
    if (cwick_read_file(path, &text, &len) != 0) {
        cwick_error_set(err, CWICK_ERR_IO, "cannot read %s", path);
        return false;
    }
    json_value *root = cwick_json_parse(text, len, err);
    free(text);
    if (!root) return false;
    json_value *v = cwick_json_get(root, key);
    bool ok = v && v->kind == JSON_NUMBER && (int64_t)v->number == want;
    if (!ok)
        cwick_error_set(err, CWICK_ERR_CONTRACT, "%s: %s must be %lld", path, key,
                        (long long)want);
    cwick_json_free(root);
    return ok;
}

static bool music_json_string_file(const char *path, const char *key,
                                   const char *want, cwick_error *err) {
    char *text = NULL;
    size_t len = 0;
    if (cwick_read_file(path, &text, &len) != 0) {
        cwick_error_set(err, CWICK_ERR_IO, "cannot read %s", path);
        return false;
    }
    json_value *root = cwick_json_parse(text, len, err);
    free(text);
    if (!root) return false;
    const char *got = cwick_json_str(cwick_json_get(root, key));
    bool ok = got && strcmp(got, want) == 0;
    if (!ok)
        cwick_error_set(err, CWICK_ERR_CONTRACT, "%s: %s must be %s", path, key, want);
    cwick_json_free(root);
    return ok;
}

static bool music_mat_get(const cwick_safetensors *saf, const char *name,
                          bool f32, int64_t rows, int64_t cols, music_mat *out,
                          cwick_error *err) {
    cwick_st_tensor t;
    if (cwick_safetensors_get(saf, name, &t, err) != CWICK_OK) return false;
    if ((f32 && t.dtype != CWICK_ST_F32) || (!f32 && t.dtype != CWICK_ST_BF16)) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "%s has unexpected dtype", name);
        return false;
    }
    bool shape_ok = t.rows == rows && t.cols == cols;
    if (!shape_ok && (rows == 1 || cols == 1) &&
        t.rows * t.cols == rows * cols) shape_ok = true;
    if (!shape_ok || !t.data) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                        "%s shape %lldx%lld, expected %lldx%lld", name,
                        (long long)t.rows, (long long)t.cols,
                        (long long)rows, (long long)cols);
        return false;
    }
    out->p = t.data;
    out->rows = rows;
    out->cols = cols;
    out->f32 = f32;
    return true;
}

static float *music_vec_copy(const music_mat *m) {
    int64_t n = m->rows * m->cols;
    float *v = music_f32(n);
    if (!v) return NULL;
    if (m->f32) memcpy(v, m->p, (size_t)n * sizeof(float));
    else {
        const uint16_t *p = (const uint16_t *)m->p;
        for (int64_t i = 0; i < n; i++) v[i] = cwick_bf16_fast(p[i]);
    }
    return v;
}

static bool music_validate_lm(music_model *m, cwick_error *err) {
    char n[192];
    music_lm *l = &m->lm;
    if (!music_mat_get(&m->lm_saf, "model.embed_tokens.weight", false,
                       MUSIC_LM_VOCAB, MUSIC_LM_HIDDEN, &l->embed, err) ||
        !music_mat_get(&m->lm_saf, "lm_head.weight", false,
                       MUSIC_LM_VOCAB, MUSIC_LM_HIDDEN, &l->head, err)) return false;
    music_mat w;
    if (!music_mat_get(&m->lm_saf, "model.norm.weight", false,
                       MUSIC_LM_HIDDEN, 1, &w, err)) return false;
    l->norm = music_vec_copy(&w);
    for (int i = 0; i < MUSIC_LM_LAYERS; i++) {
        music_lm_layer *a = &l->layer[i];
#define LM_GET(field, suffix, rows, cols)                                      \
        do {                                                                   \
            snprintf(n, sizeof(n), "model.layers.%d.%s", i, suffix);           \
            if (!music_mat_get(&m->lm_saf, n, false, rows, cols, &a->field, err)) \
                return false;                                                  \
        } while (0)
        LM_GET(q, "self_attn.q_proj.weight", MUSIC_LM_HIDDEN, MUSIC_LM_HIDDEN);
        LM_GET(k, "self_attn.k_proj.weight", MUSIC_LM_KV_HEADS * MUSIC_LM_HEAD_DIM, MUSIC_LM_HIDDEN);
        LM_GET(v, "self_attn.v_proj.weight", MUSIC_LM_KV_HEADS * MUSIC_LM_HEAD_DIM, MUSIC_LM_HIDDEN);
        LM_GET(o, "self_attn.o_proj.weight", MUSIC_LM_HIDDEN, MUSIC_LM_HIDDEN);
        LM_GET(gate, "mlp.gate_proj.weight", MUSIC_LM_INTER, MUSIC_LM_HIDDEN);
        LM_GET(up, "mlp.up_proj.weight", MUSIC_LM_INTER, MUSIC_LM_HIDDEN);
        LM_GET(down, "mlp.down_proj.weight", MUSIC_LM_HIDDEN, MUSIC_LM_INTER);
#undef LM_GET
        snprintf(n, sizeof(n), "model.layers.%d.input_layernorm.weight", i);
        if (!music_mat_get(&m->lm_saf, n, false, MUSIC_LM_HIDDEN, 1, &w, err)) return false;
        a->input_norm = music_vec_copy(&w);
        snprintf(n, sizeof(n), "model.layers.%d.post_attention_layernorm.weight", i);
        if (!music_mat_get(&m->lm_saf, n, false, MUSIC_LM_HIDDEN, 1, &w, err)) return false;
        a->post_norm = music_vec_copy(&w);
        snprintf(n, sizeof(n), "model.layers.%d.self_attn.q_norm.weight", i);
        if (!music_mat_get(&m->lm_saf, n, false, MUSIC_LM_HEAD_DIM, 1, &w, err)) return false;
        a->q_norm = music_vec_copy(&w);
        snprintf(n, sizeof(n), "model.layers.%d.self_attn.k_norm.weight", i);
        if (!music_mat_get(&m->lm_saf, n, false, MUSIC_LM_HEAD_DIM, 1, &w, err)) return false;
        a->k_norm = music_vec_copy(&w);
        if (!a->input_norm || !a->post_norm || !a->q_norm || !a->k_norm) return false;
    }
    return l->norm != NULL;
}

static bool music_validate_depth(music_model *m, cwick_error *err) {
    char n[192];
    music_depth *d = &m->depth;
    if (!music_mat_get(&m->depth_saf, "audio_embeddings.weight", false,
                       MUSIC_DEPTH_VOCAB * MUSIC_DEPTH_CODEBOOKS, MUSIC_LM_HIDDEN,
                       &d->audio_embeddings, err) ||
        !music_mat_get(&m->depth_saf, "projection.weight", false,
                       MUSIC_LM_HIDDEN, MUSIC_LM_HIDDEN, &d->projection, err) ||
        !music_mat_get(&m->depth_saf, "pos_embedding.weight", false,
                       16, MUSIC_LM_HIDDEN, &d->pos_embedding, err) ||
        !music_mat_get(&m->depth_saf, "norm.weight", false,
                       MUSIC_LM_HIDDEN, 1, &d->norm, err)) return false;
    d->norm_weight = music_vec_copy(&d->norm);
    for (int g = 0; g < MUSIC_DEPTH_CODEBOOKS; g++) {
        snprintf(n, sizeof(n), "audio_heads.%d.weight", g);
        if (!music_mat_get(&m->depth_saf, n, false, MUSIC_DEPTH_VOCAB,
                           MUSIC_LM_HIDDEN, &d->audio_heads[g], err)) return false;
    }
    for (int i = 0; i < MUSIC_DEPTH_LAYERS; i++) {
        music_depth_layer *a = &d->layer[i];
#define DEPTH_GET(field, suffix, rows, cols)                                    \
        do {                                                                   \
            snprintf(n, sizeof(n), "layers.%d.%s", i, suffix);                  \
            if (!music_mat_get(&m->depth_saf, n, false, rows, cols, &a->field, err)) \
                return false;                                                  \
        } while (0)
        DEPTH_GET(q, "attn.to_q.weight", MUSIC_LM_HIDDEN, MUSIC_LM_HIDDEN);
        DEPTH_GET(k, "attn.to_k.weight", MUSIC_LM_HIDDEN, MUSIC_LM_HIDDEN);
        DEPTH_GET(v, "attn.to_v.weight", MUSIC_LM_HIDDEN, MUSIC_LM_HIDDEN);
        DEPTH_GET(o, "attn.to_out.weight", MUSIC_LM_HIDDEN, MUSIC_LM_HIDDEN);
        DEPTH_GET(gate, "gate_proj.weight", MUSIC_DEPTH_INTER, MUSIC_LM_HIDDEN);
        DEPTH_GET(up, "up_proj.weight", MUSIC_DEPTH_INTER, MUSIC_LM_HIDDEN);
        DEPTH_GET(down, "down_proj.weight", MUSIC_LM_HIDDEN, MUSIC_DEPTH_INTER);
#undef DEPTH_GET
        music_mat w;
        snprintf(n, sizeof(n), "layers.%d.input_layernorm.weight", i);
        if (!music_mat_get(&m->depth_saf, n, false, MUSIC_LM_HIDDEN, 1, &w, err)) return false;
        a->input_norm = music_vec_copy(&w);
        snprintf(n, sizeof(n), "layers.%d.post_attention_layernorm.weight", i);
        if (!music_mat_get(&m->depth_saf, n, false, MUSIC_LM_HIDDEN, 1, &w, err)) return false;
        a->post_norm = music_vec_copy(&w);
        if (!a->input_norm || !a->post_norm) return false;
    }
    return d->norm_weight != NULL;
}

static bool music_validate_condition(music_model *m, cwick_error *err) {
    music_mat w, b, logits, scale;
    if (!music_mat_get(&m->cond_saf, "proj.weight", true, MUSIC_COND_DIM,
                       MUSIC_LM_HIDDEN * 3, &w, err) ||
        !music_mat_get(&m->cond_saf, "proj.bias", true, MUSIC_COND_DIM, 1, &b, err) ||
        !music_mat_get(&m->cond_saf, "layer_weight_logits", true,
                       MUSIC_COND_LAYERS, 1, &logits, err) ||
        !music_mat_get(&m->cond_saf, "layer_scale", true, 1, 1, &scale, err)) return false;
    m->condition.proj_w = (const float *)w.p;
    m->condition.proj_b = (const float *)b.p;
    m->condition.layer_logits = music_vec_copy(&logits);
    float *scale_value = music_vec_copy(&scale);
    if (!m->condition.layer_logits || !scale_value) {
        free(scale_value);
        return false;
    }
    m->condition.layer_scale = scale_value[0];
    free(scale_value);
    return true;
}

static bool music_validate_dit(music_model *m, cwick_error *err) {
    char n[192];
    music_dit *d = &m->dit;
#define DIT_GET(field, suffix, rows, cols)                                      \
    do {                                                                       \
        if (!music_mat_get(&m->dit_saf, suffix, true, rows, cols, &d->field, err)) \
            return false;                                                      \
    } while (0)
    DIT_GET(preprocess, "preprocess_conv.weight", MUSIC_DIT_IN, MUSIC_DIT_IN);
    DIT_GET(proj_in, "proj_in.weight", MUSIC_DIT_WIDTH, MUSIC_DIT_IN);
    DIT_GET(proj_out, "proj_out.weight", MUSIC_LATENT_CH, MUSIC_DIT_WIDTH);
    DIT_GET(postprocess, "postprocess_conv.weight", MUSIC_LATENT_CH, MUSIC_LATENT_CH);
#undef DIT_GET
    music_mat w;
#define DIT_SCALAR(ptr, suffix, rows)                                          \
    do {                                                                       \
        if (!music_mat_get(&m->dit_saf, suffix, true, rows, 1, &w, err)) return false; \
        d->ptr = (const float *)w.p;                                            \
    } while (0)
    DIT_SCALAR(time_proj, "time_proj.weight", 128);
    DIT_SCALAR(time1_b, "time_embed.linear_1.bias", MUSIC_DIT_WIDTH);
    DIT_SCALAR(time2_b, "time_embed.linear_2.bias", MUSIC_DIT_WIDTH);
#undef DIT_SCALAR
    /* The time matrices are two-dimensional, so fetch them separately. */
    if (!music_mat_get(&m->dit_saf, "time_embed.linear_1.weight", true,
                       MUSIC_DIT_WIDTH, 256, &w, err)) return false;
    d->time1_w = (const float *)w.p;
    if (!music_mat_get(&m->dit_saf, "time_embed.linear_2.weight", true,
                       MUSIC_DIT_WIDTH, MUSIC_DIT_WIDTH, &w, err)) return false;
    d->time2_w = (const float *)w.p;
    for (int i = 0; i < MUSIC_DIT_LAYERS; i++) {
        music_dit_layer *a = &d->layer[i];
#define DIT_LAYER(field, suffix, rows, cols)                                    \
        do {                                                                   \
            snprintf(n, sizeof(n), "transformer_blocks.%d.%s", i, suffix);      \
            if (!music_mat_get(&m->dit_saf, n, true, rows, cols, &a->field, err)) \
                return false;                                                  \
        } while (0)
        DIT_LAYER(q, "attn.to_q.weight", MUSIC_DIT_WIDTH, MUSIC_DIT_WIDTH);
        DIT_LAYER(k, "attn.to_k.weight", MUSIC_DIT_WIDTH, MUSIC_DIT_WIDTH);
        DIT_LAYER(v, "attn.to_v.weight", MUSIC_DIT_WIDTH, MUSIC_DIT_WIDTH);
        DIT_LAYER(o, "attn.to_out.0.weight", MUSIC_DIT_WIDTH, MUSIC_DIT_WIDTH);
        DIT_LAYER(ff_in, "ff_in.weight", 2 * MUSIC_DIT_FF, MUSIC_DIT_WIDTH);
        DIT_LAYER(ff_out, "ff_out.weight", MUSIC_DIT_WIDTH, MUSIC_DIT_FF);
#undef DIT_LAYER
#define DIT_VEC(ptr, suffix, count)                                            \
        do {                                                                   \
            snprintf(n, sizeof(n), "transformer_blocks.%d.%s", i, suffix);      \
            if (!music_mat_get(&m->dit_saf, n, true, count, 1, &w, err)) return false; \
            a->ptr = (const float *)w.p;                                        \
        } while (0)
        DIT_VEC(norm1_w, "norm1.weight", MUSIC_DIT_WIDTH); DIT_VEC(norm1_b, "norm1.bias", MUSIC_DIT_WIDTH);
        DIT_VEC(norm2_w, "norm2.weight", MUSIC_DIT_WIDTH); DIT_VEC(norm2_b, "norm2.bias", MUSIC_DIT_WIDTH);
        DIT_VEC(ff_in_b, "ff_in.bias", 2 * MUSIC_DIT_FF); DIT_VEC(ff_out_b, "ff_out.bias", MUSIC_DIT_WIDTH);
#undef DIT_VEC
    }
    return true;
}

static bool music_pt_get(const rt_pt *pt, const char *name, const int64_t *shape,
                         int ndim, const rt_tensor **out, cwick_error *err) {
    const rt_tensor *t = rt_pt_get(pt, name);
    if (!t) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "dav.pth tensor %s not found", name);
        return false;
    }
    if (t->dtype != RT_DT_F32 || !t->data || !rt_tensor_fits(t, shape, ndim)) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "dav.pth tensor %s has unexpected dtype/shape", name);
        return false;
    }
    *out = t;
    return true;
}

static bool music_conv_prepare_layout(music_conv *c, cwick_error *err) {
    if (!c->transposed) return true;
    int64_t n = (int64_t)c->K * c->O * c->I;
    c->kernel = music_f32(n);
    if (!c->kernel) {
        cwick_error_set(err, CWICK_ERR_MEMORY, "cannot allocate convolution layout");
        return false;
    }
    for (int i = 0; i < c->I; i++)
        for (int o = 0; o < c->O; o++)
            for (int k = 0; k < c->K; k++)
                c->kernel[((int64_t)k * c->O + o) * c->I + i] =
                    c->v[((int64_t)i * c->O + o) * c->K + k];
    return true;
}

static bool music_conv_build_st(const cwick_safetensors *saf, const char *prefix,
                                music_conv *c, int I, int O, int K, int stride,
                                int dil, int pad, bool transposed, bool normalized,
                                cwick_error *err) {
    memset(c, 0, sizeof(*c));
    c->I = I; c->O = O; c->K = K; c->stride = stride; c->dil = dil; c->pad = pad;
    c->transposed = transposed;
    char n[256];
    cwick_st_tensor t;
    snprintf(n, sizeof(n), "%s.weight", prefix);
    if (cwick_safetensors_get(saf, n, &t, err) == CWICK_OK) {
        if (normalized || t.dtype != CWICK_ST_F32 ||
            t.rows != (transposed ? I : O) ||
            t.cols != (transposed ? O * K : I * K)) {
            cwick_error_set(err, CWICK_ERR_CONTRACT, "%s has unexpected layout", prefix);
            return false;
        }
        c->v = (const float *)t.data;
    } else {
        if (!normalized) return false;
        snprintf(n, sizeof(n), "%s.weight_v", prefix);
        if (cwick_safetensors_get(saf, n, &t, err) != CWICK_OK) return false;
        if (t.dtype != CWICK_ST_F32 || t.rows != (transposed ? I : O) ||
            t.cols != (transposed ? O * K : I * K)) {
            cwick_error_set(err, CWICK_ERR_CONTRACT, "%s weight_v has unexpected layout", prefix);
            return false;
        }
        c->v = (const float *)t.data;
        snprintf(n, sizeof(n), "%s.weight_g", prefix);
        if (cwick_safetensors_get(saf, n, &t, err) != CWICK_OK) return false;
        if (t.dtype != CWICK_ST_F32 || t.rows * t.cols != (transposed ? I : O)) {
            cwick_error_set(err, CWICK_ERR_CONTRACT, "%s weight_g has unexpected layout", prefix);
            return false;
        }
        c->g = (const float *)t.data;
        c->scale = music_f32(transposed ? I : O);
        if (!c->scale) return false;
        for (int x = 0; x < (transposed ? I : O); x++) {
            int64_t count = (int64_t)(transposed ? O : I) * K;
            const float *row = c->v + (int64_t)x * count;
            double ss = 0.0;
            for (int64_t j = 0; j < count; j++) ss += (double)row[j] * row[j];
            double norm = sqrt(ss);
            if (!(norm > 1e-20) || !isfinite(c->g[x])) {
                cwick_error_set(err, CWICK_ERR_CONTRACT, "%s has invalid weight norm", prefix);
                return false;
            }
            c->scale[x] = (float)((double)c->g[x] / norm);
        }
    }
    snprintf(n, sizeof(n), "%s.bias", prefix);
    if (cwick_safetensors_get(saf, n, &t, err) != CWICK_OK) return false;
    if (t.dtype != CWICK_ST_F32 || t.rows * t.cols != O) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "%s bias has unexpected layout", prefix);
        return false;
    }
    c->b = (const float *)t.data;
    return music_conv_prepare_layout(c, err);
}

static bool music_conv_build_pt(const rt_pt *pt, const char *prefix, music_conv *c,
                                int I, int O, int K, int stride, int dil, int pad,
                                bool transposed, cwick_error *err) {
    memset(c, 0, sizeof(*c));
    c->I = I; c->O = O; c->K = K; c->stride = stride; c->dil = dil; c->pad = pad;
    c->transposed = transposed;
    char n[256];
    const rt_tensor *v, *g, *b;
    int64_t vs[3] = {transposed ? I : O, transposed ? O : I, K};
    int64_t gs[3] = {transposed ? I : O, 1, 1};
    int64_t bs[1] = {O};
    snprintf(n, sizeof(n), "%s.weight_v", prefix);
    if (!music_pt_get(pt, n, vs, 3, &v, err)) return false;
    snprintf(n, sizeof(n), "%s.weight_g", prefix);
    if (!music_pt_get(pt, n, gs, 3, &g, err)) return false;
    snprintf(n, sizeof(n), "%s.bias", prefix);
    if (!music_pt_get(pt, n, bs, 1, &b, err)) return false;
    c->v = (const float *)v->data;
    c->g = (const float *)g->data;
    c->b = (const float *)b->data;
    c->scale = music_f32(transposed ? I : O);
    if (!c->scale) return false;
    for (int x = 0; x < (transposed ? I : O); x++) {
        int64_t count = (int64_t)(transposed ? O : I) * K;
        const float *row = c->v + (int64_t)x * count;
        double ss = 0.0;
        for (int64_t j = 0; j < count; j++) ss += (double)row[j] * row[j];
        double norm = sqrt(ss);
        if (!(norm > 1e-20) || !isfinite(c->g[x])) {
            cwick_error_set(err, CWICK_ERR_CONTRACT, "%s has invalid weight norm", prefix);
            return false;
        }
        c->scale[x] = (float)((double)c->g[x] / norm);
    }
    return music_conv_prepare_layout(c, err);
}

static bool music_alpha_st(const cwick_safetensors *saf, const char *name, int ch,
                           const float **out, cwick_error *err) {
    music_mat w;
    if (!music_mat_get(saf, name, true, ch, 1, &w, err)) return false;
    *out = (const float *)w.p;
    return true;
}

static bool music_alpha_pt(const rt_pt *pt, const char *name, int ch,
                           const float **out, cwick_error *err) {
    const rt_tensor *t;
    int64_t shape[3] = {1, ch, 1};
    if (!music_pt_get(pt, name, shape, 3, &t, err)) return false;
    *out = (const float *)t->data;
    return true;
}

static bool music_validate_vocoder(music_model *m, cwick_error *err) {
    music_vocoder *v = &m->vocoder;
    char n[256];
    if (!music_conv_build_st(&m->voc_saf, "dec_in_proj", &v->in_proj,
                             64, 1024, 1, 1, 1, 0, false, false, err) ||
        !music_conv_build_st(&m->voc_saf, "conv_in", &v->conv_in,
                             1024, 1536, 7, 1, 1, 3, false, true, err) ||
        !music_conv_build_st(&m->voc_saf, "conv_out", &v->conv_out,
                             96, 1, 7, 1, 1, 3, false, true, err)) return false;
    if (!music_alpha_st(&m->voc_saf, "snake_out.alpha", 96, &v->alpha_out, err)) return false;
    const int rates[4] = {8, 8, 4, 2};
    for (int i = 0; i < 4; i++) {
        int in = 1536 >> i, out = 1536 >> (i + 1), r = rates[i];
        music_vocoder_block *b = &v->block[i];
        snprintf(n, sizeof(n), "blocks.%d.conv_t1", i);
        if (!music_conv_build_st(&m->voc_saf, n, &b->up, in, out, 2 * r,
                                 r, 1, r / 2, true, true, err)) return false;
        snprintf(n, sizeof(n), "blocks.%d.snake1.alpha", i);
        if (!music_alpha_st(&m->voc_saf, n, in, &b->alpha, err)) return false;
        for (int j = 0; j < 3; j++) {
            int dil[3] = {1, 3, 9};
            music_resunit *u = &b->unit[j];
            snprintf(n, sizeof(n), "blocks.%d.res_unit%d.conv1", i, j + 1);
            if (!music_conv_build_st(&m->voc_saf, n, &u->c1, out, out, 7,
                                     1, dil[j], 3 * dil[j], false, true, err)) return false;
            snprintf(n, sizeof(n), "blocks.%d.res_unit%d.conv2", i, j + 1);
            if (!music_conv_build_st(&m->voc_saf, n, &u->c2, out, out, 1,
                                     1, 1, 0, false, true, err)) return false;
            snprintf(n, sizeof(n), "blocks.%d.res_unit%d.snake1.alpha", i, j + 1);
            if (!music_alpha_st(&m->voc_saf, n, out, &u->alpha1, err)) return false;
            snprintf(n, sizeof(n), "blocks.%d.res_unit%d.snake2.alpha", i, j + 1);
            if (!music_alpha_st(&m->voc_saf, n, out, &u->alpha2, err)) return false;
        }
    }
    return true;
}

static bool music_validate_encoder(music_model *m, cwick_error *err) {
    music_vae_encoder *e = &m->encoder;
    char n[256];
    if (!music_conv_build_pt(m->dav, "encoder.block.0", &e->input,
                             1, 64, 7, 1, 1, 3, false, err)) return false;
    const int strides[4] = {2, 4, 8, 8};
    const int dil[3] = {1, 3, 9};
    for (int i = 0; i < 4; i++) {
        int ch = 64 << (i + 1), half = ch / 2, r = strides[i];
        for (int j = 0; j < 3; j++) {
            snprintf(n, sizeof(n), "encoder.block.%d.block.%d.block.1", i + 1, j);
            if (!music_conv_build_pt(m->dav, n, &e->unit[i][2 * j], half, half, 7,
                                     1, dil[j], 3 * dil[j], false, err)) return false;
            snprintf(n, sizeof(n), "encoder.block.%d.block.%d.block.3", i + 1, j);
            if (!music_conv_build_pt(m->dav, n, &e->unit[i][2 * j + 1], half, half, 1,
                                     1, 1, 0, false, err)) return false;
            snprintf(n, sizeof(n), "encoder.block.%d.block.%d.block.0.alpha", i + 1, j);
            if (!music_alpha_pt(m->dav, n, half, &e->unit_alpha[i][2 * j], err)) return false;
            snprintf(n, sizeof(n), "encoder.block.%d.block.%d.block.2.alpha", i + 1, j);
            if (!music_alpha_pt(m->dav, n, half, &e->unit_alpha[i][2 * j + 1], err)) return false;
        }
        snprintf(n, sizeof(n), "encoder.block.%d.block.3.alpha", i + 1);
        if (!music_alpha_pt(m->dav, n, half, &e->down_alpha[i], err)) return false;
        snprintf(n, sizeof(n), "encoder.block.%d.block.4", i + 1);
        if (!music_conv_build_pt(m->dav, n, &e->down[i], half, ch, 2 * r,
                                 r, 1, r / 2, false, err)) return false;
    }
    if (!music_alpha_pt(m->dav, "encoder.block.5.alpha", 1024, &e->final_alpha, err) ||
        !music_conv_build_pt(m->dav, "encoder.block.6", &e->final,
                             1024, 1024, 3, 1, 1, 1, false, err)) return false;
    /* The posterior heads are plain 1x1 convolutions. */
    const rt_tensor *mw, *mb, *lw, *lb;
    int64_t wshape[3] = {64, 1024, 1}, bshape[1] = {64};
    if (!music_pt_get(m->dav, "mean_proj.weight", wshape, 3, &mw, err) ||
        !music_pt_get(m->dav, "mean_proj.bias", bshape, 1, &mb, err) ||
        !music_pt_get(m->dav, "logs_proj.weight", wshape, 3, &lw, err) ||
        !music_pt_get(m->dav, "logs_proj.bias", bshape, 1, &lb, err)) return false;
    memset(&e->mean, 0, sizeof(e->mean));
    memset(&e->logs, 0, sizeof(e->logs));
    e->mean.I = e->logs.I = 1024; e->mean.O = e->logs.O = 64;
    e->mean.K = e->logs.K = 1; e->mean.stride = e->logs.stride = 1;
    e->mean.v = (const float *)mw->data; e->mean.b = (const float *)mb->data;
    e->logs.v = (const float *)lw->data; e->logs.b = (const float *)lb->data;
    (void)lb;
    return true;
}

static bool music_validate_configs(const char *root, cwick_error *err) {
    char p[1152];
#define CFG(dir, file, key, val)                                               \
    do {                                                                       \
        cwick_path_join(p, sizeof(p), root, dir);                              \
        cwick_path_join(p, sizeof(p), p, file);                                \
        if (!music_json_int_file(p, key, val, err)) return false;              \
    } while (0)
    cwick_path_join(p, sizeof(p), root, "config.json");
    if (!music_json_string_file(p, "model_type", "minimax_music3", err)) return false;
    CFG("language_model", "config.json", "num_hidden_layers", 36);
    CFG("language_model", "config.json", "hidden_size", 4096);
    CFG("language_model", "config.json", "num_attention_heads", 32);
    CFG("language_model", "config.json", "num_key_value_heads", 8);
    CFG("language_model", "config.json", "head_dim", 128);
    CFG("language_model", "config.json", "intermediate_size", 12288);
    CFG("language_model", "config.json", "vocab_size", 200000);
    CFG("rvq_depth_decoder", "config.json", "num_layers", 4);
    CFG("rvq_depth_decoder", "config.json", "hidden_size", 4096);
    CFG("rvq_depth_decoder", "config.json", "num_attention_heads", 16);
    CFG("rvq_depth_decoder", "config.json", "intermediate_size", 6144);
    CFG("rvq_depth_decoder", "config.json", "audio_vocab_size", 1024);
    CFG("rvq_depth_decoder", "config.json", "num_codebooks", 8);
    CFG("condition_encoder", "config.json", "num_condition_layers", 8);
    CFG("condition_encoder", "config.json", "condition_hidden_dim", 4096);
    CFG("condition_encoder", "config.json", "out_dim", 2048);
    CFG("condition_encoder", "config.json", "input_sampling_rate", 24000);
    CFG("condition_encoder", "config.json", "input_hop_length", 960);
    CFG("condition_encoder", "config.json", "output_sampling_rate", 44100);
    CFG("condition_encoder", "config.json", "output_hop_length", 512);
    CFG("transformer", "config.json", "num_layers", 36);
    CFG("transformer", "config.json", "in_channels", 128);
    CFG("transformer", "config.json", "condition_dim", 2048);
    CFG("transformer", "config.json", "num_attention_heads", 32);
    CFG("transformer", "config.json", "attention_head_dim", 64);
    CFG("transformer", "config.json", "ff_inner_dim", 8192);
    CFG("transformer", "config.json", "rotary_dim", 32);
    CFG("transformer", "config.json", "fourier_embedding_dim", 256);
    CFG("vocoder", "config.json", "latent_channels", 128);
    CFG("vocoder", "config.json", "decoder_input_dim", 1024);
    CFG("vocoder", "config.json", "decoder_hidden_dim", 1536);
    CFG("vocoder", "config.json", "sampling_rate", 44100);
#undef CFG
    cwick_path_join(p, sizeof(p), root, "scheduler/scheduler_config.json");
    if (!music_json_string_file(p, "_class_name", "FlowMatchEulerDiscreteScheduler", err)) return false;
    if (!music_json_int_file(p, "shift", 1, err) || !music_json_int_file(p, "num_train_timesteps", 1, err)) return false;
    cwick_path_join(p, sizeof(p), root, "tokenizer/tokenizer_config.json");
    if (!music_json_string_file(p, "tokenizer_class", "Qwen2Tokenizer", err)) return false;
    return true;
}

static void music_conv_free(music_conv *c) {
    free(c->scale);
    free(c->kernel);
    c->scale = NULL;
    c->kernel = NULL;
}

void music_model_close(music_model *m) {
    if (!m) return;
    if (m->gpu_adopted) h3v_gpu_shutdown();
    if (m->dav) {
        for (int i = 0; i < m->dav_locked_count; i++) {
            if (!m->dav_locked || !m->dav_locked[i]) continue;
            const rt_tensor *t = rt_pt_at(m->dav, i);
            size_t bytes = (size_t)t->numel * rt_dtype_size(t->dtype);
            cwick_mem_unlock_range(t->data, bytes);
        }
        rt_pt_close(m->dav);
    }
    free(m->dav_locked);
    if (m->lm_open) cwick_safetensors_close(&m->lm_saf);
    if (m->depth_open) cwick_safetensors_close(&m->depth_saf);
    if (m->cond_open) cwick_safetensors_close(&m->cond_saf);
    if (m->dit_open) cwick_safetensors_close(&m->dit_saf);
    if (m->voc_open) cwick_safetensors_close(&m->voc_saf);
    if (m->tok) cwick_tokenizer_free(m->tok);
    for (int i = 0; i < MUSIC_LM_LAYERS; i++) {
        free(m->lm.layer[i].input_norm); free(m->lm.layer[i].post_norm);
        free(m->lm.layer[i].q_norm); free(m->lm.layer[i].k_norm);
    }
    free(m->lm.norm);
    for (int i = 0; i < MUSIC_DEPTH_LAYERS; i++) {
        free(m->depth.layer[i].input_norm); free(m->depth.layer[i].post_norm);
    }
    free(m->depth.norm_weight);
    free(m->condition.layer_logits);
    music_conv_free(&m->vocoder.in_proj); music_conv_free(&m->vocoder.conv_in);
    music_conv_free(&m->vocoder.conv_out);
    for (int i = 0; i < 4; i++) {
        music_conv_free(&m->vocoder.block[i].up);
        for (int j = 0; j < 3; j++) {
            music_conv_free(&m->vocoder.block[i].unit[j].c1);
            music_conv_free(&m->vocoder.block[i].unit[j].c2);
        }
    }
    music_conv_free(&m->encoder.input); music_conv_free(&m->encoder.final);
    music_conv_free(&m->encoder.mean); music_conv_free(&m->encoder.logs);
    for (int i = 0; i < 4; i++) {
        music_conv_free(&m->encoder.down[i]);
        for (int j = 0; j < 6; j++) music_conv_free(&m->encoder.unit[i][j]);
    }
    pthread_mutex_destroy(&m->lock);
    free(m->root);
    free(m);
}

const char *music_model_dir(const music_model *m) { return m ? m->root : NULL; }
bool music_model_ready(const music_model *m) { return m && m->ready; }

music_model *music_model_open(const char *root, cwick_error *err) {
    cwick_error_clear(err);
    if (!root || !*root) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "music model root is empty");
        return NULL;
    }
    double t0 = music_ms();
    music_model *m = (music_model *)cwick_xmalloc(sizeof(*m));
    memset(m, 0, sizeof(*m));
    m->root = cwick_strdup(root);
    pthread_mutex_init(&m->lock, NULL);
    if (!music_validate_configs(root, err)) goto fail;
    char *dir = music_join(root, "language_model");
    if (cwick_safetensors_open(dir, &m->lm_saf, err) != CWICK_OK) { free(dir); goto fail; }
    m->lm_open = true; free(dir);
    dir = music_join(root, "rvq_depth_decoder");
    if (cwick_safetensors_open(dir, &m->depth_saf, err) != CWICK_OK) { free(dir); goto fail; }
    m->depth_open = true; free(dir);
    dir = music_join(root, "condition_encoder");
    if (cwick_safetensors_open(dir, &m->cond_saf, err) != CWICK_OK) { free(dir); goto fail; }
    m->cond_open = true; free(dir);
    dir = music_join(root, "transformer");
    if (cwick_safetensors_open(dir, &m->dit_saf, err) != CWICK_OK) { free(dir); goto fail; }
    m->dit_open = true; free(dir);
    dir = music_join(root, "vocoder");
    if (cwick_safetensors_open(dir, &m->voc_saf, err) != CWICK_OK) { free(dir); goto fail; }
    m->voc_open = true; free(dir);
    dir = music_join(root, "dav.pth");
    if (rt_pt_open(&m->dav, dir) != 0) {
        free(dir); cwick_error_set(err, CWICK_ERR_IO, "cannot open dav.pth"); goto fail;
    }
    free(dir);
    if (!music_validate_lm(m, err) || !music_validate_depth(m, err) ||
        !music_validate_condition(m, err) || !music_validate_dit(m, err) ||
        !music_validate_vocoder(m, err) || !music_validate_encoder(m, err)) goto fail;

    cwick_model_spec spec;
    memset(&spec, 0, sizeof(spec));
    snprintf(spec.model_type, sizeof(spec.model_type), "%s", "qwen3");
    spec.vocab_size = MUSIC_LM_VOCAB;
    dir = music_join(root, "tokenizer");
    if (cwick_tokenizer_load(dir, &spec, &m->tok, err) != CWICK_OK) { free(dir); goto fail; }
    free(dir);
    const int64_t required_markers[8] = {151644, 151645, MUSIC_AUDIO_CFG, 151669,
                                         MUSIC_AUDIO_END, 151671, 151673, 151674};
    unsigned markers = 0;
    for (size_t i = 0; i < m->tok->special_count; i++)
        for (int j = 0; j < 8; j++)
            if (m->tok->special[i].id == required_markers[j]) markers |= 1u << j;
    if (markers != 0xffu) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "music tokenizer is missing audio control tokens");
        goto fail;
    }

    cwick_safetensors *sets[5] = {&m->lm_saf, &m->depth_saf, &m->cond_saf, &m->dit_saf, &m->voc_saf};
    const char *names[5] = {"language_model", "rvq_depth_decoder", "condition_encoder", "transformer", "vocoder"};
    uint64_t wired = 0;
    for (int i = 0; i < 5; i++) {
        for (size_t j = 0; j < sets[i]->map_count; j++)
            madvise(sets[i]->maps[j], sets[i]->map_lens[j], MADV_NORMAL);
        uint64_t got = cwick_safetensors_make_resident(sets[i], NULL, 1);
        wired += got;
        CWICK_INFO("music", "%s mapped read-only: %.2f GiB, %s", names[i],
                   (double)got / 1073741824.0,
                   sets[i]->pinned_wired ? "resident and wired" : "resident (wiring unavailable)");
    }
    m->dav_locked_count = rt_pt_count(m->dav);
    m->dav_locked = (bool *)calloc((size_t)m->dav_locked_count, sizeof(bool));
    if (!m->dav_locked) goto fail;
    for (int i = 0; i < m->dav_locked_count; i++) {
        const rt_tensor *t = rt_pt_at(m->dav, i);
        size_t bytes = (size_t)t->numel * rt_dtype_size(t->dtype);
        m->dav_locked[i] = cwick_mem_lock_range(t->data, bytes, "music reference encoder");
        if (m->dav_locked[i]) wired += bytes;
    }
    if (h3v_gpu_on()) {
        for (int i = 0; i < 5; i++)
            for (size_t j = 0; j < sets[i]->map_count; j++)
                h3v_gpu_adopt(sets[i]->maps[j], sets[i]->map_lens[j]);
        m->gpu_adopted = true;
    }
    m->load_ms = music_ms() - t0;
    m->ready = true;
    CWICK_INFO("music", "model ready: %.2f GiB wired, load %.0f ms", (double)wired / 1073741824.0, m->load_ms);
    return m;
fail:
    music_model_close(m);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Math and sampling                                                   */
/* ------------------------------------------------------------------ */

static void music_linear_f32(float *out, const float *x, int64_t rows, int64_t k,
                             int64_t cols, const float *w, const float *bias) {
    if (rows <= 0 || cols <= 0) return;
    if (h3v_gpu_on() && rows >= 32 && k % 16 == 0 &&
        rows * k * cols >= 8 * 1024 * 1024 &&
        h3v_gpu_gemm_f32(x, w, out, (int)rows, (int)k, (int)cols,
                         (int)k, (int)cols)) {
        if (bias)
            for (int64_t r = 0; r < rows; r++)
                for (int64_t c = 0; c < cols; c++) out[r * cols + c] += bias[c];
        return;
    }
    cwick_gemm_f32(out, rows, k, cols, x, w);
    if (bias)
        for (int64_t r = 0; r < rows; r++)
            for (int64_t c = 0; c < cols; c++) out[r * cols + c] += bias[c];
}

static void music_linear_bf16_batch(const music_mat *w, const float *x,
                                    float *out, int64_t rows, int64_t k,
                                    int64_t cols, int copies) {
    if (copies <= 0) return;
    if (copies <= 4 && h3v_gpu_on() && k % 16 == 0 &&
        rows * k * cols >= 2 * 1024 * 1024) {
        h3v_gpu_bf16_item it[4];
        for (int i = 0; i < copies; i++)
            it[i] = (h3v_gpu_bf16_item){x, w->p, out + (int64_t)i * rows * cols,
                                        (int)rows, (int)k, (int)cols,
                                        (int)k, (int)cols};
        if (h3v_gpu_gemm_bf16_batch(it, copies)) return;
    }
    for (int i = 0; i < copies; i++)
        cwick_gemm_bf16(out + (int64_t)i * rows * cols, rows, k, cols, x, w->p);
}

static void music_bf16_group(const music_mat *w, const float *x, float *out,
                             int64_t rows, int64_t k, int n) {
    if (n <= 0) return;
    int64_t out_at = 0;
    if (n <= 4 && h3v_gpu_on() && k % 16 == 0 &&
        rows * k * w[0].cols >= 2 * 1024 * 1024) {
        h3v_gpu_bf16_item it[4];
        for (int i = 0; i < n; i++) {
            it[i] = (h3v_gpu_bf16_item){x, w[i].p, out + out_at,
                                        (int)rows, (int)k, (int)w[i].rows,
                                        (int)k, (int)w[i].rows};
            out_at += rows * w[i].rows;
        }
        if (h3v_gpu_gemm_bf16_batch(it, n)) return;
        out_at = 0;
    }
    for (int i = 0; i < n; i++) {
        cwick_gemm_bf16(out + out_at, rows, k, w[i].rows, x, w[i].p);
        out_at += rows * w[i].rows;
    }
}

static void music_rms_rows(float *out, const float *x, const float *weight,
                           int64_t rows, int64_t d, float eps) {
    for (int64_t r = 0; r < rows; r++)
        cwick_rms_norm(out + r * d, x + r * d, d, weight, eps);
}

static void music_rms_heads(float *x, const float *weight, int64_t rows,
                            int64_t heads, int64_t hd, float eps) {
    for (int64_t r = 0; r < rows; r++)
        for (int64_t h = 0; h < heads; h++)
            cwick_rms_norm(x + r * heads * hd + h * hd,
                           x + r * heads * hd + h * hd, hd, weight, eps);
}

static void music_layernorm(float *out, const float *x, const float *w,
                            const float *b, int64_t rows, int64_t d) {
    for (int64_t r = 0; r < rows; r++) {
        const float *src = x + r * d;
        float *dst = out + r * d;
        double mean = 0.0;
        for (int64_t i = 0; i < d; i++) mean += src[i];
        mean /= (double)d;
        double var = 0.0;
        for (int64_t i = 0; i < d; i++) {
            double q = src[i] - mean;
            var += q * q;
        }
        float inv = (float)(1.0 / sqrt(var / (double)d + 1e-5));
        for (int64_t i = 0; i < d; i++) dst[i] = (src[i] - (float)mean) * inv * w[i] + b[i];
    }
}

static void music_rope(float *x, int64_t rows, int64_t heads, int64_t hd,
                       int64_t pos0, int64_t rotary, double theta) {
    int half = (int)(rotary / 2);
    for (int64_t r = 0; r < rows; r++) {
        double pos = (double)(pos0 + r);
        for (int64_t h = 0; h < heads; h++) {
            float *v = x + r * heads * hd + h * hd;
            for (int i = 0; i < half; i++) {
                double inv = pow(theta, -(2.0 * i) / rotary);
                double ang = pos * inv;
                float cs = (float)cos(ang), sn = (float)sin(ang);
                float a = v[i], b = v[i + half];
                v[i] = a * cs - b * sn;
                v[i + half] = b * cs + a * sn;
            }
        }
    }
}

static void music_attention_cpu(float *out, const float *q, const float *k,
                                const float *v, int64_t qrows, int64_t krows,
                                int64_t heads, int64_t hd, bool causal,
                                float *scores) {
    const int64_t ld = heads * hd;
    const float scale = 1.0f / sqrtf((float)hd);
    for (int64_t h = 0; h < heads; h++) {
        const float *qh = q + h * hd, *kh = k + h * hd, *vh = v + h * hd;
        float *oh = out + h * hd;
        for (int64_t r0 = 0; r0 < qrows; r0 += 256) {
            int64_t rn = qrows - r0 < 256 ? qrows - r0 : 256;
            int64_t cols = causal ? r0 + rn : krows;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        (int)rn, (int)cols, (int)hd, scale,
                        qh + r0 * ld, (int)ld, kh, (int)ld,
                        0.0f, scores, (int)cols);
            for (int64_t r = 0; r < rn; r++) {
                int64_t valid = causal ? r0 + r + 1 : krows;
                for (int64_t j = valid; j < cols; j++) scores[r * cols + j] = -INFINITY;
                cwick_softmax(scores + r * cols, valid);
                for (int64_t j = valid; j < cols; j++) scores[r * cols + j] = 0.0f;
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        (int)rn, (int)hd, (int)cols, 1.0f,
                        scores, (int)cols, vh, (int)ld,
                        0.0f, oh + r0 * ld, (int)ld);
        }
    }
}

static uint64_t music_splitmix64(uint64_t *s) {
    uint64_t z = (*s += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

static float music_random_normal(uint64_t *state, bool *second, float *saved) {
    if (*second) { *second = false; return *saved; }
    double u1 = (double)(music_splitmix64(state) >> 11) * 0x1.0p-53;
    double u2 = (double)(music_splitmix64(state) >> 11) * 0x1.0p-53;
    if (u1 < 1e-15) u1 = 1e-15;
    double r = sqrt(-2.0 * log(u1));
    *saved = (float)(r * sin(2.0 * M_PI * u2));
    *second = true;
    return (float)(r * cos(2.0 * M_PI * u2));
}

static uint32_t music_cfg_pick(cwick_sampler *sampler, const float *cond,
                               const float *uncond, int64_t n, float cfg,
                               int conditional_topk, float *guided) {
    for (int64_t i = 0; i < n; i++) {
        float a = isfinite(cond[i]) ? cond[i] : -1e9f;
        float b = isfinite(uncond[i]) ? uncond[i] : -1e9f;
        guided[i] = b + (a - b) * cfg;
    }
    if (conditional_topk > MUSIC_AR_TOPK) conditional_topk = MUSIC_AR_TOPK;
    if (conditional_topk > 0 && conditional_topk < n) {
        float heap[MUSIC_AR_TOPK];
        int size = 0;
        for (int64_t i = 0; i < n; i++) {
            float x = cond[i];
            if (!isfinite(x)) continue;
            if (size < conditional_topk) {
                int p = size++;
                heap[p] = x;
                while (p > 0) {
                    int parent = (p - 1) / 2;
                    if (heap[parent] <= heap[p]) break;
                    float t = heap[parent]; heap[parent] = heap[p]; heap[p] = t;
                    p = parent;
                }
            } else if (x > heap[0]) {
                int p = 0;
                heap[0] = x;
                for (;;) {
                    int child = p * 2 + 1;
                    if (child >= size) break;
                    if (child + 1 < size && heap[child + 1] < heap[child]) child++;
                    if (heap[p] <= heap[child]) break;
                    float t = heap[p]; heap[p] = heap[child]; heap[child] = t;
                    p = child;
                }
            }
        }
        float threshold = size ? heap[0] : -INFINITY;
        for (int64_t i = 0; i < n; i++) if (cond[i] < threshold) guided[i] = -INFINITY;
    }
    return cwick_sampler_pick(sampler, guided, n);
}

/* ------------------------------------------------------------------ */
/* Prompt contract                                                     */
/* ------------------------------------------------------------------ */

typedef struct { char *p; size_t n, cap; } music_str;

static void music_str_add(music_str *b, const char *s, size_t n) {
    if (b->n + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 256;
        while (cap < b->n + n + 1) cap *= 2;
        b->p = (char *)cwick_xrealloc(b->p, cap);
        b->cap = cap;
    }
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = 0;
}

static void music_str_char(music_str *b, char c) { music_str_add(b, &c, 1); }

static bool music_line_is_rule(const char *s, size_t n) {
    size_t k = 0;
    while (k < n && (s[k] == ' ' || s[k] == '\t')) k++;
    if (k == n || (s[k] != '-' && s[k] != '*' && s[k] != '_')) return false;
    char c = s[k++];
    int count = 1;
    while (k < n && (s[k] == c || s[k] == ' ' || s[k] == '\t')) {
        if (s[k] == c) count++;
        k++;
    }
    return count >= 3 && k == n;
}

static char *music_clean_caption(const char *input) {
    music_str tag = {0}, all = {0};
    size_t n = strlen(input);
    for (size_t i = 0; i < n;) {
        if (i + 2 < n && input[i] == '<' && input[i + 1] == '|') {
            size_t j = i + 2;
            while (j + 1 < n && !(input[j] == '|' && input[j + 1] == '>')) j++;
            if (j + 1 < n) {
                size_t a = i + 2, b = j;
                while (a < b && (input[a] == ' ' || input[a] == '\t')) a++;
                while (b > a && (input[b - 1] == ' ' || input[b - 1] == '\t')) b--;
                size_t sp = a;
                while (sp < b && input[sp] != ' ' && input[sp] != '\t') sp++;
                music_str_add(&tag, input + a, sp - a);
                if (sp < b) {
                    size_t q = sp;
                    while (q < b && (input[q] == ' ' || input[q] == '\t')) q++;
                    music_str_add(&tag, " is ", 4);
                    music_str_add(&tag, input + q, b - q);
                }
                i = j + 2;
                continue;
            }
        }
        music_str_char(&tag, input[i++]);
    }
    n = tag.n;
    size_t i = 0;
    while (i < n) {
        size_t j = i;
        while (j < n && tag.p[j] != '\n') j++;
        const char *line = tag.p + i;
        size_t len = j - i, start = 0;
        while (start < len && (line[start] == ' ' || line[start] == '\t')) start++;
        size_t hashes = 0;
        while (start + hashes < len && line[start + hashes] == '#') hashes++;
        if (hashes >= 1 && hashes <= 6 && start + hashes < len &&
            (line[start + hashes] == ' ' || line[start + hashes] == '\t')) {
            start += hashes;
            while (start < len && (line[start] == ' ' || line[start] == '\t')) start++;
        } else if (start < len && (line[start] == '*' || line[start] == '+' || line[start] == '-') &&
                   start + 1 < len && (line[start + 1] == ' ' || line[start + 1] == '\t')) {
            start += 2;
        }
        if (music_line_is_rule(line + start, len - start)) {
            if (j < n) music_str_char(&all, '\n');
        } else {
            music_str add = {0};
            for (size_t k = start; k < len;) {
                if (k + 1 < len && line[k] == '*' && line[k + 1] == '*') {
                    size_t e = k + 2;
                    while (e + 1 < len && !(line[e] == '*' && line[e + 1] == '*')) e++;
                    if (e + 1 < len) { music_str_add(&add, line + k + 2, e - k - 2); k = e + 2; continue; }
                }
                if (line[k] == '*' && k + 1 < len && line[k + 1] != '*' &&
                    k > start && line[k - 1] != '*') {
                    size_t e = k + 1;
                    while (e < len && line[e] != '*') e++;
                    if (e < len) { music_str_add(&add, line + k + 1, e - k - 1); k = e + 1; continue; }
                }
                music_str_char(&add, line[k++]);
            }
            while (add.n && (add.p[add.n - 1] == ' ' || add.p[add.n - 1] == '\t' ||
                             add.p[add.n - 1] == '\r')) add.p[--add.n] = 0;
            for (size_t k = 0; k < add.n;) {
                if (k + 3 < add.n && add.p[k] == ' ' && add.p[k + 1] == ' ' &&
                    add.p[k + 2] == ' ' && add.p[k + 3] == ' ') {
                    k += 4;
                    continue;
                }
                music_str_char(&all, add.p[k++]);
            }
            free(add.p);
            if (j < n) music_str_char(&all, '\n');
        }
        i = j < n ? j + 1 : j;
    }
    free(tag.p);
    memset(&tag, 0, sizeof(tag));
    for (size_t k = 0; k < all.n; k++) {
        if (k + 2 < all.n && all.p[k] == '\xe2' && all.p[k + 1] == '\x80' && all.p[k + 2] == '\xa2') {
            if (k + 3 < all.n && all.p[k + 3] == ' ') k += 3;
            else k += 2;
            continue;
        }
        music_str_char(&tag, all.p[k]);
    }
    free(all.p);
    music_str out = {0};
    for (size_t k = 0; k < tag.n;) {
        if (tag.p[k] == '\n') {
            size_t e = k;
            while (e < tag.n && tag.p[e] == '\n') e++;
            if (out.n && e < tag.n) music_str_char(&out, '\n');
            k = e;
        } else music_str_char(&out, tag.p[k++]);
    }
    free(tag.p);
    if (!out.p) out.p = cwick_strdup("");
    return out.p;
}

/* The checkpoint documents a closed set of single-word lyric tags: [Intro],
 * [Verse], [Pre-Chorus], [Chorus], [Post-Chorus], [Bridge], [Instrumental],
 * [Solo] and [Outro]. The format-understanding model emits numbered variants
 * ("[Verse 1]", "[Chorus 2]") and compound forms ("[Instrumental Drop 1]"),
 * which collapse onto their documented tag here. It also emits arrangement
 * annotations as bracketed phrases ("[Synth arpeggio and pads]") next to a
 * section tag; those are not sections, so the line is dropped instead of being
 * sent as a bogus tag. A single-token unknown tag is kept verbatim: the
 * checkpoint's own reference lyrics use ad-hoc tags such as
 * "[bass-quartet-rumbles-in]", and a multi-word rule keeps those while
 * removing the prose annotations. */
static const char *music_canonical_tag(const char *name, size_t len, bool *multiword) {
    static const char *const tags[] = {
        "intro", "verse", "pre-chorus", "chorus", "post-chorus",
        "bridge", "instrumental", "solo", "outro", NULL
    };
    char buf[96];
    size_t n = 0;
    bool gap = false;
    for (size_t i = 0; i < len && n + 2 < sizeof(buf); i++) {
        unsigned char c = (unsigned char)name[i];
        if (c == ' ' || c == '\t' || c == '_' || c == '\r') { gap = n > 0; continue; }
        if (c >= 'A' && c <= 'Z') c = (unsigned char)(c - 'A' + 'a');
        if (gap) { buf[n++] = ' '; gap = false; }
        buf[n++] = (char)c;
    }
    while (n && (buf[n - 1] == ' ' || buf[n - 1] == '.' || buf[n - 1] == ':')) n--;
    buf[n] = 0;
    /* Drop a trailing index: "verse 1" and "chorus2" both become "verse"/"chorus". */
    size_t m = n;
    while (m && buf[m - 1] >= '0' && buf[m - 1] <= '9') m--;
    while (m && buf[m - 1] == ' ') m--;
    if (m && m != n) { n = m; buf[n] = 0; }
    if (multiword) {
        bool spaced = false;
        for (size_t i = 0; i < n; i++) if (buf[i] == ' ') spaced = true;
        *multiword = spaced;
    }
    for (int i = 0; tags[i]; i++)
        if (!strcmp(buf, tags[i])) return tags[i];
    /* First token: "instrumental drop", "instrumental break". */
    for (size_t i = 0; i < n; i++) {
        if (buf[i] != ' ') continue;
        buf[i] = 0;
        for (int t = 0; tags[t]; t++)
            if (!strcmp(buf, tags[t])) return tags[t];
        buf[i] = ' ';
        break;
    }
    /* Spaced spelling of a hyphenated tag: "pre chorus", "post chorus". */
    for (size_t i = 0; i < n; i++) if (buf[i] == ' ') buf[i] = '-';
    for (int i = 0; tags[i]; i++)
        if (!strcmp(buf, tags[i])) return tags[i];
    return NULL;
}

static char *music_normalize_lyrics(const char *input) {
    music_str out = {0};
    music_str_add(&out, "[start]\n", 8);
    size_t n = strlen(input), i = 0;
    while (i < n) {
        size_t j = i;
        while (j < n && input[j] != '\n') j++;
        const char *line = input + i;
        size_t len = j - i, k = 0;
        while (k < len && (line[k] == ' ' || line[k] == '\t')) k++;
        if (k + 3 <= len && line[k] == '[' && line[k + 3] == ']' &&
            ((line[k + 1] >= 'a' && line[k + 1] <= 'z') ||
             (line[k + 1] >= 'A' && line[k + 1] <= 'Z')) &&
            ((line[k + 2] >= 'a' && line[k + 2] <= 'z') ||
             (line[k + 2] >= 'A' && line[k + 2] <= 'Z'))) {
            k += 4;
            while (k < len && (line[k] == ' ' || line[k] == '\t')) k++;
            line += k;
            len -= k;
            k = 0;
        }
        music_str tags = {0};
        bool saw_tag = false, kept_tag = false;
        while (k < len && line[k] == '[') {
            size_t e = k + 1;
            while (e < len && line[e] != ']') e++;
            if (e == len) break;
            saw_tag = true;
            bool multiword = false;
            const char *canon = music_canonical_tag(line + k + 1, e - k - 1, &multiword);
            size_t a = k + 1, b = e;
            while (a < b && (line[a] == ' ' || line[a] == '\t')) a++;
            while (b > a && (line[b - 1] == ' ' || line[b - 1] == '\t')) b--;
            if (canon || !multiword) {
                if (kept_tag) music_str_char(&tags, ' ');
                music_str_char(&tags, '[');
                if (canon) {
                    music_str_add(&tags, canon, strlen(canon));
                } else {
                    for (size_t q = a; q < b; q++) {
                        unsigned char c = (unsigned char)line[q];
                        if (c >= 'A' && c <= 'Z') c = (unsigned char)(c - 'A' + 'a');
                        music_str_char(&tags, (char)c);
                    }
                }
                music_str_char(&tags, ']');
                kept_tag = true;
            }
            k = e + 1;
            while (k < len && (line[k] == ' ' || line[k] == '\t')) k++;
        }
        bool only_tags = true;
        for (size_t q = k; q < len; q++)
            if (line[q] != ' ' && line[q] != '\t') only_tags = false;
        if (tags.n) {
            music_str_add(&out, tags.p, tags.n);
            /* The checkpoint's own contract drops whatever follows a leading
             * tag on the same line. Keeping it costs nothing and stops a
             * caller who writes "[Verse] one line of lyric" from losing the
             * whole song text (which reads back as an instrumental). */
            if (!only_tags) {
                music_str_char(&out, '\n');
                for (size_t q = k; q < len; q++) music_str_char(&out, line[q]);
            }
            if (j < n) music_str_char(&out, '\n');
        } else if (saw_tag && only_tags) {
            /* Every tag on this line was an annotation; drop the whole line so
             * it leaves no empty stanza marker behind. */
        } else {
            for (size_t q = 0; q < len; q++) {
                if (q + 1 < len && line[q] == ']' && line[q + 1] == ' ') {
                    music_str_char(&out, ']'); music_str_char(&out, '\n'); q++;
                } else if (q + 1 < len && line[q] == ' ' && line[q + 1] == '[') {
                    music_str_char(&out, '\n'); q++;
                } else if (q + 2 < len && line[q] == ' ' && line[q + 1] == '^' && line[q + 2] == ' ') {
                    music_str_char(&out, '\n'); q += 2;
                } else music_str_char(&out, line[q]);
            }
            if (j < n) music_str_char(&out, '\n');
        }
        free(tags.p);
        i = j < n ? j + 1 : j;
    }
    bool in_tag = false;
    for (size_t k = 0; out.p && k < out.n; k++) {
        if (out.p[k] == '[') in_tag = true;
        else if (out.p[k] == ']') in_tag = false;
        else if (in_tag && out.p[k] >= 'A' && out.p[k] <= 'Z') out.p[k] += 'a' - 'A';
    }
    if (!out.p) out.p = cwick_strdup("");
    return out.p;
}

bool music_lyrics_are_instrumental(const char *lyrics) {
    if (!lyrics) return false;
    while (*lyrics == ' ' || *lyrics == '\t' || *lyrics == '\r' || *lyrics == '\n') lyrics++;
    size_t n = strlen(lyrics);
    while (n && (lyrics[n - 1] == ' ' || lyrics[n - 1] == '\t' ||
                 lyrics[n - 1] == '\r' || lyrics[n - 1] == '\n')) n--;
    static const char tag[] = "[instrumental]";
    if (n != sizeof(tag) - 1) return false;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)lyrics[i];
        if (c >= 'A' && c <= 'Z') c = (unsigned char)(c - 'A' + 'a');
        if (c != (unsigned char)tag[i]) return false;
    }
    return true;
}

bool music_has_sung_text(const char *lyrics) {
    if (!lyrics) return false;
    const char *p = lyrics;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
        if (!*p) break;
        if (*p == '[') {
            const char *end = strchr(p, ']');
            if (end) { p = end + 1; continue; }
        }
        return true;
    }
    return false;
}

static uint32_t music_utf8_next(const unsigned char **cursor) {
    const unsigned char *p = *cursor;
    uint32_t cp = *p++;
    if (cp >= 0xc2 && cp <= 0xdf && (p[0] & 0xc0) == 0x80) {
        cp = ((cp & 0x1f) << 6) | (p[0] & 0x3f); p++;
    } else if (cp >= 0xe0 && cp <= 0xef && (p[0] & 0xc0) == 0x80 &&
               (p[1] & 0xc0) == 0x80) {
        cp = ((cp & 0x0f) << 12) | ((p[0] & 0x3f) << 6) | (p[1] & 0x3f); p += 2;
    } else if (cp >= 0xf0 && cp <= 0xf4 && (p[0] & 0xc0) == 0x80 &&
               (p[1] & 0xc0) == 0x80 && (p[2] & 0xc0) == 0x80) {
        cp = ((cp & 0x07) << 18) | ((p[0] & 0x3f) << 12) |
             ((p[1] & 0x3f) << 6) | (p[2] & 0x3f); p += 3;
    }
    *cursor = p;
    return cp;
}

void music_lyric_stats_measure(const char *lyrics, int seconds, music_lyric_stats *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!lyrics) return;
    const unsigned char *p = (const unsigned char *)lyrics;
    bool in_tag = false, in_word = false;
    while (*p) {
        if (*p == '[') { in_tag = true; in_word = false; p++; continue; }
        if (*p == ']') { in_tag = false; in_word = false; p++; continue; }
        uint32_t cp = music_utf8_next(&p);
        if (in_tag) continue;
        bool cjk = (cp >= 0x3400 && cp <= 0x9fff) ||
                   (cp >= 0xf900 && cp <= 0xfaff) ||
                   (cp >= 0x20000 && cp <= 0x323af);
        bool word = (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') ||
                    (cp >= '0' && cp <= '9');
        if (cjk) { out->cjk_chars++; in_word = false; }
        else if (word) {
            if (!in_word) out->latin_words++;
            in_word = true;
        } else in_word = false;
    }
    out->units = out->cjk_chars + out->latin_words;
    out->units_per_second = seconds > 0 ? (double)out->units / seconds : 0.0;
}

bool music_lyric_stats_selftest(void) {
    music_lyric_stats zh, en;
    music_lyric_stats_measure("[Verse]\n霓虹月光洒在地面", 2, &zh);
    music_lyric_stats_measure("[Verse]\nMorning light\nNeon moonlight", 2, &en);
    return zh.cjk_chars == 8 && zh.latin_words == 0 && zh.units == 8 &&
           zh.units_per_second == 4.0 &&
           en.cjk_chars == 0 && en.latin_words == 4 && en.units == 4 &&
           en.units_per_second == 2.0;
}

bool music_lyrics_selftest(void) {
    /* Shape of a real format-understanding result: numbered section tags,
     * a compound "[Instrumental Drop 1]", and arrangement annotations in
     * brackets that are not sections. */
    static const char *const sample =
        "[Intro]\n[Synth arpeggio and pads]\n\n[Verse 1]\nline one\n\n"
        "[Pre-Chorus]\nline two\n\n[Instrumental Drop 1]\n"
        "[Synth lead melody with heavy beat]\n\n[Bridge]\n"
        "[Synth arpeggio and pads]\n\n[Verse 2]\nline three\n";
    char *out = music_normalize_lyrics(sample);
    bool ok = out &&
              strstr(out, "[start]\n[intro]\n") != NULL &&
              strstr(out, "[verse]\nline one") != NULL &&
              strstr(out, "[pre-chorus]\nline two") != NULL &&
              strstr(out, "[instrumental]\n") != NULL &&
              strstr(out, "[bridge]\n\n[verse]\nline three") != NULL &&
              strstr(out, "synth") == NULL &&
              strstr(out, "verse 1") == NULL &&
              strstr(out, "drop") == NULL;
    free(out);

    /* Spaced/numbered spellings collapse, single-token ad-hoc tags survive. */
    static const char *const edge =
        "[Pre Chorus 2]\nline a\n[Chorus3]\nline b\n[Verse 10]\nline c\n"
        "[Whisper]\nline d\n[bass-quartet-rumbles-in]\nline e\n";
    char *e = music_normalize_lyrics(edge);
    ok = ok && e &&
         strstr(e, "[pre-chorus]\nline a") != NULL &&
         strstr(e, "[chorus]\nline b") != NULL &&
         strstr(e, "[verse]\nline c") != NULL &&
         strstr(e, "[whisper]\nline d") != NULL &&
         strstr(e, "[bass-quartet-rumbles-in]\nline e") != NULL;
    free(e);
    const char *language = "[Verse]\n[zh] 把你的眼泪从脸上擦去\n[en] Morning light";
    char *l = music_normalize_lyrics(language);
    ok = ok && l &&
         strstr(l, "[start]\n[verse]\n把你的眼泪从脸上擦去\nMorning light") != NULL &&
         strstr(l, "[zh]") == NULL && strstr(l, "[en]") == NULL;
    free(l);
    /* A tag that shares its line with lyric text keeps the text instead of the
     * checkpoint's "drop it" rule. */
    char *inline_tag = music_normalize_lyrics("[Chorus] line one\n[bridge]line two");
    ok = ok && inline_tag &&
         strstr(inline_tag, "[chorus]\nline one") != NULL &&
         strstr(inline_tag, "[bridge]\nline two") != NULL;
    free(inline_tag);
    ok = ok && music_lyrics_are_instrumental("  [INSTRUMENTAL]\r\n") &&
         !music_lyrics_are_instrumental("[Instrumental]\n[Verse]\nhello") &&
         music_has_sung_text("[Verse]\nhello") &&
         music_has_sung_text("[Verse] hello") &&
         !music_has_sung_text(NULL) &&
         !music_has_sung_text(" \n[Verse]\n[Instrumental]\n");
    return ok;
}

static bool music_tokenize(music_model *m, const music_request *req,
                           uint32_t **ids_out, int64_t *n_out, cwick_error *err) {
    if (!req->prompt || !*req->prompt || !req->lyrics || !*req->lyrics) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "prompt and lyrics are required");
        return false;
    }
    char *caption = music_clean_caption(req->prompt);
    char *lyrics = music_normalize_lyrics(req->lyrics);
    music_str text = {0};
    music_str_add(&text, "<|im_start|><|caption_start|>", strlen("<|im_start|><|caption_start|>"));
    music_str_add(&text, caption, strlen(caption));
    music_str_add(&text, "<|caption_end|><|lyrics_start|>", strlen("<|caption_end|><|lyrics_start|>"));
    music_str_add(&text, lyrics, strlen(lyrics));
    music_str_add(&text, "<|lyrics_end|><|im_end|><|audio_start|>", strlen("<|lyrics_end|><|im_end|><|audio_start|>"));
    free(caption); free(lyrics);
    uint32_t *ids = NULL;
    int64_t n = 0;
    cwick_status st = cwick_tokenizer_encode(m->tok, text.p, &ids, &n, err);
    free(text.p);
    if (st != CWICK_OK) return false;
    if (n <= 2 || n > MUSIC_MAX_PROMPT_TOKENS) {
        free(ids);
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "assembled music prompt has %lld tokens (maximum %d)",
                        (long long)n, MUSIC_MAX_PROMPT_TOKENS);
        return false;
    }
    *ids_out = ids; *n_out = n;
    return true;
}

/* ------------------------------------------------------------------ */
/* Global Qwen3 language model                                         */
/* ------------------------------------------------------------------ */

typedef struct {
    float *keys, *values;
    int64_t capacity, len;
} music_lm_cache;

static bool music_lm_cache_init(music_lm_cache *c, int64_t capacity) {
    memset(c, 0, sizeof(*c));
    int64_t per_layer = 2 * capacity * MUSIC_LM_KV_HEADS * MUSIC_LM_HEAD_DIM;
    c->keys = music_calloc_f32(MUSIC_LM_LAYERS * per_layer);
    c->values = music_calloc_f32(MUSIC_LM_LAYERS * per_layer);
    if (!c->keys || !c->values) { free(c->keys); free(c->values); return false; }
    c->capacity = capacity;
    return true;
}

static void music_lm_cache_free(music_lm_cache *c) {
    free(c->keys); free(c->values);
    memset(c, 0, sizeof(*c));
}

static float *music_embed_rows(const music_mat *table, const uint32_t *ids,
                               int64_t rows, int64_t cols) {
    float *out = music_f32(rows * cols);
    if (!out) return NULL;
    const uint16_t *w = (const uint16_t *)table->p;
    for (int64_t r = 0; r < rows; r++) {
        if ((int64_t)ids[r] < 0 || (int64_t)ids[r] >= table->rows) {
            free(out); return NULL;
        }
        const uint16_t *src = w + (int64_t)ids[r] * cols;
        float *dst = out + r * cols;
        for (int64_t j = 0; j < cols; j++) dst[j] = cwick_bf16_fast(src[j]);
    }
    return out;
}

static bool music_lm_forward(const music_model *m, music_lm_cache *cache,
                             const float *embeds, int B, int n, int pos0,
                             float *out, float *work, float *scores,
                             cwick_error *err) {
    const int D = MUSIC_LM_HIDDEN, KV = MUSIC_LM_KV_HEADS * MUSIC_LM_HEAD_DIM;
    const int H = MUSIC_LM_HEADS, KH = MUSIC_LM_KV_HEADS, HD = MUSIC_LM_HEAD_DIM;
    const bool legacy_qkv = getenv("CWICK_MUSIC_LM_QKV_LEGACY") != NULL;
    float *x = work;
    float *norm = x + B * n * D;
    float *q = norm + B * n * D;
    float *k = q + B * n * D;
    float *v = k + B * n * KV;
    float *att = v + B * n * KV;
    float *gate = att + B * n * D;
    float *up = gate + B * n * MUSIC_LM_INTER;
    float *tmp = up + B * n * MUSIC_LM_INTER;
    memcpy(x, embeds, (size_t)B * n * D * sizeof(float));
    int64_t rows = B * n;
    for (int l = 0; l < MUSIC_LM_LAYERS; l++) {
        const music_lm_layer *a = &m->lm.layer[l];
        music_rms_rows(norm, x, a->input_norm, rows, D, 1e-6f);
        if (legacy_qkv) {
            music_linear_bf16_batch(&a->q, norm, q, rows, D, D, 1);
            music_linear_bf16_batch(&a->k, norm, k, rows, D, KV, 1);
            music_linear_bf16_batch(&a->v, norm, v, rows, D, KV, 1);
        } else {
            const music_mat qkv[3] = {a->q, a->k, a->v};
            music_bf16_group(qkv, norm, q, rows, D, 3);
        }
        if (getenv("MUSIC_DEBUG") && l == 0) {
            int bq=0,bk=0,bv=0; for(int i=0;i<rows*D;i++)if(!isfinite(q[i]))bq++; for(int i=0;i<rows*KV;i++){if(!isfinite(k[i]))bk++;if(!isfinite(v[i]))bv++;}
            fprintf(stderr, "qkv q=%g k=%g v=%g norm=%g bad q%d k%d v%d\n",
                    q[(rows-1)*D], k[(rows-1)*KV], v[(rows-1)*KV],
                    norm[(rows-1)*D], bq, bk, bv);
        }
        music_rms_heads(q, a->q_norm, rows, H, HD, 1e-6f);
        music_rms_heads(k, a->k_norm, rows, KH, HD, 1e-6f);
        for (int b = 0; b < B; b++) {
            music_rope(q + b * n * D, n, H, HD, pos0, HD, 1000000.0);
            music_rope(k + b * n * KV, n, KH, HD, pos0, HD, 1000000.0);
        }
        int64_t cache_stride = cache->capacity * KV;
        for (int b = 0; b < B; b++) {
            float *kd = cache->keys + ((int64_t)l * B + b) * cache_stride + pos0 * KV;
            float *vd = cache->values + ((int64_t)l * B + b) * cache_stride + pos0 * KV;
            memcpy(kd, k + b * n * KV, (size_t)n * KV * sizeof(float));
            memcpy(vd, v + b * n * KV, (size_t)n * KV * sizeof(float));
        }
        for (int b = 0; b < B; b++) {
            const float *qb = q + b * n * D;
            const float *kb = cache->keys + ((int64_t)l * B + b) * cache_stride;
            const float *vb = cache->values + ((int64_t)l * B + b) * cache_stride;
            float *ab = att + b * n * D;
            for (int h = 0; h < H; h++) {
                int kh = h / (H / KH);
                const float *qh = qb + h * HD;
                const float *khp = kb + kh * HD;
                const float *vh = vb + kh * HD;
                float *oh = ab + h * HD;
                float scale = 1.0f / sqrtf((float)HD);
                for (int r0 = 0; r0 < n; r0 += 256) {
                    int rn = n - r0 < 256 ? n - r0 : 256;
                    int cols = pos0 + r0 + rn;
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                rn, cols, HD, scale, qh + r0 * D, D,
                                khp, KV, 0.0f, scores, cols);
                    for (int r = 0; r < rn; r++) {
                        int valid = pos0 + r0 + r + 1;
                        for (int j = valid; j < cols; j++) scores[r * cols + j] = -INFINITY;
                        cwick_softmax(scores + r * cols, valid);
                        for (int j = valid; j < cols; j++) scores[r * cols + j] = 0.0f;
                        if (getenv("MUSIC_DEBUG") && l == 0 && b == 0 && h == 0 && r0 == 0)
                            fprintf(stderr, "score r%d valid=%d s=%g q=%g k=%g\n", r, valid, scores[r * cols], qh[r * D], khp[0]);
                    }
                    if (getenv("MUSIC_DEBUG") && l == 0 && b == 0 && h == 0 && r0 == 0)
                        fprintf(stderr, "value cache=%g scores=%g,%g\n", vb[0], scores[0], scores[1]);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                rn, HD, cols, 1.0f, scores, cols,
                                vh, KV, 0.0f, oh + r0 * D, D);
                    if (getenv("MUSIC_DEBUG") && l == 0 && b == 0 && h == 0 && r0 == 0)
                        fprintf(stderr, "att after=%g\n", oh[0]);
                }
            }
        }
        music_linear_bf16_batch(&a->o, att, tmp, rows, D, D, 1);
        if (getenv("MUSIC_DEBUG") && l == 0) fprintf(stderr, "att=%g tmp=%g x=%g\n", att[(rows-1)*D], tmp[(rows-1)*D], x[(rows-1)*D]);
        for (int64_t i = 0; i < rows * D; i++) x[i] += tmp[i];
        music_rms_rows(norm, x, a->post_norm, rows, D, 1e-6f);
        const music_mat gu[2] = {a->gate, a->up};
        music_bf16_group(gu, norm, gate, rows, D, 2);
        if (getenv("MUSIC_DEBUG") && l == 0) fprintf(stderr, "gate=%g up=%g\n", gate[(rows-1)*MUSIC_LM_INTER], up[(rows-1)*MUSIC_LM_INTER]);
        for (int64_t i = 0; i < rows * MUSIC_LM_INTER; i++)
            gate[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i];
        music_linear_bf16_batch(&a->down, gate, tmp, rows, MUSIC_LM_INTER, D, 1);
        if (getenv("MUSIC_DEBUG") && l == 0) fprintf(stderr, "down=%g\n", tmp[(rows-1)*D]);
        for (int64_t i = 0; i < rows * D; i++) x[i] += tmp[i];
        if (getenv("MUSIC_DEBUG")) {
            for (int b = 0; b < B; b++) {
                int bad = 0;
                for (int t = 0; t < n; t++)
                    for (int j = 0; j < D; j++)
                        if (!isfinite(x[(b * n + t) * D + j])) bad++;
                fprintf(stderr, "layer %d b%d bad=%d first=%g\n", l, b, bad, x[(b * n + n - 1) * D]);
            }
        }
    }
    music_rms_rows(out, x, m->lm.norm, rows, D, 1e-6f);
    cache->len = pos0 + n;
    (void)err;
    return true;
}

/* ------------------------------------------------------------------ */
/* RVQ depth decoder and autoregressive frames                         */
/* ------------------------------------------------------------------ */

static bool music_depth_forward(const music_model *m, const float *sequence,
                                int B, int n, float *last, float *work,
                                float *scores) {
    const int D = MUSIC_LM_HIDDEN, H = MUSIC_DEPTH_HEADS, HD = D / H;
    float *x = work;
    float *norm = x + B * 8 * D;
    float *q = norm + B * 8 * D;
    float *k = q + B * 8 * D;
    float *v = k + B * 8 * D;
    float *att = v + B * 8 * D;
    float *gate = att + B * 8 * D;
    float *up = gate + B * 8 * MUSIC_DEPTH_INTER;
    float *tmp = up + B * 8 * MUSIC_DEPTH_INTER;
    const uint16_t *pos = (const uint16_t *)m->depth.pos_embedding.p;
    for (int b = 0; b < B; b++)
        for (int t = 0; t < n; t++)
            for (int j = 0; j < D; j++)
                x[(b * 8 + t) * D + j] = sequence[(b * 8 + t) * D + j] +
                                          cwick_bf16_fast(pos[t * D + j]);
    int64_t rows = B * n;
    for (int l = 0; l < MUSIC_DEPTH_LAYERS; l++) {
        const music_depth_layer *a = &m->depth.layer[l];
        for (int b = 0; b < B; b++)
            for (int t = 0; t < n; t++) {
                const float *src = x + (b * 8 + t) * D;
                float *dst = norm + (b * n + t) * D;
                cwick_rms_norm(dst, src, D, a->input_norm, 1e-6f);
            }
        const music_mat qkv[3] = {a->q, a->k, a->v};
        music_bf16_group(qkv, norm, q, rows, D, 3);
        for (int b = 0; b < B; b++)
            music_attention_cpu(att + b * n * D, q + b * n * D,
                                k + b * n * D, v + b * n * D,
                                n, n, H, HD, true, scores);
        music_linear_bf16_batch(&a->o, att, tmp, rows, D, D, 1);
        for (int b = 0; b < B; b++)
            for (int t = 0; t < n; t++)
                for (int j = 0; j < D; j++)
                    x[(b * 8 + t) * D + j] += tmp[(b * n + t) * D + j];
        for (int b = 0; b < B; b++)
            for (int t = 0; t < n; t++) {
                const float *src = x + (b * 8 + t) * D;
                float *dst = norm + (b * n + t) * D;
                cwick_rms_norm(dst, src, D, a->post_norm, 1e-6f);
            }
        const music_mat gu[2] = {a->gate, a->up};
        music_bf16_group(gu, norm, gate, rows, D, 2);
        for (int64_t i = 0; i < rows * MUSIC_DEPTH_INTER; i++)
            gate[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i];
        music_linear_bf16_batch(&a->down, gate, tmp, rows, MUSIC_DEPTH_INTER, D, 1);
        for (int b = 0; b < B; b++)
            for (int t = 0; t < n; t++)
                for (int j = 0; j < D; j++)
                    x[(b * 8 + t) * D + j] += tmp[(b * n + t) * D + j];
    }
    for (int b = 0; b < B; b++) {
        const float *src = x + (b * 8 + n - 1) * D;
        cwick_rms_norm(last + b * D, src, D, m->depth.norm_weight, 1e-6f);
    }
    return true;
}

static float *music_embed_pair(const music_mat *table, uint32_t id) {
    uint32_t ids[2] = {id, id};
    return music_embed_rows(table, ids, 2, table->cols);
}

static bool music_semantic_generate(const music_model *m, const music_request *req,
                                    const uint32_t *prompt_ids, int64_t prompt_n,
                                    int max_frames, float **frames_out,
                                    int64_t *frames_n, double *global_ms,
                                    double *depth_ms, cwick_error *err) {
    const int D = MUSIC_LM_HIDDEN;
    const bool full_head = getenv("CWICK_MUSIC_FULL_HEAD") != NULL;
    const int head_rows = full_head ? MUSIC_LM_VOCAB : MUSIC_AUDIO_HEAD_ROWS;
    music_mat audio_head = m->lm.head;
    audio_head.rows = head_rows;
    audio_head.p = full_head ? m->lm.head.p :
                   (const uint16_t *)m->lm.head.p + (int64_t)MUSIC_AUDIO_END * D;
    int64_t capacity = prompt_n + max_frames + 2;
    music_lm_cache cache;
    if (!music_lm_cache_init(&cache, capacity)) {
        cwick_error_set(err, CWICK_ERR_MEMORY, "cannot allocate global LM KV cache");
        return false;
    }
    int chunk = MUSIC_LM_PREFILL;
    /* q is full-width; the GQA k/v projections need their own narrower arenas. */
    int64_t lm_work_n = 2LL * chunk * (7 * D + 2 * MUSIC_LM_KV_HEADS * MUSIC_LM_HEAD_DIM +
                                      2 * MUSIC_LM_INTER);
    float *lm_work = music_f32(lm_work_n);
    float *lm_result = music_f32(2LL * chunk * D);
    float *lm_scores = music_f32((int64_t)chunk * capacity);
    float *depth_work = music_f32(2LL * 8 * (7 * D + 2 * MUSIC_DEPTH_INTER));
    float *depth_scores = music_f32(8 * 8);
    float *last_hidden = music_f32(2 * D);
    float *logits = music_f32(2LL * head_rows);
    float *guided = music_f32(head_rows);
    float *depth_last = music_f32(2 * D);
    float *depth_logits = music_f32(2LL * MUSIC_DEPTH_VOCAB);
    float *depth_guided = music_f32(MUSIC_DEPTH_VOCAB);
    float *sequence = music_f32(2LL * 8 * D);
    float *frame_hiddens = music_f32((int64_t)max_frames * MUSIC_COND_LAYERS * D);
    uint32_t *uncond = NULL;
    bool ok;
    if (!lm_work || !lm_result || !lm_scores || !depth_work || !depth_scores ||
        !last_hidden || !logits || !guided || !depth_last || !depth_logits ||
        !depth_guided || !sequence || !frame_hiddens) {
        cwick_error_set(err, CWICK_ERR_MEMORY, "cannot allocate autoregressive workspaces");
        goto fail;
    }
    uncond = (uint32_t *)cwick_xmalloc((size_t)prompt_n * sizeof(uint32_t));
    memcpy(uncond, prompt_ids, (size_t)prompt_n * sizeof(uint32_t));
    for (int64_t i = 1; i < prompt_n - 2; i++) uncond[i] = MUSIC_AUDIO_CFG;

    double global_acc = 0.0, depth_acc = 0.0;
    music_progress(req, MUSIC_PHASE_GLOBAL, 0, max_frames);
    CWICK_INFO("music", "phase global_lm start: prompt=%lld frames<=%d kv=%lld head_rows=%d qkv=%s",
               (long long)prompt_n, max_frames, (long long)capacity, head_rows,
               getenv("CWICK_MUSIC_LM_QKV_LEGACY") ? "legacy" : "grouped");
    for (int64_t p0 = 0; p0 < prompt_n; p0 += chunk) {
        if (!music_check_cancel(req, err)) goto fail;
        int n = (int)(prompt_n - p0 < chunk ? prompt_n - p0 : chunk);
        uint32_t ids[2 * MUSIC_LM_PREFILL];
        for (int b = 0; b < 2; b++)
            for (int t = 0; t < n; t++)
                ids[b * n + t] = b ? uncond[p0 + t] : prompt_ids[p0 + t];
        float *emb = music_embed_rows(&m->lm.embed, ids, 2LL * n, D);
        if (!emb) { free(emb); cwick_error_set(err, CWICK_ERR_CONTRACT, "prompt token outside vocabulary"); goto fail; }
        double a = music_ms();
        ok = music_lm_forward(m, &cache, emb, 2, n, (int)p0,
                              lm_result, lm_work, lm_scores, err);
        global_acc += music_ms() - a;
        free(emb);
        if (!ok) goto fail;
        if (!music_audio_valid(lm_result, 2LL * n * D, "global LM prefill", err)) goto fail;
        memcpy(last_hidden, lm_result + (n - 1) * D, D * sizeof(float));
        memcpy(last_hidden + D, lm_result + (2LL * n - 1) * D, D * sizeof(float));
        music_progress(req, MUSIC_PHASE_GLOBAL, p0 + n, prompt_n + max_frames);
    }
    free(uncond); uncond = NULL;

    cwick_sampler_cfg scfg;
    cwick_sampler_cfg_default(&scfg);
    scfg.temperature = req->temperature;
    scfg.top_k = req->top_k > 0 ? req->top_k : MUSIC_AR_TOPK;
    scfg.top_p = 1.0f;
    char why[128];
    if (!cwick_sampler_cfg_validate(&scfg, why, sizeof(why))) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "invalid music sampler: %s", why);
        goto fail;
    }
    uint64_t seed = req->seed ? req->seed : UINT64_C(0x4d5553494333);
    cwick_sampler global_sampler, depth_sampler;
    cwick_sampler_init(&global_sampler, &scfg, head_rows, seed);
    uint64_t depth_seed = music_splitmix64(&seed) | 1;
    cwick_sampler_init(&depth_sampler, &scfg, MUSIC_DEPTH_VOCAB, depth_seed);
    global_sampler.lon = global_sampler.lbrk = false;
    depth_sampler.lon = depth_sampler.lbrk = false;

    int64_t made = 0;
    bool audio_end = false;
    for (int frame = 0; frame <= max_frames; frame++) {
        if (!music_check_cancel(req, err)) goto sampler_fail;
        double a = music_ms();
        /* `last_hidden` already holds the two conditional rows. One GEMM
         * produces both logit rows; passing copies=2 would write twice the
         * allocated buffer and corrupt the sampler's heap on the first pick. */
        music_linear_bf16_batch(&audio_head, last_hidden, logits,
                                2, D, head_rows, 1);
        for (int b = 0; b < 2; b++) {
            float *row = logits + b * head_rows;
            if (full_head) {
                for (int i = 0; i < MUSIC_LM_VOCAB; i++) {
                    bool allowed = (i >= MUSIC_AUDIO_OFFSET &&
                                    i < MUSIC_AUDIO_OFFSET + MUSIC_SEMANTIC_VOCAB) ||
                                   (i == MUSIC_AUDIO_END && music_early_end_enabled());
                    if (!allowed) row[i] = -INFINITY;
                }
            } else {
                for (int i = 1; i < MUSIC_AUDIO_HEAD_GAP; i++) row[i] = -INFINITY;
                if (!music_early_end_enabled()) row[0] = -INFINITY;
            }
        }
        uint32_t picked = music_cfg_pick(&global_sampler, logits,
                                         logits + head_rows, head_rows,
                                         MUSIC_AR_CFG, MUSIC_AR_TOPK, guided);
        uint32_t sampled;
        if (full_head) sampled = picked;
        else if (picked == 0) sampled = MUSIC_AUDIO_END;
        else if (picked >= MUSIC_AUDIO_HEAD_GAP) sampled = music_audio_head_token(picked);
        else {
            cwick_error_set(err, CWICK_ERR_STATE,
                            "global LM sampled a non-audio head row %u", picked);
            goto sampler_fail;
        }
        global_acc += music_ms() - a;
        if (sampled == MUSIC_AUDIO_END) {
            audio_end = true;
            break;
        }
        if (sampled < MUSIC_AUDIO_OFFSET || sampled >= MUSIC_AUDIO_OFFSET + MUSIC_SEMANTIC_VOCAB) {
            cwick_error_set(err, CWICK_ERR_STATE, "global LM sampled an invalid audio token %u", sampled);
            goto sampler_fail;
        }
        uint32_t semantic = sampled - MUSIC_AUDIO_OFFSET;
        uint32_t codes[MUSIC_DEPTH_CODEBOOKS + 1];
        codes[0] = semantic;
        a = music_ms();
        music_progress(req, MUSIC_PHASE_DEPTH, made, max_frames);
        /* The depth decoder reads rows at (b*8+t)*D, so the two batch rows
         * must be scattered instead of written contiguously. */
        for (int b = 0; b < 2; b++)
            music_linear_bf16_batch(&m->depth.projection, last_hidden + b * D,
                                    sequence + (int64_t)b * 8 * D, 1, D, D, 1);
        float *sem_emb = music_embed_pair(&m->lm.embed, sampled);
        if (!sem_emb) { cwick_error_set(err, CWICK_ERR_MEMORY, "semantic embedding allocation failed"); goto sampler_fail; }
        float *projected = music_f32(2 * D);
        if (!projected) { free(sem_emb); cwick_error_set(err, CWICK_ERR_MEMORY, "depth projection allocation failed"); goto sampler_fail; }
        music_linear_bf16_batch(&m->depth.projection, sem_emb, projected, 2, D, D, 1);
        free(sem_emb);
        for (int b = 0; b < 2; b++)
            memcpy(sequence + ((int64_t)b * 8 + 1) * D, projected + b * D,
                   (size_t)D * sizeof(float));
        free(projected);
        int seq_n = 2;
        for (int g = 0; g < MUSIC_DEPTH_CODEBOOKS; g++) {
            if (!music_check_cancel(req, err)) goto sampler_fail;
            if (!music_depth_forward(m, sequence, 2, seq_n, depth_last,
                                     depth_work, depth_scores)) {
                cwick_error_set(err, CWICK_ERR_STATE, "depth decoder failed"); goto sampler_fail;
            }
            if (!music_audio_valid(depth_last, 2 * D, "depth hidden", err)) goto sampler_fail;
            memcpy(frame_hiddens + ((int64_t)made * MUSIC_COND_LAYERS + 1 + g) * D,
                   depth_last, D * sizeof(float));
            music_linear_bf16_batch(&m->depth.audio_heads[g], depth_last,
                                    depth_logits, 2, D, MUSIC_DEPTH_VOCAB, 1);
            uint32_t code = music_cfg_pick(&depth_sampler, depth_logits,
                                            depth_logits + MUSIC_DEPTH_VOCAB,
                                            MUSIC_DEPTH_VOCAB, MUSIC_AR_CFG, 0,
                                            depth_guided);
            codes[g + 1] = code;
            if (g + 1 < MUSIC_DEPTH_CODEBOOKS) {
                float *extra = music_embed_pair(&m->depth.audio_embeddings,
                                                 code + (uint32_t)g * MUSIC_DEPTH_VOCAB);
                projected = music_f32(2 * D);
                if (!extra || !projected) {
                    free(extra); free(projected);
                    cwick_error_set(err, CWICK_ERR_MEMORY, "depth embedding allocation failed");
                    goto sampler_fail;
                }
                music_linear_bf16_batch(&m->depth.projection, extra, projected, 2, D, D, 1);
                free(extra);
                for (int b = 0; b < 2; b++)
                    memcpy(sequence + (b * 8 + seq_n) * D, projected + b * D,
                           D * sizeof(float));
                free(projected);
                seq_n++;
            }
        }
        if (frame > 0) {
            memcpy(frame_hiddens + (int64_t)made * MUSIC_COND_LAYERS * D,
                   last_hidden, D * sizeof(float));
            made++;
            depth_acc += music_ms() - a;
            music_progress(req, MUSIC_PHASE_DEPTH, made, max_frames);
            if ((made % 100) == 0)
                CWICK_INFO("music", "phase depth: frames=%lld/%d elapsed=%.0f ms",
                           (long long)made, max_frames, global_acc + depth_acc);
            if (made >= max_frames) break;
        } else {
            depth_acc += music_ms() - a;
        }
        float *feedback = music_f32(2 * D);
        if (!feedback) { cwick_error_set(err, CWICK_ERR_MEMORY, "feedback allocation failed"); goto sampler_fail; }
        float *e = music_embed_pair(&m->lm.embed, sampled);
        if (!e) { free(feedback); cwick_error_set(err, CWICK_ERR_MEMORY, "feedback embedding allocation failed"); goto sampler_fail; }
        memcpy(feedback, e, 2LL * D * sizeof(float));
        free(e);
        for (int g = 0; g < MUSIC_DEPTH_CODEBOOKS; g++) {
            e = music_embed_pair(&m->depth.audio_embeddings,
                                 codes[g + 1] + (uint32_t)g * MUSIC_DEPTH_VOCAB);
            if (!e) { free(feedback); cwick_error_set(err, CWICK_ERR_MEMORY, "feedback embedding allocation failed"); goto sampler_fail; }
            for (int i = 0; i < 2 * D; i++) feedback[i] += e[i];
            free(e);
        }
        float inv = 1.0f / sqrtf((float)(MUSIC_DEPTH_CODEBOOKS + 1));
        for (int i = 0; i < 2 * D; i++) feedback[i] *= inv;
        a = music_ms();
        ok = music_lm_forward(m, &cache, feedback, 2, 1, (int)(prompt_n + frame),
                              lm_result, lm_work, lm_scores, err);
        global_acc += music_ms() - a;
        free(feedback);
        if (!ok) goto sampler_fail;
        if (!music_audio_valid(lm_result, 2 * D, "global LM hidden", err)) goto sampler_fail;
        memcpy(last_hidden, lm_result, 2LL * D * sizeof(float));
        memcpy(last_hidden + D, lm_result + D, D * sizeof(float));
        music_progress(req, MUSIC_PHASE_GLOBAL, prompt_n + frame + 1, prompt_n + max_frames);
    }
    if (made == 0) {
        cwick_error_set(err, CWICK_ERR_STATE, "music model generated zero audio frames");
        goto sampler_fail;
    }
    *frames_out = frame_hiddens;
    *frames_n = made;
    *global_ms = global_acc;
    *depth_ms = depth_acc;
    CWICK_INFO("music", "phase global_lm done: frames=%lld/%d prompt=%lld ended_early=%s "
                         "stop=%s global=%.0f ms depth=%.0f ms",
               (long long)made, max_frames, (long long)prompt_n,
               made < max_frames ? "yes" : "no", audio_end ? "audio_end" : "frame_limit",
               global_acc, depth_acc);
    cwick_sampler_free(&global_sampler); cwick_sampler_free(&depth_sampler);
    free(uncond); free(lm_work); free(lm_result); free(lm_scores);
    free(depth_work); free(depth_scores); free(last_hidden); free(logits);
    free(guided); free(depth_last); free(depth_logits); free(depth_guided);
    free(sequence); music_lm_cache_free(&cache);
    return true;
sampler_fail:
    cwick_sampler_free(&global_sampler); cwick_sampler_free(&depth_sampler);
fail:
    free(uncond); free(lm_work); free(lm_result); free(lm_scores);
    free(depth_work); free(depth_scores); free(last_hidden); free(logits);
    free(guided); free(depth_last); free(depth_logits); free(depth_guided);
    free(sequence); free(frame_hiddens); music_lm_cache_free(&cache);
    return false;
}

/* ------------------------------------------------------------------ */
/* Hidden-state conditioning and flow transformer                      */
/* ------------------------------------------------------------------ */

static int64_t music_latent_frames(int64_t frames) {
    if (frames <= 0) return 0;
    double n = (double)frames * 44100.0 / 24000.0 * 960.0 / 512.0;
    int64_t v = (int64_t)n;
    return v > 0 ? v : 1;
}

bool music_latent_frames_selftest(void) {
    return music_latent_frames(0) == 0 && music_latent_frames(25) == 86;
}

static bool music_audio_valid(const float *audio, int64_t count,
                              const char *stage, cwick_error *err);

bool music_audio_valid_selftest(void) {
    const float good[2] = {0.01f, -0.02f};
    const float silent[2] = {0.0f, 0.0f};
    const float invalid[2] = {NAN, 0.0f};
    cwick_error err;
    return music_audio_valid(good, 2, "selftest", &err) &&
           !music_audio_valid(silent, 2, "selftest", &err) &&
           !music_audio_valid(invalid, 2, "selftest", &err);
}

bool music_flow_layout_selftest(void) {
    enum { T = 3, N = 3 * MUSIC_LATENT_CH };
    float storage[2 * N + 1];
    for (int i = 0; i < N; i++) storage[i] = (float)(i + 1);
    storage[2 * N] = 12345.0f;
    memcpy(music_batch_second(storage, T), storage, N * sizeof(float));
    bool ok = music_batch_second(storage, T) == storage + N && storage[2 * N] == 12345.0f;
    for (int i = 0; i < N && ok; i++) ok = storage[N + i] == storage[i];
    return ok;
}

bool music_flow_scheduler_selftest(void) {
    /* t = 0 is pure noise; the last step is (steps-1)/steps and each Euler step
     * advances sigma by +1/steps, matching invert_sigmas. */
    return fabs(music_flow_time(0, 30) - 0.0) < 1e-12 &&
           fabs(music_flow_time(29, 30) - 29.0 / 30.0) < 1e-12 &&
           fabs(music_flow_step_size(30) - 1.0f / 30.0f) < 1e-7f;
}

bool music_ar_stop_selftest(void) {
    return music_ar_min_frames(1500) == 375 &&
           music_ar_min_frames(625) == 157 &&
           music_ar_min_frames(25) == 7;
}

bool music_audio_head_selftest(void) {
    enum { K = 32, ROWS = 12, FIRST = 5, BATCH = 2 };
    uint16_t weights[ROWS * K];
    float x[BATCH * K], full[BATCH * ROWS], compact[BATCH * (ROWS - FIRST)];
    for (int i = 0; i < ROWS * K; i++)
        weights[i] = cwick_f32_to_bf16(((i * 17) % 29 - 14) * 0.03125f);
    for (int i = 0; i < BATCH * K; i++) x[i] = ((i * 11) % 23 - 11) * 0.0625f;
    music_mat full_head = {.p = weights, .rows = ROWS, .cols = K, .f32 = false};
    music_mat audio_head = {.p = weights + FIRST * K, .rows = ROWS - FIRST,
                            .cols = K, .f32 = false};
    music_linear_bf16_batch(&full_head, x, full, BATCH, K, ROWS, 1);
    music_linear_bf16_batch(&audio_head, x, compact, BATCH, K, ROWS - FIRST, 1);
    bool same = MUSIC_AUDIO_HEAD_GAP == FIRST &&
                MUSIC_AUDIO_HEAD_ROWS == 16389 &&
                music_audio_head_token(0) == MUSIC_AUDIO_END &&
                music_audio_head_token(MUSIC_AUDIO_HEAD_GAP) == MUSIC_AUDIO_OFFSET &&
                music_audio_head_token(MUSIC_AUDIO_HEAD_ROWS - 1) ==
                    MUSIC_AUDIO_OFFSET + MUSIC_SEMANTIC_VOCAB - 1;
    for (int b = 0; b < BATCH && same; b++)
        for (int r = 0; r < ROWS - FIRST; r++)
            if (fabsf(full[b * ROWS + FIRST + r] - compact[b * (ROWS - FIRST) + r]) > 1e-5f)
                same = false;
    return same;
}

bool music_qkv_group_selftest(void) {
    enum { K = 32, Q = 32, KV = 16, BATCH = 2 };
    uint16_t qw[Q * K], kw[KV * K], vw[KV * K];
    float x[BATCH * K], separate[BATCH * (Q + 2 * KV)];
    float grouped[BATCH * (Q + 2 * KV)];
    for (int i = 0; i < Q * K; i++)
        qw[i] = cwick_f32_to_bf16(((i * 7) % 31 - 15) * 0.03125f);
    for (int i = 0; i < KV * K; i++) {
        kw[i] = cwick_f32_to_bf16(((i * 11) % 29 - 14) * 0.03125f);
        vw[i] = cwick_f32_to_bf16(((i * 13) % 27 - 13) * 0.03125f);
    }
    for (int i = 0; i < BATCH * K; i++) x[i] = ((i * 5) % 19 - 9) * 0.0625f;
    music_mat weights[3] = {
        {.p = qw, .rows = Q, .cols = K, .f32 = false},
        {.p = kw, .rows = KV, .cols = K, .f32 = false},
        {.p = vw, .rows = KV, .cols = K, .f32 = false},
    };
    music_linear_bf16_batch(&weights[0], x, separate, BATCH, K, Q, 1);
    music_linear_bf16_batch(&weights[1], x, separate + BATCH * Q, BATCH, K, KV, 1);
    music_linear_bf16_batch(&weights[2], x, separate + BATCH * (Q + KV), BATCH, K, KV, 1);
    music_bf16_group(weights, x, grouped, BATCH, K, 3);
    for (int i = 0; i < BATCH * (Q + 2 * KV); i++)
        if (fabsf(separate[i] - grouped[i]) > 1e-5f) return false;
    return true;
}

static float *music_condition_encode(const music_model *m,
                                     const float *frames, int64_t n_frames,
                                     int64_t *n_latent_out) {
    const int D = MUSIC_LM_HIDDEN, C = MUSIC_COND_DIM;
    int64_t n_latent = music_latent_frames(n_frames);
    float *mixed = music_f32(n_frames * D);
    float *patches = music_f32(n_frames * D * 3);
    float *projected = music_f32(n_frames * C);
    float *condition = music_f32(n_latent * C);
    if (!mixed || !patches || !projected || !condition) goto fail;
    float weights[MUSIC_COND_LAYERS], sum = 0.0f;
    for (int l = 0; l < MUSIC_COND_LAYERS; l++) {
        weights[l] = expf(m->condition.layer_logits[l]);
        sum += weights[l];
    }
    for (int l = 0; l < MUSIC_COND_LAYERS; l++) weights[l] /= sum;
    for (int64_t t = 0; t < n_frames; t++)
        for (int j = 0; j < D; j++) {
            float v = 0.0f;
            for (int l = 0; l < MUSIC_COND_LAYERS; l++)
                v += weights[l] * frames[(t * MUSIC_COND_LAYERS + l) * D + j];
            mixed[t * D + j] = v * m->condition.layer_scale;
        }
    for (int64_t t = 0; t < n_frames; t++)
        for (int i = 0; i < D; i++)
            for (int k = 0; k < 3; k++) {
                int64_t src = t + k - 1;
                patches[t * D * 3 + i * 3 + k] =
                    src >= 0 && src < n_frames ? mixed[src * D + i] : 0.0f;
            }
    music_linear_f32(projected, patches, n_frames, D * 3, C,
                     m->condition.proj_w, m->condition.proj_b);
    for (int64_t t = 0; t < n_latent; t++) {
        int64_t src = (int64_t)((long double)t * n_frames / n_latent);
        if (src >= n_frames) src = n_frames - 1;
        memcpy(condition + t * C, projected + src * C, C * sizeof(float));
    }
    free(mixed); free(patches); free(projected);
    *n_latent_out = n_latent;
    return condition;
fail:
    free(mixed); free(patches); free(projected); free(condition);
    return NULL;
}

typedef struct {
    float *input, *x, *norm, *q, *k, *v, *att, *ff, *tmp, *body, *out;
    float *temb, *fourier, *scores;
    int64_t capacity;
} music_dit_work;

static void music_dit_work_free(music_dit_work *w) {
    free(w->input); free(w->x); free(w->norm); free(w->q); free(w->k);
    free(w->v); free(w->att); free(w->ff); free(w->tmp); free(w->body);
    free(w->out); free(w->temb); free(w->fourier); free(w->scores);
    memset(w, 0, sizeof(*w));
}

static bool music_dit_work_init(music_dit_work *w, int64_t T) {
    memset(w, 0, sizeof(*w));
    const int64_t B = 2, S = T + 1, D = MUSIC_DIT_WIDTH;
    w->input = music_f32(B * T * MUSIC_DIT_IN);
    w->x = music_f32(B * S * D);
    w->norm = music_f32(B * S * D);
    w->q = music_f32(B * S * D); w->k = music_f32(B * S * D);
    w->v = music_f32(B * S * D); w->att = music_f32(B * S * D);
    w->ff = music_f32(B * S * 2 * MUSIC_DIT_FF);
    w->tmp = music_f32(B * S * D);
    w->body = music_f32(B * T * D);
    w->out = music_f32(B * T * MUSIC_LATENT_CH);
    w->temb = music_f32(D); w->fourier = music_f32(256);
    w->scores = music_f32(S * S);
    if (!w->input || !w->x || !w->norm || !w->q || !w->k || !w->v ||
        !w->att || !w->ff || !w->tmp || !w->body || !w->out ||
        !w->temb || !w->fourier || !w->scores) {
        music_dit_work_free(w);
        return false;
    }
    w->capacity = T;
    return true;
}

static bool music_dit_forward(const music_model *m, const float *latents,
                              const float *condition, int64_t T, double time,
                              music_dit_work *w) {
    const int64_t B = 2, S = T + 1, D = MUSIC_DIT_WIDTH;
    const int64_t rows_t = B * T, rows_s = B * S;
    for (int64_t b = 0; b < B; b++)
        for (int64_t t = 0; t < T; t++) {
            float *dst = w->input + (b * T + t) * MUSIC_DIT_IN;
            const float *src = latents + (b * T + t) * MUSIC_LATENT_CH;
            const float *cond = condition + t * MUSIC_COND_DIM;
            memcpy(dst, src, MUSIC_LATENT_CH * sizeof(float));
            memset(dst + MUSIC_LATENT_CH, 0, MUSIC_LATENT_CH * sizeof(float));
            if (b == 0)
                memcpy(dst + 2 * MUSIC_LATENT_CH, cond, MUSIC_COND_DIM * sizeof(float));
            else
                memset(dst + 2 * MUSIC_LATENT_CH, 0, MUSIC_COND_DIM * sizeof(float));
        }
    music_linear_f32(w->ff, w->input, rows_t, MUSIC_DIT_IN, MUSIC_DIT_IN,
                     (const float *)m->dit.preprocess.p, NULL);
    for (int64_t i = 0; i < rows_t * MUSIC_DIT_IN; i++) w->input[i] += w->ff[i];
    music_linear_f32(w->body, w->input, rows_t, MUSIC_DIT_IN, D,
                     (const float *)m->dit.proj_in.p, NULL);
    for (int i = 0; i < 128; i++) {
        double angle = 2.0 * M_PI * time * m->dit.time_proj[i];
        w->fourier[i] = (float)cos(angle);
        w->fourier[128 + i] = (float)sin(angle);
    }
    music_linear_f32(w->norm, w->fourier, 1, 256, D, m->dit.time1_w, m->dit.time1_b);
    for (int i = 0; i < D; i++) w->norm[i] /= 1.0f + expf(-w->norm[i]);
    music_linear_f32(w->temb, w->norm, 1, D, D, m->dit.time2_w, m->dit.time2_b);
    for (int64_t b = 0; b < B; b++) {
        memcpy(w->x + b * S * D, w->temb, D * sizeof(float));
        memcpy(w->x + b * S * D + D, w->body + b * T * D,
               (size_t)T * D * sizeof(float));
    }
    for (int l = 0; l < MUSIC_DIT_LAYERS; l++) {
        const music_dit_layer *a = &m->dit.layer[l];
        music_layernorm(w->norm, w->x, a->norm1_w, a->norm1_b, rows_s, D);
        music_linear_f32(w->q, w->norm, rows_s, D, D, (const float *)a->q.p, NULL);
        music_linear_f32(w->k, w->norm, rows_s, D, D, (const float *)a->k.p, NULL);
        music_linear_f32(w->v, w->norm, rows_s, D, D, (const float *)a->v.p, NULL);
        for (int b = 0; b < B; b++) {
            music_rope(w->q + b * S * D, S, MUSIC_DIT_HEADS,
                       MUSIC_DIT_HEAD_DIM, 0, MUSIC_DIT_ROTARY, 10000.0);
            music_rope(w->k + b * S * D, S, MUSIC_DIT_HEADS,
                       MUSIC_DIT_HEAD_DIM, 0, MUSIC_DIT_ROTARY, 10000.0);
        }
        for (int64_t b = 0; b < B; b++) {
            float *qb = w->q + b * S * D, *kb = w->k + b * S * D;
            float *vb = w->v + b * S * D, *ab = w->att + b * S * D;
            if (!h3v_gpu_on() ||
                !h3v_gpu_attention(qb, kb, vb, ab, (int)S, (int)D,
                                   MUSIC_DIT_HEADS, MUSIC_DIT_HEAD_DIM))
                music_attention_cpu(ab, qb, kb, vb, S, S, MUSIC_DIT_HEADS,
                                    MUSIC_DIT_HEAD_DIM, false, w->scores);
        }
        music_linear_f32(w->tmp, w->att, rows_s, D, D, (const float *)a->o.p, NULL);
        for (int64_t i = 0; i < rows_s * D; i++) w->x[i] += w->tmp[i];
        music_layernorm(w->norm, w->x, a->norm2_w, a->norm2_b, rows_s, D);
        music_linear_f32(w->ff, w->norm, rows_s, D, 2 * MUSIC_DIT_FF,
                         (const float *)a->ff_in.p, a->ff_in_b);
        for (int64_t r = 0; r < rows_s; r++)
            for (int64_t i = 0; i < MUSIC_DIT_FF; i++) {
                float first = w->ff[r * 2 * MUSIC_DIT_FF + i];
                float second = w->ff[r * 2 * MUSIC_DIT_FF + MUSIC_DIT_FF + i];
                w->ff[r * 2 * MUSIC_DIT_FF + i] =
                    first * second / (1.0f + expf(-second));
            }
        for (int64_t r = 0; r < rows_s; r++)
            memmove(w->ff + r * MUSIC_DIT_FF,
                    w->ff + r * 2 * MUSIC_DIT_FF,
                    (size_t)MUSIC_DIT_FF * sizeof(float));
        music_linear_f32(w->tmp, w->ff, rows_s, MUSIC_DIT_FF, D,
                         (const float *)a->ff_out.p, a->ff_out_b);
        for (int64_t i = 0; i < rows_s * D; i++) w->x[i] += w->tmp[i];
    }
    for (int64_t b = 0; b < B; b++)
        memcpy(w->body + b * T * D, w->x + b * S * D + D,
               (size_t)T * D * sizeof(float));
    music_linear_f32(w->out, w->body, rows_t, D, MUSIC_LATENT_CH,
                     (const float *)m->dit.proj_out.p, NULL);
    music_linear_f32(w->q, w->out, rows_t, MUSIC_LATENT_CH, MUSIC_LATENT_CH,
                     (const float *)m->dit.postprocess.p, NULL);
    for (int64_t i = 0; i < rows_t * MUSIC_LATENT_CH; i++) w->out[i] += w->q[i];
    return true;
}

static bool music_audio_valid(const float *audio, int64_t count,
                              const char *stage, cwick_error *err) {
    float peak = 0.0f;
    int64_t bad = -1;
    for (int64_t i = 0; i < count; i++) {
        if (!isfinite(audio[i])) { bad = i; break; }
        float a = fabsf(audio[i]);
        if (a > peak) peak = a;
    }
    if (bad >= 0) {
        cwick_error_set(err, CWICK_ERR_STATE, "%s produced non-finite sample at index %lld",
                        stage, (long long)bad);
        return false;
    }
    if (!(peak > 1e-7f)) {
        cwick_error_set(err, CWICK_ERR_STATE, "%s produced digital silence", stage);
        return false;
    }
    return true;
}

static bool music_flow_generate(const music_model *m, const music_request *req,
                                const float *frame_hiddens, int64_t n_frames,
                                const float *reference, int64_t reference_n,
                                float ***chunks_out, int64_t **chunk_lens_out,
                                int64_t *n_chunks_out, int64_t *steps_out,
                                double *condition_ms, double *flow_ms,
                                cwick_error *err) {
    const int chunk_frames = n_frames <= MUSIC_SINGLE_CHUNK_FRAMES
                                 ? (int)n_frames : MUSIC_CHUNK_FRAMES;
    int n_chunks = n_frames <= chunk_frames ? 1 :
                   (int)((n_frames - MUSIC_CHUNK_HOP + MUSIC_CHUNK_HOP - 1) / MUSIC_CHUNK_HOP);
    int max_chunk = chunk_frames;
    int64_t max_latent = music_latent_frames(max_chunk);
    music_dit_work work;
    if (!music_dit_work_init(&work, max_latent)) {
        cwick_error_set(err, CWICK_ERR_MEMORY, "cannot allocate flow transformer workspace");
        return false;
    }
    float **chunks = (float **)cwick_xmalloc((size_t)n_chunks * sizeof(float *));
    int64_t *lens = (int64_t *)cwick_xmalloc((size_t)n_chunks * sizeof(int64_t));
    memset(chunks, 0, (size_t)n_chunks * sizeof(float *));
    float *previous = music_f32(MUSIC_CARRY_LATENT * MUSIC_LATENT_CH);
    float *previous_cond = music_f32(MUSIC_CARRY_LATENT * MUSIC_COND_DIM);
    float *noise_prompt = music_f32(MUSIC_CARRY_LATENT * MUSIC_LATENT_CH);
    float *latents = music_f32(2LL * max_latent * MUSIC_LATENT_CH);
    if (!previous || !previous_cond || !noise_prompt || !latents) {
        cwick_error_set(err, CWICK_ERR_MEMORY, "cannot allocate flow chunk state"); goto fail;
    }
    int64_t previous_n = 0, total_steps = (int64_t)n_chunks * req->steps;
    int64_t step_done = 0;
    double cond_acc = 0.0, flow_acc = 0.0;
    uint64_t rng = req->seed ? req->seed : UINT64_C(0x4d5553494333);
    rng = music_splitmix64(&rng) | 1;
    bool normal_second = false;
    float normal_saved = 0.0f;
    music_progress(req, MUSIC_PHASE_CONDITION, 0, n_chunks);
    CWICK_INFO("music", "phase condition start: frames=%lld chunks=%d", (long long)n_frames, n_chunks);
    for (int c = 0; c < n_chunks; c++) {
        if (!music_check_cancel(req, err)) goto fail;
        int64_t start = n_frames <= chunk_frames ? 0 : (int64_t)c * MUSIC_CHUNK_HOP;
        int64_t end = start + chunk_frames;
        if (end > n_frames) end = n_frames;
        int64_t nf = end - start;
        int64_t T = music_latent_frames(nf);
        double a = music_ms();
        float *condition = music_condition_encode(m, frame_hiddens + start * MUSIC_COND_LAYERS * MUSIC_LM_HIDDEN,
                                                   nf, &T);
        cond_acc += music_ms() - a;
        if (!condition) { cwick_error_set(err, CWICK_ERR_MEMORY, "condition encoder allocation failed"); goto fail; }
        if (!music_audio_valid(condition, T * MUSIC_COND_DIM, "condition hidden", err)) {
            free(condition); goto fail;
        }
        int64_t overlap = previous_n < T ? previous_n : T;
        if (overlap > 0)
            memcpy(condition, previous_cond, (size_t)overlap * MUSIC_COND_DIM * sizeof(float));
        music_progress(req, MUSIC_PHASE_CONDITION, c + 1, n_chunks);
        for (int b = 0; b < 2; b++)
            for (int64_t i = 0; i < T * MUSIC_LATENT_CH; i++)
                latents[(b * T + i)] = music_random_normal(&rng, &normal_second, &normal_saved);
        if (overlap > 0)
            memcpy(noise_prompt, latents, (size_t)overlap * MUSIC_LATENT_CH * sizeof(float));
        music_progress(req, MUSIC_PHASE_FLOW, step_done, total_steps);
        for (int step = 0; step < req->steps; step++) {
            if (!music_check_cancel(req, err)) { free(condition); goto fail; }
            double t = music_flow_time(step, req->steps);
            if (overlap > 0) {
                float nt = (float)(1.0 - (1.0 - 1e-6) * t);
                for (int64_t i = 0; i < overlap * MUSIC_LATENT_CH; i++)
                    latents[i] = nt * noise_prompt[i] + (float)t * previous[i];
            }
            memcpy(music_batch_second(latents, T), latents,
                   (size_t)T * MUSIC_LATENT_CH * sizeof(float));
            a = music_ms();
            music_dit_forward(m, latents, condition, T, t, &work);
            float dt = music_flow_step_size(req->steps);
            for (int64_t i = 0; i < T * MUSIC_LATENT_CH; i++) {
                float cond = work.out[i];
                float uncond = music_batch_second(work.out, T)[i];
                float velocity = uncond + (cond - uncond) * req->cfg_scale;
                latents[i] += dt * velocity;
            }
            flow_acc += music_ms() - a;
            step_done++;
            music_progress(req, MUSIC_PHASE_FLOW, step_done, total_steps);
        }
        if (!music_audio_valid(latents, T * MUSIC_LATENT_CH, "flow latent", err)) {
            free(condition); goto fail;
        }
        if (overlap > 0)
            memcpy(latents, previous, (size_t)overlap * MUSIC_LATENT_CH * sizeof(float));
        if (reference) {
            int64_t ref_start = music_latent_frames(start);
            int64_t use = reference_n - ref_start;
            if (use > T) use = T;
            if (use > 0) {
                double generated_ss = 0.0, reference_ss = 0.0;
                for (int64_t i = 0; i < use * MUSIC_LATENT_CH; i++) {
                    generated_ss += (double)latents[i] * latents[i];
                    reference_ss += (double)reference[ref_start * MUSIC_LATENT_CH + i] *
                                    reference[ref_start * MUSIC_LATENT_CH + i];
                }
                float generated_rms = sqrtf((float)(generated_ss / (use * MUSIC_LATENT_CH)));
                float reference_rms = sqrtf((float)(reference_ss / (use * MUSIC_LATENT_CH)));
                if (generated_rms > 1e-6f && reference_rms > 1e-6f) {
                    float scale = reference_rms / generated_rms;
                    if (isfinite(scale))
                        for (int64_t i = 0; i < use * MUSIC_LATENT_CH; i++) latents[i] *= scale;
                }
                float w = req->audio_weight;
                float generated_weight = (1.0f - w) * (1.0f - w);
                float reference_weight = 1.0f - generated_weight;
                for (int64_t i = 0; i < use * MUSIC_LATENT_CH; i++)
                    latents[i] = generated_weight * latents[i] +
                                 reference_weight * reference[ref_start * MUSIC_LATENT_CH + i];
            }
        }
        /* A window hop of 100 frames is 344 latent frames, but only the leading
         * 172 latents of the overlap are re-synthesised: the next window's
         * latents [0,172) cover the same audio as this window's
         * [T-344, T-172). Carrying the trailing 344 latents instead (and
         * blending 344 latents) both reached into the region the next window
         * re-generates from scratch and smeared the boundary of every chunk. */
        int64_t ostart = T > MUSIC_CARRY_LATENT ? T - MUSIC_CARRY_LATENT : 0;
        int64_t oend = T > MUSIC_OVERLAP_LATENT ? T - MUSIC_OVERLAP_LATENT : 0;
        if (oend < ostart) oend = ostart;
        previous_n = oend - ostart;
        if (previous_n > 0) {
            memcpy(previous, latents + ostart * MUSIC_LATENT_CH,
                   (size_t)previous_n * MUSIC_LATENT_CH * sizeof(float));
            memcpy(previous_cond, condition + ostart * MUSIC_COND_DIM,
                   (size_t)previous_n * MUSIC_COND_DIM * sizeof(float));
        }
        chunks[c] = music_f32(T * MUSIC_LATENT_CH);
        if (!chunks[c]) { free(condition); cwick_error_set(err, CWICK_ERR_MEMORY, "latent chunk allocation failed"); goto fail; }
        memcpy(chunks[c], latents, (size_t)T * MUSIC_LATENT_CH * sizeof(float));
        if (!music_audio_valid(chunks[c], T * MUSIC_LATENT_CH, "flow output latent", err)) {
            free(condition); goto fail;
        }
        lens[c] = T;
        free(condition);
        CWICK_INFO("music", "phase flow: chunk=%d/%d frames=%lld latent=%lld steps=%d/%lld elapsed=%.0f ms",
                   c + 1, n_chunks, (long long)nf, (long long)T, req->steps,
                   (long long)step_done, cond_acc + flow_acc);
    }
    CWICK_INFO("music", "phase condition done: chunks=%d elapsed=%.0f ms", n_chunks, cond_acc);
    CWICK_INFO("music", "phase flow done: chunks=%d steps=%lld elapsed=%.0f ms", n_chunks,
               (long long)step_done, flow_acc);
    *chunks_out = chunks; *chunk_lens_out = lens; *n_chunks_out = n_chunks;
    *steps_out = total_steps; *condition_ms = cond_acc; *flow_ms = flow_acc;
    free(previous); free(previous_cond); free(noise_prompt); free(latents);
    music_dit_work_free(&work);
    return true;
fail:
    for (int i = 0; i < n_chunks; i++) free(chunks[i]);
    free(chunks); free(lens); free(previous); free(previous_cond);
    free(noise_prompt); free(latents); music_dit_work_free(&work);
    return false;
}

/* ------------------------------------------------------------------ */
/* Flow-VAE encoder and vocoder                                         */
/* ------------------------------------------------------------------ */

static float *music_conv_regular(const music_conv *c, const float *x,
                                 int B, int64_t T) {
    int64_t L = (T + 2 * c->pad - c->dil * (c->K - 1) - 1) / c->stride + 1;
    if (L <= 0) return NULL;
    int64_t rows = B * L, K = (int64_t)c->I * c->K;
    float *y = music_f32(B * L * c->O);
    float *patch = music_f32(128 * K);
    float *gemm_out = music_f32(128 * c->O);
    if (!y || !patch || !gemm_out) { free(y); free(patch); free(gemm_out); return NULL; }
    for (int64_t r0 = 0; r0 < rows; r0 += 128) {
        int64_t rn = rows - r0 < 128 ? rows - r0 : 128;
        for (int64_t r = 0; r < rn; r++) {
            int64_t flat = r0 + r, b = flat / L, ot = flat % L;
            for (int i = 0; i < c->I; i++)
                for (int k = 0; k < c->K; k++) {
                    int64_t src_t = ot * c->stride - c->pad + (int64_t)k * c->dil;
                    patch[r * K + i * c->K + k] =
                        src_t >= 0 && src_t < T ? x[(b * T + src_t) * c->I + i] : 0.0f;
                }
        }
        music_linear_f32(gemm_out, patch, rn, K, c->O, c->v, NULL);
        for (int64_t r = 0; r < rn; r++) {
            float *row = y + (r0 + r) * c->O;
            for (int o = 0; o < c->O; o++) {
                float v = gemm_out[r * c->O + o];
                if (c->scale) v *= c->scale[o];
                row[o] = v + c->b[o];
            }
        }
    }
    free(patch); free(gemm_out);
    return y;
}

bool music_conv_selftest(void) {
    const float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float weight[2] = {1.0f, 2.0f};
    const float bias[1] = {0.5f};
    music_conv conv;
    memset(&conv, 0, sizeof(conv));
    conv.v = weight; conv.b = bias;
    conv.I = 1; conv.O = 1; conv.K = 2; conv.stride = 2; conv.dil = 1;
    float *out = music_conv_regular(&conv, input, 1, 4);
    if (!out) return false;
    bool ok = fabsf(out[0] - 5.5f) < 1e-6f && fabsf(out[1] - 11.5f) < 1e-6f;
    free(out);
    return ok;
}

static float *music_conv_transposed(const music_conv *c, const float *x,
                                    int B, int64_t T) {
    int64_t L = (T - 1) * c->stride + c->K - 2 * c->pad;
    if (L <= 0) return NULL;
    int64_t rows = B * T;
    float *y = music_calloc_f32(B * L * c->O);
    float *sx = music_f32(rows * c->I);
    float *tmp = music_f32(rows * c->O);
    if (!y || !sx || !tmp) { free(y); free(sx); free(tmp); return NULL; }
    for (int64_t r = 0; r < rows; r++)
        for (int i = 0; i < c->I; i++)
            sx[r * c->I + i] = x[r * c->I + i] * (c->scale ? c->scale[i] : 1.0f);
    for (int k = 0; k < c->K; k++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)rows, c->O, c->I, 1.0f, sx, c->I,
                    c->kernel + (int64_t)k * c->O * c->I, c->I,
                    0.0f, tmp, c->O);
        for (int64_t r = 0; r < rows; r++) {
            int64_t b = r / T, t = r % T;
            int64_t pos = t * c->stride + k - c->pad;
            if (pos < 0 || pos >= L) continue;
            float *dst = y + (b * L + pos) * c->O;
            const float *src = tmp + r * c->O;
            for (int o = 0; o < c->O; o++) dst[o] += src[o];
        }
    }
    for (int64_t r = 0; r < B * L; r++)
        for (int o = 0; o < c->O; o++) y[r * c->O + o] += c->b[o];
    free(sx); free(tmp);
    return y;
}

static void music_snake(float *x, int64_t rows, int C, const float *alpha) {
    for (int64_t r = 0; r < rows; r++)
        for (int c = 0; c < C; c++) {
            float a = alpha[c] + 1e-9f;
            float s = sinf(alpha[c] * x[r * C + c]);
            x[r * C + c] += s * s / a;
        }
}

static float *music_resunit_run(const music_resunit *u, const float *x,
                                int B, int64_t T, int C) {
    float *pre = music_f32(B * T * C);
    if (!pre) return NULL;
    memcpy(pre, x, (size_t)B * T * C * sizeof(float));
    music_snake(pre, B * T, C, u->alpha1);
    float *y = music_conv_regular(&u->c1, pre, B, T);
    free(pre);
    if (!y) return NULL;
    music_snake(y, B * T, C, u->alpha2);
    float *z = music_conv_regular(&u->c2, y, B, T);
    free(y);
    if (!z) return NULL;
    for (int64_t i = 0; i < B * T * C; i++) z[i] += x[i];
    return z;
}

static float *music_vocoder_decode(const music_model *m, const float *latent,
                                   int64_t T, int64_t *samples_out) {
    float *x = music_f32(2LL * T * 64);
    if (!x) return NULL;
    for (int c = 0; c < 2; c++)
        for (int64_t t = 0; t < T; t++)
            memcpy(x + (c * T + t) * 64, latent + t * MUSIC_LATENT_CH + c * 64,
                   64 * sizeof(float));
    float *y = music_conv_regular(&m->vocoder.in_proj, x, 2, T);
    free(x); x = y;
    if (!x) return NULL;
    y = music_conv_regular(&m->vocoder.conv_in, x, 2, T);
    free(x); x = y;
    if (!x) return NULL;
    int64_t len = T;
    for (int i = 0; i < 4; i++) {
        const music_vocoder_block *b = &m->vocoder.block[i];
        int C = 1536 >> i;
        music_snake(x, 2 * len, C, b->alpha);
        y = music_conv_transposed(&b->up, x, 2, len);
        free(x); x = y;
        if (!x) return NULL;
        len *= b->up.stride;
        C >>= 1;
        for (int j = 0; j < 3; j++) {
            y = music_resunit_run(&b->unit[j], x, 2, len, C);
            free(x); x = y;
            if (!x) return NULL;
        }
    }
    music_snake(x, 2 * len, 96, m->vocoder.alpha_out);
    y = music_conv_regular(&m->vocoder.conv_out, x, 2, len);
    free(x);
    if (!y) return NULL;
    float *wave = music_f32(len * 2);
    if (!wave) { free(y); return NULL; }
    for (int64_t t = 0; t < len; t++) {
        wave[t * 2] = tanhf(y[t]);
        wave[t * 2 + 1] = tanhf(y[len + t]);
    }
    free(y);
    *samples_out = len;
    return wave;
}

static float *music_encoder_segment(const music_model *m, const float *audio,
                                    int64_t T) {
    float *left = music_f32(T), *right = music_f32(T);
    if (!left || !right) { free(left); free(right); return NULL; }
    for (int64_t t = 0; t < T; t++) { left[t] = audio[t * 2]; right[t] = audio[t * 2 + 1]; }
    float *a = music_conv_regular(&m->encoder.input, left, 1, T);
    float *b = music_conv_regular(&m->encoder.input, right, 1, T);
    free(left); free(right);
    if (!a || !b) { free(a); free(b); return NULL; }
    float *x = music_f32(2LL * T * 64);
    if (!x) { free(a); free(b); return NULL; }
    memcpy(x, a, (size_t)T * 64 * sizeof(float));
    memcpy(x + T * 64, b, (size_t)T * 64 * sizeof(float));
    free(a); free(b);
    int64_t len = T;
    for (int i = 0; i < 4; i++) {
        int C = 64 << i;
        for (int j = 0; j < 3; j++) {
            music_resunit unit = {m->encoder.unit[i][2 * j],
                                  m->encoder.unit[i][2 * j + 1],
                                  m->encoder.unit_alpha[i][2 * j],
                                  m->encoder.unit_alpha[i][2 * j + 1]};
            a = music_resunit_run(&unit, x, 2, len, C);
            free(x); x = a;
            if (!x) return NULL;
        }
        music_snake(x, 2 * len, C, m->encoder.down_alpha[i]);
        a = music_conv_regular(&m->encoder.down[i], x, 2, len);
        free(x); x = a;
        if (!x) return NULL;
        len /= m->encoder.down[i].stride;
    }
    music_snake(x, 2 * len, 1024, m->encoder.final_alpha);
    a = music_conv_regular(&m->encoder.final, x, 2, len);
    free(x); x = a;
    if (!x) return NULL;
    a = music_conv_regular(&m->encoder.mean, x, 2, len);
    free(x);
    return a;
}

/* ------------------------------------------------------------------ */
/* WAV input, sample-rate conversion, WAV output                       */
/* ------------------------------------------------------------------ */

static uint16_t music_rd16(const unsigned char *p) {
    return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
}
static uint32_t music_rd32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}
static void music_wr16(unsigned char *p, uint16_t v) {
    p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8);
}
static void music_wr32(unsigned char *p, uint32_t v) {
    p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8);
    p[2] = (unsigned char)(v >> 16); p[3] = (unsigned char)(v >> 24);
}

static bool music_wav_decode(const unsigned char *data, size_t len, float **out,
                             int64_t *samples_out, int *rate_out,
                             cwick_error *err) {
    if (len < 12 || memcmp(data, "RIFF", 4) || memcmp(data + 8, "WAVE", 4)) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "reference is not a RIFF/WAVE file"); return false;
    }
    const unsigned char *fmt = NULL, *pcm = NULL;
    size_t fmt_len = 0, pcm_len = 0;
    size_t p = 12;
    while (p + 8 <= len) {
        uint32_t n = music_rd32(data + p + 4);
        size_t body = p + 8;
        if ((size_t)n > len - body) {
            cwick_error_set(err, CWICK_ERR_ARGUMENT, "reference WAV chunk is truncated"); return false;
        }
        if (!memcmp(data + p, "fmt ", 4)) { fmt = data + body; fmt_len = n; }
        else if (!memcmp(data + p, "data", 4)) { pcm = data + body; pcm_len = n; }
        p = body + n + (n & 1u);
    }
    if (!fmt || fmt_len < 16 || !pcm || pcm_len == 0) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "reference WAV lacks fmt or data"); return false;
    }
    uint16_t format = music_rd16(fmt), channels = music_rd16(fmt + 2);
    uint32_t rate = music_rd32(fmt + 4), align = music_rd16(fmt + 12);
    uint16_t bits = music_rd16(fmt + 14);
    if (format == 0xfffe && fmt_len >= 40) format = music_rd16(fmt + 24);
    if ((format != 1 && format != 3) || (channels != 1 && channels != 2) ||
        rate == 0 || rate > 384000 || align < channels * (bits / 8) ||
        (bits != 8 && bits != 16 && bits != 24 && bits != 32) ||
        (format == 3 && bits != 32 && bits != 64)) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "unsupported reference WAV format"); return false;
    }
    int bytes = bits / 8;
    int64_t frames = (int64_t)(pcm_len / align);
    if (frames <= 0 || (long double)frames / rate > MUSIC_MAX_SECONDS) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "reference WAV length is outside the supported range"); return false;
    }
    float *stereo = music_f32(frames * 2);
    if (!stereo) { cwick_error_set(err, CWICK_ERR_MEMORY, "reference WAV allocation failed"); return false; }
    for (int64_t t = 0; t < frames; t++) {
        for (int c = 0; c < channels; c++) {
            const unsigned char *q = pcm + t * align + c * bytes;
            float v = 0.0f;
            if (format == 3) {
                if (bits == 32) { uint32_t u = music_rd32(q); memcpy(&v, &u, 4); }
                else { uint64_t u = (uint64_t)music_rd32(q) | ((uint64_t)music_rd32(q + 4) << 32); double d; memcpy(&d, &u, 8); v = (float)d; }
            } else if (bits == 8) v = ((int)q[0] - 128) / 128.0f;
            else if (bits == 16) v = (int16_t)music_rd16(q) / 32768.0f;
            else if (bits == 24) {
                int32_t s = (int32_t)(q[0] | (q[1] << 8) | (q[2] << 16));
                if (s & 0x800000) s |= ~0xffffff;
                v = s / 8388608.0f;
            } else v = (int32_t)music_rd32(q) / 2147483648.0f;
            if (!isfinite(v)) {
                free(stereo); cwick_error_set(err, CWICK_ERR_ARGUMENT, "reference WAV contains non-finite samples"); return false;
            }
            stereo[t * 2 + c] = v;
            if (channels == 1) stereo[t * 2 + 1] = v;
        }
    }
    *out = stereo; *samples_out = frames; *rate_out = (int)rate;
    return true;
}

static float *music_resample_channel(const float *in, int64_t n, int stride,
                                     int from, int to) {
    if (from == to) {
        float *y = music_f32(n);
        if (y) for (int64_t i = 0; i < n; i++) y[i] = in[i * stride];
        return y;
    }
    float h[MUSIC_RS_TAPS];
    double fc = 0.5 * (to < from ? (double)to / from : 1.0);
    double mid = (MUSIC_RS_TAPS - 1) / 2.0, sum = 0.0;
    for (int i = 0; i < MUSIC_RS_TAPS; i++) {
        double x = i - mid;
        double s = x == 0.0 ? 2.0 * fc : sin(2.0 * M_PI * fc * x) / (M_PI * x);
        double w = 0.54 - 0.46 * cos(2.0 * M_PI * i / (MUSIC_RS_TAPS - 1));
        h[i] = (float)(s * w); sum += h[i];
    }
    for (int i = 0; i < MUSIC_RS_TAPS; i++) h[i] /= (float)sum;
    int64_t m = (int64_t)((long double)n * to / from);
    float *y = music_f32(m > 0 ? m : 1);
    if (!y) return NULL;
    double ratio = (double)from / to;
    int half = MUSIC_RS_TAPS / 2;
    for (int64_t j = 0; j < m; j++) {
        double src = (double)j * ratio;
        int64_t base = (int64_t)floor(src) - half + 1;
        double acc = 0.0;
        for (int k = 0; k < MUSIC_RS_TAPS; k++) {
            int64_t q = base + k;
            if (q >= 0 && q < n) acc += (double)h[k] * in[q * stride];
        }
        y[j] = (float)acc;
    }
    return y;
}

static float *music_resample_stereo(const float *in, int64_t n, int from, int to,
                                    int64_t *n_out) {
    float *a = music_resample_channel(in, n, 2, from, to);
    float *b = music_resample_channel(in + 1, n, 2, from, to);
    if (!a || !b) { free(a); free(b); return NULL; }
    int64_t m = (int64_t)((long double)n * to / from);
    float *out = music_f32(m * 2);
    if (!out) { free(a); free(b); return NULL; }
    for (int64_t i = 0; i < m; i++) { out[i * 2] = a[i]; out[i * 2 + 1] = b[i]; }
    free(a); free(b); *n_out = m;
    return out;
}

static int64_t music_select_short_window(float *wave, int64_t samples,
                                         int requested_seconds,
                                         int64_t *start_out) {
    const int rate = MUSIC_VAE_RATE;
    const int64_t want = (int64_t)requested_seconds * rate;
    if (samples <= want) return samples;
    const int64_t hop = rate / 10;
    const int64_t block = rate / 10;
    double best_score = INFINITY;
    int64_t best = 0;
    for (int64_t start = 0; start + want <= samples; start += hop) {
        double sum = 0.0, sum2 = 0.0;
        int64_t blocks = 0, silent = 0;
        for (int64_t at = start; at < start + want; at += block) {
            double ss = 0.0;
            int64_t end = at + block;
            if (end > start + want) end = start + want;
            for (int64_t i = at; i < end; i++) {
                double l = wave[i * 2], r = wave[i * 2 + 1];
                ss += l * l + r * r;
            }
            double rms = sqrt(ss / (double)((end - at) * 2));
            sum += rms; sum2 += rms * rms; blocks++;
        }
        double mean = sum / (double)blocks;
        if (mean > 1e-12) {
            for (int64_t at = start; at < start + want; at += block) {
                double ss = 0.0;
                int64_t end = at + block;
                if (end > start + want) end = start + want;
                for (int64_t i = at; i < end; i++) {
                    double l = wave[i * 2], r = wave[i * 2 + 1];
                    ss += l * l + r * r;
                }
                double rms = sqrt(ss / (double)((end - at) * 2));
                if (rms < fmax(0.001, mean * 0.06)) silent++;
            }
        }
        double var = sum2 / (double)blocks - mean * mean;
        if (var < 0.0) var = 0.0;
        double cv = mean > 1e-12 ? sqrt(var) / mean : 1000.0;
        double score = cv + 20.0 * (double)silent / (double)blocks;
        if (score < best_score) { best_score = score; best = start; }
    }
    memmove(wave, wave + best * 2, (size_t)want * 2 * sizeof(float));
    double ss = 0.0, peak = 0.0;
    for (int64_t i = 0; i < want * 2; i++) {
        double v = wave[i];
        ss += v * v;
        if (fabs(v) > peak) peak = fabs(v);
    }
    double rms = sqrt(ss / (double)(want * 2));
    if (rms > 1e-8 && peak > 1e-8) {
        double gain = fmin(0.90 / peak, 0.11 / rms);
        for (int64_t i = 0; i < want * 2; i++) wave[i] *= (float)gain;
    }
    const int64_t fade = rate / 200;
    for (int64_t i = 0; i < fade && i < want; i++) {
        float g = (float)i / (float)fade;
        wave[i * 2] *= g; wave[i * 2 + 1] *= g;
        wave[(want - 1 - i) * 2] *= g;
        wave[(want - 1 - i) * 2 + 1] *= g;
    }
    if (start_out) *start_out = best;
    return want;
}

bool music_short_window_selftest(void) {
    const int rate = MUSIC_VAE_RATE;
    const int64_t n = (int64_t)20 * rate;
    float *x = music_f32(n * 2);
    if (!x) return false;
    for (int64_t i = 0; i < n; i++) {
        float v = i < 10 * rate
            ? ((i / (rate / 2)) % 2 ? 0.2f : 0.001f)
            : 0.05f;
        x[i * 2] = x[i * 2 + 1] = v;
    }
    int64_t start = -1;
    int64_t got = music_select_short_window(x, n, 10, &start);
    bool ok = got == 10 * rate && start >= 10 * rate;
    free(x);
    return ok;
}

static float *music_reference_encode(const music_model *m, const music_request *req,
                                     int64_t *frames_out, cwick_error *err) {
    double started = music_ms();
    float *audio = NULL;
    int64_t samples = 0;
    int rate = 0;
    if (!req->reference_audio || req->reference_len == 0) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "cover requires reference_audio"); return NULL;
    }
    if (!music_wav_decode(req->reference_audio, req->reference_len, &audio, &samples, &rate, err)) return NULL;
    int64_t at_n = 0;
    float *at = music_resample_stereo(audio, samples, rate, MUSIC_VAE_RATE, &at_n);
    free(audio);
    if (!at || at_n <= 0) { free(at); cwick_error_set(err, CWICK_ERR_MEMORY, "reference resampling failed"); return NULL; }
    int64_t padded = (at_n + MUSIC_LATENT_HOP - 1) / MUSIC_LATENT_HOP * MUSIC_LATENT_HOP;
    float *padded_audio = music_calloc_f32(padded * 2);
    if (!padded_audio) { free(at); cwick_error_set(err, CWICK_ERR_MEMORY, "reference padding failed"); return NULL; }
    memcpy(padded_audio, at, (size_t)at_n * 2 * sizeof(float));
    free(at);
    int64_t total = padded / MUSIC_LATENT_HOP;
    float *latent = music_f32(total * MUSIC_LATENT_CH);
    if (!latent) { free(padded_audio); cwick_error_set(err, CWICK_ERR_MEMORY, "reference latent allocation failed"); return NULL; }
    const int64_t main_samples = 860LL * MUSIC_LATENT_HOP;
    const int64_t context = 8192;
    int64_t chunks = (padded + main_samples - 1) / main_samples;
    music_progress(req, MUSIC_PHASE_REFERENCE, 0, chunks);
    CWICK_INFO("music", "phase reference_encode start: samples=%lld rate=44100 latent=%lld chunks=%lld",
               (long long)at_n, (long long)total, (long long)chunks);
    for (int64_t c = 0; c < chunks; c++) {
        if (!music_check_cancel(req, err)) { free(latent); free(padded_audio); return NULL; }
        int64_t start = c * main_samples, end = start + main_samples;
        if (end > padded) end = padded;
        int64_t seg_start = start > context ? start - context : 0;
        int64_t seg_end = end + context < padded ? end + context : padded;
        float *mean = music_encoder_segment(m, padded_audio + seg_start * 2, seg_end - seg_start);
        if (!mean) {
            free(latent); free(padded_audio);
            cwick_error_set(err, CWICK_ERR_MEMORY, "Flow-VAE encoder failed"); return NULL;
        }
        int64_t crop_l = (start - seg_start) / MUSIC_LATENT_HOP;
        int64_t crop_r = (seg_end - end) / MUSIC_LATENT_HOP;
        int64_t seg_frames = (seg_end - seg_start) / MUSIC_LATENT_HOP;
        int64_t keep = (end - start) / MUSIC_LATENT_HOP;
        for (int64_t t = 0; t < keep; t++)
            for (int ch = 0; ch < 2; ch++)
                memcpy(latent + ((start / MUSIC_LATENT_HOP + t) * MUSIC_LATENT_CH + ch * 64),
                       mean + ((ch * seg_frames + crop_l + t) * 64),
                       64 * sizeof(float));
        (void)crop_r;
        free(mean);
        music_progress(req, MUSIC_PHASE_REFERENCE, c + 1, chunks);
    }
    free(padded_audio);
    for (int64_t i = 0; i < total * MUSIC_LATENT_CH; i++)
        if (!isfinite(latent[i])) {
            free(latent); cwick_error_set(err, CWICK_ERR_STATE, "Flow-VAE encoder produced non-finite latents"); return NULL;
        }
    *frames_out = total;
    CWICK_INFO("music", "phase reference_encode done: latent=%lld elapsed=%.0f ms",
               (long long)total, music_ms() - started);
    return latent;
}

static unsigned char *music_wav_write(const float *audio, int64_t samples,
                                      size_t *len_out, cwick_error *err) {
    int64_t at_n = 0;
    float *at = music_resample_stereo(audio, samples, MUSIC_VAE_RATE,
                                      MUSIC_SAMPLE_RATE, &at_n);
    if (!at || at_n <= 0) { free(at); cwick_error_set(err, CWICK_ERR_MEMORY, "output resampling failed"); return NULL; }
    if (!music_audio_valid(at, at_n * 2, "WAV output", err)) { free(at); return NULL; }
    float peak = 0.0f;
    for (int64_t i = 0; i < at_n * 2; i++) {
        float a = fabsf(at[i]);
        if (a > peak) peak = a;
    }
    if (peak > 0.98f) {
        float gain = 0.98f / peak;
        for (int64_t i = 0; i < at_n * 2; i++) at[i] *= gain;
    }
    uint32_t data_bytes = (uint32_t)(at_n * 4);
    size_t len = 44 + data_bytes;
    unsigned char *wav = (unsigned char *)cwick_xmalloc(len);
    memcpy(wav, "RIFF", 4); music_wr32(wav + 4, (uint32_t)(len - 8));
    memcpy(wav + 8, "WAVEfmt ", 8); music_wr32(wav + 16, 16);
    music_wr16(wav + 20, 1); music_wr16(wav + 22, 2);
    music_wr32(wav + 24, MUSIC_SAMPLE_RATE); music_wr32(wav + 28, MUSIC_SAMPLE_RATE * 4);
    music_wr16(wav + 32, 4); music_wr16(wav + 34, 16);
    memcpy(wav + 36, "data", 4); music_wr32(wav + 40, data_bytes);
    for (int64_t i = 0; i < at_n * 2; i++) {
        float v = at[i];
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        int32_t s = (int32_t)lrintf(v * 32767.0f);
        music_wr16(wav + 44 + i * 2, (uint16_t)(int16_t)s);
    }
    free(at); *len_out = len;
    return wav;
}

void music_wav_free(unsigned char *p) { free(p); }

/* ------------------------------------------------------------------ */
/* Public full-chain entry point                                       */
/* ------------------------------------------------------------------ */

bool music_condition_probe(music_model *m, const float *frames, int64_t n_frames,
                           float **out, int64_t *out_frames) {
    if (!m || !frames || !out || !out_frames || n_frames <= 0) return false;
    pthread_mutex_lock(&m->lock);
    int64_t n = 0;
    float *v = music_condition_encode(m, frames, n_frames, &n);
    pthread_mutex_unlock(&m->lock);
    if (!v) return false;
    *out = v;
    *out_frames = n;
    return true;
}

bool music_reference_probe(music_model *m, const unsigned char *wav, size_t len,
                           float **latent, int64_t *frames) {
    if (!m || !wav || !latent || !frames) return false;
    music_request req;
    memset(&req, 0, sizeof(req));
    req.task = MUSIC_TASK_COVER;
    req.reference_audio = wav;
    req.reference_len = len;
    cwick_error err;
    cwick_error_clear(&err);
    pthread_mutex_lock(&m->lock);
    float *out = music_reference_encode(m, &req, frames, &err);
    pthread_mutex_unlock(&m->lock);
    if (!out) return false;
    *latent = out;
    return true;
}

bool music_dit_probe(music_model *m, const float *latents, const float *condition,
                     int64_t frames, double time, float **out) {
    if (!m || !latents || !condition || !out || frames <= 0 ||
        frames > MUSIC_CHUNK_FRAMES) return false;
    music_dit_work work;
    if (!music_dit_work_init(&work, frames)) return false;
    pthread_mutex_lock(&m->lock);
    bool ok = music_dit_forward(m, latents, condition, frames, time, &work);
    pthread_mutex_unlock(&m->lock);
    if (!ok) { music_dit_work_free(&work); return false; }
    *out = music_f32(2LL * frames * MUSIC_LATENT_CH);
    if (!*out) { music_dit_work_free(&work); return false; }
    memcpy(*out, work.out, (size_t)2LL * frames * MUSIC_LATENT_CH * sizeof(float));
    music_dit_work_free(&work);
    return true;
}

bool music_vocoder_probe(music_model *m, const float *latent, int64_t frames,
                         float **out, int64_t *out_frames) {
    if (!m || !latent || !out || !out_frames || frames <= 0) return false;
    pthread_mutex_lock(&m->lock);
    float *pcm = music_vocoder_decode(m, latent, frames, out_frames);
    pthread_mutex_unlock(&m->lock);
    if (!pcm) return false;
    *out = pcm;
    return true;
}

bool music_generate(music_model *m, const music_request *req,
                    unsigned char **out_wav, size_t *out_len,
                    music_timing *tim, cwick_error *err) {
    cwick_error_clear(err);
    if (out_wav) *out_wav = NULL;
    if (out_len) *out_len = 0;
    if (!m || !m->ready || !req || !out_wav || !out_len) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "invalid music generation arguments");
        return false;
    }
    if (req->task != MUSIC_TASK_GENERATE && req->task != MUSIC_TASK_COVER) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "unknown music task");
        return false;
    }
    if (req->seconds < MUSIC_MIN_SECONDS || req->seconds > MUSIC_MAX_SECONDS ||
        req->steps < 1 || req->steps > 100 || !isfinite(req->cfg_scale) ||
        req->cfg_scale <= 0.0f || !isfinite(req->temperature) || req->temperature < 0.0f ||
        req->top_k < 0 || req->top_k > MUSIC_LM_VOCAB) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "invalid music duration, steps, CFG, or sampler");
        return false;
    }
    if (req->task == MUSIC_TASK_COVER && (!req->reference_audio || req->reference_len == 0)) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "cover requires reference_audio");
        return false;
    }
    if (req->task == MUSIC_TASK_COVER &&
        (!isfinite(req->audio_weight) || req->audio_weight < 0.0f || req->audio_weight > 1.0f)) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "cover audio_weight must be between 0 and 1");
        return false;
    }
    pthread_mutex_lock(&m->lock);
    if (!m->ready) {
        pthread_mutex_unlock(&m->lock);
        cwick_error_set(err, CWICK_ERR_STATE, "music model is not ready");
        return false;
    }
    music_timing t;
    memset(&t, 0, sizeof(t));
    t.load_ms = m->load_ms;
    t.cover = req->task == MUSIC_TASK_COVER;
    double started = music_ms();
    uint32_t *prompt_ids = NULL;
    int64_t prompt_n = 0, n_frames = 0, reference_n = 0;
    float *prompt_frames = NULL, *reference = NULL;
    float **chunks = NULL;
    int64_t *chunk_lens = NULL;
    int64_t n_chunks = 0, flow_steps = 0;
    float *wave = NULL;
    int64_t wave_n = 0;
    unsigned char *wav = NULL;
    size_t wav_len = 0;
    const int output_seconds = req->seconds;
    int generate_seconds = req->seconds;
    const char *extend = getenv("CWICK_MUSIC_SHORT_EXTEND");
    if (!t.cover && output_seconds <= 15 &&
        !(extend && *extend && strcmp(extend, "0") == 0))
        generate_seconds = 20;
    music_request generate_req = *req;
    generate_req.seconds = generate_seconds;
    int max_frames = generate_seconds * 25;
    if (max_frames > MUSIC_MAX_FRAMES) max_frames = MUSIC_MAX_FRAMES;
    t.requested_frames = output_seconds * 25;

    music_progress(req, MUSIC_PHASE_TOKENIZE, 0, 1);
    double a = music_ms();
    if (!music_tokenize(m, req, &prompt_ids, &prompt_n, err)) goto done;
    t.tokenize_ms = music_ms() - a;
    t.prompt_tokens = prompt_n;
    music_progress(req, MUSIC_PHASE_TOKENIZE, 1, 1);
    CWICK_INFO("music", "phase tokenize done: tokens=%lld elapsed=%.0f ms",
               (long long)prompt_n, t.tokenize_ms);

    if (t.cover) {
        a = music_ms();
        reference = music_reference_encode(m, req, &reference_n, err);
        if (!reference) goto done;
        t.reference_encode_ms = music_ms() - a;
        music_progress(req, MUSIC_PHASE_REFERENCE, 1, 1);
    }
    music_request semantic_req = generate_req;
    t.ar_attempts = 1;
    double attempt_global = 0.0, attempt_depth = 0.0;
    if (!music_semantic_generate(m, &semantic_req, prompt_ids, prompt_n, max_frames,
                                 &prompt_frames, &n_frames, &attempt_global,
                                 &attempt_depth, err)) goto done;
    t.global_lm_ms += attempt_global;
    t.depth_ms += attempt_depth;
    t.frames = n_frames;
    t.ended_early = n_frames < max_frames;
    if (n_frames < music_ar_min_frames(max_frames)) {
        cwick_error_set(err, CWICK_ERR_STATE,
                        "music model ended after %lld audio frames; minimum is %d",
                        (long long)n_frames, music_ar_min_frames(max_frames));
        goto done;
    }
    music_progress(req, MUSIC_PHASE_GLOBAL, n_frames, n_frames);

    a = music_ms();
    if (!music_flow_generate(m, &generate_req, prompt_frames, n_frames, reference, reference_n,
                             &chunks, &chunk_lens, &n_chunks, &flow_steps,
                             &t.condition_ms, &t.flow_ms, err)) goto done;
    (void)a;
    t.flow_steps = flow_steps;
    music_progress(req, MUSIC_PHASE_CONDITION, n_chunks, n_chunks);
    music_progress(req, MUSIC_PHASE_FLOW, flow_steps, flow_steps);

    int64_t total_wave = 0;
    for (int64_t c = 0; c < n_chunks; c++) {
        int64_t n = chunk_lens[c] * MUSIC_LATENT_HOP;
        if (c > 0) n -= MUSIC_CROP_LEFT * MUSIC_LATENT_HOP;
        if (c + 1 < n_chunks) n -= MUSIC_CROP_RIGHT * MUSIC_LATENT_HOP;
        if (n <= 0) {
            cwick_error_set(err, CWICK_ERR_CONTRACT, "invalid vocoder chunk crop length");
            goto done;
        }
        total_wave += n;
    }
    wave = music_f32(total_wave * 2);
    if (!wave) { cwick_error_set(err, CWICK_ERR_MEMORY, "cannot allocate stitched waveform"); goto done; }
    music_progress(req, MUSIC_PHASE_VOCODER, 0, n_chunks);
    a = music_ms();
    CWICK_INFO("music", "phase vocoder start: chunks=%lld samples=%lld", (long long)n_chunks,
               (long long)total_wave);
    int64_t wave_at = 0;
    for (int64_t c = 0; c < n_chunks; c++) {
        if (!music_check_cancel(req, err)) goto done;
        int64_t decoded_n = 0;
        float *part = music_vocoder_decode(m, chunks[c], chunk_lens[c], &decoded_n);
        if (!part || decoded_n != chunk_lens[c] * MUSIC_LATENT_HOP) {
            free(part);
            cwick_error_set(err, CWICK_ERR_CONTRACT, "vocoder returned an invalid sample count");
            goto done;
        }
        if (!music_audio_valid(part, decoded_n * 2, "vocoder output", err)) {
            free(part); goto done;
        }
        double chunk_ss = 0.0;
        for (int64_t i = 0; i < decoded_n * 2; i++) chunk_ss += (double)part[i] * part[i];
        CWICK_INFO("music", "vocoder chunk=%lld/%lld decoded=%lld rms=%.5f",
                   (long long)(c + 1), (long long)n_chunks, (long long)decoded_n,
                   sqrt(chunk_ss / (double)(decoded_n * 2)));
        int64_t left = c > 0 ? MUSIC_CROP_LEFT * MUSIC_LATENT_HOP : 0;
        int64_t right = c + 1 < n_chunks ? MUSIC_CROP_RIGHT * MUSIC_LATENT_HOP : 0;
        int64_t keep = decoded_n - left - right;
        memcpy(wave + wave_at * 2, part + left * 2, (size_t)keep * 2 * sizeof(float));
        free(part);
        wave_at += keep;
        music_progress(req, MUSIC_PHASE_VOCODER, c + 1, n_chunks);
    }
    if (wave_at != total_wave) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "vocoder stitch length mismatch");
        goto done;
    }
    wave_n = total_wave;
    if (generate_seconds > output_seconds) {
        int64_t selected = 0;
        wave_n = music_select_short_window(wave, wave_n, output_seconds, &selected);
        t.frames = output_seconds * 25;
        t.ended_early = 0;
        CWICK_INFO("music", "short-track window: generated=%ds selected=%.2fs..%.2fs output=%ds",
                   generate_seconds, (double)selected / MUSIC_VAE_RATE,
                   (double)(selected + wave_n) / MUSIC_VAE_RATE, output_seconds);
    }
    t.vocoder_ms = music_ms() - a;
    CWICK_INFO("music", "phase vocoder done: chunks=%lld samples=%lld elapsed=%.0f ms",
               (long long)n_chunks, (long long)wave_n, t.vocoder_ms);

    music_progress(req, MUSIC_PHASE_WAV, 0, 1);
    a = music_ms();
    wav = music_wav_write(wave, wave_n, &wav_len, err);
    if (!wav) goto done;
    t.wav_ms = music_ms() - a;
    t.audio_samples = (int64_t)((long double)wave_n * MUSIC_SAMPLE_RATE / MUSIC_VAE_RATE);
    t.output_bytes = (int64_t)wav_len;
    music_progress(req, MUSIC_PHASE_WAV, 1, 1);
    CWICK_INFO("music", "phase wav done: samples=%lld bytes=%lld elapsed=%.0f ms",
               (long long)t.audio_samples, (long long)t.output_bytes, t.wav_ms);
    *out_wav = wav;
    *out_len = wav_len;
    wav = NULL;
    t.total_ms = music_ms() - started + t.load_ms;
    if (tim) *tim = t;
    CWICK_INFO("music", "music done: task=%s frames=%lld steps=%lld audio=%lld total=%.0f ms "
                         "(load=%.0f tokenize=%.0f reference=%.0f global=%.0f depth=%.0f "
                         "condition=%.0f flow=%.0f vocoder=%.0f wav=%.0f)",
               t.cover ? "cover" : "generate", (long long)t.frames, (long long)t.flow_steps,
               (long long)t.audio_samples, t.total_ms, t.load_ms, t.tokenize_ms,
               t.reference_encode_ms, t.global_lm_ms, t.depth_ms, t.condition_ms,
               t.flow_ms, t.vocoder_ms, t.wav_ms);
    free(prompt_ids); free(prompt_frames); free(reference);
    for (int64_t c = 0; c < n_chunks; c++) free(chunks[c]);
    free(chunks); free(chunk_lens); free(wave);
    pthread_mutex_unlock(&m->lock);
    return true;
done:
    free(prompt_ids); free(prompt_frames); free(reference);
    for (int64_t c = 0; c < n_chunks; c++) free(chunks[c]);
    free(chunks); free(chunk_lens); free(wave); free(wav);
    t.total_ms = music_ms() - started + t.load_ms;
    if (tim) *tim = t;
    pthread_mutex_unlock(&m->lock);
    return false;
}
