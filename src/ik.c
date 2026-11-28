/*
 * ik.c — Inkling-2-Mini: checkpoint load, dequantization and the text forward.
 *
 * The routed experts ship in two quantized forms and the mix is per layer:
 * NVFP4 (e2m1 nibbles, one e4m3 block scale per 16 values, rescaled by a
 * per-expert amax and one shared denominator) or MXFP8 (e4m3 values, one
 * power-of-two block scale per 32 values). Metal runs them directly from packed
 * arena bytes; CPU and A/B runs can retain the BF16 dequantized representation,
 * while image prefill materializes only the active experts into temporary BF16.
 *
 * The whole checkpoint is resident and mlock'd, so the compressor cannot touch
 * it and inference never pages. Packed mode keeps the expert payload near its
 * checkpoint size instead of widening it to BF16.
 */
#include "ik.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "h3v_gpu.h"
#include "json.h"
#include "log.h"
#include "mem.h"
#include "metrics.h"
#include "util.h"

#define IK_CHUNK 64   /* tokens per prefill slice */
#define CWICK_IK_BATCH_MAX 128

/* ------------------------------------------------------------------ */
/* fp4 / fp8 tables                                                    */
/* ------------------------------------------------------------------ */

static float g_e2m1[16];
static float g_e4m3[256];
static int g_tables_ready;

static void ik_build_tables(void) {
    if (g_tables_ready) return;
    static const float v[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    for (int i = 0; i < 8; i++) {
        g_e2m1[i] = v[i];
        g_e2m1[i + 8] = -v[i];
    }
    for (int code = 0; code < 256; code++) {
        int e = (code & 0x7F) >> 3;
        int m = code & 7;
        float val = (e == 0) ? ldexpf((float)m, -9)
                             : ldexpf(1.0f + (float)m / 8.0f, e - 7);
        g_e4m3[code] = (code & 0x80) ? -val : val;
    }
    g_e4m3[0x7F] = NAN;
    g_e4m3[0xFF] = NAN;
    g_tables_ready = 1;
}

/* ------------------------------------------------------------------ */
/* config                                                              */
/* ------------------------------------------------------------------ */

static int64_t ik_int(json_value *o, const char *k, int64_t d) {
    return cwick_json_int(cwick_json_get(o, k), d);
}
static double ik_num(json_value *o, const char *k, double d) {
    return cwick_json_num(cwick_json_get(o, k), d);
}

static cwick_status ik_cfg_load(const char *root, ik_cfg *c, cwick_error *err) {
    char path[1200];
    cwick_path_join(path, sizeof(path), root, "config.json");
    char *text = NULL;
    size_t len = 0;
    if (cwick_read_file(path, &text, &len) != 0) {
        cwick_error_set(err, CWICK_ERR_IO, "cannot read %s", path);
        return CWICK_ERR_IO;
    }
    cwick_error e2;
    cwick_error_clear(&e2);
    json_value *j = cwick_json_parse(text, len, &e2);
    free(text);
    if (!j) {
        cwick_error_set(err, CWICK_ERR_PARSE, "config.json is not valid JSON");
        return CWICK_ERR_PARSE;
    }
    memset(c, 0, sizeof(*c));
    json_value *tc = cwick_json_get(j, "text_config");
    if (!tc) tc = j;
    json_value *vc = cwick_json_get(j, "vision_config");

    c->hidden = ik_int(tc, "hidden_size", 0);
    c->n_layers = ik_int(tc, "num_hidden_layers", 0);
    c->n_heads = ik_int(tc, "num_attention_heads", 0);
    c->n_kv = ik_int(tc, "num_key_value_heads", 0);
    c->head_dim = ik_int(tc, "head_dim", 0);
    c->vocab = ik_int(tc, "vocab_size", 0);
    c->unpadded_vocab = ik_int(tc, "unpadded_vocab_size", 0);
    c->d_rel = ik_int(tc, "d_rel", 0);
    c->rel_extent = ik_int(tc, "rel_extent", 0);
    c->swa_window = ik_int(tc, "sliding_window_size", 0);
    c->sconv_k = ik_int(tc, "sconv_kernel_size", 4);
    c->res_block = ik_int(tc, "n_layers_per_attn_res_block", 0);
    c->n_experts = ik_int(tc, "n_routed_experts", 0);
    c->top_k = ik_int(tc, "num_experts_per_tok", 0);
    c->inter = ik_int(tc, "intermediate_size", 0);
    if (c->inter <= 0) c->inter = ik_int(tc, "moe_intermediate_size", 0);
    c->eos_id = ik_int(j, "eos_token_id", ik_int(tc, "eos_token_id", 0));

    c->rms_eps = (float)ik_num(tc, "rms_norm_eps", 1e-6);
    c->qk_clip = (float)ik_num(tc, "qk_l2_clip_threshold", 22.627416997969522);
    c->rel_clip = (float)ik_num(tc, "rel_l2_clip_threshold", 8.0);
    c->log_floor = (float)ik_num(tc, "log_scaling_n_floor", 8192);
    c->log_alpha = (float)ik_num(tc, "log_scaling_alpha", 0.1);
    c->route_scale = (float)ik_num(tc, "route_scale", 8.0);
    c->gate_softcap = (float)ik_num(tc, "mlp_gate_softcap", 4.0);
    c->up_softcap = (float)ik_num(tc, "mlp_up_softcap", 25.0);
    c->final_softcap = (float)ik_num(tc, "final_logit_softcapping", 30.0);
    c->image_token_id = ik_int(j, "image_token_id", 200054);

    json_value *lp = cwick_json_get(tc, "loop_config");
    if (!lp) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "config.json has no loop_config");
        cwick_json_free(j);
        return CWICK_ERR_CONTRACT;
    }
    c->loop_start = ik_int(lp, "core_start", 0);
    c->loop_end = ik_int(lp, "core_end", 0);
    c->loop_passes = ik_int(lp, "num_passes", 1);
    c->gate_rank = ik_int(lp, "gate_rank", 0);

    if (c->hidden != 2304 || c->n_heads * c->head_dim != 3072 ||
        c->n_layers <= 0 || c->n_layers > IK_MAX_LAYERS ||
        c->n_experts <= 0 || c->n_experts > 128 ||
        c->top_k <= 0 || c->top_k > c->n_experts || c->d_rel <= 0 ||
        c->rel_extent <= 0 || c->swa_window <= 0 || c->res_block <= 0 ||
        c->loop_end <= c->loop_start || c->loop_end > c->n_layers ||
        c->gate_rank <= 0 || c->loop_passes < 1) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                        "inkling2 config is outside the shapes these kernels "
                        "implement (hidden=%lld heads=%lld kv=%lld layers=%lld "
                        "experts=%lld top=%lld dr=%lld loop=[%lld,%lld) x%lld)",
                        (long long)c->hidden, (long long)c->n_heads,
                        (long long)c->n_kv, (long long)c->n_layers,
                        (long long)c->n_experts, (long long)c->top_k,
                        (long long)c->d_rel, (long long)c->loop_start,
                        (long long)c->loop_end, (long long)c->loop_passes);
        cwick_json_free(j);
        return CWICK_ERR_CONTRACT;
    }

    for (int64_t i = 0; i < c->n_layers; i++) c->is_swa[i] = true;
    json_value *local = cwick_json_get(tc, "local_layer_ids");
    if (local && local->kind == JSON_ARRAY) {
        for (int64_t i = 0; i < c->n_layers; i++) c->is_swa[i] = false;
        for (size_t k = 0; k < cwick_json_array_len(local); k++) {
            int64_t li = cwick_json_int(cwick_json_array_at(local, k), -1);
            if (li >= 0 && li < c->n_layers) c->is_swa[li] = true;
        }
    }

    if (vc) {
        c->v_hidden = ik_int(vc, "hidden_size", 1024);
        c->v_layers = ik_int(vc, "num_hidden_layers", 23);
        c->v_heads = ik_int(vc, "num_attention_heads", 16);
        c->v_patch = ik_int(vc, "native_patch_size", 20);
        c->v_merge = ik_int(vc, "adapter_merge_size", 2);
        c->v_grid = ik_int(vc, "posemb_grid_size", 32);
        c->v_proj = ik_int(vc, "projection_hidden_dim", 4096);
        c->v_inter = ik_int(vc, "intermediate_size", 0);
        if (c->v_inter <= 0)
            c->v_inter = (int64_t)((double)c->v_hidden * ik_num(vc, "mlp_ratio", 4.0));
        c->v_ln_eps = (float)ik_num(vc, "layer_norm_eps", 1e-5);
        if (c->v_hidden <= 0 || c->v_layers <= 0 || c->v_heads <= 0 ||
            c->v_inter <= 0 || c->v_patch <= 0 || c->v_merge <= 0 ||
            c->v_grid <= 0 || c->v_proj <= 0 || c->v_hidden % c->v_heads) {
            cwick_error_set(err, CWICK_ERR_CONTRACT,
                            "inkling2 vision config has invalid geometry");
            cwick_json_free(j);
            return CWICK_ERR_CONTRACT;
        }
    }
    cwick_json_free(j);
    return CWICK_OK;
}

/* ------------------------------------------------------------------ */
/* arena                                                               */
/* ------------------------------------------------------------------ */

typedef struct {
    uint8_t *base;  /* NULL while planning */
    size_t off;
    size_t cap;
} ik_bump;

static size_t ik_bump_off(ik_bump *b, size_t n, size_t align) {
    if (align < 64) align = 64;
    size_t pad = (align - (b->off % align)) % align;
    size_t at = b->off + pad;
    b->off = at + n;
    return at;
}

/* ------------------------------------------------------------------ */
/* math helpers                                                        */
/* ------------------------------------------------------------------ */

typedef struct {
    double linear, attention, moe, conv, mix, control, total;
    double route, up, act, down, combine, pair;
    int64_t layers;
} ik_prof;

static ik_prof g_prof;
static int g_prof_mode = -1;

static bool ik_prof_on(void) {
    if (g_prof_mode < 0) {
        const char *e = getenv("CWICK_IK_PROF");
        g_prof_mode = e && *e && strcmp(e, "0") != 0;
    }
    return g_prof_mode != 0;
}

static double ik_prof_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void ik_prof_add(double *slot, double t0) {
    *slot += ik_prof_now() - t0;
}

void ik_round_bf16(float *x, int64_t n) {
    for (int64_t i = 0; i < n; i++) x[i] = cwick_bf16_to_f32(cwick_f32_to_bf16(x[i]));
}

void ik_rms_norm_bf16(float *out, const float *in, int64_t n,
                      const float *weight, float eps) {
    cwick_rms_norm(out, in, n, weight, eps);
    ik_round_bf16(out, n);
}

static void ik_rms_unweighted(float *out, const float *in, int64_t n, float eps) {
    double ss = 0.0;
    for (int64_t i = 0; i < n; i++) ss += (double)in[i] * (double)in[i];
    float inv = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (int64_t i = 0; i < n; i++) out[i] = in[i] * inv;
}

static void ik_l2_clip(float *v, int64_t heads, int64_t dim, const float *scale,
                       float threshold) {
    for (int64_t h = 0; h < heads; h++) {
        float *x = v + h * dim;
        const float *s = scale ? scale + h * dim : NULL;
        double ss = 0.0;
        if (s) {
            for (int64_t i = 0; i < dim; i++) { x[i] *= s[i]; ss += (double)x[i] * (double)x[i]; }
        } else {
            for (int64_t i = 0; i < dim; i++) ss += (double)x[i] * (double)x[i];
        }
        float k = threshold / (sqrtf((float)ss) + 1e-6f);
        if (k < 1.0f)
            for (int64_t i = 0; i < dim; i++) x[i] *= k;
        ik_round_bf16(x, dim);
    }
}

typedef struct {
    const void *w;
    const float *x;
    float *out;
    int64_t k, cols;
} ik_linear_arg;

static bool ik_gpu_enabled(void) {
    const char *e = getenv("CWICK_IK_GPU");
    return cwick_metal_backend_active() && (!e || strcmp(e, "0") != 0);
}

static bool ik_packed_experts_enabled(void) {
    const char *e = getenv("CWICK_IK_PACKED_EXPERTS");
    if (e && strcmp(e, "0") == 0) return false;
    return ik_gpu_enabled() && cwick_metal_ik_packed_ready() != 0;
}

static int ik_quant_config_load(ik_model *m, const char *root, cwick_error *err) {
    char path[1200];
    cwick_path_join(path, sizeof(path), root, "hf_quant_config.json");
    char *text = NULL;
    size_t len = 0;
    if (cwick_read_file(path, &text, &len) != 0) {
        cwick_error_set(err, CWICK_ERR_IO, "cannot read %s", path);
        return -1;
    }
    cwick_error e2;
    cwick_error_clear(&e2);
    json_value *j = cwick_json_parse(text, len, &e2);
    free(text);
    if (!j) {
        cwick_error_set(err, CWICK_ERR_PARSE, "hf_quant_config.json is not valid JSON");
        return -1;
    }
    json_value *model = cwick_json_get(j, "model");
    json_value *groups = cwick_json_get(model ? model : j, "parameter_groups");
    if (!groups || groups->kind != JSON_ARRAY) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                        "hf_quant_config.json has no model.parameter_groups");
        cwick_json_free(j);
        return -1;
    }
    for (size_t g = 0; g < cwick_json_array_len(groups); g++) {
        json_value *group = cwick_json_array_at(groups, g);
        const char *dtype = cwick_json_str(cwick_json_get(group, "compute_dtype"));
        ik_expert_format format;
        if (dtype && strcmp(dtype, "nvfp4") == 0) format = IK_EXPERT_NVFP4;
        else if (dtype && strcmp(dtype, "mxfp8") == 0) format = IK_EXPERT_MXFP8;
        else continue;
        json_value *tensors = cwick_json_get(group, "tensors");
        if (!tensors || tensors->kind != JSON_ARRAY) continue;
        for (size_t i = 0; i < cwick_json_array_len(tensors); i++) {
            const char *name = cwick_json_str(cwick_json_array_at(tensors, i));
            if (!name) continue;
            int layer = -1, bit = 0, end = 0;
            if (sscanf(name, "model.llm.layers.%d.mlp.experts.w13_weight%n",
                       &layer, &end) == 1 && end == (int)strlen(name)) bit = 1;
            else if (sscanf(name, "model.llm.layers.%d.mlp.experts.w2_weight%n",
                            &layer, &end) == 1 && end == (int)strlen(name)) bit = 2;
            if (!bit) continue;
            if (layer < 0 || layer >= m->c.n_layers ||
                (m->expert_seen[layer] & bit)) {
                cwick_error_set(err, CWICK_ERR_CONTRACT,
                                "hf_quant_config.json has invalid expert tensor %s", name);
                cwick_json_free(j);
                return -1;
            }
            if ((m->expert_seen[layer] & ~bit) &&
                m->expert_format[layer] != (uint8_t)format) {
                cwick_error_set(err, CWICK_ERR_CONTRACT,
                                "layer %d mixes expert quantization formats", layer);
                cwick_json_free(j);
                return -1;
            }
            m->expert_seen[layer] |= (uint8_t)bit;
            m->expert_format[layer] = (uint8_t)format;
        }
    }
    cwick_json_free(j);
    for (int64_t l = 0; l < m->c.n_layers; l++) {
        if (m->expert_seen[l] != 3) {
            cwick_error_set(err, CWICK_ERR_CONTRACT,
                            "hf_quant_config.json does not describe both expert matrices "
                            "for layer %lld", (long long)l);
            return -1;
        }
    }
    return 0;
}

static int ik_gpu_attn_mode(void) {
    static int cached = -2;
    if (cached == -2) {
        const char *e = getenv("CWICK_IK_GPU_ATTN");
        cached = !e ? 0 : (strcmp(e, "0") != 0);
    }
    return cached;
}

typedef struct {
    const void *w;
    const float *x;
    float *out;
    int64_t rows, k, cols;
} ik_linear_item;

static void ik_linear_items(ik_model *m, const ik_linear_item *a, int64_t count) {
    const bool prof = ik_prof_on();
    const double prof_t0 = prof ? ik_prof_now() : 0.0;
    const bool gpu = ik_gpu_enabled();
    int64_t done = 0;
    while (done < count) {
        int64_t take = 1;
        while (done + take < count &&
               (a[done + take].rows == 1) == (a[done].rows == 1)) take++;
        if (a[done].rows == 1 && take > 128) take = 128;
        if (a[done].rows > 1 && take > 64) take = 64;
        bool ok = false;
        if (gpu && a[done].rows == 1) {
            void *batch = cwick_metal_bf16_batch_begin();
            ok = batch != NULL;
            for (int64_t i = 0; i < take && ok; i++)
                ok = cwick_metal_bf16_batch_add(batch, a[done + i].w,
                                                a[done + i].x, a[done + i].out,
                                                (uint32_t)a[done + i].k,
                                                (uint32_t)a[done + i].cols) != 0;
            if (ok) ok = cwick_metal_bf16_batch_end(batch);
        } else if (gpu && a[done].rows > 1 && h3v_gpu_on()) {
            h3v_gpu_bf16_item items[64];
            for (int64_t i = 0; i < take; i++) {
                items[i] = (h3v_gpu_bf16_item){
                    a[done + i].x, a[done + i].w, a[done + i].out,
                    (int)a[done + i].rows, (int)a[done + i].k,
                    (int)a[done + i].cols, (int)a[done + i].k,
                    (int)a[done + i].cols};
            }
            ok = h3v_gpu_gemm_bf16_batch(items, (int)take) != 0;
        }
        if (ok) {
            m->gpu_calls += take;
        } else {
            for (int64_t i = 0; i < take; i++) {
                const ik_linear_item *it = &a[done + i];
                if (it->rows == 1)
                    cwick_gemv_bf16(it->out, it->k, it->cols, it->x, it->w);
                else
                    cwick_gemm_bf16(it->out, it->rows, it->k, it->cols,
                                    it->x, it->w);
            }
        }
        done += take;
    }
    if (prof) ik_prof_add(&g_prof.linear, prof_t0);
}

static void ik_linear_group(ik_model *m, int64_t n, const ik_linear_arg *a,
                            int64_t count) {
    ik_linear_item items[128];
    if (count > 128) abort();
    for (int64_t i = 0; i < count; i++)
        items[i] = (ik_linear_item){a[i].w, a[i].x, a[i].out, n, a[i].k, a[i].cols};
    ik_linear_items(m, items, count);
}

typedef struct {
    ik_linear_item linear;
    const ik_expert_pack *pack;
    int64_t expert;
} ik_expert_item;

static void ik_packed_weight_row(const ik_expert_item *it, const void **weight,
                                 const uint8_t **scale, const float **global) {
    const ik_expert_pack *q = it->pack;
    const int64_t k = it->linear.k, cols = it->linear.cols;
    const int64_t row_values = cols * k;
    const size_t wstride = q->format == IK_EXPERT_NVFP4
        ? (size_t)row_values / 2u : (size_t)row_values;
    const size_t sstride = q->format == IK_EXPERT_NVFP4
        ? (size_t)row_values / 16u : (size_t)row_values / 32u;
    *weight = (const uint8_t *)q->packed + (size_t)it->expert * wstride;
    *scale = q->scale + (size_t)it->expert * sstride;
    *global = q->format == IK_EXPERT_NVFP4 ? q->global_scale + it->expert : NULL;
}

typedef struct {
    const ik_expert_pack *w13, *w2;
    int64_t expert;
    const float *x;
    float *out;
    float gamma;
} ik_expert_pair;

static bool ik_packed_pair_batch(ik_model *m, const ik_expert_pair *pairs,
                                 int64_t count) {
    const int64_t H = m->c.hidden, I = m->c.inter;
    for (int64_t done = 0; done < count; done += CWICK_IK_BATCH_MAX) {
        int64_t take = count - done;
        if (take > CWICK_IK_BATCH_MAX) take = CWICK_IK_BATCH_MAX;
        void *batch = cwick_metal_ik_pair_batch_begin();
        bool ok = batch != NULL;
        for (int64_t i = 0; i < take && ok; i++) {
            const ik_expert_pair *p = &pairs[done + i];
            ik_expert_item w13 = {.pack = p->w13, .expert = p->expert,
                                  .linear = {NULL, NULL, NULL, 1, H, 2 * I}};
            ik_expert_item w2 = {.pack = p->w2, .expert = p->expert,
                                 .linear = {NULL, NULL, NULL, 1, I, H}};
            const void *packed13, *packed2;
            const uint8_t *scale13, *scale2;
            const float *global13, *global2;
            ik_packed_weight_row(&w13, &packed13, &scale13, &global13);
            ik_packed_weight_row(&w2, &packed2, &scale2, &global2);
            ok = cwick_metal_ik_pair_batch_add(
                batch, packed13, scale13, global13 ? *global13 : 0.0f,
                packed2, scale2, global2 ? *global2 : 0.0f,
                p->x, p->out, p->gamma, H, 2 * I, I, H, I,
                (int)p->w13->format, m->c.gate_softcap, m->c.up_softcap) != 0;
        }
        if (ok) ok = cwick_metal_ik_pair_batch_end(batch) != 0;
        if (!ok) return false;
        m->gpu_calls += take;
    }
    return true;
}

/* Safety net if a packed Metal batch cannot be submitted. It reads the same
 * resident bytes and mapping as the shader; no BF16 expert copy is needed. */
static void ik_packed_gemv_cpu(const ik_expert_item *it, int64_t half) {
    const void *weight;
    const uint8_t *scale;
    const float *global;
    ik_packed_weight_row(it, &weight, &scale, &global);
    const int64_t k = it->linear.k, cols = it->linear.cols;
    const uint8_t *w = (const uint8_t *)weight;
    for (int64_t c = 0; c < cols; c++) {
        const int64_t row = half ? ((c < half) ? 2 * c : 2 * (c - half) + 1) : c;
        double acc = 0.0;
        if (it->pack->format == IK_EXPERT_NVFP4) {
            const uint8_t *wr = w + (size_t)row * (size_t)(k / 2);
            const uint8_t *sr = scale + (size_t)row * (size_t)(k / 16);
            const float gs = global ? *global : 1.0f;
            for (int64_t j = 0; j < k; j++) {
                const uint8_t byte = wr[j >> 1];
                const float v = g_e2m1[(j & 1) ? byte >> 4 : byte & 15];
                acc += (double)(v * g_e4m3[sr[j >> 4]] * gs) * it->linear.x[j];
            }
        } else {
            const uint8_t *wr = w + (size_t)row * (size_t)k;
            const uint8_t *sr = scale + (size_t)row * (size_t)(k / 32);
            for (int64_t j = 0; j < k; j++) {
                const float v = g_e4m3[wr[j]] * ldexpf(1.0f, (int)sr[j >> 5] - 127);
                acc += (double)v * it->linear.x[j];
            }
        }
        it->linear.out[c] = (float)acc;
    }
}

typedef struct {
    const uint8_t *w, *scale;
    const float *global;
    uint16_t *out;
    int64_t k, cols, half;
    ik_expert_format format;
} ik_dequant_arg;

static void ik_dequant_range(void *v, int64_t lo, int64_t hi) {
    const ik_dequant_arg *a = (const ik_dequant_arg *)v;
    for (int64_t c = lo; c < hi; c++) {
        const int64_t row = a->half ? ((c < a->half) ? 2 * c : 2 * (c - a->half) + 1) : c;
        for (int64_t j = 0; j < a->k; j++) {
            float value;
            if (a->format == IK_EXPERT_NVFP4) {
                const uint8_t byte = a->w[(size_t)row * (size_t)(a->k / 2) + (size_t)(j >> 1)];
                const float q = g_e2m1[(j & 1) ? byte >> 4 : byte & 15];
                const float s = g_e4m3[a->scale[(size_t)row * (size_t)(a->k / 16) + (size_t)(j >> 4)]];
                value = q * s * (a->global ? *a->global : 1.0f);
            } else {
                const float q = g_e4m3[a->w[(size_t)row * (size_t)a->k + (size_t)j]];
                const int e = (int)a->scale[(size_t)row * (size_t)(a->k / 32) + (size_t)(j >> 5)] - 127;
                value = q * ldexpf(1.0f, e);
            }
            a->out[(size_t)c * (size_t)a->k + (size_t)j] = cwick_f32_to_bf16(value);
        }
    }
}

static void ik_dequant_expert_bf16(const ik_expert_item *it, int64_t half,
                                   uint16_t *out) {
    const void *weight;
    const uint8_t *scale;
    const float *global;
    ik_packed_weight_row(it, &weight, &scale, &global);
    ik_dequant_arg a = {(const uint8_t *)weight, scale, global, out,
                        it->linear.k, it->linear.cols, half, it->pack->format};
    cwick_pool_init();
    cwick_pool_for_items(a.cols, &a, ik_dequant_range);
}

static void ik_linear_expert_items(ik_model *m, const ik_expert_item *items,
                                   int64_t half, int64_t count, bool use_packed) {
    if (!use_packed) {
        if (items[0].linear.w) {
            ik_linear_item bf16[128];
            if (count > 128) abort();
            for (int64_t i = 0; i < count; i++) bf16[i] = items[i].linear;
            ik_linear_items(m, bf16, count);
            return;
        }
        if (ik_gpu_enabled() && h3v_gpu_on()) {
            for (int64_t done = 0; done < count; done += 64) {
                int64_t take = count - done < 64 ? count - done : 64;
                size_t values = 0;
                for (int64_t i = 0; i < take; i++)
                    values += (size_t)items[done + i].linear.k * (size_t)items[done + i].linear.cols;
                uint16_t *weights = (uint16_t *)malloc(values * sizeof(uint16_t));
                h3v_gpu_bf16_item gpu_items[64];
                if (!weights) abort();
                size_t offset = 0;
                for (int64_t i = 0; i < take; i++) {
                    const ik_linear_item *it = &items[done + i].linear;
                    ik_dequant_expert_bf16(&items[done + i], half, weights + offset);
                    gpu_items[i] = (h3v_gpu_bf16_item){
                        it->x, weights + offset, it->out, (int)it->rows,
                        (int)it->k, (int)it->cols, (int)it->k, (int)it->cols};
                    offset += (size_t)it->k * (size_t)it->cols;
                }
                if (!h3v_gpu_gemm_bf16_batch(gpu_items, (int)take)) {
                    for (int64_t i = 0; i < take; i++) {
                        const ik_linear_item *it = &items[done + i].linear;
                        const uint16_t *w = gpu_items[i].w;
                        ik_round_bf16((float *)it->x, it->rows * it->k);
                        if (it->rows == 1)
                            cwick_gemv_bf16(it->out, it->k, it->cols, it->x, w);
                        else
                            cwick_gemm_bf16(it->out, it->rows, it->k, it->cols, it->x, w);
                    }
                }
                free(weights);
            }
            return;
        }
        uint16_t *weight = NULL;
        size_t cap = 0;
        for (int64_t i = 0; i < count; i++) {
            const ik_linear_item *it = &items[i].linear;
            size_t need = (size_t)it->k * (size_t)it->cols;
            if (need > cap) {
                uint16_t *next = (uint16_t *)realloc(weight, need * sizeof(uint16_t));
                if (!next) abort();
                weight = next;
                cap = need;
            }
            ik_dequant_expert_bf16(&items[i], half, weight);
            ik_round_bf16((float *)it->x, it->rows * it->k);
            if (it->rows == 1)
                cwick_gemv_bf16(it->out, it->k, it->cols, it->x, weight);
            else
                cwick_gemm_bf16(it->out, it->rows, it->k, it->cols, it->x, weight);
        }
        free(weight);
        return;
    }
    int64_t total = 0;
    for (int64_t i = 0; i < count; i++) total += items[i].linear.rows;
    ik_expert_item *rows = (ik_expert_item *)malloc((size_t)total * sizeof(*rows));
    if (!rows) abort();
    int64_t at = 0;
    for (int64_t i = 0; i < count; i++) {
        for (int64_t r = 0; r < items[i].linear.rows; r++) {
            rows[at++] = (ik_expert_item){
                .linear = {items[i].linear.w,
                           items[i].linear.x + r * items[i].linear.k,
                           items[i].linear.out + r * items[i].linear.cols,
                           1, items[i].linear.k, items[i].linear.cols},
                .pack = items[i].pack, .expert = items[i].expert};
        }
    }

    for (int64_t done = 0; done < total; done += CWICK_IK_BATCH_MAX) {
        int64_t take = total - done;
        if (take > CWICK_IK_BATCH_MAX) take = CWICK_IK_BATCH_MAX;
        bool ok = false;
        if (ik_gpu_enabled()) {
            void *batch = cwick_metal_ik_packed_batch_begin();
            ok = batch != NULL;
            for (int64_t i = 0; i < take && ok; i++) {
                const ik_expert_item *it = &rows[done + i];
                const void *weight;
                const uint8_t *scale;
                const float *global;
                ik_packed_weight_row(it, &weight, &scale, &global);
                ok = cwick_metal_ik_packed_batch_add(
                    batch, weight, scale, global ? *global : 0.0f,
                    it->linear.x, it->linear.out, it->linear.k, it->linear.cols,
                    half, (int)it->pack->format) != 0;
            }
            if (ok) ok = cwick_metal_ik_packed_batch_end(batch) != 0;
        }
        if (ok) {
            m->gpu_calls += take;
        } else {
            static int warned;
            if (!warned) {
                warned = 1;
                CWICK_INFO("ikwarn", "packed expert Metal batch unavailable; "
                                   "using the CPU packed GEMV fallback");
            }
            for (int64_t i = 0; i < take; i++) ik_packed_gemv_cpu(&rows[done + i], half);
        }
    }
    free(rows);
}

/* ------------------------------------------------------------------ */
/* scratch                                                             */
/* ------------------------------------------------------------------ */

typedef struct ik_scratch {
    int64_t n;
    float *x, *xn;
    float *ain, *bin, *cin;
    float *q, *kk, *vv, *rr;
    float *ao, *gg;
    float *tt, *mo;
    float *dynk, *dynv;
    float *gsub, *asub, *dsub;
    float *blocks, *pre_blocks, *prev_blocks;
    int *nblocks;
    int pre_nb, prev_nb;
    float *pre_h, *prev_h;
    float *msgs;
    float *scores;
    int32_t *sel;
    float *wts;
    float *mixout;   /* [3][H] */
    float *mbank;    /* [IK_MAX_BLOCKS+1][H] */
    float *normbuf;  /* [IK_MAX_BLOCKS+1][H] */
    float *ascores;  /* [n][score_keys] attention softmax scratch, per token */
    int64_t score_keys;
    bool ready;
} ik_scratch;

static void ik_scratch_free(void);

static ik_scratch g_scr;

static void ik_scratch_destroy(ik_scratch *s) {
    free(s->x); free(s->xn); free(s->ain); free(s->bin); free(s->cin);
    free(s->q); free(s->kk); free(s->vv); free(s->rr);
    free(s->ao); free(s->gg); free(s->tt); free(s->mo);
    free(s->dynk); free(s->dynv);
    free(s->gsub); free(s->asub); free(s->dsub);
    free(s->blocks); free(s->pre_blocks); free(s->prev_blocks);
    free(s->nblocks); free(s->pre_h); free(s->prev_h);
    free(s->msgs); free(s->scores); free(s->sel); free(s->wts);
    free(s->mixout); free(s->mbank); free(s->normbuf); free(s->ascores);
    memset(s, 0, sizeof(*s));
}

static ik_scratch *ik_scratch_get(const ik_cfg *c, int64_t n) {
    if (g_scr.ready && g_scr.n >= n) return &g_scr;
    ik_scratch_destroy(&g_scr);
    ik_scratch *s = &g_scr;
    const int64_t H = c->hidden, QD = c->n_heads * c->head_dim;
    const int64_t KVD = c->n_kv * c->head_dim, RD = c->n_heads * c->d_rel;
    const int64_t I2 = 2 * c->inter;
#define AL(ptr, count) do { \
    (ptr) = (float *)calloc((size_t)(count), sizeof(float)); \
    if (!(ptr)) { ik_scratch_destroy(s); return NULL; } \
} while (0)
    AL(s->x, n * H); AL(s->xn, n * H);
    AL(s->ain, n * H); AL(s->bin, n * H); AL(s->cin, n * H);
    AL(s->q, n * QD); AL(s->kk, n * KVD); AL(s->vv, n * KVD); AL(s->rr, n * RD);
    AL(s->ao, n * QD); AL(s->gg, n * QD);
    AL(s->tt, n * H); AL(s->mo, n * H);
    AL(s->dynk, n * c->n_kv * c->sconv_k);
    AL(s->dynv, n * c->n_kv * c->sconv_k);
    AL(s->gsub, n * I2); AL(s->asub, n * c->inter); AL(s->dsub, n * H);
    AL(s->blocks, n * IK_MAX_BLOCKS * H);
    AL(s->pre_blocks, n * IK_MAX_BLOCKS * H);
    AL(s->prev_blocks, n * IK_MAX_BLOCKS * H);
    AL(s->pre_h, n * H); AL(s->prev_h, n * H);
    AL(s->msgs, n * 4 * H);
    AL(s->scores, n * c->n_experts);
    AL(s->wts, n * c->top_k);
    AL(s->mixout, 3 * H);
    AL(s->mbank, (IK_MAX_BLOCKS + 1) * H);
    AL(s->normbuf, (IK_MAX_BLOCKS + 1) * H);
    s->sel = (int32_t *)calloc((size_t)(n * c->top_k), sizeof(int32_t));
    s->nblocks = (int *)calloc((size_t)n, sizeof(int));
    if (!s->sel || !s->nblocks) { ik_scratch_destroy(s); return NULL; }
#undef AL
    s->n = n;
    s->ready = true;
    return s;
}

/* One softmax slot per visible key, not per chunk: a full-attention token at
 * position p sees p+1 keys. Grow on demand so no worker allocates in the pool. */
static int ik_scratch_scores(ik_scratch *s, int64_t n, int64_t keys) {
    if (keys < 1) keys = 1;
    if (s->score_keys >= keys) return 0;
    float *p = (float *)realloc(s->ascores, (size_t)(n * keys) * sizeof(float));
    if (!p) return -1;
    s->ascores = p;
    s->score_keys = keys;
    return 0;
}

static void ik_scratch_free(void) { ik_scratch_destroy(&g_scr); }

/* ------------------------------------------------------------------ */
/* weight loading                                                      */
/* ------------------------------------------------------------------ */

static const cwick_st_tensor *ik_st(ik_model *m, const char *name,
                                    cwick_error *err) {
    static cwick_st_tensor t;
    cwick_error e;
    cwick_error_clear(&e);
    if (cwick_safetensors_get(&m->saf, name, &t, &e) != CWICK_OK) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                        "Inkling-2-Mini checkpoint is missing %s", name);
        return NULL;
    }
    return &t;
}

static const void *ik_copy_bf16(ik_model *m, ik_bump *b, const char *name,
                                int64_t *rows_out, int64_t *cols_out,
                                cwick_error *err) {
    const cwick_st_tensor *t = ik_st(m, name, err);
    if (!t) return NULL;
    if (t->dtype != CWICK_ST_BF16) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "%s: expected BF16", name);
        return NULL;
    }
    int64_t rows = t->rows, cols = t->cols;
    size_t bytes = (size_t)rows * (size_t)cols * 2u;
    size_t at = ik_bump_off(b, bytes, 64);
    if (rows_out) *rows_out = rows;
    if (cols_out) *cols_out = cols;
    if (!b->base) return NULL;
    if (b->off > b->cap) return NULL;
    memcpy(b->base + at, t->data, bytes);
    return b->base + at;
}

static float *ik_copy_f32(ik_model *m, ik_bump *b, const char *name,
                          int64_t count, cwick_error *err) {
    const cwick_st_tensor *t = ik_st(m, name, err);
    if (!t) return NULL;
    int64_t have = t->rows * t->cols;
    if (count > 0 && have < count) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "%s: short tensor (%lld < %lld)",
                        name, (long long)have, (long long)count);
        return NULL;
    }
    if (count <= 0) count = have;
    size_t at = ik_bump_off(b, (size_t)count * sizeof(float), 64);
    if (!b->base) return NULL;
    if (b->off > b->cap) return NULL;
    float *dst = (float *)(b->base + at);
    if (t->dtype == CWICK_ST_F32) {
        memcpy(dst, t->data, (size_t)count * sizeof(float));
    } else if (t->dtype == CWICK_ST_BF16) {
        const uint16_t *src = (const uint16_t *)t->data;
        for (int64_t i = 0; i < count; i++) dst[i] = cwick_bf16_to_f32(src[i]);
    } else {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "%s: unsupported dtype", name);
        return NULL;
    }
    return dst;
}

/* [k][ch] taps for the dynamic short convolution (tap 0 = current token). */
static float *ik_copy_taps(ik_model *m, ik_bump *b, const char *name,
                           int64_t k, int64_t ch, cwick_error *err) {
    const cwick_st_tensor *t = ik_st(m, name, err);
    if (!t) return NULL;
    if (t->rows * t->cols != k * ch) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "%s: expected [%lld,%lld]",
                        name, (long long)k, (long long)ch);
        return NULL;
    }
    size_t at = ik_bump_off(b, (size_t)(k * ch) * sizeof(float), 64);
    if (!b->base) return NULL;
    if (b->off > b->cap) return NULL;
    float *dst = (float *)(b->base + at);
    const uint16_t *src = (const uint16_t *)t->data;
    for (int64_t i = 0; i < k * ch; i++) dst[i] = cwick_bf16_to_f32(src[i]);
    return dst;
}

void ik_static_conv_taps(float *dst, const uint16_t *src, int64_t k, int64_t ch) {
    for (int64_t c = 0; c < ch; c++)
        for (int64_t j = 0; j < k; j++)
            dst[(k - 1 - j) * ch + c] = cwick_bf16_to_f32(src[c * k + j]);
}

/* Sliding K/V conv1d weight is [ch, 1, k], oldest tap first. The kernel uses
 * [lag][ch], newest tap first, so reverse the tap axis while transposing. */
static float *ik_copy_conv_transpose(ik_model *m, ik_bump *b, const char *name,
                                     int64_t k, int64_t ch, cwick_error *err) {
    const cwick_st_tensor *t = ik_st(m, name, err);
    if (!t) return NULL;
    if (t->rows * t->cols != k * ch) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "%s: expected [%lld,%lld]",
                        name, (long long)ch, (long long)k);
        return NULL;
    }
    size_t at = ik_bump_off(b, (size_t)(k * ch) * sizeof(float), 64);
    if (!b->base || b->off > b->cap) return NULL;
    float *dst = (float *)(b->base + at);
    const uint16_t *src = (const uint16_t *)t->data;
    ik_static_conv_taps(dst, src, k, ch);
    return dst;
}

/* Depthwise conv1d weight [ch, 1, k] -> [ch][k] F32 (tap 0 = k-1 steps back). */
static float *ik_copy_depthwise(ik_model *m, ik_bump *b, const char *name,
                                int64_t ch, int64_t k, cwick_error *err) {
    const cwick_st_tensor *t = ik_st(m, name, err);
    if (!t) return NULL;
    if (t->rows * t->cols != ch * k) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "%s: expected [%lld,1,%lld]",
                        name, (long long)ch, (long long)k);
        return NULL;
    }
    size_t at = ik_bump_off(b, (size_t)(ch * k) * sizeof(float), 64);
    if (!b->base) return NULL;
    if (b->off > b->cap) return NULL;
    float *dst = (float *)(b->base + at);
    const uint16_t *src = (const uint16_t *)t->data;
    for (int64_t i = 0; i < ch * k; i++) dst[i] = cwick_bf16_to_f32(src[i]);
    return dst;
}

/* ---- expert dequantization ---------------------------------------- */

typedef struct {
    const uint8_t *packed;
    const uint8_t *scale;
    const float *amax;
    float den;
    int64_t rows, cols, half_rows;
    uint16_t *out;
    int nvfp4;
} ik_deq_arg;

static void ik_deq_range(void *argp, int64_t e0, int64_t e1) {
    ik_deq_arg *a = (ik_deq_arg *)argp;
    const int64_t rows = a->rows, cols = a->cols;
    if (a->nvfp4) {
        const int64_t pcols = cols / 2, scols = cols / 16;
        for (int64_t e = e0; e < e1; e++) {
            const uint8_t *pw = a->packed + (size_t)e * (size_t)(rows * pcols);
            const uint8_t *ps = a->scale + (size_t)e * (size_t)(rows * scols);
            uint16_t *po = a->out + (size_t)e * (size_t)(rows * cols);
            float f = a->amax[e] / a->den;
            for (int64_t r = 0; r < rows; r++) {
                int64_t orow = a->half_rows > 0 ? ((r & 1) * a->half_rows + (r >> 1)) : r;
                uint16_t *dst = po + (size_t)orow * (size_t)cols;
                const uint8_t *wr = pw + (size_t)r * (size_t)pcols;
                const uint8_t *sr = ps + (size_t)r * (size_t)scols;
                for (int64_t c = 0; c < pcols; c++) {
                    float s = g_e4m3[sr[c >> 3]] * f;
                    uint8_t byte = wr[c];
                    dst[2 * c] = cwick_f32_to_bf16(g_e2m1[byte & 15] * s);
                    dst[2 * c + 1] = cwick_f32_to_bf16(g_e2m1[byte >> 4] * s);
                }
            }
        }
    } else {
        const int64_t scols = cols / 32;
        for (int64_t e = e0; e < e1; e++) {
            const uint8_t *pw = a->packed + (size_t)e * (size_t)(rows * cols);
            const uint8_t *ps = a->scale + (size_t)e * (size_t)(rows * scols);
            uint16_t *po = a->out + (size_t)e * (size_t)(rows * cols);
            for (int64_t r = 0; r < rows; r++) {
                int64_t orow = a->half_rows > 0 ? ((r & 1) * a->half_rows + (r >> 1)) : r;
                uint16_t *dst = po + (size_t)orow * (size_t)cols;
                const uint8_t *wr = pw + (size_t)r * (size_t)cols;
                const uint8_t *sr = ps + (size_t)r * (size_t)scols;
                for (int64_t c = 0; c < cols; c++)
                    dst[c] = cwick_f32_to_bf16(
                        g_e4m3[wr[c]] * ldexpf(1.0f, (int)sr[c >> 5] - 127));
            }
        }
    }
}

static void *ik_load_experts(ik_model *m, ik_bump *b, const char *base,
                             int64_t rows, int64_t cols, bool is_w13,
                             ik_expert_format format, ik_expert_pack *pack,
                             cwick_error *err) {
    const cwick_st_tensor *wt = ik_st(m, base, err);
    if (!wt) return NULL;
    /* ik_st returns a static descriptor: snapshot every field before the next
     * lookup overwrites it. */
    const uint8_t *weight_data = (const uint8_t *)wt->data;
    cwick_st_dtype weight_dtype = wt->dtype;
    const size_t weight_bytes = wt->data_bytes;
    const int64_t experts = m->c.n_experts;
    const bool nvfp4 = weight_dtype == CWICK_ST_U8;
    const bool packed = m->packed_experts;
    const bool keep_bf16 = !packed || m->dual_experts;
    const size_t row_values = (size_t)rows * (size_t)cols;
    const size_t weight_values = row_values * (size_t)experts;
    const size_t bf16_bytes = keep_bf16 ? weight_values * sizeof(uint16_t) : 0;
    const size_t packed_bytes = packed
        ? (nvfp4 ? weight_values / 2u : weight_values) : 0;
    const size_t scale_bytes = packed
        ? (nvfp4 ? weight_values / 16u : weight_values / 32u) : 0;
    const size_t at = ik_bump_off(b, bf16_bytes, 64);
    const size_t packed_at = packed ? ik_bump_off(b, packed_bytes, 64) : 0;
    const size_t scale_at = packed ? ik_bump_off(b, scale_bytes, 64) : 0;
    const size_t global_at = (packed && nvfp4)
        ? ik_bump_off(b, (size_t)experts * sizeof(float), 64) : 0;
    pack->format = packed ? format : IK_EXPERT_BF16;
    /* Both passes must take exactly the same bumps: the first one only
     * measures, the second one writes into the mapped arena. */
    if (!b->base || b->off > b->cap) return NULL;

    char sn[256], an[256], dn[256];
    snprintf(sn, sizeof(sn), "%s_scale", base);
    snprintf(an, sizeof(an), "%s_amax", base);
    snprintf(dn, sizeof(dn), "%s_outer_scale_denominator", base);
    const cwick_st_tensor *st = ik_st(m, sn, err);
    if (!st) return NULL;
    const uint8_t *scale_data = (const uint8_t *)st->data;
    const size_t expected_scale = nvfp4 ? weight_values / 16u : weight_values / 32u;
    if (weight_bytes != (nvfp4 ? weight_values / 2u : weight_values) ||
        st->data_bytes != expected_scale) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                        "%s: expert payload size mismatch (weight %zu/%zu, scale %zu/%zu)",
                        base, weight_bytes,
                        nvfp4 ? weight_values / 2u : weight_values,
                        st->data_bytes, expected_scale);
        return NULL;
    }
    if (packed && ((nvfp4 && format != IK_EXPERT_NVFP4) ||
                   (!nvfp4 && format != IK_EXPERT_MXFP8))) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "%s: dtype disagrees with hf_quant_config.json", base);
        return NULL;
    }

    float *global = NULL;
    const float *amax_raw = NULL;
    float denom = 1.0f;
    if (nvfp4) {
        const cwick_st_tensor *ax = ik_st(m, an, err);
        if (!ax) return NULL;
        amax_raw = (const float *)ax->data;
        const cwick_st_dtype amax_dtype = ax->dtype;
        const size_t amax_bytes = ax->data_bytes;
        const cwick_st_tensor *dt = ik_st(m, dn, err);
        if (!dt) return NULL;
        if (amax_dtype != CWICK_ST_F32 || amax_bytes < (size_t)experts * sizeof(float) ||
            dt->dtype != CWICK_ST_F32 || dt->data_bytes < sizeof(float)) {
            cwick_error_set(err, CWICK_ERR_CONTRACT, "%s: expected F32 amax/denominator", base);
            return NULL;
        }
        denom = *(const float *)dt->data;
        if (!(denom > 0.0f) || !isfinite(denom)) {
            cwick_error_set(err, CWICK_ERR_CONTRACT, "%s: bad denominator", dn);
            return NULL;
        }
        global = (float *)(b->base + global_at);
        for (int64_t e = 0; e < experts; e++) global[e] = amax_raw[e] / denom;
    }

    if (packed) {
        pack->packed = b->base + packed_at;
        pack->scale = b->base + scale_at;
        pack->global_scale = global;
        memcpy((void *)pack->packed, weight_data, packed_bytes);
        memcpy((void *)pack->scale, scale_data, scale_bytes);
        if (!keep_bf16) return NULL;
    }

    ik_deq_arg a;
    memset(&a, 0, sizeof(a));
    a.packed = weight_data;
    a.scale = scale_data;
    a.rows = rows;
    a.cols = cols;
    a.half_rows = is_w13 ? m->c.inter : 0;
    a.out = (uint16_t *)(b->base + at);
    a.nvfp4 = nvfp4;
    a.amax = amax_raw;
    a.den = denom;
    cwick_pool_init();
    cwick_pool_for(experts, &a, ik_deq_range);
    return b->base + at;
}

/* ------------------------------------------------------------------ */
/* vision weights                                                      */
/* ------------------------------------------------------------------ */

static int ik_vision_load(ik_model *m, ik_bump *b, cwick_error *err) {
    const ik_cfg *c = &m->c;
    if (m->vis) {   /* the sizing pass allocated one; this pass replaces it */
        free(m->vis->layers);
        free(m->vis);
        m->vis = NULL;
    }
    if (c->v_layers <= 0) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "Inkling-2-Mini vision tower is missing");
        return -1;
    }
    ik_vision *v = (ik_vision *)calloc(1, sizeof(ik_vision));
    if (!v) {
        cwick_error_set(err, CWICK_ERR_MEMORY, "vision tower allocation failed");
        return -1;
    }
    m->vis = v;
    v->patch_embed = ik_copy_bf16(m, b, "model.visual.backbone.embeddings.patch_embedding.weight",
                                  NULL, NULL, err);
    v->pos_embed = ik_copy_f32(m, b, "model.visual.backbone.embeddings.position_embedding",
                               c->v_grid * c->v_grid * c->v_hidden, err);
    v->class_embed = ik_copy_f32(m, b, "model.visual.backbone.embeddings.class_embedding",
                                 c->v_hidden, err);
    v->cls_pos = ik_copy_f32(m, b, "model.visual.backbone.embeddings.cls_position_embedding",
                             c->v_hidden, err);
    v->pre_ln_w = ik_copy_f32(m, b, "model.visual.backbone.ln_pre.weight", c->v_hidden, err);
    v->pre_ln_b = ik_copy_f32(m, b, "model.visual.backbone.ln_pre.bias", c->v_hidden, err);
    v->proj_in = ik_copy_bf16(m, b, "model.visual.proj_in.weight", NULL, NULL, err);
    v->proj_out = ik_copy_bf16(m, b, "model.visual.proj_out.weight", NULL, NULL, err);

    v->layers = (ik_vlayer *)calloc((size_t)c->v_layers, sizeof(ik_vlayer));
    if (!v->layers) {
        cwick_error_set(err, CWICK_ERR_MEMORY, "vision layer table allocation failed");
        return -1;
    }
    char n[320];
    for (int64_t l = 0; l < c->v_layers; l++) {
        ik_vlayer *vl = &v->layers[l];
#define VN(fmt) (snprintf(n, sizeof(n), "model.visual.backbone.layers.%lld." fmt, \
                          (long long)l), n)
        vl->ln1_w = ik_copy_f32(m, b, VN("ln_1.weight"), c->v_hidden, err);
        vl->ln1_b = ik_copy_f32(m, b, VN("ln_1.bias"), c->v_hidden, err);
        vl->ln2_w = ik_copy_f32(m, b, VN("ln_2.weight"), c->v_hidden, err);
        vl->ln2_b = ik_copy_f32(m, b, VN("ln_2.bias"), c->v_hidden, err);
        vl->q = ik_copy_bf16(m, b, VN("self_attn.q_proj.weight"), NULL, NULL, err);
        vl->k = ik_copy_bf16(m, b, VN("self_attn.k_proj.weight"), NULL, NULL, err);
        vl->v = ik_copy_bf16(m, b, VN("self_attn.v_proj.weight"), NULL, NULL, err);
        vl->o = ik_copy_bf16(m, b, VN("self_attn.out_proj.weight"), NULL, NULL, err);
        vl->q_b = ik_copy_f32(m, b, VN("self_attn.q_proj.bias"), c->v_hidden, err);
        vl->k_b = ik_copy_f32(m, b, VN("self_attn.k_proj.bias"), c->v_hidden, err);
        vl->v_b = ik_copy_f32(m, b, VN("self_attn.v_proj.bias"), c->v_hidden, err);
        vl->o_b = ik_copy_f32(m, b, VN("self_attn.out_proj.bias"), c->v_hidden, err);
        vl->fc1 = ik_copy_bf16(m, b, VN("mlp.c_fc.weight"), NULL, NULL, err);
        vl->fc2 = ik_copy_bf16(m, b, VN("mlp.c_proj.weight"), NULL, NULL, err);
        vl->fc1_b = ik_copy_f32(m, b, VN("mlp.c_fc.bias"), c->v_inter, err);
        vl->fc2_b = ik_copy_f32(m, b, VN("mlp.c_proj.bias"), c->v_hidden, err);
        vl->ls1 = ik_copy_f32(m, b, VN("ls_1.gamma"), c->v_hidden, err);
        vl->ls2 = ik_copy_f32(m, b, VN("ls_2.gamma"), c->v_hidden, err);
#undef VN
        if (err->status != CWICK_OK) return -1;
    }
    v->loaded = true;
    return 0;
}

/* ------------------------------------------------------------------ */
/* open / close                                                        */
/* ------------------------------------------------------------------ */

static void ik_load_layer(ik_model *m, ik_bump *b, int64_t L, cwick_error *err) {
    ik_layer *ly = &m->layers[L];
    char n[320];
    const int64_t H = m->c.hidden;
    const int64_t QD = m->c.n_heads * m->c.head_dim;
    const int64_t KVD = m->c.n_kv * m->c.head_dim;
    const int64_t RD = m->c.n_heads * m->c.d_rel;
    const int64_t K = m->c.sconv_k;
    bool swa = m->c.is_swa[L];
    bool loop = (L >= m->c.loop_start && L < m->c.loop_end);

#define NAME(fmt, ...) (snprintf(n, sizeof(n), "model.llm.layers.%lld." fmt, \
                                 (long long)L, ##__VA_ARGS__), n)
    /* Sliding layers share one input norm; full-attention layers carry three
     * separate norms for Q, K and V instead and have no `attn_norm` at all. */
    if (swa) ly->attn_norm = ik_copy_f32(m, b, NAME("attn_norm.weight"), H, err);
    ly->mlp_norm = ik_copy_f32(m, b, NAME("mlp_norm.weight"), H, err);
    if (!swa) {
        ly->q_in_norm = ik_copy_f32(m, b, NAME("q_input_norm.weight"), H, err);
        ly->k_in_norm = ik_copy_f32(m, b, NAME("k_input_norm.weight"), H, err);
        ly->v_in_norm = ik_copy_f32(m, b, NAME("v_input_norm.weight"), H, err);
    }
    ly->a.wq = ik_copy_bf16(m, b, NAME("attn.wq_du.weight"), NULL, NULL, err);
    ly->a.wk = ik_copy_bf16(m, b, NAME("attn.wk_dv.weight"), NULL, NULL, err);
    ly->a.wv = ik_copy_bf16(m, b, NAME("attn.wv_dv.weight"), NULL, NULL, err);
    ly->a.wr = ik_copy_bf16(m, b, NAME("attn.wr_du.weight"), NULL, NULL, err);
    ly->a.wo = ik_copy_bf16(m, b, NAME("attn.wo_ud.weight"), NULL, NULL, err);
    ly->a.gate = ik_copy_bf16(m, b, NAME("attn.gate_attn.weight"), NULL, NULL, err);
    ly->a.sink = ik_copy_bf16(m, b, NAME("attn.attn_sink_proj.weight"), NULL, NULL, err);
    ly->a.q_scale = ik_copy_f32(m, b, NAME("attn.q_l2_clip.scale"), QD, err);
    ly->a.rel_scale = ik_copy_f32(m, b, NAME("attn.rel_l2_clip.scale"), RD, err);
    ly->a.rel_proj = ik_copy_f32(m, b, NAME("attn.rel_logits_proj.proj"),
                                 m->c.d_rel * m->c.rel_extent, err);
    if (swa) {
        ly->a.k_conv = ik_copy_conv_transpose(m, b, NAME("attn.k_sconv.weight"), K, KVD, err);
        ly->a.v_conv = ik_copy_conv_transpose(m, b, NAME("attn.v_sconv.weight"), K, KVD, err);
    } else {
        ly->a.k_conv = ik_copy_taps(m, b, NAME("attn.k_sconv.static_w"), K, KVD, err);
        ly->a.v_conv = ik_copy_taps(m, b, NAME("attn.v_sconv.static_w"), K, KVD, err);
        ly->a.k_conv_proj = ik_copy_bf16(m, b, NAME("attn.k_sconv.weight_proj"), NULL, NULL, err);
        ly->a.v_conv_proj = ik_copy_bf16(m, b, NAME("attn.v_sconv.weight_proj"), NULL, NULL, err);
    }
    ly->pa_conv = ik_copy_depthwise(m, b, NAME("attn_sconv.weight"), H, K, err);
    ly->pm_conv = ik_copy_depthwise(m, b, NAME("mlp_sconv.weight"), H, K, err);
    ly->gate_w = ik_copy_bf16(m, b, NAME("mlp.gate.weight"), NULL, NULL, err);
    ly->gate_bias = ik_copy_f32(m, b, NAME("mlp.gate.bias"), m->c.n_experts, err);

    if (L > 0) {
        char qn[192];
        snprintf(qn, sizeof(qn), "model.llm.attn_res.attn_queries.%lld", (long long)(L - 1));
        const cwick_st_tensor *t = ik_st(m, qn, err);
        if (t) {
            int64_t nq = (t->rows * t->cols) / H;
            if (nq < 1) nq = 1;
            ly->attn_query = ik_copy_f32(m, b, qn, nq * H, err);
        }
    }
    {
        char qn[192];
        snprintf(qn, sizeof(qn), "model.llm.attn_res.mlp_queries.%lld", (long long)L);
        ly->mlp_query = ik_copy_f32(m, b, qn, H, err);
    }

    ly->is_loop = loop;
    if (loop) {
        static const char *kind[4] = {"attn", "k", "v", "mlp"};
        for (int c = 0; c < 4; c++) {
            char cn[288];
            snprintf(cn, sizeof(cn), "model.llm.layers.%lld.controllers.%s.down_weight",
                     (long long)L, kind[c]);
            ly->ctrl[c][0] = ik_copy_bf16(m, b, cn, NULL, NULL, err);
            snprintf(cn, sizeof(cn), "model.llm.layers.%lld.controllers.%s.gain_weight",
                     (long long)L, kind[c]);
            ly->ctrl[c][1] = ik_copy_bf16(m, b, cn, NULL, NULL, err);
            snprintf(cn, sizeof(cn), "model.llm.layers.%lld.controllers.%s.forget_weight",
                     (long long)L, kind[c]);
            ly->ctrl[c][2] = ik_copy_bf16(m, b, cn, NULL, NULL, err);
        }
        size_t at = ik_bump_off(b, (size_t)(4 * H) * sizeof(float), 64);
        if (b->base && b->off <= b->cap) ly->msg_query = (float *)(b->base + at);
    }

    {
        char en[288];
        snprintf(en, sizeof(en), "model.llm.layers.%lld.mlp.experts.w13_weight",
                 (long long)L);
        ly->w13 = ik_load_experts(m, b, en, 2 * m->c.inter, H, true,
                                  (ik_expert_format)m->expert_format[L], &ly->w13_packed, err);
        snprintf(en, sizeof(en), "model.llm.layers.%lld.mlp.experts.w2_weight",
                 (long long)L);
        ly->w2 = ik_load_experts(m, b, en, H, m->c.inter, false,
                                 (ik_expert_format)m->expert_format[L], &ly->w2_packed, err);
    }
#undef NAME
}

static void ik_load_msg_queries(ik_model *m, ik_bump *b, cwick_error *err) {
    const int64_t H = m->c.hidden;
    const int64_t core = m->c.loop_end - m->c.loop_start;
    /* The checkpoint stores 2 queries per core layer in each of two tables;
     * they are interleaved per layer into the four messages the layer takes
     * (q, k, v, mlp). */
    size_t at_q = ik_bump_off(b, (size_t)(core * 2 * H) * sizeof(float), 64);
    size_t at_kv = ik_bump_off(b, (size_t)(core * 2 * H) * sizeof(float), 64);
    if (!b->base) return;
    if (b->off > b->cap) return;
    float *qq = (float *)(b->base + at_q);
    float *kv = (float *)(b->base + at_kv);
    const cwick_st_tensor *tq = ik_st(m, "model.llm.middle_cycle_cross_pass_queries", err);
    const void *qdata = tq ? tq->data : NULL;
    cwick_st_dtype qdtype = tq ? tq->dtype : CWICK_ST_BF16;
    const cwick_st_tensor *tk = ik_st(m, "model.llm.middle_cycle_cross_pass_kv_queries", err);
    if (!qdata || !tk) return;
    /* The cross-pass tables are F32 in this checkpoint (they are listed as
     * fp32 compute tensors in hf_quant_config.json); older exports kept BF16. */
    if (qdtype == CWICK_ST_F32)
        memcpy(qq, qdata, (size_t)(core * 2 * H) * sizeof(float));
    else
        for (int64_t i = 0; i < core * 2 * H; i++)
            qq[i] = cwick_bf16_to_f32(((const uint16_t *)qdata)[i]);
    if (tk->dtype == CWICK_ST_F32)
        memcpy(kv, tk->data, (size_t)(core * 2 * H) * sizeof(float));
    else
        for (int64_t i = 0; i < core * 2 * H; i++)
            kv[i] = cwick_bf16_to_f32(((const uint16_t *)tk->data)[i]);
    for (int64_t i = 0; i < core; i++) {
        float *dst = m->layers[m->c.loop_start + i].msg_query;
        if (!dst) continue;
        memcpy(dst + 0 * H, qq + (2 * i) * H, (size_t)H * sizeof(float));
        memcpy(dst + 1 * H, kv + (2 * i) * H, (size_t)H * sizeof(float));
        memcpy(dst + 2 * H, kv + (2 * i + 1) * H, (size_t)H * sizeof(float));
        memcpy(dst + 3 * H, qq + (2 * i + 1) * H, (size_t)H * sizeof(float));
    }
}

/* Cache slot holding layer L's keys/values/convolution state on a given loop
 * pass (pass 0 = the first pass, which owns slots 0..n_layers-1). */
static int64_t ik_slot_of(const ik_model *m, int64_t L, int64_t pass) {
    if (pass <= 0) return m->slot[L];
    const int64_t nloop = m->c.loop_end - m->c.loop_start;
    return m->loop_slot0 + (pass - 1) * nloop + (L - m->c.loop_start);
}

static int64_t ik_slot_cap(const ik_model *m, int64_t L) {
    return m->c.is_swa[L] ? m->c.swa_window : m->cap;
}

int64_t ik_cache_bytes(const ik_model *m, int64_t cap) {
    const int64_t KVD = m->c.n_kv * m->c.head_dim;
    int64_t per_pos = 0;
    const int64_t passes = m->c.loop_passes > 0 ? m->c.loop_passes : 1;
    for (int64_t l = 0; l < m->c.n_layers; l++) {
        int64_t len = m->c.is_swa[l] ? m->c.swa_window : cap;
        per_pos += len * (l >= m->c.loop_start && l < m->c.loop_end ? passes : 1);
    }
    int64_t bytes = 2 * per_pos * KVD * (int64_t)sizeof(float);
    for (int64_t l = 0; l < m->c.n_layers; l++) {
        int64_t mult = l >= m->c.loop_start && l < m->c.loop_end ? passes : 1;
        bytes += mult * (2 * KVD + 2 * m->c.hidden) * (m->c.sconv_k - 1) *
                 (int64_t)sizeof(float);
    }
    return bytes;
}

static int ik_kv_alloc(ik_model *m, cwick_error *err) {
    const int64_t H = m->c.hidden, KVD = m->c.n_kv * m->c.head_dim;
    const int64_t nloop = m->c.loop_end - m->c.loop_start;
    m->loop_slot0 = m->c.n_layers;
    m->n_slots = m->c.n_layers + (m->c.loop_passes - 1) * nloop;
    if (m->n_slots > IK_MAX_SLOTS) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                        "loop passes need %lld cache slots, the limit is %d",
                        (long long)m->n_slots, IK_MAX_SLOTS);
        return -1;
    }
    for (int64_t l = 0; l < m->c.n_layers; l++) m->slot[l] = l;
    for (int64_t s = 0; s < m->n_slots; s++) {
        int64_t L = s < m->c.n_layers ? s : m->c.loop_start + (s - m->loop_slot0) % nloop;
        int64_t len = ik_slot_cap(m, L);
        m->kv_cap[s] = len;
        m->kv_k[s] = (float *)calloc((size_t)(len * KVD), sizeof(float));
        m->kv_v[s] = (float *)calloc((size_t)(len * KVD), sizeof(float));
        if (!m->kv_k[s] || !m->kv_v[s]) {
            cwick_error_set(err, CWICK_ERR_MEMORY, "KV cache allocation failed");
            return -1;
        }
        const int64_t chans[4] = {KVD, KVD, H, H};
        for (int c = 0; c < 4; c++) {
            m->conv_ch[s][c] = chans[c];
            m->conv_hist[s][c] = (float *)calloc(
                (size_t)((m->c.sconv_k - 1) * chans[c]), sizeof(float));
            if (!m->conv_hist[s][c]) {
                cwick_error_set(err, CWICK_ERR_MEMORY, "conv state allocation failed");
                return -1;
            }
        }
        if (ik_gpu_enabled()) {
            size_t kb = (size_t)(len * KVD) * sizeof(float);
            h3v_gpu_adopt(m->kv_k[s], kb);
            h3v_gpu_adopt(m->kv_v[s], kb);
        }
    }
    return 0;
}

int ik_model_open(ik_model *m, const char *root, int64_t cap, cwick_error *err) {
    memset(m, 0, sizeof(*m));
    ik_build_tables();
    if (ik_cfg_load(root, &m->c, err) != CWICK_OK) return -1;
    m->packed_experts = ik_packed_experts_enabled();
    const char *dual = getenv("CWICK_IK_DUAL_EXPERTS");
    m->dual_experts = m->packed_experts && dual && strcmp(dual, "0") != 0;
    if (m->packed_experts && ik_quant_config_load(m, root, err) != 0) return -1;
    if (cwick_safetensors_open(root, &m->saf, err) != CWICK_OK) return -1;
    m->layers = (ik_layer *)calloc((size_t)m->c.n_layers, sizeof(ik_layer));
    if (!m->layers) {
        cwick_error_set(err, CWICK_ERR_MEMORY, "layer table allocation failed");
        return -1;
    }

    /* ---- pass 1: compute the arena size, touch nothing ---- */
    ik_bump plan;
    memset(&plan, 0, sizeof(plan));
    ik_load_msg_queries(m, &plan, err);
    ik_copy_bf16(m, &plan, "model.llm.embed.weight", NULL, NULL, err);
    ik_copy_bf16(m, &plan, "model.llm.unembed.weight", NULL, NULL, err);
    ik_copy_f32(m, &plan, "model.llm.embed_norm.weight", m->c.hidden, err);
    ik_copy_f32(m, &plan, "model.llm.norm.weight", m->c.hidden, err);
    ik_copy_f32(m, &plan, "model.llm.attn_res.output_query", m->c.hidden, err);
    for (int64_t l = 0; l < m->c.n_layers; l++) ik_load_layer(m, &plan, l, err);
    if (ik_vision_load(m, &plan, err) != 0) return -1;
    size_t total = plan.off;
    if (err->status != CWICK_OK) return -1;

    CWICK_INFO("ik", "reserving %.2f GiB for the resident checkpoint", cwick_gib(total));
    void *arena = mmap(NULL, total, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (arena == MAP_FAILED) {
        cwick_error_set(err, CWICK_ERR_MEMORY,
                        "cannot reserve %.2f GiB for the Inkling-2-Mini arena",
                        cwick_gib(total));
        return -1;
    }
    m->arena = arena;
    m->arena_bytes = total;

    ik_bump fill;
    fill.base = (uint8_t *)arena;
    fill.off = 0;
    fill.cap = total;

    m->embed = ik_copy_bf16(m, &fill, "model.llm.embed.weight", NULL, NULL, err);
    m->unembed = ik_copy_bf16(m, &fill, "model.llm.unembed.weight", NULL, NULL, err);
    m->embed_norm = ik_copy_f32(m, &fill, "model.llm.embed_norm.weight", m->c.hidden, err);
    m->out_norm = ik_copy_f32(m, &fill, "model.llm.norm.weight", m->c.hidden, err);
    m->final_query = ik_copy_f32(m, &fill, "model.llm.attn_res.output_query",
                                 m->c.hidden, err);
    for (int64_t l = 0; l < m->c.n_layers; l++) {
        ik_load_layer(m, &fill, l, err);
        if (err->status != CWICK_OK) return -1;
    }
    ik_load_msg_queries(m, &fill, err);
    m->vis = NULL;
    if (ik_vision_load(m, &fill, err) != 0) return -1;
    if (fill.off != total)
        CWICK_INFO("ikwarn", "arena planner and loader disagree (%zu vs %zu bytes)",
                 total, fill.off);
    if (ik_gpu_enabled()) {
        h3v_gpu_adopt(arena, total);
        cwick_metal_adopt(arena, total);
    }

    m->logits = (float *)calloc((size_t)m->c.vocab, sizeof(float));
    if (!m->logits) {
        cwick_error_set(err, CWICK_ERR_MEMORY, "logit buffer allocation failed");
        return -1;
    }

    /* Everything the arena needs has been copied out, so drop the 19.5 GiB
     * source mapping before wiring: keeping both resident would put the
     * dequantized copy and its own source in competition for the same RAM. */
    cwick_safetensors_close(&m->saf);

    m->cap = cap > 0 ? cap : 8192;
    if (ik_kv_alloc(m, err) != 0) return -1;
    if (mlock(arena, total) != 0) {
        cwick_error_set(err, CWICK_ERR_MEMORY,
                        "cannot wire %.2f GiB of Inkling-2-Mini weights",
                        cwick_gib(total));
        return -1;
    }
    m->arena_wired = true;
    int64_t cache_bytes = ik_cache_bytes(m, m->cap);
    if (mlock(m->logits, (size_t)m->c.vocab * sizeof(float)) != 0) {
        cwick_error_set(err, CWICK_ERR_MEMORY, "cannot wire logit buffer");
        return -1;
    }
    for (int64_t s = 0; s < m->n_slots; s++) {
        const int64_t KVD = m->c.n_kv * m->c.head_dim;
        size_t kb = (size_t)(m->kv_cap[s] * KVD) * sizeof(float);
        if (mlock(m->kv_k[s], kb) != 0 || mlock(m->kv_v[s], kb) != 0) {
            cwick_error_set(err, CWICK_ERR_MEMORY, "cannot wire vision/text KV cache");
            return -1;
        }
        for (int c = 0; c < 4; c++) {
            size_t cb = (size_t)((m->c.sconv_k - 1) * m->conv_ch[s][c]) * sizeof(float);
            if (cb && mlock(m->conv_hist[s][c], cb) != 0) {
                cwick_error_set(err, CWICK_ERR_MEMORY, "cannot wire convolution state");
                return -1;
            }
        }
    }
    CWICK_INFO("ik", "wired %.2f GiB weights + %.2f GiB inference state",
               cwick_gib(total), cwick_gib((double)cache_bytes));
    /* The shared Metal device builds its pipelines on first use: ~20 ms with a
     * warm driver shader cache, the full MSL compile on a cold one. Doing it
     * here keeps that off the first request's time-to-first-token. It is not a
     * large win and it is not the ~1 s a short first request shows -- measured
     * 14- and 18-token prefills cost 0.83-0.96 s whether this call sits inside
     * the request or ahead of it, so that second is prefill, not pipeline
     * setup. */
    if (ik_gpu_enabled()) (void)h3v_gpu_on();
    /* This backend has no streaming plan, so nothing else publishes these
     * gauges: without them /v1/metrics reports a resident 18 GiB model as 0. */
    uint64_t host = cwick_host_memory_bytes();
    if (host)
        cwick_metrics_set_memory_ceiling(
            (uint64_t)((double)host * CWICK_MEM_DEFAULT_FRACTION));
    cwick_metrics_set_model_residency((uint64_t)total, (uint64_t)total,
                                      (uint64_t)(total + cache_bytes), 1);
    m->pos = 0;

    int64_t nswa = 0, nnv = 0, nmx = 0;
    for (int64_t i = 0; i < m->c.n_layers; i++) {
        nswa += m->c.is_swa[i] ? 1 : 0;
        nnv += m->expert_format[i] == IK_EXPERT_NVFP4;
        nmx += m->expert_format[i] == IK_EXPERT_MXFP8;
    }
    CWICK_INFO("ik", "Inkling-2-Mini ready: %lld layers (%lld sliding), %lld experts "
             "top-%lld, loop [%lld,%lld) x%lld, %lld positions, KV %.2f GiB, "
             "linear backend=%s experts=%s",
             (long long)m->c.n_layers, (long long)nswa,
             (long long)m->c.n_experts, (long long)m->c.top_k,
             (long long)m->c.loop_start, (long long)m->c.loop_end,
             (long long)m->c.loop_passes, (long long)m->cap,
             cwick_gib((double)ik_cache_bytes(m, m->cap)),
             ik_gpu_enabled() ? "metal" : "cpu",
             m->packed_experts ? (m->dual_experts ? "packed+bf16" : "packed") : "bf16");
    if (m->packed_experts)
        CWICK_INFO("ik", "packed experts: %lld NVFP4 + %lld MXFP8 layers%s",
                   (long long)nnv, (long long)nmx,
                   m->dual_experts ? " (BF16 retained for image requests)" : "");
    return 0;
}

void ik_model_close(ik_model *m) {
    ik_vision_close(m);
    if (m->arena) {
        if (m->arena_wired) munlock(m->arena, m->arena_bytes);
        munmap(m->arena, m->arena_bytes);
        m->arena = NULL;
    }
    cwick_safetensors_close(&m->saf);
    free(m->layers);
    m->layers = NULL;
    for (int64_t s = 0; s < m->n_slots; s++) {
        free(m->kv_k[s]); free(m->kv_v[s]);
        m->kv_k[s] = NULL; m->kv_v[s] = NULL;
        for (int c = 0; c < 4; c++) { free(m->conv_hist[s][c]); m->conv_hist[s][c] = NULL; }
    }
    free(m->logits);
    m->logits = NULL;
    ik_scratch_free();
}

void ik_model_reset(ik_model *m) {
    const int64_t KVD = m->c.n_kv * m->c.head_dim;
    for (int64_t s = 0; s < m->n_slots; s++) {
        memset(m->kv_k[s], 0, (size_t)(m->kv_cap[s] * KVD) * sizeof(float));
        memset(m->kv_v[s], 0, (size_t)(m->kv_cap[s] * KVD) * sizeof(float));
        for (int c = 0; c < 4; c++)
            memset(m->conv_hist[s][c], 0,
                   (size_t)((m->c.sconv_k - 1) * m->conv_ch[s][c]) * sizeof(float));
        m->kv_len[s] = 0;
        m->kv_head[s] = 0;
    }
    m->pos = 0;
}

/* ------------------------------------------------------------------ */
/* forward pieces                                                      */
/* ------------------------------------------------------------------ */

/* Attention over the bank of block states plus `cur`, with `nq` learned
 * queries. Every entry is RMS-normalised first; the mixture is a softmax with
 * 1/sqrt(H) scaling. */
static void ik_mix(const ik_cfg *c, const float *bank, int64_t nbank,
                   const float *cur, const float *query, int64_t nq,
                   float *norm_buf, float *score_buf, float *out) {
    const bool prof = ik_prof_on();
    const double t0 = prof ? ik_prof_now() : 0.0;
    const int64_t H = c->hidden;
    const float scale = 1.0f / sqrtf((float)H);
    const int64_t total = nbank + 1;
    for (int64_t b = 0; b < nbank; b++)
        ik_rms_unweighted(norm_buf + b * H, bank + b * H, H, 1e-6f);
    ik_rms_unweighted(norm_buf + nbank * H, cur, H, 1e-6f);
    for (int64_t q = 0; q < nq; q++) {
        const float *sv = query + q * H;
        float mx = -INFINITY;
        for (int64_t b = 0; b < total; b++) {
            const float *x = norm_buf + b * H;
            double acc = 0.0;
            for (int64_t i = 0; i < H; i++) acc += (double)x[i] * (double)sv[i];
            score_buf[b] = (float)acc * scale;
            if (score_buf[b] > mx) mx = score_buf[b];
        }
        double sum = 0.0;
        for (int64_t b = 0; b < total; b++) {
            score_buf[b] = expf(score_buf[b] - mx);
            sum += score_buf[b];
        }
        float inv = (float)(1.0 / sum);
        float *dst = out + q * H;
        for (int64_t i = 0; i < H; i++) dst[i] = 0.0f;
        for (int64_t b = 0; b < total; b++) {
            float w = score_buf[b] * inv;
            const float *x = norm_buf + b * H;
            for (int64_t i = 0; i < H; i++) dst[i] += w * x[i];
        }
    }
    if (prof) ik_prof_add(&g_prof.mix, t0);
}

/* Message gate: (1 - forget) * cur + gain * message, both gated on the pair. */
void ik_control(const ik_cfg *c, const ik_layer *ly, int ctrl,
                const float *cur, const float *msg, float *out,
                float *cat, float *gate, float *gf, float *gm) {
    const bool prof = ik_prof_on();
    const double t0 = prof ? ik_prof_now() : 0.0;
    const int64_t H = c->hidden, R = c->gate_rank;
    ik_rms_unweighted(cat, cur, H, 1e-6f);
    ik_rms_unweighted(cat + H, msg, H, 1e-6f);
    ik_round_bf16(cat, 2 * H);
    cwick_gemv_bf16(gate, 2 * H, R, cat, ly->ctrl[ctrl][0]);
    ik_round_bf16(gate, R);
    cwick_gemv_bf16(gf, R, H, gate, ly->ctrl[ctrl][2]);
    cwick_gemv_bf16(gm, R, H, gate, ly->ctrl[ctrl][1]);
    for (int64_t i = 0; i < H; i++)
        out[i] = (1.0f - tanhf(gf[i])) * cur[i] + tanhf(gm[i]) * msg[i];
    if (prof) ik_prof_add(&g_prof.control, t0);
}

/* Causal short convolution over a chunk. `taps` is [k][ch] for the dynamic
 * variants (tap 0 = current token) or [ch][k] for the depthwise ones (tap 0 =
 * k-1 steps back). The output is (conv + input) / denom, in place. */
static void ik_sconv(const ik_cfg *c, float *x, int64_t n, int64_t ch,
                     const float *taps, const float *dyn, int64_t heads,
                     float *hist, bool depthwise, float denom) {
    const bool prof = ik_prof_on();
    const double t0 = prof ? ik_prof_now() : 0.0;
    const int64_t K = c->sconv_k;
    float *in = (float *)cwick_xmalloc((size_t)(n * ch) * sizeof(float));
    memcpy(in, x, (size_t)(n * ch) * sizeof(float));
    for (int64_t t = 0; t < n; t++) {
        const float *src = in + t * ch;
        float *v = x + t * ch;
        for (int64_t i = 0; i < ch; i++) {
            float acc = 0.0f;
            if (depthwise) {
                /* taps[ch][j]: tap j pairs with the input K-1-j steps back, and
                 * hist[0] is the oldest of the K-1 retained values */
                for (int64_t j = 0; j < K; j++) {
                    float s = (j < K - 1) ? hist[j * ch + i] : src[i];
                    acc += taps[i * K + j] * s;
                }
            } else {
                /* taps[lag][ch]: tap 0 is the current token */
                int64_t h = i / (ch / heads);
                const float *dt = dyn ? dyn + (size_t)t * (size_t)(heads * K) : NULL;
                for (int64_t lag = 0; lag < K; lag++) {
                    float s = (lag == 0) ? src[i]
                                         : hist[(K - 1 - lag) * ch + i];
                    float w = taps[lag * ch + i];
                    if (dt) w += dt[h * K + lag];
                    acc += w * s;
                }
            }
            v[i] = cwick_bf16_to_f32(cwick_f32_to_bf16((acc + src[i]) * denom));
        }
        const int64_t keep = K - 1;
        for (int64_t d = 0; d + 1 < keep; d++)
            memcpy(hist + d * ch, hist + (d + 1) * ch, (size_t)ch * sizeof(float));
        if (keep > 0) memcpy(hist + (keep - 1) * ch, src, (size_t)ch * sizeof(float));
    }
    free(in);
    if (prof) ik_prof_add(&g_prof.conv, t0);
}

typedef struct {
    const ik_model *m;
    const ik_layer *ly;
    int64_t L, slot, n;
    const float *q, *rr, *sink;
    float *ao, *scores;
    int64_t score_stride;
} ik_attn_arg;

/* One token: softmax over the visible history with the relative bias and the
 * sink logit for each head, then the weighted sum of values. Slices are by
 * token so the logit scratch is private to one worker; the earlier (token,
 * head) split made two workers share the same token's logit row. */
static void ik_attn_range(void *argp, int64_t i0, int64_t i1) {
    ik_attn_arg *a = (ik_attn_arg *)argp;
    const ik_cfg *c = &a->m->c;
    const int64_t NH = c->n_heads, D = c->head_dim, KVH = c->n_kv;
    const int64_t G = NH / KVH;
    const bool swa = c->is_swa[a->L];
    const int64_t win = c->swa_window;
    const float scale = 1.0f / sqrtf((float)D);
    const float *kk = a->m->kv_k[a->slot];
    const float *vv = a->m->kv_v[a->slot];
    const int64_t cap = a->m->kv_cap[a->slot];
    for (int64_t t = i0; t < i1; t++) {
        const int64_t p = a->m->pos + t;
        /* Sliding windows wrap in the cache; full layers never write past cap. */
        int64_t j1 = (!swa && p >= cap) ? cap - 1 : p;
        int64_t j0 = swa && p + 1 > win ? p + 1 - win : 0;
        if (j0 > j1) j0 = j1;
        const int64_t nk = j1 - j0 + 1;
        float *scores = a->scores + t * a->score_stride;
        for (int64_t h = 0; h < NH; h++) {
            const float *qh = a->q + (t * NH + h) * D;
            const float *rh = a->rr + (t * NH + h) * c->d_rel;
            float sink = a->sink[t * NH + h];
            int64_t kvh = h / G;
            float mx = 0.0f;                   /* the sink's own logit is 0 */
            double sum = 0.0;
            for (int64_t j = j0; j <= j1; j++) {
                int64_t slot = swa ? (j % cap) : j;
                const float *kj = kk + slot * (KVH * D) + kvh * D;
                double acc = 0.0;
                for (int64_t d = 0; d < D; d++) acc += (double)qh[d] * (double)kj[d];
                float logit = (float)acc * scale;
                int64_t dist = p - j;
                if (dist < c->rel_extent) {
                    const float *pr = a->ly->a.rel_proj;
                    double b = 0.0;
                    for (int64_t r = 0; r < c->d_rel; r++)
                        b += (double)rh[r] * (double)pr[r * c->rel_extent + dist];
                    logit += cwick_bf16_to_f32(cwick_f32_to_bf16((float)b));
                }
                logit -= sink;
                scores[j - j0] = logit;
                if (logit > mx) mx = logit;
            }
            sum = exp((double)(0.0f - mx));    /* sink mass */
            for (int64_t k = 0; k < nk; k++) sum += exp((double)(scores[k] - mx));
            float inv = (float)(1.0 / sum);
            float *dst = a->ao + (t * NH + h) * D;
            for (int64_t d = 0; d < D; d++) dst[d] = 0.0f;
            for (int64_t j = j0; j <= j1; j++) {
                float w = (float)exp((double)(scores[j - j0] - mx)) * inv;
                int64_t slot = swa ? (j % cap) : j;
                const float *vj = vv + slot * (KVH * D) + kvh * D;
                for (int64_t d = 0; d < D; d++) dst[d] += w * vj[d];
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* layer                                                               */
/* ------------------------------------------------------------------ */

static void ik_layer_forward(ik_model *m, int64_t L, ik_scratch *s, int64_t n,
                             bool have_msgs, int64_t pass);

/* Apply the routing weight before the activation's BF16/W2 boundary. */
void ik_expert_activation(float *out, const float *gate, const float *up,
                          int64_t n, float gamma, float gate_cap, float up_cap) {
    cwick_situ(out, gate, up, n, gate_cap, up_cap);
    for (int64_t i = 0; i < n; i++)
        out[i] = cwick_bf16_to_f32(cwick_f32_to_bf16(out[i] * gamma));
}

/* Run the MoE for one layer over the chunk: route, then one GEMM per expert. */
static void ik_moe_layer(ik_model *m, const ik_layer *ly, ik_scratch *s, int64_t n) {
    const bool prof = ik_prof_on();
    const double t0 = prof ? ik_prof_now() : 0.0;
    const int64_t H = m->c.hidden, I = m->c.inter, NE = m->c.n_experts;
    const int64_t K = m->c.top_k;
    float *x = s->xn;
    float *out = s->mo;
    memset(out, 0, (size_t)(n * H) * sizeof(float));
    const double route_t0 = prof ? ik_prof_now() : 0.0;

    if (n > 1) {
        ik_linear_arg gate = {ly->gate_w, x, s->scores, H, NE};
        ik_linear_group(m, n, &gate, 1);
    }
    for (int64_t t = 0; t < n; t++) {
        float *sc = s->scores + t * NE;
        if (n == 1) {
            ik_linear_arg gate = {ly->gate_w, x + t * H, sc, H, NE};
            ik_linear_group(m, 1, &gate, 1);
        }
        for (int64_t e = 0; e < NE; e++) sc[e] = 1.0f / (1.0f + expf(-sc[e]));
        int32_t *sel = s->sel + t * K;
        for (int64_t k = 0; k < K; k++) sel[k] = -1;
        for (int64_t k = 0; k < K; k++) {
            float best = -INFINITY;
            int32_t bi = -1;
            for (int64_t e = 0; e < NE; e++) {
                bool taken = false;
                for (int64_t j = 0; j < k; j++)
                    if (sel[j] == e) { taken = true; break; }
                if (taken) continue;
                float v = sc[e] + ly->gate_bias[e];
                if (v > best) { best = v; bi = (int32_t)e; }
            }
            if (bi < 0) bi = 0;   /* never let a NaN ranking index the bins */
            sel[k] = bi;
        }
        float sum = 0.0f;
        for (int64_t k = 0; k < K; k++) sum += sc[sel[k]];
        if (!(sum > 0.0f)) sum = 1.0f;
        for (int64_t k = 0; k < K; k++)
            s->wts[t * K + k] = sc[sel[k]] / sum * m->c.route_scale;
    }

    if (prof) ik_prof_add(&g_prof.route, route_t0);

    int32_t *list = (int32_t *)malloc((size_t)(n * K) * sizeof(int32_t));
    float *wlist = (float *)malloc((size_t)(n * K) * sizeof(float));
    int64_t *cursor = (int64_t *)calloc((size_t)(NE + 1), sizeof(int64_t));
    int64_t *first = (int64_t *)calloc((size_t)(NE + 1), sizeof(int64_t));
    if (!list || !wlist || !cursor || !first) {
        free(list); free(wlist); free(cursor); free(first);
        abort();
    }
    for (int64_t t = 0; t < n; t++)
        for (int64_t k = 0; k < K; k++) first[s->sel[t * K + k] + 1]++;
    for (int64_t e = 0; e < NE; e++) first[e + 1] += first[e];
    memcpy(cursor, first, (size_t)(NE + 1) * sizeof(int64_t));
    for (int64_t t = 0; t < n; t++)
        for (int64_t k = 0; k < K; k++) {
            int32_t e = s->sel[t * K + k];
            int64_t at = cursor[e]++;
            list[at] = (int32_t)t;
            wlist[at] = s->wts[t * K + k];
        }

    const bool use_packed = m->packed_experts &&
        (m->img_tokens == 0 || n == 1) && ik_gpu_enabled();
    const uint16_t *w13 = use_packed ? NULL : (const uint16_t *)ly->w13;
    const uint16_t *w2 = use_packed ? NULL : (const uint16_t *)ly->w2;
    float *xs = (float *)malloc((size_t)(n * K) * (size_t)H * sizeof(float));
    float *gate_up = (float *)malloc((size_t)(n * K) * (size_t)(2 * I) * sizeof(float));
    float *act = (float *)malloc((size_t)(n * K) * (size_t)I * sizeof(float));
    float *down = (float *)malloc((size_t)(n * K) * (size_t)H * sizeof(float));
    if (!xs || !gate_up || !act || !down) {
        free(xs); free(gate_up); free(act); free(down);
        free(list); free(wlist); free(cursor); free(first);
        abort();
    }
    ik_expert_item up_items[128], down_items[128];
    ik_expert_pair *pairs = (ik_expert_pair *)malloc((size_t)(n * K) * sizeof(*pairs));
    if (!pairs) abort();
    int64_t active = 0;
    for (int64_t e = 0; e < NE; e++) {
        int64_t lo = first[e], hi = first[e + 1];
        if (lo >= hi) continue;
        int64_t cnt = hi - lo;
        for (int64_t i = 0; i < cnt; i++) {
            memcpy(xs + (lo + i) * H, x + (int64_t)list[lo + i] * H,
                   (size_t)H * sizeof(float));
            pairs[lo + i] = (ik_expert_pair){&ly->w13_packed, &ly->w2_packed, e,
                                              x + (int64_t)list[lo + i] * H,
                                              down + (lo + i) * H, wlist[lo + i]};
        }
        const uint16_t *we = w13 ? w13 + (size_t)e * (size_t)(2 * I) * (size_t)H : NULL;
        const uint16_t *wd = w2 ? w2 + (size_t)e * (size_t)H * (size_t)I : NULL;
        up_items[active] = (ik_expert_item){
            .linear = {we, xs + lo * H, gate_up + lo * (2 * I), cnt, H, 2 * I},
            .pack = &ly->w13_packed, .expert = e};
        down_items[active] = (ik_expert_item){
            .linear = {wd, act + lo * I, down + lo * H, cnt, I, H},
            .pack = &ly->w2_packed, .expert = e};
        active++;
    }
    double stage_t0 = prof ? ik_prof_now() : 0.0;
    bool paired = false;
    const char *fused = getenv("CWICK_IK_FUSED_EXPERTS");
    if (use_packed && (!fused || strcmp(fused, "0") != 0)) {
        paired = ik_packed_pair_batch(m, pairs, n * K);
        if (prof) ik_prof_add(&g_prof.pair, stage_t0);
    }
    if (!paired) {
        stage_t0 = prof ? ik_prof_now() : 0.0;
        ik_linear_expert_items(m, up_items, I, active, use_packed);
        if (prof) ik_prof_add(&g_prof.up, stage_t0);
        stage_t0 = prof ? ik_prof_now() : 0.0;
        for (int64_t i = 0; i < n * K; i++)
            ik_expert_activation(act + i * I, gate_up + i * (2 * I),
                                 gate_up + i * (2 * I) + I, I, wlist[i],
                                 m->c.gate_softcap, m->c.up_softcap);
        if (prof) ik_prof_add(&g_prof.act, stage_t0);
        stage_t0 = prof ? ik_prof_now() : 0.0;
        ik_linear_expert_items(m, down_items, 0, active, use_packed);
        if (prof) ik_prof_add(&g_prof.down, stage_t0);
    }
    stage_t0 = prof ? ik_prof_now() : 0.0;
    for (int64_t i = 0; i < n * K; i++) {
        int64_t t = list[i];
        float *dst = out + t * H;
        const float *src = down + i * H;
        for (int64_t j = 0; j < H; j++) dst[j] += src[j];
    }
    if (prof) ik_prof_add(&g_prof.combine, stage_t0);
    ik_round_bf16(out, n * H);
    free(xs); free(gate_up); free(act); free(down); free(pairs);
    free(list); free(wlist); free(cursor); free(first);
    if (prof) ik_prof_add(&g_prof.moe, t0);
}

static void ik_layer_forward(ik_model *m, int64_t L, ik_scratch *s, int64_t n,
                             bool have_msgs, int64_t pass) {
    ik_layer *ly = &m->layers[L];
    const ik_cfg *c = &m->c;
    const int64_t slot = ik_slot_of(m, L, pass);
    const int64_t H = c->hidden;
    const int64_t NH = c->n_heads, KVH = c->n_kv, D = c->head_dim;
    const int64_t QD = NH * D, KVD = KVH * D, RD = NH * c->d_rel;
    const bool swa = c->is_swa[L];
    const int64_t K = c->sconv_k;
    const bool boundary = (L % c->res_block) == 0;
    const int64_t B = IK_MAX_BLOCKS;
    if (ik_prof_on()) g_prof.layers++;

    /* ---- 1. attention inputs ---- */
    int64_t nq = swa ? 1 : 3;
    if (!ly->attn_query) nq = 0;
    for (int64_t t = 0; t < n; t++) {
        const float *cur = s->x + t * H;
        float *qin = s->ain + t * H, *kin = s->bin + t * H, *vin = s->cin + t * H;
        if (nq > 0) {
            ik_mix(c, s->blocks + (size_t)t * B * H, s->nblocks[t], cur,
                   ly->attn_query, nq, s->normbuf, s->scores, s->mixout);
            memcpy(qin, s->mixout, (size_t)H * sizeof(float));
            memcpy(kin, s->mixout + (nq > 1 ? H : 0), (size_t)H * sizeof(float));
            memcpy(vin, s->mixout + (nq > 2 ? 2 * H : 0), (size_t)H * sizeof(float));
        } else {
            memcpy(qin, cur, (size_t)H * sizeof(float));
            memcpy(kin, cur, (size_t)H * sizeof(float));
            memcpy(vin, cur, (size_t)H * sizeof(float));
        }
    }

    if (have_msgs && ly->is_loop) {
        float *cat = (float *)cwick_xmalloc((size_t)(2 * H) * sizeof(float));
        float *gate = (float *)cwick_xmalloc((size_t)c->gate_rank * sizeof(float));
        float *gf = (float *)cwick_xmalloc((size_t)H * sizeof(float));
        float *gm = (float *)cwick_xmalloc((size_t)H * sizeof(float));
        for (int64_t t = 0; t < n; t++) {
            const float *msg = s->msgs + (size_t)t * 4 * H;
            ik_control(c, ly, 0, s->ain + t * H, msg + 0 * H, s->ain + t * H,
                       cat, gate, gf, gm);
            ik_control(c, ly, 1, s->bin + t * H, msg + 1 * H, s->bin + t * H,
                       cat, gate, gf, gm);
            ik_control(c, ly, 2, s->cin + t * H, msg + 2 * H, s->cin + t * H,
                       cat, gate, gf, gm);
        }
        free(cat); free(gate); free(gf); free(gm);
    }

    /* ---- 2. the three input norms ---- */
    {
        const float *wn = swa ? ly->attn_norm : ly->q_in_norm;
        const float *wk = swa ? ly->attn_norm : ly->k_in_norm;
        const float *wv = swa ? ly->attn_norm : ly->v_in_norm;
        if (swa) {
            /* one shared normalisation, unless the message gates made the three
             * inputs differ */
            if (have_msgs && ly->is_loop) {
                for (int64_t t = 0; t < n; t++) {
                    ik_rms_norm_bf16(s->ain + t * H, s->ain + t * H, H, wn, c->rms_eps);
                    ik_rms_norm_bf16(s->bin + t * H, s->bin + t * H, H, wn, c->rms_eps);
                    ik_rms_norm_bf16(s->cin + t * H, s->cin + t * H, H, wn, c->rms_eps);
                }
            } else {
                for (int64_t t = 0; t < n; t++) {
                    ik_rms_norm_bf16(s->ain + t * H, s->ain + t * H, H, wn, c->rms_eps);
                    memcpy(s->bin + t * H, s->ain + t * H, (size_t)H * sizeof(float));
                    memcpy(s->cin + t * H, s->ain + t * H, (size_t)H * sizeof(float));
                }
            }
        } else {
            for (int64_t t = 0; t < n; t++) {
                ik_rms_norm_bf16(s->ain + t * H, s->ain + t * H, H, wn, c->rms_eps);
                ik_rms_norm_bf16(s->bin + t * H, s->bin + t * H, H, wk, c->rms_eps);
                ik_rms_norm_bf16(s->cin + t * H, s->cin + t * H, H, wv, c->rms_eps);
            }
        }
    }

    /* ---- 3. projections, the two short convolutions, QK/rel clipping ---- */
    ik_linear_arg qkv[6] = {
        {ly->a.wq, s->ain, s->q, H, QD},
        {ly->a.wk, s->bin, s->kk, H, KVD},
        {ly->a.wv, s->cin, s->vv, H, KVD},
        {ly->a.wr, s->ain, s->rr, H, RD},
        {ly->a.gate, s->ain, s->gg, H, QD},
        {ly->a.sink, s->ain, s->tt, H, NH},
    };
    ik_linear_group(m, n, qkv, 6);
    ik_round_bf16(s->q, n * QD);
    ik_round_bf16(s->kk, n * KVD);
    ik_round_bf16(s->vv, n * KVD);
    ik_round_bf16(s->rr, n * RD);
    ik_round_bf16(s->gg, n * QD);
    ik_round_bf16(s->tt, n * NH);
    memcpy(s->mo, s->tt, (size_t)(n * NH) * sizeof(float));
    if (!swa) {
        for (int64_t chunk = 0; chunk < n; chunk += 32) {
            int64_t take = n - chunk < 32 ? n - chunk : 32;
            ik_linear_arg dyn[2] = {
                {ly->a.k_conv_proj, s->bin + chunk * H,
                 s->dynk + chunk * KVH * K, H, KVH * K},
                {ly->a.v_conv_proj, s->cin + chunk * H,
                 s->dynv + chunk * KVH * K, H, KVH * K},
            };
            ik_linear_group(m, take, dyn, 2);
            ik_round_bf16(s->dynk + chunk * KVH * K, take * KVH * K);
            ik_round_bf16(s->dynv + chunk * KVH * K, take * KVH * K);
        }
    }
    ik_sconv(c, s->kk, n, KVD, ly->a.k_conv, swa ? NULL : s->dynk, KVH,
             m->conv_hist[slot][0], false, swa ? 1.0f / sqrtf((float)(K + 1))
                                            : 1.0f / sqrtf((float)(2 * K + 1)));
    ik_sconv(c, s->vv, n, KVD, ly->a.v_conv, swa ? NULL : s->dynv, KVH,
             m->conv_hist[slot][1], false, swa ? 1.0f / sqrtf((float)(K + 1))
                                            : 1.0f / sqrtf((float)(2 * K + 1)));

    for (int64_t t = 0; t < n; t++) {
        float *q = s->q + t * QD;
        float *k = s->kk + t * KVD;
        float *v = s->vv + t * KVD;
        float *r = s->rr + t * RD;
        ik_l2_clip(q, NH, D, ly->a.q_scale, c->qk_clip);
        ik_l2_clip(k, KVH, D, NULL, c->qk_clip);
        for (int64_t h = 0; h < KVH; h++) {
            ik_rms_unweighted(v + h * D, v + h * D, D, c->rms_eps);
            ik_round_bf16(v + h * D, D);
        }
        ik_l2_clip(r, NH, c->d_rel, ly->a.rel_scale, c->rel_clip);
        if (!swa) {
            int64_t p = m->pos + t;
            float tau = 1.0f + c->log_alpha *
                logf(fmaxf(1.0f, (float)(p + 1) / c->log_floor));
            for (int64_t i = 0; i < QD; i++) q[i] *= tau;
            for (int64_t i = 0; i < RD; i++) r[i] *= tau;
            ik_round_bf16(q, QD);
            ik_round_bf16(r, RD);
        }
    }

    /* ---- 4. write the KV cache ---- */
    {
        const int64_t cap = m->kv_cap[slot];
        for (int64_t t = 0; t < n; t++) {
            int64_t p = m->pos + t;
            int64_t at = swa ? (p % cap) : p;
            if (!swa && p >= cap) continue;
            memcpy(m->kv_k[slot] + at * KVD, s->kk + t * KVD,
                   (size_t)KVD * sizeof(float));
            memcpy(m->kv_v[slot] + at * KVD, s->vv + t * KVD,
                   (size_t)KVD * sizeof(float));
        }
    }

    /* ---- 5. attention ---- */
    {
        float *sink = s->mo;
        ik_attn_arg a;
        a.m = m;
        a.ly = ly;
        a.L = L;
        a.slot = slot;
        a.n = n;
        a.q = s->q;
        a.rr = s->rr;
        a.sink = sink;
        a.ao = s->ao;
        a.scores = s->ascores;
        a.score_stride = s->score_keys;
        cwick_pool_init();
        const double attn_t0 = ik_prof_on() ? ik_prof_now() : 0.0;
        const int attn_mode = ik_gpu_attn_mode();
        const bool use_gpu_attn = attn_mode > 0;
        bool gpu_attn = false;
        if (use_gpu_attn && n <= 64 && ik_gpu_enabled() && h3v_gpu_on()) {
            if (n == 1) {
                gpu_attn = h3v_gpu_inkling_decode_attn(
                    s->q, m->kv_k[slot], m->kv_v[slot], s->rr,
                    ly->a.rel_proj, sink, s->ao, (int)m->pos,
                    (int)m->kv_cap[slot], (int)KVD, (int)NH, (int)KVH,
                    (int)D, (int)c->d_rel, (int)c->rel_extent,
                    swa ? (int)c->swa_window : 0) != 0;
            } else {
                gpu_attn = h3v_gpu_inkling_prefill_attn(
                    s->q, m->kv_k[slot], m->kv_v[slot], s->rr,
                    ly->a.rel_proj, sink, s->ao, (int)m->pos, (int)n,
                    (int)m->kv_cap[slot], (int)KVD, (int)NH, (int)KVH,
                    (int)D, (int)c->d_rel, (int)c->rel_extent,
                    swa ? (int)c->swa_window : 0) != 0;
            }
            if (gpu_attn) m->gpu_calls++;
        }
        if (!gpu_attn) cwick_pool_for_items(n, &a, ik_attn_range);
        if (ik_prof_on()) ik_prof_add(&g_prof.attention, attn_t0);
    }

    /* ---- 6. gated output projection + post-attention convolution ---- */
    ik_round_bf16(s->ao, n * QD);
    for (int64_t t = 0; t < n; t++)
        for (int64_t i = 0; i < QD; i++)
            s->ao[t * QD + i] *= 1.0f / (1.0f + expf(-s->gg[t * QD + i]));
    ik_round_bf16(s->ao, n * QD);
    ik_linear_arg out_proj = {ly->a.wo, s->ao, s->tt, QD, H};
    ik_linear_group(m, n, &out_proj, 1);
    ik_round_bf16(s->tt, n * H);
    ik_sconv(c, s->tt, n, H, ly->pa_conv, NULL, 0, m->conv_hist[slot][2], true,
             1.0f / sqrtf((float)(K + 1)));

    /* ---- 7. residual stream and the block bank ---- */
    for (int64_t t = 0; t < n; t++) {
        float *hidden = s->tt + t * H;
        float *blocks = s->blocks + (size_t)t * B * H;
        if (boundary) {
            if (s->nblocks[t] < B)
                memcpy(blocks + (size_t)s->nblocks[t] * H, s->x + t * H,
                       (size_t)H * sizeof(float));
            if (s->nblocks[t] < B) s->nblocks[t]++;
        } else {
            for (int64_t i = 0; i < H; i++) hidden[i] += s->x[t * H + i];
        }
    }

    /* ---- 8. MLP input mix, message gate, norm ---- */
    for (int64_t t = 0; t < n; t++) {
        ik_mix(c, s->blocks + (size_t)t * B * H, s->nblocks[t], s->tt + t * H,
               ly->mlp_query, 1, s->normbuf, s->scores, s->mixout);
        memcpy(s->xn + t * H, s->mixout, (size_t)H * sizeof(float));
    }
    if (have_msgs && ly->is_loop) {
        float *cat = (float *)cwick_xmalloc((size_t)(2 * H) * sizeof(float));
        float *gate = (float *)cwick_xmalloc((size_t)c->gate_rank * sizeof(float));
        float *gf = (float *)cwick_xmalloc((size_t)H * sizeof(float));
        float *gm = (float *)cwick_xmalloc((size_t)H * sizeof(float));
        for (int64_t t = 0; t < n; t++)
            ik_control(c, ly, 3, s->xn + t * H, s->msgs + (size_t)t * 4 * H + 3 * H,
                       s->xn + t * H, cat, gate, gf, gm);
        free(cat); free(gate); free(gf); free(gm);
    }
    for (int64_t t = 0; t < n; t++)
        ik_rms_norm_bf16(s->xn + t * H, s->xn + t * H, H, ly->mlp_norm, c->rms_eps);

    /* ---- 9. MoE + post-MLP convolution ---- */
    ik_moe_layer(m, ly, s, n);
    ik_sconv(c, s->mo, n, H, ly->pm_conv, NULL, 0, m->conv_hist[slot][3], true,
             1.0f / sqrtf((float)(K + 1)));
    for (int64_t t = 0; t < n; t++)
        for (int64_t i = 0; i < H; i++) s->x[t * H + i] = s->tt[t * H + i] + s->mo[t * H + i];
    ik_round_bf16(s->x, n * H);
}

/* ------------------------------------------------------------------ */
/* forward                                                             */
/* ------------------------------------------------------------------ */

void ik_message_mix(const ik_cfg *c, const float *prev_blocks, int pre_nb,
                    const float *prev_h, const float *query, float *bank,
                    float *norm_buf, float *score_buf, float *out) {
    const int64_t H = c->hidden;
    memcpy(bank, prev_blocks + (size_t)(pre_nb + 1) * H,
           (size_t)H * sizeof(float));
    ik_mix(c, bank, 1, prev_h, query, 4, norm_buf, score_buf, out);
}

/* Build the cross-pass messages for one layer from the previous pass's first
 * completed core block and its final state. The final state is already ik_mix's
 * current source and must not be duplicated in the bank. */
static void ik_build_msgs(ik_model *m, const ik_layer *ly, ik_scratch *s,
                          int64_t n) {
    const int64_t B = IK_MAX_BLOCKS, H = m->c.hidden;
    for (int64_t t = 0; t < n; t++)
        ik_message_mix(&m->c,
                       s->prev_blocks + (size_t)t * B * H, s->pre_nb,
                       s->prev_h + t * H, ly->msg_query, s->mbank,
                       s->normbuf, s->scores, s->msgs + (size_t)t * 4 * H);
}

int ik_model_forward(ik_model *m, const int32_t *ids, int64_t n, int64_t pos,
                     float *logits_out, cwick_error *err) {
    if (n <= 0) return 0;
    const bool prof = ik_prof_on();
    const double total_t0 = prof ? ik_prof_now() : 0.0;
    if (prof) memset(&g_prof, 0, sizeof(g_prof));
    const ik_cfg *c = &m->c;
    const int64_t H = c->hidden, B = IK_MAX_BLOCKS;
    ik_scratch *s = ik_scratch_get(c, IK_CHUNK);
    if (!s) {
        cwick_error_set(err, CWICK_ERR_MEMORY, "inkling2 scratch allocation failed");
        return -1;
    }
    if (ik_scratch_scores(s, IK_CHUNK, pos + n) != 0) {
        cwick_error_set(err, CWICK_ERR_MEMORY, "inkling2 attention scratch failed");
        return -1;
    }
    m->pos = pos;
    int64_t done = 0;
    int64_t last_take = 0;
    while (done < n) {
        int64_t take = n - done < IK_CHUNK ? n - done : IK_CHUNK;
        m->pos = pos + done;

        /* embeddings: RMS norm with the checkpoint's own weight; image
         * placeholders are replaced by the tower's vectors as they go by */
        for (int64_t t = 0; t < take; t++) {
            int64_t id = ids[done + t];
            float *x = s->x + t * H;
            if (id == c->image_token_id && m->img_used < m->img_tokens) {
                memcpy(x, m->img_embeds + (size_t)m->img_used * (size_t)H,
                       (size_t)H * sizeof(float));
                ik_round_bf16(x, H);
                m->img_used++;
                continue;
            }
            if (id < 0 || id >= c->vocab) id = 0;
            const uint16_t *row = (const uint16_t *)m->embed + (size_t)id * (size_t)H;
            for (int64_t i = 0; i < H; i++) x[i] = cwick_bf16_to_f32(row[i]);
            ik_rms_norm_bf16(x, x, H, m->embed_norm, c->rms_eps);
        }
        for (int64_t t = 0; t < take; t++) s->nblocks[t] = 0;

        for (int64_t l = 0; l < c->loop_start; l++)
            ik_layer_forward(m, l, s, take, false, 0);

        memcpy(s->pre_h, s->x, (size_t)(take * H) * sizeof(float));
        memcpy(s->pre_blocks, s->blocks, (size_t)(take * B * H) * sizeof(float));
        s->pre_nb = s->nblocks[0];

        for (int64_t l = c->loop_start; l < c->loop_end; l++)
            ik_layer_forward(m, l, s, take, false, 0);

        for (int64_t pass = 1; pass < c->loop_passes; pass++) {
            memcpy(s->prev_h, s->x, (size_t)(take * H) * sizeof(float));
            memcpy(s->prev_blocks, s->blocks, (size_t)(take * B * H) * sizeof(float));
            s->prev_nb = s->nblocks[0];
            memcpy(s->x, s->pre_h, (size_t)(take * H) * sizeof(float));
            memcpy(s->blocks, s->pre_blocks, (size_t)(take * B * H) * sizeof(float));
            for (int64_t t = 0; t < take; t++) s->nblocks[t] = s->pre_nb;
            for (int64_t l = c->loop_start; l < c->loop_end; l++) {
                ik_build_msgs(m, &m->layers[l], s, take);
                ik_layer_forward(m, l, s, take, true, pass);
            }
        }

        for (int64_t l = c->loop_end; l < c->n_layers; l++)
            ik_layer_forward(m, l, s, take, false, 0);

        /* final block mix + norm */
        for (int64_t t = 0; t < take; t++) {
            ik_mix(c, s->blocks + (size_t)t * B * H, s->nblocks[t], s->x + t * H,
                   m->final_query, 1, s->normbuf, s->scores, s->mixout);
            memcpy(s->x + t * H, s->mixout, (size_t)H * sizeof(float));
            ik_rms_norm_bf16(s->x + t * H, s->x + t * H, H, m->out_norm, c->rms_eps);
        }
        m->pos += take;
        done += take;
        last_take = take;
    }
    m->pos = pos + n;

    cwick_metrics_set_ik_gpu_calls(m->gpu_calls);
    if (logits_out) {
        const int64_t V = c->unpadded_vocab > 0 ? c->unpadded_vocab : c->vocab;
        ik_linear_arg head = {m->unembed, s->x + (last_take - 1) * H,
                              logits_out, H, V};
        ik_linear_group(m, 1, &head, 1);
        for (int64_t i = 0; i < V; i++)
            logits_out[i] = tanhf(logits_out[i] / c->final_softcap) * c->final_softcap;
    }
    if (prof) {
        g_prof.total = ik_prof_now() - total_t0;
        fprintf(stderr, "IKPROF n=%lld pos=%lld total=%.3f linear=%.3f attn=%.3f "
                        "moe=%.3f route=%.3f up=%.3f act=%.3f down=%.3f combine=%.3f pair=%.3f "
                        "conv=%.3f mix=%.3f ctrl=%.3f layers=%lld\n",
                (long long)n, (long long)pos, g_prof.total * 1e3,
                g_prof.linear * 1e3, g_prof.attention * 1e3,
                g_prof.moe * 1e3, g_prof.route * 1e3, g_prof.up * 1e3,
                g_prof.act * 1e3, g_prof.down * 1e3, g_prof.combine * 1e3,
                g_prof.pair * 1e3, g_prof.conv * 1e3, g_prof.mix * 1e3,
                g_prof.control * 1e3, (long long)g_prof.layers);
    }
    return 0;
}
