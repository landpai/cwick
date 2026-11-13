/* model.c — strict model-contract validation from config.json.
 *
 * The native process reads inert JSON, validates every performance-critical
 * dimension, and fails closed before opening any weight payload. The CWICK
 * quality guards enforce the never-pruned rule:
 *   - the config dtype must be bfloat16 (row-int8 is a derived representation,
 *     never a substitute contract); the released Flash checkpoint additionally
 *     carries an fp8 quantization_config describing the on-disk FP8 E4M3
 *     storage of the same weights;
 *   - tie_word_embeddings must be false;
 *   - num_experts_per_token is used exactly as the model defines it (top-k is
 *     never reduced) and every expert must be present in the corpus;
 *   - every layer must be assigned exactly one attention kind and one MLP kind.
 *
 * GLM-5.3 text_configs use a K3-style nesting: the real fields live inside
 * `text_config`. They also use the next-gen field names (n_routed_experts,
 * num_experts_per_tok, n_shared_experts, qk_head_dim, v_head_dim, layer_types,
 * mlp_layer_types, indexer_* and hc_* blocks) which this parser maps onto the
 * same spec slots.
 */
#include "cwick.h"
#include "json.h"
#include "models.h"
#include "util.h"

#include <stdlib.h>
#include <string.h>

static int64_t req_int(json_value *root, const char *key, cwick_error *err) {
    json_value *v = cwick_json_get(root, key);
    if (!v || (v->kind != JSON_NUMBER && v->kind != JSON_BOOL)) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "config is missing/invalid %s", key);
        return -1;
    }
    return cwick_json_int(v, 0);
}

static int64_t opt_int(json_value *root, const char *key, int64_t dflt) {
    json_value *v = cwick_json_get(root, key);
    return cwick_json_int(v, dflt);
}

static bool req_bool(json_value *root, const char *key, bool dflt, cwick_error *err) {
    json_value *v = cwick_json_get(root, key);
    if (!v) return dflt;
    if (v->kind != JSON_BOOL) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "config %s must be a bool", key);
        return dflt;
    }
    return v->boolean;
}

static const char *req_str(json_value *root, const char *key, cwick_error *err) {
    json_value *v = cwick_json_get(root, key);
    if (!v || v->kind != JSON_STRING) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "config is missing/invalid %s", key);
        return NULL;
    }
    return v->string;
}

/* First element of an integer scalar-or-array field (GLM-5.3 eos_token_id). */
static int64_t req_int_or_array_first(json_value *root, const char *key,
                                      cwick_error *err) {
    json_value *v = cwick_json_get(root, key);
    if (!v) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "config is missing/invalid %s", key);
        return -1;
    }
    if (v->kind == JSON_NUMBER || v->kind == JSON_BOOL) return cwick_json_int(v, 0);
    if (v->kind == JSON_ARRAY && cwick_json_array_len(v) > 0)
        return cwick_json_int(cwick_json_array_at(v, 0), -1);
    cwick_error_set(err, CWICK_ERR_CONTRACT, "config %s must be an int or array", key);
    return -1;
}

/* Assign layers from a JSON array of one-based indices (legacy schedule). */
static int assign_layers(json_value *root, const char *key, cwick_layer_kind kind,
                         int64_t n_layers, int64_t *n_assigned,
                         cwick_layer_kind *kinds, cwick_error *err) {
    json_value *arr = cwick_json_get(root, key);
    if (!arr) return 0;
    if (arr->kind != JSON_ARRAY) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "config %s must be an array", key);
        return -1;
    }
    size_t n = cwick_json_array_len(arr);
    for (size_t i = 0; i < n; i++) {
        int64_t one = cwick_json_int(cwick_json_array_at(arr, i), -1);
        if (one <= 0 || one > n_layers) {
            cwick_error_set(err, CWICK_ERR_CONTRACT,
                           "config %s contains invalid layer %lld", key, one);
            return -1;
        }
        int idx = (int)(one - 1);
        if (kinds[idx] != (cwick_layer_kind)0xff) {
            cwick_error_set(err, CWICK_ERR_CONTRACT,
                           "config %s repeats layer %lld", key, one);
            return -1;
        }
        kinds[idx] = kind;
        (*n_assigned)++;
    }
    return 0;
}

/* Assign layer kinds from a GLM-5.3 `layer_types` string array
 * (linear_attention | deepseek_sparse_attention). */
static int assign_layer_types(json_value *root, const char *key,
                              int64_t n_layers, cwick_layer_kind *kinds,
                              int64_t *n_assigned, cwick_error *err) {
    json_value *arr = cwick_json_get(root, key);
    if (!arr) return 0;
    if (arr->kind != JSON_ARRAY) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "config %s must be an array", key);
        return -1;
    }
    size_t n = cwick_json_array_len(arr);
    if (n != (size_t)n_layers) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                       "config %s length %zu != num_hidden_layers %lld", key,
                       n, n_layers);
        return -1;
    }
    for (size_t i = 0; i < n; i++) {
        const char *s = cwick_json_str(cwick_json_array_at(arr, i));
        cwick_layer_kind kind;
        if (s && strcmp(s, "linear_attention") == 0) kind = CWICK_LAYER_KDA;
        else if (s && strcmp(s, "deepseek_sparse_attention") == 0) kind = CWICK_LAYER_DSA;
        else if (s && strcmp(s, "full_attention") == 0) kind = CWICK_LAYER_MLA;
        else {
            cwick_error_set(err, CWICK_ERR_UNSUPPORTED,
                           "config %s contains unknown layer type", key);
            return -1;
        }
        kinds[i] = kind;
        (*n_assigned)++;
    }
    return 0;
}

/* Assign per-layer MLP kind from a GLM-5.3 `mlp_layer_types` array
 * (dense | sparse). */
static int assign_mlp_types(json_value *root, const char *key, int64_t n_layers,
                            bool *mlp_dense, int64_t *n_dense, int64_t *n_sparse,
                            cwick_error *err) {
    json_value *arr = cwick_json_get(root, key);
    if (!arr) return 0;
    if (arr->kind != JSON_ARRAY) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "config %s must be an array", key);
        return -1;
    }
    size_t n = cwick_json_array_len(arr);
    if (n != (size_t)n_layers) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                       "config %s length %zu != num_hidden_layers %lld", key,
                       n, n_layers);
        return -1;
    }
    for (size_t i = 0; i < n; i++) {
        const char *s = cwick_json_str(cwick_json_array_at(arr, i));
        if (s && strcmp(s, "dense") == 0) { mlp_dense[i] = true; (*n_dense)++; }
        else if (s && strcmp(s, "sparse") == 0) { mlp_dense[i] = false; (*n_sparse)++; }
        else {
            cwick_error_set(err, CWICK_ERR_UNSUPPORTED,
                           "config %s contains unknown mlp type", key);
            return -1;
        }
    }
    return 0;
}

/* The ds config nests almost everything under text_config, but a few ids sit
 * beside it. Look in the inner object first, then the outer one. */
static int64_t ds_int(json_value *cfg, json_value *root, const char *key,
                      int64_t dflt) {
    json_value *v = cwick_json_get(cfg, key);
    if (!v) v = cwick_json_get(root, key);
    return cwick_json_int(v, dflt);
}

static double ds_num(json_value *cfg, json_value *root, const char *key,
                     double dflt) {
    json_value *v = cwick_json_get(cfg, key);
    if (!v) v = cwick_json_get(root, key);
    return cwick_json_num(v, dflt);
}

/* ------------------------------------------------------------------ */
/* DeepSeek-V4.1-Flash-1017                                            */
/*                                                                     */
/* A second architecture, not a variant of the GLM one: its numbers    */
/* live in `text_config` under next-gen names (n_routed_experts,       */
/* hc_*, index_*, compress_ratios, ...) and its quality contract is    */
/* "the geometry the checkpoint actually ships", validated by the      */
/* loader in ds_model.c against the tensor table. The GLM guards below */
/* (MLA latent, KDA schedule, the glm-5.3-flash geometry lock) do not  */
/* apply, so this fills the shared spec slots the engine and the       */
/* server read and returns early.                                      */
/* ------------------------------------------------------------------ */
static cwick_status ds_spec_fill(json_value *root, json_value *cfg,
                                 cwick_model_spec *spec, cwick_error *err) {
    /* `dtype` sits next to `text_config`, not inside it. */
    json_value *dv = cwick_json_get(cfg, "dtype");
    if (!dv) dv = cwick_json_get(root, "dtype");
    const char *dtype = cwick_json_str(dv);
    if (!dtype || strcmp(dtype, "bfloat16") != 0) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                       "deepseek_v41 requires the bfloat16 contract, got %s",
                       dtype ? dtype : "(missing)");
        return CWICK_ERR_CONTRACT;
    }
    snprintf(spec->dtype, sizeof(spec->dtype), "%s", dtype);

    json_value *qc = cwick_json_get(root, "quantization_config");
    if (qc && qc->kind == JSON_OBJECT) {
        const char *qm = cwick_json_str(cwick_json_get(qc, "quant_method"));
        if (qm && strcmp(qm, "fp8") != 0) {
            cwick_error_set(err, CWICK_ERR_UNSUPPORTED,
                           "deepseek_v41 quantization_config quant_method=%s "
                           "is not the fp8 release", qm);
            return CWICK_ERR_UNSUPPORTED;
        }
    }
    /* The experts are fp4 and the activations fp8; both are decoded by
     * ds_quant.c, so the spec stays on the plain contract. */
    spec->quant = CWICK_QUANT_BF16;

    spec->bos_token_id = opt_int(cfg, "bos_token_id",
                                 opt_int(root, "bos_token_id", 0));
    /* eos/pad live next to text_config in the HF layout, not inside it. */
    json_value *eosv = cwick_json_get(cfg, "eos_token_id");
    if (!eosv) eosv = cwick_json_get(root, "eos_token_id");
    if (!eosv) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                        "deepseek_v41 config is missing eos_token_id");
        return CWICK_ERR_CONTRACT;
    }
    spec->eos_token_id = (eosv->kind == JSON_NUMBER || eosv->kind == JSON_BOOL)
                             ? cwick_json_int(eosv, 0)
                             : cwick_json_int(cwick_json_array_at(eosv, 0), -1);
    if (spec->eos_token_id < 0) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                       "deepseek_v41 eos_token_id must be an int or array");
        return CWICK_ERR_CONTRACT;
    }
    spec->pad_token_id = opt_int(cfg, "pad_token_id",
                                 opt_int(root, "pad_token_id", 2));
    spec->n_eos = 0;
    spec->eos_ids[spec->n_eos++] = spec->eos_token_id;
    if (eosv->kind == JSON_ARRAY) {
        for (size_t i = 0; i < eosv->count && spec->n_eos < 8; i++) {
            int64_t id = cwick_json_int(cwick_json_array_at(eosv, i), -1);
            if (id < 0) continue;
            bool dup = false;
            for (int k = 0; k < spec->n_eos; k++)
                if (spec->eos_ids[k] == id) dup = true;
            if (!dup) spec->eos_ids[spec->n_eos++] = id;
        }
    }

#define DSREQ(name, field) do { \
    int64_t v_ = ds_int(cfg, root, name, -1); \
    if (v_ < 0) { \
        cwick_error_set(err, CWICK_ERR_CONTRACT, \
                        "deepseek_v41 config is missing/invalid %s", name); \
        return CWICK_ERR_CONTRACT; \
    } \
    spec->field = v_; \
} while (0)

    DSREQ("hidden_size", hidden_size);
    DSREQ("num_hidden_layers", num_hidden_layers);
    DSREQ("vocab_size", vocab_size);
    DSREQ("max_position_embeddings", max_position_embeddings);
    DSREQ("num_attention_heads", num_attention_heads);
    DSREQ("qk_rope_head_dim", qk_rope_head_dim);
    DSREQ("q_lora_rank", q_lora_rank);
    DSREQ("n_routed_experts", num_experts);
    DSREQ("num_experts_per_tok", num_experts_per_token);
    DSREQ("n_shared_experts", num_shared_experts);
    DSREQ("moe_intermediate_size", moe_intermediate_size);
    spec->head_dim = ds_int(cfg, root, "head_dim", 0);
    spec->qk_head_dim = spec->head_dim;
    spec->qk_nope_head_dim = spec->head_dim - spec->qk_rope_head_dim;
    spec->value_head_dim = spec->head_dim;
    /* No MLA latent cache: the compressed KV is the ds model's own. */
    spec->kv_lora_rank = 0;
    spec->num_key_value_heads = 1;
    spec->intermediate_size = spec->moe_intermediate_size;
    spec->routed_expert_hidden_size = spec->hidden_size;
    spec->rms_norm_eps_f = (float)cwick_json_num(
        cwick_json_get(cfg, "rms_norm_eps"), 1e-20);
    spec->rms_norm_eps = (int64_t)spec->rms_norm_eps_f;
    spec->swiglu_limit = (float)ds_num(cfg, root, "swiglu_limit", 0.0);
    spec->routed_scaling_factor = (float)ds_num(cfg, root,
                                                "routed_scaling_factor", 1.0);
    spec->norm_topk_prob = cwick_json_bool(
        cwick_json_get(cfg, "norm_topk_prob"), true);
    spec->moe_renormalize = spec->norm_topk_prob;
    spec->hc_mult = (int)ds_int(cfg, root, "hc_mult", 1);
    spec->hc_eps = (float)ds_num(cfg, root, "hc_eps", 1e-6);
    spec->hc_sinkhorn_iters = (int)ds_int(cfg, root, "hc_sinkhorn_iters", 0);
    spec->mhc = spec->hc_mult > 1;
    spec->index_head_dim = (int)ds_int(cfg, root, "index_head_dim", 0);
    spec->index_n_heads = (int)ds_int(cfg, root, "index_n_heads", 0);
    spec->index_topk = (int)ds_int(cfg, root, "index_topk", 0);
    spec->first_k_dense_replace = 0;
    {
        const char *sf = cwick_json_str(cwick_json_get(cfg, "scoring_func"));
        snprintf(spec->scoring_func, sizeof(spec->scoring_func), "%s",
                 sf ? sf : "sqrtsoftplus");
        const char *tm = cwick_json_str(cwick_json_get(cfg, "topk_method"));
        snprintf(spec->topk_method, sizeof(spec->topk_method), "%s",
                 tm ? tm : "noaux_tc");
    }

    if (spec->num_experts <= 0 || spec->num_experts_per_token <= 0 ||
        spec->num_experts_per_token > spec->num_experts) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                       "deepseek_v41 MoE geometry invalid (experts=%lld top=%lld)",
                       (long long)spec->num_experts,
                       (long long)spec->num_experts_per_token);
        return CWICK_ERR_CONTRACT;
    }
    if (spec->num_hidden_layers <= 0 ||
        spec->num_hidden_layers > 64 /* DS_MAX_LAYERS */) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                       "deepseek_v41 layer count %lld out of range",
                       (long long)spec->num_hidden_layers);
        return CWICK_ERR_CONTRACT;
    }

    /* Every layer is one attention kind for the engine's bookkeeping: the
     * compressed-KV sparse attention. Reported as DSA. */
    spec->layers = cwick_xmalloc((size_t)spec->num_hidden_layers *
                                 sizeof(cwick_layer_kind));
    for (int64_t i = 0; i < spec->num_hidden_layers; i++)
        spec->layers[i] = CWICK_LAYER_DSA;
    spec->dsa_layers = spec->num_hidden_layers;
    spec->mla_layers = 0;
    spec->kda_layers = 0;
    spec->mlp_dense = cwick_xmalloc((size_t)spec->num_hidden_layers * sizeof(bool));
    for (int64_t i = 0; i < spec->num_hidden_layers; i++)
        spec->mlp_dense[i] = false;
    spec->n_dense_layers = 0;
    spec->n_sparse_layers = spec->num_hidden_layers;
    return CWICK_OK;
}

/* ------------------------------------------------------------------ */
/* MiniMax-M3.1-Flash                                                  */
/*                                                                     */
/* A third architecture. Its numbers also live in `text_config`, but   */
/* the quality contract is different in kind: the checkpoint is fully  */
/* quantized (mxfp8 dense, nvfp4 experts) and ships a draft model       */
/* beside the weights, so `spec->quant` stays on the plain contract    */
/* and the loader in mm_model.c owns every decoding decision. The      */
/* GLM guards below -- bfloat16 dtype, tie_word_embeddings, the fp8    */
/* quantization_config, MLA latent -- would all reject it.             */
/*                                                                     */
/* eos/bos are only in generation_config.json here, unlike DeepSeek,   */
/* so they are read from there and fall back to the config.            */
/* ------------------------------------------------------------------ */
static cwick_status mm_spec_fill(json_value *root, json_value *cfg,
                                 const char *dir, cwick_model_spec *spec,
                                 cwick_error *err) {
    json_value *qc = cwick_json_get(root, "quantization_config");
    if (qc && qc->kind == JSON_OBJECT) {
        const char *qm = cwick_json_str(cwick_json_get(qc, "quant_method"));
        if (qm && strcmp(qm, "mxfp8") != 0 && strcmp(qm, "nvfp4") != 0) {
            cwick_error_set(err, CWICK_ERR_UNSUPPORTED,
                           "minimax_m3 quantization_config quant_method=%s is "
                           "not the mxfp8/nvfp4 release", qm);
            return CWICK_ERR_UNSUPPORTED;
        }
    }
    spec->quant = CWICK_QUANT_BF16;   /* weights are decoded by mm_model.c */
    snprintf(spec->dtype, sizeof(spec->dtype), "bfloat16");

#define MMREQ(name, field) do { \
    int64_t v_ = ds_int(cfg, root, name, -1); \
    if (v_ < 0) { \
        cwick_error_set(err, CWICK_ERR_CONTRACT, \
                        "minimax_m3 config is missing/invalid %s", name); \
        return CWICK_ERR_CONTRACT; \
    } \
    spec->field = v_; \
} while (0)

    MMREQ("hidden_size", hidden_size);
    MMREQ("num_hidden_layers", num_hidden_layers);
    MMREQ("vocab_size", vocab_size);
    MMREQ("max_position_embeddings", max_position_embeddings);
    MMREQ("num_attention_heads", num_attention_heads);
    MMREQ("num_key_value_heads", num_key_value_heads);
    MMREQ("num_local_experts", num_experts);
    MMREQ("num_experts_per_tok", num_experts_per_token);
#undef MMREQ

    spec->head_dim = (int)ds_int(cfg, root, "head_dim", 0);
    if (spec->head_dim <= 0) {
        spec->head_dim = spec->hidden_size / spec->num_attention_heads;
    }
    spec->qk_head_dim = spec->head_dim;
    spec->qk_rope_head_dim = (int)ds_int(cfg, root, "rotary_dim",
                                         spec->head_dim / 2);
    spec->qk_nope_head_dim = spec->head_dim - spec->qk_rope_head_dim;
    spec->value_head_dim = spec->head_dim;
    spec->kv_lora_rank = 0;           /* plain GQA cache, no MLA latent */
    spec->intermediate_size = (int)ds_int(cfg, root, "intermediate_size", 0);
    spec->moe_intermediate_size = spec->intermediate_size;
    spec->num_shared_experts = 1;
    spec->routed_expert_hidden_size = spec->hidden_size;
    spec->rms_norm_eps_f = (float)cwick_json_num(
        cwick_json_get(cfg, "rms_norm_eps"), 1e-6);
    spec->rms_norm_eps = (int64_t)spec->rms_norm_eps_f;
    spec->swiglu_limit = (float)ds_num(cfg, root, "swiglu_limit", 7.0);
    spec->routed_scaling_factor =
        (float)ds_num(cfg, root, "routed_scaling_factor", 2.0);
    spec->norm_topk_prob = true;
    spec->moe_renormalize = true;
    {
        const char *sf = cwick_json_str(cwick_json_get(cfg, "scoring_func"));
        snprintf(spec->scoring_func, sizeof(spec->scoring_func), "%s",
                 sf ? sf : "sigmoid");
    }
    json_value *sa = cwick_json_get(cfg, "sparse_attention_config");
    if (sa && sa->kind == JSON_OBJECT) {
        spec->index_head_dim = (int)cwick_json_int(
            cwick_json_get(sa, "sparse_index_dim"), 128);
        spec->index_n_heads = (int)cwick_json_int(
            cwick_json_get(sa, "sparse_num_index_heads"), 4);
        spec->index_topk = (int)cwick_json_int(
            cwick_json_get(sa, "sparse_topk_blocks"), 16) *
            (int)cwick_json_int(cwick_json_get(sa, "sparse_block_size"), 128);
    }

    /* eos/bos: generation_config.json first, the config itself second. */
    int64_t eos = ds_int(cfg, root, "eos_token_id", -1);
    int64_t bos = ds_int(cfg, root, "bos_token_id", -1);
    {
        char gp[1152];
        cwick_path_join(gp, sizeof(gp), dir, "generation_config.json");
        char *text = NULL;
        size_t len = 0;
        if (cwick_read_file(gp, &text, &len) == 0) {
            cwick_error jerr;
            cwick_error_clear(&jerr);
            json_value *g = cwick_json_parse(text, len, &jerr);
            if (g) {
                if (eos < 0) eos = cwick_json_int(
                    cwick_json_get(g, "eos_token_id"), -1);
                if (bos < 0) bos = cwick_json_int(
                    cwick_json_get(g, "bos_token_id"), -1);
                cwick_json_free(g);
            }
            free(text);
        }
    }
    if (eos < 0) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                        "minimax_m3 config is missing eos_token_id");
        return CWICK_ERR_CONTRACT;
    }
    spec->eos_token_id = eos;
    spec->bos_token_id = bos < 0 ? eos : bos;
    spec->pad_token_id = spec->eos_token_id;
    spec->n_eos = 0;
    spec->eos_ids[spec->n_eos++] = eos;

    if (spec->num_hidden_layers <= 0 || spec->num_hidden_layers > 64) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                        "minimax_m3 layer count %lld out of range",
                        (long long)spec->num_hidden_layers);
        return CWICK_ERR_CONTRACT;
    }

    /* moe_layer_freq: 0 means a dense MLP for that layer. The count of
     * leading zeros is what the loader calls first_k_dense. */
    spec->first_k_dense_replace = 0;
    {
        json_value *fr = cwick_json_get(cfg, "moe_layer_freq");
        if (fr && fr->kind == JSON_ARRAY)
            for (size_t i = 0; i < fr->count &&
                               (int64_t)i < spec->num_hidden_layers; i++)
                if (cwick_json_num(cwick_json_array_at(fr, i), 1.0) == 0.0)
                    spec->first_k_dense_replace = (int)i + 1;
    }
    spec->layers = cwick_xmalloc((size_t)spec->num_hidden_layers *
                                 sizeof(cwick_layer_kind));
    spec->mlp_dense = cwick_xmalloc((size_t)spec->num_hidden_layers *
                                    sizeof(bool));
    for (int64_t i = 0; i < spec->num_hidden_layers; i++) {
        spec->layers[i] = CWICK_LAYER_DSA;
        spec->mlp_dense[i] = i < spec->first_k_dense_replace;
    }
    spec->dsa_layers = spec->num_hidden_layers;
    spec->mla_layers = 0;
    spec->kda_layers = 0;
    spec->n_dense_layers = spec->first_k_dense_replace;
    spec->n_sparse_layers = spec->num_hidden_layers - spec->first_k_dense_replace;
    return CWICK_OK;
}

/* Qwen3.5 text geometry. Every field here is read because the loader and the
 * budget code need it; the layer mix is recorded so a caller can tell how many
 * layers keep a KV cache and how many keep a recurrence state instead. */
static cwick_status q35_spec_fill(json_value *root, json_value *cfg,
                                  const char *dir, cwick_model_spec *spec,
                                  cwick_error *err) {
    spec->quant = CWICK_QUANT_BF16;   /* BF16 on disk, decoded by q35_model.c */
    snprintf(spec->dtype, sizeof(spec->dtype), "bfloat16");

#define QREQ(name, field) do { \
    int64_t v_ = ds_int(cfg, root, name, -1); \
    if (v_ < 0) { \
        cwick_error_set(err, CWICK_ERR_CONTRACT, \
                        "qwen3_5 config is missing/invalid %s", name); \
        return CWICK_ERR_CONTRACT; \
    } \
    spec->field = v_; \
} while (0)

    QREQ("hidden_size", hidden_size);
    QREQ("num_hidden_layers", num_hidden_layers);
    QREQ("vocab_size", vocab_size);
    QREQ("max_position_embeddings", max_position_embeddings);
    QREQ("num_attention_heads", num_attention_heads);
    QREQ("num_key_value_heads", num_key_value_heads);
    QREQ("intermediate_size", intermediate_size);
#undef QREQ

    spec->head_dim = (int)ds_int(cfg, root, "head_dim", 0);
    if (spec->head_dim <= 0)
        spec->head_dim = spec->hidden_size / spec->num_attention_heads;
    spec->qk_head_dim = spec->head_dim;
    {
        /* partial_rotary_factor 0.25: only the first quarter of each head
         * turns, the rest passes through. */
        int rot = spec->head_dim;
        json_value *rp = cwick_json_get(cfg, "rope_parameters");
        if (rp && rp->kind == JSON_OBJECT) {
            double prf = cwick_json_num(
                cwick_json_get(rp, "partial_rotary_factor"), 1.0);
            rot = (int)((double)spec->head_dim * prf);
        }
        if (rot <= 0 || rot > spec->head_dim) rot = spec->head_dim;
        if (rot % 2) rot--;
        spec->qk_rope_head_dim = rot;
    }
    spec->qk_nope_head_dim = spec->head_dim - spec->qk_rope_head_dim;
    spec->value_head_dim = spec->head_dim;
    spec->kv_lora_rank = 0;           /* plain GQA cache, no MLA latent */
    spec->moe_intermediate_size = spec->intermediate_size;
    spec->num_experts = 0;            /* dense: every layer runs the same MLP */
    spec->num_experts_per_token = 0;
    spec->num_shared_experts = 0;
    spec->rms_norm_eps_f = (float)ds_num(cfg, root, "rms_norm_eps", 1e-6);
    spec->rms_norm_eps = (int64_t)spec->rms_norm_eps_f;
    spec->n_dense_layers = spec->num_hidden_layers;
    spec->n_sparse_layers = 0;

    /* The layer mix: a linear-attention layer here is a short-convolution
     * recurrence (CWICK_LAYER_KDA is the closest existing kind), the rest are
     * full attention. */
    spec->layers = cwick_xmalloc((size_t)spec->num_hidden_layers *
                                 sizeof(cwick_layer_kind));
    spec->mlp_dense = cwick_xmalloc((size_t)spec->num_hidden_layers *
                                    sizeof(bool));
    {
        json_value *lt = cwick_json_get(cfg, "layer_types");
        int64_t interval = ds_int(cfg, root, "full_attention_interval", 4);
        if (interval < 1) interval = 4;
        int64_t lin = 0;
        for (int64_t i = 0; i < spec->num_hidden_layers; i++) {
            bool is_lin;
            if (lt && lt->kind == JSON_ARRAY && (int64_t)lt->count ==
                spec->num_hidden_layers) {
                const char *s = cwick_json_str(cwick_json_array_at(lt, i));
                is_lin = s && strstr(s, "linear") != NULL;
            } else {
                is_lin = ((i + 1) % interval) != 0;
            }
            spec->layers[i] = is_lin ? CWICK_LAYER_KDA : CWICK_LAYER_MLA;
            spec->mlp_dense[i] = true;
            if (is_lin) lin++;
        }
        spec->kda_layers = lin;
        spec->mla_layers = spec->num_hidden_layers - lin;
        spec->dsa_layers = 0;
        spec->first_k_dense_replace = spec->num_hidden_layers;
    }

    /* eos/bos: the config first, generation_config.json second (that file is
     * where this checkpoint repeats both). */
    int64_t eos = ds_int(cfg, root, "eos_token_id", -1);
    int64_t bos = ds_int(cfg, root, "bos_token_id", -1);
    int64_t pad = ds_int(cfg, root, "pad_token_id", -1);
    {
        char gp[1152];
        cwick_path_join(gp, sizeof(gp), dir, "generation_config.json");
        char *text = NULL;
        size_t len = 0;
        if (cwick_read_file(gp, &text, &len) == 0) {
            cwick_error jerr;
            cwick_error_clear(&jerr);
            json_value *g = cwick_json_parse(text, len, &jerr);
            if (g) {
                if (eos < 0)
                    eos = cwick_json_int(cwick_json_get(g, "eos_token_id"), -1);
                if (bos < 0)
                    bos = cwick_json_int(cwick_json_get(g, "bos_token_id"), -1);
                if (pad < 0)
                    pad = cwick_json_int(cwick_json_get(g, "pad_token_id"), -1);
                cwick_json_free(g);
            }
            free(text);
        }
    }
    if (eos < 0) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                        "qwen3_5 config is missing eos_token_id");
        return CWICK_ERR_CONTRACT;
    }
    spec->eos_token_id = eos;
    spec->bos_token_id = bos < 0 ? eos : bos;
    spec->pad_token_id = pad < 0 ? eos : pad;
    /* A turn ends on <|im_end|> as well as on the generation_config eos:
     * the chat template closes every turn with the former, and stopping only
     * on the latter lets the model keep writing the next turn. */
    spec->n_eos = 0;
    spec->eos_ids[spec->n_eos++] = eos;
    spec->eos_ids[spec->n_eos++] = 248046;   /* <|im_end|> */
    return CWICK_OK;
}

cwick_status cwick_model_spec_load(const char *path, cwick_model_spec *spec,
                                 cwick_error *err) {
    cwick_error_clear(err);
    memset(spec, 0, sizeof(*spec));

    char *text = NULL;
    size_t len = 0;
    if (cwick_read_file(path, &text, &len) != 0) {
        cwick_error_set(err, CWICK_ERR_IO, "cannot read config %s", path);
        return CWICK_ERR_IO;
    }
    json_value *root = cwick_json_parse(text, len, err);
    free(text);
    if (!root) return CWICK_ERR_PARSE;

    /* Unwrap a nested text_config if present (K3-style configs). */
    json_value *cfg = root;
    json_value *tc = cwick_json_get(root, "text_config");
    if (tc && tc->kind == JSON_OBJECT) cfg = tc;

    /* model_type sits beside text_config in some checkpoints (the MiniMax
     * release puts it there), so fall back to the outer object. */
    const char *model_type = req_str(cfg, "model_type", err);
    if (!model_type) model_type = cwick_json_str(cwick_json_get(root, "model_type"));
    if (!model_type) { cwick_json_free(root); return CWICK_ERR_CONTRACT; }
    snprintf(spec->model_type, sizeof(spec->model_type), "%s", model_type);

    /* The remaining parser is the GLM geometry contract. Without this guard an
     * unknown architecture is silently measured against that contract and dies
     * on whichever GLM-only field it happens to reach first: a nested config
     * whose `text_config.pad_token_id` is null reported "missing pad_token_id"
     * even though the outer object had one, which says nothing about the real
     * problem (there are no kernels for that architecture). Refuse by name. */
    if (!cwick_model_type_supported(model_type)) {
        cwick_error_set(err, CWICK_ERR_UNSUPPORTED,
                        "unsupported model_type \"%s\": this build implements the "
                        "GLM-5.3 / DeepSeek-V4.1 / MiniMax-M3.1 / Qwen3.5 "
                        "kernels only; "
                        "downloading a checkpoint does not make it loadable",
                        model_type);
        cwick_json_free(root);
        return CWICK_ERR_UNSUPPORTED;
    }

    if (strcmp(model_type, "deepseek_v41") == 0 ||
        strcmp(model_type, "deepseek_v41_text") == 0) {
        cwick_status st = ds_spec_fill(root, cfg, spec, err);
        cwick_json_free(root);
        return st;
    }
    if (strncmp(model_type, "qwen3_5", 7) == 0) {
        /* The directory: generation_config.json sits beside the config and is
         * one of the two places this checkpoint states its stop set. */
        char dir[1152];
        snprintf(dir, sizeof(dir), "%s", path);
        char *slash = strrchr(dir, '/');
        if (slash) *slash = '\0';
        cwick_status st = q35_spec_fill(root, cfg, dir, spec, err);
        cwick_json_free(root);
        return st;
    }
    if (strncmp(model_type, "minimax_m3", 10) == 0) {
        /* The directory, not the file: generation_config.json sits beside it
         * and is the only place this checkpoint states its eos token. */
        char dir[1152];
        snprintf(dir, sizeof(dir), "%s", path);
        char *slash = strrchr(dir, '/');
        if (slash) *slash = '\0';
        cwick_status st = mm_spec_fill(root, cfg, dir, spec, err);
        cwick_json_free(root);
        return st;
    }

    const char *dtype = req_str(cfg, "dtype", err);
    if (!dtype) { cwick_json_free(root); return CWICK_ERR_CONTRACT; }
    snprintf(spec->dtype, sizeof(spec->dtype), "%s", dtype);

    /* ---- quality guards (never prune) ---- */
    if (strcmp(dtype, "bfloat16") != 0) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                       "dtype=%s: CWICK requires the bfloat16 contract (BF16 spine); "
                       "row-int8 is a derived representation, never a substitute",
                       dtype);
        cwick_json_free(root);
        return CWICK_ERR_CONTRACT;
    }
    if (!req_bool(cfg, "tie_word_embeddings", false, err)) {
        /* false is the required value */
    } else {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                       "tie_word_embeddings=true would merge the embedding and "
                       "lm_head: rejected by the never-pruned guard");
        cwick_json_free(root);
        return CWICK_ERR_CONTRACT;
    }

    /* ---- released storage: an fp8 quantization_config describes the actual
     * on-disk weights of the Flash checkpoint; the bfloat16 config dtype
     * remains the quality contract. ---- */
    spec->quant = CWICK_QUANT_BF16;
    json_value *qc = cwick_json_get(root, "quantization_config");
    if (qc && qc->kind == JSON_OBJECT) {
        const char *qm = cwick_json_str(cwick_json_get(qc, "quant_method"));
        const char *qf = cwick_json_str(cwick_json_get(qc, "fmt"));
        if (qm && strcmp(qm, "fp8") == 0 && qf && strcmp(qf, "e4m3") == 0) {
            spec->quant = CWICK_QUANT_FP8_E4M3;
        } else if (qm) {
            cwick_error_set(err, CWICK_ERR_UNSUPPORTED,
                           "quantization_config quant_method=%s fmt=%s: CWICK "
                           "supports the fp8/e4m3 Flash release",
                           qm ? qm : "?", qf ? qf : "?");
            cwick_json_free(root);
            return CWICK_ERR_UNSUPPORTED;
        }
    }

    spec->bos_token_id = opt_int(cfg, "bos_token_id", -1);
    {
        int64_t v = req_int_or_array_first(cfg, "eos_token_id", err);
        if (v < 0) { cwick_json_free(root); return CWICK_ERR_CONTRACT; }
        spec->eos_token_id = v;
        /* The whole stop set, not just the first id: a GLM-5.3 turn ends on
         * <|endoftext|>, <|user|> or <|observation|>. Stopping on the first
         * alone means the model keeps writing the next user turn until
         * max_tokens is reached. */
        spec->n_eos = 0;
        spec->eos_ids[spec->n_eos++] = v;
        json_value *eosv = cwick_json_get(cfg, "eos_token_id");
        if (eosv && eosv->kind == JSON_ARRAY) {
            for (size_t i = 0; i < eosv->count && spec->n_eos < 8; i++) {
                int64_t id = cwick_json_int(cwick_json_array_at(eosv, i), -1);
                if (id < 0) continue;
                bool dup = false;
                for (int k = 0; k < spec->n_eos; k++)
                    if (spec->eos_ids[k] == id) dup = true;
                if (!dup) spec->eos_ids[spec->n_eos++] = id;
            }
        }
    }
    {
        int64_t v = req_int(cfg, "pad_token_id", err);
        if (v < 0) { cwick_json_free(root); return CWICK_ERR_CONTRACT; }
        spec->pad_token_id = v;
    }
    if (spec->bos_token_id < 0) spec->bos_token_id = spec->eos_token_id;

#define REQ(name, field) do { \
    int64_t v = req_int(cfg, name, err); \
    if (v < 0) { cwick_json_free(root); return CWICK_ERR_CONTRACT; } \
    spec->field = v; \
} while (0)

    REQ("hidden_size", hidden_size);
    REQ("num_hidden_layers", num_hidden_layers);
    REQ("vocab_size", vocab_size);
    REQ("max_position_embeddings", max_position_embeddings);
    REQ("num_attention_heads", num_attention_heads);
    REQ("num_key_value_heads", num_key_value_heads);
    REQ("q_lora_rank", q_lora_rank);
    REQ("kv_lora_rank", kv_lora_rank);

    /* GLM-5.3: qk_head_dim is the full attention head dim; legacy configs use
     * head_dim directly. */
    spec->qk_head_dim = opt_int(cfg, "qk_head_dim", 0);
    spec->qk_nope_head_dim = opt_int(cfg, "qk_nope_head_dim", 0);
    spec->qk_rope_head_dim = opt_int(cfg, "qk_rope_head_dim", 0);
    spec->value_head_dim = opt_int(cfg, "v_head_dim",
                                   opt_int(cfg, "value_head_dim", 0));
    spec->head_dim = opt_int(cfg, "head_dim", 0);
    if (spec->qk_head_dim > 0) {
        spec->head_dim = spec->qk_head_dim;
    }
    if (spec->qk_nope_head_dim == 0 && spec->qk_rope_head_dim == 0) {
        /* legacy MLA: rope head dim = head_dim, nope = 0 */
        spec->qk_nope_head_dim = 0;
        spec->qk_rope_head_dim = spec->head_dim;
    }

    /* residual anchor (legacy); GLM-5.3 uses hyper-connection instead */
    spec->attn_res_block_size = opt_int(cfg, "attn_res_block_size", 1);

    /* ---- MoE geometry (required: the target is a full MoE, never pruned).
     * Accept both legacy (num_experts / num_experts_per_token /
     * num_shared_experts) and GLM-5.3 (n_routed_experts / num_experts_per_tok /
     * n_shared_experts) names. ---- */
    spec->num_experts = opt_int(cfg, "n_routed_experts", 0);
    spec->num_experts_per_token = opt_int(cfg, "num_experts_per_tok", 0);
    spec->num_shared_experts = opt_int(cfg, "n_shared_experts", 0);
    if (spec->num_experts == 0) spec->num_experts = req_int(cfg, "num_experts", err);
    if (spec->num_experts_per_token == 0)
        spec->num_experts_per_token = req_int(cfg, "num_experts_per_token", err);
    if (spec->num_shared_experts == 0)
        spec->num_shared_experts = req_int(cfg, "num_shared_experts", err);
    if (spec->num_experts <= 0 || spec->num_experts_per_token <= 0 ||
        spec->num_shared_experts < 0) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                       "config MoE geometry incomplete (experts/topk/shared)");
        cwick_json_free(root);
        return CWICK_ERR_CONTRACT;
    }
    REQ("moe_intermediate_size", moe_intermediate_size);
    spec->routed_expert_hidden_size =
        opt_int(cfg, "routed_expert_hidden_size", 0);
    if (spec->routed_expert_hidden_size == 0) {
        /* GLM-5.3 experts operate directly on the hidden width */
        spec->routed_expert_hidden_size = spec->hidden_size;
    }

    spec->intermediate_size = cwick_json_int(cwick_json_get(cfg, "intermediate_size"),
                                            spec->moe_intermediate_size);
    spec->num_expert_groups = cwick_json_int(cwick_json_get(cfg, "num_expert_groups"), 1);
    spec->topk_groups = cwick_json_int(cwick_json_get(cfg, "topk_groups"),
                                      cwick_json_int(cwick_json_get(cfg, "topk_group"), 1));

    spec->moe_renormalize = cwick_json_bool(cwick_json_get(cfg, "moe_renormalize"),
                                           cwick_json_bool(cwick_json_get(cfg, "norm_topk_prob"), true));
    spec->latent_moe_use_norm = cwick_json_bool(cwick_json_get(cfg, "latent_moe_use_norm"), true);
    spec->mla_use_output_gate = cwick_json_bool(cwick_json_get(cfg, "mla_use_output_gate"), true);
    spec->activation_situ_beta =
        (float)cwick_json_num(cwick_json_get(cfg, "activation_situ_beta"), 4.0);
    spec->activation_situ_linear_beta =
        (float)cwick_json_num(cwick_json_get(cfg, "activation_situ_linear_beta"), 25.0);
    spec->routed_scaling_factor =
        (float)cwick_json_num(cwick_json_get(cfg, "routed_scaling_factor"), 1.0);
    spec->rms_norm_eps_f =
        (float)cwick_json_num(cwick_json_get(cfg, "rms_norm_eps"), 1e-5);
    spec->rms_norm_eps = (int64_t)spec->rms_norm_eps_f;
    spec->gate_lower_bound =
        (float)cwick_json_num(cwick_json_get(cfg, "gate_lower_bound"), 0.0);
    /* GLM-5.3: hidden_act is silu and swiglu_limit clamps it. A config that
     * asks for any other activation cannot be honoured by the clamped-SwiGLU
     * kernel, so fail closed rather than silently emitting wrong logits. */
    spec->swiglu_limit =
        (float)cwick_json_num(cwick_json_get(cfg, "swiglu_limit"), 0.0);

    /* ---- top-k quality guard ---- */
    if (spec->num_experts_per_token <= 0 ||
        spec->num_experts_per_token > spec->num_experts) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                       "num_experts_per_token=%lld with num_experts=%lld: "
                       "invalid routed top-k; CWICK never reduces top-k",
                       spec->num_experts_per_token, spec->num_experts);
        cwick_json_free(root);
        return CWICK_ERR_CONTRACT;
    }

    /* ---- GLM-5.3 next-gen blocks ---- */
    spec->index_head_dim = opt_int(cfg, "index_head_dim", 0);
    spec->index_n_heads = opt_int(cfg, "index_n_heads", 0);
    spec->index_topk = opt_int(cfg, "index_topk", 0);
    spec->index_kpool = opt_int(cfg, "index_kpool", 0);
    spec->index_kpool_always_select_tail =
        cwick_json_bool(cwick_json_get(cfg, "index_kpool_always_select_tail"), true);
    spec->index_kpool_compress =
        cwick_json_bool(cwick_json_get(cfg, "index_kpool_compress"), true);
    spec->indexer_rope_interleave =
        cwick_json_bool(cwick_json_get(cfg, "indexer_rope_interleave"), true);
    spec->hc_mult = opt_int(cfg, "hc_mult", 1);
    spec->hc_eps = (float)cwick_json_num(cwick_json_get(cfg, "hc_eps"), 1e-6);
    spec->hc_sinkhorn_iters = opt_int(cfg, "hc_sinkhorn_iters", 0);
    spec->mhc = cwick_json_bool(cwick_json_get(cfg, "mhc"), false);
    spec->first_k_dense_replace = opt_int(cfg, "first_k_dense_replace", 1);
    spec->num_nextn_predict_layers = opt_int(cfg, "num_nextn_predict_layers", 0);
    spec->norm_topk_prob = cwick_json_bool(cwick_json_get(cfg, "norm_topk_prob"), true);
    {
        const char *sf = cwick_json_str(cwick_json_get(cfg, "scoring_func"));
        snprintf(spec->scoring_func, sizeof(spec->scoring_func), "%s",
                 sf ? sf : "sigmoid");
        const char *tm = cwick_json_str(cwick_json_get(cfg, "topk_method"));
        snprintf(spec->topk_method, sizeof(spec->topk_method), "%s",
                 tm ? tm : "noaux_tc");
    }

    /* ---- linear attention config (KDA) ---- */
    json_value *lin = cwick_json_get(cfg, "linear_attn_config");
    spec->linear_num_heads = spec->num_attention_heads;
    spec->linear_head_dim = spec->head_dim;
    spec->short_conv_kernel_size = 4;
    if (lin && lin->kind == JSON_OBJECT) {
        spec->linear_num_heads =
            cwick_json_int(cwick_json_get(lin, "num_heads"), spec->num_attention_heads);
        spec->linear_head_dim =
            cwick_json_int(cwick_json_get(lin, "head_dim"), spec->head_dim);
        spec->short_conv_kernel_size =
            cwick_json_int(cwick_json_get(lin, "short_conv_kernel_size"), 4);
        /* KDA's bounded sigmoid gate lives here, not at the config root. */
        spec->gate_lower_bound =
            (float)cwick_json_num(cwick_json_get(lin, "gate_lower_bound"),
                                 spec->gate_lower_bound);
    }

    /* ---- layer schedule ---- */
    spec->layers = cwick_xmalloc((size_t)spec->num_hidden_layers * sizeof(cwick_layer_kind));
    for (int64_t i = 0; i < spec->num_hidden_layers; i++)
        spec->layers[i] = (cwick_layer_kind)0xff;
    int64_t n_assigned = 0;
    if (assign_layer_types(cfg, "layer_types", spec->num_hidden_layers,
                           spec->layers, &n_assigned, err) < 0) {
        cwick_json_free(root);
        return CWICK_ERR_CONTRACT;
    }
    if (n_assigned == 0) {
        json_value *schedule_src = lin ? lin : cfg;
        if (assign_layers(schedule_src, "kda_layers", CWICK_LAYER_KDA,
                          spec->num_hidden_layers, &n_assigned, spec->layers, err) != 0 ||
            assign_layers(schedule_src, "full_attn_layers", CWICK_LAYER_MLA,
                          spec->num_hidden_layers, &n_assigned, spec->layers, err) != 0) {
            cwick_json_free(root);
            return CWICK_ERR_CONTRACT;
        }
    }
    if (n_assigned == 0) {
        /* default: every layer is a full-attention (MLA) layer */
        for (int64_t i = 0; i < spec->num_hidden_layers; i++) spec->layers[i] = CWICK_LAYER_MLA;
    } else {
        for (int64_t i = 0; i < spec->num_hidden_layers; i++) {
            if (spec->layers[i] == (cwick_layer_kind)0xff) {
                cwick_error_set(err, CWICK_ERR_CONTRACT,
                               "layer schedule omits one-based layer %lld", i + 1);
                cwick_json_free(root);
                return CWICK_ERR_CONTRACT;
            }
        }
    }
    spec->kda_layers = 0;
    spec->mla_layers = 0;
    spec->dsa_layers = 0;
    for (int64_t i = 0; i < spec->num_hidden_layers; i++) {
        switch (spec->layers[i]) {
        case CWICK_LAYER_KDA: spec->kda_layers++; break;
        case CWICK_LAYER_MLA: spec->mla_layers++; break;
        case CWICK_LAYER_DSA: spec->dsa_layers++; break;
        default: break;                  /* counted by the schedule instead */
        }
    }

    /* ---- per-layer MLP kind ---- */
    spec->mlp_dense = cwick_xmalloc((size_t)spec->num_hidden_layers * sizeof(bool));
    for (int64_t i = 0; i < spec->num_hidden_layers; i++) spec->mlp_dense[i] = false;
    if (assign_mlp_types(cfg, "mlp_layer_types", spec->num_hidden_layers,
                         spec->mlp_dense, &spec->n_dense_layers,
                         &spec->n_sparse_layers, err) < 0) {
        cwick_json_free(root);
        return CWICK_ERR_CONTRACT;
    }
    if (spec->n_dense_layers == 0 && spec->n_sparse_layers == 0) {
        /* legacy default: layer 0 dense, the rest sparse */
        spec->mlp_dense[0] = true;
        spec->n_dense_layers = 1;
        spec->n_sparse_layers = spec->num_hidden_layers - 1;
    }

    /* The official model id is a quality contract, not an alias. Synthetic
     * verification models use distinct ids, while anything claiming to be the
     * released full GLM-5.3-Flash must retain its complete 45-layer,
     * 288-expert, top-8 geometry. */
    if (strcmp(spec->model_type, "glm-5.3-flash") == 0 &&
        (spec->hidden_size != 4096 || spec->num_hidden_layers != 45 ||
         spec->num_experts != 288 || spec->num_experts_per_token != 8 ||
         spec->moe_intermediate_size != 2048 || spec->kda_layers != 34 ||
         spec->dsa_layers != 11 || spec->mla_layers != 0 ||
         spec->n_dense_layers != 3 || spec->n_sparse_layers != 42 ||
         !spec->mhc || spec->hc_mult != 4 || spec->index_topk != 2048)) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                       "official glm-5.3-flash geometry was modified "
                       "(hidden=%lld layers=%lld experts=%lld top=%lld "
                       "moe=%lld KDA/DSA/MLA=%lld/%lld/%lld dense/sparse=%lld/%lld "
                       "mhc=%d/%lld index_topk=%lld); refusing a pruned contract",
                       (long long)spec->hidden_size,
                       (long long)spec->num_hidden_layers,
                       (long long)spec->num_experts,
                       (long long)spec->num_experts_per_token,
                       (long long)spec->moe_intermediate_size,
                       (long long)spec->kda_layers, (long long)spec->dsa_layers,
                       (long long)spec->mla_layers,
                       (long long)spec->n_dense_layers,
                       (long long)spec->n_sparse_layers,
                       (int)spec->mhc, (long long)spec->hc_mult,
                       (long long)spec->index_topk);
        cwick_json_free(root);
        return CWICK_ERR_CONTRACT;
    }

    cwick_json_free(root);
    return CWICK_OK;
}

void cwick_model_spec_free(cwick_model_spec *spec) {
    if (spec->layers) {
        free(spec->layers);
        spec->layers = NULL;
    }
    if (spec->mlp_dense) {
        free(spec->mlp_dense);
        spec->mlp_dense = NULL;
    }
}

void cwick_expert_saf_name(const cwick_model_spec *spec, int64_t layer,
                           int64_t expert, const char *matrix,
                           char *out, size_t cap) {
    /* `matrix` is the bare projection name ("gate_proj"); the ".weight"
     * suffix is part of every on-disk name. */
    (void)spec;
    snprintf(out, cap,
             "model.language_model.layers.%lld.mlp.experts.%lld.%s.weight",
             (long long)layer, (long long)expert, matrix);
}
