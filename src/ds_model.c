/*
 * ds_model.c — the DeepSeek-V4.1-Flash-1017 forward pass.
 *
 * Structure follows the reference implementation's module boundaries, in the
 * same order, because several of them are order-dependent: the indexer needs
 * the compressed latent *before* it is rotated and written to the cache, and
 * the attention output is un-rotated before the block-diagonal wo_a.
 *
 * Two deliberate deviations, both documented where they happen:
 *   * the residual stream is fp32 here and bf16 in the reference. Every point
 *     where the reference rounds to bf16 is reproduced with ds_bf16_round, so
 *     the streams carry the same values;
 *   * hc_post's residual mixing is selectable (CWICK_DS_HC_COMB). See the
 *     comment on ds_hc_post.
 *
 * Prefill runs layer-major over chunks of DS_CHUNK positions: the dense
 * non-expert weights are ~5 GB and re-reading them once per token instead of
 * once per chunk costs more than the whole rest of the pass.
 */
#include "ds_model.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "cwick.h"   /* cwick_pool_for: attention splits over heads */

#include "ds.h"
#include "json.h"
#include "log.h"
#include "mem.h"
#include "metrics.h"
#include "util.h"

#include <stdatomic.h>

/* ------------------------------------------------------------------ */
/* debug trace                                                          */
/*                                                                      */
/* DS_TRACE=1 makes the forward pass print named buffers as hex floats  */
/* on stdout, so tools/ds_layertest.py can recompute them from the      */
/* shard bytes alone and compare. Nothing in the model reads this.      */
/*                                                                      */
/* DS_TRACE_MAXL bounds the layer index of the heavy per-sublayer       */
/* buffers (SUBIN/ATTNOUT/FFNIN/FFNOUT/LATENT/IDXTOP). HIN and ENGHASH  */
/* are the seeds the oracle starts from, so they are always dumped and  */
/* are cheap: one [hc, dim] row and 24 integers per token.              */
/* ------------------------------------------------------------------ */

static int ds_trace_on = 0;
static int ds_trace_maxl = 2;

/* Which of the heavy buffers to print for layer L. */
static int ds_trace_layer(int L) {
    return ds_trace_on && L >= 0 && L <= ds_trace_maxl;
}

static void ds_trace(const char *tag, const float *v, int n) {
    if (!ds_trace_on) return;
    printf("%s", tag);
    for (int i = 0; i < n; i++) printf(" %a", (double)v[i]);
    printf("\n");
    fflush(stdout);
}

static void ds_trace_i(const char *tag, const int64_t *v, int n) {
    if (!ds_trace_on) return;
    printf("%s", tag);
    for (int i = 0; i < n; i++) printf(" %lld", (long long)v[i]);
    printf("\n");
    fflush(stdout);
}

/* ------------------------------------------------------------------ */
/* where the forward pass spends its time (DS_TIMING=1)                 */
/*                                                                      */
/* Guessing the bottleneck cost more time than these counters do: the    */
/* profile has moved from the dense projections to the expert reads to   */
/* the rotation/attention tail as each was fixed, and only measurement   */
/* showed which. One line per forward pass, percentage of the total.     */
/* ------------------------------------------------------------------ */

enum { DS_T_ATTN, DS_T_ROUTE, DS_T_FETCH, DS_T_PIN, DS_T_EC, DS_T_SHARED, DS_T_TAIL,
       DS_T_QKV, DS_T_INDEX, DS_T_SPARSE, DS_T_WO, DS_T_HEAD,
       DS_T_N };
static const char *const ds_t_name[DS_T_N] = {
    "attention", "routing", "expert fetch", "expert pin", "expert matmul",
    "shared+sum", "block tail", "qkv+rope", "indexer", "sparse attn", "wo", "head"
};
static int ds_timing_on;
static uint64_t ds_t_ns[DS_T_N];

static void ds_timing_init(void) {
    const char *e = getenv("DS_TIMING");
    ds_timing_on = e && *e && *e != '0';
}

/* ------------------------------------------------------------------ */
/* small helpers                                                        */
/* ------------------------------------------------------------------ */

static void *xmalloc(size_t n) { return malloc(n ? n : 1); }

/* Quantize one vector into fp8 bytes plus per-32 e8m0 scales, exactly the way
 * the reference's act_quant(..., scale_fmt="ue8m0") does before a quantized
 * GEMM. */
static void ds_quant_act(const float *x, int n, uint8_t *q, uint8_t *s) {
    for (int b = 0; b < n; b += 32)
        s[b / 32] = ds_act_quant_block(x + b, q + b);
}

/* ------------------------------------------------------------------ */
/* config                                                               */
/* ------------------------------------------------------------------ */

static int json_int_array(json_value *v, int *out, int cap, int dflt) {
    if (!v || v->kind != JSON_ARRAY) return 0;
    size_t n = cwick_json_array_len(v);
    if ((int)n > cap) n = cap;
    for (size_t i = 0; i < n; i++)
        out[i] = (int)cwick_json_int(cwick_json_array_at(v, i), dflt);
    return (int)n;
}

static int ds_load_config(ds_model *m, const char *root, int max_seq_len,
                          cwick_error *err) {
    ds_config *c = &m->cfg;
    memset(c, 0, sizeof(*c));

    char path[4096];
    snprintf(path, sizeof(path), "%s/config.json", root);
    FILE *f = fopen(path, "rb");
    if (!f) {
        cwick_error_set(err, CWICK_ERR_IO, "cannot open %s", path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *text = xmalloc((size_t)sz + 1);
    size_t got = fread(text, 1, (size_t)sz, f);
    text[got] = 0;
    fclose(f);

    json_value *rootv = cwick_json_parse(text, got, err);
    free(text);
    if (!rootv) return -1;

    json_value *tc = cwick_json_get(rootv, "text_config");
    if (!tc || tc->kind != JSON_OBJECT) tc = rootv; /* flat layout fallback */

    json_value *v;
    c->vocab_size = (int)cwick_json_int(cwick_json_get(tc, "vocab_size"), 129280);
    c->dim = (int)cwick_json_int(cwick_json_get(tc, "hidden_size"), 5120);
    c->moe_inter_dim = (int)cwick_json_int(cwick_json_get(tc, "moe_intermediate_size"), 2304);
    c->n_layers = (int)cwick_json_int(cwick_json_get(tc, "num_hidden_layers"), 40);
    c->n_heads = (int)cwick_json_int(cwick_json_get(tc, "num_attention_heads"), 64);
    c->head_dim = (int)cwick_json_int(cwick_json_get(tc, "head_dim"), 512);
    c->rope_head_dim = (int)cwick_json_int(cwick_json_get(tc, "qk_rope_head_dim"), 64);
    c->q_lora_rank = (int)cwick_json_int(cwick_json_get(tc, "q_lora_rank"), 1280);
    c->o_lora_rank = (int)cwick_json_int(cwick_json_get(tc, "o_lora_rank"), 1024);
    c->o_groups = (int)cwick_json_int(cwick_json_get(tc, "o_groups"), 8);
    c->window_size = (int)cwick_json_int(cwick_json_get(tc, "sliding_window"), 128);
    c->n_routed_experts = (int)cwick_json_int(cwick_json_get(tc, "n_routed_experts"), 384);
    c->n_shared_experts = (int)cwick_json_int(cwick_json_get(tc, "n_shared_experts"), 1);
    c->n_activated_experts = (int)cwick_json_int(cwick_json_get(tc, "num_experts_per_tok"), 6);
    c->swiglu_limit = (float)cwick_json_num(cwick_json_get(tc, "swiglu_limit"), 10.0);
    c->rms_norm_eps = (float)cwick_json_num(cwick_json_get(tc, "rms_norm_eps"), 1e-20);
    c->norm_topk_prob = (int)cwick_json_bool(cwick_json_get(tc, "norm_topk_prob"), true);
    c->route_scale = (float)cwick_json_num(cwick_json_get(tc, "routed_scaling_factor"), 1.5);
    /* the checkpoint leaves the routing temperature at its default, so a
     * missing key means 1.0; dividing by zero would turn every score into
     * inf and the normalised top-k weights into NaN */
    c->gate_temp = (float)cwick_json_num(cwick_json_get(tc, "gate_temp"), 1.0);
    if (!(c->gate_temp > 0.0f)) c->gate_temp = 1.0f;
    c->index_n_heads = (int)cwick_json_int(cwick_json_get(tc, "index_n_heads"), 32);
    c->index_head_dim = (int)cwick_json_int(cwick_json_get(tc, "index_head_dim"), 128);
    c->index_topk = (int)cwick_json_int(cwick_json_get(tc, "index_topk"), 512);
    c->candidate_source_layer = (int)cwick_json_int(cwick_json_get(tc, "candidate_source_layer_id"), 20);
    c->candidate_topk_blocks = (int)cwick_json_int(cwick_json_get(tc, "candidate_topk_blocks"), 2048);
    c->candidate_block_size = (int)cwick_json_int(cwick_json_get(tc, "candidate_block_size"), 8);
    c->hc_mult = (int)cwick_json_int(cwick_json_get(tc, "hc_mult"), 4);
    c->hc_sinkhorn_iters = (int)cwick_json_int(cwick_json_get(tc, "hc_sinkhorn_iters"), 20);
    c->hc_eps = (float)cwick_json_num(cwick_json_get(tc, "hc_eps"), 1e-6);
    c->engram_max_ngram_size = (int)cwick_json_int(cwick_json_get(tc, "engram_max_ngram_size"), 4);
    c->engram_vocab_size = (int)cwick_json_int(cwick_json_get(tc, "engram_vocab_size"), 16000000);
    c->engram_n_heads = (int)cwick_json_int(cwick_json_get(tc, "engram_n_heads"), 8);
    c->engram_head_dim = (int)cwick_json_int(cwick_json_get(tc, "engram_head_dim"), 256);
    c->engram_pad_id = (int)cwick_json_int(cwick_json_get(tc, "engram_pad_token_id"), 2);
    c->engram_compressed_vocab_size = (int)cwick_json_int(cwick_json_get(tc, "engram_compressed_vocab_size"), 99092);

    c->rope_theta = cwick_json_num(cwick_json_get(tc, "rope_theta"), 10000.0);
    c->compress_rope_theta = cwick_json_num(cwick_json_get(tc, "compress_rope_theta"), 160000.0);
    v = cwick_json_get(tc, "rope_scaling");
    c->rope_factor = cwick_json_num(cwick_json_get(v, "factor"), 16.0);
    c->original_seq_len = (int)cwick_json_int(cwick_json_get(v, "original_max_position_embeddings"), 65536);
    c->beta_fast = (int)cwick_json_int(cwick_json_get(v, "beta_fast"), 32);
    c->beta_slow = (int)cwick_json_int(cwick_json_get(v, "beta_slow"), 1);

    json_int_array(cwick_json_get(tc, "compress_ratios"), c->compress_ratios, DS_MAX_LAYERS, 0);
    c->n_kv_source = json_int_array(cwick_json_get(tc, "kv_source_layer_ids"),
                                    c->kv_source_layers, DS_MAX_LAYERS, -1);
    c->n_index_source = json_int_array(cwick_json_get(tc, "index_source_layer_ids"),
                                       c->index_source_layers, DS_MAX_LAYERS, -1);
    c->n_engram = json_int_array(cwick_json_get(tc, "engram_layer_ids"),
                                 c->engram_layer_ids, DS_MAX_LAYERS, -1);
    v = cwick_json_get(tc, "engram_num_embeddings");
    if (v && v->kind == JSON_ARRAY) {
        size_t n = cwick_json_array_len(v);
        if ((int)n > DS_MAX_LAYERS) n = DS_MAX_LAYERS;
        for (size_t i = 0; i < n; i++)
            c->engram_num_embeddings[i] = cwick_json_int(cwick_json_array_at(v, i), 0);
    }
    c->max_seq_len = max_seq_len;

    cwick_json_free(rootv);

    if (c->n_layers > DS_MAX_LAYERS) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "%d layers exceeds the %d this build supports",
                        c->n_layers, DS_MAX_LAYERS);
        return -1;
    }
    if (c->hc_mult > DS_MAX_HC) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "hc_mult %d exceeds %d", c->hc_mult, DS_MAX_HC);
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* weight binding                                                       */
/* ------------------------------------------------------------------ */

/* ---- resident pin arena ------------------------------------------------ */

/* Tensors that stay in the shard mapping: the routed experts (read through
 * the expert cache or the stream) and the engram tables (24 rows per token,
 * out of 2 x 384M). Everything else is touched in full every token. */
static int ds_pin_skip(const char *name) {
    if (strstr(name, ".experts.")) return 1;
    if (strstr(name, "engram.embed")) return 1;
    if (strstr(name, "vision")) return 1;
    if (strstr(name, "mtp")) return 1;
    if (strstr(name, "aligner")) return 1;
    return 0;
}

/* CWICK_DS_EXACT=1 keeps the fp8 dense weights exactly as the checkpoint
 * stores them: no bf16 widening at load, and the reference-order scalar
 * kernels. The widened path is lossless but sums four lanes instead of one
 * chain, so the layer-by-layer oracle check runs in this mode. */
static int ds_exact_layout(void) {
    static int cached = -1;
    if (cached < 0) cached = cwick_env_true("CWICK_DS_EXACT") ? 1 : 0;
    return cached;
}

/* How many experts of a chunked-MoE union are put in flight at once. The
 * default is `n_activated_experts`: one prefetch arena zone, the widest batch
 * whose reads all fit before the first one is needed. CWICK_DS_PF_BATCH=1
 * reproduces the old one-expert-at-a-time order exactly, which is how the
 * batching is checked to be output-neutral rather than merely plausible. */
static int ds_pf_batch(int dflt) {
    static int cached = -2;
    if (cached == -2) {
        const char *e = getenv("CWICK_DS_PF_BATCH");
        int v = e && *e ? atoi(e) : 0;
        cached = v > 0 ? v : 0;
    }
    if (cached <= 0) return dflt;
    return cached < dflt ? cached : dflt;
}

/* Point `t` at a private copy. Returns the mapped bytes when the arena is
 * exhausted or absent, so a short budget degrades instead of failing. */
static const void *ds_pin_tensor(ds_model *m, const cwick_st_tensor *t) {
    if (!t->data || !t->data_bytes || !m->pin_base) return t->data;
    size_t need = (t->data_bytes + 63u) & ~(size_t)63u;
    if (m->pin_used + need > m->pin_cap) return t->data;
    void *p = m->pin_base + m->pin_used;
    memcpy(p, t->data, t->data_bytes);
    m->pin_used += need;
    return p;
}

static int bind_q(ds_model *m, const char *name, ds_qweight *out, cwick_error *err) {
    char nm[512];
    cwick_st_tensor t;
    snprintf(nm, sizeof(nm), "%s.weight", name);
    if (cwick_safetensors_get(&m->saf, nm, &t, err) != CWICK_OK) return -1;
    out->fp4 = (t.dtype == CWICK_ST_I8);
    out->rows = t.rows;
    out->cols = t.cols;
    out->w16 = NULL;

    /* A dense fp8 tensor is widened to bf16 once and then read from there, so
     * pinning the fp8 bytes as well would cost a copy nobody touches. Leave
     * it in the shard mapping (file-backed, reclaimed on demand) and spend the
     * resident budget on the bf16 half instead. */
    int widen = (t.dtype == CWICK_ST_F8_E4M3) && m->dense16 && !ds_exact_layout();
    if (widen) {
        size_t need = (size_t)(t.rows * t.cols) * sizeof(uint16_t);
        need = (need + 63u) & ~(size_t)63u;
        if (m->dense16_used + need <= m->dense16_cap) {
            uint16_t *dst = (uint16_t *)(m->dense16 + m->dense16_used);
            ds_fp8_to_bf16_row((const uint8_t *)t.data, dst, t.rows * t.cols);
            m->dense16_used += need;
            out->w = t.data;
            out->w16 = dst;
        } else {
            out->w = t.data;
        }
    } else {
        out->w = ds_pin_tensor(m, &t);
    }

    /* `X.weight` is paired with `X.scale`, not `X.weight.scale` */
    size_t n = strlen(name);
    if (n > 7 && strcmp(name + n - 7, ".weight") == 0)
        snprintf(nm, sizeof(nm), "%.*s.scale", (int)(n - 7), name);
    else
        snprintf(nm, sizeof(nm), "%s.scale", name);
    if (cwick_safetensors_get(&m->saf, nm, &t, err) != CWICK_OK) return -1;
    out->scale = ds_pin_tensor(m, &t);
    return 0;
}

static const uint16_t *bind_bf16(ds_model *m, const char *name, cwick_error *err) {
    cwick_st_tensor t;
    if (cwick_safetensors_get(&m->saf, name, &t, err) != CWICK_OK) return NULL;
    return (const uint16_t *)ds_pin_tensor(m, &t);
}

static const float *bind_f32(ds_model *m, const char *name, cwick_error *err) {
    cwick_st_tensor t;
    if (cwick_safetensors_get(&m->saf, name, &t, err) != CWICK_OK) return NULL;
    return (const float *)ds_pin_tensor(m, &t);
}

/* ------------------------------------------------------------------ */
/* open / close                                                         */
/* ------------------------------------------------------------------ */

int ds_model_probe(const char *root) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/config.json", root);
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    char buf[8192];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = 0;
    fclose(f);
    return strstr(buf, "deepseek_v41") != NULL;
}

/* Every tensor that is neither a routed expert nor an engram row table: read
 * in full on every token, so it gets a resident copy. Sized from the shard
 * headers, not guessed.
 *
 * The fp8 half is counted separately because it is stored as bf16 (twice the
 * bytes) and its fp8 original is left in the mapping; `pin_out` is what the
 * pin arena has to hold and `w16_out` what the widened copy needs. */
static void ds_dense_sizes(const cwick_safetensors *saf, size_t *pin_out,
                           size_t *w16_out) {
    size_t pin = 0, w16 = 0;
    int widen = !ds_exact_layout();
    for (size_t i = 0; i < saf->tensor_count; i++) {
        const cwick_st_tensor *t = &saf->tensors[i];
        if (ds_pin_skip(t->name)) continue;
        if (t->dtype == CWICK_ST_F8_E4M3) {
            if (widen) {
                size_t b = (size_t)(t->rows * t->cols) * sizeof(uint16_t);
                w16 += (b + 63u) & ~(size_t)63u;
                continue;
            }
        }
        pin += (t->data_bytes + 63u) & ~(size_t)63u;
    }
    *pin_out = pin;
    *w16_out = w16;
}

static size_t ds_dense_bytes(const cwick_safetensors *saf) {
    size_t pin = 0, w16 = 0;
    ds_dense_sizes(saf, &pin, &w16);
    return pin + w16;
}

/* Bytes the KV-side state costs, sized exactly the way ds_model_open takes it
 * below (cmp_kv + idx_k over n_kv_source x max_seq_len, plus the sliding
 * window and the token cache). The pin budget has to subtract this: that
 * memory is allocated for the whole --context window up front and memset, so
 * at --context 1000000 it is ~9.5 GiB of dirty anonymous pages that a flat
 * "few GiB" reserve simply did not include. */
static size_t ds_kv_bytes(const ds_config *c) {
    int64_t src = c->n_kv_source > 0 ? c->n_kv_source : 0;
    int64_t n   = c->max_seq_len  > 0 ? c->max_seq_len  : 0;
    int64_t hd  = c->head_dim, ihd = c->index_head_dim;
    return (size_t)src * (size_t)n * (size_t)(hd + ihd) * sizeof(float)
         + (size_t)c->n_layers * (size_t)c->window_size * (size_t)hd *
               sizeof(float)
         + (size_t)n * sizeof(int64_t);
}

/* Budget for the expert pin cache.
 *
 * goal.md §3 asks the process to hold ~80% of host RAM, with the rest going
 * to SSD streaming. So the plan is: 80% of RAM, minus the resident dense copy
 * (it is a copy of 9 GiB of mapped weights, not optional), minus everything
 * else that lives in the footprint but is not a pinned expert. Whatever is
 * left is expert pins. The old rule -- host/4 -- left 60 GiB of a 128 GiB
 * machine idle while 40% of the expert reads went to the SSD.
 *
 * What changed: the reserve used to be a flat 5 GiB standing in for "the KV
 * caches, the contribution tables and the allocator". Measured, the non-cache
 * part of the footprint is ~44 GiB on this box -- 9.5 GiB of KV state, ~18 GiB
 * of GPU-side arena and driver accounting, the rest scratch. Paying for only
 * 5 GiB of it put the steady-state footprint 23.7 GiB over the ceiling, and
 * since the pin cache is ordinary malloc'ed memory the overshoot went straight
 * to the compressor: same hit rate, 3x the step time. */
static size_t ds_expert_cache_budget(const ds_config *c, size_t dense_bytes,
                                     uint64_t *target_out) {
    const uint64_t GiB = 1024ull * 1024ull * 1024ull;
    uint64_t host = cwick_host_memory_bytes();
    /* The ceiling is 80% of host RAM whether or not the cache is overridden:
     * it is what the pin arena, the KV caches and the scratch all add up to,
     * and it is what the dashboard divides the footprint by. */
    uint64_t target = host ? host / 10 * 8 : 0;
    if (target_out) *target_out = target;

    const char *e = getenv("CWICK_EXPERT_CACHE_GIB");
    if (e && *e) {
        double gib = atof(e);
        if (gib <= 0.0) return 0;
        return (size_t)(gib * 1024.0 * 1024.0 * 1024.0);
    }
    if (host == 0) return (size_t)32 * GiB;

    uint64_t kv  = (uint64_t)ds_kv_bytes(c);
    uint64_t gpu = host / 8;                  /* Metal arenas, driver, views */
    if (gpu < 2 * GiB) gpu = 2 * GiB;
    uint64_t reserve = 4 * GiB + kv + gpu;    /* + scratch and allocator slack */
    uint64_t fixed = (uint64_t)dense_bytes + reserve;
    if (target < fixed + 4 * GiB) target = fixed + 4 * GiB;
    return (size_t)(target - fixed);
}

int ds_model_open(ds_model *m, const char *root, int max_seq_len, cwick_error *err) {
    memset(m, 0, sizeof(*m));
    { const char *t = getenv("DS_TRACE"); ds_trace_on = t && *t && *t != '0'; }
    ds_timing_init();
    { const char *t = getenv("DS_TRACE_MAXL");
      ds_trace_maxl = (t && *t) ? atoi(t) : 2;
      if (ds_trace_maxl < 0) ds_trace_maxl = 0;
      if (ds_trace_maxl > 4096) ds_trace_maxl = 4096; }
    if (ds_load_config(m, root, max_seq_len, err) != 0) return -1;

    if (cwick_safetensors_open(root, &m->saf, err) != CWICK_OK) return -1;
    m->loaded = 1;

    size_t dense_bytes = ds_dense_bytes(&m->saf);

    /* Expert pin cache, straight from the GLM path's implementation: it owns
     * the LRU, the ghost admission filter and the F_NOCACHE descriptors. A
     * failure here only costs speed, so it degrades to the mapped shards
     * rather than failing the load. */
    {
        cwick_stream_options so;
        memset(&so, 0, sizeof(so));
        so.arena_bytes = 32u << 20;
        so.nocache = true;
        uint64_t ceiling = 0;
        so.expert_cache_bytes = ds_expert_cache_budget(&m->cfg, dense_bytes,
                                                       &ceiling);
        /* Publish the memory-plan ceiling for /v1/metrics. Without this the
         * gauge stayed 0 on this backend and the web UI drew the footprint
         * against a zero denominator. The decode loop steers against the same
         * number, so it has to outlive this block. */
        cwick_metrics_set_memory_ceiling(ceiling);
        m->rss_target = ceiling;
        cwick_error serr;
        cwick_error_clear(&serr);
        if (cwick_stream_open(&m->saf, &so, &m->stream, &serr) != CWICK_OK) {
            /* Say so out loud. A silent fallback to the mapped shards costs
             * roughly half the decode rate (the pin cache is where the
             * expert-hit traffic comes from), and the only symptom is a
             * metrics line reading "0/0 hit" after the fact. */
            CWICK_INFO("init", "expert cache unavailable (%s); experts will be "
                       "read through the shard mapping",
                       serr.message[0] ? serr.message : "stream open failed");
            m->stream = NULL;
        } else {
            CWICK_INFO("init", "expert pin cache %.1f GiB (dense %.1f GiB + "
                       "KV %.1f GiB + GPU/scratch reserve; ceiling %.1f GiB = "
                       "80%% of host RAM)",
                       (double)so.expert_cache_bytes / (1024.0 * 1024.0 * 1024.0),
                       (double)dense_bytes / (1024.0 * 1024.0 * 1024.0),
                       (double)ds_kv_bytes(&m->cfg) / (1024.0 * 1024.0 * 1024.0),
                       (double)ceiling / (1024.0 * 1024.0 * 1024.0));
        }
    }

    ds_config *c = &m->cfg;
    int D = c->dim, H = c->n_heads, HD = c->head_dim;

    /* Size and take the resident arena before the first bind. The budget is
     * whatever the dense tensors actually weigh (their headers are already
     * parsed), so this is not a guess: every tensor that is not an expert or
     * an engram row table gets a private copy and the mapping is then only
     * used for the streamed 500 GB. */
    {
        size_t pin = 0, w16 = 0;
        ds_dense_sizes(&m->saf, &pin, &w16);
        if (pin) {
            m->pin_base = cwick_shmem_alloc(pin);
            if (m->pin_base) m->pin_cap = pin;
        }
        if (w16) {
            m->dense16 = cwick_shmem_alloc(w16);
            if (m->dense16) m->dense16_cap = w16;
            else CWICK_INFO("init", "no room for the bf16 copy of the dense "
                            "weights (%.1f GiB); they stay in fp8",
                            cwick_gib(w16));
        }
    }

    /* Prefetch arena for the decode path: one layer's eighteen matrices fit in
     * each of the two zones (6 experts x ~18.8 MiB), so the reads submitted for
     * a layer can complete while the layer's multiply-adds run. */
    if (m->stream) {
        size_t per_layer = (size_t)c->n_activated_experts * 3 *
                           ((size_t)c->dim * (size_t)c->moe_inter_dim / 2 +
                            (size_t)c->dim * (size_t)c->moe_inter_dim / 32);
        size_t want = per_layer * 2 + (32u << 20);
        cwick_error perr;
        cwick_error_clear(&perr);
        if (cwick_stream_prefetch_open(m->stream, want, &perr) != CWICK_OK)
            CWICK_INFO("init", "demand prefetch unavailable (%s); expert reads "
                       "stay in front of the multiply-adds",
                       perr.message[0] ? perr.message : "thread failed");
        else
            CWICK_INFO("init", "demand prefetch arena %.0f MiB (%d x 3 matrices "
                       "in flight)", (double)want / (1024.0 * 1024.0),
                       c->n_activated_experts);
    }

    m->embed = bind_bf16(m, "embed.weight", err);
    m->head = bind_bf16(m, "head.weight", err);
    m->final_norm = bind_bf16(m, "norm.weight", err);
    if (!m->embed || !m->head || !m->final_norm) {
        cwick_error_set(err, CWICK_ERR_IO, "checkpoint is missing embed/head/norm");
        return -1;
    }

    /* layer -> most recent kv source at or before it: that source publishes
     * both the compressed KV and the index keys every layer in between reads */
    m->src_slot = xmalloc(sizeof(int) * DS_MAX_LAYERS);
    {
        int cur = -1;
        for (int L = 0; L < c->n_layers; L++) {
            for (int s = 0; s < c->n_kv_source; s++)
                if (c->kv_source_layers[s] == L) cur = s;
            m->src_slot[L] = cur;
        }
    }

    for (int L = 0; L < c->n_layers; L++) {
        ds_layer *y = &m->layers[L];
        char p[256];
        y->layer_id = L;
        y->compress_ratio = c->compress_ratios[L];
        y->is_kv_source = 0;
        for (int s = 0; s < c->n_kv_source; s++)
            if (c->kv_source_layers[s] == L) y->is_kv_source = 1;
        y->is_index_source = 0;
        for (int s = 0; s < c->n_index_source; s++)
            if (c->index_source_layers[s] == L) y->is_index_source = 1;
        y->is_candidate_source = (L == c->candidate_source_layer);
        y->uses_candidates = (c->candidate_source_layer >= 0 && L > c->candidate_source_layer);
        y->engram_slot = -1;
        for (int e = 0; e < c->n_engram; e++)
            if (c->engram_layer_ids[e] == L) y->engram_slot = e;

        snprintf(p, sizeof(p), "layers.%d.attn.wq_a", L);
        if (bind_q(m, p, &y->wq_a, err)) return -1;
        snprintf(p, sizeof(p), "layers.%d.attn.wq_b", L);
        if (bind_q(m, p, &y->wq_b, err)) return -1;
        snprintf(p, sizeof(p), "layers.%d.attn.wkv", L);
        if (bind_q(m, p, &y->wkv, err)) return -1;
        snprintf(p, sizeof(p), "layers.%d.attn.wo_a", L);
        if (bind_q(m, p, &y->wo_a, err)) return -1;
        snprintf(p, sizeof(p), "layers.%d.attn.wo_b", L);
        if (bind_q(m, p, &y->wo_b, err)) return -1;

        snprintf(p, sizeof(p), "layers.%d.attn.q_norm.weight", L);
        y->q_norm = bind_bf16(m, p, err);
        snprintf(p, sizeof(p), "layers.%d.attn.kv_norm.weight", L);
        y->kv_norm = bind_bf16(m, p, err);
        snprintf(p, sizeof(p), "layers.%d.attn.attn_sink", L);
        y->attn_sink = bind_f32(m, p, err);
        snprintf(p, sizeof(p), "layers.%d.attn_norm.weight", L);
        y->attn_norm = bind_bf16(m, p, err);
        if (!y->q_norm || !y->kv_norm || !y->attn_sink || !y->attn_norm) return -1;

        snprintf(p, sizeof(p), "layers.%d.ffn.gate.weight", L);
        y->gate_w = bind_bf16(m, p, err);
        snprintf(p, sizeof(p), "layers.%d.ffn.gate.bias", L);
        y->gate_bias = bind_f32(m, p, err);
        snprintf(p, sizeof(p), "layers.%d.ffn_norm.weight", L);
        y->ffn_norm = bind_bf16(m, p, err);
        if (!y->gate_w || !y->gate_bias || !y->ffn_norm) return -1;

        snprintf(p, sizeof(p), "layers.%d.ffn.shared_experts.w1", L);
        if (bind_q(m, p, &y->shared_w1, err)) return -1;
        snprintf(p, sizeof(p), "layers.%d.ffn.shared_experts.w2", L);
        if (bind_q(m, p, &y->shared_w2, err)) return -1;
        snprintf(p, sizeof(p), "layers.%d.ffn.shared_experts.w3", L);
        if (bind_q(m, p, &y->shared_w3, err)) return -1;

        /* experts: keep the resolved tensors so a token's six lookups are
         * array reads instead of 92k binary searches */
        y->expert = xmalloc(sizeof(ds_expert) * (size_t)c->n_routed_experts * 3);
        for (int e = 0; e < c->n_routed_experts; e++) {
            cwick_st_tensor t;
            for (int k = 0; k < 3; k++) {
                const char *sfx = k == 0 ? "w1" : (k == 1 ? "w2" : "w3");
                snprintf(p, sizeof(p), "layers.%d.ffn.experts.%d.%s.weight", L, e, sfx);
                if (cwick_safetensors_get(&m->saf, p, &t, err) != CWICK_OK) return -1;
                y->expert[(size_t)e * 3 + k].w = t;
                snprintf(p, sizeof(p), "layers.%d.ffn.experts.%d.%s.scale", L, e, sfx);
                if (cwick_safetensors_get(&m->saf, p, &t, err) != CWICK_OK) return -1;
                y->expert[(size_t)e * 3 + k].s = t;
                y->expert[(size_t)e * 3 + k].scale = t.data;
            }
        }

        snprintf(p, sizeof(p), "layers.%d.hc_attn_fn", L);
        y->hc_attn_fn = bind_f32(m, p, err);
        snprintf(p, sizeof(p), "layers.%d.hc_attn_base", L);
        y->hc_attn_base = bind_f32(m, p, err);
        snprintf(p, sizeof(p), "layers.%d.hc_attn_scale", L);
        y->hc_attn_scale = bind_f32(m, p, err);
        snprintf(p, sizeof(p), "layers.%d.hc_ffn_fn", L);
        y->hc_ffn_fn = bind_f32(m, p, err);
        snprintf(p, sizeof(p), "layers.%d.hc_ffn_base", L);
        y->hc_ffn_base = bind_f32(m, p, err);
        snprintf(p, sizeof(p), "layers.%d.hc_ffn_scale", L);
        y->hc_ffn_scale = bind_f32(m, p, err);
        if (!y->hc_attn_fn || !y->hc_attn_base || !y->hc_attn_scale ||
            !y->hc_ffn_fn || !y->hc_ffn_base || !y->hc_ffn_scale) return -1;

        if (y->is_kv_source) {
            snprintf(p, sizeof(p), "layers.%d.attn.compressor.wkv.weight", L);
            y->cmp_wkv = bind_bf16(m, p, err);
            snprintf(p, sizeof(p), "layers.%d.attn.compressor.norm.weight", L);
            y->cmp_norm = bind_bf16(m, p, err);
            if (!y->cmp_wkv || !y->cmp_norm) return -1;
            if (y->compress_ratio > 1) {
                snprintf(p, sizeof(p), "layers.%d.attn.compressor.wgate.weight", L);
                y->cmp_wgate = bind_bf16(m, p, err);
                if (!y->cmp_wgate) return -1;
            }
        }
        if (y->is_index_source) {
            snprintf(p, sizeof(p), "layers.%d.attn.indexer.wq_b", L);
            if (bind_q(m, p, &y->idx_wq_b, err)) return -1;
            snprintf(p, sizeof(p), "layers.%d.attn.indexer.weights_proj.weight", L);
            y->idx_weights_proj = bind_bf16(m, p, err);
            if (!y->idx_weights_proj) return -1;
            if (y->is_kv_source) { /* only a kv source owns index keys */
                snprintf(p, sizeof(p), "layers.%d.attn.indexer.wk.weight", L);
                y->idx_wk = bind_bf16(m, p, err);
                snprintf(p, sizeof(p), "layers.%d.attn.indexer.k_norm.weight", L);
                y->idx_k_norm = bind_bf16(m, p, err);
                if (!y->idx_wk || !y->idx_k_norm) return -1;
            }
        }
        if (y->engram_slot >= 0) {
            cwick_st_tensor t;
            snprintf(p, sizeof(p), "layers.%d.engram.wkv", L);
            if (bind_q(m, p, &y->engram_wkv, err)) return -1;
            snprintf(p, sizeof(p), "layers.%d.engram.embed.weight", L);
            if (cwick_safetensors_get(&m->saf, p, &t, err) != CWICK_OK) return -1;
            y->eng_embed = t.data;
            snprintf(p, sizeof(p), "layers.%d.engram.embed.scale", L);
            if (cwick_safetensors_get(&m->saf, p, &t, err) != CWICK_OK) return -1;
            y->eng_embed_scale = t.data;
            snprintf(p, sizeof(p), "layers.%d.engram.q_weight", L);
            y->eng_q_weight = bind_bf16(m, p, err);
            snprintf(p, sizeof(p), "layers.%d.engram.k_weight", L);
            y->eng_k_weight = bind_bf16(m, p, err);
            if (!y->eng_q_weight || !y->eng_k_weight) return -1;
        }

        /* two frequency tables in the whole model: YaRN off for the pure
         * sliding-window layers, YaRN on with the compressed theta elsewhere */
        if (y->compress_ratio > 0)
            ds_freqs_init(&m->freqs[L], c->rope_head_dim, c->compress_rope_theta,
                          c->rope_factor, c->original_seq_len, c->beta_fast, c->beta_slow);
        else
            ds_freqs_init(&m->freqs[L], c->rope_head_dim, c->rope_theta,
                          c->rope_factor, 0, c->beta_fast, c->beta_slow);
    }

    /* ---- caches ---- */
    int W = c->window_size;
    if (m->pin_cap)
        CWICK_INFO("init", "dense weights pinned: %.2f GiB resident (%.1f%% of "
                   "the arena used), engram and routed experts stay mapped",
                   (double)m->pin_used / (1024.0 * 1024.0 * 1024.0),
                   m->pin_cap ? 100.0 * (double)m->pin_used / (double)m->pin_cap
                              : 0.0);
    m->win_kv = xmalloc(sizeof(float) * (size_t)c->n_layers * W * HD);
    memset(m->win_kv, 0, sizeof(float) * (size_t)c->n_layers * W * HD);
    int max_cmp = c->max_seq_len; /* ratio 1 is the worst case */
    int max_ratio = 1;
    for (int s = 0; s < c->n_kv_source; s++)
        if (c->compress_ratios[c->kv_source_layers[s]] > max_ratio)
            max_ratio = c->compress_ratios[c->kv_source_layers[s]];
    /* Publish it: the compressor's partial-group buffers are sized by this,
     * and anything that walks them (the prefix snapshot, the reset) has to
     * agree with the allocation. Leaving it 0 made both silently skip two
     * arrays while the model still wrote into them. */
    c->max_ratio = max_ratio;
    m->cmp_kv = xmalloc(sizeof(float) * (size_t)c->n_kv_source * max_cmp * HD);
    memset(m->cmp_kv, 0, sizeof(float) * (size_t)c->n_kv_source * max_cmp * HD);
    m->cmp_state = xmalloc(sizeof(float) * (size_t)c->n_kv_source * max_ratio * HD);
    m->cmp_score = xmalloc(sizeof(float) * (size_t)c->n_kv_source * max_ratio * HD);
    m->idx_k = xmalloc(sizeof(float) * (size_t)c->n_kv_source * max_cmp * c->index_head_dim);
    memset(m->idx_k, 0, sizeof(float) * (size_t)c->n_kv_source * max_cmp * c->index_head_dim);
    m->tok_cache = xmalloc(sizeof(int64_t) * (size_t)c->max_seq_len);
    memset(m->tok_cache, 0, sizeof(int64_t) * (size_t)c->max_seq_len);

    /* ---- scratch ---- */
    int hc = c->hc_mult, I = c->moe_inter_dim;
    int wide = D > I ? D : I;                       /* widest quantized activation */
    int groups_wide = c->o_groups * c->o_lora_rank; /* wo_b's input */
    if (groups_wide > wide) wide = groups_wide;
    int sel_wide = c->index_topk > c->candidate_topk_blocks ? c->index_topk
                                                            : c->candidate_topk_blocks;
    m->h = xmalloc(sizeof(float) * (size_t)hc * D);
    m->res = xmalloc(sizeof(float) * (size_t)hc * D);
    m->sub = xmalloc(sizeof(float) * D);
    m->subl = xmalloc(sizeof(float) * D);
    m->tmp = xmalloc(sizeof(float) * D);
    m->q = xmalloc(sizeof(float) * (size_t)H * HD);
    m->qr = xmalloc(sizeof(float) * c->q_lora_rank);
    m->kv = xmalloc(sizeof(float) * HD);
    m->latent = xmalloc(sizeof(float) * HD);
    m->o = xmalloc(sizeof(float) * (size_t)H * HD);
    m->o_g = xmalloc(sizeof(float) * (size_t)groups_wide);
    m->attn_out = xmalloc(sizeof(float) * D);
    m->moe_x = xmalloc(sizeof(float) * D);
    m->moe_y = xmalloc(sizeof(float) * D);
    m->expert_gate = xmalloc(sizeof(float) * I);
    m->expert_up = xmalloc(sizeof(float) * I);
    m->pre_mix = xmalloc(sizeof(float) * hc);
    m->attn_pre = xmalloc(sizeof(float) * hc);
    m->attn_post = xmalloc(sizeof(float) * hc);
    m->ffn_pre = xmalloc(sizeof(float) * hc);
    m->ffn_post = xmalloc(sizeof(float) * hc);
    m->comb = xmalloc(sizeof(float) * hc * hc);
    m->q_idx = xmalloc(sizeof(float) * (size_t)c->index_n_heads * c->index_head_dim);
    m->ik = xmalloc(sizeof(float) * c->index_head_dim);
    m->iw = xmalloc(sizeof(float) * c->index_n_heads);
    m->router = xmalloc(sizeof(float) * (size_t)c->n_routed_experts);
    m->sel = xmalloc(sizeof(int) * (size_t)c->n_activated_experts);
    m->selw = xmalloc(sizeof(float) * (size_t)c->n_activated_experts);
    m->eng_w = xmalloc(sizeof(float) * (size_t)hc * D);
    m->act_q = xmalloc((size_t)wide);
    m->act_s = xmalloc((size_t)wide / 32 + 1);
    m->idxs = xmalloc(sizeof(int) * (size_t)(W + c->index_topk));
    m->scores = xmalloc(sizeof(float) * (size_t)max_cmp);
    m->att_scores = xmalloc(sizeof(float) * (size_t)c->n_heads *
                            (size_t)(W + c->index_topk));
    m->bsc = xmalloc(sizeof(float) * (size_t)(max_cmp / c->candidate_block_size + 2));
    m->heap = xmalloc(sizeof(float) * (size_t)sel_wide);
    m->heap_idx = xmalloc(sizeof(int) * (size_t)sel_wide);
    m->blk_out = xmalloc(sizeof(int) * (size_t)sel_wide);
    m->shared_topk = xmalloc(sizeof(int) * (size_t)DS_CHUNK * (W + c->index_topk));
    m->shared_cand = xmalloc((size_t)DS_CHUNK * (size_t)max_cmp);
    memset(m->shared_cand, 0, (size_t)DS_CHUNK * (size_t)max_cmp);

    int nhc = (c->engram_max_ngram_size - 1) * c->engram_n_heads;
    m->eng_hash = xmalloc(sizeof(int64_t) * (size_t)DS_CHUNK * c->n_engram * nhc);
    m->chunk_h = xmalloc(sizeof(float) * (size_t)DS_CHUNK * c->hc_mult * D);
    m->chunk_pre = xmalloc(sizeof(float) * (size_t)DS_CHUNK * c->hc_mult);
    /* chunked MoE scratch: the contribution table is the big one at
     * 512 x 6 x 5120 x 4 = 60 MiB, which is noise beside the 24 GiB the
     * weight mapping and the KV caches already cost, and it is what lets the
     * grouped pass reproduce the token-serial accumulation order exactly. */
    m->chunk_sub = xmalloc(sizeof(float) * (size_t)DS_CHUNK * D);
    m->chunk_subl = xmalloc(sizeof(float) * (size_t)DS_CHUNK * D);
    m->chunk_contrib = xmalloc(sizeof(float) * (size_t)DS_CHUNK *
                               (size_t)c->n_activated_experts * D);
    m->chunk_selw = xmalloc(sizeof(float) * (size_t)DS_CHUNK *
                            (size_t)c->n_activated_experts);
    m->chunk_sel = xmalloc(sizeof(int) * (size_t)DS_CHUNK *
                           (size_t)c->n_activated_experts);
    m->chunk_fpre = xmalloc(sizeof(float) * (size_t)DS_CHUNK * hc);
    m->chunk_fpost = xmalloc(sizeof(float) * (size_t)DS_CHUNK * hc);
    m->chunk_comb = xmalloc(sizeof(float) * (size_t)DS_CHUNK * hc * hc);
    m->chunk_aq = xmalloc((size_t)DS_CHUNK * D);
    m->chunk_as = xmalloc((size_t)DS_CHUNK * (D / 32 + 1));
    m->chunk_uni = xmalloc(sizeof(int) * (size_t)c->n_routed_experts);
    m->chunk_mark = xmalloc((size_t)c->n_routed_experts);
    m->h2 = xmalloc(sizeof(float) * (size_t)hc * D);
    m->eng_embed = xmalloc(sizeof(float) * (size_t)nhc * c->engram_head_dim);
    m->eng_kv = xmalloc(sizeof(float) * (size_t)D * (hc + 1));
    m->eng_key = xmalloc(sizeof(float) * (size_t)hc * D);
    m->eng_val = xmalloc(sizeof(float) * D);
    m->eng_rows = xmalloc(sizeof(int64_t) * (size_t)nhc);
    return 0;
}

void ds_model_close(ds_model *m) {
    if (!m) return;
    /* The stream holds descriptors into the shards, so it goes first. */
    if (m->stream) {
        cwick_stream_prefetch_close(m->stream);
        cwick_stream_close(m->stream);
        m->stream = NULL;
    }
    if (m->pin_base) { cwick_shmem_free(m->pin_base); m->pin_base = NULL; }
    /* The bf16 widening of the dense weights is a second block of the same
     * kind, sized by ds_dense_sizes(). Leaving it out of the list below was a
     * leak of ~11 GiB per load, which only showed up when the resident model
     * was replaced: the switch reported success while the outgoing model's
     * dense copy stayed resident for the rest of the process (see
     * docs/DECISIONS.md §24). */
    if (m->dense16) { cwick_shmem_free(m->dense16); m->dense16 = NULL; }
    if (m->loaded) cwick_safetensors_close(&m->saf);
    for (int L = 0; L < DS_MAX_LAYERS; L++) free(m->layers[L].expert);
    free(m->src_slot);
    free(m->win_kv); free(m->cmp_kv); free(m->cmp_state); free(m->cmp_score);
    free(m->idx_k); free(m->tok_cache);
    free(m->shared_topk); free(m->shared_cand); free(m->eng_hash); free(m->chunk_h);
    free(m->chunk_pre);
    free(m->chunk_sub); free(m->chunk_subl); free(m->chunk_contrib);
    free(m->chunk_selw); free(m->chunk_sel); free(m->chunk_fpre);
    free(m->chunk_fpost); free(m->chunk_comb); free(m->chunk_aq);
    free(m->chunk_as); free(m->chunk_uni); free(m->chunk_mark); free(m->h2);
    free(m->h); free(m->res); free(m->sub); free(m->subl); free(m->tmp);
    free(m->q); free(m->qr); free(m->kv); free(m->latent); free(m->o); free(m->o_g);
    free(m->attn_out);
    free(m->moe_x); free(m->moe_y); free(m->expert_gate); free(m->expert_up);
    free(m->pre_mix); free(m->attn_pre); free(m->attn_post);
    free(m->ffn_pre); free(m->ffn_post); free(m->comb);
    free(m->q_idx); free(m->ik); free(m->iw); free(m->router); free(m->sel);
    free(m->selw); free(m->eng_w);
    free(m->act_q); free(m->act_s); free(m->idxs); free(m->scores);
    free(m->att_scores);
    free(m->bsc); free(m->heap); free(m->heap_idx); free(m->blk_out);
    free(m->eng_embed); free(m->eng_kv); free(m->eng_key); free(m->eng_val); free(m->eng_rows);
    memset(m, 0, sizeof(*m));
}

/* ------------------------------------------------------------------ */
/* forward                                                              */
/*                                                                      */
/* Everything below runs one position at a time. That is slower than    */
/* batching a chunk through each projection, but it makes the sequential */
/* dependencies -- the ring buffer, the compressor's partial group, the  */
/* indexer's top-k, the values a source layer publishes for the layers   */
/* after it -- fall out in the same order the reference runs them, and   */
/* correctness comes first here.                                        */
/* ------------------------------------------------------------------ */

/* Route a quantized projection to whichever copy of the weight is resident:
 * the bf16 widening when the tensor got one, the fp8 bytes otherwise. Both
 * apply the same 32x32 block scales to the same block sums. */
static void ds_gemv_q(const ds_qweight *w, const uint8_t *aq, const uint8_t *as,
                      int64_t N, int64_t K, float *out) {
    if (w->w16) ds_qbf16_gemv(w->w16, w->scale, aq, as, N, K, out);
    else        ds_fp8_gemv(aq, as, w->w, w->scale, N, K, out);
}

/* Quantize x and run one quantized GEMM, rounding the result to bf16 the
 * way the reference's gemm does (its output dtype is the default, bf16). */
static void ds_lin_q(ds_model *m, const ds_qweight *w, const float *x, int N, float *out) {
    ds_quant_act(x, (int)w->cols, m->act_q, m->act_s);
    if (w->fp4) ds_fp4_gemv(m->act_q, m->act_s, w->w, w->scale, N, w->cols, out);
    else        ds_gemv_q(w, m->act_q, m->act_s, N, w->cols, out);
    for (int i = 0; i < N; i++) out[i] = ds_bf16_round(out[i]);
}

/* A plain bf16 GEMM (F.linear over a bf16 weight). `round` mirrors the
 * reference's bf16 output; the fp32 compressor projections leave it off. */
static void ds_lin_bf16(const uint16_t *w, const float *x, int N, int K,
                        float *out, int round) {
    ds_bf16_gemv(w, x, NULL, N, K, out);
    if (round) for (int i = 0; i < N; i++) out[i] = ds_bf16_round(out[i]);
}

static float ds_softplus(float x) {
    /* torch's default: identity above the threshold, log1p(exp) below */
    if (x > 20.0f) return x;
    return log1pf(expf(x));
}

/* ---- sparse attention over a window ring plus the compressed cache ---- */

/* One head's slice of the sparse attention. Heads are independent -- each
 * reads the shared window/compressed KVs and writes only its own 512 outputs
 * -- so handing them to the pool changes no arithmetic: every head still does
 * the same serial dot product over HD in the same order, and the same three
 * passes over the selected keys. */
typedef struct {
    const float *q, *win, *cmp, *sink;
    const int *idx;
    int total, W, HD, nidx, L, p;
    float scale;
    float *o;
    float *scores;   /* [n_heads][total] */
} ds_attn_arg;

static void ds_sparse_attn_heads(void *argp, int64_t h0, int64_t h1) {
    ds_attn_arg *a = (ds_attn_arg *)argp;
    const int HD = a->HD, W = a->W, total = a->total;
    const int *idx = a->idx;

    for (int64_t h = h0; h < h1; h++) {
        const float *qh = a->q + (size_t)h * HD;
        float *oh = a->o + (size_t)h * HD;
        float *sc = a->scores + (size_t)h * total;
        float mx = -1e30f;
        for (int j = 0; j < total; j++) {
            int t = idx[j];
            if (t < 0) { sc[j] = -INFINITY; continue; }
            const float *kt = t < W ? a->win + (size_t)t * HD
                                    : a->cmp + (size_t)(t - W) * HD;
            float d = 0.0f;
            for (int i = 0; i < HD; i++) d += qh[i] * kt[i];
            d *= a->scale;
            sc[j] = d;
            if (d > mx) mx = d;
        }
        float sum = 0.0f;
        for (int j = 0; j < total; j++) {
            if (idx[j] < 0) { sc[j] = 0.0f; continue; }
            /* -inf - (-1e30) is still -inf, so an all-invalid row stays zero */
            float e = expf(sc[j] - mx);
            sc[j] = e;
            sum += e;
        }
        sum += expf(a->sink[h] - mx);
        for (int i = 0; i < HD; i++) oh[i] = 0.0f;
        for (int j = 0; j < total; j++) {
            int t = idx[j];
            if (t < 0) continue;
            const float *kt = t < W ? a->win + (size_t)t * HD
                                    : a->cmp + (size_t)(t - W) * HD;
            float w = sc[j];
            for (int i = 0; i < HD; i++) oh[i] += w * kt[i];
        }
        float inv = 1.0f / sum;
        for (int i = 0; i < HD; i++) oh[i] *= inv;
        if (ds_trace_on && h == 0) {
            char tg[64];
            snprintf(tg, sizeof(tg), "PROBE_ATTW %d %d", a->L, a->p);
            ds_trace(tg, sc, total);
            snprintf(tg, sizeof(tg), "PROBE_ATTS %d %d", a->L, a->p);
            ds_trace(tg, &sum, 1);
        }
    }
}

static void ds_sparse_attn(ds_model *m, int L, int p, const float *sink, int src,
                           int nidx, float scale, float *o) {
    ds_config *c = &m->cfg;
    int H = c->n_heads, HD = c->head_dim, W = c->window_size;

    ds_attn_arg a;
    a.q = m->q;
    a.win = m->win_kv + (size_t)L * W * HD;
    /* ratio 0 layers pass src = -1 and nidx = 0, so never touch cmp there */
    a.cmp = nidx > 0 ? m->cmp_kv + (size_t)src * c->max_seq_len * HD : NULL;
    a.sink = sink;
    a.idx = m->idxs;
    a.total = W + nidx;
    a.W = W;
    a.HD = HD;
    a.nidx = nidx;
    a.L = L;
    a.p = p;
    a.scale = scale;
    a.o = o;
    a.scores = m->att_scores;

    /* 64 heads is below the pool's default "not worth a broadcast" threshold
     * of 512 items, but each head here is ~0.1 ms of work, so this is a small
     * number of large jobs and goes out through the items entry point. */
    cwick_pool_init();
    cwick_pool_for_items(H, &a, ds_sparse_attn_heads);
}

/* ---- the compressor: `ratio` tokens pooled into one KV latent ---------- */

/* Writes the latent for the group that just completed (if any) into
 * cmp_kv[src], and the matching indexer key into idx_k[src]. `latent_out`
 * receives the pre-RoPE latent when one was produced, else stays untouched. */
static int ds_compress(ds_model *m, int L, const float *sub, int64_t pos,
                       float *latent_out, int *j_out, int64_t *latent_pos) {
    ds_config *c = &m->cfg;
    ds_layer *y = &m->layers[L];
    int HD = c->head_dim, D = c->dim, rd = c->rope_head_dim, ratio = y->compress_ratio;
    int src = m->src_slot[L];
    const ds_freqs *fr = &m->freqs[L];
    int j;
    int64_t lp;

    if (ratio == 1) {
        /* a plain projection: bf16 weights, bf16 output, no gate, no state */
        ds_lin_bf16(y->cmp_wkv, sub, HD, D, m->latent, 1);
        ds_rmsnorm_bf16w(m->latent, m->latent, y->cmp_norm, HD, c->rms_norm_eps);
        j = (int)pos;
        lp = pos;
    } else {
        /* fp32 pooling: the projections stay fp32, only the norm output is bf16 */
        ds_lin_bf16(y->cmp_wkv, sub, HD, D, m->kv, 0);
        ds_lin_bf16(y->cmp_wgate, sub, HD, D, m->tmp, 0);
        int slot = (int)(pos % ratio);
        memcpy(m->cmp_state + ((size_t)src * ratio + slot) * HD, m->kv, HD * sizeof(float));
        memcpy(m->cmp_score + ((size_t)src * ratio + slot) * HD, m->tmp, HD * sizeof(float));
        if ((pos + 1) % ratio != 0) return 0;      /* group still filling */
        j = (int)((pos + 1) / ratio - 1);
        lp = (int64_t)j * ratio;
        /* element-wise softmax over the ratio axis, then a weighted sum */
        for (int i = 0; i < HD; i++) {
            const float *sc = m->cmp_score + (size_t)src * ratio * HD + i;
            float mx = -INFINITY, w[DS_MAX_HC > 8 ? DS_MAX_HC : 8], s = 0.0f, acc = 0.0f;
            for (int r = 0; r < ratio; r++) if (sc[(size_t)r * HD] > mx) mx = sc[(size_t)r * HD];
            for (int r = 0; r < ratio; r++) {
                w[r] = expf(sc[(size_t)r * HD] - mx);
                s += w[r];
            }
            for (int r = 0; r < ratio; r++)
                acc += m->cmp_state[((size_t)src * ratio + r) * HD + i] * (w[r] / s);
            m->latent[i] = acc;
        }
        ds_rmsnorm_bf16w(m->latent, m->latent, y->cmp_norm, HD, c->rms_norm_eps);
    }

    if (ds_trace_layer(L)) {
        char tag[64];
        snprintf(tag, sizeof(tag), "LATENT %d %lld", L, (long long)lp);
        ds_trace(tag, m->latent, HD);
    }
    memcpy(latent_out, m->latent, HD * sizeof(float));
    *j_out = j;
    *latent_pos = lp;

    /* the indexer needs the latent BEFORE it is rotated and quantized */
    if (y->is_index_source && y->idx_wk) {
        int IH = c->index_head_dim;
        ds_lin_bf16(y->idx_wk, m->latent, IH, HD, m->ik, 1);
        ds_rmsnorm_bf16w(m->ik, m->ik, y->idx_k_norm, IH, c->rms_norm_eps);
        ds_rope(m->ik + IH - rd, fr, lp, 0);
        for (int i = 0; i < rd; i++) m->ik[IH - rd + i] = ds_bf16_round(m->ik[IH - rd + i]);
        ds_qdq_fp4_e8m0(m->ik, IH);
        memcpy(m->idx_k + ((size_t)src * c->max_seq_len + j) * IH, m->ik,
               (size_t)IH * sizeof(float));
    }

    /* rotate, quantize and publish the latent itself */
    ds_rope(m->latent + HD - rd, fr, lp, 0);
    for (int i = 0; i < rd; i++) m->latent[HD - rd + i] = ds_bf16_round(m->latent[HD - rd + i]);
    ds_qdq_fp4_e4m3(m->latent, HD, 16);
    memcpy(m->cmp_kv + ((size_t)src * c->max_seq_len + j) * HD, m->latent,
           (size_t)HD * sizeof(float));
    if (ds_trace_layer(L)) {
        char tag[64];
        snprintf(tag, sizeof(tag), "PROBE_CKV %d %d", L, j);
        ds_trace(tag, m->latent, HD);
    }
    return 1;
}

/* ---- the indexer: which compressed positions each query keeps ---------- */

static void ds_indexer(ds_model *m, int L, int p, const float *sub, int64_t pos,
                       const float *qr, int nvis) {
    ds_config *c = &m->cfg;
    ds_layer *y = &m->layers[L];
    int IH = c->index_head_dim, NH = c->index_n_heads, rd = c->rope_head_dim;
    int W = c->window_size, src = m->src_slot[L];
    const ds_freqs *fr = &m->freqs[L];

    /* query heads, from the same qr the attention used */
    ds_lin_q(m, &y->idx_wq_b, qr, NH * IH, m->q_idx);
    for (int h = 0; h < NH; h++) {
        float *t = m->q_idx + (size_t)h * IH + (IH - rd);
        ds_rope(t, fr, pos, 0);
        for (int i = 0; i < rd; i++) t[i] = ds_bf16_round(t[i]);
    }
    ds_qdq_fp4_e8m0(m->q_idx, NH * IH);

    /* per-head weights, folded with the softmax scale */
    ds_lin_bf16(y->idx_weights_proj, sub, NH, c->dim, m->iw, 1);
    float fscale = powf((float)IH, -0.5f) * powf((float)NH, -0.5f);
    for (int h = 0; h < NH; h++) m->iw[h] = ds_bf16_round(m->iw[h] * fscale);

    if (nvis <= 0) {
        m->shared_topk[(size_t)p * (W + c->index_topk) + W] = -1;
        return;
    }

    /* score = sum_h relu(q_h . k_t) * w_h */
    const float *kc = m->idx_k + (size_t)src * c->max_seq_len * IH;
    for (int t = 0; t < nvis; t++) {
        const float *kt = kc + (size_t)t * IH;
        float acc = 0.0f;
        for (int h = 0; h < NH; h++) {
            const float *qh = m->q_idx + (size_t)h * IH;
            float d = 0.0f;
            for (int i = 0; i < IH; i++) d += qh[i] * kt[i];
            d = ds_bf16_round(d);
            if (d > 0.0f) acc += d * m->iw[h];
        }
        m->scores[t] = acc;
    }

    /* level one: the candidate source pins down whole blocks for the layers
     * after it; the layers in between only ever see inside those blocks */
    if (y->is_candidate_source) {
        int bs = c->candidate_block_size;
        int nblk = (nvis + bs - 1) / bs;
        for (int b = 0; b < nblk; b++) {
            int hi = (b + 1) * bs; if (hi > nvis) hi = nvis;
            float mx = -INFINITY;
            for (int t = b * bs; t < hi; t++) if (m->scores[t] > mx) mx = m->scores[t];
            m->bsc[b] = mx;
        }
        int last = (nvis - 1) / bs;
        m->bsc[last] = INFINITY;              /* partly-filled block, always kept */
        int k = ds_topk_scores(m->bsc, nblk, c->candidate_topk_blocks, m->blk_out,
                               m->heap, m->heap_idx);
        uint8_t *cand = m->shared_cand + (size_t)p * c->max_seq_len;
        memset(cand, 0, (size_t)nvis);
        for (int i = 0; i < k; i++) {
            int b = m->blk_out[i];
            int hi = (b + 1) * bs; if (hi > nvis) hi = nvis;
            for (int t = b * bs; t < hi; t++) cand[t] = 1;
        }
    } else if (y->uses_candidates) {
        const uint8_t *cand = m->shared_cand + (size_t)p * c->max_seq_len;
        for (int t = 0; t < nvis; t++) if (!cand[t]) m->scores[t] = -INFINITY;
    }

    /* level two: the best `index_topk` positions, re-sorted into position order */
    int cnt = nvis < c->index_topk ? nvis : c->index_topk;
    int got = ds_topk_scores(m->scores, nvis, cnt, m->blk_out, m->heap, m->heap_idx);
    int *dst = m->shared_topk + (size_t)p * (W + c->index_topk) + W;
    for (int i = 0; i < got; i++) dst[i] = m->blk_out[i] + W;
    for (int i = got; i < cnt; i++) dst[i] = -1;
    if (ds_trace_layer(L)) {
        char tag[64];
        snprintf(tag, sizeof(tag), "IDXTOP %d %d", L, p);
        for (int i = 0; i < cnt; i++)
            printf("%s %d", i ? "" : tag, dst[i]);
        printf("\n");
        fflush(stdout);
    }
}

/* ---- one attention sublayer ------------------------------------------- */

static void ds_attention(ds_model *m, int L, int p, const float *sub, int64_t pos,
                         float *qr) {
    ds_config *c = &m->cfg;
    ds_layer *y = &m->layers[L];
    int D = c->dim, H = c->n_heads, HD = c->head_dim, R = c->q_lora_rank;
    int rd = c->rope_head_dim, W = c->window_size, ratio = y->compress_ratio;
    const ds_freqs *fr = &m->freqs[L];
    int src = m->src_slot[L];
    uint64_t tq = ds_timing_on ? cwick_prof_ns() : 0;

    /* qr = q_norm(wq_a(x)) -- the indexer reuses it, so it is handed back */
    ds_lin_q(m, &y->wq_a, sub, R, qr);
    ds_rmsnorm_bf16w(qr, qr, y->q_norm, R, c->rms_norm_eps);
    if (ds_trace_on) {
        char tg[64];
        snprintf(tg, sizeof(tg), "PROBE_QR %d %d", L, p);
        ds_trace(tg, qr, R);
    }

    /* q, rope on the tail of every head */
    ds_lin_q(m, &y->wq_b, qr, H * HD, m->q);
    for (int h = 0; h < H; h++) {
        float *t = m->q + (size_t)h * HD + (HD - rd);
        ds_rope(t, fr, pos, 0);
        for (int i = 0; i < rd; i++) t[i] = ds_bf16_round(t[i]);
    }
    if (ds_trace_on) {
        char tg[64];
        snprintf(tg, sizeof(tg), "PROBE_Q %d %d", L, p);
        ds_trace(tg, m->q, H * HD);
    }

    /* window KV: one 512-dim latent per position, into this layer's ring */
    ds_lin_q(m, &y->wkv, sub, HD, m->kv);
    ds_rmsnorm_bf16w(m->kv, m->kv, y->kv_norm, HD, c->rms_norm_eps);
    ds_rope(m->kv + HD - rd, fr, pos, 0);
    for (int i = 0; i < rd; i++) m->kv[HD - rd + i] = ds_bf16_round(m->kv[HD - rd + i]);
    ds_qdq_fp8_e8m0(m->kv, HD);
    if (ds_trace_on) {
        char tg[64];
        snprintf(tg, sizeof(tg), "PROBE_WKV %d %d", L, p);
        ds_trace(tg, m->kv, HD);
    }
    memcpy(m->win_kv + ((size_t)L * W + (size_t)(pos % W)) * HD, m->kv,
           (size_t)HD * sizeof(float));

    /* window slots for this query: the W positions ending at pos, oldest
     * first, with -1 for anything the sequence has not reached yet */
    int nwin = 0;
    {
        int64_t lo = pos - W + 1;
        if (lo < 0) lo = 0;
        for (int64_t pp = lo; pp <= pos; pp++) m->idxs[nwin++] = (int)(pp % W);
        while (nwin < W) m->idxs[nwin++] = -1;
    }

    /* compressed KV and the positions this query keeps from it */
    int ncmp = 0;
    if (ds_timing_on) ds_t_ns[DS_T_QKV] += cwick_prof_ns() - tq;
    uint64_t ti = ds_timing_on ? cwick_prof_ns() : 0;
    if (ratio > 0) {
        int nvis = (int)((pos + 1) / ratio);
        if (y->is_kv_source) {
            int jj; int64_t lp;
            ds_compress(m, L, sub, pos, m->latent, &jj, &lp);
        }
        if (y->is_index_source) ds_indexer(m, L, p, sub, pos, qr, nvis);
        ncmp = nvis < c->index_topk ? nvis : c->index_topk;
        const int *srct = m->shared_topk + (size_t)p * (W + c->index_topk) + W;
        for (int j = 0; j < ncmp; j++) m->idxs[W + j] = srct[j];
    }

    if (ds_trace_on) {
        int64_t tmp[W + 1024];
        for (int j = 0; j < W + ncmp; j++) tmp[j] = m->idxs[j];
        char tg[64];
        snprintf(tg, sizeof(tg), "PROBE_IDS %d %d", L, p);
        ds_trace_i(tg, tmp, W + ncmp);
    }
    if (ds_timing_on) ds_t_ns[DS_T_INDEX] += cwick_prof_ns() - ti;
    uint64_t ts = ds_timing_on ? cwick_prof_ns() : 0;
    ds_sparse_attn(m, L, p, y->attn_sink, src, ncmp,
                   powf((float)HD, -0.5f), m->o);
    if (ds_timing_on) ds_t_ns[DS_T_SPARSE] += cwick_prof_ns() - ts;
    if (ds_trace_on) {
        char tg[64];
        snprintf(tg, sizeof(tg), "PROBE_SINK %d %d", L, p);
        ds_trace(tg, y->attn_sink, H);
    }

    /* sparse_attn hands back a bf16 tensor, so every head is rounded here,
     * before the inverse rotation reads it back in fp32 */
    for (int i = 0; i < H * HD; i++) m->o[i] = ds_bf16_round(m->o[i]);
    if (ds_trace_on) {
        char tg[64];
        snprintf(tg, sizeof(tg), "PROBE_OV %d %d", L, p);
        ds_trace(tg, m->o, H * HD);
    }

    /* undo the query rotation so the output projection sees one shared form */
    for (int h = 0; h < H; h++) {
        float *t = m->o + (size_t)h * HD + (HD - rd);
        ds_rope(t, fr, pos, 1);
        for (int i = 0; i < rd; i++) t[i] = ds_bf16_round(t[i]);
    }
    if (ds_trace_on) {
        char tg[64];
        snprintf(tg, sizeof(tg), "PROBE_O %d %d", L, p);
        ds_trace(tg, m->o, H * HD);
    }

    /* wo_a is block-diagonal over groups: einsum("bsgd,grd->bsgr"), not Linear */
    uint64_t tw = ds_timing_on ? cwick_prof_ns() : 0;
    int G = c->o_groups, OR = c->o_lora_rank, per = H * HD / G;
    for (int g = 0; g < G; g++) {
        const float *og = m->o + (size_t)g * per;
        ds_quant_act(og, per, m->act_q, m->act_s);
        if (y->wo_a.w16)
            ds_qbf16_gemv(y->wo_a.w16 + (size_t)g * OR * per,
                          y->wo_a.scale + (size_t)(g * OR / 32) * (per / 32),
                          m->act_q, m->act_s, OR, per,
                          m->o_g + (size_t)g * OR);
        else
            ds_fp8_gemv(m->act_q, m->act_s,
                        y->wo_a.w + (size_t)g * OR * per,
                        y->wo_a.scale + (size_t)(g * OR / 32) * (per / 32),
                        OR, per, m->o_g + (size_t)g * OR);
        for (int i = 0; i < OR; i++) m->o_g[(size_t)g * OR + i] =
            ds_bf16_round(m->o_g[(size_t)g * OR + i]);
    }
    if (ds_trace_on) {
        char tg[64];
        snprintf(tg, sizeof(tg), "PROBE_OG %d %d", L, p);
        ds_trace(tg, m->o_g, G * OR);
    }
    ds_lin_q(m, &y->wo_b, m->o_g, D, m->attn_out);
    if (ds_timing_on) ds_t_ns[DS_T_WO] += cwick_prof_ns() - tw;
}

/* ---- one MoE sublayer ------------------------------------------------- */

/* Router for a single position: sqrt(softplus(x/temp)), top-k on
 * score+bias, weights from the unbiased scores, renormalized then scaled.
 * Writes the K selected expert ids and their weights into `sel`/`selw`, which
 * lets the chunked path keep one set per position. */
static void ds_route_token(ds_model *m, int L, const float *sub, int *sel,
                           float *selw) {
    ds_config *c = &m->cfg;
    ds_layer *y = &m->layers[L];
    int D = c->dim, K = c->n_activated_experts;

    ds_bf16_gemv(y->gate_w, sub, NULL, c->n_routed_experts, D, m->router);
    for (int e = 0; e < c->n_routed_experts; e++)
        m->router[e] = sqrtf(ds_softplus(m->router[e] / c->gate_temp));

    for (int k = 0; k < K; k++) {
        int best = -1;
        float bv = -INFINITY;
        for (int e = 0; e < c->n_routed_experts; e++) {
            if (m->router[e] == -INFINITY) continue;
            float v = m->router[e] + y->gate_bias[e];
            if (v > bv) { bv = v; best = e; }
        }
        sel[k] = best;
        selw[k] = best >= 0 ? m->router[best] : 0.0f;
        if (best >= 0) m->router[best] = -INFINITY;   /* consumed */
    }
    if (c->norm_topk_prob && K > 1) {
        float s = 0.0f;
        for (int k = 0; k < K; k++) s += selw[k];
        /* a real division, not a multiply by the reciprocal: the two differ
         * by an ulp on some weights and that is enough to move a few
         * elements across a bf16 rounding boundary */
        for (int k = 0; k < K; k++) selw[k] = selw[k] / (s + 1e-20f);
    }
    for (int k = 0; k < K; k++) selw[k] *= c->route_scale;
}

/* Resolving one expert's matrices happens through the prefetch arena now
 * (ds_expert_prefetch + ds_expert_take below), because a single read at a
 * time is latency-bound and the chunked MoE has a whole batch's worth of
 * reads it can put in flight before it needs the first one. The pin cache is
 * consulted on the same path, so a hit still costs no read at all. */

/* ------------------------------------------------------------------ */
/* demand prefetch                                                      */
/*                                                                      */
/* A decode step reads ~4.5 GB of expert weights and cannot compute its   */
/* way out of it, so the only way to go faster is to stop waiting: the    */
/* routing for a layer is known before any of its experts is touched, so  */
/* all eighteen reads are submitted up front and the multiply-add of       */
/* expert k runs while experts k+1.. are still in flight.                 */
/*                                                                      */
/* This is not speculation -- every byte submitted here is consumed in     */
/* the same layer -- so unlike a next-layer guess it cannot waste          */
/* bandwidth. The reads land in the prefetch arena (no cache lock held     */
/* across the pread), and each one is pinned into the expert cache after   */
/* it has been used, which keeps the hit rate of the cached path.          */
/* ------------------------------------------------------------------ */

/* Submit every matrix of `e` that is not already pinned. */
static void ds_expert_prefetch(ds_model *m, int L, int e) {
    if (!m->stream) return;
    ds_layer *y = &m->layers[L];
    cwick_error ce;
    cwick_error_clear(&ce);
    for (int mat = 0; mat < 3; mat++) {
        if (cwick_stream_expert_resident(m->stream, L, e, mat)) continue;
        ds_expert *w = &y->expert[(size_t)e * 3 + mat];
        size_t shard = 0, shard_s = 0;
        uint64_t off = 0, off_s = 0;
        if (cwick_stream_locate(m->stream, &w->w, &shard, &off, &ce) != CWICK_OK)
            continue;
        if (cwick_stream_locate(m->stream, &w->s, &shard_s, &off_s, &ce) !=
            CWICK_OK)
            continue;
        cwick_stream_prefetch_submit(m->stream, L, e, mat, w->w.data_bytes,
                                     w->s.data_bytes, shard, off, off_s);
    }
}

/* Consume what ds_expert_prefetch submitted for `e`, in matrix order
 * 0=gate, 1=down, 2=up. `origin[mat]` is 2 for a pinned hit (use in place),
 * 1 for an arena slot (usable now, pin it after), 0 for the mapped shard. */
static void ds_expert_take(ds_model *m, int L, int e, const uint8_t *p[3],
                           const uint8_t *s[3], int origin[3]) {
    ds_layer *y = &m->layers[L];
    for (int mat = 0; mat < 3; mat++) {
        ds_expert *w = &y->expert[(size_t)e * 3 + mat];
        p[mat] = (const uint8_t *)w->w.data;
        s[mat] = w->scale;
        origin[mat] = 0;
        if (!m->stream) continue;
        size_t bytes = 0;
        const void *b = cwick_stream_expert_cached(m->stream, L, e, mat, &bytes);
        if (b) {
            origin[mat] = 2;
        } else {
            b = cwick_stream_prefetch_wait(m->stream, L, e, mat);
            if (b) origin[mat] = 1;
        }
        if (!b) continue;
        p[mat] = (const uint8_t *)b;
        s[mat] = p[mat] + w->w.data_bytes;
    }
}

/* Pin an arena slot into the expert cache once it has been used, so the next
 * token that routes here skips the read. The ghost filter still decides
 * whether the key deserves a slot.
 *
 * This is a second pass over bytes the read just fetched, and removing it
 * looked like free money: measured at 7.8% of a decode step (docs/
 * DEEPSEEK-PORT §10.11). It is not -- reading straight into a reserved cache
 * slot instead made the step 452.1 ms against 453.6, i.e. the copy was
 * already overlapping the SSD wait for the rest of the expert's matrices. So
 * the simpler read-into-arena-then-promote shape stays. */
static void ds_expert_pin(ds_model *m, int L, int e, int mat,
                          const uint8_t *blob) {
    if (!m->stream || !blob) return;
    if (!cwick_stream_expert_should_pin(m->stream, L, e, mat)) return;
    ds_expert *w = &m->layers[L].expert[(size_t)e * 3 + mat];
    cwick_stream_expert_put(m->stream, L, e, mat, blob,
                            w->w.data_bytes + w->s.data_bytes);
}

/* The SwiGLU half of one expert applied to one already-quantized activation:
 * gate/up, the swiglu with its clamps, requantize, down. The result lands in
 * `out` (fp32, the caller rounds). `k` is the routing slot, used only for the
 * trace tags. */
static void ds_expert_apply(ds_model *m, int L, int p, int k, const uint8_t *a,
                            const uint8_t *as, const uint8_t *p1,
                            const uint8_t *s1, const uint8_t *p3,
                            const uint8_t *s3, const uint8_t *p2,
                            const uint8_t *s2, float scale, float *out) {
    ds_config *c = &m->cfg;
    int D = c->dim, I = c->moe_inter_dim;
    float limit = c->swiglu_limit;

    ds_fp4_gemv(a, as, p1, s1, I, D, m->expert_gate);
    ds_fp4_gemv(a, as, p3, s3, I, D, m->expert_up);
    for (int i = 0; i < I; i++) {
        m->expert_gate[i] = ds_bf16_round(m->expert_gate[i]);
        m->expert_up[i] = ds_bf16_round(m->expert_up[i]);
    }
    if (ds_trace_on) {
        char tg[64];
        snprintf(tg, sizeof(tg), "PROBE_G%d %d %d", k, L, p);
        ds_trace(tg, m->expert_gate, I);
        snprintf(tg, sizeof(tg), "PROBE_U%d %d %d", k, L, p);
        ds_trace(tg, m->expert_up, I);
    }
    for (int i = 0; i < I; i++) {
        float g = m->expert_gate[i];
        float u = m->expert_up[i];
        if (limit > 0.0f) {
            if (u > limit) u = limit;
            if (u < -limit) u = -limit;
            if (g > limit) g = limit;
        }
        float hv = ds_silu(g) * u * scale;
        m->expert_gate[i] = ds_bf16_round(hv);
        if (ds_trace_on) m->moe_y[i] = hv;   /* unrounded, trace only */
    }
    if (ds_trace_on) {
        char tg[64];
        snprintf(tg, sizeof(tg), "PROBE_EXP%d %d %d", k, L, p);
        ds_trace(tg, m->moe_y, I);
        snprintf(tg, sizeof(tg), "PROBE_EXH%d %d %d", k, L, p);
        ds_trace(tg, m->expert_gate, I);
    }
    ds_quant_act(m->expert_gate, I, m->act_q, m->act_s);
    ds_fp4_gemv(m->act_q, m->act_s, p2, s2, D, I, out);
}

/* The shared expert, which runs for every position and reads no streamed
 * weight: it is FP8 and stays mapped. */
static void ds_shared_expert(ds_model *m, int L, const uint8_t *aq,
                             const uint8_t *as, float *out) {
    ds_config *c = &m->cfg;
    ds_layer *y = &m->layers[L];
    int D = c->dim, I = c->moe_inter_dim;
    float limit = c->swiglu_limit;
    ds_qweight *w1 = &y->shared_w1, *w2 = &y->shared_w2, *w3 = &y->shared_w3;

    ds_gemv_q(w1, aq, as, I, D, m->expert_gate);
    ds_gemv_q(w3, aq, as, I, D, m->expert_up);
    for (int i = 0; i < I; i++) {
        float g = ds_bf16_round(m->expert_gate[i]);
        float u = ds_bf16_round(m->expert_up[i]);
        if (limit > 0.0f) {
            if (u > limit) u = limit;
            if (u < -limit) u = -limit;
            if (g > limit) g = limit;
        }
        m->expert_gate[i] = ds_bf16_round(ds_silu(g) * u);
    }
    ds_quant_act(m->expert_gate, I, m->act_q, m->act_s);
    ds_gemv_q(w2, m->act_q, m->act_s, D, I, m->moe_y);
    for (int i = 0; i < D; i++) {
        m->moe_y[i] = ds_bf16_round(m->moe_y[i]);
        out[i] += m->moe_y[i];
    }
}

static void ds_moe(ds_model *m, int L, int p, const float *sub, float *out) {
    ds_config *c = &m->cfg;
    int D = c->dim, K = c->n_activated_experts;

    uint64_t tr = ds_timing_on ? cwick_prof_ns() : 0;
    ds_route_token(m, L, sub, m->sel, m->selw);
    if (ds_timing_on) ds_t_ns[DS_T_ROUTE] += cwick_prof_ns() - tr;
    if (ds_trace_on) {
        int64_t tmp[64];
        for (int k = 0; k < K; k++) tmp[k] = m->sel[k];
        char tg[64];
        snprintf(tg, sizeof(tg), "PROBE_SEL %d %d", L, p);
        ds_trace_i(tg, tmp, K);
        snprintf(tg, sizeof(tg), "PROBE_SELW %d %d", L, p);
        ds_trace(tg, m->selw, K);
    }

    memset(out, 0, (size_t)D * sizeof(float));
    ds_quant_act(sub, D, m->act_q, m->act_s);
    /* the activation of `sub` is the same for all six experts, so quantize
     * once into the per-position slot and let the loop reuse it */
    {
        uint8_t *aq = m->chunk_aq;
        uint8_t *as = m->chunk_as;
        memcpy(aq, m->act_q, (size_t)D);
        memcpy(as, m->act_s, (size_t)(D / 32));

        /* Submit all eighteen reads before computing any of them: the reads
         * overlap the multiply-adds instead of preceding them. */
        cwick_stream_prefetch_begin(m->stream);
        for (int k = 0; k < K; k++)
            if (m->sel[k] >= 0) ds_expert_prefetch(m, L, m->sel[k]);

        for (int k = 0; k < K; k++) {
            int e = m->sel[k];
            if (e < 0) continue;
            const uint8_t *mp[3], *ms[3];
            int origin[3];
            uint64_t tf = ds_timing_on ? cwick_prof_ns() : 0;
            ds_expert_take(m, L, e, mp, ms, origin);
            if (ds_timing_on) ds_t_ns[DS_T_FETCH] += cwick_prof_ns() - tf;
            uint64_t te = ds_timing_on ? cwick_prof_ns() : 0;
            ds_expert_apply(m, L, p, k, aq, as, mp[0], ms[0], mp[2], ms[2],
                            mp[1], ms[1], m->selw[k], m->moe_y);
            if (ds_timing_on) ds_t_ns[DS_T_EC] += cwick_prof_ns() - te;
            if (ds_trace_on) {
                char tg[64];
                snprintf(tg, sizeof(tg), "PROBE_EY%d %d %d", k, L, p);
                ds_trace(tg, m->moe_y, D);
            }
            for (int i = 0; i < D; i++)
                out[i] += ds_bf16_round(m->moe_y[i]);
            uint64_t tp = ds_timing_on ? cwick_prof_ns() : 0;
            for (int mat = 0; mat < 3; mat++)
                if (origin[mat] == 1) ds_expert_pin(m, L, e, mat, mp[mat]);
            if (ds_timing_on) ds_t_ns[DS_T_PIN] += cwick_prof_ns() - tp;
        }
    }
    if (ds_trace_on) {
        char tg[64];
        snprintf(tg, sizeof(tg), "PROBE_EX %d %d", L, p);
        ds_trace(tg, out, D);
    }
    uint64_t tsh = ds_timing_on ? cwick_prof_ns() : 0;
    ds_shared_expert(m, L, m->chunk_aq, m->chunk_as, out);
    if (ds_timing_on) ds_t_ns[DS_T_SHARED] += cwick_prof_ns() - tsh;
    if (ds_trace_on) {
        char tg[64];
        snprintf(tg, sizeof(tg), "PROBE_SHW %d %d", L, p);
        ds_trace(tg, m->moe_y, D);
    }
    for (int i = 0; i < D; i++) out[i] = ds_bf16_round(out[i]);
}

/* Chunked MoE: every position routes first, then the union of those routes is
 * walked once. Reading an expert's 18 MiB for one position is the whole cost
 * of the decode path; paying it once per *chunk* instead is what makes a long
 * prompt affordable (89 positions touch 2362 pairs but only 942 experts).
 *
 * Per-position arithmetic is untouched: each (position, slot) contribution is
 * produced by the same GEMV in the same order and stored, then summed in slot
 * order, so the grouped pass is bit-identical to the token-serial one. */
static void ds_moe_chunk(ds_model *m, int L, int n) {
    ds_config *c = &m->cfg;
    int D = c->dim, K = c->n_activated_experts, E = c->n_routed_experts;
    uint64_t t0 = ds_timing_on ? cwick_prof_ns() : 0;

    for (int p = 0; p < n; p++) {
        const float *sub = m->chunk_sub + (size_t)p * D;
        int *sel = m->chunk_sel + (size_t)p * K;
        float *selw = m->chunk_selw + (size_t)p * K;
        ds_route_token(m, L, sub, sel, selw);
        ds_quant_act(sub, D, m->chunk_aq + (size_t)p * D,
                     m->chunk_as + (size_t)p * (D / 32));
        if (ds_trace_on) {
            int64_t tmp[64];
            for (int k = 0; k < K; k++) tmp[k] = sel[k];
            char tg[64];
            snprintf(tg, sizeof(tg), "PROBE_SEL %d %d", L, p);
            ds_trace_i(tg, tmp, K);
            snprintf(tg, sizeof(tg), "PROBE_SELW %d %d", L, p);
            ds_trace(tg, selw, K);
        }
    }

    memset(m->chunk_mark, 0, (size_t)E);
    for (int p = 0; p < n; p++)
        for (int k = 0; k < K; k++) {
            int e = m->chunk_sel[(size_t)p * K + k];
            if (e >= 0) m->chunk_mark[e] = 1;
        }
    int nuni = 0;
    for (int e = 0; e < E; e++)
        if (m->chunk_mark[e]) m->chunk_uni[nuni++] = e;
    if (ds_timing_on) ds_t_ns[DS_T_ROUTE] += cwick_prof_ns() - t0;

    /* Walk the union in arena-sized batches instead of one expert at a time.
     *
     * One expert's eighteen megabytes read on its own sustains about 2.5 GB/s
     * here: the read is latency-bound, not bandwidth-bound (the same device
     * delivers ~9 GB/s with four to eight requests outstanding). A batch is
     * one prefetch zone wide -- six experts, eighteen matrices -- so all of
     * its reads are in flight before the first multiply-add starts, and the
     * remaining reads of the batch then complete underneath that arithmetic
     * instead of in front of it.
     *
     * The bytes and the arithmetic are unchanged: every (position, slot) is
     * still one GEMV into its own contribution, rounded and summed in slot
     * order afterwards. Batching only changes when the read happens. */
    const int batch = ds_pf_batch(c->n_activated_experts);   /* = one arena zone */
    for (int u = 0; u < nuni; u += batch) {
        int hi = u + batch < nuni ? u + batch : nuni;
        if (m->stream) cwick_stream_prefetch_begin(m->stream);
        for (int v = u; v < hi; v++) ds_expert_prefetch(m, L, m->chunk_uni[v]);

        for (int v = u; v < hi; v++) {
            int e = m->chunk_uni[v];
            const uint8_t *mp[3], *ms[3];
            int origin[3];
            uint64_t tf = ds_timing_on ? cwick_prof_ns() : 0;
            ds_expert_take(m, L, e, mp, ms, origin);
            if (ds_timing_on) ds_t_ns[DS_T_FETCH] += cwick_prof_ns() - tf;
            uint64_t te = ds_timing_on ? cwick_prof_ns() : 0;
            for (int p = 0; p < n; p++) {
                const int *sel = m->chunk_sel + (size_t)p * K;
                int slot = -1;
                for (int k = 0; k < K; k++)
                    if (sel[k] == e) { slot = k; break; }
                if (slot < 0) continue;
                float *cd = m->chunk_contrib +
                            ((size_t)p * K + (size_t)slot) * (size_t)D;
                ds_expert_apply(m, L, p, slot, m->chunk_aq + (size_t)p * D,
                                m->chunk_as + (size_t)p * (D / 32),
                                mp[0], ms[0], mp[2], ms[2], mp[1], ms[1],
                                m->chunk_selw[(size_t)p * K + slot], cd);
                for (int i = 0; i < D; i++) cd[i] = ds_bf16_round(cd[i]);
                if (ds_trace_on) {
                    char tg[64];
                    snprintf(tg, sizeof(tg), "PROBE_EY%d %d %d", slot, L, p);
                    ds_trace(tg, cd, D);
                }
            }
            if (ds_timing_on) ds_t_ns[DS_T_EC] += cwick_prof_ns() - te;
            if (ds_timing_on) ds_t_ns[DS_T_EC] += cwick_prof_ns() - te;
            /* Pin what came out of the arena, exactly as the decode path
             * does: the ghost filter still decides whether the key deserves a
             * slot, so an expert this chunk touches once is read and dropped
             * while one the next token will route to again stays resident.
             * Timed apart from the read: it is a second pass over the same
             * bytes (arena -> cache slot), and folding it into `fetch` is how
             * the report came to blame the disk for work the CPU was doing.
             * docs/DEEPSEEK-PORT §10.11 records why it is nonetheless not
             * worth removing. */
            uint64_t tp = ds_timing_on ? cwick_prof_ns() : 0;
            for (int mat = 0; mat < 3; mat++)
                if (origin[mat] == 1) ds_expert_pin(m, L, e, mat, mp[mat]);
            if (ds_timing_on) ds_t_ns[DS_T_PIN] += cwick_prof_ns() - tp;
        }
    }

    uint64_t ts = ds_timing_on ? cwick_prof_ns() : 0;
    for (int p = 0; p < n; p++) {
        float *out = m->chunk_subl + (size_t)p * D;
        memset(out, 0, (size_t)D * sizeof(float));
        /* slot order, so the fp32 additions happen in the same sequence the
         * token-serial loop used */
        for (int k = 0; k < K; k++) {
            int e = m->chunk_sel[(size_t)p * K + k];
            if (e < 0) continue;
            const float *cd = m->chunk_contrib +
                              ((size_t)p * K + (size_t)k) * (size_t)D;
            for (int i = 0; i < D; i++) out[i] += cd[i];
        }
        if (ds_trace_on) {
            char tg[64];
            snprintf(tg, sizeof(tg), "PROBE_EX %d %d", L, p);
            ds_trace(tg, out, D);
        }
        ds_shared_expert(m, L, m->chunk_aq + (size_t)p * D,
                         m->chunk_as + (size_t)p * (D / 32), out);
        if (ds_trace_on) {
            char tg[64];
            snprintf(tg, sizeof(tg), "PROBE_SHW %d %d", L, p);
            ds_trace(tg, m->moe_y, D);
        }
        for (int i = 0; i < D; i++) out[i] = ds_bf16_round(out[i]);
    }
    if (ds_timing_on) ds_t_ns[DS_T_SHARED] += cwick_prof_ns() - ts;
}

/* ---- engram: n-gram hash lookups written into the residual stream ------ */

/* Row ids for every position of the chunk, one set per engram layer. The
 * rolling hash is the (i+1)-gram after step i, so each n-gram size lands in
 * its own prime-sized bucket range. */
static void ds_engram_hash(ds_model *m, const int32_t *ids, int n, int64_t start_pos) {
    ds_config *c = &m->cfg;
    int ngram = c->engram_max_ngram_size, nh = c->engram_n_heads;
    int ncol = (ngram - 1) * nh;
    int32_t pad = ds_engram_pad_id;

    for (int p = 0; p < n; p++)
        m->tok_cache[start_pos + p] = ds_token_map[ids[p]];

    for (int p = 0; p < n; p++) {
        int64_t pos = start_pos + p;
        int64_t tk[8];
        for (int k = 0; k < ngram; k++) {
            int64_t sp = pos - k;
            /* look-back stops at the sequence start; text-only input has no
             * dead tokens, so nothing else can block an n-gram */
            tk[k] = sp < 0 ? pad : m->tok_cache[sp];
        }
        for (int e = 0; e < c->n_engram; e++) {
            int64_t rolling = tk[0] * ds_engram_mult[e][0];
            int col = 0;
            for (int i = 1; i < ngram; i++) {
                rolling ^= tk[i] * ds_engram_mult[e][i];
                for (int h = 0; h < nh; h++) {
                    int64_t hsh = rolling % ds_engram_primes[e][i - 1][h];
                    m->eng_hash[((size_t)p * c->n_engram + e) * ncol + col] =
                        hsh + ds_engram_offsets[e][col];
                    col++;
                }
            }
        }
    }
}

static void ds_engram_apply(ds_model *m, int L, int p) {
    ds_config *c = &m->cfg;
    ds_layer *y = &m->layers[L];
    int e = y->engram_slot;
    int D = c->dim, hc = c->hc_mult, EHD = c->engram_head_dim;
    int ngram = c->engram_max_ngram_size, nh = c->engram_n_heads;
    int ncol = (ngram - 1) * nh;
    float eps = c->rms_norm_eps;
    float *h = m->chunk_h + (size_t)p * hc * D;

    /* dequantize the rows this position's n-grams hashed to */
    const int64_t *rows = m->eng_hash + ((size_t)p * c->n_engram + e) * ncol;
    for (int col = 0; col < ncol; col++) {
        int64_t row = rows[col];
        if (row < 0) row = 0;
        const uint8_t *w = y->eng_embed + (size_t)row * EHD;
        const uint8_t *s = y->eng_embed_scale + (size_t)row * (EHD / 32);
        float *dst = m->eng_embed + (size_t)col * EHD;
        for (int b = 0; b < EHD; b += 32) {
            float sc = ds_e8m0_to_f32(s[b / 32]);
            for (int j = 0; j < 32; j++)
                dst[b + j] = ds_bf16_round(ds_e4m3_to_f32(w[b + j]) * sc);
        }
    }

    /* one projection gives every hc copy a key plus one shared value */
    ds_quant_act(m->eng_embed, ncol * EHD, m->act_q, m->act_s);
    ds_gemv_q(&y->engram_wkv, m->act_q, m->act_s, D * (hc + 1), ncol * EHD,
              m->eng_kv);
    for (int i = 0; i < D * (hc + 1); i++) m->eng_kv[i] = ds_bf16_round(m->eng_kv[i]);
    memcpy(m->eng_val, m->eng_kv + (size_t)hc * D, (size_t)D * sizeof(float));

    /* q_weight and k_weight only ever appear as a product */
    for (int j = 0; j < hc; j++)
        for (int i = 0; i < D; i++)
            m->eng_w[(size_t)j * D + i] = ds_bf16_to_f32(y->eng_q_weight[(size_t)j * D + i]) *
                                          ds_bf16_to_f32(y->eng_k_weight[(size_t)j * D + i]);

    float dscale = powf((float)D, -0.5f);
    for (int j = 0; j < hc; j++) {
        const float *hj = h + (size_t)j * D;
        const float *kj = m->eng_kv + (size_t)j * D;
        const float *wj = m->eng_w + (size_t)j * D;
        double sh = 0.0, sk = 0.0, dot = 0.0;
        for (int i = 0; i < D; i++) {
            sh += (double)hj[i] * hj[i];
            sk += (double)kj[i] * kj[i];
            dot += (double)hj[i] * wj[i] * kj[i];
        }
        float rstd = (float)(1.0 / sqrt(sh / D + eps)) * (float)(1.0 / sqrt(sk / D + eps));
        float g = (float)dot * rstd * dscale;
        /* signed sqrt before the sigmoid, matching the training kernel */
        float a = fabsf(g);
        if (a < 1e-6f) a = 1e-6f;
        float gate = 1.0f / (1.0f + expf(-copysignf(sqrtf(a), g)));
        for (int i = 0; i < D; i++)
            h[(size_t)j * D + i] = ds_bf16_round(hj[i] + gate * m->eng_val[i]);
    }
}

/* ---- one block: hc_mixes -> sublayer -> hc_post, twice -----------------
 *
 * The block is split around the MoE because the chunked path needs every
 * position's FFN input before it can group them (see ds_moe_chunk). The two
 * halves hand each other exactly the state the unsplit version kept in its
 * scratch buffers: the FFN mixes (pre/post/comb) and the sublayer input,
 * both per position. Nothing about the arithmetic changes. */

/* Attention half. Leaves: chunk_sub[p] = FFN input, chunk_fpre/fpost/comb[p]
 * = this block's FFN mixes, chunk_h[p] = residual after the attention. */
static void ds_block_attn(ds_model *m, int L, int p, int64_t pos,
                          const float *pre_mix) {
    ds_config *c = &m->cfg;
    ds_layer *y = &m->layers[L];
    int hc = c->hc_mult, D = c->dim;
    float *h = m->chunk_h + (size_t)p * hc * D;
    size_t hb = (size_t)hc * D * sizeof(float);

    memcpy(m->res, h, hb);                     /* the residual the sublayer keeps */

    ds_hc_mixes(h, y->hc_attn_fn, y->hc_attn_scale, y->hc_attn_base, hc, D,
                c->hc_sinkhorn_iters, c->hc_eps, m->attn_pre, m->attn_post, m->comb);
    ds_hc_pre(m->sub, h, pre_mix, hc, D);
    for (int i = 0; i < D; i++) m->sub[i] = ds_bf16_round(m->sub[i]);
    ds_rmsnorm_bf16w(m->sub, m->sub, y->attn_norm, D, c->rms_norm_eps);

    if (ds_trace_layer(L)) {
        char tag[64];
        snprintf(tag, sizeof(tag), "SUBIN %d %d", L, p);
        ds_trace(tag, m->sub, D);
    }
    ds_attention(m, L, p, m->sub, pos, m->qr);
    if (ds_trace_layer(L)) {
        char tag[64];
        snprintf(tag, sizeof(tag), "ATTNOUT %d %d", L, p);
        ds_trace(tag, m->attn_out, D);
    }
    ds_hc_post(h, m->attn_out, m->res, m->attn_post, m->comb, hc, D);
    for (int j = 0; j < hc; j++)
        for (int i = 0; i < D; i++)
            h[(size_t)j * D + i] = ds_bf16_round(h[(size_t)j * D + i]);

    ds_hc_mixes(h, y->hc_ffn_fn, y->hc_ffn_scale, y->hc_ffn_base, hc, D,
                c->hc_sinkhorn_iters, c->hc_eps, m->ffn_pre, m->ffn_post, m->comb);
    memcpy(m->chunk_fpre + (size_t)p * hc, m->ffn_pre, (size_t)hc * sizeof(float));
    memcpy(m->chunk_fpost + (size_t)p * hc, m->ffn_post, (size_t)hc * sizeof(float));
    memcpy(m->chunk_comb + (size_t)p * hc * hc, m->comb,
           (size_t)hc * hc * sizeof(float));
    ds_hc_pre(m->sub, h, m->attn_pre, hc, D);
    for (int i = 0; i < D; i++) m->sub[i] = ds_bf16_round(m->sub[i]);
    ds_rmsnorm_bf16w(m->sub, m->sub, y->ffn_norm, D, c->rms_norm_eps);

    if (ds_trace_layer(L)) {
        char tag[64];
        snprintf(tag, sizeof(tag), "FFNIN %d %d", L, p);
        ds_trace(tag, m->sub, D);
    }
    memcpy(m->chunk_sub + (size_t)p * D, m->sub, (size_t)D * sizeof(float));
}

/* FFN half: mix the MoE output back into the residual and hand this block's
 * ffn_pre to the next layer as its pre_mix. hc_post writes its destination
 * while still reading `res`, so the residual cannot alias the output. */
static void ds_block_ffn_tail(ds_model *m, int p) {
    ds_config *c = &m->cfg;
    int hc = c->hc_mult, D = c->dim;
    float *h = m->chunk_h + (size_t)p * hc * D;
    size_t hb = (size_t)hc * D * sizeof(float);

    memcpy(m->h2, h, hb);
    ds_hc_post(m->h, m->chunk_subl + (size_t)p * D, m->h2,
               m->chunk_fpost + (size_t)p * hc,
               m->chunk_comb + (size_t)p * hc * hc, hc, D);
    for (int j = 0; j < hc; j++)
        for (int i = 0; i < D; i++)
            m->h[(size_t)j * D + i] = ds_bf16_round(m->h[(size_t)j * D + i]);
    memcpy(h, m->h, hb);

    /* this layer's ffn_pre is the next layer's pre_mix */
    memcpy(m->chunk_pre + (size_t)p * hc, m->chunk_fpre + (size_t)p * hc,
           (size_t)hc * sizeof(float));
}

/* Token-serial block. Decode (n == 1) goes through ds_block_attn + ds_moe +
 * ds_block_ffn_tail inline in ds_model_forward; this wrapper exists for the
 * tools that drive one position at a time. */
static void __attribute__((unused))
ds_block(ds_model *m, int L, int p, int64_t pos, const float *pre_mix) {
    ds_config *c = &m->cfg;
    int D = c->dim;

    ds_block_attn(m, L, p, pos, pre_mix);
    ds_moe(m, L, p, m->chunk_sub + (size_t)p * D,
           m->chunk_subl + (size_t)p * D);
    if (ds_trace_layer(L)) {
        char tag[64];
        snprintf(tag, sizeof(tag), "FFNOUT %d %d", L, p);
        ds_trace(tag, m->chunk_subl + (size_t)p * D, D);
    }
    ds_block_ffn_tail(m, p);
}

/* ------------------------------------------------------------------ */
/* public entry points                                                  */
/* ------------------------------------------------------------------ */

/* Drop every cached KV/index state and rewind to position 0. The weights
 * stay mapped; only the per-sequence buffers are cleared, which is what a
 * new conversation needs. Stale cache contents would be harmless on their
 * own (every read is bounded by what this sequence has written), but the
 * compressor's partial-group state is read as a whole group, so it has to
 * go back to its "nothing written" shape. */
void ds_model_reset(ds_model *m) {
    ds_config *c = &m->cfg;
    int HD = c->head_dim, W = c->window_size;
    int max_cmp = c->max_seq_len;
    int max_ratio = c->max_ratio > 0 ? c->max_ratio : 1;
    if (m->win_kv)
        memset(m->win_kv, 0, sizeof(float) * (size_t)c->n_layers * W * HD);
    if (m->cmp_kv)
        memset(m->cmp_kv, 0,
               sizeof(float) * (size_t)c->n_kv_source * max_cmp * HD);
    if (m->idx_k)
        memset(m->idx_k, 0, sizeof(float) * (size_t)c->n_kv_source * max_cmp *
                                c->index_head_dim);
    if (m->cmp_state)
        memset(m->cmp_state, 0, sizeof(float) * (size_t)c->n_kv_source *
                                   max_ratio * HD);
    if (m->cmp_score)
        memset(m->cmp_score, 0, sizeof(float) * (size_t)c->n_kv_source *
                                   max_ratio * HD);
    m->pos = 0;
}

/* ------------------------------------------------------------------ */
/* prefix snapshots (see pfx.h)                                        */
/*                                                                      */
/* What crosses a token boundary here: the sliding-window KV of every   */
/* layer, the compressed KV and indexer keys of the positions built so  */
/* far, the compressor's partial pooling group, and the token ids the   */
/* engram hash looks back at. The residual stream and the per-token     */
/* scratch are intra-token and are rebuilt from the next embedding, so  */
/* they are deliberately absent (same argument as the mHC state in      */
/* pfx.h).                                                              */
/*                                                                      */
/* Restoring is a memcpy of exactly the state a full prefill would have */
/* left, which is why a replay is bit-identical to the prefill it       */
/* replaces rather than merely close.                                   */
/* ------------------------------------------------------------------ */

#define DS_SNAP_MAGIC 0x564b5344u   /* "DSKV" */
#define DS_SNAP_VER 1u

typedef struct {
    uint32_t magic, ver;
    int64_t pos, n_layers, window, head_dim, n_src, idx_head_dim, ratio;
} ds_snap_hdr;

/* Every array in the snapshot, in the order they are written. `pos` is the
 * number of tokens covered.
 *
 * cmp_kv and idx_k are [source][max_seq_len][dim], so their live prefix is a
 * per-source run of `pos` rows, not one contiguous block: copying
 * `n_kv_source * pos` rows from the base would carry source 0's rows
 * 0..(n_kv_source*pos-1) and leave every other source at zero. pack/apply walk
 * source by source for that reason. The byte count is unchanged either way,
 * which is why the bug was invisible until a replay was compared bit for bit
 * against the prefill it replaced. */
static size_t ds_snap_floats(const ds_config *c, int64_t pos) {
    size_t f = 0;
    f += (size_t)c->n_layers * (size_t)c->window_size * (size_t)c->head_dim;
    f += (size_t)c->n_kv_source * (size_t)pos * (size_t)c->head_dim;
    f += 2u * (size_t)c->n_kv_source * (size_t)c->max_ratio *
         (size_t)c->head_dim;
    f += (size_t)c->n_kv_source * (size_t)pos * (size_t)c->index_head_dim;
    return f;
}

size_t ds_state_bytes(const ds_model *m, int64_t pos) {
    if (!m) return 0;
    if (pos <= 0) return sizeof(ds_snap_hdr);
    if (pos > m->cfg.max_seq_len) pos = m->cfg.max_seq_len;
    return sizeof(ds_snap_hdr) + ds_snap_floats(&m->cfg, pos) * sizeof(float) +
           (size_t)pos * sizeof(int64_t);
}

int ds_state_pack(const ds_model *m, int64_t pos, void *buf, size_t cap) {
    if (!m || !buf) return -1;
    const ds_config *c = &m->cfg;
    if (pos < 0 || pos > c->max_seq_len) return -1;
    const size_t need = ds_state_bytes(m, pos);
    if (cap < need) return -1;

    ds_snap_hdr h;
    memset(&h, 0, sizeof(h));
    h.magic = DS_SNAP_MAGIC;
    h.ver = DS_SNAP_VER;
    h.pos = pos;
    h.n_layers = c->n_layers;
    h.window = c->window_size;
    h.head_dim = c->head_dim;
    h.n_src = c->n_kv_source;
    h.idx_head_dim = c->index_head_dim;
    h.ratio = c->max_ratio;

    uint8_t *p = (uint8_t *)buf;
    memcpy(p, &h, sizeof(h));
    p += sizeof(h);
    const size_t win = (size_t)c->n_layers * (size_t)c->window_size *
                       (size_t)c->head_dim;
    memcpy(p, m->win_kv, win * sizeof(float));
    p += win * sizeof(float);
    for (int s = 0; s < c->n_kv_source; s++) {
        memcpy(p, m->cmp_kv + ((size_t)s * (size_t)c->max_seq_len) *
                              (size_t)c->head_dim,
               (size_t)pos * (size_t)c->head_dim * sizeof(float));
        p += (size_t)pos * (size_t)c->head_dim * sizeof(float);
    }
    const size_t cst = (size_t)c->n_kv_source * (size_t)c->max_ratio *
                       (size_t)c->head_dim;
    memcpy(p, m->cmp_state, cst * sizeof(float));
    p += cst * sizeof(float);
    memcpy(p, m->cmp_score, cst * sizeof(float));
    p += cst * sizeof(float);
    for (int s = 0; s < c->n_kv_source; s++) {
        memcpy(p, m->idx_k + ((size_t)s * (size_t)c->max_seq_len) *
                             (size_t)c->index_head_dim,
               (size_t)pos * (size_t)c->index_head_dim * sizeof(float));
        p += (size_t)pos * (size_t)c->index_head_dim * sizeof(float);
    }
    memcpy(p, m->tok_cache, (size_t)pos * sizeof(int64_t));
    return 0;
}

int ds_state_apply(ds_model *m, const void *buf, size_t len, int64_t *pos_out) {
    if (!m || !buf || len < sizeof(ds_snap_hdr)) return -1;
    ds_snap_hdr h;
    memcpy(&h, buf, sizeof(h));
    const ds_config *c = &m->cfg;
    /* The snapshot is only meaningful for the geometry it was taken under.
     * A mismatch is a cache miss, not a crash: report it and let the caller
     * prefill in full. */
    if (h.magic != DS_SNAP_MAGIC || h.ver != DS_SNAP_VER) return -1;
    if (h.n_layers != c->n_layers || h.window != c->window_size ||
        h.head_dim != c->head_dim || h.n_src != c->n_kv_source ||
        h.idx_head_dim != c->index_head_dim || h.ratio != c->max_ratio)
        return -1;
    if (h.pos < 0 || h.pos > c->max_seq_len) return -1;
    if (len < ds_state_bytes(m, h.pos)) return -1;

    const int64_t pos = h.pos;
    const uint8_t *p = (const uint8_t *)buf + sizeof(h);
    const size_t win = (size_t)c->n_layers * (size_t)c->window_size *
                       (size_t)c->head_dim;
    memcpy(m->win_kv, p, win * sizeof(float));
    p += win * sizeof(float);
    for (int s = 0; s < c->n_kv_source; s++) {
        memcpy(m->cmp_kv + ((size_t)s * (size_t)c->max_seq_len) *
                           (size_t)c->head_dim,
               p, (size_t)pos * (size_t)c->head_dim * sizeof(float));
        p += (size_t)pos * (size_t)c->head_dim * sizeof(float);
    }
    const size_t cst = (size_t)c->n_kv_source * (size_t)c->max_ratio *
                       (size_t)c->head_dim;
    memcpy(m->cmp_state, p, cst * sizeof(float));
    p += cst * sizeof(float);
    memcpy(m->cmp_score, p, cst * sizeof(float));
    p += cst * sizeof(float);
    for (int s = 0; s < c->n_kv_source; s++) {
        memcpy(m->idx_k + ((size_t)s * (size_t)c->max_seq_len) *
                          (size_t)c->index_head_dim,
               p, (size_t)pos * (size_t)c->index_head_dim * sizeof(float));
        p += (size_t)pos * (size_t)c->index_head_dim * sizeof(float);
    }
    memcpy(m->tok_cache, p, (size_t)pos * sizeof(int64_t));

    /* Rows past `pos` keep whatever the previous request left there. They are
     * never read: the compressor writes sequentially from pos and the indexer
     * only scans the compressed rows pos implies. Zeroing them would cost a
     * gigabyte of memset per replay at a 128K context and buy nothing. */
    m->pos = pos;
    if (pos_out) *pos_out = pos;
    return 0;
}

/* ------------------------------------------------------------------ */
/* prefill progress                                                     */
/*                                                                      */
/* Without these the DeepSeek backend leaves /v1/metrics frozen at      */
/* prefill_layer 0 for the entire prompt, which reads as a hang on any  */
/* dashboard that divides by the layer count. The gauges are only       */
/* meaningful while a prefill is running, so `end` clears them and the  */
/* layer loop skips the write when no prefill is in flight (decode      */
/* calls the same forward pass with n == 1).                            */
/* ------------------------------------------------------------------ */

static _Atomic int64_t g_pf_total_tokens;
static _Atomic int64_t g_pf_done_tokens;
static _Atomic int g_pf_layer_count;

/* Wall clock of the prefill start and of the last progress line. A 512-token
 * chunk against 40 layers is minutes of work with nothing to show; the layer
 * loop below is the only place frequent enough to say "still moving". */
static uint64_t g_pf_t0_ns;
static uint64_t g_pf_last_ns;

static double ds_progress_min_ms(void) {
    const char *e = getenv("CWICK_PROGRESS_MIN_MS");
    if (e) { double v = atof(e); if (v > 0.0) return v; }
    return 5000.0;
}

/* One prefill progress line, throttled to CWICK_PROGRESS_MIN_MS. `done` is the
 * layer-folded estimate the metrics gauges use, so the percentage here and the
 * one on /metrics cannot disagree. */
static void ds_prefill_heartbeat(int64_t done, int64_t total, int layer,
                                 int nlayer) {
    uint64_t now = cwick_prof_ns();
    double min_ms = ds_progress_min_ms();
    bool first = (g_pf_last_ns == 0);
    if (!first && (double)(now - g_pf_last_ns) < min_ms * 1e6) return;
    g_pf_last_ns = now;

    double el = (double)(now - g_pf_t0_ns) / 1e9;    /* seconds since begin */
    /* The first layer fires microseconds after the start, so a rate taken
     * there is noise (it once printed 1091 tok/s). Report position only until
     * there is enough elapsed time to mean something. */
    if (first || el < 0.5) {
        CWICK_INFO("prefill", "%lld/%lld tokens (%.1f%%), layer %d/%d",
                   (long long)done, (long long)total,
                   total > 0 ? 100.0 * (double)done / (double)total : 0.0,
                   layer, nlayer);
        return;
    }
    double rate = (double)done / el;
    double eta = (rate > 0.0 && total > done) ? (double)(total - done) / rate : 0.0;
    CWICK_INFO("prefill",
               "%lld/%lld tokens (%.1f%%), layer %d/%d, %.2f tok/s, eta %.0fs",
               (long long)done, (long long)total,
               total > 0 ? 100.0 * (double)done / (double)total : 0.0,
               layer, nlayer, rate, eta);
}

void ds_prefill_progress_begin(int64_t total_tokens, int layers) {
    atomic_store_explicit(&g_pf_total_tokens, total_tokens, memory_order_relaxed);
    atomic_store_explicit(&g_pf_done_tokens, 0, memory_order_relaxed);
    atomic_store_explicit(&g_pf_layer_count, layers, memory_order_relaxed);
    g_pf_t0_ns = cwick_prof_ns();
    g_pf_last_ns = 0;    /* first layer of the first chunk always reports */
    cwick_metrics_set_prefill(0, total_tokens, 0, layers);
}

void ds_prefill_progress_done(int64_t done_tokens) {
    atomic_store_explicit(&g_pf_done_tokens, done_tokens, memory_order_relaxed);
}

void ds_prefill_progress_end(void) {
    atomic_store_explicit(&g_pf_total_tokens, 0, memory_order_relaxed);
    atomic_store_explicit(&g_pf_done_tokens, 0, memory_order_relaxed);
    g_pf_last_ns = 0;
    cwick_metrics_set_prefill(0, 0, 0, 0);
}

int ds_model_forward(ds_model *m, const int32_t *ids, int n, int64_t start_pos,
                     float *logits, cwick_error *err) {
    ds_config *c = &m->cfg;
    int hc = c->hc_mult, D = c->dim;
    uint64_t tw = ds_timing_on ? cwick_prof_ns() : 0;

    if (n <= 0 || n > DS_CHUNK) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "ds_model_forward: n=%d outside 1..%d", n, DS_CHUNK);
        return -1;
    }
    if (start_pos < 0 || start_pos + n > c->max_seq_len) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "ds_model_forward: positions %lld..%lld exceed max_seq_len %d",
                        (long long)start_pos, (long long)(start_pos + n - 1),
                        c->max_seq_len);
        return -1;
    }

    /* Every trace tag below carries the position as an index *within this
     * call* (that is what the layer loops iterate), so a trace of a prompt
     * split across several calls is ambiguous on its own. This line anchors
     * them: everything after it belongs to the positions starting here. */
    if (ds_trace_on) printf("FWDPOS %lld %d\n", (long long)start_pos, n);

    if (c->n_engram > 0) ds_engram_hash(m, ids, n, start_pos);
    if (ds_trace_on) {
        int ncol = (c->engram_max_ngram_size - 1) * c->engram_n_heads;
        for (int p = 0; p < n; p++)
            for (int e = 0; e < c->n_engram; e++) {
                char tag[64];
                snprintf(tag, sizeof(tag), "ENGHASH %d %d", p, e);
                ds_trace_i(tag, m->eng_hash + ((size_t)p * c->n_engram + e) * ncol, ncol);
            }
    }

    /* the residual stream starts as hc copies of the token embedding, and the
     * first block's pre_mix is the identity one-hot */
    for (int p = 0; p < n; p++) {
        float *hp = m->chunk_h + (size_t)p * hc * D;
        const uint16_t *e = m->embed + (size_t)ids[p] * D;
        for (int j = 0; j < hc; j++)
            for (int i = 0; i < D; i++)
                hp[(size_t)j * D + i] = ds_bf16_to_f32(e[i]);
        float *pm = m->chunk_pre + (size_t)p * hc;
        pm[0] = 1.0f;
        for (int j = 1; j < hc; j++) pm[j] = 0.0f;
    }

    int64_t pf_total = atomic_load_explicit(&g_pf_total_tokens,
                                            memory_order_relaxed);
    for (int L = 0; L < c->n_layers; L++) {
        if (pf_total > 0) {
            /* A chunk smaller than DS_CHUNK finishes in one go, so reporting
             * only "tokens completed" would leave done at 0 for the whole
             * prompt and a done/total bar would never move. Fold the layers
             * finished inside this chunk into the count: the number is then
             * "tokens whose last layer is behind us", monotone and bounded by
             * the chunk end. */
            int64_t base = atomic_load_explicit(&g_pf_done_tokens,
                                                memory_order_relaxed);
            int64_t within = (int64_t)((double)(L + 1) * (double)n /
                                       (double)c->n_layers);
            int nlayer = (int)atomic_load_explicit(&g_pf_layer_count,
                                                   memory_order_relaxed);
            cwick_metrics_set_prefill(base + within, pf_total, L + 1, nlayer);
            ds_prefill_heartbeat(base + within, pf_total, L + 1, nlayer);
        }
        if (m->layers[L].engram_slot >= 0)
            for (int p = 0; p < n; p++) ds_engram_apply(m, L, p);
        uint64_t ta = ds_timing_on ? cwick_prof_ns() : 0;
        uint64_t sub_before = ds_timing_on ? (ds_t_ns[DS_T_QKV] + ds_t_ns[DS_T_INDEX] +
                                              ds_t_ns[DS_T_SPARSE] + ds_t_ns[DS_T_WO])
                                          : 0;
        for (int p = 0; p < n; p++) {
            if (ds_trace_layer(L)) {
                char tag[64];
                snprintf(tag, sizeof(tag), "HIN %d %d", L, p);
                ds_trace(tag, m->chunk_h + ((size_t)p * hc + 0) * D, D);
                snprintf(tag, sizeof(tag), "HIN1 %d %d", L, p);
                ds_trace(tag, m->chunk_h + ((size_t)p * hc + 1) * D, D);
            }
            /* the previous layer's ffn_pre, so the oracle can recompute this
             * block from HIN alone without replaying the whole stack */
            if (ds_trace_layer(L)) {
                char tag[64];
                snprintf(tag, sizeof(tag), "PREMIX %d %d", L, p);
                ds_trace(tag, m->chunk_pre + (size_t)p * hc, hc);
            }
            ds_block_attn(m, L, p, start_pos + p, m->chunk_pre + (size_t)p * hc);
        }
        /* The attention bucket is what is left of the block pass once its four
         * sub-buckets are taken out; summing a gross timer with its own parts
         * is how the report once read 170% of the wall clock. */
        if (ds_timing_on) {
            uint64_t sub = ds_t_ns[DS_T_QKV] + ds_t_ns[DS_T_INDEX] +
                           ds_t_ns[DS_T_SPARSE] + ds_t_ns[DS_T_WO];
            uint64_t gross = cwick_prof_ns() - ta;
            uint64_t inner = sub - sub_before;
            ds_t_ns[DS_T_ATTN] += gross > inner ? gross - inner : 0;
        }
        if (n > 1)
            ds_moe_chunk(m, L, n);
        else
            ds_moe(m, L, 0, m->chunk_sub, m->chunk_subl);
        uint64_t tt = ds_timing_on ? cwick_prof_ns() : 0;
        for (int p = 0; p < n; p++) {
            if (ds_trace_layer(L)) {
                char tag[64];
                snprintf(tag, sizeof(tag), "FFNOUT %d %d", L, p);
                ds_trace(tag, m->chunk_subl + (size_t)p * D, D);
            }
            ds_block_ffn_tail(m, p);
        }
        if (ds_timing_on) ds_t_ns[DS_T_TAIL] += cwick_prof_ns() - tt;
    }
    /* collapse with the last layer's ffn_pre, norm, and project the final
     * position only -- that is all generation reads */
    {
        uint64_t th = ds_timing_on ? cwick_prof_ns() : 0;
        float *hp = m->chunk_h + (size_t)(n - 1) * hc * D;
        const float *pm = m->chunk_pre + (size_t)(n - 1) * hc;
        ds_hc_pre(m->sub, hp, pm, hc, D);
        for (int i = 0; i < D; i++) m->sub[i] = ds_bf16_round(m->sub[i]);
        ds_rmsnorm_bf16w(m->tmp, m->sub, m->final_norm, D, c->rms_norm_eps);
        ds_bf16_gemv(m->head, m->tmp, NULL, c->vocab_size, D, logits);
        if (ds_timing_on) ds_t_ns[DS_T_HEAD] += cwick_prof_ns() - th;
    }

    if (ds_timing_on) {
        uint64_t total = 0;
        for (int i = 0; i < DS_T_N; i++) total += ds_t_ns[i];
        char line[512];
        int off = 0;
        for (int i = 0; i < DS_T_N && off < (int)sizeof(line) - 48; i++)
            off += snprintf(line + off, sizeof(line) - (size_t)off, "%s %.0f%% ",
                            ds_t_name[i],
                            total ? 100.0 * (double)ds_t_ns[i] / (double)total
                                  : 0.0);
        /* The buckets only cover what is instrumented; `wall` is the truth the
         * decode line reports, so the difference is whatever still has no
         * bucket. The head projection sat at 40% of a step without appearing
         * in a single percentage until it got one. */
        double wall_ms = (double)(cwick_prof_ns() - tw) / 1e6;
        CWICK_INFO("ds", "forward n=%d %.1f ms wall, %.1f ms bucketed: %s", n,
                   wall_ms, (double)total / 1e6, line);
        memset(ds_t_ns, 0, sizeof(ds_t_ns));
    }
    return 0;
}
