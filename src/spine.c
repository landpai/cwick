/* spine.c — memory-mapped weight spine.
 *
 * The dense spine is a set of row-major tensors (torch linear layout
 * [out, in]) in a `resident/` tree, either raw bfloat16 or the derived
 * row-int8 + per-row fp16 scale representation. Experts are one canonical raw
 * file per layer/expert; the corpus is opened with a completeness check so a
 * missing expert fails closed (never-pruned rule).
 *
 * Two sources are supported:
 *   - a converted `resident/` tree (synthetic or converted checkpoints), and
 *   - the released HF safetensors checkpoint directly: `model.safetensors
 *     .index.json` + `model-*.safetensors` shards, resolved by name through
 *     the safetensors loader (see safetensors.c). The real GLM-5.3-Flash
 *     checkpoint is HF FP8 E4M3 with paired F32 per-128x128 scale_inv.
 *
 * Directory layout under the model root:
 *   config.json
 *   model.safetensors.index.json + model-*.safetensors   (real checkpoint)
 *   resident/tensors/<name>.bin         BF16 dense spine (source of truth)
 *   resident/experts/expert-<L>-<E>.bin canonical expert files
 *   resident-int8/tensors/<name>.i8     optional int8 data
 *   resident-int8/tensors/<name>.s      optional fp16 per-row scales
 */
#include "cwick.h"
#include "log.h"
#include "mem.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

static void *spine_maps_push(cwick_spine *spine, const void *base, size_t len) {
    size_t n = spine->map_count + 1;
    spine->maps = cwick_xrealloc(spine->maps, n * sizeof(void *));
    spine->map_lens = cwick_xrealloc(spine->map_lens, n * sizeof(size_t));
    spine->maps[spine->map_count] = (void *)(uintptr_t)base;
    spine->map_lens[spine->map_count] = len;
    spine->map_count = n;
    return spine->maps[n - 1];
}

static cwick_tensor *spine_tensor_push(cwick_spine *spine) {
    spine->tensors =
        cwick_xrealloc(spine->tensors, (spine->tensor_count + 1) * sizeof(cwick_tensor));
    cwick_tensor *t = &spine->tensors[spine->tensor_count++];
    memset(t, 0, sizeof(*t));
    t->rows = t->cols = -1;
    return t;
}

static char **spine_expert_push(cwick_spine *spine, const char *path) {
    spine->expert_files =
        cwick_xrealloc(spine->expert_files, (spine->expert_file_count + 1) * sizeof(char *));
    spine->expert_files[spine->expert_file_count] = cwick_strdup(path);
    return &spine->expert_files[spine->expert_file_count++];
}

int64_t cwick_spine_find(const cwick_spine *spine, const char *name) {
    for (size_t i = 0; i < spine->tensor_count; i++) {
        if (strcmp(spine->tensors[i].name, name) == 0) return (int64_t)i;
    }
    return CWICK_TENSOR_MISSING;
}

/* Build the expected tensor name for a slot. */
static void layer_name(char *out, size_t cap, int64_t layer, const char *suffix) {
    snprintf(out, cap, "layers.%lld.%s", layer, suffix);
}

/* Register one dense tensor in the converted-format mode. If the BF16 file is
 * missing this fails closed. */
static cwick_status add_tensor(cwick_spine *spine, const char *name, int64_t rows,
                              int64_t cols, const char *repr, cwick_error *err) {
    char dir[1024], path[1152], spath[1152];
    cwick_path_join(dir, sizeof(dir), spine->root, "resident/tensors");
    cwick_path_join(path, sizeof(path), dir, name);
    size_t data_bytes = (size_t)(rows * cols) * 2; /* BF16 */
    int kind = CWICK_WEIGHT_BF16;
    const void *scales = NULL;
    size_t scale_bytes = 0;
    const void *data = NULL;

    if (strcmp(repr, "int8") == 0) {
        cwick_path_join(dir, sizeof(dir), spine->root, "resident-int8/tensors");
        cwick_path_join(path, sizeof(path), dir, name);
        cwick_path_join(spath, sizeof(spath), dir, name);
        char data_path[1152], scale_path[1152];
        snprintf(data_path, sizeof(data_path), "%s.i8", path);
        snprintf(scale_path, sizeof(scale_path), "%s.s", spath);
        size_t dlen = 0, slen = 0;
        const void *d = cwick_mmap_file(data_path, &dlen);
        if (!d) {
            cwick_error_set(err, CWICK_ERR_IO,
                           "int8 spine missing %s (run setup/convert-spine-int8)", name);
            return CWICK_ERR_IO;
        }
        const void *s = cwick_mmap_file(scale_path, &slen);
        if (!s) {
            cwick_error_set(err, CWICK_ERR_IO, "int8 spine missing scales for %s", name);
            return CWICK_ERR_IO;
        }
        spine_maps_push(spine, d, dlen);
        spine_maps_push(spine, s, slen);
        data = d;
        scales = s;
        data_bytes = (size_t)(rows * cols);
        scale_bytes = (size_t)rows * 2;
        kind = CWICK_WEIGHT_ROW_I8_F16;
    } else {
        char binpath[1152];
        snprintf(binpath, sizeof(binpath), "%s.bin", path);
        size_t len = 0;
        const void *d = cwick_mmap_file(binpath, &len);
        if (!d) {
            cwick_error_set(err, CWICK_ERR_IO,
                           "BF16 spine missing %s (run setup)", name);
            return CWICK_ERR_IO;
        }
        if (len < data_bytes) {
            cwick_error_set(err, CWICK_ERR_IO,
                           "tensor %s truncated (%zu < %zu bytes)", name, len, data_bytes);
            return CWICK_ERR_IO;
        }
        spine_maps_push(spine, d, len);
        data = d;
        data_bytes = len;
    }

    cwick_tensor *t = spine_tensor_push(spine);
    t->rows = rows;
    t->cols = cols;
    t->kind = (cwick_weight_kind)kind;
    t->data = data;
    t->scales = scales;
    t->data_bytes = data_bytes;
    t->scale_bytes = scale_bytes;
    snprintf(t->name, sizeof(t->name), "%s", name);
    return CWICK_OK;
}

/* ---- safetensors mode -------------------------------------------------- */

/* Map a spine slot name to the released checkpoint tensor name. Returns false
 * when the slot has no counterpart in GLM-5.3 (legacy residual/anchor and
 * routed_expert_* projections do not exist there). */
static bool glm_name_for_slot(const char *slot, int64_t layer,
                              char *out, size_t cap) {
    if (strcmp(slot, "embed_tokens.weight") == 0) {
        snprintf(out, cap, "model.language_model.embed_tokens.weight");
        return true;
    }
    if (strcmp(slot, "lm_head.weight") == 0) {
        snprintf(out, cap, "lm_head.weight");
        return true;
    }
    if (strcmp(slot, "model.norm.weight") == 0) {
        snprintf(out, cap, "model.language_model.norm.weight");
        return true;
    }
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "layers.%lld.", layer);
    const char *rest = slot + strlen(prefix);
    if (strncmp(rest, "block_sparse_moe.", 17) == 0) {
        const char *r2 = rest + 17;
        if (strncmp(r2, "gate.weight", 11) == 0) {
            snprintf(out, cap, "model.language_model.layers.%lld.mlp.gate.weight", layer);
            return true;
        }
        if (strncmp(r2, "gate.e_score_correction_bias", 28) == 0) {
            snprintf(out, cap,
                     "model.language_model.layers.%lld.mlp.gate.e_score_correction_bias",
                     layer);
            return true;
        }
        if (strncmp(r2, "shared_experts.", 15) == 0) {
            snprintf(out, cap, "model.language_model.layers.%lld.mlp.shared_experts.%s",
                     layer, r2 + 15);
            return true;
        }
        /* routed_expert_down/up/norm projections do not exist in GLM-5.3:
         * experts map hidden -> intermediate -> hidden directly. */
        return false;
    }
    snprintf(out, cap, "model.language_model.layers.%lld.%s", layer, rest);
    return true;
}

/* Register one dense tensor from the safetensors source. `optional` slots
 * (legacy residual-anchor weights) are skipped when absent. */
static cwick_status add_tensor_saf(cwick_spine *spine, const char *slot,
                                  int64_t layer, bool optional, cwick_error *err) {
    char safname[192];
    if (!glm_name_for_slot(slot, layer, safname, sizeof(safname))) {
        if (optional) return CWICK_OK;
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                       "slot %s has no counterpart in this checkpoint", slot);
        return CWICK_ERR_CONTRACT;
    }
    cwick_st_tensor st;
    cwick_status r = cwick_safetensors_get(spine->saf, safname, &st, err);
    if (r != CWICK_OK) {
        if (optional) return CWICK_OK; /* legacy slot absent in the real model */
        return r;
    }
    cwick_tensor *t = spine_tensor_push(spine);
    t->rows = st.rows;
    t->cols = st.cols;
    t->data = st.data;
    t->data_bytes = st.data_bytes;
    switch (st.dtype) {
    case CWICK_ST_BF16:
        t->kind = CWICK_WEIGHT_BF16;
        break;
    case CWICK_ST_F32:
        t->kind = CWICK_WEIGHT_F32;
        break;
    case CWICK_ST_F8_E4M3:
        t->kind = CWICK_WEIGHT_FP8_E4M3;
        t->scales = st.scale_inv;
        t->scale_bytes = (size_t)(st.scale_inv_rows * st.scale_inv_cols) * 4;
        break;
    default:
        t->kind = CWICK_WEIGHT_F32;
        break;
    }
    snprintf(t->name, sizeof(t->name), "%s", slot);
    return CWICK_OK;
}

cwick_status cwick_spine_open(const char *root, const cwick_model_spec *spec,
                            const char *representation, cwick_spine *spine,
                            cwick_error *err) {
    cwick_error_clear(err);
    memset(spine, 0, sizeof(*spine));
    spine->root = cwick_strdup(root);

    char idxpath[1152];
    cwick_path_join(idxpath, sizeof(idxpath), root, "model.safetensors.index.json");
    const bool is_saf = cwick_file_exists(idxpath);

    const char *repr = representation && strcmp(representation, "int8") == 0 ? "int8" : "bf16";
    spine->have_int8 = strcmp(repr, "int8") == 0;

    if (is_saf) {
        /* --spine int8 is only implemented for the converted corpus
         * (resident-int8/tensors/<name>.i8 + .s). A safetensors checkpoint
         * keeps its own on-disk dtype, so the request cannot be honoured --
         * say so instead of silently running BF16 and letting the user
         * believe the resident spine got twice as small. */
        if (spine->have_int8)
            CWICK_INFO("init", "--spine int8 is ignored for a safetensors "
                              "checkpoint (it applies to the converted "
                              "resident/tensors corpus); running the "
                              "checkpoint's own BF16 weights");
        /* The flag was a request, not a fact: the corpus on disk keeps its
         * own dtype. Clear it so consumers (the /metrics `spine` label)
         * report what is actually resident rather than what was asked for. */
        spine->have_int8 = false;
        spine->is_safetensors = true;
        spine->saf = cwick_xmalloc(sizeof(cwick_safetensors));
        cwick_status st = cwick_safetensors_open(root, spine->saf, err);
        if (st != CWICK_OK) {
            cwick_spine_close(spine);
            return st;
        }

        /* ---- split the checkpoint before anything is pinned ------------
         * The expert cache budget depends on how much RAM the resident
         * spine will eat, so the resident/streamed split has to be known
         * *before* the streaming reader is opened (it used to be computed
         * afterwards, which meant the budget never subtracted it). */
        cwick_safetensors_split(spine->saf, &spine->resident_bytes,
                               &spine->streamed_bytes);

        /* Size of one routed expert matrix, used for the read headroom in
         * the budget below. Sampled from the first sparse layer. */
        size_t per_matrix_bytes = 0;
        int64_t first_sparse = -1;
        for (int64_t l = 0; l < spec->num_hidden_layers; l++) {
            if (!spec->mlp_dense[l]) { first_sparse = l; break; }
        }
        {
            char probe[192];
            cwick_expert_saf_name(spec, first_sparse >= 0 ? first_sparse : 0,
                                  0, "gate_proj", probe, sizeof(probe));
            cwick_st_tensor pt;
            cwick_error perr;
            cwick_error_clear(&perr);
            if (cwick_safetensors_get(spine->saf, probe, &pt, &perr) == CWICK_OK)
                per_matrix_bytes = (size_t)pt.data_bytes;
        }

        /* The streaming reader. Routed experts are 290 GiB of the 306 GiB
         * checkpoint; they are fetched through F_NOCACHE descriptors so a
         * decode step never turns the machine's RAM into a copy of the
         * expert bank. Resident tensors keep using the shared mapping. */
        {
            cwick_stream_options sopt;
            memset(&sopt, 0, sizeof(sopt));
            sopt.arena_bytes = 32u << 20;
            sopt.nocache = true;

            /* ---- memory budget: at most ‑‑mem-fraction of this host ----
             * Everything the kernel is allowed to keep resident has to fit
             * under one number, decided once from real hardware:
             *
             *   host * fraction
             *   - resident weight spine (touched by every token)
             *   - inference state (KV / KDA / anchors, see engine.c)
             *   - streaming + prefetch arenas
             *   - headroom: two routed layers of demand reads, so prefill
             *     can read the next layer while the current one computes
             *   = what is left to pin hot experts in RAM.
             *
             * Anything not pinned is re-read from SSD per decode step. */
            bool use_prefetch = cwick_env_true("CWICK_PREFETCH");
            size_t prefetch_arena = 0;
            if (use_prefetch) {
                /* two zones x one routed layer of top-k experts */
                size_t one_layer = (size_t)spec->num_experts_per_token * 3 *
                                   per_matrix_bytes + 2u * 1024u * 1024u;
                if (one_layer < (4u << 20)) one_layer = 4u << 20;
                prefetch_arena = 2 * one_layer;
            }
            /* headroom: what one decode step pulls through the arenas and
             * must not be double-booked against the pinned set. */
            uint64_t headroom = (uint64_t)prefetch_arena;
            if (per_matrix_bytes > 0) {
                uint64_t per_layer = (uint64_t)spec->num_experts_per_token * 3 *
                                     (uint64_t)per_matrix_bytes;
                headroom += per_layer * 2;   /* two-layer overlap reserve */
            }
            if (headroom < (256u << 20)) headroom = 256u << 20;

            cwick_mem_plan *plan = cwick_xmalloc(sizeof(*plan));
            cwick_status pst = cwick_mem_plan_build(cwick_host_memory_bytes(),
                                                  spine->resident_bytes,
                                                  cwick_memory_runtime_state(),
                                                  sopt.arena_bytes + prefetch_arena,
                                                  headroom,
                                                  spine->streamed_bytes,
                                                  plan, err);
            if (pst != CWICK_OK) {
                /* Fail closed: starting anyway would only mean swapping. */
                free(plan);
                cwick_spine_close(spine);
                return pst;
            }
            spine->mem_plan = plan;
            cwick_mem_plan_log(plan, cwick_log_enabled(CWICK_LOG_VERBOSE));

            if (plan->expert_cache_bytes > 0) {
                sopt.expert_cache_bytes = (size_t)plan->expert_cache_bytes;
                CWICK_VERB("stor", "pinned expert cache enabled: %.2f GiB "
                          "(~%zu expert matrices of %.1f MiB)",
                          cwick_gib(plan->expert_cache_bytes),
                          (size_t)(plan->expert_cache_bytes /
                                   (per_matrix_bytes ? per_matrix_bytes : (8u << 20))),
                          per_matrix_bytes
                              ? (double)per_matrix_bytes / (1024.0 * 1024.0)
                              : 8.0);
            } else {
                CWICK_INFO("stor", "no room for a pinned expert cache — every "
                          "routed expert matrix is re-read from SSD each step");
            }

            if (cwick_stream_open(spine->saf, &sopt, &spine->stream, err) != CWICK_OK) {
                /* A 306 GiB checkpoint without bounded demand reads would fall
                 * back to mmap and let the page cache grow without admission.
                 * Streaming is therefore part of the memory-safety contract. */
                cwick_spine_close(spine);
                return CWICK_ERR_IO;
            }
            /* Cross-layer prefetch reader (opt-in). The arena is split
             * into two zones so one layer's prefetch can be consumed while
             * the next layer's is read. */
            if (spine->stream && use_prefetch && prefetch_arena > 0) {
                if (cwick_stream_prefetch_open(spine->stream, prefetch_arena, err)
                    != CWICK_OK)
                    cwick_error_clear(err); /* optional: fall back to demand I/O */
            }
        }

        /* ---- global tensors ---- */
#define ADD_SAF(slot, layer, optional) \
    if ((st = add_tensor_saf(spine, slot, layer, optional, err)) != CWICK_OK) { \
        cwick_spine_close(spine); return st; \
    }
        ADD_SAF("embed_tokens.weight", 0, false);
        ADD_SAF("model.norm.weight", 0, false);
        ADD_SAF("lm_head.weight", 0, false);
        /* legacy output residual anchors are absent in GLM-5.3 */
        ADD_SAF("model.output_attn_res_norm.weight", 0, true);
        ADD_SAF("model.output_attn_res_proj.weight", 0, true);

        /* ---- per-layer dense spine ---- */
        for (int64_t l = 0; l < spec->num_hidden_layers; l++) {
            char name[128];
            cwick_layer_kind kind = spec->layers[l];

#define ADD_SAF2(suffix, optional) \
    layer_name(name, sizeof(name), l, suffix); \
    if ((st = add_tensor_saf(spine, name, l, optional, err)) != CWICK_OK) { \
        cwick_spine_close(spine); return st; \
    }
            ADD_SAF2("input_layernorm.weight", false);
            ADD_SAF2("post_attention_layernorm.weight", false);
            /* mHC (hyper-connection) mixing parameters. They are stored in
             * fp32/bf16 and are *not* FP8-quantized in the released
             * checkpoint (modules_to_not_convert lists hyper_connection).
             * hc_attn_* drives the pre-attention mix, hc_ffn_* the pre-MLP
             * mix; both are needed for every base layer. */
            if (spec->mhc) {
                ADD_SAF2("hc_attn_fn", false);
                ADD_SAF2("hc_attn_base", false);
                ADD_SAF2("hc_attn_scale", false);
                ADD_SAF2("hc_ffn_fn", false);
                ADD_SAF2("hc_ffn_base", false);
                ADD_SAF2("hc_ffn_scale", false);
            }
            /* legacy residual-anchor weights are absent in GLM-5.3 */
            ADD_SAF2("self_attention_res_norm.weight", true);
            ADD_SAF2("self_attention_res_proj.weight", true);
            ADD_SAF2("mlp_res_norm.weight", true);
            ADD_SAF2("mlp_res_proj.weight", true);

            if (kind == CWICK_LAYER_KDA) {
                ADD_SAF2("self_attn.A_log", false);
                ADD_SAF2("self_attn.dt_bias", false);
                ADD_SAF2("self_attn.q_conv1d.weight", false);
                ADD_SAF2("self_attn.k_conv1d.weight", false);
                ADD_SAF2("self_attn.v_conv1d.weight", false);
                ADD_SAF2("self_attn.o_norm.weight", false);
                ADD_SAF2("self_attn.q_proj.weight", false);
                ADD_SAF2("self_attn.k_proj.weight", false);
                ADD_SAF2("self_attn.v_proj.weight", false);
                /* GLM-5.3 splits the delta-rule gate into g_a + g_b; the
                 * legacy single g_proj slot has no counterpart. */
                ADD_SAF2("self_attn.g_a_proj.weight", false);
                ADD_SAF2("self_attn.g_b_proj.weight", false);
                ADD_SAF2("self_attn.f_a_proj.weight", false);
                ADD_SAF2("self_attn.f_b_proj.weight", false);
                ADD_SAF2("self_attn.b_proj.weight", false);
                ADD_SAF2("self_attn.o_proj.weight", false);
                /* legacy g_proj slot is absent */
                ADD_SAF2("self_attn.g_proj.weight", true);
            } else if (kind == CWICK_LAYER_DSA) {
                ADD_SAF2("self_attn.q_a_proj.weight", false);
                ADD_SAF2("self_attn.q_b_proj.weight", false);
                ADD_SAF2("self_attn.kv_a_proj_with_mqa.weight", false);
                ADD_SAF2("self_attn.kv_b_proj.weight", false);
                ADD_SAF2("self_attn.o_proj.weight", false);
                ADD_SAF2("self_attn.q_a_layernorm.weight", false);
                ADD_SAF2("self_attn.kv_a_layernorm.weight", false);
                /* DSA indexer weights (no legacy MLA g_proj / output gate) */
                ADD_SAF2("self_attn.indexer.wk.weight", false);
                ADD_SAF2("self_attn.indexer.wq_b.weight", false);
                ADD_SAF2("self_attn.indexer.weights_proj.weight", false);
                ADD_SAF2("self_attn.indexer.k_norm.weight", false);
                ADD_SAF2("self_attn.indexer.k_norm.bias", true);
                ADD_SAF2("self_attn.indexer.index_kpool_compress_ape", true);
                ADD_SAF2("self_attn.indexer.index_kpool_compress_gate", true);
            } else { /* CWICK_LAYER_MLA (legacy) */
                ADD_SAF2("self_attn.q_a_proj.weight", false);
                ADD_SAF2("self_attn.q_b_proj.weight", false);
                ADD_SAF2("self_attn.kv_a_proj_with_mqa.weight", false);
                ADD_SAF2("self_attn.kv_b_proj.weight", false);
                ADD_SAF2("self_attn.g_proj.weight", true);
                ADD_SAF2("self_attn.o_proj.weight", false);
                ADD_SAF2("self_attn.q_a_layernorm.weight", false);
                ADD_SAF2("self_attn.kv_a_layernorm.weight", false);
            }

            if (spec->mlp_dense[l]) {
                ADD_SAF2("mlp.gate_proj.weight", false);
                ADD_SAF2("mlp.up_proj.weight", false);
                ADD_SAF2("mlp.down_proj.weight", false);
            } else {
                ADD_SAF2("block_sparse_moe.gate.weight", false);
                ADD_SAF2("block_sparse_moe.gate.e_score_correction_bias", false);
                /* routed_expert_down/up/norm projections are absent in GLM-5.3 */
                ADD_SAF2("block_sparse_moe.routed_expert_down_proj.weight", true);
                ADD_SAF2("block_sparse_moe.routed_expert_up_proj.weight", true);
                ADD_SAF2("block_sparse_moe.routed_expert_norm.weight", true);
                ADD_SAF2("block_sparse_moe.shared_experts.gate_proj.weight", false);
                ADD_SAF2("block_sparse_moe.shared_experts.up_proj.weight", false);
                ADD_SAF2("block_sparse_moe.shared_experts.down_proj.weight", false);
            }
#undef ADD_SAF2
        }

        /* ---- the next-token-prediction (MTP) layer ----
         *
         * GLM-5.3-Flash ships num_nextn_predict_layers=1 as a *forty-sixth*
         * transformer block: `model.language_model.layers.45.*` with its own
         * 288-expert bank, its own MLA attention and its own DSA indexer. It
         * is not part of the depth-45 schedule -- that is a validated contract
         * and stays 45 -- so it hangs off num_nextn_predict_layers instead,
         * and every consumer that walks layers 0..44 keeps doing exactly what
         * it did before.
         *
         * What makes it an MTP block rather than a 46th decoder layer is the
         * three tensors at the top of the list: enorm/hnorm normalise the next
         * token's embedding and the previous token's hidden state, and
         * eh_proj (BF16 [H, 2H]) fuses them into the layer's input. The layer
         * itself is a DSA block with a sparse MLP, and shared_head.norm
         * re-normalises its output before the shared lm_head.
         *
         * Registering these does not change generation: nothing reads them
         * until the speculative path asks for them. They are registered here,
         * at load time, because the spine is the only place that knows how a
         * slot name maps onto this checkpoint's tensor names. */
        if (spec->num_nextn_predict_layers > 0) {
            const int64_t mtp = spec->num_hidden_layers;
            char name[128];
#define ADD_MTP(suffix, optional) \
    layer_name(name, sizeof(name), mtp, suffix); \
    if ((st = add_tensor_saf(spine, name, mtp, optional, err)) != CWICK_OK) { \
        cwick_spine_close(spine); return st; \
    }
            /* fusion, in checkpoint order */
            ADD_MTP("enorm.weight", false);
            ADD_MTP("hnorm.weight", false);
            ADD_MTP("eh_proj.weight", false);
            ADD_MTP("shared_head.norm.weight", false);
            /* the block itself */
            ADD_MTP("input_layernorm.weight", false);
            ADD_MTP("post_attention_layernorm.weight", false);
            ADD_MTP("self_attn.q_a_proj.weight", false);
            ADD_MTP("self_attn.q_b_proj.weight", false);
            ADD_MTP("self_attn.kv_a_proj_with_mqa.weight", false);
            ADD_MTP("self_attn.kv_b_proj.weight", false);
            ADD_MTP("self_attn.o_proj.weight", false);
            ADD_MTP("self_attn.q_a_layernorm.weight", false);
            ADD_MTP("self_attn.kv_a_layernorm.weight", false);
            ADD_MTP("self_attn.indexer.wk.weight", false);
            ADD_MTP("self_attn.indexer.wq_b.weight", false);
            ADD_MTP("self_attn.indexer.weights_proj.weight", false);
            ADD_MTP("self_attn.indexer.k_norm.weight", false);
            ADD_MTP("self_attn.indexer.k_norm.bias", true);
            ADD_MTP("self_attn.indexer.index_kpool_compress_ape", true);
            ADD_MTP("self_attn.indexer.index_kpool_compress_gate", true);
            ADD_MTP("block_sparse_moe.gate.weight", false);
            ADD_MTP("block_sparse_moe.gate.e_score_correction_bias", false);
            ADD_MTP("block_sparse_moe.shared_experts.gate_proj.weight", false);
            ADD_MTP("block_sparse_moe.shared_experts.up_proj.weight", false);
            ADD_MTP("block_sparse_moe.shared_experts.down_proj.weight", false);
#undef ADD_MTP
        }
#undef ADD_SAF

        /* Expert corpus completeness (never-pruned guard): every routed expert
         * matrix for every sparse layer must resolve. The nextn predict layer
         * (index num_hidden_layers) also carries an expert bank, but only for
         * the checkpoints that have one. */
        for (int64_t l = 0; l <= spec->num_hidden_layers; l++) {
            if (l == spec->num_hidden_layers &&
                spec->num_nextn_predict_layers <= 0)
                break;
            if (l < spec->num_hidden_layers && spec->mlp_dense[l]) continue;
            static const char *matrix_names[3] = {
                "gate_proj", "up_proj", "down_proj"
            };
            for (int64_t e = 0; e < spec->num_experts; e++) {
                for (int m = 0; m < 3; m++) {
                    char safname[192], scalename[208];
                    cwick_expert_saf_name(spec, l, e, matrix_names[m],
                                          safname, sizeof(safname));
                    cwick_st_tensor tensor;
                    if (cwick_safetensors_get(spine->saf, safname, &tensor, err)
                        != CWICK_OK) {
                        cwick_error_set(err, CWICK_ERR_CONTRACT,
                                       "expert corpus incomplete: missing %s "
                                       "(all three matrices of every expert are required)",
                                       safname);
                        cwick_spine_close(spine);
                        return CWICK_ERR_CONTRACT;
                    }
                    if (tensor.dtype == CWICK_ST_F8_E4M3) {
                        snprintf(scalename, sizeof(scalename), "%s_scale_inv", safname);
                        cwick_st_tensor scale;
                        if (cwick_safetensors_get(spine->saf, scalename, &scale, err)
                            != CWICK_OK) {
                            cwick_error_set(err, CWICK_ERR_CONTRACT,
                                           "expert corpus incomplete: missing FP8 scale %s",
                                           scalename);
                            cwick_spine_close(spine);
                            return CWICK_ERR_CONTRACT;
                        }
                    }
                }
            }
        }

        /* ---- accounting + real load -----------------------------------
         * The resident/streamed split was already computed above (it feeds
         * the memory budget); this pass warm-wires the resident tensors so
         * the first token does not pay a cold fault for every one of them,
         * and so a truncated or unreadable shard is discovered here rather
         * than mid-decode. */
        {
            cwick_safetensors_split(spine->saf, &spine->resident_bytes,
                                   &spine->streamed_bytes);
            struct timeval w0, w1;
            gettimeofday(&w0, NULL);
            for (size_t i = 0; i < spine->tensor_count; i++) {
                const cwick_tensor *t = &spine->tensors[i];
                if (t->data && t->data_bytes) {
                    volatile const uint8_t *p = (const uint8_t *)t->data;
                    /* stride-touch: enough to fault the whole range in */
                    uint64_t acc = 0;
                    for (size_t o = 0; o < t->data_bytes; o += 4096) acc ^= p[o];
                    (void)acc;
                    cwick_stream_warm(t->data, t->data_bytes);
                }
            }
            gettimeofday(&w1, NULL);
            double ms = (double)(w1.tv_sec - w0.tv_sec) * 1e3 +
                        (double)(w1.tv_usec - w0.tv_usec) / 1e3;
            CWICK_VERB("stor", "resident spine warm-wired in %.0f ms (%zu "
                      "tensors, %.2f GiB)", ms, spine->tensor_count,
                      cwick_gib(spine->resident_bytes));
            /* ---- keep it that way ----------------------------------
             * Warm-wiring only asks the VM to fault these pages in; it
             * does not stop the compressor from taking them back. Every
             * one of them is read on every decode step, so a compressed
             * page is a fault per token per tensor -- and when the GPU is
             * the one reading (the expert arena is the same story) the
             * stall measured 65 ms average and 3.6 s worst case per step.
             * Wiring is the only way to say "resident" and mean it.
             * mlock on overlapping ranges is idempotent, so wiring tensor
             * by tensor is safe even though tensors share shard mappings. */
            if (cwick_memory_noswap()) {
                size_t failed = 0;
                for (size_t i = 0; i < spine->tensor_count; i++) {
                    const cwick_tensor *t = &spine->tensors[i];
                    if (t->data && t->data_bytes &&
                        !cwick_mem_lock_range(t->data, (size_t)t->data_bytes,
                                             "resident spine"))
                        failed++;
                }
                if (failed == 0)
                    CWICK_INFO("stor", "resident spine wired: %.2f GiB held "
                              "incompressible and unswapped",
                              cwick_gib(spine->resident_bytes));
                else
                    CWICK_INFO("stor", "resident spine wiring failed for %zu "
                              "of %zu tensors — the VM may compress part of "
                              "the %.2f GiB spine (--allow-swap makes this "
                              "the expected behaviour)",
                              failed, spine->tensor_count,
                              cwick_gib(spine->resident_bytes));
            }
            if (spine->mem_plan) {
                uint64_t rss = cwick_process_rss_bytes();
                if (rss > 0)
                    CWICK_VERB("stor", "resident RSS %.2f GiB of the %.2f GiB "
                              "ceiling after warm-up", cwick_gib(rss),
                              cwick_gib(spine->mem_plan->target_bytes));
            }
        }
        return CWICK_OK;
    }

    /* ---- converted-format mode (unchanged) ---- */
    cwick_status st;
#define ADD(name, rows, cols) \
    if ((st = add_tensor(spine, name, rows, cols, repr, err)) != CWICK_OK) { \
        cwick_spine_close(spine); return st; \
    }
    ADD("model.norm.weight", 1, spec->hidden_size);
    ADD("model.output_attn_res_norm.weight", 1, spec->hidden_size);
    ADD("model.output_attn_res_proj.weight", 1, spec->hidden_size);
    ADD("lm_head.weight", spec->vocab_size, spec->hidden_size);
    ADD("embed_tokens.weight", spec->vocab_size, spec->hidden_size);
#undef ADD

    for (int64_t l = 0; l < spec->num_hidden_layers; l++) {
        char name[128];
        cwick_layer_kind kind = spec->layers[l];

#define ADD2(suffix, rows, cols) \
    layer_name(name, sizeof(name), l, suffix); \
    if ((st = add_tensor(spine, name, rows, cols, repr, err)) != CWICK_OK) { \
        cwick_spine_close(spine); return st; \
    }
        ADD2("input_layernorm.weight", 1, spec->hidden_size);
        ADD2("self_attention_res_norm.weight", 1, spec->hidden_size);
        ADD2("self_attention_res_proj.weight", 1, spec->hidden_size);
        ADD2("post_attention_layernorm.weight", 1, spec->hidden_size);
        ADD2("mlp_res_norm.weight", 1, spec->hidden_size);
        ADD2("mlp_res_proj.weight", 1, spec->hidden_size);

        /* Hyper-connection mixers (mHC models). Mirrors the safetensors
         * manifest so the BF16 layout can carry an mHC model too. */
        if (spec->mhc) {
            int64_t nstream = spec->hc_mult > 0 ? spec->hc_mult : 1;
            int64_t hc_mix = 2 * nstream + nstream * nstream;
            int64_t hc_d = nstream * spec->hidden_size;
            ADD2("hc_attn_fn", hc_mix, hc_d);
            ADD2("hc_attn_base", 1, hc_mix);
            ADD2("hc_attn_scale", 1, 3);
            ADD2("hc_ffn_fn", hc_mix, hc_d);
            ADD2("hc_ffn_base", 1, hc_mix);
            ADD2("hc_ffn_scale", 1, 3);
        }

        int64_t proj = spec->num_attention_heads * spec->head_dim; /* 96*128 */
        if (kind == CWICK_LAYER_KDA) {
            int64_t lin_proj = spec->linear_num_heads * spec->linear_head_dim;
            /* A_log has one entry per head in the released checkpoint. */
            ADD2("self_attn.A_log", 1, spec->linear_num_heads);
            ADD2("self_attn.dt_bias", 1, lin_proj);
            ADD2("self_attn.q_conv1d.weight", lin_proj, spec->short_conv_kernel_size);
            ADD2("self_attn.k_conv1d.weight", lin_proj, spec->short_conv_kernel_size);
            ADD2("self_attn.v_conv1d.weight", lin_proj, spec->short_conv_kernel_size);
            ADD2("self_attn.o_norm.weight", 1, spec->linear_head_dim);
            ADD2("self_attn.q_proj.weight", lin_proj, spec->hidden_size);
            ADD2("self_attn.k_proj.weight", lin_proj, spec->hidden_size);
            ADD2("self_attn.v_proj.weight", lin_proj, spec->hidden_size);
            ADD2("self_attn.g_a_proj.weight", spec->linear_head_dim, spec->hidden_size);
            ADD2("self_attn.g_b_proj.weight", lin_proj, spec->linear_head_dim);
            ADD2("self_attn.f_a_proj.weight", spec->linear_head_dim, spec->hidden_size);
            ADD2("self_attn.f_b_proj.weight", lin_proj, spec->linear_head_dim);
            ADD2("self_attn.b_proj.weight", spec->linear_num_heads, spec->hidden_size);
            ADD2("self_attn.o_proj.weight", spec->hidden_size, lin_proj);
        } else {
            int64_t q_out = spec->num_attention_heads *
                            (spec->qk_nope_head_dim + spec->qk_rope_head_dim);
            int64_t kv_out = spec->num_key_value_heads *
                             (spec->qk_nope_head_dim + spec->value_head_dim);
            int64_t g_out = proj;
            ADD2("self_attn.q_a_proj.weight", spec->q_lora_rank, spec->hidden_size);
            ADD2("self_attn.q_b_proj.weight", q_out, spec->q_lora_rank);
            ADD2("self_attn.kv_a_proj_with_mqa.weight",
                 spec->kv_lora_rank + spec->qk_rope_head_dim, spec->hidden_size);
            ADD2("self_attn.kv_b_proj.weight", kv_out, spec->kv_lora_rank);
            ADD2("self_attn.g_proj.weight", g_out, spec->hidden_size);
            ADD2("self_attn.o_proj.weight", spec->hidden_size, g_out);
            ADD2("self_attn.q_a_layernorm.weight", 1, spec->q_lora_rank);
            ADD2("self_attn.kv_a_layernorm.weight", 1, spec->kv_lora_rank);
        }

        if (spec->mlp_dense[l]) {
            ADD2("mlp.gate_proj.weight", spec->intermediate_size, spec->hidden_size);
            ADD2("mlp.up_proj.weight", spec->intermediate_size, spec->hidden_size);
            ADD2("mlp.down_proj.weight", spec->hidden_size, spec->intermediate_size);
        } else {
            const int64_t SI = spec->moe_intermediate_size * spec->num_shared_experts;
            ADD2("block_sparse_moe.gate.weight", spec->num_experts, spec->hidden_size);
            ADD2("block_sparse_moe.gate.e_score_correction_bias", 1, spec->num_experts);
            ADD2("block_sparse_moe.routed_expert_down_proj.weight",
                 spec->routed_expert_hidden_size, spec->hidden_size);
            ADD2("block_sparse_moe.routed_expert_up_proj.weight",
                 spec->hidden_size, spec->routed_expert_hidden_size);
            ADD2("block_sparse_moe.routed_expert_norm.weight",
                 1, spec->routed_expert_hidden_size);
            ADD2("block_sparse_moe.shared_experts.gate_proj.weight",
                 SI, spec->hidden_size);
            ADD2("block_sparse_moe.shared_experts.up_proj.weight",
                 SI, spec->hidden_size);
            ADD2("block_sparse_moe.shared_experts.down_proj.weight",
                 spec->hidden_size, SI);
        }
#undef ADD2
    }

    /* ---- expert corpus completeness (never-pruned guard) ---- */
    char edir[1024];
    cwick_path_join(edir, sizeof(edir), root, "resident/experts");
    if (!cwick_dir_exists(edir)) {
        cwick_error_set(err, CWICK_ERR_IO, "expert corpus missing: %s", edir);
        cwick_spine_close(spine);
        return CWICK_ERR_IO;
    }
    int64_t moe_layers = 0;
    for (int64_t l = 0; l < spec->num_hidden_layers; l++)
        if (!spec->mlp_dense[l]) moe_layers++;
    for (int64_t l = 0; l < spec->num_hidden_layers; l++) {
        if (spec->mlp_dense[l]) continue;   /* dense layers hold no experts */
        for (int64_t e = 0; e < spec->num_experts; e++) {
            char path[1152];
            snprintf(path, sizeof(path), "%s/expert-%lld-%lld.bin", edir, l, e);
            if (!cwick_file_exists(path)) {
                cwick_error_set(err, CWICK_ERR_CONTRACT,
                               "expert corpus incomplete: missing %s "
                               "(CWICK runs the full expert bank, never pruned)",
                               path);
                cwick_spine_close(spine);
                return CWICK_ERR_CONTRACT;
            }
            spine_expert_push(spine, path);
        }
    }
    (void)moe_layers;
    return CWICK_OK;
}

void cwick_spine_close(cwick_spine *spine) {
    if (!spine) return;
    /* The model is going away, so the mapping goes with it.
     *
     * These maps used to be left mapped for the life of the process, which was
     * right while a process served exactly one model and wrong the moment a
     * model could be swapped: the outgoing checkpoint's resident pages stayed,
     * and each switch added another model's worth. Measured on a DeepSeek ->
     * GLM switch that was ~29 GiB per model -- enough to push a second round
     * trip from 43.7 to 70.3 GiB warm-up and back over the ceiling.
     *
     * MADV_DONTNEED is not enough on macOS for a file mapping: measured, it
     * returns 0 for all 48 DeepSeek shards (475 GiB of address space) and RSS
     * does not move by a byte. Only munmap gives the pages back, so the views
     * over these weights must already be released -- cwick_engine_free does
     * that first. munlock first for the accounting: the resident spine is
     * wired, and a wired page is not reclaimable at all. */
    for (size_t i = 0; i < spine->map_count; i++) {
        if (!spine->maps[i]) continue;
        size_t len = spine->map_lens ? spine->map_lens[i] : 0;
        if (!len) continue;
        cwick_mem_unlock_range(spine->maps[i], len);
        cwick_munmap(spine->maps[i], len);
    }
    free(spine->tensors);
    for (size_t i = 0; i < spine->expert_file_count; i++) free(spine->expert_files[i]);
    free(spine->expert_files);
    free(spine->maps);
    free(spine->map_lens);
    if (spine->stream) {
        cwick_stream_prefetch_close(spine->stream);
        cwick_stream_close(spine->stream);
    }
    if (spine->saf) {
        cwick_safetensors_close(spine->saf);
        free(spine->saf);
    }
    free(spine->mem_plan);
    free((char *)spine->root);
    memset(spine, 0, sizeof(*spine));
}

/* The plan applied when this spine was opened (see mem.c). */
const cwick_mem_plan *cwick_spine_memory_plan(const cwick_spine *spine) {
    return spine ? spine->mem_plan : NULL;
}

/* Load one expert's matrices and convert to fp32. In the converted format the
 * body is w1 [I, RH], w2 [RH, I], w3 [I, RH]. In the safetensors source the
 * released GLM-5.3 experts are gate/up [I, H] and down [H, I] (FP8 E4M3 +
 * per-block scale_inv or BF16), mapped to the same w1/w2/w3 slots. */
cwick_status cwick_expert_load(const cwick_spine *spine, int64_t layer,
                             int64_t expert, const cwick_model_spec *spec,
                             float *w1, float *w2, float *w3,
                             cwick_error *err) {
    cwick_error_clear(err);
    int64_t I = spec->moe_intermediate_size;
    int64_t RH = spec->routed_expert_hidden_size;
    if (layer < 0 || expert < 0 || expert >= spec->num_experts) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "expert out of range");
        return CWICK_ERR_ARGUMENT;
    }

    if (spine->is_safetensors) {
        /* w1 = gate_proj [I, RH], w2 = down_proj [RH, I], w3 = up_proj [I, RH] */
        const char *mat[3] = { "gate_proj", "down_proj", "up_proj" };
        float *dst[3] = { w1, w2, w3 };
        int64_t n[3] = { I * RH, RH * I, I * RH };
        for (int m = 0; m < 3; m++) {
            char safname[192];
            cwick_expert_saf_name(spec, layer, expert, mat[m], safname,
                                  sizeof(safname));
            cwick_st_tensor st;
            cwick_status r = cwick_safetensors_get(spine->saf, safname, &st, err);
            if (r != CWICK_OK) return r;
            if (dst[m]) {
                if (st.dtype == CWICK_ST_F8_E4M3) {
                    const uint8_t *p = (const uint8_t *)st.data;
                    const float *si = (const float *)st.scale_inv;
                    int64_t ncb = st.scale_inv_cols;
                    for (int64_t i = 0; i < n[m]; i++) {
                        int64_t r_ = i / st.cols, c_ = i % st.cols;
                        dst[m][i] = cwick_fp8_e4m3_to_f32(p[i]) *
                                    si[(r_ / CWICK_ST_BLOCK) * ncb + (c_ / CWICK_ST_BLOCK)];
                    }
                } else {
                    const uint16_t *p = (const uint16_t *)st.data;
                    for (int64_t i = 0; i < n[m]; i++) dst[m][i] = cwick_bf16_to_f32(p[i]);
                }
            }
        }
        return CWICK_OK;
    }

    /* ---- converted expert corpus (unchanged) ---- */
    if (layer < 0 || layer >= spec->num_hidden_layers ||
        spec->mlp_dense[layer]) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "expert out of range");
        return CWICK_ERR_ARGUMENT;
    }
    /* The corpus is packed in layer order, skipping dense layers (the spine's
     * completeness loop pushes it that way), so a layer's slot is its rank
     * among the sparse layers -- not `layer - 1`. */
    int64_t rank = 0;
    for (int64_t l = 0; l < layer; l++)
        if (!spec->mlp_dense[l]) rank++;
    size_t idx = (size_t)rank * (size_t)spec->num_experts + (size_t)expert;
    if (idx >= spine->expert_file_count) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "expert corpus index out of range");
        return CWICK_ERR_ARGUMENT;
    }
    const char *path = spine->expert_files[idx];
    size_t len = 0;
    const void *base = cwick_mmap_file(path, &len);
    if (!base) {
        cwick_error_set(err, CWICK_ERR_IO, "cannot open expert file %s", path);
        return CWICK_ERR_IO;
    }
    size_t header = 20;
    size_t w1b = (size_t)(I * RH) * 2;
    size_t w2b = (size_t)(RH * I) * 2;
    size_t w3b = (size_t)(I * RH) * 2;
    if (len < header + w1b + w2b + w3b) {
        cwick_error_set(err, CWICK_ERR_IO, "expert file %s truncated", path);
        cwick_munmap(base, len);
        return CWICK_ERR_IO;
    }
    const char *magic = (const char *)base;
    if (memcmp(magic, CWICK_EXPERT_MAGIC, 8) != 0) {
        cwick_error_set(err, CWICK_ERR_IO, "expert file %s has bad magic", path);
        cwick_munmap(base, len);
        return CWICK_ERR_IO;
    }
    const uint8_t *body = (const uint8_t *)base + header;
    const uint16_t *p = (const uint16_t *)body;

    if (w1) {
        for (size_t i = 0; i < (size_t)(I * RH); i++) w1[i] = cwick_bf16_to_f32(p[i]);
    }
    p += (size_t)(I * RH);
    if (w2) {
        for (size_t i = 0; i < (size_t)(RH * I); i++) w2[i] = cwick_bf16_to_f32(p[i]);
    }
    p += (size_t)(RH * I);
    if (w3) {
        for (size_t i = 0; i < (size_t)(I * RH); i++) w3[i] = cwick_bf16_to_f32(p[i]);
    }
    cwick_munmap(base, len);
    return CWICK_OK;
}
