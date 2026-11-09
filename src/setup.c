/* setup.c — synthetic model generator and setup command.
 *
 * `cwick_setup_synthetic` writes a complete, deterministic, tiny model in the
 * CWICK storage layout (see docs/STORAGE.md) so the full engine + server can be
 * verified end-to-end without downloading real weights:
 *
 *   config.json
 *   resident/tensors/<name>.bin       BF16 dense spine
 *   resident/experts/expert-L-E.bin   BF16 routed experts
 *
 * The synthetic model exercises every path: layer 0 is a KDA (linear
 * attention) layer with a dense MLP, layer 1 is an MLA (full attention) layer
 * with a routed MoE. All weights are seeded, so runs are reproducible.
 */
#include "cwick.h"
#include "json.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

/* xorshift64 PRNG (deterministic) */
typedef struct { uint64_t s; } rng;
static uint64_t rng_next(rng *r) {
    uint64_t x = r->s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    r->s = x;
    return x;
}
static float rng_f(rng *r) {
    /* uniform in [-1, 1) */
    uint64_t u = rng_next(r);
    float v = (float)(int64_t)(u & 0xFFFF) / 32768.0f - 1.0f;
    return v;
}

/* Router-gate weight scale for the synthetic checkpoints. Deliberately large:
 * with a small scale every sigmoid score sits near 0.5, the renormalised
 * top-k weights are all ~1/k, and a per-token indexing bug inside the batched
 * MoE (reading another token's score row) still produces almost the right
 * answer -- measured rel 2e-5, invisible under the equivalence tolerance.
 * Decisive logits make the same bug an O(1) error, so the batched-vs-serial
 * equivalence tests actually guard the routing-weight gather. The post-norm
 * hidden state is ~0.1 per element (norm weights are 0.1), so the scale has
 * to be well above 1 before the logits separate. */
#define SYNTH_ROUTER_SCALE 16.0f

static uint16_t *alloc_bf16(size_t n, rng *r, float scale) {
    uint16_t *buf = cwick_xmalloc(n * sizeof(uint16_t));
    for (size_t i = 0; i < n; i++) buf[i] = cwick_f32_to_bf16(rng_f(r) * scale);
    return buf;
}

static int write_tensor(const char *dir, const char *name, int64_t rows,
                        int64_t cols, rng *r, float scale) {
    char path[1152];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    size_t n = (size_t)(rows * cols);
    uint16_t *buf = alloc_bf16(n, r, scale);
    int rc = cwick_write_file(path, buf, n * sizeof(uint16_t));
    free(buf);
    return rc;
}

static int write_norm(const char *dir, const char *name, int64_t n, rng *r,
                      float scale) {
    return write_tensor(dir, name, 1, n, r, scale);
}

static int write_expert(const char *dir, int64_t layer, int64_t expert,
                        int64_t I, int64_t RH, rng *r) {
    char path[1152];
    snprintf(path, sizeof(path), "%s/expert-%lld-%lld.bin", dir, layer, expert);
    uint8_t header[20];
    memcpy(header, CWICK_EXPERT_MAGIC, 8);
    uint32_t le[3] = { (uint32_t)layer, (uint32_t)expert, CWICK_EXPERT_MATRICES };
    memcpy(header + 8, le, 12);
    size_t w1n = (size_t)(I * RH);
    size_t w2n = (size_t)(RH * I);
    size_t w3n = (size_t)(I * RH);
    uint16_t *w1 = alloc_bf16(w1n, r, 0.1f);
    uint16_t *w2 = alloc_bf16(w2n, r, 0.1f);
    uint16_t *w3 = alloc_bf16(w3n, r, 0.1f);
    size_t total = 20 + (w1n + w2n + w3n) * sizeof(uint16_t);
    char *buf = cwick_xmalloc(total);
    memcpy(buf, header, 20);
    memcpy(buf + 16, w1, w1n * 2);
    memcpy(buf + 16 + w1n * 2, w2, w2n * 2);
    memcpy(buf + 16 + (w1n + w2n) * 2, w3, w3n * 2);
    int rc = cwick_write_file(path, buf, total);
    free(buf);
    free(w1);
    free(w2);
    free(w3);
    return rc;
}

/* The synthetic configuration (fixed for reproducibility). */
static void synth_dims(int64_t *hidden, int64_t *layers, int64_t *experts,
                       int64_t *vocab) {
    *hidden = 64;
    *layers = 2;
    *experts = 8;
    *vocab = 256;
}

static const char *synth_config_json(void) {
    return
        "{"
        "\"model_type\":\"glm_5_3_flash_synthetic\","
        "\"dtype\":\"bfloat16\","
        "\"tie_word_embeddings\":false,"
        "\"bos_token_id\":0,\"eos_token_id\":1,\"pad_token_id\":2,"
        "\"hidden_size\":64,"
        "\"num_hidden_layers\":2,"
        "\"vocab_size\":256,"
        "\"max_position_embeddings\":512,"
        "\"rms_norm_eps\":1e-5,"
        "\"num_attention_heads\":4,"
        "\"num_key_value_heads\":4,"
        "\"head_dim\":16,"
        "\"q_lora_rank\":16,"
        "\"kv_lora_rank\":16,"
        "\"qk_nope_head_dim\":8,"
        "\"qk_rope_head_dim\":8,"
        "\"value_head_dim\":8,"
        "\"mla_use_output_gate\":true,"
        "\"attn_res_block_size\":2,"
        "\"linear_attn_config\":{\"num_heads\":4,\"head_dim\":16,"
        "\"short_conv_kernel_size\":4,"
        "\"kda_layers\":[1],\"full_attn_layers\":[2]},"
        "\"intermediate_size\":128,"
        "\"num_experts\":8,"
        "\"num_experts_per_token\":2,"
        "\"num_shared_experts\":1,"
        "\"moe_intermediate_size\":32,"
        "\"routed_expert_hidden_size\":32,"
        "\"num_expert_groups\":1,"
        "\"topk_groups\":1,"
        "\"moe_renormalize\":true,"
        "\"latent_moe_use_norm\":true,"
        "\"activation_situ_beta\":4.0,"
        "\"activation_situ_linear_beta\":25.0,"
        "\"routed_scaling_factor\":1.0,"
        "\"gate_lower_bound\":-5.0"
        "}";
}

cwick_status cwick_setup_synthetic(const char *root, cwick_error *err) {
    cwick_error_clear(err);
    int64_t H, L, E, V;
    synth_dims(&H, &L, &E, &V);

    char td[1152], ed[1152];
    cwick_path_join(td, sizeof(td), root, "resident/tensors");
    cwick_path_join(ed, sizeof(ed), root, "resident/experts");
    if (cwick_mkdir_p(td) != 0 || cwick_mkdir_p(ed) != 0) {
        cwick_error_set(err, CWICK_ERR_IO, "cannot create model directories under %s", root);
        return CWICK_ERR_IO;
    }

    /* config.json */
    char cfg[1152];
    cwick_path_join(cfg, sizeof(cfg), root, "config.json");
    const char *json = synth_config_json();
    if (cwick_write_file(cfg, json, strlen(json)) != 0) {
        cwick_error_set(err, CWICK_ERR_IO, "cannot write config.json");
        return CWICK_ERR_IO;
    }

    rng r = { 0x2A6F5E3C9B8D7F41ULL };

    /* ---- global tensors ---- */
    write_norm(td, "model.norm.weight", H, &r, 0.1f);
    write_norm(td, "model.output_attn_res_norm.weight", H, &r, 0.1f);
    write_norm(td, "model.output_attn_res_proj.weight", H, &r, 0.1f);
    write_tensor(td, "lm_head.weight", V, H, &r, 0.05f);
    write_tensor(td, "embed_tokens.weight", V, H, &r, 0.05f);

    /* ---- layer 0: KDA + dense MLP ---- */
    const int64_t proj = 4 * 16;         /* lin heads * head_dim */
    const int64_t hd = 16;
    const int64_t nh_kda = 4;            /* A_log is per head */
    write_norm(td, "layers.0.input_layernorm.weight", H, &r, 0.1f);
    write_norm(td, "layers.0.self_attention_res_norm.weight", H, &r, 0.1f);
    write_norm(td, "layers.0.self_attention_res_proj.weight", H, &r, 0.1f);
    write_norm(td, "layers.0.post_attention_layernorm.weight", H, &r, 0.1f);
    write_norm(td, "layers.0.mlp_res_norm.weight", H, &r, 0.1f);
    write_norm(td, "layers.0.mlp_res_proj.weight", H, &r, 0.1f);
    /* KDA attention */
    /* A_log is per head in the released checkpoint ([num_heads]). */
    write_tensor(td, "layers.0.self_attn.A_log", 1, nh_kda, &r, 0.1f);
    write_tensor(td, "layers.0.self_attn.dt_bias", 1, proj, &r, 0.1f);
    write_tensor(td, "layers.0.self_attn.q_conv1d.weight", proj, 4, &r, 0.3f);
    write_tensor(td, "layers.0.self_attn.k_conv1d.weight", proj, 4, &r, 0.3f);
    write_tensor(td, "layers.0.self_attn.v_conv1d.weight", proj, 4, &r, 0.3f);
    write_norm(td, "layers.0.self_attn.o_norm.weight", hd, &r, 0.1f);
    write_tensor(td, "layers.0.self_attn.q_proj.weight", proj, H, &r, 0.1f);
    write_tensor(td, "layers.0.self_attn.k_proj.weight", proj, H, &r, 0.1f);
    write_tensor(td, "layers.0.self_attn.v_proj.weight", proj, H, &r, 0.1f);
    /* Two-stage output gate, as in the released checkpoint. */
    write_tensor(td, "layers.0.self_attn.g_a_proj.weight", hd, H, &r, 0.1f);
    write_tensor(td, "layers.0.self_attn.g_b_proj.weight", proj, hd, &r, 0.1f);
    write_tensor(td, "layers.0.self_attn.f_a_proj.weight", hd, H, &r, 0.1f);
    write_tensor(td, "layers.0.self_attn.f_b_proj.weight", proj, hd, &r, 0.1f);
    write_tensor(td, "layers.0.self_attn.b_proj.weight", 4, H, &r, 0.1f);
    write_tensor(td, "layers.0.self_attn.o_proj.weight", H, proj, &r, 0.1f);
    /* dense MLP */
    write_tensor(td, "layers.0.mlp.gate_proj.weight", 128, H, &r, 0.1f);
    write_tensor(td, "layers.0.mlp.up_proj.weight", 128, H, &r, 0.1f);
    write_tensor(td, "layers.0.mlp.down_proj.weight", H, 128, &r, 0.1f);

    /* ---- layer 1: MLA + MoE ---- */
    const int64_t q_dim = 8 + 8;         /* nope + rope */
    const int64_t kv_dim = 8 + 8;        /* nope + v */
    const int64_t q_out = 4 * q_dim;     /* heads * q_dim */
    const int64_t kv_out = 4 * kv_dim;   /* kv heads * kv_dim */
    write_norm(td, "layers.1.input_layernorm.weight", H, &r, 0.1f);
    write_norm(td, "layers.1.self_attention_res_norm.weight", H, &r, 0.1f);
    write_norm(td, "layers.1.self_attention_res_proj.weight", H, &r, 0.1f);
    write_norm(td, "layers.1.post_attention_layernorm.weight", H, &r, 0.1f);
    write_norm(td, "layers.1.mlp_res_norm.weight", H, &r, 0.1f);
    write_norm(td, "layers.1.mlp_res_proj.weight", H, &r, 0.1f);
    write_tensor(td, "layers.1.self_attn.q_a_proj.weight", 16, H, &r, 0.1f);
    write_tensor(td, "layers.1.self_attn.q_b_proj.weight", q_out, 16, &r, 0.1f);
    write_tensor(td, "layers.1.self_attn.kv_a_proj_with_mqa.weight", 24, H, &r, 0.1f);
    write_tensor(td, "layers.1.self_attn.kv_b_proj.weight", kv_out, 16, &r, 0.1f);
    write_tensor(td, "layers.1.self_attn.g_proj.weight", 64, H, &r, 0.1f);
    write_tensor(td, "layers.1.self_attn.o_proj.weight", H, 64, &r, 0.1f);
    write_norm(td, "layers.1.self_attn.q_a_layernorm.weight", 16, &r, 0.1f);
    write_norm(td, "layers.1.self_attn.kv_a_layernorm.weight", 16, &r, 0.1f);
    /* MoE */
    write_tensor(td, "layers.1.block_sparse_moe.gate.weight", E, H, &r, SYNTH_ROUTER_SCALE);
    write_tensor(td, "layers.1.block_sparse_moe.gate.e_score_correction_bias", 1, E, &r, 0.1f);
    write_tensor(td, "layers.1.block_sparse_moe.routed_expert_down_proj.weight", 32, H, &r, 0.1f);
    write_tensor(td, "layers.1.block_sparse_moe.routed_expert_up_proj.weight", H, 32, &r, 0.1f);
    write_norm(td, "layers.1.block_sparse_moe.routed_expert_norm.weight", 32, &r, 0.1f);
    write_tensor(td, "layers.1.block_sparse_moe.shared_experts.gate_proj.weight", 32, H, &r, 0.1f);
    write_tensor(td, "layers.1.block_sparse_moe.shared_experts.up_proj.weight", 32, H, &r, 0.1f);
    write_tensor(td, "layers.1.block_sparse_moe.shared_experts.down_proj.weight", H, 32, &r, 0.1f);

    /* ---- experts ---- */
    const int64_t I = 32, RH = 32;
    for (int64_t e = 0; e < E; e++) {
        if (write_expert(ed, 1, e, I, RH, &r) != 0) {
            cwick_error_set(err, CWICK_ERR_IO, "cannot write expert %lld", e);
            return CWICK_ERR_IO;
        }
    }
    return CWICK_OK;
}

/* Same idea as cwick_setup_synthetic, but the released checkpoint's shape:
 * every layer is a full-attention (MLA) layer with hyper-connections
 * (`mhc`/`hc_mult`) instead of the residual-anchor bank, and every layer is
 * MoE (no dense MLP). It exists so the layer-major batched prefill
 * (forward_prefill_mhc) can be diffed against the token-serial reference
 * (forward_token_mhc) on a model small enough to run in the test suite. */

static void synth_mhc_dims(int64_t *H, int64_t *L, int64_t *E, int64_t *V,
                           int64_t *ns) {
    *H = 64; *L = 2; *E = 8; *V = 256; *ns = 2;   /* hc_mult = 2 streams */
}

static const char *synth_mhc_config_json(void) {
    return
        "{"
        "\"model_type\":\"glm-5.3-flash-synthetic-mhc\","
        "\"dtype\":\"bfloat16\","
        "\"tie_word_embeddings\":false,"
        "\"bos_token_id\":0,\"eos_token_id\":1,\"pad_token_id\":2,"
        "\"hidden_size\":64,"
        "\"num_hidden_layers\":2,"
        "\"vocab_size\":256,"
        "\"max_position_embeddings\":512,"
        "\"rms_norm_eps\":1e-5,"
        "\"num_attention_heads\":4,"
        "\"num_key_value_heads\":4,"
        "\"head_dim\":16,"
        "\"q_lora_rank\":16,"
        "\"kv_lora_rank\":16,"
        "\"qk_nope_head_dim\":8,"
        "\"qk_rope_head_dim\":8,"
        "\"value_head_dim\":8,"
        "\"mla_use_output_gate\":false,"
        "\"mhc\":true,"
        "\"hc_mult\":2,"
        "\"hc_eps\":0.000001,"
        "\"hc_sinkhorn_iters\":2,"
        "\"intermediate_size\":128,"
        "\"num_experts\":8,"
        "\"num_experts_per_token\":2,"
        "\"num_shared_experts\":1,"
        "\"moe_intermediate_size\":32,"
        "\"routed_expert_hidden_size\":32,"
        "\"num_expert_groups\":1,"
        "\"topk_groups\":1,"
        "\"moe_renormalize\":true,"
        "\"latent_moe_use_norm\":true,"
        "\"activation_situ_beta\":4.0,"
        "\"activation_situ_linear_beta\":25.0,"
        "\"routed_scaling_factor\":1.0,"
        "\"gate_lower_bound\":-5.0,"
        "\"mlp_layer_types\":[\"dense\",\"sparse\"]"
        "}";
}

cwick_status cwick_setup_synthetic_mhc(const char *root, cwick_error *err) {
    cwick_error_clear(err);
    int64_t H, L, E, V, ns;
    synth_mhc_dims(&H, &L, &E, &V, &ns);

    const int64_t mix = 2 * ns + ns * ns;   /* hc_pre emits 2n + n*n values */
    const int64_t d = ns * H;               /* residual bank width          */
    const int64_t nh = 4, hd = 16;          /* heads, head_dim              */
    const int64_t q_lora = 16, kv_lora = 16;
    const int64_t nope = 8, rope = 8, vdim = 8;
    const int64_t q_dim = nope + rope;
    const int64_t q_out = nh * q_dim;
    const int64_t kv_dim = nope + vdim;
    const int64_t kv_out = nh * kv_dim;
    /* The loader sizes g_proj/o_proj by proj = nh * head_dim, so match it. */
    const int64_t proj = nh * hd;
    const int64_t I = 32, RH = 32;

    char td[1152], ed[1152];
    cwick_path_join(td, sizeof(td), root, "resident/tensors");
    cwick_path_join(ed, sizeof(ed), root, "resident/experts");
    if (cwick_mkdir_p(td) != 0 || cwick_mkdir_p(ed) != 0) {
        cwick_error_set(err, CWICK_ERR_IO, "cannot create model directories under %s", root);
        return CWICK_ERR_IO;
    }

    char cfg[1152];
    cwick_path_join(cfg, sizeof(cfg), root, "config.json");
    const char *json = synth_mhc_config_json();
    if (cwick_write_file(cfg, json, strlen(json)) != 0) {
        cwick_error_set(err, CWICK_ERR_IO, "cannot write config.json");
        return CWICK_ERR_IO;
    }

    rng r = { 0x51C3A9E7B4D2F608ULL };

    /* ---- global tensors ---- */
    write_norm(td, "model.norm.weight", H, &r, 0.1f);
    write_tensor(td, "lm_head.weight", V, H, &r, 0.05f);
    write_tensor(td, "embed_tokens.weight", V, H, &r, 0.05f);
    /* The BF16 spine loader requires the residual-anchor vectors even when the
     * forward pass is mHC (which never reads them). */
    write_norm(td, "model.output_attn_res_norm.weight", H, &r, 0.1f);
    write_norm(td, "model.output_attn_res_proj.weight", H, &r, 0.1f);

    char name[160];
    /* write_tensor/write_norm take a finished name, so expand the layer index
     * here instead of passing them a printf format. */
#define SN(fmt) snprintf(name, sizeof(name), fmt, l)
    for (int64_t l = 0; l < L; l++) {
        SN("layers.%lld.input_layernorm.weight");           write_norm(td, name, H, &r, 0.1f);
        SN("layers.%lld.post_attention_layernorm.weight");  write_norm(td, name, H, &r, 0.1f);
        SN("layers.%lld.self_attention_res_norm.weight");   write_norm(td, name, H, &r, 0.1f);
        SN("layers.%lld.self_attention_res_proj.weight");   write_norm(td, name, H, &r, 0.1f);
        SN("layers.%lld.mlp_res_norm.weight");              write_norm(td, name, H, &r, 0.1f);
        SN("layers.%lld.mlp_res_proj.weight");              write_norm(td, name, H, &r, 0.1f);

        /* hyper-connection mixers: hc_pre(x) = (x @ fn^T)*scale-ish + base */
        SN("layers.%lld.hc_attn_fn");     write_tensor(td, name, mix, d, &r, 0.1f);
        SN("layers.%lld.hc_attn_base");   write_tensor(td, name, 1, mix, &r, 0.1f);
        SN("layers.%lld.hc_attn_scale");  write_tensor(td, name, 1, 3, &r, 0.5f);
        SN("layers.%lld.hc_ffn_fn");      write_tensor(td, name, mix, d, &r, 0.1f);
        SN("layers.%lld.hc_ffn_base");    write_tensor(td, name, 1, mix, &r, 0.1f);
        SN("layers.%lld.hc_ffn_scale");   write_tensor(td, name, 1, 3, &r, 0.5f);

        /* MLA attention (full attention on every layer) */
        SN("layers.%lld.self_attn.q_a_proj.weight");           write_tensor(td, name, q_lora, H, &r, 0.1f);
        SN("layers.%lld.self_attn.q_b_proj.weight");           write_tensor(td, name, q_out, q_lora, &r, 0.1f);
        SN("layers.%lld.self_attn.kv_a_proj_with_mqa.weight"); write_tensor(td, name, kv_lora + rope, H, &r, 0.1f);
        SN("layers.%lld.self_attn.kv_b_proj.weight");          write_tensor(td, name, kv_out, kv_lora, &r, 0.1f);
        SN("layers.%lld.self_attn.o_proj.weight");             write_tensor(td, name, H, proj, &r, 0.1f);
        /* required by the loader even though mHC passes g=NULL to MLA */
        SN("layers.%lld.self_attn.g_proj.weight");             write_tensor(td, name, proj, H, &r, 0.1f);
        SN("layers.%lld.self_attn.q_a_layernorm.weight");      write_norm(td, name, q_lora, &r, 0.1f);
        SN("layers.%lld.self_attn.kv_a_layernorm.weight");     write_norm(td, name, kv_lora, &r, 0.1f);

        if (l == 0) {
            /* The converted expert corpus is 1-indexed by layer (spine.c
             * treats layer 0 as dense), so layer 0 carries a plain MLP and the
             * MoE lives on the layers above it. */
            SN("layers.%lld.mlp.gate_proj.weight"); write_tensor(td, name, 128, H, &r, 0.1f);
            SN("layers.%lld.mlp.up_proj.weight");   write_tensor(td, name, 128, H, &r, 0.1f);
            SN("layers.%lld.mlp.down_proj.weight"); write_tensor(td, name, H, 128, &r, 0.1f);
        } else {
            /* MoE (latent layout, one shared expert) */
            SN("layers.%lld.block_sparse_moe.gate.weight");                     write_tensor(td, name, E, H, &r, SYNTH_ROUTER_SCALE);
            SN("layers.%lld.block_sparse_moe.gate.e_score_correction_bias");    write_tensor(td, name, 1, E, &r, 0.1f);
            SN("layers.%lld.block_sparse_moe.routed_expert_down_proj.weight");  write_tensor(td, name, RH, H, &r, 0.1f);
            SN("layers.%lld.block_sparse_moe.routed_expert_up_proj.weight");    write_tensor(td, name, H, RH, &r, 0.1f);
            SN("layers.%lld.block_sparse_moe.routed_expert_norm.weight");       write_norm(td, name, RH, &r, 0.1f);
            SN("layers.%lld.block_sparse_moe.shared_experts.gate_proj.weight"); write_tensor(td, name, I, H, &r, 0.1f);
            SN("layers.%lld.block_sparse_moe.shared_experts.up_proj.weight");   write_tensor(td, name, I, H, &r, 0.1f);
            SN("layers.%lld.block_sparse_moe.shared_experts.down_proj.weight"); write_tensor(td, name, H, I, &r, 0.1f);

            for (int64_t e = 0; e < E; e++) {
                if (write_expert(ed, l, e, I, RH, &r) != 0) {
                    cwick_error_set(err, CWICK_ERR_IO, "cannot write expert %lld-%lld", l, e);
                    return CWICK_ERR_IO;
                }
            }
        }
    }
#undef SN
    return CWICK_OK;
}
