/*
 * ds_model.h — DeepSeek-V4.1-Flash-1017 on top of the shared safetensors loader.
 *
 * This is a second forward pass beside the GLM one, not a modification of it:
 * the two checkpoints share nothing but the process. Weights stay mmap'd and
 * are read through the loader's tensor table; only the dense non-expert weights
 * are touched every token, so a cold start costs one page fault per 16 KiB.
 *
 * Scope, matching what the reference implementation needs for text-only
 * inference: the 40 backbone layers, the sliding window plus compressed KV
 * attention, the two engram tables, and the shared/routed experts. The vision
 * tower, the aligner and the MTP draft heads are not loaded (they are separate
 * tensors; nothing reads them).
 */
#ifndef CWICK_DS_MODEL_H
#define CWICK_DS_MODEL_H

#include <stdint.h>
#include <stddef.h>

#include "cwick.h"
#include "ds_ops.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DS_MAX_LAYERS 64
#define DS_MAX_HC 8
/* Positions per prefill chunk: the forward pass is layer-major over a chunk,
 * so this is both the working-set bound and the granularity at which the
 * prompt is fed.
 *
 * 1024, for the same measured reason as MM_CHUNK: the expert read is a
 * per-block cost, so the biggest block that still fits a decision prompt
 * amortises it best. Buffers below are sized by this number. */
#define DS_CHUNK 1024

/* ------------------------------------------------------------------ */
/* config                                                               */
/* ------------------------------------------------------------------ */

typedef struct {
    int vocab_size, dim, moe_inter_dim, n_layers, n_heads, head_dim, rope_head_dim;
    int q_lora_rank, o_lora_rank, o_groups, window_size;
    int n_routed_experts, n_shared_experts, n_activated_experts;
    float swiglu_limit, rms_norm_eps, gate_temp, route_scale;
    int norm_topk_prob;
    int compress_ratios[DS_MAX_LAYERS];
    int max_ratio;                          /* largest ratio among kv sources */
    int kv_source_layers[DS_MAX_LAYERS], n_kv_source;
    int index_source_layers[DS_MAX_LAYERS], n_index_source;
    int index_n_heads, index_head_dim, index_topk;
    int candidate_source_layer, candidate_topk_blocks, candidate_block_size;
    int hc_mult, hc_sinkhorn_iters;
    float hc_eps;
    int engram_layer_ids[DS_MAX_LAYERS], n_engram;
    int64_t engram_num_embeddings[DS_MAX_LAYERS];
    int engram_max_ngram_size, engram_vocab_size, engram_n_heads, engram_head_dim;
    int engram_pad_id, engram_compressed_vocab_size;
    double rope_theta, compress_rope_theta, rope_factor;
    int original_seq_len, beta_fast, beta_slow;
    int max_seq_len;
} ds_config;

/* ------------------------------------------------------------------ */
/* one quantized weight                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    const uint8_t *w;        /* fp8 bytes, or packed fp4 bytes */
    const uint8_t *scale;    /* e8m0 bytes, same row count as the blocks */
    int64_t rows, cols;      /* logical shape of w */
    int fp4;                 /* 1: two fp4 values per byte along K */
    const uint16_t *w16;     /* the same weights widened to bf16 at load, for
                              * the tensors that are read in full every token;
                              * NULL when they still live in fp8 */
} ds_qweight;

/* ------------------------------------------------------------------ */
/* per-layer weights                                                    */
/* ------------------------------------------------------------------ */

/* One expert's weight plus its scale, resolved once at load time. */
typedef struct {
    cwick_st_tensor w;
    cwick_st_tensor s;      /* the paired *.scale tensor: the pin cache reads
                             * weight and scale in one go, so both descriptors
                             * have to survive load */
    const uint8_t *scale;
} ds_expert;

typedef struct {
    int layer_id;
    int compress_ratio;
    int is_kv_source, is_index_source, is_candidate_source, uses_candidates;

    /* attention */
    ds_qweight wq_a, wq_b, wkv, wo_b;   /* fp8 + 32x32 block scales */
    ds_qweight wo_a;                    /* fp8 + 32x32 block scales, block-diagonal */
    const uint16_t *q_norm;             /* bf16 [q_lora_rank] */
    const uint16_t *kv_norm;            /* bf16 [head_dim] */
    const float *attn_sink;             /* f32 [n_heads] */
    const uint16_t *attn_norm;          /* bf16 [dim] */

    /* ffn */
    const uint16_t *gate_w;             /* bf16 [n_experts, dim] */
    const float *gate_bias;             /* f32 [n_experts] */
    ds_qweight shared_w1, shared_w2, shared_w3;
    ds_expert *expert;                 /* [n_routed][3] w1,w2,w3 */
    const uint16_t *ffn_norm;           /* bf16 [dim] */

    /* hyper-connections (fp32) */
    const float *hc_attn_fn, *hc_attn_base, *hc_attn_scale;
    const float *hc_ffn_fn, *hc_ffn_base, *hc_ffn_scale;

    /* compressor (kv source layers only) */
    const uint16_t *cmp_wkv;            /* bf16 [head_dim, dim] */
    const uint16_t *cmp_wgate;          /* bf16 [head_dim, dim], ratio > 1 only */
    const uint16_t *cmp_norm;           /* bf16 [head_dim] */

    /* indexer (index source layers only) */
    ds_qweight idx_wq_b;                /* fp8 [n_heads*index_head_dim, q_lora_rank] */
    const uint16_t *idx_wk;             /* bf16 [index_head_dim, head_dim] */
    const uint16_t *idx_k_norm;         /* bf16 [index_head_dim] */
    const uint16_t *idx_weights_proj;   /* bf16 [n_heads, dim] */

    /* engram (two layers only) */
    int engram_slot;                    /* index into the engram tables, -1 if none */
    ds_qweight engram_wkv;              /* fp8 [dim*(hc+1), n_hash_cols*head_dim] */
    const uint8_t *eng_embed;           /* fp8 [rows, head_dim] */
    const uint8_t *eng_embed_scale;     /* e8m0 [rows, head_dim/32] */
    const uint16_t *eng_q_weight;       /* bf16 [hc, dim] */
    const uint16_t *eng_k_weight;       /* bf16 [hc, dim] */
} ds_layer;

/* ------------------------------------------------------------------ */
/* model                                                               */
/* ------------------------------------------------------------------ */

typedef struct ds_model {
    ds_config cfg;
    cwick_safetensors saf;
    /* Expert pin cache. The routed experts are ~272 GiB and are touched by six
     * routing slots per layer per token, so the same handful come back over
     * and over within a conversation. The GLM path already has a budgeted LRU
     * over raw expert blobs; pointing the DeepSeek read path at the same
     * stream gives it the same cache instead of a second one. NULL when the
     * cache is off, in which case reads fall back to the mapped shards. */
    cwick_stream *stream;
    int loaded;
    /* Whole-process memory budget this model was planned against (80% of host
     * RAM). The expert pin cache is only one line of that plan, and the decode
     * loop trims the cache against this number rather than against the cache's
     * own cap -- the KV tensors, the GPU-side arenas and the scratch are in
     * the footprint too and none of them is in the pin budget. */
    uint64_t rss_target;

    /* Resident copy of the dense weights: ~10 GiB of attention projections,
     * shared experts, embed/head and the hyper-connection coefficients. They
     * are read in full on every token, and the mapped shards they live in are
     * file-backed, so every page the kernel reclaims costs an SSD read on the
     * next token. Anonymous memory holding the only copy cannot be dropped
     * that way. Engram (203 GB) and the routed experts (297 GB) stay mapped
     * and streamed. */
    uint8_t *pin_base;
    size_t pin_cap, pin_used;

    /* bf16 copies of the fp8 dense weights (attention projections, shared
     * experts, indexer, engram key/value). Lossless, and read instead of the
     * fp8 originals -- decoding bf16 is a shift, decoding e4m3 is a table
     * lookup, which measured 119 GB/s against 38 GB/s on the same shapes. The
     * fp8 bytes stay mapped and are never touched again, so the resident cost
     * is one copy, not two. */
    uint8_t *dense16;
    size_t dense16_cap, dense16_used;

    const uint16_t *embed;              /* bf16 [vocab, dim] */
    const uint16_t *head;               /* bf16 [vocab, dim] */
    const uint16_t *final_norm;         /* bf16 [dim] */

    ds_layer layers[DS_MAX_LAYERS];
    ds_freqs freqs[DS_MAX_LAYERS];      /* two distinct tables, stored per layer */

    /* ---- caches (batch 1) ---- */
    float *win_kv;        /* [n_layers][window][head_dim] */
    float *cmp_kv;        /* [n_kv_source][max_cmp][head_dim] */
    float *cmp_state;     /* [n_kv_source][ratio][head_dim] fp32 pooling state */
    float *cmp_score;     /* [n_kv_source][ratio][head_dim] fp32 */
    float *idx_k;         /* [n_kv_source][max_cmp][index_head_dim] */
    int *src_slot;        /* [n_layers] -> most recent kv source at or before it */
    int64_t *tok_cache;   /* [max_seq] compressed token ids for the engram hash */

    /* ---- shared attention runtime (the reference's shared_attn) ---- */
    int *shared_topk;                   /* [DS_CHUNK][window + index_topk] */
    uint8_t *shared_cand;               /* [DS_CHUNK][max_cmp] candidate mask */
    int64_t *eng_hash;                  /* [DS_CHUNK][n_engram][n_hash_cols] */
    float *chunk_h;                     /* [DS_CHUNK][hc][dim] residual stream */
    float *chunk_pre;                   /* [DS_CHUNK][hc] pre_mix handed forward */

    /* ---- chunked MoE (see ds_moe_chunk) ----
     *
     * A prefill chunk routes every position first and then walks the *union*
     * of the six selections, so each expert's 18 MiB is read once per chunk
     * instead of once per position. The 89-token prompt that motivated this
     * touches 2362 (layer, expert) pairs but only 942 distinct ones. */
    float *chunk_sub;                   /* [DS_CHUNK][dim] sublayer input */
    float *chunk_subl;                  /* [DS_CHUNK][dim] sublayer output */
    float *chunk_contrib;               /* [DS_CHUNK][K][dim] routed terms, already
                                         * bf16-rounded, summed in slot order */
    float *chunk_selw;                  /* [DS_CHUNK][K] */
    int *chunk_sel;                     /* [DS_CHUNK][K] */
    float *chunk_fpre;                  /* [DS_CHUNK][hc] ffn mixes for the next layer */
    float *chunk_fpost;                 /* [DS_CHUNK][hc] */
    float *chunk_comb;                  /* [DS_CHUNK][hc][hc] */
    uint8_t *chunk_aq;                  /* [DS_CHUNK][dim] quantized sublayer input */
    uint8_t *chunk_as;                  /* [DS_CHUNK][dim/32] its scales */
    int *chunk_uni;                     /* [n_routed_experts] */
    uint8_t *chunk_mark;                /* [n_routed_experts] */
    float *h2;                          /* [hc][dim] scratch: ffn post-mix target */

    /* ---- scratch (batch 1, one query position at a time) ---- */
    int64_t pos;                        /* absolute position of the token being processed */
    float *h;                           /* [hc][dim] residual stream */
    float *res;                         /* [hc][dim] */
    float *sub;                         /* [dim] sublayer input */
    float *subl;                        /* [dim] sublayer output */
    float *tmp;                         /* [dim] */
    float *q;                           /* [n_heads][head_dim] */
    float *qr;                          /* [q_lora_rank] */
    float *kv;                          /* [head_dim] */
    float *latent;                      /* [head_dim] the RoPE-free compressed latent */
    float *o;                           /* [n_heads][head_dim] */
    float *o_g;                         /* [n_groups * o_lora_rank] wo_a output */
    float *attn_out;                    /* [dim] */
    float *moe_x;                       /* [dim] */
    float *moe_y;                       /* [dim] */
    float *expert_gate, *expert_up;     /* [moe_inter_dim] */
    float *pre_mix, *attn_pre, *attn_post, *ffn_pre, *ffn_post;
    float *comb;                        /* [hc*hc] */
    float *q_idx;                       /* [index_n_heads * index_head_dim] */
    float *ik;                          /* [index_head_dim] */
    float *iw;                          /* [index_n_heads] */
    float *router;                      /* [n_routed_experts] */
    int *sel;                           /* [n_activated_experts] */
    float *selw;                        /* [n_activated_experts] */
    float *eng_w;                       /* [hc * dim] engram q_weight * k_weight */
    uint8_t *act_q;                     /* [max(dim, moe_inter_dim, groups*lora)] */
    uint8_t *act_s;                     /* its per-32 scales */
    int *idxs;                          /* [window + index_topk] */
    float *scores;                      /* [max_cmp] indexer scores */
    float *att_scores;                  /* [n_heads][window + index_topk], one
                                         * row per head so the heads can run on
                                         * the pool at the same time */
    float *bsc;                         /* [max_cmp / block + 2] candidate block scores */
    float *heap;                        /* [max(index_topk, candidate_topk_blocks)] */
    int *heap_idx;                      /* the same, indices */
    int *blk_out;                       /* [max(index_topk, candidate_topk_blocks)] */

    /* engram scratch */
    float *eng_embed;                   /* [n_hash_cols * head_dim] */
    float *eng_kv;                      /* [dim * (hc+1)] */
    float *eng_key;                     /* [hc][dim] */
    float *eng_val;                     /* [dim] */
    int64_t *eng_rows;                  /* [n_hash_cols] */
} ds_model;

/* ------------------------------------------------------------------ */
/* api                                                                  */
/* ------------------------------------------------------------------ */

/* Read config.json (the HF layout: text_config holds the numbers), open every
 * safetensors shard through the shared loader and bind the weights by name.
 * Returns 0 on success; on failure fills err and returns nonzero. */
int ds_model_open(ds_model *m, const char *root, int max_seq_len, cwick_error *err);
void ds_model_close(ds_model *m);

/* Drop every cached KV/index state and rewind to position 0. The weights stay
 * mapped; only the per-sequence buffers are cleared. */
void ds_model_reset(ds_model *m);

/* Run positions [start_pos, start_pos+n). Fills logits[vocab] for the LAST
 * position only (all the reference computes for generation). n may be 1. */
int ds_model_forward(ds_model *m, const int32_t *ids, int n, int64_t start_pos,
                     float *logits, cwick_error *err);

/* ---- prefix snapshots (pfx.h kind 1) ----
 *
 * Snapshot the cross-token state of the first `pos` positions: window KV,
 * compressed KV, compressor pooling state, indexer keys and the engram token
 * history. `ds_state_bytes` is the exact size `ds_state_pack` writes, and
 * `ds_state_apply` puts it back so the next forward continues as if the prompt
 * had been prefilled. All three return/take 0 on success, -1 on a geometry
 * mismatch (which the caller must treat as a cache miss, never as an error). */
size_t ds_state_bytes(const ds_model *m, int64_t pos);
int ds_state_pack(const ds_model *m, int64_t pos, void *buf, size_t cap);
int ds_state_apply(ds_model *m, const void *buf, size_t len, int64_t *pos_out);

/* Prefill progress for /v1/metrics. The GLM engine reports this from its own
 * layer loop; this backend has a separate forward pass, so it needs its own
 * hooks or the dashboard shows 0/0 for the whole prompt. `begin` takes the
 * prompt length and the layer count, `done` is the token count finished so
 * far (called between chunks), `end` returns the gauges to idle. */
void ds_prefill_progress_begin(int64_t total_tokens, int layers);
void ds_prefill_progress_done(int64_t done_tokens);
void ds_prefill_progress_end(void);

/* True when root/config.json describes this architecture. */
int ds_model_probe(const char *root);

#ifdef __cplusplus
}
#endif

#endif /* CWICK_DS_MODEL_H */
