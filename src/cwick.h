/*
 * cwick.h — CWICK: a C implementation of a full-model runtime for
 * GLM-5.3-Flash on Apple Silicon.
 *
 * CWICK runs the *complete, never-pruned* target model on a single Apple
 * Silicon Mac:
 *   - one native C process owns the engine and the OpenAI-compatible server;
 *   - the model contract is validated from config.json and fails closed;
 *   - weights are memory-mapped from a BF16 resident spine (row-int8 packs
 *     optional), experts are read per routed layer, nothing is pruned;
 *   - MLA + KDA attention, MoE routing (sigmoid/tanh + top-k + renormalize),
 *     shared experts and the 12-layer residual-anchor mechanism;
 *   - the HTTP server implements /v1/models, /v1/completions and
 *     /v1/chat/completions with SSE streaming and fail-closed validation.
 */
#ifndef CWICK_H
#define CWICK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Version / ABI                                                       */
/* ------------------------------------------------------------------ */
#define CWICK_VERSION_MAJOR 0
#define CWICK_VERSION_MINOR 1
#define CWICK_VERSION_PATCH 0
#define CWICK_ABI_VERSION 1

/* ------------------------------------------------------------------ */
/* Errors                                                              */
/* ------------------------------------------------------------------ */
typedef enum {
    CWICK_OK = 0,
    CWICK_ERR_IO,
    CWICK_ERR_PARSE,
    CWICK_ERR_CONTRACT,   /* model contract mismatch (quality guard) */
    CWICK_ERR_MEMORY,
    CWICK_ERR_STATE,      /* engine lifecycle / busy */
    CWICK_ERR_ARGUMENT,
    CWICK_ERR_UNSUPPORTED,
    CWICK_ERR_NOT_FOUND,  /* a cache/prefix lookup had no matching entry */
} cwick_status;

typedef struct cwick_error {
    cwick_status status;
    char message[512];
} cwick_error;

void cwick_error_clear(cwick_error *err);
void cwick_error_set(cwick_error *err, cwick_status status, const char *fmt, ...)
    __attribute__((format(printf, 3, 4)));
const char *cwick_error_message(const cwick_error *err);

/* ------------------------------------------------------------------ */
/* Model contract                                                      */
/* ------------------------------------------------------------------ */
typedef enum {
    CWICK_LAYER_MLA = 0,   /* full (softmax) attention with latent KV (legacy) */
    CWICK_LAYER_KDA = 1,   /* linear (recurrent) attention with short conv */
    CWICK_LAYER_DSA = 2,   /* deepseek sparse attention with indexer (GLM-5.3) */
} cwick_layer_kind;

/* Released weight representation. The real GLM-5.3-Flash checkpoint ships as
 * HF FP8: F8_E4M3 weights plus a paired F32 `*_scale_inv` (one inverse scale
 * per 128x128 block). BF16/F32 tensors remain unquantized in the same shards. */
typedef enum {
    CWICK_QUANT_BF16 = 0,     /* original BF16/F32 contract (synthetic/converted) */
    CWICK_QUANT_FP8_E4M3 = 1, /* HF FP8 E4M3 + per-128x128 F32 scale_inv */
} cwick_quant_kind;

typedef struct cwick_model_spec {
    char model_type[64];
    char dtype[16];
    /* token ids */
    int64_t bos_token_id;
    int64_t eos_token_id;
    /* GLM-5.3 ends a turn on any of <|endoftext|>, <|user|> or
     * <|observation|> (generation_config.json). Stopping on the first id
     * alone leaves the model talking to itself until max_tokens. */
    int64_t eos_ids[8];
    int n_eos;
    int64_t pad_token_id;
    /* dense geometry */
    int64_t hidden_size;             /* H */
    int64_t intermediate_size;       /* dense MLP hidden (layer 0) */
    int64_t num_hidden_layers;
    int64_t vocab_size;
    int64_t max_position_embeddings;
    int64_t rms_norm_eps;            /* stored as float bits? no: float below */
    /* attention geometry */
    int64_t num_attention_heads;
    int64_t num_key_value_heads;
    int64_t head_dim;                /* rope base dimension */
    int64_t q_lora_rank;             /* MLA query latent */
    int64_t kv_lora_rank;            /* MLA key/value latent */
    int64_t qk_nope_head_dim;
    int64_t qk_rope_head_dim;
    int64_t value_head_dim;
    bool mla_use_output_gate;
    /* KDA (linear attention) geometry */
    int64_t linear_num_heads;
    int64_t linear_head_dim;       /* KDA head dim (linear_attn_config.head_dim) */
    int64_t short_conv_kernel_size;
    /* residual anchor mechanism */
    int64_t attn_res_block_size;
    /* MoE geometry */
    int64_t num_experts;
    int64_t num_experts_per_token;   /* top-k, never reduced (quality guard) */
    int64_t num_shared_experts;
    int64_t moe_intermediate_size;
    int64_t routed_expert_hidden_size;
    int64_t num_expert_groups;
    int64_t topk_groups;
    bool moe_renormalize;
    bool latent_moe_use_norm;
    float activation_situ_beta;
    float activation_situ_linear_beta;
    float routed_scaling_factor;
    float swiglu_limit;          /* GLM-5.3: clamp for the SwiGLU activation */
    float gate_lower_bound;
    float rms_norm_eps_f;
    /* layer schedule */
    cwick_layer_kind *layers;         /* length num_hidden_layers */
    int64_t kda_layers;
    int64_t mla_layers;
    int64_t dsa_layers;
    /* GLM-5.3 / next-gen fields (absent for legacy configs) */
    cwick_quant_kind quant;           /* actual released storage in the checkpoint */
    int64_t qk_head_dim;             /* full attention head dim (qk_nope + qk_rope) */
    int64_t index_head_dim;          /* DSA indexer head dim */
    int64_t index_n_heads;
    int64_t index_topk;
    int64_t index_kpool;
    bool index_kpool_always_select_tail;
    bool index_kpool_compress;
    bool indexer_rope_interleave;
    int64_t hc_mult;                 /* hyper-connection multiplier */
    float hc_eps;
    int64_t hc_sinkhorn_iters;
    bool mhc;                        /* multi-head chunking (hyper-connection) */
    int64_t first_k_dense_replace;   /* first N layers use dense MLP */
    int64_t num_nextn_predict_layers;
    bool norm_topk_prob;             /* MoE: normalize top-k probabilities */
    char scoring_func[32];           /* "sigmoid" | "softmax" */
    char topk_method[32];            /* "noaux_tc" | ... */
    bool act_silu;                   /* plain SiLU gating: silu(gate) * up */
    bool *mlp_dense;                 /* per-layer: true=dense MLP, false=sparse MoE */
    int64_t n_dense_layers;
    int64_t n_sparse_layers;
} cwick_model_spec;

/* Parse + strictly validate config.json. Fails closed on any quality-
 * reducing contract (reduced top-k, pruned experts, wrong dtype). */
cwick_status cwick_model_spec_load(const char *path, cwick_model_spec *spec,
                                 cwick_error *err);
void cwick_model_spec_free(cwick_model_spec *spec);

/* Forward declaration: the memory-bounded streaming reader (stream.c). The
 * spine owns one so routed experts can be fetched without letting the kernel
 * retain them. */
typedef struct cwick_stream cwick_stream;

/* Forward declaration: the host-memory budget decided at load time
 * (mem.c). Kept opaque here so storage code can hand it around without
 * pulling the planner's dependencies in. */
struct cwick_mem_plan;
typedef struct cwick_mem_plan cwick_mem_plan;

/* ------------------------------------------------------------------ */
/* Weight spine                                                        */
/* ------------------------------------------------------------------ */
typedef enum {
    CWICK_WEIGHT_BF16 = 1,             /* raw bfloat16 rows */
    CWICK_WEIGHT_F32 = 2,              /* raw float32 rows (biases / scores) */
    CWICK_WEIGHT_ROW_I8_F16 = 3,       /* row-int8 + per-row fp16 scale */
    CWICK_WEIGHT_FP8_E4M3 = 4,         /* FP8 E4M3 + F32 per-128x128 scale_inv */
} cwick_weight_kind;

/* FP8 E4M3 scale block size (rows x cols) for the HF `*_scale_inv` layout. */
#define CWICK_ST_BLOCK 128

typedef struct cwick_tensor {
    int64_t rows;                     /* output dim */
    int64_t cols;                     /* input dim */
    cwick_weight_kind kind;
    const void *data;                 /* mapped data (bf16 or int8) */
    const void *scales;               /* row-int8 only: fp16 per row */
    size_t data_bytes;
    size_t scale_bytes;
    char name[128];
} cwick_tensor;

/* Read one scalar from a small BF16/F32 tensor without assuming its storage
 * width. Matrices still use cwick_linear; this is for biases, gates and mHC
 * vectors whose dtype differs between HF exports and converted fixtures. */
float cwick_tensor_value(const cwick_tensor *t, int64_t index);

typedef struct cwick_spine {
    void **maps;                  /* array of void* (mmap base) */
    size_t *map_lens;             /* parallel to maps: bytes, for unload */
    size_t map_count;
    cwick_tensor *tensors;         /* dense spine tensors */
    size_t tensor_count;
    char **expert_files;          /* array of char* paths (converted format) */
    size_t expert_file_count;
    const char *root;
    bool have_int8;
    bool is_safetensors;          /* real HF checkpoint source */
    struct cwick_safetensors *saf; /* parsed HF shards when is_safetensors */
    cwick_stream *stream;          /* F_NOCACHE reader for streamed weights */
    uint64_t resident_bytes;      /* bytes that must stay in RAM */
    uint64_t streamed_bytes;      /* bytes read per decode step (experts) */
    cwick_mem_plan *mem_plan;      /* how this machine's RAM was split up */
} cwick_spine;

/* HF safetensors loader (the real GLM-5.3-Flash checkpoint source).
 *
 * The released checkpoint is a set of `model-*.safetensors` shards plus
 * `model.safetensors.index.json` mapping each tensor name to a shard. Each
 * shard starts with an 8-byte little-endian header length and a JSON header
 * naming every tensor with {dtype, shape, data_offsets}; tensor bytes begin
 * at 8 + header_len + offsets[0]. This loader mmaps every shard and resolves
 * tensors by name (binary search over a sorted index). FP8 E4M3 tensors carry
 * a paired F32 `*_scale_inv` tensor: one inverse scale per 128x128 block,
 * stored row-major as [rows/128, cols/128]. */
typedef enum {
    CWICK_ST_BF16 = 0,
    CWICK_ST_F32 = 1,
    CWICK_ST_F8_E4M3 = 2,
    CWICK_ST_U8 = 3,
    CWICK_ST_I8 = 4,       /* packed FP4 rows (the DeepSeek expert format) */
    CWICK_ST_F8_E8M0 = 5,  /* power-of-two scale bytes */
} cwick_st_dtype;

typedef struct cwick_st_tensor {
    int64_t rows, cols;
    cwick_st_dtype dtype;
    const void *data;             /* tensor bytes inside a mapped shard */
    size_t data_bytes;
    const void *scale_inv;        /* F32 [rows/128, cols/128], NULL if not FP8 */
    int64_t scale_inv_rows, scale_inv_cols;
    char name[192];
    const char *shard;            /* shard file name (for reporting) */
    int64_t shard_index;          /* index into cwick_safetensors.shard_names */
    uint64_t data_offset;         /* byte offset of the payload within the shard */
} cwick_st_tensor;

typedef struct cwick_safetensors {
    char *root;
    cwick_st_tensor *tensors;      /* one entry per tensor in the index */
    size_t tensor_count;
    int64_t *order;               /* sorted tensor indexes for binary search */
    void **maps;                  /* one mmap per shard */
    size_t *map_lens;             /* parallel to maps: file bytes */
    size_t map_count;
    char **shard_names;
    size_t shard_count;
    uint64_t total_bytes;         /* sum of all shard file sizes */
} cwick_safetensors;

cwick_status cwick_safetensors_open(const char *root, cwick_safetensors *saf,
                                  cwick_error *err);
void cwick_safetensors_close(cwick_safetensors *saf);
cwick_status cwick_safetensors_get(const cwick_safetensors *saf, const char *name,
                                 cwick_st_tensor *out, cwick_error *err);
int64_t cwick_safetensors_count_shards(const cwick_safetensors *saf);
uint64_t cwick_safetensors_total_bytes(const cwick_safetensors *saf);
/* Split into resident spine bytes and routed-expert (streamed) bytes. */
void cwick_safetensors_split(const cwick_safetensors *saf,
                            uint64_t *resident_bytes,
                            uint64_t *streamed_bytes);

/* ------------------------------------------------------------------ */
/* Memory-bounded SSD streaming (see stream.c)                         */
/*                                                                     */
/* A 306 GiB checkpoint cannot be allowed to become resident. The spine  */
/* that every token needs stays mapped and warm; everything streamed    */
/* (above all the routed expert bank) is read through F_NOCACHE         */
/* descriptors into a fixed arena so the kernel never retains it.       */
/* ------------------------------------------------------------------ */
typedef struct cwick_stream_options {
    size_t arena_bytes;          /* staging buffer for streamed reads (8 MiB) */
    bool nocache;                /* F_NOCACHE: keep streamed bytes out of RAM */
    size_t expert_cache_bytes;   /* opt-in LRU over raw expert blobs; 0 = off */
} cwick_stream_options;

/* Total physical memory of this host (0 when unknown). */
uint64_t cwick_host_memory_bytes(void);

cwick_status cwick_stream_open(const cwick_safetensors *saf,
                             const cwick_stream_options *opt,
                             cwick_stream **out, cwick_error *err);
void cwick_stream_close(cwick_stream *s);

/* Drop / warm mapped ranges: reclaim after a verification pass, warm the
 * resident spine at boot. */
void cwick_stream_reclaim(void *addr, size_t len);
void cwick_stream_warm(const void *addr, size_t len);

/* Read a tensor's payload through the streaming path. */
cwick_status cwick_stream_read(cwick_stream *s, const cwick_st_tensor *t,
                             void *dst, cwick_error *err);
cwick_status cwick_stream_locate(const cwick_stream *s, const cwick_st_tensor *t,
                               size_t *shard_index, uint64_t *offset,
                               cwick_error *err);

/* Read every byte of a tensor and fold it into `acc` (FNV-1a). This is what
 * makes `cwick doctor` a real load: O(arena) memory, whole-checkpoint cost. */
cwick_status cwick_stream_verify(cwick_stream *s, const cwick_st_tensor *t,
                               uint64_t *acc, cwick_error *err);

const void *cwick_stream_expert_cached(cwick_stream *s, int64_t layer,
                                      int64_t expert, int64_t matrix,
                                      size_t *bytes);
/* Reserve a writable cache slot for `bytes` and return its buffer, so the
 * caller can allocate every slot a layer needs on one thread and then spread
 * the preads over the worker pool. NULL when the budget cannot admit it. */
uint8_t *cwick_stream_expert_reserve(cwick_stream *s, int64_t layer,
                                    int64_t expert, int64_t matrix,
                                    size_t bytes);
/* Roll back a reserve whose read failed, so a half-filled blob is never
 * served as a cache hit. */
void cwick_stream_expert_abort(cwick_stream *s, int64_t layer, int64_t expert,
                              int64_t matrix);
/* The counterpart to the reserve above: the read that filled the slot has
 * landed, so the blob is ordinary cache data again and may be evicted. Every
 * reserved slot is either aborted (read failed) or settled (read completed),
 * and a pending slot is never an eviction candidate -- that is what keeps a
 * reserve's buffer from being handed to somebody else before its own read has
 * been dispatched. */
void cwick_stream_expert_settle(cwick_stream *s, int64_t layer, int64_t expert,
                               int64_t matrix);
/* Residency probe that does not move the hit/miss counters. Used to decide
 * whether a speculative SSD read is worth spending bandwidth on. */
int cwick_stream_expert_resident(cwick_stream *s, int64_t layer, int64_t expert,
                                int64_t matrix);
void cwick_stream_expert_put(cwick_stream *s, int64_t layer, int64_t expert,
                            int64_t matrix, const void *blob, size_t bytes);
const void *cwick_stream_expert_put_read(cwick_stream *s, int64_t layer,
                                        int64_t expert, int64_t matrix,
                                        const cwick_st_tensor *t,
                                        const cwick_st_tensor *st,
                                        cwick_error *err);
void cwick_stream_expert_decay(cwick_stream *s);

/* Ghost admission filter: returns 1 if this (layer, expert, matrix) has
 * earned a real cache slot (it has been requested-but-missing enough times
 * that pinning it pays), 0 if it should be read into a transient staging
 * buffer instead and left out of the pinned set. One-shot experts from
 * prefill and cold routes stay out of the budget, which is what lets the
 * pinned bytes concentrate on the experts decode actually reuses. */
int cwick_stream_expert_should_pin(cwick_stream *s, int64_t layer, int64_t expert,
                                  int64_t matrix);

/* Cross-layer expert prefetch (opt-in). A reader thread issues
 * F_NOCACHE preads for the expert matrices the engine predicts the next
 * layer will need, hiding read latency behind the current layer's compute. */
cwick_status cwick_stream_prefetch_open(cwick_stream *s, size_t arena_bytes,
                                      cwick_error *err);
void cwick_stream_prefetch_begin(cwick_stream *s);
void cwick_stream_prefetch_submit(cwick_stream *s, int64_t layer, int64_t expert,
                                 int64_t matrix, size_t bytes, size_t sbytes,
                                 size_t shard, uint64_t off, uint64_t soff);
const void *cwick_stream_prefetch_wait(cwick_stream *s, int64_t layer,
                                      int64_t expert, int64_t matrix);
void cwick_stream_prefetch_close(cwick_stream *s);

/* Prefetch gate: per-layer adaptive prefetch admission.
 * A layer whose predicted-expert recall EMA is below the threshold stops
 * issuing speculative reads (saving SSD bandwidth on the decode bottleneck)
 * while scoring continues so it can recover automatically. The gate never
 * changes routing — prefetch bytes are only an opportunistic cache. */
#define CWICK_PILOT_GATE_THRESHOLD 0.10f
#define CWICK_PILOT_GATE_WARMUP    16

/* Score a completed authoritative route against any outstanding prediction
 * for that layer. `pilot_ids` is the hint that was submitted; pass NULL to
 * score the prev-token predictor only. Called after the real router publishes
 * its authoritative ids. */
void cwick_stream_prefetch_score(cwick_stream *s, int64_t layer,
                                const int64_t *pilot_ids, int64_t n_pilot,
                                const int64_t *auth_ids, int64_t n_auth);
/* Record the submitted hint for a layer so it can be scored when the
 * authoritative route arrives. */
void cwick_stream_prefetch_hint_set(cwick_stream *s, int64_t layer,
                                   const int64_t *ids, int64_t n);
/* Remember this layer's authoritative ids for the prev-token predictor. */
void cwick_stream_prefetch_set_prev(cwick_stream *s, int64_t layer,
                                   const int64_t *ids, int64_t n);
/* Query whether speculative reads are currently admitted for a layer.
 * Returns 0 if reads should be issued, 1 if suppressed, -1 if unknown. */
int cwick_stream_prefetch_gated(cwick_stream *s, int64_t layer);

void cwick_stream_stats(const cwick_stream *s, uint64_t *bytes_streamed,
                       uint64_t *read_calls, uint64_t *cache_hits,
                       uint64_t *cache_misses);
size_t cwick_stream_arena_bytes(const cwick_stream *s);
bool cwick_stream_nocache_active(const cwick_stream *s);

/* ---- decode-step phase profiler (opt-in, CWICK_PROFILE=1) --------------
 * Attributes each decode step to the phase that actually consumes it, so
 * "why is a step 6 s" is answerable from inside the process instead of from
 * an Activity Monitor that shows the same idle GPU either way. Off by
 * default; every entry point is a load+test when CWICK_PROFILE is unset. */
typedef enum {
    CWICK_PH_STREAM = 0,   /* demand expert bytes: cache lookup + disk read */
    CWICK_PH_GEMV,         /* routed-expert dequant + GEMV                 */
    CWICK_PH_ATTENTION,    /* attention / KDA / norms                      */
    CWICK_PH_OTHER,        /* router, shared experts, elementwise glue     */
    CWICK_PH_COUNT
} cwick_prof_phase;

bool cwick_prof_on(void);
uint64_t cwick_prof_ns(void);            /* monotonic nanoseconds */
void cwick_prof_add(cwick_prof_phase p, uint64_t ns);
void cwick_prof_step(void);              /* count one decode step */
void cwick_prof_reset(void);
const char *cwick_prof_phase_name(cwick_prof_phase p);
/* Emit the breakdown at info level; `decode_ms` is the total decode wall
 * time the engine measured, used for the shares and for the leftover. */
void cwick_prof_report(const char *tag, double decode_ms);
/* Prefill's counterpart: no phase timers, only the fetch sub-counters. */
void cwick_prof_report_prefill(const char *tag, double prefill_ms,
                              uint64_t bytes_streamed);

/* ---- optional Metal decode backend (opt-in, CWICK_METAL=1) ---------------
 * Offloads the routed-expert FP8 GEMV (the biggest decode-step arithmetic
 * hotspot) to the GPU. Best-effort: returns 0 and the caller keeps the CPU
 * path if Metal is unavailable. `stride` lets one call cover n_rows rows of
 * a batched prefill; the kernel assumes k % 128 == 0 (CWICK_ST_BLOCK). */
/* Metal/GPU backend control: call once before any engine creation.
 * force=1 enables Metal (ignores CWICK_METAL env var), force=0 disables it.
 * Returns 0 on success, -1 if the Metal device could not be initialized. */
int cwick_metal_force(int force);
/* True when the Metal compute backend is up (device initialised and the
 * expert GEMV kernels usable). */
int cwick_metal_backend_active(void);
/* True when the routed-expert GEMV actually runs on the GPU: an explicit
 * --backend metal, or an autotune that measured Metal faster. This is what the
 * metrics `backend` label reports — "the device is up" and "the GPU is doing
 * the work" are different claims, and only the second one predicts tok/s. */
int cwick_metal_gemv_on_gpu(void);
int cwick_metal_gemv(const void *w_fp8, const float *scale_inv,
                    const float *x, float *out,
                    int64_t k, int64_t cols, int64_t ncb,
                    int64_t n_rows, int64_t stride);
/* Time the GPU and CPU FP8 expert GEMV on a real expert-shaped matrix and
 * pick the faster one. Call before the expert cache is allocated, so the
 * cache knows whether GPU-visible memory is worth wiring. */
void cwick_metal_autotune(void);
/* Bind the calling thread to a generation slot. Metal batches are per slot
 * (see cwick_metal_batch_begin_slot); the deep GEMV paths call the no-argument
 * begin/end and pick up the slot from here. Set once per request by the
 * server; anything that never sets it runs as slot 0. */
void cwick_metal_set_slot(int slot);

int cwick_metal_linear_bf16(const void *w_bf16, const float *x, float *out,
                           int64_t k, int64_t cols);
/* Batched resident-BF16 GEMV: group the independent projections of one
 * sublayer, pay one commit+wait for all of them. NULL from begin (or 0 from
 * add) means "not eligible" and the caller uses the serial path; end()
 * returns 0 when nothing was written and the group must be redone. */
void *cwick_metal_bf16_batch_begin(void);
int cwick_metal_bf16_batch_add(void *batch, const void *w_bf16,
                              const float *x, float *out,
                              int64_t k, int64_t cols);
int cwick_metal_bf16_batch_end(void *batch);

/* ---- batched GEMV ---------------------------------------------------
 * One Metal dispatch per expert matrix costs far more in launch latency than
 * in GPU work (measured here: 105 us of GPU against 295-724 us inside
 * waitUntilCompleted), and a decode step issues 1080 of them. The routed
 * group is two waves -- gate and up are independent, down needs both -- so a
 * whole layer fits in two command buffers instead of 24.
 *
 * begin() hands back a reusable batch, gemv() enqueues into it without
 * committing, end() commits once, waits once and copies every result back to
 * the CPU pointer it was given. Returns 0 if anything failed; the caller must
 * then redo the whole group on the CPU. abort() drops a batch that will not
 * be committed. */
void *cwick_metal_batch_begin(void);
int   cwick_metal_batch_gemv(void *batch, const void *w_fp8,
                            const float *scale_inv, const float *x,
                            float *out, int64_t k, int64_t cols, int64_t ncb,
                            int64_t n_rows, int64_t stride);
int   cwick_metal_batch_end(void *batch);
void  cwick_metal_batch_abort(void *batch);

/* ---- DeepSeek block-scaled GEMV batches -------------------------------
 * The DeepSeek backend quantizes every weight with one scale per 32
 * consecutive elements, either as e2m1 nibbles (routed and shared experts) or
 * as bf16 (every dense projection). These two batches run those GEMVs on the
 * GPU for a range of output rows, so the CPU pool can compute the rest of the
 * same matrix at the same time.
 *
 * Rows are independent and each row's summation order is fixed by the CPU
 * kernel, so the split cannot change the answer: it only decides who computed
 * which row. That property is what makes it safe to hand an arbitrary fraction
 * of every projection in the model to the GPU.
 *
 *   begin(slot, kind)  kind 0 = e2m1, 1 = bf16. Returns NULL when the GPU is
 *                      unavailable or the shader is missing.
 *   gemv(...)          enqueues rows [row0, row1) of an n-row matrix; `af` is
 *                      the dequantized activation (k floats) and `asf` its
 *                      32-element block scales (k/32 floats).
 *   commit()           encodes and submits without waiting, so the caller can
 *                      run its own share of the rows on the CPU pool first.
 *   wait()             waits, copies each item's rows to `out + row0`, and
 *                      returns 0 if the command buffer failed.
 * A false return from gemv() is not fatal: the caller keeps that matrix on the
 * CPU. */
void *cwick_metal_ds_batch_begin(int slot, int kind);
int   cwick_metal_ds_batch_gemv(void *batch, const void *w, const void *w_scale,
                                int64_t n, int64_t k, int64_t row0, int64_t row1,
                                const float *af, const float *asf, float *out);
int   cwick_metal_ds_batch_commit(void *batch);
int   cwick_metal_ds_batch_wait(void *batch);
int   cwick_metal_ds_ready(void);

/* GPU-visible allocation for streamed expert weights. On Apple Silicon a
 * shared MTLBuffer is ordinary RAM the GPU can also address, so reading
 * experts straight into one of these lets the GEMV bind them with an offset
 * instead of copying 8 MiB per call. Degrades to malloc when the backend is
 * off; free with cwick_shmem_free so the arena can reclaim the block. */
void  *cwick_shmem_alloc(size_t bytes);
void   cwick_shmem_free(void *p);
size_t cwick_shmem_arena_bytes(void);
/* Would cwick_shmem_alloc answer out of the GPU-visible arena right now? Used
 * by the stream layer to decide whether it may batch expert blobs into larger
 * blocks of its own, which is only safe for plain memory. */
int    cwick_shmem_gpu_visible(void);

/* Opt in to the fixed-size blob arena for streamed expert matrices. Only the
 * backend whose GEMVs run on the CPU pool wants this: for a GPU-visible blob
 * the Metal arena is the allocator, and batching blocks into larger mappings
 * changes how it grows. Call it before the first expert is reserved. */
void   cwick_stream_enable_blob_pool(cwick_stream *s);

/* Release the Metal device, command queue, pipeline and every GPU-visible
 * arena. Called once at shutdown, after all expert blobs have been returned;
 * safe to call when the backend was never brought up. */
void   cwick_metal_cleanup(void);

/* Give back one model's GPU-visible memory without tearing down the device:
 * the expert arenas and the no-copy views over its mapped weights. Called
 * when the resident model is replaced, because the outgoing model's arena is
 * otherwise still resident while the incoming one computes its budget -- the
 * buffers are wired MTLBuffers held in a malloc'd struct, so nothing else
 * frees them. Device, queue and pipeline states survive, so the next model
 * still gets the GPU. Safe to call when the backend is not up. */
void   cwick_metal_arena_release(void);

/* Release only the no-copy views over the model's mapped weights, keeping the
 * expert arenas. Must be called before the weights are unmapped: the views are
 * MTLBuffers over that address range. */
void   cwick_metal_views_release(void);

/* GPU keepalive: a background thread that dispatches a trivial kernel every
 * ~600 us so the GPU's power management does not downclock during the decode
 * loop's long CPU phases. Idempotent; returns 1 when the thread is running.
 * CWICK_KEEPALIVE=0 disables it, CWICK_KA_PERIOD=us sets the period. */
int    cwick_metal_keepalive_start(void);
void   cwick_metal_keepalive_stop(void);

/* ---- expert-cache accounting + the 80% steering valve ----------------
 * `cwick_stream_trim` / `cwick_stream_limit` give back pinned expert bytes
 * when measured RSS threatens the memory ceiling agreed at load time. */
size_t cwick_stream_cache_used(const cwick_stream *s);
size_t cwick_stream_cache_cap(const cwick_stream *s);
size_t cwick_stream_rss_bytes(const cwick_stream *s);
size_t cwick_stream_trim(cwick_stream *s, size_t target_cap);
void cwick_stream_limit(cwick_stream *s, size_t cache_cap);

/* How often (in decode steps) a loop compares its real physical footprint
 * with the memory plan and gives back pinned expert bytes when it drifted
 * over. */
#define CWICK_STEER_EVERY 1
/* The steering valve itself, lifted out of the GLM decode loop so the
 * DeepSeek loop can call it too: it takes a stream and a ceiling rather than
 * reaching into a cwick_engine. Returns the bytes it released (0 when the
 * footprint was inside the ceiling). */
size_t cwick_stream_steer_memory(cwick_stream *s, uint64_t ceiling, int64_t step,
                                 const char *tag);

typedef struct cwick_spine_index {
    int64_t input_layernorm;
    int64_t attn_res_norm;
    int64_t attn_res_proj;
    int64_t post_attn_norm;
    int64_t mlp_res_norm;
    int64_t mlp_res_proj;
    int64_t attn_a_log;
    int64_t attn_dt_bias;
    int64_t attn_q_conv;
    int64_t attn_k_conv;
    int64_t attn_v_conv;
    int64_t attn_o_norm;
    int64_t attn_q_proj;
    int64_t attn_k_proj;
    int64_t attn_v_proj;
    int64_t attn_g_proj;
    int64_t attn_fa_proj;
    int64_t attn_fb_proj;
    int64_t attn_b_proj;
    int64_t attn_o_proj;
    int64_t mla_qa_proj;
    int64_t mla_qb_proj;
    int64_t mla_kva_proj;
    int64_t mla_kvb_proj;
    int64_t mla_qa_norm;
    int64_t mla_kva_norm;
    int64_t dense_gate;
    int64_t dense_up;
    int64_t dense_down;
    int64_t moe_gate;
    int64_t moe_gate_bias;
    int64_t moe_routed_down;
    int64_t moe_routed_up;
    int64_t moe_routed_norm;
    int64_t moe_shared_gate;
    int64_t moe_shared_up;
    int64_t moe_shared_down;
    int64_t final_norm;
    int64_t output_res_norm;
    int64_t output_res_proj;
    int64_t lm_head;
    int64_t embedding;
} cwick_spine_index;

#define CWICK_TENSOR_MISSING ((int64_t)-1)

cwick_status cwick_spine_open(const char *root, const cwick_model_spec *spec,
                            const char *representation, /* "bf16" | "int8" */
                            cwick_spine *spine, cwick_error *err);
void cwick_spine_close(cwick_spine *spine);
int64_t cwick_spine_find(const cwick_spine *spine, const char *name);
/* The memory plan applied when this spine was opened (NULL for converted
 * models that are small enough to map whole). */
const cwick_mem_plan *cwick_spine_memory_plan(const cwick_spine *spine);

/* Expert corpus: one raw file per layer/expert, BF16 dense rows.
 * File name: expert-<layer>-<expert>.bin with a 20-byte header:
 *   magic "CWICKEXP1" (8 bytes), then u32 layer, u32 expert, u32 matrix_count.
 * Body: three matrices in torch linear layout ([out,in] rows):
 *   w1 [I, RH] bf16, w2 [RH, I] bf16, w3 [I, RH] bf16.
 */
#define CWICK_EXPERT_MAGIC "CWICKEXP1"
#define CWICK_EXPERT_MATRICES 3

cwick_status cwick_expert_load(const cwick_spine *spine, int64_t layer,
                             int64_t expert, const cwick_model_spec *spec,
                             float *w1, float *w2, float *w3,
                             cwick_error *err);

/* On-disk safetensors name of one routed-expert matrix, in this checkpoint's
 * naming family. Every consumer (spine completeness check, streaming reader,
 * prefetch, doctor) builds expert names through here. */
void cwick_expert_saf_name(const cwick_model_spec *spec, int64_t layer,
                           int64_t expert, const char *matrix,
                           char *out, size_t cap);

/* ------------------------------------------------------------------ */
/* Persistent fork-join worker pool (pool.c)                           */
/* ------------------------------------------------------------------ */
/* A decode step issues ~1400 GEMV/GEMM calls. Without this every one of
 * them ran on the caller's thread, leaving 11 of 12 performance cores idle
 * while a 306 GiB checkpoint streamed off the SSD. See pool.c.
 *
 * cwick_pool_for() splits [0, n) into contiguous slices and runs fn(arg, lo,
 * hi) on each worker. It is safe to call at any time: when the pool is not
 * up (not yet initialized, already shut down, or CWICK_THREADS=1) it runs the
 * whole range inline, so results are bit-identical either way. Returns the
 * number of threads that took part (0 = serial). */
typedef void (*cwick_pool_range_fn)(void *arg, int64_t lo, int64_t hi);
int cwick_pool_init(void);
int cwick_pool_threads(void);
void cwick_pool_shutdown(void);
/* Dispatch for a small number of large jobs (see pool.c). CWICK_POOL_MIN_WORK
 * would send 24 SSD reads of 8 MiB each down the inline path. */
int cwick_pool_for_items(int64_t n, void *arg, cwick_pool_range_fn fn);
int cwick_pool_for(int64_t n, void *arg, cwick_pool_range_fn fn);

/* ------------------------------------------------------------------ */
/* Math core (Apple Silicon)                                           */
/* ------------------------------------------------------------------ */
/* Dense linear: out[row][k] = sum_j in[row][j] * W[k][j]
 * (W is row-major [k, j] = torch linear layout). */
void cwick_gemm_bf16(float *out, int64_t rows, int64_t k, int64_t cols,
                    const float *in, const void *w_bf16);
void cwick_gemm_int8(float *out, int64_t rows, int64_t k, int64_t cols,
                    const float *in, const int8_t *w, const uint16_t *scales);
void cwick_gemm_f32(float *out, int64_t rows, int64_t k, int64_t cols,
                   const float *in, const float *w);
void cwick_gemm_fp8(float *out, int64_t rows, int64_t k, int64_t cols,
                   const float *in, const void *w_fp8, const float *scale_inv,
                   int64_t nrb, int64_t ncb);

/* Streaming GEMV (rows == 1) — the decode path. These never materialize the
 * weight matrix; they dequantize one tile at a time, so a [154880, 4096]
 * lm_head or a [8192, 4096] projection costs O(tile) extra memory instead of
 * 2.5 GiB / 134 MiB. Required to run a 306 GiB checkpoint inside a bounded
 * memory budget. */
void cwick_gemv_bf16(float *out, int64_t k, int64_t cols, const float *in,
                    const void *w_bf16);
void cwick_gemv_fp8_rows(float *out, int64_t rows, int64_t k, int64_t cols,
                        const float *in, const void *w_fp8,
                        const float *scale_inv, int64_t ncb);
/* Stop cwick_gemv_fp8_rows from retrying the GPU one matrix at a time. The
 * batched prefill group sets this while it redoes a group on the CPU. */
void cwick_prefill_metal_suspend(bool on);
void cwick_gemv_fp8(float *out, int64_t k, int64_t cols, const float *in,
                   const void *w_fp8, const float *scale_inv, int64_t ncb);
/* Output rows [c0, c1) only, for a caller that already owns the
 * parallelism. See cwick_gemv_fp8_sub() in tensor.c for why it exists. */
void cwick_gemv_fp8_sub(float *out, int64_t k, int64_t cols, int64_t c0, int64_t c1,
                       const float *in, const void *w_fp8, const float *scale_inv,
                       int64_t ncb);
/* Row-int8 GEMV (the --spine int8 decode path). */
void cwick_gemv_int8(float *out, int64_t k, int64_t cols, const float *in,
                    const int8_t *w, const uint16_t *scales);

/* FP8 E4M3 -> f32 (OCP: 1 sign, 4 exponent bits bias 7, 3 mantissa bits,
 * subnormal step 2^-9). */
float cwick_fp8_e4m3_to_f32(uint8_t b);

/* Linear with a spine tensor (BF16, f32, row-int8, or FP8 E4M3). */
void cwick_linear(float *out, const float *in, int64_t rows, int64_t cols_in,
                 int64_t cols_out, const cwick_tensor *t);

/* MXFP4 expert GEMV: packed e2m1 nibbles + e8m0 scale per 32 values. */
void mxfp4_gemv_rows(const uint8_t *p, const uint8_t *s, const float *x,
                     float *y, int row0, int row1, int cols);
void mxfp4_gemv_rows_ref(const uint8_t *p, const uint8_t *s, const float *x,
                         float *y, int row0, int row1, int cols);

void cwick_rms_norm(float *out, const float *in, int64_t n,
                   const float *weight, float eps);
void cwick_softmax(float *scores, int64_t n);
void cwick_silu_vec(float *x, int64_t n);
/* GLM-5.3 SwiGLU with clamp: gate clamped above only, up clamped symmetric. */
void cwick_swiglu_clamp(float *out, const float *gate, const float *up,
                       int64_t n, float limit);
void cwick_situ(float *out, const float *gate, const float *up, int64_t n,
               float beta, float linear_beta);

/* ------------------------------------------------------------------ */
/* DSA indexer (GLM-5.3 sparse attention)                              */
/* ------------------------------------------------------------------ */
/* The 11 `deepseek_sparse_attention` layers do not attend to the whole
 * prefix: the checkpoint's indexer picks `index_topk` (2048) keys for each
 * query. Inside that window "top-k of the prefix" is the whole prefix, so a
 * dense pass is exactly what the checkpoint asks for -- which is why a
 * context of up to index_topk needs no indexer at all. Beyond it, running
 * dense is a different algorithm, not a slower one, so the indexer is what
 * makes a 100K window both correct and affordable.
 *
 * Structure:
 *   - a key cache compressed by `index_kpool` (4 consecutive positions ->
 *     one 128-dim row, gated softmax over the pool with a learned APE bias),
 *   - a query projected from the *normalised attention query latent*,
 *   - one learned weight per indexer head; the per-head dot products are
 *     ReLU'd and summed into a score, and the top-scoring pools are selected.
 * GLM-5.3 has no RoPE at all (qk_rope_head_dim == 0), so neither the indexer
 * nor the MLA path rotates anything. */
typedef struct cwick_indexer_weights {
    const cwick_tensor *wq_b;         /* [index_n_heads*index_head_dim, q_lora_rank] */
    const cwick_tensor *wk;           /* [index_head_dim, hidden] */
    const cwick_tensor *gate;         /* [index_head_dim, hidden] */
    const cwick_tensor *ape;          /* bf16 [index_kpool, index_head_dim] */
    const cwick_tensor *k_norm;       /* [index_head_dim] */
    const cwick_tensor *k_norm_b;     /* [index_head_dim] */
    const cwick_tensor *weights_proj; /* [index_n_heads, hidden] */
} cwick_indexer_weights;

/* Per-layer indexer cache, owned by the engine and reset with the KV cache.
 * `pool` holds the complete compressed rows; `tail_*` hold the raw key/gate
 * rows of the pool still being filled. */
typedef struct cwick_indexer_state {
    float *pool;     /* [pool_cap * index_head_dim] */
    float *tail_k;   /* [index_kpool * index_head_dim] */
    float *tail_g;   /* [index_kpool * index_head_dim] */
    int64_t *sel;    /* scratch: selected positions [index_topk + index_kpool] */
    float *q;        /* scratch: [index_n_heads * index_head_dim] */
    float *wt;       /* scratch: [index_n_heads] */
    float *sc;       /* scratch: [pool_cap] per-pool scores */
    float *sc2;      /* scratch: [pool_cap] permutable copy for quickselect */
    int64_t pool_cap;
    int64_t n_pools; /* complete compressed rows written so far */
} cwick_indexer_state;

/* Precision of the MLA latent cache. Every attention step streams it twice
 * (score pass, accumulation pass), so its width is the bandwidth the
 * attention loop spends; halving it is the only lever left now that the
 * reductions are known to be bandwidth-bound. Engine, prefix cache and
 * attention all ask this one function so they can never disagree.
 * CWICK_LATENT_BF16=0 restores fp32 for an A/B. */
bool cwick_latent_bf16_enabled(void);
size_t cwick_latent_elem_bytes(void);

/* Bytes of indexer cache one DSA layer needs for a `capacity`-token window. */
uint64_t cwick_indexer_bytes(const cwick_model_spec *spec, int64_t capacity);
void cwick_indexer_reset(const cwick_model_spec *spec, cwick_indexer_state *st);

/* Attention decode steps (see attention.c). */
/* MLA/DSA decode. `g` (output gate) may be NULL: the released GLM-5.3
 * checkpoint has no gate projection on its sparse-attention layers. With
 * qk_rope_head_dim == 0 (mla_nope) the rope slice is empty.
 *
 * `values == NULL` selects the MLA *latent* cache: `keys` then holds
 * [layers][capacity][kv_lora_rank] normalised KV latents instead of
 * materialised keys, and the kernel absorbs kv_b_proj into the query and the
 * output. See the layout note in attention.c -- the result is identical up to
 * summation order, and the cache is 64x smaller, which is what makes a 100K
 * context window fit. `values != NULL` keeps the expanded cache. */
bool cwick_mla_latent_enabled(const cwick_model_spec *spec,
                             const cwick_tensor *kvb);
void cwick_attention_mla_decode(const cwick_model_spec *spec,
                               const cwick_tensor *qa, const cwick_tensor *qb,
                               const cwick_tensor *kva, const cwick_tensor *kvb,
                               const cwick_tensor *g, const cwick_tensor *o,
                               const cwick_tensor *qa_n, const cwick_tensor *kva_n,
                               float *keys, float *values, int64_t capacity,
                               int64_t pos, int64_t layer,
                               const cwick_indexer_weights *idx_w,
                               cwick_indexer_state *idx,
                               const float *x, float *out, cwick_error *err);

/* One sparse-attention layer for a whole prefill chunk at once: the four
 * projections become one GEMM per weight instead of one GEMV per token (a
 * GEMV re-reads the weight matrix, which is the dominant cost of a long
 * prefill). `X` is n*H attention inputs, `Out` n*H outputs.
 * Returns CWICK_ERR_UNSUPPORTED when the layer is not in the layout this path
 * covers, so the caller can fall back to the per-token loop. */
bool cwick_mla_chunk_supported(const cwick_model_spec *spec,
                              const cwick_tensor *qa, const cwick_tensor *qb,
                              const cwick_tensor *kva, const cwick_tensor *kvb,
                              const cwick_tensor *o, const float *values);

cwick_status cwick_attention_mla_prefill_chunk(const cwick_model_spec *spec,
                                             const cwick_tensor *qa,
                                             const cwick_tensor *qb,
                                             const cwick_tensor *kva,
                                             const cwick_tensor *kvb,
                                             const cwick_tensor *o,
                                             const cwick_tensor *qa_n,
                                             const cwick_tensor *kva_n,
                                             float *keys, float *values,
                                             int64_t capacity, int64_t pos0,
                                             int64_t n, int64_t layer,
                                             const cwick_indexer_weights *idx_w,
                                             cwick_indexer_state *idx,
                                             const float *X, float *Out,
                                             cwick_error *err);

/* True when this layer needs an indexer for a prefix of `npos` tokens, i.e.
 * when the prefix is longer than the checkpoint's top-k window. */
bool cwick_indexer_active(const cwick_model_spec *spec, int64_t npos);
/* KDA decode. `ga`/`gb` are the two-stage output gate; `on` is a *gated*
 * RMSNorm weight (norm output * sigmoid(gb(ga(x)))). `a_log` has one entry
 * per head, not per channel. */
void cwick_attention_kda_decode(const cwick_model_spec *spec,
                               const cwick_tensor *q_proj, const cwick_tensor *k_proj,
                               const cwick_tensor *v_proj, const cwick_tensor *q_conv,
                               const cwick_tensor *k_conv, const cwick_tensor *v_conv,
                               const cwick_tensor *a_log, const cwick_tensor *dt_bias,
                               const cwick_tensor *o_norm, const cwick_tensor *fa_proj,
                               const cwick_tensor *fb_proj, const cwick_tensor *b_proj,
                               const cwick_tensor *ga_proj, const cwick_tensor *gb_proj,
                               const cwick_tensor *o_proj,
                               float *rec, float *conv, int64_t layer,
                               const float *x, float *out, cwick_error *err);

/* MoE / dense-MLP forwards (see moe.c). */
void cwick_moe_forward(const cwick_model_spec *spec, const cwick_spine *spine,
                      int64_t layer, const float *hidden, float *out,
                      cwick_error *err);
/* Batched MoE: `B` tokens through `layer`, each routed expert loaded ONCE and
 * reused across every token that selected it. `hidden`/`out` are B*H layouts.
 * Math is identical to cwick_moe_forward (this is what makes prefill fast: a
 * token-serial pass re-streams every expert matrix from disk per token).
 *
 * `progress`, when non-NULL, is called from inside the expert loops with the
 * layer and how far the MoE phase has advanced (done/total: experts in the
 * routed loops). The batched MoE phase is the longest silent stretch of a
 * prefill (~100 s/layer on a 30K-token prompt); the engine passes a heartbeat
 * through here so the log keeps reporting while it runs. */
typedef void (*cwick_moe_progress_fn)(void *user, int64_t layer,
                                     int64_t done, int64_t total);
void cwick_moe_forward_batch(const cwick_model_spec *spec, const cwick_spine *spine,
                            int64_t layer, const float *hidden, int64_t B,
                            float *out, cwick_error *err,
                            cwick_moe_progress_fn progress, void *progress_user);
void cwick_dense_mlp(const cwick_model_spec *spec, const cwick_spine *spine,
                    int64_t layer, const float *hidden, float *out,
                    cwick_error *err);
/* Router prediction: top-k expert ids for a layer given its
 * post-attention residual (see moe.c). Returns 0, or -1 if absent. */
int cwick_router_predict(const cwick_model_spec *spec, const cwick_spine *spine,
                        int64_t layer, const float *hidden, int64_t *ids_out);

/* ------------------------------------------------------------------ */
/* Engine                                                              */
/* ------------------------------------------------------------------ */
typedef struct cwick_kv_cache {
    float *keys;      /* [layers * heads * kv_lora_rope? -> see engine] */
    float *values;
    int64_t *positions;  /* per-layer position of last stored token */
    int64_t capacity;
    int64_t len;
} cwick_kv_cache;

typedef struct cwick_engine cwick_engine;

typedef struct cwick_gen_result {
    uint32_t *tokens;
    int64_t n_tokens;
    bool stop_reason_length;
    /* Timing, filled in by the engine so a caller (the server's /metrics
     * registry) can report prefill/decode without re-deriving them from its
     * own clock and getting the streaming commit time mixed in. Zero when the
     * generation failed before either phase ran. */
    double prefill_ms;
    double decode_ms;
    int64_t decode_steps;   /* real forward steps: max_new-1, see the note in
                             * engine.c about the prefill logits */
    bool interrupted;       /* Ctrl+C / stop flag */
    bool sink_stopped;      /* the consumer hung up mid-stream */
    bool loop_stopped;      /* the loop guard ended a repetition attractor */
    bool think_budget_hit;  /* the engine closed the think block for the model
                             * because the thinking budget ran out */
    /* The ceiling that was actually in force, after the clamp to the output
     * budget (-1 when the mechanism was off). The request asked for a level;
     * this is what the level turned into, so logs can report the number that
     * really capped the turn instead of the one the client named. */
    int64_t think_budget_applied;
} cwick_gen_result;

/* How the decode loop turns logits into a token.
 *
 * The default is greedy (temperature 0), which is what this engine always did
 * and what `temperature: 0` still asks for. Anything else is honoured: a
 * checkpoint whose generation_config.json says do_sample=true / temperature
 * 1.0 / top_p 0.95 / top_k 20 is served with those numbers unless the request
 * overrides them, because greedy decoding of a sampling-trained checkpoint
 * collapses into repetition loops.
 *
 * Penalties use the OpenAI shape and are counted over the prompt tokens plus
 * the tokens generated so far: logit -= presence * 1 + frequency * count. */
typedef struct cwick_sampler_cfg {
    float temperature;        /* <= 0 -> argmax; > 0 -> scale logits by 1/T */
    float top_p;              /* (0,1] nucleus cutoff; 1.0 = off */
    int   top_k;              /* keep only the k best; <= 0 = off */
    float presence_penalty;   /* subtracted once per distinct token */
    float frequency_penalty;  /* subtracted once per occurrence */
} cwick_sampler_cfg;

typedef struct cwick_engine_options {
    const char *model_root;
    const char *spine;       /* "bf16" or "int8" */
    int64_t max_new_tokens;
    int64_t seed;
    /* KV-cache / context window size in tokens. 0 (default) keeps the engine's
     * memory-safe ceiling (4096, or the DSA indexer top-k for FP8). Any value
     * up to the checkpoint's max_position_embeddings is honoured, letting a
     * long-context checkpoint (GLM-5.3 supports 1M) serve requests longer than
     * the default window — at the cost of a proportionally larger KV cache. */
    int64_t max_context;
    bool verbose;
    /* Serve several requests at once. Slots share one weight spine: the
     * struct is all pointers, so a borrower copies it and gets the mmaps,
     * the shard index, the F_NOCACHE stream and the expert cache without
     * paying for or re-wiring them. Per-sequence state (KV, KDA recurrence,
     * indexer) is allocated per slot. `shared_spine` must outlive the
     * borrowing engine; slot 0 is always the owner and is freed last. */
    int  slot_id;                    /* which generation slot: 0..N-1 */
    const cwick_spine *shared_spine;  /* NULL = open the weights ourselves */
    bool synthetic;          /* generate a tiny deterministic model */
    int64_t synthetic_hidden;
    int64_t synthetic_layers;
    int64_t synthetic_experts;
    int64_t synthetic_vocab;
    /* Optional Ctrl+C flag. Polled between prefill/decode steps, so `serve`
     * can leave a multi-second decode of a 320B model immediately instead of
     * waiting for the answer to finish. Set it once for the whole run with
     * cwick_engine_set_stop(). */
    volatile int *stop;
} cwick_engine_options;

cwick_status cwick_engine_new(const cwick_engine_options *opt, cwick_engine **out,
                            cwick_error *err);
void cwick_engine_free(cwick_engine *eng);
const cwick_model_spec *cwick_engine_spec(const cwick_engine *eng);
/* The KV window this engine actually runs with: --context when given, else the
 * checkpoint default (top-k for an FP8 checkpoint, max_position_embeddings
 * otherwise). This — not the checkpoint's ceiling — is what /metrics reports. */
int64_t cwick_engine_context_window(const cwick_engine *eng);
const cwick_spine *cwick_engine_spine(const cwick_engine *eng);
/* Expert pin cache of the resident backend, or NULL when it is off. */
cwick_stream *cwick_engine_expert_stream(const cwick_engine *eng);
const char *cwick_engine_model_root(const cwick_engine *eng);
/* KV positions this engine was built with, so a second generation slot can be
 * opened with the same context window instead of the default one. */
int64_t cwick_engine_capacity(const cwick_engine *eng);
/* Positions of context the engine currently holds: the prompt it has consumed
 * plus every token it has generated since. A caller that wants to snapshot the
 * *conversation* (not just the prompt) needs this to check that the token
 * sequence it is about to key the snapshot on is the one the state describes. */
int64_t cwick_engine_pos(const cwick_engine *eng);
/* Wire an interrupt flag (owned by the caller, usually set from a SIGINT
 * handler) into the decode loop. */
void cwick_engine_set_stop(cwick_engine *eng, volatile int *stop);

/* Extra interrupt source polled wherever `stop` is polled. Used by the server
 * to notice a client that disconnected while its response was still being
 * generated -- otherwise the generation keeps running and holds the
 * generation slot for the whole remaining budget. */
void cwick_engine_set_abort_poll(cwick_engine *eng, bool (*poll)(void *user),
                                void *user);
void cwick_engine_set_verbose(cwick_engine *eng, bool verbose);
cwick_status cwick_engine_reset(cwick_engine *eng, cwick_error *err);

/* Bind this engine's log output to a request tag ("req 12"). Every engine-side
 * progress line -- prefill token/layer heartbeat, prefill-moe heartbeat,
 * per-step decode timing, the generation summary -- is then tagged with the
 * request being processed, so a log line can always be attributed. `tag == NULL`
 * (or "") returns to the untagged "gen" marker used by the CLI. */
void cwick_engine_set_req_tag(cwick_engine *eng, const char *tag);

/* Tell the engine which part of the answer is being decoded right now --
 * "thinking", "content" or "tools" (static strings; NULL clears the hint).
 * The server classifies each token with the state machine that splits the
 * SSE stream and publishes the label here, so the per-step decode progress
 * lines read `decode(thinking) forward 123/23999 ...` instead of a bare
 * `decode`. A request that never got a hint (or a CLI run) keeps the
 * unlabelled form. */
void cwick_engine_set_decode_phase(cwick_engine *eng, const char *phase);

/* Ask the engine to snapshot the state right after its next prefill into
 * `out` (see pfx.h). `out == NULL` cancels. One capture per request: the
 * engine clears the pointer once it has filled it, so the caller owns the
 * state from then on and frees it with cwick_prefix_state_free.
 *
 * `done`, when set, is invoked the instant the snapshot is filled -- i.e.
 * while the prefill that produced it is still the request's present. The
 * snapshot must be persisted there, not when generation returns: a 30K-token
 * prompt spends hours in prefill and hours more in decode, and a snapshot
 * that only becomes durable at the end is lost to every abort, disconnect
 * and kill in between. `fn` takes ownership of the state (frees it). */
struct cwick_prefix_state;
void cwick_engine_set_prefix_capture(cwick_engine *eng,
                                    struct cwick_prefix_state *out);
void cwick_engine_set_prefix_capture_done(
    cwick_engine *eng, void (*fn)(void *user, struct cwick_prefix_state *s),
    void *user);

/* Also snapshot every `block` tokens *during* the prefill, not only once at
 * the end, and cap the prefill chunk so chunk boundaries land on multiples of
 * `block`. A chunk boundary is the only point where the state is a legal
 * prefix (the recurrent parts can be snapshotted but not rewound), so this is
 * what lets a prompt that shares only its first 12K tokens with a cached
 * prompt replay those 12K instead of re-prefilling everything. Costs one
 * extra expert pass per additional chunk (~58 s here) and one file per chunk.
 * 0 restores "one snapshot at the end". */
void cwick_engine_set_prefix_capture_block(cwick_engine *eng, int64_t block);

/* Tell the engine the prompt's full token count when a cached prefix was
 * replayed before the call (engine-side `prompt_len` is only the unmasked
 * tail). Used by the summary line so it reports the real request size. */
void cwick_engine_set_prompt_total(cwick_engine *eng, int64_t total);

/* Choose how the next generation turns logits into a token (see
 * cwick_sampler_cfg). It is a property of the engine, not of the call, so the
 * streaming and non-streaming entry points and both backends (GLM and
 * DeepSeek) can share one code path. NULL restores the default (greedy).
 *
 * A request that says nothing gets the checkpoint's generation_config.json, so
 * the served behaviour matches how the weights were trained; a request that
 * sends temperature/top_p/top_k/penalties gets those, and is refused (400)
 * when they are out of range rather than silently clamped. */
void cwick_engine_set_sampler(cwick_engine *eng, const cwick_sampler_cfg *cfg);
void cwick_engine_sampler_default(cwick_sampler_cfg *out);
/* What the checkpoint's generation_config.json asked for, before any request
 * touched it. The server starts from this and overrides only the fields the
 * request sent. False when there was no such file (out is then greedy). */
bool cwick_engine_sampler_checkpoint(const cwick_engine *eng,
                                     cwick_sampler_cfg *out);

/* Run one decode transaction over the given prompt (token ids).
 * Fills result->tokens with prompt + generated, or only generated when
 * `keep_prompt` is false. Greedy by default; see cwick_engine_set_sampler. */
cwick_status cwick_engine_generate(cwick_engine *eng, const uint32_t *prompt,
                                 int64_t prompt_len, int64_t max_new,
                                 bool keep_prompt, cwick_gen_result *result,
                                 cwick_error *err);
void cwick_gen_result_free(cwick_gen_result *result);

/* Advance R sequences by one token each with a SHARED expert fetch: every row
 * does its own mHC mixing and attention (those are per-sequence state), then
 * all rows meet for one cwick_moe_forward_batch call, then they separate again.
 * Same arithmetic, one fetch instead of R.
 *
 * Not wired to a generate loop yet: batching R rows pays one expert fetch
 * instead of R, and the measured headroom (the fetch is the dominant phase of
 * a streaming step) is what is left to reclaim. R == 1 falls through to the plain
 * single-sequence forward, so calling it with one row is always safe. */
cwick_status cwick_engine_generate_group(cwick_engine **engs,
                                       const uint32_t *const *prompts,
                                       const int64_t *prompt_lens,
                                       int64_t R, int64_t max_new,
                                       bool keep_prompt,
                                       cwick_gen_result *results,
                                       cwick_error *err);
cwick_status cwick_forward_group(cwick_engine **engs, const uint32_t *tokens,
                               int64_t R, float **logits, cwick_error *err);

/* Called for each generated token right after it is decoded (before the next,
 * expensive forward step), so a server can push it to the client immediately.
 * `index` is the 0-based index within this response. Return false to stop
 * generation early -- e.g. the client hung up. */
typedef bool (*cwick_token_sink)(void *user, uint32_t token, int64_t index);

/* Like cwick_engine_generate, but reports every token through `sink` as it is
 * produced (incremental streaming). Tokens already produced are still returned
 * in `result`; a sink that returns false ends the turn with finish_reason
 * "stop" rather than an error. Explicit prompt tokens are never echoed. */
cwick_status cwick_engine_generate_stream(cwick_engine *eng, const uint32_t *prompt,
                                        int64_t prompt_len, int64_t max_new,
                                        cwick_token_sink sink, void *user,
                                        cwick_gen_result *result, cwick_error *err);

/* Run only the prefill and copy the next-token logits (vocab_size floats) into
 * `logits_out`. Used to diff the batched and token-serial prefills. Honours
 * CWICK_NO_BATCH_PREFILL / CWICK_PREFILL_CHUNK. */
cwick_status cwick_engine_prefill_logits(cwick_engine *eng, const uint32_t *prompt,
                                       int64_t prompt_len, float *logits_out,
                                       cwick_error *err);

/* Same, but starting at absolute position `start_pos`.
 *
 * `start_pos == 0` resets the engine and is what the wrapper above does.
 * `start_pos > 0` keeps the KV/latent state already written for [0, start_pos)
 * and overwrites everything after it, which is how one request answers several
 * questions about the same state without prefilling that state N times. The
 * caller owns the correctness of the prefix: it must be the exact token prefix
 * the state was built from.
 *
 * Two things make this safe on all three backends. GLM's forward_prefill
 * derives each token's absolute position from `eng->pos` (engine.c:1882), and
 * the DeepSeek/MiniMax forwards take `pos` as an argument, so a resume is the
 * same code path as a fresh prefill with a non-zero origin. mHC state is
 * intra-token, so nothing outside the KV survives a token boundary anyway. */
cwick_status cwick_engine_prefill_logits_at(cwick_engine *eng,
                                            const uint32_t *prompt,
                                            int64_t prompt_len, int64_t start_pos,
                                            float *logits_out, cwick_error *err);

/* ------------------------------------------------------------------ */
/* Tokenizer (BPE over vocab.json + merges.txt, byte fallback)         */
/* ------------------------------------------------------------------ */
typedef struct cwick_vocab_entry {
    char *key;      /* token string (or byte fallback marker) */
    int64_t id;
    bool is_byte;
} cwick_vocab_entry;

typedef struct cwick_merge_entry {
    char *pair;     /* "a b" merge rule */
} cwick_merge_entry;

typedef struct cwick_tokenizer {
    cwick_vocab_entry *vocab;
    size_t vocab_size;
    cwick_merge_entry *merges;
    size_t merge_count;
    int64_t *byte_to_id; /* 256 entries, -1 if absent */
    bool byte_fallback;
    bool byte_level;     /* GPT-2 byte-level BPE (GLM tokenizer.json) */
    /* Which pre-tokenizer regex this checkpoint ships. The GLM one and the
     * DeepSeek one differ in digit runs, CJK isolation and the punctuation
     * class, and the difference changes the token stream. */
    bool pretok_ds;
    int64_t vocab_cap;
    int64_t *index;      /* open-addressed token-string index (hash -> vocab slot) */
    size_t index_mask;
    /* added_tokens of the checkpoint: <sop>, <|system|>, <think>, … These are
     * single ids the model was trained on, so encoding splits text on them
     * instead of byte-pair-encoding their characters. Sorted longest-first. */
    cwick_vocab_entry *special;
    size_t special_count;
    bool special_first[256];   /* first-byte bloom: is a special worth trying? */
    int64_t *id_to_slot;   /* vocab id -> vocab slot (decode, O(1)) */
    int64_t id_to_slot_cap;
    /* merge rules resolved to (left id, right id) -> (rank, merged id) */
    uint64_t *pair_keys;
    int32_t *pair_rank;
    int64_t *pair_ab;
    size_t pair_count;
    size_t pair_mask;
} cwick_tokenizer;

cwick_status cwick_tokenizer_load(const char *dir, const cwick_model_spec *spec,
                                cwick_tokenizer **tok, cwick_error *err);
void cwick_tokenizer_free(cwick_tokenizer *tok);
cwick_status cwick_tokenizer_encode(const cwick_tokenizer *tok, const char *text,
                                  uint32_t **ids, int64_t *n, cwick_error *err);
cwick_status cwick_tokenizer_decode(const cwick_tokenizer *tok, const uint32_t *ids,
                                  int64_t n, char **out, cwick_error *err);

/* ------------------------------------------------------------------ */
/* Chat template                                                       */
/* ------------------------------------------------------------------ */
/* One rendered turn of a conversation. `tool_calls_json` is the raw OpenAI
 * `tool_calls` array an assistant message may carry (may be NULL). */
typedef struct cwick_chat_msg {
    const char *role;            /* system | user | assistant | tool */
    const char *content;
    const char *tool_calls_json;
} cwick_chat_msg;

/* Render the full conversation in the checkpoint's own chat format and
 * encode it. `tools_block` is the rendered <tools> section (or NULL) and
 * `reasoning_effort` is one of "Max" / "Low" / "High" (NULL -> "Max"). */
cwick_status cwick_chat_encode_messages(const cwick_tokenizer *tok,
                                      const cwick_chat_msg *msgs, size_t n,
                                      const char *tools_block,
                                      const char *reasoning_effort,
                                      uint32_t **ids, int64_t *n_out,
                                      cwick_error *err);
cwick_status cwick_chat_encode(const cwick_tokenizer *tok, const char *system,
                             const char *user, uint32_t **ids, int64_t *n,
                             cwick_error *err);
/* Same, with declared tools spliced into the system turn using the
 * checkpoint's own <tools>/<tool_call> protocol (see tools.c). Pass
 * tools_block == NULL for the plain chat prefix. */
cwick_status cwick_chat_encode_tools(const cwick_tokenizer *tok, const char *system,
                                   const char *user, const char *tools_block,
                                   uint32_t **ids, int64_t *n, cwick_error *err);

/* One turn whose assistant header answers instead of thinking: the reasoning
 * block is closed before the model writes anything, so the next token is the
 * answer. The decision endpoint reads that one position's distribution and
 * never decodes, which is why it needs this form rather than the chat one. */
cwick_status cwick_chat_encode_decision(const cwick_tokenizer *tok,
                                        const char *system, const char *user,
                                        uint32_t **ids, int64_t *n,
                                        cwick_error *err);

/* Synthetic model generator (see setup.c). */
cwick_status cwick_setup_synthetic(const char *root, cwick_error *err);
/* Synthetic GLM-5.3-Flash-shaped model: hyper-connections (mHC) + MLA + MoE,
 * used to verify the layer-major batched prefill against the token-serial
 * reference. */
cwick_status cwick_setup_synthetic_mhc(const char *root, cwick_error *err);

/* ------------------------------------------------------------------ */
/* OpenAI-compatible server                                            */
/* ------------------------------------------------------------------ */

/* Incremental SSE split state machine (server.c). Exposed so the test suite can
 * exercise the exact reasoning/content/tool-call boundary handling, including
 * markers split across tokens, without needing a live model. */
typedef enum {
    CWICK_SEG_REASONING = 0,   /* before </think>  -> reasoning_content */
    CWICK_SEG_CONTENT   = 1    /* before <tool_call> -> content / text  */
} cwick_segment_kind;

typedef struct {
    bool after_think;       /* past </think> (true from the start for completions) */
    bool tool_mode;         /* a <tool_call> began: the rest is not content */
    bool content_started;   /* a content byte has been emitted (leading \n strip) */
    size_t sent;            /* bytes of the answer already emitted */
    /* Which markers delimit the reasoning block on this checkpoint. NULL means
     * `<think>` / `</think>` (GLM and DeepSeek); MiniMax-M3.1 writes
     * `<mm:think>` / `</mm:think>`, and its tokenizer carries `<think>` as an
     * unrelated control token, so the pair cannot be inferred from the text. */
    const char *open_mk;
    const char *close_mk;
} cwick_stream_state;

/* Classify the next emit-ready slice of an accumulated answer. `st` carries the
 * phase across calls. Returns true and fills (*kind, *off, *len) when a slice is
 * ready, false when nothing more can be classified yet (or ever, once a
 * <tool_call> has begun). `final` releases the held-back tail. */
bool cwick_stream_segment(const char *raw, size_t raw_len, bool final,
                         cwick_stream_state *st, cwick_segment_kind *kind,
                         size_t *off, size_t *len);

/* Decode phase of a generated token, used by the per-step progress lines and
 * the access-log split. THINKING/CONTENT/TOOLS. */
typedef enum {
    CWICK_PHASE_THINKING = 0,
    CWICK_PHASE_CONTENT  = 1,
    CWICK_PHASE_TOOLS    = 2
} cwick_phase;

/* Count a finished generation's tokens by decode phase using the very state
 * machine above, so buffered requests log the same split as streaming ones:
 * before " train of thought close tag " (chat only) is thinking, from there
 * to the first " train of tool call start " is content, and from that marker
 * on everything is tool calls. A token counts towards the phase it landed in
 * when fully decoded (marker-straddling tokens go to the phase they end in).
 * out[0]=thinking, out[1]=content, out[2]=tools; they always sum to n. */
/* Same, but with the checkpoint's own reasoning markers (NULL = `<think>`).
 * MiniMax-M3.1 writes `<mm:think>`, so the replay needs to be told. */
void cwick_phase_account_mk(const cwick_tokenizer *tok, const uint32_t *tokens,
                            size_t n, bool is_chat, size_t out[3],
                            const char *open_mk, const char *close_mk,
                            bool closed_at_start);

void cwick_phase_account(const cwick_tokenizer *tok, const uint32_t *tokens,
                        size_t n, bool is_chat, size_t out[3]);

/* Thinking intensity ("思考强度"). Five client-facing levels — 低 / 中 / 高 /
 * 超高 / 极致 — plus off. The checkpoint's chat template was trained with
 * three Reasoning Effort headers (low / high / max; the C renderer passes the
 * string through verbatim), so the levels collapse onto those buckets:
 *   低 low/minimal      -> Low   (the lightest budget the model knows)
 *   中 medium           -> Low   (no middle bucket exists; stays light)
 *   高 high             -> High
 *   超高 ultra          -> Max
 *   极致 extreme/max    -> Max   (also the default when the client says nothing)
 *   off/none/disabled   -> Low   (thinking kept, at the minimum bucket)
 * English and Chinese spellings are accepted, case-insensitively; anything
 * else is refused (fail-closed, like every other unsupported field).
 * On success fills *level (static Chinese name, for logs) and *effort (the
 * header value to render) and returns true; NULL in/out are tolerated. */
bool cwick_thinking_effort_resolve(const char *raw, const char **level,
                                  const char **effort);

/* ---- thinking budget -------------------------------------------------
 * The checkpoint decides when its thought is finished; a client that asked
 * for a level meant a ceiling. Measured on agent traffic, one request spent
 * all 8096 generated tokens inside the think block and delivered no content
 * at all — a `finish=length` with `content=0` the client can only answer
 * with a full, equally expensive retry. So the engine enforces the ceiling
 * itself, by writing the close tag on the model's behalf: into the output
 * (the wire sees a closed block and everything after it is content) and into
 * the context (the model, seeing its own delimiter, starts the answer
 * instead of continuing the thought).
 *
 * The policy is a pure function so it can be tested without a model:
 *   `gen`        tokens generated in this response so far
 *   `budget`     ceiling in tokens; 0 = close before the first token
 *   `think_open` a think block is open (seeded by the template or written by
 *                the model)
 *   `closed`     the model already wrote the close tag itself
 * A negative budget disables everything (see cwick_engine_set_think_budget). */
typedef enum {
    CWICK_THINK_NONE = 0,       /* nothing to force */
    CWICK_THINK_CLOSE = 1,      /* emit the close tag */
    CWICK_THINK_OPEN_CLOSE = 2  /* never opened: emit <think></think> */
} cwick_think_action;

cwick_think_action cwick_think_plan(int64_t gen, int64_t budget,
                                   bool think_open, bool closed);

/* Minimum answer left behind the ceiling. A cap that lands on the length limit
 * leaves nothing to say, which is the failure the cap exists to prevent, so a
 * ceiling at or above max_new is clamped to `max_new - CWICK_THINK_RESERVE`
 * (or half of a smaller budget). */
#define CWICK_THINK_RESERVE 512

/* Arm the per-response ceiling. `budget` < 0 disables it (the default).
 * `open_id` / `close_id` are the tokenizer's ids for the two markers; pass
 * -1 for either when the tokenizer does not give each one as a single token,
 * which disables the forced closure rather than guessing a byte split.
 * `open_at_start` says the rendered prompt already ends with the open tag
 * (as GLM's template does) as opposed to the model writing it itself. */
/* Tell the engine the prompt already carries the block's close marker, so a
 * budget of 0 must not write a second empty block. Only the MiniMax template
 * closes the block before the first token; the other two open it. */
void cwick_engine_set_think_closed_at_start(cwick_engine *eng, bool closed);

/* How many tokens may actually be generated from `pos` with a `prompt_len`
 * prompt. `max_new` is an upper bound the client asks for ("at most N more"),
 * not a demand: on success it may be reduced to what the window still holds,
 * and only a prompt that does not fit at all is an error. Shared by the
 * DeepSeek and MiniMax loops so all three backends answer the same request the
 * same way -- they used to reject anything whose max_tokens overran the
 * window, while the GLM path clamped, which turned a 20-token answer into a
 * 400 whenever the client sent a generous max_tokens. */
cwick_status cwick_gen_window(int64_t pos, int64_t prompt_len,
                              int64_t *max_new, int64_t capacity,
                              cwick_error *err);

void cwick_engine_set_think_budget(cwick_engine *eng, int64_t budget,
                                  int64_t open_id, int64_t close_id,
                                  bool open_at_start);

typedef struct cwick_server_options {
    const char *host;      /* default 127.0.0.1 */
    int port;
    int64_t max_tokens;    /* per-response ceiling */
    int64_t max_request_bytes;
    bool verbose;
    bool trace;            /* print prompts, outputs and tool calls to stdout */
    /* Serve the built-in chat UI at `/` (and its assets under /web/). The
     * page is compiled into the binary and calls this same server over
     * /v1/chat/completions, so `cwick serve --web` is the whole setup. */
    bool web;
    /* Serve POST /v1/systemone: typed answers (choice / score / noul) read out
     * of one prefill over the resident model, for callers that need a decision
     * rather than a conversation. No second model, no second process. */
    bool jev;
    /* Single-user login password. NULL/empty disables auth; an empty string
     * passed deliberately (--lan) means "generate a random password". The
     * server hashes it with PBKDF2 and never writes it to disk. */
    const char *password;
    /* API keys for non-browser clients. `api_key` is a *name* for a key whose
     * secret cwick generates and prints once at startup; `api_key_secret` is a
     * secret the operator chose (CWICK_API_KEY), adopted as-is. Either one
     * turns single-user login on, because a key that is not checked is just a
     * string in a config file. Only hashes are ever stored. */
    const char *api_key;
    const char *api_key_secret;
    /* Read end of a self-pipe that main writes to from its SIGINT/SIGTERM
     * handler. `-1` when not supplied: the server makes its own. */
    int wake_fd;
    /* Prefix cache: snapshots the state of a token prefix to `dir` so a
     * prompt that repeats (every turn of an agent session resends the same
     * long system prompt and tool list) is prefilled once. NULL disables it;
     * `prefix_cache_gib` bounds the directory (0 = default 8 GiB). */
    const char *prefix_cache_dir;
    int64_t prefix_cache_gib;
    /* Extra local models to advertise and to allow switching to. Each entry
     * is a directory that is itself a checkpoint root, or one whose immediate
     * children are; discovered entries join the running model in GET
     * /v1/models and can be made resident with POST /v1/models/switch. NULL
     * when unset. The array is copied by reference — keep it alive for the
     * lifetime of cwick_server_run(). */
    const char *models_dirs[8];
    int models_dirs_n;
    /* Out-parameter that answers "which engine is live now?". On return this
     * holds the engine the caller should free: normally the one passed in,
     * but after a model switch it is the replacement (the original was freed
     * as part of the swap, so freeing it here would be a double free). NULL is
     * accepted and means "no switching is possible". */
    cwick_engine **live_engine;
    /* sqlite file for the built-in apps' own data (game history, move logs,
     * small preferences). NULL/empty = $CWICK_STORE, else ~/.cwick/apps.db.
     * NULL *and* CWICK_STORE unset still gives a store: an app that cannot
     * remember anything is not what the app market is for. Pass a path under
     * /dev/null only if you really mean "throw everything away". */
    const char *store_path;
} cwick_server_options;

cwick_status cwick_server_run(cwick_engine *eng, const cwick_server_options *opt,
                            volatile int *stop, cwick_error *err);

#ifdef __cplusplus
}
#endif

#endif /* CWICK_H */
