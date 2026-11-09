/*
 * mm.h — MiniMax-M3.1-Flash.
 *
 * A third forward pass beside the GLM and DeepSeek ones, sharing only the
 * process with them. The checkpoint is ~233 GiB: 60 layers, 128 routed
 * experts per MoE layer, and every weight quantized -- the routed experts to
 * packed FP4, everything else to FP8.
 *
 * Storage, measured from the shard headers rather than assumed
 * (tools/mm_probe prints it for any layer):
 *
 *   routed experts  [N,K/2] U8  two e2m1 values per byte along K
 *                   [N,K/16] F8_E4M3  one scale per 16 K elements, per row
 *                   two F32 scalars, `weight_global_scale` and
 *                   `input_global_scale`. The first is a DIVISOR: a decoded
 *                   weight is fp4 * e4m3_block / 8192. The probe settles it --
 *                   read as a multiplier the same bytes give weights around
 *                   1e6 instead of around 0.03.
 *
 *   dense weights   [N,K] F8_E4M3 payload + [N,K/32] U8 (e8m0, power of two)
 *                   one scale per 32 K elements, per row.
 *
 *   router          [n_experts, dim] F32, plus a F32 [n_experts] bias. The
 *                   only unquantized matrix in the model.
 *
 *   norms           BF16, and `use_gemma_norm` makes the multiplier (1 + w).
 *
 * Geometry, for the numbers this file is parameterised on:
 *   dim 6144, 60 layers, 64 query heads / 4 key-value heads of 128, rope on
 *   the first 64 dims of each head, MoE from layer 3 on with 128 routed (top
 *   4) plus one shared expert, a per-block indexer picking 16 blocks of 128.
 */
#ifndef CWICK_MM_H
#define CWICK_MM_H

#include <stdint.h>
#include <stddef.h>

#include "cwick.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MM_MAX_LAYERS 64
/* Positions per prefill chunk. Prefill is layer-major over a chunk, so this
 * bounds both the activations and the number of distinct experts a chunk can
 * reach -- see mm_moe_chunk. */
/* Prefill block size. 1024, not 512: the expert read is per-block, so the
 * block is what amortises it. Measured on MiniMax at 4096 context, a 512-token
 * block costs 173 ms/token while a 38-token tail costs ~545 ms/token; a prompt
 * of 550 tokens split 512+38 took 109.3 s and as one 550-token block takes
 * ~95 s. Blocks of 512-1024 are the common /v1/systemone prompt size, so the
 * ceiling sits at 1024. Buffers below are sized by this number, which costs
 * ~1 GiB of activations -- the machine has 128. */
#define MM_CHUNK 1024

/* Largest speculative block, in positions per forward. 8 is the number the
 * checkpoint's own block drafter works in (7 candidates + the sampled token),
 * so this is a ceiling on what a drafter could ask for rather than a tuning
 * knob: every extra position costs a head GEMV and widens the expert union. */
#define MM_SPEC_MAX 16

/* ------------------------------------------------------------------ */
/* config                                                               */
/* ------------------------------------------------------------------ */

typedef struct {
    int dim;                 /* 6144 */
    int n_layers;            /* 60 */
    int n_heads;             /* 64 query heads */
    int n_kv_heads;          /* 4 */
    int head_dim;            /* 128 */
    int rotary_dim;          /* 64: rope covers the first half of a head */
    int inter_dim;           /* 3072: routed / shared expert width */
    int dense_inter_dim;     /* 12288: the three dense MLP layers */
    int first_k_dense;       /* 3: layers 0..2 use a dense MLP */
    int vocab_size;          /* 200064 */
    int n_routed;            /* 128 */
    int n_activated;         /* 4 */
    int max_seq_len;

    /* indexer (the sparse-attention block picker) */
    int index_dim;           /* 128 */
    int index_n_heads;       /* 4 */
    int topk_blocks;         /* 16 */
    int block_size;          /* 128 */
    int local_blocks;        /* 1: the most recent block is always kept */

    float rms_norm_eps;      /* 1e-6 */
    double rope_theta;       /* 1e7 */
    float swiglu_alpha;      /* 1.702 */
    float swiglu_limit;      /* 7.0 */
    float route_scale;       /* 2.0 */
    int norm_topk_prob;      /* 1: renormalise the top-k weights */
} mm_config;

/* ------------------------------------------------------------------ */
/* weights                                                             */
/* ------------------------------------------------------------------ */

/* One FP8 matrix. `w16` is the same matrix widened to BF16 at load time; it
 * is what the hot path reads. The widening is exact -- e4m3 carries three
 * mantissa bits, bf16 carries eight, and the e8m0 scale is a power of two --
 * and it is worth the memory because decoding bf16 is a shift where decoding
 * e4m3 is a table lookup. */
typedef struct {
    const uint8_t *w;        /* F8_E4M3 [rows, cols] */
    const uint8_t *s;        /* U8 e8m0 [rows, cols/32] */
    const uint16_t *w16;     /* BF16 [rows, cols], scaled in; NULL if not kept */
    int64_t rows, cols;
    const float *gscale;     /* F32 scalar divisor, NVFP4 only */
} mm_qw;

/* One routed expert: three matrices (gate, down, up) plus the descriptors the
 * pin cache needs to read them as one blob each. */
typedef struct {
    cwick_st_tensor tw[3];   /* packed bytes, for cwick_stream_locate */
    cwick_st_tensor ts[3];   /* the paired e4m3 scale tensor */
    const uint8_t *p[3];     /* mapped packed bytes (cold path) */
    const uint8_t *sc[3];    /* mapped e4m3 scales */
    float inv_g[3];          /* 1 / weight_global_scale, per matrix */
} mm_expert;

typedef struct {
    int is_dense;            /* layers 0..2: a plain MLP instead of MoE */

    mm_qw wq, wk, wv, wo;              /* [n_heads*head_dim|n_kv*head_dim, dim] */
    const uint16_t *q_norm, *k_norm;   /* BF16 [head_dim] */
    mm_qw idx_q, idx_k;                /* indexer projections */
    const uint16_t *idx_q_norm, *idx_k_norm;
    const uint16_t *in_norm, *post_norm;

    const float *gate_w;               /* F32 [n_routed, dim] */
    const float *gate_bias;            /* F32 [n_routed] */
    mm_qw sw1, sw2, sw3;               /* shared expert: gate, down, up */
    mm_qw dg, du, dd;                  /* dense MLP: gate, up, down */
    mm_expert *experts;                /* [n_routed] */
} mm_layer;

typedef struct mm_model {
    mm_config cfg;
    cwick_safetensors saf;
    cwick_stream *stream;
    int loaded;
    uint64_t rss_target;

    /* Resident copy of what every token reads: the dense weights, the router,
     * the norms. The shards they came from are file-backed, so any page the
     * kernel reclaims costs an SSD read on the next token; anonymous memory
     * holding the only copy cannot be dropped that way. */
    uint8_t *pin_base;
    size_t pin_cap, pin_used;
    /* The FP8 half of the above, widened to BF16 in place of the original. */
    uint8_t *dense16;
    size_t dense16_cap, dense16_used;

    const uint16_t *embed;             /* BF16 [vocab, dim] */
    const uint16_t *head;              /* BF16 [vocab, dim] */
    const uint16_t *final_norm;        /* BF16 [dim] */

    mm_layer layers[MM_MAX_LAYERS];
    /* cos/sin per pair, per position: rope_theta is fixed and there is a
     * single rope configuration, so one table serves every layer. */
    float *rope_cos, *rope_sin;        /* [max_seq_len, rotary_dim/2] each */

    /* ---- caches (batch 1) ---- */
    uint16_t *kcache;   /* [n_layers][max_seq][n_kv_heads*head_dim] bf16 */
    uint16_t *vcache;   /* the same for V */
    uint16_t *idxcache; /* [n_layers][max_seq][index_dim] bf16 indexer keys */

    /* ---- scratch ---- */
    float *h;            /* [dim] residual stream */
    float *h2;           /* [dim] */
    /* Set by mm_model_forward_multi only: the final norm and the head then run
     * for every position of the chunk and write [n][vocab] floats here. The
     * decode loop leaves it NULL, so a normal step pays nothing for it. */
    float *spec_out;
    float *sub;          /* [dim] sublayer input */
    float *subl;         /* [dim] sublayer output */
    float *q;            /* [n_heads*head_dim] */
    float *k;            /* [n_kv_heads*head_dim] */
    float *v;            /* [n_kv_heads*head_dim] */
    float *att;          /* [n_heads] */
    float *o;            /* [n_heads*head_dim] */
    float *iq;           /* [index_n_heads*index_dim] */
    float *ik;           /* [index_dim] */
    float *bscore;       /* [max_seq/block_size + 2] block scores */
    float *heap;         /* [topk_blocks] */
    int *heap_idx;
    int *blk_out;        /* [topk_blocks] chosen block indices */
    uint8_t *selmask;    /* [max_seq/block_size + 2] */
    float *scores;       /* [max_seq] indexer per-position scores */
    float *gsm;          /* [inter_dim] gate */
    float *usm;          /* [inter_dim] up */
    float *router;       /* [n_routed] */
    int *sel;            /* [n_activated] */
    float *selw;         /* [n_activated] */
    uint16_t *act16;     /* [max(dim, inter_dim)] activation in bf16 */
    float *evbuf, *odbuf; /* [dim/2] the NVFP4 GEMV's de-interleaved x */
    float *moe_y;        /* [dim] */
    int64_t pos;         /* positions filled so far */
    int max_sel;         /* room in `att` for one head's selected positions */
    int cur_layer;       /* which layer the indexer scratch is working on */

    /* ---- chunked prefill ---- */
    float *ch_h;         /* [MM_CHUNK][dim] */
    float *ch_sub;       /* [MM_CHUNK][dim] */
    float *ch_subl;      /* [MM_CHUNK][dim] */
    float *ch_q;         /* [MM_CHUNK][n_heads*head_dim] */
    float *ch_k;         /* [MM_CHUNK][n_kv_heads*head_dim] */
    float *ch_v;         /* the same for V */
    float *ch_iq;        /* [MM_CHUNK][index_n_heads*index_dim] */
    float *ch_ik;        /* [MM_CHUNK][index_dim] */
    float *ch_o;         /* [MM_CHUNK][n_heads*head_dim] */
    float *ch_att;       /* [MM_CHUNK][n_heads][sel] */
    float *ch_contrib;   /* [MM_CHUNK][n_activated][dim] routed terms */
    int *ch_sel;         /* [MM_CHUNK][n_activated] */
    float *ch_selw;      /* [MM_CHUNK][n_activated] */
    int *ch_uni;         /* [n_routed] union of experts this chunk reaches */
    uint8_t *ch_mark;    /* [n_routed] */
    int *ch_rows;        /* [MM_CHUNK*n_activated] (position, slot) pairs */
    /* FP32 staging for one layer's dense matrix, so the batched prefill GEMM
     * has a plain sgemm to call. */
    float *wf32;
    size_t wf32_cap;

    /* Prefill MoE staging: the activations of every (position, slot) pair that
     * routed to one expert, and that expert's three results. Sized for the
     * worst case (every position picking the same expert). This is what lets
     * an expert's matrix be read once for a whole chunk instead of once per
     * position (see mm_fp4_gemm). */
    float *ex_x;    /* [MM_CHUNK*n_activated][dim] */
    float *ex_g;    /* [MM_CHUNK*n_activated][inter_dim] */
    float *ex_u;    /* the same, gate and up */
    float *ex_y;    /* [MM_CHUNK*n_activated][dim] */

    /* statistics */
    uint64_t n_expert_reads, n_expert_bytes;
} mm_model;

/* ------------------------------------------------------------------ */
/* api                                                                  */
/* ------------------------------------------------------------------ */

int mm_model_open(mm_model *m, const char *root, int max_seq_len,
                  cwick_error *err);
void mm_model_close(mm_model *m);
void mm_model_reset(mm_model *m);

/* Run positions [start, start+n). logits[vocab] receives the LAST position
 * only. n may be 1; the prefill path is chunked by the caller. */
int mm_model_forward(mm_model *m, const int32_t *ids, int n, int64_t start,
                     float *logits, cwick_error *err);

/* The same forward, but `logits_all` receives one distribution per position,
 * laid out [n][vocab]. This is what block verification needs: a candidate
 * token at position i is checked against the distribution the model produced
 * for position i-1, so anything less than every position cannot verify a
 * block. It costs one extra head GEMV per trailing position (the head is
 * vocab*dim bf16, i.e. read from the mapped checkpoint, not the SSD), which is
 * why nothing else in the engine uses it. */
int mm_model_forward_multi(mm_model *m, const int32_t *ids, int n, int64_t start,
                           float *logits_all, cwick_error *err);

/* Cross-token state for the prefix cache: the K/V/index caches of the first
 * `pos` positions. Same contract as the DeepSeek backend's pack/apply. */
/* n-gram draft for block verification: look for the most recent occurrence of
 * the history's tail and return up to `nmax` tokens that followed it. Exported
 * so the standalone driver can run the same verifier the server does. */
int mm_draft_lookup(const uint32_t *h, int64_t n, int nmax, uint32_t *out);

size_t mm_state_bytes(const mm_model *m, int64_t pos);
int mm_state_pack(const mm_model *m, int64_t pos, void *buf, size_t cap);
int mm_state_apply(mm_model *m, const void *buf, size_t len, int64_t *pos_out);

void mm_prefill_progress_begin(int64_t total_tokens, int layers);
void mm_prefill_progress_done(int64_t done_tokens);
/* Per-layer progress inside a prefill chunk: `start` is the chunk's absolute
 * first position, `n` its size, `layer` the 1-based layer just finished. */
void mm_prefill_progress_layer(int64_t start, int64_t n, int layer,
                               int nlayer);
void mm_prefill_progress_end(void);

/* True when root/config.json describes this architecture. */
int mm_model_probe(const char *root);

/* KV/index bytes for `max_seq` positions, and the bytes the dense weights
 * take. Both feed the expert-cache budget, which is the only line of the
 * memory plan that can be trimmed. */
uint64_t mm_kv_bytes(const mm_model *m, int max_seq);

/* Micro-benchmark for the NVFP4 GEMV, without a checkpoint: allocates an
 * [n,k] packed matrix, runs `iters` passes and returns ms/pass through
 * `ms_out`. The pointer arithmetic is the point -- the whole token rate is
 * this kernel's bandwidth, and measuring it through the model mixes in SSD
 * misses and cache state that change run to run. */
double mm_bench_fp4(int64_t n, int64_t k, int iters, double *gmacs_out);
/* The same shape as BF16 through cwick_gemv_bf16, for a reference ceiling. */
double mm_bench_bf16(int64_t n, int64_t k, int iters, double *gmacs_out);
/* Cross-check the grouped expert kernel against the single-row one; prints the
 * worst per-row difference. Same packing, so any difference is arithmetic. */
void mm_bench_gemm_check(int64_t N, int64_t K);
/* Time the FP8 dense GEMM on a real shape and report GMAC/s. */
void mm_bench_fp8(int64_t N, int64_t K, int64_t n, int iters);

#ifdef __cplusplus
}
#endif

#endif /* CWICK_MM_H */
