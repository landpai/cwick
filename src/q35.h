/*
 * q35.h — the Qwen3.5 hybrid-attention checkpoint family.
 *
 * A fourth forward pass beside the GLM, DeepSeek and MiniMax ones, sharing
 * only the process. What makes this checkpoint different is not its size
 * (~18 GiB of BF16, the smallest of the four) but its layer mix: 24 of the 32
 * layers have no softmax attention at all. They carry a Gated DeltaNet, a
 * linear-attention recurrence that keeps a fixed-size state instead of a KV
 * cache, so their per-token cost does not grow with context. The remaining 8
 * layers are ordinary GQA with two twists -- a per-head RMSNorm on Q and K,
 * and an output gate that doubles the width of `q_proj`.
 *
 * Geometry, read from the shard headers rather than assumed
 * (config.json alone does not say how wide `q_proj` is):
 *
 *   dim 4096, 32 layers, vocab 248320, head_dim 256 (explicit, not
 *   hidden/heads), 16 query heads / 4 KV heads, partial rope on the first
 *   64 of each head's 256 dims, theta 1e7.
 *
 *   full attention (8 layers)
 *     q_proj  [8192, 4096]   two halves: 16*256 query, 16*256 output gate
 *     k_proj  [1024, 4096]   v_proj the same
 *     q_norm / k_norm [256]  RMSNorm per head, applied before the rotation
 *     o_proj  [4096, 4096]
 *
 *   linear attention (24 layers), Gated DeltaNet
 *     in_proj_qkv [8192, 4096]   q(2048) | k(2048) | v(4096)
 *     in_proj_z   [4096, 4096]   the gate the output norm multiplies by
 *     in_proj_a   [32, 4096]     decay input, one per value head
 *     in_proj_b   [32, 4096]     write strength, one per value head
 *     conv1d      [8192, 1, 4]   depthwise causal, SiLU after it
 *     A_log, dt_bias [32]
 *     norm        [128]          RMSNormGated over one value head
 *     out_proj    [4096, 4096]
 *
 * Every layer has the same shell: input_layernorm -> mixer -> +residual ->
 * post_attention_layernorm -> SwiGLU -> +residual.
 *
 * The recurrence, per value head h (key head h/2, see Q35_LIN_KV_GROUP):
 *
 *   g    = -exp(A_log) * softplus(a + dt_bias)      the log decay
 *   beta = sigmoid(b)                               the write strength
 *   S    = exp(g) * S + k (x) (beta * (v - S^T k))   the delta rule
 *   o    = S^T q
 *
 * q and k are L2-normalised per head before the update, and S is
 * [lin_k_dim, lin_v_dim] = [128, 128] of f32 per value head. That is 2 MiB per
 * layer per sequence (50 MiB for all 24), against a KV cache that would grow
 * without bound -- which is the whole reason the architecture exists.
 *
 * Implemented: the MTP draft head (`mtp.*`, loaded from a separate
 * `model-mtp.safetensors`), which feeds the speculative decoder. Not
 * implemented: the vision tower (27 layers under `model.visual.*`), which is
 * unreachable from a text prompt because images would need `image_token_id`
 * positions to expand into patches.
 */
#ifndef CWICK_Q35_H
#define CWICK_Q35_H

#include <stdint.h>
#include <stddef.h>

#include "cwick.h"

#ifdef __cplusplus
extern "C" {
#endif

#define Q35_MAX_LAYERS 64

/* Positions per prefill block. Prefill is layer-major over a block: every
 * projection of a layer runs once for the whole block, and only the delta-rule
 * recurrence -- which is sequential by definition -- walks positions one at a
 * time. Activations for a 512-position block are 512*8192*4 = 16 MiB at the
 * widest point, so the block size is chosen for GEMM efficiency, not memory. */
#define Q35_CHUNK 512

/* Queries per attention GEMM pair. The score matrix of a tile is
 * attn_q x keys floats, so this trades call count against scratch: at 128
 * queries and a 128K context the buffer is 64 MiB, and cblas still gets an
 * M that is a multiple of its 8-row micro-kernel. */
#define Q35_ATTN_Q 128

/* Row slices in the decode output pass. Each slice accumulates its own copy of
 * the 16 output heads and they are summed once at the end, which is what lets a
 * work item read whole cache rows instead of one kv head's quarter of them. */
#define Q35_DEC_SLICES 16

/* Value heads per key head. 32 value heads read from 16 key heads, and the
 * checkpoint stores them grouped (value heads 0..1 read key head 0, 2..3 key
 * head 1, ...), which is what HF's qwen35 layout does; an interleaved store
 * would give every head the wrong key. */
#define Q35_LIN_KV_GROUP 2

/* ------------------------------------------------------------------ */
/* config                                                              */
/* ------------------------------------------------------------------ */

typedef struct {
    int dim;                 /* 4096 */
    int n_layers;            /* 32 */
    int n_heads;             /* 16 query heads, full attention */
    int n_kv_heads;          /* 4 */
    int head_dim;            /* 256 */
    int rotary_dim;          /* 64: head_dim * partial_rotary_factor 0.25 */
    int inter_dim;           /* 12288 */
    int vocab_size;          /* 248320 */

    int lin_k_heads;         /* 16 */
    int lin_v_heads;         /* 32 */
    int lin_k_dim;           /* 128 */
    int lin_v_dim;           /* 128 */
    int conv_kernel;         /* 4 */
    int n_lin_layers;        /* 24 */
    int n_full_layers;       /* 8 */

    int attn_gate;           /* 1: q_proj carries a second half for the gate */
    float rms_eps;           /* 1e-6 */
    float l2_eps;            /* 1e-6, the Q/K normalisation */
    double rope_theta;       /* 1e7 */
    int max_seq_len;

    uint8_t *is_linear;      /* [n_layers], 1 = Gated DeltaNet */
} q35_config;

/* ------------------------------------------------------------------ */
/* weights                                                             */
/* ------------------------------------------------------------------ */

/* The 1-D parameter tensors are widened to f32 once at load time: they are
 * read on every position of every token, and a bf16 -> f32 conversion inside
 * the hot loop would buy nothing for ~1 MiB of memory. Matrices stay bf16 and
 * are consumed by the GEMM/GEMV kernels directly. */
typedef struct {
    const uint16_t *w_qkv;   /* [qkv_dim, dim]  q(2*lin_k_heads*lin_k_dim) | */
    const uint16_t *w_z;     /* [lin_v_heads*lin_v_dim, dim] */
    const uint16_t *w_a;     /* [lin_v_heads, dim] */
    const uint16_t *w_b;     /* [lin_v_heads, dim] */
    const uint16_t *w_o;     /* [dim, lin_v_heads*lin_v_dim] */
    float *conv;             /* [conv_kernel][qkv_dim], transposed + widened */
    float *a_log;            /* [lin_v_heads] */
    float *dt_bias;          /* [lin_v_heads] */
    float *norm;             /* [lin_v_dim] */
} q35_lin_w;

typedef struct {
    const uint16_t *wq;      /* [(1+gate)*n_heads*head_dim, dim] */
    const uint16_t *wk;      /* [n_kv_heads*head_dim, dim] */
    const uint16_t *wv;      /* [n_kv_heads*head_dim, dim] */
    const uint16_t *wo;      /* [dim, n_heads*head_dim] */
    float *q_norm;           /* [head_dim] */
    float *k_norm;           /* [head_dim] */
} q35_attn_w;

typedef struct {
    int is_linear;
    float *in_norm;             /* [dim] */
    float *post_norm;           /* [dim] */
    const uint16_t *w_gate;     /* [inter_dim, dim] */
    const uint16_t *w_up;       /* [inter_dim, dim] */
    const uint16_t *w_down;     /* [dim, inter_dim] */
    q35_lin_w lin;
    q35_attn_w attn;
} q35_layer;

/* ---- MTP draft head ------------------------------------------------- *
 *
 * One block, trained to predict the token *after* the next one given the
 * hidden state of the position before it:
 *
 *   h'    = fc( [ pre_norm_emb(emb(x_{t+1})) , pre_norm_hid(h_t) ] )
 *   res   = h' + attn(h') + mlp(h')          one ordinary GQA block
 *   draft = lm_head( out_norm(res) )         the shared 2.03 GiB head
 *
 * It is shaped exactly like a full-attention layer of the text model --
 * 16 query heads, 4 KV heads, head_dim 256, the same interleaved output gate
 * and the same partial rope -- so the decode kernels already implement it and
 * only the schedule is new.
 *
 * `mtp_use_dedicated_embeddings` is false in this checkpoint, so the embedding
 * table and the output projection are the text model's. That single fact is
 * what makes the head expensive to use: a draft token is 486.6 MiB of block
 * weights *plus* a pass over 2.03 GiB of vocabulary rows. See q35_mtp_draft. */
typedef struct {
    const uint16_t *fc;      /* [dim, 2*dim]  rows of the concat, emb first */
    const uint16_t *wq;      /* [(1+gate)*n_heads*head_dim, dim] */
    const uint16_t *wk;      /* [n_kv_heads*head_dim, dim] */
    const uint16_t *wv;      /* [n_kv_heads*head_dim, dim] */
    const uint16_t *wo;      /* [dim, n_heads*head_dim] */
    const uint16_t *gate;    /* [inter_dim, dim] */
    const uint16_t *up;      /* [inter_dim, dim] */
    const uint16_t *down;    /* [dim, inter_dim] */
    float *pre_norm_emb;     /* [dim] */
    float *pre_norm_hid;     /* [dim] */
    float *in_norm;          /* [dim] */
    float *post_norm;        /* [dim] */
    float *out_norm;         /* [dim] */
    float *q_norm;           /* [head_dim] */
    float *k_norm;           /* [head_dim] */
} q35_mtp_w;

typedef struct q35_model {
    q35_config cfg;
    cwick_safetensors saf;
    int loaded;

    /* Full residency (see q35_model_open). `resident` says the text weights
     * were faulted in at load time; `resident_bytes` is how much of the
     * checkpoint that covered, and `state_locked_bytes` is the KV/DeltaNet
     * state that was wired along with it. */
    int resident;
    uint64_t resident_bytes;
    uint64_t state_locked_bytes;

    const uint16_t *embed;      /* [vocab, dim] */
    const uint16_t *head;       /* [vocab, dim] */
    float *final_norm;          /* [dim] */

    /* ---- MTP draft head, optional ---- *
     * `mtp_ok` is 0 when the checkpoint has no `model-mtp.safetensors`, in
     * which case every mtp_* field below is unused and the decoder drafts with
     * prompt lookup instead. The tensors are mapped from their own file (the
     * text shards' index does not name them) and are not part of `saf`. */
    int mtp_ok;
    void *mtp_map;
    size_t mtp_map_len;
    q35_mtp_w mtp;
    /* The block's own KV cache, a sliding window of the most recent
     * positions. It is *not* back-filled from the prompt: those rows would
     * have to be computed from the text model's hidden states, which prefill
     * discards. What it is filled with is every position the decoder has
     * finished -- q35_mtp_sync copies the verify pass's hidden states in
     * before each draft, so from the second step on the window holds real
     * context rather than whatever the previous chain guessed.
     *
     * Rows are indexed by absolute position minus `mtp_lo`, so the score loop
     * reads rows 0..keys-1 in order and never wraps; when the window fills,
     * mtp_compact shifts the live rows down (their rotation was baked in when
     * they were written, so the shift is exact). */
    float *mtp_kcache;          /* [mtp_win][n_kv_heads*head_dim] */
    float *mtp_vcache;
    int mtp_win;                /* rows in each cache */
    int64_t mtp_n;              /* absolute position of the next row to write */
    int64_t mtp_lo;             /* absolute position of row 0 */
    int64_t mtp_base;           /* positions confirmed by the text model */

    q35_layer layers[Q35_MAX_LAYERS];

    /* cos/sin per (position, pair). rope covers rotary_dim/2 pairs and theta is
     * fixed, so one table serves all 8 attention layers. */
    float *rope_cos, *rope_sin;   /* [max_seq][rotary_dim/2] */

    /* ---- caches, one sequence ---- */
    /* KV cache exists only for full-attention layers; the linear ones keep
     * their recurrence state instead. Indexed by layer, so a NULL row means
     * "this layer is a DeltaNet and owns no KV". */
    /* f32, because both attention passes are GEMMs and cblas takes f32. The
     * prefix snapshot still travels as bf16 rows (half the bytes on disk and in
     * the cache file); pack converts, apply converts back. */
    float **kcache;      /* [n_layers][max_seq][n_kv_heads*head_dim] f32 */
    float **vcache;
    /* DeltaNet state. conv_state holds the last (conv_kernel-1) *post*-SiLU
     * inputs of the depthwise convolution, per channel; ssm_state is the
     * [k_dim, v_dim] matrix per value head. */
    float **conv_state;  /* [n_layers][(conv_kernel-1) * qkv_dim] */
    float **ssm_state;   /* [n_layers][lin_v_heads * lin_k_dim * lin_v_dim] */

    /* ---- scratch, sized by Q35_CHUNK ---- */
    float *x;        /* [chunk][dim] residual stream */
    float *xn;       /* [chunk][dim] normalised copy */
    float *mix;      /* [chunk][max(qkv_dim, (1+gate)*heads*hd)] mixer output */
    float *gate;     /* [chunk][max(lin_v_dim*lin_v_heads, inter_dim)] */
    float *up;       /* [chunk][inter_dim] */
    float *down;     /* [chunk][dim] */
    float *att;      /* [chunk][n_heads*head_dim] attention output */
    float *qb;       /* [chunk][n_heads*head_dim] q after norm/rope */
    float *kb;       /* [chunk][n_kv_heads*head_dim] */
    float *vb;       /* [chunk][n_kv_heads*head_dim] */
    float *lin_o;    /* [chunk][lin_v_heads*lin_v_dim] recurrence output */
    float *gdec;     /* [lin_v_heads] exp(g) per position */
    float *beta;     /* [lin_v_heads] sigmoid(b) */
    /* Scores of one (head, query tile): [Q35_ATTN_Q][max_seq]. Sized by the
     * context, because the tile's score row spans every key it can see. Not
     * wired -- it is scratch, and pinning it would take memory from the cache. */
    float *attn_s;
    float *dec_out;  /* [Q35_DEC_SLICES][n_heads][head_dim] decode partials */
    size_t attn_s_cap;
    /* Set by q35_model_forward when it runs a planning chunk: the last
     * position's logits are written here. */
    float *logits;
} q35_model;

/* Fill a cwick_model_spec from config.json (called by model.c so the model
 * catalog and the switch dry-run can tell a loadable checkpoint from a
 * downloaded one). */
cwick_status q35_spec_fill(void *root, void *cfg, const char *dir,
                           cwick_model_spec *spec, cwick_error *err);

int q35_model_open(q35_model *m, const char *root, int max_seq,
                   cwick_error *err);
void q35_model_close(q35_model *m);

/* Run `n` positions starting at absolute `pos`. Writes the logits of the LAST
 * position into `logits` (size vocab). `n` may exceed Q35_CHUNK; the caller
 * normally splits, but this handles it. */
int q35_model_forward(q35_model *m, const int32_t *ids, int n, int64_t pos,
                      float *logits, cwick_error *err);

/* Run `n` consecutive positions and write ALL of their logits, row-major:
 * all[i * vocab .. (i+1) * vocab) is position pos + i. This is the verify
 * entry point -- a speculative step needs k distributions from one weight
 * pass -- and it is the only caller shape where the GPU forward works on more
 * than one row. */
int q35_model_forward_all(q35_model *m, const int32_t *ids, int n, int64_t pos,
                          float *all, cwick_error *err);

/* The same n-row pass, but leaving every row's state in its own slot so the
 * caller can choose afterwards: the verify only knows which drafts survived
 * once the logits are back. Pair it with q35_model_keep_rows. */
int q35_model_verify_rows(q35_model *m, const int32_t *ids, int n, int64_t pos,
                          float *all, cwick_error *err);

/* After q35_model_verify_rows, keep only the state as of row `j`: a verify
 * that accepted the first j drafts of n runs the recurrence over all n of them,
 * and a DeltaNet has no inverse to undo the rest. The k-row step writes each
 * row into its own state slot so this is one copy, not another pass. */
int q35_model_keep_rows(q35_model *m, int j);

/* ---- MTP drafter ------------------------------------------------------ *
 * 1 when the checkpoint ships a usable draft head and the GPU path is up
 * (the draft runs as one command buffer of existing kernels, so there is no
 * CPU implementation -- without the GPU the decoder falls back to prompt
 * lookup). */
int q35_mtp_available(const q35_model *m);

/* Write the block's cache rows for the positions that already exist, using the
 * hidden states the last forward pass left in `g_s.x` -- rows 0..hrow hold the
 * positions `pos-hrow-1 .. pos-1`, and `nxt[r]` is the token *after* row r's
 * position, i.e. the token at `pos-hrow+r`, so `nxt[hrow]` is the token at
 * `pos` itself.
 *
 * This is what turns the drafter from context-free into context-bearing: a
 * draft attends over keys, and before this the only keys in the cache were the
 * ones earlier drafts had guessed. Costs one block run per position that was
 * accepted last step (1.6 ms each, 464 MiB of weights apiece), which is why it
 * is called once per step with the rows the verify already produced rather
 * than over the prompt. */
int q35_mtp_sync(q35_model *m, int hrow, int hpost, int64_t pos,
                 const int32_t *nxt);

/* Propose up to `k` tokens (k <= Q35_GK-1) that could follow position
 * `pos - 1`, given `first` (the token already accepted for position `pos`)
 * and the hidden state that row `hrow` of the last pass left on the device
 * (`hpost` picks the post-final-norm copy rather than the residual stream).
 *
 * Each proposal runs the whole block and then projects it through the shared
 * vocabulary head restricted to `cand[0..ncand)` -- the caller supplies that
 * set, because a draft is allowed to be wrong and the verify decides, so the
 * only thing a narrow candidate set costs is acceptance rate, not accuracy.
 * Writes the proposals to `out` and returns how many it produced.
 *
 * The chain is one command buffer: proposal i's token is read by proposal
 * i+1's embedding lookup on the device, so k proposals cost one round trip
 * rather than k. */
int q35_mtp_draft(q35_model *m, int hrow, int hpost, int32_t first, int64_t pos,
                  int k, const uint32_t *cand, int ncand, int32_t *out);

/* Number of layers whose mixer is a Gated DeltaNet. */
int q35_linear_layers(const q35_config *c);

/* Bytes of cache the model needs for `max_seq` positions, for the budget
 * lines in main.c. */
uint64_t q35_cache_bytes(const q35_model *m);

/* ---- prefix cache ---------------------------------------------------- *
 * What has to travel is only what a token boundary preserves: the KV rows of
 * the 8 full-attention layers for `pos` positions, and the 24 DeltaNet
 * layers' conv/ssm state, which is a fixed size because a recurrence has no
 * history. Layout: the fixed part first (layer-major, conv then ssm), then
 * the KV rows (layer-major, keys then values). Putting the fixed part first
 * makes `pos` recoverable from the blob length alone, which is what lets a
 * truncated snapshot still apply.
 *
 * `q35_prefix_bytes(m, pos)` is exactly what `q35_prefix_pack` writes.
 * `q35_prefix_apply` restores at most `want` positions (0 = as many as the
 * blob holds) and reports how many it actually restored -- the caller always
 * re-runs the last position, because a snapshot carries no hidden state to
 * put through lm_head. */
size_t q35_prefix_bytes(const q35_model *m, int64_t pos);
int q35_prefix_pack(const q35_model *m, int64_t pos, void *buf, size_t cap);
int q35_prefix_apply(q35_model *m, const void *buf, size_t len, int64_t want,
                     int64_t *pos_out);

/* Decide whether a checkpoint should be held fully in core.
 *
 * This backend has no routed experts, so every byte of the checkpoint is read
 * by every token and there is nothing to stream -- which makes residency a
 * question of size, not of policy. The answer is yes while the weights plus
 * the caches fit in half of RAM and under the configured ceiling; above that
 * the page cache is a better citizen than wiring, because the machine still
 * has to hold the KV state of whatever else is running.
 *
 * `host` or `ceiling` of 0 means "unknown", and unknown is a no: the whole
 * point of the check is to avoid a decision made without the measurement. */
int q35_resident_wanted(uint64_t weights, uint64_t state, uint64_t host,
                        uint64_t ceiling);

#ifdef __cplusplus
}
#endif

#endif /* CWICK_Q35_H */
