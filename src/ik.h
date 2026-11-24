/*
 * ik.h — the Inkling-2-Mini (inkling2_loop) backend.
 *
 * This checkpoint is not a decoder stack with a MoE bolted on: its residual
 * stream is a *bank* of per-block states that every layer reads through a
 * learned query (attention over depth), and the six layers in the middle run
 * three times, each pass receiving messages retrieved from the previous pass.
 * The geometry below is what the kernels in ik.c actually consume, so it is
 * read out of config.json rather than hard-coded.
 *
 * Two other things are unusual and both are load-bearing:
 *
 *   - there is no rotary embedding in the text model. Position enters only
 *     through (a) a hidden-state conditioned relative-distance bias and (b) a
 *     log-scaled multiplier on Q and on the relative states of the four
 *     full-attention layers. Get either wrong and the model still emits
 *     fluent-looking text at position 0 and nonsense later.
 *   - every attention layer has a per-head sink logit. It is not a mask: it
 *     absorbs probability mass, so the weights do not sum to one.
 */
#ifndef CWICK_IK_H
#define CWICK_IK_H

#include <stdbool.h>
#include <stdint.h>

#include "cwick.h"

#define IK_MAX_LAYERS 32
#define IK_MAX_BLOCKS 32
/* The loop layers run more than once and each pass needs its own keys, values
 * and convolution history: a pass that re-read the previous pass's cache would
 * attend to states from the wrong depth. Slots 0..n_layers-1 are the first
 * pass; the extra ones are appended pass by pass, exactly like the reference
 * appends deep copies of the loop layers' caches. */
#define IK_MAX_SLOTS 64
/* tokens per prefill slice: the scratch buffers are sized for this */
#define IK_CHUNK 64

typedef struct ik_cfg {
    int64_t hidden;
    int64_t n_layers;
    int64_t n_heads;
    int64_t n_kv;
    int64_t head_dim;
    int64_t vocab;
    int64_t unpadded_vocab;

    int64_t d_rel;        /* relative-bias profile width */
    int64_t rel_extent;   /* bias table width (== full-attention distances) */
    int64_t swa_window;   /* sliding window / sliding rel extent */
    int64_t sconv_k;      /* short-convolution taps */
    int64_t res_block;    /* layers per attention-residual block */

    int64_t n_experts;
    int64_t top_k;
    int64_t inter;        /* routed expert hidden */

    int64_t loop_start, loop_end, loop_passes, gate_rank;

    float rms_eps;
    float qk_clip;
    float rel_clip;
    float log_floor;
    float log_alpha;
    float route_scale;
    float gate_softcap;
    float up_softcap;
    float final_softcap;

    int64_t eos_id;
    int64_t image_token_id;
    bool is_swa[IK_MAX_LAYERS];

    /* vision tower */
    int64_t v_hidden, v_layers, v_heads, v_inter, v_patch, v_merge, v_grid, v_proj;
    float v_ln_eps;
} ik_cfg;

/* One attention layer's weights. Everything that is multiplied by a vector is
 * BF16 row-major [out][in]; norms, clips, small tables and the gate correction
 * stay F32 because the reference computes them in F32. */
typedef struct ik_attn {
    const void *wq;   /* [n_heads*head_dim, H] */
    const void *wk;   /* [n_kv*head_dim, H] */
    const void *wv;
    const void *wr;   /* [n_heads*d_rel, H] */
    const void *wo;   /* [H, n_heads*head_dim] */
    const void *gate; /* [n_heads*head_dim, H] */
    const void *sink; /* [n_heads, H] */
    float *q_scale;   /* [n_heads*head_dim] */
    float *rel_scale; /* [n_heads*d_rel] */
    float *rel_proj;  /* [d_rel][rel_extent] F32 */
    float *k_conv;    /* [k][n_kv*head_dim] taps, tap 0 = current token */
    float *v_conv;
    const void *k_conv_proj; /* [n_kv*sconv_k, H] (full layers only) */
    const void *v_conv_proj;
} ik_attn;

typedef enum {
    IK_EXPERT_BF16 = 0,
    IK_EXPERT_NVFP4 = 1,
    IK_EXPERT_MXFP8 = 2,
} ik_expert_format;

/* Resident checkpoint form of one expert projection. In packed mode the
 * weight, block scales and (for NVFP4) amax/denominator factors are copied into
 * the arena unchanged; global_scale[e] is the precomputed amax[e] / denominator. */
typedef struct ik_expert_pack {
    const void *packed;
    const uint8_t *scale;
    const float *global_scale;
    ik_expert_format format;
} ik_expert_pack;

typedef struct ik_layer {
    float *attn_norm;
    float *mlp_norm;
    float *q_in_norm, *k_in_norm, *v_in_norm; /* full layers */
    ik_attn a;

    /* Exactly one representation is resident: BF16 on CPU/escape-hatch runs,
     * or the checkpoint's packed NVFP4/MXFP8 bytes on packed Metal runs. */
    void *w13; /* BF16 [n_experts][2*inter][H]: gate half then up half */
    void *w2;  /* BF16 [n_experts][H][inter] */
    ik_expert_pack w13_packed, w2_packed;
    const void *gate_w; /* [n_experts, H] */
    float *gate_bias;   /* [n_experts] score correction */

    float *pa_conv; /* [H][sconv_k] depthwise taps, tap 0 = k-1 steps back */
    float *pm_conv;

    /* loop core only */
    bool is_loop;
    const void *ctrl[4][3]; /* [q,k,v,mlp][down(2H->rank), gain, forget] */
    float *msg_query;       /* [4][H] cross-pass retrieval queries */

    /* layer 0 has no attention-input query */
    float *attn_query; /* [nq][H], nq = 1 (sliding) or 3 (full), NULL on layer 0 */
    float *mlp_query;  /* [H] */
} ik_layer;

/* One vision encoder layer: standard pre-norm ViT with layer scale, exact GELU
 * and axial 2-D rotary positions. Weights are BF16 (they are not quantized). */
typedef struct ik_vlayer {
    float *ln1_w, *ln1_b, *ln2_w, *ln2_b;
    const void *q, *k, *v, *o;
    float *q_b, *k_b, *v_b, *o_b;
    const void *fc1, *fc2;
    float *fc1_b, *fc2_b;
    float *ls1, *ls2;
} ik_vlayer;

typedef struct ik_vision {
    const void *patch_embed;  /* [v_hidden, 3*patch*patch] */
    float *pos_embed;         /* [grid*grid][v_hidden] */
    float *class_embed, *cls_pos;
    float *pre_ln_w, *pre_ln_b;
    const void *proj_in;      /* [v_proj, merge^2*v_hidden] */
    const void *proj_out;     /* [text_hidden, v_proj] */
    ik_vlayer *layers;
    bool loaded;
} ik_vision;

typedef struct ik_model {
    ik_cfg c;
    cwick_safetensors saf;
    void *arena;
    size_t arena_bytes;
    bool arena_wired;
    bool packed_experts;
    bool dual_experts;
    uint8_t expert_format[IK_MAX_LAYERS]; /* ik_expert_format from hf_quant_config.json */
    uint8_t expert_seen[IK_MAX_LAYERS];   /* bit 0: w13, bit 1: w2 */

    const void *embed;      /* [vocab, H] */
    float *embed_norm;      /* [H] */
    const void *unembed;    /* [unpadded_vocab, H] */
    float *out_norm;        /* [H] */
    float *final_query;     /* [H] */

    ik_layer *layers;

    /* KV: full layers keep `cap` positions, sliding layers a ring of window.
     * Indexed by cache slot, see IK_MAX_SLOTS. */
    int64_t n_slots;
    int64_t slot[IK_MAX_LAYERS];      /* cache slot of layer L on pass 1 */
    int64_t loop_slot0;               /* first appended loop-pass block */
    float *kv_k[IK_MAX_SLOTS];
    float *kv_v[IK_MAX_SLOTS];
    int64_t kv_cap[IK_MAX_SLOTS];
    int64_t kv_len[IK_MAX_SLOTS];     /* sliding: how many written, saturates */
    int64_t kv_head[IK_MAX_SLOTS];    /* sliding ring head */
    /* short-convolution history: [3 taps][channels] per slot */
    float *conv_hist[IK_MAX_SLOTS][4];
    int64_t conv_ch[IK_MAX_SLOTS][4];

    int64_t cap;       /* full-attention capacity in positions */
    int64_t pos;       /* next absolute position */
    float *logits;
    uint64_t gpu_calls;

    /* Image embeddings for the turn being prefilled. The checkpoint's own
     * model splices the tower's output over the placeholder rows *after* the
     * embedding norm, so these rows replace the lookup instead of being added
     * to it. `img_used` walks the list as the placeholders go by. */
    const float *img_embeds;
    int64_t img_tokens;
    int64_t img_used;

    /* vision */
    struct ik_vision *vis;
} ik_model;

/* ik.c */
int64_t ik_cache_bytes(const ik_model *m, int64_t cap);
int ik_model_open(ik_model *m, const char *root, int64_t cap, cwick_error *err);
void ik_model_close(ik_model *m);
void ik_model_reset(ik_model *m);
void ik_message_mix(const ik_cfg *c, const float *prev_blocks, int pre_nb,
                    const float *prev_h, const float *query, float *bank,
                    float *norm_buf, float *score_buf, float *out);
void ik_expert_activation(float *out, const float *gate, const float *up,
                          int64_t n, float gamma, float gate_cap, float up_cap);
void ik_round_bf16(float *x, int64_t n);
void ik_rms_norm_bf16(float *out, const float *in, int64_t n,
                      const float *weight, float eps);
void ik_static_conv_taps(float *dst, const uint16_t *src, int64_t k, int64_t ch);
float ik_vision_pixel(float raw, int channel);
void ik_control(const ik_cfg *c, const ik_layer *ly, int ctrl,
                const float *cur, const float *msg, float *out,
                float *cat, float *gate, float *gf, float *gm);
int ik_model_forward(ik_model *m, const int32_t *ids, int64_t n, int64_t pos,
                     float *logits, cwick_error *err);

/* ik_prompt.c */
int ik_prompt_tools_json(char *out, size_t cap, const char *tools_json);
cwick_status ik_chat_encode(const cwick_tokenizer *tok, const char *const *roles,
                            const char *const *contents, size_t n,
                            const char *tools_block, uint32_t **ids,
                            int64_t *n_out, cwick_error *err);

/* ik_vision.c */
int ik_vision_open(ik_model *m, cwick_error *err);
void ik_vision_close(ik_model *m);
/* Decode an image (PNG/JPEG/WebP — whatever ImageIO reads) and run the tower
 * exactly like the checkpoint's processor: pad to 40 px blocks with the
 * checkpoint's -1 fill, cut into blocks in raster order, bilinear-interpolate
 * the 32x32 position table, pre-LN ViT, then the merger. `embeds` receives
 * rows*cols vectors of `hidden` floats — one per image placeholder token. */
int ik_vision_embed(ik_model *m, const uint8_t *data, size_t len,
                    float **embeds, int64_t *n_tokens, int64_t *rows,
                    int64_t *cols, cwick_error *err);
/* Same, from an already-decoded RGB image (channel-interleaved, 0..1). */
int ik_vision_embed_rgb(ik_model *m, const float *rgb, int64_t height,
                        int64_t width, float **embeds, int64_t *n_tokens,
                        int64_t *rows, int64_t *cols, cwick_error *err);

#endif /* CWICK_IK_H */
