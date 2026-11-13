// q35_gpu.c — the layer schedule of a Qwen3.5 decode step, encoded for the GPU.
//
// Why this exists, in one paragraph. A decode step reads ~16 GB of weights and
// does almost no arithmetic per byte, so it is a bandwidth problem — and the
// CPU cluster's path to memory on this machine is 123 GB/s while the GPU's is
// 285-329. The catch is that the GPU only reaches that rate when a step is
// submitted as one command buffer: measured, 128 separate submissions move the
// same bytes in 118.8 ms (116 GB/s) while one submission moves them in
// 37.6 ms (368 GB/s), because a round trip costs ~928 us and each of the 128
// pays it. So this file encodes the *entire* step — 32 layers, norms, the
// recurrence, attention, the MLP, the head — into one encoder and waits once.
//
// That decision dictates everything else here:
//
//   * intermediates never leave the device during a step. There is no host
//     readback until the logits;
//   * the DeltaNet state, the convolution window and the KV cache must be
//     GPU-addressable, so they are adopted as no-copy views over the buffers
//     q35_model_open already owns. Apple Silicon shares memory, so the CPU
//     prefill and the GPU decode write the same bytes and there is no
//     synchronisation between them to get wrong;
//   * every operator is a kernel in q35_shader.c whose counterpart in
//     q35_model.c defines the arithmetic.
//
// This unit is plain C: it knows the schedule, not the Metal runtime. The
// encode calls at the top are implemented in q35_gpu.m.

#include "cwick.h"
#include "log.h"
#include "q35.h"
#include "q35_gpu.h"
#include "util.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- the ObjC side (q35_gpu.m) -------------------------------------- */
int      q35m_init(void);
void     q35m_shutdown(void);
uint64_t q35m_max_buffer(void);
void    *q35m_alloc(size_t bytes, int shared, void **host);
void     q35m_free(void *h);
int      q35m_resolve(const void *p, size_t len, void **buf, size_t *off);
int      q35m_begin(void);
int      q35m_flush(void);
void     q35m_abort(void);

int q35m_gemv(void *W, size_t woff, uint32_t k, uint32_t cols,
              void *X, size_t xoff, void *O, size_t ooff);
int q35m_gemv_n_prepare(uint32_t n);
int q35m_barrier(void);
int q35m_gemv_n(void *W, size_t woff, uint32_t k, uint32_t cols,
                void *X, size_t xoff, void *O, size_t ooff, uint32_t n,
                uint32_t sx, uint32_t so);
int q35m_rmsnorm(void *X, size_t xoff, void *W, size_t woff,
                 void *O, size_t ooff, uint32_t n, float eps, int add_one);
/* The row-batched forms. Same arithmetic, one dispatch for all rows; the
 * strides are element counts between rows. A k-row pass uses these instead of
 * looping the single-row kernels, because the loop's cost was launch latency
 * rather than work (see q35_shader.c). */
int q35m_rmsnorm_rows(void *X, size_t xoff, void *W, size_t woff,
                      void *O, size_t ooff, uint32_t n, float eps, int add_one,
                      uint32_t rows, uint32_t sx, uint32_t so);
int q35m_add_rows(void *X, size_t xoff, void *Y, size_t yoff, uint32_t n,
                  uint32_t rows, uint32_t sx, uint32_t sy);
int q35m_silu_mul_rows(void *G, size_t go, void *U, size_t uo, uint32_t n,
                       uint32_t rows, uint32_t sx, uint32_t sy);
int q35m_l2norm_rows(void *Y, size_t yoff, uint32_t nheads, uint32_t qheads,
                     float eps, float qscale, uint32_t rows, uint32_t sx);
int q35m_gdn_gates_rows(void *A, size_t aoff, void *B, size_t boff, void *Alog,
                        size_t logoff, void *Dtb, size_t dtoff, void *Decay,
                        size_t decoff, void *Beta, size_t betoff, uint32_t nh,
                        uint32_t rows, uint32_t sa, uint32_t sb, uint32_t sd,
                        uint32_t sbe);
int q35m_add(void *X, size_t xoff, void *Y, size_t yoff, uint32_t n);
int q35m_embed(void *E, size_t eoff, void *X, size_t xoff, uint32_t dim,
               uint32_t id);
int q35m_conv_silu(void *Mix, size_t moff, void *Conv, size_t coff,
                   void *St, size_t soff, void *Stp, size_t stpoff,
                   void *Y, size_t yoff, uint32_t qkv, uint32_t taps);
int q35m_l2norm(void *Y, size_t yoff, uint32_t nheads, uint32_t qheads,
                float eps, float qscale);
int q35m_gdn_gates(void *A, size_t aoff, void *B, size_t boff, void *Alog,
                   size_t logoff, void *Dtb, size_t dtoff, void *Decay,
                   size_t decoff, void *Beta, size_t betoff, uint32_t nh);
int q35m_gdn(void *S, size_t soff, void *Sp, size_t spoff, void *Qkv,
             size_t qoff, void *Z, size_t zoff, void *Nw, size_t nwoff,
             void *Decay, size_t decoff, void *Beta, size_t betoff, void *Out,
             size_t ooff, uint32_t nh, uint32_t hk, uint32_t hv,
             uint32_t key_dim, uint32_t val_dim);
int q35m_qk(void *Qsrc, size_t qso, void *Ksrc, size_t kso, void *Qout,
            size_t qoo, void *Krow, size_t kro, void *Qnw, size_t qno,
            void *Knw, size_t kno, void *Cos, size_t co, void *Sin, size_t so,
            uint32_t nq, uint32_t nkv, uint32_t hd, uint32_t rotary, float eps);
int q35m_kv_store(void *Src, size_t so, void *Dst, size_t doff, uint32_t n);
int q35m_scores(void *Q, size_t qo, void *Kc, size_t ko, void *S, size_t so,
                uint32_t heads, uint32_t hd, uint32_t kv, uint32_t keys,
                uint32_t group, float scale);
int q35m_softmax(void *S, size_t so, uint32_t heads, uint32_t keys);
int q35m_av(void *S, size_t so, void *Vc, size_t vo, void *Part, size_t po,
            uint32_t heads, uint32_t hd, uint32_t kv, uint32_t keys,
            uint32_t group, uint32_t chunk, uint32_t nchunks);
int q35m_av_combine(void *Part, size_t po, void *O, size_t oo,
                    uint32_t nchunks, uint32_t heads, uint32_t hd);
int q35m_attn_gate(void *O, size_t oo, void *Qg, size_t qo,
                   uint32_t nh, uint32_t hd);
int q35m_silu_mul(void *G, size_t go, void *U, size_t uo, uint32_t n);
int q35m_gemv_rows(void *W, size_t woff, uint32_t k, uint32_t cols,
                   void *Idx, size_t ioff, void *X, size_t xoff,
                   void *O, size_t ooff);
int q35m_embed_id(void *E, size_t eoff, void *X, size_t xoff,
                  void *Id, size_t idoff, uint32_t dim);
int q35m_argmax(void *S, size_t soff, void *Idx, size_t ioff,
                void *Out, size_t ooff, uint32_t n);

/* ------------------------------------------------------------------ */
/* binding                                                             */
/* ------------------------------------------------------------------ */

typedef struct {
    void  *b;     /* MTLBuffer handle, unretained: q35_gpu.m owns it */
    size_t o;     /* byte offset of the tensor inside that buffer */
} gbuf;

/* Weight slots. The linear-attention and full-attention groups are mutually
 * exclusive per layer, so an unbound slot is simply never read. */
enum {
    W_QKV, W_Z, W_A, W_B, W_O,
    W_WQ, W_WK, W_WV, W_WO,
    W_GATE, W_UP, W_DOWN,
    W_KC, W_VC, W_CST, W_SSM,
    W_N
};

/* f32 helper slots per layer, all packed into one host-visible buffer. */
enum { F_IN, F_POST, F_GN, F_ALOG, F_DTB, F_QN, F_KN, F_CONV, F_N };

static gbuf    g_w[Q35_MAX_LAYERS][W_N];
static gbuf    g_embed, g_head;
static gbuf    g_final, g_rope;
static size_t  g_foff[Q35_MAX_LAYERS][F_N];
static size_t  g_rope_n;
static void   *g_fbuf, *g_fhost;
static int     g_ok;
static uint64_t g_bound_bytes;
/* The model the current pass belongs to, and how many rows its last multi-row
 * pass ran. q35_gpu_commit_rows needs both: it moves the state slot the caller
 * chose into slot 0, which is the only copy a rejected draft costs. */
static struct q35_model *g_model;
static int     g_last_n;

static struct {
    gbuf   x, xn, mix, gate, up, down, att, qb, kb, vb, lin;
    gbuf   araw, braw, decay, beta, scores, part, logits;
    void  *logits_host;
    size_t scores_cap, part_cap;
} g_s;

/* ---- MTP draft head -------------------------------------------------- *
 * One block, so its weights get a flat set of slots instead of the per-layer
 * table the 32 text layers need. The f32 parameters (five norms plus the two
 * per-head ones) live in their own host-visible buffer for the same reason. */
enum { M_FC, M_WQ, M_WK, M_WV, M_WO, M_GATE, M_UP, M_DOWN, M_N };
enum { MF_EMB, MF_HID, MF_IN, MF_POST, MF_OUT, MF_QN, MF_KN, MF_N };

static gbuf    g_mtp[M_N];
/* The block's KV cache has to be a buffer handle like every other tensor the
 * kernels touch: q35m_* take MTLBuffer objects, not host pointers. */
static gbuf    g_mtpkc, g_mtpvc;
static void   *g_mtpbuf, *g_mtphost;
static size_t  g_mtpoff[MF_N];
static int     g_mtp_ok;

/* Its own scratch, dimensioned for one row: a draft chain is sequential by
 * construction, so nothing here is ever N rows deep. */
static struct {
    gbuf   e;       /* [2D]  concat(pre_norm_emb(emb), pre_norm_hid(h))   */
    gbuf   res;     /* [D]   residual stream                                */
    gbuf   xn;      /* [D]   normalised copy                                */
    gbuf   mix;     /* [qwidth] q | gate                                    */
    gbuf   qb;      /* [qdim] q after its per-head norm and the rotation    */
    gbuf   kv;      /* [2*kvw]  k then v                                    */
    gbuf   gate;    /* [I]                                                  */
    gbuf   up;      /* [I]                                                  */
    gbuf   down;    /* [D]                                                  */
    gbuf   att;     /* [qdim]                                               */
    gbuf   cand;    /* uint [vocab]  candidate row ids                      */
    gbuf   ids;     /* uint [Q35_GK+1] host-visible: the chain's tokens     */
    gbuf   sc;      /* float [vocab] host-visible scores                    */
    void  *cand_host, *ids_host, *sc_host;
    int    cand_cap;
} g_m;

void q35_gpu_close(void) {
    /* Safe whether or not the backend ever came up, and safe to call twice:
     * q35_model_close runs it unconditionally, and a partial open calls it
     * from its own failure path. */
    g_ok = 0;
    g_fbuf = NULL;
    g_fhost = NULL;
    g_mtpbuf = NULL;
    g_mtphost = NULL;
    g_mtp_ok = 0;
    memset(&g_m, 0, sizeof(g_m));
    memset(g_mtp, 0, sizeof(g_mtp));
    memset(g_mtpoff, 0, sizeof(g_mtpoff));
    memset(&g_s, 0, sizeof(g_s));
    q35m_shutdown();
}

static int bind_tensor(gbuf *g, const void *p, size_t bytes) {
    void *b = NULL;
    size_t o = 0;
    if (!p || bytes == 0) return 0;
    if (!q35m_resolve(p, bytes, &b, &o)) return 0;
    g->b = b;
    g->o = o;
    g_bound_bytes += bytes;
    return 1;
}

#define ALLOC_SCRATCH(g, bytes) do {                          \
        (g).b = q35m_alloc((bytes), 0, NULL);                 \
        (g).o = 0;                                            \
        if (!(g).b) goto fail;                                \
    } while (0)

static uint64_t f32_span(const q35_config *c) {
    const int D = c->dim;
    const int qkv_dim = 2 * c->lin_k_heads * c->lin_k_dim +
                        c->lin_v_heads * c->lin_v_dim;
    const int taps = c->conv_kernel > 0 ? c->conv_kernel : 4;
    uint64_t per_layer = (uint64_t)2 * D + (uint64_t)c->lin_v_dim +
                         2u * c->lin_v_heads + 2u * c->head_dim +
                         (uint64_t)qkv_dim * taps;
    /* One rotation table per position, Q35_GK of them: a verify row is a
     * position and each needs its own cos/sin. */
    return per_layer * (uint64_t)c->n_layers + (uint64_t)D +
           (uint64_t)Q35_GK * 2u * c->rotary_dim;
}

static int bind_params(q35_model *m, const q35_config *c) {
    const int D = c->dim;
    const int qkv_dim = 2 * c->lin_k_heads * c->lin_k_dim +
                        c->lin_v_heads * c->lin_v_dim;
    const int taps = c->conv_kernel > 0 ? c->conv_kernel : 4;
    uint64_t floats = f32_span(c);
    if (floats > (uint64_t)SIZE_MAX / sizeof(float)) return 0;

    void *host = NULL;
    g_fbuf = q35m_alloc((size_t)floats * sizeof(float), 1, &host);
    if (!g_fbuf || !host) return 0;
    memset(host, 0, (size_t)floats * sizeof(float));
    g_fhost = host;
    float *f = (float *)host;
    size_t off = 0;

    for (int l = 0; l < c->n_layers; l++) {
        const q35_layer *L = &m->layers[l];
        memcpy(f + off, L->in_norm, (size_t)D * sizeof(float));
        g_foff[l][F_IN] = off; off += (size_t)D;
        memcpy(f + off, L->post_norm, (size_t)D * sizeof(float));
        g_foff[l][F_POST] = off; off += (size_t)D;
        if (L->is_linear) {
            memcpy(f + off, L->lin.norm, (size_t)c->lin_v_dim * sizeof(float));
            g_foff[l][F_GN] = off; off += (size_t)c->lin_v_dim;
            memcpy(f + off, L->lin.a_log, (size_t)c->lin_v_heads * sizeof(float));
            g_foff[l][F_ALOG] = off; off += (size_t)c->lin_v_heads;
            memcpy(f + off, L->lin.dt_bias, (size_t)c->lin_v_heads * sizeof(float));
            g_foff[l][F_DTB] = off; off += (size_t)c->lin_v_heads;
            /* Transpose [taps][qkv] to [qkv][taps]: the convolution kernel
             * gives one thread a whole channel, and the checkpoint's layout
             * would put the lanes of a warp a whole plane apart. */
            for (int ch = 0; ch < qkv_dim; ch++)
                for (int k = 0; k < taps; k++)
                    f[off + (size_t)ch * taps + k] =
                        L->lin.conv[(size_t)k * qkv_dim + ch];
            g_foff[l][F_CONV] = off;
            off += (size_t)qkv_dim * (size_t)taps;
        } else {
            memcpy(f + off, L->attn.q_norm, (size_t)c->head_dim * sizeof(float));
            g_foff[l][F_QN] = off; off += (size_t)c->head_dim;
            memcpy(f + off, L->attn.k_norm, (size_t)c->head_dim * sizeof(float));
            g_foff[l][F_KN] = off; off += (size_t)c->head_dim;
        }
    }
    g_final.b = g_fbuf; g_final.o = off * sizeof(float);
    memcpy(f + off, m->final_norm, (size_t)D * sizeof(float));
    off += (size_t)D;
    g_rope.b = g_fbuf; g_rope.o = off * sizeof(float);
    g_rope_n = (size_t)c->rotary_dim;
    off += (size_t)Q35_GK * 2u * g_rope_n;
    return off <= (size_t)floats;
}

/* The draft block's weights and its own parameters. Returns 0 when something
 * does not fit, which is not a failure of the backend: the decoder simply
 * drafts by prompt lookup instead. */
static int bind_mtp(q35_model *m, const q35_config *c) {
    if (!m->mtp_ok) return 1;
    const size_t D = (size_t)c->dim;
    const size_t I = (size_t)c->inter_dim;
    const size_t HD = (size_t)c->head_dim;
    const size_t kvw = (size_t)c->n_kv_heads * HD;
    const size_t qdim = (size_t)c->n_heads * HD;
    const size_t qw = qdim * (size_t)(1 + c->attn_gate);

    if (!bind_tensor(&g_mtp[M_FC], m->mtp.fc, D * 2 * D * 2)) return 0;
    if (!bind_tensor(&g_mtp[M_WQ], m->mtp.wq, qw * D * 2)) return 0;
    if (!bind_tensor(&g_mtp[M_WK], m->mtp.wk, kvw * D * 2)) return 0;
    if (!bind_tensor(&g_mtp[M_WV], m->mtp.wv, kvw * D * 2)) return 0;
    if (!bind_tensor(&g_mtp[M_WO], m->mtp.wo, D * qdim * 2)) return 0;
    if (!bind_tensor(&g_mtp[M_GATE], m->mtp.gate, I * D * 2)) return 0;
    if (!bind_tensor(&g_mtp[M_UP], m->mtp.up, I * D * 2)) return 0;
    if (!bind_tensor(&g_mtp[M_DOWN], m->mtp.down, D * I * 2)) return 0;

    /* Five D-wide norms and two head-wide ones, in one host-visible buffer so
     * every kernel can take them by offset. */
    const size_t floats = 5 * D + 2 * HD;
    void *host = NULL;
    g_mtpbuf = q35m_alloc(floats * sizeof(float), 1, &host);
    if (!g_mtpbuf || !host) return 0;
    g_mtphost = host;
    float *f = (float *)host;
    size_t off = 0;
#define PUT(slot, src, n)                                                     \
    do {                                                                      \
        memcpy(f + off, (src), (size_t)(n) * sizeof(float));                  \
        g_mtpoff[slot] = off;                                                 \
        off += (size_t)(n);                                                   \
    } while (0)
    PUT(MF_EMB, m->mtp.pre_norm_emb, D);
    PUT(MF_HID, m->mtp.pre_norm_hid, D);
    PUT(MF_IN, m->mtp.in_norm, D);
    PUT(MF_POST, m->mtp.post_norm, D);
    PUT(MF_OUT, m->mtp.out_norm, D);
    PUT(MF_QN, m->mtp.q_norm, HD);
    PUT(MF_KN, m->mtp.k_norm, HD);
#undef PUT
    return off == floats;
}

static int mtp_scratch(q35_model *m) {
    if (!m) return 0;
    const q35_config *c = &m->cfg;
    const size_t D = (size_t)c->dim;
    const size_t I = (size_t)c->inter_dim;
    const size_t HD = (size_t)c->head_dim;
    const size_t kvw = (size_t)c->n_kv_heads * HD;
    const size_t qdim = (size_t)c->n_heads * HD;
    const size_t qw = qdim * (size_t)(1 + c->attn_gate);
    const size_t V = (size_t)c->vocab_size;

    ALLOC_SCRATCH(g_m.e, 2 * D * sizeof(float));
    ALLOC_SCRATCH(g_m.res, D * sizeof(float));
    ALLOC_SCRATCH(g_m.xn, D * sizeof(float));
    ALLOC_SCRATCH(g_m.mix, qw * sizeof(float));
    ALLOC_SCRATCH(g_m.qb, qdim * sizeof(float));
    ALLOC_SCRATCH(g_m.kv, 2 * kvw * sizeof(float));
    ALLOC_SCRATCH(g_m.gate, I * sizeof(float));
    ALLOC_SCRATCH(g_m.up, I * sizeof(float));
    ALLOC_SCRATCH(g_m.down, D * sizeof(float));
    ALLOC_SCRATCH(g_m.att, qdim * sizeof(float));

    /* Neither cache is ever read by the host, so they are bound where they
     * were allocated rather than adopted from a mapping. */
    if (!bind_tensor(&g_mtpkc, m->mtp_kcache,
                     (size_t)m->mtp_win * kvw * sizeof(float))) return 0;
    if (!bind_tensor(&g_mtpvc, m->mtp_vcache,
                     (size_t)m->mtp_win * kvw * sizeof(float))) return 0;

    g_m.cand_cap = (int)V;
    g_m.cand.b = q35m_alloc(V * sizeof(uint32_t), 1, &g_m.cand_host);
    g_m.cand.o = 0;
    if (!g_m.cand.b || !g_m.cand_host) goto fail;
    g_m.ids.b = q35m_alloc((size_t)(Q35_GK + 1) * sizeof(uint32_t), 1,
                           &g_m.ids_host);
    g_m.ids.o = 0;
    if (!g_m.ids.b || !g_m.ids_host) goto fail;
    g_m.sc.b = q35m_alloc(V * sizeof(float), 1, &g_m.sc_host);
    g_m.sc.o = 0;
    if (!g_m.sc.b || !g_m.sc_host) goto fail;
    /* The identity candidate list, so the "score every row" arm of an A/B is
     * the same code path as the restricted one. */
    for (size_t i = 0; i < V; i++) ((uint32_t *)g_m.cand_host)[i] = (uint32_t)i;
    g_mtp_ok = 1;
    return 1;
fail:
    return 0;
}

static int bind_layer_weights(q35_model *m, const q35_config *c, int l) {    const q35_layer *L = &m->layers[l];
    const size_t kv_row = (size_t)c->n_kv_heads * c->head_dim;
    const size_t taps = (size_t)(c->conv_kernel > 0 ? c->conv_kernel : 4);
    if (L->is_linear) {
        const size_t key_dim = (size_t)c->lin_k_heads * c->lin_k_dim;
        const size_t val_dim = (size_t)c->lin_v_heads * c->lin_v_dim;
        const size_t qkv_dim = 2 * key_dim + val_dim;
        if (!bind_tensor(&g_w[l][W_QKV], L->lin.w_qkv, qkv_dim * c->dim * 2)) return 0;
        if (!bind_tensor(&g_w[l][W_Z], L->lin.w_z, val_dim * c->dim * 2)) return 0;
        if (!bind_tensor(&g_w[l][W_A], L->lin.w_a,
                         (size_t)c->lin_v_heads * c->dim * 2)) return 0;
        if (!bind_tensor(&g_w[l][W_B], L->lin.w_b,
                         (size_t)c->lin_v_heads * c->dim * 2)) return 0;
        if (!bind_tensor(&g_w[l][W_O], L->lin.w_o, (size_t)c->dim * val_dim * 2)) return 0;
        /* The recurrent state is bound whole, all Q35_STATE_SLOTS of it: a
         * verify row writes the slot after the one it read, so a one-slot view
         * would put every row past the first out of bounds. */
        if (!bind_tensor(&g_w[l][W_CST], m->conv_state[l],
                         (size_t)Q35_STATE_SLOTS * (size_t)(taps - 1) *
                             qkv_dim * sizeof(float))) return 0;
        if (!bind_tensor(&g_w[l][W_SSM], m->ssm_state[l],
                         (size_t)Q35_STATE_SLOTS * (size_t)c->lin_v_heads *
                             c->lin_k_dim * c->lin_v_dim * sizeof(float)))
            return 0;
    } else {
        const size_t qwidth = (size_t)c->n_heads * c->head_dim * (1 + c->attn_gate);
        const size_t kvw = (size_t)c->n_kv_heads * c->head_dim;
        const size_t qdim = (size_t)c->n_heads * c->head_dim;
        if (!bind_tensor(&g_w[l][W_WQ], L->attn.wq, qwidth * c->dim * 2)) return 0;
        if (!bind_tensor(&g_w[l][W_WK], L->attn.wk, kvw * c->dim * 2)) return 0;
        if (!bind_tensor(&g_w[l][W_WV], L->attn.wv, kvw * c->dim * 2)) return 0;
        if (!bind_tensor(&g_w[l][W_WO], L->attn.wo, (size_t)c->dim * qdim * 2)) return 0;
        if (!bind_tensor(&g_w[l][W_KC], m->kcache[l],
                         (size_t)c->max_seq_len * kv_row * sizeof(float))) return 0;
        if (!bind_tensor(&g_w[l][W_VC], m->vcache[l],
                         (size_t)c->max_seq_len * kv_row * sizeof(float))) return 0;
    }
    if (!bind_tensor(&g_w[l][W_GATE], L->w_gate, (size_t)c->inter_dim * c->dim * 2)) return 0;
    if (!bind_tensor(&g_w[l][W_UP], L->w_up, (size_t)c->inter_dim * c->dim * 2)) return 0;
    if (!bind_tensor(&g_w[l][W_DOWN], L->w_down, (size_t)c->dim * c->inter_dim * 2)) return 0;
    return 1;
}

/* ------------------------------------------------------------------ */
/* open                                                                */
/* ------------------------------------------------------------------ */

int q35_gpu_available(void) {
    /* On by default wherever a Metal device comes up, because the step is
     * 2.9x the CPU pool and produces the same text (see docs/PERF.md). The
     * switch exists for A/B measurement and for `--backend cpu`, which turns
     * cwick_metal_backend_active() off anyway. */
    const char *e = getenv("CWICK_Q35_GPU");
    if (e && *e && (*e == '0' || strcmp(e, "off") == 0)) return 0;
    return cwick_metal_backend_active();
}

int q35_gpu_open(q35_model *m) {
    const q35_config *c = &m->cfg;
    if (!m || !m->loaded) return -1;
    if (!cwick_metal_backend_active()) return -1;
    if (!q35m_init()) return -1;

    const size_t D = (size_t)c->dim;
    const size_t I = (size_t)c->inter_dim;
    size_t wide = (size_t)c->lin_v_heads * c->lin_v_dim;
    if (I > wide) wide = I;
    const size_t kvw = (size_t)c->n_kv_heads * c->head_dim;
    /* Every activation is position-major and Q35_GK rows deep: a verify step
     * produces all of them at once, and the slots that are per-row scratch
     * (scores, the softmax partials) are reused in row order instead. */
    const size_t R = (size_t)Q35_GK;

    ALLOC_SCRATCH(g_s.x, R * D * sizeof(float));
    ALLOC_SCRATCH(g_s.xn, R * D * sizeof(float));
    ALLOC_SCRATCH(g_s.mix, R * 2 * wide * sizeof(float));
    ALLOC_SCRATCH(g_s.gate, R * I * sizeof(float));
    ALLOC_SCRATCH(g_s.up, R * I * sizeof(float));
    ALLOC_SCRATCH(g_s.down, R * D * sizeof(float));
    ALLOC_SCRATCH(g_s.att, R * D * sizeof(float));
    ALLOC_SCRATCH(g_s.qb, R * D * sizeof(float));
    ALLOC_SCRATCH(g_s.kb, R * kvw * sizeof(float));
    ALLOC_SCRATCH(g_s.vb, R * kvw * sizeof(float));
    ALLOC_SCRATCH(g_s.lin, R * wide * sizeof(float));
    ALLOC_SCRATCH(g_s.araw, R * (size_t)c->lin_v_heads * sizeof(float));
    ALLOC_SCRATCH(g_s.braw, R * (size_t)c->lin_v_heads * sizeof(float));
    ALLOC_SCRATCH(g_s.decay, R * (size_t)c->lin_v_heads * sizeof(float));
    ALLOC_SCRATCH(g_s.beta, R * (size_t)c->lin_v_heads * sizeof(float));

    g_s.scores_cap = (size_t)c->n_heads * (size_t)c->max_seq_len;
    ALLOC_SCRATCH(g_s.scores, g_s.scores_cap * sizeof(float));
    g_s.part_cap = (size_t)64 * (size_t)c->n_heads * (size_t)c->head_dim;
    ALLOC_SCRATCH(g_s.part, g_s.part_cap * sizeof(float));

    g_s.logits.b = q35m_alloc(R * (size_t)c->vocab_size * sizeof(float), 1,
                              &g_s.logits_host);
    g_s.logits.o = 0;
    if (!g_s.logits.b || !g_s.logits_host) goto fail;

    if (!bind_params(m, c)) goto fail;
    if (!bind_tensor(&g_embed, m->embed, (size_t)c->vocab_size * c->dim * 2)) {
        CWICK_INFO("q35gpu", "embedding table (%.2f GiB) exceeds the %.2f GiB "
                   "buffer limit; decode stays on the CPU",
                   (double)((size_t)c->vocab_size * c->dim * 2) / 1073741824.0,
                   (double)q35m_max_buffer() / 1073741824.0);
        goto fail;
    }
    if (!bind_tensor(&g_head, m->head, (size_t)c->vocab_size * c->dim * 2)) goto fail;

    g_bound_bytes = 0;
    for (int l = 0; l < c->n_layers; l++)
        if (!bind_layer_weights(m, c, l)) {
            CWICK_INFO("q35gpu", "layer %d could not be adopted by the GPU "
                       "(a tensor past the %.2f GiB buffer limit, or a weight "
                       "that cannot be wrapped); decode stays on the CPU",
                       l, (double)q35m_max_buffer() / 1073741824.0);
            goto fail;
        }
    if (!bind_mtp(m, c) || !mtp_scratch(m)) {
        /* The drafter is an optimisation and the rest of the step does not
         * depend on it, so a head that will not fit costs acceptance, not
         * service. */
        CWICK_INFO("q35gpu", "MTP draft head could not be adopted; drafts fall "
                   "back to prompt lookup");
        g_mtp_ok = 0;
        memset(g_mtp, 0, sizeof(g_mtp));
    }
    g_ok = 1;
    CWICK_INFO("q35gpu", "decode steps on the GPU: %d layers in one command "
               "buffer, %.2f GiB adopted without a copy%s",
               c->n_layers, (double)g_bound_bytes / 1073741824.0,
               g_mtp_ok ? ", MTP drafter ready" : "");
    return 0;

fail:
    q35_gpu_close();
    return -1;
}

/* ------------------------------------------------------------------ */
/* one step, n rows                                                    */
/* ------------------------------------------------------------------ */
//
// A verify step for speculative decoding: k consecutive positions, one weight
// pass. The order below is what makes that legal. Every projection that feeds
// the *same* activation runs once for all k rows (`proj_n`), and everything
// that carries state -- the convolution window, the DeltaNet recurrence, the
// KV cache -- runs per row, in position order, so the state a row sees is the
// state k-1 single steps would have left behind. Sharing is only ever about
// the weight read, never about the state.

typedef struct {
    const q35_config *c;
    int D, I;
    int key_dim, val_dim, qkv_dim;
    int H, KV, HD, qdim, kvw, group, qwidth, taps, NH;
    size_t bD, bI, bWide, bMix, bKvw, bNH, bVocab;   /* per-row strides, bytes */
    size_t slot_conv, slot_ssm;   /* one copy of the recurrent state */
} sctx;

/* Row r of a position-major scratch buffer. */
static gbuf row_at(gbuf g, size_t stride, int r) {
    gbuf o = g;
    o.o += (size_t)r * stride;
    return o;
}

/* Debug switch: run the one-row step through the k-row kernel, so a mismatch
 * between the two GEMV kernels cannot be confused with a scheduling one. */
static int q35_gpu_tiled1(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("CWICK_Q35_GPU_TILED1");
        cached = (e && *e && *e != '0') ? 1 : 0;
    }
    return cached;
}

/* CWICK_Q35_ROWFUSE=0 puts the row-local kernels back to one dispatch per row.
 * They are a pure batching change -- same arithmetic, same sums, same order --
 * so the only observable difference is launch count; the switch exists to keep
 * that claim testable in one binary. */
static int rowfuse(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("CWICK_Q35_ROWFUSE");
        cached = (e && *e) ? (cwick_env_true("CWICK_Q35_ROWFUSE") ? 1 : 0) : 1;
    }
    return cached;
}

/* One dispatch for all n rows of a row-local kernel, or n dispatches. `xs`/`os`
 * are byte strides between rows. */
static int norm_all(const sctx *s, gbuf X, void *wb, size_t wo, gbuf O, int n,
                    float eps, int add_one, size_t xs, size_t os) {
    if (n > 1 && rowfuse())
        return q35m_rmsnorm_rows(X.b, X.o, wb, wo, O.b, O.o, (uint32_t)s->D,
                                 eps, add_one, (uint32_t)n,
                                 (uint32_t)(xs / sizeof(float)),
                                 (uint32_t)(os / sizeof(float)));
    for (int r = 0; r < n; r++) {
        gbuf x = row_at(X, xs, r);
        gbuf o = row_at(O, os, r);
        if (!q35m_rmsnorm(x.b, x.o, wb, wo, o.b, o.o, (uint32_t)s->D, eps,
                          add_one))
            return 0;
    }
    return 1;
}

static int add_all(const sctx *s, gbuf X, gbuf Y, int n, size_t xs, size_t ys) {
    if (n > 1 && rowfuse())
        return q35m_add_rows(X.b, X.o, Y.b, Y.o, (uint32_t)s->D, (uint32_t)n,
                             (uint32_t)(xs / sizeof(float)),
                             (uint32_t)(ys / sizeof(float)));
    for (int r = 0; r < n; r++) {
        gbuf x = row_at(X, xs, r);
        gbuf y = row_at(Y, ys, r);
        if (!q35m_add(x.b, x.o, y.b, y.o, (uint32_t)s->D)) return 0;
    }
    return 1;
}

/* One projection over all n rows. `xstride`/`ostride` are the row spacings of
 * the two scratch buffers in bytes: they are allocation widths, not output
 * widths, because a buffer is reused by projections of different shapes (mix
 * holds the linear qkv row and the attention q row; gate holds the DeltaNet z
 * row and the MLP gate row). n == 1 keeps the plain GEMV, which is the shape
 * the 20.4 tok/s step was measured with. */
static int proj_n(const sctx *s, gbuf W, gbuf X, size_t xstride,
                  gbuf O, size_t ostride, int n, uint32_t k, uint32_t cols) {
    (void)s;
    if (n == 1 && !q35_gpu_tiled1()) return q35m_gemv(W.b, W.o, k, cols, X.b,
                                                      X.o, O.b, O.o);
    return q35m_gemv_n(W.b, W.o, k, cols, X.b, X.o, O.b, O.o, (uint32_t)n,
                       (uint32_t)(xstride / sizeof(float)),
                       (uint32_t)(ostride / sizeof(float)));
}

/* The linear-attention half of a layer: four projections batched, then the
 * recurrence per row. */
static int step_linear_rows(const sctx *s, int l, int n) {
    const gbuf *W = g_w[l];
    const float qscale = 1.0f / sqrtf((float)s->c->lin_k_dim);
    if (!proj_n(s, W[W_QKV], g_s.xn, s->bD, g_s.mix, s->bMix, n,
                (uint32_t)s->D, (uint32_t)s->qkv_dim)) return 0;
    if (!proj_n(s, W[W_Z], g_s.xn, s->bD, g_s.gate, s->bI, n,
                (uint32_t)s->D, (uint32_t)s->val_dim)) return 0;
    if (!proj_n(s, W[W_A], g_s.xn, s->bD, g_s.araw, s->bNH, n,
                (uint32_t)s->D, (uint32_t)s->NH)) return 0;
    if (!proj_n(s, W[W_B], g_s.xn, s->bD, g_s.braw, s->bNH, n,
                (uint32_t)s->D, (uint32_t)s->NH)) return 0;
    /* Row r reads state slot r and writes slot r+1. A single row stays in
     * place at slot 0 -- there is nothing to choose between afterwards, and a
     * copy per step would be 50 MiB of pure overhead. A multi-row pass needs
     * the rolling slots because the caller keeps only the first j rows of a
     * draft and a DeltaNet cannot be rewound. */
    const int roll = n > 1;
    /* The decay and the write strength depend only on this row's own a_raw and
     * b_raw, so all n rows can be gated in one dispatch before the recurrence
     * starts. conv and gdn stay per row: they walk the state slots in order. */
    const int batch_gates = roll && rowfuse();
    if (batch_gates) {
        if (!q35m_gdn_gates_rows(g_s.araw.b, g_s.araw.o, g_s.braw.b,
                                 g_s.braw.o, g_fbuf,
                                 g_foff[l][F_ALOG] * sizeof(float), g_fbuf,
                                 g_foff[l][F_DTB] * sizeof(float),
                                 g_s.decay.b, g_s.decay.o, g_s.beta.b,
                                 g_s.beta.o, (uint32_t)s->NH, (uint32_t)n,
                                 (uint32_t)(s->bNH / sizeof(float)),
                                 (uint32_t)(s->bNH / sizeof(float)),
                                 (uint32_t)(s->bNH / sizeof(float)),
                                 (uint32_t)(s->bNH / sizeof(float))))
            return 0;
    }
    for (int r = 0; r < n; r++) {
        const size_t cs = roll ? (size_t)r * s->slot_conv : 0;
        const size_t cd = roll ? (size_t)(r + 1) * s->slot_conv : 0;
        const size_t ss = roll ? (size_t)r * s->slot_ssm : 0;
        const size_t sd = roll ? (size_t)(r + 1) * s->slot_ssm : 0;
        /* Rows are a chain through the same state slots; make the ordering
         * explicit (see q35m_barrier). */
        if (r > 0 && !q35m_barrier()) return 0;
        gbuf mix = row_at(g_s.mix, s->bMix, r);
        gbuf gate = row_at(g_s.gate, s->bI, r);
        gbuf up = row_at(g_s.up, s->bI, r);
        gbuf lin = row_at(g_s.lin, s->bWide, r);
        gbuf ar = row_at(g_s.araw, s->bNH, r);
        gbuf br = row_at(g_s.braw, s->bNH, r);
        gbuf dec = row_at(g_s.decay, s->bNH, r);
        gbuf bet = row_at(g_s.beta, s->bNH, r);
        if (!batch_gates &&
            !q35m_gdn_gates(ar.b, ar.o, br.b, br.o, g_fbuf,
                            g_foff[l][F_ALOG] * sizeof(float), g_fbuf,
                            g_foff[l][F_DTB] * sizeof(float),
                            dec.b, dec.o, bet.b, bet.o,
                            (uint32_t)s->NH)) return 0;
        if (!q35m_conv_silu(mix.b, mix.o, g_fbuf,
                            g_foff[l][F_CONV] * sizeof(float),
                            W[W_CST].b, W[W_CST].o + cd,
                            W[W_CST].b, W[W_CST].o + cs, up.b, up.o,
                            (uint32_t)s->qkv_dim, (uint32_t)s->taps)) return 0;
        if (!q35m_l2norm(up.b, up.o,
                         (uint32_t)(2 * s->c->lin_k_heads),
                         (uint32_t)s->c->lin_k_heads, s->c->l2_eps,
                         qscale)) return 0;
        if (!q35m_gdn(W[W_SSM].b, W[W_SSM].o + sd,
                      W[W_SSM].b, W[W_SSM].o + ss, up.b, up.o,
                      gate.b, gate.o, g_fbuf,
                      g_foff[l][F_GN] * sizeof(float),
                      dec.b, dec.o, bet.b, bet.o, lin.b, lin.o,
                      (uint32_t)s->NH, (uint32_t)s->c->lin_k_dim,
                      (uint32_t)s->c->lin_v_dim, (uint32_t)s->key_dim,
                      (uint32_t)s->val_dim)) return 0;
    }
    if (!proj_n(s, W[W_O], g_s.lin, s->bWide, g_s.down, s->bD, n,
                (uint32_t)s->val_dim, (uint32_t)s->D)) return 0;
    return 1;
}

/* The full-attention half: projections batched, then one query per row
 * against a KV cache that grows by one entry per row. */
static int step_full_rows(const sctx *s, int l, int n, int64_t pos) {
    const gbuf *W = g_w[l];
    const float scale = 1.0f / sqrtf((float)s->HD);
    const size_t rope_stride = 2u * g_rope_n * sizeof(float);
    if (!proj_n(s, W[W_WQ], g_s.xn, s->bD, g_s.mix, s->bMix, n,
                (uint32_t)s->D, (uint32_t)s->qwidth)) return 0;
    if (!proj_n(s, W[W_WK], g_s.xn, s->bD, g_s.kb, s->bKvw, n,
                (uint32_t)s->D, (uint32_t)s->kvw)) return 0;
    if (!proj_n(s, W[W_WV], g_s.xn, s->bD, g_s.vb, s->bKvw, n,
                (uint32_t)s->D, (uint32_t)s->kvw)) return 0;
    for (int r = 0; r < n; r++) {
        if (r > 0 && !q35m_barrier()) return 0;
        const int64_t p = pos + r;
        const size_t krow = (size_t)p * (size_t)s->kvw * sizeof(float);
        gbuf mix = row_at(g_s.mix, s->bMix, r);
        gbuf qb = row_at(g_s.qb, s->bD, r);
        gbuf kb = row_at(g_s.kb, s->bKvw, r);
        gbuf vb = row_at(g_s.vb, s->bKvw, r);
        gbuf att = row_at(g_s.att, s->bD, r);
        const uint32_t keys = (uint32_t)(p + 1);
        uint32_t chunk = 256u;
        uint32_t nchunks;
        if (keys < chunk) chunk = keys;
        nchunks = (keys + chunk - 1u) / chunk;
        if (nchunks > 64u) {
            nchunks = 64u;
            chunk = (keys + 63u) / 64u;
        }
        if ((size_t)nchunks * (size_t)s->H * (size_t)s->HD > g_s.part_cap)
            return 0;
        if (!q35m_qk(mix.b, mix.o, kb.b, kb.o, qb.b, qb.o,
                     W[W_KC].b, W[W_KC].o + krow,
                     g_fbuf, g_foff[l][F_QN] * sizeof(float),
                     g_fbuf, g_foff[l][F_KN] * sizeof(float),
                     g_rope.b, g_rope.o + (size_t)r * rope_stride,
                     g_rope.b,
                     g_rope.o + (size_t)r * rope_stride +
                         (size_t)(s->c->rotary_dim / 2) * sizeof(float),
                     (uint32_t)s->H, (uint32_t)s->KV, (uint32_t)s->HD,
                     (uint32_t)s->c->rotary_dim, s->c->rms_eps)) return 0;
        if (!q35m_kv_store(vb.b, vb.o, W[W_VC].b, W[W_VC].o + krow,
                           (uint32_t)s->kvw)) return 0;
        if (!q35m_scores(qb.b, qb.o, W[W_KC].b, W[W_KC].o,
                         g_s.scores.b, g_s.scores.o, (uint32_t)s->H,
                         (uint32_t)s->HD, (uint32_t)s->kvw, keys,
                         (uint32_t)s->group, scale)) return 0;
        if (!q35m_softmax(g_s.scores.b, g_s.scores.o, (uint32_t)s->H, keys))
            return 0;
        if (!q35m_av(g_s.scores.b, g_s.scores.o, W[W_VC].b, W[W_VC].o,
                     g_s.part.b, g_s.part.o, (uint32_t)s->H, (uint32_t)s->HD,
                     (uint32_t)s->kvw, keys, (uint32_t)s->group, chunk,
                     nchunks)) return 0;
        if (!q35m_av_combine(g_s.part.b, g_s.part.o, att.b, att.o,
                             nchunks, (uint32_t)s->H, (uint32_t)s->HD))
            return 0;
        if (s->c->attn_gate)
            if (!q35m_attn_gate(att.b, att.o, mix.b, mix.o,
                                (uint32_t)s->H, (uint32_t)s->HD)) return 0;
    }
    if (!proj_n(s, W[W_WO], g_s.att, s->bD, g_s.down, s->bD, n,
                (uint32_t)s->qdim, (uint32_t)s->D)) return 0;
    return 1;
}

static int step_mlp_rows(const sctx *s, int l, int n) {
    const gbuf *W = g_w[l];
    if (!proj_n(s, W[W_GATE], g_s.xn, s->bD, g_s.gate, s->bI, n,
                (uint32_t)s->D, (uint32_t)s->I)) return 0;
    if (!proj_n(s, W[W_UP], g_s.xn, s->bD, g_s.up, s->bI, n,
                (uint32_t)s->D, (uint32_t)s->I)) return 0;
    if (n > 1 && rowfuse()) {
        if (!q35m_silu_mul_rows(g_s.gate.b, g_s.gate.o, g_s.up.b, g_s.up.o,
                                (uint32_t)s->I, (uint32_t)n,
                                (uint32_t)(s->bI / sizeof(float)),
                                (uint32_t)(s->bI / sizeof(float)))) return 0;
    } else {
        for (int r = 0; r < n; r++) {
            gbuf g = row_at(g_s.gate, s->bI, r);
            gbuf u = row_at(g_s.up, s->bI, r);
            if (!q35m_silu_mul(g.b, g.o, u.b, u.o, (uint32_t)s->I)) return 0;
        }
    }
    if (!proj_n(s, W[W_DOWN], g_s.gate, s->bI, g_s.down, s->bD, n,
                (uint32_t)s->I, (uint32_t)s->D)) return 0;
    return 1;
}

int q35_gpu_supports_n(int n) {
    if (!g_ok || n < 1 || n > Q35_GK) return 0;
    if (n == 1) return 1;
    return q35m_gemv_n_prepare((uint32_t)n);
}

int q35_gpu_step_n(q35_model *m, const int32_t *ids, int64_t pos, int n,
                   float *logits) {
    if (!g_ok || !m || !ids || !logits) return 0;
    if (n < 1 || n > Q35_GK) return 0;
    sctx s;
    s.c = &m->cfg;
    s.D = s.c->dim;
    s.I = s.c->inter_dim;
    s.key_dim = s.c->lin_k_heads * s.c->lin_k_dim;
    s.val_dim = s.c->lin_v_heads * s.c->lin_v_dim;
    s.qkv_dim = 2 * s.key_dim + s.val_dim;
    s.H = s.c->n_heads;
    s.KV = s.c->n_kv_heads;
    s.HD = s.c->head_dim;
    s.qdim = s.H * s.HD;
    s.kvw = s.KV * s.HD;
    s.group = s.KV ? s.H / s.KV : 1;
    s.qwidth = s.qdim * (1 + s.c->attn_gate);
    s.taps = s.c->conv_kernel > 0 ? s.c->conv_kernel : 4;
    s.NH = s.c->lin_v_heads;
    s.bD = (size_t)s.D * sizeof(float);
    s.bI = (size_t)s.I * sizeof(float);
    s.bWide = (size_t)(s.I > s.val_dim ? s.I : s.val_dim) * sizeof(float);
    s.bMix = 2u * s.bWide;
    s.bKvw = (size_t)s.kvw * sizeof(float);
    s.bNH = (size_t)s.NH * sizeof(float);
    s.bVocab = (size_t)s.c->vocab_size * sizeof(float);
    s.slot_conv = (size_t)(s.taps - 1) * (size_t)s.qkv_dim * sizeof(float);
    s.slot_ssm = (size_t)s.val_dim * (size_t)s.c->lin_k_dim * sizeof(float);

    if (pos < 0 || pos + n > s.c->max_seq_len) return 0;
    if ((size_t)s.H * (size_t)(pos + n) > g_s.scores_cap) return 0;
    for (int r = 0; r < n; r++)
        if (ids[r] < 0 || ids[r] >= s.c->vocab_size) return 0;

    /* This is the one place a verify step touches the host between steps: n
     * rotation tables, written into the shared parameter buffer. They are
     * bit-identical to the CPU's because they *are* the CPU's tables. */
    {
        const int pairs = s.c->rotary_dim / 2;
        const size_t stride = 2u * g_rope_n * sizeof(float);
        for (int r = 0; r < n; r++) {
            float *dst = (float *)((char *)g_fhost + g_rope.o +
                                   (size_t)r * stride);
            memcpy(dst, m->rope_cos + (size_t)(pos + r) * pairs,
                   (size_t)pairs * sizeof(float));
            memcpy(dst + pairs, m->rope_sin + (size_t)(pos + r) * pairs,
                   (size_t)pairs * sizeof(float));
        }
    }

    if (!q35m_begin()) return 0;
    for (int r = 0; r < n; r++) {
        gbuf x = row_at(g_s.x, s.bD, r);
        if (!q35m_embed(g_embed.b, g_embed.o, x.b, x.o, (uint32_t)s.D,
                        (uint32_t)ids[r])) goto bad;
    }
    /* CWICK_Q35_GPU_MAXL caps the layer count, for bisecting a mismatch
     * between the n-row and one-row paths: both paths read it, so they stay
     * comparable. Unset in production. */
    int nlayers = s.c->n_layers;
    {
        const char *e = getenv("CWICK_Q35_GPU_MAXL");
        int v = e && *e ? atoi(e) : 0;
        if (v > 0 && v < nlayers) nlayers = v;
    }
    for (int l = 0; l < nlayers; l++) {
        const q35_layer *L = &m->layers[l];
        if (!norm_all(&s, g_s.x, g_fbuf, g_foff[l][F_IN] * sizeof(float),
                      g_s.xn, n, s.c->rms_eps, 1, s.bD, s.bD))
            goto bad;
        if (L->is_linear) {
            if (!step_linear_rows(&s, l, n)) goto bad;
        } else {
            if (!step_full_rows(&s, l, n, pos)) goto bad;
        }
        if (!add_all(&s, g_s.x, g_s.down, n, s.bD, s.bD)) goto bad;
        if (!norm_all(&s, g_s.x, g_fbuf, g_foff[l][F_POST] * sizeof(float),
                      g_s.xn, n, s.c->rms_eps, 1, s.bD, s.bD))
            goto bad;
        if (!step_mlp_rows(&s, l, n)) goto bad;
        if (!add_all(&s, g_s.x, g_s.down, n, s.bD, s.bD)) goto bad;
    }

    if (!norm_all(&s, g_s.x, g_final.b, g_final.o, g_s.xn, n, s.c->rms_eps, 1,
                  s.bD, s.bD))
        goto bad;
    if (!proj_n(&s, g_head, g_s.xn, s.bD, g_s.logits, s.bVocab, n,
                (uint32_t)s.D, (uint32_t)s.c->vocab_size)) goto bad;
    if (!q35m_flush()) goto bad;
    memcpy(logits, g_s.logits_host,
           (size_t)n * (size_t)s.c->vocab_size * sizeof(float));
    /* The recurrent state now sits in slot n, whichever prefix of the rows the
     * caller ends up keeping. q35_gpu_commit_rows moves the chosen one down to
     * slot 0; a single-row step never left it. */
    g_model = m;
    g_last_n = n;
    return 1;

bad:
    /* Part of the chain may already have run, and the DeltaNet state it
     * advanced cannot be rolled back without a snapshot. Report a failure
     * rather than an answer, and stop offering the GPU for this model. */
    q35m_abort();
    g_ok = 0;
    CWICK_INFO("q35gpu", "step at position %lld failed; the GPU forward is off "
               "for the rest of this process", (long long)pos);
    return -1;
}

int q35_gpu_step(q35_model *m, int32_t id, int64_t pos, float *logits) {
    return q35_gpu_step_n(m, &id, pos, 1, logits);
}

int q35_gpu_commit_rows(int j) {
    if (!g_model) return 0;
    if (g_last_n <= 1) return 1;   /* a single row never leaves slot 0 */
    if (j < 0 || j >= g_last_n) return 0;
    const q35_config *c = &g_model->cfg;
    const int taps = c->conv_kernel > 0 ? c->conv_kernel : 4;
    const size_t qkv = (size_t)(2 * c->lin_k_heads * c->lin_k_dim +
                                c->lin_v_heads * c->lin_v_dim);
    const size_t cb = (size_t)(taps - 1) * qkv * sizeof(float);
    const size_t sb = (size_t)c->lin_v_heads * c->lin_k_dim * c->lin_v_dim *
                      sizeof(float);
    const size_t from = (size_t)(j + 1);
    for (int l = 0; l < c->n_layers; l++) {
        if (!c->is_linear[l]) continue;
        memmove(g_model->conv_state[l], (uint8_t *)g_model->conv_state[l] +
                from * cb, cb);
        memmove(g_model->ssm_state[l], (uint8_t *)g_model->ssm_state[l] +
                from * sb, sb);
    }
    g_last_n = 1;
    return 1;
}

/* ------------------------------------------------------------------ */
/* the MTP drafter                                                     */
/* ------------------------------------------------------------------ */
//
// One block that turns "the token this step is about to verify" into a guess
// at the one after it. The case for it is arithmetic rather than taste: a
// draft costs one pass over 486.6 MB of block weights, while an extra *verify*
// row costs what the arithmetic of an extra activation row costs -- so a
// drafter that is right often enough pays for itself, and one that is not is
// simply discarded by the verify.
//
// Two things make it cheap enough to be worth trying.
//
// First, the chain never leaves the device. Proposal i is the argmax of
// proposal i-1's scores, read by proposal i's embedding lookup out of a buffer
// the argmax wrote; a host round trip per proposal would be ~0.93 ms against a
// ~1.5 ms block, i.e. most of the cost. So k proposals are one command buffer
// and one wait, and only the k token ids come back.
//
// Second, the output projection is restricted to a candidate list. The shared
// vocabulary head is 2.03 GiB against the block's 0.49 GiB, so projecting a
// draft through all of it would make the head, not the drafter, the cost --
// 7.0 ms per token against 1.6. A draft is allowed to be wrong (the verify
// decides what is emitted), so narrowing it is a bandwidth decision and not an
// accuracy one: what it costs is acceptance rate, and that is measurable.
//
// The block's own KV cache is not back-filled for the prompt. Filling it would
// mean running this layer over every prefilled position -- an extra 1/32 of
// prefill, plus the hidden states to feed it -- and the fusion is a
// concatenation precisely because h_t already carries the context. Rows are
// written from the positions this decoder actually visits, so `mtp_n` (the
// number of such rows) bounds the score loop rather than the position does.

/* Nothing here can be rolled back if it fails half way except the cache, which
 * the caller rewinds anyway, so a failure only costs this step's drafts. */
/* Structural switches, so the things the checkpoint cannot tell us are
 * measurable instead of assumed. All three were swept; see DECISIONS. */
static int mtp_swap(void) {
    static int cached = -1;
    if (cached < 0) cached = cwick_env_true("CWICK_Q35_MTP_SWAP") ? 1 : 0;
    return cached;
}
/* Close the attention half of the block, leaving fc + SwiGLU + head. The cache
 * holds only the rows this decoder drafted, so at the first proposal the
 * softmax has exactly one key -- itself -- which is not a shape the layer was
 * trained on. This measures what that costs against what the context-free half
 * already gets. */
static int mtp_noattn(void) {
    static int cached = -1;
    if (cached < 0) cached = cwick_env_true("CWICK_Q35_MTP_NOATTN") ? 1 : 0;
    return cached;
}

/* Everything one position of the block needs, resolved once per call. */
typedef struct {
    const q35_config *c;
    int D, I, H, KV, HD, qwidth, qdim, group;
    size_t bD, bKv;
    size_t eoff, hoff;
    float eps, scale;
} mtpd;

/* One position: fuse, attend, SwiGLU, leave the result in g_m.res.
 *
 * `hsrc` is the row that supplies h_t -- the text model's residual when the
 * caller has it, this block's own output when it is chaining. `keys` counts the
 * rows of the block's cache that exist by this position, including the one this
 * call writes. */
static int mtp_one(const mtpd *d, gbuf hsrc, gbuf ids, size_t idoff,
                   int64_t row, uint32_t keys, uint32_t rope_i) {
    const int D = d->D, I = d->I, H = d->H, KV = d->KV, HD = d->HD;
    const size_t bKv = d->bKv;

    if (!q35m_embed_id(g_embed.b, g_embed.o, g_m.e.b, g_m.e.o + d->eoff,
                       ids.b, ids.o + idoff, (uint32_t)D)) return 0;
    if (!q35m_rmsnorm(g_m.e.b, g_m.e.o + d->eoff, g_mtpbuf,
                      g_mtpoff[MF_EMB] * sizeof(float),
                      g_m.e.b, g_m.e.o + d->eoff, (uint32_t)D, d->eps, 1))
        return 0;
    /* The hidden half of the fusion, normalised with the block's own weight.
     * Whether the chained (block-owned) hidden should be normalised as well is
     * not in the checkpoint; swept, it did not matter, so there is no switch. */
    if (!q35m_rmsnorm(hsrc.b, hsrc.o, g_mtpbuf,
                      g_mtpoff[MF_HID] * sizeof(float),
                      g_m.e.b, g_m.e.o + d->hoff, (uint32_t)D, d->eps, 1))
        return 0;
    if (!q35m_gemv(g_mtp[M_FC].b, g_mtp[M_FC].o, (uint32_t)(2 * D),
                   (uint32_t)D, g_m.e.b, g_m.e.o, g_m.res.b, g_m.res.o))
        return 0;

    if (!mtp_noattn()) {
        const size_t rope_stride = 2u * g_rope_n * sizeof(float);
        if (!q35m_rmsnorm(g_m.res.b, g_m.res.o, g_mtpbuf,
                          g_mtpoff[MF_IN] * sizeof(float),
                          g_m.xn.b, g_m.xn.o, (uint32_t)D, d->eps, 1))
            return 0;
        if (!q35m_gemv(g_mtp[M_WQ].b, g_mtp[M_WQ].o, (uint32_t)D,
                       (uint32_t)d->qwidth, g_m.xn.b, g_m.xn.o,
                       g_m.mix.b, g_m.mix.o)) return 0;
        if (!q35m_gemv(g_mtp[M_WK].b, g_mtp[M_WK].o, (uint32_t)D,
                       (uint32_t)(KV * HD), g_m.xn.b, g_m.xn.o,
                       g_m.kv.b, g_m.kv.o)) return 0;
        if (!q35m_gemv(g_mtp[M_WV].b, g_mtp[M_WV].o, (uint32_t)D,
                       (uint32_t)(KV * HD), g_m.xn.b, g_m.xn.o,
                       g_m.kv.b, g_m.kv.o + bKv)) return 0;
        if (!q35m_qk(g_m.mix.b, g_m.mix.o, g_m.kv.b, g_m.kv.o,
                     g_m.qb.b, g_m.qb.o, g_mtpkc.b, g_mtpkc.o + row * bKv,
                     g_mtpbuf, g_mtpoff[MF_QN] * sizeof(float),
                     g_mtpbuf, g_mtpoff[MF_KN] * sizeof(float),
                     g_rope.b, g_rope.o + (size_t)rope_i * rope_stride,
                     g_rope.b,
                     g_rope.o + (size_t)rope_i * rope_stride +
                         (size_t)(d->c->rotary_dim / 2) * sizeof(float),
                     (uint32_t)H, (uint32_t)KV, (uint32_t)HD,
                     (uint32_t)d->c->rotary_dim, d->eps)) return 0;
        if (!q35m_kv_store(g_m.kv.b, g_m.kv.o + bKv, g_mtpvc.b,
                           g_mtpvc.o + row * bKv, (uint32_t)(KV * HD)))
            return 0;
        {
            uint32_t chunk = 256u, nchunks;
            if (keys < chunk) chunk = keys;
            nchunks = (keys + chunk - 1u) / chunk;
            if (nchunks > 64u) { nchunks = 64u; chunk = (keys + 63u) / 64u; }
            if ((size_t)nchunks * (size_t)H * (size_t)HD > g_s.part_cap)
                return 0;
            if (!q35m_scores(g_m.qb.b, g_m.qb.o, g_mtpkc.b, g_mtpkc.o,
                             g_s.scores.b, g_s.scores.o, (uint32_t)H,
                             (uint32_t)HD, (uint32_t)(KV * HD), keys,
                             (uint32_t)d->group, d->scale)) return 0;
            if (!q35m_softmax(g_s.scores.b, g_s.scores.o, (uint32_t)H, keys))
                return 0;
            if (!q35m_av(g_s.scores.b, g_s.scores.o, g_mtpvc.b, g_mtpvc.o,
                         g_s.part.b, g_s.part.o, (uint32_t)H, (uint32_t)HD,
                         (uint32_t)(KV * HD), keys, (uint32_t)d->group, chunk,
                         nchunks)) return 0;
            if (!q35m_av_combine(g_s.part.b, g_s.part.o, g_m.att.b, g_m.att.o,
                                 nchunks, (uint32_t)H, (uint32_t)HD)) return 0;
        }
        if (d->c->attn_gate &&
            !q35m_attn_gate(g_m.att.b, g_m.att.o, g_m.mix.b, g_m.mix.o,
                            (uint32_t)H, (uint32_t)HD)) return 0;
        if (!q35m_gemv(g_mtp[M_WO].b, g_mtp[M_WO].o, (uint32_t)d->qdim,
                       (uint32_t)D, g_m.att.b, g_m.att.o, g_m.down.b,
                       g_m.down.o)) return 0;
        if (!q35m_add(g_m.res.b, g_m.res.o, g_m.down.b, g_m.down.o,
                      (uint32_t)D)) return 0;
    }

    if (!q35m_rmsnorm(g_m.res.b, g_m.res.o, g_mtpbuf,
                      g_mtpoff[MF_POST] * sizeof(float),
                      g_m.xn.b, g_m.xn.o, (uint32_t)D, d->eps, 1)) return 0;
    if (!q35m_gemv(g_mtp[M_GATE].b, g_mtp[M_GATE].o, (uint32_t)D,
                   (uint32_t)I, g_m.xn.b, g_m.xn.o,
                   g_m.gate.b, g_m.gate.o)) return 0;
    if (!q35m_gemv(g_mtp[M_UP].b, g_mtp[M_UP].o, (uint32_t)D, (uint32_t)I,
                   g_m.xn.b, g_m.xn.o, g_m.up.b, g_m.up.o)) return 0;
    if (!q35m_silu_mul(g_m.gate.b, g_m.gate.o, g_m.up.b, g_m.up.o,
                       (uint32_t)I)) return 0;
    if (!q35m_gemv(g_mtp[M_DOWN].b, g_mtp[M_DOWN].o, (uint32_t)I,
                   (uint32_t)D, g_m.gate.b, g_m.gate.o,
                   g_m.down.b, g_m.down.o)) return 0;
    if (!q35m_add(g_m.res.b, g_m.res.o, g_m.down.b, g_m.down.o,
                  (uint32_t)D)) return 0;
    return 1;
}

/* Rows keep their absolute positions when the window fills; only the index they
 * sit at changes, and the rotation was baked in at write time, so shifting the
 * live rows down is exact rather than an approximation. */
static void mtp_compact(q35_model *m, int64_t need_pos) {
    if (!m || m->mtp_win <= 0) return;
    int64_t over = need_pos - m->mtp_lo - (int64_t)(m->mtp_win - 1);
    if (over <= 0) return;
    const size_t kvw = (size_t)m->cfg.n_kv_heads * (size_t)m->cfg.head_dim;
    const size_t rowb = kvw * sizeof(float);
    int64_t live = m->mtp_n - m->mtp_lo;
    if (over > live) over = live;
    const int64_t keep = live - over;
    if (keep > 0) {
        memmove(m->mtp_kcache, m->mtp_kcache + (size_t)over * kvw,
                (size_t)keep * rowb);
        memmove(m->mtp_vcache, m->mtp_vcache + (size_t)over * kvw,
                (size_t)keep * rowb);
    }
    m->mtp_lo += over;
    if (m->mtp_n < m->mtp_lo) m->mtp_n = m->mtp_lo;
}

static void mtpd_init(mtpd *d, const q35_model *m) {
    const q35_config *c = &m->cfg;
    memset(d, 0, sizeof(*d));
    d->c = c;
    d->D = c->dim;
    d->I = c->inter_dim;
    d->H = c->n_heads;
    d->KV = c->n_kv_heads;
    d->HD = c->head_dim;
    d->qwidth = d->H * d->HD * (1 + c->attn_gate);
    d->qdim = d->H * d->HD;
    d->group = d->KV ? d->H / d->KV : 1;
    d->bD = (size_t)d->D * sizeof(float);
    d->bKv = (size_t)d->KV * (size_t)d->HD * sizeof(float);
    d->eps = c->rms_eps;
    d->scale = 1.0f / sqrtf((float)d->HD);
    /* Which half of the fusion comes first is not in the file either. */
    d->eoff = mtp_swap() ? d->bD : 0;
    d->hoff = d->bD - d->eoff;
}

int q35_mtp_sync(q35_model *m, int hrow, int hpost, int64_t pos,
                 const int32_t *nxt) {
    if (!g_ok || !g_mtp_ok || !m || !m->mtp_ok || !nxt) return 0;
    if (hrow < 0 || hrow >= Q35_GK) return 0;
    if (pos - hrow - 1 < 0) return 0;
    mtpd d;
    mtpd_init(&d, m);
    int64_t lo = pos - hrow - 1;
    /* A gap means the older rows no longer line up with these positions
     * (a fresh sequence, or a step that ran on the CPU), so the window starts
     * here instead of pretending the gap is context. */
    if (lo > m->mtp_n) m->mtp_lo = lo;
    m->mtp_n = lo;
    mtp_compact(m, pos - 1);
    if (pos - 1 - m->mtp_lo >= m->mtp_win) return 0;

    {   /* the rotation tables for these positions, in the shared slots */
        const int pairs = d.c->rotary_dim / 2;
        const size_t stride = 2u * g_rope_n * sizeof(float);
        for (int i = 0; i <= hrow; i++) {
            float *rop = (float *)((char *)g_fhost + g_rope.o +
                                   (size_t)i * stride);
            memcpy(rop, m->rope_cos + (size_t)(lo + i) * pairs,
                   (size_t)pairs * sizeof(float));
            memcpy(rop + pairs, m->rope_sin + (size_t)(lo + i) * pairs,
                   (size_t)pairs * sizeof(float));
        }
    }
    for (int i = 0; i <= hrow; i++)
        ((uint32_t *)g_m.ids_host)[i] = (uint32_t)nxt[i];

    gbuf xn = g_s.xn, xr = g_s.x;
    if (!q35m_begin()) return 0;
    for (int i = 0; i <= hrow; i++) {
        const int64_t idx = (lo + i) - m->mtp_lo;
        gbuf h = row_at(hpost ? xn : xr, d.bD, i);
        if (!mtp_one(&d, h, g_m.ids, (size_t)i * sizeof(uint32_t), idx,
                     (uint32_t)(idx + 1), (uint32_t)i))
            goto bad;
    }
    if (!q35m_flush()) goto bad;
    m->mtp_n = pos;
    m->mtp_base = pos;
    return 1;
bad:
    q35m_abort();
    m->mtp_n = m->mtp_lo;
    CWICK_INFO("q35gpu", "MTP sync failed at position %lld; the window starts "
               "over", (long long)pos);
    return 0;
}

/* Nothing here can be rolled back if it fails half way except the cache, which
 * the caller rewinds anyway, so a failure only costs this step's drafts. */
int q35_mtp_draft(q35_model *m, int hrow, int hpost, int32_t first, int64_t pos,
                  int k, const uint32_t *cand, int ncand, int32_t *out) {
    if (!g_ok || !g_mtp_ok || !m || !m->mtp_ok) return 0;
    if (!m->mtp_kcache || !m->mtp_vcache || !out) return 0;
    if (k < 1 || k > Q35_GK - 1) return 0;
    /* A NULL candidate list means "score every row": the identity list is
     * already sitting in the device buffer, so the A/B arm against the
     * restricted projection costs no upload. */
    const int every_row = (ncand <= 0 || !cand);
    if (every_row) ncand = g_m.cand_cap;
    mtpd d;
    mtpd_init(&d, m);
    const int D = d.D;
    if (ncand <= 0 || ncand > g_m.cand_cap) return 0;
    if (hrow < 0 || hrow >= Q35_GK) return 0;
    if (pos < 0 || pos + k > d.c->max_seq_len) return 0;
    if (m->mtp_n < m->mtp_lo) return 0;
    if (m->mtp_n + k - m->mtp_lo > (int64_t)g_s.scores_cap / (int64_t)d.H)
        return 0;
    mtp_compact(m, m->mtp_n + k - 1);
    if (m->mtp_n + k - m->mtp_lo > m->mtp_win) return 0;

    /* The rotation tables for the positions this chain sits at, in the same
     * shared slots the verify pass uses -- it has already consumed them. */
    {
        const int pairs = d.c->rotary_dim / 2;
        const size_t stride = 2u * g_rope_n * sizeof(float);
        for (int i = 0; i < k; i++) {
            float *rop = (float *)((char *)g_fhost + g_rope.o +
                                   (size_t)i * stride);
            memcpy(rop, m->rope_cos + (size_t)(pos + i) * pairs,
                   (size_t)pairs * sizeof(float));
            memcpy(rop + pairs, m->rope_sin + (size_t)(pos + i) * pairs,
                   (size_t)pairs * sizeof(float));
        }
    }
    if (!every_row)
        memcpy(g_m.cand_host, cand, (size_t)ncand * sizeof(uint32_t));
    ((uint32_t *)g_m.ids_host)[0] = (uint32_t)first;

    /* h_t is the hidden state of the last accepted position. Which of the two
     * the block was trained on is not in the checkpoint: the text model keeps
     * both, the residual stream (pre final norm) and the normed copy that the
     * vocabulary head consumes. */
    gbuf xrow = row_at(hpost ? g_s.xn : g_s.x, d.bD, hrow);

    if (!q35m_begin()) return 0;
    /* The write cursor must not move while the chain runs: `row` is derived
     * from it, and incrementing it inside the loop would make proposal i land
     * at base + i + (rows already written) -- every row past the first would
     * be written off the end of what the next proposal reads. */
    const int64_t base_row = m->mtp_n;
    for (int i = 0; i < k; i++) {
        const int64_t idx = base_row + i - m->mtp_lo;
        gbuf h = (i == 0) ? xrow : g_m.res;
        if (i > 0 && !q35m_barrier()) goto bad;
        if (!mtp_one(&d, h, g_m.ids, (size_t)i * sizeof(uint32_t), idx,
                     (uint32_t)(idx + 1), (uint32_t)i))
            goto bad;
        /* --- the shared vocabulary head, restricted --- */
        if (!q35m_rmsnorm(g_m.res.b, g_m.res.o, g_mtpbuf,
                          g_mtpoff[MF_OUT] * sizeof(float),
                          g_m.xn.b, g_m.xn.o, (uint32_t)D, d.eps, 1)) goto bad;
        if (!q35m_gemv_rows(g_head.b, g_head.o, (uint32_t)D, (uint32_t)ncand,
                            g_m.cand.b, g_m.cand.o, g_m.xn.b, g_m.xn.o,
                            g_m.sc.b, g_m.sc.o)) goto bad;
        if (!q35m_argmax(g_m.sc.b, g_m.sc.o, g_m.cand.b, g_m.cand.o,
                         g_m.ids.b,
                         g_m.ids.o + (size_t)(i + 1) * sizeof(uint32_t),
                         (uint32_t)ncand)) goto bad;
    }
    m->mtp_n = base_row + k;
    if (!q35m_flush()) goto bad;
    for (int i = 0; i < k; i++) out[i] = (int32_t)((uint32_t *)g_m.ids_host)[i + 1];
    return k;

bad:
    q35m_abort();
    m->mtp_n = m->mtp_base;
    CWICK_INFO("q35gpu", "MTP draft failed at position %lld; this step verifies "
               "one row", (long long)pos);
    return 0;
}
