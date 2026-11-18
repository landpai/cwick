/* rt_flow.c — see rt_flow.h.
 *
 * The estimator is a 22-layer DiT over the mel sequence. Two details of the
 * exported graph are easy to get wrong and are reproduced exactly here:
 *   - rotary embedding touches only the first 64 of the 1024 q/k channels,
 *     i.e. head 0; the other 15 heads are position-blind by construction.
 *   - the input projection concatenates [x, cond, mu, spks], in that order.
 * Both were checked against the shipped ONNX graph, not just the source. */
#include "rt_flow.h"

#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "log.h"
#include "rt_dsp.h"
#include "rt_torch.h"
#include "util.h"

#define FL_TAG "flow"
#define FL_DIM 1024
#define FL_DEPTH 22
#define FL_HEADS 16
#define FL_HD 64
#define FL_FF 2048
#define FL_MEL 80
#define FL_CHUNK 50 /* mel frames per streaming chunk (25 tokens x 2) */
#define FL_LOOKAHEAD 3
#define FL_POS_CTX 60 /* left reach of the two 31-tap position convolutions */

typedef struct {
    rt_lin norm; /* 1024 -> 6144 */
    rt_lin q, k, v, o;
    rt_lin ff_in, ff_out;
} fl_block;

struct rt_flow {
    rt_pt *pt;
    float *emb; /* [6561][80] */
    rt_lin spk; /* 192 -> 80 */
    rt_conv pre1, pre2;
    rt_lin in_proj; /* 320 -> 1024 */
    rt_conv pos1, pos2;
    rt_lin t0, t2;
    fl_block b[FL_DEPTH];
    rt_lin norm_out; /* 1024 -> 2048 */
    rt_lin proj_out; /* 1024 -> 80 */
    float inv_freq[FL_HD / 2];
    int steps;
    float cfg;
    int linear; /* 0: cosine schedule, 1: linear */
    /* scratch, grown on demand */
    int cap;
    float *x, *h, *q, *k, *v, *att, *nrm, *ff, *score, *res;
    float *rope; /* [cap][64] cos then sin interleaved per pair */
    int rope_fill;
    /* Streaming prefix reuse. The mask is chunk-causal, the position
     * convolutions are left-padded and the noise stream is sequential, so a
     * frame's activations never change once its own 50-frame chunk is
     * complete. Keeping K/V per euler step per layer therefore lets a chunk
     * compute only its new frames instead of re-rendering the whole prefix,
     * which is what made the cost quadratic in the utterance length. */
    /* One block per (euler step, layer), each [kvcap][2][FL_DIM]. Splitting
     * them keeps a frame's K/V contiguous inside its block, so growing the
     * cache is a plain realloc that preserves the prefix -- a single flat
     * arena would restride every block and scramble what was already there. */
    float **kv;
    /* The latent as it stood at the input of each euler step, for committed
     * frames. The position convolutions reach 60 frames back, so rendering a
     * chunk needs the prefix's real x -- fresh noise there would corrupt the
     * first 60 new rows. 80 floats per frame per step, so this is cheap. */
    float **kx;
    int kvcap, kvfill;
    int kv_on;
    /* Content fingerprint of the conditioning prompt. Pointer identity is not
     * enough: a voice can be freed and another allocated at the same address. */
    uint64_t kv_key;
};

static int fl_lin(rt_lin *l, const rt_pt *pt, const char *name, int bias) {
    if (rt_lin_load(l, pt, name, bias) != 0) {
        CWICK_INFO(FL_TAG, "missing linear %s", name);
        return -1;
    }
    return 0;
}

rt_flow *rt_flow_open(const char *path, cwick_error *err) {
    rt_pt *pt = NULL;
    if (rt_pt_open(&pt, path) != 0) {
        cwick_error_set(err, CWICK_ERR_IO, "cannot open flow checkpoint: %s", path);
        return NULL;
    }
    rt_flow *f = (rt_flow *)cwick_xmalloc(sizeof(*f));
    memset(f, 0, sizeof(*f));
    f->pt = pt;
    f->steps = 4;
    f->cfg = 0.0f;
    const char *e = getenv("RT_FLOW_STEPS");
    if (e && atoi(e) > 0) f->steps = atoi(e);
    e = getenv("RT_FLOW_CFG");
    if (e) f->cfg = (float)atof(e);
    e = getenv("RT_FLOW_SCHED");
    if (e && strcmp(e, "linear") == 0) f->linear = 1;
    f->kv = (float **)cwick_xmalloc((size_t)f->steps * FL_DEPTH * sizeof(float *));
    memset(f->kv, 0, (size_t)f->steps * FL_DEPTH * sizeof(float *));
    f->kx = (float **)cwick_xmalloc((size_t)f->steps * sizeof(float *));
    memset(f->kx, 0, (size_t)f->steps * sizeof(float *));
    f->kv_on = 1;
    e = getenv("RT_FLOW_KV");
    if (e && atoi(e) == 0) f->kv_on = 0;

    const rt_tensor *emb = rt_pt_get(pt, "input_embedding.weight");
    const rt_tensor *ivf = rt_pt_get(pt, "decoder.estimator.rotary_embed.inv_freq");
    if (!emb || emb->ndim != 2 || emb->shape[1] != FL_MEL || !ivf ||
        ivf->numel != FL_HD / 2) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "flow: unexpected embedding/rope");
        rt_flow_close(f);
        return NULL;
    }
    int vocab = (int)emb->shape[0];
    f->emb = (float *)cwick_xmalloc((size_t)vocab * FL_MEL * sizeof(float));
    for (int64_t i = 0; i < emb->numel; i++) f->emb[i] = rt_tensor_f32(emb, i);
    for (int i = 0; i < FL_HD / 2; i++) f->inv_freq[i] = rt_tensor_f32(ivf, i);

    int bad = 0;
    bad |= fl_lin(&f->spk, pt, "spk_embed_affine_layer", 1);
    bad |= rt_conv_load(&f->pre1, pt, "pre_lookahead_layer.conv1", 1, 1, 1, 0, 3);
    bad |= rt_conv_load(&f->pre2, pt, "pre_lookahead_layer.conv2", 1, 1, 1, 2, 0);
    bad |= fl_lin(&f->in_proj, pt, "decoder.estimator.input_embed.proj", 1);
    bad |= rt_conv_load(&f->pos1, pt, "decoder.estimator.input_embed.conv_pos_embed.conv1.0", 16, 1, 1, 30, 0);
    bad |= rt_conv_load(&f->pos2, pt, "decoder.estimator.input_embed.conv_pos_embed.conv2.0", 16, 1, 1, 30, 0);
    bad |= fl_lin(&f->t0, pt, "decoder.estimator.time_embed.time_mlp.0", 1);
    bad |= fl_lin(&f->t2, pt, "decoder.estimator.time_embed.time_mlp.2", 1);
    bad |= fl_lin(&f->norm_out, pt, "decoder.estimator.norm_out.linear", 1);
    bad |= fl_lin(&f->proj_out, pt, "decoder.estimator.proj_out", 1);
    for (int i = 0; i < FL_DEPTH && !bad; i++) {
        char nm[192];
        fl_block *b = &f->b[i];
        snprintf(nm, sizeof(nm), "decoder.estimator.transformer_blocks.%d", i);
        char full[256];
        snprintf(full, sizeof(full), "%s.attn_norm.linear", nm);
        bad |= fl_lin(&b->norm, pt, full, 1);
        snprintf(full, sizeof(full), "%s.attn.to_q", nm);
        bad |= fl_lin(&b->q, pt, full, 1);
        snprintf(full, sizeof(full), "%s.attn.to_k", nm);
        bad |= fl_lin(&b->k, pt, full, 1);
        snprintf(full, sizeof(full), "%s.attn.to_v", nm);
        bad |= fl_lin(&b->v, pt, full, 1);
        snprintf(full, sizeof(full), "%s.attn.to_out.0", nm);
        bad |= fl_lin(&b->o, pt, full, 1);
        snprintf(full, sizeof(full), "%s.ff.ff.0.0", nm);
        bad |= fl_lin(&b->ff_in, pt, full, 1);
        snprintf(full, sizeof(full), "%s.ff.ff.2", nm);
        bad |= fl_lin(&b->ff_out, pt, full, 1);
    }
    if (bad) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "flow: incomplete estimator");
        rt_flow_close(f);
        return NULL;
    }
    CWICK_INFO(FL_TAG, "flow ready: DiT %dx%d, %d heads, %d euler steps, cfg %.2f",
               FL_DEPTH, FL_DIM, FL_HEADS, f->steps, (double)f->cfg);
    return f;
}

void rt_flow_close(rt_flow *f) {
    if (!f) return;
    free(f->emb);
    rt_lin_free(&f->spk);
    rt_conv_free(&f->pre1); rt_conv_free(&f->pre2);
    rt_lin_free(&f->in_proj);
    rt_conv_free(&f->pos1); rt_conv_free(&f->pos2);
    rt_lin_free(&f->t0); rt_lin_free(&f->t2);
    rt_lin_free(&f->norm_out); rt_lin_free(&f->proj_out);
    for (int i = 0; i < FL_DEPTH; i++) {
        fl_block *b = &f->b[i];
        rt_lin_free(&b->norm); rt_lin_free(&b->q); rt_lin_free(&b->k);
        rt_lin_free(&b->v); rt_lin_free(&b->o);
        rt_lin_free(&b->ff_in); rt_lin_free(&b->ff_out);
    }
    free(f->x); free(f->h); free(f->q); free(f->k); free(f->v);
    free(f->att); free(f->nrm); free(f->ff); free(f->score); free(f->rope);
    free(f->res);
    if (f->kv) {
        for (int i = 0; i < f->steps * FL_DEPTH; i++) free(f->kv[i]);
        free(f->kv);
    }
    if (f->kx) {
        for (int i = 0; i < f->steps; i++) free(f->kx[i]);
        free(f->kx);
    }
    rt_pt_close(f->pt);
    free(f);
}

static void fl_reserve(rt_flow *f, int T) {
    if (T <= f->cap) return;
    int cap = f->cap ? f->cap : 256;
    while (cap < T) cap *= 2;
    free(f->x); free(f->h); free(f->q); free(f->k); free(f->v);
    free(f->att); free(f->nrm); free(f->ff); free(f->score); free(f->rope);
    free(f->res);
    size_t d = (size_t)cap * FL_DIM;
    f->x = (float *)cwick_xmalloc(d * sizeof(float));
    f->h = (float *)cwick_xmalloc(d * sizeof(float));
    f->q = (float *)cwick_xmalloc(d * sizeof(float));
    f->k = (float *)cwick_xmalloc(d * sizeof(float));
    f->v = (float *)cwick_xmalloc(d * sizeof(float));
    f->att = (float *)cwick_xmalloc(d * sizeof(float));
    f->nrm = (float *)cwick_xmalloc(d * sizeof(float));
    f->ff = (float *)cwick_xmalloc((size_t)cap * FL_FF * sizeof(float));
    f->score = (float *)cwick_xmalloc((size_t)cap * cap * sizeof(float));
    f->rope = (float *)cwick_xmalloc((size_t)cap * FL_HD * sizeof(float));
    f->res = (float *)cwick_xmalloc(d * sizeof(float));
    f->cap = cap;
    f->rope_fill = 0;
}

/* K/V cache for `frames` mel frames, or 0 if that much is not affordable.
 * One frame costs steps * depth * 2 * 1024 floats, so the budget is stated in
 * frames and the caller falls back to a full re-render past it. */
#define FL_KV_MAX 2048
static int fl_kv_reserve(rt_flow *f, int frames) {
    if (frames <= f->kvcap) return 1;
    if (frames > FL_KV_MAX) return 0;
    int cap = f->kvcap ? f->kvcap : 256;
    while (cap < frames) cap *= 2;
    if (cap > FL_KV_MAX) cap = FL_KV_MAX;
    int nb = f->steps * FL_DEPTH;
    size_t n = (size_t)cap * 2 * FL_DIM;
    for (int i = 0; i < nb; i++) {
        float *p = (float *)realloc(f->kv[i], n * sizeof(float));
        if (!p) {
            /* Blocks now disagree about their stride, so nothing cached can
             * be addressed any more; drop the prefix rather than trust it. */
            f->kvcap = 0; f->kvfill = 0;
            return 0;
        }
        f->kv[i] = p;
    }
    for (int i = 0; i < f->steps; i++) {
        float *p = (float *)realloc(f->kx[i], (size_t)cap * FL_MEL * sizeof(float));
        if (!p) { f->kvcap = 0; f->kvfill = 0; return 0; }
        f->kx[i] = p;
    }
    f->kvcap = cap;
    return 1;
}

static float *fl_kv_at(rt_flow *f, int step, int layer, int frame) {
    return f->kv[(size_t)step * FL_DEPTH + layer] + (size_t)frame * 2 * FL_DIM;
}

/* FNV-1a over everything the prompt contributes to the estimator. Hashing the
 * mel as well as the tokens costs one pass over ~120 KB, microseconds against
 * the render it can save. */
static uint64_t fl_prompt_key(const int *token, int pt, const float *feat, int pf,
                              const float *spk) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < pt; i++) {
        h = (h ^ (uint32_t)token[i]) * 1099511628211ULL;
    }
    const uint32_t *fb = (const uint32_t *)feat;
    for (size_t i = 0; i < (size_t)pf * FL_MEL; i++) h = (h ^ fb[i]) * 1099511628211ULL;
    const uint32_t *sb = (const uint32_t *)spk;
    for (int i = 0; i < 192; i++) h = (h ^ sb[i]) * 1099511628211ULL;
    h = (h ^ (uint32_t)pt) * 1099511628211ULL;
    return (h ^ (uint32_t)pf) * 1099511628211ULL;
}

/* cos/sin for the interleaved rotary pairs, duplicated so channel 2i and
 * 2i+1 share an angle. Only the rows past rope_fill are new: the table is a
 * pure function of the position, so re-deriving the prefix every chunk was
 * tens of thousands of needless trig calls per estimator evaluation. */
static void fl_rope(rt_flow *f, int T) {
    for (int t = f->rope_fill; t < T; t++)
        for (int i = 0; i < FL_HD / 2; i++) {
            float a = (float)t * f->inv_freq[i];
            float c = cosf(a), s = sinf(a);
            f->rope[(size_t)t * FL_HD + 2 * i] = c;
            f->rope[(size_t)t * FL_HD + 2 * i + 1] = s;
        }
    if (T > f->rope_fill) f->rope_fill = T;
}

static void fl_rotate(const float *rope, float *x, int T) {
    for (int t = 0; t < T; t++) {
        float *row = x + (size_t)t * FL_DIM;
        const float *rs = rope + (size_t)t * FL_HD;
        for (int i = 0; i < FL_HD / 2; i++) {
            float a = row[2 * i], b = row[2 * i + 1];
            float c = rs[2 * i], s = rs[2 * i + 1];
            row[2 * i] = a * c - b * s;
            row[2 * i + 1] = b * c + a * s;
        }
    }
}

/* Attention over the new rows [row0, T) against K/V for the whole sequence,
 * laid out with `stride` floats per frame (1024 for the direct buffers, 2048
 * for the interleaved cache). Row i attends [0, end_i): everything when not
 * streaming, else up to the end of i's 50-frame chunk. Every row of a chunk
 * shares that bound, so the two products run block by block over just the
 * columns that survive the mask -- half the work of a full T*T product. */
static void fl_attention_kv(rt_flow *f, int T, int streaming, int row0,
                            const float *kc, const float *vc, int stride) {
    const float scale = 1.0f / sqrtf((float)FL_HD);
    for (int h = 0; h < FL_HEADS; h++) {
        const float *q = f->q + h * FL_HD;
        float *o = f->att + h * FL_HD;
        /* Blocks follow the mask's absolute 50-frame boundaries, not row0:
         * a voice prompt's frame count need not be a multiple of 50, so the
         * first new row can sit inside a chunk that started in the prompt. */
        for (int r0 = row0; r0 < T; ) {
            int end = streaming ? (r0 / FL_CHUNK + 1) * FL_CHUNK : T;
            if (end > T) end = T;
            int n = end - r0;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, end, FL_HD,
                        scale, q + (size_t)(r0 - row0) * FL_DIM, FL_DIM,
                        kc + h * FL_HD, stride, 0.0f, f->score, T);
            for (int i = 0; i < n; i++) {
                float *s = f->score + (size_t)i * T;
                /* vForce's exp is SIMD and near-exact; the scalar expf this
                 * replaces was a large share of the estimator's wall clock. */
                float mx = 0.0f, sum = 0.0f;
                vDSP_maxv(s, 1, &mx, (vDSP_Length)end);
                float neg = -mx;
                vDSP_vsadd(s, 1, &neg, s, 1, (vDSP_Length)end);
                int nn = end;
                vvexpf(s, s, &nn);
                vDSP_sve(s, 1, &sum, (vDSP_Length)end);
                float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
                vDSP_vsmul(s, 1, &inv, s, 1, (vDSP_Length)end);
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, FL_HD, end,
                        1.0f, f->score, T, vc + h * FL_HD, stride,
                        0.0f, o + (size_t)(r0 - row0) * FL_DIM, FL_DIM);
            r0 = end;
        }
    }
}

static void fl_timestep(rt_flow *f, float t, float *temb) {
    float sin[256];
    float half = 128.0f;
    float step = logf(10000.0f) / (half - 1.0f);
    for (int i = 0; i < 128; i++) {
        float a = 1000.0f * t * expf(-(float)i * step);
        sin[i] = sinf(a);
        sin[128 + i] = cosf(a);
    }
    float mid[FL_DIM];
    rt_lin_apply(&f->t0, sin, mid, 1);
    rt_silu(mid, FL_DIM);
    rt_lin_apply(&f->t2, mid, temb, 1);
}

/* One estimator evaluation. x/cond/mu are [T][80], spks is [80], out [T][80].
 * With `step >= 0` the prefix K/V comes from the cache and only rows
 * [row0, T) are pushed through the stack; out rows below row0 are left
 * untouched. step < 0 is the uncached path: every row, K/V in the scratch. */
static void fl_estimator(rt_flow *f, const float *x, const float *cond,
                         const float *mu, const float *spks, float t, int T,
                         int streaming, int step, int row0, float *out) {
    float temb[FL_DIM], mod[6 * FL_DIM], mod2[2 * FL_DIM];
    fl_timestep(f, t, temb);

    /* Input projection over [x, cond, mu, spks]. The two position convolutions
     * are causal with 30 taps of left padding each, so a window starting 60
     * rows before row0 reproduces the new rows exactly; anything older than
     * that cannot reach them. Rows below win0 hold stale noise and are simply
     * never read. */
    int win0 = row0 > FL_POS_CTX ? row0 - FL_POS_CTX : 0;
    int nw = T - win0;
    float *cat = (float *)cwick_xmalloc((size_t)nw * 4 * FL_MEL * sizeof(float));
    for (int i = 0; i < nw; i++) {
        size_t s = (size_t)(win0 + i) * FL_MEL;
        float *dst = cat + (size_t)i * 4 * FL_MEL;
        memcpy(dst, x + s, FL_MEL * sizeof(float));
        memcpy(dst + FL_MEL, cond + s, FL_MEL * sizeof(float));
        memcpy(dst + 2 * FL_MEL, mu + s, FL_MEL * sizeof(float));
        memcpy(dst + 3 * FL_MEL, spks, FL_MEL * sizeof(float));
    }
    rt_lin_apply(&f->in_proj, cat, f->res, nw);
    free(cat);
    rt_conv_apply(&f->pos1, f->res, f->x, nw);
    rt_mish(f->x, nw * FL_DIM);
    rt_conv_apply(&f->pos2, f->x, f->h, nw);
    rt_mish(f->h, nw * FL_DIM);
    vDSP_vadd(f->h, 1, f->res, 1, f->h, 1, (vDSP_Length)nw * FL_DIM);
    fl_rope(f, T);

    /* The input stage is ~1% of the stack, so the 22 layers below see only the
     * new rows, shifted down to index 0. */
    int rows = T - row0;
    int off = row0 - win0;
    if (off)
        memmove(f->h, f->h + (size_t)off * FL_DIM,
                (size_t)rows * FL_DIM * sizeof(float));
    const float *rope = f->rope + (size_t)row0 * FL_HD;

    for (int b = 0; b < FL_DEPTH; b++) {
        const fl_block *bl = &f->b[b];
        float silu[FL_DIM];
        memcpy(silu, temb, sizeof(silu));
        rt_silu(silu, FL_DIM);
        rt_lin_apply(&bl->norm, silu, mod, 1);
        const float *shift_msa = mod, *scale_msa = mod + FL_DIM;
        const float *gate_msa = mod + 2 * FL_DIM, *shift_mlp = mod + 3 * FL_DIM;
        const float *scale_mlp = mod + 4 * FL_DIM, *gate_mlp = mod + 5 * FL_DIM;

        rt_ln(f->nrm, f->h, rows, FL_DIM, 1e-6f);
        float onep[FL_DIM];
        for (int c = 0; c < FL_DIM; c++) onep[c] = 1.0f + scale_msa[c];
        for (int i = 0; i < rows; i++) {
            float *row = f->nrm + (size_t)i * FL_DIM;
            vDSP_vmul(row, 1, onep, 1, row, 1, FL_DIM);
            vDSP_vadd(row, 1, shift_msa, 1, row, 1, FL_DIM);
        }
        rt_lin_apply(&bl->q, f->nrm, f->q, rows);
        rt_lin_apply(&bl->k, f->nrm, f->k, rows);
        rt_lin_apply(&bl->v, f->nrm, f->v, rows);
        fl_rotate(rope, f->q, rows);
        fl_rotate(rope, f->k, rows);
        if (step >= 0) {
            for (int i = 0; i < rows; i++) {
                float *slot = fl_kv_at(f, step, b, row0 + i);
                memcpy(slot, f->k + (size_t)i * FL_DIM, FL_DIM * sizeof(float));
                memcpy(slot + FL_DIM, f->v + (size_t)i * FL_DIM,
                       FL_DIM * sizeof(float));
            }
            const float *kc = fl_kv_at(f, step, b, 0);
            fl_attention_kv(f, T, streaming, row0, kc, kc + FL_DIM, 2 * FL_DIM);
        } else {
            fl_attention_kv(f, T, streaming, 0, f->k, f->v, FL_DIM);
        }
        rt_lin_apply(&bl->o, f->att, f->x, rows);
        for (int i = 0; i < rows; i++) {
            size_t j = (size_t)i * FL_DIM;
            vDSP_vma(f->x + j, 1, gate_msa, 1, f->h + j, 1, f->h + j, 1, FL_DIM);
        }

        rt_ln(f->nrm, f->h, rows, FL_DIM, 1e-6f);
        for (int c = 0; c < FL_DIM; c++) onep[c] = 1.0f + scale_mlp[c];
        for (int i = 0; i < rows; i++) {
            float *row = f->nrm + (size_t)i * FL_DIM;
            vDSP_vmul(row, 1, onep, 1, row, 1, FL_DIM);
            vDSP_vadd(row, 1, shift_mlp, 1, row, 1, FL_DIM);
        }
        rt_lin_apply(&bl->ff_in, f->nrm, f->ff, rows);
        rt_gelu_tanh(f->ff, rows * FL_FF);
        rt_lin_apply(&bl->ff_out, f->ff, f->x, rows);
        for (int i = 0; i < rows; i++) {
            size_t j = (size_t)i * FL_DIM;
            vDSP_vma(f->x + j, 1, gate_mlp, 1, f->h + j, 1, f->h + j, 1, FL_DIM);
        }
    }

    float silu[FL_DIM];
    memcpy(silu, temb, sizeof(silu));
    rt_silu(silu, FL_DIM);
    rt_lin_apply(&f->norm_out, silu, mod2, 1);
    rt_ln(f->nrm, f->h, rows, FL_DIM, 1e-6f);
    float onep[FL_DIM];
    for (int c = 0; c < FL_DIM; c++) onep[c] = 1.0f + mod2[c];
    for (int i = 0; i < rows; i++) {
        float *row = f->nrm + (size_t)i * FL_DIM;
        vDSP_vmul(row, 1, onep, 1, row, 1, FL_DIM);
        vDSP_vadd(row, 1, mod2 + FL_DIM, 1, row, 1, FL_DIM);
    }
    rt_lin_apply(&f->proj_out, f->nrm, out + (size_t)row0 * FL_MEL, rows);
}

int rt_flow_frames(int n, int lookahead, int skip) {
    int render = n - lookahead - skip;
    return render > 0 ? render * 2 : 0;
}

int rt_flow_mel(rt_flow *f, const int *prompt_token, int pt,
                const float *prompt_feat, int pf, const float *embedding,
                const int *token, int n, int lookahead, int skip, int streaming,
                float *out) {
    if (!f || n <= 0 || pt < 0 || pf < 0) return -1;
    if (lookahead < 0 || lookahead > n) return -1;
    int render = n - lookahead;
    if (render <= 0) return 0;
    if (skip < 0 || skip > render) return -1;

    /* speaker vector: L2-normalise, then affine 192 -> 80 */
    double ss = 0.0;
    for (int i = 0; i < 192; i++) ss += (double)embedding[i] * embedding[i];
    float norm[192];
    float inv = (float)(1.0 / (sqrt(ss) + 1e-12));
    for (int i = 0; i < 192; i++) norm[i] = embedding[i] * inv;
    float spks[FL_MEL];
    rt_lin_apply(&f->spk, norm, spks, 1);

    /* token embedding for prompt + new, then the pre-lookahead convolution.
     * Running it over all n tokens and truncating to `render` is identical to
     * the reference's context form: conv1 sees the same padded sequence. */
    int L = pt + n;
    float *tok = (float *)cwick_xmalloc((size_t)L * FL_MEL * sizeof(float));
    for (int i = 0; i < pt; i++) {
        int id = prompt_token[i];
        if (id < 0) id = 0;
        memcpy(tok + (size_t)i * FL_MEL, f->emb + (size_t)id * FL_MEL,
               FL_MEL * sizeof(float));
    }
    for (int i = 0; i < n; i++) {
        int id = token[i];
        if (id < 0) id = 0;
        memcpy(tok + (size_t)(pt + i) * FL_MEL, f->emb + (size_t)id * FL_MEL,
               FL_MEL * sizeof(float));
    }
    float *hid = (float *)cwick_xmalloc((size_t)L * 1024 * sizeof(float));
    rt_conv_apply(&f->pre1, tok, hid, L);
    rt_leaky(hid, L * 1024, 0.01f);
    float *pre = (float *)cwick_xmalloc((size_t)L * FL_MEL * sizeof(float));
    rt_conv_apply(&f->pre2, hid, pre, L);
    for (int i = 0; i < L * FL_MEL; i++) pre[i] += tok[i];
    free(hid);

    int keep = pt + render;
    int T = keep * 2;
    if (pf > T) { free(tok); free(pre); return -1; }
    fl_reserve(f, T);

    float *mu = (float *)cwick_xmalloc((size_t)T * FL_MEL * sizeof(float));
    float *cond = (float *)cwick_xmalloc((size_t)T * FL_MEL * sizeof(float));
    memset(cond, 0, (size_t)T * FL_MEL * sizeof(float));
    for (int i = 0; i < keep; i++)
        for (int r = 0; r < 2; r++)
            memcpy(mu + (size_t)(i * 2 + r) * FL_MEL, pre + (size_t)i * FL_MEL,
                   FL_MEL * sizeof(float));
    memcpy(cond, prompt_feat, (size_t)pf * FL_MEL * sizeof(float));
    free(tok); free(pre);

    /* fixed noise, then euler along a cosine schedule */
    rt_rng rng;
    rt_rng_seed(&rng, 0x9e3779b97f4a7c15ULL);
    float *x = (float *)cwick_xmalloc((size_t)T * FL_MEL * sizeof(float));
    float *est = (float *)cwick_xmalloc((size_t)T * FL_MEL * sizeof(float));
    for (int i = 0; i < T * FL_MEL; i++) x[i] = rt_rng_normal(&rng);

    int ns = f->steps;
    float *span = (float *)cwick_xmalloc((size_t)(ns + 1) * sizeof(float));
    for (int j = 0; j <= ns; j++)
        span[j] = f->linear ? (float)j / (float)ns
                            : 1.0f - cosf((float)j / (float)ns * (float)M_PI * 0.5f);
    float t = span[0], dt = span[1] - span[0];

    /* Prefix reuse. Only complete 50-frame chunks are stable: a row in the
     * trailing partial chunk attended a truncated window and will see more
     * columns next time, so it has to be rendered again. That makes the reuse
     * boundary min(first, kvfill), not first -- with a voice prompt whose
     * frame count is not a multiple of 50 the two differ by up to one chunk.
     *
     * The cache survives across utterances, which is what keeps a realtime
     * session's first audio short: the voice prompt is the same every turn, so
     * only the part of it whose chunk closes inside the prompt may be carried
     * over -- later prompt frames attend the previous utterance's tokens. */
    int first = pf + skip * 2;
    int row0 = 0, cached = -1;
    uint64_t key = fl_prompt_key(prompt_token, pt, prompt_feat, pf, embedding);
    if (streaming && f->kv_on && f->cfg == 0.0f && fl_kv_reserve(f, T)) {
        cached = 0;
        if (f->kvfill > 0 && f->kv_key == key) {
            int lim = f->kvfill < first ? f->kvfill : first;
            int plim = pf / FL_CHUNK * FL_CHUNK;
            if (skip == 0 && lim > plim) lim = plim;
            row0 = lim;
        }
    }

    float *zero = NULL, *cfg = NULL;
    if (f->cfg > 0.0f) {
        zero = (float *)cwick_xmalloc((size_t)T * FL_MEL * sizeof(float));
        memset(zero, 0, (size_t)T * FL_MEL * sizeof(float));
        cfg = (float *)cwick_xmalloc((size_t)T * FL_MEL * sizeof(float));
    }
    float zspk[FL_MEL] = {0};
    int stable = T / FL_CHUNK * FL_CHUNK;
    int win0 = row0 > FL_POS_CTX ? row0 - FL_POS_CTX : 0;
    for (int step = 1; step <= ns; step++) {
        int slot = cached >= 0 ? step - 1 : -1;
        if (slot >= 0) {
            /* Restore the prefix latent the convolutions will read, and commit
             * this step's input latent for the frames that become stable here.
             * Both must happen before x is advanced. */
            if (row0 > win0)
                memcpy(x + (size_t)win0 * FL_MEL,
                       f->kx[slot] + (size_t)win0 * FL_MEL,
                       (size_t)(row0 - win0) * FL_MEL * sizeof(float));
            if (stable > row0)
                memcpy(f->kx[slot] + (size_t)row0 * FL_MEL,
                       x + (size_t)row0 * FL_MEL,
                       (size_t)(stable - row0) * FL_MEL * sizeof(float));
        }
        fl_estimator(f, x, cond, mu, spks, t, T, streaming, slot, row0, est);
        if (f->cfg > 0.0f) {
            fl_estimator(f, x, zero, zero, zspk, t, T, streaming, -1, 0, cfg);
            for (int i = row0 * FL_MEL; i < T * FL_MEL; i++)
                est[i] = (1.0f + f->cfg) * est[i] - f->cfg * cfg[i];
        }
        for (int i = row0 * FL_MEL; i < T * FL_MEL; i++) x[i] += dt * est[i];
        t += dt;
        if (step < ns) dt = span[step + 1] - t;
    }
    if (cached >= 0) {
        f->kvfill = stable;
        f->kv_key = key;
    }
    memcpy(out, x + (size_t)first * FL_MEL, (size_t)(T - first) * FL_MEL * sizeof(float));
    free(cfg); free(zero); free(span); free(x); free(est); free(mu); free(cond);
    return T - first;
}