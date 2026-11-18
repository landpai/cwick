/* rt_talker.c — see rt_talker.h.
 *
 * The LM is a 24-layer Qwen2 decoder (896 wide, 14 query heads over 2 KV
 * heads, SwiGLU 4864) with the speech vocabulary bolted on: text rows come
 * from embed_tokens, speech rows from speech_embedding, and the output head
 * is llm_decoder over 6761 speech ids. The checkpoint's own lm_head is not
 * used — the reference reads the final hidden state, not the text logits.
 *
 * Sampling is the reference's repetition-aware nucleus: top-p 0.8 / top-k 25,
 * and a token that has already appeared in the last ten is masked and the
 * draw is retaken from the full distribution. */
#include "rt_talker.h"

#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "log.h"
#include "rt_dsp.h"
#include "rt_flow.h"
#include "rt_hift.h"
#include "rt_torch.h"
#include "rt_tts_tokens.h"
#include "util.h"

#define TK_TAG "talker"
#define TK_H 896
#define TK_LAYERS 24
#define TK_QH 14
#define TK_KVH 2
#define TK_HD 64
#define TK_INTER 4864
#define TK_VOCAB 151936
#define TK_SPEECH 6561
#define TK_SOS 6561
#define TK_EOS 6562
#define TK_TASK 6563
#define TK_FILL 6564
#define TK_STOP_LO 6561
#define TK_STOP_HI 6760
#define TK_OUT 6761
#define TK_EPS 1e-6f
#define TK_THETA 1000000.0f
#define TK_CHUNK 25
#define TK_LOOKAHEAD 3

typedef struct {
    rt_lin q, k, v, o;
    rt_lin gate, up, down;
    const float *ln1, *ln2;
} tk_layer;

struct rt_talker {
    rt_pt *pt;
    const float *embed;   /* [151936][896] text rows */
    const float *speech;  /* [6761][896] speech rows */
    const float *dec_w;   /* [6761][896] output head, no bias */
    const float *norm_w;  /* [896] */
    tk_layer L[TK_LAYERS];
    rt_flow *flow;
    rt_hift *hift;
    cwick_tokenizer *tok;
    int chunk;
    /* KV cache, [layer][pos][kv*hd] */
    float *kc, *vc;
    int cap, len;
    /* decode scratch */
    float logits[TK_OUT];
    float *qh, *ao, *sc;  /* attention: gathered q, gathered out, scores */
    int sccap;
    float *px, *pn, *pq, *pk, *pv, *pa, *pg, *pu, *po, *pr; /* prefill + rope */
    int pcap;
};

static void tk_rms(float *dst, const float *src, const float *w, int m) {
    for (int r = 0; r < m; r++) {
        const float *s = src + (size_t)r * TK_H;
        double ss = 0.0;
        for (int i = 0; i < TK_H; i++) ss += (double)s[i] * s[i];
        float inv = 1.0f / sqrtf((float)(ss / TK_H) + TK_EPS);
        float *d = dst + (size_t)r * TK_H;
        for (int i = 0; i < TK_H; i++) d[i] = s[i] * inv * w[i];
    }
}

/* Qwen2 half-split rotary: the first half of each head rotates against the
 * second half. The angle table is shared by every head and both projections,
 * so it is built once per position block. */
static void tk_rope_table(float *cs, int pos0, int T) {
    for (int i = 0; i < T; i++) {
        float *row = cs + (size_t)i * TK_HD;
        for (int d = 0; d < TK_HD / 2; d++) {
            float a = (float)(pos0 + i) *
                      powf(TK_THETA, -2.0f * (float)d / (float)TK_HD);
            row[d] = cosf(a);
            row[TK_HD / 2 + d] = sinf(a);
        }
    }
}

static void tk_rotate(float *vec, const float *cs) {
    for (int d = 0; d < TK_HD / 2; d++) {
        float c = cs[d], s = cs[TK_HD / 2 + d];
        float x1 = vec[d], x2 = vec[d + TK_HD / 2];
        vec[d] = x1 * c - x2 * s;
        vec[d + TK_HD / 2] = x2 * c + x1 * s;
    }
}

static int tk_reserve(rt_talker *t, int T) {
    if (T <= t->cap) return 0;
    int cap = t->cap ? t->cap : 512;
    while (cap < T) cap *= 2;
    size_t per = (size_t)cap * TK_KVH * TK_HD;
    float *kc = (float *)cwick_xmalloc(TK_LAYERS * per * sizeof(float));
    float *vc = (float *)cwick_xmalloc(TK_LAYERS * per * sizeof(float));
    if (t->kc) {
        /* [layer][pos][kv*hd]: a wider stride means one copy per layer */
        size_t old = (size_t)t->cap * TK_KVH * TK_HD;
        size_t rows = (size_t)t->len * TK_KVH * TK_HD * sizeof(float);
        for (int l = 0; l < TK_LAYERS; l++) {
            memcpy(kc + (size_t)l * per, t->kc + (size_t)l * old, rows);
            memcpy(vc + (size_t)l * per, t->vc + (size_t)l * old, rows);
        }
    }
    free(t->kc); free(t->vc);
    t->kc = kc; t->vc = vc; t->cap = cap;
    t->qh = (float *)cwick_xrealloc(t->qh, (size_t)cap * TK_HD * sizeof(float));
    t->ao = (float *)cwick_xrealloc(t->ao, (size_t)cap * TK_HD * sizeof(float));
    return 0;
}

static void tk_prefill_reserve(rt_talker *t, int T) {
    if (T <= t->pcap) return;
    int cap = t->pcap ? t->pcap : 256;
    while (cap < T) cap *= 2;
    free(t->px); free(t->pn); free(t->pq); free(t->pk); free(t->pv);
    free(t->pa); free(t->pg); free(t->pu); free(t->po); free(t->pr);
    t->px = (float *)cwick_xmalloc((size_t)cap * TK_H * sizeof(float));
    t->pn = (float *)cwick_xmalloc((size_t)cap * TK_H * sizeof(float));
    t->pq = (float *)cwick_xmalloc((size_t)cap * TK_QH * TK_HD * sizeof(float));
    t->pk = (float *)cwick_xmalloc((size_t)cap * TK_KVH * TK_HD * sizeof(float));
    t->pv = (float *)cwick_xmalloc((size_t)cap * TK_KVH * TK_HD * sizeof(float));
    t->pa = (float *)cwick_xmalloc((size_t)cap * TK_QH * TK_HD * sizeof(float));
    t->pg = (float *)cwick_xmalloc((size_t)cap * TK_INTER * sizeof(float));
    t->pu = (float *)cwick_xmalloc((size_t)cap * TK_INTER * sizeof(float));
    t->po = (float *)cwick_xmalloc((size_t)cap * TK_H * sizeof(float));
    t->pr = (float *)cwick_xmalloc((size_t)cap * TK_HD * sizeof(float));
    t->pcap = cap;
}

rt_talker *rt_talker_open(const char *dir, cwick_error *err) {
    char path[1152];
    snprintf(path, sizeof(path), "%s/llm.pt", dir);
    rt_pt *pt = NULL;
    if (rt_pt_open(&pt, path) != 0) {
        cwick_error_set(err, CWICK_ERR_IO, "cannot open %s", path);
        return NULL;
    }
    rt_talker *t = (rt_talker *)cwick_xmalloc(sizeof(*t));
    memset(t, 0, sizeof(*t));
    t->pt = pt;
    t->chunk = TK_CHUNK;
    const char *e = getenv("RT_TTS_CHUNK");
    if (e && atoi(e) > 0) t->chunk = atoi(e);

    const rt_tensor *te = rt_pt_get(pt, "llm.model.model.embed_tokens.weight");
    const rt_tensor *se = rt_pt_get(pt, "speech_embedding.weight");
    const rt_tensor *dw = rt_pt_get(pt, "llm_decoder.weight");
    const rt_tensor *nw = rt_pt_get(pt, "llm.model.model.norm.weight");
    if (!te || !se || !dw || !nw || te->shape[1] != TK_H || se->shape[0] != TK_OUT) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "talker: unexpected LM layout");
        rt_pt_close(pt); free(t);
        return NULL;
    }
    t->embed = (const float *)te->data;
    t->speech = (const float *)se->data;
    t->dec_w = (const float *)dw->data;
    t->norm_w = (const float *)nw->data;

    int bad = 0;
    for (int i = 0; i < TK_LAYERS && !bad; i++) {
        char nm[192], full[256];
        tk_layer *L = &t->L[i];
        snprintf(nm, sizeof(nm), "llm.model.model.layers.%d", i);
        snprintf(full, sizeof(full), "%s.input_layernorm.weight", nm);
        const rt_tensor *w1 = rt_pt_get(pt, full);
        snprintf(full, sizeof(full), "%s.post_attention_layernorm.weight", nm);
        const rt_tensor *w2 = rt_pt_get(pt, full);
        if (!w1 || !w2) { bad = 1; break; }
        L->ln1 = (const float *)w1->data;
        L->ln2 = (const float *)w2->data;
        snprintf(full, sizeof(full), "%s.self_attn.q_proj", nm);
        bad |= rt_lin_load(&L->q, pt, full, 1);
        snprintf(full, sizeof(full), "%s.self_attn.k_proj", nm);
        bad |= rt_lin_load(&L->k, pt, full, 1);
        snprintf(full, sizeof(full), "%s.self_attn.v_proj", nm);
        bad |= rt_lin_load(&L->v, pt, full, 1);
        snprintf(full, sizeof(full), "%s.self_attn.o_proj", nm);
        bad |= rt_lin_load(&L->o, pt, full, 0);
        snprintf(full, sizeof(full), "%s.mlp.gate_proj", nm);
        bad |= rt_lin_load(&L->gate, pt, full, 0);
        snprintf(full, sizeof(full), "%s.mlp.up_proj", nm);
        bad |= rt_lin_load(&L->up, pt, full, 0);
        snprintf(full, sizeof(full), "%s.mlp.down_proj", nm);
        bad |= rt_lin_load(&L->down, pt, full, 0);
    }
    if (bad) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "talker: incomplete LM");
        rt_talker_close(t);
        return NULL;
    }
    snprintf(path, sizeof(path), "%s/flow.pt", dir);
    t->flow = rt_flow_open(path, err);
    if (!t->flow) { rt_talker_close(t); return NULL; }
    snprintf(path, sizeof(path), "%s/hift.pt", dir);
    t->hift = rt_hift_open(path, err);
    if (!t->hift) { rt_talker_close(t); return NULL; }

    snprintf(path, sizeof(path), "%s/CosyVoice-BlankEN", dir);
    cwick_model_spec spec;
    memset(&spec, 0, sizeof(spec));
    if (cwick_tokenizer_load(path, &spec, &t->tok, err) != CWICK_OK) {
        rt_talker_close(t);
        return NULL;
    }
    for (size_t i = 0; i < sizeof(rt_tts_tokens) / sizeof(rt_tts_tokens[0]); i++)
        cwick_tokenizer_add_special(t->tok, rt_tts_tokens[i].text, rt_tts_tokens[i].id);

    CWICK_INFO(TK_TAG, "talker ready: LM %dx%d (%d/%d heads), speech vocab %d, chunk %d tokens",
               TK_LAYERS, TK_H, TK_QH, TK_KVH, TK_SPEECH, t->chunk);
    return t;
}

void rt_talker_close(rt_talker *t) {
    if (!t) return;
    for (int i = 0; i < TK_LAYERS; i++) {
        tk_layer *L = &t->L[i];
        rt_lin_free(&L->q); rt_lin_free(&L->k); rt_lin_free(&L->v); rt_lin_free(&L->o);
        rt_lin_free(&L->gate); rt_lin_free(&L->up); rt_lin_free(&L->down);
    }
    free(t->kc); free(t->vc);
    free(t->qh); free(t->ao); free(t->sc);
    free(t->px); free(t->pn); free(t->pq); free(t->pk); free(t->pv);
    free(t->pa); free(t->pg); free(t->pu); free(t->po); free(t->pr);
    if (t->flow) rt_flow_close(t->flow);
    if (t->hift) rt_hift_close(t->hift);
    if (t->tok) cwick_tokenizer_free(t->tok);
    rt_pt_close(t->pt);
    free(t);
}

int rt_voice_load(const char *path, rt_voice *v, cwick_error *err) {
    memset(v, 0, sizeof(*v));
    rt_pt *pt = NULL;
    if (rt_pt_open(&pt, path) != 0) {
        cwick_error_set(err, CWICK_ERR_IO, "cannot open voice %s", path);
        return -1;
    }
    const rt_tensor *st = rt_pt_get(pt, "speech_token");
    const rt_tensor *sf = rt_pt_get(pt, "speech_feat");
    const rt_tensor *em = rt_pt_get(pt, "embedding");
    if (!st || !sf || !em || sf->ndim != 3 || sf->shape[2] != 80 || em->numel != 192) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "voice %s: unexpected layout", path);
        rt_pt_close(pt);
        return -1;
    }
    v->n_token = (int)st->numel;
    v->n_feat = (int)sf->shape[1];
    v->token = (int *)cwick_xmalloc((size_t)v->n_token * sizeof(int));
    v->feat = (float *)cwick_xmalloc((size_t)v->n_feat * 80 * sizeof(float));
    for (int i = 0; i < v->n_token; i++) v->token[i] = (int)rt_tensor_f32(st, i);
    for (int64_t i = 0; i < sf->numel; i++) v->feat[i] = rt_tensor_f32(sf, i);
    for (int i = 0; i < 192; i++) v->embedding[i] = rt_tensor_f32(em, i);
    rt_pt_close(pt);
    return 0;
}

void rt_voice_free(rt_voice *v) {
    if (!v) return;
    free(v->token); free(v->feat);
    memset(v, 0, sizeof(*v));
}

const char *rt_talker_prompt_text(void) {
    const char *e = getenv("RT_TTS_PROMPT_TEXT");
    return (e && *e) ? e : "<|endofprompt|>";
}

int rt_talker_sample_rate(const rt_talker *t) { (void)t; return 24000; }

/* ------------------------------------------------------------- LM decode --- */

static void tk_attention(rt_talker *t, int layer, int T, int causal) {
    const float scale = 1.0f / sqrtf((float)TK_HD);
    int group = TK_QH / TK_KVH;
    size_t stride = (size_t)t->cap * TK_KVH * TK_HD;
    const float *kc = t->kc + (size_t)layer * stride;
    const float *vc = t->vc + (size_t)layer * stride;
    int total = t->len + T;
    if (T * total > t->sccap) {
        t->sccap = T * total;
        t->sc = (float *)cwick_xrealloc(t->sc, (size_t)t->sccap * sizeof(float));
    }
    for (int h = 0; h < TK_QH; h++) {
        int kvh = h / group;
        /* the head's rows are strided inside pq/pa; BLAS wants them packed */
        for (int i = 0; i < T; i++) {
            memcpy(t->qh + (size_t)i * TK_HD,
                   t->pq + (size_t)i * TK_QH * TK_HD + h * TK_HD,
                   TK_HD * sizeof(float));
        }
        if (T == 1) {
            /* Decode: 14 heads x 24 layers x 2 products = 672 BLAS calls per
             * token, each on a [1][64] operand. sgemm hands those to
             * Accelerate's thread pool and the hand-off dwarfs the arithmetic;
             * sgemv stays on this thread. */
            cblas_sgemv(CblasRowMajor, CblasNoTrans, total, TK_HD, scale,
                        kc + kvh * TK_HD, TK_KVH * TK_HD, t->qh, 1, 0.0f, t->sc, 1);
        } else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, total, TK_HD,
                        scale, t->qh, TK_HD, kc + kvh * TK_HD, TK_KVH * TK_HD,
                        0.0f, t->sc, total);
        }
        for (int i = 0; i < T; i++) {
            float *row = t->sc + (size_t)i * total;
            int end = causal ? t->len + i + 1 : total;
            float mx = -INFINITY;
            for (int j = 0; j < end; j++) if (row[j] > mx) mx = row[j];
            float sum = 0.0f;
            for (int j = 0; j < end; j++) { row[j] = expf(row[j] - mx); sum += row[j]; }
            float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int j = 0; j < end; j++) row[j] *= inv;
            for (int j = end; j < total; j++) row[j] = 0.0f;
        }
        if (T == 1) {
            cblas_sgemv(CblasRowMajor, CblasTrans, total, TK_HD, 1.0f,
                        vc + kvh * TK_HD, TK_KVH * TK_HD, t->sc, 1, 0.0f, t->ao, 1);
        } else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, TK_HD, total,
                        1.0f, t->sc, total, vc + kvh * TK_HD, TK_KVH * TK_HD,
                        0.0f, t->ao, TK_HD);
        }
        for (int i = 0; i < T; i++) {
            memcpy(t->pa + (size_t)i * TK_QH * TK_HD + h * TK_HD,
                   t->ao + (size_t)i * TK_HD, TK_HD * sizeof(float));
        }
    }
}

/* Runs the stack over T new rows starting at position t->len, appending to
 * the KV cache. `out` receives the last row's final hidden state. */
static void tk_forward(rt_talker *t, const float *in, int T, float *out) {
    tk_prefill_reserve(t, T);
    memcpy(t->px, in, (size_t)T * TK_H * sizeof(float));
    tk_rope_table(t->pr, t->len, T);
    size_t stride = (size_t)t->cap * TK_KVH * TK_HD;
    for (int l = 0; l < TK_LAYERS; l++) {
        const tk_layer *L = &t->L[l];
        tk_rms(t->pn, t->px, L->ln1, T);
        rt_lin_apply(&L->q, t->pn, t->pq, T);
        rt_lin_apply(&L->k, t->pn, t->pk, T);
        rt_lin_apply(&L->v, t->pn, t->pv, T);
        for (int i = 0; i < T; i++) {
            float *q = t->pq + (size_t)i * TK_QH * TK_HD;
            float *k = t->pk + (size_t)i * TK_KVH * TK_HD;
            const float *cs = t->pr + (size_t)i * TK_HD;
            for (int h = 0; h < TK_QH; h++) tk_rotate(q + h * TK_HD, cs);
            for (int h = 0; h < TK_KVH; h++) tk_rotate(k + h * TK_HD, cs);
            memcpy(t->kc + (size_t)l * stride + (size_t)(t->len + i) * TK_KVH * TK_HD,
                   k, TK_KVH * TK_HD * sizeof(float));
            memcpy(t->vc + (size_t)l * stride + (size_t)(t->len + i) * TK_KVH * TK_HD,
                   t->pv + (size_t)i * TK_KVH * TK_HD, TK_KVH * TK_HD * sizeof(float));
        }
        tk_attention(t, l, T, 1);
        rt_lin_apply(&L->o, t->pa, t->po, T);
        for (int i = 0; i < T * TK_H; i++) t->px[i] += t->po[i];

        tk_rms(t->pn, t->px, L->ln2, T);
        rt_lin_apply(&L->gate, t->pn, t->pg, T);
        rt_lin_apply(&L->up, t->pn, t->pu, T);
        rt_silu(t->pg, T * TK_INTER);
        for (int i = 0; i < T * TK_INTER; i++) t->pg[i] *= t->pu[i];
        rt_lin_apply(&L->down, t->pg, t->po, T);
        for (int i = 0; i < T * TK_H; i++) t->px[i] += t->po[i];
    }
    t->len += T;
    tk_rms(out, t->px + (size_t)(T - 1) * TK_H, t->norm_w, 1);
}

/* The reference's repetition-aware nucleus sampler over the 6761 speech ids.
 * Only the top 25 can survive the top-k cut, so a partial selection in
 * descending (prob, id) order reproduces the stable sort the reference does
 * without paying for all 6761! comparisons. The eos id is masked below
 * min_len. */
static int tk_sample(const float *logits, const int *hist, int nhist,
                     int ignore_eos, rt_rng *rng, int *nk_out, float *p1_out) {
    float lp[TK_OUT];
    float mx = -INFINITY;
    for (int i = 0; i < TK_OUT; i++) {
        lp[i] = logits[i];
        if (lp[i] > mx) mx = lp[i];
    }
    if (ignore_eos) lp[TK_SPEECH] = -INFINITY;
    double sum = 0.0;
    for (int i = 0; i < TK_OUT; i++) { lp[i] = expf(lp[i] - mx); sum += lp[i]; }
    float inv = (float)(1.0 / sum);
    for (int i = 0; i < TK_OUT; i++) lp[i] *= inv;

    int keep[32];
    float kp[32];
    int nk = 0;
    double cum = 0.0;
    unsigned char used[TK_OUT];
    memset(used, 0, sizeof(used));
    while (nk < 25 && cum < 0.8) {
        int best = -1;
        float bp = 0.0f;
        for (int i = 0; i < TK_OUT; i++)
            if (!used[i] && lp[i] > bp) { bp = lp[i]; best = i; }
        if (best < 0) break;
        used[best] = 1;
        keep[nk] = best;
        kp[nk] = bp;
        cum += bp;
        nk++;
    }
    if (nk_out) *nk_out = nk;
    if (p1_out) *p1_out = nk ? kp[0] : 0.0f;
    if (nk == 0) return TK_EOS;
    double tot = 0.0;
    for (int i = 0; i < nk; i++) tot += kp[i];
    double r = rt_rng_uniform(rng) * tot;
    int pick = keep[nk - 1];
    for (int i = 0; i < nk; i++) {
        r -= kp[i];
        if (r <= 0.0) { pick = keep[i]; break; }
    }
    /* repetition-aware retake: a token seen in the last ten is masked and the
     * draw is retaken from the full distribution */
    int lo = nhist > 10 ? nhist - 10 : 0;
    for (int i = lo; i < nhist; i++) {
        if (hist[i] != pick) continue;
        lp[pick] = 0.0f;
        double s2 = 0.0;
        for (int j = 0; j < TK_OUT; j++) s2 += lp[j];
        double r2 = rt_rng_uniform(rng) * s2;
        pick = TK_SPEECH - 1;
        for (int j = 0; j < TK_OUT; j++) {
            r2 -= lp[j];
            if (r2 <= 0.0) { pick = j; break; }
        }
        break;
    }
    return pick;
}

/* --------------------------------------------------------------- synthesis --- */

int rt_talker_say(rt_talker *t, const rt_voice *v, const char *text,
                  rt_talker_pcm_cb cb, void *user, rt_talker_stats *stats) {
    if (!t || !v || !text) return -1;
    rt_talker_stats st;
    memset(&st, 0, sizeof(st));
    double t_start = (double)cwick_prof_ns() / 1e9;

    /* text = prompt_text + tts_text, encoded with the voice tokenizer */
    const char *pt = rt_talker_prompt_text();
    size_t plen = strlen(pt), tlen = strlen(text);
    char *joined = (char *)cwick_xmalloc(plen + tlen + 1);
    memcpy(joined, pt, plen);
    memcpy(joined + plen, text, tlen + 1);
    uint32_t *pt_ids = NULL, *all_ids = NULL;
    int64_t n_pt = 0, n_all = 0;
    cwick_error err = {0};
    if (cwick_tokenizer_encode(t->tok, pt, &pt_ids, &n_pt, &err) != CWICK_OK ||
        cwick_tokenizer_encode(t->tok, joined, &all_ids, &n_all, &err) != CWICK_OK) {
        CWICK_INFO(TK_TAG, "tokenize failed: %s", err.message);
        free(joined); free(pt_ids); free(all_ids);
        return -1;
    }
    free(joined);
    {
        char *back = NULL;
        if (cwick_tokenizer_decode(t->tok, all_ids, n_all, &back, &err) == CWICK_OK) {
            CWICK_INFO(TK_TAG, "text %d ids (prompt %d): %s", (int)n_all, (int)n_pt, back);
            free(back);
        }
    }
    int new_text = (int)(n_all - n_pt);
    if (new_text <= 0) { free(pt_ids); free(all_ids); return -1; }
    int min_len = new_text * 2;
    int max_len = new_text * 20;
    if (max_len > 1024) max_len = 1024; /* ~41 s of audio, one conversational turn */

    /* lm_input = [sos] + text_emb + [task_id] + prompt_speech_emb.
     * The prompt speech only helps when the LM also knows what the prompt
     * said: without a transcript it cannot tell where the reference clip ends
     * and the new sentence begins, and stops after a few tokens. The shipped
     * voices carry no transcript, so that case drops the prompt speech (the
     * reference's cross-lingual mode) and leaves the timbre to the flow. */
    int nps = strcmp(pt, "<|endofprompt|>") == 0 ? 0 : v->n_token;
    int L = 1 + (int)n_all + 1 + nps;
    float *in = (float *)cwick_xmalloc((size_t)L * TK_H * sizeof(float));
    memcpy(in, t->speech + (size_t)TK_SOS * TK_H, TK_H * sizeof(float));
    for (int i = 0; i < n_all; i++) {
        uint32_t id = all_ids[i];
        if (id >= TK_VOCAB) id = 0;
        memcpy(in + (size_t)(1 + i) * TK_H, t->embed + (size_t)id * TK_H,
               TK_H * sizeof(float));
    }
    memcpy(in + (size_t)(1 + n_all) * TK_H, t->speech + (size_t)TK_TASK * TK_H,
           TK_H * sizeof(float));
    for (int i = 0; i < nps; i++) {
        int id = v->token[i];
        if (id < 0 || id >= TK_OUT) id = 0;
        memcpy(in + (size_t)(2 + n_all + i) * TK_H, t->speech + (size_t)id * TK_H,
               TK_H * sizeof(float));
    }
    free(pt_ids); free(all_ids);
    st.lm_prompt_tokens = L;

    tk_reserve(t, L + max_len + 4);
    t->len = 0;
    double t0 = (double)cwick_prof_ns() / 1e9;
    float hidden[TK_H];
    tk_forward(t, in, L, hidden);
    st.lm_prefill_s = (double)cwick_prof_ns() / 1e9 - t0;
    free(in);
    CWICK_INFO(TK_TAG, "lm prefill %d tokens in %.3f s", L, st.lm_prefill_s);

    int *toks = (int *)cwick_xmalloc((size_t)max_len * sizeof(int));
    int ntok = 0;
    rt_rng rng;
    rt_rng_seed(&rng, 0x12345678ULL);
    int chunk = t->chunk;
    int rendered = 0; /* speech tokens already turned into audio */
    float *mel = NULL;
    int mel_cap = 0;
    int aborted = 0;

    int ndraw = 0;
    t0 = (double)cwick_prof_ns() / 1e9;
    for (int i = 0; i < max_len; i++) {
        rt_gemv(t->dec_w, hidden, t->logits, TK_H, TK_OUT);
        int nk = 0;
        float p1 = 0.0f;
        int id = tk_sample(t->logits, toks, ntok, i < min_len, &rng, &nk, &p1);
        ndraw++;
        st.lm_nucleus += nk;
        st.lm_top1_prob += p1;
        if (id >= TK_STOP_LO && id <= TK_STOP_HI) break;
        toks[ntok++] = id;
        tk_reserve(t, t->len + 1);
        tk_forward(t, t->speech + (size_t)id * TK_H, 1, hidden);

        /* emit a chunk once enough tokens (plus lookahead) are in hand */
        int avail = ntok - rendered;
        int fin = (i + 1 == max_len);
        if (avail >= chunk + TK_LOOKAHEAD || (fin && avail > 0)) {
            int take = avail > chunk + TK_LOOKAHEAD ? chunk + TK_LOOKAHEAD : avail;
            int look = (take > TK_LOOKAHEAD && !fin) ? TK_LOOKAHEAD : 0;
            /* the DiT sees the whole prefix: a chunk attends to every earlier
             * one, so only the tail frames are new audio */
            int total = rendered + take;
            int frames = rt_flow_frames(total, look, rendered);
            if (frames > mel_cap) {
                mel_cap = frames + 64;
                mel = (float *)cwick_xrealloc(mel, (size_t)mel_cap * 80 * sizeof(float));
            }
            double tf = (double)cwick_prof_ns() / 1e9;
            int got = rt_flow_mel(t->flow, v->token, v->n_token, v->feat, v->n_feat,
                                  v->embedding, toks, total, look, rendered, 1, mel);
            st.flow_s += (double)cwick_prof_ns() / 1e9 - tf;
            if (got < 0) { aborted = -1; break; }
            double th = (double)cwick_prof_ns() / 1e9;
            float *pcm = NULL;
            int ns = rt_hift_synth(t->hift, mel, got, &pcm);
            st.hift_s += (double)cwick_prof_ns() / 1e9 - th;
            if (ns < 0) { aborted = -1; break; }
            st.mel_frames += got;
            st.samples += ns;
            st.chunks++;
            if (st.first_audio_s == 0.0)
                st.first_audio_s = (double)cwick_prof_ns() / 1e9 - t_start;
            CWICK_INFO(TK_TAG, "chunk %d: %d tokens -> %d mel -> %d samples (%.2f s audio)",
                       st.chunks, take - look, got, ns, ns / 24000.0);
            int rc = cb ? cb(user, pcm, ns, 0) : 0;
            free(pcm);
            if (rc) { aborted = 1; break; }
            rendered += take - look;
        }
    }
    /* The loop interleaves synthesis with decoding, so the wall clock here
     * is not the LM's: reporting it as a decode rate understates the LM by
     * the whole flow cost. */
    st.lm_decode_s = (double)cwick_prof_ns() / 1e9 - t0 - st.flow_s - st.hift_s;
    st.speech_tokens = ntok;
    if (ndraw) { st.lm_nucleus /= ndraw; st.lm_top1_prob /= ndraw; }

    /* flush whatever is left, with no lookahead */
    if (aborted == 0 && rendered < ntok) {
        int frames = rt_flow_frames(ntok, 0, rendered);
        if (frames > mel_cap) {
            mel_cap = frames + 64;
            mel = (float *)cwick_xrealloc(mel, (size_t)mel_cap * 80 * sizeof(float));
        }
        double tf = (double)cwick_prof_ns() / 1e9;
        int got = rt_flow_mel(t->flow, v->token, v->n_token, v->feat, v->n_feat,
                              v->embedding, toks, ntok, 0, rendered, 1, mel);
        st.flow_s += (double)cwick_prof_ns() / 1e9 - tf;
        if (got > 0) {
            double th = (double)cwick_prof_ns() / 1e9;
            float *pcm = NULL;
            int ns = rt_hift_synth(t->hift, mel, got, &pcm);
            st.hift_s += (double)cwick_prof_ns() / 1e9 - th;
            if (ns > 0) {
                st.mel_frames += got;
                st.samples += ns;
                st.chunks++;
                if (st.first_audio_s == 0.0)
                    st.first_audio_s = (double)cwick_prof_ns() / 1e9 - t_start;
                if (cb) cb(user, pcm, ns, 0);
                free(pcm);
            }
        }
    }
    if (cb) cb(user, NULL, 0, 1);
    free(mel);
    free(toks);
    st.total_s = (double)cwick_prof_ns() / 1e9 - t_start;
    CWICK_INFO(TK_TAG, "say done: %d speech tokens, %d mel, %.2f s audio in %.2f s "
               "(lm %.2f, flow %.2f, hift %.2f; first audio %.2f s; "
               "top1 %.3f, nucleus %.1f)",
               st.speech_tokens, st.mel_frames, st.samples / 24000.0, st.total_s,
               st.lm_prefill_s + st.lm_decode_s, st.flow_s, st.hift_s, st.first_audio_s,
               st.lm_top1_prob, st.lm_nucleus);
    if (stats) *stats = st;
    return aborted < 0 ? -1 : aborted;
}
