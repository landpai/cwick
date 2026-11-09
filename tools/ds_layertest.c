/*
 * ds_layertest.c -- an independent recomputation of whole DeepSeek-V4.1
 * blocks, checked against the engine's own debug trace.
 *
 *   ds_layertest <checkpoint-root> <max-seq-len> <ds_run-path>
 *               [--layers 0,1,2,20] [--tokens 1000 2000 3000]
 *               [--moe 2:0] [--all]
 *
 * The engine (tools/ds_run) is run with DS_TRACE=1, which makes it print the
 * buffers it computes as hex floats. This program then re-derives those same
 * buffers from the shard bytes and compares. Nothing here calls into
 * ds_model.c: the point is to catch wiring mistakes, so the layer math is
 * written out again from the reference implementation and only the already
 * cross-checked primitives (format tables, GEMVs, norms, rope, hyper
 * connections) are shared.
 *
 * What is checked, per layer and position:
 *
 *   HIN / HIN1  the residual stream entering the block, all hc copies.
 *               Layer 0 is the embedding, layer 1 the embedding plus engram,
 *               every other layer is the previous block's output -- so a
 *               block that verifies also verifies the next layer's HIN.
 *   PREMIX      the pre_mix handed down from the previous block's FFN.
 *   SUBIN       hc_pre -> attn_norm.
 *   ATTNOUT     the attention sublayer: window ring, compressor, indexer.
 *   LATENT      the compressor's pooled latent, before RoPE.
 *   IDXTOP      which compressed positions each query keeps.
 *   FFNIN       hc_post -> the FFN's own hc_pre -> ffn_norm.
 *   FFNOUT      gate + routed experts + shared expert (opt-in, it is slow).
 *   ENGHASH     the n-gram hash row ids.
 *
 * Everything runs in the same float precision and the same order as the
 * engine, so agreement is close to exact; the tolerance only exists to
 * absorb the last bit of a different summation order.
 */
#include "cwick.h"
#include "ds.h"
#include "ds_ops.h"
#include "json.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXL 64
#define MAXTOK 16
#define MAXHC 8

/* ------------------------------------------------------------------ */
/* small helpers                                                        */
/* ------------------------------------------------------------------ */

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "out of memory (%zu bytes)\n", n);
        exit(1);
    }
    return p;
}

static void die(const char *what) {
    fprintf(stderr, "ds_layertest: %s\n", what);
    exit(1);
}

/* ------------------------------------------------------------------ */
/* config, read straight from config.json                              */
/* ------------------------------------------------------------------ */

typedef struct {
    int vocab, dim, moe_inter, n_layers, n_heads, head_dim, rope_dim;
    int q_lora, o_lora, o_groups, window;
    int n_experts, topk;
    float swiglu_limit, rms_eps, gate_temp, route_scale;
    int norm_topk;
    int ratio[MAXL];
    int kv_src[MAXL], n_kv_src;
    int idx_src[MAXL], n_idx_src;
    int idx_heads, idx_hd, idx_topk, cand_layer, cand_blocks, cand_block;
    int hc, hc_iters;
    float hc_eps;
    int eng_layers[MAXL], n_eng;
    int eng_max_ngram, eng_heads, eng_hd;
    double rope_theta, crope_theta, yarn_factor;
    int yarn_orig, beta_fast, beta_slow;
} cfg_t;

static int json_int(json_value *o, const char *k, int dflt) {
    json_value *v = cwick_json_get(o, k);
    return v && v->kind == JSON_NUMBER ? (int)cwick_json_int(v, dflt) : dflt;
}

static double json_num(json_value *o, const char *k, double dflt) {
    json_value *v = cwick_json_get(o, k);
    return v && v->kind == JSON_NUMBER ? cwick_json_num(v, dflt) : dflt;
}

/* The reference's ModelArgs takes its field names verbatim, so the HF
 * config.json is flattened by hand here: the rope_* values live inside
 * rope_scaling and a few names differ from the reference's field names. */
static void load_config(const char *root, cfg_t *c) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", root);
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open config.json");
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *txt = xmalloc((size_t)n + 1);
    if (fread(txt, 1, (size_t)n, f) != (size_t)n) die("short read on config.json");
    txt[n] = 0;
    fclose(f);

    cwick_error err = {0};
    json_value *top = cwick_json_parse(txt, (size_t)n, &err);
    if (!top) die("config.json does not parse");
    json_value *j = cwick_json_get(top, "text_config");
    if (!j || j->kind != JSON_OBJECT) j = top;

    memset(c, 0, sizeof(*c));
    c->vocab = json_int(j, "vocab_size", 129280);
    c->dim = json_int(j, "hidden_size", 5120);
    c->moe_inter = json_int(j, "moe_intermediate_size", 2304);
    c->n_layers = json_int(j, "num_hidden_layers", 40);
    c->n_heads = json_int(j, "num_attention_heads", 64);
    c->head_dim = json_int(j, "head_dim", 512);
    c->rope_dim = json_int(j, "qk_rope_head_dim", 64);
    c->q_lora = json_int(j, "q_lora_rank", 1280);
    c->o_lora = json_int(j, "o_lora_rank", 1024);
    c->o_groups = json_int(j, "o_groups", 8);
    c->window = json_int(j, "sliding_window", 128);
    c->n_experts = json_int(j, "n_routed_experts", 384);
    c->topk = json_int(j, "num_experts_per_tok", 6);
    c->swiglu_limit = (float)json_num(j, "swiglu_limit", 10.0);
    c->rms_eps = (float)json_num(j, "rms_norm_eps", 1e-20);
    c->gate_temp = (float)json_num(j, "gate_temp", 1.0);
    c->route_scale = (float)json_num(j, "routed_scaling_factor", 1.5);
    json_value *nt = cwick_json_get(j, "norm_topk_prob");
    c->norm_topk = nt ? cwick_json_bool(nt, 1) : 1;

    json_value *ar = cwick_json_get(j, "compress_ratios");
    if (ar && ar->kind == JSON_ARRAY)
        for (size_t i = 0; i < ar->count && i < MAXL; i++)
            c->ratio[i] = (int)cwick_json_int(cwick_json_array_at(ar, i), 0);
    ar = cwick_json_get(j, "kv_source_layer_ids");
    if (ar && ar->kind == JSON_ARRAY)
        for (size_t i = 0; i < ar->count && i < MAXL; i++)
            c->kv_src[c->n_kv_src++] = (int)cwick_json_int(cwick_json_array_at(ar, i), -1);
    ar = cwick_json_get(j, "index_source_layer_ids");
    if (ar && ar->kind == JSON_ARRAY)
        for (size_t i = 0; i < ar->count && i < MAXL; i++)
            c->idx_src[c->n_idx_src++] = (int)cwick_json_int(cwick_json_array_at(ar, i), -1);
    c->cand_layer = json_int(j, "candidate_source_layer_id", -1);
    c->cand_blocks = json_int(j, "candidate_topk_blocks", 2048);
    c->cand_block = json_int(j, "candidate_block_size", 8);
    c->idx_heads = json_int(j, "index_n_heads", 32);
    c->idx_hd = json_int(j, "index_head_dim", 128);
    c->idx_topk = json_int(j, "index_topk", 512);

    c->hc = json_int(j, "hc_mult", 4);
    c->hc_iters = json_int(j, "hc_sinkhorn_iters", 20);
    c->hc_eps = (float)json_num(j, "hc_eps", 1e-6);

    ar = cwick_json_get(j, "engram_layer_ids");
    if (ar && ar->kind == JSON_ARRAY)
        for (size_t i = 0; i < ar->count && i < MAXL; i++)
            c->eng_layers[c->n_eng++] = (int)cwick_json_int(cwick_json_array_at(ar, i), -1);
    c->eng_max_ngram = json_int(j, "engram_max_ngram_size", 4);
    c->eng_heads = json_int(j, "engram_n_heads", 8);
    c->eng_hd = json_int(j, "engram_head_dim", 256);

    c->rope_theta = json_num(j, "rope_theta", 10000.0);
    c->crope_theta = json_num(j, "compress_rope_theta", 160000.0);
    json_value *rs = cwick_json_get(j, "rope_scaling");
    c->yarn_factor = rs ? json_num(rs, "factor", 16.0) : 1.0;
    c->yarn_orig = rs ? json_int(rs, "original_max_position_embeddings", 65536) : 0;
    c->beta_fast = rs ? json_int(rs, "beta_fast", 32) : 32;
    c->beta_slow = rs ? json_int(rs, "beta_slow", 1) : 1;

    cwick_json_free(top);
    free(txt);

    if (c->n_layers > MAXL) die("too many layers");
    if (c->hc > MAXHC) die("too many hc copies");
}

/* ------------------------------------------------------------------ */
/* checkpoint access                                                    */
/* ------------------------------------------------------------------ */

static cwick_safetensors S;

static const uint16_t *bf16(const char *name) {
    cwick_error err = {0};
    cwick_st_tensor t;
    if (cwick_safetensors_get(&S, name, &t, &err) != CWICK_OK) {
        fprintf(stderr, "missing tensor %s\n", name);
        die("checkpoint is incomplete");
    }
    return (const uint16_t *)t.data;
}

static const float *f32(const char *name) {
    cwick_error err = {0};
    cwick_st_tensor t;
    if (cwick_safetensors_get(&S, name, &t, &err) != CWICK_OK) {
        fprintf(stderr, "missing tensor %s\n", name);
        die("checkpoint is incomplete");
    }
    return (const float *)t.data;
}

/* the same, for a name built from a layer index */
static const uint16_t *bf16_(const char *fmt, int L) {
    char nm[256];
    snprintf(nm, sizeof(nm), fmt, L);
    return bf16(nm);
}

static const float *f32_(const char *fmt, int L) {
    char nm[256];
    snprintf(nm, sizeof(nm), fmt, L);
    return f32(nm);
}

/* `base` is the tensor name without the .weight / .scale suffix. */
static void qw(const char *base, const uint8_t **w, const uint8_t **sc,
               int *rows, int *cols, int *fp4) {
    char nm[512];
    cwick_error err = {0};
    cwick_st_tensor t;
    snprintf(nm, sizeof(nm), "%s.weight", base);
    if (cwick_safetensors_get(&S, nm, &t, &err) != CWICK_OK) {
        fprintf(stderr, "missing tensor %s\n", nm);
        die("checkpoint is incomplete");
    }
    *w = t.data;
    *rows = (int)t.rows;
    /* fp4 weights are two nibbles per byte, so the stored column count is
     * half the logical K the gemv wants */
    *fp4 = (t.dtype == CWICK_ST_I8);
    *cols = (int)t.cols * (*fp4 ? 2 : 1);
    snprintf(nm, sizeof(nm), "%s.scale", base);
    if (cwick_safetensors_get(&S, nm, &t, &err) != CWICK_OK) {
        fprintf(stderr, "missing tensor %s\n", nm);
        die("checkpoint is incomplete");
    }
    *sc = t.data;
}

/* ------------------------------------------------------------------ */
/* the engine's trace                                                   */
/* ------------------------------------------------------------------ */

typedef struct {
    char tag[24];
    int a, b;
    int is_int;
    float *v;
    int64_t *iv;
    int n;
} TEnt;

static TEnt *tent;
static size_t n_tent, cap_tent;

static void tent_push(const char *tag, int a, int b, int is_int) {
    if (n_tent == cap_tent) {
        cap_tent = cap_tent ? cap_tent * 2 : 4096;
        tent = realloc(tent, cap_tent * sizeof(TEnt));
        if (!tent) die("out of memory for the trace");
    }
    TEnt *e = &tent[n_tent++];
    memset(e, 0, sizeof(*e));
    snprintf(e->tag, sizeof(e->tag), "%s", tag);
    e->a = a;
    e->b = b;
    e->is_int = is_int;
    e->v = xmalloc(sizeof(float) * 64);
    e->iv = xmalloc(sizeof(int64_t) * 64);
}

static const TEnt *tfind(const char *tag, int a, int b) {
    for (size_t i = 0; i < n_tent; i++)
        if (tent[i].a == a && tent[i].b == b && strcmp(tent[i].tag, tag) == 0)
            return &tent[i];
    return NULL;
}

/* Run the engine with tracing on and parse everything it prints. */
static void capture_trace(const char *exe, const char *root, int seq,
                          const int *toks, int ntok, int maxl) {
    char cmd[4096];
    int off = snprintf(cmd, sizeof(cmd), "%s %s %d", exe, root, seq);
    for (int i = 0; i < ntok && off < (int)sizeof(cmd) - 16; i++)
        off += snprintf(cmd + off, sizeof(cmd) - (size_t)off, " %d", toks[i]);

    setenv("DS_TRACE", "1", 1);
    char ml[16];
    snprintf(ml, sizeof(ml), "%d", maxl);
    setenv("DS_TRACE_MAXL", ml, 1);

    FILE *p = popen(cmd, "r");
    if (!p) die("cannot run the engine");
    size_t cap = 1 << 20, len = 0;
    char *buf = xmalloc(cap);
    for (;;) {
        if (len + (1 << 16) + 1 > cap) {
            cap *= 2;
            buf = realloc(buf, cap);
            if (!buf) die("out of memory for the trace");
        }
        size_t got = fread(buf + len, 1, 1 << 16, p);
        if (!got) break;
        len += got;
    }
    buf[len] = 0;
    int rc = pclose(p);
    if (rc != 0) fprintf(stderr, "warning: engine exited %d\n", rc);

    char *save = NULL;
    for (char *ln = strtok_r(buf, "\n", &save); ln; ln = strtok_r(NULL, "\n", &save)) {
        char tag[24];
        int a, b;
        if (sscanf(ln, "%23s %d %d", tag, &a, &b) != 3) continue;
        int is_int = strcmp(tag, "ENGHASH") == 0 || strcmp(tag, "IDXTOP") == 0 ||
                     strcmp(tag, "PROBE_SEL") == 0 || strcmp(tag, "PROBE_IDS") == 0;
        tent_push(tag, a, b, is_int);
        TEnt *e = &tent[n_tent - 1];
        char *q = ln;
        for (int k = 0; k < 3; k++) { /* skip tag, a, b */
            while (*q == ' ') q++;
            while (*q && *q != ' ') q++;
        }
        int cnt = 0, capv = 64;
        while (*q) {
            while (*q == ' ') q++;
            if (!*q) break;
            char *end = NULL;
            if (cnt == capv) {
                capv *= 2;
                e->v = realloc(e->v, sizeof(float) * (size_t)capv);
                e->iv = realloc(e->iv, sizeof(int64_t) * (size_t)capv);
                if (!e->v || !e->iv) die("out of memory for a trace line");
            }
            if (is_int) {
                long long val = strtoll(q, &end, 10);
                if (end == q) break;
                e->iv[cnt++] = (int64_t)val;
            } else {
                double val = strtod(q, &end);
                if (end == q) break;
                e->v[cnt++] = (float)val;
            }
            q = end;
        }
        e->n = cnt;
    }
    free(buf);
    fprintf(stderr, "trace: %zu lines\n", n_tent);
}

/* ------------------------------------------------------------------ */
/* report                                                               */
/* ------------------------------------------------------------------ */

static int n_checks, n_fail;
static double worst_rel;

static void cmpv(const char *what, int L, int p, const float *got,
                 const float *ref, int n) {
    n_checks++;
    if (n <= 0) return;
    double num = 0.0, den = 0.0;
    int wi = 0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)got[i] - (double)ref[i]);
        if (d > num) {
            num = d;
            wi = i;
        }
        den += (double)ref[i] * (double)ref[i];
    }
    double rms = sqrt(den / n);
    double rel = num / (rms > 1e-30 ? rms : 1e-30);
    if (rel > worst_rel) worst_rel = rel;
    if (!(num == 0.0)) {
        n_fail++;
        int nbad = 0, shown = 0;
        for (int i = 0; i < n; i++)
            if (fabs((double)got[i] - (double)ref[i]) > 1e-4 * (rms > 1e-30 ? rms : 1)) nbad++;
        printf("  FAIL %-18s L=%-3d p=%-3d n=%-5d max|diff|=%.6g rel/rms=%.3g "
               "at %d (engine %.9g vs oracle %.9g) nbad=%d",
               what, L, p, n, num, rel, wi, got[wi], ref[wi], nbad);
        for (int i = 0; i < n && shown < 6; i++)
            if (fabs((double)got[i] - (double)ref[i]) > 1e-4 * (rms > 1e-30 ? rms : 1)) {
                printf(" [%d:%.6g/%.6g]", i, got[i], ref[i]);
                shown++;
            }
        printf("\n");
    }
}

static void cmpi(const char *what, int L, int p, const int64_t *got,
                 const int *ref, int n) {
    n_checks++;
    int bad = -1;
    for (int i = 0; i < n; i++)
        if (got[i] != ref[i]) {
            bad = i;
            break;
        }
    if (bad >= 0) {
        n_fail++;
        printf("  FAIL %-18s L=%-3d p=%-3d first difference at %d: "
               "engine %lld vs oracle %d\n",
               what, L, p, bad, (long long)got[bad], ref[bad]);
    }
}

/* ------------------------------------------------------------------ */
/* oracle state                                                         */
/* ------------------------------------------------------------------ */

/* Temporary bisection probes: the engine prints PROBE_* for intermediate
 * quantities inside the attention sublayer, and these compare them one by
 * one so the first divergence can be located. */
static int probe_want;

static void pcmp(const char *tag, int L, int p, const float *ref, int n) {
    if (!probe_want || n <= 0) return;
    const TEnt *t = tfind(tag, L, p);
    if (t) cmpv(tag, L, p, t->v, ref, n);
}

static cfg_t C;
static int D, HC, H, HD, RD, W, IH, NH, NGRP, OLORA, QLR, INTER, NEXP, TOPK;
static int max_cmp;

static float *STRM;        /* [ntok][hc][dim] the residual stream */
static float *PRE;      /* [ntok][hc] the pre_mix handed down */
static float *RES;      /* [hc][dim] scratch for hc_post's residual */
static float *winkv;    /* [n_layers][window][head_dim] */
static float *cmpkv;    /* [n_kv_src][max_cmp][head_dim] */
static float *cmpst;    /* [n_kv_src][ratio][head_dim] pooling state */
static float *cmpss;    /* [n_kv_src][ratio][head_dim] gate state */
static float *idxk;     /* [n_kv_src][max_cmp][index_head_dim] */
static int64_t *ccache; /* [seq] compressed token ids */

/* the reference's shared_attn: what a source layer publishes for the layers
 * after it. It is per position, not one scratch buffer: a layer reads the
 * selection its source made *at its own position*, and with more than one
 * token in the chunk the source has already moved on to the last one. */
static int *sh_topk;    /* [ntok][idx_topk] */
static int *sh_n;       /* [ntok] */
static uint8_t *sh_cand; /* [ntok][max_cmp] */
static int NTOK = 1;

static int src_slot[MAXL], idx_slot[MAXL];
static int is_kv_src[MAXL], is_idx_src[MAXL], is_cand_src[MAXL], uses_cand[MAXL];
static int eng_slot[MAXL];

static ds_freqs fr_yarn, fr_plain;

static ds_freqs *fr_for(int L) {
    return C.ratio[L] > 0 ? &fr_yarn : &fr_plain;
}

/* ---- scratch, one buffer per live value so nothing aliases ---- */
static float *sub, *subl, *moe_y;
static float *qr, *q, *iq, *kv, *ckv, *cgate, *latent, *o, *o_g, *attn_out;
static float *ex_g, *ex_u, *ex_h, *ex_p, *iw;
static float *apre, *apost, *fpre, *fpost, *comb, *comb2;
static float *router, *selw;
static int *sel, *idxs, *blk_out, *heap_idx;
static float *scores, *bsc, *heap;
static uint8_t *act_q;
static uint8_t *act_s;
static float *eng_rows, *eng_kv, *eng_prod;
static int64_t *eng_hash;
static float *lat;      /* [head_dim] the compressor's latent */
static int *idxtop;     /* [window + index_topk] the chosen positions */

/* quantize an activation the way the reference's act_quant does before a
 * quantized GEMM, run the GEMM, then round to bf16 the way the reference's
 * gemm output dtype does */
static void lin_q(const uint8_t *w, const uint8_t *sc, int rows, int cols,
                  int fp4, const float *x, float *out) {
    for (int b = 0; b < cols; b += 32)
        act_s[b / 32] = ds_act_quant_block(x + b, act_q + b);
    if (fp4) ds_fp4_gemv(act_q, act_s, w, sc, rows, cols, out);
    else ds_fp8_gemv(act_q, act_s, w, sc, rows, cols, out);
    for (int i = 0; i < rows; i++) out[i] = ds_bf16_round(out[i]);
}

static void lin_bf16(const uint16_t *w, const float *x, int rows, int cols,
                     float *out, int round) {
    ds_bf16_gemv(w, x, NULL, rows, cols, out);
    if (round)
        for (int i = 0; i < rows; i++) out[i] = ds_bf16_round(out[i]);
}

/* ------------------------------------------------------------------ */
/* engram                                                               */
/* ------------------------------------------------------------------ */

/* NgramHashState.forward for one position: the row ids the n-gram hash
 * picks out of the layer's table. */
static void engram_hash(int p, int e, int64_t *out) {
    int ng = C.eng_max_ngram, nh = C.eng_heads;
    int64_t tok[8];
    for (int s = 0; s < ng; s++) {
        int blocked = (p < s);
        int64_t src = 0;
        if (!blocked) {
            src = ccache[p - s];
            if (src < 0) blocked = 1; /* DEAD: image spans only */
        }
        tok[s] = blocked ? ds_engram_pad_id : src;
    }
    /* XOR the multiplied ids one lookback at a time, so the running value
     * after step i is the hash of the (i+1)-gram */
    int64_t rolling = tok[0] * ds_engram_mult[e][0];
    for (int i = 1; i < ng; i++) {
        rolling ^= tok[i] * ds_engram_mult[e][i];
        for (int h = 0; h < nh; h++) {
            int col = (i - 1) * nh + h;
            out[col] = rolling % ds_engram_primes[e][i - 1][h] +
                       ds_engram_offsets[e][col];
        }
    }
}

/* Engram.forward for one position: writes the n-gram lookup into the
 * residual stream, gated by how well it matches that stream. */
static void engram_apply(int L, int p) {
    int e = eng_slot[L];
    int ncol = (C.eng_max_ngram - 1) * C.eng_heads;
    int eh = C.eng_hd;
    int64_t rows[64];
    engram_hash(p, e, rows);

    /* the table stays fp8: dequantize the rows on lookup, then round to
     * bf16 exactly as the reference's ParallelEngramEmbedding does */
    const uint8_t *ew, *es;
    int er, ec, ed;
    char base[128];
    snprintf(base, sizeof(base), "layers.%d.engram.embed", L);
    qw(base, &ew, &es, &er, &ec, &ed);
    for (int r = 0; r < ncol; r++) {
        int64_t row = rows[r];
        const uint8_t *w = ew + (size_t)row * eh;
        const uint8_t *s = es + (size_t)row * (eh / 32);
        for (int b = 0; b < eh; b += 32) {
            float sc = ds_e8m0_to_f32(s[b / 32]);
            for (int j = 0; j < 32; j++)
                eng_rows[r * eh + b + j] =
                    ds_bf16_round(ds_e4m3_to_f32(w[b + j]) * sc);
        }
    }

    /* wkv: one key per hc copy plus a shared value */
    const uint8_t *ww, *ws;
    int wr, wc, wd;
    snprintf(base, sizeof(base), "layers.%d.engram.wkv", L);
    qw(base, &ww, &ws, &wr, &wc, &wd);
    lin_q(ww, ws, wr, wc, 0, eng_rows, eng_kv);

    const uint16_t *qwv = bf16_("layers.%d.engram.q_weight", L);
    const uint16_t *kwv = bf16_("layers.%d.engram.k_weight", L);
    /* only ever used as a product */
    for (int j = 0; j < HC; j++)
        for (int i = 0; i < D; i++)
            eng_prod[(size_t)j * D + i] =
                ds_bf16_to_f32(qwv[(size_t)j * D + i]) *
                ds_bf16_to_f32(kwv[(size_t)j * D + i]);

    for (int j = 0; j < HC; j++) {
        /* the reference keeps the stream and the key in fp32 and does each of
         * these ops in fp32, so the oracle does too: the sums are widened to
         * double only to stop the accumulation itself from drifting, which is
         * as close as a sequential loop gets to torch's blocked reduction */
        double sh = 0.0, sk = 0.0, dot = 0.0;
        for (int i = 0; i < D; i++) {
            float hj = STRM[((size_t)p * HC + j) * D + i];
            float kj = eng_kv[(size_t)j * D + i];
            sh += (double)hj * hj;
            sk += (double)kj * kj;
            dot += (double)hj * eng_prod[(size_t)j * D + i] * kj;
        }
        /* normalized per (token, hc copy) over dim, not jointly over copies */
        float rstd = (float)(1.0 / sqrt(sh / D + C.rms_eps)) *
                     (float)(1.0 / sqrt(sk / D + C.rms_eps));
        float g = (float)dot * rstd * powf((float)D, -0.5f);
        /* signed sqrt before the sigmoid, matching the training kernel */
        float a = fabsf(g) < 1e-6f ? 1e-6f : fabsf(g);
        float gate = 1.0f / (1.0f + expf(-copysignf(sqrtf(a), g)));
        for (int i = 0; i < D; i++) {
            float v = STRM[((size_t)p * HC + j) * D + i];
            STRM[((size_t)p * HC + j) * D + i] =
                ds_bf16_round(v + gate * eng_kv[HC * D + i]);
        }
    }
}

/* ------------------------------------------------------------------ */
/* compressor                                                           */
/* ------------------------------------------------------------------ */

/* Returns 1 and fills `lat` when a group completed at this position. Only
 * kv-source layers own a compressor; everyone else reads that cache. */
static int compress(int L, int p, const float *x, float *lat, int *j_out,
                    int64_t *lp_out) {
    int ratio = C.ratio[L];
    int src = src_slot[L];
    const uint16_t *nw = bf16_("layers.%d.attn.compressor.norm.weight", L);

    if (ratio == 1) {
        /* one token per group: a plain projection, bf16 in and out */
        const uint16_t *w = bf16_("layers.%d.attn.compressor.wkv.weight", L);
        lin_bf16(w, x, HD, D, lat, 1);
        ds_rmsnorm_bf16w(lat, lat, nw, HD, C.rms_eps);
        *j_out = p;
        *lp_out = p;
        return 1;
    }

    /* ratio > 1: the projections run in fp32 (bf16 weights upcast exactly),
     * the pooling is fp32, only the norm output is bf16 */
    const uint16_t *wv = bf16_("layers.%d.attn.compressor.wkv.weight", L);
    const uint16_t *wg = bf16_("layers.%d.attn.compressor.wgate.weight", L);
    lin_bf16(wv, x, HD, D, ckv, 0);
    lin_bf16(wg, x, HD, D, cgate, 0);
    int slot = p % ratio;
    memcpy(cmpst + ((size_t)src * ratio + slot) * HD, ckv, sizeof(float) * (size_t)HD);
    memcpy(cmpss + ((size_t)src * ratio + slot) * HD, cgate, sizeof(float) * (size_t)HD);
    if ((p + 1) % ratio != 0) return 0; /* group still filling */

    *j_out = (p + 1) / ratio - 1;
    *lp_out = (int64_t)*j_out * ratio;
    for (int i = 0; i < HD; i++) {
        float mx = -INFINITY, w[8], s = 0.0f, acc = 0.0f;
        for (int r = 0; r < ratio; r++) {
            float v = cmpss[((size_t)src * ratio + r) * HD + i];
            if (v > mx) mx = v;
        }
        for (int r = 0; r < ratio; r++) {
            w[r] = expf(cmpss[((size_t)src * ratio + r) * HD + i] - mx);
            s += w[r];
        }
        for (int r = 0; r < ratio; r++)
            acc += cmpst[((size_t)src * ratio + r) * HD + i] * (w[r] / s);
        lat[i] = acc;
    }
    ds_rmsnorm_bf16w(lat, lat, nw, HD, C.rms_eps);
    return 1;
}

/* ------------------------------------------------------------------ */
/* indexer                                                              */
/* ------------------------------------------------------------------ */

/* Level one of the two-level top-k: keep the `cand_blocks` best blocks of
 * `cand_block` positions, each scored by its best position, with the block
 * holding the newest position pinned in. Writes a mask over `width`. */
static void select_candidate_blocks(const float *sc, int width, uint8_t *mask) {
    int bs = C.cand_block;
    int pad = (bs - width % bs) % bs;
    int nblk = (width + pad) / bs;
    for (int b = 0; b < nblk; b++) {
        float mx = -INFINITY;
        for (int j = 0; j < bs; j++) {
            int t = b * bs + j;
            float v = t < width ? sc[t] : -INFINITY;
            if (v > mx) mx = v;
        }
        bsc[b] = mx;
    }
    /* the block with this query's newest position is only partly filled, so
     * pin it in: it holds the most recent tokens but could otherwise be
     * outscored by an older, full block */
    int last = (width - 1) / bs;
    if (last >= 0 && last < nblk) bsc[last] = INFINITY;

    int k = C.cand_blocks < nblk ? C.cand_blocks : nblk;
    int n = ds_topk_scores(bsc, nblk, k, blk_out, heap, heap_idx);
    memset(mask, 0, (size_t)width);
    for (int i = 0; i < n; i++)
        if (bsc[blk_out[i]] > -INFINITY)
            for (int j = 0; j < bs; j++) {
                int t = blk_out[i] * bs + j;
                if (t < width) mask[t] = 1;
            }
}

/* The indexer of an index-source layer: a small side attention that scores
 * the compressed positions, then keeps the best `index_topk`. Writes the
 * chosen positions, each shifted by the window offset, into `out`. */
static int run_indexer(int L, int p, const float *qr, const float *latent,
                       int compress_len, const float *x, int *out) {
    /* the index keys live with the kv source: only a kv source compresses,
     * so the index sources after it re-derive queries against its keys */
    int src = src_slot[L];
    int rd = RD;
    int ratio = C.ratio[L];
    int64_t lp = (int64_t)(((p + 1) / ratio - 1) * ratio);

    /* append an index key when this layer's compression group completed */
    if (latent) {
        const uint16_t *wk = bf16_("layers.%d.attn.indexer.wk.weight", L);
        const uint16_t *kn = bf16_("layers.%d.attn.indexer.k_norm.weight", L);
        lin_bf16(wk, latent, IH, HD, iq, 1);
        ds_rmsnorm_bf16w(iq, iq, kn, IH, C.rms_eps);
        ds_rope(iq + IH - rd, fr_for(L), lp, 0);
        for (int i = 0; i < rd; i++) iq[IH - rd + i] = ds_bf16_round(iq[IH - rd + i]);
        ds_qdq_fp4_e8m0(iq, IH);
        int j = (int)(lp / ratio);
        memcpy(idxk + ((size_t)src * max_cmp + j) * IH, iq, sizeof(float) * (size_t)IH);
    }

    /* the query side: a projection of qr, rotated at the query's position.
     * The index key has already been copied out of `iq`, so it is free to
     * hold the 32x128 indexer query -- the attention's own `q` must not be
     * touched, it is still needed. */
    const uint8_t *wq, *sq;
    int r, c, d;
    char base[128];
    snprintf(base, sizeof(base), "layers.%d.attn.indexer.wq_b", L);
    qw(base, &wq, &sq, &r, &c, &d);
    lin_q(wq, sq, r, c, 0, qr, iq);
    for (int h = 0; h < NH; h++) {
        float *qh = iq + (size_t)h * IH;
        ds_rope(qh + IH - rd, fr_for(L), p, 0);
        for (int i = 0; i < rd; i++) qh[IH - rd + i] = ds_bf16_round(qh[IH - rd + i]);
    }
    ds_qdq_fp4_e8m0(iq, NH * IH);

    const uint16_t *wp = bf16_("layers.%d.attn.indexer.weights_proj.weight", L);
    lin_bf16(wp, x, NH, D, iw, 1);
    float wscale = powf((float)IH, -0.5f) * powf((float)NH, -0.5f);

    /* one score per compressed position: relu'd per-head dots, combined by
     * the per-head weights */
    for (int t = 0; t < compress_len; t++) {
        const float *kt = idxk + ((size_t)src * max_cmp + t) * IH;
        float acc = 0.0f;
        for (int h = 0; h < NH; h++) {
            const float *qh = iq + (size_t)h * IH;
            float dot = 0.0f;
            for (int i = 0; i < IH; i++) dot += qh[i] * kt[i];
            if (dot > 0.0f) acc += dot * (iw[h] * wscale);
        }
        scores[t] = acc;
    }

    if (is_cand_src[L])
        select_candidate_blocks(scores, compress_len, sh_cand + (size_t)p * max_cmp);
    else if (uses_cand[L]) {
        const uint8_t *cand = sh_cand + (size_t)p * max_cmp;
        for (int t = 0; t < compress_len; t++)
            if (!cand[t]) scores[t] = -INFINITY;
    }

    /* top-k by score, re-sorted into position order; unreachable -> -1 */
    int k = C.idx_topk < compress_len ? C.idx_topk : compress_len;
    int n = ds_topk_scores(scores, compress_len, k, out, heap, heap_idx);
    for (int i = 0; i < n; i++)
        out[i] = out[i] < compress_len ? out[i] + W : -1;
    return n;
}

/* ------------------------------------------------------------------ */
/* attention                                                            */
/* ------------------------------------------------------------------ */

static void attention(int L, int p, const float *x, float *out,
                      float *lat_out, int *idxtop_out, int *n_idx) {
    char base[128];
    const uint8_t *w, *s;
    int r, c, d;
    int ratio = C.ratio[L];
    *n_idx = 0;
    if (lat_out) memset(lat_out, 0, sizeof(float) * (size_t)HD);

    /* ---- q: a low-rank projection, rotated at the query's position ---- */
    snprintf(base, sizeof(base), "layers.%d.attn.wq_a", L);
    qw(base, &w, &s, &r, &c, &d);
    lin_q(w, s, r, c, 0, x, qr);
    const uint16_t *qn = bf16_("layers.%d.attn.q_norm.weight", L);
    ds_rmsnorm_bf16w(qr, qr, qn, QLR, C.rms_eps);
    pcmp("PROBE_QR", L, p, qr, QLR);

    snprintf(base, sizeof(base), "layers.%d.attn.wq_b", L);
    qw(base, &w, &s, &r, &c, &d);
    lin_q(w, s, r, c, 0, qr, q);
    for (int h = 0; h < H; h++)
        ds_rope(q + (size_t)h * HD + HD - RD, fr_for(L), p, 0);
    for (int h = 0; h < H; h++)
        for (int i = 0; i < RD; i++)
            q[(size_t)h * HD + HD - RD + i] =
                ds_bf16_round(q[(size_t)h * HD + HD - RD + i]);
    pcmp("PROBE_Q", L, p, q, H * HD);

    /* ---- this layer's window K, into the ring buffer ---- */
    snprintf(base, sizeof(base), "layers.%d.attn.wkv", L);
    qw(base, &w, &s, &r, &c, &d);
    lin_q(w, s, r, c, 0, x, kv);
    const uint16_t *kn = bf16_("layers.%d.attn.kv_norm.weight", L);
    ds_rmsnorm_bf16w(kv, kv, kn, HD, C.rms_eps);
    ds_rope(kv + HD - RD, fr_for(L), p, 0);
    for (int i = 0; i < RD; i++) kv[HD - RD + i] = ds_bf16_round(kv[HD - RD + i]);
    ds_qdq_fp8_e8m0(kv, HD);
    pcmp("PROBE_WKV", L, p, kv, HD);
    memcpy(winkv + ((size_t)L * W + (p % W)) * HD, kv, sizeof(float) * (size_t)HD);

    /* window positions, oldest first; -1 marks a slot holding nothing */
    int oldest = p % W + 1;
    for (int i = 0; i < W; i++) {
        int t = i < W - oldest ? oldest + i : i - (W - oldest);
        idxs[i] = t > p ? -1 : t;
    }

    int total = W;
    const float *cmp = NULL;

    if (ratio > 0) {
        int compress_len = (p + 1) / ratio;
        int have = 0, jj = 0;
        int64_t lp = 0;
        if (is_kv_src[L]) have = compress(L, p, x, latent, &jj, &lp);
        if (have && lat_out) memcpy(lat_out, latent, sizeof(float) * (size_t)HD);

        if (compress_len > 0) {
            if (is_idx_src[L])
                sh_n[p] = run_indexer(L, p, qr, have ? latent : NULL, compress_len,
                                      x, sh_topk + (size_t)p * C.idx_topk);
            const int *mine = sh_topk + (size_t)p * C.idx_topk;
            memcpy(idxs + W, mine, sizeof(int) * (size_t)sh_n[p]);
            *n_idx = sh_n[p];
            if (idxtop_out) memcpy(idxtop_out, idxs + W, sizeof(int) * (size_t)sh_n[p]);
        }

        if (have) {
            /* a latent stands for the first token of its group, so group j
             * takes position j * ratio */
            ds_rope(latent + HD - RD, fr_for(L), lp, 0);
            for (int i = 0; i < RD; i++)
                latent[HD - RD + i] = ds_bf16_round(latent[HD - RD + i]);
            ds_qdq_fp4_e4m3(latent, HD, 16);
            int src = src_slot[L];
            memcpy(cmpkv + ((size_t)src * max_cmp + jj) * HD, latent,
                   sizeof(float) * (size_t)HD);
            pcmp("PROBE_CKV", L, jj, latent, HD);
        }
        cmp = cmpkv + (size_t)src_slot[L] * max_cmp * HD;
        total = W + *n_idx;
    }
    if (probe_want) {
        const TEnt *t = tfind("PROBE_IDS", L, p);
        if (t) {
            /* the two implementations agree on the reference's freedom to
             * order the live window slots differently, so compare the multiset */
            int n = (int)t->n;
            int a[4096], b[4096];
            if (n > 4096) n = 4096;
            for (int i = 0; i < n; i++) {
                a[i] = (int)t->iv[i];
                b[i] = idxs[i];
            }
            for (int i = 1; i < n; i++) {          /* insertion sort, n <= 640 */
                int v = a[i], j = i - 1;
                while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; j--; }
                a[j + 1] = v;
                v = b[i]; j = i - 1;
                while (j >= 0 && b[j] > v) { b[j + 1] = b[j]; j--; }
                b[j + 1] = v;
            }
            int bad = -1;
            for (int i = 0; i < n; i++)
                if (a[i] != b[i]) { bad = i; break; }
            n_checks++;
            if (bad >= 0) {
                n_fail++;
                printf("  FAIL %-18s L=%-3d p=%-3d sorted idx %d: engine %d vs oracle %d\n",
                       "PROBE_IDS", L, p, bad, a[bad], b[bad]);
            }
        }
    }

    /* ---- sparse attention over the concatenated KV ---- */
    const float *sink = f32_("layers.%d.attn.attn_sink", L);
    pcmp("PROBE_SINK", L, p, sink, H);
    float scale = powf((float)HD, -0.5f);
    for (int h = 0; h < H; h++) {
        const float *qh = q + (size_t)h * HD;
        float *oh = o + (size_t)h * HD;
        float mx = -1e30f, sum = 0.0f;
        for (int j = 0; j < total; j++) {
            int t = idxs[j];
            if (t < 0) {
                scores[j] = -INFINITY;
                continue;
            }
            const float *kt = t < W ? winkv + ((size_t)L * W + t) * HD
                                    : cmp + (size_t)(t - W) * HD;
            float dot = 0.0f;
            for (int i = 0; i < HD; i++) dot += qh[i] * kt[i];
            dot *= scale;
            scores[j] = dot;
            if (dot > mx) mx = dot;
        }
        for (int j = 0; j < total; j++) {
            if (idxs[j] < 0) {
                scores[j] = 0.0f;
                continue;
            }
            scores[j] = expf(scores[j] - mx);
            sum += scores[j];
        }
        /* the sink is an extra virtual key with a fixed score: it widens the
         * denominator but contributes nothing to the numerator */
        sum += expf(sink[h] - mx);
        for (int i = 0; i < HD; i++) oh[i] = 0.0f;
        for (int j = 0; j < total; j++) {
            int t = idxs[j];
            if (t < 0) continue;
            const float *kt = t < W ? winkv + ((size_t)L * W + t) * HD
                                    : cmp + (size_t)(t - W) * HD;
            float wgt = scores[j];
            for (int i = 0; i < HD; i++) oh[i] += wgt * kt[i];
        }
        float inv = 1.0f / sum;
        for (int i = 0; i < HD; i++) oh[i] *= inv;
        if (h == 0 && probe_want) pcmp("PROBE_ATTS", L, p, &sum, 1);
    }

    /* sparse_attn's result is a bf16 tensor: round every head, then undo the
     * query's rotation so the cache stays in one shared form */
    for (int i = 0; i < H * HD; i++) o[i] = ds_bf16_round(o[i]);
    pcmp("PROBE_OV", L, p, o, H * HD);
    for (int h = 0; h < H; h++)
        ds_rope(o + (size_t)h * HD + HD - RD, fr_for(L), p, 1);
    for (int h = 0; h < H; h++)
        for (int i = 0; i < RD; i++)
            o[(size_t)h * HD + HD - RD + i] =
                ds_bf16_round(o[(size_t)h * HD + HD - RD + i]);
    pcmp("PROBE_O", L, p, o, H * HD);

    /* ---- the grouped LoRA output projection ---- */
    int K = H * HD / NGRP;
    for (int b = 0; b < NGRP * K; b += 32)
        act_s[b / 32] = ds_act_quant_block(o + b, act_q + b);
    snprintf(base, sizeof(base), "layers.%d.attn.wo_a", L);
    qw(base, &w, &s, &r, &c, &d);
    int nsc = c / 32;
    for (int g = 0; g < NGRP; g++)
        ds_fp8_gemv(act_q + (size_t)g * K, act_s + (size_t)g * (K / 32),
                    w + (size_t)g * OLORA * c, s + (size_t)g * (OLORA / 32) * nsc,
                    OLORA, c, o_g + (size_t)g * OLORA);
    for (int i = 0; i < NGRP * OLORA; i++) o_g[i] = ds_bf16_round(o_g[i]);
    pcmp("PROBE_OG", L, p, o_g, NGRP * OLORA);

    snprintf(base, sizeof(base), "layers.%d.attn.wo_b", L);
    qw(base, &w, &s, &r, &c, &d);
    lin_q(w, s, r, c, 0, o_g, out);
}

/* ------------------------------------------------------------------ */
/* MoE                                                                  */
/* ------------------------------------------------------------------ */

static void moe(int L, int p, const float *x, float *out) {
    const uint16_t *gw = bf16_("layers.%d.ffn.gate.weight", L);
    const float *gb = f32_("layers.%d.ffn.gate.bias", L);
    ds_bf16_gemv(gw, x, NULL, NEXP, D, router);

    /* sqrtsoftplus; the correction bias then picks experts but does not
     * scale them, the weights come from the raw scores */
    for (int e = 0; e < NEXP; e++) {
        float v = router[e] / C.gate_temp;
        router[e] = sqrtf(v > 20.0f ? v : log1pf(expf(v)));
    }
    for (int i = 0; i < TOPK; i++) {
        int best = 0;
        float bv = -INFINITY;
        for (int e = 0; e < NEXP; e++) {
            float v = router[e] + gb[e];
            if (v > bv) {
                bv = v;
                best = e;
            }
        }
        sel[i] = best;
        selw[i] = router[best];
        router[best] = -INFINITY; /* taken */
    }
    if (C.norm_topk && TOPK > 1) {
        float sum = 0.0f;
        for (int i = 0; i < TOPK; i++) sum += selw[i];
        for (int i = 0; i < TOPK; i++) selw[i] /= (sum + 1e-20f);
    }
    for (int i = 0; i < TOPK; i++) selw[i] *= C.route_scale;
    if (probe_want) {
        const TEnt *t = tfind("PROBE_SEL", L, p);
        if (t) cmpi("PROBE_SEL", L, p, t->iv, sel, TOPK);
        t = tfind("PROBE_SELW", L, p);
        if (t) pcmp("PROBE_SELW", L, p, selw, TOPK);
    }

    for (int i = 0; i < D; i++) out[i] = 0.0f;
    char base[128];
    const uint8_t *w, *s;
    int r, c, d;
    /* the routed experts first, then the shared one every token goes
     * through. The reference accumulates in fp32 and rounds once at the
     * end, so `out` stays fp32 until the final loop below. */
    for (int shared = 0; shared < 2; shared++) {
        int hi = shared ? 1 : TOPK;
        for (int k = 0; k < hi; k++) {
            if (shared) {
                snprintf(base, sizeof(base), "layers.%d.ffn.shared_experts.w1", L);
            } else {
                snprintf(base, sizeof(base), "layers.%d.ffn.experts.%d.w1", L, sel[k]);
            }
            qw(base, &w, &s, &r, &c, &d);
            lin_q(w, s, r, c, shared ? 0 : 1, x, ex_g);
            if (shared) {
                snprintf(base, sizeof(base), "layers.%d.ffn.shared_experts.w3", L);
            } else {
                snprintf(base, sizeof(base), "layers.%d.ffn.experts.%d.w3", L, sel[k]);
            }
            qw(base, &w, &s, &r, &c, &d);
            lin_q(w, s, r, c, shared ? 0 : 1, x, ex_u);
            if (!shared) {
                char tg[24];
                snprintf(tg, sizeof(tg), "PROBE_G%d", k);
                pcmp(tg, L, p, ex_g, INTER);
                snprintf(tg, sizeof(tg), "PROBE_U%d", k);
                pcmp(tg, L, p, ex_u, INTER);
            }

            /* the clamps come from training: the up branch is clamped on
             * both sides, the gate branch only from above */
            for (int i = 0; i < INTER; i++) {
                float g = ex_g[i], u = ex_u[i];
                if (C.swiglu_limit > 0) {
                    if (u > C.swiglu_limit) u = C.swiglu_limit;
                    if (u < -C.swiglu_limit) u = -C.swiglu_limit;
                    if (g > C.swiglu_limit) g = C.swiglu_limit;
                }
                float wgt = shared ? 1.0f : selw[k];
                /* fp32 throughout, rounded once on the way into w2 */
                float hv = (g / (1.0f + expf(-g)) * u) * wgt;
                ex_p[i] = hv;
                ex_h[i] = ds_bf16_round(hv);
            }
            if (!shared) {
                char tg[24];
                snprintf(tg, sizeof(tg), "PROBE_EXP%d", k);
                pcmp(tg, L, p, ex_p, INTER);
                snprintf(tg, sizeof(tg), "PROBE_EXH%d", k);
                pcmp(tg, L, p, ex_h, INTER);
            }
            if (shared) {
                snprintf(base, sizeof(base), "layers.%d.ffn.shared_experts.w2", L);
            } else {
                snprintf(base, sizeof(base), "layers.%d.ffn.experts.%d.w2", L, sel[k]);
            }
            qw(base, &w, &s, &r, &c, &d);
            lin_q(w, s, r, c, shared ? 0 : 1, ex_h, moe_y);
            for (int i = 0; i < D; i++) out[i] += moe_y[i];
            if (shared) pcmp("PROBE_SHW", L, p, moe_y, D);
            else {
                char tg[24];
                snprintf(tg, sizeof(tg), "PROBE_EY%d", k);
                pcmp(tg, L, p, moe_y, D);
                if (k == TOPK - 1) pcmp("PROBE_EX", L, p, out, D);
            }
        }
    }
    for (int i = 0; i < D; i++) out[i] = ds_bf16_round(out[i]);
}

/* ------------------------------------------------------------------ */
/* driver                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s <checkpoint-root> <max-seq-len> <ds_run-path> "
                "[--layers 0,1,2] [--tokens 1000 2000 3000] [--moe 2:0] [--all]\n",
                argv[0]);
        return 2;
    }
    const char *root = argv[1];
    int seq = atoi(argv[2]);
    const char *exe = argv[3];

    int layers[MAXL], nlayers = 0;
    int toks[MAXTOK], ntok = 0;
    int all_layers = 0;

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            char *tmp = strdup(argv[++i]);
            for (char *t = strtok(tmp, ","); t; t = strtok(NULL, ","))
                if (nlayers < MAXL) layers[nlayers++] = atoi(t);
            free(tmp);
        } else if (strcmp(argv[i], "--tokens") == 0) {
            while (i + 1 < argc && argv[i + 1][0] != '-' && ntok < MAXTOK)
                toks[ntok++] = atoi(argv[++i]);
            if (ntok == 0) die("--tokens needs at least one id");
        } else if (strcmp(argv[i], "--all") == 0) {
            all_layers = 1;
        }
    }
    if (ntok == 0) {
        toks[ntok++] = 1000;
        toks[ntok++] = 2000;
        toks[ntok++] = 3000;
    }
    if (nlayers == 0) {
        layers[nlayers++] = 0;
        layers[nlayers++] = 1;
        layers[nlayers++] = 2;
    }

    load_config(root, &C);
    if (all_layers) {
        nlayers = 0;
        for (int L = 0; L < C.n_layers && nlayers < MAXL; L++)
            layers[nlayers++] = L;
    }
    int maxl = 0;
    for (int i = 0; i < nlayers; i++)
        if (layers[i] > maxl) maxl = layers[i];

    load_config(root, &C);
    D = C.dim;
    HC = C.hc;
    H = C.n_heads;
    HD = C.head_dim;
    RD = C.rope_dim;
    W = C.window;
    IH = C.idx_hd;
    NH = C.idx_heads;
    NGRP = C.o_groups;
    OLORA = C.o_lora;
    QLR = C.q_lora;
    INTER = C.moe_inter;
    NEXP = C.n_experts;
    TOPK = C.topk;

    int min_ratio = 1 << 20;
    for (int L = 0; L < C.n_layers; L++)
        if (C.ratio[L] > 0 && C.ratio[L] < min_ratio) min_ratio = C.ratio[L];
    max_cmp = seq / min_ratio + 2;

    cwick_error err = {0};
    if (cwick_safetensors_open(root, &S, &err) != CWICK_OK) {
        fprintf(stderr, "cannot open the checkpoint: %s\n", err.message);
        return 1;
    }

    ds_freqs_init(&fr_yarn, RD, C.crope_theta, C.yarn_factor, C.yarn_orig,
                  C.beta_fast, C.beta_slow);
    ds_freqs_init(&fr_plain, RD, C.rope_theta, 1.0, 0, C.beta_fast, C.beta_slow);

    /* ---- caches ---- */
    STRM = xmalloc(sizeof(float) * (size_t)ntok * HC * D);
    PRE = xmalloc(sizeof(float) * (size_t)ntok * HC);
    RES = xmalloc(sizeof(float) * (size_t)HC * D);
    winkv = xmalloc(sizeof(float) * (size_t)C.n_layers * W * HD);
    cmpkv = xmalloc(sizeof(float) * (size_t)C.n_kv_src * max_cmp * HD);
    cmpst = xmalloc(sizeof(float) * (size_t)C.n_kv_src * 8 * HD);
    cmpss = xmalloc(sizeof(float) * (size_t)C.n_kv_src * 8 * HD);
    idxk = xmalloc(sizeof(float) * (size_t)C.n_kv_src * max_cmp * IH);
    ccache = xmalloc(sizeof(int64_t) * (size_t)(seq + 1));
    NTOK = ntok;
    sh_topk = xmalloc(sizeof(int) * (size_t)ntok * C.idx_topk);
    sh_n = xmalloc(sizeof(int) * (size_t)ntok);
    sh_cand = xmalloc((size_t)ntok * max_cmp);
    memset(winkv, 0, sizeof(float) * (size_t)C.n_layers * W * HD);
    memset(cmpkv, 0, sizeof(float) * (size_t)C.n_kv_src * max_cmp * HD);
    memset(idxk, 0, sizeof(float) * (size_t)C.n_kv_src * max_cmp * IH);
    memset(cmpst, 0, sizeof(float) * (size_t)C.n_kv_src * 8 * HD);
    memset(cmpss, 0, sizeof(float) * (size_t)C.n_kv_src * 8 * HD);
    memset(sh_n, 0, sizeof(int) * (size_t)ntok);
    memset(sh_cand, 0, (size_t)ntok * max_cmp);

    /* ---- scratch, one buffer per live value ---- */
    sub = xmalloc(sizeof(float) * (size_t)D);
    subl = xmalloc(sizeof(float) * (size_t)D);
    moe_y = xmalloc(sizeof(float) * (size_t)D);
    qr = xmalloc(sizeof(float) * (size_t)QLR);
    q = xmalloc(sizeof(float) * (size_t)H * HD);
    iq = xmalloc(sizeof(float) * (size_t)(NH * IH > HD ? NH * IH : HD));
    kv = xmalloc(sizeof(float) * (size_t)HD);
    ckv = xmalloc(sizeof(float) * (size_t)HD);
    cgate = xmalloc(sizeof(float) * (size_t)HD);
    latent = xmalloc(sizeof(float) * (size_t)HD);
    o = xmalloc(sizeof(float) * (size_t)H * HD);
    o_g = xmalloc(sizeof(float) * (size_t)NGRP * OLORA);
    attn_out = xmalloc(sizeof(float) * (size_t)D);
    ex_g = xmalloc(sizeof(float) * (size_t)INTER);
    ex_u = xmalloc(sizeof(float) * (size_t)INTER);
    ex_h = xmalloc(sizeof(float) * (size_t)INTER);
    ex_p = xmalloc(sizeof(float) * (size_t)INTER);
    iw = xmalloc(sizeof(float) * (size_t)NH);
    apre = xmalloc(sizeof(float) * (size_t)HC);
    apost = xmalloc(sizeof(float) * (size_t)HC);
    fpre = xmalloc(sizeof(float) * (size_t)HC);
    fpost = xmalloc(sizeof(float) * (size_t)HC);
    comb = xmalloc(sizeof(float) * (size_t)HC * HC);
    comb2 = xmalloc(sizeof(float) * (size_t)HC * HC);
    router = xmalloc(sizeof(float) * (size_t)NEXP);
    selw = xmalloc(sizeof(float) * (size_t)TOPK);
    sel = xmalloc(sizeof(int) * (size_t)TOPK);
    idxs = xmalloc(sizeof(int) * (size_t)(W + C.idx_topk));
    scores = xmalloc(sizeof(float) * (size_t)(W + C.idx_topk + 8));
    bsc = xmalloc(sizeof(float) * (size_t)(max_cmp / C.cand_block + 4));
    heap = xmalloc(sizeof(float) * (size_t)(C.idx_topk + C.cand_blocks + 8));
    heap_idx = xmalloc(sizeof(int) * (size_t)(C.idx_topk + C.cand_blocks + 8));
    blk_out = xmalloc(sizeof(int) * (size_t)(C.cand_blocks + 8));
    {
        int big = D;
        if (INTER > big) big = INTER;
        if (NGRP * (H * HD / NGRP) > big) big = NGRP * (H * HD / NGRP);
        if (NGRP * OLORA > big) big = NGRP * OLORA;
        act_q = xmalloc((size_t)big + 64);
        act_s = xmalloc((size_t)big / 32 + 8);
    }
    eng_rows = xmalloc(sizeof(float) * (size_t)64 * C.eng_hd);
    eng_kv = xmalloc(sizeof(float) * (size_t)D * (HC + 1));
    eng_prod = xmalloc(sizeof(float) * (size_t)HC * D);
    eng_hash = xmalloc(sizeof(int64_t) * 64);
    lat = xmalloc(sizeof(float) * (size_t)HD);
    idxtop = xmalloc(sizeof(int) * (size_t)(W + C.idx_topk));

    /* ---- layer roles, resolved the way the reference resolves them ---- */
    {
        int cur = -1, curi = -1;
        for (int L = 0; L < C.n_layers; L++) {
            for (int s = 0; s < C.n_kv_src; s++)
                if (C.kv_src[s] == L) cur = s;
            for (int s = 0; s < C.n_idx_src; s++)
                if (C.idx_src[s] == L) curi = s;
            src_slot[L] = cur;
            idx_slot[L] = curi;
            is_kv_src[L] = 0;
            for (int s = 0; s < C.n_kv_src; s++)
                if (C.kv_src[s] == L) is_kv_src[L] = 1;
            is_idx_src[L] = 0;
            for (int s = 0; s < C.n_idx_src; s++)
                if (C.idx_src[s] == L) is_idx_src[L] = 1;
            is_cand_src[L] = (L == C.cand_layer);
            uses_cand[L] = (C.cand_layer >= 0 && L > C.cand_layer);
            eng_slot[L] = -1;
            for (int e = 0; e < C.n_eng; e++)
                if (C.eng_layers[e] == L) eng_slot[L] = e;
        }
    }

    for (int p = 0; p < ntok; p++) ccache[p] = ds_token_map[toks[p]];

    capture_trace(exe, root, seq, toks, ntok, maxl);

    printf("config: dim %d layers %d heads %d hd %d rope %d hc %d window %d\n",
           D, C.n_layers, H, HD, RD, HC, W);
    printf("        experts %d top-%d inter %d idx %dx%d topk %d "
           "kv-src %d idx-src %d\n",
           NEXP, TOPK, INTER, NH, IH, C.idx_topk, C.n_kv_src, C.n_idx_src);
    printf("tokens:");
    for (int i = 0; i < ntok; i++) printf(" %d", toks[i]);
    printf("\n\n");

    /* ---- the residual stream starts as hc copies of the embedding ---- */
    {
        const uint16_t *emb = bf16("embed.weight");
        for (int p = 0; p < ntok; p++) {
            const uint16_t *e = emb + (size_t)toks[p] * D;
            for (int j = 0; j < HC; j++)
                for (int i = 0; i < D; i++)
                    STRM[((size_t)p * HC + j) * D + i] = ds_bf16_to_f32(e[i]);
            /* the first block's pre_mix is the identity one-hot */
            PRE[(size_t)p * HC + 0] = 1.0f;
            for (int j = 1; j < HC; j++) PRE[(size_t)p * HC + j] = 0.0f;
        }
    }

    /* ---- engram hash, checked for every position ---- */
    if (C.n_eng > 0) {
        int ncol = (C.eng_max_ngram - 1) * C.eng_heads;
        for (int p = 0; p < ntok; p++) {
            for (int e = 0; e < C.n_eng; e++) {
                int64_t mine[64];
                engram_hash(p, e, mine);
                const TEnt *t = tfind("ENGHASH", p, e);
                if (!t) continue;
                n_checks++;
                for (int i = 0; i < ncol && i < t->n; i++)
                    if (t->iv[i] != mine[i]) {
                        n_fail++;
                        printf("  FAIL %-18s p=%-3d e=%-3d col %d: engine %lld "
                               "vs oracle %lld\n", "ENGHASH", p, e, i,
                               (long long)t->iv[i], (long long)mine[i]);
                        break;
                    }
            }
        }
    }

    /* ---- the blocks ---- */
    for (int L = 0; L <= maxl; L++) {
        int want = 0;
        for (int i = 0; i < nlayers; i++)
            if (layers[i] == L) want = 1;
        probe_want = want;

        if (eng_slot[L] >= 0)
            for (int p = 0; p < ntok; p++) engram_apply(L, p);

        for (int p = 0; p < ntok; p++) {
            const TEnt *t;
            if (want && (t = tfind("HIN", L, p)))
                cmpv("HIN", L, p, t->v, STRM + ((size_t)p * HC + 0) * D, D);
            if (want && (t = tfind("HIN1", L, p)))
                cmpv("HIN1", L, p, t->v, STRM + ((size_t)p * HC + 1) * D, D);
            if (want && (t = tfind("PREMIX", L, p)))
                cmpv("PREMIX", L, p, t->v, PRE + (size_t)p * HC, HC);

            /* ---- attention sublayer ---- */
            memcpy(RES, STRM + (size_t)p * HC * D, sizeof(float) * (size_t)HC * D);
            {
                const float *fn = f32_("layers.%d.hc_attn_fn", L);
                const float *base = f32_("layers.%d.hc_attn_base", L);
                const float *scale = f32_("layers.%d.hc_attn_scale", L);
                ds_hc_mixes(STRM + (size_t)p * HC * D, fn, scale, base, HC, D,
                            C.hc_iters, C.hc_eps, apre, apost, comb);
            }
            ds_hc_pre(sub, STRM + (size_t)p * HC * D, PRE + (size_t)p * HC, HC, D);
            for (int i = 0; i < D; i++) sub[i] = ds_bf16_round(sub[i]);
            ds_rmsnorm_bf16w(sub, sub, bf16_("layers.%d.attn_norm.weight", L),
                             D, C.rms_eps);
            if (want && (t = tfind("SUBIN", L, p)))
                cmpv("SUBIN", L, p, t->v, sub, D);

            int nidx = 0;
            attention(L, p, sub, attn_out, lat, idxtop, &nidx);
            if (want && (t = tfind("ATTNOUT", L, p)))
                cmpv("ATTNOUT", L, p, t->v, attn_out, D);
            if (want && C.ratio[L] > 0 &&
                (t = tfind("LATENT", L, p + 1 - C.ratio[L])))
                cmpv("LATENT", L, p, t->v, lat, HD);
            if (want && C.ratio[L] > 0 && (t = tfind("IDXTOP", L, p)))
                cmpi("IDXTOP", L, p, t->iv, idxtop, t->n);

            ds_hc_post(STRM + (size_t)p * HC * D, attn_out, RES, apost, comb, HC, D);
            for (int j = 0; j < HC; j++)
                for (int i = 0; i < D; i++)
                    STRM[((size_t)p * HC + j) * D + i] =
                        ds_bf16_round(STRM[((size_t)p * HC + j) * D + i]);

            /* ---- FFN sublayer ---- */
            memcpy(RES, STRM + (size_t)p * HC * D, sizeof(float) * (size_t)HC * D);
            {
                const float *fn = f32_("layers.%d.hc_ffn_fn", L);
                const float *base = f32_("layers.%d.hc_ffn_base", L);
                const float *scale = f32_("layers.%d.hc_ffn_scale", L);
                ds_hc_mixes(STRM + (size_t)p * HC * D, fn, scale, base, HC, D,
                            C.hc_iters, C.hc_eps, fpre, fpost, comb2);
            }
            /* the FFN collapses with the attention sublayer's pre_mix */
            ds_hc_pre(sub, STRM + (size_t)p * HC * D, apre, HC, D);
            for (int i = 0; i < D; i++) sub[i] = ds_bf16_round(sub[i]);
            ds_rmsnorm_bf16w(sub, sub, bf16_("layers.%d.ffn_norm.weight", L),
                             D, C.rms_eps);
            if (want && (t = tfind("FFNIN", L, p)))
                cmpv("FFNIN", L, p, t->v, sub, D);

            /* the FFN output is what the next layer's HIN is built from, so
             * it is always recomputed even when it is not being checked */
            moe(L, p, sub, subl);
            if (want && (t = tfind("FFNOUT", L, p)))
                cmpv("FFNOUT", L, p, t->v, subl, D);
            if (want)
                printf("    layer %d pos %d experts", L, p);
            for (int i = 0; i < TOPK; i++)
                if (want) printf(" %d(%.4f)", sel[i], selw[i]);
            if (want) printf("\n");

            ds_hc_post(STRM + (size_t)p * HC * D, subl, RES, fpost, comb2, HC, D);
            for (int j = 0; j < HC; j++)
                for (int i = 0; i < D; i++)
                    STRM[((size_t)p * HC + j) * D + i] =
                        ds_bf16_round(STRM[((size_t)p * HC + j) * D + i]);

            /* this layer's ffn_pre is the next layer's pre_mix */
            memcpy(PRE + (size_t)p * HC, fpre, sizeof(float) * (size_t)HC);
        }
    }

    printf("\n%d checks, %d failure(s), worst rel/rms %.3g\n",
           n_checks, n_fail, worst_rel);
    return n_fail ? 1 : 0;
}
