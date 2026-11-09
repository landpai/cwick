/* attention.c — MLA + KDA attention, decode-step forward.
 *
 *   - MLA: latent query (q_a/q_b + norm), latent KV (kv_a/kv_b + norm),
 *     RoPE on the rope slice, full attention over the KV cache, output gate.
 *   - KDA: linear attention with a 4-tap short convolution (SiLU), L2
 *     normalized query/key, gated decay (gate_lower_bound * sigmoid),
 *     beta-scaled delta rule and a per-head RMSNorm before the output gate.
 *
 * Both operate on one token at a time (fp32) and write into persistent
 * KV / KDA state owned by the engine.
 */
#include "cwick.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "util.h"


/* Optional numeric probe (CWICK_TRACE=1). */
static bool atrace_on(void) {
    static int c = -1;
    if (c < 0) c = getenv("CWICK_TRACE") != NULL ? 1 : 0;
    return c == 1;
}
/* Re-run the un-absorbed reference inside the absorbed kernel and report the
 * difference (CWICK_MLA_SELFCHECK=1). Expensive: O(nsel) full kv_b_proj
 * materialisations per layer, so it is for short prompts only. */
static bool mla_selfcheck(void) {
    static int c = -1;
    if (c < 0) c = getenv("CWICK_MLA_SELFCHECK") != NULL ? 1 : 0;
    return c == 1;
}
/* CWICK_INDEXER_TRACE=1 prints, per sparse-attention layer and position, how
 * many positions the DSA indexer kept out of the growing prefix. That ratio
 * is the whole point of the mechanism: it must stay near index_topk/npos once
 * the prefix is longer than the window, otherwise long context is just paying
 * for full attention with extra steps. */
static bool indexer_trace(void) {
    static int c = -1;
    if (c < 0) c = getenv("CWICK_INDEXER_TRACE") != NULL ? 1 : 0;
    return c == 1;
}
static void aprobe(const char *tag, const float *x, int64_t n) {
    if (!atrace_on()) return;
    double mn = INFINITY, mx = -INFINITY; int64_t nan = 0;
    for (int64_t i = 0; i < n; i++) { double v = x[i]; if (v != v) { nan++; continue; }
        if (v < mn) mn = v; if (v > mx) mx = v; }
    fprintf(stderr, "    kda %-16s n=%-7lld min=%-11.5g max=%-11.5g nan=%lld\n",
            tag, (long long)n, mn, mx, (long long)nan);
}

/* ------------------------------------------------------------------ */
/* RoPE (MLA qk_rope slice)                                            */
/* ------------------------------------------------------------------ */
static void rope_apply(float *v, int64_t n_rope, int64_t pos, float base) {
    for (int64_t i = 0; i < n_rope / 2; i++) {
        float angle = (float)pos / powf(base, 2.0f * (float)i / (float)n_rope);
        float cos_a = cosf(angle), sin_a = sinf(angle);
        float a = v[2 * i], b = v[2 * i + 1];
        v[2 * i] = a * cos_a - b * sin_a;
        v[2 * i + 1] = b * cos_a + a * sin_a;
    }
}

static void apply_bf16_vec(const cwick_tensor *t, float *x, int64_t n) {
    const uint16_t *wn = (const uint16_t *)t->data;
    for (int64_t i = 0; i < n; i++) x[i] *= cwick_bf16_fast(wn[i]);
}

/* ------------------------------------------------------------------ */
/* MLA decode                                                          */
/* ------------------------------------------------------------------ */
/* Two cache layouts, selected by the caller:
 *
 *   LATENT   (values == NULL)
 *     keys[layer][pos][0 .. kv_lora) = the normalised KV latent c_t.
 *     MLA absorption makes the expanded form unnecessary:
 *         k_t = W_k c_t          -> q·k_t = (W_k^T q)·c_t
 *         v_t = W_v c_t          -> sum_t p_t v_t = W_v (sum_t p_t c_t)
 *     Both are exact algebraic identities -- only the summation order (and
 *     therefore the last-bit rounding) changes. The cache shrinks by
 *     kv_lora / (nh*(q_dim+vdim)): for GLM-5.3 that is 512 floats per
 *     position instead of 32768, i.e. 2.05 GiB for 11 DSA layers at 100K
 *     tokens instead of 137.5 GiB. That ratio is the whole reason a 100K
 *     context window is reachable in a bounded RAM budget at all. The
 *     checkpoint's own DSA cache is the same row: 512 latent dimensions and
 *     no RoPE tail.
 *
 *   EXPANDED (values != NULL)
 *     The materialised K [nh, q_dim] / V [nh, vdim] cache. Kept as the
 *     reference path; CWICK_MLA_EXPANDED=1 forces it for A/B checks.
 */

/* Latent absorption needs a rope-free MLA: with rope the positional tail
 * cannot be folded into a static cached vector, so the optimisation is
 * simply not applied there rather than approximated. */
static bool mla_latent_ok(const cwick_model_spec *spec, const cwick_tensor *kvb) {
    if (getenv("CWICK_MLA_EXPANDED") != NULL) return false;
    const int64_t nh = spec->num_attention_heads;
    const int64_t kv_dim = spec->qk_nope_head_dim + spec->value_head_dim;
    if (kv_dim <= 0 || nh <= 0) return false;
    if (spec->qk_rope_head_dim != 0) return false;
    if (spec->kv_lora_rank <= 0) return false;
    /* kv_b_proj must be the per-query-head MLA layout [nh, kv_dim, kv_lora].
     * GQA checkpoints share one slice between heads; there is no single
     * latent to absorb into in that case. */
    return kvb->rows == nh * kv_dim && kvb->cols >= spec->kv_lora_rank;
}

bool cwick_mla_latent_enabled(const cwick_model_spec *spec,
                             const cwick_tensor *kvb) {
    return mla_latent_ok(spec, kvb);
}

bool cwick_indexer_active(const cwick_model_spec *spec, int64_t npos) {
    const int64_t topk = spec->index_topk;
    if (topk <= 0 || spec->index_head_dim <= 0 || spec->index_n_heads <= 0)
        return false;
    return npos > topk;
}

#if defined(__aarch64__) && defined(__ARM_NEON)
/* Same widening trick as tensor.c: bf16 -> fp32 is a 16-bit left shift of
 * the bit pattern, so vshll_n_u16 does four lanes per instruction. */
static inline float32x4_t attn_bf16_lo(uint16x8_t h) {
    return vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h), 16));
}
static inline float32x4_t attn_bf16_hi(uint16x8_t h) {
    return vreinterpretq_f32_u32(vshll_high_n_u16(h, 16));
}
#endif

/* out[c] = sum_r in[r] * w[r][c], w row-major [rows][cols].
 *
 * The absorbed MLA needs the *transpose* of the kv_b_proj slice: the matrix
 * is stored as [nope, kv_lora] (one row per key dimension) while the query
 * has to contract over the key dimension, not the latent one. Row-outer
 * accumulation keeps the latent-length output resident and streams each
 * weight row once, which is both cache-friendly and vectorisable. */
static void gemv_bf16_t(float *out, int64_t rows, int64_t cols,
                        const float *in, const uint16_t *w) {
    for (int64_t c = 0; c < cols; c++) out[c] = 0.0f;
#if defined(__aarch64__) && defined(__ARM_NEON)
    if (cols >= 8) {
        const int64_t vlen = cols & ~(int64_t)7;
        for (int64_t r = 0; r < rows; r++) {
            const float a = in[r];
            if (a == 0.0f) continue;
            const uint16_t *wr = w + (size_t)r * (size_t)cols;
            const float32x4_t av = vdupq_n_f32(a);
            for (int64_t c = 0; c < vlen; c += 8) {
                uint16x8_t h = vld1q_u16(wr + c);
                float32x4_t o0 = vld1q_f32(out + c);
                float32x4_t o1 = vld1q_f32(out + c + 4);
                o0 = vfmaq_f32(o0, attn_bf16_lo(h), av);
                o1 = vfmaq_f32(o1, attn_bf16_hi(h), av);
                vst1q_f32(out + c, o0);
                vst1q_f32(out + c + 4, o1);
            }
            for (int64_t c = vlen; c < cols; c++)
                out[c] += a * cwick_bf16_fast(wr[c]);
        }
        return;
    }
#endif
    for (int64_t r = 0; r < rows; r++) {
        const float a = in[r];
        if (a == 0.0f) continue;
        const uint16_t *wr = w + (size_t)r * (size_t)cols;
        for (int64_t c = 0; c < cols; c++) out[c] += a * cwick_bf16_fast(wr[c]);
    }
}

/* out[r] = dot(w[r], in, cols), w row-major [rows][cols].
 *
 * Deliberately not cwick_gemv_bf16: that one dispatches to the thread pool,
 * and this runs once per query head (64 per layer, 704 per decode step) --
 * more dispatches than there is work to spread. */
static void gemv_bf16_n(float *out, int64_t rows, int64_t cols,
                        const float *in, const uint16_t *w) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    if (cols >= 16) {
        const int64_t vlen = cols & ~(int64_t)15;
        for (int64_t r = 0; r < rows; r++) {
            const uint16_t *wr = w + (size_t)r * (size_t)cols;
            float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
            float32x4_t a2 = vdupq_n_f32(0.f), a3 = vdupq_n_f32(0.f);
            int64_t j = 0;
            for (; j < vlen; j += 16) {
                uint16x8_t h0 = vld1q_u16(wr + j);
                uint16x8_t h1 = vld1q_u16(wr + j + 8);
                a0 = vfmaq_f32(a0, attn_bf16_lo(h0), vld1q_f32(in + j + 0));
                a1 = vfmaq_f32(a1, attn_bf16_hi(h0), vld1q_f32(in + j + 4));
                a2 = vfmaq_f32(a2, attn_bf16_lo(h1), vld1q_f32(in + j + 8));
                a3 = vfmaq_f32(a3, attn_bf16_hi(h1), vld1q_f32(in + j + 12));
            }
            float32x4_t s = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
            float acc = vgetq_lane_f32(s, 0) + vgetq_lane_f32(s, 1) +
                        vgetq_lane_f32(s, 2) + vgetq_lane_f32(s, 3);
            for (; j < cols; j++) acc += cwick_bf16_fast(wr[j]) * in[j];
            out[r] = acc;
        }
        return;
    }
#endif
    for (int64_t r = 0; r < rows; r++) {
        const uint16_t *wr = w + (size_t)r * (size_t)cols;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        int64_t j = 0;
        for (; j + 4 <= cols; j += 4) {
            s0 += cwick_bf16_fast(wr[j + 0]) * in[j + 0];
            s1 += cwick_bf16_fast(wr[j + 1]) * in[j + 1];
            s2 += cwick_bf16_fast(wr[j + 2]) * in[j + 2];
            s3 += cwick_bf16_fast(wr[j + 3]) * in[j + 3];
        }
        for (; j < cols; j++) s0 += cwick_bf16_fast(wr[j]) * in[j];
        out[r] = (s0 + s1) + (s2 + s3);
    }
}

/* The single hottest reduction in the model: every attention score and every
 * indexer score is one of these, and a 30K-token prefill does ~10^11 of them.
 * NEON with the same four-lane grouping the scalar version used, so the
 * result is bit-identical to the old code and only the speed changes. */
static float dot_f32(const float *a, const float *b, int64_t n) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        a0 = vfmaq_f32(a0, vld1q_f32(a + i + 0), vld1q_f32(b + i + 0));
        a1 = vfmaq_f32(a1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    float32x4_t s = vaddq_f32(a0, a1);
    float r = vgetq_lane_f32(s, 0) + vgetq_lane_f32(s, 1) +
              vgetq_lane_f32(s, 2) + vgetq_lane_f32(s, 3);
    for (; i + 4 <= n; i += 4) {
        float32x4_t t = vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        r += vgetq_lane_f32(t, 0) + vgetq_lane_f32(t, 1) +
             vgetq_lane_f32(t, 2) + vgetq_lane_f32(t, 3);
    }
    for (; i < n; i++) r += a[i] * b[i];
    return r;
#else
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i + 0] * b[i + 0];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for (; i < n; i++) s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
#endif
}

/* ------------------------------------------------------------------ */
/* Latent cache precision                                              */
/* ------------------------------------------------------------------ */
/* The MLA latent is read twice per attention step (score pass, accumulation
 * pass), so halving its width looks like free bandwidth. It is not, and this
 * switch is OFF by default because of it -- measured on the released
 * checkpoint, 35-token prompt, 32 decode steps:
 *
 *   fp32  876.3 ms/step
 *   bf16  925.7 ms/step      (+5.6%, slower)
 *
 * The reason is that the latent was never the bandwidth problem: a step reads
 * ~92 MB of latent in ~139 ms of attention, about 660 MB/s against a memory
 * system two orders of magnitude faster than that. Halving 660 MB/s buys
 * nothing measurable, while every load now pays a bf16->fp32 conversion on
 * top of the arithmetic that was already the cost.
 *
 * Scenarios with a much longer window (nlist at the 2048 cap rather than 35)
 * shift that arithmetic, so the path stays available under CWICK_LATENT_BF16=1
 * for someone who wants to re-measure there. Do not flip the default back
 * without numbers that beat the ones above.

 * The value that gets rounded is the latent vector the checkpoint produced;
 * the query side stays fp32, so the error is one rounding of the KV rather
 * than a compounding one. Output quality held up in the A/B: the two runs
 * produced the same text apart from a couple of tokens. */
static int g_latent_bf16 = -1;

bool cwick_latent_bf16_enabled(void) {
    if (g_latent_bf16 < 0) {
        const char *e = getenv("CWICK_LATENT_BF16");
        g_latent_bf16 = (e && (e[0] == '1' || strcmp(e, "on") == 0)) ? 1 : 0;
    }
    return g_latent_bf16 == 1;
}

size_t cwick_latent_elem_bytes(void) {
    return cwick_latent_bf16_enabled() ? sizeof(uint16_t) : sizeof(float);
}

/* out[j] = sum_j a[j] * bf16(b[j]), with the same four-lane grouping as
 * dot_f32 so the two precisions stay directly comparable in an A/B. */
static float dot_f32_bf16(const float *a, const uint16_t *b, int64_t n) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t h = vld1q_u16(b + i);
        a0 = vfmaq_f32(a0, vld1q_f32(a + i + 0), attn_bf16_lo(h));
        a1 = vfmaq_f32(a1, vld1q_f32(a + i + 4), attn_bf16_hi(h));
    }
    float32x4_t s = vaddq_f32(a0, a1);
    float r = vgetq_lane_f32(s, 0) + vgetq_lane_f32(s, 1) +
              vgetq_lane_f32(s, 2) + vgetq_lane_f32(s, 3);
    for (; i + 4 <= n; i += 4) {
        float32x4_t t = vmulq_f32(vld1q_f32(a + i),
                                  attn_bf16_lo(vld1q_u16(b + i)));
        r += vgetq_lane_f32(t, 0) + vgetq_lane_f32(t, 1) +
             vgetq_lane_f32(t, 2) + vgetq_lane_f32(t, 3);
    }
    for (; i < n; i++) r += a[i] * cwick_bf16_fast(b[i]);
    return r;
#else
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i + 0] * cwick_bf16_fast(b[i + 0]);
        s1 += a[i + 1] * cwick_bf16_fast(b[i + 1]);
        s2 += a[i + 2] * cwick_bf16_fast(b[i + 2]);
        s3 += a[i + 3] * cwick_bf16_fast(b[i + 3]);
    }
    for (; i < n; i++) s0 += a[i] * cwick_bf16_fast(b[i]);
    return (s0 + s1) + (s2 + s3);
#endif
}

/* ------------------------------------------------------------------ */
/* Parallel attention reductions                                       */
/* ------------------------------------------------------------------ */
/* The two big reductions in MLA attention are nlist x nh x kv_lora MACs
 * each -- 2 x 2051 x 64 x 512 = 134 M MAC per token on the released
 * checkpoint, i.e. ~9 TFLOP over a 30K-token prompt per layer. Until now
 * they ran on one core while the other eleven sat idle, which is why a long
 * prefill took tens of minutes at 90% machine idle. Both are
 * embarrassingly parallel over independent outputs, so they go to the
 * worker pool.
 *
 * Accumulation order per output element is kept exactly as it was (each
 * head's c_acc still sums i = 0..nlist in order), so these produce
 * bit-identical results to the serial loops they replace.
 *
 * None of these may be called from inside a pool worker: the pool's
 * fork-join is not re-entrant. Every caller is on a request thread. */

/* A/B knob for the parallel reductions, and it is OFF by default.
 *
 * Measured on the real checkpoint, 3037-token prompt, --backend metal:
 *   serial reductions   879 s end to end   (19.0 s/layer over layers 29..45)
 *   parallel, (h,i) split 1091 s           (23.5 s/layer)
 *   parallel, position split  ~11% slower than serial at the same layers
 *
 * The scores loop is one pass over the selected latent rows and reuse each
 * row for all 64 heads; splitting it twelve ways does not add memory
 * bandwidth (that is shared), and the fork-join costs more than the arithmetic
 * it spreads. So the parallelism is kept — it is the right shape for a prompt
 * long enough that the kernel dominates — but it stays behind
 * CWICK_ATTN_PARALLEL=1 until it measures faster at some length. */
static int g_attn_parallel = -1;
static bool attn_parallel(void) {
    if (g_attn_parallel < 0)
        g_attn_parallel = cwick_env_true("CWICK_ATTN_PARALLEL") ? 1 : 0;
    return g_attn_parallel == 1;
}

typedef struct {
    const float *pool;    /* n_comp * hd compressed indexer rows */
    const float *q;       /* ih * hd */
    const float *wt;      /* ih */
    float *sc;            /* n_comp */
    int64_t n_comp, hd, ih;
} iscore_arg;

static void iscore_range(void *argp, int64_t c0, int64_t c1) {
    iscore_arg *a = (iscore_arg *)argp;
    if (a->n_comp <= 0) return;
    for (int64_t c = c0; c < c1; c++) {
        const float *kv = a->pool + (size_t)c * (size_t)a->hd;
        float s = 0.0f;
        for (int64_t h = 0; h < a->ih; h++) {
            float d = dot_f32(kv, a->q + h * a->hd, a->hd);
            if (d < 0.0f) d = 0.0f;               /* ReLU, per indexer head */
            s += d * a->wt[h];
        }
        a->sc[c] = s;
    }
}

/* Split over *positions*, never over heads.
 *
 * The serial loop is deliberately position-outer: one latent row c[i] is
 * loaded once and reused for all 64 heads. Splitting (h, i) instead -- the
 * obvious parallel decomposition -- makes every worker re-read every row for
 * its own heads, and on a 3037-token prompt that measured *slower* than the
 * serial loop (1091 s vs 879 s end to end): the extra traffic through the
 * latent array costs more than twelve cores save. Keeping the position-outer
 * order per slice preserves the streaming and still spreads the work. */
typedef struct {
    const float *q_abs;    /* nh * kv_lora */
    const void *lbase;     /* capacity * lat_dim; fp32 or bf16 per `bf16` */
    const int64_t *sel;    /* nlist entries, or NULL for "every position" */
    float *scores;         /* nh * nlist */
    int64_t nlist, nh, kv_lora, lat_dim;
    float scale;
    bool bf16;
} scores_arg;

static void scores_range(void *argp, int64_t i0, int64_t i1) {
    scores_arg *a = (scores_arg *)argp;
    const int64_t nlist = a->nlist;
    if (nlist <= 0 || a->nh <= 0) return;
    const float *lf = (const float *)a->lbase;
    const uint16_t *lb = (const uint16_t *)a->lbase;
    for (int64_t i = i0; i < i1; i++) {
        const size_t t = (size_t)(a->sel ? a->sel[i] : i) * (size_t)a->lat_dim;
        for (int64_t h = 0; h < a->nh; h++)
            a->scores[(size_t)h * (size_t)nlist + (size_t)i] =
                (a->bf16 ? dot_f32_bf16(a->q_abs + h * a->kv_lora, lb + t,
                                        a->kv_lora)
                         : dot_f32(a->q_abs + h * a->kv_lora, lf + t,
                                   a->kv_lora)) * a->scale;
    }
}

/* ------------------------------------------------------------------ */
/* DSA indexer                                                         */
/* ------------------------------------------------------------------ */
/* `index_kpool` consecutive positions are compressed into one 128-dim row:
 * each raw key row is layer-normalised, then a softmax over the pool -- whose
 * logits are a learned gate plus a per-pool-slot absolute-position bias
 * (ape) -- weights them, and the affine k_norm is applied on the way out.
 * Both halves of k_norm (weight and bias) are consumed: the checkpoint ships
 * the bias, so dropping it would silently ignore a parameter the model was
 * trained with. */

uint64_t cwick_indexer_bytes(const cwick_model_spec *spec, int64_t capacity) {
    const int64_t hd = spec->index_head_dim;
    const int64_t kp = spec->index_kpool > 0 ? spec->index_kpool : 1;
    const int64_t topk = spec->index_topk > 0 ? spec->index_topk : 0;
    if (hd <= 0 || spec->index_n_heads <= 0) return 0;
    const int64_t pools = capacity / kp + 2;
    const int64_t sel = topk + kp + 2;              /* + tail + self */
    return (uint64_t)((pools * hd) + (2 * kp * hd) + sel +
                      (spec->index_n_heads * hd) + spec->index_n_heads +
                      2 * pools) * sizeof(float);
}

void cwick_indexer_reset(const cwick_model_spec *spec, cwick_indexer_state *st) {
    if (!st) return;
    const int64_t hd = spec->index_head_dim;
    const int64_t kp = spec->index_kpool > 0 ? spec->index_kpool : 1;
    if (hd > 0 && st->pool) {
        memset(st->pool, 0, (size_t)st->pool_cap * (size_t)hd * sizeof(float));
        memset(st->tail_k, 0, (size_t)kp * (size_t)hd * sizeof(float));
        memset(st->tail_g, 0, (size_t)kp * (size_t)hd * sizeof(float));
    }
    st->n_pools = 0;
}

/* Project this position's raw indexer key/gate and, when the pool closes,
 * emit one compressed row. Call once per position, in increasing order. */
static void indexer_absorb(const cwick_model_spec *spec,
                           const cwick_indexer_weights *w,
                           cwick_indexer_state *st,
                           const float *x, int64_t pos) {
    const int64_t hd = spec->index_head_dim;
    const int64_t kp = spec->index_kpool > 0 ? spec->index_kpool : 1;
    if (!w || !st || hd <= 0) return;
    const int64_t r = pos % kp;

    cwick_linear(st->tail_k + r * hd, x, 1, spec->hidden_size, hd, w->wk);
    cwick_linear(st->tail_g + r * hd, x, 1, spec->hidden_size, hd, w->gate);
    if (r != kp - 1) return;           /* pool still open */

    if (st->n_pools >= st->pool_cap) return;
    float *dst = st->pool + (size_t)st->n_pools * (size_t)hd;
    const float eps = spec->rms_norm_eps_f;
    const uint16_t *ape = (const uint16_t *)w->ape->data;
    float mean[8], inv[8];
    if (kp > 8) return;                /* kpool is 4 in the released config */

    for (int64_t i = 0; i < kp; i++) {
        const float *row = st->tail_k + i * hd;
        double s = 0.0;
        for (int64_t d = 0; d < hd; d++) s += (double)row[d];
        mean[i] = (float)(s / (double)hd);
        double v = 0.0;
        for (int64_t d = 0; d < hd; d++) {
            double t = (double)row[d] - (double)mean[i];
            v += t * t;
        }
        inv[i] = 1.0f / sqrtf((float)(v / (double)hd) + eps);
    }

    for (int64_t d = 0; d < hd; d++) {
        float mx = -INFINITY;
        for (int64_t i = 0; i < kp; i++) {
            float lg = st->tail_g[i * hd + d] +
                       cwick_bf16_fast(ape[(size_t)i * (size_t)hd + (size_t)d]);
            if (lg > mx) mx = lg;
        }
        float den = 0.0f, acc = 0.0f;
        for (int64_t i = 0; i < kp; i++) {
            float lg = st->tail_g[i * hd + d] +
                       cwick_bf16_fast(ape[(size_t)i * (size_t)hd + (size_t)d]);
            float p = expf(lg - mx);
            den += p;
            float n = ((st->tail_k[i * hd + d] - mean[i]) * inv[i]) *
                      cwick_tensor_value(w->k_norm, d) +
                      cwick_tensor_value(w->k_norm_b, d);
            acc += p * n;
        }
        dst[d] = den > 0.0f ? acc / den : 0.0f;
    }
    st->n_pools++;
}

/* Hoare partition / quickselect: move the `k` largest scores to the front. */
static void score_select(float *v, int64_t n, int64_t k) {
    int64_t lo = 0, hi = n - 1;
    while (lo < hi) {
        const float pivot = v[(size_t)(lo + hi) >> 1];
        int64_t i = lo, j = hi;
        while (i <= j) {
            while (v[i] > pivot) i++;   /* descending: larger is better */
            while (v[j] < pivot) j--;
            if (i <= j) { float t = v[i]; v[i] = v[j]; v[j] = t; i++; j--; }
        }
        if (k <= j) hi = j;
        else if (k >= i) lo = i;
        else break;
    }
}

/* Pick the positions this query may attend to. Returns their count, or -1 to
 * mean "the whole prefix" -- inside index_topk that is not an approximation
 * but the identity the checkpoint itself computes. */
static int64_t indexer_select(const cwick_model_spec *spec,
                              const cwick_indexer_weights *w,
                              cwick_indexer_state *st,
                              const float *q_lat, const float *x, int64_t npos,
                              int64_t *sel) {
    const int64_t hd = spec->index_head_dim;
    const int64_t ih = spec->index_n_heads;
    const int64_t kp = spec->index_kpool > 0 ? spec->index_kpool : 1;
    const int64_t topk = spec->index_topk;
    const int64_t n_comp = st->n_pools;
    const int64_t want = (topk + kp - 1) / kp;    /* pools behind index_topk */
    if (want <= 0 || n_comp <= want) return -1;   /* whole prefix is top-k */

    float *q = st->q, *wt = st->wt, *sc = st->sc;
    cwick_linear(q, q_lat, 1, spec->q_lora_rank, ih * hd, w->wq_b);
    cwick_linear(wt, x, 1, spec->hidden_size, ih, w->weights_proj);
    const float wscale = 1.0f / sqrtf((float)((double)hd * (double)ih));
    for (int64_t h = 0; h < ih; h++) wt[h] *= wscale;

    if (attn_parallel()) {
        iscore_arg a;
        a.pool = st->pool; a.q = q; a.wt = wt; a.sc = sc;
        a.n_comp = n_comp; a.hd = hd; a.ih = ih;
        cwick_pool_for(n_comp, &a, iscore_range);
    } else {
        for (int64_t c = 0; c < n_comp; c++) {
            const float *kv = st->pool + (size_t)c * (size_t)hd;
            float s = 0.0f;
            for (int64_t h = 0; h < ih; h++) {
                float d = dot_f32(kv, q + h * hd, hd);
                if (d < 0.0f) d = 0.0f;           /* ReLU, per indexer head */
                s += d * wt[h];
            }
            sc[c] = s;
        }
    }

    /* Threshold = the `want`-th largest score. Taken on a copy: quickselect
     * permutes in place, and the per-pool identity must survive. Ties are
     * broken by pool order, which is arbitrary but deterministic. */
    memcpy(st->sc2, sc, (size_t)n_comp * sizeof(float));
    score_select(st->sc2, n_comp, want);
    const float thr = st->sc2[want - 1];

    const int64_t cap = want * kp;
    int64_t n = 0;
    for (int64_t pass = 0; pass < 2 && n < cap; pass++)
        for (int64_t c = 0; c < n_comp && n < cap; c++) {
            const bool take = pass == 0 ? (sc[c] > thr) : (sc[c] == thr);
            if (take) for (int64_t r = 0; r < kp; r++) sel[n++] = c * kp + r;
        }

    /* The positions after the last complete pool are in no compressed row
     * yet, so the indexer cannot have judged them -- always keep them
     * (index_kpool_always_select_tail). */
    const int64_t tail = npos % kp;
    for (int64_t r = 0; r < tail; r++) sel[n++] = npos - tail + r;

    /* ---- one deliberate addition to the tail rule --------------------
     * When `npos` is a multiple of index_kpool there is no incomplete tail,
     * and the query's own position can be pruned by the top-k. We always
     * keep it.
     *
     * The rationale is a safety property, not a quality claim: a query that
     * cannot attend to itself has no way to represent "copy the current
     * token", and nothing in the selection guards against an empty set
     * (which would make the softmax NaN). It only adds a position when
     * npos % kpool == 0 AND the last pool lost the top-k -- one extra
     * position out of 2048, and only on that arithmetic coincidence.
     * Set CWICK_INDEXER_STRICT=1 to drop it and take the plain top-k. */
    if (getenv("CWICK_INDEXER_STRICT") == NULL &&
        (n == 0 || sel[n - 1] != npos - 1))
        sel[n++] = npos - 1;
    return n;
}

/* Optional precomputed pieces for mla_latent_decode. The batched prefill
 * computes the projections for a whole chunk with one GEMM per weight, and
 * hands them in here so the per-token pass only does the reduction. NULL, or
 * NULL members, mean "compute them the usual way" -- which is what the decode
 * path does, unchanged.
 *
 * `attn_out` non-NULL means "stop after the attention reduction and leave
 * nh*vdim floats there": the caller batches the o_proj the same way. */
typedef struct {
    const float *q_a;      /* [q_lora], after its rms_norm + q_a_layernorm */
    const float *kv_a;     /* [kv_lora], after its rms_norm + kv_a_layernorm */
    const float *q;        /* [nh * q_dim] */
    float *attn_out;       /* [nh * vdim] destination, skips the o_proj */
} mla_pre;

/* Absorbed MLA over the latent cache. `latent` is [layers][capacity][lat_dim]
 * and already holds c_0..c_{pos-1}; c_pos is written here (unless `pre` brought
 * the projections, in which case the caller wrote it). */
static void mla_latent_decode(const cwick_model_spec *spec,
                              const cwick_tensor *qa, const cwick_tensor *qb,
                              const cwick_tensor *kva, const cwick_tensor *kvb,
                              const cwick_tensor *o,
                              const cwick_tensor *qa_n, const cwick_tensor *kva_n,
                              float *latent, int64_t capacity, int64_t pos,
                              int64_t layer,
                              const cwick_indexer_weights *idx_w,
                              cwick_indexer_state *idx,
                              const float *x, float *out, const mla_pre *pre) {
    const int64_t H = spec->hidden_size;
    const int64_t nh = spec->num_attention_heads;
    const int64_t nope = spec->qk_nope_head_dim;
    const int64_t vdim = spec->value_head_dim;
    const int64_t q_dim = nope;            /* qk_rope_head_dim == 0 here */
    const int64_t kv_dim = nope + vdim;
    const int64_t q_lora = spec->q_lora_rank;
    const int64_t kv_lora = spec->kv_lora_rank;
    const int64_t lat_dim = kv_lora;
    const float scale = 1.0f / sqrtf((float)q_dim);
    const int64_t npos = pos + 1;
    const uint16_t *kvb_w = (const uint16_t *)kvb->data;
    const bool lbf = cwick_latent_bf16_enabled();

    float *q_a = cwick_xmalloc((size_t)q_lora * sizeof(float));
    float *q = cwick_xmalloc((size_t)(nh * q_dim) * sizeof(float));
    float *kv_a = cwick_xmalloc((size_t)kv_lora * sizeof(float));
    float *q_abs = cwick_xmalloc((size_t)(nh * kv_lora) * sizeof(float));
    float *c_acc = cwick_xmalloc((size_t)(nh * kv_lora) * sizeof(float));
    float *attn_out = cwick_xmalloc((size_t)(nh * vdim) * sizeof(float));

    if (pre && pre->q_a && pre->kv_a && pre->q) {
        /* Chunk path: the projections were one GEMM for the whole chunk, the
         * norms and the latent write already happened. */
        memcpy(q_a, pre->q_a, (size_t)q_lora * sizeof(float));
        memcpy(kv_a, pre->kv_a, (size_t)kv_lora * sizeof(float));
        memcpy(q, pre->q, (size_t)(nh * q_dim) * sizeof(float));
    } else {
    /* latent query and latent KV are both functions of x: one batch. */
    {
        void *bb = cwick_metal_bf16_batch_begin();
        bool qa_ok = bb && qa->kind == CWICK_WEIGHT_BF16 &&
                     cwick_metal_bf16_batch_add(bb, qa->data, x, q_a, H, q_lora);
        bool kva_ok = bb && kva->kind == CWICK_WEIGHT_BF16 &&
                      cwick_metal_bf16_batch_add(bb, kva->data, x, kv_a, H, kv_lora);
        if (bb && !cwick_metal_bf16_batch_end(bb))
            qa_ok = kva_ok = false;
        if (!qa_ok) cwick_linear(q_a, x, 1, H, q_lora, qa);
        if (!kva_ok) cwick_linear(kv_a, x, 1, H, kv_lora, kva);
    }
    cwick_rms_norm(q_a, q_a, q_lora, NULL, spec->rms_norm_eps_f);
    apply_bf16_vec(qa_n, q_a, q_lora);
    cwick_linear(q, q_a, 1, q_lora, nh * q_dim, qb);

    /* latent KV: the only thing that has to be remembered for this position */
    cwick_rms_norm(kv_a, kv_a, kv_lora, NULL, spec->rms_norm_eps_f);
    apply_bf16_vec(kva_n, kv_a, kv_lora);
    {
        const size_t off = ((size_t)layer * (size_t)capacity + (size_t)pos) *
                           (size_t)lat_dim;
        if (lbf) {
            uint16_t *dst = (uint16_t *)latent + off;
            for (int64_t i = 0; i < lat_dim; i++)
                dst[i] = cwick_f32_to_bf16(kv_a[i]);
        } else {
            memcpy(latent + off, kv_a, (size_t)lat_dim * sizeof(float));
        }
    }
    }

    const void *lbase = (const uint8_t *)latent +
                        (size_t)layer * (size_t)capacity * (size_t)lat_dim *
                        (lbf ? sizeof(uint16_t) : sizeof(float));

    /* ---- DSA indexer: which positions may this query look at? ----------
     * Inside index_topk the answer is "all of them", which is exactly what
     * the checkpoint's top-k would return, so the indexer is skipped. */
    int64_t *sel = NULL;
    int64_t nlist = npos;
    if (idx_w && idx) {
        indexer_absorb(spec, idx_w, idx, x, pos);
        const int64_t got = indexer_select(spec, idx_w, idx, q_a, x, npos,
                                           idx->sel);
        if (got > 0) { sel = idx->sel; nlist = got; }
        if (indexer_trace()) {
            fprintf(stderr, "[indexer] layer=%lld pos=%lld pools=%lld "
                            "prefix=%lld selected=%lld (%.1f%%)\n",
                    (long long)layer, (long long)pos,
                    (long long)idx->n_pools, (long long)npos,
                    (long long)nlist,
                    100.0 * (double)nlist / (double)(npos > 0 ? npos : 1));
        }
    }
    float *scores = cwick_xmalloc((size_t)nh * (size_t)nlist * sizeof(float));

    /* absorb: q_abs[h][c] = sum_i q[h][i] * W_k[h][i][c], and later
     * out[h] = W_v[h] @ c_acc[h]. kv_b_proj is row-major [nh*kv_dim, kv_lora],
     * so head h owns rows [h*kv_dim, h*kv_dim+kv_dim): the first `nope` rows
     * are W_k, the next `vdim` rows are W_v -- both contiguous. */
    for (int64_t h = 0; h < nh; h++) {
        const uint16_t *wk_h = kvb_w + (size_t)(h * kv_dim) * (size_t)kv_lora;
        gemv_bf16_t(q_abs + h * kv_lora, nope, kv_lora, q + h * q_dim, wk_h);
    }

    /* scores[h][i] = (q_abs[h] . c_sel[i]) * scale, i indexing the *selected*
     * positions (or every position when no selection was made). Selected-
     * outer so the latent array is streamed once instead of once per head. */
    if (attn_parallel()) {
        scores_arg a;
        a.q_abs = q_abs; a.lbase = lbase; a.sel = sel; a.scores = scores;
        a.nlist = nlist; a.nh = nh; a.kv_lora = kv_lora; a.lat_dim = lat_dim;
        a.scale = scale; a.bf16 = lbf;
        cwick_pool_for(nlist, &a, scores_range);
    } else {
        const float *lf = (const float *)lbase;
        const uint16_t *lb = (const uint16_t *)lbase;
        for (int64_t i = 0; i < nlist; i++) {
            const int64_t t = sel ? sel[i] : i;
            for (int64_t h = 0; h < nh; h++)
                scores[h * nlist + i] =
                    (lbf ? dot_f32_bf16(q_abs + h * kv_lora,
                                        lb + (size_t)t * (size_t)lat_dim, kv_lora)
                         : dot_f32(q_abs + h * kv_lora,
                                   lf + (size_t)t * (size_t)lat_dim, kv_lora))
                    * scale;
        }
    }
    for (int64_t h = 0; h < nh; h++) cwick_softmax(scores + h * nlist, nlist);

    /* c_acc[h] = sum_i p[h][i] * c_sel[i].
     *
     * Deliberately *not* parallelised: it only has a head-local accumulator,
     * so the natural split (one head per worker) makes every worker walk the
     * whole selected latent set -- 12x the traffic for the same arithmetic.
     * Position-outer is what keeps it at one pass, and that order needs a
     * real reduction across workers, which is not worth it against a cost
     * that is already a minority of the step. */
    memset(c_acc, 0, (size_t)(nh * kv_lora) * sizeof(float));
    {
        const float *lf = (const float *)lbase;
        const uint16_t *lb = (const uint16_t *)lbase;
        for (int64_t i = 0; i < nlist; i++) {
            const int64_t t = sel ? sel[i] : i;
            const size_t off = (size_t)t * (size_t)lat_dim;
            for (int64_t h = 0; h < nh; h++) {
                const float p = scores[h * nlist + i];
                float *a = c_acc + h * kv_lora;
                if (lbf) {
#if defined(__aarch64__) && defined(__ARM_NEON)
                    const float32x4_t pv = vdupq_n_f32(p);
                    int64_t j = 0;
                    for (; j + 8 <= kv_lora; j += 8) {
                        uint16x8_t h = vld1q_u16(lb + off + j);
                        vst1q_f32(a + j, vfmaq_f32(vld1q_f32(a + j), pv,
                                                 attn_bf16_lo(h)));
                        vst1q_f32(a + j + 4, vfmaq_f32(vld1q_f32(a + j + 4), pv,
                                                       attn_bf16_hi(h)));
                    }
                    for (; j < kv_lora; j++) a[j] += p * cwick_bf16_fast(lb[off + j]);
#else
                    for (int64_t i2 = 0; i2 < kv_lora; i2++)
                        a[i2] += p * cwick_bf16_fast(lb[off + i2]);
#endif
                } else {
                    const float *c = lf + off;
                    for (int64_t i2 = 0; i2 < kv_lora; i2++) a[i2] += p * c[i2];
                }
            }
        }
    }
    for (int64_t h = 0; h < nh; h++) {
        const uint16_t *wv_h = kvb_w + (size_t)(h * kv_dim + nope) * (size_t)kv_lora;
        gemv_bf16_n(attn_out + h * vdim, vdim, kv_lora, c_acc + h * kv_lora, wv_h);
    }

    if (atrace_on() && (layer == 3 || layer == 7)) {
        aprobe("mla q", q, nh * q_dim);
        aprobe("mla q_abs", q_abs, nh * kv_lora);
        /* The latent probe reads fp32; with a bf16 cache the row is widened
         * into a scratch buffer first rather than inventing a second probe
         * format -- this path only exists under CWICK_ATTN_TRACE. */
        if (!lbf)
            aprobe("mla latent last",
                   (const float *)lbase + (size_t)pos * (size_t)lat_dim, kv_lora);
        aprobe("mla scores h0", scores, nlist);
        aprobe("mla c_acc", c_acc, nh * kv_lora);
        aprobe("mla attn_out_pre", attn_out, nh * vdim);
    }

    /* ---- self-check ----------------------------------------------------
     * Recompute the same attention the way the expanded cache does it:
     * materialise k_t and v_t from every cached latent with the full
     * kv_b_proj, then score and accumulate exactly as the original kernel
     * did. This is the reference the absorption has to reproduce, run on the
     * same inputs in the same process, so it isolates the optimisation from
     * everything else in the model. CWICK_MLA_SELFCHECK=1. */
    if (mla_selfcheck()) {
        float *kv_b = cwick_xmalloc((size_t)(nh * kv_dim) * sizeof(float));
        float *ref = cwick_xmalloc((size_t)(nh * vdim) * sizeof(float));
        float *rs = cwick_xmalloc((size_t)nlist * sizeof(float));
        /* One widened copy of the latent row: the reference path works in
         * fp32, so it reads whatever precision the cache holds through this
         * buffer. Both sides of the comparison see the same values, which is
         * what makes the check meaningful in either mode. */
        float *crow = cwick_xmalloc((size_t)lat_dim * sizeof(float));
        const float *lf2 = (const float *)lbase;
        const uint16_t *lb2 = (const uint16_t *)lbase;
        memset(ref, 0, (size_t)(nh * vdim) * sizeof(float));
        for (int64_t h = 0; h < nh; h++) {
            const uint16_t *wk = kvb_w + (size_t)(h * kv_dim) * (size_t)kv_lora;
            const uint16_t *wv = kvb_w + (size_t)(h * kv_dim + nope) * (size_t)kv_lora;
            float *krow = kv_b + h * kv_dim;
            float *vrow = kv_b + h * kv_dim + nope;
            for (int64_t i = 0; i < nlist; i++) {
                const int64_t t = sel ? sel[i] : i;
                const size_t to = (size_t)t * (size_t)lat_dim;
                if (lbf)
                    for (int64_t j = 0; j < lat_dim; j++)
                        crow[j] = cwick_bf16_fast(lb2[to + j]);
                else
                    memcpy(crow, lf2 + to, (size_t)lat_dim * sizeof(float));
                gemv_bf16_n(krow, nope, kv_lora, crow, wk);
                gemv_bf16_n(vrow, vdim, kv_lora, crow, wv);
                rs[i] = dot_f32(q + h * q_dim, krow, nope) * scale;
            }
            cwick_softmax(rs, nlist);
            for (int64_t i = 0; i < nlist; i++) {
                const int64_t t = sel ? sel[i] : i;
                const size_t to = (size_t)t * (size_t)lat_dim;
                if (lbf)
                    for (int64_t j = 0; j < lat_dim; j++)
                        crow[j] = cwick_bf16_fast(lb2[to + j]);
                else
                    memcpy(crow, lf2 + to, (size_t)lat_dim * sizeof(float));
                gemv_bf16_n(vrow, vdim, kv_lora, crow, wv);
                float *o2 = ref + h * vdim;
                for (int64_t j = 0; j < vdim; j++) o2[j] += rs[i] * vrow[j];
            }
        }
        double mx = 0.0, sc = 0.0;
        for (int64_t i = 0; i < nh * vdim; i++) {
            double d = fabs((double)attn_out[i] - (double)ref[i]);
            double a = fabs((double)ref[i]);
            if (d > mx) mx = d;
            if (a > sc) sc = a;
        }
        fprintf(stderr, "[mla-selfcheck] layer=%lld pos=%lld nsel=%lld "
                        "maxdiff=%.3e refmax=%.3e rel=%.3e\n",
                (long long)layer, (long long)pos, (long long)nlist,
                mx, sc, sc > 0 ? mx / sc : mx);
        free(rs); free(ref); free(kv_b); free(crow);
    }

    if (pre && pre->attn_out)
        memcpy(pre->attn_out, attn_out, (size_t)(nh * vdim) * sizeof(float));
    else
        cwick_linear(out, attn_out, 1, nh * vdim, H, o);

    free(attn_out); free(c_acc); free(scores); free(q_abs);
    free(kv_a); free(q); free(q_a);
}

/* Batched prefill of one sparse-attention layer.
 *
 * The per-token path does four GEMVs per token (q_a, kv_a, q, o_proj) and a
 * GEMV re-reads its weight matrix: at 3037 tokens that is ~700 GB of weight
 * traffic per layer, against ~230 MB if the same weights are read once and
 * applied to every token at once. That gap, not arithmetic, is why a long
 * prefill spends minutes per layer -- and it is why moving the *reductions*
 * to twelve threads did not help (they were never the wall).
 *
 * So: one GEMM per weight for the whole chunk, one serial pass for the parts
 * that genuinely depend on token order (the DSA indexer, the score/value
 * reduction), then one GEMM for the output projection. Returns
 * CWICK_ERR_UNSUPPORTED when the checkpoint is not in the layout this path
 * knows (expanded KV cache, non-BF16 projections) so the caller can fall back
 * to the per-token loop. */
/* CWICK_NO_CHUNK_ATTN=1 disables the chunk path, which is how the equivalence
 * tests isolate it from the per-token sweep. */
static int g_chunk_attn = -1;
static bool chunk_attn_enabled(void) {
    if (g_chunk_attn < 0)
        g_chunk_attn = cwick_env_true("CWICK_NO_CHUNK_ATTN") ? 0 : 1;
    return g_chunk_attn == 1;
}

/* Whether the chunk path can run for this layer. The caller needs this
 * *before* it runs the mHC pre-mix: once that has been computed for the whole
 * chunk there is no going back to the fused per-token loop without doing the
 * mix twice. */
bool cwick_mla_chunk_supported(const cwick_model_spec *spec,
                              const cwick_tensor *qa, const cwick_tensor *qb,
                              const cwick_tensor *kva, const cwick_tensor *kvb,
                              const cwick_tensor *o, const float *values) {
    if (!spec || !qa || !qb || !kva || !kvb || !o) return false;
    if (!chunk_attn_enabled()) return false;
    if (values != NULL || !mla_latent_ok(spec, kvb)) return false;
    return qa->kind == CWICK_WEIGHT_BF16 && kva->kind == CWICK_WEIGHT_BF16 &&
           qb->kind == CWICK_WEIGHT_BF16 && o->kind == CWICK_WEIGHT_BF16;
}

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
                                             cwick_error *err) {
    if (!spec || n <= 0 || !keys) return CWICK_ERR_ARGUMENT;
    if (!chunk_attn_enabled()) return CWICK_ERR_UNSUPPORTED;
    if (values != NULL || !mla_latent_ok(spec, kvb)) return CWICK_ERR_UNSUPPORTED;
    if (qa->kind != CWICK_WEIGHT_BF16 || kva->kind != CWICK_WEIGHT_BF16 ||
        qb->kind != CWICK_WEIGHT_BF16 || o->kind != CWICK_WEIGHT_BF16)
        return CWICK_ERR_UNSUPPORTED;

    const int64_t H = spec->hidden_size;
    const int64_t nh = spec->num_attention_heads;
    const int64_t vdim = spec->value_head_dim;
    const int64_t q_dim = spec->qk_nope_head_dim;
    const int64_t q_lora = spec->q_lora_rank;
    const int64_t kv_lora = spec->kv_lora_rank;
    const int64_t lat_dim = kv_lora;
    const int64_t attn_dim = nh * vdim;

    float *QA   = cwick_xmalloc((size_t)n * (size_t)q_lora * sizeof(float));
    float *KVA  = cwick_xmalloc((size_t)n * (size_t)kv_lora * sizeof(float));
    float *Q    = cwick_xmalloc((size_t)n * (size_t)nh * (size_t)q_dim * sizeof(float));
    float *ATTN = cwick_xmalloc((size_t)n * (size_t)attn_dim * sizeof(float));
    if (!QA || !KVA || !Q || !ATTN) {
        free(QA); free(KVA); free(Q); free(ATTN);
        cwick_error_set(err, CWICK_ERR_MEMORY, "mla chunk: out of memory");
        return CWICK_ERR_MEMORY;
    }

    /* The whole point: each weight is touched once for the chunk. */
    cwick_gemm_bf16(QA, n, H, q_lora, X, qa->data);
    cwick_gemm_bf16(KVA, n, H, kv_lora, X, kva->data);

    for (int64_t t = 0; t < n; t++) {
        float *qa_row = QA + t * q_lora;
        float *kv_row = KVA + t * kv_lora;
        cwick_rms_norm(qa_row, qa_row, q_lora, NULL, spec->rms_norm_eps_f);
        apply_bf16_vec(qa_n, qa_row, q_lora);
        cwick_rms_norm(kv_row, kv_row, kv_lora, NULL, spec->rms_norm_eps_f);
        apply_bf16_vec(kva_n, kv_row, kv_lora);
        /* Written at whatever width the cache is, not at sizeof(float): with a
         * bf16 latent the chunk path would otherwise write 4 bytes per element
         * into a 2-byte-per-element buffer -- twice the bytes, past the end. */
        const size_t off = ((size_t)layer * (size_t)capacity +
                            (size_t)(pos0 + t)) * (size_t)lat_dim;
        if (cwick_latent_bf16_enabled()) {
            uint16_t *dst = (uint16_t *)keys + off;
            for (int64_t i = 0; i < kv_lora; i++)
                dst[i] = cwick_f32_to_bf16(kv_row[i]);
        } else {
            memcpy(keys + off, kv_row, (size_t)lat_dim * sizeof(float));
        }
    }
    cwick_gemm_bf16(Q, n, q_lora, nh * q_dim, QA, qb->data);

    for (int64_t t = 0; t < n; t++) {
        mla_pre pre;
        pre.q_a = QA + t * q_lora;
        pre.kv_a = KVA + t * kv_lora;
        pre.q = Q + t * nh * q_dim;
        pre.attn_out = ATTN + t * attn_dim;
        mla_latent_decode(spec, qa, qb, kva, kvb, o, qa_n, kva_n,
                          keys, capacity, pos0 + t, layer, idx_w, idx,
                          X + t * H, Out + t * H, &pre);
    }
    cwick_gemm_bf16(Out, n, attn_dim, H, ATTN, o->data);

    free(QA); free(KVA); free(Q); free(ATTN);
    return CWICK_OK;
}

void cwick_attention_mla_decode(const cwick_model_spec *spec,
                               const cwick_tensor *qa, const cwick_tensor *qb,
                               const cwick_tensor *kva, const cwick_tensor *kvb,
                               const cwick_tensor *g, const cwick_tensor *o,
                               const cwick_tensor *qa_n, const cwick_tensor *kva_n,
                               float *keys, float *values, int64_t capacity,
                               int64_t pos, int64_t layer,
                               const cwick_indexer_weights *idx_w,
                               cwick_indexer_state *idx,
                               const float *x, float *out, cwick_error *err) {
    if (values == NULL && mla_latent_ok(spec, kvb)) {
        mla_latent_decode(spec, qa, qb, kva, kvb, o, qa_n, kva_n,
                          keys, capacity, pos, layer, idx_w, idx, x, out, NULL);
        return;
    }
    const int64_t H = spec->hidden_size;
    const int64_t nh = spec->num_attention_heads;
    const int64_t nope = spec->qk_nope_head_dim;
    const int64_t rope = spec->qk_rope_head_dim;
    const int64_t vdim = spec->value_head_dim;
    const int64_t q_dim = nope + rope;
    const int64_t kv_dim = nope + vdim;
    const int64_t q_lora = spec->q_lora_rank;
    const int64_t kv_lora = spec->kv_lora_rank;
    const int64_t gdim = nh * vdim;
    const float scale = 1.0f / sqrtf((float)q_dim);
    const int64_t npos = pos + 1;
    /* Two layouts share this kernel:
     *   - MLA (GLM-5.3 / DeepSeek): kv_b_proj emits [nh, nope+vdim] — one
     *     latent expanded into a private slice per query head.
     *   - GQA (legacy converted models): kv_b_proj emits [nkvh, ...] with
     *     nkvh < nh, and head h shares kv head h*nkvh/nh.
     * They are told apart by the weight's own row count, never by a flag. */
    const bool mla_layout = (kvb->rows == nh * kv_dim);
    const int64_t nkvh = mla_layout ? nh : (kv_dim > 0 ? kvb->rows / kv_dim : 1);
    float *q_a = cwick_xmalloc((size_t)q_lora * sizeof(float));
    float *q = cwick_xmalloc((size_t)(nh * q_dim) * sizeof(float));
    float *kv_a = cwick_xmalloc((size_t)(kv_lora + rope) * sizeof(float));
    float *kv_b = cwick_xmalloc((size_t)(nkvh * kv_dim) * sizeof(float));
    float *scores = cwick_xmalloc((size_t)npos * sizeof(float));
    float *attn_out = cwick_xmalloc((size_t)(nh * vdim) * sizeof(float));
    float *gated = g ? cwick_xmalloc((size_t)gdim * sizeof(float)) : NULL;
    float *gate = g ? cwick_xmalloc((size_t)gdim * sizeof(float)) : NULL;

    /* latent query */
    cwick_linear(q_a, x, 1, H, q_lora, qa);
    cwick_rms_norm(q_a, q_a, q_lora, NULL, spec->rms_norm_eps_f);
    apply_bf16_vec(qa_n, q_a, q_lora);
    cwick_linear(q, q_a, 1, q_lora, nh * q_dim, qb);

    /* latent KV */
    cwick_linear(kv_a, x, 1, H, kv_lora + rope, kva);
    cwick_rms_norm(kv_a, kv_a, kv_lora, NULL, spec->rms_norm_eps_f);
    apply_bf16_vec(kva_n, kv_a, kv_lora);
    cwick_linear(kv_b, kv_a, 1, kv_lora, nh * kv_dim, kvb);

    /* store this position's key/value. MLA keeps a private nope/v slice per
     * query head; GQA keeps one slice per kv head. */
    const int64_t kstride = nh * q_dim;   /* floats per cached position */
    const int64_t vstride = nh * vdim;
    {
        float *k = keys + layer * capacity * kstride + pos * kstride;
        float *vstore = values + layer * capacity * vstride + pos * vstride;
        for (int64_t h = 0; h < nkvh; h++) {
            const float *src = kv_b + h * kv_dim;
            memcpy(k + h * q_dim, src, (size_t)nope * sizeof(float));
            if (rope) memcpy(k + h * q_dim + nope, kv_a + kv_lora,
                             (size_t)rope * sizeof(float));
            memcpy(vstore + h * vdim, src + nope, (size_t)vdim * sizeof(float));
        }
        if (rope) {
            for (int64_t h = 0; h < nkvh; h++)
                rope_apply(k + h * q_dim + nope, rope, pos, 10000.0f);
        }
    }

    memset(attn_out, 0, (size_t)(nh * vdim) * sizeof(float));
    const float *kbase = keys + layer * capacity * kstride;
    const float *vbase = values + layer * capacity * vstride;
    for (int64_t h = 0; h < nh; h++) {
        const int64_t kh = mla_layout ? h : h * nkvh / nh;
        const float *qrow = q + h * q_dim;
        for (int64_t t = 0; t < npos; t++) {
            const float *krow = kbase + t * kstride + kh * q_dim;
            float acc = 0.0f;
            for (int64_t i = 0; i < q_dim; i++) acc += qrow[i] * krow[i];
            scores[t] = acc * scale;
        }
        cwick_softmax(scores, npos);
        float *orow = attn_out + h * vdim;
        for (int64_t t = 0; t < npos; t++) {
            const float *vrow = vbase + t * vstride + kh * vdim;
            float p = scores[t];
            for (int64_t i = 0; i < vdim; i++) orow[i] += p * vrow[i];
        }
    }
    if (atrace_on() && (layer == 3 || layer == 7)) {
        aprobe("mla q", q, nh * q_dim);
        { const float *kb = keys + layer * capacity * kstride + pos * kstride;
          aprobe("mla key last", kb, nkvh * q_dim); }
        { const float *vb = values + layer * capacity * vstride + pos * vstride;
          aprobe("mla val last", vb, nkvh * vdim); }
        aprobe("mla attn_out_pre", attn_out, nh * vdim);
    }

    /* optional output gate: g_proj(x), sigmoid, multiply, o_proj */
    if (g) {
        cwick_linear(gate, x, 1, H, gdim, g);
        for (int64_t i = 0; i < gdim; i++) {
            float sg = 1.0f / (1.0f + expf(-gate[i]));
            gated[i] = attn_out[i] * sg;
        }
        cwick_linear(out, gated, 1, gdim, H, o);
    } else {
        cwick_linear(out, attn_out, 1, gdim, H, o);
    }

    free(gate);
    free(gated);
    free(attn_out);
    free(scores);
    free(kv_b);
    free(kv_a);
    free(q);
    free(q_a);
    (void)nkvh;
    (void)err;
}

/* ------------------------------------------------------------------ */
/* KDA decode                                                          */
/* ------------------------------------------------------------------ */
/* state layout:
 *   rec:  [L * heads * head_dim * head_dim]
 *   conv: [L * proj * 3]  (last three *projected* values per channel)
 */
/* Sub-phase timers for the attention path. `attn` is one prof phase, but it
 * spans two very different layers (34 KDA linear-attention layers and 11 MLA
 * sparse ones) and, inside KDA, projections that already run on the GPU
 * alongside a recurrence that does not. Without this split, "attn is 132 ms"
 * is not actionable: it does not say which half to move.
 *
 * Reported by cwick_prof_report alongside the phases, reset with them. */
uint64_t g_attn_proj_ns;   /* KDA/MLA projections (GPU batch + CPU fallback) */
uint64_t g_attn_conv_ns;   /* short conv + sigmoid + L2 norms                */
uint64_t g_attn_gate_ns;   /* f_a/f_b/b projections + gated decay + beta     */
uint64_t g_attn_rec_ns;    /* the KDA recurrence (decay/delta/outer/core)    */
uint64_t g_attn_mla_ns;    /* MLA/DSA layers, projections included           */
uint64_t g_attn_tail_ns;   /* output gate, o_norm, o_proj                    */

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
                               const float *x, float *out, cwick_error *err) {
    const int64_t H = spec->hidden_size;
    /* KDA geometry comes from linear_attn_config, NOT from the MLA head
     * geometry: GLM-5.3 uses 64 heads of 128 while the MLA path uses 256-dim
     * heads. Mixing them silently corrupts every linear-attention layer. */
    const int64_t nh = spec->linear_num_heads;
    const int64_t hd = spec->linear_head_dim;
    const int64_t proj = nh * hd;
    const int64_t width = spec->short_conv_kernel_size; /* 4 */
    const int64_t conv_state = width - 1;               /* 3 */
    const float eps = 1.0e-5f;
    const float lower = spec->gate_lower_bound;
    const float rms_eps = spec->rms_norm_eps_f;

    float *q = cwick_xmalloc((size_t)proj * sizeof(float));
    float *k = cwick_xmalloc((size_t)proj * sizeof(float));
    float *v = cwick_xmalloc((size_t)proj * sizeof(float));
    float *f_a = cwick_xmalloc((size_t)hd * sizeof(float));
    float *g_a = cwick_xmalloc((size_t)hd * sizeof(float));
    float *gate_raw = cwick_xmalloc((size_t)proj * sizeof(float));
    float *beta = cwick_xmalloc((size_t)nh * sizeof(float));
    float *gate = cwick_xmalloc((size_t)proj * sizeof(float));

    /* q/k/v are independent functions of x, so they can share one GPU
     * command buffer: each is ~311 us of GPU work against ~745 us on the
     * CPU pool, but a commit+wait round trip costs ~450 us of scheduling,
     * which is exactly what the batch removes. Anything the batch refuses
     * (not BF16, too small, a weight Metal cannot adopt) or fails to land
     * falls back to the serial path for that projection only.
     *
     * f_a / beta / g_a are also pure functions of x, just consumed later
     * (after the conv, the recurrence and the gate respectively). Issuing
     * them in the same batch costs only their (tiny) kernel time and drops
     * three serial CPU GEMVs per KDA layer: fa [hd,H], b [nh,H] and
     * ga [hd,H] are 0.5 MiB, 0.25 MiB and 0.5 MiB of BF16, one matvec each
     * on the CPU pool. Buffers are allocated before the batch on purpose. */
    uint64_t _sp = cwick_prof_ns();
    {
        void *bb = cwick_metal_bf16_batch_begin();
        bool q_ok = bb && q_proj->kind == CWICK_WEIGHT_BF16 &&
                    cwick_metal_bf16_batch_add(bb, q_proj->data, x, q, H, proj);
        bool k_ok = bb && k_proj->kind == CWICK_WEIGHT_BF16 &&
                    cwick_metal_bf16_batch_add(bb, k_proj->data, x, k, H, proj);
        bool v_ok = bb && v_proj->kind == CWICK_WEIGHT_BF16 &&
                    cwick_metal_bf16_batch_add(bb, v_proj->data, x, v, H, proj);
        bool fa_ok = bb && fa_proj->kind == CWICK_WEIGHT_BF16 &&
                     cwick_metal_bf16_batch_add(bb, fa_proj->data, x, f_a, H, hd);
        bool b_ok = bb && b_proj->kind == CWICK_WEIGHT_BF16 &&
                    cwick_metal_bf16_batch_add(bb, b_proj->data, x, beta, H, nh);
        bool ga_ok = bb && ga_proj->kind == CWICK_WEIGHT_BF16 &&
                     cwick_metal_bf16_batch_add(bb, ga_proj->data, x, g_a, H, hd);
        if (bb && !cwick_metal_bf16_batch_end(bb))
            q_ok = k_ok = v_ok = fa_ok = b_ok = ga_ok = false;
        if (!q_ok)  cwick_linear(q, x, 1, H, proj, q_proj);
        if (!k_ok)  cwick_linear(k, x, 1, H, proj, k_proj);
        if (!v_ok)  cwick_linear(v, x, 1, H, proj, v_proj);
        if (!fa_ok) cwick_linear(f_a, x, 1, H, hd, fa_proj);
        if (!b_ok)  cwick_linear(beta, x, 1, H, nh, b_proj);
        if (!ga_ok) cwick_linear(g_a, x, 1, H, hd, ga_proj);
    }
    g_attn_proj_ns += cwick_prof_ns() - _sp;
    uint64_t _sc = cwick_prof_ns();

    /* Short convolution with state (state holds the *projected* values).
     * The conv weights are BF16 in the checkpoint, like every other weight
     * here — reading them as f32 turns the whole recurrence into NaN. */
    const uint16_t *qcw = (const uint16_t *)q_conv->data;
    const uint16_t *kcw = (const uint16_t *)k_conv->data;
    const uint16_t *vcw = (const uint16_t *)v_conv->data;
    /* conv layout: [L][3][proj][3] — three per-channel states (q,k,v) */
    float *cq = conv + layer * (3 * proj * conv_state) + 0 * (proj * conv_state);
    float *ck = conv + layer * (3 * proj * conv_state) + 1 * (proj * conv_state);
    float *cv = conv + layer * (3 * proj * conv_state) + 2 * (proj * conv_state);
    for (int64_t i = 0; i < proj; i++) {
        float *sq = cq + i * conv_state;
        float *sk = ck + i * conv_state;
        float *sv = cv + i * conv_state;
        float qacc = sq[0] * cwick_bf16_fast(qcw[i * width + 0]) +
                     sq[1] * cwick_bf16_fast(qcw[i * width + 1]) +
                     sq[2] * cwick_bf16_fast(qcw[i * width + 2]) +
                     q[i] * cwick_bf16_fast(qcw[i * width + 3]);
        float kacc = sk[0] * cwick_bf16_fast(kcw[i * width + 0]) +
                     sk[1] * cwick_bf16_fast(kcw[i * width + 1]) +
                     sk[2] * cwick_bf16_fast(kcw[i * width + 2]) +
                     k[i] * cwick_bf16_fast(kcw[i * width + 3]);
        float vacc = sv[0] * cwick_bf16_fast(vcw[i * width + 0]) +
                     sv[1] * cwick_bf16_fast(vcw[i * width + 1]) +
                     sv[2] * cwick_bf16_fast(vcw[i * width + 2]) +
                     v[i] * cwick_bf16_fast(vcw[i * width + 3]);
        /* keep the projected input for the next window */
        float qp = q[i], kp = k[i], vp = v[i];
        q[i] = qacc * (1.0f / (1.0f + expf(-qacc)));
        k[i] = kacc * (1.0f / (1.0f + expf(-kacc)));
        v[i] = vacc * (1.0f / (1.0f + expf(-vacc)));
        sq[0] = sq[1]; sq[1] = sq[2]; sq[2] = qp;
        sk[0] = sk[1]; sk[1] = sk[2]; sk[2] = kp;
        sv[0] = sv[1]; sv[1] = sv[2]; sv[2] = vp;
    }

    aprobe("qkv proj", q, proj); aprobe("conv out", k, proj);
    /* L2-normalize query/key per head; query additionally / sqrt(hd) */
    for (int64_t h = 0; h < nh; h++) {
        const float *qrow = q + h * hd;
        const float *krow = k + h * hd;
        double qn = 0.0, kn = 0.0;
        for (int64_t j = 0; j < hd; j++) {
            qn += (double)qrow[j] * qrow[j];
            kn += (double)krow[j] * krow[j];
        }
        float qinv = 1.0f / (float)(sqrt(qn) < eps ? eps : sqrt(qn));
        float kinv = 1.0f / (float)(sqrt(kn) < eps ? eps : sqrt(kn));
        for (int64_t j = 0; j < hd; j++) {
            q[h * hd + j] = qrow[j] * qinv / sqrtf((float)hd);
            k[h * hd + j] = krow[j] * kinv;
        }
    }

    g_attn_conv_ns += cwick_prof_ns() - _sc;
    uint64_t _sg = cwick_prof_ns();

    /* gated decay: lower_bound * sigmoid(exp(A_log) * (g1 + dt_bias)).
     * A_log has one entry *per head* (checkpoint: [num_heads]), while g1 and
     * dt_bias are per (head, channel). The decay then scales the recurrent
     * state along the key dimension (state[k][v] *= exp(g[k])), matching
     * llama.cpp's `g_last_exp_t` broadcast for KDA.
     * f_a came out of the batch above (it only depends on x); only the
     * second stage still runs here. */
    cwick_linear(gate_raw, f_a, 1, hd, proj, fb_proj);
    float *gdecay = cwick_xmalloc((size_t)proj * sizeof(float));
    for (int64_t i = 0; i < proj; i++) {
        int64_t h = i / hd;
        float raw = gate_raw[i] + cwick_tensor_value(dt_bias, i);
        float sg = 1.0f / (1.0f + expf(-expf(cwick_tensor_value(a_log, h)) * raw));
        gdecay[i] = lower * sg;
    }

    aprobe("gdecay", gdecay, proj);
    if (atrace_on()) {
        fprintf(stderr, "    kda inputs: lower=%g nh=%lld hd=%lld proj=%lld width=%lld\n",
                (double)lower, (long long)nh, (long long)hd, (long long)proj, (long long)width);
        fprintf(stderr, "    kda alog[0..2]=%g %g %g  dt_bias[0..2]=%g %g %g\n",
                (double)cwick_tensor_value(a_log, 0),
                (double)cwick_tensor_value(a_log, 1),
                (double)cwick_tensor_value(a_log, 2),
                (double)cwick_tensor_value(dt_bias, 0),
                (double)cwick_tensor_value(dt_bias, 1),
                (double)cwick_tensor_value(dt_bias, 2));
        fprintf(stderr, "    kda alog dtype rows=%lld cols=%lld kind=%d ; dt rows=%lld cols=%lld kind=%d\n",
                (long long)a_log->rows, (long long)a_log->cols, (int)a_log->kind,
                (long long)dt_bias->rows, (long long)dt_bias->cols, (int)dt_bias->kind);
    }
    /* beta (also a function of x: the batch above already produced it) */
    for (int64_t h = 0; h < nh; h++) {
        float b = beta[h];
        beta[h] = 1.0f / (1.0f + expf(-b));
    }

    /* The recurrent output, the gated norm and the output gate all live in
     * `proj` space (nh*hd), which is NOT the caller's `out` (hidden_size).
     * Projecting first keeps a single hidden-sized write at the end. */
    g_attn_gate_ns += cwick_prof_ns() - _sg;
    uint64_t _sr = cwick_prof_ns();

    float *core = cwick_xmalloc((size_t)proj * sizeof(float));
    float *delta = cwick_xmalloc((size_t)hd * sizeof(float));
    float *rbase = rec + layer * nh * hd * hd;
    for (int64_t h = 0; h < nh; h++) {
        const float *qrow = q + h * hd;
        const float *krow = k + h * hd;
        const float *vrow = v + h * hd;
        const float *gd = gdecay + h * hd;
        float *r = rbase + h * hd * hd;
        float *orow = core + h * hd;

        /* Apply the gated decay in place. The previous implementation copied
         * the complete 64x128 recurrent matrix for every one of 34 KDA layers
         * (over 2 GiB of extra state traffic per decoded token) and allocated
         * one delta vector per head. Both are unnecessary: delta is complete
         * before the outer-product update, so the authoritative state can be
         * transformed row by row while the output is accumulated. */
        for (int64_t kk = 0; kk < hd; kk++) {
            float g = expf(gd[kk]);
            float *row = r + kk * hd;
            for (int64_t vv = 0; vv < hd; vv++) row[vv] *= g;
        }

        /* delta = value - key @ state. Iterating key-major keeps every state
         * row contiguous (the old value-major loop walked a 128-float stride
         * for every element and defeated the prefetcher). */
        memcpy(delta, vrow, (size_t)hd * sizeof(float));
        for (int64_t kk = 0; kk < hd; kk++) {
            float coeff = krow[kk];
            const float *row = r + kk * hd;
            for (int64_t vv = 0; vv < hd; vv++) delta[vv] -= coeff * row[vv];
        }

        /* state += outer(beta*key, delta) */
        for (int64_t kk = 0; kk < hd; kk++) {
            float coeff = beta[h] * krow[kk];
            float *row = r + kk * hd;
            for (int64_t vv = 0; vv < hd; vv++) row[vv] += coeff * delta[vv];
        }

        /* core[h][v] = sum_k q[k] * state[k][v] */
        memset(orow, 0, (size_t)hd * sizeof(float));
        for (int64_t kk = 0; kk < hd; kk++) {
            float coeff = qrow[kk];
            const float *row = r + kk * hd;
            for (int64_t vv = 0; vv < hd; vv++) orow[vv] += coeff * row[vv];
        }
    }
    free(delta);
    g_attn_rec_ns += cwick_prof_ns() - _sr;
    uint64_t _st = cwick_prof_ns();
    aprobe("core", core, proj);

    /* per-head gated RMSNorm (vLLM FusedRMSNormGated, activation="sigmoid"):
     * out = rms_norm(out) * o_norm_weight * sigmoid(g_b(g_a(x))).
     * The gate is two-stage in GLM-5.3; a single projection does not exist.
     * g_a itself came out of the q/k/v batch above (it is a function of x),
     * so only the second stage runs here. */
    {
        const uint16_t *on = (const uint16_t *)o_norm->data;
        cwick_linear(gate, g_a, 1, hd, proj, gb_proj);
        for (int64_t h = 0; h < nh; h++) {
            float *orow = core + h * hd;
            double ss = 0.0;
            for (int64_t j = 0; j < hd; j++) ss += (double)orow[j] * orow[j];
            float inv = 1.0f / (float)sqrt(ss / (double)hd + (double)rms_eps);
            for (int64_t j = 0; j < hd; j++) {
                float sg = 1.0f / (1.0f + expf(-gate[h * hd + j]));
                orow[j] = orow[j] * inv * cwick_bf16_fast(on[j]) * sg;
            }
        }
    }
    aprobe("gate", gate, proj);
    aprobe("normed core", core, proj);
    cwick_linear(out, core, 1, proj, H, o_proj);
    g_attn_tail_ns += cwick_prof_ns() - _st;

    free(core);
    free(gdecay);
    free(gate);
    free(beta);
    free(gate_raw);
    free(f_a);
    free(g_a);
    free(v);
    free(k);
    free(q);
    (void)err;
}
