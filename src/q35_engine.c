/*
 * q35_engine.c — generation on top of the Qwen3.5 forward pass.
 *
 * The shape of a run is the DeepSeek and MiniMax backends' shape, deliberately:
 * prefill in chunks, then one forward per token, with the same sampler, the
 * same thinking-budget policy, the same loop guard and the same progress lines.
 * What differs is only what the model requires.
 *
 * One difference worth naming: this checkpoint has no routed experts, so there
 * is nothing to stream and no cache to budget. Every weight is resident BF16
 * (~18 GiB) and the whole of a step's cost is reading them once. That makes the
 * decode rate a pure function of memory bandwidth -- 18 GiB per step against a
 * measured ~123 GB/s ceiling is about 6.8 tok/s, and the measured rate is
 * ~7.0 tok/s, i.e. 92% of the bus. Everything below that line would have been
 * overhead, and there is almost nothing left of it: the per-token delta-rule
 * scan and the attention loop now run across cores (4 ms and 2 ms of a 129 ms
 * step), so a step is the GEMVs and nothing else.
 *
 * The 7.0 tok/s figure is with the weights faulted in and wired at load; a
 * lazily mapped checkpoint measures the same once its pages are warm, but pays
 * for them inside the first request instead of at startup (DECISIONS §52).
 */
#include "q35_engine.h"
#include "q35_gpu.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "log.h"
#include "mem.h"
#include "metrics.h"
#include "mm.h"        /* mm_prefill_progress_*, the shared prefill reporter */
#include "sample.h"
#include "util.h"

static double q35_now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1e3 + (double)tv.tv_usec / 1e3;
}

static bool q35_stopped(const struct q35_ctx *cx) {
    if (cx->stop && *cx->stop) return true;
    if (cx->abort_poll && cx->abort_poll(cx->abort_poll_user)) return true;
    return false;
}

static bool q35_is_eos(const cwick_model_spec *spec, uint32_t tok) {
    if (!spec) return false;
    for (int i = 0; i < spec->n_eos; i++)
        if ((int64_t)tok == spec->eos_ids[i]) return true;
    return (int64_t)tok == spec->eos_token_id;
}

static const char *q35_phase_label(const struct q35_ctx *cx, char *buf,
                                   size_t cap) {
    if (cx && cx->decode_phase)
        snprintf(buf, cap, "decode(%s)", cx->decode_phase);
    else
        snprintf(buf, cap, "decode");
    return buf;
}

static uint64_t q35_sampler_seed(void) {
    static uint64_t cached = 0;
    static bool read = false;
    if (!read) {
        read = true;
        const char *e = getenv("CWICK_SEED");
        if (e && *e) cached = strtoull(e, NULL, 0);
    }
    return cached;
}

/* CWICK_Q35_VERIFY=k turns each decode step into a k-row verify: one weight
 * pass producing k distributions, of which only the first is used. There is no
 * drafter here, so the text is nonsense by construction -- what this measures
 * is the cost of the verify itself, which is the half of speculative decoding
 * that costs anything. 0 (the default) is the ordinary single-row step. */
static int q35_verify_rows(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("CWICK_Q35_VERIFY");
        int v = e && *e ? atoi(e) : 0;
        if (v < 0) v = 0;
        if (v > Q35_GK) v = Q35_GK;
        cached = v;
    }
    return cached;
}

/* CWICK_Q35_SPEC=k turns each decode step into k drafts plus one (k+1)-row
 * verify, accepted greedily. 0 (the default) is the ordinary single-row step,
 * so k=1 -- one proposal, two rows -- is the arm that prices the drafter on
 * its own.
 *
 * Two drafters exist. The learned one is the checkpoint's own MTP block
 * (`q35_mtp_draft`), which costs 1.6 ms per proposal and is right often enough
 * to pay for it. The free one is prompt lookup -- the tokens that followed the
 * most recent earlier occurrence of the last few tokens -- which costs nothing
 * and, measured, returns 1.1 tokens per pass; it stays as the fallback for a
 * checkpoint with no MTP head and as the A/B baseline. CWICK_Q35_MTP=0 forces
 * it even when a head is present. */
static int q35_spec_rows(void) {
    static int cached = -2;
    if (cached < 0) {
        const char *e = getenv("CWICK_Q35_SPEC");
        int v = e && *e ? atoi(e) : 0;
        if (v < 0) v = 0;
        if (v > Q35_GK - 1) v = Q35_GK - 1;
        cached = v;
    }
    return cached;
}

/* Which copy of the text model's hidden state the block's fusion consumes: the
 * residual stream (0) or the final-normed copy the vocabulary head sees (1).
 * The checkpoint does not say, so it was measured -- 39.0% against 35.7% on
 * the ocean paragraph, same acceptance at depth 2 -- and 1 is the default now.
 * CWICK_Q35_MTP_POSTNORM=0 puts it back for the A/B. */
static int mtp_hpost(void) {
    static int cached = -2;
    if (cached < 0) {
        const char *e = getenv("CWICK_Q35_MTP_POSTNORM");
        cached = (e && *e)
                     ? (cwick_env_true("CWICK_Q35_MTP_POSTNORM") ? 1 : 0)
                     : 1;
    }
    return cached;
}

/* CWICK_Q35_MTP_FULL=1 projects a draft through every vocabulary row instead
 * of a candidate subset. It exists only to price the subset: if acceptance is
 * the same either way, the 2.03 GiB head was never needed for a draft. */

static int q35_mtp_full(void) {
    static int cached = -1;
    if (cached < 0) cached = cwick_env_true("CWICK_Q35_MTP_FULL") ? 1 : 0;
    return cached;
}

/* Candidate rows for a draft's output projection: the distinct tokens this
 * context has used recently.
 *
 * The head is 2.03 GiB and a draft needs a ranking, not a distribution, so
 * narrowing it is a bandwidth decision -- the verify decides what is emitted,
 * so a wrong draft costs a verification row and nothing else. What the set
 * buys is acceptance on the text where speculation pays at all: lists, code,
 * quoting, structured output -- anything where the next token has already
 * appeared nearby. Cold prose is where it gives up, and that is measurable:
 * on the 120-token ocean paragraph the last 2048 tokens hold only ~60 distinct
 * ids, which is a set the next token leaves about two thirds of the time. So
 * half the budget comes from the previous step's own logits instead, which is
 * what §64's "cheap way to recover most of that" turned into.
 * CWICK_Q35_CAND widens the whole set (a row costs 8 KB of head, so 16384 rows
 * is 0.45 ms per draft against 1.45 for 4096). */
#define Q35_CAND_WINDOW 2048
/* How far below the top logit a token may sit and still be worth ranking.
 * Threshold rather than top-k: a sharp distribution contributes a handful of
 * rows, a flat one is allowed to fill its half of the budget, and neither case
 * needs a sort. */
static float q35_cand_t(void) {
    static float cached = -1.0f;
    if (cached < 0.0f) {
        const char *e = getenv("CWICK_Q35_CAND_T");
        cached = (e && *e) ? (float)atof(e) : 2.5f;
        if (cached <= 0.0f) cached = 2.5f;
    }
    return cached;
}

/* How many rows the logits half may contribute. A draft pays for every row it
 * ranks (8 KB of head each), so this is the knob that decides whether the
 * shortlist is worth having at all: 2048 rows cost 4 ms and bought 2.6 points
 * of acceptance on prose, which is a loss; 512 rows cost 1 ms. */
static int q35_cand_lcap(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("CWICK_Q35_CAND_LCAP");
        int v = e && *e ? atoi(e) : 0;
        cached = v > 0 ? v : 512;
    }
    return cached;
}

static int q35_cand_cap(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("CWICK_Q35_CAND");
        int v = e && *e ? atoi(e) : 0;
        if (v <= 0) v = 4096;
        if (v > 262144) v = 262144;
        cached = v;
    }
    return cached;
}

static int q35_candidates(const int32_t *hist, int64_t n, const float *logits,
                          int vocab, uint32_t *out, int cap, uint8_t *seen) {
    if (!out || cap <= 0 || !seen) return 0;
    int got = 0;
    /* Half the budget is the distinct ids of the recent window. */
    const int hist_cap = cap / 2;
    if (hist && n > 0) {
        int64_t lo = n - Q35_CAND_WINDOW;
        if (lo < 0) lo = 0;
        for (int64_t i = n - 1; i >= lo && got < hist_cap; i--) {
            const int32_t t = hist[i];
            if (t < 0) continue;
            if (seen[t]) continue;
            seen[t] = 1;
            out[got++] = (uint32_t)t;
        }
    }
    /* The other half is the previous step's own shortlist: every row whose
     * logit is within T of the max. That is the only thing in the process that
     * knows what the text model was considering, and it is already on the host
     * -- the verify pass copied it there. */
    if (logits && vocab > 0 && got < cap) {
        float mx = logits[0];
        for (int i = 1; i < vocab; i++)
            if (logits[i] > mx) mx = logits[i];
        int room = cap - got;
        if (room > q35_cand_lcap()) room = q35_cand_lcap();
        float t = q35_cand_t();
        for (int attempt = 0; attempt < 4; attempt++, t *= 0.5f) {
            const float cut = mx - t;
            int c = 0;
            for (int i = 0; i < vocab; i++)
                if (logits[i] >= cut) c++;
            if (c > room) continue;
            for (int i = 0; i < vocab && got < cap; i++) {
                if (logits[i] < cut || seen[i]) continue;
                seen[i] = 1;
                out[got++] = (uint32_t)i;
            }
            break;
        }
    }
    for (int i = 0; i < got; i++) seen[out[i]] = 0;
    return got;
}

/* Prompt lookup: find an earlier occurrence of the last `order` tokens of
 * `hist[0..n)` and return up to `cap` tokens that followed it.
 *
 * Longest key first, so a repeated block of code continues instead of matching
 * a coincidental 2-gram. Among occurrences of the same key the newest one is
 * usually the better predictor -- except when it sits at the very end of the
 * history, which is exactly what a repeating tail does: its continuation is
 * empty and the draft would be nothing. So a match with at least `WANT` tokens
 * behind it wins over a newer one without; an occurrence's tail only shortens
 * as it gets newer, which is why one pass in order is enough.
 *
 * The scan is bounded to the last SCN tokens of history. Repetition that
 * matters is local, and an unbounded scan at a 128K context would cost more
 * than the row it saves.
 *
 * Returns 0 when nothing matched, which is not an error -- an unmatched step
 * simply verifies one row. */
#define Q35_LOOKUP_WANT 4
#define Q35_LOOKUP_SCAN 8192

/* Test hook: corrupt the n-th draft so the verify has to reject from there on.
 * A prompt-lookup drafter on a repetitive run is accepted every time, which
 * exercises only the "all rows kept" path -- and that path is the one that
 * needs no rewind. This forces the interesting case and lets the output be
 * compared against a run with speculation off: a rejected draft must leave the
 * same tokens and the same state behind. */
static int q35_spec_poison(void) {
    static int cached = -2;
    if (cached < 0) {
        const char *e = getenv("CWICK_Q35_SPEC_POISON");
        cached = e && *e ? atoi(e) : -1;
    }
    return cached;
}

static int q35_lookup_draft(const int32_t *hist, int64_t n, int32_t *out,
                            int cap) {
    if (!hist || n < 3 || !out || cap <= 0) return 0;
    int want = cap < Q35_LOOKUP_WANT ? cap : Q35_LOOKUP_WANT;
    int64_t order = 4;
    if (order > n - 1) order = n - 1;
    const int64_t lo = n - Q35_LOOKUP_SCAN > 0 ? n - Q35_LOOKUP_SCAN : 0;
    for (; order >= 2; order--) {
        const int64_t last = n - order - 1;
        const int32_t *key = hist + n - order;
        int64_t pick_s = -1, newest = -1;
        for (int64_t s = lo; s <= last; s++) {
            if (memcmp(hist + s, key, (size_t)order * sizeof(int32_t)) != 0)
                continue;
            newest = s;
            if (n - (s + order) >= want) pick_s = s;
        }
        if (pick_s < 0) pick_s = newest;
        if (pick_s < 0) continue;
        const int64_t from = pick_s + order;
        int take = 0;
        while (take < cap && from + take < n) {
            out[take] = hist[from + take];
            take++;
        }
        if (take > 0) return take;
    }
    return 0;
}

/* The draft check compares a row against the token a plain argmax would pick,
 * which is exactly what the sampler does when cwick_sampler_plain_argmax holds.
 * Left as a local scan: it runs k times per step over a 248320-wide row, which
 * is 1 us against a 61 ms step. */
static uint32_t q35_argmax_of(const float *x, size_t n) {
    uint32_t best = 0;
    float bv = n ? x[0] : 0.0f;
    for (size_t i = 1; i < n; i++)
        if (x[i] > bv) { bv = x[i]; best = (uint32_t)i; }
    return best;
}

static double q35_progress_min_ms(void) {
    static double cached = -1.0;
    if (cached < 0.0) {
        const char *e = getenv("CWICK_PROGRESS_MIN_MS");
        double v = e ? atof(e) : 0.0;
        cached = v > 0.0 ? v : 5000.0;
    }
    return cached;
}

int q35_engine_open(q35_engine *qe, const char *root, int64_t capacity,
                    cwick_error *err) {
    memset(qe, 0, sizeof(*qe));
    qe->capacity = capacity > 0 ? capacity : 4096;
    if (capacity > 0 && capacity > INT32_MAX) capacity = INT32_MAX;
    if (q35_model_open(&qe->m, root, (int)qe->capacity, err) != 0) return -1;
    qe->capacity = qe->m.cfg.max_seq_len;
    qe->logits = cwick_xmalloc(sizeof(float) * (size_t)qe->m.cfg.vocab_size);
    qe->pos = 0;
    qe->opened = 1;
    cwick_sampler_cfg_default(&qe->smp);
    /* 0 is a legal budget ("do not think"), so disabled has to be -1. */
    qe->ctx.think_budget = -1;
    qe->ctx.think_open_id = -1;
    qe->ctx.think_close_id = -1;
    qe->ctx.think_open_at_start = false;
    return 0;
}

void q35_engine_close(q35_engine *qe) {
    if (!qe || !qe->opened) return;
    free(qe->logits);
    qe->logits = NULL;
    q35_model_close(&qe->m);
    qe->opened = 0;
}

void q35_engine_reset(q35_engine *qe) {
    if (!qe || !qe->opened) return;
    const q35_config *c = &qe->m.cfg;
    qe->pos = 0;
    for (int l = 0; l < c->n_layers; l++) {
        if (qe->m.kcache[l])
            memset(qe->m.kcache[l], 0, (size_t)c->max_seq_len *
                   (size_t)c->n_kv_heads * (size_t)c->head_dim * sizeof(float));
        if (qe->m.vcache[l])
            memset(qe->m.vcache[l], 0, (size_t)c->max_seq_len *
                   (size_t)c->n_kv_heads * (size_t)c->head_dim * sizeof(float));
        if (qe->m.conv_state[l]) {
            int64_t qkv_dim = 2 * (int64_t)c->lin_k_heads * c->lin_k_dim +
                              (int64_t)c->lin_v_heads * c->lin_v_dim;
            int taps = c->conv_kernel > 0 ? c->conv_kernel : 4;
            memset(qe->m.conv_state[l], 0,
                   (size_t)(taps - 1) * (size_t)qkv_dim * sizeof(float));
        }
        if (qe->m.ssm_state[l])
            memset(qe->m.ssm_state[l], 0, (size_t)c->lin_v_heads *
                   (size_t)c->lin_k_dim * (size_t)c->lin_v_dim * sizeof(float));
    }
    /* The draft head's cache describes tokens of the sequence that just
     * ended, so it goes with the rest of the state. It is only `mtp_win` rows
     * deep -- a sliding window, not the whole context -- so the length here is
     * the window and not max_seq_len. Clearing max_seq_len rows overran the
     * allocation by 256x and took the process out with SIGBUS on the first
     * request of every serve session (KERN_PROTECTION_FAILURE inside bzero). */
    if (qe->m.mtp_kcache && qe->m.mtp_win > 0)
        memset(qe->m.mtp_kcache, 0, (size_t)qe->m.mtp_win *
               (size_t)c->n_kv_heads * (size_t)c->head_dim * sizeof(float));
    if (qe->m.mtp_vcache && qe->m.mtp_win > 0)
        memset(qe->m.mtp_vcache, 0, (size_t)qe->m.mtp_win *
               (size_t)c->n_kv_heads * (size_t)c->head_dim * sizeof(float));
    qe->m.mtp_n = 0;
    qe->m.mtp_lo = 0;
    qe->m.mtp_base = 0;
}

cwick_status q35_engine_generate(q35_engine *qe, const uint32_t *prompt,
                                 int64_t prompt_len, int64_t max_new,
                                 bool keep_prompt, cwick_token_sink sink,
                                 void *user, cwick_gen_result *result,
                                 cwick_error *err) {
    cwick_error_clear(err);
    if (!qe || !qe->opened) {
        cwick_error_set(err, CWICK_ERR_STATE, "qwen3_5 engine is not open");
        return CWICK_ERR_STATE;
    }
    const struct q35_ctx *cx = &qe->ctx;
    const cwick_model_spec *spec = cx ? cx->spec : NULL;
    const char *tag = (cx && cx->tag) ? cx->tag : "gen";
    const q35_config *c = &qe->m.cfg;

    qe->ctx.decode_phase = NULL;
    qe->next_checkpoint = qe->checkpoint_block > 0 ? qe->checkpoint_block : 0;

    /* max_new is a ceiling, not a demand: a client asking for 32K against a
     * 32K server with an 8K prompt is asking for "as much as fits". */
    int64_t total_new = max_new;
    {
        const int64_t want = max_new;
        cwick_status wst = cwick_gen_window(qe->pos, prompt_len, &max_new,
                                            qe->capacity, err);
        if (wst != CWICK_OK) return wst;
        if (max_new != want) {
            total_new = 0;
            CWICK_INFO(tag, "max_tokens %lld clamped to %lld: prompt %lld at "
                       "position %lld leaves %lld of %lld positions",
                       (long long)want, (long long)max_new,
                       (long long)prompt_len, (long long)qe->pos,
                       (long long)max_new, (long long)qe->capacity);
        }
    }

    int32_t *ids = cwick_xmalloc((size_t)(prompt_len > 1 ? prompt_len : 1) *
                                 sizeof(int32_t));
    for (int64_t i = 0; i < prompt_len; i++) ids[i] = (int32_t)prompt[i];

    /* Verify and speculative modes only allocate when they are on; the
     * ordinary path is unchanged. */
    const int vrow = q35_verify_rows();
    const int spec_k = vrow > 1 ? 0 : q35_spec_rows();
    const int vmax = spec_k > 0 ? spec_k + 1 : vrow;
    int32_t *vids = NULL;
    float *vall = NULL;
    int32_t *hist = NULL;    /* prompt + everything emitted, for the lookup */
    int32_t *pend = NULL;    /* verified but not yet emitted */
    uint32_t *cand = NULL;   /* draft head's candidate rows */
    uint8_t *seen = NULL;
    int pend_n = 0, pend_i = 0;
    int64_t hist_n = 0;
    /* Which row of the last multi-row pass holds the hidden state of the last
     * accepted position -- what the MTP block needs to propose the next token.
     * -1 means "not on the device", which is the state after a CPU prefill and
     * after a prefix-cache restore. */
    int hrow = -1;
    const int mtp_on = spec_k > 0 && q35_mtp_available(&qe->m);
    if (vmax > 1) {
        vids = cwick_xmalloc((size_t)vmax * sizeof(int32_t));
        vall = cwick_xmalloc((size_t)vmax * (size_t)c->vocab_size *
                             sizeof(float));
    }
    if (vrow > 1)
        CWICK_INFO(tag, "verify mode: %d rows per step, no drafter "
                   "(CWICK_Q35_VERIFY) -- timings are real, text is not", vrow);
    if (spec_k > 0) {
        hist = cwick_xmalloc((size_t)(prompt_len + max_new + 1) *
                             sizeof(int32_t));
        for (int64_t i = 0; i < prompt_len; i++) hist[i] = (int32_t)prompt[i];
        hist_n = prompt_len;
        pend = cwick_xmalloc((size_t)vmax * sizeof(int32_t));
        if (mtp_on && !q35_mtp_full()) {
            cand = cwick_xmalloc((size_t)q35_cand_cap() * sizeof(uint32_t));
            seen = cwick_xmalloc((size_t)c->vocab_size);
            memset(seen, 0, (size_t)c->vocab_size);
        }
        CWICK_INFO(tag, "speculative: up to %d drafts per step from %s, greedy "
                   "verify over %d rows (CWICK_Q35_SPEC)", spec_k,
                   mtp_on ? "the MTP head" : "prompt lookup", vmax);
    }

    /* ---- prefill ---- */
    double prefill_ms = 0.0;
    int64_t done = 0;
    mm_prefill_progress_begin(prompt_len, c->n_layers);
    while (done < prompt_len) {
        if (cx && q35_stopped(cx)) {
            mm_prefill_progress_end();
            free(ids);
            cwick_error_set(err, CWICK_ERR_STATE, "interrupted during prefill");
            return CWICK_ERR_STATE;
        }
        int n = (int)(prompt_len - done);
        if (n > Q35_CHUNK) n = Q35_CHUNK;
        double t0 = q35_now_ms();
        if (q35_model_forward(&qe->m, ids + done, n, qe->pos, qe->logits,
                              err) != 0) {
            mm_prefill_progress_end();
            free(ids);
            return CWICK_ERR_STATE;
        }
        prefill_ms += q35_now_ms() - t0;
        qe->pos += n;
        done += n;
        mm_prefill_progress_done(done);
        if (qe->on_prefill_checkpoint && qe->checkpoint_block > 0 &&
            qe->pos >= qe->next_checkpoint) {
            qe->on_prefill_checkpoint(qe->on_prefill_checkpoint_user, qe->pos);
            qe->next_checkpoint = qe->pos + qe->checkpoint_block;
        }
    }
    if (prompt_len == 0) {
        ids[0] = 0;
        double t0 = q35_now_ms();
        if (q35_model_forward(&qe->m, ids, 1, qe->pos, qe->logits, err) != 0) {
            mm_prefill_progress_end();
            free(ids);
            return CWICK_ERR_STATE;
        }
        prefill_ms += q35_now_ms() - t0;
        qe->pos += 1;
    }
    mm_prefill_progress_end();
    if (qe->on_prefill_done) qe->on_prefill_done(qe->on_prefill_done_user, qe->pos);

    int64_t collected = keep_prompt ? prompt_len : 0;
    uint32_t *tokens = cwick_xmalloc((size_t)(collected + max_new + 1) *
                                     sizeof(uint32_t));
    if (keep_prompt && prompt_len > 0)
        memcpy(tokens, prompt, (size_t)prompt_len * sizeof(uint32_t));

    int64_t gen = 0, decode_steps = 0;
    bool hit_eos = false, interrupted = false, sink_stopped = false;
    bool loop_stopped = false;
    double decode_t0 = q35_now_ms();
    double last_rep = decode_t0;
    int64_t steps_accounted = 0, gen_accounted = 0;
    int64_t spec_drafts = 0, spec_taken = 0;
    /* Acceptance by depth: depth d is only reached when every proposal before
     * it was accepted, so the ratio here is P(proposal d | prefix 0..d-1) --
     * the number that says whether a chain is worth running or whether only
     * the first link pays. */
    int64_t spec_try[Q35_GK], spec_at[Q35_GK];
    memset(spec_try, 0, sizeof(spec_try));
    memset(spec_at, 0, sizeof(spec_at));

    const int64_t think_open_id = cx ? cx->think_open_id : -1;
    const int64_t think_close_id = cx ? cx->think_close_id : -1;
    int64_t think_budget = cx ? cx->think_budget : -1;
    if (think_budget >= max_new) {
        int64_t room = max_new - CWICK_THINK_RESERVE;
        if (room < 0) room = max_new / 2;
        if (cx && cx->verbose)
            CWICK_VERB(tag, "thinking budget %lld clamped to %lld: %lld tokens "
                       "of output leave no room to answer",
                       (long long)(cx->think_budget), (long long)room,
                       (long long)max_new);
        think_budget = room;
    }
    bool think_open = cx ? cx->think_open_at_start : false;
    bool think_closed = cx ? cx->think_closed_at_start : false;
    bool think_budget_hit = false;

    cwick_sampler smp;
    cwick_sampler_init(&smp, &qe->smp, c->vocab_size, q35_sampler_seed());
    if (smp.counts)
        for (int64_t i = 0; i < prompt_len; i++)
            cwick_sampler_prime(&smp, prompt[i]);

    for (int64_t k = 0; k < max_new; k++) {
        if (cx && q35_stopped(cx)) { interrupted = true; break; }

        /* ---- thinking budget ---- */
        if (think_budget >= 0 && !think_closed) {
            cwick_think_action act =
                cwick_think_plan(gen, think_budget, think_open, false);
            if (act != CWICK_THINK_NONE) {
                uint32_t inj[2];
                int n_inj = 0;
                if (act == CWICK_THINK_OPEN_CLOSE)
                    inj[n_inj++] = (uint32_t)think_open_id;
                inj[n_inj++] = (uint32_t)think_close_id;
                bool aborted = false;
                int written = 0;
                for (int t = 0; t < n_inj && gen < max_new; t++) {
                    uint32_t ct = inj[t];
                    tokens[collected + gen++] = ct;
                    written++;
                    cwick_sampler_accept(&smp, ct);
                    if ((int64_t)ct == think_open_id) think_open = true;
                    if ((int64_t)ct == think_close_id) think_closed = true;
                    if (sink && !sink(user, ct, gen - 1)) {
                        sink_stopped = true;
                        aborted = true;
                        break;
                    }
                    if (k + 1 < max_new) {
                        ids[0] = (int32_t)ct;
                        if (q35_model_forward(&qe->m, ids, 1, qe->pos,
                                              qe->logits, err) != 0) {
                            cwick_sampler_free(&smp);
                            free(tokens);
                            free(ids);
                            return CWICK_ERR_STATE;
                        }
                        qe->pos++;
                        decode_steps++;
                        /* A one-row forward leaves its hidden state in row 0;
                         * anything larger is stale from the last verify. */
                        hrow = 0;
                    }
                }
                if (written) {
                    think_budget_hit = true;
                    CWICK_INFO(tag, "thinking budget: %lld tokens reached — "
                              "wrote </think> so the turn answers instead of "
                              "thinking", (long long)think_budget);
                }
                if (aborted) break;
                if (gen >= max_new) break;
                continue;
            }
        }

        /* A queued token was already verified and already counted into the
         * sampler when it was accepted, so it is emitted without running the
         * model again -- the distribution for the token after it is the row the
         * verify used to decide it, which is sitting in qe->logits. */
        bool from_pend = pend_i < pend_n;
        uint32_t next = from_pend ? (uint32_t)pend[pend_i++]
                                  : cwick_sampler_pick(&smp, qe->logits,
                                                       c->vocab_size);
        if (spec && q35_is_eos(spec, next)) { hit_eos = true; break; }
        tokens[collected + gen++] = next;
        if (hist) hist[hist_n++] = (int32_t)next;
        if (!from_pend) cwick_sampler_accept(&smp, next);
        if ((int64_t)next == think_open_id) think_open = true;
        if ((int64_t)next == think_close_id) think_closed = true;
        int lp = 0;
        int64_t lr = 0;
        if (cwick_sampler_looped(&smp, &lp, &lr)) {
            gen--;
            CWICK_INFO(tag, "loop guard: period-%d block repeated %lld times "
                      "(%lld tokens) — stopping after %lld generated; "
                      "CWICK_LOOP_DETECT=0 disables", lp,
                      (long long)(lr / (lp > 0 ? lp : 1) + 1), (long long)lr,
                      (long long)gen);
            loop_stopped = true;
            break;
        }
        if (sink && !sink(user, next, gen - 1)) { sink_stopped = true; break; }

        if (from_pend) continue;
        if (k + 1 >= max_new) break;

        double step_t0 = q35_now_ms();
        int rows = 1;
        int nd = 0, naccepted = 0;
        if (spec_k > 0) {
            int m = spec_k;
            const int64_t room = max_new - 1 - gen;
            if ((int64_t)m > room) m = (int)(room > 0 ? room : 0);
            if (qe->pos + m + 1 > qe->capacity)
                m = (int)(qe->capacity - qe->pos - 1);
            if (m > 0) {
                if (mtp_on && hrow >= 0) {
                    /* The block sits one position past the accepted prefix:
                     * hrow holds h_{P-1} (the last accepted position) and
                     * `next` is the token that goes at P, so the block's query
                     * lives at P -- `qe->pos`.
                     *
                     * Before drafting, hand the block the positions the text
                     * model has already decided on: rows 0..hrow now describe
                     * accepted text, and `nxt[r]` is the token that follows row
                     * r's position, `next` included. Without this the only keys
                     * in the cache are ones earlier drafts guessed, and the
                     * softmax at the first proposal has a single key. */
                    int32_t nxt[Q35_GK + 1];
                    const int64_t lo = qe->pos - hrow - 1;
                    for (int r = 0; r < hrow; r++)
                        nxt[r] = (lo + r < hist_n) ? hist[lo + r] : (int32_t)next;
                    nxt[hrow] = (int32_t)next;
                    (void)q35_mtp_sync(&qe->m, hrow, mtp_hpost(), qe->pos, nxt);
                    int ncand = 0;
                    if (cand) {
                        ncand = q35_candidates(hist, hist_n, qe->logits,
                                               (int)c->vocab_size, cand,
                                               q35_cand_cap(), seen);
                        if (ncand < 8) ncand = 0;   /* too thin: use every row */
                    }
                    nd = q35_mtp_draft(&qe->m, hrow, mtp_hpost(), (int32_t)next,
                                       qe->pos, m, cand, ncand, vids + 1);
                } else if (!mtp_on) {
                    nd = q35_lookup_draft(hist, hist_n, vids + 1, m);
                }
            }
        }
        if (nd > 0) {
            /* One weight pass over nd+1 positions: the row that decided each
             * draft and the row after the last one. */
            const int n = nd + 1;
            const size_t V = (size_t)c->vocab_size;
            vids[0] = (int32_t)next;
            {
                const int pi = q35_spec_poison();
                if (pi >= 0 && pi < nd)
                    vids[pi + 1] = (int32_t)((vids[pi + 1] + 1) %
                                             c->vocab_size);
            }
            int bad = q35_model_verify_rows(&qe->m, vids, n, qe->pos, vall,
                                            err) != 0;
            if (bad) {
                cwick_sampler_free(&smp);
                free(tokens);
                free(ids);
                free(vids);
                free(vall);
                free(hist);
                free(pend);
                free(cand);
                free(seen);
                return CWICK_ERR_STATE;
            }
            /* Accept the longest prefix of the draft the model agrees with.
             * Each accepted token is fed to the sampler *now*, so the next
             * comparison sees the state the live path would have had -- that is
             * what makes an armed loop-breaker end the run at the same place.
             * An eos draft is left for the normal path to emit, so the stop is
             * recorded with the position it actually happened at. */
            int j = 0;
            for (; j < nd; j++) {
                if (!cwick_sampler_plain_argmax(&smp)) break;
                uint32_t got = q35_argmax_of(vall + (size_t)j * V, V);
                if ((int32_t)got != vids[j + 1]) break;
                if (spec && q35_is_eos(spec, got)) break;
                cwick_sampler_accept(&smp, got);
                /* The loop guard is consulted once per produced token, and it
                 * latches: the first time it fires, pick() stops applying the
                 * repetition ban. Polling it here, in the same place the live
                 * path does, keeps that latch on the same token -- skipping it
                 * would leave the ban armed for a row the live path would have
                 * already unmasked. */
                int lp2 = 0;
                int64_t lr2 = 0;
                if (cwick_sampler_looped(&smp, &lp2, &lr2)) break;
            }
            /* The rows that were not kept still advanced the recurrence, so
             * point it back at the slot that belongs to the accepted prefix. */
            if (q35_model_keep_rows(&qe->m, j) != 0) {
                cwick_sampler_free(&smp);
                free(tokens);
                free(ids);
                free(vids);
                free(vall);
                free(hist);
                free(pend);
                free(cand);
                free(seen);
                cwick_error_set(err, CWICK_ERR_STATE,
                                "q35 verify could not keep row %d of %d",
                                j, n);
                return CWICK_ERR_STATE;
            }
            for (int q2 = 0; q2 < j; q2++) pend[q2] = vids[q2 + 1];
            pend_n = j;
            pend_i = 0;
            naccepted = j;
            spec_drafts += nd;
            spec_taken += j;
            for (int i2 = 0; i2 < nd; i2++) {
                if (i2 < j) { spec_try[i2]++; spec_at[i2]++; }
                else if (i2 == j) { spec_try[i2]++; }
                else break;
            }
            /* `next` plus the j accepted drafts are the positions this step
             * keeps; row j holds the distribution for the one after them. */
            rows = j + 1;
            hrow = j;
            memcpy(qe->logits, vall + (size_t)j * V, V * sizeof(float));
        } else if (vrow > 1 && qe->pos + vrow <= qe->capacity) {
            for (int r = 0; r < vrow; r++) vids[r] = (int32_t)next;
            if (q35_model_forward_all(&qe->m, vids, vrow, qe->pos, vall,
                                      err) != 0) {
                cwick_sampler_free(&smp);
                free(tokens);
                free(ids);
                free(vids);
                free(vall);
                free(hist);
                free(pend);
                free(cand);
                free(seen);
                return CWICK_ERR_STATE;
            }
            memcpy(qe->logits, vall, (size_t)c->vocab_size * sizeof(float));
            rows = vrow;
            hrow = vrow - 1;
        } else {
            /* `ids` still holds the prompt, so the token to forward has to be
             * written in: the decode step consumes the one that was just
             * picked, not prompt[0]. Feeding the prompt back made the model
             * run on its own first token for the whole generation, which
             * degenerated into a repetition within a dozen tokens. */
            ids[0] = (int32_t)next;
            if (q35_model_forward(&qe->m, ids, 1, qe->pos, qe->logits,
                                  err) != 0) {
                cwick_sampler_free(&smp);
                free(tokens);
                free(ids);
                free(vids);
                free(vall);
                free(hist);
                free(pend);
                free(cand);
                free(seen);
                return CWICK_ERR_STATE;
            }
            hrow = 0;
        }
        qe->pos += rows;
        decode_steps++;
        double step_ms = q35_now_ms() - step_t0;

        char phase_lab[32];
        q35_phase_label(cx, phase_lab, sizeof(phase_lab));
        if (spec_k > 0 && cx && cx->verbose && nd > 0) {
            CWICK_VERB(tag, "%s spec %d drafts, %d accepted, %d rows %.0f ms "
                       "(%.2f tok/s this step)", phase_lab, nd, naccepted,
                       nd + 1, step_ms,
                       step_ms > 0 ? (double)rows * 1000.0 / step_ms : 0.0);
        }
        if (vrow > 1 && cx && cx->verbose) {
            CWICK_VERB(tag, "%s verify %d rows %.0f ms: %.2f tok/s if 1 row "
                       "accepted, %.2f at %d", phase_lab, rows, step_ms,
                       step_ms > 0 ? 1000.0 / step_ms : 0.0,
                       step_ms > 0 ? (double)rows * 1000.0 / step_ms : 0.0,
                       rows);
        }
        if (cx && cx->verbose) {
            if (total_new > 0)
                CWICK_VERB(tag, "%s forward %lld/%lld %.0f ms (%.3f tok/s)",
                           phase_lab, (long long)decode_steps,
                           (long long)(total_new - 1), step_ms,
                           step_ms > 0 ? 1000.0 / step_ms : 0.0);
            else
                CWICK_VERB(tag, "%s forward %lld %.0f ms (%.3f tok/s, context "
                           "cap)", phase_lab, (long long)decode_steps, step_ms,
                           step_ms > 0 ? 1000.0 / step_ms : 0.0);
        }
        if ((q35_now_ms() - last_rep) >= q35_progress_min_ms()) {
            double el = q35_now_ms() - decode_t0;
            double per = decode_steps > 0 ? el / (double)decode_steps : 0.0;
            /* Tokens over wall time, not steps: with a speculative step those
             * differ by the acceptance rate, and the token rate is the number
             * that matters. */
            double tps = el > 0.0 ? (double)gen * 1000.0 / el : 0.0;
            last_rep = q35_now_ms();
            if (total_new > 0) {
                double eta = tps > 0.0
                                 ? (double)(total_new - 1 - gen) * 1000.0 / tps
                                 : 0.0;
                CWICK_INFO(tag, "%s %lld/%lld tokens (%.1f%%) — %.0f ms/step, "
                           "%.3f tok/s, eta %.0fs", phase_lab, (long long)gen,
                           (long long)total_new,
                           100.0 * (double)gen / (double)total_new,
                           per, tps, eta / 1000.0);
            } else {
                CWICK_INFO(tag, "%s %lld tokens (context cap) — %.0f ms/step, "
                           "%.3f tok/s", phase_lab, (long long)gen, per, tps);
            }
        }
        cwick_metrics_observe_decode_step(step_ms / 1000.0);
        cwick_metrics_observe_iteration_tokens(1);
        if ((decode_steps & 15) == 0) {
            cwick_metrics_add_decode_steps(decode_steps - steps_accounted);
            steps_accounted = decode_steps;
            if (gen > gen_accounted) {
                cwick_metrics_add_generation_tokens(gen - gen_accounted);
                gen_accounted = gen;
            }
        }
    }
    double decode_ms = q35_now_ms() - decode_t0;

    if (decode_steps > steps_accounted)
        cwick_metrics_add_decode_steps(decode_steps - steps_accounted);
    if (gen > gen_accounted)
        cwick_metrics_add_generation_tokens(gen - gen_accounted);

    {
        char pf_tps[32], dc_tps[32];
        if (prefill_ms > 0.0)
            snprintf(pf_tps, sizeof(pf_tps), "%.3f",
                     (double)prompt_len * 1000.0 / prefill_ms);
        else
            snprintf(pf_tps, sizeof(pf_tps), "n/a");
        if (decode_steps > 0 && decode_ms > 0.0)
            snprintf(dc_tps, sizeof(dc_tps), "%.3f",
                     (double)gen * 1000.0 / decode_ms);
        else
            snprintf(dc_tps, sizeof(dc_tps), "n/a");
        if (spec_k > 0 && decode_steps > 0)
            CWICK_INFO(tag, "drafts %lld, accepted %lld (%.1f%%), %.2f tokens "
                       "per weight pass", (long long)spec_drafts,
                       (long long)spec_taken,
                       spec_drafts > 0
                           ? 100.0 * (double)spec_taken / (double)spec_drafts
                           : 0.0,
                       (double)gen / (double)decode_steps);
        if (spec_k > 0 && decode_steps > 0) {
            char byd[160];
            size_t at = 0;
            byd[0] = '\0';
            for (int i2 = 0; i2 < Q35_GK && at + 24 < sizeof(byd); i2++) {
                if (spec_try[i2] <= 0) break;
                at += (size_t)snprintf(byd + at, sizeof(byd) - at, "%s%d:%.1f%%",
                                       i2 ? " " : "", i2 + 1,
                                       spec_try[i2] > 0
                                           ? 100.0 * (double)spec_at[i2] /
                                                 (double)spec_try[i2]
                                           : 0.0);
            }
            if (at > 0)
                CWICK_INFO(tag, "accept by depth %s", byd);
        }
        CWICK_INFO(tag, "prompt=%lld gen=%lld finish=%s prefill=%.0fms/%s "
                   "tok/s decode=%.0fms/%lld passes %s tok/s mem=%.2fGiB "
                   "cache=%.2fGiB",
                   (long long)prompt_len, (long long)gen,
                   interrupted ? "interrupted"
                               : loop_stopped ? "loop"
                               : ((hit_eos || sink_stopped) ? "stop" : "length"),
                   prefill_ms, pf_tps, decode_ms, (long long)decode_steps,
                   dc_tps, cwick_gib(cwick_process_footprint_bytes()),
                   cwick_gib(q35_cache_bytes(&qe->m)));
    }

    result->tokens = tokens;
    result->n_tokens = collected + gen;
    result->stop_reason_length =
        !hit_eos && !interrupted && !sink_stopped && !loop_stopped &&
        gen == max_new;
    result->prefill_ms = prefill_ms;
    result->decode_ms = decode_ms;
    result->decode_steps = decode_steps;
    result->interrupted = interrupted;
    result->sink_stopped = sink_stopped;
    result->loop_stopped = loop_stopped;
    result->think_budget_hit = think_budget_hit;
    result->think_budget_applied = think_budget;
    cwick_sampler_free(&smp);
    free(ids);
    free(vids);
    free(vall);
    free(hist);
    free(pend);
    free(cand);
    free(seen);
    if (interrupted) {
        cwick_error_set(err, CWICK_ERR_STATE, "generation interrupted");
        return CWICK_ERR_STATE;
    }
    return CWICK_OK;
}
