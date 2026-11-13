/* sample.c — temperature / top-k / top-p sampling, repetition penalties and
 * the loop guard that stops a decode that has fallen into a repetition
 * attractor.
 *
 * See sample.h for why this exists. Two properties are load-bearing:
 *
 *   - it never sorts the whole vocabulary (a 150144-way sort is ~10 ms, the
 *     decode step it would sit in is ~15 ms), so the candidate set is bounded
 *     and gathered with a bounded min-heap while the only full-vocabulary
 *     passes are max / exp / sum;
 *   - it is seed-reproducible, so a bug report that pins the seed pins the
 *     output.
 *
 * The loop guard adds a third: it costs one comparison per period per token
 * and no allocation, because the history it needs is a ring inside the sampler
 * and the check is against the tokens this run already produced. */
#include "sample.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <Accelerate/Accelerate.h>

#include "json.h"
#include "util.h"

/* ------------------------------------------------------------------ */
/* config                                                              */
/* ------------------------------------------------------------------ */

void cwick_sampler_cfg_default(cwick_sampler_cfg *c) {
    if (!c) return;
    c->temperature = 0.0f;
    c->top_p = 1.0f;
    c->top_k = 0;
    c->presence_penalty = 0.0f;
    c->frequency_penalty = 0.0f;
}

bool cwick_sampler_cfg_is_greedy(const cwick_sampler_cfg *c) {
    if (!c) return true;
    if (!(c->temperature > 0.0f)) return true;
    /* A distribution that is neither truncated nor penalised is still a
     * distribution: sampling from it is not argmax. */
    return false;
}

bool cwick_sampler_cfg_validate(const cwick_sampler_cfg *c, char *why, size_t cap) {
    if (!c) return true;
    if (c->temperature != c->temperature) {  /* NaN */
        if (why) snprintf(why, cap, "temperature must be a number");
        return false;
    }
    if (c->temperature < 0.0f || c->temperature > 2.0f) {
        if (why) snprintf(why, cap, "temperature must be in [0, 2]");
        return false;
    }
    if (c->top_p != c->top_p || c->top_p <= 0.0f || c->top_p > 1.0f) {
        if (why) snprintf(why, cap, "top_p must be in (0, 1]");
        return false;
    }
    if (c->top_k < 0) {
        if (why) snprintf(why, cap, "top_k must be >= 0");
        return false;
    }
    if (c->presence_penalty != c->presence_penalty ||
        c->presence_penalty < -2.0f || c->presence_penalty > 2.0f) {
        if (why) snprintf(why, cap, "presence_penalty must be in [-2, 2]");
        return false;
    }
    if (c->frequency_penalty != c->frequency_penalty ||
        c->frequency_penalty < -2.0f || c->frequency_penalty > 2.0f) {
        if (why) snprintf(why, cap, "frequency_penalty must be in [-2, 2]");
        return false;
    }
    return true;
}

void cwick_sampler_describe(const cwick_sampler_cfg *c, char *buf, size_t cap) {
    if (!buf || cap == 0) return;
    if (!c || cwick_sampler_cfg_is_greedy(c)) {
        snprintf(buf, cap, "greedy");
        return;
    }
    snprintf(buf, cap, "t=%.2f top_p=%.2f top_k=%d pen=%.2f/%.2f",
             (double)c->temperature, (double)c->top_p, c->top_k,
             (double)c->presence_penalty, (double)c->frequency_penalty);
}

/* What the checkpoint says about its own decoding. `generation_config.json`
 * is the HF-side half of the training contract: do_sample=false means the
 * reference implementation decodes it greedily, do_sample=true plus
 * temperature/top_p/top_k means it decodes by drawing. Reading it here is the
 * difference between "serve the model the way it was trained" and "serve it
 * with argmax and watch it repeat one clause for two thousand tokens".
 *
 * Returns false when there is no such file (or nothing usable in it), in
 * which case `out` is untouched — the caller's greedy default stands. */
bool cwick_sampler_cfg_from_checkpoint(const char *root, cwick_sampler_cfg *out) {
    if (!root || !out) return false;
    char path[1152];
    cwick_path_join(path, sizeof(path), root, "generation_config.json");
    char *text = NULL;
    size_t len = 0;
    if (cwick_read_file(path, &text, &len) != 0 || !text) return false;
    cwick_error err;
    cwick_error_clear(&err);
    json_value *v = cwick_json_parse(text, len, &err);
    free(text);
    if (!v || v->kind != JSON_OBJECT) {
        cwick_json_free(v);
        return false;
    }
    cwick_sampler_cfg c;
    cwick_sampler_cfg_default(&c);
    bool do_sample = cwick_json_bool(cwick_json_get(v, "do_sample"), false);
    json_value *t = cwick_json_get(v, "temperature");
    json_value *p = cwick_json_get(v, "top_p");
    json_value *k = cwick_json_get(v, "top_k");
    if (t && t->kind == JSON_NUMBER) c.temperature = (float)cwick_json_num(t, 0.0);
    else if (do_sample) c.temperature = 1.0f;   /* HF's default when sampling */
    if (!do_sample) c.temperature = 0.0f;       /* do_sample=false is explicit */
    if (p && p->kind == JSON_NUMBER) c.top_p = (float)cwick_json_num(p, 1.0);
    if (k && k->kind == JSON_NUMBER) c.top_k = (int)cwick_json_int(k, 0);
    cwick_json_free(v);
    if (!cwick_sampler_cfg_validate(&c, NULL, 0)) return false;
    *out = c;
    return true;
}

/* ------------------------------------------------------------------ */
/* rng                                                                 */
/* ------------------------------------------------------------------ */

/* splitmix64: small, no global state, and uniform enough for one draw per
 * decode step (a bad generator here would show up as a repeated-token bias,
 * not as a crash, so it is worth using one that is actually tested). */
static uint64_t sm64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* Uniform in [0,1): 53 bits is all a double can hold. */
static double rnd01(cwick_sampler *s) {
    return (double)(sm64(&s->rng) >> 11) * (1.0 / 9007199254740992.0);
}

/* ------------------------------------------------------------------ */
/* lifecycle                                                           */
/* ------------------------------------------------------------------ */

/* "0" / "false" / "no" / "off" are off; anything else that is set is on; unset
 * keeps `dflt`. (cwick_env_true only recognises the on-spellings, which is the
 * wrong shape for a knob that is on by default.) */
static bool loop_env_bool(const char *name, bool dflt) {
    const char *v = getenv(name);
    if (!v || !*v) return dflt;
    if (strcmp(v, "0") == 0 || strcasecmp(v, "false") == 0 ||
        strcasecmp(v, "no") == 0 || strcasecmp(v, "off") == 0)
        return false;
    return true;
}

static int64_t loop_env_int(const char *name, int64_t dflt) {
    const char *v = getenv(name);
    if (!v || !*v) return dflt;
    return strtoll(v, NULL, 0);
}

/* The guard's thresholds, read once per generation (a getenv per request is
 * nothing next to a prefill, and reading it per run keeps a test able to turn
 * the guard off for one case). */
static void loop_guard_cfg(cwick_sampler *s) {
    s->lon = loop_env_bool("CWICK_LOOP_DETECT", true);
    int64_t per = loop_env_int("CWICK_LOOP_MAX_PERIOD", CWICK_LOOP_MAX_PERIOD);
    if (per < 1) per = 1;
    /* The ring is sized for the compile-time maximum; a larger period would
     * read history it no longer has, so the knob can only lower it. */
    if (per > CWICK_LOOP_MAX_PERIOD) per = CWICK_LOOP_MAX_PERIOD;
    s->lper_max = (int)per;
    int64_t run = loop_env_int("CWICK_LOOP_MIN_RUN", CWICK_LOOP_MIN_RUN);
    if (run < 4) run = 4;
    if (run > 4096) run = 4096;
    s->lmin_run = run;
    int64_t occ = loop_env_int("CWICK_LOOP_MIN_OCC", CWICK_LOOP_MIN_OCC);
    if (occ < 2) occ = 2;
    if (occ > 64) occ = 64;
    s->lmin_occ = (int)occ;
    for (int p = 1; p <= CWICK_LOOP_MAX_PERIOD; p++) {
        int64_t need = (int64_t)(s->lmin_occ - 1) * p;
        if (need < s->lmin_run) need = s->lmin_run;
        s->lneed[p] = (int)need;
    }
    s->lhit_per = 0;
    s->lhit_run = 0;
}

static void loop_break_cfg(cwick_sampler *s) {
    s->lbrk = loop_env_bool("CWICK_LOOP_BREAK", true);
    int64_t mn = loop_env_int("CWICK_LOOP_BREAK_MIN", CWICK_LOOP_BREAK_MIN);
    if (mn < 4) mn = 4;
    if (mn > 256) mn = 256;
    for (int p = 1; p <= CWICK_LOOP_MAX_PERIOD; p++) {
        int64_t need = 2 * p;
        if (need < mn) need = mn;
        if (need > CWICK_LOOP_BREAK_MAX) need = CWICK_LOOP_BREAK_MAX;
        s->lbrk_need[p] = (int)need;
    }
    s->lbrk_steps = 0;
}

void cwick_sampler_init(cwick_sampler *s, const cwick_sampler_cfg *c,
                        int64_t vocab, uint64_t seed) {
    memset(s, 0, sizeof(*s));
    if (c) s->cfg = *c; else cwick_sampler_cfg_default(&s->cfg);
    s->vocab = vocab > 0 ? vocab : 0;
    loop_guard_cfg(s);
    loop_break_cfg(s);
    if (s->cfg.presence_penalty != 0.0f || s->cfg.frequency_penalty != 0.0f) {
        if (s->vocab > 0) {
            s->counts = cwick_xmalloc((size_t)s->vocab * sizeof(int32_t));
            memset(s->counts, 0, (size_t)s->vocab * sizeof(int32_t));
        } else
            s->cfg.presence_penalty = s->cfg.frequency_penalty = 0.0f;
    }
    if (seed != 0) {
        s->rng = seed;
    } else {
        /* Not reproducible on purpose: a caller that did not ask for a seed
         * gets a different run every time, which is the OpenAI reading of the
         * field. Mixing the clock with the pid and the object address keeps
         * two engines that start in the same millisecond apart. */
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        s->rng = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
        s->rng ^= (uint64_t)getpid() * 0x9E3779B97F4A7C15ULL;
        s->rng ^= (uint64_t)(uintptr_t)s >> 4;
    }
}

void cwick_sampler_free(cwick_sampler *s) {
    if (!s) return;
    free(s->counts);
    free(s->scratch);
    free(s->cand);
    free(s->bmass);
    free(s->bcnt);
    memset(s, 0, sizeof(*s));
}

/* One produced token into the guard: append it to the ring, then for every
 * period p extend or reset the count of trailing positions that repeat at that
 * period. The first period to reach its threshold wins and latches. */
static void loop_feed(cwick_sampler *s, uint32_t tok) {
    if (!s->lon || s->lhit_per) return;
    const int64_t H = CWICK_LOOP_HIST;
    const int64_t g = s->lgen;
    s->lhist[g % H] = tok;
    s->lgen = g + 1;
    if (g == 0) return;                    /* the first token has no partner */
    const uint32_t cur = s->lhist[g % H];
    for (int p = 1; p <= s->lper_max; p++) {
        if (p > g) break;                  /* not enough history for this p */
        if (cur != s->lhist[(g - p) % H]) { s->lmatch[p] = 0; continue; }
        if (++s->lmatch[p] >= s->lneed[p]) {
            s->lhit_per = p;
            s->lhit_run = s->lmatch[p];
            return;
        }
    }
}

void cwick_sampler_accept(cwick_sampler *s, uint32_t tok) {
    if (!s) return;
    /* The penalty table is optional; the guard is not, so it must not sit
     * behind the same early return. */
    if (s->counts && (int64_t)tok >= 0 && (int64_t)tok < s->vocab) {
        if (s->counts[tok] < INT32_MAX) s->counts[tok]++;
    }
    loop_feed(s, tok);
}

void cwick_sampler_prime(cwick_sampler *s, uint32_t tok) {
    if (!s || !s->counts) return;
    if ((int64_t)tok >= 0 && (int64_t)tok < s->vocab) {
        if (s->counts[tok] < INT32_MAX) s->counts[tok]++;
    }
}

bool cwick_sampler_looped(const cwick_sampler *s, int *period, int64_t *run) {
    if (!s || !s->lhit_per) return false;
    if (period) *period = s->lhit_per;
    if (run) *run = s->lhit_run;
    return true;
}

/* ------------------------------------------------------------------ */
/* candidate gathering (bounded min-heap, never a full sort)            */
/* ------------------------------------------------------------------ */

static void heap_up(cwick_cand *h, int64_t i) {
    while (i > 0) {
        int64_t p = (i - 1) / 2;
        if (h[p].val <= h[i].val) break;
        cwick_cand t = h[p]; h[p] = h[i]; h[i] = t;
        i = p;
    }
}

static void heap_down(cwick_cand *h, int64_t n, int64_t i) {
    for (;;) {
        int64_t l = 2 * i + 1, r = l + 1, m = i;
        if (l < n && h[l].val < h[m].val) m = l;
        if (r < n && h[r].val < h[m].val) m = r;
        if (m == i) break;
        cwick_cand t = h[i]; h[i] = h[m]; h[m] = t;
        i = m;
    }
}

/* The `size` largest (value, index) pairs, in the heap `h` of capacity `cap`.
 * One comparison per element for everything that is not in the running top
 * `cap`, which is why a 150144-vocabulary top-20 costs ~0.2 ms instead of the
 * ~10 ms a sort would. */
static int64_t gather_top(const float *x, int64_t n, int64_t cap, cwick_cand *h) {
    int64_t size = 0;
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        if (v != v) continue;            /* NaN is not a candidate */
        if (size < cap) {
            h[size].val = v;
            h[size].idx = (int32_t)i;
            heap_up(h, size);
            size++;
        } else if (v > h[0].val) {
            h[0].val = v;
            h[0].idx = (int32_t)i;
            heap_down(h, size, 0);
        }
    }
    return size;
}

static int cmp_cand_desc(const void *a, const void *b) {
    float va = ((const cwick_cand *)a)->val;
    float vb = ((const cwick_cand *)b)->val;
    if (va > vb) return -1;
    if (va < vb) return 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* pick                                                                */
/* ------------------------------------------------------------------ */

static uint32_t argmax_of(const float *x, int64_t n) {
    uint32_t best = 0;
    float bestv = -INFINITY;
    for (int64_t i = 0; i < n; i++) {
        if (x[i] > bestv) { bestv = x[i]; best = (uint32_t)i; }
    }
    return best;
}

/* Draw from the candidate set `h[0..size)` (sorted by value, descending),
 * whose un-normalised weights are `exp(val - max)`. The normaliser is summed
 * over the set being drawn from, so the draw is over that set's own
 * distribution — which is what "sample from the nucleus" means once the
 * nucleus has already been cut. */
static uint32_t draw(cwick_sampler *s, const cwick_cand *h, int64_t size,
                     double max, uint32_t fallback) {
    double total = 0.0;
    for (int64_t j = 0; j < size; j++) total += exp(h[j].val - max);
    /* Every candidate underflowed (all -inf logits): there is no distribution
     * to draw from, so take the best one. */
    if (!(total > 0.0)) return fallback;
    double u = rnd01(s) * total;
    double acc = 0.0;
    for (int64_t j = 0; j < size; j++) {
        acc += exp(h[j].val - max);
        if (u < acc) return (uint32_t)h[j].idx;
    }
    return (uint32_t)h[size - 1].idx;
}

/* Nucleus cut over a window that is already sorted by value, descending, and
 * whose values are already tempered. The normaliser is summed over that same
 * window, which is the reference order: top_k is a hard filter applied first
 * and top_p only ever sees its survivors (HF's TopKLogitsWarper feeding
 * TopPLogitsWarper, and vLLM's sampler, both work this way). */
static uint32_t window_pick(cwick_sampler *s, const cwick_cand *h, int64_t size) {
    const double max = (double)h[0].val;
    double total = 0.0;
    for (int64_t j = 0; j < size; j++) total += exp(h[j].val - max);
    if (!(total > 0.0)) return (uint32_t)h[0].idx;   /* all -inf: no distribution */
    if (s->cfg.top_p >= 1.0f) return draw(s, h, size, max, (uint32_t)h[0].idx);
    double acc = 0.0;
    int64_t keep = size;
    for (int64_t j = 0; j < size; j++) {
        acc += exp(h[j].val - max) / total;
        if (acc >= (double)s->cfg.top_p) { keep = j + 1; break; }
    }
    if (keep < 1) keep = 1;
    return draw(s, h, keep, max, (uint32_t)h[0].idx);
}

/* ------------------------------------------------------------------ */
/* nucleus without a sort                                              */
/* ------------------------------------------------------------------ */

/* Bucket of a logit inside [fmin, fmax]; -1 when the value cannot be a
 * candidate at all (NaN and -inf, which exp() maps to NaN and 0). `scale` is
 * CWICK_NUCLEUS_BUCKETS / (fmax - fmin), or 0 when the range is empty (every
 * logit equal), which puts everything in bucket 0. */
static inline int nuc_bucket(float v, float fmin, double scale, int B) {
    if (!(v > -INFINITY)) return -1;
    if (!(v > fmin)) return 0;
    const double d = ((double)v - (double)fmin) * scale;
    if (!(d >= 0.0)) return 0;   /* not finite (an infinite logit span) or < 0 */
    int b = (int)d;
    if (b >= B) b = B - 1;
    return b;
}

/* Draw from the nucleus when it is wider than the scan window. A sort of the
 * whole vocabulary would decide the boundary exactly and cost ~10 ms, which is
 * most of a decode step, so the boundary comes from a histogram of the logit
 * range: buckets from the top down until the running mass reaches the target,
 * then as many tokens out of the boundary bucket as close the remaining gap.
 *
 * The only approximation is *which* tokens of the boundary bucket are taken
 * (the first ones in index order rather than the largest ones), and the values
 * inside one bucket differ by at most exp(bucket width) - 1, which for 4096
 * buckets over a realistic logit range is under 1%. Everything else — the
 * normaliser, the cut, the draw weights — is exact. */
static uint32_t nucleus_draw(cwick_sampler *s, const float *x, int64_t n,
                             float fmax, double Z) {
    const int B = CWICK_NUCLEUS_BUCKETS;
    const double Target = (double)s->cfg.top_p * Z;
    if (!s->bmass) {
        s->bmass = cwick_xmalloc((size_t)B * sizeof(double));
        s->bcnt = cwick_xmalloc((size_t)B * sizeof(int64_t));
    }
    memset(s->bmass, 0, (size_t)B * sizeof(double));
    memset(s->bcnt, 0, (size_t)B * sizeof(int64_t));

    float fmin = INFINITY;
    vDSP_minv((float *)x, 1, &fmin, (vDSP_Length)n);
    const double span = (double)fmax - (double)fmin;
    const double scale = (span > 0.0) ? (double)B / span : 0.0;

    for (int64_t i = 0; i < n; i++) {
        int b = nuc_bucket(x[i], fmin, scale, B);
        if (b < 0) continue;
        s->bmass[b] += (double)s->scratch[i];
        s->bcnt[b]++;
    }

    /* Walk from the top bucket down: the boundary bucket is the first one
     * whose mass closes the gap to the target. `above` is the mass that is
     * certainly inside the nucleus once the boundary bucket is reached. */
    int bstar = -1;
    double above = 0.0;
    for (int b = B - 1; b >= 0; b--) {
        if (above + s->bmass[b] >= Target) { bstar = b; break; }
        above += s->bmass[b];
    }
    if (bstar < 0) { bstar = 0; above = 0.0; }   /* rounding: take everything */

    /* How many of the boundary bucket close the remaining gap. Its mass is
     * positive (otherwise the walk would have stopped higher up), so the
     * average value is a safe stand-in for the ones actually taken. */
    int64_t k = 0;
    const double need = Target - above;
    if (need > 0.0 && s->bcnt[bstar] > 0) {
        const double vtyp = s->bmass[bstar] / (double)s->bcnt[bstar];
        if (vtyp > 0.0) {
            k = (int64_t)ceil(need / vtyp);
            if (k < 1) k = 1;
            if (k > s->bcnt[bstar]) k = s->bcnt[bstar];
        }
    }

    /* Draw: one walk in index order that exits as soon as the running mass
     * passes the draw, skipping what the nucleus left out. */
    double u = rnd01(s) * Target;
    double acc = 0.0;
    int64_t taken = 0;
    for (int64_t i = 0; i < n; i++) {
        int b = nuc_bucket(x[i], fmin, scale, B);
        if (b < bstar) continue;
        if (b == bstar && taken++ >= k) continue;
        const double w = (double)s->scratch[i];
        if (!(w > 0.0)) continue;    /* NaN or underflowed: never drawable */
        acc += w;
        if (u < acc) return (uint32_t)i;
    }
    return (uint32_t)(n - 1);
}

bool cwick_sampler_plain_argmax(const cwick_sampler *s) {
    if (!s) return false;
    if (!cwick_sampler_cfg_is_greedy(&s->cfg)) return false;
    if (s->cfg.presence_penalty != 0.0f ||
        s->cfg.frequency_penalty != 0.0f)
        return false;
    if (!s->lbrk) return true;
    /* The ban is armed exactly when some period's match count has reached its
     * threshold -- the same test pick() runs before masking. */
    for (int p = 1; p <= s->lper_max; p++)
        if (s->lmatch[p] >= s->lbrk_need[p]) return false;
    return true;
}

uint32_t cwick_sampler_pick(cwick_sampler *s, float *x, int64_t n) {
    if (!s || !x || n <= 0) return 0;
    if (n > s->vocab) n = s->vocab;   /* the table sizes follow the spec */

    /* 0. the loop breaker. While the produced tail repeats exactly at period
     * p for lbrk_need[p] tokens, the token that continues the pattern — the
     * one p positions back in the ring — is banned, and the model has to say
     * something else. Everything downstream (penalties, argmax, the top_k
     * window, the nucleus) just sees a masked logit. -INFINITY survives the
     * penalty subtraction and exp() maps it to exactly 0 mass, so no other
     * path needs to know. The guard stays as the backstop for the case the
     * model escapes into a synonym and re-enters. */
    if (s->lbrk && !s->lhit_per && s->lgen > 0) {
        const int64_t H = CWICK_LOOP_HIST;
        const int64_t g = s->lgen;
        bool armed = false;
        for (int p = 1; p <= s->lper_max; p++) {
            if (s->lmatch[p] >= s->lbrk_need[p]) {
                uint32_t cont = s->lhist[(g - p) % H];
                if ((int64_t)cont < n) {
                    x[cont] = -INFINITY;
                    armed = true;
                }
            }
        }
        if (armed) s->lbrk_steps++;
    }

    /* 1. repetition penalties, on the raw logits (OpenAI/HF order). */
    if (s->counts) {
        const float pr = s->cfg.presence_penalty;
        const float fr = s->cfg.frequency_penalty;
        for (int64_t i = 0; i < n; i++) {
            int32_t c = s->counts[i];
            if (c) x[i] -= pr + fr * (float)c;
        }
    }
    if (!(s->cfg.temperature > 0.0f)) return argmax_of(x, n);

    if (!s->cand) s->cand = cwick_xmalloc((size_t)s->vocab * sizeof(cwick_cand));

    /* 2. the candidate window. */
    const bool have_topk = (s->cfg.top_k > 0 && s->cfg.top_k < n);
    int64_t want = have_topk ? (int64_t)s->cfg.top_k : CWICK_TOPP_SCAN;
    if (want > n) want = n;

    if (have_topk) {
        /* top_k keeps the `top_k` largest logits, and temperature > 0 is a
         * positive scale, so the ranking is the same before and after it: the
         * window can be gathered on the raw (penalised) logits and the
         * temperature applied to the K survivors alone. That is the whole
         * point of the order — a decode step is short and a 150144-wide
         * multiply/exp pass would be a visible slice of it. */
        int64_t size = gather_top(x, n, want, s->cand);
        if (size <= 0) return argmax_of(x, n);
        qsort(s->cand, (size_t)size, sizeof(cwick_cand), cmp_cand_desc);
        const float inv_t = 1.0f / s->cfg.temperature;
        if (inv_t != 1.0f)
            for (int64_t j = 0; j < size; j++) s->cand[j].val *= inv_t;
        return window_pick(s, s->cand, size);
    }

    /* No top_k: the nucleus is over the whole vocabulary, so its normaliser
     * has to be too. Temperature first (it changes the normaliser), then the
     * one pass that has to be vectorised: exp() over the vocabulary is ~0.2 ms
     * with vvexpf against ~2 ms for a scalar expf loop, and the sum is
     * accumulated in double so the denominator does not lose the tail. */
    const float inv_t = 1.0f / s->cfg.temperature;
    if (inv_t != 1.0f) {
        for (int64_t i = 0; i < n; i++) x[i] *= inv_t;
    }
    float fmax = -INFINITY;
    vDSP_Length nn = (vDSP_Length)n;
    vDSP_maxv(x, 1, &fmax, nn);
    if (!(fmax > -INFINITY) || fmax != fmax) return argmax_of(x, n);
    if (!s->scratch) s->scratch = cwick_xmalloc((size_t)s->vocab * sizeof(float));
    float neg = -fmax;
    vDSP_vsadd(x, 1, &neg, s->scratch, 1, nn);
    int in = (int)n;
    vvexpf(s->scratch, s->scratch, &in);
    double Z = 0.0;
    for (int64_t i = 0; i < n; i++) Z += (double)s->scratch[i];
    const double dmax = (double)fmax;
    if (!(Z > 0.0)) {
        if (Z != Z) {
            /* A NaN logit poisons the normaliser: drop it and renormalise
             * over the rest rather than falling back to greedy. */
            for (int64_t i = 0; i < n; i++)
                if (s->scratch[i] != s->scratch[i]) s->scratch[i] = 0.0f;
            Z = 0.0;
            for (int64_t i = 0; i < n; i++) Z += (double)s->scratch[i];
        }
        if (!(Z > 0.0)) return argmax_of(x, n);
    }

    /* 3. nucleus over a fixed scan window. A nucleus wider than the window is
     * possible (a flat distribution), and then the boundary is decided by
     * histogram instead of by sorting the whole vocabulary. */
    {
        int64_t size = gather_top(x, n, want, s->cand);
        if (size <= 0) return argmax_of(x, n);
        qsort(s->cand, (size_t)size, sizeof(cwick_cand), cmp_cand_desc);
        if (s->cfg.top_p >= 1.0f) {
            return draw(s, s->cand, size, dmax, (uint32_t)s->cand[0].idx);
        }
        double acc = 0.0;
        int64_t keep = size;
        for (int64_t j = 0; j < size; j++) {
            acc += exp(s->cand[j].val - dmax) / Z;
            if (acc >= (double)s->cfg.top_p) { keep = j + 1; break; }
        }
        if (acc < (double)s->cfg.top_p && want < n)
            return nucleus_draw(s, x, n, fmax, Z);
        if (keep < 1) keep = 1;
        return draw(s, s->cand, keep, dmax, (uint32_t)s->cand[0].idx);
    }
}
