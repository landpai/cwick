/* sample.h — token sampling and repetition penalties for the decode loop.
 *
 * The engine used to take a plain argmax of the logits, which is exactly
 * temperature 0. A checkpoint that ships `generation_config.json` with
 * do_sample=true / temperature 1.0 / top_p 0.95 / top_k 20 is then served
 * wrong: greedy decoding walks into repetition attractors (a sentence or a
 * label repeated until the token budget is gone) that sampling with the
 * checkpoint's own parameters never reaches.
 *
 * So the decode loop asks this module for a token instead:
 *
 *   temperature <= 0            -> argmax (what the greedy contract promised,
 *                                  and what temperature:0 still means)
 *   temperature > 0             -> top-k (a hard filter), then top-p (nucleus
 *                                  over the survivors), then one uniform draw
 *   presence / frequency penalty-> subtracted from the raw logits first, the
 *                                  standard OpenAI/HF shape:
 *                                  logit -= presence * 1 + frequency * count
 *
 * Sampling lowers the odds of the attractor; the loop guard below is what ends
 * the turn when it happens anyway, so a degenerate run costs a bounded number
 * of tokens instead of the whole budget.
 *
 * The order is the reference one: top_k filters, top_p normalises over what is
 * left. It is also what keeps the cost off the decode step — a positive
 * temperature preserves the ranking, so the top-k window is gathered on the raw
 * logits and the temperature is applied to the K survivors instead of to all
 * 150144 of them.
 *
 * Cost matters: a decode step is short and a 150144-token vocabulary
 * makes a full sort of the logits ~10 ms, i.e. most of a step. Nothing here
 * sorts the whole vocabulary: the candidate set is bounded (top-k, or a
 * fixed top-p scan window) and gathered with a bounded min-heap, and the only
 * full-vocabulary passes are max / exp / sum, which are vectorised. */
#ifndef CWICK_SAMPLE_H
#define CWICK_SAMPLE_H

#include <stddef.h>
#include <stdint.h>

#include "cwick.h"

/* How many candidates the top-p scan looks at when top_k is off. A nucleus
 * wider than this is not wrong, just unheard of for a trained LM; the sampler
 * then decides the boundary with a histogram instead of a sort (see
 * CWICK_NUCLEUS_BUCKETS). */
#define CWICK_TOPP_SCAN 4096

/* Buckets for the nucleus boundary when the scan window cannot cover it. The
 * nucleus of a flat distribution can be most of the vocabulary, and a
 * 150144-way sort is ~10 ms — most of a decode step — so the boundary is
 * decided from a histogram of the logit range instead: one pass, ~0.4 ms, and
 * the cut is exact to within the width of one bucket. */
#define CWICK_NUCLEUS_BUCKETS 4096

/* ------------------------------------------------------------------ */
/* loop guard                                                          */
/* ------------------------------------------------------------------ */
/* Sampling makes the repetition loop rare; it does not make it impossible
 * (one 250-token story in six still ended in "喷喷喷喷…"). This is the net
 * under it: watch the tokens this run has produced and stop the generation
 * when a fixed block has been repeating.
 *
 * A loop is "the last p tokens equal the p before them, for a long run". The
 * test is per period — `lmatch[p]` counts how many trailing positions repeat
 * at period p — and it fires when that count covers both `lmin_occ` whole
 * blocks and `lmin_run` tokens. The second condition is what keeps ordinary
 * structure from tripping it: a markdown rule, an ellipsis or "1. 2. 3."
 * repeats a handful of times and stops, and 16 identical tokens in a row is
 * not something an answer does on purpose.
 *
 * Cost is one comparison per period per token — 64 compares against a step
 * that costs ~15 ms — and the history is a ring inside the sampler, so there
 * is no allocation and no per-step pass over the vocabulary. */
#define CWICK_LOOP_MAX_PERIOD 64
/* Ring depth: the comparison for period p needs the last 2p tokens, so this
 * only has to cover 2*max_period. */
#define CWICK_LOOP_HIST (4 * CWICK_LOOP_MAX_PERIOD)
#define CWICK_LOOP_MIN_RUN 16   /* consecutive repeating tokens before it fires */
#define CWICK_LOOP_MIN_OCC 3    /* occurrences of the period block before it fires */

/* The breaker sits one notch below the guard: while the tail repeats exactly
 * at period p for this many tokens, pick() bans the token that would extend
 * the pattern, so the run escapes instead of coasting to the guard. Measured
 * on live agent traffic, the guard fired at 16 tokens after the whole answer
 * had degenerated; banning at 8 turns the escape into a one-step event. 2*p
 * for the larger periods keeps the requirement proportional — but the cap
 * below exists because banning can *create* a longer loop: with a period-1
 * attractor banned at 8, the run settles into "nine tokens + the escapee",
 * a period-10 pattern that a proportional threshold (20) would only arm at
 * exactly where the guard fires. The cap keeps every period armed well
 * before the guard can act. */
#define CWICK_LOOP_BREAK_MIN 8  /* trailing repeats that arm the ban */
#define CWICK_LOOP_BREAK_MAX 12 /* and the cap that keeps it under the guard */

typedef struct {
    float val;
    int32_t idx;
} cwick_cand;

typedef struct {
    cwick_sampler_cfg cfg;
    int64_t vocab;
    /* Per-token count since the start of this generation, NULL when both
     * penalties are 0 (the common case: no table, no per-step pass over it). */
    int32_t *counts;
    float *scratch;     /* [vocab] exp(x - max), only for the exp/sum pass */
    cwick_cand *cand;   /* [vocab] candidate heap, worst case = the full vocab */
    /* Nucleus histogram, allocated on first use by the no-top_k path only. */
    double *bmass;      /* [CWICK_NUCLEUS_BUCKETS] mass per logit bucket */
    int64_t *bcnt;      /* [CWICK_NUCLEUS_BUCKETS] tokens per logit bucket */
    uint64_t rng;       /* splitmix64 state */

    /* Loop guard: the tokens this run has produced, in a ring, plus per-period
     * counts of how many trailing positions repeat at that period. Filled by
     * cwick_sampler_accept, read by cwick_sampler_looped. */
    bool lon;                          /* guard enabled (CWICK_LOOP_DETECT) */
    int lper_max;                      /* highest period checked (<= 64) */
    int64_t lmin_run;                  /* run length needed, per period floor */
    int lmin_occ;                      /* occurrences needed, per period */
    int lneed[CWICK_LOOP_MAX_PERIOD + 1];  /* max((occ-1)*p, min_run) per p */
    uint32_t lhist[CWICK_LOOP_HIST];   /* ring of produced tokens */
    int64_t lgen;                      /* tokens fed so far */
    int lmatch[CWICK_LOOP_MAX_PERIOD + 1];
    int lhit_per;                      /* period that fired, 0 = none */
    int64_t lhit_run;                  /* how long it had repeated when it did */

    /* Loop breaker: when the tail repeats exactly at period p for
     * lbrk_need[p] tokens, pick() bans the token that continues the pattern
     * (the one sitting p positions back in the ring), so the generation
     * escapes the attractor before the guard has to kill it. lbrk_steps
     * counts the picks that went out with a ban armed — the engine prints
     * it next to its summary, and a stop with lb>0 is a recovery, not a
     * clean finish. */
    bool lbrk;                         /* breaker enabled (CWICK_LOOP_BREAK) */
    int lbrk_need[CWICK_LOOP_MAX_PERIOD + 1];  /* max(2*p, break_min) */
    int64_t lbrk_steps;                /* picks made while a ban was armed */
} cwick_sampler;

/* Greedy: temperature 0, no truncation, no penalty. */
void cwick_sampler_cfg_default(cwick_sampler_cfg *c);
/* True when the config cannot do anything but argmax. */
bool cwick_sampler_cfg_is_greedy(const cwick_sampler_cfg *c);
/* Reject what a client must not send (NaN, negative temperature, top_p outside
 * (0,1], top_k < 0). Returns false and fills `why` (if given) on rejection. */
bool cwick_sampler_cfg_validate(const cwick_sampler_cfg *c, char *why, size_t cap);

/* Read <root>/generation_config.json (do_sample / temperature / top_p / top_k)
 * into `out`. False when there is no such file or nothing usable in it, in
 * which case `out` is left alone. */
bool cwick_sampler_cfg_from_checkpoint(const char *root, cwick_sampler_cfg *out);

/* `seed` decides the whole run: same seed + same logits + same config -> same
 * tokens. The engine builds one sampler per generation, so a server request is
 * reproducible only if the seed is; the CLI is (it is derived from nothing
 * random). */
void cwick_sampler_init(cwick_sampler *s, const cwick_sampler_cfg *c,
                        int64_t vocab, uint64_t seed);
void cwick_sampler_free(cwick_sampler *s);

/* Count a token the run has just *produced*: penalties (vLLM and HF both
 * penalise the whole visible text) and the loop guard, which is about what the
 * model is doing now. */
void cwick_sampler_accept(cwick_sampler *s, uint32_t tok);

/* Count a token that was in the *prompt*: penalties only. Priming the prompt
 * through accept() instead would feed the tail of the prompt to the loop guard
 * and make a prompt that ends in a repeated block look like a loop. */
void cwick_sampler_prime(cwick_sampler *s, uint32_t tok);

/* True when the produced-token stream has fallen into a repetition loop.
 * `period` / `run` (optional) report which period fired and how many trailing
 * tokens had been repeating. Latched: once true it stays true. */
bool cwick_sampler_looped(const cwick_sampler *s, int *period, int64_t *run);

/* Pick the next token from `logits` (length n). The array is modified in
 * place — the caller recomputes it on every step anyway. */
uint32_t cwick_sampler_pick(cwick_sampler *s, float *logits, int64_t n);

/* One-line description for request logs: "greedy", or
 * "t=1.00 top_p=0.95 top_k=20 pen=0.00/0.00". */
void cwick_sampler_describe(const cwick_sampler_cfg *c, char *buf, size_t cap);

#endif /* CWICK_SAMPLE_H */
