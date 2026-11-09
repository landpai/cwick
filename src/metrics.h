/* metrics.h — Prometheus exposition for cwick, modelled on vLLM's metrics.
 *
 * vLLM publishes a Prometheus endpoint next to the OpenAI API and documents
 * it at https://docs.vllm.ai/en/latest/usage/metrics/ . Its value is that the
 * *shape* of the numbers is stable across engines: a dashboard written for
 * vLLM's `vllm:time_to_first_token_seconds` keeps working when the thing
 * behind the port is a 320B model streaming its experts off an SSD.
 *
 * So cwick reuses the vLLM names wherever the meaning matches — TTFT, TPOT,
 * end-to-end latency, prompt/generation token counters, cache usage — and
 * adds `cwick:*` metrics for the things vLLM has no name for because it never
 * expected weights to live on disk:
 *
 *   cwick:expert_cache_hits_total / _misses_total   expert-pin recall
 *   cwick:expert_bytes_streamed_total               SSD bytes per token
 *   cwick:decode_step_seconds                       the wall clock that
 *                                                  actually decides tok/s
 *
 * Naming follows vLLM's post-0.7 convention: no `model_name` label (the
 * engine serves exactly one checkpoint), `_total` suffix on counters, `le`
 * buckets on histograms, and a `cwick:cache_config_info` info-metric for the
 * static configuration. See the "deprecation policy" note in README: names
 * are kept stable across minor versions and renamed only with a release of
 * deprecation warning.
 *
 * Cost: everything is an atomic increment or store. The hot path (one decode
 * step) adds two relaxed atomic adds — tens of nanoseconds against a step
 * that costs hundreds of milliseconds. There is no sampler thread.
 */
#ifndef CWICK_METRICS_H
#define CWICK_METRICS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "cwick.h"

/* ------------------------------------------------------------------ */
/* Static configuration (published as an info metric)                  */
/* ------------------------------------------------------------------ */
typedef struct cwick_metrics_info {
    const char *model;        /* e.g. "glm-5.3-flash" */
    const char *backend;      /* "metal" | "cpu" */
    const char *spine;        /* "bf16" | "int8" */
    int64_t max_context;      /* KV window in tokens */
    int64_t num_layers;
    int64_t num_experts;
    int64_t experts_per_token;
} cwick_metrics_info;

void cwick_metrics_init(const cwick_metrics_info *info);

/* Replace only the static identity — used after a runtime model switch.
 * Unlike init it leaves the counters and the process start time alone: an
 * instance that swapped models is still the same instance, and a scrape that
 * suddenly shows zero requests would be read as a crash. */
void cwick_metrics_set_model_info(const cwick_metrics_info *info);

/* ------------------------------------------------------------------ */
/* Request lifecycle (server side)                                     */
/* ------------------------------------------------------------------ */
/* A request arrived but has not got the generation slot yet. Returns a
 * monotonic id; `cwick_metrics_request_admit` consumes it. */
uint64_t cwick_metrics_request_arrive(void);

/* The request got the slot (or was refused). `queue_s` is how long it sat
 * in the accept queue before that. */
void cwick_metrics_request_admit(uint64_t rid, double queue_s);
void cwick_metrics_request_refuse(const char *reason);

/* First byte of the first generated token reached the client. Only meaningful
 * for streaming requests; for a non-streaming one the whole body lands at
 * once and TTFT is reported at completion instead. The server calls this once
 * per request at the first-token moment (queue time included), so
 * `cwick_metrics_request_finish` deliberately does not observe it again. */
void cwick_metrics_observe_ttft(double seconds);

/* One request finished. `finish_reason` is vLLM's vocabulary:
 * "stop" (EOS), "length" (hit the token cap), "abort" (client hung up / Ctrl+C).
 * `success` false marks a failed request (5xx) — counted separately. */
void cwick_metrics_request_finish(int64_t prompt_tokens, int64_t gen_tokens,
                                 double prefill_s, double decode_s,
                                 double e2e_s, double ttft_s,
                                 double max_new_tokens,
                                 const char *finish_reason, bool success);

/* A request that never produced a body (busy, bad JSON, engine reset
 * failure). Still counted so `requests_total` matches the access log. */
void cwick_metrics_request_error(const char *kind);

/* ------------------------------------------------------------------ */
/* Generation internals (engine side)                                  */
/* ------------------------------------------------------------------ */
/* Token counters are maintained live by the engine (one relaxed atomic add
 * per request, not per token), so a scrape during a long stream already
 * shows the tokens processed so far. `cwick_metrics_request_finish` only
 * observes histograms for these — it must not add to the counters again. */
void cwick_metrics_add_prompt_tokens(int64_t n);
void cwick_metrics_add_generation_tokens(int64_t n);
void cwick_metrics_add_decode_steps(int64_t n);

/* One decode step's wall time. This is the distribution that decides
 * tok/s: its mean is 1/throughput, and its tail is what a user feels. */
void cwick_metrics_observe_decode_step(double seconds);

/* One engine iteration and how many sequences it advanced. vLLM publishes
 * this as `vllm:iteration_tokens_total`; with one generation slot it is 1,
 * with a batched group it is the live group size. */
void cwick_metrics_observe_iteration_tokens(int64_t n);

/* Mean seconds per decode step observed so far, or 0 before the first sample.
 * engine.c uses it to turn an uncapped token budget into a wall-clock estimate
 * so the operator can see, up front, that a request is going to run for hours. */
double cwick_metrics_decode_step_mean_seconds(void);

/* Prefix-cache lookups. A hit means the prompt started with tokens cwick has
 * already prefilled, so only the tail was run through the model; on a
 * repeating 30K-token system prompt that is the difference between hours and
 * a fraction of a second. `reused` is the number of prompt tokens skipped. */
void cwick_metrics_prefix_query(bool hit, int64_t reused);

/* ------------------------------------------------------------------ */
/* Live gauges                                                         */
/* ------------------------------------------------------------------ */
/* The expert stream is sampled at render time, so `/metrics` reports the
 * cache as it is *now* rather than as it was at some earlier step. */
void cwick_metrics_set_stream(const cwick_stream *s);
void cwick_metrics_set_memory_ceiling(uint64_t bytes);
void cwick_metrics_set_context_window(int64_t tokens);
/* Number of requests currently holding the generation slot (0 or 1 for
 * cwick's single-turn engine, >1 the moment it batches). */
void cwick_metrics_set_requests_running(int64_t n);

/* Where the running request is inside its prefill. `total > 0` means "a
 * prefill is in flight"; everything is zeroed when prefill ends.
 *
 * Long prompts are the one place cwick goes quiet for minutes at a time: the
 * prompt/generation token counters only move when a phase *completes*, so a
 * scrape during a 30K-token prefill used to show prompt_tokens_total = 0 and
 * no way to tell progress from a hang. These gauges are the live view. */
void cwick_metrics_set_prefill(int64_t done_tokens, int64_t total_tokens,
                              int64_t layer, int64_t nlayer);

/* ------------------------------------------------------------------ */
/* Exposition                                                          */
/* ------------------------------------------------------------------ */
/* Render the whole registry in Prometheus text format (v0.0.4). Returns a
 * malloc'd NUL-terminated buffer; `out_len` gets its length without the NUL.
 * The caller frees it. */
char *cwick_metrics_render(size_t *out_len);

/* Content type a Prometheus scrape expects for this payload. */
const char *cwick_metrics_content_type(void);

/* One line per headline number, through the normal logger — lets a `--verbose`
 * serve print its own metrics without a scraper. */
void cwick_metrics_log_summary(const char *tag);

#endif /* CWICK_METRICS_H */
