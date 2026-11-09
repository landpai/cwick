/* metrics.c — Prometheus registry for cwick (see metrics.h for the rationale).
 *
 * The registry is deliberately flat: one file, no registration API, no
 * samplers. A single-slot engine on one machine has a few dozen numbers, and
 * the interesting ones (expert cache recall, decode step latency, SSD bytes
 * per token) are all cheap enough to update on the hot path.
 *
 * Exposition is Prometheus text format v0.0.4:
 *   https://prometheus.io/docs/instrumenting/exposition_formats/
 * Histograms are rendered cumulatively over `le` buckets, which is what a
 * scraper expects even though the counters are stored per-bucket.
 */
#include "metrics.h"

#include <math.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "log.h"
#include "mem.h"
#include "util.h"

/* ------------------------------------------------------------------ */
/* Primitives                                                          */
/* ------------------------------------------------------------------ */
#define CWICK_MAX_BUCKETS 24

/* A double has no lock-free atomic arithmetic in the C11 sense on every
 * target, so gauges and histogram sums are stored as their IEEE-754 bit
 * pattern and updated with a compare-exchange loop. Contention is nil:
 * the only writer of any one gauge is a single thread. */
static uint64_t d2u(double d) { uint64_t u; memcpy(&u, &d, sizeof(u)); return u; }
static double u2d(uint64_t u) { double d; memcpy(&d, &u, sizeof(d)); return d; }

static double gauge_get(_Atomic uint64_t *p) {
    return u2d(atomic_load_explicit(p, memory_order_relaxed));
}
static void gauge_add(_Atomic uint64_t *p, double v) {
    uint64_t old = atomic_load_explicit(p, memory_order_relaxed);
    for (;;) {
        double d = u2d(old) + v;
        uint64_t nw = d2u(d);
        if (atomic_compare_exchange_weak_explicit(p, &old, nw,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed))
            return;
    }
}
static void counter_add(_Atomic uint64_t *p, uint64_t v) {
    if (v) atomic_fetch_add_explicit(p, v, memory_order_relaxed);
}

typedef struct {
    const char *name;
    const char *help;
    const double *buckets;   /* upper bounds, ascending */
    int nbuckets;
    _Atomic uint64_t count[CWICK_MAX_BUCKETS];
    _Atomic uint64_t sum_bits;
    _Atomic uint64_t total;
} cwick_hist;

static void hist_observe(cwick_hist *h, double v) {
    if (isnan(v) || isinf(v)) return;            /* a NaN would fall through
                                                  * every bucket and inflate
                                                  * only `+Inf` */
    gauge_add(&h->sum_bits, v);
    atomic_fetch_add_explicit(&h->total, 1, memory_order_relaxed);
    for (int i = 0; i < h->nbuckets; i++) {
        if (v <= h->buckets[i]) {
            atomic_fetch_add_explicit(&h->count[i], 1, memory_order_relaxed);
            return;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Histogram definitions (bucket sets copied from vLLM's documented ones) */
/* ------------------------------------------------------------------ */
static const double k_ttft[] = {
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75,
    1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0
};
static const double k_tpot[] = {
    0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5
};
static const double k_e2e[] = {
    1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0,
    120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0
};
static const double k_queue[] = {
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 1.0, 5.0
};
/* vLLM's build_1_2_5_buckets(6): 1,2,5,10,... up to 1e6. */
static const double k_tokens[] = {
    1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000,
    20000, 50000, 100000, 200000, 500000, 1000000
};
static const double k_iter[] = {
    1, 2, 5, 8, 16, 32, 64, 128, 256, 512, 1024, 4096
};
/* cwick-specific: one decode step. Its mean is the reciprocal of tok/s, so
 * this is the histogram that actually answers "are we at 3 tok/s?". */
static const double k_step[] = {
    0.05, 0.1, 0.15, 0.2, 0.25, 0.333, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0
};

#define HIST(name_, help_, buckets_) \
    { name_, help_, buckets_, (int)(sizeof(buckets_) / sizeof((buckets_)[0])), \
      {0}, 0, 0 }

enum {
    H_TTFT, H_TPOT, H_E2E, H_QUEUE, H_PREFILL, H_DECODE,
    H_PROMPT_TOKENS, H_GEN_TOKENS, H_MAX_GEN_TOKENS, H_ITER, H_STEP,
    H_COUNT
};

static cwick_hist g_hists[] = {
    HIST("vllm:time_to_first_token_seconds",
         "Histogram of time to first token in seconds.", k_ttft),
    HIST("vllm:time_per_output_token_seconds",
         "Histogram of time per output token in seconds.", k_tpot),
    HIST("vllm:e2e_request_latency_seconds",
         "Histogram of end to end request latency in seconds.", k_e2e),
    HIST("vllm:request_queue_time_seconds",
         "Histogram of time spent waiting for the generation slot.", k_queue),
    HIST("vllm:request_prefill_time_seconds",
         "Histogram of prefill (prompt) time in seconds.", k_e2e),
    HIST("vllm:request_decode_time_seconds",
         "Histogram of decode time in seconds.", k_e2e),
    HIST("vllm:request_prompt_tokens",
         "Histogram of prompt token counts.", k_tokens),
    HIST("vllm:request_generation_tokens",
         "Histogram of generated token counts.", k_tokens),
    HIST("vllm:request_max_num_generation_tokens",
         "Histogram of the requested generation-token ceiling.", k_tokens),
    HIST("vllm:iteration_tokens_total",
         "Histogram of tokens produced per engine iteration.", k_iter),
    HIST("cwick:decode_step_seconds",
         "Histogram of one decode step's wall time in seconds.", k_step),
};
/* Guard against reordering g_hists without renumbering the enum. */
_Static_assert(sizeof(g_hists) / sizeof(g_hists[0]) == H_COUNT,
               "cwick histogram table and H_* enum are out of sync");

/* ------------------------------------------------------------------ */
/* Counters and gauges                                                 */
/* ------------------------------------------------------------------ */
static _Atomic uint64_t m_prompt_tokens;
static _Atomic uint64_t m_generation_tokens;
static _Atomic uint64_t m_decode_steps;
static _Atomic uint64_t m_requests_total;
static _Atomic uint64_t m_requests_success;
static _Atomic uint64_t m_requests_failed;
static _Atomic uint64_t m_requests_aborted;
static _Atomic uint64_t m_requests_refused;
static _Atomic uint64_t m_finish_stop;
static _Atomic uint64_t m_finish_length;
static _Atomic uint64_t m_finish_abort;
static _Atomic uint64_t m_finish_loop;
static _Atomic uint64_t m_expert_hits;
static _Atomic uint64_t m_expert_misses;
static _Atomic uint64_t m_pfx_queries;
static _Atomic uint64_t m_pfx_hits;
static _Atomic uint64_t m_pfx_reused;    /* prompt tokens skipped by a hit */

static _Atomic uint64_t m_expert_reads;
static _Atomic uint64_t m_expert_bytes;

static _Atomic uint64_t g_pf_done;      /* prefill tokens done (live) */
static _Atomic uint64_t g_pf_total;     /* prefill tokens in this request */
static _Atomic uint64_t g_pf_layer;     /* 1-based layer now running */
static _Atomic uint64_t g_pf_layers;    /* layers in the checkpoint */

static _Atomic uint64_t g_running;      /* requests holding the slot */
static _Atomic uint64_t g_waiting;      /* requests queued behind it */
static _Atomic uint64_t g_expert_used;
static _Atomic uint64_t g_expert_cap;
static _Atomic uint64_t g_rss;
static _Atomic uint64_t g_ceiling;
static _Atomic uint64_t g_context_window;
static _Atomic uint64_t g_started_ns;

static const cwick_stream *g_stream;     /* sampled at render time */

static cwick_metrics_info g_info;
static bool g_initialised;

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */
static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

void cwick_metrics_init(const cwick_metrics_info *info) {
    if (info) {
        g_info = *info;
        /* `model` is a pointer and the callers hand it stack buffers (the
         * startup path derives the served alias into a local). Copying the
         * struct therefore published a dangling pointer, and /v1/metrics
         * reported the model as whatever byte the stack slot held next --
         * observed as model="B" for a MiniMax run. */
        if (info->model) {
            static char model_buf[128];
            snprintf(model_buf, sizeof(model_buf), "%s", info->model);
            g_info.model = model_buf;
        }
        /* The denominator of the prefill progress bar is a property of the
         * checkpoint, not of a request: without this it reads 0 whenever the
         * engine is idle, and a dashboard computing done/total sees either 0/0
         * or a bar that jumps from "unknown" to "full" at the first token. */
        if (info->num_layers > 0) g_pf_layers = (uint64_t)info->num_layers;
    }
    g_started_ns = now_ns();
    g_initialised = true;
}

/* See metrics.h: identity change without touching the accumulators. */
void cwick_metrics_set_model_info(const cwick_metrics_info *info) {
    if (info) {
        g_info = *info;
        /* A model switch replaces the checkpoint, so the prefill denominator
         * has to follow it: leaving it at the previous checkpoint's layer
         * count makes the progress bar read 0/45 while a 40-layer model runs
         * (and never reach 100%). */
        if (info->num_layers > 0) g_pf_layers = (uint64_t)info->num_layers;
    }
    g_initialised = true;
}

uint64_t cwick_metrics_request_arrive(void) {
    counter_add(&m_requests_total, 1);
    counter_add(&g_waiting, 1);
    return atomic_load_explicit(&m_requests_total, memory_order_relaxed);
}

void cwick_metrics_request_admit(uint64_t rid, double queue_s) {
    (void)rid;
    if (atomic_load_explicit(&g_waiting, memory_order_relaxed) > 0)
        atomic_fetch_sub_explicit(&g_waiting, 1, memory_order_relaxed);
    counter_add(&g_running, 1);
    if (queue_s >= 0.0) hist_observe(&g_hists[H_QUEUE], queue_s);
}

void cwick_metrics_request_refuse(const char *reason) {
    (void)reason;
    if (atomic_load_explicit(&g_waiting, memory_order_relaxed) > 0)
        atomic_fetch_sub_explicit(&g_waiting, 1, memory_order_relaxed);
    counter_add(&m_requests_refused, 1);
}

void cwick_metrics_request_error(const char *kind) {
    if (atomic_load_explicit(&g_waiting, memory_order_relaxed) > 0)
        atomic_fetch_sub_explicit(&g_waiting, 1, memory_order_relaxed);
    counter_add(&m_requests_failed, 1);
    (void)kind;
}

void cwick_metrics_observe_ttft(double seconds) {
    hist_observe(&g_hists[H_TTFT], seconds);
}

void cwick_metrics_request_finish(int64_t prompt_tokens, int64_t gen_tokens,
                                 double prefill_s, double decode_s,
                                 double e2e_s, double ttft_s,
                                 double max_new_tokens,
                                 const char *finish_reason, bool success) {
    (void)ttft_s;   /* TTFT is observed at first-token time, not here */
    if (atomic_load_explicit(&g_running, memory_order_relaxed) > 0)
        atomic_fetch_sub_explicit(&g_running, 1, memory_order_relaxed);
    if (atomic_load_explicit(&g_waiting, memory_order_relaxed) > 0)
        atomic_fetch_sub_explicit(&g_waiting, 1, memory_order_relaxed);

    /* Failures are not all the same: a request the client abandoned counts
     * as aborted too (that is where requests_aborted_total is meant to be
     * visible), a genuine error does not. */
    if (!success) {
        counter_add(&m_requests_failed, 1);
        if (finish_reason && strcmp(finish_reason, "abort") == 0)
            counter_add(&m_requests_aborted, 1);
        return;
    }
    counter_add(&m_requests_success, 1);

    /* The prompt counter carries the *full* prompt the client sent (what
     * vLLM's prompt_tokens_total means), including the part a prefix hit
     * replayed. The engine's own per-prefill add only saw the tokens it
     * actually computed, so a 499-token request with 498 replayed used to
     * land here as "1". Histograms keep observing the same full number. */
    if (prompt_tokens > 0)
        counter_add(&m_prompt_tokens, (uint64_t)prompt_tokens);

    const char *reason = finish_reason ? finish_reason : "stop";
    if (strcmp(reason, "length") == 0) counter_add(&m_finish_length, 1);
    else if (strcmp(reason, "abort") == 0) { counter_add(&m_finish_abort, 1);
                                             counter_add(&m_requests_aborted, 1); }
    else if (strcmp(reason, "loop") == 0) counter_add(&m_finish_loop, 1);
    else counter_add(&m_finish_stop, 1);

    hist_observe(&g_hists[H_E2E], e2e_s);
    hist_observe(&g_hists[H_PREFILL], prefill_s);
    hist_observe(&g_hists[H_DECODE], decode_s);
    hist_observe(&g_hists[H_PROMPT_TOKENS], (double)prompt_tokens);
    hist_observe(&g_hists[H_GEN_TOKENS], (double)gen_tokens);
    hist_observe(&g_hists[H_MAX_GEN_TOKENS], max_new_tokens);
    /* TTFT is *not* observed here: the server records it at the moment the
     * first token reaches the client (see cwick_metrics_observe_ttft), which
     * keeps the histogram live for streams that run for minutes. `ttft_s` is
     * accepted only so the signature mirrors the engine's own timings. */
    /* TPOT: mean over the decode steps actually run, which is the number the
     * engine already divides by for its own tok/s line. */
    if (decode_s > 0.0 && gen_tokens > 1)
        hist_observe(&g_hists[H_TPOT], decode_s / (double)(gen_tokens - 1));
}

void cwick_metrics_add_prompt_tokens(int64_t n) {
    counter_add(&m_prompt_tokens, n > 0 ? (uint64_t)n : 0);
}
void cwick_metrics_add_generation_tokens(int64_t n) {
    counter_add(&m_generation_tokens, n > 0 ? (uint64_t)n : 0);
}
void cwick_metrics_add_decode_steps(int64_t n) {
    counter_add(&m_decode_steps, n > 0 ? (uint64_t)n : 0);
}
void cwick_metrics_prefix_query(bool hit, int64_t reused) {
    counter_add(&m_pfx_queries, 1);
    if (hit) {
        counter_add(&m_pfx_hits, 1);
        if (reused > 0) counter_add(&m_pfx_reused, (uint64_t)reused);
    }
}

void cwick_metrics_observe_decode_step(double seconds) {
    hist_observe(&g_hists[H_STEP], seconds);
}

/* vLLM's iteration_tokens_total: one observation per engine iteration, with
 * the number of sequences (tokens) that iteration advanced. The single-slot
 * decode loop always passes 1; the batched path passes the live group size.
 * It used to be observed once per request in request_finish, which made the
 * histogram's count equal the request count rather than the step count. */
void cwick_metrics_observe_iteration_tokens(int64_t n) {
    if (n > 0) hist_observe(&g_hists[H_ITER], (double)n);
}

/* Mean seconds per decode step this process has measured, or 0 before there
 * is a single sample. Callers use it to turn a token budget into a wall-clock
 * estimate -- see the uncapped-max_tokens warning in engine.c, where a request
 * that omitted max_tokens is about to be handed a context-sized budget. */
double cwick_metrics_decode_step_mean_seconds(void) {
    cwick_hist *h = &g_hists[H_STEP];
    uint64_t n = atomic_load_explicit(&h->total, memory_order_relaxed);
    if (n == 0) return 0.0;
    double s = gauge_get(&h->sum_bits);
    return s > 0.0 ? s / (double)n : 0.0;
}

void cwick_metrics_set_stream(const cwick_stream *s) { g_stream = s; }
void cwick_metrics_set_memory_ceiling(uint64_t bytes) { g_ceiling = bytes; }
void cwick_metrics_set_context_window(int64_t tokens) {
    g_context_window = tokens > 0 ? (uint64_t)tokens : 0;
}
void cwick_metrics_set_requests_running(int64_t n) {
    g_running = n > 0 ? (uint64_t)n : 0;
}

void cwick_metrics_set_prefill(int64_t done_tokens, int64_t total_tokens,
                              int64_t layer, int64_t nlayer) {
    g_pf_done   = done_tokens  > 0 ? (uint64_t)done_tokens  : 0;
    g_pf_total  = total_tokens > 0 ? (uint64_t)total_tokens : 0;
    g_pf_layer  = layer  > 0 ? (uint64_t)layer  : 0;
    /* nlayer is the checkpoint's layer count and is identical every time it is
     * passed; 0 means "not stated" (an idle tick), which must not erase the
     * value init already published. */
    if (nlayer > 0) g_pf_layers = (uint64_t)nlayer;
}

/* ------------------------------------------------------------------ */
/* Rendering                                                           */
/* ------------------------------------------------------------------ */
typedef struct { char *p; size_t len, cap; } cwick_buf;

static void buf_reserve(cwick_buf *b, size_t extra) {
    if (b->len + extra + 1 <= b->cap) return;
    size_t cap = b->cap ? b->cap : 4096;
    while (b->len + extra + 1 > cap) cap *= 2;
    char *np = realloc(b->p, cap);
    if (!np) return;                 /* out of memory: drop the tail silently */
    b->p = np;
    b->cap = cap;
}
static void buf_addf(cwick_buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char tmp[512];
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    if ((size_t)n < sizeof(tmp)) {
        buf_reserve(b, (size_t)n);
        if (!b->p) return;
        memcpy(b->p + b->len, tmp, (size_t)n);
        b->len += (size_t)n;
        b->p[b->len] = '\0';
        return;
    }
    buf_reserve(b, (size_t)n + 1);
    if (!b->p) return;
    va_start(ap, fmt);
    vsnprintf(b->p + b->len, (size_t)n + 1, fmt, ap);
    va_end(ap);
    b->len += (size_t)n;
}

/* Prometheus label escaping: backslash, quote and newline, per the text
 * format spec. Everything else (including UTF-8 model names) passes through. */
static void buf_label_escape(cwick_buf *b, const char *s) {
    for (const char *c = s; *c; c++) {
        if (*c == '\\') buf_addf(b, "\\\\");
        else if (*c == '"') buf_addf(b, "\\\"");
        else if (*c == '\n') buf_addf(b, "\\n");
        else buf_addf(b, "%c", *c);
    }
}

static void emit_help_type(cwick_buf *b, const char *name, const char *help,
                           const char *type) {
    buf_addf(b, "# HELP %s %s\n# TYPE %s %s\n", name, help, name, type);
}
static void emit_counter(cwick_buf *b, const char *name, const char *help,
                         uint64_t v) {
    emit_help_type(b, name, help, "counter");
    buf_addf(b, "%s %llu\n", name, (unsigned long long)v);
}
static void emit_gauge(cwick_buf *b, const char *name, const char *help,
                       double v) {
    emit_help_type(b, name, help, "gauge");
    buf_addf(b, "%s %.6g\n", name, v);
}
static void emit_gauge_i(cwick_buf *b, const char *name, const char *help,
                         uint64_t v) {
    emit_help_type(b, name, help, "gauge");
    buf_addf(b, "%s %llu\n", name, (unsigned long long)v);
}
static void emit_info(cwick_buf *b, const char *name, const char *help,
                      const char *labels) {
    emit_help_type(b, name, help, "gauge");
    buf_addf(b, "%s{%s} 1\n", name, labels);
}

/* Pull the live numbers out of the stream before rendering: the expert cache
 * is the one thing that changes between scrapes without a request. */
static void sample_stream(void) {
    if (!g_stream) return;
    uint64_t bytes = 0, reads = 0, hits = 0, misses = 0;
    cwick_stream_stats(g_stream, &bytes, &reads, &hits, &misses);
    m_expert_bytes = bytes;
    m_expert_reads = reads;
    m_expert_hits = hits;
    m_expert_misses = misses;
    g_expert_used = cwick_stream_cache_used(g_stream);
    g_expert_cap = cwick_stream_cache_cap(g_stream);
}

char *cwick_metrics_render(size_t *out_len) {
    sample_stream();

    cwick_buf b;
    memset(&b, 0, sizeof(b));
    buf_reserve(&b, 8192);

    const char *model = g_info.model ? g_info.model : "unknown";
    /* Read the backend at scrape time, not just at init. The GPU path can be
     * turned off after startup (an explicit --backend arrives with the CLI, an
     * autotune can decide the CPU pool wins, a failed device drops the whole
     * path), and a dashboard that keeps reporting the value captured twenty
     * seconds into boot is worse than no label: a run observed at 1.3 tok/s
     * showed backend="cpu" while the same log said the Metal backend was up,
     * and there was no way to tell which of the two was stale. The predicate is
     * cached inside metal.c, so this costs a load, not a device probe. */
    const char *backend = cwick_metal_gemv_on_gpu() ? "metal" : "cpu";

    /* ---- static configuration (vLLM's *info* pattern) ---- */
    buf_reserve(&b, 1024);
    {
        cwick_buf lb;
        memset(&lb, 0, sizeof(lb));
        buf_addf(&lb, "model=\"");
        buf_label_escape(&lb, model);
        buf_addf(&lb, "\",backend=\"");
        buf_label_escape(&lb, backend);
        buf_addf(&lb, "\",spine=\"");
        buf_label_escape(&lb, g_info.spine ? g_info.spine : "bf16");
        buf_addf(&lb, "\",max_context=\"%lld\",num_layers=\"%lld\","
                      "num_experts=\"%lld\",experts_per_token=\"%lld\"",
                 (long long)g_info.max_context, (long long)g_info.num_layers,
                 (long long)g_info.num_experts,
                 (long long)g_info.experts_per_token);
        emit_info(&b, "cwick:model_info",
                  "Static description of the loaded checkpoint.", lb.p);
        free(lb.p);
    }

    /* ---- request counters (vLLM names) ---- */
    emit_counter(&b, "vllm:prompt_tokens_total",
                 "Number of prefill tokens processed.",
                 atomic_load_explicit(&m_prompt_tokens, memory_order_relaxed));
    emit_counter(&b, "vllm:generation_tokens_total",
                 "Number of generation tokens processed.",
                 atomic_load_explicit(&m_generation_tokens, memory_order_relaxed));
    emit_counter(&b, "vllm:request_success_total",
                 "Number of finished requests (all finish reasons).",
                 atomic_load_explicit(&m_requests_success, memory_order_relaxed));

    /* finish-reason breakdown, as vLLM labels it */
    emit_help_type(&b, "vllm:request_success_by_reason_total",
                   "Number of finished requests by finish reason.", "counter");
    buf_addf(&b, "vllm:request_success_by_reason_total{finished_reason=\"stop\"} %llu\n",
             (unsigned long long)atomic_load_explicit(&m_finish_stop,
                                                      memory_order_relaxed));
    buf_addf(&b, "vllm:request_success_by_reason_total{finished_reason=\"length\"} %llu\n",
             (unsigned long long)atomic_load_explicit(&m_finish_length,
                                                      memory_order_relaxed));
    buf_addf(&b, "vllm:request_success_by_reason_total{finished_reason=\"abort\"} %llu\n",
             (unsigned long long)atomic_load_explicit(&m_finish_abort,
                                                      memory_order_relaxed));
    /* loop is not in vLLM's vocabulary: it is the loop guard's early stop,
     * invisible as plain "stop" even though it is the quality signal that
     * matters most (a generation half wasted on a repetition). */
    buf_addf(&b, "vllm:request_success_by_reason_total{finished_reason=\"loop\"} %llu\n",
             (unsigned long long)atomic_load_explicit(&m_finish_loop,
                                                      memory_order_relaxed));

    emit_counter(&b, "vllm:request_failure_total",
                 "Number of requests that ended in an error.",
                 atomic_load_explicit(&m_requests_failed, memory_order_relaxed)
                 + atomic_load_explicit(&m_requests_refused, memory_order_relaxed));
    emit_counter(&b, "vllm:num_preemptions_total",
                 "Number of preemptions (always 0: cwick runs one request at a "
                 "time and never preempts).", 0);
    /* Real counters since the prefix cache landed (see src/pfx.c): every
     * request is looked up, and a hit skips the prefill of the matched
     * tokens. Names match vLLM so an existing dashboard panel works. */
    emit_counter(&b, "vllm:prefix_cache_queries_total",
                 "Prompt prefix lookups against the prefix cache.",
                 atomic_load_explicit(&m_pfx_queries, memory_order_relaxed));
    emit_counter(&b, "vllm:prefix_cache_hits_total",
                 "Lookups served by a cached prefix (those tokens were not "
                 "prefilled again).",
                 atomic_load_explicit(&m_pfx_hits, memory_order_relaxed));
    emit_counter(&b, "cwick:prefix_tokens_reused_total",
                 "Prompt tokens skipped because a cached prefix covered them.",
                 atomic_load_explicit(&m_pfx_reused, memory_order_relaxed));

    /* ---- expert residency: NOT a vLLM prefix/KV cache ----
     * These used to be emitted as `vllm:gpu_prefix_cache_*`, which made any
     * standard vLLM panel report the expert hit rate as "GPU prefix cache hit
     * rate" -- a number that has nothing to do with prompt-prefix reuse (that
     * is `vllm:prefix_cache_*` above). They are cwick-specific, so they are
     * named that way and no longer shadow a vLLM metric. */
    uint64_t hits = atomic_load_explicit(&m_expert_hits, memory_order_relaxed);
    uint64_t misses = atomic_load_explicit(&m_expert_misses, memory_order_relaxed);
    emit_counter(&b, "cwick:expert_cache_queries_total",
                 "Expert-weight lookups (hits + misses).", hits + misses);
    emit_counter(&b, "cwick:expert_cache_hits_total",
                 "Expert matrices served from the pinned RAM cache.", hits);
    emit_counter(&b, "cwick:expert_cache_misses_total",
                 "Expert matrices read from SSD.", misses);
    emit_counter(&b, "cwick:expert_reads_total",
                 "Expert matrix reads issued.",
                 atomic_load_explicit(&m_expert_reads, memory_order_relaxed));
    emit_counter(&b, "cwick:expert_bytes_streamed_total",
                 "Bytes streamed off SSD for expert weights.",
                 atomic_load_explicit(&m_expert_bytes, memory_order_relaxed));
    emit_counter(&b, "cwick:decode_steps_total",
                 "Decode forward steps executed.",
                 atomic_load_explicit(&m_decode_steps, memory_order_relaxed));
    emit_counter(&b, "cwick:requests_refused_total",
                 "Requests refused while another generation was running.",
                 atomic_load_explicit(&m_requests_refused, memory_order_relaxed));
    emit_counter(&b, "cwick:requests_aborted_total",
                 "Requests abandoned by the client or stopped by Ctrl+C.",
                 atomic_load_explicit(&m_requests_aborted, memory_order_relaxed));

    /* ---- gauges ---- */
    uint64_t used = atomic_load_explicit(&g_expert_used, memory_order_relaxed);
    uint64_t cap = atomic_load_explicit(&g_expert_cap, memory_order_relaxed);
    emit_gauge_i(&b, "vllm:num_requests_running",
                 "Number of requests currently holding the generation slot.",
                 atomic_load_explicit(&g_running, memory_order_relaxed));
    emit_gauge_i(&b, "vllm:num_requests_waiting",
                 "Number of requests queued behind the generation slot.",
                 atomic_load_explicit(&g_waiting, memory_order_relaxed));
    emit_gauge_i(&b, "vllm:num_requests_swapped",
                 "Number of requests swapped out (always 0: no swapping).", 0);
    /* vLLM's `gpu_cache_usage_perc` is KV-cache occupancy; this is the pinned
     * expert arena, so it gets its own name instead of borrowing that one
     * (a vLLM panel would read 99.9% as "KV cache full, preemption imminent"). */
    emit_gauge(&b, "cwick:expert_cache_usage_perc",
               "Pinned expert-cache occupancy, 0..1.", cap ? (double)used / (double)cap : 0.0);
    emit_gauge_i(&b, "cwick:expert_cache_bytes",
                 "Bytes of expert weights currently pinned in RAM.", used);
    emit_gauge_i(&b, "cwick:expert_cache_cap_bytes",
                 "Expert-cache budget in bytes.", cap);
    emit_gauge_i(&b, "cwick:memory_ceiling_bytes",
                 "Hard ceiling cwick steers its RSS toward (--mem-fraction of RAM).",
                 atomic_load_explicit(&g_ceiling, memory_order_relaxed));
    emit_gauge_i(&b, "cwick:context_window_tokens",
                 "Configured KV context window in tokens.",
                 atomic_load_explicit(&g_context_window, memory_order_relaxed));
    emit_gauge_i(&b, "cwick:prefill_tokens_total",
                 "Prompt tokens of the request currently prefilling (0 = idle).",
                 atomic_load_explicit(&g_pf_total, memory_order_relaxed));
    emit_gauge_i(&b, "cwick:prefill_tokens_done",
                 "Prompt tokens processed so far by the running prefill.",
                 atomic_load_explicit(&g_pf_done, memory_order_relaxed));
    emit_gauge_i(&b, "cwick:prefill_layer",
                 "1-based layer the running prefill is on (0 = idle).",
                 atomic_load_explicit(&g_pf_layer, memory_order_relaxed));
    emit_gauge_i(&b, "cwick:prefill_layers",
                 "Layers in the checkpoint (denominator for prefill_layer).",
                 atomic_load_explicit(&g_pf_layers, memory_order_relaxed));
    {
        uint64_t fp = cwick_process_footprint_bytes();
        g_rss = fp;
        emit_gauge_i(&b, "cwick:process_footprint_bytes",
                     "Physical footprint of the cwick process (resident + "
                     "compressed).", fp);
    }
    emit_gauge(&b, "cwick:uptime_seconds",
               "Seconds since cwick started.",
               g_initialised ? (double)(now_ns() - g_started_ns) / 1e9 : 0.0);

    /* ---- histograms ---- */
    for (int i = 0; i < H_COUNT; i++) {
        cwick_hist *h = &g_hists[i];
        emit_help_type(&b, h->name, h->help, "histogram");
        uint64_t running = 0;
        for (int k = 0; k < h->nbuckets; k++) {
            running += atomic_load_explicit(&h->count[k], memory_order_relaxed);
            double le = h->buckets[k];
            /* Whole-number bounds must not be emitted as 1e+06: some scrapers
             * match `le` as a string when joining across series. */
            if (le == floor(le) && le < 1e15)
                buf_addf(&b, "%s_bucket{le=\"%.0f\"} %llu\n", h->name, le,
                         (unsigned long long)running);
            else
                buf_addf(&b, "%s_bucket{le=\"%g\"} %llu\n", h->name, le,
                         (unsigned long long)running);
        }
        uint64_t total = atomic_load_explicit(&h->total, memory_order_relaxed);
        buf_addf(&b, "%s_bucket{le=\"+Inf\"} %llu\n", h->name,
                 (unsigned long long)total);
        buf_addf(&b, "%s_sum %.6g\n", h->name,
                 gauge_get(&h->sum_bits));
        buf_addf(&b, "%s_count %llu\n", h->name, (unsigned long long)total);
    }

    if (out_len) *out_len = b.len;
    if (!b.p) { b.p = calloc(1, 1); }
    return b.p;
}

const char *cwick_metrics_content_type(void) {
    /* Prometheus 3.x accepts this for UTF-8 names; 2.x ignores the version. */
    return "text/plain; version=0.0.4; charset=utf-8";
}

void cwick_metrics_log_summary(const char *tag) {
    sample_stream();
    uint64_t hits = atomic_load_explicit(&m_expert_hits, memory_order_relaxed);
    uint64_t misses = atomic_load_explicit(&m_expert_misses, memory_order_relaxed);
    uint64_t total = hits + misses;
    uint64_t steps = atomic_load_explicit(&m_decode_steps, memory_order_relaxed);
    double step_sum = gauge_get(&g_hists[H_STEP].sum_bits);
    uint64_t step_n = atomic_load_explicit(&g_hists[H_STEP].total,
                                           memory_order_relaxed);
    double e2e_sum = gauge_get(&g_hists[H_E2E].sum_bits);
    uint64_t e2e_n = atomic_load_explicit(&g_hists[H_E2E].total,
                                          memory_order_relaxed);

    CWICK_INFO(tag, "metrics: %llu request(s), %llu prompt + %llu generated tokens",
              (unsigned long long)atomic_load_explicit(&m_requests_success,
                                                       memory_order_relaxed),
              (unsigned long long)atomic_load_explicit(&m_prompt_tokens,
                                                       memory_order_relaxed),
              (unsigned long long)atomic_load_explicit(&m_generation_tokens,
                                                       memory_order_relaxed));
    if (step_n > 0)
        CWICK_INFO(tag, "metrics: decode step mean %.1f ms (%.3f tok/s) over "
                       "%llu step(s)",
                  step_sum * 1000.0 / (double)step_n,
                  step_sum > 0.0 ? (double)step_n / step_sum : 0.0,
                  (unsigned long long)step_n);
    if (e2e_n > 0)
        CWICK_INFO(tag, "metrics: e2e mean %.2f s over %llu request(s)",
                  e2e_sum / (double)e2e_n, (unsigned long long)e2e_n);
    if (total > 0)
        CWICK_INFO(tag, "metrics: expert cache %.1f%% (%llu/%llu), %.2f GiB "
                       "streamed, %llu decode step(s)",
                  100.0 * (double)hits / (double)total,
                  (unsigned long long)hits, (unsigned long long)total,
                  cwick_gib(atomic_load_explicit(&m_expert_bytes,
                                                memory_order_relaxed)),
                  (unsigned long long)steps);
}
