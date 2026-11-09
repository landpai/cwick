/* log.h — one logger, one stream: stdout.
 *
 * cwick serve is meant to be pointed at by a local chat app or a coding
 * agent, and its diagnostics have to land where a foreground process prints
 * them — stdout — not hidden in stderr where `--help` output would suggest.
 *
 * Three loudness levels, all going to the same stream:
 *
 *   info     always on            startup banner, memory budget, one line per
 *                                 request, one line per generation, shutdown
 *   verbose  `--verbose`          per-request detail, per-decode-step timing,
 *                                 cache/stream counters, full budget table
 *   trace    `--trace`            what the model actually saw and produced:
 *                                 prompts, outputs and tool calls
 *
 * Logging is line-atomic: locked, flushed and thread-safe, so a background
 * prefetch thread or a future worker pool cannot interleave half a line.
 *
 * Every line carries the wall-clock system time and, for the request the
 * calling thread is serving, its id:
 *
 *   cwick: 2026-10-23 15:31:30.123 [info   +12.34s] [req 7] [prefill] ...
 *   cwick: 2026-10-23 15:31:18.001 [info    +0.00s] [init] ...     (no request)
 */
#ifndef CWICK_LOG_H
#define CWICK_LOG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum {
    CWICK_LOG_INFO = 0,   /* always */
    CWICK_LOG_VERBOSE,    /* --verbose */
    CWICK_LOG_TRACE,      /* --trace */
} cwick_log_level;

/* Configure once at startup. `quiet` silences even the info level (used by
 * anyone parsing our stdout machine-readable). */
void cwick_log_config(bool verbose, bool trace, bool quiet);
bool cwick_log_enabled(cwick_log_level level);

/* Which request the calling thread is currently working on, or 0 for none.
 *
 * Every log line then carries `[req N]` right after the timestamp, so a serve
 * log can be filtered to one request without guessing whose line is whose --
 * with one thread per connection, interleaved requests used to be
 * indistinguishable. Thread-local rather than global because two connections
 * generate concurrently and must not claim each other's lines. Set to 0 when
 * the request ends (and before any line that belongs to no request, e.g. the
 * startup banner). */
void cwick_log_set_request(uint64_t id);
uint64_t cwick_log_request(void);

/* Emit one line. `tag` is a short context marker ("init", "req 7", "gen");
 * pass NULL for no marker. The line is prefixed with `cwick: `, an elapsed
 * timestamp and the level tag so long-running serves are readable hours in. */
void cwick_logf(cwick_log_level level, const char *tag, const char *fmt, ...)
    __attribute__((format(printf, 3, 4)));

/* Emit a possibly-long payload (a prompt, an assistant answer, a tool call).
 * `\n`-separated lines are re-indented under `indent` spaces and returned
 * through the same atomic path as cwick_logf. `max_chars` truncates runaway
 * generations (0 = unlimited). */
void cwick_log_block(cwick_log_level level, const char *tag, const char *body,
                    size_t max_chars, int indent);

#define CWICK_INFO(tag, fmt, ...) \
    cwick_logf(CWICK_LOG_INFO, (tag), fmt, ##__VA_ARGS__)
#define CWICK_VERB(tag, fmt, ...) \
    cwick_logf(CWICK_LOG_VERBOSE, (tag), fmt, ##__VA_ARGS__)
#define CWICK_TRAC(tag, fmt, ...) \
    cwick_logf(CWICK_LOG_TRACE, (tag), fmt, ##__VA_ARGS__)

#endif /* CWICK_LOG_H */
