/* logcap.h — capture this process's own stdout/stderr into a bounded ring.
 *
 * Why this exists: `cwick serve` prints everything to stdout (startup banner,
 * the memory budget, one line per request, prefill/decode progress, tool
 * calls), and the only way to read that stream today is to be the terminal
 * that started it. A local GUI therefore cannot show what the server is doing
 * without asking the user to keep a second window open.
 *
 * The capture is a *tee*, not a redirect: fd 1 and fd 2 are pointed at two
 * pipes, one pump thread reads both, echoes every byte straight back to the
 * original descriptors and hands each completed line to the ring. Nothing is
 * lost from the terminal, and there is exactly one copy of every line in the
 * ring (not two — going through the pipe rather than hooking log.c means the
 * lines printed by Objective-C frameworks, Metal diagnostics and plain
 * fprintf() land here too, which log.c alone could never see).
 *
 * The ring is bounded in lines *and* bytes, ring-shaped, and never blocks the
 * producer: a runaway `--trace` dump costs the oldest lines, not a stall.
 */
#ifndef CWICK_LOGCAP_H
#define CWICK_LOGCAP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum { CWICK_LOGCAP_OUT = 0, CWICK_LOGCAP_ERR = 1 } cwick_logcap_src;

typedef struct {
    uint64_t seq;
    cwick_logcap_src src;
    int64_t t_ms;      /* wall clock at capture time, ms since the epoch */
    const char *text;  /* borrowed: only valid inside the callback */
} cwick_logcap_line;

/* Start capturing. `max_lines` and `max_bytes` bound the ring (0 = defaults:
 * 2000 lines / 4 MiB). Returns false when the pipes or the pump thread cannot
 * be created, in which case nothing changed and every other call here is a
 * no-op-ish empty read. Calling it twice is safe: the second call is ignored. */
bool cwick_logcap_start(size_t max_lines, size_t max_bytes);
void cwick_logcap_stop(void);
bool cwick_logcap_active(void);

/* Retained line count, the sequence number the next line will get, the oldest
 * sequence still retained, and how many lines were evicted before their time
 * (so a consumer can say "the client missed N lines" instead of silently
 * showing an incomplete tail). */
size_t cwick_logcap_lines(void);
uint64_t cwick_logcap_head(void);
uint64_t cwick_logcap_oldest(void);
uint64_t cwick_logcap_dropped(void);

/* Walk every retained line with `seq >= since`, oldest first, at most `limit`
 * of them (0 = unlimited). The ring lock is held for the whole walk, so the
 * callback must be cheap and must not call back into this module. */
void cwick_logcap_foreach(uint64_t since, size_t limit,
                         void (*cb)(const cwick_logcap_line *line, void *user),
                         void *user);

/* Wait up to `timeout_ms` for a line newer than `after`, and return the
 * current head. Returns immediately when lines are already available, which
 * makes the poll loop "wait, then drain" rather than "sleep and hope". */
uint64_t cwick_logcap_wait(uint64_t after, int timeout_ms);

#endif /* CWICK_LOGCAP_H */
