/* logcap.c — see logcap.h.
 *
 * Two pipes, one pump thread, one bounded ring.
 *
 * fd 1 -> pipe A --\
 *                    >-- pump thread --+--> echo to the *saved* fd (unchanged
 * fd 2 -> pipe B --/                   |    terminal: nothing disappears)
 *                                      +--> ring (`since`-addressable, capped)
 *
 * The saved descriptors are what the caller used to print to, so `cwick serve`
 * looks identical whether or not the GUI is reading along. */
#include "logcap.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

/* Default ring bounds: enough history to explain a request that spent an hour
 * in prefill, small enough that a trace-mode dump cannot grow without limit. */
#define CWICK_LOGCAP_DEF_LINES 2000
#define CWICK_LOGCAP_DEF_BYTES (4u * 1024u * 1024u)

/* A single line is truncated to this (plus an ellipsis) rather than allowed to
 * pin unbounded memory: `--trace` can print a 100K-token prompt. */
#define CWICK_LOGCAP_LINE_MAX 4096

#define CWICK_LOGCAP_CHUNK 32768

typedef struct entry {
    struct entry *next;
    uint64_t seq;
    int64_t t_ms;
    cwick_logcap_src src;
    char *text;
    size_t len;
} entry;

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER;

static entry *g_head = NULL, *g_tail = NULL;
static size_t g_count = 0, g_max_lines = CWICK_LOGCAP_DEF_LINES;
static uint64_t g_bytes = 0, g_max_bytes = CWICK_LOGCAP_DEF_BYTES;
static uint64_t g_next_seq = 1, g_first_seq = 1, g_dropped = 0;

/* plumbing */
static pthread_t g_thread;
static bool g_thread_live = false;
static bool g_active = false;
static volatile bool g_stop = false;
static int g_read_fd[2] = {-1, -1};    /* pump reads here (out, err) */
static int g_write_fd[2] = {-1, -1};   /* what fd 1 / fd 2 were switched to */
static int g_saved[2] = {-1, -1};      /* the real stdout / stderr */
static int g_wake[2] = {-1, -1};       /* makes stop() immediate */

static int64_t now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000 + (int64_t)(tv.tv_usec / 1000);
}

/* ------------------------------------------------------------------ */
/* ring                                                                */
/* ------------------------------------------------------------------ */

/* Called with g_lock held. */
static void ring_evict_locked(void) {
    while (g_count > 0 &&
           (g_count > g_max_lines ||
            (g_max_bytes > 0 && g_bytes > g_max_bytes))) {
        entry *e = g_head;
        g_head = e->next;
        if (!g_head) g_tail = NULL;
        g_bytes -= (uint64_t)e->len;
        g_count--;
        g_first_seq = e->seq + 1;
        g_dropped++;
        free(e->text);
        free(e);
    }
}

/* Push one completed line. `text` is copied; `len` excludes the terminator and
 * may already be free of trailing CR/LF. */
static void ring_push(cwick_logcap_src src, const char *text, size_t len) {
    if (len == 0) return;                       /* a bare newline: nothing */

    /* A truncated line is marked rather than silently shortened: someone
     * reading it has to be able to tell "that is the whole line" from "there
     * was more". "…" is 3 bytes in UTF-8. */
    char stack[64];
    char *heap = NULL;
    char *body = NULL;
    size_t blen = 0;
    if (len > CWICK_LOGCAP_LINE_MAX) {
        size_t keep = CWICK_LOGCAP_LINE_MAX - 3;
        heap = (char *)malloc(keep + 3 + 1);
        if (!heap) return;
        memcpy(heap, text, keep);
        memcpy(heap + keep, "\xe2\x80\xa6", 3);
        heap[keep + 3] = '\0';
        body = heap;
        blen = keep + 3;
    } else {
        if (len < sizeof(stack)) {
            memcpy(stack, text, len);
            stack[len] = '\0';
            body = stack;
            blen = len;
        } else {
            heap = (char *)malloc(len + 1);
            if (!heap) return;
            memcpy(heap, text, len);
            heap[len] = '\0';
            body = heap;
            blen = len;
        }
    }

    entry *e = (entry *)calloc(1, sizeof(*e));
    if (!e) { free(heap); return; }             /* OOM: lose lines, do not die */
    if (body == stack) {
        e->text = (char *)malloc(blen + 1);
        if (!e->text) { free(e); return; }
        memcpy(e->text, body, blen + 1);
    } else {
        e->text = body;                          /* ownership moves here */
    }
    e->len = blen + 1;
    e->seq = g_next_seq;
    e->t_ms = now_ms();
    e->src = src;
    e->next = NULL;

    pthread_mutex_lock(&g_lock);
    if (g_tail) g_tail->next = e; else g_head = e;
    g_tail = e;
    if (g_count == 0) g_first_seq = e->seq;
    g_count++;
    g_bytes += (uint64_t)e->len;
    g_next_seq = e->seq + 1;
    /* Trim *after* appending, so the bound is a real bound: evicting first
     * and then adding one leaves max+1 lines retained. */
    ring_evict_locked();
    pthread_mutex_unlock(&g_lock);
    pthread_cond_broadcast(&g_cv);
}

void cwick_logcap_foreach(uint64_t since, size_t limit,
                         void (*cb)(const cwick_logcap_line *line, void *user),
                         void *user) {
    if (!cb) return;
    pthread_mutex_lock(&g_lock);
    size_t seen = 0;
    for (entry *e = g_head; e; e = e->next) {
        if (e->seq < since) continue;
        if (limit && seen >= limit) break;
        cwick_logcap_line l;
        l.seq = e->seq;
        l.src = e->src;
        l.t_ms = e->t_ms;
        l.text = e->text;
        cb(&l, user);
        seen++;
    }
    pthread_mutex_unlock(&g_lock);
}

uint64_t cwick_logcap_wait(uint64_t after, int timeout_ms) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    long ms = timeout_ms > 0 ? timeout_ms : 0;
    if (ms > 60000) ms = 60000;
    ts.tv_sec += ms / 1000;
    ts.tv_nsec += (ms % 1000) * 1000000L;
    if (ts.tv_nsec >= 1000000000L) { ts.tv_nsec -= 1000000000L; ts.tv_sec++; }

    pthread_mutex_lock(&g_lock);
    if (g_next_seq > after) {
        pthread_mutex_unlock(&g_lock);
        return g_next_seq;
    }
    pthread_cond_timedwait(&g_cv, &g_lock, &ts);
    uint64_t head = g_next_seq;
    pthread_mutex_unlock(&g_lock);
    return head;
}

size_t cwick_logcap_lines(void) {
    pthread_mutex_lock(&g_lock);
    size_t n = g_count;
    pthread_mutex_unlock(&g_lock);
    return n;
}
uint64_t cwick_logcap_head(void) {
    pthread_mutex_lock(&g_lock);
    uint64_t n = g_next_seq;
    pthread_mutex_unlock(&g_lock);
    return n;
}
uint64_t cwick_logcap_oldest(void) {
    pthread_mutex_lock(&g_lock);
    uint64_t n = g_count ? g_first_seq : g_next_seq;
    pthread_mutex_unlock(&g_lock);
    return n;
}
uint64_t cwick_logcap_dropped(void) {
    pthread_mutex_lock(&g_lock);
    uint64_t n = g_dropped;
    pthread_mutex_unlock(&g_lock);
    return n;
}
bool cwick_logcap_active(void) { return g_active; }

/* ------------------------------------------------------------------ */
/* pump                                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    char *buf;
    size_t len, cap;
} partial;

static void partial_append(partial *p, const char *data, size_t n) {
    if (p->len + n + 1 > p->cap) {
        size_t want = p->cap ? p->cap : 4096;
        while (want < p->len + n + 1) want *= 2;
        char *nb = (char *)realloc(p->buf, want);
        if (!nb) return;                 /* drop bytes rather than crash */
        p->buf = nb;
        p->cap = want;
    }
    memcpy(p->buf + p->len, data, n);
    p->len += n;
    p->buf[p->len] = '\0';
}

static void partial_reset(partial *p) { p->len = 0; if (p->buf) p->buf[0] = '\0'; }

static void emit_partial(cwick_logcap_src src, partial *p) {
    if (p->len == 0) return;
    size_t n = p->len;
    while (n > 0 && (p->buf[n - 1] == '\r' || p->buf[n - 1] == '\n')) n--;
    ring_push(src, p->buf, n);
    partial_reset(p);
}

/* Split a chunk into lines. A line without a trailing newline yet is held
 * back until it either completes or the writer goes quiet for long enough to
 * look like a prompt-style partial write. */
static void feed(cwick_logcap_src src, const char *data, size_t n, partial *p) {
    size_t i = 0;
    while (i < n) {
        const char *nl = (const char *)memchr(data + i, '\n', n - i);
        if (!nl) {
            size_t take = n - i;
            /* Keep a pathological runaway write bounded. */
            if (p->len + take >= CWICK_LOGCAP_LINE_MAX * 4) {
                partial_append(p, data + i, take);
                emit_partial(src, p);
            } else {
                partial_append(p, data + i, take);
            }
            return;
        }
        size_t take = (size_t)(nl - (data + i));
        partial_append(p, data + i, take);
        emit_partial(src, p);
        i += take + 1;
    }
}

static void write_saved(int which, const char *data, size_t n) {
    int fd = g_saved[which];
    if (fd < 0) return;
    size_t off = 0;
    while (off < n) {
        ssize_t w = write(fd, data + off, n - off);
        if (w <= 0) {
            if (errno == EINTR) continue;
            return;
        }
        off += (size_t)w;
    }
}

static void *pump_thread(void *arg) {
    (void)arg;
    char ibuf[CWICK_LOGCAP_CHUNK];
    partial part[2];
    memset(part, 0, sizeof(part));

    struct pollfd pfd[3];
    pfd[0].fd = g_read_fd[0]; pfd[1].fd = g_read_fd[1]; pfd[2].fd = g_wake[0];
    pfd[0].events = pfd[1].events = pfd[2].events = POLLIN;
    pfd[0].revents = pfd[1].revents = pfd[2].revents = 0;

    /* Non-blocking on every fd this thread reads, permanently — the pipe ends
     * *and* the wake pipe. Two different blocking reads used to park this
     * thread forever: one that exactly filled the buffer (which looked like
     * "more is obviously waiting"), and the wake-pipe drain loop, which after
     * consuming the single stop byte happily waited for a second one that is
     * never coming. Either one takes pthread_join(), and therefore Ctrl+C,
     * down with it. Non-blocking turns "nothing left" into EAGAIN, which is
     * just a return. */
    {
        int fds[3] = { g_read_fd[0], g_read_fd[1], g_wake[0] };
        for (int i = 0; i < 3; i++) {
            int fl = fcntl(fds[i], F_GETFL, 0);
            if (fl >= 0) (void)fcntl(fds[i], F_SETFL, fl | O_NONBLOCK);
        }
    }

    while (!g_stop) {
        int rc = poll(pfd, 3, 200);
        if (rc < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (pfd[2].revents & POLLIN) {
            char drain[256];
            while (read(g_wake[0], drain, sizeof(drain)) > 0) { }
        }
        for (int i = 0; i < 2; i++) {
            if (!(pfd[i].revents & (POLLIN | POLLHUP | POLLERR))) continue;
            for (;;) {
                ssize_t n = read(g_read_fd[i], ibuf, sizeof(ibuf));
                if (n > 0) {
                    write_saved(i, ibuf, (size_t)n);
                    feed((cwick_logcap_src)i, ibuf, (size_t)n, &part[i]);
                    if ((size_t)n < sizeof(ibuf)) break;   /* drained */
                    continue;                              /* maybe more */
                }
                if (n < 0 && errno == EINTR) continue;
                break;                                     /* EAGAIN or EOF */
            }
        }
    }

    /* Last look before shutdown: whatever was written after the final poll
     * must still reach the terminal and the ring. The reads are non-blocking
     * on purpose — the write ends are still open (our own fds), so a blocking
     * read here would never return EOF and the join below would hang forever,
     * taking the Ctrl+C path with it. */
    for (int i = 0; i < 2; i++) {
        int fl = fcntl(g_read_fd[i], F_GETFL, 0);
        if (fl >= 0) (void)fcntl(g_read_fd[i], F_SETFL, fl | O_NONBLOCK);
        for (;;) {
            ssize_t n = read(g_read_fd[i], ibuf, sizeof(ibuf));
            if (n > 0) {
                write_saved(i, ibuf, (size_t)n);
                feed((cwick_logcap_src)i, ibuf, (size_t)n, &part[i]);
                continue;
            }
            if (n < 0 && errno == EINTR) continue;
            break;
        }
        emit_partial((cwick_logcap_src)i, &part[i]);
    }
    free(part[0].buf);
    free(part[1].buf);
    return NULL;
}

bool cwick_logcap_start(size_t max_lines, size_t max_bytes) {
    if (g_active) return true;
    if (max_lines) g_max_lines = max_lines;
    if (max_bytes) g_max_bytes = max_bytes;

    int out_pipe[2] = {-1, -1}, err_pipe[2] = {-1, -1}, wake[2] = {-1, -1};
    if (pipe(out_pipe) != 0) return false;
    if (pipe(err_pipe) != 0) { close(out_pipe[0]); close(out_pipe[1]); return false; }
    if (pipe(wake) != 0) {
        close(out_pipe[0]); close(out_pipe[1]);
        close(err_pipe[0]); close(err_pipe[1]);
        return false;
    }

    /* Flush first: anything buffered belongs to the *old* descriptors. */
    fflush(stdout);
    fflush(stderr);

    int saved_out = dup(STDOUT_FILENO);
    int saved_err = dup(STDERR_FILENO);
    if (saved_out < 0 || saved_err < 0) {
        close(out_pipe[0]); close(out_pipe[1]);
        close(err_pipe[0]); close(err_pipe[1]);
        close(wake[0]); close(wake[1]);
        if (saved_out >= 0) close(saved_out);
        if (saved_err >= 0) close(saved_err);
        return false;
    }

    g_read_fd[0] = out_pipe[0]; g_write_fd[0] = out_pipe[1];
    g_read_fd[1] = err_pipe[0]; g_write_fd[1] = err_pipe[1];
    g_wake[0] = wake[0];         g_wake[1] = wake[1];
    g_saved[0] = saved_out;      g_saved[1] = saved_err;
    g_stop = false;

    if (pthread_create(&g_thread, NULL, pump_thread, NULL) != 0) {
        close(out_pipe[0]); close(out_pipe[1]);
        close(err_pipe[0]); close(err_pipe[1]);
        close(wake[0]); close(wake[1]);
        close(saved_out); close(saved_err);
        g_read_fd[0] = g_read_fd[1] = -1;
        g_write_fd[0] = g_write_fd[1] = -1;
        g_wake[0] = g_wake[1] = -1;
        g_saved[0] = g_saved[1] = -1;
        return false;
    }
    g_thread_live = true;

    dup2(g_write_fd[0], STDOUT_FILENO);
    dup2(g_write_fd[1], STDERR_FILENO);
    /* Line-buffered, like a terminal: every complete line goes out at once, so
     * the ring sees whole lines rather than fragments of one printf. */
    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IOLBF, 0);
    g_active = true;
    return true;
}

void cwick_logcap_stop(void) {
    if (!g_active) return;
    g_active = false;

    /* Put the descriptors back *before* stopping the pump, so the last bytes
     * leaving buffered stdout still land on the real terminal. */
    fflush(stdout);
    fflush(stderr);
    dup2(g_saved[0], STDOUT_FILENO);
    dup2(g_saved[1], STDERR_FILENO);
    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    g_stop = true;
    if (g_wake[1] >= 0) {
        char b = 1;
        ssize_t ignored = write(g_wake[1], &b, 1);
        (void)ignored;
    }
    if (g_thread_live) pthread_join(g_thread, NULL);
    g_thread_live = false;

    for (int i = 0; i < 2; i++) {
        if (g_read_fd[i] >= 0) close(g_read_fd[i]);
        if (g_write_fd[i] >= 0) close(g_write_fd[i]);
        if (g_saved[i] >= 0) close(g_saved[i]);
        g_read_fd[i] = g_write_fd[i] = g_saved[i] = -1;
    }
    if (g_wake[0] >= 0) close(g_wake[0]);
    if (g_wake[1] >= 0) close(g_wake[1]);
    g_wake[0] = g_wake[1] = -1;

    /* Keep the history: it is still the record of this run, and the server
     * may be answering a last /v1/logs read on its way out. */
}
