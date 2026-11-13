/* server.c — OpenAI-compatible HTTP server.
 *
 *   - /v1/models, /v1/completions, /v1/chat/completions (+ /models aliases);
 *   - SSE streaming: begin chunk, content deltas, finish frame, optional
 *     usage frame, then `data: [DONE]`;
 *   - fail-closed validation: anything the server cannot honor exactly is
 *     refused with an OpenAI-shaped HTTP 400 before the model is entered;
 *   - one generation at a time, queued in arrival order: a concurrent
 *     generation request takes a ticket and is served to completion rather
 *     than refused (503 while a swap or shutdown runs, `Retry-After` set);
 *   - sampling is honoured, not just accepted: temperature / top_p / top_k and
 *     the two repetition penalties are implemented (sample.c), and whatever a
 *     request leaves out comes from the checkpoint's generation_config.json --
 *     so a model trained with do_sample=true is decoded by drawing, which is
 *     the only thing standing between it and a repetition loop. Greedy stays
 *     the default for a checkpoint that says nothing, and for temperature 0.
 *
 * The server is text-only — except that it speaks the checkpoint's own tool
 * protocol (tools / tool_choice arrive asking for functions, the model
 * answers with <tool_call> blocks, and those are handed back as OpenAI
 * tool_calls).
 *
 * Diagnostics: every request and every generation prints one line to stdout
 * (`--verbose` adds detail, `--trace` also prints prompts, outputs and tool
 * calls). Ctrl+C interrupts cleanly (see main.c + the poll loop below).
 */
#include "cwick.h"
#include "ds_prompt.h"
#include "mm_prompt.h"
#include "q35_prompt.h"
#include "json.h"
#include "sample.h"
#include "tools.h"
#include "jev.h"
#include "hub.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <malloc/malloc.h>   /* malloc_zone_pressure_relief, on model unload */
#endif

#include "auth.h"
#include "log.h"
#include "logcap.h"
#include "mem.h"
#include "metrics.h"
#include "models.h"
#include "util.h"
#include "pfx.h"
#include "qr.h"
#include "store.h"
#include "webui.h"

#define CWICK_SERVER_MAX_BODY (8u * 1024u * 1024u)
#define CWICK_SERVER_ID_BUF 64
/* Read timeout on an accepted socket: long enough for a slow client, short
 * enough that a Ctrl+C is noticed while someone is trickling bytes at us. */
#define CWICK_CLIENT_READ_TIMEOUT_MS 2000
/* How long shutdown waits for in-flight connections before proceeding
 * anyway. A generation aborts within one decode step of the stop flag, and
 * a socket read cannot outlive SO_RCVTIMEO, so this is pure defence. */
#define CWICK_DRAIN_TIMEOUT_S 30

/* ---- logging: everything lands on stdout through log.c --------------
 * The request/execution lines are printed at the info level by default;
 * `--verbose` unlocks per-request detail and `--trace` also prints what the
 * model actually saw and produced. The historical CWICK_SERVER_LOG knob still
 * works: `CWICK_SERVER_LOG=0` silences these lines, `=verbose` asks for the
 * detail level without passing --verbose. */
static bool srv_log_enabled(void) {
    const char *e = getenv("CWICK_SERVER_LOG");
    if (!e || *e == '\0' || *e == '1' || strcmp(e, "verbose") == 0) return 1;
    return 0;
}
static bool srv_log_verbose_env(void) {
    const char *e = getenv("CWICK_SERVER_LOG");
    return e && strcmp(e, "verbose") == 0;
}

#define CWICK_LOG(fmt, ...) CWICK_INFO(NULL, fmt, ##__VA_ARGS__)
#define CWICK_VDLOG(fmt, ...) CWICK_VERB(NULL, fmt, ##__VA_ARGS__)

/* How many requests may be generating at once. Measured: two independent
 * server processes served two requests in 226 s against 334 s for one process
 * doing both, so the SSD and the GPU are not
 * saturated by a single stream and a second one really does add throughput
 * (1.48x at 74% parallel efficiency). Three slots is the point where the
 * extra KV and the extra cache pressure stop paying. */
#define CWICK_GEN_SLOTS 3

typedef struct {
    /* Slot 0 owns the weight spine; the others borrow it (see
     * cwick_engine_options.shared_spine) so the 15.46 GiB mlocked spine is
     * paid for once. Per-sequence state is per slot. */
    cwick_engine *eng[CWICK_GEN_SLOTS];
    cwick_tokenizer *tok;
    /* Token ids of the two think markers, probed once per loaded tokenizer
     * (startup and every model switch). -1 means the tokenizer does not hand
     * the marker over as a single token, which turns the thinking budget off
     * rather than splitting a marker across steps. */
    int64_t think_open_id;
    int64_t think_close_id;
    /* The same pair as text, for the answer splitter: the markers differ per
     * checkpoint (`<think>` vs `<mm:think>`) and the splitter works on bytes.
     * NULL means the `<think>` pair. */
    const char *think_open_mk;
    const char *think_close_mk;
    /* Guards slot_busy only. A generation holds no lock: it takes a slot,
     * releases this mutex, and decodes outside it, so /v1/metrics and
     * /v1/models stay answers while three requests are generating. */
    pthread_mutex_t gen_lock;
    bool slot_busy[CWICK_GEN_SLOTS];
    /* FIFO entry into the slots.
     *
     * A request takes a ticket under gen_lock, then waits until its ticket
     * falls inside the serving window [ticket_served, ticket_served + nslots).
     * Because tickets are handed out in arrival order, the window always
     * admits the oldest waiters first, and releasing a slot advances the
     * window by one -- so with one slot the requests are served strictly in
     * arrival order, and nothing is ever turned away for being busy.
     *
     * The wait is timed on purpose: a shutdown must be noticed even if the
     * current holder never releases. */
    pthread_cond_t gen_cv;
    uint64_t ticket_next;   /* next ticket to hand out */
    uint64_t ticket_served; /* tickets that have released their slot */
    int nwaiting;           /* queued, not yet granted */
    /* Socket of the request currently generating in each slot, so the engine
     * can notice a client that hung up and stop instead of decoding the whole
     * budget for nobody. Guarded by gen_lock. */
    int slot_fd[CWICK_GEN_SLOTS];
    /* Whether to actually abort generation when that peer is gone.
     *
     * Off by default, deliberately. Detecting a hangup means reading the
     * socket, and the only signals available are "FIN received" and "reset" --
     * both of which are also produced by clients that shut down their write
     * side yet are still waiting for the answer. Acting on that would drop a
     * live request, and to the caller a dropped request is indistinguishable
     * from a dead model; an agent that probes concurrently would read its own
     * abandoned retries as the endpoint going away.
     *
     * The only thing that keeps an abandoned request from monopolising the
     * slot is the client's own max_tokens (or the context ceiling), so this
     * stays an opt-in escape hatch: CWICK_ABORT_ON_DISCONNECT=1. */
    bool abort_on_disconnect;
    /* How many of the CWICK_GEN_SLOTS entries in `eng` are actually live.
     * The refusal message reports this rather than the array size, so a
     * single-slot server does not claim "all 3 slots are in use" and mislead
     * a client into retrying against capacity that was never there. */
    int nslots;
    /* In-flight connection tracking: every accepted socket is served by its
     * own detached thread (a generation runs for minutes and must not block
     * /metrics, /v1/models or any other client), and shutdown drains them
     * before the engine is freed. */
    pthread_mutex_t client_mtx;
    pthread_cond_t client_cv;
    int clients;
    bool running;
    const char *host;
    int port;
    int64_t max_tokens;
    int64_t max_request_bytes;
    cwick_pfx_cache *pfx;         /* prefix cache, NULL when disabled */
    /* Snapshot writes run here instead of on the request's thread. A snapshot
     * is hundreds of megabytes and used to be written between the prefill and
     * the first decode step, where it cost seconds of time-to-first-token and
     * (through the page-cache pressure it created) pushed the resident weight
     * bank out, slowing the decode that followed. See pfx_submit below. */
    struct {
        pthread_mutex_t mtx;
        pthread_cond_t cv;
        pthread_t th;
        bool started;
        bool stop;
        struct pfx_job *head, *tail;
        int queued;
    } pfxw;
    /* Guards the cache's index and its files: one connection thread can be
     * looking a prompt up (which reads and may evict) while another is
     * storing the snapshot its prefill just produced. */
    pthread_mutex_t pfx_mtx;
    int64_t pfx_block;       /* checkpoint stride during a long prefill */
    bool verbose;
    bool trace;            /* print prompts / outputs / tool calls */
    bool web;              /* serve the built-in chat UI at `/` */
    /* Serve POST /v1/systemone: typed decisions read out of one prefill,
     * using the resident model and tokenizer. Off unless asked for, because it
     * is a second way to drive the same engine and a deployment that only
     * wants chat should not advertise it. */
    bool jev;
    cwick_auth *auth;       /* single-user login; NULL = loopback-style auth off */
    /* App storage. `store` is the primary database: the API keys, and — when
     * `store_single` is set — every app's data too, which is the layout an
     * existing `--store file.db` / CWICK_STORE keeps. Otherwise the apps are
     * split: `<store_dir>/<app>.db` per app, opened on first use and kept in
     * `store_slots` so a phone and a tab share one file handle per app. */
    cwick_store *store;
    bool store_single;
    char store_dir[1024];
    struct store_slot {
        char app[64];
        cwick_store *st;
    } store_slots[16];
    int store_slots_n;
    pthread_mutex_t store_mtx;
    /* Every model this instance knows about: the resident one plus whatever
     * `--models-dir` turned up. Read-mostly; only POST /v1/models/switch
     * touches membership, and it does so with no request in flight. */
    cwick_model_list catalog;
    cwick_model_entry *resident;   /* the entry currently loaded, never NULL-less */
    const char *models_dirs[8];
    int models_dirs_n;
    /* The local model library: what is on disk under ./models, what is being
     * downloaded, and the read-only catalog used to find more. Independent of
     * the engine — a download must never hold a generation slot. */
    cwick_hub *hub;
    /* A model swap frees the engine and the tokenizer out from under anything
     * using them. `swapping` refuses new generations for the duration, and
     * `tok_users` counts the connections that may touch srv->tok, so the swap
     * can only begin when both are zero — see srv_switch_model(). */
    bool swapping;
    int tok_users;
    /* Engine template, remembered so a switch can rebuild exactly what the
     * startup built (same KV window, same spine choice, same slot count). */
    char spine_kind[16];
    int64_t context_capacity;
    int slot_want;
    /* Reports the live engine back to the caller (main) after a swap. */
    cwick_engine **live_engine;
    uint64_t request_seq;  /* access-log numbering */
    int listen_fd;
    int wake_r, wake_w;    /* internal self-pipe used as a poll() wakeup */
    volatile int *stop_owner; /* signal-safe external cancellation flag */
} server_ctx;

/* Upper bound on concurrently served connections. Generations are serialised
 * by the slot queue anyway (a second one waits its turn), so this only caps
 * the cheap endpoints and slow-loris style connection floods. */
#define CWICK_MAX_CLIENTS 64

/* Shortest prompt worth a prefix snapshot. The floor is low on purpose: a
 * "hello" turn on this checkpoint costs ~19 s of prefill (the expert-union
 * read dominates, so the cost barely depends on token count), and the file a
 * 32-token snapshot takes is under a megabyte against an 80 GiB budget. The
 * original 256 meant the first turns of a session were never cached at all,
 * so nothing could hit until the conversation had grown past a page. */
#define CWICK_PFX_MIN_TOKENS 32

/* Format a request tag: "req 12". Thread-local: with one thread per
 * connection a shared rotating buffer would hand the same tag to two
 * requests, or interleave a half-written one into the log. */
static const char *req_tag(uint64_t id) {
    static __thread char buf[32];
    snprintf(buf, sizeof(buf), "req %llu", (unsigned long long)id);
    return buf;
}

/* ------------------------------------------------------------------ */
/* HTTP plumbing                                                       */
/* ------------------------------------------------------------------ */

typedef struct {
    char method[8];
    char path[256];
    char cookie[512];
    char authorization[512];
    /* `X-API-Key:` — several agent platforms send this instead of (or in
     * addition to) an Authorization header, and it is the one credential a
     * non-browser client can set without touching cookies. */
    char api_key[512];
    int64_t content_length;
    const char *body;
    size_t body_len;
} http_req;

/* bounded substring search (buffer may contain NULs) */
static const char *find_bytes(const char *buf, size_t n, const char *needle,
                              size_t nlen) {
    if (nlen == 0) return buf;
    if (n < nlen) return NULL;
    for (size_t i = 0; i + nlen <= n; i++) {
        if (memcmp(buf + i, needle, nlen) == 0) return buf + i;
    }
    return NULL;
}

static int read_http_request(int fd, char *buf, size_t cap, http_req *req,
                             cwick_error *err) {
    size_t n = 0;
    /* read until "\r\n\r\n" (headers end) */
    for (;;) {
        if (n >= cap - 1) {
            cwick_error_set(err, CWICK_ERR_ARGUMENT, "HTTP request too large");
            return -1;
        }
        ssize_t r = read(fd, buf + n, cap - 1 - n);
        if (r <= 0) {
            cwick_error_set(err, CWICK_ERR_IO, "read failed");
            return -1;
        }
        n += (size_t)r;
        buf[n] = '\0';
        if (find_bytes(buf, n, "\r\n\r\n", 4) != NULL) break;
    }
    const char *hdr_end = find_bytes(buf, n, "\r\n\r\n", 4);
    const char *line_end = find_bytes(buf, (size_t)(hdr_end - buf), "\r\n", 2);
    if (!hdr_end || !line_end) {
        cwick_error_set(err, CWICK_ERR_PARSE, "no header terminator");
        return -1;
    }
    if (getenv("CWICK_DBG_HTTP")) {
        fprintf(stderr, "cwick: DBG n=%zu hdr_end_off=%td line_end_off=%td\n",
                n, hdr_end - buf, line_end - buf);
    }
    /* request line: buf .. line_end */
    {
        const char *sp = find_bytes(buf, (size_t)(line_end - buf), " ", 1);
        if (!sp) { cwick_error_set(err, CWICK_ERR_PARSE, "malformed request line"); return -1; }
        size_t mlen = (size_t)(sp - buf);
        if (mlen >= sizeof(req->method)) mlen = sizeof(req->method) - 1;
        memcpy(req->method, buf, mlen);
        req->method[mlen] = '\0';
        const char *path = sp + 1;
        const char *sp2 = find_bytes(path, (size_t)(line_end - path), " ", 1);
        size_t plen = sp2 ? (size_t)(sp2 - path) : (size_t)(line_end - path);
        if (plen >= sizeof(req->path)) plen = sizeof(req->path) - 1;
        memcpy(req->path, path, plen);
        req->path[plen] = '\0';
    }

    /* headers: Content-Length */
    req->content_length = 0;
    const char *h = line_end + 2;
    while (h < hdr_end) {
        const char *he = find_bytes(h, (size_t)(hdr_end + 2 - h), "\r\n", 2);
        if (!he) break;
        const char *colon = find_bytes(h, (size_t)(he - h), ":", 1);
        if (colon) {
            char name[64], value[512];
            size_t nlen = (size_t)(colon - h);
            if (nlen >= sizeof(name)) nlen = sizeof(name) - 1;
            memcpy(name, h, nlen);
            name[nlen] = '\0';
            const char *vs = colon + 1;
            while (vs < he && *vs == ' ') vs++;
            size_t vlen = (size_t)(he - vs);
            if (vlen >= sizeof(value)) vlen = sizeof(value) - 1;
            memcpy(value, vs, vlen);
            value[vlen] = '\0';
            if (strcasecmp(name, "Content-Length") == 0) {
                req->content_length = strtoll(value, NULL, 10);
            } else if (strcasecmp(name, "Cookie") == 0) {
                snprintf(req->cookie, sizeof(req->cookie), "%s", value);
            } else if (strcasecmp(name, "Authorization") == 0) {
                snprintf(req->authorization, sizeof(req->authorization), "%s", value);
            } else if (strcasecmp(name, "X-API-Key") == 0) {
                snprintf(req->api_key, sizeof(req->api_key), "%s", value);
            }
        }
        h = he + 2;
    }
    const char *body = hdr_end + 4;
    size_t got = (size_t)((buf + n) - body);
    if (req->content_length < 0 || (size_t)req->content_length > cap) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "body exceeds server limit");
        return -1;
    }
    while (got < (size_t)req->content_length) {
        if (n >= cap - 1) {
            cwick_error_set(err, CWICK_ERR_ARGUMENT, "body too large");
            return -1;
        }
        ssize_t r = read(fd, buf + n, cap - 1 - n);
        if (r <= 0) { cwick_error_set(err, CWICK_ERR_IO, "body read failed"); return -1; }
        n += (size_t)r;
        got += (size_t)r;
    }
    buf[n] = '\0';
    req->body = body;
    req->body_len = (size_t)req->content_length;
    return 0;
}

static void write_all(int fd, const char *s, size_t n) {
    size_t off = 0;
    while (off < n) {
        ssize_t w = write(fd, s + off, n - off);
        if (w <= 0) return;
        off += (size_t)w;
    }
}

/* Shared head writer. `extra` is a raw "\r\n"-terminated header block or NULL. */
static void http_response_ex(int fd, int status, const char *status_text,
                             const char *content_type, const char *extra,
                             const char *body, bool stream) {
    char head[640];
    int64_t len = body ? (int64_t)strlen(body) : 0;
    snprintf(head, sizeof(head),
             "HTTP/1.1 %d %s\r\nContent-Type: %s\r\n"
             "X-Content-Type-Options: nosniff\r\nContent-Length: %lld\r\n"
             "Cache-Control: no-store\r\nConnection: close\r\n%s\r\n",
             status, status_text, content_type, (long long)len,
             extra ? extra : "");
    write_all(fd, head, strlen(head));
    if (body) write_all(fd, body, strlen(body));
    (void)stream;
}

static void http_response(int fd, int status, const char *status_text,
                          const char *content_type, const char *body,
                          bool stream) {
    http_response_ex(fd, status, status_text, content_type, NULL, body, stream);
}

/* Binary response (the QR PNG). http_response() deliberately uses strlen for
 * JSON/text, so it cannot carry a byte string with NULs. */
static void http_response_bytes(int fd, int status, const char *status_text,
                                const char *content_type, const void *body,
                                size_t len) {
    char head[512];
    snprintf(head, sizeof(head),
             "HTTP/1.1 %d %s\r\nContent-Type: %s\r\n"
             "X-Content-Type-Options: nosniff\r\nContent-Length: %zu\r\n"
             "Cache-Control: no-store\r\nConnection: close\r\n\r\n",
             status, status_text, content_type, len);
    write_all(fd, head, strlen(head));
    if (body && len) write_all(fd, body, len);
}

/* Same as write_all but reports whether the peer is still there, so a streaming
 * generation can be aborted the moment the client hangs up. */
static bool write_all_ok(int fd, const void *buf, size_t n) {
    const char *p = (const char *)buf;
    size_t off = 0;
    while (off < n) {
        ssize_t w = write(fd, p + off, n - off);
        if (w <= 0) return false;
        off += (size_t)w;
    }
    return true;
}

static bool sse_frame(int fd, const char *json) {
    return write_all_ok(fd, "data: ", 6) &&
           write_all_ok(fd, json, strlen(json)) &&
           write_all_ok(fd, "\n\n", 2);
}

/* A named SSE frame (`event: name` + `data: {...}`). The Responses API uses
 * event names for every frame, where chat/completions puts the type inside
 * the JSON payload only. */
static bool sse_named(int fd, const char *event, const char *json) {
    return write_all_ok(fd, "event: ", 7) &&
           write_all_ok(fd, event, strlen(event)) &&
           write_all_ok(fd, "\ndata: ", 7) &&
           write_all_ok(fd, json, strlen(json)) &&
           write_all_ok(fd, "\n\n", 2);
}

/* Streaming response head: SSE has no length, so the body is delimited by the
 * connection close (the server closes every connection anyway). */
static void http_stream_head(int fd) {
    static const char head[] =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream; charset=utf-8\r\n"
        "Cache-Control: no-store\r\n"
        "X-Content-Type-Options: nosniff\r\n"
        "Connection: close\r\n\r\n";
    write_all(fd, head, sizeof(head) - 1);
}

/* ------------------------------------------------------------------ */
/* OpenAI response shapes                                              */
/* ------------------------------------------------------------------ */

static int64_t now_unix(void) { return (int64_t)time(NULL); }

/* Wall clock in milliseconds. Used for the request timings the /metrics
 * registry reports (queue time, TTFT, end-to-end latency). */
static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

static uint64_t g_response_seq;
static void make_id(char *buf, size_t cap, const char *kind) {
    uint64_t seq = ++g_response_seq;
    snprintf(buf, cap, "%s-%lld-%ld-%llu", kind, (long long)now_unix(),
             (long)getpid(), (unsigned long long)seq);
}

/* ------------------------------------------------------------------ */
/* Fail-closed request validation                                      */
/* ------------------------------------------------------------------ */

/* Which dialect this request arrived in. The prompt, the queue and the
 * engine are shared; only the envelope (and, for the non-OpenAI dialects,
 * the input shape) differs. `is_chat` stays the "messages in, messages out"
 * switch the prompt renderers read, so a Responses or Anthropic request is a
 * chat request answered in another dialect. */
typedef enum {
    SRV_API_CHAT = 0,
    SRV_API_COMPLETION = 1,
    SRV_API_RESPONSES = 2,
    SRV_API_MESSAGES = 3,   /* Anthropic /v1/messages */
} srv_api;

typedef struct {
    bool stream;
    bool include_usage;
    int64_t max_tokens;
    bool max_tokens_set;
    char *prompt;         /* raw completion prompt */
    char *system;         /* chat system (first system message; trace only) */
    char *user;           /* chat user (last user message; trace only) */
    cwick_chat_msg *msgs;  /* the whole conversation, in order */
    size_t n_msgs;
    /* The raw OpenAI `messages` / `tools` arrays. `msgs` above is the
     * flattened view GLM's template needs; a checkpoint that ships its own
     * Python prompt format (DeepSeek) reads fields GLM ignores --
     * reasoning_content, task, wo_eos -- so it renders from the original
     * JSON instead of the reduced struct. Kept for both backends because the
     * model can be switched between validation and generation. */
    char *messages_json;
    char *tools_json;
    char *reasoning_effort; /* "Max" | "Low" | "High" (GLM reasoning_effort) */
    /* Canonical thinking-intensity name for the arrival log line:
     * "低" / "中" / "高" / "超高" / "极致" / "关" (static strings, not owned).
     * set by validate_body for chat requests; NULL for completions. */
    const char *effort_level;
    bool effort_defaulted;   /* no effort field was sent (log: "default") */
    bool is_chat;
    /* Which dialect to answer in; see srv_api. `is_chat` is true for chat,
     * responses and messages, because all three hand the engine a
     * conversation rather than a raw prompt. */
    srv_api api;
    /* Responses only: the client asked for a reasoning summary
     * (`reasoning.summary`), so the reasoning block goes out as
     * reasoning_summary_* events instead of raw reasoning_text deltas. */
    bool resp_summary;
    /* Responses only: the part of the response object that repeats the
     * request (tools, tool_choice, parallel_tool_calls, temperature, ...).
     * Built once from the request, then appended to every response object
     * the endpoint emits — the buffered body, the `response.created`
     * envelope and the terminal `response.completed`. Those fields are not
     * decoration: `tools`, `tool_choice` and `parallel_tool_calls` are
     * non-optional on the wire, and a client that diffs the response
     * against what it asked for reads them. */
    char *resp_tail;
    /* tools: declarations offered to the model in its own protocol */
    cwick_tool_set tools;
    char *tools_block;    /* rendered <tools> section (NULL when unused) */
    uint64_t rid;         /* request id for logging */
    /* Filled in by run_generation once the prompt is tokenized: the caller
     * needs it for `vllm:prompt_tokens` after the result comes back. */
    int64_t prompt_tokens;
    /* Sampling: only the fields this request actually sent (the rest keep the
     * checkpoint's generation_config.json values, see apply_sampling). */
    cwick_sampler_cfg smp;
    bool has_temperature, has_top_p, has_top_k, has_presence, has_frequency;
} gen_request;

static void free_request(gen_request *gr) {
    for (size_t i = 0; i < gr->n_msgs; i++) {
        free((char *)gr->msgs[i].role);
        free((char *)gr->msgs[i].content);
        free((char *)gr->msgs[i].tool_calls_json);
    }
    free(gr->msgs);
    gr->msgs = NULL;
    gr->n_msgs = 0;
    free(gr->messages_json);
    gr->messages_json = NULL;
    free(gr->tools_json);
    gr->tools_json = NULL;
    free(gr->resp_tail);
    gr->resp_tail = NULL;
}

/* ------------------------------------------------------------------ */
/* Thinking intensity (思考强度) and thinking budget (思考预算)         */
/* ------------------------------------------------------------------ */
/* These are one knob seen from two sides, so they live in ONE table:
 *
 *   intensity  what the model is TOLD -- a "Reasoning Effort" header
 *              rendered into the prompt. The model may ignore it.
 *   budget     what the engine ENFORCES -- a token ceiling checked on every
 *              decode step, past which the engine writes the close tag
 *              itself so the turn answers instead of thinking forever.
 *
 * Keeping a level's two halves in the same row is the point: a level cannot
 * gain a budget without also gaining the header that asks the model to use
 * it, the one place where they deliberately disagree is visible rather than
 * an accident of two parallel if-chains, and `level` string comparisons
 * cannot fall out of step between the three call sites that used to exist.
 *
 * The ceiling exists because a thought that never closes costs the whole
 * turn: measured on agent traffic, one request spent all 8096 tokens it had
 * inside the think block and delivered `content=0, finish=length`, which a
 * client can only answer with a full retry.
 *
 * The `glm` column is squashed to three buckets on purpose. The GLM
 * checkpoint (zai-org/GLM-5.3-Flash/chat_template.jinja) was trained with
 * exactly three Reasoning Effort headers and its C renderer passes the
 * string through verbatim, so 低 and 中 -- which differ in budget, 512 vs
 * 2048 -- necessarily share `Low`. DeepSeek takes a number in 1..100 and
 * can express all six.
 *
 * `关` is 0, meaning "answer, do not think": the engine expresses that as
 * the template's own empty block (<think></think>), not as a truncated
 * thought. It is also the only level that never opens a block. */
typedef struct {
    const char *level;      /* canonical Chinese name; what the logs print */
    const char *glm;        /* GLM header, rendered verbatim */
    int         ds_effort;  /* DeepSeek header: 1..100 (0 = renderer default) */
    int64_t     budget;     /* token ceiling; 0 = answer without thinking */
    bool        thinks;     /* false = never open a block */
} think_level;

static const think_level THINK_LEVELS[] = {
    /* level   glm     ds   budget  thinks */
    { "关",   "Low",   0,       0,  false },
    { "低",   "Low",  25,     512,  true  },
    { "中",   "Low",  50,    2048,  true  },
    { "高",   "High", 75,    8192,  true  },
    { "超高", "Max",  90,   16384,  true  },
    { "极致", "Max", 100,   32768,  true  },
};
#define THINK_LEVEL_N (sizeof(THINK_LEVELS) / sizeof(THINK_LEVELS[0]))

/* What a request gets when it sends no reasoning/thinking field at all.
 * Kept beside the ladder rather than inside the parser, so the renderers,
 * the log fallback and the regression test all read the same value. A
 * default is an ordinary level, not an "unset" sentinel: handing the
 * renderers a sentinel used to leave each template to open a reasoning
 * block while the engine was told to close it immediately. */
#define CWICK_THINK_DEFAULT_LEVEL  "关"
#define CWICK_THINK_DEFAULT_EFFORT "Low"

static const think_level *think_level_find(const char *level) {
    if (!level) return NULL;
    for (size_t i = 0; i < THINK_LEVEL_N; i++)
        if (strcmp(level, THINK_LEVELS[i].level) == 0) return &THINK_LEVELS[i];
    return NULL;
}

/* Every spelling cwick accepts, mapped to one canonical level. English
 * spellings are matched case-insensitively; anything unrecognised is refused
 * rather than silently clamped (fail-closed, like every other field). The
 * header string a backend renders comes from THINK_LEVELS, never from here:
 * the spelling table only decides which level was asked for. */
bool cwick_thinking_effort_resolve(const char *raw, const char **level,
                                  const char **effort) {
    if (level) *level = NULL;
    if (effort) *effort = NULL;
    if (!raw || !*raw) return false;
    static const struct { const char *name; const char *level; } table[] = {
        /* Chinese spellings first: byte-exact, no case folding needed. */
        { "低",       "低"   },
        { "中",       "中"   },
        { "高",       "高"   },
        { "超高",     "超高" },
        { "极致",     "极致" },
        { "关闭",     "关"   },
        { "关",       "关"   },
        { "无",       "关"   },
        /* English (and OpenAI-compatible) spellings. */
        { "low",      "低"   },
        { "minimal",  "低"   },
        { "medium",   "中"   },
        { "mid",      "中"   },
        { "high",     "高"   },
        { "ultra",    "超高" },
        { "veryhigh", "超高" },
        { "xhigh",    "超高" },
        { "extreme",  "极致" },
        { "max",      "极致" },
        { "none",     "关"   },
        { "off",      "关"   },
        { "disabled", "关"   },
    };
    /* ASCII-lowercase copy for the English half (C locale, no ctype). */
    char buf[32];
    size_t n = strlen(raw);
    if (n >= sizeof(buf)) return false;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)raw[i];
        buf[i] = (char)((c >= 'A' && c <= 'Z') ? c + 32 : c);
    }
    buf[n] = '\0';
    for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); i++) {
        if (strcmp(raw, table[i].name) == 0 || strcmp(buf, table[i].name) == 0) {
            const think_level *tl = think_level_find(table[i].level);
            if (level) *level = table[i].level;
            if (effort) *effort = tl ? tl->glm : NULL;
            return true;
        }
    }
    return false;
}

/* The no-field default must stay equal to an explicit `none`. Exposed for
 * the regression test so the two paths cannot silently diverge again. */
const char *cwick_thinking_default_level(void) {
    return CWICK_THINK_DEFAULT_LEVEL;
}

static const cwick_model_entry *resident_entry(server_ctx *srv);
static const char *resident_served_name(server_ctx *srv);

/* The token ceiling a level turns into (THINK_LEVELS' `budget` column).
 * -1 means "no ceiling at all", which an absent or unrecognised level gets:
 * the spelling resolver has already refused anything unknown, so this is a
 * safety net rather than a policy. */
static int64_t think_budget_for_level(const char *level) {
    const think_level *tl = think_level_find(level);
    return tl ? tl->budget : -1;
}

/* CWICK_THINK_BUDGET overrides the ceiling for an A/B: a number forces one
 * budget for every request (0 = never think), negative turns the mechanism
 * off. It deliberately does NOT touch the prompt header -- the point of the
 * knob is to hold the model's instructions fixed and vary only the ceiling,
 * which is how the DeepSeek enforcement bug was found: a run with
 * CWICK_THINK_BUDGET=64 still sent the `高` header (effort 75) and still got
 * 64 tokens to honour it in. */
static int64_t think_budget_effective(const char *level) {
    int64_t from_level = think_budget_for_level(level);
    const char *v = getenv("CWICK_THINK_BUDGET");
    if (!v || !*v) return from_level;
    char *end = NULL;
    long long n = strtoll(v, &end, 10);
    if (end == v) return from_level;   /* not a number: keep the level */
    return (int64_t)n;
}

/* Whether a reasoning marker sits in the last few prompt tokens. Most
 * templates end with the marker itself, so the last token answers it. Two
 * templates instead end with the newlines that follow it (`<think>\n` or
 * `</think>\n\n`), and there the marker is the last *non-whitespace* token --
 * hence a small window rather than an exact test. The window is only used for
 * those two families, because for the others a stray marker near the end of a
 * user message would be read as part of the template. Used for both the
 * opening and the closing marker. */
static bool prompt_token_near_end(const uint32_t *ids, int64_t nids,
                                  int64_t marker_id, int window) {
    if (!ids || marker_id < 0 || nids <= 0) return false;
    int64_t start = nids - window;
    if (start < 0) start = 0;
    for (int64_t i = nids - 1; i >= start; i--)
        if ((int64_t)ids[i] == marker_id) return true;
    return false;
}

/* Probe the two markers once per loaded tokenizer. They have to be single
 * tokens: the engine writes them as whole steps, and splitting a marker
 * across steps would corrupt the block it is trying to close. A tokenizer
 * that disagrees simply gets no budget rather than a guess. */
static void srv_probe_think_ids(server_ctx *srv) {
    srv->think_open_id = -1;
    srv->think_close_id = -1;
    if (!srv->tok) return;
    /* Which markers close a reasoning block is a property of the checkpoint,
     * not of this server: GLM and DeepSeek write <think>, MiniMax writes
     * <mm:think> (and its tokenizer happens to carry <think> too, as an
     * unrelated control token, so a fixed pair would arm the wrong one). */
    static const char *glm_marks[2] = { "<think>", "</think>" };
    static const char *mm_marks[2] = { "<mm:think>", "</mm:think>" };
    const cwick_model_spec *ms = srv->eng[0] ? cwick_engine_spec(srv->eng[0])
                                             : NULL;
    const bool is_mm = ms && strncmp(ms->model_type, "minimax_m3", 10) == 0;
    const char **marks = is_mm ? mm_marks : glm_marks;
    srv->think_open_mk = marks[0];
    srv->think_close_mk = marks[1];
    int64_t *dst[2] = { &srv->think_open_id, &srv->think_close_id };
    for (int i = 0; i < 2; i++) {
        cwick_error e;
        cwick_error_clear(&e);
        uint32_t *ids = NULL;
        int64_t n = 0;
        if (cwick_tokenizer_encode(srv->tok, marks[i], &ids, &n, &e) == CWICK_OK &&
            ids && n == 1)
            *dst[i] = (int64_t)ids[0];
        free(ids);
    }
    const bool on = srv->think_open_id >= 0 && srv->think_close_id >= 0;
    CWICK_VERB("init", "think markers: <think>=%lld </think>=%lld — thinking "
               "budget %s", (long long)srv->think_open_id,
               (long long)srv->think_close_id,
               on ? "armed (ceiling per effort level)"
                  : "off (markers are not single tokens)");
    /* Print the level table once. The two halves of a level are deliberately
     * one row (see THINK_LEVELS), so an operator asking "what does 高 actually
     * do" should not have to read two functions to find out. */
    if (on) {
        char map[320];
        size_t o = 0;
        for (size_t i = 0; i < THINK_LEVEL_N && o + 1 < sizeof(map); i++) {
            int w = snprintf(map + o, sizeof(map) - o, "%s%s→%s/%lld",
                             i ? "  " : "", THINK_LEVELS[i].level,
                             THINK_LEVELS[i].glm,
                             (long long)THINK_LEVELS[i].budget);
            if (w < 0) break;
            o += (size_t)w;
            if (o >= sizeof(map)) o = sizeof(map) - 1;
        }
        map[o] = '\0';
        CWICK_VERB("init", "thinking levels (header / token ceiling): %s", map);
    }
}

/* Resolve the decision endpoint's per-checkpoint adapter. Called once after a
 * tokenizer is loaded and again after every model switch, because that is
 * exactly when it can change: the prompt grammar follows the family and the
 * readable option set follows the tokenizer. Doing it here (rather than
 * lazily inside the request) means the log answers "which interface will the
 * next decision use" without waiting for a caller. */
static void srv_probe_decision_adapter(server_ctx *srv) {
    if (!srv->jev || !srv->eng[0] || !srv->tok) return;
    const cwick_model_spec *ms = cwick_engine_spec(srv->eng[0]);
    const char *name = resident_served_name(srv);
    jev_adapter a;
    char err[256];
    if (jev_probe(ms, srv->tok, &a, err, sizeof(err)) != 0) {
        CWICK_INFO("init", "decision adapter unavailable for %s: %s", name, err);
        return;
    }
    CWICK_INFO("init", "decision adapter for %s: family=%s prompt=%s letters=%d/%d"
                      " (%s), re-resolved on every model switch",
               name, a.family, a.prompt, a.letters, JEV_MAX_OPTIONS,
               a.first_missing < 0
                   ? "every option is one token"
                   : a.space_lead ? "space-led letters" : "bare letters");
}

static int validate_body(json_value *root, gen_request *gr, char *errbuf,
                         size_t errcap, const char *model_id,
                         const cwick_model_entry *cur) {
    /* unknown / unsupported fields -> refuse (fail-closed) */
    static const char *allowed_chat[] = {
        "model", "messages", "max_tokens", "max_completion_tokens", "stream",
        "stream_options", "temperature", "top_p", "top_k", "presence_penalty",
        "frequency_penalty", "n", "best_of", "user",
        "metadata", "reasoning_effort", "store", "response_format",
        "tools", "tool_choice", "thinking", "enable_thinking", "reasoning", NULL,
    };
    static const char *allowed_comp[] = {
        "model", "prompt", "max_tokens", "stream", "stream_options",
        "temperature", "top_p", "top_k", "presence_penalty",
        "frequency_penalty", "n", "best_of", "user", "metadata",
        "tools", "tool_choice", "thinking", "enable_thinking", "reasoning", NULL,
    };
    const char **allowed = gr->is_chat ? allowed_chat : allowed_comp;

    if (root->kind != JSON_OBJECT) {
        snprintf(errbuf, errcap, "request body must be a JSON object");
        return 400;
    }
    for (size_t i = 0; i < root->count; i++) {
        const char *k = root->members[i].key;
        bool ok = false;
        for (const char **a = allowed; *a; a++) if (strcmp(k, *a) == 0) { ok = true; break; }
        if (!ok) {
            snprintf(errbuf, errcap, "unsupported field `%s`", k);
            return 400;
        }
    }

    /* model must match what we advertise.
     *
     * Up to three names are accepted: the checkpoint's own model_type (what
     * the engine spec reports), the resident entry's served alias (the name
     * /v1/models advertises — a directory-derived release name like
     * "glm-5.3-flash"), and the catalog id (it can differ from model_type
     * once two local checkpoints share one: the catalog disambiguates them as
     * `name@2`, and a caller that learned that longer name from GET /v1/models
     * must not then be told it is wrong). */
    json_value *m = cwick_json_get(root, "model");
    if (m && m->kind != JSON_STRING) {
        snprintf(errbuf, errcap, "model must be a string");
        return 400;
    }
    if (m && strcmp(m->string, model_id) != 0 &&
        !(cur && strcmp(m->string, cur->id) == 0) &&
        !(cur && cur->alias[0] && strcmp(m->string, cur->alias) == 0)) {
        snprintf(errbuf, errcap, "model `%s` is not served by this endpoint "
                                 "(the resident model is `%s`)",
                 m->string, cur && cur->alias[0] ? cur->alias : model_id);
        return 400;
    }

    /* n / best_of must be 1 */
    json_value *n = cwick_json_get(root, "n");
    if (n && n->kind != JSON_NUMBER) {
        snprintf(errbuf, errcap, "n must be a number");
        return 400;
    }
    if (n && cwick_json_int(n, 1) != 1) {
        snprintf(errbuf, errcap, "n must be 1 (one generation at a time)");
        return 400;
    }
    json_value *bo = cwick_json_get(root, "best_of");
    if (bo && bo->kind != JSON_NUMBER) {
        snprintf(errbuf, errcap, "best_of must be a number");
        return 400;
    }
    if (bo && cwick_json_int(bo, 1) != 1) {
        snprintf(errbuf, errcap, "best_of must be 1");
        return 400;
    }

    /* refused OpenAI features */
    json_value *stop = cwick_json_get(root, "stop");
    if (stop && stop->kind != JSON_NULL && !(stop->kind == JSON_STRING && stop->count == 0) &&
        !(stop->kind == JSON_ARRAY && cwick_json_array_len(stop) == 0)) {
        snprintf(errbuf, errcap, "non-default stop sequences are not implemented");
        return 400;
    }
    /* `presence_penalty` / `frequency_penalty` used to be here. They are
     * implemented now (see sample.c) — the repetition loop they break is not a
     * theoretical concern on a sampling checkpoint: a greedy decode of one
     * produced "Owner: Owner: Owner: ..." for 5282 tokens. */
    const char *refused[] = { "logit_bias",
                              "logprobs", "seed", "suffix", "echo",
                              "response_format" };
    for (size_t i = 0; i < sizeof(refused) / sizeof(refused[0]); i++) {
        json_value *v = cwick_json_get(root, refused[i]);
        if (v && v->kind != JSON_NULL) {
            if (strcmp(refused[i], "response_format") == 0) {
                /* text-only: allow {"type":"text"} */
                json_value *ty = cwick_json_get(v, "type");
                if (ty && ty->kind == JSON_STRING && strcmp(ty->string, "text") == 0) continue;
            }
            snprintf(errbuf, errcap, "`%s` is not implemented", refused[i]);
            return 400;
        }
    }
    /* store must be false if present */
    json_value *store = cwick_json_get(root, "store");
    if (store && cwick_json_bool(store, false)) {
        snprintf(errbuf, errcap, "store=true is not implemented");
        return 400;
    }

    /* ---- sampling: temperature / top_p / top_k / repetition penalties ----
     * These are read, not just tolerated. A checkpoint that ships
     * generation_config.json with do_sample=true is *trained to be drawn
     * from*, and the argmax of a sampling-trained model is the repetition
     * loop: same clause, same label, same line until the budget runs out. What
     * is recorded here is only what the client actually sent; the fields it
     * did not send keep the checkpoint's own defaults (see apply_sampling), so
     * "top_p only" does not silently mean temperature 0. */
    cwick_sampler_cfg_default(&gr->smp);
    gr->has_temperature = gr->has_top_p = gr->has_top_k = false;
    gr->has_presence = gr->has_frequency = false;
    {
        json_value *tv = cwick_json_get(root, "temperature");
        if (tv && tv->kind != JSON_NULL) {
            if (tv->kind != JSON_NUMBER) {
                snprintf(errbuf, errcap, "temperature must be a number");
                return 400;
            }
            gr->smp.temperature = (float)cwick_json_num(tv, 0.0);
            gr->has_temperature = true;
        }
        json_value *pv = cwick_json_get(root, "top_p");
        if (pv && pv->kind != JSON_NULL) {
            if (pv->kind != JSON_NUMBER) {
                snprintf(errbuf, errcap, "top_p must be a number");
                return 400;
            }
            gr->smp.top_p = (float)cwick_json_num(pv, 1.0);
            gr->has_top_p = true;
        }
        json_value *kv = cwick_json_get(root, "top_k");
        if (kv && kv->kind != JSON_NULL) {
            if (kv->kind != JSON_NUMBER) {
                snprintf(errbuf, errcap, "top_k must be a number");
                return 400;
            }
            gr->smp.top_k = (int)cwick_json_int(kv, 0);
            gr->has_top_k = true;
        }
        json_value *ppv = cwick_json_get(root, "presence_penalty");
        if (ppv && ppv->kind != JSON_NULL) {
            if (ppv->kind != JSON_NUMBER) {
                snprintf(errbuf, errcap, "presence_penalty must be a number");
                return 400;
            }
            gr->smp.presence_penalty = (float)cwick_json_num(ppv, 0.0);
            gr->has_presence = true;
        }
        json_value *fpv = cwick_json_get(root, "frequency_penalty");
        if (fpv && fpv->kind != JSON_NULL) {
            if (fpv->kind != JSON_NUMBER) {
                snprintf(errbuf, errcap, "frequency_penalty must be a number");
                return 400;
            }
            gr->smp.frequency_penalty = (float)cwick_json_num(fpv, 0.0);
            gr->has_frequency = true;
        }
        /* Fail-closed on the range: a client that sends temperature 7 wants
         * something we cannot give, and silently clamping it to 2 would look
         * like support. */
        if ((gr->has_temperature || gr->has_top_p || gr->has_top_k ||
             gr->has_presence || gr->has_frequency) &&
            !cwick_sampler_cfg_validate(&gr->smp, errbuf, errcap))
            return 400;
    }

    /* ---- tools: declared functions reach the prompt in GLM-5.3's own
     * <tools>/<tool_call> protocol, and are read back out of the raw text.
     * Anything we cannot honour exactly is still refused here. ---- */
    if (!gr->is_chat) {
        json_value *t = cwick_json_get(root, "tools");
        if (t && t->kind != JSON_NULL) {
            snprintf(errbuf, errcap, "tools are only supported on "
                                     "/v1/chat/completions");
            return 400;
        }
    }
    int tcode = cwick_tools_parse_request(root, &gr->tools, errbuf, errcap);
    if (tcode != 0) return tcode;
    if (gr->tools.present && gr->tools.advertise)
        gr->tools_block = cwick_tools_render_block(&gr->tools);
    /* Same reason as `messages_json`: DeepSeek renders the schema block from
     * the OpenAI tool objects, GLM renders it from cwick_tool_set. */
    if (gr->is_chat) {
        json_value *tj = cwick_json_get(root, "tools");
        if (tj && tj->kind == JSON_ARRAY && cwick_json_array_len(tj) > 0)
            gr->tools_json = cwick_json_write(tj);
    }

    json_value *stream_value = cwick_json_get(root, "stream");
    if (stream_value && stream_value->kind != JSON_BOOL) {
        snprintf(errbuf, errcap, "stream must be a boolean");
        return 400;
    }
    gr->stream = cwick_json_bool(stream_value, false);
    gr->max_tokens = 0;
    json_value *mt = cwick_json_get(root, "max_tokens");
    json_value *mct = cwick_json_get(root, "max_completion_tokens");
    if (mt || mct) gr->max_tokens_set = true;
    if (mt && mt->kind != JSON_NUMBER) {
        snprintf(errbuf, errcap, "max_tokens must be a number");
        return 400;
    }
    if (mct && mct->kind != JSON_NUMBER) {
        snprintf(errbuf, errcap, "max_completion_tokens must be a number");
        return 400;
    }
    if (mt) gr->max_tokens = cwick_json_int(mt, 0);
    if (mct) {
        int64_t v = cwick_json_int(mct, 0);
        if (mt && v != gr->max_tokens) {
            snprintf(errbuf, errcap,
                     "max_tokens and max_completion_tokens disagree");
            return 400;
        }
        gr->max_tokens = v;
    }
    if (gr->max_tokens < 0) {
        snprintf(errbuf, errcap, "max_tokens must be non-negative");
        return 400;
    }
    gr->include_usage = false;
    json_value *so = cwick_json_get(root, "stream_options");
    if (so && so->kind == JSON_OBJECT) {
        gr->include_usage = cwick_json_bool(cwick_json_get(so, "include_usage"), false);
    }

    if (gr->is_chat) {
        json_value *msgs = cwick_json_get(root, "messages");
        if (!msgs || msgs->kind != JSON_ARRAY || cwick_json_array_len(msgs) == 0) {
            snprintf(errbuf, errcap, "`messages` must be a non-empty JSON array");
            return 400;
        }
        /* The whole conversation is rendered, not just the last turn: this
         * checkpoint's chat format has a turn marker per message, and the
         * engine re-prefills the prompt on every request anyway. */
        gr->system = NULL;
        gr->user = NULL;
        size_t nmsg = cwick_json_array_len(msgs);
        gr->msgs = cwick_xmalloc((nmsg ? nmsg : 1) * sizeof(*gr->msgs));
        /* Kept verbatim for the DeepSeek renderer: its template is defined
         * over the OpenAI message objects themselves, not over this struct. */
        gr->messages_json = cwick_json_write(msgs);
        for (size_t i = 0; i < nmsg; i++) {
            json_value *msg = cwick_json_array_at(msgs, i);
            json_value *mr = cwick_json_get(msg, "role");
            json_value *mc = cwick_json_get(msg, "content");
            if (!mr || mr->kind != JSON_STRING) {
                snprintf(errbuf, errcap, "message role must be a string");
                return 400;
            }
            if (mc && mc->kind != JSON_STRING && mc->kind != JSON_NULL) {
                snprintf(errbuf, errcap, "multimodal content is not supported (text-only server)");
                return 400;
            }
            const char *content = (mc && mc->kind == JSON_STRING) ? mc->string : "";
            /* the tool_calls of an assistant turn are re-emitted in the
             * checkpoint's <tool_call> form so the model sees its own calls */
            char *tc = NULL;
            if (strcmp(mr->string, "assistant") == 0) {
                json_value *tcs = cwick_json_get(msg, "tool_calls");
                if (tcs && tcs->kind == JSON_ARRAY && cwick_json_array_len(tcs) > 0)
                    tc = cwick_json_write(tcs);
            }
            gr->msgs[gr->n_msgs].role = cwick_strdup(mr->string);
            gr->msgs[gr->n_msgs].content = cwick_strdup(content);
            gr->msgs[gr->n_msgs].tool_calls_json = tc;
            gr->n_msgs++;
            if (strcmp(mr->string, "system") == 0 && !gr->system)
                gr->system = cwick_strdup(content);
            if (strcmp(mr->string, "user") == 0) {
                free(gr->user);
                gr->user = cwick_strdup(content);
            }
        }
        if (!gr->user && gr->api != SRV_API_RESPONSES &&
            gr->api != SRV_API_MESSAGES) {
            snprintf(errbuf, errcap, "chat needs a user message");
            return 400;
        }

        /* reasoning_effort / thinking / reasoning / enable_thinking: GLM
         * clients steer the checkpoint's reasoning budget through any of
         * these spellings. All collapse onto the five 思考强度 levels
         * (低/中/高/超高/极致, plus 关), which map onto the three headers
         * the template was trained with — see cwick_thinking_effort_resolve.
         * Precedence: reasoning_effort > thinking > reasoning > enable_thinking.
         * An unknown spelling is a 400, never a silent clamp.
         * A body with none of these fields defaults to 关: a client that
         * never heard of the knob gets an answer, not a 32768-token
         * reasoning block. The default is a real level and reaches the
         * renderers, so the prompt and the engine agree on whether a
         * block is opened. */
        const char *level = CWICK_THINK_DEFAULT_LEVEL;
        const char *effort = CWICK_THINK_DEFAULT_EFFORT;
        bool from_client = false;
        json_value *re = cwick_json_get(root, "reasoning_effort");
        if (re && re->kind == JSON_STRING) {
            if (!cwick_thinking_effort_resolve(re->string, &level, &effort)) {
                snprintf(errbuf, errcap, "reasoning_effort must be one of "
                         "低/中/高/超高/极致 (low/medium/high/ultra/extreme, "
                         "or none/off to keep thinking at its minimum)");
                return 400;
            }
            from_client = true;
        }
        if (!from_client) {
            json_value *th = cwick_json_get(root, "thinking");
            if (th && th->kind == JSON_OBJECT) {
                json_value *ty = cwick_json_get(th, "type");
                json_value *te = cwick_json_get(th, "effort");
                if (ty && ty->kind == JSON_STRING &&
                    strcmp(ty->string, "disabled") == 0) {
                    level = "关";
                    effort = "Low";
                    from_client = true;
                } else if (te && te->kind == JSON_STRING) {
                    if (!cwick_thinking_effort_resolve(te->string, &level,
                                                      &effort)) {
                        snprintf(errbuf, errcap, "thinking.effort must be one "
                                 "of 低/中/高/超高/极致 "
                                 "(low/medium/high/ultra/extreme)");
                        return 400;
                    }
                    from_client = true;
                }
            }
        }
        if (!from_client) {
            json_value *rs = cwick_json_get(root, "reasoning");
            if (rs && rs->kind == JSON_BOOL) {
                if (cwick_json_bool(rs, true)) { level = "极致"; effort = "Max"; }
                else { level = "关"; effort = "Low"; }
                from_client = true;
            } else if (rs && rs->kind == JSON_OBJECT) {
                json_value *re2 = cwick_json_get(rs, "effort");
                if (re2 && re2->kind == JSON_STRING) {
                    if (!cwick_thinking_effort_resolve(re2->string, &level,
                                                      &effort)) {
                        snprintf(errbuf, errcap, "reasoning.effort must be one "
                                 "of 低/中/高/超高/极致 "
                                 "(low/medium/high/ultra/extreme)");
                        return 400;
                    }
                    from_client = true;
                }
            }
        }
        if (!from_client) {
            json_value *et = cwick_json_get(root, "enable_thinking");
            if (et && et->kind == JSON_BOOL) {
                if (!cwick_json_bool(et, true)) { level = "关"; effort = "Low"; }
                from_client = true;
            }
        }
        gr->effort_level = level;
        gr->effort_defaulted = !from_client;
        gr->reasoning_effort = cwick_strdup(effort);
    } else {
        json_value *p = cwick_json_get(root, "prompt");
        if (!p || p->kind != JSON_STRING) {
            snprintf(errbuf, errcap, "`prompt` must be a JSON string");
            return 400;
        }
        gr->prompt = cwick_strdup(p->string);
    }

    /* On receipt — before queueing, prefill or any engine work: the two
     * facts that correlate a client's expectations with what the engine
     * will do. `stream` decides whether the client sees tokens as they are
     * produced or waits in the dark for the whole body; the thinking
     * intensity decides how long the reasoning block runs before the first
     * content byte. Both belong to the request itself, so they are logged
     * even when the request is later refused, queued or aborted. */
    if (gr->is_chat)
        CWICK_INFO(req_tag(gr->rid), "stream=%s thinking=%s (effort=%s%s)",
                  gr->stream ? "true" : "false",
                  gr->effort_level ? gr->effort_level : CWICK_THINK_DEFAULT_LEVEL,
                  gr->reasoning_effort ? gr->reasoning_effort
                                       : CWICK_THINK_DEFAULT_EFFORT,
                  gr->effort_defaulted ? ", default" : "");
    else
        CWICK_INFO(req_tag(gr->rid),
                  "stream=%s thinking=off (completions have no reasoning)",
                  gr->stream ? "true" : "false");
    return 0;
}

/* ------------------------------------------------------------------ */
/* Generation + response                                               */
/* ------------------------------------------------------------------ */

static char *srv_asprintf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

static char *srv_asprintf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) return NULL;
    char *out = cwick_xmalloc((size_t)n + 1);
    va_start(ap, fmt);
    vsnprintf(out, (size_t)n + 1, fmt, ap);
    va_end(ap);
    return out;
}

/* A growable text buffer for endpoints whose body has no fixed bound (the
 * model list once more than one model is known, and the log tail). */
typedef struct { char *p; size_t len, cap; } srv_buf;

static void sbuf_grow(srv_buf *b, size_t extra) {
    if (b->len + extra + 1 <= b->cap) return;
    size_t want = b->cap ? b->cap : 256;
    while (want < b->len + extra + 1) want *= 2;
    b->p = cwick_xrealloc(b->p, want);
    b->cap = want;
}
static void sbuf_putn(srv_buf *b, const char *s, size_t n) {
    sbuf_grow(b, n);
    memcpy(b->p + b->len, s, n);
    b->len += n;
    b->p[b->len] = '\0';
}
static void sbuf_puts(srv_buf *b, const char *s) { sbuf_putn(b, s, strlen(s)); }
static void sbuf_putf(srv_buf *b, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));
static void sbuf_putf(srv_buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char *piece = NULL;
    if (vasprintf(&piece, fmt, ap) < 0 || !piece) { va_end(ap); return; }
    va_end(ap);
    sbuf_puts(b, piece);
    free(piece);
}

/* ------------------------------------------------------------------ */
/* Responses API: input items -> the conversation the engine already    */
/* speaks. One adapter, so the prompt renderer, the queue, the prefix   */
/* cache and the thinking budget stay the same code as chat/completions */
/* — only the envelope differs.                                         */
/* ------------------------------------------------------------------ */

typedef struct {
    char role[24];
    char call_id[96];        /* role "tool": the call this output answers */
    srv_buf text;
    srv_buf calls;           /* comma-joined function_call objects */
    size_t ncalls;
} resp_turn;

typedef struct { resp_turn *v; size_t n, cap; } resp_turns;

static void resp_turns_free(resp_turns *t) {
    for (size_t i = 0; i < t->n; i++) {
        free(t->v[i].text.p);
        free(t->v[i].calls.p);
    }
    free(t->v);
    t->v = NULL;
    t->n = t->cap = 0;
}

static resp_turn *resp_turn_new(resp_turns *t, const char *role) {
    if (t->n == t->cap) {
        t->cap = t->cap ? t->cap * 2 : 8;
        t->v = cwick_xrealloc(t->v, t->cap * sizeof(*t->v));
    }
    resp_turn *r = &t->v[t->n++];
    memset(r, 0, sizeof(*r));
    snprintf(r->role, sizeof(r->role), "%s", role);
    return r;
}

static resp_turn *resp_turn_last(resp_turns *t) {
    return t->n ? &t->v[t->n - 1] : NULL;
}

/* A Responses content field is either a string or an array of typed parts.
 * Only text parts exist here: this server has no vision or file input, and a
 * silently dropped image would be a wrong prompt rather than a smaller one. */
static bool resp_text_of(json_value *v, srv_buf *out, char *err, size_t cap) {
    if (!v || v->kind == JSON_NULL) return true;
    if (v->kind == JSON_STRING) {
        sbuf_puts(out, v->string);
        return true;
    }
    if (v->kind != JSON_ARRAY) {
        snprintf(err, cap, "content must be a string or an array of text parts");
        return false;
    }
    for (size_t i = 0; i < v->count; i++) {
        json_value *part = v->items[i];
        json_value *ty = cwick_json_get(part, "type");
        const char *t = (ty && ty->kind == JSON_STRING) ? ty->string : NULL;
        if (t && (strcmp(t, "input_image") == 0 || strcmp(t, "image") == 0 ||
                  strcmp(t, "input_file") == 0 || strcmp(t, "file") == 0 ||
                  strcmp(t, "input_audio") == 0)) {
            snprintf(err, cap, "%s content is not supported (text-only server)", t);
            return false;
        }
        json_value *tx = cwick_json_get(part, "text");
        if (tx && tx->kind == JSON_STRING) sbuf_puts(out, tx->string);
    }
    return true;
}

/* One input item -> one turn. Assistant tool calls merge into the assistant
 * turn before them, because that is how both prompt templates expect a turn
 * with several calls to look. */
static bool resp_add_item(resp_turns *t, json_value *item,
                          char *err, size_t cap) {
    if (item->kind != JSON_OBJECT) {
        snprintf(err, cap, "input items must be objects");
        return false;
    }
    json_value *ty = cwick_json_get(item, "type");
    const char *type = (ty && ty->kind == JSON_STRING) ? ty->string : "message";
    if (strcmp(type, "message") == 0) {
        json_value *rv = cwick_json_get(item, "role");
        if (!rv || rv->kind != JSON_STRING) {
            snprintf(err, cap, "message item needs a role");
            return false;
        }
        /* The Responses API renamed "system" to "developer"; both are the
         * same instruction turn to the checkpoint. */
        const char *role = strcmp(rv->string, "developer") == 0 ? "system" : rv->string;
        if (strcmp(role, "system") && strcmp(role, "user") &&
            strcmp(role, "assistant") && strcmp(role, "tool")) {
            snprintf(err, cap, "unsupported message role `%s`", role);
            return false;
        }
        resp_turn *last = resp_turn_last(t);
        resp_turn *turn = (!strcmp(role, "assistant") && last &&
                           !strcmp(last->role, "assistant")) ? last
                                                            : resp_turn_new(t, role);
        if (!resp_text_of(cwick_json_get(item, "content"), &turn->text, err, cap))
            return false;
        return true;
    }
    if (strcmp(type, "function_call") == 0) {
        json_value *cv = cwick_json_get(item, "call_id");
        json_value *nv = cwick_json_get(item, "name");
        json_value *av = cwick_json_get(item, "arguments");
        if (!cv || cv->kind != JSON_STRING || !nv || nv->kind != JSON_STRING) {
            snprintf(err, cap, "function_call needs call_id and name");
            return false;
        }
        resp_turn *last = resp_turn_last(t);
        resp_turn *turn = (last && !strcmp(last->role, "assistant"))
                              ? last : resp_turn_new(t, "assistant");
        char *jc = cwick_json_stringify(cv->string);
        char *jn = cwick_json_stringify(nv->string);
        char *ja = cwick_json_stringify((av && av->kind == JSON_STRING)
                                            ? av->string : "{}");
        if (turn->ncalls) sbuf_puts(&turn->calls, ",");
        sbuf_putf(&turn->calls, "{\"id\":%s,\"type\":\"function\","
                                "\"function\":{\"name\":%s,\"arguments\":%s}}",
                  jc, jn, ja);
        turn->ncalls++;
        free(jc);
        free(jn);
        free(ja);
        return true;
    }
    if (strcmp(type, "function_call_output") == 0) {
        json_value *cv = cwick_json_get(item, "call_id");
        if (!cv || cv->kind != JSON_STRING) {
            snprintf(err, cap, "function_call_output needs a call_id");
            return false;
        }
        resp_turn *turn = resp_turn_new(t, "tool");
        snprintf(turn->call_id, sizeof(turn->call_id), "%s", cv->string);
        return resp_text_of(cwick_json_get(item, "output"), &turn->text, err, cap);
    }
    if (strcmp(type, "reasoning") == 0) {
        /* The checkpoint's own hidden chain is not replayable input: it is
         * not part of any prompt template. Dropping it is what the chat path
         * does with a stale reasoning_content too. */
        return true;
    }
    snprintf(err, cap, "unsupported input item type `%s`", type);
    return false;
}

static char *resp_messages_json(const resp_turns *t) {
    srv_buf b = {0};
    sbuf_puts(&b, "[");
    for (size_t i = 0; i < t->n; i++) {
        const resp_turn *r = &t->v[i];
        if (i) sbuf_puts(&b, ",");
        char *role = cwick_json_stringify(r->role);
        char *text = cwick_json_stringify(r->text.p ? r->text.p : "");
        if (!strcmp(r->role, "tool")) {
            char *cid = cwick_json_stringify(r->call_id);
            sbuf_putf(&b, "{\"role\":\"tool\",\"tool_call_id\":%s,"
                          "\"content\":%s}", cid, text);
            free(cid);
        } else if (r->ncalls) {
            sbuf_putf(&b, "{\"role\":%s,\"content\":%s,\"tool_calls\":[%s]}",
                      role, text, r->calls.p);
        } else {
            sbuf_putf(&b, "{\"role\":%s,\"content\":%s}", role, text);
        }
        free(role);
        free(text);
    }
    sbuf_puts(&b, "]");
    return b.p;
}

/* Responses declares a tool flat ({type, name, description, parameters});
 * the engine's chat renderer reads the nested chat form. `strict` is accepted
 * and ignored — schema conformance is never constrained during decoding. */
static char *resp_tools_json(json_value *tools, char *err, size_t cap) {
    if (!tools || tools->kind == JSON_NULL) return NULL;
    if (tools->kind != JSON_ARRAY) {
        snprintf(err, cap, "`tools` must be an array");
        return NULL;
    }
    if (tools->count == 0) return NULL;
    srv_buf b = {0};
    sbuf_puts(&b, "[");
    for (size_t i = 0; i < tools->count; i++) {
        json_value *t = tools->items[i];
        json_value *ty = cwick_json_get(t, "type");
        if (ty && ty->kind == JSON_STRING && strcmp(ty->string, "function") != 0) {
            snprintf(err, cap, "only `function` tools are supported, got `%s`",
                     ty->string);
            free(b.p);
            return NULL;
        }
        json_value *nv = cwick_json_get(t, "name");
        if (!nv || nv->kind != JSON_STRING) {
            snprintf(err, cap, "a tool needs a name");
            free(b.p);
            return NULL;
        }
        json_value *dv = cwick_json_get(t, "description");
        json_value *pv = cwick_json_get(t, "parameters");
        char *jn = cwick_json_stringify(nv->string);
        char *jd = (dv && dv->kind == JSON_STRING) ? cwick_json_stringify(dv->string)
                                                   : cwick_strdup("null");
        char *jp = (pv && pv->kind == JSON_OBJECT)
                       ? cwick_json_write(pv)
                       : cwick_strdup("{\"type\":\"object\",\"properties\":{}}");
        if (i) sbuf_puts(&b, ",");
        sbuf_putf(&b, "{\"type\":\"function\",\"function\":{\"name\":%s,"
                      "\"description\":%s,\"parameters\":%s}}", jn, jd, jp);
        free(jn);
        free(jd);
        free(jp);
    }
    sbuf_puts(&b, "]");
    return b.p;
}

/* tool_choice: auto/none pass through; a named function becomes the chat
 * form the tool layer already validates; "required" needs constrained
 * decoding and is refused rather than silently downgraded. */
static bool resp_tool_choice(json_value *tc, char *out, size_t cap,
                             char *err, size_t errcap) {
    out[0] = '\0';
    if (!tc || tc->kind == JSON_NULL) return true;
    if (tc->kind == JSON_STRING) {
        if (!strcmp(tc->string, "auto") || !strcmp(tc->string, "none")) {
            snprintf(out, cap, "\"%s\"", tc->string);
            return true;
        }
        snprintf(err, errcap, "tool_choice `%s` is not supported "
                 "(use auto, none, or a named function)", tc->string);
        return false;
    }
    if (tc->kind == JSON_OBJECT) {
        json_value *ty = cwick_json_get(tc, "type");
        json_value *nm = cwick_json_get(tc, "name");
        if (ty && ty->kind == JSON_STRING &&
            strcmp(ty->string, "function") == 0 && nm && nm->kind == JSON_STRING) {
            char *jn = cwick_json_stringify(nm->string);
            snprintf(out, cap, "{\"type\":\"function\",\"function\":{\"name\":%s}}", jn);
            free(jn);
            return true;
        }
        snprintf(err, errcap, "only {\"type\":\"function\",\"name\":...} "
                 "tool_choice is supported");
        return false;
    }
    snprintf(err, errcap, "tool_choice must be a string or an object");
    return false;
}

/* Translate a Responses request into the chat request the rest of the server
 * already validates, then validate that. Returns 0/400 like validate_body. */
static int responses_parse(json_value *root, gen_request *gr,
                           const char *model_id, const cwick_model_entry *cur,
                           char *errbuf, size_t errcap) {
    static const char *allowed[] = {
        "model", "input", "instructions", "tools", "tool_choice",
        "max_output_tokens", "max_tokens", "temperature", "top_p", "stream",
        "reasoning", "store", "previous_response_id", "conversation",
        "metadata", "user", "include", "parallel_tool_calls", "text",
        "truncation", "background", "prompt_cache_key", "safety_identifier",
        "service_tier", "top_logprobs", "max_tool_calls", NULL,
    };
    if (root->kind != JSON_OBJECT) {
        snprintf(errbuf, errcap, "request body must be a JSON object");
        return 400;
    }
    errbuf[0] = '\0';
    for (size_t i = 0; i < root->count; i++) {
        const char *k = root->members[i].key;
        bool ok = false;
        for (const char **a = allowed; *a; a++) if (strcmp(k, *a) == 0) { ok = true; break; }
        if (!ok) {
            snprintf(errbuf, errcap, "unsupported field `%s`", k);
            return 400;
        }
    }
    /* Durable state is the one thing this server does not have. Accepting a
     * previous_response_id we cannot load would silently truncate the prompt,
     * so it is refused by name — replay the full input instead. */
    const char *stateful[] = { "previous_response_id", "conversation" };
    for (size_t i = 0; i < sizeof(stateful) / sizeof(stateful[0]); i++) {
        json_value *v = cwick_json_get(root, stateful[i]);
        if (v && v->kind != JSON_NULL) {
            snprintf(errbuf, errcap, "%s is not supported; replay the full "
                     "input items instead", stateful[i]);
            return 400;
        }
    }
    json_value *store = cwick_json_get(root, "store");
    if (store && cwick_json_bool(store, false)) {
        snprintf(errbuf, errcap, "store=true is not implemented "
                 "(responses are not persisted)");
        return 400;
    }
    json_value *bg = cwick_json_get(root, "background");
    if (bg && cwick_json_bool(bg, false)) {
        snprintf(errbuf, errcap, "background responses are not implemented");
        return 400;
    }
    json_value *tr = cwick_json_get(root, "truncation");
    if (tr && tr->kind == JSON_STRING && strcmp(tr->string, "disabled") != 0) {
        snprintf(errbuf, errcap, "truncation `%s` is not implemented "
                 "(a long prompt is refused, not truncated)", tr->string);
        return 400;
    }
    json_value *tl = cwick_json_get(root, "top_logprobs");
    if (tl && tl->kind != JSON_NULL && cwick_json_int(tl, 0) != 0) {
        snprintf(errbuf, errcap, "top_logprobs is not implemented");
        return 400;
    }
    json_value *mtc = cwick_json_get(root, "max_tool_calls");
    if (mtc && mtc->kind != JSON_NULL) {
        snprintf(errbuf, errcap, "max_tool_calls is not implemented");
        return 400;
    }
    json_value *txt = cwick_json_get(root, "text");
    if (txt && txt->kind == JSON_OBJECT) {
        json_value *fmt = cwick_json_get(txt, "format");
        json_value *fty = fmt ? cwick_json_get(fmt, "type") : NULL;
        if (fty && fty->kind == JSON_STRING && strcmp(fty->string, "text") != 0) {
            snprintf(errbuf, errcap, "text.format `%s` is not implemented",
                     fty->string);
            return 400;
        }
    }
    json_value *insv = cwick_json_get(root, "instructions");
    if (insv && insv->kind != JSON_NULL && insv->kind != JSON_STRING) {
        snprintf(errbuf, errcap, "instructions must be a string");
        return 400;
    }

    json_value *input = cwick_json_get(root, "input");
    if (!input) {
        snprintf(errbuf, errcap, "missing input");
        return 400;
    }
    resp_turns turns;
    memset(&turns, 0, sizeof(turns));
    bool ok = true;
    if (input->kind == JSON_STRING) {
        resp_turn *t = resp_turn_new(&turns, "user");
        sbuf_puts(&t->text, input->string);
    } else if (input->kind == JSON_ARRAY) {
        for (size_t i = 0; i < input->count && ok; i++)
            ok = resp_add_item(&turns, input->items[i], errbuf, errcap);
    } else {
        snprintf(errbuf, errcap, "`input` must be a string or an array of items");
        ok = false;
    }
    if (ok && !turns.n) {
        snprintf(errbuf, errcap, "`input` is empty");
        ok = false;
    }
    /* instructions is the request's system turn, and it precedes whatever the
     * caller put in `input` regardless of field order. */
    json_value *ins = cwick_json_get(root, "instructions");
    if (ok && ins && ins->kind == JSON_STRING && ins->string[0]) {
        resp_turns with_sys;
        memset(&with_sys, 0, sizeof(with_sys));
        resp_turn *sys = resp_turn_new(&with_sys, "system");
        sbuf_puts(&sys->text, ins->string);
        if (with_sys.n) {
            with_sys.v = cwick_xrealloc(with_sys.v,
                                        (with_sys.n + turns.n) * sizeof(*with_sys.v));
            memcpy(with_sys.v + with_sys.n, turns.v, turns.n * sizeof(*turns.v));
            with_sys.n += turns.n;
            with_sys.cap = with_sys.n;
            free(turns.v);
            turns = with_sys;
        }
    }
    if (!ok) {
        resp_turns_free(&turns);
        return 400;
    }

    char *messages = resp_messages_json(&turns);
    resp_turns_free(&turns);

    char *tools = NULL;
    json_value *tj = cwick_json_get(root, "tools");
    if (tj && tj->kind != JSON_NULL) {
        tools = resp_tools_json(tj, errbuf, errcap);
        if (!tools && errbuf[0]) { free(messages); return 400; }
    }
    char choice[256];
    if (!resp_tool_choice(cwick_json_get(root, "tool_choice"), choice,
                          sizeof(choice), errbuf, errcap)) {
        free(messages);
        free(tools);
        return 400;
    }

    bool summary = false;
    const char *effort = NULL;
    json_value *rz = cwick_json_get(root, "reasoning");
    if (rz && rz->kind != JSON_NULL && rz->kind != JSON_OBJECT) {
        snprintf(errbuf, errcap, "reasoning must be an object");
        free(messages);
        free(tools);
        return 400;
    }
    if (rz && rz->kind == JSON_OBJECT) {
        json_value *ev = cwick_json_get(rz, "effort");
        if (ev && ev->kind == JSON_STRING) {
            const char *lvl = NULL, *eff = NULL;
            if (!cwick_thinking_effort_resolve(ev->string, &lvl, &eff)) {
                snprintf(errbuf, errcap, "reasoning.effort must be one of "
                         "minimal/low/medium/high/xhigh/none");
                free(messages);
                free(tools);
                return 400;
            }
            effort = ev->string;
        }
        json_value *sm = cwick_json_get(rz, "summary");
        if (sm && sm->kind == JSON_STRING && strcmp(sm->string, "none") != 0)
            summary = true;
    }

    /* Build the chat-shaped body and hand it to the validator the chat
     * endpoint uses, so every rule (model identity, sampling ranges, tool
     * declarations, thinking levels) has exactly one implementation. */
    srv_buf synth = {0};
    sbuf_puts(&synth, "{");
    json_value *mv = cwick_json_get(root, "model");
    if (mv && mv->kind != JSON_NULL && mv->kind != JSON_STRING) {
        snprintf(errbuf, errcap, "model must be a string");
        free(synth.p);
        return 400;
    }
    if (mv && mv->kind == JSON_STRING) {
        char *jm = cwick_json_stringify(mv->string);
        sbuf_putf(&synth, "\"model\":%s,", jm);
        free(jm);
    }
    sbuf_putf(&synth, "\"messages\":%s", messages);
    free(messages);
    if (tools) {
        sbuf_putf(&synth, ",\"tools\":%s", tools);
        free(tools);
    }
    if (choice[0]) sbuf_putf(&synth, ",\"tool_choice\":%s", choice);
    json_value *mo = cwick_json_get(root, "max_output_tokens");
    if (!mo || mo->kind == JSON_NULL) mo = cwick_json_get(root, "max_tokens");
    if (mo && mo->kind != JSON_NULL) {
        if (mo->kind != JSON_NUMBER) {
            snprintf(errbuf, errcap, "max_output_tokens must be a number");
            free(synth.p);
            return 400;
        }
        sbuf_putf(&synth, ",\"max_tokens\":%lld", (long long)cwick_json_int(mo, 0));
    }
    json_value *tv = cwick_json_get(root, "temperature");
    if (tv && tv->kind != JSON_NULL) {
        if (tv->kind != JSON_NUMBER) {
            snprintf(errbuf, errcap, "temperature must be a number");
            free(synth.p);
            return 400;
        }
        sbuf_putf(&synth, ",\"temperature\":%.6f", cwick_json_num(tv, 1.0));
    }
    json_value *pv = cwick_json_get(root, "top_p");
    if (pv && pv->kind != JSON_NULL) {
        if (pv->kind != JSON_NUMBER) {
            snprintf(errbuf, errcap, "top_p must be a number");
            free(synth.p);
            return 400;
        }
        sbuf_putf(&synth, ",\"top_p\":%.6f", cwick_json_num(pv, 1.0));
    }
    json_value *sv = cwick_json_get(root, "stream");
    if (sv && sv->kind != JSON_NULL && sv->kind != JSON_BOOL) {
        snprintf(errbuf, errcap, "stream must be a boolean");
        free(synth.p);
        return 400;
    }
    sbuf_putf(&synth, ",\"stream\":%s", cwick_json_bool(sv, false) ? "true" : "false");
    if (effort) {
        char *je = cwick_json_stringify(effort);
        sbuf_putf(&synth, ",\"reasoning_effort\":%s", je);
        free(je);
    }
    sbuf_puts(&synth, "}");

    /* The half of the response object that echoes the request. Three of these
     * (`tools`, `tool_choice`, `parallel_tool_calls`) are not optional in the
     * schema, and the rest is what a client compares against what it sent —
     * a response that drops them looks like a server that ignored the
     * request. Built once here so the buffered body and every streaming
     * envelope carry the same copy. */
    {
        srv_buf tail = {0};
        json_value *pc = cwick_json_get(root, "parallel_tool_calls");
        bool parallel = !pc || pc->kind == JSON_NULL || cwick_json_bool(pc, true);
        sbuf_putf(&tail, ",\"parallel_tool_calls\":%s",
                  parallel ? "true" : "false");
        json_value *tcv = cwick_json_get(root, "tool_choice");
        char *jtc = (tcv && tcv->kind != JSON_NULL) ? cwick_json_write(tcv)
                                                    : cwick_strdup("\"auto\"");
        sbuf_putf(&tail, ",\"tool_choice\":%s", jtc);
        free(jtc);
        json_value *tje = cwick_json_get(root, "tools");
        char *jtl = (tje && tje->kind == JSON_ARRAY) ? cwick_json_write(tje)
                                                     : cwick_strdup("[]");
        sbuf_putf(&tail, ",\"tools\":%s", jtl);
        free(jtl);
        json_value *iev = cwick_json_get(root, "instructions");
        if (iev && iev->kind == JSON_STRING) {
            char *ji = cwick_json_stringify(iev->string);
            sbuf_putf(&tail, ",\"instructions\":%s", ji);
            free(ji);
        } else {
            sbuf_puts(&tail, ",\"instructions\":null");
        }
        json_value *mo2 = cwick_json_get(root, "max_output_tokens");
        if (!mo2 || mo2->kind == JSON_NULL) mo2 = cwick_json_get(root, "max_tokens");
        if (mo2 && mo2->kind == JSON_NUMBER)
            sbuf_putf(&tail, ",\"max_output_tokens\":%lld",
                      (long long)cwick_json_int(mo2, 0));
        else
            sbuf_puts(&tail, ",\"max_output_tokens\":null");
        static const char *nums[] = { "temperature", "top_p" };
        for (size_t i = 0; i < sizeof(nums) / sizeof(nums[0]); i++) {
            json_value *v = cwick_json_get(root, nums[i]);
            if (v && v->kind == JSON_NUMBER)
                sbuf_putf(&tail, ",\"%s\":%.6f", nums[i],
                          cwick_json_num(v, 0.0));
            else
                sbuf_putf(&tail, ",\"%s\":null", nums[i]);
        }
        sbuf_puts(&tail, ",\"truncation\":\"disabled\"");
        json_value *txv = cwick_json_get(root, "text");
        char *jtx = (txv && txv->kind == JSON_OBJECT)
                        ? cwick_json_write(txv)
                        : cwick_strdup("{\"format\":{\"type\":\"text\"}}");
        sbuf_putf(&tail, ",\"text\":%s", jtx);
        free(jtx);
        json_value *rzv = cwick_json_get(root, "reasoning");
        if (rzv && rzv->kind == JSON_OBJECT) {
            char *jr = cwick_json_write(rzv);
            sbuf_putf(&tail, ",\"reasoning\":%s", jr);
            free(jr);
        } else {
            sbuf_puts(&tail, ",\"reasoning\":null");
        }
        json_value *mdv = cwick_json_get(root, "metadata");
        if (mdv && mdv->kind == JSON_OBJECT) {
            char *jm = cwick_json_write(mdv);
            sbuf_putf(&tail, ",\"metadata\":%s", jm);
            free(jm);
        } else {
            sbuf_puts(&tail, ",\"metadata\":null");
        }
        sbuf_puts(&tail, ",\"store\":false,\"background\":false,"
                         "\"previous_response_id\":null");
        gr->resp_tail = tail.p;
    }

    gr->api = SRV_API_RESPONSES;
    gr->is_chat = true;
    gr->resp_summary = summary;
    cwick_error jerr;
    cwick_error_clear(&jerr);
    json_value *chat_root = cwick_json_parse(synth.p, synth.len, &jerr);
    free(synth.p);
    if (!chat_root) {
        snprintf(errbuf, errcap, "invalid request: %s",
                 cwick_error_message(&jerr));
        return 400;
    }
    int rc = validate_body(chat_root, gr, errbuf, errcap, model_id, cur);
    cwick_json_free(chat_root);
    return rc;
}

/* ------------------------------------------------------------------ */
/* Anthropic Messages API: request -> the same chat conversation        */
/* ------------------------------------------------------------------ */
/* The wire format is Claude's: `system` is a top-level field, content is an
 * array of typed blocks, a tool is declared flat with a JSON Schema named
 * `input_schema`, and a call/result pair is `tool_use`/`tool_result`. None of
 * that reaches the engine: the request is folded into the same chat body the
 * other two dialects build, so the prompt templates, the queue, the prefix
 * cache and the tool layer keep one implementation. */

/* Anthropic content is a string or an array of blocks. Only text carries a
 * prompt; image/document/audio blocks are refused rather than dropped,
 * because a silently missing attachment is a wrong prompt, not a smaller
 * one. */
static bool anth_blocks_text(json_value *v, srv_buf *out, char *err,
                             size_t cap, const char *what) {
    if (!v || v->kind == JSON_NULL) return true;
    if (v->kind == JSON_STRING) { sbuf_puts(out, v->string); return true; }
    if (v->kind != JSON_ARRAY) {
        snprintf(err, cap, "%s must be a string or an array of blocks", what);
        return false;
    }
    for (size_t i = 0; i < v->count; i++) {
        json_value *b = v->items[i];
        json_value *ty = cwick_json_get(b, "type");
        const char *t = (ty && ty->kind == JSON_STRING) ? ty->string : "text";
        if (!strcmp(t, "image") || !strcmp(t, "document") ||
            !strcmp(t, "audio") || !strcmp(t, "video")) {
            snprintf(err, cap, "%s blocks are not supported (text-only server)", t);
            return false;
        }
        if (!strcmp(t, "text")) {
            json_value *tx = cwick_json_get(b, "text");
            if (tx && tx->kind == JSON_STRING) sbuf_puts(out, tx->string);
        }
        /* thinking / redacted_thinking / tool_use / tool_result are handled
         * by the caller; this only collects the prose. */
    }
    return true;
}

/* Append one chat message. `calls` is a ready-made `tool_calls` array body
 * (assistant), `call_id` turns the message into a tool result. */
static void anth_emit(srv_buf *b, size_t *n, const char *role, const char *text,
                      const char *calls, const char *call_id) {
    char *jr = cwick_json_stringify(role);
    char *jt = cwick_json_stringify(text ? text : "");
    if (*n) sbuf_puts(b, ",");
    if (call_id) {
        char *jc = cwick_json_stringify(call_id);
        sbuf_putf(b, "{\"role\":\"tool\",\"tool_call_id\":%s,\"content\":%s}",
                  jc, jt);
        free(jc);
    } else if (calls && *calls) {
        sbuf_putf(b, "{\"role\":%s,\"content\":%s,\"tool_calls\":[%s]}",
                  jr, jt, calls);
    } else {
        sbuf_putf(b, "{\"role\":%s,\"content\":%s}", jr, jt);
    }
    free(jr);
    free(jt);
    (*n)++;
}

/* One Anthropic message -> one or more chat messages. A user turn may carry
 * several `tool_result` blocks, each of which is its own `tool` role in the
 * chat form, so text and results are flushed in the order they appear. */
static bool anth_add_message(json_value *m, srv_buf *msgs, size_t *n,
                             char *err, size_t cap) {
    json_value *rv = cwick_json_get(m, "role");
    json_value *cv = cwick_json_get(m, "content");
    if (!rv || rv->kind != JSON_STRING) {
        snprintf(err, cap, "message role must be a string");
        return false;
    }
    const bool is_user = strcmp(rv->string, "user") == 0;
    const bool is_asst = strcmp(rv->string, "assistant") == 0;
    if (!is_user && !is_asst) {
        snprintf(err, cap, "unsupported message role `%s`", rv->string);
        return false;
    }
    if (!cv || cv->kind == JSON_NULL) {
        snprintf(err, cap, "a message needs content");
        return false;
    }
    if (cv->kind == JSON_STRING) {
        anth_emit(msgs, n, rv->string, cv->string, NULL, NULL);
        return true;
    }
    if (cv->kind != JSON_ARRAY) {
        snprintf(err, cap, "message content must be a string or an array of blocks");
        return false;
    }

    srv_buf text = {0};
    srv_buf calls = {0};
    size_t ncalls = 0;
    bool ok = true;
    for (size_t i = 0; i < cv->count && ok; i++) {
        json_value *b = cv->items[i];
        json_value *ty = cwick_json_get(b, "type");
        const char *t = (ty && ty->kind == JSON_STRING) ? ty->string : "text";
        if (!strcmp(t, "text")) {
            json_value *tx = cwick_json_get(b, "text");
            if (tx && tx->kind == JSON_STRING) sbuf_puts(&text, tx->string);
            continue;
        }
        /* A hidden chain is not replayable input for any template; the chat
         * path drops a stale reasoning_content for the same reason. */
        if (!strcmp(t, "thinking") || !strcmp(t, "redacted_thinking")) continue;
        if (!strcmp(t, "tool_use") && is_asst) {
            json_value *iv = cwick_json_get(b, "id");
            json_value *nv = cwick_json_get(b, "name");
            json_value *av = cwick_json_get(b, "input");
            if (!iv || iv->kind != JSON_STRING || !nv || nv->kind != JSON_STRING) {
                snprintf(err, cap, "tool_use needs id and name");
                ok = false;
                break;
            }
            char *raw = (av && av->kind != JSON_NULL) ? cwick_json_write(av)
                                                      : cwick_strdup("{}");
            char *ji = cwick_json_stringify(iv->string);
            char *jn = cwick_json_stringify(nv->string);
            char *ja = cwick_json_stringify(raw);
            if (ncalls) sbuf_puts(&calls, ",");
            sbuf_putf(&calls, "{\"id\":%s,\"type\":\"function\","
                              "\"function\":{\"name\":%s,\"arguments\":%s}}",
                      ji, jn, ja);
            ncalls++;
            free(raw);
            free(ji);
            free(jn);
            free(ja);
            continue;
        }
        if (!strcmp(t, "tool_result") && is_user) {
            json_value *idv = cwick_json_get(b, "tool_use_id");
            if (!idv || idv->kind != JSON_STRING) {
                snprintf(err, cap, "tool_result needs a tool_use_id");
                ok = false;
                break;
            }
            srv_buf out = {0};
            if (!anth_blocks_text(cwick_json_get(b, "content"), &out, err, cap,
                                  "tool_result content")) {
                free(out.p);
                ok = false;
                break;
            }
            /* Prose and the result keep their order: anything written before
             * this block is its own user turn. */
            if (text.len) {
                anth_emit(msgs, n, "user", text.p, NULL, NULL);
                text.len = 0;
                if (text.p) text.p[0] = '\0';
            }
            anth_emit(msgs, n, "tool", out.p ? out.p : "", NULL, idv->string);
            free(out.p);
            continue;
        }
        if (!strcmp(t, "image") || !strcmp(t, "document") ||
            !strcmp(t, "audio") || !strcmp(t, "video")) {
            snprintf(err, cap, "%s blocks are not supported (text-only server)", t);
            ok = false;
            break;
        }
        /* tool_result in an assistant turn, tool_use in a user turn: the
         * pairing is wrong, and guessing would send the wrong prompt. */
        snprintf(err, cap, "`%s` blocks are not valid in %s turn", t,
                 is_asst ? "an assistant" : "a user");
        ok = false;
    }
    if (ok) {
        if (is_asst) {
            /* One assistant turn: its prose plus every call it made. */
            anth_emit(msgs, n, "assistant", text.p, calls.p, NULL);
        } else if (text.len) {
            anth_emit(msgs, n, "user", text.p, NULL, NULL);
        }
    }
    free(text.p);
    free(calls.p);
    return ok;
}

/* `input_schema` -> chat `parameters`; a tool here has no `type` (or
 * `"custom"`). Server-side tools (`web_search`, …) do not exist here and are
 * refused by name. */
static bool anth_tools(json_value *tools, srv_buf *out, char *err, size_t cap) {
    if (!tools || tools->kind == JSON_NULL) return true;
    if (tools->kind != JSON_ARRAY) {
        snprintf(err, cap, "`tools` must be an array");
        return false;
    }
    sbuf_puts(out, ",\"tools\":[");
    for (size_t i = 0; i < tools->count; i++) {
        json_value *t = tools->items[i];
        json_value *ty = cwick_json_get(t, "type");
        if (ty && ty->kind == JSON_STRING && strcmp(ty->string, "custom") != 0) {
            snprintf(err, cap, "only client tools are supported, got `%s`",
                     ty->string);
            return false;
        }
        json_value *nv = cwick_json_get(t, "name");
        if (!nv || nv->kind != JSON_STRING) {
            snprintf(err, cap, "a tool needs a name");
            return false;
        }
        json_value *dv = cwick_json_get(t, "description");
        json_value *sv = cwick_json_get(t, "input_schema");
        char *jn = cwick_json_stringify(nv->string);
        char *jd = (dv && dv->kind == JSON_STRING) ? cwick_json_stringify(dv->string)
                                                   : cwick_strdup("null");
        char *jp = (sv && sv->kind == JSON_OBJECT)
                       ? cwick_json_write(sv)
                       : cwick_strdup("{\"type\":\"object\",\"properties\":{}}");
        if (i) sbuf_puts(out, ",");
        sbuf_putf(out, "{\"type\":\"function\",\"function\":{\"name\":%s,"
                       "\"description\":%s,\"parameters\":%s}}", jn, jd, jp);
        free(jn);
        free(jd);
        free(jp);
    }
    sbuf_puts(out, "]");
    return true;
}

/* tool_choice: auto / none / a named tool. `any` ("must call something") is
 * the same requirement as OpenAI's `required` and needs constrained decoding,
 * so it is refused instead of being downgraded to `auto`. */
static bool anth_tool_choice(json_value *tc, srv_buf *out, char *err, size_t cap) {
    if (!tc || tc->kind == JSON_NULL) return true;
    const char *type = NULL;
    json_value *name = NULL;
    if (tc->kind == JSON_STRING) {
        type = tc->string;
    } else if (tc->kind == JSON_OBJECT) {
        json_value *ty = cwick_json_get(tc, "type");
        type = (ty && ty->kind == JSON_STRING) ? ty->string : NULL;
        name = cwick_json_get(tc, "name");
    }
    if (!type) {
        snprintf(err, cap, "tool_choice must be a string or {\"type\":…}");
        return false;
    }
    if (!strcmp(type, "auto") || !strcmp(type, "none")) {
        sbuf_putf(out, ",\"tool_choice\":\"%s\"", type);
        return true;
    }
    if (!strcmp(type, "any")) {
        snprintf(err, cap, "tool_choice `any` is not implemented "
                 "(a mandatory call needs constrained decoding; use `auto`)");
        return false;
    }
    if (!strcmp(type, "tool")) {
        if (!name || name->kind != JSON_STRING) {
            snprintf(err, cap, "tool_choice `tool` needs a name");
            return false;
        }
        char *jn = cwick_json_stringify(name->string);
        sbuf_putf(out, ",\"tool_choice\":{\"type\":\"function\","
                       "\"function\":{\"name\":%s}}", jn);
        free(jn);
        return true;
    }
    snprintf(err, cap, "unknown tool_choice `%s`", type);
    return false;
}

/* `thinking.budget_tokens` -> the closest 思考强度 level. The ladder is
 * cwick's own (512 / 2048 / 8192 / 16384 / 32768); a budget is a ceiling, so
 * rounding *down* would hand back less thinking than was paid for. */
static const char *anth_think_level(json_value *thinking, char *err, size_t cap,
                                    bool *ok) {
    *ok = true;
    if (!thinking || thinking->kind == JSON_NULL) return NULL;
    if (thinking->kind != JSON_OBJECT) {
        snprintf(err, cap, "`thinking` must be an object");
        *ok = false;
        return NULL;
    }
    json_value *ty = cwick_json_get(thinking, "type");
    const char *t = (ty && ty->kind == JSON_STRING) ? ty->string : "enabled";
    if (!strcmp(t, "disabled")) return "关";
    /* "adaptive" means the model decides; 高 is the level that asks it to
     * think without pinning the block to a small budget. */
    if (!strcmp(t, "adaptive")) return "高";
    if (strcmp(t, "enabled")) {
        snprintf(err, cap, "thinking.type must be enabled, disabled or adaptive");
        *ok = false;
        return NULL;
    }
    json_value *bv = cwick_json_get(thinking, "budget_tokens");
    if (!bv || bv->kind == JSON_NULL) return "中";
    if (bv->kind != JSON_NUMBER) {
        snprintf(err, cap, "thinking.budget_tokens must be a number");
        *ok = false;
        return NULL;
    }
    int64_t budget = cwick_json_int(bv, 0);
    if (budget <= 512) return "低";
    if (budget <= 2048) return "中";
    if (budget <= 8192) return "高";
    if (budget <= 16384) return "超高";
    return "极致";
}

static int anthropic_parse(json_value *root, gen_request *gr,
                           const char *model_id, const cwick_model_entry *cur,
                           char *errbuf, size_t errcap) {
    static const char *allowed[] = {
        "model", "messages", "system", "max_tokens", "temperature", "top_p",
        "top_k", "stop_sequences", "stream", "tools", "tool_choice",
        "thinking", "metadata", "service_tier", NULL,
    };
    if (root->kind != JSON_OBJECT) {
        snprintf(errbuf, errcap, "request body must be a JSON object");
        return 400;
    }
    errbuf[0] = '\0';
    for (size_t i = 0; i < root->count; i++) {
        const char *k = root->members[i].key;
        bool ok = false;
        for (const char **a = allowed; *a; a++) if (strcmp(k, *a) == 0) { ok = true; break; }
        if (!ok) {
            snprintf(errbuf, errcap, "unsupported field `%s`", k);
            return 400;
        }
    }
    /* Anthropic requires max_tokens; the engine would otherwise fall back to
     * the checkpoint's own default, which is not what the client asked for. */
    json_value *mtv = cwick_json_get(root, "max_tokens");
    if (!mtv || mtv->kind != JSON_NUMBER) {
        snprintf(errbuf, errcap, "max_tokens is required (a number)");
        return 400;
    }
    json_value *ssv = cwick_json_get(root, "stop_sequences");
    if (ssv && ssv->kind != JSON_NULL) {
        if (ssv->kind != JSON_ARRAY) {
            snprintf(errbuf, errcap, "stop_sequences must be an array");
            return 400;
        }
        if (ssv->count > 0) {
            snprintf(errbuf, errcap, "non-default stop_sequences are not "
                     "implemented (the engine stops on its own delimiters)");
            return 400;
        }
    }

    json_value *msgs = cwick_json_get(root, "messages");
    if (!msgs || msgs->kind != JSON_ARRAY || msgs->count == 0) {
        snprintf(errbuf, errcap, "`messages` must be a non-empty array");
        return 400;
    }

    srv_buf body = {0};
    size_t nmsg = 0;
    sbuf_puts(&body, "[");
    /* `system` is a top-level turn, and it precedes `messages` regardless of
     * field order. */
    json_value *sysv = cwick_json_get(root, "system");
    if (sysv && sysv->kind != JSON_NULL) {
        srv_buf sys = {0};
        if (!anth_blocks_text(sysv, &sys, errbuf, errcap, "system")) {
            free(sys.p);
            free(body.p);
            return 400;
        }
        if (sys.len) anth_emit(&body, &nmsg, "system", sys.p, NULL, NULL);
        free(sys.p);
    }
    for (size_t i = 0; i < msgs->count; i++) {
        if (!anth_add_message(msgs->items[i], &body, &nmsg, errbuf, errcap)) {
            free(body.p);
            return 400;
        }
    }
    sbuf_puts(&body, "]");
    if (nmsg == 0) {
        snprintf(errbuf, errcap, "`messages` is empty");
        free(body.p);
        return 400;
    }

    /* Rebuild the request as the chat body the validator already owns, so
     * every rule (model identity, sampling range, tool declarations, thinking
     * levels) keeps exactly one implementation. */
    srv_buf synth = {0};
    sbuf_puts(&synth, "{");
    json_value *mv = cwick_json_get(root, "model");
    if (mv && mv->kind != JSON_NULL && mv->kind != JSON_STRING) {
        snprintf(errbuf, errcap, "model must be a string");
        free(body.p);
        free(synth.p);
        return 400;
    }
    if (mv && mv->kind == JSON_STRING) {
        char *jm = cwick_json_stringify(mv->string);
        sbuf_putf(&synth, "\"model\":%s,", jm);
        free(jm);
    }
    sbuf_putf(&synth, "\"messages\":%s", body.p);
    free(body.p);
    sbuf_putf(&synth, ",\"max_tokens\":%lld",
              (long long)cwick_json_int(mtv, 0));
    if (!anth_tools(cwick_json_get(root, "tools"), &synth, errbuf, errcap)) {
        free(synth.p);
        return 400;
    }
    if (!anth_tool_choice(cwick_json_get(root, "tool_choice"), &synth,
                          errbuf, errcap)) {
        free(synth.p);
        return 400;
    }
    static const char *scalars[] = { "temperature", "top_p", "top_k" };
    for (size_t i = 0; i < sizeof(scalars) / sizeof(scalars[0]); i++) {
        json_value *v = cwick_json_get(root, scalars[i]);
        if (!v || v->kind == JSON_NULL) continue;
        if (v->kind != JSON_NUMBER) {
            snprintf(errbuf, errcap, "%s must be a number", scalars[i]);
            free(synth.p);
            return 400;
        }
        sbuf_putf(&synth, ",\"%s\":%.6f", scalars[i], cwick_json_num(v, 0.0));
    }
    json_value *sv = cwick_json_get(root, "stream");
    if (sv && sv->kind != JSON_NULL && sv->kind != JSON_BOOL) {
        snprintf(errbuf, errcap, "`stream` must be a boolean");
        free(synth.p);
        return 400;
    }
    sbuf_putf(&synth, ",\"stream\":%s", cwick_json_bool(sv, false) ? "true" : "false");
    bool tkok = true;
    const char *level = anth_think_level(cwick_json_get(root, "thinking"),
                                         errbuf, errcap, &tkok);
    if (!tkok) {
        free(synth.p);
        return 400;
    }
    if (level) {
        char *jl = cwick_json_stringify(level);
        sbuf_putf(&synth, ",\"reasoning_effort\":%s", jl);
        free(jl);
    }
    sbuf_puts(&synth, "}");

    gr->api = SRV_API_MESSAGES;
    gr->is_chat = true;
    cwick_error jerr;
    cwick_error_clear(&jerr);
    json_value *chat_root = cwick_json_parse(synth.p, synth.len, &jerr);
    free(synth.p);
    if (!chat_root) {
        snprintf(errbuf, errcap, "invalid request: %s", cwick_error_message(&jerr));
        return 400;
    }
    int rc = validate_body(chat_root, gr, errbuf, errcap, model_id, cur);
    cwick_json_free(chat_root);
    return rc;
}

/* ------------------------------------------------------------------ */
/* Responses API: the response object and its output items             */
/* ------------------------------------------------------------------ */

/* The `usage` object of a response (Responses spells the same numbers
 * differently from chat/completions: input/output instead of prompt/
 * completion, and reasoning tokens broken out). */
static void resp_put_usage(srv_buf *b, int64_t in_tokens, int64_t out_tokens,
                           int64_t cached_tokens, int64_t reasoning_tokens) {
    sbuf_putf(b,
              "{\"input_tokens\":%lld,\"input_tokens_details\":{\"cached_tokens\":%lld},"
              "\"output_tokens\":%lld,\"output_tokens_details\":"
              "{\"reasoning_tokens\":%lld},\"total_tokens\":%lld}",
              (long long)in_tokens, (long long)cached_tokens,
              (long long)out_tokens, (long long)reasoning_tokens,
              (long long)(in_tokens + out_tokens));
}

static void resp_put_text_part(srv_buf *b, const char *text) {
    char *j = cwick_json_stringify(text ? text : "");
    sbuf_putf(b, "{\"type\":\"output_text\",\"text\":%s,\"annotations\":[]}", j);
    free(j);
}

/* A `message` output item holding one output_text part. */
static void resp_put_message_item(srv_buf *b, const char *item_id,
                                  const char *text, const char *status) {
    sbuf_putf(b, "{\"id\":\"%s\",\"type\":\"message\",\"status\":\"%s\","
                 "\"role\":\"assistant\",\"content\":[",
              item_id, status);
    resp_put_text_part(b, text);
    sbuf_puts(b, "]}");
}

/* A `reasoning` output item. The summary is the checkpoint's own reasoning
 * block — this engine has no separate summariser, so the whole block is
 * reported as its summary when the client asks for one. */
static void resp_put_reasoning_item(srv_buf *b, const char *item_id,
                                    const char *reasoning, const char *status) {
    sbuf_putf(b, "{\"id\":\"%s\",\"type\":\"reasoning\",\"status\":\"%s\","
                 "\"summary\":[", item_id, status);
    if (reasoning && *reasoning) {
        char *j = cwick_json_stringify(reasoning);
        sbuf_putf(b, "{\"type\":\"summary_text\",\"text\":%s}", j);
        free(j);
    }
    sbuf_puts(b, "]}");
}

/* A `function_call` output item. `call_id` is what the client echoes back in
 * a function_call_output item, so it is the model-parsed call id, not the
 * item id. */
static void resp_put_call_item(srv_buf *b, const char *item_id,
                               const char *call_id, const char *name,
                               const char *arguments) {
    char *jn = cwick_json_stringify(name ? name : "");
    char *ja = cwick_json_stringify(arguments ? arguments : "{}");
    char *jc = cwick_json_stringify(call_id ? call_id : "");
    sbuf_putf(b, "{\"id\":\"%s\",\"type\":\"function_call\",\"status\":\"completed\","
                 "\"call_id\":%s,\"name\":%s,\"arguments\":%s}",
              item_id, jc, jn, ja);
    free(jn);
    free(ja);
    free(jc);
}

/* The whole `response` object. `status` is "completed" on the final body and
 * the same builder is used for `response.completed`, so the streaming and
 * buffered paths cannot drift apart.
 *
 * `reasoning` is the turn's reasoning block and `summary` says whether the
 * client asked for it (`reasoning.summary`). The *item* exists either way —
 * the stream announces it as soon as the first thinking token lands, so a
 * body without it would name a different output list than the stream did.
 * Without a request for a summary the item carries an empty one, which is
 * exactly what the chat endpoint does with `reasoning_content: null`.
 *
 * `incomplete` is the budget case: a turn that stopped because
 * `max_output_tokens` ran out is *not* a completed response, and the status
 * is the only place a client can learn that (the Anthropic dialect says
 * `stop_reason: max_tokens` instead). */
static char *responses_body(const char *id, int64_t created, const char *model,
                            const char *text, const char *reasoning,
                            bool summary, bool incomplete,
                            const cwick_tool_call *calls, size_t ncalls,
                            int64_t in_tokens, int64_t out_tokens,
                            int64_t cached_tokens, int64_t reasoning_tokens,
                            const char *tail) {
    srv_buf b = {0};
    char *jmodel = cwick_json_stringify(model ? model : "");
    /* Item ids are derived from the response id, not minted independently:
     * the terminal `response.completed` must name the same items the stream
     * announced in `output_item.added`, and a client correlates them by id. */
    char mid[96], rid[96], fid[96];
    snprintf(mid, sizeof(mid), "%s-msg", id);
    snprintf(rid, sizeof(rid), "%s-rs", id);
    sbuf_putf(&b, "{\"id\":\"%s\",\"object\":\"response\",\"created_at\":%lld,"
                  "\"status\":\"%s\",\"model\":%s,\"output\":[",
              id, (long long)created, incomplete ? "incomplete" : "completed",
              jmodel);
    free(jmodel);
    bool first = true;
    if (reasoning && *reasoning) {
        resp_put_reasoning_item(&b, rid, summary ? reasoning : NULL, "completed");
        first = false;
    }
    if ((text && *text) || ncalls == 0) {
        /* A turn that made no tool call always has a message item, even when
         * the model spent its whole budget thinking: the client reads the
         * answer from that item, and its absence means "no answer" rather
         * than "empty answer". */
        if (!first) sbuf_puts(&b, ",");
        resp_put_message_item(&b, mid, text, "completed");
        first = false;
    }
    for (size_t i = 0; i < ncalls; i++) {
        if (!first) sbuf_puts(&b, ",");
        first = false;
        snprintf(fid, sizeof(fid), "%s-fc%zu", id, i);
        resp_put_call_item(&b, fid, calls[i].id, calls[i].name,
                           calls[i].arguments_json);
    }
    sbuf_puts(&b, "],\"usage\":");
    resp_put_usage(&b, in_tokens, out_tokens, cached_tokens, reasoning_tokens);
    /* `error` is null on anything that ran to a verdict; the failed variant
     * of this object (see stream_error) fills it in. */
    sbuf_puts(&b, ",\"error\":null,\"incomplete_details\":");
    sbuf_puts(&b, incomplete ? "{\"reason\":\"max_output_tokens\"}" : "null");
    if (tail && *tail) sbuf_puts(&b, tail);
    sbuf_puts(&b, "}");
    return b.p;
}

/* ------------------------------------------------------------------ */
/* Anthropic Messages API: the `message` object                         */
/* ------------------------------------------------------------------ */

/* A `tool_use` block. Anthropic's `input` is the parsed object, not a JSON
 * string, so the arguments cwick parsed out of the model's own protocol are
 * inlined here; they are already valid JSON or the turn would have been
 * refused as a malformed call. */
static void msg_put_tool_use(srv_buf *b, const cwick_tool_call *c) {
    char *ji = cwick_json_stringify(c->id);
    char *jn = cwick_json_stringify(c->name);
    const char *args = (c->arguments_json && *c->arguments_json)
                           ? c->arguments_json : "{}";
    sbuf_putf(b, "{\"type\":\"tool_use\",\"id\":%s,\"name\":%s,\"input\":%s}",
              ji, jn, args);
    free(ji);
    free(jn);
}

/* The whole `message`. Same builder for the buffered body and — through the
 * streaming path's own envelope — the same numbers, so the two cannot drift
 * apart the way two hand-written JSON blobs would. */
static char *messages_body(const char *id, const char *model, const char *text,
                           const char *reasoning, const cwick_tool_call *calls,
                           size_t ncalls, int64_t in_tokens, int64_t out_tokens,
                           int64_t cached_tokens, const char *stop_reason) {
    srv_buf b = {0};
    char *jmodel = cwick_json_stringify(model ? model : "");
    sbuf_putf(&b, "{\"id\":\"%s\",\"type\":\"message\",\"role\":\"assistant\","
                  "\"model\":%s,\"content\":[", id, jmodel);
    free(jmodel);
    bool first = true;
    if (reasoning && *reasoning) {
        char *jr = cwick_json_stringify(reasoning);
        /* `signature` is part of the block on the wire; this engine has no
         * signing key and never verifies a replayed chain, so it is empty. */
        sbuf_putf(&b, "{\"type\":\"thinking\",\"thinking\":%s,"
                      "\"signature\":\"\"}", jr);
        free(jr);
        first = false;
    }
    if (text && *text) {
        char *jt = cwick_json_stringify(text);
        if (!first) sbuf_puts(&b, ",");
        sbuf_putf(&b, "{\"type\":\"text\",\"text\":%s}", jt);
        free(jt);
        first = false;
    }
    for (size_t i = 0; i < ncalls; i++) {
        if (!first) sbuf_puts(&b, ",");
        msg_put_tool_use(&b, &calls[i]);
        first = false;
    }
    sbuf_putf(&b, "],\"stop_reason\":\"%s\",\"stop_sequence\":null,\"usage\":{",
              stop_reason);
    sbuf_putf(&b, "\"input_tokens\":%lld,\"output_tokens\":%lld",
              (long long)in_tokens, (long long)out_tokens);
    if (cached_tokens > 0)
        sbuf_putf(&b, ",\"cache_read_input_tokens\":%lld",
                  (long long)cached_tokens);
    sbuf_puts(&b, "}}");
    return b.p;
}

/* ------------------------------------------------------------------ */
/* OpenAPI-ish helpers for the two GET endpoints that take query parameters.
 * The request line keeps its query (see read_http_request), so every query
 * has to be read where it is used. */
static void path_only(const char *path, char *out, size_t cap) {
    size_t n = 0;
    for (const char *p = path; *p && *p != '?' && *p != '#'; p++) {
        if (n + 1 >= cap) break;
        out[n++] = *p;
    }
    out[n] = '\0';
}
/* The decision endpoint answers on the published path (`/inference/v1/…`, so a
 * client pointed at `http://host:port/inference` lands on it) and on the two
 * short forms the built-in apps and the README use. */
static bool is_systemone_path(const char *p) {
    return strcmp(p, "/v1/systemone") == 0 || strcmp(p, "/systemone") == 0 ||
           strcmp(p, "/inference/v1/systemone") == 0;
}

/* Percent-decode one URL path segment. Model ids come from checkpoint
 * directory names, which on macOS may contain a space; agents escape those as
 * %20. Decoding is deliberately lossy rather than fatal: a stray '%' stays
 * literal, and the name then simply will not match a catalog entry. */
static void url_decode_segment(const char *in, char *out, size_t cap) {
    size_t n = 0;
    for (const char *p = in; *p && n + 1 < cap; p++) {
        if (*p == '%' && isxdigit((unsigned char)p[1]) &&
            isxdigit((unsigned char)p[2])) {
            char hex[3] = { p[1], p[2], 0 };
            out[n++] = (char)strtol(hex, NULL, 16);
            p += 2;
        } else {
            out[n++] = *p;
        }
    }
    out[n] = '\0';
}
static const char *query_find(const char *path, const char *key) {
    const char *q = strchr(path, '?');
    if (!q) return NULL;
    size_t klen = strlen(key);
    for (const char *p = q + 1; *p;) {
        const char *amp = strchr(p, '&');
        size_t seg = amp ? (size_t)(amp - p) : strlen(p);
        if (seg > klen && p[klen] == '=' && memcmp(p, key, klen) == 0)
            return p + klen + 1;
        if (!amp) break;
        p = amp + 1;
    }
    return NULL;
}
/* Decode one query value in place. Returns false on a malformed escape or a
 * value that would not fit; callers turn that into a 400, never a guess. */
static bool query_decode(const char *src, char *out, size_t cap) {
    if (!src || !out || cap == 0) return false;
    size_t n = 0;
    for (const char *p = src; *p && *p != '&' && *p != '#'; p++) {
        int c = (unsigned char)*p;
        if (c == '+') { c = ' '; }
        else if (c == '%') {
            if (!(p[1] && p[2])) return false;
            int hi = p[1] >= '0' && p[1] <= '9' ? p[1] - '0' :
                     p[1] >= 'a' && p[1] <= 'f' ? p[1] - 'a' + 10 :
                     p[1] >= 'A' && p[1] <= 'F' ? p[1] - 'A' + 10 : -1;
            int lo = p[2] >= '0' && p[2] <= '9' ? p[2] - '0' :
                     p[2] >= 'a' && p[2] <= 'f' ? p[2] - 'a' + 10 :
                     p[2] >= 'A' && p[2] <= 'F' ? p[2] - 'A' + 10 : -1;
            if (hi < 0 || lo < 0) return false;
            c = (hi << 4) | lo;
            p += 2;
        }
        if (c == 0 || n + 1 >= cap) return false;
        out[n++] = (char)c;
    }
    out[n] = '\0';
    return n > 0;
}

static int64_t query_int(const char *path, const char *key, int64_t dflt) {
    const char *v = query_find(path, key);
    if (!v) return dflt;
    int64_t out = 0;
    size_t n = 0;
    bool neg = false;
    if (*v == '-') { neg = true; v++; }
    while (v[n] >= '0' && v[n] <= '9' && n < 20) {
        out = out * 10 + (v[n] - '0');
        n++;
    }
    if (n == 0) return dflt;
    /* Ignore anything after the digits ('&', '#', garbage). */
    return neg ? -out : out;
}

/* OpenAI `tool_calls` array for a set of parsed calls. With `empty_first` the
 * first call's arguments are left empty: those bytes were already pushed as
 * streaming deltas, and a client concatenates them onto this field. */
static char *tool_calls_json(const cwick_tool_call *calls, size_t n,
                             bool empty_first) {
    if (!calls || n == 0) return NULL;
    char *joined = srv_asprintf("[");
    for (size_t i = 0; i < n; i++) {
        const char *raw = (empty_first && i == 0) ? ""
                         : (calls[i].arguments_json ? calls[i].arguments_json : "{}");
        char *args = cwick_json_stringify(raw);
        char *name = cwick_json_stringify(calls[i].name);
        char *next = srv_asprintf("%s%s{\"id\":\"%s\",\"type\":\"function\","
                                 "\"function\":{\"name\":%s,\"arguments\":%s}}",
                                 joined, i ? "," : "", calls[i].id, name, args);
        free(joined);
        free(name);
        free(args);
        joined = next;
    }
    char *out = srv_asprintf("%s]", joined);
    free(joined);
    return out;
}

/* ---- --trace: what the model was asked and what it answered -------- */
static void trace_request(server_ctx *srv, const gen_request *gr) {
    if (!srv->trace) return;
    const char *tag = req_tag(gr->rid);
    if (gr->tools.present) {
        char Decl[192];
        for (size_t i = 0; i < gr->tools.count; i++) {
            snprintf(Decl, sizeof(Decl), "%s tool #%zu", tag, i + 1);
            cwick_log_block(CWICK_LOG_TRACE, Decl, gr->tools.items[i].json, 4096, 4);
        }
        char msg[256];
        msg[0] = '\0';
        if (strcmp(gr->tools.choice, "named") == 0)
            snprintf(msg, sizeof(msg), ", forced=%s", gr->tools.forced);
        CWICK_TRAC(tag, "tools declared: %zu (tool_choice=%s%s)", gr->tools.count,
                  gr->tools.choice, msg);
    }
    if (gr->is_chat) {
        if (gr->system) cwick_log_block(CWICK_LOG_TRACE, req_tag(gr->rid),
                                       gr->system, 8192, 4);
        cwick_log_block(CWICK_LOG_TRACE, req_tag(gr->rid),
                       gr->user ? gr->user : "", 8192, 4);
    } else {
        cwick_log_block(CWICK_LOG_TRACE, req_tag(gr->rid),
                       gr->prompt ? gr->prompt : "", 8192, 4);
    }
    if (gr->tools_block)
        cwick_log_block(CWICK_LOG_TRACE, req_tag(gr->rid), gr->tools_block,
                       8192, 4);
}

static void trace_output(server_ctx *srv, uint64_t rid, const char *text,
                         const cwick_tool_call *calls, size_t n) {
    if (!srv->trace) return;
    const char *tag = req_tag(rid);
    if (text && *text)
        cwick_log_block(CWICK_LOG_TRACE, tag, text, 16384, 4);
    for (size_t i = 0; i < n; i++) {
        char buf[512];
        snprintf(buf, sizeof(buf), "%s > %s", tag, calls[i].name);
        cwick_log_block(CWICK_LOG_TRACE, buf, calls[i].arguments_json, 8192, 4);
    }
    if (calls && n)
        CWICK_TRAC(tag, "tool calls emitted: %zu", n);
}

/* Some checkpoints write the <think> opener themselves -- their template ends
 * at <|assistant|> instead of pre-opening the reasoning block, so the marker
 * arrives as the first generated token. It is framing, not reasoning, and a
 * client that renders reasoning_content verbatim would show it, so drop it. */
static const char *skip_think_opener(const char *text, const char *open_mk) {
    const char *open = open_mk ? open_mk : "<think>";
    size_t olen = strlen(open);
    if (text && strncmp(text, open, olen) == 0) return text + olen;
    return text;
}

/* The generation prompt ends with "<|assistant|><think>", so a chat answer
 * starts inside the reasoning block: everything up to "</think>" is the
 * model's reasoning, everything after is what the user should see. */
static void split_reasoning(const char *text, const char *open_mk,
                            const char *close_mk, bool closed_at_start,
                            char **reasoning, char **content) {
    *reasoning = NULL;
    *content = NULL;
    if (!text) return;
    /* The prompt already closed the block, so the answer starts in content:
     * there is no closing marker in the generated text to split on, and
     * without this the whole reply went out as reasoning_content. */
    if (closed_at_start) {
        *content = cwick_strdup(text);
        return;
    }
    const char *close_tag = close_mk ? close_mk : "</think>";
    text = skip_think_opener(text, open_mk);
    const char *close = strstr(text, close_tag);
    if (close) {
        size_t rlen = (size_t)(close - text);
        char *r = cwick_xmalloc(rlen + 1);
        memcpy(r, text, rlen);
        r[rlen] = '\0';
        *reasoning = r;
        const char *rest = close + strlen(close_tag);
        while (*rest == '\n' || *rest == '\r') rest++;
        *content = cwick_strdup(rest);
        return;
    }
    /* no closing tag: the model is still thinking (hit max_tokens) */
    *reasoning = cwick_strdup(text);
    *content = cwick_strdup("");
}

void cwick_phase_account(const cwick_tokenizer *tok, const uint32_t *tokens,
                        size_t n, bool is_chat, size_t out[3]) {
    cwick_phase_account_mk(tok, tokens, n, is_chat, out, NULL, NULL, false);
}

/* ------------------------------------------------------------------ */
/* Incremental SSE streaming                                           */
/* ------------------------------------------------------------------ */
/* Which part of the answer a generation is producing, in the words the log
 * should use. Derived from the same state machine that splits the stream, so
 * a logged transition is a transition on the wire. */
typedef struct {
    cwick_engine *eng;        /* engine whose progress lines carry the label */
    const char *tag;         /* "req 7" — for the transition line */
    const char *phase;       /* current label, NULL before the first token */
    int64_t first_tok;       /* generation index of the phase's first token */
    double t0_ms;            /* wall clock when that phase began */
} phase_trk;

static const char *phase_label(const cwick_stream_state *st) {
    return !st->after_think ? "thinking" : (st->tool_mode ? "tools" : "content");
}

/* Publish the current phase to the engine (and log the transition), once per
 * phase change. `gen_index` is 1-based: the phase the sink was looking at
 * when the Nth generated token was produced. */
static void phase_trk_observe(phase_trk *pt, const cwick_stream_state *st,
                              int64_t gen_index) {
    const char *ph = phase_label(st);
    if (ph == pt->phase) return;
    if (pt->phase) {
        /* Close out the previous phase with its own rate: the transition is
         * the only place that knows how many tokens that phase cost, and a
         * long tool argument is otherwise indistinguishable from a hang. */
        double secs = (now_ms() - pt->t0_ms) / 1000.0;
        int64_t ntok = gen_index - pt->first_tok;
        CWICK_INFO(pt->tag, "decode phase: %s -> %s after %lld tokens "
                   "(%.1fs, %.2f tok/s)", pt->phase, ph, (long long)ntok,
                   secs, secs > 0.0 ? (double)ntok / secs : 0.0);
    }
    else
        CWICK_INFO(pt->tag, "decode phase: %s at token %lld", ph,
                  (long long)gen_index);
    pt->phase = ph;
    pt->first_tok = gen_index;
    pt->t0_ms = now_ms();
    cwick_engine_set_decode_phase(pt->eng, ph);
}

/* Count a finished generation's tokens by phase (see cwick.h). Replays the
 * segment machine token by token; the final token releases the held tail so
 * nothing is left unclassified. */
/* Marker-less entry point, kept for the callers (and tests) that only know
 * GLM's `<think>` pair. */
void cwick_phase_account(const cwick_tokenizer *tok, const uint32_t *tokens,
                        size_t n, bool is_chat, size_t out[3]);

void cwick_phase_account_mk(const cwick_tokenizer *tok, const uint32_t *tokens,
                           size_t n, bool is_chat, size_t out[3],
                           const char *open_mk, const char *close_mk,
                           bool closed_at_start) {
    if (!out) return;
    out[0] = out[1] = out[2] = 0;
    if (!tok || !tokens || n == 0) return;
    cwick_stream_state st;
    memset(&st, 0, sizeof(st));
    st.after_think = !is_chat || closed_at_start;
    /* The markers are per checkpoint. Without them this replay looked for
     * `</think>` in a MiniMax answer that only ever writes `</mm:think>`, so
     * every access-log line reported think=<all tokens>, content=0 -- on
     * requests whose streamed phase log said "decode(content)". */
    st.open_mk = open_mk;
    st.close_mk = close_mk;
    char *buf = cwick_xmalloc(64);
    size_t len = 0, cap = 64;
    int ph = (is_chat && !closed_at_start) ? CWICK_PHASE_THINKING
                                           : CWICK_PHASE_CONTENT;
    for (size_t i = 0; i < n; i++) {
        cwick_error e;
        cwick_error_clear(&e);
        char *piece = NULL;
        if (cwick_tokenizer_decode(tok, tokens + i, 1, &piece, &e) == CWICK_OK &&
            piece) {
            size_t plen = strlen(piece);
            while (len + plen + 1 > cap) {
                cap *= 2;
                buf = cwick_xrealloc(buf, cap);
            }
            memcpy(buf + len, piece, plen);
            len += plen;
            buf[len] = '\0';
            free(piece);
        }
        for (;;) {
            cwick_segment_kind kind = CWICK_SEG_CONTENT;
            size_t off = 0, slen = 0;
            if (!cwick_stream_segment(buf, len, i + 1 == n, &st, &kind, &off,
                                     &slen)) {
                if (st.tool_mode) ph = CWICK_PHASE_TOOLS;
                break;
            }
            ph = (kind == CWICK_SEG_REASONING) ? CWICK_PHASE_THINKING
                                              : CWICK_PHASE_CONTENT;
        }
        out[ph]++;
    }
    free(buf);
}

/* The engine hands us each token the moment it is decoded; here we turn it
 * into text and push the newly-completed bytes out as `reasoning_content` /
 * `content` deltas, so the client renders the answer as it is produced instead
 * of waiting for the whole turn.
 *
 * The text is split exactly like the buffered path does after the fact
 * (split_reasoning + cwick_tools_extract): everything before `</think>` is
 * reasoning, everything before the first `<tool_call>` is content, and from
 * there on the bytes belong to tool calls. Because a marker can straddle two
 * tokens we hold back a small tail and only classify bytes that cannot be the
 * start of a marker; the tail is flushed when generation ends. */
typedef struct {
    server_ctx *srv;
    const gen_request *gr;
    int fd;
    char idbuf[CWICK_SERVER_ID_BUF];
    int64_t created;
    const char *model;

    char *raw;                /* full detokenized answer so far */
    size_t raw_len, raw_cap;
    cwick_stream_state split;  /* reasoning / content / tool-call phase */
    size_t tool_sent;         /* raw bytes already forwarded as tool progress */
    bool started;             /* response head + first frame written */
    bool dead;                /* the client went away */

    /* Decode-phase label for the engine's per-step log lines, advanced as
     * `split` classifies each token. */
    phase_trk pt;

    cwick_engine *eng;         /* engine generating this request (for pt) */

    /* Time-to-first-token: when the first engine token reached this sink,
     * measured from the moment the generation started (the caller adds the
     * queue time it spent waiting for the slot). */
    double gen_t0_ms;
    double queue_ms;        /* slot wait, folded into the reported TTFT */
    bool first_token_seen;
    double first_token_ms;

    /* ---- Responses dialect (gr->api == SRV_API_RESPONSES) --------------
     * The same token stream, framed as response.* events instead of chat
     * choices. Items open lazily, on the first byte that belongs to them, so
     * a turn that produces only tool calls never announces an empty message
     * item. */
    bool resp_started;
    int resp_index;                 /* next output_index */
    bool resp_reason_open;
    bool resp_text_open;
    /* Every Responses event carries a monotonic `sequence_number`; clients
     * use it to detect a dropped or reordered frame, and the field is not
     * optional in the schema. */
    int64_t resp_seq;
    char resp_reason_id[96];
    char resp_msg_id[96];
    srv_buf resp_reason;            /* accumulated reasoning text */
    srv_buf resp_text;              /* accumulated answer text */

    /* ---- Anthropic Messages dialect (gr->api == SRV_API_MESSAGES) ------
     * Content blocks are numbered consecutively and opened lazily, on the
     * first byte that belongs to them: reasoning and prose each get at most
     * one block, tool calls one block each at the end of the turn. */
    int64_t in_tokens;              /* message_start reports input_tokens */
    int64_t cached_tokens;          /* prefix-cache replay, for usage */
    int msg_index;                  /* next content_block index */
    int msg_think_idx;              /* -1 when no thinking block is open */
    int msg_text_idx;               /* -1 when no text block is open */

    /* ---- standard tool-call progress (all three dialects) ------------
     * `cwick_tool` is a private field only the bundled console reads. A
     * protocol-aware client needs the call to be *opened* while the model is
     * still writing it, and it needs the arguments in that dialect's own
     * delta field. These fields track the one call whose arguments are
     * dribbled out; further calls still arrive whole at the finish. */
    bool tool_std_on;               /* in-flight block is the nested dialect */
    bool tool_std_open;             /* opening frame already on the wire */
    bool tool_std_closed;           /* first call settled at the finish */
    char tool_std_id[80];
    char tool_std_name[192];
    int  tool_std_msg_idx;          /* Messages content_block index used */
    int  tool_std_resp_idx;         /* Responses output_index used */
    size_t tool_std_begin;          /* raw offset of the tool block */
    size_t tool_std_params;         /* completed <parameter=> pairs seen */
    size_t tool_std_val_len;        /* value bytes at the last in-flight scan */
    char *tool_std_prefix;          /* argument-json bytes already emitted */
    size_t tool_std_sent;           /* strlen(tool_std_prefix) */
} stream_ctx;

static void stream_ctx_free(stream_ctx *sc) {
    free(sc->raw);
    sc->raw = NULL;
    free(sc->tool_std_prefix);
    sc->tool_std_prefix = NULL;
    free(sc->resp_reason.p);
    sc->resp_reason.p = NULL;
    free(sc->resp_text.p);
    sc->resp_text.p = NULL;
}

static const char *stream_content_field(const stream_ctx *sc) {
    return sc->gr->is_chat ? "content" : "text";
}

static char *json_slice(const char *p, size_t n) {
    char *tmp = cwick_xmalloc(n + 1);
    memcpy(tmp, p, n);
    tmp[n] = '\0';
    char *esc = cwick_json_stringify(tmp);
    free(tmp);
    return esc;
}

/* ------------------------------------------------------------------ */
/* Responses streaming: the same deltas, framed as response.* events    */
/* ------------------------------------------------------------------ */

/* Send one named Responses event. Every payload here is a flat JSON object, so
 * the mandatory `sequence_number` is appended where it belongs (before the
 * closing brace) instead of being threaded through every builder. */
static void resp_sse(stream_ctx *sc, const char *event, const char *json) {
    size_t n = json ? strlen(json) : 0;
    char *with = NULL;
    if (n > 1 && json[n - 1] == '}') {
        with = srv_asprintf("%.*s,\"sequence_number\":%lld}",
                            (int)(n - 1), json, (long long)sc->resp_seq++);
    }
    if (!sse_named(sc->fd, event, with ? with : json)) sc->dead = true;
    free(with);
}

static void resp_stream_created(stream_ctx *sc, bool progress) {
    const char *ev = progress ? "response.in_progress" : "response.created";
    char *json = srv_asprintf(
        "{\"type\":\"%s\",\"response\":{\"id\":\"%s\",\"object\":\"response\","
        "\"created_at\":%lld,\"status\":\"in_progress\",\"model\":\"%s\","
        "\"output\":[],\"usage\":null,\"error\":null,"
        "\"incomplete_details\":null%s}}",
        ev, sc->idbuf, (long long)sc->created, sc->model,
        sc->gr->resp_tail ? sc->gr->resp_tail : "");
    resp_sse(sc, ev, json);
    free(json);
}

/* Close the open reasoning item. A summary was requested or it was not; the
 * closing event names differ, the item is the same. */
static void resp_close_reason(stream_ctx *sc) {
    if (!sc->resp_reason_open) return;
    sc->resp_reason_open = false;
    const char *text = sc->resp_reason.p ? sc->resp_reason.p : "";
    char *jtext = cwick_json_stringify(text);
    char *json;
    if (sc->gr->resp_summary) {
        json = srv_asprintf(
            "{\"type\":\"response.reasoning_summary_text.done\",\"item_id\":\"%s\","
            "\"output_index\":%d,\"summary_index\":0,\"text\":%s}",
            sc->resp_reason_id, sc->resp_index, jtext);
        resp_sse(sc, "response.reasoning_summary_text.done", json);
        free(json);
        json = srv_asprintf(
            "{\"type\":\"response.reasoning_summary_part.done\",\"item_id\":\"%s\","
            "\"output_index\":%d,\"summary_index\":0,"
            "\"part\":{\"type\":\"summary_text\",\"text\":%s}}",
            sc->resp_reason_id, sc->resp_index, jtext);
        resp_sse(sc, "response.reasoning_summary_part.done", json);
        free(json);
    } else {
        json = srv_asprintf(
            "{\"type\":\"response.reasoning_text.done\",\"item_id\":\"%s\","
            "\"output_index\":%d,\"content_index\":0,\"text\":%s}",
            sc->resp_reason_id, sc->resp_index, jtext);
        resp_sse(sc, "response.reasoning_text.done", json);
        free(json);
    }
    free(jtext);
    srv_buf item = {0};
    resp_put_reasoning_item(&item, sc->resp_reason_id,
                            sc->gr->resp_summary ? text : NULL, "completed");
    json = srv_asprintf("{\"type\":\"response.output_item.done\",\"output_index\":%d,"
                        "\"item\":%s}", sc->resp_index, item.p);
    resp_sse(sc, "response.output_item.done", json);
    free(json);
    free(item.p);
    sc->resp_index++;
}

static void resp_close_text(stream_ctx *sc) {
    if (!sc->resp_text_open) return;
    sc->resp_text_open = false;
    const char *text = sc->resp_text.p ? sc->resp_text.p : "";
    char *jtext = cwick_json_stringify(text);
    char *json = srv_asprintf(
        "{\"type\":\"response.output_text.done\",\"item_id\":\"%s\","
        "\"output_index\":%d,\"content_index\":0,\"text\":%s,\"logprobs\":[]}",
        sc->resp_msg_id, sc->resp_index, jtext);
    resp_sse(sc, "response.output_text.done", json);
    free(json);
    json = srv_asprintf(
        "{\"type\":\"response.content_part.done\",\"item_id\":\"%s\","
        "\"output_index\":%d,\"content_index\":0,"
        "\"part\":{\"type\":\"output_text\",\"text\":%s,\"annotations\":[]}}",
        sc->resp_msg_id, sc->resp_index, jtext);
    resp_sse(sc, "response.content_part.done", json);
    free(json);
    free(jtext);
    srv_buf item = {0};
    resp_put_message_item(&item, sc->resp_msg_id, text, "completed");
    json = srv_asprintf("{\"type\":\"response.output_item.done\",\"output_index\":%d,"
                        "\"item\":%s}", sc->resp_index, item.p);
    resp_sse(sc, "response.output_item.done", json);
    free(json);
    free(item.p);
    sc->resp_index++;
}

static void resp_open_reason(stream_ctx *sc) {
    if (sc->resp_reason_open) return;
    sc->resp_reason_open = true;
    snprintf(sc->resp_reason_id, sizeof(sc->resp_reason_id), "%s-rs", sc->idbuf);
    char *json = srv_asprintf(
        "{\"type\":\"response.output_item.added\",\"output_index\":%d,"
        "\"item\":{\"id\":\"%s\",\"type\":\"reasoning\",\"status\":\"in_progress\","
        "\"summary\":[]}}", sc->resp_index, sc->resp_reason_id);
    resp_sse(sc, "response.output_item.added", json);
    free(json);
    if (sc->gr->resp_summary) {
        json = srv_asprintf(
            "{\"type\":\"response.reasoning_summary_part.added\",\"item_id\":\"%s\","
            "\"output_index\":%d,\"summary_index\":0,"
            "\"part\":{\"type\":\"summary_text\",\"text\":\"\"}}",
            sc->resp_reason_id, sc->resp_index);
        resp_sse(sc, "response.reasoning_summary_part.added", json);
        free(json);
    }
}

static void resp_open_text(stream_ctx *sc) {
    if (sc->resp_text_open) return;
    resp_close_reason(sc);
    sc->resp_text_open = true;
    snprintf(sc->resp_msg_id, sizeof(sc->resp_msg_id), "%s-msg", sc->idbuf);
    char *json = srv_asprintf(
        "{\"type\":\"response.output_item.added\",\"output_index\":%d,"
        "\"item\":{\"id\":\"%s\",\"type\":\"message\",\"status\":\"in_progress\","
        "\"role\":\"assistant\",\"content\":[]}}",
        sc->resp_index, sc->resp_msg_id);
    resp_sse(sc, "response.output_item.added", json);
    free(json);
    json = srv_asprintf(
        "{\"type\":\"response.content_part.added\",\"item_id\":\"%s\","
        "\"output_index\":%d,\"content_index\":0,"
        "\"part\":{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]}}",
        sc->resp_msg_id, sc->resp_index);
    resp_sse(sc, "response.content_part.added", json);
    free(json);
}

/* One classified delta -> the Responses event that carries it. Tool-call
 * progress has no equivalent here: the arguments arrive complete in the
 * function_call item at the end of the turn. */
static void resp_stream_text(stream_ctx *sc, const char *field,
                             const char *p, size_t len) {
    if (sc->dead || len == 0) return;
    if (strcmp(field, "reasoning_content") == 0) {
        resp_open_reason(sc);
        sbuf_putn(&sc->resp_reason, p, len);
        char *esc = json_slice(p, len);
        char *json;
        if (sc->gr->resp_summary) {
            json = srv_asprintf(
                "{\"type\":\"response.reasoning_summary_text.delta\","
                "\"item_id\":\"%s\",\"output_index\":%d,\"summary_index\":0,"
                "\"delta\":%s}", sc->resp_reason_id, sc->resp_index, esc);
            resp_sse(sc, "response.reasoning_summary_text.delta", json);
        } else {
            json = srv_asprintf(
                "{\"type\":\"response.reasoning_text.delta\",\"item_id\":\"%s\","
                "\"output_index\":%d,\"content_index\":0,\"delta\":%s}",
                sc->resp_reason_id, sc->resp_index, esc);
            resp_sse(sc, "response.reasoning_text.delta", json);
        }
        free(json);
        free(esc);
        return;
    }
    if (strcmp(field, "content") == 0 || strcmp(field, "text") == 0) {
        resp_open_text(sc);
        sbuf_putn(&sc->resp_text, p, len);
        char *esc = json_slice(p, len);
        char *json = srv_asprintf(
            "{\"type\":\"response.output_text.delta\",\"item_id\":\"%s\","
            "\"output_index\":%d,\"content_index\":0,\"delta\":%s,"
            "\"logprobs\":[]}",
            sc->resp_msg_id, sc->resp_index, esc);
        resp_sse(sc, "response.output_text.delta", json);
        free(json);
        free(esc);
    }
}

/* ------------------------------------------------------------------ */
/* Anthropic Messages streaming: the same deltas as content blocks      */
/* ------------------------------------------------------------------ */

static void msg_sse(stream_ctx *sc, const char *event, const char *json) {
    if (!sse_named(sc->fd, event, json)) sc->dead = true;
}

static void msg_block_stop(stream_ctx *sc, int idx) {
    char *json = srv_asprintf("{\"type\":\"content_block_stop\",\"index\":%d}", idx);
    msg_sse(sc, "content_block_stop", json);
    free(json);
}

/* Blocks are sequential: a text block closes the thinking block before it
 * opens, because the protocol has one block in flight at a time. */
static void msg_open_think(stream_ctx *sc) {
    if (sc->msg_think_idx >= 0) return;
    if (sc->msg_text_idx >= 0) {
        msg_block_stop(sc, sc->msg_text_idx);
        sc->msg_text_idx = -1;
    }
    sc->msg_think_idx = sc->msg_index++;
    char *json = srv_asprintf(
        "{\"type\":\"content_block_start\",\"index\":%d,\"content_block\":"
        "{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}",
        sc->msg_think_idx);
    msg_sse(sc, "content_block_start", json);
    free(json);
}

static void msg_close_think(stream_ctx *sc) {
    if (sc->msg_think_idx < 0) return;
    msg_block_stop(sc, sc->msg_think_idx);
    sc->msg_think_idx = -1;
}

static void msg_open_text(stream_ctx *sc) {
    if (sc->msg_text_idx >= 0) return;
    msg_close_think(sc);
    sc->msg_text_idx = sc->msg_index++;
    char *json = srv_asprintf(
        "{\"type\":\"content_block_start\",\"index\":%d,\"content_block\":"
        "{\"type\":\"text\",\"text\":\"\"}}", sc->msg_text_idx);
    msg_sse(sc, "content_block_start", json);
    free(json);
}

static void msg_close_text(stream_ctx *sc) {
    if (sc->msg_text_idx < 0) return;
    msg_block_stop(sc, sc->msg_text_idx);
    sc->msg_text_idx = -1;
}

/* One classified delta -> the content block it belongs to. Tool-call bytes
 * have no block of their own until the call is parsed: `tool_use` carries
 * its `input` as a complete object, so progress is reported by the engine's
 * own decode lines rather than on this wire. */
static void msg_stream_text(stream_ctx *sc, const char *field,
                            const char *p, size_t len) {
    if (sc->dead || len == 0) return;
    if (strcmp(field, "reasoning_content") == 0) {
        msg_open_think(sc);
        char *esc = json_slice(p, len);
        char *json = srv_asprintf(
            "{\"type\":\"content_block_delta\",\"index\":%d,"
            "\"delta\":{\"type\":\"thinking_delta\",\"thinking\":%s}}",
            sc->msg_think_idx, esc);
        msg_sse(sc, "content_block_delta", json);
        free(json);
        free(esc);
        return;
    }
    if (strcmp(field, "content") == 0 || strcmp(field, "text") == 0) {
        msg_open_text(sc);
        char *esc = json_slice(p, len);
        char *json = srv_asprintf(
            "{\"type\":\"content_block_delta\",\"index\":%d,"
            "\"delta\":{\"type\":\"text_delta\",\"text\":%s}}",
            sc->msg_text_idx, esc);
        msg_sse(sc, "content_block_delta", json);
        free(json);
        free(esc);
    }
}

static void stream_delta_n(stream_ctx *sc, const char *field,
                           const char *p, size_t len) {
    if (len == 0 || sc->dead) return;
    if (sc->gr->api == SRV_API_RESPONSES) {
        resp_stream_text(sc, field, p, len);
        return;
    }
    if (sc->gr->api == SRV_API_MESSAGES) {
        msg_stream_text(sc, field, p, len);
        return;
    }
    char *esc = json_slice(p, len);
    char *json = srv_asprintf(
        "{\"id\":\"%s\",\"object\":\"%s\",\"created\":%lld,\"model\":\"%s\","
        "\"choices\":[{\"index\":0,\"delta\":{\"%s\":%s},\"finish_reason\":null}]}",
        sc->idbuf, sc->gr->is_chat ? "chat.completion.chunk" : "text_completion",
        (long long)sc->created, sc->model, field, esc);
    free(esc);
    if (!sse_frame(sc->fd, json)) sc->dead = true;
    free(json);
}

static void stream_delta(stream_ctx *sc, const char *field,
                         size_t off, size_t len) {
    stream_delta_n(sc, field, sc->raw + off, len);
}

/* Write the head and the opening role/content chunk. Sent lazily, on the first
 * token: a failure before any output can still become a normal HTTP error. */
static void stream_begin(stream_ctx *sc) {
    if (sc->started) return;
    sc->started = true;
    http_stream_head(sc->fd);
    if (sc->gr->api == SRV_API_RESPONSES) {
        /* The envelope exists before the first token, exactly like the chat
         * chunk that announces the assistant role. */
        resp_stream_created(sc, false);
        resp_stream_created(sc, true);
        return;
    }
    if (sc->gr->api == SRV_API_MESSAGES) {
        /* message_start is the whole envelope minus the content: the id, the
         * model and the input-token count are known here, because prefill has
         * not started yet. */
        char *mjson;
        if (sc->cached_tokens > 0)
            mjson = srv_asprintf(
                "{\"type\":\"message_start\",\"message\":{\"id\":\"%s\","
                "\"type\":\"message\",\"role\":\"assistant\",\"model\":\"%s\","
                "\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,"
                "\"usage\":{\"input_tokens\":%lld,\"output_tokens\":0,"
                "\"cache_read_input_tokens\":%lld}}}",
                sc->idbuf, sc->model, (long long)sc->in_tokens,
                (long long)sc->cached_tokens);
        else
            mjson = srv_asprintf(
                "{\"type\":\"message_start\",\"message\":{\"id\":\"%s\","
                "\"type\":\"message\",\"role\":\"assistant\",\"model\":\"%s\","
                "\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,"
                "\"usage\":{\"input_tokens\":%lld,\"output_tokens\":0}}}",
                sc->idbuf, sc->model, (long long)sc->in_tokens);
        msg_sse(sc, "message_start", mjson);
        free(mjson);
        return;
    }
    char *json = srv_asprintf(
        "{\"id\":\"%s\",\"object\":\"%s\",\"created\":%lld,\"model\":\"%s\","
        "\"choices\":[{\"index\":0,\"delta\":{%s\"%s\":\"\"},"
        "\"finish_reason\":null}]}",
        sc->idbuf, sc->gr->is_chat ? "chat.completion.chunk" : "text_completion",
        (long long)sc->created, sc->model,
        sc->gr->is_chat ? "\"role\":\"assistant\"," : "",
        stream_content_field(sc));
    if (!sse_frame(sc->fd, json)) sc->dead = true;
    free(json);
}

/* How many trailing bytes could still turn out to be the start of a split
 * marker? Holding exactly those back (rather than a fixed window) is what keeps
 * the stream incremental: normal text ends in no marker prefix, so every token
 * is forwarded the moment it arrives. */
static size_t marker_prefix_tail(const char *s, size_t len, const char *close_mk) {
    const char *marks[2] = { close_mk ? close_mk : "</think>", "<tool_call>" };
    size_t best = 0;
    for (int m = 0; m < 2; m++) {
        size_t mlen = strlen(marks[m]);
        size_t maxk = len < mlen - 1 ? len : mlen - 1;
        for (size_t k = maxk; k > 0; k--) {
            if (memcmp(s + len - k, marks[m], k) == 0) {
                if (k > best) best = k;
                break;
            }
        }
    }
    return best;
}

/* Bytes of a truncated UTF-8 sequence at the tail (0..3): emitting half a
 * character would put invalid UTF-8 inside a JSON string. */
static size_t utf8_incomplete_tail(const char *s, size_t len) {
    size_t back = 0;
    while (back < len && back < 3) {
        unsigned char c = (unsigned char)s[len - 1 - back];
        if ((c & 0xC0) != 0x80) {              /* lead byte */
            size_t need;
            if ((c & 0x80) == 0x00) need = 1;
            else if ((c & 0xE0) == 0xC0) need = 2;
            else if ((c & 0xF0) == 0xE0) need = 3;
            else if ((c & 0xF8) == 0xF0) need = 4;
            else return 0;                     /* not a valid lead byte */
            return (back + 1 < need) ? back + 1 : 0;
        }
        back++;
    }
    return 0;
}

bool cwick_stream_segment(const char *raw, size_t raw_len, bool final,
                         cwick_stream_state *st, cwick_segment_kind *kind,
                         size_t *off, size_t *len) {
    for (;;) {
        if (st->tool_mode) return false;

        /* Same opener as skip_think_opener(), one byte at a time: hold a
         * partial "<thi" back rather than emit it and then discover it was
         * framing. Only the very first bytes can be the opener. */
        if (!st->after_think && st->sent == 0) {
            const char *open = st->open_mk ? st->open_mk : "<think>";
            const size_t olen = strlen(open);
            size_t avail = raw_len < olen ? raw_len : olen;
            if (avail == olen && memcmp(raw, open, olen) == 0) {
                st->sent = olen;
            } else if (memcmp(raw, open, avail) == 0) {
                if (!final) return false;
            }
        }

        const char *close_mk = st->close_mk ? st->close_mk : "</think>";
        const char *marker = st->after_think ? "<tool_call>" : close_mk;
        const char *hit = strstr(raw, marker);
        size_t lim = raw_len;
        if (!final) {
            size_t hold = marker_prefix_tail(raw, raw_len, close_mk);
            size_t uhold = utf8_incomplete_tail(raw, raw_len);
            if (uhold > hold) hold = uhold;
            lim = raw_len > hold ? raw_len - hold : 0;
        }

        /* The buffered split_reasoning drops the newlines that follow
         * </think>. Here they can arrive one byte at a time, so keep dropping
         * them until the first content byte lands. */
        if (st->after_think && !st->content_started) {
            while (st->sent < lim &&
                   (raw[st->sent] == '\n' || raw[st->sent] == '\r'))
                st->sent++;
        }

        if (hit && (size_t)(hit - raw) >= st->sent) {
            size_t p = (size_t)(hit - raw);
            if (p > st->sent) {              /* prose just before the marker */
                *kind = st->after_think ? CWICK_SEG_CONTENT : CWICK_SEG_REASONING;
                *off = st->sent;
                *len = p - st->sent;
                st->sent = p;
                if (st->after_think) st->content_started = true;
                return true;
            }
            if (st->after_think) {           /* content stops at <tool_call> */
                st->tool_mode = true;
                st->sent = p;
                return false;
            }
            st->after_think = true;          /* re-classify the rest as content */
            st->sent = p + strlen(close_mk);
            continue;
        }
        if (lim > st->sent) {                /* plain projection */
            *kind = st->after_think ? CWICK_SEG_CONTENT : CWICK_SEG_REASONING;
            *off = st->sent;
            *len = lim - st->sent;
            st->sent = lim;
            if (st->after_think) st->content_started = true;
            return true;
        }
        return false;
    }
}

/* ------------------------------------------------------------------ */
/* Standard tool-call deltas                                           */
/* ------------------------------------------------------------------ */
/* `cwick_tool` is a private field only the bundled console reads. A client
 * that speaks one of the three public dialects needs the call *opened* while
 * the model is still writing it, and it needs the arguments in that dialect's
 * own delta field. The opening frame carries an empty `arguments`; every
 * later frame appends to it, and the finish appends the rest. Concatenating
 * the deltas therefore yields exactly the object in the final payload.
 *
 * Only the first call is dribbled out. If the model writes more, the rest
 * arrive whole at the finish, which is also the fallback whenever the block
 * is not in the nested <function=NAME> dialect. */

static void tool_std_open_frame(stream_ctx *sc) {
    if (sc->dead || sc->tool_std_open) return;
    sc->tool_std_open = true;
    char *jid = cwick_json_stringify(sc->tool_std_id);
    char *jname = cwick_json_stringify(sc->tool_std_name);
    char fid[96];
    snprintf(fid, sizeof(fid), "%s-fc0", sc->idbuf);
    char *json;

    if (sc->gr->api == SRV_API_RESPONSES) {
        /* Output items are sequential, so the text item has to close first. */
        resp_close_reason(sc);
        resp_close_text(sc);
        sc->tool_std_resp_idx = sc->resp_index;
        json = srv_asprintf(
            "{\"type\":\"response.output_item.added\",\"output_index\":%d,"
            "\"item\":{\"id\":\"%s\",\"type\":\"function_call\","
            "\"status\":\"in_progress\",\"call_id\":%s,\"name\":%s,"
            "\"arguments\":\"\"}}",
            sc->tool_std_resp_idx, fid, jid, jname);
        resp_sse(sc, "response.output_item.added", json);
    } else if (sc->gr->api == SRV_API_MESSAGES) {
        msg_close_think(sc);
        msg_close_text(sc);
        sc->tool_std_msg_idx = sc->msg_index++;
        json = srv_asprintf(
            "{\"type\":\"content_block_start\",\"index\":%d,\"content_block\":"
            "{\"type\":\"tool_use\",\"id\":%s,\"name\":%s,\"input\":{}}}",
            sc->tool_std_msg_idx, jid, jname);
        msg_sse(sc, "content_block_start", json);
    } else {
        json = srv_asprintf(
            "{\"id\":\"%s\",\"object\":\"%s\",\"created\":%lld,\"model\":\"%s\","
            "\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":["
            "{\"index\":0,\"id\":%s,\"type\":\"function\",\"function\":"
            "{\"name\":%s,\"arguments\":\"\"}}]},\"finish_reason\":null}]}",
            sc->idbuf, sc->gr->is_chat ? "chat.completion.chunk" : "text_completion",
            (long long)sc->created, sc->model, jid, jname);
        if (!sse_frame(sc->fd, json)) sc->dead = true;
    }
    free(json);
    free(jid);
    free(jname);
}

static void tool_std_args_frame(stream_ctx *sc, const char *piece, size_t len) {
    if (sc->dead || !sc->tool_std_open || len == 0) return;
    char *esc = json_slice(piece, len);
    char fid[96];
    snprintf(fid, sizeof(fid), "%s-fc0", sc->idbuf);
    char *json;

    if (sc->gr->api == SRV_API_RESPONSES) {
        json = srv_asprintf(
            "{\"type\":\"response.function_call_arguments.delta\","
            "\"item_id\":\"%s\",\"output_index\":%d,\"delta\":%s}",
            fid, sc->tool_std_resp_idx, esc);
        resp_sse(sc, "response.function_call_arguments.delta", json);
    } else if (sc->gr->api == SRV_API_MESSAGES) {
        json = srv_asprintf(
            "{\"type\":\"content_block_delta\",\"index\":%d,\"delta\":"
            "{\"type\":\"input_json_delta\",\"partial_json\":%s}}",
            sc->tool_std_msg_idx, esc);
        msg_sse(sc, "content_block_delta", json);
    } else {
        json = srv_asprintf(
            "{\"id\":\"%s\",\"object\":\"%s\",\"created\":%lld,\"model\":\"%s\","
            "\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":["
            "{\"index\":0,\"function\":{\"arguments\":%s}}]},"
            "\"finish_reason\":null}]}",
            sc->idbuf, sc->gr->is_chat ? "chat.completion.chunk" : "text_completion",
            (long long)sc->created, sc->model, esc);
        if (!sse_frame(sc->fd, json)) sc->dead = true;
    }
    free(json);
    free(esc);
}

static void tool_std_emit_prefix(stream_ctx *sc, const char *blk, size_t cut,
                                 bool strip_value_quote);
static void tool_std_value_scan(stream_ctx *sc, const char *blk,
                                const char *fn_end);

/* Look at everything the model has written since <tool_call> opened. Instead
 * of a second XML grammar on the streaming side, hand a *completed* prefix of
 * the block to the same cwick_tools_extract() the buffered path uses: that
 * gives the argument object for every parameter whose closing tag has
 * arrived, with the same value normalisation and the same key order. Dropping
 * its final brace leaves a prefix that can only grow, which is what makes it
 * safe to emit a little at a time. */
static void tool_std_scan(stream_ctx *sc) {
    static const char fn_open[] = "<function=";
    static const char par_close[] = "</parameter>";
    static const char fn_close[] = "</function>";
    if (sc->dead || !sc->gr->stream || !sc->gr->tools.present ||
        !sc->gr->tools.advertise || sc->raw_len <= sc->tool_std_begin) return;
    const char *blk = sc->raw + sc->tool_std_begin;

    if (!sc->tool_std_on) {
        const char *f = strstr(blk, fn_open);
        if (!f) return;                    /* a different checkpoint's dialect */
        const char *nm = f + sizeof(fn_open) - 1;
        const char *e = strchr(nm, '>');
        if (!e) return;                    /* tag still arriving */
        size_t nl = (size_t)(e - nm);
        if (nl == 0 || nl >= sizeof(sc->tool_std_name)) return;
        memcpy(sc->tool_std_name, nm, nl);
        sc->tool_std_name[nl] = '\0';
        snprintf(sc->tool_std_id, sizeof(sc->tool_std_id), "call_%ld_s%d",
                 (long)getpid() % 100000, (int)(sc->created % 100000));
        sc->tool_std_on = true;
        tool_std_open_frame(sc);
    }
    if (sc->dead) return;

    size_t np = 0, cut = 0;
    for (const char *q = blk; (q = strstr(q, par_close)) != NULL;
         q += sizeof(par_close) - 1) {
        np++;
        cut = (size_t)(q - blk) + sizeof(par_close) - 1;
    }
    /* Keep the first <function> block only; further calls arrive at the end. */
    const char *fe = strstr(blk, fn_close);
    if (fe) cut = cut < (size_t)(fe - blk) ? cut : (size_t)(fe - blk);

    if (np > sc->tool_std_params) {
        sc->tool_std_params = np;
        sc->tool_std_val_len = 0;   /* a new parameter starts from zero */
        tool_std_emit_prefix(sc, blk, cut, false);
    }
    /* A parameter value is the slow part of a real tool call (a shell script,
     * a file body) and it is written after `</parameter>` has had time to
     * arrive only at the very end. Without this the client sees the opening
     * frame and then silence for the whole value. */
    if (!sc->dead) tool_std_value_scan(sc, blk, fe);
}

/* Emit the argument JSON up to `cut` bytes of the block. `strip_value_quote`
 * is for the in-flight case: the parameter is only *going* to be closed, so
 * the closing quote the extractor adds is not part of the prefix yet. */
static void tool_std_emit_prefix(stream_ctx *sc, const char *blk, size_t cut,
                                 bool strip_value_quote) {
    static const char head[] = "<tool_call>";
    static const char tail[] = "</function></tool_call>";
    size_t plen = sizeof(head) - 1 + cut + sizeof(tail) - 1;
    char *probe = cwick_xmalloc(plen + 1);
    memcpy(probe, head, sizeof(head) - 1);
    memcpy(probe + sizeof(head) - 1, blk, cut);
    memcpy(probe + sizeof(head) - 1 + cut, tail, sizeof(tail) - 1);
    probe[plen] = '\0';

    cwick_error e;
    cwick_error_clear(&e);
    cwick_tool_call *pc = NULL;
    size_t npc = 0;
    int rc = cwick_tools_extract(probe, &pc, &npc, NULL, &e);
    free(probe);
    if (rc != 0 || npc == 0) {
        cwick_tool_calls_free(pc, npc);
        return;
    }
    const char *args = pc[0].arguments_json ? pc[0].arguments_json : "";
    size_t alen = strlen(args);
    if (alen && args[alen - 1] == '}') alen--;    /* leave the object open */
    if (strip_value_quote && alen && args[alen - 1] == '"') alen--;
    if (alen > sc->tool_std_sent &&
        strncmp(args, sc->tool_std_prefix ? sc->tool_std_prefix : "",
                sc->tool_std_sent) == 0) {
        tool_std_args_frame(sc, args + sc->tool_std_sent,
                            alen - sc->tool_std_sent);
        char *grown = cwick_xmalloc(alen + 1);
        if (sc->tool_std_sent)
            memcpy(grown, sc->tool_std_prefix, sc->tool_std_sent);
        memcpy(grown + sc->tool_std_sent, args + sc->tool_std_sent,
               alen - sc->tool_std_sent);
        grown[alen] = '\0';
        free(sc->tool_std_prefix);
        sc->tool_std_prefix = grown;
        sc->tool_std_sent = alen;
    }
    cwick_tool_calls_free(pc, npc);
}

/* Can the bytes written so far only ever become a JSON string? A value that
 * starts with `{ [ " -` or a digit may still parse as JSON, and `t`/`f`/`n`
 * may still become `true`/`false`/`null`; those are left to the closed-parameter
 * path. A bare word (a path, a sentence, a shell script) never can. This test
 * is monotone: once true it stays true as more bytes arrive, which is what
 * makes it safe to commit bytes to the wire. */
static bool value_is_definitely_string(const char *v, size_t n) {
    if (n == 0) return false;
    unsigned char c = (unsigned char)v[0];
    if (c == '{' || c == '[' || c == '"' || c == '-' ||
        (c >= '0' && c <= '9')) return false;
    if (c == 't' || c == 'f' || c == 'n') {
        const char *lit = c == 't' ? "true" : (c == 'f' ? "false" : "null");
        size_t ll = strlen(lit);
        if (n <= ll && memcmp(v, lit, n) == 0) return false;  /* still a prefix */
    }
    return true;
}

/* Stream the parameter that is still being written. The value is sliced at the
 * last byte that cannot change meaning: trailing whitespace is held back (the
 * extractor trims it) and a half UTF-8 character is held back (the wire is
 * JSON). Everything else is handed to the same extractor the closed path uses,
 * so escaping and key order cannot drift. */
static void tool_std_value_scan(stream_ctx *sc, const char *blk,
                               const char *fn_end) {
    static const char p_open[] = "<parameter=";
    static const char p_close[] = "</parameter>";
    const char *end = fn_end ? fn_end : blk + strlen(blk);

    const char *last_close = NULL;
    for (const char *q = blk; (q = strstr(q, p_close)) != NULL &&
         q < end; q += sizeof(p_close) - 1)
        last_close = q + sizeof(p_close) - 1;
    const char *tail = last_close ? last_close : blk;

    const char *po = NULL;
    for (const char *q = tail; (q = strstr(q, p_open)) != NULL && q < end;
         q += sizeof(p_open) - 1)
        po = q;
    if (!po) return;
    const char *gt = strchr(po, '>');
    if (!gt || gt >= end) return;

    const char *v0 = gt + 1;
    while (v0 < end && (*v0 == '\n' || *v0 == '\r' || *v0 == ' ' || *v0 == '\t'))
        v0++;
    if (v0 >= end) return;

    size_t n = (size_t)(end - v0);
    while (n > 0 && (v0[n - 1] == '\n' || v0[n - 1] == '\r' ||
                     v0[n - 1] == ' ' || v0[n - 1] == '\t'))
        n--;
    /* A closing tag arrives one piece at a time. A tail that could still
     * become `</parameter>` is framing, not value: emitting it would put XML
     * inside the JSON string and the closed-parameter path would then disagree
     * with what is already on the wire. */
    {
        size_t tlen = sizeof(p_close) - 1;
        size_t maxk = n < tlen - 1 ? n : tlen - 1;
        for (size_t k = maxk; k > 0; k--)
            if (memcmp(v0 + n - k, p_close, k) == 0) { n -= k; break; }
    }
    size_t u = utf8_incomplete_tail(v0, n);
    n -= u;
    if (n == 0) return;
    if (!value_is_definitely_string(v0, n)) return;

    /* Each scan re-parses the whole block, so it must not run per token for a
     * large value. Emit in steps: the first byte of a value goes out at once
     * (the client sees the parameter start), then every STEP bytes after it.
     * At 6-7 tok/s a step is a fraction of a second, which reads as live. */
    enum { STEP = 12 };
    if (sc->tool_std_val_len && n < sc->tool_std_val_len + STEP) return;
    sc->tool_std_val_len = n;

    /* The closing quote is emitted by the same extractor as part of `"value"`,
     * so the prefix has to stop one byte short of it. */
    size_t cut = (size_t)(v0 - blk) + n;
    tool_std_emit_prefix(sc, blk, cut, true);
}

/* Close out the streamed call. Returns true when the first call's arguments
 * have gone out as deltas, so the finish must not repeat them. If the final
 * parse disagrees with the bytes already on the wire, those bytes win: the
 * payload is rewritten to the object they describe, which keeps a client that
 * concatenates deltas and one that reads the final object in agreement. */
static bool tool_std_settle(stream_ctx *sc, cwick_tool_call *calls,
                            size_t ncalls) {
    if (!sc->tool_std_on || !sc->tool_std_open || sc->dead || ncalls == 0)
        return false;
    const char *final = calls[0].arguments_json ? calls[0].arguments_json : "{}";
    size_t flen = strlen(final);
    const char *pre = sc->tool_std_prefix ? sc->tool_std_prefix : "";
    if (flen >= sc->tool_std_sent &&
        strncmp(final, pre, sc->tool_std_sent) == 0) {
        if (flen > sc->tool_std_sent)
            tool_std_args_frame(sc, final + sc->tool_std_sent,
                                flen - sc->tool_std_sent);
    } else {
        char *wire = srv_asprintf("%s}", pre);
        free(calls[0].arguments_json);
        calls[0].arguments_json = wire;
        tool_std_args_frame(sc, "}", 1);
        CWICK_INFO(req_tag(sc->gr->rid),
                   "streamed tool arguments did not match the final parse; "
                   "the wire bytes became the payload");
    }
    snprintf(calls[0].id, sizeof(calls[0].id), "%s", sc->tool_std_id);
    sc->tool_std_closed = true;
    return true;
}

/* Once a <tool_call> has opened, cwick_stream_segment() stops classifying:
 * those bytes are parsing material, not content. They are still the only thing
 * happening, and a client waiting on the turn sees nothing at all for as long
 * as the block takes (tens of seconds on a slow checkpoint), so forward them as
 * `cwick_tool` — a non-standard delta field next to `content`, ignored by
 * clients that do not know it — and, in parallel, as the standard opening and
 * argument deltas above.
 *
 * The slice is cleaned of framing before it goes out, and a tail that could
 * still become a frame is held back until the next piece lands. */
static void stream_tool_progress(stream_ctx *sc, bool final) {
    if (sc->dead || sc->raw_len == 0) return;

    /* First call: skip the opener itself, it is the marker that switched us
     * into tool mode and it is not part of what the model is writing. */
    if (sc->tool_sent == 0) {
        static const char open[] = "<tool_call>";
        sc->tool_sent = sc->split.sent + (sizeof(open) - 1);
        if (sc->tool_sent > sc->raw_len) sc->tool_sent = sc->raw_len;
        sc->tool_std_begin = sc->tool_sent;
        sc->tool_std_msg_idx = -1;
        sc->tool_std_resp_idx = -1;
    }

    size_t hold = final ? 0 : cwick_tool_ns_tail(sc->raw, sc->raw_len);
    size_t lim = sc->raw_len > hold ? sc->raw_len - hold : sc->tool_sent;
    if (lim <= sc->tool_sent) return;

    char *clean = cwick_tool_ns_strip(sc->raw + sc->tool_sent,
                                      lim - sc->tool_sent);
    if (clean && *clean)
        stream_delta_n(sc, "cwick_tool", clean, strlen(clean));
    free(clean);
    sc->tool_sent = lim;
    if (!final) tool_std_scan(sc);
}

/* Emit every byte that is now unambiguously classified. With `final` set the
 * held-back tail is emitted too. */
static void stream_flush(stream_ctx *sc, bool final) {
    for (;;) {
        if (sc->dead) return;
        cwick_segment_kind kind = CWICK_SEG_CONTENT;
        size_t off = 0, len = 0;
        if (!cwick_stream_segment(sc->raw, sc->raw_len, final, &sc->split,
                                 &kind, &off, &len))
            break;
        stream_delta(sc, kind == CWICK_SEG_REASONING ? "reasoning_content"
                                                    : stream_content_field(sc),
                     off, len);
    }
    if (sc->split.tool_mode) stream_tool_progress(sc, final);
}

/* One engine token -> bytes appended to the answer -> deltas on the wire. */
static bool stream_sink(void *user, uint32_t token, int64_t index) {
    (void)index;
    stream_ctx *sc = user;
    if (sc->dead) return false;

    /* vLLM's TTFT is "request arrived -> first token on the wire". Here the
     * generation clock starts when the slot is granted, so the queue time is
     * added by the caller; what this records is prefill + first token. */
    if (!sc->first_token_seen) {
        sc->first_token_seen = true;
        sc->first_token_ms = now_ms() - sc->gen_t0_ms;
        if (sc->first_token_ms < 0.0) sc->first_token_ms = 0.0;
        /* vLLM's TTFT starts at request arrival, so the slot wait is part of
         * it. Observed here rather than at completion so the histogram is
         * live even for a stream that runs for minutes. */
        cwick_metrics_observe_ttft((sc->first_token_ms + sc->queue_ms) / 1000.0);
    }

    /* cwick_tokenizer_decode is a pure per-token byte append, so decoding one
     * token at a time is exact (not an approximation of the full decode). */
    cwick_error e;
    cwick_error_clear(&e);
    char *piece = NULL;
    if (cwick_tokenizer_decode(sc->srv->tok, &token, 1, &piece, &e) == CWICK_OK &&
        piece) {
        size_t plen = strlen(piece);
        if (sc->raw_len + plen + 1 > sc->raw_cap) {
            while (sc->raw_len + plen + 1 > sc->raw_cap) sc->raw_cap *= 2;
            sc->raw = cwick_xrealloc(sc->raw, sc->raw_cap);
        }
        memcpy(sc->raw + sc->raw_len, piece, plen);
        sc->raw_len += plen;
        sc->raw[sc->raw_len] = '\0';
        free(piece);
    }
    stream_begin(sc);
    stream_flush(sc, false);
    /* Publish the phase this token landed in (cheap no-op unless it just
     * changed): the engine's per-step decode lines are labelled with it, and
     * a transition is logged once. */
    phase_trk_observe(&sc->pt, &sc->split, index + 1);
    return !sc->dead;
}

static void stream_finish(stream_ctx *sc, const char *finish, bool include_usage,
                          const char *usage, const char *tcalls) {
    stream_begin(sc);            /* 0-token turns still get a valid envelope */
    stream_flush(sc, true);      /* emit the held-back tail */
    if (sc->dead) return;
    if (tcalls) {
        char *json = srv_asprintf(
            "{\"id\":\"%s\",\"object\":\"%s\",\"created\":%lld,\"model\":\"%s\","
            "\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":%s},"
            "\"finish_reason\":null}]}",
            sc->idbuf, sc->gr->is_chat ? "chat.completion.chunk" : "text_completion",
            (long long)sc->created, sc->model, tcalls);
        sse_frame(sc->fd, json);
        free(json);
    }
    char *json = srv_asprintf(
        "{\"id\":\"%s\",\"object\":\"%s\",\"created\":%lld,\"model\":\"%s\","
        "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}]}",
        sc->idbuf, sc->gr->is_chat ? "chat.completion.chunk" : "text_completion",
        (long long)sc->created, sc->model, finish);
    sse_frame(sc->fd, json);
    free(json);
    if (include_usage) {
        char *ujson = srv_asprintf(
            "{\"id\":\"%s\",\"object\":\"%s\",\"created\":%lld,\"model\":\"%s\","
            "\"choices\":[],%s}",
            sc->idbuf, sc->gr->is_chat ? "chat.completion.chunk" : "text_completion",
            (long long)sc->created, sc->model, usage);
        sse_frame(sc->fd, ujson);
        free(ujson);
    }
    static const char done[] = "data: [DONE]\n\n";
    write_all_ok(sc->fd, done, sizeof(done) - 1);
}

/* Close the Responses stream: finish the open items, announce the tool calls
 * the model made, then send the terminal frame carrying the same object the
 * buffered path would have returned. A turn that ran out of budget ends with
 * `response.incomplete`, not `response.completed`: the wire has no other way
 * to say "this answer was cut off". */
static void resp_stream_finish(stream_ctx *sc, const char *resp_body,
                               const cwick_tool_call *calls, size_t ncalls,
                               bool incomplete) {
    stream_begin(sc);            /* 0-token turns still get a valid envelope */
    stream_flush(sc, true);
    if (sc->dead) return;
    resp_close_reason(sc);
    resp_close_text(sc);
    if (sc->resp_msg_id[0] == '\0' && ncalls == 0) {
        /* Symmetry with the buffered body: a textless, toolless turn still
         * ends with a message item. */
        resp_open_text(sc);
        resp_close_text(sc);
    }
    for (size_t i = 0; i < ncalls && !sc->dead; i++) {
        char fid[96];
        snprintf(fid, sizeof(fid), "%s-fc%zu", sc->idbuf, i);
        /* When the first call was streamed, its `added` and argument deltas
         * are already on the wire; only the closing events remain. */
        const bool streamed = (i == 0 && sc->tool_std_closed);
        char *json = NULL;
        if (!streamed) {
            srv_buf item = {0};
            char *jc = cwick_json_stringify(calls[i].id);
            char *jn = cwick_json_stringify(calls[i].name);
            sbuf_putf(&item, "{\"id\":\"%s\",\"type\":\"function_call\","
                            "\"status\":\"in_progress\",\"call_id\":%s,\"name\":%s,"
                            "\"arguments\":\"\"}", fid, jc, jn);
            free(jc);
            free(jn);
            json = srv_asprintf("{\"type\":\"response.output_item.added\","
                                "\"output_index\":%d,\"item\":%s}",
                                sc->resp_index, item.p);
            resp_sse(sc, "response.output_item.added", json);
            free(json);
            free(item.p);
        }
        const char *args = (calls[i].arguments_json && *calls[i].arguments_json)
                               ? calls[i].arguments_json : "{}";
        char *ja = cwick_json_stringify(args);
        if (!streamed) {
            json = srv_asprintf("{\"type\":\"response.function_call_arguments.delta\","
                                "\"item_id\":\"%s\",\"output_index\":%d,\"delta\":%s}",
                                fid, sc->resp_index, ja);
            resp_sse(sc, "response.function_call_arguments.delta", json);
            free(json);
        }
        json = srv_asprintf("{\"type\":\"response.function_call_arguments.done\","
                            "\"item_id\":\"%s\",\"output_index\":%d,"
                            "\"arguments\":%s}", fid, sc->resp_index, ja);
        resp_sse(sc, "response.function_call_arguments.done", json);
        free(json);
        free(ja);
        srv_buf done = {0};
        resp_put_call_item(&done, fid, calls[i].id, calls[i].name,
                           calls[i].arguments_json);
        json = srv_asprintf("{\"type\":\"response.output_item.done\","
                            "\"output_index\":%d,\"item\":%s}",
                            sc->resp_index, done.p);
        resp_sse(sc, "response.output_item.done", json);
        free(json);
        free(done.p);
        sc->resp_index++;
    }
    if (!sc->dead && resp_body) {
        const char *ev = incomplete ? "response.incomplete" : "response.completed";
        char *json = srv_asprintf("{\"type\":\"%s\",\"response\":%s}", ev, resp_body);
        resp_sse(sc, ev, json);
        free(json);
    }
}

/* Close an Anthropic stream: finish the open blocks, announce the calls the
 * model made as `tool_use` blocks, then the two terminal events. Anthropic
 * has no [DONE]: `message_stop` is the end, and `message_delta` carries the
 * only place a streamed response reports its stop reason and output count. */
static void msg_stream_finish(stream_ctx *sc, const cwick_tool_call *calls,
                              size_t ncalls, const char *stop_reason,
                              int64_t out_tokens) {
    stream_begin(sc);            /* 0-token turns still get a valid envelope */
    stream_flush(sc, true);
    if (sc->dead) return;
    msg_close_think(sc);
    msg_close_text(sc);
    for (size_t i = 0; i < ncalls && !sc->dead; i++) {
        /* The streamed call already owns its content_block index; do not
         * allocate a second one or repeat its input_json_delta. */
        const bool streamed = (i == 0 && sc->tool_std_closed);
        int idx = streamed ? sc->tool_std_msg_idx : sc->msg_index++;
        if (!streamed) {
            char *jc = cwick_json_stringify(calls[i].id);
            char *jn = cwick_json_stringify(calls[i].name);
            char *json = srv_asprintf(
                "{\"type\":\"content_block_start\",\"index\":%d,\"content_block\":"
                "{\"type\":\"tool_use\",\"id\":%s,\"name\":%s,\"input\":{}}}",
                idx, jc, jn);
            msg_sse(sc, "content_block_start", json);
            free(json);
            free(jc);
            free(jn);
            const char *args = (calls[i].arguments_json && *calls[i].arguments_json)
                                   ? calls[i].arguments_json : "{}";
            char *ja = cwick_json_stringify(args);
            json = srv_asprintf(
                "{\"type\":\"content_block_delta\",\"index\":%d,\"delta\":"
                "{\"type\":\"input_json_delta\",\"partial_json\":%s}}", idx, ja);
            msg_sse(sc, "content_block_delta", json);
            free(json);
            free(ja);
        }
        msg_block_stop(sc, idx);
    }
    if (sc->dead) return;
    /* `message_delta` is the only place a streamed turn reports its final
     * counts. Anthropic repeats the input side here (and the cache read, when
     * a prefix hit covered part of the prompt), so a client that only reads
     * the closing events still gets a complete usage object. */
    char cache[64] = "";
    if (sc->cached_tokens > 0)
        snprintf(cache, sizeof(cache), ",\"cache_read_input_tokens\":%lld",
                 (long long)sc->cached_tokens);
    char *json = srv_asprintf(
        "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"%s\","
        "\"stop_sequence\":null},\"usage\":{\"output_tokens\":%lld,"
        "\"input_tokens\":%lld%s}}",
        stop_reason, (long long)out_tokens, (long long)sc->in_tokens, cache);
    msg_sse(sc, "message_delta", json);
    free(json);
    msg_sse(sc, "message_stop", "{\"type\":\"message_stop\"}");
}

/* A failure after the head is on the wire cannot become an HTTP status; report
 * it in-band and close the stream cleanly. */
static void stream_error(stream_ctx *sc, const char *message) {
    char *esc = cwick_json_stringify(message ? message : "generation failed");
    if (sc->gr->api == SRV_API_RESPONSES) {
        /* A response that has already been announced cannot become an HTTP
         * status, and the Responses stream has no [DONE] sentinel: the
         * terminal event ends it. For a response that exists but did not
         * finish, that event is `response.failed` — the same object as
         * `response.completed`, with status "failed" and `error` filled in.
         * (The bare `error` event is for a failure before a response is
         * created, which this server reports as an HTTP error instead.) */
        char *json = srv_asprintf(
            "{\"type\":\"response.failed\",\"response\":{\"id\":\"%s\","
            "\"object\":\"response\",\"created_at\":%lld,\"status\":\"failed\","
            "\"model\":\"%s\",\"output\":[],\"usage\":null,"
            "\"error\":{\"code\":\"server_error\",\"message\":%s},"
            "\"incomplete_details\":null%s}}",
            sc->idbuf, (long long)sc->created, sc->model, esc,
            sc->gr->resp_tail ? sc->gr->resp_tail : "");
        resp_sse(sc, "response.failed", json);
        free(json);
        free(esc);
        return;
    }
    if (sc->gr->api == SRV_API_MESSAGES) {
        /* Anthropic wraps the failure in an `error` event whose payload is
         * the same object a non-streaming call would have returned. */
        char *json = srv_asprintf(
            "{\"type\":\"error\",\"error\":{\"type\":\"api_error\","
            "\"message\":%s}}", esc);
        msg_sse(sc, "error", json);
        free(json);
        free(esc);
        return;
    }
    char *json = srv_asprintf(
        "{\"error\":{\"message\":%s,\"type\":\"server_error\","
        "\"param\":null,\"code\":null}}", esc);
    free(esc);
    sse_frame(sc->fd, json);
    free(json);
    static const char done[] = "data: [DONE]\n\n";
    write_all_ok(sc->fd, done, sizeof(done) - 1);
}

/* A post-generation failure (malformed tool call, unmet tool_choice, ...) can
 * surface after the SSE head and some deltas are already on the wire. Report it
 * in-band and swallow the error, so the caller does not append an HTTP error
 * body to a live stream. Returns true when it reported the error itself. */
static bool stream_report_error(stream_ctx *sc, bool streaming, cwick_error *err) {
    if (!streaming || !sc->started) return false;
    stream_error(sc, cwick_error_message(err));
    cwick_error_clear(err);
    return true;
}

/* True while the peer is still there.
 *
 * Generation for one request runs to completion before a single byte is
 * written, so a client that hangs up mid-generation is otherwise undetectable:
 * the loop keeps decoding, holding the generation slot for the whole budget
 * (8192 tokens at ~1.3 tok/s is ~106 minutes of 429s for everyone else).
 *
 * The probe is a one-byte MSG_PEEK, which never consumes anything:
 *   > 0  bytes pending  -> peer sent something (a pipelined request); alive
 *   = 0  orderly close  -> FIN received; gone
 *   < 0  EAGAIN         -> nothing to read, connection still open; alive
 *        ECONNRESET etc -> gone
 * The pending-data case must NOT count as a disconnect: a keep-alive client
 * that already pipelined the next request would otherwise be killed. */
static bool client_socket_alive(int fd) {
    if (fd < 0) return true;
    char probe;
    ssize_t n = recv(fd, &probe, 1, MSG_PEEK | MSG_DONTWAIT);
    if (n > 0) return true;
    if (n == 0) return false;                 /* FIN received */
    /* Err on the side of "still here". Only a reset/dead pipe counts as gone;
     * anything unexpected (EBADF from a slot whose fd was already closed,
     * EINTR, ...) must not abort a request that is perfectly fine. */
    return !(errno == ECONNRESET || errno == EPIPE);
}

/* Engine-side predicate: it polls the socket of the request being served. */
static bool client_gone_poll(void *user) {
    const int *fdp = (const int *)user;
    return fdp ? !client_socket_alive(*fdp) : false;
}

/* Hand the slot back and let the oldest waiter through. */
static void slot_release(server_ctx *srv, int slot) {
    pthread_mutex_lock(&srv->gen_lock);
    srv->slot_busy[slot] = false;
    srv->slot_fd[slot] = -1;
    srv->ticket_served++;
    pthread_cond_broadcast(&srv->gen_cv);
    pthread_mutex_unlock(&srv->gen_lock);
}

/* Wait for a generation slot. Returns the slot index, or -1 when the request
 * has to be turned away: the queue is full, or the server is shutting down
 * and this request never got a turn. `wait_ms_out` gets the time spent queued. */
/* Queue for a generation slot in arrival order.
 *
 * The window is [ticket_served, ticket_served + nslots); a ticket may take a
 * slot once it falls inside it. Tickets are handed out under gen_lock, so the
 * window always advances to the oldest waiting request -- with one slot that
 * means strict FIFO, and a request is never refused just because the slot is
 * busy.
 *
 * Returns the slot index, or -1 when the request cannot be served: the backlog
 * is full, or shutdown began before its turn came. `wait_ms_out` gets the time
 * spent queued. */
static int slot_acquire(server_ctx *srv, int fd, volatile int *stop,
                        double *wait_ms_out) {
    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    pthread_mutex_lock(&srv->gen_lock);
    const uint64_t mine = srv->ticket_next++;
    const uint64_t width = (uint64_t)(srv->nslots > 0 ? srv->nslots : 1);
    srv->nwaiting++;
    int slot = -1;

    while (!(stop && *stop)) {
        /* A model swap frees the engine and the tokenizer out from under
         * anyone holding them, so nothing new is admitted while one runs. -3
         * is distinct from -1 (no capacity) so the caller can say why. */
        if (srv->swapping) { slot = -3; break; }
        if (mine - srv->ticket_served < width) {
            for (int i = 0; i < CWICK_GEN_SLOTS; i++) {
                if (!srv->slot_busy[i] && srv->eng[i]) {
                    srv->slot_busy[i] = true;
                    srv->slot_fd[i] = fd;  /* the engine polls this socket */
                    slot = i;
                    break;
                }
            }
            if (slot >= 0) break;
        }
        /* Timed rather than plain wait: a holder that never releases must not
         * hold up shutdown. */
        struct timespec deadline;
        clock_gettime(CLOCK_MONOTONIC, &deadline);
        deadline.tv_sec += 1;
        pthread_cond_timedwait(&srv->gen_cv, &srv->gen_lock, &deadline);
        /* A client that hung up while queued is not worth serving: its
         * answer would be written to a dead socket after an hour-plus
         * prefill, and it keeps a slot from the requests behind it. The
         * probe is the same MSG_PEEK the engine-side abort uses; only an
         * orderly close or a reset counts as gone, so a keep-alive peer
         * that pipelined its next request stays queued. This can never
         * interrupt a running generation -- it only drops what nobody is
         * waiting for any more. */
        if (!(stop && *stop) && !client_socket_alive(fd)) {
            srv->nwaiting--;
            pthread_mutex_unlock(&srv->gen_lock);
            if (wait_ms_out) {
                struct timespec t1;
                clock_gettime(CLOCK_MONOTONIC, &t1);
                double ms = (double)(t1.tv_sec - t0.tv_sec) * 1e3 +
                            (double)(t1.tv_nsec - t0.tv_nsec) / 1e6;
                *wait_ms_out = ms < 0.0 ? 0.0 : ms;
            }
            return -2;
        }
    }

    srv->nwaiting--;
    pthread_mutex_unlock(&srv->gen_lock);

    if (wait_ms_out) {
        struct timespec t1;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (double)(t1.tv_sec - t0.tv_sec) * 1e3 +
                    (double)(t1.tv_nsec - t0.tv_nsec) / 1e6;
        *wait_ms_out = ms < 0.0 ? 0.0 : ms;
    }
    return slot;
}

/* `slot` is the generation slot this request was granted; `queue_ms` is how
 * long it waited for one. Several slots may be generating at once.
 * `res_out` receives the engine's own timings (ownership of `tokens` moves to
 * the caller, which must call cwick_gen_result_free); `ttft_ms_out` receives
 * the time to the first streamed token (0 for a non-streaming request, where
 * nothing is observable before the body). */
/* One snapshot waiting to be written. Owns its state and its token sequence:
 * both outlive the request that produced them, which is the whole point --
 * the prompt buffer the capture was keyed on belongs to a connection thread
 * that is about to return. */
struct pfx_job {
    struct pfx_job *next;
    cwick_prefix_state st;
    uint32_t *ids;
    int64_t P;
    char tag[32];
};

/* Two is enough: one being written, one waiting. A third means the writer is
 * far behind (a slow disk, or a burst of long prompts) and the oldest
 * snapshot is dropped rather than let the queue grow into the memory the
 * weights need -- a dropped snapshot costs one future prefill, which is
 * exactly what the cache is there to avoid, but never more than that. */
#define PFX_MAX_QUEUED 2

static void pfx_job_free(struct pfx_job *j) {
    if (!j) return;
    cwick_prefix_state_free(&j->st);
    free(j->ids);
    free(j);
}

static void pfx_write_sync(server_ctx *srv, struct pfx_job *j) {
    cwick_error e;
    cwick_error_clear(&e);
    double t0 = now_ms();
    pthread_mutex_lock(&srv->pfx_mtx);
    cwick_status s = cwick_pfx_store(srv->pfx, j->ids, j->P, &j->st, &e);
    pthread_mutex_unlock(&srv->pfx_mtx);
    if (s == CWICK_OK)
        CWICK_INFO(j->tag, "prefix cached: %lld tokens (%.0f ms, background)",
                   (long long)j->P, now_ms() - t0);
    else
        CWICK_VERB(j->tag, "prefix store failed: %s", cwick_error_message(&e));
}

static void *pfx_writer_main(void *arg) {
    server_ctx *srv = (server_ctx *)arg;
    for (;;) {
        struct pfx_job *j = NULL;
        pthread_mutex_lock(&srv->pfxw.mtx);
        while (!srv->pfxw.head && !srv->pfxw.stop)
            pthread_cond_wait(&srv->pfxw.cv, &srv->pfxw.mtx);
        if (srv->pfxw.head) {
            j = srv->pfxw.head;
            srv->pfxw.head = j->next;
            if (!srv->pfxw.head) srv->pfxw.tail = NULL;
            srv->pfxw.queued--;
        }
        const bool stopping = srv->pfxw.stop;
        pthread_mutex_unlock(&srv->pfxw.mtx);
        if (j) pfx_write_sync(srv, j);
        pfx_job_free(j);
        if (!j && stopping) break;
    }
    return NULL;
}

/* Hand a captured snapshot to the writer. Takes ownership of `st`. */
static void pfx_submit(server_ctx *srv, cwick_prefix_state *st,
                       const uint32_t *ids, int64_t P, const char *tag) {
    if (!st || !ids || P <= 0) { cwick_prefix_state_free(st); return; }
    if (!srv->pfx || !srv->pfxw.started) {
        /* No writer: fall back to a synchronous store so a cache is never
         * silently lost just because a thread could not be started. */
        struct pfx_job j;
        memset(&j, 0, sizeof(j));
        j.st = *st;
        j.ids = (uint32_t *)ids;   /* borrowed: the caller still owns it here */
        j.P = P;
        snprintf(j.tag, sizeof(j.tag), "%s", tag ? tag : "pfx");
        pfx_write_sync(srv, &j);
        cwick_prefix_state_free(&j.st);
        memset(st, 0, sizeof(*st));
        return;
    }
    struct pfx_job *j = calloc(1, sizeof(*j));
    if (!j) { cwick_prefix_state_free(st); return; }
    j->st = *st;
    j->P = P;
    snprintf(j->tag, sizeof(j->tag), "%s", tag ? tag : "pfx");
    j->ids = cwick_xmalloc((size_t)P * sizeof(uint32_t));
    if (!j->ids) { pfx_job_free(j); memset(st, 0, sizeof(*st)); return; }
    memcpy(j->ids, ids, (size_t)P * sizeof(uint32_t));
    memset(st, 0, sizeof(*st));   /* ownership moved into the job */

    bool dropped = false;
    pthread_mutex_lock(&srv->pfxw.mtx);
    if (srv->pfxw.queued >= PFX_MAX_QUEUED) {
        struct pfx_job *old = srv->pfxw.head;
        srv->pfxw.head = old->next;
        if (!srv->pfxw.head) srv->pfxw.tail = NULL;
        srv->pfxw.queued--;
        pfx_job_free(old);
        dropped = true;
    }
    j->next = NULL;
    if (srv->pfxw.tail) srv->pfxw.tail->next = j; else srv->pfxw.head = j;
    srv->pfxw.tail = j;
    srv->pfxw.queued++;
    pthread_cond_signal(&srv->pfxw.cv);
    pthread_mutex_unlock(&srv->pfxw.mtx);
    if (dropped)
        CWICK_VERB(j->tag, "prefix cache writer busy — dropped the oldest "
                   "pending snapshot of %lld tokens", (long long)j->P);
}

/* Start / stop the writer alongside the cache. */
static void pfx_writer_start(server_ctx *srv) {
    memset(&srv->pfxw, 0, sizeof(srv->pfxw));
    if (!srv->pfx) return;
    pthread_mutex_init(&srv->pfxw.mtx, NULL);
    pthread_cond_init(&srv->pfxw.cv, NULL);
    if (pthread_create(&srv->pfxw.th, NULL, pfx_writer_main, srv) == 0)
        srv->pfxw.started = true;
}

static void pfx_writer_stop(server_ctx *srv) {
    if (!srv->pfxw.started) return;
    pthread_mutex_lock(&srv->pfxw.mtx);
    srv->pfxw.stop = true;
    pthread_cond_signal(&srv->pfxw.cv);
    pthread_mutex_unlock(&srv->pfxw.mtx);
    pthread_join(srv->pfxw.th, NULL);
    while (srv->pfxw.head) {          /* anything still queued */
        struct pfx_job *j = srv->pfxw.head;
        srv->pfxw.head = j->next;
        pfx_job_free(j);
    }
    srv->pfxw.started = false;
}

/* Everything the prefix-store callback needs when the engine fires it. Lives
 * on run_generation's stack for exactly as long as the generation that can
 * fire it: the callback runs between the request's prefill and its first
 * decode step, and `ids` below is the request's own prompt buffer. */
typedef struct {
    server_ctx *srv;
    cwick_engine *eng;      /* the live engine: to capture one token short */
    const uint32_t *ids;    /* the whole prompt, as sent */
    int64_t nids;
    char tag[32];           /* req_tag() is thread-local: copy, never borrow */
} pfx_store_ctx;

/* Fired by the engine the instant the post-prefill snapshot is filled. The
 * write happens HERE rather than when generation returns: the prefill that
 * produced this snapshot cost minutes to hours, and anything that ends the
 * request in between (abort, client disconnect, a kill of the process)
 * would otherwise throw the snapshot away -- the whole point of the cache is
 * that the NEXT request with the same prompt never pays for this prefill
 * again. Takes ownership of `s`. */
static void pfx_save_now(void *user, cwick_prefix_state *s) {
    pfx_store_ctx *pc = (pfx_store_ctx *)user;
    if (s && s->pos > 0) {
        /* Stored under the snapshot's own length, not the request's: this
         * callback also fires at chunk boundaries during a long prefill, where
         * s->pos is only part of the prompt. */
        const int64_t P = s->pos;
        if (P >= CWICK_PFX_MIN_TOKENS && P <= pc->nids) {
            /* Handed to the writer thread rather than written here: this runs
             * between the prefill and the first decode step, and a synchronous
             * write of a large snapshot used to add seconds to the request and
             * displace the resident weights (see pfx_submit). The snapshot is
             * already captured, so nothing the request needs is at stake. */
            pfx_submit(pc->srv, s, pc->ids, P, pc->tag);
            s = NULL;   /* ownership moved */

            /* And one boundary shorter, keyed on the tokens up to the point
             * where the next turn will actually agree with this one.
             *
             * A prompt that opens a thinking block ends with the block itself:
             * the open marker on its own when thinking is on, or a pre-closed
             * empty block (`<think>\n\n</think>\n\n`) when the server was asked
             * to disable it. The next turn of that session renders the
             * assistant's reply as the header plus its content -- it never
             * resends the injected block -- so the two prompts agree up to the
             * FIRST token of that block and not one token further.
             *
             * Thinking on: the block is the single open marker, so P - 1 is
             * exactly the boundary, which is why that twin was enough for the
             * 262-vs-405 case measured earlier. Thinking off: the injected
             * block is four tokens on Qwen3.5, so P - 1 was still three too
             * long. Measured on a live session, turn N was 232 tokens and turn
             * N+1 was 265, they agreed on the first 228, the cache held 231 and
             * 232 -- the lookup missed and the turn fell back to the 4K block
             * checkpoint, re-prefilling 37 tokens at 30K-context rates.
             *
             * So: search the tail for the open marker and key the twin there.
             * Falls back to P - 1 when there is none, which is the
             * thinking-on case. */
            if (pc->eng && P == pc->nids && P >= 2) {
                int64_t twin = P - 1;
                if (pc->srv && pc->srv->think_open_id >= 0) {
                    int64_t from = P - 1 - 8;
                    if (from < 0) from = 0;
                    for (int64_t k = P - 1; k >= from; k--)
                        if ((int64_t)pc->ids[k] == pc->srv->think_open_id) {
                            twin = k;
                            break;
                        }
                }
                if (twin < 2) twin = P - 1;
                cwick_error e;
                cwick_error_clear(&e);
                cwick_prefix_state *sh = calloc(1, sizeof(*sh));
                if (sh) {
                    if (cwick_engine_capture_prefix(pc->eng, twin, sh, &e) ==
                        CWICK_OK)
                        pfx_submit(pc->srv, sh, pc->ids, twin, pc->tag);
                    else
                        CWICK_VERB(pc->tag, "prefix capture at %lld tokens "
                                  "failed: %s", (long long)twin,
                                  cwick_error_message(&e));
                    cwick_prefix_state_free(sh);
                    free(sh);
                }
            }
        }
    }
    cwick_prefix_state_free(s);
}

/* Snapshot the conversation as it stands when generation returns.
 *
 * The prefill-time snapshot above stops at the prompt, which is not where a
 * conversation stops: the next request's prompt is this prompt *followed by*
 * what was just generated, and that is exactly the state the engine is holding
 * here. Storing only the prefill-time one means every later turn of a session
 * falls back to the last block checkpoint below its own length and re-prefills
 * the rest from there. Measured on a 9828-token prompt whose next turn added
 * seven tokens: 8192 replayed / 1643 re-prefilled (19.6 s of prefill) before
 * this existed, the whole 9835 replayed after it.
 *
 * The key is the prompt tokens followed by the generated ones. A client that
 * re-tokenises the assistant's reply differently simply misses -- the stored
 * tokens are compared before anything is applied, so a mismatch costs one
 * lookup and never a wrong generation. That is also why this runs even when
 * the prompt itself was already cached: what is new here is the tail. */
static void pfx_save_generated(server_ctx *srv, cwick_engine *eng,
                               const uint32_t *ids, int64_t nids,
                               const uint32_t *gen, int64_t ngen,
                               const char *tag) {
    if (!srv->pfx || !ids || !gen || nids <= 0 || ngen <= 0) return;
    /* The key length is the engine's own position, not nids + ngen: the
     * prefill's logits already predict the first generated token, so a
     * generation of n tokens runs n-1 forwards and the last token emitted has
     * no row of its own yet. Keying on nids + ngen would therefore ask for one
     * position more than the state holds, and the snapshot would be rejected
     * (or worse, describe a token nobody sent). */
    const int64_t total = cwick_engine_pos(eng);
    const int64_t gen_in = total - nids;
    if (gen_in <= 0 || gen_in > ngen) return;
    if (total < CWICK_PFX_MIN_TOKENS) return;

    uint32_t *all = cwick_xmalloc((size_t)total * sizeof(uint32_t));
    if (!all) return;
    memcpy(all, ids, (size_t)nids * sizeof(uint32_t));
    memcpy(all + nids, gen, (size_t)gen_in * sizeof(uint32_t));

    cwick_prefix_state st;
    memset(&st, 0, sizeof(st));
    cwick_error e;
    cwick_error_clear(&e);
    double t0 = now_ms();
    if (cwick_engine_capture_prefix(eng, total, &st, &e) == CWICK_OK && st.pos > 0) {
        pfx_submit(srv, &st, all, total, tag);
        CWICK_INFO(tag, "conversation snapshot queued: %lld tokens "
                   "(prompt %lld + %lld of %lld generated, %.0f ms to capture) "
                   "— the next turn starts with exactly these, so it can replay "
                   "the whole conversation", (long long)total,
                   (long long)nids, (long long)gen_in, (long long)ngen,
                   now_ms() - t0);
    } else {
        CWICK_VERB(tag, "prefix capture after generation failed: %s",
                   cwick_error_message(&e));
    }
    free(all);
}

/* Buffered (non-streaming) requests have no consumer to split the text, but
 * the decode still deserves phase-labelled progress lines and transition
 * records: this sink classifies the tokens with the same state machine and
 * publishes the label to the engine, without writing anything to a socket. */
typedef struct {
    phase_trk pt;             /* label bookkeeping (eng/tag/phase/t0) */
    cwick_stream_state split;  /* the classifier this sink advances */
    const cwick_tokenizer *tok;
    char *buf;                /* detokenized bytes so far */
    size_t len, cap;
} phase_tracker;

static bool phase_track_sink(void *user, uint32_t token, int64_t index) {
    phase_tracker *ptk = (phase_tracker *)user;
    if (!ptk) return true;
    cwick_error e;
    cwick_error_clear(&e);
    char *piece = NULL;
    if (cwick_tokenizer_decode(ptk->tok, &token, 1, &piece, &e) == CWICK_OK &&
        piece) {
        size_t plen = strlen(piece);
        while (ptk->len + plen + 1 > ptk->cap) {
            ptk->cap *= 2;
            ptk->buf = cwick_xrealloc(ptk->buf, ptk->cap);
        }
        memcpy(ptk->buf + ptk->len, piece, plen);
        ptk->len += plen;
        ptk->buf[ptk->len] = '\0';
        free(piece);
        for (;;) {
            cwick_segment_kind kind = CWICK_SEG_CONTENT;
            size_t off = 0, slen = 0;
            if (!cwick_stream_segment(ptk->buf, ptk->len, false, &ptk->split,
                                     &kind, &off, &slen))
                break;
        }
    }
    phase_trk_observe(&ptk->pt, &ptk->split, index + 1);
    return true;   /* never stops the engine: this path writes no bytes */
}

/* ------------------------------------------------------------------ */
/* Chat rendering dispatch                                             */
/*                                                                     */
/* GLM's format lives in tokenizer.c; DeepSeek-V4.1 ships its own in    */
/* `encoding/encoding.py`, ported to src/ds_prompt.c. Which one applies */
/* is a property of the resident checkpoint, not of the request, so the */
/* decision is read from model_type on every request -- /v1/models/     */
/* switch can change it between two requests.                          */
/* ------------------------------------------------------------------ */

static bool model_uses_ds_template(const cwick_engine *eng) {
    const cwick_model_spec *spec = eng ? cwick_engine_spec(eng) : NULL;
    return spec && strncmp(spec->model_type, "deepseek", 8) == 0;
}

static bool model_uses_mm_template(const cwick_engine *eng) {
    const cwick_model_spec *spec = eng ? cwick_engine_spec(eng) : NULL;
    return spec && strncmp(spec->model_type, "minimax_m3", 10) == 0;
}

static bool model_uses_q35_template(const cwick_engine *eng) {
    const cwick_model_spec *spec = eng ? cwick_engine_spec(eng) : NULL;
    return spec && strncmp(spec->model_type, "qwen3_5", 7) == 0;
}

/* Qwen3.5 renders with its own template. The thinking switch has no numeric
 * knob there either: "off" is the assistant header followed by an empty, closed
 * reasoning block, which is what the template itself writes. */
static cwick_status q35_chat_encode(const cwick_tokenizer *tok,
                                    const char *messages_json,
                                    const char *tools_json,
                                    const char *effort_level,
                                    uint32_t **ids, int64_t *n_out,
                                    cwick_error *err) {
    cwick_error pe;
    cwick_error_clear(&pe);
    json_value *msgs = cwick_json_parse(messages_json, strlen(messages_json), &pe);
    if (!msgs) {
        cwick_error_set(err, CWICK_ERR_PARSE, "messages: %s",
                        cwick_error_message(&pe));
        return CWICK_ERR_PARSE;
    }
    json_value *tools = NULL;
    if (tools_json) {
        cwick_error_clear(&pe);
        tools = cwick_json_parse(tools_json, strlen(tools_json), &pe);
        if (!tools) {
            cwick_json_free(msgs);
            cwick_error_set(err, CWICK_ERR_PARSE, "tools: %s",
                            cwick_error_message(&pe));
            return CWICK_ERR_PARSE;
        }
    }

    q35_prompt_opts opts;
    q35_prompt_opts_default(&opts);
    const think_level *tl = effort_level ? think_level_find(effort_level) : NULL;
    opts.thinking = tl ? tl->thinks : true;

    cwick_error_clear(&pe);
    char *text = q35_prompt_encode(msgs, tools, &opts, NULL, &pe);
    cwick_json_free(msgs);
    if (tools) cwick_json_free(tools);
    if (!text) {
        cwick_error_set(err, CWICK_ERR_PARSE, "qwen3_5 prompt: %s",
                        cwick_error_message(&pe));
        return CWICK_ERR_PARSE;
    }
    cwick_status st = cwick_tokenizer_encode(tok, text, ids, n_out, err);
    free(text);
    return st;
}

/* MiniMax-M3.1's template has no numeric effort knob, so the "关" level is
 * expressed the way the template itself expresses it: the assistant header is
 * followed by `</mm:think>` instead of a reasoning block. */
static cwick_status mm_chat_encode(const cwick_tokenizer *tok,
                                   const char *messages_json,
                                   const char *tools_json,
                                   const char *effort_level,
                                   uint32_t **ids, int64_t *n_out,
                                   cwick_error *err) {
    cwick_error pe;
    cwick_error_clear(&pe);
    json_value *msgs = cwick_json_parse(messages_json, strlen(messages_json), &pe);
    if (!msgs) {
        cwick_error_set(err, CWICK_ERR_PARSE, "messages: %s",
                        cwick_error_message(&pe));
        return CWICK_ERR_PARSE;
    }
    json_value *tools = NULL;
    if (tools_json) {
        cwick_error_clear(&pe);
        tools = cwick_json_parse(tools_json, strlen(tools_json), &pe);
        if (!tools) {
            cwick_json_free(msgs);
            cwick_error_set(err, CWICK_ERR_PARSE, "tools: %s",
                            cwick_error_message(&pe));
            return CWICK_ERR_PARSE;
        }
    }

    mm_prompt_opts opts;
    mm_prompt_opts_default(&opts);
    const think_level *tl = effort_level ? think_level_find(effort_level) : NULL;
    opts.thinking = tl ? tl->thinks : true;

    cwick_error_clear(&pe);
    char *text = mm_prompt_encode(msgs, tools, &opts, NULL, &pe);
    cwick_json_free(msgs);
    if (tools) cwick_json_free(tools);
    if (!text) {
        cwick_error_set(err, CWICK_ERR_PARSE, "minimax prompt: %s",
                        cwick_error_message(&pe));
        return CWICK_ERR_PARSE;
    }
    cwick_status st = cwick_tokenizer_encode(tok, text, ids, n_out, err);
    free(text);
    return st;
}

/* The same THINK_LEVELS row, seen from the DeepSeek renderer: a number in
 * 1..100 for its "Reasoning Effort: N" header, plus whether to open a block
 * at all. DeepSeek's chat mode is a different thing from a small budget --
 * there the assistant header is followed by `</think>` and the model answers
 * without reasoning, which is exactly `关`. Returning 0 leaves the renderer's
 * own default (75) in place. */
static int ds_effort_from_level(const char *level, bool *thinking) {
    const think_level *tl = think_level_find(level);
    *thinking = tl ? tl->thinks : true;
    return tl ? tl->ds_effort : 0;
}

/* Render the conversation with the DeepSeek template and tokenize it. The
 * two JSON strings are the request's own arrays, kept verbatim -- the
 * DeepSeek template reads fields (reasoning_content, task, wo_eos) that the
 * flattened cwick_chat_msg struct does not carry. */
static cwick_status ds_chat_encode(const cwick_tokenizer *tok,
                                   const char *messages_json,
                                   const char *tools_json,
                                   const char *effort_level,
                                   uint32_t **ids, int64_t *n_out,
                                   cwick_error *err) {
    cwick_error pe;
    cwick_error_clear(&pe);
    json_value *msgs = cwick_json_parse(messages_json, strlen(messages_json), &pe);
    if (!msgs) {
        cwick_error_set(err, CWICK_ERR_PARSE, "messages: %s",
                        cwick_error_message(&pe));
        return CWICK_ERR_PARSE;
    }
    json_value *tools = NULL;
    if (tools_json) {
        cwick_error_clear(&pe);
        tools = cwick_json_parse(tools_json, strlen(tools_json), &pe);
        if (!tools) {
            cwick_json_free(msgs);
            cwick_error_set(err, CWICK_ERR_PARSE, "tools: %s",
                            cwick_error_message(&pe));
            return CWICK_ERR_PARSE;
        }
    }

    ds_prompt_opts opts;
    ds_prompt_opts_default(&opts);
    bool thinking = true;
    opts.reasoning_effort = ds_effort_from_level(effort_level, &thinking);
    opts.thinking = thinking ? DS_THINKING_THINKING : DS_THINKING_CHAT;
    /* An earlier turn's reasoning is only replayed while the conversation is
     * tool-free; once tools are declared the reference drops it, so the model
     * reads its own calls as the record of what happened. */
    if (tools) opts.drop_thinking = false;

    cwick_error_clear(&pe);
    char *text = ds_prompt_encode(msgs, tools, &opts, NULL, &pe);
    cwick_json_free(msgs);
    if (tools) cwick_json_free(tools);
    if (!text) {
        cwick_error_set(err, CWICK_ERR_PARSE, "deepseek prompt: %s",
                        cwick_error_message(&pe));
        return CWICK_ERR_PARSE;
    }
    cwick_status st = cwick_tokenizer_encode(tok, text, ids, n_out, err);
    free(text);
    return st;
}

/* Hand the engine the (checkpoint default + request override) sampling config
 * for this generation, and say what it ended up being — "the answer was
 * degenerate" is only diagnosable next to "the answer was drawn with t=1.0".
 *
 * greedy stays reachable in two ways, both deliberate: a request that sends
 * temperature 0 (the bundled web app does, and it wants one deterministic
 * answer), and a checkpoint with no generation_config.json or with
 * do_sample=false. */
static void apply_sampling(cwick_engine *eng, const gen_request *gr) {
    cwick_sampler_cfg cfg;
    bool from_ckpt = cwick_engine_sampler_checkpoint(eng, &cfg);
    if (!from_ckpt) cwick_sampler_cfg_default(&cfg);
    if (gr->has_temperature) cfg.temperature = gr->smp.temperature;
    if (gr->has_top_p) cfg.top_p = gr->smp.top_p;
    if (gr->has_top_k) cfg.top_k = gr->smp.top_k;
    if (gr->has_presence) cfg.presence_penalty = gr->smp.presence_penalty;
    if (gr->has_frequency) cfg.frequency_penalty = gr->smp.frequency_penalty;
    /* validate_body already refused out-of-range values; this is the belt to
     * that braces (a config assembled from two sources is a new config). */
    if (!cwick_sampler_cfg_validate(&cfg, NULL, 0)) {
        cwick_sampler_cfg_default(&cfg);
        from_ckpt = false;
    }
    cwick_engine_set_sampler(eng, &cfg);

    char sd[96];
    cwick_sampler_describe(&cfg, sd, sizeof(sd));
    CWICK_INFO(req_tag(gr->rid), "sampling=%s (%s)", sd,
               from_ckpt ? "generation_config.json" : "greedy default");
}

static void run_generation(server_ctx *srv, gen_request *gr, int fd,
                           char *detail, size_t detail_cap, int slot,
                           double queue_ms, cwick_gen_result *res_out,
                           double *ttft_ms_out, cwick_error *err) {
    cwick_engine *eng = srv->eng[slot];
    uint32_t *ids = NULL;
    int64_t nids = 0;
    cwick_status st;
    double gen_t0 = now_ms();

    /* The slot is granted: this request stops being "waiting". */
    cwick_metrics_request_admit(gr->rid, queue_ms / 1000.0);

    trace_request(srv, gr);

    if (gr->is_chat && gr->messages_json && model_uses_ds_template(eng)) {
        /* The no-field default is a real level (关), so it takes exactly the
         * same path as an explicit reasoning_effort:none. An absent field
         * used to be passed down as "unset" and each renderer fell back to
         * its own checkpoint default, which opened a reasoning block that
         * the engine's zero budget then had to close again. */
        st = ds_chat_encode(srv->tok, gr->messages_json, gr->tools_json,
                            gr->effort_level, &ids, &nids, err);
    } else if (gr->is_chat && gr->messages_json && model_uses_mm_template(eng)) {
        st = mm_chat_encode(srv->tok, gr->messages_json, gr->tools_json,
                            gr->effort_level, &ids, &nids, err);
    } else if (gr->is_chat && gr->messages_json && model_uses_q35_template(eng)) {
        st = q35_chat_encode(srv->tok, gr->messages_json, gr->tools_json,
                             gr->effort_level, &ids, &nids, err);
    } else if (gr->is_chat) {
        st = cwick_chat_encode_messages(srv->tok, gr->msgs, gr->n_msgs,
                                       gr->tools_block, gr->reasoning_effort,
                                       &ids, &nids, err);
    } else {
        st = cwick_tokenizer_encode(srv->tok, gr->prompt, &ids, &nids, err);
    }
    if (st != CWICK_OK) { cwick_metrics_request_error("tokenize"); return; }
    if (srv->trace) {
        char *rendered = NULL;
        cwick_error ignored;
        cwick_error_clear(&ignored);
        if (cwick_tokenizer_decode(srv->tok, ids, nids, &rendered, &ignored)
                == CWICK_OK && rendered) {
            char label[256];
            snprintf(label, sizeof(label), "%s rendered prompt", req_tag(gr->rid));
            cwick_log_block(CWICK_LOG_TRACE, label, rendered, 65536, 4);
        }
        free(rendered);
    }
    gr->prompt_tokens = nids;
    CWICK_VERB(req_tag(gr->rid), "prompt encoded: %lld tokens (tools %s)",
              (long long)nids, gr->tools_block ? "declared" : "none");

    /* Does the prompt already sit *past* the reasoning block? GLM and
     * DeepSeek open it (their header ends with `<think>`), MiniMax closes it
     * (`</mm:think>` right after the assistant header when the turn is not
     * meant to think). That one fact decides three things that must not
     * disagree: whether the engine may write a closing marker, and where the
     * streaming splitter and the access-log replay start classifying. Getting
     * it wrong here is not cosmetic -- with the block closed in the prompt and
     * a splitter that starts in "thinking", a whole answer is delivered as
     * reasoning_content with an empty content. */
    const bool pfx_closed_at_start =
        gr->is_chat && srv->think_close_id >= 0 && nids > 0 &&
        ((int64_t)ids[nids - 1] == srv->think_close_id ||
         ((model_uses_mm_template(eng) || model_uses_q35_template(eng)) &&
          prompt_token_near_end(ids, nids, srv->think_close_id, 4)));

    /* ---- prefix cache: replay the part of the prompt already prefilled --
     * An agent client resends the same multi-thousand-token system prompt and
     * tool list on every turn. Re-prefilling it costs the whole TTFT; the
     * state that stands for it is a sequential read from disk. */
    int64_t pfx_reused = 0;
    int64_t pfx_hit = 0;     /* length the lookup matched, before truncation */
    cwick_prefix_state pfx_cap;
    memset(&pfx_cap, 0, sizeof(pfx_cap));
    bool pfx_cap_on = false;
    pfx_store_ctx pfx_ctx;   /* read by the store callback during generation */
    if (srv->pfx && nids > 1) {
        pthread_mutex_lock(&srv->pfx_mtx);
        int64_t P = cwick_pfx_lookup(srv->pfx, ids, nids);
        pfx_hit = P;
        if (P > 0) {
            cwick_error pe;
            cwick_error_clear(&pe);
            /* A hit that covers the whole prompt is replayed all but the last
             * token: that last position still has to be pushed through the
             * model, because the prefill is what produces the first-token
             * logits and a restored state carries no hidden state to run
             * lm_head on. The snapshot is truncated, not re-looked-up --
             * looking for a shorter entry would miss the one that exists. */
            const bool whole = (P >= nids);
            if (cwick_pfx_load(srv->pfx, ids, P, &pfx_cap, &pe) == CWICK_OK) {
                if (whole && pfx_cap.pos > 0) pfx_cap.pos--;
            }
            if (pfx_cap.pos > 0 &&
                cwick_engine_apply_prefix(eng, &pfx_cap, &pe) == CWICK_OK) {
                /* What was actually replayed, not what was looked up: on a
                 * whole-prompt hit those differ by one, and the tail handed
                 * to the engine has to be the difference. */
                pfx_reused = pfx_cap.pos;
                CWICK_INFO(req_tag(gr->rid), "prefix cache hit: %lld of %lld "
                          "prompt tokens replayed, %lld left to prefill",
                          (long long)pfx_reused, (long long)nids,
                          (long long)(nids - pfx_reused));
            } else {
                CWICK_INFO(req_tag(gr->rid), "prefix cache load failed (%s) — "
                          "prefilling the prompt in full", cwick_error_message(&pe));
                cwick_error re;
                cwick_error_clear(&re);
                cwick_engine_reset(eng, &re);   /* undo a partial apply */
            }
            cwick_prefix_state_free(&pfx_cap);
        }
        /* Accounted after the replay settles, so the number is what was
         * actually replayed rather than what the lookup matched. Those differ
         * by one on a whole-prompt hit (the last token always goes through
         * the model), and a load that failed counts as a query that missed --
         * which is what a hit-rate panel is supposed to show.
         *
         * Every lookup counts, including the ones that found nothing: a miss
         * is a query too. Counting only `P > 0` made queries == hits by
         * construction, so a hit-rate panel read either 0/0 or exactly 100%
         * and could never show the cache working badly. */
        cwick_metrics_prefix_query(pfx_reused > 0, pfx_reused);
        /* Snapshot the state once this prompt has been prefilled, so the next
         * request that starts with the same tokens can replay it -- unless this
         * exact prompt is already cached, in which case capturing it again
         * costs a ~0.9 GB allocation and copy (and a write that the store
         * would discard) to re-produce bytes we already have. A partial hit
         * still captures: what is new is the tail. */
        const bool already = (pfx_hit == nids);
        if (nids >= CWICK_PFX_MIN_TOKENS && !already) {
            memset(&pfx_cap, 0, sizeof(pfx_cap));
            pfx_ctx.srv = srv;
            pfx_ctx.eng = eng;
            pfx_ctx.ids = ids;
            pfx_ctx.nids = nids;
            snprintf(pfx_ctx.tag, sizeof(pfx_ctx.tag), "%s", req_tag(gr->rid));
            /* The engine fires pfx_save_now the moment the snapshot is
             * filled -- mid-request, not at its end -- so the snapshot is
             * durable even if the generation never completes. */
            cwick_engine_set_prefix_capture(eng, &pfx_cap);
            cwick_engine_set_prefix_capture_done(eng, pfx_save_now, &pfx_ctx);
            /* Long prompts additionally leave a checkpoint behind every
             * CWICK_PFX_BLOCK tokens (default 4096), so a later prompt that
             * shares only the head of this one can still replay that head. */
            cwick_engine_set_prefix_capture_block(eng, srv->pfx_block);
            pfx_cap_on = true;
        } else if (already) {
            CWICK_VERB(req_tag(gr->rid), "prefix already cached at %lld tokens — "
                      "not capturing it again", (long long)nids);
        }
        pthread_mutex_unlock(&srv->pfx_mtx);
    }

    int64_t max_new = gr->max_tokens_set ? gr->max_tokens : srv->max_tokens;
    if (max_new < 0) max_new = 0;

    /* Sampling, decided here because it is a property of (checkpoint, request)
     * and neither side alone knows both: the checkpoint's generation_config.json
     * supplies what the request left out, and the request supplies what it
     * insists on. */
    apply_sampling(eng, gr);

    cwick_gen_result res;
    memset(&res, 0, sizeof(res));

    /* A streaming request is served incrementally: the sink detokenizes and
     * pushes each token as the engine decodes it. Everything else (reasoning
     * split, tool-call extraction, access log) works off the same result. */
    /* L4 thinking budget: arm the ceiling this request asked for. The engine
     * writes the close tag itself when it is reached, into both the output and
     * the context, so a turn that would have thought for its whole budget ends
     * with an answer instead of `content=0`. Nothing is armed for completions
     * (no think block) or when the tokenizer does not give the markers as
     * single tokens (srv_probe_think_ids). */
    int64_t tbudget = -1;
    bool tbudget_armed = false;
    if (gr->is_chat && srv->think_open_id >= 0 && srv->think_close_id >= 0) {
        tbudget = think_budget_effective(gr->effort_level);
        if (tbudget >= 0) {
            /* GLM's template ends the prompt with <think>, so the block is
             * already open; a model whose prompt is a bare assistant header
             * writes the marker itself. Reading the last prompt token tells
             * the two apart without a per-family table. Qwen3.5 ends with the
             * marker followed by a newline token, so that exact test misses
             * an open block: use the same short backwards window as the
             * close-marker check. Without it the engine waits for a marker
             * the model will never write, the budget never fires, and the
             * thinking phase runs until max_tokens with content=0. */
            bool open_at_start =
                nids > 0 && (int64_t)ids[nids - 1] == srv->think_open_id;
            if (!open_at_start && !pfx_closed_at_start &&
                model_uses_q35_template(eng))
                open_at_start = prompt_token_near_end(ids, nids,
                                                      srv->think_open_id, 4);
            /* ...and one whose template closed it (MiniMax with thinking off)
             * must not have the engine close it a second time. */
            cwick_engine_set_think_closed_at_start(eng, pfx_closed_at_start);
            cwick_engine_set_think_budget(eng, tbudget, srv->think_open_id,
                                          srv->think_close_id, open_at_start);
            tbudget_armed = true;
            CWICK_VERB(req_tag(gr->rid), "thinking budget %lld tokens "
                       "(level=%s, block %s)", (long long)tbudget,
                       gr->effort_level ? gr->effort_level : "-",
                       open_at_start ? "already open"
                       : pfx_closed_at_start ? "already closed by the template"
                                             : "opened by the model");
        }
    }

    /* Bind the engine's log lines to this request: every prefill/decode
     * heartbeat, per-step decode timing and the generation summary then
     * carries the request tag, so a line can always be attributed. Cleared
     * right after the generation returns; CLI runs never set it. */
    cwick_engine_set_req_tag(eng, req_tag(gr->rid));
    /* The engine sees only the unreplayed tail; tell it the full prompt size
     * so the summary line reports the real request (a prefix-cache hit would
     * otherwise print "prompt=1" for a 499-token request). */
    cwick_engine_set_prompt_total(eng, nids);
    /* The id is minted once and used by both paths (the streaming envelope and
     * the buffered body), so a client that compares them cannot see two
     * different ids for one request. Responses gets its own prefix. */
    char idbuf[CWICK_SERVER_ID_BUF];
    make_id(idbuf, sizeof(idbuf),
            gr->api == SRV_API_RESPONSES ? "resp"
            : gr->api == SRV_API_MESSAGES ? "msg"
            : (gr->is_chat ? "chatcmpl" : "cmpl"));
    int64_t created = now_unix();
    const char *model = resident_served_name(srv);
    stream_ctx sc;
    memset(&sc, 0, sizeof(sc));
    const bool streaming = gr->stream;
    if (streaming) {
        sc.srv = srv;
        sc.gr = gr;
        sc.fd = fd;
        sc.created = created;
        memcpy(sc.idbuf, idbuf, sizeof(sc.idbuf));
        sc.model = model;
        sc.split.after_think = !gr->is_chat || pfx_closed_at_start;
        sc.split.open_mk = srv->think_open_mk;
        sc.split.close_mk = srv->think_close_mk;
        sc.raw_cap = 256;
        sc.raw = cwick_xmalloc(sc.raw_cap);
        sc.raw[0] = '\0';
        sc.gen_t0_ms = gen_t0;
        sc.queue_ms = queue_ms;
        sc.eng = eng;
        sc.pt.eng = eng;
        sc.pt.tag = req_tag(gr->rid);
        /* Anthropic reports the prompt size in message_start, which is on the
         * wire before prefill; the prefix-cache replay rides along as the
         * protocol's own `cache_read_input_tokens`. */
        sc.in_tokens = nids;
        sc.cached_tokens = pfx_reused;
        sc.msg_think_idx = -1;
        sc.msg_text_idx = -1;
        /* Commit HTTP + SSE before the potentially minute-long prefill. A
         * client/proxy then sees the request immediately; later failures are
         * reported as an in-band SSE error. */
        stream_begin(&sc);
        st = cwick_engine_generate_stream(eng, ids + pfx_reused,
                                         nids - pfx_reused, max_new,
                                         stream_sink, &sc, &res, err);
    } else {
        /* No socket to feed, but still a decode to label: the tracker
         * classifies tokens with the same machine the stream uses and keeps
         * the engine's progress lines honest about the phase. */
        phase_tracker ptk;
        memset(&ptk, 0, sizeof(ptk));
        ptk.pt.eng = eng;
        ptk.pt.tag = req_tag(gr->rid);
        ptk.split.after_think = !gr->is_chat || pfx_closed_at_start;
        ptk.split.open_mk = srv->think_open_mk;
        ptk.split.close_mk = srv->think_close_mk;
        ptk.tok = srv->tok;
        ptk.cap = 64;
        ptk.buf = cwick_xmalloc(ptk.cap);
        ptk.buf[0] = '\0';
        st = cwick_engine_generate_stream(eng, ids + pfx_reused,
                                         nids - pfx_reused, max_new,
                                         phase_track_sink, &ptk, &res, err);
        free(ptk.buf);
        /* Nothing is observable on the wire before the whole body lands, so
         * the prefill stands in for the first token: the answer starts
         * existing once the prompt has been processed. */
        if (st == CWICK_OK && res.prefill_ms > 0.0)
            cwick_metrics_observe_ttft((res.prefill_ms + queue_ms) / 1000.0);
    }
    cwick_engine_set_req_tag(eng, NULL);
    cwick_engine_set_prompt_total(eng, 0);
    cwick_engine_set_decode_phase(eng, NULL);
    cwick_engine_set_prefix_capture_block(eng, 0);
    /* The ceiling is per request; leaving it armed would silently cap the
     * next turn at this one's level. */
    if (tbudget_armed) cwick_engine_set_think_budget(eng, -1, -1, -1, false);

    /* Nothing to persist here: pfx_save_now already stored the snapshot the
     * moment prefill finished and freed its RAM copy, which is what makes the
     * cache survive an abort or disconnect later in the generation. All that
     * is left is detaching the engine-side pointers -- which also covers the
     * paths that never reached a prefill at all, where the snapshot is an
     * empty struct and there is nothing to free. */
    if (pfx_cap_on) {
        cwick_engine_set_prefix_capture(eng, NULL);
        cwick_engine_set_prefix_capture_done(eng, NULL, NULL);
    }

    if (st != CWICK_OK) {
        if (!streaming || !sc.started) {
            /* nothing on the wire yet: the caller still owns the error reply */
            cwick_metrics_request_error("generation");
            stream_ctx_free(&sc);
            free(ids);
            return;
        }
        cwick_metrics_request_error("stream_aborted");
        /* the SSE head is already sent, so the failure has to be reported
         * in-band; swallow it so the caller does not append a second body */
        stream_error(&sc, cwick_error_message(err));
        snprintf(detail, detail_cap, "stream aborted: %s",
                 cwick_error_message(err));
        cwick_error_clear(err);
        stream_ctx_free(&sc);
        free(ids);
        cwick_gen_result_free(&res);
        return;
    }

    /* The engine still holds the whole conversation (prompt + what it just
     * generated) and this is the last moment it does, so snapshot it before
     * anything is torn down. It runs on success only: an aborted or failed
     * generation leaves the state at a position no client will ever ask to
     * continue from, and a snapshot keyed on a truncated reply is dead weight
     * in the cache. */
    if (srv->pfx && !res.interrupted)
        pfx_save_generated(srv, eng, ids, nids, res.tokens, res.n_tokens,
                           req_tag(gr->rid));

    /* decode the generated text */
    char *text = NULL;
    st = cwick_tokenizer_decode(srv->tok, res.tokens, res.n_tokens, &text, err);
    if (st != CWICK_OK) {
        if (stream_report_error(&sc, streaming, err))
            snprintf(detail, detail_cap, "stream aborted: decode failed");
        cwick_metrics_request_error("detokenize");
        stream_ctx_free(&sc);
        free(ids);
        cwick_gen_result_free(&res);
        return;
    }

    /* ---- reasoning vs. answer --------------------------------------
     * GLM answers "<reasoning></think><visible answer>", exactly like the
     * OpenAI-compatible endpoint of the model: `reasoning_content` carries
     * the first half, `content` the second. */
    char *reasoning = NULL;
    char *answer = NULL;
    if (gr->is_chat)
        split_reasoning(text, srv->think_open_mk, srv->think_close_mk,
                        pfx_closed_at_start, &reasoning, &answer);
    else answer = cwick_strdup(text);

    /* ---- did the model call anything? ------------------------------
     * Tools are advertised in the checkpoint's own XML protocol, so any
     * answer is read back out of raw text. A malformed tool call is a
     * request we cannot honour, and it is refused rather than forwarded
     * half-parsed. */
    cwick_tool_call *calls = NULL;
    size_t ncalls = 0;
    char *visible = NULL;
    if (gr->tools.present && gr->tools.advertise) {
        if (cwick_tools_extract(answer, &calls, &ncalls, &visible, err) != 0) {
            if (stream_report_error(&sc, streaming, err))
                snprintf(detail, detail_cap, "stream aborted: malformed tool call");
            cwick_metrics_request_error("malformed_tool_call");
            stream_ctx_free(&sc);
            free(text);
            free(reasoning);
            free(answer);
            free(visible);
            free(ids);
            cwick_gen_result_free(&res);
            return;
        }
        for (size_t i = 0; i < ncalls; i++) {
            char orig[192];
            snprintf(orig, sizeof(orig), "%s", calls[i].name);
            cwick_decl_match m = cwick_tool_decl_match(&calls[i], &gr->tools);
            if (m == CWICK_DECL_FOLDED)
                CWICK_TRAC(req_tag(gr->rid),
                           "tool name `%s` folded to the declared spelling `%s`",
                           orig, calls[i].name);
            if (m == CWICK_DECL_UNKNOWN) {
                cwick_error_set(err, CWICK_ERR_UNSUPPORTED,
                               "model emitted undeclared tool `%s`", calls[i].name);
                if (stream_report_error(&sc, streaming, err))
                    snprintf(detail, detail_cap,
                             "stream aborted: undeclared tool");
                cwick_metrics_request_error("undeclared_tool");
                if (calls) cwick_tool_calls_free(calls, ncalls);
                stream_ctx_free(&sc); free(text); free(reasoning); free(answer);
                free(visible); free(ids); cwick_gen_result_free(&res);
                return;
            }
        }
        if (strcmp(gr->tools.choice, "named") == 0) {
            bool matched = false;
            for (size_t i = 0; i < ncalls; i++)
                if (strcmp(calls[i].name, gr->tools.forced) == 0) matched = true;
            if (!matched) {
                cwick_error_set(err, CWICK_ERR_UNSUPPORTED,
                               "tool_choice asked for `%s` but the model did "
                               "not call it", gr->tools.forced);
                if (stream_report_error(&sc, streaming, err))
                    snprintf(detail, detail_cap,
                             "stream aborted: tool_choice not satisfied");
                cwick_metrics_request_error("tool_choice");
                if (calls) cwick_tool_calls_free(calls, ncalls);
                stream_ctx_free(&sc);
                free(text);
                free(reasoning);
                free(answer);
                free(visible);
                free(ids);
                cwick_gen_result_free(&res);
                return;
            }
        }
    }
    char *tcalls = tool_calls_json(calls, ncalls, false);
    if (reasoning && *reasoning) {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s reasoning", req_tag(gr->rid));
        if (srv->trace) cwick_log_block(CWICK_LOG_TRACE, buf, reasoning, 16384, 4);
    }
    trace_output(srv, gr->rid, ncalls ? visible : answer, calls, ncalls);

    /* ---- execution detail for the access-log line -------------------
     * The engine already prints the canonical per-generation line
     * (prefill/decode split, tok/s, cache, streamed bytes, RSS); the server
     * only adds the token counts that belong to the HTTP request itself,
     * including the decode-phase split (thinking / content / tools). */
    size_t ph_counts[3] = { 0, 0, 0 };
    cwick_phase_account_mk(srv->tok, res.tokens, (size_t)res.n_tokens,
                           gr->is_chat, ph_counts,
                           srv->think_open_mk, srv->think_close_mk,
                           pfx_closed_at_start);
    /* The ceiling this turn ran under, and whether it had to be enforced.
     * The engine may have clamped the level's number to what the output budget
     * could honour, so report what it actually applied — a level of 8192
     * against max_tokens 8096 becomes 7584, and printing 8192 there would be a
     * cap the turn never had. `(hit)` means the engine wrote the close tag. */
    char tnote[48];
    tnote[0] = '\0';
    if (tbudget_armed) {
        const int64_t applied = res.think_budget_applied >= 0
                                    ? res.think_budget_applied : tbudget;
        snprintf(tnote, sizeof(tnote), " think_cap=%lld%s", (long long)applied,
                 res.think_budget_hit ? "(hit)" : "");
    }
    if (ncalls) {
        char names[192];
        names[0] = '\0';
        for (size_t i = 0; i < ncalls && i < 4; i++) {
            size_t room = sizeof(names) - strlen(names) - 1;
            if (i) strncat(names, ",", room);
            room = sizeof(names) - strlen(names) - 1;
            strncat(names, calls[i].name, room);
        }
        snprintf(detail, detail_cap, "prompt=%lld completion=%lld "
                 "finish=tool_calls calls=%s think=%zu tools=%zu content=%zu%s",
                 (long long)nids, (long long)res.n_tokens, names,
                 ph_counts[0], ph_counts[2], ph_counts[1], tnote);
    } else {
        /* `loop` is the access-log spelling of the loop guard's early stop;
         * on the wire it is a plain "stop" (OpenAI has no loop finish reason),
         * and the engine prints the period and the run next to its summary. */
        snprintf(detail, detail_cap, "prompt=%lld completion=%lld finish=%s "
                 "think=%zu tools=%zu content=%zu%s",
                 (long long)nids, (long long)res.n_tokens,
                 res.loop_stopped ? "loop"
                                  : (res.stop_reason_length ? "length" : "stop"),
                 ph_counts[0], ph_counts[2], ph_counts[1], tnote);
    }

    /* build the OpenAI response */
    const char *finish = ncalls ? "tool_calls"
                                : (res.stop_reason_length ? "length" : "stop");
    int64_t prompt_tokens = nids;
    int64_t completion_tokens = res.n_tokens;

    /* usage
     * `prompt_tokens` is the length of the prompt the client sent, in full --
     * not the part cwick actually prefilled. When a prefix hit covered most of
     * it, `prompt_tokens_details.cached_tokens` says how much was replayed,
     * which is OpenAI's field for exactly this and lets a client see what the
     * prefix cache saved it. Omitted on a miss so the object stays identical
     * to the pre-cache shape.
     *
     * `completion_tokens_details.reasoning_tokens` is what the thinking block
     * spent, the same number the Responses dialect reports in
     * `output_tokens_details` — the two envelopes must not disagree about how
     * much of the answer was thinking. */
    char usage[480];
    if (pfx_reused > 0)
        snprintf(usage, sizeof(usage),
                 "\"usage\":{\"prompt_tokens\":%lld,\"completion_tokens\":%lld,"
                 "\"total_tokens\":%lld,\"prompt_tokens_details\":"
                 "{\"cached_tokens\":%lld},\"completion_tokens_details\":"
                 "{\"reasoning_tokens\":%lld}}",
                 (long long)prompt_tokens, (long long)completion_tokens,
                 (long long)(prompt_tokens + completion_tokens),
                 (long long)pfx_reused, (long long)ph_counts[0]);
    else
        snprintf(usage, sizeof(usage),
                 "\"usage\":{\"prompt_tokens\":%lld,\"completion_tokens\":%lld,"
                 "\"total_tokens\":%lld,\"completion_tokens_details\":"
                 "{\"reasoning_tokens\":%lld}}",
                 (long long)prompt_tokens, (long long)completion_tokens,
                 (long long)(prompt_tokens + completion_tokens),
                 (long long)ph_counts[0]);

    /* assistant message content: the prose before the first <tool_call>,
     * or the whole answer when no tool was used. */
    const char *content_src = ncalls ? (visible ? visible : "") : answer;
    /* OpenAI sends content:null when the turn consists only of tool calls */
    char *content_field;
    if (ncalls && (!content_src || !*content_src))
        content_field = cwick_strdup("null");
    else
        content_field = cwick_json_stringify(content_src ? content_src : "");

    /* reasoning_content: the checkpoint's own reasoning block, or null when
     * this turn produced none (completions endpoint, or an empty <think>). */
    char *reasoning_field = NULL;
    if (gr->is_chat && reasoning && *reasoning)
        reasoning_field = cwick_json_stringify(reasoning);

    /* The Responses object, built once and used by both paths: the buffered
     * body and the terminal frame must be the same document. The reasoning
     * item is always present when the turn reasoned; its summary is only
     * filled in when the client asked for one. */
    const bool resp_incomplete = ncalls == 0 && res.stop_reason_length;
    char *resp_body = NULL;
    if (gr->api == SRV_API_RESPONSES) {
        resp_body = responses_body(idbuf, created, model, content_src,
                                   reasoning, gr->resp_summary, resp_incomplete,
                                   calls, ncalls, prompt_tokens,
                                   completion_tokens, pfx_reused,
                                   (int64_t)ph_counts[0], gr->resp_tail);
    }
    /* Anthropic names the three ways a turn can end; there is no "length" and
     * no "tool_calls" here. */
    const char *msg_stop = ncalls ? "tool_use"
                                  : (res.stop_reason_length ? "max_tokens"
                                                            : "end_turn");

    if (gr->stream) {
        /* The reasoning/content deltas were already pushed from inside the
         * generation sink as the tokens were produced; here we only close the
         * envelope (tool calls, finish frame, [DONE]). Flush the held-back
         * tail first so the last argument bytes are the final ones a client
         * sees, then settle the streamed call against the parsed result. */
        stream_flush(&sc, true);
        if (tool_std_settle(&sc, calls, ncalls)) {
            free(tcalls);
            /* A single call is complete on the wire already. With several,
             * the closing chunk still has to announce calls 1..N; the first
             * entry carries an empty arguments field so a concatenating
             * client does not see the streamed bytes twice. */
            tcalls = ncalls > 1 ? tool_calls_json(calls, ncalls, true) : NULL;
            /* The terminal Responses document embeds the call id and its
             * arguments, so rebuild it against the settled call. */
            if (gr->api == SRV_API_RESPONSES) {
                free(resp_body);
                resp_body = responses_body(idbuf, created, model, content_src,
                                           reasoning, gr->resp_summary,
                                           resp_incomplete, calls, ncalls,
                                           prompt_tokens, completion_tokens,
                                           pfx_reused, (int64_t)ph_counts[0],
                                           gr->resp_tail);
            }
        }
        if (gr->api == SRV_API_RESPONSES)
            resp_stream_finish(&sc, resp_body, calls, ncalls, resp_incomplete);
        else if (gr->api == SRV_API_MESSAGES)
            msg_stream_finish(&sc, calls, ncalls, msg_stop, completion_tokens);
        else
            stream_finish(&sc, finish, gr->include_usage, usage, tcalls);
        if (ttft_ms_out)
            *ttft_ms_out = sc.first_token_seen ? sc.first_token_ms : 0.0;
        stream_ctx_free(&sc);
        free(resp_body);
        free(content_field);
        free(reasoning_field);
        free(tcalls);
        if (calls) cwick_tool_calls_free(calls, ncalls);
        free(visible);
        free(reasoning);
        free(answer);
        free(ids);
        free(text);
        if (res_out) *res_out = res; else cwick_gen_result_free(&res);
        return;
    }

    char *body;
    if (gr->api == SRV_API_RESPONSES) {
        body = resp_body;
        resp_body = NULL;
    } else if (gr->api == SRV_API_MESSAGES) {
        body = messages_body(idbuf, model, content_src, reasoning, calls, ncalls,
                             prompt_tokens, completion_tokens, pfx_reused,
                             msg_stop);
    } else if (gr->is_chat) {
        body = srv_asprintf(
            "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%lld,"
            "\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{"
            "\"role\":\"assistant\",\"content\":%s,\"reasoning_content\":%s"
            "%s%s},\"finish_reason\":\"%s\"}],%s}",
            idbuf, (long long)created, model, content_field,
            reasoning_field ? reasoning_field : "null",
            /* The comma belongs to the tool_calls field, not to the one
             * before it: emitting `"reasoning_content":…"tool_calls":[…]`
             * produced invalid JSON, which any OpenAI client library
             * (and `json.tool`) rejects outright. */
            tcalls ? ",\"tool_calls\":" : "", tcalls ? tcalls : "",
            finish, usage);
    } else {
        body = srv_asprintf(
            "{\"id\":\"%s\",\"object\":\"text_completion\",\"created\":%lld,"
            "\"model\":\"%s\",\"choices\":[{\"index\":0,\"text\":%s,"
            "\"finish_reason\":\"%s\"}],%s}",
            idbuf, (long long)created, model, content_field, finish, usage);
    }
    http_response(fd, 200, "OK", "application/json; charset=utf-8", body, false);
    free(body);
    free(content_field);
    free(tcalls);
    free(reasoning_field);
    if (calls) cwick_tool_calls_free(calls, ncalls);
    free(visible);
    free(reasoning);
    free(answer);
    free(ids);
    free(text);
    if (res_out) *res_out = res; else cwick_gen_result_free(&res);
}

/* Every use of srv->tok lives inside run_generation(), including the tail that
 * runs after the slot has been handed back (decoding the final text, costing
 * the thinking/content phases). Counting the requests inside it is what lets a
 * model swap prove nobody is holding the tokenizer before it frees it. */
static void run_generation_hold(server_ctx *srv, gen_request *gr, int fd,
                                char *detail, size_t detail_cap, int slot,
                                double queue_ms, cwick_gen_result *res_out,
                                double *ttft_ms_out, cwick_error *err) {
    pthread_mutex_lock(&srv->gen_lock);
    srv->tok_users++;
    pthread_mutex_unlock(&srv->gen_lock);
    run_generation(srv, gr, fd, detail, detail_cap, slot, queue_ms, res_out,
                   ttft_ms_out, err);
    pthread_mutex_lock(&srv->gen_lock);
    srv->tok_users--;
    pthread_mutex_unlock(&srv->gen_lock);
}

/* ------------------------------------------------------------------ */
/* Request dispatch                                                    */
/* ------------------------------------------------------------------ */

/* The error envelope is part of the dialect: Anthropic clients parse
 * {"type":"error","error":{…}} and would treat the OpenAI shape as a
 * malformed success. The OpenAI wording is kept byte-identical so nothing
 * that already worked against this server changes behaviour. */
typedef enum { E_BAD = 0, E_OVERLOAD, E_SERVER } err_kind;

/* `code` is a JSON literal ("null" or a quoted string) used by the OpenAI
 * shape only; Anthropic has no per-error code field. */
static char *api_error_body(srv_api api, err_kind kind, const char *json_msg,
                            const char *code) {
    if (api == SRV_API_MESSAGES) {
        const char *t = kind == E_BAD ? "invalid_request_error"
                       : kind == E_OVERLOAD ? "overloaded_error"
                       : "api_error";
        return srv_asprintf("{\"type\":\"error\",\"error\":{\"type\":\"%s\","
                            "\"message\":%s}}", t, json_msg);
    }
    const char *t = kind == E_SERVER ? "server_error" : "invalid_request_error";
    return srv_asprintf("{\"error\":{\"message\":%s,\"type\":\"%s\","
                        "\"param\":null,\"code\":%s}}", json_msg, t,
                        code ? code : "null");
}

/* One access-log line per request: method, path, status, latency and — for
 * generations — the token counts. Everything else (per token timing, cache
 * behaviour) comes from the engine's own line. */
/* `cred` is which credential authorized the request ("session" or
 * "key:cwick_xxxxxxxx"): when a third-party agent is misbehaving, the first
 * question is always "as what", and the answer has to be in the same line as
 * the request. It is a prefix, never the key itself. */
/* Endpoints a dashboard or an orchestrator polls on a timer. They are not
 * requests in any useful sense: measured under --verbose they were 62% of all
 * log lines (a scrape every ~2 s printed two lines each) and buried the one
 * generation that was actually running. */
static bool probe_path(const char *path) {
    if (!path) return false;
    return strcmp(path, "/metrics") == 0 || strcmp(path, "/v1/metrics") == 0 ||
           strcmp(path, "/health") == 0 || strcmp(path, "/healthz") == 0;
}

static void access_log(uint64_t rid, const char *method,
                       const char *path, int status, double ms,
                       const char *detail, const char *cred) {
    /* A successful probe is trace-level: still available with --trace, absent
     * from a normal run. A failing probe stays at info -- a 401 storm is
     * exactly what an operator needs to see without enabling trace. */
    if (probe_path(path) && status == 200) {
        CWICK_TRAC(req_tag(rid), "%s %s -> %d in %.1fms%s%s%s%s", method, path,
                   status, ms, detail ? " " : "", detail ? detail : "",
                   cred ? " auth=" : "", cred ? cred : "");
        return;
    }
    CWICK_INFO(req_tag(rid), "%s %s -> %d in %.1fms%s%s%s%s", method, path, status,
              ms, detail ? " " : "", detail ? detail : "",
              cred ? " auth=" : "", cred ? cred : "");
}

/* ------------------------------------------------------------------ */
/* Log tail: everything this process printed, served on this port      */
/* ------------------------------------------------------------------ */

/* The built-in UI has to show what `cwick serve` prints — memory budget,
 * request lines, prefill heartbeats — and the only copy of that today is the
 * terminal of whoever launched it. logcap.c tees fd 1/fd 2 into a ring; these
 * two endpoints read it back. Both are answered before the generation slots
 * for the same reason metrics is: watching a run must never queue behind it.
 *
 * One thing to be honest about: with `--trace` the ring holds prompts and
 * outputs, so these endpoints serve them too. That is what the flag is for,
 * and the listener is loopback-only unless CWICK_ALLOW_PUBLIC=1. */
typedef struct { srv_buf *b; bool first; } logs_json_ctx;

static void logs_json_line(const cwick_logcap_line *l, void *user) {
    logs_json_ctx *c = (logs_json_ctx *)user;
    char *msg = cwick_json_stringify(l->text);
    if (!msg) return;
    sbuf_putf(c->b, "%s{\"n\":%llu,\"src\":\"%s\",\"t\":%lld,\"msg\":%s}",
              c->first ? "" : ",", (unsigned long long)l->seq,
              l->src == CWICK_LOGCAP_ERR ? "err" : "out",
              (long long)l->t_ms, msg);
    free(msg);
    c->first = false;
}

static char *logs_json_body(uint64_t since, size_t limit) {
    srv_buf b = {0};
    sbuf_putf(&b, "{\"capture\":%s,\"seq\":%llu,\"oldest\":%llu,"
                  "\"dropped\":%llu,\"retained\":%zu,\"lines\":[",
              cwick_logcap_active() ? "true" : "false",
              (unsigned long long)cwick_logcap_head(),
              (unsigned long long)cwick_logcap_oldest(),
              (unsigned long long)cwick_logcap_dropped(),
              cwick_logcap_lines());
    logs_json_ctx ctx = { &b, true };
    cwick_logcap_foreach(since, limit, logs_json_line, &ctx);
    sbuf_puts(&b, "]}");
    return b.p;   /* NULL only if the very first allocation failed */
}

typedef struct { int fd; bool alive; } log_stream_ctx;

static void logs_stream_line(const cwick_logcap_line *l, void *user) {
    log_stream_ctx *c = (log_stream_ctx *)user;
    if (!c->alive) return;
    char *msg = cwick_json_stringify(l->text);
    if (!msg) return;
    char *frame = srv_asprintf("{\"type\":\"line\",\"n\":%llu,\"src\":\"%s\","
                               "\"t\":%lld,\"msg\":%s}",
                               (unsigned long long)l->seq,
                               l->src == CWICK_LOGCAP_ERR ? "err" : "out",
                               (long long)l->t_ms, msg);
    free(msg);
    if (!frame) return;
    c->alive = sse_frame(c->fd, frame);
    free(frame);
}

/* Long-lived SSE tail of the ring. Ends when the client hangs up (a write
 * fails) or the process is stopping — checked between waits, so Ctrl+C is
 * honoured immediately instead of holding shutdown until the drain timeout. */
static void logs_stream(server_ctx *srv, int fd, uint64_t since,
                        size_t backlog) {
    http_stream_head(fd);
    log_stream_ctx ctx = { fd, true };

    char *hello = srv_asprintf("{\"type\":\"hello\",\"seq\":%llu,"
                               "\"capture\":%s,\"dropped\":%llu}",
                               (unsigned long long)cwick_logcap_head(),
                               cwick_logcap_active() ? "true" : "false",
                               (unsigned long long)cwick_logcap_dropped());
    if (hello) { ctx.alive = sse_frame(fd, hello); free(hello); }

    if (backlog > 0) {
        uint64_t cursor = cwick_logcap_head();
        uint64_t oldest = cwick_logcap_oldest();
        if (since < oldest || since > cursor) since = cursor;
        cwick_logcap_foreach(since, backlog, logs_stream_line, &ctx);
    }
    uint64_t cursor = cwick_logcap_head();

    double last_ping = now_ms();
    while (ctx.alive) {
        uint64_t head = cwick_logcap_wait(cursor, 400);
        if (head > cursor) {
            cwick_logcap_foreach(cursor, 0, logs_stream_line, &ctx);
            cursor = head;
        }
        if (srv->stop_owner && *srv->stop_owner) break;
        if (now_ms() - last_ping > 15000) {
            /* Comment frame: keeps an idle connection from being reaped by
             * anything sitting in front of us, and costs nothing. */
            ctx.alive = write_all_ok(fd, ": idle\n\n", 8);
            last_ping = now_ms();
        }
    }
}

/* ------------------------------------------------------------------ */
/* Model catalog: list, inspect, switch                                */
/* ------------------------------------------------------------------ */

static char *json_str_alloc(const char *s) {
    return cwick_json_stringify(s ? s : "");
}

static const char *status_text_for(int code) {
    switch (code) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 409: return "Conflict";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        default:  return "Error";
    }
}

/* Which entry is resident, resolved live from the engine: comparing roots is
 * the only question that matters (an id can repeat, a root cannot). */
static const cwick_model_entry *resident_entry(server_ctx *srv) {
    const char *root = srv->eng[0] ? cwick_engine_model_root(srv->eng[0]) : NULL;
    if (!root) return NULL;
    for (size_t i = 0; i < srv->catalog.n; i++) {
        if (strcmp(srv->catalog.v[i].root, root) == 0) return &srv->catalog.v[i];
    }
    return NULL;
}

/* The name a catalog entry goes by in OpenAI payloads: the directory-derived
 * served alias when the checkpoint has one, else what its config.json says.
 * Catalog entries are built once at startup and never mutated afterwards, so
 * the returned pointer stays valid for the process lifetime. */
static const char *served_name(const cwick_model_entry *e) {
    return (e && e->alias[0]) ? e->alias : e->id;
}

/* Same question for the resident model, for call sites that run before the
 * catalog exists (the first metrics publish) or without an entry in hand.
 * Falls back to the engine spec, which is always there. */
static const char *resident_served_name(server_ctx *srv) {
    const cwick_model_entry *cur = resident_entry(srv);
    if (cur) return served_name(cur);
    const cwick_model_spec *spec =
        srv->eng[0] ? cwick_engine_spec(srv->eng[0]) : NULL;
    return spec ? spec->model_type : "";
}

static char *models_json(server_ctx *srv) {
    pthread_mutex_lock(&srv->gen_lock);
    const cwick_model_entry *cur = resident_entry(srv);
    srv_buf b = {0};
    sbuf_puts(&b, "{\"object\":\"list\",\"data\":[");
    for (size_t i = 0; i < srv->catalog.n; i++) {
        const cwick_model_entry *e = &srv->catalog.v[i];
        bool loaded = cur && strcmp(cur->root, e->root) == 0;
        char *jid = json_str_alloc(served_name(e));
        char *jroot = json_str_alloc(e->root);
        if (!jid || !jroot) { free(jid); free(jroot); continue; }
        /* The reason is only useful when the answer is no; a client that
         * checks `supported` never has to parse it. */
        char extra[512] = "";
        if (!e->supported) {
            char *jwhy = json_str_alloc(e->unsupported_reason);
            snprintf(extra, sizeof(extra), ",\"unsupported_reason\":%s",
                     jwhy ? jwhy : "\"\"");
            free(jwhy);
        }
        sbuf_putf(&b,
                  "%s{\"id\":%s,\"object\":\"model\",\"created\":%lld,"
                  "\"owned_by\":\"cwick\",\"cwick\":{\"root\":%s,"
                  "\"loaded\":%s,\"supported\":%s%s}}",
                  i ? "," : "", jid, (long long)now_unix(), jroot,
                  loaded ? "true" : "false",
                  e->supported ? "true" : "false", extra);
        free(jid);
        free(jroot);
    }
    pthread_mutex_unlock(&srv->gen_lock);
    sbuf_puts(&b, "]}");
    return b.p;
}

/* One catalog entry, in the same shape as an element of the list above.
 * OpenAI's retrieve-model call (GET /v1/models/<id>) is what several agent
 * frameworks issue before their first completion to check the name the
 * operator typed; without it they report "model not found" even though
 * /v1/chat/completions would have accepted it. NULL means "not in the
 * catalog" and the caller answers 404. */
static char *model_json_one(server_ctx *srv, const char *id) {
    pthread_mutex_lock(&srv->gen_lock);
    const cwick_model_entry *cur = resident_entry(srv);
    const cwick_model_entry *hit = NULL;
    for (size_t i = 0; i < srv->catalog.n; i++) {
        /* Matched by either name: the served alias (what GET /v1/models just
         * told the caller) or the checkpoint's own model_type. */
        if (strcmp(srv->catalog.v[i].id, id) == 0 ||
            (srv->catalog.v[i].alias[0] &&
             strcmp(srv->catalog.v[i].alias, id) == 0)) {
            hit = &srv->catalog.v[i];
            break;
        }
    }
    if (!hit) {
        pthread_mutex_unlock(&srv->gen_lock);
        return NULL;
    }
    bool loaded = cur && strcmp(cur->root, hit->root) == 0;
    char *jid = json_str_alloc(served_name(hit));
    char *jroot = json_str_alloc(hit->root);
    srv_buf b = {0};
    if (jid && jroot) {
        char extra[512] = "";
        if (!hit->supported) {
            char *jwhy = json_str_alloc(hit->unsupported_reason);
            snprintf(extra, sizeof(extra), ",\"unsupported_reason\":%s",
                     jwhy ? jwhy : "\"\"");
            free(jwhy);
        }
        sbuf_putf(&b,
                  "{\"id\":%s,\"object\":\"model\",\"created\":%lld,"
                  "\"owned_by\":\"cwick\",\"cwick\":{\"root\":%s,"
                  "\"loaded\":%s,\"supported\":%s%s}}",
                  jid, (long long)now_unix(), jroot,
                  loaded ? "true" : "false",
                  hit->supported ? "true" : "false", extra);
    }
    free(jid);
    free(jroot);
    pthread_mutex_unlock(&srv->gen_lock);
    return b.p;
}

/* What this instance is doing, for the UI's settings page: the knobs the
 * command line set, reported back so a user can see why a request was clamped
 * without going looking for the terminal that started the server. */
static char *config_json(server_ctx *srv) {
    const cwick_model_entry *cur = resident_entry(srv);
    const char *id = cur ? served_name(cur)
                         : (srv->eng[0] ? cwick_engine_spec(srv->eng[0])->model_type
                                        : "");
    const char *root = cur ? cur->root
                           : (srv->eng[0] ? cwick_engine_model_root(srv->eng[0])
                                          : "");
    char *jid = json_str_alloc(id);
    char *jroot = json_str_alloc(root);
    char *jhost = json_str_alloc(srv->host);
    char *jlib = json_str_alloc(srv->hub ? cwick_hub_root(srv->hub) : "");
    char *jcat = json_str_alloc(srv->hub ? cwick_hub_endpoint(srv->hub) : "");
    srv_buf b = {0};
    sbuf_putf(&b,
              "{\"model\":%s,\"model_root\":%s,\"host\":%s,\"port\":%d,"
              "\"slots\":%d,\"max_tokens\":%lld,"
              "\"max_request_bytes\":%lld,"
              "\"context_window\":%lld,\"catalog_size\":%zu,"
              "\"web\":%s,\"verbose\":%s,\"trace\":%s,\"systemone\":%s,"
              "\"prefix_cache\":{\"enabled\":%s,\"block_tokens\":%lld},"
              "\"log_capture\":%s,\"log_retained\":%zu,"
              "\"model_library\":%s,\"model_catalog\":%s,"
              "\"models_dirs\":[",
              jid ? jid : "\"\"", jroot ? jroot : "\"\"", jhost ? jhost : "\"\"",
              srv->port, srv->nslots, (long long)srv->max_tokens,
              (long long)srv->max_request_bytes,
              (long long)(srv->eng[0] ? cwick_engine_context_window(srv->eng[0]) : 0),
              srv->catalog.n, srv->web ? "true" : "false",
              srv->verbose ? "true" : "false", srv->trace ? "true" : "false",
              srv->jev ? "true" : "false",
              srv->pfx ? "true" : "false", (long long)srv->pfx_block,
              cwick_logcap_active() ? "true" : "false", cwick_logcap_lines(),
              jlib ? jlib : "\"\"", jcat ? jcat : "\"\"");
    free(jid); free(jroot); free(jhost); free(jlib); free(jcat);
    for (int i = 0; i < srv->models_dirs_n; i++) {
        char *jd = json_str_alloc(srv->models_dirs[i]);
        sbuf_putf(&b, "%s%s", i ? "," : "", jd ? jd : "\"\"");
        free(jd);
    }
    sbuf_puts(&b, "]}");
    return b.p;
}

/* ------------------------------------------------------------------ */
/* Model library: catalog search, local inventory, download queue      */
/* ------------------------------------------------------------------ */

static void hub_send_json(int fd, int code, const char *body) {
    http_response(fd, code, status_text_for(code),
                  "application/json; charset=utf-8", body, false);
}
static void hub_send_error(int fd, int code, const char *msg) {
    char *jm = json_str_alloc(msg ? msg : "unknown error");
    char out[1024];
    snprintf(out, sizeof(out),
             "{\"error\":{\"message\":%s,\"type\":\"invalid_request_error\"}}",
             jm ? jm : "\"unknown error\"");
    free(jm);
    hub_send_json(fd, code, out);
}

/* /v1/models is the list of switchable roots. When a checkpoint is deleted
 * from the library its catalog entry has to go with it, or the list keeps
 * advertising a root that no longer exists and every switch to it answers
 * 400. The delete is already on disk when this runs, so a concurrent reader
 * sees either the entry (and gets a 404/400 from the switch) or no entry. */
static void catalog_drop_root(server_ctx *srv, const char *root) {
    pthread_mutex_lock(&srv->gen_lock);
    for (size_t i = 0; i < srv->catalog.n; i++) {
        if (strcmp(srv->catalog.v[i].root, root) != 0) continue;
        memmove(&srv->catalog.v[i], &srv->catalog.v[i + 1],
                (srv->catalog.n - i - 1) * sizeof(*srv->catalog.v));
        srv->catalog.n--;
        break;
    }
    pthread_mutex_unlock(&srv->gen_lock);
}

/* Deleting the weights a running engine has mapped would leave it reading
 * files that no longer exist. Refuse and tell the caller to switch first. */
static bool path_is_resident(server_ctx *srv, const char *root) {
    pthread_mutex_lock(&srv->gen_lock);
    const cwick_model_entry *cur = resident_entry(srv);
    bool same = cur && strcmp(cur->root, root) == 0;
    pthread_mutex_unlock(&srv->gen_lock);
    return same;
}

/* Answers every model-library request under /v1/hub/, or returns -1 for any
 * are control-plane reads: they are served before the generation slot is
 * taken, so a catalog that is slow to answer cannot hold up a chat request.
 * Progress is polled from the UI rather than streamed, which keeps the job
 * table the only piece of shared state. */
static int hub_route(server_ctx *srv, int fd, const char *method,
                     const char *path, const char *ponly,
                     const char *body, size_t body_len,
                     char *detail, size_t detailcap) {
    if (!srv->hub || strncmp(ponly, "/v1/hub/", 8) != 0) return -1;
    const char *leaf = ponly + 8;
    char err[256] = "";
    char *json = NULL;
    int fail_status = 400;

    if (strcmp(method, "GET") == 0 && strcmp(leaf, "search") == 0) {
        char q[256] = "";
        const char *raw = query_find(path, "q");
        if (raw && !query_decode(raw, q, sizeof(q))) {
            hub_send_error(fd, 400, "malformed q");
            snprintf(detail, detailcap, "model search -> 400");
            return 400;
        }
        int limit = 12;
        const char *rl = query_find(path, "limit");
        if (rl) { int v = atoi(rl); if (v > 0) limit = v < 50 ? v : 50; }
        json = cwick_hub_search(srv->hub, q, limit, err, sizeof(err));
        snprintf(detail, detailcap, "model catalog search '%.60s'", q);
    } else if (strcmp(method, "GET") == 0 && strcmp(leaf, "models") == 0) {
        json = cwick_hub_models(srv->hub, err, sizeof(err));
        snprintf(detail, detailcap, "local model inventory");
    } else if (strcmp(method, "GET") == 0 && strcmp(leaf, "downloads") == 0) {
        json = cwick_hub_downloads(srv->hub, err, sizeof(err));
        snprintf(detail, detailcap, "download queue");
    } else if (strcmp(method, "DELETE") == 0 &&
               strncmp(leaf, "models/", 7) == 0 && leaf[7]) {
        /* Deleting a downloaded checkpoint. The id may be owner/name or a
         * bare directory name, and is percent-decoded like a model id. */
        if (strlen(leaf + 7) >= 256) {
            hub_send_error(fd, 400, "model name is too long");
            snprintf(detail, detailcap, "delete -> 400 (name too long)");
            return 400;
        }
        char name[256];
        url_decode_segment(leaf + 7, name, sizeof(name));
        char target[CWICK_HUB_PATH_MAX];
        if (!cwick_hub_model_dir(srv->hub, name, target, sizeof(target))) {
            hub_send_error(fd, 404, "no such local model");
            snprintf(detail, detailcap, "delete %.60s -> 404", name);
            return 404;
        }
        if (path_is_resident(srv, target)) {
            hub_send_error(fd, 409,
                           "the resident model cannot be deleted; switch to "
                           "another model first");
            snprintf(detail, detailcap, "delete %.60s -> 409 (resident)", name);
            return 409;
        }
        json = cwick_hub_delete(srv->hub, name, &fail_status, err, sizeof(err));
        if (!json) {
            hub_send_error(fd, fail_status, err[0] ? err : "delete failed");
            snprintf(detail, detailcap, "delete %.60s -> %d: %s",
                     name, fail_status, err);
            return fail_status;
        }
        /* Disk first, then the catalog, so the two never disagree. */
        catalog_drop_root(srv, target);
        snprintf(detail, detailcap, "deleted local model %.80s", name);
    } else if (strcmp(method, "POST") == 0 && strcmp(leaf, "download") == 0) {
        cwick_error je;
        cwick_error_clear(&je);
        json_value *root = cwick_json_parse(body, body_len, &je);
        json_value *r = root ? cwick_json_get(root, "repo") : NULL;
        if (!r || r->kind != JSON_STRING || !cwick_json_str(r) ||
            !cwick_json_str(r)[0]) {
            cwick_json_free(root);
            hub_send_error(fd, 400, "body.repo must be owner/name");
            snprintf(detail, detailcap, "download -> 400 (no repo)");
            return 400;
        }
        char repo[256];
        snprintf(repo, sizeof(repo), "%s", cwick_json_str(r));
        cwick_json_free(root);
        json = cwick_hub_download_start(srv->hub, repo, err, sizeof(err));
        snprintf(detail, detailcap, "start download %.80s", repo);
    } else if (strcmp(method, "POST") == 0 &&
               strncmp(leaf, "download/", 9) == 0) {
        const char *rest = leaf + 9;
        const char *slash = strchr(rest, '/');
        if (!slash || slash == rest || strchr(slash + 1, '/')) {
            hub_send_error(fd, 400, "expected <id>/<action>");
            snprintf(detail, detailcap, "download action -> 400");
            return 400;
        }
        char id[32], action[32];
        size_t ni = (size_t)(slash - rest);
        if (ni >= sizeof(id)) {
            hub_send_error(fd, 400, "download id is too long");
            return 400;
        }
        memcpy(id, rest, ni);
        id[ni] = '\0';
        snprintf(action, sizeof(action), "%s", slash + 1);
        json = cwick_hub_download_action(srv->hub, id, action,
                                         err, sizeof(err));
        snprintf(detail, detailcap, "download %s -> %s", id, action);
    } else {
        hub_send_error(fd, 404, "no such model-library endpoint");
        snprintf(detail, detailcap, "unknown model-library path");
        return 404;
    }

    if (!json) {
        hub_send_error(fd, fail_status, err[0] ? err : "model library error");
        snprintf(detail, detailcap, "model library -> %d: %s", fail_status, err);
        return fail_status;
    }
    hub_send_json(fd, 200, json);
    free(json);
    return 200;
}

/* Nothing may be using the engine or the tokenizer while they are freed and
 * rebuilt. Both conditions are checked under gen_lock in the same critical
 * section that flips `swapping`, so a request cannot sneak through between
 * the check and the teardown. */
static bool swap_gate_open(server_ctx *srv, char *why, size_t cap) {
    int busy = 0;
    for (int i = 0; i < CWICK_GEN_SLOTS; i++) if (srv->slot_busy[i]) busy++;
    if (busy > 0) {
        snprintf(why, cap, "%d generation slot(s) are busy", busy);
        return false;
    }
    if (srv->nwaiting > 0) {
        snprintf(why, cap, "%d request(s) are queued for a slot", srv->nwaiting);
        return false;
    }
    if (srv->tok_users > 0) {
        snprintf(why, cap, "%d request(s) still hold the tokenizer",
                 srv->tok_users);
        return false;
    }
    return true;
}

/* Replace the resident engine with another root. Returns an HTTP status and
 * explains itself in `msg`. Runs outside every generation path: while it
 * works, slot_acquire() refuses with 503 rather than queueing, because the
 * model a queued request was validated against is being destroyed. */
static int srv_switch_model(server_ctx *srv, const cwick_model_entry *target,
                            char *msg, size_t msgcap) {
    /* Validate the target *before* anything is released. The old sequence
     * tore down the engine, tokenizer, Metal arena and prefix cache first and
     * only then tried to load the replacement: a repository with no kernels
     * took the whole instance down with a 500, and every later switch saw
     * "(none)" as the resident model. Both checks here are read-only and run
     * outside the generation lock, so metrics and chat requests keep flowing
     * while a config is parsed. */
    char support_why[256];
    if (!cwick_model_root_supported(target->root, support_why,
                                    sizeof(support_why))) {
        snprintf(msg, msgcap, "cannot switch to %s: %s",
                 served_name(target), support_why);
        CWICK_INFO("model", "%s", msg);
        return 400;
    }
    char cfg[CWICK_MODEL_ROOT_MAX + 32];
    if (snprintf(cfg, sizeof(cfg), "%s/config.json", target->root) >=
        (int)sizeof(cfg)) {
        snprintf(msg, msgcap, "cannot switch to %s: model root is too long",
                 served_name(target));
        return 400;
    }
    cwick_model_spec dry;
    cwick_error derr;
    cwick_error_clear(&derr);
    cwick_status dry_st = cwick_model_spec_load(cfg, &dry, &derr);
    cwick_model_spec_free(&dry);
    if (dry_st != CWICK_OK) {
        snprintf(msg, msgcap,
                 "cannot switch to %s: its config does not satisfy the model "
                 "contract (%s); the resident model was left untouched",
                 served_name(target), cwick_error_message(&derr));
        CWICK_INFO("model", "%s", msg);
        return 400;
    }

    char why[256];
    pthread_mutex_lock(&srv->gen_lock);
    if (srv->swapping) {
        pthread_mutex_unlock(&srv->gen_lock);
        snprintf(msg, msgcap, "a model swap is already in progress");
        return 409;
    }
    if (!swap_gate_open(srv, why, sizeof(why))) {
        pthread_mutex_unlock(&srv->gen_lock);
        snprintf(msg, msgcap, "cannot switch now: %s", why);
        return 409;
    }
    if (srv->stop_owner && *srv->stop_owner) {
        pthread_mutex_unlock(&srv->gen_lock);
        snprintf(msg, msgcap, "server is shutting down");
        return 503;
    }
    if (!srv->live_engine) {
        pthread_mutex_unlock(&srv->gen_lock);
        snprintf(msg, msgcap, "this instance does not own its engine, so the "
                              "resident model cannot be replaced at runtime");
        return 400;
    }
    srv->swapping = true;
    char old_copy[CWICK_MODEL_ROOT_MAX];
    snprintf(old_copy, sizeof(old_copy), "%s",
             srv->eng[0] ? cwick_engine_model_root(srv->eng[0]) : "(none)");
    /* Sampled before anything is released: this is the number the switch has
     * to get back down to, and the only way to tell "freed the objects" from
     * "freed the pages" in the log afterwards. */
    const uint64_t rss_before = cwick_process_rss_bytes();
    pthread_mutex_unlock(&srv->gen_lock);

    CWICK_INFO("model", "switching to %s (%s) — releasing %s first: the "
                       "resident spine is most of the memory budget, so the "
                       "old weights have to go before the new ones are mapped",
              served_name(target), target->root, old_copy);

    /* Borrowers before the owner: they borrow the owner's mmaps. */
    for (int i = CWICK_GEN_SLOTS - 1; i >= 1; i--) {
        if (srv->eng[i]) { cwick_engine_free(srv->eng[i]); srv->eng[i] = NULL; }
    }
    pthread_mutex_lock(&srv->gen_lock);
    cwick_tokenizer *old_tok = srv->tok;
    srv->tok = NULL;
    srv->nslots = 0;
    pthread_mutex_unlock(&srv->gen_lock);
    if (old_tok) cwick_tokenizer_free(old_tok);
    if (srv->eng[0]) { cwick_engine_free(srv->eng[0]); srv->eng[0] = NULL; }
    if (srv->live_engine) *srv->live_engine = NULL;   /* nothing to free twice */

    /* Give the outgoing model's GPU-visible bytes back *before* the incoming
     * one plans its budget. `cwick_engine_free` releases the C objects, but the
     * expert arenas and the no-copy views over the old weights are wired
     * MTLBuffers held in malloc'd structs -- nothing frees those on unload, so
     * without this line the switch leaves the old model's arena (tens of GiB
     * after a warmed-up run) resident, the new model's plan over-commits, and
     * the 80% ceiling check spends the request dismantling the expert cache
     * rather than the leftovers. Measured on DeepSeek -> GLM: 67.19 GiB
     * warm-up without this, 43.74 GiB with it (same model started fresh:
     * 14.49 GiB -- the rest is the outgoing checkpoint's mappings, which are
     * still process-lifetime; see docs/DECISIONS.md §24). */
    cwick_metal_arena_release();
    /* free() does not give the pages back. A model that peaked at 60 GiB of
     * scratch buffers returns them to the allocator's free lists, which keeps
     * them resident: measured 29.28 GiB still resident after every object,
     * arena and mapping was released. `malloc_zone_pressure_relief` is the
     * macOS call that asks the zones to hand their free pages to the VM, and a
     * model switch is exactly the moment to ask -- nothing else in the process
     * is allocating yet, and the incoming model is about to size its budget
     * against whatever is left. */
#if defined(__APPLE__)
    malloc_zone_pressure_relief(NULL, 0);
#endif
    /* One line that says what actually came back. A switch is the only moment
     * where "the old model is gone" can be checked against "the memory is
     * gone", and without it a switch that frees every object and no pages
     * looks exactly like a good one in the log. */
    CWICK_INFO("model", "released %s: RSS %.2f GiB before %s loads "
                       "(it was %.2f GiB when the switch started)",
               old_copy, cwick_gib(cwick_process_rss_bytes()),
               served_name(target), cwick_gib(rss_before));

    /* Prefix snapshots belong to the geometry that produced them. */
    if (srv->pfx) {
        pthread_mutex_lock(&srv->pfx_mtx);
        cwick_pfx_clear(srv->pfx);
        pthread_mutex_unlock(&srv->pfx_mtx);
        CWICK_INFO("model", "prefix cache cleared: snapshots taken from %s are "
                           "not a legal state for %s", old_copy,
                           served_name(target));
    }

    /* Rebuild: the startup options with a different root. */
    cwick_engine_options eopt;
    memset(&eopt, 0, sizeof(eopt));
    eopt.model_root = target->root;
    eopt.max_context = srv->context_capacity;
    eopt.spine = srv->spine_kind;
    eopt.max_new_tokens = srv->max_tokens;
    eopt.verbose = srv->verbose;
    eopt.stop = srv->stop_owner;
    eopt.slot_id = 0;
    cwick_error e;
    cwick_error_clear(&e);
    if (cwick_engine_new(&eopt, &srv->eng[0], &e) != CWICK_OK) {
        snprintf(msg, msgcap, "loading %s failed: %s", served_name(target),
                 cwick_error_message(&e));
        CWICK_INFO("model", "%s", msg);
        /* No model until someone loads one. Every generation endpoint now
         * answers 503 rather than pretending the switch worked. The gauge has
         * to say so too: model_info is a static label and would otherwise keep
         * naming the checkpoint that was just released. */
        cwick_metrics_set_model_loaded(false);
        srv->swapping = false;
        return 500;
    }
    cwick_engine_set_stop(srv->eng[0], srv->stop_owner);
    cwick_engine_set_verbose(srv->eng[0], srv->verbose);
    if (srv->live_engine) *srv->live_engine = srv->eng[0];

    /* Extra slots, exactly as at startup (one by default — see the note in
     * cwick_server_run about why more are gated behind CWICK_GEN_SLOTS). */
    int live = 1;
    for (int i = 1; i < srv->slot_want; i++) {
        eopt.slot_id = i;
        eopt.shared_spine = cwick_engine_spine(srv->eng[0]);
        cwick_error e2;
        cwick_error_clear(&e2);
        if (cwick_engine_new(&eopt, &srv->eng[i], &e2) != CWICK_OK) {
            CWICK_INFO("model", "generation slot %d unavailable after the swap "
                               "(%s) — serving with %d slot(s)", i,
                      cwick_error_message(&e2), i);
            srv->eng[i] = NULL;
            break;
        }
        cwick_engine_set_stop(srv->eng[i], srv->stop_owner);
        live++;
    }

    cwick_tokenizer *tok = NULL;
    cwick_error terr;
    cwick_error_clear(&terr);
    if (cwick_tokenizer_load(cwick_engine_model_root(srv->eng[0]),
                            cwick_engine_spec(srv->eng[0]), &tok, &terr) != CWICK_OK) {
        snprintf(msg, msgcap, "tokenizer for %s failed to load: %s",
                 served_name(target), cwick_error_message(&terr));
        CWICK_INFO("model", "%s", msg);
        /* Half a model is not a model: without a tokenizer every generation
         * path would dereference NULL. Drop the engine too, and let the
         * "no model" guard answer 503 until someone switches again. */
        for (int i = CWICK_GEN_SLOTS - 1; i >= 0; i--) {
            if (srv->eng[i]) { cwick_engine_free(srv->eng[i]); srv->eng[i] = NULL; }
        }
        if (srv->live_engine) *srv->live_engine = NULL;
        cwick_metrics_set_model_loaded(false);
        srv->swapping = false;
        return 500;
    }

    pthread_mutex_lock(&srv->gen_lock);
    srv->tok = tok;
    srv_probe_think_ids(srv);
    srv->nslots = live;
    srv->swapping = false;
    pthread_mutex_unlock(&srv->gen_lock);
    /* Outside the lock: it encodes a handful of short strings against the new
     * tokenizer, and doing it here is what makes "switch the model, switch the
     * decision interface" one operation instead of two. */
    srv_probe_decision_adapter(srv);

    /* Republish what the metrics endpoint claims we are running. */
    {
        const cwick_model_spec *mspec = cwick_engine_spec(srv->eng[0]);
        const cwick_spine *spine = cwick_engine_spine(srv->eng[0]);
        cwick_metrics_info mi;
        memset(&mi, 0, sizeof(mi));
        mi.model = served_name(target);
        mi.backend = cwick_metal_backend_active() ? "metal" : "cpu";
        mi.spine = spine && spine->have_int8 ? "int8" : "bf16";
        mi.max_context = cwick_engine_context_window(srv->eng[0]);
        mi.num_layers = mspec->num_hidden_layers;
        mi.num_experts = mspec->num_experts;
        mi.experts_per_token = mspec->num_experts_per_token;
        cwick_metrics_set_model_info(&mi);
        cwick_metrics_set_context_window(mi.max_context);
        /* GLM keeps the pin cache on the spine; the DeepSeek backend owns its
         * own stream. Without the fallback the expert-cache gauges keep the
         * previous model's numbers (or zero) after a switch. */
        if (spine && spine->stream) cwick_metrics_set_stream(spine->stream);
        else {
            cwick_stream *es = cwick_engine_expert_stream(srv->eng[0]);
            if (es) cwick_metrics_set_stream(es);
        }
        const cwick_mem_plan *plan = spine ? cwick_spine_memory_plan(spine) : NULL;
        if (plan) cwick_metrics_set_memory_ceiling(plan->target_bytes);
    }

    const cwick_model_spec *spec = cwick_engine_spec(srv->eng[0]);
    snprintf(msg, msgcap, "now serving %s from %s (%lld layers, %lld experts, "
                          "context %lld)", served_name(target), target->root,
             (long long)spec->num_hidden_layers, (long long)spec->num_experts,
             (long long)cwick_engine_context_window(srv->eng[0]));
    CWICK_INFO("model", "switched to %s — %zu model(s) in the catalog, %d "
                       "generation slot(s) live",
              served_name(target), srv->catalog.n, live);
    return 200;
}

/* How much history a client that did not ask for anything gets: enough to
 * explain what happened a request ago, not a whole multi-hour session. */
#define CWICK_LOGS_BACKLOG 300

/* One connection, one thread. The generation itself stays behind the slot
 * queue (the engine has a single slot, so a second concurrent generation
 * waits for its turn), so the only things that actually run in parallel are
 * the cheap endpoints -- /metrics, /v1/models, 404s, malformed bodies --
 * plus the pre-tokenize work of a queued generation. That is the whole
 * point: a minutes-long chat completion must not make the process look
 * dead to a scraper or to a second client. */
typedef struct {
    server_ctx *srv;
    int fd;
    uint64_t rid;
} client_job;


/* ------------------------------------------------------------------ */
/* Single-user auth                                                    */
/* ------------------------------------------------------------------ */
#define CWICK_AUTH_COOKIE "cwick_session"

static bool cookie_get(const char *cookie, const char *name, char *out, size_t cap) {
    if (!cookie || !*cookie || !name || !out || cap == 0) return false;
    size_t nlen = strlen(name);
    const char *p = cookie;
    while (*p) {
        while (*p == ' ' || *p == ';') p++;
        const char *key = p;
        while (*p && *p != '=' && *p != ';') p++;
        if (*p != '=') { while (*p && *p != ';') p++; continue; }
        size_t klen = (size_t)(p - key);
        p++; /* '=' */
        const char *val = p;
        while (*p && *p != ';') p++;
        if (klen == nlen && strncmp(key, name, nlen) == 0) {
            size_t vlen = (size_t)(p - val);
            if (vlen >= cap) vlen = cap - 1;
            memcpy(out, val, vlen);
            out[vlen] = '\0';
            return vlen > 0;
        }
    }
    return false;
}

static bool request_auth_token(const http_req *req, char *out, size_t cap) {
    out[0] = '\0';
    const char *a = req->authorization;
    if (a && strncasecmp(a, "Bearer ", 7) == 0) {
        const char *t = a + 7;
        while (*t == ' ') t++;
        size_t n = strcspn(t, " \t\r\n");
        if (n == 0 || n >= cap) return false;
        memcpy(out, t, n);
        out[n] = '\0';
        return true;
    }
    return cookie_get(req->cookie, CWICK_AUTH_COOKIE, out, cap);
}

/* Which credential served the current request. Thread-local like req_tag():
 * one thread per connection, and the access log needs it after the check has
 * already been made. It names an API key by its *prefix*, never by its secret —
 * the log has to answer "which key" without ever carrying the key. */
static __thread char tls_auth_who[64];

/* Pull the bearer secret out of either header. `X-API-Key` wins over
 * `Authorization` only in the sense that it is tried first when a client sets
 * both to different values, which no sane client does. */
static bool request_bearer(const http_req *req, char *out, size_t cap) {
    if (req->api_key[0]) {
        size_t n = strcspn(req->api_key, " \t\r\n");
        if (n == 0 || n >= cap) return false;
        memcpy(out, req->api_key, n);
        out[n] = '\0';
        return true;
    }
    const char *a = req->authorization;
    if (a && strncasecmp(a, "Bearer ", 7) == 0) {
        const char *t = a + 7;
        while (*t == ' ') t++;
        size_t n = strcspn(t, " \t\r\n");
        if (n == 0 || n >= cap) return false;
        memcpy(out, t, n);
        out[n] = '\0';
        return true;
    }
    return false;
}

/* Session only: a browser cookie or a token minted by POST /v1/auth/login.
 * API keys are deliberately *not* accepted here — see keys_manage_allowed(). */
static bool request_has_session(server_ctx *srv, const http_req *req) {
    if (!srv->auth) return true;
    char token[128];
    if (!request_auth_token(req, token, sizeof(token))) return false;
    return cwick_auth_validate(srv->auth, token);
}

static bool request_authenticated(server_ctx *srv, const http_req *req) {
    if (!srv->auth) return true;
    tls_auth_who[0] = '\0';
    /* A session first: it is the cheap case (an array walk) and it is what a
     * logged-in browser sends on every request. */
    if (request_has_session(srv, req)) {
        snprintf(tls_auth_who, sizeof(tls_auth_who), "session");
        return true;
    }
    char secret[256];
    if (!request_bearer(req, secret, sizeof(secret))) return false;
    char id[32];
    if (!cwick_auth_key_check(srv->auth, secret, id, sizeof(id))) return false;
    /* Remember when it was used, but never block a request on the write. */
    if (srv->store) cwick_store_key_used(srv->store, id, 0);
    snprintf(tls_auth_who, sizeof(tls_auth_who), "key:%s",
             cwick_auth_key_prefix_of(srv->auth, id));
    return true;
}

static bool path_is_login_page(const char *p) {
    return strcmp(p, "/login") == 0 || strcmp(p, "/login.html") == 0;
}

static bool path_is_public_api(const char *p) {
    return strcmp(p, "/health") == 0 || strcmp(p, "/healthz") == 0 ||
           strcmp(p, "/v1/auth/login") == 0 ||
           strcmp(p, "/v1/auth/status") == 0 ||
           strcmp(p, "/v1/auth/logout") == 0;
}

/* CSS/JS/manifest are harmless without a session; the HTML that would call the
 * model is what gets a login redirect. */
static bool path_is_public_asset(const char *p) {
    if (strcmp(p, "/login.html") == 0 || strcmp(p, "/favicon.ico") == 0) return true;
    if (strncmp(p, "/web/", 5) == 0) return true;
    if (strncmp(p, "/apps/", 6) == 0) {
        size_t n = strlen(p);
        if (strstr(p, ".html") != NULL) return false;
        if (n && p[n - 1] == '/') return false;
        const char *base = strrchr(p, '/');
        if (!base || !strrchr(base, '.')) return false;  /* app dir alias */
        return true;
    }
    if (strcmp(p, "/apps") == 0 || strcmp(p, "/apps/") == 0) return true;
    return false;
}

static bool path_is_navigation(const char *method, const char *p) {
    if (strcmp(method, "GET") != 0) return false;
    if (strcmp(p, "/") == 0) return true;
    size_t n = strlen(p);
    if (n && p[n - 1] == '/') return true;
    if (strstr(p, ".html") != NULL) return true;
    if (strncmp(p, "/apps/", 6) == 0) {
        const char *base = strrchr(p, '/');
        if (base && !strrchr(base, '.')) return true;   /* /apps/<id> */
    }
    return false;
}

static void url_encode_path(const char *src, char *out, size_t cap) {
    static const char hex[] = "0123456789ABCDEF";
    size_t w = 0;
    for (const unsigned char *p = (const unsigned char *)src; *p && w + 4 < cap; p++) {
        unsigned char c = *p;
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '/') {
            out[w++] = (char)c;
        } else {
            out[w++] = '%';
            out[w++] = hex[c >> 4];
            out[w++] = hex[c & 15];
        }
    }
    out[w] = '\0';
}

static void auth_redirect_login(int fd, const char *path) {
    char enc[512], extra[700];
    url_encode_path(path, enc, sizeof(enc));
    snprintf(extra, sizeof(extra), "Location: /login?next=%s\r\n", enc);
    http_response_ex(fd, 302, "Found", "text/plain; charset=utf-8", extra,
                     "login required\n", false);
}

static void auth_unauthorized(int fd) {
    http_response_ex(fd, 401, "Unauthorized", "application/json; charset=utf-8",
                     "WWW-Authenticate: Bearer realm=\"cwick\"\r\n",
                     "{\"error\":{\"message\":\"authentication required\","
                     "\"type\":\"invalid_request_error\",\"code\":\"auth_required\"}}",
                     false);
}

/* POST /v1/auth/login. The login page sends JSON; an API client may send the
 * same body. `next` is validated to a same-site absolute path before it is
 * reflected, so a crafted form cannot turn the login page into an open
 * redirect. Returns the HTTP status; the caller only logs it. */
static int auth_login_response(server_ctx *srv, int fd, const http_req *req) {
    if (strcmp(req->method, "POST") != 0) {
        http_response(fd, 405, "Method Not Allowed",
                      "application/json; charset=utf-8",
                      "{\"error\":{\"message\":\"POST required\"}}", false);
        return 405;
    }
    if (!srv->auth) {
        http_response(fd, 200, "OK", "application/json; charset=utf-8",
                      "{\"status\":\"disabled\",\"next\":\"/\",\"enabled\":false}",
                      false);
        return 200;
    }
    cwick_error e;
    cwick_error_clear(&e);
    json_value *root = cwick_json_parse(req->body, req->body_len, &e);
    if (!root) {
        http_response(fd, 400, "Bad Request", "application/json; charset=utf-8",
                      "{\"error\":{\"message\":\"malformed JSON\"}}", false);
        return 400;
    }
    const char *pw = cwick_json_str(cwick_json_get(root, "password"));
    const char *next = cwick_json_str(cwick_json_get(root, "next"));
    if (!next || next[0] != '/' || next[1] == '/') next = "/";
    int retry = cwick_auth_retry_after(srv->auth);
    if (retry > 0) {
        char extra[128], body[192];
        snprintf(extra, sizeof(extra), "Retry-After: %d\r\n", retry);
        snprintf(body, sizeof(body),
                 "{\"error\":{\"message\":\"too many attempts; retry in %d seconds\"}}",
                 retry);
        http_response_ex(fd, 429, "Too Many Requests",
                         "application/json; charset=utf-8", extra, body, false);
        cwick_json_free(root);
        return 429;
    }
    char *token = cwick_auth_login(srv->auth, pw ? pw : "");
    if (!token) {
        http_response(fd, 401, "Unauthorized", "application/json; charset=utf-8",
                      "{\"error\":{\"message\":\"invalid password\"}}", false);
        cwick_json_free(root);
        return 401;
    }
    char extra[256], body[512];
    snprintf(extra, sizeof(extra),
             "Set-Cookie: " CWICK_AUTH_COOKIE "=%s; HttpOnly; SameSite=Strict; "
             "Path=/; Max-Age=43200\r\n", token);
    char *jnext = json_str_alloc(next);
    snprintf(body, sizeof(body),
             "{\"status\":\"ok\",\"next\":%s,\"token\":\"%s\",\"expires_in\":43200}",
             jnext ? jnext : "\"/\"", token);
    http_response_ex(fd, 200, "OK", "application/json; charset=utf-8", extra, body,
                     false);
    free(jnext);
    free(token);
    cwick_json_free(root);
    return 200;
}

static void auth_status_response(server_ctx *srv, int fd, const http_req *req) {
    char body[192];
    snprintf(body, sizeof(body),
             "{\"enabled\":%s,\"authenticated\":%s}",
             srv->auth ? "true" : "false",
             request_authenticated(srv, req) ? "true" : "false");
    http_response(fd, 200, "OK", "application/json; charset=utf-8", body, false);
}

static void auth_logout_response(server_ctx *srv, int fd, const http_req *req) {
    char token[128];
    if (srv->auth && request_auth_token(req, token, sizeof(token)))
        cwick_auth_logout(srv->auth, token);
    http_response_ex(fd, 200, "OK", "application/json; charset=utf-8",
                     "Set-Cookie: " CWICK_AUTH_COOKIE "=; HttpOnly; SameSite=Strict; "
                     "Path=/; Max-Age=0\r\n",
                     "{\"status\":\"logged_out\"}", false);
}

/* ------------------------------------------------------------------ */
/* app storage: GET /v1/store and /v1/apps/<id>/{state,events,games}    */
/* ------------------------------------------------------------------ */

static void sbuf_put_jstr(srv_buf *b, const char *s) {
    char *j = json_str_alloc(s ? s : "");
    sbuf_puts(b, j ? j : "\"\"");
    free(j);
}

/* Split `/v1/apps/<id>/<tail>`. The id is charset-checked by the store, so a
 * path like `/v1/apps/../../etc/state` or one with a quote in it is rejected
 * here rather than reaching sqlite. */
static bool app_path_parse(const char *ponly, char *app, size_t cap,
                           const char **tail) {
    static const char pre[] = "/v1/apps/";
    size_t plen = strlen(pre);
    if (strncmp(ponly, pre, plen) != 0) return false;
    const char *p = ponly + plen;
    const char *slash = strchr(p, '/');
    if (!slash) return false;
    size_t n = (size_t)(slash - p);
    if (n == 0 || n >= cap) return false;
    memcpy(app, p, n);
    app[n] = '\0';
    if (!cwick_store_id_ok(app)) return false;
    *tail = slash + 1;
    return true;
}

typedef struct { srv_buf *b; bool first; } store_out;

static void kv_each_cb(const char *k, const char *v, int64_t ts, void *user) {
    store_out *o = (store_out *)user;
    if (!o->first) sbuf_puts(o->b, ",");
    o->first = false;
    sbuf_puts(o->b, "{\"key\":");
    sbuf_put_jstr(o->b, k);
    sbuf_putf(o->b, ",\"updated_at\":%lld,\"value\":%s}", (long long)ts, v);
}

static void games_cb(const cwick_store_game *g, void *user) {
    store_out *o = (store_out *)user;
    if (!o->first) sbuf_puts(o->b, ",");
    o->first = false;
    sbuf_puts(o->b, "{\"game_id\":");
    sbuf_put_jstr(o->b, g->game_id);
    sbuf_putf(o->b, ",\"started_at\":%lld,\"ended_at\":%lld,\"moves\":%lld,"
                    "\"outcome\":",
              (long long)g->started_at, (long long)g->ended_at,
              (long long)g->moves);
    sbuf_put_jstr(o->b, g->outcome);
    sbuf_puts(o->b, "}");
}

static void events_cb(const cwick_store_event *e, void *user) {
    store_out *o = (store_out *)user;
    if (!o->first) sbuf_puts(o->b, ",");
    o->first = false;
    sbuf_putf(o->b, "{\"id\":%lld,\"seq\":%lld,\"ts\":%lld,\"game_id\":",
              (long long)e->id, (long long)e->seq, (long long)e->ts);
    sbuf_put_jstr(o->b, e->game_id);
    sbuf_puts(o->b, ",\"kind\":");
    sbuf_put_jstr(o->b, e->kind);
    sbuf_puts(o->b, ",\"payload\":");
    sbuf_puts(o->b, e->payload);
    sbuf_puts(o->b, "}");
}

static void store_fail(int fd, int code, const char *msg) {
    char *j = json_str_alloc(msg ? msg : "store unavailable");
    char body[512];
    snprintf(body, sizeof(body), "{\"error\":{\"message\":%s,"
             "\"type\":\"invalid_request_error\",\"code\":\"store_error\"}}",
             j ? j : "\"store unavailable\"");
    free(j);
    http_response(fd, code, code == 500 ? "Internal Server Error" : "Bad Request",
                  "application/json; charset=utf-8", body, false);
}

/* The database an app's rows live in. Normally `<dir>/<app>.db`, opened on
 * first use and cached: two browser tabs and a phone on the LAN must see the
 * same file, and opening it per request would re-run the schema every time.
 * Returns NULL when it cannot be opened (the caller answers 503, which is
 * what the apps read as "fall back to localStorage"). */
static cwick_store *store_for_app(server_ctx *srv, const char *app) {
    if (!srv->store) return NULL;
    if (srv->store_single) return srv->store;
    pthread_mutex_lock(&srv->store_mtx);
    cwick_store *found = NULL;
    for (int i = 0; i < srv->store_slots_n; i++) {
        if (strcmp(srv->store_slots[i].app, app) == 0) {
            found = srv->store_slots[i].st;
            break;
        }
    }
    if (!found && srv->store_slots_n <
            (int)(sizeof(srv->store_slots) / sizeof(srv->store_slots[0]))) {
        char path[1152];
        int n = snprintf(path, sizeof(path), "%s/%s.db", srv->store_dir, app);
        if (n > 0 && (size_t)n < sizeof(path)) {
            cwick_error e;
            cwick_error_clear(&e);
            cwick_store *st = cwick_store_open(path, &e);
            if (st) {
                snprintf(srv->store_slots[srv->store_slots_n].app,
                         sizeof(srv->store_slots[srv->store_slots_n].app),
                         "%s", app);
                srv->store_slots[srv->store_slots_n].st = st;
                srv->store_slots_n++;
                found = st;
                CWICK_INFO("apps", "opened per-app database %s", path);
            } else {
                CWICK_INFO("apps", "cannot open %s (%s) — this app's requests "
                                  "answer 503", path, cwick_error_message(&e));
            }
        }
    }
    pthread_mutex_unlock(&srv->store_mtx);
    return found;
}

/* Returns -1 when `ponly` is not an app-storage route, otherwise the HTTP
 * status to log. `path` keeps its query string, which is where `key`,
 * `game_id` and `limit` live. */
/* ------------------------------------------------------------------ */
/* API keys: /v1/auth/keys                                             */
/* ------------------------------------------------------------------ */

typedef struct { srv_buf *b; bool first; } keys_out;

static void keys_list_cb(const cwick_api_key_info *k, void *user) {
    keys_out *o = (keys_out *)user;
    if (!o->first) sbuf_puts(o->b, ",");
    o->first = false;
    sbuf_puts(o->b, "{\"id\":");
    sbuf_put_jstr(o->b, k->id);
    sbuf_puts(o->b, ",\"name\":");
    sbuf_put_jstr(o->b, k->name);
    sbuf_puts(o->b, ",\"prefix\":");
    sbuf_put_jstr(o->b, k->prefix);
    sbuf_putf(o->b, ",\"created_at\":%lld,\"last_used_at\":%lld,\"enabled\":%s}",
              (long long)k->created_at, (long long)k->last_used_at,
              k->enabled ? "true" : "false");
}

static void keys_unauthorized(int fd) {
    store_fail(fd, 401, "key management requires a login session, not an API key");
}

/* Returns -1 when `ponly` is not a key route.
 *
 * Only a *session* may manage keys, never an API key. That asymmetry is the
 * whole point: a key is meant to be pasted into a third-party agent's config,
 * where it will inevitably end up in a log, a backup or a screenshot. If such
 * a key could mint more keys, one leaked paste would be a foothold; as it is,
 * the worst it can do is be revoked from the console. */
static int keys_response(server_ctx *srv, int fd, const http_req *req,
                         const char *path, const char *ponly) {
    if (strcmp(ponly, "/v1/auth/keys") != 0) return -1;
    if (!srv->auth) {
        store_fail(fd, 404, "single-user login is disabled, so there are no keys");
        return 404;
    }
    if (!request_has_session(srv, req)) {
        keys_unauthorized(fd);
        return 401;
    }
    if (strcmp(req->method, "GET") == 0) {
        srv_buf b = {0};
        sbuf_puts(&b, "{\"keys\":[");
        keys_out o = { &b, true };
        cwick_auth_keys_each(srv->auth, keys_list_cb, &o);
        sbuf_puts(&b, "]}");
        http_response(fd, 200, "OK", "application/json; charset=utf-8",
                      b.p ? b.p : "{}", false);
        free(b.p);
        return 200;
    }
    if (strcmp(req->method, "POST") == 0) {
        cwick_error e;
        cwick_error_clear(&e);
        char nm[128];
        nm[0] = '\0';
        if (req->body && req->body_len) {
            json_value *root = cwick_json_parse(req->body, req->body_len, &e);
            if (root) {
                /* Copy before freeing: `name` points into the DOM. */
                const char *n = cwick_json_str(cwick_json_get(root, "name"));
                if (n) snprintf(nm, sizeof(nm), "%s", n);
                cwick_json_free(root);
            }
        }
        cwick_api_key_info info;
        char *secret = cwick_auth_key_create(srv->auth, nm[0] ? nm : NULL, &info);
        if (!secret) {
            store_fail(fd, 500, "no free key slot (32 keys is the limit)");
            return 500;
        }
        if (srv->store) {
            cwick_store_key row;
            memset(&row, 0, sizeof(row));
            row.id = info.id;
            row.name = info.name;
            row.hash_hex = info.hash_hex;
            row.prefix = info.prefix;
            row.created_at = info.created_at;
            row.enabled = true;
            if (!cwick_store_key_put(srv->store, &row, &e)) {
                /* Keep memory and disk in step: a key that was not persisted
                 * must not be accepted, or the operator would believe an agent
                 * is configured while a restart silently drops it. */
                cwick_auth_key_delete(srv->auth, info.id);
                store_fail(fd, 500, cwick_error_message(&e));
                memset(secret, 0, strlen(secret));
                free(secret);
                return 500;
            }
        }
        srv_buf b = {0};
        sbuf_puts(&b, "{\"id\":");
        sbuf_put_jstr(&b, info.id);
        sbuf_puts(&b, ",\"name\":");
        sbuf_put_jstr(&b, info.name);
        sbuf_puts(&b, ",\"prefix\":");
        sbuf_put_jstr(&b, info.prefix);
        sbuf_putf(&b, ",\"created_at\":%lld,\"secret\":", (long long)info.created_at);
        sbuf_put_jstr(&b, secret);
        sbuf_puts(&b, ",\"note\":\"shown once; the server keeps only its hash\"}");
        http_response(fd, 200, "OK", "application/json; charset=utf-8",
                      b.p ? b.p : "{}", false);
        free(b.p);
        /* The plaintext has left the building: do not leave it in a heap block
         * that outlives this request. */
        memset(secret, 0, strlen(secret));
        free(secret);
        return 200;
    }
    if (strcmp(req->method, "DELETE") == 0) {
        const char *raw = query_find(path, "id");
        char id[64];
        if (!raw || !query_decode(raw, id, sizeof(id)) || !*id) {
            store_fail(fd, 400, "DELETE needs ?id=");
            return 400;
        }
        if (!cwick_auth_key_delete(srv->auth, id)) {
            store_fail(fd, 404, "no key with that id");
            return 404;
        }
        if (srv->store) {
            cwick_error e;
            cwick_error_clear(&e);
            (void)cwick_store_key_del(srv->store, id, &e);
        }
        http_response(fd, 200, "OK", "application/json; charset=utf-8",
                      "{\"status\":\"revoked\"}", false);
        return 200;
    }
    store_fail(fd, 405, "keys supports GET, POST and DELETE");
    return 405;
}

/* Adopt the keys an earlier run persisted, so restarting cwick does not
 * invalidate what an agent platform was configured with. */
static void auth_keys_load_cb(const cwick_store_key *k, void *user) {
    cwick_auth *a = (cwick_auth *)user;
    cwick_auth_key_load(a, k->id, k->name, k->hash_hex, k->prefix,
                       k->created_at, k->last_used_at, k->enabled);
}

static void keys_count_cb(const cwick_api_key_info *k, void *user) {
    (void)k;
    int *n = (int *)user;
    (*n)++;
}

static int app_store_response(server_ctx *srv, int fd, const http_req *req,
                              const char *path, const char *app,
                              const char *tail) {
    cwick_store *st = store_for_app(srv, app);
    if (!st) {
        store_fail(fd, 503, "app storage is unavailable (see the startup log)");
        return 503;
    }
    const char *method = req->method;

    /* ---- /v1/apps/<id>/state ---------------------------------------- */
    if (strcmp(tail, "state") == 0) {
        if (strcmp(method, "GET") == 0) {
            const char *raw = query_find(path, "key");
            if (raw) {
                char key[160];
                if (!query_decode(raw, key, sizeof(key)) || !*key) {
                    store_fail(fd, 400, "bad key");
                    return 400;
                }
                char *v = cwick_store_kv_get(st, app, key);
                srv_buf b = {0};
                sbuf_puts(&b, "{\"app\":");
                sbuf_put_jstr(&b, app);
                sbuf_puts(&b, ",\"key\":");
                sbuf_put_jstr(&b, key);
                sbuf_puts(&b, ",\"value\":");
                sbuf_puts(&b, v ? v : "null");
                sbuf_puts(&b, "}");
                http_response(fd, 200, "OK", "application/json; charset=utf-8",
                              b.p ? b.p : "{}", false);
                free(b.p);
                free(v);
                return 200;
            }
            srv_buf b = {0};
            sbuf_puts(&b, "{\"app\":");
            sbuf_put_jstr(&b, app);
            sbuf_puts(&b, ",\"items\":[");
            store_out o = { &b, true };
            cwick_store_kv_each(st, app, kv_each_cb, &o);
            sbuf_puts(&b, "]}");
            http_response(fd, 200, "OK", "application/json; charset=utf-8",
                          b.p ? b.p : "{}", false);
            free(b.p);
            return 200;
        }
        if (strcmp(method, "DELETE") == 0) {
            const char *raw = query_find(path, "key");
            char key[160];
            if (!raw || !query_decode(raw, key, sizeof(key)) || !*key) {
                store_fail(fd, 400, "DELETE needs ?key=");
                return 400;
            }
            cwick_error e;
            cwick_error_clear(&e);
            if (!cwick_store_kv_del(st, app, key, &e)) {
                store_fail(fd, 500, cwick_error_message(&e));
                return 500;
            }
            http_response(fd, 200, "OK", "application/json; charset=utf-8",
                          "{\"status\":\"deleted\"}", false);
            return 200;
        }
        if (strcmp(method, "PUT") == 0 || strcmp(method, "POST") == 0) {
            cwick_error e;
            cwick_error_clear(&e);
            json_value *root = cwick_json_parse(req->body, req->body_len, &e);
            if (!root) {
                store_fail(fd, 400, "body must be a JSON object");
                return 400;
            }
            int written = 0;
            json_value *vals = cwick_json_get(root, "values");
            if (vals) {                       /* {"values":{k:v,...}} */
                for (size_t i = 0; i < vals->count; i++) {
                    char *txt = cwick_json_write(vals->members[i].value);
                    if (!txt) continue;
                    if (cwick_store_kv_put(st, app,
                                          vals->members[i].key, txt, &e)) written++;
                    free(txt);
                }
            } else {                          /* {"key":k,"value":v} */
                const char *key = cwick_json_str(cwick_json_get(root, "key"));
                json_value *val = cwick_json_get(root, "value");
                if (!key || !*key || !val) {
                    cwick_json_free(root);
                    store_fail(fd, 400, "body needs \"key\" and \"value\" "
                                        "(or \"values\")");
                    return 400;
                }
                char *txt = cwick_json_write(val);
                if (txt && cwick_store_kv_put(st, app, key, txt, &e)) written++;
                free(txt);
            }
            cwick_json_free(root);
            if (!written) {
                store_fail(fd, 400, cwick_error_message(&e));
                return 400;
            }
            srv_buf b = {0};
            sbuf_putf(&b, "{\"status\":\"ok\",\"written\":%d}", written);
            http_response(fd, 200, "OK", "application/json; charset=utf-8",
                          b.p ? b.p : "{}", false);
            free(b.p);
            return 200;
        }
        store_fail(fd, 405, "state supports GET, PUT and DELETE");
        return 405;
    }

    /* ---- /v1/apps/<id>/events --------------------------------------- */
    if (strcmp(tail, "events") == 0) {
        if (strcmp(method, "GET") == 0) {
            const char *raw = query_find(path, "game_id");
            char game[128];
            bool have_game = raw && query_decode(raw, game, sizeof(game)) && *game;
            int64_t limit = 0;
            const char *lraw = query_find(path, "limit");
            if (lraw) {
                char lbuf[32];
                if (query_decode(lraw, lbuf, sizeof(lbuf))) limit = atoll(lbuf);
            }
            srv_buf b = {0};
            sbuf_puts(&b, "{\"app\":");
            sbuf_put_jstr(&b, app);
            sbuf_puts(&b, ",\"events\":[");
            store_out o = { &b, true };
            cwick_store_events(st, app, have_game ? game : NULL, limit,
                              events_cb, &o);
            sbuf_puts(&b, "]}");
            http_response(fd, 200, "OK", "application/json; charset=utf-8",
                          b.p ? b.p : "{}", false);
            free(b.p);
            return 200;
        }
        if (strcmp(method, "POST") == 0) {
            cwick_error e;
            cwick_error_clear(&e);
            json_value *root = cwick_json_parse(req->body, req->body_len, &e);
            if (!root) {
                store_fail(fd, 400, "body must be a JSON object");
                return 400;
            }
            const char *game = cwick_json_str(cwick_json_get(root, "game_id"));
            const char *kind = cwick_json_str(cwick_json_get(root, "kind"));
            json_value *payload = cwick_json_get(root, "payload");
            int64_t seq = cwick_json_int(cwick_json_get(root, "seq"), 0);
            if (!game || !*game || !kind || !*kind || !payload) {
                cwick_json_free(root);
                store_fail(fd, 400, "body needs game_id, kind and payload");
                return 400;
            }
            char *txt = cwick_json_write(payload);
            bool ok = txt &&
                cwick_store_event_add(st, app, game, seq, kind, txt, &e);
            free(txt);
            cwick_json_free(root);
            if (!ok) {
                store_fail(fd, 400, cwick_error_message(&e));
                return 400;
            }
            http_response(fd, 200, "OK", "application/json; charset=utf-8",
                          "{\"status\":\"ok\"}", false);
            return 200;
        }
        store_fail(fd, 405, "events supports GET and POST");
        return 405;
    }

    /* ---- /v1/apps/<id>/games ---------------------------------------- */
    if (strcmp(tail, "games") == 0) {
        if (strcmp(method, "GET") == 0) {
            int64_t limit = 0;
            const char *lraw = query_find(path, "limit");
            if (lraw) {
                char lbuf[32];
                if (query_decode(lraw, lbuf, sizeof(lbuf))) limit = atoll(lbuf);
            }
            srv_buf b = {0};
            sbuf_puts(&b, "{\"app\":");
            sbuf_put_jstr(&b, app);
            sbuf_puts(&b, ",\"games\":[");
            store_out o = { &b, true };
            cwick_store_games(st, app, limit, games_cb, &o);
            sbuf_puts(&b, "]}");
            http_response(fd, 200, "OK", "application/json; charset=utf-8",
                          b.p ? b.p : "{}", false);
            free(b.p);
            return 200;
        }
        if (strcmp(method, "PUT") == 0 || strcmp(method, "POST") == 0) {
            cwick_error e;
            cwick_error_clear(&e);
            json_value *root = cwick_json_parse(req->body, req->body_len, &e);
            if (!root) {
                store_fail(fd, 400, "body must be a JSON object");
                return 400;
            }
            const char *game = cwick_json_str(cwick_json_get(root, "game_id"));
            const char *outcome = cwick_json_str(cwick_json_get(root, "outcome"));
            int64_t started = cwick_json_int(cwick_json_get(root, "started_at"), 0);
            int64_t ended = cwick_json_int(cwick_json_get(root, "ended_at"), 0);
            int64_t moves = cwick_json_int(cwick_json_get(root, "moves"), 0);
            if (!game || !*game) {
                cwick_json_free(root);
                store_fail(fd, 400, "body needs game_id");
                return 400;
            }
            bool ok = cwick_store_game_put(st, app, game, started,
                                          outcome, ended, moves, &e);
            /* A game row is also the app's "there is a match in progress":
             * record it so a half-finished game is still replayable. */
            if (ok && !outcome) {
                cwick_store_game_put(st, app, game,
                                    started ? started : (int64_t)time(NULL),
                                    NULL, 0, 0, &e);
            }
            cwick_json_free(root);
            if (!ok) {
                store_fail(fd, 400, cwick_error_message(&e));
                return 400;
            }
            http_response(fd, 200, "OK", "application/json; charset=utf-8",
                          "{\"status\":\"ok\"}", false);
            return 200;
        }
        if (strcmp(method, "DELETE") == 0) {
            const char *raw = query_find(path, "game_id");
            char game[128];
            if (!raw || !query_decode(raw, game, sizeof(game)) || !*game) {
                store_fail(fd, 400, "DELETE needs ?game_id=");
                return 400;
            }
            cwick_error e;
            cwick_error_clear(&e);
            if (!cwick_store_game_del(st, app, game, &e)) {
                store_fail(fd, 500, cwick_error_message(&e));
                return 500;
            }
            http_response(fd, 200, "OK", "application/json; charset=utf-8",
                          "{\"status\":\"deleted\"}", false);
            return 200;
        }
        store_fail(fd, 405, "games supports GET, PUT and DELETE");
        return 405;
    }

    return -1;   /* not a storage route */
}

/* GET /v1/store — where the apps' data lives and how much of it there is.
 * Returns the HTTP status so the access log can show the real answer.
 *
 * `?app=<id>` answers about *that* app's own database, with the flat
 * path/games/events/kv_rows fields the apps' badge reads; without it the same
 * fields describe the primary database (the API keys, and everything in the
 * legacy single-file layout) and `apps` lists the per-app files. */
static int store_status_response(server_ctx *srv, int fd, const char *path) {
    if (!srv->store) {
        store_fail(fd, 503, "app storage is unavailable (see the startup log)");
        return 503;
    }
    const char *araw = query_find(path, "app");
    char app[64] = "";
    bool have_app = false;
    if (araw) {
        char cand[64];
        if (query_decode(araw, cand, sizeof(cand)) && *cand &&
            cwick_store_id_ok(cand)) {
            snprintf(app, sizeof(app), "%s", cand);
            have_app = true;
        }
    }

    srv_buf b = {0};
    sbuf_puts(&b, "{\"enabled\":true,\"mode\":");
    sbuf_put_jstr(&b, srv->store_single ? "single" : "per-app");
    if (!srv->store_single) {
        sbuf_puts(&b, ",\"dir\":");
        sbuf_put_jstr(&b, srv->store_dir);
        sbuf_puts(&b, ",\"primary\":{\"path\":");
        sbuf_put_jstr(&b, cwick_store_path(srv->store) ? cwick_store_path(srv->store)
                                                      : "");
        sbuf_puts(&b, "}");
    }

    /* The database the flat fields describe. */
    cwick_store *st = srv->store;
    if (have_app && !srv->store_single) {
        st = store_for_app(srv, app);
        if (!st) {
            free(b.p);
            store_fail(fd, 503, "this app's database could not be opened");
            return 503;
        }
        sbuf_puts(&b, ",\"app\":");
        sbuf_put_jstr(&b, app);
    }
    cwick_store_stats sst;
    memset(&sst, 0, sizeof(sst));
    cwick_store_get_stats(st, &sst);
    sbuf_puts(&b, ",\"path\":");
    sbuf_put_jstr(&b, cwick_store_path(st) ? cwick_store_path(st) : "");
    sbuf_putf(&b, ",\"games\":%lld,\"events\":%lld,\"kv_rows\":%lld,"
                  "\"bytes\":%lld",
              (long long)sst.games, (long long)sst.events,
              (long long)sst.kv_rows, (long long)sst.bytes);

    if (!srv->store_single && !have_app) {
        sbuf_puts(&b, ",\"apps\":[");
        pthread_mutex_lock(&srv->store_mtx);
        for (int i = 0; i < srv->store_slots_n; i++) {
            cwick_store_stats a;
            memset(&a, 0, sizeof(a));
            cwick_store_get_stats(srv->store_slots[i].st, &a);
            if (i) sbuf_puts(&b, ",");
            sbuf_puts(&b, "{\"app\":");
            sbuf_put_jstr(&b, srv->store_slots[i].app);
            sbuf_puts(&b, ",\"path\":");
            sbuf_put_jstr(&b, cwick_store_path(srv->store_slots[i].st)
                                  ? cwick_store_path(srv->store_slots[i].st)
                                  : "");
            sbuf_putf(&b, ",\"games\":%lld,\"events\":%lld,\"kv_rows\":%lld}",
                      (long long)a.games, (long long)a.events,
                      (long long)a.kv_rows);
        }
        pthread_mutex_unlock(&srv->store_mtx);
        sbuf_puts(&b, "]");
    }
    sbuf_puts(&b, "}");
    http_response(fd, 200, "OK", "application/json; charset=utf-8",
                  b.p ? b.p : "{}", false);
    free(b.p);
    return 200;
}

static void handle_client(server_ctx *srv, int fd, uint64_t rid);

static void *client_thread(void *arg) {
    client_job *j = (client_job *)arg;
    server_ctx *srv = j->srv;
    /* Every line this thread emits while serving the connection carries
     * `[req N]`, including the ones from the engine and the model. Cleared
     * afterwards so the pool's own bookkeeping lines are not attributed. */
    cwick_log_set_request(j->rid);
    handle_client(srv, j->fd, j->rid);
    cwick_log_set_request(0);
    pthread_mutex_lock(&srv->client_mtx);
    if (--srv->clients <= 0) {
        srv->clients = 0;
        pthread_cond_broadcast(&srv->client_cv);
    }
    pthread_mutex_unlock(&srv->client_mtx);
    free(j);
    return NULL;
}

/* Register a new in-flight client. Returns false when the process is at
 * CWICK_MAX_CLIENTS, in which case the caller must answer 503 itself. */
static bool client_register(server_ctx *srv) {
    pthread_mutex_lock(&srv->client_mtx);
    bool ok = srv->clients < CWICK_MAX_CLIENTS;
    if (ok) srv->clients++;
    pthread_mutex_unlock(&srv->client_mtx);
    return ok;
}

static void handle_client(server_ctx *srv, int fd, uint64_t rid) {
    cwick_error err;
    cwick_error_clear(&err);
    /* Use the configured body limit (plus HTTP headroom) consistently rather
     * than silently enforcing a second hard-coded 8 MiB cap. */
    if (srv->max_request_bytes <= 0 ||
        (uint64_t)srv->max_request_bytes > SIZE_MAX - 4096u) {
        cwick_error_set(&err, CWICK_ERR_ARGUMENT, "invalid request byte limit");
        close(fd);
        return;
    }
    size_t request_cap = (size_t)srv->max_request_bytes + 4096u;
    char *rbuf = cwick_xmalloc(request_cap);
    http_req req;
    memset(&req, 0, sizeof(req));
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    int status = 0;
    char detail[256];
    detail[0] = '\0';

    /* Read timeout so a trickling client cannot pin the process through
     * shutdown (see the poll loop in cwick_server_run). */
    {
        struct timeval tv;
        tv.tv_sec = CWICK_CLIENT_READ_TIMEOUT_MS / 1000;
        tv.tv_usec = (CWICK_CLIENT_READ_TIMEOUT_MS % 1000) * 1000;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    }

#define DONE(code) do { \
        gettimeofday(&t1, NULL); \
        status = (code); \
        double ms = (double)(t1.tv_sec - t0.tv_sec) * 1e3 + \
                    (double)(t1.tv_usec - t0.tv_usec) / 1e3; \
        access_log(rid, req.method[0] ? req.method : "?", \
                   req.path[0] ? req.path : "-", status, ms, \
                   detail[0] ? detail : NULL, \
                   tls_auth_who[0] ? tls_auth_who : NULL); \
        free(rbuf); \
        close(fd); \
        return; \
    } while (0)

    if (read_http_request(fd, rbuf, request_cap, &req, &err) != 0) {
        /* A body over the configured cap is not a malformed request: it is a
         * request this server will not read, and the reference protocols say
         * 413 for that (a client retrying the same body gets the same 400
         * forever, one that reads the status can stop). */
        const char *why = cwick_error_message(&err);
        const bool too_big = strstr(why, "too large") != NULL ||
                             strstr(why, "exceeds") != NULL;
        const int code = too_big ? 413 : 400;
        char body[512];
        snprintf(body, sizeof(body),
                 "{\"error\":{\"message\":\"bad request: %s\",\"type\":\"invalid_request_error\",\"param\":null,\"code\":null}}",
                 why);
        http_response(fd, code, too_big ? "Payload Too Large" : "Bad Request",
                      "application/json; charset=utf-8", body, false);
        snprintf(detail, sizeof(detail), "invalid request (%s)", why);
        DONE(code);
    }
    /* Probes skip the request line: access_log already prints one line for
     * them (at trace when it succeeded), and two lines per scrape is what made
     * the log unreadable. */
    if (!probe_path(req.path))
        CWICK_VERB(req_tag(rid), "%s %s (%lld bytes body)", req.method, req.path,
                   (long long)req.content_length);

    const char *path = req.path;
    char ponly[256];
    path_only(path, ponly, sizeof(ponly));

    /* SDK base URLs disagree about who owns the "/v1": Anthropic's client
     * appends "/v1/messages" to whatever ANTHROPIC_BASE_URL it is handed, so a
     * base URL that already ends in /v1 arrives as "/v1/v1/messages" and used
     * to 404 on every route below. Collapse the repeated prefix so the
     * misconfiguration is harmless; req.path keeps the raw text, which is what
     * the access log and the trace line print. */
    while (strncmp(ponly, "/v1/v1/", 7) == 0) {
        const size_t n = strlen(ponly + 3);
        memmove(ponly, ponly + 3, n + 1);
    }

    /* ---- single-user auth --------------------------------------------
     * Login/status/logout are public; the login page is served only to an
     * unauthenticated browser. Everything else is gated below: static CSS/JS
     * and the app manifest are public (they contain no model data), but HTML
     * navigation is redirected to /login and API calls get a Bearer challenge.
     */
    if (strcmp(ponly, "/v1/auth/login") == 0) {
        int code = auth_login_response(srv, fd, &req);
        snprintf(detail, sizeof(detail), "auth login -> %d", code);
        DONE(code);
    }
    if (strcmp(ponly, "/v1/auth/status") == 0) {
        auth_status_response(srv, fd, &req);
        snprintf(detail, sizeof(detail), "auth status");
        DONE(200);
    }
    if (strcmp(ponly, "/v1/auth/logout") == 0) {
        auth_logout_response(srv, fd, &req);
        snprintf(detail, sizeof(detail), "auth logout");
        DONE(200);
    }
    {
        int code = keys_response(srv, fd, &req, path, ponly);
        if (code >= 0) {
            snprintf(detail, sizeof(detail), "api keys -> %d", code);
            DONE(code);
        }
    }
    if (path_is_login_page(ponly)) {
        if (!srv->auth || request_authenticated(srv, &req)) {
            http_response_ex(fd, 302, "Found", "text/plain; charset=utf-8",
                             "Location: /\r\n", "already authenticated\n", false);
            snprintf(detail, sizeof(detail), "login page -> /");
            DONE(302);
        }
        const cwick_webui_blob *login = srv->web ? cwick_webui_lookup(ponly) : NULL;
        if (login) {
            http_response(fd, 200, "OK", login->content_type,
                          (const char *)login->data, false);
            snprintf(detail, sizeof(detail), "login page (%zu bytes)", login->len);
            DONE(200);
        }
        http_response(fd, 404, "Not Found", "text/plain; charset=utf-8",
                      "login page unavailable\n", false);
        snprintf(detail, sizeof(detail), "login page missing");
        DONE(404);
    }
    if (srv->auth && !request_authenticated(srv, &req)) {
        if (path_is_public_api(ponly) || path_is_public_asset(ponly)) {
            /* fall through to the normal route */
        } else if (path_is_navigation(req.method, ponly)) {
            auth_redirect_login(fd, path);
            snprintf(detail, sizeof(detail), "auth redirect -> /login");
            DONE(302);
        } else {
            auth_unauthorized(fd);
            snprintf(detail, sizeof(detail), "auth required");
            DONE(401);
        }
    }

    /* Every local model this instance knows about. The resident one is first
     * (what an OpenAI client reading data[0] expects); each entry carries the
     * cwick extras — where its weights are and whether it is the one loaded —
     * because the built-in UI needs them and nobody else is harmed. */
    if (strcmp(req.method, "GET") == 0 &&
        (strcmp(ponly, "/v1/models") == 0 || strcmp(ponly, "/models") == 0)) {
        char *body = models_json(srv);
        if (!body) {
            http_response(fd, 500, "Internal Server Error",
                          "application/json; charset=utf-8",
                          "{\"error\":{\"message\":\"model list unavailable\"}}",
                          false);
            DONE(500);
        }
        http_response(fd, 200, "OK", "application/json; charset=utf-8", body,
                      false);
        snprintf(detail, sizeof(detail), "%zu known model(s)",
                 srv->catalog.n);
        free(body);
        DONE(200);
    }

    /* GET /v1/models/<id> — the OpenAI retrieve-model call. Several agent
     * frameworks probe it once with the name from their config, so a missing
     * one reads as "your model id is wrong" long before a completion is
     * attempted. `switch` is the control endpoint above, not a model, and a
     * second slash means this is not a single id. */
    if (strcmp(req.method, "GET") == 0 &&
        (strncmp(ponly, "/v1/models/", 11) == 0 ||
         strncmp(ponly, "/models/", 8) == 0)) {
        const char *seg = ponly[4] == 'm' ? ponly + 11 : ponly + 8;
        if (*seg && !strchr(seg, '/') && strcmp(seg, "switch") != 0) {
            char mid[128];
            url_decode_segment(seg, mid, sizeof(mid));
            char *body = model_json_one(srv, mid);
            if (!body) {
                char msg[256];
                snprintf(msg, sizeof(msg),
                         "{\"error\":{\"message\":\"model '%.100s' does not "
                         "exist\",\"type\":\"invalid_request_error\","
                         "\"param\":\"model\",\"code\":\"model_not_found\"}}",
                         mid);
                http_response(fd, 404, "Not Found",
                              "application/json; charset=utf-8", msg, false);
                snprintf(detail, sizeof(detail), "unknown model '%.60s'", mid);
                DONE(404);
            }
            http_response(fd, 200, "OK", "application/json; charset=utf-8",
                          body, false);
            snprintf(detail, sizeof(detail), "model '%.60s'", mid);
            free(body);
            DONE(200);
        }
    }

    /* The server's own configuration, for the UI's settings page. Read-only:
     * the run was started with these values and changing a lifetime choice
     * (context window, memory fraction) is a restart, not a toggle. What can
     * be changed at runtime — which model is resident — has its own endpoint. */
    if (strcmp(req.method, "GET") == 0 &&
        (strcmp(ponly, "/v1/config") == 0 || strcmp(ponly, "/config") == 0)) {
        char *body = config_json(srv);
        if (!body) {
            http_response(fd, 500, "Internal Server Error",
                          "application/json; charset=utf-8",
                          "{\"error\":{\"message\":\"configuration unavailable\"}}",
                          false);
            DONE(500);
        }
        http_response(fd, 200, "OK", "application/json; charset=utf-8", body,
                      false);
        snprintf(detail, sizeof(detail), "%zu bytes of server config",
                 strlen(body));
        free(body);
        DONE(200);
    }

    /* Model library (search / inventory / downloads). These never take a
     * generation slot, so a long catalog request cannot stall a chat turn. */
    {
        int hc = hub_route(srv, fd, req.method, path, ponly, req.body,
                           req.body_len, detail, sizeof(detail));
        if (hc >= 0) DONE(hc);
    }

    /* Switch the resident model. Refused (409) while anything is generating,
     * queued or holding the tokenizer, because the swap frees those exact
     * objects; answered before the generation slot, like every other control
     * endpoint. */
    if (strcmp(req.method, "POST") == 0 &&
        (strcmp(ponly, "/v1/models/switch") == 0 ||
         strcmp(ponly, "/models/switch") == 0)) {
        json_value *root = cwick_json_parse(req.body, req.body_len, &err);
        if (!root) {
            http_response(fd, 400, "Bad Request", "application/json; charset=utf-8",
                          "{\"error\":{\"message\":\"invalid JSON\",\"type\":\"invalid_request_error\"}}",
                          false);
            DONE(400);
        }
        json_value *m = cwick_json_get(root, "model");
        if (!m || m->kind != JSON_STRING) {
            cwick_json_free(root);
            http_response(fd, 400, "Bad Request", "application/json; charset=utf-8",
                          "{\"error\":{\"message\":\"body.model must be a string\",\"type\":\"invalid_request_error\"}}",
                          false);
            DONE(400);
        }
        char wanted[CWICK_MODEL_ID_MAX];
        snprintf(wanted, sizeof(wanted), "%s", cwick_json_str(m));
        cwick_json_free(root);

        const cwick_model_entry *target = NULL;
        for (size_t i = 0; i < srv->catalog.n; i++) {
            /* Either name resolves: the served alias /v1/models advertises or
             * the checkpoint's own model_type. */
            if (strcmp(srv->catalog.v[i].id, wanted) == 0 ||
                (srv->catalog.v[i].alias[0] &&
                 strcmp(srv->catalog.v[i].alias, wanted) == 0)) {
                target = &srv->catalog.v[i];
                break;
            }
        }
        /* A checkpoint downloaded after startup was not in the startup scan.
         * Register it on demand instead of asking for a restart: the library
         * only reports a path when config.json exists and no .incomplete file
         * is left, so a half-downloaded model cannot become switchable. */
        if (!target && srv->hub) {
            char *fresh = cwick_hub_local_path(srv->hub, wanted);
            if (fresh) {
                cwick_models_list_add(&srv->catalog, wanted, fresh);
                for (size_t i = 0; i < srv->catalog.n; i++) {
                    if (strcmp(srv->catalog.v[i].root, fresh) == 0) {
                        target = &srv->catalog.v[i];
                        break;
                    }
                }
                free(fresh);
            }
        }
        if (!target) {
            char body[512];
            snprintf(body, sizeof(body),
                     "{\"error\":{\"message\":\"no local model named %s (see GET /v1/models)\","
                     "\"type\":\"invalid_request_error\",\"param\":\"model\",\"code\":\"model_not_found\"}}",
                     wanted);
            http_response(fd, 404, "Not Found", "application/json; charset=utf-8",
                          body, false);
            snprintf(detail, sizeof(detail), "unknown model %s", wanted);
            DONE(404);
        }
        /* No-op switch: say so instead of paying a reload. */
        pthread_mutex_lock(&srv->gen_lock);
        const cwick_model_entry *cur = resident_entry(srv);
        bool same = cur && strcmp(cur->root, target->root) == 0;
        pthread_mutex_unlock(&srv->gen_lock);
        if (same) {
            char body[512];
            char *jid = json_str_alloc(served_name(target));
            snprintf(body, sizeof(body),
                     "{\"status\":\"already_resident\",\"model\":%s}",
                     jid ? jid : "\"\"");
            free(jid);
            http_response(fd, 200, "OK", "application/json; charset=utf-8", body,
                          false);
            snprintf(detail, sizeof(detail), "already resident: %s",
                     served_name(target));
            DONE(200);
        }

        char msg[512];
        int status = srv_switch_model(srv, target, msg, sizeof(msg));
        char *jmsg = json_str_alloc(msg);
        char *jid = json_str_alloc(served_name(target));
        char body[1024];
        snprintf(body, sizeof(body),
                 "{\"status\":%s,\"model\":%s,\"detail\":%s}",
                 status == 200 ? "\"switched\"" : "\"error\"",
                 jid ? jid : "\"\"", jmsg ? jmsg : "\"\"");
        free(jid);
        free(jmsg);
        http_response(fd, status, status_text_for(status),
                      "application/json; charset=utf-8", body, false);
        snprintf(detail, sizeof(detail), "model switch -> %d", status);
        DONE(status);
    }

    /* ---- built-in chat UI (--web) ------------------------------------
     * Served from the binary, on the same origin as the API: the page POSTs
     * to /v1/chat/completions with no CORS preflight and no proxy in front.
     * Like the metrics endpoint it answers without touching the generation
     * slot — opening a second tab must never queue behind a 10-minute decode. */
    if (srv->web && strcmp(req.method, "GET") == 0) {
        const cwick_webui_blob *asset = cwick_webui_lookup(path);
        if (asset) {
            http_response(fd, 200, "OK", asset->content_type,
                          (const char *)asset->data, false);
            snprintf(detail, sizeof(detail), "web ui %s (%zu bytes)",
                     asset->path, asset->len);
            DONE(200);
        }
    }

    /* ---- LAN share / QR ----------------------------------------------
     * The app's share button asks this process for an address a phone on the
     * same Wi-Fi can reach, then renders /v1/share/qr as a PNG. Both are
     * answered before the generation slot, like metrics. */
    {
        char probe[256];
        path_only(path, probe, sizeof(probe));
        bool wants_share = strcmp(req.method, "GET") == 0 &&
            (strcmp(probe, "/v1/share") == 0 || strcmp(probe, "/share") == 0);
        bool wants_qr = strcmp(req.method, "GET") == 0 &&
            (strcmp(probe, "/v1/share/qr") == 0 || strcmp(probe, "/share/qr") == 0);
        if (wants_share) {
            char lan[64] = "";
            bool have_lan = cwick_lan_ipv4(lan, sizeof(lan));
            bool bind_all = strcmp(srv->host, "0.0.0.0") == 0;
            bool shareable = have_lan &&
                (bind_all || strcmp(srv->host, lan) == 0);
            /* ?app=<id> so each app can share itself: the QR has to encode
             * *that* app's entry, not whichever one happens to be first. The
             * id is charset-checked before it reaches printf. */
            char app_id[80] = "tictactoe";
            bool id_ok = true;
            const char *qa = query_find(path, "app");
            if (qa) {
                size_t n = 0;
                for (const char *p = qa; *p && *p != '&' && n + 1 < sizeof(app_id);
                     p++) {
                    char c = *p;
                    bool good = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                                (c >= '0' && c <= '9') || c == '_' || c == '-';
                    if (!good) { id_ok = false; break; }
                    app_id[n++] = c;
                }
                app_id[n] = '\0';
                if (!n) id_ok = false;
            }
            char body[1024];
            if (shareable && id_ok) {
                snprintf(body, sizeof(body),
                    "{\"bind\":\"%s\",\"port\":%d,\"lan_ip\":\"%s\","
                    "\"shareable\":true,\"lan_urls\":[\"http://%s:%d\"],"
                    "\"app\":\"%s\","
                    "\"app_url\":\"http://%s:%d/apps/%s/index.html\"}",
                    srv->host, srv->port, lan, lan, srv->port, app_id,
                    lan, srv->port, app_id);
            } else {
                snprintf(body, sizeof(body),
                    "{\"bind\":\"%s\",\"port\":%d,\"lan_ip\":\"%s\","
                    "\"shareable\":false,\"lan_urls\":[],"
                    "\"app\":\"%s\",\"app_url\":\"\","
                    "\"warning\":\"%s\"}",
                    srv->host, srv->port, have_lan ? lan : "", app_id,
                    !id_ok
                        ? "unknown or malformed app id"
                        : "listener is loopback-only or has no LAN IPv4; "
                          "restart with --lan (or --host 0.0.0.0) to share on "
                          "the local network");
            }
            http_response(fd, 200, "OK", "application/json; charset=utf-8",
                          body, false);
            snprintf(detail, sizeof(detail), "share info (%s)",
                     shareable ? "lan" : "loopback");
            DONE(200);
        }
        if (wants_qr) {
            const char *raw = query_find(path, "text");
            char text[512];
            if (!raw || !query_decode(raw, text, sizeof(text)) ||
                (strncmp(text, "http://", 7) != 0 &&
                 strncmp(text, "https://", 8) != 0)) {
                http_response(fd, 400, "Bad Request", "application/json; charset=utf-8",
                              "{\"error\":{\"message\":\"text must be an http(s) URL\"}}",
                              false);
                DONE(400);
            }
            size_t png_len = 0;
            unsigned char *png = cwick_qr_png(text, &png_len);
            if (!png) {
                http_response(fd, 500, "Internal Server Error",
                              "text/plain; charset=utf-8", "qr unavailable\n", false);
                DONE(500);
            }
            http_response_bytes(fd, 200, "OK", "image/png", png, png_len);
            cwick_qr_png_free(png);
            snprintf(detail, sizeof(detail), "qr (%zu bytes)", png_len);
            DONE(200);
        }
    }

    /* ---- app storage -------------------------------------------------
     * The apps' own data (finished games, the move-by-move replay log, small
     * preferences) lives in one sqlite file next to the server rather than in
     * each browser's localStorage: a browser profile is not a database, and
     * "clear site data" should not erase a match history the model produced.
     * Answered before the generation slot, like metrics — saving a move must
     * not queue behind a decode. */
    if (strcmp(req.method, "GET") == 0 && strcmp(ponly, "/v1/store") == 0) {
        int code = store_status_response(srv, fd, path);
        snprintf(detail, sizeof(detail), "store status -> %d", code);
        DONE(code);
    }
    {
        char st_app[CWICK_STORE_MAX_APP_ID + 8];
        const char *st_tail = NULL;
        if (app_path_parse(ponly, st_app, sizeof(st_app), &st_tail)) {
            int code = app_store_response(srv, fd, &req, path, st_app, st_tail);
            if (code >= 0) {
                snprintf(detail, sizeof(detail), "store %s/%s -> %d", st_app,
                         st_tail, code);
                DONE(code);
            }
        }
    }

    /* ---- observability endpoints -------------------------------------
     * vLLM serves Prometheus metrics from the same port as the OpenAI API,
     * so a dashboard pointed at a vLLM instance keeps working against cwick.
     * These are answered before the generation-slot queue, because a metrics
     * scrape must never be told "server is busy". */
    bool is_metrics = strcmp(req.method, "GET") == 0 &&
        (strcmp(path, "/metrics") == 0 || strcmp(path, "/v1/metrics") == 0);
    bool is_health = strcmp(req.method, "GET") == 0 &&
        (strcmp(path, "/health") == 0 || strcmp(path, "/healthz") == 0);
    if (is_metrics) {
        size_t mlen = 0;
        char *mbody = cwick_metrics_render(&mlen);
        if (!mbody) {
            http_response(fd, 500, "Internal Server Error",
                          "text/plain; charset=utf-8", "metrics unavailable\n", false);
            DONE(500);
        }
        http_response(fd, 200, "OK", cwick_metrics_content_type(), mbody, false);
        free(mbody);
        snprintf(detail, sizeof(detail), "%llu bytes of metrics",
                 (unsigned long long)mlen);
        DONE(200);
    }
    if (is_health) {
        http_response(fd, 200, "OK", "application/json; charset=utf-8",
                      "{\"status\":\"ok\"}", false);
        DONE(200);
    }

    /* ---- this process's own stdout/stderr ---------------------------
     * GET /v1/logs             one JSON snapshot (`since`, `limit`)
     * GET /v1/logs/stream      SSE tail of the same ring (`since`) */
    {
        char probe[256];
        path_only(path, probe, sizeof(probe));
        bool wants_tail = strcmp(probe, "/v1/logs") == 0 || strcmp(probe, "/logs") == 0;
        bool wants_stream = strcmp(probe, "/v1/logs/stream") == 0 ||
                            strcmp(probe, "/logs/stream") == 0;
        if (strcmp(req.method, "GET") == 0 && (wants_tail || wants_stream)) {
            uint64_t since;
            {
                int64_t asked = query_int(path, "since", -1);
                uint64_t head = cwick_logcap_head();
                uint64_t oldest = cwick_logcap_oldest();
                if (asked < 0) {
                    /* No cursor: hand over the recent tail rather than every
                     * line since the process started. When fewer lines than
                     * the backlog have been printed, that is *all* of them. */
                    since = head > (uint64_t)CWICK_LOGS_BACKLOG
                                ? head - CWICK_LOGS_BACKLOG
                                : oldest;
                } else {
                    since = (uint64_t)asked;
                }
                if (since < oldest) since = oldest;   /* evicted: cannot serve */
                if (since > head) since = head;
            }
            size_t limit = (size_t)query_int(path, "limit",
                                             (int64_t)CWICK_LOGS_BACKLOG);
            if (limit > 5000) limit = 5000;

            if (wants_stream) {
                logs_stream(srv, fd, since, limit);
                snprintf(detail, sizeof(detail), "log stream since %llu",
                         (unsigned long long)since);
                DONE(200);
            }
            char *body = logs_json_body(since, limit);
            if (!body) {
                http_response(fd, 500, "Internal Server Error",
                              "application/json; charset=utf-8",
                              "{\"error\":{\"message\":\"log buffer unavailable\"}}",
                              false);
                DONE(500);
            }
            http_response(fd, 200, "OK", "application/json; charset=utf-8", body,
                          false);
            snprintf(detail, sizeof(detail), "log tail from %llu (%zu bytes)",
                     (unsigned long long)since, strlen(body));
            free(body);
            DONE(200);
        }
    }

    /* What a decision would use right now. A model switch changes the prompt
     * grammar and the readable option set, so the answer here is not a static
     * capability list: it is the adapter resolved against the resident
     * checkpoint. Clients that cache a model id across a switch can compare
     * this against what they expect instead of guessing from a failed call. */
    if (srv->jev && strcmp(req.method, "GET") == 0 && is_systemone_path(ponly)) {
        pthread_mutex_lock(&srv->gen_lock);
        if (!srv->eng[0] || !srv->tok) {
            pthread_mutex_unlock(&srv->gen_lock);
            http_response(fd, 503, "Service Unavailable",
                          "application/json; charset=utf-8",
                          "{\"error\":{\"message\":\"no model is loaded\","
                          "\"type\":\"server_error\",\"param\":null,"
                          "\"code\":\"no_model\"}}", false);
            snprintf(detail, sizeof(detail), "no resident model");
            DONE(503);
        }
        char *body = NULL;
        char jerr[256];
        int jrc = jev_describe(srv->eng[0], srv->tok, &body, jerr, sizeof(jerr));
        pthread_mutex_unlock(&srv->gen_lock);
        if (jrc != 0) {
            char *emsg = cwick_json_stringify(jerr);
            char errbody[512];
            snprintf(errbody, sizeof(errbody),
                     "{\"error\":{\"message\":%s,\"type\":\"server_error\","
                     "\"param\":null,\"code\":null}}", emsg);
            free(emsg);
            http_response(fd, 500, "Internal Server Error",
                          "application/json; charset=utf-8", errbody, false);
            snprintf(detail, sizeof(detail), "decision adapter refused: %.150s", jerr);
            DONE(500);
        }
        http_response(fd, 200, "OK", "application/json; charset=utf-8", body, false);
        snprintf(detail, sizeof(detail), "decision adapter described");
        free(body);
        DONE(200);
    }

    /* The decision endpoint. One prefill per question and no decode: each
     * answer is read out of the distribution at the answer position rather
     * than generated, so a question costs a prefill and no tokens. It takes a
     * generation slot because it drives the same engine and tokenizer; on a
     * single-slot server a decision and a chat turn cannot overlap, and
     * pretending otherwise would have them reset each other's state. */
    if (srv->jev && strcmp(req.method, "POST") == 0 && is_systemone_path(ponly)) {
        if (!srv->eng[0] || !srv->tok) {
            http_response_ex(fd, 503, "Service Unavailable",
                             "application/json; charset=utf-8", "Retry-After: 5\r\n",
                             "{\"error\":{\"message\":\"no model is loaded\","
                             "\"type\":\"server_error\",\"param\":null,"
                             "\"code\":\"no_model\"}}", false);
            snprintf(detail, sizeof(detail), "no resident model");
            DONE(503);
        }
        double queue_ms = 0.0;
        int slot = slot_acquire(srv, fd, srv->stop_owner, &queue_ms);
        if (slot < 0) {
            bool swapping = slot == -3;
            http_response(fd, 503, "Service Unavailable",
                          "application/json; charset=utf-8",
                          swapping
                          ? "{\"error\":{\"message\":\"a model switch is in progress\","
                            "\"type\":\"server_error\",\"param\":null,\"code\":null}}"
                          : "{\"error\":{\"message\":\"the request queue is full\","
                            "\"type\":\"server_error\",\"param\":null,\"code\":null}}",
                          false);
            snprintf(detail, sizeof(detail), "decision refused: %s",
                     swapping ? "model switch" : "queue full");
            DONE(503);
        }
        char *body = NULL;
        int64_t in_tokens = 0;
        char verr[256];
        /* Bind the engine's log lines to this request. A decision reaches the
         * model through cwick_engine_prefill_logits directly, and that is
         * where its prefill progress and timing lines come from; without the
         * tag they say "gen" and cannot be attributed to a request. */
        cwick_engine_set_req_tag(srv->eng[0], req_tag(rid));
        int rc = jev_decide(srv->eng[0], srv->tok, req.body, req.body_len,
                            &body, &in_tokens, verr, sizeof(verr));
        cwick_engine_set_req_tag(srv->eng[0], NULL);
        slot_release(srv, slot);
        if (rc != 0) {
            /* Cover every early refusal (bad JSON, unknown model, a broken
               adapter, a prefill failure) with one line. The successful path
               logs from inside jev_decide; this is the other half. */
            CWICK_INFO(req_tag(rid), "systemone: decision refused — %s", verr);
            /* Two classes, as the protocol defines them: -2 is "this is not a
             * decision request" (malformed JSON, a missing field, an invalid
             * question), anything else is "well-formed but not servable"
             * (a stale model id, a limit, an unsupported field, a failed
             * prefill). */
            const int code = rc == -2 ? 422 : 400;
            char *emsg = cwick_json_stringify(verr);
            char errbody[1024];
            snprintf(errbody, sizeof(errbody),
                     "{\"error\":{\"message\":%s,\"type\":\"invalid_request_error\","
                     "\"param\":null,\"code\":null}}", emsg);
            free(emsg);
            http_response(fd, code, code == 422 ? "Unprocessable Entity"
                                               : "Bad Request",
                          "application/json; charset=utf-8", errbody, false);
            snprintf(detail, sizeof(detail), "decision refused: %.150s", verr);
            DONE(code);
        }
        http_response(fd, 200, "OK", "application/json; charset=utf-8", body, false);
        snprintf(detail, sizeof(detail),
                 "decision: %lld input tokens, %.0f ms queued",
                 (long long)in_tokens, queue_ms);
        free(body);
        DONE(200);
    }

    /* Compare the query-stripped form: the Anthropic client puts "?beta=true"
     * on every /v1/messages call, and matching the raw path here (as this
     * block once did) turned each of those into a 404 "unknown endpoint". */
    bool is_chat = strcmp(req.method, "POST") == 0 &&
        (strcmp(ponly, "/v1/chat/completions") == 0 || strcmp(ponly, "/chat/completions") == 0);
    bool is_comp = strcmp(req.method, "POST") == 0 &&
        (strcmp(ponly, "/v1/completions") == 0 || strcmp(ponly, "/completions") == 0);
    bool is_resp = strcmp(req.method, "POST") == 0 &&
        (strcmp(ponly, "/v1/responses") == 0 || strcmp(ponly, "/responses") == 0);
    bool is_msgs = strcmp(req.method, "POST") == 0 &&
        (strcmp(ponly, "/v1/messages") == 0 || strcmp(ponly, "/messages") == 0);
    const bool is_gen = is_chat || is_comp || is_resp || is_msgs;
    /* Which envelope an error from here on wears. */
    const srv_api api = is_msgs ? SRV_API_MESSAGES
                      : is_resp ? SRV_API_RESPONSES
                                : SRV_API_CHAT;

    /* Every generation request is accounted for from here on, and exactly one
     * of admit/refuse/error closes it — so `waiting` can never drift. */
    if (is_gen) (void)cwick_metrics_request_arrive();

    /* No resident model. Only reachable if a switch's replacement failed to
     * load: every other path either has an engine or is a 404 above. Tell the
     * client how to fix it instead of dereferencing a NULL spec. The tokenizer
     * is checked too: an engine without one cannot encode a prompt, and a
     * switch can leave exactly that state for a moment. */
    if (is_gen && (!srv->eng[0] || !srv->tok)) {
        char *body = api_error_body(api, E_SERVER,
            "\"no model is loaded (a model switch failed to load its "
            "replacement); POST /v1/models/switch with one of the ids from "
            "GET /v1/models\"", "\"no_model\"");
        http_response_ex(fd, 503, "Service Unavailable",
                         "application/json; charset=utf-8", "Retry-After: 5\r\n",
                         body, false);
        free(body);
        snprintf(detail, sizeof(detail), "no resident model");
        cwick_metrics_request_error("no_model");
        DONE(503);
    }

    if (!is_gen) {
        http_response(fd, 404, "Not Found", "application/json; charset=utf-8",
                      "{\"error\":{\"message\":\"unknown endpoint\",\"type\":\"invalid_request_error\",\"param\":null,\"code\":null}}", false);
        DONE(404);
    }

    /* parse JSON body */
    json_value *root = cwick_json_parse(req.body, req.body_len, &err);
    if (!root) {
        char message[512];
        snprintf(message, sizeof(message), "invalid JSON: %s",
                 cwick_error_message(&err));
        char *emsg = cwick_json_stringify(message);
        char *body = api_error_body(api, E_BAD, emsg, NULL);
        free(emsg);
        http_response(fd, 400, "Bad Request", "application/json; charset=utf-8", body, false);
        free(body);
        snprintf(detail, sizeof(detail), "invalid JSON");
        cwick_metrics_request_error("invalid_json");
        DONE(400);
    }

    gen_request gr;
    memset(&gr, 0, sizeof(gr));
    gr.is_chat = is_chat || is_resp || is_msgs;
    gr.rid = rid;
    char verr[256];
    /* A request may name the resident model either by the id the checkpoint
     * reports or by its catalog id (they differ once two local models share a
     * model_type — see the comment in validate_body). */
    const cwick_model_entry *cur_ent = resident_entry(srv);
    const char *served_id = cwick_engine_spec(srv->eng[0])->model_type;
    int vcode = is_resp
        ? responses_parse(root, &gr, served_id, cur_ent, verr, sizeof(verr))
        : is_msgs
            ? anthropic_parse(root, &gr, served_id, cur_ent, verr, sizeof(verr))
            : validate_body(root, &gr, verr, sizeof(verr), served_id, cur_ent);
    cwick_json_free(root);
    if (vcode != 0) {
        char *emsg = cwick_json_stringify(verr);
        char *body = api_error_body(api, E_BAD, emsg, NULL);
        free(emsg);
        http_response(fd, vcode, vcode == 400 ? "Bad Request" : "Too Many Requests",
                      "application/json; charset=utf-8", body, false);
        free(body);
        snprintf(detail, sizeof(detail), "refused: %s", verr);
        free(gr.prompt);
        free(gr.system);
        free(gr.user);
        free(gr.reasoning_effort);
        free(gr.tools_block);
        free_request(&gr);
        cwick_tools_free_set(&gr.tools);
        cwick_metrics_request_error("refused_body");
        DONE(vcode);
    }

    /* Take a generation slot, queueing behind whatever is already generating.
     * The lock is held only while the slot table is read and written: the
     * decode itself runs outside it, so /v1/metrics and /v1/models stay
     * answerable while requests are generating. */
    double queue_ms = 0.0;
    int slot = slot_acquire(srv, fd, srv->stop_owner, &queue_ms);
    if (slot == -2) {
        /* Gone while queued. Nothing is written (the peer is not there to
         * receive it) and nothing is computed -- this is the cheap version
         * of the disconnect path: it never interrupts a running request. */
        snprintf(detail, sizeof(detail),
                 "client disconnected while queued after %.0f ms (dropped)",
                 queue_ms);
        cwick_metrics_request_finish(0, 0, 0.0, 0.0, queue_ms / 1000.0, 0.0,
                                    0.0, "abort", false);
        CWICK_VERB(req_tag(gr.rid), "%s", detail);
        free(gr.prompt);
        free(gr.system);
        free(gr.user);
        free(gr.reasoning_effort);
        free(gr.tools_block);
        free_request(&gr);
        cwick_tools_free_set(&gr.tools);
        DONE(499);
    }
    /* A model swap is running: the engine this request was validated against
     * is being freed, so queueing would mean answering with the *next* model's
     * weights under the previous model's id. Refuse with a retry hint. */
    if (slot == -3) {
        char *body = api_error_body(api, E_SERVER,
            "\"a model swap is in progress: the resident weights are being "
            "replaced; retry shortly\"", "\"model_switch\"");
        http_response_ex(fd, 503, "Service Unavailable",
                         "application/json; charset=utf-8", "Retry-After: 5\r\n",
                         body, false);
        free(body);
        snprintf(detail, sizeof(detail), "refused: model swap in progress");
        cwick_metrics_request_error("model_switch");
        free(gr.prompt);
        free(gr.system);
        free(gr.user);
        free(gr.reasoning_effort);
        free(gr.tools_block);
        free_request(&gr);
        cwick_tools_free_set(&gr.tools);
        DONE(503);
    }
    if (slot < 0) {
        /* The queue only overflows when generation is far behind demand, so
         * this is a real overload signal -- say so, and tell the client when
         * to come back instead of leaving it to guess. */
        char *jmsg = srv_asprintf(
            "\"server is overloaded: %d request(s) already queued behind %d "
            "generation slot(s); retry later\"", srv->nwaiting, srv->nslots);
        char *body = api_error_body(api, E_OVERLOAD, jmsg, NULL);
        free(jmsg);
        http_response_ex(fd, 429, "Too Many Requests",
                         "application/json; charset=utf-8",
                         "Retry-After: 5\r\n", body, false);
        free(body);
        snprintf(detail, sizeof(detail), "queue full");
        cwick_metrics_request_refuse("queue_full");
        free(gr.prompt);
        free(gr.system);
        free(gr.user);
        free(gr.reasoning_effort);
        free(gr.tools_block);
        free_request(&gr);
        cwick_tools_free_set(&gr.tools);
        DONE(429);
    }
    cwick_engine *gen_eng = srv->eng[slot];
    /* Before any generation: every Metal batch this thread takes from here on
     * must be its own slot's, or two slots would share one batch struct. */
    cwick_metal_set_slot(slot);
    /* Optional early release of an abandoned request: the engine peeks at
     * this socket at its layer/chunk boundaries and unwinds if the peer is
     * gone. Off unless CWICK_ABORT_ON_DISCONNECT=1 -- see the comment on the
     * field for why acting on it by default is not safe. */
    cwick_engine_set_abort_poll(gen_eng,
                               srv->abort_on_disconnect ? client_gone_poll
                                                         : NULL,
                               srv->abort_on_disconnect ? &srv->slot_fd[slot]
                                                         : NULL);
    CWICK_VERB(req_tag(gr.rid), "generation slot %d", slot);

    /* reset engine state to a fresh conversation */
    cwick_error reset_err;
    cwick_error_clear(&reset_err);
    cwick_status rst = cwick_engine_reset(gen_eng, &reset_err);
    if (rst != CWICK_OK) {
        char *emsg = cwick_json_stringify(cwick_error_message(&reset_err));
        char *body = api_error_body(api, E_SERVER, emsg, NULL);
        free(emsg);
        http_response(fd, 500, "Internal Server Error", "application/json; charset=utf-8", body, false);
        free(body);
        snprintf(detail, sizeof(detail), "engine reset failed");
        cwick_metrics_request_error("engine_reset");
        slot_release(srv, slot);
        free(gr.prompt);
        free(gr.system);
        free(gr.user);
        free(gr.reasoning_effort);
        free(gr.tools_block);
        free_request(&gr);
        cwick_tools_free_set(&gr.tools);
        DONE(500);
    }

    /* `queue_ms` is how long slot_acquire() made this request wait; 0 when a
     * slot was free on arrival. */
    cwick_error gen_err;
    cwick_error_clear(&gen_err);
    cwick_gen_result res;
    memset(&res, 0, sizeof(res));
    double ttft_ms = 0.0;
    run_generation_hold(srv, &gr, fd, detail, sizeof(detail), slot, queue_ms,
                        &res, &ttft_ms, &gen_err);
    if (gen_err.status != CWICK_OK) {
        /* Two generation failures belong to the client or to shutdown, not to
         * the server: a prompt that does not fit the context window is a 400,
         * a Ctrl+C mid-decode is a 503 the client can retry. */
        const bool interrupted = gen_err.status == CWICK_ERR_STATE &&
                                 strcmp(gen_err.message, "interrupted") == 0;
        const bool client_gone = gen_err.status == CWICK_ERR_STATE &&
                                 strcmp(gen_err.message, "client disconnected") == 0;
        const bool client_err = gen_err.status == CWICK_ERR_ARGUMENT;

        /* The peer hung up mid-generation. There is nobody to answer and
         * nothing is broken: hand the slot back, free, and leave -- writing an
         * error would only fail the write and be logged as a 500. 499 is the
         * usual "client closed request" marker for this case. */
        if (client_gone) {
            snprintf(detail, sizeof(detail), "aborted: client disconnected");
            cwick_gen_result_free(&res);
            slot_release(srv, slot);
            free(gr.prompt);
            free(gr.system);
            free(gr.user);
            free(gr.reasoning_effort);
            free(gr.tools_block);
            free_request(&gr);
            cwick_tools_free_set(&gr.tools);
            DONE(499);
        }
        const int code = interrupted ? 503 : (client_err ? 400 : 500);
        const char *message = interrupted ? "server is shutting down (interrupted)"
                                          : cwick_error_message(&gen_err);
        char *emsg = cwick_json_stringify(message);
        char *body = api_error_body(api, client_err ? E_BAD : E_SERVER, emsg, NULL);
        free(emsg);
        http_response(fd, code,
                      interrupted ? "Service Unavailable"
                                  : (client_err ? "Bad Request"
                                                : "Internal Server Error"),
                      "application/json; charset=utf-8", body, false);
        snprintf(detail, sizeof(detail), "%s%s",
                 interrupted ? "interrupted: "
                             : (client_err ? "invalid request: " : "generation failed: "),
                 interrupted ? "shutdown" : cwick_error_message(&gen_err));
        cwick_gen_result_free(&res);
        slot_release(srv, slot);
        free(gr.prompt);
        free(gr.system);
        free(gr.user);
        free(gr.reasoning_effort);
        free(gr.tools_block);
        free_request(&gr);
        cwick_tools_free_set(&gr.tools);
        DONE(code);
    }
    slot_release(srv, slot);

    /* ---- metrics: one finished request -----------------------------
     * Times are the engine's own (prefill / decode), plus this file's view of
     * queue time and end-to-end. A non-streaming request has no observable
     * first-token moment, so it reports the prefill as its TTFT -- that is
     * when the answer starts existing, even though the wire sees it later. */
    {
        double e2e_ms = now_ms() - ((double)t0.tv_sec * 1000.0 +
                                    (double)t0.tv_usec / 1000.0);
        if (e2e_ms < 0.0) e2e_ms = 0.0;
        double ttft = ttft_ms > 0.0 ? ttft_ms + queue_ms
                                    : res.prefill_ms + queue_ms;
        /* sink_stopped is NOT an abort: a sink that returns false is the
         * documented way to end a turn early ("finish_reason 'stop' rather
         * than an error" -- cwick.h), and the HTTP layer already reports
         * finish=stop for it. Mapping it to abort made every stream whose
         * consumer closed the loop count as an aborted request while the
         * access log said stop. Only a real interrupt is an abort. */
        const char *reason = res.interrupted ? "abort"
                           : res.loop_stopped ? "loop"
                           : (res.stop_reason_length ? "length" : "stop");
        cwick_metrics_request_finish(gr.prompt_tokens,
                                    (int64_t)res.n_tokens,
                                    res.prefill_ms / 1000.0,
                                    res.decode_ms / 1000.0,
                                    e2e_ms / 1000.0, ttft / 1000.0,
                                    (double)(gr.max_tokens_set ? gr.max_tokens
                                                               : srv->max_tokens),
                                    reason, true);
    }
    cwick_gen_result_free(&res);

    free(gr.prompt);
    free(gr.system);
    free(gr.user);
    free(gr.reasoning_effort);
    free(gr.tools_block);
    free_request(&gr);
    cwick_tools_free_set(&gr.tools);
    DONE(200);
#undef DONE
}

/* ------------------------------------------------------------------ */
/* Server lifecycle                                                    */
/* ------------------------------------------------------------------ */

cwick_status cwick_server_run(cwick_engine *eng, const cwick_server_options *opt,
                            volatile int *stop, cwick_error *err) {
    cwick_error_clear(err);
    signal(SIGPIPE, SIG_IGN); /* a disconnected SSE client must not kill serve */
    server_ctx srv;
    memset(&srv, 0, sizeof(srv));
    pthread_mutex_init(&srv.store_mtx, NULL);
    srv.eng[0] = eng;
    srv.stop_owner = stop;
    pthread_cond_init(&srv.gen_cv, NULL);
    for (int i = 0; i < CWICK_GEN_SLOTS; i++) srv.slot_fd[i] = -1;
    cwick_engine_set_stop(eng, stop);
    /* Build the extra generation slots. Each borrows slot 0's weight spine,
     * so the 15.46 GiB mlocked spine is paid for once and only the
     * per-sequence state (KV, KDA recurrence, DSA indexer) is duplicated --
     * about 2.4 GiB per extra slot. A slot that fails to come up is simply
     * left NULL and never handed out: the server runs with fewer slots
     * rather than refusing to start. */
    {
        /* Opt in with CWICK_GEN_SLOTS=2|3.
         *
         * The pieces are in place -- per-slot Metal batches, a shared spine,
         * a locked expert cache, a per-thread slot binding -- but with the
         * extra slots live a request is accepted and then makes no further
         * progress (the log stops right after "POST /v1/chat/completions"),
         * and that has not been located yet. Shipping a server that accepts a
         * request and then hangs is strictly worse than one that answers 429,
         * so the default is one slot: identical to the previous behaviour.
         * Remove this gate when the hang is fixed; everything below it is
         * already written for N. */
        int want = 1;
        const char *e = getenv("CWICK_GEN_SLOTS");
        if (e && *e) { int v = atoi(e); if (v >= 2 && v <= CWICK_GEN_SLOTS) want = v; }
        /* Remembered as the template for a later model switch: the replacement
         * engine is built with exactly these options, so swapping models does
         * not silently change the KV window or the slot count. */
        srv.slot_want = want;
        snprintf(srv.spine_kind, sizeof(srv.spine_kind), "bf16");
        srv.context_capacity = cwick_engine_capacity(eng);
        cwick_engine_options eopt;
        memset(&eopt, 0, sizeof(eopt));
        eopt.model_root = cwick_engine_model_root(eng);
        eopt.max_context = cwick_engine_capacity(eng);
        eopt.spine = "bf16";
        eopt.max_new_tokens = 1000000;
        for (int i = 1; i < want; i++) {
            if (!eopt.model_root) break;
            eopt.slot_id = i;
            eopt.shared_spine = cwick_engine_spine(eng);
            cwick_error e2;
            cwick_error_clear(&e2);
            if (cwick_engine_new(&eopt, &srv.eng[i], &e2) != CWICK_OK) {
                CWICK_INFO("init", "generation slot %d unavailable (%s) — "
                                  "serving with %d slot(s)", i,
                          cwick_error_message(&e2), i);
                srv.eng[i] = NULL;
                break;
            }
            cwick_engine_set_stop(srv.eng[i], stop);
        }
        /* Count what actually came up, not what we asked for. */
        srv.nslots = 0;
        for (int i = 0; i < CWICK_GEN_SLOTS; i++) if (srv.eng[i]) srv.nslots++;
    }
    srv.host = opt->host ? opt->host : "127.0.0.1";
    srv.port = opt->port > 0 ? opt->port : 8000;
    /* Per-response ceiling used when the request omits max_tokens; 1000000 is
     * the "no explicit cap" sentinel (the engine clamps it to the context). */
    srv.max_tokens = opt->max_tokens > 0 ? opt->max_tokens : 1000000;
    srv.max_request_bytes = opt->max_request_bytes > 0 ? opt->max_request_bytes
                                                       : CWICK_SERVER_MAX_BODY;
    {
        const char *aod = getenv("CWICK_ABORT_ON_DISCONNECT");
        srv.abort_on_disconnect = (aod && *aod &&
                                   strcmp(aod, "0") != 0 &&
                                   strcmp(aod, "off") != 0);
    }
    /* Checkpoint stride for long prefills (0 = one snapshot at the end). */
    srv.pfx_block = 4096;
    { const char *be = getenv("CWICK_PFX_BLOCK");
      if (be && *be) { int64_t v = atoll(be);
                       if (v >= 0) srv.pfx_block = v; } }
    {
        const char *dir = opt->prefix_cache_dir;
        if (!dir || !*dir) dir = getenv("CWICK_PREFIX_CACHE_DIR");
        if (dir && *dir) {
            uint64_t gib = (uint64_t)(opt->prefix_cache_gib > 0 ? opt->prefix_cache_gib : 0);
            if (gib == 0) gib = 8;
            srv.pfx = cwick_pfx_open(dir, gib * (1024ull * 1024ull * 1024ull));
            if (srv.pfx) {
                int64_t ents = 0;
                uint64_t bytes = 0;
                cwick_pfx_stats(srv.pfx, &ents, &bytes);
                CWICK_INFO("init", "prefix cache %s: %lld snapshot%s, %.2f GiB "
                                  "of %.2f GiB used",
                          dir, (long long)ents, ents == 1 ? "" : "s",
                          (double)bytes / (1024.0 * 1024.0 * 1024.0),
                          (double)gib);
            } else {
                CWICK_INFO("init", "prefix cache disabled: %s is not usable", dir);
            }
        }
    }
    pfx_writer_start(&srv);   /* after srv.pfx is known; no-op when disabled */
    srv.verbose = opt->verbose;
    srv.trace = opt->trace;
    srv.web = opt->web;
    srv.jev = opt->jev;
    srv.live_engine = opt->live_engine;
    if (srv.live_engine) *srv.live_engine = eng;
    srv.wake_r = -1;
    srv.wake_w = -1;
    cwick_status terminal = CWICK_OK;
    pthread_mutex_init(&srv.gen_lock, NULL);
    pthread_mutex_init(&srv.pfx_mtx, NULL);
    pthread_mutex_init(&srv.client_mtx, NULL);
    pthread_cond_init(&srv.client_cv, NULL);
    srv.clients = 0;

    /* Metrics registry: publish what this instance is running. The stream and
     * the memory ceiling are sampled live, so they are handed over as pointers
     * the registry can read at scrape time. */
    {
        const cwick_model_spec *mspec = cwick_engine_spec(eng);
        const cwick_spine *spine = cwick_engine_spine(eng);
        cwick_metrics_info mi;
        memset(&mi, 0, sizeof(mi));
        /* The catalog is built further down, so the served alias is derived
         * from the root directly — same answer, no entry to look up yet. */
        char served[CWICK_MODEL_ID_MAX];
        mi.model = mspec->model_type;
        if (cwick_model_served_alias(cwick_engine_model_root(eng), served,
                                     sizeof(served)) &&
            served[0] && strcmp(served, mspec->model_type) != 0)
            mi.model = served;
        mi.backend = cwick_metal_backend_active() ? "metal" : "cpu";
        mi.spine = spine && spine->have_int8 ? "int8" : "bf16";
        mi.max_context = cwick_engine_context_window(eng);
        mi.num_layers = mspec->num_hidden_layers;
        mi.num_experts = mspec->num_experts;
        mi.experts_per_token = mspec->num_experts_per_token;
        cwick_metrics_init(&mi);
        cwick_metrics_set_context_window(mi.max_context);
        if (spine && spine->stream) cwick_metrics_set_stream(spine->stream);
        else {
            cwick_stream *es = cwick_engine_expert_stream(eng);
            if (es) cwick_metrics_set_stream(es);
        }
        const cwick_mem_plan *plan = spine ? cwick_spine_memory_plan(spine) : NULL;
        if (plan) cwick_metrics_set_memory_ceiling(plan->target_bytes);
    }

    /* Fold the legacy CWICK_SERVER_LOG knob into the logger decided by the CLI
     * so every module prints at the same loudness. */
    {
        bool verbose = srv.verbose || srv_log_verbose_env();
        bool quiet = !srv_log_enabled();
        cwick_log_config(verbose, srv.trace, quiet);
        cwick_engine_set_verbose(eng, verbose);
    }

    /* tokenizer (model root: legacy tokenizer/ or HF tokenizer.json) */
    cwick_tokenizer *tok = NULL;
    cwick_status st = cwick_tokenizer_load(cwick_engine_model_root(eng),
                                        cwick_engine_spec(eng), &tok, err);
    if (st != CWICK_OK) { pthread_mutex_destroy(&srv.gen_lock); pthread_mutex_destroy(&srv.pfx_mtx);
    pthread_cond_destroy(&srv.gen_cv); return st; }
    srv.tok = tok;
    srv_probe_think_ids(&srv);

    /* ---- model catalog ----------------------------------------------
     * The resident model comes first: an OpenAI client that ignores
     * everything after data[0] must get whatever is actually answering. Then
     * whatever --models-dir found. Scanning reads only `config.json` and a
     * couple of stat() calls, never weights; an unreadable directory is
     * reported and skipped rather than failing startup. */
    {
        const char *root = cwick_engine_model_root(eng);
        const char *id = cwick_engine_spec(eng)->model_type;
        cwick_models_list_init(&srv.catalog);
        bool listed_resident = root && cwick_models_list_add(&srv.catalog, id, root);
        if (opt->models_dirs_n > 0) {
            int limit = opt->models_dirs_n < 8 ? opt->models_dirs_n : 8;
            for (int i = 0; i < limit; i++) {
                if (!opt->models_dirs[i]) continue;
                srv.models_dirs[srv.models_dirs_n++] = opt->models_dirs[i];
                (void)cwick_models_scan_dir(opt->models_dirs[i], &srv.catalog);
            }
            cwick_models_list_dedupe_ids(&srv.catalog);
            CWICK_INFO("init", "model catalog: %zu model%s known across %d "
                              "directory/ies — GET /v1/models lists them, "
                              "POST /v1/models/switch swaps the resident one",
                      srv.catalog.n, srv.catalog.n == 1 ? "" : "s",
                      srv.models_dirs_n);
        }
        /* The download library lives under ./models. Register it as one more
         * scan directory unless the operator already named it, so a model
         * fetched in an earlier run is switchable at startup. The catalog host
         * can be overridden without a new flag: CWICK_HUB_ENDPOINT wins, then
         * HF_ENDPOINT (the same variable the standalone downloader reads). */
        {
            const char *ep = getenv("CWICK_HUB_ENDPOINT");
            if (!ep || !*ep) ep = getenv("HF_ENDPOINT");
            srv.hub = cwick_hub_open("./models", ep);
        }
        if (srv.hub && srv.models_dirs_n < 8) {
            bool have = false;
            for (int i = 0; i < srv.models_dirs_n; i++)
                if (strcmp(srv.models_dirs[i], "./models") == 0) have = true;
            if (!have) {
                srv.models_dirs[srv.models_dirs_n++] = "./models";
                (void)cwick_models_scan_dir("./models", &srv.catalog);
                /* Downloads live one level down (./models/owner/name), so a
                 * single scan of ./models cannot see them. Scan each owner
                 * directory too; this reads config.json only, never weights. */
                DIR *owners = opendir("./models");
                if (owners) {
                    struct dirent *de;
                    while ((de = readdir(owners)) != NULL) {
                        if (de->d_name[0] == '.') continue;
                        char sub[1024];
                        if (snprintf(sub, sizeof(sub), "./models/%s",
                                     de->d_name) >= (int)sizeof(sub)) continue;
                        DIR *probe = opendir(sub);
                        if (!probe) continue;
                        closedir(probe);
                        (void)cwick_models_scan_dir(sub, &srv.catalog);
                    }
                    closedir(owners);
                }
                cwick_models_list_dedupe_ids(&srv.catalog);
                if (listed_resident) srv.resident = &srv.catalog.v[0];
            }
        }
        /* Resolved after every add: the list is an array that may have been
         * reallocated, so pointers taken earlier are not necessarily valid.
         * Index 0 is the resident model — that is the order we built it in. */
        if (listed_resident) srv.resident = &srv.catalog.v[0];
    }

    /* listen */
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        cwick_error_set(err, CWICK_ERR_IO, "socket: %s", strerror(errno));
        cwick_tokenizer_free(tok);
        pthread_mutex_destroy(&srv.gen_lock); pthread_mutex_destroy(&srv.pfx_mtx);
    pthread_cond_destroy(&srv.gen_cv);
        return CWICK_ERR_IO;
    }
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)srv.port);
    if (inet_pton(AF_INET, srv.host, &addr.sin_addr) != 1) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "invalid host %s", srv.host);
        close(fd);
        cwick_tokenizer_free(tok);
        pthread_mutex_destroy(&srv.gen_lock); pthread_mutex_destroy(&srv.pfx_mtx);
    pthread_cond_destroy(&srv.gen_cv);
        return CWICK_ERR_ARGUMENT;
    }
    uint32_t host_ip = ntohl(addr.sin_addr.s_addr);
    if (host_ip != INADDR_LOOPBACK && !cwick_env_true("CWICK_ALLOW_PUBLIC")) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                       "non-loopback listener refused; use --lan (single-user "
                       "login) or CWICK_ALLOW_PUBLIC=1 only behind a trusted proxy");
        close(fd);
        cwick_tokenizer_free(tok);
        pthread_mutex_destroy(&srv.gen_lock); pthread_mutex_destroy(&srv.pfx_mtx);
    pthread_cond_destroy(&srv.gen_cv);
        return CWICK_ERR_ARGUMENT;
    }
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        cwick_error_set(err, CWICK_ERR_IO, "bind %s:%d: %s", srv.host, srv.port,
                       strerror(errno));
        close(fd);
        cwick_tokenizer_free(tok);
        pthread_mutex_destroy(&srv.gen_lock); pthread_mutex_destroy(&srv.pfx_mtx);
    pthread_cond_destroy(&srv.gen_cv);
        return CWICK_ERR_IO;
    }
    if (listen(fd, 8) != 0) {
        cwick_error_set(err, CWICK_ERR_IO, "listen: %s", strerror(errno));
        close(fd);
        cwick_tokenizer_free(tok);
        pthread_mutex_destroy(&srv.gen_lock); pthread_mutex_destroy(&srv.pfx_mtx);
    pthread_cond_destroy(&srv.gen_cv);
        return CWICK_ERR_IO;
    }
    srv.listen_fd = fd;

    /* Single-user login: enabled by --password/CWICK_PASSWORD, or forced by
     * --lan (with an empty string meaning "generate one"). Created after the
     * listener is up so every earlier failure path does not need its cleanup. */
    if (opt->password) {
        srv.auth = cwick_auth_new(opt->password, 43200);
        if (!srv.auth) {
            cwick_error_set(err, CWICK_ERR_MEMORY,
                           "auth: cannot allocate the single-user login state");
            close(fd);
            cwick_tokenizer_free(tok);
            pthread_mutex_destroy(&srv.gen_lock);
            pthread_mutex_destroy(&srv.pfx_mtx);
            pthread_cond_destroy(&srv.gen_cv);
            return CWICK_ERR_MEMORY;
        }
    }

    /* Self-pipe used purely as a wakeup for the poll() below: main.c writes a
     * byte from its SIGINT handler, which is the only reliable way to break
     * out of accept() when the signal arrives (macOS restarts accept() after
     * the handler returns, so the old `while(!stop) accept()` loop simply
     * never saw Ctrl+C while idle). */
    if (opt->wake_fd >= 0) {
        srv.wake_r = opt->wake_fd;               /* owned by main */
    } else {
        int p[2];
        if (pipe(p) == 0) { srv.wake_r = p[0]; srv.wake_w = p[1]; }
    }

    CWICK_INFO("init", "OpenAI-compatible server listening on http://%s:%d/v1 "
              "(model %s)", srv.host, srv.port, resident_served_name(&srv));
    /* App storage. Opened unconditionally (not only with --web): an app is
     * served from this process either way, and a store that is missing at
     * request time is a 500 the operator has to diagnose from a log line,
     * whereas a store that failed to open is a startup error with a path.
     * A failure is still non-fatal on purpose — the model is the product, and
     * a read-only home directory should not stop a 320B engine from serving. */
    {
        cwick_error se;
        cwick_error_clear(&se);
        const char *sp = opt->store_path;
        srv.store_single = cwick_store_path_is_file(sp);
        if (srv.store_single) {
            /* An existing single-file layout: every app shares it. Kept so a
             * `--store ~/apps.db` or a CWICK_STORE pointing at a file still
             * finds the data it wrote last time. */
            srv.store = cwick_store_open(sp, &se);
        } else {
            /* One database per app. The primary file in the directory belongs
             * to the server, not to an app: it holds the API keys. A trailing
             * slash is stripped so `--store ./dbs/` logs and reports
             * `./dbs/kart.db`, not `./dbs//kart.db`. */
            const char *dir = (sp && *sp) ? sp : cwick_store_default_dir();
            snprintf(srv.store_dir, sizeof(srv.store_dir), "%s", dir);
            size_t dl = strlen(srv.store_dir);
            while (dl > 1 && srv.store_dir[dl - 1] == '/') srv.store_dir[--dl] = '\0';
            (void)cwick_mkdir_p(srv.store_dir);
            char primary[1152];
            cwick_path_join(primary, sizeof(primary), srv.store_dir, "auth.db");
            srv.store = cwick_store_open(primary, &se);
        }
        if (srv.store) {
            CWICK_INFO("apps", "sqlite store: %s%s",
                      cwick_store_path(srv.store),
                      srv.store_single ? "" : "  (one <app>.db per app in this "
                                                 "directory)");
        } else {
            CWICK_INFO("apps", "sqlite store unavailable (%s) — apps fall back "
                              "to localStorage", cwick_error_message(&se));
        }
    }

    /* API keys from an earlier run. Loaded before the new one is minted so a
     * restart never invalidates what an agent platform was configured with —
     * that is the whole reason keys are persisted and sessions are not. */
    if (srv.auth && srv.store) {
        cwick_store_keys_each(srv.store, auth_keys_load_cb, srv.auth);
        int nkeys = 0;
        cwick_auth_keys_each(srv.auth, keys_count_cb, &nkeys);
        if (nkeys)
            CWICK_INFO("auth", "%d API key(s) adopted from %s (hashes only)",
                      nkeys, cwick_store_path(srv.store));
    }

    /* --api-key / CWICK_API_KEY: mint (or adopt) one key at startup and print
     * it once. Printed, not stored: this process never keeps the plaintext. */
    if (srv.auth && (opt->api_key || opt->api_key_secret)) {
        cwick_api_key_info info;
        memset(&info, 0, sizeof(info));
        char *secret = NULL;
        bool ok = false;
        if (opt->api_key_secret && *opt->api_key_secret) {
            const char *nm = (opt->api_key && *opt->api_key) ? opt->api_key
                                                             : "startup";
            ok = cwick_auth_key_import(srv.auth, nm, opt->api_key_secret, &info);
        } else {
            secret = cwick_auth_key_create(srv.auth,
                                          opt->api_key && *opt->api_key
                                              ? opt->api_key : "startup", &info);
            ok = secret != NULL;
        }
        if (ok && srv.store) {
            cwick_store_key row;
            memset(&row, 0, sizeof(row));
            row.id = info.id; row.name = info.name;
            row.hash_hex = info.hash_hex; row.prefix = info.prefix;
            row.created_at = info.created_at; row.enabled = true;
            cwick_error e;
            cwick_error_clear(&e);
            if (!cwick_store_key_put(srv.store, &row, &e)) {
                CWICK_INFO("auth", "api key could not be persisted (%s) — it "
                                  "will not survive a restart",
                          cwick_error_message(&e));
            }
        }
        if (ok) {
            CWICK_INFO("auth", "API key \"%s\" (%s): %s", info.name, info.prefix,
                      secret ? secret : opt->api_key_secret);
            CWICK_INFO("auth", "  send it as `Authorization: Bearer <key>` or "
                              "`X-API-Key: <key>`; it is printed once and only "
                              "its hash is stored");
            if (!srv.store)
                CWICK_INFO("auth", "  no sqlite store: this key dies with the process");
        }
        if (secret) { memset(secret, 0, strlen(secret)); free(secret); }
    }

    if (srv.auth) {
        const char *gen = cwick_auth_generated_password(srv.auth);
        CWICK_INFO("auth", "single-user login enabled: PBKDF2-HMAC-SHA256, "
                          "in-memory sessions, 12 h TTL");
        if (gen)
            CWICK_INFO("auth", "generated password: %s  (set --password or "
                              "CWICK_PASSWORD to choose your own; it is printed "
                              "once and never written to disk)", gen);
        CWICK_INFO("auth", "login: POST /v1/auth/login (JSON {\"password\":...}) "
                          "-> HttpOnly cookie + Bearer token; GET /login for the UI");
    } else {
        CWICK_VERB("auth", "single-user login disabled (loopback default; set "
                          "--password or CWICK_PASSWORD to require it)");
    }
    CWICK_INFO("init", "endpoints: GET /v1/models, POST /v1/completions, "
              "POST /v1/chat/completions, POST /v1/responses, "
              "POST /v1/messages, GET /metrics%s  "
              "(press Ctrl+C to stop)", srv.trace ? " [trace on]" : "");
    /* The control surface the built-in UI is built on, for anyone driving it
     * from curl instead: without this line the only way to discover that the
     * resident model can be swapped at runtime is to read the source. */
    CWICK_INFO("init", "control: GET /v1/config, POST /v1/models/switch, "
                      "GET /v1/logs, GET /v1/logs/stream, GET /v1/share");
    CWICK_INFO("init", "apps: GET /v1/store, GET|PUT|DELETE /v1/apps/<id>/state, "
                      "GET|POST /v1/apps/<id>/events, GET|PUT|DELETE "
                      "/v1/apps/<id>/games");
    if (srv.web) {
        /* Name the URL a browser needs, not the API root: the page is at /,
         * and saying so is the difference between "there is a UI" and
         * "here it is". Loopback only, unless the operator opted out. */
        char lan[64] = "";
        bool have_lan = cwick_lan_ipv4(lan, sizeof(lan));
        const char *shown = strcmp(srv.host, "0.0.0.0") == 0
                                ? (have_lan ? lan : "127.0.0.1")
                                : srv.host;
        CWICK_INFO("init", "web chat UI: http://%s:%d/  (%zu assets embedded, "
                          "same process, same OpenAI API)",
                  shown, srv.port, cwick_webui_asset_count());
        if (strcmp(srv.host, "0.0.0.0") == 0) {
            if (have_lan)
                CWICK_INFO("init", "LAN share: http://%s:%d/ — the app market is "
                                  "at / #apps; each app has its own QR for a phone",
                          lan, srv.port);
            else
                CWICK_INFO("init", "--lan is on but no non-loopback IPv4 was found; "
                                  "the QR will explain that sharing is unavailable");
        }
    }

    /* The decision endpoint is independent of --web: an app can call it from
     * anywhere on the LAN, and an operator turning off the built-in UI should
     * not lose the ability to see which adapter is live. */
    if (srv.jev) {
        CWICK_INFO("init", "decision endpoint: POST /v1/systemone (also "
                          "/inference/v1/systemone; %d questions per request, one "
                          "prefill each, no generated tokens) and GET "
                          "/v1/systemone (the live adapter)",
                  JEV_MAX_QUESTIONS);
        srv_probe_decision_adapter(&srv);
    }

    while (!(stop && *stop)) {
        struct pollfd pfd[2];
        int nfds = 0;
        pfd[nfds].fd = fd;
        pfd[nfds].events = POLLIN;
        pfd[nfds].revents = 0;
        nfds++;
        if (srv.wake_r >= 0) {
            pfd[nfds].fd = srv.wake_r;
            pfd[nfds].events = POLLIN;
            pfd[nfds].revents = 0;
            nfds++;
        }
        /* 500 ms tick: even without a signal pipe we re-check `stop` twice a
         * second, so `serve` cannot outlive the process's Ctrl+C. */
        int r = poll(pfd, (nfds_t)nfds, 500);
        if (r < 0) {
            if (errno == EINTR) continue;
            if (stop && *stop) break;
            cwick_error_set(err, CWICK_ERR_IO, "poll: %s", strerror(errno));
            terminal = CWICK_ERR_IO;
            break;
        }
        if (r == 0) continue;
        if (srv.wake_r >= 0 && nfds > 1 && (pfd[1].revents & POLLIN)) break;
        if (stop && *stop) break;
        if (!(pfd[0].revents & POLLIN)) continue;

        int c = accept(fd, NULL, NULL);
        if (c < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            if (stop && *stop) break;
            cwick_error_set(err, CWICK_ERR_IO, "accept: %s", strerror(errno));
            terminal = CWICK_ERR_IO;
            break;
        }
        /* Serve each connection on its own thread. Handling it inline here
         * would block the accept loop for the whole generation -- a
         * /v1/metrics scrape during a 12-minute completion would sit in the
         * listen backlog, which is exactly the "API looks dead" failure.
         * The engine's single generation slot still serialises the
         * expensive work; what runs concurrently is everything else. */
        if (!client_register(&srv)) {
            const char *busy =
                "{\"error\":{\"message\":\"too many concurrent connections\","
                "\"type\":\"rate_limit_error\",\"param\":null,\"code\":null}}";
            http_response(c, 503, "Service Unavailable",
                          "application/json; charset=utf-8", busy, false);
            close(c);
            if (stop && *stop) break;
            continue;
        }
        client_job *j = cwick_xmalloc(sizeof(*j));
        j->srv = &srv;
        j->fd = c;
        j->rid = ++srv.request_seq;
        pthread_t t;
        if (pthread_create(&t, NULL, client_thread, j) == 0) {
            pthread_detach(t);   /* reaped by client_thread's own bookkeeping */
        } else {
            /* Thread creation failed (resource exhaustion): fall back to
             * inline handling rather than dropping the connection. */
            pthread_mutex_lock(&srv.client_mtx);
            if (--srv.clients <= 0) {
                srv.clients = 0;
                pthread_cond_broadcast(&srv.client_cv);
            }
            pthread_mutex_unlock(&srv.client_mtx);
            cwick_log_set_request(j->rid);
            handle_client(&srv, c, j->rid);
            cwick_log_set_request(0);
            free(j);
        }
        if (stop && *stop) break;
    }

    /* Drain in-flight connections before anything they touch is freed.
     * The stop flag is already set, so a generation aborts at its next
     * token (one decode step, ~0.7 s) and a slow reader hits its 2 s socket
     * timeout; a wedged client cannot hold shutdown hostage for more than
     * CWICK_DRAIN_TIMEOUT_S. An impatient operator's second Ctrl+C calls
     * _exit(130) in the signal handler and never reaches this wait. */
    {
        struct timespec deadline;
        clock_gettime(CLOCK_REALTIME, &deadline);
        deadline.tv_sec += CWICK_DRAIN_TIMEOUT_S;
        pthread_mutex_lock(&srv.client_mtx);
        while (srv.clients > 0) {
            int w = pthread_cond_timedwait(&srv.client_cv, &srv.client_mtx,
                                           &deadline);
            if (w == ETIMEDOUT) {
                CWICK_INFO("init", "shutdown: %d connection(s) still in flight "
                                  "after %d s — proceeding anyway",
                          srv.clients, CWICK_DRAIN_TIMEOUT_S);
                break;
            }
        }
        pthread_mutex_unlock(&srv.client_mtx);
    }

    CWICK_INFO("init", "listener closed, served %llu request(s)",
              (unsigned long long)srv.request_seq);
    /* Final metrics pass: a `--verbose` serve prints its own numbers on the
     * way out, so a scrape is not the only way to see what a session did. */
    cwick_metrics_log_summary("metrics");
    close(fd);
    if (srv.wake_w >= 0) { close(srv.wake_w); close(srv.wake_r); }
    /* Not `tok`: that is the *startup* tokenizer, and a runtime model switch
     * already freed it before loading the new one into srv.tok. Freeing the
     * local here is a double free — it segfaults on every Ctrl+C that follows
     * a switch, which is exactly how long it took to find this. Free whatever
     * is resident, once. */
    if (srv.tok) { cwick_tokenizer_free(srv.tok); srv.tok = NULL; }
    pthread_mutex_destroy(&srv.gen_lock); pthread_mutex_destroy(&srv.pfx_mtx);
    pthread_cond_destroy(&srv.gen_cv);
    pthread_mutex_destroy(&srv.client_mtx);
    pthread_cond_destroy(&srv.client_cv);
    /* Before the cache is closed: a queued snapshot can still be in flight,
     * and writing to a closed cache would be a use-after-free. */
    pfx_writer_stop(&srv);
    cwick_pfx_close(srv.pfx);   /* the snapshots stay on disk for the next run */
    cwick_auth_free(srv.auth);
    /* Closed after the listener and the auth state: an app's last write can
     * still be in flight while a connection thread drains. */
    for (int i = 0; i < srv.store_slots_n; i++)
        cwick_store_close(srv.store_slots[i].st);
    srv.store_slots_n = 0;
    pthread_mutex_destroy(&srv.store_mtx);
    cwick_store_close(srv.store);
    /* Ask the download workers to stop before their state is freed. Partial
     * files stay on disk, which is what makes the next run resumable. */
    cwick_hub_close(srv.hub);
    srv.hub = NULL;
    return terminal;
}
