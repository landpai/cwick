/* main.c — CWICK CLI: setup / run / serve / doctor.
 *
 *   cwick setup --root DIR            install a synthetic model (verification)
 *   cwick run --root DIR --prompt P [--max-new N] [--spine bf16|int8] [--chat]\
[--backend metal|cpu]
 *                                   [--verbose] [--trace]
 *   cwick serve --root DIR [--host H] [--port P] [--max-tokens N]\
[--backend metal|cpu]
 *                         [--verbose] [--trace] [--mem-fraction PCT]
 *                         [--max-memory GiB] [--web]
 *   cwick doctor --root DIR
 *
 * Ctrl+C is handled here: a single SIGINT/SIGTERM stops the server loop and
 * aborts an in-flight decode (the engine polls the same flag); a second one
 * exits immediately, because waiting politely is not what someone pressing
 * Ctrl+C twice wants.
 */
#include "cwick.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "log.h"
#include "logcap.h"
#include "mem.h"
#include "util.h"

static volatile int g_stop = 0;
static volatile int g_signals = 0;
static int g_wake_pipe[2] = { -1, -1 };

/* The signal handler does two things only: set the flag every poll loop in
 * the process reads, and write a byte into the self-pipe so anything blocked
 * in poll()/select() wakes up at once. No malloc, no printf, no syscalls that
 * can fail unpredictably. */
static void on_signal(int sig) {
    g_stop = 1;
    g_signals++;
    if (g_wake_pipe[1] >= 0) {
        char b = (char)sig;
        ssize_t ignored = write(g_wake_pipe[1], &b, 1);
        (void)ignored;
    }
    /* Impatient operator: leave now rather than waiting for a 306 GiB
     * checkpoint to finish whatever it was reading. */
    if (g_signals >= 2) _exit(130);
}

/* Install handlers that do NOT restart interrupted syscalls: on macOS
 * signal() sets SA_RESTART, which is why accept() used to swallow Ctrl+C
 * until the next request arrived. */
static void install_signal_handlers(void) {
    if (pipe(g_wake_pipe) != 0) {
        g_wake_pipe[0] = -1;
        g_wake_pipe[1] = -1;
    }
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_signal;
    sa.sa_flags = 0;                 /* deliberately NOT SA_RESTART */
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    struct sigaction ign;
    memset(&ign, 0, sizeof(ign));
    ign.sa_handler = SIG_IGN;
    sigemptyset(&ign.sa_mask);
    sigaction(SIGPIPE, &ign, NULL);
}

static void close_wake_pipe(void) {
    if (g_wake_pipe[0] >= 0) close(g_wake_pipe[0]);
    if (g_wake_pipe[1] >= 0) close(g_wake_pipe[1]);
    g_wake_pipe[0] = -1;
    g_wake_pipe[1] = -1;
}

static const char *need_arg(int *i, int argc, char **argv, const char *flag) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "cwick: %s requires a value\n", flag);
        exit(2);
    }
    return argv[++(*i)];
}

/* Configure the logger once, from the parsed flags. Every later line — model
 * load, memory budget, requests, generations — goes to stdout through it. */
static bool parse_i64(const char *s, int64_t min, int64_t max, int64_t *out) {
    if (!s || !*s) return false;
    errno = 0;
    char *end = NULL;
    long long v = strtoll(s, &end, 10);
    if (errno == ERANGE || end == s || *end != '\0' || v < min || v > max)
        return false;
    *out = (int64_t)v;
    return true;
}

static bool parse_double(const char *s, double min, double max, double *out) {
    if (!s || !*s) return false;
    errno = 0;
    char *end = NULL;
    double v = strtod(s, &end);
    if (errno == ERANGE || end == s || *end != '\0' || !isfinite(v) ||
        v < min || v > max)
        return false;
    *out = v;
    return true;
}

static void configure_logging(bool verbose, bool trace) {
    cwick_log_config(verbose, trace, false);
}

/* --backend metal|cpu: say what actually happened.
 *
 * The Metal path is best-effort by design -- a GPU must never be allowed to
 * break a generation -- so "unavailable" is a loud warning plus a fallback to
 * the CPU GEMV, not a fatal error. It is repeated on stderr as well, because
 * stdout is usually piped into a chat client or an agent log where a startup
 * warning is easy to miss. */
static void report_backend(int backend) {
    if (backend == 0) {
        /* This has to call cwick_metal_force(0), not just say so.
         *
         * g_metal_selected starts at -1 ("decide by measurement"), and
         * everything downstream reads it: metal_probe() takes -1 as
         * "bring the device up on Apple Silicon", and cwick_metal_gemv()
         * takes it as "consult the autotune". Leaving it at -1 meant
         * --backend cpu printed the right line and then ran the expert GEMV
         * on the GPU anyway whenever the autotune happened to prefer it --
         * which it did in `cwick serve` (Metal 428 us vs CPU 516 us) while
         * picking the CPU in `cwick run` (533 vs 263 us) for the same
         * machine and the same flag. The two autotunes disagree because the
         * lazy one fires during the first request and times the CPU pool
         * while that pool is busy. An explicit --backend must decide, not
         * advise. */
        cwick_metal_force(0);
        CWICK_INFO("init", "compute backend: cpu (expert GEMV stays on the CPU)");
        return;
    }
    if (cwick_metal_force(1) == 0) return;      /* logs "backend up" itself */
    fprintf(stderr,
            "cwick: warning: --backend metal was requested but no usable Metal "
            "device was found;\n"
            "cwick: falling back to the CPU expert GEMV (same output, slower).\n");
    CWICK_INFO("init", "requested backend metal is unavailable — running the "
                      "CPU expert GEMV instead; output is identical, only "
                      "decode speed differs");
}

/* Say, once at startup, whether this machine is going to fight the memory
 * budget: how much swap is already in use, and how hard the compressor is
 * working. Both are host-wide counters, so this is context for the numbers
 * cwick prints later, not an accusation -- but if swap is half full before a
 * 306 GiB checkpoint is even mapped, the 80% ceiling is not the real limit
 * and someone needs to know. */
static void report_vm_state(void) {
    cwick_vm_state vm;
    if (!cwick_vm_state_sample(&vm)) return;
    CWICK_INFO("mem", "host swap %.2f of %.2f GiB used, compressor %llu "
                     "compressions / %llu decompressions since boot",
              cwick_gib(vm.swap_used), cwick_gib(vm.swap_total),
              (unsigned long long)vm.compressions,
              (unsigned long long)vm.decompressions);
    if (cwick_memory_noswap()) {
        CWICK_INFO("mem", "no-swap mode: cwick wires the resident spine and the "
                         "expert arena so the VM can neither compress nor page "
                         "them (--allow-swap to opt out)");
    } else {
        CWICK_INFO("mem", "--allow-swap: cwick's memory may be compressed or "
                         "swapped; expect slower decode under pressure");
    }
}

/* ---- one running copy per checkpoint ----------------------------------
 * Everything a process allocates is sized against the *whole* machine: the
 * memory plan takes 80% of host RAM, the resident spine and the hot-expert
 * arena are then mlock()ed so the VM can neither compress nor page them. That
 * arithmetic is only valid for one process. Two copies of a 79.6 GiB
 * checkpoint on a 128 GiB box ask for 160 GiB of memory the kernel is not
 * allowed to reclaim, and macOS resolves the oversubscription by killing
 * things -- usually the shell that started them, sometimes the window server.
 *
 * So the CLI takes an advisory lock next to the weights before it allocates
 * anything. The kernel drops it when the holder exits, so a crash never
 * leaves a stale lock behind, and a read-only checkpoint (a model on a
 * read-only mount) is not a reason to refuse to run.
 *
 * CWICK_ALLOW_MULTI=1 turns it off for a deliberate multi-instance
 * measurement on a machine with room to spare.
 */
static struct {
    char root[1024];
    int fd;
} g_instance = { { 0 }, -1 };

static bool instance_claim(const char *root) {
    if (!root || !*root) return true;
    if (cwick_env_true("CWICK_ALLOW_MULTI")) return true;
    if (g_instance.fd >= 0 && strcmp(g_instance.root, root) == 0) return true;

    char path[1152];
    cwick_path_join(path, sizeof(path), root, ".cwick.lock");
    int fd = open(path, O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) {
        CWICK_VERB("init", "cannot create %s (%s); running without the "
                   "single-instance lock", path, strerror(errno));
        return true;
    }
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        char buf[64] = { 0 };
        ssize_t n = pread(fd, buf, sizeof(buf) - 1, 0);
        close(fd);
        char who[32] = "";
        long pid = (n > 0) ? strtol(buf, NULL, 10) : 0;
        if (pid > 0) snprintf(who, sizeof(who), " (pid %ld)", pid);
        fprintf(stderr,
                "cwick: %s is already held by another running copy%s.\n"
                "cwick: this checkpoint is budgeted against the whole machine\n"
                "cwick: (80%% of RAM, then wired so it cannot be paged), so a\n"
                "cwick: second copy does not fit -- it would push the host into\n"
                "cwick: swap or force the kernel to kill processes.\n"
                "cwick: use the running server, stop it first, or set\n"
                "cwick: CWICK_ALLOW_MULTI=1 if the box really has room.\n",
                path, who);
        return false;
    }
    if (ftruncate(fd, 0) == 0) {
        char me[32];
        int len = snprintf(me, sizeof(me), "%d\n", (int)getpid());
        if (len > 0 && write(fd, me, (size_t)len) < 0) { /* advisory only */ }
    }
    snprintf(g_instance.root, sizeof(g_instance.root), "%s", root);
    g_instance.fd = fd;
    return true;
}

static void select_backend(int backend) {
    /* Give the GPU as much of the unified pool as the memory policy allows,
     * before anything touches Metal: the driver samples the ceiling once, at
     * device creation, so raising it later has no effect on this process. */
    {
        double frac = 0.0;
        uint64_t limit = 0;
        cwick_memory_policy(&frac, &limit);
        cwick_mem_raise_gpu_wired(0, frac > 0.0 ? frac : 0.80);
    }
    if (backend >= 0) {
        report_backend(backend);
        return;
    }
    /* No --backend flag: measure instead of assuming. An expert GEMV is only
     * 8 MiB of work per dispatch, and on a machine whose CPU has 12
     * performance cores with NEON the GPU is frequently the slower of the two
     * once per-dispatch overhead is counted.
     *
     * The measurement has to happen before the engine (and therefore before
     * the expert cache) is built, because the winner also decides whether the
     * cache is worth allocating as GPU-visible memory. */
    CWICK_INFO("init", "compute backend: auto (expert GEMV backend chosen by "
                      "measurement; --backend metal|cpu to force one)");
    cwick_metal_autotune();
}

/* --trace prints the model's actual input and output to stdout. That is the
 * point, but it is also the one flag that can put user content into a log, so
 * say so once at startup and keep every payload truncated. */
static void trace_privacy_notice(void) {
    CWICK_INFO("init", "--trace is ON: prompts, outputs and tool calls are "
                      "printed to stdout (truncated). cwick writes no log file "
                      "of its own, but stdout usually is one -- start.sh tees "
                      "it into ./logs/cwick-serve.log -- so treat everything "
                      "this prints as written to disk, and do not enable it on "
                      "a shared host");
}

/* `ran` reports whether the engine actually came up, so the caller only claims
 * a clean shutdown when there was something to shut down. */
/* Startup warmup: one silent generation on a throwaway prompt, before the
 * listening socket exists.
 *
 * The first decode steps of a cold process are measured to be slow for two
 * reasons a warm request never pays: the expert arena grows under load
 * (shmem_alloc 189 ms/step) and the cache is empty (17.3% hit vs ~68%
 * warm -- 1013 vs 590 ms/step on the same build). Both halves disappear
 * once the process has generated a few dozen tokens: the arena is mapped,
 * the compressor has classified the hot blobs, and the expert LRU holds the
 * neighbourhood the real traffic will route to.
 *
 * It uses the CLI `run` path (cwick_engine_generate), which never touches
 * the prefix cache -- that is server-side, so the warmup prompt cannot
 * evict a real snapshot, and the generated text is discarded without
 * decoding. A failure here is logged and ignored: warmup is an
 * optimisation, never a precondition. */
static double now_mono_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void serve_warmup(cwick_engine *eng, int64_t warmup_tokens) {
    /* A generic, assistant-shaped prompt: the point is not this text but the
     * expert neighbourhood it activates, which is the one an agent's system
     * prompt activates too. */
    static const char *kPrompt =
        "You are a careful assistant. Answer briefly and precisely.\n"
        "Summarise the following in one sentence: a research team measured a "
        "small language model on a laptop, reported tokens per second, memory "
        "pressure and expert cache hit rate, then rewrote the slowest "
        "kernel and re-measured.\n";
    cwick_error err;
    cwick_error_clear(&err);
    cwick_tokenizer *tok = NULL;
    cwick_status     st = cwick_tokenizer_load(cwick_engine_model_root(eng),
                             cwick_engine_spec(eng), &tok, &err);
    if (st != CWICK_OK) {
        CWICK_INFO("init", "warmup skipped: tokenizer: %s",
                  cwick_error_message(&err));
        return;
    }
    uint32_t *ids = NULL;
    int64_t nids = 0;
    st = cwick_tokenizer_encode(tok, kPrompt, &ids, &nids, &err);
    if (st != CWICK_OK) {
        CWICK_INFO("init", "warmup skipped: tokenizer: %s",
                  cwick_error_message(&err));
        cwick_tokenizer_free(tok);
        return;
    }
    double t0 = now_mono_ms();
    cwick_gen_result res;
    memset(&res, 0, sizeof(res));
    st = cwick_engine_generate(eng, ids, nids, warmup_tokens,
                              false, &res, &err);
    double ms = now_mono_ms() - t0;
    if (st != CWICK_OK) {
        if (g_stop)
            CWICK_INFO("init", "warmup interrupted (Ctrl-C)");
        else
            CWICK_INFO("init", "warmup skipped: %s", cwick_error_message(&err));
        free(ids);
        cwick_tokenizer_free(tok);
        return;
    }
    const cwick_spine *sp = cwick_engine_spine(eng);
    uint64_t bs = 0, rc = 0, ch = 0, cm = 0;
    if (sp && sp->stream)
        cwick_stream_stats(sp->stream, &bs, &rc, &ch, &cm);
    CWICK_INFO("init", "warmup: %lld tokens in %.1f s (%.0f ms/token), expert "
              "cache %.1f%% hit (%llu/%llu), %.2f GiB streamed -- first real "
              "request will not pay this",
              (long long)res.n_tokens, ms / 1000.0,
              res.n_tokens ? ms / (double)res.n_tokens : 0.0,
              (ch + cm) ? 100.0 * (double)ch / (double)(ch + cm) : 0.0,
              (unsigned long long)ch, (unsigned long long)cm,
              cwick_gib(bs));
    cwick_gen_result_free(&res);
    free(ids);
    cwick_tokenizer_free(tok);
}

static int run_serve(const char *root, const char *host, int port,
                     int64_t max_tokens, int64_t max_context,
                     const char *prefix_cache_dir, int64_t prefix_cache_gib,
                     const char *spine, int verbose, int trace, int web,
                     int jev, const char *password,
                     const char *const *models_dirs, int models_dirs_n,
                     const char *store_path, const char *api_key,
                     const char *api_key_secret, int64_t warmup, int *ran) {
    cwick_error err;
    cwick_error_clear(&err);
    /* Capture our own stdout/stderr from here on. It is a tee, not a redirect:
     * the terminal still gets every byte, and the ring additionally answers
     * GET /v1/logs and /v1/logs/stream so the built-in UI can show what the
     * server is printing. Started before the engine so the memory budget and
     * the spine wiring land in it too — those are exactly the lines worth
     * reading when a startup fails. */
    if (cwick_logcap_start(0, 0)) {
        CWICK_INFO("init", "log capture on: stdout/stderr teed into a ring "
                          "(GET /v1/logs, GET /v1/logs/stream)");
    }
    cwick_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_root = root;
    opt.max_new_tokens = max_tokens > 0 ? max_tokens : 256;
    opt.max_context = max_context;
    opt.spine = spine;
    opt.verbose = verbose;
    opt.stop = &g_stop;
    /* Bring the CPU worker pool up before the engine: a 306 GiB checkpoint
     * spends every decode step in ~1400 matrix-vector products, and building
     * the pool first means the first request never pays thread-creation cost.
     * CWICK_THREADS=1 is the documented A/B escape hatch (serial kernels). */
    int nthr = cwick_pool_init();
    CWICK_INFO("init", "cpu worker pool: %d thread%s for the decode GEMV "
                      "kernels (CWICK_THREADS to override)",
              nthr > 0 ? nthr : 1, nthr > 1 ? "s" : "");
    cwick_engine *eng = NULL;
    cwick_status st = cwick_engine_new(&opt, &eng, &err);
    if (st != CWICK_OK) {
        /* Nothing was ever started, so nothing is being shut down: say that
         * plainly (a "shutdown complete" line here would be a lie) and put the
         * reason on stdout too, where the rest of the run log lives. */
        CWICK_INFO("init", "startup failed: %s", cwick_error_message(&err));
        fprintf(stderr, "cwick: %s\n", cwick_error_message(&err));
        return 1;
    }
    /* Warm before the port opens: a request that arrives during the warmup
     * would wait for the socket, not for generation. */
    if (warmup > 0) serve_warmup(eng, warmup);
    cwick_server_options sopts;
    memset(&sopts, 0, sizeof(sopts));
    sopts.host = host;
    sopts.port = port;
    sopts.max_tokens = max_tokens;
    sopts.prefix_cache_dir = prefix_cache_dir;
    sopts.prefix_cache_gib = prefix_cache_gib;
    sopts.verbose = verbose;
    sopts.trace = trace;
    sopts.web = web != 0;
    sopts.jev = jev != 0;
    sopts.password = (password && *password) ? password :
                     (password ? "" : NULL);
    sopts.api_key = api_key;
    sopts.api_key_secret = api_key_secret;
    /* A key is only meaningful if something checks it, so asking for one turns
     * login on (with a generated password when none was given) — exactly what
     * --lan does, for exactly the same reason. */
    if ((api_key || (api_key_secret && *api_key_secret)) && !sopts.password)
        sopts.password = "";
    sopts.wake_fd = g_wake_pipe[0];
    for (int i = 0; i < models_dirs_n; i++) sopts.models_dirs[i] = models_dirs[i];
    sopts.models_dirs_n = models_dirs_n;
    sopts.store_path = store_path;
    /* The engine pointer handed to the server may be replaced: a model switch
     * through POST /v1/models/switch frees the startup engine and builds a new
     * one, so the server reports which engine is live at exit and main frees
     * that one (never a pointer that was already freed). */
    cwick_engine *live = eng;
    sopts.live_engine = &live;
    *ran = 1;                     /* the engine exists from here on */
    st = cwick_server_run(eng, &sopts, &g_stop, &err);
    if (st != CWICK_OK) {
        CWICK_INFO("init", "server stopped with an error: %s",
                  cwick_error_message(&err));
        fprintf(stderr, "cwick: %s\n", cwick_error_message(&err));
        cwick_engine_free(live);
        cwick_pool_shutdown();
        return 1;
    }
    cwick_engine_free(live);
    /* After the spine: expert cache blobs live in the Metal arena, so the
     * arenas must outlive them, and the worker pool must outlive the last
     * kernel call. */
    cwick_pool_shutdown();
    return 0;
}

/* Arm the thinking ceiling for a CLI run when CWICK_THINK_BUDGET is set, the
 * same way the server does it per request from the client's level. A number is
 * all the CLI needs, and having it here is what makes the budget testable on a
 * real checkpoint: a second server would need another copy of the weights. */
static void cli_arm_think_budget(cwick_engine *eng, const cwick_tokenizer *tok,
                                 const uint32_t *ids, int64_t nids) {
    const char *v = getenv("CWICK_THINK_BUDGET");
    if (!v || !*v) return;
    char *end = NULL;
    long long budget = strtoll(v, &end, 10);
    if (end == v || budget < 0) return;
    static const char *lit[2] = { "<think>", "</think>" };
    int64_t mark[2] = { -1, -1 };
    for (int i = 0; i < 2; i++) {
        cwick_error e;
        cwick_error_clear(&e);
        uint32_t *m = NULL;
        int64_t n = 0;
        if (cwick_tokenizer_encode(tok, lit[i], &m, &n, &e) == CWICK_OK &&
            m && n == 1)
            mark[i] = (int64_t)m[0];
        free(m);
    }
    if (mark[0] < 0 || mark[1] < 0) {
        CWICK_INFO("run", "CWICK_THINK_BUDGET ignored: the tokenizer does not "
                   "give <think>/</think> as single tokens");
        return;
    }
    const bool open_at_start = nids > 0 && (int64_t)ids[nids - 1] == mark[0];
    cwick_engine_set_think_budget(eng, (int64_t)budget, mark[0], mark[1],
                                  open_at_start);
    CWICK_INFO("run", "thinking budget %lld tokens (CWICK_THINK_BUDGET)%s",
               budget, open_at_start ? ", block already open" : "");
}

static int run_generate(const char *root, const char *prompt, int64_t max_new,
                        int64_t max_context, const char *spine, bool chat,
                        bool verbose, bool trace) {
    cwick_error err;
    cwick_error_clear(&err);
    cwick_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_root = root;
    opt.max_new_tokens = max_new;
    opt.max_context = max_context;
    opt.spine = spine;
    opt.verbose = verbose;
    opt.stop = &g_stop;
    cwick_pool_init();
    cwick_engine *eng = NULL;
    cwick_status st = cwick_engine_new(&opt, &eng, &err);
    if (st != CWICK_OK) {
        fprintf(stderr, "cwick: %s\n", cwick_error_message(&err));
        return 1;
    }

    cwick_tokenizer *tok = NULL;
    st = cwick_tokenizer_load(root, cwick_engine_spec(eng), &tok, &err);
    if (st != CWICK_OK) {
        fprintf(stderr, "cwick: %s\n", cwick_error_message(&err));
        cwick_engine_free(eng);
        return 1;
    }

    uint32_t *ids = NULL;
    int64_t nids = 0;
    /* From here on the process is doing the work of one request; the engine
     * and tokenizer banners above belong to no request and stay untagged. */
    cwick_log_set_request(1);
    if (chat) {
        st = cwick_chat_encode(tok, "You are a helpful assistant.", prompt,
                              &ids, &nids, &err);
    } else {
        st = cwick_tokenizer_encode(tok, prompt, &ids, &nids, &err);
    }
    if (st != CWICK_OK) {
        fprintf(stderr, "cwick: %s\n", cwick_error_message(&err));
        cwick_tokenizer_free(tok);
        cwick_engine_free(eng);
        return 1;
    }

    if (trace) {
        CWICK_TRAC("run", "prompt: %lld tokens, max-new %lld, chat=%s",
                  (long long)nids, (long long)max_new, chat ? "yes" : "no");
        cwick_log_block(CWICK_LOG_TRACE, "run", prompt, 16384, 4);
    }

    cwick_gen_result res;
    memset(&res, 0, sizeof(res));
    cli_arm_think_budget(eng, tok, ids, nids);
    st = cwick_engine_generate(eng, ids, nids, max_new, false, &res, &err);
    if (st != CWICK_OK) {
        if (g_stop)
            fprintf(stderr, "cwick: interrupted\n");
        else
            fprintf(stderr, "cwick: %s\n", cwick_error_message(&err));
        free(ids);
        cwick_tokenizer_free(tok);
        cwick_engine_free(eng);
        return 1;
    }

    char *text = NULL;
    st = cwick_tokenizer_decode(tok, res.tokens, res.n_tokens, &text, &err);
    if (st == CWICK_OK) {
        fwrite(text, 1, strlen(text), stdout);
        fputc('\n', stdout);
        fflush(stdout);
        if (trace) cwick_log_block(CWICK_LOG_TRACE, "run", text, 16384, 4);
        free(text);
    }
    {
        /* The DeepSeek backend owns its own stream, so read the cache numbers
         * from whichever backend is live -- a spine lookup reports 0/0 for a
         * model that is in fact pinning experts. */
        cwick_stream *stream = cwick_engine_expert_stream(eng);
        uint64_t bs = 0, rc = 0, ch = 0, cm = 0;
        if (stream)
            cwick_stream_stats(stream, &bs, &rc, &ch, &cm);
        CWICK_INFO("run", "%lld generated tokens (finish: %s), expert cache "
                  "%.1f%% hit (%llu/%llu), %.2f GiB streamed, %llu reads, "
                  "rss %.2f GiB",
                  (long long)res.n_tokens,
                  res.stop_reason_length ? "length" : "stop",
                  (ch + cm) ? 100.0 * (double)ch / (double)(ch + cm) : 0.0,
                  (unsigned long long)ch, (unsigned long long)cm,
                  cwick_gib(bs), (unsigned long long)rc,
                  cwick_gib(cwick_process_rss_bytes()));
        if (verbose && stream) {
            CWICK_VERB("run", "expert pin set: %.2f GiB used of %.2f GiB cap",
                      cwick_gib(cwick_stream_cache_used(stream)),
                      cwick_gib(cwick_stream_cache_cap(stream)));
            const cwick_spine *sp = cwick_engine_spine(eng);
            const cwick_mem_plan *plan = sp ? cwick_spine_memory_plan(sp) : NULL;
            if (plan)
                CWICK_VERB("run", "memory plan: %.0f%% of %.2f GiB host "
                          "(resident %.2f GiB, cache %.2f GiB)",
                          plan->host_fraction * 100.0,
                          cwick_gib(plan->host_bytes),
                          cwick_gib(plan->resident_bytes),
                          cwick_gib(plan->expert_cache_bytes));
        }
    }
    free(ids);
    cwick_gen_result_free(&res);
    cwick_tokenizer_free(tok);
    cwick_engine_free(eng);
    cwick_pool_shutdown();
    return 0;
}

/* Decode several prompts together.
 *
 * This is the batch-decode harness: R engines share one weight spine, and
 * their decode steps share one expert fetch. The point is the two numbers it
 * prints per run -- each sequence's own tok/s, and the aggregate -- because
 * whether batching is worth its complexity is exactly the ratio between the
 * batched run and the same prompts run one at a time.
 *
 * It is also the correctness check: a batched sequence must produce byte-for-
 * byte what it produces alone, so `cwick run --prompt A` and
 * `cwick run --prompt A --prompt B` have to agree on A. */
#define CWICK_RUN_MAX_PROMPTS 8

static int run_generate_group(const char *root, const char *const *prompts,
                              int R, int64_t max_new, int64_t max_context,
                              const char *spine, bool chat, bool verbose,
                              bool trace) {
    cwick_error err;
    cwick_error_clear(&err);
    cwick_pool_init();

    cwick_engine *eng[CWICK_RUN_MAX_PROMPTS];
    cwick_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_root = root;
    opt.max_new_tokens = max_new;
    opt.max_context = max_context;
    opt.spine = spine;
    opt.verbose = verbose;
    opt.stop = &g_stop;
    opt.slot_id = 0;
    if (cwick_engine_new(&opt, &eng[0], &err) != CWICK_OK) {
        fprintf(stderr, "cwick: %s\n", cwick_error_message(&err));
        return 1;
    }
    /* Slots 1..R-1 borrow slot 0's spine: the weights, the shard index, the
     * F_NOCACHE stream and the expert cache are all pointers, so a second
     * sequence costs only its own per-sequence state. */
    for (int i = 1; i < R; i++) {
        opt.slot_id = i;
        opt.shared_spine = cwick_engine_spine(eng[0]);
        if (cwick_engine_new(&opt, &eng[i], &err) != CWICK_OK) {
            fprintf(stderr, "cwick: %s\n", cwick_error_message(&err));
            for (int j = 0; j < i; j++) cwick_engine_free(eng[j]);
            return 1;
        }
    }

    cwick_tokenizer *tok = NULL;
    if (cwick_tokenizer_load(root, cwick_engine_spec(eng[0]), &tok, &err) != CWICK_OK) {
        fprintf(stderr, "cwick: %s\n", cwick_error_message(&err));
        for (int j = 0; j < R; j++) cwick_engine_free(eng[j]);
        return 1;
    }

    uint32_t *ids[CWICK_RUN_MAX_PROMPTS];
    int64_t nids[CWICK_RUN_MAX_PROMPTS];
    for (int i = 0; i < R; i++) {
        ids[i] = NULL; nids[i] = 0;
        cwick_status st = chat
            ? cwick_chat_encode(tok, "You are a helpful assistant.", prompts[i],
                               &ids[i], &nids[i], &err)
            : cwick_tokenizer_encode(tok, prompts[i], &ids[i], &nids[i], &err);
        if (st != CWICK_OK) {
            fprintf(stderr, "cwick: %s\n", cwick_error_message(&err));
            for (int j = 0; j < i; j++) free(ids[j]);
            cwick_tokenizer_free(tok);
            for (int j = 0; j < R; j++) cwick_engine_free(eng[j]);
            return 1;
        }
        if (trace) {
            CWICK_TRAC("group", "sequence %d: %lld tokens, max-new %lld, chat=%s",
                      i, (long long)nids[i], (long long)max_new,
                      chat ? "yes" : "no");
            cwick_log_block(CWICK_LOG_TRACE, "group", prompts[i], 16384, 4);
        }
    }

    cwick_gen_result res[CWICK_RUN_MAX_PROMPTS];
    memset(res, 0, sizeof(res));
    struct timespec ts_run;
    clock_gettime(CLOCK_MONOTONIC, &ts_run);
    double t0 = (double)ts_run.tv_sec * 1000.0 + (double)ts_run.tv_nsec / 1.0e6;
    cwick_status st = cwick_engine_generate_group(eng, (const uint32_t *const *)ids,
                                                nids, R, max_new, false,
                                                res, &err);
    clock_gettime(CLOCK_MONOTONIC, &ts_run);
    double wall_ms = ((double)ts_run.tv_sec * 1000.0 +
                      (double)ts_run.tv_nsec / 1.0e6) - t0;
    if (st != CWICK_OK) {
        fprintf(stderr, "cwick: %s\n", cwick_error_message(&err));
        for (int i = 0; i < R; i++) { free(ids[i]); cwick_gen_result_free(&res[i]); }
        cwick_tokenizer_free(tok);
        for (int i = 0; i < R; i++) cwick_engine_free(eng[i]);
        return 1;
    }

    int64_t total = 0;
    for (int i = 0; i < R; i++) {
        char *text = NULL;
        if (cwick_tokenizer_decode(tok, res[i].tokens, res[i].n_tokens, &text,
                                  &err) == CWICK_OK && text) {
            printf("--- sequence %d (%lld tokens, finish=%s) ---\n", i,
                   (long long)res[i].n_tokens,
                   res[i].stop_reason_length ? "length" : "stop");
            fwrite(text, 1, strlen(text), stdout);
            fputc('\n', stdout);
            free(text);
        }
        total += res[i].n_tokens;
        CWICK_INFO("group", "seq %d: %lld tokens, prefill %.0f ms, decode %.0f ms "
                  "(%lld shared steps) -> %.3f tok/s (this sequence)",
                  i, (long long)res[i].n_tokens, res[i].prefill_ms,
                  res[i].decode_ms, (long long)res[i].decode_steps,
                  res[i].decode_ms > 0
                      ? (double)res[i].n_tokens * 1000.0 / res[i].decode_ms
                      : 0.0);
    }
    CWICK_INFO("group", "%d sequence(s), %lld tokens in %.0f ms wall -> "
              "%.3f tok/s aggregate", R, (long long)total, wall_ms,
              wall_ms > 0 ? (double)total * 1000.0 / wall_ms : 0.0);
    {
        /* Same stream accounting a single run reports: without it a group
         * that is merely re-reading (not sharing) looks like a slow group. */
        cwick_stream *stream = cwick_engine_expert_stream(eng[0]);
        uint64_t bs = 0, rc = 0, ch = 0, cm = 0;
        if (stream)
            cwick_stream_stats(stream, &bs, &rc, &ch, &cm);
        CWICK_INFO("group", "expert cache %.1f%% hit (%llu/%llu), %.2f GiB "
                  "streamed, %llu reads, rss %.2f GiB",
                  (ch + cm) ? 100.0 * (double)ch / (double)(ch + cm) : 0.0,
                  (unsigned long long)ch, (unsigned long long)cm,
                  cwick_gib(bs), (unsigned long long)rc,
                  cwick_gib(cwick_process_rss_bytes()));
    }

    for (int i = 0; i < R; i++) { free(ids[i]); cwick_gen_result_free(&res[i]); }
    cwick_tokenizer_free(tok);
    for (int i = 0; i < R; i++) cwick_engine_free(eng[i]);
    cwick_pool_shutdown();
    return 0;
}

/* Validate a required dense tensor of the real checkpoint. */
static int doctor_require(const cwick_safetensors *saf, const char *name,
                          const char *what) {
    cwick_st_tensor t;
    cwick_error err;
    cwick_error_clear(&err);
    if (cwick_safetensors_get(saf, name, &t, &err) != CWICK_OK) {
        fprintf(stderr, "cwick doctor: FAIL missing %s (%s)\n", name, what);
        return 0;
    }
    return 1;
}

static int run_doctor(const char *root, bool verbose, bool deep,
                      int64_t expert_sample) {
    cwick_error err;
    cwick_error_clear(&err);
    char cfg[1152];
    cwick_path_join(cfg, sizeof(cfg), root, "config.json");
    cwick_model_spec spec;
    cwick_status st = cwick_model_spec_load(cfg, &spec, &err);
    if (st != CWICK_OK) {
        fprintf(stderr, "cwick: %s\n", cwick_error_message(&err));
        return 1;
    }
    printf("cwick doctor: model %s (%lld hidden, %lld layers, %lld experts, "
           "top-k %lld, %d KDA / %d DSA / %d MLA layers, %s)\n",
           spec.model_type, (long long)spec.hidden_size,
           (long long)spec.num_hidden_layers, (long long)spec.num_experts,
           (long long)spec.num_experts_per_token, (int)spec.kda_layers,
           (int)spec.dsa_layers, (int)spec.mla_layers,
           spec.quant == CWICK_QUANT_FP8_E4M3 ? "fp8/e4m3" : "bf16");

    char idxpath[1152];
    cwick_path_join(idxpath, sizeof(idxpath), root, "model.safetensors.index.json");
    if (cwick_file_exists(idxpath)) {
        /* ---- real HF checkpoint: validate the full weight set ---- */
        cwick_safetensors saf;
        memset(&saf, 0, sizeof(saf));
        st = cwick_safetensors_open(root, &saf, &err);
        if (st != CWICK_OK) {
            fprintf(stderr, "cwick: %s\n", cwick_error_message(&err));
            cwick_model_spec_free(&spec);
            return 1;
        }
        printf("cwick doctor: safetensors checkpoint ok "
               "(%lld shards, %zu tensors, %.1f GiB total)\n",
               cwick_safetensors_count_shards(&saf), saf.tensor_count,
               (double)cwick_safetensors_total_bytes(&saf) / (1024.0 * 1024.0 * 1024.0));

        int ok = 1;
        /* every name below is built for this checkpoint's own tensor layout */
        const char *lp = "model.language_model.layers.%lld.%s";
        /* global tensors */
        ok &= doctor_require(&saf, "model.language_model.embed_tokens.weight",
                             "embedding");
        ok &= doctor_require(&saf, "model.language_model.norm.weight",
                             "final norm");
        ok &= doctor_require(&saf, "lm_head.weight", "lm_head");

        /* per-layer required tensors */
        for (int64_t l = 0; l < spec.num_hidden_layers && !g_stop; l++) {
            char n[192];
            snprintf(n, sizeof(n), lp, (long long)l, "input_layernorm.weight");
            ok &= doctor_require(&saf, n, "input layernorm");
            snprintf(n, sizeof(n), lp, (long long)l,
                     "post_attention_layernorm.weight");
            ok &= doctor_require(&saf, n, "post-attention norm");
            if (spec.layers[l] == CWICK_LAYER_KDA) {
                const char *need[] = { "A_log", "dt_bias", "q_conv1d.weight",
                    "k_conv1d.weight", "v_conv1d.weight", "o_norm.weight",
                    "q_proj.weight", "k_proj.weight", "v_proj.weight",
                    "g_a_proj.weight", "g_b_proj.weight", "f_a_proj.weight",
                    "f_b_proj.weight", "b_proj.weight", "o_proj.weight" };
                for (size_t i = 0; i < sizeof(need) / sizeof(need[0]); i++) {
                    snprintf(n, sizeof(n),
                             "model.language_model.layers.%lld.self_attn.%s", l, need[i]);
                    ok &= doctor_require(&saf, n, "KDA attention weight");
                }
            } else if (spec.layers[l] == CWICK_LAYER_DSA) {
                const char *need[] = { "q_a_proj.weight", "q_b_proj.weight",
                    "kv_a_proj_with_mqa.weight", "kv_b_proj.weight", "o_proj.weight",
                    "q_a_layernorm.weight", "kv_a_layernorm.weight",
                    "indexer.wk.weight", "indexer.wq_b.weight",
                    "indexer.weights_proj.weight", "indexer.k_norm.weight" };
                for (size_t i = 0; i < sizeof(need) / sizeof(need[0]); i++) {
                    snprintf(n, sizeof(n),
                             "model.language_model.layers.%lld.self_attn.%s", l, need[i]);
                    ok &= doctor_require(&saf, n, "DSA attention/indexer weight");
                }
            } else { /* MLA */
                const char *need[] = { "q_a_proj.weight", "q_b_proj.weight",
                    "kv_a_proj_with_mqa.weight", "kv_b_proj.weight", "o_proj.weight",
                    "q_a_layernorm.weight", "kv_a_layernorm.weight" };
                for (size_t i = 0; i < sizeof(need) / sizeof(need[0]); i++) {
                    snprintf(n, sizeof(n),
                             "model.language_model.layers.%lld.self_attn.%s", l, need[i]);
                    ok &= doctor_require(&saf, n, "MLA attention weight");
                }
            }
            if (spec.mlp_dense[l]) {
                const char *need[] = { "mlp.gate_proj.weight", "mlp.up_proj.weight",
                    "mlp.down_proj.weight" };
                for (size_t i = 0; i < sizeof(need) / sizeof(need[0]); i++) {
                    snprintf(n, sizeof(n), lp, (long long)l, need[i]);
                    ok &= doctor_require(&saf, n, "dense MLP weight");
                }
            } else {
                snprintf(n, sizeof(n), lp, (long long)l,
                         "block_sparse_moe.gate.weight");
                ok &= doctor_require(&saf, n, "MoE router");
                snprintf(n, sizeof(n), lp, (long long)l,
                         "block_sparse_moe.gate.e_score_correction_bias");
                ok &= doctor_require(&saf, n, "router bias");
                snprintf(n, sizeof(n), lp, (long long)l,
                         "block_sparse_moe.shared_experts.gate_proj.weight");
                ok &= doctor_require(&saf, n, "shared expert");
            }
        }
        if (g_stop) {
            cwick_safetensors_close(&saf);
            cwick_model_spec_free(&spec);
            fprintf(stderr, "cwick doctor: interrupted\n");
            return 1;
        }

        /* the complete, never-pruned expert bank: every routed expert matrix on
         * every sparse layer plus the nextn-predict layer. */
        int64_t expert_sets = 0, expert_tensors = 0;
        for (int64_t l = 0; l <= spec.num_hidden_layers; l++) {
            if (l < spec.num_hidden_layers && spec.mlp_dense[l]) continue;
            if (l == spec.num_hidden_layers && spec.num_nextn_predict_layers <= 0)
                break;   /* no nextn-predict layer in this checkpoint */
            for (int64_t e = 0; e < spec.num_experts; e++) {
                const char *mat[3] = { "gate_proj", "up_proj", "down_proj" };
                for (int m = 0; m < 3; m++) {
                    char n[192];
                    cwick_expert_saf_name(&spec, l, e, mat[m], n, sizeof(n));
                    if (!doctor_require(&saf, n, "routed expert")) ok = 0;
                    expert_tensors++;
                }
                expert_sets++;
            }
        }
        printf("cwick doctor: expert bank complete (%lld expert sets, %lld "
               "matrices; never pruned)\n", (long long)expert_sets,
               (long long)expert_tensors);

        /* ---- really load the weights ------------------------------------
         * The checks above only prove the names resolve. On a machine whose
         * RAM is smaller than the checkpoint, "loadable" has to mean the
         * bytes can actually be read under a bounded memory budget, so
         * doctor now streams them through the F_NOCACHE path: every
         * resident tensor is read in full, and the expert bank is sampled
         * (or read in full with --deep). Memory stays at O(arena). */
        cwick_stream_options sopt;
        memset(&sopt, 0, sizeof(sopt));
        sopt.arena_bytes = 8u << 20;
        sopt.nocache = true;
        cwick_stream *strm = NULL;
        st = cwick_stream_open(&saf, &sopt, &strm, &err);
        if (st != CWICK_OK) {
            fprintf(stderr, "cwick: %s\n", cwick_error_message(&err));
            cwick_safetensors_close(&saf);
            cwick_model_spec_free(&spec);
            return 1;
        }

        /* Split the checkpoint into what must be resident and what streams. */
        uint64_t resident_bytes = 0, expert_bytes = 0;
        size_t resident_count = 0, expert_count = 0;
        cwick_safetensors_split(&saf, &resident_bytes, &expert_bytes);
        for (size_t i = 0; i < saf.tensor_count; i++) {
            const cwick_st_tensor *t = &saf.tensors[i];
            if (strstr(t->name, ".mlp.experts.")) expert_count++;
            else resident_count++;
        }
        printf("cwick doctor: weight split — resident spine %zu tensors "
               "(%.2f GiB), routed expert bank %zu tensors (%.2f GiB)\n",
               resident_count, (double)resident_bytes / (1024.0 * 1024.0 * 1024.0),
               expert_count, (double)expert_bytes / (1024.0 * 1024.0 * 1024.0));

        /* 1. every resident tensor: read all of its bytes. */
        uint64_t acc = 1469598103934665603ull;
        size_t verified = 0;
        uint64_t verified_bytes = 0;
        for (size_t i = 0; i < saf.tensor_count; i++) {
            if (g_stop) break;
            const cwick_st_tensor *t = &saf.tensors[i];
            if (strstr(t->name, ".mlp.experts.")) continue;
            cwick_st_tensor copy = *t;
            if (cwick_stream_verify(strm, &copy, &acc, &err) != CWICK_OK) {
                fprintf(stderr, "cwick doctor: FAIL %s\n", cwick_error_message(&err));
                ok = 0;
                break;
            }
            verified++;
            verified_bytes += t->data_bytes;
        }
        if (ok && !g_stop) {
            printf("cwick doctor: resident spine loaded and verified — %zu "
                   "tensors, %.2f GiB read (checksum %016llx)\n",
                   verified, (double)verified_bytes / (1024.0 * 1024.0 * 1024.0),
                   (unsigned long long)acc);
        }

        /* 2. the expert bank: full pass with --deep, otherwise a strided
         *    sample that still touches every shard. */
        if (ok && !g_stop) {
            int64_t stride = deep ? 1 : (expert_sample > 0 ? expert_sample : 61);
            int64_t sampled = 0;
            uint64_t sampled_bytes = 0;
            uint64_t eacc = 1469598103934665603ull;
            for (int64_t l = 0; l <= spec.num_hidden_layers && ok; l++) {
                if (g_stop) break;
                /* the extra iteration only exists for GLM's nextn predict
                 * layers, so stop at the last real layer when there are none */
                if (l == spec.num_hidden_layers &&
                    spec.num_nextn_predict_layers <= 0)
                    break;
                if (l < spec.num_hidden_layers && spec.mlp_dense[l]) continue;
                for (int64_t e = 0; e < spec.num_experts && ok; e++) {
                    if (g_stop) break;
                    if (stride > 1 && ((l * spec.num_experts + e) % stride) != 0) continue;
                    static const char *mat[3] = { "gate_proj", "up_proj", "down_proj" };
                    for (int m = 0; m < 3; m++) {
                        char n[192], suf[160];
                        snprintf(suf, sizeof(suf),
                                 "mlp.experts.%lld.%s.weight", (long long)e, mat[m]);
                        snprintf(n, sizeof(n), lp, (long long)l, suf);
                        cwick_st_tensor t;
                        if (cwick_safetensors_get(&saf, n, &t, &err) != CWICK_OK) {
                            fprintf(stderr, "cwick doctor: FAIL %s\n",
                                    cwick_error_message(&err));
                            ok = 0;
                            break;
                        }
                        if (cwick_stream_verify(strm, &t, &eacc, &err) != CWICK_OK) {
                            fprintf(stderr, "cwick doctor: FAIL %s\n",
                                    cwick_error_message(&err));
                            ok = 0;
                            break;
                        }
                        sampled_bytes += t.data_bytes;
                    }
                    sampled++;
                }
            }
            if (ok && !g_stop) {
                printf("cwick doctor: expert bank read — %lld/%lld expert sets "
                       "(1/%lld sample), %.2f GiB (checksum %016llx)\n",
                       (long long)sampled, (long long)expert_sets, (long long)stride,
                       (double)sampled_bytes / (1024.0 * 1024.0 * 1024.0),
                       (unsigned long long)eacc);
            }
        }

        /* 3. memory verdict: does this machine actually have room? */
        if (ok && !g_stop) {
            int64_t per_set = expert_sets > 0
                ? (int64_t)(expert_bytes / (uint64_t)expert_sets) : 0;
            int64_t per_token = per_set * spec.n_sparse_layers *
                                spec.num_experts_per_token;
            uint64_t host = cwick_host_memory_bytes();
            uint64_t arena = cwick_stream_arena_bytes(strm);

            /* Same arithmetic as the runtime's own plan (mem.c), minus the
             * inference state doctor does not allocate: this is the ceiling
             * `cwick serve` will actually live under. */
            double frac;
            uint64_t limit;
            cwick_memory_policy(&frac, &limit);
            uint64_t target = (uint64_t)((double)host * frac);
            if (limit > 0 && target > limit) target = limit;

            printf("cwick doctor: host memory %.1f GiB\n",
                   (double)host / (1024.0 * 1024.0 * 1024.0));
            printf("cwick doctor: at most %.0f%% of RAM (%.2f GiB) is used by "
                   "cwick; the rest of the checkpoint streams\n",
                   frac * 100.0, (double)target / (1024.0 * 1024.0 * 1024.0));
            printf("cwick doctor: resident spine %.2f GiB + stream arena %.1f MiB "
                   "= %.2f GiB steady-state RSS\n",
                   (double)resident_bytes / (1024.0 * 1024.0 * 1024.0),
                   (double)arena / (1024.0 * 1024.0),
                   (double)(resident_bytes + arena) / (1024.0 * 1024.0 * 1024.0));
            printf("cwick doctor: per decode step streams %.2f GiB of experts "
                   "(%lld sparse layers x top-%lld x %.1f MiB)\n",
                   (double)per_token / (1024.0 * 1024.0 * 1024.0),
                   (long long)spec.n_sparse_layers,
                   (long long)spec.num_experts_per_token,
                   (double)per_set / (1024.0 * 1024.0));
            printf("cwick doctor: F_NOCACHE streaming %s\n",
                   cwick_stream_nocache_active(strm) ? "active (streamed bytes "
                   "are not retained in RAM)" : "UNAVAILABLE (kernel may retain "
                   "streamed bytes)");
            if (host > 0 && resident_bytes + arena > target) {
                fprintf(stderr, "cwick doctor: FAIL resident spine %.2f GiB exceeds "
                        "the %.2f GiB budget for this machine\n",
                        (double)(resident_bytes + arena) / (1024.0 * 1024.0 * 1024.0),
                        (double)target / (1024.0 * 1024.0 * 1024.0));
                ok = 0;
            }
        }

        /* The verification pass went through F_NOCACHE pread(), so it never
         * faulted the shared mapping in; there is nothing to reclaim. */
        cwick_stream_close(strm);
        cwick_safetensors_close(&saf);
        if (g_stop) {
            fprintf(stderr, "cwick doctor: interrupted\n");
            cwick_model_spec_free(&spec);
            return 1;
        }
        if (!ok) {
            fprintf(stderr, "cwick doctor: FAILED (weights could not be loaded)\n");
            cwick_model_spec_free(&spec);
            return 1;
        }
        printf("cwick doctor: ok — the full %s checkpoint is loadable by this "
               "runtime\n",
               "zai-org/GLM-5.3-Flash");
        cwick_model_spec_free(&spec);
        (void)verbose;
        return 0;
    }

    /* ---- converted/synthetic format ---- */
    {
        printf("cwick doctor: model %s (%lld hidden, %lld layers, %lld experts, "
               "top-k %lld, %d KDA / %d DSA / %d MLA layers)\n",
               spec.model_type, (long long)spec.hidden_size,
               (long long)spec.num_hidden_layers, (long long)spec.num_experts,
               (long long)spec.num_experts_per_token, (int)spec.kda_layers,
               (int)spec.dsa_layers, (int)spec.mla_layers);
    }
    cwick_spine spine;
    memset(&spine, 0, sizeof(spine));
    st = cwick_spine_open(root, &spec, "bf16", &spine, &err);
    if (st != CWICK_OK) {
        fprintf(stderr, "cwick: %s\n", cwick_error_message(&err));
        cwick_model_spec_free(&spec);
        return 1;
    }
    printf("cwick doctor: spine ok (%zu tensors, %zu expert files)\n",
           spine.tensor_count, spine.expert_file_count);
    cwick_spine_close(&spine);
    cwick_model_spec_free(&spec);
    (void)verbose;
    return 0;
}

static void usage(void) {
    printf("cwick — full-model C runtime + OpenAI server\n"
           "usage:\n"
           "  cwick setup --root DIR            write a synthetic verification model\n"
           "  cwick run --root DIR --prompt P [--max-new N] [--spine bf16|int8] [--chat]\n"
           "                                  [--context N] [--backend metal|cpu]\n"
           "                                  [--verbose] [--trace]\n"
           "  cwick serve --root DIR [--host H] [--port P] [--max-tokens N]\n"
           "                        [--context N] [--spine bf16|int8]\n"
           "                        [--warmup N]\n"
           "                        [--backend metal|cpu] [--verbose] [--trace]\n"
           "                        [--mem-fraction PCT] [--max-memory GiB]\n"
           "                        [--allow-swap] [--web] [--lan]\n"
           "  cwick doctor --root DIR [--deep] [--expert-sample N]\n"
           "\n"
           "compute backend:\n"
           "  --backend metal                  run the GPU decode path\n"
           "  --backend cpu                    force the CPU fallback\n"
           "  Metal is the default on Apple Silicon; CWICK_METAL=1 also enables it\n"
           "  For Qwen3.5 the GPU takes the whole decode step (32 layers in one\n"
           "  command buffer, 20 tok/s against 7 on the CPU pool); set\n"
           "  CWICK_Q35_GPU=0 to keep the CPU pool for that model.\n"
           "  If no usable Metal device is found, cwick prints a warning to\n"
           "  stderr and runs the CPU kernels instead: identical output, slower\n"
           "  decode. A GPU is never allowed to break a generation.\n"
           "\n"
           "web UI:\n"
           "  --web           serve a built-in chat app at http://HOST:PORT/\n"
           "                  (CSS/JS included, compiled into the binary). It\n"
           "                  calls this same server over /v1/chat/completions,\n"
           "                  /v1/responses or /v1/messages (the composer picks\n"
           "                  the dialect), so there is no extra process, port or\n"
           "                  CORS setup.\n"
           "                  Streams, shows the reasoning block separately,\n"
           "                  maps 低/中/高/超高/极致 to reasoning_effort, can loop\n"
           "                  tool calls through three sample local functions,\n"
           "                  and renders /v1/metrics as live cards and trend\n"
           "                  charts. Every request is same-origin: there is no\n"
           "                  option to point the console at an external OpenAI\n"
           "                  service. It also has a logs tab (GET /v1/logs/stream,\n"
           "                  the process's own stdout/stderr), a settings tab\n"
           "                  (GET /v1/config, POST /v1/models/switch, metric\n"
           "                  trends, a Prometheus series browser, model info,\n"
           "                  generation preferences) and a\n"
           "                  model tab (GET /v1/hub/models, /v1/hub/downloads,\n"
           "                  /v1/hub/search; POST /v1/hub/download).\n"
           "  --no-web        turn it back off (or CWICK_WEB=1 to default on)\n"
           "  --lan           bind 0.0.0.0 and enable --web so a phone on the\n"
           "                  same Wi-Fi can scan the app's QR. Forces single-user\n"
           "                  password login (generated if none is set); HTTP,\n"
           "                  no TLS, so use a trusted Wi-Fi or a TLS proxy.\n"
           "  --password P    single-user login password (or CWICK_PASSWORD).\n"
           "\n"
           "model library:\n"
           "  GET  /v1/hub/models, /v1/hub/downloads, /v1/hub/search?q=...\n"
           "  POST /v1/hub/download {\"repo\":\"owner/name\"}\n"
           "  POST /v1/hub/download/<id>/<pause|resume|retry|cancel>\n"
           "                  Downloads land in ./models/owner/name and resume\n"
           "                  from <file>.incomplete with a Range request. Partial\n"
           "                  files survive a restart; sizes are 64-bit, so\n"
           "                  shards above 100 GiB need no special handling.\n"
           "                  These endpoints never take the generation slot.\n"
           "                  CWICK_HUB_ENDPOINT overrides the catalog host,\n"
           "                  then HF_ENDPOINT; the default is hf-mirror.\n"
           "\n"
           "decision endpoint:\n"
           "  --jev           serve POST /v1/systemone (also reachable as\n"
           "                  /inference/v1/systemone): a state plus up to ten\n"
           "                  choice/score/yes-no questions, answered from one\n"
           "                  prefill each with no generated tokens. The readout\n"
           "                  is the next-token distribution over the option\n"
           "                  letters, so there is no JSON to parse and no retry\n"
           "                  when the model writes prose instead of an answer.\n"
           "                  The prompt grammar and the readable letter set are\n"
           "                  derived from the resident checkpoint and re-derived\n"
           "                  after every POST /v1/models/switch; GET /v1/systemone\n"
           "                  reports the live adapter. The hosted apps use this\n"
           "                  endpoint instead of /v1/chat/completions.\n"
           "  --no-jev        turn it back off (or CWICK_JEV=1 to default on)\n"
           "                  With --lan and no password, one is generated and\n"
           "                  printed once on stdout.\n"
           "  --api-key NAME  mint an API key for third-party agents and print\n"
           "                  it once on stdout (only its hash is stored, in the\n"
           "                  sqlite file, so it survives a restart). Sending one\n"
           "                  turns login on, using a generated password if none\n"
           "                  was set. Clients send it as\n"
           "                  `Authorization: Bearer <key>` or `X-API-Key: <key>`;\n"
           "                  more keys can be minted and revoked with\n"
           "                  POST|GET|DELETE /v1/auth/keys (a login session is\n"
           "                  required for those; a key cannot mint keys).\n"
           "                  CWICK_API_KEY=SECRET adopts a secret you chose\n"
           "                  instead of generating one.\n"
           "\n"
           "other local models:\n"
           "  --models-dir DIR   also scan DIR (and its immediate children) for\n"
           "                  runnable checkpoints, up to 8 times. They are only\n"
           "                  *listed* — GET /v1/models adds them to the catalog\n"
           "                  and POST /v1/models/switch makes one resident,\n"
           "                  which is how `--root A --models-dir DIR` avoids\n"
           "                  paying for two models' weights at once. A switch\n"
           "                  frees the old spine and clears the prefix cache\n"
           "                  before mapping the new one, and is refused (409)\n"
           "                  while a request is generating or queued.\n"
           "\n"
           "app storage:\n"
           "  --store DIR     one sqlite database per app, <id>.db in DIR\n"
           "                  (plus auth.db for this server's API keys).\n"
           "                  Default $CWICK_STORE, else ~/.cwick/apps.\n"
           "                  --store FILE keeps the old single shared file.\n"
           "                  Served as /v1/apps/<id>/{state,events,games};\n"
           "                  GET /v1/store reports the layout, the files and\n"
           "                  the row counts. It lives server-side rather than\n"
           "                  in localStorage so that clearing browser data\n"
           "                  does not erase history.\n"
           "\n"
           "logging (stdout):\n"
           "  --verbose   startup, model load, memory budget, per-request and\n"
           "              per-decode-step detail, errors, shutdown\n"
           "  --trace     also print what the model saw and produced: prompts,\n"
           "              outputs and tool calls. Off by default because it\n"
           "              puts user content on stdout; payloads are truncated.\n"
           "\n"
           "context:\n"
           "  --context N     KV capacity in tokens. `serve` defaults to\n"
           "                  131072 (>100K); `run` defaults to 4096, or the\n"
           "                  DSA indexer's top-k when the checkpoint is FP8.\n"
           "                  The DSA layers keep one 512-float latent vector\n"
           "                  per position (~2.75 GiB at 128K for 11 layers)\n"
           "                  and the sparse indexer selects top-2048 of them,\n"
           "                  so a 128K window fits in the 80%% budget alongside\n"
           "                  the resident weights. The checkpoint itself allows\n"
           "                  up to 1048576 positions.\n"
           "\n"
           "memory:\n"
           "  --mem-fraction PCT  share of host RAM cwick may occupy (default 80)\n"
           "  --max-memory GiB    absolute ceiling, wins when smaller\n"
           "  CWICK_MEM_PCT / CWICK_MEM_MAX_GIB do the same via the environment\n"
           "  Host RAM is detected at startup; whatever the ceiling leaves over\n"
           "  is served by F_NOCACHE SSD streaming.\n"
           "  Staying under the ceiling is not enough on its own: macOS\n"
           "  compresses anonymous pages before it swaps them, and a\n"
           "  compressed page costs a fault to read back. So by default cwick\n"
           "  *wires* the resident spine and the expert arena — the VM can\n"
           "  neither compress nor page them. --allow-swap opts out.\n"
           "\n"
           "startup warmup:\n"
           "  --warmup N     before opening the port, run one silent generation\n"
           "                  of N tokens on a throwaway prompt (CWICK_WARMUP=N\n"
           "                  does the same). The output is discarded and the\n"
           "                  prompt is not decoded. It exists to pay the cold\n"
           "                  start before a client can: the first decode steps\n"
           "                  of a fresh process grow the expert arena\n"
           "                  (shmem_alloc ~189 ms/step) and start from an\n"
           "                  empty cache (17%% vs ~68%% hit -- 1013 vs 590\n"
           "                  ms/step measured on this build). 32 tokens is\n"
           "                  ample; 0 or absent means off. A failure is\n"
           "                  logged and startup continues regardless.\n"
           "\n"
           "Ctrl+C stops `serve`: it breaks the accept loop, aborts an in-flight\n"
           "decode at the next token, then joins the prefetch thread, closes the\n"
           "shard descriptors and releases the Metal device. A second Ctrl+C\n"
           "exits immediately.\n");
}

int main(int argc, char **argv) {
    const char *cmd = argc > 1 ? argv[1] : "help";
    if (strcmp(cmd, "help") == 0 || strcmp(cmd, "--help") == 0 || strcmp(cmd, "-h") == 0) {
        usage();
        return 0;
    }
    install_signal_handlers();

    if (strcmp(cmd, "setup") == 0) {
        const char *root = NULL;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--root") == 0) root = need_arg(&i, argc, argv, "--root");
        }
        if (!root) { fprintf(stderr, "cwick setup: --root DIR required\n"); return 2; }
        cwick_error err;
        cwick_error_clear(&err);
        cwick_status st = cwick_setup_synthetic(root, &err);
        if (st != CWICK_OK) {
            fprintf(stderr, "cwick: %s\n", cwick_error_message(&err));
            return 1;
        }
        printf("cwick setup: synthetic model written under %s\n", root);
        return 0;
    }

    if (strcmp(cmd, "run") == 0) {
        /* Up to CWICK_RUN_MAX_PROMPTS prompts: with more than one, `run` stops
         * being a demo and becomes the batch-decode harness -- the sequences
         * are decoded together and share one expert fetch per step. */
        const char *root = NULL, *spine = "bf16";
        const char *prompts[CWICK_RUN_MAX_PROMPTS];
        int nprompt = 0;
        int64_t max_new = 16;
        int64_t max_context = 0;
        bool chat = false, verbose = false, trace = false;
        int backend = -1; /* -1=auto/env, 1=metal, 0=cpu */
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--root") == 0) root = need_arg(&i, argc, argv, "--root");
            else if (strcmp(argv[i], "--prompt") == 0) {
                const char *p = need_arg(&i, argc, argv, "--prompt");
                if (nprompt >= CWICK_RUN_MAX_PROMPTS) {
                    fprintf(stderr, "cwick run: at most %d --prompt arguments\n",
                            CWICK_RUN_MAX_PROMPTS);
                    return 2;
                }
                prompts[nprompt++] = p;
            }
            else if (strcmp(argv[i], "--max-new") == 0) {
                if (!parse_i64(need_arg(&i, argc, argv, "--max-new"),
                               1, 1000000, &max_new)) {
                    fprintf(stderr, "cwick run: --max-new must be 1..1000000\n"); return 2;
                }
            }
            else if (strcmp(argv[i], "--context") == 0) {
                if (!parse_i64(need_arg(&i, argc, argv, "--context"),
                               1, 1048576, &max_context)) {
                    fprintf(stderr, "cwick run: --context must be 1..1048576\n"); return 2;
                }
            }
            else if (strcmp(argv[i], "--spine") == 0) spine = need_arg(&i, argc, argv, "--spine");
            else if (strcmp(argv[i], "--chat") == 0) chat = true;
            else if (strcmp(argv[i], "--verbose") == 0) verbose = true;
            else if (strcmp(argv[i], "--trace") == 0) trace = true;
            else if (strcmp(argv[i], "--backend") == 0) {
                const char *v = need_arg(&i, argc, argv, "--backend");
                if (strcmp(v, "metal") == 0) backend = 1;
                else if (strcmp(v, "cpu") == 0) backend = 0;
                else { fprintf(stderr, "cwick run: --backend must be 'metal' or 'cpu'\n"); return 2; }
            }
            else {
                fprintf(stderr, "cwick run: unknown option `%s`\n", argv[i]);
                return 2;
            }
        }
        if (strcmp(spine, "bf16") != 0 && strcmp(spine, "int8") != 0) {
            fprintf(stderr, "cwick run: --spine must be bf16 or int8\n");
            return 2;
        }
        if (!root || nprompt == 0) { fprintf(stderr, "cwick run: --root and --prompt required\n"); return 2; }
        configure_logging(verbose, trace);
        select_backend(backend);
        if (trace) trace_privacy_notice();
        if (!instance_claim(root)) return 1;
        if (nprompt == 1)
            return run_generate(root, prompts[0], max_new, max_context, spine,
                                chat, verbose, trace);
        return run_generate_group(root, prompts, nprompt, max_new, max_context,
                                  spine, chat, verbose, trace);
    }

    if (strcmp(cmd, "serve") == 0) {
        const char *root = NULL, *host = "127.0.0.1";
        int port = 8000;
        int backend = -1; /* -1=auto/env, 1=metal, 0=cpu */
        /* Default per-response ceiling: 1000000 is the "no explicit cap"
         * sentinel -- generation runs until EOS or until the context window
         * fills (the engine clamps max_new to what is left of it). Override
         * with --max-tokens N. */
        int64_t max_tokens = 1000000;
        /* Serve defaults to a >100K window: GLM-5.3 supports 1M positions,
         * and 131072 tokens of MLA latent KV costs ~2.75 GiB, which the 80%
         * plan can afford. `run` keeps its smaller default because it is the
         * one-shot CLI. An explicit --context always wins. */
        int64_t max_context = 131072;
        int64_t prefix_cache_gib = 0;    /* 0 = default 8 GiB */
        int64_t warmup = 0;              /* 0 = no startup warmup */
        {   /* CWICK_WARMUP for scripts that do not pass flags; an explicit
             * --warmup (parsed below, later) wins. */
            const char *we = getenv("CWICK_WARMUP");
            if (we && *we) {
                long v = strtol(we, NULL, 10);
                if (v >= 1 && v <= 10000) warmup = (int64_t)v;
            }
        }
        const char *prefix_cache_dir = getenv("CWICK_PREFIX_CACHE_DIR");
        bool verbose = false, trace = false;
        bool allow_swap = false;
        /* CWICK_WEB=1 turns the UI on for everyone who starts the same
         * service script; --web / --no-web override it. */
        bool web = cwick_env_true("CWICK_WEB");
        /* CWICK_JEV=1 turns the decision endpoint on for a launcher; --jev and
         * --no-jev override it. */
        bool jev = cwick_env_true("CWICK_JEV");
        bool lan = false;
        const char *password = getenv("CWICK_PASSWORD");
        /* Extra local models to advertise and allow switching to. Repeatable;
         * each directory is scanned (itself plus its immediate children) for
         * cwick checkpoints. */
        const char *models_dirs[8];
        int models_dirs_n = 0;
        /* sqlite file for the app market's own data; NULL = $CWICK_STORE or
         * ~/.cwick/apps.db (chosen inside the store, which also creates it). */
        const char *store_path = getenv("CWICK_STORE");
        /* API key for non-browser clients (--api-key NAME, CWICK_API_KEY=secret). */
        const char *api_key = NULL;
        const char *api_key_secret = getenv("CWICK_API_KEY");
        double mem_pct = 0.0;
        uint64_t mem_limit = 0;
        /* The resident (non-expert) weights. int8 halves that budget, which
         * on a 128 GB box is the difference between a ~28% and a ~35% resident
         * expert cache -- the single biggest lever on decode speed when the
         * rest of the model has to come off the SSD. CWICK_SPINE works too. */
        const char *spine = getenv("CWICK_SPINE");
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--root") == 0) root = need_arg(&i, argc, argv, "--root");
            else if (strcmp(argv[i], "--host") == 0) host = need_arg(&i, argc, argv, "--host");
            else if (strcmp(argv[i], "--port") == 0) {
                int64_t v;
                if (!parse_i64(need_arg(&i, argc, argv, "--port"), 1, 65535, &v)) {
                    fprintf(stderr, "cwick serve: --port must be 1..65535\n"); return 2;
                }
                port = (int)v;
            }
            else if (strcmp(argv[i], "--max-tokens") == 0) {
                if (!parse_i64(need_arg(&i, argc, argv, "--max-tokens"),
                               1, 1000000, &max_tokens)) {
                    fprintf(stderr, "cwick serve: --max-tokens must be 1..1000000\n"); return 2;
                }
            }
            else if (strcmp(argv[i], "--context") == 0) {
                if (!parse_i64(need_arg(&i, argc, argv, "--context"),
                               1, 1048576, &max_context)) {
                    fprintf(stderr, "cwick serve: --context must be 1..1048576\n"); return 2;
                }
            }
            else if (strcmp(argv[i], "--warmup") == 0) {
                if (!parse_i64(need_arg(&i, argc, argv, "--warmup"),
                               1, 10000, &warmup)) {
                    fprintf(stderr, "cwick serve: --warmup must be 1..10000 "
                                    "(tokens to generate before listening, "
                                    "0/absent = off)\n");
                    return 2;
                }
            }
            else if (strcmp(argv[i], "--prefix-cache-dir") == 0)
                prefix_cache_dir = need_arg(&i, argc, argv, "--prefix-cache-dir");
            else if (strcmp(argv[i], "--prefix-cache-gib") == 0) {
                if (!parse_i64(need_arg(&i, argc, argv, "--prefix-cache-gib"),
                               1, 4096, &prefix_cache_gib)) {
                    fprintf(stderr, "cwick serve: --prefix-cache-gib must be 1..4096\n");
                    return 2;
                }
            }
            else if (strcmp(argv[i], "--verbose") == 0) verbose = true;
            else if (strcmp(argv[i], "--trace") == 0) trace = true;
            /* Built-in chat UI, compiled into the binary and served at `/`.
             * No asset directory, no static server, no CORS: the page calls
             * this same process over /v1/chat/completions. */
            else if (strcmp(argv[i], "--web") == 0) web = true;
            else if (strcmp(argv[i], "--no-web") == 0) web = false;
            /* Decision endpoint: POST /v1/systemone, answered from one prefill
             * of the resident model. Same weights, same process, no fine-tune;
             * the apps use it for every model decision. */
            else if (strcmp(argv[i], "--jev") == 0) jev = true;
            else if (strcmp(argv[i], "--no-jev") == 0) jev = false;
            /* LAN share mode: bind all interfaces and enable the web UI so a
             * phone can open the app after scanning its QR. Explicit opt-in;
             * there is no auth/TLS, so it is for a trusted same-Wi-Fi LAN. */
            else if (strcmp(argv[i], "--lan") == 0) lan = true;
            else if (strcmp(argv[i], "--password") == 0) password = need_arg(&i, argc, argv, "--password");
            else if (strcmp(argv[i], "--backend") == 0) {
                const char *v = need_arg(&i, argc, argv, "--backend");
                if (strcmp(v, "metal") == 0) backend = 1;
                else if (strcmp(v, "cpu") == 0) backend = 0;
                else { fprintf(stderr, "cwick serve: --backend must be 'metal' or 'cpu'\n"); return 2; }
            }
            /* A directory holding other local checkpoints. Nothing is loaded
             * by specifying it — each candidate is listed by GET /v1/models
             * and only made resident on POST /v1/models/switch, which is what
             * keeps `cwick serve --root DIR --models-dir ~/models` from paying
             * for two of anything. */
            else if (strcmp(argv[i], "--models-dir") == 0) {
                const char *d = need_arg(&i, argc, argv, "--models-dir");
                if (models_dirs_n == 8) {
                    fprintf(stderr, "cwick serve: at most 8 --models-dir "
                                    "entries are supported\n");
                    return 2;
                }
                models_dirs[models_dirs_n++] = d;
            }
            /* Where the built-in apps keep their own data (finished games,
             * the move-by-move replay log, small preferences). One sqlite
             * file, owned by this process, behind the same login. */
            else if (strcmp(argv[i], "--store") == 0)
                store_path = need_arg(&i, argc, argv, "--store");
            /* An API key for third-party agents. `--api-key NAME` names one
             * whose secret cwick generates and prints once; CWICK_API_KEY hands
             * over a secret you chose (for provisioning from a secret store).
             * Either way login is turned on: an unchecked key protects nothing. */
            else if (strcmp(argv[i], "--api-key") == 0)
                api_key = need_arg(&i, argc, argv, "--api-key");
            else if (strcmp(argv[i], "--mem-fraction") == 0) {
                double pct;
                if (!parse_double(need_arg(&i, argc, argv, "--mem-fraction"),
                                  50.0, 95.0, &pct)) {
                    fprintf(stderr, "cwick serve: --mem-fraction must be 50..95\n"); return 2;
                }
                mem_pct = pct / 100.0;
            }
            else if (strcmp(argv[i], "--max-memory") == 0) {
                double gib;
                if (!parse_double(need_arg(&i, argc, argv, "--max-memory"),
                                  0.001, 1024.0, &gib)) {
                    fprintf(stderr, "cwick serve: --max-memory must be 0.001..1024 GiB\n"); return 2;
                }
                mem_limit = (uint64_t)(gib * 1024.0 * 1024.0 * 1024.0);
            }
            else if (strcmp(argv[i], "--allow-swap") == 0) {
                allow_swap = true;
            }
            else if (strcmp(argv[i], "--spine") == 0) {
                spine = need_arg(&i, argc, argv, "--spine");
                if (strcmp(spine, "bf16") != 0 && strcmp(spine, "int8") != 0) {
                    fprintf(stderr, "cwick serve: --spine must be 'bf16' or 'int8'\n");
                    return 2;
                }
            }
            else {
                fprintf(stderr, "cwick serve: unknown option `%s`\n", argv[i]);
                return 2;
            }
        }
        if (!root) { fprintf(stderr, "cwick serve: --root DIR required\n"); return 2; }
        if (lan) {
            if (strcmp(host, "127.0.0.1") == 0) host = "0.0.0.0";
            web = true;
            setenv("CWICK_ALLOW_PUBLIC", "1", 1);
            /* Sharing on a LAN without a password would give every device on
             * the Wi-Fi the model, the logs and the prompts. Force login: an
             * explicitly empty password tells the server to generate one and
             * print it once at startup. */
            if (!password || !*password) password = "";
        }
        configure_logging(verbose, trace);
        if (lan)
            CWICK_INFO("init", "--lan: listening on all interfaces for a trusted "
                              "same-Wi-Fi LAN; single-user password login is on "
                              "(HTTP, no TLS: use a trusted Wi-Fi or a TLS proxy)");
        CWICK_INFO("init", "cwick serve starting (pid %d, root %s)", getpid(), root);
        select_backend(backend);
        if (trace) trace_privacy_notice();
        if (!instance_claim(root)) {
            cwick_logcap_stop();
            return 1;
        }
        cwick_memory_set_policy(mem_pct, mem_limit);
        cwick_memory_set_noswap(!allow_swap);
        report_vm_state();
        int ran = 0;
        int rc;
        rc = run_serve(root, host, port, max_tokens, max_context,
                       prefix_cache_dir, prefix_cache_gib,
                       spine, verbose, trace, web, jev, password,
                       models_dirs, models_dirs_n, store_path,
                       api_key, api_key_secret, warmup, &ran);
        close_wake_pipe();
        if (!ran) {
            /* Startup never got going: stop the tee before the last lines so
             * they land on the terminal exactly as they always have. */
            cwick_logcap_stop();
            /* Startup failed (or never got that far): no resources to release,
             * and no shutdown to report. */
            return rc;
        }
        if (g_stop) CWICK_INFO("init", "interrupted — shutting down");
        CWICK_INFO("init", "shutdown complete — Metal, threads and shard "
                          "descriptors released (exit %d)", rc);
        /* Only now: the two shutdown lines above belong in the ring too, and
         * restoring fd 1/fd 2 earlier would print them straight past it. */
        cwick_logcap_stop();
        return rc;
    }

    if (strcmp(cmd, "doctor") == 0) {
        const char *root = NULL;
        bool deep = false;
        int64_t expert_sample = 0;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--root") == 0) root = need_arg(&i, argc, argv, "--root");
            else if (strcmp(argv[i], "--deep") == 0) deep = true;
            else if (strcmp(argv[i], "--expert-sample") == 0)
                expert_sample = atoll(need_arg(&i, argc, argv, "--expert-sample"));
        }
        if (!root) { fprintf(stderr, "cwick doctor: --root DIR required\n"); return 2; }
        return run_doctor(root, true, deep, expert_sample);
    }

    fprintf(stderr, "cwick: unknown command `%s`\n", cmd);
    usage();
    return 2;
}
