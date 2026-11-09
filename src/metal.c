// metal.c — pure-C facade for the optional Metal decode backend.
//
// cwick normally runs the routed-expert FP8 GEMV as scalar dequant + GEMV on
// one CPU core (the single biggest arithmetic hotspot in a decode step).
// With CWICK_METAL=1 this unit hands that loop to the GPU via the ObjC
// bridge in metal_objc.m. Everything here is plain C; the ObjC machinery
// (struct-ABI dispatch, shader compile) is quarantined behind three C
// entry points.
//
// The backend is best-effort: if the Metal device, library or pipeline
// fails to come up, cwick_metal_gemv_fp8 returns 0 and the caller falls
// back to the CPU path. GPU offload must never break a generation.
#include "cwick.h"
#include "log.h"
#include "util.h"

#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

static int g_metal_ready = -1; /* -1 unknown, 0 unavailable, 1 ready */
static int g_metal_selected = -1; /* -1 env/auto, 0 explicit CPU, 1 explicit GPU */
/* Set by the autotune below; cwick_shmem_alloc needs it before the expert
 * cache is built, so both are declared up front. */
static int g_metal_wins = -1; /* -1 untimed, 0 CPU wins, 1 Metal wins */
static void metal_autotune(void);

int cwick_metal_gemv_fp8(const void *w_fp8, const float *scale_inv,
                        const float *x, float *out,
                        uint32_t k, uint32_t cols, uint32_t ncb,
                        uint32_t n_rows, uint32_t stride);
int cwick_metal_gemv_bf16(const void *w_bf16, const float *x, float *out,
                         uint32_t k, uint32_t cols);
void *cwick_metal_batch_begin_bridge(int slot);
int   cwick_metal_batch_gemv_bridge(void *batch, const void *w_fp8,
                                   const float *scale_inv, const float *x,
                                   float *out, uint32_t k, uint32_t cols,
                                   uint32_t ncb, uint32_t n_rows,
                                   uint32_t stride);
int   cwick_metal_batch_end_bridge(void *batch);
void  cwick_metal_batch_abort_bridge(void *batch);
void *cwick_metal_bf16_batch_begin_bridge(int slot);
int   cwick_metal_bf16_batch_add_bridge(void *batch, const void *w_bf16,
                                       const float *x, float *out,
                                       uint32_t k, uint32_t cols);
int   cwick_metal_bf16_batch_end_bridge(void *batch);
int cwick_metal_init(void);
void *cwick_metal_arena_alloc(size_t bytes);
int   cwick_metal_arena_free(void *p);
size_t cwick_metal_arena_bytes(void);
void cwick_metal_bridge_cleanup(void);
int  cwick_metal_keepalive_start(void);
void cwick_metal_keepalive_stop(void);

/* Force Metal backend on or off, overriding CWICK_METAL env var.
 * Returns 0 on success, -1 if Metal device init fails. */
int cwick_metal_force(int force) {
    g_metal_selected = force ? 1 : 0;
    if (force) {
        g_metal_ready = -1; /* reset so next probe will actually init */
        if (cwick_metal_init() != 0) {
            g_metal_ready = 1;
            CWICK_INFO("init", "Metal compute backend up (expert GEMV -> GPU)");
            return 0;
        } else {
            g_metal_ready = 0;
            CWICK_INFO("init", "Metal compute backend unavailable; "
                              "falling back to CPU GEMV");
            return -1;
        }
    } else {
        /* Explicit CPU selection wins even when CWICK_METAL=1 is present. */
        g_metal_ready = 0;
        return 0;
    }
}

/* Public entry: main.c calls this before the engine is built so the expert
 * cache already knows whether GPU-visible memory pays for itself. */
void cwick_metal_autotune(void) { metal_autotune(); }

static int env_enabled(const char *name) {
    const char *v = getenv(name);
    if (!v || !*v) return 0;
    return strcmp(v, "1") == 0 || strcasecmp(v, "true") == 0 ||
           strcasecmp(v, "yes") == 0 || strcasecmp(v, "on") == 0;
}

static int metal_probe(void) {
    if (g_metal_ready < 0) {
        /* Bring the device up even when the user asked for nothing: the
         * expert cache lives in GPU-visible memory, and the autotune needs a
         * live pipeline to time. Whether the GPU actually *runs* the GEMV is
         * decided separately, by measurement. */
        int enabled;
        if (g_metal_selected >= 0) enabled = g_metal_selected;   /* --backend */
        else if (getenv("CWICK_METAL")) enabled = env_enabled("CWICK_METAL");
#if defined(__APPLE__)
        else enabled = 1;   /* product target: device up, autotune picks */
#else
        else enabled = 0;
#endif
        if (!enabled) {
            g_metal_ready = 0;
        } else if (cwick_metal_init() != 0) {
            g_metal_ready = 1;
            CWICK_INFO("init", "Metal compute backend up (expert GEMV -> GPU)");
        } else {
            g_metal_ready = 0;
            CWICK_INFO("init", "Metal compute backend unavailable; "
                              "falling back to CPU GEMV");
        }
    }
    return g_metal_ready == 1;
}

/* Metrics / logging: is the GPU actually running the GEMV kernels? Reads the
 * same probe state as the compute path, initialising the device on first use
 * exactly like metal_probe() would. */
int cwick_metal_backend_active(void) { return metal_probe() ? 1 : 0; }

/* Metrics: is the GPU the thing actually running the routed-expert GEMV?
 *
 * That is a different question from "did the device come up". With no
 * --backend the autotune decides, and it may well pick the CPU pool; a
 * dashboard that reports "metal" for such a run claims a speedup that is not
 * happening. An explicit --backend wins, as it does everywhere else. Never
 * triggers the autotune from here (a scrape must not start a benchmark); an
 * untimed run reports its selection, and the first decode step settles it. */
int cwick_metal_gemv_on_gpu(void) {
    if (g_metal_selected == 0) return 0;
    if (!metal_probe()) return 0;
    if (g_metal_selected == 1) return 1;
    return g_metal_wins == 1;   /* -1 = not timed yet: not yet on the GPU */
}

/* ------------------------------------------------------------------ */
/* GPU-visible memory for streamed experts                             */
/* ------------------------------------------------------------------ */
/* Every expert matrix cwick pulls off the SSD is read into one of these
 * buffers. On Apple Silicon a shared MTLBuffer is ordinary RAM that the GPU
 * can also address, so preading into one costs nothing extra and saves the
 * 8 MiB memcpy the GEMV would otherwise need per call. When the backend is
 * off (or an arena cannot be grown) this degrades to plain malloc and the
 * GEMV falls back to its scratch-copy path -- same answer, slower. */
/* prof: split cwick_shmem_alloc so a slow path inside it is visible */
uint64_t g_shmem_probe_ns;    /* branch preamble (metal_probe etc.) */
uint64_t g_shmem_arena_ns;    /* the arena_alloc call itself        */
uint64_t g_shmem_fallback_ns; /* the calloc fallback, if it happens */
static uint64_t shm_now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* Whether a streamed expert blob would be served from the GPU-visible arena.
 *
 * The answer decides who allocates it: the arena has its own allocator tuned
 * for 8 MiB expert matrices, and a caller that batches them into larger blocks
 * would change how the arena grows (and how much of it is wired up front).
 * The predicate is deliberately the same one cwick_shmem_alloc uses, so the two
 * can never disagree. */
int cwick_shmem_gpu_visible(void) {
    if (g_metal_selected == 0) return 0;
    if (g_metal_selected < 0) {
        metal_autotune();
        return g_metal_wins ? 1 : 0;
    }
    return metal_probe() ? 1 : 0;
}

void *cwick_shmem_alloc(size_t bytes) {
    if (bytes == 0) return NULL;
    uint64_t _t0 = shm_now();
    /* GPU-visible only when the GPU is going to be the one reading.
     *
     * Measured on a 128 GB M3 Max: with the expert cache filling its full
     * 83.6 GiB budget, MTLBuffer-backed (wired, incompressible) pages pushed
     * the box into memory pressure and *slowed the whole decode* — the GEMV
     * phase alone went 323 -> 841 ms/step and attention 158 -> 239, far more
     * than the 206 ms/step of SSD traffic the bigger cache saved. Net: 0.594
     * -> 0.476 tok/s. When the CPU is doing the arithmetic there is nothing
     * to gain from wiring those pages, so keep them ordinary and let the OS
     * compress under pressure. */
    /* Wiring only pays when the GPU is the reader: --backend cpu, or an
     * autotune that measured the CPU as faster, both mean plain memory. */
    if (g_metal_selected == 0) return cwick_xmalloc(bytes);
    if (g_metal_selected < 0) {
        metal_autotune();
        if (!g_metal_wins) return cwick_xmalloc(bytes);
    } else if (!metal_probe()) {
        return cwick_xmalloc(bytes);
    }
    uint64_t _t1 = shm_now();
    g_shmem_probe_ns += _t1 - _t0;
    void *p = cwick_metal_arena_alloc(bytes);
    uint64_t _t2 = shm_now();
    g_shmem_arena_ns += _t2 - _t1;
    if (p) return p;
    g_shmem_fallback_ns += shm_now() - _t2;
    return cwick_xmalloc(bytes);
}

void cwick_shmem_free(void *p) {
    if (!p) return;
    if (g_metal_ready == 1 && cwick_metal_arena_free(p)) return;
    free(p);
}

/* Bytes of arena memory currently reserved for GPU-visible weights. */
size_t cwick_shmem_arena_bytes(void) {
    return (g_metal_ready == 1) ? cwick_metal_arena_bytes() : 0;
}

/* Which backend actually wins for the routed-expert shape.
 *
 * The GPU is not automatically the faster answer here, and guessing wrong
 * costs a factor of four. A decode step issues 1080 of these; each is one
 * [2048, 4096] FP8 matrix = 8 MiB, which is small enough that Metal's
 * per-dispatch cost (command buffer, commit, wait, result copy) is a
 * significant part of the total, while the CPU path now runs on every
 * performance core with NEON. Measured on an M3 Max: CPU ~280 us/matrix vs
 * Metal ~340 us/matrix in isolation, and worse again under load.
 *
 * So instead of hard-coding a winner, time both once at startup on a real
 * expert-shaped matrix and use whichever is quicker. --backend metal / cpu
 * still force the choice; this only decides the automatic default. */
static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void metal_autotune(void) {
    if (g_metal_wins >= 0) return;
    if (!metal_probe()) { g_metal_wins = 0; return; }
    /* One expert matrix exactly as the checkpoint stores it. */
    const int64_t k = 4096, cols = 2048, ncb = k / 128;
    const int reps = 8;
    size_t wbytes = (size_t)cols * k;
    size_t sbytes = (size_t)(cols / 128) * ncb * sizeof(float);
    uint8_t *w = (uint8_t *)calloc(1, wbytes ? wbytes : 1);
    float *sc = (float *)calloc(1, sbytes ? sbytes : 1);
    float *x = (float *)calloc((size_t)k, sizeof(float));
    float *out = (float *)calloc((size_t)cols, sizeof(float));
    if (!w || !sc || !x || !out) {                 /* cannot measure: keep GPU */
        free(w); free(sc); free(x); free(out);
        g_metal_wins = 1;
        return;
    }
    for (size_t i = 0; i < wbytes; i++) w[i] = (uint8_t)(i * 31u + 7u);
    for (size_t i = 0; i < (size_t)(cols / 128) * ncb; i++) sc[i] = 0.25f;
    for (int64_t i = 0; i < k; i++) x[i] = (float)((i % 11) - 5) * 0.0625f;

    /* one warm-up of each so neither side pays first-touch cost */
    cwick_metal_gemv_fp8(w, sc, x, out, (uint32_t)k, (uint32_t)cols,
                        (uint32_t)ncb, 1, (uint32_t)cols);
    cwick_gemv_fp8(out, k, cols, x, w, sc, ncb);

    double t0 = now_s();
    for (int i = 0; i < reps; i++)
        cwick_metal_gemv_fp8(w, sc, x, out, (uint32_t)k, (uint32_t)cols,
                            (uint32_t)ncb, 1, (uint32_t)cols);
    double tm = (now_s() - t0) / (double)reps;

    t0 = now_s();
    for (int i = 0; i < reps; i++) cwick_gemv_fp8(out, k, cols, x, w, sc, ncb);
    double tc = (now_s() - t0) / (double)reps;

    g_metal_wins = (tm < tc) ? 1 : 0;
    CWICK_INFO("init", "expert GEMV autotune: Metal %.0f us vs CPU %.0f us per "
                      "[%lld,%lld] FP8 matrix — using %s for the %d GEMVs of "
                      "each decode step",
              tm * 1e6, tc * 1e6, (long long)cols, (long long)k,
              g_metal_wins ? "Metal" : "the CPU pool", 1080);
    free(w); free(sc); free(x); free(out);
}

/* Run one FP8-E4M3 expert GEMV on the GPU when available; otherwise 0 so
 * moe.c keeps the CPU path. `stride` lets a batched prefill reuse one call
 * for n_rows rows. */
int cwick_metal_gemv(const void *w_fp8, const float *scale_inv,
                    const float *x, float *out,
                    int64_t k, int64_t cols, int64_t ncb,
                    int64_t n_rows, int64_t stride) {
    if (!metal_probe()) return 0;
    /* Automatic selection only: an explicit --backend always wins. */
    if (g_metal_selected < 0) {
        if (g_metal_wins < 0) metal_autotune();
        if (!g_metal_wins) return 0;
    }
    if (k <= 0 || cols <= 0 || ncb <= 0) return 0;
    if (k % 128 != 0) return 0;           /* kernel assumes CWICK_ST_BLOCK */
    if (cols > 0x7FFFFFFFu || k > 0x7FFFFFFFu) return 0;
    return cwick_metal_gemv_fp8(w_fp8, scale_inv, x, out,
                               (uint32_t)k, (uint32_t)cols, (uint32_t)ncb,
                               (uint32_t)n_rows, (uint32_t)stride);
}

/* ---- batched dispatch -----------------------------------------------
 * Public side of the two-wave batch described in metal_objc.m. The C layer
 * only owns the "is the GPU actually the backend" question and the int64 ->
 * uint32 narrowing; everything else lives behind the bridge. */
/* Which generation slot the calling thread is serving.
 *
 * Each connection runs on its own thread and is granted one slot, so the slot
 * is a property of the thread, not of the call site: the deep GEMV paths
 * (moe.c, attention.c) ask for a Metal batch without knowing where they are,
 * and this is how they get the right one. It is set once per request in the
 * server, before any generation begins. Threads that never set it (the CLI
 * `run` path, tests, the autotune probe) get slot 0, which is the same struct
 * the single-slot build always used. */
static _Thread_local int g_tls_slot;

void cwick_metal_set_slot(int slot) {
    g_tls_slot = slot < 0 ? 0 : slot;
}

void *cwick_metal_batch_begin_slot(int slot) {
    if (!metal_probe()) return NULL;
    if (g_metal_selected < 0) {
        if (g_metal_wins < 0) metal_autotune();
        if (!g_metal_wins) return NULL;
    }
    return cwick_metal_batch_begin_bridge(slot);
}

void *cwick_metal_batch_begin(void) { return cwick_metal_batch_begin_slot(g_tls_slot); }

int cwick_metal_batch_gemv(void *batch, const void *w_fp8,
                          const float *scale_inv, const float *x, float *out,
                          int64_t k, int64_t cols, int64_t ncb,
                          int64_t n_rows, int64_t stride) {
    if (!batch || k <= 0 || cols <= 0 || ncb <= 0 || n_rows <= 0) return 0;
    if (k % 128 != 0) return 0;
    if (cols > 0x7FFFFFFFl || k > 0x7FFFFFFFl) return 0;
    return cwick_metal_batch_gemv_bridge(batch, w_fp8, scale_inv, x, out,
                                        (uint32_t)k, (uint32_t)cols,
                                        (uint32_t)ncb, (uint32_t)n_rows,
                                        (uint32_t)stride);
}

int cwick_metal_batch_end(void *batch) {
    return batch ? cwick_metal_batch_end_bridge(batch) : 0;
}

void cwick_metal_batch_abort(void *batch) {
    if (batch) cwick_metal_batch_abort_bridge(batch);
}

/* Decode-only resident BF16 GEMV. Small matrices stay on the CPU so Metal
 * command setup cannot dominate; the large KDA/MoE/head projections do not
 * need a full fp32 dequantized copy on the CPU. */
int cwick_metal_linear_bf16(const void *w_bf16, const float *x, float *out,
                           int64_t k, int64_t cols) {
    /* Opt-in. In isolation this is 33-51x faster than the scalar CPU GEMV
     * (2682 -> 52-84 ms/step for the 45x4 KDA/DSA projections), but on the
     * real checkpoint it crashes during prefill and the cause is not pinned
     * down yet, so the default keeps the CPU path. Set CWICK_METAL_BF16=1 to
     * use it. */
    if (!getenv("CWICK_METAL_BF16")) return 0;
    if (!metal_probe() || !w_bf16 || !x || !out) return 0;
    if (k <= 0 || cols <= 0 || k > 0x7FFFFFFF || cols > 0x7FFFFFFF ||
        (k & 7) != 0)
        return 0;
    /* The kernel loads uint2 (8 bytes) off the row base, and every row base
     * is weights + c*k with k a multiple of 8 -- so an 8-byte-aligned weight
     * pointer is enough for every load to stay aligned. This checkpoint's
     * tensor offsets are 8-mod-16, which is exactly why the kernel does not
     * use uint4. */
    if (((uintptr_t)w_bf16 & 7u) != 0) return 0;
    const uint64_t elements = (uint64_t)k * (uint64_t)cols;
    if (elements < (1u << 18)) return 0; /* <256 Ki elements: CPU is faster */
    return cwick_metal_gemv_bf16(w_bf16, x, out, (uint32_t)k, (uint32_t)cols);
}

/* ---- batched BF16 GEMV ---------------------------------------------
 * Groups the independent resident-BF16 projections of one sublayer into a
 * single command buffer: the GPU beats the CPU pool ~2.4x on a [8192,4096]
 * matrix (311 us vs 745 us), but one commit+wait per call costs ~450 us of
 * scheduling, which is why the unbatched path is opt-in and loses. A batch
 * pays the round trip once. Same eligibility rules as
 * cwick_metal_linear_bf16; an item that cannot be served marks the batch bad
 * and the caller falls back to the serial path. */
/* Batching and the single-projection path are two different bets, so they get
 * two different defaults.
 *
 * A batch pays one commit+wait for the whole group, and measured on this
 * checkpoint that is a clear win: the KDA q/k/v group went 58.9 -> 34.4
 * ms/step and the MoE phase 158.4 -> 106.1. So it is on by default.
 *
 * The single projection is the opposite bet. Each call pays its own ~450 us
 * commit+wait, which is more than the kernel for everything except the very
 * largest matrices -- turning CWICK_METAL_BF16=1 on moved the KDA gate group
 * 6.1 -> 25.8 ms/step and the tail 24.6 -> 33.4, because those projections
 * are independent and cannot share a command buffer. Net for the whole step
 * was still positive (664 -> 600 ms) but ~28 ms of it was pure round-trip
 * overhead that a batch would not have paid. So the single path stays
 * opt-in (CWICK_METAL_BF16=1) and the batch is the default. */
static int g_bf16_batch = -1;
static bool bf16_batch_enabled(void) {
    if (g_bf16_batch < 0) {
        const char *e = getenv("CWICK_METAL_BF16_BATCH");
        g_bf16_batch = (e && (strcmp(e, "0") == 0 || strcmp(e, "off") == 0))
                           ? 0 : 1;
    }
    return g_bf16_batch == 1;
}

void *cwick_metal_bf16_batch_begin_slot(int slot) {
    if (!bf16_batch_enabled()) return NULL;
    if (!metal_probe()) return NULL;
    return cwick_metal_bf16_batch_begin_bridge(slot);
}

void *cwick_metal_bf16_batch_begin(void) { return cwick_metal_bf16_batch_begin_slot(g_tls_slot); }

int cwick_metal_bf16_batch_add(void *batch, const void *w_bf16,
                              const float *x, float *out,
                              int64_t k, int64_t cols) {
    if (!batch || k <= 0 || cols <= 0 || k > 0x7FFFFFFF || cols > 0x7FFFFFFF ||
        (k & 7) != 0 || ((uintptr_t)w_bf16 & 7u) != 0u)
        return 0;
    if ((uint64_t)k * (uint64_t)cols < (1u << 18)) return 0;  /* too small */
    return cwick_metal_bf16_batch_add_bridge(batch, w_bf16, x, out,
                                            (uint32_t)k, (uint32_t)cols);
}

/* Land the batch. Returns 1 when every item ran on the GPU, 0 when the
 * caller must redo the group itself (nothing was written). */
int cwick_metal_bf16_batch_end(void *batch) {
    return batch ? cwick_metal_bf16_batch_end_bridge(batch) : 0;
}

void cwick_metal_cleanup(void) {
    cwick_metal_bridge_cleanup();
    g_metal_ready = 0;
    g_metal_selected = -1;
}
