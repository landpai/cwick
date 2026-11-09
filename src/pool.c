/* pool.c — persistent fork-join worker pool for the CPU math kernels.
 *
 * Why this exists: every decode step runs ~1400 GEMV/GEMM calls (45 layers x
 * ~6 dense projections, plus 45 x 8 x 3 routed-expert matrices). Before this
 * file all of them executed on the calling thread, so a 12-performance-core
 * M3 Max served a 320B-parameter model with 1 core doing arithmetic and 11
 * idle. The SSD streamer already had its own thread; compute had none.
 *
 * Design notes (all deliberate):
 *   - Persistent threads. Spawning a thread costs ~20-50 us; at 1400
 *     dispatches per step that would be 30-70 ms/step of pure overhead, i.e.
 *     a meaningful fraction of the 1 s we are trying to reach. Workers are
 *     created once and parked on a condition variable.
 *   - Generation counter, not a job queue. A dispatch is: bump generation,
 *     workers run their slice, last one signals the caller. No malloc, no
 *     queue nodes, nothing to free.
 *   - No mutex on the dispatch path. This is the difference between ~25 us
 *     and ~108 us per dispatch, which at ~1500 dispatches per decode step is
 *     ~130 ms/step. The first shape made all twelve workers take the pool
 *     mutex twice per dispatch -- once to return from cond_wait, once to
 *     report completion -- and those acquisitions are serialized, so the
 *     broadcast alone cost more than some of the kernels it was starting.
 *     Now the generation and the completion count are atomics, workers spin
 *     before they park, and only the last worker out takes the lock, to wake
 *     a caller that has already gone to sleep.
 *   - Contiguous slices over [0, n). Each worker owns out[lo, hi) of the
 *     output, so kernels need no atomics and no reduction buffer.
 *   - Small work runs inline. Dispatch latency (~5-15 us) is not free; below
 *     CWICK_POOL_MIN_WORK the caller just does the whole range itself.
 *   - Lazily initialized and safe to call before/after shutdown: every entry
 *     point degrades to serial execution when the pool is not up, so the
 *     existing single-threaded code paths and tests keep their exact results.
 */

#include "cwick.h"
#include "log.h"
#include "util.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <pthread/qos.h>
#endif

#if defined(__aarch64__)
#define CWICK_CPU_RELAX() __asm__ __volatile__("yield" ::: "memory")
#else
#define CWICK_CPU_RELAX() __asm__ __volatile__("" ::: "memory")
#endif

/* How long a worker spins on the generation counter before parking. It only
 * has to cover the gap between two consecutive dispatches, which inside a
 * layer is a few microseconds (the caller does one activation or one norm
 * and dispatches again); the only long gap is the SSD fetch between layers
 * (~17 ms), and a worker that spins through that wastes the budget on one
 * core, 42 times per step.
 *
 * The spin MUST re-read the generation as it goes. Spinning for a fixed
 * budget and only then looking is worse than not spinning at all: a worker
 * caught mid-spin cannot see a job that arrived after it started, so every
 * dispatch pays up to the whole budget in added latency. Measured: 300 us of
 * blind spin turned a 1478 ms/step run into 4146 ms/step. */
#define CWICK_POOL_SPIN_NS 50000

/* How long the caller spins before parking on cv_done.
 *
 * Parking is not free and it is not the cheap option here. The caller has
 * nothing else to do, and the round trip through cv_done is a futex wake that
 * cannot begin until the last worker has taken the pool mutex — the same
 * mutex all twelve workers are using to leave their own cond_wait — so it
 * lands squarely on the critical path, once per dispatch, ~1200 times per
 * decode step. Spinning instead burns one core that would otherwise be idle
 * and gets the answer the instant it is ready. */
#define CWICK_POOL_CALLER_SPIN_MAX_NS 300000
#define CWICK_POOL_CALLER_SPIN_MIN_NS 2000

/* Exponential moving average of how long the last few dispatches took; see
 * the adaptive spin in pool_dispatch(). Guarded by the fact that only the
 * dispatching thread writes it. */
static uint64_t g_spin_budget = 20000;

/* Dispatch accounting. The expert GEMV is ~1200 dispatches per decode step,
 * so a few tens of microseconds of fixed cost each is a visible slice of the
 * step; CWICK_POOL_STATS=1 prints what a dispatch actually costs at exit. */
static uint64_t g_pool_dispatches, g_pool_dispatch_ns, g_pool_waited_ns;
static int g_pool_stats = -1;

static void pool_stats_dump(void) {
    if (g_pool_stats != 1 || !g_pool_dispatches) return;
    fprintf(stderr,
            "pool: %llu dispatch(es), %.1f us/dispatch, %.1f us of that "
            "parked, %d thread(s)\n",
            (unsigned long long)g_pool_dispatches,
            (double)g_pool_dispatch_ns / (double)g_pool_dispatches / 1e3,
            (double)g_pool_waited_ns / (double)g_pool_dispatches / 1e3,
            cwick_pool_threads());
}

/* Spin until `gen` changes, or until the budget runs out. Returns the
 * generation last seen, so the caller can tell which of the two happened. */
static uint64_t pool_spin_for_gen(const _Atomic uint64_t *gen, uint64_t seen,
                                  uint64_t budget_ns) {
    uint64_t t0 = cwick_prof_ns();
    uint64_t g;
    for (;;) {
        for (int i = 0; i < 64; i++) CWICK_CPU_RELAX();
        g = atomic_load_explicit(gen, memory_order_acquire);
        if (g != seen) return g;
        if (cwick_prof_ns() - t0 >= budget_ns) return g;
    }
}

/* Below this many output rows (or work items) a parallel dispatch costs more
 * than it saves. Tuned against the real checkpoint: a [2048, 4096] expert
 * matrix is well above it, a [4096, 1536] small projection is not. */
#define CWICK_POOL_MIN_WORK 512

typedef struct {
    pthread_mutex_t mtx;
    pthread_cond_t cv_go;
    /* The caller's wake-up gets its own mutex, deliberately separate from the
     * one guarding cv_go. Sharing them was the single largest fixed cost in a
     * dispatch: the moment a job ends, all twelve workers race back to p->mtx
     * to park again, and the caller cannot return from its own cond_wait
     * until it wins that race. Measured with CWICK_POOL_STATS=1: 548 us of
     * every 705 us dispatch was spent parked behind them. */
    pthread_mutex_t mtx_done;
    pthread_cond_t cv_done;
    pthread_t *th;
    int nthreads;
    int nperf;   /* how many of them are performance cores */
    int *share;  /* cumulative work share per worker; see pool_weights() */
    /* job description, guarded by mtx */
    cwick_pool_range_fn fn;
    void *arg;
    int64_t n;
    int nt; /* workers taking part in the current dispatch */
    /* Dispatch state, deliberately outside the mutex: the caller writes fn,
     * arg, n and nt and then bumps gen with a release store, and a worker
     * that observes the new gen with an acquire load is guaranteed to see
     * them. The caller cannot overwrite them until every worker has reported
     * completion, so a slow worker can never read the next job's fields. */
    _Atomic uint64_t gen;
    _Atomic int done;
    _Atomic int sleeping; /* workers currently parked in pthread_cond_wait */
    _Atomic int stop;
} cwick_pool;

static cwick_pool g_pool;
static int g_pool_up = 0;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;

/* Efficiency cores are worth a fraction of a performance core, and a fork-join
 * finishes when its slowest participant does — so giving an E-core an equal
 * slice makes the dispatch slower than leaving it idle. 33 % is measured, not
 * guessed: the expert GEMV is DRAM-latency-bound (see the bandwidth notes in
 * the README), and what an extra core buys in that regime is memory-level
 * parallelism, which an E-core supplies nearly as well as a P-core. */
#define CWICK_ECORE_SHARE_PCT 33

/* Cumulative work share, in 1/1024 units of the whole job, for workers
 * 0..nthreads-1. Worker i owns [share[i], share[i+1]) of the output, and the
 * denominator for a dispatch of `nt` workers is share[nt]. */
static void pool_weights(int nperf, int nthreads, int *share) {
    share[0] = 0;
    for (int i = 0; i < nthreads; i++)
        share[i + 1] = share[i] + (i < nperf ? 1024 : 1024 * CWICK_ECORE_SHARE_PCT / 100);
}

static int perf_core_count(int *nperf_out) {
    int n = 0, nperf = 0;
#if defined(__APPLE__)
    size_t sz = sizeof(n);
    if (sysctlbyname("hw.logicalcpu", &n, &sz, NULL, 0) != 0 || n <= 0) n = 0;
    sz = sizeof(nperf);
    if (sysctlbyname("hw.perflevel0.logicalcpu", &nperf, &sz, NULL, 0) != 0 || nperf <= 0)
        nperf = n;   /* homogeneous (Intel, or an unknown arm64 board) */
#endif
    if (n <= 0) n = 4;
    if (nperf <= 0) nperf = 4;
    if (nperf > n) nperf = n;
    if (n < 1) n = 1;
    if (nperf > n) nperf = n;
    if (nperf < 1) nperf = 1;
    /* Efficiency cores stay idle by default. Weighting their slices was tried
     * and lost: measured on the real checkpoint, 16 weighted workers gave
     * 1450 ms/step against 1369 for 12 performance cores. The four extra
     * workers add more wake-up latency to a fork-join that already runs
     * ~2000 times per decode step than they add memory-level parallelism to
     * kernels that are 200 us long. CWICK_THREADS overrides. */
    n = nperf;
    if (nperf_out) *nperf_out = nperf;
    return n;
}

/* Set while a thread is inside a pool job. The fork-join state (fn/arg/n/gen)
 * is a single global, so a worker that dispatched again would overwrite the
 * outer job's state and every other worker would start running the inner one.
 * `pool_dispatch` checks this and runs a nested job inline instead: slower,
 * but correct — and it means calling a pool-using kernel from inside a pool
 * job is safe rather than a hang. */
static __thread int t_in_pool_job = 0;

static void *pool_worker(void *argp) {
    cwick_pool *p = (cwick_pool *)argp;
    t_in_pool_job = 1;
    int tid = 0;
    for (int i = 0; i < p->nthreads; i++)
        if (pthread_equal(p->th[i], pthread_self())) { tid = i; break; }
    uint64_t seen = 0;

    for (;;) {
        uint64_t g = atomic_load_explicit(&p->gen, memory_order_acquire);
        if (g == seen && !atomic_load_explicit(&p->stop, memory_order_relaxed)) {
            g = pool_spin_for_gen(&p->gen, seen, CWICK_POOL_SPIN_NS);
            if (g == seen) {
                /* Park. `sleeping` is bumped with the mutex held so the
                 * caller can never miss a worker that is on its way to
                 * sleep, and the predicate is re-tested under the same mutex
                 * so a generation bump that lands in between is not lost. */
                pthread_mutex_lock(&p->mtx);
                atomic_fetch_add_explicit(&p->sleeping, 1, memory_order_relaxed);
                while (atomic_load_explicit(&p->gen, memory_order_acquire) == seen &&
                       !atomic_load_explicit(&p->stop, memory_order_relaxed))
                    pthread_cond_wait(&p->cv_go, &p->mtx);
                atomic_fetch_sub_explicit(&p->sleeping, 1, memory_order_relaxed);
                if (atomic_load_explicit(&p->stop, memory_order_relaxed)) {
                    pthread_mutex_unlock(&p->mtx);
                    return NULL;
                }
                pthread_mutex_unlock(&p->mtx);
                continue;
            }
        }
        if (atomic_load_explicit(&p->stop, memory_order_relaxed)) return NULL;
        seen = g;

        cwick_pool_range_fn fn = p->fn;
        void *a = p->arg;
        int64_t n = p->n;
        int nt = p->nt;

        /* Contiguous slice: worker i owns [share[i], share[i+1]) of [0, n),
         * weighted by what that worker's core can actually sustain. Workers
         * with tid >= nt are idle spectators for a narrow job; they skip the
         * work but still report completion so the caller's join test holds. */
        if (tid < nt && p->share) {
            int64_t denom = p->share[nt];
            int64_t lo = denom > 0 ? n * (int64_t)p->share[tid] / denom : 0;
            int64_t hi = denom > 0 ? n * (int64_t)p->share[tid + 1] / denom : n;
            if (hi > lo) fn(a, lo, hi);
        }

        /* One lock per dispatch instead of twelve: only the last worker out
         * has to wake anybody. */
        if (atomic_fetch_add_explicit(&p->done, 1, memory_order_release) + 1 ==
            p->nthreads) {
            pthread_mutex_lock(&p->mtx_done);
            pthread_cond_signal(&p->cv_done);
            pthread_mutex_unlock(&p->mtx_done);
        }
    }
    return NULL;
}

static void pool_init_once(void) {
    memset(&g_pool, 0, sizeof(g_pool));
    pthread_mutex_init(&g_pool.mtx, NULL);
    pthread_cond_init(&g_pool.cv_go, NULL);
    pthread_mutex_init(&g_pool.mtx_done, NULL);
    pthread_cond_init(&g_pool.cv_done, NULL);

    int nperf = 0;
    int want = perf_core_count(&nperf);
    const char *e = getenv("CWICK_THREADS");
    if (e && *e) {
        int v = atoi(e);
        if (v >= 1 && v <= 64) want = v;
    }
    if (want < 1) want = 1;

    g_pool.th = (pthread_t *)calloc((size_t)want, sizeof(pthread_t));
    if (!g_pool.th) return;
    g_pool.share = (int *)calloc((size_t)want + 1, sizeof(int));
    if (!g_pool.share) { free(g_pool.th); g_pool.th = NULL; return; }
    g_pool.nthreads = want;
    g_pool.nperf = want < nperf ? want : nperf;
    pool_weights(g_pool.nperf, want, g_pool.share);

    /* Quality of service is not cosmetic on Apple Silicon. A thread created
     * at the default QoS is eligible for the efficiency cores, which run
     * fp32 FMA at roughly a quarter of performance-core throughput — and a
     * fork-join finishes when its slowest participant does, so a couple of
     * E-core placements tax the whole dispatch. Measured on an M3 Max: the
     * same expert GEMV ran 250 us/call in a foreground benchmark process and
     * 538 us/call inside the model run, for no other reason than where the
     * scheduler put the workers. Ask for user-initiated QoS on every worker
     * and let the main thread inherit it. */
#if defined(__APPLE__)
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INITIATED, 0);
#endif

    int started = 0;
    for (int i = 0; i < want; i++) {
#if defined(__APPLE__)
        if (pthread_create(&g_pool.th[i], &attr, pool_worker, &g_pool) != 0) break;
#else
        if (pthread_create(&g_pool.th[i], NULL, pool_worker, &g_pool) != 0) break;
#endif
        started++;
    }
#if defined(__APPLE__)
    pthread_attr_destroy(&attr);
#endif
    CWICK_VERB("init", "compute pool: %d worker thread(s), %d weighted as "
              "performance cores (CWICK_THREADS to override)",
              started, g_pool.nperf);
    if (started <= 0) {
        free(g_pool.th);
        free(g_pool.share);
        g_pool.th = NULL;
        g_pool.share = NULL;
        g_pool.nthreads = 0;
        return;
    }
    /* If fewer threads came up than requested, still usable — slices are
     * derived from nthreads, so the work is simply split more coarsely. */
    g_pool.nthreads = started;
    g_pool_up = 1;
}

int cwick_pool_init(void) {
    pthread_once(&g_once, pool_init_once);
#if defined(__APPLE__)
    /* The caller dispatches into the pool and then waits, so a caller parked
     * at background QoS drags the workers down with it on the next dispatch.
     * Raise the thread that drives the model too. */
    if (g_pool_up) pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#endif
    return g_pool_up ? g_pool.nthreads : 0;
}

int cwick_pool_threads(void) {
    if (!g_pool_up) return 0;
    return g_pool.nthreads;
}

void cwick_pool_shutdown(void) {
    if (!g_pool_up) return;
    atomic_store_explicit(&g_pool.stop, 1, memory_order_release);
    pthread_mutex_lock(&g_pool.mtx);
    pthread_cond_broadcast(&g_pool.cv_go);
    pthread_mutex_unlock(&g_pool.mtx);
    for (int i = 0; i < g_pool.nthreads; i++) pthread_join(g_pool.th[i], NULL);
    pthread_mutex_destroy(&g_pool.mtx_done);
    pthread_cond_destroy(&g_pool.cv_done);
    free(g_pool.th);
    free(g_pool.share);
    g_pool.th = NULL;
    g_pool.share = NULL;
    g_pool.nthreads = 0;
    g_pool_up = 0;
}

/* Fork-join: run fn(arg, lo, hi) over contiguous slices of [0, n).
 * Returns the number of threads that actually participated (0 = ran inline). */
static int pool_dispatch(int64_t n, void *arg, cwick_pool_range_fn fn,
                         int64_t min_work) {
    if (n <= 0 || !fn) return 0;
    if (t_in_pool_job) {
        /* Nested dispatch: this thread is already running a slice of another
         * job, and there is only one set of dispatch globals. Run it inline.
         * Any caller that relies on pool-level parallelism from inside a job
         * gets serial execution, not corrupted state. */
        fn(arg, 0, n);
        return 0;
    }
    if (!g_pool_up) {
        /* Not initialized (or already shut down): stay serial, stay correct. */
        fn(arg, 0, n);
        return 0;
    }
    if (g_pool_stats < 0) {
        g_pool_stats = cwick_env_true("CWICK_POOL_STATS") ? 1 : 0;
        if (g_pool_stats) atexit(pool_stats_dump);
    }
    uint64_t td0 = cwick_prof_ns();
    int nt = g_pool.nthreads;
    if (nt <= 1 || n < min_work) {
        fn(arg, 0, n);
        if (g_pool_stats) {
            g_pool_dispatches++;
            g_pool_dispatch_ns += cwick_prof_ns() - td0;
        }
        return 0;
    }

    /* Skip trailing workers that would get an empty slice: waking a thread to
     * do nothing costs the same broadcast as waking one that works. */
    while (nt > 1 && (n / (int64_t)nt) < 1) nt--;
    if (nt <= 1) {
        fn(arg, 0, n);
        if (g_pool_stats) {
            g_pool_dispatches++;
            g_pool_dispatch_ns += cwick_prof_ns() - td0;
        }
        return 0;
    }

    /* The dispatching thread takes the last slice itself.
     *
     * It has to do something while the workers run, and the two options it
     * had were both bad: parking costs a futex wake on the critical path
     * (measured at 1009 us of a 1194 us dispatch), and spinning burns a core
     * that a worker could be using. Running a slice is strictly better --
     * the work gets done by a thread that was otherwise idle, and by the
     * time it finishes, the workers are usually nearly done too, so the wait
     * at the end is short.
     *
     * The workers are told to cover [0, caller_lo) with nt-1 of them; the
     * caller owns [caller_lo, n). Same weighted split, one extra
     * participant. */
    int64_t caller_lo = g_pool.share
                        ? n * (int64_t)g_pool.share[nt - 1] / (int64_t)g_pool.share[nt]
                        : n * (int64_t)(nt - 1) / (int64_t)nt;
    if (caller_lo >= n) caller_lo = n - 1;   /* never hand the caller nothing */
    if (caller_lo < 0) caller_lo = 0;

    g_pool.fn = fn;
    g_pool.arg = arg;
    g_pool.n = caller_lo;   /* workers cover [0, caller_lo) */
    g_pool.nt = nt - 1;
    atomic_store_explicit(&g_pool.done, 0, memory_order_relaxed);
    uint64_t gen = atomic_load_explicit(&g_pool.gen, memory_order_relaxed);
    atomic_store_explicit(&g_pool.gen, gen + 1, memory_order_release);

    /* Only wake the workers that actually parked. A worker that is still
     * spinning sees the new generation on its own; the rest need a broadcast,
     * and skipping it when nobody is asleep is the whole point. */
    pthread_mutex_lock(&g_pool.mtx);
    if (atomic_load_explicit(&g_pool.sleeping, memory_order_relaxed) > 0)
        pthread_cond_broadcast(&g_pool.cv_go);
    pthread_mutex_unlock(&g_pool.mtx);

    fn(arg, caller_lo, n);

    /* Adaptive spin: wait out a job that lasts about as long as the previous
     * one did, then park. A fixed budget cannot work across this range — a
     * decode step dispatches ~2000 times, from 5 us norms to 300 us expert
     * GEMVs, and spinning 200 us on a 5 us job costs forty times the job.
     * Dispatches of the same shape arrive in long runs (45 layers of the same
     * few kernels), so the previous duration predicts the next one well. */
    uint64_t budget = g_spin_budget;
    uint64_t t0 = cwick_prof_ns();
    while (atomic_load_explicit(&g_pool.done, memory_order_acquire) < g_pool.nthreads) {
        for (int i = 0; i < 64; i++) CWICK_CPU_RELAX();
        if (cwick_prof_ns() - t0 >= budget) break;
    }
    uint64_t tp0 = cwick_prof_ns();
    if (atomic_load_explicit(&g_pool.done, memory_order_acquire) < g_pool.nthreads) {
        pthread_mutex_lock(&g_pool.mtx_done);
        while (atomic_load_explicit(&g_pool.done, memory_order_acquire) < g_pool.nthreads)
            pthread_cond_wait(&g_pool.cv_done, &g_pool.mtx_done);
        pthread_mutex_unlock(&g_pool.mtx_done);
    }
    uint64_t td1 = cwick_prof_ns();
    if (g_pool_stats) {
        g_pool_dispatches++;
        g_pool_dispatch_ns += td1 - td0;
        g_pool_waited_ns += td1 - tp0;
    }
    /* Track the real duration so the next dispatch spins about this long.
     * The EMA is over 2 so it follows a change of kernel within a layer or
     * two without overreacting to one slow dispatch. */
    {
        uint64_t took = td1 - td0;
        uint64_t next = (g_spin_budget * 2 + took) / 3;
        if (next < CWICK_POOL_CALLER_SPIN_MIN_NS) next = CWICK_POOL_CALLER_SPIN_MIN_NS;
        if (next > CWICK_POOL_CALLER_SPIN_MAX_NS) next = CWICK_POOL_CALLER_SPIN_MAX_NS;
        g_spin_budget = next;
    }
    g_pool.fn = NULL;
    g_pool.arg = NULL;
    return nt;
}

int cwick_pool_for(int64_t n, void *arg, cwick_pool_range_fn fn) {
    return pool_dispatch(n, arg, fn, CWICK_POOL_MIN_WORK);
}

/* Dispatch for a *small number of large jobs*.
 *
 * CWICK_POOL_MIN_WORK exists because splitting 2048 GEMV output rows twelve
 * ways is not worth a broadcast, but that heuristic is about per-item cost,
 * not item count. One layer's routed set is only 24 items — and each is an
 * 8 MiB SSD read, the single most expensive thing a decode step does. Those
 * must go out in parallel: measured on the real checkpoint, one 8 MiB read
 * at a time sustains 6.66 GB/s while a dozen in flight sustain 9.16 GB/s. */
int cwick_pool_for_items(int64_t n, void *arg, cwick_pool_range_fn fn) {
    return pool_dispatch(n, arg, fn, 1);
}
