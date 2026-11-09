/* mem.c — see mem.h: decide, from this machine, how much of a 306 GiB
 * checkpoint may become resident.
 *
 * The plan is: target = recommendedMaxWorkingSetSize * pct (default 80%),
 * minus the non-routed (resident) weights, with a reserve so prefill can
 * overlap. 80% was measured safe on unified-memory Apple Silicon, where the
 * same RAM has to feed the OS, the page cache and transient staging buffers.
 *
 * cwick also subtracts the inference runtime state (KV/KDA/mHC) up front,
 * because GLM-5.3-Flash dedicates the better part of the remaining RAM to
 * pinning hot experts and every GiB booked twice ends up as swap.
 */
#include "mem.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "log.h"
#include "metrics.h"

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_host.h>   /* host_statistics64, vm_statistics64 */
#include <mach/task_info.h>   /* task_vm_info_data_t / TASK_VM_INFO */
#include <sys/sysctl.h>
#endif

#include <sys/mman.h>
#include <unistd.h>

#define CWICK_GIB (1024.0 * 1024.0 * 1024.0)

/* ------------------------------------------------------------------ */
/* Policy                                                              */
/* ------------------------------------------------------------------ */
static double g_host_fraction = CWICK_MEM_DEFAULT_FRACTION;
static uint64_t g_limit_bytes = 0;
static uint64_t g_runtime_state = 0;
static bool     g_noswap = true;

/* Environment fallbacks so the decision is inspectable without rebuilding
 * anything: CWICK_MEM_PCT=70 ./cwick serve … */
static double env_fraction(void) {
    const char *e = getenv("CWICK_MEM_PCT");
    if (!e || !*e) return CWICK_MEM_DEFAULT_FRACTION;
    char *end = NULL;
    double v = strtod(e, &end);
    if (end == e || *end != '\0') return CWICK_MEM_DEFAULT_FRACTION;
    if (v < CWICK_MEM_MIN_FRACTION) v = CWICK_MEM_MIN_FRACTION;
    if (v > CWICK_MEM_MAX_FRACTION) v = CWICK_MEM_MAX_FRACTION;
    return v / 100.0;
}

static uint64_t env_limit_bytes(void) {
    const char *e = getenv("CWICK_MEM_MAX_GIB");
    if (!e || !*e) return 0;
    char *end = NULL;
    double gib = strtod(e, &end);
    if (end == e || (*end != '\0' && strcasecmp(end, "gib") != 0 &&
                     strcasecmp(end, "g") != 0 && strcasecmp(end, "gb") != 0)) {
        return 0;
    }
    if (gib <= 0.0) return 0;
    double bytes = gib * CWICK_GIB;
    if (bytes > 1.6e19) return 0; /* absurd / overflow guard */
    return (uint64_t)bytes;
}

/* Set from the CLI (--mem-fraction). Until it is set, the environment wins,
 * so `CWICK_MEM_PCT=70 ./cwick serve …` stays a usable knob. */
static bool g_policy_set = false;

void cwick_memory_set_policy(double host_fraction, uint64_t limit_bytes) {
    if (host_fraction > 0.0) {
        g_host_fraction = host_fraction;
        g_policy_set = true;
    }
    if (limit_bytes > 0) g_limit_bytes = limit_bytes;
}

void cwick_memory_policy(double *host_fraction, uint64_t *limit_bytes) {
    double f = g_policy_set ? g_host_fraction : env_fraction();
    /* Both constants are already fractions (0.50 .. 0.95), not percentages. */
    if (f < CWICK_MEM_MIN_FRACTION) f = CWICK_MEM_MIN_FRACTION;
    if (f > CWICK_MEM_MAX_FRACTION) f = CWICK_MEM_MAX_FRACTION;
    if (host_fraction) *host_fraction = f;
    if (limit_bytes) {
        uint64_t lim = g_limit_bytes ? g_limit_bytes : env_limit_bytes();
        *limit_bytes = lim;
    }
}

void cwick_memory_set_runtime_state(uint64_t bytes) { g_runtime_state = bytes; }
uint64_t cwick_memory_runtime_state(void) { return g_runtime_state; }

void cwick_memory_set_noswap(bool noswap) { g_noswap = noswap; }
bool cwick_memory_noswap(void) { return g_noswap; }

/* ------------------------------------------------------------------ */
/* Host / process memory                                               */
/* ------------------------------------------------------------------ */
uint64_t cwick_host_memory_bytes(void) {
#if defined(__APPLE__)
    uint64_t mem = 0;
    size_t len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, NULL, 0) == 0 && mem > 0) return mem;
#endif
#if defined(_SC_PHYS_PAGES)
    {
        long pages = sysconf(_SC_PHYS_PAGES);
        long ps = sysconf(_SC_PAGESIZE);
        if (pages > 0 && ps > 0) return (uint64_t)pages * (uint64_t)ps;
    }
#endif
    return 0;
}

uint64_t cwick_process_rss_bytes(void) {
#if defined(__APPLE__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS) {
        return info.resident_size;
    }
#endif
    FILE *f = fopen("/proc/self/statm", "r");
    if (f) {
        unsigned long size = 0, resident = 0;
        if (fscanf(f, "%lu %lu", &size, &resident) == 2) {
            long ps = sysconf(_SC_PAGESIZE);
            fclose(f);
            if (ps > 0) return (uint64_t)resident * (uint64_t)ps;
        }
        fclose(f);
    }
    return 0;
}

/* The number the kernel actually accounts for this process: like
 * resident_size but *including* pages macOS has moved into the compressor.
 * resident_size is the wrong metric for a cache we promised to keep hot -- a
 * pinned 84 GiB can report an RSS of 40 GiB purely because the OS compressed
 * it, which is exactly the failure the memory governor exists to prevent. */
uint64_t cwick_process_footprint_bytes(void) {
#if defined(__APPLE__)
    task_vm_info_data_t vm;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vm, &count)
            == KERN_SUCCESS && vm.phys_footprint > 0) {
        return (uint64_t)vm.phys_footprint;
    }
#endif
    return cwick_process_rss_bytes();
}

double cwick_gib(uint64_t bytes) { return (double)bytes / CWICK_GIB; }

/* ------------------------------------------------------------------ */
/* Swap, the compressor, and wiring                                    */
/* ------------------------------------------------------------------ */
bool cwick_vm_state_sample(cwick_vm_state *out) {
    if (!out) return false;
    memset(out, 0, sizeof(*out));
#if defined(__APPLE__)
    bool any = false;
    /* swapusage: total/used/free, in bytes, plus an "encrypted" flag we do
     * not need. */
    {
        int mib[2] = { CTL_VM, VM_SWAPUSAGE };
        struct xsw_usage xu;
        size_t len = sizeof(xu);
        if (sysctl(mib, 2, &xu, &len, NULL, 0) == 0 && len == sizeof(xu)) {
            out->swap_used = xu.xsu_used;
            out->swap_total = xu.xsu_total;
            any = true;
        }
    }
    {
        vm_statistics64_data_t vs;
        mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
        if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                              (host_info64_t)&vs, &cnt) == KERN_SUCCESS) {
            out->compressions = (uint64_t)vs.compressions;
            out->decompressions = (uint64_t)vs.decompressions;
            out->swapins = (uint64_t)vs.swapins;
            out->swapouts = (uint64_t)vs.swapouts;
            out->wired = (uint64_t)vs.wire_count * 4096ull;
            any = true;
        }
    }
    return any;
#else
    return false;
#endif
}

static uint64_t g_locked_bytes;
static int      g_lock_warned;

bool cwick_mem_lock_range(const void *p, size_t n, const char *what) {
    if (!p || n == 0) return false;
    if (mlock(p, n) != 0) {
        if (!g_lock_warned) {
            g_lock_warned = 1;
            CWICK_INFO("mem", "could not wire %.2f GiB of %s (%s) — the VM may "
                             "compress or swap it; see --allow-swap",
                      cwick_gib((uint64_t)n), what ? what : "memory",
                      strerror(errno));
        }
        return false;
    }
    g_locked_bytes += (uint64_t)n;
    return true;
}

uint64_t cwick_mem_locked_bytes(void) { return g_locked_bytes; }

/* Undo cwick_mem_lock_range for a range that is going away.
 *
 * Needed on unload, not on exit: a mapping that a model wired stays wired
 * after the model's structures are freed, so the pages cannot be reclaimed and
 * the next model inherits them. munlock over a superset is harmless -- it
 * clears the lock on every page in the range -- and at teardown there is
 * nothing left to keep resident. */
void cwick_mem_unlock_range(const void *p, size_t n) {
    if (!p || n == 0) return;
    if (munlock(p, n) == 0 && g_locked_bytes >= (uint64_t)n)
        g_locked_bytes -= (uint64_t)n;
}

/* ------------------------------------------------------------------ */
/* GPU wired-memory ceiling                                           */
/* ------------------------------------------------------------------ */
/* macOS lets the GPU driver wire roughly three quarters of unified memory
 * by default (iogpu.wired_limit_mb == 0 means "system default"). Metal reads
 * the resulting recommendation exactly once, when the device is created, so
 * raising the ceiling has to happen before any Metal call -- which is why
 * this runs at startup, not on first use.
 *
 * This does not make the GPU faster. Bandwidth is fixed, and the ceiling
 * only decides how much of the unified pool is addressable. What it buys is
 * room: at 80% of a 128 GiB box the GPU can hold the resident spine and the
 * expert arena wired instead of pushing them back to the SSD.
 *
 * Setting a system-wide sysctl needs root, and cwick normally runs as a user,
 * so failure is the expected outcome and is reported as the one command that
 * would fix it. The ceiling is headroom, never a reason to refuse to start. */
/* Set when cwick wanted a higher ceiling but could not set it (no root). The
 * verdict is deferred to cwick_mem_report_gpu_wired: only the live device
 * knows the working set the driver really offers, and on current macOS that
 * default is frequently *larger* than what the memory plan asks for, in
 * which case "could not raise it" is not a shortfall at all. */
static uint64_t g_wired_target_mb;  /* MiB cwick asked for */
static bool     g_wired_pending;    /* asked, not set — judge at device init */

static bool sysctl_rd_u64(const char *name, uint64_t *out, size_t *size) {
    union { uint32_t u32; uint64_t u64; int32_t i32; int64_t i64; } v;
    size_t len = sizeof(v);
    memset(&v, 0, sizeof(v));
    if (sysctlbyname(name, &v, &len, NULL, 0) != 0) return false;
    if (size) *size = len;
    switch (len) {
        case 4:  *out = (uint64_t)(int64_t)v.i32; return true;
        case 8:  *out = (uint64_t)v.i64;          return true;
        default: return false;
    }
}

static bool sysctl_wr_u64(const char *name, uint64_t val, size_t size) {
    if (size == 8) {
        uint64_t v = val;
        return sysctlbyname(name, NULL, NULL, &v, sizeof(v)) == 0;
    }
    uint32_t v = (uint32_t)val;
    return sysctlbyname(name, NULL, NULL, &v, sizeof(v)) == 0;
}

bool cwick_mem_raise_gpu_wired(uint64_t host_bytes, double frac) {
    const char *env = getenv("CWICK_GPU_WIRED_MB");
    if (env && *env) {
        long long mb = atoll(env);
        if (mb == 0) {
            CWICK_INFO("mem", "GPU wired ceiling left at the system default "
                             "(CWICK_GPU_WIRED_MB=0)");
            return false;
        }
    }
    uint64_t host = host_bytes ? host_bytes : cwick_host_memory_bytes();
    if (host == 0 || frac <= 0.0) return false;

    /* The sysctl counts megabytes, and MiB (not MB) is what the rest of this
     * file means by a megabyte. */
    uint64_t target_mb =
        (uint64_t)((long double)host * (long double)frac / (1024.0L * 1024.0L));
    if (target_mb == 0) return false;

    uint64_t cur = 0;
    size_t sz = 0;
    bool mb_units = true;
    if (!sysctl_rd_u64("iogpu.wired_limit_mb", &cur, &sz)) {
        /* Ventura and older expose the same knob in bytes under a debug key. */
        if (!sysctl_rd_u64("debug.iogpu.wired_limit", &cur, &sz)) {
            CWICK_VERB("mem", "no GPU wired-ceiling sysctl on this macOS — "
                             "using the default Metal working set");
            return false;
        }
        mb_units = false;
    }
    uint64_t cur_mb = mb_units ? cur : cur / (1024 * 1024);

    if (cur_mb >= target_mb) {
        CWICK_VERB("mem", "GPU wired ceiling already %.2f GiB (>= the %.2f GiB "
                         "target) — leaving it alone",
                  cwick_gib(cur_mb * 1024ull * 1024ull),
                  cwick_gib(target_mb * 1024ull * 1024ull));
        return true;
    }

    uint64_t write_val = mb_units ? target_mb : target_mb * 1024ull * 1024ull;
    if (sysctl_wr_u64(mb_units ? "iogpu.wired_limit_mb" : "debug.iogpu.wired_limit",
                      write_val, sz)) {
        CWICK_INFO("mem", "GPU wired ceiling raised to %.2f GiB (%.0f%% of "
                         "%.2f GiB unified memory, was %.2f GiB) — the GPU can "
                         "now wire that much of the resident spine and expert "
                         "arena instead of refaulting it",
                  cwick_gib(target_mb * 1024ull * 1024ull), frac * 100.0,
                  cwick_gib(host), cwick_gib(cur_mb * 1024ull * 1024ull));
        return true;
    }

    /* Almost always EPERM: cwick is not root. Remember what we wanted and stay
     * quiet for now -- whether it matters depends on the working set the
     * driver actually hands out, which is unknowable until the Metal device
     * exists. Judging it here would either cry wolf or miss a real gap. */
    g_wired_target_mb = target_mb;
    g_wired_pending = true;
    CWICK_VERB("mem", "could not raise the GPU wired ceiling to %.2f GiB (%s) "
                     "— checking the working set the driver offers once Metal "
                     "is up", cwick_gib(target_mb * 1024ull * 1024ull),
              errno == EPERM ? "needs root" : strerror(errno));
    return false;
}

/* The first moment the ceiling can be judged: the Metal device exists, so
 * `working_set_bytes` is what the GPU can really wire. Called once, right
 * after device creation. */
bool cwick_mem_report_gpu_wired(uint64_t working_set_bytes) {
    if (!g_wired_pending || g_wired_target_mb == 0 || working_set_bytes == 0)
        return true;
    uint64_t target = g_wired_target_mb * 1024ull * 1024ull;
    g_wired_pending = false;
    if (working_set_bytes >= target) {
        CWICK_INFO("mem", "Metal working set %.2f GiB already covers the "
                         "%.2f GiB memory-plan target — this macOS needs no "
                         "wired-ceiling override",
                  cwick_gib(working_set_bytes), cwick_gib(target));
        return true;
    }
    CWICK_INFO("mem", "Metal working set is %.2f GiB, under the %.2f GiB the "
                     "memory plan targets — the GPU may not be able to wire "
                     "the whole resident spine. Run `sudo sysctl "
                     "iogpu.wired_limit_mb=%llu` before starting cwick (it "
                     "resets on reboot), or set CWICK_GPU_WIRED_MB=0 to "
                     "silence this", cwick_gib(working_set_bytes),
              cwick_gib(target), (unsigned long long)g_wired_target_mb);
    return false;
}

/* ------------------------------------------------------------------ */
/* Planning                                                            */
/* ------------------------------------------------------------------ */
static bool add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a > UINT64_MAX - b) return false;
    *out = a + b;
    return true;
}

cwick_status cwick_mem_plan_build(uint64_t host_bytes, uint64_t resident_bytes,
                                uint64_t state_bytes, uint64_t arena_bytes,
                                uint64_t headroom_bytes,
                                uint64_t streamed_bytes,
                                cwick_mem_plan *out, cwick_error *err) {
    if (!out) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "mem: null plan");
        return CWICK_ERR_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));

    double frac;
    uint64_t limit;
    cwick_memory_policy(&frac, &limit);

    uint64_t host = host_bytes ? host_bytes : cwick_host_memory_bytes();
    if (host == 0) {
        cwick_error_set(err, CWICK_ERR_MEMORY,
                       "mem: host RAM is unknown; refusing an unbounded plan");
        return CWICK_ERR_MEMORY;
    }
    if (state_bytes == 0) state_bytes = cwick_memory_runtime_state();

    long double target_ld = (long double)host * (long double)frac;
    if (target_ld > (long double)UINT64_MAX) target_ld = (long double)UINT64_MAX;
    uint64_t target = (uint64_t)target_ld;
    if (limit > 0 && target > limit) target = limit;

    out->host_bytes = host;
    out->host_fraction = frac;
    out->target_bytes = target;
    out->resident_bytes = resident_bytes;
    out->state_bytes = state_bytes;
    out->arena_bytes = arena_bytes;
    out->headroom_bytes = headroom_bytes;
    out->streamed_bytes = streamed_bytes;

    /* Admission includes every unavoidable fixed allocation, not only the
     * resident weights and runtime state. A plan that merely fits those two
     * while exceeding the target after arenas/headroom must fail closed. */
    uint64_t fixed = resident_bytes;
    if (!add_u64(fixed, state_bytes, &fixed) ||
        !add_u64(fixed, arena_bytes, &fixed) ||
        !add_u64(fixed, headroom_bytes, &fixed) || target == 0 || fixed > target) {
        out->limited_by_ram = true;
        out->expert_cache_bytes = 0;
        cwick_error_set(err, CWICK_ERR_MEMORY,
                       "fixed memory %.2f GiB (spine %.2f + state %.2f + arenas "
                       "%.2f + headroom %.2f) exceeds the %.2f GiB ceiling "
                       "(%.0f%% of %.2f GiB host)",
                       cwick_gib(fixed), cwick_gib(resident_bytes),
                       cwick_gib(state_bytes), cwick_gib(arena_bytes),
                       cwick_gib(headroom_bytes), cwick_gib(target),
                       frac * 100.0, cwick_gib(host));
        return CWICK_ERR_MEMORY;
    }

    /* What is left goes to pinning hot experts in RAM; everything else is
     * read from SSD per decode step.
     *
     * Taking the whole remainder is a trap, and it was the default until it
     * was measured. On a 128 GiB M3 Max, filling the cache to its full
     * 83.60 GiB budget scored *worse* than a 45 GiB cache:
     *
     *   cache   decode/step   tok/s    stream    gemv    attn
     *   21 GiB    1684 ms     0.594    1132 ms   323 ms  158 ms
     *   45 GiB    1665 ms     0.601     889 ms   508 ms  184 ms
     *   84 GiB    2104 ms     0.475     955 ms   834 ms  224 ms
     *
     * The 206 ms/step of SSD traffic the extra 38 GiB saves costs 518 ms/step
     * everywhere else: the resident set grows past the point where macOS can
     * keep the hot pages uncompressed, so every read of a "cached" expert
     * pays for decompression.
     *
     * Re-measured once the decode step stopped being launch-bound (batched
     * Metal dispatch) and the pages stopped being compressible (mlock), the
     * optimum moved a long way down -- the extra bytes no longer buy as much
     * because the fallback path got cheaper, while their cost did not.
     *
     * Those numbers were 15-step runs, and that turned out to matter: a cache
     * of N slots spends its first N admissions *filling*, and every one of
     * those falls through the size-class LIFO to the slow path. A 32 GiB
     * profile showed 4095 LIFO misses against 4096 slots -- the whole run was
     * the fill. Amortised over 15 steps that swamps the steady state; over 99
     * steps it does not. This is the same sweep at 99 decode steps:
     *
     *   cache   ms/step   tok/s   hit%    SSD read   gemv    alloc
     *   16 GiB    798.4   1.252   30.6%   613.5 GiB  150.5    11.2
     *   32 GiB    795.1   1.258   45.2%   484.9 GiB  153.7    24.2
     *   48 GiB    817.1   1.224   50.9%   434.3 GiB  153.2    33.4
     *   64 GiB    913.7   1.094   56.1%   387.9 GiB  197.2    43.1
     *
     * So 32 GiB matches 16 GiB on wall time while cutting SSD traffic 21% and
     * raising the hit rate by 14.6 points -- free, and worth having. Past 32
     * GiB it stops being free: the hit rate keeps climbing to 56% but the step
     * gets slower, and at 64 GiB gemv alone degrades 150 -> 197 ms.
     *
     * The reason a better hit rate stops paying is visible in the phase split:
     * `dispatch` (the reader pool's fork-join) is 435 ms/step at 16 GiB, which
     * is 97% of the entire 448 ms stream phase, while the preads themselves
     * only account for ~259 ms of it across 12 threads. Bytes removed from the
     * SSD do not shorten a phase dominated by synchronisation.
     *
     * So the cap is host/4 -- 32 GiB here. CWICK_EXPERT_CACHE_GIB overrides it
     * for machines where the tradeoff lands somewhere else; the numbers above
     * are the ones to beat, and they say do not go past ~host/4.
     *
     * Re-measured 2026-10-19 on the current build (32 prompt tokens, 64 decode
     * steps, one warm pass each, same prompt):
     *
     *   cache   ms/step   tok/s   GiB/step  reads/step  hit%
     *   32 GiB    743     1.35      5.04       645      46.6
     *   64 GiB    733     1.36      3.45       441      63.5
     *
     * The 64 GiB penalty above no longer reproduces: same wall time, 32% less
     * SSD traffic. What is left is the per-layer fork-join plus GEMV -- at
     * 3.45 GiB/step the drive only delivers 5.0 GB/s, because fewer reads
     * means a shallower queue. So the ceiling here is ~1.4 tok/s, and 64 GiB
     * is worth setting when the extra 32 GiB of RAM is otherwise idle. */
    uint64_t cache = target - fixed;
    /* The cap is on the *cache*, not on `fixed + cache`. Subtracting `fixed`
     * here is what silently disabled the cap on this machine: resident +
     * state + arena + headroom is already 16.1 GiB, so a small `host/N - fixed`
     * underflows to zero and the cache takes the whole remainder -- which is
     * how an 86 GiB cache was once measured at 920 ms/step. */
    uint64_t reserve_cap = host / 4;
    uint64_t soft_cap = reserve_cap;
    bool soft_cap_from_env = false;
    {
        const char *e = getenv("CWICK_EXPERT_CACHE_GIB");
        if (e && *e) {
            char *end = NULL;
            double gib = strtod(e, &end);
            if (end != e && gib > 0.0 && gib < 1024.0) {
                soft_cap = (uint64_t)(gib * CWICK_GIB);
                soft_cap_from_env = true;
            }
        }
    }
    out->cache_cap_from_env = soft_cap_from_env;
    if (soft_cap > 0 && cache > soft_cap) {
        out->expert_cache_bytes = soft_cap;
        out->cache_capped_by_reserve = true;
    } else {
        out->expert_cache_bytes = cache;
        out->cache_capped_by_reserve = false;
    }
    if (out->expert_cache_bytes > SIZE_MAX)
        out->expert_cache_bytes = SIZE_MAX;
    return CWICK_OK;
}

void cwick_mem_plan_log(const cwick_mem_plan *plan, bool verbose) {
    if (!plan) return;
    /* Publish the ceiling before anything else: /v1/metrics is scraped while
     * the model is still loading, and a dashboard divides the footprint by
     * this number. It used to stay 0 forever, which read as "no ceiling". */
    cwick_metrics_set_memory_ceiling(plan->target_bytes);
    CWICK_INFO("mem", "memory budget: %.2f GiB host, using at most %.0f%% "
              "(%.2f GiB)", cwick_gib(plan->host_bytes),
              plan->host_fraction * 100.0, cwick_gib(plan->target_bytes));
    if (!verbose) return;
    CWICK_VERB("mem", "budget split: resident spine %.2f GiB + runtime state "
              "%.2f GiB + arenas %.2f GiB + read headroom %.2f GiB",
              cwick_gib(plan->resident_bytes), cwick_gib(plan->state_bytes),
              cwick_gib(plan->arena_bytes), cwick_gib(plan->headroom_bytes));
    CWICK_VERB("mem", "expert cache budget %.2f GiB (routed expert bank is "
              "%.2f GiB, streamed per decode step)",
              cwick_gib(plan->expert_cache_bytes), cwick_gib(plan->streamed_bytes));
    if (plan->cache_capped_by_reserve)
        CWICK_INFO("mem", "expert cache held at %.2f GiB (%s) rather than the "
                         "%.2f GiB the budget allows — over 99 decode steps the "
                         "extra bytes stopped paying: the stream phase is owned "
                         "by the per-layer fork-join, not by the reads, so a "
                         "higher hit rate costs more than it saves; the 64 GiB "
                         "penalty no longer reproduces, so raise this by hand "
                         "when the RAM is otherwise idle",
                  cwick_gib(plan->expert_cache_bytes),
                  plan->cache_cap_from_env ? "CWICK_EXPERT_CACHE_GIB"
                                           : "the host/4 default",
                  cwick_gib(plan->target_bytes - plan->resident_bytes -
                           plan->state_bytes - plan->arena_bytes -
                           plan->headroom_bytes));
}
