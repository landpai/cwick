/* mem.h — host-memory policy for a checkpoint larger than RAM.
 *
 * CWICK's target is the *complete* zai-org/GLM-5.3-Flash checkpoint (306 GiB
 * across 62 shards) on one Apple Silicon Mac with far less RAM than that.
 * Residency therefore cannot be left to the kernel: mmap the shards, let the
 * unified buffer cache grow, and the machine swaps before the first token is
 * out. What has to be resident is decided here, once, from the machine the
 * process is actually running on.
 *
 * The plan:
 *
 *   target       = host RAM * fraction (default 0.80, i.e. at most 80% of RAM)
 *   - resident   = the weight spine every token touches (mapped + warm-wired)
 *   - state      = KV / KDA recurrent / mHC runtime state allocated once
 *   - arena      = streaming staging buffers + prefetch zones
 *   - headroom   = two full routed layers of demand reads, so prefill can
 *                  page the next layer in while the current one computes
 *   = expert cache budget (what is left for pinning hot experts in RAM)
 *
 * Everything else — the whole routed expert bank — stays on SSD and is read
 * per decode step through F_NOCACHE descriptors (stream.c). The budget is a
 * hard cap: the engine re-checks actual RSS during decode and trims the cache
 * back under `target` if the process grows past it.
 */
#ifndef CWICK_MEM_H
#define CWICK_MEM_H

#include <stdbool.h>
#include <stdint.h>

#include "cwick.h"

/* Default share of physical RAM cwick is willing to occupy while running the
 * full model. Deliberately below 100%: the rest covers the OS, the terminal,
 * the page cache the model files still need and transient allocation spikes. */
#define CWICK_MEM_DEFAULT_FRACTION 0.80
#define CWICK_MEM_MIN_FRACTION     0.50
#define CWICK_MEM_MAX_FRACTION     0.95

/* A completed budget decision. Every field is bytes.
 * (The typedef itself lives in cwick.h alongside the other opaque handles.) */
struct cwick_mem_plan {
    uint64_t host_bytes;          /* physical RAM of this host */
    uint64_t target_bytes;        /* hard ceiling: host * fraction */
    uint64_t resident_bytes;      /* resident weight spine */
    uint64_t state_bytes;         /* KV / KDA / anchors runtime state */
    uint64_t arena_bytes;         /* streaming + prefetch arenas */
    uint64_t headroom_bytes;      /* per-step transient expert reads */
    uint64_t streamed_bytes;      /* routed expert bank (read per decode) */
    uint64_t expert_cache_bytes;  /* pinned hot-expert budget (result) */
    bool cache_capped_by_reserve; /* cache < budget: leaving the machine room */
    bool cache_cap_from_env;      /* the cap came from CWICK_EXPERT_CACHE_GIB */
    double host_fraction;         /* fraction actually applied */
    bool limited_by_ram;          /* true when target < resident */
};

/* Total physical memory of this host (0 when unknown). */
uint64_t cwick_host_memory_bytes(void);

/* Resident set size of this process right now (0 when unknown). Note: on macOS
 * this EXCLUDES pages the compressor has taken, so it understates what a pinned
 * cache really costs. Prefer the footprint for budgeting. */
uint64_t cwick_process_rss_bytes(void);

/* Physical footprint (0 when unknown): resident + compressed pages, i.e. what
 * the OS accounts to this process. This is the honest number for "is the pin
 * cache actually resident", and it is what the memory governor should steer on. */
uint64_t cwick_process_footprint_bytes(void);

/* ------------------------------------------------------------------ */
/* Swap and the compressor                                             */
/* ------------------------------------------------------------------ */
/* macOS compresses anonymous pages before it swaps them, and a compressed
 * page costs a fault to read back. For a CPU that is microseconds; for the
 * GPU reading the same unified memory it measured as 65 ms average and 3.6 s
 * worst case per decode step, because the stall lands inside
 * waitUntilCompleted and never shows up in GPUEndTime.
 *
 * So "stay inside 80% of RAM" is only half the requirement: the pages cwick
 * owns must also stay uncompressed and unswapped. cwick wires what it can
 * (mlock) and reports what it could not, instead of hoping. */
typedef struct {
    uint64_t swap_used, swap_total;   /* bytes; 0 when unknown */
    uint64_t compressions;            /* host-wide, since boot */
    uint64_t decompressions;
    uint64_t swapins, swapouts;
    uint64_t wired;                   /* bytes currently wired host-wide */
} cwick_vm_state;

bool cwick_vm_state_sample(cwick_vm_state *out);

/* Wire [p, p+n) so the VM can neither compress nor swap it. Returns true on
 * success. Fails loudly (once) and returns false when the host refuses --
 * usually a privilege or a wired-memory limit, in which case the caller keeps
 * running unlocked rather than aborting a model load that has already read
 * 15 GiB of weights. */
bool cwick_mem_lock_range(const void *p, size_t n, const char *what);
/* Release a wired range whose owner is being unloaded. Without it a swapped-out
 * model leaves its spine wired, and the incoming model cannot reclaim those
 * pages -- the switch then runs over the ceiling with the pages it cannot get
 * back. */
void cwick_mem_unlock_range(const void *p, size_t n);
/* Bytes this process has successfully wired. */
uint64_t cwick_mem_locked_bytes(void);

/* ------------------------------------------------------------------ */
/* Policy (set once from the CLI before the engine is built)           */
/* ------------------------------------------------------------------ */
/* `--mem-fraction PCT` / CWICK_MEM_PCT: how much of host RAM cwick may
 * occupy (50..95). `--max-memory GiB` / CWICK_MEM_MAX_GIB: an absolute
 * ceiling that wins when it is smaller than the host fraction. */
void cwick_memory_set_policy(double host_fraction, uint64_t limit_bytes);
void cwick_memory_policy(double *host_fraction, uint64_t *limit_bytes);

/* Runtime state (KV/KDA/anchors) allocated by the engine *before* the weight
 * spine is opened, so the expert-cache budget can subtract it. */
void cwick_memory_set_runtime_state(uint64_t bytes);
uint64_t cwick_memory_runtime_state(void);

/* `--allow-swap` / CWICK_ALLOW_SWAP: let the VM compress or page cwick's own
 * memory again. Off by default, because "stay under 80% of RAM" is not the
 * same promise as "stay under 80% of RAM *and stay there*": a compressed page
 * costs a fault to read back, and this machine measured 65 ms average /
 * 3.6 s worst case per decode step when the GPU had to fault the expert
 * arena back in. */
void cwick_memory_set_noswap(bool noswap);
bool cwick_memory_noswap(void);

/* Raise the GPU's wired-memory ceiling to `frac` of host RAM (the same
 * fraction the memory plan targets, i.e. 80% by default).
 *
 * macOS defaults to letting the GPU wire about three quarters of unified
 * memory; iogpu.wired_limit_mb (0 = system default) moves that line, and
 * Metal reads it once when the device is created -- so this must be called
 * before the first Metal call, not lazily on first use.
 *
 * It is a headroom knob, not a speed knob: bandwidth is unchanged. Setting a
 * system-wide sysctl requires root, so the ordinary outcome for a server
 * running as a user is a printed `sudo sysctl ...` hint and a false return.
 * Callers must treat false as "continue", never as a startup failure.
 * CWICK_GPU_WIRED_MB=0 opts out. */
bool cwick_mem_raise_gpu_wired(uint64_t host_bytes, double frac);

/* Judge the ceiling now that the Metal device exists: `working_set_bytes` is
 * the driver's real recommendedMaxWorkingSetSize. Logs either "already
 * covers the target" or the exact `sudo sysctl` that would close the gap.
 * Safe to call when no raise was attempted (returns true, prints nothing). */
bool cwick_mem_report_gpu_wired(uint64_t working_set_bytes);

/* ------------------------------------------------------------------ */
/* Planning                                                            */
/* ------------------------------------------------------------------ */
/* Build a plan. `resident`, `state`, `arena` and `headroom` are the
 * unavoidable costs; whatever is left under `target` becomes the expert
 * cache. Returns CWICK_ERR_MEMORY when even the resident spine does not fit
 * under the ceiling — starting anyway would only mean swapping. */
cwick_status cwick_mem_plan_build(uint64_t host_bytes, uint64_t resident_bytes,
                                uint64_t state_bytes, uint64_t arena_bytes,
                                uint64_t headroom_bytes,
                                uint64_t streamed_bytes,
                                cwick_mem_plan *out, cwick_error *err);

/* Convenience: bytes -> GiB, used by every budget line we print. Returning a
 * double (rather than a formatted string) keeps the logger lock-free. */
double cwick_gib(uint64_t bytes);

/* Print the plan. `verbose` adds the per-item breakdown. */
void cwick_mem_plan_log(const cwick_mem_plan *plan, bool verbose);

#endif /* CWICK_MEM_H */
