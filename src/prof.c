/* prof.c — opt-in decode-step phase profiler (CWICK_PROFILE=1).
 *
 * The engine already prints a per-step wall time, which says nothing about
 * *why* a step costs what it does. A decode step is three very different
 * things stitched together:
 *
 *   - demand expert bytes: a sticky-cache lookup and, on a miss, a blocking
 *     F_NOCACHE pread of a multi-MiB matrix straight off the SSD;
 *   - the routed experts themselves: a dequantizing GEMV per matrix;
 *   - attention / KDA, which stays token-serial.
 *
 * Which of those dominates decides the optimization order, and it is not
 * observable from outside the process: Activity Monitor shows one busy or
 * idle process either way, whether the time is spent inside read() or inside
 * our own arithmetic. (It also shows ~0% GPU, because there is no Metal or
 * MPS path in this framework at all -- see docs/ARCHITECTURE.md.)
 *
 * Cost when disabled: one getenv() on the first probe, then a load and a
 * test per sample. Callers still pay two cwick_prof_ns() calls per sample;
 * at ~1000 expert matrices per decode step that is tens of microseconds
 * against a multi-second step, so the call sites stay unconditional.
 */
#include "cwick.h"
#include "log.h"
#include "util.h"

#include <stdlib.h>
#include <time.h>

static int g_prof = -1; /* -1 unknown, 0 off, 1 on */
static uint64_t g_ns[CWICK_PH_COUNT];
static uint64_t g_steps;

bool cwick_prof_on(void) {
    if (g_prof < 0) g_prof = cwick_env_true("CWICK_PROFILE") ? 1 : 0;
    return g_prof == 1;
}

uint64_t cwick_prof_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

void cwick_prof_add(cwick_prof_phase p, uint64_t ns) {
    /* The probe has to be armed here, not only in cwick_prof_on(): add() is
     * the call the hot paths use, so a lazy flag left at "unknown" by a
     * missing cwick_prof_on() would silently discard every sample. */
    if (!cwick_prof_on()) return;
    if ((unsigned)p >= (unsigned)CWICK_PH_COUNT) return;
    g_ns[p] += ns;
}

void cwick_prof_step(void) {
    if (cwick_prof_on()) g_steps++;
}

void cwick_prof_reset(void) {
    (void)cwick_prof_on();
    for (int i = 0; i < (int)CWICK_PH_COUNT; i++) g_ns[i] = 0;
    g_steps = 0;
    /* The fetch sub-counters live in moe.c / stream.c / metal_objc.m and are
     * reported alongside the phases, so they must reset with them -- otherwise
     * a run's prefill (which streams tens of GiB and reserves thousands of
     * slots) leaks into the decode averages. */
    {
        extern uint64_t g_fetch_pread_ns;
        extern uint64_t g_fetch_dispatch_ns;
        extern uint64_t g_fetch_alloc_ns;
        extern uint64_t g_res_probe_ns;
        extern uint64_t g_res_evict_ns;
        extern uint64_t g_res_slot_ns;
        extern uint64_t g_arena_alloc_ns;
        extern uint64_t g_arena_free_ns;
        extern uint64_t g_arena_hot_hits;
        extern uint64_t g_arena_hot_misses;
        g_fetch_pread_ns = g_fetch_dispatch_ns = g_fetch_alloc_ns = 0;
        extern uint64_t g_attn_proj_ns;
        extern uint64_t g_attn_conv_ns;
        extern uint64_t g_attn_gate_ns;
        extern uint64_t g_attn_rec_ns;
        extern uint64_t g_attn_mla_ns;
        extern uint64_t g_attn_tail_ns;
        g_attn_proj_ns = g_attn_conv_ns = g_attn_gate_ns = 0;
        g_attn_rec_ns = g_attn_mla_ns = g_attn_tail_ns = 0;
        g_res_probe_ns = g_res_evict_ns = g_res_slot_ns = 0;
        g_arena_alloc_ns = g_arena_free_ns = 0;
        g_arena_hot_hits = g_arena_hot_misses = 0;
        extern uint64_t g_arena_grow_ns;
        extern uint64_t g_arena_grows;
        g_arena_grow_ns = g_arena_grows = 0;
        extern uint64_t g_arena_hot_ns;
        g_arena_hot_ns = 0;
    }
}

const char *cwick_prof_phase_name(cwick_prof_phase p) {
    switch (p) {
    case CWICK_PH_STREAM:    return "stream";
    case CWICK_PH_GEMV:      return "gemv";
    case CWICK_PH_ATTENTION: return "attn";
    case CWICK_PH_OTHER:     return "other";
    default:                return "?";
    }
}

void cwick_prof_report(const char *tag, double decode_ms) {
    if (!cwick_prof_on()) return;
    if (g_steps == 0 || decode_ms <= 0.0) return;

    uint64_t bucket_ns = 0;
    for (int i = 0; i < (int)CWICK_PH_COUNT; i++) bucket_ns += g_ns[i];

    CWICK_INFO(tag, "profile: %llu decode steps, %.1f ms/step wall",
              (unsigned long long)g_steps, decode_ms / (double)g_steps);
    for (int i = 0; i < (int)CWICK_PH_COUNT; i++) {
        double ms = (double)g_ns[i] * 1.0e-6 / (double)g_steps;
        double pct = (double)g_ns[i] * 1.0e-6 * 100.0 / decode_ms;
        CWICK_INFO(tag, "  %-6s %9.2f ms/step  %5.1f%%",
                  cwick_prof_phase_name((cwick_prof_phase)i), ms, pct);
    }
    {
        /* Time the reader threads spent inside pread(), summed over all of
         * them. Divided by the pool width it is the serial-equivalent read
         * time; the gap to the stream phase is cache management + dispatch.
         * (Summed, not wall: twelve threads each contribute their own.) */
        extern uint64_t g_fetch_pread_ns;
        extern uint64_t g_fetch_dispatch_ns;
        extern uint64_t g_fetch_alloc_ns;
        double ms = (double)g_fetch_pread_ns * 1.0e-6 / (double)g_steps;
        CWICK_INFO(tag, "  %-6s %9.2f ms/step  %5.1f%%  (of stream: inside pread, "
                       "summed over reader threads)",
                  "pread", ms, (double)g_fetch_pread_ns * 1.0e-6 * 100.0 / decode_ms);
        CWICK_INFO(tag, "  %-6s %9.2f ms/step  %5.1f%%  (of stream: pool fork-join)",
                  "dispatch", (double)g_fetch_dispatch_ns * 1.0e-6 / (double)g_steps,
                  (double)g_fetch_dispatch_ns * 1.0e-6 * 100.0 / decode_ms);
        CWICK_INFO(tag, "  %-6s %9.2f ms/step  %5.1f%%  (of stream: slot reservation)",
                  "alloc", (double)g_fetch_alloc_ns * 1.0e-6 / (double)g_steps,
                  (double)g_fetch_alloc_ns * 1.0e-6 * 100.0 / decode_ms);
        {
        extern uint64_t g_res_probe_ns;
        extern uint64_t g_res_evict_ns;
        extern uint64_t g_res_slot_ns;
        extern uint64_t g_res_shmem_ns;
        extern uint64_t g_res_calls;
        double nres = (double)g_res_calls / (double)g_steps;
        CWICK_INFO(tag, "  %-6s %9.2f ms/step  %5.1f%%  (of alloc: hash probe)",
                  "probe", (double)g_res_probe_ns * 1.0e-6 / (double)g_steps,
                  (double)g_res_probe_ns * 1.0e-6 * 100.0 / decode_ms);
        CWICK_INFO(tag, "  %-6s %9.2f ms/step  %5.1f%%  (of alloc: eviction)",
                  "evict", (double)g_res_evict_ns * 1.0e-6 / (double)g_steps,
                  (double)g_res_evict_ns * 1.0e-6 * 100.0 / decode_ms);
        CWICK_INFO(tag, "  %-6s %9.2f ms/step  %5.1f%%  (of alloc: entry malloc, %.0f calls)",
                  "entry", (double)g_res_slot_ns * 1.0e-6 / (double)g_steps,
                  (double)g_res_slot_ns * 1.0e-6 * 100.0 / decode_ms, nres);
        CWICK_INFO(tag, "  %-6s %9.2f ms/step  %5.1f%%  (of alloc: shmem_alloc)",
                  "shmem", (double)g_res_shmem_ns * 1.0e-6 / (double)g_steps,
                  (double)g_res_shmem_ns * 1.0e-6 * 100.0 / decode_ms);
        {
            extern uint64_t g_shmem_probe_ns;
            extern uint64_t g_shmem_arena_ns;
            extern uint64_t g_shmem_fallback_ns;
            CWICK_INFO(tag, "  %-6s %9.2f ms/step  %5.1f%%  (of shmem: branch preamble)",
                      "shprobe", (double)g_shmem_probe_ns * 1.0e-6 / (double)g_steps,
                      (double)g_shmem_probe_ns * 1.0e-6 * 100.0 / decode_ms);
            CWICK_INFO(tag, "  %-6s %9.2f ms/step  %5.1f%%  (of shmem: arena call)",
                      "sharena", (double)g_shmem_arena_ns * 1.0e-6 / (double)g_steps,
                      (double)g_shmem_arena_ns * 1.0e-6 * 100.0 / decode_ms);
            CWICK_INFO(tag, "  %-6s %9.2f ms/step  %5.1f%%  (of shmem: calloc fallback)",
                      "shfallb", (double)g_shmem_fallback_ns * 1.0e-6 / (double)g_steps,
                      (double)g_shmem_fallback_ns * 1.0e-6 * 100.0 / decode_ms);
        {
            extern uint64_t g_arena_hot_hits;
            extern uint64_t g_arena_hot_misses;
            CWICK_INFO(tag, "  arena hot LIFO: %llu hits / %llu misses",
                      (unsigned long long)g_arena_hot_hits,
                      (unsigned long long)g_arena_hot_misses);
        }
        {
            /* Where the `attn` phase actually goes. KDA runs on 34 layers and
             * MLA on 11, so the two are not comparable per layer -- read them
             * as per-step totals, and use them to decide which half is worth
             * moving to the GPU. */
            extern uint64_t g_attn_proj_ns;
            extern uint64_t g_attn_conv_ns;
            extern uint64_t g_attn_gate_ns;
            extern uint64_t g_attn_rec_ns;
            extern uint64_t g_attn_mla_ns;
            extern uint64_t g_attn_tail_ns;
            static const struct { const char *name; uint64_t *p; } sub[] = {
                { "kda_proj", &g_attn_proj_ns },  /* q/k/v: already a GPU batch */
                { "kda_conv", &g_attn_conv_ns },  /* short conv + sigmoid + L2  */
                { "kda_gate", &g_attn_gate_ns },  /* f_a/f_b/b + decay + beta   */
                { "kda_rec",  &g_attn_rec_ns },   /* the recurrence itself      */
                { "mla_all",  &g_attn_mla_ns },   /* 11 sparse-attention layers */
                { "kda_tail", &g_attn_tail_ns },  /* gated RMSNorm + o_proj     */
            };
            for (unsigned i = 0; i < sizeof(sub) / sizeof(sub[0]); i++) {
                double ms = (double)*sub[i].p * 1.0e-6 / (double)g_steps;
                CWICK_INFO(tag, "  %-8s %9.2f ms/step  %5.1f%%  (of attn)",
                          sub[i].name, ms,
                          (double)*sub[i].p * 1.0e-6 * 100.0 / decode_ms);
            }
            {
                extern uint64_t g_arena_grow_ns;
                extern uint64_t g_arena_grows;
                CWICK_INFO(tag, "  arena slow path: %llu grow(s), %.2f ms/step",
                          (unsigned long long)g_arena_grows,
                          (double)g_arena_grow_ns * 1.0e-6 / (double)g_steps);
                {
                    extern uint64_t g_arena_hot_ns;
                    CWICK_INFO(tag, "  arena LIFO hit: %.2f ms/step",
                              (double)g_arena_hot_ns * 1.0e-6 / (double)g_steps);
                }
            }
            }
        }
    }
    {
        double seen = (double)bucket_ns * 1.0e-6;
        double rest = decode_ms - seen;
        if (rest < 0.0) rest = 0.0;
        CWICK_INFO(tag, "  %-6s %9.2f ms/step  %5.1f%%  (glue: router, "
                       "shared experts, norms, alloc)",
                  "rest", rest / (double)g_steps,
                  rest * 100.0 / decode_ms);
    }
    {
        /* Where the routed experts actually ran. A GEMV phase of 150-230 ms
         * against a 46 ms/step microbenchmark is only explicable if part of
         * the work is not on the GPU at all, and CWICK_GEMV_STATS cannot tell
         * a group Metal declined from a single matrix it declined. */
        extern uint64_t g_moe_group_metal;
        extern uint64_t g_moe_group_cpu;
        extern uint64_t g_moe_group_serial;
        extern uint64_t g_moe_single_metal;
        extern uint64_t g_moe_single_cpu;
        extern uint64_t g_moe_batch_metal;
        extern uint64_t g_moe_batch_cpu;
        extern uint64_t g_moe_batch_serial;
        extern uint64_t g_moe_batch_shared_gpu;
        extern uint64_t g_moe_batch_shared_cpu;
        uint64_t per = g_steps ? g_steps : 1;
        CWICK_INFO(tag, "  moe placement per step: batch/gpu %.1f, batch->cpu %.1f, "
                       "serial %.1f, single gpu %.1f, single cpu %.1f",
                  (double)g_moe_group_metal / (double)per,
                  (double)g_moe_group_cpu / (double)per,
                  (double)g_moe_group_serial / (double)per,
                  (double)g_moe_single_metal / (double)per,
                  (double)g_moe_single_cpu / (double)per);
        /* cwick_moe_forward_batch windows: how many route-first fetches ran
         * (prefill chunks + decode groups), and where each window's routed
         * experts landed. Raw counts -- the report runs once per request. */
        CWICK_INFO(tag, "  batch windows: metal %llu, ->cpu %llu, serial %llu",
                  (unsigned long long)g_moe_batch_metal,
                  (unsigned long long)g_moe_batch_cpu,
                  (unsigned long long)g_moe_batch_serial);
        {
            /* forward_group's own wall against the accounted phases: the gap
             * is time spent in the group loop around it. */
            extern double g_group_fg_ms;
            if (g_group_fg_ms > 0.0)
                CWICK_INFO(tag, "  group forward wall %.1f ms/step (decoded "
                               "steps %llu)", g_group_fg_ms / (double)per,
                          (unsigned long long)g_steps);
            /* Split of the batch function's uninstrumented completion: the
             * routed norm/up projections alone versus the whole section. */
            extern double g_batch_completion_ms;
            extern double g_batch_router_ms;
            extern double g_batch_shared_ms;
            if (g_batch_completion_ms > 0.0)
                CWICK_INFO(tag, "  batch completion (norm+up proj) %.1f ms/step",
                          g_batch_completion_ms / (double)per);
            if (g_batch_router_ms > 0.0 || g_batch_shared_ms > 0.0)
                CWICK_INFO(tag, "  batch router %.1f, shared expert %.1f ms/step "
                               "(gpu %llu / cpu %llu calls)",
                          g_batch_router_ms / (double)per,
                          g_batch_shared_ms / (double)per,
                          (unsigned long long)g_moe_batch_shared_gpu,
                          (unsigned long long)g_moe_batch_shared_cpu);
        }
    }
    }
}

/* Prefill has no phase timers of its own -- the batched path calls neither
 * cwick_prof_add nor cwick_prof_step -- so the decode report drops it wholesale.
 * What it does accumulate unconditionally is the fetch sub-counters in moe.c.
 * Sampling those either side of the prefill call splits the phase into
 * "waiting on the SSD" and "everything else" without instrumenting the batched
 * path, which is the only question that matters when prefill measures at a
 * third of decode's read bandwidth. */
void cwick_prof_report_prefill(const char *tag, double prefill_ms,
                              uint64_t bytes_streamed) {
    if (!cwick_prof_on()) return;
    if (prefill_ms <= 0.0) return;
    {
        extern uint64_t g_fetch_pread_ns;
        extern uint64_t g_fetch_dispatch_ns;
        extern uint64_t g_fetch_alloc_ns;

        double dms = (double)g_fetch_dispatch_ns * 1.0e-6;  /* fork-join wall */
        double pms = (double)g_fetch_pread_ns * 1.0e-6;     /* summed threads */
        double ams = (double)g_fetch_alloc_ns * 1.0e-6;
        double other = prefill_ms - dms - ams;
        if (other < 0.0) other = 0.0;

        CWICK_INFO(tag, "prefill profile: %.0f ms wall, %.2f GiB streamed",
                  prefill_ms, (double)bytes_streamed / 1073741824.0);
        CWICK_INFO(tag, "  %-8s %9.0f ms  %5.1f%%  (expert fetch, fork-join wall)",
                  "fetch", dms, dms * 100.0 / prefill_ms);
        CWICK_INFO(tag, "  %-8s %9.0f ms  %5.1f%%  (inside pread, summed over readers)",
                  "pread", pms, pms * 100.0 / prefill_ms);
        CWICK_INFO(tag, "  %-8s %9.0f ms  %5.1f%%  (slot reservation)",
                  "alloc", ams, ams * 100.0 / prefill_ms);
        CWICK_INFO(tag, "  %-8s %9.0f ms  %5.1f%%  (attention + GEMMs + glue)",
                  "compute", other, other * 100.0 / prefill_ms);
        if (dms > 0.0 && bytes_streamed > 0)
            CWICK_INFO(tag, "  read bandwidth: %.2f GB/s over the fetch wall",
                      (double)bytes_streamed / (dms / 1000.0) / 1.0e9);
    }
}
