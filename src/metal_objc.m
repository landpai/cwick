// metal_objc.m — ObjC compile unit for cwick's optional Metal backend.
//
// cwick is a pure-C program and its decode step has no GPU path at all: the
// routed-expert GEMV (the single biggest arithmetic hotspot of a decode step)
// runs as scalar dequant + GEMV on the CPU. This unit is the bridge that lets
// cwick hand that loop to the GPU.
//
// Why an .m file: the Metal runtime is ObjC. The device/queue/buffer APIs
// are C-callable, but dispatch takes MTLSize structs *by value*, which the
// variadic objc_msgSend ABI mangles. Compiling this unit as ObjC lets the
// compiler generate the correct struct ABI for every call, while everything
// outside stays plain C (the shader source lives in metal_shader.c).
//
// The shader is compiled at runtime from an embedded source string
// (newLibraryWithSource), so no .metallib precompile step or asset pipeline
// is needed.
//
// ---------------------------------------------------------------------------
// Zero-copy: why the expert cache lives inside Metal buffers
// ---------------------------------------------------------------------------
// On Apple Silicon the GPU and the CPU share one physical memory, so a
// MTLBuffer in StorageModeShared *is* ordinary malloc-able RAM that the GPU
// can also address. That makes the obvious implementation of this backend a
// trap: allocating a fresh 8 MiB buffer per call and memcpy()ing the expert
// weights into it costs the full memory bandwidth twice for zero benefit --
// measured at 858 us per expert GEMV, i.e. 0.93 s of a decode step spent
// purely moving bytes the SSD had just delivered.
//
// The fix is `newBufferWithBytesNoCopy` over the mapped model, done once:
// make the *destination* of the expert read itself GPU-visible. Every
// expert-cache blob and prefetch slot is carved out of a large shared
// MTLBuffer arena, so `cwick_stream_read` preads from the SSD straight into
// memory the GPU can read. `cwick_metal_gemv_fp8` then binds that arena with
// an offset instead of copying -- no allocation, no memcpy, one dispatch.
// Buffers that do not live in an arena (resident safetensors mappings, test
// harnesses) fall back to a reused scratch buffer, so the backend is correct
// either way and only the fast path changes.
//
// This backend is opt-in (--backend metal / CWICK_METAL=1). If the
// device/library/pipeline fails to come up, cwick_metal_gemv_fp8 returns 0 and
// the engine silently falls back to the CPU path -- GPU offload must never
// break a generation.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <sys/mman.h>
#include <mach/mach_time.h>

#include "cwick.h"
#include "log.h"
#include "mem.h"

/* The shader source string is defined here (pure C) so the ObjC unit and any
 * future caller share one copy. */
extern const char *cwick_metal_shader_source(void);

typedef struct {
    uint32_t k;
    uint32_t cols;
    uint32_t ncb;
    uint32_t n_rows;
    uint32_t stride;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
} cwick_metal_gemv_args;

typedef struct {
    uint32_t k;
    uint32_t cols;
    uint32_t rows_per_tg;   /* simdgroups per threadgroup: rows [gid.y*rptg, +rptg) */
    uint32_t pad1;
    uint32_t pad2;
} cwick_metal_bf16_args;

static id<MTLDevice>                g_dev;
static id<MTLCommandQueue>          g_queue;
static id<MTLComputePipelineState>  g_pso;      // FP8 expert GEMV v1
static id<MTLComputePipelineState>  g_pso_v2;   // FP8 expert GEMV v2
static id<MTLComputePipelineState>  g_pso_v3;   // FP8 expert GEMV v3 (LUT decode, default)
static id<MTLComputePipelineState>  g_pso_v3d;  // v3 without the LUT (diagnostic)
static id<MTLComputePipelineState>  g_pso_v4;   // v3 slicing + arithmetic decode (opt-in, slower)
static id<MTLComputePipelineState>  g_pso_bf16; // resident BF16 linear GEMV
/* DeepSeek's two weight formats: e2m1 with a 32-element e8m0 scale (routed and
 * shared experts) and bf16 with the same block scale (every dense projection).
 * Optional: if either function is missing the DeepSeek backend simply keeps the
 * whole GEMV on the CPU pool. */
static id<MTLComputePipelineState>  g_pso_ds_fp4;
static id<MTLComputePipelineState>  g_pso_ds_bf16;
static id<MTLComputePipelineState>  g_pso_ka;   // keepalive
static int                          g_fp8_kernel = 3;  /* CWICK_FP8_KERNEL */

/* A command buffer of compute dispatches is *serialised* by default: Metal
 * will not retire dispatch i+1 until i retires. That default is right for a
 * chain of kernels and exactly wrong for the two batches here, whose only
 * dispatches are independent GEMVs that write disjoint output slabs. With
 * the serial default each 8 MiB matrix waited for its own read to finish
 * before the next one could start, so the GPU measured ~45 GB/s on the
 * routed experts -- latency, not bandwidth. A concurrent encoder lets the
 * dispatches overlap: the same batch then runs at ~118 GB/s, which is no
 * longer the transport but the fp8 decode (see the note below).
 *
 * Only the *encoder* decides this; MTLCommandBufferDescriptor has no such
 * property. The fallback matters because a refused batch costs the caller a
 * whole CPU redo, while a serial batch costs only speed. */
static id<MTLComputeCommandEncoder> cwick_concurrent_encoder(id<MTLCommandBuffer> cb) {
    if (!cb) return nil;
    if (@available(macOS 12.0, *)) {
        id<MTLComputeCommandEncoder> e =
            [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        if (e) return e;
    }
    return [cb computeCommandEncoder];
}

/* ------------------------------------------------------------------ */
/* GPU keepalive                                                        */
/* ------------------------------------------------------------------ */
/* A decode step spends ~85% of its wall time on the CPU (streaming the
 * routed experts off the SSD, running attention) with only ~1.4 ms of GPU
 * work per layer in between. Apple's GPU power management reads those gaps
 * as idle and drops the clock, and the ramp-up cost is paid inside the next
 * command buffer's GPUEndTime -- so a slow "GEMV phase" can be a power-
 * management artefact rather than a kernel problem. Measured with
 * tools/verify/mbatch.c (16 dispatches of the real expert shape):
 *
 *   back-to-back                46 us/dispatch   (172 GB/s)
 *   ...after a 2 ms idle gap    61 us/dispatch
 *   ...after a 6 ms idle gap    96 us/dispatch    (83 GB/s)
 *
 * The real loop's gaps are 5-10 ms, i.e. the 96 us column. A thread that
 * dispatches the trivial keepalive kernel every CWICK_KA_PERIOD microseconds
 * holds the clock up and recovers ~60 ms per decode step. It costs one
 * thread and ~2% of the GPU; CWICK_KEEPALIVE=0 disables it. */
static id<MTLBuffer> g_ka_buf;
static pthread_t     g_ka_thread;
static int           g_ka_run;

static void *ka_main(void *arg) {
    (void)arg;
    const char *pe = getenv("CWICK_KA_PERIOD");
    long period = pe ? strtol(pe, NULL, 10) : 600;
    if (period < 100) period = 100;
    if (period > 20000) period = 20000;
    while (g_ka_run) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:g_pso_ka];
            [enc setBuffer:g_ka_buf offset:0 atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        usleep((useconds_t)period);
    }
    return NULL;
}

int cwick_metal_keepalive_start(void) {
    if (!g_pso_ka || !g_ka_buf || g_ka_run) return 0;
    if (getenv("CWICK_KEEPALIVE") &&
        strcmp(getenv("CWICK_KEEPALIVE"), "0") == 0) return 0;
    g_ka_run = 1;
    if (pthread_create(&g_ka_thread, NULL, ka_main, NULL) != 0) {
        g_ka_run = 0;
        return 0;
    }
    return 1;
}

void cwick_metal_keepalive_stop(void) {
    if (!g_ka_run) return;
    g_ka_run = 0;
    pthread_join(g_ka_thread, NULL);
}

/* ------------------------------------------------------------------ */
/* Shared-memory arena: GPU-visible allocation for streamed experts   */
/* ------------------------------------------------------------------ */

#define CWICK_ARENA_PAGE   4096u
#define CWICK_ARENA_MIN    (256ull << 20)   /* 256 MiB first arena        */
#define CWICK_ARENA_MAX    (2ull  << 30)    /* 2 GiB arena ceiling        */

/* Size-class LIFO for the blocks the expert cache churns.
 *
 * The arena's general free list is address-ordered and coalesced, which is
 * the right structure for merging neighbours and the wrong one for the
 * decode loop's actual traffic: ~550 admissions and ~550 evictions per step,
 * every one of them the SAME size (one FP8 matrix + its scale side). Both
 * directions walked that list, which measured 43 ms/step of alloc plus
 * ~27 ms/step of free at a 16 GiB cache -- pure bookkeeping, and the reason
 * a larger cache (more arenas, longer lists) lost more than its better hit
 * rate earned.
 *
 * These LIFOs hold blocks of exactly one size each and never coalesce, so a
 * block that lands here is invisible to the general list. That is fine for
 * the uniform blob traffic and costs nothing else: the few genuinely
 * different sizes (the 24 MiB prefill staging slab, the prefetch zones) keep
 * using the general path, and a request that misses here still falls through
 * to it.
 *
 * Why 64 bins and not 8: the bin is chosen by `(size / 4K) & (BINS-1)`, and
 * with 8 of them the checkpoint's gate/up/down matrices -- all about 7.4 MiB,
 * all differing by a few pages -- collided in one bin. A bin records exactly
 * one size, so every allocation of a size other than the one that most
 * recently landed there missed and fell through to the address-ordered free
 * list: measured 4095 misses against 6252 hits per run, 60.2 ms/step of
 * arena_alloc against 0.01 ms/step on the LIFO hit path. Widening the hash to
 * 64 bins gives the three sizes separate homes; the tables are 1 KB per
 * arena. */
#define CWICK_HOT_BINS 64u
#define CWICK_HOT_MAX  (64ull << 20)   /* bigger blocks are not churned */

typedef struct cwick_fb {
    size_t            off;
    size_t            len;
    struct cwick_fb   *next;
} cwick_fb;

typedef struct {
    uint8_t          *base;
    size_t            len;
    size_t            bump;      /* high-water mark of never-freed region */
    cwick_fb          *freelist;  /* address-ordered, coalesced on free    */
    cwick_fb          *hot[CWICK_HOT_BINS];  /* exact-size LIFOs             */
    size_t            hot_len[CWICK_HOT_BINS];
    id<MTLBuffer>     buf;
} cwick_arena;

static cwick_arena *g_arenas;
static int         g_arena_n;
static int         g_arena_cap;
static int         g_hint = -1;  /* last arena answered a view lookup */
static size_t      g_arena_bytes;

/* Wiring control + accounting (see arena_grow for the rationale). */
static int         g_nowire;         /* CWICK_METAL_NOWIRE=1              */
static int         g_wire_failed;    /* warn once, not once per arena    */
static size_t      g_wired_bytes;

/* Wall clock for the optional CWICK_METAL_TIME breakdown. */
static double cwick_metal_now_s(void) {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e9;
}

/* Threads per threadgroup for the GEMV kernel (CWICK_METAL_TPT, 32..1024). */
static NSUInteger g_tpt;

/* Persistent scratch used when a weight pointer is not arena-backed. */
static id<MTLBuffer> g_scratch_w, g_scratch_s, g_bx, g_bo;
static size_t        g_scratch_w_cap, g_scratch_s_cap, g_bx_cap, g_bo_cap;

/* Pick the expert GEMV kernel and dispatch it.
 *   v3: rows sliced across all 8 simdgroups (256 threadgroups for an expert
 *       matrix) + threadgroup-memory LUT decode. Needs tpt=256, k % 4096 == 0
 *       and cols % 8 == 0 -- that is every routed gate/up matrix. Default.
 *   v4: the same slicing with the LUT replaced by arithmetic decoding, and
 *       extended to k=2048 so the down projections get it too. Measured
 *       *slower* than v3 on both waves (1.396 -> 1.657 ms for gate/up,
 *       0.799 -> 0.851 ms for down): a bit-exact E4M3 decode is ~13 integer
 *       ops, and a shared-memory gather with its ~4-way bank conflict is
 *       cheaper than that on this GPU. Kept, opt-in, so nobody re-derives it.
 *   v2: 4 rows per simdgroup, no slicing. k % 512 == 0, cols % 32 == 0 -- the
 *       down matrices (k=2048) and anything v3/v4 decline.
 *   v1: original branchy kernel, kept for shapes neither covers.
 * CWICK_FP8_KERNEL=v1|v2|v3|v3d|v4 forces a specific one. */
static void fp8_dispatch(id<MTLComputeCommandEncoder> enc, uint32_t k,
                         uint32_t cols, uint32_t n_rows) {
    NSUInteger tpt = g_tpt ? g_tpt : 256;
    if (tpt < 32) tpt = 32;
    if (tpt > 1024) tpt = 1024;
    NSUInteger nsg = tpt / 32;
    if (nsg > cols) nsg = cols;
    tpt = nsg * 32;

    int want = g_fp8_kernel;
    id<MTLComputePipelineState> pso = NULL;
    NSUInteger gx = 0;
    if (want == 5 && g_pso_v4 && tpt == 256 && (cols % 8u) == 0u &&
        (k == 4096u || k == 2048u)) {
        pso = g_pso_v4; gx = (NSUInteger)(cols / 8u);
    } else if (want == 3 && g_pso_v3 && tpt == 256 && (k % 4096u) == 0u &&
        (cols % 8u) == 0u) {
        pso = g_pso_v3; gx = (NSUInteger)(cols / 8u);
    } else if (want == 4 && g_pso_v3d && tpt == 256 && (k % 4096u) == 0u &&
               (cols % 8u) == 0u) {
        pso = g_pso_v3d; gx = (NSUInteger)(cols / 8u);
    } else if ((want == 2 || want == 3 || want == 4 || want == 5) && g_pso_v2 &&
               (k % 512u) == 0u && (cols % (nsg * 4u)) == 0u) {
        pso = g_pso_v2; gx = (NSUInteger)(cols / (nsg * 4u));
    } else {
        pso = g_pso; gx = (NSUInteger)((cols + nsg - 1) / nsg);
    }
    [enc setComputePipelineState:pso];
    [enc dispatchThreadgroups:MTLSizeMake(gx, n_rows, 1)
           threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
}

static size_t round_up(size_t n, size_t a) { return (n + a - 1) & ~(a - 1); }

static int arena_grow(size_t need);

/* Compile the embedded shader once; returns 0 on any failure. */
static int cwick_metal_compile(void) {
    @autoreleasepool {
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) return 0;
        /* Report the working set the driver actually offers. This is the
         * number `iogpu.wired_limit_mb` moves, and it is sampled here -- once
         * -- so raising that sysctl after this point cannot help this
         * process. Seeing it in the log is what turns the memory-plan
         * percentage into something checkable. */
        CWICK_INFO("mem", "Metal recommendedMaxWorkingSetSize %.2f GiB "
                         "(%.2f GiB unified memory installed)",
                  (double)g_dev.recommendedMaxWorkingSetSize / 1073741824.0,
                  (double)[[NSProcessInfo processInfo] physicalMemory]
                      / 1073741824.0);
        /* Settle the wired-ceiling question deferred from startup: now that
         * the driver has stated its working set, say whether it covers the
         * memory plan or whether a sysctl override is genuinely needed. */
        cwick_mem_report_gpu_wired(g_dev.recommendedMaxWorkingSetSize);
        g_queue = [g_dev newCommandQueueWithMaxCommandBufferCount:64];
        if (!g_queue) return 0;

        NSError *err = nil;
        id<MTLLibrary> lib = [g_dev newLibraryWithSource:
            [NSString stringWithUTF8String:cwick_metal_shader_source()]
            options:nil error:&err];
        if (!lib) {
            fprintf(stderr, "cwick: metal shader compile failed: %s\n",
                    err ? err.localizedDescription.UTF8String : "unknown");
            return 0;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"kernel_cwick_fp8_gemv"];
        if (!fn) return 0;
        g_pso = [g_dev newComputePipelineStateWithFunction:fn error:&err];
        if (!g_pso) return 0;
        id<MTLFunction> v2fn = [lib newFunctionWithName:@"kernel_cwick_fp8_gemv_v2"];
        if (v2fn) {
            g_pso_v2 = [g_dev newComputePipelineStateWithFunction:v2fn error:&err];
        }
        id<MTLFunction> v3fn = [lib newFunctionWithName:@"kernel_cwick_fp8_gemv_v3"];
        if (v3fn) {
            g_pso_v3 = [g_dev newComputePipelineStateWithFunction:v3fn error:&err];
        }
        id<MTLFunction> v3dfn = [lib newFunctionWithName:@"kernel_cwick_fp8_gemv_v3d"];
        if (v3dfn) {
            g_pso_v3d = [g_dev newComputePipelineStateWithFunction:v3dfn error:&err];
        }
        id<MTLFunction> v4fn = [lib newFunctionWithName:@"kernel_cwick_fp8_gemv_v4"];
        if (v4fn) {
            g_pso_v4 = [g_dev newComputePipelineStateWithFunction:v4fn error:&err];
        }
        const char *kv = getenv("CWICK_FP8_KERNEL");
        if (kv) {
            if (strcmp(kv, "v1") == 0) g_fp8_kernel = 1;
            else if (strcmp(kv, "v2") == 0) g_fp8_kernel = 2;
            else if (strcmp(kv, "v3") == 0) g_fp8_kernel = 3;
            else if (strcmp(kv, "v3d") == 0) g_fp8_kernel = 4;
            else if (strcmp(kv, "v4") == 0) g_fp8_kernel = 5;
        }
        id<MTLFunction> bf16_fn = [lib newFunctionWithName:@"kernel_cwick_bf16_gemv"];
        if (!bf16_fn) return 0;
        g_pso_bf16 = [g_dev newComputePipelineStateWithFunction:bf16_fn error:&err];
        if (!g_pso_bf16) return 0;
        id<MTLFunction> dsfp4_fn = [lib newFunctionWithName:@"kernel_cwick_ds_fp4_gemv"];
        if (dsfp4_fn)
            g_pso_ds_fp4 = [g_dev newComputePipelineStateWithFunction:dsfp4_fn
                                                                error:&err];
        id<MTLFunction> dsbf16_fn = [lib newFunctionWithName:@"kernel_cwick_ds_bf16_gemv"];
        if (dsbf16_fn)
            g_pso_ds_bf16 = [g_dev newComputePipelineStateWithFunction:dsbf16_fn
                                                                 error:&err];
        id<MTLFunction> ka_fn = [lib newFunctionWithName:@"kernel_cwick_keepalive"];
        if (ka_fn) {
            g_pso_ka = [g_dev newComputePipelineStateWithFunction:ka_fn error:&err];
            if (g_pso_ka)
                g_ka_buf = [g_dev newBufferWithLength:256
                                             options:MTLResourceStorageModeShared];
        }

        const char *t = getenv("CWICK_METAL_TPT");
        if (t) {
            long v = strtol(t, NULL, 10);
            if (v >= 32 && v <= 1024) g_tpt = (NSUInteger)((v / 32) * 32);
        }
    }
    return 1;
}

/* Grow the arena array by one shared buffer big enough for `need` bytes.
 * Arena size doubles each time (capped) so a multi-GiB expert cache needs a
 * handful of buffers instead of thousands. */
static int arena_grow(size_t need) {
    size_t want = CWICK_ARENA_MIN;
    if (g_arena_n > 0) {
        size_t last = g_arenas[g_arena_n - 1].len;
        if (last * 2 > want) want = last * 2;
    }
    if (want < need) want = need;
    /* The 2 GiB per-buffer ceiling exists so a multi-GiB cache is a handful
     * of buffers rather than one giant one. A single request larger than the
     * ceiling still has to be served by one buffer -- clamping it here made
     * any allocation above ~5.75 GiB fail outright (the doubling sequence
     * 256M,512M,1G,2G,2G can never reach 6 GiB), which is how the arena
     * allocator silently refused large GPU-visible reservations. */
    if (want > CWICK_ARENA_MAX && need <= CWICK_ARENA_MAX) want = CWICK_ARENA_MAX;
    if (want < need) return 0;
    want = round_up(want, CWICK_ARENA_PAGE);

    id<MTLBuffer> buf = [g_dev newBufferWithLength:want
                                           options:MTLResourceStorageModeShared];
    if (!buf || !buf.contents) return 0;

    /* ---- wire it ----------------------------------------------------
     * A shared MTLBuffer is ordinary pageable RAM that happens to be GPU-
     * addressable. When the VM compresses or reclaims those pages the CPU
     * pays a fault of a few microseconds; the GPU pays milliseconds, and it
     * shows up inside waitUntilCompleted, not in GPUEndTime, so it looks
     * like "the GPU is slow" when it is really the compressor.
     *
     * Measured on this machine with a 47.9 GiB expert cache: 65 ms average
     * and 3.6 s worst-case commit+wait against 109 ms of actual GPU work per
     * step. Shrinking the cache to 8 GiB removed the stalls (2018 us
     * average) and cut the step from 6691 ms to 955 ms -- more streaming,
     * far less wall time.
     *
     * mlock() keeps the pages resident and incompressible, so the large
     * cache keeps its hit rate without the faults. CWICK_METAL_NOWIRE=1
     * restores the old behaviour for A/B.
     *
     * Note that mlock only locks pages that are already resident, and a
     * freshly allocated MTLBuffer has none mapped -- so this wires nothing
     * until Metal touches the pages. Pre-touching with a memset was tried
     * (to make the wire real) and measured SLOWER: 48-token decode went
     * 762 -> 838 ms/step, because faulting 16 GiB in costs ~5 s one-off
     * while the steady-state read rate does not improve. The per-read rate
     * is ~1.7 GB/s and twelve of them run concurrently for ~10.5 GB/s
     * aggregate, which is the SSD's random-read ceiling for this checkpoint
     * -- not a residency problem. */
    if (!g_nowire) {
        int w = -1;
        @autoreleasepool {
            w = mlock(buf.contents, want);
        }
        if (w != 0) {
            if (!g_wire_failed) {
                g_wire_failed = 1;
                fprintf(stderr,
                        "cwick: warning: could not wire %zu MiB of GPU-visible "
                        "memory (%s); expert cache pages may be compressed and "
                        "the GPU will stall on them\n",
                        want >> 20, strerror(errno));
            }
        } else {
            g_wired_bytes += want;
        }
    }

    if (g_arena_n == g_arena_cap) {
        int cap = g_arena_cap ? g_arena_cap * 2 : 8;
        cwick_arena *na = realloc(g_arenas, (size_t)cap * sizeof(cwick_arena));
        if (!na) return 0;
        g_arenas = na;
        g_arena_cap = cap;
    }
    cwick_arena *a = &g_arenas[g_arena_n];
    /* ARC releases the previous value of a->buf before storing the new one,
     * and this slot is fresh realloc memory holding garbage -- releasing it
     * is a wild release that shows up far from here (it crashed inside
     * objc_release under cwick_metal_arena_alloc). It only ever looked safe
     * because the allocator happened to hand back zeroed pages. */
    memset((void *)a, 0, sizeof(*a));   /* cast: ARC field, zeroed on purpose */
    a->base = (uint8_t *)buf.contents;
    a->len = want;
    a->bump = 0;
    a->freelist = NULL;
    for (unsigned b = 0; b < CWICK_HOT_BINS; b++) { a->hot[b] = NULL; a->hot_len[b] = 0; }
    a->buf = buf;
    g_arena_n++;
    g_arena_bytes += want;
    return 1;
}

/* Every block carries a 16-byte header holding its own length, so a free can
 * return exactly the bytes that were handed out. Deriving the length from the
 * neighbours instead would release live blocks sitting between the freed one
 * and the arena's high-water mark -- the cache would then hand the same bytes
 * out twice and the GEMV would read the wrong weights. */
#define CWICK_ARENA_HDR 16u

static uint8_t *arena_block(cwick_arena *a, size_t off, size_t need) {
    *((size_t *)(void *)(a->base + off)) = need;
    return a->base + off + CWICK_ARENA_HDR;
}

/* Allocate GPU-visible memory (16-byte aligned). NULL means "use malloc". */
uint64_t g_arena_alloc_ns;   /* prof: time inside arena_alloc */
uint64_t g_arena_free_ns;    /* prof: time inside arena_free  */
uint64_t g_arena_hot_hits;   /* prof: exact-size LIFO hits    */
uint64_t g_arena_hot_misses; /* prof: exact-size LIFO misses  */
uint64_t g_arena_grow_ns;     /* prof: time in the slow paths  */
uint64_t g_arena_grows;       /* prof: arena_grow calls        */
uint64_t g_arena_hot_ns;       /* prof: time in the LIFO hit path */
/* prof: time inside arena_alloc / arena_free, in NANOSECONDS.
 * mach_absolute_time() is NOT nanoseconds on this hardware -- the timebase
 * here is 125/3, so using the raw value under-reports by 41.7x (which is how
 * a 43 ms/step allocator once measured as 1.7 ms). Convert, always. */
static uint64_t arena_ticks(void) {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (uint64_t)((double)mach_absolute_time() * (double)tb.numer /
                      (double)tb.denom);
}
static void *arena_alloc_inner(size_t need);
/* The arena's free list, hot LIFOs and bump pointer are shared mutable state.
 * While there was one generation at a time the only concurrency was the
 * reader pool, which does not allocate; with several slots, two threads can
 * call this at once and a bump-plus-freelist-pop is not atomic. One mutex
 * around alloc and free is the whole cost -- the critical section is a few
 * hundred ns against a ~0.6 ms arena call. */
static pthread_mutex_t g_arena_mtx = PTHREAD_MUTEX_INITIALIZER;

void *cwick_metal_arena_alloc(size_t bytes) {
    if (!g_pso || bytes == 0) return NULL;
    size_t need = round_up(bytes + CWICK_ARENA_HDR, CWICK_ARENA_PAGE);
    if (need < bytes) return NULL;             /* overflow */
    uint64_t _ta = arena_ticks();
    pthread_mutex_lock(&g_arena_mtx);
    void *r = arena_alloc_inner(need);
    pthread_mutex_unlock(&g_arena_mtx);
    __atomic_fetch_add(&g_arena_alloc_ns, arena_ticks() - _ta, __ATOMIC_RELAXED);
    return r;
}

static void *arena_alloc_inner(size_t need) {
    /* Exact-size LIFO first: this is the path ~550 times per decode step. */
    if (need <= CWICK_HOT_MAX) {
        size_t b = (need / CWICK_ARENA_PAGE) & (CWICK_HOT_BINS - 1u);
        uint64_t _th = arena_ticks();
        for (int i = 0; i < g_arena_n; i++) {
            cwick_arena *a = &g_arenas[i];
            if (a->hot_len[b] == need && a->hot[b]) {
                cwick_fb *n = a->hot[b];
                a->hot[b] = n->next;
                size_t off = n->off;
                free(n);
                g_arena_hot_hits++;
                g_arena_hot_ns += arena_ticks() - _th;
                return arena_block(a, off, need);
            }
        }
        g_arena_hot_misses++;
    }
    /* first fit across the free lists */
    uint64_t _tg = arena_ticks();
    for (int i = 0; i < g_arena_n; i++) {
        cwick_fb **pp = &g_arenas[i].freelist;
        while (*pp) {
            if ((*pp)->len >= need) {
                size_t off = (*pp)->off;
                (*pp)->off += need;
                (*pp)->len -= need;
                if ((*pp)->len == 0) {
                    cwick_fb *dead = *pp;
                    *pp = dead->next;
                    free(dead);
                }
                return arena_block(&g_arenas[i], off, need);
            }
            pp = &(*pp)->next;
        }
    }
    /* bump from an arena that still has room */
    for (int i = 0; i < g_arena_n; i++) {
        if (g_arenas[i].len - g_arenas[i].bump >= need) {
            size_t off = g_arenas[i].bump;
            g_arenas[i].bump += need;
            return arena_block(&g_arenas[i], off, need);
        }
    }
    g_arena_grow_ns += arena_ticks() - _tg;
    g_arena_grows++;
    if (!arena_grow(need)) return NULL;
    cwick_arena *a = &g_arenas[g_arena_n - 1];
    size_t off = a->bump;
    a->bump += need;
    return arena_block(a, off, need);
}

/* Return a block to its arena's free list, coalescing with neighbours.
 * Returns 1 if the pointer was arena-owned. */
static int arena_free_inner(void *p);
int cwick_metal_arena_free(void *p) {
    if (!p) return 0;
    uint64_t _tf = arena_ticks();
    pthread_mutex_lock(&g_arena_mtx);
    int r = arena_free_inner(p);
    pthread_mutex_unlock(&g_arena_mtx);
    __atomic_fetch_add(&g_arena_free_ns, arena_ticks() - _tf, __ATOMIC_RELAXED);
    return r;
}

static int arena_free_inner(void *p) {
    if (!p) return 0;
    uintptr_t addr = (uintptr_t)p;
    for (int i = 0; i < g_arena_n; i++) {
        cwick_arena *a = &g_arenas[i];
        if (addr < (uintptr_t)a->base || addr >= (uintptr_t)(a->base + a->len))
            continue;
        size_t off = addr - (uintptr_t)a->base;
        if (off < CWICK_ARENA_HDR) return 0;    /* not one of our blocks */
        off -= CWICK_ARENA_HDR;
        size_t len = *((size_t *)(void *)(a->base + off));
        if (len == 0 || off + len > a->len) return 0;

        /* Exact-size LIFO when the block is one the cache churns: O(1), no
         * walk, no coalesce. The block stays out of the general list, which
         * is fine because nothing asks for that exact size any other way. */
        if (len <= CWICK_HOT_MAX) {
            size_t b = (len / CWICK_ARENA_PAGE) & (CWICK_HOT_BINS - 1u);
            if (a->hot_len[b] == 0 || a->hot_len[b] == len) {
                cwick_fb *node = malloc(sizeof(cwick_fb));
                if (node) {
                    node->off = off;
                    node->len = len;
                    node->next = a->hot[b];
                    a->hot[b] = node;
                    a->hot_len[b] = len;
                    return 1;
                }
            }
        }

        cwick_fb **pp = &a->freelist;
        cwick_fb  *prev = NULL;
        while (*pp && (*pp)->off < off) { prev = *pp; pp = &(*pp)->next; }
        cwick_fb *node = malloc(sizeof(cwick_fb));
        if (!node) return 1;                   /* leak the block; stay safe */
        node->off = off;
        node->len = len;
        node->next = *pp;
        *pp = node;
        /* coalesce with the following block, then with the preceding one */
        while (node->next && node->off + node->len == node->next->off) {
            cwick_fb *dead = node->next;
            node->len += dead->len;
            node->next = dead->next;
            free(dead);
        }
        if (prev && prev->off + prev->len == node->off) {
            prev->len += node->len;
            prev->next = node->next;
            free(node);
        }
        return 1;
    }
    return 0;
}

size_t cwick_metal_arena_bytes(void) { return g_arena_bytes; }

/* ------------------------------------------------------------------ */
/* No-copy views over mapped (resident) weights                        */
/* ------------------------------------------------------------------ */
/* The resident spine is mmap'd and its pages are already unified memory
 * the GPU can read, so a BF16 projection can be bound with
 * newBufferWithBytesNoCopy instead of uploading 64-134 MiB. But creating
 * that buffer is expensive, and doing it *per call* cost more than the
 * scalar GEMV it was meant to replace (measured: the whole attention phase
 * stayed at ~2.5 s/step even with the backend up). So the mapped model is
 * wrapped once at load and the result cached, so each tensor pays the wrap
 * exactly once per process. */
typedef struct {
    uintptr_t        addr;     /* page-aligned base of the mapping window */
    size_t           len;
    size_t           lead;     /* bytes from addr to the tensor's data */
    id<MTLBuffer>    buf;
} cwick_view;

static cwick_view *g_views;
static int        g_view_n, g_view_cap;
static int        g_view_hint = -1;

/* Find or create a cached no-copy view covering [p, p+len).
 *
 * CONTRACT: *buf must be initialised to nil by the caller. These helpers
 * store through an `id<MTLBuffer> __strong *` out-param, and a strong store
 * RELEASES the value already there. ARC cannot tell that a stack slot the
 * caller never initialised holds garbage (it cannot see through the
 * pointer), so an uninitialised caller slot means a wild objc_release --
 * which showed up as a rare EXC_BAD_ACCESS inside objc_release, roughly one
 * run in five, at the first bf16 or fp8 GEMV of the process. Every call
 * site below now passes a nil-initialised buffer; keep it that way. */
static int view_of(const void *p, size_t len, id<MTLBuffer> __strong *buf,
                   size_t *off) {
    uintptr_t raw = (uintptr_t)p;
    const uintptr_t page = 4096u;
    uintptr_t base = raw & ~(page - 1u);
    size_t lead = (size_t)(raw - base);
    /* newBufferWithBytesNoCopy requires a page-multiple length; handing it a
     * ragged size makes the wrap unreliable (and the GPU then reads past the
     * end of what was adopted). */
    size_t need = ((len + lead + page - 1u) / page) * page;

    /* Prefault the region before handing it to the GPU. A no-copy buffer
     * requires its backing pages to be resident; a weight that only the GPU
     * ever reads (the per-layer shared expert, since its GEMVs moved into
     * the batch) may never have been touched by the CPU, and the GPU page
     * table then lacks the mapping -- which shows up as a rare, timing
     * dependent SIGSEGV inside the kernel rather than a clean error. One
     * volatile read per page, once per wrapped region, costs microseconds
     * against the matrix load and is the documented no-copy requirement. */
    {
        volatile const unsigned char *vp = (volatile const unsigned char *)base;
        unsigned char sink = 0;
        for (size_t o = 0; o < need; o += (size_t)page) sink |= vp[o];
        (void)sink;
    }

    if (g_view_hint >= 0 && g_view_hint < g_view_n) {
        cwick_view *v = &g_views[g_view_hint];
        if (raw >= v->addr && raw + len <= v->addr + v->len) {
            *buf = v->buf; *off = raw - v->addr;
            return 1;
        }
    }
    for (int i = 0; i < g_view_n; i++) {
        cwick_view *v = &g_views[i];
        if (raw >= v->addr && raw + len <= v->addr + v->len) {
            *buf = v->buf; *off = raw - v->addr;
            g_view_hint = i;
            return 1;
        }
    }
    /* Not wrapped yet: do it once and keep it. */
    id<MTLBuffer> b = [g_dev newBufferWithBytesNoCopy:(void *)base
                                              length:need
                                             options:MTLResourceStorageModeShared
                                         deallocator:nil];
    if (!b) return 0;
    if (g_view_n == g_view_cap) {
        int cap = g_view_cap ? g_view_cap * 2 : 64;
        cwick_view *nv = realloc(g_views, (size_t)cap * sizeof(cwick_view));
        if (!nv) return 0;
        g_views = nv;
        g_view_cap = cap;
    }
    cwick_view *v = &g_views[g_view_n];
    /* ARC releases the previous value of v->buf before storing the new one,
     * and a slot past the old g_view_n is fresh realloc memory holding
     * garbage -- releasing it is a wild release. This is the same crash the
     * arena hit (see the memset in arena_grow): it surfaced as SIGBUS/SIGSEGV
     * inside objc_release under cwick_metal_gemv_bf16 during prefill, and it
     * only fired once the view cache outgrew its first 64 entries, which is
     * why short prompts looked fine. Zero the slot on purpose. */
    memset((void *)v, 0, sizeof(*v));   /* cast: ARC field, zeroed on purpose */
    v->addr = base; v->len = need; v->lead = lead; v->buf = b;
    g_view_hint = g_view_n;
    g_view_n++;
    *buf = b; *off = lead;
    return 1;
}

/* Does [p, p+len) live inside one of our arenas? */
static int arena_view(const void *p, size_t len, id<MTLBuffer> __strong *buf, size_t *off) {
    uintptr_t addr = (uintptr_t)p;
    if (g_hint >= 0 && g_hint < g_arena_n) {
        cwick_arena *a = &g_arenas[g_hint];
        if (addr >= (uintptr_t)a->base && addr + len <= (uintptr_t)(a->base + a->len)) {
            *buf = a->buf;
            *off = addr - (uintptr_t)a->base;
            return 1;
        }
    }
    for (int i = 0; i < g_arena_n; i++) {
        cwick_arena *a = &g_arenas[i];
        if (addr >= (uintptr_t)a->base && addr + len <= (uintptr_t)(a->base + a->len)) {
            *buf = a->buf;
            *off = addr - (uintptr_t)a->base;
            g_hint = i;
            return 1;
        }
    }
    return 0;
}

/* Grow-or-return a persistent scratch buffer. */
static id<MTLBuffer> scratch(id<MTLBuffer> __strong *slot, size_t *cap, size_t need) {
    if (*slot && *cap >= need) return *slot;
    size_t want = *cap ? *cap * 2 : round_up(need, CWICK_ARENA_PAGE);
    if (want < need) want = need;
    id<MTLBuffer> b = [g_dev newBufferWithLength:want
                                         options:MTLResourceStorageModeShared];
    if (!b) return nil;
    *slot = b;
    *cap = want;
    return b;
}

/* ------------------------------------------------------------------ */
/* Batched GEMV                                                        */
/* ------------------------------------------------------------------ */
/* One dispatch per expert matrix is launch-bound, not bandwidth-bound.
 * CWICK_METAL_TIME on this machine reports the split directly:
 *
 *   T enc=5-28us commit=3-11us wait=295-724us gpu=105-123us
 *
 * so ~105 us of GPU work is wrapped in ~330-570 us of round trip, and a
 * decode step issues 1080 of them -- the launch overhead alone is the GEMV
 * phase.
 *
 * The routed group has exactly two waves: gate and up are independent, down
 * consumes both. Encoding one wave into a single command buffer turns 24
 * round trips per layer into 2, and 1080 per step into 84. Same kernels,
 * same arithmetic, same answers. */

#define CWICK_ARGS_STRIDE 256u   /* Metal buffer-binding alignment, roomy */

typedef struct {
    id<MTLBuffer> b;
    size_t        off;
    void         *dst;
    size_t        bytes;
} cwick_bcopy;

struct cwick_batch_s {
    id<MTLCommandBuffer>           cb;
    id<MTLComputeCommandEncoder>   enc;
    id<MTLBuffer>                  args;     /* cwick_metal_gemv_args array */
    NSUInteger                     args_cap; /* in entries                */
    NSUInteger                     args_n;
    id<MTLBuffer>                  bx, bo;   /* x and out staging         */
    size_t                         bx_cap, bo_cap;
    size_t                         bx_used, bo_used;
    cwick_bcopy                    *cp;
    int                            ncp, cpcap;
    int                            bad;
};

/* One batch per generation slot.
 *
 * This was a single static with the comment "reused: decode is sequential",
 * and that was true while the server ran one generation at a time. With more
 * than one slot, two threads would take the same struct: each `begin` resets
 * bx_used/args_n and each `add` appends, so the second one would silently
 * truncate the first one's dispatch list and encode a command buffer over
 * buffer offsets that are being rewritten underneath it.
 *
 * Concurrency was measured to be worth it (two independent server processes
 * served two requests in 226 s against 334 s serially,
 * 1.48x), so the batch is now indexed by slot. `slot` is clamped, which keeps
 * every existing single-slot caller working unchanged. */
#define CWICK_METAL_BATCH_SLOTS 4
static struct cwick_batch_s *g_batch[CWICK_METAL_BATCH_SLOTS];

static int batch_reserve(struct cwick_batch_s *b, size_t xb, size_t ob,
                         size_t *xoff, size_t *ooff) {
    *xoff = ((b->bx_used + 63u) & ~(size_t)63u);
    *ooff = ((b->bo_used + 63u) & ~(size_t)63u);
    size_t xneed = *xoff + xb, oneed = *ooff + ob;
    if (!b->bx || b->bx_cap < xneed) {
        size_t want = b->bx_cap ? b->bx_cap * 2 : (1u << 20);
        if (want < xneed) want = xneed;
        id<MTLBuffer> nb = [g_dev newBufferWithLength:want
                                             options:MTLResourceStorageModeShared];
        if (!nb) return 0;
        b->bx = nb; b->bx_cap = want;
    }
    if (!b->bo || b->bo_cap < oneed) {
        size_t want = b->bo_cap ? b->bo_cap * 2 : (1u << 20);
        if (want < oneed) want = oneed;
        id<MTLBuffer> nb = [g_dev newBufferWithLength:want
                                             options:MTLResourceStorageModeShared];
        if (!nb) return 0;
        b->bo = nb; b->bo_cap = want;
    }
    b->bx_used = xneed; b->bo_used = oneed;
    return 1;
}

static int cwick_metal_batch_timing(void) {
    static int c = -1;
    if (c < 0) c = getenv("CWICK_METAL_BATCH_TIME") != NULL ? 1 : 0;
    return c;
}


void *cwick_metal_batch_begin_bridge(int slot) {
    if (!g_pso || !g_queue) return NULL;
    if (slot < 0) slot = 0;
    if (slot >= (int)CWICK_METAL_BATCH_SLOTS) slot = CWICK_METAL_BATCH_SLOTS - 1;
    if (!g_batch[slot]) {
        g_batch[slot] = calloc(1, sizeof(struct cwick_batch_s));
        if (!g_batch[slot]) return NULL;
    }
    struct cwick_batch_s *b = g_batch[slot];
    b->cb = nil; b->enc = nil;
    b->args_n = 0; b->bx_used = 0; b->bo_used = 0;
    b->ncp = 0; b->bad = 0;
    return b;
}


/* One-shot diagnostics for the batch bridge. When a window is refused every
 * layer of it falls back to the CPU group, and the profile only says
 * "batch->cpu" -- which does not name the precondition that failed. Six lines
 * per process, only on refusal, so this is free in the normal case. */
static int g_mdiag_left = 6;
static void mdiag(const char *why, const void *p, size_t n) {
    if (g_mdiag_left <= 0) return;
    g_mdiag_left--;
    fprintf(stderr, "cwick: metal batch refused: %s ptr=%p len=%zu\n", why, p, n);
}

int cwick_metal_batch_gemv_bridge(void *vb, const void *w_fp8, const float *scale_inv,
                          const float *x, float *out,
                          uint32_t k, uint32_t cols, uint32_t ncb,
                          uint32_t n_rows, uint32_t stride) {
    struct cwick_batch_s *b = (struct cwick_batch_s *)vb;
    if (!b || b->bad || !w_fp8 || !x || !out || !k || !cols || !ncb) return 0;
    if (k % 128u != 0u) { b->bad = 1; return 0; }

    const size_t wbytes = (size_t)cols * k;
    const size_t sbytes = (size_t)(cols / 128u) * ncb * sizeof(float);
    const size_t xbytes = (size_t)n_rows * stride * sizeof(float);
    const size_t obytes = (size_t)n_rows * cols * sizeof(float);

    /* nil first: arena_view/view_of write through an `__strong *` out-param,
     * which RELEASES the old value before storing. An uninitialised stack
     * slot therefore releases garbage -- the wild objc_release that crashed
     * roughly one run in five. ARC does not track init state through a
     * pointer, so the caller has to nil it. */
    id<MTLBuffer> bw = nil, bs = nil;
    size_t woff = 0, soff = 0;
    /* The streamed expert is already in a GPU-visible arena; a cache hit is
     * too. Resident weights (the per-layer shared expert, for one) live in
     * the mmap'd spine instead, and Metal can read those pages directly as
     * well -- `view_of` wraps them with newBufferWithBytesNoCopy, so they
     * ride the batch with no copy. Only a pointer that is neither arena- nor
     * mapping-backed is refused, because that would cost an 8 MiB memcpy per
     * matrix -- which is what made the unbatched path slow.
     *
     * Refusing the mapped case was why the shared expert's three GEMVs ran on
     * the CPU: ~190 CPU fp8 GEMVs per decode step at ~335 us each. */
    if (!arena_view(w_fp8, wbytes, &bw, &woff) &&
        !view_of(w_fp8, wbytes, &bw, &woff)) {
        mdiag("weight not arena- or mapping-backed", w_fp8, wbytes);
        b->bad = 1;
        return 0;
    }
    if (scale_inv && !arena_view(scale_inv, sbytes, &bs, &soff) &&
        !view_of(scale_inv, sbytes, &bs, &soff)) {
        mdiag("scales not arena- or mapping-backed", scale_inv, sbytes);
        b->bad = 1;
        return 0;
    }

    size_t xoff = 0, ooff = 0;
    if (!batch_reserve(b, xbytes, obytes, &xoff, &ooff)) {
        mdiag("batch_reserve full", (const void *)&xbytes, xbytes + obytes);
        b->bad = 1; return 0;
    }
    memcpy((uint8_t *)b->bx.contents + xoff, x, xbytes);

    if (b->args_n >= b->args_cap) {
        NSUInteger cap = b->args_cap ? b->args_cap * 2 : 32;
        id<MTLBuffer> na = [g_dev newBufferWithLength:(size_t)cap * CWICK_ARGS_STRIDE
                                             options:MTLResourceStorageModeShared];
        if (!na) { b->bad = 1; return 0; }
        b->args = na; b->args_cap = cap;
    }
    cwick_metal_gemv_args *ap =
        (cwick_metal_gemv_args *)((uint8_t *)b->args.contents +
                                 (size_t)b->args_n * CWICK_ARGS_STRIDE);
    ap->k = k; ap->cols = cols; ap->ncb = ncb;
    ap->n_rows = n_rows; ap->stride = stride;
    ap->pad0 = ap->pad1 = ap->pad2 = 0;

    if (!b->cb) {
        b->cb = [g_queue commandBuffer];
        if (!b->cb) { b->bad = 1; return 0; }
        b->enc = cwick_concurrent_encoder(b->cb);
        if (!b->enc) { b->bad = 1; return 0; }
        [b->enc setComputePipelineState:g_pso];
    }
    NSUInteger aoff = (NSUInteger)b->args_n * CWICK_ARGS_STRIDE;
    [b->enc setBuffer:b->args offset:aoff atIndex:0];
    [b->enc setBuffer:bw  offset:woff atIndex:1];
    [b->enc setBuffer:scale_inv ? bs : nil offset:soff atIndex:2];
    [b->enc setBuffer:b->bx offset:xoff atIndex:3];
    [b->enc setBuffer:b->bo offset:ooff atIndex:4];
    fp8_dispatch(b->enc, k, cols, n_rows);

    if (b->ncp == b->cpcap) {
        int cap = b->cpcap ? b->cpcap * 2 : 32;
        cwick_bcopy *nc = realloc(b->cp, (size_t)cap * sizeof(cwick_bcopy));
        if (!nc) { b->bad = 1; return 0; }
        /* Initialise the fresh slots: cwick_bcopy carries an ARC-strong buffer
         * and `b->cp[i].b = ...` below RELEASES the slot's old value. realloc
         * hands back dirty heap for the grown region (and malloc itself on a
         * recycled chunk), so an uninitialised slot is a wild objc_release --
         * the same crash class as the view cache. Written field by field so
         * the compiler sees the ARC store and no memset warning appears. */
        for (int i = b->cpcap; i < cap; i++) {
            nc[i].b = nil;
            nc[i].off = 0; nc[i].dst = NULL; nc[i].bytes = 0;
        }
        b->cp = nc; b->cpcap = cap;
    }
    b->cp[b->ncp].b = b->bo;
    b->cp[b->ncp].off = ooff;
    b->cp[b->ncp].dst = out;
    b->cp[b->ncp].bytes = obytes;
    b->ncp++;
    b->args_n++;
    return 1;
}

int cwick_metal_batch_end_bridge(void *vb) {
    struct cwick_batch_s *b = (struct cwick_batch_s *)vb;
    if (!b) return 0;
    if (b->bad || !b->cb) { mdiag("end: window already bad or no command buffer", vb, 0); b->bad = 1; return 0; }
    double _t0 = 0, _t1 = 0, _t2 = 0;
    if (cwick_metal_batch_timing()) _t0 = cwick_metal_now_s();
    [b->enc endEncoding];
    b->enc = nil;
    if (cwick_metal_batch_timing()) _t1 = cwick_metal_now_s();
    [b->cb commit];
    [b->cb waitUntilCompleted];
    if (cwick_metal_batch_timing()) _t2 = cwick_metal_now_s();
    int ok = (b->cb.status == MTLCommandBufferStatusCompleted && !b->cb.error);
    if (cwick_metal_batch_timing())
        fprintf(stderr, "B n=%d enc=%.1fus commit+wait=%.1fus gpu=%.1fus "
                        "bx=%zu bo=%zu\n",
                b->ncp, (_t1 - _t0) * 1e6, (_t2 - _t1) * 1e6,
                (b->cb.GPUEndTime - b->cb.GPUStartTime) * 1e6,
                b->bx_used, b->bo_used);
    b->cb = nil;
    if (!ok) {
        mdiag(b->cb.error ? "command buffer error" : "command buffer status != completed",
              vb, (size_t)b->ncp);
        b->bad = 1; return 0;
    }
    for (int i = 0; i < b->ncp; i++)
        memcpy(b->cp[i].dst, (uint8_t *)b->bo.contents + b->cp[i].off,
               b->cp[i].bytes);
    return 1;
}

void cwick_metal_batch_abort_bridge(void *vb) {
    struct cwick_batch_s *b = (struct cwick_batch_s *)vb;
    if (!b) return;
    if (b->enc) { [b->enc endEncoding]; b->enc = nil; }
    b->cb = nil;
    b->bad = 1;   /* results are unusable; caller redoes the group on CPU */
}

/* ------------------------------------------------------------------ */
/* The GEMV itself                                                     */
/* ------------------------------------------------------------------ */

/* Run one FP8-E4M3 GEMV on the GPU:
 *   out[c] = sum_j x[j] * W[c][j] * inv_scale[block(c,j)]
 * W is [cols, k] row-major uint8 E4M3; scale_inv is [cols/128, ncb] f32
 * (128 output rows share one scale block, ncb == k/128).
 * Returns 0 unless the whole dispatch completed. */
int cwick_metal_gemv_fp8(const void *w_fp8, const float *scale_inv,
                        const float *x, float *out,
                        uint32_t k, uint32_t cols, uint32_t ncb,
                        uint32_t n_rows, uint32_t stride) {
    if (!g_pso) return 0;

    const size_t wbytes = (size_t)cols * k;             /* E4M3 bytes */
    /* scale_inv layout is [cols/128, ncb] f32 (128 output rows share one
     * scale block), NOT [cols, ncb] -- the on-disk shape is [16, 32] for a
     * 2048-wide expert, i.e. 2 KiB, not 256 KiB. Reading cols*ncb here was
     * the prefill segfault (256 KiB read of a 2 KiB buffer). */
    const size_t sbytes = (size_t)(cols / 128u) * ncb * sizeof(float);
    const size_t xbytes = (size_t)n_rows * stride * sizeof(float);
    const size_t obytes = (size_t)n_rows * cols * sizeof(float);

    @autoreleasepool {
        /* nil first, see the out-param release note in the batch bridge. */
        id<MTLBuffer> bw = nil, bs = nil;
        size_t woff = 0, soff = 0;

        /* Zero-copy fast path: the weights the stream reader just pulled off
         * the SSD are already in an arena the GPU can address. */
        int w_zero = arena_view(w_fp8, wbytes, &bw, &woff);
        int s_zero = w_zero ? arena_view(scale_inv, sbytes, &bs, &soff) : 0;

        if (!w_zero) {
            bw = scratch(&g_scratch_w, &g_scratch_w_cap, wbytes);
            if (!bw) return 0;
            memcpy(bw.contents, w_fp8, wbytes);
        }
        if (!s_zero) {
            bs = scratch(&g_scratch_s, &g_scratch_s_cap, sbytes);
            if (!bs) return 0;
            memcpy(bs.contents, scale_inv, sbytes);
        }

        id<MTLBuffer> bx = scratch(&g_bx, &g_bx_cap, xbytes);
        id<MTLBuffer> bo = scratch(&g_bo, &g_bo_cap, obytes);
        if (!bx || !bo) return 0;
        memcpy(bx.contents, x, xbytes);

        cwick_metal_gemv_args args;
        args.k = k; args.cols = cols; args.ncb = ncb;
        args.n_rows = n_rows; args.stride = stride;
        args.pad0 = args.pad1 = args.pad2 = 0;

        double _t0 = 0, _t1 = 0, _t2 = 0, _t3 = 0;
        if (getenv("CWICK_METAL_TIME")) _t0 = cwick_metal_now_s();
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:bw offset:woff atIndex:1];   /* weights (E4M3 bytes) */
        [enc setBuffer:bs offset:soff atIndex:2];   /* scale_inv */
        [enc setBuffer:bx offset:0 atIndex:3];      /* x */
        [enc setBuffer:bo offset:0 atIndex:4];      /* out */

        fp8_dispatch(enc, k, cols, n_rows);
        [enc endEncoding];
        if (getenv("CWICK_METAL_TIME")) _t1 = cwick_metal_now_s();
        [cb commit];
        if (getenv("CWICK_METAL_TIME")) _t2 = cwick_metal_now_s();
        [cb waitUntilCompleted];
        if (getenv("CWICK_METAL_TIME")) _t3 = cwick_metal_now_s();
        if (cb.status != MTLCommandBufferStatusCompleted || cb.error) return 0;

        memcpy(out, bo.contents, obytes);
        if (getenv("CWICK_METAL_TIME")) {
            fprintf(stderr, "T enc=%.1fus commit=%.1fus wait=%.1fus gpu=%.1fus\n",
                    (_t1-_t0)*1e6, (_t2-_t1)*1e6, (_t3-_t2)*1e6,
                    (cb.GPUEndTime - cb.GPUStartTime)*1e6);
        }
    }
    return 1;
}

/* BF16 linear GEMV for resident/mapped weights. The old CPU path dequoted a
 * whole [rows, k] matrix to fp32 before every SGEMV; this kernel reads the
 * checkpoint's BF16 directly and shares the same unified-memory pages. */
int cwick_metal_gemv_bf16(const void *w_bf16, const float *x, float *out,
                         uint32_t k, uint32_t cols) {
    if (!g_pso_bf16 || !w_bf16 || !x || !out || !k || !cols) return 0;
    const size_t wbytes = (size_t)cols * k * sizeof(uint16_t);
    const size_t xbytes = (size_t)k * sizeof(float);
    const size_t obytes = (size_t)cols * sizeof(float);

    @autoreleasepool {
        /* nil first: arena_view/view_of release the out-param's old value. */
        id<MTLBuffer> bw = nil;
        size_t woff = 0;
        if (!arena_view(w_bf16, wbytes, &bw, &woff) &&
            !view_of(w_bf16, wbytes, &bw, &woff)) {
            /* Last resort (a mapping Metal refuses to adopt): one scratch
             * upload. Slow, so it is worth knowing about -- CWICK_METAL_TIME
             * reports the per-dispatch breakdown. */
            bw = scratch(&g_scratch_w, &g_scratch_w_cap, wbytes);
            if (!bw) return 0;
            memcpy(bw.contents, w_bf16, wbytes);
        }
        id<MTLBuffer> bx = scratch(&g_bx, &g_bx_cap, xbytes);
        id<MTLBuffer> bo = scratch(&g_bo, &g_bo_cap, obytes);
        if (!bx || !bo) return 0;
        memcpy(bx.contents, x, xbytes);

        NSUInteger tpt = g_tpt ? g_tpt : 256;
        if (tpt < 32) tpt = 32;
        if (tpt > 1024) tpt = 1024;
        NSUInteger nsg = tpt / 32;
        if (nsg > cols) nsg = cols;
        if (nsg < 1) nsg = 1;
        tpt = nsg * 32;
        /* Each simdgroup computes one distinct output row, so the grid is
         * ceil(cols / nsg) threadgroups and every row is covered exactly
         * once. The kernel used to take its row from gid.y alone, which
         * computed only every nsg-th row and left the rest of `out` holding
         * whatever the staging buffer had in it. */
        cwick_metal_bf16_args args = { k, cols, (uint32_t)nsg, 0, 0 };
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_pso_bf16];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:bw offset:woff atIndex:1];
        [enc setBuffer:bx offset:0 atIndex:2];
        [enc setBuffer:bo offset:0 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, (cols + nsg - 1) / nsg, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        double _t0 = 0, _t1 = 0, _t2 = 0;
        if (getenv("CWICK_METAL_TIME")) _t0 = cwick_metal_now_s();
        [cb commit];
        if (getenv("CWICK_METAL_TIME")) _t1 = cwick_metal_now_s();
        [cb waitUntilCompleted];
        if (getenv("CWICK_METAL_TIME")) _t2 = cwick_metal_now_s();
        if (cb.status != MTLCommandBufferStatusCompleted || cb.error) return 0;
        memcpy(out, bo.contents, obytes);
        if (getenv("CWICK_METAL_TIME"))
            fprintf(stderr, "Tb enc=%.1fus commit=%.1fus wait=%.1fus gpu=%.1fus k=%u cols=%u\n",
                    (_t1-_t0)*1e6, (_t2-_t1)*1e6,
                    (_t2-_t1)*1e6 - (cb.GPUEndTime - cb.GPUStartTime)*1e6,
                    (cb.GPUEndTime - cb.GPUStartTime)*1e6, k, cols);
    }
    return 1;
}

/* ---- batched BF16 GEMV ---------------------------------------------
 * The resident BF16 projections (KDA q/k/v, MLA q_a/kv_a) are big enough
 * that the GPU beats the CPU pool by ~2.4x (311 us vs 745 us for a
 * [8192,4096] matrix), but a commit+wait round trip costs ~450 us of
 * scheduling on top, so issuing them one at a time loses. Grouping the
 * independent projections of one sublayer into a single command buffer
 * pays the round trip once and lets the GPU work overlap inside it.
 *
 * Deliberately smaller in scope than the FP8 batch: no scale tensors, no
 * two-wave dependency tracking, and a fixed item array -- the callers only
 * ever group a handful of projections that share one input. Anything the
 * batch cannot serve (a non-arena, non-view weight pointer) marks it bad
 * and the caller redoes that group on the serial path, so a mistake costs
 * a fallback, not a wrong answer. */
#define CWICK_BF16_BATCH_MAX 8

struct cwick_bf16_batch_s {
    id<MTLCommandBuffer>         cb;
    id<MTLComputeCommandEncoder> enc;
    id<MTLBuffer>                args;      /* cwick_metal_bf16_args array */
    NSUInteger                   args_cap, args_n;
    id<MTLBuffer>                bx, bo;    /* x and out staging */
    size_t                       bx_cap, bo_cap, bx_used, bo_used;
    int                          n, bad;
    struct {
        /* The views outlive the batch (g_views / the arenas hold them for
         * the process), so an unretained reference is safe here and keeps
         * the ARC-out-of-malloc'd-memory hazard away. */
        __unsafe_unretained id<MTLBuffer> b;
        size_t                  woff;
        size_t                  xoff;
        size_t                  ooff;
        void                   *dst;
        size_t                  obytes;
        uint32_t                k, cols, nsg;
    } it[CWICK_BF16_BATCH_MAX];
};

/* Per slot, for the same reason as the FP8 batch: two concurrent generations
 * sharing one struct would each reset it and encode over the other's offsets. */
#define CWICK_METAL_BF16_BATCH_SLOTS 4
static struct cwick_bf16_batch_s *g_bf16_batch[CWICK_METAL_BF16_BATCH_SLOTS];

void *cwick_metal_bf16_batch_begin_bridge(int slot) {
    if (!g_pso_bf16 || !g_queue) return NULL;
    if (slot < 0) slot = 0;
    if (slot >= (int)CWICK_METAL_BF16_BATCH_SLOTS)
        slot = CWICK_METAL_BF16_BATCH_SLOTS - 1;
    if (!g_bf16_batch[slot]) {
        g_bf16_batch[slot] = calloc(1, sizeof(struct cwick_bf16_batch_s));
        if (!g_bf16_batch[slot]) return NULL;
    }
    struct cwick_bf16_batch_s *b = g_bf16_batch[slot];
    b->cb = nil; b->enc = nil;
    b->args_n = 0; b->bx_used = 0; b->bo_used = 0;
    b->n = 0; b->bad = 0;
    return b;
}

int cwick_metal_bf16_batch_add_bridge(void *vb, const void *w_bf16,
                                     const float *x, float *out,
                                     uint32_t k, uint32_t cols) {
    struct cwick_bf16_batch_s *b = (struct cwick_bf16_batch_s *)vb;
    if (!b || b->bad || !w_bf16 || !x || !out || !k || !cols) return 0;
    if (k % 8u != 0u || ((uintptr_t)w_bf16 & 7u) != 0u) { b->bad = 1; return 0; }
    if (b->n >= CWICK_BF16_BATCH_MAX) { b->bad = 1; return 0; }

    const size_t wbytes = (size_t)cols * k * sizeof(uint16_t);
    const size_t xbytes = (size_t)k * sizeof(float);
    const size_t obytes = (size_t)cols * sizeof(float);

    /* nil first: arena_view/view_of release the out-param's old value. */
    id<MTLBuffer> bw = nil;
    size_t woff = 0;
    if (!arena_view(w_bf16, wbytes, &bw, &woff) &&
        !view_of(w_bf16, wbytes, &bw, &woff)) {
        /* A weight Metal cannot adopt without a copy: refuse and let the
         * caller fall back, exactly like the FP8 batch does. */
        b->bad = 1;
        return 0;
    }

    size_t xoff = ((b->bx_used + 63u) & ~(size_t)63u);
    size_t ooff = ((b->bo_used + 63u) & ~(size_t)63u);
    size_t xneed = xoff + xbytes, oneed = ooff + obytes;
    if (!b->bx || b->bx_cap < xneed) {
        size_t want = b->bx_cap ? b->bx_cap * 2 : (1u << 20);
        if (want < xneed) want = xneed;
        id<MTLBuffer> nb = [g_dev newBufferWithLength:want
                                             options:MTLResourceStorageModeShared];
        if (!nb) { b->bad = 1; return 0; }
        b->bx = nb; b->bx_cap = want;
    }
    if (!b->bo || b->bo_cap < oneed) {
        size_t want = b->bo_cap ? b->bo_cap * 2 : (1u << 20);
        if (want < oneed) want = oneed;
        id<MTLBuffer> nb = [g_dev newBufferWithLength:want
                                             options:MTLResourceStorageModeShared];
        if (!nb) { b->bad = 1; return 0; }
        b->bo = nb; b->bo_cap = want;
    }
    b->bx_used = xneed; b->bo_used = oneed;
    memcpy((uint8_t *)b->bx.contents + xoff, x, xbytes);

    NSUInteger nsg = (g_tpt ? g_tpt : 256) / 32;
    if (nsg > cols) nsg = cols;
    if (nsg < 1) nsg = 1;

    b->it[b->n].b = bw;
    b->it[b->n].woff = woff;
    b->it[b->n].xoff = xoff;
    b->it[b->n].ooff = ooff;
    b->it[b->n].dst = out;
    b->it[b->n].obytes = obytes;
    b->it[b->n].k = k;
    b->it[b->n].cols = cols;
    b->it[b->n].nsg = (uint32_t)nsg;
    b->n++;
    return 1;
}

int cwick_metal_bf16_batch_end_bridge(void *vb) {
    struct cwick_bf16_batch_s *b = (struct cwick_bf16_batch_s *)vb;
    if (!b) return 0;
    if (b->bad) { b->bad = 1; return 0; }
    if (b->n == 0) return 1;

    if (b->args_n + (NSUInteger)b->n > b->args_cap || !b->args) {
        NSUInteger cap = b->args_cap ? b->args_cap : 16;
        while (cap < b->args_n + (NSUInteger)b->n) cap *= 2;
        id<MTLBuffer> na = [g_dev newBufferWithLength:(size_t)cap * CWICK_ARGS_STRIDE
                                             options:MTLResourceStorageModeShared];
        if (!na) { b->bad = 1; return 0; }
        b->args = na; b->args_cap = cap;
        b->args_n = 0;   /* the old contents were never committed */
    }

    /* Every item in this batch is a projection of the same `x` into its own
     * output buffer (see the call sites in attention.c), so the dispatches
     * are independent and may overlap. Serialised they measured ~118 GB/s. */
    b->cb = [g_queue commandBuffer];
    if (!b->cb) { b->bad = 1; return 0; }
    b->enc = cwick_concurrent_encoder(b->cb);
    if (!b->enc) { b->bad = 1; return 0; }
    [b->enc setComputePipelineState:g_pso_bf16];

    for (int i = 0; i < b->n; i++) {
        cwick_metal_bf16_args *ap =
            (cwick_metal_bf16_args *)((uint8_t *)b->args.contents +
                                     (size_t)b->args_n * CWICK_ARGS_STRIDE);
        ap->k = b->it[i].k;
        ap->cols = b->it[i].cols;
        ap->rows_per_tg = b->it[i].nsg;
        ap->pad1 = ap->pad2 = 0;
        [b->enc setBuffer:b->args offset:(NSUInteger)b->args_n * CWICK_ARGS_STRIDE
                    atIndex:0];
        [b->enc setBuffer:b->it[i].b offset:b->it[i].woff atIndex:1];
        [b->enc setBuffer:b->bx offset:b->it[i].xoff atIndex:2];
        [b->enc setBuffer:b->bo offset:b->it[i].ooff atIndex:3];
        NSUInteger nsg = b->it[i].nsg;
        [b->enc dispatchThreadgroups:MTLSizeMake(1,
                    (b->it[i].cols + nsg - 1) / nsg, 1)
                threadsPerThreadgroup:MTLSizeMake(nsg * 32, 1, 1)];
        b->args_n++;
    }
    [b->enc endEncoding];
    b->enc = nil;
    double _t0 = 0, _t1 = 0;
    if (getenv("CWICK_METAL_TIME")) _t0 = cwick_metal_now_s();
    [b->cb commit];
    [b->cb waitUntilCompleted];
    if (getenv("CWICK_METAL_TIME")) {
        _t1 = cwick_metal_now_s();
        fprintf(stderr, "Tbatch n=%d wait=%.1fus gpu=%.1fus\n", b->n,
                (_t1-_t0)*1e6, (b->cb.GPUEndTime - b->cb.GPUStartTime)*1e6);
    }
    int ok = (b->cb.status == MTLCommandBufferStatusCompleted && !b->cb.error);
    if (ok)
        for (int i = 0; i < b->n; i++)
            memcpy(b->it[i].dst, (uint8_t *)b->bo.contents + b->it[i].ooff,
                   b->it[i].obytes);
    b->n = 0;
    return ok;
}

/* ------------------------------------------------------------------ */
/* DeepSeek block-scaled GEMV batches                                  */
/* ------------------------------------------------------------------ */
/* The DeepSeek backend has always done its arithmetic on the CPU pool: */
/* it was ported second, its weights are quantized differently from the  */
/* GLM checkpoint, and the CPU was fast enough to reach ~2.5 tok/s. The  */
/* consequence is that the GPU sits completely idle for its whole decode */
/* step, while the same memory system that feeds the CPU pool can serve  */
/* the GPU at roughly three times the rate.                              */
/*                                                                       */
/* These two batches hand part of every large GEMV to the GPU while the  */
/* CPU pool computes the rest, by splitting the output rows. Rows are    */
/* independent and each row's arithmetic is fixed by the CPU kernel, so  */
/* a split cannot change the answer -- only which engine produced which  */
/* row. That is what makes this safe to apply to every projection in the */
/* model at once.                                                        */
/*                                                                       */
/* The one thing the GPU cannot read is a weight that is not in a         */
/* Metal-visible allocation. The routed experts, the pinned dense tensors */
/* and the bf16 widening all come from cwick_shmem_alloc, so under        */
/* --backend metal they are already arena-backed and cost nothing to bind. */
/* Anything else (a test fixture on the stack, the mmap'd shard fallback) */
/* is refused and the caller keeps those rows on the CPU.                 */
#define CWICK_DS_BATCH_MAX   32
#define CWICK_DS_BATCH_SLOTS 2

typedef struct {
    uint32_t k, nb, n, row0, nsg, nbmax;
} cwick_ds_args;

struct cwick_ds_batch_s {
    id<MTLCommandBuffer>         cb;
    id<MTLComputeCommandEncoder> enc;
    id<MTLBuffer>                args;
    NSUInteger                   args_cap, args_n;
    id<MTLBuffer>                stg;      /* activation + output staging */
    size_t                       stg_cap, stg_used;
    int                          n, bad, kind, committed;
    double                       t_submit;
    struct {
        void  *dst;        /* caller's out + row0 */
        size_t ooff;       /* where this item's rows landed in stg */
        size_t obytes;
        size_t woff, swoff, aoff, asoff, rows;
        uint32_t nsg;
        size_t tgm;
        /* The views outlive the batch -- the arenas hold them for the life of
         * the process -- so an unretained reference is safe here. */
        __unsafe_unretained id<MTLBuffer> bw, bsw;
    } it[CWICK_DS_BATCH_MAX];
};

static struct cwick_ds_batch_s *g_ds_batch[CWICK_DS_BATCH_SLOTS];

void *cwick_metal_ds_batch_begin(int slot, int kind) {
    if (!g_queue || (kind == 0 ? !g_pso_ds_fp4 : !g_pso_ds_bf16)) return NULL;
    if (slot < 0) slot = 0;
    if (slot >= CWICK_DS_BATCH_SLOTS) slot = CWICK_DS_BATCH_SLOTS - 1;
    if (!g_ds_batch[slot]) {
        g_ds_batch[slot] = calloc(1, sizeof(struct cwick_ds_batch_s));
        if (!g_ds_batch[slot]) return NULL;
    }
    struct cwick_ds_batch_s *b = g_ds_batch[slot];
    b->cb = nil;
    b->enc = nil;
    b->args_n = 0;
    b->stg_used = 0;
    b->n = 0;
    b->bad = 0;
    b->kind = kind;
    b->committed = 0;
    return b;
}

static int ds_stg_reserve(struct cwick_ds_batch_s *b, size_t need, size_t *off) {
    *off = (b->stg_used + 63u) & ~(size_t)63u;
    size_t want = *off + need;
    if (!b->stg || b->stg_cap < want) {
        size_t cap = b->stg_cap ? b->stg_cap : (1u << 20);
        while (cap < want) cap *= 2;
        id<MTLBuffer> nb = [g_dev newBufferWithLength:cap
                                              options:MTLResourceStorageModeShared];
        if (!nb) return 0;
        b->stg = nb;
        b->stg_cap = cap;
    }
    b->stg_used = want;
    return 1;
}

int cwick_metal_ds_batch_gemv(void *vh, const void *w, const void *ws,
                              int64_t n, int64_t k, int64_t row0, int64_t row1,
                              const float *af, const float *asf, float *out) {
    struct cwick_ds_batch_s *b = (struct cwick_ds_batch_s *)vh;
    if (!b || b->bad || b->committed) return 0;
    if (!w || !ws || !af || !asf || !out) return 0;
    if (b->n >= CWICK_DS_BATCH_MAX) return 0;
    if (row0 < 0 || row1 > n || row0 >= row1) return 0;
    int nb = (int)(k / 32);
    if (nb <= 0 || k % 32) return 0;

    size_t wbytes = (b->kind == 0)
        ? (size_t)n * (size_t)(k / 2)
        : (size_t)n * (size_t)k * sizeof(uint16_t);
    size_t sbytes = (b->kind == 0)
        ? (size_t)n * (size_t)nb
        : (size_t)((n + 31) / 32) * (size_t)nb;

    id<MTLBuffer> bw = nil, bsw = nil;
    size_t woff = 0, swoff = 0;
    if (!arena_view(w, wbytes, &bw, &woff) && !view_of(w, wbytes, &bw, &woff)) {
        mdiag(b->kind == 0 ? "ds fp4 weights not GPU-visible"
                           : "ds bf16 weights not GPU-visible", w, wbytes);
        return 0;
    }
    if (!arena_view(ws, sbytes, &bsw, &swoff) && !view_of(ws, sbytes, &bsw, &swoff)) {
        mdiag("ds weight scales not GPU-visible", ws, sbytes);
        return 0;
    }

    size_t rows = (size_t)(row1 - row0);
    size_t aoff = 0, soff = 0, ooff = 0;
    if (!ds_stg_reserve(b, (size_t)k * sizeof(float), &aoff)) { b->bad = 1; return 0; }
    if (!ds_stg_reserve(b, (size_t)nb * sizeof(float), &soff)) { b->bad = 1; return 0; }
    if (!ds_stg_reserve(b, rows * sizeof(float), &ooff)) { b->bad = 1; return 0; }
    memcpy((uint8_t *)b->stg.contents + aoff, af, (size_t)k * sizeof(float));
    memcpy((uint8_t *)b->stg.contents + soff, asf, (size_t)nb * sizeof(float));

    int i = b->n++;
    b->it[i].dst = out + row0;
    b->it[i].ooff = ooff;
    b->it[i].obytes = rows * sizeof(float);
    /* One simdgroup per row, `nsg` of them per threadgroup, and one float of
     * threadgroup memory per block per row for the ordered fold. */
    uint32_t nsg = 8;
    uint32_t nbmax = (uint32_t)(((nb + 31) / 32) * 32);
    if ((size_t)nsg * nbmax * sizeof(float) > 16384) { b->bad = 1; return 0; }
    /* Per-item args live in a buffer rather than in setBytes: a concurrent
     * encoder replays dispatches in any order, and setBytes is only guaranteed
     * to be captured per dispatch on the serial path. */
    if (b->args_n + 1 > b->args_cap || !b->args) {
        NSUInteger cap = b->args_cap ? b->args_cap * 2 : 16;
        id<MTLBuffer> na = [g_dev newBufferWithLength:(size_t)cap * CWICK_ARGS_STRIDE
                                              options:MTLResourceStorageModeShared];
        if (!na) { b->bad = 1; return 0; }
        b->args = na;
        b->args_cap = cap;
        b->args_n = 0;
    }
    cwick_ds_args *ap = (cwick_ds_args *)((uint8_t *)b->args.contents +
                                          (size_t)b->args_n * CWICK_ARGS_STRIDE);
    ap->k = (uint32_t)k;
    ap->nb = (uint32_t)nb;
    ap->n = (uint32_t)row1;      /* stop row: threads past row1 must not write */
    ap->row0 = (uint32_t)row0;
    ap->nsg = nsg;
    ap->nbmax = nbmax;
    b->args_n++;
    b->it[i].nsg = nsg;
    b->it[i].tgm = (size_t)nsg * nbmax * sizeof(float);

    /* Keep the binding for the encode pass, which happens in commit(). */
    b->it[i].woff = woff;
    b->it[i].swoff = swoff;
    b->it[i].aoff = aoff;
    b->it[i].asoff = soff;
    b->it[i].bw = bw;
    b->it[i].bsw = bsw;
    b->it[i].rows = rows;
    return 1;
}

int cwick_metal_ds_batch_commit(void *vh) {
    struct cwick_ds_batch_s *b = (struct cwick_ds_batch_s *)vh;
    if (!b) return 0;
    if (b->bad) return 0;
    if (b->n == 0) { b->committed = 1; return 1; }
    b->cb = [g_queue commandBuffer];
    if (!b->cb) { b->bad = 1; return 0; }
    b->enc = cwick_concurrent_encoder(b->cb);
    if (!b->enc) { b->bad = 1; return 0; }
    [b->enc setComputePipelineState:(b->kind == 0 ? g_pso_ds_fp4 : g_pso_ds_bf16)];
    for (int i = 0; i < b->n; i++) {
        [b->enc setBuffer:b->args offset:(NSUInteger)i * CWICK_ARGS_STRIDE atIndex:0];
        [b->enc setBuffer:b->it[i].bw offset:b->it[i].woff atIndex:1];
        [b->enc setBuffer:b->it[i].bsw offset:b->it[i].swoff atIndex:2];
        [b->enc setBuffer:b->stg offset:b->it[i].aoff atIndex:3];
        [b->enc setBuffer:b->stg offset:b->it[i].asoff atIndex:4];
        [b->enc setBuffer:b->stg offset:b->it[i].ooff atIndex:5];
        [b->enc setThreadgroupMemoryLength:b->it[i].tgm atIndex:0];
        NSUInteger rows = b->it[i].rows;
        NSUInteger nsg = b->it[i].nsg;
        [b->enc dispatchThreadgroups:MTLSizeMake((rows + nsg - 1) / nsg, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(nsg * 32, 1, 1)];
    }
    [b->enc endEncoding];
    b->enc = nil;
    b->t_submit = cwick_metal_now_s();
    [b->cb commit];
    b->committed = 1;
    return 1;
}

int cwick_metal_ds_batch_wait(void *vh) {
    struct cwick_ds_batch_s *b = (struct cwick_ds_batch_s *)vh;
    if (!b) return 0;
    int ok = 0;
    if (!b->bad && b->committed && b->n > 0 && b->cb) {
        [b->cb waitUntilCompleted];
        ok = (b->cb.status == MTLCommandBufferStatusCompleted && !b->cb.error);
        if (ok)
            for (int i = 0; i < b->n; i++)
                memcpy(b->it[i].dst, (uint8_t *)b->stg.contents + b->it[i].ooff,
                       b->it[i].obytes);
        if (getenv("CWICK_METAL_DS_TIME"))
            fprintf(stderr, "dsgpu n=%d gpu=%.1fms wall=%.1fms\n", b->n,
                    (b->cb.GPUEndTime - b->cb.GPUStartTime) * 1e3,
                    (cwick_metal_now_s() - b->t_submit) * 1e3);
    }
    b->n = 0;
    b->args_n = 0;
    b->committed = 0;
    b->stg_used = 0;
    b->cb = nil;
    return ok;
}

int cwick_metal_ds_ready(void) {
    return (g_queue && g_pso_ds_fp4 && g_pso_ds_bf16) ? 1 : 0;
}

/* Drop every arena and every no-copy weight view, releasing the MTLBuffers
 * they hold.
 *
 * Both live in malloc'd arrays of structs with an `id<MTLBuffer>` field, so
 * ARC retains them on assignment but cannot release them when the array is
 * freed -- `free(g_arenas)` left the buffers alive, wired, for the rest of the
 * process. After a warmed-up model that is tens of GiB: measured on the
 * DeepSeek -> GLM switch, the new model reported a 67.19 GiB warm-up footprint
 * against 14.49 GiB for the same model started fresh, and the miss then
 * cascaded into the ceiling check eviscerating the expert cache and growing
 * 7.9 GiB per decode step until the process was killed.
 *
 * The views matter for the same reason and one more: they are no-copy buffers
 * over the *mapped weights*, so holding them keeps the outgoing model's
 * mapping alive even after the spine is closed. Releasing them is what makes
 * the unmap actually give the pages back. */
static void arena_release_all(void) {
    for (int i = 0; i < g_arena_n; i++) {
        cwick_fb *f = g_arenas[i].freelist;
        while (f) { cwick_fb *d = f; f = f->next; free(d); }
        for (unsigned b = 0; b < CWICK_HOT_BINS; b++) {
            cwick_fb *h = g_arenas[i].hot[b];
            while (h) { cwick_fb *d = h; h = h->next; free(d); }
            g_arenas[i].hot[b] = NULL;
            g_arenas[i].hot_len[b] = 0;
        }
        g_arenas[i].freelist = NULL;
        g_arenas[i].base = NULL;
        g_arenas[i].bump = 0;
        g_arenas[i].buf = nil;      /* ARC releases the wired MTLBuffer */
    }
    free(g_arenas);
    g_arenas = NULL;
    g_arena_n = g_arena_cap = 0;
    g_hint = -1;
    g_arena_bytes = 0;
    g_wired_bytes = 0;
    g_wire_failed = 0;
}

static void views_release_all(void) {
    for (int i = 0; i < g_view_n; i++) g_views[i].buf = nil;
    free(g_views);
    g_views = NULL;
    g_view_n = g_view_cap = 0;
    g_view_hint = -1;
}

/* Release only the no-copy views, leaving the arenas alive.
 *
 * The views wrap the model's *mapped* weights, so they have to go before the
 * mapping does: `unmap_weights` in the loader unmaps the shards, and a no-copy
 * MTLBuffer left pointing into that range is a view of address space that no
 * longer exists. Called from cwick_engine_free before cwick_spine_close. */
void cwick_metal_views_release(void) {
    pthread_mutex_lock(&g_arena_mtx);
    views_release_all();
    pthread_mutex_unlock(&g_arena_mtx);
}

/* Release one model's GPU-visible memory without tearing the device down.
 *
 * The device, queue and pipeline states are model-independent and cost a
 * shader compile to rebuild, so a model *switch* needs only this: it gives
 * back the outgoing model's expert arenas and the no-copy views over its
 * weights, and leaves the next model on the GPU. cwick_metal_cleanup is the
 * shutdown case and additionally drops everything the device owns. */
void cwick_metal_arena_release(void) {
    pthread_mutex_lock(&g_arena_mtx);
    arena_release_all();
    views_release_all();
    pthread_mutex_unlock(&g_arena_mtx);
}

void cwick_metal_bridge_cleanup(void) {
    pthread_mutex_lock(&g_arena_mtx);
    arena_release_all();
    views_release_all();
    pthread_mutex_unlock(&g_arena_mtx);
    g_scratch_w = g_scratch_s = g_bx = g_bo = nil;
    g_scratch_w_cap = g_scratch_s_cap = g_bx_cap = g_bo_cap = 0;
    g_pso = nil;
    g_pso_bf16 = nil;
    g_pso_ds_fp4 = nil;
    g_pso_ds_bf16 = nil;
    for (int i = 0; i < CWICK_DS_BATCH_SLOTS; i++) {
        if (g_ds_batch[i]) { g_ds_batch[i]->stg = nil; g_ds_batch[i]->args = nil; }
        free(g_ds_batch[i]);
        g_ds_batch[i] = NULL;
    }
    g_queue = nil;
    g_dev = nil;
}

int cwick_metal_init(void) {
    const char *nw = getenv("CWICK_METAL_NOWIRE");
    g_nowire = (nw && (strcmp(nw, "1") == 0 || strcasecmp(nw, "yes") == 0 ||
                       strcasecmp(nw, "on") == 0)) ? 1 : 0;
    int ok = cwick_metal_compile();
    if (ok && cwick_metal_batch_timing())
        fprintf(stderr, "metal: GPU-visible arena wiring %s\n",
                g_nowire ? "OFF (CWICK_METAL_NOWIRE)" : "ON");
    return ok;
}

/* Bytes of GPU-visible memory currently held wired (incompressible, never
 * swapped). Reported at startup so "the memory budget said 47.9 GiB and the
 * process only shows 17 GiB of RSS" has an explanation. */
size_t cwick_metal_wired_bytes(void) { return g_wired_bytes; }
