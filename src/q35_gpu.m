// q35_gpu.m — ObjC quarantine for the Qwen3.5 full-step Metal forward.
//
// Everything above this file is plain C: q35_gpu.c owns the layer schedule
// (which projection feeds which, what gets normalised, where the recurrence
// sits) and this unit owns only the things that need the Metal runtime —
// device, library, pipeline states, buffer objects, and the encode calls.
//
// The one idea that makes this worth existing: a decode step is encoded into a
// *single* command buffer with a *serial* compute encoder, so the GPU walks the
// whole 32-layer chain without a single round trip to the host. Measured on
// this machine, a round trip costs ~928 us and a step needs ~128 of them if
// each projection is submitted on its own — which is why the per-projection
// GPU path (metal_objc.m's BF16 batch) tops out at 1.27x. Here there is one
// commit and one wait for the entire step.
//
// That is also why the encoder must NOT be the concurrent kind used by the
// expert batch: those dispatches are independent by construction, whereas here
// every kernel reads what the previous one wrote.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cwick.h"
#include "log.h"
#include "q35_gpu.h"

extern const char *cwick_q35_shader_source(void);
extern const char *cwick_q35_gemv_n_source(int n);
extern int cwick_q35_gemv_n_rows(int n);
extern int cwick_q35_gemv_n_rw(int n);

/* Pipeline states, in the order they are declared in q35_shader.c. */
enum {
    P_GEMV, P_RMSNORM, P_ADD, P_EMBED, P_CONV, P_L2NORM, P_GDN_GATES, P_GDN,
    P_QK, P_KVSTORE, P_SCORES, P_SOFTMAX, P_AV, P_AVC, P_GATE, P_SILUMUL,
    P_GEMV_ROWS, P_EMBED_ID, P_ARGMAX,
    P_RMSNORM_ROWS, P_ADD_ROWS, P_SILUMUL_ROWS, P_L2NORM_ROWS, P_GDNGATES_ROWS,
    P_N
};
static const char *const P_NAME[P_N] = {
    "q35_gemv", "q35_rmsnorm", "q35_add", "q35_embed", "q35_conv_silu",
    "q35_l2norm", "q35_gdn_gates", "q35_gdn", "q35_qk_norm_rope",
    "q35_kv_store", "q35_attn_scores", "q35_attn_softmax", "q35_attn_av",
    "q35_attn_combine", "q35_attn_gate", "q35_silu_mul",
    "q35_gemv_rows", "q35_embed_id", "q35_argmax",
    "q35_rmsnorm_rows", "q35_add_rows", "q35_silu_mul_rows",
    "q35_l2norm_rows", "q35_gdn_gates_rows",
};

/* Argument layouts. These are the structs the shader declares, and the two
 * sides have to agree byte for byte — hence the explicit padding instead of
 * letting each compiler choose. */
typedef struct { uint32_t k, cols, rows_per_tg, pad; } a_gemv;
typedef struct { uint32_t n; float eps; uint32_t add_one, pad; } a_norm;
typedef struct { uint32_t k, cols, n, sx, so, pad0, pad1, pad2; } a_gemvn;
typedef struct { uint32_t dim, id; } a_embed;
typedef struct { uint32_t qkv, taps, pad0, pad1; } a_conv;
typedef struct { uint32_t nheads, qheads; float eps, qscale; } a_l2;
typedef struct { uint32_t nkv, hd, kv, keys, group; float scale; uint32_t pad0, pad1; } a_score;
typedef struct { uint32_t hd, kv, keys, chunk, nchunks, group, heads, pad0; } a_av;
typedef struct { uint32_t nchunks, heads, hd, pad; } a_avc;
typedef struct { uint32_t nh, hd; } a_gate;
typedef struct { uint32_t nq, hd, rotary; float eps; } a_qk;
typedef struct { uint32_t hk, hv, key_dim, val_dim; } a_gdn;
typedef struct { uint32_t n; float eps; uint32_t add_one, sx, so; } a_norm_rows;
typedef struct { uint32_t n, sx, sy, pad; } a_pair;
typedef struct { uint32_t nheads, qheads; float eps, qscale; uint32_t sx; } a_l2_rows;
typedef struct { uint32_t nh, sa, sb, sd, sbe; } a_gate_rows;

static id<MTLDevice>               g_dev;
static id<MTLCommandQueue>         g_queue;
static id<MTLComputePipelineState> g_pso[P_N];
/* The k-row GEMV is generated per k (N is a literal in the source), so its
 * pipeline states live in their own lazily-filled table. */
static id<MTLComputePipelineState> g_gemvn[(Q35_GK + 1) * 9];   /* [n][rw] */
/* Buffers handed to C. The array owns the retain; the pointers C holds are
 * unretained, which is safe because this array outlives them. */
static NSMutableArray              *g_keep;
static id<MTLCommandBuffer>         g_cb;
static id<MTLComputeCommandEncoder> g_enc;
static int                          g_ready;
static uint64_t                     g_maxtime;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------ */
/* lifecycle                                                           */
/* ------------------------------------------------------------------ */

int q35m_init(void) {
    if (g_ready) return 0;
    g_dev = MTLCreateSystemDefaultDevice();
    if (!g_dev) {
        CWICK_INFO("q35gpu", "no Metal device; the CPU path stays in charge");
        return 0;
    }
    g_queue = [g_dev newCommandQueue];
    if (!g_queue) return 0;
    g_keep = [NSMutableArray arrayWithCapacity:512];

    NSError *err = nil;
    NSString *src = [NSString stringWithUTF8String:cwick_q35_shader_source()];
    double t0 = now_s();
    id<MTLLibrary> lib = [g_dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) {
        CWICK_INFO("q35gpu", "shader compile failed: %s",
                   err ? err.localizedDescription.UTF8String : "(no error)");
        return 0;
    }
    for (int i = 0; i < P_N; i++) {
        id<MTLFunction> fn = [lib newFunctionWithName:
                                  [NSString stringWithUTF8String:P_NAME[i]]];
        if (!fn) {
            CWICK_INFO("q35gpu", "kernel %s is missing from the library",
                       P_NAME[i]);
            return 0;
        }
        g_pso[i] = [g_dev newComputePipelineStateWithFunction:fn error:&err];
        if (!g_pso[i]) {
            CWICK_INFO("q35gpu", "pipeline %s failed: %s", P_NAME[i],
                       err ? err.localizedDescription.UTF8String : "(no error)");
            return 0;
        }
    }
    g_maxtime = (uint64_t)g_dev.maxBufferLength;
    g_ready = 1;
    CWICK_INFO("q35gpu", "%s up: %d kernels in %.0f ms, max buffer %.2f GiB",
               g_dev.name.UTF8String, P_N, (now_s() - t0) * 1e3,
               (double)g_maxtime / 1073741824.0);
    return 1;
}

void q35m_shutdown(void) {
    g_enc = nil; g_cb = nil;
    g_keep = nil;
    for (int i = 0; i < P_N; i++) g_pso[i] = nil;
    for (int i = 0; i < (Q35_GK + 1) * 9; i++) g_gemvn[i] = nil;
    g_queue = nil; g_dev = nil;
    g_ready = 0;
}

uint64_t q35m_max_buffer(void) { return g_maxtime; }

/* A device-private or shared buffer for scratch. Returns an unretained handle
 * and, for a shared buffer, the host pointer the caller can write through. */
void *q35m_alloc(size_t bytes, int shared, void **host) {
    if (!g_ready || bytes == 0) return NULL;
    MTLResourceOptions opt = shared ? MTLResourceStorageModeShared
                                    : MTLResourceStorageModePrivate;
    id<MTLBuffer> b = [g_dev newBufferWithLength:bytes options:opt];
    if (!b) return NULL;
    [g_keep addObject:b];
    if (host) *host = shared ? b.contents : NULL;
    return (__bridge void *)b;
}

void q35m_free(void *h) {
    if (!h || !g_keep) return;
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)h;
    [g_keep removeObjectIdenticalTo:b];
}

/* Adopt a range of the mapped checkpoint as a no-copy buffer.
 *
 * The pages are already resident (q35_model_open faults the whole checkpoint
 * in), but a no-copy buffer requires them to be *mapped into the GPU's page
 * table*, which they only are once something touches them. One volatile read
 * per page is the documented way to guarantee that; skipping it showed up as a
 * rare SIGSEGV inside the kernel rather than a clean error. */
int q35m_resolve(const void *p, size_t len, void **buf, size_t *off) {
    if (!g_ready || !p || !len) return 0;
    const uintptr_t page = 4096u;
    uintptr_t raw = (uintptr_t)p;
    uintptr_t base = raw & ~(page - 1u);
    size_t lead = (size_t)(raw - base);
    size_t need = ((len + lead + page - 1u) / page) * page;
    if ((uint64_t)need > g_maxtime) return 0;
    {
        volatile const unsigned char *vp = (volatile const unsigned char *)base;
        unsigned char sink = 0;
        for (size_t o = 0; o < need; o += page) sink |= vp[o];
        (void)sink;
    }
    id<MTLBuffer> b = [g_dev newBufferWithBytesNoCopy:(void *)base
                                              length:need
                                             options:MTLResourceStorageModeShared
                                         deallocator:nil];
    if (!b) return 0;
    [g_keep addObject:b];
    *buf = (__bridge void *)b;
    *off = lead;
    return 1;
}

/* ------------------------------------------------------------------ */
/* encoder                                                             */
/* ------------------------------------------------------------------ */

int q35m_begin(void) {
    if (!g_ready || g_enc) return 0;
    g_cb = [g_queue commandBuffer];
    if (!g_cb) return 0;
    /* Serial on purpose: see the file header. */
    g_enc = [g_cb computeCommandEncoder];
    return g_enc ? 1 : 0;
}

/* Land the whole step. Returns 1 when the GPU ran every dispatch. */
int q35m_flush(void) {
    if (!g_enc) return 0;
    [g_enc endEncoding];
    g_enc = nil;
    int timed = getenv("CWICK_METAL_TIME") != NULL;
    double t0 = timed ? now_s() : 0.0;
    [g_cb commit];
    double t1 = timed ? now_s() : 0.0;
    [g_cb waitUntilCompleted];
    double t2 = timed ? now_s() : 0.0;
    int ok = (g_cb.status == MTLCommandBufferStatusCompleted && !g_cb.error);
    if (timed)
        fprintf(stderr, "Tq commit=%.1fus wait=%.1fus gpu=%.1fus %s\n",
                (t1 - t0) * 1e6, (t2 - t1) * 1e6,
                (g_cb.GPUEndTime - g_cb.GPUStartTime) * 1e6,
                ok ? "" : "FAILED");
    if (!ok && g_cb.error)
        CWICK_INFO("q35gpu", "step failed: %s",
                   g_cb.error.localizedDescription.UTF8String);
    g_cb = nil;
    return ok;
}

void q35m_abort(void) {
    if (g_enc) [g_enc endEncoding];
    g_enc = nil;
    if (g_cb) [g_cb commit];   /* no wait: the state is unrecoverable anyway */
    g_cb = nil;
}

/* ------------------------------------------------------------------ */
/* dispatch helpers                                                    */
/* ------------------------------------------------------------------ */

#define ENC(i) id<MTLComputeCommandEncoder> e = g_enc; \
               if (!e) return 0; [e setComputePipelineState:g_pso[i]]

int q35m_gemv(void *W, size_t woff, uint32_t k, uint32_t cols,
              void *X, size_t xoff, void *O, size_t ooff) {
    if (!g_enc || !W || !X || !O) return 0;
    ENC(P_GEMV);
    a_gemv a = { k, cols, 8u, 0u };
    [e setBytes:&a length:sizeof(a) atIndex:0];
    [e setBuffer:(__bridge id<MTLBuffer>)W offset:woff atIndex:1];
    [e setBuffer:(__bridge id<MTLBuffer>)X offset:xoff atIndex:2];
    [e setBuffer:(__bridge id<MTLBuffer>)O offset:ooff atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(1, (cols + 7u) / 8u, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}

/* The k-row GEMV. The source is generated per k because N is a literal there,
 * which keeps the accumulator loop straight-line; the pipeline state is built
 * on first use and kept for the life of the process. */
static id<MTLComputePipelineState> gemvn_pso(uint32_t n) {
    if (!g_ready || n < 1u || n > (uint32_t)Q35_GK) return nil;
    const uint32_t rw = (uint32_t)cwick_q35_gemv_n_rw((int)n);
    const unsigned slot = n * 9u + rw;
    if (g_gemvn[slot]) return g_gemvn[slot];
    const char *src = cwick_q35_gemv_n_source((int)n);
    if (!src) return nil;
    NSError *err = nil;
    id<MTLLibrary> lib = [g_dev newLibraryWithSource:[NSString stringWithUTF8String:src]
                                             options:nil error:&err];
    if (!lib) {
        CWICK_INFO("q35gpu", "k=%u GEMV compile failed: %s", n,
                   err ? err.localizedDescription.UTF8String : "(no error)");
        return nil;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"q35_gemv_n"];
    id<MTLComputePipelineState> pso =
        fn ? [g_dev newComputePipelineStateWithFunction:fn error:&err] : nil;
    if (!pso) {
        CWICK_INFO("q35gpu", "k=%u GEMV pipeline failed: %s", n,
                   err ? err.localizedDescription.UTF8String : "(no error)");
        return nil;
    }
    g_gemvn[slot] = pso;
    return pso;
}

int q35m_gemv_n_prepare(uint32_t n) { return gemvn_pso(n) != nil; }

/* A buffer-scope barrier between two dispatches that touch the same buffer in
 * the same encoder. A serial encoder is supposed to make its dispatches
 * ordered by construction; the k-row path found a case where the second read
 * of a recurrence state saw the first dispatch's *input*, so the ordering is
 * made explicit here rather than assumed. */
int q35m_barrier(void) {
    if (!g_enc) return 0;
    [g_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    return 1;
}

int q35m_gemv_n(void *W, size_t woff, uint32_t k, uint32_t cols,
                void *X, size_t xoff, void *O, size_t ooff, uint32_t n,
                uint32_t sx, uint32_t so) {
    if (!g_enc || !W || !X || !O) return 0;
    id<MTLComputePipelineState> pso = gemvn_pso(n);
    if (!pso) return 0;
    id<MTLComputeCommandEncoder> e = g_enc;
    [e setComputePipelineState:pso];
    /* sx / so are element strides: a scratch buffer is wider than any single
     * projection's output (mix holds both the linear qkv and the attention q),
     * so the row spacing has to travel with the dispatch. */
    a_gemvn a = { k, cols, n, sx, so, 0u, 0u, 0u };
    [e setBytes:&a length:sizeof(a) atIndex:0];
    [e setBuffer:(__bridge id<MTLBuffer>)W offset:woff atIndex:1];
    [e setBuffer:(__bridge id<MTLBuffer>)X offset:xoff atIndex:2];
    [e setBuffer:(__bridge id<MTLBuffer>)O offset:ooff atIndex:3];
    /* Output rows per threadgroup come from the same function that generated
     * the body (RW * RP), and so does the thread count: RW simdgroups of 32. */
    const NSUInteger rw = (NSUInteger)cwick_q35_gemv_n_rows((int)n);
    const NSUInteger sgs = (NSUInteger)cwick_q35_gemv_n_rw((int)n);
    [e dispatchThreadgroups:MTLSizeMake(1, (cols + rw - 1) / rw, 1)
        threadsPerThreadgroup:MTLSizeMake(32u * sgs, 1, 1)];
    return 1;
}

/* The vocabulary projection restricted to a candidate list. Same dispatch
 * geometry as q35m_gemv, because the kernel is the same sum over a row -- it
 * only reads the row's index instead of computing it from the group id. */
int q35m_gemv_rows(void *W, size_t woff, uint32_t k, uint32_t cols,
                   void *Idx, size_t ioff, void *X, size_t xoff,
                   void *O, size_t ooff) {
    if (!g_enc || !W || !Idx || !X || !O) return 0;
    ENC(P_GEMV_ROWS);
    a_gemv a = { k, cols, 8u, 0u };
    [e setBytes:&a length:sizeof(a) atIndex:0];
    [e setBuffer:(__bridge id<MTLBuffer>)W offset:woff atIndex:1];
    [e setBuffer:(__bridge id<MTLBuffer>)Idx offset:ioff atIndex:2];
    [e setBuffer:(__bridge id<MTLBuffer>)X offset:xoff atIndex:3];
    [e setBuffer:(__bridge id<MTLBuffer>)O offset:ooff atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake(1, (cols + 7u) / 8u, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}

/* Embedding of the token an earlier dispatch chose. */
int q35m_embed_id(void *E, size_t eoff, void *X, size_t xoff,
                  void *Id, size_t idoff, uint32_t dim) {
    if (!g_enc || !E || !X || !Id) return 0;
    ENC(P_EMBED_ID);
    [e setBuffer:(__bridge id<MTLBuffer>)E offset:eoff atIndex:0];
    [e setBuffer:(__bridge id<MTLBuffer>)X offset:xoff atIndex:1];
    [e setBuffer:(__bridge id<MTLBuffer>)Id offset:idoff atIndex:2];
    [e setBytes:&dim length:sizeof(dim) atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}

/* out[0] = the candidate index whose score is highest. `idx` may be NULL, in
 * which case the score row is already position-major and the index is the
 * position. */
int q35m_argmax(void *S, size_t soff, void *Idx, size_t ioff,
                void *Out, size_t ooff, uint32_t n) {
    if (!g_enc || !S || !Out) return 0;
    ENC(P_ARGMAX);
    [e setBuffer:(__bridge id<MTLBuffer>)S offset:soff atIndex:0];
    if (Idx) [e setBuffer:(__bridge id<MTLBuffer>)Idx offset:ioff atIndex:1];
    else     [e setBuffer:nil offset:0 atIndex:1];
    [e setBuffer:(__bridge id<MTLBuffer>)Out offset:ooff atIndex:2];
    [e setBytes:&n length:sizeof(n) atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    return 1;
}

int q35m_rmsnorm(void *X, size_t xoff, void *W, size_t woff,
                 void *O, size_t ooff, uint32_t n, float eps, int add_one) {    if (!g_enc || !X || !W || !O) return 0;
    ENC(P_RMSNORM);
    a_norm a = { n, eps, add_one ? 1u : 0u, 0u };
    [e setBytes:&a length:sizeof(a) atIndex:3];
    [e setBuffer:(__bridge id<MTLBuffer>)X offset:xoff atIndex:0];
    [e setBuffer:(__bridge id<MTLBuffer>)W offset:woff atIndex:1];
    [e setBuffer:(__bridge id<MTLBuffer>)O offset:ooff atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}

int q35m_add(void *X, size_t xoff, void *Y, size_t yoff, uint32_t n) {
    if (!g_enc || !X || !Y) return 0;
    ENC(P_ADD);
    [e setBuffer:(__bridge id<MTLBuffer>)X offset:xoff atIndex:0];
    [e setBuffer:(__bridge id<MTLBuffer>)Y offset:yoff atIndex:1];
    [e setBytes:&n length:sizeof(n) atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake((n + 255u) / 256u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}

int q35m_embed(void *E, size_t eoff, void *X, size_t xoff, uint32_t dim,
               uint32_t tok) {
    if (!g_enc || !E || !X) return 0;
    ENC(P_EMBED);
    a_embed a = { dim, tok };
    [e setBytes:&a length:sizeof(a) atIndex:2];
    [e setBuffer:(__bridge id<MTLBuffer>)E offset:eoff atIndex:0];
    [e setBuffer:(__bridge id<MTLBuffer>)X offset:xoff atIndex:1];
    [e dispatchThreadgroups:MTLSizeMake((dim + 255u) / 256u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}

int q35m_conv_silu(void *Mix, size_t moff, void *Conv, size_t coff,
                   void *St, size_t soff, void *Stp, size_t stpoff, void *Y,
                   size_t yoff, uint32_t qkv, uint32_t taps) {
    if (!g_enc || !Mix || !Conv || !St || !Stp || !Y) return 0;
    ENC(P_CONV);
    a_conv a = { qkv, taps, 0u, 0u };
    [e setBytes:&a length:sizeof(a) atIndex:4];
    [e setBuffer:(__bridge id<MTLBuffer>)Mix offset:moff atIndex:0];
    [e setBuffer:(__bridge id<MTLBuffer>)Conv offset:coff atIndex:1];
    [e setBuffer:(__bridge id<MTLBuffer>)St offset:soff atIndex:2];
    [e setBuffer:(__bridge id<MTLBuffer>)Y offset:yoff atIndex:3];
    [e setBuffer:(__bridge id<MTLBuffer>)Stp offset:stpoff atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake((qkv + 255u) / 256u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}

int q35m_l2norm(void *Y, size_t yoff, uint32_t nheads, uint32_t qheads,
                float eps, float qscale) {
    if (!g_enc || !Y) return 0;
    ENC(P_L2NORM);
    a_l2 a = { nheads, qheads, eps, qscale };
    [e setBytes:&a length:sizeof(a) atIndex:1];
    [e setBuffer:(__bridge id<MTLBuffer>)Y offset:yoff atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(nheads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}

int q35m_gdn_gates(void *A, size_t aoff, void *B, size_t boff, void *Alog,
                   size_t logoff, void *Dtb, size_t dtoff, void *Decay,
                   size_t decoff, void *Beta, size_t betoff, uint32_t nh) {
    if (!g_enc || !A || !B || !Alog || !Dtb || !Decay || !Beta) return 0;
    ENC(P_GDN_GATES);
    [e setBytes:&nh length:sizeof(nh) atIndex:6];
    [e setBuffer:(__bridge id<MTLBuffer>)A offset:aoff atIndex:0];
    [e setBuffer:(__bridge id<MTLBuffer>)B offset:boff atIndex:1];
    [e setBuffer:(__bridge id<MTLBuffer>)Alog offset:logoff atIndex:2];
    [e setBuffer:(__bridge id<MTLBuffer>)Dtb offset:dtoff atIndex:3];
    [e setBuffer:(__bridge id<MTLBuffer>)Decay offset:decoff atIndex:4];
    [e setBuffer:(__bridge id<MTLBuffer>)Beta offset:betoff atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}

int q35m_gdn(void *S, size_t soff, void *Sp, size_t spoff, void *Qkv,
             size_t qoff, void *Z, size_t zoff, void *Nw, size_t nwoff,
             void *Decay, size_t decoff, void *Beta, size_t betoff, void *Out,
             size_t ooff, uint32_t nh, uint32_t hk, uint32_t hv,
             uint32_t key_dim, uint32_t val_dim) {
    if (!g_enc || !S || !Sp || !Qkv || !Z || !Nw || !Out) return 0;
    ENC(P_GDN);
    a_gdn a = { hk, hv, key_dim, val_dim };
    [e setBytes:&a length:sizeof(a) atIndex:7];
    [e setBuffer:(__bridge id<MTLBuffer>)S offset:soff atIndex:0];
    [e setBuffer:(__bridge id<MTLBuffer>)Qkv offset:qoff atIndex:1];
    [e setBuffer:(__bridge id<MTLBuffer>)Z offset:zoff atIndex:2];
    [e setBuffer:(__bridge id<MTLBuffer>)Nw offset:nwoff atIndex:3];
    [e setBuffer:(__bridge id<MTLBuffer>)Decay offset:decoff atIndex:4];
    [e setBuffer:(__bridge id<MTLBuffer>)Beta offset:betoff atIndex:5];
    [e setBuffer:(__bridge id<MTLBuffer>)Out offset:ooff atIndex:6];
    [e setBuffer:(__bridge id<MTLBuffer>)Sp offset:spoff atIndex:8];
    [e dispatchThreadgroups:MTLSizeMake(nh, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(hv, 1, 1)];
    return 1;
}

int q35m_qk(void *Qsrc, size_t qso, void *Ksrc, size_t kso, void *Qout,
            size_t qoo, void *Krow, size_t kro, void *Qnw, size_t qno,
            void *Knw, size_t kno, void *Cos, size_t co, void *Sin, size_t so,
            uint32_t nq, uint32_t nkv, uint32_t hd, uint32_t rotary, float eps) {
    if (!g_enc || !Qsrc || !Ksrc || !Qout || !Krow) return 0;
    ENC(P_QK);
    a_qk a = { nq, hd, rotary, eps };
    [e setBytes:&a length:sizeof(a) atIndex:8];
    [e setBuffer:(__bridge id<MTLBuffer>)Qsrc offset:qso atIndex:0];
    [e setBuffer:(__bridge id<MTLBuffer>)Ksrc offset:kso atIndex:1];
    [e setBuffer:(__bridge id<MTLBuffer>)Qout offset:qoo atIndex:2];
    [e setBuffer:(__bridge id<MTLBuffer>)Krow offset:kro atIndex:3];
    [e setBuffer:(__bridge id<MTLBuffer>)Qnw offset:qno atIndex:4];
    [e setBuffer:(__bridge id<MTLBuffer>)Knw offset:kno atIndex:5];
    [e setBuffer:(__bridge id<MTLBuffer>)Cos offset:co atIndex:6];
    [e setBuffer:(__bridge id<MTLBuffer>)Sin offset:so atIndex:7];
    [e dispatchThreadgroups:MTLSizeMake(nq + nkv, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}

int q35m_kv_store(void *Src, size_t so, void *Dst, size_t doff, uint32_t n) {
    if (!g_enc || !Src || !Dst) return 0;
    ENC(P_KVSTORE);
    [e setBytes:&n length:sizeof(n) atIndex:2];
    [e setBuffer:(__bridge id<MTLBuffer>)Src offset:so atIndex:0];
    [e setBuffer:(__bridge id<MTLBuffer>)Dst offset:doff atIndex:1];
    [e dispatchThreadgroups:MTLSizeMake((n + 255u) / 256u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}

int q35m_scores(void *Q, size_t qo, void *Kc, size_t ko, void *S, size_t so,
                uint32_t heads, uint32_t hd, uint32_t kv, uint32_t keys,
                uint32_t group, float scale) {
    if (!g_enc || !Q || !Kc || !S) return 0;
    ENC(P_SCORES);
    a_score a = { heads, hd, kv, keys, group, scale, 0u, 0u };
    [e setBytes:&a length:sizeof(a) atIndex:3];
    [e setBuffer:(__bridge id<MTLBuffer>)Q offset:qo atIndex:0];
    [e setBuffer:(__bridge id<MTLBuffer>)Kc offset:ko atIndex:1];
    [e setBuffer:(__bridge id<MTLBuffer>)S offset:so atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(heads, (keys + 7u) / 8u, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}

int q35m_softmax(void *S, size_t so, uint32_t heads, uint32_t keys) {
    if (!g_enc || !S) return 0;
    ENC(P_SOFTMAX);
    [e setBytes:&keys length:sizeof(keys) atIndex:1];
    [e setBuffer:(__bridge id<MTLBuffer>)S offset:so atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(heads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}

int q35m_av(void *S, size_t so, void *Vc, size_t vo, void *Part, size_t po,
            uint32_t heads, uint32_t hd, uint32_t kv, uint32_t keys,
            uint32_t group, uint32_t chunk, uint32_t nchunks) {
    if (!g_enc || !S || !Vc || !Part) return 0;
    ENC(P_AV);
    a_av a = { hd, kv, keys, chunk, nchunks, group, heads, 0u };
    [e setBytes:&a length:sizeof(a) atIndex:3];
    [e setBuffer:(__bridge id<MTLBuffer>)S offset:so atIndex:0];
    [e setBuffer:(__bridge id<MTLBuffer>)Vc offset:vo atIndex:1];
    [e setBuffer:(__bridge id<MTLBuffer>)Part offset:po atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(heads, nchunks, 1)
        threadsPerThreadgroup:MTLSizeMake(hd, 1, 1)];
    return 1;
}

int q35m_av_combine(void *Part, size_t po, void *O, size_t oo,
                    uint32_t nchunks, uint32_t heads, uint32_t hd) {
    if (!g_enc || !Part || !O) return 0;
    ENC(P_AVC);
    a_avc a = { nchunks, heads, hd, 0u };
    [e setBytes:&a length:sizeof(a) atIndex:2];
    [e setBuffer:(__bridge id<MTLBuffer>)Part offset:po atIndex:0];
    [e setBuffer:(__bridge id<MTLBuffer>)O offset:oo atIndex:1];
    [e dispatchThreadgroups:MTLSizeMake(heads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(hd, 1, 1)];
    return 1;
}

int q35m_attn_gate(void *O, size_t oo, void *Qg, size_t qo,
                   uint32_t nh, uint32_t hd) {
    if (!g_enc || !O || !Qg) return 0;
    ENC(P_GATE);
    a_gate a = { nh, hd };
    [e setBytes:&a length:sizeof(a) atIndex:2];
    [e setBuffer:(__bridge id<MTLBuffer>)O offset:oo atIndex:0];
    [e setBuffer:(__bridge id<MTLBuffer>)Qg offset:qo atIndex:1];
    uint32_t n = nh * hd;
    [e dispatchThreadgroups:MTLSizeMake((n + 255u) / 256u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}

int q35m_silu_mul(void *G, size_t go, void *U, size_t uo, uint32_t n) {
    if (!g_enc || !G || !U) return 0;
    ENC(P_SILUMUL);
    [e setBytes:&n length:sizeof(n) atIndex:2];
    [e setBuffer:(__bridge id<MTLBuffer>)G offset:go atIndex:0];
    [e setBuffer:(__bridge id<MTLBuffer>)U offset:uo atIndex:1];
    [e dispatchThreadgroups:MTLSizeMake((n + 255u) / 256u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}

/* ------------------------------------------------------------------ */
/* the row-batched forms: one dispatch for all n rows                  */
/* ------------------------------------------------------------------ */
/* sx / so are element strides between rows, so the caller passes the
 * allocation widths of the scratch buffers rather than the projection widths --
 * the same convention q35m_gemv_n uses, and for the same reason. */

int q35m_rmsnorm_rows(void *X, size_t xoff, void *W, size_t woff,
                      void *O, size_t ooff, uint32_t n, float eps, int add_one,
                      uint32_t rows, uint32_t sx, uint32_t so) {
    if (!g_enc || !X || !W || !O || rows < 1u) return 0;
    ENC(P_RMSNORM_ROWS);
    a_norm_rows a = { n, eps, add_one ? 1u : 0u, sx, so };
    [e setBytes:&a length:sizeof(a) atIndex:3];
    [e setBuffer:(__bridge id<MTLBuffer>)X offset:xoff atIndex:0];
    [e setBuffer:(__bridge id<MTLBuffer>)W offset:woff atIndex:1];
    [e setBuffer:(__bridge id<MTLBuffer>)O offset:ooff atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}

int q35m_add_rows(void *X, size_t xoff, void *Y, size_t yoff, uint32_t n,
                  uint32_t rows, uint32_t sx, uint32_t sy) {
    if (!g_enc || !X || !Y || rows < 1u) return 0;
    ENC(P_ADD_ROWS);
    a_pair a = { n, sx, sy, 0u };
    [e setBytes:&a length:sizeof(a) atIndex:2];
    [e setBuffer:(__bridge id<MTLBuffer>)X offset:xoff atIndex:0];
    [e setBuffer:(__bridge id<MTLBuffer>)Y offset:yoff atIndex:1];
    [e dispatchThreadgroups:MTLSizeMake(rows, (n + 255u) / 256u, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}

int q35m_silu_mul_rows(void *G, size_t go, void *U, size_t uo, uint32_t n,
                       uint32_t rows, uint32_t sx, uint32_t sy) {
    if (!g_enc || !G || !U || rows < 1u) return 0;
    ENC(P_SILUMUL_ROWS);
    a_pair a = { n, sx, sy, 0u };
    [e setBytes:&a length:sizeof(a) atIndex:2];
    [e setBuffer:(__bridge id<MTLBuffer>)G offset:go atIndex:0];
    [e setBuffer:(__bridge id<MTLBuffer>)U offset:uo atIndex:1];
    [e dispatchThreadgroups:MTLSizeMake(rows, (n + 255u) / 256u, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}

int q35m_l2norm_rows(void *Y, size_t yoff, uint32_t nheads, uint32_t qheads,
                     float eps, float qscale, uint32_t rows, uint32_t sx) {
    if (!g_enc || !Y || rows < 1u) return 0;
    ENC(P_L2NORM_ROWS);
    a_l2_rows a = { nheads, qheads, eps, qscale, sx };
    [e setBytes:&a length:sizeof(a) atIndex:1];
    [e setBuffer:(__bridge id<MTLBuffer>)Y offset:yoff atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(nheads, rows, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}

int q35m_gdn_gates_rows(void *A, size_t aoff, void *B, size_t boff, void *Alog,
                        size_t logoff, void *Dtb, size_t dtoff, void *Decay,
                        size_t decoff, void *Beta, size_t betoff, uint32_t nh,
                        uint32_t rows, uint32_t sa, uint32_t sb, uint32_t sd,
                        uint32_t sbe) {
    if (!g_enc || !A || !B || !Alog || !Dtb || !Decay || !Beta || rows < 1u)
        return 0;
    ENC(P_GDNGATES_ROWS);
    a_gate_rows a = { nh, sa, sb, sd, sbe };
    [e setBytes:&a length:sizeof(a) atIndex:6];
    [e setBuffer:(__bridge id<MTLBuffer>)A offset:aoff atIndex:0];
    [e setBuffer:(__bridge id<MTLBuffer>)B offset:boff atIndex:1];
    [e setBuffer:(__bridge id<MTLBuffer>)Alog offset:logoff atIndex:2];
    [e setBuffer:(__bridge id<MTLBuffer>)Dtb offset:dtoff atIndex:3];
    [e setBuffer:(__bridge id<MTLBuffer>)Decay offset:decoff atIndex:4];
    [e setBuffer:(__bridge id<MTLBuffer>)Beta offset:betoff atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}
