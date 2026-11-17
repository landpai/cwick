/* h3_gpu.m — Metal device, pipelines and the one kernel the image
 * transformer needs.
 *
 * The kernel is a 64x64 output tile per threadgroup: four simdgroups in a 2x2
 * arrangement, each owning a 32x32 block built from 8x8 matrix fragments. The
 * weight tile is staged into threadgroup memory in the checkpoint's own [n][k]
 * order and loaded with the transpose flag, so nothing on the host has to
 * rearrange 61 GiB of weights.
 *
 * Activation and weight blocks are adopted as no-copy buffers over the pages
 * they already occupy. A no-copy buffer only works if the pages are mapped into
 * the GPU's page table, which happens when something touches them: the resolve
 * below reads one byte per page first, otherwise the kernel faults rather than
 * reporting an error.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "h3_gpu.h"

int h3m_attention(const float *q, const float *k, const float *v, float *o,
                  int seq, int ld, int heads);

const char *h3m_shader_source(void);

#define H3_GPU_TG 128

/* Same kernel as the standalone probe, with the strides carried in the
 * argument block so a per-head slice of a packed activation can be read
 * without copying it out first. */
static const char *H3_MSL = R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define TG_M 64
#define TG_N 64
#define TG_K 16
#define AM 64
#define AK 16

struct gargs { int rows, k, cols, lda, ldc; long woff; };

kernel void gemm_bf16(device const float *A [[buffer(0)]],
                      device const bfloat *Wr [[buffer(1)]],
                      device float *C [[buffer(2)]],
                      constant gargs &a [[buffer(3)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint2 lid [[thread_position_in_threadgroup]],
                      uint sgi [[simdgroup_index_in_threadgroup]]) {
    threadgroup float tC[TG_M][TG_N];
    threadgroup bfloat tA[TG_M][TG_K];
    threadgroup bfloat tB[TG_N][TG_K];

    /* A projection points into a shard this kernel already holds as one
     * buffer, so the tensor's byte offset arrives as an argument rather than
     * as a buffer offset: safetensors offsets are only 8-byte aligned and a
     * buffer offset has to stay on a page boundary. */
    device const bfloat *W = (device const bfloat *)((device const char *)Wr + a.woff);

    int m0 = (int)tg.y * TG_M, n0 = (int)tg.x * TG_N;
    int mg = (int)(sgi >> 1) * 32, ng = (int)(sgi & 1) * 32;
    int lin = (int)lid.x;

    simdgroup_float8x8 acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < a.k; k0 += TG_K) {
        for (int pass = 0; pass < 8; pass++) {
            int idx = pass * 128 + lin;
            int r = idx / TG_K, c = idx % TG_K;
            int gm = m0 + r, gn = n0 + r, gk = k0 + c;
            tA[r][c] = (gm < a.rows) ? bfloat(A[(long)gm * a.lda + gk]) : bfloat(0.0f);
            tB[r][c] = (gn < a.cols) ? W[(long)gn * a.k + gk] : bfloat(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kf = 0; kf < TG_K / 8; kf++) {
            simdgroup_bfloat8x8 af[4], bf[4];
            for (int mi = 0; mi < 4; mi++)
                simdgroup_load(af[mi], &tA[mg + mi * 8][kf * 8], TG_K);
            for (int ni = 0; ni < 4; ni++)
                simdgroup_load(bf[ni], &tB[ng + ni * 8][kf * 8], TG_K,
                               ulong2(0, 0), true);
            for (int mi = 0; mi < 4; mi++)
                for (int ni = 0; ni < 4; ni++)
                    simdgroup_multiply_accumulate(acc[mi][ni], af[mi], bf[ni],
                                                  acc[mi][ni]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int mi = 0; mi < 4; mi++)
        for (int ni = 0; ni < 4; ni++)
            simdgroup_store(acc[mi][ni], &tC[mg + mi * 8][ng + ni * 8], TG_N);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int pass = 0; pass < 32; pass++) {
        int idx = pass * 128 + lin;
        int r = idx / TG_N, c = idx % TG_N;
        int gm = m0 + r, gn = n0 + c;
        if (gm < a.rows && gn < a.cols) C[(long)gm * a.ldc + gn] = tC[r][c];
    }
}

/* Attention over the packed sequence, one threadgroup per (row block, head).
 *
 * The CPU version tiles the score matrix and calls cblas twice per tile, which
 * is fine at a few hundred rows and hopeless at sixteen thousand: the cost is
 * quadratic in the sequence, and it is the larger half of an edit step. This is
 * the streaming form instead — the key range is walked in 16-key blocks, no
 * score matrix is ever materialised beyond one block, and 64 query rows share
 * each pass over the keys.
 *
 * The score block goes through threadgroup memory rather than staying in
 * fragments because the softmax needs a row-wise reduction: one thread owns one
 * row of 16 scores, which is cheaper than reducing across fragment lanes and
 * does not depend on how the compiler lays a fragment out.
 *
 * The key range is walked twice. The first pass takes the row maximum, the
 * second exponentiates against it and accumulates. A single streaming pass
 * would need the running maximum, and applying its correction means rescaling
 * the output accumulator — which is exactly what a set of matrix fragments
 * cannot do element-wise.
 */
struct aargs { int seq, ld, hoff; float scale; };

kernel void attn_fa(device const float *Q [[buffer(0)]],
                    device const float *Kb [[buffer(1)]],
                    device const float *Vb [[buffer(2)]],
                    device float *Ob [[buffer(3)]],
                    constant aargs &a [[buffer(4)]],
                    uint2 tg [[threadgroup_position_in_grid]],
                    uint2 lid [[thread_position_in_threadgroup]],
                    uint sgi [[simdgroup_index_in_threadgroup]],
                    uint slane [[thread_index_in_simdgroup]]) {
    threadgroup bfloat tQ[AM][128];
    threadgroup bfloat tK[AK][128];
    threadgroup bfloat tV[AK][128];
    /* The score block is dead by the time a straddling block is stored, so the
     * same 4 KiB carries the staged 8x128 output block. */
    threadgroup union { float s[AM][AK]; float o[8][128]; } u;
    threadgroup bfloat tP[AM][AK];
    threadgroup float tSum[AM];
    threadgroup float tMax[AM];

    const int lin = (int)lid.x;
    const int m0 = (int)tg.x * AM;
    const int hoff = (int)tg.y * a.hoff;

    if (lin < AM) { tSum[lin] = 0.0f; tMax[lin] = -INFINITY; }
    {
        int r = lin >> 2, seg = (lin & 3) * 32;      /* 64 rows x 4 spans of 32 */
        int gm = m0 + r;
        for (int i = 0; i < 32; i += 4) {
            float4 f = float4(0.0f);
            if (gm < a.seq)
                f = *(const device float4 *)(Q + (long)gm * a.ld + hoff + seg + i);
            tQ[r][seg + i + 0] = bfloat(f.x);
            tQ[r][seg + i + 1] = bfloat(f.y);
            tQ[r][seg + i + 2] = bfloat(f.z);
            tQ[r][seg + i + 3] = bfloat(f.w);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* First pass: the row maximum over the whole key range.
     *
     * The exponent is taken as exp(score - row_max). Skipping the subtraction
     * is only safe if scores are small, and in this checkpoint they are not:
     * the residual stream reaches four digits by the end of the first block, so
     * a score of a few times 10^4 overflows fp32 and every later stage turns
     * into NaN. A row's maximum cannot be known before its last key, and the
     * output accumulator is a set of matrix fragments that cannot be rescaled
     * element-wise, so the maximum is collected here first. Only the score
     * product runs in this pass — no values, no exponentiation. */
    for (int j0 = 0; j0 < a.seq; j0 += AK) {
        int r = lin >> 4, seg = (lin & 15) * 8;
        int gn = j0 + r;
        for (int i = 0; i < 8; i += 4) {
            float4 f = float4(0.0f);
            if (gn < a.seq)
                f = *(const device float4 *)(Kb + (long)gn * a.ld + hoff + seg + i);
            tK[r][seg + i + 0] = bfloat(f.x);
            tK[r][seg + i + 1] = bfloat(f.y);
            tK[r][seg + i + 2] = bfloat(f.z);
            tK[r][seg + i + 3] = bfloat(f.w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 S[AK / 8];
        for (int i = 0; i < AK / 8; i++)
            S[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (int kf = 0; kf < 16; kf++) {
            simdgroup_bfloat8x8 qf;
            simdgroup_load(qf, &tQ[sgi * 8][kf * 8], 128);
            for (int ni = 0; ni < AK / 8; ni++) {
                simdgroup_bfloat8x8 kf8;
                simdgroup_load(kf8, &tK[ni * 8][kf * 8], 128, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S[ni], qf, kf8, S[ni]);
            }
        }
        for (int ni = 0; ni < AK / 8; ni++)
            simdgroup_store(S[ni], &u.s[sgi * 8][ni * 8], AK);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lin < AM) {
            float bm = tMax[lin];
            for (int c = 0; c < AK; c++)
                if (j0 + c < a.seq) {
                    float v = u.s[lin][c] * a.scale;
                    if (v > bm) bm = v;
                }
            tMax[lin] = bm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_float8x8 O[16];
    for (int i = 0; i < 16; i++)
        O[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int j0 = 0; j0 < a.seq; j0 += AK) {
        int r = lin >> 4, seg = (lin & 15) * 8;      /* AK rows x 16 spans of 8 */
        int gn = j0 + r;
        for (int i = 0; i < 8; i += 4) {
            float4 f = float4(0.0f), g = float4(0.0f);
            if (gn < a.seq) {
                f = *(const device float4 *)(Kb + (long)gn * a.ld + hoff + seg + i);
                g = *(const device float4 *)(Vb + (long)gn * a.ld + hoff + seg + i);
            }
            tK[r][seg + i + 0] = bfloat(f.x);
            tK[r][seg + i + 1] = bfloat(f.y);
            tK[r][seg + i + 2] = bfloat(f.z);
            tK[r][seg + i + 3] = bfloat(f.w);
            tV[r][seg + i + 0] = bfloat(g.x);
            tV[r][seg + i + 1] = bfloat(g.y);
            tV[r][seg + i + 2] = bfloat(g.z);
            tV[r][seg + i + 3] = bfloat(g.w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 S[AK / 8];
        for (int i = 0; i < AK / 8; i++)
            S[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (int kf = 0; kf < 16; kf++) {
            simdgroup_bfloat8x8 qf;
            simdgroup_load(qf, &tQ[sgi * 8][kf * 8], 128);
            for (int ni = 0; ni < AK / 8; ni++) {
                simdgroup_bfloat8x8 kf8;
                simdgroup_load(kf8, &tK[ni * 8][kf * 8], 128, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S[ni], qf, kf8, S[ni]);
            }
        }
        for (int ni = 0; ni < AK / 8; ni++)
            simdgroup_store(S[ni], &u.s[sgi * 8][ni * 8], AK);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* One thread per row: exponentiate against the row maximum found in
         * the first pass, accumulate the denominator, and hand the block back
         * as the bf16 operand of the value product. */
        if (lin < AM) {
            const float mn = tMax[lin];
            float sum = 0.0f;
            for (int c = 0; c < AK; c++) {
                float p = (j0 + c < a.seq) ? exp(u.s[lin][c] * a.scale - mn) : 0.0f;
                sum += p;
                tP[lin][c] = bfloat(p);
            }
            tSum[lin] += sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int ni = 0; ni < AK / 8; ni++) {
            simdgroup_bfloat8x8 pf;
            simdgroup_load(pf, &tP[sgi * 8][ni * 8], AK);
            for (int nf = 0; nf < 16; nf++) {
                simdgroup_bfloat8x8 vf;
                simdgroup_load(vf, &tV[ni * 8][nf * 8], 128);
                simdgroup_multiply_accumulate(O[nf], pf, vf, O[nf]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int base = m0 + (int)sgi * 8;
    const int er = (int)slane >> 2, ec = (int)(slane & 3) * 32;
    const int eg = base + er;
    const float den = tSum[er + (int)sgi * 8] > 0.0f ? 1.0f / tSum[er + (int)sgi * 8] : 0.0f;
    if (base + 8 <= a.seq) {
        for (int nf = 0; nf < 16; nf++)
            simdgroup_store(O[nf], Ob + (long)base * a.ld + hoff + nf * 8, a.ld);
        /* The store writes through lanes chosen by the hardware; the scaling
         * below reads different columns of the same rows. */
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (eg < a.seq)
            for (int i = 0; i < 32; i++)
                Ob[(long)eg * a.ld + hoff + ec + i] *= den;
    } else if (base < a.seq) {
        /* Rows that run past the end share one staging block, then this
         * simdgroup's own lanes carry them out already scaled: every lane
         * writes the same columns it scales, so no cross-lane hand-off. */
        for (int nf = 0; nf < 16; nf++)
            simdgroup_store(O[nf], &u.o[0][nf * 8], 128);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (eg < a.seq)
            for (int i = 0; i < 32; i++)
                Ob[(long)eg * a.ld + hoff + ec + i] = u.o[er][ec + i] * den;
    }
}
)MSL";

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_q;
static id<MTLComputePipelineState> g_gemm;
static id<MTLComputePipelineState> g_attn;
static NSMutableArray<id<MTLBuffer>> *g_keep;   /* sparse adoptions: live for the process */
static int g_bad;

/* Shards adopted whole, so a projection inside one costs no view at all.
 * `buf` is nil for a mapping that could not be adopted; those fall back to the
 * per-call path. */
#define H3M_SPANS 32
typedef struct {
    const char *base;
    size_t len;
    id<MTLBuffer> buf;
} h3m_span;
static h3m_span g_span[H3M_SPANS];
static int g_spans;

/* Adopt `len` bytes at `p` as a no-copy buffer and report the page-aligned
 * lead the kernel has to skip. `keep` decides whether the handle survives the
 * call: weight mappings do, activations are released with the command buffer. */
static id<MTLBuffer> h3m_wrap(const void *p, size_t len, NSUInteger *off, int keep) {
    const uintptr_t page = 4096u;
    uintptr_t raw = (uintptr_t)p;
    uintptr_t base = raw & ~(page - 1u);
    size_t lead = (size_t)(raw - base);
    size_t need = ((len + lead + page - 1u) / page) * page;
    volatile const unsigned char *vp = (volatile const unsigned char *)base;
    unsigned char sink = 0;
    for (size_t o = 0; o < need; o += page) sink |= vp[o];
    if (sink == 0xff) { /* keeps the walk alive under -O2 */ }
    id<MTLBuffer> b = [g_dev newBufferWithBytesNoCopy:(void *)base
                                              length:need
                                             options:MTLResourceStorageModeShared
                                         deallocator:nil];
    if (!b) return nil;
    if (keep) [g_keep addObject:b];
    *off = lead;
    return b;
}

/* One view over a whole shard. The page walk still happens, but once per
 * process instead of once per projection: at 66 GB of weights per sampling
 * step the per-call form faults sixteen million pages a step and keeps a
 * handle for each, so step time grew with the number of steps taken. */
int h3m_adopt(const void *base, size_t len) {
    if (!g_dev || g_spans >= H3M_SPANS) return 0;
    if (!base || !len) return 0;
    for (int i = 0; i < g_spans; i++)
        if (g_span[i].base == (const char *)base) return 1;
    NSUInteger off = 0;
    id<MTLBuffer> b = h3m_wrap(base, len, &off, 1);
    if (!b || off != 0) return 0;   /* an mmap starts on a page boundary */
    g_span[g_spans].base = (const char *)base;
    g_span[g_spans].len = len;
    g_span[g_spans].buf = b;
    g_spans++;
    return 1;
}

/* Index of the adopted shard holding [p, p+n), or -1. */
static int h3m_span_idx(const void *p, size_t n) {
    uintptr_t q = (uintptr_t)p;
    for (int i = 0; i < g_spans; i++) {
        uintptr_t lo = (uintptr_t)g_span[i].base;
        if (q >= lo && q + n <= lo + g_span[i].len) return i;
    }
    return -1;
}

int h3m_init(void) {
    if (g_bad) return 0;
    if (g_dev) return 1;
    @autoreleasepool {
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) { g_bad = 1; return 0; }
        g_q = [g_dev newCommandQueue];
        g_keep = [NSMutableArray array];
        NSError *err = nil;
        MTLCompileOptions *opt = [MTLCompileOptions new];
        opt.languageVersion = MTLLanguageVersion3_1;
        id<MTLLibrary> lib = [g_dev newLibraryWithSource:
                                 [NSString stringWithUTF8String:h3m_shader_source()]
                                                 options:opt error:&err];
        if (!lib) {
            fprintf(stderr, "h3 gpu: shader compile failed: %s\n",
                    [[err localizedDescription] UTF8String]);
            g_bad = 1;
            return 0;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"gemm_bf16"];
        id<MTLFunction> fa = [lib newFunctionWithName:@"attn_fa"];
        if (!fn || !fa) { fprintf(stderr, "h3 gpu: kernel missing\n"); g_bad = 1; return 0; }
        g_attn = [g_dev newComputePipelineStateWithFunction:fa error:&err];
        g_gemm = [g_dev newComputePipelineStateWithFunction:fn error:&err];
        if (!g_gemm) {
            fprintf(stderr, "h3 gpu: pipeline failed: %s\n",
                    [[err localizedDescription] UTF8String]);
            g_bad = 1;
            return 0;
        }
    }
    return 1;
}

const char *h3m_shader_source(void) { return H3_MSL; }

void h3m_shutdown(void) {
    @autoreleasepool {
        g_gemm = nil;
        g_attn = nil;
        g_q = nil;
        for (int i = 0; i < g_spans; i++) { g_span[i].buf = nil; g_span[i].base = NULL; }
        g_spans = 0;
        [g_keep removeAllObjects];
        g_keep = nil;
        g_dev = nil;
    }
}

typedef struct { int rows, k, cols, lda, ldc; int64_t woff; } h3m_args;

int h3m_gemm(const float *a, const void *w, float *c,
             int rows, int k, int cols, int lda, int ldc) {
    if (!g_dev || !g_gemm) return 0;
    @autoreleasepool {
        NSUInteger ao = 0, wo = 0, co = 0;
        size_t wb = (size_t)k * (size_t)cols * sizeof(uint16_t);
        int si = h3m_span_idx(w, wb);
        long woff = 0;
        id<MTLBuffer> bw;
        if (si >= 0) {
            bw = g_span[si].buf;
            woff = (long)((uintptr_t)w - (uintptr_t)g_span[si].base);
        } else {
            bw = h3m_wrap(w, wb, &wo, 0);
        }
        if (!bw) return 0;
        id<MTLBuffer> ba = h3m_wrap(a, (size_t)lda * (size_t)rows * sizeof(float), &ao, 0);
        id<MTLBuffer> bc = h3m_wrap(c, (size_t)ldc * (size_t)rows * sizeof(float), &co, 0);
        if (!ba || !bc) return 0;

        h3m_args ar = {rows, k, cols, lda, ldc, woff};
        MTLSize grid = MTLSizeMake((NSUInteger)((cols + 63) / 64),
                                   (NSUInteger)((rows + 63) / 64), 1);
        MTLSize tgs = MTLSizeMake(H3_GPU_TG, 1, 1);
        id<MTLCommandBuffer> cb = [g_q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:g_gemm];
        [e setBuffer:ba offset:ao atIndex:0];
        [e setBuffer:bw offset:wo atIndex:1];
        [e setBuffer:bc offset:co atIndex:2];
        [e setBytes:&ar length:sizeof(ar) atIndex:3];
        [e dispatchThreadgroups:grid threadsPerThreadgroup:tgs];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        int ok = [cb status] == MTLCommandBufferStatusCompleted;
        /* The activation and output handles alias pages the caller owns;
         * dropping them before it frees those pages is the whole contract. */
        ba = nil; bc = nil;
        return ok;
    }
}


typedef struct { int seq, ld, hoff; float scale; } h3a_args;

int h3m_attention(const float *q, const float *k, const float *v, float *o,
                  int seq, int ld, int heads) {
    if (!g_dev || !g_attn) return 0;
    @autoreleasepool {
        NSUInteger qo = 0, ko = 0, vo = 0, oo = 0;
        id<MTLBuffer> bq = h3m_wrap(q, (size_t)ld * (size_t)seq * sizeof(float), &qo, 0);
        id<MTLBuffer> bk = h3m_wrap(k, (size_t)ld * (size_t)seq * sizeof(float), &ko, 0);
        id<MTLBuffer> bv = h3m_wrap(v, (size_t)ld * (size_t)seq * sizeof(float), &vo, 0);
        id<MTLBuffer> bo = h3m_wrap(o, (size_t)ld * (size_t)seq * sizeof(float), &oo, 0);
        if (!bq || !bk || !bv || !bo) return 0;

        h3a_args ar = {seq, ld, 128, 1.0f / sqrtf(128.0f)};
        MTLSize grid = MTLSizeMake((NSUInteger)((seq + 63) / 64), (NSUInteger)heads, 1);
        MTLSize tgs = MTLSizeMake(256, 1, 1);
        id<MTLCommandBuffer> cb = [g_q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:g_attn];
        [e setBuffer:bq offset:qo atIndex:0];
        [e setBuffer:bk offset:ko atIndex:1];
        [e setBuffer:bv offset:vo atIndex:2];
        [e setBuffer:bo offset:oo atIndex:3];
        [e setBytes:&ar length:sizeof(ar) atIndex:4];
        [e dispatchThreadgroups:grid threadsPerThreadgroup:tgs];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return [cb status] == MTLCommandBufferStatusCompleted;
    }
}
