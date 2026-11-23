/* h3v_gpu.m — Metal device, pipelines and kernels for video generation.
 *
 * Three kernels cover the whole compute cost of a clip:
 *
 *   gemm_bf16   DiT projections. The checkpoint's BF16 bytes stay in the mmap
 *               and are staged as BF16 tiles; the activation is staged to BF16
 *               as well and the product accumulates in FP32 fragments.
 *   gemm_f32    VAE projections. Those weights are F32 and stay F32 through
 *               the fragment product, so the decoder keeps its checkpoint
 *               precision instead of inheriting the DiT's BF16 activation.
 *   attn        One tiled flash kernel instantiated for 128-wide DiT heads
 *               and 64-wide decoder heads. Neither materialises a score
 *               matrix wider than one key block.
 *
 * Weight shards are adopted as no-copy buffers over the pages the loader
 * already mapped. A no-copy buffer only works when the pages are in the
 * GPU page table, so the adopt walk touches one byte per page first.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "h3v_gpu.h"

#define H3VM_TG 128

static const char *H3VM_MSL = R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define TG_M 64
#define TG_N 64
#define TG_K 16
#define AM 64
#define AK 16

/* ---- projection kernels ------------------------------------------- */
struct gargs { int rows, k, cols, lda, ldc; long woff; int qpart, qhd; };

/* The release shards store a fused QKV projection one head at a time as
 * [q | k | v]. Select one plane at GEMM time so no separate copy of the
 * 21504-row weight tile, and no post-pass over the output, is needed. */
inline int weight_row(int col, int qpart, int qhd) {
    return (col / qhd) * 3 * qhd + qpart * qhd + (col % qhd);
}

/* out[rows][cols] = A[rows][k] @ W[cols][k]^T. Both operands are staged in
 * the checkpoint's dtype; the accumulator is FP32. */
template<typename T, typename F>
static void gemm_core(device const float *A,
                      device const T *Wr,
                      device float *C,
                      constant gargs &a,
                      uint2 tg,
                      uint2 lid,
                      uint sgi,
                      threadgroup float (*tC)[TG_N],
                      threadgroup T (*tA)[TG_K],
                      threadgroup T (*tB)[TG_K]) {
    device const T *W = (device const T *)((device const char *)Wr + a.woff);

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
            int wr = (a.qpart >= 0) ? weight_row(gn, a.qpart, a.qhd) : gn;
            int wlim = (a.qpart >= 0) ? (a.cols / a.qhd) * 3 * a.qhd : a.cols;
            T av = (gm < a.rows) ? T(A[(long)gm * a.lda + gk]) : T(0.0f);
            T bv = (wr < wlim) ? W[(long)wr * a.k + gk] : T(0.0f);
            tA[r][c] = av;
            tB[r][c] = bv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kf = 0; kf < TG_K / 8; kf++) {
            F af[4], bf[4];
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
    gemm_core<bfloat, simdgroup_bfloat8x8>(A, Wr, C, a, tg, lid, sgi, tC, tA, tB);
}

kernel void gemm_bf16_qkv(device const float *A [[buffer(0)]],
                          device const bfloat *Wr [[buffer(1)]],
                          device float *C [[buffer(2)]],
                          constant gargs &a [[buffer(3)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint2 lid [[thread_position_in_threadgroup]],
                          uint sgi [[simdgroup_index_in_threadgroup]]) {
    threadgroup float tC[TG_M][TG_N];
    threadgroup bfloat tA[TG_M][TG_K];
    threadgroup bfloat tB[TG_N][TG_K];
    gemm_core<bfloat, simdgroup_bfloat8x8>(A, Wr, C, a, tg, lid, sgi, tC, tA, tB);
}

kernel void gemm_f32(device const float *A [[buffer(0)]],
                     device const float *Wr [[buffer(1)]],
                     device float *C [[buffer(2)]],
                     constant gargs &a [[buffer(3)]],
                     uint2 tg [[threadgroup_position_in_grid]],
                     uint2 lid [[thread_position_in_threadgroup]],
                     uint sgi [[simdgroup_index_in_threadgroup]]) {
    threadgroup float tC[TG_M][TG_N];
    threadgroup float tA[TG_M][TG_K];
    threadgroup float tB[TG_N][TG_K];
    gemm_core<float, simdgroup_float8x8>(A, Wr, C, a, tg, lid, sgi, tC, tA, tB);
}

/* ---- attention ---------------------------------------------------- *
 * One threadgroup owns a 64-query-row block of one head. The key range is
 * walked twice: once for the row maximum, once for the exponent and the
 * value product. A streaming single pass would have to rescale the output
 * fragments when a new maximum arrives, which fragments cannot do
 * element-wise. NF is HD / 8, the number of 8-wide fragments per head.
 */
struct aargs { int seq, ld, hoff; float scale; };

template<int HD, int NF>
static void attn_core(device const float *Q,
                      device const float *Kb,
                      device const float *Vb,
                      device float *Ob,
                      constant aargs &a,
                      uint2 tg,
                      uint2 lid,
                      uint sgi,
                      uint slane,
                      threadgroup bfloat (*tQ)[HD],
                      threadgroup bfloat (*tK)[HD],
                      threadgroup bfloat (*tV)[HD],
                      threadgroup float (*sc)[AK],
                      threadgroup float (*oc)[HD],
                      threadgroup bfloat (*tP)[AK],
                      threadgroup float *tSum,
                      threadgroup float *tMax) {
    const int QSPAN = HD / 32;   /* float4 loads per row while staging Q */
    const int KSPAN = HD / 8;    /* float4 loads per row while staging K/V */

    const int lin = (int)lid.x;
    const int m0 = (int)tg.x * AM;
    const int hoff = (int)tg.y * a.hoff;

    if (lin < AM) { tSum[lin] = 0.0f; tMax[lin] = -INFINITY; }
    {
        int r = lin / QSPAN, seg = (lin % QSPAN) * 32;
        int gm = m0 + r;
        if (r < AM)
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

    for (int j0 = 0; j0 < a.seq; j0 += AK) {
        int r = lin / KSPAN, seg = (lin % KSPAN) * 8;
        int gn = j0 + r;
        if (r < AK)
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
        for (int kf = 0; kf < NF; kf++) {
            simdgroup_bfloat8x8 qf;
            simdgroup_load(qf, &tQ[sgi * 8][kf * 8], HD);
            for (int ni = 0; ni < AK / 8; ni++) {
                simdgroup_bfloat8x8 kf8;
                simdgroup_load(kf8, &tK[ni * 8][kf * 8], HD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S[ni], qf, kf8, S[ni]);
            }
        }
        for (int ni = 0; ni < AK / 8; ni++)
            simdgroup_store(S[ni], &sc[sgi * 8][ni * 8], AK);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lin < AM) {
            float bm = tMax[lin];
            for (int c = 0; c < AK; c++)
                if (j0 + c < a.seq) {
                    float v = sc[lin][c] * a.scale;
                    if (v > bm) bm = v;
                }
            tMax[lin] = bm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_float8x8 O[NF];
    for (int i = 0; i < NF; i++)
        O[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int j0 = 0; j0 < a.seq; j0 += AK) {
        int r = lin / KSPAN, seg = (lin % KSPAN) * 8;
        int gn = j0 + r;
        if (r < AK)
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
        for (int kf = 0; kf < NF; kf++) {
            simdgroup_bfloat8x8 qf;
            simdgroup_load(qf, &tQ[sgi * 8][kf * 8], HD);
            for (int ni = 0; ni < AK / 8; ni++) {
                simdgroup_bfloat8x8 kf8;
                simdgroup_load(kf8, &tK[ni * 8][kf * 8], HD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S[ni], qf, kf8, S[ni]);
            }
        }
        for (int ni = 0; ni < AK / 8; ni++)
            simdgroup_store(S[ni], &sc[sgi * 8][ni * 8], AK);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lin < AM) {
            const float mn = tMax[lin];
            float sum = 0.0f;
            for (int c = 0; c < AK; c++) {
                float p = (j0 + c < a.seq) ? exp(sc[lin][c] * a.scale - mn) : 0.0f;
                sum += p;
                tP[lin][c] = bfloat(p);
            }
            tSum[lin] += sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int ni = 0; ni < AK / 8; ni++) {
            simdgroup_bfloat8x8 pf;
            simdgroup_load(pf, &tP[sgi * 8][ni * 8], AK);
            for (int nf = 0; nf < NF; nf++) {
                simdgroup_bfloat8x8 vf;
                simdgroup_load(vf, &tV[ni * 8][nf * 8], HD);
                simdgroup_multiply_accumulate(O[nf], pf, vf, O[nf]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int base = m0 + (int)sgi * 8;
    /* The fragment store lays an 8-row x HD-column slice across the 32 lanes
     * of the simdgroup as four column blocks of HD/4 each. Deriving the block
     * width from HD keeps the rescale in step for both 128- and 64-wide heads. */
    const int ECOL = HD / 4;
    const int er = (int)slane / 4, ec = (int)((int)slane % 4) * ECOL;
    const int eg = base + er;
    const float den = tSum[er + (int)sgi * 8] > 0.0f
                        ? 1.0f / tSum[er + (int)sgi * 8] : 0.0f;
    if (base + 8 <= a.seq) {
        for (int nf = 0; nf < NF; nf++)
            simdgroup_store(O[nf], Ob + (long)base * a.ld + hoff + nf * 8, a.ld);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (eg < a.seq)
            for (int i = 0; i < ECOL; i++)
                Ob[(long)eg * a.ld + hoff + ec + i] *= den;
    } else if (base < a.seq) {
        for (int nf = 0; nf < NF; nf++)
            simdgroup_store(O[nf], &oc[0][nf * 8], HD);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (eg < a.seq)
            for (int i = 0; i < ECOL; i++)
                Ob[(long)eg * a.ld + hoff + ec + i] = oc[er][ec + i] * den;
    }
}

kernel void attn128(device const float *Q [[buffer(0)]],
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
    threadgroup union { float s[AM][AK]; float o[8][128]; } u;
    threadgroup bfloat tP[AM][AK];
    threadgroup float tSum[AM];
    threadgroup float tMax[AM];
    attn_core<128, 16>(Q, Kb, Vb, Ob, a, tg, lid, sgi, slane,
                       tQ, tK, tV, u.s, u.o, tP, tSum, tMax);
}

kernel void attn64(device const float *Q [[buffer(0)]],
                   device const float *Kb [[buffer(1)]],
                   device const float *Vb [[buffer(2)]],
                   device float *Ob [[buffer(3)]],
                   constant aargs &a [[buffer(4)]],
                   uint2 tg [[threadgroup_position_in_grid]],
                   uint2 lid [[thread_position_in_threadgroup]],
                   uint sgi [[simdgroup_index_in_threadgroup]],
                   uint slane [[thread_index_in_simdgroup]]) {
    threadgroup bfloat tQ[AM][64];
    threadgroup bfloat tK[AK][64];
    threadgroup bfloat tV[AK][64];
    threadgroup union { float s[AM][AK]; float o[8][64]; } u;
    threadgroup bfloat tP[AM][AK];
    threadgroup float tSum[AM];
    threadgroup float tMax[AM];
    attn_core<64, 8>(Q, Kb, Vb, Ob, a, tg, lid, sgi, slane,
                     tQ, tK, tV, u.s, u.o, tP, tSum, tMax);
}
)MSL";

const char *h3v_gpu_shader_source(void) { return H3VM_MSL; }

/* ---- device ------------------------------------------------------- */
static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_q;
static id<MTLComputePipelineState> p_gemm_bf16, p_gemm_f32, p_gemm_qkv;
static id<MTLComputePipelineState> p_attn128, p_attn64;
static NSMutableArray<id<MTLBuffer>> *g_keep;
static int g_bad;

#define H3VM_SPANS 32
typedef struct {
    const char *base;
    size_t len;
    id<MTLBuffer> buf;
} h3vm_span;
static h3vm_span g_span[H3VM_SPANS];
static int g_spans;

/* Wrap a CPU address range as a shared no-copy buffer. The lead is the
 * page-aligned prefix the kernel must skip; safetensors tensor offsets are
 * byte-granular and cannot be used as a Metal buffer offset. */
static id<MTLBuffer> h3vm_wrap(const void *p, size_t len, NSUInteger *off, int keep) {
    const uintptr_t page = 4096u;
    uintptr_t raw = (uintptr_t)p;
    uintptr_t base = raw & ~(page - 1u);
    size_t lead = (size_t)(raw - base);
    size_t need = ((len + lead + page - 1u) / page) * page;
    volatile const unsigned char *vp = (volatile const unsigned char *)base;
    unsigned char sink = 0;
    for (size_t o = 0; o < need; o += page) sink |= vp[o];
    if (sink == 0xff) { }
    id<MTLBuffer> b = [g_dev newBufferWithBytesNoCopy:(void *)base
                                              length:need
                                             options:MTLResourceStorageModeShared
                                         deallocator:nil];
    if (!b) return nil;
    if (keep) [g_keep addObject:b];
    *off = lead;
    return b;
}

void h3vm_adopt(const void *base, size_t len) {
    if (!g_dev || g_spans >= H3VM_SPANS || !base || !len) return;
    for (int i = 0; i < g_spans; i++)
        if (g_span[i].base == (const char *)base) return;
    NSUInteger off = 0;
    id<MTLBuffer> b = h3vm_wrap(base, len, &off, 1);
    if (!b || off != 0) return;    /* mmap starts on a page boundary */
    g_span[g_spans].base = (const char *)base;
    g_span[g_spans].len = len;
    g_span[g_spans].buf = b;
    g_spans++;
}

static int h3vm_span_idx(const void *p, size_t n) {
    uintptr_t q = (uintptr_t)p;
    for (int i = 0; i < g_spans; i++) {
        uintptr_t lo = (uintptr_t)g_span[i].base;
        if (q >= lo && q + n <= lo + g_span[i].len) return i;
    }
    return -1;
}

int h3vm_init(void) {
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
                                 [NSString stringWithUTF8String:H3VM_MSL]
                                                 options:opt error:&err];
        if (!lib) {
            fprintf(stderr, "h3v gpu: shader compile failed: %s\n",
                    [[err localizedDescription] UTF8String]);
            g_bad = 1; return 0;
        }
        p_gemm_bf16 = [g_dev newComputePipelineStateWithFunction:
                          [lib newFunctionWithName:@"gemm_bf16"] error:&err];
        p_gemm_f32 = [g_dev newComputePipelineStateWithFunction:
                         [lib newFunctionWithName:@"gemm_f32"] error:&err];
        p_gemm_qkv = [g_dev newComputePipelineStateWithFunction:
                         [lib newFunctionWithName:@"gemm_bf16_qkv"] error:&err];
        p_attn128 = [g_dev newComputePipelineStateWithFunction:
                        [lib newFunctionWithName:@"attn128"] error:&err];
        p_attn64 = [g_dev newComputePipelineStateWithFunction:
                       [lib newFunctionWithName:@"attn64"] error:&err];
        if (!p_gemm_bf16 || !p_gemm_f32 || !p_gemm_qkv || !p_attn128 || !p_attn64) {
            fprintf(stderr, "h3v gpu: pipeline creation failed: %s\n",
                    [[err localizedDescription] UTF8String]);
            g_bad = 1; return 0;
        }
    }
    return 1;
}

void h3vm_shutdown(void) {
    @autoreleasepool {
        p_gemm_bf16 = nil; p_gemm_f32 = nil; p_gemm_qkv = nil;
        p_attn128 = nil; p_attn64 = nil;
        g_q = nil;
        for (int i = 0; i < g_spans; i++) { g_span[i].buf = nil; g_span[i].base = NULL; }
        g_spans = 0;
        [g_keep removeAllObjects];
        g_keep = nil;
        g_dev = nil;
    }
}

/* ---- dispatch ----------------------------------------------------- */
typedef struct { int rows, k, cols, lda, ldc; int64_t woff; int qpart, qhd; } h3vm_args;

static int h3vm_gemm(const float *a, const void *w, float *c,
                     int rows, int k, int cols, int wrows, int lda, int ldc,
                     id<MTLComputePipelineState> pipe, int bf16,
                     int qpart, int qhd) {
    if (!g_dev || !pipe) return 0;
    @autoreleasepool {
        NSUInteger ao = 0, wo = 0, co = 0;
        size_t unit = bf16 ? sizeof(uint16_t) : sizeof(float);
        size_t wb = (size_t)k * (size_t)wrows * unit;
        int si = h3vm_span_idx(w, wb);
        int64_t woff = 0;
        id<MTLBuffer> bw;
        if (si >= 0) {
            bw = g_span[si].buf;
            woff = (int64_t)((uintptr_t)w - (uintptr_t)g_span[si].base);
        } else {
            bw = h3vm_wrap(w, wb, &wo, 0);
            woff = 0;
        }
        if (!bw) return 0;
        id<MTLBuffer> ba = h3vm_wrap(a, (size_t)lda * (size_t)rows * sizeof(float), &ao, 0);
        id<MTLBuffer> bc = h3vm_wrap(c, (size_t)ldc * (size_t)rows * sizeof(float), &co, 0);
        if (!ba || !bc) return 0;

        h3vm_args ar = {rows, k, cols, lda, ldc, woff, qpart, qhd};
        MTLSize grid = MTLSizeMake((NSUInteger)((cols + 63) / 64),
                                   (NSUInteger)((rows + 63) / 64), 1);
        id<MTLCommandBuffer> cb = [g_q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pipe];
        [e setBuffer:ba offset:ao atIndex:0];
        [e setBuffer:bw offset:wo atIndex:1];
        [e setBuffer:bc offset:co atIndex:2];
        [e setBytes:&ar length:sizeof(ar) atIndex:3];
        [e dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(H3VM_TG, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        int ok = [cb status] == MTLCommandBufferStatusCompleted;
        ba = nil; bc = nil;
        return ok;
    }
}

int h3vm_gemm_bf16(const float *a, const void *w, float *c,
                   int rows, int k, int cols, int lda, int ldc) {
    return h3vm_gemm(a, w, c, rows, k, cols, cols, lda, ldc,
                     p_gemm_bf16, 1, -1, 0);
}

int h3vm_gemm_bf16_qkv(const float *a, const void *w, float *c,
                       int rows, int k, int cols, int wrows, int lda, int ldc,
                       int part, int hd) {
    return h3vm_gemm(a, w, c, rows, k, cols, wrows, lda, ldc,
                     p_gemm_qkv, 1, part, hd);
}

int h3vm_gemm_f32(const float *a, const float *w, float *c,
                  int rows, int k, int cols, int lda, int ldc) {
    return h3vm_gemm(a, w, c, rows, k, cols, cols, lda, ldc,
                     p_gemm_f32, 0, -1, 0);
}

typedef struct { int seq, ld, hoff; float scale; } h3vm_aargs;

int h3vm_attention(const float *q, const float *k, const float *v, float *o,
                   int seq, int ld, int heads, int hd) {
    if (!g_dev) return 0;
    id<MTLComputePipelineState> pipe = (hd == 128) ? p_attn128 : p_attn64;
    if (!pipe) return 0;
    @autoreleasepool {
        NSUInteger qo = 0, ko = 0, vo = 0, oo = 0;
        size_t bytes = (size_t)ld * (size_t)seq * sizeof(float);
        id<MTLBuffer> bq = h3vm_wrap(q, bytes, &qo, 0);
        id<MTLBuffer> bk = h3vm_wrap(k, bytes, &ko, 0);
        id<MTLBuffer> bv = h3vm_wrap(v, bytes, &vo, 0);
        id<MTLBuffer> bo = h3vm_wrap(o, bytes, &oo, 0);
        if (!bq || !bk || !bv || !bo) return 0;

        h3vm_aargs ar = {seq, ld, hd, 1.0f / sqrtf((float)hd)};
        MTLSize grid = MTLSizeMake((NSUInteger)((seq + 63) / 64),
                                   (NSUInteger)heads, 1);
        id<MTLCommandBuffer> cb = [g_q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pipe];
        [e setBuffer:bq offset:qo atIndex:0];
        [e setBuffer:bk offset:ko atIndex:1];
        [e setBuffer:bv offset:vo atIndex:2];
        [e setBuffer:bo offset:oo atIndex:3];
        [e setBytes:&ar length:sizeof(ar) atIndex:4];
        /* Eight simdgroups, eight rows each: one threadgroup covers the
         * kernel's 64 query rows. The GEMM uses four simdgroups instead. */
        [e dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return [cb status] == MTLCommandBufferStatusCompleted;
    }
}
