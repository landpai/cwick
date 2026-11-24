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
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>

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

static float bf16_rn(float x) {
    uint u = as_type<uint>(x);
    u += ((u >> 15) & 1u) + ((u >> 16) & 1u);
    return as_type<float>(u & 0xffff0000u);
}

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

template<typename T, typename F>
static void gemm_core_small(device const float *A,
                            device const T *Wr,
                            device float *C,
                            constant gargs &a,
                            uint2 tg,
                            uint2 lid,
                            uint sgi,
                            threadgroup float (*tC)[TG_N],
                            threadgroup T (*tA)[TG_K],
                            threadgroup T (*tB)[TG_K]) {
    constexpr int TM = 16;
    device const T *W = (device const T *)((device const char *)Wr + a.woff);
    int m0 = (int)tg.y * TM, n0 = (int)tg.x * TG_N;
    int mg = (int)(sgi >> 1) * 8, ng = (int)(sgi & 1) * 32;
    int lin = (int)lid.x;
    simdgroup_float8x8 acc[1][4];
    for (int i = 0; i < 1; i++)
        for (int j = 0; j < 4; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k0 = 0; k0 < a.k; k0 += TG_K) {
        for (int pass = 0; pass < 2; pass++) {
            int idx = pass * 128 + lin;
            int r = idx / TG_K, c = idx % TG_K;
            int gm = m0 + r, gk = k0 + c;
            tA[r][c] = (gm < a.rows) ? T(A[(long)gm * a.lda + gk]) : T(0.0f);
        }
        for (int pass = 0; pass < 8; pass++) {
            int idx = pass * 128 + lin;
            int r = idx / TG_K, c = idx % TG_K;
            int gn = n0 + r, gk = k0 + c;
            tB[r][c] = (gn < a.cols) ? W[(long)gn * a.k + gk] : T(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kf = 0; kf < TG_K / 8; kf++) {
            F af[1], bf[4];
            for (int mi = 0; mi < 1; mi++)
                simdgroup_load(af[mi], &tA[mg + mi * 8][kf * 8], TG_K);
            for (int ni = 0; ni < 4; ni++)
                simdgroup_load(bf[ni], &tB[ng + ni * 8][kf * 8], TG_K,
                               ulong2(0, 0), true);
            for (int mi = 0; mi < 1; mi++)
                for (int ni = 0; ni < 4; ni++)
                    simdgroup_multiply_accumulate(acc[mi][ni], af[mi], bf[ni],
                                                  acc[mi][ni]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int mi = 0; mi < 1; mi++)
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

kernel void gemm_bf16_small(device const float *A [[buffer(0)]],
                            device const bfloat *Wr [[buffer(1)]],
                            device float *C [[buffer(2)]],
                            constant gargs &a [[buffer(3)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint2 lid [[thread_position_in_threadgroup]],
                            uint sgi [[simdgroup_index_in_threadgroup]]) {
    threadgroup float tC[16][TG_N];
    threadgroup bfloat tA[16][TG_K];
    threadgroup bfloat tB[TG_N][TG_K];
    gemm_core_small<bfloat, simdgroup_bfloat8x8>(A, Wr, C, a, tg, lid, sgi,
                                                tC, tA, tB);
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

struct dargs {
    int pos, nh, kvh, d, dr, extent, window, nk, j0, score_stride, cache_rows;
    float scale;
};

/* One threadgroup per query head. Each simdgroup computes 32 key logits;
 * every lane then walks the compact score row for softmax and one output
 * channel for the value sum. The sink is the extra logit-zero entry. */
kernel void inkling_decode_attn(device const float *Q [[buffer(0)]],
                                device const float *K [[buffer(1)]],
                                device const float *V [[buffer(2)]],
                                device const float *R [[buffer(3)]],
                                device const float *Rel [[buffer(4)]],
                                device const float *Sink [[buffer(5)]],
                                device float *O [[buffer(6)]],
                                device float *Scores [[buffer(7)]],
                                constant dargs &a [[buffer(8)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint2 lid [[thread_position_in_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4];
    const int h = (int)tg.x;
    const int kv = h / (a.nh / a.kvh);
    const int ld = a.kvh * a.d;
    device float *score = Scores + (ulong)h * a.score_stride;
    int j = a.j0 + (int)sg * 32 + (int)lane;
    float dot = 0.0f;
    if (j <= a.pos) {
        int slot = a.window > 0 ? j % a.cache_rows : j;
        for (int d = 0; d < a.d; d++)
            dot = fma(Q[(ulong)h * a.d + d],
                      K[(ulong)slot * ld + kv * a.d + d], dot);
    }
    float logit = -INFINITY;
    if (j <= a.pos) {
        int dist = a.pos - j;
        float b = 0.0f;
        if (dist < a.extent) {
            for (int r = 0; r < a.dr; r++)
                b = fma(R[(ulong)h * a.dr + r],
                        Rel[(ulong)r * a.extent + dist], b);
            b = bf16_rn(b);
        }
        logit = dot * a.scale + b - Sink[h];
        score[j - a.j0] = logit;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mx = 0.0f;
    for (int i = (int)lane; i < a.nk; i += 128)
        mx = max(mx, score[i]);
    mx = simd_max(mx);
    if (lane == 0u) red[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u && sg == 0u) {
        float m = max(max(red[0], red[1]), max(red[2], red[3]));
        red[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = red[0];

    float sum = 0.0f;
    for (int i = (int)lane; i < a.nk; i += 128) {
        float p = exp(score[i] - mx);
        score[i] = p;
        sum += p;
    }
    sum = simd_sum(sum);
    if (lane == 0u) red[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u && sg == 0u)
        red[0] = red[0] + red[1] + red[2] + red[3] + exp(-mx);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = 1.0f / red[0];

    int d = (int)lane;
    if (d < a.d) {
        float acc = 0.0f;
        for (int i = 0; i < a.nk; i++) {
            int key = a.j0 + i;
            int slot = a.window > 0 ? key % a.cache_rows : key;
            acc = fma(score[i] * inv, V[(ulong)slot * ld + kv * a.d + d], acc);
        }
        O[(ulong)h * a.d + d] = acc;
    }
}

struct pargs {
    int qstart, qrows, kvlen, nh, kvh, d, dr, extent, window, cache_rows;
    float scale;
};

/* One threadgroup per head for a prefill tile. Scores live in a reusable
 * device buffer so long contexts do not consume threadgroup memory. */
kernel void inkling_prefill_attn(device const float *Q [[buffer(0)]],
                                 device const float *K [[buffer(1)]],
                                 device const float *V [[buffer(2)]],
                                 device const float *R [[buffer(3)]],
                                 device const float *Rel [[buffer(4)]],
                                 device const float *Sink [[buffer(5)]],
                                 device float *O [[buffer(6)]],
                                 device float *Scores [[buffer(7)]],
                                 constant pargs &a [[buffer(8)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint2 tid [[thread_position_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[64];
    const int h = (int)tg.x;
    const int kv = h / (a.nh / a.kvh);
    const int ld = a.kvh * a.d;
    device float *score = Scores + (ulong)h * a.qrows * a.kvlen;
    device float *prob = Scores + (ulong)a.nh * a.qrows * a.kvlen +
                         (ulong)h * a.qrows * a.kvlen;
    for (int pair = (int)tid.x; pair < a.qrows * a.kvlen; pair += 256) {
        int q = pair / a.kvlen, key = pair % a.kvlen;
        int pos = a.qstart + q;
        bool valid = key <= pos && (a.window <= 0 || key >= pos + 1 - a.window);
        float dot = 0.0f;
        if (valid) {
            int slot = a.window > 0 ? key % a.cache_rows : key;
            for (int d = 0; d < a.d; d++)
                dot = fma(Q[((ulong)q * a.nh + h) * a.d + d],
                          K[(ulong)slot * ld + kv * a.d + d], dot);
        }
        float logit = -INFINITY;
        float bias = 0.0f;
        if (valid) {
            int dist = pos - key;
            if (dist < a.extent) {
                for (int r = 0; r < a.dr; r++)
                    bias = fma(R[((ulong)q * a.nh + h) * a.dr + r],
                               Rel[(ulong)r * a.extent + dist], bias);
                bias = bf16_rn(bias);
            }
            logit = dot * a.scale + bias - Sink[(ulong)q * a.nh + h];
        }
        score[pair] = logit;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid.x < (uint)a.qrows) {
        float mx = 0.0f;
        for (int key = 0; key < a.kvlen; key++) mx = max(mx, score[(ulong)tid.x * a.kvlen + key]);
        red[tid.x] = mx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid.x < (uint)a.qrows) {
        float mx = red[tid.x], sum = exp(-mx);
        device float *row = score + (ulong)tid.x * a.kvlen;
        device float *prow = prob + (ulong)tid.x * a.kvlen;
        for (int key = 0; key < a.kvlen; key++) {
            float p = exp(row[key] - mx);
            prow[key] = p;
            sum += p;
        }
        red[tid.x] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid.x < (uint)a.qrows)
        Scores[(ulong)a.nh * a.qrows * a.kvlen * 2 + (ulong)h * a.qrows + tid.x] = red[tid.x];

    for (int out = (int)tid.x; out < a.qrows * a.d; out += 256) {
        int q = out / a.d, d = out % a.d;
        device float *row = prob + (ulong)q * a.kvlen;
        float inv = 1.0f / red[q], acc = 0.0f;
        for (int key = 0; key < a.kvlen; key++) {
            int slot = a.window > 0 ? key % a.cache_rows : key;
            acc = fma(row[key] * inv, V[(ulong)slot * ld + kv * a.d + d], acc);
        }
        O[((ulong)q * a.nh + h) * a.d + d] = acc;
    }
}
)MSL";

const char *h3v_gpu_shader_source(void) { return H3VM_MSL; }

static double h3vm_now_s(void) {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e9;
}

/* ---- device ------------------------------------------------------- */
static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_q;
static id<MTLComputePipelineState> p_gemm_bf16, p_gemm_f32, p_gemm_qkv;
static id<MTLComputePipelineState> p_gemm_bf16_small;
static id<MTLComputePipelineState> p_attn128, p_attn64;
static id<MTLComputePipelineState> p_inkling_decode_attn;
static id<MTLComputePipelineState> p_inkling_prefill_attn;
static id<MTLBuffer> p_dec_scores;
static size_t p_dec_scores_cap;
static NSMutableArray<id<MTLBuffer>> *g_keep;
static int g_bad;

#define H3VM_SPANS 128
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
        p_gemm_bf16_small = [g_dev newComputePipelineStateWithFunction:
                          [lib newFunctionWithName:@"gemm_bf16_small"] error:&err];
        p_gemm_f32 = [g_dev newComputePipelineStateWithFunction:
                         [lib newFunctionWithName:@"gemm_f32"] error:&err];
        p_gemm_qkv = [g_dev newComputePipelineStateWithFunction:
                         [lib newFunctionWithName:@"gemm_bf16_qkv"] error:&err];
        p_attn128 = [g_dev newComputePipelineStateWithFunction:
                        [lib newFunctionWithName:@"attn128"] error:&err];
        p_attn64 = [g_dev newComputePipelineStateWithFunction:
                       [lib newFunctionWithName:@"attn64"] error:&err];
        p_inkling_decode_attn = [g_dev newComputePipelineStateWithFunction:
                       [lib newFunctionWithName:@"inkling_decode_attn"] error:&err];
        p_inkling_prefill_attn = [g_dev newComputePipelineStateWithFunction:
                       [lib newFunctionWithName:@"inkling_prefill_attn"] error:&err];
        if (!p_gemm_bf16 || !p_gemm_bf16_small || !p_gemm_f32 || !p_gemm_qkv ||
            !p_attn128 || !p_attn64 || !p_inkling_decode_attn || !p_inkling_prefill_attn) {
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
        p_gemm_bf16_small = nil;
        p_attn128 = nil; p_attn64 = nil; p_inkling_decode_attn = nil;
        p_inkling_prefill_attn = nil;
        p_dec_scores = nil; p_dec_scores_cap = 0;
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
        int tile_rows = pipe == p_gemm_bf16_small ? 16 : 64;
        MTLSize grid = MTLSizeMake((NSUInteger)((cols + 63) / 64),
                                   (NSUInteger)((rows + tile_rows - 1) / tile_rows), 1);
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

int h3vm_gemm_bf16_batch(const h3v_gpu_bf16_item *items, int n) {
    if (!g_dev || !p_gemm_bf16 || !p_gemm_bf16_small || !items || n <= 0 || n > 64) return 0;
    @autoreleasepool {
        id<MTLBuffer> ba[64], bw[64], bc[64];
        NSUInteger ao[64], wo[64], co[64];
        h3vm_args ar[64];
        for (int i = 0; i < n; i++) {
            const h3v_gpu_bf16_item *it = &items[i];
            if (it->rows <= 0 || it->k <= 0 || it->cols <= 0 ||
                it->k % 16 || it->lda < it->k || it->ldc < it->cols) return 0;
            size_t wb = (size_t)it->k * (size_t)it->cols * sizeof(uint16_t);
            int si = h3vm_span_idx(it->w, wb);
            if (si >= 0) {
                bw[i] = g_span[si].buf;
                wo[i] = (NSUInteger)((uintptr_t)it->w - (uintptr_t)g_span[si].base);
            } else {
                bw[i] = h3vm_wrap(it->w, wb, &wo[i], 0);
                if (!bw[i]) return 0;
            }
            ba[i] = h3vm_wrap(it->a, (size_t)it->lda * (size_t)it->rows * sizeof(float),
                              &ao[i], 0);
            bc[i] = h3vm_wrap(it->c, (size_t)it->ldc * (size_t)it->rows * sizeof(float),
                              &co[i], 0);
            if (!ba[i] || !bc[i]) return 0;
            ar[i] = (h3vm_args){it->rows, it->k, it->cols, it->lda, it->ldc, 0, -1, 0};
        }
        id<MTLCommandBuffer> cb = [g_q commandBuffer];
        id<MTLComputeCommandEncoder> e;
        if (@available(macOS 12.0, *))
            e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        else
            e = [cb computeCommandEncoder];
        for (int i = 0; i < n; i++) {
            [e setComputePipelineState:p_gemm_bf16];
            [e setBuffer:ba[i] offset:ao[i] atIndex:0];
            [e setBuffer:bw[i] offset:wo[i] atIndex:1];
            [e setBuffer:bc[i] offset:co[i] atIndex:2];
            [e setBytes:&ar[i] length:sizeof(ar[i]) atIndex:3];
            MTLSize grid = MTLSizeMake((NSUInteger)((items[i].cols + 63) / 64),
                                       (NSUInteger)((items[i].rows + 63) / 64), 1);
            [e dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(H3VM_TG, 1, 1)];
        }
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        int ok = [cb status] == MTLCommandBufferStatusCompleted && !cb.error;
        if (!ok)
            fprintf(stderr, "h3v gpu: BF16 batch failed: %s\n",
                    cb.error ? [[cb.error localizedDescription] UTF8String] : "command buffer error");
        return ok;
    }
}

int h3vm_gemm_bf16(const float *a, const void *w, float *c,
                   int rows, int k, int cols, int lda, int ldc) {
    return h3vm_gemm(a, w, c, rows, k, cols, cols, lda, ldc,
                     rows <= 16 ? p_gemm_bf16_small : p_gemm_bf16, 1, -1, 0);
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
typedef struct {
    int pos, nh, kvh, d, dr, extent, window, nk, j0, score_stride, cache_rows;
    float scale;
} h3vm_dargs;
typedef struct {
    int qstart, qrows, kvlen, nh, kvh, d, dr, extent, window, cache_rows;
    float scale;
} h3vm_pargs;

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
        [e dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return [cb status] == MTLCommandBufferStatusCompleted;
    }
}

int h3vm_inkling_decode_attn(const float *q, const float *k, const float *v,
                             const float *r, const float *rel, const float *sink,
                             float *o, int pos, int cache_rows, int ld,
                             int nh, int kvh, int d, int dr, int extent,
                             int window) {
    if (!g_dev || !p_inkling_decode_attn || pos < 0 ||
        (window <= 0 && cache_rows <= pos) ||
        d != 128 || dr <= 0 || nh <= 0 || kvh <= 0 || nh % kvh) return 0;
    int j1 = window > 0 && pos + 1 > window ? pos + 1 - window : 0;
    if (window <= 0) j1 = 0;
    int j0 = j1, nk = pos - j0 + 1;
    if (nk <= 0 || nk > cache_rows) return 0;
    const bool prof_time = getenv("CWICK_H3V_TIME") != NULL;
    const double prof_t0 = prof_time ? h3vm_now_s() : 0.0;
    @autoreleasepool {
        NSUInteger qo = 0, ko = 0, vo = 0, ro = 0, relo = 0, so = 0, oo = 0;
        size_t kvbytes = (size_t)cache_rows * (size_t)ld * sizeof(float);
        id<MTLBuffer> bq = h3vm_wrap(q, (size_t)nh * d * sizeof(float), &qo, 0);
        id<MTLBuffer> bk = h3vm_wrap(k, kvbytes, &ko, 0);
        id<MTLBuffer> bv = h3vm_wrap(v, kvbytes, &vo, 0);
        id<MTLBuffer> br = h3vm_wrap(r, (size_t)nh * dr * sizeof(float), &ro, 0);
        size_t relbytes = (size_t)dr * (size_t)extent * sizeof(float);
        int si = h3vm_span_idx(rel, relbytes);
        id<MTLBuffer> brel;
        if (si >= 0) {
            brel = g_span[si].buf;
            relo = (NSUInteger)((uintptr_t)rel - (uintptr_t)g_span[si].base);
        } else {
            brel = h3vm_wrap(rel, relbytes, &relo, 0);
        }
        id<MTLBuffer> bs = h3vm_wrap(sink, (size_t)nh * sizeof(float), &so, 0);
        id<MTLBuffer> bo = h3vm_wrap(o, (size_t)nh * d * sizeof(float), &oo, 0);
        if (!bq || !bk || !bv || !br || !brel || !bs || !bo) return 0;
        size_t score_bytes = (size_t)nh * (size_t)nk * sizeof(float);
        if (!p_dec_scores || p_dec_scores_cap < score_bytes) {
            size_t cap = p_dec_scores_cap ? p_dec_scores_cap : 65536;
            while (cap < score_bytes) cap *= 2;
            p_dec_scores = [g_dev newBufferWithLength:cap
                                              options:MTLResourceStorageModeShared];
            if (!p_dec_scores) return 0;
            p_dec_scores_cap = cap;
        }
        h3vm_dargs ar = {pos, nh, kvh, d, dr, extent, window, nk, j0, nk, cache_rows,
                         1.0f / sqrtf((float)d)};
        id<MTLCommandBuffer> cb = [g_q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p_inkling_decode_attn];
        [e setBuffer:bq offset:qo atIndex:0];
        [e setBuffer:bk offset:ko atIndex:1];
        [e setBuffer:bv offset:vo atIndex:2];
        [e setBuffer:br offset:ro atIndex:3];
        [e setBuffer:brel offset:relo atIndex:4];
        [e setBuffer:bs offset:so atIndex:5];
        [e setBuffer:bo offset:oo atIndex:6];
        [e setBuffer:p_dec_scores offset:0 atIndex:7];
        [e setBytes:&ar length:sizeof(ar) atIndex:8];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)nh, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [e endEncoding];
        const double prof_t1 = prof_time ? h3vm_now_s() : 0.0;
        [cb commit];
        [cb waitUntilCompleted];
        if (prof_time)
            fprintf(stderr, "H3VATTN host=%.1fus wait=%.1fus gpu=%.1fus pos=%d\n",
                    (prof_t1 - prof_t0) * 1e6,
                    (h3vm_now_s() - prof_t1) * 1e6,
                    (cb.GPUEndTime - cb.GPUStartTime) * 1e6, pos);
        return [cb status] == MTLCommandBufferStatusCompleted && !cb.error;
    }
}

int h3vm_inkling_prefill_attn(const float *q, const float *k, const float *v,
                              const float *r, const float *rel, const float *sink,
                              float *o, int qstart, int qrows, int cache_rows,
                              int ld, int nh, int kvh, int d, int dr,
                              int extent, int window) {
    if (!g_dev || !p_inkling_prefill_attn || qstart < 0 || qrows <= 0 ||
        qrows > 64 || d != 128 || dr <= 0 || nh <= 0 || kvh <= 0 || nh % kvh)
        return 0;
    int kvlen = qstart + qrows;
    if ((window <= 0 && cache_rows < kvlen) ||
        (window > 0 && cache_rows <= 0)) return 0;
    @autoreleasepool {
        NSUInteger qo = 0, ko = 0, vo = 0, ro = 0, relo = 0, so = 0, oo = 0;
        size_t kvbytes = (size_t)cache_rows * (size_t)ld * sizeof(float);
        id<MTLBuffer> bq = h3vm_wrap(q, (size_t)qrows * nh * d * sizeof(float), &qo, 0);
        id<MTLBuffer> bk = h3vm_wrap(k, kvbytes, &ko, 0);
        id<MTLBuffer> bv = h3vm_wrap(v, kvbytes, &vo, 0);
        id<MTLBuffer> br = h3vm_wrap(r, (size_t)qrows * nh * dr * sizeof(float), &ro, 0);
        size_t relbytes = (size_t)dr * (size_t)extent * sizeof(float);
        int si = h3vm_span_idx(rel, relbytes);
        id<MTLBuffer> brel;
        if (si >= 0) {
            brel = g_span[si].buf;
            relo = (NSUInteger)((uintptr_t)rel - (uintptr_t)g_span[si].base);
        } else {
            brel = h3vm_wrap(rel, relbytes, &relo, 0);
        }
        id<MTLBuffer> bs = h3vm_wrap(sink, (size_t)qrows * nh * sizeof(float), &so, 0);
        id<MTLBuffer> bo = h3vm_wrap(o, (size_t)qrows * nh * d * sizeof(float), &oo, 0);
        if (!bq || !bk || !bv || !br || !brel || !bs || !bo) return 0;
        size_t score_bytes = (size_t)nh * (size_t)qrows *
                             ((size_t)kvlen * 2 + 1) * sizeof(float);
        if (!p_dec_scores || p_dec_scores_cap < score_bytes) {
            size_t cap = p_dec_scores_cap ? p_dec_scores_cap : 65536;
            while (cap < score_bytes) cap *= 2;
            p_dec_scores = [g_dev newBufferWithLength:cap
                                              options:MTLResourceStorageModeShared];
            if (!p_dec_scores) return 0;
            p_dec_scores_cap = cap;
        }
        h3vm_pargs ar = {qstart, qrows, kvlen, nh, kvh, d, dr, extent, window,
                         cache_rows, 1.0f / sqrtf((float)d)};
        id<MTLCommandBuffer> cb = [g_q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p_inkling_prefill_attn];
        [e setBuffer:bq offset:qo atIndex:0];
        [e setBuffer:bk offset:ko atIndex:1];
        [e setBuffer:bv offset:vo atIndex:2];
        [e setBuffer:br offset:ro atIndex:3];
        [e setBuffer:brel offset:relo atIndex:4];
        [e setBuffer:bs offset:so atIndex:5];
        [e setBuffer:bo offset:oo atIndex:6];
        [e setBuffer:p_dec_scores offset:0 atIndex:7];
        [e setBytes:&ar length:sizeof(ar) atIndex:8];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)nh, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return [cb status] == MTLCommandBufferStatusCompleted && !cb.error;
    }
}
