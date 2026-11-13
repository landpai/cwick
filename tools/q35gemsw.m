/* q35gemsw — sweep the k-row weight pass over "rows per simdgroup" (RP).
 *
 * tools/q35gemt established that staging the activation rows in threadgroup
 * memory turns k weight passes into one pass plus arithmetic: 38.5 ms at k=1,
 * 69.9 ms at k=8 over this shape mix. But the marginal row still costs
 * 4.5 ms, and the arithmetic does not explain it: 8 rows over 13.82 GB is
 * 110 GFLOP in 69.9 ms, i.e. 1.6 TFLOP/s, about a tenth of the FP32 FMA peak.
 *
 * The suspect is instruction mix, not ALU. One simdgroup owning one weight row
 * issues, per 4 weight elements, 2 threadgroup x loads and 4 FMAs -- a 1:2
 * load-to-FMA ratio, and threadgroup load issue is the scarcer resource.
 * Nothing about the math requires that ratio: the same staged x is valid for
 * any number of weight rows. So give each simdgroup RP of them and hold x in
 * registers across all RP, which makes the ratio RP:2.
 *
 * Cost of RP: RP * N live accumulators. At N=8, RP=4 that is 32 floats plus
 * 4*N activation registers, which still fits the register file, and beyond
 * that it should start spilling.
 *
 *   make tools/q35gemsw && ./tools/q35gemsw [layers] [reps]
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define D 4096
#define QW 8192
#define KV 1024
#define QKV 8192
#define ZD 4096
#define I 12288
#define MAXK 8
#define NT 256        /* threads per threadgroup */
#define NSG 8         /* simdgroups per threadgroup == NT / 32 */
#define TILE 512      /* k elements staged per threadgroup pass */

typedef struct { const char *name; uint32_t k, cols; } proj;

enum { S_QLIN, S_ZLIN, S_QATTN, S_KATTN, S_VATTN, S_GATE, S_UP, S_DOWN, S_OUT, S_N };

static const proj SLAB[S_N] = {
    [S_QLIN]  = {"in_proj_qkv", D, QKV},
    [S_ZLIN]  = {"in_proj_z",   D, ZD},
    [S_QATTN] = {"q_proj",      D, QW},
    [S_KATTN] = {"k_proj",      D, KV},
    [S_VATTN] = {"v_proj",      D, KV},
    [S_GATE]  = {"gate_proj",   D, I},
    [S_UP]    = {"up_proj",     D, I},
    [S_DOWN]  = {"down_proj",   I, D},
    [S_OUT]   = {"o_proj",      D, D},
};

static int group_members(int attn, int g, int *slabs) {
    if (g == 0) {
        if (attn) { slabs[0] = S_QATTN; slabs[1] = S_KATTN; slabs[2] = S_VATTN; return 3; }
        slabs[0] = S_QLIN; slabs[1] = S_ZLIN; return 2;
    }
    if (g == 1) { slabs[0] = S_GATE; slabs[1] = S_UP; return 2; }
    if (g == 2) { slabs[0] = S_DOWN; return 1; }
    slabs[0] = S_OUT;
    return 1;
}

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* One kernel body, parameterised by N (rows verified), RP (weight rows per
 * simdgroup) and VEC (elements per load). The k-chunk of every activation row
 * is staged once per (threadgroup, tile) and reused by all NSG * RP rows.
 *
 * One detail worth stating because getting it wrong makes the kernel quietly
 * compute a quarter of the sum: a simdgroup is 32 lanes, so covering one TILE
 * of 512 elements takes lane*VEC stepping 32*VEC per iteration. Stepping by
 * NT*VEC instead leaves every lane owning a 1/4 slice of the tile. */
static NSString *source_for(int n, int rp, int vec) {
    NSMutableString *s = [NSMutableString string];
    [s appendString:
    @"#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "typedef struct { uint k; uint cols; uint rows_per_tg; uint n; } args_t;\n"
    "static inline float bf2f(ushort h) { return as_type<float>((uint)h << 16); }\n"];
    [s appendFormat:@"#define N %d\n", n];
    [s appendFormat:@"#define RP %d\n", rp];
    [s appendFormat:@"#define VEC %d\n", vec];
    [s appendFormat:@"#define NSG %d\n", NSG];
    [s appendFormat:@"#define NT %d\n", NT];
    [s appendFormat:@"#define TILE %d\n", TILE];

    if (vec == 8) {
        [s appendString:
        @"static inline void load_w(device const ushort *p, thread float *o) {\n"
        "    uint4 u = *((device const uint4 *)p);\n"
        "    o[0] = bf2f((ushort)(u.x & 0xffffu));\n"
        "    o[1] = bf2f((ushort)(u.x >> 16u));\n"
        "    o[2] = bf2f((ushort)(u.y & 0xffffu));\n"
        "    o[3] = bf2f((ushort)(u.y >> 16u));\n"
        "    o[4] = bf2f((ushort)(u.z & 0xffffu));\n"
        "    o[5] = bf2f((ushort)(u.z >> 16u));\n"
        "    o[6] = bf2f((ushort)(u.w & 0xffffu));\n"
        "    o[7] = bf2f((ushort)(u.w >> 16u));\n"
        "}\n"];
    } else {
        [s appendString:
        @"static inline void load_w(device const ushort *p, thread float *o) {\n"
        "    uint2 u = *((device const uint2 *)p);\n"
        "    o[0] = bf2f((ushort)(u.x & 0xffffu));\n"
        "    o[1] = bf2f((ushort)(u.x >> 16u));\n"
        "    o[2] = bf2f((ushort)(u.y & 0xffffu));\n"
        "    o[3] = bf2f((ushort)(u.y >> 16u));\n"
        "}\n"];
    }

    [s appendString:
    @"kernel void gemv_sw(constant args_t &a [[buffer(0)]],\n"
    "                    device const ushort *w [[buffer(1)]],\n"
    "                    device const float *x [[buffer(2)]],\n"
    "                    device float *out [[buffer(3)]],\n"
    "                    uint2 gid [[threadgroup_position_in_grid]],\n"
    "                    uint ti [[thread_index_in_threadgroup]],\n"
    "                    ushort lane [[thread_index_in_simdgroup]],\n"
    "                    ushort sg [[simdgroup_index_in_threadgroup]]) {\n"
    "    threadgroup float xt[N * TILE];\n"
    "    const uint cols = a.cols, k = a.k;\n"
    "    const uint c0 = gid.y * (NSG * RP) + (uint)sg * RP;\n"
    "    float s[RP][N];\n"
    "    for (uint q = 0; q < RP; q++)\n"
    "        for (uint r = 0; r < N; r++) s[q][r] = 0.0f;\n"
    "    for (uint base = 0; base < k; base += TILE) {\n"
    "        for (uint i = ti; i < N * TILE; i += NT) {\n"
    "            const uint r = i / TILE, t = i - r * TILE;\n"
    "            const uint kk = base + t;\n"
    "            xt[i] = (kk < k) ? x[(ulong)r * k + kk] : 0.0f;\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        for (uint j = (uint)lane * VEC; j < TILE; j += 32u * VEC) {\n"
    "            const uint kk = base + j;\n"
    "            if (kk >= k) break;\n"
    "            float xv[N][VEC];\n"];
    if (vec == 8) {
        [s appendString:
        @"            for (uint r = 0; r < N; r++) {\n"
        "                *((thread float4 *)xv[r]) = *((threadgroup const float4 *)\n"
        "                                       (xt + r * TILE + j));\n"
        "                *((thread float4 *)(xv[r] + 4)) = *((threadgroup const float4 *)\n"
        "                                       (xt + r * TILE + j + 4u));\n"
        "            }\n"];
    } else {
        [s appendString:
        @"            for (uint r = 0; r < N; r++)\n"
        "                *((thread float4 *)xv[r]) = *((threadgroup const float4 *)\n"
        "                                       (xt + r * TILE + j));\n"];
    }
    [s appendString:
    @"            for (uint q = 0; q < RP; q++) {\n"
    "                const uint c = c0 + q;\n"
    "                if (c >= cols) break;\n"
    "                float wv[VEC];\n"
    "                load_w(w + (ulong)c * k + kk, wv);\n"
    "                for (uint r = 0; r < N; r++) {\n"
    "                    for (uint v = 0; v < VEC; v++)\n"
    "                        s[q][r] = fma(wv[v], xv[r][v], s[q][r]);\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "    for (uint q = 0; q < RP; q++) {\n"
    "        const uint c = c0 + q;\n"
    "        if (c >= cols) break;\n"
    "        for (uint r = 0; r < N; r++) {\n"
    "            const float v = simd_sum(s[q][r]);\n"
    "            if (lane == 0) out[(ulong)r * cols + c] = v;\n"
    "        }\n"
    "    }\n"
    "}\n"];
    return s;
}

int main(int argc, char **argv) {
    int layers = argc > 1 ? atoi(argv[1]) : 32;
    int reps = argc > 2 ? atoi(argv[2]) : 3;
    if (layers < 1) layers = 32;

    static const int RPS[] = {1, 2, 4};
    static const int VECS[] = {4, 8};

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];

        id<MTLBuffer> W[S_N];
        for (int i = 0; i < S_N; i++) {
            size_t n = (size_t)SLAB[i].k * SLAB[i].cols;
            W[i] = [dev newBufferWithLength:n * 2 options:MTLResourceStorageModePrivate];
            if (!W[i]) { fprintf(stderr, "alloc %s failed\n", SLAB[i].name); return 1; }
        }
        id<MTLBuffer> X = [dev newBufferWithLength:(size_t)I * sizeof(float) * MAXK
                                           options:MTLResourceStorageModePrivate];
        id<MTLBuffer> O = [dev newBufferWithLength:(size_t)I * sizeof(float) * MAXK
                                           options:MTLResourceStorageModePrivate];

        double moved = 0.0;
        for (int l = 0; l < layers; l++) {
            int attn = (l % 4) == 3;
            for (int g = 0; g < 4; g++) {
                int m[3]; int nr = group_members(attn, g, m);
                for (int i = 0; i < nr; i++)
                    moved += (double)SLAB[m[i]].k * (double)SLAB[m[i]].cols * 2.0;
            }
        }
        printf("%s: %d layers, %.2f GB per weight pass\n", dev.name.UTF8String,
               layers, moved / 1e9);

        for (size_t ci = 0; ci < sizeof(RPS) / sizeof(RPS[0]); ci++) {
            for (size_t vi = 0; vi < sizeof(VECS) / sizeof(VECS[0]); vi++) {
                int rp = RPS[ci], vec = VECS[vi];
                printf("\n== RP %d (rows/simdgroup), VEC %d (%d rows/threadgroup) ==\n",
                       rp, vec, NSG * rp);
                printf("  k | ms/pass | GB/s | marginal ms | vs RP1k1 | GFLOP/s\n");
                printf("  --+---------+------+-------------+----------+--------\n");
                double base_ms = 0.0, prev = 0.0;
                for (int k = 1; k <= MAXK; k++) {
                    NSError *err = nil;
                    id<MTLLibrary> lib =
                        [dev newLibraryWithSource:source_for(k, rp, vec) options:nil error:&err];
                    if (!lib) { fprintf(stderr, "  k=%d compile: %s\n", k,
                                        err.localizedDescription.UTF8String); break; }
                    id<MTLFunction> fn = [lib newFunctionWithName:@"gemv_sw"];
                    id<MTLComputePipelineState> pso =
                        [dev newComputePipelineStateWithFunction:fn error:&err];
                    if (!pso) { fprintf(stderr, "  k=%d pso: %s\n", k,
                                        err.localizedDescription.UTF8String); break; }

                    double best = 1e9;
                    for (int r = 0; r < reps; r++) {
                        double t0 = now_s();
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc setComputePipelineState:pso];
                        for (int l = 0; l < layers; l++) {
                            int attn = (l % 4) == 3;
                            for (int g = 0; g < 4; g++) {
                                int m[3]; int nr = group_members(attn, g, m);
                                for (int i = 0; i < nr; i++) {
                                    struct { uint32_t k, cols, rows_per_tg, n; } a;
                                    a.k = SLAB[m[i]].k; a.cols = SLAB[m[i]].cols;
                                    a.rows_per_tg = (uint32_t)(NSG * rp); a.n = (uint32_t)k;
                                    [enc setBytes:&a length:sizeof(a) atIndex:0];
                                    [enc setBuffer:W[m[i]] offset:0 atIndex:1];
                                    [enc setBuffer:X offset:0 atIndex:2];
                                    [enc setBuffer:O offset:0 atIndex:3];
                                    uint32_t rt = (uint32_t)(NSG * rp);
                                    [enc dispatchThreadgroups:
                                             MTLSizeMake(1, (SLAB[m[i]].cols + rt - 1) / rt, 1)
                                                 threadsPerThreadgroup:MTLSizeMake(NT, 1, 1)];
                                }
                            }
                        }
                        [enc endEncoding];
                        [cb commit];
                        [cb waitUntilCompleted];
                        double dt = now_s() - t0;
                        if (dt < best) best = dt;
                    }
                    if (k == 1 && rp == RPS[0] && vec == VECS[0]) base_ms = best * 1e3;
                    double marg = k == 1 ? 0.0 : best * 1e3 - prev;
                    prev = best * 1e3;
                    double gflop = moved / 1e9 / 2.0 * (double)k * 2.0;
                    printf("  %d | %7.1f | %4.0f | %11.1f | %8.2fx | %7.0f\n",
                           k, best * 1e3, moved / best / 1e9, marg,
                           base_ms / (best * 1e3), gflop / best / 1e9);
                }
            }
        }
    }
    return 0;
}
