/* q35gemt — the k-row weight pass with x in threadgroup memory.
 *
 * tools/q35gemn measured the obvious kernel (one simdgroup per weight row,
 * one accumulator per verified row) and it lost: 40.8 ms at k=1 but +9-10 ms
 * for every extra row, so k=4 already cost 64.5 ms. The diagnosis is not ALU
 * for the FMAs: the kernel reads the *whole* activation row for every weight
 * row, so x costs 4 bytes per weight element -- twice the weight traffic --
 * and that read amplifies with k while the weights do not.
 *
 * So this is the fix: a real tile. One threadgroup owns RW weight rows, each
 * simdgroup takes one of them, and the k-chunk of all N activation rows is
 * staged in threadgroup memory once and shared by every simdgroup in the
 * group. x traffic drops by RW, which is the whole difference between "k rows
 * cost k weight passes" and "k rows cost one weight pass plus arithmetic".
 *
 * N is a compile-time constant (the source is built once per k), so the
 * accumulator loop is straight-line and there is no guard in the hot loop.
 *
 *   make q35gemt && ./tools/q35gemt [layers] [reps]
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
#define RW 8          /* weight rows per threadgroup (one per simdgroup) */
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

static NSString *source_for(int n) {
    NSMutableString *s = [NSMutableString string];
    [s appendString:
    @"#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "typedef struct { uint k; uint cols; uint rows_per_tg; uint n; } args_t;\n"
    "static inline float bf2f(ushort h) { return as_type<float>((uint)h << 16); }\n"];
    [s appendFormat:@"#define N %d\n", n];
    [s appendFormat:@"#define TILE %d\n", TILE];
    [s appendFormat:@"#define RW %d\n", RW];
    [s appendString:
    @"kernel void gemv_t(constant args_t &a [[buffer(0)]],\n"
    "                   device const ushort *w [[buffer(1)]],\n"
    "                   device const float *x [[buffer(2)]],\n"
    "                   device float *out [[buffer(3)]],\n"
    "                   uint2 gid [[threadgroup_position_in_grid]],\n"
    "                   uint ti [[thread_index_in_threadgroup]],\n"
    "                   ushort lane [[thread_index_in_simdgroup]],\n"
    "                   ushort sg [[simdgroup_index_in_threadgroup]]) {\n"
    "    threadgroup float xt[N * TILE];\n"
    "    const uint c = gid.y * RW + (uint)sg;\n"
    "    const uint cols = a.cols, k = a.k;\n"
    "    const bool live = (c < cols);\n"
    "    device const ushort *row = w + (ulong)(live ? c : 0u) * k;\n"
    "    float s[N];\n"
    "    for (uint r = 0; r < N; r++) s[r] = 0.0f;\n"
    "    for (uint base = 0; base < k; base += TILE) {\n"
    "        for (uint i = ti; i < N * TILE; i += 256u) {\n"
    "            const uint r = i / TILE, t = i % TILE;\n"
    "            const uint kk = base + t;\n"
    "            xt[i] = (kk < k) ? x[(ulong)r * k + kk] : 0.0f;\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        if (live) {\n"
    "            for (uint j = (uint)lane * 4u; j < TILE; j += 128u) {\n"
    "                const uint kk = base + j;\n"
    "                if (kk < k) {\n"
    "                    uint2 p = *((device const uint2 *)(row + kk));\n"
    "                    const float w0 = bf2f((ushort)(p.x & 0xffffu));\n"
    "                    const float w1 = bf2f((ushort)(p.x >> 16u));\n"
    "                    const float w2 = bf2f((ushort)(p.y & 0xffffu));\n"
    "                    const float w3 = bf2f((ushort)(p.y >> 16u));\n"
    "                    for (uint r = 0; r < N; r++) {\n"
    "                        device const float *unused = 0; (void)unused;\n"
    "                        float2 xa = *((threadgroup const float2 *)(xt + r * TILE + j));\n"
    "                        float2 xb = *((threadgroup const float2 *)(xt + r * TILE + j + 2u));\n"
    "                        float t = w0 * xa.x;\n"
    "                        t = fma(w1, xa.y, t);\n"
    "                        t = fma(w2, xb.x, t);\n"
    "                        t = fma(w3, xb.y, t);\n"
    "                        s[r] += t;\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "    for (uint r = 0; r < N; r++) {\n"
    "        const float v = simd_sum(s[r]);\n"
    "        if (live && lane == 0) out[(ulong)r * cols + c] = v;\n"
    "    }\n"
    "}\n"];
    return s;
}

int main(int argc, char **argv) {
    int layers = argc > 1 ? atoi(argv[1]) : 32;
    int reps = argc > 2 ? atoi(argv[2]) : 3;
    if (layers < 1) layers = 32;

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];

        id<MTLBuffer> W[S_N];
        size_t bytes = 0;
        for (int i = 0; i < S_N; i++) {
            size_t n = (size_t)SLAB[i].k * SLAB[i].cols;
            W[i] = [dev newBufferWithLength:n * 2 options:MTLResourceStorageModePrivate];
            if (!W[i]) { fprintf(stderr, "alloc %s failed\n", SLAB[i].name); return 1; }
            bytes += n * 2;
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
        printf("%s: %zu MB of weights, %d layers, %.2f GB per weight pass\n",
               dev.name.UTF8String, bytes >> 20, layers, moved / 1e9);
        printf("tile %d k-elements, %d weight rows per threadgroup (%.0f KiB threadgroup memory at k=8)\n\n",
               TILE, RW, (double)(MAXK * TILE * 4) / 1024.0);

        printf("  k | ms/step | GB/s | marginal ms | tokens/step | tok/s at alpha=1 | alpha=0.6\n");
        printf("  --+---------+------+-------------+-------------+------------------+----------\n");
        double prev = 0.0;
        for (int k = 1; k <= MAXK; k++) {
            NSError *err = nil;
            id<MTLLibrary> lib = [dev newLibraryWithSource:source_for(k) options:nil error:&err];
            if (!lib) { fprintf(stderr, "compile k=%d: %s\n", k, err.localizedDescription.UTF8String); return 1; }
            id<MTLFunction> fn = [lib newFunctionWithName:@"gemv_t"];
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
            if (!pso) { fprintf(stderr, "pso k=%d: %s\n", k, err.localizedDescription.UTF8String); return 1; }

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
                            a.rows_per_tg = RW; a.n = (uint32_t)k;
                            [enc setBytes:&a length:sizeof(a) atIndex:0];
                            [enc setBuffer:W[m[i]] offset:0 atIndex:1];
                            [enc setBuffer:X offset:0 atIndex:2];
                            [enc setBuffer:O offset:0 atIndex:3];
                            [enc dispatchThreadgroups:
                                     MTLSizeMake(1, (SLAB[m[i]].cols + RW - 1) / RW, 1)
                                         threadsPerThreadgroup:MTLSizeMake(RW * 32, 1, 1)];
                        }
                    }
                }
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                double dt = now_s() - t0;
                if (dt < best) best = dt;
            }
            double marg = k == 1 ? 0.0 : best * 1e3 - prev;
            prev = best * 1e3;
            printf("  %d | %7.1f | %4.0f | %11.1f | %11d | %16.2f | %9.2f\n",
                   k, best * 1e3, moved / best / 1e9, marg, k,
                   (double)k / best, 0.6 * (double)k / best);
        }
        printf("\nA step also pays the non-GEMV work once per row (norms, the DeltaNet\n"
               "recurrence, attention over the KV cache): measured at ~5.7 ms/row on\n"
               "the 4K-context step, so add k * 5.7 ms before dividing.\n");
    }
    return 0;
}
