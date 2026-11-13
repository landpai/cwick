/* q35gemn — what does a k-row verify cost on the weight side?
 *
 * tools/q35one answered "one commit for the whole step" (37.6 ms for the
 * checkpoint's shape mix, 368 GB/s). Speculative decoding asks the next
 * question: verifying k drafted tokens reads the *same* weights once, so the
 * weight side of a k-row step should be ~flat in k and only the arithmetic
 * grows. If that is true, the marginal cost of a verified row is small and
 * 40 tok/s is reachable with the MTP head; if the k-row kernel degrades (more
 * registers, more L1 traffic on x, worse occupancy), it is not.
 *
 * So: same shapes, same single command buffer, one accumulator per row, and
 * the weight row is read exactly once per output row regardless of k.
 *
 *   make q35gemn && ./tools/q35gemn [layers] [reps]
 *
 * Prints ms and GB/s for k = 1..8 and the implied decode rate for each k,
 * where the step is scored as "k tokens came out of this one weight pass".
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
#define MAXR 8

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
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* The production kernel with MAXR accumulators instead of one. The weight row
 * is loaded once per j step and reused for every row, which is the whole
 * claim being tested. MAXR loops are guarded by a uniform bound so the
 * compiler keeps the accumulators in registers and the guard predication
 * costs nothing per lane. */
static const char *SRC =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "typedef struct { uint k; uint cols; uint rows_per_tg; uint n; } args_t;\n"
    "static inline float bf2f(ushort h) { return as_type<float>((uint)h << 16); }\n"
    "kernel void gemv_n(constant args_t &a [[buffer(0)]],\n"
    "                   device const ushort *w [[buffer(1)]],\n"
    "                   device const float *x [[buffer(2)]],\n"
    "                   device float *out [[buffer(3)]],\n"
    "                   uint2 gid [[threadgroup_position_in_grid]],\n"
    "                   ushort lane [[thread_index_in_simdgroup]],\n"
    "                   ushort sg [[simdgroup_index_in_threadgroup]]) {\n"
    "    const uint c = gid.y * a.rows_per_tg + (uint)sg;\n"
    "    if (c >= a.cols) return;\n"
    "    const uint k = a.k;\n"
    "    const uint n = a.n;\n"
    "    device const ushort *row = w + (ulong)c * k;\n"
    "    const uint j0 = (uint)lane * 4u;\n"
    "    float s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;\n"
    "    for (uint j = j0; j < k; j += 128u) {\n"
    "        uint2 p = *((device const uint2 *)(row + j));\n"
    "        float w0 = bf2f((ushort)(p.x & 0xffffu));\n"
    "        float w1 = bf2f((ushort)(p.x >> 16u));\n"
    "        float w2 = bf2f((ushort)(p.y & 0xffffu));\n"
    "        float w3 = bf2f((ushort)(p.y >> 16u));\n"
    "        for (uint r = 0; r < MAXR; r++) {\n"
    "            if (r >= n) break;\n"
    "            device const float *xr = x + (ulong)r * k + j;\n"
    "            float2 xa = *((device const float2 *)(xr));\n"
    "            float2 xb = *((device const float2 *)(xr + 2u));\n"
    "            float t = w0 * xa.x; t = fma(w1, xa.y, t);\n"
    "            t = fma(w2, xb.x, t); t = fma(w3, xb.y, t);\n"
    "            switch (r) {\n"
    "              case 0: s0 += t; break;\n"
    "              case 1: s1 += t; break;\n"
    "              case 2: s2 += t; break;\n"
    "              case 3: s3 += t; break;\n"
    "              case 4: s4 += t; break;\n"
    "              case 5: s5 += t; break;\n"
    "              case 6: s6 += t; break;\n"
    "              default: s7 += t; break;\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    float v[MAXR] = {s0,s1,s2,s3,s4,s5,s6,s7};\n"
    "    for (uint r = 0; r < MAXR; r++) {\n"
    "        if (r >= n) break;\n"
    "        out[(ulong)r * a.cols + c] = simd_sum(v[r]);\n"
    "    }\n"
    "}\n";

int main(int argc, char **argv) {
    int layers = argc > 1 ? atoi(argv[1]) : 32;
    int reps = argc > 2 ? atoi(argv[2]) : 3;
    if (layers < 1) layers = 32;

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        NSString *src = [[NSString stringWithUTF8String:SRC]
            stringByReplacingOccurrencesOfString:@"MAXR"
                                      withString:[NSString stringWithFormat:@"%d", MAXR]];
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { fprintf(stderr, "compile: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"gemv_n"];
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { fprintf(stderr, "pso: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];

        id<MTLBuffer> W[S_N];
        size_t bytes = 0;
        for (int i = 0; i < S_N; i++) {
            size_t n = (size_t)SLAB[i].k * SLAB[i].cols;
            W[i] = [dev newBufferWithLength:n * 2 options:MTLResourceStorageModePrivate];
            if (!W[i]) { fprintf(stderr, "alloc %s failed\n", SLAB[i].name); return 1; }
            bytes += n * 2;
        }
        /* n copies of every activation vector, so the k-row kernel reads k
         * distinct x's: same footprint the real forward needs. */
        id<MTLBuffer> X = [dev newBufferWithLength:(size_t)I * sizeof(float) * MAXR
                                           options:MTLResourceStorageModePrivate];
        id<MTLBuffer> O = [dev newBufferWithLength:(size_t)I * sizeof(float) * MAXR
                                           options:MTLResourceStorageModePrivate];

        double moved = 0.0;
        for (int l = 0; l < layers; l++) {
            int attn = (l % 4) == 3;
            for (int g = 0; g < 4; g++) {
                int m[3]; int n = group_members(attn, g, m);
                for (int i = 0; i < n; i++)
                    moved += (double)SLAB[m[i]].k * (double)SLAB[m[i]].cols * 2.0;
            }
        }
        printf("%s: %zu MB of weights, %d layers, %.2f GB per weight pass\n",
               dev.name.UTF8String, bytes >> 20, layers, moved / 1e9);

        printf("\n  k | ms/step | GB/s | tokens/step | implied tok/s\n");
        printf("  --+---------+------+-------------+--------------\n");
        for (int k = 1; k <= MAXR; k++) {
            double best = 1e9, best_gpu = 0;
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
                            NSUInteger nsg = 8;
                            a.rows_per_tg = (uint32_t)nsg;
                            a.n = (uint32_t)k;
                            [enc setBytes:&a length:sizeof(a) atIndex:0];
                            [enc setBuffer:W[m[i]] offset:0 atIndex:1];
                            [enc setBuffer:X offset:0 atIndex:2];
                            [enc setBuffer:O offset:0 atIndex:3];
                            [enc dispatchThreadgroups:
                                     MTLSizeMake(1, (SLAB[m[i]].cols + nsg - 1) / nsg, 1)
                                         threadsPerThreadgroup:MTLSizeMake(nsg * 32, 1, 1)];
                        }
                    }
                }
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                double dt = now_s() - t0;
                if (dt < best) best = dt;
                double gpu = (cb.GPUEndTime - cb.GPUStartTime);
                if (gpu > best_gpu) best_gpu = gpu;
            }
            printf("  %d | %7.1f | %4.0f | %11d | %8.2f\n", k, best * 1e3,
                   moved / best / 1e9, k, (double)k / best);
        }

        /* The marginal row is the number the decision turns on: at k rows a
         * verified token costs 1/k of a weight pass plus that row's share of
         * the non-GEMV work. */
        printf("\n40 tok/s at a %.1f ms weight pass needs k * alpha >="
               " 40 * 0.0465 = 1.86 accepted rows\n", 37.6);
    }
    return 0;
}
