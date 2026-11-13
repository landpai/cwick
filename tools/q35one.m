/* q35one — one decode step in one command buffer: what does it cost?
 *
 * tools/q35step showed the per-group GPU path is 1.27x, not 2.4x, because a
 * layer needs four commit+wait boundaries and each costs ~0.9 ms. The fix is
 * structural: encode every dispatch of the whole step into a single command
 * buffer, so the step pays one round trip instead of 128. Metal executes
 * compute encoders in submission order, so a chain of dependent GEMVs is legal
 * inside one buffer -- the CPU only has to stay out of the loop.
 *
 * This tool measures that upper bound with the production kernel: the
 * checkpoint's own shape mix, 32 layers, one command buffer, one wait.
 *
 *   make q35one && ./tools/q35one [layers] [reps]
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

/* The production kernel, unchanged: one simdgroup per output row, 8-byte
 * weight loads (this checkpoint's tensor offsets are 8-mod-16, so uint4 would
 * be misaligned off the row base). */
static const char *SRC =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "typedef struct { uint k; uint cols; uint rows_per_tg; uint pad1; uint pad2; } args_t;\n"
    "static inline float bf2f(ushort h) { return as_type<float>((uint)h << 16); }\n"
    "kernel void gemv(constant args_t &a [[buffer(0)]],\n"
    "                 device const ushort *w [[buffer(1)]],\n"
    "                 device const float *x [[buffer(2)]],\n"
    "                 device float *out [[buffer(3)]],\n"
    "                 uint2 gid [[threadgroup_position_in_grid]],\n"
    "                 ushort lane [[thread_index_in_simdgroup]],\n"
    "                 ushort sg [[simdgroup_index_in_threadgroup]]) {\n"
    "    const uint c = gid.y * a.rows_per_tg + (uint)sg;\n"
    "    if (c >= a.cols) return;\n"
    "    const uint k = a.k;\n"
    "    device const ushort *row = w + (ulong)c * k;\n"
    "    const uint j0 = (uint)lane * 4u;\n"
    "    float sum = 0.0f;\n"
    "    for (uint j = j0; j < k; j += 128u) {\n"
    "        uint2 p = *((device const uint2 *)(row + j));\n"
    "        float2 xa = *((device const float2 *)(x + j));\n"
    "        float2 xb = *((device const float2 *)(x + j + 2u));\n"
    "        sum = fma(bf2f((ushort)(p.x & 0xffffu)), xa.x, sum);\n"
    "        sum = fma(bf2f((ushort)(p.x >> 16u)), xa.y, sum);\n"
    "        sum = fma(bf2f((ushort)(p.y & 0xffffu)), xb.x, sum);\n"
    "        sum = fma(bf2f((ushort)(p.y >> 16u)), xb.y, sum);\n"
    "    }\n"
    "    out[c] = simd_sum(sum);\n"
    "}\n";

int main(int argc, char **argv) {
    int layers = argc > 1 ? atoi(argv[1]) : 32;
    int reps = argc > 2 ? atoi(argv[2]) : 3;
    int sets = argc > 3 ? atoi(argv[3]) : 1;
    if (layers < 1) layers = 32;
    if (sets < 1) sets = 1;

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:
                                 [NSString stringWithUTF8String:SRC] options:nil error:&err];
        if (!lib) { fprintf(stderr, "compile: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"gemv"];
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { fprintf(stderr, "pso: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];

        /* One buffer per distinct shape, `sets` copies so a multi-layer step
         * can rotate and keep the working set past any cache. 0.5 GB per set. */
        id<MTLBuffer> W[S_N * 4];
        id<MTLBuffer> X = [dev newBufferWithLength:(size_t)I * sizeof(float)
                                           options:MTLResourceStorageModePrivate];
        id<MTLBuffer> O = [dev newBufferWithLength:(size_t)I * sizeof(float) * 4
                                           options:MTLResourceStorageModePrivate];
        size_t bytes = 0;
        for (int s = 0; s < sets; s++)
            for (int i = 0; i < S_N; i++) {
                size_t n = (size_t)SLAB[i].k * SLAB[i].cols;
                W[(size_t)s * S_N + i] = [dev newBufferWithLength:n * 2
                                                         options:MTLResourceStorageModePrivate];
                if (!W[(size_t)s * S_N + i]) { fprintf(stderr, "alloc %s failed\n", SLAB[i].name); return 1; }
                bytes += n * 2;
            }
        printf("%s: %zu MB of weights in %d set(s), %d layers, %d reps\n",
               dev.name.UTF8String, bytes >> 20, sets, layers, reps);

        double moved = 0.0;
        for (int l = 0; l < layers; l++) {
            int attn = (l % 4) == 3;
            for (int g = 0; g < 4; g++) {
                int m[3]; int n = group_members(attn, g, m);
                for (int i = 0; i < n; i++)
                    moved += (double)SLAB[m[i]].k * (double)SLAB[m[i]].cols * 2.0;
            }
        }
        printf("%.2f GB per simulated step\n", moved / 1e9);

        double best = 1e9, best_gpu = 0;
        for (int r = 0; r < reps; r++) {
            double t0 = now_s();
            id<MTLCommandBuffer> cb = [q commandBuffer];
            /* ONE encoder for the whole step: compute encoders inside a
             * command buffer run in submission order, so each GEMV sees the
             * previous one's output without a round trip. */
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            for (int l = 0; l < layers; l++) {
                int attn = (l % 4) == 3;
                for (int g = 0; g < 4; g++) {
                    int m[3]; int n = group_members(attn, g, m);
                    for (int i = 0; i < n; i++) {
                        struct { uint32_t k, cols, rows_per_tg, p1, p2; } a;
                        a.k = SLAB[m[i]].k; a.cols = SLAB[m[i]].cols;
                        NSUInteger nsg = 8;               /* 256-thread groups */
                        a.rows_per_tg = (uint32_t)nsg;
                        a.p1 = a.p2 = 0;
                        [enc setBytes:&a length:sizeof(a) atIndex:0];
                        [enc setBuffer:W[(size_t)((l / 4) % sets) * S_N + m[i]]
                                offset:0 atIndex:1];
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

        printf("\none commit: %.1f ms  = %.1f GB/s  = %.2f tok/s at 16.5 GB/step\n",
               best * 1e3, moved / best / 1e9, 16500.0 / (best * 1e3));
        printf("gpu-reported kernel time (max of reps): %.1f ms\n", best_gpu * 1e3);
    }
    return 0;
}
