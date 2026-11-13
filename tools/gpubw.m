/* gpubw — what does the GPU side of the unified memory bus deliver?
 *
 * The CPU pool tops out at ~122 GB/s on this host (tools/membwbench, three
 * kernels, 1..16 threads, flat from 4 threads on). A decode step for a dense
 * BF16 checkpoint is a pure stream of weight bytes, so if the GPU can read the
 * same bytes faster, the GEMVs belong on the GPU and the per-token step time
 * falls by exactly that ratio. If it cannot, the CPU number is the machine and
 * there is nothing to move.
 *
 * Usage: gpubw [gib] [reps]
 *
 * The buffer is device-private and never written from the host: content is
 * irrelevant, only the byte count per unit time is being measured.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static const char *SRC =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    /* Grid-stride over 16-byte words: lane i touches word i, i+gsz, ... so a
     * warp's 32 lanes cover 512 contiguous bytes per iteration. */
    "kernel void sumread(device const uint4 *src [[buffer(0)]],\n"
    "                    device float *out [[buffer(1)]],\n"
    "                    constant ulong &n [[buffer(2)]],\n"
    "                    uint gid [[thread_position_in_grid]],\n"
    "                    uint gsz [[threads_per_grid]]) {\n"
    "    uint4 acc = uint4(0u);\n"
    "    for (ulong i = gid; i < n; i += gsz) acc += src[i];\n"
    "    ulong s = (ulong)acc.x + (ulong)acc.y + (ulong)acc.z + (ulong)acc.w;\n"
    "    out[gid & 1023u] = (float)(s & 0xffffffu);\n"
    "}\n"
    /* Same bytes, but through the BF16 widening a real dense GEMV performs,
     * so the number includes the conversion cost rather than only transport. */
    "kernel void sumread_bf16(device const ushort *src [[buffer(0)]],\n"
    "                         device float *out [[buffer(1)]],\n"
    "                         constant ulong &n [[buffer(2)]],\n"
    "                         uint gid [[thread_position_in_grid]],\n"
    "                         uint gsz [[threads_per_grid]]) {\n"
    "    float acc = 0.0f;\n"
    "    ulong i = (ulong)gid * 8ul;\n"
    "    for (; i + 8ul <= n; i += gsz * 8ul) {\n"
    "        ushort4 a = *((device const ushort4 *)(src + i));\n"
    "        ushort4 b = *((device const ushort4 *)(src + i + 4));\n"
    "        acc += as_type<float>((uint)a.x << 16) + as_type<float>((uint)a.y << 16)\n"
    "             + as_type<float>((uint)a.z << 16) + as_type<float>((uint)a.w << 16)\n"
    "             + as_type<float>((uint)b.x << 16) + as_type<float>((uint)b.y << 16)\n"
    "             + as_type<float>((uint)b.z << 16) + as_type<float>((uint)b.w << 16);\n"
    "    }\n"
    "    out[gid & 1023u] = acc;\n"
    "}\n";

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    size_t gib = argc > 1 ? (size_t)strtoul(argv[1], NULL, 10) : 4;
    int reps = argc > 2 ? atoi(argv[2]) : 3;
    if (gib < 1) gib = 1;
    size_t bytes = (gib << 30);

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no Metal device\n"); return 1; }
        printf("%s, %zu GiB buffer, %d reps\n", dev.name.UTF8String, gib, reps);

        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:
                                 [NSString stringWithUTF8String:SRC]
                                              options:nil error:&err];
        if (!lib) { fprintf(stderr, "compile: %s\n", err.localizedDescription.UTF8String); return 1; }

        id<MTLBuffer> buf = [dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        id<MTLBuffer> out = [dev newBufferWithLength:4096 * sizeof(float)
                                             options:MTLResourceStorageModePrivate];
        if (!buf || !out) { fprintf(stderr, "alloc failed\n"); return 1; }

        struct { const char *name; const char *fn; } kinds[] = {
            {"uint4", "sumread"},
            {"bf16", "sumread_bf16"},
        };
        const size_t grid_counts[] = {10240, 40960, 81920, 163840, 327680};

        for (size_t k = 0; k < sizeof(kinds) / sizeof(kinds[0]); k++) {
            id<MTLFunction> fn = [lib newFunctionWithName:
                                     [NSString stringWithUTF8String:kinds[k].fn]];
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
            if (!pso) { fprintf(stderr, "pso: %s\n", err.localizedDescription.UTF8String); return 1; }
            printf("\n%-6s %12s %10s %10s\n", "kernel", "threads", "GB/s", "ms");
            double best = 0.0;
            for (size_t g = 0; g < sizeof(grid_counts) / sizeof(grid_counts[0]); g++) {
                NSUInteger threads = grid_counts[g];
                double rate = 0.0, ms = 0.0;
                for (int r = 0; r < reps; r++) {
                    id<MTLCommandQueue> q = [dev newCommandQueue];
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pso];
                    [enc setBuffer:buf offset:0 atIndex:0];
                    [enc setBuffer:out offset:0 atIndex:1];
                    uint64_t n = (kinds[k].name[0] == 'b') ? bytes / 2 : bytes / 16;
                    [enc setBytes:&n length:sizeof(n) atIndex:2];
                    [enc dispatchThreads:MTLSizeMake(threads, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                    double t0 = now_s();
                    [cb commit];
                    [cb waitUntilCompleted];
                    double dt = now_s() - t0;
                    double gbs = ((double)bytes / 1e9) / dt;
                    if (gbs > rate) { rate = gbs; ms = dt * 1e3; }
                }
                if (rate > best) best = rate;
                printf("%-6s %12lu %10.1f %10.1f\n", kinds[k].name,
                       (unsigned long)threads, rate, ms);
            }
            printf("%-6s peak %10.1f GB/s\n", kinds[k].name, best);
        }
    }
    return 0;
}
