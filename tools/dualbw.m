/* dualbw — do the CPU pool and the GPU add up, or share one pipe?
 *
 * Measured separately on this host: the CPU pool reads at ~122 GB/s and the
 * GPU at ~284 GB/s. A dense decode step is nothing but weight bytes, so if
 * those two paths add, splitting every projection's output rows between the
 * CPU pool and the GPU would cut the step to ~16.5 GB / (sum) -- which is the
 * difference between 7 tok/s and 20.
 *
 * If they do not add, the aggregate lands at the larger of the two and the
 * split would only move bytes from a fast path to a slow one.
 *
 *   make dualbw && ./tools/dualbw [gib] [cpu_threads] [reps]
 *
 * Reports the CPU-only rate, the GPU-only rate, and the rate of both running
 * at once over the same window (bytes are counted from both sides).
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ---- CPU side: same inner loop as tools/membwbench's 16-byte kernel ---- */
typedef struct { const uint8_t *base; size_t bytes; double gbs; } cpu_job;

static void *cpu_worker(void *arg) {
    cpu_job *j = (cpu_job *)arg;
    const uint8_t *p = j->base;
    size_t n = j->bytes, i = 0;
    uint32x4_t a0 = vdupq_n_u32(0), a1 = vdupq_n_u32(0);
    for (; i + 64 <= n; i += 64) {
        a0 = vaddq_u32(a0, vld1q_u32((const uint32_t *)(p + i)));
        a1 = vaddq_u32(a1, vld1q_u32((const uint32_t *)(p + i + 32)));
    }
    uint32x4_t s = vaddq_u32(a0, a1);
    volatile uint32_t sink = vaddvq_u32(s);
    (void)sink;
    return NULL;                        /* timing is done by the caller */
}

static const char *SRC =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void sumread(device const uint4 *src [[buffer(0)]],\n"
    "                    device float *out [[buffer(1)]],\n"
    "                    constant ulong &n [[buffer(2)]],\n"
    "                    uint gid [[thread_position_in_grid]],\n"
    "                    uint gsz [[threads_per_grid]]) {\n"
    "    uint4 acc = uint4(0u);\n"
    "    for (ulong i = gid; i < n; i += gsz) acc += src[i];\n"
    "    out[gid & 1023u] = (float)((acc.x + acc.y + acc.z + acc.w) & 0xffffffu);\n"
    "}\n";

int main(int argc, char **argv) {
    size_t gib = argc > 1 ? (size_t)strtoul(argv[1], NULL, 10) : 4;
    int nthreads = argc > 2 ? atoi(argv[2]) : 12;
    int reps = argc > 3 ? atoi(argv[3]) : 2;
    size_t bytes = gib << 30;

    uint8_t *host = malloc(bytes);
    if (!host) { fprintf(stderr, "host alloc failed\n"); return 1; }
    memset(host, 0x3f, bytes);

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:
                                 [NSString stringWithUTF8String:SRC] options:nil error:&err];
        id<MTLFunction> fn = [lib newFunctionWithName:@"sumread"];
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        id<MTLBuffer> gbuf = [dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        id<MTLBuffer> out = [dev newBufferWithLength:4096 * sizeof(float)
                                             options:MTLResourceStorageModePrivate];
        if (!pso || !gbuf || !out) { fprintf(stderr, "metal setup failed\n"); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];

        printf("%s, %zu GiB per side, %d CPU threads, %d reps\n",
               dev.name.UTF8String, gib, nthreads, reps);

        double best_cpu = 0, best_gpu = 0, best_both = 0, best_cpu_alone = 0;
        for (int r = 0; r < reps; r++) {
            /* --- CPU alone --- */
            pthread_t th[16];
            cpu_job job[16];
            size_t chunk = bytes / (size_t)nthreads;
            double t0 = now_s();
            for (int i = 0; i < nthreads; i++) {
                job[i].base = host + chunk * (size_t)i; job[i].bytes = chunk;
                pthread_create(&th[i], NULL, cpu_worker, &job[i]);
            }
            for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
            double cpu_only = ((double)bytes / 1e9) / (now_s() - t0);
            if (cpu_only > best_cpu_alone) best_cpu_alone = cpu_only;

            /* --- GPU alone --- */
            uint64_t n = bytes / 16;
            t0 = now_s();
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:gbuf offset:0 atIndex:0];
            [enc setBuffer:out offset:0 atIndex:1];
            [enc setBytes:&n length:sizeof(n) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(81920, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            double gpu_only = ((double)bytes / 1e9) / (now_s() - t0);
            if (gpu_only > best_gpu) best_gpu = gpu_only;

            /* --- both at once, same window --- */
            t0 = now_s();
            for (int i = 0; i < nthreads; i++) {
                job[i].base = host + chunk * (size_t)i; job[i].bytes = chunk;
                pthread_create(&th[i], NULL, cpu_worker, &job[i]);
            }
            cb = [q commandBuffer];
            enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:gbuf offset:0 atIndex:0];
            [enc setBuffer:out offset:0 atIndex:1];
            [enc setBytes:&n length:sizeof(n) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(81920, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            [cb commit];
            for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
            [cb waitUntilCompleted];
            double both = ((double)(bytes * 2) / 1e9) / (now_s() - t0);
            if (both > best_both) best_both = both;
        }

        printf("\n%-10s %10s\n", "window", "GB/s");
        printf("%-10s %10.1f\n", "cpu", best_cpu_alone);
        printf("%-10s %10.1f\n", "gpu", best_gpu);
        printf("%-10s %10.1f   (both sides' bytes over one window)\n", "together", best_both);
        printf("\nadditive? %.2fx of the faster engine, %.2fx of the sum of the two\n",
               best_both / (best_gpu > best_cpu_alone ? best_gpu : best_cpu_alone),
               best_both / (best_gpu + best_cpu_alone));
        (void)best_cpu;
    }
    return 0;
}
