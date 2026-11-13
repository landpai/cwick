/* q35gemck — does the swept kernel compute the same thing as the shipped one?
 *
 * A 3.4x jump over a memory-bound baseline is not an optimization until the
 * arithmetic is proven identical, so this runs the shipped q35_gemv_n body
 * (copied verbatim from src/q35_shader.c) and the swept body on the same
 * bf16 weights and f32 activations, at small dimensions where a CPU reference
 * is also available, and prints the worst deviation of each.
 *
 *   make tools/q35gemck && ./tools/q35gemck
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define K_DIM 256
#define COLS  40
#define NROW  4

static NSString *ref_source(int n) {
    /* verbatim from src/q35_shader.c, RW = 8, TILE = 512 */
    return [NSString stringWithFormat:
    @"#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "typedef struct { uint k; uint cols; uint n; uint sx; uint so;\n"
    "               uint pad0; uint pad1; uint pad2; } q35_gemvn_args;\n"
    "static inline float q35_bf(ushort h) { return as_type<float>((uint)h << 16u); }\n"
    "#define N %d\n#define TILE 512\n#define RW 8\n"
    "kernel void q35_gemv_n(\n"
    "        constant q35_gemvn_args &a [[buffer(0)]],\n"
    "        device const ushort *w [[buffer(1)]],\n"
    "        device const float *x [[buffer(2)]],\n"
    "        device float *out [[buffer(3)]],\n"
    "        uint2 gid [[threadgroup_position_in_grid]],\n"
    "        uint ti [[thread_index_in_threadgroup]],\n"
    "        ushort lane [[thread_index_in_simdgroup]],\n"
    "        ushort sg [[simdgroup_index_in_threadgroup]]) {\n"
    "    threadgroup float xt[N * TILE];\n"
    "    const uint c = gid.y * RW + (uint)sg;\n"
    "    const uint cols = a.cols, k = a.k, so = a.so;\n"
    "    const bool live = (c < cols);\n"
    "    device const ushort *row = w + (ulong)(live ? c : 0u) * k;\n"
    "    float s[N];\n"
    "    for (uint r = 0; r < N; r++) s[r] = 0.0f;\n"
    "    for (uint base = 0; base < k; base += TILE) {\n"
    "        for (uint i = ti; i < N * TILE; i += 256u) {\n"
    "            const uint r = i / TILE, t = i - r * TILE;\n"
    "            const uint kk = base + t;\n"
    "            xt[i] = (kk < k) ? x[(ulong)r * a.sx + kk] : 0.0f;\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        if (live) {\n"
    "            for (uint j = (uint)lane * 4u; j < TILE; j += 128u) {\n"
    "                const uint kk = base + j;\n"
    "                if (kk < k) {\n"
    "                    uint2 p = *((device const uint2 *)(row + kk));\n"
    "                    const float w0 = q35_bf((ushort)(p.x & 0xffffu));\n"
    "                    const float w1 = q35_bf((ushort)(p.x >> 16u));\n"
    "                    const float w2 = q35_bf((ushort)(p.y & 0xffffu));\n"
    "                    const float w3 = q35_bf((ushort)(p.y >> 16u));\n"
    "                    for (uint r = 0; r < N; r++) {\n"
    "                        float2 xa = *((threadgroup const float2 *)(xt + r * TILE + j));\n"
    "                        float2 xb = *((threadgroup const float2 *)(xt + r * TILE + j + 2u));\n"
    "                        s[r] = fma(w0, xa.x, s[r]);\n"
    "                        s[r] = fma(w1, xa.y, s[r]);\n"
    "                        s[r] = fma(w2, xb.x, s[r]);\n"
    "                        s[r] = fma(w3, xb.y, s[r]);\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "    for (uint r = 0; r < N; r++) {\n"
    "        const float v = simd_sum(s[r]);\n"
    "        if (live && lane == 0) out[(ulong)r * so + c] = v;\n"
    "    }\n"
    "}\n", n];
}

static NSString *sw_source(int n, int rp, int vec) {
    return [NSString stringWithFormat:
    @"#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "typedef struct { uint k; uint cols; uint rows_per_tg; uint n; } args_t;\n"
    "static inline float bf2f(ushort h) { return as_type<float>((uint)h << 16); }\n"
    "#define N %d\n#define RP %d\n#define VEC %d\n#define NSG 8\n#define NT 256\n"
    "#define TILE 512\n"
    "static inline void load_w(device const ushort *p, thread float *o) {\n"
    "    uint2 u = *((device const uint2 *)p);\n"
    "    o[0] = bf2f((ushort)(u.x & 0xffffu));\n"
    "    o[1] = bf2f((ushort)(u.x >> 16u));\n"
    "    o[2] = bf2f((ushort)(u.y & 0xffffu));\n"
    "    o[3] = bf2f((ushort)(u.y >> 16u));\n"
    "}\n"
    "kernel void gemv_sw(constant args_t &a [[buffer(0)]],\n"
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
    "            float xv[N][VEC];\n"
    "            for (uint r = 0; r < N; r++)\n"
    "                *((thread float4 *)xv[r]) = *((threadgroup const float4 *)\n"
    "                                       (xt + r * TILE + j));\n"
    "            for (uint q = 0; q < RP; q++) {\n"
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
    "}\n", n, rp, vec];
}

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];

        const size_t nw = (size_t)COLS * K_DIM;
        ushort *hw = malloc(nw * 2);
        float *hx = malloc((size_t)K_DIM * NROW * 4);
        for (size_t i = 0; i < nw; i++) {
            /* bf16 is sign(1) exp(8) mantissa(7): randomising 0x7fff walks the
             * exponent into 0xFF, i.e. inf/NaN, which makes every comparison
             * below vacuously pass. Pin the exponent, randomise the mantissa. */
            hw[i] = (ushort)(0x3a00 | (rand() & 0x7f)); /* ~[0.0005, 0.001) */
        }
        for (int i = 0; i < K_DIM * NROW; i++)
            hx[i] = (float)((rand() % 2001) - 1000) / 1000.0f;

        float ref[NROW][COLS];
        for (int r = 0; r < NROW; r++)
            for (int c = 0; c < COLS; c++) {
                float acc = 0.0f;
                for (int kk = 0; kk < K_DIM; kk++) {
                    float w = 0.0f;
                    uint32_t bits = (uint32_t)hw[(size_t)c * K_DIM + kk] << 16;
                    memcpy(&w, &bits, 4);
                    acc = fmaf(w, hx[(size_t)r * K_DIM + kk], acc);
                }
                ref[r][c] = acc;
            }

        id<MTLBuffer> W = [dev newBufferWithBytes:hw length:nw * 2
                                          options:MTLResourceStorageModeManaged];
        id<MTLBuffer> X = [dev newBufferWithBytes:hx length:(size_t)K_DIM * NROW * 4
                                          options:MTLResourceStorageModeManaged];
        id<MTLBuffer> O = [dev newBufferWithLength:(size_t)COLS * NROW * 4
                                           options:MTLResourceStorageModeShared];

        struct { uint32_t k, cols, n, sx, so, p0, p1, p2; } ra;
        ra.k = K_DIM; ra.cols = COLS; ra.n = NROW; ra.sx = K_DIM; ra.so = COLS;
        ra.p0 = ra.p1 = ra.p2 = 0;

        printf("%s | k=%d cols=%d rows=%d\n\n", dev.name.UTF8String, K_DIM, COLS, NROW);
        printf("  variant        | max abs dev vs CPU | max dev vs shipped\n");
        printf("  ---------------+--------------------+---------------------\n");

        double shipped[NROW][COLS];
        for (int variant = 0; variant < 3; variant++) {
            int rp = variant == 0 ? 1 : (variant == 1 ? 2 : 4);
            int vec = 4;
            NSError *err = nil;
            NSString *src = variant == 0 ? ref_source(NROW) : sw_source(NROW, rp, vec);
            const char *fname = variant == 0 ? "q35_gemv_n" : "gemv_sw";
            id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
            if (!lib) { fprintf(stderr, "compile: %s\n", err.localizedDescription.UTF8String); return 1; }
            id<MTLComputePipelineState> pso =
                [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:
                    [NSString stringWithUTF8String:fname]] error:&err];
            if (!pso) { fprintf(stderr, "pso: %s\n", err.localizedDescription.UTF8String); return 1; }

            memset([O contents], 0, (size_t)COLS * NROW * 4);
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            if (variant == 0) [enc setBytes:&ra length:sizeof(ra) atIndex:0];
            else {
                struct { uint32_t k, cols, rows_per_tg, n; } sa;
                sa.k = K_DIM; sa.cols = COLS; sa.rows_per_tg = (uint32_t)(8 * rp); sa.n = NROW;
                [enc setBytes:&sa length:sizeof(sa) atIndex:0];
            }
            [enc setBuffer:W offset:0 atIndex:1];
            [enc setBuffer:X offset:0 atIndex:2];
            [enc setBuffer:O offset:0 atIndex:3];
            uint32_t rt = (uint32_t)(variant == 0 ? 8 : 8 * rp);
            [enc dispatchThreadgroups:MTLSizeMake(1, (COLS + rt - 1) / rt, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            const float *got = (const float *)[O contents];
            double dref = 0.0, dship = 0.0;
            for (int r = 0; r < NROW; r++)
                for (int c = 0; c < COLS; c++) {
                    double g = got[r * COLS + c];
                    double d = fabs(g - (double)ref[r][c]);
                    if (d > dref) dref = d;
                    if (variant == 0) shipped[r][c] = g;
                    else {
                        double d2 = fabs(g - shipped[r][c]);
                        if (d2 > dship) dship = d2;
                    }
                }
            if (getenv("DUMP")) {
                printf("   [dump] %-12s got:", variant == 0 ? "shipped" : (rp==2?"swept2":"swept4"));
                for (int c = 0; c < 4; c++) printf(" %.6e", got[c]);
                printf("  ref:");
                for (int c = 0; c < 4; c++) printf(" %.6e", ref[0][c]);
                printf("\n");
            }
            char label[32];
            snprintf(label, sizeof(label), variant == 0 ? "shipped" : "swept RP=%d", rp);
            if (variant == 0)
                printf("  %-14s | %18.3e | %19s\n", label, dref, "-");
            else
                printf("  %-14s | %18.3e | %19.3e\n", label, dref, dship);
        }

        /* how big is the sumsq, so a "deviation" can be read in context */
        double ss = 0.0;
        for (int r = 0; r < NROW; r++)
            for (int c = 0; c < COLS; c++) ss += (double)ref[r][c] * ref[r][c];
        printf("\n  |ref| (rms over %d values) = %.4e\n", NROW * COLS,
               sqrt(ss / (NROW * COLS)));
        free(hw); free(hx);
    }
    return 0;
}
