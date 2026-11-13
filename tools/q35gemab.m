/* q35gemab — what actually costs a marginal row in a k-row weight pass.
 *
 * The measurement that started this: one weight pass over this checkpoint's
 * shapes (13.82 GB) takes 38.5 ms at k=1 and 70 ms at k=8, so the marginal row
 * is 4.5 ms. Arithmetic cannot explain it -- 8 rows over 13.82 GB is 110 GFLOP,
 * and 110 GFLOP in 70 ms is 1.6 TFLOP/s, about a ninth of what this GPU issues
 * in FP32. Ablation (ABL) says what does, at RP=2:
 *
 *   ABL=0  everything                                   k=8 -> 65.8 ms
 *   ABL=1  no weight read                               k=8 -> 36.4 ms
 *   ABL=2  no activation read                           k=8 -> 45.2 ms
 *   ABL=3  one add per weight element instead of N FMAs  k=8 -> 64.0 ms
 *
 * so the weight read is ~29 ms, the activation read ~21 ms, and the multiply is
 * ~2 ms. The multiply being nearly free is the important number: the pass is
 * not compute bound at any k, and a marginal row costs what it costs because it
 * adds memory work, not arithmetic.
 *
 * That leaves two explanations for 29 + 21 adding up to 66 instead of
 * max(29, 21): the activation read is limited on its own, or the two streams
 * never overlap because a warp only ever has one weight load in flight.
 * Raising RP tests the first -- it cuts activation traffic by RP -- and it
 * makes things worse past 2, so the first is out. PF tests the second: issue
 * PF * RP weight loads back to back before consuming any of them.
 *
 * MODE=1 is the other way at the activation read: skip threadgroup memory and
 * read activations straight from device. x is N * k * 4 bytes per matrix --
 * 128 KB at k = 8 -- so it stays resident in L2 across the whole pass, and this
 * also removes the staging loop and both barriers.
 *
 *   clang -O2 -fobjc-arc -framework Foundation -framework Metal \
 *         -o tools/q35gemab tools/q35gemab.m
 *   MODE=0 PF=2 ./tools/q35gemab [layers] [reps]
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

static void append_common_head(NSMutableString *s, int n, int rp, int pf, int tile) {
    [s appendString:
    @"#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "typedef struct { uint k; uint cols; uint sx; uint so; } args_t;\n"
    "static inline float bf2f(ushort h) { return as_type<float>((uint)h << 16); }\n"
    "static inline void load_w(device const ushort *p, thread float *o) {\n"
    "    uint2 u = *((device const uint2 *)p);\n"
    "    o[0] = bf2f((ushort)(u.x & 0xffffu));\n"
    "    o[1] = bf2f((ushort)(u.x >> 16u));\n"
    "    o[2] = bf2f((ushort)(u.y & 0xffffu));\n"
    "    o[3] = bf2f((ushort)(u.y >> 16u));\n"
    "}\n"];
    [s appendFormat:@"#define N %d\n", n];
    [s appendFormat:@"#define RP %d\n", rp];
    [s appendFormat:@"#define PF %d\n", pf];
    [s appendFormat:@"#define NSG %d\n", NSG];
    [s appendFormat:@"#define NT %d\n", NT];
    [s appendFormat:@"#define TILE %d\n", tile];
}

/* Four activation floats for one (row, group), from threadgroup staging or
 * straight out of device memory. */
static void append_xv(NSMutableString *s, int mode, int xs) {
    if (mode == 1 && xs == 1) {
        /* Scalar form: the production scratch buffers do not promise 16-byte
         * alignment of every row start, and a misaligned float4 load in Metal
         * is not a slow path -- it is the wrong bytes. Four consecutive scalar
         * reads are coalesced across the warp exactly like the vector one. */
        [s appendString:
        @"                    float xv[N][4];\n"
        "                    for (uint r = 0; r < N; r++) {\n"
        "                        const device float *xp = x + (ulong)r * sx + jj;\n"
        "                        xv[r][0] = xp[0]; xv[r][1] = xp[1];\n"
        "                        xv[r][2] = xp[2]; xv[r][3] = xp[3];\n"
        "                    }\n"];
    } else if (mode == 1) {
        [s appendString:
        @"                    float xv[N][4];\n"
        "                    for (uint r = 0; r < N; r++)\n"
        "                        *((thread float4 *)xv[r]) =\n"
        "                            *((device const float4 *)(x + (ulong)r * sx + jj));\n"];
    } else {
        [s appendString:
        @"                    float xv[N][4];\n"
        "                    for (uint r = 0; r < N; r++) {\n"
        "                        float2 xa = *((threadgroup const float2 *)\n"
        "                                      (xt + r * TILE + jj2));\n"
        "                        float2 xb = *((threadgroup const float2 *)\n"
        "                                      (xt + r * TILE + jj2 + 2u));\n"
        "                        xv[r][0] = xa.x; xv[r][1] = xa.y;\n"
        "                        xv[r][2] = xb.x; xv[r][3] = xb.y;\n"
        "                    }\n"];
    }
}

static NSString *source_for(int n, int rp, int abl, int pf, int mode, int tile, int nsg, int xs) {
    NSMutableString *s = [NSMutableString string];
    append_common_head(s, n, rp, pf, tile);
    [s appendFormat:@"#undef NSG\n#define NSG %d\n", nsg];

    [s appendString:
    @"kernel void gemv_ab(constant args_t &a [[buffer(0)]],\n"
    "                   device const ushort *w [[buffer(1)]],\n"
    "                   device const float *x [[buffer(2)]],\n"
    "                   device float *out [[buffer(3)]],\n"
    "                   uint2 gid [[threadgroup_position_in_grid]],\n"
    "                   uint ti [[thread_index_in_threadgroup]],\n"
    "                   ushort lane [[thread_index_in_simdgroup]],\n"
    "                   ushort sg [[simdgroup_index_in_threadgroup]]) {\n"];
    if (mode == 0) {
        [s appendString:@"    threadgroup float xt[N * TILE];\n"];
    }
    [s appendString:
    @"    const uint cols = a.cols, k = a.k, sx = a.sx, so = a.so;\n"
    "    const uint c0 = gid.y * (NSG * RP) + (uint)sg * RP;\n"
    "    const uint qmax = (cols > c0) ? min((uint)RP, cols - c0) : 0u;\n"
    "    float s[RP][N];\n"
    "    for (uint q = 0; q < RP; q++)\n"
    "        for (uint r = 0; r < N; r++) s[q][r] = 0.0f;\n"];

    if (mode == 0) {
        [s appendString:
        @"    for (uint base = 0; base < k; base += TILE) {\n"
        "        for (uint i = ti; i < N * TILE; i += NT) {\n"
        "            const uint r = i / TILE, t = i - r * TILE;\n"
        "            const uint kk = base + t;\n"
        "            xt[i] = (kk < k) ? x[(ulong)r * sx + kk] : 0.0f;\n"
        "        }\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"];
    } else {
        [s appendString:@"    for (uint base = 0; base < k; base += TILE) {\n"];
    }

    [s appendString:
    @"        for (uint jg = (uint)lane * 4u; jg < TILE; jg += 32u * 4u * PF) {\n"
    "            /* Stage 1 issues every load of the group before stage 2 touches\n"
    "             * any of them; that is the whole point of PF. */\n"
    "            float wv[PF][RP][4];\n"
    "            for (uint t = 0; t < PF; t++) {\n"
    "                const uint jj = base + jg + t * 128u;\n"
    "                const bool ok = (jg + t * 128u) < TILE && jj < k;\n"
    "                for (uint q = 0; q < RP; q++) {\n"
    "                    if (ok && q < qmax)\n"];
    if (abl == 1) {
        [s appendString:
        @"                        { for (uint v = 0; v < 4; v++)\n"
        "                              wv[t][q][v] = (float)(jj + v + 1u) * 1e-7f; }\n"];
    } else {
        [s appendString:
        @"                        load_w(w + (ulong)(c0 + q) * k + jj, wv[t][q]);\n"];
    }
    [s appendString:
    @"                    else\n"
    "                        for (uint v = 0; v < 4; v++) wv[t][q][v] = 0.0f;\n"
    "                }\n"
    "            }\n"
    "            for (uint t = 0; t < PF; t++) {\n"
    "                const uint jj2 = jg + t * 128u;\n"
    "                const uint jj = base + jj2;\n"
    "                if (jj2 < TILE && jj < k) {\n"];

    if (abl == 2) {
        [s appendString:
        @"                    float xv[N][4];\n"
        "                    for (uint r = 0; r < N; r++)\n"
        "                        for (uint v = 0; v < 4; v++)\n"
        "                            xv[r][v] = (float)(r * 4u + v + 1u) * 1e-7f;\n"];
    } else {
        append_xv(s, mode, xs);
    }

    if (abl == 3) {
        [s appendString:
        @"                    for (uint q = 0; q < qmax; q++)\n"
        "                        s[q][0] += wv[t][q][0] + wv[t][q][1] +\n"
        "                                   wv[t][q][2] + wv[t][q][3] + xv[0][0];\n"];
    } else {
        [s appendString:
        @"                    for (uint q = 0; q < qmax; q++)\n"
        "                        for (uint r = 0; r < N; r++)\n"
        "                            for (uint v = 0; v < 4; v++)\n"
        "                                s[q][r] = fma(wv[t][q][v], xv[r][v], s[q][r]);\n"];
    }

    [s appendString:
    @"                }\n"
    "            }\n"
    "        }\n"];
    if (mode == 0) {
        [s appendString:@"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"];
    }
    [s appendString:
    @"    }\n"
    "    for (uint q = 0; q < qmax; q++)\n"
    "        for (uint r = 0; r < N; r++) {\n"
    "            const float v = simd_sum(s[q][r]);\n"
    "            if (lane == 0) out[(ulong)r * so + c0 + q] = v;\n"
    "        }\n"
    "}\n"];
    return s;
}

static const char *abl_name(int abl) {
    switch (abl) {
    case 0: return "all";
    case 1: return "no weight read";
    case 2: return "no act read";
    default: return "no act FMA";
    }
}

int main(int argc, char **argv) {
    int layers = argc > 1 ? atoi(argv[1]) : 32;
    int reps = argc > 2 ? atoi(argv[2]) : 3;
    if (layers < 1) layers = 32;

    const char *e;
    e = getenv("MODE"); int mode = (e && *e) ? atoi(e) : 0;
    e = getenv("PF");   int pf   = (e && *e) ? atoi(e) : 1;
    e = getenv("ABL");  int abl  = (e && *e) ? atoi(e) : 0;
    e = getenv("RPF");  int rpf  = (e && *e) ? atoi(e) : 0;
    e = getenv("TH");   int th   = (e && *e) ? atoi(e) : TILE;
    e = getenv("NSGV"); int nsgv = (e && *e) ? atoi(e) : NSG;
    e = getenv("XS");   int xs   = (e && *e) ? atoi(e) : 0;
    int nt = nsgv * 32;

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];

        id<MTLBuffer> W[S_N];
        for (int i = 0; i < S_N; i++) {
            size_t n = (size_t)SLAB[i].k * SLAB[i].cols;
            W[i] = [dev newBufferWithLength:n * 2 options:MTLResourceStorageModePrivate];
            if (!W[i]) { fprintf(stderr, "alloc %s failed\n", SLAB[i].name); return 1; }
        }
        id<MTLBuffer> X = [dev newBufferWithLength:(size_t)I * sizeof(float) * (MAXK + 1)
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
        printf("%s: %d layers, %.2f GB/pass | mode=%d pf=%d abl=%d(%s) rpf=%d th=%d nt=%d\n",
               dev.name.UTF8String, layers, moved / 1e9, mode, pf, abl,
               abl_name(abl), rpf, th, nt);
        printf("xs=%d\n", xs);

        int rps[4] = {1, 2, 4, 8};
        for (int ri = 0; ri < 4; ri++) {
            int rp = rpf > 0 ? rpf : rps[ri];
            printf("\n== RP %d (%d rows/threadgroup) ==\n", rp, nsgv * rp);
            printf("  k | ms/pass | GB/s | marginal ms | GFLOP/s\n");
            printf("  --+---------+------+-------------+--------\n");
            double prev = 0.0;
            for (int k = 1; k <= MAXK; k++) {
                NSError *err = nil;
                id<MTLLibrary> lib = [dev newLibraryWithSource:source_for(k, rp, abl, pf, mode, th, nsgv, xs)
                                                       options:nil error:&err];
                if (!lib) { fprintf(stderr, "  k=%d compile: %s\n", k,
                                    err.localizedDescription.UTF8String); break; }
                id<MTLFunction> fn = [lib newFunctionWithName:@"gemv_ab"];
                id<MTLComputePipelineState> pso =
                    [dev newComputePipelineStateWithFunction:fn error:&err];
                if (!pso) { fprintf(stderr, "  k=%d pso: %s\n", k,
                                    err.localizedDescription.UTF8String); break; }

                double best = 1e9;
                for (int rr = 0; rr < reps; rr++) {
                    double t0 = now_s();
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pso];
                    for (int l = 0; l < layers; l++) {
                        int attn = (l % 4) == 3;
                        for (int g = 0; g < 4; g++) {
                            int m[3]; int nr = group_members(attn, g, m);
                            for (int i = 0; i < nr; i++) {
                                struct { uint32_t k, cols, sx, so; } a;
                                a.k = SLAB[m[i]].k; a.cols = SLAB[m[i]].cols;
                                a.sx = SLAB[m[i]].k; a.so = SLAB[m[i]].cols;
                                [enc setBytes:&a length:sizeof(a) atIndex:0];
                                [enc setBuffer:W[m[i]] offset:0 atIndex:1];
                                [enc setBuffer:X offset:0 atIndex:2];
                                [enc setBuffer:O offset:0 atIndex:3];
                                uint32_t rt = (uint32_t)(nsgv * rp);
                                [enc dispatchThreadgroups:
                                         MTLSizeMake(1, (SLAB[m[i]].cols + rt - 1) / rt, 1)
                                             threadsPerThreadgroup:MTLSizeMake((NSUInteger)nt, 1, 1)];
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
                double gflop = (abl == 1 ? 0.0 : moved / 1e9 / 2.0) * (double)k * 2.0;
                printf("  %d | %7.1f | %4.0f | %11.1f | %7.0f\n",
                       k, best * 1e3, abl == 1 ? 0.0 : moved / best / 1e9, marg,
                       abl == 1 ? 0.0 : gflop / best);
            }
            if (rpf > 0) break;
        }
    }
    return 0;
}
