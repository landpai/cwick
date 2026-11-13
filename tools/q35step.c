/* q35step — can the GPU carry one Qwen3.5 decode step, and by how much?
 *
 * A dense BF16 decode step is a stream of ~16.5 GB of weights read once. The
 * CPU pool reads at ~122 GB/s on this host (tools/membwbench) and the GPU at
 * ~284 GB/s (tools/gpubw), so the arithmetic says the GPU could halve the step
 * -- but the projections of one layer depend on each other, so a GPU path pays
 * a commit+wait between every group. This tool measures the real thing: the
 * checkpoint's shape mix, batched through the production API, with the same
 * number of round trips a real step would make.
 *
 *   make gemvstep && ./tools/q35step [layers] [sets]
 */
#include "cwick.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define D 4096
#define QW 8192      /* q_proj with attn_output_gate: 2 x dim */
#define KV 1024      /* 4 kv heads x 256 */
#define QKV 8192     /* linear_attn.in_proj_qkv */
#define ZD 4096      /* linear_attn.in_proj_z */
#define I 12288      /* mlp intermediate */

typedef struct { const char *name; int64_t k, cols; } proj;

/* Slab layout inside one weight set. Order matters: the group walk indexes it
 * arithmetically so the loops below stay readable. */
enum { S_QLIN = 0, S_ZLIN, S_QATTN, S_KATTN, S_VATTN, S_GATE, S_UP, S_DOWN, S_OUT, S_N };

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

/* The four groups a layer's dependency chain allows to share one command
 * buffer. `attn` picks the input-projection variant; the mlp groups are common
 * to both layer types. */
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

static uint16_t *alloc_bf16(int64_t elems) {
    void *p = NULL;
    if (posix_memalign(&p, 16384, (size_t)elems * 2) != 0 || !p) {
        fprintf(stderr, "alloc %lld elems failed\n", (long long)elems);
        exit(1);
    }
    uint16_t *w = (uint16_t *)p;
    for (int64_t i = 0; i < elems; i++) w[i] = (uint16_t)(0x3f00u + (i & 0x7fu));
    return w;
}

int main(int argc, char **argv) {
    int layers = argc > 1 ? atoi(argv[1]) : 32;
    int sets = argc > 2 ? atoi(argv[2]) : 2;
    if (layers < 1) layers = 32;
    if (sets < 1) sets = 1;

    if (cwick_metal_force(1) != 0) {
        fprintf(stderr, "metal init failed\n");
        return 1;
    }

    uint16_t **w = calloc((size_t)sets * S_N, sizeof(*w));
    size_t bytes_alloc = 0;
    for (int s = 0; s < sets; s++)
        for (int k = 0; k < S_N; k++) {
            w[(size_t)s * S_N + k] = alloc_bf16(SLAB[k].k * SLAB[k].cols);
            bytes_alloc += (size_t)SLAB[k].k * (size_t)SLAB[k].cols * 2;
        }

    float *x = calloc((size_t)I, sizeof(float));
    float *out = calloc((size_t)I, sizeof(float));
    for (int i = 0; i < I; i++) x[i] = 0.001f * (float)(i & 15);

    double bytes = 0.0;
    for (int l = 0; l < layers; l++) {
        int attn = (l % 4) == 3;
        for (int g = 0; g < 4; g++) {
            int members[3];
            int n = group_members(attn, g, members);
            for (int i = 0; i < n; i++)
                bytes += (double)SLAB[members[i]].k * (double)SLAB[members[i]].cols * 2.0;
        }
    }
    printf("%d layers (%d linear / %d attention), %.2f GB moved, %d weight sets "
           "(%.1f GB allocated)\n", layers, layers - layers / 4, layers / 4,
           bytes / 1e9, sets, (double)bytes_alloc / 1e9);

    /* ---- CPU, same shapes and same order ---- */
    double t0 = now_s();
    for (int l = 0; l < layers; l++) {
        int attn = (l % 4) == 3;
        int set = (l / 4) % sets;
        for (int g = 0; g < 4; g++) {
            int members[3];
            int n = group_members(attn, g, members);
            for (int i = 0; i < n; i++) {
                const proj *p = &SLAB[members[i]];
                cwick_gemm_bf16(out, 1, p->k, p->cols, x, w[(size_t)set * S_N + members[i]]);
            }
        }
    }
    double cpu_ms = (now_s() - t0) * 1e3;

    /* ---- GPU, one command buffer per group ---- */
    int commits = 0, refused = 0;
    void *probe = cwick_metal_bf16_batch_begin();
    if (!probe) { fprintf(stderr, "no batch slot\n"); return 1; }
    cwick_metal_bf16_batch_end(probe);

    t0 = now_s();
    for (int l = 0; l < layers; l++) {
        int attn = (l % 4) == 3;
        int set = (l / 4) % sets;
        for (int g = 0; g < 4; g++) {
            int members[3];
            int n = group_members(attn, g, members);
            void *b = cwick_metal_bf16_batch_begin();
            if (!b) { fprintf(stderr, "no batch slot at layer %d group %d\n", l, g); return 1; }
            for (int i = 0; i < n; i++) {
                const proj *p = &SLAB[members[i]];
                if (!cwick_metal_bf16_batch_add(b, w[(size_t)set * S_N + members[i]],
                                                x, out, p->k, p->cols))
                    refused++;
            }
            if (!cwick_metal_bf16_batch_end(b)) refused++;
            commits++;
        }
    }
    double gpu_ms = (now_s() - t0) * 1e3;

    printf("\n%-6s %10s %10s %16s\n", "path", "ms", "GB/s", "step tok/s@16.5GB");
    printf("%-6s %10.1f %10.1f %16.2f\n", "cpu", cpu_ms,
           bytes / cpu_ms * 1e3 / 1e9, 16500.0 / cpu_ms);
    printf("%-6s %10.1f %10.1f %16.2f\n", "gpu", gpu_ms,
           bytes / gpu_ms * 1e3 / 1e9, 16500.0 / gpu_ms);
    printf("\n%d commits, %.0f us per commit+wait, %d refused\n",
           commits, gpu_ms * 1e3 / (double)commits, refused);
    printf("speedup %.2fx\n", cpu_ms / gpu_ms);
    return refused ? 2 : 0;
}
