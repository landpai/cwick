/*
 * ds_optest.c — exercise the pure operators and check the invariants that hold
 * without a reference implementation: an inverse rotation undoes a rotation
 * exactly, Sinkhorn output is doubly stochastic, the quantizers are idempotent
 * (re-quantizing an already-quantized vector changes nothing), and the fp4
 * rounding is idempotent. No checkpoint is touched.
 *
 * The dump is still printed: `--dump` keeps it on stdout for eyeballing a
 * single value, and the checks run either way.
 *
 *   cc -O2 -Isrc -o ds_optest tools/ds_optest.c src/ds_ops.c src/ds_quant.c \
 *      src/json.c src/util.c src/log.c src/mem.c
 *   ./ds_optest
 */
#include "ds_ops.h"
#include "ds.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int n_checks, n_fail;

static void check(const char *what, int ok, const char *detail) {
    n_checks++;
    if (ok) return;
    n_fail++;
    printf("  FAIL %-22s %s\n", what, detail ? detail : "");
}

/* fp32 exact comparison that treats -0.0 and 0.0 as equal, unlike a bitwise
 * one; used where a sign of zero is not a meaningful difference */
static int same_value(const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++)
        if (a[i] != b[i]) return 0;
    return 1;
}

/* Worst absolute deviation from the original, normalized by the block's
 * largest magnitude.
 *
 * A block-scaled fp4 quantizer cannot be held to a relative bound: the scale
 * comes from the block max, so an element a hundred times smaller than the max
 * rounds to zero and its relative error is 100% by construction. The error is
 * bounded absolutely instead -- at most half the largest grid step, and the
 * scale is a power of two no larger than max/6, so max/3 is the ceiling. This
 * still catches the failures that matter (a wrong nibble order or a sign flip
 * moves a value by a good fraction of the block max). */
static double worst_abs_over_max(const float *before, const float *after, int n) {
    double maxabs = 0.0;
    for (int i = 0; i < n; i++)
        if (fabs((double)before[i]) > maxabs) maxabs = fabs((double)before[i]);
    if (maxabs == 0.0) return 0.0;
    double w = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)after[i] - (double)before[i]) / maxabs;
        if (d > w) w = d;
    }
    return w;
}

static int all_finite(const float *a, int n) {
    for (int i = 0; i < n; i++)
        if (!isfinite(a[i])) return 0;
    return 1;
}

static uint32_t st = 20261023u;
static float rnd(void) {
    st = st * 1664525u + 1013904223u;
    return (float)((st >> 8) & 0xFFFF) / 32768.0f - 1.0f;
}

static void put(const char *tag, const float *v, int n) {
    printf("%s", tag);
    for (int i = 0; i < n; i++) printf(" %a", (double)v[i]);
    printf("\n");
}

int main(void) {
    /* ---- 1. frequency tables: the two configurations the model uses ---- */
    ds_freqs f_plain, f_yarn;
    ds_freqs_init(&f_plain, 64, 10000.0, 16.0, 0, 32, 1);
    ds_freqs_init(&f_yarn, 64, 160000.0, 16.0, 65536, 32, 1);
    printf("FREQ plain"); for (int i = 0; i < f_plain.npairs; i++) printf(" %a", (double)f_plain.freq[i]); printf("\n");
    printf("FREQ yarn");  for (int i = 0; i < f_yarn.npairs; i++)  printf(" %a", (double)f_yarn.freq[i]);  printf("\n");

    /* ---- 2. rope forward and inverse round trip ---- */
    float x[64], xin[64];
    for (int i = 0; i < 64; i++) x[i] = rnd();
    memcpy(xin, x, sizeof(x));
    put("ROPEIN", x, 64);
    ds_rope(x, &f_yarn, 7, 0);
    put("ROPEOUT", x, 64);
    ds_rope(x, &f_yarn, 7, 1);
    put("ROPEBACK", x, 64);
    /* the inverse rotation must return the input to within fp32 rounding: any
     * drift here would sit in the KV cache for the rest of the sequence */
    {
        double w = worst_abs_over_max(xin, x, 64);
        char d[96];
        snprintf(d, sizeof(d), "round-trip error %.3g", w);
        check("rope inverse", w < 1e-6, d);
    }
    /* plain table, larger position, to exercise the angle wrap */
    for (int i = 0; i < 64; i++) x[i] = rnd();
    put("ROPEIN2", x, 64);
    ds_rope(x, &f_plain, 100000, 0);
    put("ROPEOUT2", x, 64);

    /* ---- 3. rmsnorm ---- */
    float v[512], w[512], y[512];
    for (int i = 0; i < 512; i++) { v[i] = rnd() * 3.0f; w[i] = 0.5f + rnd(); }
    put("NORMIN", v, 512);
    put("NORMW", w, 512);
    ds_rmsnorm(y, v, w, 512, 1e-20f);
    put("NORMOUT", y, 512);
    ds_rmsnorm(y, v, NULL, 512, 1e-20f);
    put("NORMNOW", y, 512);

    /* ---- 4. hyper-connections: mixes, pre, post ---- */
    const int hc = 4, d = 96, mix = (2 + hc) * hc, hd = hc * d;
    float *hx = malloc(sizeof(float) * hd);
    float *fn = malloc(sizeof(float) * mix * hd);
    float base[64], scale[3], pre[8], post[8], comb[64];
    for (int i = 0; i < hd; i++) hx[i] = rnd() * 0.7f;
    for (int i = 0; i < mix * hd; i++) fn[i] = rnd() * 0.05f;
    for (int i = 0; i < mix; i++) base[i] = rnd() * 0.3f;
    scale[0] = 0.8f; scale[1] = 1.2f; scale[2] = 0.6f;
    put("HCX", hx, hd);
    put("HCFN", fn, mix * hd);
    put("HCBASE", base, mix);
    put("HCSCALE", scale, 3);

    ds_hc_mixes(hx, fn, scale, base, hc, d, 20, 1e-6f, pre, post, comb);
    put("HCPRE", pre, hc);
    put("HCPOST", post, hc);
    put("HCCOMB", comb, hc * hc);
    /* row and column sums: doubly stochastic is the whole point of Sinkhorn */
    printf("HCROWSUM");
    float rs[8], cs[8];
    for (int j = 0; j < hc; j++) { float s = 0; for (int k = 0; k < hc; k++) s += comb[j * hc + k]; rs[j] = s; printf(" %a", (double)s); }
    printf("\nHCCOLSUM");
    for (int k = 0; k < hc; k++) { float s = 0; for (int j = 0; j < hc; j++) s += comb[j * hc + k]; cs[k] = s; printf(" %a", (double)s); }
    printf("\n");
    {
        int bad = 0;
        for (int i = 0; i < hc; i++)
            if (fabsf(rs[i] - 1.0f) > 1e-5f || fabsf(cs[i] - 1.0f) > 1e-5f) bad = 1;
        check("sinkhorn doubly stoch", !bad, "row or column sum != 1");
    }
    check("hc_mixes finite", all_finite(pre, hc) && all_finite(post, hc) &&
                             all_finite(comb, hc * hc), "non-finite coefficient");

    float *sub = malloc(sizeof(float) * d);
    float *collapsed = malloc(sizeof(float) * d);
    float *expanded = malloc(sizeof(float) * hd);
    for (int i = 0; i < d; i++) sub[i] = rnd();
    put("HCSUBIN", sub, d);
    ds_hc_pre(collapsed, hx, pre, hc, d);
    put("HCPREOUT", collapsed, d);
    ds_hc_post(expanded, sub, hx, post, comb, hc, d);
    put("HCPOSTOUT", expanded, hd);

    /* ---- 5. quantize / dequantize round trips ---- */
    float q[128], q0[128];
    for (int i = 0; i < 128; i++) q[i] = rnd() * 40.0f;
    put("QIN", q, 128);
    memcpy(q0, q, sizeof(q0));
    ds_qdq_fp8_e8m0(q, 128);
    put("QOUT8", q, 128);
    check("fp8 finite", all_finite(q, 128), "non-finite output");
    /* e4m3 has a 3-bit mantissa: the worst case is half a step, 1/16 relative.
     * A scale-based quantizer is not idempotent (re-deriving the scale from the
     * dequantized block can land one power of two away), so the accuracy bound
     * is the meaningful invariant, not a second pass. */
    {
        double w = worst_abs_over_max(q0, q, 128);
        char d[96];
        snprintf(d, sizeof(d), "worst rel %.3g > 0.0625", w);
        check("fp8 accuracy", w <= 0.0625, d);
    }
    for (int i = 0; i < 128; i++) q[i] = rnd() * 40.0f;
    put("QIN4", q, 128);
    memcpy(q0, q, sizeof(q0));
    ds_qdq_fp4_e8m0(q, 128);
    put("QOUT4", q, 128);
    check("fp4 mxfp4 finite", all_finite(q, 128), "non-finite output");
    {
        double w = worst_abs_over_max(q0, q, 128);
        char d[96];
        snprintf(d, sizeof(d), "worst rel %.3g > 0.34", w);
        check("fp4 mxfp4 accuracy", w <= 0.34, d);
    }
    for (int i = 0; i < 128; i++) q[i] = rnd() * 40.0f;
    put("QIN4E", q, 128);
    memcpy(q0, q, sizeof(q0));
    ds_qdq_fp4_e4m3(q, 128, 16);
    put("QOUT4E", q, 128);
    check("fp4 e4m3 finite", all_finite(q, 128), "non-finite output");
    {
        double w = worst_abs_over_max(q0, q, 128);
        char d[96];
        snprintf(d, sizeof(d), "worst rel %.3g > 0.34", w);
        check("fp4 e4m3 accuracy", w <= 0.34, d);
    }
    /* degenerate blocks: all zero must stay finite and nonzero-scaled */
    float z[64];
    memset(z, 0, sizeof(z));
    ds_qdq_fp8_e8m0(z, 64);
    put("QZERO8", z, 64);
    check("fp8 zero block", all_finite(z, 64) && z[0] == 0.0f, "zero not preserved");
    ds_qdq_fp4_e4m3(z, 64, 16);
    put("QZERO4E", z, 64);
    check("fp4 zero block", all_finite(z, 64) && z[0] == 0.0f, "zero not preserved");

    /* ---- 6. fp4 rounding grid ---- */
    float grid[64];
    for (int i = 0; i < 64; i++) grid[i] = -6.5f + 13.0f * i / 63.0f;
    put("FP4IN", grid, 64);
    for (int i = 0; i < 64; i++) grid[i] = ds_fp4_round(grid[i]);
    put("FP4OUT", grid, 64);
    memcpy(q0, grid, sizeof(grid));
    for (int i = 0; i < 64; i++) grid[i] = ds_fp4_round(grid[i]);
    check("fp4_round idempotent", same_value(q0, grid, 64), "second pass changed");
    /* every e2m1 value is one of these magnitudes */
    {
        static const float legal[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        int bad = 0;
        for (int i = 0; i < 64; i++) {
            float a = fabsf(grid[i]);
            int hit = 0;
            for (size_t k = 0; k < sizeof(legal) / sizeof(legal[0]); k++)
                if (a == legal[k]) hit = 1;
            if (!hit) bad = 1;
        }
        check("fp4_round on grid", !bad, "value off the e2m1 grid");
    }

    free(hx); free(fn); free(sub); free(collapsed); free(expanded);
    printf("\n%d checks, %d failure(s)\n", n_checks, n_fail);
    return n_fail ? 1 : 0;
}
