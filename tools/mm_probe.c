/*
 * mm_probe.c -- read the MiniMax-M3.1-Flash quantizer constants straight out
 * of the shards and print what a decoded weight actually looks like.
 *
 *   mm_probe <checkpoint-root> [layer] [expert]
 *
 * The checkpoint stores two quantizer layouts and neither of them is
 * documented in the config:
 *
 *   dense (attention, shared expert, indexer)
 *       FP8 e4m3 payload + one U8 (e8m0, power-of-two) scale per 32 K
 *       elements, per row.                       [N,K] F8_E4M3 + [N,K/32] U8
 *
 *   routed experts
 *       FP4 e2m1 packed two per byte along K, one e4m3 scale per 16 K
 *       elements, plus a per-tensor F32 `weight_global_scale` and a second
 *       per-tensor F32 `input_global_scale`.
 *                       [N,K/2] U8 + [N,K/16] F8_E4M3 + two F32 scalars
 *
 * Only the second one is ambiguous: the global scale could be a multiplier
 * (w = fp4 * block * global) or a divisor (w = fp4 * block / global). Both
 * give finite numbers, so the check is the magnitude of the decoded tensor --
 * a 6144-wide projection has weights around 0.01, not around 1e-4 or 1e2.
 * Everything downstream depends on getting this one sign right, so the tool
 * prints both.
 */
#include "cwick.h"
#include "ds.h"
#include "json.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;

static void need(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "mm_probe: %s\n", what); g_fail = 1; }
}

/* Pull one tensor by its full checkpoint name. */
static const cwick_st_tensor *T(const cwick_safetensors *saf, const char *name) {
    static cwick_st_tensor slots[8];
    static int n = 0;
    cwick_error err;
    cwick_error_clear(&err);
    if (n >= 8) return NULL;
    cwick_st_tensor *out = &slots[n];
    if (cwick_safetensors_get(saf, name, out, &err) != CWICK_OK) {
        fprintf(stderr, "mm_probe: missing tensor %s (%s)\n", name, err.message);
        g_fail = 1;
        return NULL;
    }
    n++;
    return out;
}

static void stats(const char *label, const float *v, int64_t n) {
    double s = 0, s2 = 0, mx = 0;
    int64_t nz = 0;
    for (int64_t i = 0; i < n; i++) {
        double x = v[i];
        s += x; s2 += x * x;
        double a = fabs(x);
        if (a > mx) mx = a;
        if (a != 0) nz++;
    }
    double mean = s / (double)n;
    double sd = sqrt(s2 / (double)n - mean * mean);
    printf("    %-22s mean %+.6f  sd %.6f  max %.5f  nonzero %.1f%%\n",
           label, mean, sd, mx, 100.0 * (double)nz / (double)n);
}

/* Decode one row of an NVFP4 matrix. `gmul` picks the global-scale
 * convention: 1 -> w = fp4*block*global, 0 -> w = fp4*block/global. */
static void nvfp4_row(const uint8_t *packed, const uint8_t *escale, int64_t K,
                      float global, int gmul, float *out) {
    int64_t nb = K / 16;                 /* blocks per row */
    for (int64_t b = 0; b < nb; b++) {
        float sc = ds_e4m3_to_f32(escale[b]);
        sc = gmul ? sc * global : sc / global;
        for (int64_t j = 0; j < 16; j++) {
            int64_t k = b * 16 + j;
            uint8_t nib = (k & 1) ? (packed[k >> 1] >> 4) : (packed[k >> 1] & 0x0F);
            out[k] = ds_fp4_to_f32(nib) * sc;
        }
    }
}

/* Decode one row of the dense mxfp8 layout: e4m3 payload, e8m0 per-32. */
static void mxfp8_row(const uint8_t *w, const uint8_t *s, int64_t K, float *out) {
    for (int64_t b = 0; b * 32 < K; b++) {
        float sc = ds_e8m0_to_f32(s[b]);
        for (int64_t j = 0; j < 32 && b * 32 + j < K; j++)
            out[b * 32 + j] = ds_e4m3_to_f32(w[b * 32 + j]) * sc;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: mm_probe <checkpoint-root> [layer] [expert]\n");
        return 2;
    }
    const char *root = argv[1];
    int layer = argc > 2 ? atoi(argv[2]) : 59;
    int expert = argc > 3 ? atoi(argv[3]) : 3;

    cwick_safetensors saf;
    cwick_error err;
    cwick_error_clear(&err);
    if (cwick_safetensors_open(root, &saf, &err) != CWICK_OK) {
        fprintf(stderr, "mm_probe: %s\n", err.message);
        return 1;
    }
    printf("root %s\n", root);
    printf("  tensors %zu  shards %zu  %.1f GiB mapped\n",
           saf.tensor_count, saf.shard_count,
           (double)saf.total_bytes / (1024.0 * 1024.0 * 1024.0));

    /* ---- routed expert ---- */
    char nm[256];
    printf("\nNVFP4 routed expert (layer %d, expert %d, w1 = gate_proj):\n",
           layer, expert);
    snprintf(nm, sizeof(nm),
             "language_model.model.layers.%d.block_sparse_moe.experts.%d.w1.weight_packed",
             layer, expert);
    const cwick_st_tensor *tp = T(&saf, nm);
    snprintf(nm, sizeof(nm),
             "language_model.model.layers.%d.block_sparse_moe.experts.%d.w1.weight_scale",
             layer, expert);
    const cwick_st_tensor *ts = T(&saf, nm);
    snprintf(nm, sizeof(nm),
             "language_model.model.layers.%d.block_sparse_moe.experts.%d.w1.weight_global_scale",
             layer, expert);
    const cwick_st_tensor *tg = T(&saf, nm);
    snprintf(nm, sizeof(nm),
             "language_model.model.layers.%d.block_sparse_moe.experts.%d.w1.input_global_scale",
             layer, expert);
    const cwick_st_tensor *ti = T(&saf, nm);
    if (!tp || !ts || !tg || !ti) return 1;

    float wg = ((const float *)tg->data)[0];
    float ig = ((const float *)ti->data)[0];
    printf("  packed [%lld,%lld] %s   scale [%lld,%lld] %s\n",
           (long long)tp->rows, (long long)tp->cols,
           tp->dtype == CWICK_ST_U8 ? "U8" : "?",
           (long long)ts->rows, (long long)ts->cols,
           ts->dtype == CWICK_ST_F8_E4M3 ? "F8_E4M3" : "?");
    printf("  weight_global_scale = %.8g    input_global_scale = %.8g\n", wg, ig);
    printf("  product of the two  = %.8g\n", wg * ig);

    /* block-scale statistics, decoded */
    {
        int64_t nb = ts->rows * ts->cols;
        const uint8_t *sb = (const uint8_t *)ts->data;
        float mn = 1e30f, mx = -1e30f; double sum = 0;
        for (int64_t i = 0; i < nb; i++) {
            float v = ds_e4m3_to_f32(sb[i]);
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
        }
        printf("  per-16 e4m3 block scale: min %.6g max %.6g mean %.6g\n",
               mn, mx, sum / (double)nb);
    }

    int64_t K = tp->cols * 2;            /* logical K of w1 */
    int64_t N = tp->rows;
    printf("  logical shape [%lld,%lld]\n", (long long)N, (long long)K);

    float *row = (float *)malloc((size_t)K * sizeof(float));
    need(row != NULL, "out of memory");
    if (!row) return 1;

    int64_t probe_rows[3] = {0, N / 2, N - 1};
    for (int pass = 0; pass < 2; pass++) {
        printf("  global scale as %s:\n", pass ? "MULTIPLIER" : "DIVISOR");
        float *acc = (float *)calloc((size_t)K, sizeof(float));
        double sq = 0; int64_t cnt = 0;
        for (int r = 0; r < (int)(sizeof(probe_rows) / sizeof(probe_rows[0])); r++) {
            int64_t ri = probe_rows[r];
            nvfp4_row((const uint8_t *)tp->data + ri * tp->cols,
                      (const uint8_t *)ts->data + ri * ts->cols,
                      K, wg, pass, row);
            stats("one row", row, K);
            for (int64_t i = 0; i < K; i++) { acc[i] += row[i]; sq += (double)row[i] * row[i]; cnt++; }
        }
        printf("    %-22s rms %.6f  (a 6144-wide projection sits near 0.01)\n",
               "3 rows pooled", sqrt(sq / (double)cnt));
        free(acc);
    }

    /* ---- dense mxfp8 ---- */
    printf("\nMXFP8 dense (layer %d, self_attn.q_proj):\n", layer);
    snprintf(nm, sizeof(nm),
             "language_model.model.layers.%d.self_attn.q_proj.weight", layer);
    const cwick_st_tensor *qw = T(&saf, nm);
    snprintf(nm, sizeof(nm),
             "language_model.model.layers.%d.self_attn.q_proj.weight_scale_inv", layer);
    const cwick_st_tensor *qs = T(&saf, nm);
    if (qw && qs) {
        printf("  weight [%lld,%lld] %s   scale_inv [%lld,%lld] %s (ratio %.1f)\n",
               (long long)qw->rows, (long long)qw->cols,
               qw->dtype == CWICK_ST_F8_E4M3 ? "F8_E4M3" : "?",
               (long long)qs->rows, (long long)qs->cols,
               qs->dtype == CWICK_ST_U8 ? "U8" : "?",
               (double)(qw->rows * qw->cols) / (double)(qs->rows * qs->cols));
        int64_t K2 = qw->cols;
        float *r2 = (float *)malloc((size_t)K2 * sizeof(float));
        double sq = 0; int64_t cnt = 0;
        for (int64_t ri = 0; ri < qw->rows; ri += qw->rows / 3) {
            mxfp8_row((const uint8_t *)qw->data + ri * K2,
                      (const uint8_t *)qs->data + ri * (K2 / 32), K2, r2);
            stats("one row", r2, K2);
            for (int64_t i = 0; i < K2; i++) { sq += (double)r2[i] * r2[i]; cnt++; }
        }
        printf("    %-22s rms %.6f\n", "3 rows pooled", sqrt(sq / (double)cnt));
        free(r2);
    }

    /* ---- norms: is it gemma (1 + w)? ---- */
    printf("\nRMSNorm weights (gemma style is 1+w, so these sit near 0):\n");
    snprintf(nm, sizeof(nm),
             "language_model.model.layers.%d.input_layernorm.weight", layer);
    const cwick_st_tensor *nw = T(&saf, nm);
    if (nw) {
        const uint16_t *b = (const uint16_t *)nw->data;
        float mn = 1e30f, mx = -1e30f; double sum = 0;
        for (int64_t i = 0; i < nw->rows * nw->cols; i++) {
            float v = ds_bf16_to_f32(b[i]);
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
        }
        printf("  input_layernorm [%lld]: min %+.5f max %+.5f mean %+.5f\n",
               (long long)(nw->rows * nw->cols), mn, mx,
               sum / (double)(nw->rows * nw->cols));
    }

    /* ---- router: dtype and range ---- */
    printf("\nRouter:\n");
    snprintf(nm, sizeof(nm),
             "language_model.model.layers.%d.block_sparse_moe.gate.weight", layer);
    const cwick_st_tensor *gw = T(&saf, nm);
    if (gw) {
        printf("  gate.weight [%lld,%lld] dtype %d\n", (long long)gw->rows,
               (long long)gw->cols, (int)gw->dtype);
        const float *f = (const float *)gw->data;
        float mn = 1e30f, mx = -1e30f;
        for (int64_t i = 0; i < gw->rows * gw->cols; i++) {
            if (f[i] < mn) mn = f[i];
            if (f[i] > mx) mx = f[i];
        }
        printf("    f32 range [%+.5f, %+.5f]\n", mn, mx);
    }
    snprintf(nm, sizeof(nm),
             "language_model.model.layers.%d.block_sparse_moe.e_score_correction_bias",
             layer);
    const cwick_st_tensor *eb = T(&saf, nm);
    if (eb) {
        const float *f = (const float *)eb->data;
        float mn = 1e30f, mx = -1e30f;
        for (int64_t i = 0; i < eb->rows * eb->cols; i++) {
            if (f[i] < mn) mn = f[i];
            if (f[i] > mx) mx = f[i];
        }
        printf("  e_score_correction_bias [%lld] f32 range [%+.5f, %+.5f]\n",
               (long long)(eb->rows * eb->cols), mn, mx);
    }

    /* ---- inventory: every non-expert tensor of the layer, with geometry ----
     * The names are the contract with the loader, and several of them carry
     * their dtype in the shape rather than in the config (the per-head norms,
     * the shared expert, the routing bias). Printing the whole layer is faster
     * than guessing which of them exist. */
    printf("\nLayer %d inventory (non-expert, in shard order):\n", layer);
    for (size_t i = 0; i < saf.tensor_count; i++) {
        const cwick_st_tensor *t = &saf.tensors[i];
        char want[128];
        snprintf(want, sizeof(want), "language_model.model.layers.%d.", layer);
        if (strncmp(t->name, want, strlen(want)) != 0) continue;
        if (strstr(t->name, ".experts.")) continue;
        /* cwick_st_dtype: BF16, F32, F8_E4M3, U8, I8, F8_E8M0 */
        static const char *dn[] = {"BF16", "F32", "F8E4M3", "U8", "I8", "E8M0"};
        printf("  %-96s [%lld,%lld] %-7s %llu B\n", t->name,
               (long long)t->rows, (long long)t->cols,
               dn[(int)t->dtype < 13 ? (int)t->dtype : 0],
               (unsigned long long)t->data_bytes);
    }

    free(row);
    cwick_safetensors_close(&saf);
    return g_fail ? 1 : 0;
}
