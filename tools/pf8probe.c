/*
 * pf8probe.c — settle the FP8 E4M3 dequantization question from checkpoint
 * bytes alone.
 *
 * The released GLM-5.3-Flash checkpoint stores an FP8 E4M3 matrix next to a
 * paired F32 `*_scale_inv` of shape [rows/128, cols/128]. "scale_inv" is an
 * ambiguous name: it can mean the number the *quantizer* multiplies by
 * (448 / amax, so dequant divides) or the number the *dequantizer* multiplies
 * by (amax / 448). The two differ by scale^2, so getting it wrong is not a
 * small drift — it is a different model.
 *
 * This tool does not assume either reading. It measures two things that
 * separate them:
 *
 *   1. how much of the E4M3 range each 128x128 block actually uses. A
 *      quantizer maps the block's amax onto 448, so under the "quantizer
 *      multiplies" reading max|q| inside a block is ~448; under the other
 *      reading the codes sit far below full scale and the quantizer would be
 *      throwing away most of its 4 exponent bits for no reason.
 *   2. the magnitude of the reconstructed weights under each reading. Both
 *      are printed next to a BF16 tensor from the same checkpoint, which is
 *      the only magnitude reference nothing here controls.
 *
 *   pf8probe <root> [fp8-tensor-name] [bf16-reference-name]
 */
#include "cwick.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* e4m3fn: 3 mantissa bits, bias 7, max normal 448. Mirrors the kernel in
 * tensor.c / metal_shader.c on purpose — the point is to read the bytes the
 * way the runtime reads them, not the way a library would. */
static float e4m3_to_f32(unsigned char b) {
    int sign = (b >> 7) & 1;
    int exp = (b >> 3) & 0xF;
    int man = b & 7;
    float v;
    if (exp == 0) v = (float)man * 0x1p-9f;
    else if (exp == 15 && man == 7) return NAN;
    else v = (1.0f + (float)man / 8.0f) * exp2f((float)(exp - 7));
    return sign ? -v : v;
}

static float bf16_to_f32(unsigned short h) {
    unsigned int u = (unsigned int)h << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

typedef struct {
    double sum_abs, sum_sq;
    double max_abs;
    int64_t n;
} acc;

static void acc_add(acc *a, float v) {
    double x = fabs((double)v);
    if (!isfinite(x)) return;
    a->sum_abs += x;
    a->sum_sq += x * x;
    if (x > a->max_abs) a->max_abs = x;
    a->n++;
}

static void acc_print(const char *label, const acc *a) {
    double mean = a->n ? a->sum_abs / a->n : 0.0;
    double rms = a->n ? sqrt(a->sum_sq / a->n) : 0.0;
    printf("  %-22s n=%lld  mean|w|=%.6g  rms=%.6g  max|w|=%.6g\n", label,
           (long long)a->n, mean, rms, a->max_abs);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: pf8probe <root> [fp8-tensor] [bf16-reference]\n");
        return 2;
    }
    const char *root = argv[1];
    const char *want = argc > 2
                           ? argv[2]
                           : "model.language_model.layers.45.mlp.experts.0.gate_proj.weight";
    const char *ref = argc > 3 ? argv[3] : NULL;

    cwick_error err;
    cwick_error_clear(&err);
    cwick_safetensors saf;
    if (cwick_safetensors_open(root, &saf, &err) != CWICK_OK) {
        fprintf(stderr, "pf8probe: cannot open %s: %s\n", root,
                cwick_error_message(&err));
        return 1;
    }

    cwick_st_tensor t;
    if (cwick_safetensors_get(&saf, want, &t, &err) != CWICK_OK) {
        fprintf(stderr, "pf8probe: no tensor %s: %s\n", want,
                cwick_error_message(&err));
        return 1;
    }
    printf("tensor %s\n  shape [%lld, %lld] dtype=%d\n", t.name,
           (long long)t.rows, (long long)t.cols, (int)t.dtype);
    if (t.dtype != CWICK_ST_F8_E4M3) {
        fprintf(stderr, "pf8probe: %s is not F8_E4M3 (dtype %d)\n", t.name,
                (int)t.dtype);
        return 1;
    }
    printf("  scale_inv [%lld, %lld] %s\n", (long long)t.scale_inv_rows,
           (long long)t.scale_inv_cols, t.scale_inv ? "present" : "MISSING");
    if (!t.scale_inv) return 1;

    const unsigned char *q = (const unsigned char *)t.data;
    const float *s = (const float *)t.scale_inv;
    const int64_t R = t.rows, C = t.cols;
    const int64_t SR = t.scale_inv_rows, SC = t.scale_inv_cols;
    if (SR <= 0 || SC <= 0) return 1;
    /* The block the scale grid implies, not the block the code assumes. */
    printf("  implied block [%lld, %lld]\n", (long long)(R / SR),
           (long long)(C / SC));

    /* 1. range usage per block */
    double bmax_sum = 0.0, bmax_min = 1e30, bmax_max = 0.0;
    int64_t nblocks = 0;
    for (int64_t br = 0; br < SR; br++) {
        for (int64_t bc = 0; bc < SC; bc++) {
            double bm = 0.0;
            for (int64_t i = br * (R / SR); i < (br + 1) * (R / SR); i++) {
                const unsigned char *row = q + i * C;
                for (int64_t j = bc * (C / SC); j < (bc + 1) * (C / SC); j++) {
                    float v = fabsf(e4m3_to_f32(row[j]));
                    if (!isfinite(v)) continue;
                    if (v > bm) bm = v;
                }
            }
            bmax_sum += bm;
            if (bm < bmax_min) bmax_min = bm;
            if (bm > bmax_max) bmax_max = bm;
            nblocks++;
        }
    }
    printf("\nE4M3 range usage per 128-block (448 = full scale):\n");
    printf("  mean max|q| = %.1f   min = %.1f   max = %.1f   (%lld blocks)\n",
           bmax_sum / (double)nblocks, bmax_min, bmax_max, (long long)nblocks);

    /* 2. scale magnitudes */
    double smin = 1e30, smax = 0.0, ssum = 0.0;
    int64_t sn = 0;
    for (int64_t i = 0; i < SR * SC; i++) {
        double v = fabs((double)s[i]);
        if (!isfinite(v) || v == 0.0) continue;
        if (v < smin) smin = v;
        if (v > smax) smax = v;
        ssum += v;
        sn++;
    }
    printf("\nscale_inv magnitudes:\n");
    printf("  min=%.6g  mean=%.6g  max=%.6g  (n=%lld)\n", smin,
           ssum / (double)sn, smax, (long long)sn);

    /* 3. both readings, sampled */
    acc a_mul = {0}, a_div = {0};
    uint64_t stride = 7919; /* prime, walks the whole matrix without a pattern */
    int64_t cnt = 0;
    for (int64_t flat = 0; flat < R * C; flat += stride) {
        int64_t i = flat / C, j = flat % C;
        float qv = e4m3_to_f32(q[flat]);
        if (!isfinite(qv)) continue;
        float sv = s[(i / (R / SR)) * SC + (j / (C / SC))];
        if (!isfinite(sv) || sv == 0.0f) continue;
        acc_add(&a_mul, qv * sv);
        acc_add(&a_div, qv / sv);
        cnt++;
    }
    printf("\nreconstructed weights (%lld samples):\n", (long long)cnt);
    acc_print("w = q * scale_inv", &a_mul);
    acc_print("w = q / scale_inv", &a_div);

    if (ref) {
        cwick_st_tensor r;
        cwick_error_clear(&err);
        if (cwick_safetensors_get(&saf, ref, &r, &err) == CWICK_OK) {
            printf("\nBF16 reference %s [%lld, %lld] dtype=%d\n", r.name,
                   (long long)r.rows, (long long)r.cols, (int)r.dtype);
            acc ar = {0};
            if (r.dtype == CWICK_ST_BF16) {
                const unsigned short *p = (const unsigned short *)r.data;
                for (int64_t flat = 0; flat < r.rows * r.cols; flat += stride)
                    acc_add(&ar, bf16_to_f32(p[flat]));
            } else if (r.dtype == CWICK_ST_F32) {
                const float *p = (const float *)r.data;
                for (int64_t flat = 0; flat < r.rows * r.cols; flat += stride)
                    acc_add(&ar, p[flat]);
            } else {
                printf("  (reference is dtype %d, not BF16/F32 — skipped)\n",
                       (int)r.dtype);
            }
            if (ar.n) acc_print("reference", &ar);
        } else {
            printf("\nBF16 reference %s unavailable: %s\n", ref,
                   cwick_error_message(&err));
        }
    }

    cwick_safetensors_close(&saf);
    return 0;
}
