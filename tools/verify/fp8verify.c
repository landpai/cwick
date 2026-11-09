/* Correctness check for the FP8 E4M3 GEMV.
 *
 * Prints the worst relative error against a double-precision reference that
 * dequantizes with the exported scalar converter, so the NEON kernel (the
 * default) and the scalar reference (CWICK_GEMV_SCALAR=1) are checked by the
 * same yardstick. Also reports how many bytes in the test matrix fall into
 * the three awkward classes: exponent zero (zero and subnormals) and the
 * NaN pattern. */
#include "cwick.h"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

int main(void) {
    setenv("CWICK_THREADS", "1", 1);
    const int64_t k = 4096, cols = 256, ncb = 32;
    size_t wbytes = (size_t)k * (size_t)cols;
    size_t sbytes = (size_t)(cols / 128) * (size_t)ncb * sizeof(float);

    uint8_t *w = cwick_xmalloc(wbytes);
    float *sc = cwick_xmalloc(sbytes);
    float *x = cwick_xmalloc((size_t)k * sizeof(float));
    float *out = cwick_xmalloc((size_t)cols * sizeof(float));

    /* Deliberately biased: plenty of zeros, subnormals and full-scale values,
     * which is far harsher than a real expert matrix. */
    int n_zero = 0, n_sub = 0, n_nan = 0;
    for (size_t i = 0; i < wbytes; i++) {
        uint64_t r = rnd() % 1000;
        uint8_t b;
        if (r < 40)       b = (uint8_t)((rnd() & 0x80) | 0x00);        /* +/- zero */
        else if (r < 80)  b = (uint8_t)((rnd() & 0x87) | 0x00);        /* subnormal */
        else if (r < 82)  b = (uint8_t)((rnd() & 0x80) | 0x7F);        /* NaN pattern */
        else              b = (uint8_t)(rnd() & 0xFF);
        int e = (b >> 3) & 0xF, m = b & 7;
        if (e == 0 && m == 0) n_zero++;
        else if (e == 0) n_sub++;
        else if (e == 15 && m == 7) n_nan++;
        if (e == 15 && m == 7) b = (uint8_t)(b & 0x7E); /* keep the ref finite */
        w[i] = b;
    }
    for (size_t i = 0; i < sbytes / sizeof(float); i++)
        sc[i] = 0.25f + (float)(rnd() % 1000) * 0.001f;
    for (int64_t i = 0; i < k; i++)
        x[i] = (float)((int64_t)(rnd() % 2000) - 1000) * 0.01f;

    cwick_gemv_fp8(out, k, cols, x, w, sc, ncb);

    double worst = 0.0;
    int nan_out = 0;
    for (int64_t c = 0; c < cols; c++) {
        const uint8_t *row = w + (size_t)c * (size_t)k;
        const float *srow = sc + (size_t)(c / 128) * (size_t)ncb;
        double ref = 0.0;
        for (int64_t j = 0; j < k; j++)
            ref += (double)cwick_fp8_e4m3_to_f32(row[j]) * (double)x[j] *
                   (double)srow[j / 128];
        if (isnan(out[c])) { nan_out++; continue; }
        double den = fabs(ref) > 1e-6 ? fabs(ref) : 1e-6;
        double e = fabs((double)out[c] - ref) / den;
        if (e > worst) worst = e;
    }

    printf("weights: %zu bytes — zero %d (%.3f%%), subnormal %d (%.3f%%), "
           "nan-pattern %d (%.3f%%)\n",
           wbytes, n_zero, 100.0 * n_zero / wbytes,
           n_sub, 100.0 * n_sub / wbytes, n_nan, 100.0 * n_nan / wbytes);
    printf("worst relative error vs double reference: %.3e\n", worst);
    printf("non-finite outputs: %d\n", nan_out);
    printf("checksum: ");
    uint64_t h = 1469598103934665603ULL;
    for (int64_t c = 0; c < cols; c++) {
        uint32_t b; memcpy(&b, &out[c], 4);
        h ^= b; h *= 1099511628211ULL;
    }
    printf("%016llx\n", (unsigned long long)h);
    printf("%s\n", worst < 1e-4 && nan_out == 0 ? "RESULT: PASS" : "RESULT: FAIL");
    return worst < 1e-4 && nan_out == 0 ? 0 : 1;
}
