/*
 * ds_verify.c — exercise the DeepSeek numeric core against real checkpoint
 * bytes and print everything needed to check it from outside.
 *
 * The point is that the checking side shares no code with this one: the C
 * program prints the bytes it used plus its results, and tools/ds_verify.py
 * recomputes both from the shard headers. Anything that disagrees is a bug in
 * one of the two, and a silent one at that — a wrong dequantization still
 * produces plausible-looking floats.
 *
 *   ds_verify <checkpoint-root>
 */
#include "cwick.h"
#include "ds.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static uint64_t fnv1a(const void *p, size_t n) {
    const uint8_t *b = p;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/* deterministic activation: same sequence on both sides */
static uint32_t lcg_state = 12345u;
static float next_act(void) {
    lcg_state = lcg_state * 1103515245u + 12345u;
    return (float)((lcg_state >> 8) & 0xFFFF) / 32768.0f - 1.0f;
}

static void dump_tables(void) {
    for (int b = 0; b < 256; b++) {
        float v = ds_e4m3_to_f32((uint8_t)b);
        printf("E4M3 %02x %a\n", b, (double)v);
    }
    for (int n = 0; n < 16; n++)
        printf("FP4 %x %a\n", n, (double)ds_fp4_to_f32((uint8_t)n));
    for (int b = 0; b < 256; b++)
        printf("E8M0 %02x %a\n", b, (double)ds_e8m0_to_f32((uint8_t)b));
}

/* Round-trip property: dequant(quant(v)) must be the nearest e4m3 to v, so the
 * error is at most half an ulp of the e4m3 grid. Checks the encoder against
 * the decoder without trusting either. */
static void check_roundtrip(void) {
    int bad = 0;
    for (int i = 0; i < 200000; i++) {
        float v = next_act() * 600.0f;               /* spans past 448 */
        float c = v > 448.0f ? 448.0f : (v < -448.0f ? -448.0f : v);
        uint8_t q = ds_f32_to_e4m3(c);
        float back = ds_e4m3_to_f32(q);
        if (isnan(back)) { bad++; continue; }
        /* distance to the two neighbouring representable values */
        float lo = back, hi = back;
        for (int d = 1; d < 256 && (lo == back || hi == back); d++) {
            if (q >= d && lo == back) lo = ds_e4m3_to_f32((uint8_t)(q - d));
            if (q + d < 256 && hi == back) hi = ds_e4m3_to_f32((uint8_t)(q + d));
        }
        float best = fabsf(c - back);
        float alt = fabsf(c - lo) < fabsf(c - hi) ? fabsf(c - lo) : fabsf(c - hi);
        if (alt < best - 1e-12f) {
            printf("ROUNDTRIP-FAIL v=%a got=%a(%02x) closer=%a\n",
                   (double)c, (double)back, q, (double)(alt < best ? lo : hi));
            if (++bad > 8) return;
        }
    }
    printf("ROUNDTRIP %s (%d failures)\n", bad ? "FAIL" : "ok", bad);
}

static void check_actquant(void) {
    /* one vector of 64 floats: two blocks, so per-block scales are exercised */
    float x[64];
    for (int i = 0; i < 64; i++) x[i] = next_act();
    uint8_t q[64];
    uint8_t s0 = ds_act_quant_block(x, q);
    uint8_t s1 = ds_act_quant_block(x + 32, q + 32);
    printf("ACTQ vec");
    for (int i = 0; i < 64; i++) printf(" %a", (double)x[i]);
    printf("\n");
    printf("ACTQ scale %02x %02x\n", s0, s1);
    printf("ACTQ bytes");
    for (int i = 0; i < 64; i++) printf(" %02x", q[i]);
    printf("\n");
}

/* The checkpoint pairs `X.weight` with `X.scale` (not `X.weight.scale`). */
static void scale_name_of(const char *name, char *out, size_t cap) {
    size_t n = strlen(name);
    if (n > 7 && strcmp(name + n - 7, ".weight") == 0)
        snprintf(out, cap, "%.*s.scale", (int)(n - 7), name);
    else
        snprintf(out, cap, "%s.scale", name);
}

static void check_tensor(cwick_safetensors *saf, const char *name) {
    cwick_st_tensor t;
    cwick_error err;
    cwick_error_clear(&err);
    if (cwick_safetensors_get(saf, name, &t, &err) != CWICK_OK) {
        printf("MISSING %s (%s)\n", name, cwick_error_message(&err));
        return;
    }
    printf("TENSOR %s rows=%lld cols=%lld dtype=%d bytes=%llu fnv=%016llx\n",
           name, (long long)t.rows, (long long)t.cols, (int)t.dtype,
           (unsigned long long)t.data_bytes,
           (unsigned long long)fnv1a(t.data, t.data_bytes > (1u << 20) ? (1u << 20) : t.data_bytes));

    /* the engram tables are 91 GiB each: report their shape and checksum, but
     * never try to walk them */
    if (t.rows > 4096 || t.cols > 8192) {
        printf("SKIP %s (too large to walk)\n", name);
        return;
    }

    if (t.dtype == CWICK_ST_F8_E4M3) {
        /* weight + its .scale, quantized activation, one fp8 gemv */
        char sname[256];
        scale_name_of(name, sname, sizeof(sname));
        cwick_st_tensor st;
        if (cwick_safetensors_get(saf, sname, &st, &err) != CWICK_OK) {
            printf("MISSING %s\n", sname);
            return;
        }
        int64_t K = t.cols, N = t.rows;
        float *x = malloc(sizeof(float) * (size_t)K);
        uint8_t *a = malloc((size_t)K);
        uint8_t *as = malloc((size_t)(K / 32));
        float *out = malloc(sizeof(float) * (size_t)N);
        float *ref = malloc(sizeof(float) * (size_t)N);
        for (int64_t k = 0; k < K; k++) x[k] = next_act();
        for (int64_t b = 0; b < K / 32; b++)
            as[b] = ds_act_quant_block(x + b * 32, a + b * 32);
        ds_fp8_gemv(a, as, t.data, st.data, N, K, out);
        /* reference: dequantize the whole weight, then dot */
        for (int64_t n = 0; n < N; n++) {
            double acc = 0;
            for (int64_t b = 0; b < K / 32; b++) {
                double sb = ds_e8m0_to_f32(((const uint8_t *)st.data)[(n / 32) * (K / 32) + b]);
                double sa = ds_e8m0_to_f32(as[b]);
                double part = 0;
                for (int j = 0; j < 32; j++)
                    part += (double)ds_e4m3_to_f32(((const uint8_t *)t.data)[n * K + b * 32 + j]) *
                            (double)ds_e4m3_to_f32(a[b * 32 + j]);
                acc += part * sa * sb;
            }
            ref[n] = (float)acc;
        }
        printf("ACTQIN");
        for (int64_t k = 0; k < K; k++) printf(" %a", (double)x[k]);
        printf("\n");
        for (int64_t n = 0; n < N; n++)
            printf("FP8GEMV %s %lld %a %a\n", name, (long long)n,
                   (double)out[n], (double)ref[n]);
        free(x); free(a); free(as); free(out); free(ref);
    } else if (t.dtype == CWICK_ST_I8) {
        /* packed FP4 expert row */
        char sname[256];
        scale_name_of(name, sname, sizeof(sname));
        cwick_st_tensor st;
        if (cwick_safetensors_get(saf, sname, &st, &err) != CWICK_OK) {
            printf("MISSING %s\n", sname);
            return;
        }
        int64_t K = t.cols * 2, N = t.rows;
        float *x = malloc(sizeof(float) * (size_t)K);
        uint8_t *a = malloc((size_t)K);
        uint8_t *as = malloc((size_t)(K / 32));
        float *out = malloc(sizeof(float) * (size_t)N);
        float *ref = malloc(sizeof(float) * (size_t)N);
        for (int64_t k = 0; k < K; k++) x[k] = next_act();
        for (int64_t b = 0; b < K / 32; b++)
            as[b] = ds_act_quant_block(x + b * 32, a + b * 32);
        ds_fp4_gemv(a, as, t.data, st.data, N, K, out);
        for (int64_t n = 0; n < N; n++) {
            double acc = 0;
            for (int64_t b = 0; b < K / 32; b++) {
                double sb = ds_e8m0_to_f32(((const uint8_t *)st.data)[n * (K / 32) + b]);
                double sa = ds_e8m0_to_f32(as[b]);
                double part = 0;
                for (int j = 0; j < 16; j++) {
                    uint8_t byte = ((const uint8_t *)t.data)[n * (K / 2) + b * 16 + j];
                    part += (double)ds_fp4_to_f32(byte & 0x0F) * (double)ds_e4m3_to_f32(a[b * 32 + 2 * j]);
                    part += (double)ds_fp4_to_f32(byte >> 4) * (double)ds_e4m3_to_f32(a[b * 32 + 2 * j + 1]);
                }
                acc += part * sa * sb;
            }
            ref[n] = (float)acc;
        }
        printf("ACTQIN");
        for (int64_t k = 0; k < K; k++) printf(" %a", (double)x[k]);
        printf("\n");
        for (int64_t n = 0; n < N; n++)
            printf("FP4GEMV %s %lld %a %a\n", name, (long long)n,
                   (double)out[n], (double)ref[n]);
        free(x); free(a); free(as); free(out); free(ref);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <checkpoint-root>\n", argv[0]);
        return 2;
    }
    dump_tables();
    check_roundtrip();
    check_actquant();

    cwick_safetensors saf;
    cwick_error err;
    cwick_error_clear(&err);
    if (cwick_safetensors_open(argv[1], &saf, &err) != CWICK_OK) {
        printf("OPEN-FAIL %s\n", cwick_error_message(&err));
        return 1;
    }
    printf("SHARDS %lld bytes %llu\n", (long long)cwick_safetensors_count_shards(&saf),
           (unsigned long long)cwick_safetensors_total_bytes(&saf));

    /* one FP8 attention projection and one packed FP4 expert */
    check_tensor(&saf, "layers.0.attn.wkv.weight");
    check_tensor(&saf, "layers.0.attn.wq_a.weight");
    check_tensor(&saf, "layers.0.ffn.experts.0.w1.weight");
    check_tensor(&saf, "layers.0.ffn.experts.7.w2.weight");
    check_tensor(&saf, "layers.1.engram.embed.weight");
    cwick_safetensors_close(&saf);
    return 0;
}
