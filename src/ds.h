/*
 * ds.h — DeepSeek-V4.1-Flash-1017 port.
 *
 * Everything in this header is specific to that checkpoint and has nothing to
 * do with the GLM path in cwick.h; the two models share only the process.
 *
 * Storage recap (measured from the shard headers, not assumed):
 *   experts        FP4 e2m1 packed two-per-byte along K, one E8M0 scale per
 *                  32 K elements, per row            [N, K/2] I8 + [N, K/32] E8M0
 *   attention etc. FP8 e4m3 with one E8M0 scale per 32x32 block
 *                                                     [N, K] F8_E4M3 + [N/32, K/32] E8M0
 *   engram rows    FP8 e4m3 with one E8M0 scale per 32 elements of the row
 *   rest           BF16 / F32
 */
#ifndef CWICK_DS_H
#define CWICK_DS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* generated tables (src/ds_tables.c, produced by tools/ds_constants.py) */
/* ------------------------------------------------------------------ */
extern const int32_t ds_token_map[129280];      /* token id -> compressed id */
extern const int64_t ds_engram_primes[2][3][8]; /* [layer][ngram-2][head] */
extern const int64_t ds_engram_offsets[2][24];  /* [layer][column] */
extern const int64_t ds_engram_mult[2][4];      /* [layer][lookback] */
extern const int32_t ds_engram_pad_id;
extern const int32_t ds_compressed_vocab_size;
extern const int32_t ds_engram_n_layers;

/* ------------------------------------------------------------------ */
/* scalar format conversions                                           */
/* ------------------------------------------------------------------ */

/* Flat conversion tables, filled on first use. The GEMV inner loops index
 * these directly; ds_fp4_to_f32 / ds_e4m3_to_f32 are the scalar entry points. */
extern float DS_FP4_LUT[16];
extern float DS_E4M3_LUT[256];

float ds_fp4_to_f32(uint8_t nibble);  /* e2m1: 0,.5,1,1.5,2,3,4,6 / negatives */
float ds_e8m0_to_f32(uint8_t b);      /* 2^(b-127); 0xFF is NaN */
float ds_e4m3_to_f32(uint8_t b);      /* e4m3fn: max 448, 0x7F/0xFF NaN */
float ds_bf16_to_f32(uint16_t b);     /* bfloat16 */
uint8_t ds_f32_to_e4m3(float v);      /* round-to-nearest-even, assumes |v|<=448 */
uint8_t ds_f32_to_e8m0_exponent(float amax); /* ceil(log2(amax/448)) + 127 */

/* Quantize one 32-element block the way the reference does: per-block power
 * of two scale, clamp to +-448, e4m3. Returns the scale byte (e8m0). */
uint8_t ds_act_quant_block(const float *x, uint8_t *out_fp8);

/* ------------------------------------------------------------------ */
/* GEMMs over quantized weights                                        */
/*                                                                     */
/* Every one of these takes an already-quantized activation plus its      */
/* per-32 scales, so the caller controls when the lossy step happens.    */
/* ------------------------------------------------------------------ */

/* FP4 experts: w is [N, K/2] packed, ws is [N, K/32] e8m0.
 * out[n] = sum over 32-blocks of sa[kb]*ws[n][kb] * sum_k w[n][k]*a[k] */
void ds_fp4_gemv(const uint8_t *a, const uint8_t *a_scale,
                 const uint8_t *w, const uint8_t *w_scale,
                 int64_t N, int64_t K, float *out);

/* FP8 weights with 32x32 block scales: w is [N, K], ws is [N/32, K/32]. */
void ds_fp8_gemv(const uint8_t *a, const uint8_t *a_scale,
                 const uint8_t *w, const uint8_t *w_scale,
                 int64_t N, int64_t K, float *out);

/* The same product with the fp8 weights widened to bf16 once at load.
 *
 * The dense weights (attention projections, shared experts, indexer) are read
 * in full on every token -- about 5.4 GB per step -- and an e4m3 value fits in
 * bf16 exactly (3 mantissa bits against 8), so the widening is lossless. What
 * it buys is the inner loop: bf16 decodes with a shift instead of a table
 * lookup, which is worth roughly 3x the bytes per second. `w_scale` stays the
 * 32x32 e8m0 block scale, applied to the block sum exactly as before, so the
 * arithmetic keeps the reference's shape. */
void ds_qbf16_gemv(const uint16_t *w16, const uint8_t *w_scale,
                   const uint8_t *a, const uint8_t *a_scale,
                   int64_t N, int64_t K, float *out);

/* Widen `n` fp8 values to bf16. Exact; used once per tensor at load. */
void ds_fp8_to_bf16_row(const uint8_t *src, uint16_t *dst, int64_t n);

/* Plain BF16 GEMM (norm weights, gate, compressor, indexer keys). */
void ds_bf16_gemv(const uint16_t *w, const float *x, const float *bias,
                  int64_t N, int64_t K, float *out);

/* Plain F32 GEMM (hyper-connection projections, fp32 experts). */
void ds_f32_gemv(const float *w, const float *x, const float *bias,
                 int64_t N, int64_t K, float *out);

/* Dequantize a whole FP4 row (used by the verification tool). */
void ds_fp4_row(const uint8_t *packed, const uint8_t *scale, int64_t K, float *out);

#ifdef __cplusplus
}
#endif

#endif /* CWICK_DS_H */
