/* h3v_internal.h — pieces the video pipeline's three files share.
 *
 * h3v.c owns the model handle and the transformer; h3v_vae.c reads the visual
 * VAE shards; h3v_audio.c reads the audio VAE shards; h3v_mp4.m turns pixels
 * and samples into an mp4. They are split because the weight bytes are the
 * only thing they have in common — the visual decoder is 36 transformer blocks
 * of F32, the audio decoder is a BigVGAN, and neither shape appears in the
 * other.
 *
 * Nothing here is part of the public boundary; the HTTP layer only ever sees
 * h3v.h.
 */
#ifndef CWICK_H3V_INTERNAL_H
#define CWICK_H3V_INTERNAL_H

#include <stdbool.h>
#include <stdint.h>

#include "cwick.h"
#include "h3v.h"

/* Shard handles, owned by the h3v_model. Both stay open for the life of the
 * process: the visual VAE is 10.4 GiB and re-mapping it per request would cost
 * more than the decode. */
const cwick_safetensors *h3v_shards_vae(const h3v_model *m);
const cwick_safetensors *h3v_shards_avae(const h3v_model *m);

/* Reported through h3v_timing from the pipeline. */
/* The released decoder config uses 256 px tiles with 64 px minimum overlap.
 * In latent cells that is 16/4. Keeping every intermediate tile at the full
 * size matters: a short tail tile changes the length-normalised RoPE grid and
 * leaves a visible phase seam even when the latent itself is smooth. */
#define H3V_VAE_TILE_CELLS   16   /* latent cells per tile edge            */
#define H3V_VAE_TILE_OVERLAP 4    /* latent cells shared between neighbours */
#define H3V_VAE_LATENT_CH    24
#define H3V_VAE_PIXEL        16   /* pixels per latent cell                */
#define H3V_VAE_FRAMES       4    /* frames per latent step                */

/* Visual VAE decode.
 *
 * lat  : [H3V_VAE_LATENT_CH][lt][lh][lw] in the transformer's normalised
 *        latent space (the same space the sampler works in). The function
 *        applies `x*std+mean` per channel before the decoder's embedding, so
 *        callers never convert by hand.
 * out  : malloc'd RGB24, width = lw*16, height = lh*16, row major, three
 *        bytes per pixel. Caller frees. For lt > 5 the decoder walks the
 *        clip in 5-step chunks with a 2-step overlap and returns 17*chunks+5
 *        frames, the first 17*chunks of which are the cross-faded clip; the
 *        caller keeps those. For lt <= 5 it returns lt*4 whole-clip frames.
 *        CWICK_H3V_TEMPORAL_XFADE=0 restores the whole-clip decode for A/B.
 * cb   : progress sink for the tile loop (may be NULL). */
unsigned char *h3v_vae_decode(const h3v_model *m, const float *lat, int64_t lt,
                              int64_t lh, int64_t lw, h3v_progress_fn cb,
                              void *cbctx, cwick_error *err);

/* Visual VAE encode: one RGB24 frame (width/height multiple of 16) -> the
 * mean latent [1][24][lh][lw], lh=height/16, lw=width/16, already normalised
 * to the transformer's space so it can be written straight into latent step 0.
 * Caller frees the result. */
float *h3v_vae_encode(const h3v_model *m, const unsigned char *rgb, int width,
                      int height, cwick_error *err);

/* Audio VAE decode.
 *
 * z    : [32][t] de-normalised audio latent, t = seconds * 40.
 * out  : malloc'd interleaved stereo float32, *samples frames of two channels.
 *        Caller frees. */
float *h3v_audio_decode(const h3v_model *m, const float *z, int64_t t,
                        int64_t *samples, h3v_progress_fn cb, void *cbctx,
                        cwick_error *err);

/* Pure-C layout regression check (no weights). Fills a synthetic [C][T][H][W]
 * latent with a distinct smooth plane at t=0 and white noise elsewhere, then
 * verifies the denoise signal reads that plane, not a neighbouring channel or
 * time step. Also checks 3D pack/unpack round-trips. Returns true on success. */
bool h3v_layout_selftest(void);
bool h3v_vae_plan_selftest(void);
bool h3v_audio_latnorm_selftest(void);
bool h3v_audio_amp_selftest(void);

/* Resolve the reference image's VAE-encode raster from its natural size and
 * the requested short edge. `edge` is CWICK_H3V_REF_EDGE; 0 (or a value out of
 * [32, H3V_REF_EDGE_MAX]) leaves `*w,*h` untouched, which the caller fills
 * with the generated canvas. A valid edge scales the short side, keeps the
 * aspect ratio and rounds both axes to H3V_CANVAS_MULTIPLE. Extracted so the
 * unit test exercises the exact production arithmetic for 2048. */
void h3v_ref_encode_size(int src_w, int src_h, int edge, int *w, int *h);

/* Pure-C variant-name regression check (no weights). Verifies case-insensitive
 * matching of the shipped variant spellings and rejection of unknown names. */
bool h3v_variant_selftest(void);

/* Pure-C latent-scale regression check (no weights). Verifies the 24-channel
 * normalise/denormalise pair round-trips and that a zero latent maps to the
 * checkpoint's per-channel mean. Returns true on success. */
bool h3v_latnorm_selftest(void);

/* ---- mp4 muxing (h3v_mp4.m) ---------------------------------------- */

/* Encode `frames` RGB24 pictures at H3V_FPS plus `samples` interleaved stereo
 * float32 at H3V_AUDIO_RATE into one mp4. Returns malloc'd bytes.
 * Either stream may be empty (frames == 0 or samples == 0) but not both. */
unsigned char *h3v_mp4_encode(const unsigned char *rgb, int64_t frames, int width,
                              int height, const float *audio, int64_t samples,
                              size_t *out_len, cwick_error *err);

#endif /* CWICK_H3V_INTERNAL_H */
