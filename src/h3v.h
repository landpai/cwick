/* h3v.h — MiniMax-H3 text-to-video / image-to-video with native stereo audio.
 *
 * One checkpoint, one OpenAI route: POST /v1/videos creates a job, GET
 * /v1/videos/{id} reports it, GET /v1/videos/{id}/content streams the mp4.
 * This header is the whole boundary the HTTP layer talks to; it knows nothing
 * about sockets, auth or job bookkeeping.
 *
 * The weights are mmap-backed and read straight out of the shipped shards.
 * The transformer alone is 62 GiB and is re-read on every sampling step, so
 * the pipeline is opened once for the life of the process and generation is
 * serialised behind one mutex — the machine holds one model at a time.
 *
 * Comments describe mechanism and measured numbers only.
 */
#ifndef CWICK_H3V_H
#define CWICK_H3V_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "cwick.h"

#define H3V_SERVED_ID "MiniMax-H3"
#define H3V_FPS 24
/* The VAE is a causal 3D decoder whose temporal stride produces 17-frame
 * clips: 17 pixels map to 5 latent steps (17n+5 is the frame-count rule the
 * checkpoint trains on). A request is rounded up to whole clips. */
#define H3V_CLIP_FRAMES 17
#define H3V_CLIP_LATENT 5
#define H3V_FRAMES_PER_LATENT 4
#define H3V_LATENT_CHANNELS 24
#define H3V_AUDIO_CHANNELS 32
#define H3V_VAE_RATIO 16         /* pixels per latent cell, both axes */
#define H3V_PATCH 2              /* latent cells per transformer row axis */
#define H3V_CANVAS_MULTIPLE 32
#define H3V_MIN_SECONDS 4
#define H3V_MAX_SECONDS 15
#define H3V_SAMPLE_STEPS 30      /* flow-matching steps at the default budget */
#define H3V_MIN_SIZE 256
/* Generated canvas cap (longest axis), raised to 2K (2560, under 4K) so
 * 2560x1440 requests are admitted. DiT rows grow ~quadratically with the
 * canvas, so larger frames cost proportional wall time; this is a ceiling,
 * not a claim that 30 steps finish quickly. Reference encoding has its own
 * cap below and is never clamped by this one. */
#define H3V_MAX_SIZE 2560
/* Reference encoding is not bound to the generated canvas: the checkpoint
 * presents an image at its own 2048 short edge. Allow the full documented
 * range so CWICK_H3V_REF_EDGE=2048 reaches the encoder. */
#define H3V_REF_EDGE_MAX 6272
#define H3V_AUDIO_RATE 32000     /* the audio VAE decodes to 32 kHz stereo */
#define H3V_MAX_PROMPT_TOKENS 4096

/* Full-chain timing, all milliseconds. Every phase is wall-clock measured
 * with CLOCK_MONOTONIC so the JSON a client gets back adds up to the same
 * number the access log reports. */
typedef struct {
    double total_ms;
    double tokenize_ms;
    double text_encode_ms;   /* H3-Encoder (Qwen3-VL) conditioner */
    double encode_ms;        /* reference image -> visual VAE latent (i2v only) */
    double pack_ms;          /* latent noise + 3D packed layout */
    double dit_ms;           /* all sampling steps, summed */
    double vae_ms;           /* video VAE decode */
    double audio_vae_ms;     /* audio VAE decode */
    double mux_ms;           /* H.264 + AAC + mp4 mux */
    double load_ms;          /* first shard mmap/page-in, once per process */
    int64_t text_tokens;
    int64_t video_tokens;
    int64_t audio_tokens;
    int64_t frames;
    int64_t width, height;
    int64_t steps;
    int64_t latent_t, latent_h, latent_w;
    int64_t audio_samples;
    int cached;              /* dit phase skipped: latent restored from disk */

    /* Denoise quality signal. A clip can mux cleanly while every generated
     * frame is still sampling noise, which a byte count or an HTTP 200 cannot
     * tell apart. `spatial_corr` is the mean lag-1 horizontal autocorrelation
     * of the first *generated* latent step (the conditioning step is excluded):
     * a decoded natural latent is smooth (~0.8+), Gaussian noise is ~0. This
     * directly catches the "static noise" failure. `snr_db` is 20*log10 of the
     * generated step's std over the clean conditioning step's std (i2v only;
     * 0 for text-only jobs). Both are computed on the CPU from the latent the
     * pipeline already holds, so they add no extra model reads. `suspect_noise`
     * is 1 when spatial_corr falls below H3V_CORR_FLOOR; the server surfaces it
     * as a metric and a job field. */
    double spatial_corr;
    double snr_db;
    int suspect_noise;
    /* `clean_corr` is the same lag-1 correlation measured on the clean
     * conditioning step (i2v only). It is the natural-latent anchor used to
     * calibrate the noise floor: a converged generated step should approach
     * this value, and a still-noisy one sits near zero. 0 for text-only jobs. */
    double clean_corr;
} h3v_timing;

typedef struct h3v_model h3v_model;

/* Phase progress callback. `phase` is one of the H3V_PHASE_* names below,
 * `frac` runs 0..1 within that phase, and `steps_done/steps_total` is the
 * sampling step counter (0/0 outside the DiT phase). The server turns this
 * into the `progress` field and one stdout line per phase transition. */
typedef void (*h3v_progress_fn)(void *ctx, const char *phase,
                                int steps_done, int steps_total, double frac);

#define H3V_PHASE_TEXT    "text"
#define H3V_PHASE_PACK    "pack"
#define H3V_PHASE_DIT     "dit"
#define H3V_PHASE_VIDEO   "video_vae"
#define H3V_PHASE_AUDIO   "audio_vae"
#define H3V_PHASE_MUX     "mux"

/* Open the pipeline under `root`. Two layouts are accepted:
 *   root/Ref2VA/... (the omni-reference video+audio transformer)
 *   root/...        (the files directly, for a flattened download)
 * The first/last-frame partition is not part of the video endpoint and is
 * never probed. Each layout needs text_encoder/, tokenizer/, transformer/,
 * video_vae/ and audio_vae/. `--backend metal` is honoured through
 * cwick_metal_force(). */
h3v_model *h3v_model_open(const char *root, cwick_error *err);
void h3v_model_close(h3v_model *m);
const char *h3v_model_dir(const h3v_model *m);
bool h3v_model_ready(const h3v_model *m);
/* "Ref2VA" (or "FLAT" for a flattened checkpoint) — the video variant. */
const char *h3v_model_variant(const h3v_model *m);
/* Number of selectable transformer variants found under the model root, and
 * the name of variant `i` (0..count-1). The video endpoint exposes only the
 * Ref2VA partition, or "FLAT" for a flattened copy. */
int h3v_model_variant_count(const h3v_model *m);
const char *h3v_model_variant_name(const h3v_model *m, int i);
/* Resolve a caller-supplied name (case-insensitive) to the canonical variant
 * directory name. NULL means "no such variant"; NULL/empty want returns the
 * default. Used by the HTTP layer to validate the request before queuing. */
const char *h3v_model_resolve_variant(const h3v_model *m, const char *want);

/* Generate one clip.
 *
 * prompt : required, UTF-8, truncated to H3V_MAX_PROMPT_TOKENS.
 * w,h    : requested output size (rounded to H3V_CANVAS_MULTIPLE).
 * seconds: requested duration, clamped to [H3V_MIN_SECONDS, H3V_MAX_SECONDS]
 *          and rounded up to whole 17-frame clips at 24 fps.
 * seed   : 0 lets the engine pick one; a fixed seed is deterministic because
 *          the noise draws on the CPU.
 * cb     : optional progress sink (may be NULL).
 * out    : malloc'd mp4 bytes on success (caller frees with h3v_mp4_free).
 * tim    : optional phase breakdown.
 *
 * Returns true on success; on failure `err` carries the reason. */
bool h3v_generate(h3v_model *m, const char *prompt, int w, int h, int seconds,
                  uint64_t seed, h3v_progress_fn cb, void *cbctx,
                  unsigned char **out, size_t *out_len, h3v_timing *tim,
                  cwick_error *err);

/* First-frame-to-video variant. `ref_rgb` is a width*height*3 RGB24 picture;
 * it is resized to the render canvas, encoded with the visual VAE, and held
 * as the clean t=0 latent through every sampling step. Pass NULL for plain
 * text-to-video (identical to h3v_generate). */
bool h3v_generate_i2v(h3v_model *m, const char *prompt,
                      const unsigned char *ref_rgb, int ref_w, int ref_h,
                      int w, int h, int seconds, uint64_t seed,
                      h3v_progress_fn cb, void *cbctx,
                      unsigned char **out, size_t *out_len, h3v_timing *tim,
                      cwick_error *err);

/* Same as h3v_generate_i2v but pins the transformer variant for this job.
 * `variant` is case-insensitive ("FL2VA"/"Ref2VA"); use h3v_model_resolve_variant
 * first to validate. NULL means the model's default variant. A NULL ref_rgb
 * gives text-to-video. The VAE and text encoder are shared and byte-identical
 * between variants, so only the DiT shards are swapped. */
bool h3v_generate_variant(h3v_model *m, const char *variant, const char *prompt,
                          const unsigned char *ref_rgb, int ref_w, int ref_h,
                          int w, int h, int seconds, uint64_t seed,
                          h3v_progress_fn cb, void *cbctx,
                          unsigned char **out, size_t *out_len,
                          h3v_timing *tim, cwick_error *err);

/* Same as h3v_generate_variant but pins the flow-matching step count for this
 * call. `steps` is clamped to [1, H3V_SAMPLE_STEPS]; pass 0 to use the process
 * default (CWICK_H3V_STEPS, otherwise H3V_SAMPLE_STEPS). Small values (2-4)
 * trade sample quality for a minutes-long smoke run on one clip. */
bool h3v_generate_variant_steps(h3v_model *m, const char *variant,
                                const char *prompt,
                                const unsigned char *ref_rgb, int ref_w, int ref_h,
                                int w, int h, int seconds, uint64_t seed, int steps,
                                h3v_progress_fn cb, void *cbctx,
                                unsigned char **out, size_t *out_len,
                                h3v_timing *tim, cwick_error *err);

void h3v_mp4_free(unsigned char *p);

/* Frame count for a requested duration, already rounded up to whole clips.
 * Shared with the HTTP validator so the `seconds` the caller sees in the job
 * object matches the clip that was actually rendered. */
int64_t h3v_frames_for_seconds(int seconds);

#endif /* CWICK_H3V_H */
