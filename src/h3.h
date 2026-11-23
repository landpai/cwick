/* h3.h — MiniMax-H3-Image text-to-image and image-editing port.
 *
 * One checkpoint, two OpenAI routes: text prompts in, PNG bytes out. The
 * model is a three-stage pipeline — a Qwen3-VL text conditioner, a 50-layer
 * diffusion transformer sampled in six steps, and a 2D-ViT VAE decoder —
 * and this file is the whole boundary the HTTP layer talks to. It knows
 * nothing about sockets, auth or the chat engine.
 *
 * The weights are read straight out of the shipped safetensors shards via
 * mmap; nothing is copied into a private arena. A generation therefore has
 * the same memory profile whichever route invoked it, and the server can
 * gate concurrency with one mutex around h3_generate().
 *
 * Comments describe mechanism and measured numbers only.
 */
#ifndef CWICK_H3_H
#define CWICK_H3_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "cwick.h"

#define H3_SERVED_ID "MiniMax-H3-Image"
#define H3_STEPS 6
#define H3_LATENT_CHANNELS 24
#define H3_VAE_RATIO 16          /* pixels per latent cell */
#define H3_PATCH 2               /* latent cells per transformer row axis */
#define H3_CANVAS_MULTIPLE 32
/* Default `min_generation_area`. The checkpoint's working range is about 4 to
 * 20 MP; a request below this area is generated on a larger canvas of the same
 * aspect ratio and resized back. Measured at 256x256 with the area floor
 * lowered to 1024: the output is textured noise, and the reference
 * implementation produces the same textured noise. */
#define H3_MIN_AREA 4194304

/* Full-chain timing, all milliseconds. Every phase is wall-clock measured
 * with CLOCK_MONOTONIC so the JSON a client gets back adds up to the same
 * number the access log reports. */
typedef struct {
    double total_ms;
    double tokenize_ms;
    double text_encode_ms;      /* Qwen3-VL conditioner */
    double pack_ms;             /* latent noise + packed layout */
    double dit_ms;              /* all six DiT forwards, summed */
    double vae_ms;              /* post-quant conv + ViT decoder */
    double png_ms;              /* raw RGB -> PNG bytes */
    double load_ms;             /* first shard mmap/page-in, once per process */
    int64_t text_tokens;
    int64_t image_tokens;
    int64_t canvas_w, canvas_h;
    int64_t output_w, output_h;
} h3_timing;

typedef struct h3_model h3_model;

/* Open the three components under `root`:
 *   transformer_diffusers/, text_encoder/, vae_diffusers/, pdd_fused_heads.
 * The model is mmap-backed and safe to keep open for the life of the server.
 * `--backend metal` is honoured through cwick_metal_force() before this call. */
h3_model *h3_model_open(const char *root, cwick_error *err);
void h3_model_close(h3_model *m);
const char *h3_model_dir(const h3_model *m);
bool h3_model_ready(const h3_model *m);

/* Generate one image.
 *
 * prompt  : required, UTF-8, truncated to 4096 tokens by the tokenizer path.
 * w,h     : requested output size in pixels (rounded to H3_CANVAS_MULTIPLE
 *           for generation, then resized back to exactly w,h).
 * seed    : 0 lets the engine pick one; a fixed seed is deterministic on
 *           every machine because the noise draws on the CPU.
 * ref_png : NULL for text-to-image; for edits, an array of PNG byte buffers
 *           (the caller owns them) in prompt order: <Picture 1>, ...
 * ref_len : length of each reference PNG.
 * out     : malloc'd PNG bytes on success (caller frees with h3_png_free).
 * tim     : optional phase breakdown.
 *
 * Returns true on success; on failure `err` carries the reason. */
bool h3_generate(h3_model *m, const char *prompt, int w, int h, uint64_t seed,
                 const unsigned char **ref_png, const size_t *ref_len, int n_ref,
                 unsigned char **out, size_t *out_len, h3_timing *tim,
                 cwick_error *err);

/* Live render progress. A generation runs under the model lock, so at most
 * one is in flight per process; the six phases and the DiT step counter are
 * published through atomics for a polling client. `phase` is 0 idle, 1..6 for
 * tokenize / text_encode / pack / dit / vae / png, and 7 for a finished
 * render (cleared by the next call). `step`/`steps` are meaningful only in
 * phase 4. `stamp_ms` is a monotonic clock value for staleness checks. */
typedef struct {
    int phase;
    int step, steps;
    double stamp_ms;
} h3_progress;
void h3_progress_get(h3_progress *out);

/* PNG encoding for the API response. `rgb` is interleaved float [0,1]. */
unsigned char *h3_png_encode(const float *rgb, int w, int h,
                             size_t *out_len, cwick_error *err);
void h3_png_free(unsigned char *png);

/* PNG decode for /v1/images/edits: RGBA/RGB -> interleaved RGB float [0,1].
 * Returns true and sets w,h; caller frees *rgb with free(). */
bool h3_png_decode(const unsigned char *png, size_t len, float **rgb,
                   int *w, int *h, cwick_error *err);

/* Bounds shared with the HTTP validators. */
#define H3_MIN_SIZE 256
#define H3_MAX_SIZE 6272
#define H3_MAX_REFERENCES 16

/* Collects the `image` parts of a multipart body for /v1/images/edits, in the
 * order they appear, and returns how many were found (at most `max`). The
 * `ptr` entries alias `body` — the bytes are never copied — and the CRLF that
 * precedes the next boundary is trimmed only when it is exactly CRLF, so a
 * binary tail that happens to end in a newline byte is left alone. It lives in
 * the server but is declared here because it is a pure function of the body and
 * the unit tests drive it directly. */
int h3_multipart_refs(const char *body, size_t body_len, const char *boundary,
                      const unsigned char **ptr, size_t *len, int max);

/* Text conditioning without an image model.
 *
 * The video pipeline uses the same H3-Encoder (Qwen3-VL) the image pipeline
 * does, but it must not depend on a MiniMax-H3-Image checkpoint being present:
 * a server started with --videos only has the video root. This opens the
 * encoder and tokenizer under `te_dir` / `tok_dir`, runs the prompt through the
 * final hidden layer, and returns that activation in *hs (seq x 5120, caller
 * frees with h3_text_cond_free). Nothing is cached: the encoder is read once
 * per request, which is the same profile the image path has. */
bool h3_text_cond_dir(const char *te_dir, const char *tok_dir,
                      const char *prompt, float **hs, int64_t *seq,
                      int64_t *tokens, cwick_error *err);
/* Text conditioning with one image reference in the model's presentation:
 * `Picture 1: <|vision_start|> <|image_pad|>... <|vision_end|>` followed by
 * the prompt verbatim. `ref_rgb` is float RGB in [0,1]; the tower resizes it
 * to its own input grid. Besides the hidden states this returns one modality
 * tag per row (1 text, 0 vision) so the video denoiser can address its AdaLN
 * table per row. Caller frees both with h3_text_cond_free(). */
bool h3_text_cond_ref_dir(const char *te_dir, const char *tok_dir,
                          const char *prompt,
                          const float *ref_rgb, int ref_w, int ref_h,
                          float **hs, int64_t *seq, float **tags,
                          cwick_error *err);
void h3_text_cond_free(float *hs);

#endif /* CWICK_H3_H */
