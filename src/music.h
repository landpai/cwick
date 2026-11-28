#ifndef CWICK_MUSIC_H
#define CWICK_MUSIC_H

#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "cwick.h"

#define MUSIC_SERVED_ID "MiniMax-Music3"
#define MUSIC_SAMPLE_RATE 32000
#define MUSIC_CHANNELS 2
#define MUSIC_MAX_PROMPT_TOKENS 5000
#define MUSIC_MAX_FRAMES 9000
#define MUSIC_DEFAULT_STEPS 30
#define MUSIC_AR_MIN_PERCENT 25
#define MUSIC_LYRIC_MAX_RATE 6.0
#define MUSIC_MIN_SECONDS 1
#define MUSIC_MAX_SECONDS 360

typedef enum {
    MUSIC_TASK_GENERATE = 0,
    MUSIC_TASK_COVER = 1,
} music_task;

typedef struct {
    int64_t cjk_chars;
    int64_t latin_words;
    int64_t units;
    double units_per_second;
} music_lyric_stats;

void music_lyric_stats_measure(const char *lyrics, int seconds, music_lyric_stats *out);

typedef struct {
    double total_ms;
    double load_ms;
    double tokenize_ms;
    double reference_encode_ms;
    double global_lm_ms;
    double depth_ms;
    double condition_ms;
    double flow_ms;
    double vocoder_ms;
    double wav_ms;
    int64_t prompt_tokens;
    int64_t frames;
    int64_t requested_frames;
    int ended_early;
    int ar_attempts;
    int64_t flow_steps;
    int64_t audio_samples;
    int64_t output_bytes;
    int cover;
} music_timing;

typedef struct music_model music_model;

typedef void (*music_progress_fn)(void *ctx, const char *phase,
                                  int64_t done, int64_t total, double frac);

#define MUSIC_PHASE_LOAD      "load"
#define MUSIC_PHASE_TOKENIZE  "tokenize"
#define MUSIC_PHASE_REFERENCE "reference_encode"
#define MUSIC_PHASE_GLOBAL    "global_lm"
#define MUSIC_PHASE_DEPTH     "depth"
#define MUSIC_PHASE_CONDITION "condition"
#define MUSIC_PHASE_FLOW      "flow"
#define MUSIC_PHASE_VOCODER   "vocoder"
#define MUSIC_PHASE_WAV       "wav"

typedef struct {
    music_task task;
    const char *prompt;
    const char *lyrics;
    const unsigned char *reference_audio;
    size_t reference_len;
    int seconds;
    uint64_t seed;
    int steps;
    float cfg_scale;
    float temperature;
    int top_k;
    float audio_weight;
    music_progress_fn progress;
    void *progress_ctx;
    const _Atomic int *cancel;
} music_request;

music_model *music_model_open(const char *root, cwick_error *err);
void music_model_close(music_model *m);
bool music_model_ready(const music_model *m);
const char *music_model_dir(const music_model *m);

/* Produces a complete 32 kHz, 16-bit stereo WAV. Cover mode encodes the
 * reference with the checkpoint's Flow-VAE encoder and uses that latent as the
 * flow initialization/overlap condition; it is not a waveform mix. */
bool music_generate(music_model *m, const music_request *req,
                    unsigned char **out_wav, size_t *out_len,
                    music_timing *tim, cwick_error *err);

void music_wav_free(unsigned char *p);
bool music_conv_selftest(void);
bool music_latent_frames_selftest(void);
bool music_audio_valid_selftest(void);
bool music_flow_layout_selftest(void);
bool music_flow_scheduler_selftest(void);
bool music_ar_stop_selftest(void);
bool music_short_window_selftest(void);
bool music_audio_head_selftest(void);
bool music_qkv_group_selftest(void);
bool music_lyric_stats_selftest(void);
/* Lyric tags are reduced to the checkpoint's documented section set; see the
 * comment on music_canonical_tag in music.c. */
bool music_lyrics_selftest(void);
bool music_lyrics_are_instrumental(const char *lyrics);
bool music_has_sung_text(const char *lyrics);

#endif
