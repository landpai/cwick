/* rt_audio.h — the thinker's ear: Whisper log-mel -> Qwen3-ASR audio tower ->
 * stacked SwiGLU projector. Turns 16 kHz mono float samples into vectors that
 * sit in the prompt in place of the <|audio|> tokens.
 *
 * Frame accounting, all of it derived from the checkpoint's own config:
 *   mel frames   T  = pad16(N) / 160, three stride-2 convs, 100-frame chunks
 *   tower frames    = 13 * (T / 100) + post_cnn(T % 100)
 *   prompt tokens   = ceil(T / 16)  (stack_factor 2, truncated by the caller)
 * A caller that pads the waveform to a multiple of 2560 samples makes every
 * mel frame valid, which is what the reference does before encoding. */
#ifndef CWICK_RT_AUDIO_H
#define CWICK_RT_AUDIO_H

#include <stdint.h>

#include "cwick.h"

typedef struct rt_ear rt_ear;

/* Opens the audio tower and projector that live under `thinker_dir`. The
 * weights are widened to f32 once: the tower is ~200 M parameters and it runs
 * on every turn, so a per-call widen would cost more than the GEMMs. */
rt_ear *rt_ear_open(const char *thinker_dir, cwick_error *err);
void rt_ear_close(rt_ear *e);

/* Number of prompt tokens a clip of `frames` mel frames produces. */
int rt_ear_token_count(const rt_ear *e, int frames);

/* Mel frames a padded clip yields: samples / 160, exactly. */
int rt_ear_frames_for_samples(int samples);

/* Writes the normalised log-mel as [128][frames] (mel-major, the layout the
 * conv frontend and the reference feature extractor both use). `pcm` must
 * already be 16 kHz mono and zero-padded to a multiple of 160 samples;
 * returns the frame count or -1. */
int rt_ear_logmel(const rt_ear *e, const float *pcm, int samples, float *out);

/* Full path: 16 kHz mono -> [tokens][5120] float32, caller frees. */
int rt_ear_embed(rt_ear *e, const float *pcm, int samples, float **out, int *tokens);

int rt_ear_hidden(const rt_ear *e);
int rt_ear_frame_tokens(const rt_ear *e); /* tower width for a mel chunk */

/* Reads a 16-bit PCM WAV (mono or interleaved stereo, mixed down) into
 * [-1,1] floats; returns the sample count or -1. Used by the bring-up tools
 * and by the voice-clip path, not by the streaming socket. */
int rt_wav_load(const char *path, float **pcm, int *rate);

#endif
