/* rt_talker.h — the voice: text in, 24 kHz mono PCM out.
 *
 * Three stages run in sequence, and the middle one is streamed:
 *   text -> Qwen2-0.5B LM -> speech tokens (25 per second of audio)
 *        -> distilled flow -> mel (50 frames per second)
 *        -> causal HiFT   -> PCM (480 samples per mel frame)
 *
 * A voice prompt (a few seconds of reference audio, already tokenised and
 * reduced to a mel and a speaker vector by the checkpoint) conditions all
 * three stages, which is what makes the output sound like that speaker.
 *
 * Synthesis is chunked: every RT_TTS_CHUNK speech tokens (default 25, one
 * second) the flow renders the new region with three tokens of lookahead and
 * the vocoder turns it into audio, so the first sound arrives long before the
 * sentence is finished. */
#ifndef CWICK_RT_TALKER_H
#define CWICK_RT_TALKER_H

#include <stdint.h>

#include "cwick.h"

typedef struct rt_talker rt_talker;

/* A prepared voice prompt, as shipped in voices/<name>.pt. */
typedef struct {
    int *token;      /* speech tokens, 25 fps */
    int n_token;
    float *feat;     /* [n_feat][80] log-mel, 50 fps */
    int n_feat;
    float embedding[192]; /* speaker vector */
} rt_voice;

/* `dir` is the talker directory: llm.pt, flow.pt, hift.pt and the BlankEN
 * tokenizer live under it. */
rt_talker *rt_talker_open(const char *dir, cwick_error *err);
void rt_talker_close(rt_talker *t);

int rt_voice_load(const char *path, rt_voice *v, cwick_error *err);
void rt_voice_free(rt_voice *v);

/* Per-stage wall clock, for the realtime metrics and the stdout progress log. */
typedef struct {
    double load_s;
    double lm_prefill_s;
    double lm_decode_s;
    double flow_s;
    double hift_s;
    double first_audio_s; /* from the start of say() to the first PCM chunk */
    double total_s;
    int lm_prompt_tokens;
    int speech_tokens;
    int mel_frames;
    int samples;
    int chunks;
    /* Sampler health: a confident LM keeps a short nucleus and puts most of
     * the mass on one id. Both collapse toward uniform when the stack is
     * wrong, so they are the cheapest correctness signal available offline. */
    double lm_top1_prob;
    double lm_nucleus;
} rt_talker_stats;

/* Called with each PCM chunk. `done` is 1 on the final call, which may carry
 * no samples. Returning non-zero aborts synthesis (barge-in). */
typedef int (*rt_talker_pcm_cb)(void *user, const float *pcm, int n, int done);

/* Synthesize `text`. `stats` may be NULL. Returns 0 on success, -1 on error,
 * 1 when the callback aborted. */
int rt_talker_say(rt_talker *t, const rt_voice *v, const char *text,
                  rt_talker_pcm_cb cb, void *user, rt_talker_stats *stats);

/* The prompt transcript the voice was recorded with. The shipped prompts carry
 * no transcript, so the default is just the prompt marker; RT_TTS_PROMPT_TEXT
 * overrides it. */
const char *rt_talker_prompt_text(void);

int rt_talker_sample_rate(const rt_talker *t);

#endif /* CWICK_RT_TALKER_H */