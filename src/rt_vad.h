#ifndef CWICK_RT_VAD_H
#define CWICK_RT_VAD_H

#include <stdbool.h>

/* Turn detection for the realtime input stream.
 *
 * The detector sees 16 kHz mono and works on fixed 10 ms frames. It is kept
 * free of server/audio dependencies so the same code can be driven by the
 * production session and by a standalone signal harness for A/B.
 *
 * Two decision rules are layered on the plain energy threshold:
 *   - release hysteresis: onset and offset use different levels, so a short
 *     dip inside a word cannot accumulate toward the end-of-turn timer;
 *   - a short-fragment floor: an utterance whose active time never reaches
 *     the floor is discarded instead of being committed as a turn.
 * Both are optional at run time; `legacy` reproduces the original single
 * threshold byte for byte, which is what makes a controlled A/B possible.
 */
typedef struct {
    int in_ms;              /* ms since the last commit, advanced per frame */
    bool speech;            /* a confirmed utterance is in progress */
    int silence_ms;         /* consecutive quiet frames at the release level */
    int active_ms;          /* above-onset time accumulated in this utterance */
    int speech_start_ms;    /* reported onset, including prefix padding */
    bool pending;           /* above onset but not yet past the fragment floor */
    int pending_ms;         /* in_ms at the start of the pending burst */
} rt_vad_state;

#define RT_VAD_HYST     0.15f   /* release offset on the 0..1 threshold scale */
#define RT_VAD_MIN_ACTIVE_MS 100

void rt_vad_state_init(rt_vad_state *v);
void rt_vad_state_reset(rt_vad_state *v);

/* Feed n samples at `rate`. Returns 0. Exactly one of the out flags may be set
 * per call, except that a discarded fragment sets neither `stopped` nor
 * `start_ms` and reports through `discarded`.
 *
 * threshold    0..1, mapped to dBFS the same way the client-facing spec is
 * prefix_ms    pre-roll reported in audio_start_ms
 * silence_ms   quiet time needed to end a confirmed utterance
 * hysteresis   enable the two-level onset/release decision
 * min_active_ms  active time below which an utterance is dropped as noise
 */
int rt_vad_feed(rt_vad_state *v, const float *pcm, int n, int rate,
                float threshold, int prefix_ms, int silence_ms,
                bool hysteresis, int min_active_ms,
                int *start_ms, bool *stopped, bool *discarded);

#endif
