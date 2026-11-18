/* Pure turn-detection state machine. See rt_vad.h. */
#include "rt_vad.h"

#include <math.h>
#include <string.h>

void rt_vad_state_init(rt_vad_state *v) {
    memset(v, 0, sizeof(*v));
    v->speech_start_ms = 0;
}

void rt_vad_state_reset(rt_vad_state *v) {
    /* in_ms is the session's input clock and must keep advancing; the commit
     * path clears it separately. Everything that belongs to one utterance is
     * reset here. */
    v->speech = false;
    v->silence_ms = 0;
    v->active_ms = 0;
    v->pending = false;
    v->pending_ms = 0;
}

int rt_vad_feed(rt_vad_state *v, const float *pcm, int n, int rate,
                float threshold, int prefix_ms, int silence_ms,
                bool hysteresis, int min_active_ms,
                int *start_ms, bool *stopped, bool *discarded) {
    *start_ms = -1;
    *stopped = false;
    *discarded = false;

    /* -55 .. -25 dBFS across the 0..1 client range. 0.5 -> -40 dBFS, between
     * a quiet room and conversational speech. */
    const double db_on = -55.0 + 30.0 * (double)threshold;
    const double db_off = hysteresis ? db_on - 30.0 * RT_VAD_HYST : db_on;
    const int frame = rate / 100;   /* 10 ms */

    for (int off = 0; off + frame <= n; off += frame) {
        double sum = 0.0;
        for (int i = 0; i < frame; i++) {
            double x = pcm[off + i];
            sum += x * x;
        }
        double rms = sqrt(sum / frame);
        double db = rms > 1e-9 ? 20.0 * log10(rms) : -120.0;
        v->in_ms += 10;

        if (!hysteresis) {
            /* Original single-threshold detector, kept verbatim so the
             * harness can A/B the two rules on exactly the same signal. */
            if (db > db_on) {
                v->silence_ms = 0;
                if (!v->speech) {
                    v->speech = true;
                    v->speech_start_ms = v->in_ms - prefix_ms;
                    if (v->speech_start_ms < 0) v->speech_start_ms = 0;
                    *start_ms = v->speech_start_ms;
                }
            } else if (v->speech) {
                v->silence_ms += 10;
                if (v->silence_ms >= silence_ms) {
                    v->speech = false;
                    v->silence_ms = 0;
                    *stopped = true;
                }
            }
            continue;
        }

        if (db > db_on) {
            /* Strong frame: confirm or extend the burst. */
            v->active_ms += 10;
            if (v->speech) {
                v->silence_ms = 0;
            } else if (!v->pending) {
                v->pending = true;
                v->pending_ms = v->in_ms - 10;   /* true onset, before this frame */
            }
            if (!v->speech && v->active_ms >= min_active_ms) {
                v->speech = true;
                v->speech_start_ms = v->pending_ms - prefix_ms;
                if (v->speech_start_ms < 0) v->speech_start_ms = 0;
                *start_ms = v->speech_start_ms;
                v->silence_ms = 0;
            }
        } else if (v->speech) {
            /* Between db_off and db_on is a hold region: a short dip inside a
             * word neither counts as silence nor cancels a pending end. Only
             * below db_off does the quiet timer advance. */
            if (db <= db_off) {
                v->silence_ms += 10;
                if (v->silence_ms >= silence_ms) {
                    v->speech = false;
                    v->silence_ms = 0;
                    *stopped = true;
                    v->pending = false;
                    v->active_ms = 0;
                }
            }
        } else if (v->pending) {
            /* Fell back below the release level before the fragment floor was
             * met. A click or a keyboard tap is not an utterance. */
            if (db <= db_off) {
                v->pending = false;
                v->active_ms = 0;
                *discarded = true;
            }
        }
    }
    return 0;
}
