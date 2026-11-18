/* rt_hift.h — the talker's vocoder: mel frames -> 24 kHz mono PCM.
 *
 * A causal HiFT generator: an f0 predictor drives a harmonic-plus-noise source,
 * the source's STFT is fused into three upsample stages, and the network emits
 * magnitude/phase frames that an inverse STFT turns into samples. One mel
 * frame is 480 samples (25 fps tokens x 2 mel frames x 480). */
#ifndef CWICK_RT_HIFT_H
#define CWICK_RT_HIFT_H

#include "cwick.h"

typedef struct rt_hift rt_hift;

/* `path` is talker/hift.pt. */
rt_hift *rt_hift_open(const char *path, cwick_error *err);
void rt_hift_close(rt_hift *h);

/* mel is [T][80] time-major. Writes T*480 samples to *pcm (caller frees) and
 * returns the sample count, or -1 on failure. */
int rt_hift_synth(rt_hift *h, const float *mel, int T, float **pcm);

#endif /* CWICK_RT_HIFT_H */