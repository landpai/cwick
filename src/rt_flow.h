/* rt_flow.h — speech tokens to mel: the talker's distilled flow decoder.
 *
 * token (25 fps) -> embedding -> pre-lookahead conv -> x2 repeat -> DiT
 * conditional flow matching (4 euler steps, cosine schedule, no guidance)
 * -> mel (50 fps, 80 bins). One mel frame is 480 samples of 24 kHz audio,
 * which is exactly the vocoder's hop.
 *
 * The prompt region is carried through the whole stack and dropped at the
 * end: it is what makes the output sound like the reference speaker. */
#ifndef CWICK_RT_FLOW_H
#define CWICK_RT_FLOW_H

#include "cwick.h"

typedef struct rt_flow rt_flow;

/* Opens talker/flow.pt. Solver steps and guidance rate come from
 * RT_FLOW_STEPS (default 4) and RT_FLOW_CFG (default 0): this checkpoint is
 * distilled for four steps without classifier-free guidance. */
rt_flow *rt_flow_open(const char *path, cwick_error *err);
void rt_flow_close(rt_flow *f);

/* Renders `n` new speech tokens against a voice prompt and writes
 * (n - lookahead - skip) * 2 mel frames of 80 bins to `out`.
 *
 * `lookahead` is 3 for every streaming chunk but the last: those trailing
 * tokens feed the pre-lookahead convolution without producing audio, which is
 * what lets a chunk be rendered before the sentence is finished. Pass 0 to
 * finalize. `streaming` selects the chunk-causal attention mask (50 mel frames
 * per chunk, all left chunks) instead of full attention; the two agree on the
 * frames they share, so a caller can render chunk by chunk.
 *
 * `token` is always the whole prefix generated so far, because a chunk's
 * frames attend to every earlier chunk. `skip` is how many of those leading
 * tokens have already been emitted: their frames are recomputed (they come
 * out identical) but not written, so the caller gets only the new audio.
 *
 * prompt_token/prompt_feat/embedding are the voice prompt's own tensors
 * (150 tokens, 300 mel frames, 192-d speaker vector for the shipped voices);
 * prompt_feat must hold exactly pt * 2 frames.
 * Returns the mel frame count, or -1. */
int rt_flow_mel(rt_flow *f, const int *prompt_token, int pt,
                const float *prompt_feat, int pf, const float *embedding,
                const int *token, int n, int lookahead, int skip, int streaming,
                float *out);

/* Mel frames rt_flow_mel will write for the same arguments. */
int rt_flow_frames(int n, int lookahead, int skip);

#endif /* CWICK_RT_FLOW_H */