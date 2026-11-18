/* rt_realtime.h — one resident speech pipeline and the /v1/realtime sessions
 * that drive it.
 *
 * The checkpoint is a full-duplex pair: a 32B listening decoder (rt_thinker,
 * with rt_ear turning 16 kHz waveforms into the embeddings that sit at the
 * <|audio|> positions) and a 0.5B speaking stack (rt_talker). Both live here,
 * opened once, because they are one model and the machine holds one model.
 *
 * A session is a WebSocket speaking the OpenAI Realtime event protocol. Two
 * threads run it: the connection thread parses client events and keeps
 * consuming microphone audio, and a session worker runs the turn. That split
 * is what makes the duplex real — the reader never stops reading while the
 * talker is rendering, so a user who talks over the answer is heard and can
 * barge in instead of being buffered until the sentence ends.
 *
 * Every stage reports its own wall clock, both to stdout (tagged with the
 * request id the connection thread installed) and into the aggregate counters
 * below, which is where the full-chain numbers come from. */
#ifndef CWICK_RT_REALTIME_H
#define CWICK_RT_REALTIME_H

#include <stdbool.h>
#include <stdint.h>

#include "cwick.h"
#include "rt_thinker.h"

typedef struct rt_realtime rt_realtime;

/* Append-time sample-rate conversion. The wire rate is 24 kHz and the ear
 * wants 16 kHz. Each append is filtered against the whole turn's raw input
 * (not just the chunk) and only the newly-produced suffix is emitted, so a
 * chunked stream is bit-identical to one whole-clip resample: zero-padding
 * each 200 ms append instead gates the waveform at every boundary and the
 * audio tower hears silence. Reset between turns; free when the session ends.
 * Returns the output sample count; *out is malloc'd. from == to is a copy. */
#define RT_RS_TAPS 32
typedef struct {
    float *raw;      /* accumulated input-rate samples for the current turn */
    int raw_n, raw_cap;
    int out_n;       /* resampled samples already emitted */
} rt_resample_state;
int rt_resample_stream(rt_resample_state *st, const float *in, int n,
                       int from, int to, float **out);
void rt_resample_reset(rt_resample_state *st);
void rt_resample_free(rt_resample_state *st);

/* Full-chain counters, summed over every turn this process has served. The
 * `last_*` fields are the most recent turn, which is what a client wants to
 * see next to the answer it just heard. All seconds are wall clock. */
typedef struct {
    uint64_t sessions;          /* WebSocket sessions accepted */
    uint64_t turns;             /* turns that produced a reply */
    uint64_t cancelled;         /* turns cut short by barge-in or cancel */
    uint64_t errors;
    /* Cumulative stage seconds. */
    double ear_s, prefill_s, decode_s, talker_s, turn_s;
    double first_audio_s;       /* commit -> first PCM byte on the wire */
    /* Cumulative counts. */
    uint64_t in_samples;        /* 16 kHz samples handed to the ear */
    uint64_t out_samples;       /* 24 kHz samples sent to the client */
    uint64_t audio_tokens;      /* <|audio|> placeholder rows */
    uint64_t prompt_tokens;     /* decoder positions prefilled */
    uint64_t reply_tokens;      /* decoder tokens sampled */
    uint64_t speech_tokens;     /* talker LM tokens */
    /* Most recent turn. */
    double last_ear_s, last_prefill_s, last_decode_s, last_talker_s;
    double last_first_audio_s, last_turn_s;
    int last_prompt_tokens, last_reply_tokens, last_audio_tokens;
    double last_audio_in_s, last_audio_out_s;
} rt_realtime_stats;

/* Opens `<root>` (a sophon_realtime checkpoint: config.json naming the thinker
 * and talker subdirectories and the voices). `backend` selects the decoder's
 * GEMM path. Nothing is loaded lazily later: a session must not pay a 32 GiB
 * weight fault in the middle of a conversation. */
rt_realtime *rt_realtime_open(const char *root, rt_backend backend,
                              cwick_error *err);
void rt_realtime_close(rt_realtime *r);

/* The served name, e.g. "Sophon-Realtime-32B-Tag". */
const char *rt_realtime_model(const rt_realtime *r);
const char *rt_realtime_root(const rt_realtime *r);
/* Thinker geometry, for the metrics registry after a hot-switch. */
int rt_realtime_context(const rt_realtime *r);
int rt_realtime_layers(const rt_realtime *r);
const char *rt_realtime_default_voice(const rt_realtime *r);

/* Voices shipped in the checkpoint, in directory order. */
int rt_realtime_voice_count(const rt_realtime *r);
const char *rt_realtime_voice_name(const rt_realtime *r, int i);

/* True while any session holds the pipeline or is connected. The model swap
 * gate refuses to free weights out from under a live conversation. */
bool rt_realtime_busy(const rt_realtime *r);

void rt_realtime_stats_get(const rt_realtime *r, rt_realtime_stats *out);
/* Renders the counters as the JSON body of GET /v1/realtime/stats. Caller
 * frees. */
char *rt_realtime_stats_json(const rt_realtime *r);

/* Serve one already-upgraded socket until the peer closes or `stop` is set.
 * `rid` is the access-log request id; `query` is the raw request query (the
 * `model` parameter is validated against the resident pipeline). Returns 0 on
 * a clean close, -1 on a protocol or socket failure. The caller still owns
 * `fd` and closes it afterwards. */
int rt_realtime_serve(rt_realtime *r, int fd, uint64_t rid, const char *query,
                      const volatile int *stop);

#endif /* CWICK_RT_REALTIME_H */