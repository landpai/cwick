/* rt_realtime.c — see rt_realtime.h.
 *
 * The turn, in the order the stages actually run:
 *
 *   committed 16 kHz clip
 *     -> rt_ear_embed          log-mel, audio tower, projector   (~0.09 s)
 *     -> rt_thinker_forward    the new user block only           (KV is kept)
 *     -> sample loop           text deltas as they are decoded
 *     -> rt_talker_say         per sentence, PCM deltas as they render
 *
 * The decoder's KV cache survives across turns, so a turn prefills the audio
 * it just heard and nothing else. Re-prefilling the system prompt and the whole
 * conversation every time cost 6.8 s of the 14 s turn on this machine; keeping
 * the cache is the difference between an answer that arrives while the question
 * is still fresh and one that arrives after the user has given up. */
#include <dirent.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include "json.h"
#include "log.h"
#include "metrics.h"
#include "models.h"
#include "rt_audio.h"
#include "rt_realtime.h"
#include "rt_talker.h"
#include "rt_vad.h"
#include "rt_thinker.h"
#include "sample.h"
#include "tools.h"
#include "util.h"
#include "ws.h"

#define RT_TAG "realtime"

#define AUDIO_TOKEN 151669
#define IM_END 151645
#define ENDOFTEXT 151643

/* The wire format the OpenAI Realtime API names "pcm16": mono, little-endian,
 * 24 kHz both ways. The ear wants 16 kHz, so input is resampled on append and
 * the talker's 24 kHz output goes out untouched. */
#define WIRE_RATE 24000
#define EAR_RATE 16000
/* The ear's conv frontend is exact only on whole mel frames, and the reference
 * pads to a multiple of 8 stacked frames before encoding. */
#define EAR_PAD 2560

#define MAX_REPLY_DEFAULT 512
#define MAX_SESSIONS 8

static double now_s(void) { return (double)cwick_prof_ns() / 1e9; }

/* ------------------------------------------------------------------ */
/* Growable JSON text                                                  */
/* ------------------------------------------------------------------ */

typedef struct { char *p; size_t len, cap; } jbuf;

static void jb_grow(jbuf *b, size_t extra) {
    if (b->len + extra + 1 <= b->cap) return;
    size_t want = b->cap ? b->cap : 512;
    while (want < b->len + extra + 1) want *= 2;
    b->p = cwick_xrealloc(b->p, want);
    b->cap = want;
}
static void jb_putn(jbuf *b, const char *s, size_t n) {
    jb_grow(b, n);
    memcpy(b->p + b->len, s, n);
    b->len += n;
    b->p[b->len] = '\0';
}
static void jb_puts(jbuf *b, const char *s) { jb_putn(b, s, strlen(s)); }
static void jb_putf(jbuf *b, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));
static void jb_putf(jbuf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char *piece = NULL;
    if (vasprintf(&piece, fmt, ap) < 0 || !piece) { va_end(ap); return; }
    va_end(ap);
    jb_puts(b, piece);
    free(piece);
}
/* A quoted, escaped JSON string. cwick_json_stringify owns its result. */
static void jb_str(jbuf *b, const char *s) {
    char *q = cwick_json_stringify(s ? s : "");
    jb_puts(b, q);
    free(q);
}
static void jb_free(jbuf *b) { free(b->p); b->p = NULL; b->len = b->cap = 0; }

/* ------------------------------------------------------------------ */
/* Sample-rate conversion                                              */
/* ------------------------------------------------------------------ */

/* Windowed-sinc, 32 taps. Downsampling 24 kHz to 16 kHz moves Nyquist from
 * 12 kHz to 8 kHz, so everything above 8 kHz has to go before the decimation or
 * it folds back into the band the mel filterbank actually reads. Averaging
 * groups of three samples would be cheaper and would leave that fold-in place,
 * which the audio tower hears as a metallic edge on every sibilant. */
#define RS_TAPS RT_RS_TAPS

static float *rs_kernel(int from, int to) {
    float *h = cwick_xmalloc(RS_TAPS * sizeof(float));
    double fc = 0.5 * (to < from ? (double)to / from : 1.0);
    double mid = (RS_TAPS - 1) / 2.0;
    double sum = 0.0;
    for (int i = 0; i < RS_TAPS; i++) {
        double x = i - mid;
        double s = (x == 0.0) ? 2.0 * fc
                              : sin(2.0 * M_PI * fc * x) / (M_PI * x);
        double w = 0.54 - 0.46 * cos(2.0 * M_PI * i / (RS_TAPS - 1));
        h[i] = (float)(s * w);
        sum += h[i];
    }
    /* Unity DC gain: the sinc is already scaled by 2*fc, but the window costs a
     * little, and a quiet clip must not come out quieter still. */
    if (sum > 0.0) for (int i = 0; i < RS_TAPS; i++) h[i] /= (float)sum;
    return h;
}

/* Returns the output sample count; *out is malloc'd. `from`/`to` are Hz. */
static int resample(const float *in, int n, int from, int to, float **out) {
    if (from == to) {
        float *c = cwick_xmalloc((size_t)(n > 0 ? n : 1) * sizeof(float));
        memcpy(c, in, (size_t)n * sizeof(float));
        *out = c;
        return n;
    }
    float *h = rs_kernel(from, to);
    int64_t m = (int64_t)n * to / from;
    float *y = cwick_xmalloc((size_t)(m > 0 ? m : 1) * sizeof(float));
    double ratio = (double)from / to;
    int half = RS_TAPS / 2;
    for (int64_t j = 0; j < m; j++) {
        double src = (double)j * ratio;
        int64_t base = (int64_t)floor(src) - half + 1;
        double acc = 0.0;
        for (int t = 0; t < RS_TAPS; t++) {
            int64_t k = base + t;
            if (k < 0 || k >= n) continue;      /* zero outside the clip */
            acc += (double)h[t] * in[k];
        }
        y[j] = (float)acc;
    }
    free(h);
    *out = y;
    return (int)m;
}

/* ------------------------------------------------------------------ */
/* The resident pipeline                                               */
/* ------------------------------------------------------------------ */

typedef struct {
    char name[64];
    rt_voice v;
    bool loaded;
} voice_slot;

struct rt_realtime {
    char root[CWICK_MODEL_ROOT_MAX];
    char model[CWICK_MODEL_ID_MAX];
    char thinker_dir[CWICK_MODEL_ROOT_MAX];
    char talker_dir[CWICK_MODEL_ROOT_MAX];
    char voices_dir[CWICK_MODEL_ROOT_MAX];
    char default_voice[64];

    rt_thinker *thinker;
    rt_ear *ear;
    rt_talker *talker;
    cwick_tokenizer *tok;

    voice_slot *voice;
    int nvoice;

    /* One decoder, one KV cache: a second session's turn waits here rather
     * than corrupting the first one's positions. */
    pthread_mutex_t pipe_mtx;
    int sessions;               /* connected, guarded by pipe_mtx */

    pthread_mutex_t stat_mtx;
    rt_realtime_stats st;
};

static void rt_stat_add(rt_realtime *r, const rt_realtime_stats *d) {
    pthread_mutex_lock(&r->stat_mtx);
    r->st.turns += d->turns;
    r->st.cancelled += d->cancelled;
    r->st.ear_s += d->ear_s;
    r->st.prefill_s += d->prefill_s;
    r->st.decode_s += d->decode_s;
    r->st.talker_s += d->talker_s;
    r->st.turn_s += d->turn_s;
    r->st.first_audio_s += d->first_audio_s;
    r->st.in_samples += d->in_samples;
    r->st.out_samples += d->out_samples;
    r->st.audio_tokens += d->audio_tokens;
    r->st.prompt_tokens += d->prompt_tokens;
    r->st.reply_tokens += d->reply_tokens;
    r->st.speech_tokens += d->speech_tokens;
    r->st.last_ear_s = d->last_ear_s;
    r->st.last_prefill_s = d->last_prefill_s;
    r->st.last_decode_s = d->last_decode_s;
    r->st.last_talker_s = d->last_talker_s;
    r->st.last_first_audio_s = d->last_first_audio_s;
    r->st.last_turn_s = d->last_turn_s;
    r->st.last_prompt_tokens = d->last_prompt_tokens;
    r->st.last_reply_tokens = d->last_reply_tokens;
    r->st.last_audio_tokens = d->last_audio_tokens;
    r->st.last_audio_in_s = d->last_audio_in_s;
    r->st.last_audio_out_s = d->last_audio_out_s;
    pthread_mutex_unlock(&r->stat_mtx);

    /* The same numbers the turn summary prints go into the Prometheus
     * registry, so a scrape can chart the stages without parsing stdout. */
    cwick_rt_turn_metrics m;
    memset(&m, 0, sizeof(m));
    m.ear_s = d->ear_s;
    m.prefill_s = d->prefill_s;
    m.decode_s = d->decode_s;
    m.talker_s = d->talker_s;
    m.first_audio_s = d->first_audio_s;
    m.turn_s = d->turn_s;
    m.audio_in_s = d->last_audio_in_s;
    m.audio_out_s = (double)d->out_samples / WIRE_RATE;
    m.prompt_tokens = d->prompt_tokens;
    m.reply_tokens = d->reply_tokens;
    m.speech_tokens = d->speech_tokens;
    m.audio_tokens = d->audio_tokens;
    m.out_samples = d->out_samples;
    m.cancelled = d->cancelled != 0;
    cwick_metrics_rt_observe(&m);
}

/* config.json names the subdirectories; the defaults are the ones the released
 * checkpoint uses, so a config that omits them still opens. */
static void cfg_str(json_value *j, const char *key, char *out, size_t cap,
                    const char *dflt) {
    const char *s = cwick_json_str(cwick_json_get(j, key));
    snprintf(out, cap, "%s", (s && *s) ? s : dflt);
}

rt_realtime *rt_realtime_open(const char *root, rt_backend backend,
                              cwick_error *err) {
    if (!root || !*root) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "empty realtime model root");
        return NULL;
    }
    char cfg[CWICK_MODEL_ROOT_MAX + 32];
    snprintf(cfg, sizeof(cfg), "%s/config.json", root);
    FILE *f = fopen(cfg, "rb");
    if (!f) {
        cwick_error_set(err, CWICK_ERR_IO, "cannot read %s", cfg);
        return NULL;
    }
    char text[8192];
    size_t n = fread(text, 1, sizeof(text) - 1, f);
    fclose(f);
    text[n] = '\0';
    json_value *j = cwick_json_parse(text, n, err);
    if (!j) return NULL;

    rt_realtime *r = cwick_xmalloc(sizeof(*r));
    memset(r, 0, sizeof(*r));
    snprintf(r->root, sizeof(r->root), "%s", root);
    char sub[256];
    cfg_str(j, "thinker_subdir", sub, sizeof(sub), "thinker");
    snprintf(r->thinker_dir, sizeof(r->thinker_dir), "%s/%s", root, sub);
    cfg_str(j, "talker_subdir", sub, sizeof(sub), "talker");
    snprintf(r->talker_dir, sizeof(r->talker_dir), "%s/%s", root, sub);
    cfg_str(j, "voices_subdir", sub, sizeof(sub), "voices");
    snprintf(r->voices_dir, sizeof(r->voices_dir), "%s/%s", root, sub);
    cfg_str(j, "default_voice", r->default_voice, sizeof(r->default_voice), "vivian_zh");
    cwick_json_free(j);

    /* The served name is the directory the checkpoint was released under, not
     * the training-run artifact in config.json's model_type. Trim trailing
     * slashes first so "…/Sophon-Realtime-32B-Tag/" names the checkpoint rather
     * than the empty string after the last slash. */
    size_t rlen = strlen(root);
    while (rlen > 1 && root[rlen - 1] == '/') rlen--;
    size_t bstart = 0;
    for (size_t i = rlen; i > 0; i--)
        if (root[i - 1] == '/') { bstart = i; break; }
    snprintf(r->model, sizeof(r->model), "%.*s",
             (int)(rlen - bstart), root + bstart);

    pthread_mutex_init(&r->pipe_mtx, NULL);
    pthread_mutex_init(&r->stat_mtx, NULL);

    double t0 = now_s();
    r->thinker = rt_thinker_open(r->thinker_dir, backend, 4096, err);
    if (!r->thinker) { rt_realtime_close(r); return NULL; }
    CWICK_INFO(RT_TAG, "thinker ready: hidden %d, vocab %d, kv %d, %s (%.2f s)",
               rt_thinker_hidden(r->thinker), rt_thinker_vocab(r->thinker),
               rt_thinker_capacity(r->thinker),
               rt_thinker_on_gpu(r->thinker) ? "metal" : "cpu", now_s() - t0);

    double t1 = now_s();
    r->ear = rt_ear_open(r->thinker_dir, err);
    if (!r->ear) { rt_realtime_close(r); return NULL; }
    CWICK_INFO(RT_TAG, "ear ready: hidden %d, %d tokens per mel chunk (%.2f s)",
               rt_ear_hidden(r->ear), rt_ear_frame_tokens(r->ear), now_s() - t1);

    double t2 = now_s();
    r->talker = rt_talker_open(r->talker_dir, err);
    if (!r->talker) { rt_realtime_close(r); return NULL; }
    CWICK_INFO(RT_TAG, "talker ready: %d Hz out (%.2f s)",
               rt_talker_sample_rate(r->talker), now_s() - t2);

    cwick_model_spec spec;
    memset(&spec, 0, sizeof(spec));
    if (cwick_tokenizer_load(r->thinker_dir, &spec, &r->tok, err) != CWICK_OK) {
        rt_realtime_close(r);
        return NULL;
    }

    /* Voice prompts are a few hundred kilobytes each and every session picks
     * one, so they are enumerated now and widened on first use. */
    DIR *d = opendir(r->voices_dir);
    if (d) {
        struct dirent *de;
        while ((de = readdir(d)) != NULL) {
            size_t l = strlen(de->d_name);
            if (l < 4 || strcmp(de->d_name + l - 3, ".pt") != 0) continue;
            r->voice = cwick_xrealloc(r->voice, (size_t)(r->nvoice + 1) * sizeof(*r->voice));
            voice_slot *v = &r->voice[r->nvoice++];
            memset(v, 0, sizeof(*v));
            snprintf(v->name, sizeof(v->name), "%.*s", (int)(l - 3), de->d_name);
        }
        closedir(d);
    }
    /* Directory order is filesystem order; a stable list is what a client
     * rendering a voice picker needs. */
    for (int i = 1; i < r->nvoice; i++)
        for (int k = i; k > 0 && strcmp(r->voice[k].name, r->voice[k - 1].name) < 0; k--) {
            voice_slot tmp = r->voice[k];
            r->voice[k] = r->voice[k - 1];
            r->voice[k - 1] = tmp;
        }
    CWICK_INFO(RT_TAG, "pipeline ready: %s, %d voice(s), default %s (%.2f s total)",
               r->model, r->nvoice, r->default_voice, now_s() - t0);
    return r;
}

void rt_realtime_close(rt_realtime *r) {
    if (!r) return;
    for (int i = 0; i < r->nvoice; i++)
        if (r->voice[i].loaded) rt_voice_free(&r->voice[i].v);
    free(r->voice);
    if (r->tok) cwick_tokenizer_free(r->tok);
    if (r->talker) rt_talker_close(r->talker);
    if (r->ear) rt_ear_close(r->ear);
    if (r->thinker) rt_thinker_close(r->thinker);
    pthread_mutex_destroy(&r->pipe_mtx);
    pthread_mutex_destroy(&r->stat_mtx);
    free(r);
}

const char *rt_realtime_model(const rt_realtime *r) { return r->model; }
const char *rt_realtime_root(const rt_realtime *r) { return r->root; }
int rt_realtime_context(const rt_realtime *r) {
    return r->thinker ? rt_thinker_capacity(r->thinker) : 0;
}
int rt_realtime_layers(const rt_realtime *r) {
    return r->thinker ? rt_thinker_layers(r->thinker) : 0;
}
const char *rt_realtime_default_voice(const rt_realtime *r) { return r->default_voice; }
int rt_realtime_voice_count(const rt_realtime *r) { return r->nvoice; }
const char *rt_realtime_voice_name(const rt_realtime *r, int i) {
    return (i >= 0 && i < r->nvoice) ? r->voice[i].name : NULL;
}

bool rt_realtime_busy(const rt_realtime *r) {
    if (!r) return false;
    pthread_mutex_lock((pthread_mutex_t *)&r->pipe_mtx);
    bool b = r->sessions > 0;
    pthread_mutex_unlock((pthread_mutex_t *)&r->pipe_mtx);
    return b;
}

void rt_realtime_stats_get(const rt_realtime *r, rt_realtime_stats *out) {
    pthread_mutex_lock((pthread_mutex_t *)&r->stat_mtx);
    *out = r->st;
    pthread_mutex_unlock((pthread_mutex_t *)&r->stat_mtx);
}

char *rt_realtime_stats_json(const rt_realtime *r) {
    rt_realtime_stats s;
    rt_realtime_stats_get(r, &s);
    jbuf b;
    memset(&b, 0, sizeof(b));
    jb_puts(&b, "{\"model\":");
    jb_str(&b, r->model);
    jb_putf(&b,
            ",\"sessions\":%" PRIu64 ",\"turns\":%" PRIu64
            ",\"cancelled\":%" PRIu64 ",\"errors\":%" PRIu64
            ",\"cumulative\":{\"ear_s\":%.4f,\"prefill_s\":%.4f,\"decode_s\":%.4f,"
            "\"talker_s\":%.4f,\"turn_s\":%.4f,\"first_audio_s\":%.4f,"
            "\"audio_in_s\":%.3f,\"audio_out_s\":%.3f,"
            "\"audio_tokens\":%" PRIu64 ",\"prompt_tokens\":%" PRIu64
            ",\"reply_tokens\":%" PRIu64 ",\"speech_tokens\":%" PRIu64 "}",
            s.sessions, s.turns, s.cancelled, s.errors,
            s.ear_s, s.prefill_s, s.decode_s, s.talker_s, s.turn_s, s.first_audio_s,
            (double)s.in_samples / EAR_RATE, (double)s.out_samples / WIRE_RATE,
            s.audio_tokens, s.prompt_tokens, s.reply_tokens, s.speech_tokens);
    double it = s.turns ? (double)s.turns : 1.0;
    jb_putf(&b,
            ",\"mean_per_turn\":{\"ear_s\":%.4f,\"prefill_s\":%.4f,\"decode_s\":%.4f,"
            "\"talker_s\":%.4f,\"turn_s\":%.4f,\"first_audio_s\":%.4f}",
            s.ear_s / it, s.prefill_s / it, s.decode_s / it,
            s.talker_s / it, s.turn_s / it, s.first_audio_s / it);
    jb_putf(&b,
            ",\"rates\":{\"ear_realtime_x\":%.2f,\"prefill_tok_s\":%.2f,"
            "\"decode_tok_s\":%.2f,\"talker_realtime_x\":%.2f}",
            s.ear_s > 0 ? ((double)s.in_samples / EAR_RATE) / s.ear_s : 0.0,
            s.prefill_s > 0 ? (double)s.prompt_tokens / s.prefill_s : 0.0,
            s.decode_s > 0 ? (double)s.reply_tokens / s.decode_s : 0.0,
            s.talker_s > 0 ? ((double)s.out_samples / WIRE_RATE) / s.talker_s : 0.0);
    jb_putf(&b,
            ",\"last_turn\":{\"ear_s\":%.4f,\"prefill_s\":%.4f,\"decode_s\":%.4f,"
            "\"talker_s\":%.4f,\"turn_s\":%.4f,\"first_audio_s\":%.4f,"
            "\"prompt_tokens\":%d,\"reply_tokens\":%d,\"audio_tokens\":%d,"
            "\"audio_in_s\":%.3f,\"audio_out_s\":%.3f}}",
            s.last_ear_s, s.last_prefill_s, s.last_decode_s, s.last_talker_s,
            s.last_turn_s, s.last_first_audio_s,
            s.last_prompt_tokens, s.last_reply_tokens, s.last_audio_tokens,
            s.last_audio_in_s, s.last_audio_out_s);
    return b.p;
}

/* ------------------------------------------------------------------ */
/* Session state                                                       */
/* ------------------------------------------------------------------ */

/* One committed utterance, waiting for the worker. The audio is already at the
 * ear's rate and the reader owns nothing of it once it is queued. */
typedef struct clip {
    float *pcm;                 /* 16 kHz mono */
    int n;
    int audio_start_ms;         /* VAD offset inside the pre-commit buffer */
    struct clip *next;
} clip;

/* A past turn, kept so the KV cache can be rebuilt exactly when it fills. The
 * audio embeddings are small (25 rows x 5120 floats is half a megabyte) and
 * re-encoding them from a transcript is not possible: there is no ASR head. */
typedef struct {
    float *audio;
    int atok;
    char *reply;
} hist_turn;

typedef struct {
    rt_realtime *r;
    int fd;
    uint64_t rid;
    cwick_ws ws;
    pthread_mutex_t send_mtx;   /* whole frames, so two threads cannot interleave */

    char id[48];
    char model[CWICK_MODEL_ID_MAX];
    char voice[64];
    char instructions[4096];
    float temperature;
    float top_p;
    int top_k;
    int max_reply;
    int input_rate;
    bool want_text;             /* modalities includes "text" */
    bool want_audio;            /* modalities includes "audio" */

    /* turn detection */
    bool vad;                   /* false = the client commits by hand */
    float vad_threshold;
    int prefix_padding_ms;
    int silence_duration_ms;
    bool vad_create_response;
    bool vad_interrupt_response;
    bool legacy_vad;            /* CWICK_RT_VAD_LEGACY=1: original threshold */
    rt_vad_state vad_st;        /* turn detection, extracted for A/B testing */

    /* Guarded by mtx: everything the reader and the worker both touch. */
    pthread_mutex_t mtx;
    pthread_cond_t cv;
    pthread_t worker;
    bool worker_started;
    bool stop;
    clip *head, *tail;
    bool turn_active;
    bool cancel;                /* barge-in or response.cancel */
    uint64_t item_seq;
    uint64_t resp_seq;
    uint64_t event_seq;

    /* Input buffer at the ear's rate, plus the VAD's running state. */
    float *in;
    int in_n, in_cap;

    /* Overlap-save tail for the append-time resampler (input-rate samples). */
    rt_resample_state rs;

    /* A manual input_audio_buffer.commit announces the item but does not start
     * a response: the audio waits here for the client's response.create. */
    float *pending;
    int pending_n, pending_ms;

    /* Conversation, guarded by mtx (the worker reads it, session.update and
     * conversation.item.create write it). */
    hist_turn *hist;
    int nhist, hist_cap;
    int kv_pos;                 /* decoder positions the history accounts for */
    bool kv_valid;

    /* The response in flight, for the ids the deltas carry. */
    char resp_id[48];
    char resp_item_id[48];
    int resp_output_index;
} rt_session;

static const char *DEFAULT_INSTRUCTIONS =
    "你是智子（Sophon），AI降临派（Adventists AI）训练的 Sophon 系列模型，能听、能看，也能说。"
    "你会直接听到用户的语音，你的回复会被直接合成为语音说出来。请像真人对话一样，用自然的口语一两句话回应"
    "用户，注意用户的情绪。只写要说出口的话，不要写括号说明、表情或格式符号；数字、日期、金额、符号和英文"
    "缩写都直接写成口语念法。需要时可以用声音标签表现笑声、咳嗽和重读。";

static void new_id(char *out, size_t cap, const char *prefix, uint64_t seq) {
    snprintf(out, cap, "%s_%08" PRIx64 "%04x", prefix, seq,
             (unsigned)(cwick_prof_ns() & 0xffffu));
}

/* ---- event emission ------------------------------------------------ */

static void ev_send(rt_session *s, jbuf *b) {
    pthread_mutex_lock(&s->send_mtx);
    if (!s->ws.failed && !s->ws.peer_closed) cwick_ws_send_text(&s->ws, b->p);
    pthread_mutex_unlock(&s->send_mtx);
    jb_free(b);
}

/* Every server event carries event_id; the caller appends the payload. */
static void ev_begin(rt_session *s, jbuf *b, const char *type) {
    jb_puts(b, "{\"event_id\":");
    char eid[48];
    pthread_mutex_lock(&s->mtx);
    new_id(eid, sizeof(eid), "evt", s->event_seq++);
    pthread_mutex_unlock(&s->mtx);
    jb_str(b, eid);
    jb_putf(b, ",\"type\":");
    jb_str(b, type);
}

static void ev_error(rt_session *s, const char *type, const char *code,
                     const char *message, const char *param) {
    CWICK_INFO(RT_TAG, "error %s/%s: %s", type, code ? code : "-", message);
    pthread_mutex_lock(&s->r->stat_mtx);
    s->r->st.errors++;
    pthread_mutex_unlock(&s->r->stat_mtx);
    jbuf b;
    memset(&b, 0, sizeof(b));
    ev_begin(s, &b, "error");
    jb_puts(&b, ",\"error\":{\"type\":");
    jb_str(&b, type);
    jb_puts(&b, ",\"code\":");
    if (code) jb_str(&b, code); else jb_puts(&b, "null");
    jb_puts(&b, ",\"message\":");
    jb_str(&b, message);
    jb_puts(&b, ",\"param\":");
    if (param) jb_str(&b, param); else jb_puts(&b, "null");
    jb_puts(&b, ",\"event_id\":null}}");
    ev_send(s, &b);
}

/* The session object, as session.created / session.updated carry it. */
static void session_json(rt_session *s, jbuf *b) {
    jb_puts(b, "{\"id\":");
    jb_str(b, s->id);
    jb_puts(b, ",\"object\":\"realtime.session\",\"model\":");
    jb_str(b, s->model);
    jb_puts(b, ",\"modalities\":[");
    bool first = true;
    if (s->want_text) { jb_puts(b, "\"text\""); first = false; }
    if (s->want_audio) { if (!first) jb_puts(b, ","); jb_puts(b, "\"audio\""); }
    jb_puts(b, "],\"instructions\":");
    jb_str(b, s->instructions);
    jb_puts(b, ",\"voice\":");
    jb_str(b, s->voice);
    jb_putf(b, ",\"input_audio_format\":\"pcm16\",\"output_audio_format\":\"pcm16\""
               ",\"input_audio_sample_rate\":%d,\"output_audio_sample_rate\":%d"
               ",\"temperature\":%.2f,\"max_response_output_tokens\":%d"
               ",\"input_audio_transcription\":null,\"tools\":[]",
            s->input_rate, WIRE_RATE, (double)s->temperature, s->max_reply);
    if (s->vad) {
        jb_putf(b, ",\"turn_detection\":{\"type\":\"server_vad\",\"threshold\":%.2f"
                   ",\"prefix_padding_ms\":%d,\"silence_duration_ms\":%d"
                   ",\"create_response\":%s,\"interrupt_response\":%s}",
                (double)s->vad_threshold, s->prefix_padding_ms, s->silence_duration_ms,
                s->vad_create_response ? "true" : "false",
                s->vad_interrupt_response ? "true" : "false");
    } else {
        jb_puts(b, ",\"turn_detection\":null");
    }
    /* The voices this checkpoint actually ships, so a client can render a
     * picker without a second round trip. */
    jb_puts(b, ",\"cwick\":{\"voices\":[");
    for (int i = 0; i < s->r->nvoice; i++) {
        if (i) jb_puts(b, ",");
        jb_str(b, s->r->voice[i].name);
    }
    jb_putf(b, "],\"input_sample_rate\":%d,\"output_sample_rate\":%d}}",
            EAR_RATE, WIRE_RATE);
}

static void ev_session(rt_session *s, const char *type) {
    jbuf b;
    memset(&b, 0, sizeof(b));
    ev_begin(s, &b, type);
    jb_puts(&b, ",\"session\":");
    pthread_mutex_lock(&s->mtx);
    session_json(s, &b);
    pthread_mutex_unlock(&s->mtx);
    jb_puts(&b, "}");
    ev_send(s, &b);
}

/* ------------------------------------------------------------------ */
/* Input buffer and voice activity                                     */
/* ------------------------------------------------------------------ */

static void in_reserve(rt_session *s, int extra) {
    if (s->in_n + extra <= s->in_cap) return;
    int want = s->in_cap ? s->in_cap : EAR_RATE * 4;
    while (want < s->in_n + extra) want *= 2;
    s->in = cwick_xrealloc(s->in, (size_t)want * sizeof(float));
    s->in_cap = want;
}

static void in_clear(rt_session *s) {
    s->in_n = 0;
    rt_vad_state_reset(&s->vad_st);
    s->vad_st.in_ms = 0;
    rt_resample_reset(&s->rs);
}

/* Whole-turn resampler; see rt_realtime.h. Recomputing the cumulative buffer
 * per append keeps the sinc's left context real at every chunk boundary. We
 * additionally hold back the filter's right reach (a guard of output samples)
 * so a sample is only emitted once every tap that feeds it has real input on
 * both sides — otherwise the suffix emitted at an early append would have been
 * zero-padded on the right and would differ from the same sample in a later,
 * longer recompute. The upshot: the emitted stream is bit-identical to one
 * whole-clip resample (minus the held-back tail, ~1.6 ms of trailing audio).
 * Turns are a few seconds, so the repeated filter pass is cheap. Caller holds
 * s->mtx when the state lives in a session. */
int rt_resample_stream(rt_resample_state *st, const float *in, int n,
                       int from, int to, float **out) {
    if (from == to) {
        float *c = cwick_xmalloc((size_t)(n > 0 ? n : 1) * sizeof(float));
        memcpy(c, in, (size_t)n * sizeof(float));
        st->out_n += n;
        *out = c;
        return n;
    }
    if (st->raw_n + n > st->raw_cap) {
        int want = st->raw_cap ? st->raw_cap : 24000;
        while (want < st->raw_n + n) want *= 2;
        st->raw = cwick_xrealloc(st->raw, (size_t)want * sizeof(float));
        st->raw_cap = want;
    }
    memcpy(st->raw + st->raw_n, in, (size_t)n * sizeof(float));
    st->raw_n += n;
    float *full = NULL;
    int full_n = resample(st->raw, st->raw_n, from, to, &full);
    /* Right-reach guard: the last tap of output j sits at floor(j*from/to)+half,
     * so hold back ceil((half+1)*from/to)+1 samples to keep every emitted tap
     * inside the accumulated input. */
    double ratio = (double)from / to;
    int guard = (int)ceil((RS_TAPS / 2 + 1) * ratio) + 1;
    int safe = full_n - guard;
    if (safe < st->out_n) safe = st->out_n;
    int new_n = safe - st->out_n;
    float *y = cwick_xmalloc((size_t)(new_n > 0 ? new_n : 1) * sizeof(float));
    memcpy(y, full + st->out_n, (size_t)new_n * sizeof(float));
    free(full);
    st->out_n = safe;
    *out = y;
    return new_n;
}

void rt_resample_reset(rt_resample_state *st) {
    st->raw_n = 0;
    st->out_n = 0;
}

void rt_resample_free(rt_resample_state *st) {
    free(st->raw);
    st->raw = NULL;
    st->raw_n = st->raw_cap = st->out_n = 0;
}

/* Energy VAD on 10 ms frames. The checkpoint ships no neural detector, and
 * inventing one would be a second model resident; RMS in dBFS with the spec's
 * threshold mapped onto it is honest about what it is and is tunable from the
 * client. The decision itself lives in rt_vad.c so it can be driven offline
 * against the same signal twice (see tools/rtvad.c). CWICK_RT_VAD_LEGACY=1
 * keeps the old single-threshold behaviour for a live A/B. */
static void vad_frame(rt_session *s, const float *pcm, int n, int *start_ms,
                      bool *stopped) {
    bool discarded = false;
    if (!s->vad) {
        *start_ms = -1;
        *stopped = false;
        return;
    }
    rt_vad_feed(&s->vad_st, pcm, n, EAR_RATE, s->vad_threshold,
                s->prefix_padding_ms, s->silence_duration_ms,
                !s->legacy_vad, RT_VAD_MIN_ACTIVE_MS,
                start_ms, stopped, &discarded);
    if (discarded) {
        CWICK_INFO(RT_TAG, "vad: dropped %d ms burst below the fragment floor",
                   RT_VAD_MIN_ACTIVE_MS);
        /* A rejected burst is noise: do not let its samples accumulate in the
         * pre-commit buffer. Only drop them when no new onset started later in
         * the same append chunk. */
        if (!s->vad_st.speech && !s->vad_st.pending) s->in_n = 0;
    }
}

/* ------------------------------------------------------------------ */
/* Conversation history and the decoder's KV cache                     */
/* ------------------------------------------------------------------ */

static int encode_text(rt_session *s, const char *text, uint32_t **ids, int64_t *n) {
    cwick_error err;
    cwick_error_clear(&err);
    if (cwick_tokenizer_encode(s->r->tok, text, ids, n, &err) != CWICK_OK) {
        CWICK_INFO(RT_TAG, "encode failed: %s", err.message);
        return -1;
    }
    return 0;
}

/* Forward `n` embedding rows. Returns 0 or -1. */
static int fwd(rt_session *s, const float *emb, int n) {
    if (n <= 0) return 0;
    if (rt_thinker_forward(s->r->thinker, emb, n) != 0) {
        CWICK_INFO(RT_TAG, "decoder forward failed at %d rows", n);
        return -1;
    }
    s->kv_pos += n;
    return 0;
}

static int fwd_text(rt_session *s, const char *text) {
    uint32_t *ids = NULL;
    int64_t n = 0;
    if (encode_text(s, text, &ids, &n) != 0) return -1;
    const int H = rt_thinker_hidden(s->r->thinker);
    float *emb = cwick_xmalloc((size_t)(n > 0 ? n : 1) * H * sizeof(float));
    for (int64_t i = 0; i < n; i++)
        rt_thinker_embed_row(s->r->thinker, (int32_t)ids[i], emb + i * H);
    int rc = fwd(s, emb, (int)n);
    free(emb);
    free(ids);
    return rc;
}

/* The user block: header, one row per audio token, then the assistant opener.
 * `audio` supplies the rows that replace the <|audio|> embeddings. */
static int fwd_user_block(rt_session *s, const float *audio, int atok) {
    uint32_t *hid = NULL, *tid = NULL;
    int64_t nh = 0, nt = 0;
    if (encode_text(s, "<|im_start|>user\n", &hid, &nh) != 0) return -1;
    /* The answer starts immediately after the assistant opener. Anything
     * appended here is read as assistant-authored text: even a few newlines
     * make the decoder emit end-of-turn as its first token, so the turn
     * "completes" with an empty answer and the talker never gets a sentence
     * to synthesise. */
    if (encode_text(s, "<|im_end|>\n<|im_start|>assistant\n",
                    &tid, &nt) != 0) { free(hid); return -1; }
    int64_t n = nh + atok + nt;
    const int H = rt_thinker_hidden(s->r->thinker);
    float *emb = cwick_xmalloc((size_t)n * H * sizeof(float));
    for (int64_t i = 0; i < nh; i++)
        rt_thinker_embed_row(s->r->thinker, (int32_t)hid[i], emb + i * H);
    for (int i = 0; i < atok; i++)
        memcpy(emb + (nh + i) * H, audio + (int64_t)i * H, (size_t)H * sizeof(float));
    for (int64_t i = 0; i < nt; i++)
        rt_thinker_embed_row(s->r->thinker, (int32_t)tid[i], emb + (nh + atok + i) * H);
    int rc = fwd(s, emb, (int)n);
    free(emb);
    free(hid);
    free(tid);
    return rc;
}

static void hist_push(rt_session *s, const float *audio, int atok, const char *reply) {
    if (s->nhist == s->hist_cap) {
        s->hist_cap = s->hist_cap ? s->hist_cap * 2 : 8;
        s->hist = cwick_xrealloc(s->hist, (size_t)s->hist_cap * sizeof(*s->hist));
    }
    hist_turn *h = &s->hist[s->nhist++];
    const int H = rt_thinker_hidden(s->r->thinker);
    h->audio = cwick_xmalloc((size_t)(atok > 0 ? atok : 1) * H * sizeof(float));
    memcpy(h->audio, audio, (size_t)atok * H * sizeof(float));
    h->atok = atok;
    h->reply = cwick_xmalloc(strlen(reply) + 1);
    memcpy(h->reply, reply, strlen(reply) + 1);
}

static void hist_free(rt_session *s) {
    for (int i = 0; i < s->nhist; i++) { free(s->hist[i].audio); free(s->hist[i].reply); }
    free(s->hist);
    s->hist = NULL;
    s->nhist = s->hist_cap = 0;
}

/* Positions one history turn occupies: header + audio + tail + reply + the
 * im_end and newline that close it. */
static int turn_positions(rt_session *s, int atok, const char *reply) {
    uint32_t *ids = NULL;
    int64_t nh = 0, nt = 0, nr = 0, nc = 0;
    int total = 0;
    if (encode_text(s, "<|im_start|>user\n", &ids, &nh) == 0) { total += (int)nh; free(ids); }
    if (encode_text(s, "<|im_end|>\n<|im_start|>assistant\n", &ids, &nt) == 0) {
        total += (int)nt; free(ids);
    }
    if (reply && *reply && encode_text(s, reply, &ids, &nr) == 0) {
        total += (int)nr; free(ids);
    }
    if (encode_text(s, "<|im_end|>\n", &ids, &nc) == 0) { total += (int)nc; free(ids); }
    return total + atok;
}

/* Rebuild the cache from the system prompt and as many recent turns as fit
 * alongside the turn about to run. Called only when the cache would overflow,
 * which on a 4096-position window is roughly fifty conversational turns. */
static int kv_rebuild(rt_session *s, int needed) {
    rt_thinker *t = s->r->thinker;
    const int cap = rt_thinker_capacity(t);
    int budget = cap - needed;
    char head[8192];
    snprintf(head, sizeof(head), "<|im_start|>system\n%s<|im_end|>\n", s->instructions);
    uint32_t *ids = NULL;
    int64_t nhead = 0;
    if (encode_text(s, head, &ids, &nhead) != 0) return -1;
    free(ids);
    budget -= (int)nhead;

    int keep = 0;
    for (int i = s->nhist - 1; i >= 0; i--) {
        int p = turn_positions(s, s->hist[i].atok, s->hist[i].reply);
        if (budget - p < 0) break;
        budget -= p;
        keep++;
    }
    int from = s->nhist - keep;
    CWICK_INFO(RT_TAG, "kv window full at %d/%d: rebuilding from system + %d/%d "
                       "recent turn(s)", s->kv_pos, cap, keep, s->nhist);
    rt_thinker_reset(t);
    s->kv_pos = 0;
    if (fwd_text(s, head) != 0) return -1;
    for (int i = from; i < s->nhist; i++) {
        if (fwd_user_block(s, s->hist[i].audio, s->hist[i].atok) != 0) return -1;
        if (fwd_text(s, s->hist[i].reply) != 0) return -1;
        if (fwd_text(s, "<|im_end|>\n") != 0) return -1;
    }
    /* The dropped turns are gone from the cache; keeping them in `hist` would
     * make the next rebuild count positions it can no longer reproduce. */
    if (from > 0) {
        for (int i = 0; i < from; i++) {
            free(s->hist[i].audio);
            free(s->hist[i].reply);
        }
        memmove(s->hist, s->hist + from, (size_t)(s->nhist - from) * sizeof(*s->hist));
        s->nhist -= from;
    }
    s->kv_valid = true;
    return 0;
}

/* ------------------------------------------------------------------ */
/* The turn                                                            */
/* ------------------------------------------------------------------ */

/* What the talker callback needs to put PCM on the wire. */
typedef struct {
    rt_session *s;
    double t_commit;
    double first_audio;         /* 0 until the first chunk */
    uint64_t samples;
    int chunks;
} pcm_sink;

/* One audio.delta is 200 ms: small enough that a browser's jitter buffer stays
 * short, large enough that the base64 and frame overhead is not per-sample. */
#define DELTA_SAMPLES (WIRE_RATE / 5)

static void emit_audio(rt_session *s, const float *pcm, int n, pcm_sink *sink) {
    int16_t *q = cwick_xmalloc((size_t)(n > 0 ? n : 1) * sizeof(int16_t));
    for (int i = 0; i < n; i++) {
        float v = pcm[i] * 32767.0f;
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        q[i] = (int16_t)(v < 0 ? v - 0.5f : v + 0.5f);
    }
    for (int off = 0; off < n; off += DELTA_SAMPLES) {
        int take = n - off;
        if (take > DELTA_SAMPLES) take = DELTA_SAMPLES;
        char *b64 = cwick_b64_encode(q + off, (size_t)take * sizeof(int16_t));
        jbuf b;
        memset(&b, 0, sizeof(b));
        ev_begin(s, &b, "response.audio.delta");
        jb_putf(&b, ",\"response_id\":");
        jb_str(&b, s->resp_id);
        jb_puts(&b, ",\"item_id\":");
        jb_str(&b, s->resp_item_id);
        jb_putf(&b, ",\"output_index\":%d,\"content_index\":0,\"delta\":",
                s->resp_output_index);
        jb_str(&b, b64);
        jb_puts(&b, "}");
        ev_send(s, &b);
        free(b64);
    }
    free(q);
    sink->samples += (uint64_t)n;
    sink->chunks++;
    if (sink->first_audio == 0.0) sink->first_audio = now_s() - sink->t_commit;
}

static int on_pcm(void *user, const float *pcm, int n, int done) {
    pcm_sink *sink = (pcm_sink *)user;
    rt_session *s = sink->s;
    if (done) return 0;
    pthread_mutex_lock(&s->mtx);
    bool cancel = s->cancel;
    pthread_mutex_unlock(&s->mtx);
    if (cancel) return 1;
    if (n > 0) emit_audio(s, pcm, n, sink);
    return 0;
}

static void emit_transcript_delta(rt_session *s, const char *piece) {
    jbuf b;
    memset(&b, 0, sizeof(b));
    ev_begin(s, &b, "response.audio_transcript.delta");
    jb_puts(&b, ",\"response_id\":");
    jb_str(&b, s->resp_id);
    jb_puts(&b, ",\"item_id\":");
    jb_str(&b, s->resp_item_id);
    jb_putf(&b, ",\"output_index\":%d,\"content_index\":0,\"delta\":",
            s->resp_output_index);
    jb_str(&b, piece);
    jb_puts(&b, "}");
    ev_send(s, &b);
}

/* Sentence boundaries the talker can be handed without waiting for the whole
 * answer. A comma only counts once the clause is long enough to stand on its
 * own, otherwise every short answer is synthesised one word at a time and the
 * voice prompt prefill is paid per word. */
static int sentence_end(const char *text, int len) {
    for (int i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c == '\n') return i + 1;
        /* UTF-8 lead bytes of the CJK and full-width terminators. */
        if (c == 0xE3 && i + 2 < len &&
            (unsigned char)text[i + 1] == 0x80 &&
            (unsigned char)text[i + 2] == 0x82)          /* 。 */
            return i + 3;
        if (c == 0xEF && i + 2 < len &&
            (unsigned char)text[i + 1] == 0xBC &&
            ((unsigned char)text[i + 2] == 0x81 ||   /* ！ */
             (unsigned char)text[i + 2] == 0x9F ||   /* ？ */
             (unsigned char)text[i + 2] == 0x9B))    /* ； */
            return i + 3;
        if (c == '!' || c == '?' || c == ';') return i + 1;
        if (c == 0xEF && i + 2 < len && (unsigned char)text[i + 1] == 0xBC &&
            (unsigned char)text[i + 2] == 0x8C && i >= 24)   /* ， after ~8 chars */
            return i + 3;
    }
    return 0;
}

typedef struct {
    rt_session *s;
    rt_voice *voice;
    pcm_sink sink;
    /* Text decoded so far, and the part already handed to the talker. */
    char *text;
    int text_len, text_cap;
    int spoken_len;
    /* Decoder position after the last fully synthesised sentence, so a
     * barge-in keeps exactly what the user heard. */
    int kv_spoken;
    double talker_s;
    uint64_t speech_tokens;
    bool failed;
} turn_ctx;

static void tc_append(turn_ctx *tc, const char *piece) {
    int n = (int)strlen(piece);
    if (tc->text_len + n + 1 > tc->text_cap) {
        tc->text_cap = (tc->text_len + n + 1) * 2;
        tc->text = cwick_xrealloc(tc->text, (size_t)tc->text_cap);
    }
    memcpy(tc->text + tc->text_len, piece, (size_t)n);
    tc->text_len += n;
    tc->text[tc->text_len] = '\0';
}

/* Hand one sentence to the talker and stream what it renders. */
static void tc_say(turn_ctx *tc, const char *sentence) {
    rt_session *s = tc->s;
    double t0 = now_s();
    rt_talker_stats st;
    memset(&st, 0, sizeof(st));
    CWICK_INFO(RT_TAG, "talker: synthesising %zu bytes", strlen(sentence));
    int rc = rt_talker_say(s->r->talker, tc->voice, sentence, on_pcm, &tc->sink, &st);
    tc->talker_s += now_s() - t0;
    tc->speech_tokens += (uint64_t)st.speech_tokens;
    if (rc < 0) {
        tc->failed = true;
        CWICK_INFO(RT_TAG, "talker failed on this sentence");
    }
}

/* Synthesise every complete sentence in the pending text. `kv_now` is the
 * decoder position after the last token handed over, which is what a barge-in
 * truncates back to: everything past it was decoded but never spoken. */
static void tc_flush_sentences(turn_ctx *tc, bool final, int kv_now) {
    rt_session *s = tc->s;
    for (;;) {
        pthread_mutex_lock(&s->mtx);
        bool cancel = s->cancel;
        pthread_mutex_unlock(&s->mtx);
        if (cancel || tc->failed) return;
        int pend = tc->text_len - tc->spoken_len;
        if (pend <= 0) return;
        const char *p = tc->text + tc->spoken_len;
        int cut = final ? pend : sentence_end(p, pend);
        if (cut <= 0) return;
        char *sentence = cwick_xmalloc((size_t)cut + 1);
        memcpy(sentence, p, (size_t)cut);
        sentence[cut] = '\0';
        tc_say(tc, sentence);
        free(sentence);
        tc->spoken_len += cut;
        tc->kv_spoken = kv_now;
    }
}

static void response_done(rt_session *s, const char *status, turn_ctx *tc,
                          int prompt_tokens, int reply_tokens, int audio_tokens,
                          double ear_s, double prefill_s, double decode_s,
                          double turn_s, double audio_in_s) {
    jbuf b;
    memset(&b, 0, sizeof(b));
    ev_begin(s, &b, "response.done");
    jb_puts(&b, ",\"response\":{\"id\":");
    jb_str(&b, s->resp_id);
    jb_puts(&b, ",\"object\":\"realtime.response\",\"status\":");
    jb_str(&b, status);
    jb_puts(&b, ",\"status_details\":null,\"output\":[{\"id\":");
    jb_str(&b, s->resp_item_id);
    jb_puts(&b, ",\"object\":\"realtime.item\",\"type\":\"message\",\"role\":\"assistant\""
               ",\"content\":[{\"type\":\"audio\",\"transcript\":");
    jb_str(&b, tc->text ? tc->text : "");
    jb_puts(&b, "}]}],\"usage\":{\"total_tokens\":");
    jb_putf(&b, "%d,\"input_tokens\":%d,\"output_tokens\":%d"
                ",\"input_token_details\":{\"text_tokens\":%d,\"audio_tokens\":%d}"
                ",\"output_token_details\":{\"text_tokens\":%d,\"audio_tokens\":%" PRIu64 "}}",
            prompt_tokens + reply_tokens, prompt_tokens, reply_tokens,
            prompt_tokens - audio_tokens, audio_tokens, reply_tokens,
            tc->speech_tokens);
    /* The full-chain breakdown, next to the answer it belongs to. This is the
     * same data the stdout stage lines carry, in a form a client can chart. */
    jb_putf(&b, ",\"cwick\":{\"ear_s\":%.4f,\"prefill_s\":%.4f,\"decode_s\":%.4f"
                ",\"talker_s\":%.4f,\"turn_s\":%.4f,\"first_audio_s\":%.4f"
                ",\"audio_in_s\":%.3f,\"audio_out_s\":%.3f"
                ",\"audio_tokens\":%d,\"prompt_tokens\":%d,\"reply_tokens\":%d"
                ",\"speech_tokens\":%" PRIu64 ",\"pcm_chunks\":%d}}}",
            ear_s, prefill_s, decode_s, tc->talker_s, turn_s, tc->sink.first_audio,
            audio_in_s, (double)tc->sink.samples / WIRE_RATE,
            audio_tokens, prompt_tokens, reply_tokens, tc->speech_tokens,
            tc->sink.chunks);
    ev_send(s, &b);
}

/* Run one committed clip end to end. Called on the session worker with the
 * pipeline mutex held, so the decoder and the talker are exclusively ours. */
static void run_turn(rt_session *s, clip *c) {
    rt_realtime *r = s->r;
    double t_commit = now_s();
    /* Take ownership of the clip immediately: the finish path below is reached
     * from several places and must not have to know whether it still exists. */
    float *clip_pcm = c->pcm;
    int clip_n = c->n;
    free(c);
    CWICK_INFO(RT_TAG, "turn start: %.3f s of 16 kHz audio (%d samples)",
               (double)clip_n / EAR_RATE, clip_n);

    /* Snapshot the session settings the turn runs under. */
    pthread_mutex_lock(&s->mtx);
    char voice[64], instructions[4096];
    snprintf(voice, sizeof(voice), "%s", s->voice);
    snprintf(instructions, sizeof(instructions), "%s", s->instructions);
    float temperature = s->temperature, top_p = s->top_p;
    int top_k = s->top_k, max_reply = s->max_reply;
    bool want_audio = s->want_audio;
    new_id(s->resp_id, sizeof(s->resp_id), "resp", s->resp_seq++);
    new_id(s->resp_item_id, sizeof(s->resp_item_id), "item", s->item_seq++);
    s->resp_output_index = 0;
    s->cancel = false;
    pthread_mutex_unlock(&s->mtx);

    /* ---- response.created ------------------------------------------ */
    {
        jbuf b;
        memset(&b, 0, sizeof(b));
        ev_begin(s, &b, "response.created");
        jb_puts(&b, ",\"response\":{\"id\":");
        jb_str(&b, s->resp_id);
        jb_puts(&b, ",\"object\":\"realtime.response\",\"status\":\"in_progress\""
                   ",\"output\":[],\"usage\":null}}");
        ev_send(s, &b);
    }
    {
        jbuf b;
        memset(&b, 0, sizeof(b));
        ev_begin(s, &b, "response.output_item.added");
        jb_putf(&b, ",\"response_id\":");
        jb_str(&b, s->resp_id);
        jb_putf(&b, ",\"output_index\":0,\"item\":{\"id\":");
        jb_str(&b, s->resp_item_id);
        jb_puts(&b, ",\"object\":\"realtime.item\",\"type\":\"message\""
                   ",\"role\":\"assistant\",\"content\":[]}}");
        ev_send(s, &b);
    }
    {
        jbuf b;
        memset(&b, 0, sizeof(b));
        ev_begin(s, &b, "response.content_part.added");
        jb_puts(&b, ",\"response_id\":");
        jb_str(&b, s->resp_id);
        jb_puts(&b, ",\"item_id\":");
        jb_str(&b, s->resp_item_id);
        jb_puts(&b, ",\"output_index\":0,\"content_index\":0"
                   ",\"part\":{\"type\":\"audio\",\"transcript\":\"\"}}");
        ev_send(s, &b);
    }

    turn_ctx tc;
    memset(&tc, 0, sizeof(tc));
    tc.s = s;
    tc.sink.s = s;
    tc.sink.t_commit = t_commit;
    tc.text = cwick_xmalloc(256);
    tc.text_cap = 256;
    tc.text[0] = '\0';

    rt_realtime_stats d;
    memset(&d, 0, sizeof(d));
    d.turns = 1;
    d.in_samples = (uint64_t)clip_n;
    d.last_audio_in_s = (double)clip_n / EAR_RATE;

    const char *status = "completed";
    int prompt_tokens = 0, reply_tokens = 0, audio_tokens = 0;
    double ear_s = 0, prefill_s = 0, decode_s = 0;

    /* ---- ear -------------------------------------------------------- */
    double t0 = now_s();
    int padded = (clip_n + EAR_PAD - 1) / EAR_PAD * EAR_PAD;
    float *buf = cwick_xmalloc((size_t)padded * sizeof(float));
    memcpy(buf, clip_pcm, (size_t)clip_n * sizeof(float));
    memset(buf + clip_n, 0, (size_t)(padded - clip_n) * sizeof(float));
    float *avec = NULL;
    int atok = 0;
    if (rt_ear_embed(r->ear, buf, padded, &avec, &atok) != 0 || atok <= 0) {
        ear_s = now_s() - t0;
        CWICK_INFO(RT_TAG, "ear produced no audio tokens from %d samples", clip_n);
        ev_error(s, "invalid_request_error", "empty_audio",
                 "the committed audio produced no tokens; speak for at least "
                 "half a second", NULL);
        status = "failed";
        goto finish;
    }
    ear_s = now_s() - t0;
    audio_tokens = atok;
    CWICK_INFO(RT_TAG, "ear: %d mel frames -> %d audio tokens in %.3f s (%.1fx realtime)",
               rt_ear_frames_for_samples(padded), atok, ear_s,
               ((double)padded / EAR_RATE) / (ear_s > 0 ? ear_s : 1));

    /* ---- voice ------------------------------------------------------ */
    tc.voice = NULL;
    for (int i = 0; i < r->nvoice; i++) {
        if (strcmp(r->voice[i].name, voice) != 0) continue;
        if (!r->voice[i].loaded) {
            char path[CWICK_MODEL_ROOT_MAX * 2];
            snprintf(path, sizeof(path), "%s/%s.pt", r->voices_dir, voice);
            cwick_error ve;
            cwick_error_clear(&ve);
            double tv = now_s();
            if (rt_voice_load(path, &r->voice[i].v, &ve) != 0) {
                CWICK_INFO(RT_TAG, "voice %s failed: %s", voice, ve.message);
                break;
            }
            r->voice[i].loaded = true;
            CWICK_INFO(RT_TAG, "voice %s loaded: %d prompt tokens, %d mel frames (%.2f s)",
                       voice, r->voice[i].v.n_token, r->voice[i].v.n_feat, now_s() - tv);
        }
        tc.voice = &r->voice[i].v;
        break;
    }
    if (!tc.voice && want_audio) {
        ev_error(s, "invalid_request_error", "unknown_voice",
                 "the requested voice is not in this checkpoint", "session.voice");
        status = "failed";
        goto finish;
    }

    /* ---- prefill ---------------------------------------------------- */
    t0 = now_s();
    {
        char head[8192];
        snprintf(head, sizeof(head), "<|im_start|>system\n%s<|im_end|>\n", instructions);
        int need = turn_positions(s, atok, "") + max_reply;
        int pos0 = s->kv_valid ? s->kv_pos : 0;
        if (!s->kv_valid) {
            rt_thinker_reset(r->thinker);
            s->kv_pos = 0;
            if (fwd_text(s, head) != 0) { status = "failed"; goto finish; }
            s->kv_valid = true;
        } else if (s->kv_pos + need > rt_thinker_capacity(r->thinker)) {
            if (kv_rebuild(s, need) != 0) { status = "failed"; goto finish; }
            pos0 = 0;
        }
        if (fwd_user_block(s, avec, atok) != 0) { status = "failed"; goto finish; }
        prompt_tokens = s->kv_pos - pos0;
    }
    prefill_s = now_s() - t0;
    CWICK_INFO(RT_TAG, "prefill: %d positions (%d audio) in %.3f s (%.2f tok/s); "
                       "kv at %d/%d",
               prompt_tokens, atok, prefill_s,
               prefill_s > 0 ? (double)prompt_tokens / prefill_s : 0.0,
               s->kv_pos, rt_thinker_capacity(r->thinker));

    /* ---- decode ----------------------------------------------------- */
    cwick_sampler_cfg cfg;
    cwick_sampler_cfg_default(&cfg);
    cfg.temperature = temperature;
    cfg.top_p = top_p;
    cfg.top_k = top_k;
    cwick_sampler samp;
    cwick_sampler_init(&samp, &cfg, rt_thinker_vocab(r->thinker),
                       (uint64_t)(cwick_prof_ns() & 0xffffffffu));
    const int H = rt_thinker_hidden(r->thinker);
    float *one = cwick_xmalloc((size_t)H * sizeof(float));
    t0 = now_s();
    tc.kv_spoken = rt_thinker_pos(r->thinker);
    for (int step = 0; step < max_reply; step++) {
        pthread_mutex_lock(&s->mtx);
        bool cancel = s->cancel;
        pthread_mutex_unlock(&s->mtx);
        if (cancel) { status = "cancelled"; break; }
        /* pick() masks the row in place; the decoder recomputes it on every
         * forward, so the reused buffer is exactly the scratch it wants. */
        float *logits = (float *)rt_thinker_logits(r->thinker);
        uint32_t id = cwick_sampler_pick(&samp, logits, rt_thinker_vocab(r->thinker));
        cwick_sampler_accept(&samp, id);
        if (id == IM_END || id == ENDOFTEXT) break;
        char *piece = NULL;
        cwick_error de;
        cwick_error_clear(&de);
        if (cwick_tokenizer_decode(r->tok, &id, 1, &piece, &de) == CWICK_OK && piece) {
            tc_append(&tc, piece);
            emit_transcript_delta(s, piece);
            free(piece);
        }
        reply_tokens++;
        rt_thinker_embed_row(r->thinker, (int32_t)id, one);
        if (rt_thinker_forward(r->thinker, one, 1) != 0) { status = "failed"; break; }
        s->kv_pos++;
        /* Synthesise as soon as a sentence is complete: the talker runs at
         * about realtime, so waiting for the whole answer would add its entire
         * duration to the time the user hears nothing. This runs after the
         * forward so kv_spoken names a position that really holds the sentence. */
        if (want_audio) tc_flush_sentences(&tc, false, s->kv_pos);
    }
    decode_s = now_s() - t0;
    free(one);
    cwick_sampler_free(&samp);
    if (want_audio) tc_flush_sentences(&tc, true, s->kv_pos);
    CWICK_INFO(RT_TAG, "decode: %d tokens in %.3f s (%.2f tok/s)", reply_tokens,
               decode_s, decode_s > 0 ? (double)reply_tokens / decode_s : 0.0);

    /* Close the assistant turn in the cache. On a barge-in only the part that
     * was actually spoken is kept, so the next turn does not continue a
     * sentence the user never heard. */
    if (strcmp(status, "cancelled") == 0) {
        rt_thinker_truncate(r->thinker, tc.kv_spoken);
        s->kv_pos = tc.kv_spoken;
        tc.text_len = tc.spoken_len;
        tc.text[tc.text_len] = '\0';
    }
    if (fwd_text(s, "<|im_end|>\n") != 0) status = "failed";

    pthread_mutex_lock(&s->mtx);
    hist_push(s, avec, atok, tc.text);
    pthread_mutex_unlock(&s->mtx);

finish:
    free(avec);
    free(buf);
    free(clip_pcm);

    double turn_s = now_s() - t_commit;
    if (strcmp(status, "cancelled") == 0) d.cancelled = 1;
    d.ear_s = ear_s;
    d.prefill_s = prefill_s;
    d.decode_s = decode_s;
    d.talker_s = tc.talker_s;
    d.turn_s = turn_s;
    d.first_audio_s = tc.sink.first_audio;
    d.out_samples = tc.sink.samples;
    d.audio_tokens = (uint64_t)audio_tokens;
    d.prompt_tokens = (uint64_t)prompt_tokens;
    d.reply_tokens = (uint64_t)reply_tokens;
    d.speech_tokens = tc.speech_tokens;
    d.last_ear_s = ear_s;
    d.last_prefill_s = prefill_s;
    d.last_decode_s = decode_s;
    d.last_talker_s = tc.talker_s;
    d.last_first_audio_s = tc.sink.first_audio;
    d.last_turn_s = turn_s;
    d.last_prompt_tokens = prompt_tokens;
    d.last_reply_tokens = reply_tokens;
    d.last_audio_tokens = audio_tokens;
    d.last_audio_out_s = (double)tc.sink.samples / WIRE_RATE;
    rt_stat_add(r, &d);

    /* ---- the closing events ---------------------------------------- */
    {
        jbuf b;
        memset(&b, 0, sizeof(b));
        ev_begin(s, &b, "response.audio_transcript.done");
        jb_puts(&b, ",\"response_id\":");
        jb_str(&b, s->resp_id);
        jb_puts(&b, ",\"item_id\":");
        jb_str(&b, s->resp_item_id);
        jb_putf(&b, ",\"output_index\":0,\"content_index\":0,\"transcript\":");
        jb_str(&b, tc.text);
        jb_puts(&b, "}");
        ev_send(s, &b);
    }
    if (want_audio) {
        jbuf b;
        memset(&b, 0, sizeof(b));
        ev_begin(s, &b, "response.audio.done");
        jb_puts(&b, ",\"response_id\":");
        jb_str(&b, s->resp_id);
        jb_puts(&b, ",\"item_id\":");
        jb_str(&b, s->resp_item_id);
        jb_puts(&b, ",\"output_index\":0,\"content_index\":0}");
        ev_send(s, &b);
    }
    {
        jbuf b;
        memset(&b, 0, sizeof(b));
        ev_begin(s, &b, "response.content_part.done");
        jb_puts(&b, ",\"response_id\":");
        jb_str(&b, s->resp_id);
        jb_puts(&b, ",\"item_id\":");
        jb_str(&b, s->resp_item_id);
        jb_puts(&b, ",\"output_index\":0,\"content_index\":0"
                   ",\"part\":{\"type\":\"audio\",\"transcript\":");
        jb_str(&b, tc.text);
        jb_puts(&b, "}}");
        ev_send(s, &b);
    }
    {
        jbuf b;
        memset(&b, 0, sizeof(b));
        ev_begin(s, &b, "response.output_item.done");
        jb_puts(&b, ",\"response_id\":");
        jb_str(&b, s->resp_id);
        jb_putf(&b, ",\"output_index\":0,\"item\":{\"id\":");
        jb_str(&b, s->resp_item_id);
        jb_puts(&b, ",\"object\":\"realtime.item\",\"type\":\"message\""
                   ",\"role\":\"assistant\",\"content\":[{\"type\":\"audio\",\"transcript\":");
        jb_str(&b, tc.text);
        jb_puts(&b, "}]}}");
        ev_send(s, &b);
    }
    response_done(s, status, &tc, prompt_tokens, reply_tokens, audio_tokens,
                  ear_s, prefill_s, decode_s, turn_s, (double)clip_n / EAR_RATE);

    CWICK_INFO(RT_TAG,
               "turn %s in %.3f s | ear %.3f | prefill %.3f (%d tok) | decode %.3f "
               "(%d tok, %.2f tok/s) | talker %.3f (%.2f s audio, %d chunks) | "
               "first audio %.3f s | in %.2f s out %.2f s",
               status, turn_s, ear_s, prefill_s, prompt_tokens, decode_s,
               reply_tokens, decode_s > 0 ? (double)reply_tokens / decode_s : 0.0,
               tc.talker_s, (double)tc.sink.samples / WIRE_RATE, tc.sink.chunks,
               tc.sink.first_audio, (double)clip_n / EAR_RATE,
               (double)tc.sink.samples / WIRE_RATE);

    free(tc.text);
}

/* ------------------------------------------------------------------ */
/* The session worker                                                  */
/* ------------------------------------------------------------------ */

static void *worker_main(void *arg) {
    rt_session *s = (rt_session *)arg;
    cwick_log_set_request(s->rid);
    for (;;) {
        pthread_mutex_lock(&s->mtx);
        while (!s->head && !s->stop) pthread_cond_wait(&s->cv, &s->mtx);
        if (s->stop) { pthread_mutex_unlock(&s->mtx); break; }
        clip *c = s->head;
        s->head = c->next;
        if (!s->head) s->tail = NULL;
        s->turn_active = true;
        s->cancel = false;
        pthread_mutex_unlock(&s->mtx);

        /* One decoder, one talker: a second session waits here. The wait is
         * logged because a client whose answer is late deserves to know why. */
        double tw = now_s();
        pthread_mutex_lock(&s->r->pipe_mtx);
        double waited = now_s() - tw;
        if (waited > 0.05)
            CWICK_INFO(RT_TAG, "waited %.3f s for the pipeline (another session holds it)",
                       waited);
        run_turn(s, c);
        pthread_mutex_unlock(&s->r->pipe_mtx);

        pthread_mutex_lock(&s->mtx);
        s->turn_active = false;
        pthread_mutex_unlock(&s->mtx);
    }
    cwick_log_set_request(0);
    return NULL;
}

/* Queue a committed clip. Takes ownership of `pcm`. */
static void queue_clip(rt_session *s, float *pcm, int n, int start_ms) {
    clip *c = cwick_xmalloc(sizeof(*c));
    c->pcm = pcm;
    c->n = n;
    c->audio_start_ms = start_ms;
    c->next = NULL;
    pthread_mutex_lock(&s->mtx);
    if (s->tail) s->tail->next = c; else s->head = c;
    s->tail = c;
    pthread_cond_signal(&s->cv);
    pthread_mutex_unlock(&s->mtx);
}

/* ------------------------------------------------------------------ */
/* Client events                                                       */
/* ------------------------------------------------------------------ */

/* Runs with s->mtx held, so it must not emit events: ev_begin takes the same
 * mutex. Anything the client needs to be told about comes back through
 * `tools_requested` and is sent by the caller after unlocking. */
static void apply_session_update(rt_session *s, json_value *sess,
                                 bool *tools_requested) {
    if (tools_requested) *tools_requested = false;
    if (!sess || sess->kind != JSON_OBJECT) return;
    const char *v;
    if ((v = cwick_json_str(cwick_json_get(sess, "voice"))) && *v)
        snprintf(s->voice, sizeof(s->voice), "%s", v);
    if ((v = cwick_json_str(cwick_json_get(sess, "instructions"))) && *v)
        snprintf(s->instructions, sizeof(s->instructions), "%s", v);
    json_value *m = cwick_json_get(sess, "modalities");
    if (m && m->kind == JSON_ARRAY) {
        s->want_text = s->want_audio = false;
        for (size_t i = 0; i < cwick_json_array_len(m); i++) {
            const char *x = cwick_json_str(cwick_json_array_at(m, i));
            if (!x) continue;
            if (strcmp(x, "text") == 0) s->want_text = true;
            if (strcmp(x, "audio") == 0) s->want_audio = true;
        }
    }
    json_value *t = cwick_json_get(sess, "temperature");
    if (t) s->temperature = (float)cwick_json_num(t, s->temperature);
    json_value *mr = cwick_json_get(sess, "max_response_output_tokens");
    if (mr) {
        if (mr->kind == JSON_STRING && strcmp(cwick_json_str(mr), "inf") == 0)
            s->max_reply = MAX_REPLY_DEFAULT;
        else {
            int64_t x = cwick_json_int(mr, s->max_reply);
            if (x > 0 && x <= 4096) s->max_reply = (int)x;
        }
    }
    /* The wire rate is 24 kHz per the spec; a client that captures at the ear's
     * own rate can say so and skip a resample. */
    json_value *ir = cwick_json_get(sess, "input_audio_sample_rate");
    if (ir) {
        int64_t x = cwick_json_int(ir, WIRE_RATE);
        if (x == 16000 || x == 24000 || x == 48000 || x == 44100) {
            if (s->input_rate != (int)x) rt_resample_reset(&s->rs);
            s->input_rate = (int)x;
        }
    }
    json_value *td = cwick_json_get(sess, "turn_detection");
    if (td) {
        if (td->kind == JSON_NULL) {
            s->vad = false;
        } else if (td->kind == JSON_OBJECT) {
            const char *ty = cwick_json_str(cwick_json_get(td, "type"));
            s->vad = !ty || strcmp(ty, "server_vad") == 0;
            json_value *x;
            if ((x = cwick_json_get(td, "threshold")))
                s->vad_threshold = (float)cwick_json_num(x, s->vad_threshold);
            if ((x = cwick_json_get(td, "prefix_padding_ms")))
                s->prefix_padding_ms = (int)cwick_json_int(x, s->prefix_padding_ms);
            if ((x = cwick_json_get(td, "silence_duration_ms")))
                s->silence_duration_ms = (int)cwick_json_int(x, s->silence_duration_ms);
            if ((x = cwick_json_get(td, "create_response")))
                s->vad_create_response = cwick_json_bool(x, s->vad_create_response);
            if ((x = cwick_json_get(td, "interrupt_response")))
                s->vad_interrupt_response = cwick_json_bool(x, s->vad_interrupt_response);
            if (s->vad_threshold < 0.0f) s->vad_threshold = 0.0f;
            if (s->vad_threshold > 1.0f) s->vad_threshold = 1.0f;
            if (s->prefix_padding_ms < 0) s->prefix_padding_ms = 0;
            if (s->silence_duration_ms < 20) s->silence_duration_ms = 20;
        }
    }
    json_value *tools = cwick_json_get(sess, "tools");
    if (tools && tools->kind == JSON_ARRAY && cwick_json_array_len(tools) > 0 &&
        tools_requested)
        *tools_requested = true;
}

/* Commit the input buffer: hand it to the worker and tell the client the item
 * it became. `pcm` is consumed here. */
static void do_commit(rt_session *s, float *pcm, int n, int start_ms,
                      bool create_response) {
    char item_id[48];
    pthread_mutex_lock(&s->mtx);
    new_id(item_id, sizeof(item_id), "item", s->item_seq++);
    pthread_mutex_unlock(&s->mtx);

    CWICK_INFO(RT_TAG, "commit: %.3f s of audio -> item %s%s",
               (double)n / EAR_RATE, item_id,
               create_response ? " (auto response)" : "");
    {
        jbuf b;
        memset(&b, 0, sizeof(b));
        ev_begin(s, &b, "input_audio_buffer.committed");
        jb_puts(&b, ",\"previous_item_id\":null,\"item_id\":");
        jb_str(&b, item_id);
        jb_puts(&b, "}");
        ev_send(s, &b);
    }
    {
        jbuf b;
        memset(&b, 0, sizeof(b));
        ev_begin(s, &b, "conversation.item.created");
        jb_puts(&b, ",\"previous_item_id\":null,\"item\":{\"id\":");
        jb_str(&b, item_id);
        jb_puts(&b, ",\"object\":\"realtime.item\",\"type\":\"message\",\"role\":\"user\""
                   ",\"content\":[{\"type\":\"input_audio\",\"transcript\":null}]}}");
        ev_send(s, &b);
    }
    if (create_response) queue_clip(s, pcm, n, start_ms);
    else {
        pthread_mutex_lock(&s->mtx);
        free(s->pending);
        s->pending = pcm;
        s->pending_n = n;
        s->pending_ms = start_ms;
        pthread_mutex_unlock(&s->mtx);
    }
}

/* Barge-in: stop the reply that is rendering and let the new utterance through.
 * The worker notices the flag at its next token or PCM chunk. */
static void do_interrupt(rt_session *s) {
    pthread_mutex_lock(&s->mtx);
    bool active = s->turn_active;
    if (active) s->cancel = true;
    pthread_mutex_unlock(&s->mtx);
    if (active) CWICK_INFO(RT_TAG, "barge-in: cancelling the response in flight");
}

static void handle_append(rt_session *s, json_value *root) {
    const char *b64 = cwick_json_str(cwick_json_get(root, "audio"));
    if (!b64 || !*b64) {
        ev_error(s, "invalid_request_error", "missing_audio",
                 "input_audio_buffer.append needs a base64 `audio` field", "audio");
        return;
    }
    size_t raw_len = 0;
    void *raw = cwick_b64_decode(b64, strlen(b64), &raw_len);
    if (!raw || raw_len < 2) {
        free(raw);
        ev_error(s, "invalid_request_error", "bad_audio",
                 "audio is not base64 PCM16", "audio");
        return;
    }
    int n = (int)(raw_len / 2);
    const int16_t *q = (const int16_t *)raw;
    float *f = cwick_xmalloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) f[i] = (float)q[i] / 32768.0f;
    free(raw);

    pthread_mutex_lock(&s->mtx);
    float *at = NULL;
    int at_n = rt_resample_stream(&s->rs, f, n, s->input_rate, EAR_RATE, &at);
    free(f);
    if (at_n <= 0) { free(at); pthread_mutex_unlock(&s->mtx); return; }
    in_reserve(s, at_n);
    memcpy(s->in + s->in_n, at, (size_t)at_n * sizeof(float));
    s->in_n += at_n;
    int start_ms = -1;
    bool stopped = false;
    vad_frame(s, at, at_n, &start_ms, &stopped);
    bool interrupt = s->vad_interrupt_response;
    int prefix = s->prefix_padding_ms;
    bool create = s->vad_create_response;
    int buf_n = s->in_n;
    pthread_mutex_unlock(&s->mtx);
    free(at);

    if (start_ms >= 0) {
        CWICK_INFO(RT_TAG, "vad: speech started at %d ms", start_ms);
        if (interrupt) do_interrupt(s);
        /* Keep only the padding before the onset, so the ear is not handed the
         * silence of the whole previous turn. */
        pthread_mutex_lock(&s->mtx);
        int keep = prefix * EAR_RATE / 1000;
        if (keep < buf_n) {
            int drop = buf_n - keep;
            memmove(s->in, s->in + drop, (size_t)keep * sizeof(float));
            s->in_n = keep;
        }
        pthread_mutex_unlock(&s->mtx);
        jbuf b;
        memset(&b, 0, sizeof(b));
        ev_begin(s, &b, "input_audio_buffer.speech_started");
        jb_putf(&b, ",\"audio_start_ms\":%d,\"item_id\":null}", start_ms);
        ev_send(s, &b);
    }
    if (stopped) {
        int end_ms;
        pthread_mutex_lock(&s->mtx);
        end_ms = s->vad_st.in_ms;
        pthread_mutex_unlock(&s->mtx);
        CWICK_INFO(RT_TAG, "vad: speech stopped at %d ms", end_ms);
        jbuf b;
        memset(&b, 0, sizeof(b));
        ev_begin(s, &b, "input_audio_buffer.speech_stopped");
        jb_putf(&b, ",\"audio_end_ms\":%d,\"item_id\":null}", end_ms);
        ev_send(s, &b);
        if (create) {
            pthread_mutex_lock(&s->mtx);
            float *pcm = cwick_xmalloc((size_t)(s->in_n > 0 ? s->in_n : 1) * sizeof(float));
            memcpy(pcm, s->in, (size_t)s->in_n * sizeof(float));
            int n2 = s->in_n;
            int st = s->vad_st.speech_start_ms;
            in_clear(s);
            pthread_mutex_unlock(&s->mtx);
            do_commit(s, pcm, n2, st, true);
        }
    }
}

static void handle_event(rt_session *s, const char *text, size_t len) {
    cwick_error err;
    cwick_error_clear(&err);
    json_value *root = cwick_json_parse(text, len, &err);
    if (!root) {
        ev_error(s, "invalid_request_error", "bad_json", "event is not valid JSON", NULL);
        return;
    }
    const char *type = cwick_json_str(cwick_json_get(root, "type"));
    if (!type) {
        ev_error(s, "invalid_request_error", "missing_type", "event has no `type`", "type");
        cwick_json_free(root);
        return;
    }
    CWICK_VERB(RT_TAG, "client event %s", type);

    if (strcmp(type, "session.update") == 0) {
        bool tools = false;
        char voice[64];
        bool vad;
        int max_reply;
        pthread_mutex_lock(&s->mtx);
        apply_session_update(s, cwick_json_get(root, "session"), &tools);
        snprintf(voice, sizeof(voice), "%s", s->voice);
        vad = s->vad;
        max_reply = s->max_reply;
        pthread_mutex_unlock(&s->mtx);
        CWICK_INFO(RT_TAG, "session updated: voice %s, vad %s, max_reply %d",
                   voice, vad ? "server_vad" : "off", max_reply);
        if (tools)
            ev_error(s, "invalid_request_error", "unsupported_tools",
                     "function calling is not implemented for the realtime model",
                     "session.tools");
        ev_session(s, "session.updated");
    } else if (strcmp(type, "input_audio_buffer.append") == 0) {
        handle_append(s, root);
    } else if (strcmp(type, "input_audio_buffer.commit") == 0) {
        pthread_mutex_lock(&s->mtx);
        int n = s->in_n;
        float *pcm = cwick_xmalloc((size_t)(n > 0 ? n : 1) * sizeof(float));
        memcpy(pcm, s->in, (size_t)n * sizeof(float));
        int st = s->vad_st.speech_start_ms;
        in_clear(s);
        pthread_mutex_unlock(&s->mtx);
        if (n < EAR_RATE / 10) {
            free(pcm);
            ev_error(s, "invalid_request_error", "buffer_too_small",
                     "the input buffer holds less than 100 ms of audio", NULL);
        } else {
            do_commit(s, pcm, n, st, false);
        }
    } else if (strcmp(type, "input_audio_buffer.clear") == 0) {
        pthread_mutex_lock(&s->mtx);
        in_clear(s);
        pthread_mutex_unlock(&s->mtx);
        CWICK_INFO(RT_TAG, "input buffer cleared");
        jbuf b;
        memset(&b, 0, sizeof(b));
        ev_begin(s, &b, "input_audio_buffer.cleared");
        jb_puts(&b, "}");
        ev_send(s, &b);
    } else if (strcmp(type, "response.create") == 0) {
        json_value *resp = cwick_json_get(root, "response");
        bool tools = false;
        pthread_mutex_lock(&s->mtx);
        /* A per-response override is a session update that lasts one turn;
         * applying it to the session is what the reference does too. */
        if (resp && resp->kind == JSON_OBJECT)
            apply_session_update(s, resp, &tools);
        bool active = s->turn_active;
        int n = s->in_n;
        float *pcm = NULL;
        int st = s->vad_st.speech_start_ms;
        bool committed = false;
        if (!active && s->pending) {
            /* Already announced by input_audio_buffer.commit: start the turn
             * without repeating the committed / item.created events. */
            pcm = s->pending;
            n = s->pending_n;
            st = s->pending_ms;
            s->pending = NULL;
            s->pending_n = 0;
            committed = true;
        } else if (!active && n > 0) {
            pcm = cwick_xmalloc((size_t)n * sizeof(float));
            memcpy(pcm, s->in, (size_t)n * sizeof(float));
            in_clear(s);
        }
        int nh = s->nhist;
        pthread_mutex_unlock(&s->mtx);
        if (tools)
            ev_error(s, "invalid_request_error", "unsupported_tools",
                     "function calling is not implemented for the realtime model",
                     "response.tools");
        if (active) {
            ev_error(s, "invalid_request_error", "response_in_progress",
                     "a response is already being generated; cancel it first", NULL);
            free(pcm);
        } else if (pcm && committed) {
            queue_clip(s, pcm, n, st);
        } else if (pcm) {
            do_commit(s, pcm, n, st, true);
        } else if (nh == 0) {
            ev_error(s, "invalid_request_error", "empty_conversation",
                     "response.create needs audio in the buffer or a prior turn", NULL);
        } else {
            ev_error(s, "invalid_request_error", "no_input_audio",
                     "response.create without buffered audio is not supported by "
                     "the speech model: commit audio first", NULL);
        }
    } else if (strcmp(type, "response.cancel") == 0) {
        do_interrupt(s);
    } else if (strcmp(type, "conversation.item.create") == 0) {
        json_value *item = cwick_json_get(root, "item");
        const char *role = cwick_json_str(cwick_json_get(item, "role"));
        json_value *content = cwick_json_get(item, "content");
        /* Text items become part of the instructions the next turn runs under:
         * the decoder has no separate text channel in a speech session, and
         * silently dropping a system message would change the answer. */
        if (content && content->kind == JSON_ARRAY) {
            for (size_t i = 0; i < cwick_json_array_len(content); i++) {
                json_value *part = cwick_json_array_at(content, i);
                const char *pt = cwick_json_str(cwick_json_get(part, "type"));
                const char *tx = cwick_json_str(cwick_json_get(part, "text"));
                if (!pt || !tx || !*tx) continue;
                if (strcmp(pt, "input_text") != 0 && strcmp(pt, "text") != 0) continue;
                pthread_mutex_lock(&s->mtx);
                if (role && strcmp(role, "system") == 0) {
                    snprintf(s->instructions, sizeof(s->instructions), "%s", tx);
                    s->kv_valid = false;
                } else {
                    size_t l = strlen(s->instructions);
                    snprintf(s->instructions + l, sizeof(s->instructions) - l,
                             "\n%s", tx);
                    s->kv_valid = false;
                }
                pthread_mutex_unlock(&s->mtx);
                CWICK_INFO(RT_TAG, "conversation.item.create: %s text folded into "
                                   "instructions (%zu bytes)", role ? role : "user",
                           strlen(tx));
            }
        }
        char item_id[48];
        pthread_mutex_lock(&s->mtx);
        new_id(item_id, sizeof(item_id), "item", s->item_seq++);
        pthread_mutex_unlock(&s->mtx);
        jbuf b;
        memset(&b, 0, sizeof(b));
        ev_begin(s, &b, "conversation.item.created");
        jb_puts(&b, ",\"previous_item_id\":null,\"item\":{\"id\":");
        jb_str(&b, item_id);
        jb_puts(&b, ",\"object\":\"realtime.item\",\"type\":\"message\",\"role\":");
        jb_str(&b, role ? role : "user");
        jb_puts(&b, ",\"content\":[]}}");
        ev_send(s, &b);
    } else if (strcmp(type, "conversation.item.truncate") == 0 ||
               strcmp(type, "conversation.item.delete") == 0) {
        /* Both mean "forget the assistant's last answer". Dropping the last
         * history turn and invalidating the cache is the honest version: the
         * next turn rebuilds without it. */
        pthread_mutex_lock(&s->mtx);
        if (s->nhist > 0) {
            s->nhist--;
            free(s->hist[s->nhist].audio);
            free(s->hist[s->nhist].reply);
            s->kv_valid = false;
        }
        pthread_mutex_unlock(&s->mtx);
        CWICK_INFO(RT_TAG, "%s: last turn dropped, cache will rebuild", type);
        jbuf b;
        memset(&b, 0, sizeof(b));
        ev_begin(s, &b, "conversation.item.truncated");
        jb_puts(&b, ",\"item_id\":null,\"content_index\":0,\"audio_end_ms\":0}");
        ev_send(s, &b);
    } else {
        ev_error(s, "invalid_request_error", "unknown_event",
                 "this event type is not implemented", "type");
    }
    cwick_json_free(root);
}

/* ------------------------------------------------------------------ */
/* The connection                                                      */
/* ------------------------------------------------------------------ */

static void session_init(rt_session *s, rt_realtime *r, int fd, uint64_t rid) {
    memset(s, 0, sizeof(*s));
    s->r = r;
    s->fd = fd;
    s->rid = rid;
    cwick_ws_init(&s->ws, fd);
    pthread_mutex_init(&s->send_mtx, NULL);
    pthread_mutex_init(&s->mtx, NULL);
    pthread_cond_init(&s->cv, NULL);
    snprintf(s->id, sizeof(s->id), "sess_%016" PRIx64, cwick_prof_ns());
    snprintf(s->model, sizeof(s->model), "%s", r->model);
    snprintf(s->voice, sizeof(s->voice), "%s", r->default_voice);
    snprintf(s->instructions, sizeof(s->instructions), "%s", DEFAULT_INSTRUCTIONS);
    s->temperature = 0.7f;
    s->top_p = 0.8f;
    s->top_k = 20;
    s->max_reply = MAX_REPLY_DEFAULT;
    s->input_rate = WIRE_RATE;
    s->want_text = true;
    s->want_audio = true;
    s->vad = true;
    s->vad_threshold = 0.5f;
    s->prefix_padding_ms = 300;
    s->silence_duration_ms = 200;
    s->vad_create_response = true;
    s->vad_interrupt_response = true;
    const char *legacy = getenv("CWICK_RT_VAD_LEGACY");
    s->legacy_vad = legacy && legacy[0] == '1';
    rt_vad_state_init(&s->vad_st);
}

static void session_destroy(rt_session *s) {
    pthread_mutex_lock(&s->mtx);
    s->stop = true;
    /* A turn in flight checks `cancel`, not `stop`: without this a shutdown
     * would wait out a whole 15 s answer before the worker noticed. */
    s->cancel = true;
    pthread_cond_broadcast(&s->cv);
    pthread_mutex_unlock(&s->mtx);
    if (s->worker_started) pthread_join(s->worker, NULL);
    while (s->head) {
        clip *c = s->head;
        s->head = c->next;
        free(c->pcm);
        free(c);
    }
    hist_free(s);
    free(s->pending);
    free(s->in);
    rt_resample_free(&s->rs);
    cwick_ws_free(&s->ws);
    pthread_mutex_destroy(&s->send_mtx);
    pthread_mutex_destroy(&s->mtx);
    pthread_cond_destroy(&s->cv);
}

/* The `model` query parameter, if the client sent one. */
static bool query_model(const char *query, char *out, size_t cap) {
    out[0] = '\0';
    if (!query) return false;
    const char *p = strstr(query, "model=");
    if (!p) return false;
    p += 6;
    size_t n = strcspn(p, "& \t");
    if (n == 0 || n >= cap) return false;
    memcpy(out, p, n);
    out[n] = '\0';
    return true;
}

int rt_realtime_serve(rt_realtime *r, int fd, uint64_t rid, const char *query,
                      const volatile int *stop) {
    /* A realtime reader must notice a shutdown and a barge-in promptly; the
     * 2 s HTTP timeout would leave a socket deaf for two seconds at a time. */
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 200 * 1000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    char want[CWICK_MODEL_ID_MAX];
    if (query_model(query, want, sizeof(want)) &&
        strcasecmp(want, r->model) != 0 &&
        strcasecmp(want, "sophon_realtime") != 0) {
        CWICK_INFO(RT_TAG, "refusing session: model '%s' is not resident ('%s')",
                   want, r->model);
        cwick_ws ws;
        cwick_ws_init(&ws, fd);
        char msg[512];
        snprintf(msg, sizeof(msg),
                 "{\"event_id\":\"evt_0\",\"type\":\"error\",\"error\":{\"type\":"
                 "\"invalid_request_error\",\"code\":\"model_not_found\",\"message\":"
                 "\"model '%s' is not resident; this server holds '%s'\","
                 "\"param\":\"model\",\"event_id\":null}}", want, r->model);
        cwick_ws_send_text(&ws, msg);
        cwick_ws_close(&ws, 1008, "wrong model");
        cwick_ws_free(&ws);
        return -1;
    }

    rt_session *s = cwick_xmalloc(sizeof(*s));
    session_init(s, r, fd, rid);

    pthread_mutex_lock(&r->pipe_mtx);
    if (r->sessions >= MAX_SESSIONS) {
        pthread_mutex_unlock(&r->pipe_mtx);
        CWICK_INFO(RT_TAG, "refusing session: %d already connected", r->sessions);
        ev_error(s, "server_error", "too_many_sessions",
                 "the realtime pipeline is at its session limit", NULL);
        cwick_ws_close(&s->ws, 1013, "try later");
        session_destroy(s);
        free(s);
        return -1;
    }
    r->sessions++;
    pthread_mutex_unlock(&r->pipe_mtx);
    pthread_mutex_lock(&r->stat_mtx);
    r->st.sessions++;
    pthread_mutex_unlock(&r->stat_mtx);
    cwick_metrics_rt_session();

    CWICK_INFO(RT_TAG, "session %s open: model %s, voice %s, %d Hz in / %d Hz out",
               s->id, s->model, s->voice, s->input_rate, WIRE_RATE);

    int rc = 0;
    if (pthread_create(&s->worker, NULL, worker_main, s) != 0) {
        CWICK_INFO(RT_TAG, "cannot start the session worker");
        ev_error(s, "server_error", "worker_failed", "cannot start the turn worker", NULL);
        cwick_ws_close(&s->ws, 1011, "worker failed");
        rc = -1;
        goto out;
    }
    s->worker_started = true;

    {
        jbuf b;
        memset(&b, 0, sizeof(b));
        ev_begin(s, &b, "conversation.created");
        jb_puts(&b, ",\"conversation\":{\"id\":\"conv_0\",\"object\":\"realtime.conversation\"}}");
        ev_send(s, &b);
    }
    ev_session(s, "session.created");

    for (;;) {
        if (stop && *stop) { CWICK_INFO(RT_TAG, "session closing: server shutting down"); break; }
        unsigned char *msg = NULL;
        size_t mlen = 0;
        int op = cwick_ws_read(&s->ws, &msg, &mlen);
        if (op == -2) continue;                 /* read timeout: loop and recheck */
        if (op == 0) { CWICK_INFO(RT_TAG, "session closed by peer"); break; }
        if (op < 0) {
            CWICK_INFO(RT_TAG, "session socket failed");
            rc = -1;
            break;
        }
        if (op == CWICK_WS_TEXT) {
            handle_event(s, (const char *)msg, mlen);
        } else {
            /* Binary frames are not part of the Realtime protocol; audio rides
             * in base64 inside a JSON append. */
            ev_error(s, "invalid_request_error", "binary_unsupported",
                     "send audio as base64 in input_audio_buffer.append", NULL);
        }
        free(msg);
        if (s->ws.failed || s->ws.peer_closed) break;
    }

    cwick_ws_close(&s->ws, 1000, "session ended");
out:
    CWICK_INFO(RT_TAG, "session %s ended", s->id);
    session_destroy(s);
    pthread_mutex_lock(&r->pipe_mtx);
    r->sessions--;
    pthread_mutex_unlock(&r->pipe_mtx);
    free(s);
    return rc;
}