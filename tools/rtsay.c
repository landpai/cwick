/* rtsay — text in, 24 kHz WAV out, through the whole talker: the 0.5B speech
 * LM, the distilled flow decoder and the HiFT vocoder. Chunks are collected as
 * they arrive so the first-audio latency is the real streaming one.
 *
 *   tools/rtsay <talker_dir> <voice.pt> <text> <out.wav>
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cwick.h"
#include "rt_talker.h"
#include "util.h"

typedef struct {
    float *pcm;
    int n, cap;
    double t0, first;
    int chunks;
} sink;

static int on_pcm(void *user, const float *pcm, int n, int done) {
    sink *s = (sink *)user;
    if (done) return 0;
    double now = (double)cwick_prof_ns() / 1e9;
    if (s->first == 0.0) s->first = now - s->t0;
    s->chunks++;
    printf("  chunk %d: %d samples (%.3f s audio) at +%.3f s\n",
           s->chunks, n, n / 24000.0, now - s->t0);
    if (s->n + n > s->cap) {
        s->cap = (s->n + n) * 2 + 4096;
        s->pcm = (float *)cwick_xrealloc(s->pcm, (size_t)s->cap * sizeof(float));
    }
    memcpy(s->pcm + s->n, pcm, (size_t)n * sizeof(float));
    s->n += n;
    return 0;
}

static void wav_write(const char *path, const float *pcm, int n, int rate) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return; }
    int bytes = n * 2;
    unsigned char h[44] = {0};
    memcpy(h, "RIFF", 4);
    int riff = 36 + bytes;
    memcpy(h + 4, &riff, 4);
    memcpy(h + 8, "WAVEfmt ", 8);
    int sixteen = 16; memcpy(h + 16, &sixteen, 4);
    short one = 1, ch = 1, bits = 16;
    memcpy(h + 20, &one, 2); memcpy(h + 22, &ch, 2);
    memcpy(h + 24, &rate, 4);
    int br = rate * 2; memcpy(h + 28, &br, 4);
    short align = 2; memcpy(h + 32, &align, 2); memcpy(h + 34, &bits, 2);
    memcpy(h + 36, "data", 4); memcpy(h + 40, &bytes, 4);
    fwrite(h, 1, 44, f);
    int16_t *s = (int16_t *)malloc((size_t)n * 2);
    for (int i = 0; i < n; i++) {
        float v = pcm[i] * 32767.0f;
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        s[i] = (int16_t)(v < 0 ? v - 0.5f : v + 0.5f);
    }
    fwrite(s, 2, (size_t)n, f);
    free(s);
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <talker_dir> <voice.pt> <text> <out.wav>\n", argv[0]);
        return 2;
    }
    cwick_error err = {0};
    double t_open = (double)cwick_prof_ns() / 1e9;
    rt_talker *t = rt_talker_open(argv[1], &err);
    if (!t) { fprintf(stderr, "talker: %s\n", err.message); return 1; }
    rt_voice v;
    if (rt_voice_load(argv[2], &v, &err) != 0) {
        fprintf(stderr, "voice: %s\n", err.message);
        rt_talker_close(t);
        return 1;
    }
    printf("voice: %d prompt tokens, %d prompt mel frames, open %.3f s\n",
           v.n_token, v.n_feat, (double)cwick_prof_ns() / 1e9 - t_open);

    sink s;
    memset(&s, 0, sizeof(s));
    s.t0 = (double)cwick_prof_ns() / 1e9;
    rt_talker_stats st;
    int rc = rt_talker_say(t, &v, argv[3], on_pcm, &s, &st);
    if (rc < 0) { fprintf(stderr, "say failed\n"); return 1; }

    double peak = 0, rms = 0;
    for (int i = 0; i < s.n; i++) {
        double a = s.pcm[i] < 0 ? -s.pcm[i] : s.pcm[i];
        if (a > peak) peak = a;
        rms += (double)s.pcm[i] * s.pcm[i];
    }
    if (s.n) rms = sqrt(rms / s.n);
    printf("\n--- talker chain ---\n");
    printf("lm prompt      %d tokens\n", st.lm_prompt_tokens);
    printf("lm prefill     %.3f s (%.1f tok/s)\n", st.lm_prefill_s,
           st.lm_prompt_tokens / (st.lm_prefill_s > 0 ? st.lm_prefill_s : 1));
    printf("speech tokens  %d (%.3f s decode, %.1f tok/s)\n", st.speech_tokens,
           st.lm_decode_s, st.speech_tokens / (st.lm_decode_s > 0 ? st.lm_decode_s : 1));
    printf("sampler        top1 %.3f, nucleus %.1f ids\n", st.lm_top1_prob, st.lm_nucleus);
    printf("flow           %.3f s\n", st.flow_s);
    printf("hift           %.3f s\n", st.hift_s);
    printf("mel frames     %d\n", st.mel_frames);
    printf("audio          %d samples = %.3f s in %d chunks\n", st.samples,
           st.samples / 24000.0, st.chunks);
    printf("first audio    %.3f s\n", st.first_audio_s);
    printf("total          %.3f s = %.2fx realtime\n", st.total_s,
           (st.samples / 24000.0) / (st.total_s > 0 ? st.total_s : 1));
    printf("peak %.4f rms %.5f\n", peak, rms);
    if (s.n) wav_write(argv[4], s.pcm, s.n, rt_talker_sample_rate(t));
    free(s.pcm);
    rt_voice_free(&v);
    rt_talker_close(t);
    return 0;
}