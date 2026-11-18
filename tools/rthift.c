/* rthift — mel in, 24 kHz WAV out. Runs the causal HiFT vocoder on a voice
 * prompt's own speech_feat, so the output can be compared sample-for-sample
 * with the shipped reference clip.
 *
 *   tools/rthift <talker_dir> <voice.pt> <out.wav> [mel_frames]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cwick.h"
#include "rt_dsp.h"
#include "rt_hift.h"
#include "rt_torch.h"
#include "util.h"

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
    if (argc < 4) {
        fprintf(stderr, "usage: %s <talker_dir> <voice.pt> <out.wav> [frames]\n", argv[0]);
        return 2;
    }
    char hpath[1024];
    snprintf(hpath, sizeof(hpath), "%s/hift.pt", argv[1]);
    cwick_error err = {0};
    rt_hift *h = rt_hift_open(hpath, &err);
    if (!h) { fprintf(stderr, "hift: %s\n", err.message); return 1; }

    rt_pt *pt = NULL;
    float *m = NULL;
    int T = 0;
    size_t plen = strlen(argv[2]);
    if (plen > 4 && strcmp(argv[2] + plen - 4, ".f32") == 0) {
        FILE *fp = fopen(argv[2], "rb");
        if (!fp) { perror(argv[2]); return 1; }
        fseek(fp, 0, SEEK_END);
        long bytes = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        T = (int)(bytes / (80 * sizeof(float)));
        m = (float *)malloc((size_t)T * 80 * sizeof(float));
        if (fread(m, sizeof(float), (size_t)T * 80, fp) != (size_t)T * 80) T = 0;
        fclose(fp);
        if (argc > 4) T = atoi(argv[4]);
    } else {
        if (rt_pt_open(&pt, argv[2]) != 0) { fprintf(stderr, "voice: %s\n", argv[2]); return 1; }
        const rt_tensor *mel = rt_pt_get(pt, "speech_feat");
        if (!mel || mel->ndim != 3) { fprintf(stderr, "no speech_feat\n"); return 1; }
        T = (int)mel->shape[1];
        if (argc > 4) T = atoi(argv[4]);
        m = (float *)malloc((size_t)T * 80 * sizeof(float));
        for (int64_t i = 0; i < (int64_t)T * 80; i++) m[i] = rt_tensor_f32(mel, i);
    }
    if (T <= 0) { fprintf(stderr, "no mel\n"); return 1; }

    double t0 = (double)cwick_prof_ns() / 1e9;
    float *pcm = NULL;
    int n = rt_hift_synth(h, m, T, &pcm);
    double dt = (double)cwick_prof_ns() / 1e9 - t0;
    if (n < 0) { fprintf(stderr, "synth failed\n"); return 1; }
    double peak = 0, rms = 0;
    for (int i = 0; i < n; i++) {
        double a = pcm[i] < 0 ? -pcm[i] : pcm[i];
        if (a > peak) peak = a;
        rms += (double)pcm[i] * pcm[i];
    }
    rms = sqrt(rms / n);
    printf("mel %d frames -> %d samples (%.3f s audio) in %.3f s = %.1fx realtime\n",
           T, n, n / 24000.0, dt, (n / 24000.0) / dt);
    printf("peak %.4f rms %.5f\n", peak, rms);
    wav_write(argv[3], pcm, n, 24000);
    free(pcm); free(m);
    if (pt) rt_pt_close(pt);
    rt_hift_close(h);
    return 0;
}