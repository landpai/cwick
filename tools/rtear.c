/* rtear.c — ear smoke test.
 *
 *   tools/rtear <thinker_dir> <wav> [dump_prefix]
 *
 * Reads a 16 kHz mono WAV (the voice clips in the checkpoint are exactly
 * that), pads it the way the streaming path does, and prints the token count
 * plus a few statistics of the projected embeddings. With a dump prefix it
 * also writes the log-mel and the tower output as raw f32, which is what the
 * independent reference compares against. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cwick.h"
#include "log.h"
#include "rt_audio.h"

static int read_wav(const char *path, float **pcm, int *rate) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    unsigned char h[44];
    if (fread(h, 1, 44, f) != 44) { fclose(f); return -1; }
    if (memcmp(h, "RIFF", 4) || memcmp(h + 8, "WAVE", 4)) { fclose(f); return -1; }
    int ch = h[22] | (h[23] << 8);
    *rate = h[24] | (h[25] << 8) | (h[26] << 16) | (h[27] << 24);
    int bits = h[34] | (h[35] << 8);
    if (bits != 16 || ch != 1) {
        fprintf(stderr, "need 16-bit mono, got %d ch / %d bit\n", ch, bits);
        fclose(f);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f) - 44;
    fseek(f, 44, SEEK_SET);
    short *raw = (short *)malloc((size_t)sz);
    if (fread(raw, 1, (size_t)sz, f) != (size_t)sz) { free(raw); fclose(f); return -1; }
    fclose(f);
    int n = (int)(sz / 2);
    *pcm = (float *)malloc(sizeof(float) * (size_t)n);
    for (int i = 0; i < n; i++) (*pcm)[i] = (float)raw[i] / 32768.0f;
    free(raw);
    return n;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <thinker_dir> <wav> [dump_prefix]\n", argv[0]);
        return 2;
    }
    float *pcm = NULL;
    int rate = 0;
    int n = read_wav(argv[2], &pcm, &rate);
    if (n < 0) {
        fprintf(stderr, "cannot read %s\n", argv[2]);
        return 1;
    }
    if (rate != 16000) {
        fprintf(stderr, "need 16 kHz, got %d\n", rate);
        return 1;
    }
    cwick_error err;
    memset(&err, 0, sizeof(err));
    rt_ear *e = rt_ear_open(argv[1], &err);
    if (!e) {
        fprintf(stderr, "ear open failed: %s\n", err.message);
        return 1;
    }

    /* The streaming caller pads every clip to a whole 2560-sample step so the
     * tower's frame count is exact; do the same here. */
    const int step = 160 * 8 * 2;
    int padded = (n + step - 1) / step * step;
    float *buf = (float *)calloc((size_t)padded, sizeof(float));
    memcpy(buf, pcm, sizeof(float) * (size_t)n);
    printf("wav: %d samples (%.2f s) -> padded %d, mel frames %d\n", n,
           (double)n / 16000.0, padded, rt_ear_frames_for_samples(padded));

    if (argc > 3) {
        int frames = rt_ear_frames_for_samples(padded);
        float *mel = (float *)calloc((size_t)frames * 128, sizeof(float));
        rt_ear_logmel(e, buf, padded, mel);
        char p[1200];
        snprintf(p, sizeof(p), "%s.mel.f32", argv[3]);
        FILE *f = fopen(p, "wb");
        if (f) {
            fwrite(mel, sizeof(float), (size_t)frames * 128, f);
            fclose(f);
        }
        /* Range of the normalised features: everything at 0 means silence, and
         * a constant 1.0 means the log floor -- both are worth seeing. */
        double mn = 1e9, mx = -1e9, avg = 0;
        for (int64_t i = 0; i < (int64_t)frames * 128; i++) {
            if (mel[i] < mn) mn = mel[i];
            if (mel[i] > mx) mx = mel[i];
            avg += mel[i];
        }
        printf("log-mel: min %.3f max %.3f mean %.3f\n", mn, mx,
               avg / ((double)frames * 128));
        free(mel);
    }

    double t0 = (double)cwick_prof_ns() / 1e9;
    float *emb = NULL;
    int tok = 0;
    if (rt_ear_embed(e, buf, padded, &emb, &tok) != 0) {
        fprintf(stderr, "embed failed\n");
        return 1;
    }
    double t1 = (double)cwick_prof_ns() / 1e9;
    printf("tokens: %d in %.3f s (%.1f tokens/s, %.2f s of audio per s)\n", tok,
           t1 - t0, tok / (t1 - t0), ((double)padded / 16000.0) / (t1 - t0));

    if (tok > 0) {
        int H = rt_ear_hidden(e);
        double mn = 1e9, mx = -1e9, avg = 0, norm = 0;
        for (int64_t i = 0; i < (int64_t)tok * H; i++) {
            if (emb[i] < mn) mn = emb[i];
            if (emb[i] > mx) mx = emb[i];
            avg += emb[i];
            norm += (double)emb[i] * emb[i];
        }
        printf("embeddings: %d x %d, min %.4f max %.4f mean %.4f rms %.4f\n", tok, H, mn,
               mx, avg / ((double)tok * H), sqrt(norm / ((double)tok * H)));
        printf("tok0[0..7]:");
        for (int i = 0; i < 8; i++) printf(" %+.5f", emb[i]);
        printf("\n");
        if (argc > 3) {
            char p[1200];
            snprintf(p, sizeof(p), "%s.tok.f32", argv[3]);
            FILE *f = fopen(p, "wb");
            if (f) {
                fwrite(emb, sizeof(float), (size_t)tok * H, f);
                fclose(f);
            }
        }
    }
    free(emb);
    free(buf);
    free(pcm);
    rt_ear_close(e);
    return 0;
}
