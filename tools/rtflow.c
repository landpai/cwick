/* rtflow — speech tokens to mel, checked against a voice prompt's own mel.
 *
 *   tools/rtflow <talker_dir> <voice.pt> [out.f32] [streaming]
 *
 * Splits the prompt in half: the first half is the conditioning prompt, the
 * second half is rendered from its speech tokens alone. The result should land
 * close to the shipped speech_feat for that region, which is the only
 * end-to-end anchor available without a torch reference. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cwick.h"
#include "rt_dsp.h"
#include "rt_flow.h"
#include "rt_torch.h"
#include "util.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <talker_dir> <voice.pt> [out.f32] [streaming]\n", argv[0]);
        return 2;
    }
    char path[1024];
    snprintf(path, sizeof(path), "%s/flow.pt", argv[1]);
    cwick_error err = {0};
    rt_flow *f = rt_flow_open(path, &err);
    if (!f) { fprintf(stderr, "flow: %s\n", err.message); return 1; }

    rt_pt *pt = NULL;
    if (rt_pt_open(&pt, argv[2]) != 0) { fprintf(stderr, "voice: %s\n", argv[2]); return 1; }
    const rt_tensor *st = rt_pt_get(pt, "speech_token");
    const rt_tensor *sf = rt_pt_get(pt, "speech_feat");
    const rt_tensor *em = rt_pt_get(pt, "embedding");
    if (!st || !sf || !em) { fprintf(stderr, "voice tensors missing\n"); return 1; }
    int nt = (int)st->shape[1], nf = (int)sf->shape[1];
    int *tok = (int *)malloc((size_t)nt * sizeof(int));
    float *mel = (float *)malloc((size_t)nf * 80 * sizeof(float));
    float *spk = (float *)malloc(192 * sizeof(float));
    for (int i = 0; i < nt; i++) tok[i] = (int)rt_tensor_f32(st, i);
    for (int i = 0; i < nf * 80; i++) mel[i] = rt_tensor_f32(sf, i);
    for (int i = 0; i < 192; i++) spk[i] = rt_tensor_f32(em, i);

    int half = nt / 2;
    int pf = half * 2;
    int n = nt - half;
    int frames = rt_flow_frames(n, 0, 0);
    float *out = (float *)malloc((size_t)frames * 80 * sizeof(float));
    int streaming = argc > 4 ? atoi(argv[4]) : 0;

    double t0 = (double)cwick_prof_ns() / 1e9;
    int got;
    if (!streaming) {
        got = rt_flow_mel(f, tok, half, mel, pf, spk, tok + half, n, 0, 0, 0, out);
    } else {
        /* 25-token chunks with 3 tokens of lookahead, exactly the protocol the
         * reference uses; each call re-renders the prefix and writes only the
         * new frames. */
        got = 0;
        int chunk = 25, la = 3;
        for (int i = 0; i < n; i += chunk) {
            int take = i + chunk + la;
            int fin = take >= n;
            if (take > n) take = n;
            int look = fin ? 0 : la;
            int total = rt_flow_frames(take, look, i);
            float *tmp = (float *)malloc((size_t)(total > 0 ? total : 1) * 80 * sizeof(float));
            int r = rt_flow_mel(f, tok, half, mel, pf, spk, tok + half, take, look, i, 1, tmp);
            if (r < 0) { fprintf(stderr, "chunk %d failed\n", i); free(tmp); got = -1; break; }
            int copy = r;
            if (copy > frames - got) copy = frames - got;
            if (copy > 0) memcpy(out + (size_t)got * 80, tmp, (size_t)copy * 80 * sizeof(float));
            got += copy > 0 ? copy : 0;
            free(tmp);
        }
    }
    double dt = (double)cwick_prof_ns() / 1e9 - t0;
    if (got < 0) return 1;
    printf("rendered %d mel frames (%.2f s audio) in %.3f s = %.2fx realtime%s\n",
           got, got / 50.0, dt, (got / 50.0) / dt, streaming ? " [streaming]" : "");

    /* compare against the shipped mel for the same region */
    int cmp = got < nf - pf ? got : nf - pf;
    double num = 0, d1 = 0, d2 = 0, mae = 0, mx = 0;
    for (int i = 0; i < cmp * 80; i++) {
        double a = mel[pf * 80 + i], b = out[i];
        num += a * b; d1 += a * a; d2 += b * b;
        double e = fabs(a - b);
        mae += e;
        if (e > mx) mx = e;
    }
    printf("vs reference mel: corr %.4f  mae %.4f  max %.4f  (ref rms %.3f, out rms %.3f)\n",
           num / sqrt(d1 * d2), mae / (cmp * 80), mx, sqrt(d1 / (cmp * 80)),
           sqrt(d2 / (cmp * 80)));
    {
        int N = cmp * 80;
        double ma = 0, mb = 0;
        for (int i = 0; i < N; i++) { ma += mel[pf * 80 + i]; mb += out[i]; }
        ma /= N; mb /= N;
        double va = 0, vb = 0, cov = 0;
        for (int i = 0; i < N; i++) {
            double a = mel[pf * 80 + i] - ma, b = out[i] - mb;
            va += a * a; vb += b * b; cov += a * b;
        }
        printf("mean ref %.3f out %.3f | sd ref %.3f out %.3f | centered corr %.4f | fit ref = %.3f * out + %.3f\n",
               ma, mb, sqrt(va / N), sqrt(vb / N), cov / sqrt(va * vb),
               cov / vb, ma - (cov / vb) * mb);
        for (int g = 0; g < 5; g++) {
            int lo = g * cmp / 5, hi = (g + 1) * cmp / 5;
            double sa = 0, sb = 0, ca = 0, cb = 0, cc = 0, ea = 0;
            int M = (hi - lo) * 80;
            for (int t = lo; t < hi; t++)
                for (int c = 0; c < 80; c++) {
                    double a = mel[(pf + t) * 80 + c], b = out[t * 80 + c];
                    sa += a; sb += b; ea += fabs(a - b);
                }
            sa /= M; sb /= M;
            for (int t = lo; t < hi; t++)
                for (int c = 0; c < 80; c++) {
                    double a = mel[(pf + t) * 80 + c] - sa, b = out[t * 80 + c] - sb;
                    ca += a * a; cb += b * b; cc += a * b;
                }
            printf("  frames %3d-%3d: corr %.3f mae %.2f mean %.2f/%.2f sd %.2f/%.2f\n",
                   lo, hi, cc / sqrt(ca * cb), ea / M, sa, sb, sqrt(ca / M), sqrt(cb / M));
        }
        /* per-bin means, to see whether the mismatch is a tilt or an offset */
        printf("bin means ref/out:");
        for (int c = 0; c < 80; c += 8) {
            double sa = 0, sb = 0;
            for (int t = 0; t < cmp; t++) { sa += mel[(pf + t) * 80 + c]; sb += out[t * 80 + c]; }
            printf(" [%d]%.2f/%.2f", c, sa / cmp, sb / cmp);
        }
        printf("\n");
    }
    if (argc > 3) {
        FILE *fp = fopen(argv[3], "wb");
        if (fp) { fwrite(out, sizeof(float), (size_t)got * 80, fp); fclose(fp); }
        /* full mel (prompt + rendered) so the vocoder can be run on it */
        char full[1200];
        snprintf(full, sizeof(full), "%s.full.f32", argv[3]);
        fp = fopen(full, "wb");
        if (fp) {
            fwrite(mel, sizeof(float), (size_t)pf * 80, fp);
            fwrite(out, sizeof(float), (size_t)got * 80, fp);
            fclose(fp);
        }
    }
    {
        int floor_ref = 0, floor_out = 0;
        for (int i = 0; i < cmp * 80; i++) {
            if (mel[pf * 80 + i] < -11.0) floor_ref++;
            if (out[i] < -11.0) floor_out++;
        }
        printf("clamped-silence bins: ref %d / out %d of %d\n", floor_ref, floor_out, cmp * 80);
        printf("frame means ref/out:");
        for (int t = 0; t < cmp; t += 15) {
            double sa = 0, sb = 0;
            for (int c = 0; c < 80; c++) { sa += mel[(pf + t) * 80 + c]; sb += out[t * 80 + c]; }
            printf(" [%d]%.1f/%.1f", t, sa / 80, sb / 80);
        }
        printf("\n");
    }
    free(out); free(tok); free(mel); free(spk);
    rt_pt_close(pt);
    rt_flow_close(f);
    return 0;
}