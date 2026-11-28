/* h3vprobe — run one MiniMax-H3 clip outside the server.
 *
 * The server path needs an HTTP client, a job queue and a video player to tell
 * a wrong weight name from a wrong number, so this drives h3v_generate()
 * directly and prints the per-stage timing the /v1/videos metrics are built
 * from. CWICK_H3V_STEPS=1 collapses the sampler to a single step, which is how
 * a wiring change is checked without paying for all 30.
 *
 * usage: tools/h3vprobe ROOT PROMPT WIDTH HEIGHT SECONDS
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "h3v.h"
#include "h3v_internal.h"
#include "cwick.h"

int main(int argc, char **argv) {
    /* Match the server's `--backend metal` when asked; the probe otherwise
     * inherits the CPU/autotune default because no CLI flag is parsed here. */
    if (getenv("H3VPROBE_METAL")) cwick_metal_force(1);
    if (argc == 7 && strcmp(argv[1], "--vae-roundtrip") == 0) {
        int w = atoi(argv[4]), h = atoi(argv[5]);
        if (w <= 0 || h <= 0 || w % 16 || h % 16) {
            fprintf(stderr, "RGB dimensions must be positive multiples of 16\n");
            return 2;
        }
        FILE *rf = fopen(argv[3], "rb");
        unsigned char *rgb = (unsigned char *)malloc((size_t)w * h * 3);
        if (!rf || !rgb || fread(rgb, 1, (size_t)w * h * 3, rf) != (size_t)w * h * 3) {
            fprintf(stderr, "cannot read RGB24 %s\n", argv[3]);
            if (rf) fclose(rf);
            free(rgb);
            return 4;
        }
        fclose(rf);
        cwick_error e;
        cwick_error_clear(&e);
        h3v_model *m = h3v_model_open(argv[2], &e);
        if (!m) {
            fprintf(stderr, "open failed: %s\n", cwick_error_message(&e));
            free(rgb);
            return 1;
        }
        double t0 = (double)cwick_prof_ns() / 1e6;
        float *lat = h3v_vae_encode(m, rgb, w, h, &e);
        double t1 = (double)cwick_prof_ns() / 1e6;
        unsigned char *out = lat ? h3v_vae_decode(m, lat, 1, h / 16, w / 16,
                                                   NULL, NULL, &e) : NULL;
        double t2 = (double)cwick_prof_ns() / 1e6;
        FILE *pf = out ? fopen(argv[6], "wb") : NULL;
        if (pf) {
            fwrite(out, 1, (size_t)w * h * 3, pf);
            fclose(pf);
        }
        printf("vae-roundtrip %dx%d encode=%.1f ms decode=%.1f ms rgb=%zu bytes\n",
               w, h, t1 - t0, t2 - t1, (size_t)w * h * 3);
        free(out); free(lat); free(rgb);
        h3v_model_close(m);
        return pf ? 0 : 3;
    }
    if (argc == 6 && strcmp(argv[1], "--audio-decode") == 0) {
        int64_t t = atoll(argv[4]);
        if (t < 2 || (t & 1)) {
            fprintf(stderr, "audio latent length must be a positive even number\n");
            return 2;
        }
        FILE *lf = fopen(argv[3], "rb");
        float *z = (float *)malloc((size_t)32 * t * sizeof(float));
        if (!lf || !z || fread(z, sizeof(float), (size_t)32 * t, lf) != (size_t)32 * t) {
            fprintf(stderr, "cannot read audio latent %s\n", argv[3]);
            if (lf) fclose(lf);
            free(z);
            return 4;
        }
        fclose(lf);
        cwick_error e;
        cwick_error_clear(&e);
        h3v_model *m = h3v_model_open(argv[2], &e);
        if (!m) {
            fprintf(stderr, "open failed: %s\n", cwick_error_message(&e));
            free(z);
            return 1;
        }
        int64_t samples = 0;
        double t0 = (double)cwick_prof_ns() / 1e6;
        float *pcm = h3v_audio_decode(m, z, t, &samples, NULL, NULL, &e);
        double t1 = (double)cwick_prof_ns() / 1e6;
        FILE *pf = pcm ? fopen(argv[5], "wb") : NULL;
        if (pf) {
            fwrite(pcm, sizeof(float), (size_t)samples * 2, pf);
            fclose(pf);
        }
        double ss = 0.0, peak = 0.0;
        for (int64_t i = 0; pcm && i < samples * 2; i++) {
            ss += (double)pcm[i] * pcm[i];
            if (fabs(pcm[i]) > peak) peak = fabs(pcm[i]);
        }
        printf("audio-decode tokens=%lld samples=%lld rms=%.6f peak=%.6f decode=%.1f ms\n",
               (long long)t, (long long)samples,
               samples ? sqrt(ss / (double)(samples * 2)) : 0.0, peak, t1 - t0);
        free(pcm); free(z);
        h3v_model_close(m);
        return pf ? 0 : 3;
    }
    if (argc < 6) {
        fprintf(stderr, "usage: %s ROOT PROMPT W H SECONDS\n", argv[0]);
        return 2;
    }
    cwick_error e;
    cwick_error_clear(&e);
    h3v_model *m = h3v_model_open(argv[1], &e);
    if (!m) {
        fprintf(stderr, "open failed: %s\n", cwick_error_message(&e));
        return 1;
    }
    fprintf(stderr, "opened: default_variant=%s available=%d [",
            h3v_model_variant(m), h3v_model_variant_count(m));
    for (int i = 0; i < h3v_model_variant_count(m); i++)
        fprintf(stderr, "%s%s", i ? "," : "", h3v_model_variant_name(m, i));
    fprintf(stderr, "]\n");
    /* H3VPROBE_VARIANT pins the transformer partition for this run. */
    const char *want = getenv("H3VPROBE_VARIANT");
    const char *canon = h3v_model_resolve_variant(m, want);
    if (want && *want && !canon) {
        fprintf(stderr, "unknown variant '%s'\n", want);
        h3v_model_close(m);
        return 5;
    }
    const char *use_variant = canon ? canon : h3v_model_variant(m);
    fprintf(stderr, "probe variant: %s\n", use_variant);

    unsigned char *mp4 = NULL;
    size_t len = 0;
    h3v_timing t;
    /* H3VPROBE_REF=<rgb.u8> turns this into a first-frame-to-video smoke test:
     * the raw RGB24 file is passed straight to the i2v entry point. */
    const char *refp = getenv("H3VPROBE_REF");
    unsigned char *ref = NULL;
    int rw = 0, rh = 0;
    if (refp && *refp) {
        FILE *f = fopen(refp, "rb");
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        rw = atoi(getenv("H3VPROBE_REF_W") ? getenv("H3VPROBE_REF_W") : "64");
        rh = atoi(getenv("H3VPROBE_REF_H") ? getenv("H3VPROBE_REF_H") : "64");
        ref = (unsigned char *)malloc((size_t)rw * rh * 3);
        if (!f || sz < rw * rh * 3 ||
            fread(ref, 1, (size_t)rw * rh * 3, f) != (size_t)rw * rh * 3) {
            fprintf(stderr, "cannot read reference RGB %s\n", refp);
            return 4;
        }
        fclose(f);
    }
    int ok = h3v_generate_variant(m, use_variant, argv[2], ref, rw, rh,
                                  atoi(argv[3]), atoi(argv[4]), atoi(argv[5]),
                                  42, NULL, NULL, &mp4, &len, &t, &e);
    free(ref);
    if (!ok) {
        fprintf(stderr, "generate failed: %s\n", cwick_error_message(&e));
        h3v_model_close(m);
        return 3;
    }
    const char *out = getenv("H3VPROBE_OUT");
    FILE *f = fopen(out ? out : "/tmp/h3v_out.mp4", "wb");
    if (f) {
        fwrite(mp4, 1, len, f);
        fclose(f);
    }
    printf("OK bytes=%zu total=%.1f text=%.1f pack=%.1f dit=%.1f vae=%.1f "
           "avae=%.1f mux=%.1f frames=%lld %lldx%lld steps=%lld\n",
           len, t.total_ms, t.text_encode_ms, t.pack_ms, t.dit_ms, t.vae_ms,
           t.audio_vae_ms, t.mux_ms, (long long)t.frames, (long long)t.width,
           (long long)t.height, (long long)t.steps);
    /* Denoise quality: spatial_corr near 0 means the generated latent was
     * still white noise even though an mp4 was produced. This is the pure-C
     * signal the /v1/videos `suspect_noise` field and metric are built from. */
    printf("denoise spatial_corr=%.3f clean_corr=%.3f snr_db=%.2f "
           "suspect_noise=%d\n",
           t.spatial_corr, t.clean_corr, t.snr_db, t.suspect_noise);
    h3v_mp4_free(mp4);
    h3v_model_close(m);
    return 0;
}
