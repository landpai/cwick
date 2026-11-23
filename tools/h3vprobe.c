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

#include "h3v.h"
#include "cwick.h"

int main(int argc, char **argv) {
    /* Match the server's `--backend metal` when asked; the probe otherwise
     * inherits the CPU/autotune default because no CLI flag is parsed here. */
    if (getenv("H3VPROBE_METAL")) cwick_metal_force(1);
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
