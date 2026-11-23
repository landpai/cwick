/* h3vencprobe — exercise the visual VAE encoder outside the server.
 *
 * The first-frame-to-video path turns a reference RGB frame into one latent
 * step with the visual VAE encoder. This tool drives that encoder directly
 * (no 62 GiB transformer) and dumps the latent to /tmp so its numbers can be
 * checked. With H3VPROBE_RGB set it reads a raw RGB24 file instead of making
 * a deterministic pattern, which is how a fixed input is replayed.
 *
 * usage: tools/h3vencprobe ROOT [RGB_U8]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "h3v_internal.h"

/* Deterministic, structured pattern: horizontal ramp on R, vertical on G,
 * a centred bright block on B. Used when no RGB file is supplied. */
static void make_pattern(unsigned char *rgb, int w, int h) {
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            unsigned char *p = rgb + (y * w + x) * 3;
            p[0] = (unsigned char)(x * 255 / (w - 1));
            p[1] = (unsigned char)(y * 255 / (h - 1));
            p[2] = (x > w / 4 && x < 3 * w / 4 && y > h / 4 && y < 3 * h / 4)
                   ? 240 : 20;
        }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s ROOT [RGB_U8]\n", argv[0]);
        return 2;
    }
    cwick_error e;
    cwick_error_clear(&e);
    h3v_model *m = h3v_model_open(argv[1], &e);
    if (!m) {
        fprintf(stderr, "open failed: %s\n", cwick_error_message(&e));
        return 1;
    }
    const int W = getenv("H3VPROBE_W") ? atoi(getenv("H3VPROBE_W")) : 64;
    const int H = getenv("H3VPROBE_H") ? atoi(getenv("H3VPROBE_H")) : 64;
    unsigned char *rgb = (unsigned char *)malloc((size_t)W * H * 3);
    if (argc >= 3) {
        FILE *r = fopen(argv[2], "rb");
        if (!r || fread(rgb, 1, (size_t)W * H * 3, r) != (size_t)W * H * 3) {
            fprintf(stderr, "cannot read RGB dump %s\n", argv[2]);
            return 4;
        }
        fclose(r);
    } else {
        make_pattern(rgb, W, H);
    }

    float *z = h3v_vae_encode(m, rgb, W, H, &e);
    if (!z) {
        fprintf(stderr, "encode failed: %s\n", cwick_error_message(&e));
        free(rgb);
        h3v_model_close(m);
        return 3;
    }

    const char *dir = getenv("H3VPROBE_OUT");
    if (!dir) dir = "/tmp/h3venc_c";
    {
        char cmd[600];
        snprintf(cmd, sizeof(cmd), "mkdir -p %s", dir);
        system(cmd);
    }
    const int64_t lh = H / H3V_VAE_PIXEL, lw = W / H3V_VAE_PIXEL;
    {
        char p[512];
        snprintf(p, sizeof(p), "%s/rgb.u8", dir);
        FILE *f = fopen(p, "wb");
        if (f) { fwrite(rgb, 1, (size_t)W * H * 3, f); fclose(f); }
        snprintf(p, sizeof(p), "%s/z.f32", dir);
        f = fopen(p, "wb");
        if (f) {
            fwrite(z, sizeof(float), (size_t)H3V_VAE_LATENT_CH * lh * lw, f);
            fclose(f);
        }
    }

    double mn = 1e30, mx = -1e30, sum = 0.0, sum2 = 0.0;
    int64_t n = H3V_VAE_LATENT_CH * lh * lw;
    for (int64_t i = 0; i < n; i++) {
        double v = z[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v; sum2 += v * v;
    }
    double mean = sum / n, sd = sqrt(sum2 / n - mean * mean);
    printf("encode OK: %dx%d rgb -> 1x%d latent (%dx%d); "
           "min=%.5f max=%.5f mean=%.5f std=%.5f\n",
           W, H, H3V_VAE_LATENT_CH, (int)lw, (int)lh, mn, mx, mean, sd);
    printf("first 8: %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
           z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);

    /* Round-trip: feed the encoded latent straight back through the decoder.
     * If the encoder's numbers are right, the first reconstructed frame must
     * resemble the input pattern. This is a pure-C numerical check. */
    unsigned char *rec = h3v_vae_decode(m, z, 1, lh, lw, NULL, NULL, &e);
    if (!rec) {
        fprintf(stderr, "decode failed: %s\n", cwick_error_message(&e));
        free(z);
        free(rgb);
        h3v_model_close(m);
        return 5;
    }
    /* rec has 4 frames; compare frame 0 to the input on brightness correlation. */
    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    int npix = W * H;
    for (int i = 0; i < npix; i++) {
        double a = rgb[i * 3], b = rec[i * 3];   /* R channel */
        sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b;
    }
    double corr = (sab - sa * sb / npix) /
                  sqrt((saa - sa * sa / npix) * (sbb - sb * sb / npix) + 1e-9);
    printf("roundtrip R-channel correlation = %.4f\n", corr);
    {
        char p[512];
        snprintf(p, sizeof(p), "%s/recon.ppm", dir);
        FILE *f = fopen(p, "wb");
        if (f) {
            fprintf(f, "P6\n%d %d\n255\n", W, H);
            fwrite(rec, 1, (size_t)W * H * 3, f);
            fclose(f);
        }
    }
    h3v_mp4_free(rec);

    free(z);
    free(rgb);
    h3v_model_close(m);
    return 0;
}
