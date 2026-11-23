/* h3vmuxprobe — exercise the mp4 muxer without running the video pipeline.
 *
 * The full H3V pipeline needs minutes of DiT and VAE work before it reaches
 * the muxer, which makes a container regression expensive to notice. This
 * probe feeds a synthetic colour ramp and a sine wave straight into
 * h3v_mp4_encode() so the box layout can be checked with ffprobe in seconds.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "h3v_internal.h"

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/h3vmux.mp4";
    int w = argc > 2 ? atoi(argv[2]) : 64;
    int h = argc > 3 ? atoi(argv[3]) : 64;
    int nf = argc > 4 ? atoi(argv[4]) : 12;
    if (w <= 0 || h <= 0 || nf <= 0) return 2;

    unsigned char *rgb = (unsigned char *)malloc((size_t)w * h * 3 * nf);
    if (!rgb) return 2;
    for (int f = 0; f < nf; f++)
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                unsigned char *p = rgb + ((size_t)f * h * 3 + (size_t)y * 3) * w + x * 3;
                p[0] = (unsigned char)(x * 255 / w);
                p[1] = (unsigned char)(y * 255 / h);
                p[2] = (unsigned char)(f * 255 / nf);
            }

    int64_t samples = (int64_t)nf * H3V_AUDIO_RATE / H3V_FPS;
    float *audio = (float *)calloc((size_t)samples * 2, sizeof(float));
    if (!audio) { free(rgb); return 2; }
    for (int64_t i = 0; i < samples; i++) {
        float v = 0.2f * (float)sin(2.0 * M_PI * 440.0 * i / H3V_AUDIO_RATE);
        audio[i * 2] = audio[i * 2 + 1] = v;
    }

    cwick_error err;
    memset(&err, 0, sizeof(err));
    size_t len = 0;
    unsigned char *mp4 = h3v_mp4_encode(rgb, nf, w, h, audio, samples, &len, &err);
    if (!mp4) {
        fprintf(stderr, "mux failed: %s\n", cwick_error_message(&err));
        free(audio);
        free(rgb);
        return 1;
    }
    FILE *fp = fopen(path, "wb");
    if (!fp) { perror("fopen"); free(mp4); free(audio); free(rgb); return 1; }
    if (fwrite(mp4, 1, len, fp) != len) { perror("fwrite"); fclose(fp); return 1; }
    fclose(fp);
    printf("OK %s bytes=%zu frames=%d samples=%lld\n", path, len, nf,
           (long long)samples);
    free(mp4);
    free(audio);
    free(rgb);
    return 0;
}
