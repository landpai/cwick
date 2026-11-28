#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/resource.h>

#include "music.h"
#include "rt_torch.h"

bool music_vocoder_probe(music_model *m, const float *latent, int64_t frames,
                         float **out, int64_t *out_frames);
bool music_dit_probe(music_model *m, const float *latents, const float *condition,
                     int64_t frames, double time, float **out);
bool music_reference_probe(music_model *m, const unsigned char *wav, size_t len,
                           float **latent, int64_t *frames);
bool music_condition_probe(music_model *m, const float *frames, int64_t n_frames,
                           float **out, int64_t *out_frames);

static void probe_progress(void *ctx, const char *phase, int64_t done,
                           int64_t total, double frac) {
    (void)ctx;
    fprintf(stderr, "progress %s %lld/%lld %.3f\n", phase,
            (long long)done, (long long)total, frac);
}

static unsigned char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *p = (unsigned char *)malloc(n > 0 ? (size_t)n : 1);
    if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) {
        free(p); fclose(f); return NULL;
    }
    fclose(f);
    *len = (size_t)n;
    return p;
}

int main(int argc, char **argv) {
    if (argc == 5 && strcmp(argv[1], "--condition") == 0) {
        cwick_error err;
        cwick_error_clear(&err);
        music_model *m = music_model_open(argv[2], &err);
        if (!m) return 1;
        size_t len = 0;
        unsigned char *raw = read_file(argv[3], &len);
        int64_t frames = atoll(argv[4]);
        float *out = NULL;
        int64_t out_frames = 0;
        bool ok = raw && len == (size_t)frames * 8 * 4096 * sizeof(float) &&
                  music_condition_probe(m, (const float *)raw, frames, &out, &out_frames);
        if (ok) {
            FILE *f = fopen("/tmp/musicprobe-condition.f32", "wb");
            if (f) {
                fwrite(out, sizeof(float), (size_t)out_frames * 2048, f);
                fclose(f);
            }
            printf("ok frames=%lld\n", (long long)out_frames);
        }
        free(raw); free(out); music_model_close(m);
        return ok ? 0 : 1;
    }
    if (argc == 4 && strcmp(argv[1], "--encode") == 0) {
        cwick_error err;
        cwick_error_clear(&err);
        music_model *m = music_model_open(argv[2], &err);
        if (!m) return 1;
        size_t len = 0;
        unsigned char *wav = read_file(argv[3], &len);
        float *latent = NULL;
        int64_t frames = 0;
        bool ok = wav && music_reference_probe(m, wav, len, &latent, &frames);
        if (ok) {
            FILE *f = fopen("/tmp/musicprobe-reference.f32", "wb");
            if (f) {
                fwrite(latent, sizeof(float), (size_t)frames * 128, f);
                fclose(f);
            }
            printf("ok frames=%lld\n", (long long)frames);
        }
        free(wav); free(latent); music_model_close(m);
        return ok ? 0 : 1;
    }
    if (argc == 7 && strcmp(argv[1], "--dit") == 0) {
        cwick_error err;
        cwick_error_clear(&err);
        music_model *m = music_model_open(argv[2], &err);
        if (!m) return 1;
        size_t ll = 0, cl = 0;
        unsigned char *lat = read_file(argv[3], &ll);
        unsigned char *cond = read_file(argv[4], &cl);
        int64_t frames = atoll(argv[5]);
        double time = atof(argv[6]);
        float *out = NULL;
        bool ok = lat && cond && ll == (size_t)frames * 2 * 128 * sizeof(float) &&
                  cl == (size_t)frames * 2048 * sizeof(float) &&
                  music_dit_probe(m, (const float *)lat, (const float *)cond,
                                  frames, time, &out);
        if (ok) {
            FILE *f = fopen("/tmp/musicprobe-dit.f32", "wb");
            if (f) {
                fwrite(out, sizeof(float), (size_t)frames * 2 * 128, f);
                fclose(f);
            }
            printf("ok frames=%lld\n", (long long)frames);
        }
        free(lat); free(cond); free(out); music_model_close(m);
        return ok ? 0 : 1;
    }
    if (argc == 5 && strcmp(argv[1], "--vocoder") == 0) {
        cwick_error err;
        cwick_error_clear(&err);
        music_model *m = music_model_open(argv[2], &err);
        if (!m) return 1;
        size_t len = 0;
        unsigned char *raw = read_file(argv[3], &len);
        int64_t frames = atoll(argv[4]);
        float *pcm = NULL;
        int64_t pcm_frames = 0;
        bool ok = raw && len == (size_t)frames * 128 * sizeof(float) &&
                  music_vocoder_probe(m, (const float *)raw, frames, &pcm, &pcm_frames);
        if (ok) {
            FILE *f = fopen("/tmp/musicprobe-vocoder.f32", "wb");
            if (f) {
                fwrite(pcm, sizeof(float), (size_t)pcm_frames * 2, f);
                fclose(f);
            }
            printf("ok frames=%lld\n", (long long)pcm_frames);
        }
        free(raw); free(pcm); music_model_close(m);
        return ok ? 0 : 1;
    }
    if (argc >= 6 && strcmp(argv[1], "--generate") == 0) {
        cwick_error err;
        cwick_error_clear(&err);
        music_model *m = music_model_open(argv[2], &err);
        if (!m) {
            fprintf(stderr, "open failed: %s\n", cwick_error_message(&err));
            return 1;
        }
        size_t ref_len = 0;
        unsigned char *ref = argc > 6 ? read_file(argv[6], &ref_len) : NULL;
        if (argc > 6 && !ref) {
            fprintf(stderr, "cannot read reference\n");
            music_model_close(m);
            return 1;
        }
        music_request req;
        memset(&req, 0, sizeof(req));
        req.task = ref ? MUSIC_TASK_COVER : MUSIC_TASK_GENERATE;
        const char *probe_prompt = getenv("MUSIC_PROBE_PROMPT");
        const char *probe_lyrics = getenv("MUSIC_PROBE_LYRICS");
        const char *probe_cfg = getenv("MUSIC_PROBE_CFG");
        req.prompt = probe_prompt && *probe_prompt ? probe_prompt : "acoustic pop, warm piano, 96 BPM";
        req.lyrics = probe_lyrics && *probe_lyrics ? probe_lyrics : "[Verse]\nMorning light\n[Chorus]\nSoftly we breathe";
        req.reference_audio = ref;
        req.reference_len = ref_len;
        req.seconds = atoi(argv[3]);
        req.steps = atoi(argv[4]);
        req.seed = strtoull(argv[5], NULL, 10);
        req.cfg_scale = probe_cfg ? (float)atof(probe_cfg) : 1.5f;
        req.temperature = 1.0f;
        req.top_k = 50;
        req.audio_weight = 0.85f;
        req.progress = probe_progress;
        unsigned char *wav = NULL;
        size_t wav_len = 0;
        music_timing tim;
        bool ok = music_generate(m, &req, &wav, &wav_len, &tim, &err);
        if (ok) {
            FILE *f = fopen("/tmp/musicprobe-output.wav", "wb");
            if (f) {
                fwrite(wav, 1, wav_len, f);
                fclose(f);
            }
            printf("ok wav=%zu total=%.0f load=%.0f tokenize=%.0f reference=%.0f "
                   "global=%.0f depth=%.0f condition=%.0f flow=%.0f vocoder=%.0f "
                   "wav=%.0f frames=%lld steps=%lld samples=%lld\n",
                   wav_len, tim.total_ms, tim.load_ms, tim.tokenize_ms,
                   tim.reference_encode_ms, tim.global_lm_ms, tim.depth_ms,
                   tim.condition_ms, tim.flow_ms, tim.vocoder_ms, tim.wav_ms,
                   (long long)tim.frames, (long long)tim.flow_steps,
                   (long long)tim.audio_samples);
        } else {
            fprintf(stderr, "generate failed: %s\n", cwick_error_message(&err));
        }
        free(ref);
        music_wav_free(wav);
        music_model_close(m);
        return ok ? 0 : 1;
    }
    if (argc == 4 && strcmp(argv[1], "--tensor") == 0) {
        rt_pt *pt = NULL;
        if (rt_pt_open(&pt, argv[2]) != 0) return 1;
        const rt_tensor *t = rt_pt_get(pt, argv[3]);
        if (!t || t->dtype != RT_DT_F32) {
            fprintf(stderr, "tensor not found or not f32: %s\n", argv[3]);
            rt_pt_close(pt);
            return 1;
        }
        const float *v = (const float *)t->data;
        float lo = v[0], hi = v[0];
        double sum = 0.0;
        for (int64_t i = 0; i < t->numel; i++) {
            if (v[i] < lo) lo = v[i];
            if (v[i] > hi) hi = v[i];
            sum += fabs((double)v[i]);
        }
        printf("shape=[");
        for (int d = 0; d < t->ndim; d++) printf("%s%lld", d ? "," : "", (long long)t->shape[d]);
        printf("] min=%.9g max=%.9g mean_abs=%.9g first=", lo, hi, sum / (double)t->numel);
        for (int64_t i = 0; i < t->numel && i < 5; i++) printf("%s%.9g", i ? "," : "", v[i]);
        printf("\n");
        rt_pt_close(pt);
        return 0;
    }
    if (argc == 3 && strcmp(argv[1], "--open") == 0) {
        cwick_error err;
        cwick_error_clear(&err);
        music_model *m = music_model_open(argv[2], &err);
        if (!m) {
            fprintf(stderr, "open failed: %s\n", cwick_error_message(&err));
            return 1;
        }
        printf("ready=%d dir=%s\n", music_model_ready(m), music_model_dir(m));
        music_model_close(m);
        return 0;
    }
    if (argc == 2) {
        rt_pt *pt = NULL;
        if (rt_pt_open(&pt, argv[1]) != 0) {
            fprintf(stderr, "cannot open %s\n", argv[1]);
            return 1;
        }
        int n = rt_pt_count(pt);
        for (int i = 0; i < n; i++) {
            const rt_tensor *t = rt_pt_at(pt, i);
            printf("%s %s [", t->name, rt_dtype_name(t->dtype));
            for (int d = 0; d < t->ndim; d++)
                printf("%s%lld", d ? "," : "", (long long)t->shape[d]);
            printf("] numel=%lld\n", (long long)t->numel);
        }
        rt_pt_close(pt);
        return 0;
    }
    fprintf(stderr, "usage: musicprobe --open MODEL_ROOT | checkpoint.pth\n");
    return 2;
}
