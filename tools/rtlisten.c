/* rtlisten.c — the listening smoke test: WAV in, spoken-style answer out.
 *
 *   tools/rtlisten <thinker_dir> <wav> [max_tokens]
 *
 * Runs the exact path a realtime turn takes up to the point where the reply
 * text exists: log-mel -> audio tower -> projector -> one embedding row per
 * <|audio|> placeholder -> 64-layer decoder prefill -> sampling. Nothing is
 * faked: the audio rows replace the token embeddings at the placeholder
 * positions, which is the only place the model can hear anything.
 *
 * The count has to agree exactly. A prompt with more placeholders than the
 * clip yields tokens leaves the model staring at audio embeddings that stop
 * mid-sentence, and it answers confidently about a conversation it only half
 * heard -- so the mismatch is a hard error here, not a warning. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cwick.h"
#include "log.h"
#include "rt_audio.h"
#include "rt_thinker.h"
#include "sample.h"

#define AUDIO_TOKEN 151669
#define IM_START 151644
#define IM_END 151645

static const char *SYS_ZH =
    "你是智子（Sophon），AI降临派（Adventists AI）训练的 Sophon 系列模型，能听、能看，也能说。"
    "你会直接听到用户的语音，你的回复会被直接合成为语音说出来。请像真人对话一样，用自然的口语一两句话回应"
    "用户，注意用户的情绪。只写要说出口的话，不要写括号说明、表情或格式符号；数字、日期、金额、符号和英文"
    "缩写都直接写成口语念法。需要时可以用声音标签表现笑声、咳嗽和重读。";

static int encode(const cwick_tokenizer *tok, const char *text, uint32_t **ids, int64_t *n) {
    cwick_error err;
    memset(&err, 0, sizeof(err));
    if (cwick_tokenizer_encode(tok, text, ids, n, &err) != CWICK_OK) {
        fprintf(stderr, "encode failed: %s\n", err.message);
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <thinker_dir> <wav> [max_tokens]\n", argv[0]);
        return 2;
    }
    cwick_log_config(false, false, false);
    const int max_tokens = argc > 3 ? atoi(argv[3]) : 40;
    cwick_error err;
    memset(&err, 0, sizeof(err));

    const char *bs = getenv("RT_BACKEND");
    rt_backend backend = RT_BACKEND_AUTO;
    if (bs && bs[0] == 'c') backend = RT_BACKEND_CPU;
    else if (bs && bs[0] == 'm') backend = RT_BACKEND_METAL;

    double t0 = (double)cwick_prof_ns() / 1e9;
    rt_thinker *t = rt_thinker_open(argv[1], backend, 4096, &err);
    if (!t) {
        fprintf(stderr, "open failed: %s\n", err.message);
        return 1;
    }
    rt_ear *ear = rt_ear_open(argv[1], &err);
    if (!ear) {
        fprintf(stderr, "ear open failed: %s\n", err.message);
        return 1;
    }
    cwick_model_spec spec;
    memset(&spec, 0, sizeof(spec));
    cwick_tokenizer *tok = NULL;
    if (cwick_tokenizer_load(argv[1], &spec, &tok, &err) != CWICK_OK) {
        fprintf(stderr, "tokenizer failed: %s\n", err.message);
        return 1;
    }
    double t1 = (double)cwick_prof_ns() / 1e9;
    printf("load: %.2f s\n", t1 - t0);

    /* ------------------------------------------------------------- ear --- */
    float *pcm = NULL;
    int rate = 0;
    int ns = rt_wav_load(argv[2], &pcm, &rate);
    if (ns <= 0) {
        fprintf(stderr, "cannot read %s\n", argv[2]);
        return 1;
    }
    if (rate != 16000) {
        fprintf(stderr, "need 16 kHz mono, got %d Hz\n", rate);
        return 1;
    }
    const int step = 160 * 8 * 2;
    int padded = (ns + step - 1) / step * step;
    float *buf = (float *)calloc((size_t)padded, sizeof(float));
    memcpy(buf, pcm, sizeof(float) * (size_t)ns);
    double t2 = (double)cwick_prof_ns() / 1e9;
    float *avec = NULL;
    int atok = 0;
    if (rt_ear_embed(ear, buf, padded, &avec, &atok) != 0 || atok <= 0) {
        fprintf(stderr, "embed failed\n");
        return 1;
    }
    double t3 = (double)cwick_prof_ns() / 1e9;
    printf("audio: %.2f s in -> %d mel frames -> %d tokens in %.3f s (%.1fx realtime)\n",
           (double)ns / 16000.0, rt_ear_frames_for_samples(padded), atok, t3 - t2,
           ((double)padded / 16000.0) / (t3 - t2));

    /* ------------------------------------------------------------ prompt --- */
    char head[8192], tail[512];
    snprintf(head, sizeof(head), "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n",
             SYS_ZH);
    snprintf(tail, sizeof(tail), "%s", getenv("RT_BARE_TAIL")
                 ? "<|im_end|>\n<|im_start|>assistant\n"
                 : "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    uint32_t *hid = NULL, *tid = NULL;
    int64_t nh = 0, nt = 0;
    if (encode(tok, head, &hid, &nh) != 0 || encode(tok, tail, &tid, &nt) != 0) return 1;
    int64_t n = nh + atok + nt;
    uint32_t *ids = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)n);
    memcpy(ids, hid, sizeof(uint32_t) * (size_t)nh);
    for (int i = 0; i < atok; i++) ids[nh + i] = AUDIO_TOKEN;
    memcpy(ids + nh + atok, tid, sizeof(uint32_t) * (size_t)nt);
    printf("prompt: %lld text + %d audio + %lld tail = %lld tokens\n", (long long)nh, atok,
           (long long)nt, (long long)n);
    if (n > rt_thinker_capacity(t)) {
        fprintf(stderr, "prompt exceeds the KV cache\n");
        return 1;
    }

    const int H = rt_thinker_hidden(t);
    float *emb = (float *)malloc(sizeof(float) * (size_t)n * H);
    for (int64_t i = 0; i < n; i++) rt_thinker_embed_row(t, (int32_t)ids[i], emb + i * H);
    if (getenv("RT_NO_AUDIO"))
        fprintf(stderr, "RT_NO_AUDIO: leaving the learned <|audio|> embeddings in place\n");
    else
        for (int i = 0; i < atok; i++)
            memcpy(emb + (int64_t)(nh + i) * H, avec + (int64_t)i * H, sizeof(float) * H);

    cwick_sampler_cfg cfg;
    cwick_sampler_cfg_default(&cfg);
    cfg.temperature = 0.7f;
    cfg.top_p = 0.8f;
    cfg.top_k = 20;
    if (getenv("RT_GREEDY")) {
        cfg.temperature = 0.0f;
        cfg.top_k = 1;
    }
    cwick_sampler s;
    cwick_sampler_init(&s, &cfg, rt_thinker_vocab(t), 12345);

    double t4 = (double)cwick_prof_ns() / 1e9;
    if (rt_thinker_forward(t, emb, (int)n) != 0) {
        fprintf(stderr, "forward failed\n");
        return 1;
    }
    double t5 = (double)cwick_prof_ns() / 1e9;
    printf("prefill: %lld tokens in %.2f s (%.2f tok/s)\n", (long long)n, t5 - t4,
           (double)n / (t5 - t4));
    if (getenv("RT_TOP")) {
        /* Ranking on a copy: the sampler masks the live logits in place. An
         * immediate end-of-turn is the fingerprint of a prompt the model reads
         * as already finished, so the top rows are worth printing before any
         * sampling happens. */
        int V = rt_thinker_vocab(t);
        float *cp = (float *)malloc(sizeof(float) * (size_t)V);
        memcpy(cp, rt_thinker_logits(t), sizeof(float) * (size_t)V);
        for (int k = 0; k < 8; k++) {
            int best = -1;
            float bv = -INFINITY;
            for (int v = 0; v < V; v++)
                if (cp[v] > bv) { bv = cp[v]; best = v; }
            printf("top%d %6d %+.6f\n", k, best, bv);
            cp[best] = -INFINITY;
        }
        printf("EOS(151645)=%+.6f  EOT(151643)=%+.6f\n",
               rt_thinker_logits(t)[151645], rt_thinker_logits(t)[151643]);
        free(cp);
    }
    printf("answer: ");
    fflush(stdout);

    int produced = 0;
    for (int st = 0; st < max_tokens; st++) {
        const float *logits = rt_thinker_logits(t);
        uint32_t id = cwick_sampler_pick(&s, logits, rt_thinker_vocab(t));
        cwick_sampler_accept(&s, id);
        if (id == IM_END || id == 151643) break;
        char *piece = NULL;
        if (cwick_tokenizer_decode(tok, &id, 1, &piece, &err) == CWICK_OK && piece) {
            fputs(piece, stdout);
            fflush(stdout);
            free(piece);
        }
        produced++;
        rt_thinker_embed_row(t, (int32_t)id, emb);
        if (rt_thinker_forward(t, emb, 1) != 0) break;
    }
    double t6 = (double)cwick_prof_ns() / 1e9;
    printf("\ndecode: %d tokens in %.2f s (%.2f tok/s)\n", produced, t6 - t5,
           produced > 0 ? (double)produced / (t6 - t5) : 0.0);
    printf("total: %.2f s (ear %.3f s, prefill %.2f s)\n", t6 - t2, t3 - t2, t5 - t4);

    free(emb);
    free(ids);
    free(hid);
    free(tid);
    free(avec);
    free(buf);
    free(pcm);
    cwick_sampler_free(&s);
    cwick_tokenizer_free(tok);
    rt_ear_close(ear);
    rt_thinker_close(t);
    return 0;
}
