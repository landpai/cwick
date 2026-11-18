/* rtthink.c — text-only smoke test for the realtime decoder.
 *
 *   tools/rtthink <thinker_dir> "<user text>" [max_tokens]
 *
 * Builds the prompt the way the served interface does (system turn, user
 * turn, assistant turn with the thinking block pre-closed so the model
 * answers instead of reasoning), prefills it, then samples. This exercises the
 * whole text path -- tokenizer, FP8 projections, qk-norm, RoPE, KV cache,
 * lm_head -- without the audio tower, which is what makes it useful: if the
 * answer is coherent, everything under the audio injection point is right. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cwick.h"
#include "log.h"
#include "rt_thinker.h"
#include "sample.h"

static const char *SYS_ZH =
    "你是智子（Sophon），AI降临派（Adventists AI）训练的 Sophon 系列模型，能听、能看，也能说。"
    "你会直接听到用户的语音，你的回复会被直接合成为语音说出来。请像真人对话一样，用自然的口语一两句话回应"
    "用户，注意用户的情绪。只写要说出口的话，不要写括号说明、表情或格式符号；数字、日期、金额、符号和英文"
    "缩写都直接写成口语念法。需要时可以用声音标签表现笑声、咳嗽和重读。";

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <thinker_dir> <text> [max_tokens]\n", argv[0]);
        return 2;
    }
    cwick_log_config(false, false, false);
    int max_tokens = argc > 3 ? atoi(argv[3]) : 48;
    cwick_error err;
    memset(&err, 0, sizeof(err));

    /* RT_BACKEND=cpu forces the Accelerate path, which shares nothing with the
     * Metal kernels -- the two are the reference for each other. */
    const char *bs = getenv("RT_BACKEND");
    rt_backend backend = RT_BACKEND_AUTO;
    if (bs && bs[0] == 'c') backend = RT_BACKEND_CPU;
    else if (bs && bs[0] == 'm') backend = RT_BACKEND_METAL;
    rt_thinker *t = rt_thinker_open(argv[1], backend, 2048, &err);
    if (!t) {
        fprintf(stderr, "open failed: %s\n", err.message);
        return 1;
    }
    cwick_model_spec spec;
    memset(&spec, 0, sizeof(spec));
    cwick_tokenizer *tok = NULL;
    if (cwick_tokenizer_load(argv[1], &spec, &tok, &err) != CWICK_OK) {
        fprintf(stderr, "tokenizer failed: %s\n", err.message);
        return 1;
    }

    char prompt[8192];
    snprintf(prompt, sizeof(prompt),
             "<|im_start|>system\n%s<|im_end|>\n"
             "<|im_start|>user\n%s<|im_end|>\n"
             "<|im_start|>assistant\n<think>\n\n</think>\n\n",
             SYS_ZH, argv[2]);
    uint32_t *ids = NULL;
    int64_t n = 0;
    const char *single = getenv("RT_SINGLE_ID");
    if (single) {
        /* Explicit ids, comma separated, so a hand-built sequence can be fed
         * without the tokenizer. Position 0 alone isolates every projection
         * (RoPE is the identity, attention is V); two ids add RoPE at pos 1
         * and a real causal softmax over two keys. */
        int cnt = 1;
        for (const char *p = single; *p; p++)
            if (*p == ',') cnt++;
        ids = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)cnt);
        n = 0;
        const char *p = single;
        while (*p && n < cnt) {
            ids[n++] = (uint32_t)strtol(p, (char **)&p, 10);
            if (*p == ',') p++;
        }
        printf("single token: %lld ids\n", (long long)n);
    } else if (cwick_tokenizer_encode(tok, prompt, &ids, &n, &err) != CWICK_OK) {
        fprintf(stderr, "encode failed: %s\n", err.message);
        return 1;
    }
    printf("prompt tokens: %lld\n", (long long)n);
    if (getenv("RT_DUMP")) {
        /* Round-tripping the ids is the cheapest check that the specials
         * survived encoding: a literal "<|im_start|>" changes the whole
         * context and every downstream number looks inexplicably wrong. */
        char *back = NULL;
        if (cwick_tokenizer_decode(tok, (const uint32_t *)ids, (int)n, &back, &err) ==
                CWICK_OK && back) {
            printf("--- prompt round-trip ---\n%s\n---\n", back);
            free(back);
        }
        printf("ids:");
        for (int64_t i = 0; i < n && i < 200; i++) printf(" %u", ids[i]);
        printf("\n");
        if (getenv("RT_DUMPONLY")) return 0;
    }

    int H = rt_thinker_hidden(t);
    float *emb = (float *)malloc(sizeof(float) * (size_t)n * H);
    for (int64_t i = 0; i < n; i++) rt_thinker_embed_row(t, (int32_t)ids[i], emb + i * H);

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

    double t0 = (double)cwick_prof_ns() / 1e9;
    if (rt_thinker_forward(t, emb, (int)n) != 0) {
        fprintf(stderr, "forward failed\n");
        return 1;
    }
    double t1 = (double)cwick_prof_ns() / 1e9;
    printf("prefill: %lld tokens in %.2f s (%.2f tok/s)\n", (long long)n, t1 - t0,
           (double)n / (t1 - t0));

    double td = t1;
    if (getenv("RT_TOP")) {
        /* The first token's ranking separates a wrong kernel from a wrong
         * template: identical ids and logits on both backends means the math
         * agrees and the fault is elsewhere. Ranking runs on a copy so the
         * sampler still sees the real distribution. */
        const float *lg = rt_thinker_logits(t);
        int V = rt_thinker_vocab(t);
        float *cp = (float *)malloc(sizeof(float) * (size_t)V);
        memcpy(cp, lg, sizeof(float) * (size_t)V);
        for (int k = 0; k < 8; k++) {
            int best = -1;
            float bv = -INFINITY;
            for (int v = 0; v < V; v++)
                if (cp[v] > bv) { bv = cp[v]; best = v; }
            printf("top%d %6d %+.6f\n", k, best, bv);
            cp[best] = -INFINITY;
        }
        printf("\n");
        free(cp);
    }
    for (int step = 0; step < max_tokens; step++) {
        float *logits = (float *)rt_thinker_logits(t);
        uint32_t id = cwick_sampler_pick(&s, logits, rt_thinker_vocab(t));
        cwick_sampler_accept(&s, id);
        char *piece = NULL;
        if (cwick_tokenizer_decode(tok, &id, 1, &piece, &err) == CWICK_OK && piece) {
            fputs(piece, stdout);
            fflush(stdout);
            free(piece);
        }
        if (id == 151645 || id == 151643) break;
        rt_thinker_embed_row(t, (int32_t)id, emb);
        if (rt_thinker_forward(t, emb, 1) != 0) break;
    }
    double t2 = (double)cwick_prof_ns() / 1e9;
    if (max_tokens > 0)
        printf("\ndecode: %.2f s, %.2f tok/s (pos %d)\n", t2 - td,
               (double)max_tokens / (t2 - td), rt_thinker_pos(t));
    free(emb);
    free(ids);
    cwick_tokenizer_free(tok);
    cwick_sampler_free(&s);
    rt_thinker_close(t);
    return 0;
}
