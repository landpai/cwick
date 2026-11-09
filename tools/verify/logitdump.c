/* logitdump.c — dump the post-prefill logits of the real checkpoint.
 *
 * Used to A/B an inference-kernel change *numerically* rather than by eyeballing
 * generated text: two algebraically equivalent paths differ only in floating
 * point summation order, so their logits should agree to ~1e-6 relative and
 * certainly pick the same argmax. A real indexing bug shows up as an O(1)
 * relative difference.
 *
 *   ./logitdump <model-root> <out.bin> [ntok]
 *
 * The prompt is a fixed deterministic id sequence (no tokenizer) so the
 * comparison isolates the kernel, not the front end.
 */
#include "cwick.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <root> <out.bin> [ntok]\n", argv[0]);
        return 2;
    }
    const char *root = argv[1];
    const char *outp = argv[2];
    int64_t ntok = argc > 3 ? atoll(argv[3]) : 12;
    if (ntok < 1) ntok = 1;
    /* Long-context runs are the point of this tool: the cap is the engine's
     * own ceiling, not a token limit imposed here. */
    if (ntok > 200000) ntok = 200000;

    uint32_t *ids = malloc((size_t)ntok * sizeof(uint32_t));
    /* A spread of ids inside the vocabulary, deterministic across runs. */
    for (int64_t i = 0; i < ntok; i++) ids[i] = (uint32_t)(1000 + (i * 7919) % 150000);

    cwick_error err;
    cwick_error_clear(&err);
    cwick_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_root = root;
    opt.max_new_tokens = 1;
    opt.verbose = getenv("CWICK_VERBOSE") != NULL;
    /* Force the compute backend so a Metal change can be A/B'd numerically
     * against the CPU path on identical inputs. */
    {
        const char *be = getenv("CWICK_BACKEND");
        cwick_metal_force(be && strcmp(be, "metal") == 0 ? 1 : 0);
    }
    /* The released checkpoint defaults its KV window to index_topk (2048).
     * Ask for the whole prompt plus one decode slot so that a run longer than
     * the window actually exercises the DSA indexer instead of being clamped
     * and rejected. */
    {
        const char *ce = getenv("CWICK_DUMP_CONTEXT");
        opt.max_context = ce ? atoll(ce) : ntok + 8;
        if (opt.max_context < ntok + 1) opt.max_context = ntok + 1;
    }

    cwick_engine *eng = NULL;
    cwick_status st = cwick_engine_new(&opt, &eng, &err);
    if (st != CWICK_OK) {
        fprintf(stderr, "engine_new failed: %s\n", cwick_error_message(&err));
        return 1;
    }

    const int64_t V = cwick_engine_spec(eng)->vocab_size;
    float *logits = malloc((size_t)V * sizeof(float));
    st = cwick_engine_prefill_logits(eng, ids, ntok, logits, &err);
    if (st != CWICK_OK) {
        fprintf(stderr, "prefill failed: %s\n", cwick_error_message(&err));
        return 1;
    }

    FILE *f = fopen(outp, "wb");
    if (!f) { perror("fopen"); return 1; }
    int64_t hdr[2] = { V, ntok };
    fwrite(hdr, sizeof(hdr), 1, f);
    fwrite(logits, sizeof(float), (size_t)V, f);
    fclose(f);

    int64_t rank_idx[5];
    for (int k = 0; k < 5; k++) rank_idx[k] = -1;
    int64_t am = 0;
    for (int64_t i = 1; i < V; i++) if (logits[i] > logits[am]) am = i;
    double mn = 1e30, mx = -1e30;
    for (int64_t i = 0; i < V; i++) {
        if (logits[i] < mn) mn = logits[i];
        if (logits[i] > mx) mx = logits[i];
    }
    printf("vocab=%lld ntok=%lld argmax=%lld logit=%.6f range=[%.4f, %.4f]\n",
           (long long)V, (long long)ntok, (long long)am, (double)logits[am], mn, mx);
    /* Top-5 by repeated scan: the margin between the top two is what matters
     * when two implementations disagree on the generated text. */
    printf("top5:");
    for (int k = 0; k < 5; k++) {
        int64_t best = -1;
        for (int64_t i = 0; i < V; i++) {
            bool used = false;
            for (int j = 0; j < k; j++) if (rank_idx[j] == i) used = true;
            if (used) continue;
            if (best < 0 || logits[i] > logits[best]) best = i;
        }
        rank_idx[k] = best;
        printf(" %lld:%.4f", (long long)best, (double)logits[best]);
    }
    printf("\n");

    free(logits);
    free(ids);
    cwick_engine_free(eng);
    return 0;
}
