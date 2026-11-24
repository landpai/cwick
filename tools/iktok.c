/* iktok.c — Inkling tokenizer golden-vector probe.
 *
 * Usage: iktok MODEL_ROOT TEXT...
 * Prints one line per input with the encoded IDs and the piece for each ID.
 * Kept as a plain C tool so the checkpoint tokenizer can be checked without
 * starting the server or loading model weights.
 */
#include "cwick.h"
#include "ik.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { float value; uint32_t id; } ik_logit;

static int cmp_logits(const void *a, const void *b) {
    const ik_logit *x = (const ik_logit *)a, *y = (const ik_logit *)b;
    if (x->value < y->value) return 1;
    if (x->value > y->value) return -1;
    return (x->id > y->id) - (x->id < y->id);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL_ROOT [--logits] TEXT...\n", argv[0]);
        return 2;
    }
    bool want_logits = strcmp(argv[2], "--logits") == 0;
    const char *root = argv[1];
    char cfg[1152];
    cwick_error err;
    cwick_error_clear(&err);
    cwick_path_join(cfg, sizeof(cfg), root, "config.json");
    cwick_model_spec spec;
    memset(&spec, 0, sizeof(spec));
    if (cwick_model_spec_load(cfg, &spec, &err) != CWICK_OK) {
        fprintf(stderr, "spec: %s\n", cwick_error_message(&err));
        return 1;
    }
    cwick_tokenizer *tok = NULL;
    if (cwick_tokenizer_load(root, &spec, &tok, &err) != CWICK_OK) {
        fprintf(stderr, "tokenizer: %s\n", cwick_error_message(&err));
        return 1;
    }
    ik_model model;
    float *logits = NULL;
    int first = want_logits ? 3 : 2;
    if (want_logits) {
        memset(&model, 0, sizeof(model));
        if (ik_model_open(&model, root, 512, &err) != 0) {
            fprintf(stderr, "model: %s\n", cwick_error_message(&err));
            cwick_tokenizer_free(tok);
            return 1;
        }
        logits = (float *)malloc((size_t)model.c.vocab * sizeof(float));
        if (!logits) return 2;
    }
    for (int a = first; a < argc; a++) {
        uint32_t *ids = NULL;
        int64_t n = 0;
        if (cwick_tokenizer_encode(tok, argv[a], &ids, &n, &err) != CWICK_OK) {
            fprintf(stderr, "encode: %s\n", cwick_error_message(&err));
            continue;
        }
        printf("TEXT %s\nIDS ", argv[a]);
        for (int64_t i = 0; i < n; i++) printf("%u ", ids[i]);
        printf("\nPIECES ");
        for (int64_t i = 0; i < n; i++) {
            char *piece = NULL;
            if (cwick_tokenizer_decode(tok, ids + i, 1, &piece, &err) == CWICK_OK && piece) {
                printf("[%s]", piece);
                free(piece);
            } else {
                printf("[?]");
            }
        }
        printf("\n");
        if (want_logits) {
            int32_t *input = (int32_t *)malloc((size_t)(n ? n : 1) * sizeof(int32_t));
            if (!input) return 2;
            for (int64_t i = 0; i < n; i++) input[i] = (int32_t)ids[i];
            ik_model_reset(&model);
            if (ik_model_forward(&model, input, n, 0, logits, &err) != 0) {
                fprintf(stderr, "forward: %s\n", cwick_error_message(&err));
                free(input);
                free(ids);
                continue;
            }
            free(input);
            int64_t vocab = model.c.unpadded_vocab > 0 ? model.c.unpadded_vocab : model.c.vocab;
            ik_logit *ranked = (ik_logit *)malloc((size_t)vocab * sizeof(*ranked));
            if (!ranked) return 2;
            for (int64_t i = 0; i < vocab; i++) ranked[i] = (ik_logit){logits[i], (uint32_t)i};
            qsort(ranked, (size_t)vocab, sizeof(*ranked), cmp_logits);
            printf("TOP ");
            int64_t top = vocab < 20 ? vocab : 20;
            for (int64_t i = 0; i < top; i++) printf("%u:%.9g ", ranked[i].id, ranked[i].value);
            printf("\n");
            free(ranked);
        }
        free(ids);
    }
    cwick_tokenizer_free(tok);
    if (want_logits) {
        ik_model_close(&model);
        free(logits);
    }
    return 0;
}
