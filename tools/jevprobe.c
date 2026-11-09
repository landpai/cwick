/*
 * jevprobe — what the decision endpoint can read out of one checkpoint's
 * tokenizer.
 *
 * The decision endpoint never generates: it reads the distribution over the
 * option letters at the answer position. That only works if each letter is a
 * single token, and whether it is depends on the tokenizer, not on this code.
 * This tool loads the tokenizer alone (no weights, no Metal, a few seconds)
 * and prints the answer per family, so a new checkpoint can be checked before
 * anyone points a game at it.
 *
 *   ./tools/jevprobe <model-root> [model_type]
 *
 * model_type selects the prompt grammar (see jev_adapter_default): the strings
 * "deepseek", "minimax" or anything else for the GLM grammar. Two families do
 * not share a chat template, which is exactly what the adapter exists to keep
 * straight.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cwick.h"
#include "jev.h"
#include "log.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model-root> [model_type]\n", argv[0]);
        return 2;
    }
    cwick_log_config(false, false, false);
    fprintf(stderr, "tokenizer: loading %s\n", argv[1]);

    cwick_model_spec spec;
    memset(&spec, 0, sizeof(spec));
    snprintf(spec.model_type, sizeof(spec.model_type), "%s",
             argc > 2 && argv[2][0] ? argv[2] : "glm");
    spec.vocab_size = 200000;   /* only used as a floor by the loader */

    cwick_tokenizer *tok = NULL;
    cwick_error err;
    cwick_error_clear(&err);
    if (cwick_tokenizer_load(argv[1], &spec, &tok, &err) != CWICK_OK) {
        fprintf(stderr, "tokenizer: %s\n", cwick_error_message(&err));
        return 1;
    }

    jev_adapter a;
    char perr[256];
    int rc = jev_probe(&spec, tok, &a, perr, sizeof(perr));
    printf("root      : %s\n", argv[1]);
    printf("family    : %s\n", a.family);
    printf("prompt    : %s\n", a.prompt);
    printf("letters   : %d/%d%s\n", a.letters, JEV_MAX_OPTIONS,
           a.first_missing < 0 ? "" : "  (first gap above)");
    printf("spelling  : %s\n", a.space_lead ? "\" A\" (space-led)" : "\"A\" (bare)");
    if (rc != 0) {
        printf("probe     : FAILED — %s\n", perr);
        cwick_tokenizer_free(tok);
        return 1;
    }
    printf("\nletter  space-led  bare  one-token\n");
    for (int i = 0; i < JEV_MAX_OPTIONS; i++) {
        char sp[3] = { ' ', (char)('A' + i), '\0' };
        char bare[2] = { (char)('A' + i), '\0' };
        int64_t sn = -1, bn = -1;
        uint32_t *ids = NULL;
        cwick_error e;
        cwick_error_clear(&e);
        if (cwick_tokenizer_encode(tok, sp, &ids, &sn, &e) != CWICK_OK) sn = -1;
        free(ids);
        ids = NULL;
        cwick_error_clear(&e);
        if (cwick_tokenizer_encode(tok, bare, &ids, &bn, &e) != CWICK_OK) bn = -1;
        free(ids);
        printf("  %c       %5lld     %5lld    %s\n", (char)('A' + i),
               (long long)sn, (long long)bn,
               (sn == 1 || bn == 1) ? "yes" : "NO");
    }
    cwick_tokenizer_free(tok);
    return 0;
}
