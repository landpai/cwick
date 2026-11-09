/*
 * ds_toktest.c — the DeepSeek tokenizer, checked against the checkpoint's
 * own vocabulary.
 *
 *   ds_toktest <checkpoint-root>
 *
 * Two kinds of check:
 *
 *   1. Known ids. A list of (text, id) pairs read out of tokenizer.json's
 *      model.vocab: if the pre-tokenizer splits at the wrong place, or the
 *      byte-level alphabet is wrong, these come back as different ids.
 *   2. Round trip. encode(text) then decode() must give the text back for
 *      every case, including the ones with CJK, digits, punctuation and
 *      mixed scripts where the chunk boundaries are least obvious.
 *
 * This is the only oracle available for the tokenizer: the reference
 * implementation needs a Python environment this project does not install.
 */
#include "cwick.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *text;
    int64_t id;
} known;

/* id values copied from the checkpoint's model.vocab (byte-level keys, so
 * " world" is stored as "Ġworld"). */
static const known CASES[] = {
    {"hello", 33310},
    {" world", 2058},
    {" the", 270},
    {" Hello", 45948},
    {"The", 671},
    {"\n\n", 271},
    {" out", 798},
    {" ", 223},
    {"!!", 6909},
    {" !", 4050},
    {"123", 6895},     /* three digits: one \p{N}{1,3} chunk */
    {"a", 67},
    {"A", 35},
    {".", 16},
    {",", 14},
};

static const char *RT[] = {
    "hello world",
    "The quick brown fox jumps over the lazy dog.",
    "  leading and   multiple   spaces  ",
    "line one\nline two\n\nline four",
    "tabs\tand\tvertical\vtabs",
    "digits 1234567890 and 42 and 007",
    "CJK mixed in: 你好世界 and more text",
    "Japanese: これはテストです",
    "punctuation!!! what?! (really) [yes] {no} <maybe>",
    "code: def foo(bar): return bar + 1",
    "emoji and symbols: + = ~ ^ | \\ /",
    "trailing space ",
    "\n",
    "a",
    "",
};

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <checkpoint-root> [--dump TEXT...]\n",
                argv[0]);
        return 2;
    }
    const char *root = argv[1];
    cwick_error err = {0};

    cwick_model_spec spec;
    memset(&spec, 0, sizeof(spec));
    spec.vocab_size = 129280;
    cwick_tokenizer *tok = NULL;
    if (cwick_tokenizer_load(root, &spec, &tok, &err) != CWICK_OK) {
        fprintf(stderr, "load failed: %s\n", err.message);
        return 1;
    }
    fprintf(stderr, "vocab=%zu merges=%zu specials=%zu pretok_ds=%d\n",
            tok->vocab_size, tok->merge_count, tok->special_count,
            (int)tok->pretok_ds);

    if (argc > 2 && strcmp(argv[2], "--dump") == 0) {
        for (int i = 3; i < argc; i++) {
            uint32_t *ids = NULL;
            int64_t n = 0;
            if (cwick_tokenizer_encode(tok, argv[i], &ids, &n, &err) != CWICK_OK) {
                fprintf(stderr, "FAIL encode: %s\n", err.message);
                return 1;
            }
            printf("%s ->", argv[i]);
            for (int64_t k = 0; k < n; k++) printf(" %u", ids[k]);
            printf("  (%lld)\n", (long long)n);
            free(ids);
        }
        cwick_tokenizer_free(tok);
        return 0;
    }

    int fails = 0;

    for (size_t i = 0; i < sizeof(CASES) / sizeof(CASES[0]); i++) {
        uint32_t *ids = NULL;
        int64_t n = 0;
        if (cwick_tokenizer_encode(tok, CASES[i].text, &ids, &n, &err) != CWICK_OK) {
            fprintf(stderr, "FAIL encode %s: %s\n", CASES[i].text, err.message);
            fails++;
            continue;
        }
        /* A single-token expectation: the whole string must be one piece. */
        bool ok = (n == 1 && (int64_t)ids[0] == CASES[i].id);
        if (!ok) {
            fprintf(stderr, "FAIL \"%s\": got %lld token(s) [", CASES[i].text,
                    (long long)n);
            for (int64_t k = 0; k < n && k < 8; k++)
                fprintf(stderr, "%s%u", k ? " " : "", ids[k]);
            fprintf(stderr, "], expected id %lld\n",
                    (long long)CASES[i].id);
            fails++;
        }
        free(ids);
    }
    fprintf(stderr, "known-id cases: %zu checked, %d failed\n",
            sizeof(CASES) / sizeof(CASES[0]), fails);

    int rt_fails = 0;
    for (size_t i = 0; i < sizeof(RT) / sizeof(RT[0]); i++) {
        uint32_t *ids = NULL;
        int64_t n = 0;
        if (cwick_tokenizer_encode(tok, RT[i], &ids, &n, &err) != CWICK_OK) {
            fprintf(stderr, "FAIL rt encode %zu: %s\n", i, err.message);
            rt_fails++;
            continue;
        }
        char *back = NULL;
        if (cwick_tokenizer_decode(tok, ids, n, &back, &err) != CWICK_OK) {
            fprintf(stderr, "FAIL rt decode %zu: %s\n", i, err.message);
            free(ids);
            rt_fails++;
            continue;
        }
        if (strcmp(back, RT[i]) != 0) {
            fprintf(stderr, "FAIL round trip %zu:\n  in  %s\n  out %s\n", i,
                    RT[i], back);
            rt_fails++;
        }
        free(back);
        free(ids);
    }
    fprintf(stderr, "round trips: %zu checked, %d failed\n",
            sizeof(RT) / sizeof(RT[0]), rt_fails);

    cwick_tokenizer_free(tok);
    return (fails || rt_fails) ? 1 : 0;
}
