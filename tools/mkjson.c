/* mkjson.c — build a /v1/chat/completions body from text files.
 *
 *   mkjson --out body.json --model glm-5.3-flash --max-tokens 8 \
 *          --user prompt.txt [--user extra.txt ...]
 *
 * Each --user file becomes one {"role":"user"} message. The point of having
 * this instead of a shell here-doc is that a 100 KB prompt has to be JSON
 * escaped, and the escape has to be exact or the server sees a different
 * prompt than the one whose timing was measured. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void json_escape(FILE *out, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"':  fputs("\\\"", out); break;
        case '\\': fputs("\\\\", out); break;
        case '\n': fputs("\\n", out); break;
        case '\r': fputs("\\r", out); break;
        case '\t': fputs("\\t", out); break;
        case '\b': fputs("\\b", out); break;
        case '\f': fputs("\\f", out); break;
        default:
            if (c < 0x20) fprintf(out, "\\u%04x", c);
            else fputc(c, out);
        }
    }
}

static char *slurp(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "mkjson: cannot open %s\n", path); return NULL; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0) { fclose(f); return NULL; }
    rewind(f);
    char *buf = malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[got] = '\0';
    *len_out = got;
    return buf;
}

int main(int argc, char **argv) {
    const char *out_path = NULL, *model = "glm-5.3-flash";
    const char *files[64];
    const char *roles[64];
    size_t n_msg = 0;
    long max_tokens = 8;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--max-tokens") && i + 1 < argc)
            max_tokens = atol(argv[++i]);
        else if ((!strcmp(argv[i], "--user") || !strcmp(argv[i], "--assistant")) &&
                 i + 1 < argc && n_msg < sizeof(files) / sizeof(files[0])) {
            roles[n_msg] = strcmp(argv[i], "--user") ? "assistant" : "user";
            files[n_msg++] = argv[++i];
        } else {
            fprintf(stderr, "usage: mkjson --out F --model M --max-tokens N"
                            " (--user|--assistant) FILE [...]\n");
            return 2;
        }
    }
    if (!out_path || n_msg == 0) {
        fprintf(stderr, "mkjson: need --out and at least one message\n");
        return 2;
    }

    FILE *out = fopen(out_path, "wb");
    if (!out) { fprintf(stderr, "mkjson: cannot write %s\n", out_path); return 1; }
    fprintf(out, "{\"model\":\"%s\",\"messages\":[", model);
    for (size_t i = 0; i < n_msg; i++) {
        size_t n = 0;
        char *txt = slurp(files[i], &n);
        if (!txt) { fclose(out); return 1; }
        if (i) fputc(',', out);
        fprintf(out, "{\"role\":\"%s\",\"content\":\"", roles[i]);
        json_escape(out, txt, n);
        fputs("\"}", out);
        free(txt);
    }
    fprintf(out, "],\"max_tokens\":%ld,\"stream\":false}", max_tokens);
    fclose(out);
    return 0;
}
