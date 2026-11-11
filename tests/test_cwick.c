/* test_cwick.c — end-to-end verification suite.
 *
 * Covers the objective requirements:
 *   1. model-contract validation (config.json, never-pruned quality guards);
 *   2. BF16 conversion and the NEON MXFP4 expert GEMV vs scalar reference;
 *   3. engine forward + greedy decode on the synthetic full model;
 *   4. OpenAI-compatible server: /v1/models, /v1/chat/completions,
 *      /v1/completions, SSE streaming, fail-closed 400s and busy 429.
 */
#include "cwick.h"
#include "json.h"
#include "tools.h"
#include "jev.h"
#include "mem.h"
#include "metrics.h"
#include "../src/log.h"

#include <arpa/inet.h>
#include <errno.h>
#include <math.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "../src/util.h"
#include "../src/webui.h"
#include "../src/logcap.h"
#include "../src/auth.h"
#include "../src/store.h"
#include "../src/models.h"
#include "../src/hub.h"
#include "../src/qr.h"
#include "pfx.h"
#include "../src/sample.h"

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
        g_fail = 1; \
    } else { \
        printf("ok: %s\n", msg); \
    } \
} while (0)

static char *make_tmpdir(void) {
    static char path[128];
    snprintf(path, sizeof(path), "/tmp/cwick-test-%ld", (long)getpid());
    if (cwick_dir_exists(path)) {
        char cmd[256];
        snprintf(cmd, sizeof(cmd), "rm -rf %s", path);
        system(cmd);
    }
    cwick_mkdir_p(path);
    return path;
}

/* ------------------------------------------------------------------ */
/* 1. BF16                                                             */
/* ------------------------------------------------------------------ */
static void test_bf16(void) {
    float vals[] = { 0.0f, -0.0f, 0.5f, 1.0f, 1.5f, -2.0f, 3.5f, 123.25f, 1e-5f };
    int ok = 1;
    for (size_t i = 0; i < sizeof(vals) / sizeof(vals[0]); i++) {
        uint16_t b = cwick_f32_to_bf16(vals[i]);
        float r = cwick_bf16_to_f32(b);
        float d = fabsf(r - vals[i]);
        float tol = fmaxf(1e-3f, fabsf(vals[i]) * 1e-2f);
        if (d > tol) { ok = 0; fprintf(stderr, "  bf16 %g -> %g\n", vals[i], r); }
    }
    CHECK(ok, "bf16 round-trip");
}

/* ------------------------------------------------------------------ */
/* 2. MXFP4 GEMV (NEON vs scalar reference)                            */
/* ------------------------------------------------------------------ */
static void test_mxfp4(void) {
    const int rows = 4, cols = 32;
    uint8_t p[rows * cols / 2];
    uint8_t s[rows * cols / 32];
    float x[cols], y_neon[rows], y_ref[rows];
    for (int i = 0; i < rows * cols / 2; i++) p[i] = (uint8_t)(i * 7 + 3);
    for (int i = 0; i < rows * cols / 32; i++) s[i] = (uint8_t)(120 + i * 3);
    for (int i = 0; i < cols; i++) x[i] = 0.5f * (float)(i % 7) - 1.0f;

    mxfp4_gemv_rows(p, s, x, y_neon, 0, rows, cols);
    mxfp4_gemv_rows_ref(p, s, x, y_ref, 0, rows, cols);
    int ok = 1;
    for (int r = 0; r < rows; r++) {
        float d = fabsf(y_neon[r] - y_ref[r]);
        if (d > 1e-3f) { ok = 0; fprintf(stderr, "  mxfp4 row %d neon=%g ref=%g\n", r, y_neon[r], y_ref[r]); }
    }
    CHECK(ok, "mxfp4 NEON kernel matches scalar reference");
}

/* The test above compares the SIMD kernel against `mxfp4_gemv_rows_ref`, and
 * both read the nibbles and the e2m1 table the same way -- so it passes even
 * if that shared interpretation is wrong. That failure mode is silent:
 * reversing the nibbles does not crash, it just makes every number wrong.
 *
 * This test pins the packing contract with an expectation computed by hand
 * from the written spec (see the MXFP4 comment block in tensor.c):
 *
 *   byte b covers columns (2i, 2i+1); the LOW nibble is the EVEN column
 *   code c -> {0, .5, 1, 1.5, 2, 3, 4, 6}[c & 7], negated when c & 8
 *   scale  -> 2^(s - 127)
 *
 * With x = 1 the dot product is the sum of the dequantized values, so the
 * arithmetic stays checkable on paper. */
static void test_mxfp4_spec(void) {
    const int rows = 2, cols = 64;            /* two 32-element groups */
    uint8_t p[rows * cols / 2], s[rows * cols / 32];
    float x[cols], y[rows];

    /* One 16-byte group is 32 values: the low nibbles of the 16 bytes are the
     * 16 EVEN columns, the high nibbles the 16 ODD ones.
     *
     * row 0, group 0: 0x10 -> even 0x0 (+0), odd 0x1 (+0.5)  => 16*0.5 = 8
     * row 0, group 1: 0x32 -> even 0x2 (+1), odd  0x3 (+1.5)  => 16*1 + 16*1.5 = 40
     *                 scale 128 => 2^(128-127) = 2                => 80
     * row 1, group 0: 0x8F -> even 0xF (-6), odd  0x8 (-0)   => 16*-6 = -96
     *                 scale 129 => 4                               => -384
     * row 1, group 1: 0x00 -> both nibbles 0 (+0)                 => 0 */
    memset(p, 0x10, 16);
    memset(p + 16, 0x32, 16);
    memset(p + 32, 0x8F, 16);
    memset(p + 48, 0x00, 16);
    s[0] = 127; s[1] = 128;
    s[2] = 129; s[3] = 127;
    for (int i = 0; i < cols; i++) x[i] = 1.0f;

    const float want[2] = { 8.0f + 80.0f, -384.0f + 0.0f };

    /* Check one 32-element group at a time first: with cols=32 the row stride
     * is 16 bytes and the scale stride is 1, so each call isolates exactly one
     * (nibble pair, scale) combination and a failure names which one broke. */
    float g[1];
    mxfp4_gemv_rows(p, s, x, g, 0, 1, 32);
    if (fabsf(g[0] - 8.0f) > 1e-4f)
        fprintf(stderr, "  group(0x10, s=127): got %g want 8\n", g[0]);
    CHECK(fabsf(g[0] - 8.0f) <= 1e-4f, "mxfp4 group: 0x10 at s=127 is 16*0.5 = 8");

    mxfp4_gemv_rows(p + 16, s + 1, x, g, 0, 1, 32);
    if (fabsf(g[0] - 80.0f) > 1e-4f)
        fprintf(stderr, "  group(0x32, s=128): got %g want 80\n", g[0]);
    CHECK(fabsf(g[0] - 80.0f) <= 1e-4f,
          "mxfp4 group: 0x32 at s=128 is (16*1 + 16*1.5)*2 = 80");

    mxfp4_gemv_rows(p + 32, s + 2, x, g, 0, 1, 32);
    if (fabsf(g[0] + 384.0f) > 1e-4f)
        fprintf(stderr, "  group(0x8F, s=129): got %g want -384\n", g[0]);
    CHECK(fabsf(g[0] + 384.0f) <= 1e-4f,
          "mxfp4 group: 0x8F at s=129 is 16*(-6)*4 = -384");

    mxfp4_gemv_rows(p, s, x, y, 0, rows, cols);
    int ok = 1;
    for (int r = 0; r < rows; r++) {
        if (fabsf(y[r] - want[r]) > 1e-4f) {
            ok = 0;
            fprintf(stderr, "  mxfp4 spec row %d got %g want %g "
                            "(nibble order or e8m0 scale is wrong)\n",
                    r, y[r], want[r]);
        }
    }
    CHECK(ok, "mxfp4 packing contract: low nibble = even column, scale = 2^(s-127)");

    /* And the reference must agree with the same contract, not just with the
     * kernel: that is what makes it usable as a packer oracle later. */
    memset(y, 0, sizeof(y));
    mxfp4_gemv_rows_ref(p, s, x, y, 0, rows, cols);
    ok = 1;
    for (int r = 0; r < rows; r++)
        if (fabsf(y[r] - want[r]) > 1e-4f) ok = 0;
    CHECK(ok, "mxfp4 scalar reference agrees with the packing contract");
}

/* ------------------------------------------------------------------ */
/* 3. JSON parser                                                      */
/* ------------------------------------------------------------------ */
static void test_json(void) {
    const char *doc = "{\"a\": 1, \"b\": [true, null, {\"c\": \"x\"}], \"d\": 2.5}";
    cwick_error err;
    cwick_error_clear(&err);
    json_value *root = cwick_json_parse(doc, strlen(doc), &err);
    CHECK(root != NULL, "json parse");
    if (root) {
        CHECK(cwick_json_int(cwick_json_get(root, "a"), 0) == 1, "json int field");
        CHECK(cwick_json_bool(cwick_json_get(root, "b"), false) == false, "json array is not bool");
        json_value *b = cwick_json_get(root, "b");
        CHECK(b && cwick_json_array_len(b) == 3, "json array length");
        json_value *inner = cwick_json_array_at(b, 2);
        const char *c = cwick_json_str(cwick_json_get(inner, "c"));
        CHECK(c && strcmp(c, "x") == 0, "json nested object");
        cwick_json_free(root);
    }
    cwick_error_clear(&err);
    const char *bad_object = "{\"bad\":1,}";
    CHECK(cwick_json_parse(bad_object, strlen(bad_object), &err) == NULL,
          "json rejects a trailing comma");
    cwick_error_clear(&err);
    const char *bad_array = "[1,]";
    CHECK(cwick_json_parse(bad_array, strlen(bad_array), &err) == NULL,
          "json rejects a trailing array comma");
    cwick_error_clear(&err);
    char deep[300];
    memset(deep, '[', sizeof(deep) - 1);
    deep[sizeof(deep) - 1] = '\0';
    CHECK(cwick_json_parse(deep, strlen(deep), &err) == NULL,
          "json rejects excessive nesting");
    cwick_error_clear(&err);
    const char *duplicate = "{\"a\":1,\"a\":2}";
    CHECK(cwick_json_parse(duplicate, strlen(duplicate), &err) == NULL,
          "json rejects duplicate object keys");
}

/* ------------------------------------------------------------------ */
/* 4. Model contract + engine                                          */
/* ------------------------------------------------------------------ */
static void test_contract_and_engine(void) {
    char *root = make_tmpdir();
    cwick_error err;
    cwick_error_clear(&err);
    CHECK(cwick_setup_synthetic(root, &err) == CWICK_OK, "synthetic setup");

    char cfg[1152];
    cwick_path_join(cfg, sizeof(cfg), root, "config.json");
    cwick_model_spec spec;
    CHECK(cwick_model_spec_load(cfg, &spec, &err) == CWICK_OK, "contract loads");
    CHECK(spec.num_experts_per_token == 2 && spec.num_experts == 8,
          "contract top-k preserved (never pruned)");
    CHECK(spec.kda_layers == 1 && spec.mla_layers == 1,
          "contract layer schedule (1 KDA + 1 MLA)");

    /* tamper: reduce top-k -> must reject */
    {
        char *text = NULL;
        size_t len = 0;
        cwick_read_file(cfg, &text, &len);
        char *rep = strstr(text, "\"num_experts_per_token\":2");
        if (rep) {
            memcpy(rep, "\"num_experts_per_token\":1", 26);
            char tampered[1152];
            cwick_path_join(tampered, sizeof(tampered), root, "config-tampered.json");
            cwick_write_file(tampered, text, strlen(text));
            cwick_model_spec bad;
            cwick_status st = cwick_model_spec_load(tampered, &bad, &err);
            CHECK(st != CWICK_OK, "reduced top-k rejected (never-pruned guard)");
            unlink(tampered);
        }
        free(text);
    }

    /* An architecture without kernels is refused by name, before the
     * geometry parser reports an unrelated missing field. A nested config
     * whose inner pad_token_id is null used to produce the misleading
     * "config is missing/invalid pad_token_id" even with a valid outer one. */
    {
        char foreign[1152];
        cwick_path_join(foreign, sizeof(foreign), root, "config-foreign.json");
        const char *fj =
            "{\"model_type\":\"qwen3_5\",\"pad_token_id\":248055,"
            "\"text_config\":{\"model_type\":\"qwen3_5_text\","
            "\"pad_token_id\":null,\"dtype\":\"bfloat16\"}}";
        CHECK(cwick_write_file(foreign, fj, strlen(fj)) == 0,
              "contract: a foreign nested config is written");
        cwick_model_spec fspec;
        cwick_status st = cwick_model_spec_load(foreign, &fspec, &err);
        const char *why = cwick_error_message(&err);
        CHECK(st == CWICK_ERR_UNSUPPORTED &&
              strstr(why, "unsupported model_type") != NULL &&
              strstr(why, "qwen3_5_text") != NULL,
              "contract: unknown architecture is refused by name, not by a GLM-only field");
        cwick_model_spec_free(&fspec);
        unlink(foreign);
    }

    /* engine: generate deterministically */
    cwick_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_root = root;
    opt.max_new_tokens = 8;
    cwick_engine *eng = NULL;
    CHECK(cwick_engine_new(&opt, &eng, &err) == CWICK_OK, "engine new");
    if (eng) {
        cwick_tokenizer *tok = NULL;
        char tdir[1152];
        cwick_path_join(tdir, sizeof(tdir), root, "tokenizer");
        CHECK(cwick_tokenizer_load(tdir, cwick_engine_spec(eng), &tok, &err) == CWICK_OK,
              "tokenizer (byte fallback)");
        if (tok) {
            uint32_t *ids = NULL;
            int64_t nids = 0;
            CHECK(cwick_chat_encode(tok, "You are helpful.", "Hello", &ids, &nids, &err) == CWICK_OK,
                  "chat encode");
            cwick_gen_result r1, r2;
            memset(&r1, 0, sizeof(r1));
            memset(&r2, 0, sizeof(r2));
            cwick_engine_reset(eng, &err);
            CHECK(cwick_engine_generate(eng, ids, nids, 8, false, &r1, &err) == CWICK_OK,
                  "generate #1");
            cwick_engine_reset(eng, &err);
            CHECK(cwick_engine_generate(eng, ids, nids, 8, false, &r2, &err) == CWICK_OK,
                  "generate #2");
            int same = r1.n_tokens == r2.n_tokens;
            if (same) {
                for (int64_t i = 0; i < r1.n_tokens; i++)
                    if (r1.tokens[i] != r2.tokens[i]) { same = 0; break; }
            }
            CHECK(same, "greedy generation is reproducible");
            CHECK(r1.n_tokens > 0, "generated at least one token");
            char *text = NULL;
            CHECK(cwick_tokenizer_decode(tok, r1.tokens, r1.n_tokens, &text, &err) == CWICK_OK,
                  "decode generated tokens");
            if (text) {
                printf("  generated: \"%.40s\"\n", text);
                free(text);
            }
            cwick_gen_result_free(&r1);
            cwick_gen_result_free(&r2);
            free(ids);
            cwick_tokenizer_free(tok);
        }
        cwick_engine_free(eng);
    }
}

/* ------------------------------------------------------------------ */
/* 5. OpenAI-compatible server                                         */
/* ------------------------------------------------------------------ */

static int http_request(int port, const char *method, const char *path,
                        const char *body, char *out, size_t outcap) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) { close(fd); return -1; }

    char req[8192];
    snprintf(req, sizeof(req),
             "%s %s HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: %zu\r\n"
             "Content-Type: application/json\r\nConnection: close\r\n\r\n%s",
             method, path, body ? strlen(body) : 0, body ? body : "");
    size_t rlen = strlen(req);
    size_t off = 0;
    while (off < rlen) {
        ssize_t w = write(fd, req + off, rlen - off);
        if (w <= 0) { close(fd); return -1; }
        off += (size_t)w;
    }
    size_t n = 0;
    for (;;) {
        if (n >= outcap - 1) { close(fd); return -1; }
        ssize_t r = read(fd, out + n, outcap - 1 - n);
        if (r <= 0) break;
        n += (size_t)r;
    }
    out[n] = '\0';
    close(fd);
    return 0;
}

static volatile int g_server_stop = 0;

typedef struct {
    cwick_engine *eng;
    int port;
    const char *store_path;   /* NULL = the built-in default */
} server_arg;

static void *server_thread(void *argp) {
    server_arg *sa = argp;
    cwick_server_options opts;
    memset(&opts, 0, sizeof(opts));
    opts.host = "127.0.0.1";
    opts.port = sa->port;
    opts.max_tokens = 64;
    opts.store_path = sa->store_path;
    cwick_error err;
    cwick_error_clear(&err);
    cwick_server_run(sa->eng, &opts, &g_server_stop, &err);
    return NULL;
}

/* Body of an HTTP response = everything after the header block. */
static const char *http_body(const char *resp) {
    const char *p = strstr(resp, "\r\n\r\n");
    return p ? p + 4 : resp;
}

/* Concatenate the text of an SSE chat stream's deltas (reasoning then content,
 * in arrival order) and return how many text-bearing frames arrived. An
 * incremental stream yields one frame per token; a buffered one yields a
 * single frame holding the whole answer. */
static int sse_collect_text(const char *body, char *text, size_t cap) {
    size_t tlen = 0;
    int frames = 0;
    if (cap) text[0] = '\0';
    const char *p = body;
    while ((p = strstr(p, "data: ")) != NULL) {
        p += 6;
        const char *end = strstr(p, "\n\n");
        if (!end) break;
        if (strncmp(p, "[DONE]", 6) != 0) {
            cwick_error e;
            cwick_error_clear(&e);
            json_value *root = cwick_json_parse(p, (size_t)(end - p), &e);
            if (root) {
                json_value *ch = cwick_json_get(root, "choices");
                json_value *c0 = ch ? cwick_json_array_at(ch, 0) : NULL;
                json_value *d = c0 ? cwick_json_get(c0, "delta") : NULL;
                if (d) {
                    static const char *fields[3] = { "reasoning_content", "content", "text" };
                    for (int fi = 0; fi < 3; fi++) {
                        const char *s = cwick_json_str(cwick_json_get(d, fields[fi]));
                        if (!s || !*s) continue;
                        frames++;
                        size_t n = strlen(s);
                        if (tlen + n + 1 < cap) {
                            memcpy(text + tlen, s, n);
                            tlen += n;
                            text[tlen] = '\0';
                        }
                    }
                }
                cwick_json_free(root);
            }
        }
        p = end;
    }
    return frames;
}

/* message.<field> of a non-stream chat body (malloc'd, NULL when absent). */
static char *json_message_field(const char *body, const char *field) {
    cwick_error e;
    cwick_error_clear(&e);
    json_value *root = cwick_json_parse(body, strlen(body), &e);
    if (!root) return NULL;
    json_value *ch = cwick_json_get(root, "choices");
    json_value *c0 = ch ? cwick_json_array_at(ch, 0) : NULL;
    json_value *msg = c0 ? cwick_json_get(c0, "message") : NULL;
    const char *s = msg ? cwick_json_str(cwick_json_get(msg, field)) : NULL;
    char *out = s ? cwick_strdup(s) : NULL;
    cwick_json_free(root);
    return out;
}

static void test_auth(void) {
    cwick_auth *a = cwick_auth_new("correct horse battery staple", 3600);
    CHECK(a != NULL, "auth: create with an explicit password");
    CHECK(cwick_auth_generated_password(a) == NULL,
          "auth: an explicit password is not reported as generated");
    CHECK(!cwick_auth_validate(a, "not-a-token"), "auth: bogus token rejected");
    CHECK(cwick_auth_login(a, "wrong") == NULL, "auth: wrong password rejected");
    char *tok = cwick_auth_login(a, "correct horse battery staple");
    CHECK(tok != NULL, "auth: correct password mints a session");
    CHECK(tok && cwick_auth_validate(a, tok), "auth: minted token validates");
    CHECK(tok && !cwick_auth_validate(a, "deadbeef"), "auth: other tokens rejected");
    cwick_auth_logout(a, tok);
    CHECK(tok && !cwick_auth_validate(a, tok), "auth: logout invalidates the session");
    free(tok);
    cwick_auth_free(a);

    a = cwick_auth_new("", 3600);   /* empty = generate */
    CHECK(a != NULL, "auth: generated-password mode starts");
    const char *gen = cwick_auth_generated_password(a);
    CHECK(gen && strlen(gen) >= 10, "auth: generated password is printable");
    tok = gen ? cwick_auth_login(a, gen) : NULL;
    CHECK(tok != NULL, "auth: generated password can log in");
    free(tok);
    cwick_auth_free(a);

    a = cwick_auth_new("lock-test", 3600);
    for (int i = 0; i < 5; i++) {
        char *bad = cwick_auth_login(a, "nope");
        free(bad);
    }
    CHECK(cwick_auth_retry_after(a) > 0, "auth: repeated failures lock briefly");
    cwick_auth_free(a);
    CHECK(cwick_auth_token_equal("abc", "abc") &&
          !cwick_auth_token_equal("abc", "abd"),
          "auth: token comparison is exact");
}

/* ---------------------------- app storage ---------------------------- */

typedef struct {
    int n, saw_score, saw_board;
} store_kv_probe;

static void store_kv_cb(const char *key, const char *value, int64_t ts,
                        void *user) {
    store_kv_probe *p = (store_kv_probe *)user;
    (void)value; (void)ts;
    p->n++;
    if (strcmp(key, "score") == 0) p->saw_score = 1;
    if (strcmp(key, "board") == 0) p->saw_board = 1;
}

typedef struct {
    int n, saw_ai, saw_human;
    int64_t seq0, seq_last;
} store_ev_probe;

static void store_ev_cb(const cwick_store_event *e, void *user) {
    store_ev_probe *p = (store_ev_probe *)user;
    if (p->n == 0) p->seq0 = e->seq;
    p->seq_last = e->seq;
    p->n++;
    if (strcmp(e->kind, "ai") == 0) p->saw_ai = 1;
    if (strcmp(e->kind, "human") == 0) p->saw_human = 1;
}

typedef struct {
    int n;
    int64_t moves;
    char outcome[64];
} store_game_probe;

static void store_game_cb(const cwick_store_game *g, void *user) {
    store_game_probe *p = (store_game_probe *)user;
    p->n++;
    p->moves = g->moves;
    snprintf(p->outcome, sizeof(p->outcome), "%s", g->outcome);
}

static void test_store(void) {
    char *root = make_tmpdir();
    char path[512], path2[512];
    snprintf(path, sizeof(path), "%s/store.db", root);
    snprintf(path2, sizeof(path2), "%s/from-env.db", root);
    cwick_error err;
    cwick_error_clear(&err);

    /* The file-vs-directory decision that picks the layout: a path that exists
     * as a file, or one ending in a database suffix, is the legacy shared
     * database. Everything else is a directory holding <app>.db files. */
    CHECK(cwick_store_path_is_file(path), "store: a .db path is the shared file");
    snprintf(path2, sizeof(path2), "%s/dirish", root);
    CHECK(!cwick_store_path_is_file(path2),
          "store: a path without a suffix is a directory");
    CHECK(!cwick_store_path_is_file("/tmp/cwick-test-does-not-exist-xyz"),
          "store: a missing path with no suffix is a directory");
    snprintf(path2, sizeof(path2), "%s/future.sqlite3", root);
    CHECK(cwick_store_path_is_file(path2),
          "store: a not-yet-created .sqlite3 path is still a file");
    snprintf(path2, sizeof(path2), "%s/from-env.db", root);

    cwick_store *s = cwick_store_open(path, &err);
    CHECK(s != NULL, "store: opens a new sqlite file");
    if (!s) return;
    CHECK(cwick_store_path(s) && strstr(cwick_store_path(s), "store.db") != NULL,
          "store: reports the file it opened");

    /* The only rule the store imposes on a value: it has to be JSON, because
     * the server hands it back inside its own envelope. */
    CHECK(!cwick_store_kv_put(s, "tictactoe", "bad", "not json", &err),
          "store: a non-JSON value is refused");
    CHECK(!cwick_store_kv_put(s, "tictactoe", "bad", "", &err),
          "store: an empty value is refused");
    CHECK(cwick_store_kv_put(s, "tictactoe", "score", "{\"win\":3}", &err),
          "store: a JSON value is accepted");
    char *v = cwick_store_kv_get(s, "tictactoe", "score");
    CHECK(v && strstr(v, "\"win\":3") != NULL,
          "store: a value round-trips byte for byte");
    free(v);
    CHECK(cwick_store_kv_get(s, "tictactoe", "absent") == NULL,
          "store: an absent key reads back as absent");
    CHECK(!cwick_store_kv_put(s, "not an id", "k", "1", &err) &&
          !cwick_store_kv_put(s, "../etc", "k", "1", &err),
          "store: an app id that is not an identifier is refused");
    CHECK(cwick_store_id_ok("tictactoe") && cwick_store_id_ok("App_2") &&
          !cwick_store_id_ok("") && !cwick_store_id_ok("a/b") &&
          !cwick_store_id_ok(NULL),
          "store: the id check admits identifiers and nothing else");

    CHECK(cwick_store_kv_put(s, "tictactoe", "score", "{\"win\":4}", &err),
          "store: putting an existing key overwrites it");
    CHECK(cwick_store_kv_put(s, "tictactoe", "board", "[1,2,3]", &err),
          "store: a second key is stored alongside the first");
    store_kv_probe kp;
    memset(&kp, 0, sizeof(kp));
    cwick_store_kv_each(s, "tictactoe", store_kv_cb, &kp);
    CHECK(kp.n == 2 && kp.saw_score && kp.saw_board,
          "store: enumeration sees every key of that app");
    v = cwick_store_kv_get(s, "tictactoe", "score");
    CHECK(v && strstr(v, "\"win\":4") != NULL,
          "store: the overwrite is what reads back");
    free(v);
    CHECK(cwick_store_kv_del(s, "tictactoe", "board", &err) &&
          cwick_store_kv_get(s, "tictactoe", "board") == NULL,
          "store: a deleted key is gone");

    /* A game with three steps, one of them the model's. */
    CHECK(cwick_store_game_put(s, "tictactoe", "g1", 1000, NULL, 0, 0, &err),
          "store: a game can be opened before it is finished");
    bool steps_ok = true;
    for (int i = 0; i < 3; i++) {
        char payload[128];
        snprintf(payload, sizeof(payload), "{\"move\":%d,\"cell\":%d}", i, i + 1);
        steps_ok = cwick_store_event_add(s, "tictactoe", "g1", i,
                                        i % 2 ? "ai" : "human", payload, &err) &&
                   steps_ok;
    }
    CHECK(steps_ok, "store: every step of a game is appended");
    CHECK(!cwick_store_event_add(s, "tictactoe", "g1", 3, "ai", "{oops", &err),
          "store: a malformed payload is refused");
    CHECK(cwick_store_game_put(s, "tictactoe", "g1", 1000, "ai_wins", 1100, 3,
                              &err),
          "store: a game can be closed with an outcome");
    store_ev_probe ep;
    memset(&ep, 0, sizeof(ep));
    cwick_store_events(s, "tictactoe", "g1", 0, store_ev_cb, &ep);
    CHECK(ep.n == 3, "store: a game's steps come back");
    CHECK(ep.seq0 == 0 && ep.seq_last == 2, "store: steps come back in order");
    CHECK(ep.saw_ai && ep.saw_human, "store: both sides' moves are kept");
    memset(&ep, 0, sizeof(ep));
    cwick_store_events(s, "tictactoe", NULL, 0, store_ev_cb, &ep);
    CHECK(ep.n == 3, "store: events can be read across every game");
    store_game_probe gp;
    memset(&gp, 0, sizeof(gp));
    cwick_store_games(s, "tictactoe", 0, store_game_cb, &gp);
    CHECK(gp.n == 1 && gp.moves == 3 && strcmp(gp.outcome, "ai_wins") == 0,
          "store: the finished game lists with its outcome and move count");

    cwick_store_stats st;
    memset(&st, 0, sizeof(st));
    CHECK(cwick_store_get_stats(s, &st), "store: stats read");
    CHECK(st.games == 1 && st.events == 3 && st.kv_rows == 1 && st.bytes > 0,
          "store: stats count rows and report the file size");

    CHECK(cwick_store_game_del(s, "tictactoe", "g1", &err),
          "store: a game can be deleted");
    memset(&ep, 0, sizeof(ep));
    cwick_store_events(s, "tictactoe", "g1", 0, store_ev_cb, &ep);
    CHECK(ep.n == 0, "store: deleting a game deletes its steps with it");

    /* The whole point of a server-side store: it outlives the client. */
    cwick_store_close(s);
    cwick_store *s2 = cwick_store_open(path, &err);
    CHECK(s2 != NULL, "store: reopens the same file");
    char *v2 = s2 ? cwick_store_kv_get(s2, "tictactoe", "score") : NULL;
    CHECK(v2 && strstr(v2, "\"win\":4") != NULL,
          "store: data survives a restart (why the store is not localStorage)");
    free(v2);
    cwick_store_close(s2);

    setenv("CWICK_STORE", path2, 1);
    cwick_store *s3 = cwick_store_open(NULL, &err);
    CHECK(s3 && cwick_store_path(s3) &&
          strcmp(cwick_store_path(s3), path2) == 0,
          "store: CWICK_STORE chooses the file when no path is given");
    cwick_store_close(s3);
    unsetenv("CWICK_STORE");
}

/* ---------------------------- API keys ------------------------------ */

typedef struct {
    int n;
    int saw_named;
    char first_prefix[64];
    char first_id[64];
    int64_t last_used;
} key_probe;

static void key_each_cb(const cwick_api_key_info *k, void *user) {
    key_probe *p = (key_probe *)user;
    if (p->n == 0) {
        snprintf(p->first_prefix, sizeof(p->first_prefix), "%s", k->prefix);
        snprintf(p->first_id, sizeof(p->first_id), "%s", k->id);
        p->last_used = k->last_used_at;
    }
    p->n++;
    if (strcmp(k->name, "cursor") == 0) p->saw_named = 1;
}

static void test_auth_keys(void) {
    cwick_auth *a = cwick_auth_new("pw", 3600);
    CHECK(a != NULL, "key: auth object exists");
    if (!a) return;

    cwick_api_key_info info;
    char *secret = cwick_auth_key_create(a, "cursor", &info);
    CHECK(secret != NULL, "key: minting returns a secret");
    CHECK(secret && strncmp(secret, "cwick_", 5) == 0,
          "key: the secret is recognizable by its cwick_ prefix");
    CHECK(secret && strlen(secret) == 5 + 32,
          "key: 32 random characters follow the prefix");
    CHECK(info.hash_hex && strlen(info.hash_hex) == 64,
          "key: a 64-char hash is exposed for persistence");
    CHECK(info.prefix && strncmp(info.prefix, "cwick_", 5) == 0 &&
          strncmp(info.prefix, secret, strlen(info.prefix)) == 0,
          "key: the display prefix is a leading slice of the secret");
    CHECK(info.name && strcmp(info.name, "cursor") == 0, "key: the name is kept");

    /* A wrong secret must fail, and so must a session token: the two
     * credential types are not interchangeable. */
    char id[32];
    CHECK(cwick_auth_key_check(a, secret, id, sizeof(id)),
          "key: the secret authenticates");
    CHECK(id[0] && strcmp(id, info.id) == 0,
          "key: the check reports which key matched");
    CHECK(!cwick_auth_key_check(a, "cwick_" "00000000000000000000000000000000",
                               NULL, 0),
          "key: a wrong secret is rejected");
    CHECK(!cwick_auth_key_check(a, "", NULL, 0) &&
          !cwick_auth_key_check(a, NULL, NULL, 0),
          "key: an empty secret is rejected");
    {
        char *tok = cwick_auth_login(a, "pw");
        CHECK(tok != NULL, "key: a session can still be minted");
        CHECK(tok && !cwick_auth_key_check(a, tok, NULL, 0),
              "key: a session token is not a valid API key");
        CHECK(tok && cwick_auth_validate(a, tok), "key: and it is still a session");
        CHECK(!cwick_auth_validate(a, secret),
              "key: an API key is not a valid session either");
        free(tok);
    }

    /* last_used is stamped by a successful check. */
    key_probe p;
    memset(&p, 0, sizeof(p));
    cwick_auth_keys_each(a, key_each_cb, &p);
    CHECK(p.n == 1 && p.saw_named, "key: enumeration sees the key and its name");
    CHECK(p.last_used > 0, "key: a successful check stamps last_used");

    /* Persistence round trip: what a restart would load back. */
    cwick_auth *b = cwick_auth_new("pw", 3600);
    CHECK(b != NULL, "key: a second auth object (the 'restarted' one)");
    CHECK(cwick_auth_key_load(b, info.id, info.name, info.hash_hex, info.prefix,
                             info.created_at, 0, true),
          "key: a persisted hash can be adopted");
    CHECK(cwick_auth_key_check(b, secret, NULL, 0),
          "key: the adopted key authenticates the same secret");
    CHECK(!cwick_auth_key_load(b, info.id, "dupe", info.hash_hex, info.prefix, 1, 0, true),
          "key: the same id cannot be adopted twice");
    CHECK(!cwick_auth_key_load(b, "bogus", "x", "not-hex", "cwick_", 1, 0, true),
          "key: a malformed hash is refused, not guessed");
    CHECK(strcmp(cwick_auth_key_prefix_of(b, info.id), info.prefix) == 0,
          "key: the prefix can be looked up for the access log");
    CHECK(strcmp(cwick_auth_key_prefix_of(b, "gone"), "cwick_") == 0,
          "key: an unknown id gives a harmless placeholder");

    /* Revocation is immediate and complete. */
    CHECK(cwick_auth_key_delete(b, info.id), "key: a key can be revoked");
    CHECK(!cwick_auth_key_check(b, secret, NULL, 0),
          "key: a revoked key stops authenticating at once");
    CHECK(!cwick_auth_key_delete(b, info.id), "key: revoking twice says no");

    /* Import: an operator-chosen secret (CWICK_API_KEY). */
    cwick_api_key_info imp;
    CHECK(cwick_auth_key_import(b, "ci", "sk-my-own-secret", &imp),
          "key: a chosen secret can be adopted");
    CHECK(cwick_auth_key_check(b, "sk-my-own-secret", NULL, 0),
          "key: and it authenticates");
    CHECK(!cwick_auth_key_import(b, "x", "", &imp),
          "key: an empty chosen secret is refused");

    if (secret) { memset(secret, 0, strlen(secret)); free(secret); }
    cwick_auth_free(b);
    cwick_auth_free(a);
}

static void test_server(void) {
    char *root = make_tmpdir();
    cwick_error err;
    cwick_error_clear(&err);
    cwick_setup_synthetic(root, &err);

    cwick_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_root = root;
    opt.max_new_tokens = 64;
    cwick_engine *eng = NULL;
    if (cwick_engine_new(&opt, &eng, &err) != CWICK_OK) {
        CHECK(0, "server engine new");
        return;
    }

    /* `g_server_stop` is process-global and the previous server test leaves it
     * set, so a fresh listener would immediately see "stop" and never answer.
     * Clearing it here is part of starting a server, not a detail of this one. */
    g_server_stop = 0;
    const int port = 8123 + (getpid() % 500);
    server_arg sa = { eng, port, NULL };
    pthread_t th;
    pthread_create(&th, NULL, server_thread, &sa);
    usleep(300 * 1000); /* let the listener come up */

    char out[65536];
    /* /v1/models */
    CHECK(http_request(port, "GET", "/v1/models", NULL, out, sizeof(out)) == 0,
          "http GET /v1/models");
    CHECK(strstr(out, "glm_5_3_flash_synthetic") != NULL, "models lists the model");

    /* GET /v1/models/<id> — OpenAI's retrieve-model call. Agent frameworks
     * probe it with their configured name, so it has to answer with the same
     * object shape the list uses, and with model_not_found when it cannot. */
    CHECK(http_request(port, "GET", "/v1/models/glm_5_3_flash_synthetic", NULL,
                       out, sizeof(out)) == 0,
          "http GET /v1/models/<id>");
    CHECK(strstr(out, "\"object\":\"model\"") != NULL &&
              strstr(out, "\"loaded\":true") != NULL,
          "retrieve returns one model object, marked loaded");
    CHECK(http_request(port, "GET", "/v1/models/does-not-exist", NULL, out,
                       sizeof(out)) == 0,
          "http GET /v1/models/<unknown>");
    CHECK(strstr(out, "404") != NULL && strstr(out, "model_not_found") != NULL,
          "unknown model id is a 404 model_not_found");

    /* LAN share info exists even on a loopback listener; it must say so
     * rather than hand a phone a 127.0.0.1 URL. */
    CHECK(http_request(port, "GET", "/v1/share", NULL, out, sizeof(out)) == 0,
          "http GET /v1/share");
    CHECK(strstr(out, "\"shareable\":false") != NULL &&
          strstr(out, "\"bind\":\"127.0.0.1\"") != NULL,
          "share info refuses a loopback-only listener");

    {
        size_t qlen = 0;
        unsigned char *qr = cwick_qr_png("https://example.com/app", &qlen);
        CHECK(qr && qlen > 100 && qr[0] == 0x89 && qr[1] == 'P' &&
              qr[2] == 'N' && qr[3] == 'G',
              "qr: a share URL becomes a PNG");
        cwick_qr_png_free(qr);
        CHECK(cwick_qr_png("", &qlen) == NULL, "qr: empty input refused");
    }

    /* chat completion (non-stream) */
    const char *chat =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"max_tokens\":8,"
        "\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}";
    CHECK(http_request(port, "POST", "/v1/chat/completions", chat, out, sizeof(out)) == 0,
          "http POST chat/completions");
    CHECK(strstr(out, "chat.completion") != NULL, "chat completion object");
    CHECK(strstr(out, "finish_reason") != NULL, "chat finish_reason present");
    CHECK(strstr(http_body(out), "\"completion_tokens_details\":"
                                 "{\"reasoning_tokens\":") != NULL,
          "chat usage breaks out the thinking tokens");
    /* keep the buffered answer to compare against the streamed deltas */
    const char *plain_body = http_body(out);
    CHECK(strstr(plain_body, ",}") == NULL,
          "non-streaming chat envelope has no trailing comma");
    char *plain_content = json_message_field(plain_body, "content");
    char *plain_reasoning = json_message_field(plain_body, "reasoning_content");
    CHECK(plain_content != NULL,
          "non-streaming chat response is standards-valid JSON");

    /* raw completion */
    const char *comp =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"prompt\":\"Hi\",\"max_tokens\":8}";
    CHECK(http_request(port, "POST", "/v1/completions", comp, out, sizeof(out)) == 0,
          "http POST completions");
    CHECK(strstr(out, "text_completion") != NULL, "completion object");

    const char *comp_stream =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"stream\":true,\"max_tokens\":2,"
        "\"prompt\":\"Hi\"}";
    CHECK(http_request(port, "POST", "/v1/completions", comp_stream,
                       out, sizeof(out)) == 0,
          "http raw completion stream");
    CHECK(strstr(out, "\"delta\":{\"text\":\"\"}") != NULL,
          "raw SSE opening frame uses delta.text");

    /* SSE streaming — same prompt as the buffered request above, so the two
     * answers must be identical. */
    const char *stream =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"stream\":true,\"max_tokens\":8,"
        "\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],"
        "\"stream_options\":{\"include_usage\":true}}";
    CHECK(http_request(port, "POST", "/v1/chat/completions", stream, out, sizeof(out)) == 0,
          "http streaming chat");
    CHECK(strstr(out, "data: [DONE]") != NULL, "SSE [DONE] terminator");
    CHECK(strstr(out, "chat.completion.chunk") != NULL, "SSE chunk object");
    CHECK(strstr(out, "\"usage\"") != NULL, "SSE usage frame");
    CHECK(strstr(out, "HTTP/1.1 200") != NULL, "SSE response status");
    {   /* the stream must be delivered incrementally, not in one lump */
        char streamed[8192];
        int frames = sse_collect_text(http_body(out), streamed, sizeof(streamed));
        CHECK(frames >= 3, "SSE deltas arrive per token (incremental)");
        size_t elen = strlen(plain_reasoning ? plain_reasoning : "") +
                      strlen(plain_content ? plain_content : "");
        char *expect = cwick_xmalloc(elen + 1);
        expect[0] = '\0';
        if (plain_reasoning) strcat(expect, plain_reasoning);
        if (plain_content) strcat(expect, plain_content);
        if (frames < 3)
            fprintf(stderr, "  frames=%d streamed=\"%.80s\" expect=\"%.80s\"\n",
                    frames, streamed, expect);
        CHECK(strcmp(expect, streamed) == 0,
              "streamed deltas reassemble the buffered answer");
        free(expect);
    }
    free(plain_content);
    free(plain_reasoning);

    /* thinking intensity: all five levels accepted, unknown spelling refused.
     * Placed after the incremental-delivery checks, which still need `out`
     * holding the streaming response above. */
    static const char *levels[] = { "低", "中", "高", "超高", "极致",
                                    "low", "medium", "ultra", "off" };
    bool levels_ok = true;
    for (size_t i = 0; i < sizeof(levels) / sizeof(levels[0]); i++) {
        char body[640];
        snprintf(body, sizeof(body),
                 "{\"model\":\"glm_5_3_flash_synthetic\",\"max_tokens\":4,"
                 "\"reasoning_effort\":\"%s\","
                 "\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}",
                 levels[i]);
        if (http_request(port, "POST", "/v1/chat/completions", body, out,
                         sizeof(out)) != 0 ||
            strstr(out, "chat.completion") == NULL)
            levels_ok = false;
    }
    CHECK(levels_ok, "http chat accepts every thinking-intensity level");
    const char *bad_effort =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"max_tokens\":4,"
        "\"reasoning_effort\":\"banana\","
        "\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}";
    CHECK(http_request(port, "POST", "/v1/chat/completions", bad_effort, out,
                       sizeof(out)) == 0 &&
          strstr(out, "400 Bad Request") != NULL &&
          strstr(out, "reasoning_effort") != NULL,
          "http chat refuses unknown reasoning_effort (fail-closed)");
    const char *none_effort =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"max_tokens\":4,"
        "\"reasoning_effort\":\"none\","
        "\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}";
    CHECK(http_request(port, "POST", "/v1/chat/completions", none_effort, out,
                       sizeof(out)) == 0 &&
          strstr(out, "chat.completion") != NULL,
          "http chat accepts reasoning_effort none (thinking kept, minimal)");

    /* Explicit max_tokens=0 is a valid empty completion, not "use default". */
    const char *zero =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"max_tokens\":0,"
        "\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}";
    CHECK(http_request(port, "POST", "/v1/chat/completions", zero, out, sizeof(out)) == 0,
          "http max_tokens=0");
    CHECK(strstr(http_body(out), "\"completion_tokens\":0") != NULL &&
              strstr(http_body(out), "\"finish_reason\":\"length\"") != NULL,
          "max_tokens=0 returns an empty length completion");

    /* fail-closed: unsupported field -> 400 */
    const char *bad =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],"
        "\"logprobs\":true}";
    CHECK(http_request(port, "POST", "/v1/chat/completions", bad, out, sizeof(out)) == 0,
          "http bad field");
    CHECK(strstr(out, "400 Bad Request") != NULL, "unsupported field -> 400");
    CHECK(strstr(out, "logprobs") != NULL, "error names the refused field");

    /* model mismatch -> 400 */
    const char *wrongmodel =
        "{\"model\":\"some-other-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}";
    CHECK(http_request(port, "POST", "/v1/chat/completions", wrongmodel, out, sizeof(out)) == 0,
          "http wrong model");
    CHECK(strstr(out, "400 Bad Request") != NULL, "wrong model -> 400");

    /* tools are accepted now (they used to be refused outright) */
    const char *withtools =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"max_tokens\":8,"
        "\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{"
        "\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\"}}}],"
        "\"tool_choice\":\"none\"}";
    CHECK(http_request(port, "POST", "/v1/chat/completions", withtools, out, sizeof(out)) == 0,
          "http tools request");
    CHECK(strstr(out, "chat.completion") != NULL,
          "tools with tool_choice none are accepted (no 400)");

    /* a tool_choice naming an undeclared function is still refused */
    const char *badtool =
        "{\"model\":\"glm_5_3_flash_synthetic\","
        "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"a\"}}],"
        "\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"b\"}}}";
    CHECK(http_request(port, "POST", "/v1/chat/completions", badtool, out, sizeof(out)) == 0,
          "http unknown tool_choice");
    CHECK(strstr(out, "400 Bad Request") != NULL, "unknown tool_choice -> 400");

    /* `developer` is OpenAI's current spelling of the instruction turn. It
     * must reach the prompt as a system turn, not as a user request (the GLM
     * template falls through to `<|user|>` for an unknown role). */
    const char *chat_developer =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"max_tokens\":2,"
        "\"messages\":[{\"role\":\"developer\",\"content\":\"Be terse.\"},"
        "{\"role\":\"user\",\"content\":\"hi\"}]}";
    CHECK(http_request(port, "POST", "/v1/chat/completions", chat_developer,
                       out, sizeof(out)) == 0 &&
              strstr(http_body(out), "\"object\":\"chat.completion\"") != NULL,
          "the developer role is accepted on chat/completions");

    /* ---- /v1/responses: a second dialect over the same engine --------
     * The point of the adapter is that a Responses request buys the same
     * prompt, queue and prefix cache as a chat request; what these checks
     * pin down is the envelope (response object, named SSE events), the
     * item conversion (developer role, function_call/function_call_output)
     * and the fields it refuses rather than silently ignoring. */
    const char *resp_plain =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"input\":\"Hello\","
        "\"max_output_tokens\":4}";
    CHECK(http_request(port, "POST", "/v1/responses", resp_plain, out, sizeof(out)) == 0,
          "http POST /v1/responses");
    CHECK(strstr(out, "\"object\":\"response\"") != NULL, "responses object");
    CHECK(strstr(out, "\"type\":\"message\"") != NULL,
          "responses output carries a message item");
    CHECK(strstr(out, "\"output_text\"") != NULL, "message item has an output_text part");
    CHECK(strstr(out, "\"input_tokens\"") != NULL &&
              strstr(out, "\"output_tokens\"") != NULL,
          "responses usage uses the Responses spelling");
    /* The synthetic checkpoint never emits an end-of-turn token, so every
     * answer here runs into the budget: that is an *incomplete* response,
     * and the status has to say so (a client uses it to notice truncation). */
    CHECK(strstr(http_body(out), "\"status\":\"incomplete\"") != NULL &&
              strstr(http_body(out), "\"reason\":\"max_output_tokens\"") != NULL,
          "a budget-limited response is reported as incomplete");
    /* Fields the Responses schema marks as required, plus the echo of the
     * request: a response that omits them is not a valid `response` object. */
    CHECK(strstr(http_body(out), "\"parallel_tool_calls\":true") != NULL,
          "response echoes parallel_tool_calls");
    CHECK(strstr(http_body(out), "\"tool_choice\":\"auto\"") != NULL,
          "response echoes tool_choice");
    CHECK(strstr(http_body(out), "\"tools\":[]") != NULL,
          "response echoes an empty tools array");
    CHECK(strstr(http_body(out), "\"error\":null") != NULL &&
              strstr(http_body(out), "\"incomplete_details\":") != NULL,
          "a response carries error and incomplete_details");
    CHECK(strstr(http_body(out), "\"store\":false") != NULL,
          "response reports that nothing was stored");

    const char *resp_stream =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"input\":\"Hello\","
        "\"max_output_tokens\":3,\"stream\":true}";
    CHECK(http_request(port, "POST", "/v1/responses", resp_stream, out, sizeof(out)) == 0,
          "http streaming responses");
    CHECK(strstr(out, "event: response.created") != NULL,
          "responses stream opens with a named event");
    CHECK(strstr(out, "event: response.output_text.delta") != NULL ||
              strstr(out, "event: response.reasoning_text.delta") != NULL,
          "responses stream carries per-token deltas");
    CHECK(strstr(out, "event: response.completed") != NULL ||
              strstr(out, "event: response.incomplete") != NULL,
          "responses stream terminates with a terminal event");
    CHECK(strstr(out, "\"type\":\"response.incomplete\"") != NULL ||
              strstr(out, "\"status\":\"completed\"") != NULL,
          "the terminal event agrees with the response status");
    CHECK(strstr(out, "data: [DONE]") == NULL,
          "responses stream has no chat-style [DONE] sentinel");

    /* Every event carries a monotonic sequence number (the schema has no
     * default for it, and a client uses it to notice a dropped frame). */
    {
        size_t events = 0, seqs = 0;
        for (const char *p = out; (p = strstr(p, "event: response.")) != NULL; p++)
            events++;
        for (const char *p = out; (p = strstr(p, "\"sequence_number\":")) != NULL; p++)
            seqs++;
        CHECK(events > 0 && events == seqs,
              "every responses event carries a sequence_number");
    }

    /* A text-producing stream must label its deltas with `logprobs` (empty
     * here: this server never computes them) — the field is required on
     * response.output_text.delta. The synthetic checkpoint spends the whole
     * budget inside its thinking block, so what it proves is the shape of
     * every output_text event that does appear. */
    const char *resp_text_stream =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"input\":\"Hello\","
        "\"max_output_tokens\":3,\"stream\":true,"
        "\"reasoning\":{\"effort\":\"none\"}}";
    CHECK(http_request(port, "POST", "/v1/responses", resp_text_stream,
                       out, sizeof(out)) == 0,
          "http streaming responses without thinking");
    {
        size_t otext = 0, lp = 0;
        for (const char *p = out; (p = strstr(p, "\"type\":\"response.output_text.")) != NULL; p++)
            otext++;
        for (const char *p = out; (p = strstr(p, "\"logprobs\":[]")) != NULL; p++)
            lp++;
        CHECK(otext > 0 && otext == lp,
              "output_text events carry the required logprobs field");
    }

    /* The item announced by `output_item.added` and the item inside
     * `response.completed` are the same item, so they must have one id. */
    {
        const char *marker =
            strstr(out, "\"type\":\"message\",\"status\":\"in_progress\"");
        const char *idstart = NULL;
        for (const char *p = out; marker && p + 6 < marker; p++)
            if (!strncmp(p, "\"id\":\"", 6)) idstart = p + 6;
        char mid[128] = "";
        if (idstart) {
            size_t k = 0;
            while (idstart[k] && idstart[k] != '"' && k + 1 < sizeof(mid)) {
                mid[k] = idstart[k];
                k++;
            }
            mid[k] = '\0';
        }
        const char *done_ev = strstr(out, "event: response.incomplete");
        if (!done_ev) done_ev = strstr(out, "event: response.completed");
        CHECK(idstart && mid[0] && done_ev && strstr(done_ev, mid) != NULL,
              "the completed response names the items the stream announced");
    }

    /* items: developer role, a tool call and its output, then a new turn */
    const char *resp_items =
        "{\"model\":\"glm_5_3_flash_synthetic\","
        "\"instructions\":\"Be terse.\",\"max_output_tokens\":2,"
        "\"input\":["
        "{\"type\":\"message\",\"role\":\"developer\","
        "\"content\":[{\"type\":\"input_text\",\"text\":\"Stay brief.\"}]},"
        "{\"type\":\"message\",\"role\":\"user\",\"content\":\"read a.c\"},"
        "{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"read\","
        "\"arguments\":\"{\\\"path\\\":\\\"a.c\\\"}\"},"
        "{\"type\":\"function_call_output\",\"call_id\":\"call_1\","
        "\"output\":\"int main(void){}\"},"
        "{\"type\":\"message\",\"role\":\"user\",\"content\":\"now explain\"}]}";
    CHECK(http_request(port, "POST", "/v1/responses", resp_items, out, sizeof(out)) == 0,
          "http responses items");
    CHECK(strstr(out, "\"object\":\"response\"") != NULL,
          "developer/function_call/function_call_output items are accepted");

    /* flat Responses tool declarations, with the tool block disabled */
    const char *resp_tools =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"input\":\"hi\","
        "\"max_output_tokens\":2,\"tool_choice\":\"none\","
        "\"tools\":[{\"type\":\"function\",\"name\":\"read\","
        "\"description\":\"read a file\","
        "\"parameters\":{\"type\":\"object\"}}]}";
    CHECK(http_request(port, "POST", "/v1/responses", resp_tools, out, sizeof(out)) == 0 &&
              strstr(out, "\"object\":\"response\"") != NULL,
          "flat Responses tools are translated to the chat declaration");
    CHECK(strstr(http_body(out), "\"tools\":[{\"type\":\"function\",\"name\":\"read\"") != NULL,
          "the response echoes the flat tool declaration it was given");

    /* fail-closed: each of these is refused by name, not ignored */
    const char *resp_state =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"input\":\"hi\","
        "\"previous_response_id\":\"resp_1\"}";
    CHECK(http_request(port, "POST", "/v1/responses", resp_state, out, sizeof(out)) == 0 &&
              strstr(out, "400 Bad Request") != NULL &&
              strstr(out, "previous_response_id") != NULL,
          "durable previous_response_id is refused by name");
    const char *resp_store =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"input\":\"hi\",\"store\":true}";
    CHECK(http_request(port, "POST", "/v1/responses", resp_store, out, sizeof(out)) == 0 &&
              strstr(out, "400 Bad Request") != NULL,
          "store=true is refused (responses are not persisted)");
    const char *resp_required =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"input\":\"hi\","
        "\"tools\":[{\"type\":\"function\",\"name\":\"read\"}],"
        "\"tool_choice\":\"required\"}";
    CHECK(http_request(port, "POST", "/v1/responses", resp_required, out, sizeof(out)) == 0 &&
              strstr(out, "400 Bad Request") != NULL &&
              strstr(out, "required") != NULL,
          "tool_choice required is refused (no constrained decoding)");
    const char *resp_image =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"input\":[{\"type\":\"message\","
        "\"role\":\"user\",\"content\":[{\"type\":\"input_image\","
        "\"image_url\":\"x\"}]}]}";
    CHECK(http_request(port, "POST", "/v1/responses", resp_image, out, sizeof(out)) == 0 &&
              strstr(out, "400 Bad Request") != NULL &&
              strstr(out, "input_image") != NULL,
          "image parts are refused, not silently dropped");
    const char *resp_unknown =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"input\":\"hi\",\"banana\":1}";
    CHECK(http_request(port, "POST", "/v1/responses", resp_unknown, out, sizeof(out)) == 0 &&
              strstr(out, "400 Bad Request") != NULL &&
              strstr(out, "banana") != NULL,
          "an unknown Responses field is a 400 that names it");
    const char *resp_noinput = "{\"model\":\"glm_5_3_flash_synthetic\"}";
    CHECK(http_request(port, "POST", "/v1/responses", resp_noinput, out, sizeof(out)) == 0 &&
              strstr(out, "400 Bad Request") != NULL &&
              strstr(out, "missing input") != NULL,
          "a responses request without input is refused");

    /* ---- /v1/messages: the Anthropic dialect -------------------------
     * Same engine, different wire format: `system` is top-level, content is
     * typed blocks, a tool result is its own block, and the stream is
     * message_start … content_block_* … message_delta … message_stop with no
     * [DONE]. These checks pin the envelope, the block conversion and the
     * fields refused by name. */
    const char *anth_plain =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"max_tokens\":4,"
        "\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}";
    CHECK(http_request(port, "POST", "/v1/messages", anth_plain, out, sizeof(out)) == 0,
          "http POST /v1/messages");
    CHECK(strstr(http_body(out), "\"type\":\"message\"") != NULL,
          "anthropic body is a message object");
    CHECK(strstr(http_body(out), "\"role\":\"assistant\"") != NULL,
          "anthropic message has the assistant role");
    CHECK(strstr(http_body(out), "\"stop_reason\":\"") != NULL &&
              strstr(http_body(out), "\"stop_sequence\":null") != NULL,
          "anthropic stop reason and stop_sequence are present");
    CHECK(strstr(http_body(out), "\"input_tokens\":") != NULL &&
              strstr(http_body(out), "\"output_tokens\":") != NULL,
          "anthropic usage counts input and output tokens");

    const char *anth_stream =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"max_tokens\":3,"
        "\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}";
    CHECK(http_request(port, "POST", "/v1/messages", anth_stream, out, sizeof(out)) == 0,
          "http streaming anthropic");
    CHECK(strstr(out, "event: message_start") != NULL &&
              strstr(out, "\"type\":\"message_start\"") != NULL,
          "anthropic stream opens with message_start");
    CHECK(strstr(out, "event: content_block_start") != NULL &&
              strstr(out, "event: content_block_delta") != NULL,
          "anthropic stream carries content blocks and deltas");
    CHECK(strstr(out, "event: content_block_stop") != NULL,
          "anthropic stream closes its blocks");
    CHECK(strstr(out, "event: message_delta") != NULL &&
              strstr(out, "\"stop_reason\":\"") != NULL,
          "anthropic stream reports the stop reason in message_delta");
    CHECK(strstr(out, "event: message_stop") != NULL,
          "anthropic stream terminates with message_stop");
    CHECK(strstr(out, "data: [DONE]") == NULL,
          "anthropic stream has no [DONE] sentinel");
    /* message_delta repeats the input side so a client that only reads the
     * closing events still has a complete usage object. */
    {
        const char *md = strstr(out, "event: message_delta");
        CHECK(md != NULL && strstr(md, "\"input_tokens\":") != NULL &&
                  strstr(md, "\"output_tokens\":") != NULL,
              "message_delta repeats input_tokens alongside output_tokens");
    }

    /* top-level system + a tool_use/tool_result history + an image-free
     * block list: the shapes a coding agent replays on every turn */
    const char *anth_history =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"max_tokens\":2,"
        "\"system\":[{\"type\":\"text\",\"text\":\"Be terse.\"}],"
        "\"messages\":["
        "{\"role\":\"user\",\"content\":\"read a\"},"
        "{\"role\":\"assistant\",\"content\":[{\"type\":\"thinking\","
        "\"thinking\":\"need it\"},{\"type\":\"tool_use\",\"id\":\"c1\","
        "\"name\":\"read\",\"input\":{\"path\":\"a\"}}]},"
        "{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\","
        "\"tool_use_id\":\"c1\",\"content\":[{\"type\":\"text\",\"text\":\"ok\"}]},"
        "{\"type\":\"text\",\"text\":\"go\"}]}]}";
    CHECK(http_request(port, "POST", "/v1/messages", anth_history, out, sizeof(out)) == 0 &&
              strstr(http_body(out), "\"type\":\"message\"") != NULL,
          "system blocks and tool_use/tool_result history are accepted");

    /* fail-closed, each refused by name */
    const char *anth_nomax =
        "{\"model\":\"glm_5_3_flash_synthetic\","
        "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    CHECK(http_request(port, "POST", "/v1/messages", anth_nomax, out, sizeof(out)) == 0 &&
              strstr(out, "400 Bad Request") != NULL &&
              strstr(out, "max_tokens is required") != NULL,
          "anthropic requires max_tokens");
    const char *anth_any =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"max_tokens\":2,"
        "\"tools\":[{\"name\":\"read\",\"input_schema\":{\"type\":\"object\"}}],"
        "\"tool_choice\":{\"type\":\"any\"},"
        "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    CHECK(http_request(port, "POST", "/v1/messages", anth_any, out, sizeof(out)) == 0 &&
              strstr(out, "400 Bad Request") != NULL &&
              strstr(out, "any") != NULL,
          "tool_choice any is refused (no constrained decoding)");
    const char *anth_image =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"max_tokens\":2,"
        "\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image\","
        "\"source\":{\"type\":\"base64\",\"data\":\"x\"}}]}]}";
    CHECK(http_request(port, "POST", "/v1/messages", anth_image, out, sizeof(out)) == 0 &&
              strstr(out, "400 Bad Request") != NULL &&
              strstr(out, "image") != NULL,
          "image blocks are refused, not silently dropped");
    const char *anth_stop =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"max_tokens\":2,"
        "\"stop_sequences\":[\"END\"],"
        "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    CHECK(http_request(port, "POST", "/v1/messages", anth_stop, out, sizeof(out)) == 0 &&
              strstr(out, "400 Bad Request") != NULL &&
              strstr(out, "stop_sequences") != NULL,
          "non-default stop_sequences are refused");
    const char *anth_unknown =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"max_tokens\":2,\"banana\":1,"
        "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    CHECK(http_request(port, "POST", "/v1/messages", anth_unknown, out, sizeof(out)) == 0 &&
              strstr(out, "400 Bad Request") != NULL &&
              strstr(out, "banana") != NULL,
          "an unknown Anthropic field is a 400 that names it");
    /* the error envelope is Anthropic's, not OpenAI's */
    CHECK(strstr(http_body(out), "\"type\":\"error\"") != NULL &&
              strstr(http_body(out), "\"error\":{\"type\":") != NULL,
          "anthropic errors use the Anthropic envelope");

    /* unknown endpoint -> 404 */
    CHECK(http_request(port, "GET", "/v1/nonsense", NULL, out, sizeof(out)) == 0,
          "http unknown path");
    CHECK(strstr(out, "404 Not Found") != NULL, "unknown endpoint -> 404");

    /* ---- sampling fields are honoured, not just tolerated ------------
     * They used to be accepted-and-ignored (temperature/top_p) or refused
     * outright (presence_penalty / frequency_penalty), which is how a
     * sampling-trained checkpoint ended up decoded with argmax -- i.e. how a
     * reply degenerated into one repeated clause. */
    const char *samp =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"max_tokens\":8,"
        "\"temperature\":0.8,\"top_p\":0.95,\"top_k\":20,"
        "\"presence_penalty\":0.1,\"frequency_penalty\":0.1,"
        "\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}";
    CHECK(http_request(port, "POST", "/v1/chat/completions", samp, out, sizeof(out)) == 0,
          "http sampling request");
    CHECK(strstr(out, "chat.completion") != NULL,
          "temperature/top_p/top_k/penalties are accepted (no 400)");
    CHECK(strstr(out, "400 Bad Request") == NULL, "sampling request is served");

    /* and the range is still fail-closed: temperature 5 is not "2 in disguise" */
    const char *hot =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"max_tokens\":8,"
        "\"temperature\":5,"
        "\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}";
    CHECK(http_request(port, "POST", "/v1/chat/completions", hot, out, sizeof(out)) == 0,
          "http out-of-range temperature");
    CHECK(strstr(out, "400 Bad Request") != NULL &&
              strstr(out, "temperature") != NULL,
          "temperature outside [0,2] is a 400 that names the field");

    const char *badp =
        "{\"model\":\"glm_5_3_flash_synthetic\",\"max_tokens\":8,\"top_p\":1.5,"
        "\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}";
    CHECK(http_request(port, "POST", "/v1/chat/completions", badp, out, sizeof(out)) == 0,
          "http out-of-range top_p");
    CHECK(strstr(out, "400 Bad Request") != NULL, "top_p outside (0,1] is a 400");

    /* stop the server: set the flag and wake accept() with a dummy connect */
    g_server_stop = 1;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd >= 0) {
        struct sockaddr_in a;
        memset(&a, 0, sizeof(a));
        a.sin_family = AF_INET;
        a.sin_port = htons((uint16_t)port);
        inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
        connect(fd, (struct sockaddr *)&a, sizeof(a));
        close(fd);
    }
    pthread_join(th, NULL);
    cwick_engine_free(eng);
}

/* Stop the in-process test server: set the flag and wake accept() with a dummy
 * connect, then join. Shared by the two phases of the per-app store test. */
static void stop_test_server(int port, pthread_t th) {
    g_server_stop = 1;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd >= 0) {
        struct sockaddr_in a;
        memset(&a, 0, sizeof(a));
        a.sin_family = AF_INET;
        a.sin_port = htons((uint16_t)port);
        inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
        connect(fd, (struct sockaddr *)&a, sizeof(a));
        close(fd);
    }
    pthread_join(th, NULL);
}

/* ------------------------------------------------------------------ */
/* 5a-ii. Per-app databases (one sqlite file per app)                  */
/* ------------------------------------------------------------------ */

/* The routing under test lives in server.c: a `--store` value that names a
 * directory (or has no database suffix) means one `<app>.db` per app plus
 * `auth.db` for the keys, while a value that is a file — or ends in .db —
 * stays one shared database, because that is what an older run wrote. Two apps
 * must land in two different files, or "delete one app's data" could take the
 * other's with it. Checked through the real endpoints, since the mapping is the
 * part that matters. */
static void test_store_per_app(void) {
    char *root = make_tmpdir();
    if (!root) { CHECK(0, "per-app store: tmpdir"); return; }
    cwick_error err;
    cwick_error_clear(&err);
    CHECK(cwick_setup_synthetic(root, &err) == CWICK_OK ||
              err.status == CWICK_OK,
          "per-app store: synthetic model written");

    char dbdir[256], shared[256];
    snprintf(dbdir, sizeof(dbdir), "%s/dbroot", root);
    snprintf(shared, sizeof(shared), "%s/shared.db", root);

    cwick_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_root = root;
    opt.max_new_tokens = 8;
    cwick_engine *eng = NULL;
    if (cwick_engine_new(&opt, &eng, &err) != CWICK_OK) {
        CHECK(0, "per-app store: engine new");
        return;
    }

    char out[65536];
    const int port = 8200 + (getpid() % 300);
    const char *put_a = "{\"key\":\"score\",\"value\":{\"wins\":2,\"games\":2}}";
    const char *put_b = "{\"key\":\"score\",\"value\":{\"wins\":7,\"games\":9}}";

    /* Phase 1: a fresh directory -> one file per app. */
    g_server_stop = 0;
    server_arg sa = { eng, port, dbdir };
    pthread_t th;
    pthread_create(&th, NULL, server_thread, &sa);
    usleep(400 * 1000);
    CHECK(http_request(port, "PUT", "/v1/apps/alpha/state", put_a,
                       out, sizeof(out)) == 0 &&
          strstr(out, "200 OK") != NULL,
          "per-app store: PUT alpha state");
    CHECK(http_request(port, "PUT", "/v1/apps/beta/state", put_b,
                       out, sizeof(out)) == 0 &&
          strstr(out, "200 OK") != NULL,
          "per-app store: PUT beta state");

    char alpha_db[320], beta_db[320], auth_db[320];
    snprintf(alpha_db, sizeof(alpha_db), "%s/alpha.db", dbdir);
    snprintf(beta_db, sizeof(beta_db), "%s/beta.db", dbdir);
    snprintf(auth_db, sizeof(auth_db), "%s/auth.db", dbdir);
    CHECK(access(alpha_db, R_OK) == 0, "per-app store: alpha.db was created");
    CHECK(access(beta_db, R_OK) == 0, "per-app store: beta.db was created");
    CHECK(access(auth_db, R_OK) == 0, "per-app store: auth.db was created");

    /* The same key in two apps must not be the same row. */
    CHECK(http_request(port, "GET", "/v1/apps/alpha/state?key=score", NULL,
                       out, sizeof(out)) == 0 &&
          strstr(out, "\"wins\":2") != NULL,
          "per-app store: alpha reads back its own value");
    CHECK(http_request(port, "GET", "/v1/apps/beta/state?key=score", NULL,
                       out, sizeof(out)) == 0 &&
          strstr(out, "\"wins\":7") != NULL,
          "per-app store: beta reads back its own value");

    CHECK(http_request(port, "GET", "/v1/store", NULL, out, sizeof(out)) == 0 &&
          strstr(out, "\"mode\":\"per-app\"") != NULL &&
          strstr(out, "alpha.db") != NULL,
          "per-app store: /v1/store reports the layout and the files");

    stop_test_server(port, th);

    /* Phase 2: the legacy single-file layout still wins when the path says so,
     * and it does not disturb the per-app files from phase 1. */
    g_server_stop = 0;
    server_arg sa2 = { eng, port, shared };
    pthread_t th2;
    pthread_create(&th2, NULL, server_thread, &sa2);
    usleep(400 * 1000);
    CHECK(http_request(port, "PUT", "/v1/apps/alpha/state", put_a,
                       out, sizeof(out)) == 0,
          "per-app store: the shared layout still accepts writes");
    CHECK(http_request(port, "GET", "/v1/store", NULL, out, sizeof(out)) == 0 &&
          strstr(out, "\"mode\":\"single\"") != NULL,
          "per-app store: a .db path reports the single layout");
    CHECK(http_request(port, "GET", "/v1/apps/alpha/state?key=score", NULL,
                       out, sizeof(out)) == 0 &&
          strstr(out, "\"wins\":2") != NULL,
          "per-app store: the shared database serves its own app data");
    stop_test_server(port, th2);
    cwick_engine_free(eng);

    CHECK(access(shared, R_OK) == 0, "per-app store: shared.db was created");
    CHECK(access(alpha_db, R_OK) == 0 && access(beta_db, R_OK) == 0,
          "per-app store: the per-app files survived the other layout");
}

/* ------------------------------------------------------------------ */
/* 5b. Local model discovery (--models-dir / GET /v1/models)           */
/* ------------------------------------------------------------------ */

/* A directory is a cwick checkpoint when config.json sits next to either the
 * converted resident/ tree or the released shards; everything else (an empty
 * directory, a random folder of files) is not. */
static void test_models(void) {
    char parent[160];
    snprintf(parent, sizeof(parent), "/tmp/cwick-test-models-%ld", (long)getpid());
    {
        char cmd[256];
        snprintf(cmd, sizeof(cmd), "rm -rf %s", parent);
        if (system(cmd) != 0) { /* best effort */ }
    }
    cwick_mkdir_p(parent);

    char m1[192], m2[192], plain[192];
    snprintf(m1, sizeof(m1), "%s/m1", parent);
    snprintf(m2, sizeof(m2), "%s/m2", parent);
    snprintf(plain, sizeof(plain), "%s/plain", parent);
    cwick_mkdir_p(m1);
    cwick_mkdir_p(m2);
    cwick_mkdir_p(plain);

    cwick_error err;
    cwick_error_clear(&err);
    CHECK(cwick_setup_synthetic(m1, &err) == CWICK_OK ||
          err.status == CWICK_OK, "synthetic model A written");
    cwick_error_clear(&err);
    CHECK(cwick_setup_synthetic(m2, &err) == CWICK_OK ||
          err.status == CWICK_OK, "synthetic model B written");

    char id[128];
    CHECK(cwick_model_dir_probe(m1, id, sizeof(id)), "probe: a model root is recognised");
    CHECK(strcmp(id, "glm_5_3_flash_synthetic") == 0,
          "probe: the id comes from config.json's model_type");

    /* "Looks like a checkpoint" and "can be loaded" are different facts:
     * the library downloads any repository, but only architectures with
     * kernels can become resident. */
    CHECK(cwick_model_type_supported("glm5_next") &&
          cwick_model_type_supported("glm-5.3-flash") &&
          cwick_model_type_supported("deepseek_v41") &&
          cwick_model_type_supported("deepseek_v41_text") &&
          cwick_model_type_supported("minimax_m3_vl") &&
          cwick_model_type_supported("glm_5_3_flash_synthetic") &&
          cwick_model_type_supported("glm-5.3-flash-synthetic-mhc"),
          "compat: every architecture with kernels in this build is accepted");
    CHECK(!cwick_model_type_supported("qwen3_5_text") &&
          !cwick_model_type_supported("llama") &&
          !cwick_model_type_supported(NULL) &&
          !cwick_model_type_supported(""),
          "compat: unknown architectures are refused");
    {
        char compat[256], fcfg[320], why[192];
        snprintf(compat, sizeof(compat), "/tmp/cwick-test-compat-%ld",
                 (long)getpid());
        {
            char cmd[320];
            snprintf(cmd, sizeof(cmd), "rm -rf %s", compat);
            if (system(cmd) != 0) { /* best effort */ }
        }
        cwick_mkdir_p(compat);
        snprintf(fcfg, sizeof(fcfg), "%s/config.json", compat);
        const char *fj =
            "{\"model_type\":\"qwen3_5\",\"pad_token_id\":248055,"
            "\"text_config\":{\"model_type\":\"qwen3_5_text\","
            "\"pad_token_id\":null}}";
        CHECK(cwick_write_file(fcfg, fj, strlen(fj)) == 0,
              "compat: a foreign nested config is written");
        CHECK(cwick_model_root_supported(m1, why, sizeof(why)),
              "compat: a synthetic checkpoint is switchable");
        CHECK(!cwick_model_root_supported(compat, why, sizeof(why)) &&
              strstr(why, "unsupported model_type") != NULL &&
              strstr(why, "qwen3_5_text") != NULL,
              "compat: the inner model_type decides, and the reason names it");
        cwick_model_list fl;
        cwick_models_list_init(&fl);
        CHECK(cwick_models_list_add(&fl, "qwen3_5", compat) &&
              !fl.v[0].supported &&
              strstr(fl.v[0].unsupported_reason, "qwen3_5_text") != NULL,
              "compat: the catalog entry caches the verdict and its reason");
        cwick_models_list_free(&fl);
    }

    /* Served aliases: release checkpoints advertise their familiar names,
     * everything else keeps what its config.json says. */
    char alias[128];
    char glmroot[192], dsroot[256];
    snprintf(glmroot, sizeof(glmroot), "%s/zai-org/GLM-5.3-Flash", parent);
    snprintf(dsroot, sizeof(dsroot), "%s/deepseek-ai/DeepSeek-V4.1-Flash-1017",
             parent);
    CHECK(cwick_model_served_alias(glmroot, alias, sizeof(alias)) &&
              strcmp(alias, "glm-5.3-flash") == 0,
          "alias: the GLM release name is derived from the directory");
    CHECK(cwick_model_served_alias(dsroot, alias, sizeof(alias)) &&
              strcmp(alias, "deepSeek-v4.1-flash-1017") == 0,
          "alias: the DeepSeek release name keeps its interior capital");
    CHECK(!cwick_model_served_alias(m1, alias, sizeof(alias)) && alias[0] == '\0',
          "alias: a fixture keeps its own model_type");
    {
        cwick_model_list al;
        cwick_models_list_init(&al);
        CHECK(cwick_models_list_add(&al, "glm5_next", glmroot),
              "alias: a release root is listed");
        CHECK(strcmp(al.v[0].alias, "glm-5.3-flash") == 0,
              "alias: the listed entry carries the served name");
        CHECK(cwick_models_list_add(&al, "other_type", m1) &&
              al.v[1].alias[0] == '\0',
              "alias: a non-release entry has none");
        cwick_models_list_dedupe_ids(&al);
        CHECK(strcmp(al.v[0].alias, "glm-5.3-flash") == 0,
              "alias: dedupe leaves a unique alias alone");
        cwick_models_list_free(&al);
    }
    CHECK(!cwick_model_dir_probe(plain, id, sizeof(id)),
          "probe: a directory without config.json is not a model");
    CHECK(!cwick_model_dir_probe("/tmp/cwick-test-does-not-exist", id, sizeof(id)),
          "probe: a missing directory is not a model");

    cwick_model_list l;
    cwick_models_list_init(&l);
    int added = cwick_models_scan_dir(parent, &l);
    CHECK(added == 2, "scan: both models found, nothing else");
    CHECK(l.n == 2, "scan: no duplicates");
    CHECK(cwick_models_list_add(&l, "again", m1) == false,
          "scan: adding the same root twice is refused");
    cwick_models_list_dedupe_ids(&l);
    CHECK(strcmp(l.v[0].id, l.v[1].id) != 0,
          "scan: two checkpoints with one model_type get distinct ids");
    /* The resident one keeps the name it was started with: index 0 is never
     * renamed, or a client that learned the name from the banner breaks. */
    CHECK(strcmp(l.v[0].id, "glm_5_3_flash_synthetic") == 0,
          "scan: entry 0 keeps the advertised id");
    bool roots_ok = (strcmp(l.v[0].root, m1) == 0 && strcmp(l.v[1].root, m2) == 0) ||
                    (strcmp(l.v[0].root, m2) == 0 && strcmp(l.v[1].root, m1) == 0);
    CHECK(roots_ok, "scan: entries point at the two model roots");
    CHECK(l.v[0].supported && l.v[1].supported,
          "compat: scanned entries carry the cached supported verdict");
    cwick_models_list_free(&l);
    CHECK(cwick_models_scan_dir("/tmp/cwick-test-no-such-dir", &l) == -1,
          "scan: an unreadable directory reports -1 (does not crash)");
    {
        char cmd[256];
        snprintf(cmd, sizeof(cmd), "rm -rf %s", parent);
        if (system(cmd) != 0) { /* best effort */ }
    }
}

/* ------------------------------------------------------------------ */
/* 5c. Log capture: this process's own stdout/stderr, in a ring         */
/* ------------------------------------------------------------------ */

typedef struct {
    size_t count;
    int saw_out;
    int saw_err;
    int out_src;
    int err_src;
    int found;
    const char *needle;
    uint64_t last_seq;
    uint64_t min_seq;
} log_probe;

static void log_probe_cb(const cwick_logcap_line *l, void *user) {
    log_probe *p = (log_probe *)user;
    p->count++;
    p->last_seq = l->seq;
    if (l->seq < p->min_seq) p->min_seq = l->seq;
    if (p->needle && strstr(l->text, p->needle)) p->found = 1;
    if (strstr(l->text, "cwick-logcap-test-out")) {
        p->saw_out = 1;
        p->out_src = (int)l->src;
    }
    if (strstr(l->text, "cwick-logcap-test-err")) {
        p->saw_err = 1;
        p->err_src = (int)l->src;
    }
}

/* Poll the ring until `needle` is in it.
 *
 * Why not cwick_logcap_wait(): it returns as soon as *anything* is newer than
 * the cursor, and this test prints a CHECK line immediately before taking the
 * cursor — so a wait-based loop spins through all its iterations in
 * microseconds, exits before the pump's next poll cycle, and reports a missing
 * feature that is only a few milliseconds late. Matching on the text instead
 * makes the wait mean what the assertion means. */
static int log_wait_for(const char *needle, int budget_ms) {
    for (int waited = 0; waited < budget_ms; waited += 5) {
        log_probe p;
        memset(&p, 0, sizeof(p));
        p.needle = needle;
        cwick_logcap_foreach(1, 0, log_probe_cb, &p);
        if (p.found) return 1;
        usleep(5 * 1000);
    }
    return 0;
}

static void test_logcap(void) {
    CHECK(!cwick_logcap_active(), "logcap: inactive until started");
    /* Small ring on purpose: the eviction path is what this test is for. */
    CHECK(cwick_logcap_start(8, 0), "logcap: start installs the tee");
    CHECK(cwick_logcap_active(), "logcap: active after start");

    /* Both streams, and a partial line without a newline. */
    printf("cwick-logcap-test-out\n");
    fprintf(stderr, "cwick-logcap-test-err\n");
    printf("cwick-logcap-test-partial");
    fflush(stdout);
    fflush(stderr);

    CHECK(log_wait_for("cwick-logcap-test-out", 2000), "logcap: stdout is captured");
    CHECK(log_wait_for("cwick-logcap-test-err", 2000), "logcap: stderr is captured");
    {
        log_probe p;
        memset(&p, 0, sizeof(p));
        cwick_logcap_foreach(1, 0, log_probe_cb, &p);
        CHECK(p.saw_out && p.out_src == (int)CWICK_LOGCAP_OUT,
              "logcap: stdout keeps its source tag");
        CHECK(p.saw_err && p.err_src == (int)CWICK_LOGCAP_ERR,
              "logcap: stderr keeps its source tag");
    }

    /* A line with no trailing newline must not be lost: it is flushed when the
     * capture is stopped, but a *later* complete line pushes it first. */
    printf("cwick-logcap-test-tail\n");
    fflush(stdout);
    CHECK(log_wait_for("cwick-logcap-test-partial", 2000),
          "logcap: the partial line is completed by the next one");

    /* `since` must actually cut: everything before a cursor is not resent. */
    (void)log_wait_for("cwick-logcap-test-tail", 2000);
    uint64_t mid = cwick_logcap_head();
    printf("cwick-logcap-test-after\n");
    fflush(stdout);
    CHECK(log_wait_for("cwick-logcap-test-after", 2000),
          "logcap: later lines keep arriving");
    {
        log_probe r;
        memset(&r, 0, sizeof(r));
        r.min_seq = ~0ull;
        cwick_logcap_foreach(mid, 0, log_probe_cb, &r);
        CHECK(r.count >= 1, "logcap: new lines are visible since a cursor");
        CHECK(r.min_seq >= mid, "logcap: `since` resends nothing older than itself");
    }

    /* The ring is bounded: 8 lines of capacity, and the overflow is counted
     * rather than silently forgotten. */
    for (int i = 0; i < 50; i++) printf("cwick-logcap-fill-%d\n", i);
    fflush(stdout);
    (void)log_wait_for("cwick-logcap-fill-49", 3000);
    CHECK(cwick_logcap_lines() <= 8, "logcap: the ring never exceeds its line bound");
    CHECK(cwick_logcap_dropped() > 0, "logcap: evicted lines are counted, not hidden");

    cwick_logcap_stop();
    CHECK(!cwick_logcap_active(), "logcap: stop restores the descriptors");
    /* Everything printed while it was on is still there: the terminal was never
     * the only copy. Tests keep printing after this point, so the ring must
     * still be readable (it is — stop only ends the tee). */
    printf("cwick-logcap-test-after-stop\n");
}

/* ------------------------------------------------------------------ */
/* 5d. Server control endpoints: log tail, config, model switch        */
/* ------------------------------------------------------------------ */
typedef struct {
    cwick_engine *eng;
    int port;
    const char *models_dir;
    cwick_engine **live;
} switch_arg;

static void *switch_server_thread(void *argp) {
    switch_arg *sa = (switch_arg *)argp;
    cwick_server_options opts;
    memset(&opts, 0, sizeof(opts));
    opts.host = "127.0.0.1";
    opts.port = sa->port;
    opts.max_tokens = 64;
    opts.models_dirs[0] = sa->models_dir;
    opts.models_dirs_n = 1;
    opts.live_engine = sa->live;
    cwick_error err;
    cwick_error_clear(&err);
    cwick_server_run(sa->eng, &opts, &g_server_stop, &err);
    return NULL;
}

static void test_server_control(void) {
    char parent[160], m1[192], m2[192];
    snprintf(parent, sizeof(parent), "/tmp/cwick-test-sw-%ld", (long)getpid());
    {
        char cmd[256];
        snprintf(cmd, sizeof(cmd), "rm -rf %s", parent);
        if (system(cmd) != 0) { /* best effort */ }
    }
    snprintf(m1, sizeof(m1), "%s/alpha", parent);
    snprintf(m2, sizeof(m2), "%s/beta", parent);
    cwick_mkdir_p(m1);
    cwick_mkdir_p(m2);
    cwick_error err;
    cwick_error_clear(&err);
    (void)cwick_setup_synthetic(m1, &err);
    cwick_error_clear(&err);
    (void)cwick_setup_synthetic(m2, &err);

    cwick_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_root = m1;
    opt.max_new_tokens = 32;
    cwick_engine *eng = NULL;
    cwick_error_clear(&err);
    if (cwick_engine_new(&opt, &eng, &err) != CWICK_OK) {
        CHECK(0, "switch test: engine new");
        return;
    }
    cwick_engine *live = eng;

    /* The log tail has to have something in it while this server runs. */
    (void)cwick_logcap_start(0, 0);

    const int port = 8400 + (getpid() % 500);
    /* test_server leaves the global stop flag set for its own server; this one
     * must not inherit it. */
    g_server_stop = 0;
    switch_arg sa = { eng, port, parent, &live };
    pthread_t th;
    pthread_create(&th, NULL, switch_server_thread, &sa);
    usleep(400 * 1000);

    char out[65536];
    CHECK(http_request(port, "GET", "/v1/models", NULL, out, sizeof(out)) == 0,
          "http GET /v1/models (catalog)");
    CHECK(strstr(out, "glm_5_3_flash_synthetic") != NULL, "catalog lists the resident model");
    CHECK(strstr(out, "\"loaded\":true") != NULL, "catalog marks the loaded entry");
    CHECK(strstr(out, "\"loaded\":false") != NULL, "catalog marks the others");
    CHECK(strstr(out, "@2") != NULL, "catalog disambiguates duplicate model_types");

    CHECK(http_request(port, "GET", "/v1/config", NULL, out, sizeof(out)) == 0,
          "http GET /v1/config");
    CHECK(strstr(http_body(out), "\"model\"") != NULL, "config reports the resident model");
    {
        /* The startup scan also registers anything under ./models, so the
         * exact count depends on the checkout; what matters is that the two
         * synthetic catalog entries are both counted. */
        const char *cp = strstr(http_body(out), "\"catalog_size\":");
        int known = cp ? atoi(cp + 15) : 0;
        CHECK(known >= 2, "config reports how many models are known");
    }
    CHECK(strstr(http_body(out), "\"log_capture\":true") != NULL,
          "config reports the log tee is on");

    /* The log endpoint serves what this process printed, including the line
     * this test writes after the server came up. */
    printf("cwick-control-test-marker-line\n");
    fflush(stdout);
    usleep(300 * 1000);
    CHECK(http_request(port, "GET", "/v1/logs?limit=200", NULL, out, sizeof(out)) == 0,
          "http GET /v1/logs");
    CHECK(strstr(http_body(out), "\"capture\":true") != NULL, "logs report capture on");
    CHECK(strstr(http_body(out), "cwick-control-test-marker-line") != NULL,
          "logs contain what the process printed");

    /* unknown model -> 404, not a crash */
    CHECK(http_request(port, "POST", "/v1/models/switch",
                       "{\"model\":\"no-such-model\"}", out, sizeof(out)) == 0,
          "http POST switch (unknown)");
    CHECK(strstr(out, "404 Not Found") != NULL, "switching to an unknown model -> 404");

    /* switch to the second one, then back: the engine that must be freed is
     * whatever the endpoint reports, never the startup pointer. */
    char body[256];
    snprintf(body, sizeof(body), "{\"model\":\"glm_5_3_flash_synthetic@2\"}");
    CHECK(http_request(port, "POST", "/v1/models/switch", body, out, sizeof(out)) == 0,
          "http POST switch (beta)");
    CHECK(strstr(http_body(out), "\"status\":\"switched\"") != NULL,
          "switch to another local model succeeds");
    CHECK(http_request(port, "GET", "/v1/models", NULL, out, sizeof(out)) == 0,
          "http GET /v1/models after the switch");
    CHECK(strstr(out, "beta") != NULL && strstr(out, "\"loaded\":true") != NULL,
          "the switched-to model is the loaded one");

    /* the checkpoint's own name still works after a switch (two spellings,
     * one model) */
    CHECK(http_request(port, "POST", "/v1/chat/completions",
                       "{\"model\":\"glm_5_3_flash_synthetic\",\"max_tokens\":4,"
                       "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}",
                       out, sizeof(out)) == 0,
          "http chat after the switch");
    CHECK(strstr(http_body(out), "chat.completion") != NULL,
          "the swapped-in model answers a request");

    snprintf(body, sizeof(body), "{\"model\":\"glm_5_3_flash_synthetic\"}");
    CHECK(http_request(port, "POST", "/v1/models/switch", body, out, sizeof(out)) == 0,
          "http POST switch (back to alpha)");
    CHECK(strstr(http_body(out), "\"status\":\"switched\"") != NULL ||
          strstr(http_body(out), "already_resident") != NULL,
          "switching back succeeds");

    g_server_stop = 1;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd >= 0) {
        struct sockaddr_in a;
        memset(&a, 0, sizeof(a));
        a.sin_family = AF_INET;
        a.sin_port = htons((uint16_t)port);
        inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
        connect(fd, (struct sockaddr *)&a, sizeof(a));
        close(fd);
    }
    pthread_join(th, NULL);
    g_server_stop = 0;
    cwick_logcap_stop();
    /* `live` is the engine the server finished with: free that, not `eng`,
     * which was replaced (and freed) during the swap. */
    CHECK(live != NULL, "the server reports a live engine after switching");
    cwick_engine_free(live);
    {
        char cmd[256];
        snprintf(cmd, sizeof(cmd), "rm -rf %s", parent);
        if (system(cmd) != 0) { /* best effort */ }
    }
}

/* ------------------------------------------------------------------ */
/* 6. Host-memory budget (at most --mem-fraction of RAM)               */
/* ------------------------------------------------------------------ */
static void test_memory_plan(void) {
    const uint64_t gib = 1ull << 30;
    cwick_error err;
    cwick_error_clear(&err);
    cwick_mem_plan plan;

    /* reset to the documented default: 80% of host RAM */
    cwick_memory_set_policy(CWICK_MEM_DEFAULT_FRACTION, 0);

    /* the real GLM-5.3-Flash numbers on a 128 GiB machine */
    cwick_status st = cwick_mem_plan_build(128 * gib,
                                         (uint64_t)(15.46 * (double)gib),
                                         (uint64_t)(2.94 * (double)gib),
                                         32u << 20, 384u << 20,
                                         290 * gib, &plan, &err);
    CHECK(st == CWICK_OK, "memory plan builds for a 306 GiB checkpoint");
    CHECK(plan.host_fraction > 0.79 && plan.host_fraction < 0.81,
          "default budget is 80% of host RAM");
    CHECK(plan.target_bytes == (uint64_t)(102.4 * (double)gib),
          "80% of 128 GiB is a 102.4 GiB ceiling");
    {
        uint64_t used = plan.resident_bytes + plan.state_bytes +
                        plan.arena_bytes + plan.headroom_bytes;
        uint64_t remainder = plan.target_bytes - used;
        CHECK(plan.expert_cache_bytes > 0 &&
              used + plan.expert_cache_bytes <= plan.target_bytes,
              "the cache is booked out of the ceiling (never double-booked)");
        /* The cache deliberately does NOT take the whole remainder: on a
         * 128 GiB box filling it to 83.6 GiB measured 0.475 tok/s against
         * 0.601 at 45 GiB, because the extra residency costs more in memory
         * pressure than the saved SSD reads are worth.
         *
         * The cap used to be host/8 (16 GiB) on 15-step measurements. Those
         * runs were too short: a cache of N slots spends its first N
         * admissions filling, and every fill misses the size-class LIFO, so a
         * 32 GiB run showed 4095 LIFO misses for 4096 slots -- all fill. Over
         * 99 steps the fill amortises and 32 GiB ties 16 GiB on wall time
         * (795 vs 798 ms/step) while cutting SSD traffic 21% and lifting the
         * hit rate 30.6% -> 45.2%. Past 32 GiB it turns: 48 GiB is +2.3% and
         * 64 GiB is +14.4%, with gemv alone degrading 150 -> 197 ms. So the
         * cap is host/4.
         *
         * It is applied to the cache and not to `used + cache`: subtracting
         * the already-booked bytes is exactly what silently disabled this cap
         * on the real box, where resident + state + arena + headroom is
         * already 16.1 GiB and a small host/N - used underflowed to zero. */
        CHECK(plan.expert_cache_bytes <= remainder &&
              plan.cache_capped_by_reserve,
              "cache is capped by the host reserve, not by the ceiling");
        CHECK(plan.expert_cache_bytes == 128 * gib / 4,
              "the cap is a quarter of host RAM, on the cache itself");
    }

    /* the whole point: nothing is allowed to exceed the ceiling */
    CHECK(plan.resident_bytes + plan.state_bytes + plan.expert_cache_bytes +
              plan.arena_bytes + plan.headroom_bytes <= plan.target_bytes,
          "resident + state + cache stay under the ceiling");

    /* --mem-fraction moves the ceiling */
    cwick_memory_set_policy(0.60, 0);
    cwick_mem_plan p60;
    st = cwick_mem_plan_build(128 * gib, (uint64_t)(15.46 * (double)gib),
                            (uint64_t)(2.94 * (double)gib), 32u << 20,
                            384u << 20, 290 * gib, &p60, &err);
    /* Both fractions hit the same half-of-host cap here, so only the ceiling
     * moves; on a machine where the remainder is below the cap, the cache
     * moves too (asserted for a 64 GiB host below). */
    CHECK(st == CWICK_OK && p60.target_bytes < plan.target_bytes &&
          p60.expert_cache_bytes <= plan.expert_cache_bytes,
          "--mem-fraction 60 shrinks the ceiling and never grows the cache");
    {
        cwick_mem_plan psmall;
        cwick_memory_set_policy(CWICK_MEM_DEFAULT_FRACTION, 0);
        st = cwick_mem_plan_build(64 * gib, (uint64_t)(15.46 * (double)gib),
                                 (uint64_t)(2.94 * (double)gib), 32u << 20,
                                 384u << 20, 290 * gib, &psmall, &err);
        CHECK(st == CWICK_OK && psmall.expert_cache_bytes > 0 &&
              psmall.expert_cache_bytes < plan.expert_cache_bytes,
              "a smaller host gets a smaller pinned cache");
    }

    /* an absolute ceiling wins when it is smaller */
    cwick_memory_set_policy(0.80, 40 * gib);
    cwick_mem_plan plim;
    st = cwick_mem_plan_build(128 * gib, (uint64_t)(15.46 * (double)gib),
                            (uint64_t)(2.94 * (double)gib), 32u << 20,
                            384u << 20, 290 * gib, &plim, &err);
    CHECK(st == CWICK_OK && plim.target_bytes == 40 * gib,
          "--max-memory caps the ceiling below the host fraction");

    /* fail closed rather than swap */
    cwick_memory_set_policy(CWICK_MEM_DEFAULT_FRACTION, 0);
    cwick_mem_plan pbad;
    st = cwick_mem_plan_build(8 * gib, (uint64_t)(15.46 * (double)gib), 0,
                             32u << 20, 384u << 20, 290 * gib, &pbad, &err);
    CHECK(st == CWICK_ERR_MEMORY && pbad.limited_by_ram,
          "a spine that cannot fit is refused instead of swapped");

    /* Fixed arenas/headroom are part of admission too: resident+state alone
     * fit here, but the complete no-cache plan does not. */
    cwick_mem_plan pfixed;
    st = cwick_mem_plan_build(10 * gib, 5 * gib, 2 * gib,
                             (uint64_t)(0.75 * gib), (uint64_t)(0.50 * gib),
                             290 * gib, &pfixed, &err);
    CHECK(st == CWICK_ERR_MEMORY && pfixed.expert_cache_bytes == 0,
          "arena/headroom overflow is rejected instead of starting over budget");

    cwick_memory_set_policy(CWICK_MEM_DEFAULT_FRACTION, 0);
    CHECK(cwick_host_memory_bytes() > 0, "host memory is detected");
    CHECK(cwick_process_rss_bytes() > 0, "process RSS is measurable");
}

/* ------------------------------------------------------------------ */
/* 7. Tools: declare in the request, read back from the completion     */
/* ------------------------------------------------------------------ */
static void test_tools(void) {
    cwick_error err;
    cwick_error_clear(&err);

    /* ---- request side ---- */
    const char *body =
        "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{"
        "\"name\":\"get_weather\",\"description\":\"Get the weather\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]},"
        "\"strict\":true}}],\"tool_choice\":\"auto\"}";
    json_value *root = cwick_json_parse(body, strlen(body), &err);
    CHECK(root != NULL, "tools request parses");
    if (!root) return;

    cwick_tool_set set;
    char verr[256];
    int code = cwick_tools_parse_request(root, &set, verr, sizeof(verr));
    CHECK(code == 0, "tools + tool_choice auto are accepted");
    CHECK(set.count == 1 && strcmp(set.items[0].name, "get_weather") == 0,
          "declared function name is kept");
    CHECK(set.advertise && strcmp(set.choice, "auto") == 0,
          "tool_choice auto advertises the tools");
    CHECK(set.items[0].json && strstr(set.items[0].json, "\"name\":\"get_weather\"")
          && strstr(set.items[0].json, "\"parameters\"")
          && !strstr(set.items[0].json, "\"strict\""),
          "tool is rendered in the checkpoint's own <tools> JSON form");

    char *block = cwick_tools_render_block(&set);
    CHECK(block != NULL, "tools block renders");
    if (block) {
        CHECK(strstr(block, "<tools>") && strstr(block, "</tools>"),
              "block uses the GLM <tools> wrapper");
        CHECK(strstr(block, "<tool_call>") && strstr(block, "<arg_key>"),
              "block teaches the <tool_call> answer format");
        free(block);
    }
    cwick_json_free(root);
    cwick_tools_free_set(&set);

    /* ---- fail-closed refusals ---- */
    const char *bad =
        "{\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\"}}],"
        "\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"other\"}}}";
    root = cwick_json_parse(bad, strlen(bad), &err);
    if (root) {
        cwick_tool_set s2;
        code = cwick_tools_parse_request(root, &s2, verr, sizeof(verr));
        CHECK(code == 400 && strstr(verr, "not in `tools`") != NULL,
              "tool_choice naming an undeclared function -> 400");
        cwick_tools_free_set(&s2);
        cwick_json_free(root);
    }
    const char *req =
        "{\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\"}}],"
        "\"tool_choice\":\"required\"}";
    root = cwick_json_parse(req, strlen(req), &err);
    if (root) {
        cwick_tool_set s3;
        code = cwick_tools_parse_request(root, &s3, verr, sizeof(verr));
        CHECK(code == 400, "tool_choice \"required\" is refused (no forced calls)");
        cwick_tools_free_set(&s3);
        cwick_json_free(root);
    }

    /* ---- completion side ---- */
    const char *out =
        "Let me check.\n"
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value>"
        "<arg_key>unit</arg_key><arg_value>celsius</arg_value></tool_call>\n"
        "<tool_call>get_time<arg_key>tz</arg_key><arg_value>{\"tz\":\"UTC\"}</arg_value></tool_call>";
    cwick_tool_call *calls = NULL;
    size_t ncalls = 0;
    char *visible = NULL;
    int rc = cwick_tools_extract(out, &calls, &ncalls, &visible, &err);
    CHECK(rc == 0 && ncalls == 2, "two <tool_call> blocks are parsed");
    if (rc == 0 && ncalls == 2) {
        CHECK(strcmp(calls[0].name, "get_weather") == 0, "first call name");
        CHECK(strcmp(calls[0].arguments_json, "{\"city\":\"Beijing\","
              "\"unit\":\"celsius\"}") == 0, "arguments become one JSON object");
        CHECK(strcmp(calls[1].name, "get_time") == 0, "second call name");
        CHECK(strstr(calls[1].arguments_json, "\"tz\"") != NULL,
              "a JSON value is kept as JSON, not re-quoted");
        CHECK(calls[0].id[0] && strncmp(calls[0].id, "call_", 5) == 0,
              "calls carry an OpenAI-shaped id");
    }
    CHECK(visible && strstr(visible, "Let me check.") != NULL,
          "prose before the first call stays visible");
    free(visible);
    cwick_tool_calls_free(calls, ncalls);

    /* a malformed call is refused, never half-parsed */
    cwick_tool_call *bad_calls = NULL;
    size_t nbad = 0;
    rc = cwick_tools_extract("<tool_call>no_args<arg_key>city</arg_key>"
                            "<arg_value>Beijing", &bad_calls, &nbad, NULL, &err);
    CHECK(rc == -1, "an unterminated tool call is refused (fail-closed)");
    cwick_tool_calls_free(bad_calls, nbad);

    /* no tools in the text: the answer is returned as-is */
    cwick_tool_call *none = NULL;
    size_t nnone = 0;
    char *plain = NULL;
    rc = cwick_tools_extract("Just an answer.", &none, &nnone, &plain, &err);
    CHECK(rc == 0 && nnone == 0 && plain && strcmp(plain, "Just an answer.") == 0,
          "a plain answer is untouched");
    free(plain);
    cwick_tool_calls_free(none, nnone);

    /* ---- the same <tool_call> block as one checkpoint writes it: XML, with a
     * namespace token in front of every structural tag. That token is a single
     * entry in the checkpoint's added-token table and its template splices it
     * in, so the model is following the spec, not drifting. Scanned naively the
     * framing lands inside the tool name and the whole generation is refused as
     * an undeclared tool — which is what a real turn did. */
    {
        static const char ns[] = "]<]minimax[>[";
        char mm[1024];
        snprintf(mm, sizeof(mm), "%s<tool_call>\n%s<invoke name=\"add_numbers\">"
                 "%s<a>137</a>%s<b>258</b>%s</invoke>\n%s</tool_call>",
                 ns, ns, ns, ns, ns, ns);
        cwick_tool_call *mmc = NULL;
        size_t nmm = 0;
        rc = cwick_tools_extract(mm, &mmc, &nmm, NULL, &err);
        CHECK(rc == 0 && nmm == 1, "a namespaced <invoke> block is one call");
        if (rc == 0 && nmm == 1) {
            CHECK(strcmp(mmc[0].name, "add_numbers") == 0,
                  "invoke name= is the tool name, without the framing");
            CHECK(strcmp(mmc[0].arguments_json, "{\"a\":137,\"b\":258}") == 0,
                  "each child element is one argument");
        }
        cwick_tool_calls_free(mmc, nmm);
    }
    {
        static const char ns[] = "]<]minimax[>[";
        char mm[1024];
        snprintf(mm, sizeof(mm), "%s<tool_call>%s<invoke name=\"f\">%s<x>1</x>"
                 "%s</invoke>%s<invoke name=\"g\">%s<y>2</y>%s</invoke>%s</tool_call>",
                 ns, ns, ns, ns, ns, ns, ns, ns);
        cwick_tool_call *mmc = NULL;
        size_t nmm = 0;
        rc = cwick_tools_extract(mm, &mmc, &nmm, NULL, &err);
        CHECK(rc == 0 && nmm == 2, "two invocations in one block are two calls");
        if (rc == 0 && nmm == 2) {
            CHECK(strcmp(mmc[0].name, "f") == 0 && strcmp(mmc[1].name, "g") == 0,
                  "each invocation keeps its own name");
            CHECK(strcmp(mmc[1].arguments_json, "{\"y\":2}") == 0,
                  "arguments are not carried over between invocations");
        }
        cwick_tool_calls_free(mmc, nmm);
    }
    {
        /* The drift that was actually observed: the model opens the block,
         * writes half a tag and a sentence of prose, and only then opens the
         * one it fills in. One declared call has to come out of it. */
        static const char ns[] = "]<]minimax[>[";
        char mm[1024];
        snprintf(mm, sizeof(mm),
                 "<tool_call>\n<invoke name \xe5\x90\x8d\xe5\xad\x97 "
                 "tool_name=\"add_numbers\" \xe9\x87\x8c\xe7\x9a\x84\xe5\x8f\x82"
                 "\xe6\x95\xb0\xe6\x98\xaf a \xe5\x92\x8c b\xe3\x80\x82\n\n"
                 "Let me call the tool.<tool_call>\n%s<invoke name=\"add_numbers\">"
                 "%s<a>137</a>%s<b>258</b>%s</invoke>\n%s</tool_call>",
                 ns, ns, ns, ns, ns);
        cwick_tool_call *mmc = NULL;
        size_t nmm = 0;
        rc = cwick_tools_extract(mm, &mmc, &nmm, NULL, &err);
        CHECK(rc == 0 && nmm == 1, "a drifted opener still yields one call");
        if (rc == 0 && nmm == 1) {
            CHECK(strcmp(mmc[0].name, "add_numbers") == 0,
                  "the drifted prose is not taken as the tool name");
            CHECK(strcmp(mmc[0].arguments_json, "{\"a\":137,\"b\":258}") == 0,
                  "the arguments survive the drift");
        }
        cwick_tool_calls_free(mmc, nmm);
    }
    {
        /* The other form, written by the same checkpoint with the same
         * framing: without stripping the token the block looks unterminated
         * and the request is refused. */
        static const char ns[] = "]<]minimax[>[";
        char mm[512];
        snprintf(mm, sizeof(mm), "%s<tool_call>get_weather%s<arg_key>city</arg_key>"
                 "%s<arg_value>Beijing</arg_value>%s</tool_call>", ns, ns, ns, ns);
        cwick_tool_call *mmc = NULL;
        size_t nmm = 0;
        rc = cwick_tools_extract(mm, &mmc, &nmm, NULL, &err);
        CHECK(rc == 0 && nmm == 1, "the arg_key form survives the same framing");
        if (rc == 0 && nmm == 1) {
            CHECK(strcmp(mmc[0].name, "get_weather") == 0 &&
                  strcmp(mmc[0].arguments_json, "{\"city\":\"Beijing\"}") == 0,
                  "arg_key/arg_value is read through the namespace token");
        }
        cwick_tool_calls_free(mmc, nmm);
    }
    {
        /* The token has also landed *inside* a value. Stripping it only where a
         * tag is expected left `137]` as the value and `]minimax[` as the next
         * argument's key; it is framing wherever it appears, so it is removed
         * from the whole answer before anything parses it. */
        static const char ns[] = "]<]minimax[>[";
        char mm[512];
        snprintf(mm, sizeof(mm), "%s<tool_call>%s<invoke name=\"add_numbers\">"
                 "%s<a>137%s</a>%s<b>258%s</b>%s</invoke>%s</tool_call>",
                 ns, ns, ns, ns, ns, ns, ns, ns);
        cwick_tool_call *mmc = NULL;
        size_t nmm = 0;
        rc = cwick_tools_extract(mm, &mmc, &nmm, NULL, &err);
        CHECK(rc == 0 && nmm == 1, "a token inside a value is framing, not data");
        if (rc == 0 && nmm == 1)
            CHECK(strcmp(mmc[0].arguments_json, "{\"a\":137,\"b\":258}") == 0,
                  "the value is the value, without the token inside it");
        cwick_tool_calls_free(mmc, nmm);
    }

    /* ---- a call name that drops the declared case is folded, not refused.
     * A real session declared `Bash` and the model answered `bash`; the
     * strict compare refused the whole generation. Matching up to case and
     * rewriting to the declared spelling keeps the call usable while a name
     * that matches nothing is still fail-closed. */
    {
        cwick_tool_decl decls[2];
        memset(decls, 0, sizeof(decls));
        snprintf(decls[0].name, sizeof(decls[0].name), "Bash");
        snprintf(decls[1].name, sizeof(decls[1].name), "Read");
        cwick_tool_set set;
        memset(&set, 0, sizeof(set));
        set.present = true;
        set.advertise = true;
        snprintf(set.choice, sizeof(set.choice), "auto");
        set.items = decls;
        set.count = 2;

        cwick_tool_call c;
        memset(&c, 0, sizeof(c));
        snprintf(c.name, sizeof(c.name), "bash");
        CHECK(cwick_tool_decl_match(&c, &set) == CWICK_DECL_FOLDED &&
              strcmp(c.name, "Bash") == 0,
              "a lower-case `bash` folds to the declared `Bash`");

        snprintf(c.name, sizeof(c.name), "Read");
        CHECK(cwick_tool_decl_match(&c, &set) == CWICK_DECL_EXACT &&
              strcmp(c.name, "Read") == 0,
              "an exact name is left alone");

        snprintf(c.name, sizeof(c.name), "BASH");
        CHECK(cwick_tool_decl_match(&c, &set) == CWICK_DECL_FOLDED &&
              strcmp(c.name, "Bash") == 0,
              "an upper-case `BASH` folds too (first declaration wins)");

        snprintf(c.name, sizeof(c.name), "shell");
        CHECK(cwick_tool_decl_match(&c, &set) == CWICK_DECL_UNKNOWN,
              "a name matching nothing is still refused");
    }
}

/* ------------------------------------------------------------------ */
/* 8. Chat prompt carries the tools a request declares                 */
/* ------------------------------------------------------------------ */
static void test_chat_tools_prompt(void) {
    char *root = make_tmpdir();
    cwick_error err;
    cwick_error_clear(&err);
    cwick_setup_synthetic(root, &err);

    cwick_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_root = root;
    cwick_engine *eng = NULL;
    if (cwick_engine_new(&opt, &eng, &err) != CWICK_OK) {
        CHECK(0, "tools prompt: engine new");
        return;
    }
    cwick_tokenizer *tok = NULL;
    char tdir[1152];
    cwick_path_join(tdir, sizeof(tdir), root, "tokenizer");
    if (cwick_tokenizer_load(tdir, cwick_engine_spec(eng), &tok, &err) != CWICK_OK) {
        CHECK(0, "tools prompt: tokenizer");
        cwick_engine_free(eng);
        return;
    }
    uint32_t *a = NULL, *b = NULL;
    int64_t na = 0, nb = 0;
    const char *tools =
        "\n# Tools\n\n<tools>\n{\"name\":\"get_weather\",\"parameters\":{}}\n</tools>\n";
    CHECK(cwick_chat_encode(tok, "You are helpful.", "Hello", &a, &na, &err) == CWICK_OK,
          "plain chat encode");
    CHECK(cwick_chat_encode_tools(tok, "You are helpful.", "Hello", tools, &b, &nb, &err)
          == CWICK_OK, "chat encode with tools");
    CHECK(nb > na, "declared tools reach the prompt (longer prefix)");
    /* legacy entry point stays tool-free */
    CHECK(na > 0 && nb > 0, "both prefixes encode");
    free(a); free(b);
    cwick_tokenizer_free(tok);
    cwick_engine_free(eng);
}

/* ------------------------------------------------------------------ */
/* 8. Layer-major batched prefill == token-serial reference            */
/* ------------------------------------------------------------------ */
/* The batched prefill (forward_prefill / forward_prefill_mhc) reorders the
 * work but must produce exactly the same continuation as the token-serial
 * loop. We diff the two on a tiny model, forcing a small chunk so the
 * cross-chunk KV/KDA carry is exercised too. */
static void test_batched_prefill_equiv(const char *label,
                                       cwick_status (*setup)(const char *, cwick_error *)) {
    char *root = make_tmpdir();
    cwick_error err;
    cwick_error_clear(&err);
    if (setup(root, &err) != CWICK_OK) { CHECK(0, "equiv: model setup"); return; }

    cwick_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_root = root;
    opt.max_new_tokens = 8;
    cwick_engine *eng = NULL;
    if (cwick_engine_new(&opt, &eng, &err) != CWICK_OK) {
        CHECK(0, "equiv: engine new");
        return;
    }

    uint32_t ids[16];
    int64_t nids = 0;
    cwick_tokenizer *tok = NULL;
    char tdir[1152];
    cwick_path_join(tdir, sizeof(tdir), root, "tokenizer");
    cwick_tokenizer_load(tdir, cwick_engine_spec(eng), &tok, &err);
    if (tok) {
        uint32_t *e = NULL;
        int64_t ne = 0;
        if (cwick_chat_encode(tok, "You are helpful.", "Hello world",
                             &e, &ne, &err) == CWICK_OK && e) {
            int64_t m = ne < 16 ? ne : 16;
            memcpy(ids, e, (size_t)m * sizeof(uint32_t));
            nids = m;
            free(e);
        }
        cwick_tokenizer_free(tok);
    }
    if (nids == 0) {
        for (int i = 0; i < 12; i++) ids[i] = (uint32_t)(3 + i * 7);
        nids = 12;
    }

    const int64_t V = cwick_engine_spec(eng)->vocab_size;
    float *lb = cwick_xmalloc((size_t)V * sizeof(float));
    float *ls = cwick_xmalloc((size_t)V * sizeof(float));

    /* Batched sweep forced to small chunks (exercises the cross-chunk KV/KDA
     * carry), then the token-serial reference on the same prompt.
     *
     * Two chunk sizes on purpose. A prompt whose length is 1 mod chunk leaves
     * the last token alone in the final chunk, and a per-token bug inside the
     * batched MoE (the 2026-10-24 routing-weight gather read row 0 of the
     * per-token score matrix for every token but the first) is invisible when
     * the compared token is that chunk's first row. Chunk 4 with a length that
     * is not 1 mod 4 puts the compared token mid-chunk, where the bug shows. */
    unsetenv("CWICK_NO_BATCH_PREFILL");
    /* The chunk attention path is a separate change with its own test below;
     * this one keeps checking the per-token sweep bit-for-bit. */
    setenv("CWICK_NO_CHUNK_ATTN", "1", 1);
    int64_t plen = nids;
    if (plen % 4 == 1 && plen > 4) plen--;   /* keep the last token mid-chunk */
    double rel = 0.0;
    int same = 1;
    for (int ci = 0; ci < 2; ci++) {
        char cb[8];
        snprintf(cb, sizeof(cb), "%d", ci == 0 ? 3 : 4);
        setenv("CWICK_PREFILL_CHUNK", cb, 1);
        int okb = cwick_engine_prefill_logits(eng, ids, plen, lb, &err) == CWICK_OK;
        setenv("CWICK_NO_BATCH_PREFILL", "1", 1);
        unsetenv("CWICK_PREFILL_CHUNK");
        int oks = cwick_engine_prefill_logits(eng, ids, plen, ls, &err) == CWICK_OK;
        unsetenv("CWICK_NO_BATCH_PREFILL");

        /* The two evaluation orders differ only in floating-point summation
         * (the batched MoE uses tiled BLAS), so compare logits within a
         * tolerance and require the same argmax -- not bitwise equality. A real
         * indexing or state-carry bug shows up as an O(1) relative difference.
         * The tolerance is 1e-4, not 1e-3: a per-token score-gather bug inside
         * the batched MoE measures ~1e-3 on these fixtures (that is what the
         * decisive synthetic router scale buys), while legitimate summation
         * noise measures ~1e-7. */
        double maxabs = 0.0, maxdiff = 0.0;
        int64_t ab = 0, as = 0;
        for (int64_t i = 0; i < V; i++) {
            double v = fabs((double)lb[i]);
            double d = fabs((double)lb[i] - (double)ls[i]);
            if (v > maxabs) maxabs = v;
            if (d > maxdiff) maxdiff = d;
            if (lb[i] > lb[ab]) ab = i;
            if (ls[i] > ls[as]) as = i;
        }
        double r = maxabs > 0.0 ? maxdiff / maxabs : maxdiff;
        if (r > rel) rel = r;
        if (!okb || !oks || r >= 1e-4 || ab != as) same = 0;
        if (!same) {
            fprintf(stderr, "  %s chunk=%s rel=%.3e maxdiff=%.3e argmax batched=%lld serial=%lld\n",
                    label, cb, r, maxdiff, (long long)ab, (long long)as);
        }
    }
    unsetenv("CWICK_NO_CHUNK_ATTN");
    char msg[160];
    snprintf(msg, sizeof(msg),
             "batched prefill == token-serial prefill (%s, rel=%.1e)", label, rel);
    CHECK(same, msg);
    free(lb);
    free(ls);
    cwick_engine_free(eng);
}

/* A resumed prefill must produce the same readout as one uninterrupted pass.
 * /v1/systemone reuses the state prefix across the questions of one request,
 * so a wrong cursor or a stale KV row would silently change an answer instead
 * of failing: the endpoint would keep returning well-formed letters that mean
 * something else. Two cut points, one mid-prompt and one one-token-from-end,
 * because a resume that only works at the very end is the easy case. */
static void test_prefill_resume_equiv(const char *label,
                                      cwick_status (*setup)(const char *, cwick_error *)) {
    char *root = make_tmpdir();
    cwick_error err;
    cwick_error_clear(&err);
    if (setup(root, &err) != CWICK_OK) { CHECK(0, "resume: model setup"); return; }

    cwick_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_root = root;
    opt.max_new_tokens = 8;
    cwick_engine *eng = NULL;
    if (cwick_engine_new(&opt, &eng, &err) != CWICK_OK) {
        CHECK(0, "resume: engine new");
        return;
    }

    const int64_t n = 24;
    uint32_t ids[24];
    for (int64_t i = 0; i < n; i++) ids[i] = (uint32_t)(3 + (i * 11) % 97);

    const int64_t V = cwick_engine_spec(eng)->vocab_size;
    float *full = cwick_xmalloc((size_t)V * sizeof(float));
    float *split = cwick_xmalloc((size_t)V * sizeof(float));

    int ok = cwick_engine_prefill_logits(eng, ids, n, full, &err) == CWICK_OK;
    CHECK(ok, "resume: the reference prefill runs");

    static const int64_t cuts[] = { 7, 23 };
    for (size_t c = 0; ok && c < sizeof(cuts) / sizeof(cuts[0]); c++) {
        int64_t cut = cuts[c];
        cwick_error_clear(&err);
        if (cwick_engine_prefill_logits_at(eng, ids, cut, 0, split, &err) != CWICK_OK ||
            cwick_engine_prefill_logits_at(eng, ids + cut, n - cut, cut, split,
                                           &err) != CWICK_OK) {
            CHECK(0, "resume: the split prefill runs");
            ok = 0;
            break;
        }
        double maxabs = 0.0, maxdiff = 0.0;
        int64_t af = 0, as = 0;
        for (int64_t i = 0; i < V; i++) {
            double v = fabs((double)full[i]);
            double d = fabs((double)full[i] - (double)split[i]);
            if (v > maxabs) maxabs = v;
            if (d > maxdiff) maxdiff = d;
            if (full[i] > full[af]) af = i;
            if (split[i] > split[as]) as = i;
        }
        double rel = maxabs > 0.0 ? maxdiff / maxabs : maxdiff;
        char msg[200];
        snprintf(msg, sizeof(msg),
                 "prefill resume: split at %lld == one pass (%s, rel=%.1e, argmax %lld/%lld)",
                 (long long)cut, label, rel, (long long)af, (long long)as);
        CHECK(rel < 1e-4 && af == as, msg);
        if (rel >= 1e-4 || af != as) ok = 0;
    }

    free(full);
    free(split);
    cwick_engine_free(eng);
}

/* ------------------------------------------------------------------ */
/* 8b. Prefix cache: replaying a captured prefix == prefilling it      */
/* ------------------------------------------------------------------ */
/* The prefix cache is only worth its complexity if restoring a snapshot and
 * prefilling the tail gives exactly the same continuation as prefilling the
 * whole prompt. Byte-for-byte state, greedy decoding, so the generated tokens
 * must match exactly -- not within a tolerance. */
/* Capture the engine's state after prefilling exactly `P` tokens. The capture
 * hook fires right after the prefill and before the first decode step, so a
 * one-token generation is enough to trigger it and still leave the state at
 * the prefill boundary. */
static void capture_at(cwick_engine *eng, const uint32_t *ids, int64_t P,
                       cwick_prefix_state *out, cwick_error *err) {
    cwick_engine_reset(eng, err);
    cwick_engine_set_prefix_capture(eng, out);
    cwick_gen_result r;
    memset(&r, 0, sizeof(r));
    cwick_engine_generate(eng, ids, P, 1, false, &r, err);
    cwick_gen_result_free(&r);
    cwick_engine_set_prefix_capture(eng, NULL);
}

static void test_prefix_cache_equiv(void) {
    char *root = make_tmpdir();
    cwick_error err;
    cwick_error_clear(&err);
    if (cwick_setup_synthetic_mhc(root, &err) != CWICK_OK) {
        CHECK(0, "prefix: model setup");
        return;
    }
    cwick_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_root = root;
    opt.max_new_tokens = 8;
    cwick_engine *eng = NULL;
    if (cwick_engine_new(&opt, &eng, &err) != CWICK_OK) {
        CHECK(0, "prefix: engine new");
        return;
    }

    uint32_t ids[16];
    int64_t nids = 0;
    cwick_tokenizer *tok = NULL;
    char tdir[1152];
    cwick_path_join(tdir, sizeof(tdir), root, "tokenizer");
    cwick_tokenizer_load(tdir, cwick_engine_spec(eng), &tok, &err);
    if (tok) {
        uint32_t *e = NULL;
        int64_t ne = 0;
        if (cwick_chat_encode(tok, "You are helpful.", "Hello world",
                             &e, &ne, &err) == CWICK_OK && e) {
            int64_t m = ne < 16 ? ne : 16;
            memcpy(ids, e, (size_t)m * sizeof(uint32_t));
            nids = m;
            free(e);
        }
        cwick_tokenizer_free(tok);
    }
    if (nids == 0) {
        for (int i = 0; i < 12; i++) ids[i] = (uint32_t)(3 + i * 7);
        nids = 12;
    }

    /* Reference: one prefill of the whole prompt, then 8 greedy steps. */
    cwick_engine_reset(eng, &err);
    cwick_gen_result ref;
    memset(&ref, 0, sizeof(ref));
    int ok_ref = cwick_engine_generate(eng, ids, nids, 8, false, &ref, &err) == CWICK_OK;

    /* Capture the state after prefilling the first P tokens. The capture hook
     * fires right after the prefill, before any decode step moves the
     * recurrent state on. */
    const int64_t P = nids > 5 ? 5 : 1;
    cwick_engine_reset(eng, &err);
    cwick_prefix_state st;
    memset(&st, 0, sizeof(st));
    cwick_engine_set_prefix_capture(eng, &st);
    cwick_gen_result warm;
    memset(&warm, 0, sizeof(warm));
    int ok_cap = cwick_engine_generate(eng, ids, P, 1, false, &warm, &err) == CWICK_OK;
    cwick_engine_set_prefix_capture(eng, NULL);
    cwick_gen_result_free(&warm);

    int cap_len_ok = (st.pos == P);

    /* Replay it: reset, apply, prefill only the tail. */
    cwick_engine_reset(eng, &err);
    int ok_apply = cwick_engine_apply_prefix(eng, &st, &err) == CWICK_OK;
    cwick_gen_result split;
    memset(&split, 0, sizeof(split));
    int ok_split = cwick_engine_generate(eng, ids + P, nids - P, 8, false,
                                        &split, &err) == CWICK_OK;

    int same = ok_ref && ok_cap && ok_apply && ok_split && cap_len_ok &&
               ref.n_tokens == split.n_tokens && ref.n_tokens > 0;
    if (same) {
        for (int64_t i = 0; i < ref.n_tokens && same; i++)
            if (ref.tokens[i] != split.tokens[i]) same = 0;
    }
    if (!same) {
        fprintf(stderr, "  ref_ok=%d cap_ok=%d cap_pos=%lld apply_ok=%d split_ok=%d "
                        "ntok ref=%lld split=%lld\n",
                ok_ref, ok_cap, (long long)st.pos, ok_apply, ok_split,
                (long long)ref.n_tokens, (long long)split.n_tokens);
        for (int64_t i = 0; i < ref.n_tokens && i < split.n_tokens; i++)
            if (ref.tokens[i] != split.tokens[i])
                fprintf(stderr, "  first divergence at token %lld: %u vs %u\n",
                        (long long)i, ref.tokens[i], split.tokens[i]);
    }
    CHECK(same, "prefix cache: replayed prefix == full prefill (8 tokens)");

    /* The disk round-trip: store the snapshot, read it back, and require the
     * same continuation. A hash hit that loads the wrong bytes, or a short
     * write, has to fail here rather than in production. */
    char cdir[1152];
    snprintf(cdir, sizeof(cdir), "%s/pfxcache", root);
    cwick_pfx_cache *c = cwick_pfx_open(cdir, 64ull * 1024 * 1024);
    int ok_disk = 0;
    if (c) {
        cwick_error se;
        cwick_error_clear(&se);
        int64_t hit = 0;
        if (cwick_pfx_store(c, ids, P, &st, &se) == CWICK_OK)
            hit = cwick_pfx_lookup(c, ids, P);
        cwick_prefix_state st2;
        memset(&st2, 0, sizeof(st2));
        if (hit == P &&
            cwick_pfx_load(c, ids, P, &st2, &se) == CWICK_OK) {
            cwick_engine_reset(eng, &err);
            if (cwick_engine_apply_prefix(eng, &st2, &err) == CWICK_OK) {
                cwick_gen_result disk;
                memset(&disk, 0, sizeof(disk));
                if (cwick_engine_generate(eng, ids + P, nids - P, 8, false,
                                         &disk, &err) == CWICK_OK) {
                    ok_disk = (disk.n_tokens == ref.n_tokens && disk.n_tokens > 0);
                    for (int64_t i = 0; i < disk.n_tokens && ok_disk; i++)
                        if (disk.tokens[i] != ref.tokens[i]) ok_disk = 0;
                    cwick_gen_result_free(&disk);
                }
            }
        }
        cwick_prefix_state_free(&st2);
        /* A different prompt must not hit the cached prefix. */
        uint32_t other[16];
        memcpy(other, ids, (size_t)nids * sizeof(uint32_t));
        other[0] = (uint32_t)(ids[0] + 1);
        int64_t miss = cwick_pfx_lookup(c, other, nids);
        CHECK(miss == 0, "prefix cache: a different prompt misses");
        cwick_pfx_close(c);
    }
    CHECK(ok_disk, "prefix cache: disk round-trip == full prefill");

    /* Eviction under pressure: a session that grows its prompt every turn
     * produces a new snapshot each time, and the cache is too small to hold
     * them all. Plain LRU would drop the *oldest* entry -- which is the short
     * base prompt that every single turn still starts with, so every turn
     * after would pay a full prefill. The victim has to be one of the
     * never-replayed per-turn snapshots instead. */
    {
        /* Same shape as a real session: a short base prompt that every turn
         * extends, plus one longer snapshot per turn, in a cache with room for
         * about one entry. The per-turn snapshots must be the ones discarded.
         *
         * The lengths have to differ for this to mean anything -- a base of 3
         * tokens against 7-token turn snapshots, which is why the states are
         * captured at two lengths rather than reused. */
        const int64_t base_len = nids > 3 ? 3 : 1;
        const int64_t turn_len = nids > 6 ? 6 : nids;
        cwick_prefix_state sbase, sturn;
        memset(&sbase, 0, sizeof(sbase));
        memset(&sturn, 0, sizeof(sturn));
        capture_at(eng, ids, base_len, &sbase, &err);
        capture_at(eng, ids, turn_len, &sturn, &err);

        char edir[1152];
        snprintf(edir, sizeof(edir), "%s/pfxevict", root);
        cwick_pfx_cache *ec = cwick_pfx_open(edir, 1024);   /* ~1 entry of room */
        int ok_evict = 0;
        if (ec && sbase.pos == base_len && sturn.pos == turn_len) {
            cwick_error ee;
            cwick_error_clear(&ee);
            if (cwick_pfx_store(ec, ids, base_len, &sbase, &ee) != CWICK_OK)
                fprintf(stderr, "  evict: base store failed: %s\n",
                        cwick_error_message(&ee));
            for (int t = 0; t < 24; t++) {
                uint32_t turn[16];
                for (int i = 0; i < 16; i++)
                    turn[i] = (uint32_t)(500 + t * 16 + i);
                cwick_pfx_store(ec, turn, turn_len, &sturn, &ee);
                cwick_pfx_lookup(ec, ids, base_len);   /* every turn reuses it */
            }
            ok_evict = (cwick_pfx_lookup(ec, ids, base_len) == base_len);
            if (!ok_evict) {
                int64_t ents = 0;
                uint64_t by = 0;
                cwick_pfx_stats(ec, &ents, &by);
                fprintf(stderr, "  evict: base_len=%lld turn_len=%lld "
                                "base lookup=%lld entries=%lld bytes=%llu\n",
                        (long long)base_len, (long long)turn_len,
                        (long long)cwick_pfx_lookup(ec, ids, base_len),
                        (long long)ents, (unsigned long long)by);
            }
            cwick_pfx_close(ec);
        }
        CHECK(ok_evict, "prefix cache: 24 later snapshots do not evict the base prefix");
        cwick_prefix_state_free(&sbase);
        cwick_prefix_state_free(&sturn);
    }

    cwick_prefix_state_free(&st);
    cwick_gen_result_free(&ref);
    cwick_gen_result_free(&split);
    cwick_engine_free(eng);
}

/* ------------------------------------------------------------------ */
/* 8c. Chunk attention: GEMM projections == per-token GEMV sweep        */
/* ------------------------------------------------------------------ */
/* cwick_gemm_bf16 and cwick_gemv_bf16 both dequantise bf16 exactly and then
 * sum in fp32, so the only thing allowed to differ between them is summation
 * order. If this drifts, the chunk path below cannot be trusted. */
static void test_gemm_matches_gemv(void) {
    const int64_t rows = 4, k = 16, cols = 32;
    uint16_t *w = cwick_xmalloc((size_t)(cols * k) * sizeof(uint16_t));
    float *x = cwick_xmalloc((size_t)(rows * k) * sizeof(float));
    for (int64_t i = 0; i < cols * k; i++)
        w[i] = (uint16_t)((i * 2654435761u) & 0xffff);
    for (int64_t i = 0; i < rows * k; i++)
        x[i] = (float)((i % 17) - 8) * 0.125f;
    float *g = cwick_xmalloc((size_t)(rows * cols) * sizeof(float));
    float *v = cwick_xmalloc((size_t)cols * sizeof(float));
    cwick_gemm_bf16(g, rows, k, cols, x, w);
    cwick_gemv_bf16(v, k, cols, x, w);
    double maxd = 0.0, maxa = 0.0;
    for (int64_t c = 0; c < cols; c++) {
        double d = fabs((double)g[c] - (double)v[c]);
        double a = fabs((double)v[c]);
        if (d > maxd) maxd = d;
        if (a > maxa) maxa = a;
    }
    double rel = maxa > 0.0 ? maxd / maxa : maxd;
    char msg[128];
    snprintf(msg, sizeof(msg), "gemm(row-major) == gemv (rel=%.1e)", rel);
    CHECK(rel < 1e-5, msg);
    free(w); free(x); free(g); free(v);
}

/* The chunk path must land on the same continuation as the per-token sweep.
 * It is allowed to differ in the last bits: one GEMM for the whole chunk sums
 * in a different order than one GEMV per token. It is *not* allowed to change
 * the argmax, and on this 16-wide synthetic model -- whose logits are O(1e-2)
 * because the random weights nearly cancel -- the relative distance is a
 * pessimistic bound, not the accuracy a real checkpoint sees. */
static void test_chunk_attn_equiv(void) {
    char *root = make_tmpdir();
    cwick_error err;
    cwick_error_clear(&err);
    if (cwick_setup_synthetic_mhc(root, &err) != CWICK_OK) {
        CHECK(0, "chunk attn: model setup");
        return;
    }
    cwick_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_root = root;
    opt.max_new_tokens = 8;
    cwick_engine *eng = NULL;
    if (cwick_engine_new(&opt, &eng, &err) != CWICK_OK) {
        CHECK(0, "chunk attn: engine new");
        return;
    }
    uint32_t ids[16];
    int64_t nids = 0;
    cwick_tokenizer *tok = NULL;
    char tdir[1152];
    cwick_path_join(tdir, sizeof(tdir), root, "tokenizer");
    cwick_tokenizer_load(tdir, cwick_engine_spec(eng), &tok, &err);
    if (tok) {
        uint32_t *e = NULL;
        int64_t ne = 0;
        if (cwick_chat_encode(tok, "You are helpful.", "Hello world",
                             &e, &ne, &err) == CWICK_OK && e) {
            int64_t m = ne < 16 ? ne : 16;
            memcpy(ids, e, (size_t)m * sizeof(uint32_t));
            nids = m;
            free(e);
        }
        cwick_tokenizer_free(tok);
    }
    if (nids == 0) {
        for (int i = 0; i < 12; i++) ids[i] = (uint32_t)(3 + i * 7);
        nids = 12;
    }

    const int64_t V = cwick_engine_spec(eng)->vocab_size;
    float *lc = cwick_xmalloc((size_t)V * sizeof(float));   /* chunk path */
    float *ls = cwick_xmalloc((size_t)V * sizeof(float));   /* per-token sweep */

    unsetenv("CWICK_NO_CHUNK_ATTN");
    unsetenv("CWICK_NO_BATCH_PREFILL");
    int okc = cwick_engine_prefill_logits(eng, ids, nids, lc, &err) == CWICK_OK;
    setenv("CWICK_NO_CHUNK_ATTN", "1", 1);
    int oks = cwick_engine_prefill_logits(eng, ids, nids, ls, &err) == CWICK_OK;
    unsetenv("CWICK_NO_CHUNK_ATTN");

    double maxd = 0.0, maxa = 0.0;
    int64_t ac = 0, as = 0;
    for (int64_t i = 0; i < V; i++) {
        double d = fabs((double)lc[i] - (double)ls[i]);
        double a = fabs((double)ls[i]);
        if (d > maxd) maxd = d;
        if (a > maxa) maxa = a;
        if (lc[i] > lc[ac]) ac = i;
        if (ls[i] > ls[as]) as = i;
    }
    double rel = maxa > 0.0 ? maxd / maxa : maxd;
    int same = okc && oks && rel < 5e-2 && ac == as;
    char msg[160];
    snprintf(msg, sizeof(msg),
             "chunk attention == per-token sweep (rel=%.1e, argmax %lld/%lld)",
             rel, (long long)ac, (long long)as);
    CHECK(same, msg);
    free(lc);
    free(ls);
    cwick_engine_free(eng);
}

/* ------------------------------------------------------------------ */
/* 9. An over-long max_tokens stops at the context boundary, not errors */
/* ------------------------------------------------------------------ */
static void test_max_new_clamped(void) {
    char *root = make_tmpdir();
    cwick_error err;
    cwick_error_clear(&err);
    if (cwick_setup_synthetic(root, &err) != CWICK_OK) { CHECK(0, "clamp: setup"); return; }

    cwick_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_root = root;
    opt.max_new_tokens = 8;
    cwick_engine *eng = NULL;
    if (cwick_engine_new(&opt, &eng, &err) != CWICK_OK) { CHECK(0, "clamp: engine"); return; }

    const int64_t V = cwick_engine_spec(eng)->vocab_size;
    const int64_t cap = cwick_engine_spec(eng)->max_position_embeddings;
    const int64_t plen = cap - 4;             /* leave room for exactly 4 */
    uint32_t *ids = cwick_xmalloc((size_t)plen * sizeof(uint32_t));
    for (int64_t i = 0; i < plen; i++) ids[i] = (uint32_t)(i % V);

    cwick_gen_result r;
    memset(&r, 0, sizeof(r));
    cwick_status st = cwick_engine_generate(eng, ids, plen, 100000, false, &r, &err);
    CHECK(st == CWICK_OK, "clamp: over-long max_tokens does not error");
    CHECK(r.n_tokens == 4, "clamp: stops at the context boundary");
    CHECK(r.stop_reason_length, "clamp: finish_reason is length");
    cwick_gen_result_free(&r);
    free(ids);
    cwick_engine_free(eng);
}

/* The same rule the GLM path learned, now shared by all three backends: a
 * client that sends max_tokens=32768 against a 32768-position server with an
 * 8315-token prompt is asking for "as much as fits". The DeepSeek and MiniMax
 * loops used to reject that with a 400 ("prompt 8315 + max_new 32768 at
 * position 0 exceeds the 32768-position context window") even though the
 * answer was a few tokens away. Clamping is the honest answer; only a prompt
 * that does not fit is a client error, and it says so separately because its
 * fix (shorter prompt / larger --context) is different from "lower
 * max_tokens". */
static void test_gen_window(void) {
    cwick_error err;

    /* fits: untouched */
    int64_t mn = 50;
    cwick_error_clear(&err);
    CHECK(cwick_gen_window(0, 100, &mn, 1000, &err) == CWICK_OK, "window: fits");
    CHECK(mn == 50, "window: a budget that fits is not reduced");

    /* the reported case: 8315 + 32768 > 32768 */
    mn = 32768;
    cwick_error_clear(&err);
    CHECK(cwick_gen_window(0, 8315, &mn, 32768, &err) == CWICK_OK,
          "window: an over-long max_tokens is clamped, not rejected");
    CHECK(mn == 32768 - 8315, "window: clamped to room left");

    /* a restored prefix counts against the window too */
    mn = 40000;
    cwick_error_clear(&err);
    CHECK(cwick_gen_window(1000, 2000, &mn, 32768, &err) == CWICK_OK,
          "window: prefix replay is measured from pos");
    CHECK(mn == 32768 - 3000, "window: pos + prompt both subtracted");

    /* exactly filling the window leaves nothing to generate */
    mn = 10;
    cwick_error_clear(&err);
    CHECK(cwick_gen_window(0, 32768, &mn, 32768, &err) == CWICK_ERR_ARGUMENT,
          "window: a prompt that fills the window is an error");
    CHECK(strstr(err.message, "no room") != NULL,
          "window: the message says there is no room to generate");

    /* a prompt that does not fit is a different error, with a different fix */
    mn = 10;
    cwick_error_clear(&err);
    CHECK(cwick_gen_window(0, 40000, &mn, 32768, &err) == CWICK_ERR_ARGUMENT,
          "window: an over-long prompt is rejected");
    CHECK(strstr(err.message, "does not fit") != NULL &&
          strstr(err.message, "--context") != NULL,
          "window: the message points at --context");

    /* a zero budget stays zero: the caller asked for no output at all */
    mn = 0;
    cwick_error_clear(&err);
    CHECK(cwick_gen_window(0, 100, &mn, 1000, &err) == CWICK_OK,
          "window: max_new 0 is legal");
    CHECK(mn == 0, "window: max_new 0 stays 0");

    /* nonsense in is an error, not a clamp to something negative */
    mn = -1;
    cwick_error_clear(&err);
    CHECK(cwick_gen_window(0, 100, &mn, 1000, &err) == CWICK_ERR_ARGUMENT,
          "window: negative max_new rejected");
    mn = 10;
    cwick_error_clear(&err);
    CHECK(cwick_gen_window(0, -5, &mn, 1000, &err) == CWICK_ERR_ARGUMENT,
          "window: negative prompt rejected");
}

/* ------------------------------------------------------------------ */
/* 10. Incremental stream split: reasoning / content / tool-call edges */
/* ------------------------------------------------------------------ */
/* The streaming path classifies the answer as it grows, so the marker and
 * UTF-8 boundaries are the risky part. Drive the splitter with the whole text
 * at once and then one byte at a time (which splits `</think>`, `<tool_call>`
 * and multi-byte characters across calls) and check what comes out. */
typedef struct {
    char reasoning[1024];
    char content[1024];
    bool tool_mode;
} split_out;

static void run_split(const char *full, size_t step, split_out *o) {
    size_t rlen = 0, clen = 0;
    size_t len = strlen(full);
    o->reasoning[0] = '\0';
    o->content[0] = '\0';
    cwick_stream_state st;
    memset(&st, 0, sizeof(st));
    char *buf = cwick_xmalloc(len + 1);

    for (size_t upto = 0;;) {
        size_t n = step ? (upto + step < len ? upto + step : len) : len;
        if (n < upto) n = upto;
        upto = n;
        memcpy(buf, full, upto);
        buf[upto] = '\0';
        for (;;) {
            cwick_segment_kind kind = CWICK_SEG_CONTENT;
            size_t off = 0, slen = 0;
            if (!cwick_stream_segment(buf, upto, upto >= len, &st, &kind,
                                     &off, &slen))
                break;
            if (kind == CWICK_SEG_REASONING) {
                if (rlen + slen + 1 < sizeof(o->reasoning)) {
                    memcpy(o->reasoning + rlen, buf + off, slen);
                    rlen += slen;
                    o->reasoning[rlen] = '\0';
                }
            } else if (clen + slen + 1 < sizeof(o->content)) {
                memcpy(o->content + clen, buf + off, slen);
                clen += slen;
                o->content[clen] = '\0';
            }
        }
        if (upto >= len) break;
    }
    o->tool_mode = st.tool_mode;
    free(buf);
}

/* A JSON document is UTF-8 by definition, and the detokenizer appends raw
 * token bytes, so a byte-fallback token can put half a character into an
 * answer. Every response has to survive being read back as UTF-8. */
static void test_json_escape(void) {
    struct { const char *in; const char *want; } cases[] = {
        /* well-formed multi-byte text passes through untouched */
        { "hi \xE4\xB8\x80\xE4\xB8\xAA there", "\"hi \xE4\xB8\x80\xE4\xB8\xAA there\"" },
        { "emoji \xF0\x9F\x8E\xA8 ok", "\"emoji \xF0\x9F\x8E\xA8 ok\"" },
        /* a lone continuation byte (what a split token leaves behind) */
        { "a\xA8" "b", "\"a\xEF\xBF\xBD" "b\"" },
        /* a truncated sequence at the end of the answer */
        { "end\xE4\xB8", "\"end\xEF\xBF\xBD\"" },
        { "end\xE4", "\"end\xEF\xBF\xBD\"" },
        /* an overlong encoding and a surrogate are not characters either:
         * one U+FFFD for the maximal subpart (Unicode Table 3-8), not one per
         * byte the way Python's error handler does it */
        { "\xC0\xAF", "\"\xEF\xBF\xBD\"" },
        { "\xED\xA0\x80", "\"\xEF\xBF\xBD\"" },
        /* 0xF8-0xFF are never lead bytes, and each stray 0x80 after one is
         * its own ill-formed byte */
        { "\xF8\x80\x80\x80", "\"\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD\"" },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        char *got = cwick_json_stringify(cases[i].in);
        CHECK(strcmp(got, cases[i].want) == 0, "json escape: ill-formed bytes");
        /* and the result parses back as the same string the writer saw */
        cwick_error e;
        cwick_error_clear(&e);
        json_value *v = cwick_json_parse(got, strlen(got), &e);
        CHECK(v && v->kind == JSON_STRING, "json escape: output parses");
        cwick_json_free(v);
        free(got);
    }
    /* the plain escapes still work */
    {
        char *got = cwick_json_stringify("a\"b\\c\nd\te");
        CHECK(strcmp(got, "\"a\\\"b\\\\c\\nd\\te\"") == 0, "json escape: quotes");
        free(got);
    }
}

static void test_stream_split(void) {
    static const size_t steps[] = { 0, 1, 3 };   /* whole, byte-wise, 3 bytes */
    split_out o;

    for (size_t s = 0; s < sizeof(steps) / sizeof(steps[0]); s++) {
        size_t step = steps[s];
        char msg[128];

        run_split("plan</think>answer", step, &o);
        snprintf(msg, sizeof(msg), "split: </think> handoff (step=%zu)", step);
        CHECK(strcmp(o.reasoning, "plan") == 0 && strcmp(o.content, "answer") == 0,
              msg);

        run_split("r</think>\n\nafter", step, &o);
        snprintf(msg, sizeof(msg), "split: newlines after </think> skipped (step=%zu)", step);
        CHECK(strcmp(o.reasoning, "r") == 0 && strcmp(o.content, "after") == 0, msg);

        run_split("plan</think>prose<tool_call>{\"name\":\"f\"}", step, &o);
        snprintf(msg, sizeof(msg), "split: <tool_call> cuts content (step=%zu)", step);
        CHECK(strcmp(o.reasoning, "plan") == 0 && strcmp(o.content, "prose") == 0
              && o.tool_mode, msg);

        run_split("a<tool_call>b</think>c", step, &o);
        snprintf(msg, sizeof(msg),
                 "split: <tool_call> in reasoning is not a call (step=%zu)", step);
        CHECK(strcmp(o.reasoning, "a<tool_call>b") == 0 &&
              strcmp(o.content, "c") == 0 && !o.tool_mode, msg);

        run_split("no closing tag", step, &o);
        snprintf(msg, sizeof(msg), "split: no </think> stays reasoning (step=%zu)", step);
        CHECK(strcmp(o.reasoning, "no closing tag") == 0 && o.content[0] == '\0'
              && !o.tool_mode, msg);

        run_split("好</think>吗？", step, &o);
        snprintf(msg, sizeof(msg), "split: multibyte UTF-8 never half-emitted (step=%zu)", step);
        CHECK(strcmp(o.reasoning, "好") == 0 && strcmp(o.content, "吗？") == 0, msg);

        run_split("a</think>b</think>c", step, &o);
        snprintf(msg, sizeof(msg), "split: second </think> is plain content (step=%zu)", step);
        CHECK(strcmp(o.reasoning, "a") == 0 &&
              strcmp(o.content, "b</think>c") == 0, msg);

        /* A checkpoint whose template stops at <|assistant|> writes the
         * opener itself; it is framing, so it must not reach
         * reasoning_content -- including when it arrives one byte at a time. */
        run_split("<think>plan</think>answer", step, &o);
        snprintf(msg, sizeof(msg), "split: model-written <think> opener dropped (step=%zu)", step);
        CHECK(strcmp(o.reasoning, "plan") == 0 && strcmp(o.content, "answer") == 0, msg);

        /* Anything else that merely starts with '<' is untouched. */
        run_split("<thin ice></think>ok", step, &o);
        snprintf(msg, sizeof(msg), "split: near-miss opener kept verbatim (step=%zu)", step);
        CHECK(strcmp(o.reasoning, "<thin ice>") == 0 && strcmp(o.content, "ok") == 0, msg);
    }
}

/* ------------------------------------------------------------------ */
/* Prometheus metrics registry (vLLM-compatible names)                 */
/* ------------------------------------------------------------------ */
static bool render_has(const char *body, const char *needle) {
    return strstr(body, needle) != NULL;
}

static void test_metrics(void) {
    /* A model name with a quote and a backslash exercises the label escaping
     * a scraper would otherwise choke on. */
    cwick_metrics_info mi;
    memset(&mi, 0, sizeof(mi));
    mi.model = "glm\"5.3\\flash";
    mi.backend = "cpu";
    mi.spine = "bf16";
    mi.max_context = 202752;
    mi.num_layers = 45;
    mi.num_experts = 160;
    mi.experts_per_token = 8;
    cwick_metrics_init(&mi);

    CHECK(strcmp(cwick_metrics_content_type(),
                 "text/plain; version=0.0.4; charset=utf-8") == 0,
          "metrics: content type is Prometheus text v0.0.4");

    /* ---- request lifecycle ---------------------------------------- */
    uint64_t rid = cwick_metrics_request_arrive();
    CHECK(rid == 1, "metrics: first request gets id 1");
    cwick_metrics_request_admit(rid, 0.25);          /* queue: 250 ms */
    cwick_metrics_request_refuse("busy");            /* a second one, refused */
    cwick_metrics_request_error("invalid_json");     /* a third one, broken */

    /* ---- one finished request: 11 tokens in 0.5 s of decode --------
     * Generation tokens and decode steps are engine-side and live (the
     * engine adds them while decoding); the prompt counter is finish-time,
     * because only the server knows the full client prompt (a prefix hit
     * replays most of it). request_finish adds the prompt once and never
     * re-adds the generation side. */
    cwick_metrics_add_generation_tokens(11);
    cwick_metrics_add_decode_steps(10);
    cwick_metrics_observe_decode_step(0.05);
    for (int i = 0; i < 10; i++) cwick_metrics_observe_iteration_tokens(1);
    cwick_metrics_observe_ttft(1.5);
    cwick_metrics_request_finish(29, 11, 2.0, 0.5, 3.0, 1.5, 64,
                                "length", true);
    /* A failed request must not land in the success counters. */
    cwick_metrics_request_finish(4, 0, 0.1, 0.0, 0.2, 0.1, 8, "stop", false);

    /* ---- render ---------------------------------------------------- */
    size_t len = 0;
    char *body = cwick_metrics_render(&len);
    CHECK(body != NULL && len > 0, "metrics: render produces a payload");

    /* A switch that released the resident checkpoint but failed to load the
     * replacement must be visible to a scraper; model_info is static and
     * would keep naming the old checkpoint forever. */
    CHECK(render_has(body, "# TYPE cwick:model_loaded gauge\n"
                           "cwick:model_loaded{model=\"glm\\\"5.3\\\\flash\"} 1"),
          "metrics: a loaded instance publishes model_loaded=1");
    cwick_metrics_set_model_loaded(false);
    free(body);
    body = cwick_metrics_render(&len);
    CHECK(render_has(body, "cwick:model_loaded{model=\"glm\\\"5.3\\\\flash\"} 0"),
          "metrics: a failed switch publishes model_loaded=0");
    cwick_metrics_set_model_loaded(true);
    free(body);
    body = cwick_metrics_render(&len);
    CHECK(render_has(body, "cwick:model_loaded{model=\"glm\\\"5.3\\\\flash\"} 1"),
          "metrics: a successful switch restores model_loaded=1");

    /* static info, with the quote and backslash escaped */
    CHECK(render_has(body, "cwick:model_info{model=\"glm\\\"5.3\\\\flash\","),
          "metrics: info metric escapes label quotes/backslashes");
    CHECK(render_has(body, "backend=\"cpu\",spine=\"bf16\",max_context=\"202752\","),
          "metrics: info metric carries backend/spine/context labels");

    /* vLLM names a dashboard written for vLLM can scrape unchanged.
     * The prompt counter is finish-time (server-side, full prompt), the
     * generation counter is engine-side (live); request_finish must not
     * re-add the latter. */
    CHECK(render_has(body, "# TYPE vllm:prompt_tokens_total counter\n"
                           "vllm:prompt_tokens_total 29"),
          "metrics: prompt tokens counted once (finish-time, not engine)");
    CHECK(render_has(body, "# TYPE vllm:generation_tokens_total counter\n"
                           "vllm:generation_tokens_total 11"),
          "metrics: generation tokens counted once");
    CHECK(render_has(body,
          "vllm:request_success_by_reason_total{finished_reason=\"length\"} 1"),
          "metrics: finish reason breakdown (length)");
    CHECK(render_has(body,
          "vllm:request_success_by_reason_total{finished_reason=\"stop\"} 0"),
          "metrics: failed request adds no finish reason");
    CHECK(render_has(body, "vllm:request_failure_total 3"),
          "metrics: refusals, errors and failed finishes all counted");

    /* gauges: the slot is free again after both requests finished */
    CHECK(render_has(body, "# TYPE vllm:num_requests_running gauge\n"
                           "vllm:num_requests_running 0"),
          "metrics: running gauge returns to 0");
    CHECK(render_has(body, "# TYPE vllm:num_requests_waiting gauge\n"
                           "vllm:num_requests_waiting 0"),
          "metrics: waiting gauge drained");

    /* histograms: cumulative buckets, +Inf, _sum, _count */
    CHECK(render_has(body,
          "vllm:time_per_output_token_seconds_bucket{le=\"0.05\"} 1\n"
          "vllm:time_per_output_token_seconds_bucket{le=\"0.075\"} 1"),
          "metrics: TPOT histogram bucket is cumulative");
    CHECK(render_has(body,
          "vllm:time_per_output_token_seconds_bucket{le=\"0.025\"} 0"),
          "metrics: TPOT histogram lower bucket empty");
    CHECK(render_has(body, "vllm:time_per_output_token_seconds_count 1"),
          "metrics: TPOT count");
    CHECK(render_has(body, "vllm:time_per_output_token_seconds_sum 0.05"),
          "metrics: TPOT sum is decode/step count");
    CHECK(render_has(body,
          "cwick:decode_step_seconds_bucket{le=\"0.05\"} 1\n"
          "cwick:decode_step_seconds_bucket{le=\"0.1\"} 1"),
          "metrics: decode-step histogram (the tok/s distribution)");
    CHECK(render_has(body, "cwick:decode_step_seconds_count 1"),
          "metrics: decode-step count");
    CHECK(render_has(body,
          "vllm:iteration_tokens_total_bucket{le=\"1\"} 10\n"),
          "metrics: iteration_tokens observes every step (not once per request)");
    CHECK(render_has(body, "vllm:iteration_tokens_total_count 10"),
          "metrics: iteration_tokens count equals decode iterations");
    CHECK(render_has(body, "vllm:time_to_first_token_seconds_sum 1.5"),
          "metrics: TTFT observed at first-token time");

    /* cwick-only gauges exist even with no stream attached */
    CHECK(render_has(body, "# TYPE cwick:expert_cache_bytes gauge\n"
                           "cwick:expert_cache_bytes 0"),
          "metrics: expert-cache gauge present (no stream attached)");
    CHECK(render_has(body, "# TYPE cwick:process_footprint_bytes gauge"),
          "metrics: process footprint gauge present");
    CHECK(render_has(body, "# TYPE cwick:uptime_seconds gauge"),
          "metrics: uptime gauge present");
    CHECK(render_has(body, "# TYPE vllm:num_preemptions_total counter\n"
                           "vllm:num_preemptions_total 0"),
          "metrics: preemption counter exported for dashboard parity");

    free(body);
}

/* ------------------------------------------------------------------ */
/* ------------------------------------------------------------------ */
/* Decode-phase token accounting (thinking / content / tools)           */
/* ------------------------------------------------------------------ */
/* One encode -> account -> assert cycle. The byte-fallback tokenizer
 * normally hands the markers over one byte at a time, which is the split
 * case the accounting has to survive. */
static void test_phase_account(void) {
    char *root = make_tmpdir();
    cwick_error err;
    cwick_error_clear(&err);
    CHECK(cwick_setup_synthetic(root, &err) == CWICK_OK,
          "phase: synthetic setup");
    cwick_engine *eng = NULL;
    cwick_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_root = root;
    opt.max_new_tokens = 8;
    CHECK(cwick_engine_new(&opt, &eng, &err) == CWICK_OK, "phase: engine new");
    if (eng) {
        cwick_tokenizer *tok = NULL;
        char tdir[1152];
        cwick_path_join(tdir, sizeof(tdir), root, "tokenizer");
        CHECK(cwick_tokenizer_load(tdir, cwick_engine_spec(eng), &tok, &err)
                  == CWICK_OK,
              "phase: tokenizer (byte fallback)");
        if (tok) {
            /* The checkpoint's XML markers, spelled char-by-char: the
             * contiguous literal does not survive editors or filters intact,
             * and this test must exercise the real bytes. */
            static const char kThinkClose[] = { '<', '/', 't', 'h', 'i',
                                                'n', 'k', '>', '\0' };
            static const char kToolOpen[] = { '<', 't', 'o', 'o', 'l', '_',
                                              'c', 'a', 'l', 'l', '>', '\0' };
            const char *tclose = kThinkClose;
            const char *topen = kToolOpen;

            uint32_t *ids = NULL;
            int64_t n = 0;
            size_t out[3] = { 0, 0, 0 };

            /* 1. All three phases in one chat answer. */
            char text[320];
            snprintf(text, sizeof(text),
                     "let me think%s%s\nvisible answer%s",
                     tclose, tclose, topen);
            CHECK(cwick_tokenizer_encode(tok, text, &ids, &n, &err) == CWICK_OK
                      && n > 0,
                  "phase: encode 3-phase answer");
            if (n > 0) {
                cwick_phase_account(tok, (const uint32_t *)ids, (size_t)n, true,
                                   out);
                CHECK(out[0] + out[1] + out[2] == (size_t)n,
                      "phase: counts sum to the token count");
                CHECK(out[0] > 0 && out[1] > 0 && out[2] > 0,
                      "phase: chat answer splits into thinking/content/tools");
            }
            free(ids);
            ids = NULL;
            n = 0;
            out[0] = out[1] = out[2] = 0;

            /* 2. Completions never reason: without a tool marker every
             *    token is content, whatever the tokenization. */
            CHECK(cwick_tokenizer_encode(tok, "just a plain answer", &ids, &n,
                                        &err) == CWICK_OK && n > 0,
                  "phase: encode completion");
            if (n > 0) {
                cwick_phase_account(tok, (const uint32_t *)ids, (size_t)n, false,
                                   out);
                CHECK(out[0] == 0 && out[1] == (size_t)n && out[2] == 0,
                      "phase: completion without markers is all content");
            }
            free(ids);
            ids = NULL;
            n = 0;
            out[0] = out[1] = out[2] = 0;

            /* 3. A chat answer that never leaves the reasoning block: the
             *    budget ran out mid-think, and the log must say so. */
            CHECK(cwick_tokenizer_encode(tok, "still thinking", &ids, &n, &err)
                      == CWICK_OK && n > 0,
                  "phase: encode unclosed think");
            if (n > 0) {
                cwick_phase_account(tok, (const uint32_t *)ids, (size_t)n, true,
                                   out);
                CHECK(out[0] == (size_t)n && out[1] == 0 && out[2] == 0,
                      "phase: unclosed think is all thinking");
            }
            free(ids);
            ids = NULL;
            n = 0;
            out[0] = out[1] = out[2] = 0;

            /* 4. Degenerate calls must not crash and must report zeros. */
            cwick_phase_account(tok, NULL, 0, true, out);
            CHECK(out[0] == 0 && out[1] == 0 && out[2] == 0,
                  "phase: empty input reports zeroes");

            cwick_tokenizer_free(tok);
        }
        cwick_engine_free(eng);
    }
}

/* ------------------------------------------------------------------ */
/* Thinking intensity (思考强度)                                         */
/* ------------------------------------------------------------------ */
static void test_thinking_effort(void) {
    struct { const char *in; const char *level; const char *effort; } cases[] = {
        { "低", "低", "Low" },     { "中", "中", "Low" },
        { "高", "高", "High" },    { "超高", "超高", "Max" },
        { "极致", "极致", "Max" },  { "关", "关", "Low" },
        { "关闭", "关", "Low" },   { "无", "关", "Low" },
        { "low", "低", "Low" },    { "LOW", "低", "Low" },
        { "minimal", "低", "Low" }, { "medium", "中", "Low" },
        { "Medium", "中", "Low" }, { "high", "高", "High" },
        { "High", "高", "High" },  { "ultra", "超高", "Max" },
        { "extreme", "极致", "Max" }, { "max", "极致", "Max" },
        { "none", "关", "Low" },   { "off", "关", "Low" },
    };
    const char *level = NULL, *effort = NULL;
    bool ok = true;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        level = effort = NULL;
        if (!cwick_thinking_effort_resolve(cases[i].in, &level, &effort) ||
            strcmp(level, cases[i].level) != 0 ||
            strcmp(effort, cases[i].effort) != 0) {
            ok = false;
            fprintf(stderr, "  effort \"%s\" -> %s/%s (want %s/%s)\n",
                    cases[i].in, level ? level : "(null)",
                    effort ? effort : "(null)", cases[i].level,
                    cases[i].effort);
        }
    }
    CHECK(ok, "effort: five levels resolve (zh + en, case-insensitive)");
    CHECK(!cwick_thinking_effort_resolve("banana", &level, &effort) &&
          !cwick_thinking_effort_resolve("", &level, &effort) &&
          !cwick_thinking_effort_resolve(NULL, &level, &effort) &&
          !cwick_thinking_effort_resolve("mediumish", &level, &effort),
          "effort: unknown values refused (fail-closed)");
}

/* ------------------------------------------------------------------ */
/* Prefix checkpoints taken part-way through a prefill                  */
/* ------------------------------------------------------------------ */
/* The engine may snapshot the state at a chunk boundary while a long
 * prefill is still running, so that a later prompt sharing only the head of
 * this one can replay that head. The snapshot has to be a *legal* prefix
 * state -- the recurrent parts (DeltaNet accumulator, conv window, indexer
 * pool) are folded over the tokens so far and cannot be rewound, so this is
 * the only way to get them: apply the checkpoint and prefill the rest, and
 * the generation must come out identical to one full prefill. */
typedef struct {
    cwick_prefix_state st[8];
    int64_t pos[8];
    int n;
} cap_log;

static void pfx_block_done(void *user, cwick_prefix_state *s) {
    cap_log *l = (cap_log *)user;
    if (!s) return;
    if (l->n < 8) {
        l->st[l->n] = *s;          /* takes the buffers; the shell is freed */
        l->pos[l->n] = s->pos;
        l->n++;
        free(s);
    } else {
        cwick_prefix_state_free(s);
    }
}

static void test_prefix_block_checkpoints(void) {
    char *root = make_tmpdir();
    cwick_error err;
    cwick_error_clear(&err);
    CHECK(cwick_setup_synthetic_mhc(root, &err) == CWICK_OK, "pfxblk: setup");
    cwick_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_root = root;
    opt.max_new_tokens = 4;
    cwick_engine *eng = NULL;
    CHECK(cwick_engine_new(&opt, &eng, &err) == CWICK_OK, "pfxblk: engine new");
    if (!eng) return;   /* root is a static buffer (make_tmpdir), never freed */

    uint32_t ids[12];
    for (int i = 0; i < 12; i++) ids[i] = (uint32_t)(3 + i * 7);
    const int64_t nids = 12;

    /* Reference: prefill the whole prompt in one chunk. */
    cwick_engine_reset(eng, &err);
    cwick_gen_result ref;
    memset(&ref, 0, sizeof(ref));
    CHECK(cwick_engine_generate(eng, ids, nids, 4, false, &ref, &err) == CWICK_OK,
          "pfxblk: reference generation");

    /* Same prompt, chunked every 4 tokens, with a snapshot at each boundary. */
    cap_log log;
    memset(&log, 0, sizeof(log));
    /* Heap-allocated: the done callback owns and frees whatever it is handed,
     * and the end-of-prefill snapshot arrives through it too. */
    cwick_prefix_state *final_st = calloc(1, sizeof(*final_st));
    /* The synthetic engine has no memory plan, so its RAM-bounded chunk is 1
     * (token-serial). Force the chunk the real server would pick for a long
     * prompt; CWICK_PREFILL_CHUNK is applied after the block cap, so this
     * exercises the checkpoint-at-boundary path itself. */
    setenv("CWICK_PREFILL_CHUNK", "4", 1);
    /* The chunk-attention path has its own equivalence test; this one is about
     * the cross-chunk KV/KDA carry, which is what makes a boundary snapshot a
     * legal prefix state. */
    setenv("CWICK_NO_CHUNK_ATTN", "1", 1);
    cwick_engine_reset(eng, &err);
    cwick_engine_set_prefix_capture_block(eng, 4);
    cwick_engine_set_prefix_capture(eng, final_st);
    cwick_engine_set_prefix_capture_done(eng, pfx_block_done, &log);
    cwick_gen_result chunked;
    memset(&chunked, 0, sizeof(chunked));
    CHECK(cwick_engine_generate(eng, ids, nids, 4, false, &chunked, &err)
              == CWICK_OK, "pfxblk: chunked generation");
    unsetenv("CWICK_PREFILL_CHUNK");
    unsetenv("CWICK_NO_CHUNK_ATTN");
    cwick_engine_set_prefix_capture(eng, NULL);
    cwick_engine_set_prefix_capture_done(eng, NULL, NULL);
    cwick_engine_set_prefix_capture_block(eng, 0);

    /* 12 tokens in chunks of 4 -> boundaries at 4, 8 and the end (12). */

    /* Chunking must not change the arithmetic -- only the order. */
    int same_as_ref = ref.n_tokens > 0 && chunked.n_tokens == ref.n_tokens;
    for (int64_t i = 0; i < ref.n_tokens && same_as_ref; i++)
        if (ref.tokens[i] != chunked.tokens[i]) same_as_ref = 0;
    CHECK(same_as_ref, "pfxblk: chunked prefill == single-chunk prefill");

    /* Each intermediate checkpoint, replayed, must continue identically. */
    int all_replay_ok = log.n == 3;
    int replayed = 0;
    for (int k = 0; k < log.n && all_replay_ok; k++) {
        const int64_t P = log.pos[k];
        if (P >= nids) continue;   /* the end-of-prompt snapshot has no tail */
        replayed++;
        cwick_engine_reset(eng, &err);
        if (cwick_engine_apply_prefix(eng, &log.st[k], &err) != CWICK_OK) {
            all_replay_ok = 0;
            break;
        }
        cwick_gen_result split;
        memset(&split, 0, sizeof(split));
        if (cwick_engine_generate(eng, ids + P, nids - P, 4, false, &split,
                                 &err) != CWICK_OK) {
            all_replay_ok = 0;
        } else {
            if (split.n_tokens != ref.n_tokens || ref.n_tokens == 0)
                all_replay_ok = 0;
            for (int64_t i = 0; i < ref.n_tokens && all_replay_ok; i++)
                if (ref.tokens[i] != split.tokens[i]) all_replay_ok = 0;
            cwick_gen_result_free(&split);
        }
    }
    CHECK(all_replay_ok && replayed == 2,
          "pfxblk: every checkpoint replays to the same answer");

    cwick_gen_result_free(&chunked);

    /* With no block set, only the end-of-prefill snapshot is taken. */
    cap_log log2;
    memset(&log2, 0, sizeof(log2));
    cwick_prefix_state *final2 = calloc(1, sizeof(*final2));
    cwick_engine_reset(eng, &err);
    cwick_engine_set_prefix_capture_block(eng, 0);
    cwick_engine_set_prefix_capture(eng, final2);
    cwick_engine_set_prefix_capture_done(eng, pfx_block_done, &log2);
    cwick_gen_result plain;
    memset(&plain, 0, sizeof(plain));
    cwick_engine_generate(eng, ids, nids, 4, false, &plain, &err);
    cwick_engine_set_prefix_capture(eng, NULL);
    cwick_engine_set_prefix_capture_done(eng, NULL, NULL);
    cwick_gen_result_free(&plain);
    CHECK(log2.n == 1 && log2.pos[0] == 12,
          "pfxblk: block 0 keeps the old one-snapshot behaviour");

    for (int k = 0; k < log.n; k++) cwick_prefix_state_free(&log.st[k]);
    for (int k = 0; k < log2.n; k++) cwick_prefix_state_free(&log2.st[k]);
    cwick_gen_result_free(&ref);
    cwick_engine_free(eng);
}

/* The --web assets are compiled in, so "the UI ships" is a property of the
 * binary and can be asserted without opening a socket. What matters:
 *   - the document and both assets are present and non-trivial;
 *   - index aliases resolve to the document (a browser asks for "/");
 *   - query strings are ignored (browsers add them freely);
 *   - nothing else resolves, so an unknown path still gets the normal 404
 *     instead of accidentally serving a byte array. */

/* The decision endpoint refuses a body it cannot honour before it touches the
 * model, which is the part that can be checked without one. The shape of the
 * response is checked end to end against a resident model instead (curl in the
 * README): a unit test cannot invent a checkpoint that answers a letter. */
static void test_jev_errors(void) {
    char *out = NULL;
    int64_t n = 0;
    char err[256];
    int rc;

    rc = jev_decide(NULL, NULL, "{}", 2, &out, &n, err, sizeof(err));
    CHECK(rc != 0 && out == NULL && strstr(err, "no resident model"),
          "jev: no engine is refused by name");

    rc = jev_decide((cwick_engine *)1, (cwick_tokenizer *)1, "not json", 8,
                    &out, &n, err, sizeof(err));
    CHECK(rc != 0 && strstr(err, "invalid JSON"),
          "jev: a malformed body names the parse failure");

    {
        const char *b = "{\"questions\":{}}";
        rc = jev_decide((cwick_engine *)1, (cwick_tokenizer *)1, b, strlen(b),
                        &out, &n, err, sizeof(err));
    }
    CHECK(rc != 0 && strstr(err, "state"),
          "jev: a body without state is refused before the model");

    {
        const char *b = "{\"state\":\"x\",\"questions\":{}}";
        rc = jev_decide((cwick_engine *)1, (cwick_tokenizer *)1, b, strlen(b),
                        &out, &n, err, sizeof(err));
    }
    CHECK(rc != 0 && strstr(err, "questions"),
          "jev: an empty question set is refused");

    /* Eleven questions, one over the limit: the check must be about the count,
     * not about the model. */
    {
        char body[512];
        int w = snprintf(body, sizeof(body), "{\"state\":\"x\",\"questions\":{");
        for (int i = 0; i < JEV_MAX_QUESTIONS + 1; i++)
            w += snprintf(body + w, sizeof(body) - (size_t)w, "%s\"q%d\":{\"type\":\"noul\"}",
                          i ? "," : "", i);
        snprintf(body + w, sizeof(body) - (size_t)w, "}}");
        rc = jev_decide((cwick_engine *)1, (cwick_tokenizer *)1, body,
                        (size_t)strlen(body), &out, &n, err, sizeof(err));
        CHECK(rc != 0 && strstr(err, "exceed"),
              "jev: more questions than the limit is refused");
    }

    /* A question type outside the three the protocol defines. */
    {
        const char *b = "{\"state\":\"x\",\"questions\":{\"a\":{\"type\":\"rank\"}}}";
        rc = jev_decide((cwick_engine *)1, (cwick_tokenizer *)1, b, strlen(b),
                        &out, &n, err, sizeof(err));
    }
    CHECK(rc != 0 && strstr(err, "unknown type"),
          "jev: an unknown question type is refused");

    /* Two refusal classes. The reference protocol separates "this is not a
     * decision request" (422) from "well-formed, over a limit or unsupported"
     * (400); a client can act on the difference, and one status for both means
     * it cannot. */
    {
        const char *b = "not json";
        rc = jev_decide((cwick_engine *)1, (cwick_tokenizer *)1, b, strlen(b),
                        &out, &n, err, sizeof(err));
        CHECK(rc == -2, "jev: a malformed body is a 422, not a 400");
    }
    {
        const char *b = "{\"state\":\"x\",\"questions\":{\"a\":{\"type\":\"choice\","
                        "\"instructions\":\"i\",\"criteria\":{\"x\":\"y\"}},"
                        "\"b\":{\"type\":\"score\",\"instructions\":\"i\","
                        "\"criteria\":[\"only\"]}}}";
        rc = jev_decide((cwick_engine *)1, (cwick_tokenizer *)1, b, strlen(b),
                        &out, &n, err, sizeof(err));
        CHECK(rc == -1 && strstr(err, "2 to 10"),
              "jev: a one-level score is a 400 about the level range");
    }

    /* `state` is a string *or* an object in the reference; an object is
     * serialized. Proven the only way a unit test can: the body reaches the
     * *next* complaint, which is about the score level, not the state. */
    {
        const char *b = "{\"state\":{\"board\":[1,2],\"turn\":\"O\"},"
                        "\"questions\":{\"a\":{\"type\":\"score\","
                        "\"instructions\":\"i\",\"criteria\":[\"only\"]}}}";
        rc = jev_decide((cwick_engine *)1, (cwick_tokenizer *)1, b, strlen(b),
                        &out, &n, err, sizeof(err));
        CHECK(rc == -1 && strstr(err, "2 to 10") && !strstr(err, "state"),
              "jev: a JSON object state is accepted and serialized");
    }
    {
        const char *b = "{\"questions\":{}}";
        rc = jev_decide((cwick_engine *)1, (cwick_tokenizer *)1, b, strlen(b),
                        &out, &n, err, sizeof(err));
        CHECK(rc == -2 && strstr(err, "state is required"),
              "jev: a body without state is a 422 naming the field");
    }

    /* A decision has no stream, no reasoning pass and no image input. All
     * three are refused by name rather than accepted and silently narrowed. */
    {
        const char *b = "{\"state\":\"x\",\"stream\":true,\"questions\":{}}";
        rc = jev_decide((cwick_engine *)1, (cwick_tokenizer *)1, b, strlen(b),
                        &out, &n, err, sizeof(err));
        CHECK(rc == -1 && strstr(err, "stream"),
              "jev: stream=true is refused (a decision is one JSON object)");
    }
    {
        const char *b = "{\"state\":\"x\",\"reasoning_effort\":\"high\","
                        "\"questions\":{}}";
        rc = jev_decide((cwick_engine *)1, (cwick_tokenizer *)1, b, strlen(b),
                        &out, &n, err, sizeof(err));
        CHECK(rc == -1 && strstr(err, "reasoning_effort"),
              "jev: reasoning_effort other than none is refused");
    }
    {
        const char *b = "{\"state\":\"x\",\"thinking\":{\"type\":\"enabled\"},"
                        "\"questions\":{}}";
        rc = jev_decide((cwick_engine *)1, (cwick_tokenizer *)1, b, strlen(b),
                        &out, &n, err, sizeof(err));
        CHECK(rc == -1 && strstr(err, "thinking"),
              "jev: thinking on is refused");
    }
    /* ...while the off spellings pass, which is what a client that shares one
     * body builder across endpoints sends. */
    {
        const char *b = "{\"state\":\"x\",\"stream\":false,\"thinking\":false,"
                        "\"questions\":{\"a\":{\"type\":\"score\","
                        "\"instructions\":\"i\",\"criteria\":[\"only\"]}}}";
        rc = jev_decide((cwick_engine *)1, (cwick_tokenizer *)1, b, strlen(b),
                        &out, &n, err, sizeof(err));
        CHECK(rc == -1 && strstr(err, "2 to 10") && !strstr(err, "stream") &&
              !strstr(err, "thinking"),
              "jev: stream=false and thinking=false are accepted");
    }
    {
        const char *b = "{\"state\":\"x\",\"images\":[{\"data\":\"…\"}],"
                        "\"questions\":{}}";
        rc = jev_decide((cwick_engine *)1, (cwick_tokenizer *)1, b, strlen(b),
                        &out, &n, err, sizeof(err));
        CHECK(rc == -1 && strstr(err, "images"),
              "jev: image input is refused");
    }

    /* `instructions` is required: the old fallback ("Which option applies?")
     * was a silent substitution of the question being asked. */
    {
        const char *b = "{\"state\":\"x\",\"questions\":{\"a\":{\"type\":\"noul\"}}}";
        rc = jev_decide((cwick_engine *)1, (cwick_tokenizer *)1, b, strlen(b),
                        &out, &n, err, sizeof(err));
        CHECK(rc == -2 && strstr(err, "`instructions` is required"),
              "jev: a question without instructions is a 422 naming the field");
    }

    /* noul criteria describes the two sides of one statement. The order in the
     * JSON must not decide which side `noul` reports: the true side is always
     * A. Two spellings, both accepted, plus the refusal for anything else. */
    {
        const char *b = "{\"state\":\"x\",\"questions\":{"
                        "\"done\":{\"type\":\"noul\",\"instructions\":\"i\","
                        "\"criteria\":{\"false\":\"not done\",\"true\":\"done\"}},"
                        "\"bad\":{\"type\":\"score\",\"instructions\":\"i\","
                        "\"criteria\":[\"only\"]}}}";
        rc = jev_decide((cwick_engine *)1, (cwick_tokenizer *)1, b, strlen(b),
                        &out, &n, err, sizeof(err));
        CHECK(rc == -1 && strstr(err, "bad") && strstr(err, "2 to 10"),
              "jev: noul criteria {false,true} parses in the written order");
    }
    {
        const char *b = "{\"state\":\"x\",\"questions\":{\"a\":{\"type\":\"noul\","
                        "\"instructions\":\"i\",\"criteria\":{\"yes\":\"y\","
                        "\"no\":\"n\"}}}}";
        rc = jev_decide((cwick_engine *)1, (cwick_tokenizer *)1, b, strlen(b),
                        &out, &n, err, sizeof(err));
        CHECK(rc == -2 && strstr(err, "`true` and `false`"),
              "jev: noul criteria must name the two sides");
    }
    {
        const char *b = "{\"state\":\"x\",\"questions\":{"
                        "\"ok\":{\"type\":\"noul\",\"instructions\":\"i\","
                        "\"criteria\":[\"done\",\"not done\"]},"
                        "\"bad\":{\"type\":\"score\",\"instructions\":\"i\","
                        "\"criteria\":[\"only\"]}}}";
        rc = jev_decide((cwick_engine *)1, (cwick_tokenizer *)1, b, strlen(b),
                        &out, &n, err, sizeof(err));
        CHECK(rc == -1 && strstr(err, "bad") && strstr(err, "2 to 10"),
              "jev: noul criteria as a two-item array is accepted");
    }
}

/* The adapter is what makes one endpoint serve three model families. These
 * checks pin the mapping (spec -> prompt grammar) and the name check that
 * turns "a model id cached before a switch" into a 400 instead of a decision
 * attributed to the wrong checkpoint. */
static void test_jev_adapter(void) {
    cwick_model_spec s;
    jev_adapter a;
    memset(&s, 0, sizeof(s));

    snprintf(s.model_type, sizeof(s.model_type), "glm5_next");
    jev_adapter_default(&s, &a);
    CHECK(strcmp(a.family, "glm") == 0 && strcmp(a.prompt, "glm-chat-closed") == 0,
          "jev: a GLM checkpoint gets the closed-think chat grammar");
    CHECK(a.letters == -1 && a.first_missing == -1,
          "jev: the letter probe is unrun until a tokenizer is present");

    snprintf(s.model_type, sizeof(s.model_type), "deepseek_v4");
    jev_adapter_default(&s, &a);
    CHECK(strcmp(a.family, "deepseek") == 0 &&
          strcmp(a.prompt, "deepseek-chat-closed") == 0,
          "jev: a DeepSeek checkpoint gets its own closed-think grammar");

    snprintf(s.model_type, sizeof(s.model_type), "minimax_m3");
    jev_adapter_default(&s, &a);
    CHECK(strcmp(a.family, "minimax") == 0 &&
          strcmp(a.prompt, "minimax-chat-plain") == 0,
          "jev: MiniMax gets the plain chat grammar (thinking off)");

    /* No tokenizer: refused by name, not with a crash or a silent zero. */
    {
        char err[256];
        int rc = jev_probe(&s, NULL, &a, err, sizeof(err));
        CHECK(rc != 0 && strstr(err, "tokenizer"),
              "jev: a probe without a tokenizer is refused by name");
    }
    {
        char err[256];
        char *out = NULL;
        int rc = jev_describe(NULL, NULL, &out, err, sizeof(err));
        CHECK(rc != 0 && out == NULL,
              "jev: the adapter describe needs a resident model");
    }

    /* Which names answer depends on the resident checkpoint, not on the
     * catalog: the alias and model_type are both legal, anything else is a
     * stale id from before a switch. */
    CHECK(jev_model_is_served(NULL, "glm-5.3-flash", "glm5_next"),
          "jev: an omitted model means the resident one");
    CHECK(jev_model_is_served("", "glm-5.3-flash", "glm5_next"),
          "jev: an empty model means the resident one");
    CHECK(jev_model_is_served("glm-5.3-flash", "glm-5.3-flash", "glm5_next"),
          "jev: the served alias is accepted");
    CHECK(jev_model_is_served("GLM-5.3-Flash", "glm-5.3-flash", "glm5_next"),
          "jev: the alias compare is case-insensitive");
    CHECK(jev_model_is_served("glm5_next", "glm-5.3-flash", "glm5_next"),
          "jev: model_type is accepted too");
    CHECK(!jev_model_is_served("deepSeek-v4.1-flash-1017", "glm-5.3-flash",
                               "glm5_next"),
          "jev: a model id from before a switch is refused");
}

/* The model library is C state plus filesystem walking, so its rules can be
 * checked without a network: owner/name layout, completeness, resumability,
 * and the refuse-list for ids that could write outside ./models. */
static void test_hub(void) {
    char tmpl[] = "/tmp/cwick-test-hub-XXXXXX";
    char *root = mkdtemp(tmpl);
    CHECK(root != NULL, "hub: a temporary model library can be created");
    if (!root) return;

    char owner[512], dir[512], cfg[512], weights[512], partial[512];
    snprintf(owner, sizeof(owner), "%s/acme", root);
    snprintf(dir, sizeof(dir), "%s/tiny", owner);
    snprintf(cfg, sizeof(cfg), "%s/config.json", dir);
    snprintf(weights, sizeof(weights), "%s/model.bin", dir);
    snprintf(partial, sizeof(partial), "%s/model.bin.incomplete", dir);
    CHECK(cwick_mkdir_p(dir) == 0, "hub: owner/name layout is created");
    CHECK(cwick_write_file(cfg, "{}", 2) == 0, "hub: config.json written");
    CHECK(cwick_write_file(weights, "w", 1) == 0, "hub: a weight file written");

    cwick_hub *h = cwick_hub_open(root, NULL);
    CHECK(h != NULL, "hub: library opens without a network");
    if (!h) return;
    CHECK(strcmp(cwick_hub_endpoint(h), "https://hf-mirror.com") == 0,
          "hub: the default catalog host is used when none is given");
    CHECK(strcmp(cwick_hub_root(h), root) == 0, "hub: the library root is honoured");

    char err[256] = "";
    char *js = cwick_hub_models(h, err, sizeof(err));
    CHECK(js && strstr(js, "\"complete\":true") != NULL &&
          strstr(js, "acme/tiny") != NULL,
          "hub: a finished checkpoint under owner/name is listed as complete");
    CHECK(js && strstr(js, "\"supported\":false") != NULL &&
          strstr(js, "\"unsupported_reason\"") != NULL,
          "hub: the inventory reports whether the engine can load the checkpoint");
    free(js);

    char *p = cwick_hub_local_path(h, "acme/tiny");
    CHECK(p && strstr(p, "acme/tiny") != NULL,
          "hub: a repo id resolves to its checkpoint directory");
    free(p);
    p = cwick_hub_local_path(h, "tiny");
    CHECK(p != NULL, "hub: a bare name resolves under its owner");
    free(p);
    p = cwick_hub_local_path(h, "../escape");
    CHECK(p == NULL, "hub: path traversal is refused");
    free(p);

    CHECK(cwick_write_file(partial, "abc", 3) == 0, "hub: a partial file exists");
    p = cwick_hub_local_path(h, "acme/tiny");
    CHECK(p == NULL, "hub: a half-downloaded checkpoint cannot be switched to");
    free(p);
    {
        char resolved[1024];
        CHECK(cwick_hub_model_dir(h, "acme/tiny", resolved, sizeof(resolved)),
              "hub: a partial download still resolves, so it can be cleaned up");
    }
    js = cwick_hub_models(h, err, sizeof(err));
    CHECK(js && strstr(js, "\"resumable\":true") != NULL &&
          strstr(js, "\"complete\":false") != NULL,
          "hub: the inventory marks the same directory resumable");
    free(js);
    unlink(partial);

    js = cwick_hub_downloads(h, err, sizeof(err));
    CHECK(js && strstr(js, "\"jobs\":[]") != NULL,
          "hub: an empty download queue is valid JSON");
    free(js);

    char *bad = cwick_hub_download_start(h, "not-a-repo", err, sizeof(err));
    CHECK(bad == NULL && err[0], "hub: a malformed repo id is refused by name");
    free(bad);
    bad = cwick_hub_download_start(h, "../escape", err, sizeof(err));
    CHECK(bad == NULL, "hub: a traversing repo id is refused");
    free(bad);
    bad = cwick_hub_download_action(h, "d99", "pause", err, sizeof(err));
    CHECK(bad == NULL, "hub: an unknown job id is refused");
    free(bad);
    bad = cwick_hub_download_action(h, "d1", "explode", err, sizeof(err));
    CHECK(bad == NULL, "hub: an unknown action is refused");
    free(bad);

    /* Deleting a checkpoint: resolution and removal are separate calls so the
     * server can refuse a resident model between them. Both are confined to
     * the library root. */
    {
        char gone[512], gcfg[560], gweight[560], resolved[1024];
        snprintf(gone, sizeof(gone), "%s/acme/gone", root);
        snprintf(gcfg, sizeof(gcfg), "%s/config.json", gone);
        snprintf(gweight, sizeof(gweight), "%s/model.bin", gone);
        CHECK(cwick_mkdir_p(gone) == 0, "hub: a checkpoint to delete is created");
        CHECK(cwick_write_file(gcfg, "{}", 2) == 0, "hub: its config is written");
        CHECK(cwick_write_file(gweight, "12345", 5) == 0,
              "hub: its weight is written");

        CHECK(cwick_hub_model_dir(h, "acme/gone", resolved, sizeof(resolved)) &&
              strstr(resolved, "acme/gone") != NULL,
              "hub: owner/name resolves to a directory that may be deleted");
        CHECK(cwick_hub_model_dir(h, "gone", resolved, sizeof(resolved)),
              "hub: a bare name resolves under its owner");
        CHECK(!cwick_hub_model_dir(h, "..", resolved, sizeof(resolved)) &&
              !cwick_hub_model_dir(h, "/etc", resolved, sizeof(resolved)) &&
              !cwick_hub_model_dir(h, ".", resolved, sizeof(resolved)),
              "hub: traversal, absolute paths and the root itself resolve to nothing");

        int st = 0;
        char *del = cwick_hub_delete(h, "acme/gone", &st, err, sizeof(err));
        CHECK(del && strstr(del, "\"deleted\":true") != NULL &&
              strstr(del, "\"files\":2") != NULL,
              "hub: delete removes the checkpoint and reports what it freed");
        free(del);
        CHECK(!cwick_dir_exists(gone), "hub: the deleted directory is gone");
        CHECK(!cwick_hub_model_dir(h, "acme/gone", resolved, sizeof(resolved)),
              "hub: the deleted name no longer resolves");

        st = 0;
        del = cwick_hub_delete(h, "acme/missing", &st, err, sizeof(err));
        CHECK(del == NULL && st == 404,
              "hub: deleting an unknown checkpoint is a 404");
        st = 0;
        del = cwick_hub_delete(h, "../escape", &st, err, sizeof(err));
        CHECK(del == NULL && st == 404,
              "hub: deleting outside the library is refused");
        free(del);
    }

    cwick_hub_close(h);
    unlink(cfg); unlink(weights);
    rmdir(dir); rmdir(owner); rmdir(root);
}

/* The apps decide through the decision endpoint, never through chat. This is a
 * grep over the compiled-in bundles, which is the only form the rule can take
 * that cannot drift away from what the server actually serves: editing an app
 * back to /v1/chat/completions fails the suite before it ships. */
static void test_apps_use_systemone(void) {
    static const char *paths[] = {
        "/apps/tictactoe/app.js", "/apps/kart/app.js",
    };
    for (size_t i = 0; i < sizeof(paths) / sizeof(paths[0]); i++) {
        const cwick_webui_blob *b = cwick_webui_lookup(paths[i]);
        CHECK(b != NULL, "apps: every bundle is embedded");
        if (!b) continue;
        char *body = cwick_xmalloc(b->len + 1);
        memcpy(body, b->data, b->len);
        body[b->len] = '\0';
        CHECK(strstr(body, "/systemone") != NULL,
              "apps: the bundle calls /v1/systemone");
        CHECK(strstr(body, "/chat/completions") == NULL,
              "apps: the bundle never calls /v1/chat/completions");
        free(body);
    }
}

/* The goal names two specific app contracts, not just "an app that calls the
 * decision endpoint". Grep the compiled-in bundles -- the exact bytes the
 * server serves -- so a change to the prompt, the storage declaration or the
 * sidebar cannot quietly drift away from the requirement. */
static char *apps_blob_text(const cwick_webui_blob *b) {
    if (!b) return NULL;
    char *s = cwick_xmalloc(b->len + 1);
    memcpy(s, b->data, b->len);
    s[b->len] = '\0';
    return s;
}
static void test_apps_goal_contract(void) {
    char *t = apps_blob_text(cwick_webui_lookup("/apps/tictactoe/app.js"));
    char *k = apps_blob_text(cwick_webui_lookup("/apps/kart/app.js"));
    char *tm = apps_blob_text(cwick_webui_lookup("/apps/tictactoe/app.json"));
    char *km = apps_blob_text(cwick_webui_lookup("/apps/kart/app.json"));
    CHECK(t && k && tm && km, "apps: both bundles and metadata are embedded");
    if (!t || !k || !tm || !km) { free(t); free(k); free(tm); free(km); return; }

    /* App 1: human first, the question is which cell wins most, every cell's
       readout is shown in the sidebar, and the game data lives in sqlite. */
    CHECK(strstr(t, "X is the human and played first") != NULL,
          "apps/ttt: the human plays first");
    CHECK(strstr(t, "greatest probability of winning") != NULL &&
          strstr(t, "cellFacts") != NULL,
          "apps/ttt: the decision asks for the highest win probability");
    CHECK(strstr(t, "otherwise prefer the centre") == NULL,
          "apps/ttt: no hard-coded move priority replaces the model's answer");
    CHECK(strstr(t, "prob-row") != NULL && strstr(t, "factsZh") != NULL,
          "apps/ttt: the sidebar shows both sides and every cell's readout");
    CHECK(strstr(t, "/v1/apps/") != NULL && strstr(tm, "sqlite") != NULL,
          "apps/ttt: game data is stored in server sqlite");

    /* App 2: a Mobius loop, real-time randomised decisions aimed at maximum
       difficulty, with obstacle and kart statistics in the sidebar. */
    CHECK(strstr(k, "Mobius loop") != NULL &&
          strstr(k, "var h = u * 0.5") != NULL,
          "apps/kart: the track is a real Mobius strip (half-angle twist)");
    CHECK(strstr(k, "Maximise difficulty") != NULL &&
          strstr(k, "leave room for the player to react") == NULL,
          "apps/kart: the decision maximises lap difficulty");
    CHECK(strstr(k, "shuffled(DECISIONS)") != NULL &&
          strstr(k, "variation seed") != NULL &&
          strstr(k, "FOCUS[") != NULL,
          "apps/kart: each real-time decision is randomised");
    CHECK(strstr(k, "entropyOf") != NULL && strstr(k, "随机种子") != NULL,
          "apps/kart: the variation is visible in the sidebar");
    CHECK(strstr(k, "/v1/apps/kart/events") != NULL && strstr(km, "sqlite") != NULL,
          "apps/kart: spawns, hits and laps are stored in server sqlite");

    free(t); free(k); free(tm); free(km);
}

static void test_webui(void) {
    CHECK(cwick_webui_asset_count() >= 3, "webui: html + css + js are embedded");

    const cwick_webui_blob *doc = cwick_webui_lookup("/");
    CHECK(doc != NULL, "webui: `/` resolves");
    CHECK(doc && strncmp((const char *)doc->data, "<!doctype html", 14) == 0,
          "webui: `/` is the HTML document");
    CHECK(doc && doc->len > 2048, "webui: document is more than a stub");
    CHECK(doc && strstr(doc->content_type, "text/html") != NULL,
          "webui: document content type is text/html");

    const cwick_webui_blob *js = cwick_webui_lookup("/web/app.js");
    CHECK(js && js->len > 4096, "webui: the chat client is embedded");
    CHECK(js && strstr((const char *)js->data, "/v1/chat/completions") != NULL,
          "webui: the client calls this server's chat endpoint");
    CHECK(js && strstr((const char *)js->data, "reasoning_content") != NULL,
          "webui: the client reads the reasoning stream");
    CHECK(js && strstr((const char *)js->data, "/v1/hub/search") != NULL &&
          strstr((const char *)js->data, "/v1/hub/download") != NULL &&
          strstr((const char *)js->data, "hubStartPolling") != NULL,
          "webui: the model tab polls the model-library endpoints");
    CHECK(strstr((const char *)doc->data, "id=\"modelsView\"") != NULL &&
          strstr((const char *)doc->data, "data-view=\"models\"") != NULL,
          "webui: the model library has its own tab");
    CHECK(strstr((const char *)doc->data, "id=\"appsView\"") != NULL &&
          strstr((const char *)doc->data, "id=\"appFrame\"") != NULL,
          "webui: the app-market view hosts apps in an iframe");
    CHECK(js && strstr((const char *)js->data, "/apps/manifest.json") != NULL &&
          strstr((const char *)js->data, "openApp") != NULL,
          "webui: the app market is driven by apps/manifest.json");
    CHECK(js && strstr((const char *)js->data, "trendSample") != NULL &&
          strstr((const char *)js->data, "drawChart") != NULL &&
          strstr((const char *)js->data, "renderModelLabels") != NULL,
          "webui: the settings tab draws metric trends and full model info");
    CHECK(strstr((const char *)doc->data, "id=\"chartTokS\"") != NULL &&
          strstr((const char *)doc->data, "id=\"chartHit\"") != NULL &&
          strstr((const char *)doc->data, "id=\"settingsView\"") != NULL &&
          strstr((const char *)doc->data, "data-view=\"console\"") == NULL &&
          strstr((const char *)doc->data, "id=\"consoleView\"") == NULL,
          "webui: the console view is merged into settings");
    CHECK(js && strstr((const char *)js->data, "state.prefs.apiBase") == NULL &&
          strstr((const char *)js->data, "setApiBase") == NULL &&
          strstr((const char *)js->data, "testApi") == NULL &&
          strstr((const char *)js->data,
                 "function apiBaseUrl() { return location.origin") != NULL,
          "webui: the console is same-origin, no external endpoint");
    CHECK(strstr((const char *)doc->data, "id=\"setApiBase\"") == NULL &&
          strstr((const char *)doc->data, "id=\"apiTest\"") == NULL &&
          strstr((const char *)doc->data, "id=\"apiEffective\"") == NULL,
          "webui: the third-party API panel is gone");
    CHECK(strstr((const char *)doc->data, "id=\"pmList\"") != NULL &&
          strstr((const char *)doc->data, "id=\"pmQuery\"") != NULL &&
          strstr((const char *)doc->data, "id=\"pmRate\"") != NULL,
          "webui: the settings tab browses raw Prometheus series");
    CHECK(js && strstr((const char *)js->data, "pmSample") != NULL &&
          strstr((const char *)js->data, "# HELP") != NULL &&
          strstr((const char *)js->data, "# TYPE") != NULL &&
          strstr((const char *)js->data, "cwick.prom.v1") != NULL,
          "webui: any exported metric can be plotted as a custom trend");

    const cwick_webui_blob *manifest = cwick_webui_lookup("/apps/manifest.json");
    CHECK(manifest && strstr(manifest->content_type, "application/json") != NULL &&
          strstr((const char *)manifest->data, "\"tictactoe\"") != NULL,
          "webui: apps/manifest.json is served as JSON");
    CHECK(manifest && strstr((const char *)manifest->data, "\"kart\"") != NULL,
          "webui: the manifest lists the kart app");
    const cwick_webui_blob *apps_alias = cwick_webui_lookup("/apps");
    const cwick_webui_blob *apps_alias2 = cwick_webui_lookup("/apps/");
    CHECK(apps_alias && apps_alias2 && manifest &&
          apps_alias->len == manifest->len && apps_alias2->len == manifest->len &&
          memcmp(apps_alias->data, manifest->data, manifest->len) == 0 &&
          memcmp(apps_alias2->data, manifest->data, manifest->len) == 0,
          "webui: /apps aliases reach the manifest");
    const cwick_webui_blob *appjs = cwick_webui_lookup("/apps/tictactoe/app.js");
    CHECK(appjs && strstr(appjs->content_type, "javascript") != NULL &&
          strstr((const char *)appjs->data, "/systemone") != NULL &&
          strstr((const char *)appjs->data, "askDecision") != NULL &&
          strstr((const char *)appjs->data, "fallbackMove") != NULL,
          "webui: the tictactoe app decides through /v1/systemone");
    const cwick_webui_blob *kartjs = cwick_webui_lookup("/apps/kart/app.js");
    CHECK(appjs && kartjs &&
          strstr((const char *)appjs->data, "cwick.prefs.v1") == NULL &&
          strstr((const char *)kartjs->data, "cwick.prefs.v1") == NULL &&
          strstr((const char *)appjs->data,
                 "function apiBase() { return location.origin") != NULL &&
          strstr((const char *)kartjs->data,
                 "function apiBase() { return location.origin") != NULL,
          "webui: hosted apps never read an external endpoint");
    const cwick_webui_blob *apphtml = cwick_webui_lookup("/apps/tictactoe/index.html");
    const cwick_webui_blob *appdir = cwick_webui_lookup("/apps/tictactoe/");
    const cwick_webui_blob *appdir2 = cwick_webui_lookup("/apps/tictactoe");
    CHECK(apphtml && strstr((const char *)apphtml->data, "id=\"board\"") != NULL &&
          appdir && appdir2 && appdir->len == apphtml->len &&
          appdir2->len == apphtml->len &&
          memcmp(appdir->data, apphtml->data, apphtml->len) == 0 &&
          memcmp(appdir2->data, apphtml->data, apphtml->len) == 0,
          "webui: app directory aliases reach index.html");

    const cwick_webui_blob *login = cwick_webui_lookup("/login");
    const cwick_webui_blob *login2 = cwick_webui_lookup("/login.html");
    CHECK(login && login2 && login->len == login2->len &&
          strstr((const char *)login->data, "v1/auth/login") != NULL,
          "webui: the single-user login page is embedded and aliased");

    const cwick_webui_blob *css = cwick_webui_lookup("/web/app.css");
    CHECK(css && strstr(css->content_type, "text/css") != NULL,
          "webui: stylesheet served as text/css");

    CHECK(cwick_webui_lookup("/index.html") == doc && cwick_webui_lookup("/web") == doc,
          "webui: index aliases all reach the document");
    CHECK(cwick_webui_lookup("/?stream=1") == doc,
          "webui: query string ignored");
    CHECK(cwick_webui_lookup("/v1/metrics") == NULL,
          "webui: API paths are not assets");
    CHECK(cwick_webui_lookup("/../etc/passwd") == NULL,
          "webui: traversal resolves to nothing");
    CHECK(cwick_webui_lookup("") == NULL && cwick_webui_lookup(NULL) == NULL,
          "webui: empty paths are refused");
}

/* ------------------------------------------------------------------ */
/* sampling: the thing that stands between a sampling-trained checkpoint
 * and a repetition loop                                                */
/* ------------------------------------------------------------------ */

/* Draw `n` tokens from a fixed 3-way distribution and count them. `pick`
 * rewrites the logits it is given (temperature scaling, penalties), so every
 * draw starts from a fresh copy of the template. */
static void draw_counts(const cwick_sampler_cfg *cfg, const float *tpl,
                        int64_t ntok, int64_t n, uint64_t seed,
                        int64_t *counts) {
    float buf[16];
    cwick_sampler s;
    cwick_sampler_init(&s, cfg, ntok, seed);
    for (int64_t i = 0; i < n; i++) {
        memcpy(buf, tpl, (size_t)ntok * sizeof(float));
        uint32_t t = cwick_sampler_pick(&s, buf, ntok);
        cwick_sampler_accept(&s, t);
        if (t < ntok) counts[t]++;
    }
    cwick_sampler_free(&s);
}

static void test_sampler(void) {
    cwick_sampler_cfg c;
    cwick_sampler_cfg_default(&c);
    /* The distribution asserts below feed the *same* logits thousands of
     * times, which is exactly the signature the loop breaker exists to
     * fight; they measure the sampling math, not the breaker, so it is off
     * for the whole function (test_loop_break covers it with fresh logits
     * and its own expectations). */
    setenv("CWICK_LOOP_BREAK", "0", 1);
    CHECK(c.temperature == 0.0f && c.top_p == 1.0f && c.top_k == 0 &&
              c.presence_penalty == 0.0f && c.frequency_penalty == 0.0f,
          "sampler: the default is greedy (no truncation, no penalty)");
    {
        char d[96];
        cwick_sampler_describe(&c, d, sizeof(d));
        CHECK(strcmp(d, "greedy") == 0, "sampler: greedy is described as such");
    }

    /* temperature 0 is argmax even when the rest of the config is set */
    c.temperature = 0.0f;
    c.top_p = 0.5f;
    c.top_k = 3;
    CHECK(cwick_sampler_cfg_is_greedy(&c),
          "sampler: temperature 0 is greedy whatever top_p/top_k say");
    {
        float x[4] = { 1.0f, 5.0f, 3.0f, 2.0f };
        cwick_sampler s;
        cwick_sampler_init(&s, &c, 4, 12345);
        CHECK(cwick_sampler_pick(&s, x, 4) == 1, "sampler: temperature 0 picks the argmax");
        cwick_sampler_free(&s);
    }

    /* a draw is reproducible from its seed and moves when the seed moves */
    {
        const float tpl[3] = { 0.0f, 0.0f, 0.0f };  /* flat: every token plausible */
        cwick_sampler_cfg f = c;
        f.temperature = 1.0f;
        f.top_p = 1.0f;
        f.top_k = 0;
        int64_t a[3] = { 0, 0, 0 }, b[3] = { 0, 0, 0 }, d[3] = { 0, 0, 0 };
        draw_counts(&f, tpl, 3, 200, 0xABCDEFULL, a);
        draw_counts(&f, tpl, 3, 200, 0xABCDEFULL, b);
        draw_counts(&f, tpl, 3, 200, 0x13579BULL, d);
        CHECK(a[0] == b[0] && a[1] == b[1] && a[2] == b[2],
              "sampler: same seed -> same tokens");
        CHECK(!(d[0] == a[0] && d[1] == a[1] && d[2] == a[2]),
              "sampler: a different seed -> a different run");
    }

    /* the draw follows the softmax it is given */
    {
        const float tpl[3] = { 2.0f, 1.0f, 0.0f };
        cwick_sampler_cfg f = c;
        f.temperature = 1.0f;
        f.top_p = 1.0f;
        f.top_k = 0;
        int64_t cnt[3] = { 0, 0, 0 };
        const int64_t N = 30000;
        draw_counts(&f, tpl, 3, N, 0x5EEDULL, cnt);
        /* softmax(2,1,0) = .6652 .2447 .0900 */
        double p0 = (double)cnt[0] / (double)N, p1 = (double)cnt[1] / (double)N,
               p2 = (double)cnt[2] / (double)N;
        CHECK(fabs(p0 - 0.6652) < 0.02 && fabs(p1 - 0.2447) < 0.02 &&
                  fabs(p2 - 0.0900) < 0.02,
              "sampler: 30k draws match softmax(2,1,0) within 0.02");
    }

    /* top-k and top-p truncate: the tail must never be drawn */
    {
        const float tpl[3] = { 2.0f, 1.0f, 0.0f };
        cwick_sampler_cfg f = c;
        f.temperature = 1.0f;
        f.top_p = 1.0f;
        f.top_k = 2;
        int64_t cnt[3] = { 0, 0, 0 };
        draw_counts(&f, tpl, 3, 5000, 7, cnt);
        CHECK(cnt[2] == 0 && cnt[0] > 0 && cnt[1] > 0,
              "sampler: top_k=2 keeps the top two and drops the third");

        cwick_sampler_cfg g = c;
        g.temperature = 1.0f;
        g.top_p = 1e-6f;   /* nucleus = the argmax and nothing else */
        g.top_k = 0;
        int64_t cnt2[3] = { 0, 0, 0 };
        draw_counts(&g, tpl, 3, 2000, 9, cnt2);
        CHECK(cnt2[0] == 2000 && cnt2[1] == 0 && cnt2[2] == 0,
              "sampler: top_p -> 0 collapses onto the argmax");
    }

    /* The nucleus is normalised over the top-k survivors, not over the whole
     * vocabulary (the reference order: top_k filters, top_p sees what is
     * left). A long flat tail outside the window therefore cannot widen the
     * nucleus — only the window is drawable, however small its share of the
     * vocabulary mass is. */
    {
        const int64_t V = 200;
        float *tpl = cwick_xmalloc((size_t)V * sizeof(float));
        float *buf = cwick_xmalloc((size_t)V * sizeof(float));
        for (int64_t i = 0; i < V; i++) tpl[i] = 0.0f;
        tpl[0] = 10.0f;
        tpl[1] = 9.9f;   /* window mass / vocabulary mass ~ 0.019 << top_p */
        cwick_sampler_cfg f = c;
        f.temperature = 1.0f;
        f.top_p = 0.95f;
        f.top_k = 2;
        cwick_sampler s;
        cwick_sampler_init(&s, &f, V, 31);
        int64_t cnt[2] = { 0, 0 };
        bool tail_drawn = false;
        for (int i = 0; i < 5000; i++) {
            memcpy(buf, tpl, (size_t)V * sizeof(float));
            uint32_t t = cwick_sampler_pick(&s, buf, V);
            cwick_sampler_accept(&s, t);
            if (t >= 2) tail_drawn = true;
            else cnt[t]++;
        }
        cwick_sampler_free(&s);
        CHECK(!tail_drawn,
              "sampler: top_k is a hard filter — the tail is not drawable");
        /* renormalised over {10, 9.9}: e^0.1/(1+e^0.1) = .5250 / .4750 */
        CHECK(cnt[0] > 0 && cnt[1] > 0 &&
                  fabs((double)cnt[0] / 5000.0 - 0.5250) < 0.02,
              "sampler: the nucleus is normalised over the survivors");
        free(tpl);
        free(buf);
    }

    /* Same order, numeric case: logits (2,1,0), top_k=3 (no filter), top_p
     * 0.75. Cumulative over the window: .6652, then .9099 >= .75, so the
     * nucleus is the top two and they are renormalised to .7311 / .2689. */
    {
        const float tpl[3] = { 2.0f, 1.0f, 0.0f };
        cwick_sampler_cfg f = c;
        f.temperature = 1.0f;
        f.top_p = 0.75f;
        f.top_k = 3;
        int64_t cnt[3] = { 0, 0, 0 };
        draw_counts(&f, tpl, 3, 30000, 0x5EEDULL, cnt);
        CHECK(cnt[2] == 0 && fabs((double)cnt[0] / 30000.0 - 0.7311) < 0.02 &&
                  fabs((double)cnt[1] / 30000.0 - 0.2689) < 0.02,
              "sampler: top_p keeps the smallest prefix and renormalises it");
    }

    /* temperature changes the shape, not just the rng: at T=0.1 the argmax
     * is drawn almost every time */
    {
        const float tpl[3] = { 2.0f, 1.0f, 0.0f };
        cwick_sampler_cfg f = c;
        f.temperature = 0.1f;
        f.top_p = 1.0f;
        f.top_k = 0;
        int64_t cnt[3] = { 0, 0, 0 };
        draw_counts(&f, tpl, 3, 2000, 11, cnt);
        CHECK(cnt[0] > 1900, "sampler: low temperature is nearly deterministic");
    }

    /* repetition penalties: the whole point of L2 */
    {
        /* token 0 is 0.1 ahead; after 20 occurrences a frequency penalty of
         * 1.0 must hand the turn to token 1 even under argmax */
        float x[2] = { 10.0f, 9.9f };
        cwick_sampler_cfg f;
        cwick_sampler_cfg_default(&f);
        f.frequency_penalty = 1.0f;
        cwick_sampler s;
        cwick_sampler_init(&s, &f, 2, 1);
        uint32_t first = cwick_sampler_pick(&s, x, 2);
        CHECK(first == 0, "sampler: without history the best logit wins");
        for (int i = 0; i < 20; i++) cwick_sampler_accept(&s, 0);
        memcpy(x, ((float[2]){ 10.0f, 9.9f }), sizeof(float) * 2);
        CHECK(cwick_sampler_pick(&s, x, 2) == 1,
              "sampler: frequency_penalty dethrones a repeated token");
        cwick_sampler_free(&s);

        /* presence is charged once, whatever the count */
        cwick_sampler_cfg p;
        cwick_sampler_cfg_default(&p);
        p.presence_penalty = 0.5f;
        cwick_sampler s2;
        cwick_sampler_init(&s2, &p, 2, 1);
        float y[2] = { 10.0f, 9.9f };
        CHECK(cwick_sampler_pick(&s2, y, 2) == 0,
              "sampler: presence_penalty of an unseen token changes nothing");
        cwick_sampler_accept(&s2, 0);
        memcpy(y, ((float[2]){ 10.0f, 9.9f }), sizeof(float) * 2);
        CHECK(cwick_sampler_pick(&s2, y, 2) == 1,
              "sampler: presence_penalty is charged once per distinct token");
        for (int i = 0; i < 50; i++) cwick_sampler_accept(&s2, 0);
        memcpy(y, ((float[2]){ 10.0f, 9.9f }), sizeof(float) * 2);
        CHECK(cwick_sampler_pick(&s2, y, 2) == 1,
              "sampler: presence_penalty does not grow with the count");
        cwick_sampler_free(&s2);
    }

    /* NaN in the logits must not become a token */
    {
        float x[3] = { 1.0f, NAN, 2.0f };
        cwick_sampler_cfg f;
        cwick_sampler_cfg_default(&f);
        f.temperature = 1.0f;
        cwick_sampler s;
        cwick_sampler_init(&s, &f, 3, 3);
        bool hit_nan = false;
        for (int i = 0; i < 50; i++) {
            memcpy(x, ((float[3]){ 1.0f, NAN, 2.0f }), sizeof(float) * 3);
            if (cwick_sampler_pick(&s, x, 3) == 1) hit_nan = true;
        }
        CHECK(!hit_nan, "sampler: a NaN logit is never drawn");
        cwick_sampler_free(&s);
    }

    /* a nucleus wider than the scan window (CWICK_TOPP_SCAN) has to be found
     * anyway: a flat distribution over more tokens than the window is the only
     * case that reaches the fallback, and getting it wrong would truncate the
     * distribution silently. */
    {
        const int64_t V = 3 * CWICK_TOPP_SCAN;   /* 12288 flat tokens */
        float *flat = cwick_xmalloc((size_t)V * sizeof(float));
        float *buf = cwick_xmalloc((size_t)V * sizeof(float));
        for (int64_t i = 0; i < V; i++) flat[i] = 0.0f;
        cwick_sampler_cfg f;
        cwick_sampler_cfg_default(&f);
        f.temperature = 1.0f;
        f.top_p = 0.9f;      /* 90% of 12288 = 11059 tokens > the 4096 window */
        cwick_sampler s;
        cwick_sampler_init(&s, &f, V, 4242);
        int64_t beyond = 0;
        for (int i = 0; i < 400; i++) {
            memcpy(buf, flat, (size_t)V * sizeof(float));
            if (cwick_sampler_pick(&s, buf, V) >= (uint32_t)CWICK_TOPP_SCAN) beyond++;
        }
        cwick_sampler_free(&s);
        free(flat);
        free(buf);
        CHECK(beyond > 200,
              "sampler: a flat nucleus wider than the scan window is not truncated");
    }

    /* Same path, analytic case: 8192 tokens, one at logit 10 and the rest at 0,
     * top_p 0.95. The cut is that token plus ceil((0.95*Z - 1)/e^-10) of the
     * flat ones, so the peak is drawn with 1/(1+k*e^-10) and nothing past
     * index k is drawable at all. The histogram decides the boundary instead
     * of a sort, and has to land on the same distribution. */
    {
        const int64_t V = 8192;
        const double e = exp(-10.0);
        const double Z = 1.0 + (double)(V - 1) * e;
        const int64_t k = (int64_t)ceil((0.95 * Z - 1.0) / e);
        const double p_peak = 1.0 / (1.0 + (double)k * e);
        float *tpl = cwick_xmalloc((size_t)V * sizeof(float));
        float *buf = cwick_xmalloc((size_t)V * sizeof(float));
        for (int64_t i = 0; i < V; i++) tpl[i] = 0.0f;
        tpl[0] = 10.0f;
        cwick_sampler_cfg f = c;
        f.temperature = 1.0f;
        f.top_p = 0.95f;
        f.top_k = 0;
        cwick_sampler s;
        cwick_sampler_init(&s, &f, V, 909);
        const int64_t N = 20000;
        int64_t peak = 0, beyond = 0, bulk = 0;
        for (int64_t i = 0; i < N; i++) {
            memcpy(buf, tpl, (size_t)V * sizeof(float));
            uint32_t t = cwick_sampler_pick(&s, buf, V);
            if (t == 0) peak++;
            else {
                bulk++;
                if (t > (uint32_t)k) beyond++;
            }
        }
        cwick_sampler_free(&s);
        CHECK(beyond == 0,
              "sampler: the histogram nucleus excludes what is past the cut");
        CHECK(fabs((double)peak / (double)N - p_peak) < 0.02 &&
                  fabs((double)bulk / (double)N - (1.0 - p_peak)) < 0.02,
              "sampler: the histogram nucleus matches the exact cut");
        free(tpl);
        free(buf);
    }

    /* validation is fail-closed */
    {
        char why[64];
        cwick_sampler_cfg v;
        cwick_sampler_cfg_default(&v);
        CHECK(cwick_sampler_cfg_validate(&v, why, sizeof(why)), "sampler: default validates");
        v.temperature = 5.0f;
        CHECK(!cwick_sampler_cfg_validate(&v, why, sizeof(why)) &&
                  strstr(why, "temperature") != NULL,
              "sampler: temperature 5 is refused and says why");
        v = c; v.top_p = 0.0f;
        CHECK(!cwick_sampler_cfg_validate(&v, why, sizeof(why)), "sampler: top_p 0 refused");
        v = c; v.top_p = 1.5f;
        CHECK(!cwick_sampler_cfg_validate(&v, why, sizeof(why)), "sampler: top_p 1.5 refused");
        v = c; v.top_k = -1;
        CHECK(!cwick_sampler_cfg_validate(&v, why, sizeof(why)), "sampler: negative top_k refused");
        v = c; v.frequency_penalty = 3.0f;
        CHECK(!cwick_sampler_cfg_validate(&v, why, sizeof(why)),
              "sampler: penalty outside [-2,2] refused");
    }

    /* generation_config.json is the checkpoint's half of the contract */
    {
        char *root = make_tmpdir();
        char path[512];
        cwick_path_join(path, sizeof(path), root, "generation_config.json");
        const char *sampling =
            "{\"do_sample\": true, \"temperature\": 1.0, \"top_p\": 0.95, "
            "\"top_k\": 20}";
        cwick_write_file(path, sampling, strlen(sampling));
        cwick_sampler_cfg g;
        CHECK(cwick_sampler_cfg_from_checkpoint(root, &g),
              "sampler: generation_config.json is read");
        CHECK(g.temperature == 1.0f && fabs(g.top_p - 0.95f) < 1e-6 &&
                  g.top_k == 20,
              "sampler: do_sample=true -> temperature/top_p/top_k as written");
        CHECK(!cwick_sampler_cfg_is_greedy(&g),
              "sampler: a sampling checkpoint is not decoded greedily");

        const char *greedy = "{\"do_sample\": false, \"temperature\": 1.0}";
        cwick_write_file(path, greedy, strlen(greedy));
        CHECK(cwick_sampler_cfg_from_checkpoint(root, &g) && g.temperature == 0.0f,
              "sampler: do_sample=false wins over a stray temperature");

        cwick_sampler_cfg g2;
        cwick_sampler_cfg_default(&g2);
        char nowhere[512];
        cwick_path_join(nowhere, sizeof(nowhere), root, "no-such-dir");
        CHECK(!cwick_sampler_cfg_from_checkpoint(nowhere, &g2) &&
                  g2.temperature == 0.0f,
              "sampler: no generation_config.json leaves the caller's default");
    }
    unsetenv("CWICK_LOOP_BREAK");
}

/* The loop guard (L3): the same block repeating must end the turn, and nothing
 * that merely *looks* repetitive for a few tokens may. Thresholds are the
 * defaults (period <= 64, min 3 occurrences, min 16 repeating tokens), so a
 * period-1 loop fires on the 17th token and a period-3 loop on the 19th
 * (max(2*3,16) = 16 matches, which spans 19 tokens). */
static void test_loop_guard(void) {
    cwick_sampler_cfg c;
    cwick_sampler_cfg_default(&c);

    /* one token over and over: the measured failure mode ("喷喷喷喷…",
     * "Owner: Owner: Owner: …"). */
    {
        cwick_sampler s;
        cwick_sampler_init(&s, &c, 8, 1);
        int per = 0;
        int64_t run = 0;
        for (int i = 0; i < 16; i++) cwick_sampler_accept(&s, 5);
        CHECK(!cwick_sampler_looped(&s, &per, &run),
              "loop guard: 16 identical tokens are still within budget");
        cwick_sampler_accept(&s, 5);
        CHECK(cwick_sampler_looped(&s, &per, &run) && per == 1 && run == 16,
              "loop guard: the 17th identical token stops the run");
        /* latched: more of the same does not un-fire it */
        for (int i = 0; i < 40; i++) cwick_sampler_accept(&s, 5);
        CHECK(cwick_sampler_looped(&s, &per, &run) && per == 1,
              "loop guard: the stop is latched");
        cwick_sampler_free(&s);
    }

    /* a block of three repeating: fires later than the run floor alone, at
     * max(2*3, 16) = 16 matches. */
    {
        const uint32_t blk[3] = { 7, 2, 9 };
        cwick_sampler s;
        cwick_sampler_init(&s, &c, 16, 1);
        int per = 0;
        int64_t run = 0;
        bool fired = false;
        int64_t fired_at = 0;
        for (int i = 0; i < 40 && !fired; i++) {
            cwick_sampler_accept(&s, blk[i % 3]);
            if (cwick_sampler_looped(&s, &per, &run)) { fired = true; fired_at = i + 1; }
        }
        CHECK(fired && per == 3,
              "loop guard: a repeating 3-token block is caught at period 3");
        CHECK(fired_at == 19,
              "loop guard: it fires after 16 matches, not earlier");
        cwick_sampler_free(&s);
    }

    /* structured text that repeats a little and moves on must survive: a
     * markdown rule, an ellipsis, an enumerated list. */
    {
        cwick_sampler s;
        cwick_sampler_init(&s, &c, 4096, 1);
        bool fired = false;
        /* 8 dashes, 6 dots, then distinct list items */
        for (int i = 0; i < 8; i++) cwick_sampler_accept(&s, 100);   /* "--------" */
        for (int i = 0; i < 6; i++) cwick_sampler_accept(&s, 101);   /* "……" */
        for (int i = 0; i < 12; i++) cwick_sampler_accept(&s, 200 + (uint32_t)i);
        /* and a sentence that says "ok" three times between other words */
        const uint32_t sent[] = { 1, 2, 3, 300, 4, 5, 300, 6, 7, 300, 8, 9 };
        for (size_t i = 0; i < sizeof(sent) / sizeof(sent[0]); i++)
            cwick_sampler_accept(&s, sent[i]);
        fired = cwick_sampler_looped(&s, NULL, NULL);
        CHECK(!fired,
              "loop guard: ordinary structure does not trip the guard");
        cwick_sampler_free(&s);
    }

    /* a long non-repeating sequence (deterministic, so the test is stable) */
    {
        cwick_sampler s;
        cwick_sampler_init(&s, &c, 1 << 20, 1);
        uint64_t x = 0x123456789ABCDEFULL;
        for (int i = 0; i < 5000; i++) {
            x = x * 6364136223846793005ULL + 1442695040888963407ULL;
            cwick_sampler_accept(&s, (uint32_t)(x >> 33));
        }
        CHECK(!cwick_sampler_looped(&s, NULL, NULL),
              "loop guard: 5000 pseudo-random tokens never fire");
        cwick_sampler_free(&s);
    }

    /* prompt tokens are primed, not accepted: a prompt that ends in a repeated
     * block must not look like a loop on the first generated token. */
    {
        cwick_sampler s;
        cwick_sampler_init(&s, &c, 4096, 1);
        for (int i = 0; i < 64; i++) cwick_sampler_prime(&s, 77);
        for (int i = 0; i < 15; i++) cwick_sampler_accept(&s, 5);
        CHECK(!cwick_sampler_looped(&s, NULL, NULL),
              "loop guard: primed prompt tokens do not count as a loop");
        cwick_sampler_free(&s);
    }

    /* CWICK_LOOP_DETECT=0 turns the whole thing off. */
    {
        setenv("CWICK_LOOP_DETECT", "0", 1);
        cwick_sampler s;
        cwick_sampler_init(&s, &c, 8, 1);
        for (int i = 0; i < 200; i++) cwick_sampler_accept(&s, 5);
        CHECK(!cwick_sampler_looped(&s, NULL, NULL),
              "loop guard: CWICK_LOOP_DETECT=0 disables it");
        cwick_sampler_free(&s);
        unsetenv("CWICK_LOOP_DETECT");

        /* and a shorter period is honoured (the knob can only lower it) */
        setenv("CWICK_LOOP_MAX_PERIOD", "4", 1);
        cwick_sampler s2;
        cwick_sampler_init(&s2, &c, 4096, 1);
        const uint32_t blk[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
        bool fired = false;
        for (int i = 0; i < 80 && !fired; i++) {
            cwick_sampler_accept(&s2, blk[i % 8]);
            fired = cwick_sampler_looped(&s2, NULL, NULL);
        }
        CHECK(!fired,
              "loop guard: a period above CWICK_LOOP_MAX_PERIOD is not caught");
        cwick_sampler_free(&s2);
        unsetenv("CWICK_LOOP_MAX_PERIOD");
    }
}

/* The loop breaker: while the produced tail repeats exactly at period p for
 * lbrk_need[p] tokens, pick() bans the token that would continue the pattern,
 * so the run escapes before the guard has to kill it. Defaults: break_min 8,
 * so a period-1 loop arms after 8 identical tokens and a period-2 loop after
 * 8 trailing periodic ones. */
static void test_loop_break(void) {
    cwick_sampler_cfg c;
    cwick_sampler_cfg_default(&c);   /* greedy: pick is the argmax */

    /* a period-1 attractor: logits that keep wanting token 3, with 4 as the
     * runner-up. The breaker must let 8 through, ban the 9th, and the run
     * must never reach the guard. */
    {
        cwick_sampler s;
        cwick_sampler_init(&s, &c, 16, 1);
        float x[16];
        bool guard_fired = false;
        int run = 0, max_run = 0;
        for (int i = 0; i < 40; i++) {
            for (int t = 0; t < 16; t++) x[t] = 0.0f;
            x[3] = 10.0f;
            x[4] = 9.0f;
            uint32_t t = cwick_sampler_pick(&s, x, 16);
            if (t == 3) {
                run++;
                if (run > max_run) max_run = run;
            } else {
                run = 0;
            }
            cwick_sampler_accept(&s, t);
            if (cwick_sampler_looped(&s, NULL, NULL)) { guard_fired = true; break; }
        }
        CHECK(!guard_fired,
              "loop break: the attractor never reaches the guard");
        CHECK(max_run <= 9,
              "loop break: no more than break_min(+1) identical tokens in a row");
        CHECK(s.lbrk_steps >= 3,
              "loop break: the ban re-arms as the model drifts back in");
        cwick_sampler_free(&s);
    }

    /* a period-2 attractor driven by hand: [7,2] five times arms the breaker
     * (lmatch[2] = 9 >= 8), and the pick must refuse the continuation. */
    {
        cwick_sampler s;
        cwick_sampler_init(&s, &c, 16, 1);
        const uint32_t blk[2] = { 7, 2 };
        for (int i = 0; i < 10; i++) cwick_sampler_accept(&s, blk[i % 2]);
        CHECK(!cwick_sampler_looped(&s, NULL, NULL),
              "loop break: ten periodic tokens are below the guard");
        float x[16];
        for (int t = 0; t < 16; t++) x[t] = 0.0f;
        x[7] = 10.0f;   /* exactly the token that continues the loop */
        x[2] = 9.0f;
        uint32_t t = cwick_sampler_pick(&s, x, 16);
        CHECK(t == 2,
              "loop break: the continuation token is banned from the argmax");
        CHECK(s.lbrk_steps == 1, "loop break: one armed pick counted");
        cwick_sampler_free(&s);
    }

    /* CWICK_LOOP_BREAK=0 restores the bare guard behaviour. */
    {
        setenv("CWICK_LOOP_BREAK", "0", 1);
        cwick_sampler s;
        cwick_sampler_init(&s, &c, 16, 1);
        float x[16];
        for (int t = 0; t < 16; t++) x[t] = 0.0f;
        x[3] = 10.0f;
        x[4] = 9.0f;
        for (int i = 0; i < 12; i++) {
            uint32_t t = cwick_sampler_pick(&s, x, 16);
            cwick_sampler_accept(&s, t);
        }
        CHECK(s.lbrk_steps == 0,
              "loop break: CWICK_LOOP_BREAK=0 disables the ban");
        CHECK(!cwick_sampler_looped(&s, NULL, NULL),
              "loop break: 12 identical tokens are still below the guard");
        cwick_sampler_free(&s);
        unsetenv("CWICK_LOOP_BREAK");
    }

    /* ordinary structure with a handful of repeats must not arm: 8 dashes
     * arm exactly at break_min, so keep the structure under it and check
     * lbrk_steps stays 0 across ellipsis + list + scattered repeats. */
    {
        cwick_sampler s;
        cwick_sampler_init(&s, &c, 4096, 1);
        float x[4096];
        for (int i = 0; i < 6; i++) {          /* "……" six dots */
            for (int t = 0; t < 4096; t++) x[t] = 0.0f;
            x[101] = 5.0f;
            cwick_sampler_accept(&s, cwick_sampler_pick(&s, x, 4096));
        }
        for (int i = 0; i < 12; i++) {         /* distinct list items */
            for (int t = 0; t < 4096; t++) x[t] = 0.0f;
            x[200 + i] = 5.0f;
            cwick_sampler_accept(&s, cwick_sampler_pick(&s, x, 4096));
        }
        CHECK(s.lbrk_steps == 0,
              "loop break: short legitimate repeats never arm it");
        CHECK(!cwick_sampler_looped(&s, NULL, NULL),
              "loop break: and the guard agrees");
        cwick_sampler_free(&s);
    }
}

/* ------------------------------------------------------------------ */
/* 4f. thinking budget: the ceiling has to be enforced by the engine    */
/* ------------------------------------------------------------------ */
/* A thought that never closes costs the whole turn: measured on agent
 * traffic, 8096 of 8096 generated tokens went into the think block and the
 * client got `content=0, finish=length`. The fix is that the engine writes
 * the close tag itself, so two things have to be true at once -- the decision
 * (pure, below) and the act (the token really lands in the output and in the
 * context, which only a decode can show). */
static void test_think_budget(void) {
    /* ---- the policy ---- */
    CHECK(cwick_think_plan(0, -1, true, false) == CWICK_THINK_NONE,
          "budget: a negative ceiling is off");
    CHECK(cwick_think_plan(999, 10, true, true) == CWICK_THINK_NONE,
          "budget: a thought the model closed itself is left alone");
    CHECK(cwick_think_plan(4, 5, true, false) == CWICK_THINK_NONE,
          "budget: below the ceiling nothing happens");
    CHECK(cwick_think_plan(5, 5, true, false) == CWICK_THINK_CLOSE,
          "budget: at the ceiling an open thought is closed");
    CHECK(cwick_think_plan(0, 0, true, false) == CWICK_THINK_CLOSE,
          "budget: 0 with a block already open closes before the first token");
    CHECK(cwick_think_plan(0, 0, false, false) == CWICK_THINK_OPEN_CLOSE,
          "budget: 0 with nothing open writes the template's empty block");
    CHECK(cwick_think_plan(3, 0, false, false) == CWICK_THINK_NONE,
          "budget: never opens a block in the middle of an answer");
    CHECK(cwick_think_plan(9, 5, false, false) == CWICK_THINK_NONE,
          "budget: an answer that never thought is not touched");

    /* ---- the act ---- */
    char *root = make_tmpdir();
    cwick_error err;
    cwick_error_clear(&err);
    CHECK(cwick_setup_synthetic(root, &err) == CWICK_OK, "budget: synthetic setup");
    cwick_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_root = root;
    opt.max_new_tokens = 8;
    cwick_engine *eng = NULL;
    CHECK(cwick_engine_new(&opt, &eng, &err) == CWICK_OK, "budget: engine new");
    if (eng) {
        cwick_tokenizer *tok = NULL;
        char tdir[1152];
        cwick_path_join(tdir, sizeof(tdir), root, "tokenizer");
        CHECK(cwick_tokenizer_load(tdir, cwick_engine_spec(eng), &tok, &err) == CWICK_OK,
              "budget: tokenizer");
        uint32_t *ids = NULL;
        int64_t nids = 0;
        CHECK(cwick_chat_encode(tok, "You are helpful.", "Hi", &ids, &nids, &err) == CWICK_OK,
              "budget: prompt");
        if (ids) {
            cwick_gen_result r;
            /* 200/201 stand in for the markers: the synthetic vocabulary is
             * 256 wide and 0/1/2 are bos/eos/pad. */
            cwick_engine_set_think_budget(eng, 0, 200, 201, false);
            memset(&r, 0, sizeof(r));
            cwick_engine_reset(eng, &err);
            CHECK(cwick_engine_generate(eng, ids, nids, 6, false, &r, &err) == CWICK_OK,
                  "budget: generate under a 0 ceiling");
            CHECK(r.n_tokens >= 2 && r.tokens[0] == 200 && r.tokens[1] == 201,
                  "budget: the empty think block is the first thing written");
            CHECK(r.think_budget_hit,
                  "budget: the result says the ceiling was enforced");
            cwick_gen_result_free(&r);

            /* A turn that never opens a block is not touched, whatever the
             * ceiling says: there is no thought to cut. */
            cwick_engine_set_think_budget(eng, 4, 200, 201, false);
            memset(&r, 0, sizeof(r));
            cwick_engine_reset(eng, &err);
            CHECK(cwick_engine_generate(eng, ids, nids, 6, false, &r, &err) == CWICK_OK,
                  "budget: generate under a ceiling the thought never opened");
            CHECK(!r.think_budget_hit && r.tokens[0] != 200,
                  "budget: an answer that never thought is left alone");
            cwick_gen_result_free(&r);

            /* A ceiling at or above the whole output budget has to be clamped:
             * one that lands on the length limit leaves nothing to say, which
             * is the live failure this exists for (高=8192 against a
             * max_tokens of 8096). Here 40 against 6 becomes 3, so an open
             * block is closed three tokens in and the answer survives. */
            cwick_engine_set_think_budget(eng, 40, 200, 201, true);
            memset(&r, 0, sizeof(r));
            cwick_engine_reset(eng, &err);
            CHECK(cwick_engine_generate(eng, ids, nids, 6, false, &r, &err) == CWICK_OK,
                  "budget: generate above the whole output budget");
            CHECK(r.think_budget_hit && r.n_tokens >= 4 && r.tokens[3] == 201,
                  "budget: a ceiling above the budget is clamped and enforced");
            CHECK(r.think_budget_applied == 3,
                  "budget: the result reports the clamped ceiling, not the asked one");
            cwick_gen_result_free(&r);

            /* An id outside the vocabulary has to disarm the mechanism: a
             * capped id would be a different token, i.e. a corrupted block. */
            cwick_engine_set_think_budget(eng, 0, 9999, 201, false);
            memset(&r, 0, sizeof(r));
            cwick_engine_reset(eng, &err);
            CHECK(cwick_engine_generate(eng, ids, nids, 4, false, &r, &err) == CWICK_OK,
                  "budget: generate with an out-of-vocab marker");
            CHECK(!r.think_budget_hit,
                  "budget: an out-of-vocab marker disables the mechanism");
            cwick_gen_result_free(&r);

            cwick_engine_set_think_budget(eng, -1, -1, -1, false);
            free(ids);
        }
        cwick_tokenizer_free(tok);
        cwick_engine_free(eng);
    }
}

int main(void) {
    printf("cwick verification suite\n");
    test_bf16();
    test_mxfp4();
    test_mxfp4_spec();
    test_json();
    test_metrics();
    test_auth();
    test_auth_keys();
    test_store();
    test_contract_and_engine();
    test_memory_plan();
    test_tools();
    test_chat_tools_prompt();
    test_server();
    test_store_per_app();
    test_batched_prefill_equiv("legacy", cwick_setup_synthetic);
    test_batched_prefill_equiv("mhc", cwick_setup_synthetic_mhc);
    test_prefill_resume_equiv("legacy", cwick_setup_synthetic);
    test_prefill_resume_equiv("mhc", cwick_setup_synthetic_mhc);
    test_sampler();
    test_loop_guard();
    test_think_budget();
    test_loop_break();
    test_prefix_cache_equiv();
    test_prefix_block_checkpoints();
    test_gemm_matches_gemv();
    test_chunk_attn_equiv();
    test_max_new_clamped();
    test_gen_window();
    test_json_escape();
    test_stream_split();
    test_phase_account();
    test_thinking_effort();
    test_models();
    test_logcap();
    test_server_control();
    test_webui();
    test_apps_use_systemone();
    test_apps_goal_contract();
    test_hub();
    test_jev_errors();
    test_jev_adapter();
    if (g_fail) {
        printf("\nRESULT: FAIL\n");
        return 1;
    }
    printf("\nRESULT: PASS\n");
    return 0;
}
