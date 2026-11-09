/* tokenizer.c — native tokenizer.
 *
 * The native decoder is the sole source of token IDs. CWICK implements a
 * standard BPE over vocab.json + merges.txt (legacy) or a HuggingFace
 * tokenizer.json (the real checkpoint layout: a `model.vocab` dictionary and a
 * `model.merges` pair list, plus the GPT-2 byte-level byte<->char tables). A
 * byte-fallback covers raw-byte vocabularies (e.g. the synthetic model).
 * Small prompts stay sequential; the framework does not yet parallelize
 * segment encoding.
 */
#include "cwick.h"
#include "json.h"
#include "log.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "util.h"

/* Wall clock in milliseconds (same helper the other modules keep locally). */
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static cwick_status tokenizer_load_impl(const char *dir,
                                       const cwick_model_spec *spec,
                                       cwick_tokenizer *tok, cwick_error *err);

void cwick_tokenizer_free(cwick_tokenizer *tok) {
    if (!tok) return;
    for (size_t i = 0; i < tok->vocab_size; i++) free(tok->vocab[i].key);
    free(tok->vocab);
    for (size_t i = 0; i < tok->merge_count; i++) free(tok->merges[i].pair);
    free(tok->merges);
    for (size_t i = 0; i < tok->special_count; i++) free(tok->special[i].key);
    free(tok->special);
    free(tok->byte_to_id);
    free(tok->index);
    free(tok->id_to_slot);
    free(tok->pair_keys);
    free(tok->pair_rank);
    free(tok->pair_ab);
    free(tok);
}

/* Longest-first, so "<|begin_of_image|>" wins over "<|begin..."-style prefixes
 * and, more importantly, longer specials never get shadowed by shorter ones. */
static void specials_sort(cwick_tokenizer *tok) {
    for (size_t i = 1; i < tok->special_count; i++) {
        cwick_vocab_entry v = tok->special[i];
        size_t j = i;
        while (j > 0 && strlen(tok->special[j - 1].key) < strlen(v.key)) {
            tok->special[j] = tok->special[j - 1];
            j--;
        }
        tok->special[j] = v;
    }
}

/* vocab id -> vocab slot, so decode does not scan 154k entries per token. */
static void id_index_build(cwick_tokenizer *tok) {
    int64_t maxid = -1;
    for (size_t i = 0; i < tok->vocab_size; i++)
        if (tok->vocab[i].id > maxid) maxid = tok->vocab[i].id;
    if (maxid < 0) return;
    tok->id_to_slot_cap = maxid + 1;
    tok->id_to_slot = cwick_xmalloc((size_t)(maxid + 1) * sizeof(int64_t));
    for (int64_t i = 0; i <= maxid; i++) tok->id_to_slot[i] = -1;
    for (size_t i = 0; i < tok->vocab_size; i++)
        if (tok->vocab[i].id >= 0) tok->id_to_slot[tok->vocab[i].id] = (int64_t)i;
}

/* Built at the end of tokenizer_load (see the vocabulary index section). */
static void vocab_index_build(cwick_tokenizer *tok);
static void id_index_build(cwick_tokenizer *tok);
static void pair_index_build(cwick_tokenizer *tok);
static void specials_sort(cwick_tokenizer *tok);

/* Build a byte -> vocab id map from single-byte tokens. */
static void build_byte_map(cwick_tokenizer *tok) {
    tok->byte_to_id = cwick_xmalloc(256 * sizeof(int64_t));
    for (int i = 0; i < 256; i++) tok->byte_to_id[i] = -1;
    for (size_t i = 0; i < tok->vocab_size; i++) {
        const char *key = tok->vocab[i].key;
        size_t len = strlen(key);
        if (len == 1) {
            unsigned char b = (unsigned char)key[0];
            if (tok->byte_to_id[b] < 0) tok->byte_to_id[b] = tok->vocab[i].id;
        }
    }
    /* A model with ids 0..255 over raw bytes is byte-fallback capable. */
    int present = 1;
    for (int i = 0; i < 256; i++) if (tok->byte_to_id[i] < 0) { present = 0; break; }
    tok->byte_fallback = present && tok->vocab_size >= 256;
}

/* ------------------------------------------------------------------ */
/* GPT-2 byte-level BPE tables (used by the real GLM tokenizer.json)   */
/* ------------------------------------------------------------------ */

/* Decode one UTF-8 code point; returns -1 on invalid, sets *cplen to the
 * number of bytes consumed. */
static int64_t utf8_decode(const char *s, size_t len, size_t *cplen) {
    const unsigned char *p = (const unsigned char *)s;
    if (len == 0) { *cplen = 0; return -1; }
    if (p[0] < 0x80) { *cplen = 1; return p[0]; }
    if ((p[0] & 0xE0) == 0xC0 && len >= 2) {
        *cplen = 2;
        return ((int64_t)(p[0] & 0x1F) << 6) | (p[1] & 0x3F);
    }
    if ((p[0] & 0xF0) == 0xE0 && len >= 3) {
        *cplen = 3;
        return ((int64_t)(p[0] & 0x0F) << 12) |
               ((int64_t)(p[1] & 0x3F) << 6) | (p[2] & 0x3F);
    }
    *cplen = 1;
    return -1;
}

/* GPT-2 bytes_to_unicode: printable ranges map to themselves, every other
 * byte maps to U+0100+n in ascending byte order. cp_to_byte is indexed by
 * code point (0..402). */
static void gpt2_byte_tables(int64_t byte_to_char[256], int64_t cp_to_byte[512]) {
    int in_bs[256];
    memset(in_bs, 0, sizeof(in_bs));
    for (int b = 33; b <= 126; b++) in_bs[b] = 1;   /* '!'..'~' */
    for (int b = 161; b <= 172; b++) in_bs[b] = 1;  /* '¡'..'¬' */
    for (int b = 174; b <= 255; b++) in_bs[b] = 1;  /* '®'..'ÿ' */
    for (int i = 0; i < 512; i++) cp_to_byte[i] = -1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (in_bs[b]) {
            byte_to_char[b] = b;
        } else {
            byte_to_char[b] = 256 + n;
            n++;
        }
    }
    for (int b = 0; b < 256; b++) {
        int cp = (int)byte_to_char[b];
        if (cp < 512) cp_to_byte[cp] = b;
    }
}

/* Build the byte map for a GPT-2 byte-level vocabulary. */
static void build_byte_map_gpt2(cwick_tokenizer *tok) {
    int64_t byte_to_char[256], cp_to_byte[512];
    gpt2_byte_tables(byte_to_char, cp_to_byte);
    tok->byte_to_id = cwick_xmalloc(256 * sizeof(int64_t));
    for (int i = 0; i < 256; i++) tok->byte_to_id[i] = -1;
    for (size_t i = 0; i < tok->vocab_size; i++) {
        const char *key = tok->vocab[i].key;
        size_t len = strlen(key), cplen = 0;
        int64_t cp = utf8_decode(key, len, &cplen);
        if (cplen == len && cp >= 0 && cp < 512 && cp_to_byte[cp] >= 0) {
            int b = (int)cp_to_byte[cp];
            if (tok->byte_to_id[b] < 0) tok->byte_to_id[b] = (int64_t)tok->vocab[i].id;
        }
    }
    int present = 1;
    for (int i = 0; i < 256; i++) if (tok->byte_to_id[i] < 0) { present = 0; break; }
    tok->byte_fallback = present;
    tok->byte_level = true;
}

/* Shared cp->byte table for decoding byte-level tokens. */
static const int64_t *gpt2_cp_to_byte(void) {
    static int64_t cp_to_byte[512];
    static int ready = 0;
    if (!ready) {
        int64_t byte_to_char[256];
        gpt2_byte_tables(byte_to_char, cp_to_byte);
        ready = 1;
    }
    return cp_to_byte;
}

/* Shared byte->char table for encoding byte-level text. */
static const int64_t *gpt2_byte_to_char(void) {
    static int64_t byte_to_char[256];
    static int ready = 0;
    if (!ready) {
        int64_t cp_to_byte[512];
        gpt2_byte_tables(byte_to_char, cp_to_byte);
        ready = 1;
    }
    return byte_to_char;
}

/* Does this tokenizer.json ship the DeepSeek pre-tokenizer? The tell is the
 * CJK range in one of the Split patterns: the GLM regex has no such class.
 * Walking the (possibly nested) Sequence keeps this working whichever level
 * the Split sits at. */
static bool pattern_is_cjk(const char *re) {
    return re && (strstr(re, "\\u4e00") || strstr(re, "\\u3040") ||
                  strstr(re, "\xe4\xb8\x80"));
}

static bool pretok_is_deepseek(json_value *root) {
    json_value *pt = cwick_json_get(root, "pre_tokenizer");
    if (!pt) return false;
    if (pt->kind == JSON_OBJECT) {
        json_value *ty = cwick_json_get(pt, "type");
        if (ty && ty->kind == JSON_STRING && strcmp(ty->string, "Split") == 0) {
            json_value *pat = cwick_json_get(pt, "pattern");
            json_value *re = pat ? cwick_json_get(pat, "Regex") : NULL;
            return pattern_is_cjk(re ? re->string : NULL);
        }
        json_value *seq = cwick_json_get(pt, "pretokenizers");
        if (seq && seq->kind == JSON_ARRAY) {
            for (size_t i = 0; i < cwick_json_array_len(seq); i++) {
                json_value *sub = cwick_json_array_at(seq, i);
                if (!sub || sub->kind != JSON_OBJECT) continue;
                json_value *ty = cwick_json_get(sub, "type");
                if (!ty || ty->kind != JSON_STRING ||
                    strcmp(ty->string, "Split") != 0)
                    continue;
                json_value *pat = cwick_json_get(sub, "pattern");
                json_value *re = pat ? cwick_json_get(pat, "Regex") : NULL;
                if (pattern_is_cjk(re ? re->string : NULL)) return true;
            }
        }
    }
    return false;
}

cwick_status cwick_tokenizer_load(const char *dir, const cwick_model_spec *spec,
                                cwick_tokenizer **out, cwick_error *err) {
    cwick_error_clear(err);
    cwick_tokenizer *tok = cwick_xmalloc(sizeof(cwick_tokenizer));
    memset(tok, 0, sizeof(*tok));
    /* The real GLM tokenizer.json is 20 MB and takes tens of seconds to parse
     * and index; without this line the startup looks frozen between "engine
     * ready" and "server listening" (observed as a 26 s silent gap). */
    CWICK_INFO("init", "loading tokenizer from %s", dir);
    double t0 = now_ms();
    cwick_status st = tokenizer_load_impl(dir, spec, tok, err);
    if (st == CWICK_OK)
        CWICK_INFO("init", "tokenizer ready: %zu vocab entries, %zu merges, "
                          "%zu special tokens (%.1f s)",
                  tok->vocab_size, tok->merge_count, tok->special_count,
                  (now_ms() - t0) / 1000.0);
    /* On failure the impl has already released the half-built tokenizer. */
    if (st != CWICK_OK) return st;
    *out = tok;
    return CWICK_OK;
}

static cwick_status tokenizer_load_impl(const char *dir,
                                       const cwick_model_spec *spec,
                                       cwick_tokenizer *tok, cwick_error *err) {
    char vpath[1152], mpath[1152];
    cwick_path_join(vpath, sizeof(vpath), dir, "tokenizer/vocab.json");
    cwick_path_join(mpath, sizeof(mpath), dir, "tokenizer/merges.txt");

    char *text = NULL;
    size_t len = 0;
    if (cwick_read_file(vpath, &text, &len) == 0) {
        /* ---- legacy layout: vocab.json + merges.txt ---- */
        json_value *root = cwick_json_parse(text, len, err);
        free(text);
        if (!root || root->kind != JSON_OBJECT) {
            if (root) cwick_json_free(root);
            cwick_error_set(err, CWICK_ERR_PARSE, "tokenizer vocab is not a JSON object");
            cwick_tokenizer_free(tok);
            return CWICK_ERR_PARSE;
        }
        tok->vocab_size = root->count;
        tok->vocab = cwick_xmalloc(tok->vocab_size * sizeof(cwick_vocab_entry));
        for (size_t i = 0; i < root->count; i++) {
            const char *key = root->members[i].key;
            tok->vocab[i].key = cwick_strdup(key);
            tok->vocab[i].id = cwick_json_int(root->members[i].value, -1);
            tok->vocab[i].is_byte = strlen(key) == 1;
        }
        cwick_json_free(root);
        build_byte_map(tok);

        /* merges.txt: lines "a b" */
        if (cwick_read_file(mpath, &text, &len) == 0) {
            const char *p = text;
            size_t cap = 16, n = 0;
            tok->merges = cwick_xmalloc(cap * sizeof(cwick_merge_entry));
            while (p < text + len) {
                const char *line = p;
                while (p < text + len && *p != '\n') p++;
                size_t llen = (size_t)(p - line);
                if (llen > 0 && line[0] != '#') {
                    if (n == cap) { cap *= 2; tok->merges = cwick_xrealloc(tok->merges, cap * sizeof(cwick_merge_entry)); }
                    char *pair = cwick_xmalloc(llen + 1);
                    memcpy(pair, line, llen);
                    pair[llen] = '\0';
                    while (llen > 0 && (pair[llen - 1] == ' ' || pair[llen - 1] == '\r')) pair[--llen] = '\0';
                    tok->merges[n++].pair = pair;
                }
                if (p < text + len) p++; /* skip newline */
            }
            tok->merge_count = n;
            free(text);
        }
        vocab_index_build(tok);
        return CWICK_OK;
    }

    /* ---- real HF layout: tokenizer.json at the model root ---- */
    char tpath[1152];
    cwick_path_join(tpath, sizeof(tpath), dir, "tokenizer.json");
    if (cwick_read_file(tpath, &text, &len) == 0) {
        json_value *root = cwick_json_parse(text, len, err);
        free(text);
        if (!root || root->kind != JSON_OBJECT) {
            if (root) cwick_json_free(root);
            cwick_error_set(err, CWICK_ERR_PARSE, "tokenizer.json is not a JSON object");
            cwick_tokenizer_free(tok);
            return CWICK_ERR_PARSE;
        }
        json_value *model = cwick_json_get(root, "model");
        json_value *vocab = model ? cwick_json_get(model, "vocab") : NULL;
        json_value *merges = model ? cwick_json_get(model, "merges") : NULL;
        if (!model || !vocab || vocab->kind != JSON_OBJECT ||
            !merges || merges->kind != JSON_ARRAY) {
            cwick_error_set(err, CWICK_ERR_CONTRACT,
                           "tokenizer.json missing model.vocab / model.merges");
            cwick_json_free(root);
            cwick_tokenizer_free(tok);
            return CWICK_ERR_CONTRACT;
        }
        tok->vocab_size = vocab->count;
        tok->vocab = cwick_xmalloc(tok->vocab_size * sizeof(cwick_vocab_entry));
        for (size_t i = 0; i < vocab->count; i++) {
            tok->vocab[i].key = cwick_strdup(vocab->members[i].key);
            tok->vocab[i].id = cwick_json_int(vocab->members[i].value, -1);
            tok->vocab[i].is_byte = false;
        }
        build_byte_map_gpt2(tok);

        /* merges: either a list of [left, right] pairs or -- the GPT-2 style
         * this checkpoint uses -- a list of "left right" strings. Both end up
         * as one "left right" rule here. */
        size_t n = cwick_json_array_len(merges);
        tok->merges = cwick_xmalloc((n ? n : 1) * sizeof(cwick_merge_entry));
        tok->merge_count = 0;
        for (size_t i = 0; i < n; i++) {
            json_value *pair = cwick_json_array_at(merges, i);
            const char *left = NULL, *right = NULL;
            char *joined = NULL;
            if (pair && pair->kind == JSON_ARRAY && cwick_json_array_len(pair) == 2) {
                left = cwick_json_str(cwick_json_array_at(pair, 0));
                right = cwick_json_str(cwick_json_array_at(pair, 1));
                if (!left || !right) continue;
                size_t llen = strlen(left), rlen = strlen(right);
                joined = cwick_xmalloc(llen + rlen + 2);
                memcpy(joined, left, llen);
                joined[llen] = ' ';
                memcpy(joined + llen + 1, right, rlen);
                joined[llen + 1 + rlen] = '\0';
            } else if (pair && pair->kind == JSON_STRING) {
                /* "Ġ t": split on the single separating space. A token may
                 * itself contain a space in the byte-level alphabet (Ġ), but
                 * never a literal one, so the first space is the separator. */
                const char *s = pair->string;
                const char *sp = strchr(s, ' ');
                if (!sp || sp == s || !sp[1]) continue;
                joined = cwick_strdup(s);
            } else {
                continue;
            }
            tok->merges[tok->merge_count++].pair = joined;
        }
        /* added_tokens: the control tokens of this checkpoint. They must be
         * matched as whole strings, otherwise "<|system|>" is encoded as six
         * byte-pieces and the model reads a different conversation than the
         * one the client sent. */
        json_value *added = cwick_json_get(root, "added_tokens");
        if (added && added->kind == JSON_ARRAY) {
            size_t na = cwick_json_array_len(added);
            tok->special = cwick_xmalloc((na ? na : 1) * sizeof(cwick_vocab_entry));
            for (size_t i = 0; i < na; i++) {
                json_value *t = cwick_json_array_at(added, i);
                if (!t) continue;
                const char *content = cwick_json_str(cwick_json_get(t, "content"));
                int64_t id = cwick_json_int(cwick_json_get(t, "id"), -1);
                if (!content || !*content || id < 0) continue;
                tok->special[tok->special_count].key = cwick_strdup(content);
                tok->special[tok->special_count].id = id;
                tok->special[tok->special_count].is_byte = false;
                tok->special_count++;
            }
            specials_sort(tok);
            for (size_t i = 0; i < tok->special_count; i++)
                tok->special_first[(unsigned char)tok->special[i].key[0]] = true;
        }
        tok->pretok_ds = pretok_is_deepseek(root);
        cwick_json_free(root);
        vocab_index_build(tok);
        id_index_build(tok);
        pair_index_build(tok);
        return CWICK_OK;
    }

    /* ---- no vocab: byte fallback (synthetic / tiny models) ---- */
    if (spec->vocab_size >= 256) {
        tok->byte_to_id = cwick_xmalloc(256 * sizeof(int64_t));
        for (int i = 0; i < 256; i++) tok->byte_to_id[i] = i;
        tok->byte_fallback = true;
        tok->byte_level = false;   /* a synthetic model speaks raw bytes, not the
                                    * GPT-2 byte-level alphabet; decode must emit the
                                    * byte a token id stands for, not map it through
                                    * <-> char translation. */
        tok->vocab_size = (size_t)spec->vocab_size;
        tok->vocab = cwick_xmalloc((size_t)spec->vocab_size * sizeof(cwick_vocab_entry));
        memset(tok->vocab, 0, (size_t)spec->vocab_size * sizeof(cwick_vocab_entry));
        /* Build a 1:1 byte vocabulary so the model's own generated ids decode
         * back to bytes. byte_to_id[b] == b, so token id b IS byte b; here we
         * give the decoder a string to copy and an id->slot index to find it
         * by. Everything above 255 (if vocab_size is larger) simply has no
         * string and decodes as nothing, which is the correct behaviour for a
         * synthetic test checkpoint that only ever emits byte tokens. */
        size_t nb = tok->vocab_size < 256 ? tok->vocab_size : 256;
        for (size_t i = 0; i < nb; i++) {
            tok->vocab[i].id = (int64_t)i;
            tok->vocab[i].key = cwick_xmalloc(2);
            tok->vocab[i].key[0] = (char)(unsigned char)i;
            tok->vocab[i].key[1] = '\0';
            tok->vocab[i].is_byte = true;
        }
        vocab_index_build(tok);
        id_index_build(tok);
        return CWICK_OK;
    }
    cwick_error_set(err, CWICK_ERR_IO, "tokenizer vocab missing: %s", vpath);
    cwick_tokenizer_free(tok);
    return CWICK_ERR_IO;
}

/* ------------------------------------------------------------------ */
/* BPE encode                                                          */
/* ------------------------------------------------------------------ */

typedef struct {
    int64_t *ids;   /* current token ids (byte or merged) */
    int64_t n;
    int64_t cap;
} tokseq;

/* ------------------------------------------------------------------ */
/* Vocabulary index                                                    */
/*                                                                     */
/* vocab_lookup used to linear-scan every entry. GLM-5.3-Flash has a     */
/* 154,880-entry vocabulary and ~150k merge rules, so encoding one short */
/* prompt cost ~7e10 string comparisons — over 100 seconds. That made     */
/* startup look like a weight-loading problem when the weights were       */
/* already loaded in a few seconds. An open-addressed index makes the     */
/* lookup O(1).                                                           */
/* ------------------------------------------------------------------ */
static uint64_t vocab_hash(const char *s, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) {
        h ^= (unsigned char)s[i];
        h *= 1099511628211ull;
    }
    return h;
}

static void vocab_index_build(cwick_tokenizer *tok) {
    size_t cap = 16;
    while (cap < tok->vocab_size * 2) cap *= 2;
    tok->index_mask = cap - 1;
    tok->index = cwick_xmalloc(cap * sizeof(int64_t));
    for (size_t i = 0; i < cap; i++) tok->index[i] = -1;
    for (size_t i = 0; i < tok->vocab_size; i++) {
        const char *k = tok->vocab[i].key;
        if (!k) continue;            /* padding / unset entries carry no string */
        size_t h = (size_t)(vocab_hash(k, strlen(k)) & tok->index_mask);
        while (tok->index[h] >= 0) h = (h + 1) & tok->index_mask;
        tok->index[h] = (int64_t)i;
    }
}

static int64_t vocab_lookup(const cwick_tokenizer *tok, const char *s, int64_t len) {
    if (tok->index) {
        size_t h = (size_t)(vocab_hash(s, (size_t)len) & tok->index_mask);
        while (tok->index[h] >= 0) {
            const cwick_vocab_entry *e = &tok->vocab[(size_t)tok->index[h]];
            if (strlen(e->key) == (size_t)len &&
                strncmp(e->key, s, (size_t)len) == 0)
                return e->id;
            h = (h + 1) & tok->index_mask;
        }
        return -1;
    }
    for (size_t i = 0; i < tok->vocab_size; i++) {
        if (strlen(tok->vocab[i].key) == (size_t)len &&
            strncmp(tok->vocab[i].key, s, (size_t)len) == 0) {
            return tok->vocab[i].id;
        }
    }
    return -1;
}

/* ------------------------------------------------------------------ */
/* Encode                                                              */
/* ------------------------------------------------------------------ */

typedef struct { uint32_t *v; int64_t n, cap; } idvec;

static void iv_push(idvec *iv, uint32_t id) {
    if (iv->n == iv->cap) {
        iv->cap = iv->cap ? iv->cap * 2 : 64;
        iv->v = cwick_xrealloc(iv->v, (size_t)iv->cap * sizeof(uint32_t));
    }
    iv->v[iv->n++] = id;
}

/* ------------------------------------------------------------------ */
/* Merge-pair index                                                    */
/*                                                                     */
/* Encoding used to re-resolve all 321k merge rules (three hash lookups   */
/* each) on every call, then walk the whole prompt once per rule. The      */
/* rules are fixed at load time, so they are resolved once into           */
/* (left id, right id) -> (rank, merged id) and applied per pre-token      */
/* chunk instead: a 20-byte word now costs ~400 pair lookups instead of    */
/* 6.4M.                                                                  */
/* ------------------------------------------------------------------ */
static uint64_t pair_mix(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 29; return x;
}

static void pair_index_build(cwick_tokenizer *tok) {
    size_t cap = 16;
    while (cap < tok->merge_count * 2) cap *= 2;
    tok->pair_mask = cap - 1;
    tok->pair_keys = cwick_xmalloc(cap * sizeof(uint64_t));
    tok->pair_rank = cwick_xmalloc(cap * sizeof(int32_t));
    tok->pair_ab = cwick_xmalloc(cap * sizeof(int64_t));
    for (size_t i = 0; i < cap; i++) { tok->pair_keys[i] = 0; tok->pair_rank[i] = -1; tok->pair_ab[i] = -1; }
    for (size_t m = 0; m < tok->merge_count; m++) {
        const char *rule = tok->merges[m].pair;
        const char *sp = strchr(rule, ' ');
        if (!sp) continue;
        size_t llen = (size_t)(sp - rule), rlen = strlen(sp + 1);
        int64_t a = vocab_lookup(tok, rule, (int64_t)llen);
        int64_t b = vocab_lookup(tok, sp + 1, (int64_t)rlen);
        char *joined = cwick_xmalloc(llen + rlen + 1);
        memcpy(joined, rule, llen);
        memcpy(joined + llen, sp + 1, rlen);
        joined[llen + rlen] = '\0';
        int64_t ab = vocab_lookup(tok, joined, (int64_t)(llen + rlen));
        free(joined);
        if (a < 0 || b < 0 || ab < 0) continue;
        uint64_t key = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
        size_t h = (size_t)(pair_mix(key) & tok->pair_mask);
        while (tok->pair_rank[h] >= 0) h = (h + 1) & tok->pair_mask;
        tok->pair_keys[h] = key;
        tok->pair_rank[h] = (int32_t)m;
        tok->pair_ab[h] = ab;
        tok->pair_count++;
    }
}

/* Rank of the (a,b) merge, or -1 when the pair is not a merge rule. */
static int64_t pair_lookup(const cwick_tokenizer *tok, int64_t a, int64_t b,
                           int64_t *ab) {
    if (!tok->pair_keys) return -1;
    uint64_t key = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    size_t h = (size_t)(pair_mix(key) & tok->pair_mask);
    while (tok->pair_rank[h] >= 0) {
        if (tok->pair_keys[h] == key) {
            if (ab) *ab = tok->pair_ab[h];
            return tok->pair_rank[h];
        }
        h = (h + 1) & tok->pair_mask;
    }
    return -1;
}

/* ------------------------------------------------------------------ */
/* Pre-tokenizer                                                       */
/*                                                                     */
/* The GLM checkpoint's pre_tokenizer is                               */
/*   Split(/(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|       */
/*          \p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|   */
/*          \s+/, Isolated) + ByteLevel                                  */
/* Merges must not cross the chunks that regex produces, otherwise a      */
/* word and the space in front of it fuse into one token and the model    */
/* sees a text it was never trained on. Unicode letter/number classes     */
/* are approximated: every non-ASCII character is treated as a letter,    */
/* which is right for CJK and accented Latin and only ever over-merges    */
/* runs of exotic symbols.                                                */
/* ------------------------------------------------------------------ */
typedef enum { CC_SPACE, CC_LETTER, CC_DIGIT, CC_OTHER } cc_t;

static size_t cp_len_at(const char *s, size_t n, size_t i) {
    unsigned char c = (unsigned char)s[i];
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return i + 1 < n ? 2 : 1;
    if ((c & 0xF0) == 0xE0) return i + 2 < n ? 3 : 1;
    if ((c & 0xF8) == 0xF0) return i + 3 < n ? 4 : 1;
    return 1;
}

static cc_t cc_at(const char *s, size_t n, size_t i) {
    unsigned char c = (unsigned char)s[i];
    (void)n;
    if (c < 0x80) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f')
            return CC_SPACE;
        if (c >= '0' && c <= '9') return CC_DIGIT;
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return CC_LETTER;
        return CC_OTHER;
    }
    return CC_LETTER;
}

static size_t run_of(const char *s, size_t n, size_t i, cc_t want, size_t maxcps) {
    size_t j = i, used = 0;
    while (j < n && cc_at(s, n, j) == want && used < maxcps) {
        j += cp_len_at(s, n, j);
        used++;
    }
    return j;
}

/* End of the chunk starting at i (always > i). */
static size_t next_chunk(const char *s, size_t n, size_t i) {
    size_t j;
    switch (cc_at(s, n, i)) {
    case CC_LETTER:
        return run_of(s, n, i, CC_LETTER, SIZE_MAX);
    case CC_DIGIT:
        return run_of(s, n, i, CC_DIGIT, 3);      /* \p{N}{1,3} */
    case CC_OTHER: {
        /* "[^\r\n\p{L}\p{N}]?\p{L}+": a symbol directly followed by letters
         * travels with them ("'s", "-ing"). */
        size_t ln = cp_len_at(s, n, i);
        if (i + ln < n && cc_at(s, n, i + ln) == CC_LETTER)
            return run_of(s, n, i + ln, CC_LETTER, SIZE_MAX);
        j = run_of(s, n, i, CC_OTHER, SIZE_MAX);
        while (j < n && (s[j] == '\n' || s[j] == '\r')) j++;
        return j;
    }
    default: {  /* CC_SPACE */
        size_t k = i;
        bool nl = false;
        while (k < n && cc_at(s, n, k) == CC_SPACE) {
            if (s[k] == '\n' || s[k] == '\r') nl = true;
            k++;
        }
        if (nl || k >= n) return k;               /* \s*[\r\n]+ or trailing run */
        cc_t nx = cc_at(s, n, k);
        /* only the last whitespace character joins what follows; the rest of
         * the run is its own chunk ("   out" -> "ĠĠ" + " out") */
        if (k - 1 > i) return k - 1;
        if (nx == CC_LETTER) return run_of(s, n, k, CC_LETTER, SIZE_MAX);
        if (nx == CC_DIGIT) return run_of(s, n, k, CC_DIGIT, 3);
        j = run_of(s, n, k, CC_OTHER, SIZE_MAX);
        while (j < n && (s[j] == '\n' || s[j] == '\r')) j++;
        return j;
    }
    }
}

/* ------------------------------------------------------------------ */
/* DeepSeek-V4.1 pre-tokenizer                                          */
/*                                                                     */
/*   Split(/\p{N}{1,3}/)                                                */
/*   Split/[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+/                  */
/*   Split/[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|             */
/*         [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+|                        */
/*          ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/             */
/*   ByteLevel(add_prefix_space=false)                                  */
/*                                                                     */
/* Same job as next_chunk above with three differences that change the  */
/* token stream: digits are split first (so "abc123" is two chunks, not  */
/* one), CJK is isolated (so "hi你" is two chunks), and the punctuation  */
/* class is \p{P}\p{S} rather than "everything that is not a letter or  */
/* a digit". The classification is still approximate: non-ASCII that is */
/* not in the CJK ranges counts as a letter.                           */
/* ------------------------------------------------------------------ */
typedef enum { DS_SPACE, DS_PUNCT, DS_SYMBOL, DS_LETTER, DS_DIGIT, DS_CJK } dscc_t;

static bool ds_ascii_punct(unsigned char c) {
    return strchr("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~", c) != NULL;
}

static dscc_t ds_cc_at(const char *s, size_t n, size_t i) {
    unsigned char c = (unsigned char)s[i];
    (void)n;
    if (c < 0x80) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f')
            return DS_SPACE;
        if (c >= '0' && c <= '9') return DS_DIGIT;
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return DS_LETTER;
        if (ds_ascii_punct(c)) return DS_PUNCT;
        return DS_SYMBOL;
    }
    /* decode the code point just far enough to place the CJK ranges */
    unsigned cp = 0;
    if ((c & 0xE0) == 0xC0) cp = c & 0x1F;
    else if ((c & 0xF0) == 0xE0) cp = c & 0x0F;
    else if ((c & 0xF8) == 0xF0) cp = c & 0x07;
    else return DS_LETTER;
    for (size_t k = 1; k < 4 && i + k < n; k++) {
        unsigned char cc = (unsigned char)s[i + k];
        if ((cc & 0xC0) != 0x80) break;
        cp = (cp << 6) | (cc & 0x3F);
        if ((c == 0xE0 && k == 2) || (c == 0xF0 && k == 3)) break;
    }
    if ((cp >= 0x4E00 && cp <= 0x9FA5) || (cp >= 0x3040 && cp <= 0x309F) ||
        (cp >= 0x30A0 && cp <= 0x30FF))
        return DS_CJK;
    return DS_LETTER;
}

static size_t ds_run_of(const char *s, size_t n, size_t i, dscc_t want,
                        size_t maxcps) {
    size_t j = i, used = 0;
    while (j < n && ds_cc_at(s, n, j) == want && used < maxcps) {
        j += cp_len_at(s, n, j);
        used++;
    }
    return j;
}

/* End of the chunk starting at i (always > i). */
static size_t next_chunk_ds(const char *s, size_t n, size_t i) {
    /* \p{N}{1,3} is its own split, so a digit run is always exactly 1-3. */
    if (ds_cc_at(s, n, i) == DS_DIGIT)
        return ds_run_of(s, n, i, DS_DIGIT, 3);

    /* [CJK]+ is its own split. */
    if (ds_cc_at(s, n, i) == DS_CJK) {
        size_t j = i;
        while (j < n && ds_cc_at(s, n, j) == DS_CJK) j += cp_len_at(s, n, j);
        return j;
    }

    /* [!"#...~][A-Za-z]+ : ASCII punctuation glued to ASCII letters. */
    if (ds_cc_at(s, n, i) == DS_PUNCT) {
        size_t ln = cp_len_at(s, n, i);
        if (i + ln < n && ds_cc_at(s, n, i + ln) == DS_LETTER)
            return ds_run_of(s, n, i + ln, DS_LETTER, SIZE_MAX);
    }

    /* [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+ : a leading symbol that is neither
     * a letter nor punctuation travels with the letters after it. */
    if (ds_cc_at(s, n, i) == DS_SYMBOL) {
        size_t ln = cp_len_at(s, n, i);
        if (i + ln < n && ds_cc_at(s, n, i + ln) == DS_LETTER)
            return ds_run_of(s, n, i + ln, DS_LETTER, SIZE_MAX);
    }

    /* ?[\p{P}\p{S}]+[\r\n]* : a punctuation/symbol run, plus trailing
     * newlines. The leading space is handled by the space branch below. */
    {
        dscc_t c0 = ds_cc_at(s, n, i);
        if (c0 == DS_PUNCT || c0 == DS_SYMBOL) {
            size_t j = i;
            while (j < n) {
                dscc_t c = ds_cc_at(s, n, j);
                if (c != DS_PUNCT && c != DS_SYMBOL) break;
                j += cp_len_at(s, n, j);
            }
            while (j < n && (s[j] == '\n' || s[j] == '\r')) j++;
            return j;
        }
    }

    /* \s*[\r\n]+ : any whitespace run that contains a newline. */
    if (ds_cc_at(s, n, i) == DS_SPACE) {
        size_t k = i;
        bool nl = false;
        while (k < n && ds_cc_at(s, n, k) == DS_SPACE) {
            if (s[k] == '\n' || s[k] == '\r') nl = true;
            k++;
        }
        if (nl || k >= n) return k;
        /* \s+(?!\S) already covered by k >= n. What is left is \s+ followed
         * by something: only the last space of the run joins it. */
        if (k - 1 > i) return k - 1;
        /* exactly one space: it prefixes the next chunk */
        return next_chunk_ds(s, n, k);
    }

    /* plain letter run */
    return ds_run_of(s, n, i, DS_LETTER, SIZE_MAX);
}

/* ---- one pre-token chunk: byte-level string -> ids ------------------ */
static void encode_chunk(const cwick_tokenizer *tok, const char *s, size_t n,
                         idvec *out) {
    if (n == 0) return;
    const int64_t *b2c = gpt2_byte_to_char();
    char *buf = cwick_xmalloc(n * 2 + 1);
    size_t blen = 0;
    int64_t *seq = cwick_xmalloc(n * sizeof(int64_t));
    int64_t k = 0;
    for (size_t i = 0; i < n; i++) {
        int64_t id = tok->byte_to_id[(unsigned char)s[i]];
        if (id < 0) { free(buf); free(seq); return; }   /* incomplete byte map */
        int cp = (int)b2c[(unsigned char)s[i]];
        if (cp < 0x80) {
            buf[blen++] = (char)cp;
        } else {
            buf[blen++] = (char)(0xC0 | (cp >> 6));
            buf[blen++] = (char)(0x80 | (cp & 0x3F));
        }
        seq[k++] = id;
    }
    buf[blen] = '\0';
    /* ignore_merges: the checkpoint's BPE looks the whole chunk up in the
     * vocabulary before it considers any merge rule. */
    int64_t whole = vocab_lookup(tok, buf, (int64_t)blen);
    free(buf);
    if (whole >= 0) { iv_push(out, (uint32_t)whole); free(seq); return; }

    /* fall back to BPE: merge the lowest-ranked adjacent pair, repeatedly */
    while (k > 1) {
        int64_t best_rank = INT64_MAX, best_i = -1, best_ab = -1;
        for (int64_t i = 0; i + 1 < k; i++) {
            int64_t ab = -1;
            int64_t r = pair_lookup(tok, seq[i], seq[i + 1], &ab);
            if (r >= 0 && r < best_rank) { best_rank = r; best_i = i; best_ab = ab; }
        }
        if (best_i < 0) break;
        seq[best_i] = best_ab;
        memmove(seq + best_i + 1, seq + best_i + 2,
                (size_t)(k - best_i - 2) * sizeof(int64_t));
        k--;
    }
    for (int64_t i = 0; i < k; i++) iv_push(out, (uint32_t)seq[i]);
    free(seq);
}

/* Longest added/special token starting at i, if any. */
static bool match_special(const cwick_tokenizer *tok, const char *s, size_t n,
                          size_t i, int64_t *id, size_t *slen) {
    for (size_t k = 0; k < tok->special_count; k++) {
        size_t l = strlen(tok->special[k].key);
        if (i + l <= n && memcmp(s + i, tok->special[k].key, l) == 0) {
            *id = tok->special[k].id;
            *slen = l;
            return true;
        }
    }
    return false;
}

/* First position in [from,to) where a special token starts, or SIZE_MAX.
 * A pre-token chunk must never swallow the beginning of one: ".<|user|>" is
 * "." + <|user|>, not a byte-level merge of the two. */
static size_t next_special_at(const cwick_tokenizer *tok, const char *s,
                              size_t n, size_t from, size_t to) {
    if (tok->special_count == 0) return SIZE_MAX;
    for (size_t p = from; p < to; p++) {
        if (!tok->special_first[(unsigned char)s[p]]) continue;
        int64_t id;
        size_t sl;
        if (match_special(tok, s, n, p, &id, &sl)) return p;
    }
    return SIZE_MAX;
}

cwick_status cwick_tokenizer_encode(const cwick_tokenizer *tok, const char *text,
                                  uint32_t **ids_out, int64_t *n_out,
                                  cwick_error *err) {
    cwick_error_clear(err);
    size_t len = strlen(text);

    if (tok->byte_fallback && tok->merge_count == 0) {
        uint32_t *ids = cwick_xmalloc((len + 1) * sizeof(uint32_t));
        size_t n = 0;
        for (size_t i = 0; i < len; i++) {
            unsigned char b = (unsigned char)text[i];
            int64_t id = tok->byte_to_id[b];
            if (id < 0) {
                cwick_error_set(err, CWICK_ERR_ARGUMENT, "byte 0x%02x not in vocabulary", b);
                free(ids);
                return CWICK_ERR_ARGUMENT;
            }
            ids[n++] = (uint32_t)id;
        }
        *ids_out = ids;
        *n_out = (int64_t)n;
        return CWICK_OK;
    }

    idvec iv = {0};
    size_t i = 0;
    while (i < len) {
        int64_t sid;
        size_t sl;
        if (match_special(tok, text, len, i, &sid, &sl)) {
            iv_push(&iv, (uint32_t)sid);
            i += sl;
            continue;
        }
        size_t j = tok->pretok_ds ? next_chunk_ds(text, len, i)
                                  : next_chunk(text, len, i);
        if (j <= i) j = i + cp_len_at(text, len, i);
        size_t sp = next_special_at(tok, text, len, i + 1, j);
        if (sp != SIZE_MAX) j = sp;
        encode_chunk(tok, text + i, j - i, &iv);
        i = j;
    }
    if (!iv.v) iv.v = cwick_xmalloc(sizeof(uint32_t));
    *ids_out = iv.v;
    *n_out = iv.n;
    return CWICK_OK;
}

/* ------------------------------------------------------------------ */
/* Decode                                                              */
/* ------------------------------------------------------------------ */
static const char *piece_for_id(const cwick_tokenizer *tok, uint32_t id,
                                bool *is_special) {
    *is_special = false;
    for (size_t i = 0; i < tok->special_count; i++)
        if (tok->special[i].id == (int64_t)id) { *is_special = true; return tok->special[i].key; }
    if (tok->id_to_slot && (int64_t)id < tok->id_to_slot_cap) {
        int64_t slot = tok->id_to_slot[id];
        if (slot >= 0) return tok->vocab[slot].key;
    }
    return NULL;
}

cwick_status cwick_tokenizer_decode(const cwick_tokenizer *tok, const uint32_t *ids,
                                  int64_t n, char **out, cwick_error *err) {
    cwick_error_clear(err);
    size_t cap = 256, len = 0;
    char *buf = cwick_xmalloc(cap + 1);
    for (int64_t i = 0; i < n; i++) {
        bool special = false;
        const char *piece = piece_for_id(tok, ids[i], &special);
        if (!piece) {
            free(buf);
            cwick_error_set(err, CWICK_ERR_ARGUMENT, "unknown token id %u", ids[i]);
            return CWICK_ERR_ARGUMENT;
        }
        if (special) {
            size_t plen = strlen(piece);
            if (len + plen + 1 > cap) { while (len + plen + 1 > cap) cap *= 2; buf = cwick_xrealloc(buf, cap + 1); }
            memcpy(buf + len, piece, plen);
            len += plen;
            continue;
        }
        if (tok->byte_level) {
            const int64_t *cp_to_byte = gpt2_cp_to_byte();
            size_t plen = strlen(piece), off = 0;
            while (off < plen) {
                size_t cplen = 0;
                int64_t cp = utf8_decode(piece + off, plen - off, &cplen);
                if (cplen == 0) break;
                if (cp >= 0 && cp < 512 && cp_to_byte[cp] >= 0) {
                    unsigned char b = (unsigned char)cp_to_byte[cp];
                    if (len + 1 > cap) { cap *= 2; buf = cwick_xrealloc(buf, cap + 1); }
                    buf[len++] = (char)b;
                } else {
                    if (len + cplen + 1 > cap) { while (len + cplen + 1 > cap) cap *= 2; buf = cwick_xrealloc(buf, cap + 1); }
                    memcpy(buf + len, piece + off, cplen);
                    len += cplen;
                }
                off += cplen;
            }
            continue;
        }
        size_t plen = strlen(piece);
        if (len + plen + 1 > cap) { while (len + plen + 1 > cap) cap *= 2; buf = cwick_xrealloc(buf, cap + 1); }
        memcpy(buf + len, piece, plen);
        len += plen;
    }
    buf[len] = '\0';
    *out = buf;
    return CWICK_OK;
}

/* ------------------------------------------------------------------ */
/* Chat template                                                       */
/*                                                                     */
/* Rendered exactly as zai-org/GLM-5.3-Flash/chat_template.jinja does:   */
/*                                                                       */
/*   [gMASK]<sop><|system|>Reasoning Effort: Max                         */
/*   [<|system|>\n# Tools … </tool_call>]      <- when tools are declared */
/*   <|system|>{system}<|user|>{user}          <- one block per message   */
/*   <|assistant|><think>                                                 */
/*                                                                       */
/* Every delimiter is an added token of this checkpoint (<sop>,           */
/* <|system|>, <think>, …), so the encoder has to emit them as single     */
/* ids rather than splitting them into bytes — that is what the           */
/* special-token pass in cwick_tokenizer_encode is for.                    */
/* ------------------------------------------------------------------ */

typedef struct { char *p; size_t len, cap; } strb;

static void strb_putn(strb *b, const char *s, size_t n) {
    if (b->len + n + 1 > b->cap) {
        while (b->len + n + 1 > b->cap) b->cap = b->cap ? b->cap * 2 : 256;
        b->p = cwick_xrealloc(b->p, b->cap);
    }
    memcpy(b->p + b->len, s, n);
    b->len += n;
    b->p[b->len] = '\0';
}
static void strb_puts(strb *b, const char *s) { if (s) strb_putn(b, s, strlen(s)); }

/* One argument value. Strings go in bare (that is what the template does);
 * scalars are spelled the way JSON spells them. */
static void put_arg_value(strb *b, const json_value *v) {
    if (!v) return;
    if (v->kind == JSON_STRING) { strb_puts(b, v->string); return; }
    if (v->kind == JSON_NUMBER) {
        char num[64];
        snprintf(num, sizeof(num), "%.17g", v->number);
        strb_puts(b, num);
        return;
    }
    if (v->kind == JSON_BOOL) { strb_puts(b, v->boolean ? "true" : "false"); return; }
}

/* Assistant turns may carry OpenAI tool_calls; the template writes them
 * back out in the checkpoint's own <tool_call> form. */
static void put_tool_calls(strb *b, const char *json) {
    if (!json || !*json) return;
    cwick_error e;
    cwick_error_clear(&e);
    json_value *root = cwick_json_parse(json, strlen(json), &e);
    if (!root || root->kind != JSON_ARRAY) { if (root) cwick_json_free(root); return; }
    for (size_t i = 0; i < root->count; i++) {
        json_value *item = cwick_json_array_at(root, i);
        json_value *fn = cwick_json_get(item, "function");
        if (!fn) fn = item;
        const char *name = cwick_json_str(cwick_json_get(fn, "name"));
        if (!name) continue;
        strb_puts(b, "\n<tool_call>");
        strb_puts(b, name);
        json_value *args = cwick_json_get(fn, "arguments");
        if (args && args->kind == JSON_STRING && args->string && *args->string) {
            cwick_error e2;
            cwick_error_clear(&e2);
            json_value *a = cwick_json_parse(args->string, strlen(args->string), &e2);
            if (a && a->kind == JSON_OBJECT) {
                for (size_t k = 0; k < a->count; k++) {
                    strb_puts(b, "<arg_key>");
                    strb_puts(b, a->members[k].key);
                    strb_puts(b, "</arg_key><arg_value>");
                    put_arg_value(b, a->members[k].value);
                    strb_puts(b, "</arg_value>");
                }
            }
            if (a) cwick_json_free(a);
        } else if (args && args->kind == JSON_OBJECT) {
            for (size_t k = 0; k < args->count; k++) {
                strb_puts(b, "<arg_key>");
                strb_puts(b, args->members[k].key);
                strb_puts(b, "</arg_key><arg_value>");
                put_arg_value(b, args->members[k].value);
                strb_puts(b, "</arg_value>");
            }
        }
        strb_puts(b, "</tool_call>");
    }
    cwick_json_free(root);
}

static char *render_chat(const cwick_tokenizer *tok,
                         const cwick_chat_msg *msgs, size_t n,
                         const char *tools_block, const char *effort,
                         bool answer_at_once) {
    (void)tok;
    strb b = {0};
    strb_puts(&b, "[gMASK]<sop>");
    strb_puts(&b, "<|system|>Reasoning Effort: ");
    strb_puts(&b, effort && *effort ? effort : "Max");
    if (tools_block) strb_puts(&b, tools_block);
    for (size_t i = 0; i < n; i++) {
        const char *role = msgs[i].role ? msgs[i].role : "user";
        /* `developer` is OpenAI's current name for the instruction turn; the
         * checkpoint's template has one such role, and letting the name fall
         * through to the user branch would turn a system instruction into a
         * user request. */
        if (strcmp(role, "system") == 0 || strcmp(role, "developer") == 0) {
            strb_puts(&b, "<|system|>");
            strb_puts(&b, msgs[i].content);
        } else if (strcmp(role, "assistant") == 0) {
            strb_puts(&b, "<|assistant|><think></think>");
            strb_puts(&b, msgs[i].content);
            put_tool_calls(&b, msgs[i].tool_calls_json);
        } else if (strcmp(role, "tool") == 0) {
            strb_puts(&b, "<|observation|><tool_response>");
            strb_puts(&b, msgs[i].content);
            strb_puts(&b, "</tool_response>");
        } else {
            strb_puts(&b, "<|user|>");
            strb_puts(&b, msgs[i].content);
        }
    }
    /* The assistant header opens a reasoning block. `answer_at_once` closes it
     * on the same turn, which is the checkpoint's own way of saying "answer
     * now, do not think first": the next token is then the answer, not the
     * opening of a thinking block. */
    strb_puts(&b, answer_at_once ? "<|assistant|><think></think>"
                                 : "<|assistant|><think>");
    if (!b.p) b.p = cwick_xmalloc(1), b.p[0] = '\0';
    return b.p;
}

cwick_status cwick_chat_encode_messages(const cwick_tokenizer *tok,
                                      const cwick_chat_msg *msgs, size_t n,
                                      const char *tools_block,
                                      const char *reasoning_effort,
                                      uint32_t **ids, int64_t *n_out,
                                      cwick_error *err) {
    char *text = render_chat(tok, msgs, n, tools_block, reasoning_effort, false);
    cwick_status st = cwick_tokenizer_encode(tok, text, ids, n_out, err);
    free(text);
    return st;
}

cwick_status cwick_chat_encode_tools(const cwick_tokenizer *tok, const char *system,
                                   const char *user, const char *tools_block,
                                   uint32_t **ids, int64_t *n, cwick_error *err) {
    cwick_chat_msg msgs[2];
    size_t count = 0;
    if (system) { msgs[count].role = "system"; msgs[count].content = system; msgs[count].tool_calls_json = NULL; count++; }
    msgs[count].role = "user";
    msgs[count].content = user;
    msgs[count].tool_calls_json = NULL;
    count++;
    return cwick_chat_encode_messages(tok, msgs, count, tools_block, NULL, ids, n, err);
}

cwick_status cwick_chat_encode(const cwick_tokenizer *tok, const char *system,
                             const char *user, uint32_t **ids, int64_t *n,
                             cwick_error *err) {
    return cwick_chat_encode_tools(tok, system, user, NULL, ids, n, err);
}

/* Same conversation, but the assistant turn answers immediately: the reasoning
 * block is opened and closed before the model writes anything, so the next
 * token is the answer itself. Used by the decision endpoint, which reads one
 * token's distribution and never runs a decode loop. */
cwick_status cwick_chat_encode_decision(const cwick_tokenizer *tok,
                                        const char *system, const char *user,
                                        uint32_t **ids, int64_t *n,
                                        cwick_error *err) {
    cwick_chat_msg msgs[2];
    size_t count = 0;
    if (system) {
        msgs[count].role = "system";
        msgs[count].content = system;
        msgs[count].tool_calls_json = NULL;
        count++;
    }
    msgs[count].role = "user";
    msgs[count].content = user;
    msgs[count].tool_calls_json = NULL;
    count++;
    char *text = render_chat(tok, msgs, count, NULL, "Low", true);
    cwick_status st = cwick_tokenizer_encode(tok, text, ids, n, err);
    free(text);
    return st;
}
