/* json.c — minimal recursive-descent JSON DOM parser (config + HTTP bodies).
 *
 * Values are heap objects; strings are NUL-terminated copies. Numbers keep
 * both an int64 and a double view. This is intentionally small and strict.
 */
#include "json.h"

#include <stdio.h>
#include "util.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void json_free_value(json_value *v) {
    if (!v) return;
    switch (v->kind) {
        case JSON_STRING:
            free(v->string);
            break;
        case JSON_ARRAY:
            for (size_t i = 0; i < v->count; i++) json_free_value(v->items[i]);
            free(v->items);
            break;
        case JSON_OBJECT:
            for (size_t i = 0; i < v->count; i++) {
                free(v->members[i].key);
                json_free_value(v->members[i].value);
            }
            free(v->members);
            break;
        default:
            break;
    }
    free(v);
}

void cwick_json_free(json_value *v) { json_free_value(v); }

/* Parser state */
typedef struct {
    const char *start;
    const char *p;
    const char *end;
    cwick_error *err;
    unsigned depth;
} json_parser;

static void json_skip_ws(json_parser *ps) {
    while (ps->p < ps->end && (*ps->p == ' ' || *ps->p == '\t' ||
                               *ps->p == '\n' || *ps->p == '\r'))
        ps->p++;
}

static bool json_fail(json_parser *ps, const char *msg) {
    long off = (long)(ps->p - ps->start);
    cwick_error_set(ps->err, CWICK_ERR_PARSE, "json: %s at offset %ld", msg, off);
    return false;
}

static json_value *json_parse_value(json_parser *ps);
static json_value *json_parse_value_inner(json_parser *ps);

/* ---- duplicate-key detection --------------------------------------
 * Rejecting duplicate object keys keeps the parser strict for OpenAI
 * envelopes, but the naive "compare against every key so far" check is
 * quadratic: a real tokenizer.json has 155k vocab keys, and that check alone
 * cost ~20 s of startup (measured: 26.3 s total for the 20 MB file). An
 * open-addressing set keeps the strictness at linear cost. */
typedef struct {
    const char **slots;
    size_t mask;          /* capacity - 1; capacity is a power of two */
    size_t count;
} json_key_set;

static size_t json_key_hash(const char *s) {
    size_t h = 1469598103934665603ull;         /* FNV-1a */
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        h ^= (size_t)*p;
        h *= 1099511628211ull;
    }
    return h;
}

static void json_key_set_free(json_key_set *ks) {
    free(ks->slots);
    ks->slots = NULL;
    ks->mask = 0;
    ks->count = 0;
}

static bool json_key_set_add(json_key_set *ks, const char *key) {
    if (!ks->slots) {
        size_t cap = 64;
        ks->slots = calloc(cap, sizeof(*ks->slots));
        if (!ks->slots) return true;           /* OOM: skip the check */
        ks->mask = cap - 1;
        ks->count = 0;
    } else if (ks->count * 2 >= ks->mask + 1) {
        size_t cap = (ks->mask + 1) * 2;
        const char **ns = calloc(cap, sizeof(*ns));
        if (!ns) return true;                  /* OOM: skip the check */
        size_t nmask = cap - 1;
        for (size_t i = 0; i <= ks->mask; i++) {
            const char *k = ks->slots[i];
            if (!k) continue;
            size_t j = json_key_hash(k) & nmask;
            while (ns[j]) j = (j + 1) & nmask;
            ns[j] = k;
        }
        free(ks->slots);
        ks->slots = ns;
        ks->mask = nmask;
    }
    size_t j = json_key_hash(key) & ks->mask;
    while (ks->slots[j]) {
        if (strcmp(ks->slots[j], key) == 0) return true;   /* duplicate */
        j = (j + 1) & ks->mask;
    }
    ks->slots[j] = key;
    ks->count++;
    return false;
}

static json_value *json_new(json_kind kind) {
    json_value *v = cwick_xmalloc(sizeof(json_value));
    v->kind = kind;
    v->boolean = false;
    v->number = 0.0;
    v->integer = 0;
    v->string = NULL;
    v->items = NULL;
    v->members = NULL;
    v->count = 0;
    return v;
}

static bool json_append_char(char **buf, size_t *len, size_t *cap, char c) {
    if (*len + 1 >= *cap) { *cap *= 2; *buf = cwick_xrealloc(*buf, *cap + 1); }
    (*buf)[(*len)++] = c;
    return true;
}

static bool json_parse_string(json_parser *ps, char **out) {
    if (ps->p >= ps->end || *ps->p != '"') return json_fail(ps, "expected string");
    ps->p++;
    size_t cap = 64, len = 0;
    char *buf = cwick_xmalloc(cap + 1);
    while (ps->p < ps->end) {
        char c = *ps->p;
        if (c == '"') {
            ps->p++;
            buf[len] = '\0';
            *out = buf;
            return true;
        }
        if (c == '\\') {
            ps->p++;
            if (ps->p >= ps->end) { free(buf); return json_fail(ps, "bad escape"); }
            char e = *ps->p++;
            if (e == 'u') {
                if (ps->end - ps->p < 4) { free(buf); return json_fail(ps, "bad \\u"); }
                unsigned cp = 0;
                for (int i = 0; i < 4; i++) {
                    char h = *ps->p++;
                    cp = (cp << 4) | (unsigned)(h <= '9' ? (h - '0')
                            : (h <= 'F' ? (h - 'A' + 10) : (h - 'a' + 10)));
                }
                if (cp < 0x80) {
                    json_append_char(&buf, &len, &cap, (char)cp);
                } else if (cp < 0x800) {
                    json_append_char(&buf, &len, &cap, (char)(0xC0 | (cp >> 6)));
                    json_append_char(&buf, &len, &cap, (char)(0x80 | (cp & 0x3F)));
                } else {
                    json_append_char(&buf, &len, &cap, (char)(0xE0 | (cp >> 12)));
                    json_append_char(&buf, &len, &cap, (char)(0x80 | ((cp >> 6) & 0x3F)));
                    json_append_char(&buf, &len, &cap, (char)(0x80 | (cp & 0x3F)));
                }
                continue;
            }
            char out_c = 0;
            switch (e) {
                case '"': out_c = '"'; break;
                case '\\': out_c = '\\'; break;
                case '/': out_c = '/'; break;
                case 'b': out_c = '\b'; break;
                case 'f': out_c = '\f'; break;
                case 'n': out_c = '\n'; break;
                case 'r': out_c = '\r'; break;
                case 't': out_c = '\t'; break;
                default: free(buf); return json_fail(ps, "unknown escape");
            }
            json_append_char(&buf, &len, &cap, out_c);
            continue;
        }
        if ((unsigned char)c < 0x20) {
            free(buf);
            return json_fail(ps, "control char in string");
        }
        ps->p++;
        json_append_char(&buf, &len, &cap, c);
    }
    free(buf);
    return json_fail(ps, "unterminated string");
}

static bool json_parse_number(json_parser *ps, json_value *v) {
    const char *start = ps->p;
    bool neg = false;
    if (ps->p < ps->end && *ps->p == '-') { neg = true; ps->p++; }
    bool is_float = false;
    while (ps->p < ps->end) {
        char c = *ps->p;
        if (c >= '0' && c <= '9') { ps->p++; }
        else if (c == '.' || c == 'e' || c == 'E') { is_float = true; ps->p++; }
        else if ((c == '+' || c == '-') && (ps->p > start)) { ps->p++; }
        else break;
    }
    if (ps->p == start) return json_fail(ps, "bad number");
    char *tmp = cwick_xmalloc((size_t)(ps->p - start) + 1);
    memcpy(tmp, start, (size_t)(ps->p - start));
    tmp[ps->p - start] = '\0';
    double dbl = strtod(tmp, NULL);
    v->number = dbl;
    /* Integer view: exact for plain integers, rounded for float literals so
     * that `cwick_json_int()` on "10.0" still yields 10 rather than 0. */
    v->integer = is_float ? (int64_t)llround(dbl) : strtoll(tmp, NULL, 10);
    free(tmp);
    (void)neg;
    (void)is_float;
    return true;
}

static json_value *json_parse_value_inner(json_parser *ps) {
    json_skip_ws(ps);
    if (ps->p >= ps->end) { json_fail(ps, "unexpected end"); return NULL; }
    char c = *ps->p;
    if (c == '{') {
        ps->p++;
        json_value *obj = json_new(JSON_OBJECT);
        size_t cap = 4;
        obj->members = cwick_xmalloc(cap * sizeof(json_pair));
        json_key_set ks = {0};
        for (;;) {
            json_skip_ws(ps);
            if (ps->p < ps->end && *ps->p == '}') { ps->p++; break; }
            char *key = NULL;
            if (!json_parse_string(ps, &key)) { json_free_value(obj); json_key_set_free(&ks); return NULL; }
            if (json_key_set_add(&ks, key)) {
                cwick_error_set(ps->err, CWICK_ERR_PARSE,
                               "json: duplicate object key `%s`", key);
                free(key);
                json_free_value(obj);
                json_key_set_free(&ks);
                return NULL;
            }
            json_skip_ws(ps);
            if (ps->p >= ps->end || *ps->p != ':') {
                free(key); json_free_value(obj); json_fail(ps, "expected ':'"); json_key_set_free(&ks); return NULL;
            }
            ps->p++;
            json_value *val = json_parse_value(ps);
            if (!val) { free(key); json_free_value(obj); json_key_set_free(&ks); return NULL; }
            if (obj->count == cap) {
                cap *= 2;
                obj->members = cwick_xrealloc(obj->members, cap * sizeof(json_pair));
            }
            obj->members[obj->count].key = key;
            obj->members[obj->count].value = val;
            obj->count++;
            json_skip_ws(ps);
            if (ps->p >= ps->end) { json_free_value(obj); json_fail(ps, "unterminated object"); json_key_set_free(&ks); return NULL; }
            if (*ps->p == ',') {
                ps->p++;
                json_skip_ws(ps);
                /* RFC 8259 does not allow a trailing comma. Keeping the
                 * parser strict is important for OpenAI clients: it prevents
                 * malformed server envelopes from passing our own tests just
                 * because both sides share the same permissive parser. */
                if (ps->p >= ps->end || *ps->p == '}') {
                    json_free_value(obj);
                    json_fail(ps, "trailing comma in object");
                    json_key_set_free(&ks);
                    return NULL;
                }
                continue;
            }
            if (*ps->p == '}') { ps->p++; break; }
            json_free_value(obj);
            json_fail(ps, "expected ',' or '}'");
            json_key_set_free(&ks);
            return NULL;
        }
        json_key_set_free(&ks);
        return obj;
    }
    if (c == '[') {
        ps->p++;
        json_value *arr = json_new(JSON_ARRAY);
        size_t cap = 4;
        arr->items = cwick_xmalloc(cap * sizeof(json_value *));
        for (;;) {
            json_skip_ws(ps);
            if (ps->p < ps->end && *ps->p == ']') { ps->p++; break; }
            json_value *val = json_parse_value(ps);
            if (!val) { json_free_value(arr); return NULL; }
            if (arr->count == cap) {
                cap *= 2;
                arr->items = cwick_xrealloc(arr->items, cap * sizeof(json_value *));
            }
            arr->items[arr->count++] = val;
            json_skip_ws(ps);
            if (ps->p >= ps->end) { json_free_value(arr); json_fail(ps, "unterminated array"); return NULL; }
            if (*ps->p == ',') {
                ps->p++;
                json_skip_ws(ps);
                if (ps->p >= ps->end || *ps->p == ']') {
                    json_free_value(arr);
                    json_fail(ps, "trailing comma in array");
                    return NULL;
                }
                continue;
            }
            if (*ps->p == ']') { ps->p++; break; }
            json_free_value(arr);
            json_fail(ps, "expected ',' or ']'");
            return NULL;
        }
        return arr;
    }
    if (c == '"') {
        char *s = NULL;
        if (!json_parse_string(ps, &s)) return NULL;
        json_value *v = json_new(JSON_STRING);
        v->string = s;
        return v;
    }
    if (c == 't') {
        if (ps->end - ps->p >= 4 && strncmp(ps->p, "true", 4) == 0) {
            ps->p += 4;
            json_value *v = json_new(JSON_BOOL);
            v->boolean = true;
            return v;
        }
        json_fail(ps, "expected true"); return NULL;
    }
    if (c == 'f') {
        if (ps->end - ps->p >= 5 && strncmp(ps->p, "false", 5) == 0) {
            ps->p += 5;
            json_value *v = json_new(JSON_BOOL);
            v->boolean = false;
            return v;
        }
        json_fail(ps, "expected false"); return NULL;
    }
    if (c == 'n') {
        if (ps->end - ps->p >= 4 && strncmp(ps->p, "null", 4) == 0) {
            ps->p += 4;
            return json_new(JSON_NULL);
        }
        json_fail(ps, "expected null"); return NULL;
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
        json_value *v = json_new(JSON_NUMBER);
        if (!json_parse_number(ps, v)) { json_free_value(v); return NULL; }
        return v;
    }
    json_fail(ps, "unexpected token");
    return NULL;
}

static json_value *json_parse_value(json_parser *ps) {
    if (ps->depth >= 128) {
        json_fail(ps, "nesting exceeds 128 levels");
        return NULL;
    }
    ps->depth++;
    json_value *v = json_parse_value_inner(ps);
    ps->depth--;
    return v;
}

json_value *cwick_json_parse(const char *text, size_t len, cwick_error *err) {
    json_parser ps = { text, text, text + len, err, 0 };
    json_value *v = json_parse_value(&ps);
    if (!v) return NULL;
    json_skip_ws(&ps);
    if (ps.p != ps.end) {
        cwick_error_set(err, CWICK_ERR_PARSE, "json: trailing data");
        json_free_value(v);
        return NULL;
    }
    return v;
}

/* Accessors */
json_value *cwick_json_get(json_value *obj, const char *key) {
    if (!obj || obj->kind != JSON_OBJECT) return NULL;
    for (size_t i = 0; i < obj->count; i++) {
        if (strcmp(obj->members[i].key, key) == 0) return obj->members[i].value;
    }
    return NULL;
}

const char *cwick_json_str(json_value *v) {
    return v && v->kind == JSON_STRING ? v->string : NULL;
}

int64_t cwick_json_int(json_value *v, int64_t dflt) {
    if (!v) return dflt;
    if (v->kind == JSON_NUMBER) return v->integer;
    if (v->kind == JSON_BOOL) return v->boolean ? 1 : 0;
    return dflt;
}

double cwick_json_num(json_value *v, double dflt) {
    if (!v) return dflt;
    if (v->kind == JSON_NUMBER) return v->number;
    return dflt;
}

bool cwick_json_bool(json_value *v, bool dflt) {
    if (!v) return dflt;
    if (v->kind == JSON_BOOL) return v->boolean;
    if (v->kind == JSON_NUMBER) return v->integer != 0;
    return dflt;
}

size_t cwick_json_array_len(json_value *v) {
    return v && v->kind == JSON_ARRAY ? v->count : 0;
}

json_value *cwick_json_array_at(json_value *v, size_t i) {
    return v && v->kind == JSON_ARRAY && i < v->count ? v->items[i] : NULL;
}

/* ------------------------------------------------------------------ */
/* Serialization                                                       */
/* ------------------------------------------------------------------ */
typedef struct { char *p; size_t len, cap; } jbuf;

static void jb_putn(jbuf *b, const char *s, size_t n) {
    if (b->len + n + 1 > b->cap) {
        while (b->len + n + 1 > b->cap) b->cap = b->cap ? b->cap * 2 : 128;
        b->p = cwick_xrealloc(b->p, b->cap);
    }
    memcpy(b->p + b->len, s, n);
    b->len += n;
    b->p[b->len] = '\0';
}
static void jb_puts(jbuf *b, const char *s) { jb_putn(b, s, strlen(s)); }
static void jb_putc(jbuf *b, char c) { jb_putn(b, &c, 1); }

static void jb_put_string(jbuf *b, const char *s) {
    jb_putn(b, "\"", 1);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        switch (*p) {
        case '"':  jb_puts(b, "\\\""); break;
        case '\\': jb_puts(b, "\\\\"); break;
        case '\n': jb_puts(b, "\\n"); break;
        case '\r': jb_puts(b, "\\r"); break;
        case '\t': jb_puts(b, "\\t"); break;
        default:
            if (*p < 0x20) { char e[8]; snprintf(e, sizeof(e), "\\u%04x", *p); jb_puts(b, e); }
            else jb_putn(b, (const char *)p, 1);
        }
    }
    jb_putn(b, "\"", 1);
}

static void jb_put_value(jbuf *b, const json_value *v) {
    if (!v) { jb_puts(b, "null"); return; }
    switch (v->kind) {
    case JSON_NULL:   jb_puts(b, "null"); break;
    case JSON_BOOL:   jb_puts(b, v->boolean ? "true" : "false"); break;
    case JSON_NUMBER: {
        /* integers stay integers: tool arguments round-trip as written */
        if (v->number == (double)v->integer && v->integer <= 9007199254740991ll &&
            v->integer >= -9007199254740991ll) {
            char n[32];
            snprintf(n, sizeof(n), "%lld", (long long)v->integer);
            jb_puts(b, n);
        } else {
            /* Shortest round-trip, not %.17g: that is exact but turns 0.42
             * into 0.41999999999999998, and a stored payload is read back by
             * the app that wrote it (the replay log shows these numbers).
             * 15 digits almost always suffices; stop as soon as the text
             * parses to the same double and fall through to 17 if not. */
            char n[64];
            int prec = 15;
            for (; prec <= 17; prec++) {
                snprintf(n, sizeof(n), "%.*g", prec, v->number);
                if (strtod(n, NULL) == v->number) break;
            }
            jb_puts(b, n);
        }
        break;
    }
    case JSON_STRING: jb_put_string(b, v->string ? v->string : ""); break;
    case JSON_ARRAY:
        jb_putc(b, '[');
        for (size_t i = 0; i < v->count; i++) {
            if (i) jb_putc(b, ',');
            jb_put_value(b, v->items[i]);
        }
        jb_putc(b, ']');
        break;
    case JSON_OBJECT:
        jb_putc(b, '{');
        for (size_t i = 0; i < v->count; i++) {
            if (i) jb_putc(b, ',');
            jb_put_string(b, v->members[i].key);
            jb_putc(b, ':');
            jb_put_value(b, v->members[i].value);
        }
        jb_putc(b, '}');
        break;
    }
}

char *cwick_json_write(const json_value *v) {
    jbuf b = {0};
    jb_put_value(&b, v);
    if (!b.p) b.p = cwick_strdup("");
    return b.p;
}
