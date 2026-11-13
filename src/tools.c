/* tools.c — see tools.h.
 *
 * The wire formats come straight from the checkpoint we serve, not from a
 * guess: `zai-org/GLM-5.3-Flash/chat_template.jinja` renders tool decls into a
 * `<tools>` block and expects `<tool_call>` answers with `<arg_key>` /
 * `<arg_value>` pairs. Mirroring that exactly is what makes tool calling work
 * with the released weights instead of merely looking like it should.
 */
#include "tools.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

#include "util.h"

/* ------------------------------------------------------------------ */
/* tiny string builder                                                 */
/* ------------------------------------------------------------------ */
typedef struct strbuf {
    char *p;
    size_t len, cap;
} strbuf;

static void sb_init(strbuf *b) {
    b->cap = 256;
    b->len = 0;
    b->p = cwick_xmalloc(b->cap);
    b->p[0] = '\0';
}

static void sb_need(strbuf *b, size_t extra) {
    if (b->len + extra + 1 <= b->cap) return;
    while (b->len + extra + 1 > b->cap) b->cap *= 2;
    b->p = cwick_xrealloc(b->p, b->cap);
}

static void sb_putn(strbuf *b, const char *s, size_t n) {
    sb_need(b, n);
    memcpy(b->p + b->len, s, n);
    b->len += n;
    b->p[b->len] = '\0';
}

static void sb_put(strbuf *b, const char *s) {
    sb_putn(b, s, strlen(s));
}

static void sb_putc(strbuf *b, char c) {
    sb_need(b, 1);
    b->p[b->len++] = c;
    b->p[b->len] = '\0';
}

/* Deterministic release: the caller owns b->p after sb_take(). */
static char *sb_take(strbuf *b) {
    char *p = b->p;
    b->p = NULL;
    b->len = b->cap = 0;
    return p;
}

/* ------------------------------------------------------------------ */
/* JSON writing                                                        */
/* ------------------------------------------------------------------ */
/* Escape into a JSON string literal.
 *
 * Bytes >= 0x80 are passed through only as part of a well-formed UTF-8
 * sequence. The detokenizer appends raw token bytes, and a byte-fallback token
 * can be half a multi-byte character: a lone 0xA8 written into the document
 * makes the whole response unparseable for any client that reads it as UTF-8,
 * which is what JSON is defined to be. An ill-formed byte becomes U+FFFD (the
 * standard replacement) instead. Callers that stream already hold back an
 * incomplete tail (see utf8_incomplete_tail in server.c), so a character is
 * never split across two deltas and this never mangles one. */
char *cwick_json_stringify(const char *s) {
    strbuf b;
    sb_init(&b);
    sb_putc(&b, '"');
    const unsigned char *p = (const unsigned char *)s;
    while (*p) {
        unsigned char c = *p;
        if (c < 0x80) {
            switch (c) {
                case '"':  sb_put(&b, "\\\""); break;
                case '\\': sb_put(&b, "\\\\"); break;
                case '\n': sb_put(&b, "\\n"); break;
                case '\r': sb_put(&b, "\\r"); break;
                case '\t': sb_put(&b, "\\t"); break;
                default:
                    if (c < 0x20) {
                        char hex[8];
                        snprintf(hex, sizeof(hex), "\\u%04x", c);
                        sb_put(&b, hex);
                    } else {
                        sb_putc(&b, (char)c);
                    }
            }
            p++;
            continue;
        }

        /* Lead byte: how many continuation bytes must follow, and the bits
         * they contribute. 0x80-0xBF (a stray continuation) and 0xF8-0xFF
         * (beyond U+10FFFF) are not lead bytes at all. */
        int need;
        uint32_t cp;
        if ((c & 0xE0) == 0xC0)      { need = 1; cp = c & 0x1Fu; }
        else if ((c & 0xF0) == 0xE0) { need = 2; cp = c & 0x0Fu; }
        else if ((c & 0xF8) == 0xF0) { need = 3; cp = c & 0x07u; }
        else { need = 0; cp = 0; }   /* not a lead byte: one bad byte */

        bool ok = need > 0;
        for (int i = 1; i <= need; i++) {
            unsigned char cc = p[i];          /* '\0' ends the string: short */
            if ((cc & 0xC0) != 0x80) { ok = false; break; }
            cp = (cp << 6) | (cc & 0x3Fu);
        }
        if (ok) {
            if (need == 1 && cp < 0x80) ok = false;         /* overlong */
            else if (need == 2 && cp < 0x800) ok = false;   /* overlong */
            else if (need == 3 && cp < 0x10000) ok = false; /* overlong */
            else if (cp > 0x10FFFF) ok = false;
            else if (cp >= 0xD800 && cp <= 0xDFFF) ok = false;  /* surrogate */
        }
        if (ok) {
            for (int i = 0; i <= need; i++) sb_putc(&b, (char)p[i]);
            p += need + 1;
        } else {
            /* One U+FFFD for the whole maximal subpart of the ill-formed
             * sequence — the Unicode recommendation, and what Python's
             * errors="replace" does — so a truncated "end\xE4\xB8" costs one
             * replacement character, not two. */
            int bad = 1;
            while (bad <= need && (p[bad] & 0xC0) == 0x80) bad++;
            sb_put(&b, "\xEF\xBF\xBD");
            p += bad;
        }
    }
    sb_putc(&b, '"');
    return sb_take(&b);
}

static void ser_json(const json_value *v, strbuf *b) {
    if (!v) { sb_put(b, "null"); return; }
    switch (v->kind) {
        case JSON_NULL:  sb_put(b, "null"); break;
        case JSON_BOOL:  sb_put(b, v->boolean ? "true" : "false"); break;
        case JSON_NUMBER: {
            /* preserve integer-ness; cwick's parser records both views */
            if (cwick_json_int((json_value *)v, 0) == 0 || v->number != (double)v->integer) {
                char num[64];
                snprintf(num, sizeof(num), "%.17g", v->number);
                sb_put(b, num);
            } else {
                char num[64];
                snprintf(num, sizeof(num), "%lld", (long long)v->integer);
                sb_put(b, num);
            }
            break;
        }
        case JSON_STRING: {
            char *esc = cwick_json_stringify(v->string ? v->string : "");
            sb_put(b, esc);
            free(esc);
            break;
        }
        case JSON_ARRAY:
            sb_putc(b, '[');
            for (size_t i = 0; i < v->count; i++) {
                if (i) sb_putc(b, ',');
                ser_json(v->items[i], b);
            }
            sb_putc(b, ']');
            break;
        case JSON_OBJECT:
            sb_putc(b, '{');
            for (size_t i = 0; i < v->count; i++) {
                if (i) sb_putc(b, ',');
                char *key = cwick_json_stringify(v->members[i].key);
                sb_put(b, key);
                sb_putc(b, ':');
                free(key);
                ser_json(v->members[i].value, b);
            }
            sb_putc(b, '}');
            break;
    }
}

/* The jinja macro drops `defer_loading` and `strict` (they are transport
 * hints, not part of the signature the model is shown). */
static void ser_function(const json_value *fn, strbuf *b) {
    sb_putc(b, '{');
    bool first = true;
    for (size_t i = 0; i < fn->count; i++) {
        const char *k = fn->members[i].key;
        if (strcmp(k, "defer_loading") == 0 || strcmp(k, "strict") == 0) continue;
        if (!first) sb_putc(b, ',');
        first = false;
        char *key = cwick_json_stringify(k);
        sb_put(b, key);
        sb_putc(b, ':');
        free(key);
        ser_json(fn->members[i].value, b);
    }
    sb_putc(b, '}');
}

/* ------------------------------------------------------------------ */
/* Request side                                                        */
/* ------------------------------------------------------------------ */
int cwick_tools_parse_request(const json_value *root, cwick_tool_set *out,
                             char *errbuf, size_t errcap) {
    memset(out, 0, sizeof(*out));
    snprintf(out->choice, sizeof(out->choice), "auto");

    json_value *tools = cwick_json_get((json_value *)root, "tools");
    if (!tools || tools->kind == JSON_NULL) {
        /* no tools: tool_choice must not ask for one */
        json_value *tc = cwick_json_get((json_value *)root, "tool_choice");
        if (tc && tc->kind != JSON_NULL) {
            snprintf(errbuf, errcap, "tool_choice requires tools");
            return 400;
        }
        return 0;
    }
    if (tools->kind != JSON_ARRAY) {
        snprintf(errbuf, errcap, "`tools` must be an array");
        return 400;
    }
    size_t n = cwick_json_array_len(tools);
    if (n == 0) return 0;                 /* empty array: nothing declared */
    if (n > 64) {
        snprintf(errbuf, errcap, "at most 64 tools per request (got %zu)", n);
        return 400;
    }

    out->present = true;
    out->items = cwick_xmalloc(n * sizeof(*out->items));
    memset(out->items, 0, n * sizeof(*out->items));
    out->count = n;

    for (size_t i = 0; i < n; i++) {
        json_value *t = cwick_json_array_at(tools, i);
        if (!t || t->kind != JSON_OBJECT) {
            snprintf(errbuf, errcap, "tools[%zu] must be an object", i);
            return 400;
        }
        json_value *ty = cwick_json_get(t, "type");
        const char *tys = ty && ty->kind == JSON_STRING ? ty->string : NULL;
        if (!tys || strcmp(tys, "function") != 0) {
            snprintf(errbuf, errcap,
                     "tools[%zu]: only type \"function\" is implemented", i);
            return 400;
        }
        json_value *fn = cwick_json_get(t, "function");
        if (!fn || fn->kind != JSON_OBJECT) {
            snprintf(errbuf, errcap, "tools[%zu] needs a function object", i);
            return 400;
        }
        json_value *nm = cwick_json_get(fn, "name");
        if (!nm || nm->kind != JSON_STRING || nm->string[0] == '\0') {
            snprintf(errbuf, errcap, "tools[%zu] needs a name", i);
            return 400;
        }
        if (strlen(nm->string) >= sizeof(out->items[i].name)) {
            snprintf(errbuf, errcap, "tools[%zu] name is too long", i);
            return 400;
        }
        snprintf(out->items[i].name, sizeof(out->items[i].name), "%s", nm->string);
        strbuf line;
        sb_init(&line);
        ser_function(fn, &line);
        out->items[i].json = sb_take(&line);
    }

    /* tool_choice */
    json_value *tc = cwick_json_get((json_value *)root, "tool_choice");
    if (tc && tc->kind != JSON_NULL) {
        if (tc->kind == JSON_STRING) {
            const char *c = tc->string;
            if (strcmp(c, "auto") == 0) {
                snprintf(out->choice, sizeof(out->choice), "auto");
            } else if (strcmp(c, "none") == 0) {
                snprintf(out->choice, sizeof(out->choice), "none");
            } else if (strcmp(c, "required") == 0) {
                snprintf(errbuf, errcap,
                         "tool_choice \"required\" is not implemented "
                         "(the model decides when to call)");
                return 400;
            } else {
                snprintf(errbuf, errcap, "unknown tool_choice `%s`", c);
                return 400;
            }
        } else if (tc->kind == JSON_OBJECT) {
            json_value *ty = cwick_json_get(tc, "type");
            const char *tys = ty && ty->kind == JSON_STRING ? ty->string : NULL;
            json_value *fnv = cwick_json_get(tc, "function");
            if (!tys || strcmp(tys, "function") != 0 || !fnv ||
                fnv->kind != JSON_OBJECT) {
                snprintf(errbuf, errcap,
                         "tool_choice object must be {\"type\":\"function\","
                         "\"function\":{\"name\":…}}");
                return 400;
            }
            json_value *nmj = cwick_json_get(fnv, "name");
            if (!nmj || nmj->kind != JSON_STRING) {
                snprintf(errbuf, errcap, "tool_choice needs a function name");
                return 400;
            }
            bool known = false;
            for (size_t i = 0; i < out->count; i++)
                if (strcmp(out->items[i].name, nmj->string) == 0) known = true;
            if (!known) {
                snprintf(errbuf, errcap,
                         "tool_choice names `%s`, which is not in `tools`",
                         nmj->string);
                return 400;
            }
            snprintf(out->choice, sizeof(out->choice), "named");
            snprintf(out->forced, sizeof(out->forced), "%s", nmj->string);
        } else {
            snprintf(errbuf, errcap, "tool_choice must be a string or object");
            return 400;
        }
    }

    out->advertise = strcmp(out->choice, "none") != 0;
    return 0;
}

void cwick_tools_free_set(cwick_tool_set *set) {
    if (!set) return;
    for (size_t i = 0; i < set->count; i++) free(set->items[i].json);
    free(set->items);
    memset(set, 0, sizeof(*set));
}

char *cwick_tools_render_block(const cwick_tool_set *set) {
    if (!set || !set->present || !set->advertise || set->count == 0) return NULL;
    strbuf b;
    sb_init(&b);
    /* Byte-for-byte the section chat_template.jinja splices in when `tools`
     * is set: it starts with the <|system|> turn marker and ends with the
     * answer format the checkpoint was trained to emit. The blank lines
     * around each tool are what the template's whitespace control leaves
     * behind, and the model has seen exactly that shape. */
    sb_put(&b, "<|system|>\n# Tools\n\nYou may call one or more functions to "
               "assist with the user query.\n\nYou are provided with function "
               "signatures within <tools></tools> XML tags:\n<tools>\n");
    for (size_t i = 0; i < set->count; i++) {
        sb_putc(&b, '\n');
        sb_put(&b, set->items[i].json);
        sb_put(&b, "\n\n");
    }
    sb_put(&b, "\n</tools>\n\nFor each function call, output the function name "
               "and arguments within the following XML format:\n"
               "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key>"
               "<arg_value>{arg-value-1}</arg_value>"
               "<arg_key>{arg-key-2}</arg_key>"
               "<arg_value>{arg-value-2}</arg_value>...</tool_call>");
    return sb_take(&b);
}

/* ------------------------------------------------------------------ */
/* Completion side                                                     */
/* ------------------------------------------------------------------ */
#define CWICK_TOOL_CALL_OPEN  "<tool_call>"
#define CWICK_TOOL_CALL_CLOSE "</tool_call>"
#define CWICK_ARG_KEY_OPEN    "<arg_key>"
#define CWICK_ARG_KEY_CLOSE   "</arg_key>"
#define CWICK_ARG_VAL_OPEN    "<arg_value>"
#define CWICK_ARG_VAL_CLOSE   "</arg_value>"

static int seq; /* call id suffix */

/* Slice out the text between two markers, or NULL when absent. */
static char *slice_until(const char **p, const char *close, size_t clen,
                         int *malformed) {
    const char *end = strstr(*p, close);
    if (!end) { *malformed = 1; return NULL; }
    size_t len = (size_t)(end - *p);
    char *out = cwick_xmalloc(len + 1);
    memcpy(out, *p, len);
    out[len] = '\0';
    *p = end + clen;
    return out;
}

/* Normalize a raw <arg_value> into a JSON fragment: if the model produced
 * JSON, re-emit it compactly; otherwise treat it as a JSON string. */
static char *normalize_value(const char *raw) {
    cwick_error e;
    cwick_error_clear(&e);
    json_value *v = cwick_json_parse(raw, strlen(raw), &e);
    if (v) {
        strbuf b;
        sb_init(&b);
        ser_json(v, &b);
        cwick_json_free(v);
        return sb_take(&b);
    }
    char *esc = cwick_json_stringify(raw);
    return esc;
}

/* MiniMax-M3.1 writes tool calls as XML inside the same <tool_call> wrapper:
 *
 *   <tool_call>
 *   <invoke name="add_numbers"><a>137</a><b>258</b></invoke>
 *   </tool_call>
 *
 * with the checkpoint's namespace token (`]<]minimax[>[`) prefixed to every
 * structural tag -- that token is in added_tokens.json and the template splices
 * it in as `ns_token ~ '<invoke name=...>'`, so the model is following the
 * spec, not drifting.
 *
 * Rather than teach the loop below a second grammar, this rewrites the block
 * into the <arg_key>/<arg_value> form it already parses. One parser, one set of
 * edge cases. Nested objects are the known limit: a child that itself contains
 * elements is passed through as a string, which is what the flat tools here
 * need; a recursive renderer would have to come with a recursive parser. */
static const char MM_NS_TOKEN[] = "]<]minimax[>[";

static void strip_ns(const char **src, const char *end, strbuf *b) {
    const char *p = *src;
    while (p < end) {
        if (end - p >= (long)sizeof(MM_NS_TOKEN) - 1 &&
            memcmp(p, MM_NS_TOKEN, sizeof(MM_NS_TOKEN) - 1) == 0) {
            p += sizeof(MM_NS_TOKEN) - 1;
            continue;
        }
        sb_putc(b, *p++);
    }
    *src = p;
}

/* The same token, seen from the streaming side. A tool block arrives one
 * detokenized piece at a time, so before forwarding a slice we ask how much of
 * its tail could still grow into the namespace token: that much is held back
 * until the next piece lands. Holding the partial prefix (rather than a fixed
 * window) is what keeps the progress live — plain bytes end in no token
 * prefix, so almost every piece goes out immediately. */
size_t cwick_tool_ns_tail(const char *s, size_t n) {
    const size_t tlen = sizeof(MM_NS_TOKEN) - 1;
    size_t maxk = n < tlen - 1 ? n : tlen - 1;
    for (size_t k = maxk; k > 0; k--)
        if (memcmp(s + n - k, MM_NS_TOKEN, k) == 0) return k;
    return 0;
}

/* A copy of [s, s+n) with every complete namespace token removed, for the
 * tool-progress frames. Caller frees. */
char *cwick_tool_ns_strip(const char *s, size_t n) {
    strbuf b;
    sb_init(&b);
    const char *p = s;
    strip_ns(&p, s + n, &b);
    return sb_take(&b);
}

static char *translate_invoke(const char *text) {
    static const char open[] = "<tool_call>", close[] = "</tool_call>";
    static const char inv[] = "<invoke name=\"";
    /* The namespace token has to go whether or not the block is an <invoke>:
     * a checkpoint can also write the <arg_key>/<arg_value> form with the
     * token in front of every tag, and leaving it in makes the block look
     * unterminated. */
    if (!strstr(text, MM_NS_TOKEN) && !strstr(text, inv)) return NULL;

    strbuf b;
    sb_init(&b);
    const char *p = text;
    for (;;) {
        const char *tc = strstr(p, open);
        if (!tc) { strip_ns(&p, text + strlen(text), &b); break; }
        const char *body_end = strstr(tc + sizeof(open) - 1, close);
        strip_ns(&p, tc, &b);                      /* prose before the block */
        if (!body_end) break;                      /* unterminated: leave it */
        const char *q = tc + sizeof(open) - 1;
        int emitted = 0;
        while (1) {
            const char *i0 = strstr(q, inv);
            if (!i0 || i0 >= body_end) break;
            const char *nm = i0 + sizeof(inv) - 1;
            const char *nm_end = strchr(nm, '"');
            if (!nm_end || nm_end >= body_end) break;
            sb_put(&b, open);                      /* one block per <invoke> */
            strip_ns(&nm, nm_end, &b);
            q = nm_end + 1;
            while (q < body_end && *q != '>') q++;  /* rest of the open tag */
            if (q < body_end) q++;
            /* children: <tag>value</tag>, until </invoke> or the block ends */
            while (1) {
                while (q < body_end && (*q == ' ' || *q == '\n' || *q == '\r')) q++;
                while (body_end - q >= (long)sizeof(MM_NS_TOKEN) - 1 &&
                       memcmp(q, MM_NS_TOKEN, sizeof(MM_NS_TOKEN) - 1) == 0)
                    q += sizeof(MM_NS_TOKEN) - 1;   /* framing before the tag */
                if (q >= body_end) break;
                if (strncmp(q, "</invoke>", 9) == 0) { q += 9; break; }
                if (*q != '<') break;
                const char *tag = ++q;
                while (q < body_end && *q != '>') q++;
                if (q >= body_end) break;
                char tag_name[64];
                size_t tl = (size_t)(q - tag);
                if (tl == 0 || tl >= sizeof(tag_name)) break;
                memcpy(tag_name, tag, tl);
                tag_name[tl] = '\0';
                q++;                                   /* past '>' */
                const char *v0 = q;
                while (q < body_end && *q != '<') q++;
                sb_put(&b, "<arg_key>");
                sb_put(&b, tag_name);
                sb_put(&b, "</arg_key><arg_value>");
                const char *vp = v0;
                strip_ns(&vp, q, &b);
                sb_put(&b, "</arg_value>");
                /* skip the child's closing tag */
                if (q < body_end && *q == '<' && q + 1 < body_end && q[1] == '/') {
                    while (q < body_end && *q != '>') q++;
                    if (q < body_end) q++;
                }
            }
            sb_put(&b, close);
            emitted = 1;
        }
        if (!emitted) {          /* an <invoke>-free block: keep it as it was */
            sb_put(&b, open);
            strip_ns(&q, body_end, &b);
            sb_put(&b, close);
        }
        p = body_end + sizeof(close) - 1;
    }
    /* trailing text after the last block */
    strip_ns(&p, text + strlen(text), &b);
    return sb_take(&b);
}

/* Every complete namespace token gone, caller frees. Used on the whole answer
 * before any parsing (see cwick_tools_extract) and on the slices the server
 * forwards as tool progress. */
static char *strip_ns_all(const char *text) {
    strbuf b;
    sb_init(&b);
    const char *p = text;
    strip_ns(&p, text + strlen(text), &b);
    return sb_take(&b);
}

/* A second checkpoint keeps the <tool_call> wrapper but names the function and
 * its arguments with <function=NAME> / <parameter=KEY>VALUE</parameter>:
 *
 *   <tool_call>
 *   <function=write_file>
 *   <parameter=path>
 *   a/b.html
 *   </parameter>
 *   </function>
 *   </tool_call>
 *
 * Rewrite that into the flat <arg_key>/<arg_value> grammar extract_impl already
 * reads, so there is still one parser and one set of edge cases. NULL means
 * "not this dialect" — including a half-written block, which must reach the
 * extractor unchanged so it can refuse the turn instead of silently dropping
 * part of a call. */
static char *translate_function(const char *text) {
    static const char f_open[]  = "<function=";
    static const char f_close[] = "</function>";
    static const char p_open[]  = "<parameter=";
    static const char p_close[] = "</parameter>";

    if (!strstr(text, f_open)) return NULL;
    /* A text that already uses the flat grammar belongs to the other dialect;
     * do not splice the two together. */
    if (strstr(text, CWICK_ARG_KEY_OPEN)) return NULL;

    const char *probe = text;
    while ((probe = strstr(probe, f_open)) != NULL) {
        const char *e = strstr(probe, f_close);
        if (!e) return NULL;                 /* fail-closed */
        probe = e + sizeof(f_close) - 1;
    }

    strbuf b;
    sb_init(&b);
    const char *p = text;
    while (*p) {
        if (strncmp(p, CWICK_TOOL_CALL_OPEN, strlen(CWICK_TOOL_CALL_OPEN)) == 0) {
            p += strlen(CWICK_TOOL_CALL_OPEN);     /* outer framing */
            continue;
        }
        if (strncmp(p, CWICK_TOOL_CALL_CLOSE, strlen(CWICK_TOOL_CALL_CLOSE)) == 0) {
            p += strlen(CWICK_TOOL_CALL_CLOSE);
            continue;
        }
        if (strncmp(p, f_open, sizeof(f_open) - 1) != 0) {
            sb_putc(&b, *p++);                    /* prose before/after calls */
            continue;
        }
        const char *nm = p + sizeof(f_open) - 1;
        const char *nm_end = strchr(nm, '>');
        const char *body_end = strstr(nm, f_close);
        if (!nm_end || !body_end || nm_end >= body_end) {
            free(b.p);
            return NULL;
        }
        sb_put(&b, CWICK_TOOL_CALL_OPEN);
        sb_putn(&b, nm, (size_t)(nm_end - nm));
        const char *q = nm_end + 1;
        while (q < body_end) {
            if (strncmp(q, p_open, sizeof(p_open) - 1) != 0) { q++; continue; }
            const char *key = q + sizeof(p_open) - 1;
            const char *key_end = strchr(key, '>');
            const char *val_end = strstr(key, p_close);
            if (!key_end || key_end >= body_end) { q++; continue; }
            if (val_end && val_end > body_end) val_end = NULL;
            const char *v0 = key_end + 1;
            const char *v1 = val_end ? val_end : body_end;
            /* The template puts newlines around a value; they are framing,
             * not part of the parameter. Inner newlines are kept. */
            while (v0 < v1 && (*v0 == '\n' || *v0 == '\r' ||
                               *v0 == ' ' || *v0 == '\t')) v0++;
            while (v1 > v0 && (v1[-1] == '\n' || v1[-1] == '\r' ||
                               v1[-1] == ' ' || v1[-1] == '\t')) v1--;
            sb_put(&b, CWICK_ARG_KEY_OPEN);
            sb_putn(&b, key, (size_t)(key_end - key));
            sb_put(&b, CWICK_ARG_KEY_CLOSE);
            sb_put(&b, CWICK_ARG_VAL_OPEN);
            sb_putn(&b, v0, (size_t)(v1 - v0));
            sb_put(&b, CWICK_ARG_VAL_CLOSE);
            q = val_end ? val_end + sizeof(p_close) - 1 : body_end;
        }
        sb_put(&b, CWICK_TOOL_CALL_CLOSE);
        p = body_end + sizeof(f_close) - 1;
    }
    return sb_take(&b);
}

static int extract_impl(const char *text, cwick_tool_call **calls, size_t *n,
                        char **visible, cwick_error *err);

int cwick_tools_extract(const char *text, cwick_tool_call **calls, size_t *n,
                        char **visible, cwick_error *err) {
    if (!text) {
        *calls = NULL;
        *n = 0;
        if (visible) *visible = NULL;
        return 0;
    }
    /* Framing first, and from the whole answer rather than from the places a
     * grammar thinks a tag can be: the token has been observed in front of a
     * tag, between two tags, and inside a value (a value that ended in `137]`
     * with `]minimax[` parsed as the next argument's key). Removing it before
     * anything looks at the text is one rule instead of a rule per position. */
    char *clean = strstr(text, MM_NS_TOKEN) ? strip_ns_all(text) : NULL;
    const char *base = clean ? clean : text;

    char *tr = translate_invoke(base);
    const char *after_invoke = tr ? tr : base;
    char *tf = translate_function(after_invoke);
    int rc = extract_impl(tf ? tf : after_invoke, calls, n, visible, err);
    free(tf);
    free(tr);
    free(clean);
    return rc;
}

static int extract_impl(const char *text, cwick_tool_call **calls, size_t *n,
                       char **visible, cwick_error *err) {
    *calls = NULL;
    *n = 0;
    if (visible) *visible = NULL;
    if (!text) return 0;

    const char *first = strstr(text, CWICK_TOOL_CALL_OPEN);
    if (!first) {
        if (visible) *visible = cwick_strdup(text);
        return 0;
    }

    /* visible prose is whatever precedes the first tool call */
    if (visible) {
        size_t len = (size_t)(first - text);
        if (len > 0) {
            char *v = cwick_xmalloc(len + 1);
            memcpy(v, text, len);
            v[len] = '\0';
            *visible = v;
        }
    }

    size_t cap = 4, count = 0;
    cwick_tool_call *list = cwick_xmalloc(cap * sizeof(*list));
    memset(list, 0, cap * sizeof(*list));

    const char *p = text;
    while ((p = strstr(p, CWICK_TOOL_CALL_OPEN)) != NULL) {
        p += strlen(CWICK_TOOL_CALL_OPEN);
        /* The function name runs from here to the first <arg_key>, or — for a
         * zero-argument call — straight to </tool_call>. */
        const char *next_key = strstr(p, CWICK_ARG_KEY_OPEN);
        const char *close = strstr(p, CWICK_TOOL_CALL_CLOSE);
        const char *stop;
        size_t name_len;
        if (!close) {
            cwick_error_set(err, CWICK_ERR_PARSE,
                           "unterminated <tool_call> in model output");
            goto fail;
        }
        if (next_key && next_key < close) { stop = next_key; }
        else                              { stop = close; }
        name_len = (size_t)(stop - p);
        char *name = cwick_xmalloc(name_len + 1);
        memcpy(name, p, name_len);
        name[name_len] = '\0';
        p = stop;

        /* trim whitespace the model may have put around the name */
        {
            size_t l = strlen(name);
            while (l > 0 && (name[l - 1] == ' ' || name[l - 1] == '\n' ||
                             name[l - 1] == '\r' || name[l - 1] == '\t'))
                name[--l] = '\0';
            size_t skip = 0;
            while (name[skip] == ' ' || name[skip] == '\n' ||
                   name[skip] == '\r' || name[skip] == '\t') skip++;
            if (skip) memmove(name, name + skip, strlen(name + skip) + 1);
        }
        if (!*name) {
            free(name);
            cwick_error_set(err, CWICK_ERR_PARSE,
                           "<tool_call> without a function name");
            goto fail;
        }

        if (count == cap) {
            cap *= 2;
            list = cwick_xrealloc(list, cap * sizeof(*list));
            memset(list + count, 0, (cap - count) * sizeof(*list));
        }
        cwick_tool_call *c = &list[count];
        snprintf(c->name, sizeof(c->name), "%s", name);
        snprintf(c->id, sizeof(c->id), "call_%ld_%d", (long)getpid() % 100000, ++seq);
        free(name);

        size_t acap = 4;
        c->args = cwick_xmalloc(acap * sizeof(*c->args));
        memset(c->args, 0, acap * sizeof(*c->args));

        /* arg pairs until </tool_call> */
        while (strncmp(p, CWICK_TOOL_CALL_CLOSE, strlen(CWICK_TOOL_CALL_CLOSE)) != 0 && *p) {
            int bad = 0;
            /* p sits on "<arg_key>k</arg_key><arg_value>v</arg_value>":
             * step over the opening tag before slicing the key out. */
            if (strncmp(p, CWICK_ARG_KEY_OPEN, strlen(CWICK_ARG_KEY_OPEN)) == 0)
                p += strlen(CWICK_ARG_KEY_OPEN);
            char *key = slice_until(&p, CWICK_ARG_KEY_CLOSE,
                                    strlen(CWICK_ARG_KEY_CLOSE), &bad);
            if (!key) break;
            if (strncmp(p, CWICK_ARG_VAL_OPEN, strlen(CWICK_ARG_VAL_OPEN)) != 0) {
                free(key);
                break;
            }
            p += strlen(CWICK_ARG_VAL_OPEN);
            char *raw = slice_until(&p, CWICK_ARG_VAL_CLOSE,
                                    strlen(CWICK_ARG_VAL_CLOSE), &bad);
            if (!raw) {
                free(key);
                if (bad) {
                    cwick_error_set(err, CWICK_ERR_PARSE,
                                   "unterminated <arg_value> for `%s`",
                                   c->name);
                    goto fail;
                }
                break;
            }
            if (c->n_args == acap) {
                acap *= 2;
                c->args = cwick_xrealloc(c->args, acap * sizeof(*c->args));
            }
            c->args[c->n_args].key = key;
            c->args[c->n_args].value_json = normalize_value(raw);
            c->n_args++;
            free(raw);
        }
        if (strncmp(p, CWICK_TOOL_CALL_CLOSE, strlen(CWICK_TOOL_CALL_CLOSE)) != 0) {
            cwick_error_set(err, CWICK_ERR_PARSE,
                           "unterminated <tool_call> for `%s`", c->name);
            goto fail;
        }
        p += strlen(CWICK_TOOL_CALL_CLOSE);

        /* {"k":v,…} */
        strbuf js;
        sb_init(&js);
        sb_putc(&js, '{');
        for (size_t i = 0; i < c->n_args; i++) {
            if (i) sb_putc(&js, ',');
            char *k = cwick_json_stringify(c->args[i].key);
            sb_put(&js, k);
            sb_putc(&js, ':');
            free(k);
            sb_put(&js, c->args[i].value_json);
        }
        sb_putc(&js, '}');
        c->arguments_json = sb_take(&js);
        count++;
    }

    *calls = list;
    *n = count;
    return 0;

fail:
    /* every slot up to cap may hold partially built state */
    for (size_t i = 0; i < cap; i++) {
        free(list[i].arguments_json);
        for (size_t j = 0; j < list[i].n_args; j++) {
            free(list[i].args[j].key);
            free(list[i].args[j].value_json);
        }
        free(list[i].args);
    }
    free(list);
    if (visible && *visible) { free(*visible); *visible = NULL; }
    return -1;
}

void cwick_tool_calls_free(cwick_tool_call *calls, size_t n) {
    if (!calls) return;
    for (size_t i = 0; i < n; i++) {
        free(calls[i].arguments_json);
        for (size_t j = 0; j < calls[i].n_args; j++) {
            free(calls[i].args[j].key);
            free(calls[i].args[j].value_json);
        }
        free(calls[i].args);
    }
    free(calls);
}

cwick_decl_match cwick_tool_decl_match(cwick_tool_call *call,
                                       const cwick_tool_set *set) {
    if (!call || !set || !set->present) return CWICK_DECL_UNKNOWN;
    for (size_t j = 0; j < set->count; j++)
        if (strcmp(set->items[j].name, call->name) == 0)
            return CWICK_DECL_EXACT;
    /* The model copies tool names out of the <tools> block but does not
     * always keep their case: a declared `Bash` has come back as `bash`,
     * and the OpenAI contract is the exact declared name, so a call that
     * matches a declaration up to case is rewritten to it. The first
     * declaration wins if several differ only by case, which is already
     * a pathological declaration set. */
    for (size_t j = 0; j < set->count; j++) {
        if (strcasecmp(set->items[j].name, call->name) == 0) {
            snprintf(call->name, sizeof(call->name), "%s", set->items[j].name);
            return CWICK_DECL_FOLDED;
        }
    }
    return CWICK_DECL_UNKNOWN;
}
