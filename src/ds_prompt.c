/*
 * ds_prompt.c — DeepSeek-V4.1 prompt encoding in C.
 *
 * The reference splits the format across `encoding/encoding.py` (the renderer)
 * and `encoding/README.md` (the specification). This file is a port of the
 * text path of that renderer, keeping the same function boundaries so the two
 * can be compared side by side:
 *
 *   render_message      one message -> its bytes, plus the transition tokens
 *                       that depend on what follows
 *   merge_tool_messages `role: tool` turns are not a thing here; each one
 *                       becomes a <tool_result> block inside a user turn
 *   render_tools        the schema block spliced into the system turn
 *   encode              BOS, the drop-thinking pass, and the loop
 *
 * Two things are deliberately literal:
 *
 *   * the DSML tag names carry a leading space (`<｜DSML｜ invoke>`), which is
 *     the V4.1 change from V4;
 *   * numbers and strings are re-serialized the way `json.dumps` does with its
 *     default separators (", " and ": "), because the tool schema block is
 *     part of the prompt and a byte difference there is a different prompt.
 */
#include "ds_prompt.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* special tokens and templates                                         */
/* ------------------------------------------------------------------ */

#define SP_BOS     "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence" "\xef\xbd\x9c>"
#define SP_EOS     "<\xef\xbd\x9c" "end\xe2\x96\x81of\xe2\x96\x81sentence" "\xef\xbd\x9c>"
#define SP_USER    "<\xef\xbd\x9c" "User" "\xef\xbd\x9c>"
#define SP_ASSIST  "<\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c>"
#define SP_SYSTEM  "<\xef\xbd\x9c" "System" "\xef\xbd\x9c>"
#define SP_REMIND  "<\xef\xbd\x9c" "latest_reminder" "\xef\xbd\x9c>"

#define T_THINK_IN  "<think>"
#define T_THINK_OUT "</think>"
#define DSML        "\xef\xbd\x9c" "DSML" "\xef\xbd\x9c"

/* the leading space is the V4.1 form */
#define DSML_CALLS  " calls"
#define DSML_INVOKE " invoke"
#define DSML_PARAM  " parameter"

#define EFFORT_FMT \
    "Reasoning Effort: %d (range 1-100, the higher the value, the more " \
    "thorough the reasoning)\n\n"

#define RESPONSE_FORMAT_HEAD \
    "## Response Format:\n\nYou MUST strictly adhere to the following schema " \
    "to reply:\n"

static const char TOOLS_TEMPLATE[] =
    "## Tools\n"
    "\n"
    "You have access to a set of tools to help answer the user's question. "
    "You can invoke tools by writing a \"<" DSML " calls>\" block like the "
    "following:\n"
    "\n"
    "<" DSML " calls>\n"
    "<" DSML " invoke name=\"$TOOL_NAME\">\n"
    "<" DSML " parameter name=\"$PARAMETER_NAME\" string=\"true|false\">"
    "$PARAMETER_VALUE</" DSML " parameter>\n"
    "...\n"
    "</" DSML " invoke>\n"
    "<" DSML " invoke name=\"$TOOL_NAME2\">\n"
    "...\n"
    "</" DSML " invoke>\n"
    "</" DSML " calls>\n"
    "\n"
    "String parameters should be specified as is and set `string=\"true\"`. "
    "For all other types (numbers, booleans, arrays, objects), pass the value "
    "in JSON format and set `string=\"false\"`.\n"
    "\n"
    "If thinking_mode is enabled (triggered by " T_THINK_IN "), you MUST "
    "output your complete reasoning inside " T_THINK_IN "..." T_THINK_OUT
    " BEFORE any tool calls or final response.\n"
    "\n"
    "Otherwise, output directly after " T_THINK_OUT " with tool calls or "
    "final response.\n"
    "\n"
    "### Available Tool Schemas\n"
    "\n";

static const char TOOLS_TAIL[] =
    "\n"
    "You MUST strictly follow the above defined tool name and parameter "
    "schemas to invoke tool calls.\n";

/* ------------------------------------------------------------------ */
/* growable byte buffer                                                 */
/* ------------------------------------------------------------------ */

typedef struct {
    char *p;
    size_t n, cap;
} buf;

static void buf_grow(buf *b, size_t extra) {
    if (b->n + extra + 1 <= b->cap) return;
    size_t cap = b->cap ? b->cap : 256;
    while (cap < b->n + extra + 1) cap *= 2;
    char *np = realloc(b->p, cap);
    if (!np) {
        fprintf(stderr, "ds_prompt: out of memory\n");
        abort();
    }
    b->p = np;
    b->cap = cap;
}

static void buf_add(buf *b, const char *s, size_t n) {
    if (!n) return;
    buf_grow(b, n);
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = '\0';
}

static void buf_puts(buf *b, const char *s) { buf_add(b, s, strlen(s)); }

static void buf_putc(buf *b, char c) { buf_add(b, &c, 1); }

static void buf_printf(buf *b, const char *fmt, ...) {
    char tmp[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if ((size_t)n < sizeof(tmp)) {
        buf_add(b, tmp, (size_t)n);
        return;
    }
    buf_grow(b, (size_t)n);
    va_start(ap, fmt);
    vsnprintf(b->p + b->n, (size_t)n + 1, fmt, ap);
    va_end(ap);
    b->n += (size_t)n;
}

/* ------------------------------------------------------------------ */
/* json.dumps-compatible writer                                         */
/* ------------------------------------------------------------------ */

/* Python's json.dumps(float) uses repr(), which prints the shortest
 * round-tripping form. %.17g over-prints; trim back to the shortest form that
 * still reads back as the same double. */
static void put_double(buf *b, double v) {
    if (v != v) { buf_puts(b, "NaN"); return; }
    if (v > 1.7976931348623157e308) { buf_puts(b, "Infinity"); return; }
    if (v < -1.7976931348623157e308) { buf_puts(b, "-Infinity"); return; }
    char tmp[40];
    for (int prec = 1; prec <= 17; prec++) {
        snprintf(tmp, sizeof(tmp), "%.*g", prec, v);
        if (strtod(tmp, NULL) == v) break;
    }
    /* Python keeps a ".0" on integral floats; JSON input never tells us
     * whether the token was written as 1 or 1.0, so the integer path in
     * put_number is what normally handles those. */
    buf_puts(b, tmp);
}

static void put_number(buf *b, const json_value *v) {
    double d = v->number;
    if (d == (double)v->integer && d >= -9007199254740992.0 &&
        d <= 9007199254740992.0) {
        buf_printf(b, "%lld", (long long)v->integer);
        return;
    }
    put_double(b, d);
}

static void put_string(buf *b, const char *s) {
    buf_putc(b, '"');
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        switch (*p) {
        case '"':  buf_puts(b, "\\\""); break;
        case '\\': buf_puts(b, "\\\\"); break;
        case '\n': buf_puts(b, "\\n"); break;
        case '\r': buf_puts(b, "\\r"); break;
        case '\t': buf_puts(b, "\\t"); break;
        case '\b': buf_puts(b, "\\b"); break;
        case '\f': buf_puts(b, "\\f"); break;
        default:
            /* ensure_ascii=False: UTF-8 goes out unchanged, control bytes do
             * not */
            if (*p < 0x20) buf_printf(b, "\\u%04x", (unsigned)*p);
            else buf_putc(b, (char)*p);
        }
    }
    buf_putc(b, '"');
}

static void put_json(buf *b, const json_value *v) {
    if (!v) { buf_puts(b, "null"); return; }
    switch (v->kind) {
    case JSON_NULL:   buf_puts(b, "null"); break;
    case JSON_BOOL:   buf_puts(b, v->boolean ? "true" : "false"); break;
    case JSON_NUMBER: put_number(b, v); break;
    case JSON_STRING: put_string(b, v->string ? v->string : ""); break;
    case JSON_ARRAY:
        buf_putc(b, '[');
        for (size_t i = 0; i < v->count; i++) {
            if (i) buf_puts(b, ", ");
            put_json(b, v->items[i]);
        }
        buf_putc(b, ']');
        break;
    case JSON_OBJECT:
        buf_putc(b, '{');
        for (size_t i = 0; i < v->count; i++) {
            if (i) buf_puts(b, ", ");
            put_string(b, v->members[i].key ? v->members[i].key : "");
            buf_puts(b, ": ");
            put_json(b, v->members[i].value);
        }
        buf_putc(b, '}');
        break;
    }
}

/* ------------------------------------------------------------------ */
/* tool names and namespaces                                            */
/* ------------------------------------------------------------------ */

/* Split "ns::name" and validate any explicit namespace, as the reference
 * does. Returns the bare name; *ns receives the namespace or NULL. Takes
 * ownership-free views into `name`. */
static const char *split_tool_name(const char *name, const char *ns,
                                   const char **out_ns, char *nsbuf,
                                   size_t nsbuf_sz, cwick_error *err) {
    const char *sep = strstr(name, "::");
    *out_ns = ns;
    if (sep) {
        size_t plen = (size_t)(sep - name);
        if (ns && (strlen(ns) != plen || strncmp(ns, name, plen) != 0)) {
            cwick_error_set(err, CWICK_ERR_ARGUMENT,
                            "conflicting tool namespaces: %s != %.*s", ns,
                            (int)plen, name);
            return NULL;
        }
        if (plen + 1 > nsbuf_sz) {
            cwick_error_set(err, CWICK_ERR_ARGUMENT, "tool namespace too long");
            return NULL;
        }
        memcpy(nsbuf, name, plen);
        nsbuf[plen] = '\0';
        *out_ns = nsbuf;
        name = sep + 2;
    }
    if (strstr(name, "::")) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "tool name must not contain '::': %s", name);
        return NULL;
    }
    return name;
}

/* <ns>::<name> for a tool definition (`namespace` may be a string or an
 * object with a `name`). Writes into `out`. */
static int tool_name_for_encoding(const json_value *tool, char *out,
                                  size_t out_sz, cwick_error *err) {
    const json_value *fn = cwick_json_get((json_value *)tool, "function");
    if (!fn) fn = (json_value *)tool;
    const char *name = cwick_json_str(cwick_json_get((json_value *)fn, "name"));
    if (!name) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "tool is missing function.name");
        return -1;
    }
    const char *ns = NULL;
    const json_value *nsv = cwick_json_get((json_value *)tool, "namespace");
    if (!nsv) nsv = cwick_json_get((json_value *)fn, "namespace");
    if (nsv) {
        if (nsv->kind == JSON_STRING) ns = cwick_json_str((json_value *)nsv);
        else if (nsv->kind == JSON_OBJECT)
            ns = cwick_json_str(cwick_json_get((json_value *)nsv, "name"));
    }
    char nsbuf[128];
    const char *bare = split_tool_name(name, ns, &ns, nsbuf, sizeof(nsbuf), err);
    if (!bare) return -1;
    if (ns) snprintf(out, out_sz, "%s::%s", ns, bare);
    else snprintf(out, out_sz, "%s", bare);
    return 0;
}

/* The schema block, one JSON object per line. */
static int render_tools(buf *b, const json_value *tools, cwick_error *err) {
    buf_puts(b, TOOLS_TEMPLATE);
    for (size_t i = 0; i < cwick_json_array_len((json_value *)tools); i++) {
        json_value *tool = cwick_json_array_at((json_value *)tools, i);
        if (!tool || tool->kind != JSON_OBJECT) continue;
        char qname[256];
        if (tool_name_for_encoding(tool, qname, sizeof(qname), err) != 0)
            return -1;
        /* tools_from_openai_format: the emitted object is the function
         * definition with the namespace folded into its name, and a namespace
         * description prepended to the tool description. */
        json_value *fn = cwick_json_get(tool, "function");
        if (!fn) fn = tool;
        buf_putc(b, '{');
        int first = 1;
        for (size_t k = 0; k < fn->count; k++) {
            const char *key = fn->members[k].key;
            if (!key || strcmp(key, "namespace") == 0) continue;
            if (!first) buf_puts(b, ", ");
            first = 0;
            put_string(b, key);
            buf_puts(b, ": ");
            if (strcmp(key, "name") == 0) {
                put_string(b, qname);
            } else if (strcmp(key, "description") == 0) {
                const json_value *nsd = NULL;
                const json_value *nsv = cwick_json_get(tool, "namespace");
                if (!nsv) nsv = cwick_json_get(fn, "namespace");
                if (nsv && nsv->kind == JSON_OBJECT)
                    nsd = cwick_json_get((json_value *)nsv, "description");
                const char *desc = cwick_json_str(fn->members[k].value);
                if (nsd && cwick_json_str((json_value *)nsd))
                    buf_printf(b, "%s\n%s", cwick_json_str((json_value *)nsd),
                               desc ? desc : "");
                else
                    put_json(b, fn->members[k].value);
            } else {
                put_json(b, fn->members[k].value);
            }
        }
        /* a namespace with no description still needs the qualified name, and
         * the name may have been absent from the loop above only if the tool
         * used a bare `{name, ...}` shape */
        if (first) {
            put_string(b, "name");
            buf_puts(b, ": ");
            put_string(b, qname);
        }
        buf_putc(b, '}');
        buf_putc(b, '\n');
    }
    buf_puts(b, TOOLS_TAIL);
    return 0;
}

/* <｜DSML｜ parameter name="k" string="true|false">v</｜DSML｜ parameter> */
static void encode_arguments_to_dsml(buf *b, const json_value *args) {
    /* A string of JSON (possibly double-encoded) is parsed; anything else
     * non-object is wrapped as {"arguments": value}, like the reference. */
    json_value *parsed = NULL;
    const json_value *obj = args;
    if (args && args->kind != JSON_OBJECT) {
        if (args->kind == JSON_STRING) {
            for (int tries = 0; tries < 2; tries++) {
                cwick_error e;
                cwick_error_clear(&e);
                parsed = cwick_json_parse(args->string, strlen(args->string), &e);
                if (!parsed) break;
                if (parsed->kind == JSON_OBJECT) break;
                if (parsed->kind == JSON_STRING) {
                    cwick_json_free(parsed);
                    parsed = NULL;
                    break;
                }
                break;
            }
        }
        if (parsed && parsed->kind == JSON_OBJECT) obj = parsed;
    }
    if (!obj || obj->kind != JSON_OBJECT) {
        buf_puts(b, "<" DSML DSML_PARAM " name=\"arguments\" string=\"false\">");
        put_json(b, args);
        buf_puts(b, "</" DSML DSML_PARAM ">");
        cwick_json_free(parsed);
        return;
    }
    for (size_t i = 0; i < obj->count; i++) {
        if (i) buf_putc(b, '\n');
        const char *k = obj->members[i].key ? obj->members[i].key : "";
        const json_value *v = obj->members[i].value;
        int is_str = v && v->kind == JSON_STRING;
        buf_puts(b, "<" DSML DSML_PARAM " name=\"");
        buf_puts(b, k);
        buf_puts(b, "\" string=\"");
        buf_puts(b, is_str ? "true" : "false");
        buf_puts(b, "\">");
        if (is_str) buf_puts(b, v->string ? v->string : "");
        else put_json(b, v);
        buf_puts(b, "</" DSML DSML_PARAM ">");
    }
    cwick_json_free(parsed);
}

/* ------------------------------------------------------------------ */
/* message list                                                         */
/* ------------------------------------------------------------------ */

typedef enum { BLK_TEXT, BLK_TOOL_RESULT, BLK_UNSUPPORTED } blk_kind;

typedef struct {
    blk_kind kind;
    const char *text;          /* BLK_TEXT / BLK_UNSUPPORTED */
    const json_value *content; /* BLK_TOOL_RESULT, borrowed */
} blk;

typedef struct {
    const char *role;
    const char *content;
    const json_value *tools;
    const json_value *tool_calls;
    const json_value *content_blocks;
    const json_value *response_format;
    const char *reasoning_content;
    const char *task;
    bool wo_eos;
    blk *blocks;
    int n_blocks, cap_blocks;
} dp_msg;

typedef struct {
    dp_msg *m;
    int n, cap;
} dp_list;

static dp_msg *list_push(dp_list *l) {
    if (l->n == l->cap) {
        l->cap = l->cap ? l->cap * 2 : 16;
        dp_msg *nm = realloc(l->m, (size_t)l->cap * sizeof(dp_msg));
        if (!nm) { fprintf(stderr, "ds_prompt: out of memory\n"); abort(); }
        l->m = nm;
    }
    dp_msg *m = &l->m[l->n++];
    memset(m, 0, sizeof(*m));
    return m;
}

static void msg_add_block(dp_msg *m, blk b) {
    if (m->n_blocks == m->cap_blocks) {
        m->cap_blocks = m->cap_blocks ? m->cap_blocks * 2 : 4;
        blk *nb = realloc(m->blocks, (size_t)m->cap_blocks * sizeof(blk));
        if (!nb) { fprintf(stderr, "ds_prompt: out of memory\n"); abort(); }
        m->blocks = nb;
    }
    m->blocks[m->n_blocks++] = b;
}

static const json_value *const_str_array_at(const json_value *v, size_t i) {
    return cwick_json_array_at((json_value *)v, i);
}

/* The content of a user message as blocks, per merge_tool_messages: a string
 * becomes one text block, an array of OpenAI content parts becomes blocks. */
static int user_blocks_from(dp_msg *m, const json_value *msg, cwick_error *err) {
    const json_value *content = cwick_json_get((json_value *)msg, "content");
    if (!content) return 0;
    if (content->kind == JSON_STRING) {
        blk b = {BLK_TEXT, content->string ? content->string : "", NULL};
        msg_add_block(m, b);
        return 0;
    }
    if (content->kind != JSON_ARRAY) {
        blk b = {BLK_TEXT, "", NULL};
        (void)b;
        (void)err;
        return 0;
    }
    for (size_t i = 0; i < content->count; i++) {
        const json_value *part = const_str_array_at(content, i);
        if (!part || part->kind != JSON_OBJECT) continue;
        const char *type = cwick_json_str(cwick_json_get((json_value *)part, "type"));
        if (!type || strcmp(type, "text") == 0) {
            blk b = {BLK_TEXT,
                     cwick_json_str(cwick_json_get((json_value *)part, "text"))
                         ? cwick_json_str(cwick_json_get((json_value *)part, "text"))
                         : "",
                     NULL};
            msg_add_block(m, b);
        } else if (strcmp(type, "tool_result") == 0) {
            blk b = {BLK_TOOL_RESULT, NULL,
                     cwick_json_get((json_value *)part, "content")};
            msg_add_block(m, b);
        } else if (strcmp(type, "image") == 0 || strcmp(type, "image_url") == 0) {
            cwick_error_set(err, CWICK_ERR_UNSUPPORTED,
                            "deepseek_v41 prompt encoding does not support image "
                            "blocks in this runtime (the vision tower is not loaded)");
            return -1;
        } else {
            blk b = {BLK_UNSUPPORTED, type, NULL};
            msg_add_block(m, b);
        }
    }
    return 0;
}

/* tool -> user with a <tool_result> block, appended to the previous user turn
 * when that turn is itself a merged tool result. */
static int merge_messages(dp_list *out, const json_value *messages,
                          cwick_error *err) {
    for (size_t i = 0; i < cwick_json_array_len((json_value *)messages); i++) {
        json_value *msg = cwick_json_array_at((json_value *)messages, i);
        if (!msg || msg->kind != JSON_OBJECT) continue;
        const char *role = cwick_json_str(cwick_json_get(msg, "role"));
        if (!role) {
            cwick_error_set(err, CWICK_ERR_ARGUMENT, "message %zu has no role", i);
            return -1;
        }

        if (strcmp(role, "tool") == 0) {
            const json_value *content = cwick_json_get(msg, "content");
            blk b = {BLK_TOOL_RESULT, NULL, content};
            if (out->n > 0 && out->m[out->n - 1].role &&
                strcmp(out->m[out->n - 1].role, "user") == 0 &&
                out->m[out->n - 1].n_blocks > 0) {
                msg_add_block(&out->m[out->n - 1], b);
            } else {
                dp_msg *m = list_push(out);
                m->role = "user";
                msg_add_block(m, b);
            }
            continue;
        }

        dp_msg *m = list_push(out);
        /* OpenAI renamed the instruction turn to `developer`; this template
         * has one instruction role, and rendering `developer` as anything
         * else (an unknown role is an error here) would change the prompt. */
        m->role = strcmp(role, "developer") == 0 ? "system" : role;
        m->content = cwick_json_str(cwick_json_get(msg, "content"));
        m->tools = cwick_json_get(msg, "tools");
        m->tool_calls = cwick_json_get(msg, "tool_calls");
        m->response_format = cwick_json_get(msg, "response_format");
        m->reasoning_content =
            cwick_json_str(cwick_json_get(msg, "reasoning_content"));
        m->task = cwick_json_str(cwick_json_get(msg, "task"));
        m->wo_eos = cwick_json_bool(cwick_json_get(msg, "wo_eos"), false);

        if (strcmp(role, "user") == 0) {
            if (user_blocks_from(m, msg, err) != 0) return -1;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* rendering                                                            */
/* ------------------------------------------------------------------ */

static int reply_effort(const ds_prompt_opts *o) {
    if (o->reasoning_effort >= 1 && o->reasoning_effort <= 100)
        return o->reasoning_effort;
    return 75; /* "high" */
}

/* V4.1 supports mid-conversation system messages, which count as user messages
 * for the assistant header. A user turn made only of tool results does not
 * start a new turn. */
static int find_last_user_index(const dp_list *l) {
    for (int i = l->n - 1; i >= 0; i--) {
        const char *role = l->m[i].role;
        if (!role) continue;
        if (strcmp(role, "user") != 0 &&
            !(strcmp(role, "system") == 0 && i > 0))
            continue;
        if (l->m[i].n_blocks > 0) {
            int has_turn_block = 0;
            for (int k = 0; k < l->m[i].n_blocks; k++)
                if (l->m[i].blocks[k].kind == BLK_TEXT) has_turn_block = 1;
            if (!has_turn_block) continue;
        }
        return i;
    }
    return -1;
}

/* The tool_result block body: a string as-is, a list of parts joined with a
 * blank line, anything else serialized. */
static void put_tool_content(buf *b, const json_value *content) {
    if (!content) return;
    if (content->kind == JSON_STRING) {
        buf_puts(b, content->string ? content->string : "");
        return;
    }
    if (content->kind == JSON_ARRAY) {
        int first = 1;
        for (size_t i = 0; i < content->count; i++) {
            const json_value *part = const_str_array_at(content, i);
            const char *type =
                part ? cwick_json_str(cwick_json_get((json_value *)part, "type"))
                     : NULL;
            if (!first) buf_puts(b, "\n\n");
            first = 0;
            if (!type || strcmp(type, "text") == 0) {
                const char *t = part
                    ? cwick_json_str(cwick_json_get((json_value *)part, "text"))
                    : NULL;
                buf_puts(b, t ? t : "");
            } else {
                buf_printf(b, "[Unsupported %s]", type);
            }
        }
        return;
    }
    put_json(b, content);
}

/* One assistant tool call as a DSML invoke block. */
static int render_tool_calls(buf *b, const json_value *tool_calls,
                             cwick_error *err) {
    if (!tool_calls || tool_calls->kind != JSON_ARRAY) return 0;
    buf_puts(b, "\n\n<" DSML DSML_CALLS ">\n");
    int first = 1;
    for (size_t i = 0; i < tool_calls->count; i++) {
        json_value *tc = cwick_json_array_at((json_value *)tool_calls, i);
        if (!tc || tc->kind != JSON_OBJECT) continue;
        json_value *fn = cwick_json_get(tc, "function");
        if (!fn) fn = tc;
        const char *name = cwick_json_str(cwick_json_get(fn, "name"));
        if (!name) continue;
        const char *ns = cwick_json_str(cwick_json_get(tc, "namespace"));
        if (!ns) ns = cwick_json_str(cwick_json_get(fn, "namespace"));
        char nsbuf[128];
        const char *bare = split_tool_name(name, ns, &ns, nsbuf, sizeof(nsbuf), err);
        if (!bare) return -1;
        if (!first) buf_putc(b, '\n');
        first = 0;
        if (ns) buf_printf(b, "<" DSML DSML_INVOKE " name=\"%s::%s\">\n", ns, bare);
        else buf_printf(b, "<" DSML DSML_INVOKE " name=\"%s\">\n", bare);
        encode_arguments_to_dsml(b, cwick_json_get(fn, "arguments"));
        buf_puts(b, "\n</" DSML DSML_INVOKE ">");
    }
    buf_puts(b, "\n</" DSML DSML_CALLS ">");
    return 0;
}

static int render_message(buf *out, int index, const dp_list *l,
                          const ds_prompt_opts *o, bool drop,
                          const json_value *tools_override, int last_user_idx,
                          cwick_error *err) {
    const dp_msg *m = &l->m[index];
    const char *role = m->role;
    bool is_first = index == 0;
    int has_effort = o->thinking == DS_THINKING_THINKING && is_first;

    /* System token leads when there is an effort prefix or a system turn. */
    if (is_first && (has_effort || strcmp(role, "system") == 0))
        buf_puts(out, SP_SYSTEM);
    if (has_effort) buf_printf(out, EFFORT_FMT, reply_effort(o));

    if (strcmp(role, "system") == 0) {
        if (index > 0) buf_puts(out, SP_SYSTEM);
        buf_puts(out, m->content ? m->content : "");
        const json_value *tools = m->tools ? m->tools : tools_override;
        if (tools && cwick_json_array_len((json_value *)tools) > 0) {
            buf_puts(out, "\n\n");
            if (render_tools(out, tools, err) != 0) return -1;
        }
        if (m->response_format) {
            buf_puts(out, "\n\n" RESPONSE_FORMAT_HEAD);
            put_json(out, m->response_format);
        }
    } else if (strcmp(role, "user") == 0) {
        buf_puts(out, SP_USER);
        if (m->n_blocks > 0) {
            for (int i = 0; i < m->n_blocks; i++) {
                if (i) buf_puts(out, "\n\n");
                switch (m->blocks[i].kind) {
                case BLK_TEXT:
                    buf_puts(out, m->blocks[i].text ? m->blocks[i].text : "");
                    break;
                case BLK_TOOL_RESULT:
                    buf_puts(out, "<tool_result>");
                    put_tool_content(out, m->blocks[i].content);
                    buf_puts(out, "</tool_result>");
                    break;
                case BLK_UNSUPPORTED:
                    buf_printf(out, "[Unsupported %s]", m->blocks[i].text);
                    break;
                }
            }
        } else {
            buf_puts(out, m->content ? m->content : "");
        }
    } else if (strcmp(role, "latest_reminder") == 0) {
        buf_puts(out, SP_REMIND);
        buf_puts(out, m->content ? m->content : "");
    } else if (strcmp(role, "assistant") == 0) {
        /* Reasoning is kept only when it is part of the current turn: before
         * the last user turn it is dropped unless drop_thinking is off. */
        const char *rc = m->reasoning_content ? m->reasoning_content : "";
        int prev_has_task = index > 0 && l->m[index - 1].task != NULL;
        if (o->thinking == DS_THINKING_THINKING && !prev_has_task) {
            int keep = !drop || index > last_user_idx;
            if (keep) {
                buf_puts(out, rc);
                buf_puts(out, T_THINK_OUT);
            }
        }
        buf_puts(out, m->content ? m->content : "");
        if (render_tool_calls(out, m->tool_calls, err) != 0) return -1;
        if (!m->wo_eos) buf_puts(out, SP_EOS);
    } else if (strcmp(role, "tool") == 0) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "role=tool must be merged into a user turn first");
        return -1;
    } else {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "unknown role: %s", role);
        return -1;
    }

    /* Transition tokens. The header only goes in when the next turn wants one
     * (or when this is the last turn, i.e. the generation target). */
    if (index + 1 < l->n) {
        const char *nr = l->m[index + 1].role;
        int wants_header = nr && (strcmp(nr, "assistant") == 0 ||
                                  strcmp(nr, "latest_reminder") == 0);
        if (!wants_header) return 0;
    }

    if (m->task) {
        static const struct { const char *name, *tok; } task_tokens[] = {
            {"action", "<\xef\xbd\x9c" "action" "\xef\xbd\x9c>"},
            {"query", "<\xef\xbd\x9c" "query" "\xef\xbd\x9c>"},
            {"authority", "<\xef\xbd\x9c" "authority" "\xef\xbd\x9c>"},
            {"domain", "<\xef\xbd\x9c" "domain" "\xef\xbd\x9c>"},
            {"title", "<\xef\xbd\x9c" "title" "\xef\xbd\x9c>"},
            {"read_url", "<\xef\xbd\x9c" "read_url" "\xef\xbd\x9c>"},
        };
        const char *tok = NULL;
        for (size_t i = 0; i < sizeof(task_tokens) / sizeof(task_tokens[0]); i++)
            if (strcmp(task_tokens[i].name, m->task) == 0)
                tok = task_tokens[i].tok;
        if (!tok) {
            cwick_error_set(err, CWICK_ERR_ARGUMENT, "invalid task: '%s'", m->task);
            return -1;
        }
        if (strcmp(m->task, "action") != 0) {
            buf_puts(out, tok);
        } else {
            buf_puts(out, SP_ASSIST);
            buf_puts(out, o->thinking == DS_THINKING_THINKING ? T_THINK_IN
                                                            : T_THINK_OUT);
            buf_puts(out, tok);
        }
    } else if (strcmp(role, "user") == 0 ||
               (strcmp(role, "system") == 0 && index > 0)) {
        buf_puts(out, SP_ASSIST);
        if (!drop && o->thinking == DS_THINKING_THINKING)
            buf_puts(out, T_THINK_IN);
        else if (drop && o->thinking == DS_THINKING_THINKING &&
                 index >= last_user_idx)
            buf_puts(out, T_THINK_IN);
        else
            buf_puts(out, T_THINK_OUT);
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* public entry point                                                   */
/* ------------------------------------------------------------------ */

void ds_prompt_opts_default(ds_prompt_opts *o) {
    if (!o) return;
    o->thinking = DS_THINKING_THINKING;
    o->drop_thinking = true;
    o->add_bos = true;
    o->reasoning_effort = 0;
}

char *ds_prompt_encode(const json_value *messages, const json_value *tools,
                       const ds_prompt_opts *opts, size_t *out_len,
                       cwick_error *err) {
    ds_prompt_opts dflt;
    if (!opts) {
        ds_prompt_opts_default(&dflt);
        opts = &dflt;
    }
    if (out_len) *out_len = 0;

    dp_list l;
    memset(&l, 0, sizeof(l));
    if (merge_messages(&l, messages, err) != 0) {
        for (int i = 0; i < l.n; i++) free(l.m[i].blocks);
        free(l.m);
        return NULL;
    }

    /* A `tools` declaration anywhere disables thinking-drop: tool-calling
     * conversations need the full reasoning trail. */
    bool drop = opts->drop_thinking;
    if (tools && cwick_json_array_len((json_value *)tools) > 0) drop = false;
    for (int i = 0; i < l.n; i++)
        if (l.m[i].tools && cwick_json_array_len((json_value *)l.m[i].tools) > 0)
            drop = false;

    /* No system turn but tools to declare: the tools block needs a turn to
     * live in, and the reference only renders it there. */
    dp_msg synthetic;
    memset(&synthetic, 0, sizeof(synthetic));
    if (tools && cwick_json_array_len((json_value *)tools) > 0 &&
        !(l.n > 0 && l.m[0].role && strcmp(l.m[0].role, "system") == 0)) {
        synthetic.role = "system";
        synthetic.tools = tools;
        if (l.n == l.cap) {
            dp_msg *nm = realloc(l.m, (size_t)(l.cap + 16) * sizeof(dp_msg));
            if (!nm) { free(l.m); cwick_error_set(err, CWICK_ERR_MEMORY, "out of memory"); return NULL; }
            l.m = nm;
            l.cap += 16;
        }
        memmove(&l.m[1], &l.m[0], (size_t)l.n * sizeof(dp_msg));
        l.m[0] = synthetic;
        l.n++;
    }

    int last_user_idx = find_last_user_index(&l);

    buf out;
    memset(&out, 0, sizeof(out));
    buf_grow(&out, 64);
    out.p[0] = '\0';
    if (opts->add_bos) buf_puts(&out, SP_BOS);

    int rc = 0;
    for (int i = 0; i < l.n && rc == 0; i++)
        rc = render_message(&out, i, &l, opts, drop,
                            /* the override is only consumed by a system turn,
                             * and the synthetic one already carries it */
                            NULL, last_user_idx, err);

    for (int i = 0; i < l.n; i++) free(l.m[i].blocks);
    free(l.m);
    if (rc != 0) {
        free(out.p);
        return NULL;
    }
    if (out_len) *out_len = out.n;
    return out.p;
}
