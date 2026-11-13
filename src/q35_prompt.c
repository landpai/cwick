/*
 * q35_prompt.c — Qwen3.5 prompt encoding in C. See q35_prompt.h.
 *
 * The template is a Jinja file, but every construct that matters for a text
 * conversation is a literal block, so the port is a matter of getting the bytes
 * and their order right rather than of interpreting a language. Four details
 * are easy to get wrong and each one changes the prompt:
 *
 *   * the first one or two system/developer messages are *merged* and written
 *     once, ahead of the loop; the loop then skips them, so a second system
 *     message must not appear twice;
 *   * a tool declaration replaces the whole system block -- the merged system
 *     text moves to the end of the tool preamble rather than staying first;
 *   * consecutive `tool` messages collapse into ONE user turn, each wrapped in
 *     `<tool_response>`; and
 *   * an assistant turn replays its reasoning only when it sits after the last
 *     real user query, which is what `loop.index0 > last_query_index` means.
 */
#include "q35_prompt.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "log.h"
#include "util.h"

#define IM_START "<|im_start|>"
#define IM_END   "<|im_end|>"
#define T_OPEN   "<think>"
#define T_CLOSE  "</think>"

static const char TOOLS_HEAD[] =
    "# Tools\n\n"
    "You have access to the following functions:\n\n<tools>";

static const char TOOLS_TAIL[] =
    "\n</tools>"
    "\n\nIf you choose to call a function ONLY reply in the following format "
    "with NO suffix:\n\n"
    "<tool_call>\n"
    "<function=example_function_name>\n"
    "<parameter=example_parameter_1>\n"
    "value_1\n"
    "</parameter>\n"
    "<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\n"
    "that can span\n"
    "multiple lines\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n\n"
    "<IMPORTANT>\n"
    "Reminder:\n"
    "- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within "
    "<tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural "
    "language BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal "
    "with your current knowledge and do not tell the user about function "
    "calls\n"
    "</IMPORTANT>";

/* ------------------------------------------------------------------ */
/* a string builder that reports out-of-memory once                    */
/* ------------------------------------------------------------------ */

typedef struct {
    char *p;
    size_t len, cap;
    int oom;
} qb;

static void qb_reserve(qb *b, size_t extra) {
    if (b->oom) return;
    if (b->len + extra + 1 <= b->cap) return;
    size_t cap = b->cap ? b->cap : 1024;
    while (cap < b->len + extra + 1) cap *= 2;
    char *np = realloc(b->p, cap);
    if (!np) { b->oom = 1; return; }
    b->p = np;
    b->cap = cap;
}

static void qb_puts(qb *b, const char *s) {
    if (!s) return;
    size_t n = strlen(s);
    qb_reserve(b, n);
    if (b->oom) return;
    memcpy(b->p + b->len, s, n);
    b->len += n;
    b->p[b->len] = '\0';
}

static void qb_putf(qb *b, const char *fmt, ...) {
    char tmp[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if ((size_t)n < sizeof(tmp)) {
        qb_puts(b, tmp);
        return;
    }
    char *big = malloc((size_t)n + 1);
    if (!big) { b->oom = 1; return; }
    va_start(ap, fmt);
    vsnprintf(big, (size_t)n + 1, fmt, ap);
    va_end(ap);
    qb_puts(b, big);
    free(big);
}

/* ------------------------------------------------------------------ */
/* text extraction                                                     */
/* ------------------------------------------------------------------ */

/* The template applies `|trim` to every rendered content, so leading and
 * trailing whitespace never reaches the model. */
static void qb_puts_trimmed(qb *b, const char *s) {
    if (!s) return;
    while (*s == ' ' || *s == '\n' || *s == '\t' || *s == '\r') s++;
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == ' ' || s[n - 1] == '\n' || s[n - 1] == '\t' ||
                     s[n - 1] == '\r'))
        n--;
    qb_reserve(b, n);
    if (b->oom) return;
    memcpy(b->p + b->len, s, n);
    b->len += n;
    b->p[b->len] = '\0';
}

/* A content field is either a string or an array of typed parts. Non-text
 * parts are dropped: the vision tower is not loaded in this build, and turning
 * an image into its placeholder token without patches behind it would be a
 * prompt the model cannot read. */
static void render_content(qb *b, json_value *content, int allow_media) {
    if (!content) return;
    if (content->kind == JSON_STRING) {
        qb_puts_trimmed(b, content->string);
        return;
    }
    if (content->kind != JSON_ARRAY) return;
    for (size_t i = 0; i < content->count; i++) {
        json_value *it = cwick_json_array_at(content, i);
        if (!it || it->kind != JSON_OBJECT) continue;
        json_value *txt = cwick_json_get(it, "text");
        if (txt && txt->kind == JSON_STRING) {
            qb_puts(b, txt->string);
            continue;
        }
        const char *type = cwick_json_str(cwick_json_get(it, "type"));
        if (type && (strcmp(type, "image_url") == 0 || strcmp(type, "image") == 0 ||
                     strcmp(type, "video") == 0 || strcmp(type, "input_audio") == 0 ||
                     strcmp(type, "audio") == 0)) {
            if (!allow_media)
                CWICK_INFO("prompt", "%s part dropped: this build has no vision "
                           "or audio tower", type);
            continue;
        }
    }
}

static int content_is_tool_response(json_value *content) {
    if (!content || content->kind != JSON_STRING || !content->string) return 0;
    const char *s = content->string;
    while (*s == ' ' || *s == '\n' || *s == '\t' || *s == '\r') s++;
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == ' ' || s[n - 1] == '\n' || s[n - 1] == '\t' ||
                     s[n - 1] == '\r'))
        n--;
    const char *open = "<tool_response>";
    const char *close = "</tool_response>";
    size_t lo = strlen(open), lc = strlen(close);
    return n > lo + lc && strncmp(s, open, lo) == 0 &&
           strncmp(s + n - lc, close, lc) == 0;
}

/* OpenAI tool_calls -> the XML the template asks the model to produce. */
static void render_tool_calls(qb *b, json_value *calls) {
    if (!calls || calls->kind != JSON_ARRAY) return;
    for (size_t i = 0; i < calls->count; i++) {
        json_value *tc = cwick_json_array_at(calls, i);
        if (!tc || tc->kind != JSON_OBJECT) continue;
        json_value *fn = cwick_json_get(tc, "function");
        if (!fn || fn->kind != JSON_OBJECT) continue;
        const char *name = cwick_json_str(cwick_json_get(fn, "name"));
        json_value *args = cwick_json_get(fn, "arguments");
        qb_putf(b, "\n%s<tool_call>\n<function=%s>\n", i ? "\n" : "", name ? name : "");
        if (args && args->kind == JSON_OBJECT) {
            for (size_t a = 0; a < args->count; a++) {
                json_value *v = args->members[a].value;
                qb_putf(b, "<parameter=%s>\n", args->members[a].key);
                if (v && (v->kind == JSON_OBJECT || v->kind == JSON_ARRAY)) {
                    char *s = cwick_json_write(v);
                    qb_puts(b, s ? s : "");
                    free(s);
                } else if (v && v->kind == JSON_STRING) {
                    qb_puts(b, v->string);
                } else if (v && v->kind == JSON_NUMBER) {
                    qb_putf(b, "%g", v->number);
                } else if (v && v->kind == JSON_BOOL) {
                    qb_puts(b, v->boolean ? "true" : "false");
                }
                qb_puts(b, "\n</parameter>\n");
            }
        }
        qb_puts(b, "</function>\n</tool_call>");
    }
}

/* ------------------------------------------------------------------ */
/* encode                                                              */
/* ------------------------------------------------------------------ */

void q35_prompt_opts_default(q35_prompt_opts *o) {
    if (o) o->thinking = true;
}

bool q35_prompt_is_toolcall(const char *text) {
    if (!text) return false;
    while (*text == ' ' || *text == '\n' || *text == '\t' || *text == '\r') text++;
    return strncmp(text, "<tool_call>", 11) == 0;
}

static int role_is(json_value *m, const char *want) {
    const char *r = cwick_json_str(cwick_json_get(m, "role"));
    return r && strcmp(r, want) == 0;
}

char *q35_prompt_encode(json_value *messages, json_value *tools,
                        const q35_prompt_opts *opts, size_t *out_len,
                        cwick_error *err) {
    q35_prompt_opts dflt;
    if (!opts) { q35_prompt_opts_default(&dflt); opts = &dflt; }
    size_t nmsg = (messages && messages->kind == JSON_ARRAY) ? messages->count : 0;
    int has_tools = tools && tools->kind == JSON_ARRAY && tools->count > 0;

    qb b = {0};

    /* ---- the merged system block ---- */
    size_t nsys = 0;
    qb sys = {0};
    for (size_t i = 0; i < nmsg && i < 2; i++) {
        json_value *m = cwick_json_array_at(messages, i);
        if (!m || (m->kind != JSON_OBJECT)) break;
        if (!role_is(m, "system") && !role_is(m, "developer")) break;
        if (sys.len) qb_putf(&sys, "\n");
        render_content(&sys, cwick_json_get(m, "content"), 0);
        nsys = i + 1;
    }

    if (has_tools) {
        qb_putf(&b, IM_START "system\n%s", TOOLS_HEAD);
        for (size_t i = 0; i < tools->count; i++) {
            json_value *t = cwick_json_array_at(tools, i);
            char *s = t ? cwick_json_write(t) : NULL;
            qb_putf(&b, "\n%s", s ? s : "{}");
            free(s);
        }
        qb_puts(&b, TOOLS_TAIL);
        if (sys.len) qb_putf(&b, "\n\n%s", sys.p);
        qb_puts(&b, IM_END "\n");
    } else if (sys.len) {
        qb_putf(&b, IM_START "system\n%s" IM_END "\n", sys.p);
    }
    free(sys.p);

    /* ---- index of the last real user query ---- */
    long last_query = -1;
    for (size_t i = 0; i < nmsg; i++) {
        json_value *m = cwick_json_array_at(messages, i);
        if (!m || m->kind != JSON_OBJECT || !role_is(m, "user")) continue;
        if (!content_is_tool_response(cwick_json_get(m, "content")))
            last_query = (long)i;
    }

    /* ---- the turns ---- */
    for (size_t i = nsys; i < nmsg; i++) {
        json_value *m = cwick_json_array_at(messages, i);
        if (!m || m->kind != JSON_OBJECT) continue;
        if (role_is(m, "system") || role_is(m, "developer")) continue;
        json_value *content = cwick_json_get(m, "content");

        if (role_is(m, "user")) {
            qb_puts(&b, IM_START "user\n");
            render_content(&b, content, 0);
            qb_puts(&b, IM_END "\n");
        } else if (role_is(m, "assistant")) {
            qb_puts(&b, IM_START "assistant\n");
            /* An assistant turn after the last query is the one being
             * continued, so its reasoning is replayed; older turns keep only
             * the answer. */
            if ((long)i > last_query) {
                qb_puts(&b, T_OPEN "\n");
                json_value *rc = cwick_json_get(m, "reasoning_content");
                if (rc && rc->kind == JSON_STRING) {
                    qb_puts_trimmed(&b, rc->string);
                } else if (content && content->kind == JSON_STRING &&
                           content->string && strstr(content->string, T_CLOSE)) {
                    /* The OpenAI shape has no reasoning field, so a client that
                     * round-trips the model's own output carries it inline. */
                    const char *s = content->string;
                    const char *close = strstr(s, T_CLOSE);
                    qb_puts_trimmed(&b, s);
                    (void)close;
                }
                qb_puts(&b, "\n" T_CLOSE "\n\n");
            }
            qb_puts_trimmed(&b, (content && content->kind == JSON_STRING)
                                     ? content->string : NULL);
            if (content && content->kind == JSON_ARRAY) {
                /* Already-trimmed parts: append without a second trim pass. */
                qb spare = {0};
                render_content(&spare, content, 0);
                qb_puts(&b, spare.p ? spare.p : "");
                free(spare.p);
            }
            render_tool_calls(&b, cwick_json_get(m, "tool_calls"));
            qb_puts(&b, IM_END "\n");
        } else if (role_is(m, "tool")) {
            /* Consecutive tool results share one user turn. */
            int prev_tool = i > 0 &&
                (role_is(cwick_json_array_at(messages, i - 1), "tool"));
            if (!prev_tool) qb_puts(&b, IM_START "user");
            qb_puts(&b, "\n<tool_response>\n");
            render_content(&b, content, 0);
            qb_puts(&b, "\n</tool_response>");
            int next_tool = (i + 1 < nmsg) &&
                role_is(cwick_json_array_at(messages, i + 1), "tool");
            if (!next_tool) qb_puts(&b, IM_END "\n");
        }
    }

    /* ---- the generation prompt ---- */
    qb_puts(&b, IM_START "assistant\n");
    if (opts->thinking) {
        qb_puts(&b, T_OPEN "\n");
    } else {
        qb_puts(&b, T_OPEN "\n\n" T_CLOSE "\n\n");
    }

    if (b.oom) {
        free(b.p);
        cwick_error_set(err, CWICK_ERR_MEMORY, "out of memory rendering the prompt");
        return NULL;
    }
    if (out_len) *out_len = b.len;
    return b.p;
}
