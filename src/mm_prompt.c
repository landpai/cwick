/*
 * mm_prompt.c — MiniMax-M3.1-Flash prompt encoding in C. See mm_prompt.h.
 *
 * The template is a Jinja file, but every construct it uses that matters here
 * is a literal block, so the port is a matter of getting the bytes and their
 * order right rather than of interpreting a language. Two details are easy to
 * get wrong and both change the prompt:
 *
 *   * `]~!b[` opens the *system* block only. Every later block starts with
 *     `]~b]` alone, and there is no separator between a block's text and its
 *     `[e~[`.
 *   * the thinking instructions and `<model_id>` are appended to the system
 *     block whether or not the caller sent a system message -- only the
 *     identity paragraph is replaced.
 */
#include "mm_prompt.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "log.h"

#define BOD  "]~!b["
#define BOS  "]~b]"
#define EOS  "[e~["
#define NS   "]<]minimax[>["
#define THINK_OPEN  "<mm:think>"
#define THINK_CLOSE "</mm:think>"

static const char DEFAULT_IDENTITY[] =
    "Your model version is MiniMax-M3.1, developed by MiniMax. Knowledge "
    "cutoff: January 2026. Founded in early 2022, MiniMax is a global AI "
    "foundation model company committed to advancing the frontiers of AI "
    "towards AGI.";

static const char THINK_RULES[] =
    "\n\n<thinking_instructions>\n"
    "You have a thinking capability that allows you to reason step by step "
    "before responding. When thinking is enabled, wrap your reasoning in "
    THINK_OPEN THINK_CLOSE " tags before your response. When thinking is "
    "disabled, begin your response directly after the " THINK_CLOSE " prefix. "
    "When thinking is adaptive, decide on your own whether to think for the "
    "current turn.\n"
    "Current thinking mode: adaptive. You are encouraged to think for complex "
    "decision-making, multi-step reasoning, or when analyzing function/tool "
    "results.\n"
    "</thinking_instructions>"
    "\n\n<model_id>8bb8149fe</model_id>";

static const char TOOLS_HEAD[] =
    "\n\n# Tools\n"
    "You may call one or more tools to assist with the user query.\n"
    "Here are the tools available in JSONSchema format:\n"
    "\n<tools>\n";

static const char TOOLS_TAIL[] =
    "</tools>\n\n"
    "To call tools, wrap all invocations in a single "
    NS "<tool_call>" NS "</tool_call> block. Parameter values containing "
    "nested objects or arrays are recursively expanded into XML elements. "
    "Example:\n"
    "\n" NS "<tool_call>\n"
    NS "<invoke name=\"tool-name-1\">"
    NS "<param-1>value-1" NS "</param-1>"
    NS "<param-2>" NS "<item>" NS "<key-a>val-a" NS "</key-a>"
    NS "<key-b>val-b" NS "</key-b>" NS "</item>" NS "</param-2>"
    NS "</invoke>\n"
    NS "<invoke name=\"tool-name-2\">"
    NS "<param-1>value-1" NS "</param-1>"
    NS "</invoke>\n"
    NS "</tool_call>";

/* ------------------------------------------------------------------ */
/* a string builder that reports out-of-memory once                    */
/* ------------------------------------------------------------------ */

typedef struct {
    char *p;
    size_t n, cap;
    bool oom;
} sb;

static void sb_reserve(sb *b, size_t extra) {
    if (b->oom) return;
    if (b->n + extra + 1 <= b->cap) return;
    size_t cap = b->cap ? b->cap : 4096;
    while (cap < b->n + extra + 1) cap *= 2;
    char *np = (char *)realloc(b->p, cap);
    if (!np) { b->oom = true; return; }
    b->p = np;
    b->cap = cap;
}

static void sb_add(sb *b, const char *s, size_t len) {
    sb_reserve(b, len);
    if (b->oom) return;
    memcpy(b->p + b->n, s, len);
    b->n += len;
    b->p[b->n] = '\0';
}

static void sb_puts(sb *b, const char *s) { sb_add(b, s, strlen(s)); }

static void sb_printf(sb *b, const char *fmt, ...) {
    char tmp[512];
    va_list ap;
    va_start(ap, fmt);
    int k = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (k < 0) { b->oom = true; return; }
    if ((size_t)k < sizeof(tmp)) { sb_add(b, tmp, (size_t)k); return; }
    char *big = (char *)malloc((size_t)k + 1);
    if (!big) { b->oom = true; return; }
    va_start(ap, fmt);
    vsnprintf(big, (size_t)k + 1, fmt, ap);
    va_end(ap);
    sb_add(b, big, (size_t)k);
    free(big);
}

static void sb_str_json(sb *b, json_value *v);

/* Visible text of a message's content: a string as-is, an array of parts by
 * concatenating the text ones, anything else by serialising it. */
static void sb_content(sb *b, json_value *content) {
    if (!content) return;
    if (content->kind == JSON_STRING) {
        const char *s = cwick_json_str(content);
        if (s) sb_puts(b, s);
        return;
    }
    if (content->kind == JSON_ARRAY) {
        for (size_t i = 0; i < content->count; i++) {
            json_value *it = cwick_json_array_at(content, i);
            if (!it) continue;
            if (it->kind == JSON_STRING) {
                const char *s = cwick_json_str(it);
                if (s) sb_puts(b, s);
                continue;
            }
            const char *t = cwick_json_str(cwick_json_get(it, "type"));
            if (t && strcmp(t, "text") == 0) {
                const char *s = cwick_json_str(cwick_json_get(it, "text"));
                if (s) sb_puts(b, s);
            }
            /* image / video parts have no renderer in this port */
        }
        return;
    }
    sb_str_json(b, content);
}

static void sb_str_json(sb *b, json_value *v) {
    char *s = cwick_json_write(v);
    if (s) {
        sb_puts(b, s);
        free(s);
    }
}

/* ------------------------------------------------------------------ */
/* blocks                                                              */
/* ------------------------------------------------------------------ */

static void render_system(sb *b, json_value *sys, int effort) {
    sb_puts(b, BOD BOS "system\n");
    const char *content = sys ? cwick_json_str(cwick_json_get(sys, "content"))
                              : NULL;
    if (sys && content && *content)
        sb_puts(b, content);
    else if (sys && cwick_json_get(sys, "content"))
        sb_content(b, cwick_json_get(sys, "content"));
    else
        sb_puts(b, DEFAULT_IDENTITY);
    sb_puts(b, THINK_RULES);
    if (effort > 0) sb_printf(b, "\n\n<effort>%d</effort>", effort);
    sb_puts(b, EOS "\n");
}

static void render_developer(sb *b, json_value *dev, json_value *tools) {
    sb_puts(b, BOS "developer\n");
    const char *content = dev ? cwick_json_str(cwick_json_get(dev, "content"))
                              : NULL;
    if (dev && content && *content)
        sb_puts(b, content);
    else
        sb_puts(b, "You are a helpful assistant.");

    if (tools && tools->kind == JSON_ARRAY && tools->count > 0) {
        sb_puts(b, TOOLS_HEAD);
        for (size_t i = 0; i < tools->count; i++) {
            json_value *t = cwick_json_array_at(tools, i);
            if (!t) continue;
            json_value *fn = cwick_json_get(t, "function");
            sb_puts(b, "<tool>");
            sb_str_json(b, fn ? fn : t);
            sb_puts(b, "</tool>\n");
        }
        sb_puts(b, TOOLS_TAIL);
    }
    sb_puts(b, EOS "\n");
}

/* One assistant turn: header, the reasoning block (or the close marker that
 * stands in for one), the content, tool calls, end of turn. */
static void render_assistant(sb *b, json_value *m) {
    sb_puts(b, BOS "ai\n");

    json_value *rc = cwick_json_get(m, "reasoning_content");
    const char *reasoning = rc && rc->kind == JSON_STRING
                                ? cwick_json_str(rc) : NULL;
    json_value *content = cwick_json_get(m, "content");

    /* A client that replays the answer with its markers inside the content
     * (rather than in reasoning_content) is split here, the way the template
     * splits it. */
    char *split_think = NULL;
    if ((!reasoning || !*reasoning) && content && content->kind == JSON_STRING) {
        const char *s = cwick_json_str(content);
        if (s) {
            const char *close = strstr(s, THINK_CLOSE);
            const char *open = strstr(s, THINK_OPEN);
            if (close) {
                const char *from = open && open < close ? open + strlen(THINK_OPEN) : s;
                size_t len = (size_t)(close - from);
                split_think = (char *)malloc(len + 1);
                if (split_think) {
                    memcpy(split_think, from, len);
                    split_think[len] = '\0';
                    reasoning = split_think;
                }
                content = NULL;   /* the tail is re-read below */
                sb_puts(b, THINK_OPEN);
                sb_puts(b, reasoning);
                sb_puts(b, THINK_CLOSE);
                sb_puts(b, close + strlen(THINK_CLOSE));
                free(split_think);
                split_think = NULL;
                goto tools;
            }
        }
    }
    if (reasoning && *reasoning) {
        sb_puts(b, THINK_OPEN);
        sb_puts(b, reasoning);
        sb_puts(b, THINK_CLOSE);
    } else {
        sb_puts(b, THINK_CLOSE);
    }
    sb_content(b, content);

tools:
    {
        json_value *calls = cwick_json_get(m, "tool_calls");
        if (calls && calls->kind == JSON_ARRAY && calls->count > 0) {
            sb_puts(b, NS "<tool_call>\n");
            for (size_t i = 0; i < calls->count; i++) {
                json_value *tc = cwick_json_array_at(calls, i);
                if (!tc) continue;
                json_value *fn = cwick_json_get(tc, "function");
                if (fn) tc = fn;
                const char *name = cwick_json_str(cwick_json_get(tc, "name"));
                sb_puts(b, NS "<invoke name=\"");
                sb_puts(b, name ? name : "");
                sb_puts(b, "\">");
                json_value *args = cwick_json_get(tc, "arguments");
                /* OpenAI sends the arguments as a JSON *string*; the template
                 * walks a mapping. Both shapes are accepted here. */
                json_value *parsed = NULL;
                if (args && args->kind == JSON_STRING) {
                    cwick_error je;
                    cwick_error_clear(&je);
                    parsed = cwick_json_parse(cwick_json_str(args),
                                              strlen(cwick_json_str(args)), &je);
                    if (parsed) args = parsed;
                }
                if (args && args->kind == JSON_OBJECT) {
                    for (size_t k = 0; k < args->count; k++) {
                        const char *key = args->members[k].key;
                        json_value *val = args->members[k].value;
                        if (!key || !val || val->kind == JSON_NULL) continue;
                        sb_puts(b, NS "<");
                        sb_puts(b, key);
                        sb_puts(b, ">");
                        if (val->kind == JSON_STRING)
                            sb_puts(b, cwick_json_str(val));
                        else
                            sb_str_json(b, val);
                        sb_puts(b, NS "</");
                        sb_puts(b, key);
                        sb_puts(b, ">");
                    }
                }
                if (parsed) cwick_json_free(parsed);
                sb_puts(b, NS "</invoke>\n");
            }
            sb_puts(b, NS "</tool_call>");
        }
    }
    sb_puts(b, EOS "\n");
}

static void render_tool(sb *b, json_value *m, bool first_of_group) {
    if (first_of_group) sb_puts(b, BOS "tool");
    sb_puts(b, "\n<response>");
    sb_content(b, cwick_json_get(m, "content"));
    sb_puts(b, "</response>");
}

void mm_prompt_opts_default(mm_prompt_opts *o) {
    o->thinking = true;
    o->reasoning_effort = 0;
}

bool mm_prompt_is_toolcall(const char *text) {
    if (!text) return false;
    const char *p = text;
    while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
    return strncmp(p, NS "<tool_call>", strlen(NS "<tool_call>")) == 0;
}

char *mm_prompt_encode(json_value *messages, json_value *tools,
                       const mm_prompt_opts *opts, size_t *out_len,
                       cwick_error *err) {
    mm_prompt_opts dflt;
    if (!opts) {
        mm_prompt_opts_default(&dflt);
        opts = &dflt;
    }
    if (!messages || messages->kind != JSON_ARRAY) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "minimax prompt: messages must be an array");
        return NULL;
    }

    /* The template gives `root` (high priority) and `system`/`developer` (low
     * priority) roles a fixed place ahead of the conversation. Anything else
     * at the head of the array stays in the loop. */
    json_value *sys = NULL, *dev = NULL;
    size_t first = 0;
    if (messages->count > 0) {
        json_value *m0 = cwick_json_array_at(messages, 0);
        const char *r0 = m0 ? cwick_json_str(cwick_json_get(m0, "role")) : NULL;
        if (r0 && strcmp(r0, "root") == 0) {
            sys = m0;
            first = 1;
            if (messages->count > 1) {
                json_value *m1 = cwick_json_array_at(messages, 1);
                const char *r1 = m1 ? cwick_json_str(cwick_json_get(m1, "role"))
                                    : NULL;
                if (r1 && (strcmp(r1, "system") == 0 ||
                           strcmp(r1, "developer") == 0)) {
                    dev = m1;
                    first = 2;
                }
            }
        } else if (r0 && (strcmp(r0, "system") == 0 ||
                          strcmp(r0, "developer") == 0)) {
            dev = m0;
            first = 1;
        }
    }

    sb b;
    memset(&b, 0, sizeof(b));
    render_system(&b, sys, opts->reasoning_effort);
    render_developer(&b, dev, tools);

    for (size_t i = first; i < messages->count; i++) {
        json_value *m = cwick_json_array_at(messages, i);
        if (!m) continue;
        const char *role = cwick_json_str(cwick_json_get(m, "role"));
        if (!role) continue;
        if (strcmp(role, "assistant") == 0) {
            render_assistant(&b, m);
        } else if (strcmp(role, "tool") == 0) {
            /* Consecutive tool messages share one `]~b]tool` header. */
            bool first_of_group = true;
            if (i > first) {
                json_value *prev = cwick_json_array_at(messages, i - 1);
                const char *pr = prev
                    ? cwick_json_str(cwick_json_get(prev, "role")) : NULL;
                if (pr && strcmp(pr, "tool") == 0) first_of_group = false;
            }
            render_tool(&b, m, first_of_group);
            /* A group ends when the next message is not a tool result. */
            bool last = true;
            if (i + 1 < messages->count) {
                json_value *nx = cwick_json_array_at(messages, i + 1);
                const char *nr = nx
                    ? cwick_json_str(cwick_json_get(nx, "role")) : NULL;
                if (nr && strcmp(nr, "tool") == 0) last = false;
            }
            if (last) sb_puts(&b, EOS "\n");
        } else if (strcmp(role, "system") == 0 ||
                   strcmp(role, "developer") == 0) {
            sb_puts(&b, BOS "developer\n");
            const char *c = cwick_json_str(cwick_json_get(m, "content"));
            if (c && *c) sb_puts(&b, c);
            else sb_content(&b, cwick_json_get(m, "content"));
            sb_puts(&b, EOS "\n");
        } else {
            sb_puts(&b, BOS "user\n");
            sb_content(&b, cwick_json_get(m, "content"));
            sb_puts(&b, EOS "\n");
        }
    }

    sb_puts(&b, BOS "ai\n");
    if (!opts->thinking) sb_puts(&b, THINK_CLOSE);

    if (b.oom || !b.p) {
        free(b.p);
        cwick_error_set(err, CWICK_ERR_MEMORY,
                        "minimax prompt: out of memory rendering %zu messages",
                        messages->count);
        return NULL;
    }
    if (out_len) *out_len = b.n;
    return b.p;
}
