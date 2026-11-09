/* tools.h — OpenAI function/tool declarations and GLM-5.3 tool-call parsing.
 *
 * A coding agent pointed at `cwick serve` wants tool calling, and cwick will
 * not pretend to something it cannot honour: tools are advertized to the
 * model in the checkpoint's own protocol (from
 * `zai-org/GLM-5.3-Flash/chat_template.jinja`) and read back out of its raw
 * text output:
 *
 *   prompt side                       completion side
 *   <tools>                           <tool_call>get_weather
 *   {"name": ..., "parameters": ...}  <arg_key>city</arg_key>
 *   ...                               <arg_value>"Beijing"</arg_value>
 *   </tools>                          </tool_call>
 *   For each function call, output
 *   the function name and arguments
 *   within the following XML format:
 *   <tool_call>...</tool_call>
 *
 * Nothing here constrains decoding: there is no grammar forcing the model to
 * emit exactly one tool call, so malformed output is detected and reported
 * (fail-closed) rather than passed through to the client half-parsed.
 */
#ifndef CWICK_TOOLS_H
#define CWICK_TOOLS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "cwick.h"
#include "json.h"

/* One declared function, kept as compact JSON exactly as the model should
 * see it, plus its name for tool_choice matching. */
typedef struct cwick_tool_decl {
    char name[192];
    char *json;                  /* {"name":…, "description":…, "parameters":…} */
} cwick_tool_decl;

/* What the request asked for. */
typedef struct cwick_tool_set {
    bool present;                /* `tools` was in the body */
    bool advertise;              /* …and should reach the prompt */
    char choice[16];             /* "auto" | "none" | "required" | "named" */
    char forced[192];            /* when choice == "named": the function name */
    cwick_tool_decl *items;
    size_t count;
} cwick_tool_set;

/* One parsed tool call out of the model's completion. */
typedef struct cwick_tool_arg {
    char *key;
    char *value_json;            /* normalized JSON value */
} cwick_tool_arg;

typedef struct cwick_tool_call {
    char id[80];
    char name[192];
    cwick_tool_arg *args;
    size_t n_args;
    char *arguments_json;        /* {"k":v,…} ready for the OpenAI payload */
} cwick_tool_call;

/* ---- request side --------------------------------------------------- */
/* Validate `tools` / `tool_choice` and build the declaration set.
 * Returns 0 when the request is acceptable, 400 with `errbuf` filled when it
 * asks for something cwick cannot honour exactly (fail-closed, same rule as
 * every other refused OpenAI feature). */
int cwick_tools_parse_request(const json_value *root, cwick_tool_set *out,
                             char *errbuf, size_t errcap);
void cwick_tools_free_set(cwick_tool_set *set);

/* Render the `<tools>` section to splice into the chat prompt, or NULL when
 * no tools should be advertised (no tools declared, or tool_choice "none").
 * Caller frees. */
char *cwick_tools_render_block(const cwick_tool_set *set);

/* ---- completion side ------------------------------------------------ */
/* Extract every `<tool_call>…</tool_call>` from a completion.
 * `*visible` (may be NULL) receives the assistant's plain-text part; if there
 * is none it is set to NULL. Returns 0 on success with `*n` possibly 0, and
 * -1 when a tool call is present but malformed — the caller must refuse the
 * request rather than hand over a half-parsed tool call. */
int cwick_tools_extract(const char *text, cwick_tool_call **calls, size_t *n,
                       char **visible, cwick_error *err);
void cwick_tool_calls_free(cwick_tool_call *calls, size_t n);

/* Does a parsed call name a declared tool?
 * The model copies tool names out of the `<tools>` block but does not always
 * keep their case — a declared `Bash` has come back as `bash` — and the
 * OpenAI contract is the exact declared name, so a call that matches a
 * declaration up to case is rewritten to the declared spelling in place.
 * EXACT: already matches a declaration; FOLDED: rewritten to one;
 * UNKNOWN: nothing declared matches and the caller must refuse the call. */
typedef enum {
    CWICK_DECL_EXACT = 0,
    CWICK_DECL_FOLDED,
    CWICK_DECL_UNKNOWN
} cwick_decl_match;
cwick_decl_match cwick_tool_decl_match(cwick_tool_call *call,
                                       const cwick_tool_set *set);

/* ---- streaming side ------------------------------------------------- */
/* One checkpoint prefixes every structural tag of a tool block with a namespace
 * token that is a single entry in its added-token table, so the bytes are
 * framing and not something a user should ever read. While the block is still
 * being generated the server forwards the text as tool progress, and these two
 * keep that copy clean: `cwick_tool_ns_tail` says how many trailing bytes must
 * be held back because they could still grow into a token, and
 * `cwick_tool_ns_strip` returns a copy of a range with every complete token
 * removed (caller frees). */
size_t cwick_tool_ns_tail(const char *s, size_t n);
char *cwick_tool_ns_strip(const char *s, size_t n);

/* Escape a raw fragment into a JSON string literal (no surrounding quotes).
 * Exposed so the server can embed generated text in SSE frames. */
char *cwick_json_stringify(const char *s);

#endif /* CWICK_TOOLS_H */
