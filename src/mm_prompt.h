/*
 * mm_prompt.h — MiniMax-M3.1-Flash prompt encoding, in C.
 *
 * The checkpoint ships `chat_template.jinja`; this is a port of its text path,
 * keeping the same blocks so the two can be compared side by side:
 *
 *   system       `]~!b[]~b]system` + the identity block + the thinking
 *                instructions + <model_id>, ended by `[e~[`
 *   developer    `]~b]developer` + the developer text + the tool schema block
 *   turn         `]~b]user` / `]~b]ai` / `]~b]tool`, ended by `[e~[`
 *   generate     `]~b]ai\n`
 *
 * Out of scope, matching what this port can run: image and video content
 * blocks (the vision tower is not loaded, so an image part is dropped rather
 * than silently becoming text), and the Jinja macros' XML tool-call rendering
 * for *history* is supported only for the OpenAI shape the server receives.
 */
#ifndef CWICK_MM_PROMPT_H
#define CWICK_MM_PROMPT_H

#include <stdbool.h>
#include <stddef.h>

#include "cwick.h"
#include "json.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The checkpoint's own markers, as ids (added_tokens.json). The engine needs
 * the two think ones for its budget mechanism, and the server needs the eos
 * to stop a turn. */
#define MM_TOK_BOD        200034   /* ]~!b[ */
#define MM_TOK_BOS        200019   /* ]~b]  */
#define MM_TOK_EOS        200020   /* [e~[  */
#define MM_TOK_NS         200058   /* ]<]minimax[>[ */
#define MM_TOK_THINK_OPEN 200059   /* <mm:think>  */
#define MM_TOK_THINK_CLOSE 200060  /* </mm:think> */

typedef struct {
    /* false makes the assistant header open with `</mm:think>` so the turn
     * answers instead of reasoning -- the template's own way of saying
     * "no thinking for this one". */
    bool thinking;
    /* A number for the `</effort>` element; 0 leaves it out. */
    int reasoning_effort;
} mm_prompt_opts;

void mm_prompt_opts_default(mm_prompt_opts *opts);

/* Render an OpenAI-shaped `messages` array into the checkpoint's prompt
 * string. `tools` is an optional OpenAI tool array. Returns a heap buffer the
 * caller frees, or NULL with `err` filled. */
char *mm_prompt_encode(json_value *messages, json_value *tools,
                       const mm_prompt_opts *opts, size_t *out_len,
                       cwick_error *err);

/* True when text starts with the end-of-turn marker chain the model writes
 * (`]<]minimax[>[<tool_call>`): the hook a future tool-call parser hangs on. */
bool mm_prompt_is_toolcall(const char *text);

#ifdef __cplusplus
}
#endif

#endif /* CWICK_MM_PROMPT_H */
