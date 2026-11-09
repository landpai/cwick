/*
 * ds_prompt.h — DeepSeek-V4.1 prompt encoding, in C.
 *
 * The checkpoint ships its prompt format as a standalone Python module
 * (`encoding/encoding.py`) plus a README that specifies the byte layout. cwick
 * must produce the same bytes without a Python runtime, so this is a direct
 * port of the text path: BOS, system / mid-conversation system, user turns,
 * assistant turns with reasoning and DSML tool calls, `latest_reminder`, tool
 * results folded into the user turn, the tool schema block, the numeric
 * reasoning-effort prefix and the assistant generation header.
 *
 * Out of scope, matching the README's own split: image content blocks (the
 * vision tower is not loaded; an image block is rejected rather than silently
 * dropped) and `parse_message_from_completion_text` (the server parses output
 * with its own streaming parser).
 */
#ifndef CWICK_DS_PROMPT_H
#define CWICK_DS_PROMPT_H

#include <stdbool.h>
#include <stddef.h>

#include "cwick.h"
#include "json.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DS_THINKING_CHAT = 0,   /* `</think>` right after the assistant header */
    DS_THINKING_THINKING,   /* the model writes a reasoning block first */
} ds_thinking_mode;

typedef struct {
    ds_thinking_mode thinking;
    /* Keep reasoning_content from earlier turns. The reference turns this off
     * by itself whenever any message carries a `tools` declaration. */
    bool drop_thinking;
    bool add_bos;
    /* 1..100, the numeric budget the README specifies; 0 means "unset" and
     * resolves to the default effort (high, 75). */
    int reasoning_effort;
} ds_prompt_opts;

/* Fill `opts` with the reference defaults: thinking mode, drop_thinking on,
 * BOS on, effort unset. */
void ds_prompt_opts_default(ds_prompt_opts *opts);

/* Render an OpenAI-shaped `messages` array into the V4.1 prompt string.
 *
 *   messages  a JSON_ARRAY of message objects (role / content / tool_calls /
 *             reasoning_content / task / wo_eos ...)
 *   tools     an optional JSON_ARRAY in OpenAI tool format. The reference
 *             carries this on the system message; servers pass it beside
 *             `messages`, so when it is present it is rendered into the system
 *             turn the way the reference does, synthesizing one if the
 *             conversation has none.
 *   out_len   receives the byte length (the buffer is NUL-terminated too).
 *
 * Returns a heap buffer the caller frees, or NULL on error with `err` filled. */
char *ds_prompt_encode(const json_value *messages, const json_value *tools,
                       const ds_prompt_opts *opts, size_t *out_len,
                       cwick_error *err);

#ifdef __cplusplus
}
#endif

#endif /* CWICK_DS_PROMPT_H */
