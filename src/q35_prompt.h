/*
 * q35_prompt.h — Qwen3.5 chat rendering, in C.
 *
 * The checkpoint ships `chat_template.jinja`; this is a port of its text path.
 * The markers come from added_tokens.json, and the two thinking ones are also
 * what the engine's budget mechanism writes, so they live here as ids.
 *
 * Out of scope, matching what this port can run: image, video and audio parts
 * (the vision tower is not loaded, so a non-text part is dropped rather than
 * silently becoming text), and the `add_vision_id` variant of the template.
 */
#ifndef CWICK_Q35_PROMPT_H
#define CWICK_Q35_PROMPT_H

#include <stdbool.h>
#include <stddef.h>

#include "cwick.h"
#include "json.h"

#ifdef __cplusplus
extern "C" {
#endif

#define Q35_TOK_EOS         248044   /* <|endoftext|> */
#define Q35_TOK_IM_START    248045   /* <|im_start|>  */
#define Q35_TOK_IM_END      248046   /* <|im_end|>    */
#define Q35_TOK_THINK_OPEN  248068   /* <think>       */
#define Q35_TOK_THINK_CLOSE 248069   /* </think>      */
#define Q35_TOK_VISION_PAD  248055   /* <|vision_pad|>, also the pad token */
#define Q35_TOK_IMAGE_PAD   248056

typedef struct {
    /* false makes the assistant header open with an empty, already-closed
     * reasoning block (`<think>\n\n</think>\n\n`) -- the template's own way of
     * saying "answer without thinking". */
    bool thinking;
} q35_prompt_opts;

void q35_prompt_opts_default(q35_prompt_opts *opts);

/* Render an OpenAI-shaped `messages` array into the checkpoint's prompt text.
 * `tools` is an optional OpenAI tool array. Returns a heap buffer the caller
 * frees, or NULL with `err` filled. */
char *q35_prompt_encode(json_value *messages, json_value *tools,
                        const q35_prompt_opts *opts, size_t *out_len,
                        cwick_error *err);

/* True when `text` opens with the tool-call marker this template asks for. */
bool q35_prompt_is_toolcall(const char *text);

#ifdef __cplusplus
}
#endif

#endif /* CWICK_Q35_PROMPT_H */
