/*
 * ik_prompt.h — Inkling-2-Mini chat rendering in pure C.
 *
 * The checkpoint's template is line-oriented: every message is one
 * `<|message_role|>...<|end_message|>` record, with an extra thinking-effort
 * system record before the first non-system turn, and one placeholder token
 * per image. This renderer produces exactly that shape from the raw OpenAI /
 * Responses / Anthropic messages array.
 */
#ifndef CWICK_IK_PROMPT_H
#define CWICK_IK_PROMPT_H

#include <stddef.h>
#include <stdint.h>

#include "cwick.h"

struct json_value;

#define IK_MAX_IMAGES 16
#define IK_IMAGE_PLACEHOLDER "<|unused_200054|>"

/* Render and tokenize. When `images` is non-NULL each image part contributes
 * one owned base64 string, in prompt order; release them with
 * ik_prompt_free_images(). The caller decodes them and hands the resulting
 * embeddings to cwick_engine_set_images(). */
/* Render only, for the decision endpoint which reads one position's logits. */
char *ik_prompt_render(const char *messages_json, const char *tools_json,
                       const char *effort_level, char ***images, size_t *n_images,
                       cwick_error *err);
char *ik_prompt_encode_text(const char *messages_json, const char *tools_json,
                            cwick_error *err);
char *ik_prompt_encode_decision(const char *messages_json, const char *tools_json,
                                cwick_error *err);

cwick_status ik_prompt_encode(const cwick_tokenizer *tok,
                              const char *messages_json,
                              const char *tools_json,
                              const char *effort_level,
                              char ***images, size_t *n_images,
                              uint32_t **ids, int64_t *n_out,
                              cwick_error *err);
cwick_status ik_prompt_encode_decision_ids(const cwick_tokenizer *tok,
                                           const char *messages_json,
                                           const char *tools_json,
                                           uint32_t **ids, int64_t *n_out,
                                           cwick_error *err);
/* Decision encoding with owned image payloads retained for vision encoding. */
cwick_status ik_prompt_encode_decision_images(const cwick_tokenizer *tok,
                                              const char *messages_json,
                                              const char *tools_json,
                                              char ***images, size_t *n_images,
                                              uint32_t **ids, int64_t *n_out,
                                              cwick_error *err);
/* Resolve the checkpoint's actual placeholder ID through its tokenizer. */
cwick_status ik_prompt_image_token_id(const cwick_tokenizer *tok,
                                      uint32_t *id_out, cwick_error *err);
/* Normalize top-level systemone `images` / `image` input into owned base64. */
cwick_status ik_prompt_normalize_images(const struct json_value *root,
                                        char ***images, size_t *n_images,
                                        cwick_error *err);
void ik_prompt_free_images(char **images, size_t n);

typedef struct {
    char *name;
    char *arguments_json;
} ik_output_call;

typedef struct {
    char *reasoning;
    char *content;
    ik_output_call *calls;
    size_t n_calls;
} ik_output;

cwick_status ik_output_parse(const char *text, ik_output *out, cwick_error *err);
void ik_output_free(ik_output *out);

#endif
