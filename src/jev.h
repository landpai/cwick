/*
 * jev.h — the decision endpoint: turn one prefill into typed answers.
 *
 * A request carries a `state` (the material to judge) and a set of questions
 * about it. Each question is one of:
 *
 *   choice  pick one key, get a distribution over all keys
 *   score   pick one level, get the probability-weighted level index
 *   noul    one statement, two named sides, get P(the statement holds)
 *
 * Nothing is generated. One prefill reaches the answer position and the
 * candidate letters' logits there are the answer: a softmax over them is the
 * distribution, and the JSON is assembled in C. The model, tokenizer and
 * engine are the ones already resident, so a decision costs one prefill and
 * adds no second model, no second process and no fine-tuning.
 *
 * Cost model: one prefill per question. Within a question every option is
 * read from the same forward pass, so a nine-cell board costs the same as a
 * two-way choice. Ten questions cost ten prefills, which is why the endpoint
 * is for a handful of questions about one state, not for a ticket queue.
 *
 * Adaptation: a decision is only as good as the prompt format the resident
 * checkpoint was trained on, and the three supported families do not share
 * one. `jev_adapter` is that per-checkpoint answer -- which encoder builds the
 * turn, which spelling of "A" the tokenizer makes one token, how many options
 * are readable. It is derived from the live engine spec on every request (and
 * re-probed by the server after a model switch), so
 * `POST /v1/models/switch` also switches the LLM interface underneath this
 * endpoint; nothing here is cached across a switch.
 */
#ifndef CWICK_JEV_H
#define CWICK_JEV_H

#include <stddef.h>
#include <stdint.h>

#include "cwick.h"

#ifdef __cplusplus
extern "C" {
#endif

#define JEV_MAX_QUESTIONS 10
#define JEV_MAX_OPTIONS 26      /* A..Z */
/* The two byte caps are the reference protocol's, not ours: a state is text
 * and its size is the caller's business. The one we cannot match is the option
 * count -- the readout is one letter token, and a tokenizer has 26 of them. */
#define JEV_MAX_STATE_BYTES 4194304        /* 4 MiB, as published */
#define JEV_MAX_INSTRUCTIONS_BYTES 65536   /* 64 KiB of question text */
#define JEV_MIN_SCORE_LEVELS 2
#define JEV_MAX_SCORE_LEVELS 10

/* The per-checkpoint half of a decision: which encoder renders the turn and
 * how many options the tokenizer can name with a single token.
 *
 * `family` and `prompt` come from the engine spec alone and are always known;
 * `letters` needs the tokenizer, so it is -1 until jev_probe() has run. The
 * server probes at startup and again after every model switch, and reports
 * the result both in the log and at GET /v1/systemone -- that is how a caller
 * can tell which interface a switch selected instead of assuming one. */
typedef struct {
    const char *family;     /* "glm" | "deepseek" | "minimax" */
    const char *prompt;     /* encoder used for the decision turn */
    int letters;            /* A.. resolvable as one token, contiguous; -1 unknown */
    int first_missing;      /* first letter that is not one token, -1 if none */
    int space_lead;         /* 1 when the single-token spelling is " A" */
} jev_adapter;

/* Family name for a spec. Pure: no tokenizer, no engine, safe on any spec. */
const char *jev_family_name(const cwick_model_spec *spec);

/* Fill the spec-derived half of an adapter (letters = -1). */
void jev_adapter_default(const cwick_model_spec *spec, jev_adapter *out);

/* Resolve the option letters against a tokenizer. Returns 0 with `out` filled;
 * -1 with `err` set when not one letter is a single token (a tokenizer that
 * cannot spell an option cannot answer a choice).
 */
int jev_probe(const cwick_model_spec *spec, const cwick_tokenizer *tok,
              jev_adapter *out, char *err, size_t errcap);

/* The adapter as a JSON object, for GET /v1/systemone. `*out_json` is heap. */
int jev_describe(cwick_engine *eng, const cwick_tokenizer *tok,
                 char **out_json, char *err, size_t errcap);

/* A request may name the resident model by served alias or by model_type;
 * anything else is a stale name from before a switch. Empty/NULL means "the
 * resident one". Comparison is case-insensitive, like the chat endpoint. */
bool jev_model_is_served(const char *requested, const char *served,
                         const char *model_type);

/* Answer a `POST /v1/systemone` body.
 *
 * `eng` must be idle (the caller holds the generation slot): the function
 * resets the engine between questions, so it cannot share a live turn.
 * On success `*out_json` is a heap string holding the whole response object
 * and `*in_tokens` is the total number of prompt tokens prefilled across the
 * questions. On failure it returns a negative code with `err` filled and
 * leaves `*out_json` NULL:
 *
 *   -1  the request is well-formed but cannot be served (a stale model name,
 *       a limit exceeded, an unsupported field, a tokenizer without letters,
 *       a failed prefill)                                   -> HTTP 400
 *   -2  the body is not a valid decision request (malformed JSON, a missing
 *       required field, an invalid question)                -> HTTP 422
 *
 * The split is the reference protocol's: 400 means "over a limit or
 * unsupported", 422 means "this is not that document at all".
 */
int jev_decide(cwick_engine *eng, const cwick_tokenizer *tok,
               const char *body, size_t body_len,
               char **out_json, int64_t *in_tokens,
               char *err, size_t errcap);

#ifdef __cplusplus
}
#endif

#endif /* CWICK_JEV_H */
