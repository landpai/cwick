/*
 * jev.c — decision endpoint internals. See jev.h for the contract.
 *
 * The whole file exists to answer one question honestly: how do you get a
 * calibrated-looking distribution out of a model that was not trained to give
 * you one? The answer here is to stop asking it to say anything. A letter
 * position in the prompt has a distribution over the vocabulary; restricted to
 * the option letters and normalised, that is a distribution over the options.
 * No decode loop, no JSON to parse, no retry when the model writes prose
 * instead of a letter.
 *
 * The catch is the prior: a letter token has a frequency of its own, and that
 * frequency attaches to whatever option sits in that slot. This file does not
 * make the prior vanish -- it is a property of the model -- but it keeps the
 * prompt identical for every option and reports the whole distribution, so a
 * caller can see a flat or skewed one instead of only its argmax.
 */
#include "jev.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/time.h>

#include "ds_prompt.h"
#include "json.h"
#include "log.h"
#include "mm_prompt.h"
#include "models.h"
#include "tools.h"
#include "util.h"


/* The util layer only guarantees malloc/realloc; calloc and strdup are
 * spelled out here so the allocation sites read the same as everywhere else. */
static void *jev_calloc(size_t n, size_t sz) {
    void *p = cwick_xmalloc(n * sz);
    memset(p, 0, n * sz);
    return p;
}

static char *jev_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = cwick_xmalloc(n);
    memcpy(p, s, n);
    return p;
}

/* ------------------------------------------------------------------ */
/* small dynamic string                                                */
/* ------------------------------------------------------------------ */
typedef struct {
    char *p;
    size_t len, cap;
} jb;

static void jb_reserve(jb *b, size_t extra) {
    if (b->len + extra + 1 <= b->cap) return;
    size_t cap = b->cap ? b->cap : 256;
    while (cap < b->len + extra + 1) cap *= 2;
    b->p = cwick_xrealloc(b->p, cap);
    b->cap = cap;
}

static void jb_put(jb *b, const char *s) {
    size_t n = strlen(s);
    jb_reserve(b, n);
    memcpy(b->p + b->len, s, n);
    b->len += n;
    b->p[b->len] = '\0';
}

static void jb_putf(jb *b, const char *fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n > 0) {
        jb_reserve(b, (size_t)n);
        vsnprintf(b->p + b->len, (size_t)n + 1, fmt, ap2);
        b->len += (size_t)n;
    }
    va_end(ap2);
}

static char *jb_take(jb *b) {
    if (!b->p) return jev_strdup("");
    char *p = b->p;
    b->p = NULL;
    b->len = b->cap = 0;
    return p;
}

/* ------------------------------------------------------------------ */
/* request model                                                       */
/* ------------------------------------------------------------------ */
typedef enum { JEV_CHOICE = 0, JEV_SCORE = 1, JEV_NOUL = 2 } jev_kind;

typedef struct {
    char *name;
    jev_kind kind;
    char *instructions;
    char **keys;      /* choice: option keys; score: "0".."n-1"; noul: yes/no */
    char **descs;     /* parallel to keys; may be NULL for score levels */
    size_t n;
} jev_q;

typedef struct {
    char *state;
    char *model;
    double temperature;
    jev_q *q;
    size_t nq;
} jev_req;

static void jev_q_free(jev_q *q) {
    free(q->name);
    free(q->instructions);
    for (size_t i = 0; i < q->n; i++) {
        free(q->keys[i]);
        free(q->descs[i]);
    }
    free(q->keys);
    free(q->descs);
}

static void jev_req_free(jev_req *r) {
    free(r->state);
    free(r->model);
    for (size_t i = 0; i < r->nq; i++) jev_q_free(&r->q[i]);
    free(r->q);
    memset(r, 0, sizeof(*r));
}

static int fail_code(int code, char *err, size_t cap, const char *fmt,
                     va_list ap) {
    if (err && cap) vsnprintf(err, cap, fmt, ap);
    return code;
}

/* Two refusal classes, as the reference protocol defines them: "over a limit
 * or unsupported" is a 400, "this is not a decision request" is a 422. */
static int fail(char *err, size_t cap, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int rc = fail_code(-1, err, cap, fmt, ap);
    va_end(ap);
    return rc;
}

static int invalid(char *err, size_t cap, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int rc = fail_code(-2, err, cap, fmt, ap);
    va_end(ap);
    return rc;
}

static char *dup_str(const char *s) {
    return jev_strdup(s ? s : "");
}

/* One option list. `criteria` is either an object (key -> description) or an
 * array of names; both shapes appear in the protocol, and a score question
 * uses the array form for its levels.
 *
 * `noul` is not an option list at all: its answer is one probability, that the
 * statement holds. Its two candidates are therefore fixed here -- the true
 * side first, the false side second -- instead of being taken from the order
 * the JSON members happened to arrive in. That order *was* the answer before
 * this: a client that wrote `{"false": …, "true": …}` got P(false) reported
 * under the key `noul`, and nothing about the response said so. The criteria
 * text is what the prompt shows; the default is the protocol's yes/no. */
static int parse_options(json_value *crit, jev_q *q, const char *qname,
                         char *err, size_t cap) {
    if (q->kind == JEV_NOUL) {
        const char *yes = "yes", *no = "no";
        if (crit) {
            if (crit->kind == JSON_OBJECT) {
                if (crit->count > 2)
                    return invalid(err, cap,
                                   "question %s: noul criteria takes exactly "
                                   "{\"true\": …, \"false\": …}", qname);
                json_value *t = cwick_json_get(crit, "true");
                json_value *f = cwick_json_get(crit, "false");
                if (!t && !f)
                    return invalid(err, cap, "question %s: noul criteria must "
                                   "name `true` and `false`", qname);
                if ((t && t->kind != JSON_STRING) ||
                    (f && f->kind != JSON_STRING))
                    return invalid(err, cap, "question %s: noul criteria values "
                                   "must be strings", qname);
                if (t && t->string && t->string[0]) yes = t->string;
                if (f && f->string && f->string[0]) no = f->string;
            } else if (crit->kind == JSON_ARRAY) {
                if (crit->count != 2 || !crit->items[0] || !crit->items[1] ||
                    crit->items[0]->kind != JSON_STRING ||
                    crit->items[1]->kind != JSON_STRING)
                    return invalid(err, cap, "question %s: noul criteria takes "
                                   "two strings, the true side then the false "
                                   "side", qname);
                if (crit->items[0]->string[0]) yes = crit->items[0]->string;
                if (crit->items[1]->string[0]) no = crit->items[1]->string;
            } else {
                return invalid(err, cap, "question %s: noul criteria must be an "
                               "object or an array", qname);
            }
        }
        q->keys = jev_calloc(2, sizeof(char *));
        q->descs = jev_calloc(2, sizeof(char *));
        q->keys[0] = dup_str("true");
        q->keys[1] = dup_str("false");
        q->descs[0] = dup_str(yes);
        q->descs[1] = dup_str(no);
        q->n = 2;
        return 0;
    }

    size_t n = 0;
    if (!crit)
        return invalid(err, cap, "question %s: %s needs criteria", qname,
                       q->kind == JEV_CHOICE ? "choice" : "score");
    if (crit->kind == JSON_OBJECT) {
        if (q->kind == JEV_SCORE)
            return invalid(err, cap, "question %s: score criteria is an ordered "
                           "array of level descriptions, lowest first", qname);
        n = crit->count;
    } else if (crit->kind == JSON_ARRAY) {
        n = crit->count;
    } else {
        return invalid(err, cap,
                       "question %s: criteria must be an object or an array",
                       qname);
    }

    if (n == 0) return invalid(err, cap, "question %s: criteria is empty", qname);
    if (q->kind == JEV_CHOICE && n > JEV_MAX_OPTIONS)
        return fail(err, cap, "question %s: %zu options exceed the %d reads "
                    "this endpoint can rank (one letter token per option)",
                    qname, n, JEV_MAX_OPTIONS);
    if (q->kind == JEV_SCORE &&
        (n < JEV_MIN_SCORE_LEVELS || n > JEV_MAX_SCORE_LEVELS))
        return fail(err, cap, "question %s: %zu levels; a score question takes "
                    "%d to %d ordered levels", qname, n, JEV_MIN_SCORE_LEVELS,
                    JEV_MAX_SCORE_LEVELS);

    q->keys = jev_calloc(n, sizeof(char *));
    q->descs = jev_calloc(n, sizeof(char *));
    q->n = n;
    for (size_t i = 0; i < n; i++) {
        if (crit->kind == JSON_OBJECT) {
            q->keys[i] = dup_str(crit->members[i].key);
            json_value *v = crit->members[i].value;
            q->descs[i] = dup_str(v && v->kind == JSON_STRING ? v->string : "");
        } else {
            json_value *v = crit->items[i];
            char idx[16];
            snprintf(idx, sizeof(idx), "%zu", i);
            q->keys[i] = dup_str(q->kind == JEV_SCORE ? idx
                                : (v && v->kind == JSON_STRING ? v->string : idx));
            q->descs[i] = dup_str(v && v->kind == JSON_STRING ? v->string : "");
        }
    }
    /* A score question's keys are its indices and its descriptions are the
     * level labels, so the response can carry a legend without re-reading the
     * request. An empty description would make the legend useless. */
    if (q->kind == JEV_SCORE) {
        for (size_t i = 0; i < n; i++)
            if (!q->descs[i][0]) {
                char *k = q->descs[i];
                q->descs[i] = dup_str(q->keys[i]);
                free(k);
            }
    }
    return 0;
}

static int parse_request(json_value *root, jev_req *r, char *err, size_t cap) {
    if (!root || root->kind != JSON_OBJECT)
        return invalid(err, cap, "body must be an object");

    /* Fields this endpoint cannot honour are refused by name, not ignored. A
     * decision has no token stream and no reasoning pass, so accepting
     * `stream:true` or `reasoning_effort:"high"` and answering with one JSON
     * object would answer a question the caller did not ask. */
    json_value *stream = cwick_json_get(root, "stream");
    if (stream && !(stream->kind == JSON_BOOL && !stream->boolean))
        return fail(err, cap, "body.stream is not supported: a decision is one "
                    "JSON object, never a stream of events");
    json_value *effort = cwick_json_get(root, "reasoning_effort");
    if (effort) {
        const char *v = cwick_json_str(effort);
        if (!v || strcmp(v, "none") != 0)
            return fail(err, cap, "body.reasoning_effort must be \"none\": a "
                        "decision reads one distribution and generates nothing");
    }
    json_value *think = cwick_json_get(root, "thinking");
    if (think) {
        bool off = think->kind == JSON_BOOL && !think->boolean;
        if (think->kind == JSON_OBJECT) {
            /* The chat endpoint's own spelling, so a client can send the same
             * block it sends there -- as long as it says "off". */
            const char *t = cwick_json_str(cwick_json_get(think, "type"));
            off = t && strcmp(t, "disabled") == 0;
        }
        if (!off)
            return fail(err, cap, "body.thinking must be off: a decision reads "
                        "one distribution, there is no reasoning pass");
    }
    if (cwick_json_get(root, "image") || cwick_json_get(root, "images"))
        return fail(err, cap, "body.images is not supported: the readout is "
                    "one letter token, so the decision is text only");

    /* `state` is the material to judge. The reference takes a string as-is and
     * serializes an object; every other JSON value is serialized the same way,
     * so a caller that hands over its state as JSON gets it inside <state>
     * instead of a parse error. */
    json_value *state = cwick_json_get(root, "state");
    if (!state || state->kind == JSON_NULL)
        return invalid(err, cap, "body.state is required");
    if (state->kind == JSON_STRING) {
        if (!state->string || !state->string[0])
            return invalid(err, cap, "body.state is empty");
        if (strlen(state->string) > JEV_MAX_STATE_BYTES)
            return fail(err, cap, "body.state exceeds %d bytes",
                        JEV_MAX_STATE_BYTES);
        r->state = dup_str(state->string);
    } else {
        char *text = cwick_json_write(state);
        if (!text) return invalid(err, cap, "body.state cannot be serialized");
        if (strlen(text) > JEV_MAX_STATE_BYTES) {
            free(text);
            return fail(err, cap, "body.state exceeds %d bytes",
                        JEV_MAX_STATE_BYTES);
        }
        r->state = text;
    }

    json_value *model = cwick_json_get(root, "model");
    if (model && model->kind == JSON_STRING) r->model = dup_str(model->string);

    r->temperature = 1.0;
    json_value *t = cwick_json_get(root, "temperature");
    if (t && t->kind == JSON_NUMBER) {
        if (!(t->number > 0.0) || t->number > 100.0)
            return fail(err, cap, "body.temperature must be in (0, 100]");
        r->temperature = t->number;
    }

    json_value *qs = cwick_json_get(root, "questions");
    if (!qs || qs->kind != JSON_OBJECT)
        return invalid(err, cap, "body.questions must be an object");
    if (qs->count == 0) return invalid(err, cap, "body.questions is empty");
    if (qs->count > JEV_MAX_QUESTIONS)
        return fail(err, cap, "%zu questions exceed the limit of %d", qs->count,
                    JEV_MAX_QUESTIONS);

    r->q = jev_calloc(qs->count, sizeof(jev_q));
    for (size_t i = 0; i < qs->count; i++) {
        const char *qname = qs->members[i].key;
        json_value *qv = qs->members[i].value;
        jev_q *q = &r->q[i];
        r->nq = i + 1;
        q->name = dup_str(qname);
        if (!qv || qv->kind != JSON_OBJECT)
            return invalid(err, cap, "question %s must be an object", qname);

        json_value *type = cwick_json_get(qv, "type");
        const char *tname = type && type->kind == JSON_STRING ? type->string : NULL;
        if (!tname)
            return invalid(err, cap, "question %s: `type` is required "
                           "(choice, score or noul)", qname);
        if (strcmp(tname, "choice") == 0) q->kind = JEV_CHOICE;
        else if (strcmp(tname, "score") == 0) q->kind = JEV_SCORE;
        else if (strcmp(tname, "noul") == 0) q->kind = JEV_NOUL;
        else return invalid(err, cap, "question %s: unknown type %s", qname, tname);

        /* The question itself, in plain language. Required: the old fallback
         * text ("Which option applies?") was a silent substitution, and a
         * decision asked with the wrong question is worse than a refusal. */
        json_value *ins = cwick_json_get(qv, "instructions");
        if (!ins || ins->kind != JSON_STRING || !ins->string || !ins->string[0])
            return invalid(err, cap, "question %s: `instructions` is required",
                           qname);
        if (strlen(ins->string) > JEV_MAX_INSTRUCTIONS_BYTES)
            return fail(err, cap, "question %s: instructions exceed %d bytes",
                        qname, JEV_MAX_INSTRUCTIONS_BYTES);
        q->instructions = dup_str(ins->string);

        int prc = parse_options(cwick_json_get(qv, "criteria"), q, qname, err, cap);
        if (prc != 0) return prc;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* rendering                                                           */
/* ------------------------------------------------------------------ */
static const char *JEV_SYSTEM =
    "You are a decision engine. Read the state, then answer the question by "
    "choosing exactly one option. Reply with the option's uppercase letter and "
    "nothing else: no explanation, no punctuation, no JSON.";

/* The user turn. Every option is rendered the same way and the prompt ends at
 * the answer cue, so the only thing that differs between options is the letter
 * the readout looks at -- not the wording in front of it. */
static char *render_user(const jev_req *r, const jev_q *q) {
    jb b = {0};
    jb_put(&b, "<state>\n");
    jb_put(&b, r->state);
    jb_put(&b, "\n</state>\n\nQuestion: ");
    jb_put(&b, q->instructions);
    jb_put(&b, "\n\nOptions:\n");
    for (size_t i = 0; i < q->n; i++) {
        char letter = (char)('A' + i);
        if (q->kind == JEV_CHOICE) {
            jb_putf(&b, "%c. %s", letter, q->keys[i]);
            if (q->descs[i] && q->descs[i][0]) jb_putf(&b, ": %s", q->descs[i]);
            jb_put(&b, "\n");
        } else {
            /* Score levels are descriptions with an index for a key, and a
             * noul's two sides are descriptions with "true"/"false" for a key.
             * In both cases the description is the only thing worth showing. */
            jb_putf(&b, "%c. %s\n", letter, q->descs[i][0] ? q->descs[i] : q->keys[i]);
        }
    }
    jb_put(&b, "\nAnswer with one letter: ");
    for (size_t i = 0; i < q->n; i++) {
        if (i) jb_put(&b, ", ");
        jb_putf(&b, "%c", (char)('A' + i));
    }
    jb_put(&b, ".\nAnswer:");
    return jb_take(&b);
}

/* ------------------------------------------------------------------ */
/* encoding, per prompt format                                         */
/* ------------------------------------------------------------------ */
/* Three families, three chat grammars. The test is on model_type because
 * that is the one field every checkpoint declares about itself and the one
 * the engine has already validated to pick its forward pass. The fallback is
 * the GLM grammar rather than an error: an unknown family that still speaks
 * `<|assistant|>` gets a working readout, and one that speaks something else
 * fails inside the encoder with a tokenizer error the caller can act on
 * instead of being refused by this endpoint for a name it did not choose. */
typedef enum { JEV_FAMILY_GLM = 0, JEV_FAMILY_DS = 1, JEV_FAMILY_MM = 2 } jev_family;

static jev_family family_of(const cwick_model_spec *spec) {
    const char *t = spec && spec->model_type[0] ? spec->model_type : "";
    if (strncasecmp(t, "deepseek", 8) == 0) return JEV_FAMILY_DS;
    if (strncasecmp(t, "minimax", 7) == 0) return JEV_FAMILY_MM;
    return JEV_FAMILY_GLM;
}

const char *jev_family_name(const cwick_model_spec *spec) {
    switch (family_of(spec)) {
    case JEV_FAMILY_DS: return "deepseek";
    case JEV_FAMILY_MM: return "minimax";
    default: return "glm";
    }
}

void jev_adapter_default(const cwick_model_spec *spec, jev_adapter *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    out->family = jev_family_name(spec);
    /* -1 is "nothing missing", which is also the honest answer before the
     * tokenizer has been asked. */
    out->letters = -1;
    out->first_missing = -1;
    switch (family_of(spec)) {
    case JEV_FAMILY_DS:
        /* ds_prompt_encode closes the reasoning block on the assistant header
         * (DS_THINKING_CHAT), so the answer position is the next token. */
        out->prompt = "deepseek-chat-closed";
        break;
    case JEV_FAMILY_MM:
        out->prompt = "minimax-chat-plain";
        break;
    default:
        /* render_chat(..., answer_at_once=true): `<|assistant|><think></think>`. */
        out->prompt = "glm-chat-closed";
        break;
    }
}

/* Build the OpenAI-shaped messages array the two JSON-driven prompt encoders
 * take. The strings are escaped by the same routine that escapes SSE payloads,
 * so a state containing quotes cannot break out of the array. */
static json_value *messages_json(const char *system, const char *user, cwick_error *err) {
    char *es = cwick_json_stringify(system);
    char *eu = cwick_json_stringify(user);
    size_t cap = strlen(es) + strlen(eu) + 128;
    char *text = cwick_xmalloc(cap);
    snprintf(text, cap,
             "[{\"role\":\"system\",\"content\":%s},"
             "{\"role\":\"user\",\"content\":%s}]", es, eu);
    free(es);
    free(eu);
    json_value *arr = cwick_json_parse(text, strlen(text), err);
    free(text);
    return arr;
}

static cwick_status encode_prompt(const cwick_tokenizer *tok,
                                  const cwick_model_spec *spec,
                                  const char *system, const char *user,
                                  uint32_t **ids, int64_t *n, cwick_error *err) {
    switch (family_of(spec)) {
    case JEV_FAMILY_DS:
    case JEV_FAMILY_MM: {
        json_value *msgs = messages_json(system, user, err);
        if (!msgs) return CWICK_ERR_PARSE;
        char *text = NULL;
        if (family_of(spec) == JEV_FAMILY_DS) {
            ds_prompt_opts o;
            ds_prompt_opts_default(&o);
            /* The thinking block is closed right after the assistant header:
             * the next token is the answer, which is the one position this
             * endpoint reads. */
            o.thinking = DS_THINKING_CHAT;
            text = ds_prompt_encode(msgs, NULL, &o, NULL, err);
        } else {
            mm_prompt_opts o;
            mm_prompt_opts_default(&o);
            o.thinking = false;
            text = mm_prompt_encode(msgs, NULL, &o, NULL, err);
        }
        cwick_json_free(msgs);
        if (!text) return CWICK_ERR_PARSE;
        cwick_status st = cwick_tokenizer_encode(tok, text, ids, n, err);
        free(text);
        return st;
    }
    default:
        return cwick_chat_encode_decision(tok, system, user, ids, n, err);
    }
}

/* ------------------------------------------------------------------ */
/* readout                                                             */
/* ------------------------------------------------------------------ */
/* The token that stands for option `i` at the answer position. Two spellings
 * are possible depending on the tokenizer: with a leading space (the answer
 * cue ends at a colon and the model writes " A") or bare. Whichever is a
 * single token is the one the model can actually emit there; a spelling that
 * costs two tokens is not a candidate at all. `*lead` reports which spelling
 * won, which is what the adapter publishes. */
static int letter_token_ex(const cwick_tokenizer *tok, size_t i, int *lead) {
    char forms[2][3];
    forms[0][0] = ' ';
    forms[0][1] = (char)('A' + i);
    forms[0][2] = '\0';
    forms[1][0] = (char)('A' + i);
    forms[1][1] = '\0';
    if (lead) *lead = 0;
    for (int f = 0; f < 2; f++) {
        cwick_error e;
        cwick_error_clear(&e);
        uint32_t *ids = NULL;
        int64_t n = 0;
        cwick_status st = cwick_tokenizer_encode(tok, forms[f], &ids, &n, &e);
        if (st == CWICK_OK && n == 1) {
            int id = (int)ids[0];
            free(ids);
            if (lead) *lead = f == 0;
            return id;
        }
        free(ids);
    }
    return -1;
}

static int letter_token(const cwick_tokenizer *tok, size_t i) {
    return letter_token_ex(tok, i, NULL);
}

int jev_probe(const cwick_model_spec *spec, const cwick_tokenizer *tok,
              jev_adapter *out, char *err, size_t errcap) {
    if (!out) return fail(err, errcap, "no adapter to fill");
    jev_adapter_default(spec, out);
    if (!tok) return fail(err, errcap, "no resident tokenizer");
    int letters = 0, first_missing = -1;
    for (size_t i = 0; i < JEV_MAX_OPTIONS; i++) {
        int lead = 0;
        int id = letter_token_ex(tok, i, &lead);
        if (id < 0) {
            if (first_missing < 0) first_missing = (int)i;
            continue;
        }
        if (first_missing < 0) {
            letters++;
            if (i == 0) out->space_lead = lead;
        }
    }
    out->letters = letters;
    out->first_missing = first_missing;
    if (letters < 2)
        return fail(err, errcap,
                    "this tokenizer spells fewer than two option letters as a "
                    "single token; the decision endpoint needs one token per option");
    return 0;
}

static void softmax_over(const float *logits, const int *ids, size_t n,
                         double temp, double *p) {
    double mx = -INFINITY;
    for (size_t i = 0; i < n; i++) {
        double v = (double)logits[ids[i]] / temp;
        p[i] = v;
        if (v > mx) mx = v;
    }
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        p[i] = exp(p[i] - mx);
        sum += p[i];
    }
    if (sum > 0.0) for (size_t i = 0; i < n; i++) p[i] /= sum;
}

/* How concentrated the distribution is, on a 0..1 scale where a uniform
 * distribution is 0 and a point mass is 1. It is not a probability of being
 * right and it is not calibrated: it answers "how sure is the model about this
 * choice", which is a different question from "how often is it correct". */
static double concentration(const double *p, size_t n) {
    if (n <= 1) return 1.0;
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += p[i] * p[i];
    double v = (n * s - 1.0) / (double)(n - 1);
    return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
}

/* ------------------------------------------------------------------ */
/* identity                                                            */
/* ------------------------------------------------------------------ */
/* The public name of the resident checkpoint. The catalog alias is what
 * /v1/models advertises and what an app caches; `model_type` is the fallback
 * for a checkpoint the catalog has never seen. */
static void served_name(const cwick_engine *eng, char *out, size_t cap) {
    const cwick_model_spec *spec = cwick_engine_spec(eng);
    out[0] = '\0';
    if (cwick_model_served_alias(cwick_engine_model_root(eng), out, cap) &&
        out[0])
        return;
    snprintf(out, cap, "%s", spec && spec->model_type[0] ? spec->model_type : "");
}

bool jev_model_is_served(const char *requested, const char *served,
                         const char *model_type) {
    if (!requested || !requested[0]) return true;   /* resident one */
    if (served && served[0] && strcasecmp(requested, served) == 0) return true;
    if (model_type && model_type[0] && strcasecmp(requested, model_type) == 0)
        return true;
    return false;
}

static void put_adapter(jb *b, const jev_adapter *a) {
    jb_putf(b, "{\"family\":\"%s\",\"prompt\":\"%s\",\"readout\":\"letter\","
               "\"letters\":%d,\"letters_total\":%d,\"first_missing\":",
           a->family, a->prompt, a->letters, JEV_MAX_OPTIONS);
    if (a->first_missing < 0) jb_put(b, "null");
    else jb_putf(b, "%d", a->first_missing);
    jb_putf(b, ",\"space_lead\":%s}", a->space_lead ? "true" : "false");
}

int jev_describe(cwick_engine *eng, const cwick_tokenizer *tok,
                 char **out_json, char *err, size_t errcap) {
    if (out_json) *out_json = NULL;
    if (!eng || !tok) return fail(err, errcap, "no resident model");
    jev_adapter a;
    char perr[256];
    if (jev_probe(cwick_engine_spec(eng), tok, &a, perr, sizeof(perr)) != 0) {
        /* A tokenizer that cannot spell two options still has a family and a
         * prompt format worth reporting: say what is wrong and leave the rest
         * readable rather than 503-ing the discovery call. */
        fail(err, errcap, "%s", perr);
        a.letters = 0;
        a.first_missing = 0;
    }
    char served[CWICK_MODEL_ID_MAX];
    served_name(eng, served, sizeof(served));
    jb b = {0};
    char *em = cwick_json_stringify(served);
    jb_putf(&b, "{\"object\":\"systemone.adapter\",\"model\":%s,\"adapter\":", em);
    free(em);
    put_adapter(&b, &a);
    jb_putf(&b, ",\"questions_max\":%d,\"options_max\":%d,\"state_max_bytes\":%d,"
                "\"instructions_max_bytes\":%d,\"levels_min\":%d,\"levels_max\":%d,"
                "\"cost\":\"one prefill per question, no generated tokens\"}",
            JEV_MAX_QUESTIONS, JEV_MAX_OPTIONS, JEV_MAX_STATE_BYTES,
            JEV_MAX_INSTRUCTIONS_BYTES, JEV_MIN_SCORE_LEVELS,
            JEV_MAX_SCORE_LEVELS);
    if (out_json) *out_json = jb_take(&b);
    else { free(b.p); }
    return 0;
}

/* ------------------------------------------------------------------ */
/* response                                                            */
/* ------------------------------------------------------------------ */
typedef struct {
    double *p;
    int winner;
    double confidence;
    double score;
    double noul;
} jev_out;

static void put_probabilities(jb *b, const jev_q *q, const double *p) {
    jb_put(b, "{");
    for (size_t i = 0; i < q->n; i++) {
        char *k = cwick_json_stringify(q->keys[i]);
        jb_putf(b, "%s%s:%.6f", i ? "," : "", k, p[i]);
        free(k);
    }
    jb_put(b, "}");
}

int jev_decide(cwick_engine *eng, const cwick_tokenizer *tok,
               const char *body, size_t body_len,
               char **out_json, int64_t *in_tokens,
               char *err, size_t errcap) {
    *out_json = NULL;
    *in_tokens = 0;
    if (!eng || !tok) return fail(err, errcap, "no resident model");

    cwick_error e;
    cwick_error_clear(&e);
    json_value *root = cwick_json_parse(body, body_len, &e);
    if (!root)
        return invalid(err, errcap, "invalid JSON: %s", cwick_error_message(&e));

    jev_req req;
    memset(&req, 0, sizeof(req));
    {
        int prc = parse_request(root, &req, err, errcap);
        if (prc != 0) {
            cwick_json_free(root);
            jev_req_free(&req);
            return prc;
        }
    }
    cwick_json_free(root);
    struct timeval dec_t0;
    gettimeofday(&dec_t0, NULL);

    const cwick_model_spec *spec = cwick_engine_spec(eng);
    char served[CWICK_MODEL_ID_MAX];
    served_name(eng, served, sizeof(served));
    /* A name cached before a model switch is refused by name, exactly like the
     * chat endpoint: answering with the new model's decision while echoing the
     * old model's id is the one failure mode a caller cannot see. */
    if (!jev_model_is_served(req.model, served, spec->model_type)) {
        fail(err, errcap,
             "model `%s` is not served by this endpoint (the resident model is "
             "`%s`)", req.model, served);
        jev_req_free(&req);
        return -1;
    }

    /* The resident adapter, resolved now: a switch replaces the engine and the
     * tokenizer between requests, and this is where that lands. */
    jev_adapter adapter;
    if (jev_probe(spec, tok, &adapter, err, errcap) != 0) {
        jev_req_free(&req);
        return -1;
    }

    /* One line at the start of the request: a decision can stream tens of
     * GiB before it answers, and without this the serve log only shows the
     * access-log line at each end. The logger adds the wall-clock time and
     * the request id, so these lines are attributable to one HTTP call. */
    CWICK_INFO("systemone", "decision: %zu question(s), %zu body bytes, "
               "model %s, adapter %s — stream=false thinking=off, "
               "one prefill per question, no decode",
               req.nq, body_len, served, adapter.family);

    size_t vocab = (size_t)spec->vocab_size;
    float *logits = cwick_xmalloc(vocab * sizeof(float));
    jev_out *outs = jev_calloc(req.nq, sizeof(jev_out));
    int *letters = jev_calloc(JEV_MAX_OPTIONS, sizeof(int));
    int rc = -1;

    /* Two passes. The first encodes every question and finds the longest token
     * prefix they share; the second prefills. Encoding is microseconds against
     * a prefill and the shared prefix only exists once the second prompt does,
     * so this has to be a separate pass.
     *
     * Why it matters: every question is "system + <state> + question", and the
     * state is identical across them. Prefilling it once and resuming at the
     * prefix turns N full prefills into one plus N-1 suffixes. On the
     * 60-layer checkpoints a token costs ~180 ms, so three questions about a
     * 250-token state stop paying for that state twice. */
    uint32_t **ids_all = jev_calloc(req.nq, sizeof(uint32_t *));
    int64_t *len_all = jev_calloc(req.nq, sizeof(int64_t));

    for (size_t qi = 0; qi < req.nq; qi++) {
        jev_q *q = &req.q[qi];
        for (size_t i = 0; i < q->n; i++) {
            int id = i < (size_t)adapter.letters ? letter_token(tok, i) : -1;
            if (id < 0) {
                fail(err, errcap,
                     "this checkpoint's tokenizer has no single-token letter for "
                     "option %c; the decision endpoint needs one token per option "
                     "(family %s, %d of %d letters available)",
                     (char)('A' + i), adapter.family, adapter.letters,
                     JEV_MAX_OPTIONS);
                goto done;
            }
            if (qi == 0) letters[i] = id;
        }

        char *user = render_user(&req, q);
        if (encode_prompt(tok, spec, JEV_SYSTEM, user, &ids_all[qi],
                          &len_all[qi], &e) != CWICK_OK) {
            fail(err, errcap, "question %s: cannot encode the prompt: %s", q->name,
                 cwick_error_message(&e));
            free(user);
            goto done;
        }
        /* --trace shows the rendered prompt, the same way a chat turn shows
           its messages. A decision has no generated output, so the readout
           below is the only "answer" there is to log. */
        if (cwick_log_enabled(CWICK_LOG_TRACE)) {
            CWICK_TRAC("systemone", "question %zu/%zu (%s) prompt:",
                       qi + 1, req.nq, q->name);
            cwick_log_block(CWICK_LOG_TRACE, "systemone", JEV_SYSTEM, 16384, 4);
            cwick_log_block(CWICK_LOG_TRACE, "systemone", user, 16384, 4);
        }
        free(user);
    }

    /* Longest common token prefix. It can never reach the shortest prompt:
     * the readout sits at each prompt's own last position, so at least one
     * token must remain as that question's suffix. */
    int64_t prefix = len_all[0];
    for (size_t qi = 1; qi < req.nq; qi++) {
        int64_t n = len_all[qi];
        int64_t lim = prefix < n ? prefix : n;
        int64_t k = 0;
        while (k < lim && ids_all[qi][k] == ids_all[0][k]) k++;
        prefix = k;
    }
    for (size_t qi = 0; qi < req.nq; qi++)
        if (len_all[qi] <= prefix) prefix = len_all[qi] - 1;
    if (prefix < 0) prefix = 0;
    /* Below this the bookkeeping is worth more than the tokens it saves, and
     * a very short "prefix" would usually be the chat template's boilerplate
     * rather than the state. */
    if (prefix < 48) prefix = 0;
    /* Reuse is limited to the backend whose state can be continued exactly.
     *
     * Moving the cursor and prefilling over the tail is only safe when the
     * cross-token state is addressed by position. MiniMax is GQA + KV and
     * reproduces the uninterrupted readout bit for bit (measured |dp| = 0.0
     * on a three-question request). The other two accumulate state that a
     * rewrite does not undo: with the same three questions, moving the cursor
     * changed two of three answers on DeepSeek (|dp| = 0.56) and one on GLM
     * (|dp| = 0.41). Snapshotting that state and putting it back -- the pfx
     * mechanism -- fixed the *choices* on GLM but left |dp| = 0.11, because
     * the DSA indexer keeps a partially filled pool whose tail is not
     * restored exactly, so it is not used here either. Both keep the old
     * N-full-prefills path until their apply is exact; the interface below is
     * where that lands when it is.
     *
     * CWICK_JEV_REUSE_ALL=1 measures a backend that has been made resumable;
     * CWICK_JEV_NO_PREFIX_REUSE=1 forces the old path. */
    if (family_of(spec) != JEV_FAMILY_MM && !getenv("CWICK_JEV_REUSE_ALL"))
        prefix = 0;
    if (getenv("CWICK_JEV_NO_PREFIX_REUSE")) prefix = 0;

    int64_t prefilled = 0, reused = 0;
    for (size_t qi = 0; qi < req.nq; qi++) {
        jev_q *q = &req.q[qi];
        int64_t n = len_all[qi];
        int64_t from = (qi > 0 && prefix > 0 && n > prefix) ? prefix : 0;
        int64_t slice = n - from;
        if (from > 0) reused += from;
        prefilled += slice;

        if (from > 0)
            CWICK_INFO("systemone", "question %zu/%zu (%s): prefill %lld tokens, "
                       "%lld reused from the shared prefix",
                       qi + 1, req.nq, q->name, (long long)slice, (long long)from);
        else
            CWICK_INFO("systemone", "question %zu/%zu (%s): prefill %lld tokens",
                       qi + 1, req.nq, q->name, (long long)slice);

        cwick_error_clear(&e);
        if (cwick_engine_prefill_logits_at(eng, ids_all[qi] + from, slice, from,
                                           logits, &e) != CWICK_OK) {
            fail(err, errcap, "question %s: prefill failed: %s", q->name,
                 cwick_error_message(&e));
            goto done;
        }
        *in_tokens += n;

        outs[qi].p = jev_calloc(q->n, sizeof(double));
        softmax_over(logits, letters, q->n, req.temperature, outs[qi].p);
        outs[qi].confidence = concentration(outs[qi].p, q->n);
        int win = 0;
        for (size_t i = 1; i < q->n; i++)
            if (outs[qi].p[i] > outs[qi].p[win]) win = (int)i;
        outs[qi].winner = win;
        if (q->kind == JEV_SCORE) {
            double s = 0.0;
            for (size_t i = 0; i < q->n; i++) s += (double)i * outs[qi].p[i];
            outs[qi].score = s;
        } else if (q->kind == JEV_NOUL) {
            outs[qi].noul = outs[qi].p[0];   /* option A is "yes" */
        }

        /* Execution log: what the readout chose and how sharp it was. A flat
           distribution is itself the finding -- it says the state/question
           pair did not separate the options -- so it belongs in stdout, not
           only in the response body. */
        double runner = 0.0;
        for (size_t i = 0; i < q->n; i++)
            if ((int)i != win && outs[qi].p[i] > runner) runner = outs[qi].p[i];
        if (q->kind == JEV_CHOICE)
            CWICK_INFO("systemone", "question %zu/%zu (%s) -> %s p=%.4f "
                       "(runner-up %.4f), confidence=%.4f, %zu options",
                       qi + 1, req.nq, q->name, q->keys[win], outs[qi].p[win],
                       runner, outs[qi].confidence, q->n);
        else if (q->kind == JEV_SCORE)
            CWICK_INFO("systemone", "question %zu/%zu (%s) -> score %.4f, "
                       "confidence=%.4f, %zu levels",
                       qi + 1, req.nq, q->name, outs[qi].score,
                       outs[qi].confidence, q->n);
        else
            CWICK_INFO("systemone", "question %zu/%zu (%s) -> yes p=%.4f, "
                       "confidence=%.4f",
                       qi + 1, req.nq, q->name, outs[qi].noul,
                       outs[qi].confidence);
    }

    {
        jb b = {0};
        char *em = cwick_json_stringify(served);
        jb_putf(&b, "{\"object\":\"systemone.decision\",\"model\":%s,"
                    "\"adapter\":", em);
        free(em);
        put_adapter(&b, &adapter);
        jb_put(&b, ",\"answers\":{");
        for (size_t qi = 0; qi < req.nq; qi++) {
            jev_q *q = &req.q[qi];
            char *qn = cwick_json_stringify(q->name);
            if (qi) jb_put(&b, ",");
            if (q->kind == JEV_CHOICE) {
                char *ck = cwick_json_stringify(q->keys[outs[qi].winner]);
                jb_putf(&b, "%s:{\"type\":\"choice\",\"choice\":%s,"
                            "\"probabilities\":", qn, ck);
                free(ck);
                put_probabilities(&b, q, outs[qi].p);
                jb_putf(&b, ",\"confidence\":%.6f}", outs[qi].confidence);
            } else if (q->kind == JEV_SCORE) {
                jb_putf(&b, "%s:{\"type\":\"score\",\"score\":%.6f,"
                            "\"legend\":{", qn, outs[qi].score);
                for (size_t i = 0; i < q->n; i++) {
                    char *ld = cwick_json_stringify(q->descs[i]);
                    jb_putf(&b, "%s\"%zu\":%s", i ? "," : "", i, ld);
                    free(ld);
                }
                jb_put(&b, "},\"probabilities\":");
                put_probabilities(&b, q, outs[qi].p);
                jb_putf(&b, ",\"confidence\":%.6f}", outs[qi].confidence);
            } else {
                jb_putf(&b, "%s:{\"type\":\"noul\",\"noul\":%.6f}",
                        qn, outs[qi].noul);
            }
            free(qn);
        }
        /* `cached_input_tokens` is the reference field for "state tokens served
         * from the cache"; `prefilled_tokens`/`reused_tokens` stay next to it
         * because they are the two numbers the log line reports and a caller
         * comparing them can see the saving rather than infer it. */
        jb_putf(&b, "},\"usage\":{\"input_tokens\":%lld,"
                    "\"cached_input_tokens\":%lld,\"output_tokens\":0,"
                    "\"prefilled_tokens\":%lld,\"reused_tokens\":%lld}}",
                (long long)*in_tokens, (long long)reused, (long long)prefilled,
                (long long)reused);
        *out_json = jb_take(&b);
        if (cwick_log_enabled(CWICK_LOG_TRACE)) {
            CWICK_TRAC("systemone", "decision output:");
            cwick_log_block(CWICK_LOG_TRACE, "systemone", *out_json, 16384, 4);
        }
    }
    rc = 0;

done:
    for (size_t di = 0; di < req.nq; di++) free(ids_all[di]);
    free(ids_all);
    free(len_all);
    {
        struct timeval dec_t1;
        gettimeofday(&dec_t1, NULL);
        double ms = (double)(dec_t1.tv_sec - dec_t0.tv_sec) * 1e3 +
                    (double)(dec_t1.tv_usec - dec_t0.tv_usec) / 1e3;
        CWICK_INFO("systemone", "decision %s: %zu question(s), %lld input "
                   "tokens, %lld prefilled (%lld reused), %.0f ms wall, "
                   "%.2f tok/s prefill%s%s",
                   rc == 0 ? "done" : "failed", req.nq, (long long)*in_tokens,
                   (long long)prefilled, (long long)reused,
                   ms, ms > 0.0 ? (double)prefilled * 1000.0 / ms : 0.0,
                   rc == 0 ? "" : " — ", rc == 0 ? "" : err);
    }
    for (size_t i = 0; i < req.nq; i++) if (outs) free(outs[i].p);
    free(outs);
    free(logits);
    free(letters);
    jev_req_free(&req);
    return rc;
}
