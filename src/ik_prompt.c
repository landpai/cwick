/*
 * ik_prompt.c — Inkling-2-Mini chat rendering in pure C.
 *
 * Text parts become `<|content_text|>` records; image parts become
 * `<|content_image|><|unused_200054|>` records, one placeholder per image,
 * which the engine replaces with the vision tower's merged patch embeddings.
 */
#include "ik_prompt.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"
#include "tools.h"
#include "util.h"

typedef struct {
    size_t off, len;
    bool special;
} ik_seg;

typedef struct {
    char *s;
    size_t n, cap;
    ik_seg *seg;
    size_t nseg, segcap;
    bool track;
} ik_buf;

static void ib_write_mode(ik_buf *b, const char *s, size_t n, bool special) {
    if (n == 0) return;
    if (b->track) {
        if (b->nseg == b->segcap) {
            b->segcap = b->segcap ? b->segcap * 2 : 64;
            b->seg = (ik_seg *)realloc(b->seg, b->segcap * sizeof(*b->seg));
            if (!b->seg) abort();
        }
        b->seg[b->nseg++] = (ik_seg){b->n, n, special};
    }
    if (b->n + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 4096;
        while (cap < b->n + n + 1) cap *= 2;
        char *p = (char *)realloc(b->s, cap);
        if (!p) abort();
        b->s = p;
        b->cap = cap;
    }
    memcpy(b->s + b->n, s, n);
    b->n += n;
    b->s[b->n] = 0;
}

static void ib_write(ik_buf *b, const char *s, size_t n) {
    ib_write_mode(b, s, n, false);
}

static void ib_puts(ik_buf *b, const char *s) {
    if (s) ib_write(b, s, strlen(s));
}

static void ib_special(ik_buf *b, const char *s) {
    if (s) ib_write_mode(b, s, strlen(s), true);
}

static int ik_effort(const char *e, double *out, cwick_error *err) {
    if (!e || !*e) { *out = 0.9; return 0; }
    if (strcasecmp(e, "none") == 0) { *out = 0.0; return 0; }
    if (strcasecmp(e, "minimal") == 0) { *out = 0.1; return 0; }
    if (strcasecmp(e, "low") == 0) { *out = 0.2; return 0; }
    if (strcasecmp(e, "medium") == 0) { *out = 0.7; return 0; }
    if (strcasecmp(e, "high") == 0) { *out = 0.9; return 0; }
    if (strcasecmp(e, "xhigh") == 0 || strcasecmp(e, "max") == 0) {
        *out = 0.99; return 0;
    }
    if (strcmp(e, "关") == 0) { *out = 0.0; return 0; }
    if (strcmp(e, "低") == 0) { *out = 0.2; return 0; }
    if (strcmp(e, "中") == 0) { *out = 0.7; return 0; }
    if (strcmp(e, "高") == 0) { *out = 0.9; return 0; }
    if (strcmp(e, "超高") == 0 || strcmp(e, "极致") == 0) {
        *out = 0.99; return 0;
    }
    char *end = NULL;
    double v = strtod(e, &end);
    if (!end || *end || !isfinite(v) || v < 0.0 || v > 0.99) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "ik_prompt: unknown reasoning_effort '%s'", e);
        return -1;
    }
    *out = v;
    return 0;
}

static const char *ik_role_token(const char *role) {
    if (strcmp(role, "user") == 0) return "<|message_user|>";
    if (strcmp(role, "assistant") == 0) return "<|message_model|>";
    if (strcmp(role, "system") == 0 || strcmp(role, "developer") == 0)
        return "<|message_system|>";
    if (strcmp(role, "tool") == 0) return "<|message_tool|>";
    return NULL;
}

static json_value *ik_image_payload(json_value *part, unsigned depth) {
    if (!part || depth > 4) return NULL;
    if (part->kind == JSON_STRING) return part;
    if (part->kind != JSON_OBJECT) return NULL;
    json_value *im = cwick_json_get(part, "image_url");
    if (im) {
        if (im->kind == JSON_STRING) return im;
        if (im->kind == JSON_OBJECT) {
            json_value *url = cwick_json_get(im, "url");
            if (url && url->kind == JSON_STRING) return url;
        }
    }
    json_value *image = cwick_json_get(part, "image");
    if (image) {
        json_value *v = ik_image_payload(image, depth + 1);
        if (v) return v;
    }
    json_value *src = cwick_json_get(part, "source");
    if (src && src->kind == JSON_OBJECT) {
        json_value *ty = cwick_json_get(src, "type");
        if (ty && ty->kind == JSON_STRING && strcmp(ty->string, "url") == 0) {
            json_value *url = cwick_json_get(src, "url");
            if (url && url->kind == JSON_STRING) return url;
        } else {
            json_value *data = cwick_json_get(src, "data");
            if (data && data->kind == JSON_STRING) return data;
        }
    }
    json_value *data = cwick_json_get(part, "data");
    if (!data) data = cwick_json_get(part, "b64_json");
    if (!data) data = cwick_json_get(part, "url");
    return data && data->kind == JSON_STRING ? data : NULL;
}

static char *ik_image_b64(json_value *part, cwick_error *err) {
    json_value *v = ik_image_payload(part, 0);
    if (!v || v->kind != JSON_STRING || !v->string || !*v->string) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "ik_prompt: image must contain base64 data or a data URL");
        return NULL;
    }
    const char *s = v->string;
    while (isspace((unsigned char)*s)) s++;
    if (strstr(s, "://") || strncmp(s, "//", 2) == 0) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "remote image URLs are not supported; use base64 or a data URL");
        return NULL;
    }
    if (strncmp(s, "data:", 5) == 0) {
        const char *comma = strchr(s + 5, ',');
        bool base64 = false;
        for (const char *p = s + 5; comma && p + 7 <= comma; p++) {
            if (strncasecmp(p, ";base64", 7) == 0) { base64 = true; break; }
        }
        if (!comma || !base64) {
            cwick_error_set(err, CWICK_ERR_ARGUMENT,
                            "image data URL must declare ;base64 and include a payload");
            return NULL;
        }
        s = comma + 1;
    } else if (strchr(s, ':')) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "remote image URLs are not supported; use base64 or a data URL");
        return NULL;
    }
    return *s ? cwick_strdup(s) : NULL;
}

static cwick_status ik_add_image(char ***imgs, size_t *n, char *b64,
                                 cwick_error *err) {
    if (!imgs || !n) { free(b64); return CWICK_OK; }
    if (*n >= IK_MAX_IMAGES) {
        free(b64);
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "at most %d images are supported", IK_MAX_IMAGES);
        return CWICK_ERR_ARGUMENT;
    }
    char **p = (char **)realloc(*imgs, (*n + 1) * sizeof(char *));
    if (!p) { free(b64); abort(); }
    *imgs = p;
    (*imgs)[(*n)++] = b64;
    return CWICK_OK;
}

static int cmp_str(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static void ib_json(ik_buf *b, const json_value *v) {
    if (!v) { ib_puts(b, "null"); return; }
    switch (v->kind) {
    case JSON_NULL: ib_puts(b, "null"); break;
    case JSON_BOOL: ib_puts(b, v->boolean ? "true" : "false"); break;
    case JSON_NUMBER: {
        char num[64];
        if (v->number == (double)v->integer) snprintf(num, sizeof(num), "%lld", (long long)v->integer);
        else snprintf(num, sizeof(num), "%.17g", v->number);
        ib_puts(b, num); break;
    }
    case JSON_STRING: {
        char *s = cwick_json_stringify(v->string ? v->string : "");
        ib_puts(b, s); free(s); break;
    }
    case JSON_ARRAY:
        ib_puts(b, "[");
        for (size_t i = 0; i < v->count; i++) {
            if (i) ib_puts(b, ",");
            ib_json(b, v->items[i]);
        }
        ib_puts(b, "]");
        break;
    case JSON_OBJECT: {
        const char **keys = (const char **)malloc((v->count ? v->count : 1) * sizeof(char *));
        if (!keys) abort();
        for (size_t i = 0; i < v->count; i++) keys[i] = v->members[i].key;
        qsort(keys, v->count, sizeof(char *), cmp_str);
        ib_puts(b, "{");
        for (size_t i = 0; i < v->count; i++) {
            if (i) ib_puts(b, ",");
            char *key = cwick_json_stringify(keys[i]);
            ib_puts(b, key); free(key);
            ib_puts(b, ":");
            ib_json(b, cwick_json_get((json_value *)v, keys[i]));
        }
        ib_puts(b, "}");
        free(keys);
        break;
    }
    }
}

static void ik_emit_tools(ik_buf *b, const char *tools_json) {
    cwick_error e;
    cwick_error_clear(&e);
    json_value *tools = cwick_json_parse(tools_json, strlen(tools_json), &e);
    if (!tools || tools->kind != JSON_ARRAY) {
        cwick_json_free(tools);
        ib_puts(b, "[]");
        return;
    }
    ib_puts(b, "[");
    for (size_t i = 0; i < tools->count; i++) {
        json_value *tool = tools->items[i];
        json_value *fn = cwick_json_get(tool, "function");
        if (!fn || fn->kind != JSON_OBJECT) fn = tool;
        json_value *desc = cwick_json_get(fn, "description");
        json_value *name = cwick_json_get(fn, "name");
        json_value *params = cwick_json_get(fn, "parameters");
        json_value *type = cwick_json_get(tool, "type");
        if (i) ib_puts(b, ",");
        ib_puts(b, "{\"description\":");
        char *s = cwick_json_stringify(desc && desc->kind == JSON_STRING && *desc->string
                                       ? desc->string : "");
        ib_puts(b, s); free(s);
        ib_puts(b, ",\"name\":");
        s = cwick_json_stringify(name && name->kind == JSON_STRING ? name->string : "");
        ib_puts(b, s); free(s);
        ib_puts(b, ",\"parameters\":");
        if (params && params->kind == JSON_OBJECT) ib_json(b, params);
        else ib_puts(b, "{}");
        ib_puts(b, ",\"type\":");
        s = cwick_json_stringify(type && type->kind == JSON_STRING && *type->string
                                 ? type->string : "function");
        ib_puts(b, s); free(s);
        ib_puts(b, "}");
    }
    ib_puts(b, "]");
    cwick_json_free(tools);
}

static void ik_emit_effort(ik_buf *b, double eff) {
    int64_t millieffort = llrint(eff * 1000.0);
    int64_t hundredths = llrint((double)millieffort / 10.0);
    char num[64];
    snprintf(num, sizeof(num), "%.10g", (double)hundredths / 100.0);
    ib_special(b, "<|message_system|>"); ib_special(b, "<|content_text|>");
    ib_puts(b, "Thinking effort level: ");
    ib_puts(b, num);
    ib_special(b, "<|end_message|>");
}

static void ik_emit_part(ik_buf *b, json_value *part,
                         char ***imgs, size_t *nimg, cwick_error *err) {
    if (!part) return;
    if (part->kind == JSON_STRING) {
        ib_special(b, "<|content_text|>");
        ib_puts(b, part->string); ib_special(b, "<|end_message|>");
        return;
    }
    if (part->kind != JSON_OBJECT) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "ik_prompt: unsupported content part");
        return;
    }
    json_value *ty = cwick_json_get(part, "type");
    const char *t = ty && ty->kind == JSON_STRING ? ty->string : "text";
    if (strcmp(t, "text") == 0 || strcmp(t, "input_text") == 0) {
        json_value *tx = cwick_json_get(part, "text");
        ib_special(b, "<|content_text|>");
        if (tx && tx->kind == JSON_STRING) ib_puts(b, tx->string);
        ib_special(b, "<|end_message|>");
    } else if (strcmp(t, "image") == 0 || strcmp(t, "image_url") == 0 ||
               strcmp(t, "input_image") == 0) {
        char *b64 = ik_image_b64(part, err);
        if (!b64) return;
        if (ik_add_image(imgs, nimg, b64, err) != CWICK_OK) return;
        ib_special(b, "<|content_image|>");
        ib_special(b, IK_IMAGE_PLACEHOLDER);
        ib_special(b, "<|end_message|>");
    } else {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "ik_prompt: unsupported content part type '%s'", t);
    }
}

static void ik_emit_tool_calls(ik_buf *b, json_value *m, cwick_error *err) {
    json_value *calls = cwick_json_get(m, "tool_calls");
    if (!calls || calls->kind != JSON_ARRAY) return;
    json_value *namev = cwick_json_get(m, "name");
    const char *mname = namev && namev->kind == JSON_STRING ? namev->string : NULL;
    for (size_t i = 0; i < calls->count; i++) {
        json_value *tc = calls->items[i];
        json_value *fn = cwick_json_get(tc, "function");
        json_value *nv = fn ? cwick_json_get(fn, "name") : NULL;
        json_value *args = fn ? cwick_json_get(fn, "arguments") : NULL;
        json_value *parsed = NULL;
        if (!args || args->kind == JSON_NULL) {
            cwick_error ae;
            cwick_error_clear(&ae);
            parsed = cwick_json_parse("{}", 2, &ae);
            args = parsed;
        } else if (args->kind == JSON_STRING) {
            cwick_error ae;
            cwick_error_clear(&ae);
            const char *raw = *args->string ? args->string : "{}";
            parsed = cwick_json_parse(raw, strlen(raw), &ae);
            args = parsed;
        }
        if (!nv || nv->kind != JSON_STRING || !args || args->kind != JSON_OBJECT ||
            (mname && strcmp(mname, nv->string) != 0)) {
            cwick_json_free(parsed);
            cwick_error_set(err, CWICK_ERR_ARGUMENT,
                            "ik_prompt: tool call needs a name and arguments object");
            return;
        }
        char *name = cwick_json_stringify(nv->string);
        ib_special(b, "<|message_model|>"); ib_puts(b, nv->string);
        ib_special(b, "<|content_invoke_tool_json|>"); ib_puts(b, "{\"name\":");
        ib_puts(b, name); free(name);
        ib_puts(b, ",\"args\":");
        ib_json(b, args);
        ib_puts(b, "}"); ib_special(b, "<|end_message|>");
        cwick_json_free(parsed);
    }
}

static char *ik_prompt_render_internal(const char *messages_json,
                                      const char *tools_json,
                                      const char *effort_level, char ***images,
                                      size_t *n_images, cwick_error *err,
                                      ik_buf *segments) {
    if (images) *images = NULL;
    if (n_images) *n_images = 0;
    if (!messages_json) return NULL;
    cwick_error pe;
    cwick_error_clear(&pe);
    json_value *msgs = cwick_json_parse(messages_json, strlen(messages_json), &pe);
    if (!msgs || msgs->kind != JSON_ARRAY) {
        cwick_error_set(err, CWICK_ERR_PARSE, "ik_prompt: messages is not an array");
        cwick_json_free(msgs);
        return NULL;
    }
    double eff;
    if (ik_effort(effort_level, &eff, err) != 0) { cwick_json_free(msgs); return NULL; }
    ik_buf b = {0};
    b.track = segments != NULL;
    bool effort_emitted = false;
    if (tools_json && *tools_json) {
        cwick_error_clear(&pe);
        json_value *tools = cwick_json_parse(tools_json, strlen(tools_json), &pe);
        if (!tools || tools->kind != JSON_ARRAY) {
            cwick_json_free(tools); cwick_json_free(msgs); free(b.s); free(b.seg);
            cwick_error_set(err, CWICK_ERR_PARSE, "ik_prompt: tools is not an array");
            return NULL;
        }
        cwick_json_free(tools);
        ib_special(&b, "<|message_system|>");
        ib_puts(&b, "tool_declare"); ib_special(&b, "<|content_xml|>");
        ik_emit_tools(&b, tools_json);
        ib_special(&b, "<|end_message|>");
    }
    for (size_t i = 0; i < msgs->count && err->status == CWICK_OK; i++) {
        json_value *m = msgs->items[i];
        json_value *rv = cwick_json_get(m, "role");
        const char *role = rv && rv->kind == JSON_STRING ? rv->string : "user";
        const char *head = ik_role_token(role);
        if (!head) {
            cwick_error_set(err, CWICK_ERR_ARGUMENT, "ik_prompt: bad role '%s'", role);
            break;
        }
        if (!effort_emitted && strcmp(role, "system") != 0 &&
            strcmp(role, "developer") != 0) {
            ik_emit_effort(&b, eff);
            effort_emitted = true;
        }
        const char *tool_name = NULL;
        if (strcmp(role, "tool") == 0) {
            json_value *name = cwick_json_get(m, "name");
            json_value *cid = cwick_json_get(m, "tool_call_id");
            tool_name = name && name->kind == JSON_STRING ? name->string : NULL;
            if (!tool_name && cid && cid->kind == JSON_STRING) {
                for (size_t j = 0; j < msgs->count; j++) {
                    json_value *prev = msgs->items[j];
                    json_value *calls = cwick_json_get(prev, "tool_calls");
                    if (!calls || calls->kind != JSON_ARRAY) continue;
                    for (size_t k = 0; k < calls->count; k++) {
                        json_value *tc = calls->items[k];
                        json_value *id = cwick_json_get(tc, "id");
                        json_value *fn = cwick_json_get(tc, "function");
                        json_value *nv = fn ? cwick_json_get(fn, "name") : NULL;
                        if (id && id->kind == JSON_STRING && strcmp(id->string, cid->string) == 0 &&
                            nv && nv->kind == JSON_STRING) tool_name = nv->string;
                    }
                }
            }
            json_value *content = cwick_json_get(m, "content");
            if (!content || content->kind != JSON_ARRAY) {
                ib_special(&b, head);
                if (tool_name) ib_puts(&b, tool_name);
            }
        }
        if (strcmp(role, "assistant") == 0) {
            json_value *reason = cwick_json_get(m, "reasoning_content");
            if (reason && reason->kind == JSON_STRING && *reason->string) {
                ib_special(&b, head);
                json_value *name = cwick_json_get(m, "name");
                if (name && name->kind == JSON_STRING && *name->string) ib_puts(&b, name->string);
                ib_special(&b, "<|content_thinking|>");
                ib_puts(&b, reason->string); ib_special(&b, "<|end_message|>");
            }
        }
        json_value *content = cwick_json_get(m, "content");
        json_value *calls = cwick_json_get(m, "tool_calls");
        if (content && content->kind == JSON_STRING &&
            (*content->string || (strcmp(role, "assistant") != 0 || !calls || !calls->count))) {
            if (strcmp(role, "tool") != 0) {
                ib_special(&b, head);
                json_value *name = cwick_json_get(m, "name");
                if (strcmp(role, "assistant") == 0 && name &&
                    name->kind == JSON_STRING && *name->string) ib_puts(&b, name->string);
            }
            ib_special(&b, "<|content_text|>"); ib_puts(&b, content->string);
            ib_special(&b, "<|end_message|>");
        } else if (content && content->kind == JSON_ARRAY) {
            for (size_t j = 0; j < content->count && err->status == CWICK_OK; j++) {
                json_value *part = content->items[j];
                ib_special(&b, head);
                if (strcmp(role, "assistant") == 0) {
                    json_value *name = cwick_json_get(m, "name");
                    if (name && name->kind == JSON_STRING && *name->string) ib_puts(&b, name->string);
                } else if (strcmp(role, "tool") == 0 && tool_name) {
                    ib_puts(&b, tool_name);
                }
                if (part->kind == JSON_OBJECT) {
                    json_value *ty = cwick_json_get(part, "type");
                    const char *t = ty && ty->kind == JSON_STRING ? ty->string : "text";
                    if (strcmp(t, "image") == 0 || strcmp(t, "input_image") == 0 ||
                        strcmp(t, "image_url") == 0) {
                        if (strcmp(role, "user") != 0 && strcmp(role, "tool") != 0) {
                            cwick_error_set(err, CWICK_ERR_ARGUMENT,
                                            "ik_prompt: images require user or tool role");
                            break;
                        }
                    }
                }
                ik_emit_part(&b, part, images, n_images, err);
            }
        } else if (content && content->kind != JSON_NULL) {
            cwick_error_set(err, CWICK_ERR_ARGUMENT, "ik_prompt: bad content");
        }
        if (strcmp(role, "assistant") == 0) {
            ik_emit_tool_calls(&b, m, err);
            ib_special(&b, "<|content_model_end_sampling|>");
        }
    }
    if (err->status != CWICK_OK) {
        cwick_json_free(msgs); free(b.s); free(b.seg);
        ik_prompt_free_images(images ? *images : NULL, n_images ? *n_images : 0);
        if (images) *images = NULL;
        if (n_images) *n_images = 0;
        return NULL;
    }
    if (!effort_emitted) ik_emit_effort(&b, eff);
    cwick_json_free(msgs);
    if (segments) *segments = b;
    else free(b.seg);
    return b.s;
}

char *ik_prompt_render(const char *messages_json, const char *tools_json,
                       const char *effort_level, char ***images, size_t *n_images,
                       cwick_error *err) {
    return ik_prompt_render_internal(messages_json, tools_json, effort_level,
                                     images, n_images, err, NULL);
}

char *ik_prompt_encode_text(const char *messages_json, const char *tools_json,
                            cwick_error *err) {
    char **images = NULL;
    size_t n = 0;
    char *text = ik_prompt_render(messages_json, tools_json, NULL, &images, &n, err);
    ik_prompt_free_images(images, n);
    return text;
}

char *ik_prompt_encode_decision(const char *messages_json, const char *tools_json,
                                cwick_error *err) {
    char **images = NULL;
    size_t n_images = 0;
    char *rendered = ik_prompt_render(messages_json, tools_json, "none",
                                     &images, &n_images, err);
    ik_prompt_free_images(images, n_images);
    if (!rendered) return NULL;
    static const char suffix[] = "<|message_model|><|content_text|>";
    size_t len = strlen(rendered);
    char *text = (char *)realloc(rendered, len + sizeof(suffix));
    if (!text) abort();
    memcpy(text + len, suffix, sizeof(suffix));
    return text;
}

static cwick_status ik_encode_segments(const cwick_tokenizer *tok,
                                       const ik_buf *b, const char *suffix,
                                       uint32_t **ids, int64_t *n_out,
                                       cwick_error *err) {
    cwick_tokenizer plain = *tok;
    plain.special = NULL;
    plain.special_count = 0;
    uint32_t *all = NULL;
    size_t count = 0;
    for (size_t i = 0; i < b->nseg;) {
        size_t end = i + 1;
        if (!b->seg[i].special) {
            while (end < b->nseg && !b->seg[end].special &&
                   b->seg[end].off == b->seg[end - 1].off + b->seg[end - 1].len)
                end++;
        }
        size_t off = b->seg[i].off;
        size_t len = b->seg[end - 1].off + b->seg[end - 1].len - off;
        bool special = b->seg[i].special;
        char saved = b->s[off + len];
        b->s[off + len] = 0;
        uint32_t *piece = NULL;
        int64_t np = 0;
        cwick_status st = cwick_tokenizer_encode(special ? tok : &plain,
                                                 b->s + off, &piece, &np, err);
        b->s[off + len] = saved;
        if (st != CWICK_OK) { free(piece); free(all); return st; }
        if (special && np != 1) {
            free(piece); free(all);
            cwick_error_set(err, CWICK_ERR_ARGUMENT,
                            "ik_prompt: structural marker is not one token "
                            "(len=%zu ids=%lld, %.64s)",
                            len, (long long)np, b->s + off);
            return CWICK_ERR_ARGUMENT;
        }
        if ((size_t)np > SIZE_MAX - count) abort();
        all = (uint32_t *)cwick_xrealloc(all, (count + (size_t)np) * sizeof(uint32_t));
        memcpy(all + count, piece, (size_t)np * sizeof(uint32_t));
        count += (size_t)np;
        free(piece);
        i = end;
    }
    if (suffix) {
        uint32_t *piece = NULL;
        int64_t np = 0;
        cwick_status st = cwick_tokenizer_encode(tok, suffix, &piece, &np, err);
        if (st != CWICK_OK) { free(piece); free(all); return st; }
        if ((size_t)np > SIZE_MAX - count) abort();
        all = (uint32_t *)cwick_xrealloc(all, (count + (size_t)np) * sizeof(uint32_t));
        memcpy(all + count, piece, (size_t)np * sizeof(uint32_t));
        count += (size_t)np;
        free(piece);
    }
    if (!all) all = (uint32_t *)cwick_xmalloc(sizeof(uint32_t));
    *ids = all;
    *n_out = (int64_t)count;
    return CWICK_OK;
}

cwick_status ik_prompt_encode(const cwick_tokenizer *tok,
                              const char *messages_json,
                              const char *tools_json,
                              const char *effort_level,
                              char ***images, size_t *n_images,
                              uint32_t **ids, int64_t *n_out,
                              cwick_error *err) {
    if (!tok || !messages_json) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "ik_prompt: no messages");
        return CWICK_ERR_ARGUMENT;
    }
    ik_buf b;
    char *text = ik_prompt_render_internal(messages_json, tools_json, effort_level,
                                           images, n_images, err, &b);
    if (!text) return err->status != CWICK_OK ? err->status : CWICK_ERR_STATE;
    cwick_status st = ik_encode_segments(tok, &b, NULL, ids, n_out, err);
    free(text); free(b.seg);
    if (st != CWICK_OK && images && *images) {
        ik_prompt_free_images(*images, *n_images);
        *images = NULL;
        *n_images = 0;
        *ids = NULL;
        *n_out = 0;
    }
    return st;
}

cwick_status ik_prompt_encode_decision_images(const cwick_tokenizer *tok,
                                              const char *messages_json,
                                              const char *tools_json,
                                              char ***images, size_t *n_images,
                                              uint32_t **ids, int64_t *n_out,
                                              cwick_error *err) {
    if (!tok || !messages_json || !images || !n_images || !ids || !n_out) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "ik_prompt: no messages or outputs");
        return CWICK_ERR_ARGUMENT;
    }
    *images = NULL;
    *n_images = 0;
    *ids = NULL;
    *n_out = 0;
    ik_buf b;
    char *text = ik_prompt_render_internal(messages_json, tools_json, "none",
                                           images, n_images, err, &b);
    if (!text) return err->status != CWICK_OK ? err->status : CWICK_ERR_STATE;
    cwick_status st = ik_encode_segments(tok, &b, "<|message_model|><|content_text|>",
                                         ids, n_out, err);
    free(text); free(b.seg);
    if (st != CWICK_OK) {
        ik_prompt_free_images(*images, *n_images);
        *images = NULL;
        *n_images = 0;
        *ids = NULL;
        *n_out = 0;
    }
    return st;
}

cwick_status ik_prompt_encode_decision_ids(const cwick_tokenizer *tok,
                                           const char *messages_json,
                                           const char *tools_json,
                                           uint32_t **ids, int64_t *n_out,
                                           cwick_error *err) {
    char **images = NULL;
    size_t n_images = 0;
    cwick_status st = ik_prompt_encode_decision_images(tok, messages_json,
                                                        tools_json, &images,
                                                        &n_images, ids, n_out,
                                                        err);
    ik_prompt_free_images(images, n_images);
    return st;
}

cwick_status ik_prompt_image_token_id(const cwick_tokenizer *tok,
                                      uint32_t *id_out, cwick_error *err) {
    if (!tok || !id_out) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "ik_prompt: no tokenizer");
        return CWICK_ERR_ARGUMENT;
    }
    uint32_t *ids = NULL;
    int64_t n = 0;
    cwick_status st = cwick_tokenizer_encode(tok, IK_IMAGE_PLACEHOLDER, &ids, &n, err);
    if (st != CWICK_OK) { free(ids); return st; }
    if (n != 1) {
        free(ids);
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "ik_prompt: image placeholder is not one token (ids=%lld)",
                        (long long)n);
        return CWICK_ERR_ARGUMENT;
    }
    *id_out = ids[0];
    free(ids);
    return CWICK_OK;
}

cwick_status ik_prompt_normalize_images(const json_value *root,
                                        char ***images, size_t *n_images,
                                        cwick_error *err) {
    if (images) *images = NULL;
    if (n_images) *n_images = 0;
    if (!root || root->kind != JSON_OBJECT || !images || !n_images) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "body.images must be an array");
        return CWICK_ERR_ARGUMENT;
    }
    json_value *many = cwick_json_get((json_value *)root, "images");
    json_value *one = cwick_json_get((json_value *)root, "image");
    size_t count = 0;
    if (many && one) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "body.images and body.image are mutually exclusive");
        return CWICK_ERR_ARGUMENT;
    }
    if (many) {
        if (many->kind != JSON_ARRAY) {
            cwick_error_set(err, CWICK_ERR_ARGUMENT, "body.images must be an array");
            return CWICK_ERR_ARGUMENT;
        }
        count = many->count;
    } else if (one) {
        if (one->kind == JSON_NULL) return CWICK_OK;
        count = 1;
    } else {
        return CWICK_OK;
    }
    if (count > IK_MAX_IMAGES) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "at most %d images are supported", IK_MAX_IMAGES);
        return CWICK_ERR_ARGUMENT;
    }
    for (size_t i = 0; i < count; i++) {
        json_value *part = many ? many->items[i] : one;
        char *b64 = ik_image_b64(part, err);
        if (!b64) {
            ik_prompt_free_images(*images, *n_images);
            *images = NULL;
            *n_images = 0;
            return CWICK_ERR_ARGUMENT;
        }
        cwick_status st = ik_add_image(images, n_images, b64, err);
        if (st != CWICK_OK) {
            ik_prompt_free_images(*images, *n_images);
            *images = NULL;
            *n_images = 0;
            return st;
        }
    }
    return CWICK_OK;
}

void ik_prompt_free_images(char **images, size_t n) {
    if (!images) return;
    for (size_t i = 0; i < n; i++) free(images[i]);
    free(images);
}

void ik_output_free(ik_output *out) {
    if (!out) return;
    free(out->reasoning);
    free(out->content);
    for (size_t i = 0; i < out->n_calls; i++) {
        free(out->calls[i].name);
        free(out->calls[i].arguments_json);
    }
    free(out->calls);
    memset(out, 0, sizeof(*out));
}

cwick_status ik_output_parse(const char *text, ik_output *out, cwick_error *err) {
    static const char *const marks[] = {
        "<|message_model|>", "<|content_thinking|>", "<|content_text|>",
        "<|content_invoke_tool_json|>", "<|end_message|>",
        "<|content_model_end_sampling|>"
    };
    memset(out, 0, sizeof(*out));
    if (!text) return CWICK_OK;
    ik_buf reasoning = {0}, content = {0};
    const char *p = text;
    int mode = 1;
    bool assistant_header = false;
    for (;;) {
        const char *hit = NULL;
        int which = -1;
        for (int i = 0; i < 6; i++) {
            const char *q = strstr(p, marks[i]);
            if (q && (!hit || q < hit)) { hit = q; which = i; }
        }
        if (mode == 2) {
            if (!hit || which != 4) {
                cwick_error_set(err, CWICK_ERR_PARSE,
                                "ik_prompt: unterminated tool-call record");
                goto fail;
            }
            cwick_error pe;
            cwick_error_clear(&pe);
            json_value *obj = cwick_json_parse(p, (size_t)(hit - p), &pe);
            json_value *name = obj ? cwick_json_get(obj, "name") : NULL;
            json_value *args = obj ? cwick_json_get(obj, "args") : NULL;
            if (!obj || obj->kind != JSON_OBJECT || !name ||
                name->kind != JSON_STRING || !name->string || !*name->string ||
                !args || args->kind != JSON_OBJECT) {
                cwick_json_free(obj);
                cwick_error_set(err, CWICK_ERR_PARSE,
                                "ik_prompt: tool call needs a name and arguments object");
                goto fail;
            }
            ik_output_call *calls = realloc(out->calls,
                                            (out->n_calls + 1) * sizeof(*calls));
            if (!calls) abort();
            out->calls = calls;
            calls[out->n_calls].name = cwick_strdup(name->string);
            ik_buf canonical = {0};
            ib_json(&canonical, args);
            calls[out->n_calls].arguments_json = canonical.s;
            out->n_calls++;
            cwick_json_free(obj);
            p = hit + strlen(marks[which]);
            mode = 1;
            continue;
        }
        if (hit) {
            if (hit > p && !(which == 3 && assistant_header))
                ib_write(mode == 0 ? &reasoning : &content,
                         p, (size_t)(hit - p));
            if (which == 0) assistant_header = true;
            else {
                assistant_header = false;
                if (which == 1) mode = 0;
                else if (which == 2) mode = 1;
                else if (which == 3) mode = 2;
                else if (which == 5) break;
            }
            p = hit + strlen(marks[which]);
            continue;
        }
        if (*p) ib_puts(mode == 0 ? &reasoning : &content, p);
        break;
    }
    out->reasoning = reasoning.s ? reasoning.s : cwick_strdup("");
    out->content = content.s ? content.s : cwick_strdup("");
    return CWICK_OK;
fail:
    free(reasoning.s);
    free(content.s);
    ik_output_free(out);
    return err->status != CWICK_OK ? err->status : CWICK_ERR_PARSE;
}
