/* json.h — minimal JSON DOM parser (config + HTTP bodies). */
#ifndef CWICK_JSON_H
#define CWICK_JSON_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "cwick.h"

typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT,
} json_kind;

typedef struct json_value json_value;
typedef struct json_pair {
    char *key;
    json_value *value;
} json_pair;

/* NOTE: these payload fields must NOT live in a union.  A JSON number has
 * both an int64 view and a double view (e.g. "1e-6", "-5.0", "10.0"), and
 * aliasing them in one union silently corrupts whichever was written first:
 * `strtoll("-5.0")` yields -5, whose bit pattern reinterpreted as a double is
 * NaN.  Keep them as distinct members so `cwick_json_num()` always sees the
 * parsed double. */
struct json_value {
    json_kind kind;
    bool boolean;             /* JSON_BOOL   */
    double number;            /* JSON_NUMBER (double view)  */
    int64_t integer;          /* JSON_NUMBER (integer view) */
    char *string;             /* JSON_STRING */
    json_value **items;       /* JSON_ARRAY  */
    json_pair *members;       /* JSON_OBJECT */
    size_t count;             /* array length or object member count */
};

json_value *cwick_json_parse(const char *text, size_t len, cwick_error *err);
void cwick_json_free(json_value *v);

json_value *cwick_json_get(json_value *obj, const char *key);
const char *cwick_json_str(json_value *v);
int64_t cwick_json_int(json_value *v, int64_t dflt);
double cwick_json_num(json_value *v, double dflt);
bool cwick_json_bool(json_value *v, bool dflt);
size_t cwick_json_array_len(json_value *v);
json_value *cwick_json_array_at(json_value *v, size_t i);

/* Serialize a parsed value back to compact JSON text (owned by the caller).
 * Used to hand a sub-tree — e.g. an assistant turn's `tool_calls` — to a
 * component that speaks JSON rather than a DOM. */
char *cwick_json_write(const json_value *v);

#endif /* CWICK_JSON_H */
