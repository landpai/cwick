/* auth.h — single-user login for cwick serve.
 *
 * One password, PBKDF2-HMAC-SHA256 in memory, short-lived opaque session
 * tokens in memory, HttpOnly SameSite=Strict cookies for the web UI and
 * `Authorization: Bearer <token>` for API clients. There is deliberately no
 * user database and no multi-tenant concept: the requirement is one owner of
 * one local model, not accounts.
 *
 * Passwords are never written to disk by this module. When the server is
 * started without --password/CWICK_PASSWORD it generates one and prints it on
 * stdout (the same place every other startup secret-free diagnostic goes).
 *
 * There are two credentials, on purpose:
 *
 *   password -> a short-lived *session* token (browser: HttpOnly cookie)
 *   API key  -> a long-lived bearer secret (third-party agents: Authorization)
 *
 * A session expires and lives only in memory; an API key does not expire and is
 * meant to be pasted once into an agent platform's configuration, so cwick
 * persists only its hash (see cwick_auth_key_load) and never the secret. An API
 * key is 160 random bits, which is why it is stored as a plain SHA-256 and not
 * through PBKDF2: stretching is what makes a *low-entropy human* password
 * expensive to guess, and there is nothing to guess here.
 */
#ifndef CWICK_AUTH_H
#define CWICK_AUTH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct cwick_auth cwick_auth;

/* `password` NULL/empty means "generate one"; `ttl_s` is the session lifetime
 * (clamped to 60..604800 seconds). Returns NULL only on allocation failure. */
cwick_auth *cwick_auth_new(const char *password, int64_t ttl_s);
void cwick_auth_free(cwick_auth *a);

/* Non-NULL only when cwick_auth_new generated the password. The pointer lives
 * as long as `a`; the caller prints it once at startup, never per request. */
const char *cwick_auth_generated_password(const cwick_auth *a);

/* Check the password and, on success, mint a new session token. The returned
 * string is malloc'd and is the value to put in the cookie/Bearer header.
 * Returns NULL for a wrong password, a locked account, or allocation failure. */
char *cwick_auth_login(cwick_auth *a, const char *password);

/* Seconds until a locked-out caller may retry, or 0 when not locked. */
int cwick_auth_retry_after(cwick_auth *a);

/* Validate a session token. Expired sessions are retired in place. */
bool cwick_auth_validate(cwick_auth *a, const char *token);

/* Drop one session token (logout). */
void cwick_auth_logout(cwick_auth *a, const char *token);

/* Constant-time helpers exposed for the unit test and the cookie parser. */
bool cwick_auth_token_equal(const char *a, const char *b);

/* ------------------------------------------------------------------ */
/* API keys                                                            */
/* ------------------------------------------------------------------ */

/* The secret handed to the caller once, and what is kept about it. `prefix`
 * is the display form ("cwick_" plus the first 8 characters): enough to tell
 * two keys apart in a log or a list, and useless as a credential — the log
 * lines identify which key served a request without ever carrying the key. */
typedef struct {
    const char *id;        /* 12-char handle used to revoke it */
    const char *name;      /* what the operator called it */
    const char *prefix;    /* "cwick_xxxxxxxx" */
    const char *hash_hex;  /* what to persist; never the secret itself */
    int64_t created_at;
    int64_t last_used_at;  /* 0 = never */
    bool enabled;
} cwick_api_key_info;

/* Mint a key. Returns the secret (malloc'd, caller frees — and should clear
 * it once it has been shown), or NULL on failure. `name` may be NULL/empty.
 * `out` is optional and receives the metadata; the `id`/`name`/`prefix`
 * pointers it hands back belong to the auth object. */
char *cwick_auth_key_create(cwick_auth *a, const char *name,
                           cwick_api_key_info *out);

/* Adopt a key supplied as plaintext — CWICK_API_KEY / --api-key, where the
 * operator (or a provisioning script) already chose the secret. Same storage
 * rules as cwick_auth_key_create: only the hash is kept. */
bool cwick_auth_key_import(cwick_auth *a, const char *name, const char *secret,
                          cwick_api_key_info *out);

/* Adopt a key whose hash was persisted by an earlier run. `hash_hex` is the
 * 64-char SHA-256 of the original secret; the secret itself is gone and this
 * module never needs it. Returns false when the id is taken or the hash is
 * malformed. */
bool cwick_auth_key_load(cwick_auth *a, const char *id, const char *name,
                        const char *hash_hex, const char *prefix,
                        int64_t created_at, int64_t last_used_at, bool enabled);

/* Revoke one key. Returns false when no such id exists. */
bool cwick_auth_key_delete(cwick_auth *a, const char *id);

/* Enumerate keys for GET /v1/auth/keys. */
void cwick_auth_keys_each(cwick_auth *a,
                         void (*cb)(const cwick_api_key_info *k, void *user),
                         void *user);

/* Validate a bearer secret. `id_out`/`cap` return which key matched (for the
 * access log and for updating last_used); `id_out` may be NULL. A session
 * token is *not* accepted here: the two credential types are checked
 * separately so that a leaked session can be told from a leaked key. */
bool cwick_auth_key_check(cwick_auth *a, const char *secret, char *id_out,
                         size_t cap);

/* The display prefix of a key, by id — what the access log prints. Returns a
 * static placeholder ("cwick_") when the id is unknown or already revoked. */
const char *cwick_auth_key_prefix_of(cwick_auth *a, const char *id);

/* The secret's fixed prefix. Exposed so a client can be told what shape to
 * expect and so a startup line can print a recognizable credential. */
const char *cwick_auth_key_prefix(void);

#endif
