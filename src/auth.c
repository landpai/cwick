#include "auth.h"

#include <CommonCrypto/CommonCrypto.h>
#include <CommonCrypto/CommonKeyDerivation.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define AUTH_SESSIONS 64
#define AUTH_TOKEN_LEN 64          /* 32 random bytes, hex */
#define AUTH_SALT_LEN 16
#define AUTH_HASH_LEN 32
#define AUTH_ITERATIONS 120000
#define AUTH_MAX_FAILS 5
#define AUTH_LOCK_SECONDS 60
#define AUTH_MIN_TTL 60
#define AUTH_MAX_TTL 604800

/* API keys: "cwick_" + 32 base62 characters = 160 bits of randomness from
 * arc4random_buf. The prefix is not decoration: it tells a human (and a secret
 * scanner, and the operator reading a config file) what this string is, and it
 * costs nothing. */
#define AUTH_KEYS 32
#define AUTH_KEY_ID_LEN 12
#define AUTH_KEY_SECRET_CHARS 32
#define AUTH_KEY_PREFIX_SHOWN 8
#define AUTH_KEY_SECRET_LEN (5 + AUTH_KEY_SECRET_CHARS)   /* "cwick_" + body */
#define AUTH_KEY_NAME_LEN 64
#define AUTH_KEY_HASH_HEX (AUTH_HASH_LEN * 2 + 1)

typedef struct {
    char token[AUTH_TOKEN_LEN + 1];
    int64_t expires;
    bool used;
} auth_session;

typedef struct {
    char id[AUTH_KEY_ID_LEN + 1];
    char name[AUTH_KEY_NAME_LEN];
    char prefix[5 + AUTH_KEY_PREFIX_SHOWN + 1];
    uint8_t hash[AUTH_HASH_LEN];      /* SHA-256 of the secret, never the secret */
    char hash_hex[AUTH_KEY_HASH_HEX]; /* the persisted form of the above */
    int64_t created_at;
    int64_t last_used_at;
    bool enabled;
    bool used;
} api_key;

struct cwick_auth {
    uint8_t salt[AUTH_SALT_LEN];
    uint8_t hash[AUTH_HASH_LEN];
    char *generated;               /* only when generated, for the startup line */
    pthread_mutex_t mtx;
    auth_session sessions[AUTH_SESSIONS];
    api_key keys[AUTH_KEYS];
    int64_t ttl_s;
    int fails;
    int64_t window_start;
    int64_t locked_until;
};

static int64_t now_s(void) { return (int64_t)time(NULL); }

static void secure_clear(void *p, size_t n) {
    volatile unsigned char *v = (volatile unsigned char *)p;
    while (n--) *v++ = 0;
}

static void random_bytes(uint8_t *p, size_t n) {
    arc4random_buf(p, n);
}

static void hex_encode(const uint8_t *in, size_t n, char *out) {
    static const char h[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        out[i * 2] = h[in[i] >> 4];
        out[i * 2 + 1] = h[in[i] & 15];
    }
    out[n * 2] = 0;
}

bool cwick_auth_token_equal(const char *a, const char *b) {
    if (!a || !b) return false;
    size_t la = strlen(a), lb = strlen(b);
    unsigned diff = (unsigned)(la ^ lb);
    size_t n = la < lb ? la : lb;
    for (size_t i = 0; i < n; i++)
        diff |= (unsigned char)a[i] ^ (unsigned char)b[i];
    return diff == 0;
}

static void derive(const char *password, const uint8_t *salt, uint8_t *out) {
    /* CommonCrypto is part of libSystem on macOS; no extra dependency. */
    CCKeyDerivationPBKDF(kCCPBKDF2, password, strlen(password),
                         salt, AUTH_SALT_LEN, kCCPRFHmacAlgSHA256,
                         AUTH_ITERATIONS, out, AUTH_HASH_LEN);
}

static char *make_password(void) {
    /* Avoid 0/O, 1/l/I: this is read from a terminal/log by a human. */
    static const char alphabet[] =
        "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789";
    uint8_t r[14];
    random_bytes(r, sizeof(r));
    char *p = malloc(sizeof(r) + 1);
    if (!p) return NULL;
    for (size_t i = 0; i < sizeof(r); i++)
        p[i] = alphabet[r[i] % (sizeof(alphabet) - 1)];
    p[sizeof(r)] = 0;
    return p;
}

cwick_auth *cwick_auth_new(const char *password, int64_t ttl_s) {
    cwick_auth *a = calloc(1, sizeof(*a));
    if (!a) return NULL;
    pthread_mutex_init(&a->mtx, NULL);
    if (ttl_s < AUTH_MIN_TTL) ttl_s = AUTH_MIN_TTL;
    if (ttl_s > AUTH_MAX_TTL) ttl_s = AUTH_MAX_TTL;
    a->ttl_s = ttl_s;
    const char *pw = password && *password ? password : NULL;
    if (!pw) {
        a->generated = make_password();
        if (!a->generated) { pthread_mutex_destroy(&a->mtx); free(a); return NULL; }
        pw = a->generated;
    }
    random_bytes(a->salt, sizeof(a->salt));
    derive(pw, a->salt, a->hash);
    /* Keep a->generated alive: the startup line has to print it once. The
     * hash, not the plaintext, is what validation ever reads. */
    return a;
}

void cwick_auth_free(cwick_auth *a) {
    if (!a) return;
    pthread_mutex_lock(&a->mtx);
    secure_clear(a->hash, sizeof(a->hash));
    secure_clear(a->salt, sizeof(a->salt));
    memset(a->sessions, 0, sizeof(a->sessions));
    for (int i = 0; i < AUTH_KEYS; i++) {
        if (!a->keys[i].used) continue;
        secure_clear(a->keys[i].hash, sizeof(a->keys[i].hash));
        memset(&a->keys[i], 0, sizeof(a->keys[i]));
    }
    if (a->generated) {
        secure_clear(a->generated, strlen(a->generated));
        free(a->generated);
    }
    a->generated = NULL;
    pthread_mutex_unlock(&a->mtx);
    pthread_mutex_destroy(&a->mtx);
    free(a);
}

const char *cwick_auth_generated_password(const cwick_auth *a) {
    return a ? a->generated : NULL;
}

int cwick_auth_retry_after(cwick_auth *a) {
    if (!a) return 0;
    pthread_mutex_lock(&a->mtx);
    int64_t now = now_s();
    int retry = a->locked_until > now ? (int)(a->locked_until - now) : 0;
    pthread_mutex_unlock(&a->mtx);
    return retry;
}

char *cwick_auth_login(cwick_auth *a, const char *password) {
    if (!a || !password) return NULL;
    char *token = NULL;
    pthread_mutex_lock(&a->mtx);
    int64_t now = now_s();
    if (a->locked_until > now) {
        pthread_mutex_unlock(&a->mtx);
        return NULL;
    }
    if (now - a->window_start > AUTH_LOCK_SECONDS) {
        a->window_start = now;
        a->fails = 0;
    }
    uint8_t got[AUTH_HASH_LEN];
    derive(password, a->salt, got);
    unsigned diff = 0;
    for (size_t i = 0; i < AUTH_HASH_LEN; i++) diff |= got[i] ^ a->hash[i];
    secure_clear(got, sizeof(got));
    if (diff != 0) {
        a->fails++;
        if (a->fails >= AUTH_MAX_FAILS) {
            a->fails = 0;
            a->locked_until = now + AUTH_LOCK_SECONDS;
        }
        pthread_mutex_unlock(&a->mtx);
        return NULL;
    }
    a->fails = 0;
    int slot = -1;
    for (int i = 0; i < AUTH_SESSIONS; i++) {
        if (!a->sessions[i].used || a->sessions[i].expires <= now) {
            slot = i;
            break;
        }
    }
    if (slot >= 0) {
        uint8_t raw[32];
        random_bytes(raw, sizeof(raw));
        hex_encode(raw, sizeof(raw), a->sessions[slot].token);
        a->sessions[slot].expires = now + a->ttl_s;
        a->sessions[slot].used = true;
        token = strdup(a->sessions[slot].token);
        secure_clear(raw, sizeof(raw));
    }
    pthread_mutex_unlock(&a->mtx);
    return token;
}

bool cwick_auth_validate(cwick_auth *a, const char *token) {
    if (!a || !token || !*token) return false;
    bool ok = false;
    pthread_mutex_lock(&a->mtx);
    int64_t now = now_s();
    for (int i = 0; i < AUTH_SESSIONS; i++) {
        if (!a->sessions[i].used) continue;
        if (a->sessions[i].expires <= now) {
            a->sessions[i].used = false;
            secure_clear(a->sessions[i].token, sizeof(a->sessions[i].token));
            continue;
        }
        if (cwick_auth_token_equal(a->sessions[i].token, token)) { ok = true; break; }
    }
    pthread_mutex_unlock(&a->mtx);
    return ok;
}

void cwick_auth_logout(cwick_auth *a, const char *token) {
    if (!a || !token) return;
    pthread_mutex_lock(&a->mtx);
    for (int i = 0; i < AUTH_SESSIONS; i++) {
        if (a->sessions[i].used && cwick_auth_token_equal(a->sessions[i].token, token)) {
            a->sessions[i].used = false;
            secure_clear(a->sessions[i].token, sizeof(a->sessions[i].token));
            break;
        }
    }
    pthread_mutex_unlock(&a->mtx);
}

/* ------------------------------------------------------------------ */
/* API keys                                                            */
/* ------------------------------------------------------------------ */

static const char kKeyAlphabet[] =
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/* Rejection sampling rather than `% 62`: a plain modulo of a uniform byte
 * favours the first four characters of the alphabet. Over 160 bits that bias
 * is nowhere near exploitable, but drawing uniformly costs one comparison. */
static void random_chars(char *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint8_t r;
        do { arc4random_buf(&r, 1); } while (r >= 248);   /* 248 = 4 * 62 */
        out[i] = kKeyAlphabet[r % 62];
    }
    out[n] = '\0';
}

/* A single SHA-256, not PBKDF2. Stretching exists to make a *guessable*
 * secret expensive to test; this one is 160 random bits, so the only thing
 * iterations would buy is latency on every request of every agent. */
static void key_hash(const char *secret, uint8_t out[AUTH_HASH_LEN]) {
    CC_SHA256((const void *)secret, (CC_LONG)strlen(secret), out);
}

static int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static bool hex_decode(const char *s, uint8_t *out, size_t n) {
    if (!s) return false;
    for (size_t i = 0; i < n; i++) {
        int hi = hex_digit(s[i * 2]), lo = hex_digit(s[i * 2 + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    return s[n * 2] == '\0';
}

static void hex_of(const uint8_t *in, size_t n, char *out) {
    hex_encode(in, n, out);
}

const char *cwick_auth_key_prefix(void) { return "cwick_"; }

char *cwick_auth_key_create(cwick_auth *a, const char *name,
                           cwick_api_key_info *out) {
    if (!a) return NULL;
    pthread_mutex_lock(&a->mtx);
    int slot = -1;
    for (int i = 0; i < AUTH_KEYS; i++) if (!a->keys[i].used) { slot = i; break; }
    if (slot < 0) { pthread_mutex_unlock(&a->mtx); return NULL; }
    api_key *k = &a->keys[slot];
    memset(k, 0, sizeof(*k));
    char body[AUTH_KEY_SECRET_CHARS + 1];
    random_chars(body, AUTH_KEY_SECRET_CHARS);
    random_chars(k->id, AUTH_KEY_ID_LEN);
    char secret[AUTH_KEY_SECRET_LEN + 1];
    snprintf(secret, sizeof(secret), "cwick_%s", body);
    snprintf(k->prefix, sizeof(k->prefix), "cwick_%.*s", AUTH_KEY_PREFIX_SHOWN, body);
    snprintf(k->name, sizeof(k->name), "%s", name && *name ? name : "key");
    key_hash(secret, k->hash);
    hex_of(k->hash, AUTH_HASH_LEN, k->hash_hex);
    k->created_at = now_s();
    k->last_used_at = 0;
    k->enabled = true;
    k->used = true;
    if (out) {
        out->id = k->id;
        out->name = k->name;
        out->prefix = k->prefix;
        out->hash_hex = k->hash_hex;
        out->created_at = k->created_at;
        out->last_used_at = 0;
        out->enabled = true;
    }
    char *copy = strdup(secret);
    secure_clear(secret, sizeof(secret));
    secure_clear(body, sizeof(body));
    pthread_mutex_unlock(&a->mtx);
    return copy;
}

bool cwick_auth_key_load(cwick_auth *a, const char *id, const char *name,
                        const char *hash_hex, const char *prefix,
                        int64_t created_at, int64_t last_used_at, bool enabled) {
    if (!a || !id || !*id || !hash_hex) return false;
    uint8_t h[AUTH_HASH_LEN];
    if (!hex_decode(hash_hex, h, AUTH_HASH_LEN)) return false;
    pthread_mutex_lock(&a->mtx);
    int slot = -1;
    bool dupe = false;
    for (int i = 0; i < AUTH_KEYS; i++) {
        if (a->keys[i].used && strcmp(a->keys[i].id, id) == 0) { dupe = true; break; }
        if (slot < 0 && !a->keys[i].used) slot = i;
    }
    if (dupe || slot < 0) { pthread_mutex_unlock(&a->mtx); return false; }
    api_key *k = &a->keys[slot];
    memset(k, 0, sizeof(*k));
    snprintf(k->id, sizeof(k->id), "%s", id);
    snprintf(k->name, sizeof(k->name), "%s", name && *name ? name : "key");
    snprintf(k->prefix, sizeof(k->prefix), "%s", prefix ? prefix : "cwick_");
    memcpy(k->hash, h, AUTH_HASH_LEN);
    hex_of(k->hash, AUTH_HASH_LEN, k->hash_hex);
    k->created_at = created_at;
    k->last_used_at = last_used_at;
    k->enabled = enabled;
    k->used = true;
    pthread_mutex_unlock(&a->mtx);
    secure_clear(h, sizeof(h));
    return true;
}

bool cwick_auth_key_delete(cwick_auth *a, const char *id) {
    if (!a || !id) return false;
    pthread_mutex_lock(&a->mtx);
    bool ok = false;
    for (int i = 0; i < AUTH_KEYS; i++) {
        if (!a->keys[i].used || strcmp(a->keys[i].id, id) != 0) continue;
        secure_clear(a->keys[i].hash, sizeof(a->keys[i].hash));
        memset(&a->keys[i], 0, sizeof(a->keys[i]));
        ok = true;
        break;
    }
    pthread_mutex_unlock(&a->mtx);
    return ok;
}

void cwick_auth_keys_each(cwick_auth *a,
                         void (*cb)(const cwick_api_key_info *k, void *user),
                         void *user) {
    if (!a || !cb) return;
    /* The lock is held across the callback on purpose: the pointers handed to
     * it belong to the key slots, so a concurrent delete would dangle. The
     * callback must not call back into this module. */
    pthread_mutex_lock(&a->mtx);
    for (int i = 0; i < AUTH_KEYS; i++) {
        if (!a->keys[i].used) continue;
        cwick_api_key_info info;
        info.id = a->keys[i].id;
        info.name = a->keys[i].name;
        info.prefix = a->keys[i].prefix;
        info.created_at = a->keys[i].created_at;
        info.last_used_at = a->keys[i].last_used_at;
        info.enabled = a->keys[i].enabled;
        cb(&info, user);
    }
    pthread_mutex_unlock(&a->mtx);
}

bool cwick_auth_key_check(cwick_auth *a, const char *secret, char *id_out,
                         size_t cap) {
    if (!a || !secret || !*secret) return false;
    uint8_t got[AUTH_HASH_LEN];
    key_hash(secret, got);
    bool ok = false;
    pthread_mutex_lock(&a->mtx);
    for (int i = 0; i < AUTH_KEYS; i++) {
        api_key *k = &a->keys[i];
        if (!k->used || !k->enabled) continue;
        unsigned diff = 0;
        for (size_t j = 0; j < AUTH_HASH_LEN; j++) diff |= got[j] ^ k->hash[j];
        if (diff != 0) continue;
        ok = true;
        k->last_used_at = now_s();
        if (id_out && cap) snprintf(id_out, cap, "%s", k->id);
        break;
    }
    pthread_mutex_unlock(&a->mtx);
    secure_clear(got, sizeof(got));
    return ok;
}

const char *cwick_auth_key_prefix_of(cwick_auth *a, const char *id) {
    if (!a || !id) return "cwick_";
    const char *found = NULL;
    pthread_mutex_lock(&a->mtx);
    for (int i = 0; i < AUTH_KEYS; i++) {
        if (a->keys[i].used && strcmp(a->keys[i].id, id) == 0) {
            found = a->keys[i].prefix;
            break;
        }
    }
    pthread_mutex_unlock(&a->mtx);
    return found ? found : "cwick_";
}

bool cwick_auth_key_import(cwick_auth *a, const char *name, const char *secret,
                          cwick_api_key_info *out) {
    if (!a || !secret || !*secret) return false;
    pthread_mutex_lock(&a->mtx);
    int slot = -1;
    for (int i = 0; i < AUTH_KEYS; i++) if (!a->keys[i].used) { slot = i; break; }
    if (slot < 0) { pthread_mutex_unlock(&a->mtx); return false; }
    api_key *k = &a->keys[slot];
    memset(k, 0, sizeof(*k));
    random_chars(k->id, AUTH_KEY_ID_LEN);
    snprintf(k->name, sizeof(k->name), "%s", name && *name ? name : "key");
    /* The prefix is only for display, so it is derived from the secret the
     * operator chose; a too-short secret simply shows a short prefix. */
    snprintf(k->prefix, sizeof(k->prefix), "%.13s", secret);
    key_hash(secret, k->hash);
    hex_of(k->hash, AUTH_HASH_LEN, k->hash_hex);
    k->created_at = now_s();
    k->last_used_at = 0;
    k->enabled = true;
    k->used = true;
    if (out) {
        out->id = k->id;
        out->name = k->name;
        out->prefix = k->prefix;
        out->hash_hex = k->hash_hex;
        out->created_at = k->created_at;
        out->last_used_at = 0;
        out->enabled = true;
    }
    pthread_mutex_unlock(&a->mtx);
    return true;
}
