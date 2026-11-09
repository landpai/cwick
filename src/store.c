/* store.c — sqlite persistence for the built-in apps.
 *
 * See store.h for why this exists. Implementation notes that matter:
 *
 *  - One connection, one mutex. The HTTP server gives every accepted socket
 *    its own thread, so several connections can be writing an app's state at
 *    the same moment. sqlite3 in serialized mode would make the handle safe,
 *    but the prepared statements cached below are not re-entrant, so the
 *    mutex covers whole operations rather than single calls.
 *
 *  - WAL + NORMAL sync. An app's move log is worth nothing if the machine
 *    loses power, but it is also not worth an fsync per row: the game is
 *    playable while the writes are merely durable across process restarts,
 *    which WAL gives without the write amplification of FULL sync.
 *
 *  - Everything is bound, nothing is interpolated. app ids arrive from a URL.
 *
 *  - Payloads are parsed as JSON on the way in and handed back as the exact
 *    text that arrived. That is the only validation that lets the server stay
 *    ignorant of what an app stores while still guaranteeing an app can only
 *    emit a well-formed envelope for another app (or itself) to read.
 */
#include "store.h"

#include <pthread.h>
#include <sqlite3.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "json.h"
#include "util.h"

/* Bump only when a change cannot be expressed as an IF NOT EXISTS addition. */
#define STORE_SCHEMA_VERSION 1

struct cwick_store {
    sqlite3 *db;
    char *path;
    pthread_mutex_t mtx;
    /* Cached statements, prepared on first use. */
    sqlite3_stmt *kv_put;
    sqlite3_stmt *kv_get;
    sqlite3_stmt *kv_del;
    sqlite3_stmt *kv_each;
    sqlite3_stmt *game_put;
    sqlite3_stmt *game_del;
    sqlite3_stmt *ev_add;
};

static int64_t now_s(void) { return (int64_t)time(NULL); }

/* ------------------------------------------------------------------ */
/* validation helpers                                                  */
/* ------------------------------------------------------------------ */

bool cwick_store_id_ok(const char *app) {
    if (!app || !*app) return false;
    size_t n = strlen(app);
    if (n > CWICK_STORE_MAX_APP_ID) return false;
    for (size_t i = 0; i < n; i++) {
        char c = app[i];
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_') continue;
        return false;
    }
    return true;
}

/* A value/payload is accepted only if it parses. The parse is thrown away:
 * what gets stored is the caller's own bytes, so field order and number
 * formatting survive a round trip untouched. */
static bool json_ok(const char *text, size_t max, cwick_error *err) {
    if (!text) {
        if (err) cwick_error_set(err, CWICK_ERR_ARGUMENT, "value is missing");
        return false;
    }
    size_t n = strlen(text);
    if (n == 0) {
        if (err) cwick_error_set(err, CWICK_ERR_ARGUMENT, "value is empty");
        return false;
    }
    if (n > max) {
        if (err) cwick_error_set(err, CWICK_ERR_ARGUMENT,
                                "value is %zu bytes, limit is %zu", n, max);
        return false;
    }
    cwick_error e;
    cwick_error_clear(&e);
    json_value *v = cwick_json_parse(text, n, &e);
    if (!v) {
        if (err) cwick_error_set(err, CWICK_ERR_ARGUMENT, "value is not JSON: %s",
                                cwick_error_message(&e));
        return false;
    }
    cwick_json_free(v);
    return true;
}

/* ------------------------------------------------------------------ */
/* open / close                                                        */
/* ------------------------------------------------------------------ */

/* $CWICK_STORE, else ~/.cwick/apps.db. The directory is created because a
 * default that fails on a fresh machine is worse than one that writes. */
static char *default_path(void) {
    const char *env = getenv("CWICK_STORE");
    if (env && *env) return cwick_strdup(env);
    const char *home = getenv("HOME");
    char dir[1024];
    if (home && *home) snprintf(dir, sizeof(dir), "%s/.cwick", home);
    else snprintf(dir, sizeof(dir), "/tmp/.cwick");
    (void)cwick_mkdir_p(dir);
    char out[1152];
    cwick_path_join(out, sizeof(out), dir, "apps.db");
    return cwick_strdup(out);
}

static void make_parent_dir(const char *path) {
    const char *slash = strrchr(path, '/');
    if (!slash || slash == path) return;
    size_t n = (size_t)(slash - path);
    char parent[1024];
    if (n >= sizeof(parent)) return;
    memcpy(parent, path, n);
    parent[n] = '\0';
    (void)cwick_mkdir_p(parent);
}

static bool exec_sql(cwick_store *s, const char *sql, cwick_error *err) {
    char *msg = NULL;
    int rc = sqlite3_exec(s->db, sql, NULL, NULL, &msg);
    if (rc != SQLITE_OK) {
        if (err) cwick_error_set(err, CWICK_ERR_IO, "sqlite: %s",
                                msg ? msg : sqlite3_errstr(rc));
        sqlite3_free(msg);
        return false;
    }
    return true;
}

static const char *kSchema =
    "PRAGMA journal_mode=WAL;"
    "PRAGMA synchronous=NORMAL;"
    "PRAGMA foreign_keys=ON;"
    "CREATE TABLE IF NOT EXISTS app_kv ("
    "  app_id TEXT NOT NULL,"
    "  k TEXT NOT NULL,"
    "  v TEXT NOT NULL,"
    "  updated_at INTEGER NOT NULL,"
    "  PRIMARY KEY (app_id, k));"
    "CREATE TABLE IF NOT EXISTS app_games ("
    "  app_id TEXT NOT NULL,"
    "  game_id TEXT NOT NULL,"
    "  started_at INTEGER NOT NULL,"
    "  ended_at INTEGER NOT NULL DEFAULT 0,"
    "  outcome TEXT NOT NULL DEFAULT '',"
    "  moves INTEGER NOT NULL DEFAULT 0,"
    "  PRIMARY KEY (app_id, game_id));"
    "CREATE TABLE IF NOT EXISTS app_events ("
    "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
    "  app_id TEXT NOT NULL,"
    "  game_id TEXT NOT NULL,"
    "  seq INTEGER NOT NULL DEFAULT 0,"
    "  ts INTEGER NOT NULL,"
    "  kind TEXT NOT NULL,"
    "  payload TEXT NOT NULL);"
    "CREATE INDEX IF NOT EXISTS idx_events_app_game"
    "  ON app_events (app_id, game_id, id);"
    "CREATE INDEX IF NOT EXISTS idx_games_app_started"
    "  ON app_games (app_id, started_at DESC);"
    /* API keys: see store.h. `hash` is the only thing that identifies a key;
     * `prefix` exists so a list of keys is readable without it. */
    "CREATE TABLE IF NOT EXISTS auth_keys ("
    "  id TEXT PRIMARY KEY,"
    "  name TEXT NOT NULL,"
    "  hash TEXT NOT NULL,"
    "  prefix TEXT NOT NULL,"
    "  created_at INTEGER NOT NULL,"
    "  last_used_at INTEGER NOT NULL DEFAULT 0,"
    "  enabled INTEGER NOT NULL DEFAULT 1);";

cwick_store *cwick_store_open(const char *path, cwick_error *err) {
    if (err) cwick_error_clear(err);
    char *p = (path && *path) ? cwick_strdup(path) : default_path();
    if (!p) {
        if (err) cwick_error_set(err, CWICK_ERR_MEMORY, "out of memory");
        return NULL;
    }
    make_parent_dir(p);

    cwick_store *s = (cwick_store *)calloc(1, sizeof(*s));
    if (!s) {
        free(p);
        if (err) cwick_error_set(err, CWICK_ERR_MEMORY, "out of memory");
        return NULL;
    }
    s->path = p;
    pthread_mutex_init(&s->mtx, NULL);

    int rc = sqlite3_open_v2(p, &s->db,
                             SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, NULL);
    if (rc != SQLITE_OK) {
        if (err) cwick_error_set(err, CWICK_ERR_IO, "cannot open %s: %s", p,
                                s->db ? sqlite3_errmsg(s->db)
                                      : sqlite3_errstr(rc));
        sqlite3_close(s->db);
        pthread_mutex_destroy(&s->mtx);
        free(s->path);
        free(s);
        return NULL;
    }
    /* Two writers (two browser tabs) should wait, not fail. */
    sqlite3_busy_timeout(s->db, 3000);

    if (!exec_sql(s, kSchema, err)) {
        cwick_store_close(s);
        return NULL;
    }
    char ver[64];
    snprintf(ver, sizeof(ver), "PRAGMA user_version=%d;", STORE_SCHEMA_VERSION);
    if (!exec_sql(s, ver, err)) {
        cwick_store_close(s);
        return NULL;
    }
    return s;
}

void cwick_store_close(cwick_store *s) {
    if (!s) return;
    if (s->db) {
        sqlite3_finalize(s->kv_put);
        sqlite3_finalize(s->kv_get);
        sqlite3_finalize(s->kv_del);
        sqlite3_finalize(s->kv_each);
        sqlite3_finalize(s->game_put);
        sqlite3_finalize(s->game_del);
        sqlite3_finalize(s->ev_add);
        sqlite3_close(s->db);
    }
    pthread_mutex_destroy(&s->mtx);
    free(s->path);
    free(s);
}

const char *cwick_store_path(const cwick_store *s) { return s ? s->path : NULL; }

/* $CWICK_STORE, else ~/.cwick/apps — a directory. `--store` decides whether the
 * value it holds names this directory or one single shared file. */
const char *cwick_store_default_dir(void) {
    const char *env = getenv("CWICK_STORE");
    if (env && *env) return env;
    static char dir[1024];
    static bool built = false;
    if (!built) {
        const char *home = getenv("HOME");
        if (home && *home) snprintf(dir, sizeof(dir), "%s/.cwick/apps", home);
        else snprintf(dir, sizeof(dir), "/tmp/.cwick/apps");
        (void)cwick_mkdir_p(dir);
        built = true;
    }
    return dir;
}

bool cwick_store_path_is_file(const char *path) {
    if (!path || !*path) return false;
    struct stat st;
    if (stat(path, &st) == 0) return S_ISREG(st.st_mode) != 0;
    /* Not on disk yet: a suffix is enough to say what it will be, so that
     * `--store /data/app.db` on a fresh machine is a file and
     * `--store /data/apps` is a directory without a mode flag. */
    static const char *suf[] = { ".db", ".db3", ".sqlite", ".sqlite3" };
    size_t n = strlen(path);
    for (size_t i = 0; i < sizeof(suf) / sizeof(suf[0]); i++) {
        size_t sl = strlen(suf[i]);
        if (n > sl && strcasecmp(path + n - sl, suf[i]) == 0) return true;
    }
    return false;
}

/* Prepare on first use and keep the statement. Caller holds the mutex. */
static sqlite3_stmt *prepared(cwick_store *s, sqlite3_stmt **slot,
                              const char *sql, cwick_error *err) {
    if (*slot) {
        sqlite3_reset(*slot);
        sqlite3_clear_bindings(*slot);
        return *slot;
    }
    sqlite3_stmt *st = NULL;
    int rc = sqlite3_prepare_v2(s->db, sql, -1, &st, NULL);
    if (rc != SQLITE_OK || !st) {
        if (err) cwick_error_set(err, CWICK_ERR_IO, "sqlite prepare: %s",
                                sqlite3_errmsg(s->db));
        return NULL;
    }
    *slot = st;
    return st;
}

static bool step_done(cwick_store *s, sqlite3_stmt *st, cwick_error *err) {
    int rc = sqlite3_step(st);
    if (rc != SQLITE_DONE) {
        if (err) cwick_error_set(err, CWICK_ERR_IO, "sqlite: %s",
                                sqlite3_errmsg(s->db));
        return false;
    }
    return true;
}

/* ------------------------------------------------------------------ */
/* key / value                                                         */
/* ------------------------------------------------------------------ */

bool cwick_store_kv_put(cwick_store *s, const char *app, const char *key,
                       const char *value, cwick_error *err) {
    if (!s || !cwick_store_id_ok(app) || !key || !*key ||
        strlen(key) > CWICK_STORE_MAX_KEY) {
        if (err) cwick_error_set(err, CWICK_ERR_ARGUMENT, "bad app id or key");
        return false;
    }
    if (!json_ok(value, CWICK_STORE_MAX_VALUE, err)) return false;

    pthread_mutex_lock(&s->mtx);
    sqlite3_stmt *st = prepared(s, &s->kv_put,
        "INSERT INTO app_kv (app_id, k, v, updated_at) VALUES (?, ?, ?, ?) "
        "ON CONFLICT (app_id, k) DO UPDATE SET v = excluded.v, "
        "updated_at = excluded.updated_at;", err);
    if (!st) { pthread_mutex_unlock(&s->mtx); return false; }
    sqlite3_bind_text(st, 1, app, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(st, 2, key, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(st, 3, value, -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(st, 4, now_s());
    bool ok = step_done(s, st, err);
    sqlite3_reset(st);
    pthread_mutex_unlock(&s->mtx);
    return ok;
}

char *cwick_store_kv_get(cwick_store *s, const char *app, const char *key) {
    if (!s || !cwick_store_id_ok(app) || !key || !*key) return NULL;
    pthread_mutex_lock(&s->mtx);
    sqlite3_stmt *st = prepared(s, &s->kv_get,
        "SELECT v FROM app_kv WHERE app_id = ? AND k = ?;", NULL);
    if (!st) { pthread_mutex_unlock(&s->mtx); return NULL; }
    sqlite3_bind_text(st, 1, app, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(st, 2, key, -1, SQLITE_TRANSIENT);
    char *out = NULL;
    if (sqlite3_step(st) == SQLITE_ROW) {
        const unsigned char *v = sqlite3_column_text(st, 0);
        if (v) out = cwick_strdup((const char *)v);
    }
    sqlite3_reset(st);
    pthread_mutex_unlock(&s->mtx);
    return out;
}

bool cwick_store_kv_del(cwick_store *s, const char *app, const char *key,
                       cwick_error *err) {
    if (!s || !cwick_store_id_ok(app) || !key || !*key) {
        if (err) cwick_error_set(err, CWICK_ERR_ARGUMENT, "bad app id or key");
        return false;
    }
    pthread_mutex_lock(&s->mtx);
    sqlite3_stmt *st = prepared(s, &s->kv_del,
        "DELETE FROM app_kv WHERE app_id = ? AND k = ?;", err);
    if (!st) { pthread_mutex_unlock(&s->mtx); return false; }
    sqlite3_bind_text(st, 1, app, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(st, 2, key, -1, SQLITE_TRANSIENT);
    bool ok = step_done(s, st, err);
    sqlite3_reset(st);
    pthread_mutex_unlock(&s->mtx);
    return ok;
}

void cwick_store_kv_each(cwick_store *s, const char *app,
                        void (*cb)(const char *key, const char *value,
                                   int64_t updated_at, void *user),
                        void *user) {
    if (!s || !cb || !cwick_store_id_ok(app)) return;
    pthread_mutex_lock(&s->mtx);
    sqlite3_stmt *st = prepared(s, &s->kv_each,
        "SELECT k, v, updated_at FROM app_kv WHERE app_id = ? "
        "ORDER BY updated_at;", NULL);
    if (!st) { pthread_mutex_unlock(&s->mtx); return; }
    sqlite3_bind_text(st, 1, app, -1, SQLITE_TRANSIENT);
    while (sqlite3_step(st) == SQLITE_ROW) {
        const char *k = (const char *)sqlite3_column_text(st, 0);
        const char *v = (const char *)sqlite3_column_text(st, 1);
        cb(k ? k : "", v ? v : "null", sqlite3_column_int64(st, 2), user);
    }
    sqlite3_reset(st);
    pthread_mutex_unlock(&s->mtx);
}

/* ------------------------------------------------------------------ */
/* games                                                               */
/* ------------------------------------------------------------------ */

bool cwick_store_game_put(cwick_store *s, const char *app, const char *game_id,
                         int64_t started_at, const char *outcome,
                         int64_t ended_at, int64_t moves, cwick_error *err) {
    if (!s || !cwick_store_id_ok(app) || !game_id || !*game_id ||
        strlen(game_id) > CWICK_STORE_MAX_GAME_ID) {
        if (err) cwick_error_set(err, CWICK_ERR_ARGUMENT, "bad app id or game id");
        return false;
    }
    if (outcome && strlen(outcome) > CWICK_STORE_MAX_KIND) {
        if (err) cwick_error_set(err, CWICK_ERR_ARGUMENT, "outcome is too long");
        return false;
    }
    pthread_mutex_lock(&s->mtx);
    sqlite3_stmt *st = prepared(s, &s->game_put,
        "INSERT INTO app_games (app_id, game_id, started_at, ended_at, outcome, moves)"
        " VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (app_id, game_id) DO UPDATE SET"
        "  ended_at = excluded.ended_at,"
        "  outcome  = CASE WHEN excluded.outcome = '' THEN app_games.outcome"
        "                  ELSE excluded.outcome END,"
        "  moves    = CASE WHEN excluded.moves = 0 THEN app_games.moves"
        "                  ELSE excluded.moves END,"
        "  started_at = CASE WHEN excluded.started_at = 0 THEN app_games.started_at"
        "                    ELSE excluded.started_at END;", err);
    if (!st) { pthread_mutex_unlock(&s->mtx); return false; }
    sqlite3_bind_text(st, 1, app, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(st, 2, game_id, -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(st, 3, started_at);
    sqlite3_bind_int64(st, 4, ended_at);
    sqlite3_bind_text(st, 5, outcome ? outcome : "", -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(st, 6, moves);
    bool ok = step_done(s, st, err);
    sqlite3_reset(st);
    pthread_mutex_unlock(&s->mtx);
    return ok;
}

void cwick_store_games(cwick_store *s, const char *app, int64_t limit,
                      void (*cb)(const cwick_store_game *g, void *user),
                      void *user) {
    if (!s || !cb || !cwick_store_id_ok(app)) return;
    if (limit <= 0 || limit > CWICK_STORE_MAX_EVENTS) limit = CWICK_STORE_MAX_EVENTS;
    pthread_mutex_lock(&s->mtx);
    sqlite3_stmt *st = NULL;
    const char *sql = "SELECT game_id, started_at, ended_at, outcome, moves "
                      "FROM app_games WHERE app_id = ? "
                      "ORDER BY started_at DESC, rowid DESC LIMIT ?;";
    int rc = sqlite3_prepare_v2(s->db, sql, -1, &st, NULL);
    if (rc != SQLITE_OK || !st) { pthread_mutex_unlock(&s->mtx); return; }
    sqlite3_bind_text(st, 1, app, -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(st, 2, limit);
    while (sqlite3_step(st) == SQLITE_ROW) {
        cwick_store_game g;
        g.game_id = (const char *)sqlite3_column_text(st, 0);
        g.started_at = sqlite3_column_int64(st, 1);
        g.ended_at = sqlite3_column_int64(st, 2);
        g.outcome = (const char *)sqlite3_column_text(st, 3);
        g.moves = sqlite3_column_int64(st, 4);
        if (!g.game_id) g.game_id = "";
        if (!g.outcome) g.outcome = "";
        cb(&g, user);
    }
    sqlite3_finalize(st);
    pthread_mutex_unlock(&s->mtx);
}

bool cwick_store_game_del(cwick_store *s, const char *app, const char *game_id,
                         cwick_error *err) {
    if (!s || !cwick_store_id_ok(app) || !game_id || !*game_id) {
        if (err) cwick_error_set(err, CWICK_ERR_ARGUMENT, "bad app id or game id");
        return false;
    }
    /* One transaction: a half-deleted game leaves orphan steps that the replay
     * view would happily show without a parent row. */
    pthread_mutex_lock(&s->mtx);
    bool ok = exec_sql(s, "BEGIN IMMEDIATE;", err);
    if (ok) {
        sqlite3_stmt *st = prepared(s, &s->game_del,
            "DELETE FROM app_events WHERE app_id = ? AND game_id = ?;", err);
        if (!st) {
            ok = false;
        } else {
            sqlite3_bind_text(st, 1, app, -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(st, 2, game_id, -1, SQLITE_TRANSIENT);
            ok = step_done(s, st, err);
            sqlite3_reset(st);
        }
    }
    if (ok) {
        sqlite3_stmt *g = NULL;
        if (sqlite3_prepare_v2(s->db,
                "DELETE FROM app_games WHERE app_id = ? AND game_id = ?;",
                -1, &g, NULL) == SQLITE_OK && g) {
            sqlite3_bind_text(g, 1, app, -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(g, 2, game_id, -1, SQLITE_TRANSIENT);
            if (sqlite3_step(g) != SQLITE_DONE) {
                if (err) cwick_error_set(err, CWICK_ERR_IO, "sqlite: %s",
                                        sqlite3_errmsg(s->db));
                ok = false;
            }
            sqlite3_finalize(g);
        } else {
            if (err) cwick_error_set(err, CWICK_ERR_IO, "sqlite prepare: %s",
                                    sqlite3_errmsg(s->db));
            ok = false;
        }
    }
    if (ok) ok = exec_sql(s, "COMMIT;", err);
    else (void)exec_sql(s, "ROLLBACK;", NULL);
    pthread_mutex_unlock(&s->mtx);
    return ok;
}

/* ------------------------------------------------------------------ */
/* events                                                              */
/* ------------------------------------------------------------------ */

bool cwick_store_event_add(cwick_store *s, const char *app, const char *game_id,
                          int64_t seq, const char *kind, const char *payload,
                          cwick_error *err) {
    if (!s || !cwick_store_id_ok(app) || !game_id || !*game_id ||
        strlen(game_id) > CWICK_STORE_MAX_GAME_ID) {
        if (err) cwick_error_set(err, CWICK_ERR_ARGUMENT, "bad app id or game id");
        return false;
    }
    if (!kind || !*kind || strlen(kind) > CWICK_STORE_MAX_KIND) {
        if (err) cwick_error_set(err, CWICK_ERR_ARGUMENT, "bad event kind");
        return false;
    }
    if (!json_ok(payload, CWICK_STORE_MAX_PAYLOAD, err)) return false;

    pthread_mutex_lock(&s->mtx);
    sqlite3_stmt *st = prepared(s, &s->ev_add,
        "INSERT INTO app_events (app_id, game_id, seq, ts, kind, payload)"
        " VALUES (?, ?, ?, ?, ?, ?);", err);
    if (!st) { pthread_mutex_unlock(&s->mtx); return false; }
    sqlite3_bind_text(st, 1, app, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(st, 2, game_id, -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(st, 3, seq);
    sqlite3_bind_int64(st, 4, now_s());
    sqlite3_bind_text(st, 5, kind, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(st, 6, payload, -1, SQLITE_TRANSIENT);
    bool ok = step_done(s, st, err);
    sqlite3_reset(st);
    pthread_mutex_unlock(&s->mtx);
    return ok;
}

void cwick_store_events(cwick_store *s, const char *app, const char *game_id,
                       int64_t limit,
                       void (*cb)(const cwick_store_event *e, void *user),
                       void *user) {
    if (!s || !cb || !cwick_store_id_ok(app)) return;
    if (limit <= 0 || limit > CWICK_STORE_MAX_EVENTS) limit = CWICK_STORE_MAX_EVENTS;
    pthread_mutex_lock(&s->mtx);
    const char *sql = game_id && *game_id
        ? "SELECT id, game_id, seq, ts, kind, payload FROM app_events"
          " WHERE app_id = ? AND game_id = ? ORDER BY id LIMIT ?;"
        : "SELECT id, game_id, seq, ts, kind, payload FROM app_events"
          " WHERE app_id = ? ORDER BY id LIMIT ?;";
    sqlite3_stmt *st = NULL;
    if (sqlite3_prepare_v2(s->db, sql, -1, &st, NULL) != SQLITE_OK || !st) {
        pthread_mutex_unlock(&s->mtx);
        return;
    }
    int idx = 1;
    sqlite3_bind_text(st, idx++, app, -1, SQLITE_TRANSIENT);
    if (game_id && *game_id) sqlite3_bind_text(st, idx++, game_id, -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(st, idx, limit);
    while (sqlite3_step(st) == SQLITE_ROW) {
        cwick_store_event e;
        e.id = sqlite3_column_int64(st, 0);
        e.game_id = (const char *)sqlite3_column_text(st, 1);
        e.seq = sqlite3_column_int64(st, 2);
        e.ts = sqlite3_column_int64(st, 3);
        e.kind = (const char *)sqlite3_column_text(st, 4);
        e.payload = (const char *)sqlite3_column_text(st, 5);
        if (!e.game_id) e.game_id = "";
        if (!e.kind) e.kind = "";
        if (!e.payload) e.payload = "null";
        cb(&e, user);
    }
    sqlite3_finalize(st);
    pthread_mutex_unlock(&s->mtx);
}

/* ------------------------------------------------------------------ */
/* API keys                                                            */
/* ------------------------------------------------------------------ */

bool cwick_store_key_put(cwick_store *s, const cwick_store_key *k, cwick_error *err) {
    if (!s || !k || !k->id || !*k->id || !k->hash_hex) {
        if (err) cwick_error_set(err, CWICK_ERR_ARGUMENT, "bad key row");
        return false;
    }
    pthread_mutex_lock(&s->mtx);
    sqlite3_stmt *st = NULL;
    const char *sql = "INSERT INTO auth_keys (id, name, hash, prefix, created_at,"
                      " last_used_at, enabled) VALUES (?, ?, ?, ?, ?, ?, ?)"
                      " ON CONFLICT (id) DO UPDATE SET name = excluded.name,"
                      " prefix = excluded.prefix, enabled = excluded.enabled, "
                      " last_used_at = excluded.last_used_at;";
    bool ok = sqlite3_prepare_v2(s->db, sql, -1, &st, NULL) == SQLITE_OK && st;
    if (!ok && err)
        cwick_error_set(err, CWICK_ERR_IO, "sqlite prepare: %s", sqlite3_errmsg(s->db));
    if (ok) {
        sqlite3_bind_text(st, 1, k->id, -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(st, 2, k->name ? k->name : "key", -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(st, 3, k->hash_hex, -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(st, 4, k->prefix ? k->prefix : "", -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(st, 5, k->created_at);
        sqlite3_bind_int64(st, 6, k->last_used_at);
        sqlite3_bind_int(st, 7, k->enabled ? 1 : 0);
        ok = step_done(s, st, err);
    }
    if (st) sqlite3_finalize(st);
    pthread_mutex_unlock(&s->mtx);
    return ok;
}

bool cwick_store_key_del(cwick_store *s, const char *id, cwick_error *err) {
    if (!s || !id || !*id) {
        if (err) cwick_error_set(err, CWICK_ERR_ARGUMENT, "bad key id");
        return false;
    }
    pthread_mutex_lock(&s->mtx);
    sqlite3_stmt *st = NULL;
    bool ok = sqlite3_prepare_v2(s->db, "DELETE FROM auth_keys WHERE id = ?;",
                                 -1, &st, NULL) == SQLITE_OK && st;
    if (ok) {
        sqlite3_bind_text(st, 1, id, -1, SQLITE_TRANSIENT);
        ok = step_done(s, st, err);
    }
    if (st) sqlite3_finalize(st);
    pthread_mutex_unlock(&s->mtx);
    return ok;
}

void cwick_store_key_used(cwick_store *s, const char *id, int64_t when) {
    if (!s || !id || !*id) return;
    pthread_mutex_lock(&s->mtx);
    sqlite3_stmt *st = NULL;
    if (sqlite3_prepare_v2(s->db,
            "UPDATE auth_keys SET last_used_at = ? WHERE id = ?;",
            -1, &st, NULL) == SQLITE_OK && st) {
        sqlite3_bind_int64(st, 1, when > 0 ? when : now_s());
        sqlite3_bind_text(st, 2, id, -1, SQLITE_TRANSIENT);
        (void)sqlite3_step(st);
    }
    if (st) sqlite3_finalize(st);
    pthread_mutex_unlock(&s->mtx);
}

void cwick_store_keys_each(cwick_store *s,
                          void (*cb)(const cwick_store_key *k, void *user),
                          void *user) {
    if (!s || !cb) return;
    pthread_mutex_lock(&s->mtx);
    sqlite3_stmt *st = NULL;
    if (sqlite3_prepare_v2(s->db,
            "SELECT id, name, hash, prefix, created_at, last_used_at, enabled"
            " FROM auth_keys ORDER BY created_at;", -1, &st, NULL) != SQLITE_OK || !st) {
        pthread_mutex_unlock(&s->mtx);
        return;
    }
    while (sqlite3_step(st) == SQLITE_ROW) {
        cwick_store_key k;
        k.id = (const char *)sqlite3_column_text(st, 0);
        k.name = (const char *)sqlite3_column_text(st, 1);
        k.hash_hex = (const char *)sqlite3_column_text(st, 2);
        k.prefix = (const char *)sqlite3_column_text(st, 3);
        k.created_at = sqlite3_column_int64(st, 4);
        k.last_used_at = sqlite3_column_int64(st, 5);
        k.enabled = sqlite3_column_int(st, 6) != 0;
        if (!k.id) k.id = "";
        if (!k.name) k.name = "";
        if (!k.hash_hex) k.hash_hex = "";
        if (!k.prefix) k.prefix = "";
        cb(&k, user);
    }
    sqlite3_finalize(st);
    pthread_mutex_unlock(&s->mtx);
}

/* ------------------------------------------------------------------ */
/* stats                                                               */
/* ------------------------------------------------------------------ */

static int64_t count_of(sqlite3 *db, const char *sql) {
    sqlite3_stmt *st = NULL;
    if (sqlite3_prepare_v2(db, sql, -1, &st, NULL) != SQLITE_OK || !st) return 0;
    int64_t n = 0;
    if (sqlite3_step(st) == SQLITE_ROW) n = sqlite3_column_int64(st, 0);
    sqlite3_finalize(st);
    return n;
}

bool cwick_store_get_stats(const cwick_store *s, cwick_store_stats *out) {
    if (!s || !out) return false;
    memset(out, 0, sizeof(*out));
    cwick_store *m = (cwick_store *)s;   /* stats only reads; the mutex is shared */
    pthread_mutex_lock(&m->mtx);
    out->kv_rows = count_of(s->db, "SELECT count(*) FROM app_kv;");
    out->games = count_of(s->db, "SELECT count(*) FROM app_games;");
    out->events = count_of(s->db, "SELECT count(*) FROM app_events;");
    pthread_mutex_unlock(&m->mtx);
    struct stat sb;
    if (s->path && stat(s->path, &sb) == 0) out->bytes = (int64_t)sb.st_size;
    return true;
}
