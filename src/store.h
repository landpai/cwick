/* store.h — sqlite persistence for the built-in apps.
 *
 * The app market serves HTML that runs in a browser, and a browser tab is a
 * bad place to keep the one thing an app actually owns: its history. The
 * tic-tac-toe app wants to remember finished games and replay how the model
 * decided each move, and `localStorage` disappears with the profile, the
 * private window and "clear site data". So the server keeps it instead, in one
 * sqlite file, behind the same login as everything else.
 *
 * The shape is deliberately small and generic — two tables, no schema per app:
 *
 *   app_kv    (app_id, k, v, updated_at)              small state, scores
 *   app_games (app_id, game_id, started_at, ...)      one row per match
 *   app_events(app_id, game_id, seq, ts, kind, payload)  the replay log
 *
 * The same file also holds `auth_keys` (see the bottom of this header): one
 * sqlite file for everything this process has to remember, rather than a
 * second database whose backup story nobody wrote down.
 *
 * `payload` is stored as the exact JSON text the app sent and handed back verbatim,
 * so the app's own fields never have to be known here. It is parsed on the way
 * in only to prove it is valid JSON — an unvalidated blob would let one app
 * forge the JSON envelope another app reads back.
 *
 * There is no ORM and no migration framework: `PRAGMA user_version` carries a
 * single integer and the schema is created with IF NOT EXISTS, which is all a
 * single-file store opened by one process needs.
 */
#ifndef CWICK_STORE_H
#define CWICK_STORE_H

#include <stdbool.h>
#include <stdint.h>

#include "cwick.h"

typedef struct cwick_store cwick_store;

/* Upper bounds, so one app cannot grow the file without limit. */
#define CWICK_STORE_MAX_APP_ID   64
#define CWICK_STORE_MAX_KEY      128
#define CWICK_STORE_MAX_KIND     48
#define CWICK_STORE_MAX_GAME_ID  96
#define CWICK_STORE_MAX_VALUE    (1 << 20)   /* 1 MiB per KV value   */
#define CWICK_STORE_MAX_PAYLOAD  (1 << 20)   /* 1 MiB per event      */
#define CWICK_STORE_MAX_EVENTS   5000        /* rows returned per read */

/* A row of app_events, handed to the enumeration callback. */
typedef struct {
    int64_t id;
    const char *game_id;
    int64_t seq;
    int64_t ts;          /* unix seconds */
    const char *kind;
    const char *payload; /* raw JSON text, valid for the duration of the call */
} cwick_store_event;

/* A row of app_games. `outcome` is the app's own word for how it ended. */
typedef struct {
    const char *game_id;
    int64_t started_at;
    int64_t ended_at;    /* 0 while the game is still running */
    const char *outcome; /* may be "" */
    int64_t moves;
} cwick_store_game;

typedef struct {
    int64_t games;
    int64_t events;
    int64_t kv_rows;
    int64_t bytes;       /* file size, 0 when it cannot be stat'ed */
} cwick_store_stats;

/* Open (creating if needed) the database at `path`. `path` NULL or "" means
 * the default: $CWICK_STORE, else ~/.cwick/apps.db. Returns NULL and fills `err`
 * when the file cannot be opened or the schema cannot be created. */
cwick_store *cwick_store_open(const char *path, cwick_error *err);
void cwick_store_close(cwick_store *s);

/* The path actually opened (owned by the store). */
const char *cwick_store_path(const cwick_store *s);

/* ---- per-app database layout -------------------------------------------
 *
 * Most deployments want one database file per app: a game's history is then
 * bounded by that game, deleting one app's file cannot touch another, and a
 * phone or a backup copies one file per app. `--store` therefore names a
 * *directory* by default ($CWICK_STORE, else ~/.cwick/apps) holding
 * `<app>.db` for each app plus `auth.db` for the server's own API keys.
 *
 * A path that already exists as a regular file, or one with a database suffix
 * (.db/.sqlite/.sqlite3), is still a single shared database — the layout the
 * first versions used, kept so an existing --store/CWICK_STORE keeps working.
 * cwick_store_path_is_file is how the caller decides which one it has. */
const char *cwick_store_default_dir(void);
bool cwick_store_path_is_file(const char *path);

/* ---- key/value ------------------------------------------------------- */

/* `value` must be valid JSON; it is stored verbatim. Returns false on a
 * malformed value or a sqlite failure. */
bool cwick_store_kv_put(cwick_store *s, const char *app, const char *key,
                       const char *value, cwick_error *err);

/* Returns a malloc'd copy of the stored JSON text, or NULL when absent. */
char *cwick_store_kv_get(cwick_store *s, const char *app, const char *key);

bool cwick_store_kv_del(cwick_store *s, const char *app, const char *key,
                       cwick_error *err);

/* Enumerate one app's keys in `updated_at` order. */
void cwick_store_kv_each(cwick_store *s, const char *app,
                        void (*cb)(const char *key, const char *value,
                                   int64_t updated_at, void *user),
                        void *user);

/* ---- games ------------------------------------------------------------ */

/* Start or touch a game row. Called with `outcome` NULL while it runs, and
 * again with an outcome and a move count when it ends. */
bool cwick_store_game_put(cwick_store *s, const char *app, const char *game_id,
                         int64_t started_at, const char *outcome,
                         int64_t ended_at, int64_t moves, cwick_error *err);

/* Newest first. `limit` <= 0 means CWICK_STORE_MAX_EVENTS worth of games. */
void cwick_store_games(cwick_store *s, const char *app, int64_t limit,
                      void (*cb)(const cwick_store_game *g, void *user),
                      void *user);

/* ---- events ----------------------------------------------------------- */

/* Append one step to a game's replay log. `payload` must be valid JSON.
 * `seq` is the app's own step numbering (0-based); it is stored but never
 * trusted for ordering — `id` is. */
bool cwick_store_event_add(cwick_store *s, const char *app, const char *game_id,
                          int64_t seq, const char *kind, const char *payload,
                          cwick_error *err);

/* Read a game's steps in order. `game_id` NULL means every game of the app. */
void cwick_store_events(cwick_store *s, const char *app, const char *game_id,
                       int64_t limit,
                       void (*cb)(const cwick_store_event *e, void *user),
                       void *user);

/* Drop one game and its steps (the app's "clear history"). */
bool cwick_store_game_del(cwick_store *s, const char *app, const char *game_id,
                         cwick_error *err);

/* ---- housekeeping ------------------------------------------------------ */

bool cwick_store_get_stats(const cwick_store *s, cwick_store_stats *out);

/* `app` is an identifier the server has already taken from a URL path: it is
 * length-checked here and bound as a parameter everywhere, but a caller that
 * builds SQL by hand is one refactor away from an injection, so the check is
 * cheap insurance and it keeps '.' and '/' out of the app namespace. */
bool cwick_store_id_ok(const char *app);

/* ---- API keys -------------------------------------------------------
 *
 * Only hashes live here. An API key is 160 random bits, so the stored value is
 * a SHA-256 and the plaintext is never written down by anything; the `prefix`
 * column is the display form ("cwick_" plus 8 characters) so an operator can
 * tell which key served a request without the row ever carrying the key.
 *
 * Keys outlive sessions, which is the point: an agent platform is configured
 * once, and a restart of cwick must not invalidate it.
 */
typedef struct {
    const char *id;
    const char *name;
    const char *hash_hex;
    const char *prefix;
    int64_t created_at;
    int64_t last_used_at;
    bool enabled;
} cwick_store_key;

bool cwick_store_key_put(cwick_store *s, const cwick_store_key *k, cwick_error *err);
bool cwick_store_key_del(cwick_store *s, const char *id, cwick_error *err);
/* `when` <= 0 means "now". Best-effort: a stale timestamp is not worth
 * failing a request over, so this never reports an error. */
void cwick_store_key_used(cwick_store *s, const char *id, int64_t when);
void cwick_store_keys_each(cwick_store *s,
                          void (*cb)(const cwick_store_key *k, void *user),
                          void *user);

#endif /* CWICK_STORE_H */
