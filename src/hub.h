/* hub.h — the local model library and its download queue.
 *
 * One process owns one library. `./models` is the only destination: a download
 * never writes above it, and a completed checkpoint lands as one immediate
 * child so the ordinary model-directory scan can find it without a restart.
 *
 * The remote side is a read-only model catalog. Search and file listings are
 * ordinary GETs; a download is a background thread that writes each file to
 * `<name>.incomplete` and only renames it once the byte count matches the
 * listing. That single convention gives resumption for free: an interrupted
 * transfer, a paused job, a killed process and a reboot all leave the same
 * partial file behind, and the next run starts from its size with a Range
 * request. Sizes are curl_off_t / int64_t everywhere, because a checkpoint
 * with a few hundred GiB of shards is the normal case here, not an edge case.
 *
 * Nothing in this file touches the inference engine. The server calls these
 * functions from its own control endpoints, before the generation queue, so a
 * slow download cannot delay a request that wants the resident model.
 */
#ifndef CWICK_HUB_H
#define CWICK_HUB_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct cwick_hub cwick_hub;

/* Longest path the library composes. Public so the server can hand
 * cwick_hub_model_dir() a buffer of the right size. */
#define CWICK_HUB_PATH_MAX 1024

/* `root` is where checkpoints live (the caller passes "./models"); `endpoint`
 * is the catalog host, default "https://hf-mirror.com" when NULL or empty.
 * Returns NULL only if libcurl cannot be initialised. */
cwick_hub *cwick_hub_open(const char *root, const char *endpoint);

/* Ask every worker to stop and join them. Partial files stay on disk by
 * design, so the next process can resume them. */
void cwick_hub_close(cwick_hub *h);

/* All JSON-returning functions hand back a malloc'd string, or NULL with a
 * message in `err` on a hard failure. A remote that answers with an HTTP error
 * is reported as JSON with an "error" field (and a 200 at the HTTP layer), so
 * the UI can show the reason instead of a blank panel. */

/* GET /v1/hub/search?q=<text>&limit=<n> — a small, curated projection of the
 * remote catalog: id, downloads, likes, tags, pipeline tag, gated flag. */
char *cwick_hub_search(cwick_hub *h, const char *query, int limit,
                       char *err, size_t errcap);

/* GET /v1/hub/models — what is already on disk under `root`, with byte counts,
 * file counts and whether any .incomplete file is waiting to be resumed. */
char *cwick_hub_models(cwick_hub *h, char *err, size_t errcap);

/* GET /v1/hub/downloads — the live job table. One entry per download, with
 * progress, rate and ETA; polled by the UI, never pushed. */
char *cwick_hub_downloads(cwick_hub *h, char *err, size_t errcap);

/* POST /v1/hub/download {"repo":"owner/name"} — start (or resume) a job. */
char *cwick_hub_download_start(cwick_hub *h, const char *repo,
                               char *err, size_t errcap);

/* POST /v1/hub/download/<id>/<pause|resume|cancel|retry> */
char *cwick_hub_download_action(cwick_hub *h, const char *id,
                                const char *action, char *err, size_t errcap);

/* A completed directory that the model catalog has not seen yet. Returns a
 * copy of its path, or NULL when `wanted` is not a finished local download.
 * Used by POST /v1/models/switch so a freshly downloaded checkpoint is
 * switchable in the same process that downloaded it. */
char *cwick_hub_local_path(cwick_hub *h, const char *wanted);

/* Resolve `owner/name`, a bare directory name, or a flat `./models/<name>`
 * entry to the directory that holds it. The answer is guaranteed to exist,
 * to sit strictly inside the library root after symlinks are resolved, and
 * never to be the root itself. Read-only; the switch and delete paths share
 * it so they agree on what a "local model" is. */
bool cwick_hub_model_dir(cwick_hub *h, const char *wanted,
                         char *out, size_t outcap);

/* DELETE /v1/hub/models/<name> — remove a downloaded checkpoint (complete or
 * partially downloaded) from the library. Refused with 409 while a download
 * worker is still writing into that directory, because deleting its files
 * under the worker would turn a resumable transfer into a corrupt one.
 * Returns a JSON object on success; NULL with `err` and `*status` on failure
 * (404 unknown, 409 busy, 400 invalid). Removal is recursive, irreversible,
 * does not follow symlinks, and cannot escape the library root. */
char *cwick_hub_delete(cwick_hub *h, const char *wanted, int *status,
                       char *err, size_t errcap);

/* The library root, for the startup banner and /v1/config. */
const char *cwick_hub_root(const cwick_hub *h);
const char *cwick_hub_endpoint(const cwick_hub *h);

#endif /* CWICK_HUB_H */
