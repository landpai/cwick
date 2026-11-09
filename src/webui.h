/* webui.h — the chat UI that `cwick serve --web` serves from the binary.
 *
 * The page is a single-file app (HTML + CSS + JS) that talks to this same
 * process over /v1/chat/completions and /v1/metrics. It is compiled in, not
 * read from disk, because the point of the flag is "start one binary and open
 * a browser": no asset directory to locate, no path to get wrong, nothing to
 * version-skew against the server that is already running.
 *
 * The bytes live in src/webui_embed.c, generated from web/ by
 * tools/embed_web.c (see the Makefile). This header is the only interface
 * server.c needs.
 */
#ifndef CWICK_WEBUI_H
#define CWICK_WEBUI_H

#include <stddef.h>

typedef struct cwick_webui_blob {
    const char *path;           /* canonical request path, e.g. "/web/app.js" */
    const char *content_type;   /* full Content-Type header value */
    const unsigned char *data;  /* NUL-terminated, but `len` is authoritative */
    size_t len;
} cwick_webui_blob;

/* Resolve a request path to a built-in asset. Query strings and fragments are
 * ignored; "/" and the usual index aliases map to the HTML document.
 * Returns NULL for anything that is not part of the UI, so the caller can
 * fall through to its normal 404. */
const cwick_webui_blob *cwick_webui_lookup(const char *path);

/* How many assets were compiled in (4 today: document, CSS, JS, icon). */
size_t cwick_webui_asset_count(void);

#endif /* CWICK_WEBUI_H */
