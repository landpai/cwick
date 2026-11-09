/* webui.c — path resolution for the embedded chat UI.
 *
 * Deliberately a table lookup and nothing else: no filesystem, no path
 * canonicalisation, no way for a request path to escape into a directory.
 * A path that is not in the table is not served, which is also the reason
 * `--web` adds no attack surface to a server that already refuses to bind a
 * non-loopback address without CWICK_ALLOW_PUBLIC=1.
 */
#include "webui.h"

#include <string.h>

/* Filled in by the generated file. */
extern const cwick_webui_blob cwick_webui_blobs[];
extern const size_t cwick_webui_blob_count;

/* Every spelling a browser or a person might use for "the app". */
static int is_index_path(const char *p) {
    static const char *aliases[] = {
        "/", "/index.html", "/web", "/web/", "/chat", "/chat/", "/app", NULL
    };
    for (const char **a = aliases; *a; a++)
        if (strcmp(p, *a) == 0) return 1;
    return 0;
}

const cwick_webui_blob *cwick_webui_lookup(const char *path) {
    if (!path || !*path) return NULL;

    /* Copy up to the query/fragment and bound the work a hostile URL can ask
     * for. 255 bytes is longer than every path the UI uses. */
    char buf[256];
    size_t n = 0;
    for (const char *p = path; *p && *p != '?' && *p != '#'; p++) {
        if (n + 1 >= sizeof(buf)) return NULL;
        buf[n++] = *p;
    }
    buf[n] = '\0';
    if (n == 0) return NULL;

    const char *key = is_index_path(buf) ? "/index.html" :
                      strcmp(buf, "/login") == 0 ? "/login.html" : buf;
    for (size_t i = 0; i < cwick_webui_blob_count; i++) {
        if (strcmp(cwick_webui_blobs[i].path, key) == 0) return &cwick_webui_blobs[i];
    }
    return NULL;
}

size_t cwick_webui_asset_count(void) { return cwick_webui_blob_count; }
