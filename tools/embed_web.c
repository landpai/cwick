/* embed_web.c — compile web/ and apps/ into src/webui_embed.c.
 *
 *     make                      (builds tools/embed_web, then runs it)
 *     ./tools/embed_web         (run it directly)
 *
 * This used to be a Python script. The server is a single native binary and
 * the build should not need a second language just to turn files into byte
 * arrays, so the generator is C too. It walks web/ and apps/, writes one
 * string-literal array per source file, and emits the exact lookup table
 * src/webui.c uses. No filesystem is touched at request time: every route is
 * a row in that table.
 *
 * The output is deliberately ASCII: non-printable/UTF-8 bytes become
 * three-digit octal escapes, so the generated C is portable and the diff is
 * readable.
 */
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAX_PATH_LEN 4096
#define CHUNK 56

typedef struct {
    char *path;          /* served path, e.g. /apps/tictactoe/app.js */
    const char *ctype;   /* static content type */
} route_t;

typedef struct {
    char *ident;         /* C identifier of the byte array */
    char *rel;           /* source path relative to the repo root */
    unsigned char *data;
    size_t len;
    route_t *routes;
    size_t nroutes, cap_routes;
} blob_t;

typedef struct {
    blob_t *v;
    size_t n, cap;
} blob_list_t;

typedef struct {
    char **v;
    size_t n, cap;
} str_list_t;

static void die(const char *msg) {
    fprintf(stderr, "embed_web: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) die("out of memory");
    return q;
}

static char *xstrdup(const char *s) {
    char *p = strdup(s ? s : "");
    if (!p) die("out of memory");
    return p;
}

static void str_push(str_list_t *l, const char *s) {
    if (l->n == l->cap) {
        l->cap = l->cap ? l->cap * 2 : 16;
        l->v = xrealloc(l->v, l->cap * sizeof(*l->v));
    }
    l->v[l->n++] = xstrdup(s);
}

static void str_free(str_list_t *l) {
    for (size_t i = 0; i < l->n; i++) free(l->v[i]);
    free(l->v);
    memset(l, 0, sizeof(*l));
}

static const char *content_type(const char *path) {
    const char *dot = strrchr(path, '.');
    if (!dot) return "application/octet-stream";
    if (strcmp(dot, ".html") == 0 || strcmp(dot, ".htm") == 0) return "text/html; charset=utf-8";
    if (strcmp(dot, ".css") == 0) return "text/css; charset=utf-8";
    if (strcmp(dot, ".js") == 0 || strcmp(dot, ".mjs") == 0) return "text/javascript; charset=utf-8";
    if (strcmp(dot, ".json") == 0) return "application/json; charset=utf-8";
    if (strcmp(dot, ".svg") == 0) return "image/svg+xml";
    if (strcmp(dot, ".png") == 0) return "image/png";
    if (strcmp(dot, ".webp") == 0) return "image/webp";
    if (strcmp(dot, ".jpg") == 0 || strcmp(dot, ".jpeg") == 0) return "image/jpeg";
    if (strcmp(dot, ".gif") == 0) return "image/gif";
    if (strcmp(dot, ".woff2") == 0) return "font/woff2";
    if (strcmp(dot, ".txt") == 0 || strcmp(dot, ".md") == 0) return "text/plain; charset=utf-8";
    return "application/octet-stream";
}

static bool read_file(const char *path, unsigned char **out, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "embed_web: cannot open %s: %s\n", path, strerror(errno));
        return false;
    }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return false; }
    long n = ftell(f);
    if (n < 0) { fclose(f); return false; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return false; }
    unsigned char *buf = xmalloc((size_t)n + 1);
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    if (got != (size_t)n) {
        fprintf(stderr, "embed_web: short read on %s\n", path);
        free(buf);
        return false;
    }
    /* Match the old generator: serve logical LF, never CRLF. */
    size_t w = 0;
    for (size_t i = 0; i < got; i++) {
        if (buf[i] == '\r' && i + 1 < got && buf[i + 1] == '\n') continue;
        buf[w++] = buf[i];
    }
    buf[w] = 0;
    *out = buf;
    *out_len = w;
    return true;
}

static char *sanitize_ident(const char *served) {
    const char *src = served;
    if (*src == '/') src++;
    size_t n = strlen(src);
    char *out = xmalloc(n + 8);
    size_t w = 0;
    out[w++] = 'a';
    out[w++] = 'p';
    out[w++] = 'p';
    out[w++] = '_';
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)src[i];
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9'))
            out[w++] = (char)c;
        else
            out[w++] = '_';
    }
    out[w] = 0;
    return out;
}

static bool ident_used(const blob_list_t *l, const char *ident) {
    for (size_t i = 0; i < l->n; i++)
        if (l->v[i].ident && strcmp(l->v[i].ident, ident) == 0) return true;
    return false;
}

static void blob_add(blob_list_t *l, const char *rel, const char *served,
                     const char *data, size_t len) {
    if (l->n == l->cap) {
        l->cap = l->cap ? l->cap * 2 : 16;
        l->v = xrealloc(l->v, l->cap * sizeof(*l->v));
    }
    blob_t *b = &l->v[l->n++];
    memset(b, 0, sizeof(*b));
    char *base = sanitize_ident(served);
    if (ident_used(l, base)) {
        char suffix[32];
        for (int k = 2;; k++) {
            snprintf(suffix, sizeof(suffix), "_%d", k);
            char *cand = xmalloc(strlen(base) + strlen(suffix) + 1);
            strcpy(cand, base);
            strcat(cand, suffix);
            if (!ident_used(l, cand)) { free(base); base = cand; break; }
            free(cand);
        }
    }
    b->ident = base;
    b->rel = xstrdup(rel);
    b->data = xmalloc(len + 1);
    memcpy(b->data, data, len);
    b->data[len] = 0;
    b->len = len;
}

static void route_add(blob_t *b, const char *path, const char *ctype) {
    if (b->nroutes == b->cap_routes) {
        b->cap_routes = b->cap_routes ? b->cap_routes * 2 : 4;
        b->routes = xrealloc(b->routes, b->cap_routes * sizeof(*b->routes));
    }
    b->routes[b->nroutes].path = xstrdup(path);
    b->routes[b->nroutes].ctype = ctype;
    b->nroutes++;
}

static void add_file(blob_list_t *l, const char *root, const char *rel,
                     const char *served) {
    char path[MAX_PATH_LEN];
    snprintf(path, sizeof(path), "%s/%s", root, rel);
    unsigned char *data = NULL;
    size_t len = 0;
    if (!read_file(path, &data, &len)) exit(1);
    blob_add(l, rel, served, (const char *)data, len);
    free(data);
    blob_t *b = &l->v[l->n - 1];
    const char *ctype = content_type(served);
    route_add(b, served, ctype);
    if (strcmp(served, "/apps/manifest.json") == 0) {
        route_add(b, "/apps", ctype);
        route_add(b, "/apps/", ctype);
    }
    const char *slash = strrchr(served, '/');
    if (strncmp(served, "/apps/", 6) == 0 && slash &&
        strcmp(slash + 1, "index.html") == 0) {
        char alias[MAX_PATH_LEN];
        snprintf(alias, sizeof(alias), "%.*s", (int)(slash - served + 1), served);
        route_add(b, alias, ctype);          /* /apps/tictactoe/ */
        if (alias[0] && alias[strlen(alias) - 1] == '/') {
            alias[strlen(alias) - 1] = 0;
            route_add(b, alias, ctype);      /* /apps/tictactoe  */
        }
    }
}

/* Runtime data never belongs in the binary. `--store ./apps` is a documented
 * way to keep each app's sqlite next to its source, so the walker would
 * otherwise embed `<app>.db` (and the -wal/-shm sidecars of a running server)
 * as assets: the image grows on every race played, and the copy baked in is
 * stale the moment it is served. Everything under apps/ must be an asset the
 * app fetches, so anything with a database suffix is skipped here. */
static bool is_runtime_data(const char *name) {
    static const char *suf[] = {
        ".db", ".db3", ".sqlite", ".sqlite3", ".db-wal", ".db-shm", ".db-journal"
    };
    size_t n = strlen(name);
    for (size_t i = 0; i < sizeof(suf) / sizeof(suf[0]); i++) {
        size_t sl = strlen(suf[i]);
        if (n > sl && strcasecmp(name + n - sl, suf[i]) == 0) return true;
    }
    return false;
}

static void scan_dir(const char *root, const char *rel, str_list_t *files) {
    char dir_path[MAX_PATH_LEN];
    snprintf(dir_path, sizeof(dir_path), "%s/%s", root, rel);
    DIR *d = opendir(dir_path);
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') continue;
        char child_rel[MAX_PATH_LEN];
        snprintf(child_rel, sizeof(child_rel), "%s/%s", rel, e->d_name);
        char child_abs[MAX_PATH_LEN];
        snprintf(child_abs, sizeof(child_abs), "%s/%s", root, child_rel);
        struct stat st;
        if (lstat(child_abs, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            scan_dir(root, child_rel, files);
        } else if (S_ISREG(st.st_mode)) {
            if (is_runtime_data(e->d_name)) continue;
            str_push(files, child_rel);
        }
    }
    closedir(d);
}

static void emit_byte(FILE *f, unsigned char c) {
    switch (c) {
    case '"': fputs("\\\"", f); break;
    case '\\': fputs("\\\\", f); break;
    case '\n': fputs("\\n", f); break;
    case '\t': fputs("\\t", f); break;
    default:
        if (c >= 0x20 && c < 0x7f) fputc(c, f);
        else fprintf(f, "\\%03o", c);
        break;
    }
}

static void emit_blob(FILE *f, const blob_t *b) {
    fprintf(f, "static const unsigned char %s[] =\n", b->ident);
    for (size_t i = 0; i < b->len; i += CHUNK) {
        size_t n = b->len - i < CHUNK ? b->len - i : CHUNK;
        fputs("    \"", f);
        for (size_t k = 0; k < n; k++) emit_byte(f, b->data[i + k]);
        fputs("\"\n", f);
    }
    fputs("    \"\\000\";\n\n", f);
}

static int cmp_files(const void *a, const void *b) {
    const char *sa = *(const char *const *)a;
    const char *sb = *(const char *const *)b;
    bool ma = strcmp(sa, "apps/manifest.json") == 0;
    bool mb = strcmp(sb, "apps/manifest.json") == 0;
    if (ma != mb) return ma ? -1 : 1;
    return strcmp(sa, sb);
}

int main(int argc, char **argv) {
    char root[MAX_PATH_LEN];
    if (argc > 1) {
        snprintf(root, sizeof(root), "%s", argv[1]);
    } else {
        char exe[PATH_MAX];
        if (!realpath(argv[0], exe)) {
            fprintf(stderr, "embed_web: realpath(%s): %s\n", argv[0], strerror(errno));
            return 1;
        }
        char *slash = strrchr(exe, '/');
        if (!slash) { snprintf(root, sizeof(root), "."); }
        else {
            *slash = 0;
            snprintf(root, sizeof(root), "%s/..", exe);
        }
    }
    char resolved[PATH_MAX];
    if (!realpath(root, resolved)) {
        fprintf(stderr, "embed_web: cannot resolve root %s: %s\n", root, strerror(errno));
        return 1;
    }

    blob_list_t blobs;
    memset(&blobs, 0, sizeof(blobs));

    /* Fixed shell assets keep their historical identifiers. */
    add_file(&blobs, resolved, "web/index.html", "/index.html");
    add_file(&blobs, resolved, "web/app.css", "/web/app.css");
    add_file(&blobs, resolved, "web/app.js", "/web/app.js");
    add_file(&blobs, resolved, "web/login.html", "/login.html");

    str_list_t files;
    memset(&files, 0, sizeof(files));
    scan_dir(resolved, "apps", &files);
    qsort(files.v, files.n, sizeof(files.v[0]), cmp_files);
    for (size_t i = 0; i < files.n; i++) {
        const char *rel = files.v[i];
        if (strncmp(rel, "apps/", 5) != 0 || strstr(rel, "/.") != NULL) continue;
        char served[MAX_PATH_LEN];
        snprintf(served, sizeof(served), "/%s", rel);
        add_file(&blobs, resolved, rel, served);
    }

    char out_path[MAX_PATH_LEN];
    snprintf(out_path, sizeof(out_path), "%s/src/webui_embed.c", resolved);
    FILE *f = fopen(out_path, "w");
    if (!f) {
        fprintf(stderr, "embed_web: cannot write %s: %s\n", out_path, strerror(errno));
        return 1;
    }
    fputs("/* Generated by tools/embed_web (C) from web/ and apps/ — do not edit.\n"
          " * Rebuild with `make`; the generator is tools/embed_web.c. */\n"
          "#include \"webui.h\"\n\n", f);
    for (size_t i = 0; i < blobs.n; i++) emit_blob(f, &blobs.v[i]);

    fputs("const cwick_webui_blob cwick_webui_blobs[] = {\n", f);
    size_t route_count = 0;
    size_t total_bytes = 0;
    for (size_t i = 0; i < blobs.n; i++) {
        blob_t *b = &blobs.v[i];
        total_bytes += b->len;
        for (size_t r = 0; r < b->nroutes; r++) {
            fprintf(f, "    { \"%s\", \"%s\", %s, sizeof(%s) - 1 },\n",
                    b->routes[r].path, b->routes[r].ctype, b->ident, b->ident);
            route_count++;
        }
    }
    fputs("};\n\nconst size_t cwick_webui_blob_count =\n"
          "    sizeof(cwick_webui_blobs) / sizeof(cwick_webui_blobs[0]);\n", f);
    fclose(f);

    printf("embed_web: %zu blobs, %zu route(s), %zu bytes -> %s\n",
           blobs.n, route_count, total_bytes, out_path);
    for (size_t i = 0; i < blobs.n; i++) {
        for (size_t r = 0; r < blobs.v[i].nroutes; r++) free(blobs.v[i].routes[r].path);
        free(blobs.v[i].routes);
        free(blobs.v[i].ident);
        free(blobs.v[i].rel);
        free(blobs.v[i].data);
    }
    free(blobs.v);
    str_free(&files);
    return 0;
}
