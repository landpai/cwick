/* hub.c — a local model library with a resumable background downloader.
 *
 * The rules that matter here, in the order they tend to be broken:
 *
 *   1. Every partial file is `<target>.incomplete`. Nothing else marks an
 *      unfinished download, so a crash, a pause and a Ctrl+C are the same
 *      state. The next run reads that file's size and asks for the rest.
 *   2. A file is only renamed into place when its size matches the listing.
 *      A model directory therefore never looks complete while it is not.
 *   3. All sizes are 64-bit. Shards of tens of GiB and repositories of a few
 *      hundred GiB are the expected input; `size_t` and `long` would both be
 *      wrong somewhere on this machine's usual checkpoints.
 *   4. The download thread never holds the state lock across a network call.
 *      A UI poll must not wait on a slow CDN.
 */
#include "hub.h"

#include "cwick.h"
#include "json.h"
#include "log.h"
#include "util.h"

#include <curl/curl.h>
#include <dirent.h>
#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define HUB_MAX_JOBS   8
#define HUB_MAX_FILES  2048
#define HUB_PATH_MAX   1024
#define HUB_DEFAULT_ENDPOINT "https://hf-mirror.com"

typedef struct { char *p; size_t len, cap; } hbuf;

static bool hb_reserve(hbuf *b, size_t extra) {
    if (b->len + extra + 1 <= b->cap) return true;
    size_t want = b->cap ? b->cap : 256;
    while (want < b->len + extra + 1) want *= 2;
    char *p = (char *)realloc(b->p, want);
    if (!p) return false;
    b->p = p;
    b->cap = want;
    return true;
}
static void hb_putn(hbuf *b, const char *s, size_t n) {
    if (!hb_reserve(b, n)) return;
    memcpy(b->p + b->len, s, n);
    b->len += n;
    b->p[b->len] = '\0';
}
static void hb_puts(hbuf *b, const char *s) { hb_putn(b, s, strlen(s)); }
static void hb_putf(hbuf *b, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));
static void hb_putf(hbuf *b, const char *fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n > 0 && hb_reserve(b, (size_t)n)) {
        vsnprintf(b->p + b->len, (size_t)n + 1, fmt, ap2);
        b->len += (size_t)n;
    }
    va_end(ap2);
}
static char *hb_take(hbuf *b) {
    if (!b->p) {
        b->p = (char *)malloc(1);
        if (b->p) b->p[0] = '\0';
    }
    return b->p;
}

/* Quote a C string for JSON. Escaping control bytes matters: a model id or a
 * file name comes from a remote listing and lands inside an HTML page later. */
static void hb_json_str(hbuf *b, const char *s) {
    hb_puts(b, "\"");
    for (const unsigned char *p = (const unsigned char *)(s ? s : ""); *p; p++) {
        switch (*p) {
        case '"':  hb_puts(b, "\\\""); break;
        case '\\': hb_puts(b, "\\\\"); break;
        case '\n': hb_puts(b, "\\n");  break;
        case '\r': hb_puts(b, "\\r");  break;
        case '\t': hb_puts(b, "\\t");  break;
        default:
            if (*p < 0x20) hb_putf(b, "\\u%04x", (unsigned)*p);
            else hb_putn(b, (const char *)p, 1);
        }
    }
    hb_puts(b, "\"");
}

static void url_encode(hbuf *b, const char *s, bool keep_slash) {
    static const char hex[] = "0123456789ABCDEF";
    for (const unsigned char *p = (const unsigned char *)(s ? s : ""); *p; p++) {
        unsigned char c = *p;
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_' ||
            c == '.' || c == '~' || (keep_slash && c == '/')) {
            hb_putn(b, (const char *)p, 1);
        } else {
            char e[3] = { '%', hex[c >> 4], hex[c & 15] };
            hb_putn(b, e, 3);
        }
    }
}

/* A repository id is a path component source. Accept only the characters a
 * catalog id can have, and never a dot-dot segment: the downloader writes
 * under ./models and uses these names verbatim. */
static bool repo_safe(const char *repo) {
    if (!repo || !*repo || strlen(repo) > 200) return false;
    if (strstr(repo, "..") || strchr(repo, '\\') || repo[0] == '/') return false;
    for (const char *p = repo; *p; p++) {
        unsigned char c = (unsigned char)*p;
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
              (c >= '0' && c <= '9') || c == '/' || c == '-' ||
              c == '_' || c == '.')) return false;
    }
    return strchr(repo, '/') != NULL;
}

static bool file_path_safe(const char *p) {
    if (!p || !*p || p[0] == '/' || strstr(p, "..")) return false;
    return true;
}

typedef struct {
    char path[HUB_PATH_MAX];
    int64_t size;
} hub_file;

typedef enum {
    JOB_QUEUED = 0, JOB_LISTING, JOB_RUNNING, JOB_PAUSED,
    JOB_DONE, JOB_ERROR, JOB_CANCELED
} job_state;

static const char *job_state_name(job_state s) {
    switch (s) {
    case JOB_QUEUED:   return "queued";
    case JOB_LISTING:  return "listing";
    case JOB_RUNNING:  return "running";
    case JOB_PAUSED:   return "paused";
    case JOB_DONE:     return "done";
    case JOB_ERROR:    return "error";
    case JOB_CANCELED: return "canceled";
    }
    return "unknown";
}

typedef struct {
    cwick_hub *owner;
    char id[16];
    char repo[256];
    char dir[HUB_PATH_MAX];
    char current[256];
    char error[256];
    job_state state;
    hub_file *files;
    int nfiles, filecap;
    int64_t total_bytes, done_bytes, partial_bytes;
    int files_done;
    double speed_bps;
    int64_t eta_s;
    time_t started, updated;
    bool cancel_req, pause_req;
    bool thread_running;
    pthread_t thread;
} hub_job;

struct cwick_hub {
    pthread_mutex_t mtx;
    char root[HUB_PATH_MAX];
    char endpoint[256];
    hub_job jobs[HUB_MAX_JOBS];
    int njobs;
    uint32_t next_id;
    bool stopping;
    bool curl_ready;
};

/* ------------------------------------------------------------------ */
/* one HTTP GET                                                        */
/* ------------------------------------------------------------------ */
static size_t http_sink(char *ptr, size_t size, size_t nmemb, void *userdata) {
    size_t n = size * nmemb;
    hb_putn((hbuf *)userdata, ptr, n);
    return n;
}

static long hub_http_get(cwick_hub *h, const char *url, hbuf *out,
                         char *err, size_t errcap) {
    CURL *c = curl_easy_init();
    if (!c) {
        snprintf(err, errcap, "cannot create an HTTP handle");
        return 0;
    }
    curl_easy_setopt(c, CURLOPT_URL, url);
    curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(c, CURLOPT_MAXREDIRS, 8L);
    curl_easy_setopt(c, CURLOPT_USERAGENT, "cwick/1.0 (+local model library)");
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, http_sink);
    curl_easy_setopt(c, CURLOPT_WRITEDATA, out);
    curl_easy_setopt(c, CURLOPT_ACCEPT_ENCODING, "");
    curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);       /* required in a thread */
    curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, 12L);
    curl_easy_setopt(c, CURLOPT_TIMEOUT, 120L);
    curl_easy_setopt(c, CURLOPT_SSL_VERIFYPEER, 1L);
    char errbuf[CURL_ERROR_SIZE];
    errbuf[0] = '\0';
    curl_easy_setopt(c, CURLOPT_ERRORBUFFER, errbuf);

    CURLcode rc = curl_easy_perform(c);
    long code = 0;
    curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
    if (rc != CURLE_OK) {
        snprintf(err, errcap, "%s", errbuf[0] ? errbuf : curl_easy_strerror(rc));
        if (!code) code = -1;
    }
    curl_easy_cleanup(c);
    (void)h;
    return code;
}

/* ------------------------------------------------------------------ */
/* search                                                             */
/* ------------------------------------------------------------------ */
static void tag_list(hbuf *b, json_value *tags) {
    hb_puts(b, "[");
    size_t n = cwick_json_array_len(tags);
    if (n > 6) n = 6;
    for (size_t i = 0; i < n; i++) {
        const char *t = cwick_json_str(cwick_json_array_at(tags, i));
        if (!t || !*t) continue;
        if (i) hb_puts(b, ",");
        hb_json_str(b, t);
    }
    hb_puts(b, "]");
}

char *cwick_hub_search(cwick_hub *h, const char *query, int limit,
                       char *err, size_t errcap) {
    if (!h) { snprintf(err, errcap, "model library is not initialised"); return NULL; }
    if (limit < 1) limit = 12;
    if (limit > 50) limit = 50;

    hbuf url = {0}, enc = {0}, out = {0};
    hb_puts(&url, h->endpoint);
    hb_puts(&url, "/api/models?search=");
    url_encode(&enc, query ? query : "", false);
    hb_puts(&url, enc.p ? enc.p : "");
    hb_putf(&url, "&limit=%d&full=false", limit);
    free(enc.p);

    hbuf body = {0};
    char why[256] = "";
    long code = hub_http_get(h, url.p, &body, why, sizeof(why));
    free(url.p);

    hb_puts(&out, "{\"endpoint\":");
    hb_json_str(&out, h->endpoint);
    hb_puts(&out, ",\"query\":");
    hb_json_str(&out, query ? query : "");
    hb_puts(&out, ",\"models\":[");

    int count = 0;
    if (code == 200 && body.p) {
        cwick_error je;
        cwick_error_clear(&je);
        json_value *arr = cwick_json_parse(body.p, body.len, &je);
        size_t n = cwick_json_array_len(arr);
        for (size_t i = 0; i < n && count < limit; i++) {
            json_value *m = cwick_json_array_at(arr, i);
            if (!m || m->kind != JSON_OBJECT) continue;
            const char *id = cwick_json_str(cwick_json_get(m, "id"));
            if (!id || !*id) continue;
            if (count++) hb_puts(&out, ",");
            hb_puts(&out, "{\"id\":");       hb_json_str(&out, id);
            hb_puts(&out, ",\"downloads\":");
            hb_putf(&out, "%lld", (long long)cwick_json_int(cwick_json_get(m, "downloads"), 0));
            hb_puts(&out, ",\"likes\":");
            hb_putf(&out, "%lld", (long long)cwick_json_int(cwick_json_get(m, "likes"), 0));
            const char *pt = cwick_json_str(cwick_json_get(m, "pipeline_tag"));
            hb_puts(&out, ",\"pipeline_tag\":");
            hb_json_str(&out, pt ? pt : "");
            const char *lm = cwick_json_str(cwick_json_get(m, "lastModified"));
            if (!lm) lm = cwick_json_str(cwick_json_get(m, "updatedAt"));
            hb_puts(&out, ",\"last_modified\":");
            hb_json_str(&out, lm ? lm : "");
            hb_putf(&out, ",\"private\":%s,\"gated\":%s,\"tags\":",
                    cwick_json_bool(cwick_json_get(m, "private"), false) ? "true" : "false",
                    cwick_json_bool(cwick_json_get(m, "gated"), false) ? "true" : "false");
            tag_list(&out, cwick_json_get(m, "tags"));
            hb_puts(&out, "}");
        }
        cwick_json_free(arr);
    } else if (code != 200 && !why[0]) {
        snprintf(why, sizeof(why), "catalog returned HTTP %ld", code);
    } else if (code < 0 && !why[0]) {
        snprintf(why, sizeof(why), "catalog request failed");
    }
    free(body.p);
    hb_putf(&out, "],\"count\":%d,\"error\":", count);
    hb_json_str(&out, why[0] ? why : "");
    hb_puts(&out, "}");
    return hb_take(&out);
}

/* ------------------------------------------------------------------ */
/* local library                                                      */
/* ------------------------------------------------------------------ */
static void dir_stats(const char *dir, int64_t *bytes, int *files,
                      int64_t *partial_bytes, int *partials, int depth) {
    DIR *d = opendir(dir);
    if (!d || depth > 4) { if (d) closedir(d); return; }
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        /* Our own bookkeeping file is not part of the checkpoint; a repository
         * file such as .gitattributes is, and counting it keeps the displayed
         * file count equal to the remote listing. */
        if (strcmp(de->d_name, ".cwick-repo") == 0) continue;
        char p[HUB_PATH_MAX];
        if (snprintf(p, sizeof(p), "%s/%s", dir, de->d_name) >= (int)sizeof(p))
            continue;
        struct stat st;
        if (stat(p, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            dir_stats(p, bytes, files, partial_bytes, partials, depth + 1);
        } else if (S_ISREG(st.st_mode)) {
            size_t n = strlen(de->d_name);
            if (n >= 11 && strcmp(de->d_name + n - 11, ".incomplete") == 0) {
                (*partials)++;
                *partial_bytes += (int64_t)st.st_size;
            } else {
                (*files)++;
                *bytes += (int64_t)st.st_size;
            }
        }
    }
    closedir(d);
}

static void append_local_one(hbuf *out, const char *dir, const char *name,
                            const char *repo, bool *first) {
    int64_t bytes = 0, partial_bytes = 0;
    int files = 0, partials = 0;
    dir_stats(dir, &bytes, &files, &partial_bytes, &partials, 0);

    char cfg[HUB_PATH_MAX];
    snprintf(cfg, sizeof(cfg), "%s/config.json", dir);
    bool has_config = cwick_file_exists(cfg);
    bool complete = has_config && partials == 0 && files > 0;

    if (!*first) hb_puts(out, ",");
    *first = false;
    hb_puts(out, "{\"id\":");            hb_json_str(out, name);
    hb_puts(out, ",\"repo\":");          hb_json_str(out, repo);
    hb_puts(out, ",\"path\":");          hb_json_str(out, dir);
    hb_putf(out, ",\"bytes\":%lld,\"files\":%d,\"partial_bytes\":%lld,"
                 "\"partial_files\":%d,\"complete\":%s,\"resumable\":%s}",
            (long long)bytes, files, (long long)partial_bytes, partials,
            complete ? "true" : "false", partials > 0 ? "true" : "false");
}

/* A directory that is itself a checkpoint has config.json; one that is still
 * downloading only has the .cwick-repo marker. Anything else at the first
 * level is an owner container, and its children are the checkpoints. */
static bool looks_like_checkpoint(const char *dir) {
    char p[HUB_PATH_MAX];
    snprintf(p, sizeof(p), "%s/config.json", dir);
    if (cwick_file_exists(p)) return true;
    snprintf(p, sizeof(p), "%s/.cwick-repo", dir);
    return cwick_file_exists(p) ? true : false;
}

static void emit_dir(hbuf *out, const char *dir, const char *name,
                     const char *fallback_repo, bool *first) {
    char marker[HUB_PATH_MAX], *text = NULL;
    size_t n = 0;
    char repo[256];
    snprintf(repo, sizeof(repo), "%s", fallback_repo);
    snprintf(marker, sizeof(marker), "%s/.cwick-repo", dir);
    if (cwick_read_file(marker, &text, &n) == 0 && text && n) {
        size_t k = 0;
        while (k < n && k + 1 < sizeof(repo) &&
               text[k] != '\n' && text[k] != '\r') { repo[k] = text[k]; k++; }
        repo[k] = '\0';
    }
    free(text);
    append_local_one(out, dir, name, repo, first);
}

char *cwick_hub_models(cwick_hub *h, char *err, size_t errcap) {
    if (!h) { snprintf(err, errcap, "model library is not initialised"); return NULL; }
    hbuf out = {0};
    hb_puts(&out, "{\"root\":");
    hb_json_str(&out, h->root);
    hb_puts(&out, ",\"endpoint\":");
    hb_json_str(&out, h->endpoint);
    hb_puts(&out, ",\"models\":[");
    bool first = true;
    DIR *d = opendir(h->root);
    if (d) {
        struct dirent *de;
        while ((de = readdir(d)) != NULL) {
            if (de->d_name[0] == '.') continue;
            char dir[HUB_PATH_MAX];
            if (snprintf(dir, sizeof(dir), "%s/%s", h->root, de->d_name) >=
                (int)sizeof(dir)) continue;
            if (!cwick_dir_exists(dir)) continue;
            if (looks_like_checkpoint(dir)) {
                emit_dir(&out, dir, de->d_name, de->d_name, &first);
                continue;
            }
            DIR *d2 = opendir(dir);
            if (!d2) continue;
            struct dirent *de2;
            while ((de2 = readdir(d2)) != NULL) {
                if (de2->d_name[0] == '.') continue;
                char child[HUB_PATH_MAX], repo2[256];
                if (snprintf(child, sizeof(child), "%s/%s", dir, de2->d_name) >=
                    (int)sizeof(child)) continue;
                if (!cwick_dir_exists(child)) continue;
                if (!looks_like_checkpoint(child)) continue;
                snprintf(repo2, sizeof(repo2), "%s/%s", de->d_name, de2->d_name);
                emit_dir(&out, child, de2->d_name, repo2, &first);
            }
            closedir(d2);
        }
        closedir(d);
    }
    hb_puts(&out, "]}");
    return hb_take(&out);
}

static char *checked_path(const char *dir) {
    char cfg[HUB_PATH_MAX];
    snprintf(cfg, sizeof(cfg), "%s/config.json", dir);
    if (!cwick_file_exists(cfg)) return NULL;
    int64_t bytes = 0, partial = 0;
    int files = 0, partials = 0;
    dir_stats(dir, &bytes, &files, &partial, &partials, 0);
    if (partials > 0 || files == 0) return NULL;     /* still downloading */
    return cwick_strdup(dir);
}

char *cwick_hub_local_path(cwick_hub *h, const char *wanted) {
    if (!h || !wanted || !*wanted) return NULL;
    if (strstr(wanted, "..") || wanted[0] == '/') return NULL;
    const char *base = strrchr(wanted, '/');
    base = base ? base + 1 : wanted;

    /* Two layouts are recognised: a repository kept under its owner
     * (./models/owner/name, what a download produces) and a flat directory
     * holding just the name. */
    char dir[HUB_PATH_MAX];
    if (snprintf(dir, sizeof(dir), "%s/%s", h->root, wanted) < (int)sizeof(dir)) {
        char *p = checked_path(dir);
        if (p) return p;
    }
    if (base != wanted &&
        snprintf(dir, sizeof(dir), "%s/%s", h->root, base) < (int)sizeof(dir)) {
        char *p = checked_path(dir);
        if (p) return p;
    }
    /* A bare name is ambiguous under an owner/name layout. Look one level
     * down rather than making the caller guess the owner. */
    if (!strchr(wanted, '/')) {
        DIR *d = opendir(h->root);
        if (d) {
            struct dirent *de;
            while ((de = readdir(d)) != NULL) {
                if (de->d_name[0] == '.') continue;
                if (snprintf(dir, sizeof(dir), "%s/%s/%s", h->root,
                             de->d_name, wanted) >= (int)sizeof(dir)) continue;
                char *p = checked_path(dir);
                if (p) { closedir(d); return p; }
            }
            closedir(d);
        }
    }
    return NULL;
}

const char *cwick_hub_root(const cwick_hub *h) { return h ? h->root : ""; }
const char *cwick_hub_endpoint(const cwick_hub *h) { return h ? h->endpoint : ""; }

/* ------------------------------------------------------------------ */
/* job table                                                          */
/* ------------------------------------------------------------------ */
static void job_json(hbuf *out, const hub_job *j, bool comma) {
    if (comma) hb_puts(out, ",");
    double progress = j->total_bytes > 0
        ? (double)j->done_bytes / (double)j->total_bytes : 0.0;
    if (progress < 0) progress = 0;
    if (progress > 1) progress = 1;
    hb_puts(out, "{\"id\":");        hb_json_str(out, j->id);
    hb_puts(out, ",\"repo\":");      hb_json_str(out, j->repo);
    hb_puts(out, ",\"dir\":");       hb_json_str(out, j->dir);
    hb_puts(out, ",\"state\":");     hb_json_str(out, job_state_name(j->state));
    hb_puts(out, ",\"current\":");   hb_json_str(out, j->current);
    hb_puts(out, ",\"error\":");     hb_json_str(out, j->error);
    hb_putf(out, ",\"progress\":%.6f,\"bytes_done\":%lld,\"bytes_total\":%lld,"
                 "\"files_done\":%d,\"files_total\":%d,"
                 "\"partial_bytes\":%lld,\"speed_bps\":%.0f,\"eta_seconds\":%lld,"
                 "\"started_at\":%lld,\"updated_at\":%lld,"
                 "\"range_resume\":%s}",
            progress, (long long)j->done_bytes, (long long)j->total_bytes,
            j->files_done, j->nfiles, (long long)j->partial_bytes,
            j->speed_bps, (long long)j->eta_s,
            (long long)j->started, (long long)j->updated,
            j->partial_bytes > 0 ? "true" : "false");
}

char *cwick_hub_downloads(cwick_hub *h, char *err, size_t errcap) {
    if (!h) { snprintf(err, errcap, "model library is not initialised"); return NULL; }
    hbuf out = {0};
    hb_puts(&out, "{\"root\":");
    hb_json_str(&out, h->root);
    hb_puts(&out, ",\"endpoint\":");
    hb_json_str(&out, h->endpoint);
    hb_puts(&out, ",\"jobs\":[");
    pthread_mutex_lock(&h->mtx);
    for (int i = 0; i < h->njobs; i++) job_json(&out, &h->jobs[i], i > 0);
    pthread_mutex_unlock(&h->mtx);
    hb_puts(&out, "]}");
    return hb_take(&out);
}

static char *job_only_json(cwick_hub *h, hub_job *j) {
    hbuf out = {0};
    pthread_mutex_lock(&h->mtx);
    job_json(&out, j, false);
    pthread_mutex_unlock(&h->mtx);
    return hb_take(&out);
}

static hub_job *find_job(cwick_hub *h, const char *id) {
    for (int i = 0; i < h->njobs; i++)
        if (strcmp(h->jobs[i].id, id) == 0) return &h->jobs[i];
    return NULL;
}

/* ------------------------------------------------------------------ */
/* file listing                                                       */
/* ------------------------------------------------------------------ */
static bool tree_add(hub_job *j, const char *path, int64_t size) {
    if (!file_path_safe(path) || j->nfiles >= HUB_MAX_FILES) return false;
    if (j->nfiles == j->filecap) {
        int cap = j->filecap ? j->filecap * 2 : 32;
        if (cap > HUB_MAX_FILES) cap = HUB_MAX_FILES;
        hub_file *nf = (hub_file *)realloc(j->files, (size_t)cap * sizeof(*nf));
        if (!nf) return false;
        j->files = nf;
        j->filecap = cap;
    }
    hub_file *f = &j->files[j->nfiles++];
    snprintf(f->path, sizeof(f->path), "%s", path);
    f->size = size;
    return true;
}

static bool fetch_tree(cwick_hub *h, hub_job *j, char *why, size_t whycap) {
    hbuf url = {0};
    hb_puts(&url, h->endpoint);
    hb_puts(&url, "/api/models/");
    url_encode(&url, j->repo, true);
    hb_puts(&url, "/tree/main?recursive=true&expand=false");
    hbuf body = {0};
    char net[256] = "";
    long code = hub_http_get(h, url.p, &body, net, sizeof(net));
    free(url.p);
    if (code != 200 || !body.p) {
        snprintf(why, whycap, "%s", net[0] ? net : "file listing failed");
        free(body.p);
        return false;
    }
    cwick_error je;
    cwick_error_clear(&je);
    json_value *arr = cwick_json_parse(body.p, body.len, &je);
    free(body.p);
    size_t n = cwick_json_array_len(arr);
    if (n == 0) {
        snprintf(why, whycap, "the repository has no files (or the listing was empty)");
        cwick_json_free(arr);
        return false;
    }
    for (size_t i = 0; i < n; i++) {
        json_value *f = cwick_json_array_at(arr, i);
        if (!f || f->kind != JSON_OBJECT) continue;
        const char *type = cwick_json_str(cwick_json_get(f, "type"));
        if (type && strcmp(type, "file") != 0) continue;
        const char *path = cwick_json_str(cwick_json_get(f, "path"));
        int64_t size = cwick_json_int(cwick_json_get(f, "size"), -1);
        json_value *lfs = cwick_json_get(f, "lfs");
        if (lfs && lfs->kind == JSON_OBJECT) {
            int64_t ls = cwick_json_int(cwick_json_get(lfs, "size"), -1);
            if (ls > 0) size = ls;
        }
        if (!path || size < 0) continue;
        if (strcmp(path, ".cwick-repo") == 0) continue;
        tree_add(j, path, size);
    }
    cwick_json_free(arr);
    if (j->nfiles == 0) {
        snprintf(why, whycap, "the listing contained no downloadable files");
        return false;
    }
    j->total_bytes = 0;
    for (int i = 0; i < j->nfiles; i++) j->total_bytes += j->files[i].size;
    return true;
}

/* ------------------------------------------------------------------ */
/* one transfer                                                       */
/* ------------------------------------------------------------------ */
typedef struct {
    cwick_hub *h;
    hub_job *j;
    CURL *easy;
    FILE *fp;
    curl_off_t base;      /* bytes in files already renamed into place */
    curl_off_t offset;    /* bytes already in this file's .incomplete */
    bool first_write;
} xfer;

static size_t xfer_write(char *ptr, size_t size, size_t nmemb, void *userdata) {
    xfer *x = (xfer *)userdata;
    size_t n = size * nmemb;
    if (!x->first_write) {
        long code = 0;
        curl_easy_getinfo(x->easy, CURLINFO_RESPONSE_CODE, &code);
        if (code == 200 && x->offset > 0) {
            /* The remote ignored Range and is sending the whole object. Drop
             * the partial bytes instead of appending a second copy of the
             * file to them. */
            fflush(x->fp);
            if (ftruncate(fileno(x->fp), 0) == 0) {
                fseeko(x->fp, 0, SEEK_SET);
                x->offset = 0;
            }
        }
        x->first_write = true;
    }
    return fwrite(ptr, 1, n, x->fp);
}

static int xfer_progress(void *userdata, curl_off_t dltotal, curl_off_t dlnow,
                         curl_off_t ultotal, curl_off_t ulnow) {
    (void)dltotal; (void)ultotal; (void)ulnow;
    xfer *x = (xfer *)userdata;
    cwick_hub *h = x->h;
    hub_job *j = x->j;
    pthread_mutex_lock(&h->mtx);
    bool stop = h->stopping || j->cancel_req || j->pause_req;
    if (!stop) {
        j->done_bytes = x->base + x->offset + dlnow;
        double spd = 0;
        if (curl_easy_getinfo(x->easy, CURLINFO_SPEED_DOWNLOAD, &spd) == CURLE_OK)
            j->speed_bps = spd;
        int64_t left = j->total_bytes - j->done_bytes;
        j->eta_s = (j->speed_bps > 1.0 && left > 0)
                 ? (int64_t)((double)left / j->speed_bps) : -1;
        j->updated = time(NULL);
    }
    pthread_mutex_unlock(&h->mtx);
    return stop ? 1 : 0;
}

static void ensure_parent(const char *path) {
    char tmp[HUB_PATH_MAX];
    snprintf(tmp, sizeof(tmp), "%s", path);
    char *slash = strrchr(tmp, '/');
    if (!slash || slash == tmp) return;
    *slash = '\0';
    (void)cwick_mkdir_p(tmp);
}

/* Download one file into `<path>.incomplete`, then rename. Returns 0 on
 * success, 1 when the worker asked to pause/cancel, -1 on a real failure. */
static int download_file(cwick_hub *h, hub_job *j, int idx,
                         char *why, size_t whycap) {
    hub_file *f = &j->files[idx];
    char path[HUB_PATH_MAX], tmp[HUB_PATH_MAX];
    if (snprintf(path, sizeof(path), "%s/%s", j->dir, f->path) >=
        (int)sizeof(path)) { snprintf(why, whycap, "path too long"); return -1; }
    if (snprintf(tmp, sizeof(tmp), "%s.incomplete", path) >=
        (int)sizeof(tmp)) { snprintf(why, whycap, "path too long"); return -1; }

    struct stat st;
    if (stat(path, &st) == 0 && S_ISREG(st.st_mode) &&
        (int64_t)st.st_size == f->size) return 0;      /* already complete */

    curl_off_t offset = 0;
    if (stat(tmp, &st) == 0 && S_ISREG(st.st_mode)) offset = (curl_off_t)st.st_size;
    if (offset > (curl_off_t)f->size) {
        /* A stale partial (listing changed under us): start that file over. */
        unlink(tmp);
        offset = 0;
    }
    ensure_parent(path);
    FILE *fp = fopen(tmp, offset > 0 ? "ab" : "wb");
    if (!fp) {
        snprintf(why, whycap, "cannot open %s: %s", tmp, strerror(errno));
        return -1;
    }

    hbuf url = {0};
    hb_puts(&url, h->endpoint);
    hb_puts(&url, "/");
    url_encode(&url, j->repo, true);
    hb_puts(&url, "/resolve/main/");
    url_encode(&url, f->path, true);

    hbuf range = {0};
    struct curl_slist *headers = NULL;
    if (offset > 0) {
        hb_putf(&range, "Range: bytes=%lld-", (long long)offset);
        headers = curl_slist_append(headers, range.p);
    }

    CURL *c = curl_easy_init();
    if (!c) {
        fclose(fp); free(url.p); free(range.p);
        if (headers) curl_slist_free_all(headers);
        snprintf(why, whycap, "cannot create an HTTP handle");
        return -1;
    }
    xfer x = { h, j, c, fp, 0, offset, false };

    /* `base` counts the files before this one that are already in place;
     * recomputed for every file so a resumed job reports the right total. */
    for (int k = 0; k < idx; k++) {
        char done[HUB_PATH_MAX];
        struct stat ds;
        if (snprintf(done, sizeof(done), "%s/%s", j->dir, j->files[k].path) <
            (int)sizeof(done) && stat(done, &ds) == 0 &&
            S_ISREG(ds.st_mode) && (int64_t)ds.st_size == j->files[k].size)
            x.base += (curl_off_t)j->files[k].size;
    }

    curl_easy_setopt(c, CURLOPT_URL, url.p);
    curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(c, CURLOPT_MAXREDIRS, 8L);
    curl_easy_setopt(c, CURLOPT_USERAGENT, "cwick/1.0 (+local model library)");
    curl_easy_setopt(c, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, xfer_write);
    curl_easy_setopt(c, CURLOPT_WRITEDATA, &x);
    curl_easy_setopt(c, CURLOPT_XFERINFOFUNCTION, xfer_progress);
    curl_easy_setopt(c, CURLOPT_XFERINFODATA, &x);
    curl_easy_setopt(c, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, 15L);
    curl_easy_setopt(c, CURLOPT_LOW_SPEED_LIMIT, 1L);
    curl_easy_setopt(c, CURLOPT_LOW_SPEED_TIME, 60L);
    curl_easy_setopt(c, CURLOPT_BUFFERSIZE, 262144L);
    char errbuf[CURL_ERROR_SIZE];
    errbuf[0] = '\0';
    curl_easy_setopt(c, CURLOPT_ERRORBUFFER, errbuf);

    CURLcode rc = curl_easy_perform(c);
    long code = 0;
    curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
    curl_easy_cleanup(c);
    if (headers) curl_slist_free_all(headers);
    free(url.p);
    free(range.p);
    if (fclose(fp) != 0 && rc == CURLE_OK) {
        snprintf(why, whycap, "cannot flush %s: %s", tmp, strerror(errno));
        return -1;
    }

    pthread_mutex_lock(&h->mtx);
    bool asked_stop = j->pause_req || j->cancel_req || h->stopping;
    pthread_mutex_unlock(&h->mtx);
    if (asked_stop) return 1;

    if (rc != CURLE_OK) {
        snprintf(why, whycap, "%s", errbuf[0] ? errbuf : curl_easy_strerror(rc));
        return -1;
    }
    if (code != 200 && code != 206) {
        snprintf(why, whycap, "HTTP %ld for %s", code, f->path);
        return -1;
    }
    if (stat(tmp, &st) != 0 || (int64_t)st.st_size != f->size) {
        snprintf(why, whycap, "%s stopped at %lld of %lld bytes; resume to retry",
                 f->path, (long long)(stat(tmp, &st) == 0 ? (int64_t)st.st_size : -1),
                 (long long)f->size);
        return -1;
    }
    if (rename(tmp, path) != 0) {
        snprintf(why, whycap, "cannot rename %s: %s", tmp, strerror(errno));
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* worker                                                             */
/* ------------------------------------------------------------------ */
static void job_finish(cwick_hub *h, hub_job *j, job_state state,
                       const char *why) {
    pthread_mutex_lock(&h->mtx);
    j->state = state;
    if (why) snprintf(j->error, sizeof(j->error), "%s", why);
    j->speed_bps = 0;
    j->eta_s = -1;
    j->updated = time(NULL);
    j->thread_running = false;
    pthread_mutex_unlock(&h->mtx);
}

static void *hub_worker(void *arg) {
    hub_job *j = (hub_job *)arg;
    cwick_hub *h = j->owner;

    pthread_mutex_lock(&h->mtx);
    j->state = j->nfiles > 0 ? JOB_RUNNING : JOB_LISTING;
    j->error[0] = '\0';
    j->updated = time(NULL);
    if (!j->started) j->started = j->updated;
    pthread_mutex_unlock(&h->mtx);

    char why[256] = "";
    if (j->nfiles == 0) {
        if (!fetch_tree(h, j, why, sizeof(why))) {
            job_finish(h, j, JOB_ERROR, why);
            return NULL;
        }
        pthread_mutex_lock(&h->mtx);
        j->state = JOB_RUNNING;
        j->updated = time(NULL);
        pthread_mutex_unlock(&h->mtx);
    }

    for (int i = 0; i < j->nfiles; i++) {
        pthread_mutex_lock(&h->mtx);
        bool stop = h->stopping || j->cancel_req || j->pause_req;
        if (!stop) {
            snprintf(j->current, sizeof(j->current), "%s", j->files[i].path);
            j->state = JOB_RUNNING;
        }
        pthread_mutex_unlock(&h->mtx);
        if (stop) break;

        why[0] = '\0';
        int rc = download_file(h, j, i, why, sizeof(why));
        if (rc == 1) break;                 /* pause / cancel / shutdown */
        if (rc != 0) {
            job_finish(h, j, JOB_ERROR, why[0] ? why : "transfer failed");
            return NULL;
        }

        pthread_mutex_lock(&h->mtx);
        j->files_done = i + 1;
        j->done_bytes = 0;
        for (int k = 0; k <= i; k++) j->done_bytes += j->files[k].size;
        j->updated = time(NULL);
        pthread_mutex_unlock(&h->mtx);
    }

    pthread_mutex_lock(&h->mtx);
    bool canceled = j->cancel_req || h->stopping;
    bool paused = j->pause_req && !canceled;
    /* How much is still sitting in .incomplete files: this is what "resume"
     * will pick up, and what the UI shows as resumable bytes. */
    int64_t partial = 0;
    for (int k = 0; k < j->nfiles; k++) {
        char tmp[HUB_PATH_MAX];
        struct stat st;
        if (snprintf(tmp, sizeof(tmp), "%s/%s.incomplete", j->dir,
                     j->files[k].path) < (int)sizeof(tmp) &&
            stat(tmp, &st) == 0 && S_ISREG(st.st_mode))
            partial += (int64_t)st.st_size;
    }
    j->partial_bytes = partial;
    pthread_mutex_unlock(&h->mtx);

    if (canceled)      job_finish(h, j, JOB_CANCELED, NULL);
    else if (paused)   job_finish(h, j, JOB_PAUSED, NULL);
    else {
        pthread_mutex_lock(&h->mtx);
        j->done_bytes = j->total_bytes;
        j->files_done = j->nfiles;
        j->current[0] = '\0';
        pthread_mutex_unlock(&h->mtx);
        job_finish(h, j, JOB_DONE, NULL);
        CWICK_INFO("hub", "download %s complete: %d files, %.2f GiB -> %s",
                   j->repo, j->nfiles, (double)j->total_bytes / 1073741824.0,
                   j->dir);
    }
    return NULL;
}

static bool hub_launch(cwick_hub *h, hub_job *j) {
    (void)h;   /* the worker reads the hub through j->owner */
    j->thread_running = true;
    if (pthread_create(&j->thread, NULL, hub_worker, j) != 0) {
        j->thread_running = false;
        return false;
    }
    return true;
}

/* ------------------------------------------------------------------ */
/* public control                                                     */
/* ------------------------------------------------------------------ */
static void write_marker(const char *dir, const char *repo) {
    char p[HUB_PATH_MAX];
    if (snprintf(p, sizeof(p), "%s/.cwick-repo", dir) >= (int)sizeof(p)) return;
    (void)cwick_write_file(p, repo, strlen(repo));
}

char *cwick_hub_download_start(cwick_hub *h, const char *repo,
                               char *err, size_t errcap) {
    if (!h) { snprintf(err, errcap, "model library is not initialised"); return NULL; }
    if (!repo_safe(repo)) {
        snprintf(err, errcap, "repo must look like owner/name");
        return NULL;
    }
    if (cwick_mkdir_p(h->root) != 0) {
        snprintf(err, errcap, "cannot create the model directory %s", h->root);
        return NULL;
    }

    /* Keep the owner as a directory level, so two repositories with the same
     * short name cannot collide and the library matches the catalog's ids. */
    const char *last = strrchr(repo, '/');
    size_t owner_len = (size_t)(last - repo);
    char owner[256];
    if (owner_len >= sizeof(owner)) {
        snprintf(err, errcap, "repo owner is too long");
        return NULL;
    }
    memcpy(owner, repo, owner_len);
    owner[owner_len] = '\0';
    const char *base = last + 1;
    if (!*base) {
        snprintf(err, errcap, "repo must look like owner/name");
        return NULL;
    }
    char dir[HUB_PATH_MAX];
    if (snprintf(dir, sizeof(dir), "%s/%s/%s", h->root, owner, base) >=
        (int)sizeof(dir)) {
        snprintf(err, errcap, "repo name is too long");
        return NULL;
    }
    if (cwick_mkdir_p(dir) != 0) {
        snprintf(err, errcap, "cannot create %s", dir);
        return NULL;
    }
    write_marker(dir, repo);

    pthread_mutex_lock(&h->mtx);
    hub_job *j = NULL;
    /* An active job for the same repository is idempotent: clicking twice
     * returns the same id instead of starting two writers on one file. */
    for (int i = 0; i < h->njobs; i++) {
        hub_job *c = &h->jobs[i];
        if (strcmp(c->repo, repo) == 0 && c->thread_running) { j = c; break; }
    }
    if (!j) {
        for (int i = 0; i < h->njobs; i++) {
            hub_job *c = &h->jobs[i];
            if (strcmp(c->repo, repo) == 0 && !c->thread_running) { j = c; break; }
        }
    }
    if (!j && h->njobs < HUB_MAX_JOBS) j = &h->jobs[h->njobs++];
    if (!j) {
        /* Reuse a finished slot only when no worker can still touch it. */
        for (int i = 0; i < h->njobs; i++) {
            hub_job *c = &h->jobs[i];
            if (!c->thread_running &&
                (c->state == JOB_DONE || c->state == JOB_ERROR ||
                 c->state == JOB_CANCELED)) { j = c; break; }
        }
    }
    if (!j) {
        pthread_mutex_unlock(&h->mtx);
        snprintf(err, errcap, "the download queue is full (%d jobs)", HUB_MAX_JOBS);
        return NULL;
    }

    bool fresh = j->id[0] == '\0' || strcmp(j->repo, repo) != 0;
    if (fresh) {
        free(j->files);
        memset(j, 0, sizeof(*j));
        j->owner = h;
        snprintf(j->id, sizeof(j->id), "d%u", ++h->next_id);
        snprintf(j->repo, sizeof(j->repo), "%s", repo);
        snprintf(j->dir, sizeof(j->dir), "%s", dir);
        j->eta_s = -1;
    }
    j->pause_req = false;
    j->cancel_req = false;
    j->state = JOB_QUEUED;
    j->speed_bps = 0;
    j->error[0] = '\0';
    j->started = time(NULL);
    j->updated = j->started;
    if (!hub_launch(h, j)) {
        j->state = JOB_ERROR;
        snprintf(j->error, sizeof(j->error), "cannot start a download thread");
    }
    pthread_mutex_unlock(&h->mtx);
    return job_only_json(h, j);
}

char *cwick_hub_download_action(cwick_hub *h, const char *id,
                                const char *action, char *err, size_t errcap) {
    if (!h) { snprintf(err, errcap, "model library is not initialised"); return NULL; }
    bool pause = strcmp(action, "pause") == 0;
    bool resume = strcmp(action, "resume") == 0 || strcmp(action, "retry") == 0;
    bool cancel = strcmp(action, "cancel") == 0;
    if (!pause && !resume && !cancel) {
        snprintf(err, errcap, "action must be pause, resume, retry or cancel");
        return NULL;
    }

    pthread_mutex_lock(&h->mtx);
    hub_job *j = find_job(h, id);
    if (!j) {
        pthread_mutex_unlock(&h->mtx);
        snprintf(err, errcap, "no download with id %s", id);
        return NULL;
    }
    if (pause) {
        if (j->thread_running) j->pause_req = true;
        else j->state = JOB_PAUSED;
    } else if (cancel) {
        j->cancel_req = true;
        if (!j->thread_running) j->state = JOB_CANCELED;
    } else {
        /* resume / retry: the listing is cached in the job, so retrying does
         * not pay for another catalog call, and .incomplete files are kept. */
        j->pause_req = false;
        j->cancel_req = false;
        j->error[0] = '\0';
        j->speed_bps = 0;
        j->eta_s = -1;
        if (j->state == JOB_DONE) j->state = JOB_QUEUED;   /* verify again */
        if (!j->thread_running) {
            j->state = JOB_QUEUED;
            if (!hub_launch(h, j)) {
                j->state = JOB_ERROR;
                snprintf(j->error, sizeof(j->error),
                         "cannot start a download thread");
            }
        } else {
            j->state = JOB_RUNNING;
        }
    }
    j->updated = time(NULL);
    hub_job *out = j;
    pthread_mutex_unlock(&h->mtx);
    return job_only_json(h, out);
}

/* ------------------------------------------------------------------ */
/* lifecycle                                                          */
/* ------------------------------------------------------------------ */
cwick_hub *cwick_hub_open(const char *root, const char *endpoint) {
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) return NULL;
    cwick_hub *h = (cwick_hub *)calloc(1, sizeof(*h));
    if (!h) { curl_global_cleanup(); return NULL; }
    pthread_mutex_init(&h->mtx, NULL);
    snprintf(h->root, sizeof(h->root), "%s", root && *root ? root : "./models");
    snprintf(h->endpoint, sizeof(h->endpoint), "%s",
             endpoint && *endpoint ? endpoint : HUB_DEFAULT_ENDPOINT);
    /* A trailing slash would produce a double slash in every catalog URL. */
    size_t n = strlen(h->endpoint);
    while (n && h->endpoint[n - 1] == '/') h->endpoint[--n] = '\0';
    h->curl_ready = true;
    (void)cwick_mkdir_p(h->root);
    CWICK_INFO("hub", "model library %s, catalog %s", h->root, h->endpoint);
    return h;
}

void cwick_hub_close(cwick_hub *h) {
    if (!h) return;
    pthread_mutex_lock(&h->mtx);
    h->stopping = true;
    for (int i = 0; i < h->njobs; i++) h->jobs[i].cancel_req = true;
    pthread_t threads[HUB_MAX_JOBS];
    int nt = 0;
    for (int i = 0; i < h->njobs; i++)
        if (h->jobs[i].thread_running) threads[nt++] = h->jobs[i].thread;
    pthread_mutex_unlock(&h->mtx);

    for (int i = 0; i < nt; i++) pthread_join(threads[i], NULL);
    for (int i = 0; i < h->njobs; i++) free(h->jobs[i].files);
    pthread_mutex_destroy(&h->mtx);
    if (h->curl_ready) curl_global_cleanup();
    free(h);
}
