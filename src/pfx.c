/* pfx.c — see pfx.h. Prefix snapshots in one file per prefix, named by a
 * 128-bit hash of the token sequence, with an in-memory index rebuilt by
 * reading headers on open (so a restart reuses the cache). */
#include "pfx.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "util.h"

#define PFX_MAGIC   0x464b50474c4743ULL   /* "CWICKPFK" */
#define PFX_VERSION 1
#define PFX_EXT     ".pfx"

/* Two independent 64-bit hashes over the token bytes. 128 bits is not about
 * defeating an adversary -- a collision costs a memcmp against the stored
 * tokens before anything is applied -- it is about never having to think
 * about birthday bounds with millions of prompt turns. */
#define H1_OFF 0xcbf29ce484222325ULL
#define H1_MUL 0x100000001b3ULL
#define H2_OFF 0x9e3779b97f4a7c15ULL
#define H2_MUL 0xff51afd7ed558ccdULL

typedef struct {
    uint64_t h1, h2;
    int64_t pos;
    uint64_t bytes;
    uint64_t seq;      /* creation order, for oldest-first eviction */
    uint64_t hits;     /* replays served from this entry */
    char *path;
} pfx_entry;

struct cwick_pfx_cache {
    char *dir;
    uint64_t cap_bytes;
    pfx_entry *ents;
    int n_ents, cap_ents;
    uint64_t total_bytes;
    uint64_t seq;
};

/* ------------------------------------------------------------------ */
/* Rolling hashes                                                      */
/* ------------------------------------------------------------------ */
static void hash_tokens(const uint32_t *ids, int64_t n, uint64_t *h1_out,
                        uint64_t *h2_out) {
    uint64_t h1 = H1_OFF, h2 = H2_OFF;
    for (int64_t i = 0; i < n; i++) {
        uint32_t t = ids[i];
        for (int b = 0; b < 4; b++) {
            uint8_t byte = (uint8_t)(t >> (8 * b));
            h1 = (h1 ^ byte) * H1_MUL;
            h2 = (h2 ^ byte) * H2_MUL;
        }
    }
    *h1_out = h1;
    *h2_out = h2;
}

/* ------------------------------------------------------------------ */
/* Header                                                              */
/* ------------------------------------------------------------------ */
typedef struct {
    int64_t pos, nslots, lat_stride, v_stride, L, lin_heads, lin_dim;
    int64_t kpool, hd, pool_cap, H, max_anchors, n_anchors;
    uint64_t h1, h2;
    /* Backend kind, stored in the header word that used to be padding. Value 0
     * is the original layout, so snapshots written before this field existed
     * still parse as kind 0 with no size change. Kind 1 (DeepSeek) appends one
     * opaque blob, preceded by its u64 length, after the token list; the
     * backend that wrote a blob owns its layout. */
    uint32_t kind;
} pfx_header;

static bool w64(FILE *f, uint64_t v) { return fwrite(&v, 8, 1, f) == 1; }
static bool w32(FILE *f, uint32_t v) { return fwrite(&v, 4, 1, f) == 1; }
static bool r64(FILE *f, uint64_t *v) { return fread(v, 8, 1, f) == 1; }
static bool r32(FILE *f, uint32_t *v) { return fread(v, 4, 1, f) == 1; }

static bool write_header(FILE *f, const pfx_header *h) {
    if (!w64(f, PFX_MAGIC)) return false;
    if (!w32(f, PFX_VERSION)) return false;
    if (!w32(f, h->kind)) return false;
    const int64_t fld[] = { h->pos, h->nslots, h->lat_stride, h->v_stride,
                            h->L, h->lin_heads, h->lin_dim, h->kpool, h->hd,
                            h->pool_cap, h->H, h->max_anchors, h->n_anchors };
    for (int i = 0; i < 13; i++)
        if (!w64(f, (uint64_t)fld[i])) return false;
    if (!w64(f, h->h1)) return false;
    return w64(f, h->h2);
}

static bool read_header(FILE *f, pfx_header *h) {
    uint64_t magic = 0;
    uint32_t ver = 0, kind = 0;
    if (!r64(f, &magic) || magic != PFX_MAGIC) return false;
    if (!r32(f, &ver) || ver != PFX_VERSION) return false;
    if (!r32(f, &kind) || kind > 2) return false;   /* unknown backend */
    h->kind = kind;
    int64_t *fld[] = { &h->pos, &h->nslots, &h->lat_stride, &h->v_stride,
                       &h->L, &h->lin_heads, &h->lin_dim, &h->kpool, &h->hd,
                       &h->pool_cap, &h->H, &h->max_anchors, &h->n_anchors };
    for (int i = 0; i < 13; i++) {
        uint64_t v = 0;
        if (!r64(f, &v)) return false;
        *fld[i] = (int64_t)v;
    }
    return r64(f, &h->h1) && r64(f, &h->h2);
}

/* `total_pools` is the sum of the per-slot valid pool counts: the indexer
 * grows its pool array as it compresses, and writing the whole pool_cap
 * would put 25,002 rows on disk for the handful a 400-token prefix actually
 * built (measured: 140 MB of zeros inside a 352 MB snapshot). */
static uint64_t blob_bytes(const pfx_header *h, int64_t total_pools,
                           uint64_t blob_len) {
    uint64_t n = 0;
    const uint64_t P = (uint64_t)(h->pos > 0 ? h->pos : 0);
    const uint64_t S = (uint64_t)(h->nslots > 0 ? h->nslots : 0);
    const uint64_t TP = (uint64_t)(total_pools > 0 ? total_pools : 0);
    n += P * 4;                                             /* tokens */
    /* Kind 1/2 store one opaque blob, length-prefixed, and nothing else. */
    if (h->kind >= 1) return n + 8 + blob_len;
    n += S * P * (uint64_t)h->lat_stride * 4;               /* keys */
    if (h->v_stride > 0)
        n += S * P * (uint64_t)h->v_stride * 4;             /* values */
    n += (uint64_t)h->L * (uint64_t)h->lin_heads *
         (uint64_t)h->lin_dim * (uint64_t)h->lin_dim * 4;   /* kda_rec */
    n += (uint64_t)h->L * 3u *
         ((uint64_t)h->lin_heads * (uint64_t)h->lin_dim) * 3u * 4; /* kda_conv */
    n += S * 8;                                             /* idx_npools */
    n += TP * (uint64_t)h->hd * 4;                          /* idx_pool */
    n += S * 2u * (uint64_t)h->kpool * (uint64_t)h->hd * 4; /* idx_tail */
    n += (uint64_t)h->max_anchors * (uint64_t)h->H * 4;     /* anchors */
    return n;
}

/* ------------------------------------------------------------------ */
/* State allocation / free                                             */
/* ------------------------------------------------------------------ */
void cwick_prefix_state_free(cwick_prefix_state *s) {
    if (!s) return;
    free(s->keys); free(s->values); free(s->kda_rec); free(s->kda_conv);
    free(s->idx_pool); free(s->idx_tail); free(s->idx_npools); free(s->anchors);
    free(s->blob);
    memset(s, 0, sizeof(*s));
}

static bool state_alloc(cwick_prefix_state *s, const pfx_header *h) {
    memset(s, 0, sizeof(*s));
    s->kind = (int)h->kind;
    /* A kind-1/2 snapshot is one opaque blob: the backend that wrote it owns
     * the layout, so there is nothing here to allocate (and nothing to
     * mis-size from header fields that mean something else for that backend). */
    if (h->kind >= 1) {
        s->pos = h->pos;
        return true;
    }
    s->pos = h->pos; s->nslots = h->nslots; s->lat_stride = h->lat_stride;
    s->v_stride = h->v_stride; s->L = h->L; s->lin_heads = h->lin_heads;
    s->lin_dim = h->lin_dim; s->kpool = h->kpool; s->hd = h->hd;
    s->pool_cap = h->pool_cap; s->H = h->H; s->max_anchors = h->max_anchors;
    s->n_anchors = h->n_anchors;

    const size_t P = (size_t)(h->pos > 0 ? h->pos : 0);
    const size_t S = (size_t)(h->nslots > 0 ? h->nslots : 0);
    s->keys = cwick_xmalloc(S * P * (size_t)h->lat_stride * sizeof(float));
    if (h->v_stride > 0)
        s->values = cwick_xmalloc(S * P * (size_t)h->v_stride * sizeof(float));
    s->kda_rec = cwick_xmalloc((size_t)h->L * (size_t)h->lin_heads *
                              (size_t)h->lin_dim * (size_t)h->lin_dim *
                              sizeof(float));
    s->kda_conv = cwick_xmalloc((size_t)h->L * 3 *
                               ((size_t)h->lin_heads * (size_t)h->lin_dim) * 3 *
                               sizeof(float));
    s->idx_pool = cwick_xmalloc(S * (size_t)h->pool_cap * (size_t)h->hd *
                               sizeof(float));
    s->idx_tail = cwick_xmalloc(S * 2 * (size_t)h->kpool * (size_t)h->hd *
                               sizeof(float));
    s->idx_npools = cwick_xmalloc(S * sizeof(int64_t));
    s->anchors = cwick_xmalloc((size_t)(h->max_anchors > 0 ? h->max_anchors : 1) *
                              (size_t)h->H * sizeof(float));
    return s->keys && s->kda_rec && s->kda_conv && s->idx_pool &&
           s->idx_tail && s->idx_npools && s->anchors &&
           (h->v_stride <= 0 || s->values);
}

/* Exported for engine.c, which owns the live state and so does the capture:
 * it allocates the same buffers the loader uses, from the same dimensions. */
bool cwick_prefix_state_alloc_dims(cwick_prefix_state *s, int64_t pos,
                                  int64_t nslots, int64_t lat_stride,
                                  int64_t v_stride, int64_t L, int64_t lin_heads,
                                  int64_t lin_dim, int64_t kpool, int64_t hd,
                                  int64_t pool_cap, int64_t H, int64_t max_anchors) {
    pfx_header h;
    memset(&h, 0, sizeof(h));
    h.pos = pos; h.nslots = nslots; h.lat_stride = lat_stride;
    h.v_stride = v_stride; h.L = L; h.lin_heads = lin_heads; h.lin_dim = lin_dim;
    h.kpool = kpool; h.hd = hd; h.pool_cap = pool_cap; h.H = H;
    h.max_anchors = max_anchors;
    return state_alloc(s, &h);
}

/* ------------------------------------------------------------------ */
/* Cache lifecycle                                                     */
/* ------------------------------------------------------------------ */
static void pfx_path(char *buf, size_t cap, const cwick_pfx_cache *c,
                     uint64_t h1, uint64_t h2, int64_t pos) {
    snprintf(buf, cap, "%s/%016llx%016llx_%lld%s", c->dir,
             (unsigned long long)h1, (unsigned long long)h2,
             (long long)pos, PFX_EXT);
}

static bool index_add(cwick_pfx_cache *c, uint64_t h1, uint64_t h2, int64_t pos,
                      uint64_t bytes, const char *path) {
    if (c->n_ents == c->cap_ents) {
        int ncap = c->cap_ents ? c->cap_ents * 2 : 16;
        pfx_entry *ne = realloc(c->ents, (size_t)ncap * sizeof(pfx_entry));
        if (!ne) return false;
        c->ents = ne;
        c->cap_ents = ncap;
    }
    pfx_entry *e = &c->ents[c->n_ents++];
    memset(e, 0, sizeof(*e));
    e->h1 = h1; e->h2 = h2; e->pos = pos; e->bytes = bytes;
    e->seq = ++c->seq;
    e->path = cwick_strdup(path);
    c->total_bytes += bytes;
    return e->path != NULL;
}

static void index_drop(cwick_pfx_cache *c, int i) {
    c->total_bytes -= c->ents[i].bytes;
    unlink(c->ents[i].path);
    free(c->ents[i].path);
    memmove(&c->ents[i], &c->ents[i + 1],
            (size_t)(c->n_ents - i - 1) * sizeof(pfx_entry));
    c->n_ents--;
}

/* Eviction is deliberately not plain LRU.
 *
 * An agent session grows its prompt every turn: turn 1 caches 403 tokens,
 * turn 2 caches 500, turn 3 caches 600, and so on. Under plain LRU the
 * *oldest* entry goes first -- and the oldest entry is the short base prompt
 * that every later turn still starts with, i.e. the single most valuable
 * thing in the cache. Losing it costs a full prefill on every turn after.
 *
 * So the victim is chosen by usefulness first: an entry that has actually
 * served replays is kept over one that has never been hit, and only among
 * equally-useful entries does age decide. */
static uint64_t entry_cost(const pfx_entry *e, int64_t longest) {
    /* Lower is more evictable. Three keys, in order of weight:
     *
     *   1. replays served -- an entry nobody has asked for again is worth
     *      less than one that is demonstrably in use;
     *   2. length -- a *short* prefix is the base that longer prompts extend,
     *      so it can be replayed by every turn of a session, while a long
     *      per-turn snapshot can only ever serve itself. This is the key that
     *      saves the base prompt in the window before it has earned a hit;
     *   3. age -- among equals, the older one goes first.
     *
     * Length has to outrank age: with plain LRU the base prompt is evicted
     * the moment the next turn's snapshot arrives, because it is the oldest
     * thing in the cache and has not yet served a replay. */
    uint64_t h = e->hits;
    if (h > 1023) h = 1023;
    uint64_t len_rank = (uint64_t)(longest > e->pos ? longest - e->pos : 0);
    if (len_rank > 4095) len_rank = 4095;
    return (h << 52) | (len_rank << 40) | ((uint64_t)e->seq & 0xFFFFFFFFFull);
}

static void evict_to_fit(cwick_pfx_cache *c) {
    while (c->total_bytes > c->cap_bytes && c->n_ents > 1) {
        int64_t longest = 0;
        for (int i = 0; i < c->n_ents; i++)
            if (c->ents[i].pos > longest) longest = c->ents[i].pos;
        int victim = 0;
        for (int i = 1; i < c->n_ents; i++)
            if (entry_cost(&c->ents[i], longest) <
                entry_cost(&c->ents[victim], longest))
                victim = i;
        index_drop(c, victim);
    }
}

cwick_pfx_cache *cwick_pfx_open(const char *dir, uint64_t cap_bytes) {
    if (!dir || !*dir) return NULL;
    if (mkdir(dir, 0755) != 0 && errno != EEXIST) return NULL;

    cwick_pfx_cache *c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->dir = cwick_strdup(dir);
    c->cap_bytes = cap_bytes;
    if (!c->dir) { free(c); return NULL; }

    /* Rebuild the index from the headers already on disk: a restart should
     * reuse what the previous run cached, and the header is 136 bytes. */
    DIR *d = opendir(dir);
    if (d) {
        struct dirent *de;
        while ((de = readdir(d)) != NULL) {
            const char *n = de->d_name;
            size_t len = strlen(n);
            if (len < 4 || strcmp(n + len - 4, PFX_EXT) != 0) continue;
            char path[4096];
            snprintf(path, sizeof(path), "%s/%s", dir, n);
            FILE *f = fopen(path, "rb");
            if (!f) continue;
            pfx_header h;
            bool ok = read_header(f, &h);
            fclose(f);
            if (!ok) { unlink(path); continue; }
            struct stat st;
            uint64_t bytes = 0;
            if (stat(path, &st) == 0) bytes = (uint64_t)st.st_size;
            index_add(c, h.h1, h.h2, h.pos, bytes, path);
        }
        closedir(d);
    }
    evict_to_fit(c);
    return c;
}

void cwick_pfx_close(cwick_pfx_cache *c) {
    if (!c) return;
    for (int i = 0; i < c->n_ents; i++) free(c->ents[i].path);
    free(c->ents);
    free(c->dir);
    free(c);
}

void cwick_pfx_stats(const cwick_pfx_cache *c, int64_t *entries, uint64_t *bytes) {
    if (entries) *entries = c ? c->n_ents : 0;
    if (bytes) *bytes = c ? c->total_bytes : 0;
}

/* Drop every snapshot. A prefix cache entry is the *inference state of one
 * token sequence under one checkpoint* — layer counts, hidden size, KV
 * strides — so a model switch invalidates all of them, and a snapshot written
 * by another checkpoint must never be replayed into this one even though the
 * tokens it was keyed by hash the same way. Called from the switch path, which
 * runs with no request in flight. */
void cwick_pfx_clear(cwick_pfx_cache *c) {
    if (!c) return;
    for (int i = 0; i < c->n_ents; i++) {
        if (c->ents[i].path) {
            unlink(c->ents[i].path);
            free(c->ents[i].path);
        }
    }
    c->n_ents = 0;
    c->total_bytes = 0;
}

/* ------------------------------------------------------------------ */
/* Lookup                                                              */
/* ------------------------------------------------------------------ */
int64_t cwick_pfx_lookup(cwick_pfx_cache *c, const uint32_t *ids, int64_t n) {
    if (!c || !ids || n <= 0 || c->n_ents == 0) return 0;

    /* One pass: two rolling hashes, sampled at every cached prefix length.
     * A 30K-token prompt against a handful of entries is a few microseconds
     * of hashing, versus re-prefilling those 30K tokens for hours. */
    uint64_t h1 = H1_OFF, h2 = H2_OFF;
    int64_t best = 0;
    for (int64_t i = 0; i < n; i++) {
        uint32_t t = ids[i];
        for (int b = 0; b < 4; b++) {
            uint8_t byte = (uint8_t)(t >> (8 * b));
            h1 = (h1 ^ byte) * H1_MUL;
            h2 = (h2 ^ byte) * H2_MUL;
        }
        const int64_t P = i + 1;
        for (int k = 0; k < c->n_ents; k++) {
            pfx_entry *e = &c->ents[k];
            if (e->pos != P || P <= best) continue;
            if (e->h1 == h1 && e->h2 == h2) {
                best = P;
                /* Counted here rather than at apply time: what matters for
                 * eviction is that a prefix was *asked for* again.
                 *
                 * The count is in memory only -- it is not persisted, so a
                 * restart starts every entry at zero. That is a deliberate
                 * trade: writing it back would mean touching a file on every
                 * request, and the worst case after a restart is one eviction
                 * that picks the base prompt, which costs a single prefill
                 * while the longer snapshots are still there to serve. */
                e->hits++;
                break;
            }
        }
    }
    return best;
}

/* ------------------------------------------------------------------ */
/* Load / store                                                        */
/* ------------------------------------------------------------------ */
cwick_status cwick_pfx_load(cwick_pfx_cache *c, const uint32_t *ids, int64_t P,
                          cwick_prefix_state *out, cwick_error *err) {
    cwick_error_clear(err);
    if (!c || !ids || !out || P <= 0) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "pfx: bad load arguments");
        return CWICK_ERR_ARGUMENT;
    }
    uint64_t h1 = 0, h2 = 0;
    hash_tokens(ids, P, &h1, &h2);

    const pfx_entry *hit = NULL;
    for (int k = 0; k < c->n_ents; k++)
        if (c->ents[k].pos == P && c->ents[k].h1 == h1 && c->ents[k].h2 == h2) {
            hit = &c->ents[k];
            break;
        }
    if (!hit) {
        cwick_error_set(err, CWICK_ERR_NOT_FOUND, "pfx: no entry of length %lld",
                       (long long)P);
        return CWICK_ERR_NOT_FOUND;
    }

    FILE *f = fopen(hit->path, "rb");
    if (!f) {
        cwick_error_set(err, CWICK_ERR_IO, "pfx: cannot open %s (%s)", hit->path,
                       strerror(errno));
        return CWICK_ERR_IO;
    }
    pfx_header h;
    if (!read_header(f, &h)) {
        fclose(f);
        cwick_error_set(err, CWICK_ERR_IO, "pfx: bad header in %s", hit->path);
        return CWICK_ERR_IO;
    }
    if (h.pos != P) {
        fclose(f);
        cwick_error_set(err, CWICK_ERR_IO, "pfx: length mismatch in %s", hit->path);
        return CWICK_ERR_IO;
    }

    /* Confirm the tokens before anything is applied: the hash narrows it to
     * one candidate, this decides. */
    uint32_t *stored = cwick_xmalloc((size_t)P * sizeof(uint32_t));
    if (!stored) {
        fclose(f);
        cwick_error_set(err, CWICK_ERR_MEMORY, "pfx: out of memory");
        return CWICK_ERR_MEMORY;
    }
    if (fread(stored, sizeof(uint32_t), (size_t)P, f) != (size_t)P ||
        memcmp(stored, ids, (size_t)P * sizeof(uint32_t)) != 0) {
        free(stored);
        fclose(f);
        cwick_error_set(err, CWICK_ERR_NOT_FOUND, "pfx: token mismatch");
        return CWICK_ERR_NOT_FOUND;
    }
    free(stored);

    if (!state_alloc(out, &h)) {
        fclose(f);
        cwick_prefix_state_free(out);
        cwick_error_set(err, CWICK_ERR_MEMORY, "pfx: out of memory");
        return CWICK_ERR_MEMORY;
    }
    if (h.kind >= 1) {
        uint64_t blen = 0;
        if (!r64(f, &blen) || blen == 0) {
            fclose(f);
            cwick_prefix_state_free(out);
            cwick_error_set(err, CWICK_ERR_IO, "pfx: bad blob length in %s",
                            hit->path);
            return CWICK_ERR_IO;
        }
        out->blob = cwick_xmalloc((size_t)blen);
        if (!out->blob ||
            fread(out->blob, 1, (size_t)blen, f) != (size_t)blen) {
            fclose(f);
            cwick_prefix_state_free(out);
            cwick_error_set(err, CWICK_ERR_IO, "pfx: truncated blob in %s",
                            hit->path);
            return CWICK_ERR_IO;
        }
        out->blob_len = (size_t)blen;
        fclose(f);
        return CWICK_OK;
    }

#define READ_BLOB(ptr, count)                                              \
    do {                                                                   \
        size_t cnt = (size_t)(count);                                      \
        if (cnt && fread((ptr), sizeof(float), cnt, f) != cnt) {           \
            fclose(f);                                                     \
            cwick_prefix_state_free(out);                                   \
            cwick_error_set(err, CWICK_ERR_IO, "pfx: truncated read in %s",  \
                           hit->path);                                     \
            return CWICK_ERR_IO;                                            \
        }                                                                  \
    } while (0)

    const size_t S = (size_t)(h.nslots > 0 ? h.nslots : 0);
    const size_t Pz = (size_t)P;
    READ_BLOB(out->keys, S * Pz * (size_t)h.lat_stride);
    if (h.v_stride > 0)
        READ_BLOB(out->values, S * Pz * (size_t)h.v_stride);
    READ_BLOB(out->kda_rec, (size_t)h.L * (size_t)h.lin_heads *
                            (size_t)h.lin_dim * (size_t)h.lin_dim);
    READ_BLOB(out->kda_conv, (size_t)h.L * 3 *
                             ((size_t)h.lin_heads * (size_t)h.lin_dim) * 3);
    /* The pool counts come before the pool rows: only the rows a prefix
     * actually built are on disk, so their number has to be read first. */
    if (S && fread(out->idx_npools, sizeof(int64_t), S, f) != S) {
        fclose(f);
        cwick_prefix_state_free(out);
        cwick_error_set(err, CWICK_ERR_IO, "pfx: truncated npools in %s", hit->path);
        return CWICK_ERR_IO;
    }
    size_t pool_rows = 0, off = 0;
    for (size_t k = 0; k < S; k++) {
        int64_t np = out->idx_npools[k];
        if (np < 0) np = 0;
        if (np > h.pool_cap) np = h.pool_cap;
        out->idx_npools[k] = np;
        pool_rows += (size_t)np;
    }
    if (pool_rows &&
        fread(out->idx_pool, sizeof(float), pool_rows * (size_t)h.hd, f) !=
            pool_rows * (size_t)h.hd) {
        fclose(f);
        cwick_prefix_state_free(out);
        cwick_error_set(err, CWICK_ERR_IO, "pfx: truncated pools in %s", hit->path);
        return CWICK_ERR_IO;
    }
    /* Scatter the rows back to their per-slot offsets: on disk they are
     * packed, in memory each slot owns a pool_cap-row window. */
    for (size_t k = S; k-- > 0;) {
        const size_t np = (size_t)out->idx_npools[k];
        if (!np) continue;
        off += np;
        const size_t dst = k * (size_t)h.pool_cap * (size_t)h.hd;
        const size_t src = (off - np) * (size_t)h.hd;
        memmove(out->idx_pool + dst, out->idx_pool + src,
                np * (size_t)h.hd * sizeof(float));
    }
    READ_BLOB(out->idx_tail, S * 2 * (size_t)h.kpool * (size_t)h.hd);
    READ_BLOB(out->anchors, (size_t)h.max_anchors * (size_t)h.H);
#undef READ_BLOB

    fclose(f);
    return CWICK_OK;
}

cwick_status cwick_pfx_store(cwick_pfx_cache *c, const uint32_t *ids, int64_t P,
                           const cwick_prefix_state *s, cwick_error *err) {
    cwick_error_clear(err);
    if (!c || !ids || !s || P <= 0 || P != s->pos) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "pfx: bad store arguments");
        return CWICK_ERR_ARGUMENT;
    }
    uint64_t h1 = 0, h2 = 0;
    hash_tokens(ids, P, &h1, &h2);
    for (int k = 0; k < c->n_ents; k++)
        if (c->ents[k].pos == P && c->ents[k].h1 == h1 && c->ents[k].h2 == h2)
            return CWICK_OK;   /* already cached */

    pfx_header h;
    memset(&h, 0, sizeof(h));
    h.pos = s->pos; h.nslots = s->nslots; h.lat_stride = s->lat_stride;
    h.v_stride = s->v_stride; h.L = s->L; h.lin_heads = s->lin_heads;
    h.lin_dim = s->lin_dim; h.kpool = s->kpool; h.hd = s->hd;
    h.pool_cap = s->pool_cap; h.H = s->H; h.max_anchors = s->max_anchors;
    h.n_anchors = s->n_anchors; h.h1 = h1; h.h2 = h2;
    /* Kind 0 = GLM layout, 1 = DeepSeek blob: preserve the writer's kind so
     * the right backend claims the snapshot on load. */
    h.kind = (uint32_t)(s->kind >= 1 ? s->kind : 0);

    char path[4096], tmp[4096];
    pfx_path(path, sizeof(path), c, h1, h2, P);
    snprintf(tmp, sizeof(tmp), "%s.tmp%lld", path, (long long)getpid());

    /* Written to a temp name and renamed: a serve killed mid-write must not
     * leave a half file that a later run reads back as a valid prefix.
     *
     * F_NOCACHE, for the same reason the expert stream uses it: a snapshot is
     * hundreds of megabytes of dirty pages that will never be read back in
     * this process's lifetime, and letting them into the page cache makes the
     * VM evict the resident weight bank instead. Measured on a 8147-token
     * snapshot: the write stalled 3.2 s and dropped RSS from 72 GiB to
     * 41 GiB, which turned the next decode step from 0.7 s into 4.6 s. */
    FILE *f = fopen(tmp, "wb");
    if (!f) {
        cwick_error_set(err, CWICK_ERR_IO, "pfx: cannot write %s (%s)", tmp,
                       strerror(errno));
        return CWICK_ERR_IO;
    }
    if (fcntl(fileno(f), F_NOCACHE, 1) < 0) {
        /* Not fatal: the cache still works, it just competes with the working
         * set. Every version of macOS that runs this has F_NOCACHE. */
    }
    if (!write_header(f, &h) ||
        fwrite(ids, sizeof(uint32_t), (size_t)P, f) != (size_t)P) {
        fclose(f); unlink(tmp);
        cwick_error_set(err, CWICK_ERR_IO, "pfx: header/token write failed");
        return CWICK_ERR_IO;
    }
    const size_t S = (size_t)(s->nslots > 0 ? s->nslots : 0);
    const size_t Pz = (size_t)P;
    bool ok = true;
    int64_t total_pools = 0;
    if (s->kind >= 1) {
        /* One opaque blob, length-prefixed so the loader can size it without
         * knowing the backend's layout. */
        const uint64_t blen = (uint64_t)s->blob_len;
        if (!s->blob || !blen || !w64(f, blen) ||
            fwrite(s->blob, 1, (size_t)blen, f) != (size_t)blen)
            ok = false;
    } else {
#define WRITE_BLOB(ptr, count)                                             \
    do {                                                                   \
        size_t cnt = (size_t)(count);                                      \
        if (cnt && fwrite((ptr), sizeof(float), cnt, f) != cnt) ok = false; \
    } while (0)
    WRITE_BLOB(s->keys, S * Pz * (size_t)h.lat_stride);
    if (ok && h.v_stride > 0)
        WRITE_BLOB(s->values, S * Pz * (size_t)h.v_stride);
    WRITE_BLOB(s->kda_rec, (size_t)h.L * (size_t)h.lin_heads *
                           (size_t)h.lin_dim * (size_t)h.lin_dim);
    WRITE_BLOB(s->kda_conv, (size_t)h.L * 3 *
                            ((size_t)h.lin_heads * (size_t)h.lin_dim) * 3);
    /* Pool counts first, then only the rows they say are valid. */
    for (size_t k = 0; k < S; k++) {
        int64_t np = s->idx_npools ? s->idx_npools[k] : 0;
        if (np < 0) np = 0;
        if (np > h.pool_cap) np = h.pool_cap;
        total_pools += np;
    }
    if (ok && S && fwrite(s->idx_npools, sizeof(int64_t), S, f) != S) ok = false;
    for (size_t k = 0; k < S && ok; k++) {
        const size_t np = (size_t)(s->idx_npools ? s->idx_npools[k] : 0);
        if (np && fwrite(s->idx_pool + k * (size_t)h.pool_cap * (size_t)h.hd,
                         sizeof(float), np * (size_t)h.hd, f) !=
                      np * (size_t)h.hd)
            ok = false;
    }
    WRITE_BLOB(s->idx_tail, S * 2 * (size_t)h.kpool * (size_t)h.hd);
    WRITE_BLOB(s->anchors, (size_t)h.max_anchors * (size_t)h.H);
#undef WRITE_BLOB
    }
    if (!ok || fflush(f) != 0) {
        fclose(f); unlink(tmp);
        cwick_error_set(err, CWICK_ERR_IO, "pfx: state write failed");
        return CWICK_ERR_IO;
    }
    /* A short write on a 0.9 GB snapshot is the failure mode that would be
     * read back later as a corrupt prefix, so the size is checked here rather
     * than discovered at restore time. */
    const long wrote = ftell(f);
    const long want = (long)(136 + blob_bytes(&h, total_pools,
                                              (uint64_t)s->blob_len));
    fclose(f);
    if (wrote >= 0 && want > 0 && wrote != want) {
        unlink(tmp);
        cwick_error_set(err, CWICK_ERR_IO,
                       "pfx: short write (%ld of %ld bytes)", wrote, want);
        return CWICK_ERR_IO;
    }
    if (rename(tmp, path) != 0) {
        unlink(tmp);
        cwick_error_set(err, CWICK_ERR_IO, "pfx: rename failed (%s)", strerror(errno));
        return CWICK_ERR_IO;
    }

    struct stat st;
    uint64_t bytes = 0;
    if (stat(path, &st) == 0) bytes = (uint64_t)st.st_size;
    if (!index_add(c, h1, h2, P, bytes, path)) {
        cwick_error_set(err, CWICK_ERR_MEMORY, "pfx: index full");
        return CWICK_ERR_MEMORY;
    }
    evict_to_fit(c);
    return CWICK_OK;
}
