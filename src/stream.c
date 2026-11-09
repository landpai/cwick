/* stream.c — memory-bounded SSD streaming for checkpoints larger than RAM.
 *
 * The released zai-org/GLM-5.3-Flash checkpoint is 306 GiB across 62
 * safetensors shards. On a laptop with far less RAM than that, the runtime
 * must never let the kernel decide how much of the checkpoint becomes
 * resident: mmap'ing the shards and touching expert bytes on demand lets the
 * unified page cache grow until the machine is in swap, which on macOS means
 * the whole process stalls.
 *
 * This module owns the resident-spine + on-demand-expert layout, the
 * next-layer prefetch and the three explicit controls:
 *
 *   1. `F_NOCACHE` read descriptors. Experts are read with pread() on a file
 *      descriptor whose buffer cache is disabled, so streamed expert bytes
 *      are never retained by the kernel. Resident spine tensors keep using
 *      the shared mmap (they are wanted in RAM and are warm-wired at boot).
 *   2. A fixed staging arena. Every streamed read lands in one reusable
 *      buffer, so RSS is bounded by construction rather than by luck.
 *   3. An optional byte-capped expert cache. Off by default; when enabled it
 *      is a plain LRU over raw expert blobs whose total size is a hard cap.
 *
 * Nothing here changes what is computed. The same bytes reach the same
 * kernels; only their residency is controlled.
 */
#include "cwick.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

#include "log.h"
#include "mem.h"
#include "util.h"

#define CWICK_STREAM_CHUNK (1u << 20) /* 1 MiB read granularity */

typedef struct cwick_pool_chunk {
    uint8_t *base;
    size_t len;
} cwick_pool_chunk;

struct cwick_stream {
    const cwick_safetensors *saf;
    int *fds;              /* one O_RDONLY (+F_NOCACHE) fd per shard */
    size_t fd_count;
    size_t arena_bytes;
    uint8_t *arena;
    bool nocache;

    /* byte-capped sticky heat pin set over raw expert blobs (opt-in) */
    bool cache_on;
    size_t cache_cap;
    size_t cache_used;
    /* Steering cooldown (see cwick_stream_steer_memory). Freeing bytes does
     * not shrink phys_footprint in the same instant, so the valve trims once
     * and then goes quiet for a while instead of trimming again on the next
     * step against a number that has not caught up yet. */
    uint64_t steer_cooldown_until_ms;
    struct cwick_stream_entry **slots;
    size_t slot_count;
    size_t cursor;      /* rotating eviction cursor */
    uint32_t pass;      /* decay clock: bumped once per decode step */

    /* Fixed-size blob arena (see pool_init). A routed-expert matrix is the
     * same size for every expert, so the cache's unit of allocation is a
     * constant and it can hand blocks out of one mapping instead of
     * allocating ~600 times per decode step. Grown a chunk at a time, because
     * the Metal arena that backs a GLM or DeepSeek expert blob tops out at
     * 2 GiB per buffer. */
    uint8_t *pool;        /* tail chunk: the one still being filled */
    size_t pool_chunk;    /* its capacity */
    size_t pool_bytes;    /* total mapped, for the budget */
    size_t blk_bytes;
    size_t pool_used;
    void *pool_free;      /* free-list head, threaded through the blocks */
    bool pool_off;        /* arena declined: fall back to per-blob allocation */
    bool pool_enable;     /* set by the backend that wants it (see mm_model) */
    struct cwick_pool_chunk *chunks;
    size_t chunk_count, chunk_cap;
    uint32_t heat_avg;  /* admission baseline: newcomers start mid-pack */

    /* ghost admission filter: a small parallel metadata table that counts how
     * many times each (layer, expert, matrix) was requested but NOT resident,
     * without allocating the 8 MiB blob. A key is only admitted into the real
     * cache after GHOST_THRESHOLD misses, so one-shot experts from prefill
     * never displace the experts decode actually reuses. */
    uint8_t *ghost;       /* [ghost_slots] miss counters (saturate) */
    size_t ghost_slots;
    size_t ghost_cursor;  /* rotating eviction cursor for ghost slots */
    uint32_t ghost_epoch; /* decay clock shared with pass */

    /* counters */
    uint64_t bytes_streamed;
    uint64_t read_calls;
    uint64_t cache_hits;
    uint64_t cache_misses;
    pthread_mutex_t stats_mtx; /* shard preads may run concurrently */

    /* Guards the hash table itself: slots[], heat/last_use, cache_used, ghost.
     * With one generation slot at a time the only concurrency here was the
     * reader pool, which touches none of it -- so it went unprotected and
     * that was fine. With several generation slots sharing this stream,
     * two of them can probe-and-insert the same key at once: the second
     * would see the first's half-initialised entry (blob NULL, bytes not yet
     * accounted) or free a blob the other is about to GEMV. Uncontended this
     * costs ~20 ns against ~1000 lookups per step, i.e. ~20 us of a ~590 ms
     * step. Every internal helper (probe_slot, slot_release, cache_make_room,
     * ghost_clear) is called from inside a locked public entry point, so the
     * mutex is never taken recursively. */
    pthread_mutex_t cache_mtx;

    /* optional prefetch reader (see below) */
    struct cwick_pf *pf;
};

typedef struct cwick_stream_entry {
    int64_t key;        /* layer * 1e6 + expert * 10 + matrix */
    uint8_t *blob;
    size_t bytes;
    uint32_t heat;      /* sticky heat: decays toward 0, promotes on hit */
    uint32_t last_use;  /* pass clock for decay bookkeeping */
    bool from_pool;     /* blob is a block of cwick_stream.pool, not its own */
    /* Reserved but not yet filled in. A reserve hands the caller a writable
     * buffer and the read happens afterwards (on the pool, for the whole
     * layer's misses at once), so for that window the slot holds nothing
     * valid and nothing may take its memory away.
     *
     * Before this flag existed that is exactly what happened: reserve ran a
     * loop over a layer's misses, each insert called cache_make_room, and a
     * slot reserved a few iterations earlier could be evicted before its own
     * read was ever dispatched. The caller's `dst` pointer then belonged to a
     * released block, the next admission handed the same block out again, and
     * two reads landed in overlapping memory. The model still ran -- the bytes
     * were mostly right -- but which one won depended on thread scheduling, so
     * two identical runs produced different answers. */
    bool pending;
} cwick_stream_entry;

/* Heat model — frequency with aging (LFU), not recency.
 *
 * Why not recency: measured on the real checkpoint, consecutive decode steps
 * reuse only ~24% of each layer's top-8, so "most recently used" is a poor
 * predictor of what comes next. But the routing distribution is heavily
 * skewed — per layer, the 45 most popular experts carry 82% of the traffic
 * and the top 20 alone carry 60% — so *frequency* is a very good predictor.
 * A recency-ordered cache measured 53% hit where frequency ordering reaches
 * ~82% at the same byte budget; that gap is 4.3 GiB of SSD traffic per
 * decode step.
 *
 *   - heat is an access counter, not a timestamp;
 *   - every CWICK_HEAT_AGE passes all counters halve, so an expert that stops
 *     being routed decays away and the pinned set can follow a topic shift;
 *   - a fresh entry starts at CWICK_HEAT_START and has to earn its place;
 *   - eviction takes the coldest of CWICK_HEAT_SAMPLES random slots. A full
 *     scan of a 16k-slot table per miss is real time spent not reading. */
#define CWICK_HEAT_START   1
#define CWICK_HEAT_QUANT   1
#define CWICK_HEAT_AGE     16      /* passes between halvings */
#define CWICK_HEAT_MAX     1000000
#define CWICK_HEAT_SAMPLES 64      /* power-of-K victim sampling */

/* forward: sizing is decided by the byte budget at open time */
static void cache_resize(cwick_stream *s, size_t cap_bytes);

/* ------------------------------------------------------------------ */
/* Host memory                                                         */
/*                                                                     */
/* cwick_host_memory_bytes() now lives in mem.c alongside the budget     */
/* planner that consumes it.                                            */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* Open / close                                                        */
/* ------------------------------------------------------------------ */
cwick_status cwick_stream_open(const cwick_safetensors *saf,
                             const cwick_stream_options *opt,
                             cwick_stream **out, cwick_error *err) {
    cwick_error_clear(err);
    if (!saf || !out) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "stream: bad argument");
        return CWICK_ERR_ARGUMENT;
    }
    cwick_stream *s = cwick_xmalloc(sizeof(*s));
    memset(s, 0, sizeof(*s));
    pthread_mutex_init(&s->stats_mtx, NULL);
    pthread_mutex_init(&s->cache_mtx, NULL);
    s->saf = saf;
    s->arena_bytes = (opt && opt->arena_bytes) ? opt->arena_bytes : (32u << 20);
    s->nocache = !opt || opt->nocache;
    if (opt && opt->expert_cache_bytes > 0) {
        s->cache_on = true;
        s->cache_cap = opt->expert_cache_bytes;
        cache_resize(s, s->cache_cap);
    }
    s->arena = cwick_xmalloc(s->arena_bytes);

    /* One descriptor per shard. F_NOCACHE keeps streamed bytes out of the
     * unified buffer cache; F_RDAHEAD still lets the device read ahead. */
    s->fd_count = saf->shard_count;
    s->fds = cwick_xmalloc(s->fd_count * sizeof(int));
    for (size_t i = 0; i < s->fd_count; i++) s->fds[i] = -1;
    for (size_t i = 0; i < s->fd_count; i++) {
        char path[1152];
        cwick_path_join(path, sizeof(path), saf->root, saf->shard_names[i]);
        int fd = open(path, O_RDONLY);
        if (fd < 0) {
            cwick_error_set(err, CWICK_ERR_IO, "stream: cannot open shard %s (%s)",
                           saf->shard_names[i], strerror(errno));
            cwick_stream_close(s);
            return CWICK_ERR_IO;
        }
        if (s->nocache) {
            if (fcntl(fd, F_NOCACHE, 1) < 0) {
                int saved = errno;
                close(fd);
                s->fds[i] = -1;
                cwick_error_set(err, CWICK_ERR_IO,
                               "stream: bounded F_NOCACHE reads unavailable on %s (%s)",
                               saf->shard_names[i], strerror(saved));
                cwick_stream_close(s);
                return CWICK_ERR_IO;
            }
            (void)fcntl(fd, F_RDAHEAD, 1);
        }
        s->fds[i] = fd;
    }
    *out = s;
    return CWICK_OK;
}

void cwick_stream_close(cwick_stream *s) {
    if (!s) return;
    for (size_t i = 0; i < s->fd_count; i++)
        if (s->fds[i] >= 0) close(s->fds[i]);
    free(s->fds);
    if (s->slots) {
        for (size_t i = 0; i < s->slot_count; i++) {
            if (s->slots[i]) {
                /* blobs come from cwick_shmem_alloc (a GPU-visible arena when
                 * the Metal backend is up), so they must go back through
                 * cwick_shmem_free -- free() on an arena pointer corrupts the
                 * heap. Blocks of the blob arena are not freed one by one: the
                 * mapping goes back whole below. */
                if (!s->slots[i]->from_pool) cwick_shmem_free(s->slots[i]->blob);
                free(s->slots[i]);
            }
        }
        free(s->slots);
    }
    free(s->ghost);
    free(s->arena);
    for (size_t i = 0; i < s->chunk_count; i++) cwick_shmem_free(s->chunks[i].base);
    free(s->chunks);
    pthread_mutex_destroy(&s->stats_mtx);
    pthread_mutex_destroy(&s->cache_mtx);
    free(s);
}

/* Drop a mapped range from core without unmapping it (MADV_DONTNEED). Used
 * after a bulk verification pass so the pages are reclaimed immediately
 * instead of waiting for memory pressure. */
void cwick_stream_reclaim(void *addr, size_t len) {
    if (!addr || len == 0) return;
#if defined(MADV_DONTNEED)
    long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0) return;
    uintptr_t start = (uintptr_t)addr & ~(uintptr_t)(ps - 1);
    uintptr_t end = ((uintptr_t)addr + len + (uintptr_t)ps - 1) &
                    ~(uintptr_t)(ps - 1);
    if (end > start) madvise((void *)start, (size_t)(end - start), MADV_DONTNEED);
#endif
}

/* Warm a mapped range (MADV_WILLNEED) — the resident spine. */
void cwick_stream_warm(const void *addr, size_t len) {
    if (!addr || len == 0) return;
#if defined(MADV_WILLNEED)
    long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0) return;
    uintptr_t start = (uintptr_t)addr & ~(uintptr_t)(ps - 1);
    uintptr_t end = ((uintptr_t)addr + len + (uintptr_t)ps - 1) &
                    ~(uintptr_t)(ps - 1);
    if (end > start)
        madvise((void *)start, (size_t)(end - start), MADV_WILLNEED);
#endif
}

/* ------------------------------------------------------------------ */
/* Low-level reads                                                     */
/* ------------------------------------------------------------------ */
/* Read `n` bytes at `off` of shard `si` into `dst`, looping over short
 * reads. Returns the number of bytes read. */
static size_t shard_pread(cwick_stream *s, size_t si, void *dst, size_t n,
                          off_t off, cwick_error *err) {
    size_t done = 0;
    while (done < n) {
        ssize_t r = pread(s->fds[si], (uint8_t *)dst + done, n - done, off + (off_t)done);
        if (r < 0) {
            if (errno == EINTR) continue;
            cwick_error_set(err, CWICK_ERR_IO, "stream: read failed on %s (%s)",
                           s->saf->shard_names[si], strerror(errno));
            return done;
        }
        if (r == 0) break; /* truncated file */
        done += (size_t)r;
    }
    pthread_mutex_lock(&s->stats_mtx);
    s->bytes_streamed += done;
    s->read_calls++;
    pthread_mutex_unlock(&s->stats_mtx);
    return done;
}

/* Resolve the shard index and byte offset of a tensor's payload.
 * `cwick_st_tensor.data` is already a pointer inside the shard's mapping, so
 * the offset is recovered from the shard base recorded by safetensors.c. */
cwick_status cwick_stream_locate(const cwick_stream *s, const cwick_st_tensor *t,
                               size_t *shard_index, uint64_t *offset,
                               cwick_error *err) {
    if (!s || !t) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "stream: bad tensor");
        return CWICK_ERR_ARGUMENT;
    }
    if (t->shard_index < 0 || (size_t)t->shard_index >= s->fd_count) {
        cwick_error_set(err, CWICK_ERR_IO, "stream: tensor %s has no shard mapping",
                       t->name);
        return CWICK_ERR_IO;
    }
    *shard_index = (size_t)t->shard_index;
    *offset = t->data_offset;
    return CWICK_OK;
}

/* Read a whole tensor into `dst` (which must hold data_bytes). This is the
 * streaming path: bytes go through the F_NOCACHE descriptor into the caller's
 * buffer, never into the page cache. */
cwick_status cwick_stream_read(cwick_stream *s, const cwick_st_tensor *t, void *dst,
                             cwick_error *err) {
    size_t si;
    uint64_t off;
    cwick_status st = cwick_stream_locate(s, t, &si, &off, err);
    if (st != CWICK_OK) return st;
    if (!dst) return CWICK_OK;
    size_t n = t->data_bytes;
    size_t got = shard_pread(s, si, dst, n, (off_t)off, err);
    if (got != n) {
        cwick_error_set(err, CWICK_ERR_IO,
                       "stream: tensor %s truncated (%zu of %zu bytes)",
                       t->name, got, n);
        return CWICK_ERR_IO;
    }
    return CWICK_OK;
}

/* ------------------------------------------------------------------ */
/* Verification (doctor)                                               */
/* ------------------------------------------------------------------ */
/* Read every byte of a tensor through the streaming path and fold it into
 * `acc`. Memory use is O(arena_bytes) regardless of tensor size. This is what
 * turns `cwick doctor` from a name-lookup into a real load. */
cwick_status cwick_stream_verify(cwick_stream *s, const cwick_st_tensor *t,
                               uint64_t *acc, cwick_error *err) {
    size_t si;
    uint64_t base;
    cwick_status st = cwick_stream_locate(s, t, &si, &base, err);
    if (st != CWICK_OK) return st;
    off_t off = (off_t)base;

    /* dtype sanity: the F32 scale side must have the block geometry the
     * decoder assumes, otherwise the dequantized values are silently wrong. */
    if (t->dtype == CWICK_ST_F8_E4M3) {
        if (!t->scale_inv || t->scale_inv_rows <= 0 || t->scale_inv_cols <= 0) {
            cwick_error_set(err, CWICK_ERR_CONTRACT,
                           "tensor %s is F8_E4M3 but has no paired *_scale_inv",
                           t->name);
            return CWICK_ERR_CONTRACT;
        }
        int64_t want_r = (t->rows + CWICK_ST_BLOCK - 1) / CWICK_ST_BLOCK;
        int64_t want_c = (t->cols + CWICK_ST_BLOCK - 1) / CWICK_ST_BLOCK;
        if (t->scale_inv_rows != want_r || t->scale_inv_cols != want_c) {
            cwick_error_set(err, CWICK_ERR_CONTRACT,
                           "tensor %s scale_inv is [%lld,%lld], expected [%lld,%lld]",
                           t->name, (long long)t->scale_inv_rows,
                           (long long)t->scale_inv_cols, (long long)want_r,
                           (long long)want_c);
            return CWICK_ERR_CONTRACT;
        }
    }
    if (t->rows <= 0 || t->cols <= 0 || t->data_bytes == 0) {
        cwick_error_set(err, CWICK_ERR_CONTRACT, "tensor %s has an empty shape",
                       t->name);
        return CWICK_ERR_CONTRACT;
    }

    uint64_t local = 1469598103934665603ull; /* FNV-1a 64 */
    size_t remaining = t->data_bytes;
    off_t cur = off;
    while (remaining > 0) {
        size_t chunk = remaining < s->arena_bytes ? remaining : s->arena_bytes;
        size_t got = shard_pread(s, si, s->arena, chunk, cur, err);
        if (got != chunk) {
            cwick_error_set(err, CWICK_ERR_IO,
                           "tensor %s unreadable at +%lld (%zu of %zu bytes)",
                           t->name, (long long)(cur - off), got, chunk);
            return CWICK_ERR_IO;
        }
        /* Fold 8 bytes at a time. A byte-wise FNV over 300 GiB costs ~3
         * cycles per byte and would dominate the whole pass; this keeps the
         * verification I/O-bound instead of hash-bound. */
        size_t words = chunk / 8;
        const uint64_t *wp = (const uint64_t *)(void *)s->arena;
        for (size_t i = 0; i < words; i++) {
            uint64_t v = wp[i] * 0x100000001b3ull;
            v ^= v >> 29;
            local ^= v;
            local *= 0x9e3779b97f4a7c15ull;
        }
        for (size_t i = words * 8; i < chunk; i++) {
            local ^= s->arena[i];
            local *= 1099511628211ull;
        }
        remaining -= chunk;
        cur += (off_t)chunk;
    }
    if (acc) *acc ^= local;
    return CWICK_OK;
}

/* ------------------------------------------------------------------ */
/* Byte-capped sticky expert heat cache                                */
/*                                                                     */
/* Referencing the measured route-hotness distribution: hot expert      */
/* blobs are pinned in RAM up to a byte budget, everything else is     */
/* re-read from SSD every decode step.                                 */
uint64_t g_res_probe_ns;   /* prof: probe_slot    */
uint64_t g_res_evict_ns;   /* prof: cache_make_room */
uint64_t g_res_slot_ns;    /* prof: entry malloc  */
uint64_t g_res_shmem_ns;   /* prof: cwick_shmem_alloc */
uint64_t g_res_calls;      /* prof: reserve calls */
/*                                                                     */
/* Two things were wrong with a direct-mapped 1024-slot table once the */
/* budget grew to tens of GiB:                                         */
/*   1. 42 sparse layers x 288 experts x 3 matrices is ~36k blobs; with */
/*      1024 buckets every insert evicted a live neighbour, so the byte */
/*      cap could never be reached and the "hot set" was really         */
/*      "whatever collided last". The table now sizes itself to the     */
/*      budget (capacity / expected blob size, power of two, capped)    */
/*      and probes a short window, so occupancy tracks the budget.      */
/*   2. Eviction scanned the whole table per insert (O(slots) x ~1000   */
/*      reads per token). Eviction now picks the coldest entry inside   */
/*      the probe window, which is O(window) and keeps hot neighbours.  */
/* ------------------------------------------------------------------ */
#define CWICK_CACHE_PROBES 8       /* linear probe window */
#define CWICK_SLOTS_MIN    1024
#define CWICK_SLOTS_MAX    262144
#define CWICK_EXPECT_BLOB  (8u << 20)   /* one FP8 expert matrix, ~8 MiB */

/* Ghost admission: a key must be missed this many times (over separate
 * requests) before it earns a real 8 MiB cache slot. This is the filter that
 * keeps prefill's one-shot experts and cold routes out of the pinned set. */
#define CWICK_GHOST_THRESHOLD 2
/* Ghost table: one 8-bit counter per slot, power-of-two sized like the main
 * table. Far smaller than the blob cache it guards. */
#define CWICK_GHOST_SLOTS    262144

static size_t slot_for(const cwick_stream *s, int64_t key) {
    return (size_t)(((uint64_t)key * 2654435761ull) & (uint64_t)(s->slot_count - 1));
}

/* ------------------------------------------------------------------ */
/* Ghost admission filter                                              */
/*                                                                     */
/* Every cache miss (a key requested but not resident) bumps a byte     */
/* counter in the ghost table. A key whose counter crosses              */
/* CWICK_GHOST_THRESHOLD is then allowed into the real cache; the blob   */
/* allocation path checks ghost_pass() before committing bytes.         */
/*                                                                     */
/* Each slot stores {key, count} so counters are attributable, at 16    */
/* bytes per key. The table is open-addressed like the main cache but   */
/* with 16-byte slots instead of 8 MiB blobs, so it holds far more      */
/* keys than the blob cache at a tiny fraction of the bytes. Eviction   */
/* is the same power-of-K sampling over cheap metadata.                 */
/* ------------------------------------------------------------------ */
#define CWICK_GHOST_PROBES 16

typedef struct {
    int64_t key;
    uint8_t count;
    uint8_t _pad[7];
} cwick_ghost_slot;

static size_t ghost_home(const cwick_stream *s, int64_t key) {
    return (size_t)(((uint64_t)key * 2654435761ull) &
                    (uint64_t)(s->ghost_slots - 1));
}

static cwick_ghost_slot *ghost_at(cwick_stream *s, size_t i) {
    return &((cwick_ghost_slot *)s->ghost)[i];
}

/* Ghost admission deliberately costs one extra miss per cold key before that
 * key is allowed into the cache: first miss records the key, second miss
 * admits it. The bet was that this keeps prefill's one-shot experts from
 * evicting genuinely hot ones.
 *
 * It lost, and measurably. A/B on the real 320B checkpoint, 16 GiB cache,
 * 15 decode steps each:
 *
 *   ghost on   1098 ms/step   0.910 tok/s   hit 32.2%   RSS 81.62 GiB
 *   ghost off   593 ms/step   1.687 tok/s   hit 36.5%   RSS 32.89 GiB
 *
 * Both halves of the bet failed at once. The hit rate went *down*, so the
 * filter was not keeping cold experts out -- with 1080 matrices touched per
 * step and a 16 GiB cache, nearly every key is cold often enough to clear a
 * threshold of 2, so the filter admitted almost everything while still making
 * each of them pay an extra miss. And refusing a slot sends the caller down
 * the owned-malloc fallback, which inflated the process from 32.89 to
 * 81.62 GiB and put the machine under memory pressure -- enough on its own to
 * explain the slowdown.
 *
 * So it is off by default. CWICK_GHOST_ADMIT=1 turns it back on for anyone who
 * wants to retry it against a different cache size or workload. */
static int g_ghost_admit = -1;   /* -1 = not read yet */

static bool ghost_admit_enabled(void) {
    if (g_ghost_admit < 0) {
        const char *e = getenv("CWICK_GHOST_ADMIT");
        g_ghost_admit = (e && *e && atoi(e) != 0) ? 1 : 0;
    }
    return g_ghost_admit != 0;
}

/* Record one miss of `key` in the ghost table. Returns 1 if the key has
 * now crossed the admission threshold (so the caller should admit it into
 * the real cache), 0 otherwise. The counter saturates at 255. */
static int ghost_touch(cwick_stream *s, int64_t key) {
    if (!s->ghost) return 1;   /* no ghost table: admit unconditionally */
    if (!ghost_admit_enabled()) return 1;
    size_t base = ghost_home(s, key);
    size_t empty = SIZE_MAX, victim = SIZE_MAX;
    uint8_t coldest = 255;
    for (size_t p = 0; p < CWICK_GHOST_PROBES; p++) {
        size_t i = (base + p) & (s->ghost_slots - 1);
        cwick_ghost_slot *g = ghost_at(s, i);
        if (g->key == 0) { empty = i; break; }     /* empty slot */
        if (g->key == key) {                       /* same key: bump */
            if (g->count == 255) return 1;
            uint8_t c = ++g->count;
            return c >= CWICK_GHOST_THRESHOLD;
        }
        if (g->count < coldest) { coldest = g->count; victim = i; }
    }
    /* probe window full or empty slot found: place / evict */
    if (empty != SIZE_MAX) {
        cwick_ghost_slot *g = ghost_at(s, empty);
        g->key = key; g->count = 1;
        return 0;
    }
    if (victim == SIZE_MAX) victim = base;
    cwick_ghost_slot *g = ghost_at(s, victim);
    g->key = key; g->count = 1;
    return 0;
}

/* Clear the ghost counter for `key` once it has been admitted, so the
 * table does not keep promoting a key that is already resident. */
static void ghost_clear(cwick_stream *s, int64_t key) {
    if (!s->ghost) return;
    size_t base = ghost_home(s, key);
    for (size_t p = 0; p < CWICK_GHOST_PROBES; p++) {
        size_t i = (base + p) & (s->ghost_slots - 1);
        cwick_ghost_slot *g = ghost_at(s, i);
        if (g->key == 0) return;
        if (g->key == key) { g->key = 0; g->count = 0; return; }
    }
}

/* Public ghost admission probe. Records this miss and reports whether the
 * key has earned a pinned slot. Callers that hit a non-resident key in the
 * reserve/put path call this once per miss; the first GHOST_THRESHOLD-1
 * misses return 0 (read transiently, don't pin) and the next one returns 1
 * (allocate the blob). A key already resident never reaches this probe. */
int cwick_stream_expert_should_pin(cwick_stream *s, int64_t layer,
                                  int64_t expert, int64_t matrix) {
    if (!s || !s->cache_on) return 0;
    int64_t key = layer * 1000000 + expert * 10 + matrix;
    return ghost_touch(s, key);
}

/* Decay the ghost table: halve every counter so admission follows a topic
 * shift the way the main heat cache does. Runs from cwick_stream_expert_decay,
 * which is once per decode step (cheap: 262144 x 16 bytes, ~microseconds). */
static void ghost_decay(cwick_stream *s) {
    if (!s->ghost) return;
    size_t n = s->ghost_slots;
    cwick_ghost_slot *t = (cwick_ghost_slot *)s->ghost;
    for (size_t k = 0; k < n; k++) {
        if (t[k].key == 0) continue;
        uint8_t v = t[k].count;
        t[k].count = (uint8_t)((v >> 1) + (v & 1)); /* round half up */
        if (t[k].count <= 1) { t[k].key = 0; t[k].count = 0; } /* cooled to nothing */
    }
}

/* Size the open-addressed table so 60% occupancy can hold `bytes` worth of
 * blobs. Called from cwick_stream_open once the budget is known. */
static void cache_resize(cwick_stream *s, size_t cap_bytes) {
    size_t want = cap_bytes / CWICK_EXPECT_BLOB;
    want += want / 2;                     /* keep load factor <= ~0.66 */
    size_t slots = CWICK_SLOTS_MIN;
    while (slots < want && slots < CWICK_SLOTS_MAX) slots <<= 1;
    s->slot_count = slots;
    free(s->slots);
    s->slots = cwick_xmalloc(slots * sizeof(*s->slots));
    memset(s->slots, 0, slots * sizeof(*s->slots));
    s->cache_used = 0;
    s->heat_avg = CWICK_HEAT_START;

    /* Ghost table is a fixed-size open-addressed metadata table (16 bytes
     * per slot) that does not need to size with the budget because it stores
     * {key,count}, not the 8 MiB blob. */
    s->ghost_slots = CWICK_GHOST_SLOTS;
    free(s->ghost);
    s->ghost = cwick_xmalloc(s->ghost_slots * sizeof(cwick_ghost_slot));
    memset(s->ghost, 0, s->ghost_slots * sizeof(cwick_ghost_slot));
    s->ghost_cursor = 0;
}

static size_t slot_home(const cwick_stream *s, int64_t key) {
    return (size_t)(((uint64_t)key * 2654435761ull) &
                    (uint64_t)(s->slot_count - 1));
}

/* ------------------------------------------------------------------ */
/* Blob arena                                                          */
/* ------------------------------------------------------------------ */

/* Every routed-expert matrix of a given checkpoint has the same shape, so the
 * pin cache allocates one constant-size block at a time: [inter, hidden] for
 * gate and up, [hidden, inter] for down. That makes a fixed-size arena
 * possible, and two costs go away with it -- both of which scale with the miss
 * count rather than with the arithmetic:
 *
 *   1. cwick_shmem_alloc + free per blob, ~600 pairs per decode step. Through
 *      the Metal arena that is a lock and a pool search; through the fallback
 *      it is malloc/munmap of 10 MiB, which returns the pages to the kernel.
 *   2. The re-fault that follows. A munmap'd block comes back as fresh zero
 *      pages, so the next reserve pays for its pages a second time before a
 *      single byte of the read lands.
 *
 * Measured on MiniMax-M3.1-Flash (48 GiB cap, same prompt): GEMV 400-780 ->
 * 300-410 ms/step, and the footprint at which the compressor starts to take
 * pages moved out with the reuse instead of staying put. CWICK_NO_SLAB=1 keeps
 * the old per-blob path for an A/B. */
/* One chunk of the arena. The mapping is grown a chunk at a time rather than
 * asked for in one piece: the Metal shared-memory arena that backs a GLM or
 * DeepSeek expert blob tops out at CWICK_ARENA_MAX (2 GiB) per buffer, so a
 * single 21 GiB request would fail there and quietly push every expert back to
 * a copied path. Chunks of 1 GiB stay inside that ceiling and still cut the
 * allocation count from ~600 per step to a handful per process. */
#define POOL_CHUNK_BYTES (1ull << 30)

void cwick_stream_enable_blob_pool(cwick_stream *s) {
    if (s) s->pool_enable = true;
}

static void pool_init(cwick_stream *s, size_t bytes) {
    if (s->blk_bytes || s->pool_off || !s->cache_on) return;
    s->pool_off = true;                       /* until it is proven otherwise */
    if (!s->pool_enable) return;
    if (bytes < sizeof(void *) || (bytes & 15u) != 0) return;
    const char *e = getenv("CWICK_NO_SLAB");
    if (e && *e && strcmp(e, "0") != 0) return;
    /* Not when the blobs are GPU-visible. The Metal arena has an allocator
     * built for 8 MiB expert matrices and grows (and wires) a buffer at a
     * time; handing it 1 GiB blocks instead changes that growth curve, and
     * measured on GLM-5.3-Flash it costs 17% of decode (2.09 -> 1.74 tok/s on
     * the same 7 steps). The churn this arena exists to remove is the
     * malloc/munmap of the plain-memory path, so only that path gets it. */
    if (cwick_shmem_gpu_visible()) return;
    if (s->cache_cap / bytes < 8) return;     /* too small to be worth it */
    s->blk_bytes = bytes;
    s->pool_free = NULL;
    s->pool_off = false;
}

/* A block for one expert matrix, or NULL to let the caller allocate its own. */
static uint8_t *pool_alloc(cwick_stream *s, size_t bytes) {
    if (!s->blk_bytes || bytes != s->blk_bytes) return NULL;
    void *b = s->pool_free;
    if (b) {
        s->pool_free = *(void **)b;
        return (uint8_t *)b;
    }
    if (s->pool_used + bytes <= s->pool_chunk) {
        uint8_t *p = s->pool + s->pool_used;
        s->pool_used += bytes;
        return p;
    }
    if (s->pool_bytes >= s->cache_cap) return NULL;
    size_t want = POOL_CHUNK_BYTES;
    if (want > s->cache_cap - s->pool_bytes) want = s->cache_cap - s->pool_bytes;
    want -= want % bytes;                     /* whole blocks only */
    if (want < bytes) return NULL;
    uint8_t *p = cwick_shmem_alloc(want);     /* arena-backed when Metal is on */
    if (!p) {
        s->pool_off = true;
        return NULL;
    }
    if (s->chunk_count == s->chunk_cap) {
        size_t nc = s->chunk_cap ? s->chunk_cap * 2 : 8;
        cwick_pool_chunk *v = realloc(s->chunks, nc * sizeof(*v));
        if (!v) {
            cwick_shmem_free(p);
            s->pool_off = true;
            return NULL;
        }
        s->chunks = v;
        s->chunk_cap = nc;
    }
    s->chunks[s->chunk_count].base = p;
    s->chunks[s->chunk_count].len = want;
    s->chunk_count++;
    s->pool = p;
    s->pool_chunk = want;
    s->pool_bytes += want;
    s->pool_used = bytes;
    return p;
}

/* Hand a block back. Only blocks this arena handed out get here -- the entry
 * remembers which allocator answered -- so there is nothing to range-check. */
static void pool_release(cwick_stream *s, uint8_t *b) {
    if (!b) return;
    *(void **)b = s->pool_free;
    s->pool_free = b;
}

static size_t slot_release(cwick_stream *s, size_t i) {
    if (!s->slots[i]) return i;
    s->cache_used -= s->slots[i]->bytes;
    if (s->slots[i]->from_pool) pool_release(s, s->slots[i]->blob);
    else cwick_shmem_free(s->slots[i]->blob);
    free(s->slots[i]);
    s->slots[i] = NULL;

    /* Linear probing cannot leave a tombstone: a later colliding key would
     * become unreachable at the first empty slot. Backward-shift the cluster
     * while each entry's home is on the far side of the new hole. The bytes
     * stay accounted for; only their slots move. */
    size_t hole = i;
    size_t pos = (i + 1) & (s->slot_count - 1);
    for (size_t scanned = 0; scanned < s->slot_count; scanned++) {
        cwick_stream_entry *e = s->slots[pos];
        if (!e) break;
        size_t home = slot_home(s, e->key);
        size_t mask = s->slot_count - 1;
        if (((pos - home) & mask) >= ((hole - home) & mask)) {
            s->slots[hole] = e;
            s->slots[pos] = NULL;
            hole = pos;
        }
        pos = (pos + 1) & mask;
    }
    return hole;
}

/* Look up a cached raw expert-matrix blob. Returns NULL on a miss.
 * `matrix` selects one of an expert's three matrices (0=gate,1=up,2=down). */
const void *cwick_stream_expert_cached(cwick_stream *s, int64_t layer,
                                      int64_t expert, int64_t matrix,
                                      size_t *bytes) {
    if (!s || !s->cache_on) return NULL;
    int64_t key = layer * 1000000 + expert * 10 + matrix;
    pthread_mutex_lock(&s->cache_mtx);
    size_t base = slot_for(s, key);
    for (size_t p = 0; p < CWICK_CACHE_PROBES; p++) {
        size_t i = (base + p) & (s->slot_count - 1);
        cwick_stream_entry *e = s->slots[i];
        if (!e) break;                       /* empty probe: definitively absent */
        if (e->key != key) continue;
        /* Frequency promotion: routing is skewed, not recent-heavy, so the
         * counter (not the clock) is what earns an expert its pin. */
        if (e->heat < CWICK_HEAT_MAX) e->heat += CWICK_HEAT_QUANT;
        e->last_use = s->pass;
        s->cache_hits++;
        if (bytes) *bytes = e->bytes;
        const void *hit = e->blob;
        pthread_mutex_unlock(&s->cache_mtx);
        return hit;
    }
    s->cache_misses++;
    pthread_mutex_unlock(&s->cache_mtx);
    return NULL;
}

/* Is this matrix already resident? Same lookup as the hit path but it does
 * NOT touch the hit/miss counters: callers use it to decide whether to spend
 * SSD bandwidth, and counting those probes would flatter the hit rate with
 * reads that never reached the consumer. */
int cwick_stream_expert_resident(cwick_stream *s, int64_t layer, int64_t expert,
                                int64_t matrix) {
    if (!s || !s->cache_on) return 0;
    int64_t key = layer * 1000000 + expert * 10 + matrix;
    size_t base = slot_for(s, key);
    for (size_t p = 0; p < CWICK_CACHE_PROBES; p++) {
        size_t i = (base + p) & (s->slot_count - 1);
        cwick_stream_entry *e = s->slots[i];
        if (!e) return 0;
        if (e->key == key) return 1;
    }
    return 0;
}

/* Find where `key` lives or should go. Returns the slot index and, in
 * `found`, whether it is occupied by that key. When every slot in the probe
 * window is taken by another key, the coldest one becomes the victim. */
/* A slot may only be taken away from a blob nobody is waiting to fill. */
static bool slot_evictable(const cwick_stream_entry *e) {
    return e != NULL && !e->pending;
}

static size_t probe_slot(cwick_stream *s, int64_t key, bool *found) {
    size_t base = slot_for(s, key);
    size_t empty = SIZE_MAX, victim = SIZE_MAX;
    uint32_t coldest = UINT32_MAX;
    for (size_t p = 0; p < CWICK_CACHE_PROBES; p++) {
        size_t i = (base + p) & (s->slot_count - 1);
        cwick_stream_entry *e = s->slots[i];
        if (!e) { empty = i; break; }
        if (e->key == key) { *found = true; return i; }
        if (slot_evictable(e) && e->heat < coldest) {
            coldest = e->heat; victim = i;
        }
    }
    *found = false;
    if (empty != SIZE_MAX) return empty;
    if (victim == SIZE_MAX) victim = slot_for(s, key);
    return slot_release(s, victim);
}

/* Pick an eviction victim by sampling: draw CWICK_HEAT_SAMPLES slots and
 * take the coldest. Sampling instead of a full sweep because this runs once
 * per cache miss (~550 per decode step) and a 16k-slot sweep is ~30 us of
 * pointer chasing per miss — time the SSD could be reading instead.
 * Power-of-K choices is not exact LFU but is close enough that the hit-rate
 * difference does not show up against the I/O it saves. */
static size_t cache_pick_victim(cwick_stream *s) {
    size_t mask = s->slot_count - 1;
    size_t victim = SIZE_MAX;
    uint32_t coldest = UINT32_MAX;
    uint64_t x = (uint64_t)s->cursor * 2654435761ull + 1442695040888963407ull;
    for (int k = 0; k < CWICK_HEAT_SAMPLES; k++) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        size_t i = (size_t)((x >> 33) & (uint64_t)mask);
        cwick_stream_entry *e = s->slots[i];
        if (slot_evictable(e) && e->heat < coldest) {
            coldest = e->heat; victim = i;
        }
    }
    if (victim == SIZE_MAX) {
        /* every sampled slot empty (table still filling): fall back to home */
        victim = (size_t)((x >> 33) & (uint64_t)mask);
    }
    s->cursor = (victim + 1) & mask;
    return victim;
}

/* Make room for `bytes` (budget or external pressure), evicting cold
 * entries anywhere in the table. Used when the probe window cannot absorb
 * the insert and by cwick_stream_trim. */
static void cache_make_room(cwick_stream *s, size_t bytes) {
    int guard = 0;
    while (s->cache_used + bytes > s->cache_cap && guard++ < 4096) {
        size_t victim = cache_pick_victim(s);
        if (victim >= s->slot_count) break;
        if (!s->slots[victim]) {
            /* sampled an empty slot: nothing was freed, find an occupant */
            int found = 0;
            for (size_t k = 0; k < s->slot_count; k++) {
                size_t i = (s->cursor + k) & (s->slot_count - 1);
                if (slot_evictable(s->slots[i])) { victim = i; found = 1; break; }
            }
            if (!found) break;
        }
        slot_release(s, victim);
    }
}

/* Insert a raw expert-matrix blob, evicting the coldest nearby entry until
 * it fits. */
void cwick_stream_expert_put(cwick_stream *s, int64_t layer, int64_t expert,
                            int64_t matrix, const void *blob, size_t bytes) {
    if (!s || !s->cache_on || bytes > s->cache_cap) return;
    int64_t key = layer * 1000000 + expert * 10 + matrix;
    bool found = false;
    size_t i = probe_slot(s, key, &found);
    if (found) {
        /* refresh in place -- arena blocks cannot be realloc'd, so a bigger
         * blob is re-acquired through the shared allocator */
        cwick_stream_entry *e = s->slots[i];
        s->cache_used -= e->bytes;
        if (bytes > e->bytes) {
            cwick_shmem_free(e->blob);
            e->blob = cwick_shmem_alloc(bytes);
            if (!e->blob) { s->slots[i] = NULL; s->cache_used += 0; return; }
        }
        e->bytes = bytes;
        memcpy(e->blob, blob, bytes);
        s->cache_used += bytes;
        return;
    }
    if (s->cache_used + bytes > s->cache_cap) cache_make_room(s, bytes);
    if (s->cache_used + bytes > s->cache_cap) return;
    cwick_stream_entry *e = s->slots[i];
    if (e) slot_release(s, i);
    e = cwick_xmalloc(sizeof(*e));
    e->key = key;
    e->bytes = bytes;
    e->heat = s->heat_avg;   /* admitted mid-pack: must earn a lower or higher place */
    e->last_use = s->pass;
    e->pending = false;      /* memcpy below fills it in: evictable at once */
    e->blob = cwick_shmem_alloc(bytes);
    if (!e->blob) { free(e); return; }
    memcpy(e->blob, blob, bytes);
    s->slots[i] = e;
    s->cache_used += bytes;
}

/* Read tensor `t` (and its FP8 scale side `st`, if any) straight into a
 * cache slot, avoiding the staging -> put double-copy. On a budget-evicting
 * miss this allocates the slot first, then preads into it. Returns the
 * slot's data pointer (matrix bytes, then scale bytes) or NULL if the cache
 * is off / the tensor cannot be cached. */
const void *cwick_stream_expert_put_read(cwick_stream *s, int64_t layer,
                                        int64_t expert, int64_t matrix,
                                        const cwick_st_tensor *t,
                                        const cwick_st_tensor *st,
                                        cwick_error *err) {
    if (!s || !s->cache_on) return NULL;
    size_t tbytes = t->data_bytes;
    size_t sbytes = st && st->data_bytes ? st->data_bytes : 0;
    size_t bytes = tbytes + sbytes;
    if (bytes > s->cache_cap) return NULL;

    int64_t key = layer * 1000000 + expert * 10 + matrix;
    bool found = false;
    size_t i = probe_slot(s, key, &found);
    if (found) {
        /* already resident; nothing to read */
        s->cache_hits++;
        return s->slots[i]->blob;
    }
    /* Ghost admission gate (same rule as reserve): a cold key is read into a
     * transient staging buffer by the caller, not pinned into the budget. */
    if (!cwick_stream_expert_should_pin(s, layer, expert, matrix))
        return NULL;
    if (s->cache_used + bytes > s->cache_cap) cache_make_room(s, bytes);
    if (s->cache_used + bytes > s->cache_cap) {
        pthread_mutex_unlock(&s->cache_mtx);
        return NULL;
    }
    if (s->slots[i]) slot_release(s, i);

    cwick_stream_entry *e = cwick_xmalloc(sizeof(*e));
    e->key = key;
    e->bytes = bytes;
    e->heat = s->heat_avg;   /* admitted mid-pack: must earn a lower or higher place */
    e->last_use = s->pass;
    e->pending = true;       /* the preads below are what makes it valid */
    /* GPU-visible when the Metal backend is up: the SSD read lands in memory
     * the GEMV can bind directly instead of copying per call. */
    e->blob = cwick_shmem_alloc(bytes);
    if (!e->blob) { free(e); pthread_mutex_unlock(&s->cache_mtx); return NULL; }
    s->slots[i] = e;
    s->cache_used += bytes;
    ghost_clear(s, key);

    /* pread the matrix bytes, then the scale bytes, directly into the slot.
     * Roll a partial/failed admission back out of the index; otherwise a
     * transient SSD error would leave a permanently resident, invalid entry
     * that later looks like a valid cache hit. */
    if (cwick_stream_read(s, t, e->blob, err) != CWICK_OK) {
        slot_release(s, i);
        pthread_mutex_unlock(&s->cache_mtx);
        return NULL;
    }
    if (sbytes && cwick_stream_read(s, st, e->blob + tbytes, err) != CWICK_OK) {
        slot_release(s, i);
        pthread_mutex_unlock(&s->cache_mtx);
        return NULL;
    }
    e->pending = false;      /* filled: it is cache data now, evictable */
    pthread_mutex_unlock(&s->cache_mtx);
    return e->blob;
}

/* Reserve a writable cache slot for `bytes` without reading anything.
 *
 * This exists so the caller can allocate every slot a layer needs on one
 * thread (the hash table is not thread-safe) and then dispatch the actual
 * preads concurrently. Measured on the 62-shard checkpoint: one 8 MiB read
 * at a time sustains 6.66 GB/s, four to eight in flight sustain 9.16 GB/s.
 * Since a decode step is dominated by exactly these reads, that is a third
 * off the largest phase in the profile.
 *
 * Returns NULL when the budget will not admit the blob (caller falls back to
 * a synchronous staging read). The slot is indexed straight away, so its
 * contents must be filled before anyone looks the key up — and on a failed
 * read the caller must call cwick_stream_expert_abort so a half-written blob
 * never masquerades as a cache hit. */
uint8_t *cwick_stream_expert_reserve(cwick_stream *s, int64_t layer,
                                    int64_t expert, int64_t matrix,
                                    size_t bytes) {
    int64_t key = layer * 1000000 + expert * 10 + matrix;
    bool found = false;
    uint64_t _t0 = cwick_prof_ns();
    pthread_mutex_lock(&s->cache_mtx);
    size_t i = probe_slot(s, key, &found);
    g_res_probe_ns += cwick_prof_ns() - _t0;
    if (found) {                            /* already resident */
        uint8_t *b = s->slots[i]->blob;
        pthread_mutex_unlock(&s->cache_mtx);
        return b;
    }
    /* Ghost admission: a miss only earns a pinned slot after it has been
     * requested-and-missing enough times. One-shot experts from prefill and
     * cold routes are read transiently and dropped, so the budget stays on
     * the experts decode reuses. */
    if (!cwick_stream_expert_should_pin(s, layer, expert, matrix)) {
        g_res_calls++;   /* counted as a filtered reserve */
        pthread_mutex_unlock(&s->cache_mtx);
        return NULL;
    }
    _t0 = cwick_prof_ns();
    if (s->cache_used + bytes > s->cache_cap) cache_make_room(s, bytes);
    g_res_evict_ns += cwick_prof_ns() - _t0;
    /* The mutex must come back on every path: this one used to return with
     * cache_mtx held, which deadlocks the next decode step's lookup -- and a
     * budget that will not admit the blob is reachable in normal operation
     * (trim lowered the cap, or the arena refused), not just on a bad day. */
    if (s->cache_used + bytes > s->cache_cap) {
        pthread_mutex_unlock(&s->cache_mtx);
        return NULL;
    }
    if (s->slots[i]) slot_release(s, i);

    _t0 = cwick_prof_ns();
    cwick_stream_entry *e = cwick_xmalloc(sizeof(*e));
    g_res_slot_ns += cwick_prof_ns() - _t0;
    g_res_calls++;
    e->key = key;
    e->bytes = bytes;
    e->heat = s->heat_avg;
    e->last_use = s->pass;
    e->from_pool = false;
    e->pending = true;
    _t0 = cwick_prof_ns();
    if (!s->pool && !s->pool_off) pool_init(s, bytes);
    e->blob = pool_alloc(s, bytes);
    e->from_pool = e->blob != NULL;
    if (!e->blob) e->blob = cwick_shmem_alloc(bytes);
    g_res_shmem_ns += cwick_prof_ns() - _t0;
    if (!e->blob) { free(e); pthread_mutex_unlock(&s->cache_mtx); return NULL; }
    s->slots[i] = e;
    s->cache_used += bytes;
    ghost_clear(s, key);   /* now resident: stop counting it as a miss */
    pthread_mutex_unlock(&s->cache_mtx);
    return e->blob;
}

/* The counterpart to cwick_stream_expert_reserve: the read that filled this
 * slot has completed, so it is ordinary cached data again and may be evicted.
 * Called once per completed read -- a batch that is accounted is either
 * aborted (failure) or settled (success), so nothing stays protected. */
void cwick_stream_expert_settle(cwick_stream *s, int64_t layer, int64_t expert,
                               int64_t matrix) {
    if (!s || !s->cache_on) return;
    int64_t key = layer * 1000000 + expert * 10 + matrix;
    pthread_mutex_lock(&s->cache_mtx);
    size_t base = slot_for(s, key);
    for (size_t p = 0; p < CWICK_CACHE_PROBES; p++) {
        size_t i = (base + p) & (s->slot_count - 1);
        cwick_stream_entry *e = s->slots[i];
        if (!e) break;
        if (e->key == key) { e->pending = false; break; }
    }
    pthread_mutex_unlock(&s->cache_mtx);
}

/* Drop a slot reserved by cwick_stream_expert_reserve whose read failed. */
void cwick_stream_expert_abort(cwick_stream *s, int64_t layer, int64_t expert,
                              int64_t matrix) {
    if (!s || !s->cache_on) return;
    int64_t key = layer * 1000000 + expert * 10 + matrix;
    pthread_mutex_lock(&s->cache_mtx);
    size_t base = slot_for(s, key);
    for (size_t p = 0; p < CWICK_CACHE_PROBES; p++) {
        size_t i = (base + p) & (s->slot_count - 1);
        cwick_stream_entry *e = s->slots[i];
        if (!e) break;
        if (e->key == key) { slot_release(s, i); break; }
    }
    pthread_mutex_unlock(&s->cache_mtx);
}

/* Bump the decay clock. Called once per decode step; every CWICK_HEAT_AGE
 * passes all frequency counters halve, so an expert that stops being routed
 * cools to zero and loses its pin. Without this the counters only grow and
 * the cache freezes on whatever it happened to admit first — which is
 * exactly what the old recency model did, at 53% hit instead of ~82%.
 * The sweep is O(slot_count) but runs once per AGE steps, not per miss. */
void cwick_stream_expert_decay(cwick_stream *s) {
    if (!s || !s->cache_on) return;
    s->pass++;
    if ((s->pass % CWICK_HEAT_AGE) != 0) return;
    uint64_t sum = 0;
    uint64_t n = 0;
    for (size_t k = 0; k < s->slot_count; k++) {
        cwick_stream_entry *e = s->slots[k];
        if (!e) continue;
        e->heat >>= 1;   /* geometric aging; reaches 0 and becomes evictable */
        sum += e->heat;
        n++;
    }
    /* A fresh entry is admitted at the average, not at zero. Starting a
     * newcomer below every resident is the classic LFU trap: once the cache
     * is full nothing new can ever earn a place, so the pinned set freezes
     * on whatever the prefill happened to touch first. */
    s->heat_avg = n ? (uint32_t)(sum / n) : CWICK_HEAT_START;
    if (s->heat_avg < CWICK_HEAT_START) s->heat_avg = CWICK_HEAT_START;
    ghost_decay(s);
}

/* ------------------------------------------------------------------ */
/* Memory steering (keeps the 80% ceiling honest)                      */
/* ------------------------------------------------------------------ */
size_t cwick_stream_cache_used(const cwick_stream *s) {
    return s ? s->cache_used : 0;
}

size_t cwick_stream_cache_cap(const cwick_stream *s) {
    return s ? (s->cache_on ? s->cache_cap : 0) : 0;
}

/* Lower the byte ceiling and evict until the cache obeys it. Called when
 * measured RSS drifts above the plan's target: give back expert-cache bytes
 * rather than let the OS start compressing/swapping a 320B decode. */
size_t cwick_stream_trim(cwick_stream *s, size_t target_cap) {
    if (!s || !s->cache_on) return 0;
    size_t before = s->cache_used;
    if (target_cap > s->cache_cap) target_cap = s->cache_cap;
    size_t freed = 0;
    int guard = 0;
    while (s->cache_used > target_cap && guard++ < 4096) {
        size_t victim = SIZE_MAX;
        uint32_t coldest = UINT32_MAX;
        for (size_t k = 0; k < s->slot_count; k++) {
            cwick_stream_entry *e = s->slots[k];
            if (e && e->heat < coldest) { coldest = e->heat; victim = k; }
        }
        if (victim == SIZE_MAX) break;
        slot_release(s, victim);
    }
    freed = before - s->cache_used;
    return freed;
}

/* Shrink the *hard* ceiling too, so a trim survives the next refill. */
void cwick_stream_limit(cwick_stream *s, size_t cache_cap) {
    if (!s || !s->cache_on) return;
    if (cache_cap < s->cache_cap) {
        s->cache_cap = cache_cap;
        cwick_stream_trim(s, cache_cap);
    }
}

/* How long the valve stays quiet after a trim, in milliseconds. It has to be
 * long enough for the freed bytes to show up in phys_footprint; at ~1.4 tok/s
 * 12 s is a handful of decode steps. */
#define CWICK_STEER_COOLDOWN_MS 12000

/* Monotonic milliseconds for the cooldown. */
static uint64_t steer_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ull + (uint64_t)(ts.tv_nsec / 1000000ull);
}

/* Give back pinned expert bytes when the process is over its memory plan.
 *
 * The point is to be the one who decides which bytes leave, not the VM. The
 * pin cache is ordinary malloc'ed memory; left alone it fills to its cap and
 * then the box goes over its RAM, and macOS answers that by compressing and
 * swapping the very pages the decode is about to read. On this machine that
 * cost 3x the step time at an unchanged cache-hit rate: the "hits" were
 * decompression faults. Trimming a few GiB of cold experts is far cheaper
 * than letting the compressor own the working set.
 *
 * `ceiling` is the whole-process budget (80% of host RAM by default), not the
 * cache budget, so this also accounts for the KV caches and the GPU-side
 * allocations the pin budget cannot see. */
size_t cwick_stream_steer_memory(cwick_stream *s, uint64_t ceiling, int64_t step,
                                 const char *tag) {
    if (!s || ceiling == 0) return 0;
    if ((step % CWICK_STEER_EVERY) != 0) return 0;
    uint64_t now = steer_now_ms();

    /* Cooldown first. Trimming does not shrink phys_footprint in the same
     * instant -- the freed blobs are still counted until the VM has accounted
     * for them -- so a valve that runs every step measures the same overshoot
     * again and again and walks the cap down until there is nothing left.
     * Measured with a deliberately oversized 90 GiB cap: the cap reached 0 in
     * about two seconds while every line reported "released 0.00 GiB" and the
     * footprint had not moved. One trim, then silence until it can have taken
     * effect. */
    if (s->steer_cooldown_until_ms && now < s->steer_cooldown_until_ms) return 0;

    uint64_t footprint = cwick_process_footprint_bytes();
    if (footprint == 0 || footprint <= ceiling) {
        s->steer_cooldown_until_ms = 0;
        return 0;
    }

    uint64_t over = footprint - ceiling;
    size_t cap = cwick_stream_cache_cap(s);
    if (cap == 0) return 0;
    /* Ask for a bit more than the overshoot, so the next refill does not
     * re-trip the check immediately -- but never hand back the whole cache,
     * or the decode loses its working set outright. */
    const size_t margin = 1ull << 30;
    size_t want = (cap > over + margin) ? (size_t)(cap - over - margin) : 0;
    size_t floor_cap = cap / 4;
    if (floor_cap < (256ull << 20)) floor_cap = 256ull << 20;
    if (want < floor_cap) want = floor_cap;
    size_t freed = cwick_stream_trim(s, want);
    /* remember the lower ceiling so the next refill does not undo this */
    cwick_stream_limit(s, want);
    s->steer_cooldown_until_ms = now + CWICK_STEER_COOLDOWN_MS;
    CWICK_INFO(tag ? tag : "mem", "physical footprint %.2f GiB is %.2f GiB over "
               "the ceiling — released %.2f GiB of pinned experts (cap now "
               "%.2f GiB, %.0f s before the next trim)",
               cwick_gib(footprint), cwick_gib(over), cwick_gib(freed),
               cwick_gib(cwick_stream_cache_cap(s)),
               (double)CWICK_STEER_COOLDOWN_MS / 1000.0);
    return freed;
}

/* ------------------------------------------------------------------ */
/* Cross-layer expert prefetch (opt-in, CWICK_PREFETCH=1)               */
/*                                                                     */
/* One background reader thread issues F_NOCACHE preads for the expert */
/* matrices the engine predicts the *next* layer will need (the        */
/* current layer's post-attention residual feeds the next layer's      */
/* router, so reads can start before the compute step). The MoE        */
/* forward consults the prefetch arena first and only falls back to a  */
/* blocking demand read on a miss. Prediction never changes routing;   */
/* the authoritative router still selects the experts.                 */
/* ------------------------------------------------------------------ */

typedef struct cwick_pf_job {
    int64_t layer, expert, matrix;
    uint8_t *dst;          /* arena slot */
    size_t bytes, sbytes;  /* matrix bytes + scale bytes */
    size_t shard;
    uint64_t off, soff;
    int status;
    int zone;
    struct cwick_pf_job *next;
} cwick_pf_job;

typedef struct cwick_pf_layer {
    int64_t layer;            /* routed MoE layer index */
    float ema_pilot;          /* EMA recall of the hint predictor */
    float ema_prev;           /* EMA recall of the prev-token predictor */
    uint32_t samples;         /* sample counter (saturates) */
    int64_t prev_ids[16];     /* last authoritative ids (prev-token predictor) */
    int64_t n_prev;
    int64_t hint_ids[16];     /* submitted hint for this layer */
    int64_t n_hint;
} cwick_pf_layer;

typedef struct cwick_pf {
    cwick_stream *s;
    pthread_t th[8];
    int n_readers;
    volatile int run;
    cwick_pf_job *head, *tail;         /* pending FIFO */
    cwick_pf_job *done_head, *done_tail; /* completed, unclaimed */
    cwick_pf_job *active[8];            /* one in-flight job per reader */
    pthread_mutex_t mtx;
    pthread_cond_t cv_new, cv_done;
    uint8_t *arena;
    size_t arena_cap, zone_size; /* two zones: current + the one being consumed */
    size_t zone_used[2];
    int zone;                    /* current write zone (0/1) */

    /* Prefetch gate: per-layer adaptive prefetch admission.
     * Scoring never stops; only speculative *reads* are suppressed once a
     * layer's best-predictor EMA is below threshold, so a layer whose recall
     * recovers automatically resumes prefetching. */
    cwick_pf_layer *layers;
    size_t layer_cap, layer_count;
    int gated;                 /* env: CWICK_PILOT_GATE (on/measure/off) */
    float threshold;           /* EMA below which reads stop (default 0.10) */
    uint32_t warmup;           /* samples before the gate may act (default 16) */
} cwick_pf;

/* Drop every job belonging to `zone` from one of the two FIFOs.
 *
 * The tail is recomputed by walking whatever survives rather than patched up
 * in place: a stale tail is how this list used to end up pointing at a node
 * that had already been freed, and the next append would then write through
 * freed memory. */
static void discard_zone_jobs(cwick_pf_job **head, cwick_pf_job **tail, int zone) {
    cwick_pf_job **pp = head;
    while (*pp) {
        cwick_pf_job *j = *pp;
        if (j->zone == zone) {
            *pp = j->next;
            free(j);
        } else {
            pp = &j->next;
        }
    }
    cwick_pf_job *last = NULL;
    for (cwick_pf_job *j = *head; j; j = j->next) last = j;
    *tail = last;
}

/* Reset the write zone for a new submit batch (called once per layer). The
 * previous zone is drained and retired before its arena bytes are reused. */
void cwick_stream_prefetch_begin(cwick_stream *s) {
    cwick_pf *pf = s->pf;
    if (!pf) return;
    pthread_mutex_lock(&pf->mtx);
    int old_zone = pf->zone;
    for (;;) {
        int busy = 0;
        for (int i = 0; i < pf->n_readers; i++)
            if (pf->active[i] && pf->active[i]->zone == old_zone) busy = 1;
        if (!busy) break;
        pthread_cond_wait(&pf->cv_done, &pf->mtx);
    }
    discard_zone_jobs(&pf->head, &pf->tail, old_zone);
    if (pf->head) {
        cwick_pf_job *last = pf->head;
        while (last->next) last = last->next;
        pf->tail = last;
    } else {
        pf->tail = NULL;
    }
    discard_zone_jobs(&pf->done_head, &pf->done_tail, old_zone);
    if (pf->done_head) {
        cwick_pf_job *last = pf->done_head;
        while (last->next) last = last->next;
        pf->done_tail = last;
    } else {
        pf->done_tail = NULL;
    }
    pf->zone_used[old_zone] = 0;
    pf->zone ^= 1;
    pthread_mutex_unlock(&pf->mtx);
}

/* carve a slot out of the current write zone */
static uint8_t *pf_alloc(cwick_pf *pf, size_t bytes) {
    size_t *used = &pf->zone_used[pf->zone];
    if (*used + bytes > pf->zone_size) return NULL;
    uint8_t *p = pf->arena + pf->zone * pf->zone_size + *used;
    *used += bytes;
    return p;
}

typedef struct {
    cwick_pf *pf;
    int idx;
} cwick_pf_reader;

static void *pf_reader(void *arg) {
    cwick_pf_reader *rd = (cwick_pf_reader *)arg;
    cwick_pf *pf = rd->pf;
    int idx = rd->idx;
    free(rd);
#if defined(__APPLE__)
    /* Speculative reads must lose every argument with the demand path.
     *
     * Measured: CWICK_PREFETCH=1 took a 265-token prompt from 714 to 1873
     * ms/step (2.6x slower), and 590 to 2111 ms/step in an 80-step warm-cache
     * A/B after this change. The cause is not SSD bandwidth — it is cores.
     * The demand path reads a whole layer with a twelve-worker fork-join
     * (fetch_range), and a fork-join GEMV is paced by its slowest
     * participant; a prefetch thread at default QoS preempts those workers
     * mid-kernel and the layer gets *slower* than if the read had never
     * happened.
     *
     * Raising the demand readers instead is not available here: they are
     * pool workers inside a fork-join whose GEMVs already run at
     * USER_INITIATED. So the complementary move is the one that fits: push
     * the *speculative* thread down until it cannot preempt anything that
     * matters.
     *
     * QOS_CLASS_BACKGROUND is the only class the scheduler will not let
     * preempt USER_INITIATED work, which is what the GEMV pool runs at.
     * A wrong prediction therefore costs progress only when a core is
     * genuinely idle, which is exactly the intent of speculative I/O.
     * Measured result: the prefetch is still a loss (it reads 2.7x the
     * bytes for no hit-rate gain), so it stays opt-in; see docs/PERF.md. */
    pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);
#endif
    for (;;) {
        pthread_mutex_lock(&pf->mtx);
        while (pf->head == NULL && pf->run) pthread_cond_wait(&pf->cv_new, &pf->mtx);
        if (!pf->run && pf->head == NULL) { pthread_mutex_unlock(&pf->mtx); break; }
        cwick_pf_job *j = pf->head;
        pf->head = j->next;
        if (pf->head == NULL) pf->tail = NULL;
        /* CRITICAL: a job is on exactly one list at a time. Leftover `next`
         * from the pending FIFO used to stay attached while the job was being
         * read, so the completed-FIFO walk in prefetch_wait could reach an
         * in-flight job, hand its buffer out as if the bytes were there, and
         * free it while this thread was still pread-ing into it. Worse, the
         * stale link could close the done list into a ring, which parks the
         * consumer in `while (last->next)` at 100% CPU holding the mutex this
         * thread needs -- a hang that looks exactly like a slow prefill. */
        j->next = NULL;
        pf->active[idx] = j;
        pthread_mutex_unlock(&pf->mtx);

        /* Complete both payloads before publishing the job. A waiter can then
         * consume the slot without observing a half-written scale tensor. */
        size_t got = 0, sgot = 0;
        while (got < j->bytes) {
            ssize_t r = pread(pf->s->fds[j->shard], j->dst + got,
                              j->bytes - got, (off_t)j->off + (off_t)got);
            if (r < 0) { if (errno == EINTR) continue; break; }
            if (r == 0) break;
            got += (size_t)r;
        }
        if (got == j->bytes && j->sbytes) {
            while (sgot < j->sbytes) {
                ssize_t r = pread(pf->s->fds[j->shard], j->dst + j->bytes + sgot,
                                  j->sbytes - sgot,
                                  (off_t)j->soff + (off_t)sgot);
                if (r < 0) { if (errno == EINTR) continue; break; }
                if (r == 0) break;
                sgot += (size_t)r;
            }
        }
        j->status = (got == j->bytes && sgot == j->sbytes) ? 0 : -1;

        pthread_mutex_lock(&pf->mtx);
        pf->active[idx] = NULL;
        j->next = NULL;
        if (pf->done_tail) pf->done_tail->next = j;
        else pf->done_head = j;
        pf->done_tail = j;
        pthread_mutex_lock(&pf->s->stats_mtx);
        pf->s->bytes_streamed += got + sgot;
        pf->s->read_calls++;
        pthread_mutex_unlock(&pf->s->stats_mtx);
        pthread_cond_broadcast(&pf->cv_done);
        pthread_mutex_unlock(&pf->mtx);
    }
    return NULL;
}

/* Spawn the prefetch reader. `arena_bytes` bounds the prefetch arena; it is
 * split into two zones so one layer's prefetch can be consumed while the
 * next layer's is being read. Returns CWICK_OK, or CWICK_ERR_IO if threads. */
cwick_status cwick_stream_prefetch_open(cwick_stream *s, size_t arena_bytes,
                                      cwick_error *err) {
    cwick_pf *pf = cwick_xmalloc(sizeof(*pf));
    memset(pf, 0, sizeof(*pf));
    pf->s = s;
    pf->arena_cap = arena_bytes;
    pf->zone_size = arena_bytes / 2;
    pf->zone = 0;
    /* GPU-visible: prefetched expert bytes are bound by the GEMV directly. */
    pf->arena = cwick_shmem_alloc(arena_bytes);
    if (!pf->arena) { free(pf); return CWICK_ERR_MEMORY; }
    pthread_mutex_init(&pf->mtx, NULL);
    pthread_cond_init(&pf->cv_new, NULL);
    pthread_cond_init(&pf->cv_done, NULL);
    pf->run = 1;
    pf->gated = 1;  /* default: gate on */
    pf->threshold = CWICK_PILOT_GATE_THRESHOLD;
    pf->warmup = CWICK_PILOT_GATE_WARMUP;
    {
        const char *g = getenv("CWICK_PILOT_GATE");
        if (g && strcmp(g, "off") == 0) pf->gated = 0;
        else if (g && strcmp(g, "measure") == 0) pf->gated = 2; /* score, no suppress */
        const char *t = getenv("CWICK_PILOT_GATE_THRESHOLD");
        if (t) { float v = strtof(t, NULL); if (v >= 0.0f && v < 1.0f) pf->threshold = v; }
        const char *w = getenv("CWICK_PILOT_GATE_WARMUP");
        if (w) { long v = strtol(w, NULL, 10); if (v >= 1 && v <= 100000) pf->warmup = (uint32_t)v; }
    }
    pf->layers = cwick_xmalloc(64 * sizeof(*pf->layers));
    pf->layer_cap = 64;
    pf->layer_count = 0;
    /* One reader leaves the SSD at queue depth 1: measured 973 MB of expert
     * bytes cost 243 ms of a decode step, which is 4 GB/s and exactly as long
     * as the reads take with nothing overlapping them. The consumer waits for
     * expert k while readers could be filling k+1.. , so the queue depth is
     * what decides whether that wait is paid at all. Six readers (env
     * CWICK_PF_READERS, 1..8) keep several 5.6 MiB preads in flight without
     * competing for cores: the readers sit at BACKGROUND QoS, so they only run
     * when a performance core is idle. */
    pf->n_readers = 6;
    {
        const char *r = getenv("CWICK_PF_READERS");
        if (r && *r) {
            int v = atoi(r);
            if (v >= 1 && v <= 8) pf->n_readers = v;
        }
    }
    for (int i = 0; i < pf->n_readers; i++) {
        cwick_pf_reader *rd = cwick_xmalloc(sizeof(*rd));
        rd->pf = pf;
        rd->idx = i;
        if (pthread_create(&pf->th[i], NULL, pf_reader, rd) != 0) {
            cwick_error_set(err, CWICK_ERR_IO, "prefetch: pthread_create failed");
            pf->n_readers = i;
            pf->run = 0;
            pthread_cond_broadcast(&pf->cv_new);
            for (int j = 0; j < i; j++) pthread_join(pf->th[j], NULL);
            cwick_shmem_free(pf->arena);
            free(pf->layers);
            pthread_mutex_destroy(&pf->mtx);
            pthread_cond_destroy(&pf->cv_new);
            pthread_cond_destroy(&pf->cv_done);
            free(pf);
            return CWICK_ERR_IO;
        }
    }
    s->pf = pf;
    return CWICK_OK;
}

/* Enqueue a prefetch read for one expert matrix, carved from the current
 * write zone. The reader fills the slot asynchronously. */
void cwick_stream_prefetch_submit(cwick_stream *s, int64_t layer, int64_t expert,
                                 int64_t matrix, size_t bytes, size_t sbytes,
                                 size_t shard, uint64_t off, uint64_t soff) {
    cwick_pf *pf = s->pf;
    if (!pf) return;
    uint8_t *dst;
    pthread_mutex_lock(&pf->mtx);
    int zone = pf->zone;
    dst = pf_alloc(pf, bytes + sbytes);
    pthread_mutex_unlock(&pf->mtx);
    if (!dst) return; /* arena full: fall back to demand I/O */
    cwick_pf_job *j = cwick_xmalloc(sizeof(*j));
    j->layer = layer; j->expert = expert; j->matrix = matrix;
    j->dst = dst; j->bytes = bytes; j->sbytes = sbytes;
    j->shard = shard; j->off = off; j->soff = soff;
    j->status = 0; j->zone = zone; j->next = NULL;
    pthread_mutex_lock(&pf->mtx);
    if (pf->tail) pf->tail->next = j; else pf->head = j;
    pf->tail = j;
    pthread_cond_signal(&pf->cv_new);
    pthread_mutex_unlock(&pf->mtx);
}

/* Block until the (layer, expert, matrix) prefetch completes. Returns the
 * arena slot holding matrix bytes then scale bytes, or NULL if the job is
 * absent (a router miss — fall back to a demand read). */
const void *cwick_stream_prefetch_wait(cwick_stream *s, int64_t layer,
                                      int64_t expert, int64_t matrix) {
    cwick_pf *pf = s->pf;
    if (!pf) return NULL;
    pthread_mutex_lock(&pf->mtx);
    for (;;) {
        /* Scan-and-unlink with `pp` pointing at the link that leads to the
         * current node: on a match the node is spliced out and the scan does
         * NOT advance, because `prev` must stay a node that is still on the
         * list for the tail fix-up below. (Advancing after a removal would
         * leave prev pointing at a job we just freed.) */
        cwick_pf_job **pp = &pf->done_head, *prev = NULL;
        while (*pp) {
            cwick_pf_job *j = *pp;
            if (j->layer == layer && j->expert == expert && j->matrix == matrix) {
                *pp = j->next;
                if (pf->done_tail == j) pf->done_tail = prev;
                if (!pf->done_head) pf->done_tail = NULL;
                const void *ret = j->status == 0 ? j->dst : NULL;
                free(j);
                pthread_mutex_unlock(&pf->mtx);
                return ret;
            }
            pp = &j->next;
            prev = j;
        }
        bool pending = false;
        for (int i = 0; i < pf->n_readers && !pending; i++)
            pending = pf->active[i] && pf->active[i]->layer == layer &&
                      pf->active[i]->expert == expert &&
                      pf->active[i]->matrix == matrix;
        for (cwick_pf_job *j = pf->head; !pending && j; j = j->next)
            pending = j->layer == layer && j->expert == expert &&
                      j->matrix == matrix;
        /* Never block on a reader that has been told to stop: fall back to a
         * demand read instead of waiting for a completion that cannot come. */
        if (!pending || !pf->run) {
            pthread_mutex_unlock(&pf->mtx);
            return NULL;
        }
        pthread_cond_wait(&pf->cv_done, &pf->mtx);
    }
}

void cwick_stream_prefetch_close(cwick_stream *s) {
    cwick_pf *pf = s->pf;
    if (!pf) return;
    pthread_mutex_lock(&pf->mtx);
    pf->run = 0;
    pthread_cond_broadcast(&pf->cv_new);
    pthread_mutex_unlock(&pf->mtx);
    for (int i = 0; i < pf->n_readers; i++) pthread_join(pf->th[i], NULL);
    cwick_pf_job *j = pf->head;
    while (j) { cwick_pf_job *n = j->next; free(j); j = n; }
    j = pf->done_head;
    while (j) { cwick_pf_job *n = j->next; free(j); j = n; }
    for (int i = 0; i < pf->n_readers; i++) free(pf->active[i]);
    cwick_shmem_free(pf->arena);
    free(pf->layers);
    pthread_mutex_destroy(&pf->mtx);
    pthread_cond_destroy(&pf->cv_new);
    pthread_cond_destroy(&pf->cv_done);
    free(pf);
    s->pf = NULL;
}

/* ------------------------------------------------------------------ */
/* PILOT gate: per-layer adaptive prefetch admission                   */
/* ------------------------------------------------------------------ */
/* Look up (or create) the gate state for a routed MoE layer. */
static cwick_pf_layer *pf_layer(cwick_pf *pf, int64_t layer) {
    for (size_t i = 0; i < pf->layer_count; i++)
        if (pf->layers[i].layer == layer) return &pf->layers[i];
    if (pf->layer_count >= pf->layer_cap) {
        size_t nc = pf->layer_cap * 2;
        cwick_pf_layer *nl = cwick_xrealloc(pf->layers, nc * sizeof(*nl));
        pf->layers = nl;
        pf->layer_cap = nc;
    }
    cwick_pf_layer *pl = &pf->layers[pf->layer_count++];
    memset(pl, 0, sizeof(*pl));
    pl->layer = layer;
    return pl;
}

/* Score a completed authoritative route against an outstanding prediction.
 * `pilot_ids` may be NULL to use the stored hint (the engine records the
 * submitted PILOT hint via cwick_stream_prefetch_hint_set). `auth_ids` are the
 * authoritative router ids. Recall = |pred ∩ auth| / T per layer. This feeds
 * the per-layer EMA; it never changes what is executed. */
void cwick_stream_prefetch_score(cwick_stream *s, int64_t layer,
                                const int64_t *pilot_ids, int64_t n_pilot,
                                const int64_t *auth_ids, int64_t n_auth) {
    cwick_pf *pf = s->pf;
    if (!pf) return;
    pthread_mutex_lock(&pf->mtx);
    cwick_pf_layer *pl = pf_layer(pf, layer);

    /* prev-token predictor recall: its remembered ids vs this authoritative
     * route. */
    if (pl->n_prev > 0 && n_auth > 0) {
        int64_t hit = 0;
        for (int64_t i = 0; i < pl->n_prev; i++)
            for (int64_t j = 0; j < n_auth; j++)
                if (pl->prev_ids[i] == auth_ids[j]) hit++;
        float sample = (float)hit / (float)n_auth;
        if (sample > 1.0f) sample = 1.0f;
        if (pl->samples) pl->ema_prev += 0.2f * (sample - pl->ema_prev);
        else pl->ema_prev = sample;
    }

    /* PILOT predictor recall: the hint that was submitted for this layer. */
    const int64_t *p = pilot_ids;
    int64_t np = n_pilot;
    if (!p && pl->n_hint > 0) { p = pl->hint_ids; np = pl->n_hint; }
    if (p && np > 0 && n_auth > 0) {
        int64_t hit = 0;
        for (int64_t i = 0; i < np; i++)
            for (int64_t j = 0; j < n_auth; j++)
                if (p[i] == auth_ids[j]) hit++;
        float sample = (float)hit / (float)n_auth;
        if (sample > 1.0f) sample = 1.0f;
        if (pl->samples) pl->ema_pilot += 0.2f * (sample - pl->ema_pilot);
        else pl->ema_pilot = sample;
    }

    pl->samples++;
    if (pl->samples > 0xFFFFFF) pl->samples = 0xFFFFFF;
    pthread_mutex_unlock(&pf->mtx);
}

/* Record the submitted PILOT hint for a layer so it can be scored when the
 * authoritative route arrives. */
void cwick_stream_prefetch_hint_set(cwick_stream *s, int64_t layer,
                                   const int64_t *ids, int64_t n) {
    cwick_pf *pf = s->pf;
    if (!pf) return;
    pthread_mutex_lock(&pf->mtx);
    cwick_pf_layer *pl = pf_layer(pf, layer);
    pl->n_hint = n > 16 ? 16 : (n < 0 ? 0 : n);
    for (int64_t i = 0; i < pl->n_hint; i++) pl->hint_ids[i] = ids[i];
    pthread_mutex_unlock(&pf->mtx);
}

/* Remember this layer's authoritative ids for the prev-token predictor. */
void cwick_stream_prefetch_set_prev(cwick_stream *s, int64_t layer,
                                   const int64_t *ids, int64_t n) {
    cwick_pf *pf = s->pf;
    if (!pf) return;
    pthread_mutex_lock(&pf->mtx);
    cwick_pf_layer *pl = pf_layer(pf, layer);
    pl->n_prev = n > 16 ? 16 : (n < 0 ? 0 : n);
    for (int64_t i = 0; i < pl->n_prev; i++) pl->prev_ids[i] = ids[i];
    pthread_mutex_unlock(&pf->mtx);
}

/* Query whether speculative reads are admitted for a layer. Returns 0 (read)
 * if the best warmed predictor is at/above threshold or the gate is off,
 * 1 (suppressed) if below, -1 if unknown (no pf / unwarmed with no fallback).
 * The prev-token predictor is a fallback for layers PILOT is measurably
 * worse on; the stronger warmed predictor wins ties. */
int cwick_stream_prefetch_gated(cwick_stream *s, int64_t layer) {
    cwick_pf *pf = s->pf;
    if (!pf) return -1;
    if (pf->gated != 1) return 0; /* off or measure: never suppress */
    pthread_mutex_lock(&pf->mtx);
    cwick_pf_layer *pl = pf_layer(pf, layer);
    int result = -1;
    uint32_t warmed = pl->samples >= pf->warmup;
    if (warmed) {
        float best = pl->ema_pilot;
        if (pl->ema_prev > best) best = pl->ema_prev; /* prev-token wins ties */
        result = best < pf->threshold ? 1 : 0;
    } else {
        /* unwarmed: preserve legacy behavior (prefetch the PILOT hint). */
        result = 0;
    }
    pthread_mutex_unlock(&pf->mtx);
    return result;
}


/* ------------------------------------------------------------------ */
/* Reporting                                                           */
/* ------------------------------------------------------------------ */
void cwick_stream_stats(const cwick_stream *s, uint64_t *bytes_streamed,
                       uint64_t *read_calls, uint64_t *cache_hits,
                       uint64_t *cache_misses) {
    if (!s) return;
    pthread_mutex_lock((pthread_mutex_t *)&s->stats_mtx);
    if (bytes_streamed) *bytes_streamed = s->bytes_streamed;
    if (read_calls) *read_calls = s->read_calls;
    if (cache_hits) *cache_hits = s->cache_hits;
    if (cache_misses) *cache_misses = s->cache_misses;
    pthread_mutex_unlock((pthread_mutex_t *)&s->stats_mtx);
}

/* Total anonymous buffers this reader owns (staging arena + prefetch zones).
 * The memory planner has to count them before the decision is made. */
size_t cwick_stream_rss_bytes(const cwick_stream *s) {
    if (!s) return 0;
    return s->arena_bytes + (s->pf ? s->pf->arena_cap : 0) + s->cache_used;
}

size_t cwick_stream_arena_bytes(const cwick_stream *s) {
    return s ? s->arena_bytes : 0;
}

bool cwick_stream_nocache_active(const cwick_stream *s) {
    return s ? s->nocache : false;
}
