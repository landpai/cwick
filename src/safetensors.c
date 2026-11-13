/* safetensors.c — HF safetensors checkpoint loader (the real checkpoint
 * source). The header contract:
 *   - the file starts with an 8-byte little-endian header length;
 *   - the header is a JSON object keyed by tensor name, each descriptor
 *     holding {dtype, shape, data_offsets};
 *   - tensor bytes begin at 8 + header_len + offsets[0].
 *
 * The released GLM-5.3-Flash checkpoint is 62 `model-*.safetensors` shards
 * plus `model.safetensors.index.json` (weight_map: tensor name -> shard).
 * Weights are HF FP8 E4M3 with a paired F32 `*_scale_inv` (one inverse scale
 * per 128x128 block, stored row-major as [rows/128, cols/128]); norms, router
 * and some projections stay BF16 in the same shards. This loader mmaps every
 * shard once and resolves any tensor by name (binary search on a sorted
 * index), so the multi-hundred-GB checkpoint is never resident in RAM.
 */
#include "cwick.h"
#include "json.h"
#include "log.h"
#include "mem.h"
#include "util.h"

#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static cwick_safetensors *g_cmp_saf; /* qsort comparator context (startup only) */

static int cmp_tensor_idx(const void *a, const void *b) {
    const int64_t ia = *(const int64_t *)a;
    const int64_t ib = *(const int64_t *)b;
    return strcmp(g_cmp_saf->tensors[ia].name, g_cmp_saf->tensors[ib].name);
}

static void *saf_map_push(cwick_safetensors *saf, const void *base, size_t len) {
    size_t n = saf->map_count + 1;
    saf->maps = cwick_xrealloc(saf->maps, n * sizeof(void *));
    saf->map_lens = cwick_xrealloc(saf->map_lens, n * sizeof(size_t));
    saf->maps[saf->map_count] = (void *)(uintptr_t)base;
    saf->map_lens[saf->map_count] = len;
    saf->map_count = n;
    return saf->maps[n - 1];
}

static char *saf_shard_push(cwick_safetensors *saf, const char *name) {
    saf->shard_names =
        cwick_xrealloc(saf->shard_names, (saf->shard_count + 1) * sizeof(char *));
    saf->shard_names[saf->shard_count] = cwick_strdup(name);
    return saf->shard_names[saf->shard_count++];
}

static cwick_st_tensor *saf_tensor_push(cwick_safetensors *saf) {
    saf->tensors = cwick_xrealloc(saf->tensors,
                                 (saf->tensor_count + 1) * sizeof(cwick_st_tensor));
    cwick_st_tensor *t = &saf->tensors[saf->tensor_count++];
    memset(t, 0, sizeof(*t));
    t->rows = t->cols = -1;
    return t;
}

static cwick_st_dtype dtype_from_string(const char *s) {
    if (strcmp(s, "BF16") == 0) return CWICK_ST_BF16;
    if (strcmp(s, "F32") == 0) return CWICK_ST_F32;
    if (strcmp(s, "F8_E4M3") == 0) return CWICK_ST_F8_E4M3;
    if (strcmp(s, "U8") == 0) return CWICK_ST_U8;
    if (strcmp(s, "I8") == 0) return CWICK_ST_I8;
    if (strcmp(s, "F8_E8M0") == 0) return CWICK_ST_F8_E8M0;
    return (cwick_st_dtype)-1;
}

static size_t dtype_elem_size(cwick_st_dtype d) {
    switch (d) {
    case CWICK_ST_BF16: return 2;
    case CWICK_ST_F8_E4M3:
    case CWICK_ST_U8:
    case CWICK_ST_I8:
    case CWICK_ST_F8_E8M0: return 1;
    case CWICK_ST_F32: return 4;
    }
    return 0;
}

/* Parse one shard header and append every tensor to the flat table. */
static cwick_status saf_parse_shard(cwick_safetensors *saf, const char *shard,
                                   int64_t shard_index,
                                   const char *base, size_t file_len,
                                   cwick_error *err) {
    if (file_len < 8) {
        cwick_error_set(err, CWICK_ERR_IO, "safetensors shard %s too small", shard);
        return CWICK_ERR_IO;
    }
    uint64_t hlen64;
    memcpy(&hlen64, base, 8);
    if (hlen64 > (uint64_t)(file_len - 8)) {
        cwick_error_set(err, CWICK_ERR_IO,
                       "safetensors shard %s header length out of range", shard);
        return CWICK_ERR_IO;
    }
    size_t hlen = (size_t)hlen64;
    const char *hdr = base + 8;
    cwick_error_clear(err);
    json_value *root = cwick_json_parse(hdr, hlen, err);
    if (!root || root->kind != JSON_OBJECT) {
        if (root) cwick_json_free(root);
        cwick_error_set(err, CWICK_ERR_PARSE,
                       "safetensors shard %s header is not a JSON object", shard);
        return CWICK_ERR_PARSE;
    }
    const uint8_t *data = (const uint8_t *)base + 8 + hlen;

    for (size_t i = 0; i < root->count; i++) {
        const char *name = root->members[i].key;
        json_value *desc = root->members[i].value;
        if (desc->kind != JSON_OBJECT) continue;
        /* The safetensors spec reserves every key starting with "__" for
         * non-tensor metadata (`__metadata__` carries the framework tag). It
         * is an object, so it looks like a descriptor until it is asked for
         * its dtype. Skip it rather than rejecting the shard: some released
         * checkpoints ship it in one shard and not the others. */
        if (name[0] == '_' && name[1] == '_') continue;
        const char *dtype_s = cwick_json_str(cwick_json_get(desc, "dtype"));
        json_value *shape_v = cwick_json_get(desc, "shape");
        json_value *off_v = cwick_json_get(desc, "data_offsets");
        if (!dtype_s || !shape_v || shape_v->kind != JSON_ARRAY || !off_v ||
            off_v->kind != JSON_ARRAY || cwick_json_array_len(off_v) != 2) {
            cwick_error_set(err, CWICK_ERR_PARSE,
                           "safetensors shard %s tensor %s: bad descriptor", shard, name);
            cwick_json_free(root);
            return CWICK_ERR_PARSE;
        }
        cwick_st_dtype dtype = dtype_from_string(dtype_s);
        if (dtype == (cwick_st_dtype)-1) {
            cwick_error_set(err, CWICK_ERR_UNSUPPORTED,
                           "safetensors shard %s tensor %s: unsupported dtype %s",
                           shard, name, dtype_s);
            cwick_json_free(root);
            return CWICK_ERR_UNSUPPORTED;
        }
        size_t ndim = cwick_json_array_len(shape_v);
        int64_t rows = 1, cols = 1;
        if (ndim == 1) {
            rows = 1;
            cols = cwick_json_int(cwick_json_array_at(shape_v, 0), 1);
        } else if (ndim >= 2) {
            /* linear weights are [out, in]; conv1d weights are [out, 1, k] and
             * flatten to [out, k]. */
            rows = cwick_json_int(cwick_json_array_at(shape_v, 0), 1);
            cols = 1;
            for (size_t d = 1; d < ndim; d++)
                cols *= cwick_json_int(cwick_json_array_at(shape_v, d), 1);
        }
        if (rows <= 0 || cols <= 0) {
            cwick_error_set(err, CWICK_ERR_PARSE,
                           "safetensors shard %s tensor %s: bad shape", shard, name);
            cwick_json_free(root);
            return CWICK_ERR_PARSE;
        }
        int64_t beg = cwick_json_int(cwick_json_array_at(off_v, 0), -1);
        int64_t end = cwick_json_int(cwick_json_array_at(off_v, 1), -1);
        if (beg < 0 || end < beg || (size_t)(8 + hlen + end) > file_len) {
            cwick_error_set(err, CWICK_ERR_IO,
                           "safetensors shard %s tensor %s: offsets out of range",
                           shard, name);
            cwick_json_free(root);
            return CWICK_ERR_IO;
        }

        cwick_st_tensor *t = saf_tensor_push(saf);
        t->rows = rows;
        t->cols = cols;
        t->dtype = dtype;
        t->data = data + (size_t)beg;
        t->data_bytes = (size_t)(end - beg);
        t->shard = cwick_strdup(shard);
        /* Byte offset of the payload inside the shard file. The streaming
         * layer reads experts with pread() on an F_NOCACHE descriptor
         * instead of touching this mapping, so it needs the file offset,
         * not just the mapped pointer. */
        t->shard_index = shard_index;
        t->data_offset = (uint64_t)(8 + hlen + (size_t)beg);
        snprintf(t->name, sizeof(t->name), "%s", name);

        if (dtype == CWICK_ST_F8_E4M3) {
            /* expected bytes: rows*cols (1 byte per element); scale_inv is the
             * paired F32 [rows/128, cols/128] tensor in the same shard. */
            size_t expect = (size_t)(rows * cols);
            if (t->data_bytes != expect) {
                cwick_error_set(err, CWICK_ERR_IO,
                               "safetensors shard %s tensor %s: FP8 size mismatch",
                               shard, name);
                cwick_json_free(root);
                return CWICK_ERR_IO;
            }
            char sname[256];
            snprintf(sname, sizeof(sname), "%s_scale_inv", name);
            json_value *sv = cwick_json_get(root, sname);
            if (sv && sv->kind == JSON_OBJECT) {
                json_value *soff = cwick_json_get(sv, "data_offsets");
                json_value *sshape = cwick_json_get(sv, "shape");
                if (!soff || soff->kind != JSON_ARRAY ||
                    cwick_json_array_len(soff) != 2 || !sshape ||
                    sshape->kind != JSON_ARRAY || cwick_json_array_len(sshape) != 2) {
                    cwick_error_set(err, CWICK_ERR_PARSE,
                                   "safetensors shard %s tensor %s: bad scale_inv",
                                   shard, name);
                    cwick_json_free(root);
                    return CWICK_ERR_PARSE;
                }
                int64_t sbeg = cwick_json_int(cwick_json_array_at(soff, 0), -1);
                int64_t send = cwick_json_int(cwick_json_array_at(soff, 1), -1);
                int64_t srows = cwick_json_int(cwick_json_array_at(sshape, 0), 1);
                int64_t scols = cwick_json_int(cwick_json_array_at(sshape, 1), 1);
                if (sbeg < 0 || send < sbeg || (size_t)(8 + hlen + send) > file_len) {
                    cwick_error_set(err, CWICK_ERR_IO,
                                   "safetensors shard %s tensor %s: scale_inv out of range",
                                   shard, name);
                    cwick_json_free(root);
                    return CWICK_ERR_IO;
                }
                /* contract: scale_inv is [rows/128, cols/128] F32.
                 *
                 * That is the layout of the checkpoints this loader was first
                 * written for. Others (MiniMax-M3.1 among them) ship a
                 * `weight_scale_inv` of a different dtype and block geometry
                 * -- U8 e8m0 at one scale per 32 K elements per row, for
                 * example -- and their backends resolve the scale by name the
                 * way the branch below describes. A foreign shape is therefore
                 * not a broken shard: leave `scale_inv` NULL and let the port
                 * bind its own. Only a size that contradicts the header's own
                 * byte range is an error. */
                if (srows != rows / CWICK_ST_BLOCK || scols != cols / CWICK_ST_BLOCK ||
                    (rows % CWICK_ST_BLOCK) != 0 || (cols % CWICK_ST_BLOCK) != 0) {
                    continue;
                }
                if ((size_t)(srows * scols) * 4 != (size_t)(send - sbeg)) {
                    cwick_error_set(err, CWICK_ERR_IO,
                                   "safetensors shard %s tensor %s: scale_inv size mismatch",
                                   shard, name);
                    cwick_json_free(root);
                    return CWICK_ERR_IO;
                }
                t->scale_inv = data + (size_t)sbeg;
                t->scale_inv_rows = srows;
                t->scale_inv_cols = scols;
            } else {
                /* Other checkpoint families pair FP8 weights with a `.scale`
                 * tensor of a different dtype and block geometry; that port
                 * resolves the scale by name itself. Two namings exist in the
                 * wild: `<name>.scale` and `<name minus .weight>.scale`. Still
                 * refuse an FP8 weight that has no scale of any kind. */
                char dname[256];
                snprintf(dname, sizeof(dname), "%s.scale", name);
                if (!cwick_json_get(root, dname)) {
                    size_t nlen = strlen(name);
                    if (nlen > 7 && strcmp(name + nlen - 7, ".weight") == 0) {
                        snprintf(dname, sizeof(dname), "%.*s.scale",
                                 (int)(nlen - 7), name);
                    }
                }
                /* A tensor that names itself a scale has no scale of its own.
                 * MiniMax-M3.1 stores the expert block scales as e4m3 -- a
                 * `*.weight_scale` beside a `*.weight_packed` -- so the byte
                 * format is the same as a weight's while the role is not. */
                if (!cwick_json_get(root, dname)) {
                    size_t nlen = strlen(name);
                    const char *tail = nlen >= 6 ? name + nlen - 6 : name;
                    bool is_scale = strcmp(tail, "_scale") == 0 ||
                                    strcmp(tail, ".scale") == 0 ||
                                    (nlen >= 13 &&
                                     strcmp(name + nlen - 13, "_global_scale") == 0);
                    if (is_scale) {
                        t->scale_inv = NULL;
                        t->scale_inv_rows = 0;
                        t->scale_inv_cols = 0;
                        continue;
                    }
                }
                if (!cwick_json_get(root, dname)) {
                    cwick_error_set(err, CWICK_ERR_CONTRACT,
                                   "safetensors shard %s tensor %s: missing scale",
                                   shard, name);
                    cwick_json_free(root);
                    return CWICK_ERR_CONTRACT;
                }
                t->scale_inv = NULL;
                t->scale_inv_rows = 0;
                t->scale_inv_cols = 0;
            }
        } else {
            /* BF16: rows*cols*2; F32: rows*cols*4; the byte-wide formats 1 */
            size_t elem = dtype_elem_size(dtype);
            if (elem && t->data_bytes != (size_t)(rows * cols) * elem) {
                cwick_error_set(err, CWICK_ERR_IO,
                               "safetensors shard %s tensor %s: %s size mismatch",
                               shard, name, dtype_s);
                cwick_json_free(root);
                return CWICK_ERR_IO;
            }
        }
    }
    cwick_json_free(root);
    return CWICK_OK;
}

cwick_status cwick_safetensors_open(const char *root, cwick_safetensors *saf,
                                  cwick_error *err) {
    cwick_error_clear(err);
    memset(saf, 0, sizeof(*saf));
    saf->root = cwick_strdup(root);

    char idxpath[1152];
    cwick_path_join(idxpath, sizeof(idxpath), root, "model.safetensors.index.json");
    char *text = NULL;
    size_t len = 0;
    if (cwick_read_file(idxpath, &text, &len) != 0) {
        cwick_error_set(err, CWICK_ERR_IO, "missing safetensors index %s", idxpath);
        return CWICK_ERR_IO;
    }
    json_value *rootv = cwick_json_parse(text, len, err);
    free(text);
    if (!rootv) return CWICK_ERR_PARSE;
    json_value *wm = cwick_json_get(rootv, "weight_map");
    if (!wm || wm->kind != JSON_OBJECT) {
        cwick_error_set(err, CWICK_ERR_CONTRACT,
                       "model.safetensors.index.json missing weight_map");
        cwick_json_free(rootv);
        return CWICK_ERR_CONTRACT;
    }

    /* Collect unique shard file names from the weight map. */
    size_t shard_cap = 8;
    char **shards = cwick_xmalloc(shard_cap * sizeof(char *));
    size_t n_shards = 0;
    for (size_t i = 0; i < wm->count; i++) {
        const char *sh = cwick_json_str(wm->members[i].value);
        if (!sh) {
            cwick_error_set(err, CWICK_ERR_PARSE,
                           "weight_map value %s is not a string",
                           wm->members[i].key);
            cwick_json_free(rootv);
            return CWICK_ERR_PARSE;
        }
        int dup = 0;
        for (size_t k = 0; k < n_shards; k++)
            if (strcmp(shards[k], sh) == 0) { dup = 1; break; }
        if (!dup) {
            if (n_shards == shard_cap) { shard_cap *= 2; shards = cwick_xrealloc(shards, shard_cap * sizeof(char *)); }
            shards[n_shards++] = cwick_strdup(sh);
        }
    }

    /* Map and parse every shard once. */
    for (size_t i = 0; i < n_shards; i++) {
        char path[1152];
        cwick_path_join(path, sizeof(path), root, shards[i]);
        size_t flen = 0;
        const void *base = cwick_mmap_file(path, &flen);
        if (!base) {
            cwick_error_set(err, CWICK_ERR_IO, "cannot map shard %s", shards[i]);
            for (size_t k = 0; k < n_shards; k++) free(shards[k]);
            free(shards);
            cwick_json_free(rootv);
            cwick_safetensors_close(saf);
            return CWICK_ERR_IO;
        }
        saf_map_push(saf, base, flen);
        saf_shard_push(saf, shards[i]);
        saf->total_bytes += flen;
        cwick_status st = saf_parse_shard(saf, shards[i], (int64_t)i, base, flen, err);
        if (st != CWICK_OK) {
            for (size_t k = 0; k < n_shards; k++) free(shards[k]);
            free(shards);
            cwick_json_free(rootv);
            cwick_safetensors_close(saf);
            return st;
        }
    }
    for (size_t k = 0; k < n_shards; k++) free(shards[k]);
    free(shards);
    cwick_json_free(rootv);

    /* Sorted name index for binary-search lookups. */
    saf->order = cwick_xmalloc(saf->tensor_count * sizeof(int64_t));
    for (size_t i = 0; i < saf->tensor_count; i++) saf->order[i] = (int64_t)i;
    g_cmp_saf = saf;
    qsort(saf->order, saf->tensor_count, sizeof(int64_t), cmp_tensor_idx);
    return CWICK_OK;
}

cwick_status cwick_safetensors_get(const cwick_safetensors *saf, const char *name,
                                 cwick_st_tensor *out, cwick_error *err) {
    cwick_error_clear(err);
    size_t lo = 0, hi = saf->tensor_count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        int64_t idx = saf->order[mid];
        int c = strcmp(saf->tensors[idx].name, name);
        if (c == 0) {
            *out = saf->tensors[idx];
            return CWICK_OK;
        }
        if (c < 0) lo = mid + 1;
        else hi = mid;
    }
    cwick_error_set(err, CWICK_ERR_ARGUMENT, "safetensors tensor %s not found", name);
    return CWICK_ERR_ARGUMENT;
}

int64_t cwick_safetensors_count_shards(const cwick_safetensors *saf) {
    return (int64_t)saf->shard_count;
}

uint64_t cwick_safetensors_total_bytes(const cwick_safetensors *saf) {
    return saf->total_bytes;
}

/* Split a mapped checkpoint into what must stay resident and what streams.
 *
 * `cwick serve` has to know this *before* the streaming reader is opened:
 * how much RAM the expert cache may pin is defined as what is left after the
 * resident spine, the inference state and the I/O headroom are paid for.
 * Routed experts — identified by `.mlp.experts.` in the released tensor name —
 * are the only part that streams; everything else is touched by every single
 * token and belongs in RAM. */
void cwick_safetensors_split(const cwick_safetensors *saf,
                            uint64_t *resident_bytes,
                            uint64_t *streamed_bytes) {
    uint64_t resident = 0, streamed = 0;
    if (!saf) {
        if (resident_bytes) *resident_bytes = 0;
        if (streamed_bytes) *streamed_bytes = 0;
        return;
    }
    for (size_t i = 0; i < saf->tensor_count; i++) {
        const cwick_st_tensor *t = &saf->tensors[i];
        if (strstr(t->name, ".mlp.experts.")) streamed += t->data_bytes;
        else resident += t->data_bytes;
    }
    if (resident_bytes) *resident_bytes = resident;
    if (streamed_bytes) *streamed_bytes = streamed;
}

/* ------------------------------------------------------------------ */
/* residency                                                           */
/* ------------------------------------------------------------------ */

uint64_t cwick_safetensors_bytes_under(const cwick_safetensors *saf,
                                       const char *prefix) {
    if (!saf) return 0;
    size_t plen = prefix ? strlen(prefix) : 0;
    uint64_t total = 0;
    for (size_t i = 0; i < saf->tensor_count; i++) {
        const cwick_st_tensor *t = &saf->tensors[i];
        if (prefix && strncmp(t->name, prefix, plen) != 0) continue;
        total += t->data_bytes;
    }
    return total;
}

/* One worker's slice of the fault-in pass. Reading one byte per page is
 * enough: the kernel maps a whole page per fault, and doing more would only
 * add bandwidth for data the caller is about to read anyway. */
typedef struct {
    const uint8_t *base;
    size_t page;
} touch_arg;

static volatile uint64_t g_touch_sink;

static void touch_range(void *argp, int64_t p0, int64_t p1) {
    const touch_arg *a = (touch_arg *)argp;
    uint64_t acc = 0;
    for (int64_t i = p0; i < p1; i++) acc += a->base[i * (int64_t)a->page];
    g_touch_sink += acc;
}

static int cmp_range(const void *a, const void *b) {
    const cwick_st_range *ra = (const cwick_st_range *)a;
    const cwick_st_range *rb = (const cwick_st_range *)b;
    if (ra->p != rb->p) return ra->p < rb->p ? -1 : 1;
    return ra->n < rb->n ? -1 : (ra->n > rb->n ? 1 : 0);
}

uint64_t cwick_safetensors_make_resident(cwick_safetensors *saf,
                                         const char *prefix, int wire) {
    if (!saf || !saf->map_count) return 0;
    if (saf->pinned_count) return saf->pinned_bytes;   /* idempotent */

    size_t plen = prefix ? strlen(prefix) : 0;
    cwick_st_range *r = cwick_xmalloc((saf->tensor_count ? saf->tensor_count : 1) *
                                      sizeof(*r));
    size_t n = 0;
    for (size_t i = 0; i < saf->tensor_count; i++) {
        const cwick_st_tensor *t = &saf->tensors[i];
        if (prefix && strncmp(t->name, prefix, plen) != 0) continue;
        if (!t->data || t->data_bytes == 0) continue;
        r[n].p = (void *)(uintptr_t)t->data;
        r[n].n = t->data_bytes;
        n++;
    }
    if (!n) {
        free(r);
        return 0;
    }
    qsort(r, n, sizeof(*r), cmp_range);

    /* Merge touching or overlapping runs: mlock() is a syscall per call and a
     * layer's weight matrices sit back to back inside a shard, so this turns a
     * few hundred calls into a few dozen. */
    size_t m = 0;
    for (size_t i = 1; i < n; i++) {
        uint8_t *end = (uint8_t *)r[m].p + r[m].n;
        if ((uint8_t *)r[i].p <= end) {
            uint8_t *e2 = (uint8_t *)r[i].p + r[i].n;
            if (e2 > end) r[m].n = (size_t)(e2 - (uint8_t *)r[m].p);
        } else {
            r[++m] = r[i];
        }
    }
    m++;

    long ps = sysconf(_SC_PAGESIZE);
    size_t page = ps > 0 ? (size_t)ps : 16384;
    cwick_pool_init();

    uint64_t bytes = 0;
    int wired = 0;
    for (size_t i = 0; i < m; i++) {
        uint8_t *p = (uint8_t *)r[i].p;
        size_t len = r[i].n;
        /* Ask for the whole run first: the readahead this queues is what makes
         * the per-page faults below cheap instead of one round trip each. */
        madvise(p, len, MADV_WILLNEED);
        int locked = 0;
        if (wire) {
            /* mlock() faults the range in and wires it in one syscall, which
             * is both cheaper than a user-space pass over 1.2 M pages and the
             * only version that cannot be undone by reclaim between the two
             * steps. Page-align down/up: the first tensor of a shard starts at
             * a file offset inside the mapping, not at its page boundary. */
            uintptr_t lo = (uintptr_t)p & ~(uintptr_t)(page - 1);
            uintptr_t hi = ((uintptr_t)p + len + page - 1) & ~(uintptr_t)(page - 1);
            locked = cwick_mem_lock_range((const void *)lo, (size_t)(hi - lo),
                                          "model weights");
            if (locked) wired++;
        }
        if (!locked) {
            touch_arg a;
            a.base = p;
            a.page = page;
            int64_t pages = (int64_t)((len + page - 1) / page);
            cwick_pool_for(pages, &a, touch_range);
        }
        bytes += len;
    }
    /* A sequential hint was installed at mmap time, when the only assumption
     * was "this file is read once, front to back". That is false for a model
     * that lives here: the kernel may drop pages behind the read pointer, and
     * the next token would fault them back in. */
    for (size_t i = 0; i < saf->map_count; i++)
        if (saf->maps[i] && saf->map_lens[i])
            madvise(saf->maps[i], saf->map_lens[i], MADV_NORMAL);

    saf->pinned = r;
    saf->pinned_count = m;
    saf->pinned_bytes = bytes;
    saf->pinned_wired = (wired == (int)m);
    return bytes;
}

void cwick_safetensors_release_residency(cwick_safetensors *saf) {
    if (!saf || !saf->pinned) return;
    long ps = sysconf(_SC_PAGESIZE);
    size_t page = ps > 0 ? (size_t)ps : 16384;
    for (size_t i = 0; i < saf->pinned_count; i++) {
        uint8_t *p = (uint8_t *)saf->pinned[i].p;
        size_t len = saf->pinned[i].n;
        uintptr_t lo = (uintptr_t)p & ~(uintptr_t)(page - 1);
        uintptr_t hi = ((uintptr_t)p + len + page - 1) & ~(uintptr_t)(page - 1);
        cwick_mem_unlock_range((const void *)lo, (size_t)(hi - lo));
    }
    free(saf->pinned);
    saf->pinned = NULL;
    saf->pinned_count = 0;
    saf->pinned_bytes = 0;
    saf->pinned_wired = 0;
}

void cwick_safetensors_close(cwick_safetensors *saf) {
    if (!saf) return;
    /* Give back the wiring before the mapping goes away: an mlock()ed range
     * that is only unmapped later leaves the accounting (and, on a host that
     * counts wired pages against a limit, the budget) claiming bytes that no
     * longer exist. */
    cwick_safetensors_release_residency(saf);
    /* Drop the checkpoint's resident pages before letting go of the index.
     *
     * The maps themselves stay mapped, deliberately: the stream reads through
     * pread on separate descriptors, and pointers into these mappings are not
     * supposed to outlive the model. MADV_DONTNEED is the right primitive for
     * that shape -- the pages leave core immediately, and a stale pointer
     * would refault the same bytes from the file rather than fault. Doing it
     * here is what stops a model switch from inheriting the outgoing
     * checkpoint's residency (measured ~29 GiB for DeepSeek; a second switch
     * round trip took the incoming model from 43.7 to 70.3 GiB warm-up). */
    /* Unmap the checkpoint. On macOS MADV_DONTNEED leaves a file mapping's
     * pages resident -- measured: it reports success on all 48 DeepSeek shards
     * and RSS does not change -- so munmap is the only thing that actually
     * gives the memory back. The tensors' `data` pointers all point into these
     * ranges, which is why this happens after the index is no longer needed
     * and why nothing may hold a tensor past close. */
    size_t n_live = 0, n_unmapped = 0;
    size_t map_mb = 0;
    uint64_t rss0 = cwick_process_rss_bytes();
    for (size_t i = 0; i < saf->map_count; i++) {
        if (!saf->maps[i] || !saf->map_lens || !saf->map_lens[i]) continue;
        n_live++;
        map_mb += saf->map_lens[i] >> 20;
        cwick_munmap(saf->maps[i], saf->map_lens[i]);
        n_unmapped++;
    }
    if (n_live)
        CWICK_VERB("mem", "checkpoint unmapped: %zu of %zu shards "
                  "(%.2f GiB of address space), RSS %.2f -> %.2f GiB",
                  n_unmapped, n_live, cwick_gib((uint64_t)map_mb << 20),
                  cwick_gib(rss0), cwick_gib(cwick_process_rss_bytes()));
    for (size_t i = 0; i < saf->tensor_count; i++) {
        free((char *)saf->tensors[i].shard);
    }
    free(saf->tensors);
    free(saf->order);
    for (size_t i = 0; i < saf->shard_count; i++) free(saf->shard_names[i]);
    free(saf->shard_names);
    free(saf->maps);
    free(saf->map_lens);
    free(saf->root);
    memset(saf, 0, sizeof(*saf));
}
