/* rt_torch.c — torch.save checkpoint reader (zip central directory + pickle).
 *
 * Two layers, both read-only over one mmap:
 *
 *   1. zip     — locate the end-of-central-directory record, walk the central
 *                directory and resolve each entry's payload offset from its
 *                local header. Checkpoint payloads are stored uncompressed,
 *                so the entry data is the tensor data.
 *   2. pickle  — a stack machine for the subset of opcodes a torch state dict
 *                emits. The only Python objects it materialises are the ones
 *                the graph needs to describe a tensor: strings, ints, tuples
 *                (shape/stride/args), dicts (the state dict itself), globals
 *                (the rebuild function names) and reduce results.
 *
 * `_rebuild_tensor_v2(storage, offset, size, stride, ...)` is where the two
 * layers meet: its first argument arrives through BINPERSID as
 * `('storage', dtype, key, location, numel)`, which names the blob entry
 * `data/<key>` and the element type. */
#include "rt_torch.h"

#include <sys/mman.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "log.h"
#include "util.h"

#define RT_MAX_DIM 8
#define RT_TAG "tensor"

/* ------------------------------------------------------------------ zip --- */

typedef struct {
    char name[128];
    uint32_t local_off;
    uint64_t csize;
    uint64_t usize;
    uint16_t method;
    const uint8_t *data; /* resolved payload, NULL when not resolvable */
    uint64_t avail;
} zip_entry;

typedef struct {
    zip_entry *e;
    int n;
} zip_dir;

static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | ((uint16_t)p[1] << 8)); }

static uint32_t rd32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t rd64(const uint8_t *p) {
    return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32);
}

/* Returns the offset of the zip64 end-of-central-directory record, or 0. */
static uint64_t zip64_eocd(const uint8_t *m, size_t len, size_t eocd_off) {
    if (eocd_off < 20) return 0;
    /* The locator sits immediately before the classic EOCD record. */
    size_t loc = eocd_off - 20;
    if (rd32(m + loc) != 0x07064b50u) return 0;
    uint64_t off = rd64(m + loc + 8);
    if (off + 56 > len) return 0;
    if (rd32(m + off) != 0x06064b50u) return 0;
    return off;
}

static int zip_open(zip_dir *z, const uint8_t *m, size_t len) {
    memset(z, 0, sizeof(*z));
    if (len < 22) return -1;
    /* EOCD is the last record; the trailing 64 KiB may hold a comment. */
    size_t scan_from = len > (size_t)(22 + 65535) ? len - (size_t)(22 + 65535) : 0;
    size_t eocd = 0;
    bool found = false;
    for (size_t i = len - 22 + 1; i-- > scan_from;) {
        if (rd32(m + i) == 0x06054b50u) {
            eocd = i;
            found = true;
            break;
        }
    }
    if (!found) return -1;

    uint64_t total = rd16(m + eocd + 10);
    uint64_t cd_size = rd32(m + eocd + 12);
    uint64_t cd_off = rd32(m + eocd + 16);

    uint64_t z64 = zip64_eocd(m, len, eocd);
    if (z64) {
        total = rd64(m + z64 + 32);
        cd_size = rd64(m + z64 + 40);
        cd_off = rd64(m + z64 + 48);
    }
    if (total == 0 || total > (1u << 20)) return -1;
    if (cd_off >= len || cd_size > len || cd_off + cd_size > len) return -1;

    z->e = (zip_entry *)cwick_xmalloc(sizeof(zip_entry) * (size_t)total);
    memset(z->e, 0, sizeof(zip_entry) * (size_t)total);
    z->n = 0;

    size_t p = (size_t)cd_off;
    size_t end = (size_t)(cd_off + cd_size);
    for (uint64_t k = 0; k < total; k++) {
        if (p + 46 > end || rd32(m + p) != 0x02014b50u) break;
        zip_entry *e = &z->e[z->n];
        e->method = rd16(m + p + 10);
        e->csize = rd32(m + p + 20);
        e->usize = rd32(m + p + 24);
        size_t nlen = rd16(m + p + 28);
        size_t xlen = rd16(m + p + 30);
        size_t clen = rd16(m + p + 32);
        e->local_off = rd32(m + p + 42);
        if (nlen >= sizeof(e->name) || p + 46 + nlen > end) break;
        memcpy(e->name, m + p + 46, nlen);
        e->name[nlen] = '\0';

        /* zip64 extended information: fields appear in a fixed order, only
         * those whose 32-bit slot was saturated. */
        if (e->csize == 0xFFFFFFFFu || e->usize == 0xFFFFFFFFu ||
            e->local_off == 0xFFFFFFFFu) {
            size_t x = p + 46 + nlen;
            size_t xe = x + xlen;
            while (x + 4 <= xe) {
                uint16_t id = rd16(m + x);
                uint16_t sz = rd16(m + x + 2);
                if (id == 0x0001) {
                    size_t q = x + 4;
                    if (e->usize == 0xFFFFFFFFu && q + 8 <= xe) {
                        e->usize = rd64(m + q);
                        q += 8;
                    }
                    if (e->csize == 0xFFFFFFFFu && q + 8 <= xe) {
                        e->csize = rd64(m + q);
                        q += 8;
                    }
                    if (e->local_off == 0xFFFFFFFFu && q + 8 <= xe) {
                        e->local_off = (uint32_t)rd64(m + q);
                        q += 8;
                    }
                    break;
                }
                x += 4u + sz;
            }
        }

        /* Resolve the payload through the local header: its name/extra
         * lengths can differ from the central directory's. */
        size_t lo = e->local_off;
        if (lo + 30 <= len && rd32(m + lo) == 0x04034b50u) {
            size_t ln = rd16(m + lo + 26);
            size_t lx = rd16(m + lo + 28);
            size_t d = lo + 30 + ln + lx;
            if (d <= len) {
                uint64_t room = len - d;
                e->avail = room < e->csize ? room : e->csize;
                if (e->method == 0) e->data = m + d;
            }
        }
        z->n++;
        p += 46 + nlen + xlen + clen;
    }
    return z->n > 0 ? 0 : -1;
}

static const zip_entry *zip_find(const zip_dir *z, const char *name) {
    for (int i = 0; i < z->n; i++)
        if (strcmp(z->e[i].name, name) == 0) return &z->e[i];
    return NULL;
}

/* ----------------------------------------------------------------- vals --- */

typedef enum {
    V_NONE = 0,
    V_INT,
    V_FLOAT,
    V_STR,
    V_TUPLE,
    V_LIST,
    V_DICT,
    V_GLOBAL,
    V_MARK,
    V_STORAGE,
    V_TENSOR,
} vkind;

typedef struct value {
    vkind k;
    int64_t i;
    double f;
    char *s;
    struct value **items;
    int n, cap;
    /* storage / tensor */
    rt_dtype dt;
    int64_t numel;
    int64_t stkey;
    int ndim;
    int64_t shape[RT_MAX_DIM];
    int64_t stride[RT_MAX_DIM];
    int64_t offset;
} value;

typedef struct {
    void **ptrs;
    int n, cap;
} arena;

static void *arena_add(arena *a, void *p) {
    if (a->n == a->cap) {
        a->cap = a->cap ? a->cap * 2 : 256;
        a->ptrs = (void **)cwick_xrealloc(a->ptrs, sizeof(void *) * (size_t)a->cap);
    }
    a->ptrs[a->n++] = p;
    return p;
}

static value *vnew(arena *a, vkind k) {
    value *v = (value *)cwick_xmalloc(sizeof(value));
    memset(v, 0, sizeof(*v));
    v->k = k;
    v->stkey = -1;
    arena_add(a, v);
    return v;
}

static void vpush(value *v, value *item) {
    if (v->n == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 8;
        v->items = (value **)cwick_xrealloc(v->items, sizeof(value *) * (size_t)v->cap);
    }
    v->items[v->n++] = item;
}

static char *arena_str(arena *a, const uint8_t *p, size_t n) {
    char *s = (char *)cwick_xmalloc(n + 1);
    if (n) memcpy(s, p, n);
    s[n] = '\0';
    arena_add(a, s);
    return s;
}

/* ------------------------------------------------------------- opcodes --- */

static const char *global_tail(const char *s) {
    const char *dot = s ? strrchr(s, '.') : NULL;
    return dot ? dot + 1 : s;
}

static bool name_eq(const char *a, const char *b) { return a && b && strcmp(a, b) == 0; }

static rt_dtype dtype_from_storage_name(const char *s) {
    if (!s) return RT_DT_UNKNOWN;
    /* Order matters: "BFloat16Storage" also contains "Float". */
    if (strstr(s, "BFloat16")) return RT_DT_BF16;
    if (strstr(s, "Half")) return RT_DT_F16;
    if (strstr(s, "Double")) return RT_DT_F64;
    if (strstr(s, "Float")) return RT_DT_F32;
    if (strstr(s, "Long")) return RT_DT_I64;
    if (strstr(s, "Short")) return RT_DT_I16;
    if (strstr(s, "Char")) return RT_DT_I8;
    if (strstr(s, "Byte")) return RT_DT_U8;
    if (strstr(s, "Int")) return RT_DT_I32;
    if (strstr(s, "Bool")) return RT_DT_BOOL;
    return RT_DT_UNKNOWN;
}

size_t rt_dtype_size(rt_dtype d) {
    switch (d) {
    case RT_DT_F64: return 8;
    case RT_DT_F32: return 4;
    case RT_DT_F16: return 2;
    case RT_DT_BF16: return 2;
    case RT_DT_I64: return 8;
    case RT_DT_I32: return 4;
    case RT_DT_I16: return 2;
    case RT_DT_I8: return 1;
    case RT_DT_U8: return 1;
    case RT_DT_BOOL: return 1;
    default: return 0;
    }
}

const char *rt_dtype_name(rt_dtype d) {
    switch (d) {
    case RT_DT_F64: return "f64";
    case RT_DT_F32: return "f32";
    case RT_DT_F16: return "f16";
    case RT_DT_BF16: return "bf16";
    case RT_DT_I64: return "i64";
    case RT_DT_I32: return "i32";
    case RT_DT_I16: return "i16";
    case RT_DT_I8: return "i8";
    case RT_DT_U8: return "u8";
    case RT_DT_BOOL: return "bool";
    default: return "?";
    }
}

/* --------------------------------------------------------------- parse --- */

typedef struct {
    const uint8_t *p;
    const uint8_t *end;
    const uint8_t *start;
    arena *ar;
    value **stack;
    int sp, scap;
    value **memo;
    int nmemo, mcap;
    bool bad;
} pkst;

static void pk_error(pkst *pk, const char *what) {
    if (!pk->bad) CWICK_INFO(RT_TAG, "pickle: %s at byte %ld", what, (long)(pk->p - pk->start));
    pk->bad = true;
}

static void spush(pkst *pk, value *v) {
    if (pk->sp == pk->scap) {
        pk->scap = pk->scap ? pk->scap * 2 : 1024;
        pk->stack = (value **)cwick_xrealloc(pk->stack, sizeof(value *) * (size_t)pk->scap);
    }
    pk->stack[pk->sp++] = v;
}

static value *spop(pkst *pk) {
    if (pk->sp <= 0) {
        pk_error(pk, "stack underflow");
        return vnew(pk->ar, V_NONE);
    }
    return pk->stack[--pk->sp];
}

static void memo_put(pkst *pk, int idx, value *v) {
    if (idx < 0) {
        pk_error(pk, "negative memo index");
        return;
    }
    if (idx >= pk->mcap) {
        int nc = pk->mcap ? pk->mcap : 256;
        while (nc <= idx) nc *= 2;
        pk->memo = (value **)cwick_xrealloc(pk->memo, sizeof(value *) * (size_t)nc);
        for (int i = pk->mcap; i < nc; i++) pk->memo[i] = NULL;
        pk->mcap = nc;
    }
    pk->memo[idx] = v;
    if (idx + 1 > pk->nmemo) pk->nmemo = idx + 1;
}

static value *memo_get(pkst *pk, int idx) {
    if (idx < 0 || idx >= pk->nmemo || !pk->memo[idx]) {
        pk_error(pk, "memo miss");
        return vnew(pk->ar, V_NONE);
    }
    return pk->memo[idx];
}

/* Pops down to the nearest MARK and returns the items in push order. */
static value **pop_mark(pkst *pk, int *nout) {
    int mark = -1;
    for (int i = pk->sp - 1; i >= 0; i--)
        if (pk->stack[i]->k == V_MARK) {
            mark = i;
            break;
        }
    if (mark < 0) {
        pk_error(pk, "missing MARK");
        *nout = 0;
        return NULL;
    }
    int n = pk->sp - mark - 1;
    value **out = NULL;
    if (n > 0) {
        out = (value **)cwick_xmalloc(sizeof(value *) * (size_t)n);
        for (int i = 0; i < n; i++) out[i] = pk->stack[mark + 1 + i];
    }
    pk->sp = mark;
    *nout = n;
    return out;
}

static value *make_container(pkst *pk, vkind k) { return vnew(pk->ar, k); }

static void dict_set(value *d, value *key, value *val) {
    for (int i = 0; i + 1 < d->n; i += 2)
        if (d->items[i] == key) {
            d->items[i + 1] = val;
            return;
        }
    vpush(d, key);
    vpush(d, val);
}

/* Builds the tensor value from `_rebuild_tensor_v2` arguments. */
static value *make_tensor(pkst *pk, value *args) {
    if (!args || args->n < 4) {
        pk_error(pk, "bad rebuild args");
        return vnew(pk->ar, V_NONE);
    }
    value *st = args->items[0];
    value *sz = args->items[2];
    value *sd = args->items[3];
    if (st->k != V_STORAGE || sz->k != V_TUPLE || sd->k != V_TUPLE) {
        /* Not a storage-backed tensor we can resolve; keep the graph going. */
        return vnew(pk->ar, V_NONE);
    }
    value *t = vnew(pk->ar, V_TENSOR);
    t->dt = st->dt;
    t->numel = st->numel;
    t->stkey = st->stkey;
    t->offset = args->items[1]->i;
    t->ndim = sz->n < RT_MAX_DIM ? sz->n : RT_MAX_DIM;
    if (sz->n > RT_MAX_DIM) {
        pk_error(pk, "rank too large");
        return vnew(pk->ar, V_NONE);
    }
    for (int i = 0; i < t->ndim; i++) {
        t->shape[i] = sz->items[i]->i;
        t->stride[i] = i < sd->n ? sd->items[i]->i : 1;
    }
    return t;
}

static int pk_reduce(pkst *pk) {
    value *args = spop(pk);
    value *fn = spop(pk);
    const char *tail = global_tail(fn->k == V_GLOBAL ? fn->s : NULL);
    if (name_eq(tail, "_rebuild_tensor_v2") || name_eq(tail, "_rebuild_tensor")) {
        spush(pk, make_tensor(pk, args));
        return 0;
    }
    if (name_eq(tail, "_rebuild_parameter") || name_eq(tail, "_rebuild_parameter_with_state")) {
        spush(pk, args->n > 0 ? args->items[0] : vnew(pk->ar, V_NONE));
        return 0;
    }
    if (name_eq(tail, "_rebuild_from_type_v2")) {
        spush(pk, args->n > 2 ? args->items[2] : vnew(pk->ar, V_NONE));
        return 0;
    }
    if (name_eq(tail, "OrderedDict") || name_eq(tail, "defaultdict") || name_eq(tail, "dict")) {
        spush(pk, make_container(pk, V_DICT));
        return 0;
    }
    /* Anything else (dtypes, devices, storages via constructors) only ever
     * feeds a field we do not read. */
    spush(pk, vnew(pk->ar, V_NONE));
    return 0;
}

static void pk_binpersid(pkst *pk) {
    value *pid = spop(pk);
    if (pid->k != V_TUPLE || pid->n < 3) {
        spush(pk, vnew(pk->ar, V_NONE));
        return;
    }
    value *dts = pid->items[1];
    value *key = pid->items[2];
    const char *dname = dts->k == V_STR ? dts->s : (dts->k == V_GLOBAL ? dts->s : NULL);
    value *sv = vnew(pk->ar, V_STORAGE);
    sv->dt = dtype_from_storage_name(dname);
    if (key->k == V_STR)
        sv->stkey = strtoll(key->s, NULL, 10);
    else if (key->k == V_INT)
        sv->stkey = key->i;
    sv->numel = pid->n > 4 && pid->items[4]->k == V_INT ? pid->items[4]->i : 0;
    spush(pk, sv);
}

/* Reads a little-endian two's-complement integer of `n` bytes. */
static int64_t pk_long_le(const uint8_t *p, int n) {
    uint64_t u = 0;
    for (int i = n - 1; i >= 0; i--) u = (u << 8) | p[i];
    if (n < 8 && (p[n - 1] & 0x80)) u |= ~(uint64_t)0 << (n * 8);
    return (int64_t)u;
}

static int pk_parse(pkst *pk) {
    while (pk->p < pk->end && !pk->bad) {
        uint8_t op = *pk->p++;
        switch (op) {
        case 0x80: /* PROTO */
            pk->p += 1;
            break;
        case 0x95: /* FRAME */
            pk->p += 8;
            break;
        case '.': /* STOP */
            return 0;
        case '(': spush(pk, vnew(pk->ar, V_MARK)); break;
        case 'N': spush(pk, vnew(pk->ar, V_NONE)); break;
        case 0x88: { value *v = vnew(pk->ar, V_INT); v->i = 1; spush(pk, v); break; }
        case 0x89: { value *v = vnew(pk->ar, V_INT); v->i = 0; spush(pk, v); break; }
        case 'K': {
            if (pk->p + 1 > pk->end) { pk_error(pk, "eof K"); break; }
            value *v = vnew(pk->ar, V_INT);
            v->i = pk->p[0];
            pk->p += 1;
            spush(pk, v);
            break;
        }
        case 'M': {
            if (pk->p + 2 > pk->end) { pk_error(pk, "eof M"); break; }
            value *v = vnew(pk->ar, V_INT);
            v->i = rd16(pk->p);
            pk->p += 2;
            spush(pk, v);
            break;
        }
        case 'J': {
            if (pk->p + 4 > pk->end) { pk_error(pk, "eof J"); break; }
            value *v = vnew(pk->ar, V_INT);
            v->i = (int32_t)rd32(pk->p);
            pk->p += 4;
            spush(pk, v);
            break;
        }
        case 0x8a: { /* LONG1 */
            if (pk->p + 1 > pk->end) { pk_error(pk, "eof LONG1"); break; }
            int n = pk->p[0];
            pk->p += 1;
            if (pk->p + n > pk->end) { pk_error(pk, "eof LONG1 body"); break; }
            value *v = vnew(pk->ar, V_INT);
            v->i = n ? pk_long_le(pk->p, n) : 0;
            pk->p += n;
            spush(pk, v);
            break;
        }
        case 0x8b: { /* LONG4 */
            if (pk->p + 4 > pk->end) { pk_error(pk, "eof LONG4"); break; }
            int n = (int)(int32_t)rd32(pk->p);
            pk->p += 4;
            if (n < 0 || pk->p + n > pk->end) { pk_error(pk, "eof LONG4 body"); break; }
            value *v = vnew(pk->ar, V_INT);
            v->i = n ? pk_long_le(pk->p, n) : 0;
            pk->p += n;
            spush(pk, v);
            break;
        }
        case 'G': { /* BINFLOAT, big-endian */
            if (pk->p + 8 > pk->end) { pk_error(pk, "eof G"); break; }
            uint64_t u = 0;
            for (int i = 0; i < 8; i++) u = (u << 8) | pk->p[i];
            pk->p += 8;
            value *v = vnew(pk->ar, V_FLOAT);
            memcpy(&v->f, &u, sizeof(v->f));
            spush(pk, v);
            break;
        }
        case 'X': { /* BINUNICODE */
            if (pk->p + 4 > pk->end) { pk_error(pk, "eof X"); break; }
            uint32_t n = rd32(pk->p);
            pk->p += 4;
            if (pk->p + n > pk->end) { pk_error(pk, "eof X body"); break; }
            value *v = vnew(pk->ar, V_STR);
            v->s = arena_str(pk->ar, pk->p, n);
            pk->p += n;
            spush(pk, v);
            break;
        }
        case 0x8c: { /* SHORT_BINUNICODE */
            if (pk->p + 1 > pk->end) { pk_error(pk, "eof 8c"); break; }
            uint32_t n = pk->p[0];
            pk->p += 1;
            if (pk->p + n > pk->end) { pk_error(pk, "eof 8c body"); break; }
            value *v = vnew(pk->ar, V_STR);
            v->s = arena_str(pk->ar, pk->p, n);
            pk->p += n;
            spush(pk, v);
            break;
        }
        case 0x8d: { /* BINUNICODE8 */
            if (pk->p + 8 > pk->end) { pk_error(pk, "eof 8d"); break; }
            uint64_t n = rd64(pk->p);
            pk->p += 8;
            if (n > (uint64_t)(pk->end - pk->p)) { pk_error(pk, "eof 8d body"); break; }
            value *v = vnew(pk->ar, V_STR);
            v->s = arena_str(pk->ar, pk->p, (size_t)n);
            pk->p += n;
            spush(pk, v);
            break;
        }
        case 'T': { /* BINSTRING */
            if (pk->p + 4 > pk->end) { pk_error(pk, "eof T"); break; }
            uint32_t n = rd32(pk->p);
            pk->p += 4;
            if (pk->p + n > pk->end) { pk_error(pk, "eof T body"); break; }
            value *v = vnew(pk->ar, V_STR);
            v->s = arena_str(pk->ar, pk->p, n);
            pk->p += n;
            spush(pk, v);
            break;
        }
        case 'q': /* BINPUT */
        case 'h': /* BINGET */
        {
            if (pk->p + 1 > pk->end) { pk_error(pk, "eof q/h"); break; }
            int idx = pk->p[0];
            pk->p += 1;
            if (op == 'q') memo_put(pk, idx, pk->sp > 0 ? pk->stack[pk->sp - 1] : NULL);
            else spush(pk, memo_get(pk, idx));
            break;
        }
        case 'r': /* LONG_BINPUT */
        case 'j': /* LONG_BINGET */
        {
            if (pk->p + 4 > pk->end) { pk_error(pk, "eof r/j"); break; }
            int idx = (int)rd32(pk->p);
            pk->p += 4;
            if (op == 'r') memo_put(pk, idx, pk->sp > 0 ? pk->stack[pk->sp - 1] : NULL);
            else spush(pk, memo_get(pk, idx));
            break;
        }
        case 0x94: /* MEMOIZE */
            memo_put(pk, pk->nmemo, pk->sp > 0 ? pk->stack[pk->sp - 1] : NULL);
            break;
        case '}': spush(pk, make_container(pk, V_DICT)); break;
        case ']': spush(pk, make_container(pk, V_LIST)); break;
        case ')': spush(pk, make_container(pk, V_TUPLE)); break;
        case 't': { /* TUPLE */
            int n = 0;
            value **items = pop_mark(pk, &n);
            value *t = vnew(pk->ar, V_TUPLE);
            for (int i = 0; i < n; i++) vpush(t, items[i]);
            free(items);
            spush(pk, t);
            break;
        }
        case 0x85: case 0x86: case 0x87: { /* TUPLE1/2/3 */
            int n = op - 0x84;
            value *t = vnew(pk->ar, V_TUPLE);
            value *tmp[3];
            for (int i = 0; i < n; i++) tmp[i] = spop(pk);
            for (int i = n - 1; i >= 0; i--) vpush(t, tmp[i]);
            spush(pk, t);
            break;
        }
        case 'a': { /* APPEND */
            value *v = spop(pk);
            if (pk->sp > 0) vpush(pk->stack[pk->sp - 1], v);
            break;
        }
        case 'e': { /* APPENDS */
            int n = 0;
            value **items = pop_mark(pk, &n);
            if (pk->sp > 0) {
                value *c = pk->stack[pk->sp - 1];
                for (int i = 0; i < n; i++) vpush(c, items[i]);
            }
            free(items);
            break;
        }
        case 's': { /* SETITEM */
            value *val = spop(pk);
            value *key = spop(pk);
            if (pk->sp > 0 && pk->stack[pk->sp - 1]->k == V_DICT)
                dict_set(pk->stack[pk->sp - 1], key, val);
            break;
        }
        case 'u': { /* SETITEMS */
            int n = 0;
            value **items = pop_mark(pk, &n);
            if (pk->sp > 0 && pk->stack[pk->sp - 1]->k == V_DICT) {
                value *d = pk->stack[pk->sp - 1];
                for (int i = n - 1; i - 1 >= 0; i -= 2) dict_set(d, items[i - 1], items[i]);
            }
            free(items);
            break;
        }
        case 'c': { /* GLOBAL: two newline-terminated lines */
            const uint8_t *nl = (const uint8_t *)memchr(pk->p, '\n', (size_t)(pk->end - pk->p));
            if (!nl) { pk_error(pk, "eof GLOBAL"); break; }
            const uint8_t *mod_b = pk->p;
            size_t mod_n = (size_t)(nl - pk->p);
            pk->p = nl + 1;
            const uint8_t *nl2 = (const uint8_t *)memchr(pk->p, '\n', (size_t)(pk->end - pk->p));
            if (!nl2) { pk_error(pk, "eof GLOBAL name"); break; }
            size_t name_n = (size_t)(nl2 - pk->p);
            pk->p = nl2 + 1;
            value *v = vnew(pk->ar, V_GLOBAL);
            size_t total = mod_n + 1 + name_n;
            char *s = (char *)cwick_xmalloc(total + 1);
            memcpy(s, mod_b, mod_n);
            s[mod_n] = '.';
            memcpy(s + mod_n + 1, nl2 - name_n, name_n);
            s[total] = '\0';
            arena_add(pk->ar, s);
            v->s = s;
            spush(pk, v);
            break;
        }
        case 0x93: { /* STACK_GLOBAL */
            value *nm = spop(pk);
            value *mod = spop(pk);
            value *v = vnew(pk->ar, V_GLOBAL);
            const char *a = mod->k == V_STR ? mod->s : "";
            const char *b = nm->k == V_STR ? nm->s : "";
            size_t total = strlen(a) + 1 + strlen(b);
            char *s = (char *)cwick_xmalloc(total + 1);
            snprintf(s, total + 1, "%s.%s", a, b);
            arena_add(pk->ar, s);
            v->s = s;
            spush(pk, v);
            break;
        }
        case 'R': pk_reduce(pk); break;
        case 'b': { /* BUILD: pop state, leave object */
            value *state = spop(pk);
            if (state->k == V_DICT && pk->sp > 0) {
                value *obj = pk->stack[pk->sp - 1];
                /* `requires_grad` and friends describe autograd, not data. */
                if (obj->k == V_TENSOR) {
                    for (int i = 0; i + 1 < state->n; i += 2) {
                        value *k = state->items[i];
                        if (k->k == V_STR && strcmp(k->s, "requires_grad") == 0) continue;
                    }
                }
            }
            break;
        }
        case 'Q': pk_binpersid(pk); break;
        case 'P': { /* PERSID: line-terminated, not used by torch.save */
            const uint8_t *nl = (const uint8_t *)memchr(pk->p, '\n', (size_t)(pk->end - pk->p));
            if (!nl) { pk_error(pk, "eof PERSID"); break; }
            pk->p = nl + 1;
            spush(pk, vnew(pk->ar, V_NONE));
            break;
        }
        case 0x8e: { /* BINBYTES8 */
            if (pk->p + 8 > pk->end) { pk_error(pk, "eof 8e"); break; }
            uint64_t n = rd64(pk->p);
            pk->p += 8;
            if (n > (uint64_t)(pk->end - pk->p)) { pk_error(pk, "eof 8e body"); break; }
            pk->p += n;
            spush(pk, vnew(pk->ar, V_NONE));
            break;
        }
        case 'B': case 'C': { /* BINBYTES / SHORT_BINBYTES */
            uint32_t n;
            if (op == 'B') {
                if (pk->p + 4 > pk->end) { pk_error(pk, "eof B"); break; }
                n = rd32(pk->p);
                pk->p += 4;
            } else {
                if (pk->p + 1 > pk->end) { pk_error(pk, "eof C"); break; }
                n = pk->p[0];
                pk->p += 1;
            }
            if (pk->p + n > pk->end) { pk_error(pk, "eof bytes body"); break; }
            pk->p += n;
            spush(pk, vnew(pk->ar, V_NONE));
            break;
        }
        case 0x81: /* EXT1 */
        case 0x82: /* EXT2 */
        case 0x83: { /* EXT4 */
            int n = op == 0x81 ? 1 : (op == 0x82 ? 2 : 4);
            pk->p += n + (op == 0x81 ? 1 : 4);
            spush(pk, vnew(pk->ar, V_NONE));
            break;
        }
        default:
            CWICK_INFO(RT_TAG, "pickle: unknown opcode 0x%02x at byte %ld", op,
                       (long)(pk->p - 1 - pk->start));
            pk->bad = true;
            break;
        }
    }
    pk_error(pk, "unterminated stream");
    return -1;
}

/* --------------------------------------------------------- extraction --- */

static bool stride_is_contig(const value *t) {
    int64_t want = 1;
    for (int i = t->ndim - 1; i >= 0; i--) {
        if (t->stride[i] != want) return false;
        want *= t->shape[i] == 0 ? 1 : t->shape[i];
    }
    return true;
}

/* Copies a non-contiguous tensor into a fresh contiguous buffer. */
static void gather(rt_tensor *out, const uint8_t *base, uint64_t avail, const value *t) {
    size_t es = rt_dtype_size(t->dt);
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    if (n <= 0) {
        out->data = base;
        out->numel = 0;
        return;
    }
    uint8_t *buf = (uint8_t *)cwick_xmalloc((size_t)(n * (int64_t)es));
    int64_t idx[RT_MAX_DIM] = {0};
    for (int64_t k = 0; k < n; k++) {
        int64_t off = t->offset;
        for (int i = 0; i < t->ndim; i++) off += idx[i] * t->stride[i];
        uint64_t byte = (uint64_t)off * es;
        if (byte + es <= avail) memcpy(buf + (size_t)k * es, base + byte, es);
        else memset(buf + (size_t)k * es, 0, es);
        for (int i = t->ndim - 1; i >= 0; i--) {
            if (++idx[i] < t->shape[i]) break;
            idx[i] = 0;
        }
    }
    out->data = buf;
    out->owned = true;
    out->numel = n;
}

struct rt_pt {
    void *map;
    size_t map_len;
    int n;
    rt_tensor *t;
};

static void add_tensor(rt_pt *pt, const char *name, const value *v, const zip_dir *z,
                       const char *prefix) {
    char key[192];
    snprintf(key, sizeof(key), "%sdata/%lld", prefix, (long long)v->stkey);
    const zip_entry *e = zip_find(z, key);
    if (!e || !e->data) {
        CWICK_VERB(RT_TAG, "skip %s: no blob %s", name, key);
        return;
    }
    if (e->usize && e->usize < (uint64_t)v->numel * rt_dtype_size(v->dt)) {
        CWICK_INFO(RT_TAG, "skip %s: blob %s too small (%llu < %lld elems)", name, key,
                   (unsigned long long)e->usize, (long long)v->numel);
        return;
    }
    pt->t = (rt_tensor *)cwick_xrealloc(pt->t, sizeof(rt_tensor) * (size_t)(pt->n + 1));
    rt_tensor *d = &pt->t[pt->n++];
    memset(d, 0, sizeof(*d));
    snprintf(d->name, sizeof(d->name), "%s", name);
    d->dtype = v->dt;
    d->ndim = v->ndim;
    d->numel = v->numel;
    int64_t prod = 1;
    for (int i = 0; i < v->ndim; i++) {
        d->shape[i] = v->shape[i];
        prod *= v->shape[i];
    }
    if (prod != v->numel && v->numel != 0) {
        /* Storage longer than the view is normal (offset views); trust shape. */
        d->numel = prod;
    }
    if (stride_is_contig(v)) {
        size_t es = rt_dtype_size(v->dt);
        uint64_t byte = (uint64_t)v->offset * es;
        if (byte <= e->avail) {
            d->data = e->data + byte;
            d->numel = prod;
        } else {
            gather(d, e->data, e->avail, v);
        }
    } else {
        gather(d, e->data, e->avail, v);
    }
}

/* Flattens dicts of dicts into dotted names, so a nested checkpoint and a
 * flat one are addressable the same way. */
static void walk(rt_pt *pt, const value *d, const zip_dir *z, const char *prefix,
                 const char *name_prefix, int depth) {
    if (!d || d->k != V_DICT || depth > 6) return;
    for (int i = 0; i + 1 < d->n; i += 2) {
        const value *k = d->items[i];
        const value *v = d->items[i + 1];
        if (k->k != V_STR) continue;
        char name[160];
        if (name_prefix && *name_prefix)
            snprintf(name, sizeof(name), "%s.%s", name_prefix, k->s);
        else
            snprintf(name, sizeof(name), "%s", k->s);
        if (v->k == V_TENSOR) add_tensor(pt, name, v, z, prefix);
        else if (v->k == V_DICT) walk(pt, v, z, prefix, name, depth + 1);
    }
}

int rt_pt_open(rt_pt **out, const char *path) {
    *out = NULL;
    size_t len = 0;
    /* The speech checkpoints are re-read in full on every decode step, so
     * they must stay resident: MADV_SEQUENTIAL would let the kernel drop each
     * page behind the cursor and every step would fault it back off the SSD. */
    const void *map = cwick_mmap_file_advice(path, &len, MADV_NORMAL);
    if (!map) return -1;

    zip_dir z;
    if (zip_open(&z, (const uint8_t *)map, len) != 0) {
        CWICK_INFO(RT_TAG, "%s: not a zip container", path);
        cwick_munmap(map, len);
        return -1;
    }

    const zip_entry *pkl = zip_find(&z, "data.pkl");
    if (!pkl || !pkl->data) {
        /* The container prefix is not fixed; accept any entry ending in
         * data.pkl so callers never have to know how it was written. */
        for (int i = 0; i < z.n; i++)
            if (strstr(z.e[i].name, "data.pkl") && z.e[i].data) {
                pkl = &z.e[i];
                break;
            }
    }
    if (!pkl || !pkl->data) {
        CWICK_INFO(RT_TAG, "%s: no data.pkl", path);
        cwick_munmap(map, len);
        free(z.e);
        return -1;
    }
    char prefix[128];
    snprintf(prefix, sizeof(prefix), "%s", pkl->name);
    char *slash = strstr(prefix, "data.pkl");
    if (slash) *slash = '\0';

    arena ar;
    memset(&ar, 0, sizeof(ar));
    pkst pk;
    memset(&pk, 0, sizeof(pk));
    pk.p = pk.start = pkl->data;
    pk.end = pkl->data + pkl->avail;
    pk.ar = &ar;

    int rc = pk_parse(&pk);
    rt_pt *pt = NULL;
    if (rc == 0 && pk.sp >= 1) {
        value *root = pk.stack[pk.sp - 1];
        if (root->k == V_DICT) {
            pt = (rt_pt *)cwick_xmalloc(sizeof(*pt));
            memset(pt, 0, sizeof(*pt));
            pt->map = (void *)map;
            pt->map_len = len;
            walk(pt, root, &z, prefix, "", 0);
            if (pt->n == 0) {
                free(pt->t);
                free(pt);
                pt = NULL;
            }
        }
    }

    for (int i = 0; i < ar.n; i++) free(ar.ptrs[i]);
    free(ar.ptrs);
    free(pk.stack);
    free(pk.memo);
    free(z.e);

    if (!pt) {
        cwick_munmap(map, len);
        *out = NULL;
        return -1;
    }
    *out = pt;
    return 0;
}

void rt_pt_close(rt_pt *pt) {
    if (!pt) return;
    for (int i = 0; i < pt->n; i++)
        if (pt->t[i].owned) free((void *)pt->t[i].data);
    free(pt->t);
    if (pt->map) cwick_munmap(pt->map, pt->map_len);
    free(pt);
}

int rt_pt_count(const rt_pt *pt) { return pt ? pt->n : 0; }

const rt_tensor *rt_pt_at(const rt_pt *pt, int i) {
    if (!pt || i < 0 || i >= pt->n) return NULL;
    return &pt->t[i];
}

const rt_tensor *rt_pt_get(const rt_pt *pt, const char *name) {
    if (!pt || !name) return NULL;
    for (int i = 0; i < pt->n; i++)
        if (strcmp(pt->t[i].name, name) == 0) return &pt->t[i];
    return NULL;
}

int rt_pt_count_prefix(const rt_pt *pt, const char *prefix) {
    if (!pt || !prefix) return 0;
    size_t n = strlen(prefix);
    int c = 0;
    for (int i = 0; i < pt->n; i++)
        if (strncmp(pt->t[i].name, prefix, n) == 0) c++;
    return c;
}

bool rt_tensor_fits(const rt_tensor *t, const int64_t *want, int ndim) {
    if (!t || t->ndim != ndim) return false;
    for (int i = 0; i < ndim; i++)
        if (t->shape[i] != want[i]) return false;
    return true;
}

float rt_tensor_f32(const rt_tensor *t, int64_t i) {
    if (!t || !t->data || i < 0 || i >= t->numel) return 0.0f;
    switch (t->dtype) {
    case RT_DT_F32: return ((const float *)t->data)[i];
    case RT_DT_F64: return (float)((const double *)t->data)[i];
    case RT_DT_F16: {
        uint16_t h = ((const uint16_t *)t->data)[i];
        uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
        uint32_t exp = (h >> 10) & 0x1f;
        uint32_t man = h & 0x3ff;
        uint32_t u;
        if (exp == 0) {
            if (man == 0) u = sign;
            else {
                exp = 127 - 15 + 1;
                while (!(man & 0x400)) {
                    man <<= 1;
                    exp--;
                }
                man &= 0x3ff;
                u = sign | (exp << 23) | (man << 13);
            }
        } else if (exp == 0x1f) {
            u = sign | 0x7f800000u | (man << 13);
        } else {
            u = sign | ((exp + 127 - 15) << 23) | (man << 13);
        }
        float f;
        memcpy(&f, &u, sizeof(f));
        return f;
    }
    case RT_DT_BF16: return cwick_bf16_fast(((const uint16_t *)t->data)[i]);
    case RT_DT_I64: return (float)((const int64_t *)t->data)[i];
    case RT_DT_I32: return (float)((const int32_t *)t->data)[i];
    case RT_DT_I16: return (float)((const int16_t *)t->data)[i];
    case RT_DT_I8: return (float)((const int8_t *)t->data)[i];
    case RT_DT_U8: return (float)((const uint8_t *)t->data)[i];
    case RT_DT_BOOL: return ((const uint8_t *)t->data)[i] ? 1.0f : 0.0f;
    default: return 0.0f;
    }
}

void rt_tensor_row_f32(const rt_tensor *t, int64_t row, float *out) {
    if (!t || t->ndim != 2 || !out) return;
    int64_t cols = t->shape[1];
    for (int64_t i = 0; i < cols; i++) out[i] = rt_tensor_f32(t, row * cols + i);
}
