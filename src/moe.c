/* moe.c — routed-MoE + shared-expert forward (decode step).
 *
 *   router: logits = gate(hidden); scores = sigmoid(logits);
 *           choice = scores + gate.e_score_correction_bias;
 *           top-k ids; weights = scores[ids] / (sum + 1e-20) (renormalize);
 *   expert: gate=w1(x), up=w3(x), h=situ(gate,up), y=w2(h), scaled by weight;
 *   completion: routed_up(rms_norm(routed_sum, routed_norm)) + shared_expert;
 *   shared: shared_down(situ(shared_gate(x), shared_up(x))).
 *
 * The full expert bank is always loaded; the top-k is exactly the model's
 * num_experts_per_token (never reduced).
 */
#include "cwick.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

/* Top-k indices (descending) over `n` scores. */
static void topk_indices(const float *scores, int64_t n, int64_t k,
                        int64_t *ids) {
    /* simple O(n*k) selection; k <= n by the quality guard */
    int64_t *idx = cwick_xmalloc((size_t)n * sizeof(int64_t));
    for (int64_t i = 0; i < n; i++) idx[i] = i;
    for (int64_t i = 0; i < k; i++) {
        int64_t best = i;
        for (int64_t j = i + 1; j < n; j++) {
            if (scores[idx[j]] > scores[idx[best]]) best = j;
        }
        int64_t tmp = idx[i];
        idx[i] = idx[best];
        idx[best] = tmp;
        ids[i] = idx[i];
    }
    free(idx);
}

/* Router prediction: given the current layer's
 * post-attention residual, predict which experts the *next* layer's router
 * will select, so reads can be scheduled before the compute step. This is
 * the same math as the real router (sigmoid + correction bias + top-k); with
 * n_group==1 the group mask degenerates to "all selected", matching the
 * authoritative top-k. Prediction never replaces the real router.
 * Returns 0 on success, -1 if the layer's router tensors are absent. */
int cwick_router_predict(const cwick_model_spec *spec, const cwick_spine *spine,
                        int64_t layer, const float *hidden, int64_t *ids_out) {
    const int64_t H = spec->hidden_size;
    const int64_t E = spec->num_experts;
    const int64_t T = spec->num_experts_per_token;
    char name[128];
    snprintf(name, sizeof(name), "layers.%lld.block_sparse_moe.gate.weight", layer);
    int64_t gi = cwick_spine_find(spine, name);
    if (gi == CWICK_TENSOR_MISSING) return -1;
    const cwick_tensor *gt = &spine->tensors[(size_t)gi];

    float *logits = cwick_xmalloc((size_t)E * sizeof(float));
    float *scores = cwick_xmalloc((size_t)E * sizeof(float));
    float *choice = cwick_xmalloc((size_t)E * sizeof(float));
    cwick_linear(logits, hidden, 1, H, E, gt);
    for (int64_t i = 0; i < E; i++) scores[i] = 1.0f / (1.0f + expf(-logits[i]));
    snprintf(name, sizeof(name),
             "layers.%lld.block_sparse_moe.gate.e_score_correction_bias", layer);
    int64_t bi = cwick_spine_find(spine, name);
    const float *bias = bi != CWICK_TENSOR_MISSING
                        ? (const float *)spine->tensors[(size_t)bi].data : NULL;
    for (int64_t i = 0; i < E; i++) choice[i] = scores[i] + (bias ? bias[i] : 0.0f);
    topk_indices(choice, E, T, ids_out);
    free(choice);
    free(scores);
    free(logits);
    return 0;
}


/* ------------------------------------------------------------------ */
/* Routed-expert access                                                */
/* ------------------------------------------------------------------ */
/* One routed expert matrix, still in its on-disk representation. `data`
 * points either into the shared mapping or into the caller's staging buffer;
 * the bytes are never expanded to fp32, because one [2048, 4096] expert
 * matrix would cost 33 MiB as fp32 and there are 8 of them per layer. */
typedef struct {
    const void *data;
    const float *scale_inv; /* NULL unless F8_E4M3 */
    int64_t ncb;            /* scale_inv row width (cols / 128) */
    int64_t rows, cols;
    bool owned;             /* prefill staging buffer must be released */
    size_t data_bytes;      /* weight bytes (excl. scale side) */
    size_t scale_bytes;     /* scale_inv bytes when streaming */
    cwick_st_dtype dtype;
} cwick_expert_view;

static cwick_status expert_view_impl(const cwick_spine *spine,
                                    const cwick_model_spec *spec, int64_t layer,
                                    int64_t expert, const char *matrix,
                                    uint8_t *staging, size_t staging_off,
                                    size_t staging_cap,
                                    cwick_expert_view *v, cwick_error *err) {
    char n[192], sn[192];
    cwick_expert_saf_name(spec, layer, expert, matrix, n, sizeof(n));
    cwick_st_tensor t, st;
    memset(&st, 0, sizeof(st));
    if (cwick_safetensors_get(spine->saf, n, &t, err) != CWICK_OK) return CWICK_ERR_IO;

    bool have_scale = false;
    if (t.dtype == CWICK_ST_F8_E4M3) {
        snprintf(sn, sizeof(sn), "%s_scale_inv", n);
        if (cwick_safetensors_get(spine->saf, sn, &st, err) != CWICK_OK)
            return CWICK_ERR_IO;
        have_scale = true;
    }

    memset(v, 0, sizeof(*v));
    v->rows = t.rows;
    v->cols = t.cols;
    v->dtype = t.dtype;
    v->data_bytes = t.data_bytes;
    v->scale_bytes = have_scale ? st.data_bytes : 0;

    size_t need = t.data_bytes + (have_scale ? st.data_bytes : 0);
    if (!spine->stream) {
        /* no streaming reader: read straight from the mapping */
        v->data = t.data;
        v->scale_inv = have_scale ? (const float *)st.data : NULL;
        v->ncb = have_scale ? st.cols : 0;
        return CWICK_OK;
    }

    /* ---- sticky heat cache: hit avoids the F_NOCACHE read ----
     * `matrix` maps to a 0..2 index so the (layer, expert, matrix) key is
     * unique. The same expert matrices are re-requested every decode step,
     * so pinning hot ones in RAM is the whole point of the budget. */
    int64_t m_idx = strcmp(matrix, "gate_proj") == 0 ? 0 :
                    strcmp(matrix, "up_proj")   == 0 ? 1 : 2;

    /* ---- prefetch: the engine predicted this expert and the reader
     * thread has (or will have) its bytes already. No F_NOCACHE demand read,
     * no staging, no copy. */
    if (cwick_env_true("CWICK_PREFETCH")) {
        const void *pf = cwick_stream_prefetch_wait(spine->stream, layer,
                                                   expert, m_idx);
        if (pf) {
            v->data = pf;
            v->scale_inv = have_scale ? (const float *)(pf + t.data_bytes) : NULL;
            v->ncb = have_scale ? st.cols : 0;
            return CWICK_OK;
        }
    }

    size_t cbytes = 0;
    const void *cblob = cwick_stream_expert_cached(spine->stream, layer,
                                                  expert, m_idx, &cbytes);
    if (cblob && cbytes >= need) {
        v->data = cblob;
        v->scale_inv = have_scale ? (const float *)(cblob + t.data_bytes) : NULL;
        v->ncb = have_scale ? st.cols : 0;
        return CWICK_OK;
    }

    /* ---- cache miss: read the matrix (and scale) ----
     * With the sticky pin cache on, read straight into a cache-owned slot
     * (avoids the staging -> put double copy). Without it, stage in the
     * reusable buffer as before. */
    const void *slot = cwick_stream_expert_put_read(spine->stream, layer, expert,
                                                   m_idx, &t, have_scale ? &st : NULL,
                                                   err);
    if (slot) {
        v->data = slot;
        v->scale_inv = have_scale ? (const float *)(slot + t.data_bytes) : NULL;
        v->ncb = have_scale ? st.cols : 0;
        return CWICK_OK;
    }

    if (staging_off + need > staging_cap) {
        cwick_error_set(err, CWICK_ERR_MEMORY,
                       "expert staging buffer too small for %s (%zu > %zu)",
                       n, staging_off + need, staging_cap);
        return CWICK_ERR_MEMORY;
    }
    uint8_t *dst = staging + staging_off;
    if (cwick_stream_read(spine->stream, &t, dst, err) != CWICK_OK) return CWICK_ERR_IO;
    if (have_scale) {
        if (cwick_stream_read(spine->stream, &st, dst + t.data_bytes, err) != CWICK_OK)
            return CWICK_ERR_IO;
    }
    v->data = dst;
    v->scale_inv = have_scale ? (const float *)(dst + t.data_bytes) : NULL;
    v->ncb = have_scale ? st.cols : 0;
    return CWICK_OK;
}

/* How the routed group was actually computed. CWICK_GEMV_STATS counts only the
 * CPU fp8 kernel, which is why a run can report ~190 CPU GEMVs per decode step
 * while every group looks batched: these three counters say whether that is a
 * group that Metal declined, a single-matrix dispatch that Metal declined, or
 * a group that never reached the batch at all. */
uint64_t g_moe_group_metal;   /* fused + Metal batch ran                     */
uint64_t g_moe_group_cpu;     /* fused, but the batch was aborted -> CPU     */
uint64_t g_moe_group_serial;  /* per-expert path (no route-first fetch)      */
uint64_t g_moe_single_metal;  /* one matrix, Metal accepted                  */
uint64_t g_moe_single_cpu;    /* one matrix, Metal declined -> CPU kernel    */
/* cwick_moe_forward_batch (prefill windows and decode groups): whether the
 * two-wave Metal batch carried the window or the window fell back to the
 * CPU group / serial paths. Without these, a group that is silently on the
 * CPU shows up only as "rest" in the profile and looks like a mystery. */
uint64_t g_moe_batch_metal;   /* batched window: Metal batch ran             */
uint64_t g_moe_batch_cpu;     /* batched window: aborted -> CPU group        */
uint64_t g_moe_batch_serial;  /* window went per-expert (no route-first)     */
/* Wall ms the batch function's completion section (router-free: routed
 * norm/up projections and the shared expert) has accumulated, for the
 * profiler's split line. */
double g_batch_completion_ms;
double g_batch_router_ms;   /* router + routed_in projection, per the split line */
double g_batch_shared_ms;  /* the shared expert's three wide GEMMs */
uint64_t g_moe_batch_shared_gpu;  /* shared expert rode the Metal batch   */
uint64_t g_moe_batch_shared_cpu;  /* shared expert fell back to the CPU  */


/* y[rows] = W x, straight off the raw bytes. */
static void expert_gemv_impl(float *out, const cwick_expert_view *v,
                             const float *x) {
    if (v->dtype == CWICK_ST_F8_E4M3) {
        /* Offload the FP8 GEMV to the Metal backend when it is up; the CPU
         * dequant+GEMV is the decode step's biggest arithmetic hotspot and
         * the GPU twin (cwick_metal_gemv) has the same per-block scale
         * semantics. On failure this keeps the scalar path. */
        if (cwick_metal_gemv(v->data, v->scale_inv, x, out,
                            v->cols, v->rows, v->ncb, 1, v->cols) != 0) {
            g_moe_single_metal++;
            return;
        }
        g_moe_single_cpu++;
        cwick_gemv_fp8(out, v->cols, v->rows, x, v->data, v->scale_inv, v->ncb);
    }
    else if (v->dtype == CWICK_ST_BF16)
        cwick_gemv_bf16(out, v->cols, v->rows, x, v->data);
    else
        cwick_gemm_f32(out, 1, v->cols, v->rows, x, (const float *)v->data);
}

/* Both of the above are the decode step's real hotspots, so they carry the
 * profiler directly -- every return path, every caller, no per-call-site
 * bookkeeping. The cost is two clock reads per matrix (CWICK_PROFILE unset:
 * a load and a test inside cwick_prof_add). */
static cwick_status expert_view(const cwick_spine *spine,
                               const cwick_model_spec *spec, int64_t layer,
                               int64_t expert, const char *matrix,
                               uint8_t *staging, size_t staging_off,
                               size_t staging_cap,
                               cwick_expert_view *v, cwick_error *err) {
    uint64_t t0 = cwick_prof_ns();
    cwick_status st = expert_view_impl(spine, spec, layer, expert, matrix,
                                      staging, staging_off, staging_cap, v,
                                      err);
    cwick_prof_add(CWICK_PH_STREAM, cwick_prof_ns() - t0);
    return st;
}

static void expert_gemv(float *out, const cwick_expert_view *v, const float *x) {
    uint64_t t0 = cwick_prof_ns();
    expert_gemv_impl(out, v, x);
    cwick_prof_add(CWICK_PH_GEMV, cwick_prof_ns() - t0);
}

/* ------------------------------------------------------------------ */
/* Whole-layer expert fetch                                           */
/*                                                                    */
/* The moment the router has run, every matrix this layer will touch is  */
/* known: T experts x 3 projections, all of them needed, none of them    */
/* speculative. Reading them one blocking pread at a time leaves the SSD  */
/* idle for the whole latency of each request: measured across the 62     */
/* shards of this checkpoint, one 8 MiB read in flight sustains 6.66      */
/* GB/s while a dozen in flight sustain 9.16 GB/s. Since these reads are  */
/* the largest single phase of a decode step, that difference is worth     */
/* having — and there is no prediction to get wrong.                     */
/* ------------------------------------------------------------------ */
typedef struct {
    int64_t expert;
    int64_t m_idx;
    cwick_st_tensor t;
    cwick_st_tensor st;
    bool have_scale;
    uint8_t *blob;      /* cache slot holding the bytes */
    bool needs_read;    /* slot reserved, contents not filled yet */
    bool owned;         /* blob came from malloc, must be freed by the caller */
    int rc;             /* 0 = ok */
} cwick_fetch_req;

typedef struct {
    const cwick_spine *spine;
    int64_t layer;
    cwick_fetch_req *reqs;
    const int64_t *order;   /* dispatch order: only the items that need a read */
    cwick_error *err;
} cwick_fetch_ctx;

static const char *k_expert_matrices[3] = { "gate_proj", "up_proj", "down_proj" };

/* Upper bound on the dispatch order list: a decode layer is top-8 x 3 = 24
 * items, the batched prefill window is GW(32) x 3 = 96. Anything larger
 * falls back to dispatching the raw range (see layer_fetch_ex). */
#define CWICK_FETCH_ORDER_MAX 128

/* The FP8 scale side is 2 KiB per expert matrix. Streaming it the way the
 * weights are streamed costs a *second* random pread on every miss — half of
 * all read syscalls in a decode step — for a tensor class whose entire
 * population (42 layers x 288 experts x 3) is only ~72 MiB. Copy it out of
 * the shard mapping instead: the mapping already exists for the resident
 * spine, and 72 MiB of page cache against a 128 GB box is not what the
 * F_NOCACHE discipline is protecting. Returns false when there is no mapping
 * and the caller should fall back to a read. */
static bool scale_from_map(const cwick_st_tensor *st, void *dst) {
    if (!st || !st->data || st->data_bytes == 0) return false;
    memcpy(dst, st->data, st->data_bytes);
    return true;
}

/* Fill one slice of the pending reads. Each item owns its own destination
 * buffer and the stream's byte counters are mutex-guarded, so slices are
 * independent and run concurrently. */
uint64_t g_fetch_pread_ns;   /* summed over reader threads: time in pread() */
uint64_t g_fetch_dispatch_ns;/* wall time of the pool fork-join             */
uint64_t g_fetch_alloc_ns;   /* wall time of slot reservation (main thread)  */
static void fetch_range(void *argp, int64_t lo, int64_t hi) {
    cwick_fetch_ctx *c = (cwick_fetch_ctx *)argp;
    for (int64_t i = lo; i < hi; i++) {
        cwick_fetch_req *r = &c->reqs[c->order[i]];
        if (!r->needs_read) continue;
        uint64_t _t0 = cwick_prof_ns();
        if (cwick_stream_read(c->spine->stream, &r->t, r->blob, c->err) != CWICK_OK) {
            r->rc = -1;
            continue;
        }
        __atomic_fetch_add(&g_fetch_pread_ns, cwick_prof_ns() - _t0,
                           __ATOMIC_RELAXED);
        if (r->have_scale && r->st.data_bytes &&
            !scale_from_map(&r->st, r->blob + r->t.data_bytes) &&
            cwick_stream_read(c->spine->stream, &r->st,
                             r->blob + r->t.data_bytes, c->err) != CWICK_OK)
            r->rc = -1;
    }
}

/* Resolve, reserve and read every matrix this layer needs. Slot allocation
 * happens here on one thread (the cache index is not thread-safe) and only
 * the preads are spread over the pool. Returns CWICK_OK with reqs[].blob
 * pointing at usable bytes; anything else means the caller should fall back
 * to the serial path.
 *
 * The pool dispatch reads the whole layer with all twelve workers and then
 * joins. Two alternatives were measured against it and both lost:
 *   - one reader thread per layer, waited for up front: 851 ms/step of reads
 *     against 695 for the twelve. A single 8 MiB pread sustains 5.9 GB/s on
 *     this checkpoint, twelve sustain 3.9 GB/s *per thread* but finish the
 *     layer sooner because the layer is only ~8 reads deep and the extra
 *     requests overlap the latency instead of queueing behind it.
 *   - the same reader thread consumed item by item, so the reads overlapped
 *     the GEMVs: 1512 ms/step. The reader and the twelve GEMV workers land on
 *     the same cores, and a fork-join GEMV is paced by its slowest
 *     participant, so the concurrency cost more in GEMV time (458 -> 595
 *     ms/step) than it saved in read time. QoS on the reader thread made no
 *     difference. Creating and joining a thread per layer also costs ~200
 *     ms/step on its own, which showed up as unexplained "rest" time. */
/* Where a prefill staging blob lives.
 *
 * The GPU can only bind a weight blob that sits in the GPU-visible arena, so
 * running the batched prefill GEMV on the GPU (see cwick_gemv_fp8_rows in
 * tensor.c) requires the staging allocation to come from there too. Off by
 * default: arena allocation once turned a 10-token prefill from ~43 s into one
 * that never finished, and that was measured before the arena had its
 * size-classed LIFO. The two halves opt in together so they can never disagree
 * about where a blob lives -- and if they somehow do, cwick_metal_gemv refuses
 * and the caller redoes the group on the CPU, so a mistake costs a fallback,
 * not a wrong answer. */
/* Largest prefill group sent to the GPU as one pair of command buffers,
 * in token rows. The staging blocks are rows x (RH + 3*I + RH) floats, so
 * 8192 rows is ~470 MiB -- affordable against a 102 GiB ceiling, and it
 * covers a 1024-token chunk at top-8. Bigger chunks fall back to the CPU,
 * where a long prompt already amortises each matrix over many rows.
 *
 * MEASURED: this loses. Same 29-token prompt, three configurations:
 *
 *   staging memory   GEMV        fetch    pread(summed)  compute   wall
 *   ordinary malloc  CPU NEON     2596 ms    33506 ms    10163 ms  12760 ms
 *   arena            CPU NEON     3855 ms    41240 ms     9523 ms  13378 ms
 *   arena            GPU batch    7982 ms    84608 ms     7822 ms  15059 ms
 *
 * The batch does what it was built to do -- compute drops 23% (10163 -> 7822),
 * which is the whole point of batching instead of one commit-and-wait per
 * matrix. But the SSD gets roughly twice as slow while it runs, and that costs
 * more than the GEMV saves. Splitting the two effects (CWICK_PREFILL_BATCH_ROWS=0
 * keeps the arena but drops the batch) shows arena memory alone accounts for
 * only about a fifth of the slowdown; the rest is the GPU work itself. So the
 * switch stays off by default and lives on only as the experiment above.
 *
 * 2026-10-16, re-measured on the released checkpoint with a 403-token prompt
 * (max_new 4, --backend metal):
 *
 *   default (malloc staging, CPU fp8)    prefill ~159 s
 *   arena + CPU (METAL=1, BATCH_ROWS=0)  prefill ~142 s   <-- fastest
 *   arena + Metal batch (METAL=1)        SIGSEGV in the first layer's MoE
 *
 * The batched GPU path now crashes outright, so the switch is not merely a
 * net loss but unusable; do not turn it on without fixing the staging lifetime
 * first, and expect it to lose anyway: pread dominates a prefill (~41 s of the
 * ~50 s in the table above), so even a GPU that halved the GEMV could not buy
 * back more than a few percent. The arena alone is worth ~10% and is the part
 * worth taking. */
#define CWICK_PREFILL_BATCH_ROWS 8192

/* Overridable so the two effects can be separated: CWICK_PREFILL_BATCH_ROWS=0
 * keeps the arena-backed staging buffers (and whatever they cost the SSD) but
 * sends every group to the CPU, which isolates "arena memory makes pread slow"
 * from "a busy GPU makes pread slow". */
static int64_t prefill_batch_rows(void) {
    static int64_t v = -1;
    if (v < 0) {
        const char *e = getenv("CWICK_PREFILL_BATCH_ROWS");
        v = (e && *e) ? atoll(e) : CWICK_PREFILL_BATCH_ROWS;
    }
    return v;
}

static int g_prefill_metal = -1;
static bool prefill_metal(void) {
    if (g_prefill_metal < 0)
        g_prefill_metal = cwick_env_true("CWICK_PREFILL_METAL") ? 1 : 0;
    return g_prefill_metal == 1;
}

static void *prefill_blob_alloc(size_t bytes) {
    return prefill_metal() ? cwick_shmem_alloc(bytes) : cwick_xmalloc(bytes);
}
static void prefill_blob_free(void *p) {
    if (prefill_metal()) cwick_shmem_free(p);
    else free(p);
}

/* May a Metal batch bind this matrix?
 *
 * The batch adopts a weight pointer as a no-copy view, and metal_objc.m's
 * view cache keeps that wrapper for the life of the process. So the bytes
 * behind the pointer must outlive the layer: arena blocks do (a cache hit, a
 * reserve, or the arena-allocated staging fallback), and the mmap'd spine
 * does (the shared expert). A malloc'd staging blob does NOT -- it is freed
 * at the end of the layer, the wrapper dangles, and the next allocation to
 * land on that address is read through the stale view. That is the crash
 * class behind "SIGSEGV in the first layer's MoE" (2026-10-16) and the rare
 * objc_release faults chased in 2026-10-18: both had a freed blob behind a
 * live GPU binding. Refusing the matrix drops the whole group to the CPU
 * path, which is correct and merely slower.
 *
 * `owned` alone is not enough to refuse: with CWICK_PREFILL_METAL=1 the
 * fallback blob comes from the arena and is stable. */
static bool batch_safe(const cwick_expert_view *v) {
    return !v->owned || prefill_metal();
}

/* Batched decode across sequences (CWICK_BATCH_DECODE>1) sends several rows
 * through this same group MoE, and it wants the GPU batch for the same reason
 * decode does: the blobs it fetched are arena-backed, so binding them by
 * offset is free, and the group is a handful of rows rather than prefill's
 * tens of thousands of blob allocate/free cycles. That last difference is why
 * this is a separate switch from CWICK_PREFILL_METAL -- turning the *prefill*
 * staging arena on measured slower, while a decode group of 2-4 rows is
 * exactly the shape the batch was written for. */
static int g_group_metal = -1;
static bool group_metal(void) {
    if (g_group_metal < 0) {
        const char *e = getenv("CWICK_GROUP_METAL");
        g_group_metal = (e && *e) ? (atoi(e) != 0 ? 1 : 0) : 1;
    }
    return g_group_metal == 1;
}

static cwick_status layer_fetch_ex(const cwick_spine *spine,
                                  const cwick_model_spec *spec, int64_t layer,
                                  const int64_t *ids, int64_t T,
                                  const bool *admit,
                                  cwick_fetch_req *reqs, cwick_error *err) {
    for (int64_t e = 0; e < T; e++) {
        for (int m = 0; m < 3; m++) {
            cwick_fetch_req *r = &reqs[e * 3 + m];
            char n[192], sn[192];
            memset(r, 0, sizeof(*r));
            r->expert = ids[e];
            r->m_idx = m;
            cwick_expert_saf_name(spec, layer, ids[e], k_expert_matrices[m],
                                  n, sizeof(n));
            if (cwick_safetensors_get(spine->saf, n, &r->t, err) != CWICK_OK)
                return CWICK_ERR_IO;
            if (r->t.dtype == CWICK_ST_F8_E4M3) {
                snprintf(sn, sizeof(sn), "%s_scale_inv", n);
                if (cwick_safetensors_get(spine->saf, sn, &r->st, err) != CWICK_OK)
                    return CWICK_ERR_IO;
                r->have_scale = true;
            }
            size_t need = r->t.data_bytes + (r->have_scale ? r->st.data_bytes : 0);

            size_t cbytes = 0;
            const void *cblob = cwick_stream_expert_cached(spine->stream, layer,
                                                          ids[e], m, &cbytes);
            if (cblob && cbytes >= need) { r->blob = (uint8_t *)cblob; continue; }

            if (!admit || admit[e]) {
                uint64_t _ta0 = cwick_prof_ns();
                uint8_t *slot = cwick_stream_expert_reserve(spine->stream, layer,
                                                           ids[e], m, need);
                g_fetch_alloc_ns += cwick_prof_ns() - _ta0;
                if (slot) {
                    r->blob = slot;
                    r->needs_read = true;
                } else {
                    /* Budget refused or the ghost admission gate held the key
                     * back: this expert is not worth a pinned slot right now
                     * (one-shot prefill route, or a cold key that has not yet
                     * earned its place). Read it into an owned buffer that is
                     * freed after the GEMV instead of pinning bytes that would
                     * evict a hotter neighbour. */
                    uint8_t *own = prefill_blob_alloc(need);
                    if (!own) return CWICK_ERR_MEMORY;
                    r->blob = own;
                    r->needs_read = true;
                    r->owned = true;
                }
            } else {
                /* Not worth a cache slot: during prefill a dozen tokens can
                 * touch thirty-odd experts per layer, and admitting all of
                 * them would evict the experts the decode steps actually reuse.
                 * Same reasoning about malloc-vs-arena as in
                 * expert_view_into — this is ordinary malloc on purpose. */
                uint8_t *slot = prefill_blob_alloc(need);
                if (!slot) return CWICK_ERR_MEMORY;
                r->blob = slot;
                r->needs_read = true;
                r->owned = true;
            }
        }
    }

    /* ---- compact the reads, then split them one per participant -------
     * A layer routes ~11 of its 24 matrices as misses (54% hit rate), and
     * the misses land wherever the router put them. Dispatching the raw
     * [0, T*3) range hands every participant two *adjacent* items, so a
     * participant whose pair are both cache hits does nothing while
     * another's pair of misses sets the wall: measured 336 ms/step of
     * fork-join against 156 ms of actual read work — 47% utilisation, and
     * the wall was two reads even though only ~11 of 12 participants had
     * any work at all.
     *
     * Listing the misses first and dispatching only those gives every
     * participant at most one read, which is the floor for this shape:
     * the wall becomes the slowest single read instead of the slowest
     * pair, and the participants that would have idled now never wake.
     * The reqs array itself is left in [expert][matrix] order — the GEMV
     * and the release paths index it directly — so the compaction lives
     * in a side list the readers follow.
     *
     * MEASURED, and it LOSES (A/B, same prompt, 19 decode steps each):
     *   compact on   stream 574 ms/step   pread summed 4024 ms/step
     *   compact off  stream 534 ms/step   pread summed 2690 ms/step
     * One read per participant issues the whole layer's misses at once and
     * then lets the SSD queue drain; the raw range hands each participant
     * two adjacent items, so the participants holding a second miss keep a
     * request queued behind the first and the drive sees a deeper queue for
     * longer. On this NVMe sustained depth is worth more than balance, so
     * the compaction stays off by default (CWICK_NO_FETCH_COMPACT=0 forces
     * it back on for re-measurement). */
    int64_t order[CWICK_FETCH_ORDER_MAX];
    int64_t nreads = 0;
    const int64_t nreqs = T * 3;
    static int compact = -1;
    if (compact < 0)
        compact = cwick_env_true("CWICK_FETCH_COMPACT") ? 1 : 0;
    if (compact && nreqs <= CWICK_FETCH_ORDER_MAX) {
        for (int64_t i = 0; i < nreqs; i++)
            if (reqs[i].needs_read) order[nreads++] = i;
    } else if (!compact) {
        for (int64_t i = 0; i < nreqs; i++) order[i] = i;
        nreads = nreqs;
    } else {
        /* Cannot happen for this checkpoint (max 96), but a dropped read
         * would hand the GEMV an unfilled blob, so fall back to the raw
         * range rather than risk it. */
        for (int64_t i = 0; i < nreqs; i++) order[i] = i;
        nreads = nreqs;
    }

    cwick_fetch_ctx ctx;
    ctx.spine = spine;
    ctx.layer = layer;
    ctx.reqs = reqs;
    ctx.order = order;
    ctx.err = err;
    uint64_t _td0 = cwick_prof_ns();
    cwick_pool_for_items(nreads, &ctx, fetch_range);
    g_fetch_dispatch_ns += cwick_prof_ns() - _td0;

    for (int64_t i = 0; i < T * 3; i++) {
        if (reqs[i].rc == 0) continue;
        /* A half-filled slot must not survive to be served as a cache hit. */
        if (reqs[i].needs_read && !reqs[i].owned)
            cwick_stream_expert_abort(spine->stream, layer, reqs[i].expert,
                                     reqs[i].m_idx);
        return CWICK_ERR_IO;
    }
    return CWICK_OK;
}

static cwick_status layer_fetch(const cwick_spine *spine,
                               const cwick_model_spec *spec, int64_t layer,
                               const int64_t *ids, int64_t T,
                               cwick_fetch_req *reqs, cwick_error *err) {
    return layer_fetch_ex(spine, spec, layer, ids, T, NULL, reqs, err);
}

/* Give the bytes back: owned buffers to free(), abandoned reservations to the
 * cache that made them. Harmless to call on a fetch that never had readers. */
static void layer_fetch_release(const cwick_spine *spine, int64_t layer,
                                cwick_fetch_req *reqs, int64_t n) {
    if (!reqs) return;
    for (int64_t i = 0; i < n; i++) {
        if (reqs[i].owned) { prefill_blob_free(reqs[i].blob); continue; }
        if (reqs[i].rc != 0 && reqs[i].needs_read)
            cwick_stream_expert_abort(spine->stream, layer, reqs[i].expert,
                                     reqs[i].m_idx);
    }
}

/* Build an expert view over bytes a fetch already placed in memory. */
static void fetch_view(cwick_expert_view *v, const cwick_fetch_req *r) {
    memset(v, 0, sizeof(*v));
    v->data = r->blob;
    v->rows = r->t.rows;
    v->cols = r->t.cols;
    v->dtype = r->t.dtype;
    v->data_bytes = r->t.data_bytes;
    v->scale_bytes = r->have_scale ? r->st.data_bytes : 0;
    v->scale_inv = r->have_scale
                       ? (const float *)(r->blob + r->t.data_bytes) : NULL;
    v->ncb = r->have_scale ? r->st.cols : 0;
    v->owned = r->owned;
}

/* Full MoE forward for one token. `hidden` is the post-normed MLP input.
 * Writes the MoE output (routed + shared) into `out`.
 *
 * Two layouts are supported:
 *   - GLM-5.3 (released checkpoint): each routed expert maps hidden ->
 *     moe_intermediate -> hidden directly, so the routed output is summed
 *     in hidden space and there is no latent projection or latent norm.
 *   - the legacy/converted layout, which routes through a latent
 *     `routed_expert_down_proj` / `routed_expert_up_proj` pair. That path
 *     exists only when those tensors are present, so the real checkpoint
 *     never silently runs the wrong math. */
/* SwiGLU used by the released GLM-5.3 checkpoint: clamped silu(gate) * up.
 * Falls back to the legacy situ activation only when the config carries no
 * swiglu_limit (converted/legacy models). */
static void moe_act(const cwick_model_spec *spec, float *out, const float *gate,
                    const float *up, int64_t n) {
    if (spec->swiglu_limit > 0.0f)
        cwick_swiglu_clamp(out, gate, up, n, spec->swiglu_limit);
    else if (spec->act_silu) {
        /* plain SwiGLU: silu(gate) * up, for a config that asks for
         * hidden_act=silu without a clamp */
        for (int64_t i = 0; i < n; i++)
            out[i] = gate[i] * (1.0f / (1.0f + expf(-gate[i]))) * up[i];
    } else
        cwick_situ(out, gate, up, n, spec->activation_situ_beta,
                  spec->activation_situ_linear_beta);
}

/* ------------------------------------------------------------------ */
/* Fused routed-expert group: two dispatches per layer, not 3T         */
/*                                                                     */
/* A decode step runs the expert GEMV 42 x 8 x 3 = 1008 times, and    */
/* every one of them used to be its own fork-join. Waking twelve       */
/* workers and then the caller costs more than the 280 us kernel it    */
/* starts -- measured as 696 us per dispatch against a 278 us kernel.  */
/* The three projections are genuinely sequential (down needs the      */
/* activation), but the eight experts are not, and neither are the     */
/* output rows inside one matrix. So the parallelism moves up one      */
/* level: the caller dispatches once over (expert, row) for gate+up,   */
/* applies the activation, and dispatches once more for down.          */
/*                                                                     */
/* The activation is folded into the first kernel rather than left as  */
/* its own pass, because a worker that just produced gate[r] and       */
/* up[r] still has them in L1.                                         */
/* ------------------------------------------------------------------ */
typedef struct {
    const cwick_expert_view *vg;  /* [T] gate_proj */
    const cwick_expert_view *vu;  /* [T] up_proj   */
    const float *in;             /* [k]           */
    float *gate;                 /* [T][rows]     */
    float *up;                   /* [T][rows]     */
    float *act;                  /* [T][rows]     */
    const cwick_model_spec *spec;
    int64_t T, rows;
} fused_gu_arg;

/* Walk [i0, i1) of the flattened (expert, row) space, one expert at a time
 * so the input vector and the weight tile stay hot. Both kernels below are
 * FP8-only by construction: the caller only takes this path when every view
 * in the group is CWICK_ST_F8_E4M3, because cwick_gemv_int8 has no sub-range
 * form and the --spine int8 corpus is not the path being optimized. */
#define FUSED_FOR_EACH_ROW(i0, i1, ROWS)                                       \
    int64_t _e = (i0) / (ROWS), _r = (i0) % (ROWS), _at = (i0);                \
    int64_t _take;                                                             \
    for (; _at < (i1); _at += _take) {                                         \
        _take = (ROWS) - _r;                                                   \
        if (_take > (i1) - _at) _take = (i1) - _at;

#define FUSED_ROW_NEXT(ROWS)                                                   \
        _r += _take;                                                           \
        if (_r >= (ROWS)) { _e++; _r = 0; }                                    \
    }

static void fused_gu_range(void *argp, int64_t i0, int64_t i1) {
    fused_gu_arg *a = (fused_gu_arg *)argp;
    const int64_t rows = a->rows;

    /* The output pointer handed to the kernel must be the BASE of expert
     * e's slab, not the first row this worker owns: gemv_fp8_range() writes
     * out[c] for c in [c0, c1) with an absolute index. Passing a pointer
     * already advanced by _r applies the offset twice and lands the rows in
     * the wrong place. */
    FUSED_FOR_EACH_ROW(i0, i1, rows)
        const cwick_expert_view *g = &a->vg[_e], *u = &a->vu[_e];
        float *gp = a->gate + (size_t)_e * (size_t)rows;
        float *up = a->up + (size_t)_e * (size_t)rows;
        cwick_gemv_fp8_sub(gp, g->cols, g->rows, _r, _r + _take,
                          a->in, g->data, g->scale_inv, g->ncb);
        cwick_gemv_fp8_sub(up, u->cols, u->rows, _r, _r + _take,
                          a->in, u->data, u->scale_inv, u->ncb);
        /* The activation touches what this worker just wrote, so it reads
         * back out of L1 instead of going through DRAM a second time. */
        moe_act(a->spec, a->act + (size_t)_e * (size_t)rows + _r,
                gp + _r, up + _r, _take);
    FUSED_ROW_NEXT(rows)
}

typedef struct {
    const cwick_expert_view *vd;  /* [T] down_proj */
    const float *act;            /* [T][k]        */
    float *y;                    /* [T][rows]     */
    int64_t T, rows;
} fused_d_arg;

static void fused_d_range(void *argp, int64_t i0, int64_t i1) {
    fused_d_arg *a = (fused_d_arg *)argp;
    const int64_t rows = a->rows;

    FUSED_FOR_EACH_ROW(i0, i1, rows)
        const cwick_expert_view *d = &a->vd[_e];
        cwick_gemv_fp8_sub(a->y + (size_t)_e * (size_t)rows,   /* base, see above */
                          d->cols, d->rows, _r, _r + _take,
                          a->act + (size_t)_e * (size_t)d->cols,
                          d->data, d->scale_inv, d->ncb);
    FUSED_ROW_NEXT(rows)
}

void cwick_moe_forward(const cwick_model_spec *spec, const cwick_spine *spine,
                      int64_t layer, const float *hidden, float *out,
                      cwick_error *err) {
    const int64_t H = spec->hidden_size;
    const int64_t I = spec->moe_intermediate_size;
    const int64_t SI = spec->moe_intermediate_size * spec->num_shared_experts;
    const int64_t RH = spec->routed_expert_hidden_size;
    const int64_t E = spec->num_experts;
    const int64_t T = spec->num_experts_per_token;

    cwick_tensor *t = NULL;
    bool latent_layout = false;
#define TENSOR(name) \
    do { \
        int64_t _ti = cwick_spine_find(spine, name); \
        if (_ti == CWICK_TENSOR_MISSING) { \
            cwick_error_set(err, CWICK_ERR_STATE, "missing tensor %s", name); return; \
        } \
        t = &spine->tensors[(size_t)_ti]; \
    } while (0)

    /* ---- router ---- */
    float *logits = cwick_xmalloc((size_t)E * sizeof(float));
    {
        char name[128];
        snprintf(name, sizeof(name), "layers.%lld.block_sparse_moe.gate.weight", layer);
        TENSOR(name);
        cwick_linear(logits, hidden, 1, H, E, t);
    }
    float *scores = cwick_xmalloc((size_t)E * sizeof(float));
    for (int64_t i = 0; i < E; i++) scores[i] = 1.0f / (1.0f + expf(-logits[i]));

    /* correction bias added before top-k; a checkpoint without one
     * simply selects on the plain sigmoid scores */
    float *choice = cwick_xmalloc((size_t)E * sizeof(float));
    {
        char name[128];
        snprintf(name, sizeof(name),
                 "layers.%lld.block_sparse_moe.gate.e_score_correction_bias", layer);
        int64_t bi = cwick_spine_find(spine, name);
        const float *bias = bi != CWICK_TENSOR_MISSING
                            ? (const float *)spine->tensors[(size_t)bi].data
                            : NULL;
        for (int64_t i = 0; i < E; i++)
            choice[i] = scores[i] + (bias ? bias[i] : 0.0f);
    }

    int64_t *ids = cwick_xmalloc((size_t)T * sizeof(int64_t));
    topk_indices(choice, E, T, ids);

    /* Access-trace tap (CWICK_EXPERT_DUMP=path): one "layer expert" line per
     * routed expert per decode step. Used offline to bound how well any
     * pinned-expert cache can possibly do at a given byte budget; compiled
     * in but dormant unless the env var is set. */
    {
        static FILE *dump;
        static int dump_init;
        if (!dump_init) {
            dump_init = 1;
            const char *p = getenv("CWICK_EXPERT_DUMP");
            if (p) dump = fopen(p, "w");
        }
        if (dump) {
            for (int64_t i = 0; i < T; i++)
                fprintf(dump, "%lld %lld\n", (long long)layer, (long long)ids[i]);
        }
    }

    /* Gate feedback: score the authoritative route against the outstanding
     * hint and remember it for the prev-token predictor. This only feeds the
     * per-layer EMA that decides whether speculative *reads* are admitted
     * next pass — it never changes routing. */
    if (spine->stream) {
        cwick_stream_prefetch_score(spine->stream, layer, NULL, 0, ids, T);
        cwick_stream_prefetch_set_prev(spine->stream, layer, ids, T);
    }

    /* weights = renormalized gather of plain scores */
    float *weights = cwick_xmalloc((size_t)T * sizeof(float));
    {
        double sum = 0.0;
        for (int64_t i = 0; i < T; i++) weights[i] = scores[ids[i]];
        if (spec->moe_renormalize) {
            for (int64_t i = 0; i < T; i++) sum += weights[i];
            float inv = 1.0f / (float)(sum + 1.0e-20);
            for (int64_t i = 0; i < T; i++) weights[i] *= inv;
        }
        if (spec->routed_scaling_factor != 1.0f) {
            for (int64_t i = 0; i < T; i++) weights[i] *= spec->routed_scaling_factor;
        }
    }
    if (cwick_env_true("CWICK_TRACE") && layer == 3) {
        fprintf(stderr, "    moe L3 ids =");
        for (int64_t i = 0; i < T; i++) fprintf(stderr, " %lld", (long long)ids[i]);
        fprintf(stderr, "\n    moe L3 weights =");
        for (int64_t i = 0; i < T; i++) fprintf(stderr, " %.5g", weights[i]);
        fprintf(stderr, "\n");
    }

    /* ---- routed experts (full bank, exact top-k) ----
     * Which layout the checkpoint uses is decided by the weights themselves,
     * not by a guess: GLM-5.3 has no routed_expert_* projections. */
    {
        char probe[128];
        snprintf(probe, sizeof(probe),
                 "layers.%lld.block_sparse_moe.routed_expert_down_proj.weight", layer);
        latent_layout = cwick_spine_find(spine, probe) != CWICK_TENSOR_MISSING;
    }

    float *gate_buf = cwick_xmalloc((size_t)I * sizeof(float));
    float *up_buf = cwick_xmalloc((size_t)I * sizeof(float));
    float *act_buf = cwick_xmalloc((size_t)I * sizeof(float));
    float *y_buf = cwick_xmalloc((size_t)RH * sizeof(float));
    float *combined = cwick_xmalloc((size_t)RH * sizeof(float));
    float *routed_in = cwick_xmalloc((size_t)RH * sizeof(float));
    memset(combined, 0, (size_t)RH * sizeof(float));

    if (latent_layout) {
        char name[128];
        snprintf(name, sizeof(name),
                 "layers.%lld.block_sparse_moe.routed_expert_down_proj.weight", layer);
        TENSOR(name);
        cwick_linear(routed_in, hidden, 1, H, RH, t);
    } else {
        /* GLM-5.3: experts consume the post-normed hidden state directly. */
        memcpy(routed_in, hidden, (size_t)H * sizeof(float));
    }

    /* Route-first fetch: reserve every matrix this layer needs, then fill the
     * slots on a reader thread while the GEMVs consume them. */
    cwick_fetch_req *reqs = NULL;
    uint8_t *staging = NULL;
    const size_t staging_cap = (size_t)(3 * I * H) + 64u * 1024u;
    if (spine->is_safetensors && spine->stream &&
        !cwick_env_true("CWICK_PREFETCH")) {
        reqs = cwick_xmalloc((size_t)T * 3 * sizeof(*reqs));
        uint64_t tf0 = cwick_prof_ns();
        cwick_status fs = layer_fetch(spine, spec, layer, ids, T, reqs, err);
        cwick_prof_add(CWICK_PH_STREAM, cwick_prof_ns() - tf0);
        if (fs != CWICK_OK) {
            /* The batch failed part-way. layer_fetch_ex has already aborted
             * the pinned slots it filled, so the only buffers still owned
             * here are the ones it allocated itself -- and those are one full
             * layer's staging bytes, not a rounding error. */
            for (int64_t i = 0; i < T * 3; i++)
                if (reqs[i].owned && reqs[i].blob) {
                    prefill_blob_free(reqs[i].blob);
                    reqs[i].blob = NULL;
                }
            free(reqs);
            reqs = NULL;
        }
    }
    if (!reqs && spine->is_safetensors) {
        /* Fallback: prefetch is enabled, or the byte budget refused the
         * batch. Stage one expert at a time in a reusable buffer instead. */
        staging = cwick_shmem_alloc(staging_cap);
        if (!staging) { cwick_error_set(err, CWICK_ERR_MEMORY, "expert staging"); goto fail; }
    }

    /* Materials for the fused group. Built only when the route-first fetch
     * succeeded AND every matrix in the group is FP8: cwick_gemv_int8 has no
     * sub-range form, and the per-expert loop below stays correct for it. */
    cwick_expert_view *vgs = NULL, *vus = NULL, *vds = NULL;
    float *gate_all = NULL, *up_all = NULL, *act_all = NULL, *y_all = NULL;
    size_t gw = 0;              /* slab stride of gate/up/act: max(I, RH) */
    static int fused_allowed = -1;
    if (fused_allowed < 0)
        fused_allowed = getenv("CWICK_MOE_FUSED") &&
                        atoi(getenv("CWICK_MOE_FUSED")) == 0 ? 0 : 1;
    /* `reqs` must be part of this test, not just the guard on the block
     * below. The fused path runs off buffers that only exist when the
     * route-first fetch succeeded; when there is no stream at all -- any
     * model small enough to sit in RAM, which is what the verification suite
     * uses -- reqs is NULL, the block below is skipped, and a `fused` that
     * stayed true would walk into the dispatch with null slabs. */
    bool fused = fused_allowed != 0 && reqs != NULL;
    if (fused) {
        vgs = cwick_xmalloc((size_t)T * sizeof(*vgs));
        vus = cwick_xmalloc((size_t)T * sizeof(*vus));
        vds = cwick_xmalloc((size_t)T * sizeof(*vds));
        fused = true;
        for (int64_t e = 0; e < T && fused; e++) {
            fetch_view(&vgs[e], &reqs[e * 3 + 0]);
            fetch_view(&vus[e], &reqs[e * 3 + 1]);
            fetch_view(&vds[e], &reqs[e * 3 + 2]);
            if (vgs[e].dtype != CWICK_ST_F8_E4M3 ||
                vus[e].dtype != CWICK_ST_F8_E4M3 ||
                vds[e].dtype != CWICK_ST_F8_E4M3) fused = false;
            /* The fused kernel indexes its slabs with one stride per expert,
             * so it only describes a group whose geometry is uniform. Insist
             * on that -- and on it matching the shapes the caller is going
             * to read back -- rather than carrying per-expert strides: the
             * synthetic test model has a routed width that differs from the
             * down_proj row count, and indexing y_all by vds[0].rows while
             * allocating it by RH walked off the end of the buffer. Anything
             * that does not fit this shape takes the per-expert path below,
             * which is slower and has no such assumption. */
            if (vgs[e].rows != vgs[0].rows || vus[e].rows != vgs[0].rows ||
                vds[e].rows != vds[0].rows || vds[e].cols != vgs[0].rows ||
                vds[0].rows != RH || vgs[0].rows <= 0 ||
                !vgs[e].scale_inv || !vus[e].scale_inv || !vds[e].scale_inv ||
                vgs[e].ncb <= 0 || vus[e].ncb <= 0 || vds[e].ncb <= 0)
                fused = false;
        }
            if (fused) {
            gw = (size_t)(I > RH ? I : RH);
            gate_all = cwick_xmalloc((size_t)T * gw * sizeof(float));
            up_all   = cwick_xmalloc((size_t)T * gw * sizeof(float));
            act_all  = cwick_xmalloc((size_t)T * gw * sizeof(float));
            y_all    = cwick_xmalloc((size_t)T * (size_t)RH * sizeof(float));
            if (!gate_all || !up_all || !act_all || !y_all) fused = false;
        }
    }

    /* ---- shared expert ----------------------------------------------
     * One shared expert per MoE layer: three more FP8 GEMVs (gate, up,
     * down) on every layer, 126 per decode step. They are resident in the
     * mapped spine rather than streamed, so they used to go through
     * cwick_linear -> cwick_gemm_fp8 on the CPU: measured ~190 CPU fp8 GEMVs
     * per step at ~335 us each, i.e. ~63 ms/step of the GEMV phase spent on
     * work the GPU was already sitting next to.
     *
     * gate and up need only `hidden`, so they can be appended to wave one;
     * down needs the activation of the two, which the CPU computes while the
     * routed group's activation is computed, so it rides wave two. Nothing
     * about the arithmetic changes -- and if the batch refuses them, the CPU
     * path below still produces the same answer. */
    float *sg = NULL, *su = NULL, *sh = NULL, *shared_out = NULL;
    const cwick_tensor *t_sg = NULL, *t_su = NULL, *t_sd = NULL;
    bool shared_folded = false;   /* gate/up were dispatched with wave one */
    bool shared_dn_ok  = false;   /* down was dispatched with wave two     */
    if (SI > 0) {
        char nm[160];
        int64_t gi, ui, di;
        sg = cwick_xmalloc((size_t)SI * sizeof(float));
        su = cwick_xmalloc((size_t)SI * sizeof(float));
        sh = cwick_xmalloc((size_t)SI * sizeof(float));
        shared_out = cwick_xmalloc((size_t)H * sizeof(float));
        snprintf(nm, sizeof(nm),
                 "layers.%lld.block_sparse_moe.shared_experts.gate_proj.weight", layer);
        gi = cwick_spine_find(spine, nm);
        snprintf(nm, sizeof(nm),
                 "layers.%lld.block_sparse_moe.shared_experts.up_proj.weight", layer);
        ui = cwick_spine_find(spine, nm);
        snprintf(nm, sizeof(nm),
                 "layers.%lld.block_sparse_moe.shared_experts.down_proj.weight", layer);
        di = cwick_spine_find(spine, nm);
        if (sg && su && sh && shared_out &&
            gi != CWICK_TENSOR_MISSING && ui != CWICK_TENSOR_MISSING &&
            di != CWICK_TENSOR_MISSING) {
            t_sg = &spine->tensors[(size_t)gi];
            t_su = &spine->tensors[(size_t)ui];
            t_sd = &spine->tensors[(size_t)di];
        }
    }

    /* ---- Metal: two command buffers per layer instead of 24 ----------
     * A single expert-matrix dispatch is launch-bound: CWICK_METAL_TIME
     * splits it as enc 5-28 us, commit 3-11 us, waitUntilCompleted
     * 295-724 us against 105-123 us of actual GPU work. At 1080 matrices a
     * step that overhead is the GEMV phase.
     *
     * The routed group is two waves -- gate and up are independent, down
     * consumes both -- so wave one is 16 dispatches in one command buffer
     * and wave two is 8. The activation between them is 16 K elementwise
     * ops, far below what a GPU round trip costs, so it stays on the CPU.
     *
     * Any failure (a weight pointer outside the GPU-visible arena, a buffer
     * that cannot grow) aborts the batch and the whole group is redone by
     * the CPU path below, which overwrites the same slabs. */
    bool metal_fused = false;
    if (fused) {
        const int64_t rows = vgs[0].rows;
        uint64_t tg0 = cwick_prof_ns();
        void *mb = cwick_metal_batch_begin();
        if (mb) {
            metal_fused = true;
            for (int64_t e = 0; e < T && metal_fused; e++) {
                /* A malloc'd staging blob dies at the end of the layer and the
                 * view cache outlives it -- see batch_safe. */
                if (!batch_safe(&vgs[e]) || !batch_safe(&vus[e]) ||
                    !batch_safe(&vds[e])) {
                    metal_fused = false;
                    break;
                }
                if (!cwick_metal_batch_gemv(mb, vgs[e].data, vgs[e].scale_inv,
                                           routed_in,
                                           gate_all + (size_t)e * gw,
                                           vgs[e].cols, vgs[e].rows,
                                           vgs[e].ncb, 1, vgs[e].cols) ||
                    !cwick_metal_batch_gemv(mb, vus[e].data, vus[e].scale_inv,
                                           routed_in,
                                           up_all + (size_t)e * gw,
                                           vus[e].cols, vus[e].rows,
                                           vus[e].ncb, 1, vus[e].cols))
                    metal_fused = false;
            }
            /* Wave one is still open: append the shared expert's gate and up.
             * Both read `hidden` and write their own buffers, so they neither
             * depend on nor disturb the routed group. */
            bool shared_try = false;
            if (metal_fused && t_sg && t_su &&
                t_sg->kind == CWICK_WEIGHT_FP8_E4M3 &&
                t_su->kind == CWICK_WEIGHT_FP8_E4M3 &&
                t_sg->scales && t_su->scales && t_sg->cols > 0) {
                shared_try = true;
                int64_t ncb_gu = t_sg->cols / CWICK_ST_BLOCK;
                if (!cwick_metal_batch_gemv(mb, t_sg->data,
                                           (const float *)t_sg->scales, hidden,
                                           sg, t_sg->cols, t_sg->rows,
                                           ncb_gu, 1, t_sg->cols) ||
                    !cwick_metal_batch_gemv(mb, t_su->data,
                                           (const float *)t_su->scales, hidden,
                                           su, t_su->cols, t_su->rows,
                                           ncb_gu, 1, t_su->cols))
                    metal_fused = false;
            }
            if (metal_fused) metal_fused = cwick_metal_batch_end(mb) != 0;
            else cwick_metal_batch_abort(mb);
            shared_folded = metal_fused && shared_try;
        }
        if (metal_fused) g_moe_group_metal++;
        else             g_moe_group_cpu++;
        if (metal_fused) {
            for (int64_t e = 0; e < T; e++)
                moe_act(spec, act_all + (size_t)e * gw,
                        gate_all + (size_t)e * gw,
                        up_all + (size_t)e * gw, rows);
            if (shared_folded) moe_act(spec, sh, sg, su, SI);

            void *mb2 = cwick_metal_batch_begin();
            bool d_ok = false;
            if (mb2) {
                d_ok = true;
                for (int64_t e = 0; e < T && d_ok; e++)
                    if (!cwick_metal_batch_gemv(mb2, vds[e].data,
                                               vds[e].scale_inv,
                                               act_all + (size_t)e * gw,
                                               y_all + (size_t)e * (size_t)RH,
                                               vds[e].cols, vds[e].rows,
                                               vds[e].ncb, 1, vds[e].cols))
                        d_ok = false;
                if (d_ok && shared_folded && t_sd &&
                    t_sd->kind == CWICK_WEIGHT_FP8_E4M3 && t_sd->scales &&
                    t_sd->cols > 0) {
                    int64_t ncb_d = t_sd->cols / CWICK_ST_BLOCK;
                    if (!cwick_metal_batch_gemv(mb2, t_sd->data,
                                               (const float *)t_sd->scales, sh,
                                               shared_out, t_sd->cols,
                                               t_sd->rows, ncb_d, 1, t_sd->cols))
                        d_ok = false;
                    else
                        shared_dn_ok = true;
                }
                if (d_ok) d_ok = cwick_metal_batch_end(mb2) != 0;
                else cwick_metal_batch_abort(mb2);
            }
            if (!d_ok) {
                /* act_all is already right, so only the down GEMVs are lost */
                fused_d_arg da;
                da.vd = vds; da.act = act_all; da.y = y_all;
                da.T = T; da.rows = vds[0].rows;
                cwick_pool_for((int64_t)T * vds[0].rows, &da, fused_d_range);
            }
        }
        cwick_prof_add(CWICK_PH_GEMV, cwick_prof_ns() - tg0);
    }

    if (fused && !metal_fused) {
            uint64_t tg0 = cwick_prof_ns();
        fused_gu_arg ga;
        ga.vg = vgs; ga.vu = vus; ga.in = routed_in;
        ga.gate = gate_all; ga.up = up_all; ga.act = act_all;
        ga.spec = spec; ga.T = T; ga.rows = vgs[0].rows;
        cwick_pool_for((int64_t)T * vgs[0].rows, &ga, fused_gu_range);

        fused_d_arg da;
        da.vd = vds; da.act = act_all; da.y = y_all;
        da.T = T; da.rows = vds[0].rows;
        cwick_pool_for((int64_t)T * vds[0].rows, &da, fused_d_range);
        cwick_prof_add(CWICK_PH_GEMV, cwick_prof_ns() - tg0);
    }
    if (fused) {
        for (int64_t e = 0; e < T; e++) {
            float w = weights[e];
            const float *y = y_all + (size_t)e * (size_t)RH;
            for (int64_t i = 0; i < RH; i++) combined[i] += w * y[i];
        }
    } else {
        g_moe_group_serial++;
    }

    for (int64_t e = 0; !fused && e < T; e++) {
        int64_t expert = ids[e];
        cwick_expert_view vg, vu, vd;
        if (reqs) {
            fetch_view(&vg, &reqs[e * 3 + 0]);
            fetch_view(&vu, &reqs[e * 3 + 1]);
            fetch_view(&vd, &reqs[e * 3 + 2]);
            expert_gemv(gate_buf, &vg, routed_in);
            expert_gemv(up_buf, &vu, routed_in);
            moe_act(spec, act_buf, gate_buf, up_buf, vg.rows);
            expert_gemv(y_buf, &vd, act_buf);
        } else if (!spine->is_safetensors) {
            /* converted corpus: the legacy fp32 materialization path */
            float *w1 = cwick_xmalloc((size_t)(I * RH) * sizeof(float));
            float *w2 = cwick_xmalloc((size_t)(RH * I) * sizeof(float));
            float *w3 = cwick_xmalloc((size_t)(I * RH) * sizeof(float));
            if (cwick_expert_load(spine, layer, expert, spec, w1, w2, w3, err) != CWICK_OK) {
                free(w3); free(w2); free(w1);
                cwick_shmem_free(staging);
                goto fail;
            }
            cwick_gemm_f32(gate_buf, 1, RH, I, routed_in, w1);
            cwick_gemm_f32(up_buf, 1, RH, I, routed_in, w3);
            moe_act(spec, act_buf, gate_buf, up_buf, I);
            cwick_gemm_f32(y_buf, 1, I, RH, act_buf, w2);
            free(w3); free(w2); free(w1);
        } else {
            if (expert_view(spine, spec, layer, expert, "gate_proj", staging, 0,
                            staging_cap, &vg, err) != CWICK_OK ||
                expert_view(spine, spec, layer, expert, "up_proj", staging,
                            vg.data_bytes + vg.scale_bytes,
                            staging_cap, &vu, err) != CWICK_OK) {
                cwick_shmem_free(staging);
                goto fail;
            }
            expert_gemv(gate_buf, &vg, routed_in);
            expert_gemv(up_buf, &vu, routed_in);
            moe_act(spec, act_buf, gate_buf, up_buf, vg.rows);
            if (expert_view(spine, spec, layer, expert, "down_proj", staging, 0,
                            staging_cap, &vd, err) != CWICK_OK) {
                cwick_shmem_free(staging);
                goto fail;
            }
            expert_gemv(y_buf, &vd, act_buf);
        }
        float w = weights[e];
        for (int64_t i = 0; i < RH; i++) combined[i] += w * y_buf[i];
    }
    free(y_all); free(act_all); free(up_all); free(gate_all);
    free(vds); free(vus); free(vgs);
    cwick_shmem_free(staging);
    /* Everything this layer fetched is finished with once the group GEMV has
     * returned -- the Metal batches above wait, and the CPU path consumes the
     * views in the loop before this point. `layer_fetch_release` frees the
     * buffers the stream had to own (the ones it could not pin) and leaves
     * cached slots where they are; `reqs` itself is per layer.
     *
     * Omitting this was not a small leak: a decode step fetches ~24 matrices
     * per layer and owns most of the ones the pin cache refuses, so each step
     * retained a whole step's staging bytes (~6 GiB on GLM-5.3-Flash). The
     * footprint ran 16 -> 67 -> 117 -> 179 GiB over 24 decoded tokens and the
     * process was killed. See docs/PERF.md (2026-11-07). */
    layer_fetch_release(spine, layer, reqs, T * 3);
    free(reqs);
    reqs = NULL;
    if (cwick_env_true("CWICK_TRACE") && layer == 3) {
        double mn = INFINITY, mx = -INFINITY;
        for (int64_t i = 0; i < RH; i++) { double v = combined[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
        fprintf(stderr, "    moe L3 routed_sum n=%lld min=%.5g max=%.5g\n", (long long)RH, mn, mx);
    }

    /* ---- completion ---- */
    float *routed_normed = cwick_xmalloc((size_t)RH * sizeof(float));
    if (latent_layout && spec->latent_moe_use_norm) {
        char name[128];
        snprintf(name, sizeof(name),
                 "layers.%lld.block_sparse_moe.routed_expert_norm.weight", layer);
        TENSOR(name);
        const uint16_t *wn = (const uint16_t *)t->data;
        cwick_rms_norm(routed_normed, combined, RH, NULL, spec->rms_norm_eps_f);
        for (int64_t i = 0; i < RH; i++) routed_normed[i] *= cwick_bf16_to_f32(wn[i]);
    } else {
        memcpy(routed_normed, combined, (size_t)RH * sizeof(float));
    }

    float *routed_full = cwick_xmalloc((size_t)H * sizeof(float));
    if (latent_layout) {
        char name[128];
        snprintf(name, sizeof(name),
                 "layers.%lld.block_sparse_moe.routed_expert_up_proj.weight", layer);
        TENSOR(name);
        cwick_linear(routed_full, routed_normed, 1, RH, H, t);
    } else {
        /* GLM-5.3: the routed sum is already in hidden space. */
        memcpy(routed_full, routed_normed, (size_t)H * sizeof(float));
    }

    /* shared expert: only the part the GPU batch did not take */
    if (SI > 0 && t_sg && t_su && t_sd) {
        if (!shared_folded) {
            cwick_linear(sg, hidden, 1, H, SI, t_sg);
            cwick_linear(su, hidden, 1, H, SI, t_su);
            moe_act(spec, sh, sg, su, SI);
        }
        if (!shared_dn_ok)
            cwick_linear(shared_out, sh, 1, SI, H, t_sd);
    }

    if (SI > 0 && shared_out)
        for (int64_t i = 0; i < H; i++) out[i] = routed_full[i] + shared_out[i];
    else
        memcpy(out, routed_full, (size_t)H * sizeof(float));
    if (cwick_env_true("CWICK_TRACE") && layer == 3 && shared_out) {
        double mn = INFINITY, mx = -INFINITY;
        for (int64_t i = 0; i < H; i++) { double v = shared_out[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
        fprintf(stderr, "    moe L3 shared n=%lld min=%.5g max=%.5g\n", (long long)H, mn, mx);
    }

    free(shared_out);
    free(sh);
    free(su);
    free(sg);
    free(routed_full);
    free(routed_normed);
    free(combined);
    free(y_buf);
    free(act_buf);
    free(up_buf);
    free(gate_buf);
    free(weights);
    free(ids);
    free(choice);
    free(scores);
    free(logits);
    free(routed_in);
    return;

fail:
    free(combined);
    free(y_buf);
    free(act_buf);
    free(up_buf);
    free(gate_buf);
    free(weights);
    free(ids);
    free(choice);
    free(scores);
    free(logits);
    free(routed_in);
#undef TENSOR
}

/* Dense MLP for layer 0: out = down(situ(gate(x), up(x))). */
void cwick_dense_mlp(const cwick_model_spec *spec, const cwick_spine *spine,
                    int64_t layer, const float *hidden, float *out,
                    cwick_error *err) {
    const int64_t H = spec->hidden_size;
    const int64_t I = spec->intermediate_size;
    cwick_tensor *t = NULL;
#define TENSOR(name) \
    do { \
        int64_t _ti = cwick_spine_find(spine, name); \
        if (_ti == CWICK_TENSOR_MISSING) { \
            cwick_error_set(err, CWICK_ERR_STATE, "missing tensor %s", name); return; \
        } \
        t = &spine->tensors[(size_t)_ti]; \
    } while (0)

    char dname[128];
    float *gate = cwick_xmalloc((size_t)I * sizeof(float));
    float *up = cwick_xmalloc((size_t)I * sizeof(float));
    float *act = cwick_xmalloc((size_t)I * sizeof(float));
    snprintf(dname, sizeof(dname), "layers.%lld.mlp.gate_proj.weight", layer);
    TENSOR(dname);
    if (cwick_env_true("CWICK_TRACE")) {
        fprintf(stderr, "  dense L%lld I=%lld kind=%d scales=%p rows=%lld cols=%lld\n",
                layer, I, (int)t->kind, (void *)t->scales, t->rows, t->cols);
    }
    cwick_linear(gate, hidden, 1, H, I, t);
    snprintf(dname, sizeof(dname), "layers.%lld.mlp.up_proj.weight", layer);
    TENSOR(dname);
    cwick_linear(up, hidden, 1, H, I, t);
    moe_act(spec, act, gate, up, I);
    if (cwick_env_true("CWICK_TRACE")) {
        int n1 = 0, n2 = 0, n3 = 0;
        for (int64_t i = 0; i < I; i++) {
            if (isnan(gate[i])) n1++;
            if (isnan(up[i])) n2++;
            if (isnan(act[i])) n3++;
        }
        fprintf(stderr, "  dense L%lld nan gate=%d up=%d act=%d limit=%.3f\n",
                layer, n1, n2, n3, spec->swiglu_limit);
    }
    snprintf(dname, sizeof(dname), "layers.%lld.mlp.down_proj.weight", layer);
    TENSOR(dname);
    cwick_linear(out, act, 1, I, H, t);
    free(act);
    free(up);
    free(gate);
#undef TENSOR
}

/* ------------------------------------------------------------------ */
/* Batched MoE forward (prefill)                                      */
/* ------------------------------------------------------------------ */
/* Read one expert matrix (and its FP8 scale side, if any) for a batched
 * prefill. Cache hits are always reusable. On a miss, recent prompt routes
 * are admitted into the byte-budgeted cache so the first decode steps start
 * hot; older/cold routes stay ephemeral and are released immediately. This
 * preserves the layer-major compute order and never changes routing. */
static cwick_status expert_view_into(const cwick_spine *spine,
                                    const cwick_model_spec *spec, int64_t layer,
                                    int64_t expert, const char *matrix,
                                    bool admit_cache, cwick_expert_view *v,
                                    cwick_error *err) {
    char n[192], sn[192];
    cwick_expert_saf_name(spec, layer, expert, matrix, n, sizeof(n));
    cwick_st_tensor t, st;
    memset(&st, 0, sizeof(st));
    if (cwick_safetensors_get(spine->saf, n, &t, err) != CWICK_OK) return CWICK_ERR_IO;

    bool have_scale = false;
    if (t.dtype == CWICK_ST_F8_E4M3) {
        snprintf(sn, sizeof(sn), "%s_scale_inv", n);
        if (cwick_safetensors_get(spine->saf, sn, &st, err) != CWICK_OK)
            return CWICK_ERR_IO;
        have_scale = true;
    }

    memset(v, 0, sizeof(*v));
    v->rows = t.rows;
    v->cols = t.cols;
    v->dtype = t.dtype;
    v->data_bytes = t.data_bytes;
    v->scale_bytes = have_scale ? st.data_bytes : 0;

    size_t need = t.data_bytes + (have_scale ? st.data_bytes : 0);
    if (!spine->stream) {
        /* no reader: point straight at the mapping (the GEMV copies into its
         * scratch buffer when the mapping is not GPU-visible) */
        v->data = t.data;
        v->scale_inv = have_scale ? (const float *)st.data : NULL;
        v->ncb = have_scale ? st.cols : 0;
        return CWICK_OK;
    }

    int64_t m_idx = strcmp(matrix, "gate_proj") == 0 ? 0 :
                    strcmp(matrix, "up_proj") == 0 ? 1 : 2;
    size_t cached_bytes = 0;
    const void *cached = cwick_stream_expert_cached(spine->stream, layer, expert,
                                                   m_idx, &cached_bytes);
    if (cached && cached_bytes >= need) {
        v->data = cached;
        v->scale_inv = have_scale ? (const float *)(cached + t.data_bytes) : NULL;
        v->ncb = have_scale ? st.cols : 0;
        return CWICK_OK;
    }

    if (admit_cache) {
        const void *slot = cwick_stream_expert_put_read(
            spine->stream, layer, expert, m_idx, &t, have_scale ? &st : NULL, err);
        if (slot) {
            v->data = slot;
            v->scale_inv = have_scale ? (const float *)(slot + t.data_bytes) : NULL;
            v->ncb = have_scale ? st.cols : 0;
            return CWICK_OK;
        }
    }

    /* Deliberately NOT cwick_shmem_alloc.
     *
     * GPU-visible (Metal arena) memory is what makes the *decode* path fast:
     * the expert blob the stream reader just filled is already addressable by
     * the GPU, so the GEMV binds it with an offset instead of copying 8 MiB.
     * Prefill is the opposite shape: it allocates and frees a blob per
     * (expert, matrix) — tens of thousands of 8 MiB cycles per request — and
     * at that rate arena allocation dominates everything else. Measured on
     * this machine, a 10-token prefill went from ~43 s with ordinary malloc
     * to >200 s (never finishing) with arena memory.
     *
     * So prefill pays one 8 MiB scratch copy per GEMV instead. Decode, where
     * the same blob is reused every step, keeps the zero-copy arena. */
    uint8_t *dst = prefill_blob_alloc(need);
    if (!dst) {
        cwick_error_set(err, CWICK_ERR_MEMORY, "expert prefill buffer");
        return CWICK_ERR_MEMORY;
    }
    if (cwick_stream_read(spine->stream, &t, dst, err) != CWICK_OK) {
        prefill_blob_free(dst); return CWICK_ERR_IO;
    }
    if (have_scale) {
        if (cwick_stream_read(spine->stream, &st, dst + t.data_bytes, err) != CWICK_OK) {
            prefill_blob_free(dst); return CWICK_ERR_IO;
        }
    }
    v->data = dst;
    v->scale_inv = have_scale ? (const float *)(dst + t.data_bytes) : NULL;
    v->ncb = have_scale ? st.cols : 0;
    v->owned = true;
    return CWICK_OK;
}

static void expert_view_release(cwick_expert_view *v) {
    /* `owned` is set only by the ephemeral prefill staging buffer, which is
     * deliberately ordinary malloc (see expert_view_into) — so it must be
     * released with free(), not returned to the Metal arena. */
    if (v && v->owned && v->data) prefill_blob_free((void *)v->data);
    if (v) memset(v, 0, sizeof(*v));
}

/* Number of prompt-tail token-equivalents to admit during prefill. This is
 * opt-in: filling tens of GiB before the first decode trades a large amount
 * of allocator/SSD work for uncertain route locality, and measurements on the
 * 128 GiB M3 Max were worse by default. CWICK_PREFILL_CACHE_TOKENS=N enables
 * the experiment with an explicit recency window. When N is larger than the
 * calculated budget, clamp to what the byte cache can actually hold. */
static int64_t prefill_cache_tokens(const cwick_model_spec *spec,
                                    const cwick_spine *spine) {
    if (!spine || !spine->stream) return 0;
    const char *override = getenv("CWICK_PREFILL_CACHE_TOKENS");
    if (!override || !*override) return 0;
    char *end = NULL;
    unsigned long long requested = strtoull(override, &end, 10);
    if (end == override || *end != '\0') return 0;
    if (requested == 0) return 0;
    int64_t requested_tokens = requested > (unsigned long long)INT64_MAX
                                   ? INT64_MAX : (int64_t)requested;

    size_t cap = cwick_stream_cache_cap(spine->stream);
    if (cap == 0) return 0;

    int64_t sparse_layers = 0, first_sparse = -1;
    for (int64_t l = 0; l < spec->num_hidden_layers; l++)
        if (!spec->mlp_dense[l]) {
            sparse_layers++;
            if (first_sparse < 0) first_sparse = l;
        }
    if (sparse_layers <= 0 || first_sparse < 0) return 0;

    char n[192], sn[192];
    cwick_expert_saf_name(spec, first_sparse, 0, "gate_proj", n, sizeof(n));
    cwick_st_tensor t, st;
    memset(&st, 0, sizeof(st));
    cwick_error ignored;
    cwick_error_clear(&ignored);
    if (cwick_safetensors_get(spine->saf, n, &t, &ignored) != CWICK_OK) return 0;
    size_t matrix_bytes = t.data_bytes;
    if (t.dtype == CWICK_ST_F8_E4M3) {
        snprintf(sn, sizeof(sn), "%s_scale_inv", n);
        if (cwick_safetensors_get(spine->saf, sn, &st, &ignored) == CWICK_OK)
            matrix_bytes += st.data_bytes;
    }
    if (matrix_bytes == 0) return 0;
    uint64_t per_token = (uint64_t)matrix_bytes * 3ull *
                         (uint64_t)spec->num_experts_per_token *
                         (uint64_t)sparse_layers;
    if (per_token == 0 || cap < per_token) return 0;
    uint64_t affordable = (uint64_t)cap / per_token;
    uint64_t ntok = affordable < (uint64_t)requested_tokens
                        ? affordable : (uint64_t)requested_tokens;
    return ntok > INT64_MAX ? INT64_MAX : (int64_t)ntok;
}

void cwick_moe_forward_batch(const cwick_model_spec *spec, const cwick_spine *spine,
                            int64_t layer, const float *hidden, int64_t B,
                            float *out, cwick_error *err,
                            cwick_moe_progress_fn progress, void *progress_user) {
    const int64_t H = spec->hidden_size;
    const int64_t I = spec->moe_intermediate_size;
    const int64_t SI = spec->moe_intermediate_size * spec->num_shared_experts;
    const int64_t RH = spec->routed_expert_hidden_size;
    const int64_t E = spec->num_experts;
    const int64_t T = spec->num_experts_per_token;

    cwick_tensor *t = NULL;
    bool latent_layout = false;
#define TENSOR(name) \
    do { \
        int64_t _ti = cwick_spine_find(spine, name); \
        if (_ti == CWICK_TENSOR_MISSING) { \
            cwick_error_set(err, CWICK_ERR_STATE, "missing tensor %s", name); return; \
        } \
        t = &spine->tensors[(size_t)_ti]; \
    } while (0)

    /* ---- per-token router (identical math to cwick_moe_forward) ---- */
    uint64_t _tr0 = cwick_prof_ns();
    float *logits = cwick_xmalloc((size_t)E * B * sizeof(float));
    float *scores = cwick_xmalloc((size_t)E * B * sizeof(float));
    float *choice = cwick_xmalloc((size_t)E * B * sizeof(float));
    int64_t *ids = cwick_xmalloc((size_t)T * B * sizeof(int64_t));
    float *weights = cwick_xmalloc((size_t)T * B * sizeof(float));
    {
        char name[128];
        snprintf(name, sizeof(name), "layers.%lld.block_sparse_moe.gate.weight", layer);
        TENSOR(name);
        for (int64_t b = 0; b < B; b++)
            cwick_linear(logits + b * E, hidden + b * H, 1, H, E, t);
    }
    for (int64_t b = 0; b < B; b++)
        for (int64_t i = 0; i < E; i++)
            scores[b * E + i] = 1.0f / (1.0f + expf(-logits[b * E + i]));
    {
        char name[128];
        snprintf(name, sizeof(name),
                 "layers.%lld.block_sparse_moe.gate.e_score_correction_bias", layer);
        int64_t bi = cwick_spine_find(spine, name);
        const float *bias = bi != CWICK_TENSOR_MISSING
                            ? (const float *)spine->tensors[(size_t)bi].data
                            : NULL;
        for (int64_t b = 0; b < B; b++)
            for (int64_t i = 0; i < E; i++)
                choice[b * E + i] = scores[b * E + i] + (bias ? bias[i] : 0.0f);
    }
    for (int64_t b = 0; b < B; b++)
        topk_indices(choice + b * E, E, T, ids + b * T);
    if (spine->stream) {
        for (int64_t b = 0; b < B; b++) {
            cwick_stream_prefetch_score(spine->stream, layer, NULL, 0, ids + b * T, T);
            cwick_stream_prefetch_set_prev(spine->stream, layer, ids + b * T, T);
        }
    }
    for (int64_t b = 0; b < B; b++) {
        double sum = 0.0;
        /* scores is [B x E]: the gather needs the token's own row, not row 0 */
        for (int64_t e = 0; e < T; e++)
            weights[b * T + e] = scores[(size_t)b * E + ids[b * T + e]];
        if (spec->moe_renormalize) {
            for (int64_t e = 0; e < T; e++) sum += weights[b * T + e];
            float inv = 1.0f / (float)(sum + 1.0e-20);
            for (int64_t e = 0; e < T; e++) weights[b * T + e] *= inv;
        }
        if (spec->routed_scaling_factor != 1.0f)
            for (int64_t e = 0; e < T; e++) weights[b * T + e] *= spec->routed_scaling_factor;
    }

    {
        char probe[128];
        snprintf(probe, sizeof(probe),
                 "layers.%lld.block_sparse_moe.routed_expert_down_proj.weight", layer);
        latent_layout = cwick_spine_find(spine, probe) != CWICK_TENSOR_MISSING;
    }

    float *routed_in = cwick_xmalloc((size_t)RH * B * sizeof(float));
    if (latent_layout) {
        char name[128];
        snprintf(name, sizeof(name),
                 "layers.%lld.block_sparse_moe.routed_expert_down_proj.weight", layer);
        TENSOR(name);
        for (int64_t b = 0; b < B; b++)
            cwick_linear(routed_in + b * RH, hidden + b * H, 1, H, RH, t);
    } else {
        for (int64_t b = 0; b < B; b++)
            memcpy(routed_in + b * RH, hidden + b * H, (size_t)H * sizeof(float));
    }
    cwick_prof_add(CWICK_PH_OTHER, cwick_prof_ns() - _tr0);
    g_batch_router_ms += (cwick_prof_ns() - _tr0) / 1.0e6;

    float *combined = cwick_xmalloc((size_t)RH * B * sizeof(float));
    memset(combined, 0, (size_t)RH * B * sizeof(float));
    float *gate_buf = cwick_xmalloc((size_t)I * sizeof(float));
    float *up_buf   = cwick_xmalloc((size_t)I * sizeof(float));
    float *act_buf  = cwick_xmalloc((size_t)I * sizeof(float));
    float *y_buf    = cwick_xmalloc((size_t)RH * sizeof(float));
    /* Per-expert token group buffers for prefill: one row per token so a
     * shared expert's GEMVs can be done as a single row-major block. */
    float *rows_in  = cwick_xmalloc((size_t)B * (size_t)RH * sizeof(float));
    float *gate_mat = cwick_xmalloc((size_t)B * (size_t)I * sizeof(float));
    float *up_mat   = cwick_xmalloc((size_t)B * (size_t)I * sizeof(float));
    float *act_mat  = cwick_xmalloc((size_t)B * (size_t)I * sizeof(float));
    float *y_mat    = cwick_xmalloc((size_t)B * (size_t)RH * sizeof(float));
    int64_t *tok_of = cwick_xmalloc((size_t)B * sizeof(int64_t));
    float *row_w    = cwick_xmalloc((size_t)B * sizeof(float));
    uint64_t tg_ns = 0;   /* GEMV phase, summed over this call's windows */

    /* Admit only the recency tail that fits the global expert budget. Every
     * layer receives the same token window, so early layers cannot monopolise
     * the cache before the later sparse layers are reached.
     *
     * That policy is for a PREFILL window (hundreds of token rows, opt-in via
     * CWICK_PREFILL_CACHE_TOKENS). A decode group (progress == NULL, a handful
     * of rows that keep stepping) must admit like the single-sequence path
     * does: with the same cutoff its B rows never reach B, nothing is ever
     * admitted, and every step re-reads all 14.66 experts -- measured 0% cache
     * hit, 359 GiB streamed, 3.8 s/step, i.e. batching without sharing. */
    const bool decode_group = (progress == NULL);
    int64_t hot_tokens = spine->stream ? prefill_cache_tokens(spec, spine) : 0;
    if (hot_tokens > B) hot_tokens = B;
    int64_t hot_cutoff = hot_tokens > 0 ? B - hot_tokens : B;
    int64_t *last_seen = cwick_xmalloc((size_t)E * sizeof(int64_t));
    for (int64_t e = 0; e < E; e++) last_seen[e] = -1;
    for (int64_t b = 0; b < B; b++)
        for (int64_t k = 0; k < T; k++)
            last_seen[ids[b * T + k]] = b;

    /* ---- routed experts: each loaded ONCE, applied to all routed tokens ---- */
    if (!spine->is_safetensors) {
        float *w1 = cwick_xmalloc((size_t)(I * RH) * sizeof(float));
        float *w2 = cwick_xmalloc((size_t)(RH * I) * sizeof(float));
        float *w3 = cwick_xmalloc((size_t)(I * RH) * sizeof(float));
        for (int64_t e = 0; e < E; e++) {
            bool used = false;
            for (int64_t b = 0; b < B && !used; b++)
                for (int64_t k = 0; k < T; k++)
                    if (ids[b * T + k] == e) { used = true; break; }
            if (!used) continue;
            if (cwick_expert_load(spine, layer, e, spec, w1, w2, w3, err) != CWICK_OK) {
                free(last_seen);
                free(w3); free(w2); free(w1); free(y_buf); free(act_buf);
                free(row_w); free(tok_of); free(y_mat); free(act_mat);
                free(up_mat); free(gate_mat); free(rows_in);
                free(up_buf); free(gate_buf); free(combined); free(routed_in);
                free(weights); free(ids); free(choice); free(scores); free(logits);
                return;
            }
            for (int64_t b = 0; b < B; b++) {
                int64_t eidx = -1;
                for (int64_t k = 0; k < T; k++) if (ids[b * T + k] == e) { eidx = k; break; }
                if (eidx < 0) continue;
                cwick_gemm_f32(gate_buf, 1, RH, I, routed_in + b * RH, w1);
                cwick_gemm_f32(up_buf,  1, RH, I, routed_in + b * RH, w3);
                moe_act(spec, act_buf, gate_buf, up_buf, I);
                cwick_gemm_f32(y_buf, 1, I, RH, act_buf, w2);
                float wv = weights[b * T + eidx];
                for (int64_t i = 0; i < RH; i++) combined[b * RH + i] += wv * y_buf[i];
            }
            if (progress) progress(progress_user, layer, e + 1, E);
        }
        free(w3); free(w2); free(w1);
    } else {
        /* Every expert that any token in this window routed to, ascending.
         * Ascending because consecutive experts live next to each other inside
         * a shard, and monotonic progress through a shard is worth real
         * bandwidth when the drive readahead is involved. */
        int64_t *used_list = cwick_xmalloc((size_t)E * sizeof(int64_t));
        bool *admit = cwick_xmalloc((size_t)E * sizeof(bool));
        int64_t nused = 0;
        for (int64_t e = 0; e < E; e++) {
            if (last_seen[e] < 0) continue;
            used_list[nused] = e;
            admit[nused] = decode_group || last_seen[e] >= hot_cutoff;
            nused++;
        }

        /* Windows: a long prompt can route to every one of the 288 experts in
         * a layer, and 288 x 3 x 8 MiB is 6.9 GiB of staging. Thirty-two at a
         * time is 768 MiB and keeps twelve preads in flight, which is where
         * this checkpoint's bandwidth curve flattens. */
        const int64_t GW = 32;
        cwick_fetch_req *reqs = cwick_xmalloc((size_t)GW * 3 * sizeof(*reqs));
        cwick_expert_view *vgs = cwick_xmalloc((size_t)GW * sizeof(*vgs));
        cwick_expert_view *vus = cwick_xmalloc((size_t)GW * sizeof(*vus));
        cwick_expert_view *vds = cwick_xmalloc((size_t)GW * sizeof(*vds));
        int64_t *Sof  = cwick_xmalloc((size_t)GW * sizeof(*Sof));
        int64_t *roff = cwick_xmalloc((size_t)GW * sizeof(*roff));
        for (int64_t g = 0; g < nused; g += GW) {
            int64_t gn = nused - g < GW ? nused - g : GW;
            /* Heartbeat before the window's fetch+compute: the window is the
             * unit of expert I/O, so `g` is how many routed experts this
             * layer's MoE phase has finished. */
            if (progress) progress(progress_user, layer, g, nused);
            uint64_t _tw0 = cwick_prof_ns();
            bool conc = spine->stream &&
                        layer_fetch_ex(spine, spec, layer, used_list + g, gn,
                                       admit + g, reqs, err) == CWICK_OK;
            cwick_prof_add(CWICK_PH_STREAM, cwick_prof_ns() - _tw0);

            /* Pass 1: the views, and how many token rows each expert owns.
             * Counting first is what lets pass 2 lay every expert's rows out
             * contiguously in one shared block, which is what the batch needs. */
            int64_t rows_total = 0;
            bool views_ok = true;
            for (int64_t j = 0; j < gn; j++) {
                int64_t e = used_list[g + j];
                memset(&vgs[j], 0, sizeof(vgs[j]));
                memset(&vus[j], 0, sizeof(vus[j]));
                memset(&vds[j], 0, sizeof(vds[j]));
                if (conc) {
                    fetch_view(&vgs[j], &reqs[j * 3 + 0]);
                    fetch_view(&vus[j], &reqs[j * 3 + 1]);
                    fetch_view(&vds[j], &reqs[j * 3 + 2]);
                    /* Ownership of the ephemeral staging buffers moves to the
                     * views; expert_view_release frees them and the batch-level
                     * release below must not do it a second time. */
                    reqs[j * 3 + 0].owned = false;
                    reqs[j * 3 + 1].owned = false;
                    reqs[j * 3 + 2].owned = false;
                } else if (expert_view_into(spine, spec, layer, e, "gate_proj",
                                            admit[g + j], &vgs[j], err) != CWICK_OK ||
                           expert_view_into(spine, spec, layer, e, "up_proj",
                                            admit[g + j], &vus[j], err) != CWICK_OK ||
                           expert_view_into(spine, spec, layer, e, "down_proj",
                                            admit[g + j], &vds[j], err) != CWICK_OK) {
                    views_ok = false;
                    break;
                }
                /* Every token that routed to this expert. Usually three or four,
                 * but a long prompt can hand an expert dozens. */
                int64_t S = 0;
                for (int64_t b = 0; b < B; b++)
                    for (int64_t k2 = 0; k2 < T; k2++)
                        if (ids[b * T + k2] == e) { S++; break; }
                Sof[j] = S;
                roff[j] = rows_total;
                rows_total += S;
            }
            if (!views_ok) {
                for (int64_t j = 0; j < gn; j++) {
                    expert_view_release(&vds[j]);
                    expert_view_release(&vus[j]);
                    expert_view_release(&vgs[j]);
                }
                layer_fetch_release(spine, layer, reqs, gn * 3);
                free(roff); free(Sof); free(vds); free(vus); free(vgs);
                free(reqs); free(admit); free(used_list);
                free(last_seen);
                free(y_buf); free(act_buf); free(up_buf); free(gate_buf);
                free(row_w); free(tok_of); free(y_mat); free(act_mat);
                free(up_mat); free(gate_mat); free(rows_in);
                free(combined); free(routed_in); free(weights); free(ids);
                free(choice); free(scores); free(logits);
                return;
            }

            /* Pass 2: gather the rows, with each row's owning token and
             * routing weight beside it. */
            float *rows_all = NULL, *gate_all = NULL, *up_all = NULL,
                  *act_all = NULL, *y_all = NULL, *w_all = NULL;
            int64_t *tok_all = NULL;
            if (rows_total > 0) {
                rows_all = cwick_xmalloc((size_t)rows_total * (size_t)RH * sizeof(float));
                gate_all = cwick_xmalloc((size_t)rows_total * (size_t)I  * sizeof(float));
                up_all   = cwick_xmalloc((size_t)rows_total * (size_t)I  * sizeof(float));
                act_all  = cwick_xmalloc((size_t)rows_total * (size_t)I  * sizeof(float));
                y_all    = cwick_xmalloc((size_t)rows_total * (size_t)RH * sizeof(float));
                w_all    = cwick_xmalloc((size_t)rows_total * sizeof(float));
                tok_all  = cwick_xmalloc((size_t)rows_total * sizeof(int64_t));
                for (int64_t j = 0; j < gn; j++) {
                    int64_t e = used_list[g + j], S = 0;
                    for (int64_t b = 0; b < B; b++) {
                        int64_t eidx = -1;
                        for (int64_t k2 = 0; k2 < T; k2++)
                            if (ids[b * T + k2] == e) { eidx = k2; break; }
                        if (eidx < 0) continue;
                        int64_t r = roff[j] + S;
                        memcpy(rows_all + r * (size_t)RH,
                               routed_in + b * (size_t)RH,
                               (size_t)RH * sizeof(float));
                        tok_all[r] = b;
                        w_all[r] = weights[b * T + eidx];
                        S++;
                    }
                }
            }
            /* Two-wave Metal batch, the shape the decode group already uses:
             * gate and up are independent, down consumes both.
             *
             * This is why the loop above was restructured. Submitting one
             * matrix at a time measured 0.97 ms per 8 MiB matrix -- the same as
             * the CPU path -- because commit-and-wait dwarfs the ~0.1 ms of real
             * GPU work; batched, decode pays 0.188 ms per matrix. A group past
             * CWICK_PREFILL_BATCH_ROWS stays on the CPU because its staging
             * blocks would cost more RAM than the batch wins back.
             *
             * The gate is the caller, because the two callers fetch into
             * different memory: prefill (progress != NULL, it reports layer
             * progress) stages in malloc'd blobs unless CWICK_PREFILL_METAL=1
             * put them in the arena, and the batch binds them as no-copy
             * views that outlive the free() at the end of the layer --
             * 2026-10-16 measured exactly that combination (malloc staging +
             * forced batch) as "SIGSEGV in the first layer's MoE", see the
             * staging-lifetime note above. The decode group (progress ==
             * NULL) fetches through the stream's arena by construction, so
             * it keeps its own switch, CWICK_GROUP_METAL (default on). */
            bool batched = false;
            uint64_t _tg0 = cwick_prof_ns();
            if (rows_total > 0 && rows_total <= prefill_batch_rows() &&
                (progress ? prefill_metal() : group_metal())) {                void *mb = cwick_metal_batch_begin();
                if (mb) {
                    batched = true;
                    for (int64_t j = 0; j < gn && batched; j++) {
                        if (Sof[j] <= 0) continue;
                        /* Same lifetime rule as the decode group: never bind a
                         * blob the layer is about to free. */
                        if (!batch_safe(&vgs[j]) || !batch_safe(&vus[j]) ||
                            !batch_safe(&vds[j])) {
                            batched = false;
                            break;
                        }
                        batched =
                          cwick_metal_batch_gemv(mb, vgs[j].data, vgs[j].scale_inv,
                                                rows_all + roff[j] * (size_t)RH,
                                                gate_all + roff[j] * (size_t)I,
                                                vgs[j].cols, vgs[j].rows,
                                                vgs[j].ncb, Sof[j], vgs[j].cols) &&
                          cwick_metal_batch_gemv(mb, vus[j].data, vus[j].scale_inv,
                                                rows_all + roff[j] * (size_t)RH,
                                                up_all + roff[j] * (size_t)I,
                                                vus[j].cols, vus[j].rows,
                                                vus[j].ncb, Sof[j], vus[j].cols);
                    }
                    if (batched) batched = cwick_metal_batch_end(mb) != 0;
                    else cwick_metal_batch_abort(mb);
                }
                if (batched) {
                    for (int64_t j = 0; j < gn; j++)
                        for (int64_t s = 0; s < Sof[j]; s++)
                            moe_act(spec, act_all + (roff[j] + s) * (size_t)I,
                                    gate_all + (roff[j] + s) * (size_t)I,
                                    up_all + (roff[j] + s) * (size_t)I, I);

                    void *mb2 = cwick_metal_batch_begin();
                    bool d_ok = false;
                    if (mb2) {
                        d_ok = true;
                        for (int64_t j = 0; j < gn && d_ok; j++) {
                            if (Sof[j] <= 0) continue;
                            d_ok = cwick_metal_batch_gemv(mb2, vds[j].data,
                                                         vds[j].scale_inv,
                                                         act_all + roff[j] * (size_t)I,
                                                         y_all + roff[j] * (size_t)RH,
                                                         vds[j].cols, vds[j].rows,
                                                         vds[j].ncb, Sof[j],
                                                         vds[j].cols);
                        }
                        if (d_ok) d_ok = cwick_metal_batch_end(mb2) != 0;
                        else cwick_metal_batch_abort(mb2);
                    }
                    if (!d_ok) batched = false;   /* redo the group on the CPU */
                }
            }
            if (batched) {
                for (int64_t j = 0; j < gn; j++)
                    for (int64_t s = 0; s < Sof[j]; s++) {
                        float wv = w_all[roff[j] + s];
                        const float *y = y_all + (roff[j] + s) * (size_t)RH;
                        float *acc = combined + tok_all[roff[j] + s] * (size_t)RH;
                        for (int64_t i = 0; i < RH; i++) acc[i] += wv * y[i];
                    }
            } else {
                /* CPU group. The single-matrix GPU path is suspended while it
                 * runs: retrying here would pay one commit-and-wait per matrix,
                 * which is slower than never having tried. */
                cwick_prefill_metal_suspend(true);
                for (int64_t j = 0; j < gn; j++) {
                    if (Sof[j] <= 0) continue;
                    /* Rows per expert at which the GEMM below beats per-row
                     * GEMVs (see the comment on the non-FP8 branch). */
                    const int64_t kMin_rows = 8;
                    const float *xin = rows_all + roff[j] * (size_t)RH;
                    float *g = gate_all + roff[j] * (size_t)I;
                    float *u = up_all   + roff[j] * (size_t)I;
                    float *a = act_all  + roff[j] * (size_t)I;
                    float *y = y_all    + roff[j] * (size_t)RH;
                    if (vgs[j].dtype == CWICK_ST_F8_E4M3) {
                        cwick_gemv_fp8_rows(g, Sof[j], vgs[j].cols, vgs[j].rows,
                                           xin, vgs[j].data, vgs[j].scale_inv,
                                           vgs[j].ncb);
                        cwick_gemv_fp8_rows(u, Sof[j], vus[j].cols, vus[j].rows,
                                           xin, vus[j].data, vus[j].scale_inv,
                                           vus[j].ncb);
                    } else {
                        /* One GEMM for the expert's whole row group, not one
                         * GEMV per row. `cwick_gemm_bf16` dequantizes a tile
                         * of the matrix once and hands the group to sgemm,
                         * where the per-row loop paid a fork-join per row per
                         * matrix -- with a long prompt that came to tens of
                         * thousands of wake-ups per layer, and it made the
                         * sweep slower than the disk reads it used to wait on.
                         *
                         * The group has to be worth it: below CWICK_MOE_GEMV_
                         * ROWS the expansion to f32 costs more than the rows
                         * amortise (measured: a 209-token window, ~5 rows per
                         * expert, is 12% slower as a GEMM; an 833-token
                         * window, ~20 rows, is 18% faster). So small groups
                         * keep the GEMV, which is also what a decode group
                         * always has. */
                        if (Sof[j] >= kMin_rows) {
                            cwick_gemm_bf16(g, Sof[j], vgs[j].cols, vgs[j].rows,
                                            xin, vgs[j].data);
                            cwick_gemm_bf16(u, Sof[j], vus[j].cols, vus[j].rows,
                                            xin, vus[j].data);
                        } else {
                            for (int64_t s = 0; s < Sof[j]; s++) {
                                expert_gemv(g + s * (size_t)vgs[j].rows, &vgs[j],
                                            xin + s * (size_t)RH);
                                expert_gemv(u + s * (size_t)vus[j].rows, &vus[j],
                                            xin + s * (size_t)RH);
                            }
                        }
                    }
                    for (int64_t s = 0; s < Sof[j]; s++)
                        moe_act(spec, a + s * (size_t)I, g + s * (size_t)I,
                                u + s * (size_t)I, I);
                    if (vds[j].dtype == CWICK_ST_F8_E4M3) {
                        cwick_gemv_fp8_rows(y, Sof[j], vds[j].cols, vds[j].rows,
                                           a, vds[j].data, vds[j].scale_inv,
                                           vds[j].ncb);
                    } else if (Sof[j] >= kMin_rows) {
                        cwick_gemm_bf16(y, Sof[j], vds[j].cols, vds[j].rows,
                                        a, vds[j].data);
                    } else {
                        for (int64_t s = 0; s < Sof[j]; s++)
                            expert_gemv(y + s * (size_t)vds[j].rows, &vds[j],
                                        a + s * (size_t)vds[j].cols);
                    }
                    for (int64_t s = 0; s < Sof[j]; s++) {
                        float wv = w_all[roff[j] + s];
                        float *acc = combined + tok_all[roff[j] + s] * (size_t)RH;
                        for (int64_t i = 0; i < RH; i++)
                            acc[i] += wv * y[s * (size_t)RH + i];
                    }
                }
                cwick_prefill_metal_suspend(false);
            }
            /* Where this window's routed experts actually ran. A window that
             * reached the Metal batch and kept it is g_moe_batch_metal; one
             * that reached it and was refused (batch_safe, or a bridge
             * failure) is g_moe_batch_cpu; a window that never got views
             * (no route-first fetch) ran the serial per-expert loop. */
            if (conc) {
                if (batched) g_moe_batch_metal++;
                else         g_moe_batch_cpu++;
            } else {
                g_moe_batch_serial++;
            }
            tg_ns += cwick_prof_ns() - _tg0;
            /* Cache-backed views stay resident; only ephemeral prefill
             * staging allocations are owned by this iteration. */
            for (int64_t j = 0; j < gn; j++) {
                expert_view_release(&vds[j]);
                expert_view_release(&vus[j]);
                expert_view_release(&vgs[j]);
            }
            free(rows_all); free(gate_all); free(up_all); free(act_all);
            free(y_all); free(w_all); free(tok_all);
            layer_fetch_release(spine, layer, reqs, gn * 3);
            /* Window progress: the loop above is where the layer's ~100 s
             * live (288 experts streamed and applied in GW-sized groups), so
             * reporting only on completion leaves the operator blind for the
             * whole stretch. One line per window, time-gated by the callee. */
            if (progress) progress(progress_user, layer, g + gn, nused);
        }
        if (progress) progress(progress_user, layer, nused, nused);
        free(roff); free(Sof); free(vds); free(vus); free(vgs);
        free(reqs); free(admit); free(used_list);
    }
    free(last_seen);
    free(y_buf); free(act_buf); free(up_buf); free(gate_buf);
    free(row_w); free(tok_of); free(y_mat); free(act_mat);
    free(up_mat); free(gate_mat); free(rows_in);

    /* ---- completion (identical to cwick_moe_forward) ---- */
    uint64_t _tc0 = cwick_prof_ns();
    float *routed_normed = cwick_xmalloc((size_t)RH * B * sizeof(float));
    if (latent_layout && spec->latent_moe_use_norm) {
        char name[128];
        snprintf(name, sizeof(name),
                 "layers.%lld.block_sparse_moe.routed_expert_norm.weight", layer);
        TENSOR(name);
        const uint16_t *wn = (const uint16_t *)t->data;
        for (int64_t b = 0; b < B; b++) {
            cwick_rms_norm(routed_normed + b * RH, combined + b * RH, RH, NULL, spec->rms_norm_eps_f);
            for (int64_t i = 0; i < RH; i++)
                routed_normed[b * RH + i] *= cwick_bf16_to_f32(wn[i]);
        }
    } else {
        for (int64_t b = 0; b < B; b++)
            memcpy(routed_normed + b * RH, combined + b * RH, (size_t)RH * sizeof(float));
    }
    float *routed_full = cwick_xmalloc((size_t)H * B * sizeof(float));
    if (latent_layout) {
        char name[128];
        snprintf(name, sizeof(name),
                 "layers.%lld.block_sparse_moe.routed_expert_up_proj.weight", layer);
        TENSOR(name);
        for (int64_t b = 0; b < B; b++)
            cwick_linear(routed_full + b * H, routed_normed + b * RH, 1, RH, H, t);
    } else {
        for (int64_t b = 0; b < B; b++)
            memcpy(routed_full + b * H, routed_normed + b * RH, (size_t)H * sizeof(float));
    }
    cwick_prof_add(CWICK_PH_OTHER, cwick_prof_ns() - _tc0);
    g_batch_completion_ms += (cwick_prof_ns() - _tc0) / 1.0e6;

    /* The shared expert runs on every token in the chunk, so these are
     * B-wide: gate/up are [B x SI] and the activation output is [B x SI].
     *
     * Same two-wave GPU batch the single-sequence path uses (see
     * cwick_moe_forward): on CPU this was 3 x 8 MiB fp8 matrices per token
     * row -- with B=2 that is ~1 GiB of dequant per step, measured 3247 ms
     * of a 5126 ms/step group. gate and up read only `hidden`, so they ride
     * wave one; down consumes sh, which moe_act produces on the CPU while
     * the waves run, so it rides wave two. Any refusal falls back to the
     * CPU GEMVs, which produce the identical answer. */
    uint64_t _tsh = cwick_prof_ns();
    float *sg = cwick_xmalloc((size_t)SI * B * sizeof(float));
    float *su = cwick_xmalloc((size_t)SI * B * sizeof(float));
    float *sh = cwick_xmalloc((size_t)SI * B * sizeof(float));
    float *shared_out = cwick_xmalloc((size_t)H * B * sizeof(float));
    const cwick_tensor *t_sg = NULL, *t_su = NULL, *t_sd = NULL;
    if (SI > 0) {
        char name[128];
        int64_t gi, ui, di;
        snprintf(name, sizeof(name),
                 "layers.%lld.block_sparse_moe.shared_experts.gate_proj.weight", layer);
        gi = cwick_spine_find(spine, name);
        snprintf(name, sizeof(name),
                 "layers.%lld.block_sparse_moe.shared_experts.up_proj.weight", layer);
        ui = cwick_spine_find(spine, name);
        snprintf(name, sizeof(name),
                 "layers.%lld.block_sparse_moe.shared_experts.down_proj.weight", layer);
        di = cwick_spine_find(spine, name);
        if (gi != CWICK_TENSOR_MISSING && ui != CWICK_TENSOR_MISSING &&
            di != CWICK_TENSOR_MISSING) {
            t_sg = &spine->tensors[(size_t)gi];
            t_su = &spine->tensors[(size_t)ui];
            t_sd = &spine->tensors[(size_t)di];
        }
    }
    /* The batch GEMV bridge takes one x vector with a row stride, which is
     * exactly `hidden` [B x H] -- the same shape the routed group feeds it. */
    bool sh_gpu = false;
    if (t_sg && t_su && t_sd &&
        t_sg->kind == CWICK_WEIGHT_FP8_E4M3 && t_sg->scales && t_sg->cols > 0 &&
        t_su->kind == CWICK_WEIGHT_FP8_E4M3 && t_su->scales && t_su->cols > 0) {
        const int64_t ncb_g = t_sg->cols / CWICK_ST_BLOCK;
        void *mb = cwick_metal_batch_begin();
        if (mb) {
            bool ok = cwick_metal_batch_gemv(mb, t_sg->data,
                                            (const float *)t_sg->scales,
                                            hidden, sg, t_sg->cols,
                                            t_sg->rows, ncb_g, B, H) &&
                      cwick_metal_batch_gemv(mb, t_su->data,
                                            (const float *)t_su->scales,
                                            hidden, su, t_su->cols,
                                            t_su->rows, ncb_g, B, H);
            if (ok) ok = cwick_metal_batch_end(mb) != 0;
            else    cwick_metal_batch_abort(mb);
            if (ok) {
                moe_act(spec, sh, sg, su, SI * B);
                if (t_sd->kind == CWICK_WEIGHT_FP8_E4M3 && t_sd->scales &&
                    t_sd->cols > 0) {
                    const int64_t ncb_d = t_sd->cols / CWICK_ST_BLOCK;
                    void *mb2 = cwick_metal_batch_begin();
                    if (mb2) {
                        bool ok2 = cwick_metal_batch_gemv(mb2, t_sd->data,
                                                         (const float *)t_sd->scales,
                                                         sh, shared_out,
                                                         t_sd->cols, t_sd->rows,
                                                         ncb_d, B, SI);
                        if (ok2) ok2 = cwick_metal_batch_end(mb2) != 0;
                        else     cwick_metal_batch_abort(mb2);
                        sh_gpu = ok2;
                    }
                }
            }
        }
    }
    if (!sh_gpu) {
        char name[128];
        snprintf(name, sizeof(name),
                 "layers.%lld.block_sparse_moe.shared_experts.gate_proj.weight", layer);
        TENSOR(name);
        cwick_linear(sg, hidden, B, H, SI, t);
        snprintf(name, sizeof(name),
                 "layers.%lld.block_sparse_moe.shared_experts.up_proj.weight", layer);
        TENSOR(name);
        cwick_linear(su, hidden, B, H, SI, t);
        moe_act(spec, sh, sg, su, SI * B);
        snprintf(name, sizeof(name),
                 "layers.%lld.block_sparse_moe.shared_experts.down_proj.weight", layer);
        TENSOR(name);
        cwick_linear(shared_out, sh, B, SI, H, t);
    }
    if (sh_gpu) g_moe_batch_shared_gpu++;
    else         g_moe_batch_shared_cpu++;
    g_batch_shared_ms += (cwick_prof_ns() - _tsh) / 1.0e6;
    cwick_prof_add(CWICK_PH_GEMV, tg_ns);
    for (int64_t b = 0; b < B; b++)
        for (int64_t i = 0; i < H; i++)
            out[b * H + i] = routed_full[b * H + i] + shared_out[b * H + i];

    free(shared_out); free(sh); free(su); free(sg);
    free(routed_full); free(routed_normed); free(combined); free(routed_in);
    free(weights); free(ids); free(choice); free(scores); free(logits);
#undef TENSOR
}
