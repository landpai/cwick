/*
 * ds_run.c -- smallest possible driver for the DeepSeek forward pass.
 *
 *   ds_run <checkpoint-root> <max-seq-len> <token-id>...
 *   ds_run <checkpoint-root> <max-seq-len> --prefix-test <cut> <token-id>...
 *
 * Loads the checkpoint, runs one forward over the given ids from position 0
 * and reports the top logits. This is a smoke test: the point is to see the
 * whole 40-layer path execute end to end and produce finite numbers, not to
 * measure throughput. Timing is printed because it is the only performance
 * signal there is until the Metal kernels exist.
 *
 * `--prefix-test` checks the prefix snapshot instead: it prefills the whole
 * sequence for reference, then repeats it as "prefill [0,cut), snapshot,
 * reset, restore, prefill [cut,n)" and compares the resulting logits. Equal
 * logits mean a cache replay is indistinguishable from prefilling the prompt,
 * which is the property the prefix cache depends on.
 *
 * The other modes answer "which half is wrong" when that comparison fails:
 *
 *   --split-test <cut>  the same prompt as one call and as [0,cut)+[cut,n),
 *                       logits compared. Separates "chunking is broken" from
 *                       "the snapshot is broken": only the second involves a
 *                       round-trip through pack/apply.
 *   --state-diff <cut>  the two prefills' snapshots compared segment by
 *                       segment, with the first differing byte decoded back
 *                       into (layer, position, element).
 *   --dump <cut>        trace only: one call, or [0,cut)+[cut,n) when cut > 0.
 *   --snap-dump <cut>   trace only, through the snapshot.
 *
 * The two dump modes exist to be diffed against each other with a per-tensor
 * comparator; DS_TRACE prints every position as an index inside the call, so
 * ds_model_forward emits an FWDPOS line that puts them back into one global
 * numbering.
 */
#include "ds_model.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_desc(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return x < y ? 1 : (x > y ? -1 : 0);
}

/* Prefill [0,cut), snapshot, reset, restore, prefill [cut,n). Returns the
 * number of logit positions that differ from `ref` (0 = bit-identical). */
static int prefix_roundtrip(ds_model *m, const int32_t *ids, int n, int cut,
                            const float *ref, bool use_snapshot,
                            cwick_error *err) {
    const int vocab = m->cfg.vocab_size;
    float *logits = malloc(sizeof(float) * (size_t)vocab);
    if (!logits) return -1;

    ds_model_reset(m);
    if (ds_model_forward(m, ids, cut, 0, logits, err) != 0) {
        free(logits);
        return -1;
    }
    if (use_snapshot) {
        const size_t need = ds_state_bytes(m, cut);
        void *snap = malloc(need);
        if (!snap || ds_state_pack(m, cut, snap, need) != 0) {
            fprintf(stderr, "prefix-test: snapshot failed (%zu bytes)\n", need);
            free(snap);
            free(logits);
            return -1;
        }
        fprintf(stderr, "prefix-test: snapshot %zu bytes for %d tokens\n", need, cut);

        ds_model_reset(m);
        int64_t pos = -1;
        if (ds_state_apply(m, snap, need, &pos) != 0 || pos != cut) {
            fprintf(stderr, "prefix-test: restore failed (pos=%lld)\n",
                    (long long)pos);
            free(snap);
            free(logits);
            return -1;
        }
        free(snap);
    }

    if (ds_model_forward(m, ids + cut, n - cut, cut, logits, err) != 0) {
        free(logits);
        return -1;
    }

    int diff = 0, am_out = 0, am_ref = 0;
    double worst = 0.0, se = 0.0, ss = 0.0;
    for (int i = 0; i < vocab; i++) {
        const double d = (double)logits[i] - (double)ref[i];
        if (logits[i] != ref[i]) {
            diff++;
            if (fabs(d) > worst) worst = fabs(d);
        }
        se += d * d;
        ss += (double)ref[i] * (double)ref[i];
        if (logits[i] > logits[am_out]) am_out = i;
        if (ref[i] > ref[am_ref]) am_ref = i;
    }
    const double rms = sqrt(se / (double)vocab);
    const double rel = ss > 0.0 ? rms / sqrt(ss / (double)vocab) : rms;
    printf("prefix-test: cut=%d of %d  differing logits=%d/%d  "
           "argmax %d vs %d (%s)  rel-rms=%.3g  worst |delta|=%.6g\n",
           cut, n, diff, vocab, am_out, am_ref,
           am_out == am_ref ? "same" : "DIFFERENT", rel, worst);
    free(logits);
    return diff;
}

/* Which part of the cross-token state two ways of prefilling the same tokens
 * disagree on. Runs the sequence once in one chunk and once split at `cut`,
 * then compares the snapshots segment by segment -- the logits only say "it
 * diverged", this says where. */
static int state_diff(ds_model *m, const int32_t *ids, int n, int cut,
                      cwick_error *err) {
    const ds_config *c = &m->cfg;
    const size_t win_f = (size_t)c->n_layers * (size_t)c->window_size *
                         (size_t)c->head_dim;
    const size_t cmp_f = (size_t)c->n_kv_source * (size_t)cut *
                         (size_t)c->head_dim;
    const size_t cst_f = (size_t)c->n_kv_source * (size_t)c->max_ratio *
                         (size_t)c->head_dim;
    const size_t idx_f = (size_t)c->n_kv_source * (size_t)cut *
                         (size_t)c->index_head_dim;
    const size_t need = ds_state_bytes(m, n);
    void *a = malloc(need), *b = malloc(need);
    float *logits = malloc(sizeof(float) * (size_t)c->vocab_size);
    if (!a || !b || !logits) return -1;

    ds_model_reset(m);
    if (ds_model_forward(m, ids, n, 0, logits, err) != 0) return -1;
    if (ds_state_pack(m, n, a, need) != 0) return -1;

    ds_model_reset(m);
    if (ds_model_forward(m, ids, cut, 0, logits, err) != 0) return -1;
    if (ds_model_forward(m, ids + cut, n - cut, cut, logits, err) != 0) return -1;
    if (ds_state_pack(m, n, b, need) != 0) return -1;

    /* Segment boundaries inside the blob, in write order. */
    const size_t hdr = sizeof(uint32_t) * 2 + sizeof(int64_t) * 7;
    struct { const char *name; size_t off, bytes; } seg[] = {
        {"header", 0, hdr},
        {"win_kv", hdr, win_f * sizeof(float)},
        {"cmp_kv", hdr + win_f * 4, (size_t)c->n_kv_source * (size_t)n *
                                     (size_t)c->head_dim * 4},
        {"cmp_state", hdr + win_f * 4 + (size_t)c->n_kv_source * (size_t)n *
                        (size_t)c->head_dim * 4, cst_f * 4},
        {"cmp_score", 0, cst_f * 4},
        {"idx_k", 0, (size_t)c->n_kv_source * (size_t)n *
                     (size_t)c->index_head_dim * 4},
        {"tok_cache", 0, (size_t)n * sizeof(int64_t)},
    };
    /* Fill the running offsets after the first three fixed ones. */
    seg[4].off = seg[3].off + seg[3].bytes;
    seg[5].off = seg[4].off + seg[4].bytes;
    seg[6].off = seg[5].off + seg[5].bytes;
    (void)cmp_f; (void)idx_f;

    int bad = 0;
    for (size_t s = 0; s < sizeof(seg) / sizeof(seg[0]); s++) {
        const unsigned char *pa = (const unsigned char *)a + seg[s].off;
        const unsigned char *pb = (const unsigned char *)b + seg[s].off;
        size_t nbad = 0, first = (size_t)-1;
        for (size_t k = 0; k < seg[s].bytes; k++)
            if (pa[k] != pb[k]) { if (first == (size_t)-1) first = k; nbad++; }
        printf("state-diff: %-10s %8zu bytes  %8zu differ\n", seg[s].name,
               seg[s].bytes, nbad);
        if (nbad) {
            bad++;
            if (s == 1) {   /* win_kv: decode (layer, position, element) */
                const size_t fi = first / sizeof(float);
                const size_t HD = (size_t)c->head_dim;
                const size_t W = (size_t)c->window_size;
                printf("            first at layer %zu pos %zu elem %zu\n",
                       fi / (HD * W), (fi / HD) % W, fi % HD);
            } else if (s == 2) {   /* cmp_kv: (src, compressed row, elem) */
                const size_t fi = first / sizeof(float);
                const size_t HD = (size_t)c->head_dim;
                printf("            first at src %zu row %zu elem %zu\n",
                       fi / (HD * (size_t)n), (fi / HD) % (size_t)n, fi % HD);
            }
        }
    }
    free(a); free(b); free(logits);
    return bad;
}

int main(int argc, char **argv) {
    bool prefix_test = false, split_test = false, state_test = false;
    bool dump_test = false, snap_dump = false;
    int cut = 0;
    if (argc >= 5 && strcmp(argv[3], "--prefix-test") == 0) {
        prefix_test = true;
        cut = atoi(argv[4]);
    } else if (argc >= 5 && strcmp(argv[3], "--split-test") == 0) {
        split_test = true;
        cut = atoi(argv[4]);
    } else if (argc >= 5 && strcmp(argv[3], "--state-diff") == 0) {
        state_test = true;
        cut = atoi(argv[4]);
    } else if (argc >= 5 && strcmp(argv[3], "--dump") == 0) {
        dump_test = true;
        cut = atoi(argv[4]);
    } else if (argc >= 5 && strcmp(argv[3], "--snap-dump") == 0) {
        dump_test = true;
        snap_dump = true;
        cut = atoi(argv[4]);
    }
    const bool mode = prefix_test || split_test || state_test || dump_test;
    if (argc < (mode ? 6 : 4)) {
        fprintf(stderr, "usage: %s <checkpoint-root> <max-seq-len> "
                        "[--prefix-test|--split-test|--state-diff|--dump|"
                        "--snap-dump <cut>] <token-id>...\n",
                argv[0]);
        return 2;
    }
    const char *root = argv[1];
    int max_seq = atoi(argv[2]);
    /* The optional mode flag sits between the two positional arguments and the
     * token list, so the ids start at 5 when it is present and 3 when not. */
    const int id_start = mode ? 5 : 3;
    int n = argc - id_start;
    int32_t *ids = malloc(sizeof(int32_t) * (size_t)(n > 0 ? n : 1));
    for (int i = 0; i < n; i++) ids[i] = atoi(argv[id_start + i]);

    cwick_error err = {0};
    ds_model m;
    double t0 = now_s();
    if (ds_model_open(&m, root, max_seq, &err) != 0) {
        fprintf(stderr, "open failed: %s\n", err.message);
        return 1;
    }
    double t1 = now_s();
    fprintf(stderr, "loaded in %.2fs: %d layers, dim %d, %d experts, hc %d, "
                    "kv sources %d, index sources %d, engram layers %d, "
                    "window %d, max_ratio %d\n",
            t1 - t0, m.cfg.n_layers, m.cfg.dim, m.cfg.n_routed_experts,
            m.cfg.hc_mult, m.cfg.n_kv_source, m.cfg.n_index_source,
            m.cfg.n_engram, m.cfg.window_size, m.cfg.max_ratio);

    float *logits = malloc(sizeof(float) * (size_t)m.cfg.vocab_size);

    /* `--dump`: trace-only mode. Runs the prompt as one call (cut 0) or as
     * [0,cut) + [cut,n) and prints nothing but whatever DS_TRACE emits, so
     * the two traces can be diffed tensor by tensor. */
    if (dump_test) {
        if (cut < 0 || cut >= n) {
            fprintf(stderr, "dump: cut must be in 0..%d\n", n - 1);
            return 2;
        }
        ds_model_reset(&m);
        if (!snap_dump) {
            if (cut == 0) {
                if (ds_model_forward(&m, ids, n, 0, logits, &err) != 0) {
                    fprintf(stderr, "forward failed: %s\n", err.message);
                    return 1;
                }
            } else {
                if (ds_model_forward(&m, ids, cut, 0, logits, &err) != 0 ||
                    ds_model_forward(&m, ids + cut, n - cut, cut, logits, &err) != 0) {
                    fprintf(stderr, "forward failed: %s\n", err.message);
                    return 1;
                }
            }
        } else {
            /* [0,cut) prefill, snapshot, rewind, restore, then the tail: the
             * prefix-cache path, traced so it can be diffed against --dump 0 */
            if (ds_model_forward(&m, ids, cut, 0, logits, &err) != 0) {
                fprintf(stderr, "forward failed: %s\n", err.message);
                return 1;
            }
            const size_t need = ds_state_bytes(&m, cut);
            void *snap = malloc(need);
            if (!snap || ds_state_pack(&m, cut, snap, need) != 0) {
                fprintf(stderr, "snap-dump: pack failed\n");
                return 1;
            }
            ds_model_reset(&m);
            int64_t pos = -1;
            if (ds_state_apply(&m, snap, need, &pos) != 0 || pos != cut) {
                fprintf(stderr, "snap-dump: restore failed (pos=%lld)\n",
                        (long long)pos);
                return 1;
            }
            free(snap);
            if (ds_model_forward(&m, ids + cut, n - cut, cut, logits, &err) != 0) {
                fprintf(stderr, "forward failed: %s\n", err.message);
                return 1;
            }
        }
        free(logits);
        free(ids);
        ds_model_close(&m);
        return 0;
    }

    t0 = now_s();
    if (ds_model_forward(&m, ids, n, 0, logits, &err) != 0) {
        fprintf(stderr, "forward failed: %s\n", err.message);
        return 1;
    }
    t1 = now_s();

    int nan = 0, inf = 0;
    for (int i = 0; i < m.cfg.vocab_size; i++) {
        if (isnan(logits[i])) nan++;
        else if (isinf(logits[i])) inf++;
    }
    printf("forward %d token(s) in %.2fs (%.3fs/token)  nan=%d inf=%d\n",
           n, t1 - t0, (t1 - t0) / n, nan, inf);

    if (mode) {
        if (cut <= 0 || cut >= n) {
            fprintf(stderr, "prefix-test: cut must be in 1..%d\n", n - 1);
            return 2;
        }
        if (state_test) {
            int bad = state_diff(&m, ids, n, cut, &err);
            free(logits);
            free(ids);
            ds_model_close(&m);
            return bad < 0 ? 1 : (bad ? 3 : 0);
        }
        int diff = prefix_roundtrip(&m, ids, n, cut, logits, prefix_test, &err);
        free(logits);
        free(ids);
        ds_model_close(&m);
        if (diff < 0) return 1;
        return diff == 0 ? 0 : 3;
    }

    /* top 5, plus the argmax */
    int topn = m.cfg.vocab_size < 5 ? m.cfg.vocab_size : 5;
    float *best = malloc(sizeof(float) * (size_t)topn);
    memcpy(best, logits, sizeof(float) * (size_t)topn);
    qsort(best, (size_t)topn, sizeof(float), cmp_desc);
    printf("top logits:");
    for (int k = 0; k < topn; k++) {
        int id = -1;
        for (int i = 0; i < m.cfg.vocab_size; i++)
            if (logits[i] == best[k]) { id = i; break; }
        printf(" %d:%.4f", id, best[k]);
    }
    printf("\n");
    free(best);
    free(logits);
    free(ids);
    ds_model_close(&m);
    return nan || inf ? 1 : 0;
}
