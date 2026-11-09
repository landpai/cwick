/*
 * mm_run.c — drive the MiniMax-M3.1-Flash port on its own, with no server.
 *
 *   mm_run <checkpoint-root> <max-seq> <max-new> [--prompt TEXT]
 *          [--ids 200019,8833,...] [--topk 5] [--trace]
 *
 * This is the only oracle the port has: the released checkpoint is a real
 * instruct model, so a forward pass that is wired correctly keeps producing
 * coherent text, and one that is not produces noise within a few tokens -- a
 * rotated rope half, a global scale read the wrong way round, or a routing
 * weight taken from the biased scores all show up that way rather than as an
 * error. There is no reference implementation on this machine to diff against.
 *
 * It also prints the per-step breakdown, because the token rate is the whole
 * point: prefill is chunked, decode is one position at a time, and the two
 * want different things.
 */
#include "cwick.h"
#include "mm.h"
#include "mem.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h>

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

static int argmax(const float *v, int n) {
    int b = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i;
    return b;
}

static int top_ids[16];
static float top_vals[16];

static void print_top(const float *v, int n, int k) {
    int idx[16];
    float val[16];
    int m = 0;
    for (int i = 0; i < n && m < k; i++) { idx[m] = i; val[m] = v[i]; m++; }
    for (int i = k; i < n; i++) {
        int w = m - 1;
        if (v[i] <= val[w]) continue;
        idx[w] = i; val[w] = v[i];
        while (w > 0 && val[w] > val[w - 1]) {
            float t = val[w]; val[w] = val[w - 1]; val[w - 1] = t;
            int ti = idx[w]; idx[w] = idx[w - 1]; idx[w - 1] = ti;
            w--;
        }
    }
    for (int i = 0; i < m; i++) printf("      %6d  %9.4f\n", idx[i], val[i]);
    for (int i = 0; i < m; i++) { top_ids[i] = idx[i]; top_vals[i] = val[i]; }
}

/* One greedy pass over the same prompt, optionally with block verification.
 *
 * This exists so "is speculation worth it" and "does it change the answer" are
 * answered by one binary on one loaded model. Two processes would each spend a
 * minute streaming 174 GiB before they could start, and their numbers would
 * then differ by more than the effect being measured.
 *
 * The pass is deliberately written as "emit, then look ahead": a token is
 * always the argmax of its own position's distribution, so a correct block
 * verifier produces the same sequence as a step-at-a-time run, and the
 * comparison below is exact rather than statistical. */
typedef struct {
    int64_t tokens, steps, accepted, proposed;
    double ms;
    uint32_t *out;
} mm_pass_res;

static int mm_pass(mm_model *m, const int32_t *pid, int64_t n_ids, int max_new,
                   int spec_n, uint32_t *hist, uint32_t *out, mm_pass_res *r) {
    const int V = (int)m->cfg.vocab_size;
    cwick_error err;
    cwick_error_clear(&err);
    memset(r, 0, sizeof(*r));
    r->out = out;
    float *logits = (float *)cwick_xmalloc(sizeof(float) * (size_t)V);
    float *slog = NULL;
    int32_t *ids = (int32_t *)cwick_xmalloc(sizeof(int32_t) * 16);
    uint32_t *draft = (uint32_t *)cwick_xmalloc(sizeof(uint32_t) * 16);

    mm_model_reset(m);
    double t0 = now_ms();
    if (mm_model_forward(m, pid, (int)n_ids, 0, logits, &err) != 0) {
        fprintf(stderr, "prefill: %s\n", err.message);
        goto fail;
    }
    int64_t hist_n = n_ids;
    for (int64_t i = 0; i < n_ids; i++) hist[i] = (uint32_t)pid[i];
    int64_t pos = n_ids;
    int32_t cur = (int32_t)argmax(logits, V);
    int64_t n_out = 0;

    while (n_out < max_new && cur != 200020 && cur != 200019) {
        out[n_out++] = (uint32_t)cur;
        hist[hist_n++] = (uint32_t)cur;
        if (n_out >= max_new) break;
        int nd = 0;
        if (spec_n >= 2) nd = mm_draft_lookup(hist, hist_n, spec_n - 1, draft);
        if (nd > 0) {
            if (!slog)
                slog = (float *)cwick_xmalloc(sizeof(float) * (size_t)V *
                                              (size_t)(spec_n + 1));
            ids[0] = cur;
            for (int i = 0; i < nd; i++) ids[i + 1] = (int32_t)draft[i];
            if (mm_model_forward_multi(m, ids, nd + 1, pos, slog, &err) != 0) {
                fprintf(stderr, "verify: %s\n", err.message);
                goto fail;
            }
            int acc = 0;
            for (int i = 0; i < nd; i++) {
                if ((int32_t)argmax(slog + (size_t)i * (size_t)V, V) !=
                    (int32_t)draft[i])
                    break;
                acc++;
            }
            r->proposed += nd;
            r->accepted += acc;
            r->steps++;
            for (int i = 0; i < acc && n_out < max_new; i++) {
                out[n_out++] = draft[i];
                hist[hist_n++] = draft[i];
            }
            if (n_out >= max_new) break;
            cur = (int32_t)argmax(slog + (size_t)acc * (size_t)V, V);
            pos += acc + 1;
        } else {
            if (mm_model_forward(m, &cur, 1, pos, logits, &err) != 0) {
                fprintf(stderr, "step: %s\n", err.message);
                goto fail;
            }
            pos++;
            r->steps++;
            cur = (int32_t)argmax(logits, V);
        }
    }
    r->tokens = n_out;
    r->ms = now_ms() - t0;
    free(logits);
    free(slog);
    free(ids);
    free(draft);
    return 0;
fail:
    free(logits);
    free(slog);
    free(ids);
    free(draft);
    return -1;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s <root> <max-seq> <max-new> [--prompt TEXT]\n"
                "                 [--ids a,b,c] [--topk N] [--trace]\n", argv[0]);
        return 2;
    }
    const char *root = argv[1];
    int max_seq = atoi(argv[2]);
    int max_new = atoi(argv[3]);
    const char *prompt = "The capital of France is";
    const char *ids_arg = NULL;
    int topk = 3;
    bool trace = false;
    int ab_n = 0;
    /* The checkpoint's own bos (200019): without it the model still produces
     * English but the distribution is visibly flatter and the obvious
     * completion loses to a generic one. */
    bool prepend_bos = false;
    const uint32_t bos_id = 200019;

    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--ids") && i + 1 < argc) ids_arg = argv[++i];
        else if (!strcmp(argv[i], "--topk") && i + 1 < argc) topk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--trace")) trace = true;
        else if (!strcmp(argv[i], "--bos")) prepend_bos = true;
        else if (!strcmp(argv[i], "--ab") && i + 1 < argc) ab_n = atoi(argv[++i]);
    }

    cwick_pool_init();

    /* --io FILE MB ITERS: is a cache miss paying for the read, or for the
     * allocation the read lands in? Same 10 MiB pread twice: once into a
     * freshly malloc'ed buffer (which is what the expert cache does on every
     * miss) and once into a buffer that already exists. */
    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--io") != 0 || i + 3 >= argc) continue;
        const char *path = argv[i + 1];
        size_t mb = (size_t)atoll(argv[i + 2]);
        int iters = atoi(argv[i + 3]);
        size_t bytes = mb << 20;
        int fd = open(path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "open %s: %m\n", path); return 1; }
        double t0 = now_ms();
        for (int it = 0; it < iters; it++) {
            uint8_t *p = (uint8_t *)cwick_xmalloc(bytes);
            ssize_t got = pread(fd, p, bytes, (off_t)(it % 8) * (off_t)bytes);
            if (got <= 0) { fprintf(stderr, "pread: %m\n"); return 1; }
            free(p);
        }
        double fresh = (now_ms() - t0) / (double)iters;
        enum { NB = 4 };
        uint8_t *buf[NB];
        for (int k = 0; k < NB; k++) buf[k] = (uint8_t *)cwick_xmalloc(bytes);
        t0 = now_ms();
        for (int it = 0; it < iters; it++) {
            uint8_t *p = buf[it % NB];
            ssize_t got = pread(fd, p, bytes, (off_t)(it % 8) * (off_t)bytes);
            if (got <= 0) { fprintf(stderr, "pread: %m\n"); return 1; }
            /* touch it the way a GEMV would, so neither side is lazy */
            volatile uint64_t acc = 0;
            for (size_t o = 0; o < bytes; o += 4096) acc += p[o];
            (void)acc;
        }
        double reuse = (now_ms() - t0) / (double)iters;
        printf("pread %zu MiB: fresh-buffer %.1f ms (%.2f GB/s), "
               "reused-buffer %.1f ms (%.2f GB/s)\n", mb, fresh,
               (double)bytes / (fresh * 1e6), reuse,
               (double)bytes / (reuse * 1e6));
        for (int k = 0; k < NB; k++) free(buf[k]);
        close(fd);
        return 0;
    }

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--f8") != 0) continue;
        int64_t bn = 8192, bk = 6144, bnn = 302;
        int bi = 3;
        if (i + 3 < argc) { bn = atoll(argv[i+1]); bk = atoll(argv[i+2]);
                            bnn = atoll(argv[i+3]); }
        if (i + 4 < argc) bi = atoi(argv[i+4]);
        mm_bench_fp8(bn, bk, bnn, bi);
        cwick_pool_shutdown();
        return 0;
    }

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--cvt") != 0) continue;
        int64_t cn = 3072, ck = 6144;
        if (i + 2 < argc) { cn = atoll(argv[i + 1]); ck = atoll(argv[i + 2]); }
        mm_bench_gemm_check(cn, ck);
        cwick_pool_shutdown();
        return 0;
    }

    /* --bench N K [iters]: time the two GEMV kernels on a synthetic matrix
     * and exit. No checkpoint is opened, so the numbers are the kernels. */
    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--bench") != 0) continue;
        int64_t bn = 3072, bk = 6144;
        int iters = 20;
        if (i + 2 < argc) { bn = atoll(argv[i + 1]); bk = atoll(argv[i + 2]); }
        if (i + 3 < argc) iters = atoi(argv[i + 3]);
        double gp = 0.0, gb = 0.0;
        double msp = mm_bench_fp4(bn, bk, iters, &gp);
        double msb = mm_bench_bf16(bn, bk, iters, &gb);
        double bytes_p = (double)bn * bk * 0.5 + (double)bn * (bk / 16);
        double bytes_b = (double)bn * bk * 2.0;
        printf("nvfp4 [%lldx%lld]: %.1f ms  %.2f GMAC/s  %.1f GB/s\n",
               (long long)bn, (long long)bk, msp, gp,
               bytes_p / (msp * 1e6));
        printf("bf16  [%lldx%lld]: %.1f ms  %.2f GMAC/s  %.1f GB/s\n",
               (long long)bn, (long long)bk, msb, gb,
               bytes_b / (msb * 1e6));
        cwick_pool_shutdown();
        return 0;
    }

    cwick_error err;
    cwick_error_clear(&err);

    mm_model m;
    double t0 = now_ms();
    if (mm_model_open(&m, root, max_seq, &err) != 0) {
        fprintf(stderr, "mm_model_open: %s\n", err.message);
        return 1;
    }
    printf("open %.0f ms\n", now_ms() - t0);

    /* ---- prompt ---- */
    uint32_t *ids = NULL;
    int64_t n_ids = 0;
    if (ids_arg) {
        size_t cap = 64;
        ids = (uint32_t *)malloc(cap * sizeof(uint32_t));
        char *copy = strdup(ids_arg);
        for (char *p = strtok(copy, ", "); p; p = strtok(NULL, ", ")) {
            if ((size_t)n_ids == cap) {
                cap *= 2;
                ids = (uint32_t *)realloc(ids, cap * sizeof(uint32_t));
            }
            ids[n_ids++] = (uint32_t)strtoul(p, NULL, 0);
        }
        free(copy);
    } else {
        cwick_model_spec spec;
        memset(&spec, 0, sizeof(spec));
        spec.vocab_size = m.cfg.vocab_size;
        cwick_tokenizer *tok = NULL;
        if (cwick_tokenizer_load(root, &spec, &tok, &err) != CWICK_OK) {
            fprintf(stderr, "tokenizer: %s\n", err.message);
            return 1;
        }
        if (cwick_tokenizer_encode(tok, prompt, &ids, &n_ids, &err) != CWICK_OK) {
            fprintf(stderr, "encode: %s\n", err.message);
            return 1;
        }
        if (prepend_bos) {
            uint32_t *ni = (uint32_t *)malloc(sizeof(uint32_t) *
                                              (size_t)(n_ids + 1));
            ni[0] = bos_id;
            memcpy(ni + 1, ids, sizeof(uint32_t) * (size_t)n_ids);
            free(ids);
            ids = ni;
            n_ids++;
        }
        printf("prompt %lld tokens:", (long long)n_ids);
        for (int64_t i = 0; i < n_ids && i < 16; i++) printf(" %u", ids[i]);
        printf("%s\n", n_ids > 16 ? " ..." : "");
        cwick_tokenizer_free(tok);
    }
    if (n_ids <= 0) { fprintf(stderr, "empty prompt\n"); return 1; }

    int32_t *pid = (int32_t *)malloc(sizeof(int32_t) * (size_t)n_ids);
    for (int64_t i = 0; i < n_ids; i++) pid[i] = (int32_t)ids[i];

    /* ---- --ab N: does block verification help, and does it change the
     * answer? Same process, same warm weights, three passes: plain,
     * speculative, plain again. The repeat exists because by the third pass
     * the pin cache is warm, and on this model cache warmth is worth more
     * than speculation -- without it the speculative pass would be credited
     * with a cache the first pass paid for. ---- */
    if (ab_n >= 2) {
        if (ab_n > 8) ab_n = 8;
        int max_out = max_new > 0 ? max_new : 1;
        uint32_t *hist = (uint32_t *)malloc(sizeof(uint32_t) *
                                            (size_t)(n_ids + max_out));
        uint32_t *sa = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)max_out);
        uint32_t *sb = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)max_out);
        uint32_t *sc = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)max_out);
        mm_pass_res r1, r2, r3;
        printf("\n=== ab: plain / spec=%d / plain again ===\n", ab_n);
        if (mm_pass(&m, pid, n_ids, max_new, 0, hist, sa, &r1) != 0) return 1;
        if (mm_pass(&m, pid, n_ids, max_new, ab_n, hist, sb, &r2) != 0) return 1;
        if (mm_pass(&m, pid, n_ids, max_new, 0, hist, sc, &r3) != 0) return 1;
        const char *nm[3] = { "plain", "spec", "plain2" };
        mm_pass_res *rs[3] = { &r1, &r2, &r3 };
        const uint32_t *outs[3] = { sa, sb, sc };
        for (int i = 0; i < 3; i++) {
            mm_pass_res *r = rs[i];
            printf("  %-6s %3lld tok in %7.0f ms = %5.3f tok/s  (%lld forwards, "
                   "%.2f tok each", nm[i], (long long)r->tokens, r->ms,
                   r->ms > 0 ? (double)r->tokens / (r->ms / 1000.0) : 0.0,
                   (long long)r->steps,
                   r->steps > 0 ? (double)r->tokens / (double)r->steps : 0.0);
            if (r->proposed)
                printf(", accepted %lld/%lld = %.0f%%", (long long)r->accepted,
                       (long long)r->proposed,
                       100.0 * (double)r->accepted / (double)r->proposed);
            printf(")\n");
        }
        for (int i = 1; i < 3; i++) {
            int64_t diff = -1;
            int64_t n = rs[0]->tokens < rs[i]->tokens ? rs[0]->tokens : rs[i]->tokens;
            for (int64_t k = 0; k < n; k++)
                if (outs[0][k] != outs[i][k]) { diff = k; break; }
            printf("  %s vs plain: %s", nm[i],
                   diff < 0 ? (rs[0]->tokens == rs[i]->tokens
                                   ? "identical token for token"
                                   : "identical over the shorter run")
                            : "DIFFERS");
            if (diff >= 0)
                printf(" at token %lld: plain %u, %s %u", (long long)diff,
                       outs[0][diff], nm[i], outs[i][diff]);
            printf("\n");
        }
        printf("  speed: spec %.2fx plain, %.2fx the warm plain pass\n",
               r1.ms > 0 ? r1.ms / r2.ms : 0.0, r3.ms > 0 ? r3.ms / r2.ms : 0.0);
        free(hist); free(sa); free(sb); free(sc);
        free(pid); free(ids);
        mm_model_close(&m);
        cwick_pool_shutdown();
        return 0;
    }

    float *logits = (float *)cwick_xmalloc(sizeof(float) *
                                           (size_t)m.cfg.vocab_size);

    /* ---- prefill ---- */
    t0 = now_ms();
    if (mm_model_forward(&m, pid, (int)n_ids, 0, logits, &err) != 0) {
        fprintf(stderr, "prefill: %s\n", err.message);
        return 1;
    }
    double prefill_ms = now_ms() - t0;
    printf("prefill %.0f ms for %lld tokens (%.1f tok/s)\n", prefill_ms,
           (long long)n_ids,
           (double)n_ids / (prefill_ms / 1000.0));
    printf("  next-token distribution:\n");
    print_top(logits, m.cfg.vocab_size, topk);
    int last_top[16];
    float last_topv[16];
    for (int i = 0; i < topk && i < 16; i++) {
        last_top[i] = top_ids[i];
        last_topv[i] = top_vals[i];
    }

    /* ---- decode ---- */
    int64_t pos = n_ids;
    int32_t cur = (int32_t)argmax(logits, m.cfg.vocab_size);
    double dec_ms = 0.0;
    uint32_t *out_ids = (uint32_t *)malloc(sizeof(uint32_t) *
                                           (size_t)(max_new > 0 ? max_new : 1));
    int n_out = 0;
    printf("decode:");
    for (int s = 0; s < max_new; s++) {
        printf(" %d", cur);
        fflush(stdout);
        if (n_out < max_new) out_ids[n_out++] = (uint32_t)cur;
        double ts = now_ms();
        if (mm_model_forward(&m, &cur, 1, pos, logits, &err) != 0) {
            fprintf(stderr, "\nstep %d: %s\n", s, err.message);
            return 1;
        }
        dec_ms += now_ms() - ts;
        pos++;
        cur = (int32_t)argmax(logits, m.cfg.vocab_size);
        if (trace) {
            printf("\n  step %d (%.1f ms):\n", s, now_ms() - ts);
            print_top(logits, m.cfg.vocab_size, topk);
        }
        if (cur == 200020 || cur == 200019) break;
    }
    printf("\ndecode %.1f tok/s over %d steps (%.0f ms)\n",
           (double)max_new / (dec_ms / 1000.0), max_new, dec_ms);

    /* ---- what the ids actually mean ----
     * A coherent checkpoint answers "The capital of France is" with " Paris".
     * Reading the predicted ids back as text is the only check available
     * without a reference implementation, so it is worth doing for the
     * predictions as well as for the prompt. */
    {
        cwick_model_spec spec;
        memset(&spec, 0, sizeof(spec));
        spec.vocab_size = m.cfg.vocab_size;
        cwick_tokenizer *tok = NULL;
        if (cwick_tokenizer_load(root, &spec, &tok, &err) == CWICK_OK) {
            char *txt = NULL;
            if (cwick_tokenizer_decode(tok, ids, n_ids, &txt, &err) == CWICK_OK) {
                printf("prompt text: %s\n", txt);
                free(txt);
            }
            printf("generated text: [");
            for (int i = 0; i < n_out; i++) {
                char *one = NULL;
                if (cwick_tokenizer_decode(tok, &out_ids[i], 1, &one, &err) ==
                    CWICK_OK) {
                    fputs(one, stdout);
                    free(one);
                }
            }
            printf("]\n");

            printf("top predictions as text:\n");
            for (int i = 0; i < topk; i++) {
                uint32_t id = (uint32_t)last_top[i];
                char *one = NULL;
                printf("  [%6d] ", last_top[i]);
                if (cwick_tokenizer_decode(tok, &id, 1, &one, &err) == CWICK_OK) {
                    printf("'%s'  %.4f\n", one, last_topv[i]);
                    free(one);
                } else printf("\n");
            }
            cwick_tokenizer_free(tok);
        }
    }

    printf("expert reads %llu, %.2f GiB\n",
           (unsigned long long)m.n_expert_reads,
           (double)m.n_expert_bytes / (1024.0 * 1024.0 * 1024.0));
    if (m.stream) {
        uint64_t bs = 0, rc = 0, ch = 0, cm = 0;
        cwick_stream_stats(m.stream, &bs, &rc, &ch, &cm);
        printf("pin cache: %.1f%% hit (%llu/%llu), %llu read calls, "
               "%.2f GiB streamed, footprint %.2f GiB\n",
               (ch + cm) ? 100.0 * (double)ch / (double)(ch + cm) : 0.0,
               (unsigned long long)ch, (unsigned long long)cm,
               (unsigned long long)rc, cwick_gib(bs),
               cwick_gib(cwick_process_footprint_bytes()));
    } else {
        printf("pin cache: off, footprint %.2f GiB\n",
               cwick_gib(cwick_process_footprint_bytes()));
    }

    free(logits);
    free(pid);
    free(ids);
    mm_model_close(&m);
    cwick_pool_shutdown();
    return 0;
}
