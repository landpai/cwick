/* mm_decodeprobe — the two hardware ceilings a decode step is scored against.
 *
 * Before tuning a decode loop it is worth knowing what the machine can do, so
 * that "slow" can be attributed instead of argued about. Two numbers decide
 * almost everything for a memory-bound MoE:
 *
 *   --mem   the read bandwidth of ordinary DRAM as a function of thread count.
 *           This is the ceiling for any step whose weights are resident; on a
 *           128 GiB Apple Silicon box it saturates at four threads and does not
 *           improve with sixteen.
 *
 *   --ssd   the read bandwidth of the checkpoint's shards, split by whether the
 *           threads read different files or the same one. A decode layer
 *           misses three or four experts, and those usually live in one shard,
 *           so the same-file number -- not the cross-file one -- is what bounds
 *           a step. They are far apart, which is why the distinction has to be
 *           measured rather than assumed.
 *
 * Both modes verify a byte of every buffer they read, so a number that comes
 * out of page cache is reported as such instead of being mistaken for the
 * device.
 *
 * Everything here is measurement only: no model, no kernels, no arithmetic.
 * Build: `make mmdecodeprobe`.
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* ------------------------------------------------------------------ */
/* DRAM read bandwidth                                                 */
/* ------------------------------------------------------------------ */

typedef struct {
    const uint8_t *p;
    size_t bytes;
    int64_t stride;   /* interleave step, in chunks */
    int id, nthreads;
    uint64_t sum;
} mem_job;

#define MEM_CHUNK 4096

static void *mem_worker(void *ap) {
    mem_job *j = (mem_job *)ap;
    /* One contiguous slice per thread: a strided walk measures prefetcher
     * behaviour on the stride, not the DRAM. */
    size_t per = j->bytes / (size_t)j->nthreads;
    const uint8_t *p = j->p + per * (size_t)j->id;
    size_t n = (j->id == j->nthreads - 1) ? j->bytes - per * (size_t)j->id : per;
    uint64_t s = 0;
    size_t i = 0;
#if defined(__ARM_NEON)
    uint8x16_t a0 = vdupq_n_u8(0), a1 = a0, a2 = a0, a3 = a0;
    for (; i + 128 <= n; i += 128) {
        a0 = veorq_u8(a0, vld1q_u8(p + i));
        a1 = veorq_u8(a1, vld1q_u8(p + i + 16));
        a2 = veorq_u8(a2, vld1q_u8(p + i + 32));
        a3 = veorq_u8(a3, vld1q_u8(p + i + 48));
        a0 = veorq_u8(a0, vld1q_u8(p + i + 64));
        a1 = veorq_u8(a1, vld1q_u8(p + i + 80));
        a2 = veorq_u8(a2, vld1q_u8(p + i + 96));
        a3 = veorq_u8(a3, vld1q_u8(p + i + 112));
    }
    a0 = veorq_u8(veorq_u8(a0, a1), veorq_u8(a2, a3));
    s += vgetq_lane_u64(vreinterpretq_u64_u8(a0), 0) +
         vgetq_lane_u64(vreinterpretq_u64_u8(a0), 1);
#endif
    for (; i < n; i++) s += p[i];
    j->sum = s;
    return NULL;
}

static void run_mem(double gib, int nthreads) {
    size_t bytes = (size_t)(gib * 1073741824.0);
    uint8_t *p = malloc(bytes);
    if (!p) {
        printf("  mem   cannot allocate %.1f GiB\n", gib);
        return;
    }
    memset(p, 1, bytes);       /* fault it in: we want DRAM, not zero pages */
    mem_job *js = calloc((size_t)nthreads, sizeof(*js));
    pthread_t *th = calloc((size_t)nthreads, sizeof(*th));
    double t0 = now_ms();
    for (int i = 0; i < nthreads; i++) {
        js[i].p = p; js[i].bytes = bytes; js[i].id = i; js[i].nthreads = nthreads;
        pthread_create(&th[i], NULL, mem_worker, &js[i]);
    }
    uint64_t sum = 0;
    for (int i = 0; i < nthreads; i++) { pthread_join(th[i], NULL); sum += js[i].sum; }
    double ms = now_ms() - t0;
    printf("  mem   %5.1f GiB / %2d thread(s): %8.1f ms  %6.2f GB/s  (sum %08x)\n",
           gib, nthreads, ms, (double)bytes / (ms * 1e-3) / 1e9,
           (unsigned)(sum & 0xffffffffu));
    free(th); free(js); free(p);
}

/* ------------------------------------------------------------------ */
/* shard read bandwidth                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *path;
    int fd;
    size_t bytes;     /* per read */
    int nreads;
    int64_t span;     /* how far into the file the reads walk */
    int verify;
    uint64_t sum;
} ssd_job;

static void *ssd_worker(void *ap) {
    ssd_job *j = (ssd_job *)ap;
    uint8_t *buf = malloc(j->bytes);
    if (!buf) return NULL;
    struct stat st;
    if (fstat(j->fd, &st) != 0 || st.st_size <= 0) { free(buf); return NULL; }
    int64_t limit = st.st_size - (int64_t)j->bytes;
    if (limit < 0) limit = 0;
    uint64_t s = 0;
    for (int r = 0; r < j->nreads; r++) {
        int64_t off = j->span > 0 ? (limit > 0 ? (int64_t)((double)r * (double)limit /
                                                          (double)j->nreads) : 0) : 0;
        ssize_t got = pread(j->fd, buf, j->bytes, off);
        if (got <= 0) break;
        s += buf[0] + buf[got - 1];
        if (j->verify) {
            uint64_t h = 0;
            for (size_t i = 0; i < (size_t)got; i += 4096) h += buf[i];
            s += h;
        }
    }
    j->sum = s;
    free(buf);
    return NULL;
}

static void shards_of(const char *root, char ***out, int *n) {
    /* Either one shard file or a model directory whose *.safetensors are
     * enumerated. `across` only means anything with more than one file. */
    struct stat st;
    char **v = calloc(1024, sizeof(char *));
    int c = 0;
    if (stat(root, &st) == 0 && S_ISREG(st.st_mode)) {
        v[c++] = strdup(root);
    } else {
        DIR *d = opendir(root);
        if (!d) { free(v); *out = NULL; *n = 0; return; }
        struct dirent *de;
        while ((de = readdir(d)) && c < 1024) {
            size_t l = strlen(de->d_name);
            if (l < 12 || strcmp(de->d_name + l - 12, ".safetensors") != 0) continue;
            char *p = malloc(strlen(root) + l + 2);
            sprintf(p, "%s/%s", root, de->d_name);
            v[c++] = p;
        }
        closedir(d);
    }
    *out = v; *n = c;
}

static void run_ssd(const char *root, int nreads, int nthreads, size_t bytes,
                    int one_file, int verify) {
    char **sh = NULL;
    int nsh = 0;
    shards_of(root, &sh, &nsh);
    if (nsh == 0) return;
    pthread_t *th = calloc((size_t)nthreads, sizeof(*th));
    ssd_job *js = calloc((size_t)nthreads, sizeof(*js));
    int *opened = calloc((size_t)nthreads, sizeof(int));
    int nf = 0;
    double t0 = now_ms();
    for (int i = 0; i < nthreads; i++) {
        /* A descriptor per thread. Sharing one was measured and makes no
         * difference: the ceiling is per file, not per descriptor. */
        int fd = open(sh[one_file ? 0 : (i % nsh)], O_RDONLY);
        if (fd < 0) continue;
#if defined(__APPLE__)
        /* Without this the shard is cached after the first pass and the rest of
         * the run measures memory (measured: 10.9 -> 49.4 GB/s at 8 threads).
         * The engine reads through F_NOCACHE too, so this is the honest one. */
        (void)fcntl(fd, F_NOCACHE, 1);
#endif
        js[nf].path = sh[one_file ? 0 : (i % nsh)];
        js[nf].fd = fd;
        js[nf].bytes = bytes;
        js[nf].nreads = nreads;
        js[nf].span = 1;
        js[nf].verify = verify;
        pthread_create(&th[nf], NULL, ssd_worker, &js[nf]);
        opened[nf] = fd;
        nf++;
    }
    uint64_t sum = 0;
    for (int i = 0; i < nf; i++) { pthread_join(th[i], NULL); sum += js[i].sum; }
    double ms = now_ms() - t0;
    double total = (double)bytes * (double)nreads * (double)nf;
    printf("  ssd   %s  %4d x %5.1f MiB x %2d thread(s): %8.1f ms  %6.2f GB/s  (sum %08x)\n",
           one_file ? "same shard  " : "across shards", nreads,
           (double)bytes / 1048576.0, nf, ms, total / (ms * 1e-3) / 1e9,
           (unsigned)(sum & 0xffffffffu));
    for (int i = 0; i < nf; i++) close(opened[i]);
    free(opened); free(js); free(th);
    for (int i = 0; i < nsh; i++) free(sh[i]);
    free(sh);
}

static void usage(void) {
    printf("usage:\n"
           "  mm_decodeprobe --mem GiB [threads...]\n"
           "  mm_decodeprobe --ssd PATH READS THREADS MiB [same|across]\n"
           "\n"
           "  --mem  read bandwidth of resident memory, one line per thread count\n"
           "  --ssd  pread bandwidth of the shards; `same` puts every thread on one\n"
           "         file (what a decode layer does), `across` spreads them\n");
}

int main(int argc, char **argv) {
    if (argc >= 3 && strcmp(argv[1], "--mem") == 0) {
        double gib = atof(argv[2]);
        printf("\n[a] resident memory read bandwidth\n");
        if (argc > 3) {
            for (int i = 3; i < argc; i++) run_mem(gib, atoi(argv[i]));
        } else {
            int tc[] = {1, 2, 4, 8, 12, 16};
            for (size_t i = 0; i < sizeof(tc) / sizeof(tc[0]); i++) run_mem(gib, tc[i]);
        }
        return 0;
    }
    if (argc >= 6 && strcmp(argv[1], "--ssd") == 0) {
        int reads = atoi(argv[3]);
        int threads = atoi(argv[4]);
        size_t mb = (size_t)atoll(argv[5]);
        int one = argc > 6 && strcmp(argv[6], "same") == 0;
        printf("\n[b] shard read bandwidth (data verified)\n");
        run_ssd(argv[2], reads, threads, mb * 1024 * 1024, one, 1);
        return 0;
    }
    usage();
    return 1;
}
