/* util.c — errors, memory, mmap, file I/O helpers. */
#include "cwick.h"
#include "util.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <errno.h>

void cwick_error_clear(cwick_error *err) {
    if (err) err->status = CWICK_OK, err->message[0] = '\0';
}

void cwick_error_set(cwick_error *err, cwick_status status, const char *fmt, ...) {
    if (!err) return;
    err->status = status;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err->message, sizeof(err->message), fmt, ap);
    va_end(ap);
}

const char *cwick_error_message(const cwick_error *err) {
    return err && err->message[0] ? err->message : "no error";
}

void *cwick_xmalloc(size_t n) {
    void *p = calloc(1, n ? n : 1);
    if (!p) {
        fprintf(stderr, "cwick: out of memory (%zu bytes)\n", n);
        abort();
    }
    return p;
}

void *cwick_xrealloc(void *p, size_t n) {
    void *q = realloc(p, n);
    if (!q) {
        fprintf(stderr, "cwick: out of memory (%zu bytes)\n", n);
        abort();
    }
    return q;
}

char *cwick_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = cwick_xmalloc(n);
    memcpy(p, s, n);
    return p;
}

bool cwick_env_true(const char *name) {
    const char *v = getenv(name);
    if (!v || !*v) return false;
    return strcmp(v, "1") == 0 || strcasecmp(v, "true") == 0 ||
           strcasecmp(v, "yes") == 0 || strcasecmp(v, "on") == 0;
}

/* Read a whole file into a malloc'd buffer. */
int cwick_read_file(const char *path, char **out, size_t *out_len) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return -1; }
    size_t n = (size_t)st.st_size;
    char *buf = cwick_xmalloc(n + 1);
    size_t off = 0;
    while (off < n) {
        ssize_t r = read(fd, buf + off, n - off);
        if (r <= 0) { close(fd); free(buf); return -1; }
        off += (size_t)r;
    }
    buf[n] = '\0';
    close(fd);
    *out = buf;
    *out_len = n;
    return 0;
}

/* Memory-map a file read-only; returns base or NULL. */
const void *cwick_mmap_file(const char *path, size_t *out_len) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0) { close(fd); return NULL; }
    size_t n = (size_t)st.st_size;
    if (n == 0) { close(fd); *out_len = 0; return (const void *)""; }
    void *p = mmap(NULL, n, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return NULL;
    /* Advise sequential for the big weight spines. */
    madvise(p, n, MADV_SEQUENTIAL);
    *out_len = n;
    return p;
}

void cwick_munmap(const void *base, size_t len) {
    if (base && len) munmap((void *)(uintptr_t)base, len);
}

int cwick_write_file(const char *path, const void *data, size_t n) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return -1;
    size_t off = 0;
    while (off < n) {
        ssize_t w = write(fd, (const char *)data + off, n - off);
        if (w <= 0) { close(fd); return -1; }
        off += (size_t)w;
    }
    close(fd);
    return 0;
}

int cwick_mkdir_p(const char *path) {
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s", path);
    size_t len = strlen(tmp);
    if (len == 0) return 0;
    if (tmp[len - 1] == '/') tmp[len - 1] = '\0';
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
            *p = '/';
        }
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
    return 0;
}

int cwick_file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

int cwick_dir_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

/* Join two path components into out (capacity cap). */
void cwick_path_join(char *out, size_t cap, const char *a, const char *b) {
    snprintf(out, cap, "%s/%s", a, b);
}

/* bf16 <-> f32 conversion helpers (round-to-nearest-even, as the checkpoint
 * stores its spine in bfloat16). */
float cwick_bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    uint32_t sign = u & 0x80000000u;
    uint32_t exp = u & 0x7F800000u;
    uint32_t man = u & 0x007FFFFFu;
    if (exp == 0x7F800000u && man != 0) { /* NaN -> quiet NaN */
        return 0.0f / 0.0f;
    }
    if (exp == 0x7F800000u) { /* inf */
        return sign ? -(1.0f / 0.0f) : (1.0f / 0.0f);
    }
    if (exp == 0 && man == 0) { /* +/-0 */
        return sign ? -0.0f : 0.0f;
    }
    if (exp == 0 || exp == 0x7F800000u - 0x00800000u) {
        /* subnormal / tiny: normalize */
        uint32_t e = exp ? (uint32_t)(exp >> 23) : 0;
        uint32_t m = man;
        if (exp == 0) {
            if (m == 0) return sign ? -0.0f : 0.0f;
            while (!(m & 0x00800000u)) { m <<= 1; e--; }
            m &= 0x007FFFFFu;
        }
        e += 127;
        uint32_t bits = (sign) | (e << 23) | m;
        float f;
        memcpy(&f, &bits, 4);
        return f;
    }
    /* normal */
    uint32_t e = (uint32_t)(exp >> 23) - 127 + 127;
    uint32_t bits = (sign) | (e << 23) | man;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

uint16_t cwick_f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    /* round-to-nearest-even on the retained 16 bits */
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t round = ((u >> 15) & 1u) + lsb;
    uint32_t r = u + round;
    return (uint16_t)(r >> 16);
}

bool cwick_lan_ipv4(char *out, size_t cap) {
    if (!out || cap < 8) return false;
    /* The default route's source address is the one a phone on the same Wi-Fi
     * is most likely to reach. A UDP connect sends no packet; it only asks
     * the kernel which local address it would use. */
    {
        int s = socket(AF_INET, SOCK_DGRAM, 0);
        if (s >= 0) {
            struct sockaddr_in dst;
            memset(&dst, 0, sizeof(dst));
            dst.sin_family = AF_INET;
            dst.sin_port = htons(53);
            dst.sin_addr.s_addr = htonl(0x08080808u);  /* 8.8.8.8 */
            if (connect(s, (struct sockaddr *)&dst, sizeof(dst)) == 0) {
                struct sockaddr_in self;
                socklen_t slen = sizeof(self);
                if (getsockname(s, (struct sockaddr *)&self, &slen) == 0 &&
                    self.sin_family == AF_INET) {
                    uint32_t ip = ntohl(self.sin_addr.s_addr);
                    if ((ip >> 24) != 127 && (ip >> 16) != 0xA9FE &&
                        inet_ntop(AF_INET, &self.sin_addr, out, (socklen_t)cap)) {
                        close(s);
                        return true;
                    }
                }
            }
            close(s);
        }
    }
    struct ifaddrs *ifs = NULL;
    if (getifaddrs(&ifs) != 0) return false;
    bool found = false;
    /* Prefer Wi-Fi/Ethernet names, but accept any up, non-loopback IPv4. */
    const char *preferred[] = {"en0", "en1", "en2", "bridge0", NULL};
    for (int pass = 0; pass < 2 && !found; pass++) {
        for (struct ifaddrs *ifa = ifs; ifa && !found; ifa = ifa->ifa_next) {
            if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET) continue;
            if (!(ifa->ifa_flags & IFF_UP) || (ifa->ifa_flags & IFF_LOOPBACK)) continue;
            if (pass == 0) {
                bool match = false;
                for (int i = 0; preferred[i]; i++)
                    if (strcmp(ifa->ifa_name, preferred[i]) == 0) { match = true; break; }
                if (!match) continue;
            }
            struct sockaddr_in *sin = (struct sockaddr_in *)ifa->ifa_addr;
            uint32_t ip = ntohl(sin->sin_addr.s_addr);
            if ((ip >> 24) == 127 || (ip >> 16) == 0xA9FE) continue; /* loopback/link-local */
            if (!inet_ntop(AF_INET, &sin->sin_addr, out, (socklen_t)cap)) continue;
            found = true;
        }
    }
    freeifaddrs(ifs);
    return found;
}
