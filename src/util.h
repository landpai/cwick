/* util.h — internal helpers shared by cwick sources. */
#ifndef CWICK_UTIL_H
#define CWICK_UTIL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

void *cwick_xmalloc(size_t n);
void *cwick_xrealloc(void *p, size_t n);
char *cwick_strdup(const char *s);
bool cwick_env_true(const char *name);

int cwick_read_file(const char *path, char **out, size_t *out_len);
const void *cwick_mmap_file(const char *path, size_t *out_len);
void cwick_munmap(const void *base, size_t len);
int cwick_write_file(const char *path, const void *data, size_t n);
int cwick_mkdir_p(const char *path);
int cwick_file_exists(const char *path);
int cwick_dir_exists(const char *path);
void cwick_path_join(char *out, size_t cap, const char *a, const char *b);

/* First non-loopback, non-link-local IPv4 address (e.g. en0's 192.168.x.x).
 * Used only to print the LAN URL / build a share QR; returns false when the
 * machine is offline or has no address a phone on the same Wi-Fi could use. */
bool cwick_lan_ipv4(char *out, size_t cap);

float cwick_bf16_to_f32(uint16_t b);
uint16_t cwick_f32_to_bf16(float f);

/* Hot-path BF16 decode: the 16-bit left shift that defines the format, and
 * nothing else.
 *
 * cwick_bf16_to_f32 lives in util.c and canonicalises NaN/inf, which is right
 * for reading a file that might be malformed and wrong for an inner loop.
 * The absorbed MLA kernel calls it once per weight element: 64 heads x 256 x
 * 512 twice per sparse layer, ~92 M calls per decode step, and a call that
 * cannot be inlined costs more than the multiply it feeds. Measured as a
 * 2.5x slowdown of the whole attention phase (423 ms/step absorbed vs
 * 172 ms/step expanded).
 *
 * A shifted BF16 NaN still produces a float NaN, and inf stays inf, so the
 * arithmetic is identical for every value the kernels can see. */
static inline float cwick_bf16_fast(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

#endif /* CWICK_UTIL_H */
