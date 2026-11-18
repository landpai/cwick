/* rtws.c — local harness: drive /v1/realtime as a WebSocket client.
 *
 * Not part of the build. Streams a 24 kHz mono WAV in as input_audio_buffer
 * appends, prints every server event with a client-side timestamp, and writes
 * the returned audio deltas to a WAV so the reply can be listened to.
 *
 *   tools/rtws --in /tmp/ask24.wav --out /tmp/reply.wav [--voice vivian_zh]
 *              [--burst] [--turns 2] [--text "..."] [--vad] [--token BEARER]
 */
#include <arpa/inet.h>
#include <errno.h>
#include <math.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "json.h"
#include "util.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static double t0;
static void say(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    printf("[%7.3f] ", now_s() - t0);
    vprintf(fmt, ap);
    va_end(ap);
    fflush(stdout);
}

/* ---- socket + RFC 6455 client half ---------------------------------- */

static int dial(const char *host, int port) {
    struct sockaddr_in a;
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &a.sin_addr) != 1) {
        struct hostent *he = gethostbyname(host);
        if (!he) return -1;
        memcpy(&a.sin_addr, he->h_addr_list[0], (size_t)he->h_length);
    }
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    if (connect(fd, (struct sockaddr *)&a, sizeof(a)) != 0) { close(fd); return -1; }
    struct timeval tv = { .tv_sec = 120, .tv_usec = 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    return fd;
}

static int ws_write_all(int fd, const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    while (n) {
        ssize_t k = write(fd, b, n);
        if (k <= 0) { if (k < 0 && errno == EINTR) continue; return -1; }
        b += k; n -= (size_t)k;
    }
    return 0;
}

static int ws_read_all(int fd, void *p, size_t n) {
    unsigned char *b = (unsigned char *)p;
    while (n) {
        ssize_t k = read(fd, b, n);
        if (k <= 0) { if (k < 0 && errno == EINTR) continue; return -1; }
        b += k; n -= (size_t)k;
    }
    return 0;
}

static char g_auth[512];

static int ws_handshake(int fd, const char *host, int port, const char *path) {
    char req[1024];
    int n = snprintf(req, sizeof(req),
                     "GET %s HTTP/1.1\r\nHost: %s:%d\r\nUpgrade: websocket\r\n"
                     "Connection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                     "Sec-WebSocket-Version: 13\r\n%s\r\n", path, host, port, g_auth);
    if (ws_write_all(fd, req, (size_t)n)) return -1;
    char line[2048];
    size_t used = 0;
    int status = 0;
    /* Read the response head one byte at a time: anything after the blank line
     * belongs to the first frame and must stay in the socket. */
    for (;;) {
        char c;
        if (ws_read_all(fd, &c, 1)) return -1;
        if (used + 1 >= sizeof(line)) return -1;
        line[used++] = c;
        line[used] = 0;
        if (used >= 4 && strcmp(line + used - 4, "\r\n\r\n") == 0) break;
        if (used == 12 && sscanf(line, "HTTP/1.1 %d", &status) != 1) return -1;
    }
    if (status != 101) { fprintf(stderr, "handshake: %s", line); return -1; }
    return 0;
}

static uint32_t rng_s = 0x2f6e2b1u;
static uint32_t rng(void) { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 17; rng_s ^= rng_s << 5; return rng_s; }

static int ws_send_text(int fd, const char *s) {
    size_t n = strlen(s);
    unsigned char h[14];
    size_t hn = 2;
    h[0] = 0x81;
    if (n < 126) h[1] = (unsigned char)(0x80 | n);
    else if (n < 65536) { h[1] = 0x80 | 126; h[2] = (unsigned char)(n >> 8); h[3] = (unsigned char)n; hn = 4; }
    else {
        h[1] = 0x80 | 127;
        for (int i = 0; i < 8; i++) h[2 + i] = (unsigned char)((uint64_t)n >> (56 - 8 * i));
        hn = 10;
    }
    uint32_t m = rng();
    memcpy(h + hn, &m, 4); hn += 4;
    if (ws_write_all(fd, h, hn)) return -1;
    unsigned char *masked = (unsigned char *)malloc(n ? n : 1);
    if (!masked) return -1;
    const unsigned char *mb = (const unsigned char *)&m;
    for (size_t i = 0; i < n; i++) masked[i] = (unsigned char)s[i] ^ mb[i & 3];
    int rc = ws_write_all(fd, masked, n);
    free(masked);
    return rc;
}

/* One complete text message. Returns 0 on close, -1 on error. */
static char *ws_recv(int fd) {
    unsigned char *msg = NULL;
    size_t len = 0, cap = 0;
    for (;;) {
        unsigned char h[2];
        if (ws_read_all(fd, h, 2)) { free(msg); return NULL; }
        int fin = (h[0] & 0x80) != 0, op = h[0] & 0x0f;
        int masked = (h[1] & 0x80) != 0;
        uint64_t n = h[1] & 0x7f;
        if (n == 126) { unsigned char e[2]; if (ws_read_all(fd, e, 2)) { free(msg); return NULL; } n = ((uint64_t)e[0] << 8) | e[1]; }
        else if (n == 127) {
            unsigned char e[8]; if (ws_read_all(fd, e, 8)) { free(msg); return NULL; }
            n = 0; for (int i = 0; i < 8; i++) n = (n << 8) | e[i];
        }
        unsigned char mk[4];
        if (masked && ws_read_all(fd, mk, 4)) { free(msg); return NULL; }
        unsigned char *pl = (unsigned char *)malloc((size_t)n ? (size_t)n : 1);
        if (!pl) { free(msg); return NULL; }
        if (n && ws_read_all(fd, pl, (size_t)n)) { free(pl); free(msg); return NULL; }
        if (masked) for (uint64_t i = 0; i < n; i++) pl[i] ^= mk[i & 3];
        if (op == 0x8) { free(pl); free(msg); return NULL; }
        if (op == 0x9 || op == 0xa) { free(pl); continue; }
        if (len + n + 1 > cap) {
            cap = (len + (size_t)n + 1) * 2;
            unsigned char *g = (unsigned char *)realloc(msg, cap);
            if (!g) { free(pl); free(msg); return NULL; }
            msg = g;
        }
        memcpy(msg + len, pl, (size_t)n);
        len += (size_t)n;
        free(pl);
        msg[len] = 0;
        if (fin) return (char *)msg;
    }
}

/* ---- WAV ------------------------------------------------------------- */

static int16_t *wav_read(const char *path, int *nsamples, int *rate) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    unsigned char h[44];
    if (fread(h, 1, 44, f) != 44 || memcmp(h, "RIFF", 4) || memcmp(h + 8, "WAVE", 4)) {
        fprintf(stderr, "%s is not a 44-byte-header WAV\n", path); fclose(f); return NULL;
    }
    int ch = h[22] | (h[23] << 8), bits = h[34] | (h[35] << 8);
    *rate = h[24] | (h[25] << 8) | (h[26] << 16) | (h[27] << 24);
    unsigned dsz = (unsigned)h[40] | ((unsigned)h[41] << 8) | ((unsigned)h[42] << 16) | ((unsigned)h[43] << 24);
    if (ch != 1 || bits != 16) { fprintf(stderr, "%s: need mono 16-bit\n", path); fclose(f); return NULL; }
    size_t n = dsz / 2;
    int16_t *p = (int16_t *)malloc(n * 2 + 2);
    if (!p) { fclose(f); return NULL; }
    if (fread(p, 2, n, f) != n) { free(p); fclose(f); return NULL; }
    fclose(f);
    *nsamples = (int)n;
    return p;
}

static void wav_write(const char *path, const int16_t *p, int n, int rate) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    unsigned dsz = (unsigned)n * 2, rsz = dsz + 36;
    unsigned char h[44] = { 'R','I','F','F', 0,0,0,0, 'W','A','V','E', 'f','m','t',' ',
                            16,0,0,0, 1,0, 1,0, 0,0,0,0, 0,0,0,0, 2,0, 16,0,
                            'd','a','t','a', 0,0,0,0 };
    for (int i = 0; i < 4; i++) h[4 + i] = (unsigned char)(rsz >> (8 * i));
    for (int i = 0; i < 4; i++) h[24 + i] = (unsigned char)((unsigned)rate >> (8 * i));
    unsigned br = (unsigned)rate * 2;
    for (int i = 0; i < 4; i++) h[28 + i] = (unsigned char)(br >> (8 * i));
    for (int i = 0; i < 4; i++) h[40 + i] = (unsigned char)(dsz >> (8 * i));
    fwrite(h, 1, 44, f);
    fwrite(p, 2, (size_t)n, f);
    fclose(f);
}

/* ---- event plumbing -------------------------------------------------- */

typedef struct {
    int16_t *pcm;
    int n, cap;
    char transcript[8192];
    int tlen;
    double first_audio, commit_at, create_at;
    int deltas, done;
    char usage[4096];
} turn_out;

static void out_pcm(turn_out *o, const char *b64) {
    size_t n = 0;
    void *raw = cwick_b64_decode(b64, strlen(b64), &n);
    if (!raw) return;
    int ns = (int)(n / 2);
    if (o->n + ns > o->cap) {
        o->cap = (o->n + ns) * 2 + 4800;
        o->pcm = (int16_t *)realloc(o->pcm, (size_t)o->cap * 2);
    }
    memcpy(o->pcm + o->n, raw, (size_t)ns * 2);
    o->n += ns;
    free(raw);
}

static const char *jstr(json_value *v, const char *k) {
    const char *s = cwick_json_str(cwick_json_get(v, k));
    return s ? s : "";
}

/* Consume events until response.done (or error). Returns 0 on success. */
static int drain(int fd, turn_out *o, int verbose) {
    for (;;) {
        char *m = ws_recv(fd);
        if (!m) { say("connection closed by server"); return -1; }
        cwick_error err;
        cwick_error_clear(&err);
        json_value *v = cwick_json_parse(m, strlen(m), &err);
        if (!v) { free(m); continue; }
        const char *type = jstr(v, "type");
        if (strcmp(type, "response.audio.delta") == 0 ||
            strcmp(type, "response.output_audio.delta") == 0) {
            const char *d = jstr(v, "delta");
            out_pcm(o, d);
            o->deltas++;
            if (o->first_audio == 0.0) {
                o->first_audio = now_s();
                say("first audio delta (%.3f s after commit, %.3f s after response.create)",
                    o->first_audio - o->commit_at, o->first_audio - o->create_at);
            }
            if (verbose) say("%s #%d (%.2f s audio so far)", type, o->deltas, o->n / 24000.0);
        } else if (strcmp(type, "response.audio_transcript.delta") == 0 ||
                   strcmp(type, "response.output_audio_transcript.delta") == 0) {
            const char *d = jstr(v, "delta");
            int k = (int)strlen(d);
            if (o->tlen + k < (int)sizeof(o->transcript)) {
                memcpy(o->transcript + o->tlen, d, (size_t)k);
                o->tlen += k;
                o->transcript[o->tlen] = 0;
            }
            if (verbose) say("transcript delta: %s", d);
        } else if (strcmp(type, "response.done") == 0) {
            snprintf(o->usage, sizeof(o->usage), "%s", m);
            o->done = 1;
            say("response.done");
            cwick_json_free(v);
            free(m);
            return 0;
        } else if (strcmp(type, "error") == 0) {
            say("ERROR event: %s", m);
            cwick_json_free(v);
            free(m);
            return -1;
        } else if (verbose || strcmp(type, "input_audio_buffer.speech_started") == 0 ||
                   strcmp(type, "input_audio_buffer.speech_stopped") == 0 ||
                   strcmp(type, "input_audio_buffer.committed") == 0) {
            say("%s", type);
        }
        cwick_json_free(v);
        free(m);
    }
}

static void send_json(int fd, const char *fmt, ...) {
    char buf[262144];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (ws_send_text(fd, buf)) { fprintf(stderr, "send failed\n"); exit(1); }
}

int main(int argc, char **argv) {
    t0 = now_s();
    const char *host = "127.0.0.1", *in = NULL, *out = "/tmp/rtws_out.wav";
    const char *voice = "vivian_zh", *text = NULL, *model = "Sophon-Realtime-32B-Tag";
    int port = 8000, turns = 1, verbose = 0, burst = 0, vad = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--in") && i + 1 < argc) in = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--voice") && i + 1 < argc) voice = argv[++i];
        else if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--text") && i + 1 < argc) text = argv[++i];
        else if (!strcmp(argv[i], "--turns") && i + 1 < argc) turns = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--burst")) burst = 1;
        else if (!strcmp(argv[i], "--vad")) vad = 1;
        else if (!strcmp(argv[i], "--token") && i + 1 < argc)
            snprintf(g_auth, sizeof(g_auth), "Authorization: Bearer %s\r\n", argv[++i]);
        else if (!strcmp(argv[i], "--verbose")) verbose = 1;
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    int ns = 0, rate = 0;
    int16_t *pcm = NULL;
    if (in) {
        pcm = wav_read(in, &ns, &rate);
        if (!pcm) return 1;
        say("input %s: %d samples, %d Hz (%.3f s)", in, ns, rate, (double)ns / rate);
    } else if (!text) {
        fprintf(stderr, "need --in WAV or --text\n");
        return 2;
    }
    int fd = dial(host, port);
    if (fd < 0) { fprintf(stderr, "connect %s:%d failed\n", host, port); return 1; }
    char path[512];
    snprintf(path, sizeof(path), "/v1/realtime?model=%s", model);
    if (ws_handshake(fd, host, port, path)) { close(fd); return 1; }
    say("websocket open");

    /* session.created + conversation.created arrive first */
    for (int i = 0; i < 2; i++) {
        char *m = ws_recv(fd);
        if (!m) { fprintf(stderr, "no greeting\n"); return 1; }
        if (verbose) say("<< %s", m);
        free(m);
    }
    send_json(fd,
              "{\"type\":\"session.update\",\"session\":{"
              "\"modalities\":[\"text\",\"audio\"],\"voice\":\"%s\","
              "\"input_audio_format\":\"pcm16\",\"output_audio_format\":\"pcm16\","
              "\"input_audio_sample_rate\":%d,"
              "\"input_audio_transcription\":null,"
              "\"turn_detection\":%s}}",
              voice, rate ? rate : 24000,
              vad ? "{\"type\":\"server_vad\",\"threshold\":0.5,"
                    "\"prefix_padding_ms\":300,\"silence_duration_ms\":500}"
                  : "null");
    { char *m = ws_recv(fd); if (m) { if (verbose) say("<< session.updated"); free(m); } }
    say("session configured (voice %s, vad %s)", voice, vad ? "on" : "off");

    int16_t *all = NULL;
    int all_n = 0, all_cap = 0;
    for (int t = 0; t < turns; t++) {
        turn_out o;
        memset(&o, 0, sizeof(o));
        if (text) {
            send_json(fd, "{\"type\":\"conversation.item.create\",\"item\":{"
                          "\"type\":\"message\",\"role\":\"user\",\"content\":[{"
                          "\"type\":\"input_text\",\"text\":\"%s\"}]}}", text);
            o.create_at = now_s();
            o.commit_at = o.create_at;
            send_json(fd, "{\"type\":\"response.create\"}");
            say("turn %d: text request sent", t + 1);
        } else {
            /* 200 ms appends at the wire rate, paced like a microphone unless
             * --burst asks for the fastest possible upload. */
            int chunk = rate / 5;
            double start = now_s();
            for (int p = 0; p < ns; p += chunk) {
                int k = ns - p < chunk ? ns - p : chunk;
                char *b64 = cwick_b64_encode(pcm + p, (size_t)k * 2);
                send_json(fd, "{\"type\":\"input_audio_buffer.append\",\"audio\":\"%s\"}", b64);
                free(b64);
                if (!burst) {
                    double want = start + (double)(p + k) / rate;
                    double left = want - now_s();
                    if (left > 0) usleep((useconds_t)(left * 1e6));
                }
            }
            say("turn %d: %d appends sent (%.3f s of audio, upload %.3f s)",
                t + 1, (ns + chunk - 1) / chunk, (double)ns / rate, now_s() - start);
            if (vad) {
                /* Server VAD closes the turn only after silence_duration_ms of
                 * silent *received* audio; a mic keeps streaming silence, so
                 * the harness must too or the turn never commits. */
                int16_t *sil = calloc((size_t)chunk, sizeof(int16_t));
                int sil_chunks = (rate * 4 / 5) / chunk; /* ~0.8 s */
                for (int q = 0; q < sil_chunks; q++) {
                    char *b64 = cwick_b64_encode(sil, (size_t)chunk * 2);
                    send_json(fd, "{\"type\":\"input_audio_buffer.append\",\"audio\":\"%s\"}", b64);
                    free(b64);
                    usleep((useconds_t)(chunk * 1e6 / rate));
                }
                free(sil);
                say("turn %d: trailing silence sent (VAD auto-commit)", t + 1);
            }
            o.commit_at = now_s();
            if (!vad) send_json(fd, "{\"type\":\"input_audio_buffer.commit\"}");
            o.create_at = now_s();
            if (!vad) send_json(fd, "{\"type\":\"response.create\"}");
            say("turn %d: %s", t + 1, vad ? "awaiting VAD commit" : "committed, response.create sent");
        }
        int rc = drain(fd, &o, verbose);
        double end = now_s();
        say("turn %d: %d audio deltas, %.3f s of audio, transcript \"%s\"",
            t + 1, o.deltas, o.n / 24000.0, o.transcript);
        say("turn %d wall clock %.3f s (client first-audio %.3f s)",
            t + 1, end - o.commit_at, o.first_audio ? o.first_audio - o.commit_at : 0.0);
        printf("%s\n", o.usage);
        if (o.n) {
            if (all_n + o.n > all_cap) {
                all_cap = (all_n + o.n) * 2 + 24000;
                all = (int16_t *)realloc(all, (size_t)all_cap * 2);
            }
            memcpy(all + all_n, o.pcm, (size_t)o.n * 2);
            all_n += o.n;
            if (turns == 1) wav_write(out, o.pcm, o.n, 24000);
        }
        free(o.pcm);
        if (rc) break;
    }
    if (turns > 1 && all_n) wav_write(out, all, all_n, 24000);
    if (all_n) say("wrote %s (%.3f s)", out, all_n / 24000.0);
    free(all);
    free(pcm);
    close(fd);
    return 0;
}