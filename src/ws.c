/* ws.c — see ws.h.
 *
 * The masking rule is the whole security surface of this file: a client frame
 * that arrives unmasked must be rejected, and the mask must be applied before
 * the payload is looked at. Everything else is bookkeeping.
 */
#include "ws.h"

#include <CommonCrypto/CommonCrypto.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "util.h"

bool cwick_ws_handshake(int fd, const char *key) {
    /* The GUID is fixed by the RFC; the accept value proves the peer's key was
     * seen, which is what stops a cross-protocol request from being replayed. */
    char joined[256];
    int n = snprintf(joined, sizeof(joined), "%s258EAFA5-E914-47DA-95CA-C5AB0DC85B11",
                     key ? key : "");
    if (n < 0 || (size_t)n >= sizeof(joined)) return false;
    unsigned char digest[CC_SHA1_DIGEST_LENGTH];
    CC_SHA1(joined, (CC_LONG)n, digest);
    char *accept = cwick_b64_encode(digest, sizeof(digest));
    char head[512];
    snprintf(head, sizeof(head),
             "HTTP/1.1 101 Switching Protocols\r\n"
             "Upgrade: websocket\r\n"
             "Connection: Upgrade\r\n"
             "Sec-WebSocket-Accept: %s\r\n\r\n", accept);
    free(accept);
    size_t len = strlen(head), off = 0;
    while (off < len) {
        ssize_t w = write(fd, head + off, len - off);
        if (w <= 0) return false;
        off += (size_t)w;
    }
    return true;
}

void cwick_ws_init(cwick_ws *w, int fd) {
    memset(w, 0, sizeof(*w));
    w->fd = fd;
}

void cwick_ws_free(cwick_ws *w) {
    free(w->in);
    free(w->msg);
    memset(w, 0, sizeof(*w));
}

static bool buf_reserve(unsigned char **p, size_t *cap, size_t need) {
    if (need <= *cap) return true;
    size_t c = *cap ? *cap : 4096;
    while (c < need) c *= 2;
    unsigned char *q = (unsigned char *)realloc(*p, c);
    if (!q) return false;
    *p = q;
    *cap = c;
    return true;
}

bool cwick_ws_send(cwick_ws *w, int opcode, const void *data, size_t len) {
    if (w->failed || w->peer_closed) return false;
    unsigned char hdr[10];
    size_t hn = 2;
    hdr[0] = (unsigned char)(0x80 | (opcode & 0x0f));   /* FIN + opcode */
    if (len < 126) {
        hdr[1] = (unsigned char)len;
    } else if (len <= 0xffff) {
        hdr[1] = 126;
        hdr[2] = (unsigned char)(len >> 8);
        hdr[3] = (unsigned char)len;
        hn = 4;
    } else {
        hdr[1] = 127;
        for (int i = 0; i < 8; i++)
            hdr[2 + i] = (unsigned char)((uint64_t)len >> (56 - 8 * i));
        hn = 10;
    }
    /* Header and payload in one write: two writes would put a 40-byte TCP
     * segment on the wire ahead of every audio delta. */
    size_t total = hn + len;
    unsigned char stackbuf[2048];
    unsigned char *buf = total <= sizeof(stackbuf)
                             ? stackbuf : (unsigned char *)malloc(total);
    if (!buf) return false;
    memcpy(buf, hdr, hn);
    if (len) memcpy(buf + hn, data, len);
    size_t off = 0;
    bool ok = true;
    while (off < total) {
        ssize_t n = write(w->fd, buf + off, total - off);
        if (n <= 0) { ok = false; break; }
        off += (size_t)n;
    }
    if (buf != stackbuf) free(buf);
    if (!ok) w->failed = true;
    return ok;
}

bool cwick_ws_send_text(cwick_ws *w, const char *s) {
    return cwick_ws_send(w, CWICK_WS_TEXT, s, strlen(s));
}

void cwick_ws_close(cwick_ws *w, int code, const char *reason) {
    if (!w->peer_closed && !w->failed) {
        unsigned char body[128];
        size_t n = 0;
        body[n++] = (unsigned char)(code >> 8);
        body[n++] = (unsigned char)code;
        if (reason) {
            size_t r = strlen(reason);
            if (r > sizeof(body) - 2) r = sizeof(body) - 2;
            memcpy(body + n, reason, r);
            n += r;
        }
        cwick_ws_send(w, CWICK_WS_CLOSE, body, n);
    }
    w->peer_closed = true;
}

/* Consume `n` parsed bytes from the front of the input buffer. */
static void in_consume(cwick_ws *w, size_t n) {
    if (n >= w->in_len) { w->in_len = 0; return; }
    memmove(w->in, w->in + n, w->in_len - n);
    w->in_len -= n;
}

/* Handle one complete frame. Returns 1 when a data message finished, 0 when
 * more frames are needed, -1 on a protocol violation. */
static int frame_take(cwick_ws *w, const unsigned char *f, size_t flen,
                      int fin, int opcode, bool masked) {
    if (!masked) return -1;              /* RFC 6455 5.1: client frames mask */
    if (opcode >= 0x8) {
        /* Control frames must be short and unfragmented. */
        if (!fin || flen > 125) return -1;
        if (opcode == CWICK_WS_CLOSE) {
            if (!w->peer_closed) cwick_ws_send(w, CWICK_WS_CLOSE, f, flen >= 2 ? 2 : 0);
            w->peer_closed = true;
            return 0;
        }
        if (opcode == CWICK_WS_PING) {
            if (!cwick_ws_send(w, CWICK_WS_PONG, f, flen)) return -1;
            return 0;
        }
        return 0;                        /* unsolicited pong: ignore */
    }
    if (opcode == 0) {
        if (w->msg_op == 0) return -1;    /* continuation with nothing to continue */
    } else {
        if (w->msg_op != 0) return -1;    /* new data frame mid-message */
        w->msg_op = opcode;
    }
    if (flen > CWICK_WS_MAX_MSG || w->msg_len > CWICK_WS_MAX_MSG - flen) return -1;
    if (!buf_reserve(&w->msg, &w->msg_cap, w->msg_len + flen + 1)) return -1;
    memcpy(w->msg + w->msg_len, f, flen);
    w->msg_len += flen;
    if (!fin) return 0;
    w->msg[w->msg_len] = '\0';
    return 1;
}

int cwick_ws_read(cwick_ws *w, unsigned char **out, size_t *out_len) {
    *out = NULL;
    *out_len = 0;
    if (w->failed) return -1;

    for (;;) {
        /* Parse whatever is already buffered before touching the socket: a
         * single read often carries several small audio appends. */
        size_t off = 0;
        while (w->in_len - off >= 2) {
            const unsigned char *h = w->in + off;
            int fin = (h[0] & 0x80) != 0;
            if (h[0] & 0x70) { w->failed = true; return -1; }   /* reserved bits */
            int opcode = h[0] & 0x0f;
            bool masked = (h[1] & 0x80) != 0;
            uint64_t len = h[1] & 0x7f;
            size_t hn = 2;
            if (len == 126) {
                if (w->in_len - off < 4) break;
                len = ((uint64_t)h[2] << 8) | h[3];
                hn = 4;
            } else if (len == 127) {
                if (w->in_len - off < 10) break;
                len = 0;
                for (int i = 0; i < 8; i++) len = (len << 8) | h[2 + i];
                hn = 10;
                if (len >> 63) { w->failed = true; return -1; }
            }
            if (len > CWICK_WS_MAX_MSG) { w->failed = true; return -1; }
            size_t need = hn + (masked ? 4 : 0) + (size_t)len;
            if (w->in_len - off < need) break;
            unsigned char *payload = w->in + off + hn;
            if (masked) {
                const unsigned char *mask = payload;
                payload += 4;
                for (size_t i = 0; i < (size_t)len; i++)
                    payload[i] ^= mask[i & 3];
            }
            int r = frame_take(w, payload, (size_t)len, fin, opcode, masked);
            off += need;
            if (r < 0) { in_consume(w, off); w->failed = true; return -1; }
            if (r == 1) {
                *out = w->msg;
                *out_len = w->msg_len;
                int op = w->msg_op;
                /* Ownership moves to the caller; the next message starts fresh. */
                w->msg = NULL;
                w->msg_len = w->msg_cap = 0;
                w->msg_op = 0;
                in_consume(w, off);
                return op;
            }
        }
        in_consume(w, off);
        if (w->peer_closed) return 0;

        if (!buf_reserve(&w->in, &w->in_cap, w->in_len + 16384)) {
            w->failed = true;
            return -1;
        }
        ssize_t n = read(w->fd, w->in + w->in_len, w->in_cap - w->in_len);
        if (n == 0) { w->peer_closed = true; return 0; }
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) return -2;
            if (errno == EINTR) continue;
            w->failed = true;
            return -1;
        }
        w->in_len += (size_t)n;
    }
}