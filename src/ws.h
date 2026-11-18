/* ws.h — the server half of RFC 6455, enough for /v1/realtime.
 *
 * Only what a realtime audio session needs: the handshake, text/binary
 * messages with continuation, and the three control frames. No extensions,
 * no permessage-deflate, no client role.
 */
#ifndef CWICK_WS_H
#define CWICK_WS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define CWICK_WS_TEXT  0x1
#define CWICK_WS_BIN   0x2
#define CWICK_WS_CLOSE 0x8
#define CWICK_WS_PING  0x9
#define CWICK_WS_PONG  0xa

/* Largest message the server will assemble. A realtime client sends audio in
 * ~100 ms appends and the server answers with ~1 s deltas, so this is generous
 * while still bounding what one peer can make us allocate. */
#define CWICK_WS_MAX_MSG (16u * 1024u * 1024u)

typedef struct {
    int fd;
    /* Bytes read but not yet parsed. Kept across calls because a read timeout
     * can land in the middle of a frame header. */
    unsigned char *in;
    size_t in_len, in_cap;
    /* The message being assembled from continuation frames. */
    unsigned char *msg;
    size_t msg_len, msg_cap;
    int msg_op;
    bool peer_closed;
    bool failed;
} cwick_ws;

/* Write the 101 response for a validated `Sec-WebSocket-Key`. Returns false
 * when the write failed. The caller has already checked the method, the
 * Upgrade/Connection headers and the version. */
bool cwick_ws_handshake(int fd, const char *key);

void cwick_ws_init(cwick_ws *w, int fd);
void cwick_ws_free(cwick_ws *w);

/* Pull one complete data message.
 *   >0  opcode (CWICK_WS_TEXT or CWICK_WS_BIN); *out is malloc'd, caller frees
 *    0  the peer closed cleanly
 *   -1  protocol error or broken socket
 *   -2  read timed out with nothing complete; the caller may check its own
 *       stop flag and call again
 * Ping is answered and close is acknowledged inside, so neither surfaces. */
int cwick_ws_read(cwick_ws *w, unsigned char **out, size_t *out_len);

/* One frame, unmasked as the server must. Returns false if the peer is gone. */
bool cwick_ws_send(cwick_ws *w, int opcode, const void *data, size_t len);
bool cwick_ws_send_text(cwick_ws *w, const char *s);
/* Close handshake: send the close frame and mark the session over. */
void cwick_ws_close(cwick_ws *w, int code, const char *reason);

#endif /* CWICK_WS_H */