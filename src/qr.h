/* qr.h — QR code PNG generation for the LAN share button.
 *
 * Native CoreImage/ImageIO, no third-party library: the only thing cwick
 * needs is a short URL encoded as a PNG for a phone camera. */
#ifndef CWICK_QR_H
#define CWICK_QR_H

#include <stddef.h>

/* Encode `text` as a QR code PNG. Returns a malloc'd buffer and its length,
 * or NULL on failure. Free with cwick_qr_png_free. */
unsigned char *cwick_qr_png(const char *text, size_t *out_len);
void cwick_qr_png_free(unsigned char *png);

#endif
