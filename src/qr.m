/* qr.m — QR code generation via macOS CoreImage + ImageIO.
 *
 * The web UI is compiled into the binary, so a share QR should be there too:
 * no JS QR library, no network call, no asset. CIQRCodeGenerator emits a
 * small CIImage; scale it with nearest-neighbour semantics (the module grid
 * stays crisp) and write PNG bytes with ImageIO. */
#import <Foundation/Foundation.h>
#import <CoreImage/CoreImage.h>
#import <ImageIO/ImageIO.h>

#include <stdlib.h>
#include <string.h>

#include "qr.h"

unsigned char *cwick_qr_png(const char *text, size_t *out_len) {
    if (out_len) *out_len = 0;
    if (!text || !*text) return NULL;
    @autoreleasepool {
        NSData *msg = [NSData dataWithBytes:text length:strlen(text)];
        CIFilter *f = [CIFilter filterWithName:@"CIQRCodeGenerator"];
        if (!f) return NULL;
        [f setValue:msg forKey:@"inputMessage"];
        [f setValue:@"M" forKey:@"inputCorrectionLevel"];
        CIImage *img = f.outputImage;
        if (!img) return NULL;

        /* 8 px per module: a 33-module v4 QR becomes 264 px, enough for a
         * phone camera without making the PNG large. */
        img = [img imageByApplyingTransform:CGAffineTransformMakeScale(8.0, 8.0)];
        CIContext *ctx = [CIContext context];
        CGImageRef cg = [ctx createCGImage:img fromRect:img.extent];
        if (!cg) return NULL;

        NSMutableData *png = [NSMutableData data];
        CGImageDestinationRef dest =
            CGImageDestinationCreateWithData((__bridge CFMutableDataRef)png,
                                             CFSTR("public.png"), 1, NULL);
        if (!dest) { CGImageRelease(cg); return NULL; }
        CGImageDestinationAddImage(dest, cg, NULL);
        bool ok = CGImageDestinationFinalize(dest);
        CFRelease(dest);
        CGImageRelease(cg);
        if (!ok || png.length == 0) return NULL;

        unsigned char *out = malloc(png.length);
        if (!out) return NULL;
        memcpy(out, png.bytes, png.length);
        if (out_len) *out_len = png.length;
        return out;
    }
}

void cwick_qr_png_free(unsigned char *png) { free(png); }
