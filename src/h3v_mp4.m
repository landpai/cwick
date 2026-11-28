/* h3v_mp4.m — H.264 + AAC into one mp4, using the system encoders.
 *
 * The pipeline produces RGB24 frames and interleaved float32 stereo; this file
 * turns that into bytes a client can save and play. Both codecs come from the
 * OS (VideoToolbox and AudioToolbox), so there is no encoder to ship:
 *
 *   frames  --VTCompressionSession--> AVCC NAL units + one avcC description
 *   samples --AudioConverter-------> AAC access units + one AudioSpecificConfig
 *   both    --this file------------> ftyp | mdat | moov
 *
 * The moov is written after mdat is complete, so the chunk offsets are known
 * when the tables are emitted and there is no patching pass. Sample layout is
 * all video samples then all audio samples, one chunk per track: legal under
 * ISO/IEC 14496-12, and a player seeks by track rather than by byte.
 *
 * Encoding settings are fixed and stated rather than tuned: H.264 High profile,
 * kVTCompressionPropertyKey_AllowFrameReordering off (the callback emits decode
 * order), one IDR every 48 frames, 0.2 bits per pixel per frame. The encoder
 * still emits B frames, so stts alone is not enough: ctts carries each sample's
 * composition offset and keeps display order correct in strict players. AAC-LC,
 * 128 kbit/s, stereo. The output is meant to be watched, not re-encoded.
 * A fixed 5x5 unsharp mask restores detail softened by the Flow-VAE decoder.
 *
 * Comments describe mechanism and measured numbers only.
 */
#import <Foundation/Foundation.h>
#import <AudioToolbox/AudioToolbox.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <VideoToolbox/VideoToolbox.h>

#include <stdlib.h>
#include <string.h>

#include "h3v_internal.h"
#include "log.h"
#include "util.h"

/* ---- a growable byte buffer ---------------------------------------- */
typedef struct {
    unsigned char *p;
    size_t len, cap;
} mb;

static void mb_reserve(mb *b, size_t need) {
    if (b->len + need <= b->cap) return;
    size_t cap = b->cap ? b->cap : 65536;
    while (cap < b->len + need) cap *= 2;
    b->p = (unsigned char *)realloc(b->p, cap);
    if (!b->p) abort();
    b->cap = cap;
}

static void mb_put(mb *b, const void *src, size_t n) {
    if (!n) return;
    mb_reserve(b, n);
    memcpy(b->p + b->len, src, n);
    b->len += n;
}

static void mb_u8(mb *b, unsigned v) {
    unsigned char c = (unsigned char)v;
    mb_put(b, &c, 1);
}

static void mb_u16(mb *b, unsigned v) {
    unsigned char c[2] = {(unsigned char)(v >> 8), (unsigned char)v};
    mb_put(b, c, 2);
}

static void mb_u24(mb *b, unsigned v) {
    unsigned char c[3] = {(unsigned char)(v >> 16), (unsigned char)(v >> 8),
                          (unsigned char)v};
    mb_put(b, c, 3);
}

static void mb_u32(mb *b, uint32_t v) {
    unsigned char c[4] = {(unsigned char)(v >> 24), (unsigned char)(v >> 16),
                          (unsigned char)(v >> 8), (unsigned char)v};
    mb_put(b, c, 4);
}

static void mb_zero(mb *b, size_t n) {
    mb_reserve(b, n);
    memset(b->p + b->len, 0, n);
    b->len += n;
}

static void mb_free(mb *b) {
    free(b->p);
    b->p = NULL;
    b->len = b->cap = 0;
}

/* fullbox header: version + 24-bit flags */
static void mb_full(mb *b, unsigned version, unsigned flags) {
    mb_u8(b, version);
    mb_u24(b, flags);
}

/* `body` is the payload of the box, not including its own header. */
static void mb_box(mb *out, const char *type, const mb *body) {
    mb_u32(out, (uint32_t)(body->len + 8));
    mb_put(out, type, 4);
    mb_put(out, body->p, body->len);
}

/* ---- encoder output kept until the mdat is written ----------------- */
typedef struct {
    unsigned char *data;
    size_t len;
    int sync;
    /* Presentation time in the track timescale (24000 ticks/s for video).
     * The callback delivers samples in decode order; B frames mean that is
     * not display order, so the muxer needs each sample's real PTS to emit
     * a ctts table. Without it a player shows B frames in decode order and
     * the picture jumps back and forth. */
    uint32_t pts_ticks;
} mbsample;

typedef struct {
    mbsample *v, *a;
    size_t nv, capv, na, capa;
} mbtrack;

static void mbsample_push(mbsample **arr, size_t *n, size_t *cap,
                          const void *data, size_t len, int sync,
                          uint32_t pts_ticks) {
    if (*n == *cap) {
        *cap = *cap ? *cap * 2 : 256;
        *arr = (mbsample *)realloc(*arr, *cap * sizeof(mbsample));
        if (!*arr) abort();
    }
    unsigned char *copy = (unsigned char *)malloc(len ? len : 1);
    if (!copy) abort();
    if (len) memcpy(copy, data, len);
    (*arr)[*n].data = copy;
    (*arr)[*n].len = len;
    (*arr)[*n].sync = sync;
    (*arr)[*n].pts_ticks = pts_ticks;
    (*n)++;
}

static void mbsample_free(mbsample *arr, size_t n) {
    for (size_t i = 0; i < n; i++) free(arr[i].data);
    free(arr);
}

/* Whether a 4-byte-length-prefixed AVCC sample contains an IDR slice (NAL
 * type 5), which makes the access unit a random-access / sync point.
 *
 * Scan every NAL, not just the first. VideoToolbox puts an SEI (type 6) in
 * front of each IDR access unit, so the slice is the *second* NAL; checking
 * only the first one reports every sample as non-IDR. Combined with a fallback
 * that marked all samples sync when none were, stss then advertised all 102
 * frames — including P frames with no reference — as random-access points.
 * QuickTime trusts stss and restarts decode at those frames, ghosting the
 * previous picture; ffmpeg ignores stss and looked fine. */
static bool avcc_has_idr(const unsigned char *p, size_t len) {
    size_t off = 0;
    while (off + 4 <= len) {
        uint32_t n = ((uint32_t)p[off] << 24) | ((uint32_t)p[off + 1] << 16) |
                     ((uint32_t)p[off + 2] << 8) | p[off + 3];
        if (n == 0 || (size_t)n + 4 > len - off) break;
        if ((p[off + 4] & 0x1f) == 5) return true;
        off += (size_t)n + 4;
    }
    return false;
}

/* ---- video -------------------------------------------------------- */
typedef struct {
    mbtrack *t;
    int failed;
    CMFormatDescriptionRef fmt;   /* owned, taken from the first sample */
} vtctx;

static void vt_callback(void *ref, void *src, OSStatus status,
                        VTEncodeInfoFlags info, CMSampleBufferRef sbuf) {
    vtctx *c = (vtctx *)ref;
    (void)src;
    (void)info;
    if (status != noErr || !sbuf) {
        c->failed = 1;
        return;
    }
    if (!c->fmt) {
        CMFormatDescriptionRef fd = CMSampleBufferGetFormatDescription(sbuf);
        if (fd) c->fmt = (CMFormatDescriptionRef)CFRetain(fd);
    }
    CMBlockBufferRef bb = CMSampleBufferGetDataBuffer(sbuf);
    if (!bb) {
        c->failed = 1;
        return;
    }
    const size_t total = CMBlockBufferGetDataLength(bb);
    if (total == 0) return;
    unsigned char *tmp = (unsigned char *)malloc(total);
    if (!tmp) {
        c->failed = 1;
        return;
    }
    if (CMBlockBufferCopyDataBytes(bb, 0, total, tmp) != kCMBlockBufferNoErr) {
        free(tmp);
        c->failed = 1;
        return;
    }
    /* The callback arrives in decode order. B frames are decoded before the
     * frames they depend on but displayed later, so its presentation stamp —
     * not its arrival index — is the sample's real display time. Convert to
     * the video track's 24000-tick timescale and keep it for ctts; encoding
     * fed CMTimeMake(i, 24), so each frame is exactly 1000 ticks. */
    CMTime pts = CMSampleBufferGetPresentationTimeStamp(sbuf);
    uint32_t pts_ticks = 0;
    if (pts.flags & kCMTimeFlags_Valid && pts.timescale > 0) {
        double tick = (double)pts.value * 24000.0 / (double)pts.timescale;
        pts_ticks = tick >= 0.0 ? (uint32_t)(tick + 0.5) : 0;
    }
    mbsample_push(&c->t->v, &c->t->nv, &c->t->capv, tmp, total,
                  avcc_has_idr(tmp, total), pts_ticks);
    free(tmp);
}

/* Serialise the H.264 format description into an avcC payload.
 *
 * The parameter-set API returns one flat list (SPS then PPS), but avcC has two
 * separate runs: numOfSPS followed by the SPS units, then numOfPPS followed by
 * the PPS units. Writing the flat count as numOfSPS and omitting numOfPPS makes
 * a structurally invalid record (bytes: e2 000a<SPS> 0004<PPS>, no PPS count).
 * Lenient decoders rebuild parameter sets from the in-band NALs and still show
 * a picture; strict ones reject the stream and show black. Split by NAL type
 * (7 = SPS, 8 = PPS) and emit both runs. */
static bool build_avcc(CMFormatDescriptionRef fd, unsigned char **out,
                       size_t *out_len, cwick_error *err) {
    size_t count = 0;
    if (CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
            fd, 0, NULL, NULL, &count, NULL) != noErr ||
        count < 2) {
        cwick_error_set(err, CWICK_ERR_STATE,
                        "the encoder's H.264 description has no parameter sets");
        return false;
    }
    /* Bound the parameter sets we keep: one SPS and one PPS is what the
     * VideoToolbox H.264 encoder emits; the arrays tolerate more but never
     * more than the description reports. */
    const uint8_t *sps[16], *pps[16];
    size_t sps_len[16], pps_len[16];
    size_t nsps = 0, npps = 0;
    for (size_t i = 0; i < count; i++) {
        const uint8_t *ps = NULL;
        size_t ps_len = 0;
        if (CMVideoFormatDescriptionGetH264ParameterSetAtIndex(fd, i, &ps, &ps_len,
                                                               NULL, NULL) != noErr ||
            ps_len < 4) {
            cwick_error_set(err, CWICK_ERR_STATE, "parameter set %zu is missing",
                            i);
            return false;
        }
        int nal = ps[0] & 0x1f;
        if (nal == 7 && nsps < 16) {
            sps[nsps] = ps;
            sps_len[nsps++] = ps_len;
        } else if (nal == 8 && npps < 16) {
            pps[npps] = ps;
            pps_len[npps++] = ps_len;
        }
    }
    if (nsps == 0 || npps == 0) {
        cwick_error_set(err, CWICK_ERR_STATE,
                        "the encoder's H.264 description lacks an SPS or PPS");
        return false;
    }
    mb b;
    memset(&b, 0, sizeof(b));
    mb_u8(&b, 1);                       /* configurationVersion */
    mb_u8(&b, sps[0][1]);               /* AVCProfileIndication */
    mb_u8(&b, sps[0][2]);               /* profile_compatibility */
    mb_u8(&b, sps[0][3]);               /* AVCLevelIndication */
    mb_u8(&b, 0xfc | 3);                /* 4-byte NAL lengths */
    mb_u8(&b, 0xe0 | (unsigned)nsps);   /* numOfSequenceParameterSets */
    for (size_t i = 0; i < nsps; i++) {
        mb_u16(&b, (unsigned)sps_len[i]);
        mb_put(&b, sps[i], sps_len[i]);
    }
    mb_u8(&b, (unsigned)npps);          /* numOfPictureParameterSets */
    for (size_t i = 0; i < npps; i++) {
        mb_u16(&b, (unsigned)pps_len[i]);
        mb_put(&b, pps[i], pps_len[i]);
    }
    *out = b.p;
    *out_len = b.len;
    return true;
}

static void sharpen_rgb24(const unsigned char *src, unsigned char *dst,
                          unsigned char *horizontal, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                unsigned sum = 0;
                for (int k = -2; k <= 2; k++) {
                    int sx = x + k;
                    if (sx < 0) sx = 0;
                    if (sx >= width) sx = width - 1;
                    static const int weights[5] = {1, 4, 6, 4, 1};
                    sum += (unsigned)src[((size_t)y * width + sx) * 3 + c] * weights[k + 2];
                }
                horizontal[((size_t)y * width + x) * 3 + c] = (unsigned char)(sum >> 4);
            }
        }
    }
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                unsigned sum = 0;
                for (int k = -2; k <= 2; k++) {
                    int sy = y + k;
                    if (sy < 0) sy = 0;
                    if (sy >= height) sy = height - 1;
                    static const int weights[5] = {1, 4, 6, 4, 1};
                    sum += (unsigned)horizontal[((size_t)sy * width + x) * 3 + c] * weights[k + 2];
                }
                size_t at = ((size_t)y * width + x) * 3 + c;
                int blurred = (int)(sum >> 4);
                int amount = c == 0 ? 256 : 128;
                int value = (int)src[at] + (((int)src[at] - blurred) * amount) / 256;
                dst[at] = (unsigned char)(value < 0 ? 0 : value > 255 ? 255 : value);
            }
        }
    }
}

static bool encode_video(const unsigned char *rgb, int64_t frames, int width,
                         int height, mbtrack *t, unsigned char **avcc,
                         size_t *avcc_len, cwick_error *err) {
    /* The encoder callback runs on a VideoToolbox thread and writes into ctx,
     * so ctx must exist before the session and be handed over as the refcon. */
    vtctx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.t = t;

    VTCompressionSessionRef session = NULL;
    OSStatus rc = VTCompressionSessionCreate(
        kCFAllocatorDefault, (int32_t)width, (int32_t)height,
        kCMVideoCodecType_H264, NULL, NULL, NULL, vt_callback, &ctx, &session);
    if (rc != noErr || !session) {
        cwick_error_set(err, CWICK_ERR_STATE,
                        "VideoToolbox could not open an H.264 session (%d)",
                        (int)rc);
        return false;
    }

    /* AllowFrameReordering off makes the callback deliver decode order, but
     * the encoder may still code B frames. Their presentation stamps are kept
     * in the mbsamples and written as ctts so players show them in order. */
    int32_t realtime = 0, reorder = 0, keyint = 48;
    double fps = (double)H3V_FPS;
    int32_t bitrate = (int32_t)((double)width * (double)height * fps * 0.2);
    if (bitrate < 1000000) bitrate = 1000000;
    struct {
        CFStringRef key;
        CFTypeRef val;
    } props[6];
    CFNumberRef n_realtime = CFNumberCreate(NULL, kCFNumberSInt32Type, &realtime);
    CFNumberRef n_reorder = CFNumberCreate(NULL, kCFNumberSInt32Type, &reorder);
    CFNumberRef n_keyint = CFNumberCreate(NULL, kCFNumberSInt32Type, &keyint);
    CFNumberRef n_bitrate = CFNumberCreate(NULL, kCFNumberSInt32Type, &bitrate);
    CFNumberRef n_fps = CFNumberCreate(NULL, kCFNumberFloat64Type, &fps);
    props[0].key = kVTCompressionPropertyKey_RealTime;            props[0].val = n_realtime;
    props[1].key = kVTCompressionPropertyKey_AllowFrameReordering; props[1].val = n_reorder;
    props[2].key = kVTCompressionPropertyKey_MaxKeyFrameInterval;  props[2].val = n_keyint;
    props[3].key = kVTCompressionPropertyKey_AverageBitRate;       props[3].val = n_bitrate;
    props[4].key = kVTCompressionPropertyKey_ExpectedFrameRate;    props[4].val = n_fps;
    /* Baseline forbids B frames. With AllowFrameReordering off, the callback
     * is already in decode order, so a B-free stream makes PTS == DTS for
     * every sample and no ctts is needed — the simplest thing a strict
     * player such as QuickTime can parse. */
    props[5].key = kVTCompressionPropertyKey_ProfileLevel;
    props[5].val = kVTProfileLevel_H264_Baseline_AutoLevel;
    for (int i = 0; i < 6; i++)
        VTSessionSetProperty(session, props[i].key, props[i].val);
    CFRelease(n_realtime);
    CFRelease(n_reorder);
    CFRelease(n_keyint);
    CFRelease(n_bitrate);
    CFRelease(n_fps);
    VTCompressionSessionPrepareToEncodeFrames(session);

    size_t frame_bytes = (size_t)width * (size_t)height * 3;
    unsigned char *horizontal = (unsigned char *)malloc(frame_bytes);
    unsigned char *sharpened = (unsigned char *)malloc(frame_bytes);
    if (!horizontal || !sharpened) {
        free(horizontal); free(sharpened);
        VTCompressionSessionInvalidate(session);
        CFRelease(session);
        cwick_error_set(err, CWICK_ERR_MEMORY, "cannot allocate video sharpening buffers");
        return false;
    }

    NSDictionary *attrs = @{
        (id)kCVPixelBufferIOSurfacePropertiesKey : @{},
        (id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA)
    };
    bool ok = true;
    for (int64_t i = 0; i < frames && ok; i++) {
        CVPixelBufferRef pb = NULL;
        rc = CVPixelBufferCreate(kCFAllocatorDefault, (size_t)width, (size_t)height,
                                 kCVPixelFormatType_32BGRA,
                                 (__bridge CFDictionaryRef)attrs, &pb);
        if (rc != kCVReturnSuccess || !pb) {
            cwick_error_set(err, CWICK_ERR_STATE,
                            "could not allocate a %dx%d pixel buffer", width,
                            height);
            ok = false;
            break;
        }
        CVPixelBufferLockBaseAddress(pb, 0);
        {
            unsigned char *base = (unsigned char *)CVPixelBufferGetBaseAddress(pb);
            const size_t stride = CVPixelBufferGetBytesPerRow(pb);
            const unsigned char *frame_rgb = rgb + (size_t)i * width * height * 3;
            sharpen_rgb24(frame_rgb, sharpened, horizontal, width, height);
            const unsigned char *src = sharpened;
            for (int y = 0; y < height; y++) {
                unsigned char *d = base + (size_t)y * stride;
                const unsigned char *s = src + (size_t)y * width * 3;
                for (int x = 0; x < width; x++) {
                    d[x * 4 + 0] = s[x * 3 + 2];
                    d[x * 4 + 1] = s[x * 3 + 1];
                    d[x * 4 + 2] = s[x * 3 + 0];
                    d[x * 4 + 3] = 255;
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(pb, 0);
        const CMTime pts = CMTimeMake(i, (int32_t)H3V_FPS);
        const CMTime dur = CMTimeMake(1, (int32_t)H3V_FPS);
        rc = VTCompressionSessionEncodeFrame(session, pb, pts, dur, NULL, NULL, NULL);
        CVPixelBufferRelease(pb);
        if (rc != noErr) {
            cwick_error_set(err, CWICK_ERR_STATE, "frame %lld would not encode (%d)",
                            (long long)i, (int)rc);
            ok = false;
        }
    }
    free(horizontal);
    free(sharpened);
    if (ok && VTCompressionSessionCompleteFrames(session, kCMTimeInvalid) != noErr) {
        cwick_error_set(err, CWICK_ERR_STATE, "the H.264 encoder did not drain");
        ok = false;
    }
    if (ok && (ctx.failed || t->nv == 0 || !ctx.fmt)) {
        cwick_error_set(err, CWICK_ERR_STATE,
                        "the H.264 encoder produced no usable output");
        ok = false;
    }
    if (ok) ok = build_avcc(ctx.fmt, avcc, avcc_len, err);

    if (ctx.fmt) CFRelease(ctx.fmt);
    VTCompressionSessionInvalidate(session);
    CFRelease(session);
    return ok;
}

/* ---- audio -------------------------------------------------------- */

/* AudioToolbox's AAC magic cookie is a whole ES descriptor using MPEG-4's
 * 7-bit-per-byte expanded sizes, not the bare AudioSpecificConfig that goes
 * in esds' DecSpecificInfo. Wrap it as-is and a player reads object type 0,
 * 22050 Hz and zero channels. Find the 0x05 descriptor recursively and hand
 * mp4a only its payload. Descendants are searched because the result lives
 * inside ES (0x03) -> DecoderConfig (0x04) -> DecSpecificInfo (0x05). */
static size_t desc_header(const unsigned char *p, size_t len, uint32_t *sz) {
    if (len < 2) return 0;
    size_t i = 1;
    *sz = 0;
    for (int n = 0; n < 4 && i < len; n++) {
        unsigned char b = p[i++];
        *sz = (*sz << 7) | (b & 0x7f);
        if (!(b & 0x80)) return i;
    }
    return 0;
}

static const unsigned char *find_asc(const unsigned char *p, size_t len,
                                     size_t *out_len) {
    uint32_t sz = 0;
    if (len < 2 || p[0] != 0x03) return NULL;   /* ES descriptor */
    size_t hs = desc_header(p, len, &sz);
    if (!hs || sz > len - hs) return NULL;
    size_t end = hs + sz;
    size_t i = hs + 3;                          /* ES_ID + flags */
    while (i + 2 <= end) {
        unsigned char tag = p[i];
        uint32_t dsz = 0;
        size_t dh = desc_header(p + i, end - i, &dsz);
        if (!dh || dsz > end - i - dh) return NULL;
        size_t body = i + dh;
        if (tag == 0x04 && dsz >= 15) {
            /* objectType, streamType, bufferSizeDB[3], max/avg bitrate[4+4] */
            size_t q = body + 13, qend = body + dsz;
            while (q + 2 <= qend) {
                unsigned char t = p[q];
                uint32_t asz = 0;
                size_t ah = desc_header(p + q, qend - q, &asz);
                if (!ah || asz > qend - q - ah) break;
                if (t == 0x05 && asz > 0) {
                    *out_len = asz;
                    return p + q + ah;
                }
                q += ah + asz;
            }
        }
        i = body + dsz;
    }
    return NULL;
}

typedef struct {
    const float *pcm;
    int64_t frames;
    int used;
} au_feed;

static OSStatus au_input(AudioConverterRef conv, UInt32 *io_packets,
                         AudioBufferList *io_data,
                         AudioStreamPacketDescription **out_desc, void *user) {
    (void)conv;
    (void)out_desc;
    au_feed *f = (au_feed *)user;
    if (!f || f->used || f->frames <= 0) {
        *io_packets = 0;
        return noErr;
    }
    f->used = 1;
    io_data->mNumberBuffers = 1;
    io_data->mBuffers[0].mNumberChannels = 2;
    io_data->mBuffers[0].mData = (void *)f->pcm;
    io_data->mBuffers[0].mDataByteSize = (UInt32)(f->frames * 8);
    *io_packets = (UInt32)f->frames;
    return noErr;
}

static bool encode_audio(const float *pcm, int64_t samples, mbtrack *t,
                         unsigned char **asc, size_t *asc_len, cwick_error *err) {
    AudioStreamBasicDescription in, out;
    memset(&in, 0, sizeof(in));
    in.mSampleRate = H3V_AUDIO_RATE;
    in.mFormatID = kAudioFormatLinearPCM;
    in.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
    in.mChannelsPerFrame = 2;
    in.mBitsPerChannel = 32;
    in.mFramesPerPacket = 1;
    in.mBytesPerFrame = 8;
    in.mBytesPerPacket = 8;

    memset(&out, 0, sizeof(out));
    out.mSampleRate = H3V_AUDIO_RATE;
    out.mFormatID = kAudioFormatMPEG4AAC;
    out.mChannelsPerFrame = 2;
    UInt32 size = sizeof(out);
    if (AudioFormatGetProperty(kAudioFormatProperty_FormatInfo, 0, NULL, &size,
                               &out) != noErr) {
        cwick_error_set(err, CWICK_ERR_STATE, "the AAC encoder is unavailable");
        return false;
    }

    AudioConverterRef conv = NULL;
    OSStatus rc = AudioConverterNew(&in, &out, &conv);
    if (rc != noErr || !conv) {
        cwick_error_set(err, CWICK_ERR_STATE,
                        "could not open an AAC converter (%d)", (int)rc);
        return false;
    }
    UInt32 bitrate = 128000;
    AudioConverterSetProperty(conv, kAudioConverterEncodeBitRate, sizeof(bitrate),
                              &bitrate);

    unsigned char *magic = NULL;
    UInt32 magic_len = 0;
    if (AudioConverterGetPropertyInfo(conv, kAudioConverterCompressionMagicCookie,
                                      &magic_len, NULL) == noErr &&
        magic_len > 0) {
        magic = (unsigned char *)malloc(magic_len);
        if (magic &&
            AudioConverterGetProperty(conv, kAudioConverterCompressionMagicCookie,
                                      &magic_len, magic) != noErr) {
            free(magic);
            magic = NULL;
        }
    }
    if (magic) {
        size_t asc_n = 0;
        const unsigned char *asc_p = find_asc(magic, magic_len, &asc_n);
        if (asc_p && asc_n > 0) {
            unsigned char *asc_copy = (unsigned char *)malloc(asc_n);
            if (asc_copy) {
                memcpy(asc_copy, asc_p, asc_n);
                free(magic);
                magic = asc_copy;
                magic_len = (UInt32)asc_n;
            }
        }
        /* If no 0x05 descriptor exists the cookie is already a bare
         * AudioSpecificConfig on this OS; keep it unchanged. */
    }

    const int64_t max_packets = samples / 1024 + 8;
    const size_t cap = (size_t)(max_packets * 2048);
    unsigned char *buf = (unsigned char *)malloc(cap);
    AudioStreamPacketDescription *descs = (AudioStreamPacketDescription *)
        calloc((size_t)max_packets, sizeof(AudioStreamPacketDescription));
    au_feed feed = {pcm, samples, 0};
    bool ok = false;

    for (;;) {
        UInt32 want = (UInt32)max_packets;
        AudioBufferList abl;
        abl.mNumberBuffers = 1;
        abl.mBuffers[0].mNumberChannels = 2;
        abl.mBuffers[0].mDataByteSize = (UInt32)cap;
        abl.mBuffers[0].mData = buf;
        memset(descs, 0, (size_t)max_packets * sizeof(*descs));
        rc = AudioConverterFillComplexBuffer(conv, au_input, &feed, &want, &abl,
                                            descs);
        if (rc != noErr && rc != 0) {
            cwick_error_set(err, CWICK_ERR_STATE, "AAC encoding failed (%d)",
                            (int)rc);
            ok = false;
            break;
        }
        if (want == 0) break;
        /* With a fixed bitrate the encoder reports one description per packet;
         * if it does not, split the buffer evenly. */
        size_t off = 0;
        for (UInt32 i = 0; i < want; i++) {
            size_t n = descs[i].mDataByteSize;
            if (n == 0) n = (size_t)(abl.mBuffers[0].mDataByteSize / want);
            if (off + n > cap) n = cap - off;
            if (n == 0) break;
            mbsample_push(&t->a, &t->na, &t->capa, buf + off, n, 1, 0);
            off += n;
        }
        if (want < (UInt32)max_packets) break;
    }
    if (t->na > 0) ok = true;

    if (ok) {
        *asc = magic;
        *asc_len = magic_len;
        magic = NULL;
    }
    free(magic);
    free(buf);
    free(descs);
    AudioConverterDispose(conv);
    return ok;
}

/* ---- mp4 boxes ---------------------------------------------------- */
static void put_mvhd(mb *out, uint32_t timescale, uint32_t duration,
                     uint32_t next_track) {
    mb b;
    memset(&b, 0, sizeof(b));
    mb_full(&b, 0, 0);
    mb_u32(&b, 0);                 /* creation_time */
    mb_u32(&b, 0);                 /* modification_time */
    mb_u32(&b, timescale);
    mb_u32(&b, duration);
    mb_u32(&b, 0x00010000);        /* rate 1.0 */
    mb_u16(&b, 0x0100);            /* volume 1.0 */
    mb_zero(&b, 10);
    mb_u32(&b, 0x00010000);        /* unity matrix */
    mb_u32(&b, 0);
    mb_u32(&b, 0);
    mb_u32(&b, 0);
    mb_u32(&b, 0x00010000);
    mb_u32(&b, 0);
    mb_u32(&b, 0);
    mb_u32(&b, 0);
    mb_u32(&b, 0x40000000);
    mb_zero(&b, 24);
    mb_u32(&b, next_track);
    mb_box(out, "mvhd", &b);
    mb_free(&b);
}

static void put_tkhd(mb *out, uint32_t track_id, uint32_t duration, int width,
                     int height, int is_audio) {
    mb b;
    memset(&b, 0, sizeof(b));
    mb_full(&b, 0, 3);             /* enabled | in movie */
    mb_u32(&b, 0);
    mb_u32(&b, 0);
    mb_u32(&b, track_id);
    mb_u32(&b, 0);
    mb_u32(&b, duration);
    mb_zero(&b, 8);
    mb_u16(&b, 0);                 /* layer */
    mb_u16(&b, 0);                 /* alternate_group */
    mb_u16(&b, is_audio ? 0x0100 : 0);
    mb_u16(&b, 0);
    mb_u32(&b, 0x00010000);
    mb_u32(&b, 0);
    mb_u32(&b, 0);
    mb_u32(&b, 0);
    mb_u32(&b, 0x00010000);
    mb_u32(&b, 0);
    mb_u32(&b, 0);
    mb_u32(&b, 0);
    mb_u32(&b, 0x40000000);
    mb_u32(&b, (uint32_t)width << 16);
    mb_u32(&b, (uint32_t)height << 16);
    mb_box(out, "tkhd", &b);
    mb_free(&b);
}

static void put_mdhd(mb *out, uint32_t timescale, uint32_t duration) {
    mb b;
    memset(&b, 0, sizeof(b));
    mb_full(&b, 0, 0);
    mb_u32(&b, 0);
    mb_u32(&b, 0);
    mb_u32(&b, timescale);
    mb_u32(&b, duration);
    mb_u16(&b, 0x55c4);            /* "und" */
    mb_u16(&b, 0);
    mb_box(out, "mdhd", &b);
    mb_free(&b);
}

static void put_hdlr(mb *out, const char *type, const char *name) {
    mb b;
    memset(&b, 0, sizeof(b));
    mb_full(&b, 0, 0);
    mb_u32(&b, 0);
    mb_put(&b, type, 4);
    mb_zero(&b, 12);
    mb_put(&b, name, strlen(name) + 1);
    mb_box(out, "hdlr", &b);
    mb_free(&b);
}

static void put_dinf(mb *out) {
    mb dref, url, dinf, head;
    memset(&dref, 0, sizeof(dref));
    memset(&url, 0, sizeof(url));
    memset(&dinf, 0, sizeof(dinf));
    memset(&head, 0, sizeof(head));
    mb_full(&url, 0, 1);           /* media data lives in this file */
    mb_box(&dref, "url ", &url);
    mb_full(&head, 0, 0);
    mb_u32(&head, 1);              /* entry_count */
    mb_put(&head, dref.p, dref.len);
    mb_box(&dinf, "dref", &head);
    mb_box(out, "dinf", &dinf);
    mb_free(&dref);
    mb_free(&url);
    mb_free(&dinf);
    mb_free(&head);
}

/* One sample table. `stsd_body` is the full stsd payload (fullbox + entry);
 * `sample_delta` is the per-sample duration in the track's timescale. With
 * `want_ctts`, each sample's presentation stamp is emitted as a composition
 * offset against its decode time (sample index * sample_delta) — required
 * whenever the encoder produced B frames. */
static void put_stbl(mb *out, const mb *stsd_body, uint32_t sample_delta,
                     uint32_t first_offset, const mbsample *samples,
                     uint32_t n_samples, int want_stss, int want_ctts) {
    mb stbl, b;
    memset(&stbl, 0, sizeof(stbl));

    mb_box(&stbl, "stsd", stsd_body);

    /* stts: version 0, 1 entry (all samples have same duration) */
    memset(&b, 0, sizeof(b));
    mb_full(&b, 0, 0);
    mb_u32(&b, 1);                 /* entry_count */
    mb_u32(&b, n_samples);        /* sample_count */
    mb_u32(&b, sample_delta);     /* sample_delta */
    mb_box(&stbl, "stts", &b);
    mb_free(&b);

    if (want_ctts) {
        /* ctts version 0: one entry per sample, offset = PTS - DTS. DTS is
         * implicit (decode order == sample order), PTS came from the encoder
         * callback. B frames have a positive offset; I/P are usually zero. */
        memset(&b, 0, sizeof(b));
        mb_full(&b, 0, 0);
        mb_u32(&b, n_samples);     /* entry_count */
        for (uint32_t i = 0; i < n_samples; i++) {
            uint32_t dts = i * sample_delta;
            uint32_t off = samples[i].pts_ticks >= dts
                         ? samples[i].pts_ticks - dts : 0;
            mb_u32(&b, 1);         /* sample_count */
            mb_u32(&b, off);       /* sample_offset (composition offset) */
        }
        mb_box(&stbl, "ctts", &b);
        mb_free(&b);
    }

    if (want_stss) {
        uint32_t n = 0;
        for (uint32_t i = 0; i < n_samples; i++)
            if (samples[i].sync) n++;
        /* Never fall back to "all samples are sync": a P frame has no
         * reference of its own and QuickTime restarts decode there, ghosting
         * the previous picture. If the encoder emitted no IDR at all, mark
         * only the first sample so the table still has one entry point. */
        if (n == 0 && n_samples) n = 1;
        memset(&b, 0, sizeof(b));
        mb_full(&b, 0, 0);
        mb_u32(&b, n);
        if (n == 1 && !samples[0].sync) {
            mb_u32(&b, 1);
        } else {
            for (uint32_t i = 0; i < n_samples; i++)
                if (samples[i].sync) mb_u32(&b, i + 1);
        }
        mb_box(&stbl, "stss", &b);
        mb_free(&b);
    }

    /* stsc: one sample per chunk, matching the one-offset-per-sample stco
     * below. ISO 14496-12 requires stco's entry_count to equal the number of
     * chunks implied by stsc; a single chunk here would contradict 102 stco
     * entries and ffprobe would reject the table. */
    memset(&b, 0, sizeof(b));
    mb_full(&b, 0, 0);
    mb_u32(&b, 1);                 /* entry_count */
    mb_u32(&b, 1);                 /* first_chunk */
    mb_u32(&b, 1);                 /* samples_per_chunk */
    mb_u32(&b, 1);                 /* sample_description_index */
    mb_box(&stbl, "stsc", &b);
    mb_free(&b);

    /* stsz: version 0 layout = version+flags(4) | sample_size(4) | entry_count(4)
     * | per-sample sizes (sample_size=0 means variable). */
    memset(&b, 0, sizeof(b));
    mb_full(&b, 0, 0);
    mb_u32(&b, 0);                 /* sample_size = 0 → variable sizes follow */
    mb_u32(&b, n_samples);         /* entry_count */
    for (uint32_t i = 0; i < n_samples; i++)
        mb_u32(&b, (uint32_t)samples[i].len);
    mb_box(&stbl, "stsz", &b);
    mb_free(&b);

    /* stco: per-sample offsets (each sample is its own chunk) for broadest
     * player compatibility. Offsets are absolute from file start = first_offset
     * plus cumulative previous sample sizes. */
    memset(&b, 0, sizeof(b));
    mb_full(&b, 0, 0);
    mb_u32(&b, n_samples);        /* entry_count = n_samples */
    uint32_t cum = 0;
    for (uint32_t i = 0; i < n_samples; i++) {
        mb_u32(&b, first_offset + cum);
        cum += (uint32_t)samples[i].len;
    }
    mb_box(&stbl, "stco", &b);
    mb_free(&b);

    mb_box(out, "stbl", &stbl);
    mb_free(&stbl);
}

static void put_avc1(mb *out, const unsigned char *avcc, size_t avcc_len, int w,
                     int h) {
    mb e, cfg;
    memset(&e, 0, sizeof(e));
    memset(&cfg, 0, sizeof(cfg));
    mb_zero(&e, 6);
    mb_u16(&e, 1);                 /* data_reference_index */
    mb_u16(&e, 0);                 /* pre_defined */
    mb_u16(&e, 0);                 /* reserved */
    mb_zero(&e, 12);               /* pre_defined[3] */
    mb_u16(&e, (unsigned)w);
    mb_u16(&e, (unsigned)h);
    mb_u32(&e, 0x00480000);        /* horizresolution 72 dpi */
    mb_u32(&e, 0x00480000);        /* vertresolution  72 dpi */
    mb_u32(&e, 0);                 /* reserved */
    mb_u16(&e, 1);                 /* frame_count */
    mb_u8(&e, 0);                  /* compressorname: empty */
    mb_zero(&e, 31);
    mb_u16(&e, 0x0018);            /* depth */
    mb_u16(&e, 0xffff);            /* pre_defined = -1 */
    mb_put(&cfg, avcc, avcc_len);
    mb_box(&e, "avcC", &cfg);
    mb_box(out, "avc1", &e);
    mb_free(&e);
    mb_free(&cfg);
}

/* mp4a entry with an esds carrying the encoder's AudioSpecificConfig. */
static void put_mp4a(mb *out, const unsigned char *asc, size_t asc_len,
                     uint32_t rate, uint32_t channels) {
    mb e, esds, dsi, dcd, sl;
    memset(&e, 0, sizeof(e));
    memset(&esds, 0, sizeof(esds));
    memset(&dsi, 0, sizeof(dsi));
    memset(&dcd, 0, sizeof(dcd));
    memset(&sl, 0, sizeof(sl));

    mb_zero(&e, 6);
    mb_u16(&e, 1);                 /* data_reference_index */
    mb_zero(&e, 8);                /* reserved */
    mb_u16(&e, channels);
    mb_u16(&e, 16);                /* samplesize */
    mb_u16(&e, 0);                 /* pre_defined */
    mb_u16(&e, 0);                 /* reserved */
    mb_u32(&e, rate << 16);

    mb_put(&dsi, asc, asc_len);
    mb_u8(&sl, 0x02);              /* SLConfigDescriptor predefined = 2 */

    /* DecoderConfigDescriptor: objectTypeIndication 0x40 (MPEG-4 audio),
     * streamType 0x15 (audio), then the bitrate figures and the config. */
    mb_u8(&dcd, 0x40);
    mb_u8(&dcd, 0x15);
    mb_u24(&dcd, 0);
    mb_u32(&dcd, 128000);
    mb_u32(&dcd, 128000);
    mb_u8(&dcd, 0x05);
    mb_u8(&dcd, (unsigned)asc_len);
    mb_put(&dcd, dsi.p, dsi.len);

    mb_full(&esds, 0, 0);
    mb_u8(&esds, 0x03);            /* ES_Descriptor */
    /* ES_ID(2) + flags(1) + DCD tag/size(2) + SL tag/size(2). */
    mb_u8(&esds, (unsigned)(dcd.len + sl.len + 7));
    mb_u16(&esds, 1);              /* ES_ID */
    mb_u8(&esds, 0x00);            /* no flags, no dependency */
    mb_u8(&esds, 0x04);            /* DecoderConfigDescriptor */
    mb_u8(&esds, (unsigned)dcd.len);
    mb_put(&esds, dcd.p, dcd.len);
    mb_u8(&esds, 0x06);            /* SLConfigDescriptor */
    mb_u8(&esds, (unsigned)sl.len);
    mb_put(&esds, sl.p, sl.len);

    mb_box(&e, "esds", &esds);
    mb_box(out, "mp4a", &e);
    mb_free(&e);
    mb_free(&esds);
    mb_free(&dsi);
    mb_free(&dcd);
    mb_free(&sl);
}

unsigned char *h3v_mp4_encode(const unsigned char *rgb, int64_t frames, int width,
                              int height, const float *audio, int64_t samples,
                              size_t *out_len, cwick_error *err) {
    if ((!rgb || frames <= 0) && (!audio || samples <= 0)) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "nothing to mux");
        return NULL;
    }
    mbtrack t;
    memset(&t, 0, sizeof(t));

    unsigned char *avcc = NULL, *asc = NULL;
    size_t avcc_len = 0, asc_len = 0;
    bool have_video = false, have_audio = false;

    if (rgb && frames > 0)
        have_video = encode_video(rgb, frames, width, height, &t, &avcc, &avcc_len,
                                  err);
    if (audio && samples > 0)
        have_audio = encode_audio(audio, samples, &t, &asc, &asc_len, err);

    unsigned char *result = NULL;
    if (have_video || have_audio) {
        mb out, ftyp, mdat, moov;
        memset(&out, 0, sizeof(out));
        memset(&ftyp, 0, sizeof(ftyp));
        memset(&mdat, 0, sizeof(mdat));
        memset(&moov, 0, sizeof(moov));

        mb_put(&ftyp, "isom", 4);
        mb_u32(&ftyp, 512);
        mb_put(&ftyp, "isomiso2avc1mp41", 16);
        mb_box(&out, "ftyp", &ftyp);

        /* Offsets are relative to the start of the file and are known now
         * because mdat's payload is appended straight after the ftyp. */
        const uint32_t video_off = (uint32_t)(out.len + 8);
        for (size_t i = 0; i < t.nv; i++) mb_put(&mdat, t.v[i].data, t.v[i].len);
        const uint32_t audio_off = (uint32_t)(out.len + 8 + mdat.len);
        for (size_t i = 0; i < t.na; i++) mb_put(&mdat, t.a[i].data, t.a[i].len);

        const uint32_t v_dur = (uint32_t)(t.nv * 1000u);   /* timescale 24000 */
        const uint32_t a_dur = (uint32_t)(t.na * 1024u);   /* timescale 32000 */
        uint32_t mv_dur = v_dur / 24;
        if (a_dur / 32 > mv_dur) mv_dur = a_dur / 32;
        if (mv_dur == 0) mv_dur = 1;

        put_mvhd(&moov, 1000, mv_dur, have_video && have_audio ? 3 : 2);
        uint32_t track_id = 0;

        if (have_video) {
            mb trak, mdia, minf, stsd, entry, hdr;
            memset(&trak, 0, sizeof(trak));
            memset(&mdia, 0, sizeof(mdia));
            memset(&minf, 0, sizeof(minf));
            memset(&stsd, 0, sizeof(stsd));
            memset(&entry, 0, sizeof(entry));
            memset(&hdr, 0, sizeof(hdr));

            /* tkhd's duration is in the movie timescale, not the media
             * timescale used by mdhd. */
            put_tkhd(&trak, ++track_id, mv_dur, width, height, 0);
            put_mdhd(&mdia, 24000, v_dur);
            put_hdlr(&mdia, "vide", "VideoHandler");
            mb_full(&hdr, 0, 1);
            mb_zero(&hdr, 8);
            mb_box(&minf, "vmhd", &hdr);
            put_dinf(&minf);
            put_avc1(&entry, avcc, avcc_len, width, height);
            mb_full(&stsd, 0, 0);
            mb_u32(&stsd, 1);
            mb_put(&stsd, entry.p, entry.len);
            put_stbl(&minf, &stsd, 1000, video_off, t.v, (uint32_t)t.nv, 1, 1);
            mb_box(&mdia, "minf", &minf);
            mb_box(&trak, "mdia", &mdia);
            mb_box(&moov, "trak", &trak);
            mb_free(&trak); mb_free(&mdia); mb_free(&minf);
            mb_free(&stsd); mb_free(&entry); mb_free(&hdr);
        }
        if (have_audio) {
            mb trak, mdia, minf, stsd, entry, hdr;
            memset(&trak, 0, sizeof(trak));
            memset(&mdia, 0, sizeof(mdia));
            memset(&minf, 0, sizeof(minf));
            memset(&stsd, 0, sizeof(stsd));
            memset(&entry, 0, sizeof(entry));
            memset(&hdr, 0, sizeof(hdr));

            put_tkhd(&trak, ++track_id, mv_dur, 0, 0, 1);
            put_mdhd(&mdia, 32000, a_dur);
            put_hdlr(&mdia, "soun", "SoundHandler");
            mb_full(&hdr, 0, 0);
            mb_u16(&hdr, 0);
            mb_u16(&hdr, 0);
            mb_box(&minf, "smhd", &hdr);
            put_dinf(&minf);
            put_mp4a(&entry, asc, asc_len, 32000, 2);
            mb_full(&stsd, 0, 0);
            mb_u32(&stsd, 1);
            mb_put(&stsd, entry.p, entry.len);
            put_stbl(&minf, &stsd, 1024, audio_off, t.a, (uint32_t)t.na, 0, 0);
            mb_box(&mdia, "minf", &minf);
            mb_box(&trak, "mdia", &mdia);
            mb_box(&moov, "trak", &trak);
            mb_free(&trak); mb_free(&mdia); mb_free(&minf);
            mb_free(&stsd); mb_free(&entry); mb_free(&hdr);
        }

        mb_box(&out, "mdat", &mdat);
        /* `moov` holds box payloads; it still needs its own box header.
         * Writing its contents bare leaves no moov atom at all and every
         * player reports "moov atom not found". */
        mb_box(&out, "moov", &moov);
        result = out.p;
        *out_len = out.len;
        mb_free(&mdat);
        mb_free(&moov);
        mb_free(&ftyp);
    }

    free(avcc);
    free(asc);
    mbsample_free(t.v, t.nv);
    mbsample_free(t.a, t.na);
    return result;
}
