/* Standalone VAD A/B harness. Build: make rtvad
 *
 * Feeds identical synthetic 16 kHz signals through the production state
 * machine twice: once with the original single threshold, once with
 * hysteresis + short-fragment rejection. Prints the decision events and a
 * summary so the change can be judged on numbers, not impressions.
 */
#include "rt_vad.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RATE 16000

static const float THRESH = 0.5f;
static const int PREFIX = 300;
static const int SILENCE = 200;

/* Fill a frame-aligned buffer at a given dBFS (rms == 10^(db/20)). */
static void tone(float *p, int n, double db) {
    double a = pow(10.0, db / 20.0), phase = 0.0;
    for (int i = 0; i < n; i++) {
        p[i] = (float)(a * sin(phase));
        phase += 2.0 * M_PI * 180.0 / RATE;   /* 180 Hz, rms ~ a/sqrt(2) */
    }
    /* Normalise so rms is exactly a/sqrt(2) independent of partial cycles. */
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)p[i] * p[i];
    double rms = sqrt(s / n);
    if (rms > 1e-12) {
        double g = (a / sqrt(2.0)) / rms;
        for (int i = 0; i < n; i++) p[i] = (float)(p[i] * g);
    }
}

static void run(const float *pcm, int n, bool hyst) {
    rt_vad_state v;
    rt_vad_state_init(&v);
    int starts = 0, stops = 0, drops = 0;
    int first_start = -1, first_stop = -1;
    printf("  %-9s ", hyst ? "enhanced" : "legacy");
    for (int off = 0; off + RATE / 100 <= n; off += RATE / 100) {
        int st; bool stopped, discarded;
        rt_vad_feed(&v, pcm + off, RATE / 100, RATE, THRESH, PREFIX, SILENCE,
                    hyst, RT_VAD_MIN_ACTIVE_MS, &st, &stopped, &discarded);
        if (st >= 0) {
            starts++;
            if (first_start < 0) first_start = v.in_ms;
            printf("S%d ", v.in_ms);
        }
        if (stopped) {
            stops++;
            if (first_stop < 0) first_stop = v.in_ms;
            printf("E%d ", v.in_ms);
        }
        if (discarded) { drops++; printf("x%d ", v.in_ms); }
    }
    printf("| starts=%d ends=%d drops=%d%s\n", starts, stops, drops,
           (first_start >= 0 && first_stop >= 0)
               ? (first_stop > first_start ? "  [one full turn]" : "  [ordering!]") : "");
}

/* 1. Clean speech: 1.0 s at -25 dB, 0.4 s silence, 1.0 s at -25 dB.
 *    A word-sized gap should not split the utterance (silence 200 ms). */
static void case_word_gap(void) {
    int a = RATE, gap = (int)(0.4 * RATE), n = 2 * a + gap;
    float *p = calloc((size_t)n, sizeof(float));
    tone(p, a, -25);
    tone(p + a + gap, a, -25);
    printf("word gap (1s speech / 0.4s quiet / 1s speech, quiet=200ms):\n");
    run(p, n, false);
    run(p, n, true);
    free(p);
}

/* 2. A click: 40 ms at -20 dB between two quiet stretches. Nothing else.
 *    Should never be committed as a turn. */
static void case_click(void) {
    int pre = RATE / 2, click = (int)(0.04 * RATE), post = RATE / 2;
    int n = pre + click + post;
    float *p = calloc((size_t)n, sizeof(float));
    tone(p + pre, click, -20);
    printf("click (40 ms burst, no real speech):\n");
    run(p, n, false);
    run(p, n, true);
    free(p);
}

/* 3. Mid-word dip: speech at -25 dB with a 260 ms dip to -38 dB (below the
 *    legacy -40 dB onset, above the -44.5 dB release). Legacy ends the turn;
 *    the hold region should keep it open. */
static void case_midword_dip(void) {
    int a = RATE / 2, dip = (int)(0.26 * RATE), b = RATE / 2, tail = RATE / 2;
    int n = a + dip + b + tail;
    float *p = calloc((size_t)n, sizeof(float));
    tone(p, a, -25);
    tone(p + a, dip, -38);
    tone(p + a + dip, b, -25);
    printf("mid-word dip (0.5s speech / 0.26s at -38dB / 0.5s speech / 0.5s quiet):\n");
    run(p, n, false);
    run(p, n, true);
    free(p);
}

/* 4. Ordinary speech with a normal 500 ms end pause: both must still end. */
static void case_normal_end(void) {
    int a = RATE, tail = RATE / 2;
    int n = a + tail;
    float *p = calloc((size_t)n, sizeof(float));
    tone(p, a, -25);
    printf("normal end (1s speech / 0.5s quiet):\n");
    run(p, n, false);
    run(p, n, true);
    free(p);
}

int main(void) {
    printf("VAD A/B: threshold=%.2f prefix=%dms silence=%dms min_active=%dms\n\n",
           (double)THRESH, PREFIX, SILENCE, RT_VAD_MIN_ACTIVE_MS);
    case_word_gap();
    printf("\n");
    case_click();
    printf("\n");
    case_midword_dip();
    printf("\n");
    case_normal_end();
    return 0;
}
