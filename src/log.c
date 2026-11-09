/* log.c — see log.h. Everything lands on stdout, line-atomic. */
#include "log.h"

#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static bool g_verbose = false;
static bool g_trace = false;
static bool g_quiet = false;
static bool g_clock_started = false;
static struct timeval g_t0;

/* Request the calling thread is serving. Thread-local: one thread per
 * connection, so two concurrent requests must not read each other's id. */
static _Thread_local uint64_t g_req_id;   /* 0 = line belongs to no request */

/* Monotonic-ish seconds since the first log line, so timestamps do not jump
 * when NTP steps the wall clock mid-run. */
static double elapsed_s(void) {
    struct timeval now;
    gettimeofday(&now, NULL);
    if (!g_clock_started) {
        g_t0 = now;
        g_clock_started = true;
    }
    return (double)(now.tv_sec - g_t0.tv_sec) +
           (double)(now.tv_usec - g_t0.tv_usec) / 1e6;
}

/* Wall-clock system time, rendered as YYYY-MM-DD HH:MM:SS.mmm. Every log
 * line carries the current system time so a long-running serve (or a coding
 * agent tailing stdout) can correlate lines with the wall clock. */
static void wallclock(char *buf, size_t cap) {
    struct timeval now;
    gettimeofday(&now, NULL);
    time_t s = (time_t)now.tv_sec;
    struct tm tmv;
    localtime_r(&s, &tmv);
    snprintf(buf, cap, "%04d-%02d-%02d %02d:%02d:%02d.%03d",
             tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
             tmv.tm_hour, tmv.tm_min, tmv.tm_sec,
             (int)(now.tv_usec / 1000));
}

void cwick_log_config(bool verbose, bool trace, bool quiet) {
    g_verbose = verbose;
    g_trace = trace;
    g_quiet = quiet;
}

void cwick_log_set_request(uint64_t id) { g_req_id = id; }
uint64_t cwick_log_request(void) { return g_req_id; }

bool cwick_log_enabled(cwick_log_level level) {
    switch (level) {
        case CWICK_LOG_INFO:    return !g_quiet;
        case CWICK_LOG_VERBOSE: return g_verbose && !g_quiet;
        case CWICK_LOG_TRACE:   return g_trace && !g_quiet;
        default:               return false;
    }
}

static const char *level_tag(cwick_log_level level) {
    switch (level) {
        case CWICK_LOG_INFO:    return "info";
        case CWICK_LOG_VERBOSE: return "verb";
        case CWICK_LOG_TRACE:   return "trace";
        default:               return "?";
    }
}

/* `[req N] ` for the request this thread is serving, or "" when it is not
 * serving one. Kept short because it rides on every line. */
static void req_prefix(char *buf, size_t cap, uint64_t id) {
    if (id == 0) { buf[0] = '\0'; return; }
    snprintf(buf, cap, "[req %llu] ", (unsigned long long)id);
}

/* Callers that were already tagging their lines "req 12" would now print the
 * id twice. The prefix is the canonical one, so drop the tag when it says the
 * same thing; anything else ("prefill", "gen", "http") is still useful. */
static bool tag_repeats_request(const char *tag, uint64_t id) {
    if (!tag || id == 0 || strncmp(tag, "req ", 4) != 0) return false;
    char *end = NULL;
    unsigned long long v = strtoull(tag + 4, &end, 10);
    return end && *end == '\0' && v == (unsigned long long)id;
}

void cwick_logf(cwick_log_level level, const char *tag, const char *fmt, ...) {
    if (!cwick_log_enabled(level)) return;

    char msg[4096];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);

    double t = elapsed_s();
    char wc[48], rq[40];
    wallclock(wc, sizeof(wc));
    req_prefix(rq, sizeof(rq), g_req_id);
    pthread_mutex_lock(&g_lock);
    if (tag && *tag && !tag_repeats_request(tag, g_req_id))
        fprintf(stdout, "cwick: %s [%s %+7.2fs] %s[%s] %s\n", wc,
                level_tag(level), t, rq, tag, msg);
    else
        fprintf(stdout, "cwick: %s [%s %+7.2fs] %s%s\n", wc, level_tag(level), t,
                rq, msg);
    fflush(stdout);
    pthread_mutex_unlock(&g_lock);
}

void cwick_log_block(cwick_log_level level, const char *tag, const char *body,
                    size_t max_chars, int indent) {
    if (!cwick_log_enabled(level)) return;
    if (!body) body = "";
    if (indent < 0) indent = 0;
    if (indent > 32) indent = 32;

    pthread_mutex_lock(&g_lock);
    double t = elapsed_s();
    char wc[48], rq[40];
    wallclock(wc, sizeof(wc));
    req_prefix(rq, sizeof(rq), g_req_id);
    fprintf(stdout, "cwick: %s [%s %+7.2fs] %s[%s]\n", wc, level_tag(level), t, rq,
            (tag && *tag && !tag_repeats_request(tag, g_req_id)) ? tag : "-");
    {
        char pad[64];
        size_t pn = (size_t)indent;
        for (size_t i = 0; i < pn; i++) pad[i] = ' ';
        pad[pn] = '\0';

        size_t shown = 0;
        const char *p = body;
        while (*p && shown < max_chars) {
            const char *nl = strchr(p, '\n');
            size_t len = nl ? (size_t)(nl - p) : strlen(p);
            size_t room = max_chars - shown;
            int truncated = 0;
            if (len > room) { len = room; truncated = 1; }
            fputs(pad, stdout);
            fwrite(p, 1, len, stdout);
            fputc('\n', stdout);
            shown += len;
            p += len;
            if (truncated) { fputs(pad, stdout); fputs("…\n", stdout); break; }
            if (!nl) break;
            p = nl + 1;
        }
        if (!*p) { /* emitted everything */ }
    }
    fflush(stdout);
    pthread_mutex_unlock(&g_lock);
}
