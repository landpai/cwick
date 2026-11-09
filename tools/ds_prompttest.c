/*
 * ds_prompttest.c — check the C prompt encoder against the byte strings the
 * checkpoint's own encoding README prints.
 *
 *   cc -O2 -Isrc -o tools/ds_prompttest tools/ds_prompttest.c \
 *      src/ds_prompt.c src/json.c src/util.c src/log.c src/mem.c
 *   ./tools/ds_prompttest
 *
 * Every case below is copied from `encoding/README.md`, so a pass means the C
 * renderer reproduces the reference format exactly, not merely something that
 * looks similar.
 */
#include "ds_prompt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int n_checks, n_fail;

static void expect(const char *what, const char *got, const char *want) {
    n_checks++;
    if (!got) {
        n_fail++;
        printf("  FAIL %-22s encoder returned NULL\n", what);
        return;
    }
    if (strcmp(got, want) == 0) return;
    n_fail++;
    printf("  FAIL %-22s\n    want: %s\n    got : %s\n", what, want, got);
    /* first differing byte, which is what actually needs fixing */
    size_t i = 0;
    while (got[i] && want[i] && got[i] == want[i]) i++;
    printf("    first difference at byte %zu\n", i);
}

static char *encode(const char *json, const char *tools_json,
                    ds_thinking_mode mode, int effort, bool drop) {
    cwick_error err;
    cwick_error_clear(&err);
    json_value *msgs = cwick_json_parse(json, strlen(json), &err);
    if (!msgs) {
        printf("  parse error: %s\n", cwick_error_message(&err));
        return NULL;
    }
    json_value *tools = NULL;
    if (tools_json) {
        tools = cwick_json_parse(tools_json, strlen(tools_json), &err);
        if (!tools) {
            printf("  tools parse error: %s\n", cwick_error_message(&err));
            cwick_json_free(msgs);
            return NULL;
        }
    }
    ds_prompt_opts o;
    ds_prompt_opts_default(&o);
    o.thinking = mode;
    o.reasoning_effort = effort;
    o.drop_thinking = drop;
    size_t n = 0;
    char *p = ds_prompt_encode(msgs, tools, &o, &n, &err);
    if (!p) printf("  encode error: %s\n", cwick_error_message(&err));
    cwick_json_free(msgs);
    cwick_json_free(tools);
    return p;
}

#define BOS "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence" "\xef\xbd\x9c>"
#define EOS "<\xef\xbd\x9c" "end\xe2\x96\x81of\xe2\x96\x81sentence" "\xef\xbd\x9c>"
#define USR "<\xef\xbd\x9c" "User" "\xef\xbd\x9c>"
#define AST "<\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c>"
#define SYS "<\xef\xbd\x9c" "System" "\xef\xbd\x9c>"

int main(void) {
    char *p;

    /* README "Quick start": system + user, thinking mode, effort 75. */
    p = encode("[{\"role\":\"system\",\"content\":\"You are a helpful assistant.\"},"
               "{\"role\":\"user\",\"content\":\"What is 2+2?\"}]",
               NULL, DS_THINKING_THINKING, 75, true);
    expect("quickstart/thinking", p,
           BOS SYS "Reasoning Effort: 75 (range 1-100, the higher the value, "
           "the more thorough the reasoning)\n\nYou are a helpful assistant."
           USR "What is 2+2?" AST "<think>");
    free(p);

    /* Same conversation in chat mode: the thinking block is closed at once. */
    p = encode("[{\"role\":\"user\",\"content\":\"hi\"}]",
               NULL, DS_THINKING_CHAT, 0, true);
    expect("chat-mode header", p, BOS USR "hi" AST "</think>");
    free(p);

    /* README "Basic chat": a completed turn keeps reasoning then content. */
    p = encode("[{\"role\":\"user\",\"content\":\"a\"},"
               "{\"role\":\"assistant\",\"content\":\"b\"},"
               "{\"role\":\"user\",\"content\":\"c\"}]",
               NULL, DS_THINKING_CHAT, 0, true);
    expect("two turns", p,
           BOS USR "a" AST "</think>b" EOS USR "c" AST "</think>");
    free(p);

    /* Default effort: "high" is 75. A bare reasoning_effort of 0 means unset. */
    p = encode("[{\"role\":\"system\",\"content\":\"s\"},"
               "{\"role\":\"user\",\"content\":\"u\"}]",
               NULL, DS_THINKING_THINKING, 0, true);
    expect("default effort", p,
           BOS SYS "Reasoning Effort: 75 (range 1-100, the higher the value, "
           "the more thorough the reasoning)\n\ns" USR "u" AST "<think>");
    free(p);

    /* Effort 1 and 100 are the ends of the documented range. */
    p = encode("[{\"role\":\"user\",\"content\":\"u\"}]",
               NULL, DS_THINKING_THINKING, 1, true);
    expect("effort 1", p,
           BOS SYS "Reasoning Effort: 1 (range 1-100, the higher the value, "
           "the more thorough the reasoning)\n\n" USR "u" AST "<think>");
    free(p);

    /* Reasoning is dropped for turns before the last user turn. */
    p = encode("[{\"role\":\"user\",\"content\":\"q1\"},"
               "{\"role\":\"assistant\",\"content\":\"a1\","
               "\"reasoning_content\":\"R1\"},"
               "{\"role\":\"user\",\"content\":\"q2\"}]",
               NULL, DS_THINKING_THINKING, 0, true);
    expect("drop_thinking", p,
           BOS SYS "Reasoning Effort: 75 (range 1-100, the higher the value, "
           "the more thorough the reasoning)\n\n" USR "q1" AST "</think>a1" EOS
           USR "q2" AST "<think>");
    free(p);

    /* With drop_thinking off the earlier reasoning survives. The user turn
     * still opens a thinking block, so the turn reads <think>R1</think>. */
    p = encode("[{\"role\":\"user\",\"content\":\"q1\"},"
               "{\"role\":\"assistant\",\"content\":\"a1\","
               "\"reasoning_content\":\"R1\"},"
               "{\"role\":\"user\",\"content\":\"q2\"}]",
               NULL, DS_THINKING_THINKING, 0, false);
    expect("keep_thinking", p,
           BOS SYS "Reasoning Effort: 75 (range 1-100, the higher the value, "
           "the more thorough the reasoning)\n\n" USR "q1" AST "<think>R1</think>a1" EOS
           USR "q2" AST "<think>");
    free(p);

    /* Mid-conversation system message: <｜System｜> ranks as a user turn for
     * reasoning retention, but the transition rule only emits a header when
     * the next turn is an assistant one, so none follows the note here. */
    p = encode("[{\"role\":\"user\",\"content\":\"u1\"},"
               "{\"role\":\"assistant\",\"content\":\"a1\"},"
               "{\"role\":\"system\",\"content\":\"note\"},"
               "{\"role\":\"user\",\"content\":\"u2\"}]",
               NULL, DS_THINKING_CHAT, 0, true);
    expect("mid system", p,
           BOS USR "u1" AST "</think>a1" EOS SYS "note"
           USR "u2" AST "</think>");
    free(p);

    /* role:tool folds into a <tool_result> block inside a user turn. */
    p = encode("[{\"role\":\"user\",\"content\":\"u1\"},"
               "{\"role\":\"assistant\",\"content\":\"\","
               "\"tool_calls\":[{\"type\":\"function\",\"function\":"
               "{\"name\":\"f\",\"arguments\":\"{\\\"x\\\":1}\"}}]},"
               "{\"role\":\"tool\",\"tool_call_id\":\"c1\","
               "\"content\":\"{\\\"ok\\\":true}\"}]",
               NULL, DS_THINKING_CHAT, 0, true);
    expect("tool result", p,
           BOS USR "u1" AST "</think>"
           "\n\n<\xef\xbd\x9c" "DSML" "\xef\xbd\x9c calls>\n"
           "<\xef\xbd\x9c" "DSML" "\xef\xbd\x9c invoke name=\"f\">\n"
           "<\xef\xbd\x9c" "DSML" "\xef\xbd\x9c parameter name=\"x\" "
           "string=\"false\">1</\xef\xbd\x9c" "DSML" "\xef\xbd\x9c parameter>"
           "\n</\xef\xbd\x9c" "DSML" "\xef\xbd\x9c invoke>"
           "\n</\xef\xbd\x9c" "DSML" "\xef\xbd\x9c calls>" EOS
           USR "<tool_result>{\"ok\":true}</tool_result>" AST "</think>");
    free(p);

    /* The empty prompt still gets a BOS and nothing else. */
    p = encode("[]", NULL, DS_THINKING_CHAT, 0, true);
    expect("empty", p, BOS);
    free(p);

    /* An image block is refused rather than silently flattened. */
    {
        cwick_error err;
        cwick_error_clear(&err);
        const char *j = "[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\","
                        "\"image_url\":{\"url\":\"x\"}}]}]";
        json_value *msgs = cwick_json_parse(j, strlen(j), &err);
        ds_prompt_opts o;
        ds_prompt_opts_default(&o);
        char *q = ds_prompt_encode(msgs, NULL, &o, NULL, &err);
        n_checks++;
        if (q) {
            n_fail++;
            printf("  FAIL %-22s image block accepted\n", "image rejected");
            free(q);
        } else if (err.status != CWICK_ERR_UNSUPPORTED) {
            n_fail++;
            printf("  FAIL %-22s wrong error: %s\n", "image rejected",
                   cwick_error_message(&err));
        }
        cwick_json_free(msgs);
    }

    printf("\n%d checks, %d failure(s)\n", n_checks, n_fail);
    return n_fail ? 1 : 0;
}
