/* engine.c — one-process native target engine.
 *
 *   attention residual: softmax-weighted combination of the saved anchor
 *   rows (every attn_res_block_size layers) and the current prefix, using a
 *   learned score vector (res_norm * res_proj);
 *   per layer: rms_norm(input) -> attention -> prefix = pre_attn + attn_out
 *   -> residual-combine -> rms_norm(post) -> dense/MoE -> hidden = prefix + out;
 *   tail: residual-combine with the final anchors, final rms_norm, lm_head.
 *
 * The decode loop is greedy by default and reproducible whenever the sampler
 * is: with temperature <= 0 (the default, and what `temperature: 0` asks for)
 * the next token is the argmax, and with temperature > 0 the draw comes from a
 * seeded generator (see sample.c). Which of the two a request gets is decided
 * by the checkpoint's generation_config.json unless the request overrides it.
 */
#include "cwick.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "log.h"
#include "mem.h"
#include "metrics.h"
#include "util.h"
#include "pfx.h"
#include "ds_engine.h"
#include "mm_engine.h"
#include "q35_engine.h"
#include "sample.h"

/* A turn ends on any id in the checkpoint's stop set, not just the first. */
static bool is_eos(const cwick_model_spec *spec, uint32_t tok) {
    for (int i = 0; i < spec->n_eos; i++)
        if ((int64_t)tok == spec->eos_ids[i]) return true;
    return (int64_t)tok == spec->eos_token_id;
}

/* How often (in decode steps) the inference loop compares its real RSS with
 * the memory plan and gives back pinned expert bytes if it drifted over.
 * CWICK_STEER_EVERY lives in cwick.h: the DeepSeek decode loop steers too. */
/* How often a progress line is printed (see maybe_progress). The default is
 * verbose-only; CWICK_PROGRESS_EVERY overrides it. */
#define CWICK_PROGRESS_EVERY 8
/* A layer-granular heartbeat is also due when this much wall time has passed
 * since the last one, whatever the layer counter says. A prefill layer can
 * take minutes on a 30K-token prompt, and "every 8 layers" then means one
 * line every quarter of an hour -- indistinguishable from a hang.
 * CWICK_PROGRESS_MIN_MS overrides. */
#define CWICK_PROGRESS_MIN_MS 5000.0
/* Token heartbeat inside one layer's serial sweep: one line per this many
 * tokens, further throttled by CWICK_PROGRESS_MIN_MS. */
#define CWICK_PREFILL_TOKEN_EVERY 2048
/* Above this prompt length, warn once per request that the first token will
 * take minutes: an agent client that sends a 30K-token system prompt
 * otherwise just hangs with no explanation. */
#define CWICK_LONG_PROMPT_TOKENS 4096

struct cwick_engine {
    cwick_model_spec spec;
    /* DeepSeek-V4.1 backend. Non-NULL exactly when spec.model_type is
     * deepseek_v41*, in which case every GLM member below is unused and the
     * spine stays closed: that architecture has its own loader, caches and
     * forward pass in ds_*.c, and the two share only this struct. */
    ds_engine *ds;
    /* MiniMax-M3.1-Flash backend. Same deal as `ds`: non-NULL exactly when
     * the checkpoint is that architecture, and then every GLM member below is
     * unused. It is a separate pointer rather than a union because the rest of
     * this file asks "which backend owns this engine" at a dozen points, and a
     * wrong guess there is a crash rather than a stale number. */
    mm_engine *mm;
    /* Qwen3.5 hybrid backend. Same deal as `ds` and `mm`: non-NULL exactly
     * when the checkpoint is that architecture. Dense BF16, so unlike the
     * other two it streams nothing and budgets no expert cache. */
    q35_engine *q35;
    cwick_spine spine;
    /* `spine` is embedded, not allocated, so a second engine can share the
     * weights by copying the struct: every member is a pointer to something
     * owned elsewhere (mmaps, the parsed shard index, the F_NOCACHE stream,
     * the expert cache, the memory plan). Copying it shares all of them,
     * which is what lets several generation slots exist without paying for
     * the resident spine more than once -- a spine is 15.46 GiB and cwick
     * mlocks it, so N engines each opening their own would need N x 15.46.
     * `spine_owned` says whether this engine is the one that must release it.
     *
     * What is NOT shared is the per-sequence state: keys, kda_rec, kda_conv,
     * idx_state, anchors, pos. Each slot decodes its own sequence. */
    bool spine_owned;
    int  slot_id;        /* which generation slot this engine serves */
    /* Request tag for log attribution ("req 12"). Empty for CLI runs. */
    char req_tag[24];
    char *model_root;
    int64_t capacity;    /* KV positions */
    /* keys holds either the MLA latent ([L][capacity][kv_lora]) or the
     * materialised K ([L][capacity][nh*q_dim]); `values` is NULL exactly when
     * the latent layout is in use (see attention.c). */
    float *keys;
    float *values;
    float *kda_rec;      /* [L * lin_heads * head_dim * head_dim] */
    float *kda_conv;     /* [L * 3 * proj * 3] */
    float *anchors;      /* [max_anchors * H] */
    int64_t n_anchors;
    int64_t max_anchors;
    int64_t kv_layers;       /* number of layers that own a KV cache slot */
    int64_t *kv_slot;        /* layer -> compact KV slot, or -1 */
    bool mla_latent;         /* KV cache stores the MLA latent, not expanded K/V */
    /* DSA indexer (per KV slot, NULL when the checkpoint has no indexer) */
    cwick_indexer_state *idx_state;
    cwick_indexer_weights *idx_w;
    int64_t pos;         /* next position */
    /* ---- MTP draft (CWICK_MTP=1) --------------------------------------
     * A decode step is dominated by pulling one token's routed experts off
     * the SSD, and the routers of two adjacent tokens name 13.0 experts on
     * average rather than 16 (measured, docs/PERF.md), so a second token in
     * the same batch costs 1.63x the expert bytes for two tokens of work.
     * That is the only per-token saving left in this design, and it needs a
     * guess at token t+2 cheap enough to throw away.
     *
     * GLM-5.3-Flash trains that guess: layer `num_hidden_layers` takes h_t
     * and the embedding of t+1, fuses them through eh_proj, and runs one
     * full DSA+MoE block to predict t+2. Everything below is inert unless
     * the checkpoint ships that block and CWICK_MTP asks for it.
     *
     * This stage only *drafts and scores*: it does not change what the model
     * emits, so a wrong draft costs nothing but a log line. */
    bool mtp_on;
    float *mtp_h;        /* h_t: contracted hidden of the most recent forward */
    float *mtp_keys;     /* the MTP block's own KV: [capacity][latent] */
    float *mtp_work;     /* 2H + H scratch */
    float *mtp_logits;   /* vocab scratch for the draft */
    uint32_t mtp_draft;  /* the token drafted on the previous step */
    bool mtp_have_draft;
    int64_t mtp_tries;   /* drafts whose real continuation became known */
    int64_t mtp_hits;    /* ... and matched it */
    int64_t max_new;
    /* How the decode loop turns logits into the next token. Defaulted from
     * the checkpoint's generation_config.json at open (a checkpoint that was
     * trained with do_sample=true is served with its own temperature/top_p/
     * top_k), overridden per request by cwick_engine_set_sampler. */
    cwick_sampler_cfg smp;
    /* What generation_config.json asked for, kept beside the active config:
     * a request that sends only top_p must inherit the checkpoint's
     * temperature, not a hard-coded one, so the server needs the original
     * defaults every time it rebuilds `smp`. */
    cwick_sampler_cfg smp_ckpt;
    bool smp_ckpt_have;    /* generation_config.json was present and usable */
    bool verbose;
    uint64_t streamed_per_token; /* expert bytes re-read per decode step */
    volatile int *stop;          /* Ctrl+C: polled between decode steps */
    /* Optional extra interrupt source, polled at the same points as `stop`.
     *
     * The engine decodes a whole response before the server writes a byte, so
     * without this a client that gives up (Ctrl+C, proxy timeout, closed tab)
     * is invisible here: generation runs to the end of the budget anyway,
     * holding the generation slot the entire time. With one slot that means
     * every other request gets 429 for as long as the abandoned one would
     * have taken -- 8192 tokens at ~1.3 tok/s is ~106 minutes. The server
     * passes a predicate that peeks at the socket so the loop can bail out. */
    bool (*abort_poll)(void *user);
    void *abort_poll_user;
    uint64_t rss_target;         /* memory ceiling to steer toward */
    /* Where to snapshot the state right after the next prefill (prefix
     * cache). NULL unless a request asked for it; cleared once filled. */
    struct cwick_prefix_state *pfx_cap;
    /* Invoked the moment pfx_cap is filled, so the caller can persist the
     * snapshot while the prefill that produced it is still the request's
     * present (see cwick_engine_set_prefix_capture_done). Takes ownership of
     * the state. */
    void (*pfx_cap_done)(void *user, struct cwick_prefix_state *s);
    void *pfx_cap_done_user;
    /* Full prompt size in tokens, BEFORE any prefix-cache replay. The engine
     * itself only ever sees the replayed remainder (it starts at the first
     * unmasked token), so without this the summary line reports the tail
     * length -- "prompt=1" for a 499-token request whose prefix was replayed.
     * 0 means "not told"; consumers then fall back to the actual length. */
    int64_t prompt_total;
    /* Decode-phase hint for the progress lines: "thinking" | "content" |
     * "tools" (static strings, or NULL = unknown). The server classifies each
     * token with the same state machine that splits the SSE stream, so the
     * per-step lines say which part of the answer the step produced. Not the
     * engine's business to know the semantics, so it just prints the hint. */
    const char *decode_phase;
    /* Snapshot the prefix state every this many prefilled tokens, not only
     * once at the end (see pfx_checkpoint_now). 0 = end only. Caps the prefill
     * chunk so every chunk boundary lands on a multiple of it. */
    int64_t pfx_block;
    /* Thinking budget (see cwick_engine_set_think_budget). `think_budget` is
     * the ceiling in generated tokens; -1 disables the mechanism entirely.
     * The two ids are the tokenizer's single-token markers; -1 means "the
     * tokenizer does not give this marker as one token", which also disables
     * it. The engine writes them itself when the ceiling is reached, which is
     * the difference between a thought that is cut short and a turn that
     * spends every token it has inside one. */
    int64_t think_budget;
    int64_t think_open_id;
    int64_t think_close_id;
    bool think_open_at_start;   /* the template already appended <think> */
};

void cwick_engine_set_prefix_capture(cwick_engine *eng,
                                    struct cwick_prefix_state *out) {
    if (eng) eng->pfx_cap = out;
}

void cwick_engine_set_prefix_capture_block(cwick_engine *eng, int64_t block) {
    if (!eng) return;
    eng->pfx_block = block > 0 ? block : 0;
    /* The DeepSeek backend runs its own prefill loop, so it needs the stride
     * to know when to fire a checkpoint. */
    if (eng->ds) eng->ds->checkpoint_block = eng->pfx_block;
    if (eng->mm) eng->mm->checkpoint_block = eng->pfx_block;
}

void cwick_engine_set_prefix_capture_done(
    cwick_engine *eng, void (*fn)(void *user, struct cwick_prefix_state *s),
    void *user) {
    if (!eng) return;
    eng->pfx_cap_done = fn;
    eng->pfx_cap_done_user = user;
}

/* Tell the engine the prompt's full token count when the server replayed a
 * cached prefix beforehand (engine-side `prompt_len` is then only the tail).
 * The summary line and prefill tok/s use it so operators see the real size. */
void cwick_engine_set_prompt_total(cwick_engine *eng, int64_t total) {
    if (eng) eng->prompt_total = total;
}

/* Arm (or clear) the per-response thinking ceiling. See cwick.h for why the
 * engine, and not the server, writes the close tag: the context has to move
 * with it, and only the decode loop can feed a token it did not sample. */
void cwick_engine_set_think_budget(cwick_engine *eng, int64_t budget,
                                  int64_t open_id, int64_t close_id,
                                  bool open_at_start) {
    if (!eng) return;
    /* Without both single-token markers there is no way to write a valid
     * block, so the mechanism stays off rather than emitting half a marker. */
    if (budget < 0 || open_id < 0 || close_id < 0 ||
        open_id >= eng->spec.vocab_size || close_id >= eng->spec.vocab_size) {
        eng->think_budget = -1;
        if (eng->ds) eng->ds->ctx.think_budget = -1;
        if (eng->mm) eng->mm->ctx.think_budget = -1;
        if (eng->q35) eng->q35->ctx.think_budget = -1;
        return;
    }
    eng->think_budget = budget;
    eng->think_open_id = open_id;
    eng->think_close_id = close_id;
    eng->think_open_at_start = open_at_start;
    /* The DeepSeek backend decodes with its own loop, so it needs its own copy
     * of the ceiling -- the same reason ctx.stop, ctx.tag and ctx.spec are
     * duplicated there. Arming only the GLM half is what let a request run
     * 16000 thinking tokens against a configured 8192: the server logged the
     * budget as armed and the loop that actually decodes never read it. */
    if (eng->ds) {
        eng->ds->ctx.think_budget = budget;
        eng->ds->ctx.think_open_id = open_id;
        eng->ds->ctx.think_close_id = close_id;
        eng->ds->ctx.think_open_at_start = open_at_start;
    }
    if (eng->mm) {
        eng->mm->ctx.think_budget = budget;
        eng->mm->ctx.think_open_id = open_id;
        eng->mm->ctx.think_close_id = close_id;
        eng->mm->ctx.think_open_at_start = open_at_start;
    }
    if (eng->q35) {
        eng->q35->ctx.think_budget = budget;
        eng->q35->ctx.think_open_id = open_id;
        eng->q35->ctx.think_close_id = close_id;
        eng->q35->ctx.think_open_at_start = open_at_start;
    }
}

void cwick_engine_set_think_closed_at_start(cwick_engine *eng, bool closed) {
    if (!eng) return;
    if (eng->mm) eng->mm->ctx.think_closed_at_start = closed;
    if (eng->q35) eng->q35->ctx.think_closed_at_start = closed;
}

cwick_status cwick_gen_window(int64_t pos, int64_t prompt_len,
                              int64_t *max_new, int64_t capacity,
                              cwick_error *err) {
    if (!max_new || prompt_len < 0 || *max_new < 0 || capacity <= 0) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "generation window: invalid request (position %lld, "
                        "prompt %lld, max_new %lld, context %lld)",
                        (long long)pos, (long long)prompt_len,
                        (long long)(max_new ? *max_new : -1),
                        (long long)capacity);
        return CWICK_ERR_ARGUMENT;
    }
    const int64_t room = capacity - pos - prompt_len;
    if (room < 0) {
        /* The prompt itself does not fit: nothing can be generated, and the
         * client's fix is a shorter prompt or a larger --context. */
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "prompt %lld tokens at position %lld does not fit the "
                        "%lld-position context window (short by %lld); shorten "
                        "the prompt or raise --context",
                        (long long)prompt_len, (long long)pos,
                        (long long)capacity, (long long)(-room));
        return CWICK_ERR_ARGUMENT;
    }
    if (room == 0) {
        /* Fits exactly, so every position is already spoken for. */
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                        "prompt %lld tokens at position %lld fills the "
                        "%lld-position context window exactly; there is no room "
                        "left to generate",
                        (long long)prompt_len, (long long)pos,
                        (long long)capacity);
        return CWICK_ERR_ARGUMENT;
    }
    if (*max_new > room) *max_new = room;
    return CWICK_OK;
}

/* The forced-closure policy, isolated so it is testable without a model. */
cwick_think_action cwick_think_plan(int64_t gen, int64_t budget,
                                   bool think_open, bool closed) {
    if (budget < 0 || closed || gen < budget) return CWICK_THINK_NONE;
    if (think_open) return CWICK_THINK_CLOSE;
    /* Nothing opened a block: the model went straight to the answer (nothing
     * to cut), unless the client asked for no thinking at all and we are
     * still at the very first token -- then write the template's empty block
     * so this turn is an answer, not a thought. */
    return (budget == 0 && gen == 0) ? CWICK_THINK_OPEN_CLOSE
                                     : CWICK_THINK_NONE;
}

/* Log tag for everything this engine emits while generating a request. The
 * server sets it per request so every progress line can be attributed. */
void cwick_engine_set_req_tag(cwick_engine *eng, const char *tag) {
    if (!eng) return;
    if (tag && *tag)
        snprintf(eng->req_tag, sizeof(eng->req_tag), "%s", tag);
    else
        eng->req_tag[0] = '\0';
}

static const char *eng_tag(const cwick_engine *eng) {
    return (eng && eng->req_tag[0]) ? eng->req_tag : "gen";
}

/* Which part of the answer is being decoded right now, as the progress line
 * should label it: "decode" until the server can classify anything, then
 * "decode(thinking)" / "decode(content)" / "decode(tools)". */
static void decode_phase_label(const cwick_engine *eng, char *buf, size_t cap) {
    if (eng->decode_phase)
        snprintf(buf, cap, "decode(%s)", eng->decode_phase);
    else
        snprintf(buf, cap, "decode");
}

/* Publish the decode phase hint (static string, or NULL to clear). Every
 * backend keeps its own generation context, so the hint has to be copied to
 * each one: a backend left out prints plain "decode" on every progress line,
 * which is how the Qwen3.5 path lost its thinking/content/tools labels.
 * set_think_budget, set_stop, set_abort_poll and set_verbose have the same
 * obligation: every backend must appear in every setter. */
void cwick_engine_set_decode_phase(cwick_engine *eng, const char *phase) {
    if (!eng) return;
    eng->decode_phase = phase;
    if (eng->ds) eng->ds->ctx.decode_phase = phase;
    if (eng->mm) eng->mm->ctx.decode_phase = phase;
    if (eng->q35) eng->q35->ctx.decode_phase = phase;
}

/* Wall clock helper: enough resolution to price a single decode step. */
static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1e3 + (double)tv.tv_usec / 1e3;
}

/* ------------------------------------------------------------------ */
/* Phase heartbeat state (shared by the attention sweeps and the MoE)  */
/* ------------------------------------------------------------------ */
/* Wall clock of the last progress line, so the heartbeat can fire on
 * elapsed time instead of only on a fixed counter cadence -- a phase whose
 * steps are slower than the cadence would otherwise print nothing for
 * minutes while still being the thing the request is waiting on. */
static double g_progress_ms = -1e18;

static double progress_min_ms(void) {
    const char *e = getenv("CWICK_PROGRESS_MIN_MS");
    if (e) { double v = atof(e); if (v > 0.0) return v; }
    return CWICK_PROGRESS_MIN_MS;
}

/* The recurring "... — mem ... streamed ... reads" tail of every progress
 * line, factored out so the attention and MoE heartbeats cannot drift. */
static void progress_mem_tail(const cwick_engine *eng, char *buf, size_t cap) {
    const cwick_spine *sp = &eng->spine;
    uint64_t bs = 0, rc = 0, ch = 0, cm = 0;
    if (sp->stream) cwick_stream_stats(sp->stream, &bs, &rc, &ch, &cm);
    snprintf(buf, cap,
             "mem %.2f GiB footprint (%.2f GiB rss) "
             "of %.2f GiB ceiling, expert cache %.2f GiB pinned, "
             "%.2f GiB streamed, %llu reads",
             cwick_gib(cwick_process_footprint_bytes()),
             cwick_gib(cwick_process_rss_bytes()),
             cwick_gib(eng->rss_target),
             cwick_gib(cwick_stream_cache_used(sp->stream)),
             cwick_gib(bs), (unsigned long long)rc);
}

/* Progress callback handed to cwick_moe_forward_batch. The batched MoE phase
 * (router aside: 288 routed experts x 3 matrices, streamed and applied in
 * windows) is the longest silent stretch of a layer -- ~100 s on a 30K-token
 * prompt -- and prints nothing at all without this. Time-gated, so it adds
 * at most one line per progress_min_ms() however many experts there are. */
/* Wall clock of the last MoE heartbeat. Deliberately SEPARATE from
 * g_progress_ms: the token heartbeat fires every ~5 s during a KDA layer's
 * attention sweep and would otherwise keep refreshing the shared stamp,
 * starving the MoE windows of a fast layer (they are shorter than the
 * throttle window then) -- which is exactly the 90 s of silence layers 1-3
 * showed on a 30K prompt while the token heartbeats beside them kept going. */
static double g_moe_progress_ms = -1e18;

/* The prompt sweep the running prefill is inside, in tokens: how many were
 * already done when this chunk started, how many this chunk covers, and how
 * long the whole prompt is. moe_batch_progress needs them because it is the
 * only callback a chunked prefill gets from inside a chunk, and a chunk can be
 * the entire prompt: without them /v1/metrics reported prefill_tokens_done 0
 * and prefill_layer 0 for the whole of a 30K-token sweep. */
static int64_t g_pf_base_tokens = 0;
static int64_t g_pf_chunk_tokens = 0;
static int64_t g_pf_all_tokens = 0;

static void moe_batch_progress(void *user, int64_t layer, int64_t done,
                               int64_t total) {
    cwick_engine *eng = (cwick_engine *)user;
    if (!eng) return;
    const int64_t nlayer = eng->spec.num_hidden_layers > 0
                               ? eng->spec.num_hidden_layers : 1;
    /* Publish the position before the throttle: a gauge that only moves when
     * a line is printed is a gauge that lies about a slow layer. The layer is
     * one-based here and the token count is interpolated inside the chunk, so
     * both advance every time the sweep does. */
    if (g_pf_chunk_tokens > 0) {
        int64_t within = (int64_t)((double)(layer + 1) *
                                   (double)g_pf_chunk_tokens / (double)nlayer);
        if (within > g_pf_chunk_tokens) within = g_pf_chunk_tokens;
        cwick_metrics_set_prefill(g_pf_base_tokens + within, g_pf_all_tokens,
                                  layer + 1, nlayer);
    }
    double tnow = now_ms();
    if (tnow - g_moe_progress_ms < progress_min_ms()) return;
    g_moe_progress_ms = tnow;
    char tail[320];
    progress_mem_tail(eng, tail, sizeof(tail));
    CWICK_VERB(eng_tag(eng), "prefill-moe %lld/%lld (layer %lld/%lld) — %s",
              (long long)done, (long long)total,
              (long long)(layer + 1),
              (long long)eng->spec.num_hidden_layers, tail);
}

/* The engine is a long-running single-threaded decode loop; Ctrl+C has to be
 * noticed *between* tokens, not after the whole answer is finished, or the
 * flag would only be read hours later. */
void cwick_engine_set_stop(cwick_engine *eng, volatile int *stop) {
    if (!eng) return;
    eng->stop = stop;
    if (eng->ds) eng->ds->ctx.stop = stop;
    if (eng->mm) eng->mm->ctx.stop = stop;
    if (eng->q35) eng->q35->ctx.stop = stop;
}

void cwick_engine_set_abort_poll(cwick_engine *eng, bool (*poll)(void *),
                                void *user) {
    if (!eng) return;
    eng->abort_poll = poll;
    eng->abort_poll_user = user;
    if (eng->ds) {
        eng->ds->ctx.abort_poll = poll;
        eng->ds->ctx.abort_poll_user = user;
    }
    if (eng->mm) {
        eng->mm->ctx.abort_poll = poll;
        eng->mm->ctx.abort_poll_user = user;
    }
    if (eng->q35) {
        eng->q35->ctx.abort_poll = poll;
        eng->q35->ctx.abort_poll_user = user;
    }
}

void cwick_engine_set_verbose(cwick_engine *eng, bool verbose) {
    if (eng) eng->verbose = verbose;
    if (eng && eng->ds) eng->ds->ctx.verbose = verbose;
    if (eng && eng->mm) eng->mm->ctx.verbose = verbose;
    if (eng && eng->q35) eng->q35->ctx.verbose = verbose;
}

static bool engine_stopped(const cwick_engine *eng) {
    if (!eng) return false;
    if (eng->stop && *eng->stop) return true;
    if (eng->abort_poll && eng->abort_poll(eng->abort_poll_user)) return true;
    return false;
}

/* What to tell the caller about an interrupt. "interrupted" is the shutdown
 * case (Ctrl+C: the client may retry elsewhere); anything the abort predicate
 * raised means the client itself went away, and there is nobody left to tell.
 * Kept as a literal so the existing error-code mapping still matches it. */
static const char *engine_interrupt_message(const cwick_engine *eng) {
    if (eng && !(eng->stop && *eng->stop)) return "client disconnected";
    return "interrupted";
}

/* ------------------------------------------------------------------ */
/* Residual-anchor combination                                     */
/* ------------------------------------------------------------------ */
static void residual_combine(float *out, const float *prefix,
                             const float *anchors, int64_t n_anchors,
                             const float *score, int64_t H, float eps) {
    int64_t n = n_anchors + 1;
    float *vals = cwick_xmalloc((size_t)n * H * sizeof(float));
    float *keys = cwick_xmalloc((size_t)n * H * sizeof(float));
    float *s = cwick_xmalloc((size_t)n * sizeof(float));
    for (int64_t i = 0; i < n_anchors; i++)
        memcpy(vals + i * H, anchors + i * H, (size_t)H * sizeof(float));
    memcpy(vals + n_anchors * H, prefix, (size_t)H * sizeof(float));
    for (int64_t row = 0; row < n; row++) {
        const float *v = vals + row * H;
        double ss = 0.0;
        for (int64_t j = 0; j < H; j++) ss += (double)v[j] * v[j];
        float inv = 1.0f / (float)sqrt(ss / (double)H + (double)eps);
        for (int64_t j = 0; j < H; j++) keys[row * H + j] = v[j] * inv;
    }
    for (int64_t row = 0; row < n; row++) {
        const float *k = keys + row * H;
        float acc = 0.0f;
        for (int64_t j = 0; j < H; j++) acc += k[j] * score[j];
        s[row] = acc;
    }
    cwick_softmax(s, n);
    for (int64_t j = 0; j < H; j++) {
        float acc = 0.0f;
        for (int64_t row = 0; row < n; row++) acc += s[row] * vals[row * H + j];
        out[j] = acc;
    }
    free(s);
    free(keys);
    free(vals);
}

/* score vector = res_norm * res_proj (both BF16 rows) */
static void score_weight(float *out, const cwick_tensor *norm,
                         const cwick_tensor *proj, int64_t n) {
    const uint16_t *a = (const uint16_t *)norm->data;
    const uint16_t *b = (const uint16_t *)proj->data;
    for (int64_t i = 0; i < n; i++)
        out[i] = cwick_bf16_to_f32(a[i]) * cwick_bf16_to_f32(b[i]);
}

/* ------------------------------------------------------------------ */
/* mHC: manifold-constrained hyper-connection                          */
/*                                                                     */
/* Reference: vLLM model_executor/kernels/mhc/torch.py (mhc_pre_torch  */
/* / mhc_post_torch) as called from models/glm5next/common/model.py.   */
/*                                                                     */
/* pre:  mixes  = (x @ fn^T) * rsqrt(mean(x^2) + rms_eps)              */
/*       pre_mix  = sigmoid(mixes[:n]      * scale[0] + base[:n]) + eps */
/*       post_mix = sigmoid(mixes[n:2n]    * scale[1] + base[n:2n]) * P */
/*       comb     = sinkhorn(softmax(mixes[2n:]) + eps)                */
/*       layer_in = rms_norm(sum_i pre_mix[i] * x[i])                  */
/* post: out[j]  = post_mix[j] * y + sum_i comb[i][j] * x[i]           */
/* ------------------------------------------------------------------ */
#define MHC_POST_MULT 2.0f  /* every real call site uses 2.0 (vLLM) */

static void hc_expand(const float *x, float *res, int64_t n, int64_t H) {
    for (int64_t i = 0; i < n; i++)
        memcpy(res + i * H, x, (size_t)H * sizeof(float));
}

static void hc_contract(const float *res, float *out, int64_t n, int64_t H) {
    for (int64_t j = 0; j < H; j++) {
        double acc = 0.0;
        for (int64_t i = 0; i < n; i++) acc += res[i * H + j];
        out[j] = (float)(acc / (double)n);
    }
}

static void mhc_pre(const float *residual, int64_t n, int64_t H,
                    const cwick_tensor *fn, const cwick_tensor *base,
                    const cwick_tensor *scale, const cwick_tensor *norm_w,
                    float rms_eps, float hc_eps, int sinkhorn,
                    float *post, float *comb, float *layer_in,
                    float *mixes, float *pre) {
    const int64_t d = n * H;
    const int64_t mix = 2 * n + n * n;
    /* HF exports these small vectors as F32, while converted fixtures may
     * store them as BF16. Read through the dtype-aware accessor rather than
     * casting four bytes per value unconditionally. */
    /* mixes = (x @ fn^T) * rsqrt(mean(x^2) + rms_eps) */
    cwick_linear(mixes, residual, 1, d, mix, fn);
    double ss = 0.0;
    for (int64_t i = 0; i < d; i++) ss += (double)residual[i] * residual[i];
    float inv = 1.0f / sqrtf((float)(ss / (double)d) + rms_eps);
    for (int64_t i = 0; i < mix; i++) mixes[i] *= inv;

    /* `base` and `scale` are tensors, not float arrays: read them through the
     * dtype-aware accessor so an F32 export and a BF16 fixture both work. */
    for (int64_t i = 0; i < n; i++)
        pre[i] = 1.0f / (1.0f + expf(-(mixes[i] * cwick_tensor_value(scale, 0)
                                      + cwick_tensor_value(base, i)))) + hc_eps;
    for (int64_t i = 0; i < n; i++)
        post[i] = (1.0f / (1.0f + expf(-(mixes[n + i] * cwick_tensor_value(scale, 1)
                                        + cwick_tensor_value(base, n + i)))))
                  * MHC_POST_MULT;

    /* comb: softmax over the last dim, then Sinkhorn row/column sweeps. */
    for (int64_t i = 0; i < n; i++) {
        float mx = -INFINITY;
        for (int64_t j = 0; j < n; j++) {
            float v = mixes[2 * n + i * n + j] * cwick_tensor_value(scale, 2)
                      + cwick_tensor_value(base, 2 * n + i * n + j);
            comb[i * n + j] = v;
            if (v > mx) mx = v;
        }
        double sum = 0.0;
        for (int64_t j = 0; j < n; j++) {
            float e = expf(comb[i * n + j] - mx);
            comb[i * n + j] = e;
            sum += e;
        }
        float isum = 1.0f / (float)sum;
        for (int64_t j = 0; j < n; j++) comb[i * n + j] = comb[i * n + j] * isum + hc_eps;
    }
    for (int64_t it = 0; it < sinkhorn; it++) {
        /* column normalisation (dim=-2) */
        for (int64_t j = 0; j < n; j++) {
            double cs = 0.0;
            for (int64_t i = 0; i < n; i++) cs += comb[i * n + j];
            float ic = 1.0f / (float)(cs + (double)hc_eps);
            for (int64_t i = 0; i < n; i++) comb[i * n + j] *= ic;
        }
        if (it == sinkhorn - 1) break;
        /* row normalisation (dim=-1) */
        for (int64_t i = 0; i < n; i++) {
            double rs = 0.0;
            for (int64_t j = 0; j < n; j++) rs += comb[i * n + j];
            float ir = 1.0f / (float)(rs + (double)hc_eps);
            for (int64_t j = 0; j < n; j++) comb[i * n + j] *= ir;
        }
    }

    /* layer_in = rms_norm(sum_i pre[i] * residual[i]) */
    for (int64_t j = 0; j < H; j++) {
        double acc = 0.0;
        for (int64_t i = 0; i < n; i++) acc += (double)pre[i] * residual[i * H + j];
        layer_in[j] = (float)acc;
    }
    {
        const uint16_t *wn = (const uint16_t *)norm_w->data;
        double s2 = 0.0;
        for (int64_t j = 0; j < H; j++) s2 += (double)layer_in[j] * layer_in[j];
        float r = 1.0f / sqrtf((float)(s2 / (double)H) + rms_eps);
        for (int64_t j = 0; j < H; j++)
            layer_in[j] = layer_in[j] * r * cwick_bf16_to_f32(wn[j]);
    }
}

/* out[j][k] = post[j] * y[k] + sum_i comb[i][j] * residual[i][k] */
static void mhc_post(float *out, const float *y, const float *residual,
                     const float *post, const float *comb, int64_t n, int64_t H) {
    for (int64_t j = 0; j < n; j++) {
        float *orow = out + j * H;
        for (int64_t k = 0; k < H; k++) {
            double acc = (double)post[j] * y[k];
            for (int64_t i = 0; i < n; i++)
                acc += (double)comb[i * n + j] * residual[i * H + k];
            orow[k] = (float)acc;
        }
    }
}

/* Optional per-stage numeric probe: a NaN usually enters in one sub-layer and
 * then silently poisons everything downstream, so the first step prints a
 * min/max/mean signature for every stage. Enable with CWICK_TRACE=1. */
static bool trace_on(void) {
    static int cached = -1;
    if (cached < 0) cached = cwick_env_true("CWICK_TRACE") ? 1 : 0;
    return cached == 1;
}
static void probe(const char *tag, const float *x, int64_t n) {
    if (!trace_on()) return;
    double mn = INFINITY, mx = -INFINITY, sum = 0.0;
    int64_t nan = 0;
    for (int64_t i = 0; i < n; i++) {
        double v = x[i];
        if (v != v) { nan++; continue; }
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
    }
    fprintf(stderr, "  trace %-26s n=%-7lld min=%-11.5g max=%-11.5g mean=%-11.5g nan=%lld\n",
            tag, (long long)n, mn, mx, n ? sum / (double)n : 0.0, (long long)nan);
}

static void prefetch_next_layer(const cwick_model_spec *spec,
                                const cwick_spine *spine, int64_t L,
                                const float *post);

/* ------------------------------------------------------------------ */
/* Forward one token — released GLM-5.3-Flash (mHC + KDA/DSA + MoE)    */
/*                                                                     */
/* Reference: vLLM models/glm5next/common/model.py Glm5NextDecoderLayer.
 * Every base layer runs:                                              */
/*   (fused) residual = mhc_post(prev_sublayer_out, residual, post, comb)
 *   post, comb, attn_in = mhc_pre(residual, hc_attn_*, input_layernorm)
 *   attn_out = attention(attn_in)
 *   residual = mhc_post(attn_out, residual, post, comb)
 *   post, comb, mlp_in = mhc_pre(residual, hc_ffn_*, post_attention_ln)
 *   mlp_out = mlp(mlp_in)
 * The last layer materialises its final mhc_post and contracts the n
 * residual streams to one; every other layer defers its hc_post to the
 * next layer's fused pre.                                             */
/* ------------------------------------------------------------------ */
static cwick_status forward_token_mhc(cwick_engine *eng, uint32_t token,
                                     float *logits, cwick_error *err) {
    const cwick_model_spec *spec = &eng->spec;
    const cwick_spine *spine = &eng->spine;
    const int64_t H = spec->hidden_size;
    const int64_t L = spec->num_hidden_layers;
    const int64_t n = spec->hc_mult > 0 ? spec->hc_mult : 1;
    const int64_t pos = eng->pos;
    char name[192];

    if (pos >= eng->capacity) {
        cwick_error_set(err, CWICK_ERR_STATE,
                       "context capacity %lld exhausted at position %lld",
                       eng->capacity, pos);
        return CWICK_ERR_STATE;
    }

    int64_t ei = cwick_spine_find(spine, "embed_tokens.weight");
    if (ei == CWICK_TENSOR_MISSING) {
        cwick_error_set(err, CWICK_ERR_STATE, "missing embedding table");
        return CWICK_ERR_STATE;
    }
    const cwick_tensor *emb = &spine->tensors[(size_t)ei];
    if (token >= (uint32_t)emb->rows) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "token %u out of embedding range", token);
        return CWICK_ERR_ARGUMENT;
    }

    const int64_t mix = 2 * n + n * n;
    float *residual = cwick_xmalloc((size_t)(n * H) * sizeof(float));
    float *scratch = cwick_xmalloc((size_t)(n * H) * sizeof(float));
    float *post = cwick_xmalloc((size_t)n * sizeof(float));
    float *comb = cwick_xmalloc((size_t)(n * n) * sizeof(float));
    float *mixes = cwick_xmalloc((size_t)mix * sizeof(float));
    float *pre = cwick_xmalloc((size_t)n * sizeof(float));
    float *sub_in = cwick_xmalloc((size_t)H * sizeof(float));
    float *sub_out = cwick_xmalloc((size_t)H * sizeof(float));
    float *prev_out = cwick_xmalloc((size_t)H * sizeof(float));
    float *hidden = cwick_xmalloc((size_t)H * sizeof(float));

    {
        const uint16_t *erow = (const uint16_t *)emb->data + (size_t)token * H;
        for (int64_t i = 0; i < H; i++) hidden[i] = cwick_bf16_to_f32(erow[i]);
    }
    /* layer 0 has no incoming state: expand the embedding to n streams */
    hc_expand(hidden, residual, n, H);
    probe("embed", hidden, H);

#define M_FIND(suffix, out) do { \
        snprintf(name, sizeof(name), "layers.%lld.%s", l, suffix); \
        int64_t _ix = cwick_spine_find(spine, name); \
        if (_ix == CWICK_TENSOR_MISSING) { \
            cwick_error_set(err, CWICK_ERR_STATE, "missing weight %s", name); \
            goto fail; \
        } \
        out = &spine->tensors[(size_t)_ix]; \
    } while (0)

    for (int64_t l = 0; l < L; l++) {
        if (engine_stopped(eng)) {
            cwick_error_set(err, CWICK_ERR_STATE, "%s", engine_interrupt_message(eng));
            goto fail;
        }
        const cwick_tensor *fn, *bs, *sc, *nw;

        /* ---- deferred hc_post from the previous sublayer, then pre ---- */
        if (l > 0) {
            mhc_post(scratch, prev_out, residual, post, comb, n, H);
            memcpy(residual, scratch, (size_t)(n * H) * sizeof(float));
        }
        M_FIND("hc_attn_fn", fn);
        M_FIND("hc_attn_base", bs);
        M_FIND("hc_attn_scale", sc);
        M_FIND("input_layernorm.weight", nw);
        mhc_pre(residual, n, H, fn, bs, sc, nw, spec->rms_norm_eps_f,
                spec->hc_eps, spec->hc_sinkhorn_iters,
                post, comb, sub_in, mixes, pre);
        if (l < 2 || 1) { {char _t[64];snprintf(_t,sizeof(_t),"L%lld hc_pre_attn mixes",l);probe(_t,mixes,mix);} {char _t[64];snprintf(_t,sizeof(_t),"L%lld attn_in",l);probe(_t,sub_in,H);} }

        /* ---- attention ---- */
        uint64_t _attn_t0 = cwick_prof_ns();
        if (spec->layers[l] == CWICK_LAYER_DSA || spec->layers[l] == CWICK_LAYER_MLA) {
            const cwick_tensor *qa, *qb, *kva, *kvb, *o, *qa_n, *kva_n;
            M_FIND("self_attn.q_a_proj.weight", qa);
            M_FIND("self_attn.q_b_proj.weight", qb);
            M_FIND("self_attn.kv_a_proj_with_mqa.weight", kva);
            M_FIND("self_attn.kv_b_proj.weight", kvb);
            M_FIND("self_attn.o_proj.weight", o);
            M_FIND("self_attn.q_a_layernorm.weight", qa_n);
            M_FIND("self_attn.kv_a_layernorm.weight", kva_n);
            int64_t slot = eng->kv_slot[l];
            if (slot < 0 || !eng->keys) {
                cwick_error_set(err, CWICK_ERR_STATE,
                               "layer %lld has no KV cache slot", l);
                goto fail;
            }
            uint64_t _mla_t0 = cwick_prof_ns();
            cwick_attention_mla_decode(spec, qa, qb, kva, kvb, NULL, o, qa_n, kva_n,
                                      eng->keys, eng->values, eng->capacity,
                                      pos, slot,
                                      eng->idx_w ? &eng->idx_w[slot] : NULL,
                                      eng->idx_state ? &eng->idx_state[slot] : NULL,
                                      sub_in, sub_out, err);
            extern uint64_t g_attn_mla_ns;
            g_attn_mla_ns += cwick_prof_ns() - _mla_t0;
        } else {
            const cwick_tensor *qp, *kp, *vp, *qc, *kc, *vc, *al, *db, *on;
            const cwick_tensor *fa, *fb, *bp, *ga, *gb, *op;
            M_FIND("self_attn.q_proj.weight", qp);
            M_FIND("self_attn.k_proj.weight", kp);
            M_FIND("self_attn.v_proj.weight", vp);
            M_FIND("self_attn.q_conv1d.weight", qc);
            M_FIND("self_attn.k_conv1d.weight", kc);
            M_FIND("self_attn.v_conv1d.weight", vc);
            M_FIND("self_attn.A_log", al);
            M_FIND("self_attn.dt_bias", db);
            M_FIND("self_attn.o_norm.weight", on);
            M_FIND("self_attn.f_a_proj.weight", fa);
            M_FIND("self_attn.f_b_proj.weight", fb);
            M_FIND("self_attn.b_proj.weight", bp);
            M_FIND("self_attn.g_a_proj.weight", ga);
            M_FIND("self_attn.g_b_proj.weight", gb);
            M_FIND("self_attn.o_proj.weight", op);
            cwick_attention_kda_decode(spec, qp, kp, vp, qc, kc, vc, al, db, on,
                                      fa, fb, bp, ga, gb, op, eng->kda_rec,
                                      eng->kda_conv, l, sub_in, sub_out, err);
        }
        cwick_prof_add(CWICK_PH_ATTENTION, cwick_prof_ns() - _attn_t0);

        if (l < 2 || 1) {char _t[64];snprintf(_t,sizeof(_t),"L%lld attn_out",l);probe(_t,sub_out,H);}

        /* ---- hc_post(attention) + hc_pre(MLP) ---- */
        mhc_post(scratch, sub_out, residual, post, comb, n, H);
        memcpy(residual, scratch, (size_t)(n * H) * sizeof(float));
        M_FIND("hc_ffn_fn", fn);
        M_FIND("hc_ffn_base", bs);
        M_FIND("hc_ffn_scale", sc);
        M_FIND("post_attention_layernorm.weight", nw);
        mhc_pre(residual, n, H, fn, bs, sc, nw, spec->rms_norm_eps_f,
                spec->hc_eps, spec->hc_sinkhorn_iters,
                post, comb, sub_in, mixes, pre);
        if (l < 2 || 1) {char _t[64];snprintf(_t,sizeof(_t),"L%lld mlp_in",l);probe(_t,sub_in,H);}

        /* ---- MLP ---- */
        if (spec->mlp_dense && spec->mlp_dense[l])
            cwick_dense_mlp(spec, spine, l, sub_in, sub_out, err);
        else
            cwick_moe_forward(spec, spine, l, sub_in, sub_out, err);
        if (err && err->status != CWICK_OK) goto fail;
        if (l < 2 || 1) { {char _t[64];snprintf(_t,sizeof(_t),"L%lld mlp_out",l);probe(_t,sub_out,H);} {char _t[64];snprintf(_t,sizeof(_t),"L%lld residual",l);probe(_t,residual,n*H);} }
        memcpy(prev_out, sub_out, (size_t)H * sizeof(float));
        /* The predictor is advisory; authoritative routing still happens in
         * the next layer, so this only overlaps SSD demand reads. */
        if (cwick_env_true("CWICK_PREFETCH"))
            prefetch_next_layer(spec, spine, l, prev_out);
    }
#undef M_FIND

    /* ---- tail: last hc_post, contract, final norm, lm_head ---- */
    mhc_post(scratch, prev_out, residual, post, comb, n, H);
    probe("final streams", scratch, n * H);
    hc_contract(scratch, hidden, n, H);
    probe("contracted", hidden, H);
    {
        int64_t fi = cwick_spine_find(spine, "model.norm.weight");
        int64_t li = cwick_spine_find(spine, "lm_head.weight");
        if (fi == CWICK_TENSOR_MISSING || li == CWICK_TENSOR_MISSING) {
            cwick_error_set(err, CWICK_ERR_STATE, "missing final norm or lm_head");
            goto fail;
        }
        const uint16_t *wn = (const uint16_t *)spine->tensors[(size_t)fi].data;
        cwick_rms_norm(sub_in, hidden, H, NULL, spec->rms_norm_eps_f);
        for (int64_t i = 0; i < H; i++) sub_in[i] *= cwick_bf16_to_f32(wn[i]);
        cwick_linear(logits, sub_in, 1, H, spec->vocab_size,
                    &spine->tensors[(size_t)li]);
    }
    /* h_t is the MTP block's second input, and it is only valid until the
     * next forward overwrites it -- so the draft has to be taken from here,
     * on the step after this one, not from a re-derivation. */
    if (eng->mtp_on) memcpy(eng->mtp_h, hidden, (size_t)H * sizeof(float));

    free(hidden);
    free(prev_out);
    free(sub_out);
    free(sub_in);
    free(pre);
    free(mixes);
    free(comb);
    free(post);
    free(scratch);
    free(residual);
    return CWICK_OK;

fail:
    free(hidden);
    free(prev_out);
    free(sub_out);
    free(sub_in);
    free(pre);
    free(mixes);
    free(comb);
    free(post);
    free(scratch);
    free(residual);
    return CWICK_ERR_STATE;
}

/* ------------------------------------------------------------------ */
/* MTP draft: one extra block that guesses token t+2                    */
/* ------------------------------------------------------------------ */
/* A decode step's cost is set by the routed experts it pulls: ~8.06 GiB
 * of weights for one token. Two tokens in one batch name 13.0 experts per
 * layer rather than 16 (measured over a routing trace), so the second
 * token rides along for 0.63x of a step's expert bytes instead of 1.0x --
 * that is the whole economic case for speculating, and it is why the
 * drafter only has to be *cheap*, not good.
 *
 * GLM-5.3-Flash ships exactly this drafter: layer `num_hidden_layers` (45,
 * one past the depth-45 schedule) is a full DSA block plus the fusion that
 * makes it an MTP head --
 *
 *     h' = eh_proj( [ enorm(emb(x_{t+1})) , hnorm(h_t) ] )
 *     z  = DSA-block(h')                         at position t+1
 *     draft = lm_head( shared_head.norm(z) )
 *
 * and it is trained to predict x_{t+2}. The concatenation order is the one
 * the tensor names imply (enorm then hnorm); CWICK_MTP_SWAP=1 flips it,
 * because it is the one thing about this block the checkpoint cannot tell
 * us and a wrong guess would show up as a low acceptance rate rather than
 * as anything diagnosable.
 *
 * The block owns a KV row per drafted position. Its rows for the *prompt*
 * were never computed -- the prompt is forwarded by layers 0..44, and
 * filling layer 45 for it would cost a whole extra layer of prefill -- so
 * the draft attends over zeros there. That is not free information loss:
 * h_t already carries the entire context, which is why the fusion is a
 * concatenation and not an attention lookup. If acceptance turns out too
 * low, back-filling this cache is the next thing to try.
 *
 * Nothing here changes what the model emits; a wrong draft is discarded. */
static cwick_status mtp_draft(cwick_engine *eng, uint32_t next_token,
                              uint32_t *draft, cwick_error *err) {
    const cwick_model_spec *spec = &eng->spec;
    const cwick_spine *spine = &eng->spine;
    const int64_t H = spec->hidden_size;
    const int64_t L = spec->num_hidden_layers;
    const float eps = spec->rms_norm_eps_f;
    char nb[192];

#define MTP_FIND(var, suffix) do { \
        snprintf(nb, sizeof(nb), "layers.%lld." suffix, (long long)L); \
        var = cwick_spine_find(spine, nb); \
        if (var == CWICK_TENSOR_MISSING) { \
            cwick_error_set(err, CWICK_ERR_STATE, "MTP: missing %s", nb); \
            return CWICK_ERR_STATE; \
        } \
    } while (0)
    int64_t i_eh, i_en, i_hn, i_shn, i_in, i_post;
    int64_t i_qa, i_qb, i_kva, i_kvb, i_o, i_qan, i_kvan;
    MTP_FIND(i_eh, "eh_proj.weight");
    MTP_FIND(i_en, "enorm.weight");
    MTP_FIND(i_hn, "hnorm.weight");
    MTP_FIND(i_shn, "shared_head.norm.weight");
    MTP_FIND(i_in, "input_layernorm.weight");
    MTP_FIND(i_post, "post_attention_layernorm.weight");
    MTP_FIND(i_qa, "self_attn.q_a_proj.weight");
    MTP_FIND(i_qb, "self_attn.q_b_proj.weight");
    MTP_FIND(i_kva, "self_attn.kv_a_proj_with_mqa.weight");
    MTP_FIND(i_kvb, "self_attn.kv_b_proj.weight");
    MTP_FIND(i_o, "self_attn.o_proj.weight");
    MTP_FIND(i_qan, "self_attn.q_a_layernorm.weight");
    MTP_FIND(i_kvan, "self_attn.kv_a_layernorm.weight");
#undef MTP_FIND
    const int64_t i_emb = cwick_spine_find(spine, "embed_tokens.weight");
    const int64_t i_lm = cwick_spine_find(spine, "lm_head.weight");
    if (i_emb == CWICK_TENSOR_MISSING || i_lm == CWICK_TENSOR_MISSING) {
        cwick_error_set(err, CWICK_ERR_STATE, "MTP: missing embedding or head");
        return CWICK_ERR_STATE;
    }

    float *cat = eng->mtp_work;            /* [2H] fused input */
    float *tmp = eng->mtp_work + 2 * H;    /* [H]  norm scratch */
    float *res = eng->mtp_work + 3 * H;    /* [H]  residual stream */
    float *xa  = eng->mtp_work + 4 * H;    /* [H]  block input */
    float *xb  = eng->mtp_work + 5 * H;    /* [H]  block output */

    const bool swap = cwick_env_true("CWICK_MTP_SWAP");
    const int64_t e_off = swap ? H : 0;
    const int64_t h_off = swap ? 0 : H;

    /* concat( enorm(emb(x_{t+1})) , hnorm(h_t) ) */
    const uint16_t *emb = (const uint16_t *)spine->tensors[(size_t)i_emb].data;
    for (int64_t i = 0; i < H; i++)
        tmp[i] = cwick_bf16_to_f32(emb[(size_t)next_token * (size_t)H + (size_t)i]);
    cwick_rms_norm(xa, tmp, H, NULL, eps);
    {
        const uint16_t *w = (const uint16_t *)spine->tensors[(size_t)i_en].data;
        for (int64_t i = 0; i < H; i++)
            cat[e_off + i] = xa[i] * cwick_bf16_to_f32(w[i]);
    }
    cwick_rms_norm(xa, eng->mtp_h, H, NULL, eps);
    {
        const uint16_t *w = (const uint16_t *)spine->tensors[(size_t)i_hn].data;
        for (int64_t i = 0; i < H; i++)
            cat[h_off + i] = xa[i] * cwick_bf16_to_f32(w[i]);
    }

    /* the block's residual stream starts at eh_proj(cat) */
    cwick_linear(res, cat, 1, 2 * H, H, &spine->tensors[(size_t)i_eh]);

    /* attention at the position the verify pass will occupy */
    cwick_rms_norm(xa, res, H, NULL, eps);
    {
        const uint16_t *w = (const uint16_t *)spine->tensors[(size_t)i_in].data;
        for (int64_t i = 0; i < H; i++) xa[i] *= cwick_bf16_to_f32(w[i]);
    }
    /* the block's own attention; it reports failure through `err` */
    cwick_attention_mla_decode(
        spec, &spine->tensors[(size_t)i_qa], &spine->tensors[(size_t)i_qb],
        &spine->tensors[(size_t)i_kva], &spine->tensors[(size_t)i_kvb], NULL,
        &spine->tensors[(size_t)i_o], &spine->tensors[(size_t)i_qan],
        &spine->tensors[(size_t)i_kvan], eng->mtp_keys, NULL, eng->capacity,
        eng->pos, 0, NULL, NULL, xa, xb, err);
    if (err && err->status != CWICK_OK) return err->status;
    for (int64_t i = 0; i < H; i++) res[i] += xb[i];

    /* the sparse MLP */
    cwick_rms_norm(xa, res, H, NULL, eps);
    {
        const uint16_t *w = (const uint16_t *)spine->tensors[(size_t)i_post].data;
        for (int64_t i = 0; i < H; i++) xa[i] *= cwick_bf16_to_f32(w[i]);
    }
    cwick_moe_forward(spec, spine, L, xa, xb, err);
    if (err && err->status != CWICK_OK) return err->status;
    for (int64_t i = 0; i < H; i++) res[i] += xb[i];

    /* shared_head.norm then the tied lm_head */
    cwick_rms_norm(xa, res, H, NULL, eps);
    {
        const uint16_t *w = (const uint16_t *)spine->tensors[(size_t)i_shn].data;
        for (int64_t i = 0; i < H; i++) xa[i] *= cwick_bf16_to_f32(w[i]);
    }
    cwick_linear(eng->mtp_logits, xa, 1, H, spec->vocab_size,
                 &spine->tensors[(size_t)i_lm]);

    uint32_t best = 0;
    float bv = eng->mtp_logits[0];
    for (int64_t i = 1; i < spec->vocab_size; i++) {
        if (eng->mtp_logits[i] > bv) { bv = eng->mtp_logits[i]; best = (uint32_t)i; }
    }
    *draft = best;
    return CWICK_OK;
}

/* ------------------------------------------------------------------ */
/* Batched decode: one expert fetch + one GEMV group for R sequences   */
/* ------------------------------------------------------------------ */
/* Why this exists. A single-sequence decode step streams ~4.4 GiB of expert
 * weights off the SSD to produce ONE token, because every token picks its own
 * eight experts per layer. Two sequences picking in the same layer need, on
 * average, 14.7 distinct experts between them -- not 16 (measured: unrelated
 * sequences share 1.34 of 8 per layer) -- and the whole group is fetched once
 * and applied to every row that routed to it. The SSD reads therefore cost
 * ~0.9x per token instead of 1.0x, and the GEMV, which is bandwidth-bound on
 * the same matrices, costs ~1/R per token. Attention and the KDA/DSA state
 * stay per sequence and cannot be shared -- that part is simply run once per
 * row.
 *
 * The rows have to move through the layers together, so this mirrors
 * forward_token_mhc layer by layer: every row does its own mHC mixing and
 * attention, then the rows meet for one cwick_moe_forward_batch call, then
 * they separate again. Same arithmetic, same order, one shared fetch.
 *
 * CWICK_BATCH_DECODE sets how many sequences a group may hold (1 = the old
 * behaviour, and the default). */
/* Wall time spent inside cwick_forward_group during the last group request,
 * accumulated per step. Compared against the phase buckets it tells the
 * forward call apart from the group loop around it (prof.c prints it). */
double g_group_fg_ms;
/* Batch-MoE section timers (moe.c), split out of the uninstrumented "rest"
 * bucket so a group regression can be attributed. Reset at each decode
 * start alongside the profiler phase buckets. */
extern double g_batch_completion_ms;
extern double g_batch_router_ms;
extern double g_batch_shared_ms;

typedef struct {
    float *residual, *scratch, *post, *comb, *mixes, *pre;
    float *sub_in, *sub_out, *prev_out, *hidden;
    int row;                 /* group member index, -1 when unset (probe tag) */
} mhc_row;

static void mhc_row_alloc(mhc_row *r, const cwick_model_spec *spec) {
    const int64_t H = spec->hidden_size;
    const int64_t n = spec->hc_mult > 0 ? spec->hc_mult : 1;
    const int64_t mix = 2 * n + n * n;
    r->residual = cwick_xmalloc((size_t)(n * H) * sizeof(float));
    r->scratch  = cwick_xmalloc((size_t)(n * H) * sizeof(float));
    r->post     = cwick_xmalloc((size_t)n * sizeof(float));
    r->comb     = cwick_xmalloc((size_t)(n * n) * sizeof(float));
    r->mixes    = cwick_xmalloc((size_t)mix * sizeof(float));
    r->pre      = cwick_xmalloc((size_t)n * sizeof(float));
    r->sub_in   = cwick_xmalloc((size_t)H * sizeof(float));
    r->sub_out  = cwick_xmalloc((size_t)H * sizeof(float));
    r->prev_out = cwick_xmalloc((size_t)H * sizeof(float));
    r->hidden   = cwick_xmalloc((size_t)H * sizeof(float));
    r->row      = -1;
}

static void mhc_row_free(mhc_row *r) {
    free(r->hidden); free(r->prev_out); free(r->sub_out); free(r->sub_in);
    free(r->pre); free(r->mixes); free(r->comb); free(r->post);
    free(r->scratch); free(r->residual);
}

/* Row-tagged probe so a group member's stages can be diffed against the
 * single-sequence path's untagged ones (CWICK_TRACE=1). */
static void rowprobe(const mhc_row *r, const char *stage, const float *x,
                     int64_t n) {
    if (r->row < 0) { probe(stage, x, n); return; }
    char tag[96];
    snprintf(tag, sizeof(tag), "g%d %s", r->row, stage);
    probe(tag, x, n);
}

/* embedding -> expanded residual streams; identical to forward_token_mhc. */
static cwick_status mhc_row_start(cwick_engine *eng, mhc_row *r, uint32_t token,
                                 cwick_error *err) {
    const cwick_model_spec *spec = &eng->spec;
    const cwick_spine *spine = &eng->spine;
    const int64_t H = spec->hidden_size;
    const int64_t n = spec->hc_mult > 0 ? spec->hc_mult : 1;
    int64_t ei = cwick_spine_find(spine, "embed_tokens.weight");
    if (ei == CWICK_TENSOR_MISSING) {
        cwick_error_set(err, CWICK_ERR_STATE, "missing embedding table");
        return CWICK_ERR_STATE;
    }
    const cwick_tensor *emb = &spine->tensors[(size_t)ei];
    if (token >= (uint32_t)emb->rows) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "token %u out of embedding range",
                       token);
        return CWICK_ERR_ARGUMENT;
    }
    const uint16_t *erow = (const uint16_t *)emb->data + (size_t)token * H;
    for (int64_t i = 0; i < H; i++) r->hidden[i] = cwick_bf16_to_f32(erow[i]);
    hc_expand(r->hidden, r->residual, n, H);
    rowprobe(r, "embed", r->hidden, H);
    return CWICK_OK;
}

/* One layer up to the MLP input: the deferred hc_post from the previous
 * sublayer, both mhc_pre mixes, and this sequence's own attention. Leaves the
 * row's MLP input in r->sub_in -- which is exactly where the shared MoE group
 * picks it up. */
static cwick_status mhc_row_attn(cwick_engine *eng, mhc_row *r, int64_t l,
                                cwick_error *err) {
    const cwick_model_spec *spec = &eng->spec;
    const cwick_spine *spine = &eng->spine;
    const int64_t H = spec->hidden_size;
    const int64_t n = spec->hc_mult > 0 ? spec->hc_mult : 1;
    const int64_t pos = eng->pos;
    char name[192];

#define G_FIND(suffix, out) do { \
        snprintf(name, sizeof(name), "layers.%lld.%s", l, suffix); \
        int64_t _ix = cwick_spine_find(spine, name); \
        if (_ix == CWICK_TENSOR_MISSING) { \
            cwick_error_set(err, CWICK_ERR_STATE, "missing weight %s", name); \
            return CWICK_ERR_STATE; \
        } \
        out = &spine->tensors[(size_t)_ix]; \
    } while (0)

    if (l > 0) {
        mhc_post(r->scratch, r->prev_out, r->residual, r->post, r->comb, n, H);
        memcpy(r->residual, r->scratch, (size_t)(n * H) * sizeof(float));
    }
    const cwick_tensor *fn, *bs, *sc, *nw;
    G_FIND("hc_attn_fn", fn);
    G_FIND("hc_attn_base", bs);
    G_FIND("hc_attn_scale", sc);
    G_FIND("input_layernorm.weight", nw);
    mhc_pre(r->residual, n, H, fn, bs, sc, nw, spec->rms_norm_eps_f,
            spec->hc_eps, spec->hc_sinkhorn_iters,
            r->post, r->comb, r->sub_in, r->mixes, r->pre);
    {
        char _t[64];
        snprintf(_t, sizeof(_t), "L%lld attn_in", l);
        rowprobe(r, _t, r->sub_in, H);
    }

    uint64_t _attn_t0 = cwick_prof_ns();
    if (spec->layers[l] == CWICK_LAYER_DSA || spec->layers[l] == CWICK_LAYER_MLA) {
        const cwick_tensor *qa, *qb, *kva, *kvb, *o, *qa_n, *kva_n;
        G_FIND("self_attn.q_a_proj.weight", qa);
        G_FIND("self_attn.q_b_proj.weight", qb);
        G_FIND("self_attn.kv_a_proj_with_mqa.weight", kva);
        G_FIND("self_attn.kv_b_proj.weight", kvb);
        G_FIND("self_attn.o_proj.weight", o);
        G_FIND("self_attn.q_a_layernorm.weight", qa_n);
        G_FIND("self_attn.kv_a_layernorm.weight", kva_n);
        int64_t slot = eng->kv_slot[l];
        if (slot < 0 || !eng->keys) {
            cwick_error_set(err, CWICK_ERR_STATE, "layer %lld has no KV cache slot", l);
            return CWICK_ERR_STATE;
        }
        uint64_t _mla_t0 = cwick_prof_ns();
        cwick_attention_mla_decode(spec, qa, qb, kva, kvb, NULL, o, qa_n, kva_n,
                                  eng->keys, eng->values, eng->capacity,
                                  pos, slot,
                                  eng->idx_w ? &eng->idx_w[slot] : NULL,
                                  eng->idx_state ? &eng->idx_state[slot] : NULL,
                                  r->sub_in, r->sub_out, err);
        extern uint64_t g_attn_mla_ns;
        g_attn_mla_ns += cwick_prof_ns() - _mla_t0;
    } else {
        const cwick_tensor *qp, *kp, *vp, *qc, *kc, *vc, *al, *db, *on;
        const cwick_tensor *fa, *fb, *bp, *ga, *gb, *op;
        G_FIND("self_attn.q_proj.weight", qp);
        G_FIND("self_attn.k_proj.weight", kp);
        G_FIND("self_attn.v_proj.weight", vp);
        G_FIND("self_attn.q_conv1d.weight", qc);
        G_FIND("self_attn.k_conv1d.weight", kc);
        G_FIND("self_attn.v_conv1d.weight", vc);
        G_FIND("self_attn.A_log", al);
        G_FIND("self_attn.dt_bias", db);
        G_FIND("self_attn.o_norm.weight", on);
        G_FIND("self_attn.f_a_proj.weight", fa);
        G_FIND("self_attn.f_b_proj.weight", fb);
        G_FIND("self_attn.b_proj.weight", bp);
        G_FIND("self_attn.g_a_proj.weight", ga);
        G_FIND("self_attn.g_b_proj.weight", gb);
        G_FIND("self_attn.o_proj.weight", op);
        cwick_attention_kda_decode(spec, qp, kp, vp, qc, kc, vc, al, db, on,
                                  fa, fb, bp, ga, gb, op, eng->kda_rec,
                                  eng->kda_conv, l, r->sub_in, r->sub_out, err);
    }
    cwick_prof_add(CWICK_PH_ATTENTION, cwick_prof_ns() - _attn_t0);
    {
        char _t[64];
        snprintf(_t, sizeof(_t), "L%lld attn_out", l);
        rowprobe(r, _t, r->sub_out, H);
    }

    mhc_post(r->scratch, r->sub_out, r->residual, r->post, r->comb, n, H);
    memcpy(r->residual, r->scratch, (size_t)(n * H) * sizeof(float));
    G_FIND("hc_ffn_fn", fn);
    G_FIND("hc_ffn_base", bs);
    G_FIND("hc_ffn_scale", sc);
    G_FIND("post_attention_layernorm.weight", nw);
    mhc_pre(r->residual, n, H, fn, bs, sc, nw, spec->rms_norm_eps_f,
            spec->hc_eps, spec->hc_sinkhorn_iters,
            r->post, r->comb, r->sub_in, r->mixes, r->pre);
    {
        char _t[64];
        snprintf(_t, sizeof(_t), "L%lld mlp_in", l);
        rowprobe(r, _t, r->sub_in, H);
    }
#undef G_FIND
    return CWICK_OK;
}

/* final hc_post, contract, norm, lm_head -- per row, after the last layer. */
static cwick_status mhc_row_tail(cwick_engine *eng, mhc_row *r, float *logits,
                                cwick_error *err) {
    const cwick_model_spec *spec = &eng->spec;
    const cwick_spine *spine = &eng->spine;
    const int64_t H = spec->hidden_size;
    const int64_t n = spec->hc_mult > 0 ? spec->hc_mult : 1;
    mhc_post(r->scratch, r->prev_out, r->residual, r->post, r->comb, n, H);
    rowprobe(r, "final streams", r->scratch, n * H);
    hc_contract(r->scratch, r->hidden, n, H);
    rowprobe(r, "contracted", r->hidden, H);
    int64_t fi = cwick_spine_find(spine, "model.norm.weight");
    int64_t li = cwick_spine_find(spine, "lm_head.weight");
    if (fi == CWICK_TENSOR_MISSING || li == CWICK_TENSOR_MISSING) {
        cwick_error_set(err, CWICK_ERR_STATE, "missing final norm or lm_head");
        return CWICK_ERR_STATE;
    }
    const uint16_t *wn = (const uint16_t *)spine->tensors[(size_t)fi].data;
    cwick_rms_norm(r->sub_in, r->hidden, H, NULL, spec->rms_norm_eps_f);
    for (int64_t i = 0; i < H; i++) r->sub_in[i] *= cwick_bf16_to_f32(wn[i]);
    cwick_linear(logits, r->sub_in, 1, H, spec->vocab_size,
                &spine->tensors[(size_t)li]);
    return CWICK_OK;
}

/* Advance R sequences by one token each, sharing the expert work. */
cwick_status cwick_forward_group(cwick_engine **engs, const uint32_t *tokens,
                               int64_t R, float **logits, cwick_error *err) {
    if (!engs || !tokens || !logits || R <= 0) return CWICK_ERR_ARGUMENT;
    if (R == 1)
        return forward_token_mhc(engs[0], tokens[0], logits[0], err);
    /* The ds backend shares nothing to amortise: each engine owns its own
     * ds_model, so a "group" forward is R independent single steps. */
    if (engs[0]->mm) {
        for (int64_t i = 0; i < R; i++) {
            int32_t id = (int32_t)tokens[i];
            if (mm_model_forward(&engs[i]->mm->m, &id, 1, engs[i]->mm->pos,
                                 engs[i]->mm->logits, err) != 0)
                return CWICK_ERR_STATE;
            engs[i]->mm->pos++;
            memcpy(logits[i], engs[i]->mm->logits,
                   (size_t)engs[i]->spec.vocab_size * sizeof(float));
        }
        return CWICK_OK;
    }
    if (engs[0]->ds) {
        for (int64_t i = 0; i < R; i++) {
            int32_t id = (int32_t)tokens[i];
            if (ds_model_forward(&engs[i]->ds->m, &id, 1, engs[i]->ds->pos,
                                 engs[i]->ds->logits, err) != 0)
                return CWICK_ERR_STATE;
            engs[i]->ds->pos++;
            memcpy(logits[i], engs[i]->ds->logits,
                   (size_t)engs[i]->spec.vocab_size * sizeof(float));
        }
        return CWICK_OK;
    }

    const cwick_model_spec *spec = &engs[0]->spec;
    const cwick_spine *spine = &engs[0]->spine;
    const int64_t H = spec->hidden_size;
    const int64_t L = spec->num_hidden_layers;

    mhc_row *rows = cwick_xmalloc((size_t)R * sizeof(mhc_row));
    float *in_mat  = cwick_xmalloc((size_t)R * (size_t)H * sizeof(float));
    float *out_mat = cwick_xmalloc((size_t)R * (size_t)H * sizeof(float));
    for (int64_t i = 0; i < R; i++) {
        mhc_row_alloc(&rows[i], spec);
        rows[i].row = (int)i;
    }

    cwick_status st = CWICK_OK;
    {
        uint64_t _t0 = cwick_prof_ns();
        for (int64_t i = 0; i < R; i++) {
            if (engs[i]->pos >= engs[i]->capacity) {
                cwick_error_set(err, CWICK_ERR_STATE, "context capacity exhausted");
                st = CWICK_ERR_STATE; goto done;
            }
            st = mhc_row_start(engs[i], &rows[i], tokens[i], err);
            if (st != CWICK_OK) goto done;
        }
        cwick_prof_add(CWICK_PH_OTHER, cwick_prof_ns() - _t0);
    }

    for (int64_t l = 0; l < L; l++) {
        uint64_t _ta = cwick_prof_ns();
        for (int64_t i = 0; i < R; i++) {
            if (engine_stopped(engs[i])) {
                cwick_error_set(err, CWICK_ERR_STATE, "%s",
                               engine_interrupt_message(engs[i]));
                st = CWICK_ERR_STATE; goto done;
            }
            st = mhc_row_attn(engs[i], &rows[i], l, err);
            if (st != CWICK_OK) goto done;
            memcpy(in_mat + (size_t)i * H, rows[i].sub_in,
                   (size_t)H * sizeof(float));
        }
        cwick_prof_add(CWICK_PH_OTHER, cwick_prof_ns() - _ta);
        /* The shared part: one route, one fetch, one GEMV group for all rows.
         * The dense layers write sub_out themselves and must NOT be
         * overwritten from out_mat -- doing so unconditionally (as this did
         * once) wiped layers 0-2's MLP output with an uninitialised slab and
         * every group token after the first was sampled from a poisoned
         * residual. out_mat is only touched by the MoE branch. */
        if (spec->mlp_dense && spec->mlp_dense[l]) {
            for (int64_t i = 0; i < R; i++)
                cwick_dense_mlp(spec, spine, l, rows[i].sub_in, rows[i].sub_out,
                               err);
        } else {
            cwick_moe_forward_batch(spec, spine, l, in_mat, R, out_mat, err,
                                   NULL, NULL);
            if (err && err->status != CWICK_OK) {
                st = CWICK_ERR_STATE;
                goto done;
            }
            for (int64_t i = 0; i < R; i++)
                memcpy(rows[i].sub_out, out_mat + (size_t)i * H,
                       (size_t)H * sizeof(float));
        }
        if (err && err->status != CWICK_OK) { st = CWICK_ERR_STATE; goto done; }
        uint64_t _tc = cwick_prof_ns();
        for (int64_t i = 0; i < R; i++) {
            memcpy(rows[i].prev_out, rows[i].sub_out, (size_t)H * sizeof(float));
            {
                char _t[64];
                snprintf(_t, sizeof(_t), "L%lld mlp_out", l);
                rowprobe(&rows[i], _t, rows[i].sub_out, H);
            }
        }
        cwick_prof_add(CWICK_PH_OTHER, cwick_prof_ns() - _tc);
    }

    {
        uint64_t _tt = cwick_prof_ns();
        for (int64_t i = 0; i < R; i++) {
            st = mhc_row_tail(engs[i], &rows[i], logits[i], err);
            if (st != CWICK_OK) goto done;
        }
        cwick_prof_add(CWICK_PH_OTHER, cwick_prof_ns() - _tt);
    }

done:
    free(in_mat); free(out_mat);
    for (int64_t i = 0; i < R; i++) mhc_row_free(&rows[i]);
    free(rows);
    return st;
}

/* ------------------------------------------------------------------ */
/* Next-layer expert prefetch (opt-in, CWICK_PREFETCH=1)                 */
/* ------------------------------------------------------------------ */
/* Given layer `L`'s post-attention residual, predict which experts the
 * next layer's router will select and submit their F_NOCACHE reads to the
 * background reader while layer L's MoE computes. Reads that miss (router
 * chose differently) are simply demand-fetched; prediction never changes
 * the authoritative route.
 *
 * The prefetch gate adaptively suppresses reads on
 * layers whose measured recall is below threshold; scoring continues so the
 * layer recovers automatically. The hint is still recorded for scoring even
 * when reads are suppressed. */
static void prefetch_next_layer(const cwick_model_spec *spec,
                                const cwick_spine *spine, int64_t L,
                                const float *post) {
    if (!spine->stream) return;
    const int64_t NL = L + 1;
    if (NL >= spec->num_hidden_layers) return;
    if (spec->mlp_dense && spec->mlp_dense[NL]) return; /* next layer dense */

    const int64_t T = spec->num_experts_per_token;
    int64_t *ids = cwick_xmalloc((size_t)T * sizeof(int64_t));
    if (cwick_router_predict(spec, spine, NL, post, ids) != 0) {
        free(ids);
        return;
    }

    /* Record the hint for scoring even if reads are gated off. */
    cwick_stream_prefetch_hint_set(spine->stream, NL, ids, T);

    /* Gate: suppress speculative reads on layers whose best-predictor
     * EMA is below threshold (saving SSD bandwidth on the decode bottleneck).
     * Scoring continues regardless, so recovery needs no special path. */
    if (cwick_stream_prefetch_gated(spine->stream, NL) != 0) {
        free(ids);
        return;
    }

    cwick_stream_prefetch_begin(spine->stream);
    static const char *matrices[3] = {"gate_proj", "up_proj", "down_proj"};
    char n[224], sn[224];
    cwick_error err0;
    memset(&err0, 0, sizeof(err0));
    for (int64_t e = 0; e < T; e++) {
        for (int m = 0; m < 3; m++) {
            snprintf(n, sizeof(n),
                     "model.language_model.layers.%lld.mlp.experts.%lld.%s.weight",
                     NL, ids[e], matrices[m]);
            cwick_st_tensor t, st;
            memset(&st, 0, sizeof(st));
            if (cwick_safetensors_get(spine->saf, n, &t, &err0) != CWICK_OK) continue;
            size_t si; uint64_t off, soff = 0;
            if (cwick_stream_locate(spine->stream, &t, &si, &off, &err0) != CWICK_OK)
                continue;
            size_t sbytes = 0;
            if (t.dtype == CWICK_ST_F8_E4M3) {
                snprintf(sn, sizeof(sn), "%s_scale_inv", n);
                cwick_st_tensor st2;
                memset(&st2, 0, sizeof(st2));
                if (cwick_safetensors_get(spine->saf, sn, &st2, &err0) == CWICK_OK) {
                    size_t si2; uint64_t o2;
                    if (cwick_stream_locate(spine->stream, &st2, &si2, &o2, &err0) == CWICK_OK) {
                        sbytes = st2.data_bytes;
                        soff = o2;
                    }
                }
            }
            /* Skip a speculative read when the matrix is already pinned in
             * RAM. Without this, a steady-state decode re-reads its whole
             * ~8 GiB working set from the SSD every step even though the
             * cache holds it: the prefetch slot wins over the cache in
             * expert_view, so the pinned copy is never consulted and the
             * 6 GB/s SSD (not the CPU) sets the token rate. */
            if (cwick_stream_expert_resident(spine->stream, NL, ids[e], m))
                continue;
            cwick_stream_prefetch_submit(spine->stream, NL, ids[e], m,
                                        t.data_bytes, sbytes, si, off, soff);
        }
    }
    free(ids);
}

/* ------------------------------------------------------------------ */
/* Forward one token                                                   */
/* ------------------------------------------------------------------ */
static cwick_status forward_token(cwick_engine *eng, uint32_t token,
                                 float *logits, cwick_error *err) {
    const cwick_model_spec *spec = &eng->spec;
    const cwick_spine *spine = &eng->spine;
    const int64_t H = spec->hidden_size;
    const int64_t L = spec->num_hidden_layers;
    const int64_t res = spec->attn_res_block_size;
    const float eps = spec->rms_norm_eps_f;
    const int64_t pos = eng->pos;

    if (pos >= eng->capacity) {
        cwick_error_set(err, CWICK_ERR_STATE,
                       "context capacity %lld exhausted at position %lld",
                       eng->capacity, pos);
        return CWICK_ERR_STATE;
    }

    /* Released GLM-5.3-Flash: hyper-connection layers with KDA/DSA. The
     * legacy residual-anchor path below belongs to the converted (synthetic)
     * models and must never be run on these tensors. */
    if (spec->mhc && spec->hc_mult > 1)
        return forward_token_mhc(eng, token, logits, err);

    if (spec->quant == CWICK_QUANT_FP8_E4M3) {
        cwick_error_set(err, CWICK_ERR_UNSUPPORTED,
                       "GLM-5.3-Flash generation forward requires mHC "
                       "(hc_mult=%lld); the checkpoint reports mhc=%d",
                       spec->hc_mult, (int)spec->mhc);
        return CWICK_ERR_UNSUPPORTED;
    }

    /* The residual-anchor bank is per-token state (block residual): each
     * token independently accumulates its boundary-layer pre-attention rows
     * as it passes through the layers. */
    eng->n_anchors = 0;

    /* ---- embedding ---- */
    const cwick_tensor *emb = &spine->tensors[(size_t)cwick_spine_find(spine,
                               "embed_tokens.weight")];
    if (cwick_spine_find(spine, "embed_tokens.weight") == CWICK_TENSOR_MISSING ||
        token >= (uint32_t)emb->rows) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "token %u out of embedding range", token);
        return CWICK_ERR_ARGUMENT;
    }
    float *hidden = cwick_xmalloc((size_t)H * sizeof(float));
    const uint16_t *erow = (const uint16_t *)emb->data + (size_t)token * H;
    for (int64_t i = 0; i < H; i++) hidden[i] = cwick_bf16_to_f32(erow[i]);

    float *attn_in = cwick_xmalloc((size_t)H * sizeof(float));
    float *normed = cwick_xmalloc((size_t)H * sizeof(float));
    float *attn_out = cwick_xmalloc((size_t)H * sizeof(float));
    float *prefix = cwick_xmalloc((size_t)H * sizeof(float));
    float *mixed = cwick_xmalloc((size_t)H * sizeof(float));
    float *post = cwick_xmalloc((size_t)H * sizeof(float));
    float *mlp_out = cwick_xmalloc((size_t)H * sizeof(float));
    float *score = cwick_xmalloc((size_t)H * sizeof(float));
    char name[160];

    for (int64_t l = 0; l < L; l++) {
        if (engine_stopped(eng)) {
            cwick_error_set(err, CWICK_ERR_STATE, "%s", engine_interrupt_message(eng));
            goto fail;
        }

        /* ---- attention residual + anchor boundary ---- */
        const bool boundary = (l % res) == 0;
        const float *anchor_use = eng->n_anchors > 0 ? eng->anchors : NULL;
        if (anchor_use) {
            snprintf(name, sizeof(name),
                     "layers.%lld.self_attention_res_norm.weight", l);
            int64_t ni = cwick_spine_find(spine, name);
            snprintf(name, sizeof(name),
                     "layers.%lld.self_attention_res_proj.weight", l);
            int64_t pi = cwick_spine_find(spine, name);
            if (ni == CWICK_TENSOR_MISSING || pi == CWICK_TENSOR_MISSING) {
                cwick_error_set(err, CWICK_ERR_STATE, "missing attention residual weights");
                goto fail;
            }
            score_weight(score, &spine->tensors[(size_t)ni],
                         &spine->tensors[(size_t)pi], H);
            residual_combine(attn_in, hidden, eng->anchors, eng->n_anchors,
                             score, H, eps);
        } else {
            memcpy(attn_in, hidden, (size_t)H * sizeof(float));
        }
        bool has_prefix = true;
        memcpy(prefix, hidden, (size_t)H * sizeof(float));
        if (boundary) {
            if (eng->n_anchors >= eng->max_anchors) {
                cwick_error_set(err, CWICK_ERR_STATE, "anchor capacity exceeded");
                goto fail;
            }
            memcpy(eng->anchors + eng->n_anchors * H, hidden, (size_t)H * sizeof(float));
            eng->n_anchors++;
            has_prefix = false;
        }

        /* ---- input norm + attention ---- */
        snprintf(name, sizeof(name), "layers.%lld.input_layernorm.weight", l);
        int64_t in = cwick_spine_find(spine, name);
        if (in == CWICK_TENSOR_MISSING) {
            cwick_error_set(err, CWICK_ERR_STATE, "missing input layernorm %lld", l);
            goto fail;
        }
        {
            const uint16_t *wn = (const uint16_t *)spine->tensors[(size_t)in].data;
            cwick_rms_norm(normed, attn_in, H, NULL, eps);
            for (int64_t i = 0; i < H; i++) normed[i] *= cwick_bf16_to_f32(wn[i]);
        }

        if (spec->layers[l] == CWICK_LAYER_MLA) {
            cwick_tensor *qa, *qb, *kva, *kvb, *g, *o, *qa_n, *kva_n;
#define FIND(suffix, out) do { \
            snprintf(name, sizeof(name), "layers.%lld.self_attn.%s", l, suffix); \
            int64_t ix = cwick_spine_find(spine, name); \
            if (ix == CWICK_TENSOR_MISSING) { \
                cwick_error_set(err, CWICK_ERR_STATE, "missing MLA weight %s", name); \
                goto fail; \
            } \
            out = &spine->tensors[(size_t)ix]; \
} while (0)
            FIND("q_a_proj.weight", qa);
            FIND("q_b_proj.weight", qb);
            FIND("kv_a_proj_with_mqa.weight", kva);
            FIND("kv_b_proj.weight", kvb);
            FIND("g_proj.weight", g);
            FIND("o_proj.weight", o);
            FIND("q_a_layernorm.weight", qa_n);
            FIND("kv_a_layernorm.weight", kva_n);
#undef FIND
            cwick_attention_mla_decode(spec, qa, qb, kva, kvb, g, o, qa_n, kva_n,
                                      eng->keys, eng->values, eng->capacity,
                                      pos, eng->kv_slot[l],
                                      NULL, NULL,
                                      normed, attn_out, err);
        } else {
            cwick_tensor *qp, *kp, *vp, *qc, *kc, *vc, *al, *db, *on, *fa, *fb, *bp, *ga, *gb, *op;
#define FIND(suffix, out) do { \
            snprintf(name, sizeof(name), "layers.%lld.self_attn.%s", l, suffix); \
            int64_t ix = cwick_spine_find(spine, name); \
            if (ix == CWICK_TENSOR_MISSING) { \
                cwick_error_set(err, CWICK_ERR_STATE, "missing KDA weight %s", name); \
                goto fail; \
            } \
            out = &spine->tensors[(size_t)ix]; \
} while (0)
            FIND("q_proj.weight", qp);
            FIND("k_proj.weight", kp);
            FIND("v_proj.weight", vp);
            FIND("q_conv1d.weight", qc);
            FIND("k_conv1d.weight", kc);
            FIND("v_conv1d.weight", vc);
            FIND("A_log", al);
            FIND("dt_bias", db);
            FIND("o_norm.weight", on);
            FIND("f_a_proj.weight", fa);
            FIND("f_b_proj.weight", fb);
            FIND("b_proj.weight", bp);
            FIND("g_a_proj.weight", ga);
            FIND("g_b_proj.weight", gb);
            FIND("o_proj.weight", op);
#undef FIND
            cwick_attention_kda_decode(spec, qp, kp, vp, qc, kc, vc, al, db, on,
                                      fa, fb, bp, ga, gb, op, eng->kda_rec,
                                      eng->kda_conv, l, normed, attn_out, err);
        }

        /* ---- prefix = pre_attn + attn_out (or attn_out at boundary) ---- */
        if (has_prefix) {
            for (int64_t i = 0; i < H; i++) prefix[i] += attn_out[i];
        } else {
            memcpy(prefix, attn_out, (size_t)H * sizeof(float));
        }

        /* ---- MLP residual + post norm + MLP ---- */
        if (eng->n_anchors > 0) {
            snprintf(name, sizeof(name), "layers.%lld.mlp_res_norm.weight", l);
            int64_t ni = cwick_spine_find(spine, name);
            snprintf(name, sizeof(name), "layers.%lld.mlp_res_proj.weight", l);
            int64_t pi = cwick_spine_find(spine, name);
            if (ni == CWICK_TENSOR_MISSING || pi == CWICK_TENSOR_MISSING) {
                cwick_error_set(err, CWICK_ERR_STATE, "missing MLP residual weights");
                goto fail;
            }
            score_weight(score, &spine->tensors[(size_t)ni],
                         &spine->tensors[(size_t)pi], H);
            residual_combine(mixed, prefix, eng->anchors, eng->n_anchors,
                             score, H, eps);
        } else {
            memcpy(mixed, prefix, (size_t)H * sizeof(float));
        }
        snprintf(name, sizeof(name), "layers.%lld.post_attention_layernorm.weight", l);
        int64_t pn = cwick_spine_find(spine, name);
        if (pn == CWICK_TENSOR_MISSING) {
            cwick_error_set(err, CWICK_ERR_STATE, "missing post-attention norm %lld", l);
            goto fail;
        }
        {
            const uint16_t *wn = (const uint16_t *)spine->tensors[(size_t)pn].data;
            cwick_rms_norm(post, mixed, H, NULL, eps);
            for (int64_t i = 0; i < H; i++) post[i] *= cwick_bf16_to_f32(wn[i]);
        }
        /* Schedule next layer's expert reads while this layer's MoE
         * (and dense MLP) compute. Prediction never replaces the router. */
        if (cwick_env_true("CWICK_PREFETCH"))
            prefetch_next_layer(spec, spine, l, post);
        if (spec->mlp_dense && spec->mlp_dense[l]) {
            cwick_dense_mlp(spec, spine, l, post, mlp_out, err);
        } else {
            cwick_moe_forward(spec, spine, l, post, mlp_out, err);
        }
        for (int64_t i = 0; i < H; i++) hidden[i] = prefix[i] + mlp_out[i];
    }

    /* ---- tail: output residual, final norm, lm_head ---- */
    {
        snprintf(name, sizeof(name), "model.output_attn_res_norm.weight");
        int64_t ni = cwick_spine_find(spine, name);
        snprintf(name, sizeof(name), "model.output_attn_res_proj.weight");
        int64_t pi = cwick_spine_find(spine, name);
        if (ni == CWICK_TENSOR_MISSING || pi == CWICK_TENSOR_MISSING) {
            cwick_error_set(err, CWICK_ERR_STATE, "missing output residual weights");
            goto fail;
        }
        score_weight(score, &spine->tensors[(size_t)ni], &spine->tensors[(size_t)pi], H);
        residual_combine(mixed, hidden, eng->anchors, eng->n_anchors, score, H, eps);
        int64_t fn = cwick_spine_find(spine, "model.norm.weight");
        if (fn == CWICK_TENSOR_MISSING) {
            cwick_error_set(err, CWICK_ERR_STATE, "missing final norm");
            goto fail;
        }
        {
            const uint16_t *wn = (const uint16_t *)spine->tensors[(size_t)fn].data;
            cwick_rms_norm(post, mixed, H, NULL, eps);
            for (int64_t i = 0; i < H; i++) post[i] *= cwick_bf16_to_f32(wn[i]);
        }
        int64_t lh = cwick_spine_find(spine, "lm_head.weight");
        if (lh == CWICK_TENSOR_MISSING) {
            cwick_error_set(err, CWICK_ERR_STATE, "missing lm_head");
            goto fail;
        }
        cwick_linear(logits, post, 1, H, spec->vocab_size, &spine->tensors[(size_t)lh]);
    }

    free(score);
    free(mlp_out);
    free(post);
    free(mixed);
    free(prefix);
    free(attn_out);
    free(normed);
    free(attn_in);
    free(hidden);
    return CWICK_OK;

fail:
    free(score);
    free(mlp_out);
    free(post);
    free(mixed);
    free(prefix);
    free(attn_out);
    free(normed);
    free(attn_in);
    free(hidden);
    return CWICK_ERR_STATE;
}

/* The prefill sweep below reports progress and steers memory with the same
 * helpers the decode loop uses; those definitions live further down. */
static void maybe_progress(cwick_engine *eng, const char *phase, int64_t done,
                           int64_t total);
static void maybe_progress_at(cwick_engine *eng, const char *phase, int64_t done,
                              int64_t total, int64_t layer, int64_t nlayer);
static void maybe_steer_memory(cwick_engine *eng, int64_t step);

/* Layer-major, expert-sharing prefill.
 *
 * The token-serial forward_token() re-streams every routed expert matrix from
 * disk for every prompt token, which is why a 27k-token prompt takes ~days on a
 * model bigger than RAM. Here we sweep layer-major: for each layer we route all
 * prompt tokens and then load each selected expert ONCE, applying it to every
 * token that chose it. Disk is touched once per expert per layer, not once per
 * token, so total expert I/O collapses from (prompt_len x experts/token) to
 * (union of experts in the chunk) -- the difference between ~48h and minutes.
 * The compute is mathematically identical to forward_token(); only the order
 * and expert residency differ. The prompt is processed in RAM-bounded chunks
 * (the pinned decode cache is unused during prefill, so we can borrow RAM for
 * prompt buffers); the KV cache is written in causal order exactly as decode
 * would. */
static cwick_status forward_prefill(cwick_engine *eng, const uint32_t *prompt,
                                   int64_t prompt_len, float *logits,
                                   cwick_error *err) {
    const cwick_model_spec *spec = &eng->spec;
    const cwick_spine *spine = &eng->spine;
    const int64_t H = spec->hidden_size;
    const int64_t L = spec->num_hidden_layers;
    const int64_t res = spec->attn_res_block_size;
    const float eps = spec->rms_norm_eps_f;
    const int64_t max_anchor = eng->max_anchors;

    if (eng->pos + prompt_len > eng->capacity) {
        cwick_error_set(err, CWICK_ERR_STATE,
                       "context capacity %lld exhausted at position %lld",
                       eng->capacity, eng->pos + prompt_len);
        return CWICK_ERR_STATE;
    }

    /* chunk size bounded by RAM: the pinned expert cache is unused during
     * prefill (each expert is streamed into a tiny local buffer and reused),
     * so borrow ~half the RSS ceiling for prompt buffers. */
    int64_t per_tok = H * (8 + max_anchor) * (int64_t)sizeof(float);
    int64_t budget = eng->rss_target / 2;
    if (per_tok < 1) per_tok = 1;
    int64_t chunk = budget / per_tok;
    if (chunk < 1) chunk = 1;
    if (chunk > prompt_len) chunk = prompt_len;
    /* CWICK_PREFILL_CHUNK forces the chunk size (test/A-B knob). */
    { const char *ce = getenv("CWICK_PREFILL_CHUNK");
      if (ce) { int64_t v = atoll(ce);
                if (v > 0) { chunk = v > prompt_len ? prompt_len : v; } } }

    float *last_hidden = cwick_xmalloc((size_t)H * sizeof(float));
    float *last_anchors = cwick_xmalloc((size_t)max_anchor * H * sizeof(float));
    int64_t last_n_anch = 0;

    /* Per-chunk buffers: declared at function scope (NULL until a chunk is
     * live) so the error path can free whichever chunk is in flight. */
    float *hidden = NULL, *attn_in = NULL, *normed = NULL, *attn_out = NULL,
          *prefix = NULL, *mixed = NULL, *post = NULL, *mlp_out = NULL,
          *score = NULL, *anchors_b = NULL;
    int64_t *n_anch = NULL;

    for (int64_t base = 0; base < prompt_len; base += chunk) {
        if (engine_stopped(eng)) {
            free(last_anchors); free(last_hidden);
            cwick_error_set(err, CWICK_ERR_STATE, "%s", engine_interrupt_message(eng));
            return CWICK_ERR_STATE;
        }
        int64_t n = prompt_len - base; if (n > chunk) n = chunk;
        int64_t end = base + n;

        hidden  = cwick_xmalloc((size_t)n * H * sizeof(float));
        attn_in = cwick_xmalloc((size_t)n * H * sizeof(float));
        normed  = cwick_xmalloc((size_t)n * H * sizeof(float));
        attn_out= cwick_xmalloc((size_t)n * H * sizeof(float));
        prefix  = cwick_xmalloc((size_t)n * H * sizeof(float));
        mixed   = cwick_xmalloc((size_t)n * H * sizeof(float));
        post    = cwick_xmalloc((size_t)n * H * sizeof(float));
        mlp_out = cwick_xmalloc((size_t)n * H * sizeof(float));
        score   = cwick_xmalloc((size_t)H * sizeof(float));
        anchors_b = cwick_xmalloc((size_t)n * max_anchor * H * sizeof(float));
        n_anch = cwick_xmalloc((size_t)n * sizeof(int64_t));
        for (int64_t t = 0; t < n; t++) n_anch[t] = 0;

        /* ---- embedding ---- */
        int64_t emb_i = cwick_spine_find(spine, "embed_tokens.weight");
        const cwick_tensor *emb = &spine->tensors[(size_t)emb_i];
        if (emb_i == CWICK_TENSOR_MISSING) {
            cwick_error_set(err, CWICK_ERR_STATE, "missing embedding");
            goto prefill_fail;
        }
        for (int64_t t = 0; t < n; t++) {
            uint32_t tok = prompt[base + t];
            if (tok >= (uint32_t)emb->rows) {
                cwick_error_set(err, CWICK_ERR_ARGUMENT, "token %u out of embedding range", tok);
                goto prefill_fail;
            }
            const uint16_t *erow = (const uint16_t *)emb->data + (size_t)tok * H;
            for (int64_t i = 0; i < H; i++) hidden[t * H + i] = cwick_bf16_to_f32(erow[i]);
        }

        char name[160];
        for (int64_t l = 0; l < L; l++) {
            if (engine_stopped(eng)) {
                cwick_error_set(err, CWICK_ERR_STATE, "%s", engine_interrupt_message(eng));
                goto prefill_fail;
            }
            bool boundary = (l % res) == 0;
            /* attention phase: build KV in causal order, reuse the decode kernels */
            for (int64_t t = 0; t < n; t++) {
                /* eng->pos is this chunk's start (advanced by `n` after each
                 * chunk), so the absolute position is pos + t -- NOT
                 * eng->pos + base + t, which double-counts the chunk offset. */
                int64_t pos = eng->pos + t;
                if (t > 0 && n > CWICK_PREFILL_TOKEN_EVERY &&
                    (t % CWICK_PREFILL_TOKEN_EVERY) == 0)
                    maybe_progress_at(eng, "prefill", base + t, prompt_len,
                                      l + 1, L);
                if (n_anch[t] > 0) {
                    snprintf(name, sizeof(name), "layers.%lld.self_attention_res_norm.weight", l);
                    int64_t ni = cwick_spine_find(spine, name);
                    snprintf(name, sizeof(name), "layers.%lld.self_attention_res_proj.weight", l);
                    int64_t pi = cwick_spine_find(spine, name);
                    if (ni == CWICK_TENSOR_MISSING || pi == CWICK_TENSOR_MISSING) {
                        cwick_error_set(err, CWICK_ERR_STATE, "missing attention residual weights");
                        goto prefill_fail;
                    }
                    score_weight(score, &spine->tensors[(size_t)ni], &spine->tensors[(size_t)pi], H);
                    residual_combine(attn_in + t * H, hidden + t * H,
                                     anchors_b + t * max_anchor * H, n_anch[t], score, H, eps);
                } else {
                    memcpy(attn_in + t * H, hidden + t * H, (size_t)H * sizeof(float));
                }
                if (boundary && n_anch[t] < max_anchor) {
                    memcpy(anchors_b + (t * max_anchor + n_anch[t]) * H, hidden + t * H,
                           (size_t)H * sizeof(float));
                    n_anch[t]++;
                }
                snprintf(name, sizeof(name), "layers.%lld.input_layernorm.weight", l);
                int64_t in = cwick_spine_find(spine, name);
                if (in == CWICK_TENSOR_MISSING) {
                    cwick_error_set(err, CWICK_ERR_STATE, "missing input layernorm %lld", l);
                    goto prefill_fail;
                }
                {
                    const uint16_t *wn = (const uint16_t *)spine->tensors[(size_t)in].data;
                    cwick_rms_norm(normed + t * H, attn_in + t * H, H, NULL, eps);
                    for (int64_t i = 0; i < H; i++) (normed + t * H)[i] *= cwick_bf16_to_f32(wn[i]);
                }
                if (spec->layers[l] == CWICK_LAYER_MLA) {
                    cwick_tensor *qa, *qb, *kva, *kvb, *g, *o, *qa_n, *kva_n;
#define FIND(suffix, out) do { \
    snprintf(name, sizeof(name), "layers.%lld.self_attn.%s", l, suffix); \
    int64_t ix = cwick_spine_find(spine, name); \
    if (ix == CWICK_TENSOR_MISSING) { cwick_error_set(err, CWICK_ERR_STATE, "missing MLA weight %s", name); goto prefill_fail; } \
    out = &spine->tensors[(size_t)ix]; } while (0)
                    FIND("q_a_proj.weight", qa); FIND("q_b_proj.weight", qb);
                    FIND("kv_a_proj_with_mqa.weight", kva); FIND("kv_b_proj.weight", kvb);
                    FIND("g_proj.weight", g); FIND("o_proj.weight", o);
                    FIND("q_a_layernorm.weight", qa_n); FIND("kv_a_layernorm.weight", kva_n);
#undef FIND
                    cwick_attention_mla_decode(spec, qa, qb, kva, kvb, g, o, qa_n, kva_n,
                                              eng->keys, eng->values, eng->capacity, pos,
                                              eng->kv_slot[l], NULL, NULL,
                                              normed + t * H, attn_out + t * H, err);
                } else {
                    cwick_tensor *qp, *kp, *vp, *qc, *kc, *vc, *al, *db, *on, *fa, *fb,
                               *bp, *ga, *gb, *op;
#define FIND(suffix, out) do { \
    snprintf(name, sizeof(name), "layers.%lld.self_attn.%s", l, suffix); \
    int64_t ix = cwick_spine_find(spine, name); \
    if (ix == CWICK_TENSOR_MISSING) { cwick_error_set(err, CWICK_ERR_STATE, "missing KDA weight %s", name); goto prefill_fail; } \
    out = &spine->tensors[(size_t)ix]; } while (0)
                    FIND("q_proj.weight", qp); FIND("k_proj.weight", kp); FIND("v_proj.weight", vp);
                    FIND("q_conv1d.weight", qc); FIND("k_conv1d.weight", kc); FIND("v_conv1d.weight", vc);
                    FIND("A_log", al); FIND("dt_bias", db); FIND("o_norm.weight", on);
                    FIND("f_a_proj.weight", fa); FIND("f_b_proj.weight", fb); FIND("b_proj.weight", bp);
                    FIND("g_a_proj.weight", ga); FIND("g_b_proj.weight", gb); FIND("o_proj.weight", op);
#undef FIND
                    cwick_attention_kda_decode(spec, qp, kp, vp, qc, kc, vc, al, db, on, fa, fb,
                                              bp, ga, gb, op, eng->kda_rec, eng->kda_conv, l,
                                              normed + t * H, attn_out + t * H, err);
                }
            }
            /* prefix = pre_attn + attn_out (or attn_out at a boundary) */
            for (int64_t t = 0; t < n; t++) {
                if (boundary) memcpy(prefix + t * H, attn_out + t * H, (size_t)H * sizeof(float));
                else for (int64_t i = 0; i < H; i++)
                    (prefix + t * H)[i] = hidden[t * H + i] + (attn_out + t * H)[i];
            }
            /* MLP phase */
            for (int64_t t = 0; t < n; t++) {
                if (n_anch[t] > 0) {
                    snprintf(name, sizeof(name), "layers.%lld.mlp_res_norm.weight", l);
                    int64_t ni = cwick_spine_find(spine, name);
                    snprintf(name, sizeof(name), "layers.%lld.mlp_res_proj.weight", l);
                    int64_t pi = cwick_spine_find(spine, name);
                    if (ni == CWICK_TENSOR_MISSING || pi == CWICK_TENSOR_MISSING) {
                        cwick_error_set(err, CWICK_ERR_STATE, "missing MLP residual weights");
                        goto prefill_fail;
                    }
                    score_weight(score, &spine->tensors[(size_t)ni], &spine->tensors[(size_t)pi], H);
                    residual_combine(mixed + t * H, prefix + t * H, anchors_b + t * max_anchor * H,
                                     n_anch[t], score, H, eps);
                } else {
                    memcpy(mixed + t * H, prefix + t * H, (size_t)H * sizeof(float));
                }
                snprintf(name, sizeof(name), "layers.%lld.post_attention_layernorm.weight", l);
                int64_t pn = cwick_spine_find(spine, name);
                if (pn == CWICK_TENSOR_MISSING) {
                    cwick_error_set(err, CWICK_ERR_STATE, "missing post-attention norm %lld", l);
                    goto prefill_fail;
                }
                {
                    const uint16_t *wn = (const uint16_t *)spine->tensors[(size_t)pn].data;
                    cwick_rms_norm(post + t * H, mixed + t * H, H, NULL, eps);
                    for (int64_t i = 0; i < H; i++) (post + t * H)[i] *= cwick_bf16_to_f32(wn[i]);
                }
            }
            if (spec->mlp_dense && spec->mlp_dense[l]) {
                for (int64_t t = 0; t < n; t++)
                    cwick_dense_mlp(spec, spine, l, post + t * H, mlp_out + t * H, err);
            } else {
                cwick_moe_forward_batch(spec, spine, l, post, n, mlp_out, err,
                                      moe_batch_progress, eng);
            }
            for (int64_t t = 0; t < n; t++)
                for (int64_t i = 0; i < H; i++)
                    (hidden + t * H)[i] = (prefix + t * H)[i] + (mlp_out + t * H)[i];
        }

        /* remember the last token's state for the tail (lm_head) */
        if (end == prompt_len) {
            int64_t last = n - 1;
            memcpy(last_hidden, hidden + last * H, (size_t)H * sizeof(float));
            last_n_anch = n_anch[last];
            memcpy(last_anchors, anchors_b + last * max_anchor * H,
                   (size_t)last_n_anch * H * sizeof(float));
        }

        eng->pos += n;
        maybe_progress(eng, "prefill", end, prompt_len);
        maybe_steer_memory(eng, base);

        free(n_anch); free(anchors_b); free(score); free(mlp_out); free(post);
        free(mixed); free(prefix); free(attn_out); free(normed); free(attn_in); free(hidden);
        n_anch = NULL; anchors_b = NULL; score = NULL; mlp_out = NULL; post = NULL;
        mixed = NULL; prefix = NULL; attn_out = NULL; normed = NULL;
        attn_in = NULL; hidden = NULL;
    }

    /* ---- tail: output residual, final norm, lm_head (on the last token) ---- */
    {
        char name[160];
        snprintf(name, sizeof(name), "model.output_attn_res_norm.weight");
        int64_t ni = cwick_spine_find(spine, name);
        snprintf(name, sizeof(name), "model.output_attn_res_proj.weight");
        int64_t pi = cwick_spine_find(spine, name);
        if (ni == CWICK_TENSOR_MISSING || pi == CWICK_TENSOR_MISSING) {
            free(last_anchors); free(last_hidden);
            cwick_error_set(err, CWICK_ERR_STATE, "missing output residual weights");
            return CWICK_ERR_STATE;
        }
        float *tscore = cwick_xmalloc((size_t)H * sizeof(float));
        float *tmixed = cwick_xmalloc((size_t)H * sizeof(float));
        float *tpost  = cwick_xmalloc((size_t)H * sizeof(float));
        score_weight(tscore, &spine->tensors[(size_t)ni], &spine->tensors[(size_t)pi], H);
        residual_combine(tmixed, last_hidden, last_anchors, last_n_anch, tscore, H, eps);
        int64_t fn = cwick_spine_find(spine, "model.norm.weight");
        if (fn == CWICK_TENSOR_MISSING) {
            free(tpost); free(tmixed); free(tscore); free(last_anchors); free(last_hidden);
            cwick_error_set(err, CWICK_ERR_STATE, "missing final norm");
            return CWICK_ERR_STATE;
        }
        {
            const uint16_t *wn = (const uint16_t *)spine->tensors[(size_t)fn].data;
            cwick_rms_norm(tpost, tmixed, H, NULL, eps);
            for (int64_t i = 0; i < H; i++) tpost[i] *= cwick_bf16_to_f32(wn[i]);
        }
        int64_t lh = cwick_spine_find(spine, "lm_head.weight");
        if (lh == CWICK_TENSOR_MISSING) {
            free(tpost); free(tmixed); free(tscore); free(last_anchors); free(last_hidden);
            cwick_error_set(err, CWICK_ERR_STATE, "missing lm_head");
            return CWICK_ERR_STATE;
        }
        cwick_linear(logits, tpost, 1, H, spec->vocab_size, &spine->tensors[(size_t)lh]);
        free(tpost); free(tmixed); free(tscore);
    }
    free(last_anchors); free(last_hidden);
    return CWICK_OK;

prefill_fail:
    free(n_anch); free(anchors_b); free(score); free(mlp_out); free(post);
    free(mixed); free(prefix); free(attn_out); free(normed); free(attn_in); free(hidden);
    free(last_anchors); free(last_hidden);
    return CWICK_ERR_STATE;
}

/* ------------------------------------------------------------------ */
/* Prefix checkpoints taken part-way through a prefill                  */
/* ------------------------------------------------------------------ */
/* One snapshot at the end of a prefill only ever matches a prompt of that
 * exact length. An agent session does not work like that: the system prompt
 * and tool list stay put while the history after them is rewritten, compacted
 * or truncated, so turn 40's prompt shares 90% of its tokens with turn 39's
 * and matches no cache entry at all -- and the whole multi-hour prefill runs
 * again.
 *
 * A chunk boundary is the only place a snapshot can legally be taken. The
 * recurrent state (the DeltaNet accumulator, the conv window, the indexer
 * pool) is a fold over the tokens seen so far: it can be snapshotted whole
 * but not rewound, so there is no way to derive the state at 4096 tokens from
 * the state at 12626. Chunks are sequential sweeps through every layer, so at
 * the end of one the engine holds exactly the state of the tokens consumed so
 * far -- which is a prefix state, and can be written out.
 *
 * The cost of asking for that is one more expert pass per extra chunk (each
 * chunk re-reads its layer's expert union). Measured on this machine a pass
 * saturates at ~285 GiB and ~58 s, against a per-token cost of ~0.30 s: on a
 * 12,626-token prefill three chunks instead of one add ~116 s to ~3,846 s,
 * i.e. 3%, and leave checkpoints at 4096 / 8192 / 12288 behind. */
static void pfx_checkpoint_now(cwick_engine *eng) {
    if (!eng->pfx_cap_done || eng->pos <= 0) return;
    struct cwick_prefix_state *st = calloc(1, sizeof(*st));
    if (!st) return;
    cwick_error e;
    cwick_error_clear(&e);
    if (cwick_engine_capture_prefix(eng, eng->pos, st, &e) != CWICK_OK) {
        CWICK_VERB(eng_tag(eng), "prefix checkpoint at %lld tokens failed: %s",
                  (long long)eng->pos, cwick_error_message(&e));
        cwick_prefix_state_free(st);
        return;
    }
    CWICK_INFO(eng_tag(eng), "prefix checkpoint at %lld tokens — written before "
              "the rest of the prefill, so a kill from here on still leaves "
              "this much of the prompt reusable", (long long)eng->pos);
    eng->pfx_cap_done(eng->pfx_cap_done_user, st);   /* takes ownership */
}

/* Fill the armed snapshot buffer with the state of `pos` tokens and hand it
 * to the persistence callback. Both backends capture at their own "prompt is
 * prefilled" moment -- the GLM one at the end of engine_prefill, the DeepSeek
 * one from its prefill hook -- but what happens to the snapshot afterwards is
 * identical, so it lives here. A failed capture must not fail the request:
 * the prefix cache is an optimisation. */
static void pfx_capture_and_handoff(cwick_engine *eng, int64_t pos) {
    if (!eng->pfx_cap) return;
    cwick_error cap_err;
    cwick_error_clear(&cap_err);
    if (cwick_engine_capture_prefix(eng, pos, eng->pfx_cap, &cap_err) != CWICK_OK) {
        CWICK_VERB(eng_tag(eng), "prefix capture failed: %s",
                   cwick_error_message(&cap_err));
        return;
    }
    /* Hand the snapshot over NOW, not when generation returns: the caller
     * persists it while the slot is still busy with this request anyway,
     * instead of leaving an hours-long decode between the snapshot and its
     * durability. The callback takes ownership; both pointers are cleared so
     * the engine can never fire it twice or point a later request at a freed
     * state. */
    void (*done)(void *, struct cwick_prefix_state *) = eng->pfx_cap_done;
    void *done_user = eng->pfx_cap_done_user;
    struct cwick_prefix_state *snap = eng->pfx_cap;
    eng->pfx_cap = NULL;
    eng->pfx_cap_done = NULL;
    eng->pfx_cap_done_user = NULL;
    if (done) done(done_user, snap);
}

/* DeepSeek prefill hook: see ds_engine.h's on_prefill_done. */
static void backend_prefix_hook(void *user, int64_t pos) {
    cwick_engine *eng = (cwick_engine *)user;
    if (eng && pos > 0) pfx_capture_and_handoff(eng, pos);
}

/* DeepSeek mid-prefill checkpoint: same idea as pfx_checkpoint_now, but the
 * position comes from the DS engine (eng->pos stays 0 on this backend). It
 * captures into its own state so the final snapshot stays armed. */
static void backend_prefix_checkpoint(void *user, int64_t pos) {
    cwick_engine *eng = (cwick_engine *)user;
    if (!eng || !eng->pfx_cap_done || pos <= 0) return;
    struct cwick_prefix_state *st = calloc(1, sizeof(*st));
    if (!st) return;
    cwick_error e;
    cwick_error_clear(&e);
    if (cwick_engine_capture_prefix(eng, pos, st, &e) != CWICK_OK) {
        CWICK_VERB(eng_tag(eng), "prefix checkpoint at %lld tokens failed: %s",
                   (long long)pos, cwick_error_message(&e));
        cwick_prefix_state_free(st);
        free(st);
        return;
    }
    CWICK_INFO(eng_tag(eng), "prefix checkpoint at %lld tokens — reusable "
               "before the rest of the prefill finishes", (long long)pos);
    eng->pfx_cap_done(eng->pfx_cap_done_user, st);   /* takes ownership */
}

/* ------------------------------------------------------------------ */
/* Layer-major, expert-sharing prefill for the released checkpoint     */
/* (mHC + KDA/DSA + MoE)                                                */
/* ------------------------------------------------------------------ */
/* forward_token_mhc() re-streams every routed expert matrix from disk for
 * every prompt token -- 45 layers x 8 experts x 3 matrices per token, times
 * a 27k-token prompt, i.e. tens of millions of reads, which is why prefill
 * takes ~days on a model larger than RAM. Here we sweep layer-major over a
 * RAM-bounded chunk: within a layer every token still runs its own mHC
 * pre/post and its attention (the KV cache is written in causal order,
 * exactly as decode would), but the MLP/MoE of the whole chunk is evaluated
 * in a single batched pass, so each selected expert is read ONCE and applied
 * to every token that chose it. The math is identical to forward_token_mhc();
 * only the order and expert residency differ. */
static cwick_status forward_prefill_mhc(cwick_engine *eng, const uint32_t *prompt,
                                       int64_t prompt_len, float *logits,
                                       cwick_error *err) {
    const cwick_model_spec *spec = &eng->spec;
    const cwick_spine *spine = &eng->spine;
    const int64_t H = spec->hidden_size;
    const int64_t L = spec->num_hidden_layers;
    const int64_t ns = spec->hc_mult > 0 ? spec->hc_mult : 1;
    const int64_t mix = 2 * ns + ns * ns;
    const int64_t d = ns * H;            /* width of one token's residual bank */
    const float eps = spec->rms_norm_eps_f;
    const float hce = spec->hc_eps;
    const int sinkhorn = (int)spec->hc_sinkhorn_iters;

    if (eng->pos + prompt_len > eng->capacity) {
        cwick_error_set(err, CWICK_ERR_STATE,
                       "context capacity %lld exhausted at position %lld",
                       eng->capacity, eng->pos + prompt_len);
        return CWICK_ERR_STATE;
    }

    /* Chunk bounded by RAM: the pinned decode cache is unused during prefill,
     * so borrow half the RSS ceiling for the per-token state banks. The larger
     * the chunk, the fewer times an expert is re-read across chunks. */
    int64_t per_tok = (d + 2 * H + ns + ns * ns) * (int64_t)sizeof(float);
    int64_t budget = eng->rss_target / 2;
    if (per_tok < 1) per_tok = 1;
    int64_t chunk = budget / per_tok;
    if (chunk < 1) chunk = 1;
    if (chunk > prompt_len) chunk = prompt_len;
    /* Checkpoint granularity (see pfx_checkpoint_now): cap the chunk so its
     * boundaries land on multiples of the block, which is where a prefix
     * snapshot may legally be taken. Only for prompts with room for at least
     * two blocks -- on anything shorter the extra expert pass would cost more
     * than the one snapshot at the end is worth. */
    if (eng->pfx_block > 0 && prompt_len >= 2 * eng->pfx_block &&
        chunk > eng->pfx_block)
        chunk = eng->pfx_block;
    /* CWICK_PREFILL_CHUNK forces the chunk size (test/A-B knob): a small value
     * exercises the cross-chunk KV/KDA carry, a value of 1 degrades to the
     * token-serial order for comparison. */
    { const char *ce = getenv("CWICK_PREFILL_CHUNK");
      if (ce) { int64_t v = atoll(ce);
                if (v > 0) { chunk = v > prompt_len ? prompt_len : v; } } }

    float *last_contract = cwick_xmalloc((size_t)H * sizeof(float));

    /* per-chunk state + per-token scratch, at function scope (NULL until a
     * chunk is live) so the error path can free whichever chunk is in flight */
    float *res = NULL, *prev_out = NULL, *post = NULL, *comb = NULL;
    float *mlp_in = NULL, *mlp_out = NULL;
    float *attn_in = NULL, *attn_out = NULL;   /* chunk-wide attention I/O */
    float *scratch = NULL, *sub_in = NULL, *sub_out = NULL;
    float *mixes = NULL, *pre = NULL;

    int64_t ei = cwick_spine_find(spine, "embed_tokens.weight");
    if (ei == CWICK_TENSOR_MISSING) {
        free(last_contract);
        cwick_error_set(err, CWICK_ERR_STATE, "missing embedding table");
        return CWICK_ERR_STATE;
    }
    const cwick_tensor *emb = &spine->tensors[(size_t)ei];

    char name[192];
    for (int64_t base = 0; base < prompt_len; base += chunk) {
        if (engine_stopped(eng)) {
            free(last_contract);
            cwick_error_set(err, CWICK_ERR_STATE, "%s", engine_interrupt_message(eng));
            return CWICK_ERR_STATE;
        }
        int64_t n = prompt_len - base; if (n > chunk) n = chunk;

        res      = cwick_xmalloc((size_t)n * d * sizeof(float));
        prev_out = cwick_xmalloc((size_t)n * H * sizeof(float));
        post     = cwick_xmalloc((size_t)n * ns * sizeof(float));
        comb     = cwick_xmalloc((size_t)n * ns * ns * sizeof(float));
        mlp_in   = cwick_xmalloc((size_t)n * H * sizeof(float));
        mlp_out  = cwick_xmalloc((size_t)n * H * sizeof(float));
        attn_in  = cwick_xmalloc((size_t)n * H * sizeof(float));
        attn_out = cwick_xmalloc((size_t)n * H * sizeof(float));
        scratch  = cwick_xmalloc((size_t)d * sizeof(float));
        sub_in   = cwick_xmalloc((size_t)H * sizeof(float));
        sub_out  = cwick_xmalloc((size_t)H * sizeof(float));
        mixes    = cwick_xmalloc((size_t)mix * sizeof(float));
        pre      = cwick_xmalloc((size_t)ns * sizeof(float));

        /* embedding -> expand to ns streams, per token */
        for (int64_t t = 0; t < n; t++) {
            uint32_t tok = prompt[base + t];
            if (tok >= (uint32_t)emb->rows) {
                cwick_error_set(err, CWICK_ERR_ARGUMENT,
                               "token %u out of embedding range", tok);
                goto pf_mhc_fail;
            }
            const uint16_t *erow = (const uint16_t *)emb->data + (size_t)tok * H;
            for (int64_t i = 0; i < H; i++) sub_in[i] = cwick_bf16_to_f32(erow[i]);
            hc_expand(sub_in, res + t * d, ns, H);
        }

        for (int64_t l = 0; l < L; l++) {
            if (engine_stopped(eng)) {
                cwick_error_set(err, CWICK_ERR_STATE, "%s", engine_interrupt_message(eng));
                goto pf_mhc_fail;
            }
            /* Layer-granular heartbeat. A chunk is one atomic sweep of all L
             * layers, so the token counter cannot advance inside it: without
             * this, a prompt that fits in a single chunk (the common serve
             * case -- 115 tokens) streams for a minute with no output at all,
             * because the chunk-end token count never lands on a multiple of
             * CWICK_PROGRESS_EVERY. */
            if ((l % CWICK_PROGRESS_EVERY) == 0)
                maybe_progress_at(eng, "prefill", base + n, prompt_len,
                                  l + 1, L);
            const cwick_tensor *a_fn = NULL, *a_bs = NULL, *a_sc = NULL, *a_nw = NULL;
            const cwick_tensor *f_fn = NULL, *f_bs = NULL, *f_sc = NULL, *f_nw = NULL;
            const cwick_tensor *qa = NULL, *qb = NULL, *kva = NULL, *kvb = NULL;
            const cwick_tensor *o_p = NULL, *qa_n = NULL, *kva_n = NULL;
            const cwick_tensor *qp = NULL, *kp = NULL, *vp = NULL;
            const cwick_tensor *qc = NULL, *kc = NULL, *vc = NULL;
            const cwick_tensor *al = NULL, *db = NULL, *on = NULL;
            const cwick_tensor *fa = NULL, *fb = NULL, *bp = NULL;
            const cwick_tensor *ga = NULL, *gb = NULL, *op = NULL;
            const bool attn_mla = (spec->layers[l] == CWICK_LAYER_DSA ||
                                   spec->layers[l] == CWICK_LAYER_MLA);
            const int64_t slot = eng->kv_slot[l];

            /* Resolve this layer's weights ONCE (they are the same for every
             * token); the token loop below only does arithmetic. */
#define MHCF(suffix, out) do { \
            snprintf(name, sizeof(name), "layers.%lld.%s", l, suffix); \
            int64_t _ix = cwick_spine_find(spine, name); \
            if (_ix == CWICK_TENSOR_MISSING) { \
                cwick_error_set(err, CWICK_ERR_STATE, "missing weight %s", name); \
                goto pf_mhc_fail; \
            } \
            out = &spine->tensors[(size_t)_ix]; \
} while (0)
            MHCF("hc_attn_fn", a_fn);
            MHCF("hc_attn_base", a_bs);
            MHCF("hc_attn_scale", a_sc);
            MHCF("input_layernorm.weight", a_nw);
            MHCF("hc_ffn_fn", f_fn);
            MHCF("hc_ffn_base", f_bs);
            MHCF("hc_ffn_scale", f_sc);
            MHCF("post_attention_layernorm.weight", f_nw);
            if (attn_mla) {
                if (slot < 0 || !eng->keys) {
                    cwick_error_set(err, CWICK_ERR_STATE,
                                   "layer %lld has no KV cache slot", l);
                    goto pf_mhc_fail;
                }
                MHCF("self_attn.q_a_proj.weight", qa);
                MHCF("self_attn.q_b_proj.weight", qb);
                MHCF("self_attn.kv_a_proj_with_mqa.weight", kva);
                MHCF("self_attn.kv_b_proj.weight", kvb);
                MHCF("self_attn.o_proj.weight", o_p);
                MHCF("self_attn.q_a_layernorm.weight", qa_n);
                MHCF("self_attn.kv_a_layernorm.weight", kva_n);
            } else {
                MHCF("self_attn.q_proj.weight", qp);
                MHCF("self_attn.k_proj.weight", kp);
                MHCF("self_attn.v_proj.weight", vp);
                MHCF("self_attn.q_conv1d.weight", qc);
                MHCF("self_attn.k_conv1d.weight", kc);
                MHCF("self_attn.v_conv1d.weight", vc);
                MHCF("self_attn.A_log", al);
                MHCF("self_attn.dt_bias", db);
                MHCF("self_attn.o_norm.weight", on);
                MHCF("self_attn.f_a_proj.weight", fa);
                MHCF("self_attn.f_b_proj.weight", fb);
                MHCF("self_attn.b_proj.weight", bp);
                MHCF("self_attn.g_a_proj.weight", ga);
                MHCF("self_attn.g_b_proj.weight", gb);
                MHCF("self_attn.o_proj.weight", op);
            }
#undef MHCF

            /* ---- chunk-wide attention for the sparse layers --------------
             * Their four projections become one GEMM per weight for the whole
             * chunk instead of one GEMV per token: a GEMV re-reads the weight
             * matrix, and at 3037 tokens that is ~700 GB of traffic per layer
             * against ~230 MB read once. The parts that genuinely depend on
             * token order (the DSA indexer, the score and value reduction)
             * still run per token, in order. */
            const bool can_chunk = attn_mla && cwick_mla_chunk_supported(
                    spec, qa, qb, kva, kvb, o_p, eng->values);
            bool did_chunk = false;
            if (can_chunk) {
                for (int64_t t = 0; t < n; t++) {
                    float *r  = res + t * d;
                    float *po = post + t * ns;
                    float *co = comb + t * ns * ns;
                    if (t > 0 && n > CWICK_PREFILL_TOKEN_EVERY &&
                        (t % CWICK_PREFILL_TOKEN_EVERY) == 0)
                        maybe_progress_at(eng, "prefill", base + t, prompt_len,
                                          l + 1, L);
                    if (l > 0) {
                        mhc_post(scratch, prev_out + t * H, r, po, co, ns, H);
                        memcpy(r, scratch, (size_t)d * sizeof(float));
                    }
                    mhc_pre(r, ns, H, a_fn, a_bs, a_sc, a_nw, eps, hce, sinkhorn,
                            po, co, attn_in + t * H, mixes, pre);
                }
                cwick_error chunk_err;
                cwick_error_clear(&chunk_err);
                did_chunk = cwick_attention_mla_prefill_chunk(
                                spec, qa, qb, kva, kvb, o_p, qa_n, kva_n,
                                eng->keys, eng->values, eng->capacity, eng->pos,
                                n, slot,
                                eng->idx_w ? &eng->idx_w[slot] : NULL,
                                eng->idx_state ? &eng->idx_state[slot] : NULL,
                                attn_in, attn_out, &chunk_err) == CWICK_OK;
                if (!did_chunk) {
                    /* The mHC pre-mix for this chunk is already done, so the
                     * fallback is only the attention itself -- running the
                     * fused per-token loop here would mix the streams twice. */
                    CWICK_VERB(eng_tag(eng), "layer %lld: chunk attention unavailable "
                              "(%s) — attention per token instead",
                              (long long)l, cwick_error_message(&chunk_err));
                    for (int64_t t = 0; t < n; t++)
                        cwick_attention_mla_decode(spec, qa, qb, kva, kvb, NULL,
                                                  o_p, qa_n, kva_n, eng->keys,
                                                  eng->values, eng->capacity,
                                                  eng->pos + t, slot,
                                                  eng->idx_w ? &eng->idx_w[slot] : NULL,
                                                  eng->idx_state ? &eng->idx_state[slot] : NULL,
                                                  attn_in + t * H, attn_out + t * H,
                                                  err);
                }
                for (int64_t t = 0; t < n; t++) {
                    float *r  = res + t * d;
                    float *po = post + t * ns;
                    float *co = comb + t * ns * ns;
                    mhc_post(scratch, attn_out + t * H, r, po, co, ns, H);
                    memcpy(r, scratch, (size_t)d * sizeof(float));
                    mhc_pre(r, ns, H, f_fn, f_bs, f_sc, f_nw, eps, hce,
                            sinkhorn, po, co, mlp_in + t * H, mixes, pre);
                }
            }
            if (!did_chunk)
            /* ---- per token: mHC state + attention (serial, causal KV) ---- */
            for (int64_t t = 0; t < n; t++) {
                float *r  = res + t * d;
                float *po = post + t * ns;
                float *co = comb + t * ns * ns;
                /* eng->pos is this chunk's start (advanced by `n` after each
                 * chunk), so the absolute position is pos + t -- NOT
                 * eng->pos + base + t, which double-counts the chunk offset. */
                int64_t pos = eng->pos + t;
                /* Token heartbeat: this sweep is the single-threaded part of
                 * prefill, so on a 30K-token prompt the layer counter alone
                 * leaves the operator blind for minutes. */
                if (t > 0 && n > CWICK_PREFILL_TOKEN_EVERY &&
                    (t % CWICK_PREFILL_TOKEN_EVERY) == 0)
                    maybe_progress_at(eng, "prefill", base + t, prompt_len,
                                      l + 1, L);

                if (l > 0) {
                    mhc_post(scratch, prev_out + t * H, r, po, co, ns, H);
                    memcpy(r, scratch, (size_t)d * sizeof(float));
                }
                mhc_pre(r, ns, H, a_fn, a_bs, a_sc, a_nw, eps, hce, sinkhorn,
                        po, co, sub_in, mixes, pre);
                if (attn_mla) {
                    cwick_attention_mla_decode(spec, qa, qb, kva, kvb, NULL, o_p,
                                              qa_n, kva_n, eng->keys, eng->values,
                                              eng->capacity, pos, slot,
                                              eng->idx_w ? &eng->idx_w[slot] : NULL,
                                              eng->idx_state ? &eng->idx_state[slot] : NULL,
                                              sub_in, sub_out, err);
                } else {
                    cwick_attention_kda_decode(spec, qp, kp, vp, qc, kc, vc, al, db, on,
                                              fa, fb, bp, ga, gb, op, eng->kda_rec,
                                              eng->kda_conv, l, sub_in, sub_out, err);
                }
                mhc_post(scratch, sub_out, r, po, co, ns, H);
                memcpy(r, scratch, (size_t)d * sizeof(float));
                mhc_pre(r, ns, H, f_fn, f_bs, f_sc, f_nw, eps, hce, sinkhorn,
                        po, co, mlp_in + t * H, mixes, pre);
            }

            /* ---- one batched MLP for the whole chunk: each expert once ---- */
            if (spec->mlp_dense && spec->mlp_dense[l]) {
                for (int64_t t = 0; t < n; t++)
                    cwick_dense_mlp(spec, spine, l, mlp_in + t * H, mlp_out + t * H, err);
            } else {
                cwick_moe_forward_batch(spec, spine, l, mlp_in, n, mlp_out, err,
                                      moe_batch_progress, eng);
            }
            if (err && err->status != CWICK_OK) goto pf_mhc_fail;

            /* carry the MLP output as the next layer's deferred hc_post input */
            for (int64_t t = 0; t < n; t++)
                memcpy(prev_out + t * H, mlp_out + t * H, (size_t)H * sizeof(float));
        }

        /* Fold the last token's final hc_post and contract it while its state
         * banks are still live (mirrors forward_token_mhc's tail). */
        if (base + n == prompt_len) {
            int64_t t = n - 1;
            mhc_post(scratch, prev_out + t * H, res + t * d, post + t * ns,
                     comb + t * ns * ns, ns, H);
            hc_contract(scratch, last_contract, ns, H);
        }

        eng->pos += n;
        maybe_progress(eng, "prefill", base + n, prompt_len);
        maybe_steer_memory(eng, base);
        /* Every chunk boundary except the last is a usable prefix state, and
         * the last one is captured after the prefill returns (where the tail
         * contraction has happened). Writing the intermediate ones here means
         * a prefill that is killed -- this process has been SIGKILLed mid-
         * prefill more than once -- still leaves the part it finished behind
         * as a reusable checkpoint. */
        if (base + n < prompt_len && eng->pfx_block > 0)
            pfx_checkpoint_now(eng);

        free(pre); free(mixes); free(sub_out); free(sub_in); free(scratch);
        free(mlp_out); free(mlp_in); free(comb); free(post); free(prev_out); free(res);
        free(attn_out); free(attn_in);
        pre = mixes = sub_out = sub_in = scratch = NULL;
        mlp_out = mlp_in = comb = post = prev_out = res = NULL;
        attn_out = attn_in = NULL;
    }

    /* ---- tail: final norm + lm_head on the contracted last token ---- */
    {
        float *hidden = cwick_xmalloc((size_t)H * sizeof(float));
        int64_t fi = cwick_spine_find(spine, "model.norm.weight");
        int64_t li = cwick_spine_find(spine, "lm_head.weight");
        if (fi == CWICK_TENSOR_MISSING || li == CWICK_TENSOR_MISSING) {
            free(hidden); free(last_contract);
            cwick_error_set(err, CWICK_ERR_STATE, "missing final norm or lm_head");
            return CWICK_ERR_STATE;
        }
        const uint16_t *wn = (const uint16_t *)spine->tensors[(size_t)fi].data;
        cwick_rms_norm(hidden, last_contract, H, NULL, eps);
        for (int64_t i = 0; i < H; i++) hidden[i] *= cwick_bf16_to_f32(wn[i]);
        cwick_linear(logits, hidden, 1, H, spec->vocab_size,
                    &spine->tensors[(size_t)li]);
        free(hidden);
    }
    free(last_contract);
    return CWICK_OK;

pf_mhc_fail:
    free(pre); free(mixes); free(sub_out); free(sub_in); free(scratch);
    free(mlp_out); free(mlp_in); free(comb); free(post); free(prev_out); free(res);
    free(attn_out); free(attn_in);
    free(last_contract);
    return CWICK_ERR_STATE;
}

/* ------------------------------------------------------------------ */
/* Engine construction                                                 */
/* ------------------------------------------------------------------ */
static int64_t next_capacity(const cwick_model_spec *spec, int64_t req) {
    int64_t max = spec->max_position_embeddings;
    if (max < 64) max = 64;
    int64_t cap;
    if (req > 0) {
        /* explicit --context: honour it up to the checkpoint's own ceiling */
        cap = req;
    } else {
        /* memory-safe default for the released checkpoint */
        cap = max > 4096 ? 4096 : max;
    }
    if (cap > max) cap = max;
    if (cap < 64) cap = 64;
    return cap;
}

cwick_status cwick_engine_new(const cwick_engine_options *opt, cwick_engine **out,
                            cwick_error *err) {
    cwick_error_clear(err);
    cwick_engine *eng = cwick_xmalloc(sizeof(cwick_engine));
    memset(eng, 0, sizeof(*eng));
    /* 0 is a legal budget (no thinking at all), so the disabled state has to
     * be an explicit -1 -- a zeroed struct must not read as "close the think
     * block before the first token". */
    eng->think_budget = -1;
    eng->think_open_id = -1;
    eng->think_close_id = -1;
    eng->model_root = cwick_strdup(opt->model_root);
    eng->max_new = opt->max_new_tokens > 0 ? opt->max_new_tokens : 4;
    eng->verbose = opt->verbose;
    eng->stop = opt->stop;
    eng->slot_id = opt->slot_id;

    char cfg[1152];
    cwick_path_join(cfg, sizeof(cfg), opt->model_root, "config.json");
    cwick_status st = cwick_model_spec_load(cfg, &eng->spec, err);
    if (st != CWICK_OK) {
        cwick_engine_free(eng);
        return st;
    }
    {
        const cwick_model_spec *s = &eng->spec;
        CWICK_VERB("init", "contract ok: %s hidden=%lld layers=%lld experts=%lld "
                  "top-%lld (%lld KDA / %lld DSA / %lld MLA)",
                  s->model_type, (long long)s->hidden_size,
                  (long long)s->num_hidden_layers, (long long)s->num_experts,
                  (long long)s->num_experts_per_token, (long long)s->kda_layers,
                  (long long)s->dsa_layers, (long long)s->mla_layers);
    }

    /* Sampling defaults come from the checkpoint, not from us: a model that
     * ships do_sample=true / temperature 1.0 / top_p 0.95 / top_k 20 was
     * trained to be drawn from, and decoding it with argmax walks straight
     * into repetition attractors (the same clause, label or line over and
     * over until the token budget runs out). Nothing changes for a checkpoint
     * with no generation_config.json -- that stays greedy. */
    cwick_sampler_cfg_default(&eng->smp);
    if (cwick_sampler_cfg_from_checkpoint(opt->model_root, &eng->smp)) {
        char sd[96];
        cwick_sampler_describe(&eng->smp, sd, sizeof(sd));
        CWICK_INFO("init", "sampling defaults from generation_config.json: %s", sd);
        eng->smp_ckpt_have = true;
    }
    eng->smp_ckpt = eng->smp;

    /* ---- DeepSeek-V4.1: its own loader, caches and forward pass ---- */
    if (strcmp(eng->spec.model_type, "deepseek_v41") == 0 ||
        strcmp(eng->spec.model_type, "deepseek_v41_text") == 0) {
        eng->capacity = next_capacity(&eng->spec, opt->max_context);
        eng->ds = cwick_xmalloc(sizeof(ds_engine));
        if (ds_engine_open(eng->ds, opt->model_root, eng->capacity, err) != 0) {
            free(eng->ds);
            eng->ds = NULL;
            cwick_engine_free(eng);
            return CWICK_ERR_STATE;
        }
        eng->ds->ctx.stop = opt->stop;
        eng->ds->ctx.abort_poll = NULL;
        eng->ds->ctx.abort_poll_user = NULL;
        eng->ds->ctx.verbose = opt->verbose;
        eng->ds->ctx.tag = eng->req_tag;
        eng->ds->ctx.spec = &eng->spec;
        /* Let the prefix cache snapshot this backend when its state equals
         * "prompt prefilled" -- the one moment such a snapshot is valid --
         * and every pfx_block positions during a long prefill. */
        eng->ds->on_prefill_done = backend_prefix_hook;
        eng->ds->on_prefill_done_user = eng;
        eng->ds->on_prefill_checkpoint = backend_prefix_checkpoint;
        eng->ds->on_prefill_checkpoint_user = eng;
        eng->ds->checkpoint_block = eng->pfx_block;
        eng->ds->smp = eng->smp;   /* same sampling defaults as the GLM path */
        eng->max_new = opt->max_new_tokens > 0 ? opt->max_new_tokens : 512;
        CWICK_INFO("init", "ds engine ready: %s dim=%lld layers=%lld experts=%lld "
                  "top-%lld context=%lld",
                  eng->spec.model_type, (long long)eng->spec.hidden_size,
                  (long long)eng->spec.num_hidden_layers,
                  (long long)eng->spec.num_experts,
                  (long long)eng->spec.num_experts_per_token,
                  (long long)eng->capacity);
        *out = eng;
        return CWICK_OK;
    }

    /* ---- MiniMax-M3.1-Flash: its own loader and forward pass, same shape
     * as the DeepSeek branch above ---- */
    if (strncmp(eng->spec.model_type, "minimax_m3", 10) == 0) {
        eng->capacity = next_capacity(&eng->spec, opt->max_context);
        /* This checkpoint's KV cache is bf16 GQA, not a compressed latent, so
         * it costs 2.3 KiB per position per layer: 4.2 GiB at 32K but 129 GiB
         * at 1M. A window that does not leave room for the expert cache is
         * not a slow window, it is a thrashing one, so cap the default and
         * say what the cap costs. --context still overrides. */
        {
            if (opt->max_context <= 0 && eng->capacity > 32768) {
                CWICK_INFO("init", "minimax_m3: context capped at 32768 "
                           "positions (asked %lld); this checkpoint's KV is "
                           "2.3 KiB per position per layer, so every MiB of it "
                           "is a MiB the expert cache cannot hold",
                           (long long)eng->capacity);
                eng->capacity = 32768;
            }
        }
        eng->mm = cwick_xmalloc(sizeof(mm_engine));
        if (mm_engine_open(eng->mm, opt->model_root, eng->capacity, err) != 0) {
            free(eng->mm);
            eng->mm = NULL;
            cwick_engine_free(eng);
            return CWICK_ERR_STATE;
        }
        eng->mm->ctx.stop = opt->stop;
        eng->mm->ctx.abort_poll = NULL;
        eng->mm->ctx.abort_poll_user = NULL;
        eng->mm->ctx.verbose = opt->verbose;
        eng->mm->ctx.tag = eng->req_tag;
        eng->mm->ctx.spec = &eng->spec;
        eng->mm->on_prefill_done = backend_prefix_hook;
        eng->mm->on_prefill_done_user = eng;
        eng->mm->on_prefill_checkpoint = backend_prefix_checkpoint;
        eng->mm->on_prefill_checkpoint_user = eng;
        eng->mm->checkpoint_block = eng->pfx_block;
        eng->mm->smp = eng->smp;
        eng->max_new = opt->max_new_tokens > 0 ? opt->max_new_tokens : 512;
        CWICK_INFO("init", "mm engine ready: %s dim=%lld layers=%lld experts=%lld "
                  "top-%lld context=%lld",
                  eng->spec.model_type, (long long)eng->spec.hidden_size,
                  (long long)eng->spec.num_hidden_layers,
                  (long long)eng->spec.num_experts,
                  (long long)eng->spec.num_experts_per_token,
                  (long long)eng->capacity);
        *out = eng;
        return CWICK_OK;
    }

    /* ---- Qwen3.5 hybrid: its own loader and forward pass, same shape as the
     * two branches above. Dense BF16, so there is no expert cache to budget
     * and no stream to steer; the only sizing decision is the KV window, and
     * that is 32 KiB per position (8 attention layers of 4x256 keys and
     * values) plus a fixed 50 MiB of DeltaNet state. ---- */
    if (strncmp(eng->spec.model_type, "qwen3_5", 7) == 0) {
        eng->capacity = next_capacity(&eng->spec, opt->max_context);
        eng->q35 = cwick_xmalloc(sizeof(q35_engine));
        if (q35_engine_open(eng->q35, opt->model_root, eng->capacity, err) != 0) {
            free(eng->q35);
            eng->q35 = NULL;
            cwick_engine_free(eng);
            return CWICK_ERR_STATE;
        }
        /* The loader caps the window at the checkpoint's own maximum, so take
         * its answer rather than the request. */
        eng->capacity = eng->q35->capacity;
        eng->q35->ctx.stop = opt->stop;
        eng->q35->ctx.abort_poll = NULL;
        eng->q35->ctx.abort_poll_user = NULL;
        eng->q35->ctx.verbose = opt->verbose;
        eng->q35->ctx.tag = eng->req_tag;
        eng->q35->ctx.spec = &eng->spec;
        eng->q35->on_prefill_done = backend_prefix_hook;
        eng->q35->on_prefill_done_user = eng;
        eng->q35->on_prefill_checkpoint = backend_prefix_checkpoint;
        eng->q35->on_prefill_checkpoint_user = eng;
        eng->q35->checkpoint_block = eng->pfx_block;
        eng->q35->smp = eng->smp;
        eng->max_new = opt->max_new_tokens > 0 ? opt->max_new_tokens : 512;
        CWICK_INFO("init", "qwen3_5 engine ready: %s dim=%lld layers=%lld "
                   "(%lld linear + %lld full) context=%lld cache=%.2fGiB",
                   eng->spec.model_type, (long long)eng->spec.hidden_size,
                   (long long)eng->spec.num_hidden_layers,
                   (long long)eng->spec.kda_layers,
                   (long long)eng->spec.mla_layers,
                   (long long)eng->capacity,
                   cwick_gib(q35_cache_bytes(&eng->q35->m)));
        *out = eng;
        return CWICK_OK;
    }

    /* ---- size the inference state *before* the weights are opened ------
     * KV/KDA/anchor buffers are allocated once and live for the whole run,
     * so the expert-cache budget has to subtract them first. This is also
     * where the context window is decided. */
    {
        const cwick_model_spec *s = &eng->spec;
        eng->capacity = next_capacity(s, opt->max_context);
        if (opt->max_context <= 0 && s->quant == CWICK_QUANT_FP8_E4M3) {
            /* The released checkpoint: by default cap the KV window at the DSA
             * indexer's top-k (2048). Inside that window top-2048 selection
             * is the whole prefix, so full attention is exact — and it keeps
             * the KV cache at a size that still leaves RAM for experts. An
             * explicit --context overrides this so long-context prompts run. */
            if (eng->capacity > s->index_topk && s->index_topk > 0)
                eng->capacity = s->index_topk;
        }
        int64_t nslots = 0;
        for (int64_t i = 0; i < s->num_hidden_layers; i++)
            if (s->layers[i] == CWICK_LAYER_DSA || s->layers[i] == CWICK_LAYER_MLA)
                nslots++;
        const int64_t q_dim = s->qk_nope_head_dim + s->qk_rope_head_dim;
        const int64_t vdim = s->value_head_dim;
        const int64_t kstride = s->num_attention_heads * q_dim;
        const int64_t vstride = s->num_attention_heads * vdim;
        /* Cache layout is a property of the checkpoint, but kv_b_proj only
         * becomes visible once the spine is open -- which needs this number
         * first. Decide from the spec here and re-verify below; a mismatch
         * falls back to the expanded cache (see the allocation block). */
        eng->mla_latent = (q_dim == s->qk_nope_head_dim &&
                           s->kv_lora_rank > 0 && nslots > 0 &&
                           getenv("CWICK_MLA_EXPANDED") == NULL);
        const int64_t cache_stride = eng->mla_latent
                                   ? s->kv_lora_rank : (kstride + vstride);
        uint64_t state = 0;
        state += (uint64_t)nslots * (uint64_t)eng->capacity *
                 (uint64_t)cache_stride * sizeof(float);
        state += (uint64_t)s->num_hidden_layers * (uint64_t)s->linear_num_heads *
                 (uint64_t)s->linear_head_dim * (uint64_t)s->linear_head_dim *
                 sizeof(float);
        state += (uint64_t)s->num_hidden_layers * 3 *
                 (uint64_t)(s->linear_num_heads * s->linear_head_dim) * 3 *
                 sizeof(float);
        /* DSA indexer: one compressed 128-dim row per index_kpool positions,
         * plus the open-pool tail, per sparse-attention layer. */
        if (eng->mla_latent)
            state += (uint64_t)nslots *
                     cwick_indexer_bytes(s, eng->capacity);
        if (!s->mhc)
            state += (uint64_t)(s->num_hidden_layers / (s->attn_res_block_size > 0
                                ? s->attn_res_block_size : 1) + 1) *
                     (uint64_t)s->hidden_size * sizeof(float);
        /* working buffers are per-token and small, but they exist too */
        state += (uint64_t)s->vocab_size * sizeof(float);
        /* The MTP block's own latent cache and scratch, when speculation is
         * requested. Booked separately because it is allocated separately
         * (see the MTP block in engine_open, which also refuses to turn on
         * for a non-latent cache -- the same condition tested here). */
        if (s->num_nextn_predict_layers > 0 && eng->mla_latent &&
            cwick_env_true("CWICK_MTP"))
            state += (uint64_t)eng->capacity * (uint64_t)s->kv_lora_rank *
                     cwick_latent_elem_bytes() +
                     (uint64_t)(6 * s->hidden_size + s->vocab_size) * sizeof(float);
        cwick_memory_set_runtime_state(state);
        CWICK_VERB("init", "inference state booked: %.2f GiB (KV %lld slots x "
                  "%lld positions x %lld floats [%s cache], %lld KDA layers)",
                  cwick_gib(state), (long long)nslots, (long long)eng->capacity,
                  (long long)cache_stride,
                  eng->mla_latent ? "MLA latent" : "expanded K/V",
                  (long long)s->kda_layers);
    }

    const char *repr = opt->spine ? opt->spine : "bf16";
    if (opt->shared_spine) {
        /* Borrow the weights another slot already opened. The struct is all
         * pointers, so this shares the mmaps, the parsed shard index, the
         * F_NOCACHE stream and the expert cache -- and it must NOT re-wire
         * them (the owner did) nor close them on free. Everything below that
         * sizes or allocates per-sequence state is unaffected. */
        memcpy(&eng->spine, opt->shared_spine, sizeof(eng->spine));
        eng->spine_owned = false;
        CWICK_VERB("init", "slot %d: sharing the weight spine (%.2f GiB "
                          "resident, %.2f GiB expert cache) opened by slot 0",
                  eng->slot_id,
                  cwick_gib(eng->spine.resident_bytes),
                  cwick_gib(cwick_stream_cache_cap(eng->spine.stream)));
    } else {
        st = cwick_spine_open(opt->model_root, &eng->spec, repr,
                             &eng->spine, err);
        if (st != CWICK_OK) {
            cwick_engine_free(eng);
            return st;
        }
        eng->spine_owned = true;
    }

    const cwick_model_spec *s = &eng->spec;
    const int64_t L = s->num_hidden_layers;
    const int64_t q_dim = s->qk_nope_head_dim + s->qk_rope_head_dim;
    const int64_t vdim = s->value_head_dim;
    /* eng->capacity was decided above (before the spine opened) so the KV
     * could be counted in the memory budget. */
    eng->pos = 0;
    eng->n_anchors = 0;

    /* Only the sparse-attention layers hold a KV cache; the 34 KDA layers
     * hold a recurrent state instead. Allocating per layer would waste ~5x
     * the memory, so KV layers get compact slots. */
    eng->kv_slot = cwick_xmalloc((size_t)L * sizeof(int64_t));
    for (int64_t i = 0; i < L; i++) eng->kv_slot[i] = -1;
    {
        int64_t nslots = 0;
        for (int64_t i = 0; i < L; i++)
            if (s->layers[i] == CWICK_LAYER_DSA || s->layers[i] == CWICK_LAYER_MLA)
                eng->kv_slot[i] = nslots++;
        eng->kv_layers = nslots;
        const int64_t kstride = s->num_attention_heads * q_dim;
        const int64_t vstride = s->num_attention_heads * vdim;
        /* Confirm the layout against the actual kv_b_proj now that the spine
         * is mapped: only the per-query-head MLA layout has one latent to
         * absorb into. A GQA checkpoint (shared KV heads) keeps the expanded
         * cache rather than silently changing the maths. */
        if (eng->mla_latent && nslots > 0) {
            int64_t dl = -1;
            for (int64_t i = 0; i < L; i++)
                if (s->layers[i] == CWICK_LAYER_DSA || s->layers[i] == CWICK_LAYER_MLA) { dl = i; break; }
            eng->mla_latent = false;
            if (dl >= 0) {
                char nb[192];
                snprintf(nb, sizeof(nb), "layers.%lld.self_attn.kv_b_proj.weight", dl);
                int64_t bi = cwick_spine_find(&eng->spine, nb);
                if (bi != CWICK_TENSOR_MISSING)
                    eng->mla_latent = cwick_mla_latent_enabled(
                        s, &eng->spine.tensors[(size_t)bi]);
            }
        }
        if (nslots > 0) {
            if (eng->mla_latent) {
                /* The latent is the hottest thing in the step; at bf16 it is
                 * also half the bytes, and the KV window is the one place
                 * where that shows up as both RAM and bandwidth. */
                eng->keys = cwick_xmalloc((size_t)(nslots * eng->capacity *
                                                  s->kv_lora_rank) *
                                         cwick_latent_elem_bytes());
                eng->values = NULL;
            } else {
                eng->keys = cwick_xmalloc((size_t)(nslots * eng->capacity * kstride) * sizeof(float));
                eng->values = cwick_xmalloc((size_t)(nslots * eng->capacity * vstride) * sizeof(float));
            }
        } else {
            eng->keys = NULL;
            eng->values = NULL;
        }
        CWICK_INFO("init", "KV cache: %s layout, %lld slots x %lld positions = "
                  "%.2f GiB",
                  eng->mla_latent ? "MLA latent" : "expanded K/V",
                  (long long)nslots, (long long)eng->capacity,
                  cwick_gib((uint64_t)nslots * (uint64_t)eng->capacity *
                           (uint64_t)(eng->mla_latent ? s->kv_lora_rank
                                                      : (kstride + vstride)) *
                           (eng->mla_latent ? cwick_latent_elem_bytes()
                                            : sizeof(float))));
    }

    /* ---- DSA indexer state ------------------------------------------
     * One per sparse-attention layer: the compressed key cache, the open-pool
     * tail and the selection scratch. Wired only when every tensor the
     * indexer needs is present -- a checkpoint that lacks them (or a route
     * that cannot use them, e.g. the expanded cache) runs without one, which
     * is correct for any prefix inside index_topk. */
    {
        const int64_t hd = s->index_head_dim;
        const int64_t ih = s->index_n_heads;
        const int64_t kp = s->index_kpool > 0 ? s->index_kpool : 1;
        const int64_t topk = s->index_topk > 0 ? s->index_topk : 0;
        const bool usable = eng->mla_latent && hd > 0 && ih > 0 && topk > 0;
        eng->idx_state = NULL;
        eng->idx_w = NULL;
        if (usable && eng->kv_layers > 0) {
            eng->idx_w = cwick_xmalloc((size_t)eng->kv_layers *
                                      sizeof(cwick_indexer_weights));
            eng->idx_state = cwick_xmalloc((size_t)eng->kv_layers *
                                          sizeof(cwick_indexer_state));
            memset(eng->idx_w, 0,
                   (size_t)eng->kv_layers * sizeof(cwick_indexer_weights));
            memset(eng->idx_state, 0,
                   (size_t)eng->kv_layers * sizeof(cwick_indexer_state));
            for (int64_t i = 0; i < eng->kv_layers; i++)
                eng->idx_w[i].wq_b = NULL;

            const int64_t pools = eng->capacity / kp + 2;
            bool complete = true;
            for (int64_t i = 0; i < L && complete; i++) {
                if (eng->kv_slot[i] < 0) continue;
                cwick_indexer_weights *w = &eng->idx_w[eng->kv_slot[i]];
                char nb[192];
#define IDX_FIND(field, suffix) do { \
                snprintf(nb, sizeof(nb), "layers.%lld.self_attn.indexer.%s", i, suffix); \
                int64_t _i = cwick_spine_find(&eng->spine, nb); \
                if (_i == CWICK_TENSOR_MISSING) { complete = false; break; } \
                w->field = &eng->spine.tensors[(size_t)_i]; \
            } while (0)
                IDX_FIND(wq_b, "wq_b.weight");
                IDX_FIND(wk, "wk.weight");
                IDX_FIND(gate, "index_kpool_compress_gate");
                IDX_FIND(ape, "index_kpool_compress_ape");
                IDX_FIND(k_norm, "k_norm.weight");
                IDX_FIND(k_norm_b, "k_norm.bias");
                IDX_FIND(weights_proj, "weights_proj.weight");
#undef IDX_FIND
            }
            if (!complete) {
                free(eng->idx_w);
                free(eng->idx_state);
                eng->idx_w = NULL;
                eng->idx_state = NULL;
                CWICK_INFO("init", "DSA indexer unavailable (incomplete weight "
                          "set): sparse attention is exact only up to "
                          "index_topk=%lld", (long long)topk);
            } else {
                for (int64_t i = 0; i < eng->kv_layers; i++) {
                    cwick_indexer_state *stt = &eng->idx_state[i];
                    stt->pool_cap = pools;
                    stt->n_pools = 0;
                    stt->pool = cwick_xmalloc((size_t)pools * (size_t)hd * sizeof(float));
                    stt->tail_k = cwick_xmalloc((size_t)kp * (size_t)hd * sizeof(float));
                    stt->tail_g = cwick_xmalloc((size_t)kp * (size_t)hd * sizeof(float));
                    stt->sel = cwick_xmalloc((size_t)(topk + kp + 2) * sizeof(int64_t));
                    stt->q = cwick_xmalloc((size_t)(ih * hd) * sizeof(float));
                    stt->wt = cwick_xmalloc((size_t)ih * sizeof(float));
                    stt->sc = cwick_xmalloc((size_t)pools * sizeof(float));
                    stt->sc2 = cwick_xmalloc((size_t)pools * sizeof(float));
                    cwick_indexer_reset(s, stt);
                }
                CWICK_INFO("init", "DSA indexer ready: top-%lld of %lld pools "
                          "(kpool %lld, %lld heads x %lld dims), %.2f GiB",
                          (long long)topk, (long long)pools, (long long)kp,
                          (long long)ih, (long long)hd,
                          cwick_gib((uint64_t)eng->kv_layers *
                                   cwick_indexer_bytes(s, eng->capacity)));
            }
        }
    }

    eng->kda_rec = cwick_xmalloc((size_t)(L * s->linear_num_heads * s->linear_head_dim *
                                        s->linear_head_dim) * sizeof(float));
    eng->kda_conv = cwick_xmalloc((size_t)(L * 3 * (s->linear_num_heads * s->linear_head_dim) *
                                         3) * sizeof(float));
    if (s->mhc) {
        eng->max_anchors = 0;
        eng->anchors = NULL;
    } else {
        eng->max_anchors = L / s->attn_res_block_size + 1;
        eng->anchors = cwick_xmalloc((size_t)eng->max_anchors * s->hidden_size * sizeof(float));
    }

    /* ---- MTP draft buffers (opt-in via CWICK_MTP=1) -------------------
     * Deliberately outside the kv_slot/idx_state arrays: those are sized and
     * indexed by the depth-45 schedule, and threading a 46th entry through
     * them would touch the KV allocation, the memory plan and the prefix
     * snapshot for every run, including the ones that never speculate. The
     * block gets its own latent cache and its own slot 0 instead; the only
     * shared thing is `capacity`. */
    if (s->num_nextn_predict_layers > 0 && cwick_env_true("CWICK_MTP")) {
        char probe[192];
        snprintf(probe, sizeof(probe), "layers.%lld.eh_proj.weight", (long long)L);
        const bool checkpoint_has_block =
            cwick_spine_find(&eng->spine, probe) != CWICK_TENSOR_MISSING;
        if (!checkpoint_has_block) {
            CWICK_INFO("init", "MTP: no nextn block in this checkpoint; "
                              "drafting stays off");
        } else if (!eng->mla_latent) {
            /* The MTP block's attention needs a latent KV like every other
             * DSA layer; the expanded layout has no wiring for it here. */
            CWICK_INFO("init", "MTP: needs the MLA latent cache; drafting "
                              "stays off");
        } else {
            const int64_t stride = s->kv_lora_rank;
            const size_t kb = (size_t)eng->capacity * (size_t)stride *
                              cwick_latent_elem_bytes();
            eng->mtp_h = cwick_xmalloc((size_t)s->hidden_size * sizeof(float));
            eng->mtp_work = cwick_xmalloc((size_t)(6 * s->hidden_size) * sizeof(float));
            eng->mtp_logits = cwick_xmalloc((size_t)s->vocab_size * sizeof(float));
            eng->mtp_keys = cwick_xmalloc(kb);
            memset(eng->mtp_keys, 0, kb);
            eng->mtp_on = true;
            CWICK_INFO("init", "MTP draft on: nextn block at layer %lld body, "
                      "%.0f MiB own KV, %.0f MiB scratch",
                      (long long)L, (double)kb / 1048576.0,
                      (double)(6 * s->hidden_size + s->vocab_size) * 4.0 / 1048576.0);
        }
    }

    /* ---- keep the inference state uncompressed too -------------------
     * cwick_spine_open() wires the weight spine; these buffers are the other
     * long-lived pages every token touches. Left unwired they are ordinary
     * anonymous memory, and the prefill's streaming churn (hundreds of GiB
     * through RAM) gives the compressor every reason to take them: a serve
     * log shows RSS dropping by exactly the KV size (2.1 GiB) mid-prefill
     * while the footprint stays flat — compressed KV, faulted back one page
     * at a time. Wiring them costs 2.2 GiB of the 102 GiB budget and makes
     * the "no-swap mode" promise true for the whole working set. */
    if (cwick_memory_noswap()) {
        const size_t fsz = sizeof(float);
        size_t failed = 0, locked = 0;
        uint64_t bytes = 0;
        if (eng->keys) {
            bytes = (uint64_t)eng->kv_layers * (uint64_t)eng->capacity *
                    (uint64_t)(eng->mla_latent ? s->kv_lora_rank
                                               : s->num_attention_heads * q_dim) *
                    (eng->mla_latent ? cwick_latent_elem_bytes() : fsz);
            if (cwick_mem_lock_range(eng->keys, (size_t)bytes, "KV cache")) locked++;
            else failed++;
        }
        if (eng->values) {
            bytes = (uint64_t)eng->kv_layers * (uint64_t)eng->capacity *
                    (uint64_t)(s->num_attention_heads * vdim) * fsz;
            if (cwick_mem_lock_range(eng->values, (size_t)bytes, "KV values")) locked++;
            else failed++;
        }
        if (eng->kda_rec) {
            bytes = (uint64_t)L * (uint64_t)s->linear_num_heads *
                    (uint64_t)s->linear_head_dim * (uint64_t)s->linear_head_dim * fsz;
            if (cwick_mem_lock_range(eng->kda_rec, (size_t)bytes, "KDA state")) locked++;
            else failed++;
        }
        if (eng->kda_conv) {
            bytes = (uint64_t)L * 3u * (uint64_t)(s->linear_num_heads *
                                                  s->linear_head_dim) * 3u * fsz;
            if (cwick_mem_lock_range(eng->kda_conv, (size_t)bytes, "KDA conv")) locked++;
            else failed++;
        }
        if (eng->anchors) {
            bytes = (uint64_t)eng->max_anchors * (uint64_t)s->hidden_size * fsz;
            if (cwick_mem_lock_range(eng->anchors, (size_t)bytes, "residual anchors")) locked++;
            else failed++;
        }
        for (int64_t i = 0; eng->idx_state && i < eng->kv_layers; i++) {
            cwick_indexer_state *stt = &eng->idx_state[i];
            const int64_t hd = s->index_head_dim;
            const int64_t kp = s->index_kpool > 0 ? s->index_kpool : 1;
            const int64_t topk = s->index_topk > 0 ? s->index_topk : 0;
            struct { void *p; size_t n; const char *what; } parts[] = {
                { stt->pool,   (size_t)stt->pool_cap * (size_t)hd * fsz, "indexer pools" },
                { stt->tail_k, (size_t)kp * (size_t)hd * fsz,             "indexer tail K" },
                { stt->tail_g, (size_t)kp * (size_t)hd * fsz,             "indexer tail G" },
                { stt->sel,    (size_t)(topk + kp + 2) * sizeof(int64_t), "indexer selection" },
                { stt->q,      (size_t)s->index_n_heads * (size_t)hd * fsz, "indexer query" },
                { stt->wt,     (size_t)s->index_n_heads * fsz,            "indexer weights" },
                { stt->sc,     (size_t)stt->pool_cap * fsz,               "indexer scores" },
                { stt->sc2,    (size_t)stt->pool_cap * fsz,               "indexer scores 2" },
            };
            for (size_t k = 0; k < sizeof(parts) / sizeof(parts[0]); k++)
                if (parts[k].p && parts[k].n) {
                    if (cwick_mem_lock_range(parts[k].p, parts[k].n, parts[k].what)) locked++;
                    else failed++;
                }
        }
        if (failed == 0)
            CWICK_INFO("init", "inference state wired: KV cache, DSA indexer and "
                              "KDA state held incompressible and unswapped "
                              "(%zu ranges)", locked);
        else
            CWICK_INFO("init", "inference state wiring failed for %zu of %zu "
                              "ranges — the VM may compress part of the KV "
                              "cache (--allow-swap makes this the expected "
                              "behaviour)", failed, failed + locked);
    }

    /* ---- the memory plan decided when the spine was opened -----------
     * The checkpoint is far larger than RAM, so "loaded" can only mean the
     * part that must be resident is resident and the rest is streamed; the
     * plan proved at spike time that resident + state fit under at most 80%
     * of host RAM. From here on the decode loop steers real RSS toward that
     * ceiling (see maybe_steer_memory). */
    if (eng->spine.is_safetensors) {
        eng->streamed_per_token = 0;
        {
            /* experts are re-read every decode step: sparse layers x top-k */
            int64_t sets = 0;
            uint64_t bank = eng->spine.streamed_bytes;
            for (int64_t l = 0; l < L; l++)
                if (!s->mlp_dense[l]) sets++;
            if (sets > 0 && s->num_experts > 0) {
                uint64_t per_set = bank / (uint64_t)(sets * s->num_experts + s->num_experts);
                eng->streamed_per_token = per_set * (uint64_t)sets *
                                          (uint64_t)s->num_experts_per_token;
            }
        }
        const cwick_mem_plan *plan = cwick_spine_memory_plan(&eng->spine);
        if (plan) eng->rss_target = plan->target_bytes;

        CWICK_INFO("mem", "weights resident %.2f GiB, %.2f GiB of experts "
                  "re-read per decode step",
                  cwick_gib(eng->spine.resident_bytes),
                  cwick_gib(eng->streamed_per_token));
        CWICK_VERB("mem", "expert pin set holds up to %.2f GiB; the other "
                  "%.2f GiB stay on SSD",
                  cwick_gib(cwick_stream_cache_cap(eng->spine.stream)),
                  cwick_gib(eng->spine.streamed_bytes));
        if (eng->rss_target > 0)
            CWICK_VERB("mem", "steering RSS toward %.2f GiB (%.0f%% of %.2f GiB)",
                      cwick_gib(eng->rss_target),
                      plan ? plan->host_fraction * 100.0 : 0.0,
                      cwick_gib(plan ? plan->host_bytes : 0));
        CWICK_INFO("stor", "F_NOCACHE expert streaming %s",
                  cwick_stream_nocache_active(eng->spine.stream)
                      ? "active (streamed bytes are not retained in RAM)"
                      : "inactive (kernel may retain streamed bytes)");
    }

    CWICK_INFO("init", "engine ready: %s hidden=%lld layers=%lld experts=%lld "
              "top-%lld context=%lld",
              s->model_type, (long long)s->hidden_size, (long long)L,
              (long long)s->num_experts, (long long)s->num_experts_per_token,
              (long long)eng->capacity);
    *out = eng;
    return CWICK_OK;
}

const cwick_model_spec *cwick_engine_spec(const cwick_engine *eng) { return &eng->spec; }

int64_t cwick_engine_context_window(const cwick_engine *eng) {
    return eng ? eng->capacity : 0;
}
const cwick_spine *cwick_engine_spine(const cwick_engine *eng) { return &eng->spine; }

/* The expert pin cache differs by backend: GLM keeps it on the spine, the
 * DeepSeek port owns its own stream. /metrics reads whichever is live, so a
 * cache scrape is not silently empty after a model switch. */
cwick_stream *cwick_engine_expert_stream(const cwick_engine *eng) {
    if (!eng) return NULL;
    if (eng->ds) return eng->ds->m.stream;
    if (eng->mm) return eng->mm->m.stream;
    return eng->spine.stream;
}
const char *cwick_engine_model_root(const cwick_engine *eng) { return eng->model_root; }
int64_t cwick_engine_capacity(const cwick_engine *eng) { return eng->capacity; }
int64_t cwick_engine_pos(const cwick_engine *eng) {
    if (!eng) return 0;
    if (eng->ds) return eng->ds->pos;
    if (eng->mm) return eng->mm->pos;
    if (eng->q35) return eng->q35->pos;
    return eng->pos;
}

/* Reset to a fresh conversation: clear positions, anchors and all KV/KDA
 * state. The spine and weights stay mapped. */
cwick_status cwick_engine_reset(cwick_engine *eng, cwick_error *err) {
    cwick_error_clear(err);
    if (!eng) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "reset: no engine");
        return CWICK_ERR_ARGUMENT;
    }
    if (eng->ds) {
        ds_model_reset(&eng->ds->m);
        eng->ds->pos = 0;
        eng->pos = 0;
        return CWICK_OK;
    }
    if (eng->mm) {
        mm_model_reset(&eng->mm->m);
        eng->mm->pos = 0;
        eng->pos = 0;
        return CWICK_OK;
    }
    if (eng->q35) {
        q35_engine_reset(eng->q35);
        eng->pos = 0;
        return CWICK_OK;
    }
    const cwick_model_spec *s = &eng->spec;
    const int64_t L = s->num_hidden_layers;
    const int64_t q_dim = s->qk_nope_head_dim + s->qk_rope_head_dim;
    const int64_t vdim = s->value_head_dim;
    /* KV/KDA state exists for both the released checkpoint and the converted
     * models now; only the legacy anchor bank is mHC-free. */
    {
        const int64_t kstride = s->num_attention_heads * q_dim;
        const int64_t vstride = s->num_attention_heads * vdim;
        if (eng->keys)
            memset(eng->keys, 0,
                   (size_t)(eng->kv_layers * eng->capacity *
                            (eng->mla_latent ? s->kv_lora_rank : kstride))
                   * (eng->mla_latent ? cwick_latent_elem_bytes()
                                      : sizeof(float)));
        if (eng->values)
            memset(eng->values, 0,
                   (size_t)(eng->kv_layers * eng->capacity * vstride) * sizeof(float));
        if (eng->kda_rec)
            memset(eng->kda_rec, 0,
                   (size_t)(L * s->linear_num_heads * s->linear_head_dim *
                            s->linear_head_dim) * sizeof(float));
        if (eng->kda_conv)
            memset(eng->kda_conv, 0,
                   (size_t)(L * 3 * (s->linear_num_heads * s->linear_head_dim) * 3)
                   * sizeof(float));
        if (eng->anchors)
            memset(eng->anchors, 0,
                   (size_t)eng->max_anchors * s->hidden_size * sizeof(float));
        for (int64_t i = 0; i < eng->kv_layers; i++)
            if (eng->idx_state) cwick_indexer_reset(s, &eng->idx_state[i]);
    }
    eng->pos = 0;
    eng->n_anchors = 0;
    return CWICK_OK;
}

void cwick_engine_free(cwick_engine *eng) {
    if (!eng) return;
    if (eng->ds) {
        ds_engine_close(eng->ds);
        free(eng->ds);
        eng->ds = NULL;
        cwick_model_spec_free(&eng->spec);
        free(eng->model_root);
        free(eng);
        return;
    }
    if (eng->mm) {
        mm_engine_close(eng->mm);
        free(eng->mm);
        eng->mm = NULL;
        cwick_model_spec_free(&eng->spec);
        free(eng->model_root);
        free(eng);
        return;
    }
    if (eng->q35) {
        q35_engine_close(eng->q35);
        free(eng->q35);
        eng->q35 = NULL;
        cwick_model_spec_free(&eng->spec);
        free(eng->model_root);
        free(eng);
        return;
    }
    free(eng->anchors);
    free(eng->kda_conv);
    free(eng->kda_rec);
    free(eng->values);
    free(eng->keys);
    if (eng->idx_state) {
        for (int64_t i = 0; i < eng->kv_layers; i++) {
            free(eng->idx_state[i].pool);
            free(eng->idx_state[i].tail_k);
            free(eng->idx_state[i].tail_g);
            free(eng->idx_state[i].sel);
            free(eng->idx_state[i].q);
            free(eng->idx_state[i].wt);
            free(eng->idx_state[i].sc);
            free(eng->idx_state[i].sc2);
        }
        free(eng->idx_state);
    }
    free(eng->idx_w);
    free(eng->kv_slot);
    if (eng->spine_owned) {
        /* The no-copy views are MTLBuffers over the mapped weights, and the
         * close below unmaps them. Drop the views first: a view left pointing
         * at an unmapped range is worse than a leaked one. */
        cwick_metal_views_release();
        cwick_spine_close(&eng->spine);  /* joins the prefetch thread, closes
                                          * shard fds, returns cache blobs */
    }
    /* A slot that borrowed the weights must not also close them: the owner
     * slot does that once, after every borrower is gone. */
    cwick_model_spec_free(&eng->spec);
    free(eng->model_root);
    free(eng);
    /* The keepalive thread dispatches on the Metal queue, so it has to be
     * joined before the device goes away. */
    cwick_metal_keepalive_stop();
    /* Last: drop the Metal device, queue, pipeline and weight arenas. It has
     * to come after the spine is closed, because every cached expert blob
     * still points into an arena. */
    cwick_metal_cleanup();
}

/* ------------------------------------------------------------------ */
/* Generation (greedy by default, sampled when the checkpoint says so)  */
/* ------------------------------------------------------------------ */
void cwick_engine_sampler_default(cwick_sampler_cfg *out) {
    cwick_sampler_cfg_default(out);
}

bool cwick_engine_sampler_checkpoint(const cwick_engine *eng,
                                     cwick_sampler_cfg *out) {
    if (!out) return false;
    if (eng) {
        *out = eng->smp_ckpt;
        return eng->smp_ckpt_have;
    }
    cwick_sampler_cfg_default(out);
    return false;
}

void cwick_engine_set_sampler(cwick_engine *eng, const cwick_sampler_cfg *cfg) {
    if (!eng) return;
    if (cfg) {
        eng->smp = *cfg;
        /* Operator kill switch: a sampling regression in production has to be
         * answerable without a rebuild. */
        if (cwick_env_true("CWICK_GREEDY")) cwick_sampler_cfg_default(&eng->smp);
    } else {
        cwick_sampler_cfg_default(&eng->smp);
    }
    /* The DeepSeek backend runs its own decode loop; hand it the same config
     * so the two backends cannot disagree about what a request asked for. */
    if (eng->ds) eng->ds->smp = eng->smp;
    if (eng->mm) eng->mm->smp = eng->smp;
}

/* The seed for a generation. CWICK_SEED pins it (an A/B that wants the same
 * draw twice needs this; without it every run draws differently, which is the
 * OpenAI reading of a request that sent no seed). Cached: getenv per step
 * would be a system call in the hottest loop in the program. */
static uint64_t sampler_seed(void) {
    static uint64_t cached = 0;
    static bool read = false;
    if (!read) {
        read = true;
        const char *e = getenv("CWICK_SEED");
        if (e && *e) cached = strtoull(e, NULL, 0);
    }
    return cached;
}

/* One sampler per generation: the token counts that drive the repetition
 * penalties belong to this sequence, not to the engine, and the seed must be
 * per-run so two slots do not share a stream. */
static void sampler_begin(cwick_sampler *s, const cwick_engine *eng,
                          const uint32_t *prompt, int64_t prompt_len,
                          uint64_t seed) {
    cwick_sampler_init(s, &eng->smp, eng->spec.vocab_size, seed);
    /* The prompt counts too (vLLM and HF both penalise the whole visible
     * text): a system block repeated three times in the prompt is exactly the
     * pattern the model then copies into the answer. Primed, not accepted —
     * the loop guard watches what this run produces, not what it was given. */
    if (s->counts)
        for (int64_t i = 0; i < prompt_len; i++) cwick_sampler_prime(s, prompt[i]);
}

/* Kept for the debug window below, which prints the top of the distribution
 * without going through the sampler (it runs before the token is chosen and
 * must not consume a draw). */
static uint32_t argmax(const float *logits, int64_t n) {
    uint32_t best = 0;
    float bestv = logits[0];
    for (int64_t i = 1; i < n; i++) {
        if (logits[i] > bestv) { bestv = logits[i]; best = (uint32_t)i; }
    }
    return best;
}

/* Keep the promise made by the memory plan. A pinned expert cache is the only
 * elastic thing in this process: everything else (resident spine, KV/KDA,
 * arenas) is fixed at load time, so when real RSS creeps past the ceiling the
 * honest fix is to give back expert bytes and re-read them instead of letting
 * macOS compress a 306 GiB working set. */
static void maybe_steer_memory(cwick_engine *eng, int64_t step) {
    if (!eng || eng->rss_target == 0) return;
    cwick_stream_steer_memory(eng->spine.stream, eng->rss_target, step, "mem");
}

/* Periodic progress line. On a 320B model a single prefill token streams
 * ~7.9 GiB of experts, so silence for minutes is indistinguishable from a
 * hang: this prints where the run is and what it costs, every few steps.
 *
 * `layer >= 0` marks a layer-granular call from inside a batched prefill
 * chunk. Those print unconditionally when reached: the chunk is one atomic
 * sweep of all layers, so token counts cannot advance inside it, and without
 * a layer counter a prompt that fits in a single chunk prints nothing at all
 * for its entire (minute-long) duration.
 *
 * `done == total` also always prints, so a run whose length is not a multiple
 * of CWICK_PROGRESS_EVERY still reports its final state. */
static void maybe_progress_at(cwick_engine *eng, const char *phase, int64_t done,
                              int64_t total, int64_t layer, int64_t nlayer) {
    /* `total <= 0` means "no meaningful denominator": the decode budget was
     * clamped to whatever is left of the KV window, so it is a capacity
     * ceiling rather than a requested length. Printing "1/99884" there tells
     * the operator nothing about how far along the answer is. */
    bool due = (total > 0) ? ((done % CWICK_PROGRESS_EVERY) == 0 || done == total)
                           : ((done % CWICK_PROGRESS_EVERY) == 0);
    if (layer >= 0) {
        /* Layer-granular call from inside a batched prefill chunk: print on
         * the every-8th-layer cadence *or* when the last line is older than
         * the interval, whichever comes first, so a slow layer still reports. */
        double tnow = now_ms();
        bool every_nth = ((layer - 1) % CWICK_PROGRESS_EVERY) == 0;
        if (!every_nth && (tnow - g_progress_ms) < progress_min_ms()) return;
        g_progress_ms = tnow;
        cwick_metrics_set_prefill(done, total, layer, nlayer);
    } else {
        if (!due) return;
        g_progress_ms = now_ms();
    }
    /* Report the footprint (resident + compressed) next to the RSS: they only
     * diverge when macOS is compressing the pinned expert cache, which is the
     * one thing the memory plan promises not to let happen. */
    char tail[320];
    progress_mem_tail(eng, tail, sizeof(tail));
    /* A decode line says which part of the answer it is: "decode(thinking)",
     * "decode(content)", "decode(tools)". A prefill line has no such split. */
    char phbuf[48];
    const char *ph = phase;
    if (strcmp(phase, "decode") == 0) {
        decode_phase_label(eng, phbuf, sizeof(phbuf));
        ph = phbuf;
    }
    if (layer >= 0 && nlayer > 0)
        CWICK_INFO(eng_tag(eng), "%s %lld/%lld tokens (%.1f%%), layer %lld/%lld — %s",
                  ph, (long long)done, (long long)total,
                  total > 0 ? 100.0 * (double)done / (double)total : 0.0,
                  (long long)layer, (long long)nlayer, tail);
    else if (total > 0)
        CWICK_INFO(eng_tag(eng), "%s %lld/%lld tokens (%.1f%%) — %s",
                  ph, (long long)done, (long long)total,
                  100.0 * (double)done / (double)total, tail);
    else
        CWICK_INFO(eng_tag(eng), "%s %lld tokens (uncapped) — %s",
                  ph, (long long)done, tail);
}

static void maybe_progress(cwick_engine *eng, const char *phase, int64_t done,
                           int64_t total) {
    maybe_progress_at(eng, phase, done, total, -1, 0);
}

/* Run the prompt through the forward pass, leaving the next-token logits in
 * `logits`. Default is the layer-major batched sweep (forward_prefill_mhc for
 * the released mHC checkpoint, forward_prefill for converted models); setting
 * CWICK_NO_BATCH_PREFILL=1 forces the token-serial reference. */
static cwick_status engine_prefill(cwick_engine *eng, const uint32_t *prompt,
                                  int64_t prompt_len, float *logits,
                                  cwick_error *err) {
    if (cwick_env_true("CWICK_NO_BATCH_PREFILL")) {
        for (int64_t i = 0; i < prompt_len; i++) {
            cwick_status st = forward_token(eng, prompt[i], logits, err);
            if (st != CWICK_OK) return st;
            eng->pos++;
            cwick_stream_expert_decay(eng->spine.stream);
            maybe_steer_memory(eng, i);
            if (engine_stopped(eng)) {
                cwick_error_set(err, CWICK_ERR_STATE, "%s", engine_interrupt_message(eng));
                return CWICK_ERR_STATE;
            }
            maybe_progress(eng, "prefill", i + 1, prompt_len);
        }
        return CWICK_OK;
    }
    if (eng->spec.mhc && eng->spec.hc_mult > 1)
        return forward_prefill_mhc(eng, prompt, prompt_len, logits, err);
    return forward_prefill(eng, prompt, prompt_len, logits, err);
}

/* ------------------------------------------------------------------ */
/* Prefix capture / restore (see pfx.h)                                 */
/* ------------------------------------------------------------------ */
/* Everything a prefix cache has to carry is here, and `struct cwick_engine`
 * is private to this file, so the engine side of the cache lives with it.
 *
 * The list is short because the mHC state is *intra-token*: layer 0 expands
 * the embedding and every later layer mixes the previous layer's output for
 * the same token, so nothing in it survives a token boundary. What does
 * survive is the KV/latent rows, the two DeltaNet states and the DSA indexer
 * pools -- and those are exactly the things that make a 30K-token prefill
 * cost hours. */
cwick_status cwick_engine_capture_prefix(cwick_engine *eng, int64_t pos,
                                       cwick_prefix_state *out, cwick_error *err) {
    cwick_error_clear(err);
    if (!eng || !out || pos < 0 || pos > cwick_engine_pos(eng)) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                       "prefix: capture position %lld outside 0..%lld",
                       (long long)pos,
                       (long long)(eng ? cwick_engine_pos(eng) : 0));
        return CWICK_ERR_ARGUMENT;
    }
    if (eng->mm) {
        /* Same opaque-blob contract as the DeepSeek branch, different kind: a
         * snapshot taken by one backend must never be applied to another, and
         * the file header carries this number. */
        free(out->blob);
        out->blob = NULL;
        out->blob_len = mm_state_bytes(&eng->mm->m, pos);
        out->kind = 2;
        out->pos = pos;
        out->blob = cwick_xmalloc(out->blob_len);
        if (!out->blob ||
            mm_state_pack(&eng->mm->m, pos, out->blob, out->blob_len) != 0) {
            free(out->blob);
            out->blob = NULL;
            out->blob_len = 0;
            cwick_error_set(err, CWICK_ERR_MEMORY,
                            "prefix: MiniMax snapshot at %lld tokens failed",
                            (long long)pos);
            return CWICK_ERR_MEMORY;
        }
        return CWICK_OK;
    }
    if (eng->q35) {
        /* Qwen3.5 keeps the recurrence state of 24 layers plus the KV of 8,
         * so it travels as one opaque blob with kind 3 -- the same contract
         * as DeepSeek and MiniMax, and the file header carries the number so
         * a snapshot can only be applied by the backend that wrote it. */
        free(out->blob);
        out->blob = NULL;
        out->blob_len = q35_prefix_bytes(&eng->q35->m, pos);
        out->kind = 3;
        out->pos = pos;
        out->blob = cwick_xmalloc(out->blob_len);
        if (!out->blob ||
            q35_prefix_pack(&eng->q35->m, pos, out->blob, out->blob_len) != 0) {
            free(out->blob);
            out->blob = NULL;
            out->blob_len = 0;
            cwick_error_set(err, CWICK_ERR_MEMORY,
                            "prefix: Qwen3.5 snapshot at %lld tokens failed",
                            (long long)pos);
            return CWICK_ERR_MEMORY;
        }
        return CWICK_OK;
    }
    if (eng->ds) {
        /* The DeepSeek state is a different set of buffers, so it travels as
         * one opaque blob with kind 1 in both the struct and the file header.
         * Reusing a state buffer frees the previous blob first. */
        free(out->blob);
        out->blob = NULL;
        out->blob_len = ds_state_bytes(&eng->ds->m, pos);
        out->kind = 1;
        out->pos = pos;
        out->blob = cwick_xmalloc(out->blob_len);
        if (!out->blob ||
            ds_state_pack(&eng->ds->m, pos, out->blob, out->blob_len) != 0) {
            free(out->blob);
            out->blob = NULL;
            out->blob_len = 0;
            cwick_error_set(err, CWICK_ERR_MEMORY,
                            "prefix: DeepSeek snapshot at %lld tokens failed",
                            (long long)pos);
            return CWICK_ERR_MEMORY;
        }
        return CWICK_OK;
    }
    const cwick_model_spec *sp = &eng->spec;
    const int64_t nslots = eng->kv_layers;
    const int64_t lat_stride = eng->mla_latent
        ? sp->kv_lora_rank
        : sp->num_attention_heads * (sp->qk_nope_head_dim + sp->qk_rope_head_dim);
    const int64_t v_stride = eng->mla_latent
        ? 0
        : sp->num_attention_heads * sp->value_head_dim;
    const int64_t kpool = sp->index_kpool > 0 ? sp->index_kpool : 1;
    const int64_t hd = sp->index_head_dim > 0 ? sp->index_head_dim : 0;
    const int64_t pool_cap = eng->idx_state ? eng->idx_state[0].pool_cap : 0;

    if (!cwick_prefix_state_alloc_dims(out, pos, nslots, lat_stride, v_stride,
                                      sp->num_hidden_layers, sp->linear_num_heads,
                                      sp->linear_head_dim, kpool, hd, pool_cap,
                                      sp->hidden_size, eng->max_anchors)) {
        cwick_prefix_state_free(out);
        cwick_error_set(err, CWICK_ERR_MEMORY, "prefix: out of memory");
        return CWICK_ERR_MEMORY;
    }
    out->n_anchors = eng->n_anchors;

    const size_t S = (size_t)(nslots > 0 ? nslots : 0);
    const size_t Pz = (size_t)pos;
    const size_t stride = (size_t)lat_stride;
    /* A snapshot is always stored fp32 even when the live cache is bf16: the
     * file format then does not depend on a build-time precision switch, and
     * a snapshot written by one setting can be restored by the other. The
     * conversion is one pass over the prefix rows, which is small against the
     * prefill it replaces. */
    {
        const bool lbf = cwick_latent_bf16_enabled();
        const uint16_t *src16 = (const uint16_t *)eng->keys;
        const float *src32 = eng->keys;
        for (size_t s = 0; s < S; s++) {
            const size_t from = (size_t)s * (size_t)eng->capacity * stride;
            float *dst = out->keys + s * Pz * stride;
            if (lbf)
                for (size_t i = 0; i < Pz * stride; i++)
                    dst[i] = cwick_bf16_fast(src16[from + i]);
            else
                memcpy(dst, src32 + from, Pz * stride * sizeof(float));
        }
    }
    if (v_stride > 0 && eng->values) {
        const size_t vs = (size_t)v_stride;
        for (size_t s = 0; s < S; s++)
            memcpy(out->values + s * Pz * vs,
                   eng->values + (size_t)s * (size_t)eng->capacity * vs,
                   Pz * vs * sizeof(float));
    }
    const size_t rec = (size_t)sp->num_hidden_layers *
                       (size_t)sp->linear_num_heads *
                       (size_t)sp->linear_head_dim * (size_t)sp->linear_head_dim;
    if (eng->kda_rec) memcpy(out->kda_rec, eng->kda_rec, rec * sizeof(float));
    const size_t conv = (size_t)sp->num_hidden_layers * 3 *
                        ((size_t)sp->linear_num_heads * (size_t)sp->linear_head_dim) * 3;
    if (eng->kda_conv) memcpy(out->kda_conv, eng->kda_conv, conv * sizeof(float));

    if (eng->idx_state) {
        for (size_t s = 0; s < S; s++) {
            const cwick_indexer_state *is = &eng->idx_state[s];
            out->idx_npools[s] = is->n_pools;
            const size_t np = (size_t)(is->n_pools > 0 ? is->n_pools : 0);
            if (np && is->pool)
                memcpy(out->idx_pool + s * (size_t)pool_cap * (size_t)hd,
                       is->pool, np * (size_t)hd * sizeof(float));
            if (is->tail_k)
                memcpy(out->idx_tail + s * 2 * (size_t)kpool * (size_t)hd,
                       is->tail_k, (size_t)kpool * (size_t)hd * sizeof(float));
            if (is->tail_g)
                memcpy(out->idx_tail + s * 2 * (size_t)kpool * (size_t)hd +
                       (size_t)kpool * (size_t)hd,
                       is->tail_g, (size_t)kpool * (size_t)hd * sizeof(float));
        }
    }
    if (eng->anchors && eng->max_anchors > 0)
        memcpy(out->anchors, eng->anchors,
               (size_t)eng->max_anchors * (size_t)sp->hidden_size * sizeof(float));
    return CWICK_OK;
}

cwick_status cwick_engine_apply_prefix(cwick_engine *eng, const cwick_prefix_state *s,
                                     cwick_error *err) {
    cwick_error_clear(err);
    if (!eng || !s) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "prefix: bad state");
        return CWICK_ERR_ARGUMENT;
    }
    if (eng->mm) {
        if (s->kind != 2 || !s->blob || s->blob_len == 0) {
            cwick_error_set(err, CWICK_ERR_STATE,
                            "prefix: state was not captured by this backend");
            return CWICK_ERR_STATE;
        }
        int64_t pos = 0;
        if (mm_state_apply(&eng->mm->m, s->blob, s->blob_len, &pos) != 0) {
            cwick_error_set(err, CWICK_ERR_STATE,
                            "prefix: MiniMax snapshot does not match this "
                            "checkpoint");
            return CWICK_ERR_STATE;
        }
        eng->mm->pos = pos;
        if (eng->pos < pos) eng->pos = pos;
        return CWICK_OK;
    }
    if (eng->ds) {
        if (s->kind != 1 || !s->blob || s->blob_len == 0) {
            cwick_error_set(err, CWICK_ERR_STATE,
                            "prefix: state was not captured by this backend");
            return CWICK_ERR_STATE;
        }
        int64_t pos = 0;
        if (ds_state_apply(&eng->ds->m, s->blob, s->blob_len, &pos) != 0) {
            /* Geometry mismatch or a truncated blob: a cache miss, so say so
             * and let the caller prefill the whole prompt. */
            cwick_error_set(err, CWICK_ERR_STATE,
                            "prefix: DeepSeek snapshot does not match this "
                            "checkpoint");
            return CWICK_ERR_STATE;
        }
        eng->ds->pos = pos;
        if (eng->pos < pos) eng->pos = pos;   /* mirror for the summary line */
        return CWICK_OK;
    }
    if (eng->q35) {
        if (s->kind != 3 || !s->blob || s->blob_len == 0) {
            cwick_error_set(err, CWICK_ERR_STATE,
                            "prefix: state was not captured by this backend");
            return CWICK_ERR_STATE;
        }
        int64_t pos = 0;
        /* `s->pos` is the truncation the caller asked for; 0 would mean "as
         * much as the blob holds", which only happens for a snapshot that was
         * stored at exactly this length. */
        if (q35_prefix_apply(&eng->q35->m, s->blob, s->blob_len, s->pos,
                             &pos) != 0) {
            cwick_error_set(err, CWICK_ERR_STATE,
                            "prefix: Qwen3.5 snapshot does not match this "
                            "checkpoint");
            return CWICK_ERR_STATE;
        }
        eng->q35->pos = pos;
        if (eng->pos < pos) eng->pos = pos;   /* mirror for the summary line */
        return CWICK_OK;
    }
    const cwick_model_spec *sp = &eng->spec;
    if (s->nslots != eng->kv_layers || s->L != sp->num_hidden_layers ||
        s->lin_heads != sp->linear_num_heads || s->lin_dim != sp->linear_head_dim ||
        s->H != sp->hidden_size) {
        cwick_error_set(err, CWICK_ERR_STATE,
                       "prefix: state does not match this checkpoint");
        return CWICK_ERR_STATE;
    }
    if (s->pos > eng->capacity) {
        cwick_error_set(err, CWICK_ERR_STATE,
                       "prefix: %lld tokens exceed the %lld window",
                       (long long)s->pos, (long long)eng->capacity);
        return CWICK_ERR_STATE;
    }
    const size_t S = (size_t)(s->nslots > 0 ? s->nslots : 0);
    const size_t Pz = (size_t)s->pos;
    const size_t stride = (size_t)s->lat_stride;
    {
        const bool lbf = cwick_latent_bf16_enabled();
        uint16_t *dst16 = (uint16_t *)eng->keys;
        float *dst32 = eng->keys;
        for (size_t k = 0; k < S; k++) {
            const size_t to = (size_t)k * (size_t)eng->capacity * stride;
            const float *src = s->keys + k * Pz * stride;
            if (lbf)
                for (size_t i = 0; i < Pz * stride; i++)
                    dst16[to + i] = cwick_f32_to_bf16(src[i]);
            else
                memcpy(dst32 + to, src, Pz * stride * sizeof(float));
        }
    }
    if (s->v_stride > 0 && eng->values && s->values) {
        const size_t vs = (size_t)s->v_stride;
        for (size_t k = 0; k < S; k++)
            memcpy(eng->values + (size_t)k * (size_t)eng->capacity * vs,
                   s->values + k * Pz * vs, Pz * vs * sizeof(float));
    }
    const size_t rec = (size_t)s->L * (size_t)s->lin_heads *
                       (size_t)s->lin_dim * (size_t)s->lin_dim;
    if (eng->kda_rec) memcpy(eng->kda_rec, s->kda_rec, rec * sizeof(float));
    const size_t conv = (size_t)s->L * 3 *
                        ((size_t)s->lin_heads * (size_t)s->lin_dim) * 3;
    if (eng->kda_conv) memcpy(eng->kda_conv, s->kda_conv, conv * sizeof(float));

    if (eng->idx_state) {
        for (size_t k = 0; k < S; k++) {
            cwick_indexer_state *is = &eng->idx_state[k];
            int64_t np = s->idx_npools ? s->idx_npools[k] : 0;
            if (np < 0) np = 0;
            if (np > is->pool_cap) np = is->pool_cap;
            is->n_pools = np;
            if (np && is->pool)
                memcpy(is->pool,
                       s->idx_pool + k * (size_t)s->pool_cap * (size_t)s->hd,
                       (size_t)np * (size_t)s->hd * sizeof(float));
            if (is->tail_k)
                memcpy(is->tail_k,
                       s->idx_tail + k * 2 * (size_t)s->kpool * (size_t)s->hd,
                       (size_t)s->kpool * (size_t)s->hd * sizeof(float));
            if (is->tail_g)
                memcpy(is->tail_g,
                       s->idx_tail + k * 2 * (size_t)s->kpool * (size_t)s->hd +
                       (size_t)s->kpool * (size_t)s->hd,
                       (size_t)s->kpool * (size_t)s->hd * sizeof(float));
        }
    }
    if (eng->anchors && s->max_anchors > 0) {
        eng->n_anchors = s->n_anchors;
        memcpy(eng->anchors, s->anchors,
               (size_t)s->max_anchors * (size_t)s->H * sizeof(float));
    }
    eng->pos = s->pos;
    return CWICK_OK;
}

/* Expose just the prefill logits (no decode), so callers can diff the batched
 * and token-serial paths numerically -- and answer /v1/systemone, where a
 * decision is one readout at the end of one prefill.
 *
 * A decision prefill costs exactly what a chat prefill costs, but it reaches
 * the model through this separate entry point, which used to bypass the
 * progress logger, the cwick:prefill_* gauges and the timing line entirely:
 * a multi-question decision could stream tens of GiB with nothing in stdout
 * between the access-log lines. The server binds a request tag before calling
 * this for a live request; CLI verifiers and numeric A/B tests leave it empty
 * and stay quiet (logitdump parses stdout, so an unconditional line here
 * would corrupt its output). */
/* How a prompt longer than one chunk is split.
 *
 * Fill-then-crumb, deliberately. An even split looks fairer but measured
 * slower on the MiniMax path (60 layers, 128 experts, top-4): 550 tokens as
 * 275+275 took 113.6 s against 109.3 s for 512+38, because the expert read is
 * a block-level cost that the biggest possible block amortises best -- a
 * 38-token tail costs ~20 s, while halving the main block costs the whole
 * pipeline more than the tail saves. The lever that does pay is making the
 * block bigger (see MM_CHUNK/DS_CHUNK in this change), not chopping it evenly.
 * CWICK_JEV_CHUNK_EVEN=1 remains only as a way to re-run that A/B. */
static int64_t jev_block_len(int64_t total, int64_t blocks, int64_t idx) {
    int64_t base = total / blocks, rem = total % blocks;
    return base + (idx < rem ? 1 : 0);
}
static bool jev_blocks_even(void) {
    const char *e = getenv("CWICK_JEV_CHUNK_EVEN");
    return e && atoi(e) != 0;       /* off by default; see the measurement */
}

cwick_status cwick_engine_prefill_logits_at(cwick_engine *eng,
                                            const uint32_t *prompt,
                                            int64_t prompt_len, int64_t start_pos,
                                            float *logits_out, cwick_error *err) {
    cwick_error_clear(err);
    int64_t cap = eng ? eng->capacity : 0;
    if (prompt_len < 1 || start_pos < 0 || start_pos + prompt_len > cap) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                       "prefill window %lld..%lld outside 0..%lld",
                       (long long)start_pos,
                       (long long)(start_pos + prompt_len), (long long)cap);
        return CWICK_ERR_ARGUMENT;
    }

    const bool resume = start_pos > 0;
    if (!resume) {
        cwick_engine_reset(eng, err);
        if (err->status != CWICK_OK) return err->status;
    } else {
        /* Keep [0, start_pos) and overwrite from there. Each backend keeps its
         * own cursor, so all three are set rather than inferred; leaving one
         * behind would make that backend append instead of overwrite. */
        eng->pos = start_pos;
        if (eng->ds) eng->ds->pos = start_pos;
        if (eng->mm) eng->mm->pos = start_pos;
        if (eng->q35) eng->q35->pos = start_pos;
    }

    const bool report = eng->req_tag[0] != '\0';
    const char *tag = eng_tag(eng);
    double t0 = 0.0;
    if (report) {
        t0 = now_ms();
        cwick_metrics_set_prefill(start_pos, start_pos + prompt_len, 0,
                                  eng->spec.num_hidden_layers);
        if (!resume) {
            CWICK_INFO(tag, "prefill %lld tokens (context %lld, %lld layers) — "
                       "decision readout, no decode",
                       (long long)prompt_len, (long long)cap,
                       (long long)eng->spec.num_hidden_layers);
            if (eng->ds)
                ds_prefill_progress_begin(prompt_len, (int)eng->spec.num_hidden_layers);
            if (eng->mm)
                mm_prefill_progress_begin(prompt_len, (int)eng->spec.num_hidden_layers);
        } else {
            /* A resumed slice is short by construction; the heartbeat would
             * only add noise to a line that already says where it starts. */
            CWICK_INFO(tag, "prefill resume at %lld: %lld more tokens%s",
                       (long long)start_pos, (long long)prompt_len,
                       eng->mm ? " (MiniMax)" : eng->ds ? " (DeepSeek)" : "");
        }
    }

    cwick_status st = CWICK_OK;
    if (eng->mm) {
        int32_t *ids = cwick_xmalloc((size_t)prompt_len * sizeof(int32_t));
        for (int64_t i = 0; i < prompt_len; i++) ids[i] = (int32_t)prompt[i];
        int64_t blocks = (prompt_len + MM_CHUNK - 1) / MM_CHUNK;
        const bool even = jev_blocks_even();
        int64_t pos = 0;
        for (int64_t b = 0; b < blocks; b++) {
            int64_t take = even ? jev_block_len(prompt_len, blocks, b)
                                : (prompt_len - pos > MM_CHUNK
                                       ? MM_CHUNK : prompt_len - pos);
            int n = (int)take;
            if (mm_model_forward(&eng->mm->m, ids + pos, n, eng->mm->pos,
                                 eng->mm->logits, err) != 0) {
                free(ids);
                st = CWICK_ERR_STATE;
                goto done;
            }
            eng->mm->pos += n;
            pos += n;
        }
        memcpy(logits_out, eng->mm->logits,
               (size_t)eng->spec.vocab_size * sizeof(float));
        free(ids);
    } else if (eng->q35) {
        int32_t *ids = cwick_xmalloc((size_t)prompt_len * sizeof(int32_t));
        for (int64_t i = 0; i < prompt_len; i++) ids[i] = (int32_t)prompt[i];
        int64_t pos = 0;
        while (pos < prompt_len) {
            int n = (int)(prompt_len - pos > Q35_CHUNK ? Q35_CHUNK
                                                       : prompt_len - pos);
            if (q35_model_forward(&eng->q35->m, ids + pos, n, eng->q35->pos,
                                  eng->q35->logits, err) != 0) {
                free(ids);
                st = CWICK_ERR_STATE;
                goto done;
            }
            eng->q35->pos += n;
            pos += n;
        }
        memcpy(logits_out, eng->q35->logits,
               (size_t)eng->spec.vocab_size * sizeof(float));
        free(ids);
    } else if (eng->ds) {
        int32_t *ids = cwick_xmalloc((size_t)prompt_len * sizeof(int32_t));
        for (int64_t i = 0; i < prompt_len; i++) ids[i] = (int32_t)prompt[i];
        int64_t blocks = (prompt_len + DS_CHUNK - 1) / DS_CHUNK;
        const bool even = jev_blocks_even();
        int64_t pos = 0;
        for (int64_t b = 0; b < blocks; b++) {
            int64_t take = even ? jev_block_len(prompt_len, blocks, b)
                                : (prompt_len - pos > DS_CHUNK
                                       ? DS_CHUNK : prompt_len - pos);
            int n = (int)take;
            if (ds_model_forward(&eng->ds->m, ids + pos, n, eng->ds->pos,
                                 eng->ds->logits, err) != 0) {
                free(ids);
                st = CWICK_ERR_STATE;
                goto done;
            }
            eng->ds->pos += n;
            pos += n;
        }
        memcpy(logits_out, eng->ds->logits,
               (size_t)eng->spec.vocab_size * sizeof(float));
        free(ids);
    } else {
        /* GLM chunks internally and starts from eng->pos, so a resume needs
         * nothing beyond the cursor already set above. */
        float *logits = cwick_xmalloc((size_t)eng->spec.vocab_size * sizeof(float));
        st = engine_prefill(eng, prompt, prompt_len, logits, err);
        if (st == CWICK_OK)
            memcpy(logits_out, logits, (size_t)eng->spec.vocab_size * sizeof(float));
        free(logits);
    }

done:
    if (report) {
        double ms = now_ms() - t0;
        if (eng->ds && !resume) ds_prefill_progress_end();
        if (eng->mm && !resume) mm_prefill_progress_end();
        cwick_metrics_set_prefill(0, 0, 0, 0);
        CWICK_INFO(tag, "prefill %s: %lld tokens in %.0f ms (%.2f tok/s)%s",
                   resume ? "slice done" : "done",
                   (long long)prompt_len, ms,
                   ms > 0.0 ? (double)prompt_len * 1000.0 / ms : 0.0,
                   st == CWICK_OK ? "" : " — failed");
    }
    return st;
}

cwick_status cwick_engine_prefill_logits(cwick_engine *eng, const uint32_t *prompt,
                                       int64_t prompt_len, float *logits_out,
                                       cwick_error *err) {
    return cwick_engine_prefill_logits_at(eng, prompt, prompt_len, 0, logits_out, err);
}

static cwick_status generate_impl(cwick_engine *eng, const uint32_t *prompt,
                                 int64_t prompt_len, int64_t max_new,
                                 bool keep_prompt, cwick_token_sink sink,
                                 void *sink_user, cwick_gen_result *result,
                                 cwick_error *err) {
    cwick_error_clear(err);
    result->tokens = NULL;
    result->n_tokens = 0;
    result->stop_reason_length = false;

    if (engine_stopped(eng)) {
        cwick_error_set(err, CWICK_ERR_STATE, "%s", engine_interrupt_message(eng));
        return CWICK_ERR_STATE;
    }
    int64_t cap = eng->capacity;
    if (!prompt || prompt_len <= 0) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "prompt must contain at least one token");
        return CWICK_ERR_ARGUMENT;
    }
    if (eng->pos + prompt_len > cap) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT,
                       "prompt length %lld outside 1..%lld (raise with --context to "
                       "serve longer prompts)", prompt_len, cap - eng->pos);
        return CWICK_ERR_ARGUMENT;
    }
    if (max_new == 0) {
        result->stop_reason_length = true;
        return CWICK_OK;
    }
    if (max_new < 0) max_new = eng->max_new;

    /* Bound the new-token budget by what is left of the KV window. Without
     * this a request that omits max_tokens (the serve default is "no explicit
     * cap") decodes until forward_token() trips its capacity guard, so the
     * whole request FAILS after hours of decoding instead of stopping cleanly
     * with finish_reason=length at the context boundary. */
    bool budget_is_ceiling = false;   /* max_new came from the KV window */
    {
        int64_t remaining = cap - eng->pos - prompt_len;
        if (remaining < 1) remaining = 1;
        /* `>=`, not `>`: a request that asks for exactly the remaining window
         * (which is what every request that omits max_tokens ends up with,
         * since it inherits the server default) is just as much a capacity
         * ceiling as one that asks for more. */
        if (max_new >= remaining) {
            max_new = remaining;
            /* This number is a capacity ceiling, not a requested length.
             * Progress lines must not present it as a denominator --
             * "1/99884" reads as 0.001% done and tells the operator nothing,
             * while the real answer stops at EOS long before it. */
            budget_is_ceiling = true;
        }
    }

    float *logits = cwick_xmalloc((size_t)eng->spec.vocab_size * sizeof(float));

    if (budget_is_ceiling) {
        CWICK_VERB(eng_tag(eng), "prefill %lld tokens (context %lld, max_new %lld "
                         "= context cap, uncapped request)",
                  (long long)prompt_len, (long long)cap, (long long)max_new);
        /* A client that omits max_tokens gets the whole remaining context as
         * its budget, which is the OpenAI reading of the field but on a ~1.3
         * tok/s local engine means a request measured in hours: 99501 tokens
         * is about 21 h. The client sees none of this until it times out and
         * the request lands as an abort, so say it out loud at WARN level and
         * put a number on it. This is the single most common way a session
         * against this server looks "hung". */
        if (max_new > 4096) {
            double step_s = cwick_metrics_decode_step_mean_seconds();
            if (step_s <= 0.0) step_s = 0.78;   /* measured mean before samples */
            double tps = 1.0 / step_s;
            double secs = (double)max_new * step_s;
            CWICK_INFO(eng_tag(eng), "request omitted max_tokens: budget is the context "
                             "ceiling, %lld tokens, which at the measured %.2f "
                             "tok/s is about %.1f h of decoding (EOS will "
                             "usually stop it sooner). Pass max_tokens to bound "
                             "the turn.",
                      (long long)max_new, tps, secs / 3600.0);
        }
    } else {
        CWICK_VERB(eng_tag(eng), "prefill %lld tokens (context %lld, max_new %lld)",
                  (long long)prompt_len, (long long)cap, (long long)max_new);
    }
    /* Prefill streams tens of GiB while the decode report deliberately drops
     * it, so the fetch counters have to be zeroed here to be attributable. */
    cwick_prof_reset();
    uint64_t pf_bytes0 = 0;
    {
        uint64_t rc = 0, ch = 0, cm = 0;
        if (eng->spine.stream)
            cwick_stream_stats(eng->spine.stream, &pf_bytes0, &rc, &ch, &cm);
    }
    double prefill_t0 = now_ms();

    /* Prefill: layer-major, expert-sharing by default (see engine_prefill);
     * CWICK_NO_BATCH_PREFILL=1 selects the token-serial reference. */
    /* Publish the prefill before it starts: the token counters only move when
     * a phase completes, so a 30K-token prefill used to be invisible to
     * /metrics for its whole (minute-long) duration. */
    cwick_metrics_set_prefill(0, prompt_len, 0, eng->spec.num_hidden_layers);
    if (prompt_len > CWICK_LONG_PROMPT_TOKENS)
        CWICK_INFO(eng_tag(eng), "long prompt: %lld tokens x %lld layers — the "
                  "per-token attention sweep is single-threaded, so expect "
                  "minutes before the first token (live progress on "
                  "cwick:prefill_* and in the verbose heartbeat)",
                  (long long)prompt_len,
                  (long long)eng->spec.num_hidden_layers);
    {
        cwick_status pf = engine_prefill(eng, prompt, prompt_len, logits, err);
        if (pf != CWICK_OK) {
            cwick_metrics_set_prefill(0, 0, 0, 0);
            free(logits);
            return pf;
        }
    }
    cwick_metrics_set_prefill(0, 0, 0, 0);
    double prefill_ms = now_ms() - prefill_t0;
    /* prompt_tokens_total is added by the server once the request finishes,
     * with the full prompt length (prefix-replayed tokens included); adding
     * the prefilled-only count here as well would double-count. */

    /* Prefix snapshot, taken here and nowhere else: eng->pos is now the whole
     * prompt (a restored prefix plus whatever tail was just prefilled), and
     * one decode step later that state has moved on. A failed capture must
     * not fail the request -- it is an optimisation. */
    pfx_capture_and_handoff(eng, eng->pos);

    {
        uint64_t pf_bytes1 = pf_bytes0, rc = 0, ch = 0, cm = 0;
        if (eng->spine.stream)
            cwick_stream_stats(eng->spine.stream, &pf_bytes1, &rc, &ch, &cm);
        cwick_prof_report_prefill(eng_tag(eng), prefill_ms, pf_bytes1 - pf_bytes0);
    }

    /* decode loop: pick a token per step (argmax unless the checkpoint or the
     * request asked for sampling), stop at EOS or max_new.
     * The logits left over from the prefill already predict the token *after*
     * the last prompt token, so the first step consumes them instead of
     * re-feeding the last prompt token (which would duplicate it). */
    cwick_sampler smp;
    sampler_begin(&smp, eng, prompt, prompt_len, sampler_seed());
    int64_t collected = keep_prompt ? prompt_len : 0;
    uint32_t *tokens = cwick_xmalloc((size_t)(collected + max_new + 1) * sizeof(uint32_t));
    if (keep_prompt) memcpy(tokens, prompt, (size_t)prompt_len * sizeof(uint32_t));

    int64_t gen = 0;
    int64_t decode_steps = 0;
    /* The ceiling this turn runs under, clamped to what the output budget can
     * honour. A ceiling at or above max_new can never fire, and the turn it
     * exists to save is exactly the one that dies at the length limit: a live
     * request asked for 高 (8192) with max_tokens 8096 and spent every token
     * thinking. Leaving a minimum answer behind the cap (CWICK_THINK_RESERVE)
     * is the difference between "thought about it and ran out" and "thought
     * about it, then said something". */
    int64_t think_budget = eng->think_budget;
    if (think_budget >= max_new) {
        int64_t room = max_new - CWICK_THINK_RESERVE;
        if (room < 0) room = max_new / 2;
        think_budget = room;
        CWICK_VERB(eng_tag(eng), "thinking budget %lld clamped to %lld: %lld "
                   "tokens of output leave no room to answer",
                   (long long)eng->think_budget, (long long)think_budget,
                   (long long)max_new);
    }
    bool hit_eos = false;
    bool interrupted = false;
    bool sink_stopped = false;   /* the consumer asked us to stop */
    bool loop_stopped = false;   /* the loop guard ended a repetition run */
    bool think_budget_hit = false;   /* we closed the think block ourselves */
    /* Think-block state for the budget. `think_open` starts true when the
     * template already appended the open tag (GLM), and flips when the model
     * writes it; `think_closed` latches once either side produces the
     * close tag, after which the budget has nothing left to do. */
    bool think_open = eng->think_open_at_start;
    bool think_closed = false;
    const bool debug = cwick_env_true("CWICK_DEBUG_LOGITS");
    cwick_prof_reset();   /* prefill shares the same counters; drop its noise */
    double decode_t0 = now_ms();
    /* A stale hint from an aborted previous request would mislabel this one's
     * first steps, so start from "unknown" and let the sink re-classify. */
    eng->decode_phase = NULL;
    /* Live accounting of the two token counters (see the flush below): the
     * scrape must see a decode moving while it runs, not only when it ends. */
    int64_t steps_accounted = 0, gen_accounted = 0;
    /* Hold the GPU clock up through this loop's CPU phases (expert streaming
     * and attention are ~85% of a decode step). Without it the power manager
     * reads the gaps as idle and the next MoE batch pays the ramp-up inside
     * its own GPUEndTime -- measured at 96 us/dispatch against 46 us with the
     * keepalive running, i.e. ~60 ms per step. Stopped in cwick_engine_free. */
    cwick_metal_keepalive_start();
    for (int64_t k = 0; k < max_new; k++) {
        if (engine_stopped(eng)) { interrupted = true; break; }
        if (debug) {
            /* cheapest possible window onto a wrong forward: the shape of the
             * distribution tells you whether logits collapsed or saturated. */
            double mn = INFINITY, mx = -INFINITY, sum = 0.0;
            for (int64_t i = 0; i < eng->spec.vocab_size; i++) {
                double v = logits[i];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
                sum += v;
            }
            fprintf(stderr, "cwick: step %lld logits min=%.4f max=%.4f mean=%.4f",
                    k, mn, mx, sum / (double)eng->spec.vocab_size);
            for (int t = 0; t < 5; t++) {
                uint32_t bi = argmax(logits, eng->spec.vocab_size);
                fprintf(stderr, " top%d=%u(%.3f)", t + 1, bi, logits[bi]);
                logits[bi] = -INFINITY;
            }
            fprintf(stderr, "\n");
            for (int t = 0; t < 5; t++) { /* restore not needed: recomputed */ }
            break;
        }
        /* Thinking budget (L4). Reaching the ceiling is not a reason to stop
         * the turn -- it is a reason to stop *thinking*. The engine writes the
         * close tag itself, once: the sink sees it (so the wire carries a
         * closed think block and the server's state machine reclassifies the
         * rest as content) and the forward pass consumes it (so the model,
         * which has just written its own delimiter, answers instead of
         * continuing the thought). Both need to happen in the same step, and
         * only this loop can feed a token it did not sample. */
        if (think_budget >= 0 && !think_closed) {
            cwick_think_action act =
                cwick_think_plan(gen, think_budget, think_open, false);
            if (act != CWICK_THINK_NONE) {
                uint32_t inj[2];
                int n_inj = 0;
                if (act == CWICK_THINK_OPEN_CLOSE)
                    inj[n_inj++] = (uint32_t)eng->think_open_id;
                inj[n_inj++] = (uint32_t)eng->think_close_id;
                bool aborted = false;
                int written = 0;
                for (int t = 0; t < n_inj && gen < max_new; t++) {
                    uint32_t ct = inj[t];
                    tokens[collected + gen++] = ct;
                    written++;
                    cwick_sampler_accept(&smp, ct);   /* it is now history */
                    if ((int64_t)ct == eng->think_open_id) think_open = true;
                    if ((int64_t)ct == eng->think_close_id) think_closed = true;
                    if (sink && !sink(sink_user, ct, gen - 1)) {
                        sink_stopped = true;
                        aborted = true;
                        break;
                    }
                    if (k + 1 < max_new) {
                        cwick_status ist = forward_token(eng, ct, logits, err);
                        if (ist != CWICK_OK) {
                            cwick_sampler_free(&smp);
                            free(tokens);
                            free(logits);
                            return ist;
                        }
                        eng->pos++;
                        decode_steps++;
                    }
                }
                if (written) {
                    think_budget_hit = true;
                    CWICK_INFO(eng_tag(eng), "thinking budget: %lld tokens "
                              "reached — wrote %s so the turn answers instead "
                              "of thinking; CWICK_THINK_BUDGET overrides the "
                              "ceiling", (long long)think_budget,
                              act == CWICK_THINK_OPEN_CLOSE ? "<think></think>"
                                                            : "</think>");
                }
                if (aborted) break;
                continue;   /* sample the first content token next */
            }
        }
        if (gen >= max_new) break;   /* an injected pair may have consumed it */
        uint32_t next = cwick_sampler_pick(&smp, logits, eng->spec.vocab_size);
        if (is_eos(&eng->spec, next)) { hit_eos = true; break; }
        tokens[collected + gen++] = next;
        /* Count it only once it is part of the answer: the repetition
         * penalties are about what the model has already written. */
        cwick_sampler_accept(&smp, next);
        /* Keep the budget's view of the block honest: the model closes its own
         * thought in the normal case, and that must silence the ceiling. */
        if ((int64_t)next == eng->think_open_id) think_open = true;
        if ((int64_t)next == eng->think_close_id) think_closed = true;
        /* Loop guard: sampling makes a repetition attractor rare, not
         * impossible, and the cost of reaching max_new inside one is the whole
         * budget (measured: 5282 tokens / 294 s on one request, with an empty
         * answer at the end of it). Stop instead, and say which period looped
         * so the log distinguishes it from a model that simply finished. */
        int lp = 0;
        int64_t lr = 0;
        if (cwick_sampler_looped(&smp, &lp, &lr)) {
            gen--;   /* the token that closed the loop is framing, not content */
            CWICK_INFO(eng_tag(eng), "loop guard: period-%d block repeated %lld "
                      "times (%lld tokens) — stopping after %lld generated; "
                      "CWICK_LOOP_DETECT=0 disables", lp,
                      (long long)(lr / (lp > 0 ? lp : 1) + 1), (long long)lr,
                      (long long)gen);
            loop_stopped = true;
            break;
        }
        /* hand the token to the consumer BEFORE the next forward step, which
         * is the whole point of streaming: the client sees it in milliseconds,
         * not after the remaining (minutes of) decoding. */
        if (sink && !sink(sink_user, next, gen - 1)) { sink_stopped = true; break; }
        /* ---- MTP draft: score the previous guess, then make the next one ---
         * The draft predicts the token *after* `next`, and it can only be
         * taken here: it needs h_t (still in eng->mtp_h, from the forward
         * that produced `next`) and emb(next), and both expire at the next
         * forward. Scoring the previous draft at this same point is what
         * makes this a measurement rather than an assertion -- the token we
         * just sampled IS the real continuation of the position that draft
         * covered, so hits/tries is the acceptance rate with no extra work
         * and no change to what the model emits. */
        if (eng->mtp_on) {
            if (eng->mtp_have_draft) {
                eng->mtp_tries++;
                if (eng->mtp_draft == next) eng->mtp_hits++;
            }
            uint32_t d = 0;
            cwick_status mst = mtp_draft(eng, next, &d, err);
            if (mst == CWICK_OK) {
                eng->mtp_draft = d;
                eng->mtp_have_draft = true;
            } else {
                eng->mtp_have_draft = false;
                eng->mtp_on = false;
                CWICK_INFO(eng_tag(eng), "MTP: drafting switched off after %s",
                          err && err->message[0] ? err->message : "an error");
            }
        }
        double step_t0 = now_ms();
        bool did_forward = false;
        if (k + 1 < max_new) {
            cwick_status st = forward_token(eng, next, logits, err);
            if (st != CWICK_OK) {
                cwick_sampler_free(&smp);
                free(tokens);
                free(logits);
                return st;
            }
            eng->pos++;
            decode_steps++;
            did_forward = true;
            cwick_prof_step();
        }
        cwick_stream_expert_decay(eng->spine.stream);
        maybe_steer_memory(eng, prompt_len + gen);
        double step_ms = now_ms() - step_t0;
        char phase_lab[32];
        decode_phase_label(eng, phase_lab, sizeof(phase_lab));
        if (eng->verbose && did_forward) {
            if (budget_is_ceiling)
                CWICK_VERB(eng_tag(eng), "%s forward %lld %.0f ms (%.3f tok/s)",
                          phase_lab,
                          (long long)decode_steps, step_ms,
                          step_ms > 0 ? 1000.0 / step_ms : 0.0);
            else
                CWICK_VERB(eng_tag(eng), "%s forward %lld/%lld %.0f ms (%.3f tok/s)",
                          phase_lab,
                          (long long)decode_steps,
                          (long long)(max_new > 0 ? max_new - 1 : 0), step_ms,
                          step_ms > 0 ? 1000.0 / step_ms : 0.0);
        }
        if (did_forward) {
            /* The one number that decides throughput. Observed here rather
             * than in the server because the server only sees the whole
             * request, and its e2e time includes the HTTP commit. */
            cwick_metrics_observe_decode_step(step_ms / 1000.0);
            cwick_metrics_observe_iteration_tokens(1);
            /* Same for the two token counters: a scrape mid-stream must
             * already show what this request has produced so far. The
             * every-16-steps cadence keeps the atomic traffic negligible
             * next to the step itself; the remainder is flushed after the
             * loop so the totals stay exact. */
            if ((decode_steps & 15) == 0) {
                cwick_metrics_add_decode_steps(decode_steps - steps_accounted);
                steps_accounted = decode_steps;
                if (gen > gen_accounted) {
                    cwick_metrics_add_generation_tokens(gen - gen_accounted);
                    gen_accounted = gen;
                }
            }
        }
        maybe_progress(eng, phase_lab, gen,
                       budget_is_ceiling ? 0 : (max_new > 0 ? max_new - 1 : 0));
    }
    double decode_ms = now_ms() - decode_t0;

    /* one execution line per generation — this is what `serve` prints with */
    {
        const cwick_spine *sp = cwick_engine_spine(eng);
        uint64_t bs = 0, rc = 0, ch = 0, cm = 0;
        if (sp && sp->stream)
            cwick_stream_stats(sp->stream, &bs, &rc, &ch, &cm);
        /* The prefill logits already predict token 1. Consequently a
         * max_new=N generation performs N-1 decode forwards, not N. Report
         * those real steps (and say n/a when N=1) instead of turning sink
         * bookkeeping into a fictitious thousands-of-tokens/s result. */
        char prefill_tps[32], decode_tps[32];
        /* How this answer was produced. Greedy is silent (it is the default
         * and the whole history of this engine); anything else is worth a
         * line, because "the reply degenerated" and "the reply was drawn with
         * t=1.0" are the same fact read from two ends. */
        char smp_note[72] = "";
        if (!cwick_sampler_cfg_is_greedy(&smp.cfg)) {
            char sd[48];
            cwick_sampler_describe(&smp.cfg, sd, sizeof(sd));
            snprintf(smp_note, sizeof(smp_note), " sample=%s", sd);
        }
        /* Steps the loop breaker armed on: a "stop" with lb>0 is a recovery
         * (the model was escaping a repetition attractor), not a clean
         * finish, and it is the number to watch when judging loop health. */
        char lb_note[24] = "";
        if (smp.lbrk_steps > 0)
            snprintf(lb_note, sizeof(lb_note), " lb=%lld",
                     (long long)smp.lbrk_steps);
        /* Report the full prompt size (server-side token count) when the
         * engine was told it: prompt_len here is only the not-yet-replayed
         * tail, which reads as "prompt=1" on a cache hit. */
        const int64_t shown_prompt =
            eng->prompt_total > 0 ? eng->prompt_total : prompt_len;
        /* Prefix-cache replay is invisible to the stream counters below
         * (those are the weight-stream page cache), so a hit used to print
         * cache=0.0%(0/0) while 28k of 28k prompt tokens were replayed.
         * Report reused/full explicitly: the gap between prompt_len and
         * prompt_total is exactly what the prefix cache saved. */
        const int64_t pfx_reuse =
            (eng->prompt_total > 0 && eng->prompt_total > prompt_len)
                ? eng->prompt_total - prompt_len : 0;
        if (prefill_ms > 0.0)
            snprintf(prefill_tps, sizeof(prefill_tps), "%.3f",
                     (double)shown_prompt * 1000.0 / prefill_ms);
        else
            snprintf(prefill_tps, sizeof(prefill_tps), "n/a");
        if (decode_steps > 0 && decode_ms > 0.0)
            snprintf(decode_tps, sizeof(decode_tps), "%.3f",
                     (double)decode_steps * 1000.0 / decode_ms);
        else
            snprintf(decode_tps, sizeof(decode_tps), "n/a");
        CWICK_INFO(eng_tag(eng), "prompt=%lld gen=%lld finish=%s%s%s "
                  "prefill=%.0fms/%s tok/s decode=%.0fms/%lld steps %s tok/s "
                  "cache=%.1f%%(%llu/%llu) pfx=%lld/%lld "
                  "streamed=%.2fGiB mem=%.2fGiB",
                  (long long)shown_prompt, (long long)gen,
                  interrupted ? "interrupted"
                              : loop_stopped ? "loop"
                              : ((hit_eos || sink_stopped) ? "stop" : "length"),
                  smp_note, lb_note, prefill_ms, prefill_tps, decode_ms,
                  (long long)decode_steps, decode_tps,
                  (ch + cm) ? 100.0 * (double)ch / (double)(ch + cm) : 0.0,
                  (unsigned long long)ch, (unsigned long long)cm,
                  (long long)pfx_reuse, (long long)shown_prompt,
                  cwick_gib(bs), cwick_gib(cwick_process_footprint_bytes()));    }
    if (eng->mtp_on && eng->mtp_tries > 0)
        CWICK_INFO(eng_tag(eng), "MTP draft: %lld/%lld accepted (%.1f%%) — a "
                  "2-token verify step needs ~43%% to pay for itself",
                  (long long)eng->mtp_hits, (long long)eng->mtp_tries,
                  100.0 * (double)eng->mtp_hits / (double)eng->mtp_tries);
    cwick_prof_report(eng_tag(eng), decode_ms);

    result->stop_reason_length =
        !hit_eos && !interrupted && !sink_stopped && !loop_stopped &&
        gen == max_new;
    result->prefill_ms = prefill_ms;
    result->decode_ms = decode_ms;
    result->decode_steps = decode_steps;
    result->interrupted = interrupted;
    result->sink_stopped = sink_stopped;
    result->loop_stopped = loop_stopped;
    result->think_budget_hit = think_budget_hit;
    result->think_budget_applied = think_budget;
    /* Flush whatever the live accounting did not reach yet, so the totals
     * stay exact even for a generation that ends between cadences (EOS,
     * abort, or the last step of the budget). */
    if (decode_steps > steps_accounted)
        cwick_metrics_add_decode_steps(decode_steps - steps_accounted);
    if (gen > gen_accounted) {
        cwick_metrics_add_generation_tokens(gen - gen_accounted);
        gen_accounted = gen;
    }
    eng->decode_phase = NULL;   /* no stale hint for the next request */
    result->tokens = tokens;
    result->n_tokens = collected + gen;
    cwick_sampler_free(&smp);
    free(logits);
    if (interrupted) {
        cwick_error_set(err, CWICK_ERR_STATE, "%s", engine_interrupt_message(eng));
        return CWICK_ERR_STATE;
    }
    return CWICK_OK;
}

cwick_status cwick_engine_generate(cwick_engine *eng, const uint32_t *prompt,
                                 int64_t prompt_len, int64_t max_new,
                                 bool keep_prompt, cwick_gen_result *result,
                                 cwick_error *err) {
    if (eng && eng->ds)
        return ds_engine_generate(eng->ds, prompt, prompt_len, max_new,
                                  keep_prompt, NULL, NULL, result, err);
    if (eng && eng->mm)
        return mm_engine_generate(eng->mm, prompt, prompt_len, max_new,
                                  keep_prompt, NULL, NULL, result, err);
    if (eng && eng->q35)
        return q35_engine_generate(eng->q35, prompt, prompt_len, max_new,
                                   keep_prompt, NULL, NULL, result, err);
    return generate_impl(eng, prompt, prompt_len, max_new, keep_prompt,
                         NULL, NULL, result, err);
}

/* ------------------------------------------------------------------ */
/* Batched generation: R sequences, one shared expert fetch per step   */
/* ------------------------------------------------------------------ */
/* The decode loop above drives ONE sequence: every step it routes eight
 * experts per layer, streams whatever the cache missed, and produces one
 * token. This drives R of them through the layers together, so the routing
 * finds the union of their experts, the stream reads it once, and the GEMV
 * applies each matrix to every row that asked for it.
 *
 * Rows leave the group the moment they are done (EOS, or their own token
 * budget), which is the whole point of doing the membership per step: a short
 * answer must not be dragged along at the speed of the longest one in the
 * batch.
 *
 * Prefill stays per sequence -- it is a different shape (thousands of rows,
 * expert-sharing within the prompt already) and batching it is a separate
 * optimisation. */
cwick_status cwick_engine_generate_group(cwick_engine **engs,
                                       const uint32_t *const *prompts,
                                       const int64_t *prompt_lens,
                                       int64_t R, int64_t max_new,
                                       bool keep_prompt,
                                       cwick_gen_result *results,
                                       cwick_error *err) {
    if (!engs || !prompts || !prompt_lens || !results || R < 1) {
        cwick_error_set(err, CWICK_ERR_ARGUMENT, "generate_group: bad arguments");
        return CWICK_ERR_ARGUMENT;
    }
    if (R == 1 || (engs[0] && (engs[0]->ds || engs[0]->mm))) {
        /* The ds backend has no shared expert fetch to amortise -- every
         * layer already reads its experts from one mmap -- so a "group" is
         * just the sequences run one after another. */
        cwick_status st = CWICK_OK;
        for (int64_t i = 0; i < R; i++) {
            cwick_error per;
            cwick_error_clear(&per);
            cwick_status s = cwick_engine_generate(engs[i], prompts[i],
                                                   prompt_lens[i], max_new,
                                                   keep_prompt, &results[i],
                                                   &per);
            if (s != CWICK_OK && st == CWICK_OK) {
                st = s;
                if (err) *err = per;
            }
        }
        return st;
    }

    cwick_error_clear(err);
    const int64_t V = engs[0]->spec.vocab_size;

    float **logits = cwick_xmalloc((size_t)R * sizeof(float *));
    /* One sampler per row: the repetition counts are per sequence. */
    cwick_sampler *smps = cwick_xmalloc((size_t)R * sizeof(cwick_sampler));
    uint64_t seed_base = sampler_seed();
    int64_t *budget = cwick_xmalloc((size_t)R * sizeof(int64_t));
    int64_t *gen = cwick_xmalloc((size_t)R * sizeof(int64_t));
    int64_t *steps = cwick_xmalloc((size_t)R * sizeof(int64_t));
    uint32_t **toks = cwick_xmalloc((size_t)R * sizeof(uint32_t *));
    bool *active = cwick_xmalloc((size_t)R * sizeof(bool));
    bool *hit_eos = cwick_xmalloc((size_t)R * sizeof(bool));
    double *pf_ms = cwick_xmalloc((size_t)R * sizeof(double));

    for (int64_t i = 0; i < R; i++) {
        sampler_begin(&smps[i], engs[i], prompts[i], prompt_lens[i],
                      seed_base ? seed_base + (uint64_t)i : 0);
    }
    for (int64_t i = 0; i < R; i++) {
        results[i].tokens = NULL;
        results[i].n_tokens = 0;
        results[i].stop_reason_length = false;
        results[i].prefill_ms = 0.0;
        results[i].decode_ms = 0.0;
        results[i].decode_steps = 0;
        results[i].interrupted = false;
        results[i].sink_stopped = false;
        results[i].loop_stopped = false;
        logits[i] = cwick_xmalloc((size_t)V * sizeof(float));
        gen[i] = 0; steps[i] = 0; pf_ms[i] = 0.0;
        hit_eos[i] = false;
        active[i] = true;

        int64_t remaining = engs[i]->capacity - engs[i]->pos - prompt_lens[i];
        if (remaining < 1) remaining = 1;
        int64_t b = (max_new < 0) ? engs[i]->max_new : max_new;
        if (b >= remaining) b = remaining;
        budget[i] = b;
        if (b <= 0) { active[i] = false; continue; }

        int64_t collected = keep_prompt ? prompt_lens[i] : 0;
        toks[i] = cwick_xmalloc((size_t)(collected + b + 1) * sizeof(uint32_t));
        if (keep_prompt)
            memcpy(toks[i], prompts[i], (size_t)prompt_lens[i] * sizeof(uint32_t));
    }

    /* Prefill: one sequence at a time, each leaving its own next-token logits. */
    for (int64_t i = 0; i < R; i++) {
        if (!active[i]) continue;
        if (engine_stopped(engs[i])) { active[i] = false; continue; }
        double t0 = now_ms();
        cwick_status pf = engine_prefill(engs[i], prompts[i], prompt_lens[i],
                                        logits[i], err);
        pf_ms[i] = now_ms() - t0;
        if (pf != CWICK_OK) { active[i] = false; goto done; }
    }

    cwick_metal_keepalive_start();
    cwick_prof_reset();   /* decode owns the counters; prefill's are noise */
    g_group_fg_ms = 0.0;
    g_batch_completion_ms = 0.0;
    g_batch_router_ms = 0.0;
    g_batch_shared_ms = 0.0;
    {
        double dec_t0 = now_ms();
        double dec_ms = 0.0;   /* stays 0 on the goto paths below */
        int64_t group_steps = 0;
        for (;;) {
            int64_t n = 0;
            uint64_t _ts = cwick_prof_ns();
            /* Sample every live row from the logits its last step produced. */
            for (int64_t i = 0; i < R; i++) {
                if (!active[i]) continue;
                if (engine_stopped(engs[i])) {
                    active[i] = false;
                    results[i].interrupted = true;
                    continue;
                }
                uint32_t next = cwick_sampler_pick(&smps[i], logits[i], V);
                if (is_eos(&engs[i]->spec, next)) {
                    hit_eos[i] = true;
                    active[i] = false;
                    continue;
                }
                int64_t collected = keep_prompt ? prompt_lens[i] : 0;
                toks[i][collected + gen[i]++] = next;
                cwick_sampler_accept(&smps[i], next);
                /* Loop guard, same rule as the single-sequence path: a
                 * repetition attractor costs the whole budget if nothing ends
                 * it. The row drops out of the group for this step. */
                int lp = 0;
                int64_t lr = 0;
                if (cwick_sampler_looped(&smps[i], &lp, &lr)) {
                    gen[i]--;   /* the token that closed the loop is framing */
                    results[i].loop_stopped = true;
                    active[i] = false;
                    CWICK_INFO(eng_tag(engs[i]), "loop guard: period-%d block "
                              "repeated %lld times (%lld tokens) — stopping "
                              "after %lld generated; CWICK_LOOP_DETECT=0 "
                              "disables", lp,
                              (long long)(lr / (lp > 0 ? lp : 1) + 1),
                              (long long)lr, (long long)gen[i]);
                    continue;
                }
                if (gen[i] >= budget[i]) active[i] = false;   /* length */
            }
            /* Rebuild the live set: rows that just finished drop out here. */
            cwick_engine **g_eng = cwick_xmalloc((size_t)R * sizeof(cwick_engine *));
            uint32_t *g_tok = cwick_xmalloc((size_t)R * sizeof(uint32_t));
            float **g_log = cwick_xmalloc((size_t)R * sizeof(float *));
            int64_t *g_idx = cwick_xmalloc((size_t)R * sizeof(int64_t));
            for (int64_t i = 0; i < R; i++) {
                if (!active[i]) continue;
                int64_t collected = keep_prompt ? prompt_lens[i] : 0;
                g_eng[n] = engs[i];
                g_tok[n] = toks[i][collected + gen[i] - 1];
                g_log[n] = logits[i];
                g_idx[n] = i;
                n++;
            }
            free(g_idx);
            if (n == 0) { free(g_eng); free(g_tok); free(g_log); break; }
            cwick_prof_add(CWICK_PH_OTHER, cwick_prof_ns() - _ts);

            double step_t0 = now_ms();
            cwick_status st = cwick_forward_group(g_eng, g_tok, n, g_log, err);
            double step_ms = now_ms() - step_t0;
            g_group_fg_ms += step_ms;
            free(g_eng); free(g_tok); free(g_log);
            if (st != CWICK_OK) {
                cwick_metrics_observe_decode_step(step_ms / 1000.0);
                goto done_after_loop;
            }
            for (int64_t i = 0; i < R; i++) {
                if (!active[i]) continue;
                engs[i]->pos++;
                steps[i]++;
                cwick_stream_expert_decay(engs[i]->spine.stream);
            }
            group_steps++;
            cwick_metrics_observe_decode_step(step_ms / 1000.0);
            cwick_metrics_observe_iteration_tokens(n);
            cwick_prof_step();   /* one shared step, like the single path */
            cwick_metrics_add_decode_steps(1);
            cwick_metrics_add_generation_tokens(n);
            maybe_steer_memory(engs[0], prompt_lens[0] + gen[0]);
            if (engs[0]->verbose)
                CWICK_VERB("group", "step %lld: %lld row(s), %.0f ms (%.3f tok/s "
                          "aggregate)", (long long)group_steps, (long long)n,
                          step_ms, step_ms > 0 ? (double)n * 1000.0 / step_ms : 0.0);
        }
        dec_ms = now_ms() - dec_t0;
        for (int64_t i = 0; i < R; i++) {
            results[i].decode_ms = dec_ms;
            results[i].decode_steps = steps[i];
        }
    done_after_loop:
        (void)group_steps;
        /* Same phase breakdown the single-sequence path prints; without it a
         * group regression is invisible (the group is where the sharing is
         * supposed to pay). The report self-guards on CWICK_PROFILE. */
        cwick_prof_report(eng_tag(engs[0]), dec_ms);
    }

done:
    for (int64_t i = 0; i < R; i++) {
        cwick_sampler_free(&smps[i]);
        int64_t collected = keep_prompt ? prompt_lens[i] : 0;
        results[i].tokens = toks[i];
        results[i].n_tokens = collected + gen[i];
        results[i].stop_reason_length =
            !hit_eos[i] && !results[i].interrupted && !results[i].loop_stopped &&
            gen[i] >= budget[i];
        results[i].prefill_ms = pf_ms[i];
        free(logits[i]);
    }
    free(logits); free(budget); free(gen); free(steps); free(toks);
    free(active); free(hit_eos); free(pf_ms); free(smps);
    return (err && err->status != CWICK_OK) ? err->status : CWICK_OK;
}

cwick_status cwick_engine_generate_stream(cwick_engine *eng, const uint32_t *prompt,
                                        int64_t prompt_len, int64_t max_new,
                                        cwick_token_sink sink, void *user,
                                        cwick_gen_result *result, cwick_error *err) {
    if (eng && eng->ds)
        return ds_engine_generate(eng->ds, prompt, prompt_len, max_new, false,
                                  sink, user, result, err);
    if (eng && eng->mm)
        return mm_engine_generate(eng->mm, prompt, prompt_len, max_new, false,
                                  sink, user, result, err);
    if (eng && eng->q35)
        return q35_engine_generate(eng->q35, prompt, prompt_len, max_new, false,
                                   sink, user, result, err);
    return generate_impl(eng, prompt, prompt_len, max_new, false,
                         sink, user, result, err);
}

void cwick_gen_result_free(cwick_gen_result *result) {
    free(result->tokens);
    result->tokens = NULL;
    result->n_tokens = 0;
}
