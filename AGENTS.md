# CWICK — development handoff

<p align="center"><a href="AGENTS.md">English</a> · <a href="AGENTS.zh-CN.md">简体中文</a></p>

This document is for an AI agent or a developer who has **just cloned this repo on a new
machine** and needs to understand the current state before touching anything. The
user-facing guide is [README.md](./README.md); the Chinese version of this file is
[AGENTS.zh-CN.md](./AGENTS.zh-CN.md).
Deep documentation lives in `docs/`, where every page has an English `X.md` and a Chinese
`X.zh-CN.md` that switch to each other on their third line.

## 1. What this is

Run **complete, never-pruned** GLM-5.3-flash (≈306 GiB), DeepSeek-V4.1-Flash-1017
(≈475 GiB) or MiniMax-M3.1-Flash (≈233 GiB) on a single **Apple Silicon Mac, 128 GiB as
the design point**: one C binary that reads the checkpoint exactly as published (no
conversion, no pruning, top-k never reduced), with an OpenAI-compatible server, a
single-page console, a model library and two H5 apps built in. The weights stay on the SSD and stream in;
only a few gigabytes per token actually cross the bus — which is what the name means: a
wick is the narrow thread that lets a large body of wax burn.

Two things define the product, and both are load-bearing constraints rather than
aspirations: the checkpoint runs **whole** (no pruning, no reduced top-k, no lowered
precision) on a 128 GiB machine, and the resident model is reachable **two ways** — an
OpenAI-compatible server, and `POST /v1/systemone`, which reads a typed decision out of
one prefill's distribution so the same weights act as a probability-based decision
service without a retrain or a different call stack.

Measured decode on this machine is 3.63 / 2.78 / 1.48 tok/s (`--backend metal`, full API
request). What separates them is bytes per step against the 123.5 GB/s memory read
ceiling, not a missing optimisation — the numbers and the arithmetic are in `docs/PERF.md`.

## 2. Hard constraints (read before you touch anything — these are not suggestions)

- **C only.** C11 plus two Objective-C files (`src/metal_objc.m`, the Metal bridge for the
  expert/backbone GEMVs, and `src/q35_gpu.m`, the encode layer of the Qwen3.5 whole-step
  forward). They link nothing but macOS system frameworks (Accelerate / Metal / CoreImage /
  ImageIO / CoreGraphics) and the system `sqlite3` (`Makefile:15-16`). No third-party
  libraries, no Python, no weight-conversion step. One-off verification tools are plain C too
  (`tools/*.c`); shell only drives the compiler and the runs.
- **Four model paths**: each owns its loader, forward pass and prompt template; they share
  only the process, the sampler and the server. Adding a model is one more entry, not a
  rewrite of the others.
  - GLM-5.3-flash: `spine.c` (resident weights, memory plan), `engine.c` (request →
    prefill/decode), `attention.c` (MLA/GQA/DeltaNet), `moe.c` (routing + expert GEMV/GEMM);
  - DeepSeek-V4.1: `ds_model.c` / `ds_engine.c` / `ds_ops.c` / `ds_quant.c` / `ds_prompt.c`;
  - MiniMax-M3.1: `mm_model.c` / `mm_engine.c` / `mm_prompt.c` (+ `mm.h`);
  - Qwen3.5: `q35_model.c` / `q35_engine.c` / `q35_prompt.c` (+ `q35.h`), and the GPU
    decode step in `q35_gpu.c` / `q35_gpu.m` / `q35_shader.c`. Dense BF16 with a
    Gated DeltaNet on 24 of its 32 layers, so it streams nothing and needs no expert cache.
    It is the only checkpoint small enough to be held whole: `q35_model_open` faults all
    17.98 GiB in and wires it (plus the KV/DeltaNet state) when the size policy allows, and
    its prefill is parallel over value heads (delta rule) — 58% of prefill on one core
    before it was split. Attention is **two GEMMs** (score and output) through Accelerate,
    with the causal mask written into the rectangle as -1e30 so the softmax argument is
    unchanged; hand-written tiles ran at 22 GMAC/s against cblas's 750 at the same shape
    (`tools/attnbench`). Decode is a separate path (`dec_scores`/`dec_out`): one position,
    sixteen heads, so the code walks K/V rows and lets each 4 KiB row serve sixteen dots —
    a per-head matrix-vector call reads 1 KiB out of every 4 KiB and runs at 13 GB/s. The prefix
    cache carries this backend's state as `kind = 3` (`q35_prefix_pack`/`_apply`), and a
    capture is only useful if it stops where the *next* turn stops agreeing — the last
    8 tokens are searched for a thinking marker for exactly that reason. `CWICK_Q35_PROF=1`
    prints the per-block split. Two norm conventions live in that checkpoint: plain RMSNorm
    uses `1 + w`, the gated one inside the DeltaNet uses `w`. RoPE is `rotate_half`, not
    adjacent pairs, and `q_proj` interleaves `q | gate` per head. All three are documented
    in DECISIONS §51 because all three are silent when wrong; residency and the parallel
    phases are §52, the GEMM rewrite and the f32 cache §54. Decode takes the GPU as one
    command buffer (§61). Speculative verify is `CWICK_Q35_VERIFY=k` (k consecutive
    positions, one weight pass, no drafter — timings are real, text is not) and
    `CWICK_Q35_SPEC=k` (k drafts then a k+1-row verify; k drafts, not k rows — `SPEC=1` is
    one draft and a two-row pass). The drafter is the checkpoint's own MTP block when
    `model-mtp.safetensors` is present, prompt lookup otherwise; `CWICK_Q35_MTP=0` forces
    lookup. It loses in every shape measured — 20.21 tok/s against 20.66 — so it is off by
    default and §64 has the numbers. Its knobs: `CWICK_Q35_MTP_FULL=1` projects a draft
    through all 2.03 GiB of vocabulary instead of the candidate slice (acceptance 0.356
    against 0.168 on prose, for 5.5 ms), `CWICK_Q35_CAND=n` sizes that slice (default
    4096), `CWICK_Q35_MTP_WIN=n` sizes the block's own KV window (default 512 rows, 4 MiB),
    `CWICK_Q35_MTP_POSTNORM=0` feeds the fusion the residual stream instead of the
    final-normed hidden, `CWICK_Q35_MTP_SWAP=1` inverts the fusion order (which zeroes
    acceptance, so leave it), and `CWICK_Q35_MTP_NOATTN=1` closes the block's attention
    half for an ablation, `CWICK_Q35_CAND_T=f` / `CWICK_Q35_CAND_LCAP=n` set how
    much of the shortlist comes from the previous step's logits (DECISIONS §65),
    `CWICK_Q35_ROWFUSE=0` puts the k-row pass back to one dispatch per row for
    the row-local kernels, and `CWICK_Q35_GEMVN_RW=n` sweeps the k-row GEMV's
    simdgroups per threadgroup (measured flat, §65). `CWICK_Q35_GPU_CHECK=1` compares every row of a k-row pass against the
    same GPU path run one row at a time and must print `0.00000`; `CWICK_Q35_GEMVN_RP=1|2`
    forces the verify kernel's rows-per-simdgroup for a same-session A/B, since the baseline
    drifts up to 20% between sessions. `CWICK_Q35_SPEC_POISON=i` deliberately corrupts draft
    i to exercise the partial-accept path. The verify kernel reads activations straight from
    device memory and runs 128-thread groups; it is ablative-measured by `tools/q35gemab`
    (`ABL=0..3`, `MODE`, `PF`, `RPF`, `TH`, `NSGV`), which is how its marginal row was traced
    to activation traffic rather than arithmetic. §62 is the verify and the drafter, §63 the
    per-row cost of the kernel that does it.
  Supporting a new model means a new loader + new dequantisation + a new prompt template
  + new kernels. **It is not a config change.**
- **Quality rules are non-negotiable**: top-k comes from the checkpoint's own
  `num_experts_per_token`, experts are never pruned, embeddings are never tied, and a
  missing shard fails closed (better it refuses to start than quietly runs a smaller model).
- **Documentation voice**: comments and docs describe the mechanism and the measured
  number, and **never name the projects this work learned from**. The "documentation"
  section of `goal.md` makes this a hard rule, and it covers code comments, `docs/` and
  the README alike.
- **Docs are bilingual**: `docs/X.md` (English) + `docs/X.zh-CN.md` (Chinese), with the
  same switcher line on line 3 of both. `README.md` / `README.zh-CN.md` follow the same
  pattern, as do these two files. A new document gets both versions or none.

## 3. Current state

### ✅ Landed

- Three checkpoints end to end: load + `/v1/models` + generate + switch, verified
  numerically (DeepSeek is bit-exact across the whole chain; GLM gets a structural check
  from `doctor` — see `docs/PRECISION*.md`).
- OpenAI-compatible API: `/v1/models`, `/v1/models/<id>`, `/v1/completions`,
  `/v1/chat/completions` (SSE, `reasoning_content` separate from `content`, `tool_calls`
  generated in each checkpoint's own protocol and parsed back to the OpenAI shape).
  Streaming tool calls open as soon as the function name is written and append
  standard argument shards, including **inside** a long parameter value, so a
  one-parameter call fills in instead of landing whole at the end (§59).
  `delta.cwick_tool` remains for the bundled console. Two tool XML dialects are
  accepted (flat `<arg_key>` and nested `<function=>`), both rewritten to one
  parser (§58). The DeepSeek DSML dialect is **not** parsed yet (§59).
- Responses API (`POST /v1/responses`): `input` items (message / function_call /
  function_call_output, flat `function` tools) are translated into the chat request the
  rest of the server already validates, then answered as a `response` object or as
  `response.*` events. One adapter, not a second engine — same prompt, queue, prefix
  cache and thinking budget. No durable store: `previous_response_id` / `conversation`
  are refused by name. Code is in `server.c` (`responses_parse`, `responses_body`,
  `resp_stream_*`); tests pin the envelope, the item conversion and the refusals.
- Anthropic Messages API (`POST /v1/messages`): `system` (string or blocks), content
  blocks (`text` / `thinking` / `tool_use` / `tool_result`), flat tools with
  `input_schema`, `thinking.budget_tokens` → 思考强度, and a
  `message_start → content_block_* → message_delta → message_stop` stream with no
  `[DONE]`. Same adapter rule: `anthropic_parse` rebuilds the chat body and defers to
  `validate_body`; `messages_body` / `msg_stream_*` only write the envelope. Errors use
  `{"type":"error","error":{…}}`. `x-api-key` authenticates the same as `Authorization:
  Bearer`. Refused by name: `stop_sequences`, image/document blocks, `tool_choice.any`.
- Decision endpoint (`--jev`): `POST /v1/systemone` (also `/inference/v1/systemone`)
  reads one prefill's distribution over the option letters and generates no tokens;
  `GET` on either path reports the live adapter (family / prompt grammar / readable
  letters). The adapter is re-resolved when the tokenizer loads and after every
  `POST /v1/models/switch`, so a model switch also switches the interface; a request
  naming a model the server no longer serves is a 400 that names the resident one.
  `make jevprobe` prints the letter table from a tokenizer alone. The wire format
  follows the published decision protocol: `state` is a string or any JSON value,
  `noul`'s `criteria` names its two sides (true first, always), `instructions` is
  required, `stream`/`reasoning_effort`/`thinking`/`images` are refused by name, and
  the status is 422 for "not a decision request" against 400 for "will not serve it".
  Ten questions and 26 options are the deliberate differences (`docs/DECISIONS.md` §48).
- Context: `--context` is KV capacity; `serve` defaults to 131072, `start.sh` uses 500000
  for GLM and DeepSeek and 32768 for MiniMax (its KV is 2.3 KiB per position per layer, so
  500K would be 64 GiB).
- Thinking depth: 关/低/中/高/超高/极致 → 0/512/2048/8192/16384/32768, and **the engine
  enforces the ceiling itself** (it writes the closing marker into both the output and the
  context when the budget is spent). **A request with no reasoning field defaults to 关**
  (`server.c`: `CWICK_THINK_DEFAULT_LEVEL`), identically to an explicit `none`; the
  level is passed to the renderers as a real level, not as an "unset" sentinel.
  The bundled web console sends `reasoning_effort` explicitly and is unaffected.
- Prefix cache (`pfx.c`): the inference state is snapshotted to `--prefix-cache-dir` at the
  end of each prompt and replayed when the next request starts with the same tokens. A
  403-token prompt: 169.4 s cold → 9.95 s warm.
- Memory plan (`spine.c` + `stream.c`): the ceiling is 80% of RAM by default, movable with
  `--mem-fraction` / `--max-memory`; the spine and the expert arena are **wired** by
  default (`--allow-swap` opts out); near the ceiling the runtime evicts with a cooldown
  rather than trimming every step.
- Server (`server.c`): single-slot FIFO queue with every other endpoint concurrent; model
  swap and shutdown answer 503 + `Retry-After`; a queued client that disconnected is
  dropped before it costs a prefill.
- Logs: one line per event on stdout with a wall-clock time and a request id; prefill
  reports layer progress (the block size is `MM_CHUNK` / `DS_CHUNK` = 1024, so a typical
  decision prompt is *one* block and a full block costs 140–200 s on the 60-layer
  checkpoint — per-layer is the only useful granularity, otherwise a single-block prompt
  prints nothing for minutes), decode reports `decode(thinking)` / `decode(content)` /
  `decode(tools)` separately on every per-step and periodic line. The phase hint is
  pushed to all four backends (`engine.c`: DS / MiniMax / Qwen3.5 / GLM), and the
  transition line also carries the finished phase's rate. Qwen3.5's prompt ends with
  `<think>\n`, so its open marker is found in a 4-token window (§56).
- Decision prefill (`jev.c` + `engine.c`, `docs/DECISIONS.md` §44): the questions of one
  request are encoded up front, the longest common *token* prefix is prefilled once and
  each question resumes from it. MiniMax: 1145 input tokens → 449 prefilled,
  213.8 → 102.0 s (2.10×), every probability bit-identical. `usage` reports
  `input_tokens` / `cached_input_tokens` / `prefilled_tokens` / `reused_tokens`.
  **Reuse is enabled only on MiniMax** — force it on the others and the answer changes
  (§44 table). Raising the block ceiling is what paid (550 tokens: 109.3 → 95.5 s);
  splitting the prompt evenly is 4% *slower* and stays behind `CWICK_JEV_CHUNK_EVEN=1`.
- `/v1/metrics`: vLLM naming, readable while the slot is busy.
- Console (`webui.c` + generated `src/webui_embed.c`): five tabs — chat, apps,
  models, logs, settings; the chat tab speaks **all three** dialects
  (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`, chosen per turn by the
  composer's `接口` select), the models tab drives `/v1/hub/*`. The translations live in
  `web/app.js` as pure functions (`respInput`/`respEventToChat`/`respFullToChat`,
  `msgMessages`/`msgEventToChat`/`msgFullToChat`) and end at `postChat`, which hands the
  widgets a chat-shaped event either way — do not add a second rendering path. Settings
  is the old console merged in: live metrics
  with four canvas trend charts, a Prometheus series browser (HELP/TYPE,
  search, any four metrics on private axes, raw or per-second delta), all
  `cwick:model_info` labels, generation preferences, read-only config and
  resident-model switching. Every request is
  same-origin; no external OpenAI endpoint is configurable. `--lan` adds a QR
  code per app.
- App market: tic-tac-toe (human first; the decision asks which cell has the
  highest win probability, sidebar shows every cell's readout) and 3D kart
  (Mobius loop; real-time shuffled 12-way decisions aimed at maximum
  difficulty, sidebar shows obstacle + kart statistics) in `apps/`, each with
  its own sqlite (`store.c`, `--store DIR`, default `~/.cwick/apps`); both
  decide through `/v1/systemone`. State lives server-side rather than in
  localStorage.
- Model library (`hub.c`): `./models/<owner>/<name>/`, resumable `.incomplete` files,
  a background job table, read-only catalog search; five `/v1/hub/*` endpoints are
  answered before the generation slot. `DELETE /v1/hub/models/<name>` removes a
  downloaded checkpoint (recursive, realpath-confined to the root, refuses the
  resident model and an active download), and the catalog entry is dropped in the
  same request. `CWICK_HUB_ENDPOINT` / `HF_ENDPOINT` override the catalog host;
  `./models` is scanned into the catalog at startup.
- Auth: PBKDF2 single-user login + Bearer + HttpOnly cookie; `--api-key NAME` mints a named
  key for agent platforms (hash only), managed via `POST|GET|DELETE /v1/auth/keys`.
- Model switching: `POST /v1/models/switch` runs a read-only model-contract dry
  run before releasing anything. Unknown architectures and broken configs are
  400 with the resident model untouched; 409 while a request is generating or
  queued. `/v1/models` and `/v1/hub/models` report `supported` (cached at
  discovery, so the polled list does no I/O under the generation lock), and
  `cwick:model_loaded` is the live 1/0 gauge.
- `make test`: **651 assertions, all PASS** (`./tests/test_cwick` prints one `ok:` per check).
- `cwick doctor` and `download_model.sh` (resume, >100 GiB, one progress line).

### ⚠️ Known, and still usable

- **Decode is bandwidth-bound on all three**: GLM 3.63 / DeepSeek 2.78 / MiniMax 1.48 tok/s
  (the last one after the decode work in `docs/DECISIONS.md` §27). What stands in the way is
  bytes per step against the same-shard read ceiling, not a cache that was not written —
  read that section and the roofline part of `docs/PERF*.md` before starting.
- `cwick doctor`'s tensor checklist is written against the `model.language_model.*`
  naming (all three branches — KDA / DSA / MLA — exist). A checkpoint that uses another
  prefix (MiniMax does) prints a run of `FAIL missing`: that is **the checklist not
  matching, not corrupt weights**. Judge by whether `serve` loads it and `make test` passes.
- The queue has no depth limit (`slot_acquire()` accepts everything); `429` is only
  reachable on the shutdown path.
- `/v1/metrics` carries no swap or compressor numbers (vLLM's `num_requests_swapped` is
  request preemption, unrelated to OS swap — do not copy it in).
- MiniMax runs on the CPU pool, not Metal — the GPU measured *slower* at these shapes, so
  do not "just wire it up".
- **GLM and DeepSeek cannot resume a prefill in place.** Forcing it moved a decision
  probability by 0.41 (GLM) and 0.56 (DeepSeek) and changed 1–2 of 3 answers, because the
  compressed KV and the partially-filled DSA pools are not addressed by position alone; a
  GLM snapshot round-trip is better but still `|dp| = 0.11`. So `/v1/systemone` reuses the
  shared prefix on MiniMax only and pays one full prefill per question on the other two.
  Making those two exact is the next engineering item, not a flag.

### ⬜ Not done / deliberately not done

- Decode much past the rates measured here on this hardware: the levers that failed are in
  §7, and what is left is architectural — fewer bytes per step, or more than one token per
  weight read.
- Wiring the DeepSeek expert cache: **don't** — measured slower (hit rate unchanged, step
  time longer).
- MiniMax on the GPU: needs its access pattern reworked first; it is not a backend switch.

## 4. Layout

```text
src/
  main.c           CLI: setup | run | serve | doctor, flag parsing, single-instance lock
  server.c         HTTP/SSE, queue, auth, metrics, log stream, app-storage routes
  engine.c         request → render prompt → prefill/decode → sample → stream
  spine.c          resident weights, memory plan, expert pin cache
  stream.c         F_NOCACHE SSD streaming, eviction, prefetch, the cooling valve
  attention.c      MLA / GQA / DeltaNet kernels (NEON)
  moe.c            routing + expert GEMV/GEMM
  sample.c         sampling (t<=0 argmax; top_k → top_p → draw, HF order)
  pfx.c            prefix-cache snapshots (KV checkpoints on disk)
  models.c         model discovery + served aliases (glm-5.3-flash /
                   deepSeek-v4.1-flash-1017 / MiniMax-M3.1-Flash)
  tokenizer.c      BPE; log.c logging; logcap.c the ring behind /v1/logs
  mem.c / pool.c   memory and the NEON thread pool; prof.c phase timing (CWICK_PROFILE=1)
  metal*.c/.m      Metal kernels and the zero-copy arena
  ds_*.c           DeepSeek backend (FP4 experts, FP8 attention, prompt template)
  mm_*.c           MiniMax backend (NVFP4 experts, bf16 GQA, the <mm:think> markers)
  jev.c            decision endpoint: per-family prompt grammar, letter readout,
                   in-request shared prefix
  hub.c            model library: resumable downloads (libcurl) + catalog search
  metrics.c        vLLM-named counters behind /v1/metrics
  store.c          one sqlite per app; auth.c API keys/sessions; qr.m QR codes
  webui.c          console backend; webui_embed.c is generated — do not hand-edit
web/              console source: index.html / app.js / app.css / login.html
apps/             two H5 apps + manifest.json
models/           downloaded checkpoints (gitignored, scanned at startup)
docs/             architecture, storage, performance, decisions, server, precision,
                  DeepSeek port (each in English and Chinese)
tests/test_cwick.c 651 assertions
tools/            standalone C probes: kernel benchmarks, layer tests, model runners
start.sh          recommended entry point: prefix cache + web + LAN + three model dirs
                  + password
```

## 5. Conventions an agent must keep

- **Run `make cwick` before judging a kernel's behaviour.** A stale binary has lied to this
  project more than once.
- After touching `web/` or `apps/` you must run `make src/webui_embed.c && make cwick`,
  otherwise the binary still carries the old assets (404, or a page that never changes).
- **One change at a time**, behind a flag that defaults off, so the comparison arm is the
  other half of the same binary.
- **"Faster" has to pass the anchor first**: a fixed prompt must produce the same top-1
  token id and logit. Without it, a kernel that computes garbage is also 3× faster.
- **A/B on the same machine, back to back.** The baseline drifts by ~20%, and ten minutes
  apart has been measured at 2.3×; anything under 30 steps is a cold cache.
- **`grep` in this repo's bash often returns nothing at all** (measured repeatedly). Search
  text with `rg` or the Grep tool; `sed -i ''` fails here, use `sed -i.bak`.
- A model directory is held by `.cwick.lock` for the process lifetime, so the CLI cannot
  start while a server is running. `CWICK_ALLOW_MULTI=1` bypasses that — **don't use it**:
  two processes evict each other's pages.
- Per-request state must be pushed explicitly into DeepSeek's own loop (initialise it in
  `ds_engine_open`; `eng->ds` is not memset). Past incident: the think budget only applied
  to the GLM loop, and DeepSeek free-ran for 16568 tokens.
- Run `make test` when done. **A fix the suite does not cover is not finished.**
- Commit style: `feat:` / `fix:` / `docs:` / `refactor:` plus a Chinese or English
  description; pushes are manual.

## 6. Getting started on a new machine

```sh
git clone <repo> cwick && cd cwick
make && make test                      # ~12 s, 651 checks

./download_model.sh deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    -o ./deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    --include '*.safetensors' '*.json' -j 2

sh ./start.sh deepseek                 # 0.0.0.0:8001, password cwick-local
```

Getting a token and the first curl request are in `README.md` under "Usage → Chat over
HTTP". Without the real weights you can still build a synthetic model with
`./cwick setup --root /tmp/synth` and exercise `make test` plus the whole server path.

## 7. Traps, and things that must not be tried again

**Block verification (`MM_SPEC`) is not bit-identical to single-token decode.**
It exists, it is exact with respect to the block's own arithmetic, and with
`MM_CHUNK_CAP=1` it reproduces the plain greedy output byte for byte — but with
n=8 the projections run as a chunk GEMM and a near-tie in the argmax can flip
(`short` -> `shorter`). Off by default. The economics are in
`docs/DECISIONS.md` §28: a block of 8 streams ~2x the expert bytes of one token
step, so a drafter needs > 2.5 tokens per forward on ordinary text to pay.

**The read path (there is nothing left to win here)**

- The 6.7 GB/s that `dd` reports is the page cache; the stream fd uses `F_NOCACHE`, so
  **3.4–4.0 GB/s is the device limit**.
- Sorting expert reads **by file offset** is slower, not faster (3.65 vs 3.98 GB/s) —
  ordering is not the bottleneck.
- Prefetch stays off by default: it measured worse, and the root cause is bytes, not cores.
  `CWICK_PREFETCH` exists for comparisons only.

**Memory**

- On macOS `madvise(MADV_DONTNEED)` does **nothing for file mappings** — only `munmap`
  really reclaims (that is what the hot-switch release uses: DS→GLM drops RSS from 99.7 to
  17.6 GiB).
- Trimming every step together with `cwick_stream_limit` collapses the valve (a 90 GiB cap
  reaches zero within two seconds and the log fills with "released 0.00 GiB") ⇒ it needs a
  12 s cooldown, a 1 GiB overshoot, and a floor of `max(cap/4, 256 MiB)`.
- Copying the dense weights into anonymous memory is a net loss: footprint 82 → 92 GiB, and
  it eats the expert page cache.

**Kernels (all rejected by measurement — do not reopen)**

- fp4 kernels in bf16 / fp16: instruction count only drops from 12 to 10 (they are
  issue-bound, not MAC-bound); the 16.8 / 23.1 / 38.5 GMAC/s figures are different
  measurements, not a conversion table.
- fp8 dense through native e4m3: subnormals and NaN, disproven.
- The fp4 grouped GEMM at 8 activations per pass is slower than 4 (16 accumulators
  overflow the register file).
- A single Metal matrix reaches only ~80 GB/s; the answer is batching submissions and
  waiting later (`CWICK_DS_GPU` defaults to 0).
- Sampling costs 0.10 ms/step and is not a bottleneck; `t<=0` goes through argmax, which is
  what makes a run reproducible.

**Decision prefill (`/v1/systemone`, `docs/DECISIONS.md` §44)**

- Prefill cost is **per block at the expert read**, so do not "balance" a prompt: an even
  275+275 split of 550 tokens measured 113.6 s against 109.3 s for 512+38. Raise the
  ceiling instead (512 → 1024 is the 1.14×); the extra activation buffers cost ~1.6 GiB.
- The shared-prefix resume is exact **only on MiniMax**. On GLM and DeepSeek it changes
  answers (`|dp|` 0.41 / 0.56); a GLM snapshot round-trip still leaves 0.11. Never turn it
  on for a decision endpoint to win a benchmark — a 2× that occasionally changes the
  decision is a bug, not an optimisation.
- The switch is per family in `jev_decide` (`CWICK_JEV_REUSE_ALL=1` to measure a repaired
  backend, `CWICK_JEV_NO_PREFIX_REUSE=1` to force the old path).

**Model details**

- MiniMax's thinking markers are **200059 / 200060 `<mm:think>`**, not 200050 (which is
  also in the vocabulary, and will bite).
- `DS_CHUNK` is defined in `ds_model.h` and nowhere else — a second `#define` in the `.c`
  trips `-Wmacro-redefined` and the two values then disagree.
- `ds_e4m3_to_f32()` is a function call with a `build_luts` branch — **inside a hot loop it
  must be replaced by a local table** (2.6× measured).
- Prefix cache off-by-one: `pfx_save_now` must also save the `nids-1` snapshot;
  `CWICK_PFX_MIN_TOKENS` is 32. When debugging, `cmp` the two rendered prompts byte by
  byte — it is faster than reading the code.
- When the output looks wrong, find a trustworthy baseline before suspecting a kernel: a
  "corrupted batched prefill" once turned out to be greedy decoding repeating itself on a
  prompt that repeated one phrase 30 times (`mechanical, mechanical,,`).
