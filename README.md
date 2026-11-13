<h1 align="center">CWICK</h1>

<p align="center">
  <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a>
</p>

**CWICK runs a complete, never-pruned large model on a single Apple Silicon Mac.**
Not a quantized sketch of one, not a distilled stand-in: the whole checkpoint,
every expert on disk, every token decided by the model itself. One C binary with
no third-party dependencies, no conversion step, and no Python — it reads the
checkpoint as published and serves it through an OpenAI-compatible API.

Serve `:8001` · `./cwick run` · [Architecture](docs/ARCHITECTURE.md) ·
[Performance](docs/PERF.md) · [Decisions](docs/DECISIONS.md) ·
[Server reference](docs/SERVER.md) · [Storage](docs/STORAGE.md) ·
[Development handoff](AGENTS.md)

The name is the picture: a wick is the narrow thread that lets a large body of
wax burn. Here the wax is a 233–475 GiB weight bank on the SSD and the wick is
the few gigabytes per token that actually cross the bus.

Three checkpoints are supported end to end today — not by design a closed list, but
each needs its own loader, quantizer, prompt renderer and kernels:

| model | on disk | shape | why it is here |
|---|---|---|---|
| **GLM-5.3-flash** | ≈306 GiB, 62 shards | 45 layers, 288 routed experts, top-8, FP8-E4M3 | the original model, ≈320B parameters |
| **DeepSeek-V4.1-Flash-1017** | ≈475 GiB, 48 shards | 40 layers, 384 routed experts top-6, MLA + FP4 experts + FP8 attention | the largest one that still fits a laptop |
| **MiniMax-M3.1-Flash** | ≈233 GiB, 48 shards | 60 layers (3 dense), 128 experts top-4, NVFP4 experts + FP8 dense | the smallest footprint of the three |
| **Qwen3.5-9B** | ≈18 GiB, 4 shards | 32 layers: 24 Gated DeltaNet + 8 GQA, dense BF16, no experts | hybrid linear attention; the only one that is not an MoE, and the only one held wholly in RAM |

The quality rule is not negotiable and is enforced at load time: **top-k is the
model's own `num_experts_per_token`** (never reduced), experts are never pruned,
embeddings are never tied, and a missing shard fails closed instead of quietly
running a smaller model.

# Contents

- [Why](#why)
- [How this was built: an agent in a loop](#how-this-was-built-an-agent-in-a-loop)
- [Supported hardware](#supported-hardware)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [API](#api)
- [Models, context and memory](#models-context-and-memory)
- [Long prompts and the prefix cache](#long-prompts-and-the-prefix-cache)
- [Thinking, streaming and tools](#thinking-streaming-and-tools)
- [Speed](#speed)
- [Verification](#verification)
- [Detailed Guides](#detailed-guides)
- [FAQ](#faq)
- [Directory Layout](#directory-layout)
- [License](#license)

## Why

The experiment is whether a complete engine for specific checkpoints can be
**designed and written from zero by an AI agent** — loader, quantizers, kernels,
server, web console — instead of adapted from a general-purpose runtime, and
whether that agent can then improve it on its own: optimize → compile → deploy →
feedback, each turn ending in a number rather than an opinion. The engine is the
output of that loop, and the loop is why it exists at this size at all.

Two commitments follow from that, and everything else is downstream of them.
**The checkpoint runs whole** — no pruning, no reduced top-k, no lowered precision — on
a 128 GiB machine, with the SSD treated as part of the memory hierarchy rather than an
exception. And **the same resident model answers two ways**: an OpenAI-compatible chat
API, and a typed decision read out of one prefill's distribution, so the deployed model
becomes a probability-based decision service without retraining, re-weighting or a
different call stack.

Everything below is a reason it has to look the way it does:

* The best open weights now fit on machines people can own: 233–475 GiB of
  weights on a 128 GiB laptop is a scheduling problem, not a reason to ship a
  smaller model.
* MoE models tolerate aggressive expert quantization far better than dense ones,
  so the sparse part can live on disk while the arithmetic stays exact where it
  matters.
* An inference engine specialized for three checkpoints can be far simpler, and
  far more honest about its numbers, than a general-purpose runner.

## How this was built: an agent in a loop

The work was split the way the loop needs it: the human set the brief and judged
the verdicts; the agent wrote the loader, the kernels, the server, and the loop
that measures them. No part of it was ported from a general-purpose engine.

What made that possible is not code generation. It is that every change goes
through the same loop, and the loop is cheap enough to run dozens of times a day:

```
   hypothesis ──► optimize ──► compile ──► anchor ──► real-hardware bench
        ▲                                                      │
        └────────────── feedback ◄── deploy (serve) ◄──────────┘
```

* **optimize / compile** — one change at a time, behind a flag that defaults off,
  so the comparison arm is the same binary.
* **anchor** — a fixed prompt must produce the same top-1 token id and logit.
  This is what makes "faster" meaningful; without it a kernel that computes
  garbage is also 3× faster.
* **real-hardware bench** — this pool, this SSD, this memory ceiling. Every number
  in this document came from this machine, with its conditions written next to it.
* **deploy → feedback** — a running serve's `/v1/metrics` and request log are the
  next hypothesis' input: which phase grew, which cache stopped hitting, which
  request spent its time queued behind a single long prefill.

The output is a list of changes that survived and a longer list that did not
([docs/DECISIONS.md](docs/DECISIONS.md)). Three that survived were found by the
loop, not by a plan: attention's inner loop vectorised (398 → 30 ms per token),
expert reads overlapped with compute (io 96 → 50 ms per token), the FP8 dense GEMM
tiled 4×4 (96 → 400 GMAC/s). Three that did not looked obvious and measured
slower: a speculative prefetcher, sorting expert reads by file offset, and copying
the dense weights into anonymous memory.

The loop can be run without a human in it — `make test`, one fixed anchor, one
benchmark command — which is the actual point. A verdict is a file, not a feeling.

## Supported hardware

* **Apple Silicon**, with 128 GiB as the design point. The planner reads the
  machine's RAM at startup, puts the ceiling at 80% of it, and fills the expert
  cache with whatever the resident spine and the KV cache leave over; the rest of
  the bank stays on the SSD. Steady state sits under that ceiling, which is what
  keeps `swap` flat instead of climbing into the tens of gigabytes and keeps the
  compressor off the pages every token touches.
* **Smaller Macs** work through the same path — SSD streaming is not a fallback
  mode, it is how every model larger than RAM runs.
* **Metal** accelerates the GLM and DeepSeek backends (expert GEMV, batched
  projections) and is the default on Apple Silicon; `--backend cpu` forces the
  CPU path. At startup it raises the GPU wired-memory ceiling to 80% of RAM
  and wires the inference state — KV cache, indexer pools, recurrent state — so
  the pages every token touches are neither compressed nor swapped. Expert blobs
  that live in the GPU arena are bound to the next dispatch without a copy, which
  is the whole reason unified memory is worth using here. The MiniMax backend
  stays on the CPU pool: its kernels measured *slower* on the GPU at these shapes,
  and the numbers are in [docs/DECISIONS.md](docs/DECISIONS.md) rather than hidden
  behind a flag.
* **NVIDIA and ROCm are not supported.** This is a deliberately narrow engine for
  one machine class.

## Architecture

```text
client (curl / coding agent / desktop chat app / web console)
  └── HTTP :8001 ──► server.c
        ├── /v1/chat/completions ──► queue (single slot, arrival order)
        │        └── engine.c ──► render prompt, prefill, decode, sample
        │              ├── GLM-5.3-flash : spine.c  attention.c  moe.c  pfx.c
        │              ├── DeepSeek-V4.1  : ds_model.c  ds_ops.c  ds_quant.c
        │              ├── MiniMax-M3.1   : mm_model.c  mm_prompt.c
        │              └── Qwen3.5-9B      : q35_model.c  q35_prompt.c
        ├── /v1/systemone ──► jev.c ──► one prefill, read the distribution
        ├── /v1/metrics  /v1/models  /v1/config  /v1/logs/stream
        ├── /v1/hub/*    ──► hub.c (download jobs, model catalog)
        │        └── answered while the slot is busy — this is how a 30-minute
        │            prefill stays watchable
        └── /  ├── web/     single-page console, embedded at build time
                ├── apps/   two H5 apps, one sqlite file each (store.c)
                └── models/ downloaded checkpoints, scanned at startup

state on this machine
  ├── resident : bf16 spine + KV cache + indexer pools   (wired, never swapped)
  ├── experts  : pin cache in shared memory + F_NOCACHE reads from the SSD
  ├── snapshot : pfx.c checkpoints under --prefix-cache-dir, replayed not re-prefilled
  └── compute  : pool.c (NEON threads) or metal.c (zero-copy arena blobs)
```

- **Language and dependencies**: C11 plus two Objective-C files for the Metal
  bridge. It links nothing but macOS system frameworks — Accelerate, Metal,
  CoreImage, ImageIO, CoreGraphics — and the system `sqlite3`. No Python, no
  BLAS, no runtime conversion step.
- **One binary, four subcommands**: `setup` (synthetic verification model), `run`
  (one prompt, or a batch-decode harness), `serve` (the API server), `doctor`
  (read every resident byte and print the plan).
- **The web console is compiled in** (`src/webui_embed.c`), so `--web` needs no
  assets on disk and no second process; the apps are files under `apps/`, served
  by the same server and stored in sqlite.
- **One model resident at a time.** That is a bandwidth decision, not a missing
  feature: two generations would halve each other's cache.

## Quick Start

### 0. Prerequisites

- macOS on Apple Silicon, and `clang`. Nothing else — see
  [Supported hardware](#supported-hardware) for the memory design point.
- A checkpoint: ~233 GiB (MiniMax), ~306 GiB (GLM) or ~475 GiB (DeepSeek) of SSD
  free, downloaded as published.

### 1. Build and test

```sh
git clone <this repo> cwick
cd cwick
make            # builds ./cwick
make test       # the verification suite: engines, sampler, prefix cache, server, apps
```

### 2. Fetch a checkpoint

`download_model.sh` is curl plus bash — it resumes from the byte it reached,
handles single files above 100 GiB, and prints one progress line for the whole
repository instead of one bar per shard:

```sh
# what a repository contains and how big it is, without downloading anything
./download_model.sh deepseek-ai/DeepSeek-V4.1-Flash-1017 --list

# 48 shards, weights and config only, into the directory start.sh expects
./download_model.sh deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    -o ./deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    --include '*.safetensors' '*.json' -j 2
```

A downloaded directory already *is* a model root: `config.json` plus shards is
all the loader needs, and the contract validator will refuse a checkpoint whose
quality knobs were tampered with.

### 3. Serve

```sh
sh ./start.sh glm              # GLM-5.3-flash,      0.0.0.0:8001, 500K context
sh ./start.sh deepseek         # DeepSeek-V4.1-Flash, same
sh ./start.sh minimax          # MiniMax-M3.1-Flash, 32K context (its KV is 2.3 KiB/position/layer)
sh ./start.sh qwen3.5-9b       # Qwen3.5-9B, 128K context (dense BF16, 18 GiB wired resident at load)
```

`start.sh` is the recommended entry point: it turns on the prefix cache, the web
console, LAN sharing, three model directories and a password, all of which are
easy to forget one at a time. Without a script the same server is:

```sh
./cwick serve --root ./zai-org/GLM-5.3-Flash --port 8001 --web --prefix-cache-dir ./pfxcache
```

The binary's own default port is 8000; `start.sh` serves 8001. Switching models
later needs no restart — `POST /v1/models/switch` unloads one and loads another
in the same process.

## Usage

### Serve, run, inspect

```sh
./cwick run    --root ./zai-org/GLM-5.3-Flash --prompt "Explain MoE routing" --max-new 128
./cwick serve  --root ./zai-org/GLM-5.3-Flash --port 8001 --web --prefix-cache-dir ./pfxcache
./cwick doctor --root ./zai-org/GLM-5.3-Flash     # reads every resident byte and reports the plan
```

### Chat over HTTP

```sh
# single-user login (start.sh sets the password to cwick-local)
TOKEN=$(curl -s localhost:8001/v1/auth/login -H 'Content-Type: application/json' \
  -d '{"password":"cwick-local"}' | sed -n 's/.*"token":"\([^"]*\)".*/\1/p')

curl -N localhost:8001/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","stream":true,"max_tokens":128,
       "messages":[{"role":"user","content":"Explain MoE routing in one paragraph."}]}'
```

Streaming and non-streaming use the same request; `reasoning_effort` adds a
thinking depth, and a `tools` array turns the reply into `tool_calls`. Any
OpenAI-compatible client works — a desktop chat app, a coding agent, or the
built-in console. `POST /v1/responses` is the same engine in the Responses
dialect (`input` items in, a `response` object or `response.*` events out), for
clients that speak that API; the field and event lists are in
[docs/SERVER.md](docs/SERVER.md#responses-api). `POST /v1/messages` is the
Anthropic one — the shape Claude Code speaks — with `system`, content blocks,
`tool_use`/`tool_result`, `thinking.budget_tokens` and a
`message_start…message_stop` stream. All three dialects reach the same engine,
and both named API keys and `x-api-key` authenticate them. The built-in console
speaks all three: its **接口** selector decides which endpoint the next turn goes
to, and every dialect feeds the same bubbles, tool cards and stats line.

### Logs and metrics

The server prints every diagnostic to **stdout**, one line per event, each with a
wall-clock time and — once a request is in flight — that request's id, so a
concurrent log is still greppable:

```
cwick: 2026-10-29 21:52:01.902 [info +183.71s] [req 4] stream=true thinking=高 (effort=High)
cwick: 2026-10-29 21:52:02.654 [info +184.46s] [req 4] [prefill] ~310/333 tokens (93.1%), layer 56/60 — 4.5 tok/s, eta 5s
cwick: 2026-10-29 21:53:15.215 [info +251.55s] [req 6] decode(content) 12/24 tokens (50.0%) — 690 ms/step, 1.45 tok/s, eta 8s
cwick: 2026-10-29 21:53:30.104 [info +266.44s] [req 6] POST /v1/chat/completions -> 200 in 12014.7ms prompt=201 completion=18 finish=stop think=12 tools=0 content=6
```

A decode line names which part of the answer it is counting — `decode(thinking)`,
`decode(content)` or `decode(tools)`. A prefill line has no such split, so it
reports a layer instead, and its token count is marked `~` because it is an
estimate derived from layers done rather than a count. A transition closes out
the finished phase with its token count, elapsed time and rate
(`decode phase: thinking -> content after 8 tokens (1.1s, 7.01 tok/s)`); the
hint is pushed to all four backends, so the label on every progress line and
the transition always agree (DECISIONS §56).

`--verbose` adds per-step detail and the memory budget; `--trace` also prints the
rendered prompt, the output and every tool call (off by default: it puts user
content on stdout). `Ctrl+C` stops cleanly at any time — mid-generation the
in-flight request ends with a 503, snapshots are flushed, and the last line says
so.

`/v1/metrics` carries the same state under vLLM-compatible names, so it can be
scraped or watched in the console's **控制台** tab.

### Web console, model library and the app market

`--web` serves a single-page console from the binary itself — no assets on disk,
no second process, same port as the API. Five tabs: **对话** chat (streamed over
`/v1/chat/completions`, `/v1/responses` or `/v1/messages` — the composer's
**接口** selector picks the dialect per turn, and all three are rendered by the
same code path, with reasoning and tool calls shown separately),
**应用** the app market, **模型** the model library, **日志** this process's own
stdout and stderr (a bounded ring, so the browser follows a running serve), and
**设置** — the console and the settings in one place: live `/v1/metrics` with
four-minute trend charts (decode speed, footprint, both cache hit rates, TTFT),
a Prometheus series browser that lists every exported metric with its HELP/TYPE
and current value and can plot any four of them on private axes (raw value or
per-second delta), every label from `cwick:model_info`, the live
`cwick:model_loaded` bit, generation preferences, read-only runtime config and
switching the resident local model (entries this build cannot load are shown
as unsupported and cannot be selected). Every request is same-origin;
there is no option to point the console at an external OpenAI-compatible
service. Named API keys can still be minted for other clients to call *this*
cwick instance (see authentication below).

The **模型** tab lists what is already downloaded under `./models`, the live
download queue with progress, rate and ETA, and a read-only search over the
model catalog. A download is a server-side job: pause it, resume it, retry a
failed file or cancel it; partial files survive a restart and resume with a
Range request. Repositories of a few hundred GiB and shards above 100 GiB are
handled with 64-bit sizes throughout. `docs/SERVER.md` has the five endpoints
and the on-disk layout.

The market ships two H5 apps in `apps/`, each with its own sqlite database and
its own manifest. In **tic-tac-toe** the human plays X first; every machine
move is a `/v1/systemone` decision asking which empty cell gives O the greatest
probability of winning, with per-cell tactical facts in the state and the full
probability readout in the right-hand column. In the **3D kart** the model
places every obstacle on the Mobius strip in real time: each decision shuffles
its twelve option letters, draws a variation seed and a pressure point, and
asks for the hardest possible obstacle, while the sidebar shows obstacle and
kart statistics. Both call `POST /v1/systemone`; the chat tab goes to whichever
endpoint its selector names. Each app lays itself out for a phone screen as well as
a desktop one; the console only hosts them. With `--lan` each app has a QR code,
so a phone on the same Wi-Fi can open it without knowing the address.

What they ask the model for is a decision, not prose, and the state each one
keeps (a game, its decision log, a lap) goes into its own sqlite file under
`--store`.

`--password` (or `--lan`, which requires one) turns on single-user login: a
PBKDF2 hash, a Bearer token and an HttpOnly session cookie. For third-party agent
platforms, `--api-key NAME` issues a named key (printed once at creation, only
its hash is stored) that can be revoked with `DELETE /v1/auth/keys`.

### Concurrency and queueing

Generation is single-slot: one answer at a time, because two would halve each
other's bandwidth and neither would finish. Everything else is concurrent — a slow
generation does not block `/v1/metrics`, `/v1/models`, the log stream or the web
console, which is what makes watching a 30-minute prefill possible at all.

Requests queue in arrival order and each is served to completion. A queued client
that has disconnected is detected and dropped before it costs a prefill; a model
swap in progress answers 503 with `Retry-After`, and so does shutdown, so a caller
always knows whether to retry instead of guessing at a timeout.

## Configuration

Most knobs are flags on `serve` (or `run`); the ones that matter for a long-running
serve also have an environment twin:

| flag | default | what it does |
|---|---|---|
| `--root DIR` | — | model root: `config.json` plus the published shards |
| `--host` / `--port` | `127.0.0.1` / `8000` | `start.sh` binds `0.0.0.0:8001` |
| `--context N` | 131072 | KV capacity in tokens (>100K); `run` defaults to 4096. `start.sh` uses 500000, or 32768 for MiniMax |
| `--max-tokens N` | 1000000 | default `max_tokens` for requests that omit it, clamped to whatever the window leaves; `start.sh` passes 8192 |
| `--backend metal\|cpu` | metal on Apple Silicon | expert GEMV, batched projections, and the whole Qwen3.5 decode step on the GPU |
| `--mem-fraction PCT` | 80 | share of host RAM the process may occupy |
| `--max-memory GiB` | — | absolute ceiling; wins when it is smaller |
| `--allow-swap` | off | stop wiring the spine and the expert arena (debugging only) |
| `--prefix-cache-dir` / `--prefix-cache-gib` | off / 8 | KV snapshots on disk and their budget |
| `--warmup N` | 0 | silent N-token generation before the port opens, to pay the cold start |
| `--verbose` / `--trace` | off | per-step detail / prompts and outputs on stdout |
| `--web` / `--lan` | off | embedded console / bind `0.0.0.0` + QR sharing + forced login |
| `--jev` / `--no-jev` | off | enable `POST|GET /v1/systemone`; `start.sh` turns it on |
| `--password P` / `--api-key NAME` | — | single-user login / named key for agent platforms |
| `--models-dir DIR` | — | extra checkpoints to list (up to 8); switching loads one at a time |
| `--store DIR` | `~/.cwick/apps` | one sqlite file per app plus `auth.db` |

Environment twins: `CWICK_MEM_PCT`, `CWICK_MEM_MAX_GIB`, `CWICK_STORE`,
`CWICK_PASSWORD`, `CWICK_API_KEY`, `CWICK_WARMUP`, `CWICK_WEB`, `CWICK_METAL`,
`CWICK_SEED` (sampling seed, so a run is repeatable), `CWICK_HUB_ENDPOINT` /
`HF_ENDPOINT` (model catalog host for the **模型** tab). The library always
writes to `./models`; it is in `.gitignore`.

## API

| method | path | notes |
|---|---|---|
| GET | `/v1/models` | the resident checkpoint plus anything found under `--models-dir`; each entry carries `supported` / `unsupported_reason` |
| POST | `/v1/models/switch` | validate the target first, then unload one and map another; 400 for an architecture this build cannot load (resident model untouched), 409 while generating or queued |
| POST | `/v1/completions` | raw text completion, streaming or not |
| POST | `/v1/chat/completions` | messages, `stream`, `tools`, `reasoning_effort` |
| POST | `/v1/responses` | Responses dialect: `input` items in, a `response` object / `response.*` events out |
| POST | `/v1/messages` | Anthropic dialect: `system` + content blocks in, a `message` object / `message_start…message_stop` out |
| POST | `/v1/systemone` | typed decisions (choice / score / noul) read from one prefill; requires `--jev`; also `/inference/v1/systemone` |
| GET | `/v1/systemone` | the live decision adapter: family, prompt grammar, readable letters, limits |
| GET | `/v1/hub/models` | checkpoints already under `./models`, with size and completeness |
| DELETE | `/v1/hub/models/<name>` | delete a downloaded checkpoint (recursive, confined to `./models`); 409 for the resident model or an active download |
| GET | `/v1/hub/downloads` | live download jobs: progress, rate, ETA, state, error |
| GET | `/v1/hub/search?q=…` | read-only search over the model catalog (default hf-mirror) |
| POST | `/v1/hub/download` | start/resume `{"repo":"owner/name"}` into `./models/owner/name` |
| POST | `/v1/hub/download/<id>/<action>` | `pause` / `resume` / `retry` / `cancel` a job |
| GET | `/v1/metrics` | also `/metrics` and `/health`; vLLM-compatible names |
| GET | `/v1/config` | what the settings panel edits |
| POST | `/v1/auth/login` | password → Bearer token (12 h) + `cwick_session` cookie |
| POST/GET/DELETE | `/v1/auth/keys` | mint, list and revoke API keys (session required) |
| GET | `/v1/logs`, `/v1/logs/stream` | this process's stdout/stderr, tail or follow |
| GET | `/v1/store` | app storage layout, files and row counts |
| PUT/POST/GET | `/v1/apps/<id>/{state,events,games}` | per-app sqlite, so clearing browser data does not erase history |
| GET | `/v1/share/qr` | QR for LAN sharing |

The full reference — auth rules, error shapes, storage schema, app endpoints — is
in [docs/SERVER.md](docs/SERVER.md).

### Decisions: `POST /v1/systemone`

`--jev` adds a second way to use the resident model. Instead of asking it to
write an answer, the server reads the distribution at the answer position and
returns a typed one: which option, how likely each one was, how peaked the
answer was. One question costs one prefill and generates nothing.

```bash
curl -s http://127.0.0.1:8001/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"My order arrived broken and I need it replaced today.",
       "questions":{"intent":{"type":"choice","instructions":"What does the customer want?",
                              "criteria":{"refund":"Money back","replace":"A replacement",
                                          "info":"Just information"}}}}'
```

```json
{"object":"systemone.decision","model":"glm-5.3-flash",
 "adapter":{"family":"glm","prompt":"glm-chat-closed","readout":"letter",
            "letters":26,"letters_total":26,"first_missing":null,"space_lead":true},
 "answers":{"intent":{"type":"choice","choice":"replace",
                      "probabilities":{"refund":0.0002,"replace":0.9998,"info":0.000005},
                      "confidence":0.999}},
 "usage":{"input_tokens":118,"cached_input_tokens":0,"output_tokens":0,
          "prefilled_tokens":118,"reused_tokens":0}}
```

Three question types: `choice` returns the winning key and the distribution over
every option; `score` returns the probability-weighted level index plus a legend;
`noul` returns the probability that the statement holds (its two sides are named
by `criteria` — `{"true": …, "false": …}`). `state` may be a string or any JSON
value. `instructions` is required on every question. Up to ten questions per
request, up to 26 options each. The published path is `/inference/v1/systemone`;
`/v1/systemone` is the short form.

**It follows the model.** Each supported checkpoint gets its own prompt grammar
and its own answer spelling: GLM and DeepSeek get the turn with the thinking
block already closed (`glm-chat-closed`, `deepseek-chat-closed`), MiniMax gets
its plain chat turn (`minimax-chat-plain`). The adapter is resolved when the
tokenizer loads and again after every `POST /v1/models/switch`, so switching the
model switches the interface underneath this endpoint — there is no path where a
MiniMax prompt is read out of GLM logits. `GET /v1/systemone` publishes the live
one; `make jevprobe` prints the per-letter token table for a tokenizer without
loading any weights (all three shipped checkpoints: 26/26 letters, space-led).

A request that names a model the server no longer serves is refused with a 400
naming the resident one. That is deliberate: an app that cached a model id
across a switch must not receive the new model's decision under the old model's
name.

What the numbers mean matters more than the JSON shape. The probability is the
model's own next-token distribution restricted to the option letters — it is
**not calibrated**, and `confidence` is a concentration score (0 = all options
equal, 1 = one option took everything), not an estimate of being right. A flat
distribution is the honest answer to a question the model cannot decide, which
is why the whole distribution is returned rather than only its argmax.

Cost is one prefill per question; every option of a question is read from that
same pass, so a nine-cell board costs the same as a two-way choice. On this
machine a 550-token decision on MiniMax takes 95.5 s, and a three-question
request on the 320B checkpoint takes 306 s — the prefill is essentially all of
it, because the endpoint generates no tokens and there is nothing else to
amortise.

**A request's questions share one prefill.** Every question in a request begins
with the same `system` + `<state>` turn, so the endpoint encodes all of them
first, takes the longest common *token* prefix, prefills it once and resumes
from it for each question. Three questions sharing a 348-token prefix (1145
input tokens) prefilled 449: **213.8 s → 102.0 s (2.10×)**, with every
probability identical to the run without reuse.

That resume is only switched on where it is exact, which is measurement, not
policy: on MiniMax the in-place resume reproduces the distribution to
`|dp| = 0.0`, while forcing it on GLM and DeepSeek moved a probability by 0.41
and 0.56 and changed answers (the compressed KV and the partially-filled
attention pools are not addressed by position alone). Those two keep one full
prefill per question; `usage` still reports `input_tokens`,
`cached_input_tokens`, `prefilled_tokens` and `reused_tokens`, so the saving — or
its absence — is visible per request instead of inferred.

Both hosted apps use this endpoint for their model decisions: tic-tac-toe
asks which empty cell has the highest win probability (one letter per cell,
the argmax is the move), and the kart asks for the obstacle kind and lane that
maximise lap difficulty (a 12-way choice whose order is shuffled on every
request). Neither generates an object the browser has to parse, and each keeps
a local fallback for a refused or slow decision.

## Models, context and memory

| model | context (`start.sh`) | KV cache | experts per step | decode |
|---|---|---|---|---|
| GLM-5.3-flash | 500K | MLA latent + DeltaNet state | 8 of 288 × 42 layers | 3.6 tok/s warm, 1.7 streaming |
| DeepSeek-V4.1-Flash-1017 | 500K | compressed MLA | 6 of 384 × 40 layers | 2.8 tok/s |
| MiniMax-M3.1-Flash | 32K | bf16 GQA, 2.3 KiB/position/layer | 4 of 128 × 57 layers | 1.48 tok/s |
| Qwen3.5-9B | 128K | f32 GQA 64 KiB/position + 50 MiB DeltaNet state | none — dense, held wholly in RAM | **20.0 tok/s** |

Context is not free and it is not abstract: 500K positions on MiniMax would be
64 GiB of KV, which comes straight out of the expert cache and therefore out of
the token rate. That is why `start.sh minimax` caps it at 32K — measured, not
assumed.

The memory plan is one decision made at startup: **80% of RAM is the ceiling**,
the resident spine and the KV cache come out of it first, and whatever is left
becomes the expert pin cache. When the ceiling is approached the runtime evicts
cache, with a cooldown so the valve cannot collapse to nothing. `--mem-fraction`
moves the ceiling and `--max-memory GiB` pins it to an absolute number, for when
something else on the machine needs the RAM. The point is that steady-state
footprint stays under the ceiling, `swap` stays flat, and the compressor never
holds the working set — a step that hits a compressed page looks exactly like a
step that hit the SSD, and is just as slow.

Which is why the pin cache is *not* filled to the ceiling just because it fits:
past roughly a quarter of RAM the compressor starts taking pages and the whole
step slows down even as the hit rate rises, so the cap is `host / 6` and the
numbers behind it are in [docs/DECISIONS.md](docs/DECISIONS.md) §27.

Staying under the ceiling is not enough on its own, so the spine and the expert
arena are **wired** by default: the VM can neither compress nor page the bytes
every token reads. `--allow-swap` opts out, and exists for debugging a plan that
does not fit, not for running one.

## Long prompts and the prefix cache

An agent session resends the same multi-thousand-token system prompt every turn.
Re-prefilling it *is* the time to first token, so the engine snapshots the
inference state at the end of each prompt and replays it when the next request
starts with the same tokens. Same 403-token prompt, twice:

| request | wall | what happened |
|---|---|---|
| first (cold) | 169.4 s | full prefill, then `prefix cached: 403 tokens (83 ms)` |
| second (identical) | **9.95 s** | `prefix cache hit: 402 of 403 prompt tokens replayed` — the 9.95 s is 16 decode steps |

A snapshot is a checkpoint of the inference state written under
`--prefix-cache-dir`, not a cached copy of the text: 29,798 tokens of prompt is
~0.9 GB of KV and reads back in ~0.3 s against hours of prefill. Eviction is
deliberately not LRU: a session's prompt grows every turn, so age-based eviction
throws away the short base prompt that every turn still needs. Entries are ranked
by replays served, then by length, and only then by age.

## Thinking, streaming and tools

A request that sends no `reasoning_effort` (or `thinking` / `reasoning` /
`enable_thinking`) field **does not think**: the default is level 关, exactly
what an explicit `reasoning_effort:"none"` asks for. A client that never heard
of the knob gets an answer, not a 32768-token reasoning block. Thinking depth
is a request field, mapped to a token ceiling that the engine **enforces
itself**: when the ceiling is reached the engine writes the closing marker
into both the output and the context, so a turn that ran out of budget still
answers instead of returning `content=""`.

| level | 关 | 低 | 中 | 高 | 超高 | 极致 |
|---|---|---|---|---|---|---|
| budget | 0 | 512 | 2048 | 8192 | 16384 | 32768 |

`reasoning_content` and `content` are separated on both the streaming and the
non-streaming path, from the same decision — the checkpoint's own reasoning
markers (`<think>` for GLM and DeepSeek, `<mm:think>` for MiniMax), not a fixed
string. Tool declarations are rendered into each checkpoint's own
`<tools>`/`<tool_call>` protocol and parsed back into OpenAI `tool_calls`; while a
tool call is being emitted, the stream is labelled `decode(tools)` rather than
leaking it into the answer.

A streaming tool call is **visible while it is being written**: once the model
has produced the function name, all three dialects open the call with an empty
`arguments` and then append argument shards (chat
`delta.tool_calls[].function.arguments`, Responses
`response.function_call_arguments.delta`, Messages `input_json_delta`).
Shards continue **inside** a parameter value, so a tool call that is one long
shell script or file body fills in progressively instead of appearing whole at
the end. Concatenating the shards gives byte-for-byte the object in the final
payload; the private `delta.cwick_tool` progress field is still sent for the
bundled console. See DECISIONS §58 and §59.

## Speed

Measured on an M3 Max, 128 GiB, `--backend metal`, one commit — not a fresh
benchmark of every later change. Conditions matter more than the digits: a
"resident" row is a state where the experts that reply needs are already pinned,
a "streaming" row is the same request reading them from the SSD, and the two
differ by 2× on the same binary. The MiniMax rows are newer than the rest:
decode is the default 21 GiB cache from the work in
[docs/DECISIONS.md](docs/DECISIONS.md) §27, prefill is the decision-path work in
§44 (1024-token blocks, `--backend metal`).

| model | phase | wall | tok/s | note |
|---|---|---|---|---|
| GLM-5.3-flash | decode, experts resident | 275 ms | **3.63** | 10-token reply |
| GLM-5.3-flash | decode, streaming | 590 ms | 1.69 | 83 GiB cache, 80 steps |
| GLM-5.3-flash | prefill, 403 tokens cold | 169 s | 2.4 | the second identical turn is 9.95 s |
| DeepSeek-V4.1 | decode | 360 ms | 2.78 | steady state after the memory rework |
| MiniMax-M3.1 | decode | 677 ms | **1.48** | 23 steps, default 21 GiB cache |
| MiniMax-M3.1 | prefill, 550-token decision | 95.5 s | **5.8** | one 1024-token block; 109.3 s at 512 |
| Qwen3.5-9B | decode, whole step on the GPU | 49 ms | **20.4** | 32 layers in one command buffer (DECISIONS §61); 7.0 tok/s with `CWICK_Q35_GPU=0` |
| Qwen3.5-9B | decode, CPU pool | 139 ms | **7.2** | all 17.98 GiB resident and wired; 6.2 tok/s at a 31K context |
| Qwen3.5-9B | prefill, 760-token prompt | 10.0 s | **76.2** | 512-position blocks; both attention passes are GEMMs |
| Qwen3.5-9B | prefill, 7,600-token prompt | 94.8 s | **80.2** | the same rate at 10x the context |
| Qwen3.5-9B | prefill, 14,060-token prompt | 184 s | **76.2** | flat: both attention passes are GEMMs (DECISIONS §54) |
| Qwen3.5-9B | prefill, 30,970-token prompt | 436 s | **71.0** | 94% of the flat rate at 30K — a block spends 2.3 s in attention, 4.5 s in projections |
| Qwen3.5-9B | decision, 3 questions / 525 tokens | 9.44 s | **55.6** | three prefills, no decode (see [docs/PERF.md](docs/PERF.md)) |

### What sets the token rate

The three rates are far apart for one reason: they move different numbers of bytes
per step. Same method as the table above:

| model | best decode | bytes per step | what sets the ceiling |
|---|---|---|---|
| GLM-5.3-flash | 3.63 tok/s | ~8.06 GiB of experts per step | the expert bank does not fit in RAM |
| DeepSeek-V4.1 | 2.78 tok/s | ~18 GB, mostly bf16 dense | expert fp4 GEMV measures 58.5 GB/s |
| MiniMax-M3.1 | 1.48 tok/s | ~5 GiB of misses | same-shard read ceiling; a cache above 30 GiB feeds the compressor |
| Qwen3.5-9B | 20.0 tok/s | ~16.5 GB, all resident | the GPU's path to memory: 341 GB/s of a measured 368 (§61) |

The limits are arithmetic rather than a missing optimisation: an MoE step moves 20–30 GB,
the CPU cluster reads at most 123 GB/s, and the engine serves one token per weight read
(the full derivation is in [docs/PERF.md](docs/PERF.md)). The dense model is where that
argument breaks — 16.5 GB over the GPU's 368 GB/s is 45 ms a step, and that is the rate
it now runs at. What the same loop already bought elsewhere is the prefill side — 550
tokens in 95.5 s on MiniMax (5.8 tok/s), and a three-question decision sharing one prefix
in 102 s instead of 214 s. What is left for the dense model is one lever only: more than
one token per weight read.

That lever is built and measured, and it does not pay yet. `CWICK_Q35_VERIFY=k` runs k
consecutive positions through a single weight pass, and `CWICK_Q35_GPU_CHECK=1` finds every
row of it bitwise equal to the same GPU path run one row at a time — the recurrence state
rolls through k+1 slots, so keeping the accepted prefix is one `memmove`. That pass started
at 6.4 ms per extra row and is now 4.9 (81.8 ms at k = 8 against 47.4), because ablating the
kernel showed the marginal row was activation traffic through threadgroup memory rather than
arithmetic — the multiply-adds are 1.8 ms of a 65.8 ms pass — and reading the activations
straight out of device memory removed the staging and its barriers. The only drafter on hand
that reads no weights — continuing from an earlier occurrence of the last 2-6 tokens —
returns 1.08 to 1.24 tokens per pass, which is measurably slower end to end than not
drafting at all. The learned head on disk is now bound and measured rather than priced: it
costs 2.52 GB per draft token because it shares the 2.03 GB vocabulary projection (7.0 ms,
or ~1.7 ms against a candidate slice), and it is right 35.6% of the time at depth 1, 12.5%
at depth 2 and 0% at depth 3 — so every shape of `CWICK_Q35_SPEC` is slower than one row per
step (20.19 tok/s against 20.74) and it stays off by default. What blocks 40 is no longer
the price of a draft but the number of tokens one weight pass can yield: 40 needs 2.08, the
drafter tops out at 1.40, and that ceiling is a property of the weights. Every part and the
arithmetic are in [docs/DECISIONS.md](docs/DECISIONS.md) §62 through §65.

## Verification

Beta, and it moves fast. Every change is measured on this machine before it is
kept, and the ones that lost are written down in
[docs/DECISIONS.md](docs/DECISIONS.md) — including the ones that looked obvious.

```sh
make test        # 678 checks: engines, sampler, prefix cache, server, web UI, apps
make bench       # kernel timings without loading a checkpoint
```

Correctness is anchored, not eyeballed: a fixed prompt must produce a fixed
top-1 token id and logit, and every kernel change is cross-checked against a
reference implementation to 1e-5 (the fp8 tiled GEMM measures 8.3e-6, the FP4
grouped kernel is bit-identical to the single-row one). `cwick doctor` reads
every resident byte plus a strided sample of the expert bank, so a corrupt shard
is found before a request is, and `/v1/metrics` publishes the same state the logs
print.

## Detailed Guides

Deep documentation lives in `docs/`, and every page switches between
[English](docs/ARCHITECTURE.md) and [简体中文](docs/ARCHITECTURE.zh-CN.md) on its
first line:

- [Runtime architecture and component map](docs/ARCHITECTURE.md)
- [Memory-bounded streaming and the storage split](docs/STORAGE.md)
- [Measured performance, phase by phase](docs/PERF.md)
- [Decisions: what was tried, rejected, and why](docs/DECISIONS.md)
- [OpenAI-compatible server reference](docs/SERVER.md) — API, auth, apps, storage
- [The DeepSeek backend](docs/DEEPSEEK-PORT.md)
- [Precision and per-checkpoint quantization](docs/PRECISION.md)
- [The verification suite](tests/test_cwick.c)

## FAQ

- **The first tokens are slow and the log shows a low cache hit.** A fresh
  process starts with an empty expert arena; `--warmup 32` pays that before the
  port opens (1013 → 590 ms/step, 17% → 68% hit on this build).
- **Every turn re-prefills the same prompt.** Nothing is being reused: check that
  `--prefix-cache-dir` is set and that the prompt really starts with the same
  tokens (a changed system prompt or timestamp breaks the prefix).
- **`swap` climbs and the machine gets hot.** Something else claimed the RAM:
  lower `--mem-fraction` or set `--max-memory`. Never run two cwick processes on
  one Mac, even with `CWICK_ALLOW_MULTI=1` — they evict each other's pages and
  both slow down by 2–3×.
- **A request returns 503 with `Retry-After`.** A model switch or a shutdown is in
  progress. Retry after the header's delay; a queued request is not an error and
  will be served.
- **Metal printed a warning and the output is identical but slower.** No usable
  GPU device was found; cwick falls back to the CPU GEMV rather than failing a
  generation. `--backend cpu` does the same on purpose.
- **A `doctor` run takes minutes.** It reads the whole resident spine and samples
  the expert bank; `--deep` reads the bank in full instead of sampling it.

## Directory Layout

```text
cwick/
├── cwick              the built binary
├── src/               the engine — C11, no third-party code
│   ├── main.c         CLI: setup | run | serve | doctor, and flag parsing
│   ├── server.c       HTTP, SSE, queue, auth, metrics, app-storage routes
│   ├── engine.c       request → prompt render → prefill/decode → stream
│   ├── spine.c        resident weights, memory plan, expert pin cache
│   ├── stream.c       F_NOCACHE SSD streaming, eviction, prefetch
│   ├── attention.c    MLA / GQA / DeltaNet kernels (NEON)
│   ├── moe.c          router and expert GEMV/GEMM
│   ├── ds_*.c         DeepSeek-V4.1 backend: FP4 experts, FP8 attention
│   ├── q35_*.c        Qwen3.5 backend: Gated DeltaNet + gated GQA, dense BF16
│   ├── mm_*.c         MiniMax-M3.1 backend: NVFP4 experts, bf16 GQA
│   ├── metal*.c/.m    Metal kernels and the zero-copy arena
│   ├── pfx.c          prefix-cache snapshots on disk
│   ├── jev.c          decision endpoint: three prompt grammars, letter readout,
│   │                  in-request shared prefix
│   ├── hub.c          model library: download jobs and catalog search
│   ├── store.c        one sqlite file per app
│   └── webui*.c       the console, embedded at build time
├── web/               console source: index.html, app.js, app.css, login.html
├── apps/              two H5 apps plus manifest.json
├── docs/              architecture, storage, performance, decisions, server, precision
├── tests/             test_cwick.c — 651 checks
├── tools/             standalone C probes: kernels, layer tests, model runners
├── start.sh           recommended entry point (prefix cache + web + LAN + password)
└── download_model.sh  resumable checkpoint download
```

## License

MIT. See [LICENSE](LICENSE).
