# Architecture — one C process, three checkpoints, one Mac

<p align="center"><a href="ARCHITECTURE.md">English</a> · <a href="ARCHITECTURE.zh-CN.md">简体中文</a></p>

CWICK runs complete, never-pruned checkpoints on a single Apple Silicon Mac: GLM-5.3-flash
(≈306 GiB), DeepSeek-V4.1-Flash-1017 (≈475 GiB) and MiniMax-M3.1-Flash (≈233 GiB) — one
binary, no helper processes, no conversion step. This document maps each C component to
the part of the model it implements, and says which parts are shared and which belong to
one backend.

## One-process design

`main.c` owns the CLI, `server.c` owns the HTTP surface, `engine.c` owns the request path
(render → prefill → decode → stream), `hub.c` owns the model library and `webui.c` serves
the console. There are no internal HTTP calls and no helper process per token: a request
entering `/v1/chat/completions` reaches the weights without leaving the address space.

## Model contract

`model.c` reads inert JSON, validates every performance-critical dimension, and fails
closed before opening any weight payload. The quality guards hold for all three
checkpoints:

- `tie_word_embeddings` must be false;
- `num_experts_per_token` is used exactly as the model defines it (top-k is never
  reduced) and every expert must be present in the corpus;
- a missing shard is a refusal to start, not a smaller model;
- dtype and block sizes are validated per backend rather than assumed: the three
  checkpoints do not share a quantization scheme

## Backends

A checkpoint is dispatched by family once, at load time; the request path then calls the
same two entry points (`prefill`, `decode`) on whichever backend is resident.

| family | files | weights | attention |
|---|---|---|---|
| GLM-5.3-flash | `spine.c` `attention.c` `moe.c` | FP8-E4M3 + per-128×128 F32 `weight_scale_inv` | MLA + DeltaNet (KDA), DSA pools |
| DeepSeek-V4.1 | `ds_model.c` `ds_quant.c` `ds_ops.c` `ds_engine.c` `ds_prompt.c` | FP4-E2M1 experts + E8M0, FP8 attention + block scale | compressed MLA with an indexer |
| MiniMax-M3.1 | `mm_model.c` `mm_engine.c` `mm_prompt.c` (+ `mm.h`) | NVFP4 experts, FP8 dense | bf16 GQA, 4 KV heads |

`pfx.c`, `stream.c`, `sample.c`, `tokenizer.c`, `tools.c` and the server are shared by all
three; only the loader, the dequantizer, the prompt template and the kernels differ.

Adding a fourth checkpoint means a new loader, a new dequantizer, a new prompt template
and new kernels. It is not a config change — which is why the list stays at three.

## Weight spine (the GLM path)

`spine.c` memory-maps the checkpoint:

```
config.json
resident/tensors/<name>.bin        BF16 dense spine (row-major [out,in])
resident/experts/expert-L-E.bin     one canonical raw file per layer/expert
resident-int8/tensors/<name>.i8     optional int8 data (+ .s fp16 scales)
```

Per-layer dense tensors: input norm, attention-residual norm/proj, post-attention norm,
MLP-residual norm/proj, and either the KDA set (`A_log`, `dt_bias`, q/k/v conv1d,
`o_norm`, q/k/v/g/fa/fb/b projections, `o_proj`) or the MLA set (`q_a/q_b`, `kv_a/kv_b`,
norms, `g_proj`, `o_proj`), then the dense MLP (layer 0) or the MoE spine (gate +
correction bias + routed down/up/norm + shared gate/up/down). The expert corpus is
checked for completeness at open — a missing expert fails closed (never-pruned rule).

The other two backends map the published safetensors shards directly and keep the same
split: everything read on every token stays resident, routed experts stream.

## Compute

`tensor.c` implements:

- fp32 GEMMs through Accelerate (`cblas_sgemm`), with BF16 / row-int8 weights
  dequantized into an execution arena.
- RMSNorm, softmax and SiTU with a fixed operation order:
  `situ(g,u) = tanh(g/beta)*beta*sigmoid(g) * tanh(u/linear_beta)*linear_beta`,
  `rms_norm = w * x / sqrt(mean(x^2)+eps)`.
- A NEON MXFP4 expert GEMV kernel for the routed-expert path: packed e2m1 nibbles +
  e8m0 scales, a 256-row scale LUT folded into fp32 top-16 halves, `vzip` interleave
  into element order, and `vqtbl1q_u8` + `vshll` reconstruction. A scalar reference is
  kept for the unit test, which cross-checks both paths.

Per backend, on top of that:

- **Metal** (`metal.c`, `metal_shader.c`, `metal_objc.m`): the zero-copy arena. Expert
  blobs and batched projections are bound to a dispatch without a copy; the arena is
  released with `munmap` on a model switch. GLM and DeepSeek use it; MiniMax measured
  slower at these shapes and stays on the NEON pool.
- **DeepSeek**: fp4 grouped GEMM (4 activations per pass, 2 vector accumulators each),
  fp8 dense GEMM tiled 4 rows × 4 activations, and a local e4m3→f32 table because the
  conversion helper's `build_luts` branch dominates a hot loop.
- **MiniMax**: the same shape of kernels over NVFP4 (block scale is a *divisor*) and fp8
  with per-row per-32 `U8(e8m0)` scales; everything runs a chunk at a time, never one
  token at a time — a 302-token prompt as 34,428 single-row GEMVs measured 67 GMAC/s
  against 228 GMAC/s for the chunked form.

## Attention

`attention.c` implements the GLM kernels; the other two backends have their own files
with the same interface.

- **MLA**: latent query (`q_a` + norm + `q_b`), latent KV (`kv_a_with_mqa` + norm +
  `kv_b`), RoPE on the rope slice, full attention over the KV cache
  (`scale = 1/sqrt(qk_nope+qk_rope)`), output gate (`sigmoid(g_proj)`), then `o_proj`.
  GQA maps q-heads to kv-heads.
- **KDA**: short convolution (4 taps, SiLU, three per-channel states), L2 normalized
  query/key (query additionally `/sqrt(head_dim)`), gated decay
  `gate_lower_bound * sigmoid(exp(A_log) * (gate_raw + dt_bias))`, the beta-scaled delta
  rule, per-head RMSNorm with `o_norm`, output gate, `o_proj`.
- **DeepSeek**: compressed KV plus an indexer that selects which positions the sparse
  attention reads; the indexer pools are per-request state and are snapshotted, not
  recomputed.
- **MiniMax**: bf16 GQA with 4 KV heads, RoPE applied only to the first 64 dimensions of
  each head, and `(1 + w)` norm weights (gemma-style) — a detail that silently yields the
  wrong scale if borrowed from another backend.

Prefill attention is the same math over a block of positions; the NEON inner loop is
what turned 398 ms/token into 30 ms/token on MiniMax, and it is the reason a long prompt
is bandwidth-bound rather than compute-bound.

## MoE

`moe.c` implements the exact router and expert math:

```
logits = gate(hidden)                    # [E]
scores = sigmoid(logits)
choice = scores + gate.e_score_correction_bias
top-k ids                                # num_experts_per_token, never reduced
weights = scores[ids] / (sum + 1e-20)    # moe_renormalize, routed_scaling_factor
routed_in = routed_down(hidden)          # [RH]
per expert: gate=w1(x), up=w3(x), h=situ(gate,up), y=w2(h), scaled by weight
out = routed_up(rms_norm(sum, routed_norm)) + shared_down(situ(shared_gate,
      shared_up))
```

Experts are loaded and combined in authoritative top-k order. On a prefill the selected
experts of a whole block are batched into one GEMM per expert; on a decode step they are
one GEMV each. That difference — block GEMM versus per-token GEMV — is worth up to 5×
and is the first thing to check when a prefill looks slow.

## Layer tape and decode

`engine.c` implements the per-token forward tape for the GLM path; `ds_engine.c` and
`mm_engine.c` are the same tape with their own norms, attention and quantized MLPs.

```
# per layer
attn_in = residual_combine(hidden, anchors, self_attn_res_norm * res_proj)
if layer % attn_res_block_size == 0: anchors.append(hidden); prefix = attn_out
else prefix = hidden + attn_out
normalized = rms_norm(attn_in, input_layernorm)
attn_out = attention(normalized)                    # MLA or KDA
mixed = residual_combine(prefix, anchors, mlp_res_norm * res_proj)
post = rms_norm(mixed, post_attention_layernorm)
mlp_out = dense_mlp(post) or moe(post)
hidden = prefix + mlp_out

# tail
mixed = residual_combine(hidden, anchors, output_attn_res_norm * res_proj)
logits = lm_head(rms_norm(mixed, final_norm))
```

The residual-anchor bank is per-token state (block residual): each token independently
accumulates its boundary-layer pre-attention rows. Decode samples from the last row's
logits (`t<=0` goes through argmax, which is what makes a run reproducible); prefill
advances in blocks — GLM sizes the block from its activation budget, DeepSeek and MiniMax
use `MM_CHUNK` / `DS_CHUNK` = 1024 — because the expert read is a per-block cost and a
tail block costs what a full one does.

## Server

`server.c` implements the OpenAI surface and its two siblings: `/v1/models`,
`/v1/models/switch`, `/v1/completions`, `/v1/chat/completions`, `/v1/responses`,
`/v1/messages` (+ `/models` aliases), SSE streaming (begin chunk → content deltas →
tool-call deltas → finish frame → optional usage frame → `data: [DONE]` for chat;
named events with no sentinel for the other two), and fail-closed 400s for anything not
honored exactly. Both extra endpoints are adapters, not a second pipeline: Responses
`input` items and Anthropic content blocks are translated into the same conversation the
chat validator produces, and only the envelope (`responses_body` / `resp_stream_*`,
`messages_body` / `msg_stream_*`) differs. One generation at
a time, in arrival order: the slot is single because two answers would halve each other's
bandwidth. Everything else — `/v1/metrics`, `/v1/models`, `/v1/config`, the log stream,
`/v1/hub/*`, the console — is answered while the slot is busy, which is what makes a
30-minute prefill watchable. A model switch or a shutdown answers 503 with `Retry-After`;
a queued client that disconnected is dropped before it costs a prefill. See
[SERVER.md](SERVER.md).

`metrics.c` publishes the same state under vLLM-compatible names; `auth.c` holds the
single-user session and named API keys (hash only); `logcap.c` keeps the ring that
`/v1/logs` and `/v1/logs/stream` serve.

Observability and lifecycle are part of the contract, not an afterthought:

- `log.c` — one line-atomic logger to stdout with `--verbose` (per-request and per-step
  detail) and `--trace` (prompts, outputs, tool calls) levels, every line carrying a
  wall-clock time and, once a request is in flight, its id;
- `main.c` — `sigaction` handlers without `SA_RESTART` plus a self-pipe, so a Ctrl+C
  breaks `poll()`/generation instead of waiting for the next request;
- `mem.c` — the load-time memory plan; `stream.c` — the trimmable expert pin set and the
  cooling valve that keeps the process under it.

## Decision endpoint

`jev.c` is the second way to use the resident model: instead of asking it to write, it
renders a decision prompt, prefills it, and reads the distribution at the answer
position over the option letters. No token is generated, so the whole latency is prefill.

Three parts, and each is per family rather than global:

- **grammar** — GLM and DeepSeek get the turn with the thinking block already closed
  (`glm-chat-closed`, `deepseek-chat-closed`), MiniMax gets its plain chat turn;
- **readout** — the letters are resolved against the resident tokenizer (26/26, space-led
  on all three shipped checkpoints), which is why switching models also switches the
  interface;
- **resume** — all questions of a request share the `system` + `<state>` turn, so the
  endpoint prefills the longest common *token* prefix once and resumes each question
  from there through `cwick_engine_prefill_logits_at()`. That resume is exact only where
  the cross-token state is addressed by position: MiniMax yes, GLM and DeepSeek no
  (`|dp|` 0.41 / 0.56, answers change), so those two pay one full prefill per question.
  The measurements are in [DECISIONS.md](DECISIONS.md) §44.

## Console and model library

`webui.c` serves a single page compiled into the binary (`src/webui_embed.c`, generated
from `web/` by `tools/embed_web`, never hand-edited), so `--web` needs no assets on disk
and no second process. Five tabs: chat over `/v1/chat/completions`, the app market, the
model library, the process's own stdout/stderr, and settings — which carries the live
metrics, the trend charts, a Prometheus series browser, the resident-model switch and
read-only config. Every request is same-origin; there is no outbound endpoint to point
the console at.

`hub.c` backs the library: `./models/<owner>/<name>/`, a server-side job table with
pause/resume/retry/cancel, `.incomplete` files resumed with a `Range` request, 64-bit
sizes throughout, and a read-only catalog search. Its five endpoints answer before the
generation slot, so a 400 GiB download never blocks a chat request.

## Tools (checkpoint protocol, not an invented one)

`tools.c` renders OpenAI tool declarations into the block GLM-5.3 itself was trained
with — copied verbatim from `zai-org/GLM-5.3-Flash/chat_template.jinja` (`<tools>` + one
JSON line per function, plus the `<tool_call><arg_key>/<arg_value>` instruction) — and
parses the model's answer back out. DeepSeek and MiniMax have their own renderers in
`ds_prompt.c` / `mm_prompt.c`. Decoding is never constrained, so malformed output is
refused rather than forwarded.

## What is intentionally not in this framework

- **No batched GPU prefill.** Prefill runs as chunked GEMMs on the CPU pool (or the
  Metal arena for the parts it wins); batching several requests into one forward is not
  implemented, and would trade latency for throughput on a machine serving one user.
- **No parallel tokenization.** One tokenizer, one request at a time.
- **No draft-model speculation on the default path.** Block verification exists
  (`MM_SPEC`) and is exact with respect to its own block arithmetic, but at n=8 a chunk
  GEMM can flip a near-tie in the argmax, and a block of 8 streams ~2× the expert bytes
  of one step — see [DECISIONS.md](DECISIONS.md) §28. It is off by default.

What is in place is the full math, the full storage contract, the full server contract,
and the Apple Silicon compute path (Accelerate + NEON + Metal). The verification suite
proves the pipeline on tiny synthetic checkpoints that exercise every layer kind.
