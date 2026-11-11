# Decisions — what was tried, what stands, and what must not be tried again

<p align="center"><a href="DECISIONS.md">English</a> · <a href="DECISIONS.zh-CN.md">简体中文</a></p>

Every entry here is a decision this codebase made, with the measurement that
justified it. Numbers are from this machine (M3 Max, 128 GiB, GLM-5.3-Flash)
unless marked *external*.

## 1. Expert prefetch stays opt-in (default off)

`CWICK_PREFETCH=1` issues speculative next-layer reads from a background
thread. Measured, 80 decode steps, warm cache, 83 GiB expert cache:

| arm | ms/step | stream | gemv | attn | hit rate |
|---|---|---|---|---|---|
| prefetch off | **590** | 271.6 | 151.1 | 134.4 | 68.5% |
| `CWICK_PREFETCH=1` | **2111** | 490.8 | 712.3 | 284.2 | 68.6% |

The prefetch arm reads 2.7x the bytes (647.6 vs 236.8 GiB, 95 396 vs 30 315
reads) for zero hit-rate gain, and the MoE phase explodes from cache/arena
thrash. (An earlier run measured 714 -> 1873 ms; the machine drifts ~20%
between sessions, so A/B runs must be same-session.)

Mechanism: not SSD bandwidth — **cores**. The demand path reads a whole layer
with a twelve-worker fork-join, and a fork-join GEMV is paced by its slowest
participant. A speculative thread at default QoS preempts those workers
mid-kernel, so the layer finishes slower than if the read had never been
issued. `pf_reader` therefore runs at `QOS_CLASS_BACKGROUND` (`stream.c`) — the
one class the scheduler will not let preempt `USER_INITIATED` work — and the
A/B above is *after* that change. The mechanism is now right and the prefetch
is still a loss, because the bytes are wrong, not the scheduling.

The prefetch gate (per-layer EMA recall, threshold 0.10, warmup 16 samples)
suppresses speculative reads on layers whose prediction is poor. It cannot fix
a predictor whose recall is structurally low; it only stops paying for it.

## 2. MXFP4 expert packs were rejected

Halving expert bytes would take stream 271.6 -> ~140 ms and gemv 151 -> ~75 ms
(projected 413 ms/step, 2.4 tok/s), but it costs a **290 GiB one-time repack**
and 4-bit weights that are not bit-exact. Rejected 2026-10-20. The packing
contract is kept in `tensor.c` (`mxfp4_gemv_rows`, the scalar reference and
`test_mxfp4_spec`) so the option can be revisited without re-deriving it:

```
packed:  U8 [rows, cols/2]   two e2m1 nibbles per byte, LOW nibble = even column
scales:  U8 [rows, cols/32]  e8m0, scale = 2^(s - 127)
nibble c -> {0, .5, 1, 1.5, 2, 3, 4, 6}[c & 7], negated when c & 8
```

A 16-byte group is 32 values: 16 even columns + 16 odd columns, one scale.

## 3. Cache sizing stops short of the ceiling

83 GiB of expert cache (the most the 80% budget allows) measured only ~10%
better than 64 GiB (714 vs 798 ms/step) while pushing the footprint to 102.04
of 102.40 GiB. Filling further is worse, not better: past ~45 GiB the resident
set is large enough that macOS can no longer keep hot pages uncompressed, and
a "cached" expert starts paying decompression on every read. `start.sh` runs
64 GiB and leaves headroom.

## 4. Prefill runs the MoE on the CPU

`batch windows: metal 0, ->cpu 190` is not a bug. Prefill windows fetch into
malloc'd staging blobs, and the Metal batch only adopts no-copy arena/cache
views, so prefill groups take the CPU fork-join path by design. Forcing the
GPU batch (`CWICK_PREFILL_METAL=1`) wins 23% of the compute but makes the SSD
about twice as slow — a net loss. Decode is unaffected: its views are
arena-backed by construction.

## 5. `--warmup` pays the cold tax before the port opens

Measured 2026-10-20, first request after a fresh serve, 253-token prompt,
10 generated tokens, identical output text in both arms:

| first request | decode ms/step | decode tok/s | hit rate | prefill |
|---|---|---|---|---|
| cold serve | 1304 | 0.77 | 13.2% | 96.9 s |
| `--warmup 32` | **939 (-28%)** | **1.07** | **39.6%** | 98.9 s |

The warmup itself costs 74.8 s for 32 tokens — that *is* the cold tax, moved
off the first client request. A 253-token prompt is 90% prefill, so the win
shows in decode rather than in end-to-end wall time; short prompts (and
prefix-cache hits) shift the balance toward decode. The warmup path never
touches the prefix cache, and a failure is logged and ignored.

## 6. Not to be tried again

Each of these fails for a structural reason, so re-testing is waste:

| proposal | why it loses |
|---|---|
| `madvise(WILLNEED)` on the expert path | the pages it advises are `F_NOCACHE` streams; the advice has no cache to warm |
| speculative "predicted ∪ authoritative" double read | doubles bytes to avoid a round trip that the fork-join already hides |
| 20-level priority bands / route-ahead promotion | priority fixes starvation, not prediction accuracy; demand gets starved in exchange |
| host↔device expert offload on unified memory | there is one physical memory; the copy is pure loss (*external*: +130 s per request) |
| dropping a preempted request | every request must finish; the single slot is the contract |
| `F_RDADVISE` together with `F_NOCACHE` | the advice is discarded with the cache |
| GEMM token tile 64 | neutral to slower at these shapes |
| one dispatch over all grouped experts | binding only the active experts is much faster |
| `mmap` fallback for expert reads on macOS | cold faults run ~10x slower than a pread pool (0.66 vs 6.6 GB/s, *external*) |
| thread-per-row GEMV | 7.5x off the weight-bandwidth floor |
| attention split-K with `nwg=8` | saves ~1 µs of attention and costs ~50 µs of the following FFN |
| closing fast-math to chase numerical drift | diagnostic only; it hides real bugs and slows the kernels |

## 7. How to re-measure the prefetch

One process, two runs, one environment variable apart:

```sh
cd cwick
CWICK_PROFILE=1 CWICK_EXPERT_CACHE_GIB=83 \
  ./cwick run --root ./zai-org/GLM-5.3-Flash --backend metal \
  --prompt "$(python3 -c 'print("You are a careful assistant. "*40)')" \
  --max-new 80
```

Read the `profile: N decode steps, X ms/step wall` line and the `stream` /
`gemv` / `attn` breakdown. **Use ~80 steps**: a 10-step run is 17.3% hit rate
(614 ms/step stream) and is a cold-start number, not steady state. Baseline is
590 ms/step; a prefetch arm that cannot beat it should not be defaulted on,
regardless of how the mechanism is tuned.

## 8. Decoding draws from the checkpoint's own distribution

`generation_config.json` is half of the training contract: the released
checkpoints ship `do_sample=true, temperature 1.0, top_p 0.95,
top_k 20`. Serving them with a plain argmax is serving temperature 0, and the
argmax of a sampling-trained model is the repetition loop — seen in the serve
log as one request that spent its whole 5282-token budget on
`Owner: Owner: Owner: ...` and, across five requests, three answers that were
empty because every token went into thinking.

So `src/sample.c` picks the token and the engine asks it every step:

| request | what runs |
|---|---|
| `temperature: 0` | argmax (what the bundled web app sends, and what it wants) |
| anything else | top-k (hard filter), then top-p over the survivors, then one draw |
| `presence_penalty` / `frequency_penalty` | subtracted from the raw logits first |

Fields the request omits keep the checkpoint's own values, so a client that
sends `temperature` alone does not silently mean temperature 0. A request that
sends none of them gets the checkpoint's; a checkpoint with no
`generation_config.json` gets greedy. `CWICK_GREEDY=1` forces greedy. Every
request logs the config it was served with (`sampling=... (generation_config.json)`
or `(greedy default)`), so "the answer repeated itself" is diagnosable next to
"the answer was drawn with t=1.0".

Cost, 150144-way vocabulary, one draw per decode step:

| config | ms/step |
|---|---|
| greedy | 0.17 |
| t=1.00 top_p=0.95 top_k=20 | **0.10** |
| same + presence 0.6 / frequency 0.4 | 0.13 |
| t=1.00 top_p=0.95, `top_k: 0` forced | 1.45 |

Two things keep those numbers off the step. A positive temperature preserves
the ranking, so the top-k window is gathered on the raw logits and the
temperature is applied to the K survivors — no full-vocabulary multiply, no
full-vocabulary exp. And when a client does force `top_k: 0`, the nucleus can
be wider than the 4096-token scan window (it was, in the benchmark above), and
a 150144-way sort costs ~10 ms — most of a step — so the boundary comes from a
histogram of the logit range: one pass, and the cut is exact to within one
bucket's width.

What this does *not* fix: sampling makes the repetition loop rare, not
impossible — a 250-token story still looped on one run in six
(`喷喷喷喷...`). Closing that needs a loop detector and a thinking budget,
which are separate decisions.

## 9. A response is valid UTF-8, or it is not a response

The detokenizer appends raw token bytes, and a byte-fallback token can be half
a character: one greedy run put a lone `0xA8` into an answer, and every client
that read the body as UTF-8 — which is what JSON is defined to be — failed to
parse the whole response. `cwick_json_stringify` now validates every sequence
and substitutes U+FFFD for the maximal ill-formed subpart (Unicode Table 3-8),
so the document always parses back.

Streaming needed no change for this: `utf8_incomplete_tail` already holds a
half character back until the next token completes it, so a character is never
split across two deltas and the substitution never mangles one.

## 10. A repetition loop ends the turn; it does not spend the budget

Sampling (§8) makes the attractor rare. The net under it is a loop guard that
watches the tokens a run *produces* and stops the generation when a fixed
block has been repeating.

The test is per period: `lmatch[p]` counts how many trailing positions repeat
at period p (`token[i] == token[i-p]`), and it fires when that count covers
both `lmin_occ` whole blocks and `lmin_run` tokens — 3 occurrences of a
period-64 block would be 192 wasted tokens, so the run floor dominates there;
16 identical tokens in a row is what fires for period 1. The second condition
is what keeps ordinary structure from tripping it: a markdown rule, an
ellipsis or "1. 2. 3." numbering repeats a handful of times and stops, and no
answer emits 16 identical tokens on purpose.

Two details that are easy to get wrong:

- **The prompt must not feed the guard.** `cwick_sampler_accept` counts a
  produced token for both penalties and the guard; `cwick_sampler_prime`
  counts a prompted token for penalties only. Priming the prompt through
  `accept` would feed the tail of the prompt to the guard, and a prompt that
  ends in a repeated block (a few-shot template, a table) would stop the
  answer before it starts. vLLM and HF penalise the whole visible text, so
  the penalty half of priming is required — only the guard half is prompt-blind.
- **The token that closes the loop is framing, not content.** The single path
  keeps it in `gen` (it is part of the generated text and the client sees it),
  but the batch path decrements `gen[i]` so a batched sequence reports the
  same `n_tokens` it would have without the guard.

Cost is one comparison per period per token — 64 compares against a ~15 ms
decode step — and the history is a 256-token ring inside the sampler, so there
is no allocation and no per-step pass over anything. Thresholds are
environment-tunable per generation (`CWICK_LOOP_DETECT=0` disables,
`CWICK_LOOP_MAX_PERIOD` / `CWICK_LOOP_MIN_RUN` / `CWICK_LOOP_MIN_OCC` tune the
rest; the period knob can only lower the compile-time maximum of 64 because
the ring is sized for it).

On the wire the stop is a plain `"stop"` — OpenAI has no loop finish reason —
with `finish=loop` in the access log and the period and run length in the
engine's summary line. Covered by 11 unit tests (the 16-vs-17 identical-token
boundary, period 3, latching, ordinary structure, 5000 pseudo-random tokens,
prompt priming, the disable switch, and the period ceiling).

## 11. The loop guard stops a repetition; the loop breaker has to break it

A health check of live agent traffic showed 7 of 13 requests ending in
`finish=loop`. The per-request lines made the shape clear: the guard fired
correctly (a period-4 block, 16 trailing tokens), but by then the model had
burned 450-570 thinking tokens, the client received zero content, and the
retry paid a full decode round again. Stopping the loop is not the same as
escaping it — the sampler keeps drawing from the same attractor every step
until the guard kills the turn.

So the sampler gained a breaker one notch below the guard. The loop guard
already tracks `lmatch[p]`, how many trailing produced tokens repeat exactly
at period p. While that count reaches `max(2*p, 8)` (capped at 12), pick()
bans the token that continues the pattern — the one sitting p positions back
in the history ring — by setting its logit to -INFINITY before penalties,
argmax, top_k and the nucleus see it. Everything downstream just sees a
masked logit; -INFINITY survives the penalty subtraction and exp() maps it
to exactly zero mass. The model has to say something else; if it drifts
back into the attractor the ban re-arms, and the guard (16+ trailing tokens)
stays as the backstop for anything the ban cannot shake.

The cap is not decoration: banning creates longer loops. With a period-1
attractor banned at 8, the run settles into "nine tokens + the escapee" —
a period-10 pattern whose proportional threshold (2*p = 20) would arm at
exactly where the guard fires, so the guard killed a run the breaker had
been "handling". The cap at 12 keeps every period armed strictly before the
guard can act, for every period.

`CWICK_LOOP_BREAK=0` (default on) disables it; `CWICK_LOOP_BREAK_MIN`
(default 8) moves the arming point. The engine prints `lb=N` — picks made
while a ban was armed — next to its summary, so a `stop` with `lb>0` is
visible as a recovery rather than a clean finish.

Two things it deliberately does not do: it does not apply a blanket
repetition penalty (requests that send no penalty fields keep the
checkpoint's sampling defaults — a hidden penalty would change every
answer, not the degenerate ones), and it does not count prompt tokens
(primed, not accepted — a prompt ending in a repeated block is not a loop).
Exact periodicity of 8 trailing tokens does not occur in ordinary text
(structure repeats *similar* blocks, not token-identical ones), so the
false-positive cost is one forced synonym on a stylistic run of
punctuation, against a measured 54% loop rate on agent traffic.

Verification: unit tests cover the ban (continuation refused at the argmax,
re-arming, the disable knob, short legitimate repeats never arming), and
the sampler distribution tests run with the breaker off — they feed the
same logits thousands of times, which is the loop signature by
construction. `make test` passes; seed-pinned metal generation stays
bit-identical to the pre-breaker build on clean prompts.

## 12. A thinking budget has to be written by the engine, not cut by the server

A health check of live agent traffic found one request that spent all 8096
tokens it had inside the think block and delivered `content=0,
finish=length`: 193 s of decode, nothing to show, and the client's only move
is a full retry. The trace made the shape unambiguous — the reasoning had
degenerated into a fragment loop within the first few hundred tokens and
never recovered (see §11 for the token-level half of this).

Cutting the *presentation* would not have been enough. A server-side
truncation can close the block on the wire, but the model keeps writing the
same thought, so the "content" that follows is the thought wearing a
different label. The mechanism therefore lives in the engine's decode loop
(`cwick_engine_set_think_budget`), where a token can be fed that was not
sampled: at the ceiling the engine writes the close tag itself, once, and it
lands in two places at the same time —

- **the output**, through the sink: the server's state machine sees
  `</think>` and reclassifies everything after it as content, in both the
  streaming (`cwick_stream_segment`) and buffered (`split_reasoning`) paths;
- **the context**, through the forward pass: the model has just written its
  own delimiter, so it starts the answer instead of continuing the thought.

The ceilings are per effort level (关=0 / 低=512 / 中=2048 / 高=8192 /
超高=16384 / 极致=32768 — one table since §22). 关 is 0, and 0 means "answer,
do not think": when
no block is open yet the engine writes the template's own empty form,
`<think></think>`, which is exactly how the checkpoint renders an assistant
turn that did not think — not a truncated thought.

Two details decide whether this is safe rather than clever. First, the
markers must be single tokens: the engine writes them as whole steps, and
splitting one would corrupt the block it is closing. The server probes the
tokenizer once per load (`srv_probe_think_ids`) and disables the mechanism
if either marker is not exactly one id, rather than guessing a byte split.
Second, the engine only closes a block that is actually open: it tracks
`think_open` (either seeded by the template — GLM's prompt ends with
`<think>`, which the server detects by comparing the last prompt id — or
written by the model), and a
turn that never opened a block, or that closed it itself, is never touched.
The budget can only ever change a thought that was going to run long.

`CWICK_THINK_BUDGET=<n>` overrides the level (0 = never think, negative =
off) both for the server and for `cwick run`, which is what makes the
mechanism measurable on a real checkpoint without a second copy of the
weights. The access log gains `think_cap=<n>` and marks it `(hit)` when the
closure fired, so a capped turn is distinguishable from one that finished
on its own.

Verification: `cwick_think_plan` is a pure function and is tested for the
whole matrix (below / at / above the ceiling, closed already, never opened,
budget 0 with and without an open block); a real decode on the synthetic
model asserts the injected pair is the first thing written and that
`think_budget_hit` follows, that an unreached ceiling injects nothing, and
that an out-of-vocabulary marker disables the mechanism instead of writing
a different token. `make test` passes.

The level table alone would not have saved the request that started this. It
asked for 高 (8192) with `max_tokens` 8096, so a ceiling of 8192 could never
be reached before the length limit — a cap that cannot fire is not a cap, and
the turn it exists to save is exactly the one that dies at the limit. The
engine therefore clamps any ceiling at or above `max_new` down to
`max_new - CWICK_THINK_RESERVE` (512, or half a smaller budget) and logs the
clamp next to the request. What is left is the honest version of "think, then
answer": the thought is cut where the answer still has room to exist.

## 13. A chunked MoE reads its experts in arena-width batches

`ds_moe_chunk` (the `n > 1` path, i.e. prefill) walks the union of the chunk's
routes and reads each distinct expert once. That part is right — the union is
what makes a long prompt affordable. What was wrong is *how* it read them: one
`ds_expert_fetch` per expert, each a blocking pread, so the whole union went
over the wire one 6 MiB request at a time.

That is the worst possible shape for this device. A single 6 MiB read sustains
about **2.5 GB/s** here, and the limit is latency, not bandwidth: the same SSD
delivers ~6.7 GB/s with one request at a time in a tight loop and ~9 GB/s with
four to eight outstanding. Measured on a 17-token prompt, expert fetch was
**68% of a 12.7 s prefill** and the bucket sum equalled the wall clock to
within 0.2 s — nothing was overlapping with anything.

The fix is the sequence the decode path already used, applied to the chunk's
union: `ds_expert_prefetch` submits, `ds_expert_take` consumes,
`ds_expert_apply` computes, `ds_expert_pin` admits. Batches are one prefetch
arena zone wide — `n_activated_experts` (6) experts, 18 matrices, which is
exactly what the arena was sized for — so a batch's reads are all in flight
before the first multiply-add starts and the rest complete underneath it.

| batch width | prefill, 17-token prompt |
|---|---|
| 1 (= the old order) | 11.75 s |
| 2 | 9.09 s |
| 4 | 8.13 s |
| 6 (default) | **8.00 s** |

Saturation at 4–6 is the arena zone width, which is why the batch is clamped to
`n_activated_experts` rather than made a free knob: a wider batch does not fit
and `cwick_stream_prefetch_submit` would silently fall back to demand I/O,
which is the behaviour this change exists to remove.

**The output is bit-identical, not merely plausible.** Batch width changes when
a read happens and nothing else: every (position, slot) is still one GEMV into
its own contribution, still rounded to bf16, still summed in slot order
afterwards. `CWICK_DS_PF_BATCH` exists so that claim is testable rather than
asserted — builds at width 1, 3 and 6 emit the same md5 on the same prompt and
seed. `make test` passes.

One side effect worth knowing about before comparing hit rates across versions:
the `cache=` field in the access log changed denominator. It used to be
`hits / (hits + misses that passed the ghost gate)`, because
`cwick_stream_expert_put_read` returns early — without counting anything — when
the ghost filter refuses a cold key. Routing through
`cwick_stream_expert_cached` counts every query, so one and the same request
now reads `8.5% (494/5299)` where the old path printed `68.6% (494/226)`. The
numerator is unchanged; the denominator is finally the number of questions
asked.

What is left in prefill is the exposed tail between batches (28% of the time).
Removing it needs the next batch's reads in flight while the current one
computes, and `cwick_stream_prefetch_begin` cannot express that: it drains the
zone it is leaving and discards that zone's unclaimed jobs, so a rolling window
would drop work that has already been paid for. A cross-batch pipeline means
owning job lifetime across the flip, which is a change to the threading
contract rather than to a loop. Not done.

## 14. A Metal batch must ask for concurrent dispatch

`cwick_metal_batch_*` collects independent GEMVs into one command buffer on
purpose: a single 8 MiB expert matrix costs ~300-700 us of host round trip
against ~105 us of GPU work, so one commit per matrix is launch-bound no
matter how fast the kernel is. What the batch did not ask for was permission
to *overlap* them. Metal serialises the compute dispatches inside one command
buffer by default — dispatch *i+1* does not begin until *i* retires — so the
batch removed the round trips and then ran the dispatches one after another,
each waiting for its own 8 MiB to arrive. The GPU measured 45 GB/s on a
transport that does ~227.

The fix is one API: `-computeCommandEncoderWithDispatchType:` with
`MTLDispatchTypeConcurrent`. It is safe here for a structural reason rather
than a hopeful one: every dispatch in a wave writes its own output slab and
reads only shared, read-only inputs (`x` for the routed group, the resident
weights for the shared expert), and the only real dependency in the layer —
`down` needing `silu(gate) * up` — sits between the two waves, i.e. between
two command buffers, where ordering is still guaranteed. Measured: the wave
went 2.07 ms -> 1.27 ms per layer, `gemv` 170.6 -> 126.4 ms/step, 727.8 ->
663.0 ms/step, and the generated text is byte-identical because dispatch
order was never part of the arithmetic.

Two things this does *not* fix, and they are worth writing down so the next
person does not read 118 GB/s as "the memory is saturated":

- the fp8 kernel still decodes every weight byte through a `threadgroup`
  LUT indexed by that byte, so 32 lanes gather 32 unrelated entries and pay
  bank conflicts. Its diagnostic twin `v3d` — same loads, same accumulator
  shape, `int`->`float` instead of a table read — runs the same wave in
  0.67 ms, i.e. **227 GB/s**, which invites the conclusion that the table
  costs 1.9x and dropping it is free. `v4` tests exactly that and **loses**:
  a bit-exact E4M3 decode is ~13 integer ops, and the gather plus its
  conflicts is cheaper (1.396 -> 1.657 ms on the gate/up wave, 0.799 -> 0.851
  ms on down, which v4 also extended to k=2048). So 227 GB/s is the price of
  the loads with *no* decode at all — an upper bound nothing correct can
  reach — and ~110 GB/s is where a correct fp8 GEMV lands on this GPU. The
  experiment is kept compiled behind `CWICK_FP8_KERNEL=v4`; numbers in
  `docs/PERF.md`.
- `MTLDispatchTypeConcurrent` needs macOS 12. The helper falls back to the
  serial encoder rather than failing, because a refused batch sends the
  caller down the CPU path for the whole layer, which costs far more than a
  slow GPU.

## 15. A one-matrix GPU split does not pay for itself

The DeepSeek decode path runs entirely on the CPU pool, so `--backend metal`
only buys it an arena for the expert blobs. Splitting each GEMV by output rows
between the pool and the GPU is *arithmetically* free — every row's chain is
fixed by the CPU kernel and a row is written by exactly one engine, which the
new `CWICK_DS_GPU_VERIFY=1` self-check (recompute the GPU's rows on the CPU and
compare) confirms: fp4 never reports a difference, bf16 differs only in the
last bit on ~1% of rows at some shapes. It is not *temporally* free, and that
is what decides it.

A submit+wait costs ~0.2 ms regardless of matrix size, and a decode step issues
~1400 GEMVs. Measured at the production entry point on the real dense shapes
(`tools/ds_gemvbench --gemv shapes`): the GPU alone gets ~80 GB/s per 84 MB
matrix while the pool gets 100-270 GB/s on the same bytes. The GPU only reaches
198 GB/s when a dozen matrices share one command buffer, and this interface
submits one matrix per buffer. End to end the split made decode 763 ms/step
against 461 all-CPU; a 16 MB size gate kept the profitable shapes and brought
it to 518, still a loss.

So `CWICK_DS_GPU` defaults to 0 and the kernels stay. What would change the
answer is a submission that carries a wave of independent matrices — the eight
block-diagonal `wo_a` groups and `shared w1/w3` are naturally that — with the
wait deferred to the dependency point. Until then, the honest summary is that
this machine's DeepSeek decode is a 17.9 GB/step streaming problem the CPU
already runs at 78 GB/s on the dense half, and the GPU can only add bandwidth
if it is fed a batch, not a matrix.

## 16. Two cheaper answers than the GPU batch, and why both are closed

Both were tried so the next attempt does not repeat them.

**Removing the expert pin copy is worth nothing.** Every miss is read into the
prefetch arena and then copied into its cache slot, which measured 7.8% of a
decode step (35 ms) once the timer was split out of `expert fetch`. Reading
straight into a reserved cache slot is expressible — `cwick_stream_expert_reserve`
hands back the slot and `prefetch_submit_into` can land the pread there — and it
was implemented end to end with byte-identical output. The step went 453.6 ->
452.1 ms, i.e. nowhere: the copy was already overlapping the SSD wait for the
other matrices of the same expert. The remaining 18 ms reappeared inside
`fetch` and `matmul`. Reverted, because a more complex read path that also
breaks the hit counter's meaning (a reserved slot is indexed before its bytes
arrive, so the cache query in `take` has to move) is not worth zero.

**The expert hit rate has no headroom.** 48.7% looks bad against the 81.9 GiB
cap until you ask how full the cap is: `expert pin set: 79.43 GiB used of 81.86
GiB cap`, 97%. The budget is saturated, 271 GB of expert weights do not fit in
82 GiB, and uniform routing would give 30% — so 48.7% is the concentration
routing already provides. Raising it needs RAM the machine does not have or a
lower-precision expert pack, which is out of bounds.

What is left is the one thing not yet disproved: submitting a wave of
independent matrices per command buffer so the GPU adds bandwidth *while* the
pool streams, rather than replacing it matrix by matrix.

## 17. The machine reads memory at 123 GB/s, and that is what decides 5 tok/s

Everything in section 16 was a search for a cheap win. This section is the
measurement that says there isn't one, plus the one promising idea that the
checkpoint itself rules out.

**Peak read bandwidth is ~123.5 GB/s and it does not move.** A bare 12-thread
streaming 64-bit sum over a cold buffer gives 123.5 GB/s at 4 threads, 123.2 at
12, 122.3 at 32; 1 GiB, 4 GiB and 16 GiB are identical; widening to eight
independent accumulators and 32 loads per turn changes nothing. Four threads
saturating it is the signature of a DRAM controller limit, not a bad loop.

That reframes the whole profile. A decode step moves 17.9 GB, so even a pure
streaming read with no arithmetic costs 145 ms, and 5 tok/s (200 ms) needs
89.5 GB/s sustained -- 73% of the ceiling -- while carrying the SSD stream at
the same time. Today's 453 ms is 39.5 GB/s, 32%. And the dense half is *already*
at 80% of the ceiling (the `wo` bucket runs 99 GB/s, `--gemv shapes` shows
125-127 GB/s on wo_b and the indexer), so no amount of kernel work helps dense:
only fewer bytes could.

**The one way to halve dense bytes is dead, and the checkpoint kills it.** Dense
weights are FP8 e4m3 in the file and are widened to bf16 at load
(`ds_model.dense16`, 11 GiB). Reading them natively would halve dense traffic
(13.7 -> 6.85 GB/step), free 5.5 GiB of residency for the expert cache, and
needs no new plumbing because `ds_gemv_q` already dispatches to `ds_fp8_gemv`
when `w16` is NULL. The fast NEON decode is three bit ops --
`((b & 0x7F) << 7) + 0x2000 | ((b & 0x80) << 8)`, verified by hand on 1.0, 0.5,
-1.0 and 448 -- and it is exact for every normal value. It is wrong for two
corners: e==0 is subnormal (m * 2^-9, not 2^-7 * (1 + m/8); at m=4 that is
0.0078 against 0.0117, a 50% error) and e==15,m==7 is NaN. A scan added to
`bind_q` found **all 330 fp8 tensors contain at least one such byte**, so the
corners are not rare. Fixing them needs two `vqtbl1q_u8` gathers, a zip, a sign
merge and a bsl -- about 15 extra instructions per 16 values, taking the decode
from 6 to 29 per 16 values where bf16 needs 6 (it is two shifts). That is ~1.5
instructions per byte against bf16's 0.48, which flips fp8 from
bandwidth-bound to issue-bound and erases the gain from halving the bytes.
Abandoned; the scan was removed with it.

**And the rest, measured, are zero.** Unrolling the fp4 GEMV to two blocks with
a prefetch: gate/up 65.5 -> 75.6 GB/s but down 79.1 -> 53.5, and end to end 493
ms against 453. Reader threads 2/6/12: expert matmul 26.3/26.5/26.6%.
`CWICK_EXPERT_CACHE_GIB=95` against the automatic 82: 2.067 vs 2.055 tok/s,
hit rate 33.2% vs 32.5%, residency 95.78 GiB -- the machine is already full,
which is why the hit rate has no headroom either.

So the honest number for this machine is ~2.1 tok/s today and maybe ~3.3 tok/s
if the one remaining real gap is closed: expert matmul runs at 36 GB/s in situ
against the same kernel's 65 GB/s cold bench, worth about 55 ms. Nothing found
here gets to 200 ms/step, because 200 ms/step asks for 73% of a bandwidth
ceiling that the dense half already spends 80% of.

## 18. A prefix cache that keys on exact lengths misses every multi-turn chat

The prefix cache compared a prompt against stored snapshots by *exact length*:
`cwick_pfx_lookup` samples the rolling hash at every position and looks for an
entry whose `pos` equals that position (`pfx.c:383`). That is correct and cheap,
and it is why a 0% hit rate took a live audit to find rather than showing up as
an error.

The lengths that exist are the prompt itself, every `pfx_block` positions during
a chunked prefill, and prompt+generated at the end. The length a continuing
conversation needs is none of those. A request that opens a thinking block ends
its prompt with the think-open marker, which the server appends; the next turn
of the same session renders that assistant turn as the think-close marker plus
its content. The two prompts diverge at exactly that last token, so the prefix
they share is one token shorter than the prompt just captured. Turn N was 262
tokens, turn N+1 was 405, and they agreed on the first 261; the cache held 262
and 551. The same arithmetic failed every other transition in the session, and
with `CWICK_PFX_MIN_TOKENS` at 256 the opening turns were not written at all
while the block checkpoints needed an 8192-token prompt to fire. Four requests,
four queries, zero hits, 260 s of prefill re-paid.

So `pfx_save_now` now also stores a twin at `nids - 1`, keyed on the same tokens
minus the last. This is not a new mechanism: the whole-prompt hit path already
rolls a snapshot back by one position on load (`server.c`'s `pfx_cap.pos--`), so
capturing one is the symmetric half of a rollback the code already trusted. It
costs one extra `cwick_engine_capture_prefix`, which takes any position
`<= eng->pos`, i.e. a state pack rather than a prefill, and the writer queue is
ordered so a snapshot that ever has to be dropped is the twin rather than the
full-length one. `CWICK_PFX_MIN_TOKENS` drops to 32 — on this checkpoint a
33-token turn costs 19 s of prefill because the expert-union read dominates, so
the old "short prompts are not worth a file" reasoning does not hold.

Measured on three turns with byte-identical requests in both arms: hit rate
2/3, 73 of 125 prompt tokens reused, prefill 53.6 s -> 27.3 s, and the three
responses identical in `content`, `reasoning_content` and `finish_reason`. The
remaining miss is the first turn, which has nothing to replay.

The generalisable part is not about thinking markers. It is that a cache keyed
on exact lengths can only serve the transitions whose length it happens to have
captured, so the capture policy has to be derived from *how the next request's
prompt is built*, not from the current one. Here the next prompt is the current
one with its tail rewritten, which makes "one shorter" the right policy; a
client that re-tokenised the reply would need a different one, and would miss
loudly rather than silently, which is the property worth keeping.

## 19. Per-request engine state has to be pushed to every backend that decodes

The thinking budget (L4) was implemented in `engine.c`'s decode loop and armed
by `server.c` for every chat request, with a log line that said so. DeepSeek
does not decode through that loop -- `ds_engine.c` has its own -- so the ceiling
was configured, reported, and never read. A live request generated 16,568
thinking tokens against a configured 8192 and ran for 2 h 35 m with ~30 h to go.
The loop guard could not catch it: `finished_reason=loop` stays 0 when the model
is not repeating, and this one was genuinely still thinking.

The pattern was already in the codebase and simply not followed for this field.
`ctx.stop`, `ctx.tag`, `ctx.spec` and `decode_phase` are all pushed into
`eng->ds->ctx` by the owning engine precisely because the DeepSeek backend runs
a separate loop. `think_budget` was the fifth such field and the only one left
on the GLM side, which is what made the omission invisible: the arming code is
shared, the log line is shared, and only the loop that enforces it is not.

So the rule this establishes: **a per-request engine setting is not armed until
every backend that can decode has a copy of it**, and the thing to check is not
"is it set on the engine" but "does the loop that reads logits read it". A log
line that reports configuration is not evidence of enforcement -- here it was
actively misleading, because `think_cap=` printed 0 for every DeepSeek request
(`res.think_budget_applied` was never set on that path), which reads as "no
thinking requested" rather than "the ceiling is not being applied".

Two details worth keeping from the implementation:

- The close tag has to be written *and* forwarded in the same step. The sink
  must see it so the wire carries a closed block and the server's segment
  machine reclassifies the rest as content, and the forward pass must consume
  it so the model -- having just written its own delimiter -- answers instead
  of continuing the thought. Only the decode loop can feed a token it did not
  sample, which is why this cannot live in the server.
- `eng->ds` is allocated with `cwick_xmalloc` and no memset, so the disabled
  state must be an explicit -1 in `ds_engine_open`. A zeroed struct reads as
  budget 0, i.e. "close the think block before the first token" -- the exact
  trap `cwick_engine_new`'s own comment warns about, and the first cut of this
  fix fell into it. The armed case looked correct; only the control run, with
  the environment variable unset, showed the mechanism firing when it should
  have been silent. Run the control.

## 20. --max-output-tokens removed: a clamp that could only ever miss

It existed to bound a request that omitted `max_tokens`, which would otherwise
be handed the whole remaining context (99,501 tokens on a 100K window, about
21 h at 1.3 tok/s, always ending as a client-side abort). That part worked.

It could not help the case it was added for. The clamp was gated on
`!gr->max_tokens_set`, so it applied only to requests that omitted the field --
and every real client sets it. The built-in web UI sends `max_tokens: 100000`
by default, which is why the live session that motivated the whole audit
(`start.sh` had `--max-output-tokens 10000`) ran a single turn for 2 h 35 m with
~30 h to go: the clamp never saw it. Measured against the three failure modes it
was supposed to prevent, it prevented none of them.

Worse, it was actively misleading in two places that report configuration.
`/v1/config` published `max_output_tokens`, and the settings panel rendered it
as a row, so an operator reading either would conclude requests were bounded at
10000 when the only requests actually bounded were the ones nobody sends. A
limit that is reported but not enforced is worse than no limit, because it
stops anyone from looking for the real one.

Removed: the option, the `cwick_server_options` field, the clamp, the
`/v1/config` key, and the settings-panel row. `--max-tokens` keeps its meaning
(the default for a request that omits the field), and the engine's own
context-ceiling clamp still bounds an uncapped request, with a log line that
prices it in hours at the measured tok/s. Bounding a turn is the client's
`max_tokens`, which is the OpenAI reading of the field anyway.

The generalisable part: before adding a server-side safety clamp, check which
requests it can actually reach. A clamp gated on "the client did not ask for
this" is a clamp for clients that do not exist.

## 21. --max-prompt-tokens removed: the one knob that only ever cost something

It refused a prompt longer than N with a 400, on the reasoning that a 30K-token
prefill owns the single generation slot for minutes and the client sees nothing
until it finishes. The default was 0 (no limit), so it only ever mattered to
whoever typed the flag -- and `start.sh` typed it as 100000.

That number is where it stops making sense. At `--context 1000000` the window
itself admits a million-token prompt, so a 100000 gate refuses nothing the
engine would not already refuse; at the old `--context 131072` it refused only
prompts between 100000 and 131072, a band that is simultaneously too long to
prefill in reasonable time and too short to be worth a separate code path. It
was a policy knob bolted onto a capacity limit, and it could only ever remove
capability: a request it refused is a request that gets a 400 instead of the
engine's own error, which names the exact window and says how to raise it.

Removed: the option, the `cwick_server_options` field, the refusal block, the
`/v1/config` key, the settings-panel row, the usage text and the README
paragraph. What bounds a prompt now is `--context` and nothing else, enforced
inside the engine (`prompt 400 + max_new 4 at position 0 exceeds the
64-position context window`), reported as a generation error for a
non-streaming request and an in-band SSE error for a streaming one.

That leaves the server with exactly one length-related knob, `--context`, which
is the honest shape: capacity is a property of the machine, and a policy limit
that disagrees with it is a second source of truth waiting to be wrong. If a
long prefill ever needs refusing, the right place is in front of the server --
a proxy or the client, which knows whether *this* prompt is worth waiting for.

## 22. The two halves of a thinking level are one table row

A request names an intensity (`低`/`中`/`高`/`超高`/`极致`/`关`, plus the English
aliases). That single name was being consumed by three separate lookups, each
with its own copy of the level strings:

- `cwick_thinking_effort_resolve` → the level *and* the GLM header (`Low`/`High`/`Max`)
- `think_budget_for_level` → the token ceiling (`0`/`512`/…/`32768`)
- `ds_effort_from_level` → DeepSeek's `1..100` header *and* whether to think at all

Three if-chains keyed on the same six strings is three chances for a typo to
make one of them silently fall through to its default. It would not crash: a
missed `strcmp` in `think_budget_for_level` returns -1, which means "no ceiling"
— the exact failure the budget exists to prevent, and one that shows up as a
long turn rather than an error. The GLM/DeepSeek asymmetry had the same
exposure: it lived in a comment above the resolver, not next to the numbers.

They are now one array of rows, `THINK_LEVELS`, with a column per consumer:

```
level   glm     ds   budget  thinks
关      Low      0        0  false
低      Low     25      512  true
中      Low     50     2048  true
高      High    75     8192  true
超高    Max     90    16384  true
极致    Max    100    32768  true
```

Two things fall out of writing it this way. First, the asymmetry is now visible
instead of documented: `低` and `中` share `Low` because the GLM checkpoint was
trained with exactly three headers, and the table shows that they nevertheless
differ by 4x in budget. Second, a level cannot gain a budget without gaining the
header that asks the model to use it — the two cannot be edited apart, which is
what they are: the intensity is what the model is *told*, the ceiling is what
the engine *enforces*.

`CWICK_THINK_BUDGET` remains the deliberate exception and only touches the
budget column. That asymmetry is the point of the knob — hold the model's
instructions fixed and vary the ceiling — and it is how the DeepSeek
enforcement bug in §19 was found: `CWICK_THINK_BUDGET=64` still sent the `高`
header (effort 75) and still got 64 tokens to honour it in.

The table is printed at startup so the mapping is checkable from a log rather
than from the source:

```
thinking levels (header / token ceiling): 关→Low/0  低→Low/512  中→Low/2048
  高→High/8192  超高→Max/16384  极致→Max/32768
```

Verified on the real checkpoint that the behaviour is unchanged: `高` still
renders `Reasoning Effort: 75` and arms 8192, `关` still arms 0 and fires before
the first token (`think_cap=0(hit)`, and its prompt is 5 tokens because
`thinks=false` skips the header entirely). `make test` covers every spelling
against the GLM path (`test_cwick.c`, "http chat accepts every
thinking-intensity level").

## 24. A model switch has to give the pages back, not just the objects

A live `serve` session switched DeepSeek -> GLM-5.3-Flash and then died. The
log is unambiguous: the switch reported success in 21.8 s, the incoming model
warmed up at **67.19 GiB** resident, its first decode step already sat over the
102.40 GiB ceiling, the ceiling check then did the only thing it can — it gave
back the expert cache, 32.00 -> 0.00 GiB over seven steps — the footprint did
not move while doing so, and from there it grew **7.9 GiB per step** (exactly
the 7.88 GiB of experts each step streams) until the process was killed at
185.93 GiB. Two measurements separate the cause from the symptoms:

- The same GLM started fresh warms up at **14.49 GiB**, and decodes nine steps
  at a flat **71.51 GiB** with its 32 GiB expert cache intact. So neither the
  plan nor the decode path is wrong on its own.
- The difference, 52.7 GiB, is almost exactly what DeepSeek held when the
  switch began (51.70 in the reproduction, 53.71 in the failing session).

So the switch freed the engine objects and left the pages. Three separate
reasons, all of them "unload is not the same as exit":

1. **`ds_model_close` never freed `dense16`.** The bf16 widening of the dense
   weights is an arena block sized by `ds_dense_sizes` (~11 GiB here) and it
   had no release call at all, so every load left it behind.
2. **Arena buffers were never released.** `cwick_arena` keeps its backing
   `id<MTLBuffer>` in a malloc'd struct. ARC retains on assignment, but
   `free(g_arenas)` cannot release through it, so the wired arena — the whole
   expert-cache high-water mark — stayed resident for the rest of the process.
   The same was true of the no-copy views, which are worse: they wrap the
   model's *mapped* weights, so holding them keeps the outgoing mapping alive
   even after the spine is closed.
3. **The memory plan is static.** It is computed from the model's own sizes, so
   the incoming model books the same `spine 15.46 + runtime 22.50 + arenas
   0.03 + headroom 0.38 + cache 32.00` whether it is alone in the process or
   standing on 50 GiB of its predecessor. The over-commit is therefore not
   detected at plan time, only by the runtime ceiling check — which reacts by
   destroying the one component that was working.

The fix is aimed at (1) and (2) and measured on the same switch:

| | before | after |
|---|---|---|
| incoming model, warm-up | 67.19 GiB | **43.74 GiB** |
| expert cache during decode | 32.00 -> 0.00 GiB | **32.00 GiB, held** |
| footprint over 9 decode steps | 107.41 -> 185.93 GiB | **89.15 GiB, flat** |
| process | killed | serving |

`ds_model_close` frees `dense16`; `cwick_metal_arena_release()` is a new entry
point that drops every arena block and every no-copy view, releasing the
buffers through ARC, while leaving the device, queue and pipeline states alive
— a switch does not need a shader recompile, only the outgoing model's
memory. `cwick_metal_bridge_cleanup` now goes through the same helper, so the
shutdown path no longer leaks the buffers either. The switch path calls it
between `cwick_engine_free` and the incoming `cwick_engine_new`, before the new
model plans its budget.

What was left after that was another 29 GiB, and it grew with each switch (a
second round trip peaked at 70.27 GiB). Both `cwick_safetensors_close` and
`cwick_spine_close` documented their mappings as **process-lifetime** and never
unmapped them — a policy that was correct while a process served exactly one
model. Both now record each mapping's length and drop it on close, because the
obvious alternative does not work: **`madvise(MADV_DONTNEED)` is a no-op for
file mappings on macOS.** Measured, it returns success for all 48 DeepSeek
shards (475 GiB of address space) and RSS does not move by a byte, so it cannot
be used to make a claim about released memory. `munmap` gives back 9.20 GiB on
the DeepSeek side and 14.44 GiB on the GLM side. That forces one ordering
constraint: the no-copy Metal views are buffers over the mapped weights, so
`cwick_engine_free` releases the views (`cwick_metal_views_release`) *before*
`cwick_spine_close` unmaps, and the arenas are released after the stream has
returned its blobs to them (releasing an arena first would make
`cwick_shmem_free` call `free()` on an arena pointer).

Measured on the switch, in three stages:

| | incoming model, warm-up |
|---|---|
| original | 67.19 GiB |
| `dense16` + arena/views released | 43.74 GiB |
| + checkpoint unmapped on unload | **34.57 GiB** |

and the accumulation is what actually mattered: three switches used to end at
70.27 GiB warm-up (+26.5 per round trip) and now end at 37.50 (+2.94). What
remains is the malloc zones: `free()` hands pages back to the allocator, not to
the VM, and `malloc_zone_pressure_relief` does not return them either — the
switch path calls it anyway, and the residual 20 GiB is the price of a process
that once held 60 GiB of scratch. It stays under the ceiling with the expert
cache intact, which is the property that was broken.

The generalisable part: an unload path that only frees the bookkeeping looks
identical to a correct one until a second model wants the memory. Every
allocator here that outlives an engine — the arena, the views, the maps — has
to be asked "who gives this back when the model changes", and the answer has to
be code, not a comment saying the maps live for the process. And when the
answer involves `madvise`, check that RSS actually moved before believing it.

## 25. The pin budget has to pay for the whole footprint, and both loops steer

### What was wrong

The DeepSeek expert pin cache was budgeted as `80% of RAM − dense − 5 GiB`
(`ds_model.c`), i.e. 81.9 GiB on this 128 GB box. The 5 GiB was meant to cover
"the KV caches, the contribution tables and the allocator". Measured, the
non-cache part of the footprint is ~44 GiB:

| part | bytes |
| --- | --- |
| expert pin cache | 81.86 GiB |
| dense (fp8 arena + bf16 copy) | 15.5 GiB |
| compressed KV `cmp_kv` + indexer `idx_k` | 9.5 GiB at `--context 1000000` |
| Metal arenas / IOAccelerator / driver | ~18 GiB |
| scratch, prefix cache, misc | ~1 GiB |

so the steady state was **126.11 GiB against a 102.40 GiB ceiling**, 23.7 GiB
over. The pin cache is ordinary `malloc` memory (the arena path falls back to
`cwick_xmalloc`), so the overshoot went straight to the VM: in 14 minutes swap
went 2.56 → 17.4 GiB (macOS grew its own swapfiles) and the compressor did
+38.0 M compressions / +30.8 M decompressions — about 1.1 TB of VM traffic for
a workload that read 158 GiB off the SSD. The signature was unmistakable: same
cache-hit rate (75.3%), *fewer* bytes streamed per step, and a step time that
wandered between 330 and 3303 ms. The hits were decompression faults. DeepSeek
fresh was 1.887 / 2.220 tok/s; one switch round trip later the same model ran
at 0.622 tok/s.

Three things made it structural, not incidental — all three "GLM has it,
DeepSeek does not":

1. the budget never looked at `cwick_process_footprint_bytes()`;
2. `maybe_steer_memory` returned at `if (!eng->spine.stream)` — GLM only — and
   `cwick_stream_trim` had exactly one caller in the tree;
3. `cwick_mem_lock_range` runs only on the GLM spine and GLM KV, so the startup
   line claiming "no-swap mode" did not describe the DeepSeek cache at all.

### What changed

**A — the budget pays for the footprint.** `ds_expert_cache_budget()` now takes
the config and subtracts what is actually there: `reserve = 4 GiB + KV bytes
+ host/8 (GPU arenas and driver)`. The KV term is `ds_kv_bytes()`, the same
arithmetic `ds_model_open` allocates by, so it tracks `--context` instead of
assuming 5 GiB. At `--context 500000` this plans **62.1 GiB** (was 81.9); at
`--context 1000000` it plans ~57 GiB. `CWICK_EXPERT_CACHE_GIB` still overrides.

**B — one valve, called from both loops.** The body of `maybe_steer_memory`
moved to `cwick_stream_steer_memory(stream, ceiling, step, tag)` in `stream.c`.
The GLM loop calls it as before; the DeepSeek decode loop calls it every step
next to `cwick_stream_expert_decay`, and the prefill chunk loop calls it too
(that phase fills the cache for minutes at a time).

### The trap in B: do not trim on every step

The first version inherited the GLM body verbatim, including
`cwick_stream_limit()` — which *permanently* lowers the cap. `free()` does not
shrink `phys_footprint` in the same instant, so every step measured the same
overshoot and cut the cap again. With a deliberately oversized 90 GiB cache the
cap reached **0 in about two seconds**, every line reporting "released 0.00
GiB" against a footprint that had not moved, and the decode lost its whole
working set. Fixed with:

- a **12 s cooldown** after a trim (`CWICK_STEER_COOLDOWN_MS`), so the next
  decision reads a footprint that has had time to account for the release;
- **trimming `over + 1 GiB`** so the next refill does not re-trip immediately;
- a **floor** of `max(cap/4, 256 MiB)` so a bad day cannot hand back the cache.

With the same 90 GiB cap it now logs one trim: 90 → 88.61 GiB for a 0.39 GiB
overshoot.

### Measured

Same box, same prompt, `--context 500000`:

| build | steady footprint | decode |
| --- | --- | --- |
| before (81.9 GiB plan, no steering) | 94–100 GiB and still climbing to 126 | 1.421 / 1.436 tok/s |
| after (62.1 GiB plan + steering) | **84.76 GiB, flat** across requests | **2.146 → 2.781 tok/s** |
| after, with a deliberately oversized 90 GiB cache | 102.79 GiB, one trim to 88.61 GiB | 2.328 tok/s |

Swap stayed flat at ~3.2 of 4.1 GiB throughout the runs after the fix, against
17.4 of 18.0 GiB before.

### Still open

- Do **not** wire the DeepSeek pin cache. Wired pages measured slower here (the
  note at `metal.c`), and the point is to give bytes back, not to hide them.
- `/v1/metrics` has no swap or compressor gauge at all, which is why this took
  `vm_stat` sampling to find. `vllm:num_requests_swapped` is request
  preemption and has nothing to do with OS swap.

## 26. A third backend: what MiniMax-M3.1-Flash needed that the other two did not

The port is `src/mm_model.c` (weights + forward), `src/mm_engine.c` (generation,
a copy of the DeepSeek contract) and `src/mm_prompt.c` (the checkpoint's
`chat_template.jinja`). The engine-level integration is deliberately boring:
`cwick_engine` grew one more pointer next to `ds`, and every site that used to
ask "is this the DeepSeek backend" now asks the same question about either.
Boring is the requirement — the server, the prefix cache and `/v1/metrics` must
not be able to tell the three apart.

What was *not* boring, and why it is written down:

**The reasoning markers are per checkpoint.** `srv_probe_think_ids` used to
probe a fixed `<think>`/`</think>` pair. MiniMax writes `<mm:think>` (ids
200059/200060) and its tokenizer *also* carries `<think>` (200050) as an
unrelated control token, so the fixed pair armed the wrong ids and the answer
splitter looked for a closing tag that never arrived: every reply came back with
an empty `content`. The marker pair is now chosen from the resident model type
and carried in `cwick_stream_state` (`open_mk`/`close_mk`), because the splitter
works on bytes and cannot look it up itself.

**Two memory-safety bugs, both found by the first 128-position prefill chunk.**
The scalar path never reached them because every earlier test ran nine tokens:

* the expert loop held the gate/down/up pointers of one expert simultaneously,
  and a cache miss in between evicts other slots — so the blob a later GEMV read
  was freed underneath it (SIGSEGV inside the NEON kernel);
* `gsm`/`usm` were sized to `inter_dim` (3072) but the three dense layers index
  them with `dense_inter_dim` (12288), i.e. 1.5M floats written past a 393K
  buffer, which surfaced as a crash two layers later in unrelated code.

The generalisable part: in a port like this, "it produces the right token on a
short prompt" says nothing about bounds. A 128-token chunk is the first input
that touches every buffer at its real size, and it should be in the smoke test
from the beginning.

**The FP4 GEMV was 2.6x slower than it needed to be, for one function call.**
`ds_e4m3_to_f32()` asks `build_luts()` whether the table is ready, then indexes
it. The kernel called it once per 16-element block — 384 times per row, ~1.2
million times per expert matrix. Replacing it with a locally built 256-entry
table moved the same kernel from 16.8 to 43.5 GB/s (30 → 77 GMAC/s), and a
vector accumulator instead of a per-block horizontal reduction took it the rest
of the way. For reference the bf16 GEMV on the same pool measures 136 GB/s,
which is this machine's memory ceiling — so the fp4 kernel is no longer the
outlier it was.

**The step is I/O, and the arithmetic caps the token rate.** One decode step
reads 19.4 GiB (dense 12.6 + routed experts 6.9 + shared 6.4 shared with the
resident part) and performs 24.3 G MAC. With the expert cache at 28% of the
216 GiB bank, ~1 GiB of that per step still comes from the SSD, and a
synchronous 10 MiB pread sustains only 1.3–2 GB/s here, so the phase nominally
called "experts" is more than half waiting. Reserving every missing blob on one
thread and spreading the preads over the pool (`cwick_stream_expert_reserve`,
which existed for exactly this) took the step from 1.2 s to ~1.0 s — less than
the comment in stream.c promises, because a layer misses about 0.7 matrices on
average, so there is nothing to spread.

The ceiling matters more than the tuning: max(19.4 GiB / 130 GB/s, 24.3 G MAC /
77 GMAC/s) ≈ 315 ms per token, i.e. **3.2 tok/s with a perfect cache**. Sequential
decoding is capped there on this machine; the checkpoint ships a
7-token block draft model (`dspark/`, 2.3 GB) for precisely this reason, and
amortising one weight read over several accepted tokens is the only path that
multiplies instead of shaving.

### Metal for the MiniMax backend: batched, measured, and off

The obvious wiring is five projections of a layer into one command buffer via
`cwick_metal_bf16_batch_add` — the API the GLM path already uses, so no new
shader was needed to try it. Measured on a decode step: **590 ms -> 1400 ms**,
with the projection phase alone going 142 ms -> 932 ms. That is ~15 ms of GPU
work per layer against 2.4 ms on the CPU pool, for a matrix that is 100 MB and
should move in ~0.3 ms at this machine's GPU bandwidth.

So the batching was never the problem — one commit-and-wait per matrix was a
symptom. A 8192x6144 bf16 GEMV taking 3 ms means the shader is leaving the GPU
idle (order of magnitude: 33 GB/s against a ~400 GB/s device). Before any of
this is worth revisiting, the kernel needs to be tiled — several output rows per
threadgroup, the activation staged in threadgroup memory, and a dispatch grid
that keeps 40 cores busy — and that has to be measured against the CPU pool on
its own before it is wired into a step.

The plumbing stays behind `CWICK_MM_METAL=1` as the harness for that work; the
default must not pay for an experiment.

### FP8 dense: the bf16 copy was buying nothing

The MiniMax checkpoint already stores every non-expert matrix as e4m3 with one
e8m0 scale per 32 K elements, per row. The port widened all 540 of them to bf16
at load time (19.67 GiB, 40 s) and then read 21.1 GB of them per token. The
widening was exact, but the bf16 GEMV was **bandwidth-bound at 136 GB/s** — the
machine's ceiling — so doubling the bytes doubled the time and bought nothing.

Now they are used as stored, decoded inside the GEMV (`mm_fp8_gemm`): the e4m3
byte becomes an f16 by bit shuffle (sign kept, exponent +8, mantissa <<7), f16
to f32 is one instruction, and the per-32 block scale multiplies in as a vector
constant before the block sum. Everything is exact except e4m3 subnormals, which
decode as 2^-7*(1+m/8) instead of 2^-6*(m/8) — a 2x error on weights at most
3.5e-5 of their block maximum, i.e. below the block scale's own resolution.

Measured on a decode step (same prompt, same cache): projections 142 -> 88 ms,
`wo` 52 -> 30, dense traffic 21.1 -> 10.6 GB, footprint 74 -> 60 GiB, load
40 -> 32 s, best step 590 -> 462 ms. Verdict on the next token is unchanged
(`the quick brown fox … lazy` -> ` dog`, 19.70 vs 18.84 before).

Two things this cost, both of which the memory plan now has to pay for:

* the FP8 weights are file-backed pages rather than an anonymous copy, so they
  can be evicted; the budget reserves `fp8_total` for them, which is the reason
  the expert cache did not simply grow by the 19.67 GiB that was freed
* the first cut decoded a row into a 24 KB scratch buffer and dotted it
  afterwards. That moves 2x the data and measured *worse* than bf16 (250 ms).
  The fused version — decode eight values into two f32 vectors and FMA them
  straight into the block accumulator — is the one that wins.

### Long-prompt prefill: what the numbers say, and one dead end

Prefill of a 302-token prompt: **199.2 s** (1.5 tok/s) after the FP8 dense change,
against 205.6 s before it. Three things were measured on the way there, and two
of them are worth keeping:

* **The prompt is not I/O-bound at the disk.** It streams 125 GiB, and `--io`
  shows the device sustains 9.35 GB/s for 10 MiB reads. 302 tokens cost 7.5 T
  MAC in total, i.e. the phase runs at ~37 GMAC/s — half what the decode step
  achieves. Prefill is compute-bound, which is why neither a bigger cache nor a
  faster reader moves it.
* **The prefix cache is the real lever for long prompts**, and it was broken
  until the `mm_state_apply` fix: a 400-token turn now replays 262 tokens (66%)
  from a snapshot instead of re-prefilling them, which is exactly需求 12's
  intent (前缀缓存 / 磁盘 KV checkpoint). Everything after the first turn of a
  conversation is cheap; the first one is not, and cannot be made cheap without
  reading the expert bank.

The dead end: **grouping an expert's rows into one GEMM**. Prefill routes a whole
chunk before walking the expert union, so an expert is typically asked for ~4
positions at once, and the per-row GEMV read its 10.6 MB matrix once per
position. Reading it once for the group is obviously right — and it measured
**212.4 s against 199.2 s**, because the grouped kernel as written accumulates
with a horizontal add per 16-element block per activation, the exact pattern that
made the FP4 GEMV 2.6x slow before. To pay off it needs what the decode kernel
has: one vector accumulator per activation that stays in registers across the
whole row. That is the next step here, and it is worth ~4x less expert traffic
per chunk at 128 positions (16x at 512).

It sits behind `MM_PREFILL_GEMM=1`, default off, with `--cvt` in `tools/mm_run`
and `MM_CHUNK_CAP` as the harness for that work: both kernels agree to 1e-6 on
synthetic data and inside the model, so what is missing is throughput, not
arithmetic.

### Long-prompt prefill, part two: 199 s -> 104 s, and where the remaining time is

Four changes, each measured on the same 302-token prompt (`tools/mm_run`, top-2,
`--context 8192`). The prompt is 302 tokens, which matters: chunking decides how
many times the expert bank is walked, and that turned out to be the largest term.

| | prefill | tok/s | streamed |
|---|---|---|---|
| before this round | 199.2 s | 1.5 | 125 GiB |
| + attention vectorised | 136.8 s | 2.2 | — |
| + `MM_CHUNK` 128 → 256 | 127.7 s | 2.4 | — |
| + `MM_CHUNK` → 512 | **104.2 s** | **2.9** | 111 GiB |

**The bottleneck was attention, not experts.** Phase counters (now printed for
prefill too, per token) showed `attn` at 91 / 273 / 398 ms per token across the
three chunks — growing with position and, in the last chunk, the single largest
item. The inner loops were scalar with a per-element `cwick_bf16_to_f32` call:
one multiply per iteration, no ILP, and 8 query heads re-reading the same KV head
they share under GQA. bf16 is the upper half of an f32, so widening is one shift
(`vshll_n_u16(h, 16)` → reinterpret); with four accumulators the dot product went
from 398 to **30 ms** per token at that position, a 13x on the phase and 1.45x on
the whole prefill. This is the one change in this whole exercise that paid more
than predicted, and the reason is simply that nobody had looked at the loop.

**Chunk size is a real lever here, for an I/O reason.** At 128 positions a
302-token prompt is three chunks, and each chunk routes 512 (position, slot) pairs
over 128 experts — i.e. essentially the whole expert bank per chunk per pass. So
the cost of the prompt is proportional to the number of chunks, not to its
length: one 512-position chunk walks the bank once (110.97 GiB streamed) where
three 128-position chunks walked it three times (125–136 GiB). Buffers scale
linearly and total ~250 MB at 512, which comes out of the fixed scratch reserve.

**The grouped expert kernel now wins, and the earlier verdict was wrong for the
right reason.** Re-measured on synthetic matrices (`--cvt`, which also checks it
against the single-row kernel to 0.0):

| group | N x single-row GEMV | grouped GEMM | ratio |
|---|---|---|---|
| 1 | 80 GMAC/s | 72 GMAC/s | 0.91x |
| 2 | 77 | 117 | **1.51x** |
| 4 | 74 | 133 | **1.78x** |
| 8 | 77 | 143 | **1.85x** |

The first version lost because it kept its accumulators in a stack array and did
a horizontal reduction per block per activation. Splitting into two vector
accumulators per activation, four activations per register group, is all it took.
It is now the default; `MM_PREFILL_GEMM=0` reverts. A third of the experts in a
real chunk are asked for exactly one position (`fallback` in the profile), and for
those the single-row GEMV is still the right kernel — that path is kept.

**What is left, in order of size** (per token, 512-chunk): `io` 108 ms, `experts`
86, `proj` 81, `wo` 33, `attn` 16. `io` is ~31% and is the expert bank being read:
110.97 GiB in 32.6 s is 3.4 GB/s, and `--io` shows the device itself sustains
8.5–12.8 GB/s for 10 MiB preads, so the gap is real but the bytes cannot be
avoided — 216 GiB of experts against a 68 GiB pin cache. The next lever is the
FP8 dense GEMM's register blocking: it runs at ~82 GMAC/s while its inner loop
does one FMA per activation load, so a 4-row x 4-activation tile (16 accumulators,
one activation load reused across four rows) should be worth most of `proj`+`wo`.

### Long-prompt prefill, part three: 199 -> 77 s, and three changes that lost

Same 302-token prompt, `tools/mm_run`, top-2, `--context 8192`. The round that
followed the attention fix went further by looking at *shapes* rather than at the
phases: a per-shape counter (`mm_fp8_stats_print`, `MM_PROF=1`) showed which
matrix was paying, and two of the three things it found were in code that had
never been batched at all.

| | prefill | tok/s |
|---|---|---|
| start of this round | 199.2 s | 1.5 |
| attention vectorised, `MM_CHUNK` 512 | 104.2 s | 2.9 |
| FP8 dense GEMM tiled 4x4 | 89.1 s | 3.4 |
| shared expert + dense MLP batched per chunk | **76.7 s** | **3.9** |

**The FP8 dense GEMM was issue-bound on activation loads.** Its inner loop did
one FMA per activation load and re-read the whole activation matrix once per
weight row -- 60 GB of L2 traffic for 15 G MAC on an 8192x6144 matrix. Tiled
4 rows x 4 activations with 16 accumulators (four weight vectors and four
activation vectors live, which fits the 32 NEON registers), each load now serves
four FMAs: **96 -> 400 GMAC/s** standalone, 8.3e-6 relative error against a naive
triple loop that also exercises the tails. `--f8 N K n` reproduces the number.

**Two whole matrices were still running as one-row GEMVs per position.** The
shared expert (sw1/sw2/sw3) and the three dense layers' MLPs were invoked inside
a `for (t = 0; t < n; t++)` loop, so a 302-token chunk issued 34,428 single-row
GEMVs for the shared expert alone: 14.8 s, at 67 GMAC/s, for the largest single
compute item in the prefill -- and it was invisible in the phase profile because
it landed inside `proj`. As chunk GEMMs the same work runs at 228 GMAC/s. This is
the same lesson as the projections, one level up: at n=302 the weight bytes are
read once and used 302 times, and only a GEMM does that.

Three changes were measured and rejected, which is worth as much as the wins:

* **Eight activations per expert group** instead of four: 125 GMAC/s against 143
  at n=8. Sixteen accumulators plus the decode spills out of the 32 registers.
* **A blob recycling pool in `stream.c`**: expert reads looked like they were
  paying for an mmap-per-miss (3.4 GB/s against 6.7-7.0 from `dd`), so blobs were
  held in a bounded free list. Prefill did not move (89 -> 93 s, i.e. noise). The
  `dd` comparison was the wrong one: those files were page-cached, while the
  stream fds carry `F_NOCACHE`, so 3.4 GB/s is the device with caching off.
  Reverted rather than kept "because it should help".
* **Copying the FP8 dense weights into anonymous memory** to guarantee residency:
  footprint 82 -> 92 GiB, and the 10 GiB came straight out of the page cache the
  expert reads live in. io went 102 -> 116 ms/token and net time got worse. The
  mapped pages were already resident.

What is left is now dominated by things that are not arithmetic: **io 96 ms per
token (38%)** -- 110 GiB of expert bank at 3.4 GB/s, i.e. the device with
`F_NOCACHE`, against a 216 GiB bank and an 81.5 GiB cache -- then experts 93,
proj 24, attn 16, wo 10. The compute side is close to what these kernels can do
on 12 cores; prefill of a *cold* 302-token prompt is within 2x of the time it
takes to read the experts it needs.

One more negative result on that `io` term, because it is the one people reach
for first: **sorting a layer's expert reads by file offset so the device sees
sequential sweeps makes it slower**, 3.65 GB/s against 3.98, on the same prompt
(108.83 GiB in 32.0 s against 107.50 GiB in 29.0 s). A 10 MiB pread is already
large enough that randomness is not the problem, and twelve streams each walking
its own region beat one sweep. The `io` term is therefore where it is: 3.4-4.0
GB/s is what `F_NOCACHE` reads deliver on this machine, and the only ways past it
are fewer bytes (cache hits) or different hardware -- not a smarter read loop.

Which leaves the FP4 expert kernel as the last 2x on the table. It runs at
139 GMAC/s, about 70% of this core's sustained f32 FMA issue rate, so the ops per
MAC have to come down: the nibble decode plus four f32 FMAs per 16 elements
should become `bfdot`/`vfmmla`, which do 8 MAC per instruction instead of 4. That
is a rewrite of the kernel plus its LUT layout, not a tweak, and it is the only
remaining item with a factor of two in it.

### The FP4 kernel's last 2x: measured, and it is not there

The plan for the expert kernel was to swap four f32 FMAs per 16 weights for bf16
dot products (8 MAC per instruction instead of 4). `tools/bfdotbench` exists to
check the premise before writing the kernel -- one thread, same loop shape,
`make bfdotbench`:

| kernel | GMAC/s (1 thread) | vs f32 |
|---|---|---|
| f32 `fma`, 4 accumulators | 16.8 | — |
| bf16 `bfdot` | 23.1 | **1.37x** |
| fp16 `vfmlal` (widening) | 38.5 | **2.29x** |

Two things fall out. First, **bf16 is the wrong target** -- it is barely faster
than f32 here, and it also has the narrowest mantissa. Second, **fp16 is
genuinely 2.3x on a dot product**, and it is the natural format for this kernel
anyway, because the existing FP4 decode already produces f16 bit patterns
(`tbl` gives the f16 high byte, one shift completes it) -- so "decode to f16 and
use widening MAC" removes the f32 conversion as well.

Then the instruction count says no. Per (weight row, activation) per 16-element
block:

* today, f32: 4 loads + 4 mul + 2 add + 2 scaled-FMA = **12 ops**
* with fp16: 2 loads (half the bytes) + 4 widening MAC + 2 add + 2 scaled-FMA = **10 ops**

so 1.2x, not 2.3x. The 2.3x is the *MAC* rate, and this kernel is not
MAC-limited: it is issue-limited, because every block needs its activation
vectors loaded and its partial sum scaled by a per-block e8m0 factor. The bench
measures a pure dot product, which has neither. Adding activations per group is
the only way to amortise the decode, and eight activations was already measured
and spills the 32 registers (125 vs 143 GMAC/s). Row blocking on top of it needs
4 x 4 x 2 accumulators for the same reason.

So the expert kernel stays as it is at 139 GMAC/s, and the honest statement is
that the compute side of this model on 12 cores is finished: io 96 ms/token,
experts 93, proj 24, attn 16, wo 10, route 6.

The one structural change that would still pay is not a kernel at all: **the
read-ahead and the expert compute are serialised**. `cwick_stream_expert_reserve`
makes the reads and then waits for all of them before the first GEMV, and the
reading threads are idle the whole time because 3.4 GB/s is the device, not the
core. Two or three threads reading the rest of the layer while the other ten
compute experts that have already landed would hide most of the 96 ms. That needs
a completion signal per blob and a second dispatch path, i.e. a design, not a
tweak -- and it is the only item left with a factor of two in it.

### Read-ahead and compute now overlap: io 96 -> 50 ms/token

The last item with a factor of two in it was not a kernel. `mm_expert_prefetch_layer`
reserved every blob a layer needed, read them all (blocking), and only then let the
first GEMV start — so the twelve compute threads sat idle for 96 ms of a 250 ms
token while a device that sustains 3.4-4.0 GB/s was the only thing working. Every
attempt to make that read *faster* had failed (blob pool, offset sorting, resident
copy), because the rate is the hardware's. What was fixable is that nothing else
happened during it.

The layer's reads are now split: the first half on the pool as before, the second
half handed to four helper threads that read **directly** — not through the pool,
so they run alongside the GEMVs instead of queueing behind them — and the first
half's experts are computed while they work. The union loop joins at the union
index where the second half begins (`umid`), and again after the loop as a net.

| | before | after |
|---|---|---|
| io | 96 ms/token | **49-55** |
| experts (now includes join waits and contention) | 92 | 106-108 |
| token | 250 ms | **219-227 ms** |
| 302-token prefill | 75.5 s | **66-68 s (4.4-4.6 tok/s)** |

Output is unchanged bit for bit (`the quick brown fox ... lazy` -> ` dog` 19.8344),
`make test` passes, and decode is untouched: the overlap is gated on `n > 1`,
because a decode step has four experts per layer and one position, so the handoff
would be 57 x 4 thread spawns for nothing.

Three tunings measured on the way:

* **Four helpers, not more.** Six measured 74.9 s against 66.1: the machine is
  12P+4E, so six readers plus twelve pool threads oversubscribe it and `experts`
  goes 106 -> 115 while `io` goes 49 -> 56. The helpers are I/O-blocked, but not
  free.
* **A 50/50 split.** The second half must finish inside the first half's compute
  time; halving both is the natural first guess and it left 55 ms exposed, so it
  is roughly balanced rather than badly tuned.
* **`MM_IO_OVERLAP=0`** reverts to the serial path, which is worth keeping: it is
  the reference for exactly this comparison.

### Decode: where the rate stands, and why the prefill trick does not transfer

Prefill measures 4.5 tok/s (302 tokens in 66.9 s). Decode does
not: measured 513-577 ms per step after a shallow prefill (1.7 tok/s), 857-1058
ms after a 302-token one (1.1 tok/s). The difference between those two is not
code, it is cache state: `io` is 109-210 ms in the first case and 250-383 in the
second, because the prefill has just walked the whole expert bank and evicted
what decode was using. In a real API request the prefill always happens first, so
the second number is the one users see.

A decode step moves ~22 GB of weights: `wq` 2.87, `wo` 2.87, `wk`/`wv` 0.36, the
shared expert 3.2, the 228 routed matrices 12.9. A 250 ms step would mean
**88 GB/s sustained** for the whole step, against this machine's 123.5 GB/s
ceiling and kernels that measure 78 GMAC/s (fp4 GEMV) and 114 GMAC/s (fp8 GEMV).
So it is not impossible, but it leaves no room for a stall: every stream has to
run at ~71% of peak and the SSD has to be invisible.

Where the 513 ms goes on a warm-ish step, and what each part would have to be:

| phase | now | at 100 GB/s | note |
|---|---|---|---|
| io | 109-210 | 0 | 5-14 misses x 19 MB at 3.4 GB/s |
| experts | 165-198 | ~130 | 12.9 GB, 65-78 GB/s, 684 separate GEMVs |
| proj | 83-139 | ~32 | 3.2 GB but only 23-38 GB/s |
| wo | 31-34 | ~29 | 2.87 GB at 84 GB/s, already near peak |

`proj` is the outlier: the same-shaped `wo` runs at 84 GB/s while `proj` runs at
23-38. It is five GEMVs per layer (285 per step), and the per-shape counters show
the small ones are all overhead -- `[512x6144]` 180 calls for 18 ms (30 GB/s) and
`[128x6144]` 60 calls for 4 ms (11 GB/s) when the bytes involved are 3 MB.

The prefill overlap does **not** transfer, and that was measured rather than
assumed: routing every decode miss to the four helper threads, computing the
resident experts first and joining at the boundary, is inert because `overlap` is
`n > 1`. Forcing it on would be wrong anyway -- a decode layer has four experts
and one position, so there is no second half to overlap, and four serial readers
are slower than the twelve-thread pool that was doing the reads. Decode's `io`
needs a cache answer (raise the hit rate from 94-97% toward 100%, or stop the
prefill from evicting the decode working set), not a scheduling one.

The two changes that would move decode, in order:

1. **Merge the five projections into one dispatch per layer** (285 -> 57 pool
   calls per step). `proj` is overhead-bound, not bandwidth-bound, so this is
   worth ~50-90 ms of the 513 with no arithmetic change.
2. **Merge the twelve expert GEMVs of a layer into one dispatch.** Same argument
   at a larger scale, plus longer streams: 684 -> 57 calls and 12.9 GB of
   back-to-back reads instead of 19 MB bursts.

Both need a range function that maps a global row index to (matrix, row), which
is mechanical; neither changes any result bit.

### Decode: the arithmetic says the machine is the limit

60 decode steps after a 302-token prefill (the shape a real API request has):
**1.4 tok/s**, 694 ms mean, individual steps 436-820 ms depending on whether they
hit the SSD. Output unchanged (` dog` 19.8344).

A step moves 22.2 GB of weights: 12.9 of routed experts, 2.87 `wq`, 2.87 `wo`,
3.2 shared expert, 0.36 `wk`/`wv`. So the measured effective rate is
22.2 / 0.694 = **32 GB/s**, against a 123.5 GB/s ceiling. A 250 ms step would be
**89 GB/s sustained**.

Two decompositions, both from measurement:

1. **`io` is 109-239 ms of the step** (hit rate 94-97%: 5-14 misses of 19 MB at
   3.4 GB/s). Removing it *entirely* leaves 524 ms = **2.0 tok/s**. That is the
   ceiling with no storage in the step at all.
2. Those 524 ms move 22.2 GB at 42 GB/s composite, while the phases measured
   individually run at 60-84 GB/s. A 250 ms step needs another **2.1x** on
   streams that already sit at 50-68% of this machine's peak.

So further tuning does not close it here: the shortfall is 2.1x on
bandwidth-bound kernels plus a perfect cache, and the machine has neither. The
only two ways past this are (a) **speculative decoding** -- the checkpoint ships
`dspark/`, a 7-token block draft model, which is the one way to get several tokens
per weight read -- or (b) different hardware/an implementation that can actually
use the GPU (the Metal kernels here measure 33 GB/s against a ~400 GB/s device
because they are not tiled).

Recorded so that the next session does not repeat the loop: prefill is done
(4.5 tok/s), decode is 1.4 and the gap is not a bug.

One negative result from this round, for the same reason: merging the five
projections into one pool dispatch measured proj 92-97 ms against 83-139 before,
i.e. inside the noise, and was reverted. The per-call overhead hypothesis was
wrong -- `[512x6144]` runs at 30 GB/s not because the dispatch costs 50 us but
because 512 rows over twelve threads is 42 rows of work each.

### max_tokens is a ceiling, not a demand: clamp it, do not 400

A client that sent `max_tokens: 32768` against the 32768-position MiniMax
server with an 8K prompt got

```
prompt 8315 + max_new 32768 at position 0 exceeds the 32768-position context window
```

and no answer — while the reply it wanted was a few dozen tokens away. The GLM
path had already learned this (its decode budget is clamped to what the KV
window still holds, `test_max_new_clamped`), so the bug was really that the two
newer backends disagreed with the oldest one about the same request.

`cwick_gen_window(pos, prompt_len, &max_new, capacity)` now owns the rule for
all three: reduce `max_new` to `capacity - pos - prompt_len` when it overruns,
and fail only when the *prompt* does not fit — those two cases have different
fixes (lower max_tokens vs. shorter prompt or bigger `--context`), and the old
message conflated them. Measured end to end: the failing request becomes
`200`, `content="Hi there, friend!"`, 28 tokens, and the log says

```
max_tokens 32768 clamped to 32570: prompt 1 at position 197 leaves 32570 of 32768 positions
```

A clamped budget is a *ceiling*, not a length, so the progress lines stop using
it as a denominator: `decode 6 tokens (context cap) — 895 ms/step` instead of
`1/32570 (0.0%)`, which reads as a stalled request. Same reasoning GLM already
had in `maybe_progress_at`.

One consequence worth stating: honouring a large `max_tokens` can now mean a
very long run (24151 tokens at 1.3 tok/s is hours) where it used to be an
instant error. That is the right trade — `max_tokens` is what the client asked
for, and the model stops at EOS long before it — but a client that wants a
short answer should send a small `max_tokens`, and the server should keep
reporting the ETA per progress line so the ceiling is visible.

### Prefill progress: the line existed, but not while work was happening

Three things were wrong with the MiniMax prefill progress, and only the first one
is about printing at all.

**It fired only at chunk boundaries.** `mm_prefill_progress_done` is called once
per `MM_CHUNK`, and a 512-position chunk takes ~140-170 s here, so the log showed
`1024/8321` at 21:41:01 and `1536/8321` at 21:43:51 -- one line every ~170 s. A
prompt that fits in a single chunk (any request under 512 tokens, i.e. most of
them) printed **nothing at all** until the prefill was over: the first line a
201-token request ever produced was `1/1 tokens (100%)`. The GLM path had already
solved this with a layer counter (`maybe_progress_at(..., layer, nlayer)`); the
MiniMax loop had no equivalent.

**The layer gauge was pinned at the maximum.** `cwick_metrics_set_prefill(done,
total, layers, layers)` passed the checkpoint's layer count as the *current*
layer, so `/v1/metrics` reported `cwick:prefill_layer 60` with `prefill_layers 60`
from the first second onward -- a dashboard draws that as "almost done" while 30
minutes of work remain. The token gauge had the same shape of problem from the
other side: it stepped 1024 -> 1536 every 2.5 minutes and read 0 for the whole of
a single-chunk prefill.

**Nothing in the line said where it was.** No layer, no footprint, so "slow" and
"stuck on the SSD" were indistinguishable. The GLM line carries the memory tail
for exactly this reason.

The fix is `mm_prefill_progress_layer()`, called at the end of each layer inside a
chunk: one line per 8th layer or per `CWICK_PROGRESS_MIN_MS` (5 s by default),
whichever comes first, and the same values published to the gauges.

```
[prefill] ~38/333 tokens (11.4%), layer 7/60 — 6.3 tok/s, eta 47s
[prefill] ~310/333 tokens (93.1%), layer 56/60 — 4.5 tok/s, eta 5s
[prefill] 333/333 tokens (100.0%), layer 60/60 — 4.5 tok/s, eta 0s
```

The token count inside a chunk is an **estimate** (`~`) because the positions of a
chunk advance together through the layers: `done + layer/nlayer × n`. Without it a
single-chunk prompt prints `0/201 (0.0%)` for two minutes and then jumps to 100%,
which is the same information content as no progress line at all. At the last
layer the estimate becomes exact, and `_done` prints the exact number at the
boundary, so the `~` always disappears before the next chunk starts.

Decode is unaffected: `mm_prefill_progress_layer` returns immediately unless
`begin()` armed it, and `mm_forward_chunk` only calls it with `n > 1`.

Measured on a 333-token prompt: 15 lines, 3-6 s apart, eta 47 s -> 0 s monotonically,
`cwick:prefill_layer` moving 40 -> 48 in the same window that used to sit at 60.

Two things this does **not** fix, both bigger than the logging:

* an 8321-token prefill still takes ~30 minutes at 4.5 tok/s, i.e. this machine's
  ceiling. The run I interrupted had streamed **616.65 GiB** (2.7x the whole
  checkpoint) with 0 decode steps -- it was working the entire time, it just had
  no way to say so;
* one long prefill holds the single generation slot, so later requests queue and
  then get `429 queue full`. That is a scheduling decision, not a logging one.

## 27. MiniMax decode: the ceilings first, then three changes worth 2.1x

The plan for this backend was borrowed from the GLM profile: attention is 117 of
a 275 ms step there, so moving the attention core onto the GPU was the first
item. One `MM_PROF=1` run on MiniMax killed that item before any work went into
it:

```
mm prof  974 ms: proj 137  attn 4  wo 41  route 7  experts 391  io 359  dense 11
```

Attention is 0.4% of the step. What is left is `io` (waiting for the SSD) and
`experts` (the fp4 GEMV), and they are not independent: the byte count decides
the first and the cache decides how much of it lands on the second.

**Measure the machine, not the kernel.** `make mmdecodeprobe` reports the two
ceilings a step is scored against. Resident-memory reads: 37 GB/s on one thread,
91 on four, 119 on eight, 122 on sixteen — the ceiling is reached at eight and
more threads only add contention. Shard reads are a different animal and depend
on *which* files are being read: a decode layer's misses land in one shard, and
the same-shard rate is a fraction of the cross-shard one, with read size and
per-thread descriptors both measured as no help. (That tool's absolute numbers
drift once the file is warm, so the engine-level A/B below is what the decisions
rest on.)

Then three changes, each measured on its own against the one before it, in one
session, same binary, same prompt, 23 decode steps, `--context 2048`:

| | change | tok/s |
|---|---|---|
| A | as it was: serial reads, per-blob allocation, 80% cache budget | 0.686 |
| B | + fixed-size blob arena (block reuse instead of malloc/munmap) | 0.778 |
| C | + read dispatch by item count instead of by row count | 1.020 |
| D | + cache budget capped at `host / 6` instead of the 80% plan | **1.450** |
| E | C with the 80% plan kept (63 GiB) | 1.122 |

**1. Reads have to be dispatched by item count.** `cwick_pool_for` splits work
by `CWICK_POOL_MIN_WORK`, a per-*row* threshold, so a decode layer's three or
four misses fell below it and ran inline — one 10 MiB pread after another on the
calling thread. The whole read phase is worth 4.9 GB/s that way and 10.9 GB/s
with four in flight. `cwick_pool_for_items` already existed for exactly this
(`pool.c` says so in a comment) and was simply not used here.

**2. The blob arena is for plain memory, and only for it.** Every routed-expert
matrix of a checkpoint has the same shape, so the pin cache's unit of allocation
is a constant and ~600 malloc/munmap pairs per step can be replaced by blocks
from a few large mappings. It is worth +13% on MiniMax. Handing the same arena
to GLM is a **17% loss** (2.09 -> 1.74 tok/s, reproducible), because those blobs
are GPU-visible and the Metal arena has its own allocator tuned for 8 MiB units
— batching them into 1 GiB mappings changes how that arena grows and wires. So
the pool is opt-in per backend (`cwick_stream_enable_blob_pool`, called from
`mm_model_open`) and the other two backends keep the path their numbers were
measured on.

**3. The pin cache does not want the whole plan.** This is the largest single
lever, and it is a subtraction:

| cache | 14 GiB | 18 | 22 | 30 | 44 | 63 (the 80% plan) |
|---|---|---|---|---|---|---|
| tok/s | 1.39 | 1.58 | **1.62** | 1.58 | 1.16 | 1.08 |
| hit rate | — | — | 67% | — | — | 78% |

A bigger cache does stream fewer bytes and does raise the hit rate, and the step
still gets longer. Past ~30 GiB the compressor starts taking pages, and a page
that has to be uncompressed on the way to a GEMV costs what a page from the SSD
costs — while the phases that never touch the cache (projections, attention)
slow down with it. The plateau is 18-30 GiB, which `host / 6` (21.3 GiB on a
128 GiB machine) lands in the middle of, and it scales down on smaller machines
instead of assuming they have room.

**What was tried and is not in the table**, because each was measured and lost:

* attributing the slowdown to mmap page faults: the blobs are 100% pin-cache
  (a counter says so), 0% file mapping;
* wiring the blob pool (`mlock`): compression disappears and GEMV stays flat at
  117-123 ms even at 56 GiB, but end-to-end is *worse* (1.45 tok/s) — a wired
  21 GiB is memory the rest of the process cannot use;
* 2 MiB superpages for the arena: 408 -> 312 ms peak GEMV, i.e. the knee at a
  given footprint did not move;
* a bigger cache with any of the above: E in the table.

This measures 1.48 tok/s; the gap is bytes per step against a
same-shard read ceiling, not a kernel. The next lever is the one direction that
does not need more bandwidth: produce several tokens per weight read (the
checkpoint ships `dspark/`, a 7-token block drafter), with the contracts that
come with it — state rollback to the accepted prefix, sampling that stays
correct, and a prefix snapshot that carries the drafter's state too.

## 28. Block verification: the machinery, and what an n-gram drafter says about it

Every step above this one bought time by moving fewer bytes. There is one
direction that does not need bandwidth at all: produce several tokens per
weight read, and verify them in one forward. This is the machinery for that,
measured with the cheapest possible drafter so that the *cost of verifying a
block* is separable from the quality of whatever proposes it.

**The three pieces.** `mm_model_forward_multi` emits a distribution for every
position of a chunk (`spec_out`), because candidate *i* can only be judged by
the distribution of position *i-1*; producing them one at a time would cost
what speculation is meant to save. Acceptance is argmax equality — greedy only,
since exact speculative sampling at temperature > 0 needs the sampler's
filtered distribution per position and it is not exposed; doing it approximately
would silently change the output distribution. Rollback needs no code: a
rejected candidate was written into the K/V of a position beyond `pos`, and the
next block overwrites it.

**Is it exact?** With the block forced back to single positions
(`MM_SPEC=8 MM_CHUNK_CAP=1`), the output is byte-identical to the same request
without speculation. Without that, it is not: the n=8 path runs the projections
as a chunk GEMM and the n=1 path as a GEMV, the two round differently in the
last bits, and a near-tie in the argmax flips (`short` -> `shorter`). So the
claim this engine can make is "exact with respect to the block's own
arithmetic", not "bit-identical to the single-row path" — the same caveat the
prefill path has always had, now on the decode path.

**What a block costs.** One forward over 8 positions streams about twice the
expert bytes of a one-token forward: 7.2 GiB/step against 3.6 GiB/step on the
same prompt. The union of 8 positions' experts is much smaller than 8x — adjacent
positions overlap heavily — but it is not free, and it is the number every
drafter has to be judged against.

**What an n-gram drafter buys.** Drafts come from the token stream itself (the
nearest earlier occurrence of the trailing 1-4 gram), so they cost nothing.

| prompt | acceptance | tokens/forward | tok/s |
|---|---|---|---|
| repeat one line five times | 28/44 = 64% | 1.76 | 1.486 -> **1.612** |
| "explain why the sky is blue" | ~0% | 1.00 | 1.163 -> 1.100 |

So: on text that repeats itself, +8.5%; on prose, -5%. The break-even is
tokens/forward > 2, i.e. roughly 85-90% acceptance at a block of 8, and even a
perfect drafter is capped near 2x by the union cost above.

**Which is the answer to the real question.** The checkpoint ships `dspark/`, a
7-token block drafter (5 layers, dense FFN, rank-256 markov head, confidence
head, no embed or head of its own — it reads the target's, and it consumes the
hidden states of target layers 3/17/31/45/59). Porting it is a research project
of its own, with no reference implementation in the checkpoint and no
specification for its block semantics. The measurement above says what it would
have to clear: >= 2.5 tokens per forward on ordinary text. That is a plausible
target for a trained drafter and an implausible one for n-grams, which is why
this stops here with the machinery in place, `MM_SPEC` off by default, and a
number to hold the next attempt to.

One unrelated observation from the same sessions: the same greedy request
produced different tokens when the prefix snapshot was replayed (210 positions
restored) than when the same prompt was prefilled from scratch. Both runs were
greedy with the same seed, and neither had speculation enabled. It is filed
here because it will mislead the next person who compares two runs and expects
determinism; the cause is not known.

## 29. Tool calls: what the stream shows while the block is still being written

A tool call is the longest silence this server has. `cwick_stream_segment`
classifies a `<tool_call>` body as neither reasoning nor content — correctly, it
is parsing material — so from the moment the opener lands until the request ends
the client receives nothing at all. On a slow checkpoint that is tens of seconds
of a chat bubble that looks hung, and the parsed `tool_calls` only arrive at the
end because they cannot be produced before the block closes.

**The fix is a second field, not a second copy.** Those bytes now go out as
`delta.cwick_tool`, a non-standard field next to `content` that clients which do
not know it ignore. The web UI renders them as a mono card with a live pulse and
clears it the moment the parsed `tool_calls` arrive, so the raw and parsed forms
never overlap. Framing is stripped before the slice is sent, and a tail that
could still grow into a frame is held back (`cwick_tool_ns_tail`) so a partially
received marker is never shown — the same hold-back rule the content/reasoning
split already uses.

**Why a checkpoint's tool block needed a translator.** That checkpoint writes XML
inside the same `<tool_call>` wrapper, with a namespace token — one entry in its
added-token table, spliced in by its template — in front of *every* structural
tag:

```
]<]minimax[>[<invoke name="add_numbers">]<]minimax[>[<a>137</a>]<]minimax[>[<b>258</b>]<]minimax[>[</invoke>
```

The model is following its own spec; a naive scan reads the framing as part of
the tool name and refuses the whole generation as an undeclared tool, which is
what a real turn did. Rather than teach the parser a second grammar,
`translate_invoke` rewrites the block into the `<arg_key>/<arg_value>` form the
parser already handles — one parser, one set of edge cases.

Four of its bugs were only visible to a probe fed the real bytes, not to the
unit tests: the original opener was emitted twice so the name came back as
`<tool_call>add_numbers`; scanning for children started *on* the `>` of
`<invoke ...>` so every argument was dropped; a block in the other form with the
same framing looked unterminated (`strstr` for `</tool_call>` never matched
through the token); and a token that landed *inside* a value left `137]` as the
value with `]minimax[` parsed as the next argument's key — the token contains a
`<`, so any scan that stops at a tag boundary stops inside the framing. The last
one is the reason the token is now removed from the whole answer before any
grammar looks at it: framing is framing wherever it appears, and one rule beats
one rule per position. All four are assertions in `tests/test_cwick.c`,
including the observed drift where the model opens a block, writes prose, and
only then opens the one it fills in — that has to yield one declared call.

Measured end to end on a real turn: 24 `cwick_tool` frames over the life of the
call, no framing in any of them, `finish_reason: tool_calls`, and
`{"a":137,"b":258}` as the parsed argument — where the same request before this
returned `{"a":"137]","]minimax[":"[",...}` and was refused as an undeclared
tool.

## 30. The browser side of a turn: three live regions, and running what it wrote

A turn produces three things at once and the bubble used to show them one at a
time: reasoning was painted, but a call in flight was not (it was parked inside
the prose container), and a *parsed* call was not painted at all until the whole
message was re-rendered at the end. All three are now separate live regions
above the prose — the reasoning panel, a tool container holding either the
block being written or the parsed card, and the prose — repainted from one
snapshot, so none of them waits for another.

Two details the streaming painter has to get right. The containers are created
lazily (an empty reasoning box on every direct answer is worse than no box),
and the reasoning panel is inserted *before* the tool container when both
exist, so the order stays the order the model produced them in rather than the
order the deltas happened to be handled in.

Fenced blocks now carry the bytes out of the page: copy and download on every
block, and **run** on a block that is a page — language is `html`/`svg`/`xml`,
or no language with `<!doctype html>`/`<html` as the first tag. Running opens a
modal with a sandboxed iframe (`allow-scripts allow-modals allow-forms
allow-popups`, deliberately **without** `allow-same-origin`): the snippet runs,
and it cannot read this origin's storage, cookies or the session token. Verified
in a real browser against a same-origin parent: the script ran, and parent DOM,
localStorage, cookies and XHR were all denied.

The preview is a modal rather than an inline frame for one reason: the bubble is
re-rendered on every streamed token, so an inline iframe would reload on every
token — and moving an existing iframe in the DOM reloads it too. The modal
lives outside the bubble and survives the repaint; closing it swaps the iframe
node, which is the only reliable way to stop timers a snippet left running
(setting `srcdoc` to nothing still leaves a live document).

Verified by driving the real `web/app.js` in a DOM against a synthetic SSE
stream (28 assertions): the three regions all present mid-stream, the parsed
card replacing the in-flight block, the html block getting run/download/copy and
the python block getting only copy/download, the sandbox string, and the modal's
copy/download/close/Esc paths.

## 31. Two runs of the same prompt produced different answers -- and why

The MiniMax port was not deterministic. Two `mm_run` invocations with the same
prompt produced different next-token logits (18.44 vs 19.13 for the same
token), and the two runs disagreed on how many expert matrices they had read
(4458 vs 4406). Every decode measurement this backend reported was therefore
taken on a machine that was quietly changing its answers between samples.

### How it was localised

Guessing did not work; measuring did. Two gated debug aids (each off unless an
environment variable is set) turned "the model disagrees with itself" into a
single line:

* `MM_TRACE_HASH=1` hashes the residual stream after every phase of every
  layer and prints `trace L<layer> <phase> <hash>`. Comparing two runs showed
  the first difference was always *after* routing and *before* the layer's
  residual was combined -- i.e. inside the expert FFN, with the router's own
  selection identical on both sides.
* `MM_TRACE_W=1` hashes the bytes of every expert matrix at the moment it is
  handed to a GEMV. That pinned it exactly: at layer 36 the same expert's
  `up` and `down` matrices hashed the same in both runs while its `gate`
  matrix did not. The weights themselves were different, not the arithmetic.

### The cause

`cwick_stream_expert_reserve()` hands the caller a writable buffer and the read
happens *afterwards* -- the whole point is that a layer's misses are reserved
on one thread and then pread on the pool. But a reserved slot was an ordinary
cache entry with `heat = heat_avg`, so the next reserve in the same loop, if it
had to make room, could evict a slot whose read had not been dispatched yet.
The caller's `dst` then pointed at a released block, the following admission
handed the same block out again, and two reads landed in overlapping memory.
The model kept running -- most bytes were right -- but which read won depended
on thread scheduling, which is why it looked like non-determinism rather than
corruption.

The fix is one state bit: a reserved slot is `pending` until its read lands,
and a pending slot is never an eviction candidate (in the probe-window victim
choice, in the sampling victim choice, and in `cache_make_room`'s fallback
sweep). `cwick_stream_expert_settle()` clears it once the read is accounted;
every reserved slot is either settled (success) or aborted (failure), so
nothing stays protected forever. Two `cwick_stream_expert_put_read()` paths
also returned with the cache mutex held, which would have deadlocked the next
lookup; those are fixed too.

Verified: three consecutive runs of the same prompt now agree on all 5957
traced phases, including every weight-byte hash, and report the same expert
read count.

### What it changed about the profile

The corrupted entries were also wasted bandwidth. Decode steps now stream
2.55-4.38 GiB at a 67-81% hit rate, with the read-ahead phase at 277-384 ms of
a 574-692 ms step; before the fix the corresponding lines were 4.31 GiB at 67%
and 420 ms of 714 ms. Those two sets are not a controlled comparison -- the
runs decoded different tokens, which is the whole point of the section -- but
the direction is the one the mechanism predicts: a slot evicted before its read
is a read paid for and thrown away.

## 32. One reduction order, so a verified block is the same computation

Speculative decoding here works by reading the weights once and scoring several
positions against them, then keeping the candidates whose argmax the model
itself produced. For that to be a speed-up rather than a different model, the
batched pass has to return the *same bits* as the step-at-a-time run it
replaces.

It did not, and `MM_SPEC` was left off by default for exactly that reason. The
cause is in `src/mm_model.c`: the FP8 dense kernel had **four** reduction trees
for the same product -- the single-activation path summed inside each
32-element block and applied the block scale afterwards; the row-tiled
prefill path accumulated flat across the whole row with the scale folded into
each element; and the two tail paths (rows past a multiple of four, and
activations past a multiple of four) used a pair of accumulators eight
elements apart. Feed the same matrix to two of them and you get two answers:
`--f8` now measures 130805 of 131072 values differing at n=16.

Reassociating 6144 products moves only the last bit or two, but over 60 layers
that flips a near-tied argmax, which is why a speculative run produced
`short` where the plain run produced `shorter`.

The fix is a threshold, not a rewrite. `MM_FP8_EXACT_N` (default 8) is the
largest `n` that pays for agreement: at or below it every activation goes
through the single-activation kernel, which is the reference by construction,
because a decode step *is* that kernel. Above it -- a prefill chunk, whose
outputs are never compared against anything -- the row-tiled kernel is used as
before. The threshold is also a speed win below four activations: the tiling
measured 2.4 ms at n=2 where doing the fused pass twice costs 1.4 ms.

What it costs, measured with `--f8 8192 6144 <n>` in one sitting (each number
is the whole matrix, not per activation):

| n | before | exact path | tiled (unchanged) |
|---|---|---|---|
| 1 | 0.7 ms | 0.6 ms | - |
| 2 | 2.4 ms | 1.4 ms | - |
| 4 | 1.4 ms | 2.8 ms | 1.4 ms |
| 8 | 2.0 ms | 4.5 ms | 2.0 ms |

So a verified block of 8 costs about 2.3x the dense-GEMM time of the step it
replaces. That is the price of agreement, and it is the number that decides
whether a given block length is worth it -- which is exactly the cost model
`verify_select` uses, and which this backend still does not have.

`--f8` now also prints a bit-exactness check on every run: every position of a
block against the same position computed alone, with full-mantissa synthetic
activations (small multiples of a power of two round identically in any order
and would pass either way). It reports `0 of 65536 values differ` at n<=8 and
fails loudly at n>=16, so the threshold cannot drift without being noticed.

The FP4 expert kernel has the same problem and the same answer in a different
form: its grouped kernel now combines the two nibbles of a packed byte with a
fused multiply-add, matching the per-row kernel instruction for instruction
-- same lane mapping, same two accumulators, same block order, same final
reduction. `--cvt` checks this on every run and reports the number of rows
that are not bit-identical (currently 0 of 3072).

## 33. Block verification is now exact and still does not pay -- measured, closed

With the two reduction orders unified (SS32) the question "is speculation worth
it on this model" could finally be answered by measurement instead of by a
guess about acceptance rates. `tools/mm_run --ab N` runs three passes in one
process on one loaded model -- plain greedy, speculative, plain again -- and
diffs the token sequences. The third pass is the control that matters: by then
the pin cache is warm, and on this model cache warmth is worth more than
speculation, so a single before/after comparison would credit the speculative
pass with a cache the plain pass paid for.

Repetitive prompt (asking for the same sentence five times), block of 8:

| pass | tokens | ms | tok/s | forwards | tokens per forward |
|---|---|---|---|---|---|
| plain | 40 | 37578 | 1.064 | 39 | 1.03 |
| spec=8 | 40 | 56002 | 0.714 | 23 | 1.74 (accepted 16/71 = 23%) |
| plain again | 40 | 35408 | 1.130 | 39 | 1.03 |

Prose prompt, block of 5:

| pass | tokens | ms | tok/s | forwards | tokens per forward |
|---|---|---|---|---|---|
| plain | 32 | 35134 | 0.911 | 31 | 1.03 |
| spec=5 | 32 | 32471 | 0.986 | 21 | 1.52 (accepted 11/29 = 38%) |
| plain again | 32 | 27068 | 1.182 | 31 | 1.03 |

Both runs print `spec vs plain: identical token for token`, and so does the
plain/plain control. That is the result the fix in SS32 was for: block
verification no longer changes the answer, so the only remaining question is
arithmetic.

And the arithmetic says no. A forward pass with 8 positions in flight costs
56002/23 = 2435 ms against 963 ms for a single step -- **2.5x**, for 1.74
tokens. Speculation needs more than 2.5 tokens per forward here and the n-gram
drafter delivers 1.74 on text built to be repetitive and 1.52 on prose.

The 2.5x has two causes and they scale differently:

* **The expert union grows almost linearly with the block.** Routing is
  top-4 of 128, so eight positions touch up to 32 distinct experts against 4
  for one position. Bytes per forward are proportional to the union, so the
  only way batching pays on a model whose weights do not fit is if positions
  reuse experts -- and they do not.
* **The exact FP8 path is expensive at n=8** (SS32: 4.5 ms against 0.6 ms for
  one 8192x6144 matrix). That is the price of `MM_FP8_EXACT_N=8` and it is not
  optional: without it the accepted token is not the token plain decoding would
  have produced. Cheaper exactness is possible -- give the tiled kernel the
  reference tree instead of routing small `n` to the per-row kernel -- but the
  union term above is already decisive on its own.

So `MM_SPEC` stays off by default, and this time it is a measured decision
rather than a caveat. The mechanism stays in the tree because it is now exact:
if a drafter that accepts more than 2.5 tokens per forward ever appears, it is
a configuration change, not a research project. The 7-token block drafter that
would have to be ported for that is 2.3 GiB resident, and it would have to beat
2.5 accepted tokens per forward by enough to also pay for its own reads.

Closing this reorders the MiniMax plan: the lever is bytes and hit rate, not
batching. The step is 574-692 ms of which `io` is 277-384 ms (waiting for the
SSD) and `experts` 132 ms (the GEMV work on what arrived); `proj` 92, `wo` 33,
`attn` 3, `route` 5. The expert reads cannot be prefetched across layers
either -- the router runs on the layer's own input, so layer L+1's expert set is
not known until layer L's FFN output exists. That is a structural serialisation
of a data-dependent router, not a tuning problem.

## 34. Two things that are not the lever: a bigger pin cache, and batching

Both were plausible and both are now measured, with the cache corruption of
SS31 gone -- which matters, because the earlier cache-size comparison was made
on a cache that was quietly serving damaged blobs.

**A bigger pin cache is slower, not faster.** `CWICK_EXPERT_CACHE_GIB` at the
default 21.3 GiB against 40 GiB, alternating so that neither side gets a warm
machine, two rounds each, same prompt:

| cache | hit rate | streamed | tok/s |
|---|---|---|---|
| 21.3 GiB | 64.5% | 190.85 GiB | 1.5, 1.5 |
| 40.0 GiB | 72.3% | 149.93 GiB | 1.3, 1.2 |

More capacity buys 8 points of hit rate and 21% fewer bytes, and loses 20% of
the token rate. So hit rate is not the binding constraint once the cache is
this size: what grows is the cost of *being* large -- a bigger index to probe
per lookup, more eviction work per miss, and a 40 GiB anonymous footprint whose
TLB behaviour a 21 GiB one does not have. The `host/6` cap in
`mm_expert_cache_budget` is therefore a local optimum that happens to be right,
and it should stay there until somebody measures the index and eviction costs
separately rather than guessing at them. Do not "just give it more memory".

**Read/compute overlap inside a layer is real but small in steady state** --
and it is now on by default; see the comment on `mm_io_overlap_decode()`. Over
24 steady steps: step 664 -> 638 ms, io 388 -> 200, experts 123 -> 282. The
188 ms that leaves `io` mostly reappears in `experts`, because that is where
the join is: what disappears for good is 26 ms. On the first steps of a session
the same switch is worth 277 ms a step (713 -> 436), because then there is six
times more io to hide.

Put together with SS33, the MiniMax picture is: a step is 664 ms of which 388 is
waiting for the SSD and 123 is GEMV on what arrived, the expert set for layer
L+1 is unknowable until layer L's FFN output exists (data-dependent router), the
weights do not fit so every step pays in bytes, and adding capacity or batch
width makes it worse rather than better. That is not a tuning surface; the
remaining wins are in *bytes per step* (a smaller expert footprint, i.e. fewer
distinct experts touched) and in the 84 ms of `proj` + 30 of `wo`, both of which
already run at the memory ceiling.

## 35. A decision endpoint: read the distribution instead of asking for prose

`--jev` serves `POST /v1/systemone`: a state, a handful of questions, and typed
answers with a distribution. The point is what it does *not* do — it never runs
the decode loop. One prefill reaches the answer position, the next-token
distribution there is restricted to the option letters, and that restriction is
the answer. Nothing is generated, so nothing has to be parsed, retried when the
model wraps its JSON in prose, or repaired when it writes a cell that is already
occupied.

**Readout.** Coded letters, not option text. Scoring each option's own words is
more principled — it removes the letter prior — but it costs one continuation
per option, and on this machine a token is the expensive unit (MiniMax 664 ms,
GLM ~275 ms per decoded token, versus one prefill for the whole question). The
letter path costs one prefill per question regardless of how many options there
are: the nine-cell board and the yes/no question cost the same. What it gives up
is the letter prior, which is measured rather than argued: reordering the same
three options (info / replace / refund) selected the same key with the same
probability to four digits, so on this checkpoint the prior is not what drives
the answer. That is one checkpoint and one prompt; it is not a general claim.

**Prompt.** Each question is rendered through the checkpoint's own chat format
with the thinking block closed before the model writes (`<|assistant|><think>
</think>` for GLM, the equivalent header for the other two), so the next token
is the answer rather than the opening of a reasoning block. Every option is
rendered identically and the prompt ends at `Answer:`; the only thing that
differs between options is the letter the readout looks at. A tokenizer with no
single-token spelling for a letter is refused by name rather than scored on a
two-token approximation.

**Cost, measured on the 320B checkpoint, one resident model:**

| request | prompt tokens | wall clock |
|---|---|---|
| one choice, 3 options | 118 | 66 s (cold) / 83 s |
| choice + score + noul | 331 | 206 s |
| one choice, 7 options (board) | 245 | 110 s |

The cost is the prefill and it scales with questions, not with options. Ten
questions would be ten prefills; that is why the limit is ten and why the
endpoint is for a handful of questions about one state. Sharing the state's KV
across questions would cut most of the 2nd..Nth prefill, which needs an engine
entry point that continues from a position instead of resetting — not done here,
and worth doing only once a caller actually asks more than two questions.

**What the numbers mean.** `probabilities` is the model's own next-token
distribution over the option letters; `confidence` is a concentration score
`(K·Σp² − 1)/(K − 1)`, 0 for uniform and 1 for a point mass. It is not
calibrated and it is not a probability of being correct — a flat distribution on
an undecidable question is the correct output, and the endpoint returns the whole
distribution so a caller can see that instead of only the argmax.

**Apps.** The tic-tac-toe opponent and the kart's obstacle generator now call
this endpoint instead of `/v1/chat/completions`: the board becomes a choice over
its empty cells, and the obstacle becomes a 12-way choice of kind and lane with
the distance derived locally from speed. Both keep their local fallback, so a
refused or slow decision degrades the opponent, not the game.

## 36. One endpoint, three prompt grammars: the decision adapter

SS35 shipped `/v1/systemone` with the family test inline in `jev.c`
(`is_ds`, `is_mm`, else GLM). That was enough to answer three questions about
one resident model and wrong the moment the requirement became "switch the
model and switch the LLM interface with it": the prompt grammar, the answer
spelling and the readable option set are all properties of the checkpoint, and
none of them was named anywhere. They are now a struct.

**`jev_adapter`** — `family` (glm / deepseek / minimax), `prompt` (which
encoder builds the decision turn), `letters` (how many of A..Z are one token),
`first_missing`, `space_lead`. The spec-derived half (`family`, `prompt`) is
computed from `model_type` alone; the tokenizer half needs `jev_probe()`.

**Where it is resolved.** Once at startup after the tokenizer loads, and again
after every `POST /v1/models/switch` — not cached in the endpoint, and not
derived once per process. A switch replaces `srv->eng[0]` and `srv->tok`, so a
request that arrives afterwards probes the new tokenizer and renders through the
new grammar; there is no path where a MiniMax prompt is read out of GLM logits.

**Measured letter tables** (`make jevprobe`, tokenizer only, no weights):

| checkpoint | vocab | prompt grammar | letters | spelling |
|---|---|---|---|---|
| GLM-5.3-Flash | 154,820 | `glm-chat-closed` | 26/26 | `" A"` |
| DeepSeek-V4.1-Flash-1017 | 128,000 | `deepseek-chat-closed` | 26/26 | `" A"` |
| MiniMax-M3.1-Flash | 200,000 | `minimax-chat-plain` | 26/26 | `" A"` |

All three spell every letter as one space-led token, so the readout itself is
portable and the only real difference is the turn's grammar. That is a
measurement, not an assumption: the tool prints the per-letter token count for
both spellings, and the endpoint refuses a checkpoint whose letters are not
single tokens instead of scoring a two-token approximation.

**Stale names are refused.** The request's `model` is checked against the
resident alias and `model_type` (case-insensitive) and anything else is a 400
naming the resident model. Before this, an app that cached `MiniMax-M3.1-Flash`
and then switched the server to GLM would get GLM's decisions under MiniMax's
id — the one failure a caller cannot see. `GET /v1/systemone` publishes the live
adapter (`family`, `prompt`, `letters`, `first_missing`) so a client can check
what it is talking to rather than infer it from a failed call.

**Apps.** Both hosted apps call only this endpoint: tic-tac-toe asks a choice
over the empty cells, the kart a 12-way kind x lane choice. Each keeps its local
fallback and retries once after a 400 whose text says the model is no longer
served, which is how a phone with a cached model id survives a switch made on
the desktop.

## 37. A per-layer release that was missing for one path, and what it cost

Measured while producing the endpoint matrix (SS36 companion, docs/PERF.md
2026-11-07): GLM's chat could not finish a 48-token reply. Footprint went 67 ->
117 -> 179 GiB over 24 decoded tokens while RSS *fell* from 50 to 35 GiB, and
jetsam killed the process. It reproduced with a 32 GiB pin cache, an 8 GiB pin
cache and `--backend cpu`, so it was not the cache and not the GPU.

**The increment was the tell.** +6 GiB per token is one decode step's expert
bytes (7.88 GiB). A leak of exactly one step per step means the buffers are
allocated per step and never given back, not that a cache is thrashing.

**Where.** `moe.c` has two ways to fetch a layer's experts: the batched paths,
which allocate a `cwick_fetch_req` array, fetch, GEMV, then call
`layer_fetch_release` + `free(reqs)`; and the single-token path, which did
everything except the last two calls. Every matrix the pin cache refuses is a
buffer the fetch owns (`prefill_blob_alloc` / `cwick_shmem_alloc`), so the
missing pair leaked one layer's staging bytes per layer, forever: 8 experts x 3
matrices x 45 layers x 8 MiB ~= 5.8 GiB per step.

**Why only GLM.** DeepSeek and MiniMax have their own model files and their own
fetch loops; this path is the GLM one. That is also why the leak survived: the
three-checkpoint matrix is the first measurement that ran a long GLM generation
after the route-first fetch was introduced.

**Why a short test misses it.** At 8 tokens the process already held 67 GiB and
still completed; the kill lands at 24-32 tokens. Anything that measures decode
on a streamed checkpoint has to generate at least 32 tokens and read
`cwick:process_footprint_bytes` — RSS moves the wrong way, because what grows is
compressed/page-cache accounting, not live heap.

**Fix.** Release the batch and free the array after the group GEMV returns (both
Metal waves wait before that point, the CPU path consumes the views in the loop
above it), and free the buffers the batch owns on the mid-read failure path too.
Re-measured with the same request: 48 tokens complete, footprint flat at 49.98
GiB with the default 32 GiB cache (25.94 GiB at 8 GiB — same 1.25 tok/s, so the
cache size is not what limits this checkpoint's decode), and the answer text is
coherent. `make test` passes.

The general rule this restates: in a fetch path, "owned" is a lifetime, and every
exit -- success, budget refusal, read error -- has to end it. The batched paths
had that rule written down; the single-token path was written later and did not.

## 38. Regression on the decision path: the same test, a different answer

SS37 fixed a per-step retention in the decode loop. `/v1/systemone` never
enters that loop, which sounds like a reason to leave it alone -- but "this path
does not take the code that leaked" is exactly the kind of claim that should be
measured once rather than assumed. So the decision endpoint got the same
long-run test, on two checkpoints whose prefill paths are different code:
GLM (whole-prompt prefill) and MiniMax (fixed 512-token chunks, the one place
where a chunk buffer could plausibly be retained).

**The test.** One process per checkpoint, no model switch. Several requests in
a row; each request carries 2-3 questions over one state, so each request is
2-3 prefills. State and question set change every round, so no round can be
answered out of the previous round's prefix cache. `cwick:process_footprint_bytes`
is sampled immediately before and after each request.

**The result.** GLM: 753 / 891 / 937 input tokens across three requests, footprint
15.46 -> 16.15 -> 16.14 -> 16.20 GiB. MiniMax: 817 / 1659 / 817 tokens, 22.11 ->
22.22 -> 22.23 GiB (the first sample, 0.12 GiB, predates the lazy mapping). The
first request pays once for pools and mapping; everything after it moves
+0.06 GiB (GLM) and +0.12 GiB (MiniMax) over roughly 2,000 tokens each. The
decode leak was +6 GiB per *step*. Three orders of magnitude apart, and the
increment does not grow with the request count -- GLM's third round ends below
its second.

**Two facts this pinned down.**

1. *A request does not reuse its own state's KV.* `input_tokens` is exactly
   questions x prompt (753 = 3 x 251, 1659 = 2 x 829). Every question is a full
   prefill of state + question, not a readout appended to a shared prefill. That
   is the cost model the endpoint shipped with, and it is now measured rather
   than asserted: three questions cost three prefills.
2. *The pin cache is irrelevant to a prefill.* The log lines read
   `expert cache 0.00 GiB pinned, 686.53 GiB streamed` -- 251 tokens route to
   essentially the whole expert set, so there is nothing worth keeping between
   steps. Decision latency is bounded by the device read rate, which is why the
   footprint is flat: this path has no cache that could grow.

**Determinism, for free.** The same GLM request run in two separate processes
returned identical probabilities to four decimals (0.9989 / 0.6459 / 0.9977), and
MiniMax's repeated round picked the same answers. That is the property SS31 was
about, now observed on the endpoint rather than on the cache: a changed decision
means a changed prompt or a changed model.

**What to reuse from this.** The regression is `tools/decision-longrun.sh`: it
starts its own server, sends N requests of M questions, and samples the footprint
metric around each one. It asserts nothing automatically -- the judgement is
"does the last sample differ from the first by more than the model's own working
set". That is deliberately loose: a threshold tight enough to catch 0.12 GiB
would also fire on allocator behaviour. The failure mode it does catch is the one
that actually happened, which was GiB per step and obvious on a plot.

## 39. A model library that lives in the server, not the browser

The requirement was a "模型" tab showing downloaded models, in-flight downloads,
catalog search, progress, resumable transfers and checkpoints of a few hundred
GiB, all landing under `./models`. The tempting implementation is a browser
that fetches the catalog directly and shells out to a script. Both were
rejected: the first puts a third-party origin (and CORS) between the console and
its own server, the second splits progress across two processes and two
languages. `src/hub.c` is C, runs in the server process, and owns one job table.

**Five control endpoints, no generation slot.** All `/v1/hub/*` routes are
answered before `slot_acquire`, next to `/v1/metrics` and `/v1/models`. A wedged
catalog GET or a multi-hour download cannot make a chat request wait. The UI
polls a state table; there is no SSE stream and no push channel, so the mutex
around that table is the entire concurrency story. libcurl's
`CURLOPT_NOSIGNAL` is set because these are real threads, and the progress
callback is the only place a worker touches shared state.

**One on-disk convention replaces a state database.** Every transfer writes
`<target>.incomplete` and renames only when the size matches the listing.
Pause, error, killed process and reboot are then the same state; the next run
reads the file size and sends `Range: bytes=<size>-`. `curl_off_t`/`int64_t`
everywhere is what makes a 100 GiB shard and a 300 GiB repository ordinary. If
a remote ignores `Range` and answers `200`, the write callback truncates the
partial file before appending, rather than producing a file with two copies.
A `LOW_SPEED_TIME` timeout turns a stalled transfer into a retryable `error`
job instead of a bar frozen at 93% — which is exactly what happened in the
end-to-end run below.

**Two-level layout, discovered on demand.** Downloads go to
`./models/<owner>/<name>/`, matching repository ids and avoiding short-name
collisions; a `.cwick-repo` marker records the full id. Startup scans both that
layout and a flat `./models/<name>` one into the ordinary model catalog. A
checkpoint that finishes *during* the process is registered lazily by
`POST /v1/models/switch`: the switch looks for a complete local directory when
the name is not yet in the catalog. That avoided adding a sixth "refresh"
endpoint and a second writer to the catalog. A directory with any
`.incomplete` file is never switchable.

**Measured** (one process, catalog host hf-mirror, 38.4 MB / 13-file test
repository): search answered in ~2 s; download reached 3/13 files before pause,
leaving a 51,408-byte `.incomplete`; resume continued from 0.14% without
re-fetching completed files; a stalled transfer became `error` after 60 s;
`retry` finished in 12 s with all 13 sizes matching the remote listing and zero
`.incomplete` files left. `/v1/hub/models` then reported both local checkpoints
as `complete:true`. The chat tab was left on `/v1/chat/completions` — the
library is a control-plane feature and does not change what generates tokens.

**Apps.** The market carries the two apps the goal names (tic-tac-toe, kart).
Both call `/v1/systemone` — that separation (chat uses chat, apps decide) is
asserted in `tests/test_cwick.c` on the compiled-in assets.

## 40. One settings tab, real trends, and no outbound endpoint

The goal calls this view "设置（控制台） tab" — one tab — but the console had
grown into two: metrics in **控制台**, configuration and model switching in
**设置**. The request was to merge them, add trend charts and richer model
information, and remove the option to point the console at a third-party API.

**Merge.** `consoleView` and its nav item are gone; the four metric cards, the
memory/cache meters, the model panel, the trend panel, the generation
preferences, authentication and the read-only run config all live in
`settingsView`. `VIEW_META` is five entries. The live-refresh checkbox still
drives the same 2 s scrape, and the rail's memory bar keeps its 5 s fallback,
so nothing about the metrics path changed — only where it is displayed. A
view that lets you *see* a number and the view that lets you *change* what
produced it no longer require a tab switch.

**Trends are deltas, not counters.** `/v1/metrics` publishes cumulative
counters; plotting `1/(sum/count)` draws a lifetime average, which is almost
flat and says nothing about a run. Each point is therefore the difference
between two scrapes: `dSteps/dSeconds` for decode speed, `dSum/dCount` for
TTFT, `dHits/(dHits+dMisses)` and `dQueries` windows for the two cache rates,
and the instantaneous footprint. A window with no decode steps is stored as
`null` and drawn as a break, not a false zero. 120 samples at 2 s is four
minutes, held in one in-memory object and drawn on four canvases with no chart
library. A hidden section has `clientWidth === 0`, so the canvas falls back to
its markup width and is redrawn on view change and resize. Clearing or pausing
the plot touches only the browser: the server's metrics are untouched.

**Model information.** The six headline cells stayed; every label from
`cwick:model_info` is now rendered underneath them, so a switch to a different
checkpoint family is visible without opening the Prometheus text. `/v1/config`
adds the model-library root and catalog host to the run-config grid.

**Removing the outbound endpoint.** The settings panel with API Base / key /
model, its test button and its persisted preferences were removed;
`apiBaseUrl()` is now a constant `location.origin + '/v1'`, `apiHeaders()`
carries only `Content-Type`, and `apiModel()` is the resident id. The two
hosted apps shared that same `cwick.prefs.v1` entry and would have gone on
sending `/v1/systemone` to a foreign host, so their copies of the same helper
were removed too: each now discovers the resident model from `/v1/models` and
re-discovers it after a switch, which is what the decision endpoint's
model-switch contract asks for. `loadPrefs()` deletes stale `apiBase` /
`apiKey` / `apiModel` keys from an existing localStorage entry, so a leftover
value cannot be picked up later by anything on the same origin.

**What was kept, deliberately.** `--api-key NAME` and `/v1/auth/keys` remain.
They are credentials for *other clients to call this cwick instance* — the
goal's "支持 API Key 方式，确保 cwick API 可以方便接入第三方 Agent 平台" —
which is the opposite direction from the console dialling out. The security
panel now says so explicitly. Chat still uses `/v1/chat/completions`, and the
apps still use `/v1/systemone`; neither endpoint changed.

**Guardrails.** `tests/test_cwick.c` asserts on the compiled-in assets: the
trend functions and canvases exist, no `consoleView`/`data-view="console"`
survives, no `setApi*`/`testApi`/`state.prefs.apiBase` survives, and neither
app bundle reads `cwick.prefs.v1`. That is a grep over what the server
actually serves, so the removal cannot quietly regress.

**Measured.** `make test` passes with five new assertions on the compiled-in
assets. A synthetic-checkpoint server was driven through CDP on the real clock
while eleven short generations ran over ten seconds: the plot collected six
samples, the incremental decode rate read 13,746.91 tok/s while the headline
card — a lifetime average — read 13,340.13 tok/s, and all four canvases had
drawn pixels. With no generation, the three rate charts show the idle state and
the footprint line stays continuous: the null-point rule behaves as designed.
The DOM also confirmed exactly five nav items, no third-party panel, all
`cwick:model_info` labels rendered, and `./models` reported as the library
root. The trend panel names its Prometheus series and links to `/v1/metrics`,
which is the "包括 prometheus" part of the goal's wording.

## 41. "Including prometheus" means the exporter, not a fixed projection

SS40 shipped four trend charts fed by `/v1/metrics` and a link to the raw
text. That is Prometheus *data*, but the goal's "趋势图（包括prometheus）" is
better read as: let the operator see what the exporter actually publishes, and
let them choose what to plot. A fixed projection silently decides that four
series are the interesting ones; anyone debugging a cache, a queue or a new
checkpoint has to leave the console to find out.

**What was added.** A "Prometheus 指标" panel in the merged settings tab. The
scrape parser now also keeps `# HELP` and `# TYPE`, so the panel lists every
numeric series with its type, its description and its live value, sortable by
name and filterable by substring. Clicking a row attaches a private canvas
(up to four) and a per-metric history of 120 samples. A counter can be plotted
as a raw value or as a per-second delta; the choice and the selected series are
a browser preference (`cwick.prom.v1`), not a server setting.

**Design calls.**

- *No Prometheus server, no TSDB.* cwick is the exporter; `/v1/metrics` is a
  snapshot of cumulative counters. Adding a scrape database or a query-range
  endpoint would duplicate what the browser can hold for four minutes and
  would put a long-lived writer on the generation host. The client keeps the
  short history and the server stays stateless.
- *One y-axis per series.* A byte counter, a ratio and a latency must never
  share a scale; drawing them on one chart would make the small ones flat.
  The two cache rates are the only pair that shares an axis, because both are
  percentages with a natural 0-100 domain.
- *Delta is opt-in, and only for counters.* A gauge plotted as a per-second
  difference is a bug (uptime turns into a constant ~1), and a counter plotted
  raw is a ramp. The checkbox derives only `# TYPE counter` series, with the
  usual `_total` / `_count` / `_sum` suffixes as a fallback; gauges keep their
  raw values and say so in the axis legend. Switching clears the history so
  two units never mix in one line. The four built-in charts keep their own
  fixed, meaningful units.
- *The same scrape clock.* Custom series are sampled inside `trendSample`,
  not on a second timer, so pausing "实时刷新" freezes every chart together
  and one `/v1/metrics` fetch serves both panels. `drawChart` gained an
  optional time array because a series selected later is younger than the
  session's first sample.

**Measured.** On the synthetic checkpoint the panel listed 65 exported series
with their HELP/TYPE; filtering on `cache` narrowed it to 8. Selecting
`cwick:decode_steps_total` and `cwick:uptime_seconds` produced two canvases
with independent axes and four samples each. The per-second toggle changed
only the counter's legend and value (329 raw steps -> ~47 steps/s); the
uptime gauge stayed `原始值` with its raw value. The selection and the toggle
survive a reload in `cwick.prom.v1`. `make test` passes with two more
assertions over the compiled-in assets. No engine, endpoint or model path
changed.

## 42. A decision prefill finally shows up in stdout

`/v1/systemone` never enters `generate_impl`: `jev_decide` calls
`cwick_engine_prefill_logits` once per question, which used to be a bare
forward pass. The chat path prints a prefill start, token/layer heartbeats,
timing, and drives `cwick:prefill_*`; the decision path printed none of it,
and the server never bound a request tag for it, so the few engine lines that
did appear said `gen`. A multi-question decision could stream tens of GiB
between two access-log lines with nothing in stdout.

**Root cause, in one sentence:** the observability lived in the generation
loop, not in the prefill entry point. Moving it down one level fixes every
caller at once.

**What changed.** `cwick_engine_prefill_logits` was restructured around one
common tail (the three backends used to `return` from their own branch). When
a request tag is bound, it now publishes `cwick_metrics_set_prefill`, prints a
start line, opens the per-layer heartbeat for the DeepSeek and MiniMax
backends (their heartbeat is gated on a "prefill in flight" flag that only
`ds_engine_generate` / `mm_engine_generate` used to set), then prints a timing
line with tok/s. GLM already had `maybe_progress_at` inside `engine_prefill`.
`server.c` binds and clears `eng->req_tag` around `jev_decide`, which also
makes DeepSeek/MiniMax internal lines carry the request id — their `ctx.tag`
is a pointer to that same buffer.

**The quiet-by-default rule.** `logitdump` parses stdout and the test suite
calls the same function for numeric A/B comparisons. An unconditional INFO
line there would corrupt both. The engine therefore reports only when
`eng->req_tag[0]` is non-empty, which is exactly the condition "the server
bound this prefill to an HTTP request". No new API, no flag to forget.

**Request and execution logs.** `jev_decide` prints one line at receipt
(questions, body bytes, resident model, adapter, `stream=false thinking=off`
— the decision contract), one `question i/N` line before each prefill, one
line per readout with the winning key, its probability, the runner-up and the
concentration, and a summary with input tokens, wall time and prefill tok/s.
`--trace` adds each rendered prompt and the finished decision object.
`server.c` prints `systemone: decision refused — <reason>` for every early
exit, so a 400 has a line of its own. All of it is `CWICK_INFO`/`CWICK_TRAC`,
so every line carries the wall-clock time and `[req N]` as the goal requires;
`log.c` writes and flushes stdout.

**Measured.** A two-question decision on the synthetic checkpoint produced
`decision: 2 question(s) ...`, two `question i/2` lines, two prefill starts,
55 and 54 token-progress heartbeats, two `prefill done` lines (466 tokens /
37 ms and 432 tokens / 30 ms), two readout lines (`-> a p=0.3368
(runner-up 0.3325)` and `-> x p=0.5046`), `decision done: 2 question(s), 898
input tokens, 67 ms wall`, and the 200 access-log line — all stamped
`[req 1]`. An empty question set produced `[req 2] systemone: decision
refused — body.questions is empty` and a 400. `make test` still prints zero
`[systemone]` lines, confirming the CLI path stayed quiet.

## 43. The two apps now ask the questions the goal actually names

Both apps already had the right plumbing: a real Mobius strip, one
`/v1/systemone` prefill per decision, per-app sqlite, a right-hand column, and
no generated objects to parse. What was wrong was the *wording of the
questions* — small in code, large in meaning.

**Tic-tac-toe.** The machine's instructions were
`Win if you can, otherwise block X, otherwise prefer the centre, then a corner,
then a side.` That is a complete local strategy handed to the model as a
question; the prefill was reduced to executing a rule the browser already had.
The goal asks for the cell with the *greatest probability of winning*. Now the
app computes board **facts** (`cellFacts`: immediate win, forced block, number
of winning replies, open lines) for every empty cell, puts them in the state
and in each option's description, and asks which cell has the highest win
probability. No ranking, no recommendation: the checkpoint's softmax over the
option letters is the decision and its argmax is the move. `fallbackMove()`
survives, but only on a failed request. The sidebar draws one bar per empty
cell, highlights the played cell and also shows the human's tactical facts, so
"人与机器的对弈决策过程" has both sides. The heading states that the bars are
one prefill's option distribution, **not** a simulated win rate — a claim the
endpoint cannot make.

**Kart.** The instructions literally said `leave room for the player to react`,
which contradicts "确保跑圈难度最大". They now ask for the obstacle that makes
the lap hardest, and the state carries hits, spawned count, the difficulty
level, which side of the strip is up and a pressure point. "实时随机决策" was
the subtler gap: the readout is deterministic for a fixed prompt, and a fixed
A..L order puts the model's letter bias on the same obstacle forever. Each
request now Fisher-Yates shuffles the twelve options (the server maps ordered
`criteria` members to letters), draws a fresh variation seed and one of six
pressure points. **The variation lives in the question, never on the road** —
there is still no local obstacle generator, and every obstacle remains a model
prefill. The seed, focus and the distribution's normalised entropy are shown in
the sidebar and stored in the sqlite `agent`/`spawn` events, so the randomness
is auditable. Spacing still comes from speed (a "hardest" obstacle you cannot
see is an unfair collision), but tightens with the lap above a 10 m floor, so
the difficulty target is actually felt rather than just written in the prompt.

**Measured.** On a synthetic checkpoint (context raised 512 -> 4096 so a
656-token 12-option decision fits), a real browser played one human move:
the request carried 8 options with tactical facts, the machine answered, and
the sidebar rendered 1 probability block with 8 rows, 1 highlighted, plus the
human and machine rows. Fourteen seconds of racing produced 7 decisions with
**7 distinct seeds, 7 distinct option orders and 4 distinct pressure points**,
12 options each, 0 failures, 127 ms mean response, 5 obstacles placed, and the
sidebar showing the seed and the decision spread. `make test` passes with 11
new assertions over the compiled-in bundles, so the prompt wording, the
shuffle, the sqlite declaration and both sidebars cannot silently regress.

As before, `goal.md` was read but not edited, and the chat tab still uses
`/v1/chat/completions`.

## 44. Decision prefill: measured, then optimised only where it is exact

`/v1/systemone` prefills once per question. On a 60-layer 128-expert checkpoint
that is 1.7 minutes per question, so the endpoint's cost is almost entirely
prefill — and a request that asks three questions paid for the shared state
three times.

**Where the time actually goes.** Per-token profiling on MiniMax at 4096
context: a full 512-token block costs 173 ms/token (`experts 82, io 28, attn 26,
proj 20, wo 8, route 6`), while a 9-token trailing block costs 606 ms/token
(`io 220, experts 276`). Two facts follow, and both were counter-intuitive:

1. The cost is **per block at the expert read**, so smaller blocks are worse.
   An even 275+275 split of a 550-token prompt measured **113.6 s against
   109.3 s** for 512+38. The fair-looking split was 4% slower because it halved
   the block that amortises the expert read. That variant is still behind
   `CWICK_JEV_CHUNK_EVEN=1`, but it is off.
2. Raising the ceiling is what pays: `MM_CHUNK`/`DS_CHUNK` went 512 -> 1024, so
   the common 500-1000-token decision prompt fits in one block. MiniMax single
   question, 550 tokens: **109.3 -> 95.5 s (1.14x)**. The activation buffers
   cost ~1.6 GiB more (footprint 23.8 GB, swap 1.36/3.0 GiB on this machine).

**Shared prefix, per question.** All questions in a request begin with the same
`system + <state>` turn. `jev_decide` now encodes every question first, takes
the longest common *token* prefix, prefills it once, and resumes at it for each
question. A three-question request sharing a 348-token prefix (1145 input
tokens) prefillled 449 tokens instead of 1145: **213.8 -> 102.0 s (2.10x)**, with
every probability identical to the non-reusing run (max |dp| = 0.0).

**The part that had to be thrown away.** Resuming by moving `eng->pos` and
rewriting from the prefix is only valid if the cross-token state is addressed by
position. It is not, on two of the three checkpoints:

| backend | force reuse | result |
|---|---|---|
| MiniMax (GQA + KV) | in-place resume | **|dp| = 0.0**, choices identical |
| DeepSeek (compressed KV + indexer pool) | in-place resume | **|dp| = 0.56**, 2 of 3 answers changed |
| GLM (DeltaNet + DSA pools) | in-place resume | **|dp| = 0.41**, 1 of 3 changed |
| GLM | snapshot + apply (`pfx`) | choices correct, **|dp| = 0.11** |

So the endpoint enables reuse only on MiniMax and keeps N full prefills on the
other two (`CWICK_JEV_REUSE_ALL=1` measures a backend that has been fixed;
`CWICK_JEV_NO_PREFIX_REUSE=1` forces the old path anywhere). The snapshot route
looked like the general answer — `pfx.h` says outright that these states cannot
be *rewound* but can be snapshotted whole, and `test_prefix_cache_equiv` proves
it for generation — but on the real GLM the restored DSA pool left a partially
filled tail behind, which is enough to move a probability by 0.11. A decision
endpoint cannot ship a 2x saving that occasionally changes the decision.

**What this means outside the endpoint.** For a single-question request on GLM
or DeepSeek there is no algorithmic win here: a 250-token prefill already routes
to essentially the whole expert set (GLM streamed 686.53 GiB for 251 tokens), so
the cost is "read the model once" and token count barely moves it. Reuse is the
only lever, and it needs a backend whose apply is exact — which is the next
piece of work, not a tuning parameter.

**Interface.** `cwick_engine_prefill_logits_at(eng, ids, n, start_pos, logits,
err)`: `start_pos == 0` resets, `> 0` continues from an existing prefix. It is
the same code path on all three backends (GLM derives absolute positions from
`eng->pos`; the other two take `pos` as an argument), which is why the
exactness question is about state, not about plumbing.
`test_prefill_resume_equiv` checks the split against one pass at two cut points
(mid-prompt and one-token-from-end) on both synthetic checkpoints: rel = 0.0.

## 45. A second dialect, not a second engine: `POST /v1/responses`

Coding agents increasingly speak the Responses API rather than chat/completions, and a
server that answers only chat forces every such client through an adapter it has to
maintain. The obvious implementation is a second request path; that would be a second
place for the prompt template, the queue, the prefix cache and the thinking budget to
drift apart.

**The adapter ends at the conversation.** `responses_parse` walks `input` items and builds
an OpenAI `messages` array — `message` (with `developer` folded into `system`), consecutive
`function_call` items merged into one assistant turn, `function_call_output` into a `tool`
turn, `reasoning` dropped because a hidden chain is not replayable input. It then
serialises that array with the request's scalar fields into a chat-shaped body and calls
`validate_body`, the function `/v1/chat/completions` already uses. Everything after that
point — model identity, sampling ranges, tool declarations, tokenisation, prefix cache,
generation, access log — is the same code. `gr->api` only selects the envelope.

**What the envelope costs.** The response object needs a different usage spelling
(`input_tokens` / `output_tokens` / `output_tokens_details.reasoning_tokens`), output items
instead of `choices`, and named SSE events instead of bare `data:` frames. `responses_body`
is built once and used by both the buffered body and the terminal `response.completed`
frame, so the two cannot disagree. Reasoning is published only when the client asked for
`reasoning.summary`; a turn with no tool call always carries a message item, because its
absence means "no answer" rather than "empty answer".

**What is refused, and why.** A durable response store does not exist here, so a non-null
`previous_response_id` or `conversation` is a 400 that names the field and says to replay
the input items — accepting it would silently truncate the prompt. `store:true`,
`background:true`, `truncation:"auto"`, `top_logprobs`, `max_tool_calls`, non-`function`
tools, image/file parts and `tool_choice:"required"` are refused for the same reason every
other unsupported field is: a server that pretends is worse than one that says no. Fields
that are inert here (`metadata`, `include`, `parallel_tool_calls`, `service_tier`,
`prompt_cache_key`) are accepted and ignored, and the documentation says so.

**Verification.** Ten assertions in `test_server` cover the buffered envelope, the named
stream ending in `response.completed` with no `[DONE]`, the item round-trip (developer role,
function_call + function_call_output), flat tool translation, and six refusals. They run
against the same synthetic checkpoint as the chat tests, so the equivalence claim is tested
rather than asserted.

**The console had to pick a side, and picks neither.** Being able to answer both dialects is
worth little if the only client you ship cannot be exercised in both, so the composer's
`接口` select chooses the endpoint per turn. The translation is one-way and lives in
`web/app.js`: `respInput` turns a session into items, `respEventToChat` turns each named
event into a chat chunk, and `postChat` hands everything above it the chat shape. That is why
the switch cost no new bubble, tool card or stats path — the alternative (a second renderer)
would have drifted from the first the moment either changed. Two details follow from the
dialect rather than from taste: the request carries `reasoning.summary:"auto"` so the thinking
panel has something to show (the engine reports its own block as the summary), and a tool
call can only be drawn when its item completes, because there are no argument deltas to
render.


## 46. A third dialect: `POST /v1/messages`

The project already served chat/completions and Responses. Anthropic's Messages API is the
one a coding agent sends when it is configured for Claude Code, and a server that speaks
only the OpenAI pair forces that client through a proxy. The implementation question was
the same one §45 answered, and so is the answer: **adapt at the conversation boundary, not
at the engine**.

**What the adapter has to reconcile.** Three things are structurally different, not just
spelled differently. `system` is a top-level field rather than a message, so it is prepended
to the conversation as the system turn. Content is an array of typed blocks, so a turn can
carry prose *and* calls, and a result is a `tool_result` block inside the next user turn
rather than a message with its own role — several results in one turn stay in order, and
the prose after them becomes the user message that follows. And the stop vocabulary is
`end_turn` / `max_tokens` / `tool_use`, with the count reported in `message_delta` instead
of a usage frame. None of that touches the prompt template: `anthropic_parse` rebuilds the
chat body (including the flat-tool → nested-tool translation) and calls the same
`validate_body`, so model identity, sampling ranges, tool declaration checks and thinking
levels have exactly one implementation across all three dialects.

**Verification that the adapter is really an adapter.** The strongest available check is
that the same request must price identically. A tool declaration that renders a 640-token
prompt through chat/completions renders exactly 640 through `/v1/messages` — measured, not
argued. The suite adds 19 assertions for the envelope, the block conversion (system blocks,
`tool_use`, `tool_result`, a dropped `thinking` block) and five refusals.

**What is refused, and why.** `stop_sequences` (the engine stops on its own delimiters; a
client-supplied string would have to be matched post hoc, which is a different feature),
image/document/audio/video blocks (text-only server, and a silently dropped attachment is a
wrong prompt), `tool_choice:{"type":"any"}` (the same constrained-decoding requirement as
OpenAI's `required`), and a `tool_result` in an assistant turn or `tool_use` in a user turn
(the pairing is wrong; guessing would send the wrong prompt). `thinking.signature` is emitted
empty because there is no signing key and no chain is ever verified — which is also why a
`thinking` block in the request is dropped, exactly as Responses drops a `reasoning` item.

**The console gained the third option, and it is not a third renderer.** `msgMessages` /
`msgEventToChat` / `msgFullToChat` map the dialect onto the same chat-shaped event the other
two produce, so the bubbles, the tool loop and the stats line were untouched. One thing came
out better than Responses: this dialect *does* stream a call's arguments
(`input_json_delta`), so the tool card fills in while the model writes them, where Responses
can only announce the call once the item is complete.

## 47. Auditing the three dialects against their reference docs

The two adapters were written from the shape of the protocols, not from a field-by-field
reading of them, so the question "where does this differ from the reference" deserves a
measurement rather than an opinion. Checking the envelopes against the published schemas
(the OpenAI Responses event models, the Anthropic `Message` / `RawMessageDeltaEvent`
types) turned up six places where we emitted a *legal-looking* document that was not the
document the schema describes. All six were fixed; the refusals stayed refusals.

**What was simply missing.** Every Responses event must carry a monotonic
`sequence_number`; clients use it to notice a dropped frame, and the field has no default.
`response.output_text.delta` and `.done` must carry `logprobs` (empty here — this server
never computes them). The `response` object must carry `tools`, `tool_choice` and
`parallel_tool_calls` (not optional) plus the rest of the request echo, and `error` /
`incomplete_details`. `message_delta` repeats `input_tokens` alongside `output_tokens`, so
a client that only reads the closing event still has complete usage. None of these change
what the engine does; they change whether the reply is the object the client thinks it is.

**What was actively wrong.** Two things. `status` was always `"completed"`, even when the
turn stopped because the budget ran out — so a truncated answer was indistinguishable from
a finished one; it is now `"incomplete"` with
`incomplete_details.reason = "max_output_tokens"`, and the stream ends at
`response.incomplete` (or `response.failed` when a response exists but could not finish,
where it used to send a bare `error` event). And the item ids in the terminal frame were
minted independently of the ones `output_item.added` announced, so `response.completed`
named a different message item than the stream had just described; ids are now derived
from the response id (`…-msg`, `…-rs`, `…-fc<i>`).

**One asymmetry removed.** With no `reasoning.summary`, the buffered body omitted the
reasoning item while the stream announced it and streamed `reasoning_text.delta` — the two
paths disagreed about the output list. The item now always exists when the turn reasoned;
its `summary` is populated only when asked for, which keeps the chain out of the body
without changing the shape.

**Chat needed one fix of the same kind.** `developer` is OpenAI's current spelling of the
instruction turn. The GLM template's role chain ends in a `user` branch, so a `developer`
message silently became a *user request*, and the DeepSeek renderer rejected the unknown
role outright. Both now treat it as the instruction turn.

**What stayed a refusal.** The durable half of the Responses API (`previous_response_id`,
`conversation`, `store`, `background`), `truncation:"auto"`, `top_logprobs`,
`max_tool_calls`, non-function tools, image/file input, `tool_choice:"required"` and
Anthropic's `{"type":"any"}`. `parallel_tool_calls` is echoed but not enforced (the
checkpoint decides how many calls it emits), `text.format` is text-only, and
`/v1/messages/count_tokens` does not exist — there is no entry point that counts a prompt
without rendering it. Those are listed in `docs/SERVER.md` under each dialect rather than
left for a client to discover.

**Coverage.** 613 assertions, twelve of them new: the echoed response fields, the
sequence-number count matching the event count, `logprobs` on every `output_text` event,
the item id in `response.completed` matching the one the stream announced, the incomplete
status, the flat tool declaration being echoed back, `input_tokens` in `message_delta`, and
the `developer` role on chat.

## 48. The fourth dialect, audited against its reference

`/v1/systemone` was written from the shape of the protocol too, so it got the same treatment as
§47: read the published document field by field and ask where this differs. It differed in two
places that matter and four that did not, and the two that mattered were both silent.

**Two bugs.** `state` accepted only a string. The reference takes a string or an object and
serializes the object; every example in the published document sends an object, so the endpoint
refused the canonical request shape. It now serializes any non-string JSON value into `<state>`.

`noul`'s criteria were parsed as a generic option list. The two sides became options named
`true` and `false` in whatever order the JSON members arrived, their descriptions were dropped
from the prompt, and the answer was the probability of whichever side came first. A client that
wrote `{"false": ..., "true": ...}` got P(false) reported under the key `noul`: a wrong number
in a correct-looking shape, which is the worst kind. Both sides now have a fixed order -- the
true side is always A -- and their text is what the prompt shows.

**What was missing.** `usage.cached_input_tokens`, the reference's field for state tokens served
from the cache; `reused_tokens` was our name for the same number, and both are now reported. And
the status split: a body that is not a decision request (malformed JSON, a missing field, an
invalid question) is a 422, a well-formed request this server will not serve is a 400. Bodies
over the request cap answer 413 on every endpoint now instead of 400.

**What is refused instead of ignored.** `stream:true`, `reasoning_effort` other than `"none"`,
a `thinking` block that is not off, and `images`. All four used to be accepted and silently
narrowed to one JSON object with no decode; a client that asked for a stream and got a single
body learned nothing from the reply.

**What was tightened to the published limits.** `score` levels are 2 to 10 (a one-level score
used to be accepted and would have returned 0 forever), state 4 MiB (was 32 KiB), question text
64 KiB (was 4 KiB), and `instructions` is required per question -- the old fallback text ("Which
option applies?") was a silent substitution of the question being asked.

**What stays different, on purpose.** Ten questions and 26 options against the reference's 256
and 208: the readout is one letter token per option and a tokenizer has 26 of them, and the goal
is one state plus a handful of questions. `model` is the resident alias rather than a deployment
path, `temperature` and the `object`/`adapter` keys are extensions, and the OpenAI-shaped
`/v1/decisions` is not implemented. The endpoint does answer on the published path
(`/inference/v1/systemone`), so a client whose base URL ends in `/inference` needs no change.

**The other three, re-checked against a live server.** The same audit was repeated end to end on
all four endpoints with a synthetic checkpoint: `sequence_number` present and strictly increasing
with no `[DONE]` in Responses, the terminal response object carrying the whole request echo plus
`error`/`incomplete_details`, `message_start`/`message_delta`/`message_stop` with `input_tokens`
in the delta, `stop_reason` `max_tokens` when the budget ran out, chat emitting a usage chunk
under `stream_options.include_usage` and the `[DONE]` sentinel, and 413/422/400 coming back
where the references say they should. The gaps recorded in §47 are unchanged and still documented
per dialect (chat omits the optional `system_fingerprint`/`service_tier`; no `count_tokens`; no
`ping` events).

**Coverage.** 626 assertions, thirteen of them new: the two refusal classes, a JSON object state,
the four unsupported fields (and the two accepted "off" spellings), the required `instructions`,
the score level range, and `noul` criteria in object and array form.

## 49. A downloaded model is not necessarily a loadable model

The 11-10 serve log ended with five `POST /v1/models/switch` calls to a freshly
downloaded 18 GiB checkpoint, every one a 500 `config is missing/invalid
pad_token_id`, followed by a server answering 503 `no_model` to every chat
request. The download itself was fine (19 files, no partials). Three separate
things were wrong.

**The diagnosis was misleading.** `cwick_model_spec_load()` unwraps
`text_config` and reads every field from the inner object. The outer object had
`pad_token_id=248055`, the inner one had `null`; `model_type=qwen3_5_text`
matched neither specialised branch, so it was measured against the default
geometry contract and died on an unrelated GLM-only field. The real reason — a
dense, mixed linear-attention, vision checkpoint with no kernels in this build
— never appeared in the error. `q_lora_rank`, `kv_lora_rank` and the MoE fields
are missing too, so editing the config could never have worked.

**The swap had the wrong order.** `srv_switch_model()` released the engines,
tokenizer, Metal arena and all 27 prefix snapshots *before* trying to load the
replacement. A failure therefore left the instance with no model, and each
retry logged `releasing (none)`. The first attempt had already unmapped 48
shards (475 GiB of address space). Metrics made it worse: `cwick:model_info`
is a static info metric updated only on the success path, so a scraper kept
seeing the old model name while every chat request returned 503.

**The inventory lied by omission.** Directory discovery only checked for
`config.json` plus a weight file, and the library's `complete` flag only meant
"all files arrived". Any repository could be downloaded, appear in the Web
model tab with a switch button, and then take the server down.

**Four changes.** (1) One shared type table declares the architectures with
kernels; `cwick_model_spec_load()` now refuses an unknown `model_type` by name
with `CWICK_ERR_UNSUPPORTED` instead of falling through to the default parser.
(2) Every `/v1/models` and `/v1/hub/models` entry carries `supported` and,
when false, `unsupported_reason`. The verdict is computed once at discovery
and cached in the catalog entry, so a polled model list does no file I/O while
holding the generation lock — control endpoints and generation stay concurrent.
The Web console disables those switch buttons. (3) A switch runs a read-only
`cwick_model_spec_load()` dry run on the target config *first*; an unsupported
architecture or a broken contract answers 400 and leaves the resident engine,
tokenizer, arena and prefix cache untouched. (4) `cwick:model_loaded` is a
live 1/0 gauge, set to 0 when a post-teardown load fails and back to 1 on a
successful switch; `model_info` keeps its labels because renaming or
relabelling an exported metric breaks scrapers.

**The list is deliberately not closed.** The type table is the registry of
*implemented kernels*, not a promise that only three models can ever run:
adding an architecture means adding its kernels and one line to the table. The
library still downloads any repository and still lists it; only the "make it
resident" action is gated.

**What is still not rolled back.** If the config passes but a weight tensor is
missing or the tokenizer cannot load, the old checkpoint has already been
released. Rebuilding it would mean a second full load inside the failure path,
so the instance still answers 503 `no_model` — but now with `model_loaded=0`
and an explicit 500 detail, and one successful switch restores service.

**Coverage.** 639 assertions, thirteen of them new. Against a live server with
one resident synthetic checkpoint, one supported spare and two unsupported
entries: both unsupported switches answered 400 in 0.1 ms with no
`switching to` or `released` log line, `/v1/models` kept `loaded:true`, chat
stayed 200 and `model_loaded` stayed 1; the supported switch returned 200 and
republished the identity; a checkpoint with deliberately missing weights
returned 500, `model_loaded=0` and 503 on chat; switching back returned 200
and 1. The compiled-in Web asset matched `web/app.js` byte for byte.

## 50. Deleting a downloaded checkpoint

The model tab could fetch a repository but not remove one: a wrong-architecture
checkpoint of tens of GiB stayed on disk until someone opened a terminal.
`DELETE /v1/hub/models/<name>` closes that loop.

**Resolution and removal are separate calls.** `cwick_hub_model_dir()` turns a
user-supplied name into a directory; `cwick_hub_delete()` removes it. The server
runs between them, so it can refuse the resident checkpoint (409) without
teaching the library about the engine, and the switch path uses the same
resolver — "what is a local model" has one definition. A partial download
resolves for cleanup but not for switching, because those two questions really
are different.

**The boundary is realpath, not string matching.** A name is refused if it
contains `..`, a backslash, an absolute path, or an empty or `.` component; the
resolved directory must sit strictly inside the library root after symlinks are
resolved, and the root itself is rejected. Removal uses `lstat` and only
descends into real directories, so a symlink is unlinked rather than followed.
Recursion is capped at depth 8.

**One bug the tests found.** `"."` passed the character test (no `..` in it),
resolved to `<root>/.`, correctly failed the root check — and then fell through
to the owner lookup, which turned it into `./models/<owner>`. Deleting `"."`
would have removed an entire owner tree. The fix is a component check that also
refuses empty components, so `./name`, `owner/` and `a//b` are invalid names
rather than paths to normalise.

**What is refused.** The resident checkpoint (409): its weights are mapped by
the running engine. A directory a worker is still writing (409): the user is
told to cancel the job, which is what stops the thread. Both checks run before
anything is removed.

**What stays consistent afterwards.** The catalog entry is dropped in the same
request; otherwise `/v1/models` would keep advertising a root that no longer
exists and every switch to it would answer 400. Finished jobs pointing at the
removed directory are marked `canceled` with "local directory deleted", so the
UI does not keep claiming a completed download whose files are gone.

**What is deliberately not offered.** No trash or soft delete: a checkpoint is
tens of GiB and a copy is not affordable, so the confirmation lives in the Web
console (file count and size spelled out) and the server performs an
irreversible removal. No background deletion job either: unlinking a directory
tree is metadata work, and the request returns the freed byte count when it is
done.

**Coverage.** 651 assertions, twelve of them new: resolution of owner/name,
bare and flat names; refusal of `..`, absolute paths and `.`; a successful
delete that reports two files and leaves no directory; 404 for a missing name;
404 for a traversing name. Against a live server: three layouts deleted with
200 and the right byte count, `/v1/hub/models` and `/v1/models` both dropping
them, the resident checkpoint refused with 409, and `..` / absolute paths 404.
