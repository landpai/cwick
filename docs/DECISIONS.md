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

## 51. The checkpoint that answered "4" only after five wrong turns

The fourth entry in the goal's model list is Qwen3.5-9B: 32 layers, of which 24
have no softmax attention at all. They carry a Gated DeltaNet -- a recurrent
linear attention that keeps a [128,128] state per value head instead of a KV
cache -- and the remaining 8 are ordinary GQA with three twists: a per-head
RMSNorm on Q and K, a partial rotation covering a quarter of each head, and an
output gate that doubles the width of `q_proj`. All BF16, 18 GiB, no experts,
so nothing streams and nothing needs a cache budget.

**The port ran and produced garbage for five rounds.** Every round looked
plausible in the numbers and useless in the text: endless spaces, then endless
newlines, then punctuation. The way out was not more staring at the code. It
was a second implementation in numpy that reads the same shards and prints the
same statistics, and then comparing, layer by layer, against the reference C.

That comparison is what made the problem visible. Layer 0's post-mixer mean,
absolute maximum and first four values matched **to the last printed digit**
(`mean=0.00455 absmax=13.636 x[0..3]=0.1058 0.2135 0.0251 -0.0635`), and so did
layers 1-3 and the final logit ranking (`220(11.874)` against `220(11.877)`).
Two independent implementations agreeing that closely does not mean the
port is right. It means the port faithfully implements *the same wrong idea*,
and that the idea -- not the code -- is what has to be fixed. From there the
route was the published model source, read line by line.

**Five separate mistakes, all silent:**

1. *The convolution read the wrong row.* `src` is an absolute position inside
the block, but the code added it to a per-position base, so position t read
block position t+src. It looks right until the block is shorter than the
window demands: the first runs read uninitialised memory past the end and
conv output went to zero from the eighth token on. That is what turned the
first diagnosis from "the model is broken" into "the model is fine, the
convolution is off by a position".

2. *Two different norms live in one file.* Every RMSNorm in this checkpoint
stores its weight as an offset from one (`Qwen3NextRMSNorm`, initialised at 0,
multiplier `1 + w`) -- layer 0's input norm averages 0.033, not 1. But the
gated norm inside the DeltaNet stores a plain multiplier (`RMSNormGated`,
initialised at 1). Applying `1 + w` to both, or to neither, breaks the model
in two different ways; the tell is that one family of weights sits near zero
and the other near one.

3. *The rotation was the wrong shape.* RoPE here is `rotate_half`: the two
halves of the rotated slice swap, and index i pairs with i + rotary/2. The
port used adjacent pairs (2i, 2i+1), the GPT-J layout. Both are rotations, both
produce numbers in a sane range, and they disagree on every position. This was
the single largest error.

4. *The recurrence decayed at the wrong moment.* The defined step is
`S = decay*S` first, then `delta = beta*(v - S^T k)` **reading the decayed
state**, then `S += k (x) delta`. Decaying after the read leaves every
position's correction scaled by a decay it should not have.

5. *`q_proj` is interleaved per head.* The output is `n_heads x (q | gate)`,
because the reference views it as `(..., n_heads, 2*head_dim)` and splits that
last axis. Reading the first half of the whole matrix as q mixes one head's
gate into the next head's query -- and the numbers stay in range, so nothing
complains.

**What settled each one.** (2) came from the weight statistics: a norm
averaging 0.033 cannot be a multiplier. (1), (3), (4) came from reading the
published implementation after the numpy cross-check had proved the code was
doing exactly what it was written to do. (5) came from the same reading, and
is the reason the answer to "2+2" changed from a run of "2"s to `4`.

**Where it ends up.** With all five fixed the checkpoint answers correctly in
both languages, closes its turn on `<|im_end|>`, and streams: `What is 2+2?`
returns `4` and stops; `用一句中文介绍你自己` returns a coherent Chinese
sentence; `17 times 23` produces a 7-step reasoning block ending `391` and the
answer `391`. Decode measured **4.85 tok/s** over 124 tokens -- 18 GiB of
weights per step against a measured 123 GB/s ceiling is 6.8 tok/s, so the
kernel is at ~71% of what the memory bus allows. Prefill is layer-major over
512-position blocks, the recurrence walking positions inside a block because a
scan cannot be widened, and its state is 50 MiB for all 24 layers against a KV
cache that would grow without bound.

**Not implemented, and why.** The vision tower (27 layers, `model.visual.*`)
and the MTP draft head (`mtp.*`). Neither is reachable from a text prompt:
images would need `image_token_id` positions expanded into patches, and MTP
only pays off for a speculative decoder this build does not run. The bytes stay
on disk and are never bound. Also absent: `mrope`'s multimodal position
allocation, which for text collapses to ordinary RoPE because all three
position streams are equal.

**Coverage.** The architecture is admitted by the same type table as the other
three (`qwen3_5`, `qwen3_5_text`), so the catalog marks it supported and the
switch dry-run lets it through. The regression suite keeps its "unknown
architecture is refused by name" cases by using a name that does not exist
(`mamba9`) instead of this one.

## 52. 18 GiB fits, so it lives in core — and that is what exposed two serial phases

Qwen3.5-9B is the first checkpoint in this tree that is *smaller than the
machine*. There are no routed experts, so nothing streams and there is no cache
to budget: every one of the 18 GiB is read by every position. That turns "keep
the weights in RAM" from a preference into a size question with an answer.

**The policy.** `q35_resident_wanted(weights, state, host, ceiling)` says yes
while weights + state fit in half of RAM and under the configured ceiling with
4 GiB to spare. Half, not all: wired pages cannot be reclaimed, so a checkpoint
that wants more than half the machine leaves nothing for a long context's KV,
for the other checkpoints this server can switch to, or for the OS. An unknown
host size is a no -- the point of the check is to avoid a decision made without
the measurement. `CWICK_Q35_RESIDENT=0|1` overrides either way.

**The mechanism.** `cwick_safetensors_make_resident(saf, prefix, wire)` collects
the byte range of every tensor under a name prefix, sorts them, merges the
touching runs (a layer's matrices sit back to back inside a shard, so a few
hundred `mlock` calls collapse into a few dozen), then for each run:
`MADV_WILLNEED` to queue the readahead, `mlock` to fault in and wire in one
syscall, and — only when the wiring is refused — a parallel page-touch pass
through the pool. Finally `MADV_NORMAL`, because the `MADV_SEQUENTIAL` installed
at mmap time says "read this once, front to back", which is false for a mapping
that lives here and would let the kernel drop pages behind the read pointer.
Release is symmetric and happens in `cwick_safetensors_close` before `munmap`
(a wired range that is only unmapped later leaves the accounting claiming bytes
that no longer exist). The KV and DeltaNet state is wired too — that is
anonymous memory, which the VM would rather compress than page, and a compressed
KV page becomes a stall inside the attention call instead of at its edges.

**What it buys.** 17.98 GiB faulted in and wired in **2.6 s** (7.5 GB/s) at load
time, paid once. On a 44-token prompt prefill drops from **2370 to 1569 ms**
(1.5x) because the first block no longer takes 1.2 M minor faults on the
request's critical path; on a 760-token prompt it is 13415 to 12207 ms (1.1x),
the rest being real work. Decode does not move: 18 GiB per step against a
measured 123 GB/s ceiling is 6.8 tok/s, and the kernels were already at 92% of
the bus at 7.0 tok/s. A single short request pays the 2.6 s back; a server
pay it once.

**What it exposed.** With the fault storm gone, prefill could finally be
profiled honestly (`CWICK_Q35_PROF=1`, one line per block). For a 176-position
block: mixer GEMM **748 ms**, MLP **1787 ms**, delta-rule **2142 ms**, attention
**1415 ms**. That is 58% of prefill on a single core, in two loops that are
embarrassingly parallel and had simply been written serially:

- the delta rule cannot parallelise over positions (it is a scan) but it can
  over value heads: each head owns its own `[128,128]` state and never reads
  another's. One dispatch per layer, one head per work item.
- attention only looks serial: after the KV for the block is stored, every
  (position, head) pair is independent. Items are `(head, position group)` with
  positions **interleaved** across groups rather than sliced in blocks — a query
  at position 400 costs four times what one at 100 does, and contiguous slices
  hand one worker all the expensive ones. Each worker allocates its own score
  vector; `m->scores` is sized for one query and is shared scratch.

Result, same block: delta-rule 2142 → **379 ms**, attention 1415 → **142 ms**,
block total 6092 → **3009 ms**, and a decode step 147 → **129 ms** because the
per-token recurrence was 12 ms of it. End to end on a 760-token prompt prefill
went from **45.1 s to 12.2 s** (17 → 62 tok/s), and the output is bit-identical
to the serial build (`CWICK_THREADS=1` diffs clean) — the parallel version
reorders nothing that any other thread reads.

**One bug found on the way.** `ssm_state` was allocated as
`val_dim x key_dim` over the *concatenated* dimensions, 16x the size the forward
pass indexes. It still worked, because the head stride is what the code walks —
it just reserved 800 MiB that nothing ever touched. Per-head allocation
(`heads x 128 x 128` = 2 MiB per layer) drops idle RSS from 19.95 to 19.24 GiB
and makes `q35_cache_bytes` and the resized state agree.

## 53. The prefix cache was writing nothing for Qwen3.5, and attention was 40x slower than its own GEMM

Two defects, found by reading a live server's log rather than by reading code:
`prefix capture failed: prefix: capture position 74 outside 0..0` on every
request, and a 31,690-token prompt whose prefill rate fell from 73 tok/s to
14.8 tok/s while the ETA grew.

**The cache never held a byte.** `cwick_engine_pos()` answers "how many positions
does this engine hold" and had branches for DeepSeek and MiniMax only; Qwen3.5
keeps its cursor in `eng->q35->pos`. So every capture was rejected as
out-of-range, `prefix_cache_hits_total` could never rise, and `./pfxcache` stayed
at 0 bytes -- the feature was off without saying so. Adding the branch is not the
fix: `cwick_engine_capture_prefix` has no q35 branch either, so it would fall
through to the mHC/KDA layout and read `eng->keys`, which this backend never
allocates. Qwen3.5 therefore needed the same opaque-blob contract the other two
backends have: `q35_prefix_pack` / `q35_prefix_apply` with `kind = 3`, and a
branch in both capture and apply. The blob is the fixed part first (24 layers of
conv + DeltaNet state, 50 MiB) and then the KV rows of the 8 attention layers,
which makes `pos` recoverable from the length alone -- so a truncated snapshot
still applies, which is what the whole-prompt path needs.

**Then the extension case still missed.** With capture working, an exact repeat
of a prompt hit (587 tokens: 9556 ms of prefill became **147 ms**), but the next
turn of the same conversation did not. Measured on a live session: turn N was 232
tokens, turn N+1 was 265, and they agreed on the first 228. Dumping both token
lists out of the `.pfx` files showed why: the server ends a prompt with the
thinking block, and with `reasoning_effort: none` that block is the *pre-closed*
one, four tokens (`<think>`, `\n\n`, `</think>`, `\n\n`). The next turn renders
the assistant's reply as header plus content and never resends it, so the two
prompts diverge at the first of those four. The existing "one token shorter" twin
covered the thinking-on case (where the block is the single open marker) and was
three tokens too long here. The fix searches the last eight tokens for the open
marker and keys the twin there; with it, turn N+1 replays 228 and prefills 37
instead of falling back to the 4 KiB block checkpoint.

**Attention ran at 8 GMAC/s.** The profile is unambiguous once the numbers are
read together: the bf16 GEMM path runs at ~550 GMAC/s, and the attention inner
loop -- one scalar multiply-add per iteration with a bf16 conversion in between,
and every K/V row re-read for every query -- ran at 8. Two changes, neither of
which reorders a sum:

- **Eight queries per pass.** Work items stay `(head, interleaved position
  group)`, so the tile is every G-th position and the eight queries in it share
  all but `7*G` keys. Each K row and each V row is now read once per tile instead
  of once per query, which is the difference between 0.5 MAC/byte and 4; the
  scalar kernel was not bus-limited (16 GB/s of a measured 123) but
  instruction-limited, and vectorising alone would have moved the bottleneck
  straight onto the bus.
- **NEON dots and axpy.** bf16 is the top half of an f32, so widening by 16 bits
  and reinterpreting is exact: no table, no rounding, four lanes per instruction.

Measured on prompts of the same shape as the failing request, one process each:

| prompt | before | after | block at ~14K |
|---|---:|---:|---:|
| 760 tokens | 12.3 s (61.6 tok/s) | **10.0 s (76.2 tok/s)** | — |
| 7,600 tokens | 314 s (24.2 tok/s) | **102 s (74.4 tok/s)** | — |
| 14,060 tokens | 935 s (15.0 tok/s) | **214 s (65.8 tok/s)** | 63.3 s → **9.6 s** |

The "before" column is the curve fitted to the logged 31K run
(`t(n) = 0.0117n + 3.90e-6n²`, which predicted that run's progress to within
1%); the same fit puts the 31,690-token agent prompt at 4285 s, and the new
profile -- 4.5 s of GEMM, 1.1 s of delta rule, 3.7 s of attention at 14K --
projects it at roughly 13 minutes. Attention is still the only term that grows
with the square of the context (38% of a 14K block), but at 130 GMAC/s instead
of 8.

Output is unchanged: three fixed prompts through the API produce byte-identical
text to the build before this change, and a 760-token prompt is byte-identical
between `CWICK_THREADS=1` and the 12-thread pool -- the tile changes who reads
what, not the order anything is summed in.

**And the window was one turn from failing.** `max_tokens 65536 clamped to
1078: prompt 31690` -- at `--context 32768` a 30K-token agent prompt leaves 1078
output tokens and the *next* turn does not fit at all. The checkpoint allows
262144 and its KV is 33.6 KiB/position, so `start.sh` now asks for 131072: 4.05
GiB of cache on top of 17.98 GiB of wired weights, which the residency policy
still approves.

## 54. Attention was the one kernel in this engine that was not a GEMM

The 30K-token prefill degraded because of one term, and the previous section
already named it: attention. What §53 left behind was a hand-written tile kernel
running at 130 GMAC/s inside an engine whose GEMM path runs at 790 — so the term
was not "irreducible quadratic work", it was quadratic work done badly. This
section is the measurement that separated the two, and the rewrite that followed.

**The shape of the problem.** Per 512-position block, everything except
attention is flat: GEMM 4.5 s, Delta-rule 1.1 s, activation 0.27 s. Attention
grows with the number of keys the block can see -- linearly per block, which
makes the whole prompt quadratic. Measured on one run (block index -> ms):

    1: 75   2: 182   3: 271   5: 438   8: 872   10: 1081   15: 1558   22: 2867

That is 0.26 ms per 1000 keys per block, so at 30K positions a block spent 7.8 s
in attention against 4.5 s in all 32 layers' projections.

**Where it went, before.** Instrumenting the pass by phase (thread-time, one
block at 11K): QK 43%, softmax 6%, AV 51%. Not one bottleneck but two, and
neither was arithmetic: the QK loop did one `vshll` + one `fma` per 4 MAC and
reloaded the query vector from memory for every key, while the AV loop loaded
and stored the 1 KiB output accumulator once per key. At 1.45 GMAC/s per thread
the kernel was 11x below what the same thread does on a plain f32 dot.

**What each loop could be.** `tools/attnbench` runs both passes at the real shape
(512 queries, 16 heads, 256 dims) and the same MAC count, one thread:

| pass | as shipped | hand-reordered | cblas_sgemm |
|---|---:|---:|---:|
| scores (QK) | 22.2 GMAC/s | 31.0 GMAC/s (1.4x) | **750.6 (33.8x)** |
| output (AV) | 18.9 GMAC/s | 43.8 GMAC/s (2.3x) | **477.1 (25.2x)** |

A re-run of `make tools/attnbench && ./tools/attnbench` on the restored tree
(11,000 keys) reads 21.9 -> 920.3 GMAC/s (42x) for scores and 18.7 -> 888.9
(48x) for output: the absolute cblas number moves with shape and thermals, the
40x gap does not.

Hand-reordering means four queries per key row with the widened key held in
registers and sixteen accumulators live. It is real, and it is 20x short. The
reason is not subtle: these two are `S = Q·K^T` and `O = S·V`, and every other
projection in this model already goes through Accelerate. The answer was to stop
hand-writing them.

**The rewrite.** `mixer_attn` now computes, per (query head, query tile of 128):

    S[nq x keys] = Q[nq x 256] · K[keys x 256]^T     (cblas, alpha = 1/sqrt(256))
    mask: columns past each row's own position set to -1e30, then softmax
    O[nq x 256]  = S[nq x keys] · V[keys x 256]      (cblas)

The causal mask is applied to a rectangle instead of skipping a triangle: the
masked entries exponentiate to exactly zero, so the sum is the same one the old
code computed, and the GEMM gets a shape it likes. Measured on the same blocks:
QK **818 GMAC/s**, AV **537 GMAC/s**.

**One trap, found before it shipped.** The first version kept the bf16 KV cache
and widened the visible rows once per layer into an f32 scratch. That is fine for
prefill -- 4 ms per layer at 30K -- and catastrophic for decode: with n = 1, each
token would widen the *entire* cache, eight times, to compute one query. The
cache itself is f32 now and the widening is gone. The snapshot on disk stays
bf16 (half the bytes, twice as many snapshots under the same budget) with
`q35_narrow_rows` on pack and `q35_widen_rows` on apply; the cost is 64 KiB of
cache per position instead of 32, i.e. 8.1 GiB at 128K against 18 GiB of weights.

**Result.** Same prompts, one process each, `--context 131072`:

| prompt | before | after |
|---|---:|---:|
| 760 tokens | 10.0 s (76.2 tok/s) | 10.0 s (76.2 tok/s) |
| 14,060 tokens | 214 s (65.8 tok/s) | **184 s (76.2 tok/s)** |
| 30,970 tokens | ~700 s (extrapolated from §53's curve; 37 tok/s at the tail) | **436 s (71.0 tok/s)** |
A block at 11K positions went from 8.9 s to 6.8 s, and its attention term from
2.87 s to 0.83 s. What is left of the growth is the honest part: QK and AV are
`O(keys)` per block by definition, and softmax over `16 heads x 512 queries`
scores is now 28% of the attention term (230 ms per block at 11K). Attention is
still the only term that grows with context, but it grows at 0.075 ms per 1000
keys per block instead of 0.26, so the rate stays close to the flat part's 76
tok/s instead of collapsing toward 37.

**Decode is a different shape.** One position, and its sixteen heads read the
same cache row -- 4 KiB that can serve sixteen dot products (four kv slices x
four query heads per group). The first version called sgemm per head, and with
M=1 Accelerate takes a path that cost 149 ms per step at 30K (3.4 tok/s). gemv
was better but still ran at 13 GB/s: 1 KiB read out of every 4 KiB defeats the
prefetcher. `dec_scores` / `dec_out_slices` walk rows instead, one read serving sixteen
uses. At 11K a step went from 190 ms to 139 ms (7.2 tok/s) and at 31K from
294 ms to 161 ms (6.2 tok/s); attention is 8 ms of the score pass and a few ms
of the output pass, and the remaining 111 ms is the weights.

**What this costs in exactness.** cblas accumulates in a different order than the
four-accumulator dot did, so prefill is no longer bit-identical to the previous
build. The inputs are still bf16 keys widened exactly to f32, and the arithmetic
is still f32; the difference is where the additions group. Three fixed prompts
through the API produce the same text as before the rewrite, and greedy decoding
is what would expose it if that were not true. The old kernel's property -- a
760-token prompt identical between `CWICK_THREADS=1` and the pool -- is now
cblas's property, not ours.


## 55. A request with no reasoning field now answers without thinking

**Observation.** Agent-style clients pointed at an OpenAI-compatible endpoint
usually send `messages`, `tools` and `max_tokens`, but no reasoning field at
all. `validate_body` defaulted that case to `极致` (budget 32768, header
`Max`), and the request line read `thinking=极致 (effort=Max, default)`. On a
37,638-token request the run went `decode phase: thinking` -> `content` after
52 tokens -> `tools` after 21 more, then streamed a long tool argument for
thousands of steps at ~5 tok/s. The client renders the opening sentence and
then shows "generating" with nothing new: tool-argument deltas are not
rendered as answer text. With one generation slot and
`CWICK_ABORT_ON_DISCONNECT=0`, stopping the client does not free the slot;
the client's own `max_tokens` (65536) also overrode the startup
`--max-tokens`.

**The second half of the root cause.** The old default was not only a level,
it was *absent*. `run_generation` passed
`gr->effort_defaulted ? NULL : gr->effort_level` to all three template
renderers, deliberately, so each checkpoint could apply its own default
(DeepSeek's header 75, `thinking=true` for MiniMax and Qwen3.5). Changing
only the default level would therefore have left the prompt opening a
reasoning block while the engine, now armed with a zero budget, closed it
immediately -- two halves disagreeing about the same turn.

**Change.** `CWICK_THINK_DEFAULT_LEVEL` / `_EFFORT` are `关` / `Low`, defined
next to `THINK_LEVELS`; the log fallback uses the same pair. The three
`effort_defaulted ? NULL : ...` ternaries are gone, so an absent field
travels as the real level `关` and takes exactly the path an explicit
`reasoning_effort:"none"` takes. `cwick_thinking_default_level()` exposes the
value to the test suite, which asserts it equals `resolve("none")`. All four
explicit spellings still override it and an unknown spelling is still a 400.

**Reach.** chat, responses and messages all synthesise a chat body; with no
reasoning field the synthesised body has none either, so all three follow
the new default without a per-dialect switch. The decision endpoint is
unchanged (it already refuses any effort other than `none`). The bundled web
console sends `reasoning_effort` explicitly and keeps its own selected level.
CLI `run` has no request body and is unrelated.

**Verification.** `make test` passes with one new assertion (671 total). A
temporary server on port 18001 (Qwen3.5, `--context 8192`) answered five
bodies:

| request | log line | first decode phase | think tokens |
|---|---|---|---:|
| chat, no field | `thinking=关 (effort=Low, default)` | `content` | 0 |
| responses, no field | same | `content` | 0 |
| messages, no field | same | `content` | 0 |
| chat, `reasoning_effort:"high"` | `thinking=高 (effort=High)` | `thinking` | 1 |
| chat, `reasoning_effort:"none"` | `thinking=关 (effort=Low)` | `content` | 0 |

The no-field chat response was `content:"2"`, `reasoning_content:null`,
`reasoning_tokens:0` and the summary line `think=0 tools=0 content=1
think_cap=0`. No no-field request entered `decode phase: thinking`, and an
explicit level still wins. To restore the old behaviour, put the two macros
back to `极致` / `Max` and restore the three ternaries -- but that also
restores the prompt/engine disagreement.

**Not part of this change.** `max_tokens` remains client-controlled and still
overrides `--max-tokens`; a server-side cap is a separate decision.


## 56. The phase label existed but never reached the fourth engine

**Acceptance check against goal #10.** A 573-line serve log from the Qwen3.5
build (`--verbose --trace`, 15:03) has 96 `decode forward ...` lines and two
periodic `decode N/32768 tokens ...` lines. It contains **zero** occurrences
of `decode(`. Only the two transition lines named a phase
(`decode phase: content at token 1`, `content -> tools after 27 tokens`).
Prefill was fine: `[prefill] ~91/364 tokens (25.0%), layer 8/32 — 68.3 tok/s,
eta 4s`, and every line carried a wall-clock time and `[req N]`.

**Cause 1 — the phase hint stopped at three backends.** `phase_trk_observe`
calls `cwick_engine_set_decode_phase` on every token. That setter copied the
hint into the DeepSeek and MiniMax contexts but not the Qwen3.5 one, so
`q35_ctx.decode_phase` stayed NULL and `q35_phase_label` printed plain
`decode` on both the per-step and the periodic lines. The same copy-and-miss
was present in `set_stop`, `set_abort_poll` and `set_verbose`; all four
setters now reach every backend. This is the same class of bug as §19 (a
per-request field that only one loop saw), one backend later.

**Cause 2 — the thinking phase never ended.** Qwen3.5's generation prompt
ends with `<think>\n`, i.e. the opening marker followed by a newline token.
`run_generation` tested only `ids[nids-1] == think_open_id`, concluded
"opened by the model", and armed the budget with `open_at_start=false`.
`cwick_think_plan` injects the closing marker only when `think_open` is true;
the model has no reason to write an opening marker the template already
wrote, so the budget never fired. With `CWICK_THINK_BUDGET=8` a 130-token
`high` request produced `think=130 content=0`. The close-marker check already
used a 4-token backwards window for the two templates that trail whitespace;
the same helper (renamed `prompt_token_near_end`) is now used for the opening
marker on Qwen3.5, guarded by `!pfx_closed_at_start` so a template that
already closed the block still reports "already closed".

**Change.** `engine.c`: Qwen3.5 branches in four setters. `server.c`: shared
window helper, Qwen3.5 open-marker detection, and a per-phase rate on the
transition line. No kernel, precision or sampling change.

**Measured** (temporary server, port 18001, Qwen3.5, `--context 8192`,
`--verbose`, `CWICK_PROGRESS_MIN_MS=1000`, `CWICK_THINK_BUDGET=8` for the
budget check):

| | before | after |
|---|---|---|
| per-step line | `decode forward 1/32767 161 ms` | `decode(tools) forward 1/69 140 ms` |
| periodic line | `decode 34/32768 tokens (0.2%)` | `decode(thinking) 8/26 tokens (30.8%)` |
| phase labels | none | thinking / content / tools, on every line |
| transition | `... after 27 tokens (3.9s)` | `... after 8 tokens (1.1s, 7.01 tok/s)` |
| budget 8 | `think=130 content=0`, never fired | `wrote </think>`, `think=10 content=20` |
| no reasoning field | — | unchanged: `reasoning_content:null`, content `2` |

The budget fix is a real behaviour change for explicitly enabled thinking:
a long think block now ends at its ceiling and the turn answers. The
default-no-thinking path is unchanged (`block already closed by the
template`, `decode phase: content at token 1`). `make test` 671 assertions
PASS. The open-marker window is limited to Qwen3.5; the MiniMax template
still uses the exact-last-token test for the opening marker, because no
measurement here justifies changing it.

**Out of scope.** One hand-written tool request produced `model emitted
undeclared tool <function=get_time>` — a tool-format mismatch, not a logging
or phase problem. Left alone.


## 57. Why a streaming client goes silent during the tools phase

**Question.** After the opening sentence the front end shows "generating" and
nothing else for the whole `decode(tools)` phase, while the engine is
demonstrably still working: a real 38-tool request logged 637
`decode(tools)` steps at ~7 tok/s after `content -> tools after 27 tokens
(4.1s, 6.53 tok/s)`.

**Measured.** Temporary server, port 18001, Qwen3.5, `stream:true`, one
declared tool, `--verbose`. Raw SSE frames with arrival timestamps:

| dialect | frames in 5.6 s | standard tool delta | private progress |
|---|---:|---:|---:|
| chat | 15 | 0 | 12 x `delta.cwick_tool`, ~141 ms apart |
| responses | 3 (`created`, `in_progress`, `failed`) | 0 | 0 |
| messages | 2 (`message_start`, `error`) | 0 | 0 |

The chat frames carry `delta.content:""` once, then twelve chunks of the
in-flight tool block in a **non-standard** field `delta.cwick_tool` (e.g.
`"<"`, `"function"`, `"=get"`, `"_time"`, `"</tool_call>"`). No
`delta.tool_calls[].function.arguments` delta is ever emitted.

**Cause 1 — progress is private, not standard.** `stream_tool_progress()`
sends the in-flight block through `stream_delta_n(..., "cwick_tool", ...)`;
its own comment says the field is "ignored by clients that do not know it".
The only standard `tool_calls` chunk is built in `stream_finish()` from the
fully parsed array, once, at the end of the turn (`server.c` ~3508).
`resp_stream_text()` states the same policy outright: "Tool-call progress
has no equivalent here: the arguments arrive complete in the function_call
item at the end of the turn".

**Cause 2 — two dialects drop even the private field.** `stream_delta_n()`
routes `cwick_tool` into `resp_stream_text()` / `msg_stream_text()`, which
handle only content / reasoning / text and discard unknown fields. Hence
responses and messages emit *zero* deltas for the entire phase. The built-in
web console is the only consumer: `web/app.js` accumulates `d.cwick_tool`
into `assistantMsg.toolStream`, which is why the requirement to render
streaming toolCalls holds for cwick's own UI and nowhere else.

**Cause 3 (this request) — the tool dialect did not match.** The model
emitted `<function=get_time>\n</function>` while `cwick_tools_extract()`
expects `<tool_call>NAME<arg_key>...`. Parsing failed, the server sent
`{"error": ... "model emitted undeclared tool"}` and `[DONE]`, so even the
final all-at-once `tool_calls` chunk never arrived. This is the mismatch
recorded as out-of-scope in §56.

**Why the front end freezes.** Content-phase deltas are standard
`delta.content`, so the opening sentence renders. At the first `<tool_call>`
the splitter stops classifying and the only bytes left on the wire are
unknown fields or nothing. A strict client keeps its last rendered frame and
shows its "generating" indicator until the request ends; if the call parses
successfully it sees one single `tool_calls` chunk at the very end.

**Not changed in this round.** No protocol code was touched. Fixing this
means emitting standard tool deltas per dialect (chat: a `tool_calls[i]`
opening chunk with id/type/function.name followed by
`function.arguments` shards; responses: `response.output_item.added` +
`response.function_call_arguments.delta`; messages:
`content_block_start(tool_use)` + `content_block_delta(input_json_delta)`),
while retaining `cwick_tool` for the built-in console. The in-flight text is
cleaned XML, not JSON arguments, so a streaming XML-to-JSON mapper is part
of that work. Cause 3 is smaller, independent, and should land first:
otherwise a client can watch a tool call that still fails at the end.


## 58. Fixing the silent tools phase, in the order diagnosed in §57

§57 recorded three causes and deliberately changed no protocol ("not changed
in this round" there is a historical note). This section implements its
proposed fixes in that order: the dialect mismatch first, then the standard
streaming deltas.

### 1. The tool dialect mismatch (cause 3)

The nested template answers with

```
<tool_call>
<function=save_note>
<parameter=title>
Memo
</parameter>
</function>
</tool_call>
```

while `extract_impl` only reads `<tool_call>NAME<arg_key>k</arg_key>
<arg_value>v</arg_value></tool_call>`. The turn therefore failed with
`model emitted undeclared tool` and no `tool_calls` at all, not even the
end-of-turn one. `tools.c:translate_function()` rewrites the nested form into
the flat one before the single existing parser sees it, exactly as
`translate_invoke()` already does for another checkpoint. Whitespace framing
around a value is trimmed, inner newlines are kept, `</tool_call>` wrappers
are discarded as framing, and prose outside the block is preserved. A text
that already uses `<arg_key>` is left alone; a `<function=` without its
`</function>` is left unchanged so the extractor still refuses the turn
(fail-closed). Seven new unit assertions cover zero-argument calls, multiple
parameters, inner newlines, numeric and JSON values, prose, multiple
functions in one wrapper and the unterminated case.

Measured on the real checkpoint: the same hand-written tool request that used
to end in `error` now returns `tool_calls:[{name:"get_time",arguments:"{}"}]`,
`finish_reason:"tool_calls"`, access log `finish=tool_calls calls=get_time`.

### 2. Standard deltas in all three dialects (causes 1 and 2)

`stream_tool_progress()` still emits the private `delta.cwick_tool` for the
bundled console. In parallel, `tool_std_scan()` watches the in-flight block:
once the model has written `<function=NAME>`, `tool_std_open_frame()` opens
the call with an empty `arguments` in the dialect's own protocol, and each
completed parameter appends a shard.

| dialect | opening frame | argument shards |
|---|---|---|
| chat | `delta.tool_calls[0]` with `id` / `type` / `function.name` | `delta.tool_calls[0].function.arguments` |
| responses | `response.output_item.added` (`function_call`) | `response.function_call_arguments.delta` |
| messages | `content_block_start` (`tool_use`) | `content_block_delta` (`input_json_delta`) |

At the finish the three closing functions skip the first call's `added` /
`delta` / `block_start` events but still send `done`, `output_item.done` and
`content_block_stop`, so a client that reads only the terminal events sees the
same object. `tool_calls_json()` gained an `empty_first` mode so a
concatenating chat client never sees the streamed bytes twice.

**Why the wire bytes and the payload cannot disagree.** The shards are not
produced by a second XML grammar. `tool_std_scan()` takes the in-flight text
up to the last `</parameter>`, closes the block synthetically, and calls the
same `cwick_tools_extract()` the buffered path uses; dropping the object's
final brace leaves a prefix that can only grow. At the finish
`tool_std_settle()` compares it with the parsed `arguments_json` and sends the
remaining suffix. If they disagree, the bytes already on the wire become the
payload — a client concatenating deltas and one reading the final object
still agree. The call id is allocated once on the opening frame and written
back to the parsed call, and the Responses terminal document is rebuilt
afterwards, so envelope, item and deltas all name the same call.

### Measured

Temporary server on port 18001, Qwen3.5, `--context 8192`, `CWICK_SEED=42`,
one request per dialect, raw SSE with arrival timestamps.

| | §57 before | after |
|---|---|---|
| chat | 15 frames, **0** standard tool deltas, 12 private frames | 43 frames, 1 opening + 3 argument shards, 37 private frames |
| responses | 3 frames (`created` / `in_progress` / `failed`), 0 deltas | 9 frames, 3 argument deltas |
| messages | 2 frames (`message_start` / `error`), 0 deltas | 8 frames, 3 `input_json_delta` events |
| first standard frame | none (only the end of the turn) | 5.0 s, i.e. as soon as the name was written |
| parsed call | refused (`undeclared tool`) | `save_note`, arguments complete |

Chat standard frames landed at 5.0 / 6.5 / 8.3 / 9.3 s of a 9.6 s turn, so
progress is genuinely in flight. All three dialects concatenated to exactly
`{"title":"Memo","body":"hello world"}`, identical to the non-streaming
payload and to Responses' `function_call_arguments.done`. Zero-argument calls
produced `{}` on all three. `make test` 678 assertions PASS (7 new), nine
end-to-end checks PASS, and a plain answer with no tools emitted no tool
frames at all.

### Bundled web console

`web/app.js` used to treat every `tool_calls` frame as a new call, so the
shards would have created empty duplicate cards and repeatedly cleared the
raw progress view. `applyToolCallDeltas()` now folds chunks by `index`, keeps
the first id/name, appends arguments, and mutes `cwick_tool` once a standard
card is open. `respEventToChat()` also maps
`response.function_call_arguments.delta` to progress (the Messages adapter
already consumed `input_json_delta`). Nine pure-function checks in Node PASS,
including legacy whole-frame calls, multi-call arrays and a streamed first
call followed by a multi-call finish. `src/webui_embed.c` was regenerated.

### Boundaries, deliberately left as a follow-up

- Shards are emitted **per completed parameter**. A single very long value
  (the whole HTML file from the original report) still arrives when its
  `</parameter>` closes. Streaming *inside* a value needs a state machine
  that keeps JSON type detection, incomplete UTF-8 and trailing-whitespace
  trimming monotone; that is a separate decision, not a tuning tweak.
- Only the first call is streamed; further calls arrive whole at the finish.
- A block in another dialect (flat `<arg_key>`, or the `<invoke>` form) has no
  `<function=` opener, so nothing is opened early and the original
  end-of-turn behaviour is preserved unchanged.

## 59. Saying it is streaming is not the same as looking like it

§58 opened the call while it was being written and pushed each *completed
parameter* as a shard. Measured against a real agent-shaped call, that was
still a silent phase: an agent's tool call is almost always **one long
parameter** (a shell script, a file body), and a shard per parameter means the
whole value lands in a single frame at the very end.

Reproduced on the real checkpoint with the same prompt an agent sends:

| | before | after |
|---|---|---|
| standard `tool_calls` frames | 3 | 20 |
| arrival | opening 5.1 s, then **4.1 s of nothing**, then the whole value | every ~0.15 s from 5.2 s to 10.2 s |
| largest gap between frames | 4.1 s | 1.1 s |

Three frames is arithmetically "streaming" and reads as a hang. A client that
only paints after the first argument byte shows nothing for the entire phase,
which is exactly the report: the frontend shows no streaming during
`decode(tools)`.

### What changed

`tool_std_value_scan()` now slices the value of the **parameter still being
written**, not just the parameters whose `</parameter>` has arrived. The value
is handed to the same `cwick_tools_extract()` as the closed path — the block is
truncated at the safe byte and closed synthetically — so escaping, whitespace
trimming and key order cannot drift from the final payload.

Three things make a mid-value slice safe to put on the wire:

1. **Type monotonicity.** A prefix is only emitted when it can never stop
   being a JSON string: the value must not start with `{ [ " -` or a digit,
   and `t`/`f`/`n` are held while they are still a prefix of
   `true`/`false`/`null`. A shell script, a path, a sentence passes; `137`
   and `{"a":1}` do not, and keep the closed-parameter path. The test is
   monotone — once true it stays true — so committed bytes can never need
   retracting.
2. **Trailing-whitespace hold.** The extractor trims leading/trailing
   whitespace from a value. Emitting a trailing space that is later trimmed
   would make the streamed prefix disagree with the payload, so the tail is
   held until a non-space byte lands.
3. **`</parameter>` hold.** A closing tag arrives one piece at a time; a tail
   that could still grow into `</parameter>` is framing, not value. Without
   this the first version emitted `</parameter` inside the JSON string and the
   concatenated result was invalid — caught by the end-to-end check, not by
   the unit test, which is why the check below exists.

The closing quote is stripped from the in-flight prefix (the extractor adds it
as part of `"value"`), so the byte after the last value byte is the one that
turns a prefix into a complete string. Incomplete UTF-8 at the tail is held
back for the same reason as in the content path.

### Granularity

Each scan re-parses the whole block, so it does not run per token for a large
value: after the first byte of a value, a frame goes out every 12 bytes. At
6-7 tok/s that is a frame every ~0.15 s for ASCII and ~0.4 s for CJK, while
the parse cost stays bounded. Measured decode rate is unchanged (6.82-7.43
tok/s against a 6.4-6.87 baseline), so the scan does not compete with the
forward pass.

### Verified

Eleven end-to-end checks PASS on the real checkpoint: chat concatenation
equals the non-streaming payload byte for byte, exactly one opening frame
carries the id, the private `cwick_tool` progress is still emitted, and zero-
argument calls yield `{}` in all three dialects. Numeric, boolean and object
parameters keep their JSON types (`count:137`, `meta:{"a":1}`) because they
never enter the in-flight path. All three dialects concatenate to the same
bytes and every result parses. `make test` 683 assertions PASS (5 new: a
mid-value prefix must be a byte-prefix of the final JSON, and must be strictly
shorter). The bundled console was re-checked with seven pure-function cases,
including a long shard sequence and the closing frame.

A turn truncated by `max_tokens` mid-call still ends with the existing
`unterminated <tool_call>` error frame — unchanged, and correct: the call was
never finished, so there is nothing valid to hand back.

### Separately found, not fixed

The DeepSeek prompt template teaches its own dialect
(`<｜DSML｜ calls>` / `<｜DSML｜ invoke name="…">` / `<｜DSML｜ parameter …>`),
but `cwick_tools_extract()` only knows `<tool_call>`, `<function=`,
`<invoke name="` and the MiniMax namespace token — DSML matches none of them.
`cwick_stream_segment()` likewise only enters tool mode on a literal
`<tool_call>`. So on a checkpoint that follows its own template exactly, the
tools phase produces neither a phase label nor any delta of any kind. That is a
different defect from the one fixed here (it is a missing dialect, not missing
granularity) and it needs its own measurement against a DS turn before being
changed; it is recorded here so the next reader does not mistake this section
for a complete fix of the tools phase.

## 60. The Anthropic client's own URL shape was a 404

**Symptom.** An agent harness pointed at this server gets
`404 {"error":{"message":"unknown endpoint","type":"invalid_request_error"}}`
on every turn. One serve log line carries the whole story:

```
[req 1] POST /v1/v1/messages?beta=true (46045 bytes body)
[req 1] POST /v1/v1/messages?beta=true -> 404 in 1.1ms auth=key:cwick_N5TmAqb
```

Two independent defects sit in that one path, and either of them alone is
enough to produce the 404.

**Cause 1 — the generation routes compared the raw path.** Every other route
in the handler matches `ponly`, the query-stripped form that `path_only()`
writes at the top of the function; the four generation routes were the only
ones still comparing `req.path`. The Anthropic client puts `?beta=true` on
every `/v1/messages` call, so `strcmp(path, "/v1/messages")` never matched —
the endpoint was unreachable for the one client it exists for, and
`/v1/chat/completions?x=1` was unreachable the same way.

**Cause 2 — a base URL that already ends in `/v1`.** The Anthropic SDK appends
`/v1/messages` to whatever base URL it is given. Configure it as
`http://host:8001/v1` and the request arrives as `/v1/v1/messages`, which no
route in the table knows. Rejecting it is technically correct and practically
useless: the fix is on the client's side of a config field the client does not
own, and the 404 text does not say so.

**Change.** `server.c`: after `path_only()`, collapse a repeated `/v1/v1/`
prefix in place (loop + `memmove`) so every route benefits, and match the four
generation routes against `ponly`. The raw path is untouched, so the access
log and the `--trace` line still print what arrived. No kernel, precision,
sampling or prompt change.

**Measured** (same server, same auth, POST with `{}` so a routed request
fails body validation instead of routing):

| request | before | after |
| --- | --- | --- |
| `POST /v1/v1/messages?beta=true` | 404 unknown endpoint | 400 `max_tokens is required` |
| `POST /v1/messages?beta=true` | 404 unknown endpoint | 400 `max_tokens is required` |
| `POST /v1/messages` | 400 (already routed) | 400 (unchanged) |
| `POST /messages?beta=true` | 404 unknown endpoint | 400 `max_tokens is required` |
| `POST /v1/chat/completions?x=1` | 404 unknown endpoint | 400 `messages must be a non-empty JSON array` |
| `POST /v1/responses?beta=true` | 404 unknown endpoint | 400 `missing input` |
| `GET /v1/v1/models` | 404 | 200, 5 known models |

A real streaming call then completes on the exact URL that used to 404:
`POST /v1/v1/messages?beta=true`, `max_tokens 24`, one user turn →
`message_start … content_block_stop … message_delta … message_stop`,
`200 in 1341.9ms prompt=15 completion=1 finish=stop`. `POST /v1/v1/models`
still answers 404 — normalisation is about the path, not the method.

**Not fixed here.** After routing, the same harness will be told
`model \`qwen3.5-9b\` is not served by this endpoint (the resident model is
\`qwen3_5_text\`)`: this checkpoint's directory is not in
`cwick_model_served_alias()`, so it answers to its own `model_type`. That is
the documented contract (see /v1/models), not a routing bug — send the id the
catalog advertises.

## 61. Qwen3.5 decode: the whole step in one command buffer (7.0 -> 20.0 tok/s)

**The measurement this rests on.** A decode step moves ~16 GB and does almost
no arithmetic per byte, so it is a bandwidth question. This machine has two
paths to memory, and they are not the same width: the CPU cluster saturates at
**123.8 GB/s** from four threads up, while the GPU reads at **284.6 GB/s**
(and 368 GB/s in the chained shape a real step uses). Nothing about the second
number is reachable from the CPU. The old ruling — "decode already runs at 93%
of the bus, stop looking" — measured that 93% against the CPU's path.

**What the GPU does not give you for free.** Submitting a projection at a time
is a loss. Measured with `tools/q35step`: 128 GEMVs of one step (the four sync
points a layer needs, 32 layers) cost **118.8 ms** as 128 submissions, because
each commit+wait is ~928 us and only ~58 ms of that is the kernels. The same
128 GEMVs encoded into **one** command buffer (`tools/q35one`) cost **37.6 ms**
— 368 GB/s, and identical with a 4x larger working set, so it is the DRAM rate
and not a cache artifact. The dispatch term, not the bandwidth, was the wall.

**So: one command buffer per step, and the recurrence on the GPU too.** Splitting
a projection's output rows between the CPU pool and the GPU — what the
DeepSeek backend does — is a net loss here: measured with `tools/dualbw`,
12 CPU threads and one GPU dispatch running at once reach **213.7 GB/s**
together against 328.9 for the GPU alone. The two engines share one pipe and
the CPU's share of it is the slower one. That leaves exactly one shape that
works: every operator of the step on the GPU, 32 layers inside a single serial
compute encoder, one commit, one wait, and one host readback — the logits.

**Implementation.** Three new units, split so that only the last one is ObjC:

| unit | role |
|---|---|
| `q35_shader.c` | 16 Metal kernels, each the twin of a function in `q35_model.c`: BF16 GEMV, RMSNorm with the `1+w` flag, embedding gather, depthwise conv + SiLU + window rotate, per-head L2 norm, the DeltaNet gates and recurrence, per-head Q/K norm with partial rotation, KV append, score pass, softmax, weighted sum, output gate, SwiGLU, residual add |
| `q35_gpu.c` | plain C. The schedule: which projection feeds what, where the norms sit, the position's rotation table, the buffer bindings. Knows nothing about Metal |
| `q35_gpu.m` | the Metal quarantine: device, library, pipeline states, no-copy buffer adoption, the encode calls |

The weights are adopted with `newBufferWithBytesNoCopy` over the mmap'd
safetensors — 13.19 GiB at a 4K context, 20.94 GiB at 128K, **zero copies and
zero extra resident memory**. The KV cache, the convolution window and the
DeltaNet state are adopted the same way out of the buffers `q35_model_open`
already owns, which is what makes the CPU prefill and the GPU decode write the
same bytes with no synchronisation between them. Prefill stays on the CPU,
where it already runs at 71-80 tok/s; only one-position steps take the new path.

**Correctness.** `CWICK_Q35_GPU_CHECK=1` runs the CPU step on the same position
and compares. It has to restore the recurrence state first (`q35_prefix_pack` /
`apply` — a DeltaNet state cannot be truncated like a KV cache), otherwise the
second run is simply a different step. Measured, three positions:

| position | max abs diff | mean abs diff | argmax |
|---|---|---|---|
| 43 | 0.00804 | 0.001855 | agree |
| 44 | 0.00332 | 0.000564 | agree |
| 45 | 0.00235 | 0.000277 | agree |

That is f32 reordering noise against bf16 weights, and it *decays* with
position instead of accumulating — the two paths are computing the same thing.
End to end, 90 greedy tokens from the same prompt come out **byte-identical**
to the CPU path.

One real bug was found by that check and not by inspection: the DeltaNet
output gate read `z[j]` where the gate projection is indexed by value head
(`z[vh*hv + j]`), so all 32 heads read head 0's gate. Mean logit error was
1.77 with a different argmax; after the fix it is 0.0002.

**Result** (`cwick run`, 44-token prompt, 63 decode steps, context 128K):

| decode path | ms/step | tok/s |
|---|---|---|
| CPU pool | 143 | **7.00** |
| GPU, whole step | 49.8 | **20.10** |
| GPU, whole step, `--context 131072` + prefix cache, over HTTP | 48.9 | **20.45** |

2.9x. The GPU time attributable to kernels is 46.5 ms of the 47.4 ms wall, i.e.
341 GB/s against the 368 GB/s the isolated GEMV-chain measured — 93% of the
number the shape mix allows, so there is no dispatch overhead left to reclaim.
`CWICK_METAL_TIME=1` prints `commit=6us wait=47ms gpu=46.5ms` per step.

**Not done here.** Prefill is still the CPU's, and at a 31K agent prompt it is
390 s of TTFT against 1.5 s of decode — the next thing worth moving, and a
different problem (GEMM, not GEMV). And 40 tok/s needs the MTP draft head on
top of this step, which is what the next section is about.

## 62. Checking k tokens costs less than k steps, but a draft costs more than the check saves

**The half that worked.** A decode step reads 16.4 GB of weights to produce one
token, so the way past 20 tok/s is to produce more than one token per read:
draft k tokens, then check all k in a single pass whose weight read is the same
16.4 GB. The checking half is done, and it is exact.

The obstacle is that the DeltaNet layers carry a matrix recurrence rather than a
KV cache — state that cannot be truncated, only recomputed. Keeping the first j
of k checked rows therefore needs the state as it stood after row j, which is
either a separate pass per row (k passes, no saving) or a 50 MiB copy per step.
What it is instead: **the state is allotted k+1 slots and rolls**. Row r reads
slot r and writes slot r+1, so no slot is overwritten by the row that follows
it, slot 0 is always live, and keeping j rows is one `memmove` from slot j+1 to
slot 0 — no extra traffic during the pass, and one copy at the end instead of k.
The convolution kernel wrote its window in place (`st` was both its input and
its output), which a rolling slot breaks; it now takes a separate source
pointer, and with both pointers on the same slot it is bit-for-bit the in-place
version.

`CWICK_Q35_GPU_CHECK=1` compares *every* row of a k-row pass against the same
GPU path run one row at a time. At k = 2, 3, 4 and 8 the answer is **0.00000
max, 0 of N differ** across all 32 layers — logits, KV, convolution window and
recurrence state. That is what the rest of this section stands on: a k-row pass
is not an approximation of k steps, it is the same arithmetic.

**What a row costs, measured as a real decode step** (`CWICK_Q35_VERIFY=k`, 19
passes, one prompt, one session):

| k | ms/pass | marginal ms/row |
|---|---|---|
| 1 | 55.5 | — |
| 2 | 63.6 | 8.1 |
| 4 | 73.4 | 4.9 |
| 6 | 85.6 | 6.1 |
| 8 | 98.4 | 6.4 |

The marginal cost *rises* with k, which rules out both comfortable explanations.
Not bandwidth: a pass reads 16.4 GB whatever k is, so a bandwidth limit would
make the marginal zero. Not a per-dispatch overhead either: that would inflate
the k = 1 row and leave the marginal flat. It is the arithmetic of the extra
rows, at a rate that degrades as the register footprint grows — the isolating
benchmark (`tools/q35gemsw`, the checkpoint's own 13.82 GB of shapes, 32 layers)
gets 4.5 ms/row at k = 8 where the step gets 6.4.

That benchmark is also what found the one knob taken here. The kernel is a
threadgroup tile: the k activation rows are staged once and shared by the whole
group, one simdgroup per weight row. Letting a simdgroup own RP weight rows
reuses the same staged slice RP times and cuts the activation side of the pass
by RP, at the cost of RP*N live accumulators:

| rows n | RP1 ms | RP2 ms | RP4 ms |
|---|---|---|---|
| 4 | 45.3 | 45.8 | 47.1 |
| 5 | 49.9 | 49.0 | 66.6 |
| 6 | 54.4 | 52.5 | 72.5 |
| 8 | 70.1 | 63.3 | — |

RP4 spills from n = 5 up; RP2 wins from n = 5 up and loses slightly below it,
which is the policy. RP is the only knob in that kernel that can be turned
without moving an FMA within an accumulation, and that is what keeps the bitwise
check above true; widening the load to 8 bf16 was 3 ms faster at n = 2..4 and
was not taken for the same reason. In a real step it is worth **102.2 -> 98.4 ms**
at k = 8 (same session, three interleaved reps, best of each) — less than the
benchmark's 6.8 ms, because part of a real pass does not scale with RP.

**The drafter, priced.** With the check half attached, 40 tok/s becomes a
statement about drafts: a pass of k+1 rows has to yield `A` accepted tokens with
`A / T(k) >= 40`, and `A` is the draft's business, not the verify's.

The zero-cost draft is a prompt lookup — match the last 2-6 tokens against an
earlier occurrence in the context and continue from there. It reads no weights,
and it cannot corrupt anything, because the accept test is the full model's own
greedy argmax; with it off the output is byte-identical. It also does not work.
Two prompts, 40 tokens each: **1.08 to 1.24 tokens per pass** (52% of drafts
accepted on prose, 23% on code), and end to end **2022 ms against 2011 ms** of
baseline — the wider pass costs more than the extra tokens are worth. Free is
not the same as useful.

The learned draft on disk is the MTP head, and its file settles the price:
**486.6 MB**, fifteen tensors — `mtp.fc[4096,8192]`, one full attention layer
(GQA, `q_proj[8192,4096]` with the interleaved gate, `k_proj`/`v_proj` at
`[1024,4096]`, `o_proj[4096,4096]`, SwiGLU at 12288) and three norms — and **no
output projection of its own**. So a draft token reads 486.6 MB *plus the shared
2.03 GB vocabulary head*: 2.52 GB, 7.0 ms at the measured rate.

| k | T(k) = pass + k x 7.0 ms | A at p = 0.8 | tok/s |
|---|---|---|---|
| 2 | 82 | 1.80 | 21.9 |
| 4 | 109 | 2.95 | 27.1 |
| 8 | 162 | 4.16 | 25.7 |

**That table is the answer, and it is not 40.** The head is 81% of a draft token,
so drafting eight tokens costs 56 ms against the 43 ms the seven extra rows cost
the verify pass — the gain is handed straight back. Accepting more does not
rescue it either: at p = 0.9, eight drafts return 5.70 tokens on a 162 ms pass,
which is 35 tok/s, and 40 would need **p >= 0.94**.

Two things would change the answer, in order of size. **Make a draft token cost
~1.4 ms by projecting only a candidate slice of the vocabulary** — a draft is
allowed to be wrong, the verify decides, so this is a bandwidth question and not
an accuracy one; the top 4096 rows are 33.5 MB instead of 2.03 GB, which puts
k = 8 at 118 ms and 4.16 accepted tokens at **35 tok/s**, with the bar for 40
falling to p >= 0.87. And **cut the 6.4 ms
marginal row**, which is now throughput-bound rather than bandwidth-bound and is
the only lever that helps every k at once. Both need the MTP bound first, and
neither is a tuning change.

**One bug found on the way, unrelated to any of the above.** The single-row
decode path called the forward with `ids` still holding the prompt and never
wrote the token it had just sampled into `ids[0]`, so every step of every
generation was conditioned on `prompt[0]` instead of on the token it had
produced. That degenerates into a repetition within a dozen tokens
(`TheTheThe...`, `MASKMASK[g[g[g...`) and it invalidated every earlier reading
of output text in this document. The k-row path does not share the mistake,
which is how it surfaced: the same prompt produced fluent text through
`CWICK_Q35_VERIFY` and did not through the ordinary path. The fix is one line.

## 63. The marginal row was activation traffic, and it did not have to go through threadgroup memory

§62 left one number unexplained: a k-row verify pass should cost one weight read,
but the marginal row measured 6.4 ms in a real step (55.5 ms at k = 1, 98.4 at
k = 8). Arithmetic is not the reason, and guessing at the reason had already
produced two wrong answers, so the kernel was ablated instead
(`tools/q35gemab`, 13.82 GB per pass over this checkpoint's own shape mix, one
term removed at a time, k = 8):

| what the kernel keeps | ms | what it removes |
|---|---|---|
| everything | 65.8 | — |
| no weight read | 36.4 | the 13.82 GB DRAM sweep |
| no activation read | 45.2 | the threadgroup read of the k rows |
| one add per weight element | 64.0 | every N-way multiply |

Three things fall out. The multiply-adds are worth **1.8 ms** in total, so a
marginal row never costs arithmetic. The weight read is **29 ms** and the
activation read **21 ms** — and 36.4 + 29.4 is 65.8, the measured total, which
means the two memory streams were not overlapping *at all*. That second point is
the one worth stating plainly: this was never a bandwidth ceiling, it was a
kernel that serialised two streams it could have interleaved.

The activation read existed because the rows were staged: one copy of the k rows
into threadgroup memory per threadgroup, shared by every simdgroup in the group,
two barriers around each staged tile. That design was chosen in §62 to trade
threadgroup bandwidth for weight bandwidth, and the ablation shows the trade was
a loss. x is N x k x 4 bytes for the whole pass — 128 KB at k = 8 — so it stays
resident in L2 for the duration, and every threadgroup sweeping those same lines
costs nothing like a DRAM read. Reading it straight from device memory removes
the staging loop, removes both barriers, and makes the k loop flat.

| | k=1 | k=2 | k=4 | k=6 | k=8 | marginal row |
|---|---|---|---|---|---|---|
| staged, 256 threads | 37.7 | 41.0 | 49.3 | 62.4 | 65.8 | 4.0 |
| device, 128 threads | 39.4 | 41.9 | 42.2 | 46.5 | 52.6 | 1.9 |

Narrowing the threadgroup to 4 simdgroups is part of the same change: with the
staging gone a group holds no shared state, so a narrower group just means more
of them resident. 128 threads beat 256 at every k (53.1 against 57.7 at k = 8,
both with the activation read vectorised); a 512-thread group was worse than
both.

Two other settings were measured and not taken. Prefetching the weight loads
(PF = 2 or 4, issuing several loads before consuming any) changed nothing, which
is consistent with the overhead being the staging structure rather than memory
level parallelism. Reading a wider vector per lane is faster on the cheap passes
and spills on the expensive ones, and it would re-associate the sum, which costs
the bitwise check — the same reason §62 declined it.

The activation is read as four scalars rather than a `float4`. The scratch buffer
a projection reads from is sized by allocation width, not output width, so a row
start is not guaranteed 16-byte aligned, and a misaligned vector load in Metal
returns the wrong bytes instead of taking a slow path. The scalar form costs
0.5 ms of the pass (52.6 against 52.1) and removes the obligation to prove the
alignment.

In a real decode step (99 passes, best of three interleaved reps) the same change
is **55.5 -> 47.4 ms at k = 1** and **98.4 -> 81.8 ms at k = 8**, marginal row
**6.4 -> 4.9 ms**. The residual 2.1 ms per row is not in this kernel: the 24
linear layers run their convolution and recurrence once *per row*, six dispatches
and a barrier each, so a share of every marginal row is dispatch overhead rather
than bytes. That is the next thing to attack, and it is a different change.

End to end through the API (200 tokens, `max_tokens` honoured, chat completions)
the step is **21.3 tok/s** against 18.3 before. The bitwise invariant is intact:
`CWICK_Q35_GPU_CHECK=1` still reports `0.00000` on all 8 rows across 32 layers,
because the change removes a copy and does not move a single FMA within an
accumulation.

**What this does to the draft arithmetic of §62.** The verify half got cheaper,
so the price a draft has to beat moved with it. A pass of k rows yields k tokens
and only k-1 of them need a draft, so at accept rate p the expected accepted
count is A = sum of p^i for i < k — 1.80 / 2.95 / 3.95 at p = 0.8 for k = 2 / 4 /
8. With the MTP head at 7.0 ms per draft token the best case is k = 4: a 62 ms
pass plus three drafts is 83 ms for 2.95 tokens, **35.5 tok/s**. Still not 40,
because the head is now the larger half of the cost. Project a draft against a
candidate slice of the vocabulary instead (33.5 MB, ~1.45 ms; legitimate because
the verify decides, not the draft) and k = 4 becomes 67 ms, **44.4 tok/s**, and
k = 8 becomes 92 ms for 3.95 tokens, **43.0 tok/s**. So "40 tok/s needs
p >= 0.94" no longer holds: it needs a draft that costs 1.5 ms, which is a
bandwidth change rather than an accuracy one.

**Superseded by §64.** The head *was* bound, the slice *was* built, and the
measured accept rate is 0.36 at depth 1 and 0.17 against the slice — not 0.8.
Every number in this section's last three paragraphs is replaced by §64's table.

## 64. The MTP head is bound, correct, and still a loss

§62 projected 35.5 tok/s and §63 projected 44.4 from an MTP drafter whose accept
rate was *assumed* at p = 0.8. This section binds the head, measures p, and
reports that the projections were wrong by more than their own error bars: at the
measured rate every speculative shape is slower than one row per step.

**Binding.** `model-mtp.safetensors` is its own file and is not named in
`model.safetensors.index.json`, so the text shards' reader cannot reach it;
`mtp_open()` maps it directly (8-byte little-endian header length, JSON, then
each tensor by name with dtype, shape and `data_offsets` checked against the file
length). Fifteen tensors, 486.6 MB: `fc[4096,8192]`, one GQA layer (16 q heads /
4 kv heads, head_dim 256, `q_proj[8192,4096]` because of the interleaved output
gate, partial rope 0.25, per-head q/k norm), SwiGLU at 12288, and no output
projection of its own — the vocabulary head is shared. It has to open **before**
`q35_gpu_open`, because the GPU adopts its two cache buffers by
`newBufferWithBytesNoCopy`.

One `mtp_one()` now runs a single position (fuse, attend, SwiGLU, block output
left in `g_m.res`) and both callers use it: the chain and the cache fill. Three
kernels were added. `q35_gemv_rows` is `q35_gemv` with the row ids read from a
device buffer. `q35_embed_id` reads the token from a device buffer, so a
proposal's token never goes back to the host before the next proposal reads it —
a round trip is 0.93 ms against a 1.6 ms block. `q35_argmax` returns an index and
breaks ties towards the smaller id, like the host scan it is compared against.

**Two things the file does not state, both measured.** Whether the fusion puts
`emb` first or `h` first: inverting it takes acceptance to zero (0 of 89), so
`emb` first is right. And which copy of the text model's hidden state `h` is —
the residual stream, or the final-normed copy the vocabulary head consumes:
**39.0% against 35.7%** at depth 1 for the normed copy, which is now the default
(`CWICK_Q35_MTP_POSTNORM=0` selects the other arm).

**The block's cache.** A draft attends over keys, and at the first proposal of a
fresh sequence the cache held exactly one key — itself, which is not a shape the
layer was trained on. Rows are now indexed by absolute position over a 512-row
window (`CWICK_Q35_MTP_WIN`), and `q35_mtp_sync()` writes the rows for positions
the text model has already decided on, straight from the hidden states the verify
pass left in `g_s.x`: one block run per accepted position, 1.6 ms each. When the
window fills, `mtp_compact` shifts the live rows down — the rotation was baked in
when the row was written, so the shift is exact rather than approximate. The
prompt's rows are still not back-filled (that needs the hidden states prefill
discards), so the window starts at the last prompt position.

The sync is worth **35.6% against 31.5%** at depth 1 for its 1.6 ms per step. It
does nothing for depth 2 or 3.

**Measured acceptance** (200-token prompt, 161 generated, greedy, one run each,
by depth — depth d is only reached when every earlier proposal was accepted):

| arm | depth 1 | depth 2 | depth 3 |
|---|---|---|---|
| full 2.03 GB head | 35.6% | 12.5% | 0.0% |
| 4096-row candidate slice | 16.8% | 4.3% | 0.0% |

Acceptance saturates after the first link. Chaining is exactly why one would
expect otherwise: proposal i+1 is fed this block's own residual instead of the
text model's hidden state, and the checkpoint was trained for one depth only. The
useful shape is therefore one proposal, and that is what the numbers say.

**Cost of a row** (same prompt, `CWICK_Q35_VERIFY`):

| rows | 1 | 2 | 3 | 4 | 5 | 9 |
|---|---|---|---|---|---|---|
| ms | 48.4 | 57.4 | 58.2 | 60.7 | 64.0 | 83.5 |

Leaving one row costs **+9.0 ms**; every row after that **+1.6 ms**. That column
is the real obstacle, and it is not bandwidth: four rows read exactly one copy of
the weights just as one row does. The first extra row is 19% of a whole pass.

**End to end** (same prompt, one run each, `tok/s this arm` against `20.66`):

| arm | tokens/pass | ms/pass | tok/s |
|---|---|---|---|
| one row, no drafter | 1.00 | 48.4 | 20.66 |
| SPEC=1, full head | 1.35 | 66.9 | 20.21 |
| SPEC=2, full head | 1.39 | 74.7 | 18.59 |
| SPEC=1, 4096-row slice | 1.17 | 60.8 | 19.03 |
| SPEC=2, 4096-row slice | 1.18 | 63.4 | 18.54 |
| SPEC=3, 4096-row slice | 1.18 | 67.8 | 17.33 |

Every arm loses, so `CWICK_Q35_SPEC` stays off by default.

**Not 40, and not 35 either.** One weight read per token is the constraint; 40
tok/s means 1.93 tokens per pass, and this drafter's ceiling is 1.35. §62's
"p >= 0.94" and §63's "make a draft cost 1.5 ms" had the right algebra and the
wrong input: a draft against the slice *does* cost ~1.7 ms, and the pass is still
60.8 ms because the second row costs 9 ms — while p is 0.17, not 0.8. Both
sections' 35.5 / 44.4 / 43.0 are replaced by the table above.

**Correctness.** Output text is byte-identical across `CWICK_Q35_SPEC=0/1/2/3` on
the same seed: only a plain argmax loop may accept drafts, so a sampled request
never speculates at all, and a greedy one emits exactly what the single-row path
emits.

**What is left, in order of size.** First, make a two-row verify cost 50 ms
instead of 57.4 — the +9.0/+1.6 step is a kernel-shape anomaly (four rows cost
one weight read either way), and it taxes every speculative shape at once.
Second, improve the candidate set: on prose the recent-token window holds about
60 distinct ids, so the slice gives up 19 points of acceptance to save 5.5 ms;
seeding the set with the top-K rows of the previous step's own logits, which are
already on the host, is the cheap way to recover most of that and would put a
one-draft step near 21.4 against 20.7. Neither one changes the answer to 40.

## 65. The second row is 8 ms, not 9, and the drafter still does not pay for it

§64 ended with two items in order of size: make a two-row verify cost 50 ms
instead of 57.4, and improve the candidate set so the slice stops giving up 19
points of acceptance. Both were done. The first recovered 2 ms of the 9, the
second recovered 2 points of the 19, and neither changes the conclusion.

**Where the second row's cost actually is.** The first thing to rule out was the
k-row GEMV, since that is the only part of the pass that changes kernel:
`CWICK_Q35_GPU_TILED1=1` runs one row through it and costs **49.5 ms against
48.0** — 3%. So 7 of the 9 ms were in everything else, which at n ≥ 2 ran once
per row: two norms, two residual adds and the SwiGLU multiply per layer, plus
the DeltaNet gate pair. Nine and a half dispatches per layer, 0.21 ms per layer
by the `CWICK_Q35_GPU_MAXL` slope (8/16/32 layers gave +2.66/+4.51/+7.71 ms, so
0.21 ms per layer and 1.0 ms of fixed cost).

**Batching them** is a pure change of launch count — same arithmetic, same sums,
same order — so it cannot move a single bit. Five kernels were added
(`q35_rmsnorm_rows`, `q35_add_rows`, `q35_silu_mul_rows`, `q35_l2norm_rows`,
`q35_gdn_gates_rows`), each taking the row from the grid and the row stride as an
argument, and `CWICK_Q35_ROWFUSE=0` puts the per-row loop back so the claim stays
testable in one binary. Result, back to back in one session at n = 2: **56.61 →
55.81 ms over 199 passes**. So a dispatch is worth about 5 µs, not 22, and the
9 ms was never launch latency.

**RP for n = 2.** `cwick_q35_gemv_n_rp` was a constant 2 because the table it
came from was taken at n = 4..8, where RP4 spills. At n = 2 there are eight live
accumulators, and the extra activation reuse is free: **54.61 against 55.67 ms**
at RP4. n = 3 already regresses (54.41 at RP2 against 61.35 at RP3), so the
policy is `n <= 2 ? 4 : 2`. RP moves no FMA within an accumulation, so the
bitwise invariant `CWICK_Q35_GPU_CHECK` asserts is untouched — the two-row check
still reports `max|gpu-cpu| = 0.00000` on every position.

**What is left is not activation re-reading.** A threadgroup covers RW x RP = 8
weight rows, so the activation slice is read `cols/8` times; widening the group
should cut that. `CWICK_Q35_GEMVN_RW` was added to sweep it and the sweep says
no: **55.9 / 54.9 / 56.3 ms at RW 4 / 6 / 8**, inside the run-to-run spread, and
at n = 1 the same kernel goes from 49.5 to 52.1 ms. The remaining 6 ms per extra
row is not understood and is no longer worth chasing — with everything else
fixed, a two-row pass is 54.6 ms, and that number only matters if a draft can be
right often enough to fill the row.

**The candidate set.** §64 proposed seeding the shortlist from the previous
step's own logits, which the verify has already copied to the host. Done:
half of `CWICK_Q35_CAND` is the recent window and the other half is every row
within `CWICK_Q35_CAND_T` (default 2.5) of the top logit, capped at
`CWICK_Q35_CAND_LCAP` (default 512). Acceptance at depth 1 goes **16.8% → 18.5%**
— and a draft pays 8 KB of head per ranked row, so 2048 rows cost 4 ms and 512
cost 1. The wide version bought 2.6 points for 4 ms, which is a loss; the narrow
one buys 1.7 for 1 ms, which is a wash. Threshold rather than top-k because a
sharp distribution contributes a handful of rows and a flat one fills its half,
and neither case needs a sort.

**End to end, one session, 200-token prompt, 161-200 generated, greedy:**

| arm | tokens/pass | ms/pass | tok/s |
|---|---|---|---|
| one row, no drafter | 1.00 | 48.1 | **20.74** |
| VERIFY=2, no drafter | 1.00 | 56.2 | 17.79 |
| SPEC=1, 512-row slice | 1.18 | 60.0 | 19.72 |
| SPEC=1, 2048-row slice | 1.18 | 59.2 | 19.94 |
| SPEC=1, full 2.03 GB head | 1.35 | 66.8 | 20.19 |

Every arm still loses. The drafter is now good for 0.35 tokens per pass at best
(the full head), and the row it fills costs 8 ms, so the crossover sits at
0.35 > 8/48, i.e. acceptance above 16.7% for the slice — which it does not have.

**Why 40 is not reachable on this checkpoint.** One weight read is 48 ms at this
machine's 341 GB/s, so 40 tok/s needs **2.08 tokens per pass**, and the
acceptance that would produce it is the sum of p^i with p >= 0.52 at every depth.
The full head delivers 0.356 / 0.125 / 0.0, which sums to 1.40 and is the ceiling
for any k, because the block was trained for one depth and chaining its own
residual is out of distribution. Nothing in this repository can raise a rate that
is a property of the weights. What would: a checkpoint whose MTP head is trained
on multi-depth chains, or a second model small enough to draft in 2 ms. Neither
is in the model directory.

**Correctness.** `make test` PASS (678 assertions). Output text is byte-identical
across `CWICK_Q35_SPEC=0/1/2` and across the two-row check's `max|gpu-cpu| =
0.00000`. `CWICK_Q35_SPEC` stays off by default.

## 66. `serve` died on the first request of every session: a 256x overrun in reset

The log stopped at `[req 1] generation slot 0` with no error, and the process was
gone. Two crash reports from the same morning have the same stack:

```
__bzero + 12400
q35_engine_reset
cwick_engine_reset
handle_client
client_thread
```

`EXC_BAD_ACCESS / SIGBUS / KERN_PROTECTION_FAILURE`, writing at the *start* of a
~11.86 GiB region (`bytes after start: 0`). `__bzero` is the large-block memset
path, and the caller clears the draft head's KV caches.

**Root cause.** When the MTP cache became a 512-row sliding window (§64),
allocation, `bind_tensor`, the cache accounting and `mtp_compact` were all moved
to `mtp_win` — but `q35_engine_reset` was not:

```c
/* allocated: mtp_win rows */
m->mtp_kcache = xcalloc((size_t)m->mtp_win, (size_t)kvw * sizeof(float));
/* cleared: max_seq_len rows */
memset(qe->m.mtp_kcache, 0, (size_t)c->max_seq_len * ...);
```

512 MiB written into each 2 MiB allocation, twice, i.e. **256x**. The overflow
ran out of the heap into a mapping that was not writable and the kernel killed
the process on the first page of it. Nothing printed, because the fault is
immediate and there is no handler.

**Why the CLI never showed it.** `q35_engine_reset` is reached through
`cwick_engine_reset`, which only `handle_client` calls — the server resets state
between requests. `cwick run` builds one engine and never resets it, so every
`run` measurement in §62–§65 was unaffected. The regression was introduced with
the sliding window and could only ever appear in `serve`.

**Fix.** Clear `mtp_win` rows, guarded on `mtp_win > 0`. Verified on the same
configuration the crashed server ran (context 131072, 80 GiB prefix cache): a
125 KB body / 25298-token request returns 200, three consecutive request cycles
do not reset-fault, and 8001 serves again after `start.sh`.

**Checked while there, and not defects.**

- *A fixed ~1.1 s of prefill on a short prompt* (6 tokens, CLI and server
  alike). That is one full read of 17.98 GiB of weights — a dense 9B model has to
  touch every matrix to produce one token, and the CPU prefill path runs it at
  ~16 GB/s against the 120 GB/s the machine can do. It disappears into the
  average on long prompts: 25298 tokens measured 74.4 tok/s.
- *`prefix_cache_queries_total` at 0 in one test* was my own configuration — the
  cache only opens when `--prefix-cache-dir` is given. With it, the counters are
  right: queries 2, hits 1, 25297 tokens reused, and the same 25298-token prompt
  went from **340 s of prefill to a 0.206 s TTFT** on the second request.
- *79.65 of 80.00 GiB prefix cache used* is the cache doing its job, not a leak:
  `evict_to_fit` runs on open and after every store. It is worth knowing the
  arithmetic though — one 25K-token request writes three snapshots (the prompt
  prefix, the off-by-one copy, and the prompt plus the generated tokens), about
  1.3 GB each for this checkpoint, so 80 GiB is roughly 20 turns of an agent
  session before the oldest entries start going.
- *The `--trace` line in the log* is `start.sh`, which passes it deliberately
  (goal.md asks for the flag); prompts and outputs are truncated when printed.

**The lesson is mechanical.** A buffer has four places that must agree on its
size: the allocation, the reset, the accounting, and anything that serialises it.
§64 moved three of them. A `_Static_assert`-style guard is not available across
those sites, so the cheapest defence is that reset should never take a size from
the config when the allocation took it from a runtime field — `mtp_win` is a
field, `max_seq_len` is the config, and that mismatch is exactly what the two
lines above disagree on.

## 67. Reading the serve log as a cache-hit problem, not a latency problem

`logs/cwick-serve.log` (10:07:25 to 10:25:25, three 42K-token agent turns, all
200, no warn/error/failure lines) looks healthy if you read only the status
codes. The numbers say otherwise: every turn logged

    prefix cache hit: 31785 of 42158 prompt tokens replayed, 10373 left to prefill
    ...
    prefix cached: 48087 tokens (251 ms, background)

i.e. each turn stored a full 42K/48K snapshot and the *next* turn still replayed
only the same 31785-token static system+tools head. The three turns were
189.2 s, 603.6 s and 196.5 s end to end; roughly 178 s of each was re-prefilling
the ~10K-token tail that the snapshot it had just written should have covered.
`logs/cwick-serve.log.crash-20261113` is a separate, earlier failure — the log
stops at `[req 1] generation slot 0` with no response, which is the §66 SIGBUS.

**Root cause: the victim was the entry the store had just written.** The old
cost function ranked by `longest - pos`, making the longest entry the most
evictable. A growing session's newest snapshot is by construction the longest
thing in the cache, so `evict_to_fit` ran with the new file as its own victim and
unlinked it microseconds after reporting success. The cache was 79.13 of 80.00
GiB with 184 snapshots, so *every* store triggered an eviction — the bug was not
intermittent, it was every turn.

The victim is now chosen in two tiers (`hits > 0` entries are kept over ones
never replayed, then last-use decides), with two absolute protections: the entry
just written, and the shortest prefix (the shared base of all longer prompts,
which on a cold start has not earned a hit when the first snapshot arrives).
Everything else stays eligible — including hit entries, ordered by last use — so
the cache cannot grow without bound. When only the two protected entries remain,
one file of overshoot is allowed rather than deleting either. Two regression
tests cover 24 later snapshots and a monotonically growing session; `make test`
is 686/686 PASS.

**A second, build-level fault was found while verifying.** The pipeline enum had
26 entries but `P_NAME[]` had 24, and the two extra names belonged to
group-shared attention kernels that had no dispatch function and no call site.
The running binary was built before that enum change, which is why every start
in the log still reports `24 kernels`. Rebuilding would have made `q35m_init`
walk past the end of the name table into NULL and fail GPU setup — a silent
regression that no test covers. The dead kernels and the enum entries were
removed; a standalone Metal probe now compiles the exact production shader
source and builds all 24 pipelines: `OK: 24 kernels, 0 bad`. That probe is a
one-off verification, not a shipped tool.

**The metrics scrape was healthy but incomplete.** `/v1/metrics` exposed 44
families; the snapshot taken at `uptime_seconds 12.5` shows every counter at 0
simply because no request had finished yet, and `process_footprint_bytes`
28.50 GB matches 17.98 GiB of weights plus 8.44 GiB of state plus the 464 MiB
MTP head. The real gap is that goal.md forbids swap and compression, yet the
only evidence was one startup line (`host swap 0.86 of 2.00 GiB used,
compressor ... since boot`). Five host-wide gauges now expose it continuously:
`cwick:host_swap_used_bytes`, `host_swap_total_bytes`, `host_wired_bytes`,
`host_compressions_total`, `host_swapouts_total`. They are host-wide because the
VM APIs have no per-process view: a nonzero level is not by itself a violation,
the delta over a scrape window is. A standalone render probe confirmed the
values match the startup line (928514048 / 2147483648 / 1225496103).

No model precision, weight, backend or context behaviour changed in this pass.

## 68. An empty answer was two state machines that never met

A 50,783-token agent turn ran for 773.9 s and returned `content: ""`. Every
surface said success: HTTP 200, `finish_reason: "stop"`,
`request_success_total` +1, and not one warn or error line in 2,161 lines of
log. The only trace was the access-log tail `think=25 tools=0 content=0
think_cap=512`, and the trace block that showed why:

```
<tool_call>
<function=Bash>
</function>
</tool_call>
```

The model had decided to run a command, wrote the call while still inside its
thinking block, and stopped. It never emitted `</think>`.

**Four layers each did the reasonable thing.** `split_reasoning()` splits on
`</think>`; with no close tag it files the whole text under reasoning and hands
back an empty content. `cwick_tools_extract()` reads the answer only, so it saw
no call. The finish reason is derived from the call count, so it said "stop".
The Responses rule "a textless turn still needs a message item" then sent the
client a message whose content was `""`. Nothing failed; the user got a blank
bubble and no action.

**The mechanism needs thinking to be on, and that is the point.** With the
block closed by the template (`<think></think>`, thinking off) the answer starts
in content (`closed_at_start`), so a `<tool_call>` can only be read as a call.
Of the 13 turns on that instance, the only empty one was one of the 7 that
thought; the 8 that did not never produced an empty answer. Turning thinking on
by default would not have prevented this turn — it would have made every turn a
candidate. A/B on the same checkpoint, 3 turns at `think_cap=512`, all closed
their block normally and returned `tool_calls`: this is a tail risk that grows
with context and tool count, not a broken switch.

**Two changes, both narrow.** The server now recovers a trailing call when, and
only when, all of these hold: the turn is chat with tools advertised, the block
is genuinely unterminated (no `</think>` anywhere in the output), the text ends
(after whitespace) with whole `<tool_call>` blocks, every block names a declared
tool, and every block carries at least one argument. The argument rule carries
the weight: the observed call had none, and promoting that shape would have
handed the client `Bash({})` — a request certain to fail, which an agent answers
by retrying. A second rule refuses when an earlier call would be left behind in
the reasoning text, because a silently dropped action is worse than a visible
empty answer. `cwick_tools_trailing_calls()` decides all of it, and nine unit
cases pin the matrix (recovered, whitespace, zero arguments, undeclared tool,
call not at the end, two consecutive calls, prose between calls, no call, folded
name).

When nothing qualifies the turn is still empty — but no longer silent:
`cwick:empty_answers_total` counts every finished request that generated tokens
and delivered neither content nor a call, and the access log gains an
`empty answer:` line at info level. That counter is the only signal this failure
produces, so it belongs next to the failure rate on any dashboard. A turn that
was recovered carries `promoted=N` on its access-log line, because the phase
counter reads markers out of the token stream and therefore still reports
`tools=0` there.

Streaming needs no special case: the arguments were already forwarded as
reasoning deltas and no `tool_calls` delta was sent, so `stream_finish()`
announces the parsed call in its closing chunk (`tool_std_settle()` is false
when nothing was streamed), and `resp_stream_finish()` emits the function_call
item the same way. The duplicated text is reasoning, which clients render as
thinking, not as the answer.

**What this does not fix.** The same session re-prefilled a 45K prompt it
already had on disk, 688 s of the 773.9 s it spent. That is the cache defect in
§67, not this one.

## 69. A turn writes three snapshots and §67 protected only the last one

§67 removed "the longest entry is the most evictable". Production says that was
not enough. Across the 53 chat requests of 14:09-14:41 the cache sat at
79.90 of 80.00 GiB with 49 snapshots; the disk held the contiguous chain
`50458 -> 54690` plus the very last `55023`, while `54726 .. 54965` -- the
eight most recent turns -- were all gone. Of 50 warm requests, **none was a full
hit**: the replayed length stopped at "the previous prompt minus two tokens"
every turn, re-prefilling 81-318 tokens (2.4-7.1 s), 100,522 tokens in total,
and the last three turns degraded from 81 to 160 / 239 / 318 (`left`, 2.9 s ->
7.1 s of prefill). The small-prompt sessions in the same window were fine: six
of eight warm requests fully hit, 0.35 s of prefill on average.

**Why: a turn writes a batch, and the guard covered one member.** A turn stores
three snapshots -- the full prompt P, the twin at the thinking-block boundary,
and the conversation once generation ends -- but `last == c->seq` protects only
the **last** one written. So the third store evicted the first two, the second
evicted the first, and the next turn's own store evicted the third. The twin is
the only entry the next turn actually agrees with (a client that does not echo
`reasoning_content` re-renders the assistant turn, so it matches up to the block
boundary and no further); it lived for exactly zero turns. Above ~45K tokens a
session therefore only ever had one frozen base that could never advance.

**The fix promotes "the entry just written" to "the batch written this turn".**
`cwick_pfx_lookup()` records `batch_mark = seq + 1` on entry and
`evict_to_fit()` skips every entry with `seq >= batch_mark` -- everything stored
between the last lookup and the next one, which is exactly the window in which
it is about to be read back. Older entries stay eligible (hit entries still rank
by last use), so a long session still cannot grow without bound; when only
protected entries remain, the whole batch may overshoot the cap instead of one
of its members being deleted.

The regression test `prefix cache: a turn's boundary twin survives the turn's
other two stores` pins it: one turn stores P / twin / conv, the next turn
queries with a prompt that diverges after the boundary, and the hit must be the
twin. Disable the guard and it fails (`next-turn hit=3`, the base, instead of 4;
two entries left); enable it and it passes. `make test` 699/699 PASS, no
warnings.

**Measured A/B** (same machine, same 8-9K-token growing session, 10 turns,
`--prefix-cache-gib 2` so eviction pressure arrives by turn 6):

| policy | `left to prefill`, turns 3-10 | prefill of the last turn |
|---|---|---|
| before | 26, 26, 26, 36, **69, 102, 135, 168** | 3467 ms |
| after | 26, 26, 26, 36, **35, 35, 35, 35** | 1669 ms |

Before, every turn from the seventh paid "previous generation + new turn + 2"
tokens again -- the same cadence the live 45K session showed (81 -> 160 -> 239 ->
318). After, it is a constant.

**The other snapshot in the same turn that nobody reads.** P ends *inside* the
injected thinking block, which the next turn never resends when it re-renders
the assistant message, so it matched on zero of 50 warm requests while costing
1.63 GiB and one eviction per turn. The twin covers it for the price of
re-prefilling two tokens. `pfx_save_now()` therefore computes the twin first and
does not submit P at all when there is one (verbose log: `prompt snapshot at N
tokens not stored: the boundary twin at N-2 covers it`), which takes the writes
per turn from three to two (measured: 30 -> 20 stores over 10 turns).

**And a 4x instrument error on the way past.** `host_wired_bytes` was computed
at 4096 bytes per page, but Apple Silicon has used 16 KiB pages since M1, so the
one gauge meant to prove "no swap" under-reported by exactly 4x: 8.37 GB against
33.4 GB actually wired. `src/mem.c` now reads and caches
`sysconf(_SC_PAGESIZE)`; the same instant reads 33.59 GB.

No change to model precision, weights, backend or context behaviour, and nothing
in §68 was touched.

## 70. A dropped download is a retry, not an error the user has to notice

The model tab already had the hard parts: a local inventory under `./models`, a
read-only hf-mirror catalog search, per-file `.incomplete` files with Range
resumption, 64-bit sizes for the multi-hundred-GiB checkpoints this machine
actually downloads, and recursive delete confined to the library root. What it
did not have was the one behaviour a long download cannot live without: when
the connection drops, the job used to go straight to `error` and stay there
until somebody clicked "断点续传".

That is not an edge case here. A several-hundred-GiB checkpoint over a mirror
sees a dropped transfer as routine. Measuring it was simple: point
`CWICK_HUB_ENDPOINT` at a local catalog that serves half a file and closes the
socket. The old worker entered `error` in 2 s and was still `error` 10 s later,
with `transfer closed with 3145728 bytes remaining to read` — correct diagnosis,
useless response. The bytes already on disk were intact; they just needed a
human to ask for them.

**What changed.** Both fallible network phases now share one retry path. The
file listing (`/api/models/<repo>/tree/main`) and each shard download are
retried in the worker with exponential backoff — 1 s, 2 s, 4 s … capped at
30 s — up to a budget of five attempts by default (`CWICK_HUB_RETRIES`, 0
disables). Every retry re-enters `download_file()`, which reads the size of the
existing `.incomplete` and asks only for the remaining bytes with a Range
request, so an attempt never re-downloads what is already on disk. A successful
file or listing resets the counter: the budget is per stalled step, not per
job, so a 200-file checkpoint that fails once early is not punished at file
150.

The wait is interruptible in 100 ms ticks. `pause`, `cancel`, and server
shutdown take effect during a backoff instead of after it, which matters when
the cap grows to 30 s. A retry keeps the job in `running`, not `error`: from the
outside a transient outage is a slow download, not a failed one. Only an
exhausted budget finishes as `error`, and the message keeps the last real cause
with a `gave up after N attempts` suffix — it never replaces the diagnosis with
a generic summary.

`GET /v1/hub/downloads` now exposes `attempts`, `max_attempts` and
`retry_in_seconds`. The model tab renders a `自动重试 N/M · X s 后` badge while
waiting and an `已重试 N 次` badge after a recovered attempt, so a pause that
would otherwise look like a hang has a visible reason. The heading says the
same thing in words: default 5 retries, resumed from `.incomplete`, no duplicate
bytes.

**Measurement.** Two runs against the controlled catalog, same binary,
`--backend metal` throughout (the download path does not touch the engine):

| scenario | before | after |
|---|---|---|
| one mid-stream drop, then a clean retry | `error` at 2 s, still `error` at 10 s | `retry 1/5 in 1 s` in the log, `done` at 4 s, 5 MiB complete |
| every transfer dropped, `CWICK_HUB_RETRIES=2` | `error` immediately | `running`, `attempts` 1/2 → 2/2, then `error` with `gave up after 2 attempts` |
| every transfer dropped, `CWICK_HUB_RETRIES=8` | — | `running`, `attempts 5/8`, `retry_in_seconds` counting down, UI badge visible |

The last row was checked in a real browser: the "正在下载" panel showed
`mock/retry 下载中 自动重试 5/8 · 1 s 后` alongside progress, current file and
the pause/cancel buttons. A separate probe confirmed that `pause` during a
16 s backoff moves the job to `paused` without waiting out the timer.

`make test` is 702/702 (the three new checks pin the default budget, the
override and the zero-disables case without needing a network). The default
stays 5 because that covers a mirror blip plus one re-dial at a 30 s ceiling;
an operator on a stable link can set 0 and get the old fail-fast behaviour.

## 71. The image output was two problems, and neither was the one the docs blamed

The image endpoints had been returning HTTP 200 with an unusable picture since
they were wired: at 256×256 an all-black frame, at larger sizes textured noise.
`docs/IMAGE.md` listed the suspects — the conditioner's chat template and
multi-dimensional rotary positions, the DiT's token refiner and three-axis
RoPE, the row order of the packed sequence — and inference sat behind
`CWICK_IMAGE_UNVERIFIED=1` so that "the pipeline completes" could not be
mistaken for "the model works".

The list was wrong. Two things were going on, and the first one hides the
second.

**The checkpoint does not work below about 4 MP.** Its released resolutions
start at 4096×4096. Asking the reference implementation for 256×256 returns
the *same* textured noise this port produced, which rules out a porting error
by construction: that size is outside anything the weights were trained on.
`h3_canvas_size()` now scales a request up to `min_generation_area`
(4194304 by default, `CWICK_H3_MIN_AREA` overrides), keeps the aspect ratio,
rounds both sides to a multiple of 32, and Lanczos-3 resizes the decoded image
back to the requested size. A 512×512 request therefore costs one 2048×2048
generation and returns it downsampled.

**Then the stages could be compared one at a time.** With the canvas fixed,
`CWICK_H3_DUMP` writes every intermediate and `CWICK_H3_LATENT` feeds a
reference latent straight into the decoder, so a divergence can be attributed
to a stage instead of to the stack. Fixed seed, 256² canvas, rel_rms against
the reference:

| tensor | rel_rms | cos |
|---|---:|---:|
| conditioner, 50 layers | 1.14e-2 | 0.999985 |
| timestep embedding | 3.0e-6 | 1.000000 |
| token refiner | 1.26e-2 | 0.999921 |
| packed sequence | 4.4e-3 | 0.999991 |
| DiT block 0 / block 49 | 6.4e-3 / 4.2e-2 | 0.999987 / 0.999125 |
| final velocity | 2.9e-2 | 0.999591 |
| decoded latent | 5.4e-2 | 0.998514 |

Nothing here is a defect. The conditioner's 1.14e-2 is the largest relative
number and the smallest actual error: 99.4% of its energy sits on the 1% of
channels with the biggest activations, but it is uniform across tokens and
channels (correlation −0.096 with row index, −0.114 with magnitude), which is
what float noise looks like and not what a wrong template looks like. Two
attempts to remove it by rounding — the residual stream in bf16, then thirteen
individual conditioner operators — made it worse (1.14e-2 → 2.4e-2 and
1.9e-1) and were reverted. Feeding the reference conditioner output in place
of this port's moves the decoded latent by 5.45e-2 → 5.57e-2, which says the
remaining error is in the transformer's accumulation order, not in
conditioning.

**The DiT was also ten times slower than it needed to be.** Its attention was
a hand-written loop that streamed a whole row of keys per score: 4.2 GFLOPS
measured, and 60% of a 12.0 s step at 512². Replacing it with two
`cblas_sgemm` calls per 256-row band, a vDSP softmax with the max subtracted
before `exp` (a raw score above 88 overflows fp32 and turns the row into NaN),
and folding `1/sum` into the 128-wide output instead of normalising the score
block took the 512² DiT from 344.7 s to 35.7 s with bit-identical output. At
2048² the same change is gemm 95 s → 64 s and attention 129 s → 64 s per step;
widening `H3_DEQ_TILE` from the 128-row default to 512 once the activation
stack is at least 1024 rows tall is what carries the GEMM to 2.47 TFLOPS,
close to the 2.7 TFLOPS a bare `cblas_sgemm` reaches on these shapes.

The gate is gone: `h3_generate()` no longer consults `CWICK_IMAGE_UNVERIFIED`,
and `start.sh` no longer sets it. The one request it still refuses is an image
edit, and that refusal is about a path that genuinely does not exist (§73).

## 72. Tiling the decoder was the correct answer and four times the speed

Every image came back with a regular grid over it. The decoder had just been
split into a whole-image pass, so the first suspicion was the transformer
again — but decoding a *reference* latent with the DiT skipped reproduced the
grid exactly, which put the fault in the last 24 layers and nowhere else.

The reason is in the released module: `MiniMaxH3ImageViTDecoder2d` ships with
`use_tiling = True` — 256 px tiles, a 64 px minimum overlap, linear
cross-fade — and its docstring says the released images *are* the blended-tile
ones, so turning tiling off changes the output. A 128×128 latent decoded in
one pass puts a ViT that was released for a 16×16 window far outside that
window, and the per-cell modulation it returns is the grid. A 16×16 latent is
exactly one tile, which is why the earlier 256² alignment work never saw it.

`h3_vae_decode()` now reproduces the released geometry (`_split_tiles`: count
tiles until their overlaps can span the length, then hand the slack out one
cell at a time so every boundary stays cell-aligned — at 2048² that is 11×11
tiles with 80 px overlaps and 64 px on the last two boundaries), the
`1 - i/e` / `i/e` weights, and the stitch order: cross-fade against the raw
tile above, then against the raw tile to its left, then drop the far edges.

Three things fell out of it:

- **Memory.** The first tiled build was killed with SIGKILL (exit 137) at tile
  108 of 121 with a 112 GiB peak. Every tile was re-copying all 391 decoder
  tensors out of the mapping through `h3_standalone_f32` — 1.6 GiB of fp32
  per tile, 121 times. Those tensors are read-only; the loader now returns a
  pointer into the mapping, resolves all 391 once per image, and parses the
  header once instead of 24 times per tile. Peak RSS 112 GiB → 48 GiB.
- **Speed.** Attention inside a tile is 16×16 rather than 128×128. VAE time at
  2048² went from 114028 ms to 26469 ms.
- **Fidelity.** `c_rgb` rel_rms went from 8.2e-2 to 2.8e-3; 99.96% of pixels
  are identical to the reference after the 8-bit truncation, at 79.2 dB PSNR.

Two details each cost a pass. The first is that the 8-bit conversion belongs
at the very end, not inside each tile: rounding the tiles before blending them
costs up to two levels across every overlap band. The second is the blend
itself. The weight belongs to the offset *inside* the band, so the horizontal
branch has to carry `i` down to the pixel; blending the whole `e`-wide strip
once per offset leaves only the last offset's weight applied to the strip, and
because `cur <- a*(1 - i/e) + cur*(i/e)` is idempotent once `cur == a`, the
band silently became a hard cut on the left tile's side. What gave it away was
that vertical bands were exact and horizontal bands were not — the vertical
branch writes one row per offset, the horizontal branch wrote the same strip
every time. Laying the raw tiles back out through the reference
`_stitch_tiles` in numpy reproduced the reference image to a mean absolute
error of 0.00001 in the band, so the tiles themselves were never in doubt; the
dump is kept as `c_rawrow%02d` under `CWICK_H3_DUMP` for the next time.

## 73. `images/edits` answered with a fresh image instead of refusing

The edit endpoint is supposed to return 501, because the reference-image path
(VAE-encoding the references, `<Picture N>` labels, the vision blocks) is not
bound. It returned 200 and a generated image.

The refusal is decided by `n_ref`, and `n_ref` is counted by the multipart
field extractor. That function walked the parts and returned as soon as it
found the field it was asked for — so anything positioned *after* `prompt` was
never seen, and `curl -F prompt=… -F image=@…` puts the image second. An edit
request therefore turned into a plain text-to-image generation that ignored
the reference entirely, which is precisely the failure the 501 exists to
prevent, and it did it silently.

The extractor now walks every part, counts the `image` parts and remembers the
first matching named field instead of returning from inside the loop. Both
field orders answer 501 in about a millisecond:
`image editing is not implemented: reference images are not conditioned on
yet`.

## 74. The edit path is one image, two size rules, two tag vectors

Binding `images/edits` means binding four stages that only exist when a
reference is present: the VAE encoder, the vision tower, an edit-shaped
conditioner, and a DiT layout with reference rows in front of the generated
ones. Each one had a place where the released pipeline does something that
looks like a detail and is not.

**Sizes.** One reference is fed to two consumers and they disagree. The VAE
copy targets a short edge in `[1024, 2048]`; the vision copy targets exactly
`1024`; neither is derived from the other, both scale from the original, and
neither is ever scaled to the generated canvas. A reference whose short edge is
already inside the band keeps its own size — which the packed layout cannot
take unless both sides divide by 32 twice, and every camera image hits that
case, so the port rounds each side to the nearest 32 (at most 16 px) instead of
raising.

**Two tag vectors, and they are not the same vector.** The conditioner's
`mm_token_type_ids` is 1 only on `<|image_pad|>`, which is what segments the
3-axis mrope: `<|vision_start|>` and `<|vision_end|>` are text there, so the
label and the start token form one text run. The transformer's `token_tags`
marks the **whole** vision block including both delimiters, and that is what
selects the AdaLN modality row. DiT step-0 velocity was 6.46e-1 off until the
second vector was threaded through separately; text rows are modality 1, image
rows 0, and a reference's vision block is image-tagged even though it lives in
the text stream. Nothing in the layout announces that: the two vectors agree
everywhere except at four `<|vision_start|>`/`<|vision_end|>` positions per
reference, so the error looks like a broken rope table.

**Timesteps.** The reference rows are noised to 0.999 and held; the generated
rows follow the sampler. `unique(sorted=True)` over `[0.999, t]` is what turns
that into two AdaLN rows, and which index each row gets depends on whether `t`
is above or below 0.999 — reversing it silently conditions the generated rows
at the reference level. The scheduler writes only `latents[num_reference_rows:]`,
and the output head computes all image rows and then drops the reference ones.

**Deepstack lands before `all_hidden_states`.** `hs[k]` is the layer-k output
*without* `ds[k-1]` added, because the addition happens to the hidden state
after the list entry is appended. Diffing against `hs[k]` and getting 4.9e-1 is
not a bug in the tower.

Alignment, 1024² reference and 1036 text rows, every number against the
released pipeline on the same seed:

| Tensor | rel_rms |
|---|---:|
| VAE encoder moments | 1.75e-6 |
| reference latent | 2.71e-5, 98.4% bit-identical |
| patch embed / `x0` / block 0 | 3.8e-7 / 3.5e-7 / 2.6e-6 |
| vision last layer / merger / three deepstacks | 8.5e-5 / 2.8e-5 / 7e-6…2.6e-5 |
| ids / types / tags, 1036 rows | zero mismatches |
| 3-axis mrope positions | zero mismatches |
| `hidden_states[1…3]` | 2.6e-5 / 2.1e-5 / 1.8e-5 |
| conditioner `hs50` | 4.5e-5 |
| reference rows, packed and noised | 4.4e-8 |
| step-0 velocity, 1088 image rows | 6.5e-3 |

The last row is the same magnitude as the text-to-image path, so the layout is
right and the rest is the transformer's accumulation order.

## 75. A null store one branch deep became `brk #1` 117 seconds into the request

The first real `images/edits` call killed the server: exit code 133, SIGTRAP,
and `serve.log` had no line for the request at all — the previous line was the
400 from the malformed-image test. The crash report put the PC on a `brk #0x1`
as the last instruction of `h3_vae_encode`, which is not something that
function contains.

`h3_vae_encode` wrote its latent dims back through `int64_t *o_lw, *o_lh`, and
both call sites passed `NULL` — the callers compute the dims themselves. Two
dead stores through null pointers. With both call sites inlining, the optimizer
constant-propagated `null` into the function, recognised the stores as UB, and
replaced the whole success block — the three stores and the `return true` — with
`unreachable`, which the backend emits as `brk #1`. So the trap appeared at the
*end* of a function on its success path, after 117 s of correct work, with no
allocation or bounds symptom anywhere near it.

Two things made this worth a section:

- The distinguishing evidence is cheap and decisive: compiling one source file
  to IR and counting `store …, ptr null` and `unreachable` finds this class
  before it runs. The whole module now reports zero of both.
- CLI testing had not covered it because the CLI run that exercised the same
  code path injected `CWICK_H3_REF_LATENT` and skipped the encoder. Every stage
  that a debug switch can replace still has to run once with the switch off.

The two parameters are gone; the callee writes only `*out_lat`.

## 76. Handing the image transformer's step to the GPU was two problems, and the arithmetic was neither

One sampling step at 4096 image rows issues 66.3 GB of weight reads and 1346.6
GMAC. The byte count is *identical* at 5 rows and at 4096 — it does not shrink
with the sequence — and on the CPU each of those bytes made three passes through
memory: the projection kernel dequantised every weight tile into an f32 arena
and called `cblas_sgemm` on the copy. Measured, that is 12.6 GB/s of *weight*
throughput against 1.24 TFLOPS of dense arithmetic.

A standalone BF16 simdgroup kernel on the same shapes measures 3.8-5.4 TFLOPS,
so the work was to feed it the checkpoint's own bytes. Two kernels:

- **Projection**: a 64×64 output tile per threadgroup, 128 threads, four
  simdgroups in a 2×2 arrangement of 32×32 blocks built from 8×8 fragments, K
  walked in 16-wide blocks staged through 10 KB of threadgroup memory. The
  weight tile is staged in the checkpoint's own `[n][k]` order and loaded with
  the transpose flag, so 61 GiB of weights are never rearranged.
- **Attention**: one threadgroup per (64 query rows, head), key range walked in
  16-key blocks, no score matrix beyond one block. This is the term that grows
  quadratically: 64.5 s of a 136.6 s step at 4096 rows, and 933 s/step if the
  same shape is given a 2048² canvas worth of references.

Four things were worth writing down.

**A non-zero buffer offset cost 1.75x.** The same kernel and the same tiles:
passing the tensor's page-aligned lead as `setBuffer:offset:` measured 291.7 ms
on a 4103×5376×14336 projection; passing a buffer base of zero and adding the
byte offset inside the kernel measured 167.0 ms. The offset also has to be a
page for a no-copy buffer, while a safetensors payload is only 8-byte aligned,
so the argument form is the one that generalises.

**One view per shard, not per projection.** A step runs a few hundred
projections over the same 61 GiB. Building a no-copy view per projection faulted
that tensor's pages and kept the handle for the life of the process, so step
time grew with the number of steps already taken — 62, 68, 72 s over steps 2-4.
The shard mappings are adopted once at model open (3.1 s for 61.7 GiB, one page
walk) and the tensor offset rides in the argument block.

**The exponent needs the row maximum.** The first version took `exp(score)`
directly, justified by "every head is RMS-normalised before the rope". That is
true and irrelevant: the residual stream reaches four digits inside the first
block, so a score of a few times 10⁴ overflows fp32 and everything after the
first block is NaN — while `c_block0` still matched to 1.2e-3, which is what
made it look like an accumulation problem rather than a range problem. A random
probe cannot find this, because random inputs never produce a large score; the
probe now takes an input scale and reproduces the failure at 100× and 4700×.
The range fix is a first pass over the keys that collects the row maximum and a
second that exponentiates against it: a streaming pass would need the running
maximum, and applying its correction means rescaling the output accumulator,
which is exactly what a set of matrix fragments cannot do element-wise. It costs
2.3 s of a 74 s step.

**The straddling rows of a threadgroup were written by the wrong threads.** Two
independent bugs at the boundary of the last 64-row block, both invisible at run
lengths that are multiples of 64: the staging copy-out advanced by 128 instead
of the threadgroup's 256 threads, and the per-row normalise ran without a
barrier against the store it was scaling. The row-wise probe found them in one
run — `row 64…68 rel 0.86, row 68 rel 1.00` — and the fix is to let the owning
simdgroup carry its own rows out and scale them in the same pass. `rel_rms`
against a double reference is now 2.1-2.6e-3 at every length from 69 to 1024,
and length-independent, which is what distinguishes rounding from geometry.

Net effect at 4096 image rows: step 136.6 → 74.3 s, DiT 874 → 446 s, whole
request 949 → 536 s. Attention 64.5 → 9.0 s; projections only 63.7 → 54.3 s,
because the CPU path was already at 1.24 TFLOPS and the kernel reaches 1.51.
The decoded pixels differ from the f32-activation CPU path by 8.6e-2 RMS — BF16
staging of the activations, accumulating over 50 blocks × 6 steps, same scene
and composition. `CWICK_H3_GPU=0` keeps any of this from happening.

## 77. Wiring the transformer removes the page-reclaim tax, not the arithmetic

A sampling step reads the same 66.3 GB of transformer weights six times. The
checkpoint ships as two 62 GiB trees (text encoder and transformer), so with
124 GiB of mappings on a 128 GiB machine something has to give: the OS reclaims
weight pages between steps and the kernel refaults them in the middle of the
projection loop. `CWICK_H3_WIRE=1` calls `cwick_safetensors_make_resident()`
with `wire=1` on the transformer only. The text encoder stays reclaimable on
purpose — it is read once per request, and wiring both trees would ask for more
physical memory than exists.

A/B, same machine, same 1536×2048 generation, seed 7, `--backend metal`, six
steps, A run before B:

| | unwired | wired |
|---|---:|---:|
| Model open | 3.5 s | 26.9 s (61.74 GiB, 2.5 GB/s) |
| Projections, 6-step mean | 52.57 s | 44.93 s |
| Step, 6-step range | 62.7 … 79.9 s | 60.2 … 60.7 s |
| Step, standard deviation | 6.29 s | 0.17 s |
| DiT | 419.7 s | 362.2 s |
| Whole request | 506.8 s | 449.6 s |

The mean gain is 13.7% on DiT; the standard deviation collapsing by 37× is the
actual diagnosis. An unwired projection spends anywhere from 46.0 to 61.1 s on
the same bytes because the page cache is revolting between steps, so the number
to plan with was never the mean. After wiring, all six land in 44.6–45.1 s.
The wiring cost is paid once per process: a one-shot CLI loses 23.5 s at open
and still finishes 33 s earlier, and a long-lived server pays it only once.

Two things did not change: the PNGs are byte-identical (same SHA-256), and the
text-encode and VAE timings are inside noise. The 61.74 GiB does not become
faster to read — it stops being reclaimed. This is why the earlier 2.2% delta
between two 3-reference requests under 45 GiB of swap was misleading: that
comparison held the cache state roughly constant, so it measured a workload
whose pages happened to be hot. The variance across a single unwired run was
the evidence hidden in plain sight.

## 78. One image directory, two entry points, no text-model coupling

`--img DIR` is now the documented way to configure the image checkpoint;
`CWICK_IMAGE_MODEL` remains as a launcher fallback and loses when both are
set. The directory is a lifetime choice copied into the server at startup,
so an HTTP client can ask for `MiniMax-H3-Image` but cannot choose an
arbitrary path. That keeps the switch endpoint's path handling unchanged.

The image pipeline lives beside the resident text engine on `server_ctx.h3`,
not in its generation slot. `GET /v1/models` therefore adds an entry with
`cwick.kind:"image"`, `GET /v1/models/MiniMax-H3-Image` retrieves it, and
`GET /v1/config` carries `image_model.dir` / `loaded`. A switch to
`MiniMax-H3-Image` opens the pipeline from the configured directory and
returns `switched` or `already_resident`; it returns 400 with no directory,
409 while a *text* swap is running, and 500 if opening the checkpoint fails.
Nothing releases the text model on that path, so the image endpoints can be
configured, advertised and loaded without taking generation offline. The
built-in settings panel reads the same config fields and posts the same
switch request.

For the console, language is a stored preference, not a page reload:
`data-i18n` / `data-i18n-ph` / `data-i18n-title` cover static nodes, a small
JS dictionary covers dynamic strings, and `setLang()` repaints, re-renders
messages with the newly translated empty state, replays the content fade and
moves focus to the view title. The entrance fade/slide, background gradient
flow, active-language glow and hover/focus transitions are plain CSS;
`prefers-reduced-motion` disables all of them. Verification was static —
matching key sets, a syntax check of the embedded script, and HTTP fetches of
the regenerated assets — plus `make cwick` and `make test`.

## 79. The model library must recognise a pipeline directory

The **模型** tab showed `MiniMax-H3-Image` as `未下载完` and refused to switch
to it. The inventory had exactly one definition of "downloaded": a
`config.json` at the checkpoint root, plus no `.incomplete` file. That is true
for the text checkpoints and false for an image pipeline, whose root carries
`model_index.json`, a fused-head shard and four submodule directories. The UI
was reporting the layout mismatch correctly and the state was still wrong.

Completeness is now format-aware: a text root still needs `config.json`; an
image root needs a manifest naming the image pipeline plus the transformer,
VAE, text encoder, tokenizer and fused-head shard, and neither may have a
partial file. The inventory adds `kind` and, for images, `served_id`, so the
button posts `MiniMax-H3-Image` instead of the repository path. The same
completeness gate is used by `cwick_hub_local_path()`, and the switch handler
falls back to the library when no `--img` was given, so a downloaded pipeline
can be hot-loaded without adding a second way to describe its directory.
Tests build a synthetic pipeline with no `config.json` and assert that it is
listed as complete and switchable while a text checkpoint with an `.incomplete`
file stays resumable.

## 80. One resident model, and the H5 image studio

The two pipelines used to coexist: `main` built the text engine before
`cwick_server_run`, and `--img` then mapped the image checkpoint on top of it.
On this machine that left roughly 63 GiB of image weights resident alongside
the text weights, which is exactly the condition a 128 GiB host cannot afford.
The rule is now one model at a time.

When `--img` loads successfully at startup, the text engine is released as soon
as every initialization step that reads it — metrics, tokenizer, catalog — has
copied what it needs, just before the listener opens. Switching to
`MiniMax-H3-Image` releases the text engine (all slots, tokenizer, Metal arena,
prefix snapshots, live-engine pointer) before opening the image checkpoint;
switching to a text id releases the image pipeline first, dropping the GPU's
no-copy views before unmapping the shards. Both directions were measured on a
live server: startup with `--img` left only the image entry `loaded`, a switch
back to text dropped the image entry to `loaded:false`, and a switch to image
again took 2.8 s. The catalog keeps the startup root as a switch-back target.

A render lasts about 40 minutes, so freeing its model mid-flight would be a
use-after-free. An `img_busy` counter under `gen_lock` joins the swap gate:
while either a text generation or an image render is in flight a switch is
refused with 409, and image requests are refused with 503 during a switch.
Both pipelines report their `loaded` state from the one object that is actually
resident, so the model list cannot claim two loaded models.

The third hosted app, `apps/imagegen`, is a self-contained H5 studio for the
two image endpoints. It takes a prompt plus optional face, outfit and pose
uploads; the edit mode posts a multipart body with one `image` part per upload
and calls `/v1/images/edits`, while the plain mode posts JSON to
`/v1/images/generations`. Progress shows elapsed time and the per-stage timings
from the response's `cwick` object (the API has no progress stream), and the
result is shown inline and downloadable as PNG. History is mirrored to the
per-app sqlite with a localStorage fallback, and the share button uses the same
LAN QR endpoint as the other apps. Embedded-route assertions were added to the
test suite, alongside `make cwick` and `make test`.


## 81. Image generation logged each phase to stdout, not just the total

A render is minutes long, but the image routes printed exactly one line — after
the PNG was already encoded. During a 40-minute request stdout was silent, so
there was no way to tell a progressing render from a hung one, and no way to
attribute the wait to a stage without parsing the JSON response afterwards.

The phase boundaries already existed in `h3_generate()` as the `h3_timing`
fields, so the fix is to log them where they are measured rather than to invent
a second progress mechanism. `CWICK_INFO("image", …)` now emits an arrival line
in `srv_image_route()` (mode, requested size, seed, reference count) and, inside
`h3_generate()`, one line per phase: tokenize, text-encode, pack, dit, vae and
png. The DiT phase — the long pole at six sampling steps — gets a line per step
with its own duration and cumulative elapsed time, plus a phase total. The
completion line in the server was expanded from `(dit, vae)` to all six phases.
No new flag was added: the shared logger already stamps the wall clock and the
thread-local `[req N]`, and `info` is the level a foreground `serve` prints by
default. Everything goes through the same logger, so the lines also land in the
ring served by `GET /v1/logs`.

Live `--backend metal` runs confirmed the format for both endpoints. A full
generation (256x256, debug area floor lowered) printed all six phases and all
six DiT steps under one `[req 1]`, 117284 ms total with dit 50673 ms; an
`images/edits` request with one reference printed the `edit request` arrival
line and the same phase sequence under `[req 2]`. The edits client hit the test
harness timeout mid-render while the server kept rendering and logging — the
stdout sequence stayed readable throughout. `make cwick` and `make test` pass.


## 82. A "completed" turn with no audio was an assistant-template bug

TeachMe showed a transcript and `response.done`, but never spoke. The chain
summary in stdout made the failure unambiguous:
`turn completed … decode 0 tokens … talker 0.000 (0.00 s audio, 0 chunks)`.
The WebSocket layer was fine: the transcript was streamed, the closing events
were sent, and the browser had no audio buffers to play.

The decoder was being handed
`<|im_end|>\n<|im_start|>assistant\n\n\n\n`. Those four newlines are inside the
assistant turn, so the model reads the reply as already begun and finished: the
first sampled token is end-of-turn, no sentence ever reaches the talker, and
the state machine honestly reports a completed response.

Isolating it took three A/B steps. A text-only turn with the same weights still
answered coherently, so the decoder and kernels were not at fault. Feeding a
fixed WAV through the offline listener reproduced zero decoded tokens and put
EOS at the top of the distribution (20.30, ahead of the best content token at
19.58); replacing the audio vectors with the learned placeholder embeddings
did not help, which ruled out the audio frontend. Ending the prompt at the bare
assistant opener dropped EOS to 4.67 and produced a normal answer. The same
tail is used in `fwd_user_block()` and `turn_positions()`, so both were changed
together — the latter must stay in step or the KV budget drifts from what is
actually forwarded.

After the fix, a live `--backend metal` WS run returned 53 `response.audio.delta`
events (10.44 s, PCM16/24 kHz) for 40 reply tokens. The reply WAV is not silent:
RMS 0.087, peak 0.84. A two-turn run produced 21 deltas per turn and confirmed
the warm turn prefill drops from 8.16 s to 2.83 s. `make test` passes. The full
breakdown and both failure modes are written up in `docs/REALTIME.md` and its
Chinese counterpart.


## 83. Turn detection needed hysteresis and a noise floor, not a lower threshold

The realtime input VAD was a single RMS threshold: above it, speech starts
immediately; below it, a fixed quiet timer ends the turn. Two field failures
follow from that shape. A 40 ms click is indistinguishable from a word and was
committed as a full turn, which could cancel a reply in progress. And a quiet
syllable between two loud ones — common in connected speech — lets the quiet
timer reach its limit and split one utterance into two, so the model answers
half a sentence.

A study of a full-duplex voice-agent stack pointed at two small mechanisms
rather than a neural detector: a release level below the onset level, and a
minimum active duration before a burst counts as speech. Both were adopted in
`src/rt_vad.c`. Onset and release are 0.15 apart on the client's 0..1 threshold
scale (`-40` vs `-44.5 dBFS` at 0.5); the band between them is a hold region
that does not advance the end timer. A burst must stay above onset for 100 ms
before it is reported, and a rejected burst is dropped from the pre-commit
buffer so repeated clicks cannot grow it without bound.

The decision was extracted into a pure module before changing it, so the same
code can be driven twice on one signal. `tools/rtvad.c` (`make rtvad`) shows
the click going from 1 start/1 end to 1 drop, the mid-word dip from 2 turns to
1, and a real 400 ms pause still splitting in both modes. The original rule is
kept behind `CWICK_RT_VAD_LEGACY=1`; five assertions in `make test` pin the new
behaviour. A live `--backend metal` manual-commit turn still produced 51 audio
deltas / 10.08 s of non-silent speech (RMS 0.095, peak 0.82).

`tools/rtws --vad` replays a finished recording plus trailing silence. A clip
with pauses longer than `silence_duration_ms` is split several times and the
rest is interpreted as barge-in; the same three commits and cancelled response
appear with the legacy flag, so this is a harness property, not a regression.
Use `--burst` when measuring a pre-recorded clip.

## 84. The Qwen3.5 checkpoint answers to its release name, not its type tag

The hybrid checkpoint's `config.json` carries two internal names: an outer
`model_type=qwen3_5` and an inner `text_config.model_type=qwen3_5_text`. Both
are kernel selectors — the loader unwraps `text_config` first, and `engine.c` /
`server.c` branch on the `qwen3_5` prefix — so neither may be rewritten. But
`cwick_model_served_alias()` derives the advertised API name from the directory
basename, and the Qwen3.5 directory was the only release checkpoint missing
from that table. It therefore fell back to the inner type tag: clients saw
`qwen3_5_text` in `GET /v1/models`, in the `model` field they had to send, in
`/v1/metrics` (`mi.model`) and in the startup log, while `start.sh` already
accepted the real name as a launch argument. That mismatch — launch by release
name, answer by type tag — was the bug.

Fix: one entry in the table mapping the release directory
`DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic` to that same name as its served
alias. `served_name()` (catalog), the metrics publisher (root-derived alias)
and `validate_body()` (through `cur->alias`) now all report and accept the real
name; `qwen3_5_text` and `qwen3_5` stay accepted as `model_id` / catalog id, so
nothing that learned the old name breaks. The alias differs from those ids, so
`cwick_models_list_add()` keeps it rather than blanking it as "no better name".
Two assertions in `make test` pin both the directory lookup and the catalog
entry; the suite is green at 749.

## 85. A downloaded checkpoint is not the same as a loadable one

The model-library tab called a completely downloaded checkpoint "Incomplete"
(`hub.partial`) when its directory had no `config.json` and was not an image
pipeline. `checkpoint_ready()` used those two markers as its only definition of
"downloaded", so a payload that ships plain `.pt` / `.ckpt` weights — the SVC
pipeline under `models/GiantAILab/YingMusic-SVC`, 2.03 GB in four files — was
reported as unfinished even though it had zero `.incomplete` files. The
download rule in `hub.c` is that `<target>.incomplete` is the *only* marker of
an unfinished transfer: a file is renamed into place only once its size
matches the listing. Every regular file in that directory had therefore
finished downloading; only the "is this a text model" heuristic was wrong.

Fix: `checkpoint_ready()` now also accepts a directory containing a recognised
weight payload (`.safetensors`, `.bin`, `.pt`, `.ckpt`, `.pth`, `.gguf`,
`.onnx`, `.npz`) when no `.incomplete` file is present. The reason text is no
longer "config.json is not present yet" but "weights-only checkpoint (no
config.json); not a cwick-loadable architecture". "Downloaded" and "loadable"
stay separate facts: the card renders "Complete" plus the architecture-mismatch
badge, and still offers no switch button (the checkpoint has no cwick kernel) —
the same treatment `Ace-Step1.5` already received via its `config.json`.

Measured on the live server: `GET /v1/hub/models` went from
`complete=false, supported=false` to `complete=true, supported=false` for that
one entry; the other seven models were byte-identical. `make test` stays green
at 749.

## 86. A hand-downloaded pipeline is invisible to the library scan

`models/MiniMaxAI/MiniMax-H3/` holds a complete 498 GB diffusers pipeline
(485 files) that the model-library tab never listed. Its files are intact and
there is not a single `.incomplete`, so this was not a download-state problem
like §85 — the directory never entered the inventory at all.

The scan gate `looks_like_checkpoint()` accepted only three markers:
`config.json`, the strict H3 *image* manifest (`_class_name` equal to
`MiniMaxH3ImagePipeline` or `partition:"image"`), and the `.cwick-repo` marker
that cwick's own downloader writes. A checkpoint downloaded by hand has no
such marker, and this one's manifest says `MiniMaxH3ModularPipeline`, so both
the owner directory and its child were skipped.

Fix: `looks_like_checkpoint()` now also accepts any top-level
`model_index.json`, independent of its class and of how the files arrived.
The strict image manifest is still used for the `image` kind and for
`supported`, so the modular pipeline is listed as complete but not loadable
(`complete=true, supported=false`) rather than offered as a switchable image
model. A half-downloaded pipeline has the index but no weight payload, so it
stays visible as incomplete.

Verified offline (no server started): a small probe calling
`cwick_hub_models("./models")` reports nine entries with MiniMax-H3 present at
498 GB / 485 files; `make test` is green at 755 (six new assertions).

## 87. The H5 studio closes the loop from image to video

App 3 (`apps/imagegen`) is a two-step H5 studio. Step one already called the
two image endpoints; step two now takes the PNG from step one (or an uploaded
image) plus a video prompt and calls the async `/v1/videos` resource, polling
`GET /v1/videos/{id}` for the real `progress` field and streaming
`GET /v1/videos/{id}/content` for download. Progress is server-driven, not a
timer: the bar tracks the job's phase weights (text 0-5%, pack 5-10%, dit
10-80%, video VAE 80-92%, audio VAE 92-97%, mux 97-100%).

The sampler is text conditioned — the visual VAE has only a decoder, so there
is no encoder to turn the reference image into an initial latent. Rather than
pretend to do image-to-video, the app submits the image as an optional `image`
field (data URL or bare base64) and the server validates it (PNG/JPEG magic),
keeps the bytes with the job and reports `reference_image` with
`conditioning:"text_only"` in the detail view. The source image is shrunk to a
1024px JPEG in the browser first: a full 1536x2048 PNG data URL can exceed the
server's 8 MiB request-body ceiling, and a reference does not need full
resolution.

The server recycles a video slot by freeing the previous `prompt`, `mp4` and
`ref_image` before `memset`, which also fixes a pre-existing mp4 leak on job
reuse. The `image` failure paths free the decoded bytes. `make test` stays
green; embed rebuild reports the updated 22 blobs.

## 88. First-frame-to-video needs the encoder we said did not exist

Section 87 claimed the visual VAE had only a decoder. That was wrong. The
shipped `FL2VA/video_vae/source/model.safetensors` contains all 116 `encoder.*`
tensors, plus `quant_conv` and `post_quant_conv`: a causal 3D CNN with ch=128,
`ch_mult=[1,2,2,4,4,8]`, two residual blocks per level and spatial strides
`[2,2,2,2,1,1]`. The missing piece was the implementation, not the weights.

`h3v_vae_encode()` in `h3v_vae.c` now turns one RGB24 frame into the 24-channel
latent mean. The key reduction is temporal: with a single frame, causal padding
puts `KT-1` zeros in front and only the last temporal kernel slice contributes,
so every 3D convolution becomes a 2D convolution with kernel `w[:,:,KT-1,:,:]`.
Convolutions run as im2col + sgemm; spatial pad is reflect for residual/conv
blocks and a lower-only zero pad for strided downsamplers. GroupNorm(32, eps
1e-6) is per-frame, followed by SiLU. `norm_out -> conv_out(48) -> quant_conv`
yields the moments; the first 24 channels are the latent (eps=0 for a
conditioning frame).

Verification is pure C and does not trust the old numpy replay (whose conv
helper padded the wrong axis and read only the zero frame, leaving just the
bias). A hand-computed interior pixel of `conv_in` matched the C kernel to
1.3e-7 once the bias was read from the right shard offset. End to end, encoding
a structured 64x64 pattern and decoding the latent straight back through the
already-verified ViT decoder gave an R-channel correlation of 0.9186 — an
encoder that produced garbage could not reconstruct its input.

Conditioning uses the existing video-token channel: no new DiT input exists.
The reference is resized to the render canvas, encoded, and written into
latent step 0; a packed copy of those rows is restored after every Euler
update so the frame stays clean while the DiT keeps attending to it. The cache
shortcut is disabled when a reference is present. The HTTP detail now reports
`reference_image.conditioning` as `first_frame` when encoding succeeded and
`text_only` only on decode failure. Both `h3vprobe` and `h3vencprobe` are
plain C tools in `tools/` with Makefile targets.

## 89. The mp4 muxer wrote no moov atom, and esds wrapped a descriptor in a descriptor

The first end-to-end i2v clip could not be opened at all: ffprobe reported
`moov atom not found` and `wrong sample count`. A pure-C box walker on the
output showed why. `h3v_mp4_encode()` built `moov` as a buffer of *children* and
then appended `moov.p` with `mb_put()` — there was never a box header for the
atom itself, so `mvhd` and both `trak` boxes sat loose at top level. Separately
`stsc` declared one chunk containing all samples while `stco` had one offset
per sample (102 entries); ISO 14496-12 requires those counts to agree.

Four fixes in `h3v_mp4.m`: emit `mb_box(&out, "moov", &moov)`, make stsc
`samples_per_chunk = 1`, express `tkhd` duration in the movie timescale (it was
using the media timescale, 24x too long), and correct two descriptor length
fields in `esds`.

Audio still failed after the box layout was fixed: `Audio object type 0`,
`22050 Hz, 0 channels`. The AudioToolbox magic cookie is a complete ES
descriptor with 7-bit expanded sizes, not the bare AudioSpecificConfig that
belongs in `DecSpecificInfo`; we had nested the former inside the latter.
`find_asc()` now walks ES -> DecoderConfig -> DecSpecificInfo and extracts the
two-byte config (`12 90` for 32 kHz stereo AAC-LC). ffprobe then reports
`AAC (LC), 32000 Hz, stereo` and both streams decode to exactly the expected
sample counts (AAC output is padded to its 1024-sample packet boundary).

Regression is cheap now: `make h3vmuxprobe` feeds synthetic pixels and PCM
straight to the muxer, and `test_h3v_mp4` in the suite checks box coverage,
stsc/stco agreement and the bare AAC-LC config in memory. Neither needs a
checkpoint or a four-minute DiT pass.

## 90. Video timings were only in the job detail, not in /v1/metrics

A client polling `GET /v1/videos/{id}` saw a full `cwick` timing object after
completion, but a Prometheus scrape saw no video metrics at all — the same
blind spot that had made the 10%-stuck job look healthy. `metrics.{c,h}` now
publishes `cwick:video_jobs_total`, `_succeeded_total`, `_failed_total`,
`_first_frame_jobs_total`, cumulative `cwick:video_stage_seconds_total{stage=...}`
(text, pack, encode, dit, vae, audio_vae, mux, load, total), live
`cwick:video_dit_steps_done` / `_total`, `cwick:video_last_total_seconds`,
frame/byte counters and `cwick:video_tokens_total{kind=...}`. The DiT gauges
are set from the existing progress callback, so a scrape during a long sample
distinguishes progress from a hang. `h3v_timing` gained `encode_ms` for the
reference-image VAE pass; the job detail and the counters are measured by the
same monotonic clock.

## 91. First-frame conditioning verified on a playable clip

Section 88's encoder numbers were tensor-level. The full i2v path was then run
with the fixed muxer (§89) at 256x256 / 1 step and produced a 571,855-byte mp4
that ffprobe reads as H.264 High 24 fps plus AAC-LC 32 kHz stereo, duration
4.32 s. Decoding frame 0 and frame 40 and comparing luma against the 64x64
reference upscaled in pure C gives correlations **0.567** and **−0.036**. The
reference is therefore genuinely present in the first decoded frame and absent
later, which is the behaviour the latent-step-0 injection is supposed to
produce. Full stage timings are in PERF.md.

## 92. Video DiT and VAE now have a Metal path; output stays byte-close to CPU

Sections 89-91 fixed correctness and observability but left the video DiT and
VAE on CPU fp32 SGEMM. `--backend metal` initialised a device that no video
kernel ever used. This adds `h3v_gpu.{h,c,m}`: a BF16 tensor-core GEMM for the
50 DiT blocks, an F32 tensor-core GEMM for the VAE's F32 checkpoints, and a
tiled flash-attention kernel instantiated at 128 channels (DiT) and 64
channels (VAE decoder). All share the same no-copy adoption used by the image
transformer, so a 62 GiB shard is wrapped once rather than once per projection.

Two kernel details mattered. Metal forbids threadgroup arrays in a non-kernel
function, so the kernels declare the staging memory and pass it to the shared
core. And the attention rescale maps a simdgroup's 8xHD output over 32 lanes as
four blocks of HD/4: hard-coding the 128-wide layout made the 64-wide instance
write garbage. The h3vgpuprobe tool caught both before any weights were loaded.

The policy is per call: unsupported shapes, no device, or `CWICK_H3V_GPU=0`
fall back to the existing CPU code, so the GPU can never block a clip.
Measured on one machine, 256x256 / 1 step: total 269.4 s -> 168.2 s, DiT step
149.5 s -> 51.6 s, video VAE 105.8 s -> 50.5 s. Decoded frames from the two
paths correlate 0.9989 (frame 0) and 0.9944 (frame 40); first-frame conditioning
still gives 0.567 against the reference at frame 0 and -0.037 at frame 40.
Full table in PERF.md.

## 93. Image endpoints open the model on demand and report real progress

Application 3 (the two-step image + video wizard) had two usability gaps left
after the video work. First, the image endpoints still required the operator to
pre-switch to `MiniMax-H3-Image`: a cold POST answered 503 and the H5 studio
showed a blocking "switch in settings first" confirm. The video endpoint had
already solved this by loading its pipeline on first use, through the same
one-model slot and swap gate the explicit switch uses. `srv_image_route` now
does the same: if no image pipeline is resident it calls `srv_switch_image`,
which closes the gate, releases the text/realtime weights, and maps the image
checkpoint. A concurrent switch answers 503 instead of double-loading. The H5
confirm was removed and the model pill now reads "load on first use".

Second, the image progress panel was a fake timer (`6 + secs/30`) that filled
up regardless of what the renderer was doing. The renderer already publishes a
live snapshot (phases 0 idle, 1 tokenize, 2 text_encode, 3 pack, 4 dit with a
step/steps counter, 5 vae, 6 png, 7 done); it just was not exposed. A new
`GET /v1/images/progress` answers before admission with `phase`, `phase_name`,
`step`, `steps`, `idle` and `age_ms`, so a poll never queues behind the ~40-minute
job it is watching. The H5 app polls it once a second, weights DiT at 60% of
the bar, and still fills the exact per-stage timings from the response's
`cwick` object at the end. A failed or stale poll only leaves the last snapshot
on screen; it cannot disturb the render.

End-to-end on this machine with the rebuilt binary: a cold
`POST /v1/images/generations` released the text model and loaded the image
pipeline on demand, then returned a clean 400 for an empty prompt (no render
started); `GET /v1/images/progress` reported `idle:true` at rest. `make test` passes and
the served `app.js` is the rebuilt 33420-byte asset with three references to
the real endpoint.

## 94. A "black" video was white noise behind a valid mp4, and the dashboard called it success

The log for `video_1795193877_1` and `/v1/metrics` agreed on the mechanics: the job
finished, muxed 2,570,415 bytes, reported `video_jobs_succeeded_total 1` and
`video_first_frame_jobs_total 1`. The total was 20,752.6 s, of which the DiT alone
took 19,948.8 s across 30 steps (~665 s/step at 736x1280). Nothing in that data
described the pixels. Pulling `GET /v1/videos/{id}/content` and reading frames back
showed why it looked black: frame 0 held the recognizable conditioning image under
a heavy tile grid, but frames 20 and 100 were high-frequency static (mean luma
70-73, smoothness ratio ~0.53 versus ~0.75 for frame 0). Adjacent-frame
correlation fell from +0.97 across the held frames to +0.77-0.80 deep in the clip
— the generated latent was still noise, and the VAE faithfully decoded noise.

The sampler's sign, timestep embedding and sigma-to-timestep mapping were checked
against the image line, which produces good output with the same flow-matching
family; they match. The concrete defect found was the audio shift: the checkpoint
declares `sigma_shift_scales` video=12.0, audio=3.0 in `model_index.json`, but the
sampler applied 12.0 to both streams, making the audio Euler step 4x too large.
That does not explain the video frames, so it was fixed without claiming it did.

Two things were added because a 5.7-hour generation cannot be the feedback loop:

1. A CPU-only denoise signal, computed from the latent the pipeline already
   holds. `spatial_corr` is the mean lag-1 horizontal autocorrelation of the
   first generated latent step (the conditioning step is excluded); natural
   decoded latents sit near 0.8, white noise near 0. `snr_db` is the generated
   step's std over the clean step's std. Both land in `h3v_timing`, in the job
   detail's `cwick` object, and `suspect_noise` adds a
   `cwick:video_suspect_noise_jobs_total` counter so a noise clip is no longer
   indistinguishable from a good one in a dashboard. The floor is provisional
   (0.35, `CWICK_H3V_CORR_FLOOR` overrides it) until a known-good clip is
   measured. `tools/h3vprobe` prints the same three numbers for direct A/B.
2. Config-backed shifts: `h3v_model_open` reads `_minimax_h3.sigma_shift_scales`
   (defaults 12.0/3.0), video and audio each step on their own sigma grid while
   the single shared time embedder is unchanged, and `CWICK_H3V_VIDEO_SHIFT` /
   `CWICK_H3V_AUDIO_SHIFT` permit A/B. `CWICK_H3V_REF_NOISE_AUG=<level>` also
   exposes the image line's noise-augmented first-frame conditioning as an opt-in
   lever; the default stays clean so existing output does not move until it is
   measured.

`make test` passes, including a new assertion for the suspect-noise counter, and
`h3vprobe` builds. The open question is now measurable in one short probe instead
of one long render: which DiT convention leaves the generated frames at noise
level. No claim is made that the black screen is fully fixed by the shift change.

## 95. Calibrating the noise floor and A/B-ing the DiT timestep convention

The provisional 0.35 correlation floor in §94 needed real numbers. Two pure-C
knobs were added to `h3vprobe` (no model reads, no Python):
`CWICK_H3V_CORR_TRACE` prints the generated step's lag-1 latent correlation after
every Euler update, and `CWICK_H3V_DIAG_MIX=<m>` / `CWICK_H3V_DIAG_TS=<t>` start
from a known `m*noise + (1-m)*clean` mix and force one conditioning timestep, so
the velocity's direction can be read in one step instead of one 30-step render.
`h3v_timing.clean_corr` (the same correlation on the clean conditioning step)
and `H3VPROBE_METAL=1` (force Metal to match `--backend metal`) were added to the
probe as well.

Calibration at 256x256, 2 steps: the VAE-encoded clean reference step measured
0.769-0.814, pure/near noise measured -0.145 to 0.068. The 0.35 floor separates
them cleanly and stays the default (`CWICK_H3V_CORR_FLOOR` still overrides it).

A one-step probe at sigma=0.9 appeared to show the checkpoint wants `t=sigma`:
`ts=0.1` (the code's `1-sigma`) drove corr 0.068 -> 0.025 while `ts=0.9` drove it
0.068 -> 0.290. That was a false lead — the forced `dt=0.9` is a giant step and
the latent std exploded from 0.947 to 2.042, past the clean anchor, so it measured
an out-of-distribution overshoot rather than a convention. Two matched full 8-step
runs settled it: `t=1-sigma` (the original) ended at corr 0.360 (`ok`), `t=sigma`
at 0.288 (`suspect`). The change was reverted; the default stays `t=1-sigma`,
matching the image line. `CWICK_H3V_TS=sigma` keeps the alternative for a test at
>=4 MP, where the checkpoint is actually in distribution — at 256x256 it returns
textured noise by design, so a short run cannot judge denoising quality at all.

What is now established: the noise floor is calibrated; the audio shift was a
real bug (checkpoint declares 3.0, code applied 12.0) and is fixed via config;
and a noise clip can no longer masquerade as `succeeded`. The 30-step black
screen's root cause is still open. The remaining suspects are the shift=12 grid
back-loading nearly all denoising into the last step, and the reference-level
difference (the image line conditions reference rows at their own 0.999 timestep
with a `0.999*clean + 0.001*noise` latent; the video line uses a pure-clean latent
and one shared timestep). Both need a >=4 MP A/B to confirm; no fix is claimed.

## 96. The denoise signal was reading the wrong tensor plane; layout fixed and re-measured

The noise-floor calibration and the timestep A/B in §95 were built on a broken
reader. `vlat` is `[C][T][H][W]` — `h3v_pack3d()` indexes it as
`((c*lt + t)*lh + y)*lw + x`, and the VAE decoder gathers a cell's 24 channels
with the same stride. But `h3v_lat_step_stats()` computed its plane as
`t*VD_LATENT*lh*lw` (a `[T][C][H][W]` assumption), so for t=1 it measured a mix
of channel 0's late time steps and channel 1's early ones, not the first
generated frame. The reference injection had the mirror bug: it `memcpy`'d the
encoder's contiguous `[C][H][W]` output into the front of `vlat`, which in
`[C][T][H][W]` scatters the 24 channels down channel 0's time axis instead of
placing them all at t=0. Frame 0 still looked partly right because the packed
reference rows were restored after every Euler step, but the DiT's first step
conditioned on a corrupted t=0 latent.

Both are fixed in `src/h3v.c`. The reference frame is assembled in encoder
layout, scattered channel-by-channel to `vlat` t=0, and packed from that same
buffer; the stats function takes `lt` and strides `(c*lt + t)*lh*lw`. A weight-
free `h3v_layout_selftest()` in `h3v_internal.h` checks that a smooth plane at
t=0 reads high, a t=1 plane reads low, and pack/unpack round-trips exactly;
`make test` asserts it.

Re-measured at 256x256, 4 s, seed 42, Metal, 8 steps (all four runs matched):

| config | final spatial_corr | clean_corr | snr_db | suspect |
| --- | --- | --- | --- | --- |
| default (t=1-sigma, shift 12, clean ref) | 0.247 | 0.848 | +0.63 | yes |
| `REF_NOISE_AUG=0.999` | 0.247 | 0.848 | +0.63 | yes |
| `VIDEO_SHIFT=3` | 0.145 | 0.848 | +0.91 | yes |
| both | 0.145 | 0.848 | +0.91 | yes |

The corrected trace starts at corr -0.003 (true white noise) and rises
monotonically to 0.247; the old trace's 0.404 start was a cross-channel read.
`clean_corr=0.848` is the real smooth reference. The 0.35 floor still separates
clean from noise and is unchanged. Reference noise augmentation is a no-op at
0.999 — that is ~0.1% noise over a clean latent, and the image line's actual
mechanism is a separate 0.999 timestep, which this checkpoint cannot express:
its `adaln_proj.linear.weight` is `[96768, 2688]` = 18 rows = 6 vectors x 3
modalities, with no timestep axis. Lowering the video shift to 3 spreads the
denoising across steps (step 3 corr 0.072 vs 0.007 at shift 12) but ends lower
at 8 steps (0.145 vs 0.247). All four remain under the floor at 256x256, which
the checkpoint treats as out of distribution (>=4 MP is the supported size), so
no conclusion about 30-step quality is drawn here. The 30-step black screen's
root cause is still open; the layout fix removes a real defect but does not by
itself prove the fault is gone.

## 97. Both FL2VA and Ref2VA are selectable per request

The shipped MiniMax-H3 root contains two complete partitions, `FL2VA/` and
`Ref2VA/`, each with a 62 GiB `transformer/`. Their DiT configs and all 538
weight keys are identical, and the visual VAE, audio VAE and text encoder are
byte-identical (`FL2VA/video_vae/source/model.safetensors` and the Ref2VA copy
hash the same, as do the text-encoder indexes). Only the trained transformer
weights differ: FL2VA is first/last-frame-to-video+audio, Ref2VA is omni-
reference-to-video+audio. The old `h3v_model_open()` picked the first directory
that existed — FL2VA on a full download — and offered no way to reach Ref2VA,
so half the checkpoint was unusable through `/v1/videos`.

`h3v_model_open()` now enumerates every usable variant (a variant counts only
if its `transformer/` actually holds shards), records each canonical name with
its DiT directory, and picks a default from `CWICK_H3V_VARIANT` or the first
found. `h3v_generate_variant()` pins one partition for a job; the shared VAE
and text encoder stay mapped. Because all generation already runs under the
model mutex and each job opens and closes its own DiT shards, only one 62 GiB
transformer is resident at a time and the two variants never contend.

`POST /v1/videos` accepts `variant` (alias `model_variant`), validates it
before queuing (HTTP 400 with the available names on a typo), and reports the
canonical value in the job detail next to `reference_image`. stdout logs the
chosen variant and the exact shard path at DiT open. `tools/h3vprobe.c` takes
`H3VPROBE_VARIANT` for the same path. A pure-C `h3v_variant_selftest()` covers
case-insensitive name matching and is asserted by `make test`.

End-to-end, `--backend metal`, 256x256, 4 s, 2 steps, seed 42, same
`input/sample.png` (1152x1536) and `input/sample_video.txt` prompt for both:

| variant | status | DiT shards opened | conditioning | total_ms | dit_ms | spatial_corr | suspect |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FL2VA | completed | `FL2VA/transformer` | first_frame | 285894 | 165533 | 0.376 | 0 |
| Ref2VA | completed | `Ref2VA/transformer` | first_frame | 256562 | 127090 | 0.360 | 0 |

Both downloads produced a valid H.264 + AAC mp4 (559228 vs 568343 bytes,
different hashes), first-frame conditioning succeeded, and `suspect_noise=0`
for both. `/v1/metrics` read `video_jobs_total 2`, `succeeded_total 2`,
`failed_total 0`, `first_frame_jobs_total 2`, `suspect_noise_jobs_total 0`.
The two step smoke test proves both partitions load and run on this machine;
it does not make a 30-step quality claim (256x256 is below the checkpoint's
supported size).

## 98. The video VAE was decoding an un-normalised latent

`vae/config.json` ships 24 per-channel `latents_mean` / `latents_std` values.
The transformer samples in that normalised space, but the video path ignored
both tables: `h3v_vae_encode()` returned the `quant_conv` output raw, and
`h3v_vae_decode()` fed the sampled latent straight into the folded
`post_quant_conv` + `x_embedder`. `h3v_internal.h` even documented the decoder
input as "already de-normalised by the caller", but no caller did it. The
per-channel error is large — std spans 0.45-3.28 and mean spans -1.37-1.07 —
so the 36-block ViT decoded a picture unrelated to the sampled clip, and the
conditioning frame written into latent step 0 was equally wrong for the DiT.

Fix, all in `src/h3v_vae.c`: the two tables, `vl_lat_norm()` on the encoder
output (raw -> normalised, so a reference frame lands in the same space the
noise lives in) and `vl_lat_denorm()` on the decoder input (normalised -> raw)
before the tile loop. `h3v_latnorm_selftest()` checks the round-trip and that a
zero latent maps exactly onto the per-channel mean; `make test` asserts it.

Three measurements separate this from the older "black screen" theory:

* encode -> decode at 256x320, no DiT: R-channel correlation **0.8762** and the
  reconstruction is a recognisable portrait. Before the fix the same latent
  decoded into a 16-pixel colour lattice.
* a constant (zero) latent decodes to a smooth field, so the decoder and its
  tiling are healthy; the lattice came from the values, not the tiling.
* the end-to-end frame 0 (the held reference step) went from a flat grid to a
  clear person silhouette.

End-to-end on this machine, `--backend metal`, `input/sample.png` (1152x1536)
plus the `input/sample_video.txt` prompt, 768x1024, **4 steps**, seed 42,
FL2VA: completed in 2,953,739 ms (DiT 2,241,221 ms, video VAE 678,956 ms),
`spatial_corr 0.309`, `suspect_noise 1`, 2,923,218-byte mp4. Frame 0 carries the
silhouette, frames 40/100 decay to the lattice again. Four steps are not a
quality verdict: with `shift=12` the sigma grid is 1.0, 0.973, 0.923, 0.8, 0, so
the last Euler step is 0.8 wide and does almost all of the denoising in one
unconditional jump.

Two probe results bound the remaining work:

* timestep convention is correct. A known mix of 0.9*noise + 0.1*clean with one
  step at sigma=0.9 moved `spatial_corr` 0.008 -> 0.334 under `t=1-sigma` and
  0.008 -> -0.042 under `t=sigma`.
* the Euler direction is not settled. At a small step (dt=0.1 and 0.2) the
  *minus* direction moves toward the clean anchor (corr 0.216 -> 0.289, std
  0.699 -> 0.886 at dt=0.2) while plus degrades it (0.216 -> 0.146), but a
  same-size/same-step end-to-end A/B at 4 steps prefers plus (0.300 vs 0.277).
  At 4 steps the 0.8-wide final step dominates and overshoot masks direction, so
  the end-to-end test cannot decide. Default stays `plus` (measured winner);
  `CWICK_H3V_SIGN=minus` exposes the other direction for a 30-step A/B. A
  30-step 768p run is ~5 hours on this machine and is the next measurement, not
  a guess.

Probe-only knobs added for that A/B: `CWICK_H3V_DIAG_DT` (shrink the forced
step) and `CWICK_H3V_SIGN`; `tools/h3vencprobe` now takes `H3VPROBE_W/H` so the
encoder can be exercised at the render size instead of a fixed 64x64.

## 99. The DiT read release-layout tensors in converted-layout order

The transformer shards keep the release tensor names and layout. Two reorder
steps that the reference loader performs are not implicit in the bytes:

* a fused QKV weight row is `head0: q|k|v, head1: q|k|v, ...`. Splitting the
  output into three contiguous thirds means head *h*'s query was head 0's key
  and so on: every attention head mixed the wrong streams.
* `mlp.fc1` emits `[gate | value]`, while the SwiGLU read `[value | gate]`, so
  the activation was applied to the value and multiplied by the raw gate.

Fix in `src/h3v.c`: `h3v_split_qkv()` scatters the raw output on the CPU path,
and a new `gemm_bf16_qkv` Metal kernel permutes the weight rows while loading
the tile, so the GPU path writes q/k/v straight into three planes with no
extra pass. SwiGLU now takes the gate from the first half. Both the 50-block
stack and the two token-refiner blocks use the same projection layout.
`CWICK_H3V_LEGACY_LAYOUT=1` restores the old decoding for an A/B.

Same-input A/B (256x256, one step, DIAG_MIX 0.6, dt 0.1, seed 42, Metal):

| signal | old layout | fixed |
| --- | --- | --- |
| cos(velocity, data-ward reference) | -0.142 | **+0.954** |
| best scale for the reference field | -0.042 | 0.919 |
| \|velocity\| | 206.73 | 679.06 |
| MLP block 0 output rms | 86.32 | 13.98 |
| residual rms, block 49 | 22497 | 7570 |
| generated-step `spatial_corr` | 0.338 SUSPECT NOISE | **0.501 ok** |

The fused kernel is bit-identical to the host-side scatter: both report
`|v| 679.0618`, `cos 0.954`, `corr 0.501`. `h3vgpuprobe` covers the new
pipeline.

End-to-end on this machine, `--backend metal`, `input/sample.png` plus
`input/sample_video.txt`, 256x256, 4 steps, seed 42: the DiT ended at
`spatial_corr 0.927` (`suspect_noise 0`), the mux wrote a 609,140-byte mp4 in
384.1 s total (text 70.1 s, DiT 244.8 s, video VAE 63.2 s, audio VAE 4.3 s).
Decoded frames 20, 60 and 100 show the reference person instead of the
lattice/static that the same prompt produced before this fix. Four steps at
256x256 are only a correctness check — the 768p / 30-step quality run is a
separate measurement.

The same four-step input at 768x1024 also cleared the floor: `spatial_corr
0.963`, `clean_corr 0.929`, `snr_db 2.56`, `suspect_noise 0`, a 1,249,044-byte
mp4 in 3,008.3 s total (text 63.3 s, DiT 2,282.4 s, video VAE 650.4 s, audio
VAE 4.4 s, mux 0.4 s). Where the original clip was a green lattice in every
sampled frame, frames 60 and 100 are a clean, moving person; frame 20 still
shows a one-sided lattice seam and frame 0 a horizontal band, so the temporal
VAE boundary is a separate remaining artefact. The 30-step quality run at this
size is still outstanding.

## §100 Quick-verify preset in the image + video app

The H5 studio (app 3) already exposed the two-step image -> video flow and the
server accepts a 256x256 canvas and a 4-second clip (the engine rounds both to
its 32-pixel canvas multiple and to whole 17-frame clips). What it could not do
was shorten the sampling: the DiT budget was fixed at 30 steps, reachable only
through a process-wide environment variable, so even the smallest canvas took
tens of minutes — not a quick verification.

`POST /v1/videos` now accepts an optional `steps` integer. `h3v_generate_variant_steps()`
carries it per job and clamps it to [1, `H3V_SAMPLE_STEPS`=30]; the old
`h3v_generate_variant()` keeps its signature and passes 0 (engine default).
This also closes a latent bug: the environment variable used to accept 1..256
while the sigma/timestep scratch arrays are sized for exactly 30, so a value
above 30 wrote past them. Both paths are now bounded by the same ceiling. The
job object reports the requested `steps`, and the completed `cwick` object still
reports the actual steps used. Default behaviour (no field, or 0) is unchanged.

The app's step-2 panel gains a 256x256 / 320x320 canvas option, a step counter
and a one-click "quick verify" preset (256x256, 4 s, 4 steps). On this machine
that 256^2 / 4-step path produced a clean, moving person in 384.1 s (§99),
which is the measurement the preset is built around. The full 30-step schedule
remains the default for quality renders.

## §101 Memory-aware residency for the video pipeline

The text/model lines already route their weight stages through the shared
memory budget (`--mem-fraction`, default 80% of host RAM; `--max-memory`;
no-swap wiring). The video line did not. Its DiT (61.73 GiB) was demand-mapped
on every request and unmapped on exit — never wired — so the VM was free to
compress or page the very pages the 30-step sampling loop re-reads on every
step. Its two VAEs were mapped for the process but also left unwired.

`cwick_mem_fit_model()` is now the shared decision for a single weight stage:
given the stage's mapped bytes, its runtime scratch and a reserve, it compares
the whole total against the ceiling and, in no-swap mode, elects to fault in
and wire only when the stage fits. An over-budget stage is deliberately left
as a lazy, SSD-backed mapping — wiring part of an over-budget stage would only
push something else into the compressor. The video line calls it right after
each safetensors open: the two VAEs at model load (resident for the process)
and the DiT at generation time (resident for one sampling pass; `close()`
unwires automatically). `CWICK_H3V_WIRE=0/1` overrides the decision for an A/B.

The two 62 GiB stages are kept staggered on purpose. The text encoder is mapped
only during phase 1 and released before the DiT is mapped in phase 3; wiring
both at once would peak at 134 GiB, above the 102.4 GiB ceiling, and force
exactly the swap this avoids. The text encoder is a read-once streaming access,
so leaving it lazy is both correct and cheaper than wiring it.

Four gauges expose the decision: `cwick:model_weight_bytes`,
`cwick:model_resident_bytes`, `cwick:model_target_bytes`, `cwick:model_wired`.

Measurement (same machine, Metal, 256x256 / 1 step, resident path):
VAE 9.70 GiB + audio VAE 0.56 GiB + DiT 61.73 GiB all logged `wire resident`
under a 102.40 GiB ceiling (80% of 128 GiB host), ~72 GiB wired in total. The
kernel never reported a wiring failure. The decisive no-swap evidence is the
page counter, not the `swapused` level: `Swapouts` went 137479946 -> 137479946
(delta **0**) across the whole run, and `swapused` actually fell from 6052 to
5916 MiB. `Compressions` rose by 0.009% of its since-boot value. A 2.1 GiB
`swapused` rise seen in an earlier run was global noise from other processes,
not the model's wired pages: with `mlock` succeeding those pages cannot be
paged. `denoise check` reported spatial corr 0.955 (floor 0.35). `make test`
and `make h3vprobe` pass.

## §102 The video endpoint uses Ref2VA only

Section 97 made both checkpoint partitions reachable, which was the right way
to find out what each one does. `POST /v1/videos` takes a prompt plus an
optional image and asks for a generated clip; that is the omni-reference
workflow, so Ref2VA is the transformer that matches the endpoint. The
first/last-frame partition belongs to a different request shape and was only
ever the default because it sorts first in the download. In this build
`h3v_model_open()` discovers only `Ref2VA/` (or a flat copy of it), the default
is that single variant, and a `variant` naming the first/last-frame partition
is rejected with HTTP 400 before a job is queued. Its shards are never probed
or mapped. This removes the accidental quality A/B from the endpoint without
changing the shared VAE, text encoder, job model or progress API.

## §103 Ref2VA reference presentation and reference geometry

Selecting Ref2VA exposed two places where the pipeline still used the
first/last-frame request shape:

1. The image was encoded at the generated canvas and entered the sequence only
   as latent rows. In the omni-reference workflow a picture is also part of
   the text presentation (`Picture 1: <vision block>` before the prompt), and
   the vision-token rows carry the video modality tag for AdaLN. The new
   `h3_text_cond_ref_dir()` in the text-encoder helper builds that
   presentation with the same vision tower the image path uses and returns
   the per-row tags. `CWICK_H3V_REF_PRESENT=0` restores the bare prompt.
2. An image reference is encoded at its own resolution; it does not inherit
   the generated canvas. `CWICK_H3V_REF_EDGE` (32..6272, unset = canvas) scales
   the reference's short edge before encoding, derives its own latent grid and
   packed-row count, and gives the condition rows their own spatial rotary
   grid. An image reference also advances the shared audio/video rotary clock
   (32..6272 is set by `H3V_REF_EDGE_MAX`, deliberately separate from the
   generated-canvas cap `H3V_MAX_SIZE`=2560 (2K): an earlier build clamped
   REF_EDGE to the canvas cap and silently fell back for an explicit 2048;
   the two are now split, so 2048 encodes a 1152x1536 reference as 2048x2720).
   The canvas cap itself was raised from 1280 to 2560 (under 4K): app 3's
   `vsize` gains 1600x900 / 1920x1080 / 2560x1440 and `sizeHint` follows; the
   image path's `H3_MAX_SIZE` is already 6272 (past 4K) and needed no change.
   The size arithmetic lives in `h3v_ref_encode_size()` shared by production
   and `make test`, which covers 2048, the unset fallback, out-of-range, plus
   2560 admitted / 3840 refused. DiT rows grow roughly quadratically with the
   canvas, so a 2K full 30-step job takes hours; the UI steers users to the
   4-step verification preset first.
   by one slot, which was previously missing. `CWICK_H3V_CONDROWS=0` and an
   explicit edge are not combined: the legacy target-t0 layout needs the
   reference grid to equal the target grid.

Neither knob changes checkpoint precision, the sampler, or the job/HTTP model.

Endpoint measurement before the new presentation was exercised (`--backend
metal`, 256x256, 4 s, 30 steps, seed 42, `input/sample.png` plus
`input/sample_video.txt`, POST `/v1/videos` with `variant:Ref2VA`): completed
in 1,342,128 ms total (text 58,397 ms, DiT 1,232,712 ms, video VAE 45,123 ms,
audio VAE 3,906 ms), 102 frames, 612 KiB mp4, `spatial_corr 0.899`,
`snr_db 0.17`, `suspect_noise 0`. `cwick:host_swapouts_total` stayed at
137485839 across the run while 66.28 GiB of weights were wired under the
102.40 GiB ceiling. Sampled frames 0/20/60/100 show the seated subject in the
cream fuzzy coat and beret, frame 100 turns as the prompt asks, and the old
horizontal band on frame 0 is gone. The remaining defect is texture: heavy
grain and intermittent green speckles on the coat, with soft focus throughout.
That is the target of the four-step A/B below.

Matched four-step A/B (`--backend metal`, 256x256, 4 s, seed 42, same prompt,
same RGB reference, 4 sampling steps):

| run | text tokens | sequence rows | total_ms | dit_ms | spatial_corr | clean_corr | snr_db |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `REF_PRESENT=0`, canvas reference | 34 | 2358 | 321737 | 202266 | 0.891 | 0.885 | -0.60 |
| `REF_PRESENT=1`, canvas reference | 1417 | 3741 | 514776 | 342490 | 0.905 | 0.877 | -0.36 |
| `REF_PRESENT=1`, `REF_EDGE=512` (336 condition rows) | 1417 | 4013 | 522574 | 356200 | 0.921 | 0.905 | -0.17 |

All three produced 102-frame mp4s with `suspect_noise 0`. The vision
presentation costs ~60 s of extra text encoding and ~140 s of DiT over four
steps, but frames 20/60/100 are visibly smoother and the face, short hair,
beret and cream coat read closer to the reference; the bare-prompt run is a
larger, blurrier face with more colour speckle. Encoding the reference at a
512 short edge instead of the canvas raises the three correlation figures
again but is only a marginal visual change at 256x256, so it stays opt-in.
The presentation is therefore the new default.

The same 30-step endpoint run was repeated with the presentation enabled:
2,145,126 ms total (text 111,825 ms, DiT 1,981,593 ms, video VAE 45,705 ms,
audio VAE 3,849 ms), 102 frames, 596 KiB mp4, `spatial_corr 0.814`,
`snr_db -0.52`, `suspect_noise 0`. Frames 0/20/60/100 show the same face,
beret and cream coat without the green speckles of the earlier run, though the
framing is a closer crop and the texture is still soft at 256x256. The lower
`spatial_corr` is not a quality verdict: that gauge measures horizontal
adjacency inside the latent, not sharpness or reference similarity. Wired
weight bytes stayed at 66.28 GiB; `cwick:host_swapouts_total` rose by 988 host
pages (16 MiB) over the whole session while other processes ran, far too small
and too scattered to be the mlocked model pages. Neither 30-step run should be
read as a 768p quality claim: 256x256 is the checkbox smoke test.

## §104 QuickTime black frames came from a malformed avcC, not the generator

A 256x256 / 30-step Ref2VA mp4 looked correct when decoded by ffmpeg
(clear, moving person in extracted frames) but opened as a black window in
QuickTime. That split pointed at the decoder path, not DiT or VAE output: the
generators had already been fixed in §99 and §103.

A small probe using AVFoundation — the decode stack QuickTime itself uses —
reproduced the failure deterministically: `AVAssetReader` returned
`AVFoundationErrorDomain -11821 Cannot Decode`, underlying OSStatus `-12137`,
on the first sample of every cwick mp4. The same files decode fine in ffmpeg
because ffmpeg rebuilds missing parameter sets from the in-band SPS/PPS NALs
VideoToolbox writes before each IDR; a strict `h264_mp4toannexb` conversion
that trusts only the container header failed with `Global extradata truncated,
corrupted stream or invalid MP4/AVCC bitstream`.

The root cause was in `build_avcc()` in `h3v_mp4.m`. The format-description
API returns one flat list of parameter sets (SPS followed by PPS, two entries
for this encoder). The code wrote that list count into avcC's
`numOfSequenceParameterSets` and then emitted the flat list, omitting the
separate `numOfPictureParameterSets` field entirely. The resulting record was
`01 64 00 14 ff e2 000a<SPS> 0004<PPS>`: it claimed two SPS units, fed the PPS
as the second SPS, and left the PPS-count byte absent. ISO/IEC 14496-15
requires two separate runs — numOfSPS + SPS units, then numOfPPS + PPS units.
Strict parsers reject the malformed record and render black; lenient ones
recover from the in-band NALs.

`build_avcc()` now classifies each parameter set by its NAL type (7 = SPS,
8 = PPS), collects them into two arrays, and emits both runs with the correct
counts. The fixed record is `01 64 00 14 ff e1 000c<SPS> 01 0004<PPS>`.

Verification used a synthetic colour ramp through `tools/h3vmuxprobe`, so the
muxer is exercised in seconds without minutes of DiT/VAE work:

- avcC bytes now carry `e1` (one SPS) and a trailing `01` PPS count;
- the AVFoundation probe decodes the first frame (`OK 64x64 mean_luma=87.07`)
  where the old files returned `Cannot Decode`;
- strict `h264_mp4toannexb` conversion of the container header succeeds.

This fix is the precondition for app 3's inline preview and download: a
visually correct file that QuickTime or Safari refuses to decode is a black
player to the user. The app's own copy was also brought in line with the
current video semantics: the reference-image checkbox no longer calls the
field `reference_image` (the wire field is `image`) and no longer claims the
image replaces the first-frame latent — Ref2VA folds it into the text
condition as Picture 1 (§103). The job's `variant` is now surfaced in the
progress panel so the Ref2VA default is visible end to end.

A real generation confirmed the muxer fix outside the synthetic probe: job
`video_1795373529_1` (256x256, 4 s, 4 steps, `sample.png` reference, Ref2VA
default) completed in 488.2 s (text 69.1, DiT 356.7, video VAE 56.1, audio VAE
4.1, mux 0.3) and produced 615,030 bytes / 102 frames. Its avcC is
`01 64 00 14 ff e1 000a<SPS> 01 00 04<PPS>`; the AVFoundation probe returns
`OK 256x256 mean_luma=156.36` where the pre-fix files returned `Cannot Decode`,
strict Annex-B conversion succeeds, and extracted frames show the referenced
woman, beret and coat with no black frame and no green grid seam. Softness at
256p with 4 steps is a resolution/step-count limit, not a decode failure.

## §105 Host memory pressure during video jobs: per-stage residency without a shared budget

The 30-step 256x256 Ref2VA job ran DiT at 200–241 s/step (216072, 206066,
201178, 199627, 227345, 241420 ms for steps 1–6), 2.4x slower than the 89.2
s/step of the preceding 4-step job — same canvas, same sequence length (3677
rows), same wired DiT (61.73 GiB). Host metrics collected through `/v1/metrics`
explained the gap:

- at the 02:51:49 boot, swap was 4.28 / 5.00 GiB and
  `host_compressions_total` = 2226116165;
- at the 03:28:15 boot, after both video jobs, swap was 6.61 / 8.00 GiB
  (macOS grew the swap file 5 -> 8 GiB) and compressions = 2279810266;
- that is **+2.33 GiB swap and +53.7M compressions during the two video jobs**;
- the transient unmap log shows the DiT-sampling RSS peak was 73.92 GiB
  (`RSS 73.92 -> 12.20 GiB` when the transformer closed);
- at idle with only the root model resident (`process_footprint_bytes` =
  28.5 GiB), a 5-second scrape showed zero compressions/swapouts delta — the
  pressure is created by the video job, not by the idle server.

Root cause. `cwick_mem_fit_model()` evaluates a single stage at a time against
the 102.40 GiB (80% of 128 GiB) ceiling and has no cross-stage accumulator.
The video line calls it once per stage, so each stage independently reports
"fits under budget" and is wired. `h3v_model_open()` wires the video VAE
(9.70 GiB) and audio VAE (0.56 GiB) at model-open and holds them for the whole
model lifetime; `h3v_generate()` then opens and wires the DiT (61.73 GiB) on
top for sampling. The concurrent mlocked set is therefore ~72 GiB (73.92 GiB
RSS peak), not one stage at a time. Wired pages cannot be compressed or
paged, so the pressure is shifted onto every other process on the host: the
global compressor and swap file climb. Non-wired scratch (attention buffers,
Metal arena temporaries) faults inside `waitUntilCompleted`, which is the
likely source of the 2.4x step-time regression. This is an association, not a
strict A/B: the two jobs ran back-to-back in one process, so accumulated
system state is a confounder.

A later job confirmed the extreme end of this curve: `video_1795375995_1`
(736x1280, 30 steps, seq 29357 = 8x the 256p case) ran DiT at 731656 ms on
step 11 rising monotonically to 788257 ms on step 30 — 6.3 h total, and
~13 min/step versus 89 s/step for a 4-step 256p clip. The per-step drift
within one run, with identical weights and rows, is the signature of a
page/compression fault rate climbing as system pressure builds.

Proposed fix, consistent with the goal constraints (no pruning, no precision
loss, Metal, 80% budget, no swap): make VAE residency lazy — fault in the
video/audio VAE at phases 4/5 and unwire them afterwards, so DiT sampling
holds only 61.73 GiB (peak 73.92 -> ~63 GiB); add an `already_resident_bytes`
accumulator to `cwick_mem_fit_model()` and one ledger shared by all live video
stages; treat the 80% ceiling as a global budget across stages, not a
per-stage test. Verify with a 4-step A/B comparing RSS peak, compression delta
and DiT s/step before promoting any default. Not yet applied — diagnosis only.

## §106 Blocky video at large canvases came from too-small VAE decode tiles

A 736x1280 / 30-step Ref2VA clip (`video_1795375995_1`, sample.png reference)
opened correctly in QuickTime after §104 but looked blocky: faces read cleanly
while clothing and outlines carried a regular grid. This was not the black
frame, not a decode failure and not undersampling — the job ran all 30 steps.

Evidence, in order:

- ffprobe: 736x1280, H.264 High, yuv420p, 4.71 Mb/s, 102 frames. avcC was
  already the valid §104 form (`ff e1 ... 01 00 04`), and the AVFoundation
  probe returned `OK 736x1280 mean_luma=63.89`; strict Annex-B conversion
  passed. So the container and decoder were fine.
- The job was an outlier on time: DiT ran 731656 -> 788257 ms per step (6.3 h
  total) because the canvas produced 27600 video rows / seq 29357 (8x the
  256p case) under the §105 residency pressure. Slowness and blockiness are
  separate symptoms; only the latter is addressed here.
- Extracted frames had a gradient-energy peak at lag 16 px (one latent cell,
  `16` pixels per cell) and at lag 192 px (one VAE decode tile). Horizontal
  tile boundaries at x=160/320/480/640 showed 2.4-3.6x the neighbouring
  gradient. That periodicity points at the VAE decoder, not the DiT.
- `tools/h3vencprobe` at 736x1280 round-trips a smooth ramp straight through
  the VAE with no DiT at all. The reconstruction reproduced the seams: mean
  boundary/non-boundary gradient ratio 4.14x horizontally, 5.33x vertically.
  Decoding a **constant zero latent** also produced a 16 px cell texture and a
  tile-period step, confirming the artefact is introduced by the decoder's
  tiling, not by the sampled latent content.

Root cause. The video VAE decoder cuts the latent into 12x12-cell tiles with
a 2-cell overlap. Its rotary tables use tile-local coordinates
(`2*(i+0.5)/n - 1`), so each tile spans the full [-1,1] range and adjacent
tiles predict the same row with different phases. A 2-cell (32 px) linear
cross-fade cannot hide that discontinuity; the boundary leaks as a 192 px
grid. 256p has a 16x16 latent and is effectively one tile, which is why the
problem was invisible in every earlier 256p test.

A first attempt rewrote the stitch to mirror the image VAE's validated
`_stitch_tiles` (whole-row raw buffer, in-band bidirectional blend, trim far
edges, quantise last). It was rejected by measurement: round-trip correlation
stayed 0.9913, boundary ratios did not improve (one variant made x seams much
worse at 35x), so the change was fully reverted. Shifting blend weights was
not the fix; the tile geometry itself was.

The working fix is to enlarge the tile. `H3V_VAE_TILE_CELLS` 12 -> 48 and
`H3V_VAE_TILE_OVERLAP` 2 -> 8, both overridable at runtime with
`CWICK_H3V_VAE_TILE=N` (overlap becomes N/6). Measured on the same 736x1280
smooth-ramp round trip:

| tile cells | seam ratio x | seam ratio y | R corr | decode time |
|---|---|---|---|---|
| 12 (old) | 4.48 | 5.33 | 0.9913 | 56.0 s |
| 24 | 1.89 | 6.50 | 0.9928 | 42.1 s |
| 48 (new) | 0.00 | 1.61 | 0.9937 | 31.8 s |

Larger tiles also decode faster (fewer overlapping passes) and keep the
attention sequence small enough for an 80x46 latent. After recompiling with
the new default, the boundary gradient at x=640 / y=640 was 0.63 / 0.71
versus 4-10 before. No weights, precision or sampler steps changed.

The end-to-end 736x1280 / 4-step clip has since completed
(`/tmp/h3vprobe/fix736.mp4`, 102 frames / 4.25 s, 2.46 MB). The sampler
reported `spatial corr 0.979 / clean corr 0.981 / snr -0.07 dB /
suspect_noise 0`; six extracted frames (n=0/20/40/60/80/101) show three
separate outcomes:

- The 192 px tile seam is gone. A frame-for-frame comparison against the
  earlier tile=12 / 30-step clip shows the horizontal bars at
  x=160/320/480/640 no longer appear. The tile geometry fix holds
  end-to-end, not just on the smooth-ramp round trip.
- A fine 16 px checker/dot texture still covers the whole frame. Magnified
  crops show a regular cell-and-triangle pattern on skin, hair and clothing
  alike. This is the decoder's cell-level artefact already reproduced by
  decoding a constant-zero latent; enlarging the tile does not address it.
  §106 fixes the large tile grid only.
- The 4-step clip is visibly softer than the 30-step one: fabric weave and
  facial detail no longer resolve. That is undersampling, independent of
  the tile change.

Identity is partly preserved (same long-haired, smiling subject with a
gentle head turn across the clip) but the reference outfit is not:
`sample.png` is a cream fuzzy coat / slip dress / knee boots, while the
generated frames show a plain dark turtleneck. Four steps carry too little
reference detail through the text-presentation path. No black frames,
green blocks or one-sided grids remained.

Conclusion: on a 736x1280 canvas, sharpness is now limited by sampler steps
and the VAE's intrinsic 16 px cell texture, not by tile seams. More steps
are the only in-pipeline lever for detail, at the measured ~6 h cost. The
run also re-confirmed §105 live: free pages fell to ~27k (436 MiB) with swap
at 5.6/7.2 GiB; DiT took 752-813 s per step and the VAE decode 979.7 s
(4307 s total).

## §107 256x256/30-step clip: QuickTime softness is a resolution ceiling, and the §105 step-time cause was misattributed

The clip `output/cwick-video-video_1795406010_1.mp4` is 256x256 / 102
frames / 4.25 s / 30 steps, Ref2VA with `sample.png` as reference and
`sample_video.txt` as prompt, served through `/v1/videos` under Metal.

Container is not the cause. avcC is the valid §104 form
(`01 64 00 14 ff e1 00 0a<SPS> 01 00 04<PPS>`: one SPS, one PPS), H.264 High
plus AAC 32 kHz stereo, packet DTS runs 0..101000 monotonically at 24 fps,
and a full decode produced no frame errors. Any softness seen in QuickTime
is the generated image, not muxing, avcC or timestamps.

Six extracted frames (n=0/20/40/60/80/101): identity and wardrobe now match
the reference (cream fuzzy coat, slip dress, beret) — the 30-step run carries
reference detail that the earlier 4-step clip dropped (§106). No tile seams,
green blocks or black frames. A mild 16 px cell texture remains on face and
background and the whole frame is soft. Both are expected at this canvas:
256x256 is the minimum verification canvas and maps through the VAE's 16x
spatial compression to a 16x16 latent, so the decoder's intrinsic cell
texture (§106, reproduced by decoding a zero latent) occupies a large
fraction of every displayed pixel, and any full-screen playback upscales
256x256. This is a resolution ceiling, not a regression: no sampler or
wiring change makes a 16x16 latent render like an 80x46 one.

Log and metrics: text 112110.5 ms (1417 tokens after Ref2VA presentation),
DiT 30 steps at 60-63 s/step = 1979.5 s, denoise `spatial corr 0.847`,
`gen/ref std -0.74 dB => ok`, video VAE 54114.3 ms, audio VAE 3951.9 ms,
mux 200.9 ms, 2196.4 s total. Peak RSS was 73.52 GiB (unmapped
`73.52 -> 11.79 GiB`) with the same wired set as §105 — VAE 9.70 GiB and
audio VAE 0.56 GiB held from model-open, DiT 61.73 GiB during sampling.
At job end free pages were 4.8M (95% free), swap 3930/5120 MiB, compressor
339 MiB: no host memory pressure. The serve log recorded 2096 `/v1/metrics`
scrapes, all HTTP 200 at 25133 bytes; the process had stopped by the time
of this analysis, so no live scrape was available.

**Correction to §105.** The same 256x256/30-step shape, same 3677 rows, same
61.73 GiB wired DiT and the same 73.52 GiB RSS peak now ran at 60-63 s/step,
versus the 200-241 s/step recorded in §105. The wired set was identical; what
differed was host swap/compression pressure. The §105 step-time regression
was a function of system-wide pressure, not of the VAEs being wired early.
The proposed lazy-VAE-residency fix is therefore not supported by this
comparison and was **not applied**. Keep the existing residency model; to
protect step time, finish or clean up other swap-heavy work before a long
video job instead.

Practical takeaway: 256x256 is a correctness/speed smoke test, not a quality
setting. For detail, raise the canvas — at low host pressure it is far from
the 6.3 h outlier in §105. For speed, lower the step count, but at 4 steps
the reference wardrobe is lost (§106). The 16 px cell texture is intrinsic
to the VAE decoder and is not tunable from the sampler.

## §108 QuickTime "frame skipping" was a temporal VAE seam, not a container fault

A 544x960 / 30-step Ref2VA clip (`video_1795410431_1`, 102 frames, 4.25 s)
looked like it skipped frames in QuickTime. Three independent checks ruled
out the container: the packet table has one stts entry with a uniform
1000/24000 s step and strictly increasing dts; avcC carries one SPS and one
PPS; and an AVFoundation AssetReader decoded all 102 frames with a single
interval value of 0.041667 s and zero abnormal gaps. The `non monotonically
increasing dts` warnings from `ffmpeg -f null` come from the null muxer
re-encoding the decoded stream, not from the source file. Nothing is wrong
with timestamps, keyframe placement (IDR at 0/48/95), or MP4 boxes.

The jumps are in the pixels. Measuring mean absolute difference between
consecutive decoded frames (median 4.52) shows a clean periodic impulse: the
transitions into frames 20, 40, 60, 80 and 100 average **21.34**, 2-14x every
other bucket, with the immediately preceding frame at 1-4. Smaller peaks sit
at frames divisible by 4 (9-11.6). The period is exactly 20 frames = 5 latent
steps = one temporal chunk; the 4-frame peaks are individual latent steps.

Root cause: the decoder geometry folds 17 pixel frames into 5 latent steps
(17n+5 pixel frames <-> 5n+2 latent frames). Adjacent latent chunks overlap
by 2 latent steps, and a correct decode decodes that overlap and linearly
cross-fades the shared frames. The decoder in this engine decodes all latent
steps in one pass and cross-fades only the spatial tiles; there is no temporal
overlap decode and no temporal cross-fade. Each 20-frame boundary is therefore
a hard cut, which reads as a visible jump every ~0.83 s. Frame count 102 is
also not on the checkpoint's legal 17n+5 grid (nearest: 90 or 107).

Fix direction (not applied): decode each chunk with a 2-latent overlap and
blend the shared frames, mirroring the spatial tile stitch. This changes the
VAE decode loop and intersects the spatial tiling, so it needs its own A/B
and regression before landing. A cheap mitigation is to request a legal frame
count, which removes the extra truncation but not the hard-cut seams.

Log/metrics: text 112.1 s, DiT 30 steps at 363-409 s/step = 11093.8 s, video
VAE 465.0 s, audio VAE 3.9 s, mux 0.3 s, 11678.2 s total. Denoise was healthy
(`spatial corr 0.901`, `gen/ref std -0.43 dB => ok`). The 6x step time versus
the 256x256 run in §107 is the expected cost of ~4.2x the transformer rows
(15300 vs 3677): attention grows quadratically. Host was at 95% free with
swap 3691/5120 MiB before and after, so no memory pressure was involved.

## §109 Temporal VAE overlap + cross-fade removed the chunk-boundary hard cut

§108 traced the QuickTime "frame skip" to the visual VAE decoding all latent
steps in one pass with no temporal overlap, leaving a hard cut at every 5-step
chunk boundary. The training rule folds 17 pixel frames into 5 latent steps
and drops 3 tail latents per encoded chunk, so a correct decode re-decodes a
2-latent overlap and linearly blends the shared 5 frames; rope is
length-normalised over the chunk's own 7 latents. The whole-clip decode did
neither.

Fix: `h3v_vae_decode` now walks the clip in 5-step chunks with a 2-step
overlap. Each chunk decodes 7 latents with local rope, contributes 17 main
frames taken from decoded frames [3,20), and blends its first 5 frames with
the previous chunk's 5-frame tail at weights 1->0 / 0->1. The buffer holds
17*chunks+5 frames and the mux keeps the first 17*chunks, so every muxed frame
now comes from a cross-faded main chunk. `lt<=5` and
`CWICK_H3V_TEMPORAL_XFADE=0` keep the whole-clip path for the reference-frame
round trip and for A/B. Spatial tiling and its own cross-fade are unchanged
and run inside each chunk.

A/B on the same latent (256x256, 1 step, seed 42, Metal, only the switch
differs): boundary MAE 5.358 -> 4.212 (-21%), single-frame peak 6.05 -> 4.69
(-23%), 5-frame energy 11.57 -> 8.40 (-27%). The old one-frame impulse (5.43
into frame 20, then 0.43 into 21) becomes a 5-frame ramp (4.57, 2.48, 0.34
into frames 17/18/19). Denoise was identical (`spatial corr 0.987` in both)
and the contact sheet shows no new spatial seams, black frames or block
artifacts. VAE decode rose 39.2 -> 53.1 s (+35%) for the redundant overlap
work. Total wall time is not comparable across the two runs: the DiT stage
varied 126.5 -> 85.7 s independently of this switch.

The blend's first weight is 1 by the training rule, so the first shared frame
still jumps ~4.5; that residual is the sampler's own 5-step boundary, not a
VAE seam. Removing it would need more overlap latents than the checkpoint was
trained with. `make test` PASS.

## §110 First real 30-step clip after the temporal cross-fade fix

The 256x256 / 30-step Ref2VA clip `video_1795424360_1` (102 frames, 4.25 s,
24 fps) is the first full-quality clip produced after §109 landed. It was
checked for the two symptoms reported earlier: timeline jumps and blocky
texture.

Container is clean. stts has one entry, pts==dts runs 0..101000 in steps of
1000, avcC carries one SPS and one PPS, and an AVFoundation AssetReader
decoded all 102 frames at a single 0.041667 s interval with zero abnormal
gaps. The `non monotonically increasing dts` line from `ffmpeg -f null` is the
null muxer re-encoding, as in §108.

The frame-to-frame MAE (median 4.46) shows none of the old 20-frame impulse:
the transitions into frames 20/40/60/80/100 are 3.17/3.76/7.48/6.51/2.66
(mean 4.72), and the global maximum is 12.53 at frame 83. Against the
pre-fix 256x256 clip (same 30 steps, different seed), boundary mean fell
20.08 -> 4.72 (-76%) and the max fell 33.17 -> 12.53 (-62%). The two largest
luminance changes (dL 8.77/8.43 at frames 85/84) coincide with the subject
turning from front to profile, which the prompt asks for; no consecutive
frames are duplicates (min MAE > 0.5). Contact sheets show no black frames,
green blocks, grid seams, ghosting or frozen frames, and the fur coat and
facial features are clearly resolved at native size.

This is the first evidence that the §109 path holds at real sampler quality,
not just on the 1-step A/B: cross-fade removed the periodic jump without
adding spatial seams or new blur. The clip does differ in composition from
the pre-fix one (different seed), so the overall sharpness gain is not
attributed to the switch — only the boundary-impulse change is.

Log/metrics: text 84.45 s, DiT 30 steps at 60-67 s/step = 1971.1 s, video VAE
63.77 s, audio VAE 3.97 s, mux 0.26 s, 2125.73 s total. Denoise healthy
(`spatial corr 0.847`, `gen/ref std -0.74 dB => ok`). Video VAE rose from
54.1 s (§107) to 63.77 s, the expected cost of decoding the 2-latent overlap.
Counters: 1 job, 1 succeeded, 0 failed, 0 suspect noise; resident 66.28 GiB,
wired, RSS peak 73.97 GiB. Host swap was 8.06/9.00 GiB at startup and fell
to 6.4/7.5 GiB at the end — a high boot-time baseline, but step time stayed
at the healthy 60-63 s level, again confirming §107: step-time regressions
track pressure *during* a job, not the absolute swap level.

## §111 Why the face is soft at 256x256, and what can actually move it

The same 256x256 / 30-step clip in §110 was checked specifically for a soft
face. The blur is resolution, not a defect in sampling, the cross-fade, or
the muxer. Three independent facts locate it.

1. Output grid. The VAE compresses space 16x, so a 256x256 canvas is only a
   16x16 latent. The face occupies roughly a 4x4 patch of that grid; there is
   no detail in the latent to recover. Laplacian variance over the same
   face-sized crop is 3991 on the reference downscaled to 256 but 1730-2547
   across the generated frames — about half the high-frequency energy before
   generation even starts, and the decoder cannot invent the rest.

2. Reference conditioning rows. The first-frame latent path encodes the
   reference at the *generated canvas* unless `CWICK_H3V_REF_EDGE` is set.
   The log is explicit: `reference image encoded: 256x256 RGB -> 1x24 latent
   (16x16)`, while the same image arrives as 768x1024 RGB. The identity
   signal that path can carry is capped at 16x16 latents.

3. Text presentation is already high-resolution and is not the bottleneck.
   `h3_text_cond_ref_dir` runs the reference through the vision tower at a
   1024 short edge (`h3_vision_size` -> `h3_bound_edge(1024,1024)`), giving
   1417 text tokens. This is why identity and the fur coat still match the
   reference: the structure comes from the high-res presentation, while the
   per-pixel softness comes from the 16x16 output grid.

The face comparison confirms the form: facial structure, eye count and
expression are all correct, with no mosaic or ghosting — it is uniformly
soft, which is what a 16x16 grid reconstructed to 256 pixels looks like. Frame-to-frame MAE (§110) and
cross-fade A/B (§109) already ruled out temporal seams as a cause.

Optimization boundary under the goal.md constraints: no sharpening filter,
no super-resolution model, no precision or pruning changes are applied. The
only in-engine lever that raises reference-detail transfer is
`CWICK_H3V_REF_EDGE=N`, which scales the conditioning rows to the
reference's own short edge instead of the canvas; it is a process-level
env var and needs a service restart. The checkpoint presents references at a
2048 short edge, and this build admits the full `H3V_REF_EDGE_MAX` (6272),
so `CWICK_H3V_REF_EDGE=2048` reproduces the released presentation (e.g. a
1152x1536 reference becomes 2048x2720) instead of being clamped to the
generated-canvas cap (now 2560, and independent of the reference range).
A larger short edge adds conditioning rows and DiT time grows
roughly quadratically; at 2048 the transient VAE-encode buffers are ~46 GiB and
must be budgeted together with the resident weights under 80% / no-swap.
The real fix for face detail is still a larger canvas (512x512 gives a 32x32
latent, ~4x the detail capacity), at the same roughly quadratic DiT cost.
A controlled A/B of `REF_EDGE` was not run here: the service holds the resident
model and a second DiT instance would breach the 80% / no-swap budget; only
the weight-free size arithmetic was unit-tested (`make test` passes).

## §112 App 3 binds both canvas sizes through one shared size-pair table

App 3 (`apps/imagegen`) already uploaded face/outfit/pose references, called
both image endpoints, polled the async video job, and offered a quick-verify
preset. The remaining gap was that step 1's `size` and step 2's `size` were
independent: a 1536x2048 portrait render could be handed to a 1280x720 video
request, so the conditioning image and the generated frames did not share a
framing.

Both steps now read one `SIZE_PAIRS` table in `apps/imagegen/app.js`. Each row
carries an image canvas and an aspect-ratio-matched video canvas; every video
size is an exact multiple of `H3V_CANVAS_MULTIPLE` (32) inside 256..2560 and
every image size is inside 256..6272 (`H3_MIN_SIZE`/`H3_MAX_SIZE`). The seven
rows are 256x256→256x256 (quick verify), 1024x1024→1024x1024,
1536x2048→768x1024, 1024x1536→768x1152, 2048x1536→1280x960,
2048x1152→1024x576 and 2560x1440→2560x1440 (the 2K upper bound from §102).
Choosing a row in step 1 mirrors the video size into step 2 and updates a
visible pair hint; entering step 2 from a generated image re-asserts the pair
in case the video select was changed by hand. The quick-verify button now sets
both canvases plus 4 s and 4 steps in one click, so the minimum canvas and
shortest clip are available to both steps at once.

Manual video sizes remain in a separate `optgroup` labelled "extended canvas
(not paired)", so the larger canvases stayed reachable; selecting one clears
the pairing and the hint states that explicitly. No server code changed: the
image endpoints accept any `WxH` in their range and `/v1/videos` keeps Ref2VA
as the only variant. `make src/webui_embed.c && make cwick` refreshes the
embedded UI; `make test` adds two assertions that the shared table and both
select groups reach the binary, and `node --check` validates the JS. No render
was run here: the service holds resident weights and a second instance would
breach the 80% / no-swap budget.

## §113 Symmetric video teardown in the image and realtime switches

Log evidence from a session that mixed image edits with a video clip: switching
to the image model left the video VAEs resident, so the 61.74 GiB image DiT
stacked on the 10.26 GiB video side for a ~72 GiB RSS. At 18:58:07 loading the
video DiT back logged `could not wire 4.81 GiB ... (Resource temporarily
unavailable)` — `mlock` returning EAGAIN — and the first sampling step took
171.9 s against a 60–67 s healthy value. `cwick_mem_lock_range` logs only the
first failure, so 4.81 GiB is a lower bound. The "wire resident ... fits under
budget" line just above it reports the *attempted* byte count, not the pinned
bytes; do not read it as success.

Root cause: `srv_switch_model` and `srv_switch_video` released all three
other weight banks, but `srv_switch_image` released only text and realtime and
`srv_switch_realtime` only text and image. A video pipeline therefore survived
a switch away from it. `swap_gate_open` already refuses when `vid_busy` is
non-zero, so no clip is reading the shards when the switch tears them down.

Both switches now call `srv_release_video_model()` before opening the incoming
checkpoint, and include `H3V_SERVED_ID` in the "releasing" log line. The four
switch paths are now symmetric: each releases the other three banks.
`make cwick` is clean and `make test` passes; no render was run for an A/B
because the service holds resident weights.

## §114 B frames without ctts made QuickTime show frames out of order

A 256x256 / 30-step clip (`video_1795432400_1`) looked unstable in QuickTime:
the picture jumped instead of flowing. ffmpeg extracted the same jumps, so
the encoder/container was at fault, not the player or the H.264 bitrate
(1.0 Mbps at 256x256 is ample). Frame-level probe of the shipped MP4 showed
56 B / 43 P / 3 I frames with non-monotonic presentation stamps
(0, 3000, 2000, 4000, 1000...), two duplicate tail timestamps and no `ctts`
box; the muxer assumed `AllowFrameReordering=0` meant no B frames and stamped
every sample's PTS equal to its DTS. Disabling output reordering does not stop
VideoToolbox from *coding* B frames.

Root fix: pin the encoder to Baseline (`kVTProfileLevel_H264_Baseline_AutoLevel`,
avcC profile 0x42), which forbids B frames, and additionally emit a `ctts`
table from each sample's `CMSampleBufferGetPresentationTimeStamp` as a guard
(all offsets are zero in a B-free stream, which is legal). The muxer now writes
`ctts` for the video track only.

`h3vmuxprobe` A/B at 60 frames: before = 43 B frames, unsorted PTS, 58 unique
display timestamps; after = 0 B frames, 60 unique PTS, strictly monotonic
0/1000/2000... . At 256x256/30 frames: 1 I + 29 P, 30 unique sorted PTS.
`make test` adds assertions that avcC reports Baseline and all ctts offsets are
zero. The running service was started with the old binary, so a restart is
required for new clips to use the fix; the existing MP4 cannot be repaired by
remuxing because its B frames are already in the bitstream.

## §115 stss marking every frame sync made QuickTime ghost P frames

A 256x256 / 30-step clip (`video_1795439347_1`) looked correct when decoded
with ffmpeg but ghosted in QuickTime. §114's B-frame fix was confirmed active
on this file: Baseline (avcC 0x42), 3 I / 99 P, no B frames, and 102 unique
monotonic PTS with zero ctts offsets. The remaining difference was the sync
table: stss listed all 102 samples as random-access points although only three
were IDR.

Root cause: `avcc_first_nal_type()` inspected only the first NAL of each AVCC
sample, but VideoToolbox prefixes each IDR access unit with an SEI (type 6),
placing the IDR slice (type 5) second. No sample tested positive, and
`put_stbl`'s fallback (`n ? n : n_samples`) then marked every sample sync.
QuickTime trusts stss and restarts decode at each advertised sync point;
starting at a P frame with no reference leaves the previous picture behind as
a ghost. ffmpeg ignores stss, which is why frame extraction looked fine.

Fix: `avcc_has_idr()` scans every NAL in the sample for type 5; the stss
fallback now marks only the first sample when no IDR is present, never all.
`h3vmuxprobe` at 102 frames: sync count 102 -> 3, at samples 1/49/97 (the
keyint=48 IDRs), matching the bitstream. `make test` adds two assertions that
stss lists only IDR samples and does not advertise P frames as random-access
points. As in §114, the running service predates this binary; new clips need a
restart, and an existing MP4 cannot be repaired by remuxing alone because the
wrong sync table is already written (a clean remux with the fixed muxer from
the elementary streams would be needed).
