# Decode performance: where the 0.72 s step goes

<p align="center"><a href="PERF.md">English</a> · <a href="PERF.zh-CN.md">简体中文</a></p>

Everything here was measured on the reference machine — M3 Max, 16 cores (12
performance), 128 GiB unified memory — with the full GLM-5.3-flash checkpoint
(45 layers, 288 routed experts, top-8, FP8-E4M3 on disk), `--backend metal`,
`--context 131072`. Numbers are for **one decode step** (a single token), taken
from `CWICK_PROFILE=1`, which prints a phase breakdown at the end of each
request.

Two numbers decide everything:

- **the complete routed-expert bank is 290.32 GiB** — 42 sparse layers x 288
  experts x 3 matrices x 8.0 MiB — and it does not fit in the 80% budget
  (102.40 GiB) no matter how the plan is cut;
- **one decode step touches 8.06 GiB of it** (42 x 8 experts x 3 x 8 MiB).
  That is the whole cost of a step: everything else is a rounding error.

## The measurements

| condition | ms/step | tok/s | stream | gemv | attn |
|---|---|---|---|---|---|
| Metal backend, experts resident (10-token reply) | **275** | **3.63** | 4.9 | 120 | 117 |
| Metal backend, 265-token prompt, 80 steps, 83 GiB cache (2026-10-20) | **590** | **1.69** | 271.6 | 151.1 | 134.4 |
| same + `CWICK_PREFETCH=1` (2026-10-20, QoS fix in) | **2111** | **0.47** | 490.8 | 712.3 | 284.2 |
| Metal backend, 265-token prompt, 79 steps, 83 GiB cache (2026-10-19) | 714 | 1.40 | 362 | 177 | 142 |
| Metal backend, same, 64 GiB cache | 798 | 1.25 | 462 | 208 | 151 |
| CPU backend (`--backend cpu`), cold | 1195 | 0.84 | 711 | 291 | 136 |
| Metal backend + `CWICK_PREFETCH=1` (2026-10-19, before QoS fix) | **1873** | **0.53** | — | — | — |

The 714 and 590 rows are the same configuration on different days: machine
state (thermals, other load) moves the step by ~20%. Treat 590 as the current
honest baseline and 714 as the worst case.

Four things fall out of that table.

1. **The runtime reaches 3.6 tok/s** when the experts a reply needs are in RAM: a
   step is 275 ms, and at that point neither the arithmetic, the attention nor the
   HTTP layer is the limit — the storage path is, as soon as a request has to stream.
2. **`--backend metal` is doing real work.** The expert GEMVs drop from 291 ms
   to 120 ms per step on the GPU (2.4x). The batch is accepted zero-copy for
   arena- and mapping-backed blobs, so there is no 8 MiB memcpy per matrix.
3. **The prefetch is a trap, and the QoS fix did not rescue it.** The 80-step
   A/B above is the verdict: `CWICK_PREFETCH=1` went 714 -> 1873 ms before the
   `QOS_CLASS_BACKGROUND` change and 590 -> 2111 ms after it. The prefetch arm
   reads 2.7x the bytes (647.6 vs 236.8 GiB, 95396 vs 30315 reads) for *zero*
   hit-rate gain (68.6% vs 68.5%), its demand-side pread/dispatch counters go
   to zero, and the MoE phase explodes (151 -> 712 ms) from cache/arena
   thrash. It stays opt-in, default off.

   The cause was always cores, not bytes: the demand path reads a whole layer
   with a twelve-worker fork-join (`moe.c:348-367`), a fork-join GEMV is paced
   by its slowest participant, and a prefetch thread at the default QoS
   preempts those workers mid-kernel — so the layer ends up slower than if the
   read had never been issued. The demand readers cannot be raised any
   further (they are pool workers inside a fork-join whose GEMVs already run
   `USER_INITIATED`), so `pf_reader` instead drops the *speculative* thread to
   `QOS_CLASS_BACKGROUND` (`stream.c:1110-1129`) — the one class the scheduler
   will not let preempt `USER_INITIATED` work.
   The A/B says the mechanism was not the problem; the speculative *bytes*
   were.

## Why streaming costs 362 ms and not 20 ms

At 83 GiB the cache holds ~10 600 of the 36 288 blobs (29%). Zipfian routing
rescues most of the rest: measured hits cover ~76% of the reads, so a step
streams ~1.9 GiB. 1.9 GiB / 362 ms = **5.4 GiB/s**, which is this SSD's
sustained read rate for 8 MiB random extents. The phase is at the hardware
limit, not at a software limit:

- `pread`, summed over the 12 reader threads, is 1572 ms/step; divided by 12
  that is 131 ms, but the fork-join wall is 359 ms. The gap is imbalance — one
  slow read per wave sets the wall — not idle threads.
- Slot reservation is 0.7 ms/step and cache probing 0.01 ms/step: the cache
  itself is not the problem.
- The OS page cache cannot help: expert fds are opened `F_NOCACHE` so that
  streamed bytes never evict the wired expert arena.

So the storage path is a bandwidth wall. Raising the cache from 64 to 83 GiB
buys 10% (798 -> 714 ms) and puts the footprint at 102.04 of the 102.40 GiB
ceiling, which is why the default stays at 64 GiB.

## The 714 ms row is a *warm* number — short replies pay a cold tax

The 79-step run above had time to warm the cache: by its end the hit rate was
~76% and only ~1.9 GiB/step was streamed. A 10-token reply measured on
2026-10-20 with the same 265-token prompt, same 83 GiB cache, same backend:

| | 10 steps (cold) | 79 steps (warm) |
|---|---|---|
| total | **979 ms/step** | 714 ms/step |
| stream | 614.5 | 362 |
| gemv | 208.7 | 177 |
| attn | 155.5 | 142 |
| hit rate | **17.3%** (4416/21153, prefill included) | ~76% |
| streamed | 165.26 GiB over the whole run | ~1.9 GiB/step |

Two consequences for the server:

1. **The first requests after a restart are ~37% slower per step** than the
   steady state, because the expert cache fills as requests route. The
   request-level `cache=` field in the summary line is the number to watch.
2. Cold steps also pay a one-off `alloc` cost the warm profile never shows:
   `shmem_alloc` was 189 ms/step (arena grows 21 times, `arena hot LIFO: 0
   hits / 4656 misses`). It disappears once the arena is at size.

Mitigation implemented 2026-10-20: `--warmup N` (alias `CWICK_WARMUP=N`) runs
one silent N-token generation on a throwaway prompt *before* the port opens
(`serve + serve_warmup` in `main.c`). It uses the CLI generation path, so it
never touches the prefix cache, and a failure is logged and ignored. The
generated text is discarded without decoding.

Measured 2026-10-20 (`tools/warm-ab.sh`, 253-token prompt, 10 generated
tokens, same machine, 83 GiB cache; identical output text both arms, so the
cache changes speed only):

| first request | decode ms/step | decode tok/s | hit rate | prefill wall |
|---|---|---|---|---|
| cold serve | 1304 | 0.77 | 13.2% | 96.9 s |
| `--warmup 32` | **939 (-28%)** | **1.07** | **39.6%** | 98.9 s |

The warmup itself cost 74.8 s for 32 tokens (1113 ms/step) — that *is* the
cold tax being moved off the first client request. Note prefill dominates a
253-token request (97 of 108 s end to end), so the win shows up in the decode
phase; short prompts and prefix-cache-hit prefill move the balance toward
decode, where warmup helps most. At 32 tokens the hit rate only reached 39.5%;
a longer warmup (64-128) buys more cache but pays proportionally more front
cost, and the expert set a prompt activates is prompt-specific anyway — the
durable fix for the steady state is hit-rate governance (below).

## Why prefill runs the MoE on the CPU

`batch windows: metal 0, ->cpu 190` in the profile is not a bug. Prefill
windows fetch into malloc'd staging blobs (the arena once turned a 43 s
prefill into one that never finished, `expert_view_into` in `moe.c`), and the
Metal batch only adopts no-copy arena/cache views, so prefill groups take the
CPU fork-join path by design. Forcing it (`CWICK_PREFILL_METAL=1`) was
measured: the batch wins 23% of the compute but makes the SSD about twice as
slow, which loses overall (`tensor.c:611-614`). Decode is unaffected — its
views are arena-backed by construction (`CWICK_GROUP_METAL`, default on).

## What would actually reach 3 tok/s while streaming

**MXFP4 expert packs are off the table** (decided 2026-10-20): halving the
expert bytes would take stream 271.6 -> ~140 and gemv 151 -> ~75 ms, i.e. the
projected 413 ms row below, but it costs a 290 GiB one-time repack and 4-bit
weights that are not bit-exact. See `docs/DECISIONS.md` §2 for the packing
contract that was worked out before the decision, in case it is revisited.

Without touching the bytes, the levers that remain, each measured or
measurable and none of them lossy:

| lever | expected | status |
|---|---|---|
| KDA fa/b/ga projections into the q/k/v GPU batch | -2.3 ms attn (moves work, does not remove it) | **done, measured** |
| o_proj per layer onto the GPU (batching across the recurrence boundary) | -20..-25 ms | blocked: it depends on the recurrence output, so it cannot share the q/k/v batch; a solo command buffer costs as much as the CPU pool |
| expert hit rate 68.5% -> 82% (cache grooming, evict tuning) | -115 ms stream | next |
| `--warmup N` pre-generation at serve start (default off) | first request -37% | **implemented 2026-10-20**, measured with `tools/warm-ab.sh` |
| GPU kernel clock/occupancy work | unknown | gemv 151 ms = 8.06 GiB/step = 45 GB/s, 8x under the unified-memory ceiling — the largest unmeasured candidate |

Order of work, by measured payoff:

1. **Hit-rate治理** (cache at the 80% ceiling, evict policy): -115 ms.
2. **`--warmup`**: removes the cold tax on early requests.
3. **GPU kernel profiling**: the gemv phase is 8x off the bandwidth ceiling;
   unlike everything else here, the headroom is arithmetic, not bytes.

Nothing here prunes the model, skips an expert, or reduces top-k: the router
still picks 8 of 288 per token and every one of them is applied. The honest
ceiling without MXFP4, from the 590 ms baseline: fusion + hit-rate + warmup
together land around 2.2 tok/s steady state; 3.0 tok/s needs the GPU kernel
work to pay off as well.

## Reproducing

```sh
CWICK_PROFILE=1 ./cwick serve --root ./zai-org/GLM-5.3-Flash --backend metal \
    --context 131072 --verbose --trace
# then, at the end of every request, the phases above are printed once
```

The request-level summary line is the one to watch in production:

```
prompt=265 gen=80 prefill=…/…tok/s decode=…ms/79 steps X tok/s cache=63.5%(…) streamed=…
```

`cache=` is the expert hit rate; `streamed=` is what the SSD produced. When
`X` falls below 1, `cache=` is the number to look at first.

## 2026-10-25: the step after the GPU batch stopped serialising

Same machine, same checkpoint, same `--backend metal --context 131072`,
26-token prompt, 60 decode tokens, `CWICK_PROFILE=1`. The 590 ms row above is
from 2026-10-20; this is today's build measured twice, before and after one
change.

| | ms/step | tok/s | stream | gemv | attn | rest |
|---|---|---|---|---|---|---|
| before | 727.8 | 1.374 | 390.2 | 170.6 | 132.7 | 34.3 |
| after | **663.0** | **1.508** | 379.3 | **126.4** | 124.2 | 33.2 |

The change is one function, `cwick_concurrent_encoder` in `metal_objc.m`.
Compute dispatches inside one `MTLCommandBuffer` are **serialised by
default** — dispatch *i+1* does not start until *i* retires — and both GPU
batches here consist of independent GEMVs writing disjoint output slabs, so
the default was costing exactly the overlap the batch exists to create.
`computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent` fixes it.
The generated text is byte-identical to the serial build (same md5), which is
the expected result: dispatch order was never part of the arithmetic.

A third run of the same build measured 682.4 ms/step (gemv 108.4, stream
409.2), so read the "after" row above as 1.47-1.51 tok/s rather than a
constant: the `stream` phase moves by 8% between runs on this drive while
`gemv` is stable to about the same. Everything below is a same-run comparison
for that reason.

Same run through the server (`sh ./start.sh glm-5.3-flash`, streaming, 200
generated tokens, three requests): decode **1.517 / 1.556 / 1.507 tok/s**,
end to end 1.31 tok/s. So the CLI number carries to the API path unchanged.

### What the GPU is actually limited by

`CWICK_METAL_BATCH_TIME=1` splits each wave into encode / commit+wait / GPU
time. A decode layer is two waves, 18 dispatches (eight experts' gate+up, the
shared expert's gate+up) and 9 (eight down, one shared down):

| kernel | GPU time, wave 1 | bytes | rate |
|---|---|---|---|
| v3 (threadgroup LUT), serial dispatch | 2.07 ms/layer | 192 MiB | 45 GB/s |
| v3, concurrent dispatch | 1.27 ms | 144 MiB | **118 GB/s** |
| v3d (same loads, `int`→`float` instead of a LUT) | 0.67 ms | 144 MiB | **227 GB/s** |

The third row is the interesting one, and it is easy to over-read. `v3d` is
the diagnostic twin of `v3` — identical loads, identical accumulator shape,
the only difference is that it reads no table — and it is **1.9x faster**. The
tempting conclusion is that the 256-entry `threadgroup` LUT is costing 1.9x
and should be replaced by arithmetic. That conclusion is wrong, and the next
section is the experiment that shows why: `v3d` replaces the decode with
*nothing*, so its 0.67 ms is the price of the loads alone and not a target any
correct kernel can reach. A real decode — arithmetic or table — is what puts
the kernel at ~110 GB/s.

### Decoding the byte is the floor, not the table

The obvious suspect for the remaining gap is the LUT's bank conflicts: 32
lanes index a 256-entry table with 32 unrelated bytes. So `v4` keeps v3's
slicing (8 output rows x 8 k-slices, 256 threadgroups) and swaps the table for
`cwick_fp8_v2`, a bit-exact branchless E4M3 decode built from shifts, one add
and one select — and extends the sliced shape to `k = 2048` so the down
projections get it too, which v3 cannot do. `CWICK_FP8_KERNEL=v3|v4`, same
prompt, 39 decode steps each:

| wave | dispatches | bytes | v3 (LUT) | v4 (arithmetic) |
|---|---|---|---|---|
| gate + up | 18 | 144 MiB | **1.396 ms** = 103 GB/s | 1.657 ms = 87 GB/s |
| down | 9 | 72 MiB | **0.799 ms** = 90 GB/s | 0.851 ms = 85 GB/s |

`gemv` goes 119.8 -> 151.4 ms/step; v4 is slower on both waves. A bit-exact
decode is ~13 integer ops per byte, and on this GPU a shared-memory gather
with a ~4-way bank conflict is simply cheaper than that. So:

- ~110 GB/s is where a *correct* fp8 GEMV lands here, and it is a decode
  bound, not a transport bound;
- 227 GB/s is what the transport does when nothing decodes the bytes;
- the 1.9x is therefore **not** available, and the item was removed from the
  work list below rather than left as an invitation.

`v4` stays compiled and tested behind the env var so the comparison is one
command to repeat instead of a rewrite. The negative result is the artifact.

### Why the expert cache has no headroom left

The obvious lever is the hit rate: a decode step touches 42 layers x 8 experts
x 3 matrices x 8 MiB = **8.06 GiB**, and at 50% every byte above the hit rate
is an SSD byte. `CWICK_EXPERT_DUMP` writes the routing trace, and replaying it
gives the bound any policy is playing against — the hit rate of a cache that
holds each layer's *most frequently routed* experts, i.e. the limit a perfect
frequency policy converges to:

| experts/layer | cache | oracle hit | SSD GiB/step |
|---|---|---|---|
| 16 | 15.8 GiB | 48.1% | 4.18 |
| 32 | 31.5 GiB | 65.3% | 2.80 |
| 64 | 63.0 GiB | 82.8% | 1.38 |
| 128 | 126.0 GiB | 96.3% | 0.30 |

Against that bound, an online power-of-K LFU replaying the same trace reaches
**58.4% at 31.5 GiB and 74.1% at 85 GiB**; raising the victim sample from 32
to 1024 slots moves it by 0.2 points, so the gap to the oracle is the cost of
*learning* the distribution, not of picking the victim. The build measures
50.2%, and the 8 points between that and 58.4% are prefill's one-shot experts
plus the half of the table that the open-addressed layout leaves empty.

**Measured, not assumed:** the cache size itself is a wash. Same prompt, 40
steps, `CWICK_EXPERT_CACHE_GIB` swept:

| cache | wall | stream | gemv | alloc (fill) | hit | SSD/step |
|---|---|---|---|---|---|---|
| 32 GiB | 727.8 | 390.2 | 170.6 | 36.5 | 50.4% | 3.29 GiB |
| 64 GiB | 774.5 | 413.4 | 191.0 | 88.6 | 56.5% | 2.37 GiB |
| 85 GiB | 786.4 | 424.1 | 197.4 | 107.3 | 57.3% | 2.29 GiB |

Bytes fall 30% and the phase gets *slower*: with fewer reads outstanding the
drive delivers 7.3 GiB/s instead of 9.3, which gives back what the hit rate
bought. The `alloc` column is the one-time cost of filling N slots (4095 for
32 GiB, 10675 for 85 GiB, ~0.52 ms each) and is why a short run flatters the
small cache. Subtracting it, all three land within 13 ms of each other.
(`alloc` is already inside `stream` in the profile — it is reported as
"of stream: slot reservation" — so it must not be added on top of it.)

### What `attn` is made of, and why it will not shrink

The phases add up as `total = stream + gemv + attn + rest`, which is the
invariant to hold when projecting: 390.2 + 170.6 + 132.7 + 34.3 = 727.8 for
the "before" row above. `attn` is 17-18% of a step and is **not** part of the
fp8 expert work, so it does not move when the LUT goes away. Same machine,
12-token prompt, 40 steps, split into its six parts:

| part | ms/step | share | what it is |
|---|---|---|---|
| kda_proj | 57.5 | 43% | q/k/v of the 34 KDA layers, already one bf16 GPU batch |
| mla_all | 32.5 | 24% | the 11 sparse-attention layers, end to end |
| kda_tail | 24.5 | 18% | gated RMSNorm + o_proj, another bf16 GPU batch |
| kda_rec | 9.4 | 7% | the KDA recurrence itself, CPU |
| kda_gate | 4.5 | 3% | f_a/f_b/b, decay and beta |
| kda_conv | 3.3 | 2% | short conv + sigmoid + L2 |

That run measured a cold 46.5% hit rate and 768.5 ms/step, so read the
*shares*, not the absolute milliseconds; scale by 0.86 to compare against the
663 ms row. The six parts sum to 131.6 of the 133.3 ms the profile reports
(the last 1.7 ms is timer overhead, spread over 45 layers). The point is the
composition: `kda_proj` + `kda_tail` are 82 ms of bf16 GEMV through the same
batch the fp8 experts use, and they run at ~125 GB/s with no decode at all —
they are a byte-widening shift, not a table lookup or a bit scramble. So
~110-125 GB/s is what this GPU does with a near-ideal GEMV feed, and `attn`
will not shrink along with anything that happens to the fp8 kernel.

### The arithmetic of a resident step

The cleanest way to see the wall is to ask what a step costs when the SSD is
taken out of it entirely — an expert cache large enough to hold all 290.32 GiB,
which no part of this machine can be. Every expert byte then comes from unified
memory, `stream` goes to zero, and the step is the two GPU phases plus the glue:

    126 (gemv) + 124 (attn) + 33 (rest) = 283 ms/step = 3.5 tok/s

That is the floor with **infinite memory**. So the rate is not a memory-sizing
problem, a cache-policy problem, or a tuning problem: on this GPU, with this
decode, the compute alone sets it, and no amount of RAM changes that.

The one experiment that could have moved that floor was the fp8 LUT, and it was
run and lost — `gemv` stays at 126, not 66. Had it worked, the resident step
would be 223 ms (4.5 tok/s) and the SSD would get 27 ms of the step, i.e.
0.25 GiB/step, i.e. a 97% hit rate = 281 GiB pinned on a 128 GB machine. Both
branches end in the same place — bytes per step set the rate — one of them just
ends closer.

Per decode step the model moves ~18.5 GiB of weights: 8.06 GiB of routed
experts plus ~10.4 GiB of KDA/MLA projections and shared experts, which are
resident by definition. A 250 ms step would mean moving that at 74 GB/s
aggregate, against an SSD at ~9.3 GiB/s and unified memory at ~227 GB/s:

| variant | SSD/step | stream | gemv | attn | rest | total | tok/s |
|---|---|---|---|---|---|---|---|
| before the concurrent encoder (measured) | 3.29 | 390 | 171 | 133 | 34 | 728 | 1.37 |
| today, 32 GiB (measured) | 3.29 | 379 | 126 | 124 | 33 | 663 | 1.51 |
| 85 GiB cache, 57% hit (measured, pre-concurrency build) | 2.29 | 424 | 197 | 132 | 33 | 786 | 1.27 |
| 85 GiB cache, concurrency + a gemv that does not grow | 2.29 | 424 | 126 | 132 | 33 | 715 | 1.40 |
| 88% hit, ~87 GiB pinned (does not fit, projected) | 0.97 | ~150 | 126 | 132 | 33 | 441 | 2.27 |
| every expert resident (impossible; the floor above) | 0 | 0 | 126 | 124 | 33 | 283 | 3.5 |

Row three is the surprise, and it is measured: raising the cache to 85 GiB buys
30% fewer bytes and *loses* wall clock. `stream` goes 379 -> 424 because the
queue gets shallower, and `gemv` goes 171 -> 197 **within that same sweep**, as
85 GiB of pinned expert pages crowd the GPU's working set. Row four gives that
row every benefit of the doubt — concurrency, and a `gemv` that does not grow
at all — and it still only ties today. Enlarging the cache is not a lever here;
it is a memory-pressure experiment.

Row five is the best a cache that does not fit could do: 2.27 tok/s. So the
honest spread is 1.5 today, ~2.3 from a cache that would leave nothing for the
KV cache, the arenas or the OS, and 3.5 if the expert bank were resident — which
is the machine's entire reason for existing as a streaming design.

### What MTP actually buys, measured

The speculative idea is seductive and the arithmetic kills most of it. A
2-token batch runs the MoE over the **union** of the two tokens' routing, not
over 8 experts twice, so the "one weight read, two tokens" framing holds only
if the union is near 8. It is not. Replaying `CWICK_EXPERT_DUMP` over 79 steps
and 3276 (layer, step-pair) samples:

| | experts |
|---|---|
| one token | 8.00 |
| union of two adjacent tokens | **13.03** |
| expert bytes per token, 2-token batch vs 1-token | **0.815x** |

So the second token costs 0.63x of a step's expert bytes instead of 1.0x. It
is a real saving and an 18.5% one, not 50%. That fixes the ceiling:

    step(1 token)  = 663 ms
    step(2 tokens) ~ stream 379*1.63 + gemv 126*1.63 + attn 136 + rest 40
                   ~ 999 ms
    oracle drafter (accept 1.0):  999/2     = 500 ms/token  -> 2.0 tok/s (+33%)
    measured acceptance 0.586:    999/1.586 = 630 ms/token  -> 1.59 tok/s (+5%)

**+33% is the ceiling on this model even with a perfect drafter**, because the
expert union — not the drafter — is what bounds it. Break-even is an
acceptance of 0.507, which is uncomfortably close to what the drafter actually
achieves.

The drafter itself is real. GLM-5.3-Flash ships `num_nextn_predict_layers=1`
as a complete forty-sixth block at `layers.45.*` — ominously complete: its own
288-expert bank, its own MLA attention, its own DSA indexer — plus the three
tensors that make it an MTP head (`enorm`, `hnorm`, `eh_proj`) and
`shared_head.norm`. Driving it is one fusion plus one DSA block:

    h'    = eh_proj([ enorm(emb(t+1)) , hnorm(h_t) ])
    draft = lm_head( shared_head.norm( DSA-block(h') ) )

and scored against the token the model really emits next, on greedy decoding,
30 tokens: **17/29 accepted = 58.6%**. The concatenation order is not
documented anywhere in the checkpoint; `CWICK_MTP_SWAP=1` inverts it and
measures **0/29**, which is what makes 58.6% a result rather than a
coincidence — the block is demonstrably being driven the way it was trained.

Two things are still missing before any of that turns into tokens:

- **The verify pass.** `forward_prefill_mhc` returns only the last position's
  logits and advances `pos` unconditionally. Speculation needs both positions
  (position 1 decides the draft, position 2 is the free token) and a `pos`
  the caller commits. That is a second forward entry point, not a flag.
- **A KDA rewind.** The 34 linear-attention layers fold the sequence into a
  136 MiB DeltaNet accumulator plus a 9 MiB conv window. Neither can be
  unwound — the conv window has already discarded the input it would need —
  so a rejected draft requires snapshotting ~145 MiB before the verify and
  restoring it on a miss. That is ~3 ms against a 999 ms step, so it is
  affordable; it is just not free, and it is the one place where a mistake
  corrupts output silently instead of slowing it down.

Given +5% at the measured acceptance rate and a +33% ceiling that needs a
near-perfect drafter, this item is worth finishing only if the drafter can be
pushed past ~0.75. The two candidates for that are back-filling the MTP
block's KV over the prompt (it currently attends over zeros there — h_t
carries the context, which is why it works at all) and sampling the draft from
the request's own distribution instead of taking its argmax.

The honest order of work from here:

1. **Overlap the expert fetch with the layer's own GEMV.** They are strictly
   serial today (fetch 7.5 ms, then 3.0 ms of GPU). Splitting the fetch into
   gate+up then down and committing wave one before the down reads would hide
   about a third of the read time, ~-55 ms.
2. **Batch a second token onto the same weight read.** Measured, and smaller
   than it sounds — see "What MTP actually buys" below. The drafter works
   (58.6% accepted on greedy), but a 2-token batch costs 1.63x the expert
   bytes, so the realised gain is ~5% and the ceiling even with an oracle
   drafter is +33%. The verify loop is not implemented; the draft and its
   scoring are (`CWICK_MTP=1`).

There used to be a third item here — "fp8 decode without the LUT gather",
priced at ~-60 ms off a v3d measurement that ran the same wave with no decode
at all. It was implemented as `v4` and it loses; that section is the record.
The lesson worth keeping: a diagnostic twin that removes work entirely gives
an *upper bound*, and an upper bound is not a plan.

Neither item prunes an expert, skips a layer, or narrows a weight: the router
still picks 8 of 288 and all three matrices of each are applied.

## 2026-10-26: the prefix cache hit 0% and the reason was one token

A live `serve` session was audited from its access log and `/v1/metrics`:
four chat requests, one continuous conversation, and

```
vllm:prefix_cache_queries_total       4
vllm:prefix_cache_hits_total          0     → 0.0%
cwick:prefix_tokens_reused_total      0     → 0 of 751 prompt tokens
```

Not one `prefix cache hit` line in the log, and not one `load failed` either —
the lookup itself was returning 0. The cache was healthy on every other axis
(`pfxcache/` held 8 snapshots / 149 MiB against an 80 GiB budget, so nothing
was evicted), and the prefill it was failing to skip cost 260 s of the
session's 778 s.

### The off-by-one

Turn N's prompt ends with `履衅`: the server appends it so the model
starts reasoning. Turn N+1 of the same session does not resend it — the client
renders the assistant's turn as `履畔` followed by its content, so the two
prompts diverge at exactly this last token.

Measured by extracting both rendered prompts from the trace and comparing them
byte by byte: turn N was 674 characters / 262 tokens, turn N+1 was 973 / 405,
and they agreed on the first 667 characters. The divergence lands *inside* the
`履衅` token. So the prefix two consecutive turns actually share is

```
turn N's prompt            = 262 tokens
what they really share      = 261 tokens
snapshots the cache held   = {262, 551}
```

The lookup is exact-length (`pfx.c:383` compares `e->pos` against the sampled
position), so an entry one token too long is as useless as no entry at all.
The same thing happened on every other transition: 33→51 needed 32 and held
nothing; 51→262 needed 50 and held 704.

Two more things kept a shorter entry from existing:

* `CWICK_PFX_MIN_TOKENS` was 256, so the first turns of a session were never
  written at all.
* The mid-prefill checkpoints only fire once the prompt is chunked, i.e.
  `prompt_len >= 2 * pfx_block` (8192 tokens at the default 4096). Every
  prompt in this session was under 500, so there were no intermediate entries
  to fall back on.

### The fix

Capture a twin snapshot one token shorter, keyed on the same tokens minus the
last, alongside the full-length one. `cwick_engine_capture_prefix` already
accepts any position `<= eng->pos`, so this is one extra state pack (a
~11 MiB copy here) rather than a second prefill — and it is the same one-token
rollback the whole-prompt hit path already applies on load. `CWICK_PFX_MIN_TOKENS`
drops to 32: a 33-token turn costs 19 s of prefill on this checkpoint (the
expert-union read dominates, so the cost barely depends on token count) and its
snapshot is under a megabyte.

### A/B, same requests both arms

Three turns, `reasoning_effort=High`, `max_tokens=48`, history carrying a fixed
assistant turn so both arms issue byte-identical bodies:

| turn | prompt | replayed | prefill, cache off | prefill, cache on |
|---|---|---|---|---|
| 1 | 33 | — | 19.083 s | 19.354 s (cold) |
| 2 | 42 | 32 | 15.210 s | **4.407 s** |
| 3 | 50 | 41 | 19.279 s | **3.522 s** |
| | | | 53.572 s | **27.283 s** |

Hit rate 2/3 = 66.7%, 73 of 125 prompt tokens reused. Snapshot load is ~0.1 s
(the hit lands 102 ms and 117 ms after the request arrives, then the remaining
9–10 tokens are prefilled — the residual floor is what is left, not the cache).

Correctness: the three responses are byte-identical between the arms apart from
the `id`/`created` timestamps and the `usage.prompt_tokens_details.cached_tokens`
field the hit now reports. `content`, `reasoning_content` and `finish_reason`
all match, which is the real test — a replayed state that was one position off
would show up immediately as different text. `make test` PASS and the layer
oracle is 204 checks / 0 failure / worst rel 0.

The remaining miss is turn 1, which is correct: there is nothing to replay yet.

## 2026-10-26 (later): the thinking budget was never enforced on DeepSeek

An audit of the same `serve` session (log + `/v1/metrics`) found three problems.
One of them was burning the machine: a request had generated **16,568 thinking
tokens against a configured ceiling of 8192**, had been running 2 h 35 m, and
still had ~30 h to go at its degraded rate. This section is that one.

### What was wrong

`server.c` arms the ceiling unconditionally for any chat request whose
tokenizer gives both markers as single tokens, and logs it:

```
[req 179] thinking budget 8192 tokens (level=高, block already open)
```

The enforcement lives in `engine.c`'s decode loop (`eng->think_budget` is read
at 3708, the close tag is written at 3775-3822). DeepSeek does not use that
loop: `ds_engine.c` runs its own, and a search for `think` in that file returns
two comments about the *progress label* and nothing else. So the budget was set
on the engine, reported as armed, and never read by the code that decodes.

The loop guard did not save it either -- `finished_reason=loop` stayed 0 for the
whole run, because the model was not in a repetition attractor. It was genuinely
still thinking, which is the one failure mode a loop detector cannot see.

### The fix

Push the ceiling into the backend that has its own loop, the same way
`ctx.stop`, `ctx.tag`, `ctx.spec` and `decode_phase` already are:

- `ds_engine.h`: `ctx` gains `think_budget` / `think_open_id` / `think_close_id`
  / `think_open_at_start`.
- `engine.c`'s `cwick_engine_set_think_budget` copies all four into
  `eng->ds->ctx`, including on the disarm path.
- `ds_engine.c`'s decode loop gets the same L4 block the GLM loop has: clamp to
  `max_new - CWICK_THINK_RESERVE`, ask the pure `cwick_think_plan`, and on a
  non-NONE action write the marker(s) into the output *and* feed them through
  `ds_model_forward` in the same step, so the model that just wrote its own
  delimiter answers instead of continuing the thought. `think_open` /
  `think_closed` latch exactly as on the GLM side, and the result now reports
  `think_budget_hit` / `think_budget_applied`.

### A bug the first cut introduced, and the control run that caught it

`eng->ds` comes from `cwick_xmalloc` with **no memset**, so `ctx.think_budget`
was uninitialised. A control run with `CWICK_THINK_BUDGET` unset printed
`thinking budget: 0 tokens reached` -- a zeroed struct read as "close the think
block before the first token", which is precisely the trap `cwick_engine_new`'s
own comment warns about ("0 is a legal budget, so the disabled state has to be
an explicit -1"). `ds_engine_open` now initialises all four fields to the
disabled state. Worth noting the control run is what found it: the armed case
had already looked correct.

### Verification, real checkpoint, `--backend metal`

| case | expected | observed |
|---|---|---|
| no `CWICK_THINK_BUDGET` | silent | no budget line at all; `gen=60 finish=length` (the old behaviour) |
| `CWICK_THINK_BUDGET=32` | fire at 32 | `thinking budget: 32 tokens reached`, then `gen=60 finish=stop` |
| `CWICK_THINK_BUDGET=0` | close before token 1 | `0 tokens reached`, `gen=40 finish=length` |
| `CWICK_THINK_BUDGET=200`, `--max-new 40` | clamp to 20 | `200 tokens reached` is not what appears; `20 tokens reached` is |
| server, `reasoning_effort=High`, `max_tokens=48` | clamp 8192 to 24, fire, report | `8192 clamped to 24: 48 tokens of output leave no room to answer`, then `think=25 content=23 think_cap=24(hit)` |

`make test` PASS (the existing `test_think_budget` covers the GLM path; the DS
path has no synthetic fixture, so it is covered by the runs above, which is what
`cli_arm_think_budget` exists for -- "a second server would need another copy of
the weights").

The operator-visible payoff is that `think_cap=` is now a real number. Before,
`res.think_budget_applied` was 0 for every DeepSeek request, so the access log
printed `think_cap=0` -- indistinguishable from "no thinking requested" -- while
the request was actually running under an unenforced 8192.

## 2026-11-07: the two endpoints, end to end, one model per process

Two ways to use the same resident weights, measured back to back: a chat turn
(`POST /v1/chat/completions`) and a decision (`POST /v1/systemone`). The point is
to see where the wall clock goes in each, not to crown a model.

**Method.** Every checkpoint was started in its **own fresh process** -- no
`POST /v1/models/switch` was called anywhere in this table, so nothing here is
contaminated by an outgoing model's mappings or cache. Prefix cache off (so every
prefill is cold and comparable), `--backend metal`, `--context 32768` for all
three (the same KV budget; `start.sh`'s 500K default would give GLM and DS a
different memory plan), greedy sampling, one request each. Chat prompt is the
same paragraph used as the decision `state` plus "Summarize the paragraph."; the
decision is that state with one 8-option choice. Times are the server's own
histograms (`vllm:time_to_first_token_seconds`, `request_prefill_time_seconds`,
`request_decode_time_seconds`) read from `/metrics` around the request, so they
are the numbers an operator sees.

### Chat: prefill plus one step per output token

| checkpoint | load | prompt tok | TTFT | prefill | out | decode | decode | e2e | footprint |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniMax-M3.1-Flash | 18.3 s | 261 | 55.90 s | 4.67 tok/s | 48 | 31.41 s | 1.53 tok/s | 87.31 s | 22.13 GiB |
| DeepSeek-V4.1-Flash-1017 | 16.3 s | 108 | 50.31 s | 2.15 tok/s | 32 | 14.58 s | 2.19 tok/s | 64.89 s | 84.16 GiB |
| GLM-5.3-flash | 22.4 s | 93 | 56.09 s | 1.66 tok/s | 48 | 38.28 s | 1.25 tok/s | 94.38 s | 49.98 GiB |

### Decision: one prefill, no decode

| checkpoint | input tok | wall | prefill | footprint | answer |
|---|---:|---:|---:|---:|---|
| MiniMax-M3.1-Flash | 427 | 80.91 s | 5.28 tok/s | 22.13 GiB | replace-white 0.996 |
| DeepSeek-V4.1-Flash-1017 | 249 | 84.46 s | 2.95 tok/s | 84.16 GiB | replace-white 0.998 |
| GLM-5.3-flash | 259 | 106.59 s | 2.43 tok/s | 16.08 GiB | replace-white 0.999 |

**Load** is process start to the first answered `GET /v1/models`: engine map,
tokenizer, catalog. It is not part of either request.

### What the two tables say

**A decision costs exactly one prefill.** Its wall clock equals the chat turn's
TTFT for the same prompt, and it adds no second term: GLM 106.59 s is prefill of
259 tokens at 2.43 tok/s, not "generation that failed to stop". That is also why
the answer is identical across the three checkpoints (`replace-white`, 0.996 /
0.998 / 0.999): the readout does not depend on how fast the machine is.

**The same conversation is not the same number of prompt tokens.** The same two
messages encode to 93 tokens under GLM, 108 under DeepSeek and 261 under MiniMax:
the MiniMax template carries a system scaffold the other two do not. A decision
on the same state inherits that: 259 / 249 / 427 input tokens. MiniMax still
finishes first because its prefill rate is roughly twice the other two -- it pays
more tokens at a better rate.

**Prefill rate is the family constant, not the request's.** 4.67-5.28 tok/s
(MiniMax), 2.15-2.95 (DeepSeek), 1.67-2.43 (GLM). The spread inside a family is
n=1 noise plus prompt length; the spread between families is the memory path.

**Decode is the only term that differs in kind.** 0.80 s per token on GLM, 0.65
on MiniMax, 0.46 on DeepSeek -- all far above the 275 ms/664 ms steps recorded
elsewhere in this file for short context, because a cold expert set is being read
on every step and the pin cache cannot hold the 7.88 GiB a step touches. The
first version of this measurement could not finish GLM at all; that was a leak,
not a step cost, and it is the next section.

### The leak this measurement found, and the fix

The first attempt at the GLM row was killed by jetsam three times -- 32 GiB cache
(metal), 8 GiB cache (metal), 8 GiB cache (CPU) -- always in the same way:

| decoded tokens | footprint before | RSS before | footprint after |
|---:|---:|---:|---:|
| 8 | 67.17 GiB | 50.14 GiB | 49.94 GiB |
| 16 | 117.39 GiB | 38.45 GiB | 49.95 GiB |
| 24 | 179.37 GiB | 34.91 GiB | 49.96 GiB |
| 48 | killed | -- | 49.99 GiB |

About **+6 GiB of footprint per decoded token while RSS fell**: the growth was
compressed/page-cache accounting, and the increment matched the 7.88 GiB of
experts a decode step reads. The cause was in `moe.c`, not in the stream: the
per-token MoE path allocates a `cwick_fetch_req` array for the route-first fetch
and **never called `layer_fetch_release` on it**. Every matrix the pin cache
refused is a buffer the fetch owns (`prefill_blob_alloc`), and those were simply
dropped -- one layer's staging bytes per layer, per step, forever. With eight
experts x three matrices x 45 layers that is exactly one step's worth of bytes
retained per step.

The batched paths (`moe.c` 1745 / 1969) always released; only the single-token
path added later was missing it. The fix releases the batch and frees the array
after the group GEMV returns (both Metal waves wait before that point), and the
error path frees the buffers it owns when a read fails mid-batch. `make test`
still passes.

Two consequences worth keeping:

- **GLM is usable for chat now.** 48 tokens at 1.25 tok/s, footprint flat at
  49.98 GiB with the default 32 GiB pin cache (25.94 GiB with an 8 GiB cache --
  the same speed, so the cache size is not what limits this). The answer text
  stays coherent; only the memory changed. A repeat of the same request gave
  54.06 s TTFT / 92.15 s end to end and **byte-identical output**, so the fix did
  not perturb the arithmetic.
- **A short test would not have found it.** Eight tokens already cost 67 GiB but
  the run still completed; the kill lands at 24-32 tokens. Anything that measures
  decode on this checkpoint needs at least a 32-token generation, and should read
  `cwick:process_footprint_bytes`, not RSS -- RSS *fell* while the process was
  being killed.

The logs for every run in this section are `/tmp/cwickbench-<label>/serve.log`;
the driver is a throwaway shell script, so a re-run costs one process per model
and no model switch.

### The decision path, same regression: nine prefills, no retention

The leak above was in the decode loop, and `/v1/systemone` never enters it. That
is an argument, not a measurement, so the same long-run test was run against the
decision endpoint: several requests in a row, each carrying more than one
question, footprint sampled before and after every request.

**Method.** One process per checkpoint (no model switch), `--backend metal`,
`--context 32768`, prefix cache capped at 1 GiB, state and question set changed
per round so a later round cannot be answered out of the previous round's cache.
Each question inside a request is its own prefill by design, so a request with
three questions is three prefills.

**GLM-5.3-flash** (`/tmp/jevlong-glm-5.3-flash/serve.log`):

| round | prompt tok/question | questions | input tok | wall | footprint before | after |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 251 | 3 | 753 | 364.5 s | 15.46 GiB | 16.15 GiB |
| 2 | 297 | 3 | 891 | 433.1 s | 16.15 GiB | 16.14 GiB |
| 3 | 468 | 2 | 937 | 376.7 s | 16.14 GiB | 16.20 GiB |

**MiniMax-M3.1-Flash** (`/tmp/jevlong-minimax/serve.log`; the first row's
footprint before is 0.12 GiB because the model maps lazily and the sample was
taken before the first request touched it):

| round | prompt tok/question | questions | input tok | wall | footprint before | after |
|---|---:|---:|---:|---:|---:|---:|
| short | 408 | 2 | 817 | 165.7 s | 0.12 GiB | 22.11 GiB |
| long (crosses two 512-token chunks) | 829 | 2 | 1659 | 327.5 s | 22.11 GiB | 22.22 GiB |
| short again | 408 | 2 | 817 | 153.0 s | 22.22 GiB | 22.23 GiB |

**What it shows.**

- **The decision path retains nothing.** After the first request (which maps the
  weights and allocates the pools once), GLM moves +0.06 GiB over 1,828 input
  tokens and MiniMax +0.12 GiB over 2,476. The leak fixed above was +6 GiB per
  *step*; this is two to three orders of magnitude smaller and does not grow with
  the number of requests -- round 3 is below round 2 on GLM.
- **Every question is a full prefill.** `input_tokens` is exactly questions x
  prompt (753 = 3 x 251, 1659 = 2 x 829). A request does not reuse the state's KV
  across its own questions, so a three-question decision costs three prefills,
  not one prefill plus three readouts. That is the standing limitation noted
  when the endpoint shipped; this run is the measurement of it.
- **Readouts are deterministic.** The same GLM request run twice in two separate
  processes returned `replace` 0.9989, `escalate` 0.6459, `replace` 0.9977 both
  times -- identical to the fourth decimal. MiniMax's repeated short round picked
  the same two answers. So a changed decision means a changed model or a changed
  prompt, never a changed run.
- **The pin cache does nothing for a prefill.** GLM's log lines during these
  requests read `expert cache 0.00 GiB pinned, 686.53 GiB streamed`: 251 tokens
  touch essentially the whole expert set, so there is nothing to keep between
  steps and the cost is bounded by the device read rate. That is also why the
  footprint is flat -- the decision path has no cache that could grow.

Prefill rate in this run: GLM 2.07-2.48 tok/s, MiniMax 4.93-5.34 tok/s, the same
family constants the tables above show. Nothing in this test changed a byte of
the engine; it is a regression guard, and it passes. The guard is checked in as
`tools/decision-longrun.sh` (`sh tools/decision-longrun.sh <label> <root>
[rounds] [questions]`), so the next change to prefill can re-run it instead of
re-deriving the method.

### Decision prefill: one block, one shared prefix, exact only where it is

Measured on the three resident checkpoints, `--backend metal`, context 4096,
one request per row, wall clock around `POST /v1/systemone`. "Before" is the
build at the start of this change (512-token blocks, one full prefill per
question); "after" is 1024-token blocks plus in-request prefix reuse.

| request | backend | input tok | prefill tok | before | after | speedup |
|---|---|---:|---:|---:|---:|---:|
| one question, kart-sized prompt | MiniMax | 550 | 550 | 109.3 s | 95.5 s | **1.14x** |
| three questions, 348-token shared state | MiniMax | 1145 | 449 | 213.8 s | 102.0 s | **2.10x** |
| three questions, 172-token shared state | DeepSeek | 605 | 605 | 240.1 s | 237.5 s | 1.01x |
| three questions, 173-token shared state | GLM | 611 | 611 | 306.1 s | 306.1 s | 1.00x |
| one question, same body as the kart | DeepSeek | 370 | 370 | — | 143.3 s | — |

DeepSeek and GLM keep every token because their reuse is not exact (see
`DECISIONS.md` §44): forcing it changed answers (`|dp|` 0.56 and 0.41), and a
snapshot round-trip on GLM left `|dp| = 0.11`. Their single-question cost is
"read the model once" — GLM streams 686.53 GiB for 251 tokens — so token count
barely moves them either.

**The even-split negative result**, worth keeping because it looks like it
should win: 550 tokens as 275+275 measured 113.6 s against 109.3 s for
512+38. The expert read is a per-block cost, so halving the big block costs
more than the small tail saves. Raising the ceiling (512 -> 1024) is the lever
that works; chopping evenly is not.

Memory cost of the larger block on MiniMax: footprint 23.8 GB, swap
1.36/3.0 GiB, no growth over the run.
