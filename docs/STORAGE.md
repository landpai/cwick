# Native storage preparation

<p align="center"><a href="STORAGE.md">English</a> · <a href="STORAGE.zh-CN.md">简体中文</a></p>

CWICK memory-maps a BF16 resident spine (the source of truth) and reads routed
experts per layer.

## Two checkpoint layouts, one reader

The loader accepts either shape under a model root, and decides at open time:

```
model.safetensors.index.json + model-*.safetensors   the released checkpoint
resident/tensors/<name>.bin                          a converted `resident/` tree
```

The three shipped checkpoints are read **as published** — no conversion step. The
`resident/` tree is what `cwick setup` writes for the synthetic verification models and
what a converted corpus looks like; it is also the only shape `--spine int8` applies to
(the flag is ignored, with a log line, for a safetensors checkpoint).

Downloads from the console or `/v1/hub/download` land in `./models/<owner>/<name>/`, which
is exactly such a root: `config.json` plus shards is all the loader needs.

## Directory layout of a `resident/` tree

```
config.json
resident/
  tensors/<name>.bin        BF16 dense spine, row-major [out, in]
  experts/expert-L-E.bin    one canonical raw file per layer/expert
resident-int8/
  tensors/<name>.i8         optional row-int8 data
  tensors/<name>.s          optional fp16 per-row scales
tokenizer/
  vocab.json                optional BPE vocabulary (token -> id)
  merges.txt                optional BPE merges
```

The BF16 tensors are raw little-endian bfloat16 rows in torch linear layout
(`weight[k][j]`, output-major). A norm vector `[n]` is stored as a single row.
The optional int8 representation stores `int8` rows plus one fp16 scale per row.

## Expert file format

Each routed expert is one file: a 20-byte header
(`"CWICKEXP1"` + u32 layer + u32 expert + u32 matrix count), then three BF16
matrices in linear layout: `w1 [I, RH]`, `w2 [RH, I]`, `w3 [I, RH]`, where
`I` = `moe_intermediate_size` and `RH` = `routed_expert_hidden_size`.

The complete expert corpus must be present: `cwick_spine_open` fails closed if
any layer/expert file is missing (never-pruned rule).

## Representations

- `bf16` (default) — the original BF16 weights, weight-exact.
- `int8` — row-int8 + fp16 scales; selected with `--spine int8`. The BF16 spine
  remains the verification authority.

## Setup

`cwick setup` writes a deterministic tiny model in this layout (layer 0: KDA +
dense MLP; layer 1: MLA + MoE), so the whole engine and server can be verified
without downloading real weights. The real checkpoints need no such conversion — they are
read from their safetensors shards in place; `model.c` enforces the full no-pruning
contract before any weight payload is opened.

For the released HF checkpoint no conversion is needed: `cwick` reads the
`model-*.safetensors` shards directly through the split below.

## Memory-bounded streaming (`src/stream.c`)

The released checkpoint (306 GiB / 62 shards) does not fit in RAM, so storage
is split by how often a tensor is touched, not by shard boundaries:

- **Resident** — everything read on every token: norms, KDA/DSA attention,
  shared experts, embeddings, `lm_head`. ~15.46 GiB for GLM-5.3-Flash. Kept in
  the shared `mmap`, warm-wired with `madvise(WILLNEED)` at boot, and checked
  against host memory in `cwick_engine_new`.
- **Streamed** — the routed expert bank (`*.mlp.experts.*`), 290.32 GiB. Read
  with `pread()` on descriptors opened `F_NOCACHE`, into a fixed 32 MiB arena.
  Nothing is retained: the kernel never builds a page-cache copy of the expert
  bank, and RSS is bounded by construction.

Every tensor carries `shard_index` and `data_offset` alongside its mapped
pointer, so the same descriptor table serves both paths.

## How much RAM is used: the load-time plan (`src/mem.c`)

Residency is decided once, from the machine the process is running on, and the
decision is a **hard ceiling**, not an aspiration:

```
target      = host RAM x --mem-fraction            (default 0.80)
- resident  = the spine every token touches        (~15.46 GiB)
- state     = KV / KDA / mHC buffers               (~2.94 GiB)
- arena     = staging + prefetch arenas            (~0.03 GiB)
- headroom  = two routed layers of demand reads     (~0.38 GiB)
= expert cache budget                              (~83.60 GiB on 128 GiB)
```

Measured on this machine (128 GiB, GLM-5.3-Flash):

```
cwick: [info] [mem] memory budget: 128.00 GiB host, using at most 80% (102.40 GiB)
cwick: [verb] [mem] budget split: resident spine 15.46 GiB + runtime state 2.94 GiB + arenas 0.03 GiB + read headroom 0.38 GiB
cwick: [verb] [mem] expert cache budget 83.60 GiB (routed expert bank is 290.32 GiB, streamed per decode step)
cwick: [info] [gen] prefill 24/147 — rss 31.71 GiB of 102.40 GiB ceiling, expert cache 82.12 GiB pinned, 82.77 GiB streamed
```

The plan: `recommendedMaxWorkingSetSize * pct`, minus the non-routed weights,
minus an overlap reserve, minus the runtime state.
Two properties earn their keep here:

- the inference state is booked **before** the spine opens, so the expert
  cache is sized against real remaining RAM (the old code computed the
  resident split *after* allocating the cache and therefore never subtracted
  it — on 128 GiB it reserved 96 GiB of cache on top of a 15.5 GiB spine);
- the ceiling is **enforced while running**. Every 8 decode steps the engine
  compares `task_info()` RSS with the plan and, if it drifted over, releases
  pinned expert bytes (`cwick_stream_trim`) and lowers the cache cap. The
  expert cache is the only elastic thing in the process; everything else is
  fixed at load time.

## Expert pin set

A byte-budgeted, heat-ranked pin set over raw expert blobs
(`cwick_stream_options.expert_cache_bytes`, on by default with whatever the plan
leaves). Insertion probes a short window in an open-addressed table sized to
the budget, and eviction drops the coldest entry in that window — the previous
direct-mapped 1024-slot table could not hold ~36k blobs, so it silently
degraded to "whatever collided last" once the budget grew past a few GiB.

Heat model: a hit adds a quantum, everything decays geometrically every 64
steps, and the coldest entry is what goes when byte pressure arrives. Pinning
is an optimization: a miss costs one SSD read, never a wrong token.

Override the share of RAM with `--mem-fraction PCT`, cap it absolutely with
`--max-memory GiB`, or set `CWICK_MEM_PCT` / `CWICK_MEM_MAX_GIB`.

`cwick doctor` uses this path to verify the checkpoint by actually reading it:
resident tensors in full, experts by strided sample (`--deep` for all
306 GiB), with a running FNV-1a checksum. Measured peak RSS for a full
resident pass plus a 4.76 GiB expert sample: **72.9 MB**.
