# DeepSeek-V4.1-Flash-1017 port specification

<p align="center"><a href="DEEPSEEK-PORT.md">English</a> · <a href="DEEPSEEK-PORT.zh-CN.md">简体中文</a></p>

> Goal: let cwick run `deepseek-ai/DeepSeek-V4.1-Flash-1017` (item 2 of the "currently adapted model list" in goal.md).
> This file records **facts such as identifiers, purposes and performance**; it is open to discussion.
> All architecture details come from the `inference/` reference implementation shipped inside the model checkpoint, not from any external project.
> Items marked ⚠️ are places where the reference implementation itself is ambiguous and where the spec can only be finalised after numerical verification.

---

## 0. Conclusions first

| Item | Conclusion |
|---|---|
| Can it be supported | Yes, but it requires **a whole new set of forward operators (dequantisation / attention / MoE / engram)**, not a relaxation of the validation on `model_type` |
| Workload | Roughly 2500–3000 lines of C: FP4 dequantisation GEMM, activation quantisation, Hyper-Connections Sinkhorn, engram hash lookup, indexer second-level top-k, block-diagonal wo_a |
| Early judgement on the decode rate | **Judged physically unattainable at the time**, on the basis of "every token must read ~3.2 GB from disk ÷ local SSD 5.4 GiB/s ≈ 1.6 tok/s". **That conclusion was overturned by measurement on 2026-10-25**: with an 82 GiB expert pin cache and an 86% hit rate in steady state, the SSD carries only about 0.6 GB/token, and decode measures **2.3–2.7 tok/s**. See §10.10 |
| Realistic ceiling | Bare decode measures 2.3–2.7 tok/s (119 steps server-side, `--backend metal`); going past that still requires speculative decoding (the checkpoint ships 3 nextn layers) |
| Suggested order | M0 numerical oracle → M1 dequantisation → M2 single-operator validation → M3 attention → M4 MoE → M5 engram → M6 full pipeline |

---

## 1. Current state: why the engine rejects it

```
$ ./cwick serve --root deepseek-ai/DeepSeek-V4.1-Flash-1017 --port 18001 --backend metal
cwick: 2026-10-23 02:32:22.503 [info +0.00s] [init] startup failed: config is missing/invalid dtype
```

The failure point is the contract validation in `src/model.c`; two hard mismatches:

1. `quantization_config` only has `quant_method: fp8`, with no `fmt: e4m3` field;
2. `text_config` has no `kv_lora_rank` — because this model **does not use MLA's KV low-rank compression**, and head_dim=512 is full-rank.

Today's `src/model.c` assumes "every model = GLM": on -eck- that is correct (better to refuse than to emit wrong output), but it means DeepSeek support is **a new forward pass**, not a relaxation of validation.

The key differences between the two models:

| | GLM-5.3-Flash | DeepSeek-V4.1-Flash-1017 |
|---|---|---|
| Expert dtype | FP8 e4m3 | **FP4 e2m1 (packed into int8)** |
| Number of experts / topk | 160 / ? | 384 / **6** + 1 shared |
| Attention | KDA | MLA variant + **compressed KV + indexer top-k** |
| Residual structure | plain `x+f(x)` | **Hyper-Connections (Sinkhorn doubly stochastic matrix)** |
| Extra modules | — | **engram (203 GB lookup table)**, MTP/speculative head, vision tower |
| Expert bytes per token | ~1.9 GiB | **~5.93 GB** |

---

## 2. Weight inventory (measured from the 48 shard headers, not guessed)

Total on disk **510.29 GB**, broken down:

| Category | Size |
|---|---|
| experts (40 layers × 384 + shared) | 297.52 GB |
| **engram embed (2 tables)** | **202.76 GB** |
| attention / norm / embed / head / hc_* | 8.58 GB |
| vision | 0.82 GB |
| mtp | 0.61 GB |

dtype distribution:

| Category | dtype | Representative tensors |
|---|---|---|
| FP8_E4M3 + E8M0 **32×32 block** scale | attention `wq_a/wq_b/wkv/wo_b`, `ffn.shared_experts.*`, `indexer.wq_b`, `engram.wkv`, `mtp.main_proj` | `wkv.weight [512,5120]` / `wkv.scale [16,160]` |
| **FP4 packed int8 + E8M0 1×32 row scale** | only `ffn.experts.{i}.w1/w2/w3` | `w1.weight I8 [2304,2560]` / `w1.scale [2304,160]` |
| FP8_E4M3 + E8M0 **1×32 row-block** scale | `engram.embed` | `embed.weight [384006168,256]` / `embed.scale [384006168,8]` |
| Raw BF16 (no scale) | all norms, `ffn.gate.weight`, `embed`/`head`, `compressor.*`, `indexer.wk/weights_proj`, vision/aligner | |
| F32 | `hc_*_{base,fn,scale}` (6 per layer), `attn_sink`, `ffn.gate.bias`/`bias_vl` | `attn_sink [64]` |

> Note where the 160 comes from: FP4 uses MXFP4's **1×32 per-row block** (one scale per 32 elements of each row), FP8 uses **32×32 two-dimensional blocks**. The two cannot be mixed.

---

## 3. forward Overview

```
input_ids [B,L]
  → embed [B,L,5120]
  → repeat → [B,L,4,5120]           # hc_mult=4, the residual stream is 4 copies
  → layer 0..39:
      engram write (only before layer 1, 14)        # additive, bypasses hc_pre/hc_post
      hc_mixes(x) → pre/post/comb              # RMS-scaled Linear + Sinkhorn
      hc_pre(x, pre_mix) → attn_norm → attention → hc_post
      hc_mixes(x) → hc_pre → ffn_norm → MoE → hc_post
  → hc_pre (with the last layer's ffn_pre) → norm → head → logits [B,129280]
```

**There is no `x + f(x)`**: the residual is mixed by a Sinkhorn doubly stochastic matrix, see § 4.1.

---

## 4. Operators that must be added (cwick currently has none of them)

### 4.1 Hyper-Connections (6 small tensors per layer, yet on the hot path)

Six tensors per layer: `hc_attn_{base[24], fn[24,20480], scale[3]}`, and likewise for `hc_ffn_*`.

```c
// ffn/base/scale produce 24 numbers: an RMS-scaled Linear over the flattened 4×5120
// ⚠️ this uses norm_eps=1e-20, not hc_eps=1e-6
x = flatten(h);                       // [4*5120]
rsqrt = 1/sqrt(mean(x^2) + 1e-20);
mixes = fn @ x * rsqrt;               // [24] fp32 GEMM
pre[j]  = sigmoid(mixes[j]      * scale[0] + base[j])      + 1e-6
post[j] = 2*sigmoid(mixes[4+j]  * scale[1] + base[4+j])
C[j][k] = mixes[8+j*4+k] * scale[2] + base[8+j*4+k]
// row softmax(+eps) → column normalise → 19 more rounds (row normalise ÷(s+eps), column normalise ÷(s+eps))
```

Read literally from the code, the mixing formula is **`y = Cᵀ · residual + post ⊙ x_out`** (transposed multiply).
⚠️ **Ambiguity**: the reference implementation's docstring does not say `C` or `Cᵀ`. It must be finalised after numerical verification — Sinkhorn only guarantees that every row and every column sums to 1, and `C` and `Cᵀ` are numerically different.

### 4.2 FP4 dequantisation + mixed-precision GEMM

```c
static const float FP4_E2M1[16] = {0,0.5,1,1.5,2,3,4,6, -0,-0.5,-1,-1.5,-2,-3,-4,-6};
// within one int8 byte: **the low 4 bits are the even index**, the high 4 bits are the odd index
fp4_get(row, k) = FP4_E2M1[(k & 1) ? (byte >> 4) : (byte & 0x0F)];
e8m0_to_f32(b)  = 2^(b - 127);        // b=0xFF is NaN

// out[m][n] = Σ_kb sa[m][kb]*sb[n][kb] * Σ_{k∈kb} fp4_get(n,k) * e4m3(a[m][k])
// FB: one pair of sa/sb per 32 K elements
```

This is the dominant per-token cost (5.93 GB), and also the part the Metal kernels must be rewritten for — the existing GEMV kernels assume an FP8 + scale layout.

### 4.3 Activation quantisation (ue8m0 + round-to-power-of-2)

```c
amax = max(|x| in 32-block);  amax = max(amax, 1e-4);
e = ceil_log2(amax / 448.0f);           // must be implemented bit-exactly; the fast approximative version will not do
s = 2^e;                                 // exact power of two
scale_byte = e + 127;
q = clamp(x / s, -448, 448) → e4m3
```

In the reference implementation `act_quant` is mostly `inplace`: quantise and immediately dequantise back (`x ← e4m3(x/s)*s`), and the result is still bf16. This is **lossy** and must be matched operator by operator, otherwise the accumulated error will make the output drift.

### 4.4 Attention (substantially different from what exists)

Shapes step by step (S=5120, hidden; 64 heads, head_dim=512, rope applies only to the last 64 dims):

```
qr = wq_a(x)  [S→1280] → q_norm(1280, eps=1e-20) → wq_b → [32768] → view(64,512)
rope(q[..., -64:])
kv = wkv(x)   [S→512]  → kv_norm(512, 1e-20)      # num_key_value_heads=1, each position has only one 512-dim entry
kv_cat = concat(window_kv[128], compress_kv[n_comp])   # <-- see below
o = sparse_attn(q, kv_cat, sink, topk_idxs, scale=1/sqrt(512))   → [64,512]
rope^-1(o[..., -64:])
o = view(8, 4096) → act_quant → einsum("bsgd,grd->bsgr", o, wo_a.view(8,1024,4096)) → [8,1024]
x = wo_b(flatten → 8192) → [5120]
```

Key points:

- **`wo_a` is block-diagonal** (8 groups × [4096→1024]), not a plain Linear; and the reference implementation dequantises it to BF16.
- **`attn_sink` is an (n+1)-th virtual KV slot whose value is always 0**, entering only the softmax denominator: `Σ = Σ_j exp(s_j-m) + exp(sink_h - m)`. It is not a logit bias.
- **The number of KVs per query is fixed at 640** (128 window + 512 top-k), independent of context length → attention cost is O(640) and does not grow as the context lengthens. This is good news.

Compressed KV and the indexer (only `compress_ratios` in the config has 40 entries; the 3 extra of the 43 in `inference/config.json` belong to MTP):

```
layer 0,1    ratio 0  —— does not participate in compressed KV at all
layer 2..19  ratio 2  —— KV count = T/2
layer 20..39 ratio 1  —— = T (one 512-dim latent per token)
```

Only `kv_source_layer_ids=[2,8,14,20]` **generate** compressed KV; the other layers read the most recently written one:
2..7 use layer2, 8..13 use layer8, 14..19 use layer14, 20..39 use layer20.

Only `index_source_layer_ids=[2,8,14,20,24,28,32,36]` **re-run top-k**; the other layers reuse the result.
Two-level selection: layer 20 first takes the max over each block of 8 positions to select ≤2048 blocks (covering ≤16384 positions),
and layers 21..39 set candidates outside that to −inf and then take the **top-512 compressed positions** (not blocks).

Each layer has two RoPE configurations (**they differ per layer, easy to trip over**):

- layer 0, 1 (ratio=0): theta=10000, **YaRN off**
- layer 2..39: theta=**160000**, **YaRN on** (factor=16, original=65536)
  - the corrected dim boundaries are computed from `64*ln(65536/(rot*2π))/(2*ln(160000))`: low=15, high=25
  - `ramp = clamp((i-15)/10, 0, 1)`, `freqs *= 1 - 0.9375*ramp`
  - i≤15 unchanged, i≥25 frequency ÷16, linear ramp in between

The compressed latent takes its position from the first token of its group (prefill steps by ratio, decode takes `start_pos+1-ratio`).

### 4.5 MoE routing (different from common implementations)

```
logits = x_fp32 @ gate.weight_fp32^T / 1.0        # gate is BF16 on disk, the code promotes it to fp32; dense is not quantised
s      = sqrt(softplus(logits))
idx    = topk(s + bias, 6)                        # bias only affects selection
w      = gather(s, idx)                           # weights are taken from s without bias
w     /= (Σw + 1e-20)                             # norm_topk_prob, note it is 1e-20
w     *= 1.5                                      # routed_scaling_factor
```

⚠️ **Ambiguity**: the HF main config says `topk_method: noaux_tc`, but the reference implementation has **no `n_group`/`topk_group`** — it is simply top-6 over all 384. Follow the reference implementation.

- Expert order: **w1=gate, w2=down, w3=up** (not gate/up/down)
- SwiGLU clamp: `up` is clamped both ways to ±10, `gate` is only clamped from above at ≤10 (the lower bound is not clamped)
- the shared expert weight is always 1.0, is not multiplied by a routing coefficient, and runs for every token
- `bias_vl` replaces `bias` only for multimodal input (not needed in text mode)

### 4.6 engram (the biggest engineering risk)

**Each token looks up 48 rows** (2 layers × 24 columns), each row 256 B + 8 B scale = 264 B → **12.7 KB/token**, 96 small random reads.

24 columns = (2-gram, 3-gram, 4-gram) × 8 heads; each table is cut into 24 disjoint **prime bucket ranges**
(primes searched upward from 15999999), the layer1 sum is 384006168 and the layer14 sum is 384016682, matching the config exactly (this is strong validation of the algorithm).

```c
// the 4 multipliers for each layer are generated by numpy PCG64: np.random.default_rng(10007 * layer_id)
// multiplier_bound = (2^63-1) // 99092 // 2 = 46539438283891, take 4 of them, then *2+1
c[k] = blocked ? pad_id : compressed_cache[t-k];      // k=0..3, DEAD=-1, out of range is also blocked
r = c[0]*mult[0];
h2 = (r ^ (c[1]*mult[1])) % prime[ngram2][head];      // note the XOR uses the **unreduced rolling** value
h3 = (h2raw ^ (c[2]*mult[2])) % prime[ngram3][head];
h4 = (h3raw ^ (c[3]*mult[3])) % prime[ngram4][head];
row = h + offset[layer][col];
```

After that: embed(row) → 24×256=6144 dims → `wkv` → the first 4×5120 are keys, the remaining 5120 are values →

```
weight = q_weight * k_weight                          # the two only ever appear as a product, so they can be pre-multiplied
rstd   = rsqrt(mean(h^2)+1e-20) * rsqrt(mean(key^2)+1e-20)
dot    = Σ(h * weight * key) * rstd / sqrt(5120)
gate   = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6)))   # signed sqrt then sigmoid
h     += gate * value                                   # additive write
```

**Constants that must be exported from Python first and hard-coded into C**:

| Constant | Shape | Source |
|---|---|---|
| `token_map` (compressed token map) | int32[129280] | needs the HF normalizer (NFKC/NFD/StripAccents/Lowercase…), must be exported on the Python side |
| `multipliers` | int64[2][4] | numpy PCG64, **cannot be computed by hand**, must be exported on the Python side |
| `primes` | int64[2][3][8] | already computed (see §8) |
| `offsets` | int64[2][24] | already computed |
| `freqs_cis` | complex64[65536][32] × 2 sets | theta=10000 without YaRN / theta=160000 with YaRN |

The 203 GB table **cannot be materialised** (the reference implementation splits it across 8×64 GB with TP=8, which this machine does not have). You must do your own **mmap + MADV_RANDOM**, and it is strongly recommended to repack it offline into "264 B contiguous records" (interleaving the weight row and the scale row), which cuts per-token random reads from 96 to 48 and halves the number of page touches.

---

## 5. Parts that can be ignored for now

| Module | Can it be skipped | Cost |
|---|---|---|
| `mtp.*` (3 layers of nextn / dspark) | not required for inference — the reference implementation's `generate.py` never calls the speculative path | saves 0.61 GB, but **gives up the possibility of using it as a speculative draft head** (see §6) |
| `vision.*` + `aligner.*` | yes, do text-only mode | saves 1.01 GB, and `Gate.bias_vl` becomes None, engram does no masking |
| tokenizer chat template | there is **no** jinja in the repo, the template is Python logic in `encoding/encoding.py` | must be rewritten on the C side, or pre-encoded in Python and fed as token ids |

---

## 6. Performance budget (this is the key question for whether to start)

### Bytes that must be read per decode token (batch=1)

```
routed experts:  6 experts/layer × 40 layers × 18,800,640 B  = 4.20 GiB
shared experts:  40 layers × 35,424,000 B              = 1.32 GiB
------------------------------------------------------------
expert subtotal                                     ≈ 5.93 GB
engram:  2 layers × 24 rows × 264 B                       =   12.7 KB
```

Reference point: GLM-5.3-Flash is about 1.9 GiB per step. **This model is about 2.9× that.**

### How much memory can cache

128 GiB minus about 9.5 GB resident (attention weights 5.0, indexer/compressor 0.5, embed 1.32, head 1.32, engram.wkv 0.31, KV cache 0.2) →

```
Available expert cache ≈ 118 GB. Total experts 297.52 GB → hit-rate ceiling ≈ 40%
Even with an ideal cache, every token still has to read ≥ 3.2 GB from disk
```

### tok/s ceiling

| Assumption | Result |
|---|---|
| SSD random read 5.4 GiB/s (measured local ceiling), 3.2 GB/token | **≈ 1.6 tok/s** |
| Estimated from GLM's measured effective rate (IOPS/latency cannot be avoided) | **≈ 0.7 tok/s** |
| If every step went to disk | SSD sustained **13 GB/s**, impossible on this machine |

**Conclusion: with the whole expert bank on disk, bare decode reaches about 1.6 tok/s.**

> **2026-10-25 correction**: the table above counts the whole "4.5 GiB of expert bytes per token" as disk reads, whereas in
> reality an 82 GiB pin cache absorbs ~86% of it. Only ~0.6 GB/token actually goes to the SSD, so the bottleneck
> is not SSD bandwidth but kernel efficiency + how well the read latency is hidden. Measured decode has already reached **2.3–2.7 tok/s**;
> that rate is no longer written off. See §10.10.

### The only way to raise throughput: the built-in MTP/speculative head

The checkpoint ships 3 layers of nextn (`mtp.0/1/2`) + `markov_head` + `confidence_head`,
and the reference implementation's comments explicitly call the speculative decoding loop "out of scope". But for us it is precisely the means to raise throughput:

```
effective tok/s = (average tokens accepted per step) / per-step latency
```

Per-step latency is unchanged (the MoE still has to be run once), but one forward pass yields 2–4 tokens → effective throughput ×2~4.
If a single step is 0.7 tok/s, an acceptance rate of 3/4 gets you to about 2 tok/s. **This is a capability the models ship with, and it should be the main direction after M6.**

---

## 7. Milestones (each step is verifiable; do not move to the next until it is finished)

| # | Content | Acceptance criteria |
|---|---|---|
| **M0** | numerical oracle | install torch (the current system has no torch/numpy) or write a pure-C single-operator checker; able to run the reference implementation's first 3 dims on any 1 of the 40 layers and export the intermediate activations |
| **M1** | dequantisation assay layer | a standalone `tools/` mini program: reads FP4/I8 + E8M0, FP8 + E8M0, bit-exact against the oracle (including the `__builtin` NEON version) |
| **M2** | dtype contract | `src/model.c` recognises `deepseek_v41`; reports an explicit error for the parts that are not enabled (not today's misleading `missing/invalid dtype`) |
| **M3** | attention single operator | use the oracle to check Hyper-Connections, Sinkhorn, two-level top-k, block-diagonal wo_a, the dual RoPE; error vs the reference implementation < 1e-4 |
| **M4** | MoE | routing/SwiGLU clamp, FP4 expert GEMM; first get a single layer running on the CPU backend |
| **M5** | engram | mmap(MADV_RANDOM) + offline repacking tool (264 B interleaved records); verify that the row numbers computed by the hash are exactly identical to the oracle — **this is the step most likely to go wrong** |
| **M6** | full pipeline | 40 layers + text mode; output token sequence exactly identical to `generate.py` (temperature=0) |
| **M7** | performance Metal | FP4 GEMV Metal kernel, cross-layer prefetch, expert LRU (118 GB) |
| **M8** | MTP speculative | reuse `mtp.*`, push effective tok/s toward 2 |

---

## 8. Constants already computed (primes and offsets recomputed with the reference implementation, matching the config exactly)

layer 1 (`engram_layer_ids[0]`, sums to 384006168):

```
ngram2 primes  16000057 16000079 16000081 16000097 16000121 16000129 16000133 16000183
ngram3 primes  16000189 16000207 16000211 16000253 16000277 16000289 16000307 16000321
ngram4 primes  16000339 16000381 16000393 16000399 16000403 16000409 16000447 16000463
offsets      0 16000057 32000136 48000217 64000314 80000435 96000564 112000697
             128000880 144001069 160001276 176001487 192001740 208002017 224002306 240002613
             256002934 272003273 288003654 304004047 320004446 336004849 352005258 368005705
```

layer 14 (sums to 384016682):

```
ngram2 primes  16000477 16000487 16000499 16000507 16000511 16000573 16000609 16000627
ngram3 primes  16000667 16000669 16000693 16000697 16000711 16000729 16000759 16000769
ngram4 primes  16000781 16000799 16000813 16000819 16000841 16000877 16000879 16000889
offsets      0 16000477 32000964 48001463 64001970 80002481 96003054 112003663
             128004290 144004957 160005626 176006319 192007016 208007727 224008456 240009215
             256009984 272010765 288011564 304012377 320013196 336014037 352014914 368015793
```

Still to be exported from Python: `token_map[129280]`, `multipliers[2][4]`, `pad_id`.

---

## 9. Risk list

1. **You cannot start without an oracle**: you need torch to run the reference implementation; the current system has neither torch nor numpy (there is no way to validate a single line of C).
2. ⚠️ **Two ambiguities in the reference implementation**: whether Hyper-Connections is `C·R` or `Cᵀ·R`; whether `noaux_tc` really has no grouped top-k. These must be settled numerically; getting them wrong means silently wrong output.
3. **engram is a hotbed of silent errors**: one misaligned hash and every number is wrong, the output just becomes "similar but wrong". Row numbers must be compared one by one against the oracle.
4. **act_quant's inplace quantise-dequantise**: easy to assume it does not matter and skip it, but it is in fact the source of global bit-level accumulated error.
5. **Bare decode has a hard ceiling**: without speculative decoding, this model's decode rate is pinned by bytes per step and does not get past it on this machine. This conclusion has to be stated now, rather than waiting for the numbers and being embarrassed then.

---

## 10. Landed and measured (as of 2026-10-23)

Item 1 of the risk list in §9 above no longer holds: **you can validate without torch** — the verifier was rewritten in pure C,
implementing the same mathematics in parallel with the code under test, and comparing bit for bit against the hex-float traces the engine itself writes.

### 10.1 Validation tools (all pure C, standalone binaries outside `make cwick`)

| Tool | Purpose | Result |
|---|---|---|
| `tools/ds_verify` | all 48 shards / 510.29 GB opened, quantisation entry counts, real-weight GEMV | 256 e4m3 + 16 fp4 + 256 e8m0 correct; FP8 GEMV worst rel 1.4e-07, FP4 6e-08 |
| `tools/ds_optest` | per-operator invariants (rope inverse, Sinkhorn doubly stochastic, quantisation error bound, fp4 grid) | all 13 pass |
| `tools/ds_toktest` | tokenizer known ids + round-trip | all 30 pass |
| `tools/ds_layertest` | whole-block recompute vs engine trace, 40 layers × 3 positions | **6039 checks, 0 failure, worst rel/rms = 0 (bit-identical)** |
| `tools/ds_prompttest` | prompt encoding vs the byte strings in the reference README | all 11 pass |

`ds_layertest` uses `--tokens 1000 2000 3000` to cover prefill p=0/1/2, and `--all` to cover all
40 layers; the quantities compared include the 4 copies of the residual stream, hc_pre/post/comb, the compressed latent, the indexer's selected set,
the attention weight sum, each expert's gate/up/SwiGLU product/w2 output, and the routing weights and selected set.

### 10.2 Real engine defects fixed during bit-exact alignment

1. `ds_bf16_round` only added the rounding bias and **never cleared the low bits**, i.e. it never actually rounded to the bf16 grid.
2. `gate_temp` was never parsed (the key does not exist in config.json, the reference defaults to 1.0) → division by zero →
   router scores all `inf` → `sqrt(softplus(inf))` multiplied by `0` → MoE output all NaN.
3. top-k weight normalisation used "multiply by the reciprocal" whereas the reference uses "divide", a 1 ULP difference, enough to push some elements across the bf16 boundary.
4. `sparse_attn`'s return value is a bf16 tensor, so before the inverse rotation it must be rounded as a whole once.
5. the rope segment of the query vector must be bf16-rounded after rope (the reference writes back into the bf16 tensor in place).

### 10.3 Two defects in the verifier itself (the symptoms look a lot like an engine bug)

* From the second position onwards, `shared_topk` / `shared_cand` shared one buffer, whereas the engine stores per position — L=21
  was using the selection result of the last position. Fixed by making it `[ntok]`.
* The index key cache was indexed by "index source ordinal (0..7)" but allocated by "number of KV sources (4)", so the 5th index source
  (L=28) corrupted the heap outright. The reference semantics are that index keys follow the KV sources; fixed by switching to `src_slot`.

### 10.4 prompt encoding

`src/ds_prompt.c` is a C port of the text path of the reference `encoding/encoding.py`: BOS,
system / system mid-conversation, user, assistant (reasoning + content + DSML tool calls),
`latest_reminder`, `tool` folded into the user's `<tool_result>`, the tool schema block,
the numeric reasoning budget prefix, `drop_thinking` parsing, and the `task` shortcut token.
Image blocks are rejected by design (the vision tower is not loaded), not silently dropped.

`tools/ds_prompttest`'s cases are copied verbatim from the byte strings printed by `encoding/README.md`, so passing is
equivalent to "byte-identical to the reference format", not "looks about right".

**Wired up**: `run_generation` in `server.c` dispatches on the `model_type` prefix — `deepseek*` goes to
`ds_chat_encode()`, everything else goes to the GLM template. Two details:

* the request validation stage uses `cwick_json_write()` to preserve OpenAI's `messages` / `tools` arrays verbatim,
  because the DeepSeek template needs to read `reasoning_content`, `task`, `wo_eos` — fields that the `cwick_chat_msg`
  compact structure does not have;
* an absent effort used to be passed as "unset" so each checkpoint kept its own default budget (DeepSeek 75,
  GLM Max). Since §55 the absent case is the real level 关 and takes the same path as `reasoning_effort:"none"`.
  The rendering below is the historical "unset" output, kept as a byte-level rendering fixture:

```
<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)

<｜User｜>Say hi in one word.<｜Assistant｜><think>
```

### 10.5 Current performance

Current (evening of 2026-10-23, 128 GiB machine, 40-token list prompt, expert cache 94.3% hit):

| Metric | Value |
|---|---|
| decode | **1.083 s/token** (0.923 tok/s) |
| prefill | **1.374 tok/s** |
| Expert bytes per token | ≈4.5 GiB (6/384 × 40 layers × 18.8 MiB), of which about 1.1 GiB goes to the SSD |

decode single-step breakdown (`DS_TIMING=1`, 1083 ms total): expert matmul 43–52%,
attention 26–31% (including qkv+rope 7–9%, wo 12–14%, indexer 2%, sparse attn 2%).

Bottleneck ranking (by measurement):

1. **expert matmul takes nearly half, but it is not a compute problem**. The 8.49 GMAC kernel reaches
   54.8 GMAC/s on `tools/ds_gemvbench` (≈155 ms), yet measures ~500 ms;
   the missing 350 ms is waiting for the weight stream itself: 4.5 GiB per token is fed in from an 88.2 GiB cache mixed with
   the SSD, i.e. only ~9 GB/s effective. **Reducing bytes is the only real lever**, not swapping kernels.
2. **wo 140 ms / 3.02 GMAC** = 21.6 GMAC/s, also below the kernel ceiling (31.9–34.3).
   wo_a reads 1.34 GB of dense fp8 group by group, on the same ~9 GB/s road.
3. **prefill is still dominated by expert bytes**: grouping by expert within a chunk (see 10.7) already compresses the 2362 expert accesses
   of 89 tokens down to 942 distinct experts, but the larger n is, the lower the repetition, so the returns diminish.
4. Metal still does not help: autotune measures `[2048,4096]` FP8 GEMV at 514 µs on Metal vs
   330 µs on the CPU pool, so CPU is clearly chosen. There is no Metal kernel for FP4 yet.

> Long runs and short runs are not comparable: a short run (<30 steps) measures cold-cache numbers. On this machine the same
> configuration can drift by ~20%, so any A/B must be run alternately in the same session.


### 10.6 Four defects exposed by server-side diagnostics (afternoon of 2026-10-23)

Starting from a 263-line `cwick-serve.log` and `/v1/metrics`, four defects were located; all are fixed:

1. **the prompt template was crossed over** (P0). After switching to DeepSeek it was still using GLM's
   `[gMASK]<sop>...Reasoning Effort: Max`, while the tokenizer was already DeepSeek's
   128k vocabulary — the model received a sequence it had never been trained on. See 10.4.
2. **prefill metrics frozen** (P0). `g_pf_*` was only written by the GLM engine path, DeepSeek never updated it,
   so the dashboard showed `prefill_layer 0/45` throughout (45 being GLM's layer count). Now
   `ds_engine_generate` reports the token count at chunk boundaries, `ds_model_forward` reports the
   layer number in the layer loop, and folds the fraction completed within this chunk into `done`, otherwise a prompt smaller than `DS_CHUNK`
   would leave `done` at 0 forever. On model switch `cwick_metrics_set_model_info` also gained
   a reset of `g_pf_layers` (previously only `cwick_metrics_init` did that).
3. **GEMV single-threaded** (P0). `ds_fp4_gemv` / `ds_fp8_gemv` were scalar double loops, `sample(1)`
   caught 12 workers all in `__psynch_cvwait` with only 1 core computing. Changed to slice by **output row**
   through `cwick_pool_for` — there is no reduction between rows, so the split changes no accumulation order, and therefore
   the `--all` regression is still 6039 checks / worst rel-rms = 0.
4. **expert cache not wired up** (P1). There was no caching logic at all in `ds_engine.c`, `expert_cache_cap_bytes`
   was always 0, and every token re-read all experts from the SSD. Changed to reuse GLM's `cwick_stream`
   (LRU + ghost admission + F_NOCACHE): `ds_model_open` creates the stream, `ds_moe` fetches weights with
   `cwick_stream_expert_put_read`, and a cold key rejected by ghost falls back to mapping the shard —
   both paths point at the same bytes, so correctness does not depend on a hit.

There is also an operational trap worth remembering: cwick's listening socket uses `SO_REUSEPORT`, so an old `cwick serve` that has not exited cleanly
will bind 8001 at the same time as the new one, and the kernel hashes traffic between the two. The symptom is that `/v1/config`
and the log disagree (one says context 4096, the other says 131072), and `/v1/models` randomly shows different
`loaded` states. When investigating, `pgrep -x cwick` first to confirm there is only one process.

### 10.7 GEMV kernels and parallel attention (evening of 2026-10-23)

Three changes, all keeping the `--all` regression at 6039 checks / worst rel-rms = 0.

**One, a GEMV that computes 4 rows at a time.** The old inner loop was
`part += w.lo*x; part += w.hi*x+1;` — a **serial dependency chain** of 32 multiply-adds. An fp32 add
issues in one cycle and produces its result four cycles later, so per cycle a single row spends 3/4 of its time waiting on itself. Changed to compute
four rows at once: each row still has exactly the same serial summation order, only four independent chains run side by side. There is no
reduction between rows, so it is bit-exact.

| Shape | Single row | Four rows | Multiple |
|---|---|---|---|
| FP4 `[5120,2304]` (expert gate/up/down) | 2.55 GMAC/s | **54.78** | 21× |
| FP4 `[2048,4096]` | 2.55 | **51.68** | 20× |
| FP8 `[1024,4096]` (wo_a per group) | 2.43 | **31.88** | 13× |
| FP8 `[5120,8192]` (wo_b) | 2.43 | **34.33** | 14× |

("Single row" is the reference implementation built into `tools/ds_gemvbench`; `DS_GEMV_ROWS=1` switches this kernel
back to the single-row version for comparison.)

**Two, activations are dequantised only once.** Previously every row computed had to look up the K activation values in the table; with `af` hoisted out of the
GEMV, each row is left with only multiply-adds. `DS_ACT_MAX` was raised from 5120 to 8192: wo_b's
K = o_groups × o_lora_rank = 8192 was exactly out of bounds before, falling into the serial fallback, which at one point accounted for 30% of decode.
Now exceeding the limit falls back to a heap buffer — **the fast kernel is the only kernel**.

**Three, sparse attention dispatched to the pool per head.** Each head's QK dot product is again a 512-long serial
dependency chain, and the 64 heads are completely independent (each writes only its own 512 outputs and only reads shared KV), so slicing by
head across the thread pool changes no arithmetic. For that, `ds_sparse_attn` needs **one copy of the scores per head**:
the old `m->scores` was the indexer's scratch, and sharing it made parallelism impossible.

**The trap hit (worth remembering):** `cwick_pool_for` has an internal threshold `CWICK_POOL_MIN_WORK = 512`,
and for `n < 512` the whole thing just runs on the calling thread. The head count is 64, so the first change made
no difference at all (sparse attn was still 11%). Switching to `cwick_pool_for_items` (threshold 1, specifically
for "a few large tasks") brought it down to 2% immediately. **Before dispatching to the pool, confirm the task count clears the threshold.**

Same-session A/B (40-token list prompt, identical expert cache hit rate):

| Version | prefill | decode |
|---|---|---|
| Before this round | 38.3 s / 1.148 tok/s | 50.9 s / **0.767 tok/s** |
| Four-row GEMV | — | 0.812 tok/s |
| + parallel sparse attention | 32.0 s / **1.374 tok/s** | 42.3 s / **0.923 tok/s** |

Total decode 1.20×, prefill 1.20×. The room to keep going is not in compute: the kernel is already 2.7–3× faster than the measured
path, what is missing is the supply bandwidth for the 4.5 GiB weight stream per token.

### 10.8 Server side: metric semantics and log noise (early morning of 2026-10-24)

Four problems read out of `cwick serve`'s live logs and `/v1/metrics`, all of them "looks
normal, but actually you cannot read anything from it":

1. **the prefix query counted only the hits.** `cwick_metrics_prefix_query` used to be called only when
   `P > 0` (a hit was found), so `queries == hits` always held, and the hit-rate panel was either
   0/0 or exactly 100%, never able to show a cache miss. Now every query counts, misses included.
2. **`cwick:memory_ceiling_bytes` always 0.** The memory ceiling was only computed on the GLM side and never written into
   the registry; now both `cwick_mem_plan_log` (GLM) and `ds_model_open` (DeepSeek)
   publish it, and the log says whether the ceiling came from `CWICK_EXPERT_CACHE_GIB` or the `host/4`
   default.
3. **expert residency metrics disguised as vLLM's prefix cache.** `vllm:gpu_prefix_cache_*` /
   `vllm:gpu_cache_usage_perc` were actually reporting "residency of expert weight blocks", which any vLLM
   dashboard reads as prompt prefix hit rate (measured 84.3%). Renamed to
   `cwick:expert_cache_queries_total` / `cwick:expert_cache_hits_total` /
   `cwick:expert_cache_usage_perc`, no longer occupying the vLLM namespace.
4. **185 of 300 log lines were metrics polling.** `/metrics` `/health` are now logged at trace
   level and no longer print the request line a second time. DeepSeek's summary line gained
   the `cache=`/`streamed=`/`mem=` fields, consistent with the GLM side.

### 10.9 DeepSeek prefix cache (early morning of 2026-10-24)

In agent scenarios with long repeated prompts, the vast majority of TTL is spent recomputing the same system prompt.
The DeepSeek-side prefix snapshot is now usable, and **replay is bit-identical to the prefill segment it replaces**.

**What has to be stored.** There are only six pieces of cross-token state: each layer's sliding-window KV `win_kv`, compressed KV
`cmp_kv`, the compressor's not-yet-grouped pooling intermediates `cmp_state`/`cmp_score`, the indexer's keys
`idx_k`, engram's lookback `tok_cache`, and the position itself. The residual stream and per-token scratch
are internal to the token and are recomputed by the next layer, so they are not stored (the same reasoning as for the mHC state on the GLM side).

**How it is stored.** GLM's `pfx` file format is "a set of fixed-length arrays", and DeepSeek's arrays are completely different,
so the snapshot travels as an **opaque blob**: the word in the header that used to be padding is now
`kind` (0 = original layout, 1 = blob), the header byte count is unchanged, so old snapshots still load.
`ds_state_apply` validates magic/version/geometry (`n_layers`/`window`/`head_dim`/
`n_src`/`idx_head_dim`/`ratio`), and on a mismatch treats it as a miss and lets the caller prefill the whole segment rather than
crashing. The server takes one snapshot at the end of prefill and every `CWICK_PFX_BLOCK` tokens on long prompts
(DeepSeek has its own prefill loop, so the hook is attached to `ds_engine`).

**Two real defects fixed along the way**, both "computed fast but computed wrong":

1. `c->max_ratio` was never assigned (always 0). The compressor's partial grouping buffers are allocated from it,
   and the snapshot and reset follow it too, so both sides **silently skipped two arrays**, while the model kept writing into them.
   This defect also made "prefill in two segments" look non-equivalent to "prefill in one go" — in fact chunking itself was
   not the problem, it was that the partial grouping state was not carried over.
2. **`cmp_kv` / `idx_k` are `[source][max_seq_len][dim]`, but the snapshot copied them as one contiguous block
   of memory.** Copying `n_kv_source × pos` rows amounts to moving only the first
   `n_kv_source × pos` rows of source 0, leaving the other three sources all zero. The byte count is unchanged, so
   only a bit-exact comparison against "the prefill it replaced" exposes it. The symptom is that from layer 8 onwards (the first
   layer that reads source 1) the attention weights start to drift, rel-rms 0.58.

**How it was verified (pure C).** `tools/ds_run` gained three modes, and `DS_TRACE` gained the `FWDPOS`
anchor line (positions in the trace are "indices within this call", so they do not line up across calls):

| Mode | What it does |
|---|---|
| `--split-test <cut>` | whole prefill vs two segments, compare logits |
| `--state-diff <cut>` | compare the two prefills' snapshots segment by segment, report the first differing byte |
| `--prefix-test <cut>` | first segment prefill → snapshot → rewind → restore → second segment, compare logits |
| `--dump / --snap-dump <cut>` | trace only, handed to `/tmp/ds_trdiff` to compare tensor by tensor |

`ds_trdiff` pairs the two traces by (tag, layer, global position) and reports the first (and all)
differing elements. Locating defect 2 above, it pointed straight at it: **from layer 10, position 2 onwards**,
`PROBE_ATTW` had 3/129 differing elements — tracing back up, that is "the source 1 read from layer 8 onwards is zero".

**Measured results:**

| Case | Result |
|---|---|
| 6 tokens, cut = 1/2/3/4/5 | all 129280 logits **equal**, rel-rms = 0 |
| 32 tokens, cut = 8/16/31 | equal, rel-rms = 0 |
| 150 tokens, cut = 130 (crossing the 128 sliding-window wrap) | equal, rel-rms = 0, snapshot 11.85 MB |
| `ds_layertest --all --tokens 1000 2000 3000` | 6039 checks / 0 failure / rel-rms 0 |
| `make test` | PASS |

Snapshot size grows linearly with context: about 1.35 GB at 131072 context (win_kv 10.5 MB +
cmp_kv 1.07 GB + idx_k 268 MB), so `--prefix-cache-gib` on the DeepSeek side must be estimated
in "how many snapshots" rather than "how many tokens".


### 10.10 Throughput push: from 0.86 to 2.7 tok/s (2026-10-25)

Acceptance line: with `--backend metal`, going through the full API path (server + streaming), decode output ≥2 tok/s,
without pruning and without reducing precision. Every item below is a **same-session A/B**, 60-token long run taking steady state
(`DS_TIMING=1` + the access log's `decode=…ms/… steps … tok/s`).

#### Evolution of single-step latency

| Step | ms/step | tok/s | Dominant term at the time |
|---|---|---|---|
| Starting point | 1050 | 0.86 | head 37–40%, fp4 experts 26–36% |
| head parallel (12 threads split by output row) | 571 | 1.45 | fp4 experts 43–47% |
| fp4 expert GEMV on NEON | 530 | 1.48 | expert fetch 25–34% |
| dense fp8 weights expanded into a bf16 copy | 550 | 1.66 | expert fetch 37–52% |
| prefetch readers 1 → 4 | 404 | 2.47 | — |
| prefetch readers 6 (default) | **382** | **2.65** | expert matmul 33–36%, wo 15–16% |

Details of the four changes:

1. **head (129280×5120 bf16) used to be the single most expensive point in the whole path**: 448 ms, 40% of a step,
   while 12 cores used only 1 of them for it. Split by output row across `cwick_pool_for`, then use
   `vshll_n_u16(v,16)` to treat bf16 directly as the high 16 bits of fp32 (that is literally the definition of bf16),
   with four-way accumulation breaking the dependency chain: **448 → 11.1 ms** (`make bench` measures 119.2 GB/s).
2. **NEON for the fp4 expert GEMV**: all 16 e2m1 values have zero low bytes in fp16, so keep only a **high-byte** table,
   use `vqtbl1q_u8` to gather the bit patterns in one go, then `vshll_n_u8` up to fp16 and convert to fp32.
   Activations are split in advance into even/odd halves, avoiding shifts in the inner loop.
   `tools/ds_gemvbench --gemv fp4` measures **0.10 ms/call, 58.5 GB/s**; the pre-change
   scalar version is reproduced by the `fp4 old` line of `--gemv compare` (28.3 GB/s).
   Note that this tool **can only A/B back to back in the same session**: the same code measured 0.075 / 0.111 / 0.171 ms ten minutes apart,
   that is thermal throttling, not a regression.
3. **dense weights expanded into a bf16 copy at load time** (`ds_model.dense16`, about 11 GiB).
   fp8 GEMV's bottleneck is one table lookup + scaling per element; bf16 is pure `vld1q` + `vfmaq`, and can also
   use 4 independent accumulators: **38.5 → 119.2 GB/s**. The expansion is **exact** — e4m3 has only
   8 significant bits, which bf16's 8-bit mantissa holds; the worst relative error of `--gemv check` is
   **7.6e-7 (1 ulp)**, originating only from the order of the 4-way sum; the products themselves are bit-identical.
   `CWICK_DS_EXACT=1` turns this path off and returns to the reference order (scalar kernel + no expansion),
   measured cost **1.72 vs 2.24 tok/s (about −23%)**, used for per-layer oracle validation.
4. **prefetch readers 1 → 6** (`CWICK_PF_READERS`, 1..8). Key evidence: the hit rate and the
   `streamed` byte count are **completely unchanged** at the 1/4/6 settings (`84.0%(29028/5532)`,
   `45.86 GiB`), so the entire speedup comes from hiding SSD latency, not from reading more data.
   That is also what makes it fundamentally different from `CWICK_PREFETCH=1` (speculative reads, 2.7× the bytes, pure loss).

#### chunked MoE's expert reads changed to batched arenas

`n>1` prefill goes through `ds_moe_chunk`: compute the whole block's routing first, then read each expert in the union only once.
The problem is that it used **blocking** `ds_expert_fetch`, reading one expert at a time —
measured 2.5 GB/s, because a single 6 MiB read is **latency**-bound, not bandwidth-bound
(the same device reaches 9 GB/s with 4–8 in flight).

Changed to the same "prefetch → take → compute → pin" sequence as the decode path, rolling in batches sized to one prefetch zone's
capacity (6 experts × 3 matrices = 18 matrices): all 18 reads of a batch are issued before
the first multiply-add starts, and the remaining reads complete underneath that arithmetic.

| Batch width (`CWICK_DS_PF_BATCH`) | prefill (17 tokens) |
|---|---|
| 1 (= the pre-change order) | 11.75 s |
| 2 | 9.09 s |
| 4 | 8.13 s |
| 6 (default) | **8.00 s** |

The gain saturates at 4–6 — exactly the width of one arena zone, anything wider does not fit. The output is
**bit-identical**: the 1/3/6 batch-width builds produce the same md5, because each
(position, slot) is still an independent GEMV written into its own contribution, summed in slot
order at the end; batching only changes "when the reads happen". `make test` PASS.

Also fixed a diagnostic distortion along the way: the `cache=` line used to use "misses that passed the ghost admission gate" as the denominator
(`put_read` returns immediately without counting when it rejects a cold key), so it showed a nice but
incomparable number like 68.6%; after switching to `expert_cached` every query counts, and the same request in the same process reads
`8.5%(494/5299)` — **the numerator did not change, the denominator became honest**. Watch out for this when comparing hit rates across versions.

#### Full API path measurements (`sh ./start.sh deepSeek-v4.1-flash-1017`, `--backend metal`)

Server on 8001, `/v1/chat/completions` (streaming), `max_tokens` as in the gen column, numbers taken from the
access log's `prefill=… decode=…ms/… steps … tok/s`:

| req | prompt | gen | prefill | decode | end-to-end |
|---|---|---|---|---|---|
| 2 (first request in the process, cold cache) | 36 | 120 | 20.96 s | 2.294 tok/s | 1.65 tok/s |
| 3 | 36 | 120 | 15.68 s | 2.535 tok/s | 1.91 tok/s |
| 4 | 36 | 120 | 14.74 s | **2.599 tok/s** | 1.98 tok/s |
| 5 | 36 | 120 | 16.53 s | **2.724 tok/s** | 1.99 tok/s |
| 6 | 42 | 400 | 17.97 s | **2.438 tok/s** | **2.20 tok/s** |

Before the changes, the same request (req 7, 36→120): prefill 20.09 s, decode 2.568 tok/s, end-to-end 1.80.
So prefill went **20.1 → 15.7 s (−22%)**, decode unchanged (2.5–2.7).

**How to read this table.** decode is the "output rate", i.e. the speed a streaming client sees:
**every request is ≥2.29**, steady state 2.5–2.7. End-to-end also counts prefill, and prefill for a short prompt is
expensive: the union for 36 tokens is about 167 distinct experts/layer, requiring about **60 GiB** of reads — amortised per token
that is 0.43 s/token, slower than a decode step. This is the inevitable consequence of choosing 6 of 384 (only a large chunk
amortises it well: a 512-token chunk reads only 90 matrices/token, whereas 36 tokens need 557).
So **short prompt + short answer** end-to-end gets dragged below 2 by TTFT, whereas for **answers ≥200 token
end-to-end is already ≥2.2**.

About 28% of prefill time remains as exposed latency "after a batch of reads finishes and before the next batch starts"
(the `expert fetch` bucket under `DS_TIMING=1`). To eliminate it, batch b+1's reads must be in flight while batch b is
computing, but the semantics of `cwick_stream_prefetch_begin` are "drain the old zone first, then flip pages,
discarding the old zone's unclaimed tasks when flipping" — building a cross-batch pipeline would require changing thread lifetimes and task ownership,
and the risk exceeds the benefit, so it is not done this round.

### 10.11 Single-step byte accounting and the GPU offload verdict (2026-10-26)

The previous section pushed decode to 382 ms/step. This section first works out **where the 453 ms actually goes** (same machine,
same checkpoint, 60-token prompt, 24 new, `DS_TIMING=1`, GPU off), and then judges whether "hand some of the GEMVs to the GPU" is worth it
right now.

#### Phase breakdown (average of 23 decode steps, wall 453.6 ms)

| Bucket | Share | ms | Notes |
|---|---|---|---|
| expert matmul | 28.2% | 128 | 18 fp4 matrices/layer × 40 layers = 4.2 GB |
| expert fetch | 20.6% | 93 | 1.83 GB read from SSD (hit rate 48.7%) |
| wo | 13.4% | 61 | wo_a 8 groups + wo_b |
| qkv+rope | 9.1% | 41 | |
| **expert pin** | **7.8%** | **35** | see below, previously folded into fetch |
| attention | 7.7% | 35 | |
| shared+sum | 5.8% | 26 | |
| sparse attn | 3.0% | 14 | |
| head | 2.5% | 11 | |
| routing | 1.0% | 4.5 | |
| indexer | 0.5% | 2 | |

`pin` is **a bucket newly split out this round**: `ds_expert_pin` memcpys the expert blob in the arena one more time
into a cache slot, walking the same bytes twice. Previously it was folded into `expert fetch`, so "the disk is slow" was
overestimated and "bandwidth is double-charged" was hidden. Measures 35 ms (1.83 GB → about 52 GB/s, single-threaded memcpy).

#### How many bytes are actually moved per step

From the (n, k) printed by `dsgpu-verify` and the per-step call counts (bf16 dense, 2 bytes/element):

| Shape | Calls/step | GB/step |
|---|---|---|
| wo_b 5120×8192 | 40 | 3.36 |
| indexer 32768×1280 | 40 | 3.36 |
| wo_a 1024×4096 ×8 groups | 320 | 2.68 |
| shared w1/w3 2304×5120 | 80 | 1.89 |
| shared w2 5120×2304 | 40 | 0.94 |
| engram 25600×6144 | 2 | 0.63 |
| q 1280×5120 | 40 | 0.52 |
| kv 512×5120 + the rest | — | 0.32 |
| **dense total** | ~600 calls | **13.7** |
| **experts (fp4) total** | 720 calls | **4.2** |

**17.9 GB per step, 453 ms ⇒ 39.5 GB/s aggregate bandwidth.** Two conclusions drop out directly:

1. **dense is the bulk, not the experts** (13.7 GB vs 4.2 GB), but it only takes 176 ms ⇒ 78 GB/s,
   already close to the measured ceiling of these kernels (`--gemv shapes`: 100–270 GB/s, hot buffers).
2. **experts are only 24% of the bytes but eat 56% of the time**: fetch 93 + pin 35 + matmul 128.
   fp4 decoding is ALU-heavy (32 lookups + 16 multiply-adds per 32 elements), matmul achieves only 33 GB/s
   over 4.2 GB, whereas the same kernel does 48 GB/s on a hot buffer.

#### GPU offload: the kernels are correct, the interface is wrong

The two new kernels (`kernel_cwick_ds_fp4_gemv` / `kernel_cwick_ds_bf16_gemv`) reproduce the CPU
summation tree bit for bit and split by output row, **validation shows zero error**: `--gemv gpucheck` on the real shape
`[2304, 5120]` gives `worst rel 0.000e+00`, and still 0 on the second call; `CWICK_DS_GPU_VERIFY=1`'s
built-in runtime check (recomputing the GPU's rows and comparing row by row) never reported a single fp4 error across a whole forward pass.

But throughput shows this interface cannot win:

| Shape | CPU only | Offload 0.72 | GPU only |
|---|---|---|---|
| q 1280×5120 | **0.050 ms** | 0.129 | 0.115 |
| wo_a 1024×4096 | **0.084** | 0.106 | 0.043 |
| wo_b 5120×8192 | **0.673** | 0.940 | 1.037 |
| shared w2 5120×2304 | **0.136** | 0.448 | 0.494 |
| indexer 32768×1280 | **0.663** | 1.012 | 1.217 |
| engram 25600×6144 | 2.666 | **2.180** | 2.266 |

(`make bench && tools/ds_gemvbench --gemv shapes`, CPU pool of 12 cores.)

A single command buffer is only ~80 GB/s, whereas 12 matrices sharing one buffer is **198 GB/s**
(`gpucheck`); the CPU pool does 100–270 GB/s on the same bytes. The reason is simply that one submit+wait
costs a fixed ~0.2 ms regardless of matrix size — and a decode step has ~1400 GEMVs. End-to-end consequence:
**763 ms/step (vs 461 all-CPU)**; adding a "nothing under 16 MB goes to the GPU" gate brings it back to
**518 ms/step**, still a loss (`CWICK_DS_GPU_MIN_MB`).

So **`CWICK_DS_GPU`'s default is back to 0**, and the kernels and the verification tool are kept: the bf16 kernel does
198 GB/s under batched submission and is bit-exact; what is missing is an interface for "submit several matrices at once", not a kernel.

#### A trap worth writing down

The first end-to-end A/B produced garbled output + 0.58 tok/s + the hit rate cut in half, and was briefly judged a kernel arithmetic error.
The real cause was that **`./cwick` was a stale binary** (after changing the NEON branch of `ds_fp4_gemv` it was not rebuilt with
`make cwick`), while `tools/ds_gemvbench` was built later, hence "bench correct, engine garbled".
The way to tell them apart now: `CWICK_DS_GPU_VERIFY=1` recomputes "the GPU's rows" and compares row by row,
and 0 errors means it is not the kernel; when the output is garbled, **run `make cwick` first**.

#### What still stands between here and 5 tok/s

200 ms/step means 17.9 GB has to be moved at **89 GB/s**. Between today's 39.5 GB/s and that number
there are only three routes, and none of them is tuning:

1. ~~**Eliminate pin's second copy** (35 ms)~~ **Tried, zero gain, rolled back.** The approach was to let the
   ghost gate decide residency **before the read**, have `cwick_stream_expert_reserve` grab a slot,
   `cwick_stream_prefetch_submit_into` land the pread directly into it, and `take` get owned=1,
   so `pin` disappears entirely. Fully implemented and working, output byte-for-byte unchanged, but **wall clock 452.1 ms vs 453.6**:
   of the 35 ms in the `pin` bucket, 17 ms was already overlapping with the SSD wait (it waits on the same batch of 18 reads,
   and the CPU does the copy in the gaps), and the remaining 18 ms moved into `fetch` and `matmul`. Conclusion:
   **this is not where the bottleneck is, and it is not worth complicating the read path for**, the code has been reverted.
2. **Batched GPU submission**: CPU and GPU streaming the same unified memory in parallel is what could push aggregate bandwidth
   above 100 GB/s. The prerequisite is putting naturally independent matrices like wo_a's 8 groups and shared's w1/w3 into
   one command buffer, and deferring the wait to the dependency point. This is the only big lever not yet disproved.
3. ~~**Raise the expert hit rate**~~ **Checked, capacity is exhausted, no headroom.** `--verbose`'s
   `expert pin set: 79.43 GiB used of 81.86 GiB cap` shows the cache is already **97% full**,
   so 48.7% is not the admission gate (ghost gate) being too tight, it is the direct consequence of 81.86 GiB against 271 GB of expert
   weights — by capacity ratio that is only 30%, and 48.7% is already the surplus that "routing concentration" brings.
   Raising the hit rate further would need more memory or a different weight precision, and both are blocked by constraints.

If item 2 works out, only then is there a chance of reaching 250–280 ms/step (3.5–4 tok/s); even so, 5 tok/s
(200 ms/step) requires the dense 13.7 GB and the expert 4.2 GB to run almost fully in parallel on two engines,
and the SSD's 1.83 GB to be completely hidden — that is the ceiling of this machine's current structure, not something tuning can reach.

### 10.12 Hardware bandwidth ceiling and the 5 tok/s verdict (2026-10-26)

The previous section listed three routes. This section **runs all three**, and adds the most critical measurement: how much memory bandwidth
this machine can actually deliver. The conclusion is that 5 tok/s is unreachable on this machine's structure, and that this is not an implementation-quality problem.

#### Bandwidth is 123 GB/s, not 400

Measured with a bare read loop, 12 threads, each thread doing an independent 64-bit streaming sum (`/tmp/bwtest2` and similar):

| Working set | 4 threads | 8 threads | 12 threads | 24 threads | 32 threads |
|---|---|---|---|---|---|
| 1 GiB | 118.0 | 121.1 | — | — | — |
| 4 GiB | 123.5 | 123.9 | 123.2 | 122.7 | 122.3 |
| 16 GiB | — | — | 123.6 | — | — |

**4 threads already saturate it; adding up to 32 changes nothing; 1/4/16 GiB are exactly the same.** Improving the MLP (8-way accumulation,
32 8-byte loads per round) does not change it either. This is the top of the DRAM controller, not a badly written loop.

Two direct corollaries:

1. **dense is already against the ceiling**: the `wo` bucket runs at 99 GB/s = 80% of peak, and in `--gemv shapes`
   wo_b/indexer are 125-127 GB/s. **Further optimising the dense kernels is pointless; only reducing bytes helps.**
2. one step is 17.9 GB, so even a pure streaming read takes **145 ms**; 5 tok/s (200 ms) requires **89.5 GB/s =
   73% of peak sustained bandwidth**, while also carrying the SSD path. Today's 453 ms = **39.5 GB/s =
   32%**.

#### The way to reduce dense bytes: native fp8, vetoed by the checkpoint

The dense weights are **FP8 e4m3** in the checkpoint, expanded into a bf16 copy at load time by
`bind_q` (`ds_model.dense16`, 11 GiB). Going native fp8 would kill three birds with one stone: dense bytes halve (13.7→6.85 GB),
5.5 GiB of resident memory freed (returnable to the expert cache), and `ds_gemv_q` already routes to
`ds_fp8_gemv` when `w16==NULL`.

**Veto reason (measured)**: after adding a scan to `bind_q`, 40 layers × 8 fp8 tensors per layer =
**330 tensors, all containing exception bytes**. The fast decode `((b & 0x7F) << 7) + 0x2000 | ((b & 0x80) << 8)`
(hand-verified correct for 1.0 / 0.5 / −1.0 / 448) is exact for normal values, but two corners are not affine:

| Byte | True value | Fast decode gives | Error |
|---|---|---|---|
| e=0, m=4 (subnormal) | 4·2⁻⁹ = 0.0078 | 2⁻⁷·1.5 = 0.0117 | **+50%** |
| e=0, m=1 | 2⁻⁹ = 0.00195 | 2⁻⁷·1.125 = 0.00879 | +350% |
| e=15, m=7 | NaN | 480 | — |

This is not a rounding difference, it is a real error, and it would directly break `ds_layertest`'s bit-exactness. Patching the exceptions takes two
`vqtbl1q_u8` + zip + sign merge + bsl blend, about 15 instructions/16 values, taking decoding from 6 instructions/16 values to
29; **bf16 needs only 6 instructions/16 values** (`vshll_n_u8(v,8)` then `vshll_n_u16(.,16)`, just shifts).
Converted to instructions per byte: fp8 ≈ 1.5, bf16 ≈ 0.48. fp8 would immediately go from bandwidth-bound to issue-bound,
and even halving the bytes would not catch up. Verdict: **abandoned**.

#### The other four, all zero gain

| Change | Result |
|---|---|
| Eliminate pin's second copy (§10.11) | 453.6 → 452.1 ms, the copy already overlapped with the SSD wait |
| GPU split by row | 763 ms/step (vs 461), single-matrix submit+wait is a fixed ~0.2 ms |
| `CWICK_PF_READERS` 2 / 6 / 12 | matmul bucket 26.3% / 26.5% / 26.6%, no difference |
| fp4 kernel unrolled 2 blocks + `prfm` | gate/up 65.5→75.6 GB/s, but down 79.1→**53.5**; end-to-end 493 ms (vs 453). Reverted |
| `CWICK_EXPERT_CACHE_GIB=95` (vs auto 82) | 2.067 vs 2.055 tok/s, hit 33.2% vs 32.5%, mem 95.78 GiB — **RAM is nearly full, no space** |

#### Verdict

200 ms/step needs 89.5 GB/s. The ceiling is 123 GB/s. Between the two there is only 27% headroom, and today
not even half of it is used (39.5); the gap breaks down as: dense is already at 80% (no space), expert matmul is at 36 GB/s
(that kernel's cold-data bench is 65 GB/s, that ~55 ms is the only genuinely available headroom), and pin and fetch
are each covered by SSD latency. Draining that one remaining headroom completely gets you to about 300 ms (3.3 tok/s) at best.

**5 tok/s is unreachable on this machine**, unless: the bytes per step are reduced (at full precision 271 GB of experts does not fit in 128 GB,
and fp8 dense has been disproved), or you move to a machine with higher memory bandwidth.
