# Precision status: weight formats, numeric paths and verification coverage for both models

<p align="center"><a href="PRECISION.md">English</a> · <a href="PRECISION.zh-CN.md">简体中文</a></p>

This document answers one question: **how far are the numbers cwick produces from the
numbers this checkpoint was supposed to produce.**
The answer is completely different for the two models, because they sit on two different
weight formats, and the verification effort differs as well.

Bottom line first:

| Model | Source weight precision | Does cwick re-quantize | Activations | Verification effort | Confidence |
|---|---|---|---|---|---|
| DeepSeek-V4.1-Flash-1017 | FP4 e2m1 experts + FP8 e4m3 attention (native) | **No** | quantized to e4m3 (same as reference) | 40 layers × 3 positions, bit-exact comparison | **Highest** |
| GLM-5.3-Flash | FP8 e4m3 + F32 per-128×128 scale (native) | **No** | **no quantization** (fp32) | only the `doctor` structural check | Medium (see §2) |

Neither model introduces any quantization on top of the checkpoint — every bit of loss
along the flow comes from the single quantization the checkpoint itself carries.
The only difference is whether there is evidence.

---

## 1. DeepSeek-V4.1-Flash-1017: FP4 experts, verified bit-exact

Weight formats (measured from 48 shard headers, not guessed):

| Part | Format | Tensor dtype |
|---|---|---|
| routed experts | FP4 e2m1, two nibbles packed per byte, one E8M0 scale per 32 K elements | `I8` + `F8_E8M0` |
| attention / dense | FP8 e4m3 + 32×32 block scale | `F8_E4M3` + `F8_E8M0` |
| engram | FP8 e4m3 + one scale per 32 elements | `F8_E4M3` + `F8_E8M0` |
| norm / bias / sink | BF16, F32 | — |

Tensor names are **unprefixed `layers.N.*`** (plus `vision.*` / `aligner.*` in the multimodal
shards).

Activations go through dynamic quantization to e4m3 + ue8m0 scale (`ds_act_quant_block`),
matching the reference.

**Verification (re-run 2026-10-25 01:12 with the current binary)**:

```
tools/ds_layertest ./deepseek-ai/DeepSeek-V4.1-Flash-1017 4096 ./tools/ds_run \
    --tokens 1000 2000 3000 --all
-> 6039 checks, 0 failure(s), worst rel/rms 0
tools/ds_optest
-> 13 checks, 0 failure(s)
```

`ds_layertest` recomputes whole blocks and compares the result against the engine trace bit
for bit: the 4 copies of the residual stream, hc_pre/post/comb, the compressed latent, the
indexer selection set, the attn weight sums, each expert's gate/up/SwiGLU product/w2 output,
and the routing weights and selection set. **worst rel/rms = 0 means bit-identical**, not
"the error is very small".

Five real engine defects were fixed during the reconciliation (DEEPSEEK-PORT §10.2), one of
which — `ds_bf16_round` adding only the rounding bias without clearing the low bits —
amounted to never actually rounding onto the bf16 grid.

### 1.1 A tooling defect: `doctor` is useless for DeepSeek

`cwick doctor --root ./deepseek-ai/DeepSeek-V4.1-Flash-1017` prints dozens of FAIL lines:

```
cwick doctor: FAIL missing model.language_model.layers.0.input_layernorm.weight
```

The reason is that doctor's required-tensor list only spells out GLM's
`model.language_model.*` naming, whereas DeepSeek's tensors are `layers.N.*`. **This is a
defect in the self-check tool, not a runtime adaptation problem** (serve goes through
`ds_model.c` and loads fine). It makes the "health check" meaningless on DeepSeek; to be
fixed.

---

## 2. GLM-5.3-Flash: FP8 e4m3, semantics confirmed by measurement, but no bit-exact comparison

`cwick doctor`: `fp8/e4m3`, 62 shards / 76108 tensors / 305.8 GiB, expert bank 290.32 GiB.

The weights are FP8 E4M3, paired with an F32 `*_scale_inv` of shape `[rows/128, cols/128]`:

```
model.language_model.layers.45.mlp.experts.0.gate_proj.weight         F8_E4M3 [2048, 4096]
model.language_model.layers.45.mlp.experts.0.gate_proj.weight_scale_inv  F32   [16, 32]
```

### 2.1 Is `scale_inv` a multiplier or a divisor — measured conclusion: a multiplier

"scale_inv" is an ambiguous name: it could be the number the quantizer multiplies by
(448/amax, so dequantization must **divide**), or the number the dequantizer multiplies by
(amax/448, so dequantization must **multiply**). The two differ by scale²; picking the wrong
one means you are running a different model.

`tools/pf8probe` assumes neither, and only measures the two things that can distinguish them
(built with `make probe`):

```
$ ./tools/pf8probe ./zai-org/GLM-5.3-Flash \
      model.language_model.layers.45.mlp.experts.0.gate_proj.weight \
      model.language_model.layers.45.eh_proj.weight

E4M3 range usage per 128-block (448 = full scale):
  mean max|q| = 371.0   min = 224.0   max = 448.0   (512 blocks)

scale_inv magnitudes:
  min=0.000244141  mean=0.000266075  max=0.000976562  (n=512)

reconstructed weights (1060 samples):
  w = q * scale_inv      mean|w|=0.0168385  rms=0.0211805  max|w|=0.078125
  w = q / scale_inv      mean|w|=265667     rms=344560     max|w|=1.31072e+06

BF16 reference ...eh_proj.weight [4096, 8192]
  reference              mean|w|=0.015971   rms=0.0395914  max|w|=2.23438
```

Both criteria point to the same answer:

1. **Mean within-block max|q| 371 against a full scale of 448** — the quantizer pushes each
   block's amax close to the e4m3 ceiling, which is exactly the usage of "q = w / s,
   s = amax/448". If scale_inv were 448/amax, the code values would bunch up at the bottom of
   the range, throwing away several of the 4 exponent bits for nothing; no quantizer does
   that.
2. **Multiplying gives weights with mean|w| = 0.0168, the same order of magnitude as 0.0160
   for the BF16 reference from the same checkpoint**; dividing gives 2.7e5, which is absurd.

The runtime code at `spine.c:892` uses exactly the multiplication
(`cwick_fp8_e4m3_to_f32(p[i]) * si[...]`, `CWICK_ST_BLOCK = 128`), consistent with the
verdict.

### 2.2 Activations are not quantized — more accurate than the reference, but not equivalent

The config says `activation_scheme: dynamic`. The reference implementation (vLLM/HF)
dynamically quantizes the activations to e4m3 as well;
**cwick does not do this step**: the activation parameters of `cwick_gemv_fp8` and of its
Metal twin are both `const float *` (raw fp32 values), and only the weights are fp8.

So the semantics on the GLM side are **fp32 activations × fp8 weights**, which is **more
precise** than the reference's fp8×fp8 (one activation quantization loss fewer), but **not
equivalent** to the reference. This is a known deviation, not a bug — but no test pins it
down.

### 2.3 Coverage gaps (the weakest point for GLM right now)

| Gap | Description |
|---|---|
| No FP8 unit tests | **zero matches** for fp8 / e4m3 / scale_inv in `tests/test_cwick.c` (bf16 and mxfp4 both have them) |
| No layer-by-layer comparison | GLM has no tool like `ds_layertest`; the 34 KDA layers + 11 DSA layers are hand-written operators with no reference to compare against |
| Three GEMV implementations with no cross-check | the fp8 GEMV exists in three copies — NEON (`tensor.c`), scalar (`CWICK_GEMV_SCALAR=1`) and Metal GPU — and no equivalence test pins them together |
| No end-to-end baseline | the logs contain no generation record from the real GLM checkpoint to compare against |

Compared with DeepSeek's 6039 bit-exact checks, GLM currently has **only one piece of
measured evidence: "the dequantization direction for the weights is right"**, and the
correctness of the forward operators rests on a structural check (`doctor`) and code review.

---

## 3. Reproduction commands

```sh
make test                      # fixture-level equivalence (bf16/mxfp4 contract, gemm==gemv)
make probe                     # build tools/pf8probe
./tools/pf8probe ./zai-org/GLM-5.3-Flash
./tools/ds_optest                                    # DeepSeek operator invariants
./tools/ds_layertest ./deepseek-ai/DeepSeek-V4.1-Flash-1017 4096 ./tools/ds_run \
    --tokens 1000 2000 3000 --all                    # DeepSeek layer-by-layer bit-exact comparison
./cwick doctor --root ./zai-org/GLM-5.3-Flash        # structural self-check (false positives on DeepSeek, see §1.1)
```
