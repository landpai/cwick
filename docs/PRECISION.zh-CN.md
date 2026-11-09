# 精度现状：两个模型各自的权重精度、数值路径与验证覆盖

<p align="center"><a href="PRECISION.md">English</a> · <a href="PRECISION.zh-CN.md">简体中文</a></p>

这份文档回答一个问题：**cwick 跑出来的数，离这个 checkpoint 本来该算出的数有多远。**
两个模型的答案完全不同，因为它们落在两种不同的权重格式上，而验证力度也不相同。

结论先行：

| 模型 | 权重来源精度 | cwick 是否再量化 | 激活 | 验证力度 | 可信度 |
|---|---|---|---|---|---|
| DeepSeek-V4.1-Flash-1017 | FP4 e2m1 专家 + FP8 e4m3 attention（原生） | **否** | 量化到 e4m3（与参考一致） | 40 层 × 3 位置逐位对拍 | **最高** |
| GLM-5.3-Flash | FP8 e4m3 + F32 per-128×128 scale（原生） | **否** | **不量化**（fp32） | 仅 `doctor` 结构检查 | 中（见 §2） |

两个模型都没有在 checkpoint 基础之上再引入量化——流量的每一处损失都来自 checkpoint
自己带的那一次量化。区别只在有没有证据。

---

## 1. DeepSeek-V4.1-Flash-1017：FP4 专家，逐位对拍过

权重格式（实测 48 个分片 header，不是猜的）：

| 部分 | 格式 | 张量 dtype |
|---|---|---|
| routed experts | FP4 e2m1 双 nibble 打包，每 32 个 K 元素一个 E8M0 scale | `I8` + `F8_E8M0` |
| attention / dense | FP8 e4m3 + 32×32 block scale | `F8_E4M3` + `F8_E8M0` |
| engram | FP8 e4m3 + 每 32 元素一个 scale | `F8_E4M3` + `F8_E8M0` |
| norm / bias / sink | BF16、F32 | — |

张量命名是**无前缀的 `layers.N.*`**（另有 `vision.*` / `aligner.*` 在多模态分片里）。

激活走 dynamic quantization 到 e4m3 + ue8m0 scale（`ds_act_quant_block`），与参考一致。

**验证（2026-10-25 01:12 用当前二进制重跑）**：

```
tools/ds_layertest ./deepseek-ai/DeepSeek-V4.1-Flash-1017 4096 ./tools/ds_run \
    --tokens 1000 2000 3000 --all
-> 6039 checks, 0 failure(s), worst rel/rms 0
tools/ds_optest
-> 13 checks, 0 failure(s)
```

`ds_layertest` 是把整块重算的结果和引擎 trace 逐位比：残差流 4 份拷贝、hc_pre/post/comb、
压缩 latent、indexer 选中集、attn 权重和、每个专家的 gate/up/SwiGLU 乘积/w2 输出、
路由权重与选中集。**worst rel/rms = 0 是逐位一致**，不是"误差很小"。

调和过程中修掉过 5 个真实引擎缺陷（DEEPSEEK-PORT §10.2），其中 `ds_bf16_round`
只加舍入偏置不清低位这一条，等于从来没真正舍入到 bf16 网格。

### 1.1 一个工具缺陷：`doctor` 对 DeepSeek 无效

`cwick doctor --root ./deepseek-ai/DeepSeek-V4.1-Flash-1017` 会打几十行 FAIL：

```
cwick doctor: FAIL missing model.language_model.layers.0.input_layernorm.weight
```

原因是 doctor 的必备张量清单只写了 GLM 的 `model.language_model.*` 命名，而 DeepSeek
的张量是 `layers.N.*`。**这是自检工具的缺陷，不是运行时适配问题**（serve 走的是
`ds_model.c`，能正常加载）。它让"体检"在 DeepSeek 上完全失去意义，待修。

---

## 2. GLM-5.3-Flash：FP8 e4m3，语义已实测确认，但对拍缺失

`cwick doctor`：`fp8/e4m3`，62 shards / 76108 tensors / 305.8 GiB，专家 bank 290.32 GiB。

权重是 FP8 E4M3，配一个 F32 `*_scale_inv`，形状 `[rows/128, cols/128]`：

```
model.language_model.layers.45.mlp.experts.0.gate_proj.weight         F8_E4M3 [2048, 4096]
model.language_model.layers.45.mlp.experts.0.gate_proj.weight_scale_inv  F32   [16, 32]
```

### 2.1 `scale_inv` 到底是乘还是除——实测结论：乘

"scale_inv" 是个有歧义的名字：它可能是量化器乘的数（448/amax，反量化要**除**），
也可能是反量化器乘的数（amax/448，反量化要**乘**）。两者差 scale²，选错就是换了个模型。

`tools/pf8probe` 不假设任何一种，只测两件能区分它们的事（`make probe` 构建）：

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

两条判据指向同一个答案：

1. **块内 max|q| 均值 371 / 满量程 448**——量化器把块的 amax 顶到了接近 e4m3 上限，
   这正是"q = w / s，s = amax/448"的用法。若 scale_inv 是 448/amax，码值会挤在
   量程底部，白白丢掉 4 位指数里的几位，没有量化器会这么干。
2. **乘出来的权重 mean|w| = 0.0168，与同 checkpoint 的 BF16 参照 0.0160 同一量级**；
   除出来是 2.7e5，荒谬。

运行时代码 `spine.c:892` 用的正是乘法
（`cwick_fp8_e4m3_to_f32(p[i]) * si[...]`，`CWICK_ST_BLOCK = 128`），与判定一致。

### 2.2 激活不量化——比参考更准，但不等价

config 里 `activation_scheme: dynamic`。参考实现（vLLM/HF）会把激活也动态量化到 e4m3；
**cwick 没有做这一步**：`cwick_gemv_fp8` 和它的 Metal twin 的激活参数都是
`const float *`（fp32 原值），只有权重是 fp8。

所以 GLM 侧的语义是 **fp32 激活 × fp8 权重**，比参考的 fp8×fp8 **更精确**（少了一次
激活量化损失），但与参考**不等价**。这是一处已知偏差，不是 bug——但没有测试钉住它。

### 2.3 覆盖缺口（这是 GLM 目前最弱的地方）

| 缺口 | 说明 |
|---|---|
| 无 FP8 单测 | `tests/test_cwick.c` 里 fp8 / e4m3 / scale_inv **零匹配**（bf16、mxfp4 都有） |
| 无逐层对拍 | GLM 没有 `ds_layertest` 那样的工具；34 层 KDA + 11 层 DSA 是自研算子，无参考比对 |
| 三条 GEMV 实现无互验 | fp8 GEMV 有 NEON（`tensor.c`）、scalar（`CWICK_GEMV_SCALAR=1`）、Metal GPU 三份，没有等价性测试把它们钉在一起 |
| 无端到端基准 | 日志里没有 GLM 真实 checkpoint 的生成记录可对照 |

对比 DeepSeek 的 6039 项逐位对拍，GLM 目前**只有"权重反量化方向是对的"这一条实测证据**，
前向算子的正确性靠的是结构检查（`doctor`）和代码审阅。

---

## 3. 复现命令

```sh
make test                      # 夹具级等价性（bf16/mxfp4 契约、gemm==gemv）
make probe                     # 构建 tools/pf8probe
./tools/pf8probe ./zai-org/GLM-5.3-Flash
./tools/ds_optest                                    # DeepSeek 算子不变量
./tools/ds_layertest ./deepseek-ai/DeepSeek-V4.1-Flash-1017 4096 ./tools/ds_run \
    --tokens 1000 2000 3000 --all                    # DeepSeek 逐层逐位对拍
./cwick doctor --root ./zai-org/GLM-5.3-Flash        # 结构自检（DeepSeek 上误报，见 §1.1）
```
