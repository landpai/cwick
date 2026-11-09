# DeepSeek-V4.1-Flash-1017 移植规格

<p align="center"><a href="DEEPSEEK-PORT.md">English</a> · <a href="DEEPSEEK-PORT.zh-CN.md">简体中文</a></p>

> 目标：让 cwick 能跑 `deepseek-ai/DeepSeek-V4.1-Flash-1017`（goal.md「当前适配模型列表」第 2 条）。
> 本文件记录**标识符、用途、性能这类事实**，可自由讨论。
> 所有架构细节来自模型 checkpoint 自带的 `inference/` 参考实现，而非任何外部工程。
> 标注 ⚠️ 的是参考实现本身有歧义、需要数值验证后才能定稿的地方。

---

## 0. 结论先行

| 项目 | 结论 |
|---|---|
| 能否支持 | 可以，但要**新增一套前向算子（反量化 / 注意力 / MoE / engram）**，不是给 `model_type` 放宽校验 |
| 工作量 | 约 2500–3000 行 C：FP4 反量化 GEMM、activation 量化、Hyper-Connections Sinkhorn、engram 哈希查表、indexer 二级 top-k、块对角 wo_a |
| 早期对 decode 速率的判断 | **当时判定物理上达不到**，依据是「每 token 必从磁盘读 ~3.2 GB ÷ 本机 SSD 5.4 GiB/s ≈ 1.6 tok/s」。**该结论已于 2026-10-25 被实测推翻**：专家 pin 缓存 82 GiB + 86% 命中的稳态下，SSD 只承担约 0.6 GB/token，decode 实测 **2.3–2.7 tok/s**。见 §10.10 |
| 现实上限 | 裸 decode 实测 2.3–2.7 tok/s（服务端 119 步，`--backend metal`）；要再往上仍需投机解码（checkpoint 自带 3 层 nextn 头） |
| 建议顺序 | M0 数值 oracle → M1 反量化 → M2 单算子校验 → M3 attention → M4 MoE → M5 engram → M6 全链路 |

---

## 1. 现状：引擎为什么拒绝它

```
$ ./cwick serve --root deepseek-ai/DeepSeek-V4.1-Flash-1017 --port 18001 --backend metal
cwick: 2026-10-23 02:32:22.503 [info +0.00s] [init] startup failed: config is missing/invalid dtype
```

失败点在 `src/model.c` 的契约校验，两处硬性不匹配：

1. `quantization_config` 只有 `quant_method: fp8`，没有 `fmt: e4m3` 字段；
2. `text_config` 没有 `kv_lora_rank`——因为这个模型**不用 MLA 的 KV 低秩压缩**，head_dim=512 是全秩的。

目前的 `src/model.c` 假定「所有模型 = GLM」：-eck- 上这是正确的（宁可拒绝也不给错误输出），但它意味着 DeepSeek 支持是**新增一套前向**，不是放宽校验。

两个模型的关键差异：

| | GLM-5.3-Flash | DeepSeek-V4.1-Flash-1017 |
|---|---|---|
| 专家 dtype | FP8 e4m3 | **FP4 e2m1（打包进 int8）** |
| 专家数量 / topk | 160 / ? | 384 / **6** + 1 shared |
| 注意力 | KDA | MLA 变体 + **压缩 KV + 索引器 top-k** |
| 残差结构 | 普通 `x+f(x)` | **Hyper-Connections（Sinkhorn 双随机矩阵）** |
| 额外模块 | — | **engram（203 GB 查找表）**、MTP/投机头、视觉塔 |
| 每 token 专家字节 | ~1.9 GiB | **~5.93 GB** |

---

## 2. 权重清单（实测 48 个分片 header，不是猜的）

磁盘总量 **510.29 GB**，分类：

| 类别 | 体积 |
|---|---|
| experts（40 层 × 384 + shared） | 297.52 GB |
| **engram embed（2 张表）** | **202.76 GB** |
| attention / norm / embed / head / hc_* | 8.58 GB |
| vision | 0.82 GB |
| mtp | 0.61 GB |

dtype 分布：

| 类别 | dtype | 代表张量 |
|---|---|---|
| FP8_E4M3 + E8M0 **32×32 块** scale | attention `wq_a/wq_b/wkv/wo_b`、`ffn.shared_experts.*`、`indexer.wq_b`、`engram.wkv`、`mtp.main_proj` | `wkv.weight [512,5120]` / `wkv.scale [16,160]` |
| **FP4 packed int8 + E8M0 1×32 行 scale** | 仅 `ffn.experts.{i}.w1/w2/w3` | `w1.weight I8 [2304,2560]` / `w1.scale [2304,160]` |
| FP8_E4M3 + E8M0 **1×32 行块** scale | `engram.embed` | `embed.weight [384006168,256]` / `embed.scale [384006168,8]` |
| 裸 BF16（无 scale） | 所有 norm、`ffn.gate.weight`、`embed`/`head`、`compressor.*`、`indexer.wk/weights_proj`、vision/aligner | |
| F32 | `hc_*_{base,fn,scale}`（每层 6 个）、`attn_sink`、`ffn.gate.bias`/`bias_vl` | `attn_sink [64]` |

> 注意 160 的来由：FP4 走的是 MXFP4 的 **1×32 逐行块**（每行每 32 个元素一个 scale），FP8 走的是 **32×32 二维块**。两者不能混用。

---

## 3. 前向 Overview

```
input_ids [B,L]
  → embed [B,L,5120]
  → repeat → [B,L,4,5120]           # hc_mult=4，残差流是 4 份
  → layer 0..39:
      engram 写入（仅 layer 1、14 之前）        # 加性，不过 hc_pre/hc_post
      hc_mixes(x) → pre/post/comb              # RMS-scaled Linear + Sinkhorn
      hc_pre(x, pre_mix) → attn_norm → attention → hc_post
      hc_mixes(x) → hc_pre → ffn_norm → MoE → hc_post
  → hc_pre（用最后一层 FFN 的 ffn_pre）→ norm → head → logits [B,129280]
```

**没有 `x + f(x)`**：残差是 Sinkhorn 双随机矩阵混合，见 § 4.1。

---

## 4. 必须新增的算子（cwick 目前一个都没有）

### 4.1 Hyper-Connections（每层 6 个小张量却在热路径上）

每层 6 个张量：`hc_attn_{base[24], fn[24,20480], scale[3]}`、`hc_ffn_*` 同。

```c
// ffn/base/scale 生成 24 个数：对 flatten 后的 4×5120 做 RMS-scaled Linear
// ⚠️ 这里用的是 norm_eps=1e-20，不是 hc_eps=1e-6
x = flatten(h);                       // [4*5120]
rsqrt = 1/sqrt(mean(x^2) + 1e-20);
mixes = fn @ x * rsqrt;               // [24] fp32 GEMM
pre[j]  = sigmoid(mixes[j]      * scale[0] + base[j])      + 1e-6
post[j] = 2*sigmoid(mixes[4+j]  * scale[1] + base[4+j])
C[j][k] = mixes[8+j*4+k] * scale[2] + base[8+j*4+k]
// 行 softmax(+eps) → 列归一 → 再 19 轮（行归一 ÷(s+eps)、列归一 ÷(s+eps)）
```

混合公式按代码字面是 **`y = Cᵀ · residual + post ⊙ x_out`**（转置相乘）。
⚠️ **二义点**：参考实现 docstring 没写 `C` 还是 `Cᵀ`。必须数值验证后定稿——Sinkhorn 只保证每行每列和为 1，`C` 与 `Cᵀ` 数值不同。

### 4.2 FP4 反量化 + 混合精度 GEMM

```c
static const float FP4_E2M1[16] = {0,0.5,1,1.5,2,3,4,6, -0,-0.5,-1,-1.5,-2,-3,-4,-6};
// 一个 int8 字节里：**低 4 位是偶数下标**，高 4 位是奇数下标
fp4_get(row, k) = FP4_E2M1[(k & 1) ? (byte >> 4) : (byte & 0x0F)];
e8m0_to_f32(b)  = 2^(b - 127);        // b=0xFF 为 NaN

// out[m][n] = Σ_kb sa[m][kb]*sb[n][kb] * Σ_{k∈kb} fp4_get(n,k) * e4m3(a[m][k])
// FB: sa/sb 每 32 个 K 元素一对
```

这是每 token 的主成本（5.93 GB），也是 Metal 内核必须重写的部分——现有 GEMV 内核假定 FP8 + scale 布局。

### 4.3 Activation 量化（ue8m0 + round-to-power-of-2）

```c
amax = max(|x| in 32-block);  amax = max(amax, 1e-4);
e = ceil_log2(amax / 448.0f);           // 必须按位实现，快 approximative 版不行
s = 2^e;                                 // 精确 2 的幂
scale_byte = e + 127;
q = clamp(x / s, -448, 448) → e4m3
```

参考实现里 `act_quant` 多数是 `inplace`：量化后立即反量化写回（`x ← e4m3(x/s)*s`），结果仍是 bf16。这是**有损**的，必须逐算子对齐，否则累积误差会让输出漂掉。

### 4.4 Attention（大幅不同于现有）

逐步形状（S=5120，hidden；64 头，head_dim=512，rope 只作用最后 64 维）：

```
qr = wq_a(x)  [S→1280] → q_norm(1280, eps=1e-20) → wq_b → [32768] → view(64,512)
rope(q[..., -64:])
kv = wkv(x)   [S→512]  → kv_norm(512, 1e-20)      # num_key_value_heads=1，每个位置只有 1 个 512 维
kv_cat = concat(window_kv[128], compress_kv[n_comp])   # <-- 见下
o = sparse_attn(q, kv_cat, sink, topk_idxs, scale=1/sqrt(512))   → [64,512]
rope^-1(o[..., -64:])
o = view(8, 4096) → act_quant → einsum("bsgd,grd->bsgr", o, wo_a.view(8,1024,4096)) → [8,1024]
x = wo_b(flatten → 8192) → [5120]
```

关键点：

- **`wo_a` 是块对角的**（8 组 × [4096→1024]），不是普通 Linear；且参考实现把它反量化成 BF16 了。
- **`attn_sink` 是一个 value 恒为 0 的第 n+1 个虚拟 KV 槽**，只进 softmax 分母：`Σ = Σ_j exp(s_j-m) + exp(sink_h - m)`。不是 logit 偏置。
- **每 query 的 KV 条数固定 640**（128 window + 512 top-k），与上下文长度无关 → attention 成本 O(640)，上下文拉长也不涨。这是好消息。

压缩 KV 与索引器（config 里的 `compress_ratios` 才有 40 项；`inference/config.json` 里 43 项多出的 3 个是 MTP 的）：

```
layer 0,1    ratio 0  —— 完全不参与压缩 KV
layer 2..19  ratio 2  —— KV 条数 = T/2
layer 20..39 ratio 1  —— = T（每 token 一个 512 维 latent）
```

只有 `kv_source_layer_ids=[2,8,14,20]` **生成**压缩 KV，其余层读最近一次写好的：
2..7 用 layer2、8..13 用 layer8、14..19 用 layer14、20..39 用 layer20。

只有 `index_source_layer_ids=[2,8,14,20,24,28,32,36]` **重新跑 top-k**，其余层复用。
两级选择：layer 20 先按 8 个位置一块取 max 选出 ≤2048 块（覆盖 ≤16384 位置），
其余 21..39 层在候选外置 −inf 后取 **top-512 个压缩 position**（不是 block）。

每层两套 RoPE（**按层不同，容易踩坑**）：

- layer 0、1（ratio=0）：theta=10000，**关 YaRN**
- layer 2..39：theta=**160000**，**开 YaRN**（factor=16，original=65536）
  - 修正后的 dim 分界由 `64*ln(65536/(rot*2π))/(2*ln(160000))` 算出：low=15、high=25
  - `ramp = clamp((i-15)/10, 0, 1)`，`freqs *= 1 - 0.9375*ramp`
  - i≤15 不变，i≥25 频率 ÷16，中间线性过渡

压缩 latent 的位置取它那一组的第一个 token（prefill 步长 ratio，decode 取 `start_pos+1-ratio`）。

### 4.5 MoE 路由（与常见实现不同）

```
logits = x_fp32 @ gate.weight_fp32^T / 1.0        # gate 是 BF16 磁盘、代码升 fp32，稠密不量化
s      = sqrt(softplus(logits))
idx    = topk(s + bias, 6)                        # bias 只影响选择
w      = gather(s, idx)                           # 权重取未加 bias 的 s
w     /= (Σw + 1e-20)                             # norm_topk_prob，注意是 1e-20
w     *= 1.5                                      # routed_scaling_factor
```

⚠️ **二义点**：HF 主配置写了 `topk_method: noaux_tc`，但参考实现里**没有 `n_group`/`topk_group`**，就是全 384 纯 top-6。按参考实现写。

- 专家顺序：**w1=gate、w2=down、w3=up**（不是 gate/up/down）
- SwiGLU clamp：`up` 双向 clamp 到 ±10，`gate` 只 clamp 上界 ≤10（下界不 clamp）
- shared expert 权重恒为 1.0、不加路由系数，所有 token 都跑
- `bias_vl` 只在多模态输入时替换 `bias`（文本模式不需要）

### 4.6 engram（最大的工程风险）

**每个 token 查 48 行**（2 层 × 24 列），每行 256 B + 8 B scale = 264 B → **12.7 KB/token**，96 次随机小读。

24 列 = (2-gram, 3-gram, 4-gram) × 8 头；每张表被切成 24 个互不相交的**素数桶区间**
（素数从 15999999 往后找），layer1 之和 384006168、layer14 之和 384016682，与 config 完全吻合（这是对本算法的强验证）。

```c
// 每层的 4 个乘数由 numpy PCG64 生成：np.random.default_rng(10007 * layer_id)
// multiplier_bound = (2^63-1) // 99092 // 2 = 46539438283891，取 4 个后 *2+1
c[k] = blocked ? pad_id : compressed_cache[t-k];      // k=0..3，DEAD=-1，越界也 blocked
r = c[0]*mult[0];
h2 = (r ^ (c[1]*mult[1])) % prime[ngram2][head];      // 注意 XOR 用的是**未取模的 rolling**
h3 = (h2raw ^ (c[2]*mult[2])) % prime[ngram3][head];
h4 = (h3raw ^ (c[3]*mult[3])) % prime[ngram4][head];
row = h + offset[layer][col];
```

之后：embed(row) → 24×256=6144 维 → `wkv` → 前 4×5120 是 key、后 5120 是 value →

```
weight = q_weight * k_weight                          # 两者只以乘积出现，可预乘
rstd   = rsqrt(mean(h^2)+1e-20) * rsqrt(mean(key^2)+1e-20)
dot    = Σ(h * weight * key) * rstd / sqrt(5120)
gate   = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6)))   # signed sqrt 后 sigmoid
h     += gate * value                                   # 加性写入
```

**必须先从 Python 导出、写死进 C 的常量**：

| 常量 | 形状 | 来源 |
|---|---|---|
| `token_map`（compressed token map） | int32[129280] | 需要 HF normalizer（NFKC/NFD/StripAccents/Lowercase…），必须 Python 侧导出 |
| `multipliers` | int64[2][4] | numpy PCG64，**不能手算**，必须 Python 侧导出 |
| `primes` | int64[2][3][8] | 已算出（见 §8） |
| `offsets` | int64[2][24] | 已算出 |
| `freqs_cis` | complex64[65536][32] × 2 套 | theta=10000 无 YaRN / theta=160000 有 YaRN |

203 GB 的表**不能物化**（参考实现靠 TP=8 拆到 8×64 GB，这台机器没有）。必须自己做 **mmap + MADV_RANDOM**，并且强烈建议离线重打包成「264 B 连续记录」（把 weight 行和 scale 行交错），这样每 token 随机读从 96 次降到 48 次、页触碰量减半。

---

## 5. 可以先不管的部分

| 模块 | 能否跳过 | 代价 |
|---|---|---|
| `mtp.*`（3 层 nextn / dspark） | 推理不必需——参考实现的 `generate.py` 从不调用投机路径 | 省 0.61 GB，但**丢掉把它当投机草稿头的可能**（见 §6） |
| `vision.*` + `aligner.*` | 能，做纯文本模式 | 省 1.01 GB，且 `Gate.bias_vl` 变 None、engram 不做 mask |
| tokenizer chat template | 仓库里**没有** jinja，模板是 `encoding/encoding.py` 的 Python 逻辑 | 要在 C 侧重写，或 Python 预编码后喂 token id |

---

## 6. 性能预算（这是要不要开工的关键）

### 每 decode token 必须读的字节（batch=1）

```
routed experts:  6 专家/层 × 40 层 × 18,800,640 B  = 4.20 GiB
shared experts:  40 层 × 35,424,000 B              = 1.32 GiB
------------------------------------------------------------
专家小计                                            ≈ 5.93 GB
engram:  2 层 × 24 行 × 264 B                       =   12.7 KB
```

对比参照：GLM-5.3-Flash 每步约 1.9 GiB。**这个模型是它的约 2.9 倍。**

### 内存能缓存多少

128 GiB 扣掉常驻约 9.5 GB（attention 权重 5.0、indexer/compressor 0.5、embed 1.32、head 1.32、engram.wkv 0.31、KV cache 0.2）→

```
可用专家缓存 ≈ 118 GB。专家总量 297.52 GB → 命中率上限 ≈ 40%
即使理想缓存，每 token 仍要从盘读 ≥ 3.2 GB
```

### tok/s 上限

| 假设 | 结果 |
|---|---|
| SSD 随机读 5.4 GiB/s（本机实测上限），3.2 GB/token | **≈ 1.6 tok/s** |
| 按 GLM 实测的有效速率（避不开 IOPS/延迟）估算 | **≈ 0.7 tok/s** |
| 若每一步都走盘 | SSD 需持续 **13 GB/s**，本机不可能 |

**结论：专家全走盘时，裸 decode 只能到约 1.6 tok/s。**

> **2026-10-25 修正**：上表把「每 token 4.5 GiB 专家字节」全算成磁盘读，而实际有
> 82 GiB 的 pin 缓存兜住其中 ~86%。真正走 SSD 的只有 ~0.6 GB/token，所以瓶颈
> 不是 SSD 带宽而是内核效率 + 读延迟的隐藏程度。实测 decode 已达 **2.3–2.7 tok/s**，
> 这个速率不再被判死。见 §10.10。

### 唯一能提吞吐的路：自带的 MTP/投机头

checkpoint 里带了 3 层 nextn（`mtp.0/1/2`）+ `markov_head` + `confidence_head`，
参考实现注释明确说投机解码循环「out of scope」。但对我们来说它恰恰是提吞吐的手段：

```
有效 tok/s = (每步平均接受 token 数) / 单步延迟
```

单步延迟不变（还是要把 MoE 过一遍），但一次前向产出 2–4 个 token → 有效吞吐 ×2~4。
若单步 0.7 tok/s，接受率 3/4 时能到约 2 tok/s。**这是 models 自带的能力，应当作为 M6 之后的主要方向。**

---

## 7. 里程碑（每步都可验证，不写完不许进下一步）

| # | 内容 | 验收标准 |
|---|---|---|
| **M0** | 数值 oracle | 装 torch（当前系统无 torch/numpy）或写纯 C 单算子检验器；能对 40 层里任意 1 层跑参考实现的前 3 维并导出中间激活 |
| **M1** | 反量化测定层 | 独立的 `tools/` 小程序：读 FP4/I8 + E8M0、FP8 + E8M0，与 oracle 逐位对齐（含 `__builtin` NEON 版本） |
| **M2** | dtype 契约 | `src/model.c` 认识 `deepseek_v41`；因未提供 enable 的部分而明确报错（不是现在的 `missing/invalid dtype` 误导） |
| **M3** | attention 单算子 | 用 oracle 校 Hyper-Connections、Sinkhorn、二级 top-k、块对角 wo_a、双套 RoPE；与参考实现误差 < 1e-4 |
| **M4** | MoE | 路由/SwiGLU clamp、FP4 专家 GEMM；先用 CPU 后端跑通单层 |
| **M5** | engram | mmap(MADV_RANDOM) + 离线重打包工具（264 B 交错记录）；校验 hash 算出来的行号与 oracle 完全一致——**这是最容易出事的一步** |
| **M6** | 全链路 | 40 层 + 文本模式；对比 `generate.py` 输出 token 序列完全一致（temperature=0） |
| **M7** | 性能 Metal | FP4 GEMV Metal 内核、跨层预取、专家 LRU（118 GB） |
| **M8** | MTP 投机 | 复用 `mtp.*`，把有效 tok/s 向 2 推 |

---

## 8. 已算出的常量（素数与 offset 已用参考实现复算，与 config 完全吻合）

layer 1（`engram_layer_ids[0]`，和 384006168）：

```
ngram2 素数  16000057 16000079 16000081 16000097 16000121 16000129 16000133 16000183
ngram3 素数  16000189 16000207 16000211 16000253 16000277 16000289 16000307 16000321
ngram4 素数  16000339 16000381 16000393 16000399 16000403 16000409 16000447 16000463
offsets      0 16000057 32000136 48000217 64000314 80000435 96000564 112000697
             128000880 144001069 160001276 176001487 192001740 208002017 224002306 240002613
             256002934 272003273 288003654 304004047 320004446 336004849 352005258 368005705
```

layer 14（和 384016682）：

```
ngram2 素数  16000477 16000487 16000499 16000507 16000511 16000573 16000609 16000627
ngram3 素数  16000667 16000669 16000693 16000697 16000711 16000729 16000759 16000769
ngram4 素数  16000781 16000799 16000813 16000819 16000841 16000877 16000879 16000889
offsets      0 16000477 32000964 48001463 64001970 80002481 96003054 112003663
             128004290 144004957 160005626 176006319 192007016 208007727 224008456 240009215
             256009984 272010765 288011564 304012377 320013196 336014037 352014914 368015793
```

仍需 Python 导出：`token_map[129280]`、`multipliers[2][4]`、`pad_id`。

---

## 9. 风险清单

1. **没有 oracle 就不能动手**：装 torch 才能跑参考实现；当前系统既无 torch 也无 numpy（没办法验证任何一行 C）。
2. ⚠️ **两处参考实现二义**：Hyper-Connections 是 `C·R` 还是 `Cᵀ·R`；`noaux_tc` 是否真没有分组 top-k。必须数值确定；定错就是沉默的错误输出。
3. **engram 是 静默错误的高发区**：一个 hash 错位就是数值全错，只会让输出变得"像但不对"。行号必须与 oracle 逐比较。
4. **act_quant 的 inplace 量化-反量化**：容易以为无关紧要而省掉，实际是全局位数累积误差的来源。
5. **裸 decode 有硬上限**：不靠投机解码时，这个模型的 decode 速率被每步字节数钉住，在本机上物理上过不去。这个结论现在就要说，不要等指标出来再被打脸。

---

## 10. 已落地与实测（截至 2026-10-23）

上面 §9 的第 1 条风险已经不成立：**没有 torch 也能验证**——验证器用纯 C 重写，
与被测代码并行实现同一份数学，靠引擎自己打的 hex-float trace 逐位比对。

### 10.1 验证工具（全部纯 C，`make cwick` 之外的独立二进制）

| 工具 | 作用 | 结果 |
|---|---|---|
| `tools/ds_verify` | 48 分片 / 510.29 GB 全开，量化条目数、真实权重 GEMV | 256 e4m3 + 16 fp4 + 256 e8m0 正确；FP8 GEMV worst rel 1.4e-07，FP4 6e-08 |
| `tools/ds_optest` | 逐算子不变量（rope 逆变换、Sinkhorn 双随机、量化误差上界、fp4 网格） | 13 项全过 |
| `tools/ds_toktest` | 分词器已知 id + round-trip | 30 项全过 |
| `tools/ds_layertest` | 整块重算 vs 引擎 trace，40 层 × 3 位置 | **6039 checks，0 failure，worst rel/rms = 0（逐位一致）** |
| `tools/ds_prompttest` | prompt 编码 vs 参考 README 里的字节串 | 11 项全过 |

`ds_layertest` 用 `--tokens 1000 2000 3000` 覆盖了 prefill 的 p=0/1/2，`--all` 覆盖全部
40 层；比对的量包括残差流 4 份拷贝、hc_pre/post/comb、压缩 latent、indexer 选中集、
attn 权重和、每个专家的 gate/up/SwiGLU 乘积/w2 输出、路由权重与选中集。

### 10.2 逐位对齐过程中修掉的真实引擎缺陷

1. `ds_bf16_round` 只加了舍入偏置、**没有把低位清零**，等于从未真正舍入到 bf16 网格。
2. `gate_temp` 从未解析（config.json 里没有这个键，参考默认 1.0）→ 除零 →
   router 分数全 `inf` → `sqrt(softplus(inf))` 与 `0` 相乘 → MoE 输出全 NaN。
3. top-k 权重归一化用「乘倒数」而参考用「除法」，差 1 ULP，足以把部分元素推过 bf16 边界。
4. `sparse_attn` 的返回值是 bf16 tensor，逆旋转之前必须整体舍入一次。
5. 查询向量的 rope 段在 rope 之后要 bf16 舍入（参考是原地写回 bf16 tensor）。

### 10.3 验证器自己的两个缺陷（症状很像引擎错）

* 第二个位置起 `shared_topk` / `shared_cand` 共用一个缓冲区，而引擎按位置存——L=21
  用到的是最后一个位置的选择结果。改成 `[ntok]` 后消失。
* 索引键缓存按「索引源序号（0..7）」索引，但只按「KV 源个数（4）」分配，第 5 个索引源
  （L=28）直接踩坏堆。参考语义是索引键跟随 KV 源，改成 `src_slot` 后消失。

### 10.4 prompt 编码

`src/ds_prompt.c` 是参考 `encoding/encoding.py` 文本路径的 C 移植：BOS、
system / 会话中 system、user、assistant（reasoning + 内容 + DSML 工具调用）、
`latest_reminder`、`tool` 折进 user 的 `<tool_result>`、工具 schema 块、
数值推理预算前缀、`drop_thinking` 解析、`task` 快捷指令 token。
图像块按设计拒绝（视觉塔未加载），不静默丢弃。

`tools/ds_prompttest` 的用例逐条抄自 `encoding/README.md` 打印出来的字节串，因此通过即
等价于「与参考格式字节一致」，而不是「看起来差不多」。

**已接线**：`server.c` 的 `run_generation` 按 `model_type` 前缀分派——`deepseek*` 走
`ds_chat_encode()`，其余走 GLM 模板。两个细节：

* 请求校验阶段用 `cwick_json_write()` 原样保留 OpenAI 的 `messages` / `tools` 数组，
  因为 DeepSeek 模板要读 `reasoning_content`、`task`、`wo_eos` 这些 `cwick_chat_msg`
  压缩结构里没有的字段；
* 客户端没写 effort 时传「未设定」而不是 GLM 的“极致”——两个 checkpoint 的默认预算
  不同（DeepSeek 是 75，GLM 是 Max）。实测渲染结果：

```
<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)

<｜User｜>Say hi in one word.<｜Assistant｜><think>
```

### 10.5 性能现状

当前（2026-10-23 晚，128 GiB 机器，40 token 列表 prompt，专家缓存 94.3% 命中）：

| 指标 | 数值 |
|---|---|
| decode | **1.083 s/token**（0.923 tok/s） |
| prefill | **1.374 tok/s** |
| 每 token 专家字节 | ≈4.5 GiB（6/384 × 40 层 × 18.8 MiB），其中约 1.1 GiB 走 SSD |

decode 单步分桶（`DS_TIMING=1`，总 1083 ms）：expert matmul 43–52%、
attention 26–31%（内含 qkv+rope 7–9%、wo 12–14%、indexer 2%、sparse attn 2%）。

瓶颈排序（按实测）：

1. **expert matmul 占了近一半，但不是算力问题**。8.49 GMAC 的内核在
   `tools/ds_gemvbench` 上能跑到 54.8 GMAC/s（≈155 ms），实测却要 ~500 ms，
   差出来的 350 ms 是权重流本身的等待：每 token 4.5 GiB 从 88.2 GiB 缓存与
   SSD 混合喂进来，等效只有 ~9 GB/s。**降字节是唯一真正的杠杆**，不是换内核。
2. **wo 140 ms / 3.02 GMAC** = 21.6 GMAC/s，同样低于内核上限（31.9–34.3）。
   wo_a 逐 group 读 1.34 GB 稠密 fp8，走的是同一条 ~9 GB/s 的路。
3. **prefill 仍受专家字节支配**：chunk 内按专家分组（见 10.7）已把 89 token 的
   2362 个专家访问压到 942 个不同专家，但 n 越大重复度越低，收益会递减。
4. Metal 仍然帮不上：autotune 实测 `[2048,4096]` FP8 GEMV 是 Metal 514 µs vs
   CPU 池 330 µs，明确选 CPU。FP4 还没有 Metal 内核。

> 长跑与短跑不能比：短跑（<30 步）量到的是冷缓存数字。同一配置本机
> 漂移可达 ~20%，任何 A/B 必须同场交替跑。


### 10.6 服务端诊断暴露的四个缺陷（2026-10-23 下午）

从一份 263 行的 `cwick-serve.log` 与 `/v1/metrics` 出发定位，四个都已修：

1. **prompt 模板串了**（P0）。切到 DeepSeek 后仍在用 GLM 的
   `[gMASK]<sop>...Reasoning Effort: Max`，tokenizer 却已经是 DeepSeek 的
   128k 词表——模型收到它没训过的序列。见 10.4。
2. **prefill 指标冻结**（P0）。`g_pf_*` 只有 GLM 引擎路径写，DeepSeek 从不更新，
   dashboard 全程显示 `prefill_layer 0/45`（45 还是 GLM 的层数）。现在
   `ds_engine_generate` 在 chunk 边界报 token 数、`ds_model_forward` 在层循环里报
   层号，并把本 chunk 内已完成的比例折进 `done`，否则小于 `DS_CHUNK` 的 prompt
   会让 `done` 一直是 0。切模型时 `cwick_metrics_set_model_info` 也补上了
   `g_pf_layers` 的重置（原来只有 `cwick_metrics_init` 会做）。
3. **GEMV 单线程**（P0）。`ds_fp4_gemv` / `ds_fp8_gemv` 是标量双循环，`sample(1)`
   拍到 12 个 worker 全在 `__psynch_cvwait`，只有 1 个核在算。改成按**输出行**切片
   走 `cwick_pool_for`——行与行之间没有归约，切分不改变任何累加顺序，所以
   `--all` 回归仍是 6039 checks / worst rel-rms = 0。
4. **专家缓存没接**（P1）。`ds_engine.c` 里没有任何缓存逻辑，`expert_cache_cap_bytes`
   恒为 0，每个 token 从 SSD 重读全部专家。改为复用 GLM 的 `cwick_stream`
   （LRU + ghost 准入 + F_NOCACHE）：`ds_model_open` 建流，`ds_moe` 用
   `cwick_stream_expert_put_read` 取权重，冷 key 被 ghost 拒绝时回落到映射分片——
   两条路径指向同一批字节，因此正确性不依赖命中与否。

还有一个值得记住的运维坑：cwick 的监听 socket 用了 `SO_REUSEPORT`，一个没退干净的
旧 `cwick serve` 会和新的同时绑 8001，内核在两者之间做哈希分流。表现是 `/v1/config`
和日志对不上（一个说 context 4096、一个说 131072），`/v1/models` 随机显示不同的
`loaded` 状态。排查时先 `pgrep -x cwick` 确认只有一个进程。

### 10.7 GEMV 内核与并行注意力（2026-10-23 晚）

三个改动，全部保持 `--all` 回归 6039 checks / worst rel-rms = 0。

**一、每 4 行一起算的 GEMV。** 原来的内层是
`part += w.lo*x; part += w.hi*x+1;`——一条 32 次乘加的**串行依赖链**。fp32 加法
一个周期发射、四个周期后才出结果，所以单行每个周期有 3/4 在等自己。改成一次算
四行：每行仍然是完全相同的串行求和顺序，只是四条独立的链并排跑。行之间没有
归约，所以是逐位不变的。

| 形状 | 单行 | 四行 | 倍数 |
|---|---|---|---|
| FP4 `[5120,2304]`（专家 gate/up/down） | 2.55 GMAC/s | **54.78** | 21× |
| FP4 `[2048,4096]` | 2.55 | **51.68** | 20× |
| FP8 `[1024,4096]`（wo_a 每 group） | 2.43 | **31.88** | 13× |
| FP8 `[5120,8192]`（wo_b） | 2.43 | **34.33** | 14× |

（"单行"是 `tools/ds_gemvbench` 内建的参考实现；`DS_GEMV_ROWS=1` 可以把本内核
切回单行版本用于对比。）

**二、激活只反量化一次。** 原来每算一行都要把 K 个激活值查一遍表，`af` 提到
GEMV 外层后每行只剩乘加。`DS_ACT_MAX` 从 5120 提到 8192：wo_b 的
K = o_groups × o_lora_rank = 8192 之前正好越界，掉进了串行兜底，一度占 decode
30%。现在超限会退到堆缓冲——**快内核是唯一内核**。

**三、稀疏 attention 按 head 上池。** 每个 head 的 QK 点积又是 512 次串行的
依赖链，64 个 head 完全独立（各自只写自己那 512 个输出、只读共享 KV），所以按
head 切给线程池不改变任何算术。为此 `ds_sparse_attn` 需要**每 head 一份
scores**：原来的 `m->scores` 是 indexer 的 scratch，共享它就不可能并行。

**踩到的坑（值得记住）：** `cwick_pool_for` 内部有 `CWICK_POOL_MIN_WORK = 512`
的阈值，`n < 512` 直接整段在调用者线程上跑完。head 数是 64，所以第一次改完
一点没变快（sparse attn 仍是 11%）。改用 `cwick_pool_for_items`（阈值 1，专门
给"少量大任务"）后立刻降到 2%。**打池之前先确认任务数过不过阈值。**

同场 A/B（40 token 列表 prompt，专家缓存命中率一致）：

| 版本 | prefill | decode |
|---|---|---|
| 本轮之前 | 38.3 s / 1.148 tok/s | 50.9 s / **0.767 tok/s** |
| 四行 GEMV | — | 0.812 tok/s |
| + 并行稀疏 attention | 32.0 s / **1.374 tok/s** | 42.3 s / **0.923 tok/s** |

合计 decode 1.20×、prefill 1.20×。继续往上走的空间不在算力：内核已经比实测
路径快 2.7–3×，差的是每 token 4.5 GiB 权重流的供给带宽。

### 10.8 服务侧：指标语义与日志噪声（2026-10-24 凌晨）

从 `cwick serve` 的实时日志和 `/v1/metrics` 里读出来的四个问题，都是"看起来
正常、其实读不出东西"：

1. **前缀查询只统计命中的那次。** `cwick_metrics_prefix_query` 原来只在
   `P > 0`（查到了）时调用，于是 `queries == hits` 恒成立，命中率面板要么是
   0/0 要么正好 100%，永远显示不出缓存失效。现在每次查询都计数，miss 也算。
2. **`cwick:memory_ceiling_bytes` 恒为 0。** 内存上限只在 GLM 侧算出来却没写进
   注册表；现在 `cwick_mem_plan_log`（GLM）和 `ds_model_open`（DeepSeek）都会
   发布，并且日志会说明这个上限是来自 `CWICK_EXPERT_CACHE_GIB` 还是 `host/4`
   默认值。
3. **专家驻留指标伪装成 vLLM 的前缀缓存。** `vllm:gpu_prefix_cache_*` /
   `vllm:gpu_cache_usage_perc` 报的其实是"专家权重块的驻留"，被任何 vLLM
   看板当成提示词前缀命中率读（实测 84.3%）。已改名为
   `cwick:expert_cache_queries_total` / `cwick:expert_cache_hits_total` /
   `cwick:expert_cache_usage_perc`，不再占用 vLLM 的命名空间。
4. **300 行日志里 185 行是 metrics 轮询。** `/metrics` `/health` 现在按 trace
   级记录，且不再额外打一遍请求行。DeepSeek 的汇总行补齐了
   `cache=`/`streamed=`/`mem=` 三个字段，与 GLM 一侧一致。

### 10.9 DeepSeek 前缀缓存（2026-10-24 凌晨）

长 prompt 重复的 agent 场景下，TTL 里绝大部分是重算同一个 system prompt。
DeepSeek 侧的前缀快照现在可用，且**回放与它替代的那段 prefill 逐位一致**。

**要存什么。** 跨 token 的状态只有六样：每层的滑窗 KV `win_kv`、压缩 KV
`cmp_kv`、压缩器未成组的池化中间量 `cmp_state`/`cmp_score`、indexer 的键
`idx_k`、engram 回看的 `tok_cache`，以及位置本身。残差流和 per-token scratch
是 token 内部的，下一层会重新算出来，所以不存（与 GLM 侧 mHC 状态同样的理由）。

**怎么存。** GLM 的 `pfx` 文件格式是"若干定长数组"，DeepSeek 的数组完全不同，
所以快照作为一个**不透明 blob** 走：header 里原本是 padding 的那个 word 现在
是 `kind`（0 = 原布局，1 = blob），header 字节数不变，因此旧快照照样能读。
`ds_state_apply` 会校验 magic/版本/几何（`n_layers`/`window`/`head_dim`/
`n_src`/`idx_head_dim`/`ratio`），不匹配就当 miss 让调用者整段 prefill，而不是
崩。服务侧在 prefill 结束时、以及长 prompt 每 `CWICK_PFX_BLOCK` 个 token 各
抓一次（DeepSeek 有自己的 prefill 循环，所以钩子挂在 `ds_engine` 上）。

**过程中修掉的两个真实缺陷**，都是"算得很快但算错了"：

1. `c->max_ratio` 从来没有被赋值（一直是 0）。压缩器的部分分组缓冲按它开，
   快照和 reset 也按它走，于是两边都**静默跳过两个数组**，而模型照旧往里写。
   这个缺陷同时让"分两段 prefill"看起来不等价于"整段 prefill"——其实分块本身
   没问题，是部分分组状态没被带过去。
2. **`cmp_kv` / `idx_k` 是 `[source][max_seq_len][dim]`，快照却当成一整段连续
   内存拷。** 拷 `n_kv_source × pos` 行等于只把 source 0 的前
   `n_kv_source × pos` 行搬走，其余三个 source 全是 0。字节数一点没变，所以
   只有跟"被它替代的那次 prefill"逐位比对才会暴露。症状是第 8 层起（第一个
   读 source 1 的层）注意力权重开始偏，rel-rms 0.58。

**怎么验的（纯 C）。** `tools/ds_run` 加了三个模式，`DS_TRACE` 加了 `FWDPOS`
锚点行（trace 里的位置是"本次调用内的下标"，跨调用就对不上了）：

| 模式 | 干什么 |
|---|---|
| `--split-test <cut>` | 整段 prefill vs 分两段，比 logits |
| `--state-diff <cut>` | 两种 prefill 的快照逐段比，报第一个不同的字节 |
| `--prefix-test <cut>` | 前段 prefill → 快照 → rewind → 恢复 → 后段，比 logits |
| `--dump / --snap-dump <cut>` | 只打 trace，交给 `/tmp/ds_trdiff` 逐个张量比 |

`ds_trdiff` 按 (tag, layer, 全局 position) 配对两个 trace，报第一个（以及全部）
不同的元素。定位上面第 2 个缺陷时它直接指出：**第 10 层、第 2 个位置起**，
`PROBE_ATTW` 有 3/129 个元素不同——往上追就是"第 8 层开始读的 source 1 是零"。

**实测结果：**

| 用例 | 结果 |
|---|---|
| 6 token，cut = 1/2/3/4/5 | 129280 个 logit **全等**，rel-rms = 0 |
| 32 token，cut = 8/16/31 | 全等，rel-rms = 0 |
| 150 token，cut = 130（跨过 128 滑窗折返） | 全等，rel-rms = 0，快照 11.85 MB |
| `ds_layertest --all --tokens 1000 2000 3000` | 6039 checks / 0 failure / rel-rms 0 |
| `make test` | PASS |

快照体积随 context 线性增长：131072 context 时约 1.35 GB（win_kv 10.5 MB +
cmp_kv 1.07 GB + idx_k 268 MB），所以 `--prefix-cache-gib` 在 DeepSeek 一侧要
按"几条快照"而不是"多少个 token"来估。


### 10.10 吞吐攻坚：从 0.86 到 2.7 tok/s（2026-10-25）

验收线：`--backend metal` 下走完整 API 链路（服务端 + 流式）时 decode 输出 ≥2 tok/s，
且不剪枝、不降精度。下面每一条都是**同场 A/B**，60 token 长跑取稳态
（`DS_TIMING=1` + 访问日志的 `decode=…ms/… steps … tok/s`）。

#### 单步耗时的演进

| 步骤 | ms/step | tok/s | 当时的主导项 |
|---|---|---|---|
| 起点 | 1050 | 0.86 | head 37–40%、fp4 专家 26–36% |
| head 并行（12 线程按输出行切） | 571 | 1.45 | fp4 专家 43–47% |
| fp4 专家 GEMV 上 NEON | 530 | 1.48 | 专家 fetch 25–34% |
| 稠密 fp8 权重展开成 bf16 副本 | 550 | 1.66 | 专家 fetch 37–52% |
| 预取读者 1 → 4 | 404 | 2.47 | — |
| 预取读者 6（默认） | **382** | **2.65** | expert matmul 33–36%、wo 15–16% |

四条改动的细节：

1. **head（129280×5120 bf16）原本是整条链路最贵的单点**：448 ms，占一步的
   40%，而 12 个核对它只用了 1 个。按输出行切给 `cwick_pool_for`，再用
   `vshll_n_u16(v,16)` 把 bf16 直接当成 fp32 的高 16 位（bf16 的定义就是它），
   四路累加打断依赖链：**448 → 11.1 ms**（`make bench` 实测 119.2 GB/s）。
2. **fp4 专家 GEMV 的 NEON 化**：16 个 e2m1 值在 fp16 下低字节全为 0，所以
   只留一张**高字节**表，用 `vqtbl1q_u8` 一次 gather 拿到位型，再 `vshll_n_u8`
   升到 fp16 转 fp32。激活预先拆成偶数位/奇数位两份，避免在内层循环里移位。
   `tools/ds_gemvbench --gemv fp4` 实测 **0.10 ms/call、58.5 GB/s**，改造前的
   标量版由 `--gemv compare` 的 `fp4 old` 一行复现（28.3 GB/s）。
   注意这个工具**只能同场连续 A/B**：同一份代码间隔十分钟分别量到
   0.075 / 0.111 / 0.171 ms，那是热降频不是回归。
3. **稠密权重在加载时展开成 bf16 副本**（`ds_model.dense16`，约 11 GiB）。
   fp8 GEMV 的瓶颈是每元素一次查表 + 缩放，bf16 是纯 `vld1q` + `vfmaq`，还能
   用上 4 路独立累加：**38.5 → 119.2 GB/s**。展开是**精确**的——e4m3 只有
   8 位有效位，bf16 的 8 位尾数装得下；`--gemv check` 的最坏相对误差是
   **7.6e-7（1 ulp）**，来源只有 4 路求和的顺序，乘积本身逐位相同。
   `CWICK_DS_EXACT=1` 关掉这条路径回到参考顺序（标量内核 + 不展开），
   代价实测 **1.72 vs 2.24 tok/s（约 −23%）**，用于逐层 oracle 校验。
4. **预取读者 1 → 6**（`CWICK_PF_READERS`，1..8）。关键证据：命中率与
   `streamed` 字节数在 1/4/6 三个档位下**完全不变**（`84.0%(29028/5532)`、
   `45.86 GiB`），所以提速全部来自把 SSD 延迟藏起来，而不是多读了数据。
   这也是它与 `CWICK_PREFETCH=1`（投机读，2.7× 字节、纯亏）的本质区别。

#### chunked MoE 的专家读改成 arena 分批

`n>1` 的 prefill 走 `ds_moe_chunk`：先算整块的路由，再把并集里的专家每个只
读一次。问题是它用的是**阻塞式** `ds_expert_fetch`，一个专家一个专家地读——
实测 2.5 GB/s，因为单条 6 MiB 的读是**延迟**受限的，不是带宽受限
（同一设备 4–8 个在飞时能到 9 GB/s）。

改成与 decode 路径相同的「预取 → 取用 → 计算 → pin」序列，并按一个预取 zone
的容量（6 专家 × 3 矩阵 = 18 个矩阵）滚动分批：一批的 18 个读全部发出去之后
才开始第一个乘加，剩下的读就在这段算术之下完成。

| 批宽（`CWICK_DS_PF_BATCH`） | prefill（17 token） |
|---|---|
| 1（＝改造前的顺序） | 11.75 s |
| 2 | 9.09 s |
| 4 | 8.13 s |
| 6（默认） | **8.00 s** |

收益在 4–6 处饱和——正好是 arena 一个 zone 的宽度，再宽也装不下。输出是
**逐位不变**的：批宽 1/3/6 三个构建产出同一个 md5，因为每个
(position, slot) 仍然是一次独立的 GEMV 写进自己的 contribution，最后按 slot
顺序求和，分批只改变了「读发生的时间」。`make test` PASS。

顺带修了一处诊断失真：`cache=` 这行原来以「通过 ghost 准入门的 miss」为分母
（`put_read` 在拒绝冷 key 时直接返回、不计数），所以会显示 68.6% 这种好看但
不可比的数；改走 `expert_cached` 后每次查询都计数，同一进程同一请求读作
`8.5%(494/5299)`——**分子没变，分母变诚实了**。跨版本比命中率时要注意这一点。

#### 全链路 API 实测（`sh ./start.sh deepSeek-v4.1-flash-1017`，`--backend metal`）

服务端 8001，`/v1/chat/completions`（流式），`max_tokens` 见 gen 列，数字取自
访问日志的 `prefill=… decode=…ms/… steps … tok/s`：

| req | prompt | gen | prefill | decode | 端到端 |
|---|---|---|---|---|---|
| 2（进程首个请求，冷缓存） | 36 | 120 | 20.96 s | 2.294 tok/s | 1.65 tok/s |
| 3 | 36 | 120 | 15.68 s | 2.535 tok/s | 1.91 tok/s |
| 4 | 36 | 120 | 14.74 s | **2.599 tok/s** | 1.98 tok/s |
| 5 | 36 | 120 | 16.53 s | **2.724 tok/s** | 1.99 tok/s |
| 6 | 42 | 400 | 17.97 s | **2.438 tok/s** | **2.20 tok/s** |

改造前同一请求（req 7，36→120）：prefill 20.09 s、decode 2.568 tok/s、端到端 1.80。
所以 prefill 这一项 **20.1 → 15.7 s（−22%）**，decode 不变（2.5–2.7）。

**怎么读这张表。** decode 是「输出速率」，也就是流式客户端看到的速度：
**每一条请求都 ≥2.29**，稳态 2.5–2.7。端到端还多算了 prefill，而短 prompt 的
prefill 很贵：36 token 的并集约 167 个不同专家/层，要读约 **60 GiB**——按 token
摊是 0.43 s/token，比一个 decode step 还慢。这是 384 选 6 的必然结果（大 chunk
才能摊薄：512 token 的 chunk 只读 90 个矩阵/token，36 token 要读 557 个）。
因此**短 prompt + 短回答**的端到端会被 TTFT 拖到 2 以下，而 **回答 ≥200 token
时端到端已经 ≥2.2**。

还剩约 28% 的 prefill 时间是「一批读完之后、下一批读之前」的暴露延迟
（`DS_TIMING=1` 的 `expert fetch` 桶）。要消掉它必须让第 b+1 批的读在第 b 批
计算时就在飞，而 `cwick_stream_prefetch_begin` 的语义是「先排空旧 zone 再翻页、
翻页时丢弃旧 zone 未领取的任务」——做跨批流水线要改线程生命周期与任务所有权，
风险大于收益，本轮不做。

### 10.11 单步的字节账与 GPU 分流判决（2026-10-26）

上一节把 decode 推到 382 ms/step。本节先把 **453 ms 到底花在哪** 算清楚（同机
同 checkpoint，60 token prompt、24 new，`DS_TIMING=1`，GPU 关），再判定「把一部分
GEMV 交给 GPU」这条路当前值不值。

#### 分相（23 个 decode step 平均，wall 453.6 ms）

| 桶 | 占比 | ms | 说明 |
|---|---|---|---|
| expert matmul | 28.2% | 128 | 18 个 fp4 矩阵/层 × 40 层 = 4.2 GB |
| expert fetch | 20.6% | 93 | SSD 读 1.83 GB（命中率 48.7%） |
| wo | 13.4% | 61 | wo_a 8 组 + wo_b |
| qkv+rope | 9.1% | 41 | |
| **expert pin** | **7.8%** | **35** | 见下，原先并进 fetch |
| attention | 7.7% | 35 | |
| shared+sum | 5.8% | 26 | |
| sparse attn | 3.0% | 14 | |
| head | 2.5% | 11 | |
| routing | 1.0% | 4.5 | |
| indexer | 0.5% | 2 | |

`pin` 是**这一轮新拆出来的桶**：`ds_expert_pin` 把 arena 里的专家 blob 再 memcpy
一份进缓存槽位，两遍走同样的字节。以前它并进 `expert fetch`，于是「磁盘慢」被
高估、「带宽被重复计费」被掩盖。实测 35 ms（1.83 GB → 约 52 GB/s，单线程 memcpy）。

#### 每步真正搬了多少字节

按 `dsgpu-verify` 打印的 (n, k) 与每步调用次数统计（bf16 稠密，2 字节/元素）：

| 形状 | 次/步 | GB/步 |
|---|---|---|
| wo_b 5120×8192 | 40 | 3.36 |
| indexer 32768×1280 | 40 | 3.36 |
| wo_a 1024×4096 ×8 组 | 320 | 2.68 |
| shared w1/w3 2304×5120 | 80 | 1.89 |
| shared w2 5120×2304 | 40 | 0.94 |
| engram 25600×6144 | 2 | 0.63 |
| q 1280×5120 | 40 | 0.52 |
| kv 512×5120 + 其余 | — | 0.32 |
| **稠密合计** | ~600 次 | **13.7** |
| **专家（fp4）合计** | 720 次 | **4.2** |

**一步 17.9 GB，453 ms ⇒ 39.5 GB/s 聚合带宽。** 两个结论直接掉出来：

1. **稠密是大头，不是专家**（13.7 GB vs 4.2 GB），但它只占 176 ms ⇒ 78 GB/s，
   已经接近这些内核的实测上限（`--gemv shapes`：100–270 GB/s，热缓冲）。
2. **专家只占 24% 的字节却吃掉 56% 的时间**：fetch 93 + pin 35 + matmul 128。
   fp4 解码是 ALU 密集的（每 32 元素 32 次查表 + 16 次乘加），matmul 在 4.2 GB
   上只有 33 GB/s，而同一内核在热缓冲上是 48 GB/s。

#### GPU 分流：内核正确，接口错了

新增的两个内核（`kernel_cwick_ds_fp4_gemv` / `kernel_cwick_ds_bf16_gemv`）按 CPU
求和树逐位复现，按输出行切分，**验证为零误差**：`--gemv gpucheck` 在真实形状
`[2304, 5120]` 上 `worst rel 0.000e+00`，第二次调用仍为 0；`CWICK_DS_GPU_VERIFY=1`
内建的运行时校验（重算 GPU 那部分行并逐行对比）在整段前向里 fp4 一次都没报错。

但吞吐说明这条接口不可能赢：

| 形状 | 只用 CPU | 分流 0.72 | 只走 GPU |
|---|---|---|---|
| q 1280×5120 | **0.050 ms** | 0.129 | 0.115 |
| wo_a 1024×4096 | **0.084** | 0.106 | 0.043 |
| wo_b 5120×8192 | **0.673** | 0.940 | 1.037 |
| shared w2 5120×2304 | **0.136** | 0.448 | 0.494 |
| indexer 32768×1280 | **0.663** | 1.012 | 1.217 |
| engram 25600×6144 | 2.666 | **2.180** | 2.266 |

（`make bench && tools/ds_gemvbench --gemv shapes`，CPU 池 12 核。）

单个 command buffer 只有 ~80 GB/s，而 12 个矩阵共用一个 buffer 时是 **198 GB/s**
（`gpucheck`）；CPU 池在同一批字节上是 100–270 GB/s。原因就是一次 submit+wait
固定 ~0.2 ms，与矩阵大小无关——而一个 decode step 有 ~1400 次 GEMV。端到端后果：
**763 ms/step（对 461 全 CPU）**；加一道「小于 16 MB 不给 GPU」的闸门后回到
**518 ms/step**，仍然是亏的（`CWICK_DS_GPU_MIN_MB`）。

所以 **`CWICK_DS_GPU` 默认值改回 0**，内核与校验工具保留：bf16 内核在成批提交下
是 198 GB/s、逐位正确，缺的是「一次提交多个矩阵」的接口，不是内核。

#### 一个值得记下来的坑

第一次端到端 A/B 得到乱码输出 + 0.58 tok/s + 命中率腰斩，一度被判成内核算错。
真因是 **`./cwick` 是旧二进制**（改完 `ds_fp4_gemv` 的 NEON 分支后没有重新
`make cwick`），而 `tools/ds_gemvbench` 是后编的，于是「bench 正确、引擎乱码」。
现在的分辨方法：`CWICK_DS_GPU_VERIFY=1` 会把「GPU 那部分行」重算一遍逐行比对，
0 报错就说明不是内核；输出乱码时**先 `make cwick`**。

#### 到 5 tok/s 还差什么

200 ms/step 意味着 17.9 GB 要按 **89 GB/s** 跑完。今天的 39.5 GB/s 与这个数字之间
只有三条路，且都不是调参：

1. ~~**消灭 pin 的第二遍拷贝**（35 ms）~~ **已试，收益为零，已回滚。** 做法是让
   ghost 门在**读之前**决定常驻，`cwick_stream_expert_reserve` 拿到槽位，
   `cwick_stream_prefetch_submit_into` 把 pread 直接落进去，`take` 拿到 owned=1，
   于是 `pin` 整段消失。实现完整跑通、输出逐字不变，但**墙钟 452.1 ms 对 453.6**：
   `pin` 桶的 35 ms 里有 17 ms 本来就和 SSD 等待重叠（它等的是同一批 18 个读，
   CPU 在等的空档里做拷贝），剩下的 18 ms 挪进了 `fetch` 和 `matmul`。结论：
   **这一项不是瓶颈所在，不值得为它把读路径改复杂**，代码已还原。
2. **成批提交 GPU**：CPU 与 GPU 并行流式读同一块统一内存，才有机会把聚合带宽推到
   100 GB/s 以上。前提是把 wo_a 的 8 组、shared 的 w1/w3 这类天然独立的矩阵放进
   一个 command buffer，并且让等待延后到依赖点。这是目前唯一还没被证否的大杠杆。
3. ~~**提高专家命中率**~~ **已查，容量见底，没有余量。** `--verbose` 的
   `expert pin set: 79.43 GiB used of 81.86 GiB cap` 说明缓存已经 **97% 满**，
   所以 48.7% 不是准入闸门（ghost 门）抠得太狠，而是 81.86 GiB 对 271 GB 专家
   权重的直接结果——按容量比例只有 30%，48.7% 已经是「路由集中」带来的超额部分。
   要让命中率再涨只能加内存或改权重精度，两条都被约束挡住。

第 2 条若成立，才有机会进 250–280 ms/step（3.5–4 tok/s）；即便如此，5 tok/s
（200 ms/step）要求稠密的 13.7 GB 与专家的 4.2 GB 在两个引擎上近乎完全并行，
且 SSD 那 1.83 GB 被完全藏住——这是本机当前结构的上限，不是调参能到的。

### 10.12 硬件带宽天花板与 5 tok/s 的裁决（2026-10-26）

上一节列了三条路。本节把**这三条全部跑完**，并补一条最关键的测量：这台机器到底
能给多少内存带宽。结论是 5 tok/s 在本机结构上不可达，且这不是实现质量问题。

#### 带宽是 123 GB/s，不是 400

用 12 线程、每线程独立 64 位流式求和的裸读循环测（`/tmp/bwtest2` 同类）：

| 工作集 | 4 线程 | 8 线程 | 12 线程 | 24 线程 | 32 线程 |
|---|---|---|---|---|---|
| 1 GiB | 118.0 | 121.1 | — | — | — |
| 4 GiB | 123.5 | 123.9 | 123.2 | 122.7 | 122.3 |
| 16 GiB | — | — | 123.6 | — | — |

**4 线程就打满，加到 32 线程一动不动；1/4/16 GiB 完全一样。** 提高 MLP（8 路累加、
每轮 32 次 8 字节载入）也不改变。这是 DRAM 控制器的顶，不是循环写坏了。

两个直接推论：

1. **稠密已经贴着天花板了**：`wo` 桶实跑 99 GB/s = 峰值的 80%，`--gemv shapes` 里
   wo_b/indexer 是 125-127 GB/s。**稠密再优化内核没有意义，只能减少字节。**
2. 一步 17.9 GB，即便纯流式读也要 **145 ms**；5 tok/s（200 ms）要求 **89.5 GB/s =
   峰值的 73% 持续带宽**，而且要同时扛住 SSD 那一路。今天 453 ms = **39.5 GB/s =
   32%**。

#### 减少稠密字节的路：原生 fp8，被 checkpoint 否决

稠密权重在 checkpoint 里是 **FP8 e4m3**，`bind_q` 加载时展开成 bf16 副本
（`ds_model.dense16`，11 GiB）。走原生 fp8 能一举三得：稠密字节减半（13.7→6.85 GB）、
常驻内存省 5.5 GiB（可还给专家缓存）、`ds_gemv_q` 本来就已在 `w16==NULL` 时走
`ds_fp8_gemv`。

**否决原因（实测）**：给 `bind_q` 加了扫描后，40 层 × 每层 8 个 fp8 张量 =
**330 个张量，全部含有例外字节**。快解码 `((b & 0x7F) << 7) + 0x2000 | ((b & 0x80) << 8)`
（已手工验算 1.0 / 0.5 / −1.0 / 448 均正确）对正规值精确，但两个角落不是仿射的：

| 字节 | 真值 | 快解码给出 | 误差 |
|---|---|---|---|
| e=0, m=4（次正规） | 4·2⁻⁹ = 0.0078 | 2⁻⁷·1.5 = 0.0117 | **+50%** |
| e=0, m=1 | 2⁻⁹ = 0.00195 | 2⁻⁷·1.125 = 0.00879 | +350% |
| e=15, m=7 | NaN | 480 | — |

这不是舍入差异，是真错，会直接打破 `ds_layertest` 的逐位一致。补例外要两次
`vqtbl1q_u8` + zip + 符号合并 + bsl 混合，约 15 条/16 值，把解码从 6 条/16 值抬到
29 条；**bf16 只要 6 条/16 值**（`vshll_n_u8(v,8)` 再 `vshll_n_u16(.,16)`，位移而已）。
折算成每字节指令数：fp8 ≈ 1.5，bf16 ≈ 0.48。fp8 会立刻从带宽受限转成发射受限，
即使字节减半也追不平。判定：**放弃**。

#### 其余四条，全部零收益

| 改动 | 结果 |
|---|---|
| 消灭 pin 第二遍拷贝（§10.11） | 453.6 → 452.1 ms，拷贝本就与 SSD 等待重叠 |
| GPU 按行拆分 | 763 ms/step（对 461），单矩阵 submit+wait 固定 ~0.2 ms |
| `CWICK_PF_READERS` 2 / 6 / 12 | matmul 桶 26.3% / 26.5% / 26.6%，无差异 |
| fp4 内核展开 2 块 + `prfm` | gate/up 65.5→75.6 GB/s，但 down 79.1→**53.5**；端到端 493 ms（对 453）。已回退 |
| `CWICK_EXPERT_CACHE_GIB=95`（对自动 82） | 2.067 对 2.055 tok/s，命中 33.2% 对 32.5%，mem 95.78 GiB——**RAM 已近满，没有空间** |

#### 裁决

200 ms/step 需要 89.5 GB/s。天花板 123 GB/s。二者之间只剩 27% 的余量，而今天
连一半都没用到（39.5），缺口分布是：稠密已在 80%（没空间），专家 matmul 在 36 GB/s
（该内核冷数据 bench 是 65 GB/s，这 ~55 ms 是唯一真实存在的余量），pin 与 fetch
分别被 SSD 延迟盖住。把唯一那条余量吃干净最好约 300 ms（3.3 tok/s）。

**5 tok/s 在本机不可达**，除非：减少每步字节（全精度下 271 GB 专家装不进 128 GB，
且 fp8 稠密已证伪），或者换一台内存带宽更高的机器。
