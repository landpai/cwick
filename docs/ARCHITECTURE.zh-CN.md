# 架构 —— 一个 C 进程，三个 checkpoint，一台 Mac

<p align="center"><a href="ARCHITECTURE.md">English</a> · <a href="ARCHITECTURE.zh-CN.md">简体中文</a></p>

CWICK 在一台 Apple Silicon Mac 上运行**完整、未剪枝**的 checkpoint：GLM-5.3-flash
（≈306 GiB）、DeepSeek-V4.1-Flash-1017（≈475 GiB）、MiniMax-M3.1-Flash（≈233 GiB）——
一个二进制、没有辅助进程、没有转换步骤。本文档把每个 C 组件映射到它所实现的那部分模型，
并说明哪些部分是共用的、哪些只属于某一个后端。

## 单进程设计

`main.c` 负责 CLI，`server.c` 负责 HTTP 接口层，`engine.c` 负责请求路径
（渲染 → prefill → decode → 流式），`hub.c` 负责模型库，`webui.c` 提供控制台。
不存在内部 HTTP 调用，也不存在按 token 启动的辅助进程：进入 `/v1/chat/completions`
的请求不离开地址空间就摸到了权重。

## 模型契约

`model.c` 读取惰性 JSON，校验每一个性能关键的维度，并在打开任何权重数据之前就 fail
closed。三条质量守卫对三个 checkpoint 都成立：

- `tie_word_embeddings` 必须为 false；
- `num_experts_per_token` 完全按模型的定义使用（top-k 绝不缩减），并且每个 expert
  都必须存在于语料之中；
- 缺一个分片就是拒绝启动，而不是悄悄跑一个小一号的模型；
- dtype 与块大小按后端分别校验，不做假设：三个 checkpoint 并不共用同一套量化方案。

## 后端

家族在加载时解析一次，之后请求路径对常驻的那个后端调用同样的两个入口
（`prefill` / `decode`）。

| 家族 | 文件 | 权重 | attention |
|---|---|---|---|
| GLM-5.3-flash | `spine.c` `attention.c` `moe.c` | FP8-E4M3 + 每 128×128 的 F32 `weight_scale_inv` | MLA + DeltaNet（KDA）、DSA 池 |
| DeepSeek-V4.1 | `ds_model.c` `ds_quant.c` `ds_ops.c` `ds_engine.c` `ds_prompt.c` | 专家 FP4-E2M1 + E8M0，attention FP8 + block scale | 压缩 MLA + indexer |
| MiniMax-M3.1 | `mm_model.c` `mm_engine.c` `mm_prompt.c`（+ `mm.h`） | NVFP4 专家、FP8 稠密 | bf16 GQA，4 个 KV 头 |

`pfx.c`、`stream.c`、`sample.c`、`tokenizer.c`、`tools.c` 以及服务端是三者共用的；
有差别的只有加载器、反量化、prompt 模板与内核。

接第四个 checkpoint = 新加载器 + 新反量化 + 新 prompt 模板 + 新内核，**不是改一个
config**——这也是这份名单停在三个的原因。

## 权重 spine（GLM 路径）

`spine.c` 以 mmap 方式映射 checkpoint：

```
config.json
resident/tensors/<name>.bin        BF16 稠密 spine（行优先 [out,in]）
resident/experts/expert-L-E.bin     每层/每个 expert 一个规范原始文件
resident-int8/tensors/<name>.i8     可选的 int8 数据（+ .s fp16 缩放）
```

逐层稠密张量包括：input norm、attention-residual norm/proj、post-attention norm、
MLP-residual norm/proj，以及 KDA 集合（`A_log`、`dt_bias`、q/k/v conv1d、`o_norm`、
q/k/v/g/fa/fb/b 投影、`o_proj`）或 MLA 集合（`q_a/q_b`、`kv_a/kv_b`、各个 norm、
`g_proj`、`o_proj`），然后是稠密 MLP（第 0 层）或 MoE spine（gate + correction bias +
routed down/up/norm + shared gate/up/down）。expert 语料在打开时做完整性校验——
缺少任何一个 expert 都会 fail closed（never-pruned 规则）。

另两个后端直接映射发布原样的 safetensors 分片，并保持同一种切分：每个 token 都要读的
常驻，routed 专家流式。

## 计算

`tensor.c` 实现：

- 经 Accelerate（`cblas_sgemm`）的 fp32 GEMM，BF16 / row-int8 权重会反量化到一个
  执行用的 arena 中。
- RMSNorm、softmax 和 SiTU，运算顺序固定：
  `situ(g,u) = tanh(g/beta)*beta*sigmoid(g) * tanh(u/linear_beta)*linear_beta`，
  `rms_norm = w * x / sqrt(mean(x^2)+eps)`。
- 用于 routed-expert 路径的 NEON MXFP4 expert GEMV kernel：打包的 e2m1 nibble +
  e8m0 缩放，一个 256 行的 scale LUT 折叠进 fp32 的 top-16 半区，用 `vzip` 交织成
  元素顺序，再用 `vqtbl1q_u8` + `vshll` 重建。单元测试保留一份标量参考实现，对两条
  路径做交叉校验。

在此之上，每个后端各有自己的部分：

- **Metal**（`metal.c`、`metal_shader.c`、`metal_objc.m`）：零拷贝 arena。专家数据块与
  批式投影直接绑定到一次 dispatch，不做拷贝；换模型时用 `munmap` 释放 arena。
  GLM 与 DeepSeek 用它；MiniMax 在这些形状下实测更慢，留在 NEON 线程池上。
- **DeepSeek**：fp4 分组 GEMM（一次 4 个激活、每个 2 个向量累加器）、fp8 稠密 GEMM
  按 4 行 × 4 激活分块，以及一张本地的 e4m3→f32 表——那个转换函数里的 `build_luts`
  分支会主宰热循环。
- **MiniMax**：同样形状的 kernel 跑在 NVFP4（块 scale 是**除数**）与 fp8（per-row
  per-32 的 `U8(e8m0)` scale）上；一切按块做，绝不按 token 做——302 token 的 prompt
  拆成 34,428 次单行 GEMV 实测 67 GMAC/s，改成按块 GEMM 之后是 228 GMAC/s。

## Attention

`attention.c` 实现 GLM 的 kernel；另两个后端各有一份自己的文件，接口相同。

- **MLA**：潜变量 query（`q_a` + norm + `q_b`）、潜变量 KV（`kv_a_with_mqa` + norm +
  `kv_b`）、在 rope 切片上施加 RoPE、在 KV cache 上做完整 attention
  （`scale = 1/sqrt(qk_nope+qk_rope)`）、输出门控（`sigmoid(g_proj)`），最后 `o_proj`。
  GQA 把 q-head 映射到 kv-head。
- **KDA**：短卷积（4 个 tap、SiLU、三个 per-channel 状态）、L2 归一化的 query/key
  （query 额外除以 `/sqrt(head_dim)`）、门控衰减
  `gate_lower_bound * sigmoid(exp(A_log) * (gate_raw + dt_bias))`、按 beta 缩放的
  delta 规则、per-head RMSNorm 配 `o_norm`、输出门控、`o_proj`。
- **DeepSeek**：压缩 KV，加一个 indexer 决定稀疏 attention 读哪些位置；indexer 池是
  逐请求状态，靠快照而不是重算。
- **MiniMax**：bf16 GQA、4 个 KV 头，RoPE 只作用在每个头的前 64 维，norm 权重是
  `(1 + w)`（gemma 风格）——照搬别的后端会悄悄错一个尺度。

prefill 的 attention 是同一套数学跑在一整块位置上；把内层循环向量化，是 MiniMax 上
398 ms/token → 30 ms/token 的来源，也是长 prompt 变成带宽受限而非算力受限的原因。

## MoE

`moe.c` 实现精确的 router 与 expert 数学：

```
logits = gate(hidden)                    # [E]
scores = sigmoid(logits)
choice = scores + gate.e_score_correction_bias
top-k ids                                # num_experts_per_token，绝不缩减
weights = scores[ids] / (sum + 1e-20)    # moe_renormalize, routed_scaling_factor
routed_in = routed_down(hidden)          # [RH]
per expert: gate=w1(x), up=w3(x), h=situ(gate,up), y=w2(h)，再按 weight 缩放
out = routed_up(rms_norm(sum, routed_norm)) + shared_down(situ(shared_gate,
      shared_up))
```

各个 expert 按权威的 top-k 顺序被加载与组合。prefill 时一整块位置选中的专家会被并成
每个专家一次 GEMM；decode 一步则是每个专家一次 GEMV。这个差别——块 GEMM 对逐 token
GEMV——最高值 5 倍，也是 prefill 看起来慢时第一个该查的地方。

## 层 tape 与 decode

`engine.c` 实现 GLM 路径的逐 token forward tape；`ds_engine.c` 与 `mm_engine.c` 是
同一个 tape，换成各自的 norm、attention 与量化 MLP。

```
# 每层
attn_in = residual_combine(hidden, anchors, self_attn_res_norm * res_proj)
if layer % attn_res_block_size == 0: anchors.append(hidden); prefix = attn_out
else prefix = hidden + attn_out
normalized = rms_norm(attn_in, input_layernorm)
attn_out = attention(normalized)                    # MLA 或 KDA
mixed = residual_combine(prefix, anchors, mlp_res_norm * res_proj)
post = rms_norm(mixed, post_attention_layernorm)
mlp_out = dense_mlp(post) or moe(post)
hidden = prefix + mlp_out

# 尾部
mixed = residual_combine(hidden, anchors, output_attn_res_norm * res_proj)
logits = lm_head(rms_norm(mixed, final_norm))
```

这个 residual-anchor 库是逐 token 的状态（block residual）：每个 token 独立累积自己在
边界层的 pre-attention 行。decode 从最后一行的 logits 采样（`t<=0` 走 argmax，这是
一次运行可复现的原因）；prefill 按块推进——GLM 按自己的激活预算定块大小，DeepSeek 与
MiniMax 用 `MM_CHUNK` / `DS_CHUNK` = 1024——因为专家读取是按块计的成本，一个尾巴块
和满块一样贵。

## Server

`server.c` 实现 OpenAI 接口与它的两个同族：`/v1/models`、`/v1/models/switch`、
`/v1/completions`、`/v1/chat/completions`、`/v1/responses`、`/v1/messages`
（以及 `/models` 别名）、SSE 流式（chat 是 begin chunk → content deltas → tool-call deltas →
finish frame → 可选的 usage frame → `data: [DONE]`；另两个是具名事件、没有终止哨兵），
以及对任何无法被精确履行的请求 fail closed 返回 400。两个额外端点都是适配器，不是第二条
流水线：Responses 的 `input` items 与 Anthropic 的 content blocks 都被翻译成 chat 校验器
产出的同一种会话，只有信封不同（`responses_body` / `resp_stream_*`、
`messages_body` / `msg_stream_*`）。同一时间只做一次生成，按到达顺序：槽位是
单的，因为两个答案会互相腰斩带宽。其余一切都是并发的——`/v1/metrics`、`/v1/models`、
`/v1/config`、日志流、`/v1/hub/*`、控制台——这正是"能看着一次 30 分钟的 prefill"成立
的前提。换模型或关闭时返回 503 带 `Retry-After`；排队中断连的客户端会在烧掉一次
prefill 之前被移出队列。详见 [SERVER.zh-CN.md](SERVER.zh-CN.md)。

`metrics.c` 用 vLLM 兼容命名发布同一份状态；`auth.c` 管单用户会话与具名 API key
（只存哈希）；`logcap.c` 保存 `/v1/logs` 与 `/v1/logs/stream` 背后的环形缓冲。

可观测性与生命周期是契约的组成部分，而不是事后补上的东西：

- `log.c` —— 一个行原子的 logger 写到 stdout，带 `--verbose`（逐请求、逐步细节）与
  `--trace`（prompt、输出、tool 调用）两档级别，每行都带墙上时间，请求在飞时还带请求号；
- `main.c` —— 不带 `SA_RESTART` 的 `sigaction` 处理器加一条 self-pipe，这样一次
  Ctrl+C 会打断 `poll()`/生成，而不是等到下一个请求；
- `mem.c` —— 加载时的内存计划；`stream.c` —— 可裁剪的 expert pin 集合与让它待在
  计划之下的冷却阀门。

## 决策端点

`jev.c` 是常驻模型的第二种用法：不让它写，而是渲染一段决策 prompt、prefill，然后读
答案位置上选项字母的分布。不生成任何 token，所以整段延迟都是 prefill。

三件事，而且每一件都是按家族而不是全局：

- **语法** —— GLM 与 DeepSeek 用思考块已关闭的那一轮（`glm-chat-closed`、
  `deepseek-chat-closed`），MiniMax 用它的普通对话轮；
- **读出** —— 字母表对着常驻 tokenizer 解析（三个自带 checkpoint 都是 26/26、
  带前导空格），所以切模型也就切了这个接口；
- **续算** —— 一个请求里的问题共享 `system` + `<state>` 那一轮，端点只 prefill 一次
  token 级最长公共前缀，每个问题通过 `cwick_engine_prefill_logits_at()` 从那里续算。
  这个续算只在跨 token 状态按位置寻址的模型上精确：MiniMax 是，GLM 与 DeepSeek 不是
  （`|dp|` 0.41 / 0.56，答案会变），所以那两个后端每个问题付一次完整 prefill。
  实测在 [DECISIONS.zh-CN.md](DECISIONS.zh-CN.md) §44。

## 控制台与模型库

`webui.c` 提供编译进二进制里的单页控制台（`src/webui_embed.c` 由 `tools/embed_web`
从 `web/` 生成，勿手改），所以 `--web` 不需要磁盘上的静态资源，也不需要第二个进程。
五个 tab：走 `/v1/chat/completions` 的对话、应用市场、模型库、本进程的 stdout/stderr、
设置——设置里有实时指标、趋势图、Prometheus 序列浏览器、常驻模型切换与只读配置。
所有请求同源，没有把控制台指向外部端点的选项。

`hub.c` 支撑模型库：`./models/<owner>/<name>/`、服务端任务表（暂停/继续/重试/取消）、
用 `Range` 续传的 `.incomplete` 文件、全程 64 位尺寸、只读的目录检索。五个端点都在
生成槽位之前应答，所以一个 400 GiB 的下载不会挡住聊天请求。

## 工具（沿用 checkpoint 协议，而非自创协议）

`tools.c` 把 OpenAI 的 tool 声明渲染成 GLM-5.3 自己训练时所用的那个 block——逐字抄自
`zai-org/GLM-5.3-Flash/chat_template.jinja`（`<tools>` + 每个函数一行 JSON，外加
`<tool_call><arg_key>/<arg_value>` 指令）——再把模型的回答解析回来。DeepSeek 与
MiniMax 各有自己的渲染器（`ds_prompt.c` / `mm_prompt.c`）。decode 从不施加约束，
因此格式错误的输出会被拒绝，而不是被转发。

## 本框架有意不包含的东西

- **不做批式 GPU prefill。** prefill 是按块的 CPU 线程池 GEMM（以及 Metal arena 在该
  赢的部分）；把多个请求并进一次前向没有实现，在只服务一个用户的机器上那是拿延迟换
  吞吐。
- **不做并行 tokenization。** 一个 tokenizer，一次一个请求。
- **默认路径上没有草稿模型投机。** 块验证存在（`MM_SPEC`），相对它自己的块算术是精确
  的，但 n=8 时块 GEMM 会翻转近似并列的 argmax，而且 8 个位置一块要读约 2 倍于单步的
  专家字节——见 [DECISIONS.zh-CN.md](DECISIONS.zh-CN.md) §28。默认关闭。

已经就位的是完整的数学、完整的存储契约、完整的 server 契约，以及 Apple Silicon 的
计算路径（Accelerate + NEON + Metal）。验证套件在用到了每一种层的微型合成模型上证明
整条流水线。
