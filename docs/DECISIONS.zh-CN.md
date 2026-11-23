# 决策记录——试过什么、留下了什么、什么不能再试

<p align="center"><a href="DECISIONS.md">English</a> · <a href="DECISIONS.zh-CN.md">简体中文</a></p>

这里的每一条都是本代码库做过的决策，并附上支撑它的测量结果。数字来自这台机器
（M3 Max，128 GiB，GLM-5.3-Flash），除非标注为 *external*。

## 1. expert prefetch 保持 opt-in（默认关闭）

`CWICK_PREFETCH=1` 会从后台线程发起对下一层的推测性读取。实测，80 个 decode
step，缓存预热，83 GiB expert cache：

| arm | ms/step | stream | gemv | attn | hit rate |
|---|---|---|---|---|---|
| prefetch off | **590** | 271.6 | 151.1 | 134.4 | 68.5% |
| `CWICK_PREFETCH=1` | **2111** | 490.8 | 712.3 | 284.2 | 68.6% |

prefetch arm 读取的字节数是 2.7 倍（647.6 对 236.8 GiB，95 396 对 30 315 次
读取），hit-rate 却毫无收益，而且 MoE 阶段因 cache/arena thrash 而爆炸。
（更早的一次运行测得 714 -> 1873 ms；机器在两次会话之间会漂移 ~20%，所以 A/B
必须在同一会话内跑。）

机制上：瓶颈不是 SSD 带宽，而是 **核心**。demand 路径用十二个 worker 的
fork-join 读取整层，而 fork-join 的 GEMV 由最慢的参与者决定节奏。一个默认 QoS
的推测线程会在 kernel 执行中途抢占这些 worker，于是这一层反而比根本不发起读取
还慢。因此 `pf_reader` 运行在 `QOS_CLASS_BACKGROUND`（`stream.c`）——这是调度器
唯一不会让它抢占 `USER_INITIATED` 工作的类别——而上面的 A/B 是*在*这项改动*之后*
做的。机制现在是对的，prefetch 仍然亏，因为错的是字节，不是调度。

prefetch 闸门（逐层 EMA recall，阈值 0.10，warmup 16 个样本）会在预测很差的那
些层上抑制推测性读取。它无法修复一个 recall 结构性偏低的 predictor，只能停止为
它付费。

## 2. MXFP4 expert pack 被否决

把 expert 字节数减半可以把 stream 从 271.6 降到 ~140 ms、gemv 从 151 降到 ~75 ms
（预计 413 ms/step，2.4 tok/s），但它要付一次性的 **290 GiB 重新打包**，而且 4-bit
权重不是 bit-exact。2026-10-20 否决。打包契约保留在 `tensor.c` 中
（`mxfp4_gemv_rows`，即标量参考实现以及 `test_mxfp4_spec`），这样这个选项日后
可以重新评估而不必重新推导：

```
packed:  U8 [rows, cols/2]   two e2m1 nibbles per byte, LOW nibble = even column
scales:  U8 [rows, cols/32]  e8m0, scale = 2^(s - 127)
nibble c -> {0, .5, 1, 1.5, 2, 3, 4, 6}[c & 7], negated when c & 8
```

一个 16 字节的 group 是 32 个值：16 个偶数列 + 16 个奇数列，共用一个 scale。

## 3. 缓存尺寸停在低于天花板的位置

83 GiB 的 expert cache（80% 预算允许的上限）实测只比 64 GiB 好 ~10%
（714 对 798 ms/step），却把 footprint 推到 102.40 GiB 中的 102.04。继续填满只会
更差而不是更好：超过 ~45 GiB 之后，常驻集已经大到 macOS 无法再让热页保持未压缩，
一个"已缓存"的 expert 开始在每次读取时付解压成本。`start.sh` 跑 64 GiB，留出
余量。

## 4. prefill 在 CPU 上跑 MoE

`batch windows: metal 0, ->cpu 190` 不是 bug。prefill window 会把数据取到 malloc
出来的 staging blob 里，而 Metal batch 只接受 no-copy 的 arena/cache 视图，所以
prefill group 按设计走 CPU fork-join 路径。强制走 GPU batch
（`CWICK_PREFILL_METAL=1`）能赢下 23% 的计算，却让 SSD 慢约一倍——净亏。decode
不受影响：它的视图在构造上就是 arena-backed 的。

## 5. `--warmup` 在端口打开之前替人付掉冷启动税

2026-10-20 实测，全新 serve 之后的第一个请求，253-token prompt，生成 10 个
token，两个 arm 的输出文本完全相同：

| first request | decode ms/step | decode tok/s | hit rate | prefill |
|---|---|---|---|---|
| cold serve | 1304 | 0.77 | 13.2% | 96.9 s |
| `--warmup 32` | **939 (-28%)** | **1.07** | **39.6%** | 98.9 s |

warmup 本身为 32 个 token 花掉 74.8 s——这*就是*那笔冷启动税，只是从第一个客户端
请求上挪走了。253-token 的 prompt 有 90% 是 prefill，所以收益体现在 decode 而不是
端到端 wall time 上；短 prompt（以及 prefix-cache 命中）会把平衡推向 decode。
warmup 路径从不触碰 prefix cache，失败会被记录并忽略。

## 6. 不要再试

以下每一条都因结构性原因失败，重测就是浪费：

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

## 7. 怎样重新测量 prefetch

一个进程，两次运行，只差一个环境变量：

```sh
cd cwick
CWICK_PROFILE=1 CWICK_EXPERT_CACHE_GIB=83 \
  ./cwick run --root ./zai-org/GLM-5.3-Flash --backend metal \
  --prompt "$(python3 -c 'print("You are a careful assistant. "*40)')" \
  --max-new 80
```

读 `profile: N decode steps, X ms/step wall` 这一行，以及 `stream` / `gemv` /
`attn` 的分解。**用 ~80 个 step**：10-step 的运行是 17.3% hit rate
（614 ms/step stream），那是冷启动数字，不是稳态。baseline 是 590 ms/step；
一个赢不了它的 prefetch arm 就不该被默认打开，无论机制调得多好。

## 8. decoding 从 checkpoint 自己的分布中抽样

`generation_config.json` 是训练契约的一半：发布的 checkpoint 带的是
`do_sample=true, temperature 1.0, top_p 0.95, top_k 20`。用朴素的 argmax 服务它们
就是在服务 temperature 0，而一个用采样训练出来的模型的 argmax 就是重复循环——
在 serve log 里表现为：一个请求把它的整个 5282-token 预算花在
`Owner: Owner: Owner: ...` 上；以及五个请求中有三个答案是空的，因为每个 token
都进了 thinking。

所以 `src/sample.c` 负责挑选 token，engine 每一步都向它要：

| request | what runs |
|---|---|
| `temperature: 0` | argmax（随附的 web app 发的就是这个，它要的也是这个） |
| anything else | top-k（硬过滤），然后在幸存者上做 top-p，再抽一次 |
| `presence_penalty` / `frequency_penalty` | 先从原始 logits 中减去 |

请求中省略的字段保留 checkpoint 自己的值，所以一个只发 `temperature` 的客户端
不会被静默地当成 temperature 0。一个这些字段都不发的请求会拿到 checkpoint 的值；
一个没有 `generation_config.json` 的 checkpoint 拿到 greedy。`CWICK_GREEDY=1`
强制 greedy。每个请求都会记录它是按什么配置服务的
（`sampling=... (generation_config.json)` 或 `(greedy default)`），于是"答案自己
重复了"可以和"答案是以 t=1.0 抽出来的"并排诊断。

成本，150144 路的词表，每个 decode step 抽一次：

| config | ms/step |
|---|---|
| greedy | 0.17 |
| t=1.00 top_p=0.95 top_k=20 | **0.10** |
| same + presence 0.6 / frequency 0.4 | 0.13 |
| t=1.00 top_p=0.95, `top_k: 0` forced | 1.45 |

有两件事让这些数字不落在 step 上。正 temperature 保持排序不变，所以 top-k 窗口
是在原始 logits 上取出的，temperature 只作用于 K 个幸存者——没有全词表乘法，没有
全词表 exp。而当客户端确实强制 `top_k: 0` 时，nucleus 可能比 4096-token 的扫描
窗口更宽（在上面的 benchmark 里就是这样），而一次 150144 路的排序要花 ~10 ms——
大半个 step——所以边界来自 logit 范围的直方图：一趟扫描，切点精确到一个 bucket
的宽度之内。

这*没有*修复的：采样让重复循环变罕见，而不是不可能——一个 250-token 的故事在六次
运行中仍有一次陷入循环（`喷喷喷喷...`）。要堵住它需要 loop detector 和 thinking
budget，那是另外的决策。

## 9. 一个响应要么是合法 UTF-8，要么就不是响应

detokenizer 追加的是原始 token 字节，而一个 byte-fallback token 可能是半个字符：
一次 greedy 运行把孤立的 `0xA8` 放进了答案，而每个把 body 当作 UTF-8 读的客户端
——JSON 的定义就是如此——都无法解析整个响应。`cwick_json_stringify` 现在会校验
每一个序列，并对最大不合法子部分替换为 U+FFFD（Unicode Table 3-8），于是文档
总能被解析回来。

流式输出在这方面不需要改动：`utf8_incomplete_tail` 本来就会把半个字符留到下一个
token 补全它，所以一个字符永远不会被拆到两个 delta 里，替换也永远不会破坏它。

## 10. 重复循环结束这一轮，而不是花掉预算

采样（§8）让这个吸引子变罕见。兜在它下面的网是一个 loop guard，它监视一次运行
*产出*的 token，并在某个固定块持续重复时停止生成。

判定是按 period 做的：`lmatch[p]` 统计尾部有多少个位置以 period p 重复
（`token[i] == token[i-p]`），当这个计数同时覆盖 `lmin_occ` 个完整块和
`lmin_run` 个 token 时触发——period-64 的块出现 3 次就是 192 个浪费的 token，
所以那里由 run 下限主导；而 period 1 触发靠的是连续 16 个相同 token。第二个条件
正是让普通结构不会误触的原因：一条 markdown 分隔线、一个省略号或"1. 2. 3."
编号会重复几次然后停下，而没有答案会故意吐出 16 个相同的 token。

两个容易搞错的细节：

- **prompt 不能喂给 guard。** `cwick_sampler_accept` 把一个产出的 token 同时计入
  penalties 和 guard；`cwick_sampler_prime` 把一个 prompt 中的 token 只计入
  penalties。用 `accept` 来 prime prompt 会把 prompt 的尾部喂给 guard，而一个以
  重复块结尾的 prompt（few-shot 模板、一张表）会在答案开始之前就把它停掉。vLLM
  和 HF 对整个可见文本施加 penalty，所以 priming 的 penalty 那一半是必需的——
  只有 guard 那一半对 prompt 视而不见。
- **闭合循环的那个 token 是框定，不是内容。** 单条路径把它留在 `gen` 里（它是
  生成文本的一部分，客户端看得见），但 batch 路径会把 `gen[i]` 减一，好让一个
  batch 序列报告的 `n_tokens` 与没有 guard 时相同。

成本是每个 token 每个 period 一次比较——64 次比较对 ~15 ms 的 decode step——历史
是 sampler 内部一个 256-token 的环形缓冲，所以没有分配，也没有对任何东西的逐步
遍历。阈值按每次生成可调（`CWICK_LOOP_DETECT=0` 关闭，
`CWICK_LOOP_MAX_PERIOD` / `CWICK_LOOP_MIN_RUN` / `CWICK_LOOP_MIN_OCC` 调其余部分；
period 旋钮只能把编译期上限 64 往下调，因为环形缓冲是按它确定大小的）。

在 wire 上这次停止就是一个普通的 `"stop"`——OpenAI 没有 loop finish reason——
access log 里带 `finish=loop`，engine 的 summary 行里带 period 和 run length。
由 11 个单元测试覆盖（16 对 17 个相同 token 的边界、period 3、latching、普通
结构、5000 个伪随机 token、prompt priming、禁用开关，以及 period 上限）。

## 11. loop guard 停止重复；loop breaker 必须打破它

对线上 agent 流量做的一次健康检查显示，13 个请求中有 7 个以 `finish=loop` 结束。
逐请求日志把形状说得很清楚：guard 正确触发了（一个 period-4 的块，16 个尾部
token），但那时模型已经烧掉了 450-570 个 thinking token，客户端收到零内容，重试
又要再付一整轮 decode。停止循环不等于逃出循环——sampler 每一步都还在从同一个
吸引子里抽，直到 guard 杀掉这一轮。

于是 sampler 在 guard 下面一级多了一个 breaker。loop guard 本来就跟踪 `lmatch[p]`，
即尾部有多少个产出的 token 以 period p 精确重复。当这个计数达到 `max(2*p, 8)`
（上限 12）时，pick() 会在 penalties、argmax、top_k 和 nucleus 看到它之前，把延续
该模式的那个 token——也就是历史环中往前 p 个位置的那个——的 logit 设为
-INFINITY，从而禁用它。下游的一切看到的只是一个被 mask 的 logit；-INFINITY 能
经受 penalty 减法，而 exp() 把它映射为恰好零质量。模型必须说点别的；如果它漂回
吸引子，禁用会重新武装，而 guard（16+ 个尾部 token）仍作为禁用压不住的一切的
最后防线。

这个上限不是装饰：禁用会造就更长的循环。一个 period-1 吸引子在 8 时被禁用之后，
这段运行会安定成"九个 token + 那个逃脱者"——一个 period-10 的模式，它的比例阈值
（2*p = 20）恰好会在 guard 触发的位置武装，于是 guard 杀掉了一段 breaker 一直在
"处理"的运行。上限 12 保证对每一个 period，武装都严格早于 guard 能行动。

`CWICK_LOOP_BREAK=0`（默认开）关闭它；`CWICK_LOOP_BREAK_MIN`（默认 8）移动武装
点。engine 会在 summary 旁边打印 `lb=N`——禁用武装期间做出的 pick 数——所以一个
带 `lb>0` 的 `stop` 看上去是一次恢复，而不是干净结束。

有两件事它刻意不做：它不施加一刀切的 repetition penalty（不发任何 penalty 字段的
请求保留 checkpoint 的采样默认值——隐藏的 penalty 会改变每一个答案，而不只是退化
的那些），它也不统计 prompt token（prime 过，未被 accept——以重复块结尾的 prompt
不是循环）。8 个尾部 token 的精确周期性在普通文本中不会出现（结构重复的是*相似*
块，不是 token 相同），所以误报的代价是在一段风格化标点运行上被迫换一个同义词，
换来的是在 agent 流量上实测 54% 的 loop 率。

验证：单元测试覆盖禁用逻辑（argmax 处拒绝延续、重新武装、禁用旋钮、短的合法重复
永不武装），而 sampler 的分布测试在 breaker 关闭的情况下运行——它们把同样的 logits
喂进去数千次，按构造那就是循环的特征。`make test` 通过；seed 固定的 metal 生成在
干净 prompt 上与 breaker 之前的构建保持 bit-identical。

## 12. thinking budget 必须由 engine 来写，而不是由 server 来切

对线上 agent 流量做的一次健康检查发现，有一个请求把它拥有的 8096 个 token 全花在
think 块里，交付的是 `content=0, finish=length`：193 s 的 decode，一无所获，而
客户端唯一能做的就是整轮重试。trace 把形状说得毫无歧义——推理在头几百个 token 内
就退化成片段循环，再也没恢复（token 层面的一半见 §11）。

只切掉*呈现*是不够的。服务端截断可以在 wire 上关闭这个块，但模型还在写同一个
念头，于是随后的"content"只是换了标签的念头。因此机制放在 engine 的 decode 循环
里（`cwick_engine_set_think_budget`），在那里可以喂入一个并非采样得到的 token：
到达上限时 engine 自己写入闭合标签，只写一次，而它同时落到两个地方——

- **输出**，经由 sink：server 的状态机看到 `</think>`，把它之后的一切重新归类为
  content，流式（`cwick_stream_segment`）和缓冲（`split_reasoning`）两条路径都是
  如此；
- **上下文**，经由前向传播：模型刚刚写下了自己的分隔符，于是它开始作答，而不是
  继续那个念头。

上限按 effort level 划分（关=0 / 低=512 / 中=2048 / 高=8192 /
超高=16384 / 极致=32768——自 §22 起就是同一张表）。关 是 0，而 0 意味着"作答，
不要思考"：当还没有块打开时，engine 写入模板自己的空形式
`<think></think>`，这正是 checkpoint 渲染一个没有思考的 assistant turn 的方式——
不是被截断的念头。

有两个细节决定这件事是安全而不只是聪明。第一，标记必须是单个 token：engine 把它们
当作完整的 step 写入，拆开一个就会破坏它正在关闭的块。server 每次加载时探测一次
tokenizer（`srv_probe_think_ids`），如果任一标记不是恰好一个 id，就禁用这个机制，
而不是去猜字节切分。第二，engine 只关闭真正打开的块：它跟踪 `think_open`
（要么由模板种下——GLM 的 prompt 以 `<think>` 结尾，server 通过比较最后一个
prompt id 检测到——要么由模型写入），而一个从未打开过块、或自己已经关闭了它的
turn 永远不会被碰。budget 只能改变一个本来就打算跑很久的念头。

`CWICK_THINK_BUDGET=<n>` 覆盖 level（0 = 从不思考，负值 = 关闭），对 server 和
`cwick run` 都生效，这才使得这个机制可以在真实 checkpoint 上测量而不必再来一份
权重。access log 增加 `think_cap=<n>`，并在闭包触发时标记 `(hit)`，于是被 cap 的
turn 可以和自行结束的 turn 区分开。

验证：`cwick_think_plan` 是纯函数，并针对整个矩阵做了测试（低于 / 等于 / 高于
上限、已经关闭、从未打开、budget 0 且有无打开的块）；在合成模型上做的一次真实
decode 断言注入的这对标记是写入的第一样东西，且 `think_budget_hit` 紧随其后；
未到达上限时不注入任何东西；词表外的标记会禁用机制，而不是写入另一个 token。
`make test` 通过。

光有 level 表救不了开启这一切的那个请求。它请求的是 高（8192）配
`max_tokens` 8096，所以 8192 的上限在长度限制之前永远不可能到达——一个无法触发的
cap 不是 cap，而它存在就是为了救的那个 turn，恰恰就是死在限制上的那一个。因此
engine 会把任何大于等于 `max_new` 的上限 clamp 到
`max_new - CWICK_THINK_RESERVE`（512，或更小预算的一半），并在请求旁边记录这次
clamp。剩下的就是"先思考，再作答"的诚实版本：念头被切在答案仍有空间存在的地方。

## 13. 分块 MoE 以 arena 宽度分批读取它的 expert

`ds_moe_chunk`（`n > 1` 的路径，也就是 prefill）遍历 chunk 各 route 的并集，并把
每个不同的 expert 读一次。这部分是对的——正是并集让长 prompt 变得可负担。错的
是*如何*读：每个 expert 一次 `ds_expert_fetch`，每次都是阻塞式 pread，于是整个
并集一次一个 6 MiB 请求地过线。

对这个设备来说这是最糟的形状。单次 6 MiB 读取在这里只能维持约 **2.5 GB/s**，
而限制在于延迟，不是带宽：同一块 SSD 在紧密循环中一次一个请求能给出 ~6.7 GB/s，
四到八个在途请求能给出 ~9 GB/s。在一个 17-token prompt 上实测，expert fetch 占了
**12.7 s prefill 的 68%**，而各 bucket 之和与 wall clock 相差在 0.2 s 之内——什么
都没有和任何东西重叠。

修复办法就是 decode 路径已经用过的那套序列，应用到 chunk 的并集上：
`ds_expert_prefetch` 提交，`ds_expert_take` 消费，`ds_expert_apply` 计算，
`ds_expert_pin` 纳入。批就是一个 prefetch arena zone 的宽度——`n_activated_experts`
（6）个 expert、18 个矩阵，这正是 arena 当初确定尺寸的依据——于是一个批的读取在
第一个 multiply-add 开始之前就全部在途，其余的在它底下完成。

| batch width | prefill, 17-token prompt |
|---|---|
| 1 (= the old order) | 11.75 s |
| 2 | 9.09 s |
| 4 | 8.13 s |
| 6 (default) | **8.00 s** |

在 4–6 处饱和就是 arena zone 的宽度，这也是为什么批被 clamp 到
`n_activated_experts` 而不是做成一个自由旋钮：更宽的批放不下，
`cwick_stream_prefetch_submit` 会静默退回到 demand I/O，而那个行为正是这次改动要
消除的。

**输出是 bit-identical 的，不只是看起来合理。** 批宽度改变的是读取*何时*发生，
别的什么都没变：每个 (position, slot) 仍然是一次进入自身贡献的 GEMV，仍然舍入到
bf16，仍然事后按 slot 顺序求和。`CWICK_DS_PF_BATCH` 的存在就是为了让这个断言可以
被测试而不是被宣称——宽度 1、3、6 的构建在同一个 prompt 和 seed 上输出相同的
md5。`make test` 通过。

有一个副作用，在跨版本比较 hit rate 之前值得知道：access log 里的 `cache=` 字段
换了分母。它以前是 `hits / (hits + misses that passed the ghost gate)`，因为
`cwick_stream_expert_put_read` 会在 ghost filter 拒绝一个 cold key 时提前返回——
什么都不计。经由 `cwick_stream_expert_cached` 路由会统计每一次查询，于是同一个
请求现在读作 `8.5% (494/5299)`，而旧路径打印的是 `68.6% (494/226)`。分子没变；
分母终于是被问到的那些问题数。

prefill 里剩下的是批之间暴露出来的尾部（28% 的时间）。要消除它，需要在当前批
计算时让下一批的读取在途，而 `cwick_stream_prefetch_begin` 表达不了这一点：它会
排空正在离开的那个 zone 并丢弃该 zone 中未被认领的 job，所以滚动窗口会丢掉已经
付过钱的工作。跨批 pipeline 意味着要在翻转期间持有 job 生命周期，那是对线程契约
的改动，而不是对循环的改动。未做。

## 14. 一个 Metal batch 必须主动要求 concurrent dispatch

`cwick_metal_batch_*` 有意把独立的 GEMV 收集到一个 command buffer 里：单个 8 MiB
的 expert 矩阵要花 ~300-700 us 的 host 往返，而 GPU 工作只有 ~105 us，所以每个
矩阵一次 commit 无论 kernel 多快都是 launch-bound。batch 没有要求的是让它们
*重叠*的许可。Metal 默认会把一个 command buffer 内的 compute dispatch 串行化——
dispatch *i+1* 要等 *i* 退役才开始——于是 batch 消除了往返，然后又一个接一个地跑
dispatch，每个都在等自己的 8 MiB 到位。GPU 在一条能跑 ~227 的传输上测得 45 GB/s。

修复只需一个 API：`-computeCommandEncoderWithDispatchType:` 配
`MTLDispatchTypeConcurrent`。它在这里是安全的，理由来自结构而非期望：一个 wave
里的每个 dispatch 只写自己的输出 slab，只读共享的只读输入（routed group 的 `x`、
shared expert 的常驻权重），而层里唯一真正的依赖——`down` 需要
`silu(gate) * up`——位于两个 wave 之间，也就是两个 command buffer 之间，那里的
顺序仍然有保证。实测：这个 wave 从 2.07 ms 降到 1.27 ms 每层，`gemv` 从 170.6
降到 126.4 ms/step，727.8 降到 663.0 ms/step，而生成的文本逐字节相同，因为
dispatch 顺序从来不是算术的一部分。

有两件事这*没有*修复，值得写下来，免得下一个人把 118 GB/s 读成"内存已经饱和"：

- fp8 kernel 仍然要通过一个以该字节为索引的 `threadgroup` LUT 解码每一个权重
  字节，于是 32 个 lane 收集 32 个互不相关的表项并付出 bank conflict。它的诊断
  孪生 `v3d`——相同的 load、相同的累加器形状，用 `int`->`float` 替代查表——在
  0.67 ms 就跑完同一个 wave，即 **227 GB/s**，这会诱人得出"表要花 1.9 倍、
  去掉它就是白赚"的结论。`v4` 正是测这个，而且**输了**：一个 bit-exact 的 E4M3
  解码是 ~13 个整数运算，而 gather 加上它的冲突更便宜（gate/up wave 上
  1.396 -> 1.657 ms，down 上 0.799 -> 0.851 ms，v4 还把它扩展到 k=2048）。所以
  227 GB/s 是完全*没有*解码的纯 load 的代价——一个任何正确实现都到不了的上界
  ——而 ~110 GB/s 才是正确的 fp8 GEMV 在这块 GPU 上的落点。这个实验保留编译，藏
  在 `CWICK_FP8_KERNEL=v4` 后面；数字在 `docs/PERF.zh-CN.md`。
- `MTLDispatchTypeConcurrent` 需要 macOS 12。helper 会退回到串行 encoder 而不是
  失败，因为一个被拒绝的 batch 会把调用者整层推到 CPU 路径上，那比慢 GPU 贵得多。

## 15. 一个矩阵级的 GPU 切分赚不回自己

DeepSeek 的 decode 路径完全跑在 CPU 池上，所以 `--backend metal` 只给它买到一
个装 expert blob 的 arena。按输出行把每个 GEMV 在池和 GPU 之间切开在*算术上*是
免费的——每一行的链条由 CPU kernel 固定，一行恰好由一个 engine 写出，新的
`CWICK_DS_GPU_VERIFY=1` 自检（在 CPU 上重算 GPU 的那些行并比较）确认了这一点：
fp4 从不报差异，bf16 只在某些形状下 ~1% 的行上差最后一位。它在*时间上*不免费，
而这才是决定性的。

一次 submit+wait 无论矩阵多大都要 ~0.2 ms，而一个 decode step 会发出 ~1400 个
GEMV。在真实 dense 形状的生产入口处实测（`tools/ds_gemvbench --gemv shapes`）：
对于 84 MB 的矩阵，GPU 单独只有 ~80 GB/s，而池在同样的字节上是 100-270 GB/s。
GPU 只有在一打矩阵共享一个 command buffer 时才达到 198 GB/s，而
这个接口一个矩阵一个 buffer。端到端，切分让 decode 变成 763 ms/step，全 CPU 是
461；一个 16 MB 的尺寸闸门保留了有收益的形状，把它带到 518，仍然是亏。

所以 `CWICK_DS_GPU` 默认 0，kernel 保留。能改变答案的是一次携带一批独立矩阵的
提交——八个 block-diagonal 的 `wo_a` 组和 `shared w1/w3` 天然就是——并把等待推迟
到依赖点。在那之前，诚实的总结是：这台机器上的 DeepSeek decode 是一个
17.9 GB/step 的流式问题，CPU 在 dense 那一半已经跑到 78 GB/s，而 GPU 只有在被喂
一个批而不是一个矩阵时才能增加带宽。

## 16. 比 GPU batch 更便宜的两个答案，以及为什么两个都已关闭

两个都试过，好让下一次尝试不再重复它们。

**去掉 expert pin copy 一文不值。** 每次 miss 都读进 prefetch arena，然后拷进它
的 cache slot，把计时器从 `expert fetch` 里拆出来之后测得这占一个 decode step 的
7.8%（35 ms）。直接读进一个预留的 cache slot 是可以表达的——
`cwick_stream_expert_reserve` 交出 slot，`prefetch_submit_into` 可以把 pread 落到
那里——而且它端到端实现过，输出逐字节相同。step 从 453.6 变成 452.1 ms，也就是
毫无变化：那次拷贝本来就已经和同一个 expert 其他矩阵的 SSD 等待重叠了。剩下的
18 ms 在 `fetch` 和 `matmul` 内部重新出现。回滚了，因为一条更复杂的读取路径还
破坏了 hit counter 的含义（预留的 slot 在其字节到达之前就被索引，所以 `take` 里
的 cache 查询必须搬家），不值零收益。

**expert hit rate 没有余量。** 48.7% 对着 81.9 GiB 的 cap 看起来很差，直到你问
这个 cap 有多满：`expert pin set: 79.43 GiB used of 81.86 GiB cap`，97%。预算已经
饱和，271 GB 的 expert 权重装不进 82 GiB，而均匀 routing 只会给 30%——所以 48.7%
就是 routing 的集中度已经提供的。要提高它需要这台机器没有的 RAM，或者一个更低
精度的 expert pack，而那越界了。

剩下的就是唯一还没被否证的东西：每个 command buffer 提交一批独立矩阵，让 GPU 在
池流式读取*的同时*增加带宽，而不是一个矩阵一个矩阵地取代它。

## 17. 这台机器读内存 123 GB/s，而这决定了 5 tok/s

§16 里的一切都是在找一场便宜的胜利。这一节是说明没有这种胜利的测量，外加一个
很有希望、却被 checkpoint 本身排除的想法。

**峰值读带宽是 ~123.5 GB/s，而且不动。** 一个裸的 12 线程流式 64-bit 求和扫过
冷 buffer，4 线程给 123.5 GB/s，12 线程 123.2，32 线程 122.3；1 GiB、4 GiB 和
16 GiB 完全相同；放宽到八个独立累加器和每轮 32 次 load 也什么都没变。四个线程就
把它跑满，这是 DRAM 控制器极限的特征，不是循环写得差。

这重新框定了整个 profile。一个 decode step 移动 17.9 GB，所以即使是一次完全没有
算术的纯流式读取也要花 145 ms，而 5 tok/s（200 ms）需要持续 89.5 GB/s——天花板
的 73%——同时还要扛着 SSD stream。今天的 453 ms 是 39.5 GB/s，32%。而 dense 那
一半*已经*在天花板的 80%（`wo` bucket 跑到 99 GB/s，`--gemv shapes` 在 wo_b 和
indexer 上显示 125-127 GB/s），所以再多的 kernel 工作也帮不了 dense：只有更少的
字节才行。

**把 dense 字节减半的唯一办法已经死了，而且是 checkpoint 杀的。** dense 权重在
文件里是 FP8 e4m3，加载时被放宽成 bf16（`ds_model.dense16`，11 GiB）。原生读取
它们会把 dense 流量减半（13.7 -> 6.85 GB/step），为 expert cache 腾出 5.5 GiB 的
常驻空间，而且不需要新的管线，因为 `w16` 为 NULL 时 `ds_gemv_q` 已经会 dispatch
到 `ds_fp8_gemv`。快速的 NEON 解码是三个位运算——
`((b & 0x7F) << 7) + 0x2000 | ((b & 0x80) << 8)`，在 1.0、0.5、-1.0 和 448 上手工
验证过——并且对每个 normal 值都精确。它在两个角落上是错的：e==0 是 subnormal
（m * 2^-9，而不是 2^-7 * (1 + m/8)；在 m=4 时是 0.0078 对 0.0117，50% 的误差），
而 e==15,m==7 是 NaN。加到 `bind_q` 里的一次扫描发现 **全部 330 个 fp8 tensor 都
至少含有一个这样的字节**，所以这些角落并不罕见。修复它们需要两次 `vqtbl1q_u8`
gather、一次 zip、一次 sign merge 和一次 bsl——每 16 个值约 15 条额外指令，把解码
从每 16 个值 6 条抬到 29 条，而 bf16 只需要 6 条（两次移位）。那是每个字节 ~1.5
条指令对 bf16 的 0.48，把 fp8 从 bandwidth-bound 翻成 issue-bound，并抹掉字节减半
带来的全部收益。放弃；扫描也随之移除。

**而其余那些，实测，都是零。** 把 fp4 GEMV 展开成两个块并加一个 prefetch：
gate/up 65.5 -> 75.6 GB/s，但 down 79.1 -> 53.5，端到端 493 ms 对 453。
Reader 线程 2/6/12：expert matmul 26.3/26.5/26.6%。
`CWICK_EXPERT_CACHE_GIB=95` 对自动的 82：2.067 对 2.055 tok/s，hit rate 33.2% 对
32.5%，residency 95.78 GiB——机器已经满了，这也是 hit rate 同样没有余量的原因。

所以这台机器诚实的数字是今天 ~2.1 tok/s，而如果剩下那一个真实缺口被补上，也许
~3.3 tok/s：expert matmul 在实况中跑到 36 GB/s，而同一个 kernel 的冷 bench 是
65 GB/s，值大约 55 ms。这里找到的任何东西都到不了 200 ms/step，因为 200 ms/step
要求带宽天花板的 73%，而 dense 那一半已经花掉了其中的 80%。

## 18. 按精确长度取键的 prefix cache 会错过每一次多轮对话

prefix cache 用*精确长度*把 prompt 与已存快照比较：`cwick_pfx_lookup` 在每个位置
采样滚动 hash，并寻找 `pos` 等于该位置的条目（`pfx.c:383`）。那是对的、便宜的，
也正是为什么 0% 的 hit rate 要靠一次线上审计才发现，而不是作为错误冒出来。

存在的那些长度是 prompt 本身、分块 prefill 期间每隔 `pfx_block` 个位置、以及末尾
的 prompt+generated。一段继续的对话需要的长度不在其中。一个打开 thinking 块的
请求以 think-open 标记结尾，那是 server 追加的；同一 session 的下一轮把这个
assistant turn 渲染成 think-close 标记加它的内容。两个 prompt 恰好在最后一个 token
上分叉，所以它们共享的前缀比刚捕获的那个 prompt 短一个 token。第 N 轮是 262 个
token，第 N+1 轮是 405，它们在前 261 个上一致；缓存里存的是 262 和 551。同样的
算术让 session 里其他每一次转换都失败，而 `CWICK_PFX_MIN_TOKENS` 在 256 时开头
几轮根本没被写入，block checkpoint 又需要一个 8192-token 的 prompt 才触发。四个
请求，四次查询，零命中，260 s 的 prefill 重付。

所以 `pfx_save_now` 现在还会在 `nids - 1` 存一份孪生，用同样那些 token 减去最后
一个来取键。这不是新机制：整 prompt 的命中路径在加载时本来就会把一个快照回滚一个
位置（`server.c` 的 `pfx_cap.pos--`），所以捕获一份正是代码早已信任的回滚的对称
一半。它的代价是一次额外的 `cwick_engine_capture_prefix`，它可以接受任何
`<= eng->pos` 的位置，也就是一次状态打包而不是一次 prefill，而 writer 队列是有序
的，所以万一要丢弃一个快照，丢的是孪生而不是全长的那份。
`CWICK_PFX_MIN_TOKENS` 降到 32——在这个 checkpoint 上，33-token 的一轮要花 19 s
的 prefill，因为 expert-union 读取占主导，所以旧的"短 prompt 不值得一个文件"的
推理不成立。

在三轮、两个 arm 请求逐字节相同的条件下实测：hit rate 2/3，125 个 prompt token
中复用了 73 个，prefill 53.6 s -> 27.3 s，三个响应在 `content`、
`reasoning_content` 和 `finish_reason` 上完全相同。剩下的那一次 miss 是第一轮，
它没有东西可以重放。

可推广的部分不是关于 thinking 标记的。而是：一个按精确长度取键的缓存，只能服务
那些长度恰好被它捕获过的转换，所以捕获策略必须从*下一个请求的 prompt 是如何构造
的*推导，而不是从当前这个推导。这里下一个 prompt 是当前这个把尾部重写，于是
"短一个"就是正确的策略；一个会重新 tokenize 回复的客户端需要另一种策略，而它会
响亮地 miss 而不是静默地 miss，这正是值得保留的性质。

## 19. 逐请求的 engine 状态必须推送到每一个会 decode 的 backend

thinking budget（L4）实现在 `engine.c` 的 decode 循环里，并为一个 chat 请求由
`server.c` 武装，还带一行这么说的日志。DeepSeek 不走那个循环 decode——
`ds_engine.c` 有自己的——所以上限被配置、被上报，却从未被读取。一个线上请求在
配置为 8192 的情况下生成了 16,568 个 thinking token，跑了 2 h 35 m 且还有 ~30 h
要走。loop guard 抓不到它：模型没有重复时 `finished_reason=loop` 保持 0，而这一个
确实还在思考。

这个模式在代码库里早就有，只是这个字段没有遵循它。`ctx.stop`、`ctx.tag`、
`ctx.spec` 和 `decode_phase` 都由各自的 engine 推入 `eng->ds->ctx`，正是因为
DeepSeek backend 跑一个独立的循环。`think_budget` 是第五个这样的字段，也是唯一
留在 GLM 一侧的，这才让遗漏变得不可见：武装代码是共享的，日志行是共享的，唯一
不共享的是执行它的那个循环。

所以这确立的规则是：**一个逐请求的 engine 设置，在每个能 decode 的 backend 都有
它的副本之前，都不算武装**，而要检查的不是"它在 engine 上设了吗"，而是"读 logits
的那个循环读它吗"。一行报告配置的日志不是执行的证据——在这里它主动误导，因为
`think_cap=` 对每个 DeepSeek 请求都打印 0（那条路径上 `res.think_budget_applied`
从未被设置），这读起来像"没有请求 thinking"，而不是"上限没有被应用"。

实现里有两个细节值得保留：

- 闭合标签必须在同一个 step 里既被写入*又*被前送。sink 必须看到它，这样 wire 上
  才是一个已关闭的块，server 的 segment 机才会把其余部分重新归类为 content；而
  前向传播必须消费它，这样模型——刚写下自己的分隔符——才会作答而不是继续那个念头。
  只有 decode 循环能喂入一个它没有采样得到的 token，这就是为什么这件事不能住在
  server 里。
- `eng->ds` 由 `cwick_xmalloc` 分配且没有 memset，所以禁用状态必须在
  `ds_engine_open` 里显式为 -1。一个清零的结构读作 budget 0，也就是"在第一个
  token 之前关闭 think 块"——正是 `cwick_engine_new` 自己的注释警告的那个陷阱，
  而这次修复的第一版就掉了进去。武装的情形看起来是对的；只有对照组——环境变量
  未设置的那次运行——才显示出机制在本该沉默时触发了。要跑对照组。

## 20. 移除 --max-output-tokens：一个只可能错过的 clamp

它存在的目的是给一个省略了 `max_tokens` 的请求设界，否则那个请求会拿到整个剩余
上下文（100K 窗口上的 99,501 个 token，按 1.3 tok/s 约 21 h，最终总是以客户端
abort 结束）。那部分管用。

它救不了它被加进来要救的那个场景。clamp 以 `!gr->max_tokens_set` 为条件，所以它
只作用于省略该字段的请求——而每个真实客户端都会设置它。内置 web UI 默认发
`max_tokens: 100000`，这就是为什么促成整场审计的那个线上 session（`start.sh` 里
有 `--max-output-tokens 10000`）让一轮跑了 2 h 35 m 且还有 ~30 h 要走：clamp 从
来没看到它。对着它本应防止的三种失败模式测量，它一种都没防住。

更糟的是，它在两个报告配置的地方主动误导。`/v1/config` 发布了
`max_output_tokens`，设置面板把它渲染成一行，于是读其中任何一个的运维都会以为
请求被限制在 10000，而真正被限制的只有没人会发的那些请求。一个被报告却不被执行的
限制比没有限制更糟，因为它让所有人都停止去寻找真正的那个。

已移除：选项、`cwick_server_options` 字段、clamp、`/v1/config` 的键，以及设置
面板那一行。`--max-tokens` 保留它的含义（省略该字段的请求的默认值），而 engine
自己的上下文天花板 clamp 仍然会给一个未设上限的请求设界，并有一行日志按实测
tok/s 把它折算成小时。给一轮设界是客户端的 `max_tokens`，这本来也是 OpenAI 对该
字段的解读。

可推广的部分：在加一个服务端安全 clamp 之前，先检查它实际能触达哪些请求。一个以
"客户端没有要求这个"为条件的 clamp，是给不存在的客户端准备的 clamp。

## 21. 移除 --max-prompt-tokens：唯一一个只会带来成本的旋钮

它用 400 拒绝超过 N 的 prompt，理由是 30K-token 的 prefill 会独占单个生成槽位
好几分钟，而客户端在它结束之前什么都看不到。默认值是 0（不限制），所以它只对
敲下这个 flag 的人有意义——而 `start.sh` 把它敲成了 100000。

那个数字正是它开始说不通的地方。在 `--context 1000000` 下，窗口本身就容纳一个
百万 token 的 prompt，所以 100000 的闸门什么都没拒绝，engine 本来就会拒绝；在旧的
`--context 131072` 下，它只拒绝 100000 到 131072 之间的 prompt，这个区间既长得
无法在合理时间内 prefill，又短得不值得一条单独的代码路径。它是一个被螺栓固定在
容量限制上的策略旋钮，而且只可能削减能力：一个被它拒绝的请求，变成一个 400 而不是
engine 自己的错误，后者会点明确切的窗口并说明怎么提高它。

已移除：选项、`cwick_server_options` 字段、拒绝块、`/v1/config` 的键、设置面板那
一行、usage 文本和 README 段落。现在给 prompt 设界的只有 `--context`，别无其他，
在 engine 内部执行（`prompt 400 + max_new 4 at position 0 exceeds the
64-position context window`），对非流式请求报告为生成错误，对流式请求报告为带内
SSE 错误。

这样 server 就只剩一个与长度有关的旋钮，`--context`，这才是诚实的形状：容量是
机器的属性，而与它不一致的策略限制是第二个迟早会错的事实来源。如果哪天真的需要
拒绝一次长 prefill，正确的位置在 server 前面——一个代理或客户端，它知道*这个*
prompt 值不值得等。

## 22. 一个 thinking level 的两半是同一行表

一个请求会指定一种强度（`低`/`中`/`高`/`超高`/`极致`/`关`，外加英文别名）。这一个
名字被三处独立的查表消费，每一处都有自己的 level 字符串副本：

- `cwick_thinking_effort_resolve` → level *以及* GLM header（`Low`/`High`/`Max`）
- `think_budget_for_level` → token 上限（`0`/`512`/…/`32768`）
- `ds_effort_from_level` → DeepSeek 的 `1..100` header *以及*是否思考

三条基于同样六个字符串的 if 链，就是三次让某处笔误静默落到默认值的机会。它不会
崩溃：`think_budget_for_level` 里漏掉一个 `strcmp` 会返回 -1，含义是"没有上限"
——正是 budget 存在要防止的失败，而且它表现为一轮跑很久而不是一个错误。
GLM/DeepSeek 的不对称有同样的暴露面：它住在 resolver 上方的一段注释里，而不是
在数字旁边。

它们现在是一个行数组 `THINK_LEVELS`，每个消费者一列：

```
level   glm     ds   budget  thinks
关      Low      0        0  false
低      Low     25      512  true
中      Low     50     2048  true
高      High    75     8192  true
超高    Max     90    16384  true
极致    Max    100    32768  true
```

这样写出来会有两件事自然浮现。第一，不对称现在可见而不是被文档化：`低` 和 `中`
共用 `Low`，因为 GLM checkpoint 训练时恰好只有三个 header，而表显示它们的 budget
相差 4 倍。第二，一个 level 无法在不获得 header 的情况下获得 budget——两者无法
分开编辑，因为它们本来就是：强度是模型被*告知*的东西，上限是 engine *执行*的
东西。

`CWICK_THINK_BUDGET` 仍是刻意的例外，只触碰 budget 那一列。这种不对称正是这个
旋钮的意义——保持模型的指令不变，只改变上限——也正是 §19 里 DeepSeek 执行 bug
被发现的方式：`CWICK_THINK_BUDGET=64` 仍然发送 `高` 的 header（effort 75），
仍然只得到 64 个 token 去遵守它。

这张表在启动时打印，所以映射可以从日志核对而不是从源码：

```
thinking levels (header / token ceiling): 关→Low/0  低→Low/512  中→Low/2048
  高→High/8192  超高→Max/16384  极致→Max/32768
```

在真实 checkpoint 上验证行为未变：`高` 仍然渲染 `Reasoning Effort: 75` 并武装
8192，`关` 仍然武装 0 并在第一个 token 之前触发（`think_cap=0(hit)`，它的 prompt
是 5 个 token，因为 `thinks=false` 完全跳过 header）。`make test` 针对 GLM 路径
覆盖每一种拼写（`test_cwick.c`，"http chat accepts every
thinking-intensity level"）。

## 24. 一次模型切换必须把页还回去，而不只是把对象还回去

一个线上的 `serve` session 从 DeepSeek 切换到 GLM-5.3-Flash，然后死了。日志毫无
歧义：切换在 21.8 s 内报告成功，进来的模型以 **67.19 GiB** 常驻完成 warm-up，它的
第一个 decode step 就已经压在 102.40 GiB 的 ceiling 之上，ceiling 检查随后做了它
唯一能做的事——把 expert cache 还回去，七个 step 内从 32.00 降到 0.00 GiB——而
footprint 在还的过程中纹丝不动，从那以后它每一步长 **7.9 GiB**（恰好是每步流式
读取的 7.88 GiB expert），直到进程在 185.93 GiB 处被杀。两个测量把原因和症状分开：

- 同一个 GLM 全新启动时 warm-up 只到 **14.49 GiB**，并以其 32 GiB expert cache
  完好无损的状态在平坦的 **71.51 GiB** 上 decode 九个 step。所以计划和 decode 路径
  本身都没错。
- 差别，52.7 GiB，几乎正好是切换开始时 DeepSeek 占用的量（复现中 51.70，失败
  session 中 53.71）。

所以切换释放了 engine 对象，把页留下了。三个彼此独立的原因，都是"unload 不等于
exit"：

1. **`ds_model_close` 从未释放 `dense16`。** dense 权重的 bf16 放宽是一个由
   `ds_dense_sizes` 确定大小的 arena block（这里 ~11 GiB），它根本没有释放调用，
   所以每次加载都把它留在身后。
2. **Arena buffer 从未被释放。** `cwick_arena` 把它的后备 `id<MTLBuffer>` 放在一个
   malloc 出来的结构里。ARC 会在赋值时 retain，但 `free(g_arenas)` 无法通过它
   release，于是那块 wired arena——整个 expert-cache 高水位——在进程余下的生命里
   一直常驻。no-copy 视图也是如此，而且更糟：它们包裹的是模型*映射*的权重，所以
   持有它们会让即将离开的映射一直活着，即使在 spine 关闭之后。
3. **内存计划是静态的。** 它由模型自己的尺寸算出，所以进来的模型无论在这个进程里
   是独占，还是站在前任的 50 GiB 之上，都照样预订
   `spine 15.46 + runtime 22.50 + arenas 0.03 + headroom 0.38 + cache 32.00`。
   因此超订在计划阶段根本检测不到，只能由运行时 ceiling 检查发现——而它的反应是
   摧毁那唯一在正常工作的组件。

修复针对 (1) 和 (2)，并在同一次切换上实测：

| | before | after |
|---|---|---|
| incoming model, warm-up | 67.19 GiB | **43.74 GiB** |
| expert cache during decode | 32.00 -> 0.00 GiB | **32.00 GiB, held** |
| footprint over 9 decode steps | 107.41 -> 185.93 GiB | **89.15 GiB, flat** |
| process | killed | serving |

`ds_model_close` 释放 `dense16`；`cwick_metal_arena_release()` 是一个新的入口，
丢弃每一个 arena block 和每一个 no-copy 视图，通过 ARC 释放 buffer，同时让
device、queue 和 pipeline state 保持存活——一次切换不需要重新编译 shader，只需要
即将离开的模型的内存。`cwick_metal_bridge_cleanup` 现在也走同一个 helper，所以关闭
路径也不再泄漏 buffer。切换路径在 `cwick_engine_free` 和进来的 `cwick_engine_new`
之间调用它，在新模型规划它的预算之前。

那之后剩下的还有 29 GiB，而且它随每次切换增长（第二次往返峰值 70.27 GiB）。
`cwick_safetensors_close` 和 `cwick_spine_close` 都把它们的映射记录为
**process-lifetime** 且从不 unmap——这个策略在一个进程只服务一个模型时是正确的。
两者现在都记录每个映射的长度并在 close 时丢弃它，因为那个显而易见的替代方案不
可行：**`madvise(MADV_DONTNEED)` 对 macOS 上的文件映射是 no-op。** 实测，它对全部
48 个 DeepSeek shard（475 GiB 地址空间）返回成功，而 RSS 一个字节都不动，所以不能
用它来宣称内存已释放。`munmap` 在 DeepSeek 一侧还回 9.20 GiB，在 GLM 一侧 14.44
GiB。这强制了一个顺序约束：no-copy 的 Metal 视图是覆盖映射权重的 buffer，所以
`cwick_engine_free` 要在 `cwick_spine_close` unmap *之前*释放视图
（`cwick_metal_views_release`），而 arena 在 stream 把它的 blob 还回它们之后再
释放（先释放 arena 会让 `cwick_shmem_free` 对一个 arena 指针调用 `free()`）。

在这次切换上实测，分三个阶段：

| | incoming model, warm-up |
|---|---|
| original | 67.19 GiB |
| `dense16` + arena/views released | 43.74 GiB |
| + checkpoint unmapped on unload | **34.57 GiB** |

而真正重要的是累积效应：三次切换以前结束在 70.27 GiB warm-up（每次往返 +26.5），
现在结束在 37.50（+2.94）。剩下的是 malloc zone：`free()` 把页还给分配器，而不是
还给 VM，`malloc_zone_pressure_relief` 也不把它们还回去——切换路径照样会调用它，
而那残留的 20 GiB 是一个曾经持有 60 GiB scratch 的进程的代价。它留在 ceiling 以下
且 expert cache 完好，而这正是被破坏的那个性质。

可推广的部分：一条只释放记账数据的 unload 路径，在第二个模型来要内存之前，看起来
和正确的一模一样。这里每一个比 engine 活得更久的分配器——arena、视图、映射——都
必须被问一句"模型换了之后谁来把它还回去"，而答案必须是代码，不是一句说映射活到
进程结束的注释。而当答案涉及 `madvise` 时，先确认 RSS 真的动了再相信它。

## 25. pin 预算必须为整个 footprint 付账，而且两个循环都要 steer

### 哪里错了

DeepSeek expert pin cache 的预算是按 `80% of RAM − dense − 5 GiB` 算的
（`ds_model.c`），在这台 128 GB 的机器上就是 81.9 GiB。那 5 GiB 本意是覆盖
"KV cache、contribution table 和分配器"。实测，footprint 中非缓存的部分是
~44 GiB：

| part | bytes |
| --- | --- |
| expert pin cache | 81.86 GiB |
| dense (fp8 arena + bf16 copy) | 15.5 GiB |
| compressed KV `cmp_kv` + indexer `idx_k` | 9.5 GiB at `--context 1000000` |
| Metal arenas / IOAccelerator / driver | ~18 GiB |
| scratch, prefix cache, misc | ~1 GiB |

所以稳态是 **126.11 GiB 对着 102.40 GiB 的 ceiling**，超了 23.7 GiB。pin cache 是
普通的 `malloc` 内存（arena 路径会退回到 `cwick_xmalloc`），所以超出的部分直接进了
VM：14 分钟内 swap 从 2.56 涨到 17.4 GiB（macOS 自己长出了 swapfile），compressor
做了 +38.0 M 次压缩 / +30.8 M 次解压——对一个只从 SSD 读了 158 GiB 的工作负载，
这是约 1.1 TB 的 VM 流量。特征无可混淆：同样的 cache-hit rate（75.3%）、每步流式
读取的字节*更少*、而 step time 在 330 到 3303 ms 之间游荡。那些命中是解压 fault。
DeepSeek 全新启动是 1.887 / 2.220 tok/s；一次切换往返之后同一个模型只跑到
0.622 tok/s。

有三件事让它成为结构性的而不是偶然的——三件都是"GLM 有，DeepSeek 没有"：

1. 预算从不看 `cwick_process_footprint_bytes()`；
2. `maybe_steer_memory` 在 `if (!eng->spine.stream)` 处返回——只有 GLM——而
   `cwick_stream_trim` 在整棵树里只有一个调用者；
3. `cwick_mem_lock_range` 只在 GLM 的 spine 和 GLM 的 KV 上跑，所以那行宣称
   "no-swap mode" 的启动日志根本没有描述 DeepSeek 的 cache。

### 改了什么

**A —— 预算为 footprint 付账。** `ds_expert_cache_budget()` 现在接收 config 并
减去实际存在的东西：`reserve = 4 GiB + KV bytes + host/8 (GPU arenas and driver)`。
KV 那项是 `ds_kv_bytes()`，与 `ds_model_open` 分配时用的是同一套算术，所以它跟随
`--context` 而不是假设 5 GiB。在 `--context 500000` 下它规划 **62.1 GiB**（原来是
81.9）；在 `--context 1000000` 下它规划 ~57 GiB。`CWICK_EXPERT_CACHE_GIB` 仍然
可以覆盖。

**B —— 一个阀门，两个循环都调。** `maybe_steer_memory` 的 body 搬到了 `stream.c`
里的 `cwick_stream_steer_memory(stream, ceiling, step, tag)`。GLM 循环照旧调用它；
DeepSeek 的 decode 循环每一步在 `cwick_stream_expert_decay` 旁边调用它，prefill
的 chunk 循环也调用它（那个阶段会连续几分钟往 cache 里填）。

### B 里的陷阱：不要每一步都 trim

第一版逐字继承了 GLM 的 body，包括 `cwick_stream_limit()`——它会*永久性地*降低
cap。`free()` 不会在同一瞬间缩小 `phys_footprint`，所以每一步都测到同样的超出量，
又把 cap 砍一次。在一个刻意放大的 90 GiB cache 下，cap 在**大约两秒内降到 0**，
每一行都在对着一个没动过的 footprint 报告"released 0.00 GiB"，而 decode 失去了
它的整个工作集。用以下办法修复：

- trim 之后一个 **12 s 的 cooldown**（`CWICK_STEER_COOLDOWN_MS`），让下一次决策
  读到一个有时间反映这次释放的 footprint；
- **trim 到 `over + 1 GiB`**，这样下一次回填不会立刻再次触发；
- 一个 `max(cap/4, 256 MiB)` 的**下限**，这样糟糕的一天也不能把 cache 交出去。

在同样的 90 GiB cap 下，它现在只记录一次 trim：为了 0.39 GiB 的超出，从 90 降到
88.61 GiB。

### 实测

同一台机器，同一个 prompt，`--context 500000`：

| build | steady footprint | decode |
| --- | --- | --- |
| before (81.9 GiB plan, no steering) | 94–100 GiB and still climbing to 126 | 1.421 / 1.436 tok/s |
| after (62.1 GiB plan + steering) | **84.76 GiB, flat** across requests | **2.146 → 2.781 tok/s** |
| after, with a deliberately oversized 90 GiB cache | 102.79 GiB, one trim to 88.61 GiB | 2.328 tok/s |

修复之后的各次运行里，swap 一直平在 4.1 GiB 中的 ~3.2，而之前是 18.0 GiB 中的
17.4。

### 仍未关闭

- **不要**把 DeepSeek 的 pin cache wired 起来。wired 页在这里测得*更慢*（`metal.c`
  里的注记），而重点是还回字节，不是把它们藏起来。
- `/v1/metrics` 根本没有 swap 或 compressor 的 gauge，这就是为什么这件事要靠采样
  `vm_stat` 才发现。`vllm:num_requests_swapped` 是请求抢占，与操作系统 swap 毫无
  关系。

## 26. 第三个 backend：MiniMax-M3.1-Flash 需要而另两个不需要的东西

port 是 `src/mm_model.c`（权重 + forward）、`src/mm_engine.c`（生成，DeepSeek
契约的副本）和 `src/mm_prompt.c`（checkpoint 的 `chat_template.jinja`）。
engine 层的集成刻意做得无聊：`cwick_engine` 在 `ds` 旁边多了一个指针，而每一个
以前问"这是不是 DeepSeek backend"的地方，现在对两者问同一个问题。无聊是要求——
server、prefix cache 和 `/v1/metrics` 必须无法分辨这三者。

*不*无聊的是下面这些，也是它们被写下来的原因：

**reasoning 标记是按 checkpoint 走的。** `srv_probe_think_ids` 以前探测固定的
`<think>`/`</think>` 对。MiniMax 写的是 `<mm:think>`（id 200059/200060），而它的
tokenizer *也*带着 `<think>`（200050）作为一个无关的控制 token，于是那个固定的对
武装了错误的 id，答案切分器在找一个永远不来的闭合标签：每个回复返回的 `content`
都是空的。标记对现在按常驻模型类型选取，并携带在 `cwick_stream_state`
（`open_mk`/`close_mk`）里，因为切分器是按字节工作的，自己查不了。

**两个内存安全 bug，都是被第一个 128-position 的 prefill chunk 找出来的。** 标量
路径从未触到它们，因为此前每次测试都只跑九个 token：

* expert 循环同时持有同一个 expert 的 gate/down/up 指针，而中间的一次 cache miss
  会驱逐其他 slot——于是后面某个 GEMV 读的 blob 在它脚下被释放了（SIGSEGV 发生在
  NEON kernel 内部）；
* `gsm`/`usm` 按 `inter_dim`（3072）分配，但三个 dense 层用 `dense_inter_dim`
  （12288）去索引它们，即 1.5M 个 float 写在 393K 的 buffer 之外，表现出来的却是
  两层之后在不相关的代码里崩溃。

可推广的部分：在这样的 port 里，"它在短 prompt 上产出正确的 token"对边界一无所知。
一个 128-token 的 chunk 是第一个按真实尺寸触到每个 buffer 的输入，它从一开始就该
在 smoke test 里。

**FP4 GEMV 慢到了它应有的 2.6 倍，原因是一次函数调用。** `ds_e4m3_to_f32()` 会问
`build_luts()` 表准备好了没有，然后索引它。kernel 每 16 个元素的块调用一次——
每行 384 次，每个 expert 矩阵约 1.2M 次。换成一个本地构建的 256 项表，把同一个
kernel 从 16.8 带到 43.5 GB/s（30 → 77 GMAC/s），再用向量累加器取代每块的水平
归约，走完了剩下的路。作为参照，同一个池上的 bf16 GEMV 测得 136 GB/s，那是这台
机器的内存天花板——所以 fp4 kernel 不再是那个离群值了。

**这个 step 是 I/O，而算术封顶了 token 速率。** 一个 decode step 读 19.4 GiB
（dense 12.6 + routed experts 6.9 + 与常驻部分共享的 shared 6.4），并执行
24.3 G MAC。expert cache 在 216 GiB 库的 28% 时，每步仍有 ~1 GiB 来自 SSD，而一次
同步的 10 MiB pread 在这里只能维持 1.3–2 GB/s，所以名义上叫"experts"的那个阶段
有一半以上在等待。在一个线程上预留每个缺失的 blob，并把 pread 铺到池上
（`cwick_stream_expert_reserve`，它存在的目的正是这个），把 step 从 1.2 s 带到
~1.0 s——比 stream.c 里注释承诺的少，因为一层平均只 miss 约 0.7 个矩阵，所以没有
东西可铺。

天花板比调优更重要：max(19.4 GiB / 130 GB/s, 24.3 G MAC / 77 GMAC/s) ≈ 315 ms 每
token，即 **3.2 tok/s 配一个完美的 cache**。在这台机器上顺序 decoding 的上限就在
这里；checkpoint 附带了一个 7-token 的 block draft model（`dspark/`，
2.3 GB）正是为此，而把一次权重读取摊销到多个被接受的 token 上，是唯一能倍增而不是
削减的路径。

### MiniMax backend 的 Metal：批处理过，测量过，关掉

显而易见的接法是把一层的五个 projection 通过 `cwick_metal_bf16_batch_add` 放进
一个 command buffer——GLM 路径已经在用这个 API，所以试它不需要新的 shader。在一个
decode step 上实测：**590 ms -> 1400 ms**，光 projection 阶段就从 142 ms 变成
932 ms。那相当于每层 ~15 ms 的 GPU 工作对 CPU 池上的 2.4 ms，而这个矩阵是
100 MB，按这台机器的 GPU 带宽应该在 ~0.3 ms 内移完。

所以批处理从来不是问题——每个矩阵一次 commit-and-wait 只是症状。一个
8192x6144 的 bf16 GEMV 要 3 ms，意味着 shader 让 GPU 闲着（数量级：33 GB/s 对
~400 GB/s 的设备）。在这件事值得重新审视之前，kernel 需要分块——每个 threadgroup
多个输出行、激活暂存在 threadgroup memory 里、以及一个能让 40 个核心都忙的
dispatch 网格——而那必须在接进一个 step 之前，单独对着 CPU 池测量。

这套管线留在 `CWICK_MM_METAL=1` 后面作为那项工作的 harness；默认值不能为一个实验
付账。

### FP8 dense：bf16 副本什么都没买到

MiniMax 的 checkpoint 本来就把每个非 expert 矩阵按 e4m3 存储，每行每 32 K 个元素
一个 e8m0 scale。port 在加载时把全部 540 个都放宽成 bf16（19.67 GiB，40 s），然后
每个 token 读其中 21.1 GB。放宽是精确的，但 bf16 GEMV **在 136 GB/s 上是
bandwidth-bound**——那是机器的天花板——所以字节翻倍时间就翻倍，什么都没买到。

现在它们按存储的样子使用，在 GEMV 内部解码（`mm_fp8_gemm`）：e4m3 字节通过位重排
变成 f16（保留符号、指数 +8、尾数 <<7），f16 到 f32 是一条指令，而每 32 个一块的
scale 在块求和之前作为向量常量乘进去。除了 e4m3 的 subnormal 一切都精确，那些
解码成 2^-7*(1+m/8) 而不是 2^-6*(m/8)——对权重来说最多 2 倍的误差，而那些权重最多
是其块最大值的 3.5e-5，即在块 scale 自身分辨率之下。

在一个 decode step 上实测（同一个 prompt，同一个 cache）：projection 142 -> 88 ms，
`wo` 52 -> 30，dense 流量 21.1 -> 10.6 GB，footprint 74 -> 60 GiB，加载
40 -> 32 s，最佳 step 590 -> 462 ms。对下一个 token 的判定未变
（`the quick brown fox … lazy` -> ` dog`，之前 19.70，之后 18.84）。

这付了两项代价，现在都要由内存计划来承担：

* FP8 权重是 file-backed 的页而不是匿名副本，所以它们可能被驱逐；预算为它们保留
  `fp8_total`，这也是 expert cache 没有简单地因为释放出的 19.67 GiB 而长大的原因
* 第一版把一行解码到一个 24 KB 的 scratch buffer 然后再做点积。那移动了 2 倍的
  数据，实测比 bf16 *更差*（250 ms）。融合版本——把八个值解码进两个 f32 向量并
  直接 FMA 进块累加器——才是赢的那个。

### 长 prompt prefill：数字怎么说，以及一条死路

302-token prompt 的 prefill：在 FP8 dense 改动之后是 **199.2 s**（1.5 tok/s），
之前是 205.6 s。路上测了三件事，其中两件值得保留：

* **prompt 在磁盘层面不是 I/O-bound 的。** 它流式读了 125 GiB，而 `--io` 显示设备
  对 10 MiB 读取能维持 9.35 GB/s。302 个 token 总共花 7.5 T MAC，即这个阶段跑到
  ~37 GMAC/s——是 decode step 的一半。prefill 是 compute-bound，这就是为什么更大
  的 cache 和更快的 reader 都推不动它。
* **prefix cache 才是长 prompt 的真正杠杆**，而在 `mm_state_apply` 修复之前它是
  坏的：一次 400-token 的轮次现在从快照重放 262 个 token（66%），而不是重新
  prefill 它们，这恰好是需求 12 的意图（前缀缓存 / 磁盘 KV checkpoint）。一段
  对话第一轮之后的一切都便宜；第一轮不便宜，而且在不读 expert 库的前提下无法变得
  便宜。

那条死路：**把一个 expert 的行归组进一个 GEMM**。prefill 在遍历 expert 并集之前
会先 route 整个 chunk，所以一个 expert 通常被一次请求 ~4 个位置，而逐行 GEMV 每个
位置都要读一次它那 10.6 MB 的矩阵。为整组只读一次显然是对的——而它实测
**212.4 s 对 199.2 s**，因为所写的分组 kernel 对每个激活的每 16 元素块都用一次
水平加来累加，这正是当初让 FP4 GEMV 慢 2.6 倍的模式。要回本，它需要 decode kernel
已经有的东西：每个激活一个向量累加器，在整个行上留在寄存器里。那是这里的下一步，
在 128 个位置时值约 4 倍更少的 expert 流量（512 时 16 倍）。

它在 `MM_PREFILL_GEMM=1` 后面，默认关闭，`tools/mm_run` 里的 `--cvt` 和
`MM_CHUNK_CAP` 作为那项工作的 harness：两个 kernel 在合成数据上和模型内部都一致
到 1e-6，所以缺的是吞吐，不是算术。

### 长 prompt prefill，第二部分：199 s -> 104 s，以及剩余时间在哪里

四项改动，每一项都在同一个 302-token prompt 上实测（`tools/mm_run`，top-2，
`--context 8192`）。prompt 是 302 个 token，这很重要：分块决定了 expert 库被遍历
多少次，而那最终成了最大的一项。

| | prefill | tok/s | streamed |
|---|---|---|---|
| before this round | 199.2 s | 1.5 | 125 GiB |
| + attention vectorised | 136.8 s | 2.2 | — |
| + `MM_CHUNK` 128 → 256 | 127.7 s | 2.4 | — |
| + `MM_CHUNK` → 512 | **104.2 s** | **2.9** | 111 GiB |

**瓶颈是 attention，不是 experts。** 阶段计数器（现在 prefill 也按 token 打印）
显示 `attn` 在三个 chunk 上分别是 91 / 273 / 398 ms 每 token——随位置增长，而且在
最后一个 chunk 里是单项最大。内层循环是标量的，每个元素一次
`cwick_bf16_to_f32` 调用：每次迭代一次乘法，没有 ILP，而且 8 个 query head 在 GQA
下重复读取它们共享的同一个 KV head。bf16 是 f32 的上半部分，所以放宽只需一次移位
（`vshll_n_u16(h, 16)` → reinterpret）；配上四个累加器，点积在那个位置上从 398 降
到 **30 ms** 每 token，这个阶段 13 倍，整个 prefill 1.45 倍。这是整个练习里唯一
一项超出预期的改动，而原因很简单：以前没人看过那个循环。

**chunk 大小在这里是一个真杠杆，而理由是 I/O。** 在 128 个位置时，一个 302-token
的 prompt 是三个 chunk，每个 chunk 把 512 个 (position, slot) 对 route 到 128 个
expert 上——也就是每个 chunk 每一趟基本上都是整个 expert 库。所以 prompt 的成本与
chunk 数量成正比，而不是与它的长度成正比：一个 512 位置的 chunk 遍历库一次
（流式 110.97 GiB），而三个 128 位置的 chunk 遍历了它三次（125–136 GiB）。buffer
线性增长，在 512 时总共 ~250 MB，从固定的 scratch 预留里出。

**分组 expert kernel 现在赢了，而之前的判定错得有理有据。** 在合成矩阵上重新测量
（`--cvt`，它还会拿它对着单行 kernel 校验到 0.0）：

| group | N x single-row GEMV | grouped GEMM | ratio |
|---|---|---|---|
| 1 | 80 GMAC/s | 72 GMAC/s | 0.91x |
| 2 | 77 | 117 | **1.51x** |
| 4 | 74 | 133 | **1.78x** |
| 8 | 77 | 143 | **1.85x** |

第一版输，是因为它把累加器留在栈数组里，并且每个激活的每个块都做一次水平归约。
把它拆成每个激活两个向量累加器、每组寄存器四个激活，就是这样而已。它现在是默认；
`MM_PREFILL_GEMM=0` 回退。一个真实 chunk 里有三分之一的 expert 只被请求一个位置
（profile 里的 `fallback`），对它们来说单行 GEMV 仍然是正确的 kernel——那条路径
保留。

**剩下的是什么，按大小排序**（每 token，512 chunk）：`io` 108 ms，`experts`
86，`proj` 81，`wo` 33，`attn` 16。`io` 占 ~31%，就是正在被读的 expert 库：
32.6 s 内 110.97 GiB 是 3.4 GB/s，而 `--io` 显示设备本身对 10 MiB pread 能维持
8.5–12.8 GB/s，所以差距是真的，但字节无法回避——216 GiB 的 expert 对 68 GiB 的
pin cache。下一个杠杆是 FP8 dense GEMM 的寄存器分块：它跑在 ~82 GMAC/s，而它的
内层循环每次激活加载只做一次 FMA，所以一个 4 行 x 4 激活的 tile（16 个累加器，
一次激活加载在四行上复用）应该值 `proj`+`wo` 的大部分。

### 长 prompt prefill，第三部分：199 -> 77 s，以及三项输掉的改动

同一个 302-token prompt，`tools/mm_run`，top-2，`--context 8192`。attention 修复
之后的那一轮通过看*形状*而不是看阶段走得更远：一个按形状的计数器
（`mm_fp8_stats_print`，`MM_PROF=1`）显示是哪个矩阵在付账，而它找到的三件事里有
两件在从未被批处理过的代码里。

| | prefill | tok/s |
|---|---|---|
| start of this round | 199.2 s | 1.5 |
| attention vectorised, `MM_CHUNK` 512 | 104.2 s | 2.9 |
| FP8 dense GEMM tiled 4x4 | 89.1 s | 3.4 |
| shared expert + dense MLP batched per chunk | **76.7 s** | **3.9** |

**FP8 dense GEMM 在激活加载上是 issue-bound 的。** 它的内层循环每次激活加载只做
一次 FMA，并且每条权重行都把整个激活矩阵重读一遍——在 8192x6144 的矩阵上为
15 G MAC 产生 60 GB 的 L2 流量。按 4 行 x 4 激活分块，配 16 个累加器（四个权重
向量和四个激活向量同时活着，正好放进 32 个 NEON 寄存器），每次加载现在服务四次
FMA：单独跑 **96 -> 400 GMAC/s**，对一个同时覆盖尾部的朴素三重循环是 8.3e-6 的
相对误差。`--f8 N K n` 能复现这个数字。

**还有两个完整矩阵仍在按位置跑单行 GEMV。** shared expert（sw1/sw2/sw3）和三个
dense 层的 MLP 是在一个 `for (t = 0; t < n; t++)` 循环里调用的，于是一个 302-token
的 chunk 光 shared expert 就发出了 34,428 个单行 GEMV：14.8 s，67 GMAC/s，是整个
prefill 里最大的单项计算——而它在阶段 profile 里不可见，因为它落在 `proj` 里面。
作为 chunk GEMM，同样的工作跑到 228 GMAC/s。这与 projection 是同一个教训，只是高
一层：在 n=302 时权重字节被读一次、用 302 次，而这只有 GEMM 做得到。

有三项改动被测过并否决，这与胜利同样有价值：

* **每个 expert 组八个激活**而不是四个：n=8 时是 125 GMAC/s 对 143。十六个累加器
  加上解码溢出 32 个寄存器。
* **`stream.c` 里的 blob 回收池**：expert 读取看起来像是在为每次 miss 一次 mmap
  付账（3.4 GB/s 对 `dd` 的 6.7-7.0），所以 blob 被留在一个有界的 free list 里。
  prefill 没动（89 -> 93 s，即噪声）。那个 `dd` 对比是错的对象：那些文件是
  page-cached 的，而 stream 的 fd 带着 `F_NOCACHE`，所以 3.4 GB/s 就是关掉缓存
  的设备。回滚了，而不是"因为它应该有用"而留着。
* **把 FP8 dense 权重拷进匿名内存**以保证常驻：footprint 82 -> 92 GiB，而那
  10 GiB 直接从 expert 读取所依赖的 page cache 里来。io 从 102 变成 116 ms/token，
  净时间变差。那些映射页本来就已经常驻。

剩下的现在由不做算术的东西主导：**每 token io 96 ms（38%）**——110 GiB 的 expert
库在 3.4 GB/s，也就是设备在 `F_NOCACHE` 下的水平，对着 216 GiB 的库和 81.5 GiB
的 cache——然后是 experts 93，proj 24，attn 16，wo 10。计算侧已经接近这些 kernel
在 12 个核心上能做到的；一个*冷* 302-token prompt 的 prefill，与读完它所需的
experts 所花的时间相差在 2 倍之内。

关于那个 `io` 项还有一个负面结果，因为那是人们最先想到的：**把一层的 expert 读取
按文件偏移排序，让设备看到顺序扫描，反而更慢**，同一个 prompt 上 3.65 GB/s 对
3.98（32.0 s 内 108.83 GiB 对 29.0 s 内 107.50 GiB）。一个 10 MiB 的 pread 已经
大到随机性不是问题，而十二条各自走自己区域的流胜过一次扫描。所以 `io` 项就停在它
所在的位置：3.4-4.0 GB/s 是 `F_NOCACHE` 读取在这台机器上能给出的，而越过它的
唯一办法是更少的字节（cache 命中）或不同的硬件——不是一个更聪明的读取循环。

这就把 FP4 expert kernel 留作桌上最后的 2 倍。它跑在 139 GMAC/s，大约是这个核心
持续 f32 FMA 发射率的 70%，所以每个 MAC 的操作数必须降下来：每 16 个元素的 nibble
解码加四次 f32 FMA，应该变成 `bfdot`/`vfmmla`，它们每条指令做 8 个 MAC 而不是 4
个。那是对 kernel 及其 LUT 布局的重写，不是微调，而它是唯一还剩两倍的项目。

### FP4 kernel 最后的 2 倍：测过了，它不在那里

expert kernel 的计划是把每 16 个权重的四次 f32 FMA 换成 bf16 点积（每条指令 8 个
MAC 而不是 4 个）。`tools/bfdotbench` 的存在就是在写 kernel 之前检验这个前提——
单线程，同样的循环形状，`make bfdotbench`：

| kernel | GMAC/s (1 thread) | vs f32 |
|---|---|---|
| f32 `fma`, 4 accumulators | 16.8 | — |
| bf16 `bfdot` | 23.1 | **1.37x** |
| fp16 `vfmlal` (widening) | 38.5 | **2.29x** |

两件事浮现。第一，**bf16 是错误的目标**——它在这里只比 f32 快一点点，而且它的
尾数是最窄的。第二，**fp16 在点积上真的有 2.3 倍**，而且它本来就是最适合这个
kernel 的格式，因为现有的 FP4 解码已经产出 f16 的位模式（`tbl` 给出 f16 的高
字节，一次移位就补全）——所以"解码到 f16 并用 widening MAC"连 f32 转换都省掉了。

然后指令数说不。每个 (weight row, activation) 每 16 元素块：

* 今天，f32：4 次 load + 4 次 mul + 2 次 add + 2 次 scaled-FMA = **12 ops**
* 用 fp16：2 次 load（一半字节）+ 4 次 widening MAC + 2 次 add + 2 次 scaled-FMA = **10 ops**

所以是 1.2 倍，不是 2.3 倍。那 2.3 倍是 *MAC* 速率，而这个 kernel 不是 MAC-limited
的：它是 issue-limited 的，因为每个块都需要加载它的激活向量，并把它的部分和按
每块的 e8m0 因子缩放。bench 测的是纯点积，两者都没有。在每组里增加激活是唯一能
摊销解码的办法，而八个激活已经测过并溢出 32 个寄存器（125 对 143 GMAC/s）。在
它上面再做行分块，出于同样的原因需要 4 x 4 x 2 个累加器。

所以 expert kernel 就停在 139 GMAC/s 不动，而诚实的说法是：这个模型在 12 核上的
计算侧已经做完了：io 96 ms/token，experts 93，proj 24，attn 16，wo 10，route 6。

仍然会回本的那一项结构性改动根本不是 kernel：**预读和 expert 计算是串行的。**
`cwick_stream_expert_reserve` 发起读取，然后在第一个 GEMV 之前等它们全部完成，而
那些读取线程全程闲着，因为 3.4 GB/s 是设备的上限，不是核心的。两三个线程读这一层
剩下的部分，另外十个计算已经落地的 expert，能掩盖那 96 ms 的大部分。那需要每个
blob 一个完成信号和第二条 dispatch 路径，也就是一个设计，不是一次微调——而它是
唯一还剩两倍的项目。

### 预读和计算现在重叠：io 96 -> 50 ms/token

最后那个还剩两倍的项目不是 kernel。`mm_expert_prefetch_layer` 预留一层需要的每个
blob，把它们全部读进来（阻塞），然后才让第一个 GEMV 开始——于是十二个计算线程在
250 ms token 里的 96 ms 中坐着不动，而唯一在工作的是一个能维持 3.4-4.0 GB/s 的
设备。每一次让那次读取*更快*的尝试都失败了（blob 池、偏移排序、常驻副本），因为
那个速率就是硬件的。可修的是：在那段时间里别的事情什么都没发生。

这一层的读取现在被拆开：前半照旧在池上，后半交给四个 helper 线程**直接**读——不
经过池，于是它们与 GEMV 并行而不是排在它们后面——前半的 expert 在它们工作时被
计算。并集循环在后半开始的并集索引处（`umid`）汇合，并在循环之后再汇合一次作为
兜底。

| | before | after |
|---|---|---|
| io | 96 ms/token | **49-55** |
| experts (now includes join waits and contention) | 92 | 106-108 |
| token | 250 ms | **219-227 ms** |
| 302-token prefill | 75.5 s | **66-68 s (4.4-4.6 tok/s)** |

输出逐位未变（`the quick brown fox ... lazy` -> ` dog` 19.8344），`make test`
通过，而 decode 未受影响：重叠以 `n > 1` 为条件，因为一个 decode step 每层四个
expert、一个位置，交接就成了 57 x 4 次白费的线程 spawn。

路上测了三个调优：

* **四个 helper，不要更多。** 六个测得 74.9 s 对 66.1：这台机器是 12P+4E，所以
  六个 reader 加十二个池线程会超订，`experts` 从 106 变成 115，而 `io` 从 49 变成
  56。helper 是 I/O 阻塞的，但不是免费的。
* **50/50 切分。** 后半必须在前半的计算时间内完成；把两者都减半是自然的第一个
  猜法，而它留下了 55 ms 暴露在外，所以它大致是平衡的，而不是调得很差。
* **`MM_IO_OVERLAP=0`** 回到串行路径，这个值得保留：它正是这个比较的参照。

### Decode：速率停在哪里，以及为什么 prefill 的技巧不能搬过来

prefill 实测 4.5 tok/s（302 个 token 用 66.9 s）。decode 没有：浅
prefill 之后实测每步 513-577 ms（1.7 tok/s），302-token prefill 之后 857-1058 ms
（1.1 tok/s）。两者之间的差别不是代码，是 cache 状态：`io` 在前一种情况下是
109-210 ms，在后一种情况下是 250-383，因为 prefill 刚走完整个 expert 库，把
decode 正在用的东西驱逐了。在一个真实的 API 请求里 prefill 总是先发生，所以用户
看到的是第二个数字。

一个 decode step 移动 ~22 GB 的权重：`wq` 2.87，`wo` 2.87，`wk`/`wv` 0.36，
shared expert 3.2，228 个 routed 矩阵 12.9。一步 250 ms 意味着整个 step 持续
**88 GB/s**，而在这台机器 123.5 GB/s 的天花板下，kernel 测得 78 GMAC/s（fp4 GEMV）
和 114 GMAC/s（fp8 GEMV）。所以它不是不可能，但它不给停顿留任何余地：每条流都得
跑到峰值的 ~71%，而 SSD 必须隐形。

在一个偏暖的 step 上这 513 ms 去了哪里，以及每一部分本来应该是什么：

| phase | now | at 100 GB/s | note |
|---|---|---|---|
| io | 109-210 | 0 | 5-14 misses x 19 MB at 3.4 GB/s |
| experts | 165-198 | ~130 | 12.9 GB, 65-78 GB/s, 684 separate GEMVs |
| proj | 83-139 | ~32 | 3.2 GB but only 23-38 GB/s |
| wo | 31-34 | ~29 | 2.87 GB at 84 GB/s, already near peak |

`proj` 是那个离群值：同样形状的 `wo` 跑到 84 GB/s，而 `proj` 跑在 23-38。它每层
五个 GEMV（每 step 285 个），而按形状的计数器显示小的那些全是开销——`[512x6144]`
180 次调用花 18 ms（30 GB/s），`[128x6144]` 60 次调用花 4 ms（11 GB/s），而涉及的
字节只有 3 MB。

prefill 的重叠**不能**搬过来，而这是实测而非假设的：把每次 decode miss 都 route
到四个 helper 线程，先算常驻的 expert 并在边界汇合，是无效的，因为 `overlap` 的
条件是 `n > 1`。强行打开它无论如何都是错的——一个 decode 层有四个 expert 和
一个位置，没有后半可以重叠，而四个串行 reader 比原来做读取的十二线程池更慢。
decode 的 `io` 需要一个 cache 层面的答案（把 hit rate 从 94-97% 提到 100% 附近，
或者别让 prefill 驱逐 decode 的工作集），不是一个调度层面的答案。

会推动 decode 的两项改动，按顺序：

1. **把五个 projection 合并成每层一次 dispatch**（每 step 285 -> 57 次池调用）。
   `proj` 是 overhead-bound 而不是 bandwidth-bound，所以这值 513 ms 里的 ~50-90 ms，
   且没有任何算术变化。
2. **把一层的十二个 expert GEMV 合并成一次 dispatch。** 同样的论证，规模更大，流
   也更长：684 -> 57 次调用，12.9 GB 的连续读取而不是 19 MB 的爆发。

两者都需要一个把全局行索引映射到 (matrix, row) 的 range 函数，那是机械工作；两者
都不改变任何结果位。

### Decode：算术说这台机器就是上限

302-token prefill 之后的 60 个 decode step（真实 API 请求具有的形状）：
**1.4 tok/s**，均值 694 ms，单步 436-820 ms，取决于它们是否碰到 SSD。输出未变
（` dog` 19.8344）。

一个 step 移动 22.2 GB 权重：12.9 是 routed experts，2.87 `wq`，2.87 `wo`，
3.2 shared expert，0.36 `wk`/`wv`。所以实测有效速率是
22.2 / 0.694 = **32 GB/s**，对着 123.5 GB/s 的天花板。一步 250 ms 意味着
**持续 89 GB/s**。

两个分解，都来自测量：

1. **`io` 占这个 step 的 109-239 ms**（hit rate 94-97%：19 MB 的 5-14 次 miss 在
   3.4 GB/s 下）。把它*完全*去掉还剩 524 ms = **2.0 tok/s**。这就是 step 里完全
   没有存储时的上限。
2. 那 524 ms 以 42 GB/s 的复合速率移动 22.2 GB，而各阶段单独测时跑在 60-84 GB/s。
   一步 250 ms，需要在已经位于这台机器峰值 50-68% 的流上再要 **2.1 倍**。

所以进一步调优在这里收不掉它：缺口是 bandwidth-bound kernel 上的
2.1 倍加上一个完美的 cache，而机器两样都没有。越过这里的路只有两条：(a)
**speculative decoding**——checkpoint 附带 `dspark/`，一个 7-token 的 block draft
model，这是每次权重读取拿到多个 token 的唯一办法——或者 (b) 不同的硬件／一个真正
能用上 GPU 的实现（这里的 Metal kernel 测得 33 GB/s，对着一个 ~400 GB/s 的设备，
因为它们没有分块）。

记录下来，好让下一次会话不再重复这个循环：prefill 完成了（4.5 tok/s），decode 是
1.4，而差距不是 bug。

这一轮还有一个负面结果，出于同样的理由：把五个 projection 合并成一次池 dispatch
测得 proj 92-97 ms，而之前是 83-139，即在噪声之内，于是回滚了。每次调用开销的
假设是错的——`[512x6144]` 跑到 30 GB/s 不是因为 dispatch 花了 50 us，而是因为
512 行摊到十二个线程是每人 42 行的工作。

### max_tokens 是上限，不是需求：clamp 它，不要 400

一个客户端对着 32768 位置的 MiniMax server、带 8K prompt 发送
`max_tokens: 32768`，得到的是

```
prompt 8315 + max_new 32768 at position 0 exceeds the 32768-position context window
```

而且没有答案——而它想要的回复只差几十个 token。GLM 路径早就学过了这一点（它的
decode 预算被 clamp 到 KV 窗口还能容纳的量，`test_max_new_clamped`），所以这个 bug
其实是两个较新的 backend 在最老的 backend 已经解决过的同一个请求上不一致。

`cwick_gen_window(pos, prompt_len, &max_new, capacity)` 现在为三者统管这条规则：
当 `max_new` 溢出时把它减到 `capacity - pos - prompt_len`，只在 *prompt* 放不下时
才失败——这两种情况的修法不同（降低 max_tokens 对更短的 prompt 或更大的
`--context`），而旧消息把它们混为一谈。端到端实测：失败的请求变成 `200`、
`content="Hi there, friend!"`、28 个 token，日志说

```
max_tokens 32768 clamped to 32570: prompt 1 at position 197 leaves 32570 of 32768 positions
```

被 clamp 的预算是一个*上限*，不是一个长度，所以进度行不再把它当作分母：
`decode 6 tokens (context cap) — 895 ms/step` 而不是 `1/32570 (0.0%)`，后者读起来
像一个卡住的请求。GLM 在 `maybe_progress_at` 里早已有同样的推理。

有一个后果值得说明：遵从一个很大的 `max_tokens` 现在可能意味着一次很长的运行
（1.3 tok/s 下 24151 个 token 是几小时），而以前它是一个即时的错误。这是正确的
取舍——`max_tokens` 是客户端要求的，而模型早在 EOS 就停了——但一个想要短答案的
客户端应该发一个小的 `max_tokens`，而 server 应该继续在每一行进度的同时报告 ETA，
好让上限是可见的。

### prefill 进度：那一行存在过，但不在工作发生的时候

MiniMax 的 prefill 进度有三处问题，而只有第一个真的与打印有关。

**它只在 chunk 边界触发。** `mm_prefill_progress_done` 每个 `MM_CHUNK` 调用一次，
而一个 512 位置的 chunk 在这里要 ~140-170 s，于是日志在 21:41:01 显示
`1024/8321`、在 21:43:51 显示 `1536/8321`——每 ~170 s 一行。一个能装进单个 chunk
的 prompt（任何 512 token 以下的请求，也就是大多数）在 prefill 结束之前**什么都
不打印**：一个 201-token 请求产出的第一行是 `1/1 tokens (100%)`。GLM 路径早已用
层计数器解决了这个问题（`maybe_progress_at(..., layer, nlayer)`）；MiniMax 的循环
没有等价物。

**层 gauge 被钉在最大值上。**
`cwick_metrics_set_prefill(done, total, layers, layers)` 把 checkpoint 的层数当作
*当前*层传进去，于是 `/v1/metrics` 从第一秒起就报告 `cwick:prefill_layer 60` 配
`prefill_layers 60`——仪表盘会把它画成"快完了"，
而实际还有 30 分钟的工作。token gauge 有同样形状的问题，只是方向相反：它每 2.5
分钟跳一次 1024 -> 1536，而在整个单 chunk prefill 期间读作 0。

**那一行里没有任何东西说它在哪。** 没有层，没有 footprint，于是"慢"和"卡在 SSD
上"无法区分。GLM 那一行带着内存尾巴，正是为此。

修复是 `mm_prefill_progress_layer()`，在一个 chunk 内每层结束时调用：每 8 层或
每 `CWICK_PROGRESS_MIN_MS`（默认 5 s）一行，以先到者为准，同样的值发布到 gauge。

```
[prefill] ~38/333 tokens (11.4%), layer 7/60 — 6.3 tok/s, eta 47s
[prefill] ~310/333 tokens (93.1%), layer 56/60 — 4.5 tok/s, eta 5s
[prefill] 333/333 tokens (100.0%), layer 60/60 — 4.5 tok/s, eta 0s
```

chunk 内的 token 计数是一个**估计**（`~`），因为一个 chunk 的各位置是随层一起推进
的：`done + layer/nlayer × n`。没有它，一个单 chunk 的 prompt 会连着两分钟打印
`0/201 (0.0%)` 然后跳到 100%，这与完全没有进度行的信息量相同。在最后一层时估计
变成精确的，而 `_done` 在边界处打印精确数字，所以 `~` 总是在下一个 chunk 开始之前
消失。

decode 不受影响：`mm_prefill_progress_layer` 除非 `begin()` 武装了它，否则立即
返回，而 `mm_forward_chunk` 只在 `n > 1` 时调用它。

在一个 333-token 的 prompt 上实测：15 行，间隔 3-6 s，eta 从 47 s 单调降到 0 s，
`cwick:prefill_layer` 在同一个原本一直停在 60 的窗口里从 40 走到 48。

有两件事这**没有**修复，都比日志更大：

* 一个 8321-token 的 prefill 在 4.5 tok/s 下仍要 ~30 分钟，即这台机器的天花板。
  我打断的那次运行已经流式读了 **616.65 GiB**（整个 checkpoint 的 2.7 倍）且 0 个
  decode step——它全程都在工作，只是没有途径说出来；
* 一次长 prefill 独占唯一的生成槽位，所以后来的请求排队，然后拿到
  `429 queue full`。那是一个调度决策，不是日志决策。

## 27. MiniMax 的 decode：先量天花板，再改三处，值 2.1 倍

这个后端的改进顺序本来是从 GLM 的剖面借来的：那边 attention 占 275 ms 里的
117 ms，所以把 attention 核搬上 GPU 是第一条。一次 `MM_PROF=1` 就把这条否掉了：

```
mm prof  974 ms: proj 137  attn 4  wo 41  route 7  experts 391  io 359  dense 11
```

attention 只占一步的 0.4%。剩下的是 `io`（等 SSD）和 `experts`（fp4 GEMV），
而这两者并不独立：字节数决定前者，缓存决定前者有多少能落到后者身上。

**先量机器，再谈内核。** `make mmdecodeprobe` 给出一步要对标的两条天花板。常驻内存
读带宽：单线程 37 GB/s，4 线程 91，8 线程 119，16 线程 122——天花板在 8 线程就到了，
再多只是加争用。shard 读是另一回事，而且取决于读的是**哪些**文件：一个 decode 层
的未命中落在同一个 shard 里，同 shard 速率只是跨 shard 的一个零头，而读大小和每线程
独立 fd 都实测无效。（这个工具一旦文件变热，绝对值就会漂，所以下面的决策依据是引擎
层面的 A/B。）

然后三处改动，每一处都和前一版对比、同一场会话、同一个二进制、同一个 prompt、
23 个 decode step、`--context 2048`：

| | 改动 | tok/s |
|---|---|---|
| A | 原状：串行读、每个 blob 单独分配、80% 缓存预算 | 0.686 |
| B | + 定长 blob 竞技场（复用块，不再 malloc/munmap） | 0.778 |
| C | + 读按**任务数**派发，而不是按行数 | 1.020 |
| D | + 缓存预算封顶 `host / 6`，不再走 80% 计划 | **1.450** |
| E | C 但保留 80% 计划（63 GiB） | 1.122 |

**一、读必须按任务数派发。** `cwick_pool_for` 按 `CWICK_POOL_MIN_WORK` 切分，而那
是**每行**的成本阈值，于是一个 decode 层的三四个未命中落在阈值之下，全部退化成调用
线程上串行 pread。整个读相位这样只有 4.9 GB/s，四个在飞时 10.9 GB/s。
`cwick_pool_for_items` 本来就是为这件事写的（`pool.c` 的注释里写着），只是这里没用。

**二、blob 竞技场只给普通内存，且只给它。** 一个 checkpoint 里每个路由专家矩阵形状
相同，所以 pin 缓存的分配单位是个常数，每步约 600 次 malloc/munmap 可以换成从少数
几个大映射里切块。这在 MiniMax 上值 +13%。把同一个竞技场给 GLM 是 **17% 的损失**
（2.09 -> 1.74 tok/s，可复现），因为那边的 blob 是 GPU 可见的，Metal 竞技场有自己
按 8 MiB 单元调好的分配器——把它们并成 1 GiB 映射会改变那个竞技场的增长与
wired 行为。所以这个池按后端 opt-in（`cwick_stream_enable_blob_pool`，由
`mm_model_open` 调用），另两个后端保留它们数字被量出来的那条路径。

**三、pin 缓存不该吃掉整个计划。** 这是最大的单一杠杆，而且是一次减法：

| 缓存 | 14 GiB | 18 | 22 | 30 | 44 | 63（80% 计划） |
|---|---|---|---|---|---|---|
| tok/s | 1.39 | 1.58 | **1.62** | 1.58 | 1.16 | 1.08 |
| 命中率 | — | — | 67% | — | — | 78% |

更大的缓存确实流式读更少字节、也确实提高命中率，而步时还是变长。过了 ~30 GiB 压缩器
开始收页，而一个页在送进 GEMV 前要先解压，代价和从 SSD 读一个页一样——同时那些根本
不碰缓存的相位（投影、attention）也跟着变慢。平台期是 18-30 GiB，`host / 6`
（128 GiB 机器上是 21.3 GiB）正好落在中间，并且在小机器上会等比例缩小，而不是假设
它们也有余量。

**试过但不在表里的**，因为每一项都量了，都输：

* 把变慢归因于 mmap 缺页：blob 100% 来自 pin 缓存（有计数器为证），0% 走文件映射；
* 给 blob 池加 wire（`mlock`）：压缩消失、GEMV 在 56 GiB 下仍是 117-123 ms 平稳，
  但端到端**更差**（1.45 tok/s）——wired 的 21 GiB 是进程其余部分用不到的内存；
* 竞技场用 2 MiB 超级页：GEMV 峰值 408 -> 312 ms，即给定 footprint 下的那个拐点没动；
* 上面任意一条配更大的缓存：就是表里的 E。

这里实测是 1.48；差距是每步字节数对上同 shard 的读天花板，不是内核。
下一个杠杆是唯一不需要更多带宽的方向：一次权重读取产出多个 token（这个 checkpoint
自带 `dspark/`，一个 7-token 块的草稿模型），并连同它的契约一起做——状态回滚到已接受
前缀、采样仍然正确、以及前缀快照必须连草稿状态一起存。

## 28. 块验证：机制本身，以及一个 n-gram 草稿器说明了什么

上面每一步都是靠"搬更少字节"换时间。只有一个方向完全不需要带宽：一次权重读取
产出多个 token，再在一次前向里验证它们。这一节是那套机制，并且故意配最便宜的草稿器，
这样"验证一个块要多少钱"就能和"谁来提候选"分开量。

**三块拼图。** `mm_model_forward_multi` 为一个 chunk 的每个位置各给出一个分布
（`spec_out`），因为候选 *i* 只能由位置 *i-1* 的分布判定；逐个位置取分布，代价正好
等于投机想省下的那部分。接受判据是 argmax 相等——只做贪婪：温度 > 0 时精确的投机
采样需要采样器逐位置过滤后的分布，而那是没有暴露出来的，近似做会静默改变输出分布。
回滚不需要代码：被拒的候选写进了 `pos` 之后位置的 K/V，下一个块会覆盖它。

**它是精确的吗？** 把块强制拆回单位置（`MM_SPEC=8 MM_CHUNK_CAP=1`），输出与同一请求
不开投机时逐字节相同。不拆则不然：n=8 走 chunk GEMM、n=1 走 GEMV，两者末位舍入不同，
近似并列的 argmax 就会翻转（`short` -> `shorter`）。所以这台引擎能主张的是"相对块
自身的算术是精确的"，而不是"与单行路径逐位一致"——prefill 路径一直有这个注意事项，
现在 decode 路径也有了。

**一个块要多少钱。** 一次 8 个位置的前向，专家字节约为单 token 前向的两倍：同一个
prompt 上 7.2 GiB/步 对 3.6 GiB/步。8 个位置的专家并集远小于 8 倍——相邻位置的专家
高度重叠——但它不免费，而这个数字就是每个草稿器要对的标尺。

**n-gram 草稿器买到什么。** 候选来自 token 流本身（尾部 1-4 gram 最近一次出现之后的
内容），所以成本为零。

| prompt | 接受率 | token/前向 | tok/s |
|---|---|---|---|
| 重复同一行五次 | 28/44 = 64% | 1.76 | 1.486 -> **1.612** |
| "解释天空为什么是蓝的" | ~0% | 1.00 | 1.163 -> 1.100 |

也就是说：自我重复的文本 +8.5%，散文 -5%。盈亏平衡点是 token/前向 > 2，即块为 8 时
接受率约 85-90%，而且即使草稿器完美，上面那个并集成本也把上限压在 2 倍附近。

**这就是对真正那个问题的回答。** 这个 checkpoint 自带 `dspark/`，一个 7-token 块
草稿器（5 层、dense FFN、秩 256 的 markov 头、confidence 头，自己没有 embed 和 head
——复用目标模型的，并消费目标模型第 3/17/31/45/59 层的 hidden state）。移植它本身是
一个研究项目：checkpoint 里没有参考实现，也没有它的块语义规格。上面的测量给出了它
必须跨过的线：在普通文本上每前向 ≥ 2.5 个 token。对一个训练过的草稿器这是合理的
目标，对 n-gram 则不可能——所以这一步停在这里：机制留在代码里，`MM_SPEC` 默认关闭，
下一个尝试有一个可以对标的数字。

同一批会话里一个无关的观察：同一条 greedy 请求，命前缀快照（恢复 210 个位置）与
从头 prefill 同一个 prompt，产出的 token 不同。两次都是 greedy、同一种子，也都没开
投机。记在这里是因为它会让下一个比较两次运行、期待确定性的人误判；原因未知。

## 29. 工具调用：块还没写完时，流上该显示什么

工具调用是这台服务最长的一段沉默。`cwick_stream_segment` 把 `<tool_call>` 的正文
既不归类为 reasoning 也不归类为 content——这是对的，它是待解析的材料——于是从开标签
落地到请求结束，客户端什么都收不到。在慢的 checkpoint 上这是几十秒，聊天气泡看起来
就是卡住了；而解析后的 `tool_calls` 只能在结尾给出，因为块没闭合之前它根本产不出来。

**修法是加一个字段，不是加一份副本。** 这些字节现在以 `delta.cwick_tool` 发出，是
`content` 旁边一个非标准字段，不认识它的客户端会直接忽略。web UI 把它渲染成带脉冲
的等宽卡片，解析后的 `tool_calls` 一到就清空，所以原始形态与解析形态不会同时出现。
发出前会剥掉 framing，而"还可能长成 framing"的那截尾部会被扣住
（`cwick_tool_ns_tail`），绝不显示半个标记——与 content/reasoning 切分一直用的扣尾
规则是同一条。

**为什么某个 checkpoint 的工具块需要一层翻译。** 那个 checkpoint 在同一个
`<tool_call>` 包裹里写 XML，并且在**每一个**结构性标签前面加一个 namespace token
（它 added-token 表里的一项，由它的模板拼进去）：

```
]<]minimax[>[<invoke name="add_numbers">]<]minimax[>[<a>137</a>]<]minimax[>[<b>258</b>]<]minimax[>[</invoke>
```

模型是在遵守它自己的规格；朴素扫描会把 framing 读进工具名，然后把整轮生成当作未声明
的工具拒绝掉——真实的一轮就是这样失败的。与其让解析器再学一套语法，`translate_invoke`
把这个块改写成解析器本来就认识的 `<arg_key>/<arg_value>` 形态：一个解析器，一套边界
情况。

其中四个 bug 只有喂真实字节的探针能看见，单元测试看不见：原始开标签被输出了两次，
于是名字变成 `<tool_call>add_numbers`；扫描子元素的起点落在 `<invoke ...>` 的 `>`
上，于是所有参数被丢掉；另一种写法带上同样的 framing 时看起来没闭合（`</tool_call>`
的 `strstr` 隔着 token 永远匹配不上）；以及 token 落在**值里面**时，值变成 `137]`、
而 `]minimax[` 被当成下一个参数的键——token 里含 `<`，所以任何"停在标签边界"的扫描
都会停在 framing 内部。最后一条正是现在改成"先在整个回答里去掉 token，再让任何语法
看它"的原因：framing 出现在哪里都是 framing，一条规则胜过每个位置一条规则。四条现在
都是 `tests/test_cwick.c` 里的断言，包括实测到的漂移：模型先开一个块、写一段散文、
然后才开它真正填的那个块——这种情况必须产出一个已声明的调用。

真实一轮的端到端实测：整次调用共 24 个 `cwick_tool` 帧，其中没有任何 framing，
`finish_reason: tool_calls`，解析出的参数是 `{"a":137,"b":258}`；而同一个请求在这
之前返回的是 `{"a":"137]","]minimax[":"[",...}`，并被当作未声明的工具拒绝。

## 30. 一轮生成在浏览器这一侧：三个活动区，以及运行它写出来的东西

一轮会同时产出三样东西，而气泡以前是一次只显示一样：思考会被画出来，但进行中的工具
调用不会（它被塞在正文容器里），而**解析后的**调用在整条消息最后重绘之前根本不画。
现在三者都是正文之上的独立活动区——思考面板、工具容器（里面是正在写的块或解析后的
卡片）、正文——由同一份快照重绘，谁都不等谁。

流式绘制有两个必须做对的细节：容器是懒创建的（每个直接回答都挂一个空思考框，比没有
更糟）；当思考面板与工具容器同时存在时，思考面板插在工具容器**之前**，这样顺序是模型
产出的顺序，而不是 delta 恰好被处理的顺序。

代码块现在能把字节带出页面：每个块都有复制和下载；**是页面的块**多一个运行——语言是
`html`/`svg`/`xml`，或者没写语言但首标签是 `<!doctype html>`/`<html`。运行会打开一个
带沙箱 iframe 的弹层（`allow-scripts allow-modals allow-forms allow-popups`，故意**不**
给 `allow-same-origin`）：片段能跑，但读不到本源的 storage、cookie 和会话 token。
在真实浏览器上对着同源父页面验过：脚本跑了，父页面 DOM、localStorage、cookie、XHR
全部被拒。

预览做成弹层而不是内联 iframe，只有一个原因：气泡每个 token 都会重绘，内联 iframe 会每
token 重新加载一次——而把已有 iframe 在 DOM 里挪位置也会重载。弹层在气泡之外，重绘
不受影响；关闭时**换掉** iframe 节点，这是唯一能可靠停掉片段留下的定时器的办法（把
`srcdoc` 置空仍然留着一个活文档）。

验证方式是用合成 SSE 流在 DOM 里驱动真实的 `web/app.js`（28 条断言）：流中途三个活动区
同时在、解析后的卡片替换掉进行中的块、html 块有运行/下载/复制而 python 块只有复制/
下载、沙箱串、以及弹层的复制/下载/关闭/Esc 各条路径。

## 31. 同一个 prompt 跑两次结果不同——以及为什么

MiniMax 后端以前不是确定性的。同一个 prompt 两次 `mm_run`，下一个 token 的 logit
不一样（同一个 token 18.44 对 19.13），两次连读了多少个专家矩阵都不一致（4458 对
4406）。也就是说，这个后端此前报出的每一个 decode 数字，都是在一台会在两次采样之间
悄悄改答案的机器上量的。

### 怎么定位的

猜没用，量才有用。两个由环境变量控制的调试辅助（默认关闭）把「模型自相矛盾」变成
了一行：

* `MM_TRACE_HASH=1` 在每层每个阶段之后对残差流做哈希，打印
  `trace L<层> <阶段> <哈希>`。两次运行对比显示第一次差异总在 routing **之后**、
  本层残差合并**之前**——也就是专家 FFN 内部，而两侧 router 的选择完全一致。
* `MM_TRACE_W=1` 在每个专家矩阵被交给 GEMV 的那一刻对其字节做哈希。这一步把问题
  钉死了：第 36 层同一个专家的 `up` 和 `down` 两次哈希相同，而 `gate` 不同。变的是
  权重的字节本身，不是算术。

### 原因

`cwick_stream_expert_reserve()` 交给调用方一个可写缓冲区，读取发生在**之后**——
整个设计就是在一层里先在一个线程上把所有的 miss 占好槽，再丢到线程池上去 pread。
但被占的槽是一个普通的缓存条目（`heat = heat_avg`），于是同一个循环里下一次 reserve
如果需要腾地方，就可能把**自己这层还没派发读取**的槽驱逐掉。调用方的 `dst` 随即指向
一块已释放的内存，下一次准入又把同一块发了出去，两次读取落在重叠的内存上。模型照跑
——大部分字节是对的——但谁最后写入取决于线程调度，所以看起来像非确定性，不像损坏。

修复就是一个状态位：被 reserve 的槽在读取落地前是 `pending`，pending 的槽永远不是
驱逐候选（探测窗口的 victim 选择、采样 victim 选择、`cache_make_room` 的兜底扫描，
三处都排除）。`cwick_stream_expert_settle()` 在读取被结算后清除它；每个被 reserve 的
槽要么 settle（成功）要么 abort（失败），不会有永久受保护的槽。另外
`cwick_stream_expert_put_read()` 有两条失败路径拿着缓存互斥量返回，会锁死下一次
lookup，也一并修了。

验证：同一 prompt 连续三次运行，5957 个被 trace 的阶段全部一致，包括每一个权重字节
哈希，专家读取次数也相同。

### 对 profile 的影响

被损坏的条目同时也是浪费的带宽。现在 decode 一步读 2.55–4.38 GiB、命中率 67–81%，
read-ahead 阶段 277–384 ms（总步时 574–692 ms）；修复前对应的几行是 4.31 GiB / 67%
和 420 ms / 714 ms。这两组**不是**受控对比——两次运行解码出的 token 都不一样，而那正是
这一节要讲的事——但方向正是机制所预测的：一个在读取之前就被驱逐的槽，等于付了钱又把
读到的字节扔掉。

## 32. 唯一的归约顺序：验证过的块必须是同一次计算

这里的投机解码是「读一次权重、同时给多个位置打分」，然后保留模型自己 argmax 出来的
候选。要让它成为加速而不是换了一个模型，批处理那一遍必须返回与它替代的逐步运行
**逐位相同**的结果。

以前不是，所以 `MM_SPEC` 默认关着。原因在 `src/mm_model.c`：FP8 密集核对同一个乘积
有**四套**归约树——单激活路径在 32 元素块内先求和再乘块缩放；行分块的 prefill 路径
把缩放折进每个元素后跨整行平铺累加；两条尾巴路径（行不是 4 的倍数的余量、激活不是
4 的倍数的余量）用一对间隔 8 元素的累加器。同一个矩阵喂给其中任意两个，会得到两个
答案：`--f8` 现在能测出 n=16 时 131072 个值里 130805 个不同。

重排 6144 个乘积只动最后一位或两位，但跨 60 层足以翻转一个接近并列的 argmax——这就
是投机版输出 `short` 而普通版输出 `shorter` 的原因。

修复是加一道阈值，不是重写。`MM_FP8_EXACT_N`（默认 8）是愿意为「一致」买单的最大
`n`：不超过它时每个激活都走单激活内核，而单激活内核天然就是参考——因为一个 decode
步**就是**那个内核。超过它（prefill 块，其输出永远不跟任何东西比对）仍走行分块内核。
这道阈值在 n<4 时还是一次提速：分块在 n=2 量到 2.4 ms，而把融合路径跑两遍只要 1.4 ms。

代价，用 `--f8 8192 6144 <n>` 同场量得（每个数字是整个矩阵，不是每个激活）：

| n | 改前 | 精确路径 | 分块路径（未改） |
|---|---|---|---|
| 1 | 0.7 ms | 0.6 ms | - |
| 2 | 2.4 ms | 1.4 ms | - |
| 4 | 1.4 ms | 2.8 ms | 1.4 ms |
| 8 | 2.0 ms | 4.5 ms | 2.0 ms |

也就是说 8 长度的验证块，密集 GEMM 的时间约为它替代的那一单步的 2.3 倍。这是「一致」
的价钱，也正是决定某个块长值不值得的数字——即成本模型该用的数字，而这个后端目前还没有。

`--f8` 现在每次运行都会打印一条逐位一致性检查：块里每个位置对同一个位置单独算一次，
用的是满尾数合成激活（2 的小整数倍在任何顺序下舍入都相同，怎么测都会通过）。n<=8 时
报 `0 of 65536 values differ`，n>=16 时明确失败，阈值不会再悄悄漂移。

FP4 专家核有同一类问题，答案形式不同：它的分组内核现在用融合乘加合并一个打包字节的
两个半字节，与逐行内核逐指令对齐——同样的 lane 映射、同样的两个累加器、同样的块
顺序、同样的最终归约。`--cvt` 每次运行都检查，报告不是逐位一致的行数（当前 0/3072）。

## 33. 块验证现在逐位正确，但仍然不划算——已量、已关闭

把两套归约顺序统一之后（§32），「投机在这个模型上划不划算」终于能靠测量回答，而不是靠
猜接受率。`tools/mm_run --ab N` 在同一个进程、同一份已加载的模型上跑三趟——普通贪婪、
投机、再普通贪婪——并逐 token 比对。第三趟才是关键对照：那时 pin cache 已经热了，而这个
模型上「缓存热」比「投机」更值钱，只做一次前后对比会把普通那一趟付钱买来的缓存算到投机
头上。

重复文本 prompt（要求把同一句话写五遍），块长 8：

| 趟 | token | ms | tok/s | 前向次数 | 每前向 token |
|---|---|---|---|---|---|
| 普通 | 40 | 37578 | 1.064 | 39 | 1.03 |
| spec=8 | 40 | 56002 | 0.714 | 23 | 1.74（接受 16/71 = 23%） |
| 再普通 | 40 | 35408 | 1.130 | 39 | 1.03 |

散文 prompt，块长 5：

| 趟 | token | ms | tok/s | 前向次数 | 每前向 token |
|---|---|---|---|---|---|
| 普通 | 32 | 35134 | 0.911 | 31 | 1.03 |
| spec=5 | 32 | 32471 | 0.986 | 21 | 1.52（接受 11/29 = 38%） |
| 再普通 | 32 | 27068 | 1.182 | 31 | 1.03 |

两次运行都打印 `spec vs plain: identical token for token`，普通/普通对照也是。
这正是 §32 的修复要达到的结果：块验证不再改变答案，剩下的只是算术。

而算术说不划算。8 个位置在飞的一次前向要 56002/23 = 2435 ms，而单步是 963 ms——
**2.5 倍**，换来 1.74 个 token。这里投机需要每个前向超过 2.5 个 token，而 n-gram 草稿
在专门写成重复的文本上给 1.74，在散文上给 1.52。

这 2.5 倍有两个成因，量级规律不同：

* **专家并集几乎随块长线性增长。** 路由是 128 选 4，八个位置最多碰到 32 个不同专家，
  单个位置只碰 4 个。每前向的字节数正比于并集，所以在一个权重放不下的模型上，批处理
  唯一能赚的前提是位置之间复用专家——而它们不复用。
* **精确 FP8 路径在 n=8 时很贵**（§32：同一个 8192x6144 矩阵 4.5 ms 对 0.6 ms）。这是
  `MM_FP8_EXACT_N=8` 的价钱，且不是可选项：不这么做，被接受的 token 就不是普通解码会
  产出的那个 token。更便宜的精确性是有可能的——把参考归约树给行分块核，而不是把小 `n`
  路由到逐行核——但上面那条并集项本身已经足够定论。

所以 `MM_SPEC` 保持默认关，而这一次是测量后的决定，不是遗留说明。机制留在代码里，因为
它现在是精确的：将来若出现每前向接受超过 2.5 个 token 的草稿器，那是改配置，不是开课题。
要做到这一点需要移植的 7-token 块草稿器常驻 2.3 GiB，而且它得明显超过每前向 2.5 个接受
token，还得顺便付掉自己的读取开销。

关闭这条路会把 MiniMax 的计划重排：杠杆是字节和命中率，不是批处理。一步 574–692 ms，
其中 `io` 277–384 ms（等 SSD）、`experts` 132 ms（对已到达数据做 GEMV）、`proj` 92、
`wo` 33、`attn` 3、`route` 5。而且专家读取也无法跨层预取——router 跑在本层的输入上，
第 L+1 层的专家集合要等第 L 层 FFN 输出存在才知道。这是数据相关 router 的结构性串行，
不是调参问题。

## 34. 两件不是杠杆的事：更大的 pin cache，以及批处理

两件都很合理，现在都量过了；而且这次是 §31 的缓存损坏修好之后——这一点很重要，因为
之前的缓存容量对比是在一个会悄悄供应损坏 blob 的缓存上做的。

**更大的 pin cache 更慢，不是更快。** `CWICK_EXPERT_CACHE_GIB` 默认 21.3 GiB 对 40 GiB，
交错运行（两边都不会拿到热机），各两轮，同一 prompt：

| 缓存 | 命中率 | 流式读 | tok/s |
|---|---|---|---|
| 21.3 GiB | 64.5% | 190.85 GiB | 1.5, 1.5 |
| 40.0 GiB | 72.3% | 149.93 GiB | 1.3, 1.2 |

多出的容量买到 8 个点命中率、少读 21% 的字节，却丢掉 20% 的 token 速率。所以在这个
尺寸上，命中率不是约束：变大本身有成本——每次 lookup 要探测更大的索引、每次 miss 的驱逐
工作量更多、40 GiB 匿名足迹的 TLB 行为不同于 21 GiB。`mm_expert_cache_budget` 里那个
`host/6` 上限因此是一个恰好正确的局部最优，在有人把索引成本和驱逐成本分开量出来之前
应该保持。**不要再"多给它点内存"。**

**层内读/算重叠是真的，但稳态下很小**——而且现在默认打开，见 `mm_io_overlap_decode()`
上的注释。24 个稳态步：步时 664 → 638 ms，io 388 → 200，experts 123 → 282。从 `io` 里
离开的那 188 ms 大部分又在 `experts` 里出现，因为 join 就在那儿：真正消失的是 26 ms。
而在一个会话的头几步，同一个开关值 277 ms/步（713 → 436），因为那时要藏的 io 有六倍多。

和 §33 合起来，MiniMax 的图景是：一步 664 ms，其中 388 在等 SSD、123 是对已到达数据做
GEMV；第 L+1 层的专家集合要等第 L 层 FFN 输出才知道（数据相关 router）；权重放不下，
所以每步都在为字节付钱；而**加容量或加批宽都让它更差而不是更好**。这不是一个调参面；
剩下的收益在**每步字节数**（更小的专家足迹，即更少的不同专家）以及 `proj` 84 ms 与
`wo` 30 ms，而这两项已经跑在内存天花板上了。

## 35. 决策端点：读分布，而不是让它写一段散文

`--jev` 提供 `POST /v1/systemone`：一段 state、几个问题，返回带分布的类型化答案。要点在它
**不做什么**——它从不跑 decode 循环。一次 prefill 到达答案位置，该位置的下一 token 分布
截取到选项字母上，那个截取就是答案。什么都不生成，于是没有东西需要解析、不需要在模型把
JSON 包在散文里时重试、也不需要在它写出一个已被占用的格子时修补。

**读法。** 用字母码，不读选项文本。读每个选项自己的文字更讲道理——它去掉了字母先验——
但代价是每个选项一段续写，而这台机器上 token 才是贵的单位（MiniMax 664 ms、GLM 约
275 ms 一个解码 token，而整个问题只要一次 prefill）。字母路径每问题一次 prefill，与选项
多少无关：九宫格和是非题的代价一样。它放弃的是字母先验，而这件事是量出来的而不是争出来
的：把同样三个选项换序（info / replace / refund），选中的 key 与四位小数的概率都不变，
说明在这个 checkpoint 上先验不是决定答案的东西。这是一个模型、一个 prompt 的结论，不是
普遍断言。

**prompt。** 每个问题走 checkpoint 自己的对话格式，并且在模型开口前就把思考块关掉
（GLM 是 `<|assistant|><think></think>`，另两个用各自的头部），于是下一个 token 是答案而
不是思考块的开始。每个选项渲染方式完全相同，prompt 结束在 `Answer:`；选项之间唯一不同的
是读出头去看的那个字母。分词器若没有该字母的单 token 写法，会被点名拒绝，而不是用两 token
的近似去打分。

**成本，在 320B checkpoint、单模型常驻下实测：**

| 请求 | prompt token | 墙钟 |
|---|---|---|
| 一个 choice，3 选项 | 118 | 66 s（冷）/ 83 s |
| choice + score + noul | 331 | 206 s |
| 一个 choice，7 选项（棋盘） | 245 | 110 s |

成本是 prefill，并且随问题数增长而不是随选项数。十个问题就是十次 prefill，这也是上限取十
的原因，也是这个端点只适合"一段材料上的少量问题"的原因。把 state 的 KV 在问题之间共享能
省掉第 2..N 次 prefill 的大部分，但那需要引擎提供"从某个位置继续"而不是 reset 的入口——
这次没做，等真有调用方问超过两个问题时再做。

**数字的含义。** `probabilities` 是模型自己的下一 token 分布在选项字母上的取值；
`confidence` 是集中度 `(K·Σp² − 1)/(K − 1)`，均匀为 0，点质量为 1。它没有校准，也不是
"答对的概率"——在一道无法判断的题上，平坦的分布就是正确输出，端点返回整个分布正是为了让
调用方看见这一点，而不只是拿到 argmax。

**应用。** 井字棋对手和卡丁车路障生成器改为调用这个端点而不是 `/v1/chat/completions`：
棋盘变成空格子上的 choice，路障变成"种类 × 车道"的 12 选 1，距离由速度在本地算出。两者
都保留本地兜底，所以一次被拒或过慢的决策降级的是对手，不是游戏。

## 36. 一个端点、三种 prompt 语法：决策适配器

§35 落 `--jev` 时，家族判断是写在 `jev.c` 里的 `is_ds` / `is_mm` / 否则 GLM。回答"某个常驻
模型上的三个问题"够用，但需求一变成"切换模型的同时切换适配的 LLM 接口"就不够了：prompt
语法、答案拼写、可读选项集都是 checkpoint 的属性，而当时没有一个被命名。现在它们是一个
结构体。

**`jev_adapter`** —— `family`（glm / deepseek / minimax）、`prompt`（用哪个编码器拼决策
轮）、`letters`（A..Z 中有多少是单 token）、`first_missing`、`space_lead`。前两项只由
`model_type` 决定；tokenizer 那半要靠 `jev_probe()`。

**解析时机。** 启动时 tokenizer 就绪后一次，之后每次 `POST /v1/models/switch` 再一次——不
在端点里缓存、也不是每进程只算一次。切换会替换 `srv->eng[0]` 与 `srv->tok`，因此切换之后
到达的请求探测的是新 tokenizer、走的是新语法；不存在"拿 MiniMax 的 prompt 去读 GLM 的
logits"这条路径。

**字母表实测**（`make jevprobe`，只读 tokenizer，不加载权重）：

| checkpoint | 词表 | prompt 语法 | 字母 | 拼写 |
|---|---|---|---|---|
| GLM-5.3-Flash | 154,820 | `glm-chat-closed` | 26/26 | `" A"` |
| DeepSeek-V4.1-Flash-1017 | 128,000 | `deepseek-chat-closed` | 26/26 | `" A"` |
| MiniMax-M3.1-Flash | 200,000 | `minimax-chat-plain` | 26/26 | `" A"` |

三家的每个字母都是一个带前导空格的单 token，所以读数本身是通用的，真正的差别只在那一轮
的语法。这是量出来的：工具会按两种写法逐字母打印 token 数；分词器不能单 token 拼出字母的
checkpoint 会被点名拒绝，而不是用两 token 的近似去打分。

**过期模型名会被拒。** 请求里的 `model` 会与常驻别名和 `model_type` 比对（大小写不敏感），
其它一律 400 并点名当前常驻模型。在这之前，一个缓存了 `MiniMax-M3.1-Flash` 的应用在服务端
切到 GLM 后，会拿到 GLM 的决策却挂在 MiniMax 的名字下——这是调用方唯一看不见的故障。
`GET /v1/systemone` 公布实时适配器（`family`/`prompt`/`letters`/`first_missing`），客户端
可以直接核对，而不必从一次失败调用里反推。

**应用。** 两个托管应用都只调这个端点：井字棋问空格子上的 choice，卡丁车问"种类 × 车道"的
12 选 1。两者都保留本地兜底，也都在收到"该模型已不再服务"的 400 后重试一次——这就是一台
缓存了模型 id 的手机，在桌面切了模型之后还能活下来的方式。

## 37. 一条路径漏掉的一行释放，以及它的代价

在做两接口矩阵（§36 的配套，docs/PERF.md 2026-11-07）时发现：GLM 的对话跑不完 48 token。
24 个解码 token 里 footprint 从 67 → 117 → 179 GiB，而 RSS 从 50 **降到** 35 GiB，然后被
jetsam 杀掉。32 GiB pin 缓存、8 GiB pin 缓存、`--backend cpu` 三种配置都复现，所以既不是缓存
也不是 GPU。

**增量本身就是线索。** 每 token +6 GiB，正好是一步要读的专家字节（7.88 GiB）。"每步留下整整
一步"只能是每步分配、从不归还，而不是缓存在抖动。

**位置。** `moe.c` 取一层专家有两条路：批量路径分配 `cwick_fetch_req` 数组、取数、GEMV，
然后调 `layer_fetch_release` + `free(reqs)`；单 token 路径什么都做了，只差这两句。pin 缓存
拒绝的每个矩阵都是 fetch 自己拥有的缓冲（`prefill_blob_alloc` / `cwick_shmem_alloc`），所以
漏掉这一对调用就是每层漏一层的 staging 字节，永不回收：8 专家 × 3 矩阵 × 45 层 × 8 MiB ≈
每步 5.8 GiB。

**为什么只有 GLM。** DeepSeek 与 MiniMax 有各自的模型文件和取数循环，这条是 GLM 的路径。
泄漏能活下来也是这个原因：三 checkpoint 矩阵是 route-first fetch 引入之后第一次跑长 GLM 生成。

**为什么短测试抓不到。** 八个 token 时进程已经吃掉 67 GiB，但那次运行仍然"跑完"；杀落在
24–32 token。任何在流式 checkpoint 上量 decode 的东西，都要至少生成 32 token，并且看
`cwick:process_footprint_bytes`——RSS 是朝反方向走的，因为涨的是压缩/页缓存记账，不是活堆。

**修复。** 在组 GEMV 返回之后释放这一批并 free 数组（Metal 两个 wave 在那之前都已 wait，CPU
路径在上面的循环里已经消费完视图），读取中途失败时也把本批自己拥有的缓冲还掉。用同一个请求
重测：48 token 跑完，默认 32 GiB 缓存下 footprint 平在 49.98 GiB（8 GiB 时 25.94 GiB，速度
同为 1.25 tok/s，说明限制不在缓存大小），回答文字连贯。`make test` 全过。

这条规则被再次印证：在取数路径里，"owned"是一种生命周期，每一个出口——成功、预算拒绝、读取
失败——都必须结束它。批量路径把这条规则写下来了；单 token 路径是后来写的，没有写。

## 38. 决策路径上的同款回归：同样的测试，不同的答案

§37 修的是 decode 循环里每一步的驻留，而 `/v1/systemone` 从进不了那个循环。听起来这就不必
管了——但"这条路径不走漏那段代码"恰恰是那种应该量一次、而不是想当然的断言。于是决策端点跑了
同一套长跑，而且挑了两个 prefill 路径代码完全不同的 checkpoint：GLM（整段 prefill）与
MiniMax（固定 512 token 分块，唯一有可能把分块缓冲留下的地方）。

**测试。** 一个 checkpoint 一个进程，不切模型。连续多个请求，每个请求在同一段 state 上带
2–3 个问题，因此一个请求就是 2–3 次 prefill。每轮换 state 与问题集，后一轮不可能用上一轮的
前缀缓存作答。`cwick:process_footprint_bytes` 在每个请求前后各采一次。

**结果。** GLM：三个请求 753 / 891 / 937 输入 token，footprint 15.46 → 16.15 → 16.14 →
16.20 GiB。MiniMax：817 / 1659 / 817 token，22.11 → 22.22 → 22.23 GiB（最初那次 0.12 GiB
的采样早于懒映射）。第一个请求一次性付掉池与映射，之后所有请求加起来：GLM **+0.06 GiB**、
MiniMax **+0.12 GiB**，各自约 2000 token。decode 那次泄漏是每**步** +6 GiB。差三个数量级，
而且增量不随请求数增长——GLM 第三轮的结束值比第二轮还低。

**这次钉下的两个事实。**

1. *一个请求内部不会复用它自己那段 state 的 KV。* `input_tokens` 正好是 问数 × prompt
   （753 = 3 × 251，1659 = 2 × 829）。每个问题都是 state + question 的完整 prefill，而不是
   在共享 prefill 后面接一个读数。这是端点落地时的成本模型，现在它是量出来的而不是写出来的：
   三个问题就是三次 prefill。
2. *pin 缓存对 prefill 没有意义。* 日志是 `expert cache 0.00 GiB pinned, 686.53 GiB
   streamed`——251 个 token 几乎路由到整个专家集，步与步之间没有值得留存的东西。决策延迟被
   设备读速定住，这也正是 footprint 平的原因：这条路径没有能涨的缓存。

**顺带得到的确定性。** 同一个 GLM 请求在两个独立进程里跑，概率到小数点后四位完全相同
（0.9989 / 0.6459 / 0.9977），MiniMax 重复的那轮也选了同样两个答案。这就是 §31 说的性质，
只是这次是在端点上观察到的、而不是在缓存上：决策变了意味着 prompt 或模型变了。

**可以复用的部分。** 这套回归就是 `tools/decision-longrun.sh`：自己起服务、发 N 个带 M 个
问题的请求、在每个请求前后采 footprint 指标。它不做自动断言——判断标准是"最后一个采样比第一个
多出的部分，是否超过模型自身的工作集"。这是故意放宽的：紧到能抓住 0.12 GiB 的阈值，也会被
分配器行为触发。它真正能抓的失效模式，是真实发生过的那个——每步 GiB 级增长，画在图上就很明显。

## 39. 模型库放在服务端，不放在浏览器里

需求是一个「模型」标签页：展示已下载、正在下载、目录检索、进度、断点续传、几百 GiB 的
checkpoint，统一落到 `./models`。最顺手的做法是浏览器直连目录站、再调一个脚本下载，两条都
否掉了：前者把第三方源（以及 CORS）塞在控制台和自己的服务端之间，后者把进度状态拆到两个
进程、两种语言里。`src/hub.c` 是 C，跑在服务端进程里，只持有一张任务表。

**五个控制端点，不占生成槽位。** 所有 `/v1/hub/*` 都在 `slot_acquire` 之前应答，和
`/v1/metrics`、`/v1/models` 一样。目录站卡住、或者下载要跑几个小时，都不能让对话请求排队。
页面轮询状态表，没有 SSE 推送通道，所以这张表的互斥锁就是全部的并发故事。因为这些是真实
线程，libcurl 的 `CURLOPT_NOSIGNAL` 必须开；进度回调是工作线程唯一碰共享状态的地方。

**一条落盘约定代替状态数据库。** 每次传输写 `<目标>.incomplete`，大小与清单一致才 rename。
于是暂停、出错、进程被杀、重启留下的是同一种状态；下一次读文件大小、发
`Range: bytes=<size>-` 接着追加。全程 `curl_off_t` / `int64_t`，100 GiB 的单个分片、300 GiB
的仓库都不是特例。远端若忽略 `Range` 返回 `200`，写回调会先截断部分文件再追加，不会拼出两份。
`LOW_SPEED_TIME` 把卡死的传输变成可重试的 `error` 任务，而不是一根永远停在 93% 的进度条——
下面的实测里正好发生过。

**两层布局，按需登记。** 下载落到 `./models/<owner>/<name>/`，与仓库 id 一致、也避免短名撞车，
`.cwick-repo` 标记记录完整 id。启动扫描同时把这种布局和扁平的 `./models/<name>` 登记进普通
模型目录；**本轮进程内**下载完的 checkpoint，由 `POST /v1/models/switch` 在名字还不在目录里时
按需查找一个完整的本地目录来登记。这样不必新增第六个「刷新」端点，也不给 catalog 添第二个
写者。只要目录里还有 `.incomplete`，就永远不可切换。

**实测**（单进程，目录站 hf-mirror，38.4 MB / 13 文件的测试仓库）：检索约 2 s 应答；下载到
第 3/13 个文件时暂停，留下 51,408 字节的 `.incomplete`；继续时从 0.14% 接着走，没有重下已完成
文件；一次传输卡死 60 s 后变成 `error`；`retry` 在 12 s 内跑完，13 个文件大小与远端清单逐个
吻合、`.incomplete` 归零。随后 `/v1/hub/models` 把两个本地 checkpoint 都报成 `complete:true`。
对话页保持 `/v1/chat/completions` 不变——模型库是控制面功能，不改变生成 token 的那条路。

**应用。** 应用市场承载目标里点名的两个（井字棋、卡丁车）。两者都走 `/v1/systemone`；
「对话用 chat、应用做决策」这条分界由 `tests/test_cwick.c` 直接对内嵌资源做断言。

## 40. 一个设置 tab、真趋势图、没有出口

目标文档里写的是「设置（控制台） tab」——**一个** tab——但实现长成了两个：指标在
**控制台**，配置与模型切换在**设置**。这次把两者合并，补上趋势图和更完整的模型信息，
并去掉把控制台指向第三方 API 的选项。

**合并。** 删掉 `consoleView` 与导航项；四张指标卡、内存/缓存条、模型面板、趋势面板、
生成参数、安全与只读运行配置全部落进 `settingsView`，`VIEW_META` 变成五项。「实时刷新」
仍驱动同一个 2 秒抓取，rail 的常驻内存条仍保留 5 秒兜底——指标路径一个字节没改，只是换了
展示位置。看到数字和改掉产生这个数字的开关，不再需要切 tab。

**趋势是差值，不是计数器。** `/v1/metrics` 发布的是累计计数器；直接画 `1/(sum/count)` 得到
的是全进程平均，几乎一条平线，对一次运行没有信息量。所以每个点都是相邻两次采样的差：
解码速度取 `dSteps/dSeconds`、TTFT 取 `dSum/dCount`、两级缓存命中率取各自窗口内的
`dHits/(dHits+dMisses)` 与 `dQueries`，内存取瞬时 footprint。没有新增解码步的窗口存 `null`、
画成断点，而不是画一个假的 0。120 个点 × 2 秒 = 四分钟，只存在浏览器内存里，四张 canvas
不引入任何图表库。隐藏的 section `clientWidth === 0`，所以画布回退到 markup 里的宽度，
并在切视图和 resize 时重绘。清空或暂停只影响浏览器，服务端指标不受影响。

**模型信息。** 六个头条字段保留，下面把 `cwick:model_info` 的**全部**标签渲染出来，切到不同
checkpoint 家族时不必打开 Prometheus 文本就能看见。运行配置里补上 `/v1/config` 的模型库
根目录与目录服务地址。

**去掉对外出口。** 设置页的 API Base / key / 模型名面板、测试连接按钮及对应持久化字段全部
删除：`apiBaseUrl()` 现在恒为 `location.origin + '/v1'`，`apiHeaders()` 只带 `Content-Type`，
`apiModel()` 只取常驻模型 id。两个托管应用原本共享同一份 `cwick.prefs.v1`，不改的话它们会
继续把 `/v1/systemone` 发往外部主机，因此应用里那套 helper 也一起删掉：各自从
`/v1/models` 发现常驻模型，切换后重新发现——这正是决策接口「切模型自动切适配」要求的。
`loadPrefs()` 会主动从已有的 localStorage 条目里删掉残留的 `apiBase` / `apiKey` /
`apiModel`，避免同源的其它页面日后又捡起来。

**刻意保留的部分。** `--api-key NAME` 与 `/v1/auth/keys` 没动。它们是**让其他客户端接入本机
cwick** 的凭证，对应目标文档「支持 API Key 方式，确保 cwick API 可以方便接入第三方 Agent
平台」，方向与「控制台拨出去」正好相反；安全面板里也写明了这一点。对话仍用
`/v1/chat/completions`，应用仍用 `/v1/systemone`，两个端点都没改。

**护栏。** `tests/test_cwick.c` 直接对内嵌资源断言：趋势函数与 canvas 存在、
`consoleView` / `data-view="console"` 已消失、`setApi*` / `testApi` /
`state.prefs.apiBase` 已消失、两个应用 bundle 都不再读 `cwick.prefs.v1`。这是对服务端真正
发出的字节做检查，所以移除不会悄悄回退。

**实测。** `make test` 通过，新增五条针对内嵌资源的断言全绿。用 CDP 让合成模型服务的
设置页按真实时钟跑 10 秒，其间发 11 次短生成：趋势图采到 6 个点，增量解码速率
13746.91 tok/s，而头条卡片（全进程累计均值）是 13340.13 tok/s——两者确实不是同一个量；
四张 canvas 都有绘制像素。没有生成活动时，三张速率图显示空闲态、内存线仍连续，
null 断点的规则按设计生效。DOM 同时确认：导航恰好五项、没有第三方面板、
`cwick:model_info` 标签全部渲染、模型库根目录报 `./models`。趋势面板列出对应的
Prometheus 指标名并直达 `/v1/metrics`，对应目标里「趋势图（包括 prometheus）」那句。

## 41. 「包括 prometheus」指的是 exporter 本身，不是一份固定投影

§40 交付了四张由 `/v1/metrics` 驱动的趋势图和一个原文链接。那是 Prometheus **数据**，但目标里
「趋势图（包括prometheus）」更合理的读法是：让操作者看见 exporter 到底发布了什么，并自己选
要画哪条。固定投影等于替用户决定了「只有这四条有意思」；调缓存、调队列、换 checkpoint 时，
还是得离开控制台去翻原文。

**补了什么。** 合并后的设置页新增「Prometheus 指标」面板。抓取解析现在同时保留 `# HELP`
与 `# TYPE`，因此面板能列出**每一条数值序列**的类型、说明和实时值，支持按名筛选。点一行就
为它挂一张独立 canvas（最多 4 张）和 120 点的历史。计数器可以画原始值，也可以画每秒增量；
这个选择和勾选的序列一起存在浏览器里（`cwick.prom.v1`），不是服务端设置。

**几个决定。**

- **不引入 Prometheus 服务器、不做 TSDB。** cwick 自己是 exporter，`/v1/metrics` 是累计
  计数器的一次快照。再加一个抓取库或 query-range 端点，只是把浏览器能存四分钟的东西复制
  一份，还会在推理机上多一个长驻写者。短历史留在客户端，服务端保持无状态。
- **每个序列一根 y 轴。** 字节计数器、比率、延迟绝不能共用一个刻度，否则小的那条会被压平。
  只有两条缓存命中率共用 0–100% 这一根轴，因为它们天然同域。
- **增量是可选的，且只对 counter。** gauge 求每秒差是 bug（uptime 会变成恒定的 ~1），
  counter 画原始值是一条斜坡。开关只对 `# TYPE counter` 的序列求导（没有 TYPE 注释时按
  `_total` / `_count` / `_sum` 后缀退化判断），gauge 保持原始值并在图例里注明；切换时清空
  历史，避免同一条线里混进两种单位。四张内置图保持各自固定、有含义的单位。
- **共用同一个采样时钟。** 自定义序列在 `trendSample` 内部采样，不另起定时器，所以关掉
  「实时刷新」会同时冻住所有图，一次 `/v1/metrics` 抓取同时喂两个面板。`drawChart` 增加
  了可选时间数组，因为后来才勾选的序列比会话的第一个采样点年轻。

**实测。** 合成模型上，面板列出 65 条导出序列及其 HELP/TYPE；筛选 `cache` 后剩 8 条。
勾选 `cwick:decode_steps_total` 与 `cwick:uptime_seconds` 得到两根独立坐标轴的图、各 4 个
采样点。切到「每秒增量」后只有 counter 的图例与数值变化（329 步 → 约 47 步/s），
uptime gauge 仍是「原始值」和原始读数。勾选与开关都写进 `cwick.prom.v1`，刷新后保留。
`make test` 通过并新增两条对内嵌资源的断言。引擎、端点、模型路径一个字节没改。

## 42. 决策 prefill 终于出现在 stdout 里

`/v1/systemone` 从不进 `generate_impl`：`jev_decide` 对每个问题调用一次
`cwick_engine_prefill_logits`，而这个函数以前只是一段裸 forward。chat 路径会打印
prefill 开始、逐 token/逐层心跳、耗时，并驱动 `cwick:prefill_*`；决策路径一条都没有，
服务端也没给它绑定请求标签，所以偶有引擎日志也写着 `gen`。一个多问决策可以在两条
访问日志之间流式读几十 GiB 而 stdout 一片空白。

**一句话根因**：可观测性写在生成循环里，而不是 prefill 入口里。把它下沉一层，所有调用方
一次性都修好。

**改了什么。** `cwick_engine_prefill_logits` 重构成单一收尾（原来三个后端各自 `return`）。
绑定了请求标签时，它现在发布 `cwick_metrics_set_prefill`、打印开始行、为 DeepSeek 与
MiniMax 打开逐层心跳（心跳受"prefill 进行中"标志控制，以前只有 `ds_engine_generate` /
`mm_engine_generate` 会设置），最后打印带 tok/s 的耗时行；GLM 的 `engine_prefill` 内部
本来就有 `maybe_progress_at`。`server.c` 在 `jev_decide` 前后绑定/清除 `eng->req_tag`，
顺带让 DS/MM 内部日志也带上请求编号——它们的 `ctx.tag` 是指向同一缓冲区的指针。

**默认静默规则。** `logitdump` 解析 stdout，测试套件也用同一个函数做数值 A/B。这里无条件
打 INFO 会同时毁掉两者。所以引擎只在 `eng->req_tag[0]` 非空（即"服务端把这次 prefill
绑定到了一个 HTTP 请求"）时输出。不新增 API，也没有需要记得设置的开关。

**请求与执行日志。** `jev_decide` 在收到请求时打一行（问题数、body 字节、常驻模型、适配器、
`stream=false thinking=off`——决策契约），每次 prefill 前打一行 `question i/N`，每个读数
打一行（获胜 key、概率、次高概率、集中度），最后汇总输入 token、墙钟与 prefill tok/s。
`--trace` 额外打印每个渲染后的 prompt 与完整决策对象。`server.c` 对所有早退路径统一打印
`systemone: decision refused — <原因>`，所以 400 也有自己的一行。全部走
`CWICK_INFO` / `CWICK_TRAC`，每行都带系统时间与 `[req N]`；`log.c` 写并 flush stdout。

**实测。** 合成模型上两问决策产出：`decision: 2 question(s) ...`、两行 `question i/2`、
两次 prefill 开始、55 / 54 行逐 token 心跳、两行 `prefill done`（466 tokens / 37 ms、
432 tokens / 30 ms）、两行读数（`-> a p=0.3368 (runner-up 0.3325)` 与 `-> x p=0.5046`）、
`decision done: 2 question(s), 898 input tokens, 67 ms wall` 以及 200 访问日志——全部带
`[req 1]`。空问题集产出 `[req 2] systemone: decision refused — body.questions is empty`
与 400。`make test` 输出里 `[systemone]` 仍为 0 行，确认 CLI 路径保持静默。

## 43. 两个应用现在问的，就是目标里写的那个问题

两个应用的骨架本来都对：真莫比乌斯环、每次决策一次 `/v1/systemone` prefill、各自独立的
sqlite、右侧栏、不生成任何需要解析的对象。错的是**问题的措辞**——代码里很小，语义上很大。

**井字棋。** 原来给机器的 instructions 是「能赢就赢，否则堵 X，否则占中心，再角，再边」。
这其实是一整套本地策略伪装成一个问题，prefill 退化成执行浏览器里已经有的规则。目标要的是
「落子在哪个位置**赢的概率最大**」。现在 app 为每个空格计算棋盘**事实**（`cellFacts`：立即
成三、强制拦截、后续制胜点数、可用连线数），写进 state 与每个选项的描述，然后问哪一格获胜
概率最大。不排序、不建议：checkpoint 对选项字母的 softmax 就是决策，argmax 就是落子。
`fallbackMove()` 保留，但只在请求失败时启用。右侧栏为每个空格画一根概率条、高亮实际落子，
并显示人这一手的战术事实，于是「人与机器的对弈决策过程」两边都有。标题明确写了这些条是
**一次 prefill 的选项分布，不是模拟胜率**——后者是这个端点给不出的承诺。

**卡丁车。** 原来的 instructions 字面上写着「给玩家留反应空间」，与「确保跑圈难度最大」正好
相反。现在问的是哪颗路障让这一圈最难，state 里带碰撞数、已放数量、难度等级、当前是环的哪一面
以及一个加压点。「实时随机决策」是更隐蔽的缺口：固定 prompt 的读数是确定的，固定的 A..L 顺序
会让模型的字母偏好永远落在同一颗路障上。现在每次请求都用 Fisher-Yates 洗牌 12 个选项（服务端
把有序的 `criteria` 成员映射成字母），抽一个新的随机种子和六种加压点之一。**随机的是问题，
不是赛道**——依然没有本地路障生成器，每颗路障仍来自模型的一次 prefill。种子、加压点与分布的
归一化熵显示在右侧栏，并写进 sqlite 的 `agent`/`spawn` 事件，随机性可审计。距离仍由车速派生
（看不见的「最难关」只是一次不公平碰撞），但会随圈数在 10 m 下限之上收紧，让难度目标真的被
感受到，而不只是写在 prompt 里。

**实测。** 合成模型（context 从 512 提到 4096，才放得下 656 token 的 12 选项决策）上用真实
浏览器下了一手人棋：请求带 8 个含战术事实的选项，机器作答，右侧栏渲染出 1 个概率块 / 8 行 /
1 行高亮，加上人与机器各一行。跑圈 14 秒产生 7 次决策：**7 个不同种子、7 种不同选项顺序、
4 种不同加压点**，每次 12 选项，失败 0，平均响应 127 ms，已放 5 颗路障，右侧栏显示种子与
决策分散度。`make test` 通过并新增 11 条针对内嵌资源的断言，prompt 措辞、洗牌、sqlite 声明
与两个右侧栏都不会悄悄回退。

和之前一样，`goal.md` 只读、未改；对话页仍走 `/v1/chat/completions`。

## 44. 决策 prefill：先测量，再只在答案精确的地方优化

`/v1/systemone` 每个问题做一次 prefill。在 60 层、128 专家的 checkpoint 上这是每问 1.7 分钟，
所以端点的成本几乎全是 prefill——而一个问三个问题的请求，会把共享的 state 付三遍。

**时间到底花在哪。** MiniMax 在 4096 上下文下的逐 token profiling：满 512 token 的块是
173 ms/token（`experts 82, io 28, attn 26, proj 20, wo 8, route 6`），而 9 token 的尾块是
606 ms/token（`io 220, experts 276`）。由此得到两个都和直觉相反的结论：

1. 成本是**块级的专家读取**，所以块越小越差。550 token 均匀切成 275+275 实测 **113.6 s**，
   而 512+38 是 **109.3 s**——看起来更公平的切法慢了 4%，因为它把最能摊销专家读的那块砍半了。
   这个变体仍留在 `CWICK_JEV_CHUNK_EVEN=1` 后面，但默认关闭。
2. 真正有效的是提高上限：`MM_CHUNK`/`DS_CHUNK` 从 512 提到 1024，让常见的 500–1000 token
   决策 prompt 落在一块里。MiniMax 单问 550 token：**109.3 → 95.5 s（1.14×）**。代价是
   activation 缓冲多占约 1.6 GiB（本机 footprint 23.8 GB，swap 1.36/3.0 GiB）。

**请求内共享前缀。** 一个请求里的所有问题都以同一段 `system + <state>` 开头。`jev_decide`
现在先编码全部问题、取最长的公共 **token** 前缀、只 prefill 一次，然后每个问题从前缀续写。
三问请求共享 348-token 前缀（输入 1145 token），实际只 prefill 449 个：**213.8 → 102.0 s
（2.10×）**，且每个概率与不复用的那次**完全相同**（max |dp| = 0.0）。

**必须丢掉的那部分。** 靠移动 `eng->pos` 并在前缀之后重写，只在「跨 token 状态按位置寻址」
时成立。三个 checkpoint 里有两个不成立：

| 后端 | 强制复用 | 结果 |
|---|---|---|
| MiniMax（GQA + KV） | 原地续写 | **|dp| = 0.0**，选择完全一致 |
| DeepSeek（压缩 KV + indexer pool） | 原地续写 | **|dp| = 0.56**，3 个答案里改了 2 个 |
| GLM（DeltaNet + DSA pools） | 原地续写 | **|dp| = 0.41**，3 个里改了 1 个 |
| GLM | 快照 + apply（`pfx`） | 选择正确，但 **|dp| = 0.11** |

因此端点只在 MiniMax 上启用复用，另外两个保留 N 次完整 prefill（`CWICK_JEV_REUSE_ALL=1`
用于测量已修好的后端，`CWICK_JEV_NO_PREFIX_REUSE=1` 可在任何后端强制回旧路径）。快照路线
本来像通用答案——`pfx.h` 明确写着这些状态**不能回退但可以整体快照**，`test_prefix_cache_equiv`
也证明了生成逐字节一致——但真实 GLM 上恢复出来的 DSA pool 留下了未填满的尾巴，足以把概率
推动 0.11。决策端点不能交付一个「偶尔改变决策」的 2× 加速。

**端点之外的含义。** GLM 或 DeepSeek 的单问请求在这里没有算法级收益：250 token 的 prefill
已经路由到几乎整个专家集（GLM 251 token 流式读了 686.53 GiB），成本就是「把模型读一遍」，
token 数几乎不影响它。复用是唯一的杠杆，而它需要一个 apply 精确的后端——那是下一步的工作，
不是一个可调参数。

**接口。** `cwick_engine_prefill_logits_at(eng, ids, n, start_pos, logits, err)`：
`start_pos == 0` 复位，`> 0` 从已有前缀续写。三个后端走的是同一条代码路径（GLM 从
`eng->pos` 推导绝对位置，另外两个把 `pos` 当参数），所以「精确性」问题在状态，不在管线。
`test_prefill_resume_equiv` 在两个合成 checkpoint 上、用两个切点（prompt 中部与倒数第二个
token）对比整体 prefill：rel = 0.0。

## 45. 第二种方言，不是第二个引擎：`POST /v1/responses`

编程 Agent 越来越多地说 Responses API 而不是 chat/completions，只答 chat 的服务端会逼每个
这样的客户端自带一层要自己维护的适配器。最直觉的实现是再写一条请求路径；那会让 prompt 模板、
队列、前缀缓存和思考预算各多一个会漂移的地方。

**适配器到会话为止。** `responses_parse` 遍历 `input` items，构造 OpenAI 的 `messages`
数组——`message`（`developer` 归一到 `system`）、连续 `function_call` 合并成同一个 assistant
轮、`function_call_output` 变成 `tool` 轮、`reasoning` 丢弃（隐藏思维链不是可回放的输入）。
然后把它和请求里的标量字段序列化成一个 chat 形状的 body，交给 `/v1/chat/completions` 早就
在用的 `validate_body`。此后的全部环节——模型身份、采样范围、工具声明、tokenize、前缀缓存、
生成、访问日志——都是同一份代码，`gr->api` 只决定信封。

**信封的代价。** response 对象要用另一套 usage 拼写（`input_tokens` / `output_tokens` /
`output_tokens_details.reasoning_tokens`）、用 output items 而不是 `choices`、用具名 SSE
事件而不是裸 `data:` 帧。`responses_body` 只构造一次，缓冲 body 与终止帧
`response.completed` 共用它，所以两者不可能不一致。思考只在客户端要了 `reasoning.summary`
时才发布；没有工具调用的回合一定带 message item，因为它的缺失意味着"没有回答"，而不是
"空回答"。

**拒绝什么，为什么。** 这里没有持久 response store，所以非 null 的 `previous_response_id`
或 `conversation` 是 400：点名该字段，并要求回放 input items——接受它只会悄悄截断 prompt。
`store:true`、`background:true`、`truncation:"auto"`、`top_logprobs`、`max_tool_calls`、
非 `function` 工具、图像/文件 part、`tool_choice:"required"` 都被拒绝，理由与其它不支持的
字段一致：装作支持的服务器比直接说不更糟。在这里无作用的字段（`metadata`、`include`、
`parallel_tool_calls`、`service_tier`、`prompt_cache_key`）会被接受并忽略，文档里写清楚。

**验证。** `test_server` 里十条断言覆盖缓冲信封、以 `response.completed` 结束且没有 `[DONE]`
的具名流、item 往返（developer 角色、function_call + function_call_output）、扁平工具转换，
以及六条拒绝路径。它们与 chat 测试跑在同一个合成 checkpoint 上，所以"等价"是被测出来的，
不是声称的。

**控制台必须选边，而它两边都不选。** 两种方言都能答，却只有一个客户端、还只能在一种方言下
验证，这件事本身没什么价值，所以面板上的 `接口` 下拉按轮选择端点。转换是单向的，住在
`web/app.js` 里：`respInput` 把会话变成 items，`respEventToChat` 把每个具名事件变成 chat
chunk，`postChat` 把 chat 形状交给它上面的所有代码。切换没有新增气泡、工具卡片或统计路径
就是因为它——换成"第二套渲染"的话，两边任何一处改动都会立刻漂移。有两个细节是方言决定的、
不是口味决定的：请求带上 `reasoning.summary:"auto"`，思考面板才有东西可显示（引擎把模型
自己的思考块当摘要报出来）；工具调用只能等 item 完成才画得出来，因为没有参数增量可渲染。

## 46. 第三种方言：`POST /v1/messages`

项目已经会说 chat/completions 和 Responses。Anthropic 的 Messages API 是编程 Agent 配成
Claude Code 时发的那一套，只认 OpenAI 两种的服务端会逼这个客户端再挂一层代理。实现上的问题
和 §45 是同一个，答案也是同一个：**在会话边界适配，不在引擎里适配**。

**适配器要对齐的三件事。** 它们不只是拼写不同，结构上就不同。`system` 是顶层字段而不是一条
消息，所以它被前置成 system 轮。content 是带类型的 block 数组，所以一轮里可以同时有正文和
调用；工具结果是一个 `tool_result` block，待在下一个 user 轮里，而不是一条自带 role 的消息
——同一轮里的多个结果保持顺序，它们之后的正文成为随后的 user 消息。停止词表是
`end_turn` / `max_tokens` / `tool_use`，计数在 `message_delta` 里报，而不是一个 usage 帧。
这些都不碰 prompt 模板：`anthropic_parse` 重建 chat body（包括扁平工具 → 嵌套工具的转换）
并调用同一个 `validate_body`，所以模型身份、采样范围、工具声明校验与思考档位在三种方言下
只有一份实现。

**"适配器真的是适配器"的验证。** 最硬的检查是同一个请求必须等价计费：一份在
chat/completions 下渲染出 640 token prompt 的工具声明，在 `/v1/messages` 下渲染出的也是
640 —— 量出来的，不是论证的。测试新增 19 条断言，覆盖信封、block 转换（system blocks、
`tool_use`、`tool_result`、被丢弃的 `thinking` block）与五条拒绝路径。

**拒绝什么，为什么。** `stop_sequences`（引擎在自己的分隔符上停；客户端给的字符串只能事后
匹配，那是另一个特性）、image/document/audio/video block（纯文本服务端，而且悄悄丢掉附件
等于送出错的 prompt）、`tool_choice:{"type":"any"}`（与 OpenAI 的 `required` 同一种约束解码
需求）、assistant 轮里的 `tool_result` 或 user 轮里的 `tool_use`（配对错了，猜一个就会送出
错的 prompt）。`thinking.signature` 发空串，因为没有签名密钥、也从不校验思维链——这也是
请求里的 `thinking` block 被丢弃的原因，与 Responses 丢弃 `reasoning` item 完全一致。

**控制台多了第三个选项，但不是第三套渲染。** `msgMessages` / `msgEventToChat` /
`msgFullToChat` 把这个方言映射到另外两个产出的同一种 chat 形状事件，所以气泡、工具循环与
统计行一行没改。有一点比 Responses 更好：这个方言**确实**流式发调用参数
（`input_json_delta`），所以工具卡片在模型写参数时就成型，而 Responses 只能等 item 完成
才能公布这次调用。

## 47. 按参考文档审这三种方言

两个适配器是照着协议的"形状"写的，不是逐字段对着文档写的，所以"哪里和参考不一致"这个
问题值得量一次，而不是凭印象回答。把信封对着公开 schema（OpenAI 的 Responses 事件模型、
Anthropic 的 `Message` / `RawMessageDeltaEvent` 类型）核了一遍，发现六处属于"看起来合法、
但不是 schema 描述的那份文档"的问题。六处都修了；拒绝清单仍然是拒绝。

**纯粹缺字段。** 每个 Responses 事件都必须带单调递增的 `sequence_number`——客户端靠它发现
丢帧，而且这个字段没有默认值。`response.output_text.delta` 与 `.done` 必须带 `logprobs`
（这里是空数组，本服务端不算对数概率）。response 对象必须带 `tools`、`tool_choice`、
`parallel_tool_calls`（都不可选）以及请求回显的其余字段，还有 `error` /
`incomplete_details`。`message_delta` 要同时重复 `input_tokens` 与 `output_tokens`，这样
只读收尾事件的客户端也有完整 usage。这些都不改引擎行为，改的是"客户端拿到的还是不是它以为
的那个对象"。

**真正错的。** 两处。`status` 永远是 `"completed"`，哪怕这一轮是预算用尽才停的——被截断的
答案和答完的答案分不出来；现在是 `"incomplete"` 且
`incomplete_details.reason = "max_output_tokens"`，流式终止事件相应是
`response.incomplete`（响应已建立但没能完成时是 `response.failed`，原先只发一个裸 `error`）。
另外终止帧里的 item id 是独立生成的，和 `output_item.added` 公布的不是同一个，
`response.completed` 描述的是另一个 message item；现在 id 由 response id 派生
（`…-msg`、`…-rs`、`…-fc<i>`）。

**去掉一处自相矛盾。** 没请求 `reasoning.summary` 时，缓冲 body 里没有 reasoning item，
而流式既公布了 item 又在推 `reasoning_text.delta`——两条路径对 output 列表的说法不同。现在
只要这一轮思考过，item 就存在；`summary` 只在被要求时填充，思维链仍然不进 body，但形状
不再变。

**chat 有一处同类问题。** `developer` 是 OpenAI 现在对指令轮的拼写。GLM 模板的角色链最后
落在 `user` 分支，于是一条 `developer` 消息悄悄变成了**用户请求**；DeepSeek 的渲染器则直接
把未知角色判错。两边现在都当成指令轮。

**仍然是拒绝的。** Responses 里需要持久化的那一半（`previous_response_id`、`conversation`、
`store`、`background`）、`truncation:"auto"`、`top_logprobs`、`max_tool_calls`、非 function
工具、图像/文件输入、`tool_choice:"required"`，以及 Anthropic 的 `{"type":"any"}`。
`parallel_tool_calls` 会回显但不强制（发几次调用由 checkpoint 决定），`text.format` 只有
纯文本，也没有 `/v1/messages/count_tokens`——引擎没有"不渲染 prompt 就能数 token"的入口。
这些都写在 `docs/SERVER.zh-CN.md` 各自方言那一节里，而不是留给客户端去踩。

**覆盖。** 613 项断言，新增十二条：回显字段、sequence_number 数量与事件数相等、每个
`output_text` 事件都带 `logprobs`、`response.completed` 里的 item id 与流式公布的一致、
incomplete 状态、扁平工具声明被回显、`message_delta` 带 `input_tokens`、以及 chat 的
`developer` 角色。

## 48. 第四种方言：对着参考文档审一遍

`/v1/systemone` 当初也是照着协议的形状写的，所以用了和 §47 一样的办法：逐字段读发布文档，
问"这里到底差在哪"。结果有两处要紧的偏差、四处不要紧的，而两处要紧的都是**静默**的。

**两个 bug。** 一是 `state` 只收字符串。参考协议收字符串或对象，对象就序列化——发布文档
里每一个示例发的都是对象，也就是说这个端点在拒绝最标准的请求形状。现在任何非字符串的
JSON 值都会被序列化进 `<state>`。

二是 `noul` 的 `criteria` 被当成普通选项列表解析：陈述的两侧变成名为 `true`/`false` 的
选项，顺序取决于 JSON 成员的书写顺序，两侧的描述被丢在 prompt 之外，而答案取的是"排在前
面的那一侧"的概率。于是先写 `false` 再写 `true` 的调用方，会拿到挂在 `noul` 键下的
P(false)——形状正确、数字错误，这是最坏的一种错。现在两侧顺序固定（真的一侧永远是 A），
描述也进 prompt。

**缺的字段。** `usage.cached_input_tokens`（参考协议里"从缓存提供的 state token"，
`reused_tokens` 是同一个数在我们这边的叫法，两个都报）。以及状态码的分类：body 根本不是
决策请求（JSON 损坏、缺字段、问题非法）返回 422，请求合法但本服务不提供返回 400；
超过请求体积上限的 body 现在在所有端点上返回 413，而不是 400。

**改成拒绝的字段。** `stream:true`、非 `"none"` 的 `reasoning_effort`、未关闭的
`thinking`、`images`。这四者原本被接受并静默收窄成一个不生成的 JSON 对象——要流的调用方
拿到一个单体，从回复里什么也看不出来。

**收紧到发布上限的部分。** `score` 档位 2..10（原先一级也收，会永远返回 0）、state 4 MiB
（原 32 KiB）、题面 64 KiB（原 4 KiB），以及 `instructions` 变为必填——旧的兜底文案
（"Which option applies?"）其实是在替调用方换掉问题。

**有意保留的差异。** 十个问题、26 个选项，对参考的 256 与 208：读数就是每个选项一个字母
token，而 tokenizer 只有 26 个字母；goal 要的也是一个 state 配一小把问题。`model` 用常驻
别名而不是部署路径，`temperature` 与 `object`/`adapter` 是扩展，OpenAI 形状的
`/v1/decisions` 未实现。另外端点确实响应发布路径 `/inference/v1/systemone`，所以 base
URL 以 `/inference` 结尾的客户端不用改。

**另外三个，对着一台真服务复核。** 用合成 checkpoint 把四个端点端到端重跑了一遍：
Responses 的 `sequence_number` 存在且严格递增、没有 `[DONE]`，终止 response 对象带整套回显
与 `error`/`incomplete_details`；Messages 的 `message_start`/`message_delta`/`message_stop`
齐全、delta 里带 `input_tokens`、预算用尽时 `stop_reason` 是 `max_tokens`；chat 在
`stream_options.include_usage` 下发出 usage 块并有 `[DONE]`；413/422/400 都出现在参考文档
说的位置上。§47 记录的那些缺口没有变化，仍按方言分别写在文档里（chat 省略可选的
`system_fingerprint`/`service_tier`；没有 `count_tokens`；不发 `ping`）。

**覆盖。** 626 项断言，新增十三条：两类拒绝、JSON 对象 state、四个不支持字段（以及两种
"关"的写法被接受）、必填的 `instructions`、`score` 档位范围，以及 `noul` criteria 的对象
与数组两种形态。

## 49. 「下载好了」不等于「能装载」

11-10 的 serve 日志收尾是五次 `POST /v1/models/switch`，目标是一个刚下载完的
18 GiB checkpoint，五次全是 500 `config is missing/invalid pad_token_id`；此后服务对每个
对话请求都返回 503 `no_model`。下载本身没有问题（19 个文件、无残留分片）。这里面有三个
各自独立的问题。

**报错在误导。** `cwick_model_spec_load()` 会拆开 `text_config`，之后所有字段都从内层对象读。
外层写着 `pad_token_id=248055`，内层是 `null`；`model_type=qwen3_5_text` 又不匹配两个专门
分支，于是被拿去按默认几何契约逐项校验，最后死在一个无关的 GLM 专有字段上。真正的原因——
这是一个稠密、混合线性注意力、带视觉塔的 checkpoint，本构建没有对应内核——从头到尾没出现在
错误里。`q_lora_rank`、`kv_lora_rank` 和 MoE 字段同样缺失，所以改 config 永远不可能修好。

**拆解顺序反了。** `srv_switch_model()` 在尝试装载替换模型**之前**就释放了所有引擎、
tokenizer、Metal arena 和 27 个前缀快照。一旦失败，实例就变成「没有模型」，之后每次重试
日志都写着 `releasing (none)`；第一次尝试已经 unmap 了 48 个 shard（475 GiB 地址空间）。
指标让情况更糟：`cwick:model_info` 是只在成功路径更新的静态 info 指标，于是采集端一直看到
旧模型名，而每次对话都返回 503。

**清单少说了一句话。** 目录发现只检查 `config.json` 加一个权重文件，模型库的 `complete`
也只表示「文件都到了」。任何仓库都能下载完、带着切换按钮出现在 Web 模型 tab 里，然后把服务
弄瘫。

**四处改动。**（1）共用一张类型表登记已有内核的架构；`cwick_model_spec_load()` 现在对未知
`model_type` 按名字返回 `CWICK_ERR_UNSUPPORTED`，不再掉进默认解析器。（2）`/v1/models` 与
`/v1/hub/models` 的每个条目都带 `supported`，为 false 时带 `unsupported_reason`。判定在目录
被发现时算一次并缓存进 catalog 条目，所以被轮询的模型列表不会在持有生成锁时读文件——控制面
与生成面仍然并发。Web 控制台禁用这些条目的切换按钮。（3）切换先对目标 config 跑一次只读的
`cwick_model_spec_load()` dry-run；架构不受支持或契约不通过就返回 400，常驻引擎、tokenizer、
arena 与前缀缓存一概不动。（4）新增实时 1/0 指标 `cwick:model_loaded`：拆解后装载失败置 0，
成功切换置 1；`model_info` 保留标签，因为给已导出的指标改名或改标签会打断采集端。

**这份清单有意不做成封闭清单。** 类型表登记的是*已实现的内核*，不是「永远只能跑三个模型」的
承诺：要支持新架构，就是加内核、再在表里加一行。模型库照旧下载任意仓库、照旧列出来，被拦住的
只有「设为常驻」这一个动作。

**仍未做回滚的部分。** 如果 config 通过了，但权重张量缺失或 tokenizer 装载不了，旧 checkpoint
已经被释放。在失败路径里再完整装载一次代价太大，所以实例仍然应答 503 `no_model`——但现在有
`model_loaded=0` 和明确的 500 原因，而且一次成功的切换就能恢复服务。

**覆盖。** 639 项断言，新增十三条。对着一台真服务（一个常驻合成 checkpoint、一个受支持的备用、
两个不受支持的条目）：两次不受支持的切换都是 0.1 ms 内 400，日志里没有 `switching to` /
`released`，`/v1/models` 仍 `loaded:true`，chat 仍 200，`model_loaded` 仍 1；受支持的切换返回
200 并重新发布身份；故意缺权重的 checkpoint 返回 500、`model_loaded=0`、chat 503；切回来
返回 200 且 gauge 回到 1。内嵌 Web 资源与 `web/app.js` 逐字节一致。

## 50. 删除已下载的 checkpoint

模型 tab 之前能下载仓库、却不能删：一个架构不对、几十 GiB 的 checkpoint 会一直躺在磁盘上，
直到有人打开终端。`DELETE /v1/hub/models/<name>` 把这一环补上。

**解析与删除是两次调用。** `cwick_hub_model_dir()` 把用户给的名字变成一个目录，
`cwick_hub_delete()` 负责删除。服务端跑在两者之间，所以它可以在不让模型库知道引擎的前提下
拒绝常驻模型（409）；切换路径复用同一个解析函数——「什么算一个本地模型」只有一处定义。
未下载完的目录可以解析用于清理，但不能用于切换：这两个问题本来就不同。

**边界靠 realpath，不靠字符串。** 名字含 `..`、反斜杠、绝对路径、空分量或 `.` 分量一律
拒绝；解析出的目录在 realpath 之后必须严格位于库根内，库根本身也不允许。删除用 `lstat`，
只对真正的目录递归，符号链接是被 unlink 而不是被跟随。递归深度上限 8。

**单测抓到的一个 bug。** `"."` 通过了字符检查（不含 `..`），解析成 `<root>/.`，在库根检查处
正确地被拒——然后掉进 owner 层查找，变成了 `./models/<owner>`。删 `"."` 会删掉一整棵 owner
目录。修法是加一道路径分量检查，同时拒绝空分量，于是 `./name`、`owner/`、`a//b` 都成为非法
名字，而不是被规范化成别的路径。

**拒绝什么。** 常驻模型（409）：它的权重正被运行中的引擎映射。仍有下载线程在写的目录
（409）：提示用户先取消任务，那才是让线程停下来的动作。两个检查都在真正删除之前。

**删完之后的一致性。** catalog 条目在同一请求里移除，否则 `/v1/models` 会继续宣传一个不存在的
root，每次切过去都是 400。指向被删目录的已结束任务标为 `canceled` 并写明「local directory
deleted」，UI 不会继续显示一个文件已经不在的「已完成」。

**有意不做的。** 没有回收站或软删除：一个 checkpoint 几十 GiB，副本不现实，所以确认放在 Web
控制台（写明文件数与大小），服务端执行不可逆删除。也没有后台删除任务：unlink 一棵目录树是
元数据操作，请求在完成时直接返回释放的字节数。

**覆盖。** 651 项断言，新增十二条：owner/name、裸名、扁平名三种解析；拒绝 `..`、绝对路径与
`.`；成功删除并报告两个文件、目录不再存在；不存在的名字 404；穿越的名字 404。对着一台真服务：
三种布局删除都 200 且字节数正确，`/v1/hub/models` 与 `/v1/models` 同步消失，常驻模型 409，
`..` / 绝对路径 404。

## 51. 那个走了五次弯路才答出「4」的 checkpoint

goal 模型清单的第四项是 Qwen3.5-9B：32 层，其中 24 层根本没有 softmax 注意力。它们
带的是 Gated DeltaNet——一种循环线性注意力，用每个 value head 一个 [128,128] 状态
代替 KV cache；剩下 8 层是普通 GQA，但有三处不同：Q/K 上各有一个 per-head RMSNorm、
只旋转每个 head 四分之一维度的 partial RoPE、以及把 `q_proj` 宽度翻倍的输出门。全部
BF16、18 GiB、没有专家，所以既不需要流式读盘，也不需要专家缓存预算。

**移植跑起来了，然后连着五轮吐垃圾。** 每一轮的数字看着都合理，文字全都无用：先是无尽
的空格，然后是无尽的换行，再然后是标点。出路不是继续盯着代码，而是**用 numpy 再写一份
实现**，读同一批 shard、打印同一批统计量，再和参考 C 逐层对比。

正是这个对比让问题显形。第 0 层 post-mixer 的均值、绝对最大值、前四个值**逐位到打印精度**
都相同（`mean=0.00455 absmax=13.636 x[0..3]=0.1058 0.2135 0.0251 -0.0635`），
第 1-3 层和最终 logits 排名也一样（`220(11.874)` 对 `220(11.877)`）。两份独立实现吻合到
这个程度，并不说明移植对了，而是说明移植**忠实地实现了同一个错误的想法**——要修的是那个
想法，不是代码。于是路线改成逐行读发布的模型源码。

**五个彼此独立的错误，全都是静默的：**

1. *卷积读错了行。* `src` 是块内的绝对位置，代码却把它加到了逐位置的基址上，于是位置 t
   读到的是块内位置 t+src。块长足够时看不出来，一旦块比窗口要求的短，前几次运行就越界读了
   未初始化内存，从第八个 token 起 conv 输出变成 0。正是这一点把最初的判断从「模型坏了」
   改成了「模型没坏，是卷积差了一个位置」。

2. *同一个文件里有两种 norm。* 这个 checkpoint 的每个 RMSNorm 都把权重存成相对 1 的偏移
   （`Qwen3NextRMSNorm`，初始化为 0，乘子是 `1 + w`）——第 0 层 input norm 的均值是
   0.033 而不是 1。但 DeltaNet 内部那个门控 norm 存的是普通乘子（`RMSNormGated`，初始
   化为 1）。两边都加 1 或者都不加，会以两种不同方式把模型弄坏；判断依据就是「一族权重贴着
   0、另一族贴着 1」。

3. *旋转的形状错了。* 这里的 RoPE 是 `rotate_half`：被旋转的那段前后两半互换，索引 i 与
   i + rotary/2 配对。移植用的是相邻配对 (2i, 2i+1)，也就是 GPT-J 的排布。两者都是旋转，
   数值范围都正常，但在每个位置上都不同。**这是最大的一处错误。**

4. *递推衰减的时机错了。* 定义的动作是先 `S = decay*S`，再用**衰减后的状态**算
   `delta = beta*(v - S^T k)`，最后 `S += k (x) delta`。先读后衰减，会让每个位置的修正
   都乘上一个本不该乘的衰减。

5. *`q_proj` 是按 head 交错的。* 输出是 `n_heads x (q | gate)`，因为参考实现把它 view 成
   `(..., n_heads, 2*head_dim)` 再切最后一维。把整块的前一半当 q 读，会把某个 head 的
   gate 混进下一个 head 的 query——而数值范围依然正常，所以没有任何东西报错。

**每一处是怎么定的。**（2）来自权重统计：均值为 0.033 的 norm 不可能是乘子。（1）（3）（4）
来自在 numpy 对拍证明「代码确实在做它被写成要做的事」之后去读发布实现。（5）同样来自那次
阅读，也正是它让「2+2」的回答从一串「2」变成了 `4`。

**最终结果。** 五处全修之后，checkpoint 中英文都能正确作答，用 `<|im_end|>` 干净收尾，
流式正常：`What is 2+2?` 返回 `4` 并停止；`用一句中文介绍你自己` 返回一句通顺的中文；
`17 times 23` 产出 7 步推理链、以 `391` 结束，正文也是 `391`。decode 实测 **4.85 tok/s**
（124 token 样本）——每步读 18 GiB、本机实测带宽上限 123 GB/s 对应 6.8 tok/s，也就是说
内核跑到了内存总线允许的约 71%。prefill 是 512 位置一块、按层推进，递推在块内逐位置走
（扫描本身不能加宽），24 层的状态合计 50 MiB，而 KV cache 若按同样长度会无界增长。

**没有实现的部分，以及原因。** 视觉塔（27 层 `model.visual.*`）和 MTP 草稿头（`mtp.*`）。
两者都无法从纯文本 prompt 触发：图像需要 `image_token_id` 位置展开成 patch，MTP 只对
本构建并未使用的投机解码有意义。字节留在磁盘上，从未被绑定。另外也没有实现 `mrope` 的
多模态位置分配——对文本而言三路位置相同，它退化为普通 RoPE。

**覆盖。** 该架构和其它三个一样由同一张类型表准入（`qwen3_5`、`qwen3_5_text`），所以
catalog 会标 supported、切换 dry-run 会放行。回归套件里「未知架构按名字拒绝」的用例改用
了一个不存在的名字（`mamba9`），不再用这一个。

## 52. 18 GiB 装得下，那就放进核里——顺带照出了两个串行段

Qwen3.5-9B 是这棵树里第一个**比机器小**的 checkpoint。它没有路由专家，所以什么都不流式、
也没有缓存预算要算：18 GiB 的每一个字节，每个位置都要读一遍。于是「让权重常驻内存」不再
是偏好，而是一个有答案的尺寸问题。

**策略。** `q35_resident_wanted(weights, state, host, ceiling)` 在「权重 + 状态不超过内存
一半」且「在配置上限之下还留 4 GiB」时回答 yes。是一半，不是全部：被 wire 的页收不回来，
一个要吃满半台机器的 checkpoint，会让长上下文的 KV、让这台服务器还能切过去的别的
checkpoint、以及让系统都没有地方待。内存总量未知一律回答 no——这个判断的意义就在于不做
没有测量依据的决定。`CWICK_Q35_RESIDENT=0|1` 两个方向都能覆盖。

**机制。** `cwick_safetensors_make_resident(saf, prefix, wire)` 先把某个名字前缀下所有张量
的字节区间收集起来、排序、把首尾相接的段合并（一层的那几个矩阵在分片里是挨着的，于是几百
次 `mlock` 收成几十次），然后逐段：`MADV_WILLNEED` 把预读排上，`mlock` 一次系统调用完成
「缺页装入 + wire」，**只有 wire 被拒绝时**才退化成走线程池的并行触页；最后 `MADV_NORMAL`
——因为 mmap 时装的 `MADV_SEQUENTIAL` 说的是「这份文件顺序读一遍就完了」，而这里要住下来，
留着它等于允许内核在读指针后面把页丢掉。释放是对称的，发生在
`cwick_safetensors_close` 里 `munmap` 之前（只 munmap 不 munlock，记账里会留下已经不存在的
字节）。KV 和 DeltaNet 状态也一起 wire——那是匿名内存，VM 宁可压缩也不换出，而被压缩的 KV
页会变成 attention 里面的一次停顿，而不是边角上的一次。

**买到什么。** 17.98 GiB 在加载期用 **2.6 s**（7.5 GB/s）装完并 wire 上，只付一次。44 token
的 prompt 上 prefill 从 **2370 → 1569 ms**（1.5×），因为第一块不用再在请求的关键路径上
吃 120 万次次缺页；760 token 的 prompt 是 13415 → 12207 ms（1.1×），剩下的都是真活。
decode 不动：每步 18 GiB 对实测 123 GB/s 上限就是 6.8 tok/s，而内核本来就在 7.0 tok/s
上跑着 92% 的总线。单发一个短请求刚好把 2.6 s 还回去；常驻服务只付一次。

**照出了什么。** 缺页风暴一停，prefill 才第一次能被诚实地分相（`CWICK_Q35_PROF=1`，每块
一行）。176 位置的块：mixer GEMM **748 ms**、MLP **1787 ms**、delta 递推 **2142 ms**、
attention **1415 ms**。也就是 58% 的 prefill 压在一个核上，而这两段本来是尴尬并行的，只是
当初照着公式串行写下来了：

- delta 递推在位置上不能并行（它是 scan），但在 value head 上可以：每个 head 独占自己的
  `[128,128]` 状态，从不读别人的。每层一次派发，一个 head 一个工作项。
- attention 只是看起来串行：这一块的 KV 存完之后，每个 (位置, head) 都是独立的。工作项是
  `(head, 位置分组)`，位置在各组之间**交错**分配而不是切成连续段——位置 400 的 query 的
  成本是位置 100 的四倍，切连续段等于把所有贵活塞给同一个 worker。每个 worker 自己申请
  打分向量；`m->scores` 只够一个 query，是共享 scratch。

同一块的结果：delta 2142 → **379 ms**，attention 1415 → **142 ms**，整块 6092 → **3009 ms**，
decode 一步 147 → **129 ms**（每 token 的递推原本就占 12 ms）。端到端，760 token 的 prefill
从 **45.1 s 降到 12.2 s**（17 → 62 tok/s），而输出与串行版**逐位相同**（`CWICK_THREADS=1`
对拍无差异）——并行版没有重排任何别的线程会读的东西。

**顺手发现的一个 bug。** `ssm_state` 是按**拼接后**的 `val_dim × key_dim` 分配的，是前向
真正索引到的 16 倍。它照样能跑，因为代码走的是 head 步长——只是白白占了 800 MiB 从来没人
碰的内存。改成按 head 分配（`heads × 128 × 128` = 每层 2 MiB）后，空闲 RSS 从 19.95 降到
19.24 GiB，`q35_cache_bytes` 与实际分配也对上了。

## 53. Qwen3.5 的前缀缓存一个字节都没写进去，而 attention 比它自己的 GEMM 慢 40 倍

两个缺陷都是读一个在跑的服务的日志读出来的，不是读代码读出来的：每个请求都打
`prefix capture failed: prefix: capture position 74 outside 0..0`；以及一条 31,690 token 的
prompt，prefill 速率从 73 tok/s 一路掉到 14.8 tok/s，eta 却越涨越大。

**缓存从来没存过一个字节。** `cwick_engine_pos()` 回答「这个引擎手里有多少个位置」，它只有
DeepSeek 与 MiniMax 两个分支；Qwen3.5 的游标在 `eng->q35->pos`。于是每次捕获都因为越界被拒，
`prefix_cache_hits_total` 永远涨不起来，`./pfxcache` 一直是 0 字节——功能是关着的，而且不说。
补上那个分支并不等于修好：`cwick_engine_capture_prefix` 也没有 q35 分支，补完游标它会掉进
mHC/KDA 那套布局去读 `eng->keys`，而这个后端从来不分配它。所以 Qwen3.5 需要的是另外两个后端
早就有的那套不透明 blob 契约：`q35_prefix_pack` / `q35_prefix_apply`，`kind = 3`，捕获与恢复
各一个分支。blob 先放固定部分（24 层的 conv + DeltaNet 状态，50 MiB），再放 8 个注意力层的
KV 行，这样 `pos` 可以只从长度反推出来——截断后的快照仍然能应用，而这正是「整段命中」那条路
需要的能力。

**然后是扩展仍然不命中。** 捕获通了之后，同一个 prompt 原样重发会命中（587 token：9556 ms 的
prefill 变成 **147 ms**），但同一段对话的下一轮不命中。实测：第 N 轮 232 token、第 N+1 轮
265 token，两者前 228 个 token 一致。把两个 `.pfx` 里的 token 列表 dump 出来才看清原因：
服务端会在 prompt 末尾补上思考块，而 `reasoning_effort: none` 时补的是**预先闭合**的那一个，
共 4 个 token（`<think>`、`\n\n`、`</think>`、`\n\n`）。下一轮把 assistant 的回复渲染成
「header + 正文」，从不重发这个块，于是两个 prompt 正好在这 4 个 token 的第一个上分开。
原来那个「少一个 token 的孪生快照」覆盖的是思考开启的情况（那时块就是单个开启标记），在这里
差 3 个 token。修法是：在最后 8 个 token 里找开启标记，把孪生快照建在那个位置；有了它，
第 N+1 轮复用 228 个 token、只需要 prefill 37 个，而不是退回 4 KiB 的块检查点。

**attention 跑在 8 GMAC/s。** 把数字摆在一起看毫无歧义：bf16 GEMM 路径约 550 GMAC/s，而
attention 内层——每次迭代一个标量乘加、中间夹一次 bf16 转换，并且每来一个 query 就把所有
K/V 行重读一遍——只有 8。两处改动，都不改变任何求和的顺序：

- **一次处理 8 个 query。** 工作项仍然是 `(head, 交错位置分组)`，所以一个 tile 是每隔 G 个的
  位置，里面 8 个 query 除了 `7*G` 个 key 之外全共享。每行 K、每行 V 现在每个 tile 只读一次，
  而不是每个 query 读一次——这就是 0.5 MAC/字节 与 4 MAC/字节 的区别；标量内核原本不是被总线
  卡着（实测 123 GB/s 里只用了 16），而是被指令数卡着，只做向量化会把瓶颈直接推到总线上。
- **点积与 axpy 走 NEON。** bf16 就是 f32 的高 16 位，左移 16 位再重新解释是精确的：没有查表、
  没有舍入，一条指令四个 lane。

用与出问题那条请求同形状的 prompt 各起一个进程实测：

| prompt | 改前 | 改后 | ~14K 处的单块 |
|---|---:|---:|---:|
| 760 token | 12.3 s（61.6 tok/s） | **10.0 s（76.2 tok/s）** | — |
| 7,600 token | 314 s（24.2 tok/s） | **102 s（74.4 tok/s）** | — |
| 14,060 token | 935 s（15.0 tok/s） | **214 s（65.8 tok/s）** | 63.3 s → **9.6 s** |

「改前」一列是拿那次 31K 实跑的日志拟合出来的曲线（`t(n) = 0.0117n + 3.90e-6n²`，对它自己的
进度预测误差 1%）；同一条曲线给出那条 31,690 token 的 agent prompt 是 4285 s，而新的分相
（14K 处 GEMM 4.5 s、delta 递推 1.1 s、attention 3.7 s）推算它大约 13 分钟。attention 仍然是
唯一随上下文平方增长的项（14K 时占一块的 38%），但它是 130 GMAC/s，不是 8。

输出没有变：三个固定 prompt 走接口产出的文本与改动前逐字节相同；760 token 的 prompt 在
`CWICK_THREADS=1` 与 12 线程池之间也逐字节相同——分块改变的是谁读什么，不是任何求和的顺序。

**而且窗口离失败只差一轮。** `max_tokens 65536 clamped to 1078: prompt 31690`——在
`--context 32768` 下，一条 30K token 的 agent prompt 只剩 1078 个输出 token，**下一轮根本放不
下**。这个 checkpoint 允许 262144，KV 是每个位置 33.6 KiB，所以 `start.sh` 现在要 131072：
在 17.98 GiB 已 wire 的权重之上再加 4.05 GiB 缓存，常驻策略依旧批准。

## 54. attention 是这棵树里唯一一个不是 GEMM 的内核

30K token 的 prefill 退化只来自一项，上一节已经点名了：attention。但 §53 留下来的是
一个手写分块内核，跑在 130 GMAC/s，而这棵树的 GEMM 路径跑在 790——所以那一项不是
「不可约的二次工作」，而是做坏了的二次工作。本节是把两者分开的那组测量，以及随后的重写。

**问题的形状。** 每个 512 位置的块里，除 attention 外都是平的：GEMM 4.5 s、delta 递推
1.1 s、激活 0.27 s。attention 随这个块能看到的 key 数增长——每个块线性，于是整条 prompt
是二次的。一次实跑的记录（块序号 → ms）：

    1: 75   2: 182   3: 271   5: 438   8: 872   10: 1081   15: 1558   22: 2867

也就是每 1000 个 key、每块 0.26 ms。折算到 30K 位置，一个块要在 attention 上花 7.8 s，而
32 层的全部投影只花 4.5 s。

**改之前它花在哪。** 按阶段插桩（一块、11K 处的线程时间）：QK 43%、softmax 6%、AV 51%。
不是单点瓶颈而是两个，而且都不是算力：QK 内层每 4 个 MAC 做一次 `vshll` + 一次 `fma`，并且
**每个 key 都把 query 向量从内存重读一遍**；AV 内层**每个 key 都把 1 KiB 的输出累加器读出来
再写回去**。1.45 GMAC/s 每线程——同一个线程在普通 f32 点积上能做到 16.8。

**两段各自能到多少。** `tools/attnbench` 在真实形状（512 query、16 头、256 维）和相同 MAC
数下单线程跑两段：

| 阶段 | 现有实现 | 手写重排 | cblas_sgemm |
|---|---:|---:|---:|
| score（QK） | 22.2 GMAC/s | 31.0 GMAC/s（1.4×） | **750.6（33.8×）** |
| output（AV） | 18.9 GMAC/s | 43.8 GMAC/s（2.3×） | **477.1（25.2×）** |

在恢复后的代码上重跑 `make tools/attnbench && ./tools/attnbench`（11,000 个 key）：
score 是 21.9 → 920.3 GMAC/s（42×），output 是 18.7 → 888.9（48×）。cblas 的绝对值会随
形状和机器状态变，40 倍这个差距不会。

「手写重排」是指每个 key 行服务四个 query、把展宽后的 key 留在寄存器里、16 个累加器全程存活。
它是真的有效，但离 GEMM 路径差 20 倍。原因不复杂：这两段就是 `S = Q·K^T` 和 `O = S·V`，而这个
模型里其他每一个投影早就走 Accelerate 了。结论是别再手写。

**重写。** `mixer_attn` 现在按（query 头, 128 query 的 tile）计算：

    S[nq x keys] = Q[nq x 256] · K[keys x 256]^T     （cblas，alpha = 1/sqrt(256)）
    掩码：每行超出自己位置的列置 -1e30，然后 softmax
    O[nq x 256]  = S[nq x keys] · V[keys x 256]      （cblas）

因果掩码作用在一个矩形上而不是跳过三角形：被掩的位置指数后恰好是 0，所以和原代码算的是同一
个和，而 GEMM 拿到了它喜欢的形状。同样的块上实测：QK **818 GMAC/s**、AV **537 GMAC/s**。

**一个在发布前就抓到的坑。** 第一版保留 bf16 KV 缓存，每层把可见行展宽到一块 f32 scratch。
这对 prefill 没问题（30K 时每层 4 ms），对 decode 是灾难：n = 1 时每个 token 会把**整个**缓存
展宽一遍、八次，只为算一个 query。现在缓存本身就是 f32，展宽整个没了。盘上的快照仍然是 bf16
（一半字节、同样预算下一倍快照数），pack 走 `q35_narrow_rows`、apply 走 `q35_widen_rows`；
代价是每个位置 64 KiB 缓存而不是 32，即 128K 时 8.1 GiB，对着 18 GiB 权重。

**decode 是另一种形状。** 一次只有一个位置，16 个头读的是同一行——一行 4 KiB 可以服务 16 个
点积（4 个 kv 切片 × 每组 4 个 query 头）。第一版用「每头一次 sgemm」，M=1 让 Accelerate 走进
一条坏路径：30K 上下文每步 149 ms，decode 掉到 3.4 tok/s。改成 gemv 仍然只有 13 GB/s——每次
只读 4 KiB 里的 1 KiB，预取器没法工作。最终 `dec_scores` / `dec_out_slices` 是行走内核：一行读一次、
用十六次。11K 处一步从 190 ms 降到 139 ms（7.2 tok/s），31K 处从 294 ms 降到 161 ms
（6.2 tok/s）；其中 score 段 8 ms、output 段几 ms，剩下的 111 ms 是权重。

**结果。** 同样的 prompt、各一个进程、`--context 131072`：

| prompt | 改前 | 改后 |
|---|---:|---:|
| 760 token | 10.0 s（76.2 tok/s） | 10.0 s（76.2 tok/s） |
| 14,060 token | 214 s（65.8 tok/s） | **184 s（76.2 tok/s）** |
| 30,970 token | 约 700 s（§53 曲线外推，末段 37 tok/s） | **436 s（71.0 tok/s）** |

11K 位置的一个块从 8.9 s 降到 6.8 s，其中 attention 从 2.87 s 降到 0.83 s。剩下那部分增长是
诚实的：QK 与 AV 每个块按定义就是 `O(keys)`，而 `16 头 × 512 query` 个分数上的 softmax 现在
占 attention 的 28%（11K 时每块 230 ms）。attention 仍然是唯一随上下文增长的项，但增长系数从
每 1000 key 每块 0.26 ms 降到 0.075 ms，于是速率贴着平地那段的 76 tok/s，而不是往 37 掉。

**精确性上付出了什么。** cblas 的累加顺序和原来四累加器的点积不同，所以 prefill 不再与上一版
逐位相同。输入仍然是 bf16 的 key 精确展宽成 f32、算术仍然是 f32，差别只在加法如何分组。三个
固定 prompt 走接口产出的文本与重写前相同，而贪心解码正是「如果不同就会暴露」的那个检验。原来
那个性质——760 token 的 prompt 在 `CWICK_THREADS=1` 与线程池之间逐位相同——现在是 cblas 的
性质，不是我们的。


## 55. 请求不带思考字段时，默认直接回答

**现象。** 指向 OpenAI 兼容端点的 agent 类客户端通常只发 `messages`、`tools`、
`max_tokens`，完全不带思考字段。`validate_body` 过去把这个情况默认成「极致」
（预算 32768、header `Max`），请求行打印 `thinking=极致 (effort=Max, default)`。
一个 37,638 token 的请求实测：`decode phase: thinking` → 52 token 后转 `content`
→ 再 21 token 转 `tools`，随后按 ~5 tok/s 流式吐出很长的工具参数、跑了几千步。
客户端只渲染那句开场白，之后一直停在「生成回复中」——工具参数增量不作为答案文本
渲染。单槽 + `CWICK_ABORT_ON_DISCONNECT=0` 意味着客户端停止也不会释放槽位；
客户端自己的 `max_tokens`（65536）还覆盖了启动参数 `--max-tokens`。

**根因的后半段。** 旧默认值不只是一个档位，它还是「缺失」。
`run_generation` 给三个模板渲染器传的是
`gr->effort_defaulted ? NULL : gr->effort_level`，这是有意为之：让每个 checkpoint
用自己的默认（DeepSeek header 75、MiniMax / Qwen3.5 的 `thinking=true`）。
所以只把默认档位改成「关」是不够的——prompt 仍会打开思考块，而引擎拿着 0 预算
立刻把它关掉，同一轮里两半互相矛盾。

**改动。** `CWICK_THINK_DEFAULT_LEVEL` / `_EFFORT` 定义为 `关` / `Low`，放在
`THINK_LEVELS` 旁边；日志回退也用同一对宏。删掉三处
`effort_defaulted ? NULL : ...`，让「字段缺失」以真实档位「关」走下去，与显式
`reasoning_effort:"none"` 完全同一条路径。`cwick_thinking_default_level()`
把该值暴露给测试套件，断言它等于 `resolve("none")`。四种显式拼写仍然优先，
未知拼写仍然 400。

**影响面。** chat / responses / messages 都要合成一个 chat body；不带思考字段时
合成出来的 body 也不带，于是三个方言自动跟随新默认，不需要各自加开关。
决策端点不变（它本来就拒绝任何非 `none` 的 effort）。内置 Web 控制台会显式发送
`reasoning_effort`，保持它自己选的档位。CLI `run` 没有请求体，与此无关。

**验证。** `make test` PASS，新增 1 条断言（合计 671）。在 18001 端口起了一个
临时服务（Qwen3.5，`--context 8192`），发五个请求实测：

| 请求 | 日志行 | 首个 decode 阶段 | think token |
|---|---|---|---:|
| chat，不带字段 | `thinking=关 (effort=Low, default)` | `content` | 0 |
| responses，不带字段 | 同上 | `content` | 0 |
| messages，不带字段 | 同上 | `content` | 0 |
| chat，显式 `reasoning_effort:"high"` | `thinking=高 (effort=High)` | `thinking` | 1 |
| chat，显式 `reasoning_effort:"none"` | `thinking=关 (effort=Low)` | `content` | 0 |

不带字段的 chat 响应是 `content:"2"`、`reasoning_content:null`、
`reasoning_tokens:0`，汇总行 `think=0 tools=0 content=1 think_cap=0`；
三个不带字段的请求都没有进入 `decode phase: thinking`，显式档位仍然优先。
要恢复旧行为：把两个宏改回 `极致` / `Max` 并恢复三处三元表达式——但那也会
把 prompt / 引擎不一致一起恢复。

**不在本次范围内。** `max_tokens` 仍由客户端控制、仍会覆盖 `--max-tokens`；
服务端硬上限是另一个独立决策。


## 56. 阶段标签一直存在，但没送到第四个引擎

**按 goal 第 10 条验收。** 一份 Qwen3.5 的 573 行服务日志（`--verbose --trace`，
15:03）里有 96 行 `decode forward ...` 和 2 行定期 `decode N/32768 tokens ...`，
`decode(` 出现 **0 次**。只有两行阶段切换提到了阶段名
（`decode phase: content at token 1`、`content -> tools after 27 tokens`）。
prefill 是好的：`[prefill] ~91/364 tokens (25.0%), layer 8/32 — 68.3 tok/s,
eta 4s`，且每行都带墙上时间和 `[req N]`。

**根因一：阶段提示只送到三个后端。** `phase_trk_observe` 每个 token 都会调
`cwick_engine_set_decode_phase`，而这个 setter 只把提示复制进 DeepSeek 与
MiniMax 的上下文，**漏了 Qwen3.5** ⇒ `q35_ctx.decode_phase` 恒为 NULL，
`q35_phase_label` 在每步行和定期进度行上都打印光秃秃的 `decode`。
`set_stop` / `set_abort_poll` / `set_verbose` 有同样的三缺一；现在四个 setter
都覆盖全部后端。这与 §19（每请求字段只被一个循环看到）是同一类问题，
只是晚了一个后端。

**根因二：thinking 阶段根本不会结束。** Qwen3.5 的生成 prompt 以
`<think>\n` 结尾，即开标记后面还跟一个换行 token。`run_generation` 只测
`ids[nids-1] == think_open_id`，于是判成「opened by the model」，以
`open_at_start=false` 装上预算。而 `cwick_think_plan` 只有在 `think_open`
为真时才注入闭合标记；模板已经写过开标记，模型不会重写，预算因此永不触发。
`CWICK_THINK_BUDGET=8` 下一个 130 token 的 `high` 请求实测
`think=130 content=0`。闭合标记的检测本来就对两个「尾部带空白」的模板用了
4 token 回溯窗口；现在同一个辅助函数（改名 `prompt_token_near_end`）也用于
Qwen3.5 的开标记，并用 `!pfx_closed_at_start` 保护，保证已由模板关闭块的
情况仍报「already closed」。

**改动。** `engine.c`：四个 setter 补 Qwen3.5 分支。`server.c`：共用窗口函数、
Qwen3.5 开标记检测、阶段切换行补该段速率。没有改内核、精度或采样。

**实测**（18001 临时服务，Qwen3.5，`--context 8192`、`--verbose`、
`CWICK_PROGRESS_MIN_MS=1000`，预算检查用 `CWICK_THINK_BUDGET=8`）：

| | 改前 | 改后 |
|---|---|---|
| 每步行 | `decode forward 1/32767 161 ms` | `decode(tools) forward 1/69 140 ms` |
| 定期进度行 | `decode 34/32768 tokens (0.2%)` | `decode(thinking) 8/26 tokens (30.8%)` |
| 阶段标签 | 无 | thinking / content / tools，每行都有 |
| 阶段切换 | `... after 27 tokens (3.9s)` | `... after 8 tokens (1.1s, 7.01 tok/s)` |
| 预算 8 | `think=130 content=0`，从未触发 | `wrote </think>`、`think=10 content=20` |
| 不带思考字段 | — | 不变：`reasoning_content:null`、content `2` |

预算修复对「显式开思考」是真实的行为变化：长思考块现在会在上限处收尾并给出答案。
默认不思考路径不变（`block already closed by the template`、
`decode phase: content at token 1`）。`make test` 671 断言 PASS。开标记窗口
只用于 Qwen3.5；MiniMax 仍用「最后一个 token」的精确判定，因为本次没有
实测支持改它。

**不在范围内。** 一个手写工具请求报 `model emitted undeclared tool
<function=get_time>`，那是工具格式不匹配，与日志/阶段无关，未动。


## 57. 为什么 tools 阶段前端没有流式展示

**问题。** 开场白之后，外部前端在整段 `decode(tools)` 里一直停在「生成回复中」，
而引擎明明还在跑：一个 38 工具的真实请求在
`content -> tools after 27 tokens (4.1s, 6.53 tok/s)` 之后又打了 637 行
`decode(tools)`，约 7 tok/s。

**实测。** 18001 临时服务，Qwen3.5，`stream:true`，声明一个工具，`--verbose`。
带到达时间戳的原始 SSE 帧：

| 方言 | 5.6 s 内帧数 | 标准工具增量 | 私有进度 |
|---|---:|---:|---:|
| chat | 15 | 0 | 12 帧 `delta.cwick_tool`，间隔约 141 ms |
| responses | 3（`created` / `in_progress` / `failed`） | 0 | 0 |
| messages | 2（`message_start` / `error`） | 0 | 0 |

chat 的第一帧是 `delta.content:""`，随后 12 帧把正在生成的工具块放进
**非标准**字段 `delta.cwick_tool`（如 `"<"`、`"function"`、`"=get"`、
`"_time"`、`"</tool_call>"`）。`delta.tool_calls[].function.arguments`
增量一次都没发。

**根因一：进度是私有字段，不是标准协议。** `stream_tool_progress()` 把
正在生成的块通过 `stream_delta_n(..., "cwick_tool", ...)` 发出，注释自己
写明这个字段「会被不认识它的客户端忽略」。唯一标准的 `tool_calls` chunk
在 `stream_finish()` 里由**完整解析结果**一次性构造，也就是这一轮结束时
才发（`server.c` ~3508）。`resp_stream_text()` 的注释更直接：「工具调用
进度在这里没有对应物：参数在轮末的 function_call item 里完整到达」。

**根因二：另外两个方言连私有字段都丢了。** `stream_delta_n()` 把
`cwick_tool` 交给 `resp_stream_text()` / `msg_stream_text()`，这两个只处理
content / reasoning / text，未知字段直接丢弃 ⇒ responses 与 messages
在整个阶段发 **0 个** delta。内置 Web 控制台是唯一消费者：
`web/app.js` 把 `d.cwick_tool` 累积进 `assistantMsg.toolStream`，
所以「渲染流式 toolCalls」这条要求只在自家 UI 成立。

**根因三（本次请求）：工具方言不匹配。** 模型输出
`<function=get_time>\n</function>`，而 `cwick_tools_extract()` 期望
`<tool_call>NAME<arg_key>...`。解析失败 ⇒ 服务端发
`{"error": ... "model emitted undeclared tool"}` 和 `[DONE]`，
**连轮末那次完整 `tool_calls` 都没发**。这正是 §56 标为「未做」的那个
不匹配。

**前端为什么卡住。** content 阶段的增量是标准 `delta.content`，开场白能渲染；
第一个 `<tool_call>` 之后 splitter 停止分类，线上只剩未知字段或什么都没有。
严格按协议解析的客户端会保留上一帧并显示「生成回复中」直到请求结束；
如果工具调用最终解析成功，它会在末尾一次性看到一个 `tool_calls` chunk。

**本轮未改协议。** 修复方向是三个方言各自补标准增量（chat：先发带
id/type/function.name 的 `tool_calls[i]` 起始 chunk，再逐片发
`function.arguments`；responses：`response.output_item.added` +
`response.function_call_arguments.delta`；messages：
`content_block_start(tool_use)` + `content_block_delta(input_json_delta)`），
同时保留 `cwick_tool` 给内置控制台。在途文本是清洗后的 XML 而非 JSON
arguments，所以还需要一个流式 XML→JSON 映射。根因三更小、更独立，
应当先修：否则客户端只是「看着一个最终仍会失败的工具调用」。


## 58. 按 §57 诊断的顺序修复 tools 阶段静默

§57 记录了三层原因并刻意没有改协议（那里「本轮未改协议」是历史叙述）。本节按当时给出的顺序落实：先修工具方言不匹配，再补标准流式增量。

### 一、工具方言不匹配（根因三）

嵌套模板的回答是

```
<tool_call>
<function=save_note>
<parameter=title>
Memo
</parameter>
</function>
</tool_call>
```

而 `extract_impl` 只认 `<tool_call>NAME<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`。于是整轮以 `model emitted undeclared tool` 失败，**连轮末那次全量 `tool_calls` 都没有**。`tools.c:translate_function()` 在唯一那个解析器看到文本之前，把嵌套形式改写成扁平形式——与 `translate_invoke()` 对另一个 checkpoint 的做法一致。值前后的换行/空格作为框架裁掉、内部换行保留，`</tool_call>` wrapper 视为框架丢弃，块外散文保留。已经用 `<arg_key>` 的文本不动；`<function=` 缺 `</function>` 时也不改写，让解析器照旧拒绝（fail-closed）。新增 7 条单测覆盖零参数、多参数、内部换行、数字与 JSON 值、散文、一个 wrapper 内多个 function、以及未闭合块。

真实 checkpoint 实测：原来以 `error` 收尾的手写工具请求，现在返回 `tool_calls:[{name:"get_time",arguments:"{}"}]`、`finish_reason:"tool_calls"`，访问日志 `finish=tool_calls calls=get_time`。

### 二、三个方言的标准增量（根因一、二）

`stream_tool_progress()` 仍把私有 `delta.cwick_tool` 发给内置控制台。与此同时 `tool_std_scan()` 观察在途块：模型一写出 `<function=NAME>`，`tool_std_open_frame()` 就用该方言自己的协议、以空的 `arguments` 开启调用；此后每完成一个参数追加一片增量。

| 方言 | 起始帧 | 参数增量 |
|---|---|---|
| chat | 带 `id` / `type` / `function.name` 的 `delta.tool_calls[0]` | `delta.tool_calls[0].function.arguments` |
| responses | `response.output_item.added`（`function_call`） | `response.function_call_arguments.delta` |
| messages | `content_block_start`（`tool_use`） | `content_block_delta`（`input_json_delta`） |

轮末三个收尾函数跳过第一个调用重复的 `added` / `delta` / `block_start`，但仍发 `done`、`output_item.done`、`content_block_stop`，所以只读终止事件的客户端看到的还是同一个对象。`tool_calls_json()` 增加 `empty_first` 模式，保证按增量拼接的 chat 客户端不会把已发的字节再算一遍。

**为什么线上字节与 payload 不会互相矛盾。** 分片不是靠第二套 XML 文法生成的：`tool_std_scan()` 取在途文本到最后一个 `</parameter>`，人工补齐闭合标签，然后调用非流式路径使用的**同一个** `cwick_tools_extract()`；把结果对象末尾的 `}` 去掉，剩下的就是一个只会增长的前缀。轮末 `tool_std_settle()` 拿它与解析出的 `arguments_json` 比较并补发后缀；若不一致，就让**已经发出去的字节成为 payload**——拼接增量的客户端与读取最终对象的客户端仍然一致。call id 在起始帧一次性分配并写回解析结果，Responses 的终止文档随后重建，信封、item 与增量指向同一个调用。

### 实测

18001 临时服务，Qwen3.5，`--context 8192`，`CWICK_SEED=42`，每个方言一个请求，抓带到达时间戳的原始 SSE。

| | §57 改前 | 改后 |
|---|---|---|
| chat | 15 帧，**标准工具增量 0**、私有帧 12 | 43 帧：1 个起始帧 + 3 片参数增量 + 37 私有帧 |
| responses | 3 帧（`created` / `in_progress` / `failed`），0 增量 | 9 帧，含 3 片参数增量 |
| messages | 2 帧（`message_start` / `error`），0 增量 | 8 帧，含 3 个 `input_json_delta` |
| 首个标准帧 | 无（只有轮末） | 5.0 s，即写完函数名那一刻 |
| 解析结果 | 拒绝（`undeclared tool`） | `save_note`，参数完整 |

一轮 9.6 s 的请求，chat 标准帧分别落在 5.0 / 6.5 / 8.3 / 9.3 s，进度确实在途。三个方言拼接结果都是
`{"title":"Memo","body":"hello world"}`，与非流式 payload、以及 Responses 的
`function_call_arguments.done` 逐字节相同；零参数调用三个方言都是 `{}`。
`make test` **678 断言 PASS**（新增 7 条），端到端 9 项检查 PASS，不带工具的普通问答 0 个工具帧。

### 内置 Web 控制台

`web/app.js` 过去把每个 `tool_calls` 帧都当成一个新调用，分片会产生空的重复卡片并反复清空原始进度视图。现在 `applyToolCallDeltas()` 按 `index` 归并：保留首个 id/name、追加 arguments、标准卡片开启后静音 `cwick_tool`。`respEventToChat()` 也把 `response.function_call_arguments.delta` 映射成进度（Messages 适配器本来就会消费 `input_json_delta`）。Node 里 9 项纯函数检查 PASS，覆盖旧式整帧、多调用数组、以及「首个调用流式 + 轮末多调用」。`src/webui_embed.c` 已重编。

### 明确留下的边界

- 分片粒度是**已完成的单个参数**。一个很长的参数值（最初报告里整份 HTML）仍在该 `</parameter>` 闭合时整体到达。要在值内部逐片流式，需要一个能同时保证 JSON 类型判定、不完整 UTF-8 与尾部空白裁剪三者单调的状态机，属于独立决策而不是调参。
- 只有第一个调用走流式，其余调用轮末整体给出。
- 其它方言的块（扁平 `<arg_key>`、`<invoke>` 形式）没有 `<function=` 开标记，因此不会提前开启，原有轮末全量行为原样保留。

## 59. 「在流式」和「看起来在流式」不是一回事

§58 让调用在书写过程中就开启，并按**已完成的参数**推分片。拿真实 agent 形态的
调用去量，这个阶段依然是静默的：agent 的工具调用几乎总是**一个很长的参数**
（一段 shell 脚本、一个文件正文），而「一个参数一片」意味着整个值在最后一次性
到达。

在真实 checkpoint 上、用 agent 实际会发的那种 prompt 复现：

| | 改前 | 改后 |
|---|---|---|
| 标准 `tool_calls` 帧 | 3 | 20 |
| 到达形态 | 5.1 s 开帧，然后**静默 4.1 s**，然后整个值 | 5.2 s 到 10.2 s 之间每 ~0.15 s 一帧 |
| 最大帧间隔 | 4.1 s | 1.1 s |

3 帧在算术上算「流式」，观感上是卡死。只认「第一个参数字节」才渲染的客户端，
整个阶段什么都不会显示——这正是反馈里说的「decode(tools) 阶段前端没有流式」。

### 改了什么

`tool_std_value_scan()` 现在切的是**正在写、还没闭合的那个参数**的值，而不是
只切 `</parameter>` 已经到达的参数。值交给与闭合路径**同一个**
`cwick_tools_extract()`：把块在安全字节处截断、人工补上闭合标签，因此转义、
空白裁剪和键顺序都不可能和最终 payload 分叉。

三件事让「值中间的切片」可以安全上线：

1. **类型单调性。** 只有当前缀绝不可能不再是 JSON 字符串时才发：值不能以
   `{ [ " -` 或数字开头；`t`/`f`/`n` 在仍是 `true`/`false`/`null` 前缀期间一律
   扣住。脚本、路径、句子放行；`137` 和 `{"a":1}` 不放行，仍走闭合路径。这个
   判定是单调的——一旦为真就不会再变——所以已发出的字节永远不需要撤回。
2. **尾部空白扣留。** 提取器会裁掉值首尾空白。若把稍后会被裁掉的尾随空格发出去，
   流式前缀就与 payload 不一致，因此尾部一直扣到非空白字节到达为止。
3. **`</parameter>` 扣留。** 闭合标签是逐片到达的；一个还可能长成
   `</parameter>` 的尾部属于框架而非值。没有这一条时，第一版把 `</parameter`
   发进了 JSON 字符串里，拼接结果非法——这个问题是被端到端检查抓住的，不是单测，
   所以下面那条检查才必须存在。

流式前缀会去掉末尾那个引号（提取器把它作为 `"value"` 的一部分加上），因此最后
一个值字节之后的那一字节，正好把前缀变成一个完整字符串。尾部不完整 UTF-8 与
正文路径同样扣留。

### 粒度

每次扫描都要重新解析整个块，所以对大值不是逐 token 跑：一个值的第一个字节立即
发出，之后每 12 字节一帧。6–7 tok/s 下，ASCII 约每 0.15 s 一帧、CJK 约 0.4 s，
解析开销有界。实测 decode 速率无变化（6.82–7.43 tok/s，基线 6.4–6.87），说明
扫描没有和 forward 抢资源。

### 验证

真实 checkpoint 上 11 项端到端检查全 PASS：chat 拼接与非流式 payload 逐字节
相同；有且仅有 1 个带 id 的起始帧；私有 `cwick_tool` 进度仍在发；零参数调用
三方言都是 `{}`。数字、布尔、对象参数保持 JSON 类型（`count:137`、
`meta:{"a":1}`），因为它们从不进入在途路径。三方言拼接字节完全一致且都能解析。
`make test` **683 断言 PASS**（新增 5 条：值中间的 prefix 必须是最终 JSON 的
字节前缀，且必须严格更短）。内置控制台用 7 项纯函数检查复核，含长分片序列和
收尾帧。

被 `max_tokens` 在调用中途截断的轮次，仍以既有的 `unterminated <tool_call>`
错误帧收尾——行为未变，且是对的：调用从未完成，没有合法结果可交回。

### 另外发现，未修

DeepSeek 的 prompt 模板教的是它自己的方言
（`<｜DSML｜ calls>` / `<｜DSML｜ invoke name="…">` / `<｜DSML｜ parameter …>`），
但 `cwick_tools_extract()` 只认 `<tool_call>`、`<function=`、`<invoke name="`
和 MiniMax 的命名空间 token —— DSML 一个都不命中。`cwick_stream_segment()`
同样只在字面量 `<tool_call>` 上进入工具模式。因此在一个严格遵循自身模板的
checkpoint 上，工具阶段既不会产生阶段标签，也不会有任何增量。这与本节修掉的是
**不同缺陷**（是方言缺失，不是粒度不足），需要先对一轮真实的 DS 请求做测量再改；
记录在此，以免后来者把本节当成工具阶段的完整修复。

## 60. Anthropic 客户端自己的 URL 形状就是一个 404

**症状。** 一个 agent harness 指向本服务，每一轮都收到
`404 {"error":{"message":"unknown endpoint","type":"invalid_request_error"}}`。
serve 日志里一行就说清了全部：

```
[req 1] POST /v1/v1/messages?beta=true (46045 bytes body)
[req 1] POST /v1/v1/messages?beta=true -> 404 in 1.1ms auth=key:cwick_N5TmAqb
```

这一条路径里躺着两个互不相干的缺陷，任一个单独存在都足以产生 404。

**原因 1 —— 生成路由比较的是原始路径。** 这个 handler 里其他所有路由都拿
`ponly` 比，那是函数开头 `path_only()` 写出的、剥掉查询串的形式；四个生成路由
是仅剩的仍在比 `req.path` 的地方。Anthropic 客户端给每个 `/v1/messages` 调用都
带上 `?beta=true`，于是 `strcmp(path, "/v1/messages")` 永远不成立 —— 这个端点
对它唯一存在的那类客户端不可达；`/v1/chat/completions?x=1` 同理不可达。

**原因 2 —— base URL 已经以 `/v1` 结尾。** Anthropic SDK 会在给定的 base URL
后面拼 `/v1/messages`。把它配成 `http://host:8001/v1`，请求就以
`/v1/v1/messages` 到达，路由表里没有任何一条认得它。拒绝它在技术上正确、在实用
上无用：要改的是客户端并不拥有的那个配置字段，而 404 的文本也没说这件事。

**改动。** `server.c`：在 `path_only()` 之后就地折叠重复的 `/v1/v1/` 前缀
（循环 + `memmove`），所有路由一起受益；四个生成路由改为比 `ponly`。原始路径不动，
访问日志和 `--trace` 行仍然打印收到的东西。不涉及内核、精度、采样或 prompt。

**实测**（同一服务、同一鉴权，POST 空 `{}`，让已路由的请求倒在 body 校验而不是
路由上）：

| 请求 | 改前 | 改后 |
| --- | --- | --- |
| `POST /v1/v1/messages?beta=true` | 404 unknown endpoint | 400 `max_tokens is required` |
| `POST /v1/messages?beta=true` | 404 unknown endpoint | 400 `max_tokens is required` |
| `POST /v1/messages` | 400（本就路由成功） | 400（不变） |
| `POST /messages?beta=true` | 404 unknown endpoint | 400 `max_tokens is required` |
| `POST /v1/chat/completions?x=1` | 404 unknown endpoint | 400 `messages must be a non-empty JSON array` |
| `POST /v1/responses?beta=true` | 404 unknown endpoint | 400 `missing input` |
| `GET /v1/v1/models` | 404 | 200，5 个已知模型 |

随后在**原来 404 的那个 URL** 上跑真实流式调用：`POST /v1/v1/messages?beta=true`、
`max_tokens 24`、一轮 user → `message_start … content_block_stop … message_delta …
message_stop`，`200 in 1341.9ms prompt=15 completion=1 finish=stop`。
`POST /v1/v1/models` 仍然 404 —— 归一化只关于路径，不关于方法。

**本节未修。** 路由通了之后，同一个 harness 会被告知
`model \`qwen3.5-9b\` is not served by this endpoint (the resident model is
\`qwen3_5_text\`)`：这个 checkpoint 的目录不在 `cwick_model_served_alias()` 里，
所以它用自己 `model_type` 的名字应答。这是 `/v1/models` 写明的契约，不是路由
缺陷 —— 按目录公布的那个 id 发请求即可。

## 61. Qwen3.5 decode：整步装进一个 command buffer（7.0 → 20.0 tok/s）

**这一节立足的那次测量。** 一个 decode 步搬约 16 GB，每字节几乎不做算术，所以它是带宽问题。
这台机器有两条通往内存的通路，宽度不同：CPU 集群从四个线程起就饱和在 **123.8 GB/s**，而 GPU 读到
**284.6 GB/s**（真实的链接式步进形状下是 368 GB/s）。第二个数字从 CPU 上一点都拿不到。旧结论
"decode 已经跑到总线的 93%，别再找了"——那个 93% 是拿 CPU 的通路算的。

**GPU 不会白送的东西。** 一次提交一条投影是亏的。用 `tools/q35step` 实测：一步里的 128 个 GEMV
（一层需要四个同步点 × 32 层）作为 128 次提交要 **118.8 ms**，因为每次 commit+wait 约 928 µs，
而其中只有约 58 ms 是内核。同样这 128 个 GEMV 编进 **一个** command buffer（`tools/q35one`）只要
**37.6 ms**——368 GB/s，而且工作集放大 4 倍结果不变，所以那是这个访问模式的 DRAM 速率，不是缓存假象。
墙是调度项，不是带宽。

**于是：每步一个 command buffer，递推也必须在 GPU 上。** 把一条投影的输出行拆给 CPU 池和 GPU
并行——DeepSeek 后端正是这么做的——在这里是净亏：`tools/dualbw` 实测 12 个 CPU 线程与一次 GPU
dispatch 同时跑合计 **213.7 GB/s**，而 GPU 单独是 328.9。两条引擎共用一根管子，CPU 分到的是更慢的
那份。剩下的唯一可行形状是：一步的全部算子都在 GPU 上，32 层放进一个**串行** compute encoder，
一次 commit、一次 wait、一次回传——logits。

**实现。** 三个新单元，分开写是为了让只有最后一个需要 ObjC：

| 单元 | 职责 |
|---|---|
| `q35_shader.c` | 16 个 Metal 内核，每一个都是 `q35_model.c` 里某个函数的孪生：BF16 GEMV、带 `1+w` 开关的 RMSNorm、embedding 取行、depthwise 卷积 + SiLU + 窗口滚动、逐头 L2 归一、DeltaNet 的门控与递推、逐头 Q/K 归一加部分旋转、KV 追加、打分、softmax、加权求和、输出门、SwiGLU、残差加 |
| `q35_gpu.c` | 纯 C。调度本身：哪条投影喂什么、norm 在哪、这一位置的旋转表、缓冲绑定。完全不认识 Metal |
| `q35_gpu.m` | Metal 隔离层：device、library、pipeline state、零拷贝缓冲绑定、encode 调用 |

权重用 `newBufferWithBytesNoCopy` 直接套在 mmap 的 safetensors 上——4K 上下文时 13.19 GiB，
128K 时 20.94 GiB，**零拷贝、零额外常驻内存**。KV 缓存、卷积窗口、DeltaNet 状态同样从
`q35_model_open` 已经拥有的缓冲上套过来，所以 CPU prefill 和 GPU decode 写的是同一批字节，
两者之间没有任何需要同步的东西。prefill 留在 CPU（它本来就有 71-80 tok/s）；只有单位置步进走新路。

**正确性。** `CWICK_Q35_GPU_CHECK=1` 会在同一位置上再跑一遍 CPU 步进并比较。它必须先恢复递推状态
（`q35_prefix_pack` / `apply`——DeltaNet 状态不像 KV 缓存那样能截断），否则第二次跑的就是另一步。
实测三个位置：

| 位置 | 最大绝对差 | 平均绝对差 | argmax |
|---|---|---|---|
| 43 | 0.00804 | 0.001855 | 一致 |
| 44 | 0.00332 | 0.000564 | 一致 |
| 45 | 0.00235 | 0.000277 | 一致 |

这是 bf16 权重下的 f32 重排噪声，而且随位置**衰减**而不是累积——两条路径算的是同一件事。端到端，
同一 prompt 下 90 个贪心 token **逐字节相同**。

这个对照抓到一个肉眼看不出来的真 bug：DeltaNet 的输出门读的是 `z[j]`，而门控投影是按 value head
索引的（`z[vh*hv + j]`），于是 32 个头全读了 0 号头的门。那时平均 logit 误差 1.77、argmax 不同；
修好后是 0.0002。

**结果**（`cwick run`，44 token prompt，63 个 decode 步，上下文 128K）：

| decode 路径 | ms/步 | tok/s |
|---|---|---|
| CPU 池 | 143 | **7.00** |
| GPU 整步 | 49.8 | **20.10** |
| GPU 整步，`--context 131072` + 前缀缓存，走 HTTP | 48.9 | **20.45** |

2.9×。47.4 ms 的墙里有 46.5 ms 是内核时间，即 341 GB/s，对上隔离测量里这条 GEMV 链的 368 GB/s——
已经到了这个形状配比允许的 93%，没有可回收的调度开销了。`CWICK_METAL_TIME=1` 每步打印
`commit=6us wait=47ms gpu=46.5ms`。

**本节没做。** prefill 仍在 CPU 上，而 31K 的 agent prompt 是 390 s 的 TTFT 对上 1.5 s 的 decode——
下一个值得搬的东西，而且是另一个问题（GEMM，不是 GEMV）。40 tok/s 则需要在这一步之上叠 MTP 草稿头。

## 62. 校验 k 个 token 比跑 k 步便宜，但草稿比省下来的还贵

**做成的那一半。** 一个 decode 步要读 16.4 GB 权重才产 1 个 token，所以超过 20 tok/s 的唯一
出路是「一次读权重产不止一个 token」：先草稿 k 个，再用一趟读同样 16.4 GB 的 pass 把它们
全部校验掉。校验这一半已经做完了，而且是精确的。

障碍在 DeltaNet 层：它带的是矩阵递推状态，不是 KV 缓存——只能重算、不能截断。要保留 k 个
已校验行里的前 j 行，就需要「第 j 行跑完时」的状态，而这要么每行单独再跑一趟（k 趟，白干），
要么每步拷 50 MiB。实际做法是：**状态开 k+1 个槽并滚动**。第 r 行读槽 r、写槽 r+1，因此任何
槽都不会被紧随其后的那一行覆盖，槽 0 永远是 live，保留 j 行就是把槽 j+1 `memmove` 到槽 0——
pass 期间零额外流量，结尾一次拷贝而不是 k 次。卷积内核原本是原地窗口（`st` 既是输入又是
输出），滚动槽会打破它；现在它多带一个源指针，两个指针指同一个槽时与原版逐位相同。

`CWICK_Q35_GPU_CHECK=1` 拿 k 行 pass 的**每一行**去对「同一条 GPU 路径逐行跑」的结果。
k = 2、3、4、8 在全部 32 层上都是 **max 0.00000、N 个里 0 个不同**——logits、KV、卷积窗口、
递推状态全对。本节余下的结论都立在这上面：k 行 pass 不是 k 步的近似，是同一种算术。

**一行的真实价格**（真实 decode 步，`CWICK_Q35_VERIFY=k`，19 趟，同一个 prompt，同一次会话）：

| k | ms/趟 | 边际 ms/行 |
|---|---|---|
| 1 | 55.5 | — |
| 2 | 63.6 | 8.1 |
| 4 | 73.4 | 4.9 |
| 6 | 85.6 | 6.1 |
| 8 | 98.4 | 6.4 |

边际成本随 k **上升**，这否掉了两种舒服的解释。不是带宽：无论 k 是多少，一趟都是读 16.4 GB，
真是带宽限制，边际就该是 0。也不是每次派发的固定开销：那会把 k = 1 那一行抬高、让边际变平。
它是多出来的那几行的算术，而且速率随寄存器占用变差——隔离基准（`tools/q35gemsw`，用这个
checkpoint 自己的 13.82 GB 形状配比、32 层）在 k = 8 是 4.5 ms/行，真实步里是 6.4。

也正是那个基准找到了本节唯一采纳的旋钮。内核是个 threadgroup 分块：k 条激活行只在块内暂存
一次，整组共享，每个 simdgroup 负责一个权重行。让一个 simdgroup 负责 RP 个权重行，同一份
暂存切片就复用 RP 次，激活侧流量降为 1/RP，代价是 RP*N 个活累加器：

| 行数 n | RP1 ms | RP2 ms | RP4 ms |
|---|---|---|---|
| 4 | 45.3 | 45.8 | 47.1 |
| 5 | 49.9 | 49.0 | 66.6 |
| 6 | 54.4 | 52.5 | 72.5 |
| 8 | 70.1 | 63.3 | — |

RP4 从 n = 5 起就溢出；RP2 从 n = 5 起赢、以下略输，这就是策略。RP 是这个内核里唯一能拧、
又不改变任何一次 FMA 在累加中位置的旋钮，也正是它让上面那条逐位校验还成立；把读宽放到
8 个 bf16 在 n = 2..4 又快 3 ms，正因如此没有采纳。落到真实步里，k = 8 是
**102.2 → 98.4 ms**（同场三次交错取最好）——比基准的 6.8 ms 小，因为真实 pass 里有一部分
不随 RP 缩放。

**草稿的价格。** 校验这一半接上之后，40 tok/s 就变成一句关于草稿的话：k+1 行的一趟必须产出
`A` 个被接受的 token，且 `A / T(k) >= 40`；而 `A` 是草稿的事，不是校验的事。

零成本草稿是 prompt 查表——拿末尾 2–6 个 token 去上下文里找上一次出现并顺着抄。它不读任何
权重，也不可能把结果搞坏，因为接受判定用的是全量模型自己的贪心 argmax，关掉它输出逐字节
相同。它也确实没用。两个 prompt、各 40 个 token：**每趟 1.08 到 1.24 个 token**（散文上
52% 草稿被接受，代码上 23%），端到端 **2022 ms 对基线 2011 ms**——更宽的那一趟比多出来的
token 更贵。免费不等于有用。

盘上那个学出来的草稿头是 MTP，它的文件把价格定死了：**486.6 MB**，十五个张量——
`mtp.fc[4096,8192]`、一整层 attention（GQA，`q_proj[8192,4096]` 带交错 gate，
`k_proj`/`v_proj` 是 `[1024,4096]`，`o_proj[4096,4096]`，SwiGLU 12288）加三个 norm——
而且**没有自己的输出投影**。所以每个草稿 token 要读 486.6 MB **再加上共享的 2.03 GB 词表
投影**：2.52 GB，按实测速率是 7.0 ms。

| k | T(k) = 一趟 + k × 7.0 ms | p = 0.8 时的 A | tok/s |
|---|---|---|---|
| 2 | 82 | 1.80 | 21.9 |
| 4 | 109 | 2.95 | 27.1 |
| 8 | 162 | 4.16 | 25.7 |

**这张表就是答案，而它不是 40。** 词表投影占一个草稿 token 的 81%，所以草稿 8 个要 56 ms，
而多出的 7 行只让校验贵了 43 ms——省下的正好又交回去了。接受率更高也救不了：p = 0.9 时
8 个草稿在 162 ms 的一趟上拿回 5.70 个 token，即 35 tok/s，要 40 得 **p ≥ 0.94**。

能改变答案的有两件事，按收益排。**让一个草稿 token 只花 ~1.4 ms：只投影词表里的候选切片**
——草稿是允许错的，判定在校验那边，所以这是个带宽问题、不是精度问题；前 4096 行是 33.5 MB
而不是 2.03 GB，这样 k = 8 是 118 ms、接受 4.16 个，即 **35 tok/s**，而 40 的门槛降到
p ≥ 0.87。以及**砍掉那 6.4 ms
的边际行**，它现在卡的是吞吐而不是带宽，也是唯一能同时改善所有 k 的杠杆。两件事都要先把
MTP 接上，而且都不是调参。

**顺带发现一个与上面全部无关的 bug。** 单行 decode 路径调用 forward 时，`ids` 里还是
prompt，刚采到的 token 从来没有写进 `ids[0]`，于是每一次生成的每一步都是在拿 `prompt[0]`
当条件，而不是拿它自己刚产出的那个 token。这会在十几个 token 内退化成重复
（`TheTheThe…`、`MASKMASK[g[g[g…`），也让本文档此前所有关于输出文本的判断失效。k 行路径
没有这个错误，这也正是它暴露出来的方式：同一个 prompt 走 `CWICK_Q35_VERIFY` 是通顺的、
走普通路径不是。修法一行。

## 63. 边际行的成本是激活流量，而它本来不必走 threadgroup 内存

§62 留了一个没解释的数：k 行校验一趟本该只花一次权重读，但真实步里边际行是 6.4 ms
（k = 1 时 55.5 ms，k = 8 时 98.4）。算术不是原因，而靠猜已经被证伪过两次，所以改成
把内核逐项摘掉来量（`tools/q35gemab`，用这个 checkpoint 自己的 13.82 GB 形状配比，
k = 8）：

| 内核保留什么 | ms | 它去掉了什么 |
|---|---|---|
| 全部 | 65.8 | — |
| 不读权重 | 36.4 | 那趟 13.82 GB 的 DRAM 扫读 |
| 不读激活 | 45.2 | k 行的 threadgroup 读 |
| 每个权重元素只做一次加法 | 64.0 | 全部 N 路乘法 |

三个结论。乘法加在一起只值 **1.8 ms**，所以边际行从来不是算术。权重读 **29 ms**、激活读
**21 ms**——而 36.4 + 29.4 正好是实测的 65.8，说明两条内存流**完全没有重叠**。第二点值得说白：
这从来不是带宽天花板，而是一个把两条本可交错的流串起来的写法。

激活读之所以存在，是因为这些行被暂存：每个 threadgroup 把 k 行拷一份进 threadgroup 内存，
组内所有 simdgroup 共享，每块前后各一个 barrier。那个设计是 §62 拿来用 threadgroup 带宽换
权重带宽的，而消融说明这笔换亏了。整个 pass 的 x 只有 N × k × 4 字节——k = 8 时 128 KB——
因此它在整趟里都驻留 L2，每个 threadgroup 扫同一批缓存行的代价远不是一次 DRAM 读。
直接从 device 内存读，就去掉了暂存循环、去掉了两个 barrier，k 循环也变平了。

| | k=1 | k=2 | k=4 | k=6 | k=8 | 边际行 |
|---|---|---|---|---|---|---|
| 暂存版，256 线程 | 37.7 | 41.0 | 49.3 | 62.4 | 65.8 | 4.0 |
| device 直读，128 线程 | 39.4 | 41.9 | 42.2 | 46.5 | 52.6 | 1.9 |

把线程组缩到 4 个 simdgroup 是同一个改动的一部分：暂存没了之后组内没有共享状态，更窄的组
只是意味着更多组能同时驻留。128 线程在每个 k 上都赢过 256（k = 8 是 53.1 对 57.7，两者都把
激活读向量化了）；512 线程比两者都差。

另外两个设置量过但没采纳。预取权重（PF = 2 或 4，先发若干 load 再消费）没有任何变化，这与
「开销来自暂存结构而不是内存级并行」的判断一致。每 lane 读更宽的向量在便宜的 pass 上更快、
在贵的 pass 上溢出，而且它会把求和重新结合，代价是逐位校验——与 §62 拒绝它的理由相同。

激活按四个标量读而不是 `float4`。投影读的那个 scratch 缓冲是按分配宽度而不是输出宽度排的，
所以行首不保证 16 字节对齐，而 Metal 里未对齐的向量读是**读错字节**、不是走慢路径。
标量形式只贵 0.5 ms（52.6 对 52.1），换来不必去证明对齐。

在真实 decode 步里（99 趟，三次交错取最好），同一个改动是 **k = 1 从 55.5 → 47.4 ms**、
**k = 8 从 98.4 → 81.8 ms**，边际行 **6.4 → 4.9 ms**。剩下的 2.1 ms/行不在这个内核里：24 个
线性层把卷积和递推**按行**跑一遍，每行六次 dispatch 加一个 barrier，所以每个边际行里都有一份
是派发开销而不是字节。这是下一件要处理的事，而且是另一个改动。

端到端走 API（200 token，`max_tokens` 生效，chat completions）是 **21.3 tok/s**，此前 18.3。
逐位不变量完好：`CWICK_Q35_GPU_CHECK=1` 仍在 32 层的 8 行上全部报 `0.00000`，因为这个改动
只删掉一次拷贝，没有移动任何一次 FMA 在累加中的位置。

**这对 §62 的草稿算术意味着什么。** 校验这一半变便宜了，草稿要跨过的门槛也跟着移动。
k 行的一趟产出 k 个 token，其中只有 k−1 个需要草稿，所以接受率 p 下期望接受数是
A = Σ_{i<k} p^i——p = 0.8 时 k = 2 / 4 / 8 分别是 1.80 / 2.95 / 3.95。按 MTP 头
7.0 ms/草稿 token 算，最好的是 k = 4：62 ms 的一趟加三个草稿是 83 ms 换 2.95 个 token，
**35.5 tok/s**。仍然不到 40，因为这时头已经占了成本的大半。改成让草稿只对一个候选词表
切片做投影（33.5 MB，约 1.45 ms；合法，因为判定在校验侧而不是草稿侧），k = 4 就是 67 ms、
**44.4 tok/s**，k = 8 是 92 ms 换 3.95 个、**43.0 tok/s**。所以「40 tok/s 需要 p ≥ 0.94」
不再成立：它需要的是一份只要 1.5 ms 的草稿，而那是带宽问题、不是精度问题。

**已被 §64 取代。** 头真的绑上了、切片也做了，实测接受率是深度 1 的 0.36、切片下的
0.17——不是 0.8。本节最后三段里的每个数字都由 §64 的表替换。

## 64. MTP 头接上了、算对了，依然是净亏

§62 预测 35.5 tok/s、§63 预测 44.4，两处都建立在「接受率 p = 0.8」这个**假设**上。
这一节把头真正接上、把 p 量出来，结论是那两个预测错得比它们自己的误差棒还大：按实测
接受率，任何投机形状都比「一步一行」更慢。

**绑定。** `model-mtp.safetensors` 是独立文件，而且不在 `model.safetensors.index.json`
里，所以文本分片的读取器够不到它；`mtp_open()` 直接映射（8 字节小端头长、JSON，然后
按名逐个解析张量，dtype、shape、`data_offsets` 与文件长度都校验）。十五个张量、
486.6 MB：`fc[4096,8192]`、一层 GQA（16 q 头 / 4 kv 头、head_dim 256、因为有交错的
输出门所以 `q_proj[8192,4096]`、partial rope 0.25、每头 q/k norm）、12288 的 SwiGLU，
**没有自己的输出投影**——词表头是共享的。它必须在 `q35_gpu_open` **之前**打开，因为
GPU 会用 `newBufferWithBytesNoCopy` 直接套用它的两块缓存。

现在 `mtp_one()` 只负责跑一个位置（融合、attention、SwiGLU，块输出留在 `g_m.res`），
链式草稿和缓存回填两个调用方共用它。新增三个内核：`q35_gemv_rows` 就是 `q35_gemv`
把行号改成从 device buffer 读；`q35_embed_id` 的 token 也从 device buffer 读，这样一个
proposal 的 token 在下一个 proposal 读它之前不必回主机（一次往返 0.93 ms，而一个块
1.6 ms）；`q35_argmax` 返回 index、平局取小，与它对照的 host 扫描一致。

**文件没写、只能实测的两件事。** 一是融合里 `emb` 在前还是 `h` 在前：对调之后接受率
归零（0/89），所以 `emb` 在前是对的。二是文本模型的隐藏态 `h` 取哪一份——残差流，还是
词表头看到的那份最终 norm 之后的拷贝：后者在第一层深度上是 **39.0% 对 35.7%**，现在是
默认（`CWICK_Q35_MTP_POSTNORM=0` 切回另一臂）。

**块自己的缓存。** 草稿要靠 keys 做 attention，而新序列的第一个 proposal 时缓存里只有
一个 key——它自己，这不是这一层训练时见过的形状。现在行按绝对位置索引、窗口 512 行
（`CWICK_Q35_MTP_WIN`），`q35_mtp_sync()` 把文本模型已经决定过的那些位置的行写进去，
数据直接取自校验趟留在 `g_s.x` 里的隐藏态：每个被接受的位置跑一次块，1.6 ms。窗口满了
`mtp_compact` 把活着的行整体下移——旋转在写行时就烘进去了，所以这个平移是精确的、不是
近似。prompt 的行仍然不回填（那需要 prefill 丢掉的隐藏态），窗口从最后一个 prompt 位置
开始。

回填买到的是 **35.6% 对 31.5%**（深度 1），代价 1.6 ms/步。它对深度 2、深度 3 没有用。

**实测接受率**（200 token prompt、生成 161、贪心、各跑一次；按深度看——只有前面全部被
接受才会走到深度 d）：

| 形态 | 深度 1 | 深度 2 | 深度 3 |
|---|---|---|---|
| 全 2.03 GB 头 | 35.6% | 12.5% | 0.0% |
| 4096 行候选切片 | 16.8% | 4.3% | 0.0% |

接受率在第一条链之后就饱和了。恰恰是「链」这一步没法指望：proposal i+1 喂的是本块
自己的残差，而不是文本模型的隐藏态，而检查点只按一个深度训练过。所以有用的形状是一个
proposal，数字就是这么说的。

**一行的成本**（同一 prompt，`CWICK_Q35_VERIFY`）：

| 行数 | 1 | 2 | 3 | 4 | 5 | 9 |
|---|---|---|---|---|---|---|
| ms | 48.4 | 57.4 | 58.2 | 60.7 | 64.0 | 83.5 |

离开一行要 **+9.0 ms**，之后每行只要 **+1.6 ms**。这一列才是真正的障碍，而且它不是
带宽：四行和一行一样只读一份权重。第一个额外行就吃掉整趟的 19%。

**端到端**（同一 prompt，各跑一次，与 20.66 对照）：

| 形态 | token/趟 | ms/趟 | tok/s |
|---|---|---|---|
| 一行、不投机 | 1.00 | 48.4 | 20.66 |
| SPEC=1，全头 | 1.35 | 66.9 | 20.21 |
| SPEC=2，全头 | 1.39 | 74.7 | 18.59 |
| SPEC=1，4096 行切片 | 1.17 | 60.8 | 19.03 |
| SPEC=2，4096 行切片 | 1.18 | 63.4 | 18.54 |
| SPEC=3，4096 行切片 | 1.18 | 67.8 | 17.33 |

每一臂都是亏的，所以 `CWICK_Q35_SPEC` 默认保持关闭。

**不是 40，也不是 35。** 约束是「一个 token 读一份权重」：40 tok/s 要求一趟产出 1.93 个
token，而这个草稿侧的上限是 1.35。§62 说「需要 p ≥ 0.94」、§63 说「需要一份只要 1.5 ms
的草稿」——代数都对，输入错了：切片下的草稿**确实**只要约 1.7 ms，但一趟仍然是 60.8 ms，
因为第二个行要 9 ms；而 p 是 0.17，不是 0.8。这两节里的 35.5 / 44.4 / 43.0 用上面的表替换。

**正确性。** 同一 seed 下 `CWICK_Q35_SPEC=0/1/2/3` 的输出文本逐字节相同：只有纯 argmax
的采样器才允许接受草稿，所以带采样的请求根本不会投机，贪心请求的输出与单行路径完全一致。

**还剩什么，按体量排序。** 第一，把两行的校验从 57.4 压到 50 ms——+9.0/+1.6 这个台阶是
内核形状造成的异常（四行和一行一样只读一份权重），它一次性地对所有投机形状收税。第二，
改善候选集：散文里最近 token 窗口只有约 60 个不同 id，切片为了省 5.5 ms 放弃了 19 个点
的接受率；用上一步自己 logits 的 top-K 行（本来就在主机上）去扩充这个集合是最便宜的办法，
能把单草稿一步拉到约 21.4 对 20.7。这两件都不会把 40 变成可能。

## 65. 第二行是 8 ms 而不是 9，草稿侧依然付不起它

§64 结尾留了两件事、按体量排序：把两行校验从 57.4 压到 50 ms，以及改善候选集、
让切片不再丢 19 个点的接受率。两件都做了。第一件从 9 ms 里拿回 2 ms，第二件从
19 个点里拿回 2 个，结论都没变。

**第二行的成本到底在哪。** 第一个要排除的是 k 行 GEMV，因为它是整趟里唯一换了内核的
部分：`CWICK_Q35_GPU_TILED1=1` 让单行走它，**49.5 对 48.0 ms**，3%。所以 9 ms 里有
7 ms 在别处 —— 那些在 n ≥ 2 时按行各跑一遍的东西：每层两个 norm、两个残差加、SwiGLU
乘，外加 DeltaNet 的门对。每层 9.5 个 dispatch，按 `CWICK_Q35_GPU_MAXL` 的斜率算是
**0.21 ms/层**（8/16/32 层分别是 +2.66/+4.51/+7.71 ms，即 0.21 ms 每层 + 1.0 ms 固定）。

**把它们批量化**只是改发射次数 —— 同样的算术、同样的求和、同样的顺序，所以一个 bit 都
不会动。新增五个内核（`q35_rmsnorm_rows`、`q35_add_rows`、`q35_silu_mul_rows`、
`q35_l2norm_rows`、`q35_gdn_gates_rows`），行号从 grid 拿、行 stride 作参数；
`CWICK_Q35_ROWFUSE=0` 把逐行循环放回去，这样这个断言在一个二进制里就可测。结果，同一
session 内 back-to-back、n = 2：**199 趟 56.61 → 55.81 ms**。所以一个 dispatch 值
约 5 µs、不是 22 µs，那 9 ms 从来不是发射延迟。

**n = 2 时的 RP。** `cwick_q35_gemv_n_rp` 一直是常量 2，因为那张表是在 n = 4..8 时测的
（那里 RP4 会 spill）。n = 2 时只有八个活累加器，多出来的激活复用是白得的：RP4 是
**54.61 对 55.67 ms**。n = 3 就已经退化（RP2 的 54.41 对 RP3 的 61.35），所以策略是
`n <= 2 ? 4 : 2`。RP 不移动一个累加内的任何 FMA，所以 `CWICK_Q35_GPU_CHECK` 断言的
逐位不变量不受影响 —— 两行校验仍然每个位置都是 `max|gpu-cpu| = 0.00000`。

**剩下那部分不是激活重读。** 一个 threadgroup 覆盖 RW × RP = 8 行权重，所以激活切片被
读 `cols/8` 次；把 group 加宽应该能砍掉一半。加了 `CWICK_Q35_GEMVN_RW` 去扫，结果是
**RW 4 / 6 / 8 → 55.9 / 54.9 / 56.3 ms**，落在跑间噪声里；而 n = 1 时同一个内核反而从
49.5 掉到 52.1。每多一行剩下的那 6 ms 没有解释清楚，也不值得再追 —— 其它都固定之后，
两行一趟是 54.6 ms，而这个数只有在「草稿足够对、能把这个行填上」时才有意义。

**候选集。** §64 提过用上一步自己的 logits 去喂短名单（校验趟已经把它拷到主机上了）。
做了：`CWICK_Q35_CAND` 的一半是最近窗口，另一半是与最高 logit 相差
`CWICK_Q35_CAND_T`（默认 2.5）以内的所有行，上限 `CWICK_Q35_CAND_LCAP`（默认 512）。
深度 1 的接受率 **16.8% → 18.5%** —— 而一份草稿要为每个参评行付 8 KB 的词表头，所以
2048 行要 4 ms、512 行要 1。宽的那版用 4 ms 买 2.6 个点，是亏的；窄的那版用 1 ms 买
1.7 个点，是打平。用阈值不用 top-k，因为尖的分布只给出几行、平的分布能填满它那一半，
两种都不需要排序。

**端到端，同一 session，200 token prompt，生成 161-200，贪心：**

| 形态 | token/趟 | ms/趟 | tok/s |
|---|---|---|---|
| 一行、不投机 | 1.00 | 48.1 | **20.74** |
| VERIFY=2、不带草稿 | 1.00 | 56.2 | 17.79 |
| SPEC=1，512 行切片 | 1.18 | 60.0 | 19.72 |
| SPEC=1，2048 行切片 | 1.18 | 59.2 | 19.94 |
| SPEC=1，全 2.03 GB 头 | 1.35 | 66.8 | 20.19 |

每一臂仍然亏。草稿最好能给 0.35 token/趟（全头），而它填的那个行要 8 ms，所以交叉点在
「0.35 > 8/48」，即切片接受率得超过 16.7% —— 它没有。

**为什么 40 在这个检查点上不可达。** 在本机 341 GB/s 下一份权重读是 48 ms，所以 40 tok/s
需要**每趟 2.08 个 token**，对应的接受率是 Σ p^i 且每个深度 p ≥ 0.52。全头给的是
0.356 / 0.125 / 0.0，求和 1.40，而且这就是任何 k 的上限 —— 因为那个块只按一个深度训练过，
再往后链自己的残差就出了分布。这个仓库里没有任何东西能提高一个属于权重的比率。能改变它
的是：一个 MTP 头按多深度链训练过的检查点，或者一个能在 2 ms 内出草稿的小模型。两者都不在
模型目录里。

**正确性。** `make test` PASS（678 断言）。`CWICK_Q35_SPEC=0/1/2` 输出逐字节相同，两行
校验的 `max|gpu-cpu| = 0.00000`。`CWICK_Q35_SPEC` 默认仍关闭。

## 66. `serve` 每个会话的第一个请求就死：reset 里 256 倍越界

日志停在 `[req 1] generation slot 0`，没有任何错误行，而进程已经不在了。同一个上午的两份崩溃
报告栈完全相同：

```
__bzero + 12400
q35_engine_reset
cwick_engine_reset
handle_client
client_thread
```

`EXC_BAD_ACCESS / SIGBUS / KERN_PROTECTION_FAILURE`，写在**一个 ~11.86 GiB 区域的最开头**
（`bytes after start: 0`）。`__bzero` 是大块 memset 的路径，而调用方在清草稿头的 KV 缓存。

**根因。** MTP 缓存改成 512 行滑窗（§64）时，分配、`bind_tensor`、缓存计费、`mtp_compact`
全都跟着改成了 `mtp_win`——但 `q35_engine_reset` 没改：

```c
/* 分配：mtp_win 行 */
m->mtp_kcache = xcalloc((size_t)m->mtp_win, (size_t)kvw * sizeof(float));
/* 清零：max_seq_len 行 */
memset(qe->m.mtp_kcache, 0, (size_t)c->max_seq_len * ...);
```

每个 2 MiB 的分配被写了 512 MiB，两块，即 **256 倍**。越界一路冲出堆，撞进一段不可写的映射，
内核在那一页上直接杀进程。什么都不打印，因为故障是即时的、没有 handler。

**为什么 CLI 从来没露过。** `q35_engine_reset` 只经由 `cwick_engine_reset` 到达，而后者只有
`handle_client` 调用——服务在请求之间重置状态。`cwick run` 建一个引擎、从不 reset，所以
§62–§65 里所有 `run` 的测量都不受影响。这个回归是随滑窗引入的，也只可能在 `serve` 里出现。

**修复。** 按 `mtp_win` 清，并加 `mtp_win > 0` 保护。在与崩溃那次相同的配置上验证（context
131072、80 GiB 前缀缓存）：125 KB body / 25298 token 的请求返回 200，连续三轮请求不再
reset 故障，`start.sh` 起回 8001 后正常服务。

**顺手核查过、不是缺陷的几件事。**

- *短 prompt 固定约 1.1 s 的 prefill*（6 token，CLI 与服务一致）。那是把 17.98 GiB 权重完整
  读一遍——稠密 9B 模型产出任何一个 token 都必须碰每个矩阵，而 CPU prefill 路径只跑到约
  16 GB/s（机器能到 120 GB/s）。长 prompt 上它被摊掉了：25298 token 实测 74.4 tok/s。
- *某次测试里 `prefix_cache_queries_total` 为 0* 是我自己的配置——只有给了
  `--prefix-cache-dir` 才开缓存。开了之后计数器是对的：queries 2、hits 1、复用 25297 token，
  同一条 25298 token 的 prompt 第二次从 **340 s prefill 变成 0.206 s 的 TTFT**。
- *79.65 / 80.00 GiB 的缓存占用*是缓存正常工作、不是泄漏：`evict_to_fit` 在 open 时和每次
  store 之后都跑。但账值得记住——一个 25K token 请求会写三份快照（prompt 前缀、off-by-one
  那份、prompt 加生成 token），这个检查点每份约 1.3 GB，所以 80 GiB 大约是 20 轮 agent
  会话，之后最旧的开始被淘汰。
- *日志里那行 `--trace` ON* 来自 `start.sh`，是它故意传的（goal.md 要求这个开关）；打印
  prompt 与输出时是截断的。

**教训是机械的。** 一块缓冲区有四处必须对尺寸达成一致：分配、重置、计费、以及任何把它序列化
的地方。§64 动了其中三处。这四处之间没有 `_Static_assert` 可用，所以最便宜的一道防线是：
**分配取的是运行时字段时，重置就不该取配置里的常量**——`mtp_win` 是字段、`max_seq_len` 是
配置，上面两行不一致的正是这一点。

## 67. serve 日志该按缓存命中率读，不该按延迟读

`logs/cwick-serve.log`（10:07:25 至 10:25:25，三轮 42K token 的 agent 会话，全部 200，
没有 warn/error/failure 行）只看状态码会显得一切正常。数字不是这样：每一轮都写着

    prefix cache hit: 31785 of 42158 prompt tokens replayed, 10373 left to prefill
    ...
    prefix cached: 48087 tokens (251 ms, background)

也就是每轮都存了一份完整的 42K/48K 快照，而**下一轮仍然只复用了那 31785 token 的静态
system+tools 头**。三轮端到端分别 189.2 s、603.6 s、196.5 s；其中约 178 s 都在重新
prefill 那约 10K 的尾部——本该由刚写下的快照覆盖。`logs/cwick-serve.log.crash-20261113`
是另一次更早的故障，日志停在 `[req 1] generation slot 0` 且没有响应，即 §66 的 SIGBUS。

**根因：被淘汰的正是刚写入的那一条。** 旧的代价函数用 `longest - pos` 排序，把最长的
条目算成最该淘汰。而增长型会话里最新的快照按定义就是缓存中最长的，于是 `evict_to_fit`
把刚写的文件选成自己的牺牲品，在报告成功后几微秒就 unlink 掉。缓存是 79.13 / 80.00 GiB、
184 份快照，所以**每次** store 都会触发淘汰——这不是偶发，是每轮必现。

现在淘汰分两级（`hits > 0` 的条目优先于从未被复用的，同级按最后使用时间），并有两条绝对
保护：刚写入的条目，以及最短的前缀（所有更长 prompt 的公共基座；冷启动时第一份快照到达时
它还没来得及攒到命中）。其余条目仍然可被淘汰——包括命中过的，按最后使用时间排序——
所以缓存不会无限增长。当只剩这两条受保护条目时，允许超出一个文件的容量，而不是删掉其中
任何一个。两条回归测试分别覆盖「24 份后续快照」与「单调增长的会话」；`make test`
686/686 PASS。

**核验时还发现一个构建级故障。** 管线枚举有 26 项而 `P_NAME[]` 只有 24 项，多出的两项
属于两个既没有派发函数、也没有调用点的分组共享 attention 内核。正在跑的二进制是在这次枚举
改动之前编译的，所以日志里每次启动都还是 `24 kernels`。一旦重新编译，`q35m_init` 会走到
名字表末尾读 NULL，GPU 初始化失败——一条没有任何测试覆盖的静默回归。已删除死内核与枚举
项；用独立 Metal 探针编译生产 shader 源码并建全部 24 条管线：`OK: 24 kernels, 0 bad`。
该探针只是一次性验证，不是随项目交付的工具。

**metrics 数据是健康的，但不完整。** `/v1/metrics` 暴露 44 组指标；在
`uptime_seconds 12.5` 抓的快照里所有 counter 都是 0，只是因为当时还没有请求完成；
`process_footprint_bytes` 28.50 GB 与 17.98 GiB 权重 + 8.44 GiB 状态 + 464 MiB MTP 头
相符。真正的缺口是：goal.md 禁止 swap 与压缩，但唯一证据只有启动那一行
（`host swap 0.86 of 2.00 GiB used, compressor ... since boot`）。现在新增五个主机级
gauge 持续暴露：`cwick:host_swap_used_bytes`、`host_swap_total_bytes`、
`host_wired_bytes`、`host_compressions_total`、`host_swapouts_total`。之所以是主机级，
是因为 VM 接口没有进程视角：非零的绝对值本身不构成违规，要看一个采集窗口内的增量。独立
渲染探针确认数值与启动行一致（928514048 / 2147483648 / 1225496103）。

本轮未改动任何模型精度、权重、后端或上下文行为。

## 68. 空回复是两个状态机没有碰头

一个 50,783 token 的 agent 轮次跑了 773.9 s，返回 `content: ""`。所有表面都是
成功：HTTP 200、`finish_reason: "stop"`、`request_success_total` +1，2161 行日志里
没有一条 warn 或 error。唯一痕迹是访问日志尾部 `think=25 tools=0 content=0
think_cap=512`，以及 trace 打出来的那段原文：

```
<tool_call>
<function=Bash>
</function>
</tool_call>
```

模型决定了要执行一条命令，在思考块内部就把调用写了出来，然后停止 —— 从头到尾没写
`</think>`。

**四层各自都做了合理的事。** `split_reasoning()` 按 `</think>` 切；没有闭合标记就把
全文归到 reasoning，交回空的 content。`cwick_tools_extract()` 只读 answer，于是看不到
调用。`finish_reason` 由调用个数推出，于是写 "stop"。Responses 那条「没有文本的轮次
也要发一个 message item」的规则，于是把 content 为 `""` 的 message 发给了客户端。
没有任何环节失败，用户看到的是一个空气泡，没有任何动作被执行。

**这个机制需要思考是开着的，而这正是关键。** 思考关闭时模板已经把块闭合
（`<think></think>`），答案从 content 起步（`closed_at_start`），`<tool_call>` 只能被
当成调用读出来。该实例上的 13 轮里，唯一一次空回复出在 7 个思考轮次中；不思考的 8 轮
一次也没有空过。所以「默认打开思考」不仅治不了这一轮 —— 它会让每一轮都成为候选。同一
检查点上的 A/B：`think_cap=512` 连跑 3 轮，全部正常闭合、全部返回 `tool_calls`。这是
一条随上下文长度与工具数量增长的长尾，不是一个坏掉的开关。

**两处改动，都很窄。** 现在会挽回尾部的调用，但必须同时满足：本轮是 chat 且声明了
工具、块确实没闭合（输出里任何位置都没有 `</think>`）、文本末尾（去掉空白后）是完整的
`<tool_call>` 块、每个块都指向已声明的工具、且每个块至少有 1 个参数。参数这条是承重的：
观测到的那次调用参数为空，把它提升出来等于递给客户端一个 `Bash({})` —— 必然失败，而
agent 对失败的回答是重试。第二条规则是：如果还有更早的调用会被留在 reasoning 里，就
拒绝提升 —— 被悄悄丢掉的动作用比一个看得见的空回复更糟。全部判定在
`cwick_tools_trailing_calls()` 里，九个单测用例钉住了这张矩阵（可挽回、尾部空白、
零参数、未声明工具、调用不在末尾、两个连续调用、调用之间有散文、没有调用、大小写折叠）。

什么都不满足时这一轮依然是空的 —— 但不再沉默：`cwick:empty_answers_total` 统计每一个
「生成了 token、却既没 content 也没调用」的完成请求，访问日志在 info 级别多一条
`empty answer:`。这个 counter 是这种失败唯一发出的信号，所以任何看板都该把它和失败率
放在一起。被挽回的那一轮访问日志尾部带 `promoted=N`：阶段计数是从 token 流里读标记的，
因此那里仍然写着 `tools=0`。

流式不需要特殊处理：参数此前是作为 reasoning 增量发出去的，一个 `tool_calls` 增量都没
发过，所以 `stream_finish()` 会在收尾块里把解析出的调用一次性宣告出去
（`tool_std_settle()` 在没有流式调用时返回 false），`resp_stream_finish()` 同样会补发
function_call item。重复的那段文本落在 reasoning 里，客户端会当思考渲染，不会当答案。

**这次没有修的东西。** 同一会话里那 45K 提示在盘上明明有快照，却整段重新 prefill，
占掉 773.9 s 中的 688 s。那是 §67 的缓存缺陷，不是这一条。

## 69. 一个 turn 写三份快照，§67 的保护只盖住最后一份

§67 修掉了「按长度淘汰刚写入的条目」。生产日志说明还不够：14:09–14:41 的 53 个 chat
请求里，缓存 79.90 / 80.00 GiB、49 份快照，盘上留着 `50458 → 54690` 的连续链和最后一份
`55023`，而 `54726 … 54965`（最近 8 轮写入的）全都不在。50 个温请求**零个完全命中**，
命中点固定停在「上一轮 prompt 末尾 −2」，每轮重复 prefill 81–318 token（2.4–7.1 s），
累计 100,522 token；最后 3 轮的 `left` 从 81 涨到 160 / 239 / 318，prefill 从 2.9 s 涨到
7.1 s。对照：同期小 prompt 的会话正常，8 个温请求 6 个完全命中、平均 prefill 0.35 s。

**根因：一个 turn 写的是一批，保护只认最后一份。** 一轮会存三份快照——prompt 全长 P、
思考块边界的孪生 twin、生成结束后的会话快照——而 `last == c->seq` 只保护刚写入的**那一条**。
于是第 3 次 store 淘汰第 1、2 次写的，第 2 次淘汰第 1 次，下一轮自己的 store 再淘汰上一轮
的第 3 份。孪生是唯一与下一轮真正对齐的那份（客户端不回传 `reasoning_content`，助手回合
重渲染后只在块边界之前一致），存活时间正好是 0 个 turn —— 超过 45K 的会话因此永远只有
一个不再前进的固定基座。

**修法：把「刚写入的一条」升级成「本轮写入的一整批」。** `cwick_pfx_lookup()` 在入口记下
`batch_mark = seq + 1`，`evict_to_fit()` 跳过 `seq >= batch_mark` 的条目 —— 即从上一次
查询到下一次查询之间写下的全部快照，那正是它们即将被读回的窗口。更早的条目照旧可淘汰
（命中过的仍按最后使用时间排），所以缓存不会无限增长；只剩受保护条目时允许整批超容，
而不是删掉其中一份。

单测 `prefix cache: a turn's boundary twin survives the turn's other two stores` 钉住它：
同一 turn 连写 P / twin / conv，下一轮用「边界之后分叉」的 prompt 查，必须命中 twin。
把保护关掉这条会失败（`next-turn hit=3` 是基座而不是 4，条目数 2），打开后通过。
`make test` 699/699 PASS，零告警。

**实测 A/B**（同一台机、同一个 8–9K token 增长会话、10 轮，用 `--prefix-cache-gib 2`
把淘汰压力提前到第 6 轮）：

| 策略 | `left to prefill` 第 3–10 轮 | 末轮 prefill |
|---|---|---|
| 旧 | 26, 26, 26, 36, **69, 102, 135, 168** | 3467 ms |
| 新 | 26, 26, 26, 36, **35, 35, 35, 35** | 1669 ms |

旧策略从第 7 轮起每轮多付「上一轮生成 + 新提问 + 2」个 token，节奏与线上 45K 会话一致
（81 → 160 → 239 → 318）；新策略把它压成常数。

**同一 turn 里另一份没人读的快照。** P 结束在被注入的思考块内部，下一轮渲染助手回合时
从不重发它，所以它在 50 个温请求里命中 0 次，却每轮占 1.63 GiB 并触发一次淘汰。twin 覆盖
它，代价只是重算最后两个 token。现在 `pfx_save_now()` 先算 twin，算出来就不再提交 P
（verbose 日志：`prompt snapshot at N tokens not stored: the boundary twin at N-2 covers
it`），每轮写盘从 3 份降到 2 份（10 轮实测 30 → 20 次 store）。

**顺带一个 4 倍的仪表错误。** `host_wired_bytes` 一直按 4096 字节/页折算，而 Apple Silicon
自 M1 起是 16 KiB 页，于是这个唯一用来证明「没有 swap」的 gauge 少了 4 倍：报 8.37 GB，
真值 33.4 GB。`src/mem.c` 改为读并缓存 `sysconf(_SC_PAGESIZE)`；修复后同一时刻读数为
33.59 GB。

未改动模型精度、权重、后端、上下文行为，也未动 §68 的两处。

## 70. 下载断线是一次重试，不是要用户去处理的错误

模型 tab 难的部分早就有：`./models` 下的本地清单、只读的 hf-mirror 目录检索、逐文件的
`.incomplete` 加 Range 续传、给本机真正会下的几百 GiB 检查点用的 64 位大小、以及限制在
库根目录内的递归删除。它缺的只是长下载离不开的那一个行为：连接断了，任务以前直接进
`error`，然后一直待在那里，直到有人点一下「断点续传」。

在这台机器上这不是边缘情况。通过镜像拉一个几百 GiB 的检查点，传输中断是常事。量它很
简单：把 `CWICK_HUB_ENDPOINT` 指向一个只发半个文件就关 socket 的本地目录服务。旧 worker
2 秒进 `error`，10 秒后还是 `error`，报着
`transfer closed with 3145728 bytes remaining to read` —— 诊断是对的，回应是没用的。盘上
已有的字节完好无损，只是需要一个人去要。

**改了什么。** 两个会失败的网络阶段现在共用一条重试路径：文件列表
（`/api/models/<repo>/tree/main`）和每个分片下载都在 worker 里按指数退避重试 —— 1 秒、
2 秒、4 秒……封顶 30 秒 —— 默认五次预算（`CWICK_HUB_RETRIES`，0 关闭）。每次重试都重新进
`download_file()`，它读现有 `.incomplete` 的大小，只用 Range 请求剩下的字节，所以一次
尝试绝不会重下已在盘上的内容。一个文件或列表成功就把计数清零：预算是按「卡住的那一步」
算，不是按整个任务算，一个 200 文件的检查点早期失败一次，不会在第 150 个文件上受罚。

等待按 100 ms 的小步可中断。`pause`、`cancel` 和服务退出在退避期间就生效，而不是等退避
走完 —— 退避封顶到 30 秒时这点很重要。重试期间任务保持 `running` 而不是 `error`：从外面
看，一次短暂断网就是慢，不是失败。只有预算耗尽才以 `error` 结束，并且消息保留最后一次
真实原因，附一个 `gave up after N attempts` 后缀 —— 绝不把诊断换成一句笼统的总结。

`GET /v1/hub/downloads` 现在多给 `attempts`、`max_attempts`、`retry_in_seconds`。模型 tab
在等待时显示 `自动重试 N/M · X s 后`，在重试成功后显示 `已重试 N 次`，于是本来像卡死的
停顿有了看得见的原因。小标题也用文字说了同一件事：默认 5 次、从 `.incomplete` 续传、不
重复下载。

**实测。** 对着那个可控目录服务跑，同一个二进制，全程 `--backend metal`（下载路径不碰
推理引擎）：

| 场景 | 改前 | 改后 |
|---|---|---|
| 中途断一次，随后干净重试 | 2 秒进 `error`，10 秒仍 `error` | 日志 `retry 1/5 in 1 s`，4 秒 `done`，5 MiB 完整 |
| 每次传输都断，`CWICK_HUB_RETRIES=2` | 立刻 `error` | `running`，`attempts` 1/2 → 2/2，然后 `error` 带 `gave up after 2 attempts` |
| 每次传输都断，`CWICK_HUB_RETRIES=8` | — | `running`，`attempts 5/8`，`retry_in_seconds` 倒数，UI badge 可见 |

最后一行在真实浏览器里核过：「正在下载」面板显示
`mock/retry 下载中 自动重试 5/8 · 1 s 后`，旁边是进度、当前文件和暂停/取消按钮。另一次
探针确认：在 16 秒退避期间发 `pause`，任务 1 秒内变 `paused`，不等定时器走完。

`make test` 为 702/702（新增的三条检查在不联网的情况下钉住默认预算、覆盖值与 0 关闭）。
默认保持 5，因为它够覆盖一次镜像抖动加一次 30 秒封顶内的重拨；链路稳定的运维可以设 0，
回到原来的快速失败行为。

## 71. 图像输出是两个问题，都不是文档里写的那个

图像 endpoint 从接通那天起就一直返回 HTTP 200 加一张不能用的图：256×256 全黑，
更大尺寸是带纹理的噪声。`docs/IMAGE.md` 当时列的嫌疑是条件编码器的 chat 模板与
多维旋转位置、DiT 的 token refiner 与三轴 RoPE、packed 序列的行顺序，并且把推理
路径挡在 `CWICK_IMAGE_UNVERIFIED=1` 后面，免得「管线能跑完」被当成「模型可用」。

这份清单是错的。实际是两件事，而第一件把第二件挡住了。

**这个 checkpoint 在约 4 MP 以下不工作。** 它公布的分辨率从 4096×4096 起。向参
考实现要 256×256，得到的是**同样的**带纹理噪声——这在结构上就排除了移植错误：那
个尺寸不在权重见过的范围内。`h3_canvas_size()` 现在会把请求按同宽高比放大到
`min_generation_area`（默认 4194304，`CWICK_H3_MIN_AREA` 可覆盖），两轴取整到 32
的倍数，解码后用 Lanczos-3 缩回请求尺寸。所以 512×512 请求的开销等于一次
2048×2048 生成，返回的是它缩小的结果。

**然后各阶段才能逐个对比。** 画布固定之后，`CWICK_H3_DUMP` 导出全部中间结果，
`CWICK_H3_LATENT` 把参考 latent 直接喂进解码器，于是偏差可以归到某个阶段而不是
整个栈。固定 seed、256² 画布，对参考实现的 rel_rms：

| 张量 | rel_rms | cos |
|---|---:|---:|
| 条件编码器 50 层 | 1.14e-2 | 0.999985 |
| 时间步嵌入 | 3.0e-6 | 1.000000 |
| token refiner | 1.26e-2 | 0.999921 |
| packed 序列 | 4.4e-3 | 0.999991 |
| DiT 第 0 / 第 49 层 | 6.4e-3 / 4.2e-2 | 0.999987 / 0.999125 |
| 最终速度场 | 2.9e-2 | 0.999591 |
| 解码后 latent | 5.4e-2 | 0.998514 |

这里没有一个是缺陷。条件编码器的 1.14e-2 是相对值最大的一个，也是实际误差最小
的一个：99.4% 的能量落在激活幅值最大的那 1% 通道上，但误差在 token 与通道两维
上均匀（与行号相关 −0.096，与幅值相关 −0.114）——这是浮点噪声的样子，不是模板
写错的样子。两次想靠舍入消掉它（残差流降到 bf16、条件编码器十三个算子逐个降到
bf16）都让它更差（1.14e-2 → 2.4e-2 与 1.9e-1），都已回滚。把参考条件编码器的
输出注入进来替代本移植的结果，解码 latent 只从 5.45e-2 动到 5.57e-2，说明剩下的
误差在 transformer 的累加顺序里，不在条件化里。

**DiT 还比它需要的慢了十倍。** 它的 attention 是手写循环，每算一个 score 都要
流式读一整行 key：实测 4.2 GFLOPS，占 512² 下单步 12.0 s 的 60%。改成每 256 行
一次 `cblas_sgemm` 做两次矩阵乘、vDSP 做 softmax（`exp` 之前先减行最大值——原始
score 超过 88 会让 fp32 溢出成 NaN）、把 `1/sum` 折进 128 宽的输出而不是归一化
整块 score，512² 的 DiT 从 344.7 s 降到 35.7 s，输出逐位不变。2048² 下同样的改
动是每步 gemm 95 s → 64 s、attention 129 s → 64 s；把 `H3_DEQ_TILE` 从默认的
128 行放宽到 512（激活栈至少 1024 行时才走）把 GEMM 推到 2.47 TFLOPS，接近裸
`cblas_sgemm` 在这些形状上的 2.7 TFLOPS。

门控已经删掉：`h3_generate()` 不再读 `CWICK_IMAGE_UNVERIFIED`，`start.sh` 也不再
设置它。唯一还会拒绝的是图像编辑，而那个拒绝针对的是一条确实不存在的路径（§73）。

## 72. 解码器分块既是正确答案，也是四倍速

每张图出来都带一层规则网格。解码器刚被改成整图一遍，所以第一嫌疑又是
transformer——但跳过 DiT、直接用**参考** latent 解码，网格一模一样，这就把问题
钉在最后 24 层里，别处都不用看。

原因写在发布的模块里：`MiniMaxH3ImageViTDecoder2d` 的 `use_tiling = True` 是发
布默认值——256 px 分块、最小 64 px 重叠、线性 cross-fade——而且它的 docstring 明
说公布出来的图像**就是**分块混合的那一版，关掉分块会改变输出。整张 128×128
latent 一次解码，等于把一个按 16×16 窗口发布的 ViT 推到那个窗口之外，它返回的
逐 cell 调制就是那层网格。16×16 latent 恰好只有一块，这正是此前 256² 的对齐工
作从来没看到它的原因。

`h3_vae_decode()` 现在复现发布的几何（`_split_tiles`：先数出块数直到重叠能覆盖
全长，再把余量按 cell 轮流摊到各边界上，保证每个边界都落在 cell 边界——2048² 下
是 11×11 块、80 px 重叠、最后两个边界 64 px）、`1 - i/e` / `i/e` 权重，以及
stitch 顺序：先与上方 raw 块垂直混合，再与左侧 raw 块水平混合，最后裁掉远端边缘。

由此掉出来三件事：

- **内存。** 第一个分块版本在第 108/121 块被 SIGKILL（退出码 137）掉，峰值
  112 GiB。原因是每一块都通过 `h3_standalone_f32` 把 391 个解码器张量从映射里
  拷一遍——每块 1.6 GiB fp32，121 遍。这些张量是只读的：加载器现在直接返回映射
  里的指针，每张图只解析一次全部 391 个张量，header 也从「每块 24 次」变成一次。
  峰值 RSS 112 GiB → 48 GiB。
- **速度。** 块内 attention 是 16×16 而不是 128×128。2048² 下 VAE 从 114028 ms
  降到 26469 ms。
- **保真度。** `c_rgb` 的 rel_rms 从 8.2e-2 降到 2.8e-3；按 8 位截断后 99.96% 的
  像素与参考完全一致，79.2 dB PSNR。

有两处细节各花了一轮。第一处是 8 位截断必须放在最后，不能放在每块里：先把块舍
入再混合，会让每条重叠带差出最多两个色阶。第二处是混合本身。权重属于**带内偏
移**，所以水平方向必须把 `i` 落到像素上；若每个偏移都把整条 `e` 宽的带重新混一
遍，最后只有最后一个偏移的权重留在整条带上，而递推
`cur <- a*(1 - i/e) + cur*(i/e)` 在 `cur == a` 之后是幂等的，于是那条带静默变成
了「左边一块直接覆盖」。线索是垂直带全对、水平带全错——垂直分支每个偏移写一行，
水平分支每次都写同一条带。把 raw 块按参考的 `_stitch_tiles` 在 numpy 里重新拼一
遍，得到与参考图在带内平均绝对误差 0.00001 的结果，说明块本身从来没问题；这个
dump 保留为 `CWICK_H3_DUMP` 下的 `c_rawrow%02d`，下次还能用。

## 73. `images/edits` 没有拒绝，而是返回了一张新图

编辑端点本该返回 501，因为参考图路径（参考图的 VAE 编码、`<Picture N>` 标签、
vision block）还没绑定。它返回了 200 和一张生成图。

拒绝与否取决于 `n_ref`，而 `n_ref` 是 multipart 字段提取器数的。那个函数边走各
个 part 边返回——一找到它要的字段就 return，于是排在 `prompt` **之后**的 part
从未被看到，而 `curl -F prompt=… -F image=@…` 正好把 image 放在第二位。于是一个
编辑请求变成了彻底忽略参考图的文生图，而这正是 501 存在的意义所在，而且它做得
毫无声息。

提取器现在走完每一个 part，数 `image` part，并把第一个匹配的命名字段记下来而不
是在循环里直接返回。两种字段顺序都在约 1 ms 内返回 501：
`image editing is not implemented: reference images are not conditioned on yet`。

## 74. 编辑路径是一张图、两套尺寸规则、两个 tag 向量

把 `images/edits` 接上，等于接上四段只在有参考图时才存在的流程：VAE 编码器、vision tower、编辑型条件编码器，以及把参考行放在生成行前面的 DiT 布局。每一段都有一处「看起来是细节、其实不是」的地方。

**尺寸。** 同一张参考图喂给两个消费者，而两者的规则不一致。VAE 副本要短边落在 `[1024, 2048]`；vision 副本固定 `1024`；两者互不派生，都从原图各自缩一次，且都不会缩到生成画布的尺寸。短边本就在区间内的参考图会保留原尺寸——而 packed 布局要求两边都能被 32 整除两次，手机照片几乎必然踩到，所以本实现把每边按最近的 32 取整（代价最多 16 像素），而不是抛错。

**两个 tag 向量，而且不是同一个向量。** 条件编码器自己的 `mm_token_type_ids` 只在 `<|image_pad|>` 上为 1，它决定 3 轴 mrope 的分段：`<|vision_start|>` 和 `<|vision_end|>` 在那里算文本，所以标签与 start 构成同一段 text run。transformer 的 `token_tags` 覆盖**整个** vision block（含两个分隔符），它决定 AdaLN 的模态行。把第二个向量单独接进去之前，DiT 第 0 步速度场差了 6.46e-1；text 行是模态 1、image 行是 0，而参考图的 vision block 虽然活在 text 流里，tag 却是 image。这一点布局上没有任何提示：两个向量除每张参考的四个 `<|vision_start|>`/`<|vision_end|>` 位置外处处相同，于是错误看起来像 rope 表坏了。

**时间步。** 参考行被加噪到 0.999 并保持不变，生成行跟着采样器走。对 `[0.999, t]` 做 `unique(sorted=True)` 才是把它们变成两行 AdaLN 的原因，而每行拿到哪个下标取决于 `t` 在 0.999 之上还是之下——反了就会静默地把生成行按参考水平条件化。调度器只写 `latents[num_reference_rows:]`，输出头把全部 image 行算完再丢掉参考行。

**deepstack 加在 `all_hidden_states` 之前。** `hs[k]` 是第 k 层输出但**没有**加 `ds[k-1]`，因为相加发生在列表追加之后。拿它去比得 4.9e-1 不是 tower 的错。

对齐（1024² 参考 + 1036 行文本，全部对同一 seed 的参考实现）：

| 张量 | rel_rms |
|---|---:|
| VAE 编码器 moments | 1.75e-6 |
| 参考 latent | 2.71e-5，98.4% 逐位相同 |
| patch embed / `x0` / block 0 | 3.8e-7 / 3.5e-7 / 2.6e-6 |
| vision 末层 / merger / 三个 deepstack | 8.5e-5 / 2.8e-5 / 7e-6…2.6e-5 |
| ids / types / tags（1036 行） | 零 mismatch |
| 3 轴 mrope 位置 | 零 mismatch |
| `hidden_states[1…3]` | 2.6e-5 / 2.1e-5 / 1.8e-5 |
| 条件编码器 `hs50` | 4.5e-5 |
| 参考行 packed + 加噪 | 4.4e-8 |
| 第 0 步速度场（1088 图像行） | 6.5e-3 |

最后一行与文生图同级，说明布局是对的，剩下的残差在 transformer 的累加顺序里。

## 75. 一层之外的 null 写入，在请求第 117 秒变成 `brk #1`

第一次真实的 `images/edits` 调用直接打死服务：exit 133、SIGTRAP，而 `serve.log` 里**根本没有**这次请求的记录——上一行还是坏图那条 400。崩溃报告把 PC 指在 `h3_vae_encode` 最后一条指令 `brk #0x1` 上，而那个函数里没有这种东西。

`h3_vae_encode` 通过 `int64_t *o_lw, *o_lh` 回写 latent 尺寸，而两个调用点都传 `NULL`——调用者自己算得出尺寸。两次穿过空指针的死写入。两个调用点都内联之后，优化器把 `null` 常量传播进函数，判定这两次写入是 UB，于是把整个成功块——三次写入加 `return true`——换成了 `unreachable`，后端把它输出成 `brk #1`。所以这个陷阱出现在函数的**成功路径末尾**，在 117 秒正确计算之后，周围没有任何分配或越界的迹象。

这件事值得单开一节，原因有两条：

- 判别证据便宜且决定性：把单个源文件编成 IR，数一下 `store …, ptr null` 和 `unreachable`，这一类问题在运行之前就能找到。现在整个模块两项都是零。
- CLI 测试没覆盖到它，因为跑同一段代码的那次 CLI 用了 `CWICK_H3_REF_LATENT` 注入、跳过了编码器。任何能被调试开关替掉的阶段，都必须在不带开关的情况下真跑一次。

两个参数已删除，被调函数只写 `*out_lat`。

## 76. 把图像 transformer 的一步交给 GPU 是两个问题，而且都不是算术

4096 图像行的一步要读 66.3 GB 权重、发 1346.6 GMAC。这个字节数在 5 行和
4096 行**完全相同** —— 它不随序列变短 —— 而在 CPU 上每个字节要走三趟内存：
投影内核先把每个权重块反量化进 f32 arena，再对副本调 `cblas_sgemm`。实测是
12.6 GB/s 的**权重**吞吐，对应 1.24 TFLOPS 的稠密算术。

同样形状上，独立的 BF16 simdgroup 内核实测 3.8–5.4 TFLOPS，所以要做的是把
checkpoint 自己的字节喂给它。两个内核：

- **投影**：每个 threadgroup 一块 64×64 输出，128 线程，四个 simdgroup 按 2×2
  各持 32×32，由 8×8 fragment 搭成；K 以 16 宽分块、经 10 KB threadgroup 内存
  暂存。权重块按 checkpoint 自己的 `[n][k]` 顺序暂存并用转置标志装载，所以
  61 GiB 权重一个字节都不重排。
- **注意力**：每个（64 查询行, 头）一个 threadgroup，键区间按 16 键一块走，
  除当前块外不生成任何分数矩阵。这一项是二次增长的：4096 行时占 136.6 s 单步
  里的 64.5 s；同样形状若配上 2048² 画布所需的参考图，就是 933 s/步。

有四件事值得写下来。

**非零 buffer 偏移贵 1.75 倍。** 同一个内核、同一块 tile：把张量按页对齐后的
前导量当成 `setBuffer:offset:` 传进去，4103×5376×14336 的投影实测 291.7 ms；
把 buffer 基准传 0、把字节偏移改成内核参数，实测 167.0 ms。而且 no-copy buffer
的偏移必须是整页，safetensors 的载荷只有 8 字节对齐，所以参数形式才是能推广的
那个。

**一个 shard 一份视图，不是一个投影一份。** 一步要跑几百个投影，全都落在同一
61 GiB 上。按投影建 no-copy 视图会把它那一块的页全部 fault 进来、并且把句柄留
到进程结束，于是步时间随已跑步数增长 —— 第 2–4 步是 62、68、72 s。现在在模型
打开时一次性 adopt 所有 shard 映射（61.7 GiB 走一遍页表 3.1 s），张量偏移放在
参数块里。

**指数必须减掉行最大值。** 第一版直接算 `exp(score)`，理由是「每个头在 rope
之前都做过 RMSNorm」。这句话是对的，但没用：残差流在第一个块内就到了四位数，
分数到 10⁴ 量级就让 fp32 的 exp 溢出，第一个块之后全是 NaN —— 而 `c_block0`
当时仍然对到 1.2e-3，这正是它看起来像「累积误差」而不是「量程问题」的原因。
随机输入的探针永远测不到它，因为随机输入从来产生不了大分数；现在探针带一个
输入倍率，在 100× 和 4700× 上都能复现。修法是先走一遍键区间取行最大值、第二
遍对着它取指数：单趟流式需要「运行中最大值」，而应用它的修正意味着缩放输出累加
器 —— 这恰好是一组矩阵 fragment 做不到逐元素的事。代价是 74 s 单步里的 2.3 s。

**threadgroup 的跨界行是被错误的线程写出去的。** 最后一个 64 行块的边界上有两个
独立缺陷，在长度是 64 的倍数时都看不见：暂存转出的步长用了 128 而不是这个
threadgroup 的 256 线程；逐行归一化与它要缩放的写之间没有屏障。逐行探针一次就
抓到了 —— `row 64…68 rel 0.86, row 68 rel 1.00` —— 修法是让拥有这些行的
simdgroup 自己把行搬出去、并在同一趟里完成缩放。现在对 double 参考的 `rel_rms`
在 69 到 1024 的每个长度上都是 2.1–2.6e-3，且与长度无关 —— 这才是「舍入」和
「几何错」的区别。

4096 图像行上的净效果：单步 136.6 → 74.3 s，DiT 874 → 446 s，整个请求
949 → 536 s。注意力 64.5 → 9.0 s；投影只从 63.7 到 54.3 s，因为 CPU 路径本来
就有 1.24 TFLOPS 而内核到 1.51。解码像素与 f32 激活的 CPU 路径差 8.6e-2 RMS ——
来自激活的 BF16 暂存，经 50 块 × 6 步累积，场景与构图一致。`CWICK_H3_GPU=0`
可以让这一切都不发生。

## 77. wire 住 transformer 去掉的是页回收税，不是算术

一个采样步要把同一份 66.3 GB transformer 权重读六遍。checkpoint 是两棵各
62 GiB 的目录（文本编码器与 transformer），128 GiB 机器上挂着 124 GiB 映射，
总要有东西让位：系统在步与步之间回收权重页，内核在投影循环中途把它们重新换进来。
`CWICK_H3_WIRE=1` 只对 transformer 调 `cwick_safetensors_make_resident(..., wire=1)`。
文本编码器刻意保持可回收 —— 它每个请求只读一次，而把两棵都 wire 需要的物理内存
超过机器实际拥有的。

A/B：同机、同一个 1536×2048 生成、seed 7、`--backend metal`、6 步，先 A 后 B。

| | 不驻留 | 驻留 |
|---|---:|---:|
| 模型打开 | 3.5 s | 26.9 s（61.74 GiB，2.5 GB/s） |
| 投影，6 步均值 | 52.57 s | 44.93 s |
| 单步，6 步范围 | 62.7 … 79.9 s | 60.2 … 60.7 s |
| 单步，标准差 | 6.29 s | 0.17 s |
| DiT | 419.7 s | 362.2 s |
| 整个请求 | 506.8 s | 449.6 s |

均值收益是 DiT 13.7%；标准差塌掉 37 倍才是真正的诊断。不驻留时，同一批字节的投影
耗时在 46.0 到 61.1 s 之间游走，因为页缓存在步与步之间被换出，所以从来就不该拿均值
做容量规划。wire 之后六步全落在 44.6–45.1 s。wire 的代价每个进程只付一次：一次性
CLI 在打开时多花 23.5 s，仍然提前 33 s 结束；常驻服务只付一次。

有两样东西没变：PNG 逐字节相同（SHA-256 一致），文本编码与 VAE 耗时在噪声范围内。
61.74 GiB 并没有变得更好读 —— 它只是不再被回收。这也解释了此前两次三参考图请求
只差 2.2% 为什么具有误导性：那次比较把缓存状态保持得差不多，于是测的是一个页恰好
都热的工作负载。真正的证据是单次不驻留运行内部的方差，它就摆在那里。

## 78. 一个图像目录、两个入口、不耦合文本模型

`--img DIR` 成为配置图像 checkpoint 的正式方式；`CWICK_IMAGE_MODEL` 保留给不方便传参数的启动器，两者都给出时后者让位。目录是装载期的选择，启动时复制进 server，因此 HTTP 客户端只能请求 `MiniMax-H3-Image`，不能指定任意路径 —— 切换接口的路径处理规则不需要为此改动。

图像管线挂在 `server_ctx.h3`，与常驻文本引擎并列，而不是占用它的生成槽位。于是 `GET /v1/models` 多出一条带 `cwick.kind:"image"` 的条目，`GET /v1/models/MiniMax-H3-Image` 可取回它，`GET /v1/config` 带 `image_model.dir` / `loaded`。切换到 `MiniMax-H3-Image` 会从配置目录打开管线，返回 `switched` 或 `already_resident`；没配置目录返回 400，**文本**模型切换进行中返回 409，打开 checkpoint 失败返回 500。这条路径不释放文本模型，所以图像接口可以配置、公布、按需装载，而不用让生成下线。内置设置页读的是同一份配置字段，提交的是同一个切换请求。

控制台的语言是保存下来的偏好，不做页面刷新：`data-i18n` / `data-i18n-ph` / `data-i18n-title` 覆盖静态节点，少量 JS 字典覆盖动态字符串；`setLang()` 重绘文案、用新语言的空状态重新渲染消息、重放内容淡入，并把焦点移到视图标题。卡片入场淡入上浮、背景渐变流动、当前语言按钮光晕、悬停/聚焦过渡都是纯 CSS；`prefers-reduced-motion` 下全部关闭。验证是静态的 —— 键集合检查、内嵌脚本语法检查、重新生成资源的 HTTP 抓取 —— 加上 `make cwick` 与 `make test`。

## 79. 模型库必须认识 pipeline 目录

**模型**tab 把 `MiniMax-H3-Image` 显示成「未下载完」并且不能切换。清单里「已下载」只有一条定义：checkpoint 根目录有 `config.json` 且没有 `.incomplete`。这对文本 checkpoint 成立，对图像管线不成立 —— 它的根目录是 `model_index.json`、融合头分片和四个子模块目录。UI 如实反映了布局不匹配，但状态本身是错的。

完整性现在按格式判定：文本根仍要 `config.json`；图像根要有一个声明图像管线的清单，外加 transformer、VAE、文本编码器、tokenizer 和融合头分片，两者都不能有部分文件。清单新增 `kind`，图像条目还带 `served_id`，所以按钮提交的是 `MiniMax-H3-Image` 而不是仓库路径。`cwick_hub_local_path()` 用同一道完整性门；切换处理器在没有 `--img` 时回退到模型库，于是一个已下载的管线不必再用第二种方式描述目录就能热装载。测试构造了一个没有 `config.json` 的合成管线，断言它被列为完整且可切换，同时带 `.incomplete` 的文本 checkpoint 仍是可续传。

## 80. 单模型驻留与 H5 图像工作台

过去两条管线并存：`main` 先建文本引擎再进入 `cwick_server_run`，`--img` 又把图像 checkpoint 映射到它上面。在这台机器上，这意味着约 63 GiB 图像权重与文本权重同时驻留，正是 128 GiB 主机承受不起的状态。现在同一时刻只运行一个模型。

启动时 `--img` 加载成功后，所有读取文本引擎的初始化步骤（指标、tokenizer、catalog）都已复制到自己需要的数据，此时立即释放文本引擎，直到监听端口打开。切换到 `MiniMax-H3-Image` 时先释放文本引擎（所有槽位、tokenizer、Metal arena、前缀快照、live-engine 指针），再打开图像 checkpoint；切换到文本 id 时先释放图像管线，在解除权重映射之前先丢掉 GPU 的 no-copy 视图。两个方向都在实机上量过：带 `--img` 启动后只有图像条目为 `loaded`，切回文本后图像条目变为 `loaded:false`，再切回图像耗时 2.8 s。catalog 保留启动时的文本根目录作为切回目标。

一次渲染约 40 分钟，期间释放它的模型会造成 use-after-free。`gen_lock` 下的 `img_busy` 计数加入切换门：文本生成或图像渲染进行中时，切换返回 409；切换期间图像请求返回 503。两条管线的 `loaded` 状态都取自实际驻留的那一个对象，因此模型列表不可能声称同时加载了两个模型。

第三个托管应用 `apps/imagegen` 是面向两个图像接口的独立 H5 工作台。它接收提示词与可选的头像/服装/姿势上传；编辑模式提交 multipart（每个上传一个 `image` part）调用 `/v1/images/edits`，纯文生图模式提交 JSON 调用 `/v1/images/generations`。进度显示已用时间与响应中 `cwick` 对象的分阶段耗时（该接口没有进度流），结果内联展示并可下载为 PNG。记录镜像到每个应用独立的 sqlite，localStorage 作为回退；分享按钮复用其他应用相同的局域网二维码接口。测试套件新增了内嵌路由断言，并跑过 `make cwick` 与 `make test`。


## 81. 图像生成把每个阶段写进 stdout，而不只是总计

一次渲染耗时数分钟，但图像接口只在 PNG 编码完成后打印一行。在长达 40 分钟的请求期间 stdout 完全沉默，无法区分渲染在推进还是卡死；要判断时间花在哪个阶段，只能事后解析 JSON 响应。

阶段边界本来就作为 `h3_timing` 字段存在于 `h3_generate()`，因此修复方式是在测量处直接打印，而不是另造一套进度机制。现在 `srv_image_route()` 用 `CWICK_INFO("image", …)` 打印到达行（模式、请求尺寸、seed、参考图数），`h3_generate()` 内每个阶段打印一行：tokenize、text-encode、pack、dit、vae、png。DiT 是最长的一步（六次采样），因此每步单独一行，带本步耗时与累计经过时间，并附阶段合计。服务端的完成行也从只在括号内给出 `(dit, vae)` 扩展为六个阶段全部列出。没有新增开关：共享 logger 已经盖上系统时间和线程局部的 `[req N]`，而前台 `serve` 默认就打印 `info` 级别。所有行都走同一个 logger，因此同样进入 `GET /v1/logs` 的日志环。

实机 `--backend metal` 运行确认了两个接口的格式。一次完整生成（256x256，调低调试面积下限）在同一个 `[req 1]` 下打印了六个阶段和六行 DiT 步骤，总计 117284 ms，其中 dit 50673 ms；一个带单张参考图的 `images/edits` 请求在 `[req 2]` 下打印了 `edit request` 到达行和同样的阶段序列。编辑请求的客户端在渲染中途撞上测试工具超时，而服务端继续渲染与打印 —— 整个过程的 stdout 序列保持可读。`make cwick` 与 `make test` 均通过。


## 82. 「已完成却没声音」是 assistant 模板 bug，不是模型沉默

TeachMe 能看到转写和 `response.done`，但始终不出声。stdout 的整轮摘要把问题
钉死了：`turn completed … decode 0 tokens … talker 0.000 (0.00 s audio, 0 chunks)`。
WebSocket 层是好的：转写增量与收尾事件都发了，浏览器只是根本没有音频
缓冲区可播。

当时喂给解码器的是
`<|im_end|>\n<|im_start|>assistant\n\n\n\n`。那四个换行落在 assistant 回合
**内部**，模型于是把回答读成「已经开口且说完」：首个采样 token 就是结束符，
没有任何句子送到 talker，状态机如实报告 completed。

定位做了三步 A/B。同样权重的纯文本回合仍能对答，先排除解码器和内核；用固定
WAV 走离线监听路径复现 0 token，EOS 以 20.30 排在第一（最好的内容 token 为
19.58）；把音频向量换成占位符自带的 embedding 仍然如此，排除音频前端。把
prompt 直接终止在裸的 assistant 起始符后，EOS 掉到 4.67，回答正常。
`fwd_user_block()` 与 `turn_positions()` 用同一个尾部，所以两处一起改 —— 后者
不跟着改，KV 预算就会和实际转发的位置数错位。

修复后 `--backend metal` 实机 WS 运行对 40 个回复 token 返回 53 个
`response.audio.delta`（10.44 s，24 kHz PCM16）。回复 WAV 不是静音：RMS 0.087、
峰值 0.84。两轮运行每轮各 21 个 delta，并确认热轮 prefill 由 8.16 s 降至
2.83 s。`make test` 通过。完整拆分与两个易混淆的故障模式写进
`docs/REALTIME.zh-CN.md` 及英文版。


## 83. 断句需要的是迟滞和噪声底，不是再调低阈值

realtime 的输入 VAD 原来只有一个 RMS 阈值：超过就起音，低于就走固定静音计时
结束回合。这个形状直接带来两个现场问题。40 ms 的咔哒声和一个词没有区别，会
被提交成完整回合，可能取消正在进行的回答；连续语流中两个响音节之间那个轻读
音节会让静音计时走满，把一句话切成两句，模型于是只回答半句。

调研一个全双工语音栈时找到的不是又一个神经网络，而是两个小机制：释放电平
低于起音电平，以及声音要先持续一小段时间才算语音。两者都落进
`src/rt_vad.c`。在客户端 0..1 阈值刻度上，起音与释放相差 0.15（0.5 时 −40 对
−44.5 dBFS），中间是保持区，不推进结束计时；一段声音必须连续高于起音 100 ms
才上报，被拒绝的片段从提交前缓冲中丢弃，避免连续咔哒声无限累积。

改之前先把决策抽成纯模块，因此同一段代码能在同一信号上跑两遍。
`tools/rtvad.c`（`make rtvad`）显示咔哒声从 1 起点/1 终点变成 1 次丢弃，词内
回落从 2 句变成 1 句，400 ms 真停顿在两种模式下都仍会断句。原规则保留在
`CWICK_RT_VAD_LEGACY=1` 之后，`make test` 里五条断言固定新行为。实机
`--backend metal` 手动提交一轮仍返回 51 个音频 delta / 10.08 s 非静音语音
（RMS 0.095、峰值 0.82）。

`tools/rtws --vad` 会播完成品录音再补静音。停顿超过 `silence_duration_ms` 的
音频会被切几次，其余部分被当成插话；用 legacy 标志跑得到同样的三次提交和
取消，所以这是 harness 的性质，不是回归。测成品音频请用 `--burst`。

## 84. Qwen3.5 checkpoint 用发布名应答，不用类型标签

这个混合架构 checkpoint 的 `config.json` 里有两个内部名字：外层
`model_type=qwen3_5`、内层 `text_config.model_type=qwen3_5_text`。它们都是内核
选择器——loader 先解包 `text_config`，`engine.c` / `server.c` 按 `qwen3_5` 前缀
分支——所以两个都不能改。但 `cwick_model_served_alias()` 是从目录名派生对外
API 名的，而 Qwen3.5 目录是唯一没进那张表的发布 checkpoint，于是对外名回退到
了内层类型标签：客户端在 `GET /v1/models`、必须填的 `model` 字段、`/v1/metrics`
（`mi.model`）和启动日志里看到的都是 `qwen3_5_text`，而 `start.sh` 早就接受真实
名作启动参数。这种「按发布名启动、按类型标签应答」的错位就是问题。

修法：在表里加一条，把发布目录 `DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic`
映射成同名的对外 alias。`served_name()`（catalog）、metrics 发布（从目录派生
alias）、`validate_body()`（经 `cur->alias`）现在都报告并接受真实名；`qwen3_5_text`
和 `qwen3_5` 仍作为 `model_id` / catalog id 被接受，学过旧名的客户端不会坏。
alias 与这两个 id 都不同，所以 `cwick_models_list_add()` 会保留它，而不是当成
「没有更好的名字」清空。`make test` 里两条断言固定了目录查表与 catalog entry
两个环节，全套 749 条通过。

## 85. 下载完整与可加载是两件事

模型列表 tab 会把一个已经下完的 checkpoint 显示成「未下载完」（`hub.partial`）——只要它的目录里没有 `config.json`，又不是图像 pipeline。`checkpoint_ready()` 把这两个标记当成「已下载」的唯一定义，所以一个只带 `.pt` / `.ckpt` 权重的载荷（`models/GiantAILab/YingMusic-SVC` 下的 SVC pipeline，四个文件共 2.03 GB）即使一个 `.incomplete` 都没有，也被报成未完成。`hub.c` 的下载规则是：`<target>.incomplete` 是未传完的**唯一**标记，文件大小与清单一致后才会 rename 到位。因此该目录里每个普通文件都已经下完，错的只是「这是不是文本模型」这条启发式规则。

修法：`checkpoint_ready()` 在没有 `.incomplete` 的前提下，也接受包含已知权重载荷（`.safetensors`、`.bin`、`.pt`、`.ckpt`、`.pth`、`.gguf`、`.onnx`、`.npz`）的目录。原因文案从「config.json is not present yet」改为「weights-only checkpoint (no config.json); not a cwick-loadable architecture」。「已下载」和「可加载」仍是两件事：卡片显示「完整」+ 架构不匹配徽章，依然没有切换按钮（该 checkpoint 没有 cwick 内核），和 `Ace-Step1.5` 靠 `config.json` 走的路径一致。

实测：在线服务上 `GET /v1/hub/models` 中这一条从 `complete=false, supported=false` 变成 `complete=true, supported=false`，其余七个模型逐字段不变。`make test` 保持 749 条全绿。

## 86. 手动下载的 pipeline 在模型库扫描中不可见

`models/MiniMaxAI/MiniMax-H3/` 是一份完整下载的 498 GB diffusers pipeline（485 个文件），但模型列表 tab 从来不显示它。文件完好、一个 `.incomplete` 都没有，所以这不是 §85 那种下载状态问题——这个目录根本没进清单。

扫描入口 `looks_like_checkpoint()` 只认三种标记：`config.json`、严格的 H3 **图像** manifest（`_class_name` 为 `MiniMaxH3ImagePipeline` 或 `partition:"image"`），以及 cwick 自带下载器写的 `.cwick-repo`。手动下载的 checkpoint 没有最后这个标记，而这份的 manifest 写的是 `MiniMaxH3ModularPipeline`，于是 owner 目录和它的子目录都被跳过。

修法：`looks_like_checkpoint()` 现在也接受任意顶层 `model_index.json`，不限制 class、不问文件怎么来的。严格图像 manifest 仍用于 `kind=image` 和 `supported` 判定，所以这个 modular pipeline 显示为「完整但不可加载」（`complete=true, supported=false`），而不是被当成可切换的图像模型。下载到一半的 pipeline 有 index 但没权重载荷，仍会作为未完成显示。

离线验证（未启动服务）：探针调用 `cwick_hub_models("./models")` 返回 9 条，MiniMax-H3 在列、498 GB / 485 文件；`make test` 755 条全绿（新增 6 条断言）。

## 87. H5 工作室把图像到视频的闭环接通

应用 3（`apps/imagegen`）是两步 H5 工作室。第一步调用两个图像接口；第二步把第一步生成的 PNG（或上传的图片）连同视频提示词交给异步的 `/v1/videos`，轮询 `GET /v1/videos/{id}` 取真实的 `progress` 字段，再用 `GET /v1/videos/{id}/content` 在线播放与下载。进度由服务端驱动而非本地计时器：进度条跟随任务各阶段权重（文本 0-5%、打包 5-10%、DiT 10-80%、视频 VAE 80-92%、音频 VAE 92-97%、合成 97-100%）。

采样器是文本条件的——视觉 VAE 只有 decoder，没有把参考图编码成初始 latent 的 encoder。所以没有假装做了图生视频：应用把图像作为可选 `image` 字段（data URL 或裸 base64）提交，服务端校验（PNG/JPEG 魔数）、随任务保存字节，并在详情里回显 `reference_image`，其中 `conditioning:"text_only"` 如实标注。源图先在浏览器里缩成 1024px 的 JPEG：完整的 1536×2048 PNG data URL 可能超过服务端 8 MiB 请求体上限，而参考图并不需要原始分辨率。

服务端回收视频槽位时，在 `memset` 前释放上一条任务的 `prompt`、`mp4` 和 `ref_image`，顺带修掉了任务复用时 mp4 的既有泄漏；`image` 的各条失败路径也释放了解码字节。`make test` 保持全绿；embed 重建报告更新后的 22 个 blob。

## 88. 真正的图生视频需要那个我们说并不存在的 encoder

第 87 节说视觉 VAE 只有 decoder，这是错的。随模型分发的
`FL2VA/video_vae/source/model.safetensors` 里有全部 116 个 `encoder.*` 张量，
以及 `quant_conv` 和 `post_quant_conv`：一个因果 3D CNN，ch=128、
`ch_mult=[1,2,2,4,4,8]`、每层两个残差块、空间步长 `[2,2,2,2,1,1]`。缺的是
实现，不是权重。

`h3v_vae.c` 里的 `h3v_vae_encode()` 现在把一帧 RGB24 变成 24 通道 latent
均值。关键是时间维退化：单帧时因果填充在前面补 `KT-1` 个零，只有最后一个
时间核切片有贡献，于是每个 3D 卷积都化成核为 `w[:,:,KT-1,:,:]` 的 2D 卷积。
卷积走 im2col + sgemm；残差块与 conv 用 reflect 填充，带步长的下采样用只在
远端补零的 (0,1,0,1)。GroupNorm(32, eps 1e-6) 逐帧计算，后接 SiLU。
`norm_out -> conv_out(48) -> quant_conv` 得到 moments，前 24 通道即 latent
（条件帧取 eps=0）。

验证全程纯 C，不信任旧的 numpy 重放——它的卷积辅助函数填错了轴，只读到零帧，
结果只剩 bias。用正确的 shard 偏移读出 bias 后，手算 `conv_in` 的一个内部像素
与 C 核吻合到 1.3e-7。端到端上，把结构化的 64×64 图案编码后再直接送进已验证
的 ViT decoder 解码，重建图 R 通道相关度 0.9186——输出若是乱码不可能重建出
输入。

条件注入复用已有的 video token 通道（DiT 没有新的输入口）。参考图缩放到渲染
画布后编码，写进 latent 第 0 步；每步 Euler 更新后把这段 packed 行重置回去，
使首帧保持干净而 DiT 每步都在注意它。有参考图时禁用缓存捷径。HTTP 详情里的
`reference_image.conditioning` 在编码成功时报告 `first_frame`，只有解码失败
才回退 `text_only`。`h3vprobe` 与 `h3vencprobe` 都是 `tools/` 下的纯 C 工具，
各有 Makefile 目标。

## 89. 封装器根本没写 moov，esds 又把描述符套进了描述符

第一段端到端 i2v 视频完全打不开：ffprobe 报 `moov atom not found` 和
`wrong sample count`。用纯 C 的 box 遍历器看输出才明白：`h3v_mp4_encode()`
把 `moov` 构造成一个装着子 box 的缓冲区，最后却用 `mb_put(&out, moov.p, ...)`
直接追加内容——atom 自己的头从来没写过，于是 `mvhd` 和两个 `trak` 裸放在顶层。
另一个问题是 `stsc` 声明「一个 chunk 装下所有样本」，`stco` 却按每样本一个偏移
写了 102 条；ISO 14496-12 要求两者计数一致。

`h3v_mp4.m` 四处修复：改用 `mb_box(&out, "moov", &moov)`；stsc 的
`samples_per_chunk` 改为 1；`tkhd` 的时长改用电影时间刻度（原来用媒体时间刻度，
长了 24 倍）；修正 `esds` 两个描述符长度字段。

box 修好后音频仍然失败：`Audio object type 0`、`22050 Hz, 0 channels`。
AudioToolbox 的 magic cookie 是一个完整的 ES 描述符（7 位变长长度），不是
`DecSpecificInfo` 里该放的裸 AudioSpecificConfig；我们把前者塞进了后者。
`find_asc()` 现在沿 ES → DecoderConfig → DecSpecificInfo 找到两字节配置
（32 kHz 立体声 AAC-LC 是 `12 90`）。之后 ffprobe 正确报 `AAC (LC), 32000 Hz,
stereo`，两条流都能解码到预期样本数（AAC 输出按 1024 样本包边界补齐）。

回归现在很便宜：`make h3vmuxprobe` 把合成像素和 PCM 直接送进封装器；测试套件里的
`test_h3v_mp4` 在内存中检查 box 覆盖、stsc/stco 一致性和裸 AAC-LC 配置。两者都不
需要权重，也不需要四分钟的 DiT。

## 90. 视频计时只在任务详情里，/v1/metrics 里没有

轮询 `GET /v1/videos/{id}` 的客户端在完成后能看到完整的 `cwick` 计时对象，但
Prometheus 抓取里一个视频指标都没有——和那个卡在 10% 却看起来健康的任务是同一
种盲区。`metrics.{c,h}` 现在发布 `cwick:video_jobs_total`、`_succeeded_total`、
`_failed_total`、`_first_frame_jobs_total`、累计的
`cwick:video_stage_seconds_total{stage=...}`（text、pack、encode、dit、vae、
audio_vae、mux、load、total）、实时的 `cwick:video_dit_steps_done` / `_total`、
`cwick:video_last_total_seconds`、帧数/字节数计数器以及
`cwick:video_tokens_total{kind=...}`。DiT gauge 直接由已有的进度回调写入，所以
长采样期间抓取能区分「在推进」和「卡死」。`h3v_timing` 新增 `encode_ms` 记录参考图
VAE 编码；任务详情和计数器用同一个单调时钟测量。

## 91. 在可播放的视频上验证了首帧条件注入

第 88 节的 encoder 数字还只是张量层面。之后用修好的封装器（第 89 节）以
256x256 / 1 步跑通完整 i2v，产出 571,855 字节的 mp4；ffprobe 识别为 H.264 High
24 fps 加 AAC-LC 32 kHz 立体声，时长 4.32 秒。把第 0 帧和第 40 帧解码出来，与
放大后的 64x64 参考图用纯 C 比较亮度，相关度分别是 **0.567** 和 **−0.036**。
参考图确实出现在第一帧，之后消失——正是 latent 第 0 步注入应有的行为。完整阶段
耗时见 PERF.zh-CN.md。

## 92. 视频 DiT 与 VAE 补上 Metal 路径，输出仍与 CPU 高度一致

第 89–91 节修复了正确性和可观测性，但视频 DiT 与 VAE 仍走 CPU fp32 SGEMM；
`--backend metal` 虽然初始化了设备，却没有视频内核用它。现在新增
`h3v_gpu.{h,c,m}`：50 层 DiT 用 BF16 张量核 GEMM，VAE 的 F32 权重用 F32 张量核
GEMM，分块 flash attention 分别实例化为 128 通道（DiT）和 64 通道（VAE 解码器）。
它们沿用图像变换器的零拷贝页采用方式，62 GiB 分片只包一次，而不是每个投影包一次。

两个内核细节很关键。Metal 不允许在非 kernel 函数里声明 threadgroup 数组，所以由
kernel 声明暂存内存再传给公共核心。注意力的归一化要把一个 simdgroup 的 8xHD 输出
按每块 HD/4 映射到 32 个 lane：照搬 128 宽的布局会让 64 宽实例写出乱码。
h3vgpuprobe 工具在加载任何权重之前就抓到了这两处问题。

策略是逐次判断：形状不支持、没有设备或 `CWICK_H3V_GPU=0` 都回退到原有 CPU 代码，
GPU 路径永远不会卡住视频生成。同机 256x256 / 1 步实测：总耗时 269.4 s → 168.2 s，
DiT 单步 149.5 s → 51.6 s，视频 VAE 105.8 s → 50.5 s。两条路径解码后的帧相关度
为 0.9989（第 0 帧）和 0.9944（第 40 帧）；首帧条件注入依旧是第 0 帧对参考图
0.567、第 40 帧 −0.037。完整表格见 PERF.zh-CN.md。

## 93. 图像接口按需装载模型并上报真实进度

视频线完成后，应用 3（图像 + 视频两步向导）还剩两个易用性缺口。其一，图像接口
仍要求操作者先手动切到 `MiniMax-H3-Image`：冷启动的首次 POST 返回 503，H5 界面
弹出「请先在设置里切换」的阻塞确认。视频接口早已通过首次使用时装载管线解决了
这个问题——走的是与显式切换同一个单模型槽位与交换门。`srv_image_route` 现在照此
处理：若没有常驻图像管线，就调用 `srv_switch_image`，关闭交换门、释放文本/实时
权重、映射图像 checkpoint；并发切换返回 503 而不是双重装载。H5 的确认框已去掉，
模型状态标签改为「按需加载 · 首次生成自动热切换」。

其二，图像进度条是假计时（`6 + secs/30`），不管渲染在做什么都会涨满。渲染器其实
已经在发布实时快照（阶段 0 idle、1 tokenize、2 text_encode、3 pack、4 dit 带
step/steps 计数、5 vae、6 png、7 done），只是没有暴露出来。新增的
`GET /v1/images/progress` 在准入之前应答，返回 `phase`、`phase_name`、`step`、
`steps`、`idle`、`age_ms`，所以轮询永远不会排在那个人工约 40 分钟的任务后面。
H5 应用每秒轮询一次，把 DiT 计为进度条的 60%，结束时仍从响应的 `cwick` 对象填入
精确的各阶段耗时。轮询失败或过期只会让屏幕停在最后的快照上，不影响渲染。

本机用重编译后的二进制端到端验证：冷启动的 `POST /v1/images/generations`
自动释放文本模型并按需装载图像管线，随后对空 prompt 返回干净的 400（未启动渲染）；
`GET /v1/images/progress` 在静止时上报 `idle:true`。`make test` 通过，服务的 `app.js` 是重编译后的 33420 字节版本，
含三处对真实端点的引用。

## 94. "黑屏"其实是有效 mp4 里的白噪声，而监控把它记成了成功

`video_1795193877_1` 的日志与 `/v1/metrics` 在机械层面完全一致：任务完成、封装出
2,570,415 字节、上报 `video_jobs_succeeded_total 1` 和 `video_first_frame_jobs_total 1`。
总耗时 20,752.6 s，其中 DiT 在 30 步里占了 19,948.8 s（736x1280 下每步约 665 s）。
这些数据没有一处描述像素本身。取 `GET /v1/videos/{id}/content` 回读帧之后才看清
为什么看起来是黑屏：第 0 帧还留着可辨认的条件图像，只是压着很重的分块网格；
第 20、100 帧却是高频静态噪声（平均亮度 70-73，平滑度比约 0.53，而第 0 帧约 0.75）。
相邻帧相关从保持区间的 +0.97 掉到片段深处的 +0.77-0.80——生成的 latent 仍是噪声，
VAE 只是忠实地把噪声解码了出来。

采样器的符号、timestep 嵌入和 sigma 到 timestep 的映射都与图像线逐项核对过；
图像线用同一套流匹配能产出正常结果，这几处一致。查到的具体缺陷是音频 shift：
checkpoint 在 `model_index.json` 的 `sigma_shift_scales` 里声明视频 12.0、音频 3.0，
但采样器对两条流都用 12.0，使音频的欧拉步长大了 4 倍。这解释不了视频帧，所以修它
时没有宣称它解决了黑屏。

由于一次生成要 5.7 小时，不可能拿它当反馈回路，于是加了两样东西：

1. 纯 CPU 的去噪信号，取自管线本来就持有的 latent。`spatial_corr` 是第一个生成
   latent 步（排除条件步）的横向 lag-1 自相关均值；自然解码的 latent 约 0.8，
   白噪声约 0。`snr_db` 是生成步与干净步的标准差之比。两者都进 `h3v_timing`、
   任务详情的 `cwick` 对象，并由 `suspect_noise` 新增
   `cwick:video_suspect_noise_jobs_total` 计数器，这样噪声片段在监控里不再与好片段
   无法区分。门限是暂定的（0.35，可用 `CWICK_H3V_CORR_FLOOR` 覆盖），等测过一段
   已知正常的片段再定。`tools/h3vprobe` 直接打印同样三个数，便于直接 A/B。
2. 配置驱动的 shift：`h3v_model_open` 读取 `_minimax_h3.sigma_shift_scales`
   （缺省 12.0/3.0），视频与音频各按自己的 sigma 网格步进，唯一的共享时间嵌入层
   不变；`CWICK_H3V_VIDEO_SHIFT` / `CWICK_H3V_AUDIO_SHIFT` 可在不重编译下 A/B。
   `CWICK_H3V_REF_NOISE_AUG=<level>` 还把图像线的带噪声首帧条件暴露成可选杠杆；
   缺省仍是干净首帧，在测过之前不改动既有输出。

`make test` 通过（含新增的 suspect-noise 计数器断言），`h3vprobe` 构建通过。
悬而未决的问题现在可以用一次短探针测量，而不必等一次长渲染：究竟是哪一处 DiT
约定让生成帧停在噪声水平。这里不宣称 shift 的修改已经彻底修复黑屏。

## 95. 标定噪声门限并对 DiT timestep 约定做 A/B

§94 里暂定的 0.35 相关门限需要实测数字。给 `h3vprobe` 加了两个纯 C 旋钮（不额外读模型、不用
Python）：`CWICK_H3V_CORR_TRACE` 在每次欧拉更新后打印生成步的 latent lag-1 自相关；
`CWICK_H3V_DIAG_MIX=<m>` / `CWICK_H3V_DIAG_TS=<t>` 从已知的
`m*noise + (1-m)*clean` 混合出发并强制一个条件 timestep，于是一步就能读出速度方向，
不必等 30 步渲染。`h3v_timing.clean_corr`（干净条件步上的同样自相关）和
`H3VPROBE_METAL=1`（probe 强制 Metal 以匹配 `--backend metal`）也一并加入。

256x256、2 步标定：VAE 编码的干净参考步测得 0.769-0.814，纯/近噪声测得 -0.145 到 0.068。
0.35 门限能干净分开，保留为缺省（`CWICK_H3V_CORR_FLOOR` 仍可覆盖）。

sigma=0.9 的单步探针一度显示 checkpoint 要 `t=sigma`：`ts=0.1`（代码的 `1-sigma`）
把 corr 从 0.068 推到 0.025，`ts=0.9` 推到 0.290。这是假象 —— 强制的 `dt=0.9` 是巨步，
latent std 从 0.947 爆到 2.042、越过干净锚，测的是分布外 overshoot，不是约定信号。
两次同配置的完整 8 步定论：`t=1-sigma`（原行为）最终 corr 0.360（`ok`），`t=sigma`
为 0.288（`suspect`）。改动已回退，缺省保持 `t=1-sigma`，与图像线一致。
`CWICK_H3V_TS=sigma` 保留该备选，供在 >=4 MP（checkpoint 真正在分布内）下再测 —— 
256x256 下它按设计就返回纹理噪声，短生成根本无法判断去噪质量。

现已确认：噪声门限已标定；音频 shift 是真 bug（checkpoint 声明 3.0，代码用 12.0），
已按配置修复；噪声片段不再能冒充 `succeeded`。30 步黑屏的根因仍未定论。剩余嫌疑是
shift=12 网格把几乎全部去噪压在末步，以及参考帧级别差异（图像线把参考行放在自己的
0.999 timestep、latent 用 `0.999*clean + 0.001*noise`；视频线用纯 clean latent 和单一
共享 timestep）。两者都需要 >=4 MP 的 A/B 才能确认，这里不宣称已修复。

## 96. 去噪信号读错了张量平面：修正布局后重测

§95 的噪声门限标定和 timestep A/B 建立在一个坏掉的读取器上。`vlat` 的布局是
`[C][T][H][W]` —— `h3v_pack3d()` 用 `((c*lt + t)*lh + y)*lw + x` 索引它，VAE 解码器
也用同样的步长取一个 cell 的 24 个通道。但 `h3v_lat_step_stats()` 把平面算成
`t*VD_LATENT*lh*lw`（这是 `[T][C][H][W]` 的假设），所以 t=1 时它测到的是通道 0 的
后段时间步与通道 1 的前段时间步的混合，而不是第一个生成帧。参考帧注入有镜像的
bug：它把编码器连续的 `[C][H][W]` 输出 `memcpy` 到 `vlat` 开头，在 `[C][T][H][W]`
里这会把 24 个通道撒到通道 0 的时间轴上，而不是全部放在 t=0。首帧看着仍 partially
正确，是因为每个 Euler 步后都会恢复打包好的参考行；但 DiT 第一步是在被破坏的
t=0 latent 上做条件。

两处都已在 `src/h3v.c` 修复。参考帧先按编码器布局组装，再逐通道散布到 `vlat`
的 t=0，并从同一块 buffer 打包；统计函数接收 `lt` 并按 `(c*lt + t)*lh*lw` 步长读取。
`h3v_internal.h` 里新增不依赖权重的 `h3v_layout_selftest()`：检查 t=0 平滑平面读数高、
t=1 平面读数低、pack/unpack 精确往返；`make test` 会断言它。

在 256x256、4 秒、seed 42、Metal、8 步下重测（四组配置完全一致）：

| 配置 | 最终 spatial_corr | clean_corr | snr_db | suspect |
| --- | --- | --- | --- | --- |
| 默认（t=1-sigma, shift 12, 纯 clean 参考） | 0.247 | 0.848 | +0.63 | 是 |
| `REF_NOISE_AUG=0.999` | 0.247 | 0.848 | +0.63 | 是 |
| `VIDEO_SHIFT=3` | 0.145 | 0.848 | +0.91 | 是 |
| 两者都开 | 0.145 | 0.848 | +0.91 | 是 |

修正后的 trace 从 corr -0.003（真正的白噪声）起步，单调升到 0.247；旧 trace 的
0.404 起点是跨通道误读。`clean_corr=0.848` 才是平滑参考帧的真实值。0.35 门限仍能
分开干净与噪声，保持不变。参考噪声增强在 0.999 上是 no-op —— 那只是在干净 latent
上叠约 0.1% 噪声；图像线真正的机制是独立的 0.999 timestep，而这个 checkpoint 表达
不了：`adaln_proj.linear.weight` 形状是 `[96768, 2688]` = 18 行 = 6 向量 x 3 模态，
没有 timestep 轴。把视频 shift 降到 3 会把去噪摊到各步（第 3 步 corr 0.072，
shift 12 是 0.007），但 8 步末值更低（0.145 对 0.247）。四组在 256x256 都低于门限，
而该尺寸属于 checkpoint 的分布外（支持尺寸 >=4 MP），所以这里不对 30 步质量下结论。
30 步黑屏的根因仍未定论；布局修复去掉了一个真实缺陷，但本身不证明故障已消失。

## 97. FL2VA 与 Ref2VA 都支持按请求选择

MiniMax-H3 下载包根目录含两个完整分区 `FL2VA/` 与 `Ref2VA/`，各有一个 62 GiB
的 `transformer/`。两者的 DiT 配置和全部 538 个权重键相同，视觉 VAE、音频 VAE
与文本编码器逐字节相同（`FL2VA/video_vae/source/model.safetensors` 与 Ref2VA
的副本哈希一致，文本编码器索引也一致）。只有训练出的 transformer 权重不同：
FL2VA 是首帧/尾帧转视频+音频，Ref2VA 是全模态参考转视频+音频。旧的
`h3v_model_open()` 取第一个存在的目录（完整下载包里就是 FL2VA），没有任何途径
访问 Ref2VA，于是一半 checkpoint 在 `/v1/videos` 上无法使用。

现在 `h3v_model_open()` 枚举所有可用变体（只有当其 `transformer/` 确实含分片时
才算可用），记录每个规范名及其 DiT 目录，并从 `CWICK_H3V_VARIANT` 或第一个找到
的项中挑默认值。`h3v_generate_variant()` 为每个任务固定一个分区；共享的 VAE 与
文本编码器保持映射。因为所有生成本就在模型锁下进行、且每个任务各自打开并关闭
DiT 分片，同一时刻只有一个 62 GiB transformer 常驻，两个变体不会争用。

`POST /v1/videos` 接受 `variant`（别名 `model_variant`），在入队前校验（拼错时
HTTP 400 并列出可用名字），并在任务详情的 `reference_image` 旁回报规范值。
stdout 在 DiT 打开处记录所选变体与确切分片路径。`tools/h3vprobe.c` 用
`H3VPROBE_VARIANT` 走同一条路径。纯 C 的 `h3v_variant_selftest()` 覆盖大小写
不敏感的名字匹配，`make test` 会断言。

端到端测试：`--backend metal`、256x256、4 秒、2 步、seed 42，两次使用同一个
`input/sample.png`（1152x1536）与 `input/sample_video.txt` prompt：

| 变体 | 状态 | 打开的 DiT 分片 | 条件 | total_ms | dit_ms | spatial_corr | suspect |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FL2VA | completed | `FL2VA/transformer` | first_frame | 285894 | 165533 | 0.376 | 0 |
| Ref2VA | completed | `Ref2VA/transformer` | first_frame | 256562 | 127090 | 0.360 | 0 |

两次都产出合法的 H.264 + AAC mp4（559228 对 568343 字节，哈希不同），首帧条件
均成功，`suspect_noise=0`。`/v1/metrics` 读到 `video_jobs_total 2`、
`succeeded_total 2`、`failed_total 0`、`first_frame_jobs_total 2`、
`suspect_noise_jobs_total 0`。两步冒烟测试只证明两个分区都能在这台机器上加载
并运行，不对 30 步质量下结论（256x256 低于 checkpoint 支持的尺寸）。

## 98. 视频 VAE 一直在解码未归一化的 latent

`vae/config.json` 带有 24 通道的 `latents_mean` / `latents_std`。Transformer 在
这个归一化空间里采样，但视频链路把两张表都忽略了：`h3v_vae_encode()` 直接返回
`quant_conv` 的原始输出，`h3v_vae_decode()` 把采样得到的 latent 直接送进折叠后的
`post_quant_conv` + `x_embedder`。`h3v_internal.h` 甚至写着「already
de-normalised by the caller」，但没有任何调用方做这一步。逐通道误差很大——std
范围 0.45-3.28、mean 范围 -1.37-1.07——36 层 ViT 解出来的画面对应不上采样结果，
写进 latent step 0 的参考帧对 DiT 也同样是错的。

修法都在 `src/h3v_vae.c`：两张表 + `vl_lat_norm()`（编码输出 raw -> 归一化，让
参考帧与噪声处在同一空间）与 `vl_lat_denorm()`（解码输入 归一化 -> raw，在分块
之前做一次）。`h3v_latnorm_selftest()` 校验往返误差，以及零 latent 必须正好落在
逐通道均值上，已进 `make test`。

三组实测把这件事与旧的「黑屏」结论区分开：

* 256x320 的编码->解码往返（不过 DiT）：R 通道相关 **0.8762**，重建是清晰人像；
  修复前同一 latent 解出来是 16 像素彩色网格。
* 常量（全零）latent 解码为平滑色场，说明解码器与分块本身正常；网格来自数值，
  不是分块。
* 端到端第 0 帧（被钉住的参考步）从平板网格变成了清晰人形轮廓。

本机端到端，`--backend metal`，`input/sample.png`（1152x1536）+ 
`input/sample_video.txt` 提示词，768x1024、**4 步**、seed 42、FL2VA：完成耗时
2,953,739 ms（DiT 2,241,221 ms、视频 VAE 678,956 ms），`spatial_corr 0.309`、
`suspect_noise 1`、mp4 2,923,218 字节。第 0 帧有人形，第 40/100 帧又退化为网格。
4 步不构成质量结论：`shift=12` 下 sigma 网格是 1.0、0.973、0.923、0.8、0，最后
一步宽 0.8，几乎全部去噪都挤在这一次无条件跳步里。

两组探针界定了剩余工作：

* timestep 约定是对的。已知混合 0.9*噪声 + 0.1*干净、在 sigma=0.9 走一步：
  `t=1-sigma` 把 `spatial_corr` 从 0.008 推到 0.334，`t=sigma` 推到 -0.042。
* 欧拉方向尚未定论。小步长（dt=0.1、0.2）下**减号**朝干净锚移动（dt=0.2 时
  corr 0.216 -> 0.289、std 0.699 -> 0.886），加号反而降（0.216 -> 0.146）；但
  同尺寸同步数的 4 步端到端 A/B 更偏加号（0.300 对 0.277）。4 步被 0.8 宽的末步
  主导，过冲会掩盖方向，端到端测不出来。默认保留实测胜出的 `plus`；
  `CWICK_H3V_SIGN=minus` 暴露另一方向，供 30 步 A/B 使用。本机 768p、30 步约 5
  小时，这是下一次测量，不是猜测。

为这次 A/B 新增的探针旋钮：`CWICK_H3V_DIAG_DT`（缩小强制步长）、
`CWICK_H3V_SIGN`；`tools/h3vencprobe` 支持 `H3VPROBE_W/H`，可以在渲染尺寸而不是
固定 64x64 下驱动编码器。

## 99. DiT 按转换后布局读取了原始权重

transformer 分片保留的是原始张量名与布局，参考加载器做的两次重排并不会隐含在字节里：

* 融合 QKV 的权重行是 `head0: q|k|v, head1: q|k|v, ...`。把输出直接切成三个连续块后，
  第 h 个头的查询 actual 上来自第 0 个头的键，以此类推：每个注意力头都在混合错误的流。
* `mlp.fc1` 输出的是 `[gate | value]`，而 SwiGLU 按 `[value | gate]` 解读，结果把激活施加
  在 value 上，又乘了未激活的 gate。

修复在 `src/h3v.c`：CPU 路径用 `h3v_split_qkv()` 把原始输出散开；Metal 路径新增
`gemm_bf16_qkv` 内核，在装载 tile 时对权重行做置换，直接把 q/k/v 写进三个平面，
不再多一遍拷贝。SwiGLU 改从前半取 gate。50 层 block 和 2 层 token refiner 用的是
同一种投影布局。`CWICK_H3V_LEGACY_LAYOUT=1` 可恢复旧解读做对照。

同输入 A/B（256x256、1 步、DIAG_MIX 0.6、dt 0.1、seed 42、Metal）：

| 信号 | 旧布局 | 修正后 |
| --- | --- | --- |
| cos(速度, data-ward 参考方向) | -0.142 | **+0.954** |
| 参考场最佳缩放 | -0.042 | 0.919 |
| \|速度\| | 206.73 | 679.06 |
| 第 0 层 MLP 输出 rms | 86.32 | 13.98 |
| 第 49 层残差 rms | 22497 | 7570 |
| 生成步 `spatial_corr` | 0.338 判为噪声 | **0.501 正常** |

融合内核与主机散开逐位一致，两边都给出 `|v| 679.0618`、`cos 0.954`、`corr 0.501`；
`h3vgpuprobe` 已覆盖新管线。

本机端到端验证：`--backend metal`，`input/sample.png` + `input/sample_video.txt`，
256x256、4 步、seed 42，DiT 结束时 `spatial_corr 0.927`（`suspect_noise 0`），
609,140 字节 mp4，全程 384.1 s（文本 70.1 s、DiT 244.8 s、视频 VAE 63.2 s、音频 VAE 4.3 s）。
解出的第 20、60、100 帧是参考人物，不再是修复前同一 prompt 的格子/雪花。256x256、
4 步只做正确性判定；768p / 30 步的质量数据仍需单独测量。

同样 4 步、768x1024 也越过了门限：`spatial_corr 0.963`、`clean_corr 0.929`、
`snr_db 2.56`、`suspect_noise 0`，mp4 1,249,044 字节，全程 3,008.3 s（文本 63.3 s、
DiT 2,282.4 s、视频 VAE 650.4 s、音频 VAE 4.4 s、封装 0.4 s）。原视频抽帧全是绿色格子，
修复后第 60、100 帧是清晰且在动的人物；第 20 帧仍有单侧格子接缝、第 0 帧有横向条带，
这是时间维 VAE 边界的独立遗留问题。768p / 30 步的质量测量仍未做。

## §100 图像+视频应用的快速验证档

H5 工作室（应用 3）此前已具备两步「图像 → 视频」流程，服务端也接受 256×256 画布和
4 秒片段（引擎会把两者分别向上取整到 32 像素画布倍数和 17 帧整段）。无法缩短的是采样：
DiT 步数固定 30，只能通过进程级环境变量调整，因此即便最小画布也要几十分钟，算不上快速验证。

`POST /v1/videos` 现在接受可选的 `steps` 整数。`h3v_generate_variant_steps()` 按任务传入
并把它限制在 [1, `H3V_SAMPLE_STEPS`=30]；旧的 `h3v_generate_variant()` 保持原签名并传 0
（引擎默认）。这同时修掉一个潜在缺陷：环境变量此前接受 1..256，而 sigma/时间步暂存数组
只按 30 分配，大于 30 会越界写。两条路径现在共用同一个上限。任务对象回报请求的 `steps`，
完成后的 `cwick` 对象仍回报实际步数。默认行为（不传字段或传 0）不变。

应用第二步新增 256×256 / 320×320 画布选项、步数输入框，以及一键「快速验证档」
（256×256、4 秒、4 步）。本机上这条 256² / 4 步路径在 384.1 s 内生成了清晰且在动的人物
（§99），预设正是基于这次实测。完整 30 步调度仍是质量渲染的默认值。

## §101 视频线的内存感知驻留

文本/模型线早就把各权重阶段接入了共享内存预算（`--mem-fraction`，默认主机内存的
80%；`--max-memory`；禁 swap 时 wire）。视频线没有：它的 DiT（61.73 GiB）每次请求
按需映射、退出时解除映射，从不 wire，于是 VM 可以自由压缩/换出 30 步采样循环每步都要
重读的那些页；两个 VAE 虽在进程启动时映射，同样没有 wire。

`cwick_mem_fit_model()` 现在承担单个权重阶段的统一决策：给定该阶段的映射字节数、
运行期暂存和预留量，把总和与上限比较，在禁 swap 模式下只在该阶段装得下时才 fault-in
并 wire。超预算的阶段刻意保留为惰性、SSD 支持的映射——wire 一个超预算阶段的一部分只会
把别的东西挤进压缩器。视频线在每次 safetensors 打开后调用它：两个 VAE 在模型加载时
（进程常驻），DiT 在生成时（一次采样期常驻；`close()` 自动 unwire）。
`CWICK_H3V_WIRE=0/1` 可覆盖决策做 A/B。

两个 62 GiB 阶段刻意错峰：text encoder 只在第 1 阶段映射，第 3 阶段映射 DiT 前就释放；
若两者同时 wire，峰值会到 134 GiB、超过 102.4 GiB 上限，恰恰造成本打算避免的 swap。
text encoder 是只读一次的流式访问，保持惰性既正确又比 wire 更省。

新增四个指标：`cwick:model_weight_bytes`、`cwick:model_resident_bytes`、
`cwick:model_target_bytes`、`cwick:model_wired`。

实测（同机、Metal、256×256 / 1 步、驻留路径）：VAE 9.70 GiB + audio VAE 0.56 GiB +
DiT 61.73 GiB 全部记录为 `wire resident`，上限 102.40 GiB（128 GiB 主机的 80%），
合计约 72 GiB 已 wire，全程无 wire 失败。禁 swap 的决定性证据是换页计数而非
`swapused` 电平：`Swapouts` 从 137479946 → 137479946（增量 **0**），`swapused`
反而从 6052 降到 5916 MiB；`Compressions` 增量只占其开机累计值的 0.009%。早先一次
观察到的 `swapused` 上涨 2.1 GiB 是其他进程的全局噪声，不是模型驻留页被换出：`mlock`
成功后这些页不可能被 paging。`denoise check` 报空间相关 0.955（门槛 0.35）。
`make test` 与 `make h3vprobe` 均通过。

## §102 视频接口只使用 Ref2VA

§97 让下载包里的两个分区都能被选中，这是弄清各自行为的正确手段。但
`POST /v1/videos` 的输入是 prompt 加可选参考图、输出是一段生成视频，对应的是
全模态参考工作流，因此与接口语义匹配的 transformer 是 Ref2VA。首帧/尾帧分区
服务于另一种请求形态，此前只是因为目录排序在前才成为默认。

本版 `h3v_model_open()` 只发现 `Ref2VA/`（或它的扁平副本），默认变体就是它；
请求里 `variant` 指定首帧/尾帧分区会在入队前被 HTTP 400 拒绝，其分片不再探测、
不再映射。这消除了接口上意外的质量 A/B，同时不改变共享 VAE、文本编码器、
任务模型与进度 API。

## §103 Ref2VA 参考图呈现与参考几何

切到 Ref2VA 后，管线里还有两处沿用首帧请求形态：

1. 参考图被缩放到生成画布，只作为 latent 行进入序列。全模态参考工作流里，
   图片同时也是文本呈现的一部分（prompt 前是 `Picture 1: <视觉块>`），视觉
   token 行在 AdaLN 上取视频模态。新增的 `h3_text_cond_ref_dir()` 复用图像线
   的视觉塔构建该呈现，并返回逐行模态标签。`CWICK_H3V_REF_PRESENT=0` 可回到
   裸 prompt。
2. 图片参考按自身分辨率编码，不继承生成画布。`CWICK_H3V_REF_EDGE`
   （32..6272，未设置 = 画布）缩放参考短边后编码，条件行按自身 latent 网格
   打包并使用自己的空间 RoPE 网格；图片参考还让共享音视频 rotary 时钟前进
   一格，此前缺失。`CWICK_H3V_CONDROWS=0` 与显式短边不叠加：旧的目标 t=0
   布局要求参考网格与目标网格相同。
   注意：上限 `H3V_REF_EDGE_MAX`(6272) 与生成画布上限 `H3V_MAX_SIZE`(2560，2K)
   是两个独立常量。早期实现误用画布上限截断 REF_EDGE，导致显式 2048 被静默回落到
   画布尺寸；现已拆开，设 2048 会真正按参考图自身短边编码（1152×1536 → 2048×2720）。
   画布上限本身也从 1280 放开到 2560（低于 4K）：应用 3 的 `vsize` 增加 1600×900 /
   1920×1080 / 2560×1440 三档，`sizeHint` 同步；图像线 `H3_MAX_SIZE` 本就是 6272、
   已覆盖 4K，无需改。尺寸换算提成纯函数 `h3v_ref_encode_size()`，生产路径与单测
   共用；`make test` 覆盖 2048 生效、未设置回落、越界忽略，以及 2560 准入 / 3840 拒绝。
   参考短边只影响 VAE 编码栅格与条件行，不改变生成分辨率；DiT 行数随画布近似平方
   增长，2K 全 30 步要数小时——UI 提示引导先用 4 步验证档。

两个旋钮都不改变权重精度、采样器或任务/HTTP 模型。

新呈现生效前的接口实测（`--backend metal`、256x256、4 秒、30 步、seed 42、
`input/sample.png` 加 `input/sample_video.txt`，POST `/v1/videos` 指定
`variant:Ref2VA`）：1,342,128 ms 完成（文本 58,397、DiT 1,232,712、视频 VAE
45,123、音频 VAE 3,906 ms），102 帧、612 KiB mp4，`spatial_corr 0.899`、
`snr_db 0.17`、`suspect_noise 0`。全程 `cwick:host_swapouts_total` 保持
137485839 不变，102.40 GiB 上限下 wire 66.28 GiB。抽帧 0/20/60/100 都是
米色毛绒外套、贝雷帽的坐姿人物，第 100 帧按 prompt 转身，旧的第 0 帧横条已消失。
遗留问题是质感：外套上颗粒与零散绿色斑点明显，整体偏软，这正是下面 4 步 A/B
要验证的点。

同条件 4 步 A/B（`--backend metal`、256x256、4 秒、seed 42、同一 prompt 与
RGB 参考图、4 步采样）：

| 配置 | 文本 token | 序列行数 | 总耗时 ms | DiT ms | spatial_corr | clean_corr | snr_db |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `REF_PRESENT=0`，参考图按画布 | 34 | 2358 | 321737 | 202266 | 0.891 | 0.885 | -0.60 |
| `REF_PRESENT=1`，参考图按画布 | 1417 | 3741 | 514776 | 342490 | 0.905 | 0.877 | -0.36 |
| `REF_PRESENT=1`，`REF_EDGE=512`（336 条件行） | 1417 | 4013 | 522574 | 356200 | 0.921 | 0.905 | -0.17 |

三组都产出 102 帧 mp4 且 `suspect_noise 0`。视觉呈现多花约 60 s 文本编码与
140 s DiT（4 步），但第 20/60/100 帧更平滑，脸型、短发、贝雷帽与米色外套
更贴近参考；裸 prompt 那组脸更大更糊、彩斑更多。参考图改按 512 短边编码后
三个相关系数都再升，但在 256x256 下视觉只是边际改善，因此保留为可选。
视觉呈现成为新默认。

同一 30 步接口任务在开启呈现后复跑：2,145,126 ms（文本 111,825、DiT
1,981,593、视频 VAE 45,705、音频 VAE 3,849 ms），102 帧、596 KiB mp4，
`spatial_corr 0.814`、`snr_db -0.52`、`suspect_noise 0`。抽帧 0/20/60/100
仍是同一张脸、贝雷帽与米色外套，且没有上一版的绿色斑点，只是构图更近、
256x256 下质感依旧偏软。`spatial_corr` 下降不是画质结论：它量的是 latent
内水平相邻相关性，不是清晰度或参考相似度。wire 权重仍是 66.28 GiB；
`cwick:host_swapouts_total` 整个会话涨了 988 个宿主页（16 MiB），来自其他
进程，远小于且不可能来自已 mlock 的模型页。两次 30 步都不构成 768p 画质
结论：256x256 只是最小可跑验证。

## §104 QuickTime 黑屏的根因是 avcC 结构非法，不是生成器

一个 256×256 / 30 步的 Ref2VA mp4 用 ffmpeg 解码时画面正常（抽帧是清晰、在动的人物），
但用 QuickTime 打开却是黑屏。这种「解码器之间分裂」的表现指向解码路径，而不是 DiT 或
VAE 输出——生成器在 §99 和 §103 已经修好。

用一个走 AVFoundation（QuickTime 自身使用的解码栈）的小 probe 稳定复现了失败：
`AVAssetReader` 在第一帧就返回 `AVFoundationErrorDomain -11821 Cannot Decode`，
底层 OSStatus `-12137`。同样的文件在 ffmpeg 里能解码，是因为 ffmpeg 会从 VideoToolbox
在每个 IDR 前写入的带内 SPS/PPS NAL 重建缺失的参数集；而只信任容器头的严格
`h264_mp4toannexb` 转换则报 `Global extradata truncated, corrupted stream or
invalid MP4/AVCC bitstream`。

根因在 `h3v_mp4.m` 的 `build_avcc()`。format-description API 返回的是一份扁平的
参数集列表（本编码器为 SPS 后跟 PPS，共两项）。代码把这份列表的项数写进了 avcC 的
`numOfSequenceParameterSets`，然后原样输出整份列表，完全漏掉了独立的
`numOfPictureParameterSets` 字段。得到的记录是
`01 64 00 14 ff e2 000a<SPS> 0004<PPS>`：它声称有两个 SPS，把 PPS 当成第二个
SPS 喂进去，还缺了 PPS 计数字节。ISO/IEC 14496-15 要求两段分开写——先 numOfSPS +
各 SPS，再 numOfPPS + 各 PPS。严格解析器拒绝这条非法记录并渲染黑屏；宽松解析器
靠带内 NAL 恢复。

现在 `build_avcc()` 按 NAL type 给每个参数集分类（7 = SPS、8 = PPS），分别收进两个
数组，再用正确计数输出两段。修后的记录是
`01 64 00 14 ff e1 000c<SPS> 01 0004<PPS>`。

验证走 `tools/h3vmuxprobe` 的合成色带，几秒内就能测封装器，不用等几分钟的 DiT/VAE：

- avcC 字节现在带 `e1`（一个 SPS）和末尾的 `01` PPS 计数；
- AVFoundation probe 成功解出第一帧（`OK 64x64 mean_luma=87.07`），而旧文件返回
  `Cannot Decode`；
- 只依赖容器头的严格 `h264_mp4toannexb` 转换成功。

这个修复是应用 3 在线预览与下载的前提：一个画面正确但 QuickTime / Safari 拒绝解码的
文件，对用户而言就是一个黑屏播放器。应用自身的文案也一并对齐了当前视频语义：参考图
复选框不再把字段写成 `reference_image`（线上字段是 `image`），也不再声称图片会替换
首帧 latent——Ref2VA 是把它作为 Picture 1 编入文本条件（§103）。任务的 `variant`
现在显示在进度面板里，让 Ref2VA 默认值从接口到界面全程可见。

真实生成也确认了修复，不只靠合成探针：任务 `video_1795373529_1`（256×256、4 秒、4 步、
`sample.png` 参考图、默认 Ref2VA）用时 488.2 s（文本 69.1、DiT 356.7、视频 VAE 56.1、
音频 VAE 4.1、封装 0.3），产出 615,030 字节 / 102 帧。其 avcC 为
`01 64 00 14 ff e1 000a<SPS> 01 00 04<PPS>`；AVFoundation probe 返回
`OK 256x256 mean_luma=156.36`（修复前是 `Cannot Decode`），严格 Annex-B 转换通过，
抽帧可见参考图中的人物、贝雷帽与外套，无黑帧、无绿格子接缝。256p / 4 步下的偏软是
分辨率与步数上限，不是解码失败。

## §105 视频任务期间的内存打满：各 stage 独立驻留、没有共享预算

30 步 256×256 Ref2VA 任务的 DiT 跑到 200–241 s/步（第 1–6 步分别是 216072、206066、201178、199627、227345、241420 ms），
是紧邻其前那个 4 步任务 89.2 s/步的 2.4 倍——画布相同、序列长度相同（3677 行）、wire resident 的 DiT 也相同（61.73 GiB）。
通过 `/v1/metrics` 采集的主机指标解释了差距：

- 02:51:49 启动时，swap 为 4.28 / 5.00 GiB，`host_compressions_total` = 2226116165；
- 03:28:15 启动时，即两个视频任务之后，swap 为 6.61 / 8.00 GiB（macOS 把 swap 文件从 5 扩到 8 GiB），compressions = 2279810266；
- 也就是**两个视频任务期间 swap +2.33 GiB、compressions +5370 万次**；
- 临时的 unmap 日志显示 DiT 采样期 RSS 峰值是 73.92 GiB（transformer 关闭时 `RSS 73.92 -> 12.20 GiB`）；
- 空闲时只有 root 模型驻留（`process_footprint_bytes` = 28.5 GiB），5 秒采集窗口里 compressions/swapouts 增量为 0——压力由视频任务制造，不是空闲服务。

根因。`cwick_mem_fit_model()` 一次只评估单个 stage 是否能放进 102.40 GiB（128 GiB 的 80%）上限，
没有跨 stage 的累加账本。视频线对每个 stage 调用一次，于是每个 stage 都独立报告「fits under budget」并被 wire。
`h3v_model_open()` 在模型打开时就把 video VAE（9.70 GiB）和 audio VAE（0.56 GiB）wire resident，
并且整个模型生命周期都持有；`h3v_generate()` 随后在采样时再打开并 wire DiT（61.73 GiB）叠加上去。
因此并发 mlock 的集合是约 72 GiB（RSS 峰值 73.92 GiB），而不是一次只有一个 stage。
wired 页不能被压缩或换出，压力就转嫁到主机上所有其他进程：全局 compressor 与 swap 文件上升。
非 wire 的临时区（attention buffer、Metal arena 临时页）在 `waitUntilCompleted` 内产生缺页，这是 2.4 倍步时劣化的可能来源。
这是关联而非严格 A/B：两个任务在同一进程里背靠背运行，系统状态的累积是混杂因素。

后续一个任务确认了这条曲线的极端端：`video_1795375995_1`（736×1280、30 步、seq 29357 = 256p 的 8 倍）
DiT 第 11 步 731656 ms 单调劣化到第 30 步 788257 ms——总共 6.3 小时、约 13 min/步，而 256p/4 步是 89 s/步。
同一运行内、权重和行数都不变而步时持续上升，正是系统压力累积导致缺页/压缩率上升的特征。

建议修复，与 goal 约束一致（不剪枝、不降精度、Metal、80% 预算、禁 swap）：让 VAE 驻留变懒——
在第 4/5 阶段才 fault in video/audio VAE，用完即 unwire，使 DiT 采样期只持有 61.73 GiB（峰值 73.92 → 约 63 GiB）；
给 `cwick_mem_fit_model()` 增加 `already_resident_bytes` 累加参数，并让所有活动视频 stage 共用一份账本；
把 80% 上限当作跨 stage 的全局预算，而不是逐 stage 的判断。
改前先做 4 步 A/B，对比 RSS 峰值、compressions 增量与 DiT 步时，再决定是否提升为默认。尚未实施——本节仅为诊断。

## §106 大画布视频的格子伪影来自过小的 VAE 解码分块

一个 736×1280 / 30 步的 Ref2VA 片段（`video_1795375995_1`，sample.png 参考图）在 §104 修复后能被 QuickTime 正确打开，
但画质发糊、带网格：脸部清楚，服装纹理和轮廓却覆盖着规则方格。这不是黑屏、不是解码失败，也不是采样不足——任务跑满了 30 步。

证据按取证顺序：

- ffprobe：736×1280、H.264 High、yuv420p、4.71 Mb/s、102 帧。avcC 已是 §104 的合法形式
  （`ff e1 ... 01 00 04`），AVFoundation probe 返回 `OK 736x1280 mean_luma=63.89`，严格 Annex-B 转换通过。容器与解码器没问题。
- 这个 job 在耗时上是离群点：DiT 每步 731656 → 788257 ms（共 6.3 小时），因为画布产生 27600 个视频行 / seq 29357
  （256p 的 8 倍）叠加 §105 的驻留压力。慢和格子是两个独立症状，本节只处理后者。
- 抽帧的梯度能量在 lag=16 px（一个 latent cell = 16 像素）和 lag=192 px（一个 VAE 解码 tile）出现峰值；
  水平 tile 边界 x=160/320/480/640 的梯度是邻域的 2.4–3.6 倍。周期性指向 VAE 解码器，不是 DiT。
- `tools/h3vencprobe` 在 736×1280 下把平滑 ramp 不经过 DiT 直接做 VAE 往返，重建图复现了接缝：
  边界/非边界梯度均值比水平 4.14 倍、垂直 5.33 倍。解码**常量全零 latent** 同样产生 16 px 的 cell 纹理和 tile 周期台阶，
  确认伪影由解码器的分块引入，与采样出的 latent 内容无关。

根因。视频 VAE 解码器把 latent 切成 12×12 cell 的 tile、2 cell 重叠。它的 rotary 表用 tile 局部坐标
（`2*(i+0.5)/n - 1`），每块都铺满整个 [-1,1]，于是相邻块对同一行给出不同相位的预测。2 cell（32 px）的线性 cross-fade
盖不住这个不连续，边界以 192 px 网格泄漏。256p 的 latent 是 16×16、实际只有一块，所以之前所有 256p 测试都看不到。

第一次尝试按图像线已验证的 `_stitch_tiles` 重写拼接（整行 raw buffer、带内双向混合、裁远端边、最后量化）。
实测否决：往返相关系数仍是 0.9913，边界比没有改善（一个变体把 x 接缝恶化到 35 倍），于是完整回退。调混合权重不是解法，
问题在 tile 几何本身。

有效的修复是放大 tile。`H3V_VAE_TILE_CELLS` 从 12 改 48、`H3V_VAE_TILE_OVERLAP` 从 2 改 8，
两者可用 `CWICK_H3V_VAE_TILE=N` 在运行时覆盖（重叠变成 N/6）。同一 736×1280 平滑 ramp 往返实测：

| tile cell | 接缝比 x | 接缝比 y | R 相关系数 | 解码耗时 |
|---|---|---|---|---|
| 12（旧） | 4.48 | 5.33 | 0.9913 | 56.0 s |
| 24 | 1.89 | 6.50 | 0.9928 | 42.1 s |
| 48（新） | 0.00 | 1.61 | 0.9937 | 31.8 s |

更大的 tile 也更快（重叠往返次数少），且 80×46 latent 的 attention 序列仍可控。用新默认重编译后，
x=640 / y=640 的边界梯度是 0.63 / 0.71，而此前是 4–10。权重、精度、采样步数都没变。

736×1280 / 4 步端到端片段随后完成（`/tmp/h3vprobe/fix736.mp4`，102 帧 / 4.25 s，2.46 MB）。
采样报告 `spatial corr 0.979 / clean corr 0.981 / snr -0.07 dB / suspect_noise 0`；
抽取 6 帧（n=0/20/40/60/80/101）得到三点独立结论：

- **192 px 的 tile 接缝消失**。与早先 tile=12 / 30 步片段逐帧对比，x=160/320/480/640 的横向条带不再出现。
  tile 几何修复在端到端链路上成立，不只是平滑 ramp 往返里成立。
- **仍有细密的 16 px 棋盘/点阵纹理覆盖全画面**。放大后皮肤、头发、衣服上都是规则的 cell-三角格纹。
  这是解码常量全零 latent 即可复现的 cell 级固有伪影，放大 tile 并不解决它。§106 只修掉了大尺度 tile 格子。
- **4 步片段明显比 30 步片段软**：衣料纹理和五官细节不再可辨。这是采样步数不足，与 tile 改动无关。

身份部分保留（同一位长发微笑人物，片段内有轻微转头），但参考服装没有保留：`sample.png` 是米色毛绒外套 /
吊带裙 / 长筒靴，生成画面是素色深色高领毛衣。4 步经过文本呈现路径传递的参考细节太少。
没有黑帧、绿块或单侧格子残留。

结论：736×1280 画布上，清晰度现在受限于采样步数和 VAE 固有的 16 px cell 纹理，而不是 tile 接缝。
增加步数是管线内唯一能提升细节的手段，代价是实测约 6 h。该运行也实时印证了 §105：
空闲页掉到约 27k（436 MiB），swap 5.6/7.2 GiB；DiT 每步 752–813 s，VAE 解码 979.7 s（合计 4307 s）。

## §107 256×256/30 步片段：QuickTime 发糊是分辨率天花板，§105 的步时归因被证伪

片段 `output/cwick-video-video_1795406010_1.mp4`：256×256 / 102 帧 / 4.25 s / 30 步，
Ref2VA，参考图 `sample.png`、prompt `sample_video.txt`，经 `/v1/videos` 在 Metal 下生成。

容器不是原因。avcC 是 §104 的合法形式
（`01 64 00 14 ff e1 00 0a<SPS> 01 00 04<PPS>`：一段 SPS、一段 PPS），H.264 High + AAC 32 kHz 立体声，
包 DTS 以 24 fps 从 0 到 101000 单调递增，完整解码无错误帧。QuickTime 里看到的发糊是生成图像本身，
不是封装、avcC 或时间戳问题。

抽取 6 帧（n=0/20/40/60/80/101）：身份与服装现在贴近参考（米色毛绒外套、吊带裙、贝雷帽）——
30 步把早先 4 步片段丢掉的参考细节传过来了（§106）。无 tile 接缝、无绿块、无黑帧。
脸部与背景仍有轻微 16 px cell 纹理，整体偏软。在这个画布下两者都在预期内：
256×256 是最小验证画布，经 VAE 的 16× 空间压缩映射到 16×16 latent，
解码器固有的 cell 纹理（§106 用全零 latent 复现）占据每个显示像素的很大比例，
任何全屏播放都要把 256×256 放大。这是分辨率天花板，不是退化：
没有采样器或驻留方式的改动能让 16×16 latent 渲染得像 80×46 一样。

日志与 metrics：text 112110.5 ms（呈现后 1417 token），DiT 30 步、60–63 s/步、合计 1979.5 s，
去噪 `spatial corr 0.847`、`gen/ref std -0.74 dB => ok`，video VAE 54114.3 ms、
audio VAE 3951.9 ms、mux 200.9 ms，总计 2196.4 s。
峰值 RSS 73.52 GiB（unmapped `73.52 -> 11.79 GiB`），wired 集合与 §105 相同——
model-open 起持有 VAE 9.70 GiB + audio VAE 0.56 GiB，采样期 DiT 61.73 GiB。
任务结束时空闲页 4.8M（95% 空闲）、swap 3930/5120 MiB、compressor 339 MiB：无主机内存压力。
serve 日志记录了 2096 次 `/v1/metrics` 抓取，全部 HTTP 200、25133 字节；
本次分析时进程已停止，因此没有实时抓取可用。

**对 §105 的修正**：同样的 256×256/30 步形状、同样 3677 行、同样 61.73 GiB wired DiT、
同样 73.52 GiB RSS 峰值，这次跑出 60–63 s/步，而 §105 记录的是 200–241 s/步。
wired 集合完全相同，差异在主机 swap/压缩压力。§105 的步时回归是系统级压力的函数，
不是 VAE 提前 wire 造成的。因此 §105 提议的 VAE 懒驻留修复得不到这次对比的支持，**未实施**。
保留现有驻留模型；要保护步时，应在长时间视频任务前结束或清理其他吃 swap 的作业，而不是改驻留。

实用结论：256×256 是正确性/速度冒烟测试，不是画质档位。要细节就放大画布——
在主机压力低时远达不到 §105 的 6.3 h 异常。要速度就降步数，但 4 步会丢参考服装（§106）。
16 px cell 纹理是 VAE 解码器固有属性，无法从采样器侧调掉。

## §108 QuickTime 里的「跳帧」是时间维 VAE 接缝，不是封装问题

一段 544×960 / 30 步 Ref2VA 成片（`video_1795410431_1`，102 帧、4.25 s）在
QuickTime 里看着像跳帧。三项独立检查都排除了容器：包表里只有一个 stts 项、
步长恒为 1000/24000 s、dts 严格递增；avcC 含一个 SPS 和一个 PPS；用
AVFoundation AssetReader 严格解码，102 帧全部解出、帧间隔只有 0.041667 s 一个值、
零异常间隔。`ffmpeg -f null` 报的 `non monotonically increasing dts` 来自 null
muxer 重新编码解码流，不是源文件错误。时间戳、关键帧位置（IDR 在 0/48/95）、
MP4 box 都没有问题。

跳变在像素里。逐帧平均绝对差（中位数 4.52）呈现干净的周期性脉冲：进入第
20/40/60/80/100 帧的跳变均值 **21.34**，是其他区间的 2–14 倍，而紧邻的前一帧
只有 1–4。较小的峰值落在 4 的倍数帧（9–11.6）。周期正好是 20 帧 = 5 个 latent step
= 一个时间 chunk，4 帧峰值对应单个 latent step。

根因：解码器几何把 17 个像素帧折成 5 个 latent step（`17n+5` 像素帧 ↔ `5n+2`
latent 帧）。相邻 latent chunk 重叠 2 个 latent step，正确解码应把重叠部分也解出来、
对共享帧做线性交叉淡化。本引擎的解码器一次性解完所有 latent step，只对空间 tile
做 cross-fade，**没有时间维重叠解码、也没有时间维 cross-fade**。于是每 20 帧的边界
就是硬切，观感上每约 0.83 s 跳一下。102 帧也不在 checkpoint 合法的 `17n+5` 网格上
（最近的是 90 或 107）。

修复方向（未实施）：像空间 tile 拼接那样，每个 chunk 带 2 个 latent 的重叠解码，
并对共享帧做混合。这会改动 VAE 解码循环并和空间分块交叉，需要单独 A/B 与回归后再落地。
低成本缓解是请求合法帧数，能去掉额外的截断，但消不掉硬切接缝。

日志/metrics：text 112.1 s、DiT 30 步每步 363–409 s 共 11093.8 s、video VAE
465.0 s、audio VAE 3.9 s、mux 0.3 s，合计 11678.2 s。去噪健康（`spatial corr 0.901`、
`gen/ref std -0.43 dB => ok`）。相比 §107 中 256×256 的步时慢 6 倍，是 transformer
行数约 4.2 倍（15300 vs 3677）的预期代价——attention 随行数平方增长。任务前后主机
都是 95% 空闲、swap 3691/5120 MiB，不存在内存压力。

## §109 时间维 VAE overlap + cross-fade 消除了分块边界硬切

§108 把 QuickTime 里的「跳帧」定位到视觉 VAE 一次性解完所有 latent、时间维
没有重叠，于是每个 5 步分块边界都是硬切。训练规则把 17 个像素帧折成 5 个
latent step，每个编码分块丢掉 3 个尾部 latent，所以正确解码要把 2 个 latent
的重叠部分也解出来，并对共享的 5 帧做线性混合；rope 按分块自身 7 个 latent
做长度归一化。原先的整段解码两样都没做。

修复：`h3v_vae_decode` 改为按 5 步分块、带 2 步重叠逐块解码。每块用局部 rope
解 7 个 latent，取解码帧 [3,20) 的 17 帧作为主输出，并把开头 5 帧与上一块的
5 帧尾部按 1→0 / 0→1 的权重线性混合。缓冲区放 17*chunks+5 帧，mux 取前
17*chunks，于是所有入片帧都来自做过 cross-fade 的主输出。空间分块及其自身的
cross-fade 不变，在每块内部照常执行。`lt<=5` 与 `CWICK_H3V_TEMPORAL_XFADE=0`
保留整段解码，供参考帧往返测试和 A/B 使用。

同一 latent 的 A/B（256×256、1 步、seed 42、Metal，只差开关）：边界 MAE
5.358 → 4.212（-21%），单帧峰值 6.05 → 4.69（-23%），边界 5 帧能量
11.57 → 8.40（-27%）。旧的单帧脉冲（第 20 帧 5.43、第 21 帧 0.43）变成
5 帧渐变（第 17/18/19 帧 4.57、2.48、0.34）。两次去噪一致（`spatial corr` 均
0.987），抽帧拼图没有新的空间接缝、黑帧或块状伪影。VAE 解码因重叠的冗余计算
从 39.2 s 升到 53.1 s（+35%）。总墙钟时间两次不可比：DiT 阶段自身从 126.5 s
波动到 85.7 s，与本开关无关。

按训练规则混合的首帧权重是 1，所以第一个共享帧仍有约 4.5 的跳变；这部分残余
是采样器自身的 5 步边界，不是 VAE 接缝。要消掉它需要比 checkpoint 训练时更多
的重叠 latent。`make test` PASS。

## §110 时间维 cross-fade 修复后的第一条真实 30 步成片

256×256 / 30 步 Ref2VA 成片 `video_1795424360_1`（102 帧、4.25 s、24 fps）是
§109 落地后第一条完整质量的成片。按此前报过的两个症状复查：时间跳变与块状纹理。

容器干净：stts 只有一项，pts==dts 从 0 到 101000 步长 1000，avcC 含一个 SPS
和一个 PPS；AVFoundation AssetReader 解出全部 102 帧，帧间隔恒为 0.041667 s、
零异常。`ffmpeg -f null` 报的 dts 非单调和 §108 一样，是 null muxer 重新编码所致。

逐帧 MAE（中位数 4.46）里已没有旧的 20 帧脉冲：进入第 20/40/60/80/100 帧的
跳变是 3.17/3.76/7.48/6.51/2.66（均值 4.72），全片最大只有 12.53（第 83 帧）。
对比修复前的 256×256 成片（同为 30 步、不同 seed），边界均值 20.08 → 4.72
（−76%），最大值 33.17 → 12.53（−62%）。两处最大的亮度变化（第 85/84 帧 dL
8.77/8.43）正好是人物从正面转到侧面，属于 prompt 要求的动作；没有相邻重复帧
（最小 MAE > 0.5）。抽帧拼图看不到黑帧、绿块、格子接缝、重影或冻结帧，原始尺寸下
毛绒外套与五官都清晰可辨。

这是 §109 路径在真实采样质量下（而不只是 1 步 A/B）站得住的第一份证据：
cross-fade 消除了周期跳变，且没有引入空间接缝或新的模糊。两条成片构图不同
（seed 不同），所以整体清晰度的差异不归因于开关，只有边界脉冲的变化归因于它。

日志/metrics：text 84.45 s、DiT 30 步每步 60–67 s 共 1971.1 s、video VAE
63.77 s、audio VAE 3.97 s、mux 0.26 s，合计 2125.73 s。去噪健康（`spatial corr 0.847`、
`gen/ref std -0.74 dB => ok`）。video VAE 从 §107 的 54.1 s 升到 63.77 s，是符合
预期的重叠 latent 解码成本。计数器：1 个 job、成功 1、失败 0、可疑白噪声 0；
驻留 66.28 GiB、wired、RSS 峰值 73.97 GiB。主机 swap 启动时 8.06/9.00 GiB、
结束时降到 6.4/7.5 GiB——这是开机以来的高基线，但步时仍保持健康的 60–63 s，
再次印证 §107：步时回归跟任务期间的压力有关，而不是 swap 的绝对水位。

## §111 256×256 下脸为什么糊,以及什么真正能改善

针对 §110 那条 256×256 / 30 步成片单独复查了「脸糊」。发糊来自分辨率,不是采样、
cross-fade 或封装的缺陷。三项独立事实定位了它。

1. 输出网格。VAE 空间压缩 16×,所以 256×256 画布只有 16×16 latent,脸大约只占
   其中 4×4 个格子,latent 里没有可恢复的细节。同一脸部裁剪的拉普拉斯方差:参考图
   缩到 256 是 3991,生成帧是 1730–2547——生成前就少了一半高频,解码器补不回来。

2. 参考条件行。first-frame latent 路径默认按**生成画布**编码参考图,除非设置
   `CWICK_H3V_REF_EDGE`。日志写得很明确:`reference image encoded: 256x256 RGB ->
   1x24 latent (16x16)`,而同一张图是以 768×1024 RGB 到达的。这条路径能传递的身份
   细节上限就是 16×16 latent。

3. 文本呈现已经是高分辨率,不是瓶颈。`h3_text_cond_ref_dir` 把参考图经视觉塔按
   1024 短边编码(`h3_vision_size` → `h3_bound_edge(1024,1024)`),得到 1417 个文本
   token。这正是身份和毛绒外套仍贴近参考的原因:结构来自高分辨率呈现,逐像素的
   发糊来自 16×16 输出网格。

脸部对比印证了这个形态:五官结构、眼睛数、表情都正确,没有马赛克或重影——是均匀
发软,正是 16×16 网格重建到 256 像素的样子。逐帧 MAE(§110)与 cross-fade A/B(§109)
已经排除了时间接缝这个原因。

在 goal.md 约束下的优化边界:不加锐化滤镜、不加超分模型、不改精度或剪枝。引擎内
唯一能提升参考细节传递的旋钮是 `CWICK_H3V_REF_EDGE=N`,它把条件行按参考图
自身短边而非画布缩放;这是进程级环境变量,需重启服务生效。checkpoint 默认按
2048 短边编码参考图,本机放开到 `H3V_REF_EDGE_MAX`(6272),因此设
`CWICK_H3V_REF_EDGE=2048` 可复现官方呈现(如 1152×1536 → 2048×2720),而不是
被生成画布上限（现为 2560，且与参考范围相互独立）截断。短边越大条件行越多、DiT 耗时近似平方增长;本机在
80% / 禁 swap 预算下,2048 编码的瞬时缓冲约 46 GiB,需与驻留权重合计评估。
真正提升脸部细节的办法仍是加大画布(512×512 给 32×32 latent,细节容量约 4×),
代价同样是 DiT 时间近似平方增长。
此轮未跑 `REF_EDGE` 的受控 A/B:服务已驻留模型,第二个 DiT 实例会突破 80% / 禁 swap
预算;只做了无权重的尺寸换算单测(`make test` 通过)。

## §112 应用 3 用同一张尺寸配对表绑定两步画布

应用 3（`apps/imagegen`）此前已支持头像/服装/姿势三类上传、两个图像接口、异步视频任务轮询、
进度/在线查看/下载以及快速验证档。缺口是第一步 `size` 与第二步 `size` 相互独立：可能用
1536×2048 竖版图去配 1280×720 视频，条件图与生成帧的构图不一致。

现在两步共用 `apps/imagegen/app.js` 里的 `SIZE_PAIRS` 表：每一行同时给出图像画布与同宽高比
的视频画布。视频尺寸都是 `H3V_CANVAS_MULTIPLE`(32) 的整倍数、落在 256..2560；图像尺寸落在
256..6272（`H3_MIN_SIZE`/`H3_MAX_SIZE`）。七档为 256×256→256×256（快速验证）、
1024×1024→1024×1024、1536×2048→768×1024、1024×1536→768×1152、2048×1536→1280×960、
2048×1152→1024×576、2560×1440→2560×1440（§102 的 2K 上限）。第一步选档会同步写入第二步
的视频尺寸并刷新配对提示；从生成的图进入第二步时会再次断言配对，避免手改视频尺寸后错位。
「快速验证档」一键把两步画布都设为 256×256，并把时长设为最短 4 秒、采样 4 步。

手动视频尺寸保留在单独的 `<optgroup>`（标注"扩展画布 · 不与步骤 1 配对"）里，大画布仍然可
选；选中时配对解除，提示文案会明确说明。服务端未改：两个图像接口接受范围内的任意 `WxH`，
`/v1/videos` 仍只有 Ref2VA。`make src/webui_embed.c && make cwick` 刷新内嵌 UI；`make test`
新增两条断言确认共享表与两个 optgroup 进了二进制，`node --check` 校验 JS。本轮未实跑生成：
服务驻留权重，起第二个实例会突破 80% / 禁 swap 预算。

## §113 图像与 realtime 切换对称释放 video 权重

一次图像编辑与视频任务混跑的日志证据：切到图像模型时 video 的 VAE 仍驻留，61.74 GiB 图像
DiT 叠在 10.26 GiB video 侧上，RSS 约 72 GiB。18:58:07 重新加载 video DiT 时打出
`could not wire 4.81 GiB ... (Resource temporarily unavailable)`，即 `mlock` 返回 EAGAIN；
采样第一步耗时 171.9 s，健康值为 60–67 s。`cwick_mem_lock_range` 只在首次失败时打日志，
所以 4.81 GiB 只是下界。它上面那句 "wire resident ... fits under budget" 报的是**尝试**字节数，
不是真正 pin 住的字节数，不能当成成功。

根因：`srv_switch_model` 与 `srv_switch_video` 都会释放另外三份权重，但 `srv_switch_image`
只释放 text 和 realtime，`srv_switch_realtime` 只释放 text 和 image，导致离开 video 后它仍驻留。
`swap_gate_open` 在 `vid_busy` 非零时已拒绝切换，所以拆除时没有 clip 在读这些映射。

两处切换现在都会在加载新权重前调用 `srv_release_video_model()`，并把 `H3V_SERVED_ID`
写进 "releasing" 日志。四条切换路径现已对称：各自释放另外三份权重。`make cwick` 编译干净、
`make test` 通过；因服务驻留权重，本轮未实跑 A/B。

## §114 无 ctts 的 B 帧导致 QuickTime 播放顺序错乱

一条 256×256 / 30 步视频（`video_1795432400_1`）在 QuickTime 里画面跳动、不稳定；ffmpeg 直接
抽帧同样跳，所以问题在编码/封装，不在播放器，也不在码率（256² 下 1.0 Mbps 充足）。对该 MP4
逐帧探测：56 B / 43 P / 3 I，显示时间戳非单调（0、3000、2000、4000、1000…），尾部两帧时间戳
重复，且容器没有 `ctts` box——封装器以为 `AllowFrameReordering=0` 就不会有 B 帧，把每个 sample
的 PTS 都写成等于 DTS。实测关闭输出重排并不能阻止 VideoToolbox 继续**编码** B 帧。

根治：把编码器固定到 Baseline（`kVTProfileLevel_H264_Baseline_AutoLevel`，avcC profile 0x42），
该 profile 不允许 B 帧；同时对每个 sample 用 `CMSampleBufferGetPresentationTimeStamp` 写出
`ctts` 表作为兜底（无 B 帧时 offset 全为 0，合法）。现在只给视频轨写 `ctts`。

`h3vmuxprobe` 60 帧 A/B：修复前 43 个 B 帧、PTS 乱序、仅 58 个唯一显示时间戳；修复后 0 个 B 帧、
60 个唯一 PTS、严格单调 0/1000/2000…。256×256/30 帧下：1 I + 29 P、30 个唯一且有序 PTS。
`make test` 新增断言校验 avcC 为 Baseline、ctts offset 全为 0。当前服务是用旧二进制启动的，
新视频需重启服务才会用上修复；已有的 MP4 无法靠重封装修好，因为 B 帧已在码流里。

## §115 stss 把全部帧标成 sync 导致 QuickTime 播放 P 帧出现重影

一条 256×256 / 30 步视频（`video_1795439347_1`）用 ffmpeg 解码画面正常，但在 QuickTime 里有重影。
§114 的 B 帧修复在该文件上已确认生效：Baseline（avcC 0x42）、3 I / 99 P、无 B 帧、102 个唯一且
单调的 PTS、ctts offset 全为 0。唯一剩下的差异是 sync 表：stss 把全部 102 个 sample 都列为随机
访问点，而实际只有 3 个 IDR。

根因：`avcc_first_nal_type()` 只看每个 AVCC sample 的第一个 NAL，但 VideoToolbox 在每个 IDR 访问
单元前加了一个 SEI（type 6），IDR slice（type 5）排在第二位。于是没有任何 sample 判定为 sync，
`put_stbl` 的兜底（`n ? n : n_samples`）又把全部 sample 标成 sync。QuickTime 信任 stss，在每个
声明的 sync 点重启解码；从没有参考帧的 P 帧起解就留下上一帧画面，形成重影。ffmpeg 忽略 stss，
所以抽帧看不出问题。

修复：`avcc_has_idr()` 扫描 sample 内所有 NAL 找 type 5；stss 兜底在没有 IDR 时只标首帧，绝不全标。
`h3vmuxprobe` 102 帧实测：sync 数 102 → 3，落在 sample 1/49/97（keyint=48 的 IDR），与码流一致。
`make test` 新增两条断言：stss 只列 IDR sample、不把 P 帧当随机访问点。与 §114 一样，当前服务早于
该二进制，新视频需重启才会用上；已有 MP4 无法只靠重封装修好（错误的 sync 表已写进文件，需要用修好的
muxer 从基本码流重新封装）。
