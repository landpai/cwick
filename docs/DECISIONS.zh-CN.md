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
