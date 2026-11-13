# Decode 性能：这 0.72 s 的一步花在哪

<p align="center"><a href="PERF.md">English</a> · <a href="PERF.zh-CN.md">简体中文</a></p>

这里的一切都测于参考机器 —— M3 Max，16 核（12 个性能核），128 GiB 统一内存 —— 用的是完整的
GLM-5.3-flash checkpoint（45 层，288 个 routed expert，top-8，磁盘上为 FP8-E4M3），`--backend metal`，
`--context 131072`。数字都是**一个 decode step**（单个 token）的，取自 `CWICK_PROFILE=1`，它会在每个
请求结束时打印一份阶段分解。

两个数字决定了一切：

- **完整的 routed-expert bank 是 290.32 GiB** —— 42 sparse layers x 288
  experts x 3 matrices x 8.0 MiB —— 无论计划怎么切，它都装不进 80% 的预算
  （102.40 GiB）；
- **一个 decode step 会碰到其中 8.06 GiB**（42 x 8 experts x 3 x 8 MiB）。
  这就是一步的全部成本：其余都只是舍入误差。

## 测量结果

| 条件 | ms/step | tok/s | stream | gemv | attn |
|---|---|---|---|---|---|
| Metal backend, experts resident (10-token reply) | **275** | **3.63** | 4.9 | 120 | 117 |
| Metal backend, 265-token prompt, 80 steps, 83 GiB cache (2026-10-20) | **590** | **1.69** | 271.6 | 151.1 | 134.4 |
| same + `CWICK_PREFETCH=1` (2026-10-20, QoS fix in) | **2111** | **0.47** | 490.8 | 712.3 | 284.2 |
| Metal backend, 265-token prompt, 79 steps, 83 GiB cache (2026-10-19) | 714 | 1.40 | 362 | 177 | 142 |
| Metal backend, same, 64 GiB cache | 798 | 1.25 | 462 | 208 | 151 |
| CPU backend (`--backend cpu`), cold | 1195 | 0.84 | 711 | 291 | 136 |
| Metal backend + `CWICK_PREFETCH=1` (2026-10-19, before QoS fix) | **1873** | **0.53** | — | — | — |

714 和 590 这两行是同一套配置在不同日期的测量：机器状态（温度、其他负载）会让一步移动约 20%。
把 590 当作当前诚实的基线，把 714 当作最坏情况。

从这张表里可以得出四件事。

1. **运行时能到 3.6 tok/s**：当一次回复需要的 expert 都在 RAM 里时，一步是 275 ms，而那时限制它
   的既不是算术、也不是 attention 或 HTTP 层——一旦请求不得不 stream，限制就来自存储路径。
2. **`--backend metal` 确实在干活。** expert GEMV 在 GPU 上从每步 291 ms 降到 120 ms（2.4x）。对于
   arena 和 mapping 支撑的 blob，batch 是 zero-copy 接受的，所以每个矩阵没有 8 MiB 的 memcpy。
3. **prefetch 是个陷阱，QoS 修复也没能救它。** 上面那个 80 步的 A/B 就是判决：`CWICK_PREFETCH=1`
   在 `QOS_CLASS_BACKGROUND` 改动前从 714 -> 1873 ms，改动后从 590 -> 2111 ms。prefetch 那一路读的
   字节是 2.7 倍（647.6 对 236.8 GiB，95396 对 30315 次读），命中率却*零*增益（68.6% 对 68.5%），
   它 demand 侧的 pread/dispatch 计数归零，MoE 阶段因缓存/arena 抖动而爆炸（151 -> 712 ms）。它
   保持 opt-in，默认关闭。

   原因从来都是核，而不是字节：demand 路径用十二个 worker 的 fork-join 读整个层（`moe.c:348-367`），
   而 fork-join 的 GEMV 由最慢的参与者定速，默认 QoS 下的 prefetch 线程会在 kernel 中途抢占这些
   worker —— 于是这一层比根本不发那次读还慢。demand 侧的读者没法再往上提（它们是 fork-join 里的
   pool worker，而那些 GEMV 本来就跑在 `USER_INITIATED`），所以 `pf_reader` 改为把*投机*线程降到
   `QOS_CLASS_BACKGROUND`（`stream.c:1110-1129`）—— 那是调度器唯一不许它抢占 `USER_INITIATED`
   工作的类。
   A/B 说明机制不是问题；*投机字节*才是。

## 为什么 streaming 要花 362 ms 而不是 20 ms

在 83 GiB 时，缓存装下 36 288 个 blob 里的 ~10 600 个（29%）。Zipf 式的路由救回了其余的大多数：
实测命中覆盖了约 76% 的读，于是一步 stream 约 1.9 GiB。1.9 GiB / 362 ms = **5.4 GiB/s**，这正是这块
SSD 对 8 MiB 随机区间的持续读速率。这个阶段处在硬件极限上，不是软件极限：

- 把 12 个 reader 线程加总，`pread` 是 1572 ms/step；除以 12 是 131 ms，但 fork-join 的墙钟是
  359 ms。这个差距是不均衡 —— 每一波里有一次慢读就定了墙钟 —— 而不是线程空闲。
- slot 预留是 0.7 ms/step，缓存探测是 0.01 ms/step：缓存本身不是问题。
- 操作系统 page cache 帮不上忙：expert 的 fd 是以 `F_NOCACHE` 打开的，这样 stream 出来的字节
  永远不会把 wired 的 expert arena 挤出去。

所以存储路径是一堵带宽墙。把缓存从 64 提到 83 GiB 只买到 10%（798 -> 714 ms），还把占用推到
102.40 GiB 天花板下的 102.04，这就是默认值留在 64 GiB 的原因。

## 714 ms 那一行是 *warm* 数字 —— 短回复要付 cold 的税

上面那次 79 步的运行有时间把缓存焐热：到它结束时命中率约 76%，每步只 stream 了约 1.9 GiB。
2026-10-20 用同样的 265-token prompt、同样的 83 GiB 缓存、同样的 backend 测的一次 10-token 回复：

| | 10 steps（cold） | 79 steps（warm） |
|---|---|---|
| total | **979 ms/step** | 714 ms/step |
| stream | 614.5 | 362 |
| gemv | 208.7 | 177 |
| attn | 155.5 | 142 |
| hit rate | **17.3%** (4416/21153, prefill included) | ~76% |
| streamed | 165.26 GiB over the whole run | ~1.9 GiB/step |

对 server 有两个后果：

1. **重启之后最初的几个请求，每步要慢约 37%**，比稳态更慢，因为 expert 缓存随着请求路由而填充。
   汇总行里请求级的 `cache=` 字段就是要盯的数字。
2. cold 的 step 还要付一次性的 `alloc` 成本，warm 的 profile 从来不会显示它：`shmem_alloc` 是
   189 ms/step（arena 增长了 21 次，`arena hot LIFO: 0 hits / 4656 misses`）。一旦 arena 到了
   尺寸，它就消失了。

2026-10-20 落地的缓解措施：`--warmup N`（别名 `CWICK_WARMUP=N`）会在 port 打开*之前*，用一个一次性的
prompt 跑一次静默的 N-token 生成（`main.c` 里的 `serve + serve_warmup`）。它走的是 CLI 生成路径，
所以从不碰 prefix cache，失败会被记录并忽略。生成的文本被丢弃，不做 decode。

2026-10-20 实测（`tools/warm-ab.sh`，253-token prompt，生成 10 个 token，同一台机器，83 GiB 缓存；
两臂输出的文本完全相同，所以缓存只改变速度）：

| 首次请求 | decode ms/step | decode tok/s | 命中率 | prefill 墙钟 |
|---|---|---|---|---|
| cold serve | 1304 | 0.77 | 13.2% | 96.9 s |
| `--warmup 32` | **939 (-28%)** | **1.07** | **39.6%** | 98.9 s |

warmup 本身为 32 个 token 花了 74.8 s（1113 ms/step）—— 那*正是*被从第一个客户端请求上挪走的 cold
税。注意一个 253-token 的请求由 prefill 主导（端到端 108 s 里占 97 s），所以收益体现在 decode 阶段；
短 prompt 以及 prefix-cache 命中的 prefill 会把天平推向 decode，而那里正是 warmup 帮助最大的地方。
32 个 token 时命中率只到 39.5%；更长的 warmup（64-128）买到更多缓存，但按比例付出更多前端成本，
而且一个 prompt 激活的 expert 集合本来就随 prompt 而变 —— 稳态的耐久解法是命中率治理（见下文）。

## 为什么 prefill 在 CPU 上跑 MoE

profile 里的 `batch windows: metal 0, ->cpu 190` 不是 bug。prefill 的 window 会取到 malloc 出来的
staging blob 里（arena 曾经把一次 43 s 的 prefill 变成一次永远跑不完的 prefill，见 `moe.c` 里的
`expert_view_into`），而 Metal batch 只接受 no-copy 的 arena/cache 视图，所以 prefill 的 group 按设计
走 CPU 的 fork-join 路径。强行改走 Metal（`CWICK_PREFILL_METAL=1`）也测过：batch 赢下 23% 的计算，
但让 SSD 慢了一倍左右，总体上反而亏（`tensor.c:611-614`）。Decode 不受影响 —— 它的视图按构造就是
arena 支撑的（`CWICK_GROUP_METAL`，默认开启）。

## 在 streaming 下究竟怎样才能到 3 tok/s

**MXFP4 的 expert pack 已经不在桌上**（2026-10-20 决定）：把 expert 字节减半会让 stream 从 271.6
降到 ~140、gemv 从 151 降到 ~75 ms，也就是下面那一行投影出来的 413 ms，但它要付一次 290 GiB 的
重新打包，以及不再是 bit-exact 的 4-bit 权重。打包契约在那个决定之前就已经推出来了，见
`docs/DECISIONS.zh-CN.md` §2，以备日后重提。

在不碰字节的前提下，剩下的杠杆如下，每一条都是已测或可测的，而且没有一条是有损的：

| 手段 | 预期 | 状态 |
|---|---|---|
| KDA fa/b/ga projections into the q/k/v GPU batch | -2.3 ms attn (moves work, does not remove it) | **done, measured** |
| o_proj per layer onto the GPU (batching across the recurrence boundary) | -20..-25 ms | blocked: it depends on the recurrence output, so it cannot share the q/k/v batch; a solo command buffer costs as much as the CPU pool |
| expert hit rate 68.5% -> 82% (cache grooming, evict tuning) | -115 ms stream | next |
| `--warmup N` pre-generation at serve start (default off) | first request -37% | **implemented 2026-10-20**, measured with `tools/warm-ab.sh` |
| GPU kernel clock/occupancy work | unknown | gemv 151 ms = 8.06 GiB/step = 45 GB/s, 8x under the unified-memory ceiling — the largest unmeasured candidate |

工作顺序，按实测回报排：

1. **命中率治理**（缓存顶到 80% 天花板，evict 策略）：-115 ms。
2. **`--warmup`**：把 early request 上的 cold 税去掉。
3. **GPU kernel profiling**：gemv 阶段离带宽天花板还差 8 倍；与这里其他一切不同，这点余量是算术，
   不是字节。

这里没有任何一项剪枝模型、跳过 expert 或减少 top-k：router 仍然每个 token 从 288 个里挑 8 个，而且
每一个都会被应用。不用 MXFP4 的诚实天花板，从 590 ms 基线算起：fusion + 命中率 + warmup 加起来
落在稳态约 2.2 tok/s；3.0 tok/s 还需要 GPU kernel 的工作也兑现出来。

## 复现

```sh
CWICK_PROFILE=1 ./cwick serve --root ./zai-org/GLM-5.3-Flash --backend metal \
    --context 131072 --verbose --trace
# then, at the end of every request, the phases above are printed once
```

生产环境里要盯的是请求级的汇总行：

```
prompt=265 gen=80 prefill=…/…tok/s decode=…ms/79 steps X tok/s cache=63.5%(…) streamed=…
```

`cache=` 是 expert 命中率；`streamed=` 是 SSD 实际吐出的量。当 `X` 掉到 1 以下时，第一个该看的
数字就是 `cache=`。

## 2026-10-25：GPU batch 不再串行之后的那一步

同一台机器，同一个 checkpoint，同样的 `--backend metal --context 131072`，26-token prompt，60 个
decode token，`CWICK_PROFILE=1`。上面 590 ms 那一行来自 2026-10-20；这是今天的构建，在一个改动
前后各测一次。

| | ms/step | tok/s | stream | gemv | attn | rest |
|---|---|---|---|---|---|---|
| before | 727.8 | 1.374 | 390.2 | 170.6 | 132.7 | 34.3 |
| after | **663.0** | **1.508** | 379.3 | **126.4** | 124.2 | 33.2 |

改动是一个函数，`metal_objc.m` 里的 `cwick_concurrent_encoder`。一个 `MTLCommandBuffer` 内的
compute dispatch **默认是串行的** —— dispatch *i+1* 要等 *i* 退休才开始 —— 而这里两个 GPU batch
都由彼此独立的 GEMV 组成，各自写不相交的输出 slab，所以默认行为恰好抵消了 batch 本应创造的重叠。
`computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent` 修好了它。生成的文本与串行构建
逐字节相同（同样的 md5），这是预期结果：dispatch 顺序从来就不属于算术的一部分。

同一构建的第三次运行测得 682.4 ms/step（gemv 108.4，stream 409.2），所以上面「after」那一行应该读成
1.47-1.51 tok/s 而不是一个常数：在这块盘上 `stream` 阶段会跑与跑之间移动 8%，`gemv` 的稳定度也
差不多。正因如此，下面的一切都是同一次运行内的比较。

同一次运行走 server（`sh ./start.sh glm-5.3-flash`，streaming，生成 200 个 token，三个请求）：
decode **1.517 / 1.556 / 1.507 tok/s**，端到端 1.31 tok/s。所以 CLI 的数字原样带到了 API 路径上。

### GPU 究竟受限于什么

`CWICK_METAL_BATCH_TIME=1` 把每一波拆成 encode / commit+wait / GPU 时间。一个 decode 层是两波，
18 个 dispatch（八个 expert 的 gate+up，shared expert 的 gate+up）和 9 个（八个 down，一个 shared
down）：

| kernel | GPU 时间，wave 1 | bytes | rate |
|---|---|---|---|
| v3 (threadgroup LUT), serial dispatch | 2.07 ms/layer | 192 MiB | 45 GB/s |
| v3, concurrent dispatch | 1.27 ms | 144 MiB | **118 GB/s** |
| v3d (same loads, `int`→`float` instead of a LUT) | 0.67 ms | 144 MiB | **227 GB/s** |

第三行才是有意思的，而且很容易读过头。`v3d` 是 `v3` 的诊断孪生 —— 同样的 load，同样的累加器形状，
唯一的区别是它不读任何表 —— 而它快 **1.9x**。诱人的结论是：256 项的 `threadgroup` LUT 要付 1.9x 的
代价，应该用算术替换掉。这个结论是错的，下一节就是说明为什么的实验：`v3d` 把 decode 换成了*什么都不
做*，所以它的 0.67 ms 只是 load 本身的价格，不是任何正确 kernel 能到达的目标。真正的 decode —— 无论
算术还是查表 —— 才是把 kernel 按在 ~110 GB/s 上的东西。

### 解码字节才是地板，不是查表

剩下这点差距最明显的嫌疑是 LUT 的 bank conflict：32 个 lane 用 32 个互不相关的字节去索引一张 256 项
的表。于是 `v4` 保留 v3 的切分（8 个输出行 x 8 个 k-slice，256 个 threadgroup），把表换成
`cwick_fp8_v2` —— 一个由移位、一次加法和一次选择构成的、bit-exact 无分支的 E4M3 decode —— 并把切分
形状延伸到 `k = 2048`，让 down projection 也能用上，而 v3 做不到这点。`CWICK_FP8_KERNEL=v3|v4`，
同样的 prompt，各 39 个 decode step：

| wave | dispatches | bytes | v3（LUT） | v4（算术） |
|---|---|---|---|---|
| gate + up | 18 | 144 MiB | **1.396 ms** = 103 GB/s | 1.657 ms = 87 GB/s |
| down | 9 | 72 MiB | **0.799 ms** = 90 GB/s | 0.851 ms = 85 GB/s |

`gemv` 从 119.8 走到 151.4 ms/step；v4 在两波上都更慢。一个 bit-exact 的 decode 大约每字节 13 个整数
运算，而在这块 GPU 上，一次带约 4 路 bank conflict 的 shared-memory gather 就是比它便宜。所以：

- ~110 GB/s 就是一个*正确*的 fp8 GEMV 在这里落地的地方，而且那是 decode 界的限制，不是搬运界的
  限制；
- 227 GB/s 是没有任何东西去解码这些字节时，搬运能做到的事；
- 因此那 1.9x **并不**可得，这一项被从下面的工作清单里删掉了，而不是留成一个邀请。

`v4` 仍然在环境变量后面被编译和测试，这样复现这个比较只需要一条命令，而不是一次重写。这个负结果
就是产出物。

### 为什么 expert 缓存已经没有余量了

最明显的杠杆是命中率：一个 decode step 会碰到 42 layers x 8 experts x 3 matrices x 8 MiB =
**8.06 GiB**，在 50% 时，命中率之上的每一个字节都是 SSD 的字节。`CWICK_EXPERT_DUMP` 会写下路由
trace，重放它就能得到任何策略都在对抗的那条界 —— 一个持有每层*最常被路由*的 expert 的缓存的命中率，
也就是一个完美的频率策略收敛到的上限：

| experts/layer | 缓存 | oracle 命中率 | SSD GiB/step |
|---|---|---|---|
| 16 | 15.8 GiB | 48.1% | 4.18 |
| 32 | 31.5 GiB | 65.3% | 2.80 |
| 64 | 63.0 GiB | 82.8% | 1.38 |
| 128 | 126.0 GiB | 96.3% | 0.30 |

对照这条界，重放同一条 trace 的在线 power-of-K LFU 达到 **31.5 GiB 时 58.4%、85 GiB 时 74.1%**；
把 victim 采样从 32 提到 1024 个 slot 只让它移动 0.2 个百分点，所以与 oracle 之间的差距是*学习*这个
分布的成本，不是挑 victim 的成本。构建实测 50.2%，它与 58.4% 之间的 8 个百分点，是 prefill 的一次性
expert，加上开放寻址布局留空的那半张表。

**是测出来的，不是假设的：** 缓存大小本身是一场平局。同样的 prompt，40 个 step，
`CWICK_EXPERT_CACHE_GIB` 扫描：

| 缓存 | 墙钟 | stream | gemv | alloc（填充） | 命中率 | SSD/step |
|---|---|---|---|---|---|---|
| 32 GiB | 727.8 | 390.2 | 170.6 | 36.5 | 50.4% | 3.29 GiB |
| 64 GiB | 774.5 | 413.4 | 191.0 | 88.6 | 56.5% | 2.37 GiB |
| 85 GiB | 786.4 | 424.1 | 197.4 | 107.3 | 57.3% | 2.29 GiB |

字节少了 30%，阶段反而*更慢*：在途的读变少之后，这块盘只交付 7.3 GiB/s 而不是 9.3，把命中率买到的
东西又还了回去。`alloc` 那一列是填充 N 个 slot 的一次性成本（32 GiB 是 4095 个，85 GiB 是 10675
个，每个约 0.52 ms），也是为什么一次短运行会显得小缓存更好看。把它减掉，三者彼此相距在 13 ms 之内。
（`alloc` 在 profile 里已经在 `stream` 之内 —— 它报告为 "of stream: slot reservation" —— 所以不能
在它之上再加一次。）

### `attn` 由什么构成，以及为什么它不会缩小

各阶段相加为 `total = stream + gemv + attn + rest`，这是做投影时要守住的不变量：上面「before」那一行
是 390.2 + 170.6 + 132.7 + 34.3 = 727.8。`attn` 占一步的 17-18%，而且**不**属于 fp8 expert 的工作，
所以 LUT 消失时它不会移动。同一台机器，12-token prompt，40 个 step，拆成六部分：

| 部分 | ms/step | 占比 | 说明 |
|---|---|---|---|
| kda_proj | 57.5 | 43% | q/k/v of the 34 KDA layers, already one bf16 GPU batch |
| mla_all | 32.5 | 24% | the 11 sparse-attention layers, end to end |
| kda_tail | 24.5 | 18% | gated RMSNorm + o_proj, another bf16 GPU batch |
| kda_rec | 9.4 | 7% | the KDA recurrence itself, CPU |
| kda_gate | 4.5 | 3% | f_a/f_b/b, decay and beta |
| kda_conv | 3.3 | 2% | short conv + sigmoid + L2 |

那次运行测到的是 cold 的 46.5% 命中率和 768.5 ms/step，所以请读*占比*，不要读绝对毫秒；乘 0.86 去和
663 ms 那一行比较。这六部分相加是 131.6，而 profile 报告的是 133.3 ms（最后的 1.7 ms 是 timer 开销，
摊在 45 层上）。要点在构成：`kda_proj` + `kda_tail` 是 82 ms 的 bf16 GEMV，走的是 fp8 expert 用的
同一个 batch，而且完全没有 decode 的情况下就跑在 ~125 GB/s —— 它们是字节变宽的一次搬移，不是查表
也不是位重排。所以 ~110-125 GB/s 就是这块 GPU 在接近理想的 GEMV 供给下能做到的，而 `attn` 不会跟着
fp8 kernel 上发生的任何事一起缩小。

### 常驻一步的算术

看清这堵墙最干净的方式，是问：把 SSD 完全从一步里拿掉之后，一步要花多少 —— 一个大到装得下全部
290.32 GiB 的 expert 缓存，而这台机器没有任何一部分做得到。那时每个 expert 字节都来自统一内存，
`stream` 归零，一步就是两个 GPU 阶段加上胶水：

    126 (gemv) + 124 (attn) + 33 (rest) = 283 ms/step = 3.5 tok/s

那就是**无限内存**下的地板。所以这个速率不是内存容量问题、不是缓存策略问题、也不是调参问题：在这块
GPU 上、用这个 decode，光是计算就定了它，加多少 RAM 都改变不了这一点。

唯一有可能移动那块地板的实验是 fp8 LUT，它跑过了，而且输了 —— `gemv` 停在 126，不是 66。假如它成功
了，常驻的一步会是 223 ms（4.5 tok/s），SSD 只需要这一步里的 27 ms，也就是 0.25 GiB/step，也就是 97%
的命中率 = 在 128 GB 的机器上 pinned 281 GiB。两条分支最后落在同一处——每步字节数定速率——只是其中
一条更近。

每个 decode step，模型要搬约 18.5 GiB 的权重：8.06 GiB 的 routed expert，加上约 10.4 GiB 的
KDA/MLA projection 和 shared expert，后者按定义就是常驻的。一步 250 ms 意味着以合计 74 GB/s 的
速度搬运它，而 SSD 只有约 9.3 GiB/s、统一内存约 227 GB/s：

| 变体 | SSD/step | stream | gemv | attn | rest | total | tok/s |
|---|---|---|---|---|---|---|---|
| before the concurrent encoder (measured) | 3.29 | 390 | 171 | 133 | 34 | 728 | 1.37 |
| today, 32 GiB (measured) | 3.29 | 379 | 126 | 124 | 33 | 663 | 1.51 |
| 85 GiB cache, 57% hit (measured, pre-concurrency build) | 2.29 | 424 | 197 | 132 | 33 | 786 | 1.27 |
| 85 GiB cache, concurrency + a gemv that does not grow | 2.29 | 424 | 126 | 132 | 33 | 715 | 1.40 |
| 88% hit, ~87 GiB pinned (does not fit, projected) | 0.97 | ~150 | 126 | 132 | 33 | 441 | 2.27 |
| every expert resident (impossible; the floor above) | 0 | 0 | 126 | 124 | 33 | 283 | 3.5 |

第三行是那个意外，而且是实测的：把缓存提到 85 GiB 买到少 30% 的字节，却*输掉*墙钟。`stream` 从 379
走到 424，因为队列变浅了，而 `gemv` 从 171 走到 197 是**在同一次扫描之内**发生的，因为 85 GiB 被
pinned 的 expert 页挤压了 GPU 的工作集。第四行把一切好处都让给了那一行 —— 并发，以及一个完全不增长
的 `gemv` —— 它也只是和今天打平。扩大缓存在这里不是杠杆；它是一次内存压力的实验。

第五行是一个装不下的缓存能做到的最好情况：2.27 tok/s。所以诚实的分布是：今天 1.5，约 2.3 来自一个
会把 KV cache、arena 和 OS 都挤到没有余地的缓存，3.5 则要 expert bank 全部常驻 —— 而那正是这台机器
之所以以 streaming 设计存在的全部理由。

### MTP 实际买到了什么，实测

投机的想法很诱人，而算术杀掉了它的大部分。一个 2-token 的 batch 是在两个 token 路由的**并集**上跑
MoE，而不是把 8 个 expert 跑两遍，所以「读一次权重、出两个 token」这个说法只有在并集接近 8 时才成立。
它不是。重放 79 个 step 上的 `CWICK_EXPERT_DUMP`，3276 个（layer, step-pair）样本：

| | experts |
|---|---|
| one token | 8.00 |
| union of two adjacent tokens | **13.03** |
| expert bytes per token, 2-token batch vs 1-token | **0.815x** |

所以第二个 token 花的是一个 step expert 字节的 0.63x，而不是 1.0x。这是真实的节省，而且是 18.5% 的
节省，不是 50%。这就定住了天花板：

    step(1 token)  = 663 ms
    step(2 tokens) ~ stream 379*1.63 + gemv 126*1.63 + attn 136 + rest 40
                   ~ 999 ms
    oracle drafter (accept 1.0):  999/2     = 500 ms/token  -> 2.0 tok/s (+33%)
    measured acceptance 0.586:    999/1.586 = 630 ms/token  -> 1.59 tok/s (+5%)

**即使有完美的 drafter，+33% 也是这个模型上的天花板**，因为约束它的是 expert 并集，而不是 drafter。
盈亏平衡点是 0.507 的接受率，而它离 drafter 实际达到的水平近得让人不舒服。

drafter 本身是真的。GLM-5.3-Flash 带 `num_nextn_predict_layers=1`，作为完整的第四十六个 block 放在
`layers.45.*` —— 完整得有点不祥：它有自己的 288-expert bank、自己的 MLA attention、自己的 DSA
indexer —— 外加三个让它成为 MTP head 的张量（`enorm`、`hnorm`、`eh_proj`）和 `shared_head.norm`。
驱动它就是一次 fusion 加一个 DSA block：

    h'    = eh_proj([ enorm(emb(t+1)) , hnorm(h_t) ])
    draft = lm_head( shared_head.norm( DSA-block(h') ) )

拿模型真正接下来吐出的 token 去评分，贪心 decode，30 个 token：**17/29 接受 = 58.6%**。拼接顺序在
checkpoint 里任何地方都没有文档；`CWICK_MTP_SWAP=1` 把它反过来，测得 **0/29**，这才让 58.6% 成为一个
结果而不是巧合 —— 这个 block 显然是被按它训练时的方式驱动的。

在这之前，还差两样东西，才能让这一切变成 token：

- **verify pass。** `forward_prefill_mhc` 只返回最后一个位置的 logits，并且无条件推进 `pos`。投机
  需要两个位置（位置 1 决定 draft，位置 2 是白送的 token）和一个由调用方提交的 `pos`。那是第二个
  forward 入口点，不是一个 flag。
- **一次 KDA rewind。** 34 个 linear-attention 层把序列折进一个 136 MiB 的 DeltaNet 累加器，外加一个
  9 MiB 的 conv window。两者都无法解开 —— conv window 已经丢掉了它需要的输入 —— 所以被拒绝的 draft
  要求在 verify 之前快照 ~145 MiB，并在 miss 时恢复。那是 999 ms 的一步上的 ~3 ms，所以付得起；
  它只是不免费，而且是唯一一处错误会静默损坏输出、而不是把它变慢的地方。

考虑到实测接受率下只有 +5%、以及需要一个近乎完美 drafter 的 +33% 天花板，这一项只有在 drafter 能被
推过 ~0.75 时才值得做完。两个候选方案是：把 MTP block 的 KV 在 prompt 上回填（它在那里目前是对零做
attention —— 上下文是 h_t 携带的，这也是它能工作的原因），以及从请求自己的分布里采样 draft，而不是
取它的 argmax。

从这里出发，诚实的工作顺序是：

1. **把 expert 的取数与这一层自己的 GEMV 重叠起来。** 今天它们是严格串行的（取数 7.5 ms，然后 3.0 ms
   的 GPU）。把取数拆成 gate+up 再 down，并在 down 的读之前提交第一波，大约能藏掉三分之一的读时间，
   约 -55 ms。
2. **把第二个 token 拼到同一次权重读上。** 已测，而且比听起来的小 —— 见下面「MTP 实际买到了什么」。
   drafter 是能工作的（贪心下 58.6% 接受），但一个 2-token batch 要付 1.63x 的 expert 字节，所以
   实际收益约 5%，而即使有 oracle drafter，天花板是 +33%。verify 循环没有实现；draft 及其评分有
   （`CWICK_MTP=1`）。

这里曾经有第三项 —— 「不带 LUT gather 的 fp8 decode」，价格是按一次 v3d 测量估的 ~-60 ms，而那次
测量跑的是完全没有 decode 的同一波。它被实现为 `v4`，而且它输了；那一节就是记录。值得留下的教训是：
一个把工作完全去掉的诊断孪生给出的是*上界*，而上界不是计划。

这两项都不会剪掉一个 expert、跳过一层或收窄一个权重：router 仍然从 288 里挑 8 个，而每一个的三个矩阵
都会被应用。

## 2026-10-26：prefix cache 命中 0%，原因是一个 token

一次真实的 `serve` 会话，从它的 access log 和 `/v1/metrics` 审计：四个 chat 请求，一段连续的对话，
以及

```
vllm:prefix_cache_queries_total       4
vllm:prefix_cache_hits_total          0     → 0.0%
cwick:prefix_tokens_reused_total      0     → 0 of 751 prompt tokens
```

日志里没有一行 `prefix cache hit`，也没有一行 `load failed` —— 查询本身就返回 0。缓存在其他每个轴
上都是健康的（`pfxcache/` 在 80 GiB 预算下持有 8 个快照 / 149 MiB，所以什么都没被驱逐），而它没能
跳过的 prefill 花了这次会话 778 s 里的 260 s。

### 那处 off-by-one

第 N 轮的 prompt 以 `履衅` 结尾：server 把它追加在后面，好让模型开始推理。同一会话的第 N+1 轮不会
重新发送它 —— 客户端把 assistant 的那一轮渲染成 `履畔` 加上它的内容，于是两个 prompt 恰好在这最后
一个 token 上分叉。

把两份渲染后的 prompt 从 trace 里取出来逐字节比较测得：第 N 轮是 674 个字符 / 262 个 token，第 N+1
轮是 973 / 405，而它们在前 667 个字符上一致。分叉落在 `履衅` 这个 token *内部*。所以连续两轮真正
共享的前缀是

```
turn N's prompt            = 262 tokens
what they really share      = 261 tokens
snapshots the cache held   = {262, 551}
```

查询是精确长度的（`pfx.c:383` 把 `e->pos` 与采样位置比较），所以一条长了一个 token 的条目和一条都
没有一样没用。每一次其他转换上也发生了同样的事：33→51 需要 32，却什么都没留；51→262 需要 50，却
留着 704。

还有两件事让更短的条目不存在：

* `CWICK_PFX_MIN_TOKENS` 是 256，于是一个会话最初的几轮根本不会被写入。
* prefill 中途的 checkpoint 只在 prompt 被分块之后才触发，即 `prompt_len >= 2 * pfx_block`（默认
  4096 时是 8192 个 token）。这次会话里每个 prompt 都在 500 以下，所以没有中间条目可以退而求其次。

### 修复

在全长快照旁边，再抓一份少一个 token 的孪生快照，用同样的 token 序列减去最后一个来做键。
`cwick_engine_capture_prefix` 本来就接受任何 `<= eng->pos` 的位置，所以这只是多一次 state pack
（这里是一次 ~11 MiB 的拷贝），而不是第二次 prefill —— 它和整段 prompt 命中路径在 load 时已经做的
那个一 token 回滚是同一件事。`CWICK_PFX_MIN_TOKENS` 降到 32：在这个 checkpoint 上，一轮 33 个 token
要花 19 s 的 prefill（expert 并集的读占主导，所以成本几乎不取决于 token 数），而它的快照不到
一个兆字节。

### A/B，两臂用同样的请求

三轮，`reasoning_effort=High`，`max_tokens=48`，历史里带着一个固定的 assistant 轮次，好让两臂发出
逐字节相同的 body：

| 轮次 | prompt | replayed | prefill，缓存关 | prefill，缓存开 |
|---|---|---|---|---|
| 1 | 33 | — | 19.083 s | 19.354 s (cold) |
| 2 | 42 | 32 | 15.210 s | **4.407 s** |
| 3 | 50 | 41 | 19.279 s | **3.522 s** |
| | | | 53.572 s | **27.283 s** |

命中率 2/3 = 66.7%，125 个 prompt token 里复用了 73 个。快照 load 约 0.1 s（命中落在请求到达后
102 ms 和 117 ms，然后剩下的 9–10 个 token 走 prefill —— 残余的地板就是剩下的那部分，不是缓存）。

正确性：三次响应在两臂之间逐字节相同，只有 `id`/`created` 时间戳和命中现在会报告的
`usage.prompt_tokens_details.cached_tokens` 字段不同。`content`、`reasoning_content` 和
`finish_reason` 全都一致，这才是真正的测试 —— 一个差了一个位置的 replay 状态会立刻表现为不同的
文本。`make test` PASS，layer oracle 是 204 checks / 0 failure / worst rel 0。

剩下的 miss 是第 1 轮，而这是正确的：那时还没有任何东西可以 replay。

## 2026-10-26（稍后）：thinking budget 在 DeepSeek 上从未被执行

对同一个 `serve` 会话的审计（log + `/v1/metrics`）发现三个问题。其中一个在烧机器：一个请求在配置的
8192 天花板下生成了 **16,568 个 thinking token**，已经跑了 2 h 35 m，按它退化后的速率还要约 30 h。
本节讲的就是这一个。

### 问题在哪

对于任何 chat 请求，只要 tokenizer 把两个标记都给出为单个 token，`server.c` 就无条件地武装这个
天花板，并把它记进日志：

```
[req 179] thinking budget 8192 tokens (level=高, block already open)
```

执行逻辑住在 `engine.c` 的 decode 循环里（`eng->think_budget` 在 3708 被读，close tag 在 3775-3822
被写入）。DeepSeek 不走那个循环：`ds_engine.c` 跑自己的循环，而在那个文件里搜 `think` 只会返回两条
关于*进度标签*的注释，别的什么都没有。于是 budget 被设在 engine 上、被报告为已武装，却从未被真正
decode 的代码读取。

loop guard 也没能救它 —— `finished_reason=loop` 整段运行都是 0，因为模型并不在重复吸引子里。它是
真的还在思考，而这正是 loop 检测器唯一看不见的故障模式。

### 修复

把这个天花板推进到那个有自己循环的 backend 里，就像 `ctx.stop`、`ctx.tag`、`ctx.spec` 和
`decode_phase` 已经做的那样：

- `ds_engine.h`：`ctx` 增加 `think_budget` / `think_open_id` / `think_close_id`
  / `think_open_at_start`。
- `engine.c` 的 `cwick_engine_set_think_budget` 把四个全部拷进 `eng->ds->ctx`，包括在解除武装的
  路径上。
- `ds_engine.c` 的 decode 循环拿到与 GLM 循环相同的 L4 block：clamp 到
  `max_new - CWICK_THINK_RESERVE`，问一下纯函数 `cwick_think_plan`，并在动作为非 NONE 时把标记写进
  输出，*并且*在同一步里让它们流过 `ds_model_forward`，这样刚刚写下自己定界符的模型会去回答，而不是
  继续思考。`think_open` / `think_closed` 的锁存与 GLM 侧完全一致，结果现在会报告
  `think_budget_hit` / `think_budget_applied`。

### 第一版引入的一个 bug，以及抓住它的对照运行

`eng->ds` 来自 `cwick_xmalloc`，**没有 memset**，所以 `ctx.think_budget` 是未初始化的。一次不设
`CWICK_THINK_BUDGET` 的对照运行打印出 `thinking budget: 0 tokens reached` —— 一个被清零的结构体被
读成「在第一个 token 之前就关掉 think block」，而这恰恰是 `cwick_engine_new` 自己的注释所警告的
陷阱（「0 是合法的 budget，所以禁用状态必须是一个显式的 -1」）。`ds_engine_open` 现在把四个字段都
初始化为禁用状态。值得注意的是，发现它的是对照运行：已武装的那一例看起来早就正确了。

### 验证，真实 checkpoint，`--backend metal`

| 场景 | 预期 | 实测 |
|---|---|---|
| no `CWICK_THINK_BUDGET` | silent | no budget line at all; `gen=60 finish=length` (the old behaviour) |
| `CWICK_THINK_BUDGET=32` | fire at 32 | `thinking budget: 32 tokens reached`, then `gen=60 finish=stop` |
| `CWICK_THINK_BUDGET=0` | close before token 1 | `0 tokens reached`, `gen=40 finish=length` |
| `CWICK_THINK_BUDGET=200`, `--max-new 40` | clamp to 20 | `200 tokens reached` is not what appears; `20 tokens reached` is |
| server, `reasoning_effort=High`, `max_tokens=48` | clamp 8192 to 24, fire, report | `8192 clamped to 24: 48 tokens of output leave no room to answer`, then `think=25 content=23 think_cap=24(hit)` |

`make test` PASS（现有的 `test_think_budget` 覆盖 GLM 路径；DS 路径没有合成 fixture，所以由上面这些
运行覆盖，而 `cli_arm_think_budget` 存在的意义正是如此 —— 「第二个 server 会需要另一份权重拷贝」）。

运维可见的回报是 `think_cap=` 现在是一个真实的数字。在此之前，`res.think_budget_applied` 对每个
DeepSeek 请求都是 0，于是 access log 打印 `think_cap=0` —— 与「没有请求 thinking」无法区分 —— 而
请求实际上是在一个未执行的 8192 下运行。

## 2026-11-07：两个接口的全链路，一个模型一个进程

同一份常驻权重有两种用法，这次背靠背各测一遍：对话（`POST /v1/chat/completions`）与决策
（`POST /v1/systemone`）。目的不是给模型排名，而是看清各自的墙钟时间花在哪一段。

**方法。** 每个 checkpoint 都在**自己全新的进程**里启动——这张表里没有任何一次
`POST /v1/models/switch`，所以没有一个数字被上一个模型的映射或缓存污染。前缀缓存关闭（每次
prefill 都是冷的、可比的），`--backend metal`，三个模型统一 `--context 32768`（同样的 KV
预算；`start.sh` 的 500K 默认会给 GLM/DS 另一套内存计划），贪心采样，每个组合一次请求。
对话提示词就是决策里那段 `state` 加上"Summarize the paragraph."；决策是同一段 state 加一道
8 选 1。时间取自服务端自己的直方图（`vllm:time_to_first_token_seconds`、
`request_prefill_time_seconds`、`request_decode_time_seconds`），在请求前后从 `/metrics`
读差值——也就是运维实际能看到的数。

### 对话：prefill 加每输出一个 token 一步

| checkpoint | 装载 | prompt token | 首 token | prefill | 输出 | decode | decode | 端到端 | footprint |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniMax-M3.1-Flash | 18.3 s | 261 | 55.90 s | 4.67 tok/s | 48 | 31.41 s | 1.53 tok/s | 87.31 s | 22.13 GiB |
| DeepSeek-V4.1-Flash-1017 | 16.3 s | 108 | 50.31 s | 2.15 tok/s | 32 | 14.58 s | 2.19 tok/s | 64.89 s | 84.16 GiB |
| GLM-5.3-flash | 22.4 s | 93 | 56.09 s | 1.66 tok/s | 48 | 38.28 s | 1.25 tok/s | 94.38 s | 49.98 GiB |

### 决策：一次 prefill，没有 decode

| checkpoint | 输入 token | 墙钟 | prefill | footprint | 答案 |
|---|---:|---:|---:|---:|---|
| MiniMax-M3.1-Flash | 427 | 80.91 s | 5.28 tok/s | 22.13 GiB | replace-white 0.996 |
| DeepSeek-V4.1-Flash-1017 | 249 | 84.46 s | 2.95 tok/s | 84.16 GiB | replace-white 0.998 |
| GLM-5.3-flash | 259 | 106.59 s | 2.43 tok/s | 16.08 GiB | replace-white 0.999 |

**装载**是进程启动到第一次应答 `GET /v1/models`：映射引擎、分词器、目录扫描。它不属于任何一个
请求。

### 两张表说明什么

**一次决策正好等于一次 prefill。** 它的墙钟时间等于同一提示词下对话的首 token 时间，且没有
第二项：GLM 的 106.59 s 就是 259 token 以 2.43 tok/s 做前向，不是"生成没停下来"。也正因如此，
三个 checkpoint 给出同一个答案（`replace-white`，0.996 / 0.998 / 0.999）：读数不依赖机器快慢。

**同一段对话，不是同样多的 prompt token。** 同样的两条消息，GLM 编成 93 token、DeepSeek 108、
MiniMax 261——MiniMax 的模板自带一段其它两家没有的系统脚手架。同一段 state 上的决策继承了
这一点：259 / 249 / 427 输入 token。MiniMax 仍然最先完成，因为它的 prefill 速率约为另两家的
两倍：它付更多 token，但单价更低。

**prefill 速率是家族常数，不是请求的属性。** MiniMax 4.67–5.28 tok/s、DeepSeek 2.15–2.95、
GLM 1.67–2.43。家族内部的差异是 n=1 的噪声加上提示词长度；家族之间差的是内存路径。

**decode 是唯一性质不同的项。** GLM 0.80 s/token、MiniMax 0.65、DeepSeek 0.46——都远高于
本文其它小节记录的短上下文步时（275 ms / 664 ms），因为每一步都在冷读专家集，而 pin 缓存装不下
一步要碰的 7.88 GiB。这份测量的第一版在 GLM 上根本跑不完；那是泄漏，不是步时代价，见下一节。

### 这次测量发现的泄漏，以及修复

GLM 那一行的第一次尝试被 jetsam 杀了三次——32 GiB 缓存（metal）、8 GiB 缓存（metal）、
8 GiB 缓存（CPU）——模式完全一致：

| 已解码 token | 修复前 footprint | 修复前 RSS | 修复后 footprint |
|---:|---:|---:|---:|
| 8 | 67.17 GiB | 50.14 GiB | 49.94 GiB |
| 16 | 117.39 GiB | 38.45 GiB | 49.95 GiB |
| 24 | 179.37 GiB | 34.91 GiB | 49.96 GiB |
| 48 | 被杀 | — | 49.99 GiB |

大约**每解码一个 token 涨 6 GiB footprint，而 RSS 在下降**：涨的是压缩/页缓存记账，增量正好
对上一步 7.88 GiB 的专家读取量。根因在 `moe.c`，不在流式层：单 token 的 MoE 路径为
route-first fetch 分配了一块 `cwick_fetch_req` 数组，却**从没调用
`layer_fetch_release`**。pin 缓存拒绝的每个矩阵都是 fetch 自己拥有的缓冲
（`prefill_blob_alloc`），这些缓冲被直接丢掉——一层一份、每一步一次，永不回收。八个专家 ×
三个矩阵 × 45 层，正好是"每一步留下一整步的字节"。

批量路径（`moe.c` 1745 / 1969）一直有释放；只有后来加的单 token 路径漏了。修复是在组 GEMV
返回之后释放这一批并 free 数组（Metal 的两个 wave 在那之前都已经 wait），并在读取中途失败时
把本批自己拥有的缓冲还掉。`make test` 仍然全过。

两个值得记住的后果：

- **GLM 现在可以对话了。** 48 token、1.25 tok/s，默认 32 GiB pin 缓存下 footprint 平在
  49.98 GiB（8 GiB 缓存时 25.94 GiB，速度一样——限制不在缓存大小）。输出文字连贯，变的只有内存。
  同一请求复跑一次：TTFT 54.06 s、端到端 92.15 s，输出**逐字节相同**，说明修复没有扰动算术。
- **短测试抓不到它。** 八个 token 时已经吃掉 67 GiB，但那次运行仍然"跑完"；杀落在 24–32
  token。任何在这个 checkpoint 上量 decode 的东西都要至少跑 32 token 的生成，而且要看
  `cwick:process_footprint_bytes` 而不是 RSS——进程被杀的时候 RSS 是**下降**的。

本节所有运行的日志在 `/tmp/cwickbench-<label>/serve.log`；驱动脚本是一次性的 shell，重跑只需
一个模型一个进程，不需要切换模型。

### 决策路径的同款回归：九次 prefill，没有驻留

上面那个泄漏在 decode 循环里，而 `/v1/systemone` 从不进那个循环。那只是论证，不是测量，
所以对决策端点跑了同一套长跑：连续多个请求，每个请求带一个问题以上，每个请求前后各采一次
footprint。

**方法。** 一个 checkpoint 一个进程（不切模型），`--backend metal`，`--context 32768`，
前缀缓存上限 1 GiB，每轮换 state 与问题集，保证后一轮不可能用上一轮的缓存作答。请求内的
每个问题按设计各自一次 prefill，所以一个带三个问题的请求就是三次 prefill。

**GLM-5.3-flash**（`/tmp/jevlong-glm-5.3-flash/serve.log`）：

| 轮 | prompt token/问 | 问数 | 输入 token | 墙钟 | footprint 前 | 后 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 251 | 3 | 753 | 364.5 s | 15.46 GiB | 16.15 GiB |
| 2 | 297 | 3 | 891 | 433.1 s | 16.15 GiB | 16.14 GiB |
| 3 | 468 | 2 | 937 | 376.7 s | 16.14 GiB | 16.20 GiB |

**MiniMax-M3.1-Flash**（`/tmp/jevlong-minimax/serve.log`；第一行"前"是 0.12 GiB，因为
模型是懒映射的，那次采样发生在第一个请求碰它之前）：

| 轮 | prompt token/问 | 问数 | 输入 token | 墙钟 | footprint 前 | 后 |
|---|---:|---:|---:|---:|---:|---:|
| short | 408 | 2 | 817 | 165.7 s | 0.12 GiB | 22.11 GiB |
| long（跨两个 512 分块） | 829 | 2 | 1659 | 327.5 s | 22.11 GiB | 22.22 GiB |
| short 再来一次 | 408 | 2 | 817 | 153.0 s | 22.22 GiB | 22.23 GiB |

**说明什么。**

- **决策路径什么都不留。** 第一个请求之后（它一次性映射权重、分配池），GLM 在 1828 个输入
  token 上只动了 **+0.06 GiB**，MiniMax 在 2476 个上动了 **+0.12 GiB**。上面修掉的泄漏是
  每**步** +6 GiB，这里小两到三个数量级，而且不随请求数增长——GLM 的第三轮比第二轮还低。
- **每个问题都是一次完整 prefill。** `input_tokens` 正好是 问数 × prompt（753 = 3 × 251，
  1659 = 2 × 829）。一个请求内部不会跨问题复用 state 的 KV，所以三问的决策是三次 prefill，
  不是"一次 prefill 加三次读数"。这正是端点落地时记下的欠账，这次把它量了出来。
- **读数是确定的。** 同一个 GLM 请求在两个独立进程里各跑一次，两次都给出 `replace` 0.9989、
  `escalate` 0.6459、`replace` 0.9977——小数点后第四位相同。MiniMax 重复的那轮也选了同样两个
  答案。所以决策变了只可能是模型或 prompt 变了，不是"这次跑偏了"。
- **pin 缓存对 prefill 不起作用。** GLM 在这些请求期间的日志是 `expert cache 0.00 GiB pinned,
  686.53 GiB streamed`：251 个 token 几乎碰到整个专家集，步与步之间没有可留存的东西，成本
  被设备读取速度定住。这也正是 footprint 平的原因——决策路径没有能涨的缓存。

本轮 prefill 速率：GLM 2.07–2.48 tok/s，MiniMax 4.93–5.34 tok/s，就是上面表格里的家族常数。
这次测试没有改引擎一个字节，它是一道回归护栏，并且通过了。护栏本身已入库为
`tools/decision-longrun.sh`（`sh tools/decision-longrun.sh <label> <root> [轮数] [每轮问题数]`），
下次改 prefill 直接重跑，不必再重新推导方法。

### 决策 prefill：一块、一个共享前缀，且只在答案精确处启用

三个常驻 checkpoint、`--backend metal`、context 4096，一行一个请求，
`POST /v1/systemone` 的墙钟。「改前」是这次改动开始时的二进制（512 token 块、每问一次完整
prefill）；「改后」是 1024 token 块 + 请求内前缀复用。

| 请求 | 后端 | 输入 tok | 实际 prefill tok | 改前 | 改后 | 加速 |
|---|---|---:|---:|---:|---:|---:|
| 单问，卡丁车量级的 prompt | MiniMax | 550 | 550 | 109.3 s | 95.5 s | **1.14×** |
| 三问，共享 348 token state | MiniMax | 1145 | 449 | 213.8 s | 102.0 s | **2.10×** |
| 三问，共享 172 token state | DeepSeek | 605 | 605 | 240.1 s | 237.5 s | 1.01× |
| 三问，共享 173 token state | GLM | 611 | 611 | 306.1 s | 306.1 s | 1.00× |
| 单问，与卡丁车同体 | DeepSeek | 370 | 370 | — | 143.3 s | — |

DeepSeek 与 GLM 仍然 prefill 全部 token，因为它们的复用不精确（见
`DECISIONS.md` §44）：强制复用会改变答案（`|dp|` 0.56 与 0.41），而在 GLM 上走一次快照
往返仍留下 `|dp| = 0.11`。它们的单问成本是「把模型读一遍」——GLM 251 token 流式读了
686.53 GiB——所以减少 token 也几乎不动它。

**均匀切块的反例**，值得留着，因为它看起来应该更快：550 token 切成 275+275 实测 113.6 s，
而 512+38 是 109.3 s。专家读是块级成本，把大块砍半的损失大于小尾巴带来的收益。有效的手段是
提高上限（512 → 1024），不是切得更均匀。

MiniMax 上更大块的代价：footprint 23.8 GB，swap 1.36/3.0 GiB，运行期间不增长。

## 2026-11-12：同一个 checkpoint 走完四个方言，以及 40 tok/s 得从哪里来

Qwen3.5-9B 是这里唯一整份装进内存的 checkpoint：17.98 GiB BF16 权重在加载期 2.6 s
装入并 wire，32K 位置只要 ~1.05 GiB 的 KV 加 DeltaNet 状态（`DECISIONS.md` §52）。
所以「API 在引擎之上还加了多少」这个问题只能用它来问——没有存储路径可以躲在后面。

**方法。** 一个全新进程，`--backend metal --context 32768`，**前缀缓存关闭**，四个方言打同一个
81 token 的 prompt、`max_tokens=64`，逐方言关掉思考（chat 用 `reasoning_effort=none`，
responses 用 `reasoning={"effort":"none"}`，messages 用 `thinking={"type":"disabled"}`——决策
接口本来就没有思考可关）。每个方言跑三次，顺序 chat → responses → messages → systemone，
所以每一行都是热模型。驱动脚本 `tools/apibench.sh`（见「复现」）：请求墙钟由 curl 打印，
token 数直接读服务端自己的汇总行。

### 四个方言，一个 prompt，一个进程

| 方言 | prompt token | 输出 token | prefill tok/s | decode tok/s | 首字节 | 端到端 |
|---|---:|---:|---:|---:|---:|---:|
| `POST /v1/chat/completions` | 81 | 41 | 37.65 | **7.04** | 6 ms | 7.98 s |
| `POST /v1/responses` | 81 | 41 | 37.69 | **7.09** | 6 ms | 7.93 s |
| `POST /v1/messages` | 81 | 41 | 37.66 | **7.09** | 6 ms | 7.94 s |
| `POST /v1/systemone`（3 题） | 525 | 0 | **55.60** | — | — | 9.44 s |

三次测量的区间：prefill 37.3–37.9 tok/s、decode 6.98–7.12 tok/s、墙钟 7.91–8.05 s。三个
生成方言互相之间差 **0.2%**，这是应该的：方言只是请求形状，三者最后都进同一个
`qwen3_5` 前向和同一个采样器。这些数字真正说明的是 **HTTP 与 SSE 编码没有可测成本**——
41 token 的 8 s 墙钟 = 5.9 s decode + 2.2 s prefill，整个请求/响应信封不到 10 ms。

`首字节` 是 SSE 流的第一个字节，也就是响应头那一段，**不是 TTFT**。首 token 时间是
prefill 加上一步：81 token 时是 2.15 s + 0.14 s。之所以还是列出来，是因为只等 socket
而不等 delta 的客户端会看到 6 ms，然后以为模型瞬间就答了。

**一次决策是三次 prefill，不是一次。** 三题 `/v1/systemone` 发起了三次约 175 token 的
prefill（178 / 174 / 173）、零 decode，于是 525 token / 9.44 s = 55.6 tok/s。它按 token 算
**比** chat（37.7 tok/s）**更快**，原因和「长 prompt prefill 更快」是同一个：权重读是按块付的，
175 token 把 16.5 GB 摊到了更多活上。决策接口同时也证明了它存在的意义——没有采样器、没有
decode 循环、没有第二个成本项。

### 天花板：清单里每一个模型都算一遍

40 tok/s 是被点名的目标。它值得写清楚需要什么，因为它不是一个调优项：一次 decode 步读一次
它需要的权重，读一次出一个 token。

| 模型 | 每步字节 | 40 tok/s 需要 | 本机提供 | 实测最好 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B | ~16.5 GB（全部权重减去没读的视觉塔和只读一行的 embedding） | 660 GB/s | CPU 123.5 GB/s，GPU 368 GB/s | **20.4 tok/s**，是 GPU 368 GB/s 的 93% |
| GLM-5.3-flash | ~19.9 GB（8.06 GiB 专家 + ~10.4 GiB 常驻） | 795 GB/s | 123.5 GB/s | 3.63 tok/s（专家常驻时） |
| DeepSeek-V4.1-Flash-1017 | ~17.9 GB | 716 GB/s | 123.5 GB/s | 2.78 tok/s |
| MiniMax-M3.1-Flash | ~20.8 GB（19.36 GiB） | 832 GB/s | 123.5 GB/s | 1.48 tok/s |

关于这个 checkpoint 上 `--backend metal`，这里事后修正一句：CPU 池的 7.0 tok/s 确实是
**CPU** 总线的 93%，「这条路上没空间了」这个判断没错——但那条总线本来就不该是它。Qwen3.5 是
稠密 BF16，一个 decode 步就是 16.5 GB 权重的一条流；只要整步装进**一个** command buffer
而不是 128 次 commit，GPU 能以 368 GB/s 搬它（下面一节，以及 `DECISIONS.zh-CN.md` §61）：
**20.4 tok/s**，CPU 池的 2.9 倍，文本逐字节相同。「GPU 无处可施」从来不是稠密模型的事实，它
是**专家散列**那种形状的事实。

123.5 GB/s 是这台机器实测出来的（4 线程即饱和，1/4/16 GiB 缓冲同速率）。所以 40 tok/s 等于
要 **本机内存带宽的 5.3 到 6.7 倍**，清单里每一个模型都如此——而在唯一能用上 GPU 的稠密
checkpoint 上，是 GPU 总线的 1.8 倍。要填这么大的差距只能让每个 token 少搬字节，而那条路上
每一条岔路都被 `goal.md` 的质量铁律否掉了（不剪枝、不降 top-k、不用有损权重格式，装载期就
检查）。9B 稠密模型按 4 bit 算每步 ~4.1 GB，在 GPU 总线上限 30 tok/s——还是不够，而且那已经
不是盘上的这个模型了。

**剩下的唯一杠杆值多少。** 不碰精度还能减少「每 token 字节」的办法只有一条：一次权重读产出
不止一个 token。对稠密模型这笔买卖比 MoE 划算得多（MoE 的第二个 token 会把专家并集撑大而不是
共享，本文件上面量过：每 token 字节 0.815×）。

这里原来是从 CPU 步推算的，得到 k=4 → 13 tok/s、k=8 → 17 tok/s，结论「40 的一半」。那个推算
已被 §61 作废：现在的一步是 GPU 的 46.5 ms，不是 CPU 的 129 ms，而 GPU 上「多校验几行」的算术
不再是没有余量的东西——M3 Max 有 ~14 TFLOP/s，对上一个 token 16.5 GFLOP 的预算，同一次权重
读可以校验好几行。替代它的算式有两个未知量，都可以量：

    tok/s = k * alpha / (46.5 ms + k * c)      alpha = 草稿接受率
    -> 40 tok/s 需要  k * alpha >= 1.86 + 0.040 * k * c

`c` 是多校验一行的一行边际成本（投影从 GEMV 变成 k 行 GEMM，读的是**同一批字节**，所以 `c`
是算术和发射、不是 DRAM），它由把整步前向从 `n == 1` 扩到 `n == k` 直接量出来。`alpha` 需要
把草稿头绑上：`model-mtp.safetensors`，486 MB，`num_nextn_predict_layers=1`，在盘上、未绑定。
实现成本不在草稿模型——而在 k 行前向入口，加上 50 MiB DeltaNet 状态的快照/回滚，因为被拒绝的
草稿没法像 KV cache 那样截断回来。两项都已定价，都没开工。

### 复现

```sh
./cwick serve --root ./models/xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic \
    --backend metal --jev --context 32768 --port 8004 --password cwick-local \
    --models-dir ./models 2>&1 | tee logs/apibench-q35.log

sh tools/apibench.sh --host http://127.0.0.1:8004 --password cwick-local \
    --tokens 64 --repeat 3 --log ./logs/apibench-q35.log
```

脚本每个请求打一行：端点、curl 的墙钟，以及从服务端汇总行读回来的 prompt / 生成 / prefill /
decode——客户端自己再算一遍，只会和引擎报的数漂开。

### 上面那张表的 prefill 是 81 token 的数字，而 prefill 不是平的

四方言那张表用的是 81 个 prompt token，此时 attention 基本不花钱。到 agent 的量级看的是曲线，
而这条曲线原来是弯的：一次被记录的 31,690 token 请求里，速率从 73.4 tok/s（第一块）一路掉到
14K 处的 14.8 tok/s，eta 还随着往下掉而变大。对那次实跑拟合得
`t(n) = 0.0117n + 3.90e-6n²`，误差 1%——一个二次项，也就是 attention，在一个 GEMM 路径能跑
550 GMAC/s 的引擎里只跑到 8。

`DECISIONS.md` §53 修了前半段（每次 K/V 扫描服务 8 个 query、点积走 NEON），§54 修了后半段：
两段 attention 现在就是 GEMM，而 K/V 缓存改成 f32，于是它们和模型里其他每个投影一样走
Accelerate。调用方看到的是，同形状 prompt、各起一个进程、`--context 131072`：

| prompt | prefill | 速率 |
|---|---:|---:|
| 760 token | 10.0 s | 76.2 tok/s |
| 7,600 token | 94.8 s | **80.2 tok/s** |
| 14,060 token | 184 s | 76.2 tok/s |
| 30,970 token | **436 s** | **71.0 tok/s** |

代码树恢复后用同一个二进制复跑（同样三种 prompt 形状、各起一个进程）：760 → 9.79 s /
77.6 tok/s，7,600 → 96.9 s / 78.5，30,970 → 434.0 s / 71.4。这是常见的百分之几的机器
漂移，曲线形状不变。

速率到 14K 都是平的，31K 还有平地的 94%，这才是原来缺的性质：剩下的退化是取消不掉的那部分
——score 与 output 每个块按定义就是 `O(keys)`。它们的单价从每 1000 key 每块 0.26 ms 降到
0.075 ms，于是 30K 位置的一个块在 attention 上花 2.3 s 而不是 7.8 s，而 32 层的投影（4.5 s）
仍然是最大的一项。

decode 每个 token 读一次权重、与上下文无关——11K 时是 139 ms 步长里的 111 ms——而 attention 是
唯一随上下文增长的那一项。§54 的 decode 内核之前它是 174 ms 步长里的 52 ms，之后只有几 ms：
11K 时 7.2 tok/s，短 prompt 7.0，也就是说花钱的不是缓存，是权重。

还有一个值得记住的数字：命中前缀缓存的 587 token prompt，prefill 是 **147 ms** 而不是
9556 ms——状态是一次顺序读一个快照，而不是 587 个位置的计算量。第一轮之后，决定 agent 首 token
时间的就是这一项。

## 2026-11-13: 当初否决 GPU 时，用错了天花板

上面那条结论说这个 checkpoint 上 `--backend metal` 无事可做，理由是"它的 GEMV 已经跑到内存总线的
93%"。那 93% 是拿 **123.5 GB/s** 算的，而 123.5 是 **CPU 集群的通路**，不是这台机器的内存系统。
三组测量把这件事定下来。

**CPU 读带宽**（`tools/membwbench`，8 GiB，三种内核，1..16 线程）：

| 内核 | 1 线程 | 2 | 4 | 8 | 12 | 16 | 峰值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 字节标量累加 | 75.0 | 114.9 | 122.1 | 121.7 | 121.8 | 116.8 | 123.8 |
| 16 字节向量累加 | 40.3 | 80.1 | 120.4 | 117.9 | 119.3 | 113.1 | 120.4 |
| 16 字节 + bf16 展宽 | 21.8 | 43.1 | 84.1 | 119.1 | 117.5 | 116.7 | 119.8 |

四个线程之后就平了，而单线程已经能拉 75 GB/s。这是**CPU 侧 120-124 GB/s 的共享上限**，不是单核极限。

**GPU 读带宽**（`tools/gpubw`，4 GiB private 缓冲，同样的循环形状，10K..327K 线程）：

| 内核 | 峰值 |
|---|---:|
| `uint4` grid-stride 累加 | **284.6 GB/s** |
| `bf16` 展宽 + 每 16 B 八次 f32 加 | **284.3 GB/s** |

bf16 变体与纯搬运逐字节持平才是重点：真实 BF16 GEMV 要付的转换在这个速率下是免费的，所以
285 GB/s 对跑投影的那个内核是可用的。对 4 GiB 缓冲做一次 dispatch 实测峰值 **329 GB/s**
（`tools/dualbw`）。

**两者不叠加**（`tools/dualbw`，12 个 CPU 线程与一次 GPU dispatch 同一时间窗，每边 4 GiB）：

| 窗口 | GB/s |
|---|---:|
| 只 CPU | 123.5 |
| 只 GPU | **328.9** |
| 同时跑 | **213.7** |

同时跑既到不了两者之和（452），也守不住 GPU 自己的速率：只有它的 65%。所以把一条投影的输出行
拆给 CPU 池和 GPU 并行——另外两个后端用的正是这个拆法——在这里是净亏。两条引擎共用一根管子，
而 CPU 分到的是**更慢**的那一份。

**这对步长预算意味着什么。** 每个 token 16.5 GB，一次提交：

| 引擎 | 步长 | tok/s |
|---|---:|---:|
| CPU 池 @123 GB/s | 134 ms | 7.4（实测 7.0） |
| GPU @285 GB/s | 58 ms | 17.1 |
| GPU @329 GB/s | 50 ms | 19.8 |

所以余量是真的，是 2.4×，不是零。**不**真的是：靠一条一条投影走 `cwick_metal_*` 去拿它。

**调度项把它吃掉了。** `tools/q35step` 按 checkpoint 自己的形状配比（32 层、13.82 GB、两套轮换权重
让缓存失效）复现生产用的 batched-BF16 API，并且保持真实层的分组边界——CPU 必须先拿到 `in_proj`
才能做递推、先拿到 `gate/up` 才能算 `down`，所以每层是四次提交：

| 路径 | ms | GB/s | 提交次数 | 每次 commit+wait |
|---|---:|---:|---:|---:|
| CPU 池，`cwick_gemm_bf16` | 151.0 | 91.5 | — | — |
| GPU，按组批 | 118.8 | 116.3 | 128 | **928 µs** |

1.27×，不是 2.4×。八层时是 0.95×。`CWICK_METAL_TIME=1` 说明了原因：`wait` 稳定约为 `gpu` 的两倍，
也就是每个批有一半是调度，128 次合起来 ~119 ms，而内核预算只有 58 ms。这与源码里为非批路径记下的
是同一堵墙（"一次 commit+wait 约 450 µs……非批路径因此是 opt-in，会输"），只是对一个每层需要四个
同步点的模型更狠。

**剩下的，标好价。** 每 **步** 一次提交而不是每组一次：为整个 decode 步写一条 Metal 前向
（norm、DeltaNet 递推、attention、MLP、lm_head），32 层放进同一个 command buffer。按实测传输量，
那是 50-58 ms 内核加一次往返，即 **17-20 tok/s**，而且需要比现在"每行一个 simdgroup"更好的 GEMV
才能摸到 285 GB/s。在此之上，已躺在盘上的 MTP 头能把 50-58 ms 的一步变成 2 到 2.5 个 token：
`40 tok/s` 需要每步被接受 2.0-2.4 个 token，所以它只能作为 **整步 GPU 前向 × MTP** 出现，两半都不是
调参级别的改动。

这条前向现在存在了，实测 **49 ms / 20.45 tok/s**（下一节）；被它替掉的 CPU 池保留为
`CWICK_Q35_GPU=0`，它仍然是 CPU 总线能给的 95%——这正是要点：95% 的**错的那条**总线，值 2.9 倍。

### 一次提交的界，实测

`tools/q35one` 把一步里的同样 128 个 GEMV 编进 **一个** command buffer
（同一 buffer 内的 compute encoder 按提交顺序执行，所以依赖链无需往返），只等一次：

| 编码方式 | ms | GB/s | 折算 16.5 GB 步长的 tok/s |
|---|---:|---:|---:|
| 128 次提交（`tools/q35step`，GPU） | 118.8 | 116 | 8.2 |
| **1 次提交，1 套权重（496 MB）** | **37.6** | **368** | **22.3** |
| 1 次提交，4 套权重（2 GB） | 38.0 | 364 | 22.1 |

工作集这一项很关键：轮换四份不同的拷贝结果不变，说明 368 GB/s 是这个访问模式的 DRAM 速率，
不是缓存假象；而且它**高于**对单个 4 GiB 缓冲做一次 grid-stride dispatch 的 285 GB/s
（`tools/gpubw`）。链接式形状——很多次短 dispatch，每次扫一个 8..96 MB 的矩阵——比一次长扫
调度得更好。

所以调度墙就是全部故事：118.8 ms 和 37.6 ms 搬的是同样的字节。真实前向的步长预算因此是
37.6 ms 的 GEMV 加上非 GEMV 部分的代价（norm、DeltaNet 递推、走 KV 缓存的 attention、
把 logits 拷回 CPU）。这就是纯 GEMV 的 22 tok/s 与上面估计的 17–20 tok/s 之间的差距，
也是整步前向必须守住的东西。

### 整步前向，已落地

`CWICK_Q35_GPU=1`（有 Metal 设备时即为默认）把一次 decode 位置交给 `src/q35_gpu.c`——
32 层、16 个内核、**一个** command buffer、一次等待。生产配置（128K 上下文、前缀缓存、走 HTTP）实测：

| decode 路径 | ms/步 | tok/s |
|---|---|---|
| CPU 池 | 143 | 7.00 |
| GPU 整步，`cwick run`，4K 上下文 | 49.8 | 20.10 |
| GPU 整步，serve，128K 上下文 + 前缀缓存 | 48.9 | **20.45** |

47.4 ms 的墙里内核占 46.5 ms——341 GB/s，对上隔离测量里这条 GEMV 链的 368 GB/s。也就是说这个形状配比
允许的 93% 已经全花在搬权重上，其余是 norm、递推、卷积和走 KV 缓存的 attention。同一 prompt 下 90 个
贪心 token 与 CPU 路径逐字节相同；logits 差 0.008（最大）/ 0.0002（平均），见 `CWICK_Q35_GPU_CHECK=1`。

### 校验这一半到了，但还不够

40 tok/s 是一个关于「草稿」的问题：46.5 ms 的一步需要每步被接受 1.86 个 token，而「校验 k 个」
的那一趟读的权重和一个的那一趟一样多。这一半现在做完了，而且是精确的——`CWICK_Q35_VERIFY=k`
把 k 个连续位置塞进一趟权重 pass，`CWICK_Q35_GPU_CHECK=1` 发现每一行都与「同一条 GPU 路径
逐行跑」逐位相同（max 0.00000、N 个里 0 个不同，32 层，k = 2/3/4/8）。只保留被接受的前缀
只需一次 `memmove`，因为递推状态在 k+1 个槽里滚动，而不是原地重写。

真实 decode 步的价格（99 趟，三次交错取最好）：

| k | ms/趟 | 边际 ms/行 |
|---|---|---|
| 1 | 47.4 | — |
| 8 | 81.8 | 4.9 |

只报两端：这台机器的基线在每次运行之间能漂 20% 以上，中间那些 k 跨会话也漂这么多，所以边际
按两端斜率读，而不是取某个跑了十几趟的行。同一趟隔离测量（`tools/q35gemab`，同进程、同权重、
13.82 GB 每趟）在 k = 1 是 39.4 ms、k = 8 是 52.6，边际 1.9 ms——与真实步的差就是递推和卷积，
24 个线性层把它们**按行**各跑一遍。

那个内核原本把 k 行激活暂存在 threadgroup 内存里。逐项摘掉来量，才看出暂存就是成本：k = 8 时
权重读 29 ms、激活读 21 ms、全部 N 路乘法加在一起 1.8 ms，而 36.4 + 29.4 正好是实测的 65.8——
两条内存流是串起来的，没有重叠。整个 pass 的 x 只有 128 KB，所以直接从 device 内存读
（驻留 L2、不暂存、无 barrier）在 k = 8 是 52.6 ms 对 65.8。细节在 DECISIONS §63。

于是终点落在草稿上，而且现在是实测而不是预测。prompt 查表不花权重，每趟只能拿回
1.08 到 1.24 个 token（散文上 52% 草稿被接受，代码上 23%）——端到端比不投机还慢，
2022 ms 对 2011。MTP 头已经绑上（DECISIONS §64）：486.6 MB 权重、**没有自己的输出
投影**，所以一个草稿 token 要读 2.52 GB（含共享的 2.03 GB 词表投影），即 7.0 ms；
换成 4096 行候选切片约 1.7 ms。

本文件早期版本的表格是按 p = 0.8 算的。实测按深度是 **0.356 / 0.125 / 0.0**（全头）、
**0.185 / 0.038 / 0.0**（切片），所以期望接受数 A = Σ p^i 饱和在 1.40，而不是 2.95。
同一 session、200 token prompt、贪心：

| 形态 | token/趟 | ms/趟 | tok/s |
|---|---|---|---|
| 一行、不投机 | 1.00 | 48.1 | **20.74** |
| VERIFY=2、不带草稿 | 1.00 | 56.2 | 17.79 |
| SPEC=1，512 行切片 | 1.18 | 60.0 | 19.72 |
| SPEC=1，2048 行切片 | 1.18 | 59.2 | 19.94 |
| SPEC=1，全 2.03 GB 头 | 1.35 | 66.8 | 20.19 |

每一臂都亏。原因就在行成本：离开一行要 **+8.0 ms**（56.2 对 48.1），而之后每行只要
**+1.6 ms**。DECISIONS §65 从这一步里拿回 2 ms —— 把逐行的 norm、残差加、SwiGLU 各自
合并成一次 dispatch，n = 2 时用 RP 4 —— 但符号没变，因为一份草稿要先赢过 8 ms，而最好
的草稿只对 35.6%。

所以 40 tok/s 不是「一份只花 1.5 ms 的草稿」——那样的草稿已经做出来了，它照样亏。
一个 token 读一份权重的前提下，40 tok/s 需要**每趟 2.08 个 token**，而这里的上限是
1.40：这个检查点只带一个按单个深度训练的 MTP 块，再往后链自己的残差就是分布外
（深度 2 条件接受 12.5%，深度 3 为 0）。要补上这个差距，需要一个训练过多深度的
草稿模型，而它不在文件里。

prefill 仍留在 CPU，71-80 tok/s。
