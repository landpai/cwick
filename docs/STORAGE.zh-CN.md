# 原生存储准备

<p align="center"><a href="STORAGE.md">English</a> · <a href="STORAGE.zh-CN.md">简体中文</a></p>

CWICK 以 mmap 方式映射一个 BF16 的常驻 spine（作为事实来源），并逐层
读取 routed expert。

## 两种 checkpoint 布局，同一个读取器

加载器在打开时判断模型根目录属于哪一种：

```
model.safetensors.index.json + model-*.safetensors   发布原样的 checkpoint
resident/tensors/<name>.bin                          转换过的 `resident/` 树
```

三个自带 checkpoint 都是**按发布原样**读的，没有转换步骤。`resident/` 树是
`cwick setup` 为合成验证模型写出的形状，也是转换语料的样子；它同时是 `--spine int8`
唯一适用的形状（对 safetensors checkpoint 该参数会被忽略，并打一行日志）。

控制台或 `/v1/hub/download` 下载的权重落在 `./models/<owner>/<name>/`，它本身就是
这样一个根目录：`config.json` 加分片就是加载器要的全部。

## `resident/` 树的目录布局

```
config.json
resident/
  tensors/<name>.bin        BF16 稠密 spine，行优先 [out, in]
  experts/expert-L-E.bin    每层/每个 expert 一个规范原始文件
resident-int8/
  tensors/<name>.i8         可选的 row-int8 数据
  tensors/<name>.s          可选的 fp16 逐行缩放
tokenizer/
  vocab.json                可选的 BPE 词表（token -> id）
  merges.txt                可选的 BPE merges
```

BF16 张量是 torch 线性布局（`weight[k][j]`，以输出为主序）下的原始小端
bfloat16 行。一个 norm 向量 `[n]` 存储为单行。
可选的 int8 表示存储 `int8` 行，外加每行一个 fp16 缩放因子。

## Expert 文件格式

每个 routed expert 是一个文件：一个 20 字节的头部
（`"CWICKEXP1"` + u32 layer + u32 expert + u32 matrix count），然后是线性布局的
三个 BF16 矩阵：`w1 [I, RH]`、`w2 [RH, I]`、`w3 [I, RH]`，其中
`I` = `moe_intermediate_size`，`RH` = `routed_expert_hidden_size`。

完整的 expert 语料必须齐备：若任何 layer/expert 文件缺失，
`cwick_spine_open` 会 fail closed（never-pruned 规则）。

## 表示形式

- `bf16`（默认）—— 原始 BF16 权重，逐权重精确（weight-exact）。
- `int8` —— row-int8 + fp16 缩放；用 `--spine int8` 选择。BF16 spine
  仍然是校验的权威基准。

## 初始化

`cwick setup` 会按这一布局写出一个确定性的微型模型（第 0 层：KDA +
稠密 MLP；第 1 层：MLA + MoE），因此无需下载真实权重就能验证整个
engine 与 server。真实 checkpoint 不需要这种转换——它们就地按 safetensors 分片读取；
`model.c` 里的契约校验器会在任何权重数据被打开之前
强制执行完整的 no-pruning 契约。

对已发布的 HF checkpoint 无需任何转换：`cwick` 通过下面的切分方式
直接读取 `model-*.safetensors` 分片。

## 内存有界的流式加载（`src/stream.c`）

已发布的 checkpoint（306 GiB / 62 个 shard）装不进 RAM，因此存储是按
张量被触及的频率来切分的，而不是按 shard 边界：

- **常驻（Resident）** —— 每个 token 都要读取的一切：norm、KDA/DSA attention、
  shared expert、embedding、`lm_head`。GLM-5.3-Flash 约 15.46 GiB。保存在
  共享的 `mmap` 中，启动时用 `madvise(WILLNEED)` 做 warm-wired，并在
  `cwick_engine_new` 中对照主机内存做校验。
- **流式（Streamed）** —— routed expert bank（`*.mlp.experts.*`），290.32 GiB。
  在以 `F_NOCACHE` 打开的 descriptor 上用 `pread()` 读取，读入一块固定的
  32 MiB arena。不做任何保留：kernel 绝不会为 expert bank 建立 page-cache
  副本，RSS 从构造上就是有界的。

每个张量除映射指针之外还携带 `shard_index` 与 `data_offset`，
因此同一张 descriptor 表可以同时服务这两条路径。

## 用了多少 RAM：加载时的计划（`src/mem.c`）

常驻量只决定一次，依据进程所运行的这台机器，而这个决定是**硬上限**，
不是愿望：

```
target      = host RAM x --mem-fraction            （默认 0.80）
- resident  = 每个 token 都要触及的 spine           （~15.46 GiB）
- state     = KV / KDA / mHC 缓冲区                 （~2.94 GiB）
- arena     = 暂存与 prefetch arena                 （~0.03 GiB）
- headroom  = 两层 routed 的按需读取                （~0.38 GiB）
= 专家缓存预算                                       （128 GiB 机器上约 83.60 GiB）
```

在本机实测（128 GiB，GLM-5.3-Flash）：

```
cwick: [info] [mem] memory budget: 128.00 GiB host, using at most 80% (102.40 GiB)
cwick: [verb] [mem] budget split: resident spine 15.46 GiB + runtime state 2.94 GiB + arenas 0.03 GiB + read headroom 0.38 GiB
cwick: [verb] [mem] expert cache budget 83.60 GiB (routed expert bank is 290.32 GiB, streamed per decode step)
cwick: [info] [gen] prefill 24/147 — rss 31.71 GiB of 102.40 GiB ceiling, expert cache 82.12 GiB pinned, 82.77 GiB streamed
```

这个计划是：`recommendedMaxWorkingSetSize * pct`，减去非 routed 权重，
减去一块重叠预留，再减去运行时状态。
有两点特性在这里真正发挥了作用：

- 推理状态在 spine 打开**之前**就已入账，因此专家缓存是按真正剩余的 RAM
  来定尺寸的（旧代码在分配完缓存**之后**才计算常驻切分，因此从来没有把它
  减掉 —— 在 128 GiB 机器上，它在 15.5 GiB 的 spine 之上又预留了 96 GiB
  缓存）；
- 这个上限是**在运行过程中**被强制执行的。每 8 个 decode 步，engine 就把
  `task_info()` 得到的 RSS 与计划比对，一旦发现超支，就释放已 pin 的
  expert 字节（`cwick_stream_trim`）并下调缓存上限。专家缓存是进程里
  唯一有弹性的东西；其余一切在加载时就已固定。

## Expert pin 集

一个建立在原始 expert blob 之上、按字节预算、按热度排序的 pin 集
（`cwick_stream_options.expert_cache_bytes`，默认开启，取计划剩下的额度）。
插入时在一张按预算定尺寸的开放寻址表里探测一小段窗口，驱逐时丢掉该窗口中
最冷的条目 —— 旧的直接映射 1024 槽表容纳不下约 36k 个 blob，所以一旦预算
长到几 GiB 以上，它就无声地退化成“最后撞上的那个”。

热度模型：一次命中增加一个 quantum，每 64 步所有条目按几何级数衰减，
字节压力到来时先淘汰最冷的条目。pin 只是一种优化：一次 miss 的代价是
一趟 SSD 读取，绝不会产出一个错误的 token。

可用 `--mem-fraction PCT` 覆盖 RAM 的占用比例，用 `--max-memory GiB`
给出绝对上限，或者设置 `CWICK_MEM_PCT` / `CWICK_MEM_MAX_GIB`。

`cwick doctor` 走的就是这条路径，通过真正读取 checkpoint 来校验它：
常驻张量全量读取，expert 按跨步采样读取（`--deep` 则读全部
306 GiB），并随读取不断计算 FNV-1a 校验和。一次完整常驻遍历
加 4.76 GiB expert 采样的实测峰值 RSS：**72.9 MB**。
