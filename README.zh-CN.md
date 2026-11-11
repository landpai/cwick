<h1 align="center">CWICK</h1>

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong>
</p>

**CWICK 在一台 Apple Silicon Mac 上跑完整、未剪枝的大模型。** 不是量化素描版，也不是蒸馏替身：整个 checkpoint，每个专家都在磁盘上，每个 token 都由模型自己决定。一个 C 二进制，没有第三方依赖，没有转换步骤，没有 Python——它按发布原样读取 checkpoint，并通过 OpenAI 兼容 API 提供服务。

服务 `:8001` · `./cwick run` · [运行时架构](docs/ARCHITECTURE.zh-CN.md) · [逐相性能](docs/PERF.zh-CN.md) · [决策记录](docs/DECISIONS.zh-CN.md) · [服务端参考](docs/SERVER.zh-CN.md) · [存储分层](docs/STORAGE.zh-CN.md) · [开发交接](AGENTS.zh-CN.md)

名字就是画面：灯芯是让一大块蜡持续燃烧的那根细线。这里的"蜡"是 SSD 上 233–475 GiB 的权重库，"灯芯"是每个 token 真正过总线的那几 GB。

三个 checkpoint 目前端到端支持——不是一份设计上封闭的清单，但每一个都需要自己的加载器、反量化、模板渲染与内核：

| 模型 | 磁盘占用 | 结构 | 为什么收它 |
| --- | --- | --- | --- |
| **GLM-5.3-flash** | ≈306 GiB，62 分片 | 45 层，288 路由专家，top-8，FP8-E4M3 | 最初的模型，≈320B 参数 |
| **DeepSeek-V4.1-Flash-1017** | ≈475 GiB，48 分片 | 40 层，384 路由专家 top-6，MLA + FP4 专家 + FP8 attention | 仍能进笔记本的最大一个 |
| **MiniMax-M3.1-Flash** | ≈233 GiB，48 分片 | 60 层（3 层稠密），128 专家 top-4，NVFP4 专家 + FP8 稠密 | 三者中占用最小 |

质量铁律不可协商，且在加载期强制：**top-k 就是模型自己的 `num_experts_per_token`**（从不降低），专家从不剪枝，embedding 从不共享，分片缺失时直接失败，而不是悄悄跑一个小一点的模型。

# 目录

- [为什么](#为什么)
- [它是怎么造出来的：Agent 的闭环](#它是怎么造出来的agent-的闭环)
- [支持的硬件](#支持的硬件)
- [架构](#架构)
- [快速开始](#快速开始)
- [用法](#用法)
- [配置](#配置)
- [API](#api)
- [模型、上下文与内存](#模型上下文与内存)
- [长 prompt 与前缀缓存](#长-prompt-与前缀缓存)
- [思考、流式与工具](#思考流式与工具)
- [速度](#速度)
- [验证](#验证)
- [详细文档](#详细文档)
- [FAQ](#faq)
- [目录结构](#目录结构)
- [许可证](#许可证)

## 为什么

这个实验问的是：一个完整的引擎——加载器、反量化、内核、服务端、web 控制台——能不能由 **AI Agent 从零**设计并写出来，而不是从通用 runtime 上改；以及这个 Agent 之后能不能自己把它改好：优化 → 编译 → 部署 → 反馈，每一轮结束时的产出是一个数字，而不是一个观点。引擎就是这个闭环的产物——一个项目能做到这个规模，原因也在这里。

由这件事推出两条不变的前提，其余一切都建立在这两条之上。**checkpoint 整份跑**——不剪枝、不降 top-k、不降精度——在 128 GiB 的机器上，把 SSD 当作内存层次的一部分，而不是例外。以及**同一个常驻模型有两种答法**：OpenAI 兼容的聊天 API，以及从一次 prefill 的分布里读出的类型化决策；于是已经部署好的模型不必重训、不必换权重、也不必换调用栈，就成了一个基于概率的决策服务。

下面几条是"为什么引擎必须长成这样"：

- 最好的开放权重现在装得进普通人能拥有的机器：128 GiB 的笔记本上放 233–475 GiB 的权重，是一个调度问题，不是"换个小模型"的理由。
- MoE 模型对专家量化的容忍度远高于稠密模型，所以稀疏部分可以放在磁盘上，而该精确的地方仍然精确。
- 只服务三个 checkpoint 的推理引擎，可以比通用 runner 简单得多，也可以诚实得多。

## 它是怎么造出来的：Agent 的闭环

这个引擎是为这三个 checkpoint 从零设计、从零写出来的：一个 AI Agent 与人协作完成。人定需求、判结论；Agent 写加载器、写内核、写服务端，以及写那套用来量它们的闭环。它不是对某个现成 runtime 的封装，也没有任何一段是从通用引擎搬过来的。

让它成立的并不是"能生成代码"，而是**每个改动都走同一条闭环**，而这条闭环便宜到一天能跑几十次：

```
假设 → 优化 → 编译 → 锚点 → 真机跑分 → 部署（起服务） → 反馈 → 假设
```

* **优化 / 编译**——一次只改一件事，放在默认关闭的开关后面，这样对照臂永远是同一个二进制。
* **锚点**——固定 prompt 必须产出相同的 top-1 token id 与 logit。这是"更快"有意义的前提；没有它，一个算错的内核同样会快 3 倍。
* **真机跑分**——就是这个线程池、这块 SSD、这个内存天花板。本文件里每个数字都来自这台机器，并且把条件写在数字旁边。
* **部署 → 反馈**——一个跑着的服务，它的 `/v1/metrics` 与请求日志就是下一个假设的输入：哪一相涨了、哪个缓存不再命中、哪个请求的时间全花在排队等那一次长 prefill 上。

产出是一张"活下来的改动"清单，和一张更长的"没活下来"清单（[docs/DECISIONS.zh-CN.md](docs/DECISIONS.zh-CN.md)）。活下来的三个都不是计划出来的，是闭环找出来的：attention 内层向量化（398 → 30 ms/token）、专家读盘与计算重叠（io 96 → 50 ms/token）、FP8 稠密 GEMM 做 4×4 分块（96 → 400 GMAC/s）。没活下来的三个当初看起来都很显然：投机预取、把专家读按文件偏移排序、把稠密权重拷进匿名内存——三个都实测更慢。

这条闭环可以不需要人在里面转——`make test`、一个固定锚点、一条跑分命令，就是一次判定。**判定是一个文件，不是一种感觉。**

## 支持的硬件

- **Apple Silicon**，以 128 GiB 为设计点。规划器启动时读取本机内存，把上限设在它的 80%，常驻 spine 与 KV 缓存先从这里出，剩下的全给专家缓存，其余留在 SSD。稳态因此落在上限之下：`swap` 保持平，不会一路涨到几十 GB，压缩器也碰不到每个 token 都要访问的那些页。
- **更小的 Mac** 走同一条路径——SSD 流式不是降级模式，它就是"模型比 RAM 大"时的运行方式。
- **Metal** 加速 GLM 与 DeepSeek 后端（专家 GEMV、批式投影），在 Apple Silicon 上就是默认后端，`--backend cpu` 强制走 CPU 路径。启动时它把 GPU 的 wired 内存上限提到 RAM 的 80%，并把推理状态（KV 缓存、索引器池、循环状态）锁成 wired——每个 token 都要碰的那些页既不被压缩也不被换出。住在 GPU arena 里的专家 blob 下一次 dispatch 直接绑定、不做拷贝，这正是统一内存在这里值得用的原因。MiniMax 后端留在 CPU 线程池：它的内核在这些形状下实测**更慢**，数字在 [docs/DECISIONS.zh-CN.md](docs/DECISIONS.zh-CN.md)，而不是藏在一个开关后面。
- **不支持 NVIDIA 与 ROCm。** 这是为一类机器刻意做窄的引擎。

## 架构

```text
客户端（curl / 编程 Agent / 桌面聊天应用 / web 控制台）
  └── HTTP :8001 ──► server.c
        ├── /v1/chat/completions ──► 队列（单槽，按到达顺序）
        │        └── engine.c ──► 渲染 prompt、prefill、decode、采样
        │              ├── GLM-5.3-flash : spine.c  attention.c  moe.c  pfx.c
        │              ├── DeepSeek-V4.1  : ds_model.c  ds_ops.c  ds_quant.c
        │              └── MiniMax-M3.1   : mm_model.c  mm_prompt.c
        ├── /v1/systemone ──► jev.c ──► 一次 prefill，读答案位置的分布
        ├── /v1/metrics  /v1/models  /v1/config  /v1/logs/stream
        │        └── 槽位忙时照样应答——"能看着一次 30 分钟的 prefill"就靠它
        ├── /v1/hub/*    ──► hub.c（下载任务、模型目录）
        └── /  ├── web/     单页控制台，构建时嵌进二进制
                └── apps/   两个 H5 应用，各自一个 sqlite 文件（store.c）
                └── models/ 在线下载的 checkpoint，启动时扫描

这台机器上的状态
  ├── 常驻   : bf16 spine + KV 缓存 + 索引器池（wired，不压缩不换出）
  ├── 专家   : 共享内存里的 pin 缓存 + SSD 上的 F_NOCACHE 直读
  ├── 快照   : pfx.c 写在 --prefix-cache-dir 的 checkpoint，复用而不是重算
  └── 计算   : pool.c（NEON 线程池）或 metal.c（零拷贝 arena blob）
```

- **语言与依赖**：C11，外加一个 Objective-C 文件做 Metal 桥。链接的只有 macOS 系统框架——Accelerate、Metal、CoreImage、ImageIO、CoreGraphics——以及系统自带的 `sqlite3`。没有 Python，没有 BLAS，没有运行期转换步骤。
- **一个二进制，四个子命令**：`setup`（生成合成验证模型）、`run`（单条 prompt，或多条一起批解码）、`serve`（API 服务）、`doctor`（读一遍常驻字节并打印内存计划）。
- **web 控制台是编译进去的**（`src/webui_embed.c`），所以 `--web` 不需要磁盘上的静态资源，也没有第二个进程；应用是 `apps/` 下的文件，由同一个服务端托管，状态进 sqlite。
- **同一时刻只常驻一个模型。** 这是带宽决定，不是缺功能：两次生成会互相腰斩缓存。

## 快速开始

### 0. 前置条件

- Apple Silicon 上的 macOS，以及 `clang`。别的都不要——内存设计点见[支持的硬件](#支持的硬件)。
- 一个 checkpoint：≈233 GiB（MiniMax）、≈306 GiB（GLM）或 ≈475 GiB（DeepSeek）的空闲 SSD，按发布原样下载。

### 1. 构建与测试

```sh
git clone <this repo> cwick
cd cwick
make            # 生成 ./cwick
make test       # 验证套件：引擎、采样、前缀缓存、服务端、应用
```

### 2. 取一个 checkpoint

`download_model.sh` 是 curl 加 bash——断点续传、支持 100 GiB 以上的单文件、整个仓库只画一行进度而不是每个分片一条：

```sh
# 只看仓库里有什么、有多大，不下载
./download_model.sh deepseek-ai/DeepSeek-V4.1-Flash-1017 --list

# 48 分片，只要权重和 config，下到 start.sh 期望的目录
./download_model.sh deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    -o ./deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    --include '*.safetensors' '*.json' -j 2
```

下载好的目录**本身就是**模型根目录：`config.json` 加分片就是加载器要的全部，契约校验器会拒绝那些质量参数被改过的 checkpoint。

### 3. 起服务

```sh
sh ./start.sh glm              # GLM-5.3-flash，      0.0.0.0:8001，500K 上下文
sh ./start.sh deepseek         # DeepSeek-V4.1-Flash，同上
sh ./start.sh minimax          # MiniMax-M3.1-Flash，32K 上下文（KV 是 2.3 KiB/位置/层）
```

`start.sh` 是推荐的入口：它打开了前缀缓存、web 控制台、局域网共享、三个模型目录和密码——这些一项一项都容易忘。不用脚本时，同一个服务是：

```sh
./cwick serve --root ./zai-org/GLM-5.3-Flash --port 8001 --web --prefix-cache-dir ./pfxcache
```

二进制自己的默认端口是 8000，`start.sh` 服务的是 8001。之后换模型不用重启——`POST /v1/models/switch` 在同一进程里卸载一个、加载另一个。

## 用法

### 起服务、跑单次、自检

```sh
./cwick run    --root ./zai-org/GLM-5.3-Flash --prompt "Explain MoE routing" --max-new 128
./cwick serve  --root ./zai-org/GLM-5.3-Flash --port 8001 --web --prefix-cache-dir ./pfxcache
./cwick doctor --root ./zai-org/GLM-5.3-Flash     # 读一遍常驻字节并报告内存计划
```

### 用 HTTP 聊天

```sh
# 单用户登录（start.sh 把密码设成 cwick-local）
TOKEN=$(curl -s localhost:8001/v1/auth/login -H 'Content-Type: application/json' \
  -d '{"password":"cwick-local"}' | sed -n 's/.*"token":"\([^"]*\)".*/\1/p')

curl -N localhost:8001/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","stream":true,"max_tokens":128,
       "messages":[{"role":"user","content":"用一段话解释 MoE 路由。"}]}'
```

流式与非流式用的是同一个请求；加 `reasoning_effort` 就是调思考深度，加 `tools` 数组就会返回 `tool_calls`。任何 OpenAI 兼容客户端都能接上来——桌面聊天应用、编程 Agent，或内置控制台。`POST /v1/responses` 是同一个引擎的 Responses 方言（输入 `input` items，输出 `response` 对象或 `response.*` 事件），给说那套 API 的客户端用；字段与事件清单在 [docs/SERVER.zh-CN.md](docs/SERVER.zh-CN.md#responses-api)。`POST /v1/messages` 是 Anthropic 那一套——Claude Code 说的就是它——`system`、content blocks、`tool_use`/`tool_result`、`thinking.budget_tokens`，流是 `message_start…message_stop`。三种方言打的是同一个引擎，具名 API key 与 `x-api-key` 都能鉴权。内置控制台三种都说：面板上的**接口**选择决定这一轮发给哪个端点，而三种方言渲染进同一套气泡、工具卡片和统计行。

### 日志与指标

服务端把每条诊断打到 **stdout**，一事件一行，带墙上时间，请求在飞时还带请求号，所以并发日志照样能 grep：

```
cwick: 2026-10-29 21:52:01.902 [info +183.71s] [req 4] stream=true thinking=高 (effort=High)
cwick: 2026-10-29 21:52:02.654 [info +184.46s] [req 4] [prefill] ~310/333 tokens (93.1%), layer 56/60 — 4.5 tok/s, eta 5s
cwick: 2026-10-29 21:53:15.215 [info +251.55s] [req 6] decode(content) 12/24 tokens (50.0%) — 690 ms/step, 1.45 tok/s, eta 8s
cwick: 2026-10-29 21:53:30.104 [info +266.44s] [req 6] POST /v1/chat/completions -> 200 in 12014.7ms prompt=201 completion=18 finish=stop think=12 tools=0 content=6
```

decode 行会说明自己在数哪一段：`decode(thinking)`、`decode(content)` 或 `decode(tools)`。prefill 行没有这个区分，所以它报的是层号，token 数前面带 `~`——那是按完成的层数估出来的，不是数出来的。

`--verbose` 增加每步细节和内存预算；`--trace` 还会打印渲染后的 prompt、输出和每次工具调用（默认关闭：它会把用户内容打到 stdout）。`Ctrl+C` 任何时候都能干净停下——生成中则在下一个 token 处中断，在飞请求得到 503，快照落盘，最后一行会说明。

`/v1/metrics` 用 vLLM 兼容的命名发布同一份状态，可以被抓取，也可以直接在**设置**标签页里看。

### Web 控制台、模型库与应用市场

`--web` 由二进制自己提供单页控制台——磁盘上没有静态资源，没有第二个进程，和 API 同端口。五个标签页：**对话**（流式，端点在 `/v1/chat/completions`、`/v1/responses`、`/v1/messages` 之间由面板上的**接口**选择切换，三者走同一条渲染路径，reasoning 与 tool calls 分区显示）、**应用**（应用市场）、**模型**（模型库）、**日志**（本进程的 stdout/stderr，环形缓冲，所以浏览器能跟住一个长期运行的 serve）、**设置**（控制台与设置合一）。设置页同时承载实时 `/v1/metrics`：解码速度、常驻内存、专家/前缀缓存命中率、TTFT 的四分钟趋势图；还有一个 Prometheus 指标浏览器：列出 `/v1/metrics` 全部序列的
HELP/TYPE 与当前值，可按名筛选、点选任意指标（最多 4 个）为它单独画趋势，计数器还能切成每秒
增量；再加 `cwick:model_info` 的全部标签、`cwick:model_loaded` 实时状态位、生成参数自定义、
只读运行配置，以及本地模型切换（本构建无法装载的条目标为不支持且不可选）。所有请求都是同源的，**没有**把控制台指向外部 OpenAI 兼容地址的选项；仍可签发具名 API key 供其他客户端接入本机 cwick。

**模型**标签页列出 `./models` 下已下载的权重、正在下载的任务（进度、速率、ETA），并提供模型目录的只读检索。下载由服务端后台执行：可暂停、继续、失败重试、取消；中断留下的部分文件在重启后用 Range 请求续传。单分片超过 100 GiB、整个仓库几百 GiB 全程用 64 位尺寸处理。五个接口与落盘布局见 `docs/SERVER.zh-CN.md`。

市场里随 `apps/` 带两个 H5 应用，各自有 sqlite 库和 manifest。**井字棋**你执 X 先手，机器每一手都把棋盘与各空格的战术事实交给 `POST /v1/systemone`，问「落在哪一格赢的概率最大」，右侧栏画出每个空格的概率读数与双方的对弈过程。**3D 卡丁车**在莫比乌斯环上跑圈，模型实时随机决策在前方生成路障：每次重新洗牌 12 个选项、换随机种子与加压点，目标是把跑圈难度拉满；右侧栏分别是 AI 路障统计与卡丁车统计。对话页走面板上**接口**选中的那个端点；控制台只负责托管，每个应用在手机屏和桌面屏上各自排版。开 `--lan` 后每个应用有二维码，同 Wi-Fi 的手机不用知道地址就能打开。

模型被问的是**决策**而不是文案；每个应用把自己那份状态（一局棋、一份决策日志、一圈成绩）写进 `--store` 下自己的 sqlite 文件。

`--password`（或 `--lan`，它要求必须有密码）打开单用户登录：PBKDF2 哈希、Bearer token、HttpOnly session cookie。给第三方 Agent 平台用的话，`--api-key NAME` 签发一个具名 key（创建时打印一次，只存哈希），可用 `DELETE /v1/auth/keys` 吊销。

### 并发与排队

生成是单槽的：一次只答一个，因为两个会互相腰斩带宽、谁也跑不完。其余一切都是并发的——一个慢生成不会挡住 `/v1/metrics`、`/v1/models`、日志流或 web 控制台，这也是"能看着一次 30 分钟的 prefill"成立的前提。

请求按到达顺序排队，每个都跑到完成。排队中已断开连接的客户端会被检测并移出队列，而不是先烧掉一次 prefill；正在换模型时返回 503 带 `Retry-After`，关闭服务时同理，所以调用方总能知道该不该重试，而不是对着超时猜。

## 配置

多数开关是 `serve`（或 `run`）的参数；常驻服务常用的那几个还有环境变量孪生：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--root DIR` | — | 模型根目录：`config.json` 加按发布原样的分片 |
| `--host` / `--port` | `127.0.0.1` / `8000` | `start.sh` 绑 `0.0.0.0:8001` |
| `--context N` | 131072 | KV 容量（token，>100K）；`run` 默认 4096。`start.sh` 用 500000，MiniMax 用 32768 |
| `--max-tokens N` | 1000000 | 请求没带 `max_tokens` 时的默认值，会被窗口裁到剩余空间；`start.sh` 传 8192 |
| `--backend metal\|cpu` | Apple Silicon 上为 metal | 专家 GEMV 与批式投影走 GPU |
| `--mem-fraction PCT` | 80 | 本进程可占用的宿主内存比例 |
| `--max-memory GiB` | — | 绝对上限，更小时优先 |
| `--allow-swap` | 关 | 不再锁 wired（只用于调试放不下的计划） |
| `--prefix-cache-dir` / `--prefix-cache-gib` | 关 / 8 | 磁盘上的 KV 快照及其预算 |
| `--warmup N` | 0 | 开端口前先跑 N 个 token，把冷启动付掉 |
| `--verbose` / `--trace` | 关 | 每步细节 / 把 prompt 与输出打到 stdout |
| `--web` / `--lan` | 关 | 内置控制台 / 绑 `0.0.0.0` + 二维码分享 + 强制登录 |
| `--jev` / `--no-jev` | 关 | 打开 `POST|GET /v1/systemone`；`start.sh` 已打开 |
| `--password P` / `--api-key NAME` | — | 单用户登录 / 给 Agent 平台的具名 key |
| `--models-dir DIR` | — | 额外列出（最多 8 个）可切换的 checkpoint，一次只加载一个 |
| `--store DIR` | `~/.cwick/apps` | 每个应用一个 sqlite，外加 `auth.db` |

环境变量孪生：`CWICK_MEM_PCT`、`CWICK_MEM_MAX_GIB`、`CWICK_STORE`、`CWICK_PASSWORD`、`CWICK_API_KEY`、`CWICK_WARMUP`、`CWICK_WEB`、`CWICK_METAL`、`CWICK_SEED`（采样种子，让一次运行可复现）、`CWICK_HUB_ENDPOINT` / `HF_ENDPOINT`（**模型**标签页的目录服务地址）。在线下载一律写入 `./models`，该目录已在 `.gitignore` 中。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/v1/models` | 常驻模型，加上 `--models-dir` 下找到的模型；每条带 `supported` / `unsupported_reason` |
| POST | `/v1/models/switch` | 先校验目标，再卸一个、映射另一个；本构建无法装载的架构返回 400（常驻模型不动），生成或排队中返回 409 |
| POST | `/v1/completions` | 纯文本补全，流式或非流式 |
| POST | `/v1/chat/completions` | messages、`stream`、`tools`、`reasoning_effort` |
| POST | `/v1/responses` | Responses 方言：输入 `input` items，输出 `response` 对象 / `response.*` 事件 |
| POST | `/v1/messages` | Anthropic 方言：输入 `system` + content blocks，输出 `message` 对象 / `message_start…message_stop` |
| POST | `/v1/systemone` | 一次 prefill 读出类型化决策（choice / score / noul），需要 `--jev`；也可用 `/inference/v1/systemone` |
| GET | `/v1/systemone` | 当前决策适配器：家族、prompt 语法、可读字母、各项上限 |
| GET | `/v1/hub/models` | `./models` 下已下载的 checkpoint，含大小与完整状态 |
| DELETE | `/v1/hub/models/<name>` | 删除已下载的 checkpoint（递归，限制在 `./models` 内）；常驻模型或正在下载时返回 409 |
| GET | `/v1/hub/downloads` | 下载任务实时状态：进度、速率、ETA、错误 |
| GET | `/v1/hub/search?q=…` | 模型目录只读检索（默认 hf-mirror） |
| POST | `/v1/hub/download` | 启动/续传 `{"repo":"owner/name"}` 到 `./models/owner/name` |
| POST | `/v1/hub/download/<id>/<动作>` | `pause` / `resume` / `retry` / `cancel` |
| GET | `/v1/metrics` | 也可用 `/metrics` 与 `/health`；vLLM 兼容命名 |
| GET | `/v1/config` | 设置面板读写的配置 |
| POST | `/v1/auth/login` | 密码 → Bearer token（12 小时）+ `cwick_session` cookie |
| POST/GET/DELETE | `/v1/auth/keys` | 签发、列出、吊销 API Key（需要登录会话） |
| GET | `/v1/logs`、`/v1/logs/stream` | 本进程的 stdout/stderr，尾部或跟随 |
| GET | `/v1/store` | 应用存储布局、文件与行数 |
| PUT/POST/GET | `/v1/apps/<id>/{state,events,games}` | 每应用 sqlite，清浏览器数据不会清掉历史 |
| GET | `/v1/share/qr` | 局域网分享用的二维码 |

完整参考——鉴权规则、错误结构、存储表、应用端点——在 [docs/SERVER.zh-CN.md](docs/SERVER.zh-CN.md)。

### 决策：`POST /v1/systemone`

`--jev` 给常驻模型加了第二种用法。不再让它写出一段答案，而是读答案位置的分布，直接返回
类型化的结果：选了哪个选项、每个选项概率多少、这次判断有多集中。一个问题一次 prefill，
不生成任何 token。

```bash
curl -s http://127.0.0.1:8001/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"我的订单到货时已经破损，我今天就要换货。",
       "questions":{"intent":{"type":"choice","instructions":"客户想要什么？",
                              "criteria":{"refund":"退款","replace":"换货",
                                          "info":"只是问信息"}}}}'
```

三种题型：`choice` 返回选中项和全部分布；`score` 返回概率加权的档位索引与 legend；
`noul` 返回"陈述成立"的概率（两侧由 `criteria` 的 `{"true": …, "false": …}` 指定）。
`state` 可以是字符串，也可以是任意 JSON 值；每个问题的 `instructions` 必填。一次请求
最多十个问题，每个问题最多 26 个选项。对外发布的路径是 `/inference/v1/systemone`，
`/v1/systemone` 是短写法。

**它跟着模型走。** 每个受支持的 checkpoint 有自己的 prompt 语法与答案拼写：GLM 与
DeepSeek 用"思考块已关闭"的那一轮（`glm-chat-closed`、`deepseek-chat-closed`），MiniMax
用它的普通对话轮（`minimax-chat-plain`）。适配器在 tokenizer 加载时解析一次，之后每次
`POST /v1/models/switch` 都重新解析——切换模型就是切换这个端点底下的接口，不存在"拿
MiniMax 的 prompt 去读 GLM logits"的路径。`GET /v1/systemone` 公布当前生效的那一个；
`make jevprobe` 只读 tokenizer、不加载权重，逐字母打印 token 表（三个自带 checkpoint
都是 26/26、带前导空格）。

请求里写了服务端已不再服务的模型名，会被 400 拒绝并点名当前常驻模型。这是有意的：一个
跨过一次切换缓存了模型 id 的应用，不该拿到新模型的决策却挂在旧模型的名字下。

数字的含义比 JSON 形状更重要。概率是模型自己的下一个 token 分布，只截取到选项字母上——
它**没有校准**；`confidence` 是集中度（0 = 所有选项等概率，1 = 全压在一个上），不是
"答对的概率"。分布很平时，那就是模型对这个问题的诚实回答，所以返回整个分布而不是只返回
argmax。

成本是每个问题一次 prefill；同一问题的所有选项共用那一次前向，所以九宫格和两选一的代价
一样。这台机器上，MiniMax 上 550 token 的一次决策是 95.5 s，320B 权重上一个三问请求是
306 s——几乎全部都是 prefill，因为这个端点不生成 token，也没有别的东西可以摊薄。

**一个请求里的问题共用同一段 prefill。** 每个问题都以同样的 `system` + `<state>` 那一轮开头，
所以端点先把所有问题编码一遍，取 token 级最长公共前缀，只 prefill 一次，再从这里续着算每个
问题。三个问题共享 348 token 前缀（输入 1145 token）时实际只 prefill 449：
**213.8 s → 102.0 s（2.10×）**，每个概率都与不复用的那一轮逐位相同。

续算只在能精确复现的那个后端上打开，这是量出来的边界而不是策略：MiniMax 上原地续写把分布
复现到 `|dp| = 0.0`，而在 GLM 与 DeepSeek 上强行打开会让某个概率偏移 0.41 和 0.56、并改变
答案（压缩 KV 与未填满的 attention 池不是只按位置寻址的）。这两个后端仍然每个问题一次完整
prefill；`usage` 照样报 `input_tokens`、`cached_input_tokens`、`prefilled_tokens`、`reused_tokens`，省了多少——或者
一点没省——是每个请求可见的，而不是靠推断。

两个内置应用都使用这个接口：井字棋按「哪个空格赢的概率最大」在空格上做一次选择，取概率最高的落子；卡丁车的路障来自「种类 × 车道」的 12 选 1，每次请求的选项顺序都会重新洗牌。没有一个在生成需要浏览器解析的对象，每个都保留本地兜底，用于被拒或过慢的决策。

## 模型、上下文与内存

| 模型 | 上下文（`start.sh`） | KV 缓存 | 每步专家 | decode |
| --- | --- | --- | --- | --- |
| GLM-5.3-flash | 500K | MLA latent + DeltaNet 状态 | 8/288 × 42 层 | 3.6 tok/s 常驻，1.7 流式 |
| DeepSeek-V4.1-Flash-1017 | 500K | 压缩 MLA | 6/384 × 40 层 | 2.8 tok/s |
| MiniMax-M3.1-Flash | 32K | bf16 GQA，2.3 KiB/位置/层 | 4/128 × 57 层 | 1.48 tok/s |

上下文不抽象，也不免费：MiniMax 上 500K 位置就是 64 GiB 的 KV，直接从专家缓存里出，也就是直接从 token 速率里出。这就是 `start.sh minimax` 把它压在 32K 的原因——量过，不是拍脑袋。

内存计划在启动时做一次决定：**RAM 的 80% 是天花板**，常驻 spine 和 KV 缓存先从这里出，剩下的全给专家 pin 缓存。`--mem-fraction` 改这个比例，`--max-memory GiB` 直接给绝对值——留给"机器上还有别的进程要内存"这种情况。逼近天花板时 runtime 会回收缓存，并带冷却，避免阀门一路塌到零。要点是稳态 footprint 保持在天花板之下、`swap` 不动、压缩器不必攥着工作集——踩到被压缩页的一步，和踩到 SSD 的一步看起来一模一样，也一样慢。

正因为如此，pin 缓存**不会**因为装得下就填到天花板：过了大约 RAM 的四分之一，压缩器开始收页，命中率还在涨而整个步反而变长，所以上限是 `host / 6`，背后的实测数字在 [docs/DECISIONS.zh-CN.md](docs/DECISIONS.zh-CN.md) §27。

只待在天花板之下还不够，所以 spine 与专家 arena 默认**锁成 wired**：VM 既不能压缩也不能换出每个 token 都要读的那些字节。`--allow-swap` 可以关掉它，那是用来调试"计划放不下"的，不是用来长期跑的。

## 长 prompt 与前缀缓存

Agent 会话每轮都重发同一段几千 token 的系统提示。重新 prefill 它**就是**首 token 时延，所以引擎在每个 prompt 结束时把推理状态快照下来，下一个请求开头相同时直接复用。同一个 403 token 的 prompt，跑两次：

| 请求 | 耗时 | 发生了什么 |
| --- | --- | --- |
| 第一次（冷） | 169.4 s | 完整 prefill，然后 `prefix cached: 403 tokens (83 ms)` |
| 第二次（完全相同） | **9.95 s** | `prefix cache hit: 402 of 403 prompt tokens replayed` —— 这 9.95 s 是 16 步 decode |

快照是写进 `--prefix-cache-dir` 的推理状态 **checkpoint**，不是文本缓存：29,798 token 的 prompt 约 0.9 GB KV，读回约 0.3 s，对应的是几个小时的 prefill。淘汰策略刻意不是 LRU：会话的 prompt 每轮都在变长，按时间淘汰会把每轮都还要用的那段短基础提示扔掉。排序依次是"服务过多少次复用、长度、时间"。

## 思考、流式与工具

思考默认打开，深度是请求字段 `reasoning_effort`，映射到一个 token 上限，而**引擎自己执行**这个上限：到达上限时引擎把闭合标记同时写进输出和上下文，所以预算用尽的这一轮仍然给答案，而不是返回 `content=""`。

| 档位 | 关 | 低 | 中 | 高 | 超高 | 极致 |
| --- | --- | --- | --- | --- | --- | --- |
| 预算 | 0 | 512 | 2048 | 8192 | 16384 | 32768 |

`reasoning_content` 与 `content` 在流式和非流式两条路径上都分离，且来自同一个判断——checkpoint 自己的推理标记（GLM 与 DeepSeek 是 `<think>`，MiniMax 是 `<mm:think>`），不是写死的字符串。工具声明被渲染进各 checkpoint 自己的 `<tools>`/`<tool_call>` 协议，再解析回 OpenAI `tool_calls`；工具调用正在生成时，流被标成 `decode(tools)`，而不是漏进正文。

## 速度

在这台 M3 Max、128 GiB、`--backend metal`、某个 commit 上测得——不是每个后续改动的重测。条件比数字重要："常驻"是一次回复需要的专家已经 pin 在内存里，"流式"是同一个请求从 SSD 读，同一个二进制上两者差 2×。MiniMax 的两行比其余的新：decode 是 [docs/DECISIONS.zh-CN.md](docs/DECISIONS.zh-CN.md) §27 那轮改动后的默认 21 GiB 缓存，prefill 是 §44 的决策路径改动（1024 token 一块）。

| 模型 | 阶段 | 耗时 | tok/s | 备注 |
| --- | --- | --- | --- | --- |
| GLM-5.3-flash | decode，专家常驻 | 275 ms | **3.63** | 10 token 的回复 |
| GLM-5.3-flash | decode，流式 | 590 ms | 1.69 | 83 GiB 缓存，80 步 |
| GLM-5.3-flash | prefill，403 token 冷启动 | 169 s | 2.4 | 第二次同样请求 9.95 s |
| DeepSeek-V4.1 | decode | 360 ms | 2.78 | 内存重构后的稳态 |
| MiniMax-M3.1 | decode | 677 ms | **1.48** | 23 步，默认 21 GiB 缓存 |
| MiniMax-M3.1 | prefill，550 token 的一次决策 | 95.5 s | **5.8** | 1024 token 一块；512 时是 109.3 s |

### 什么决定了 token 速率

三个数字差得远，原因只有一个：每步要搬的字节数不同。测法与上表相同：

| 模型 | 最好 decode | 每步字节 | 天花板由什么决定 |
| --- | --- | --- | --- |
| GLM-5.3-flash | 3.63 tok/s | 每步约 8.06 GiB 专家 | 专家库装不进 RAM |
| DeepSeek-V4.1 | 2.78 tok/s | 约 18 GB，大多是 bf16 稠密 | 专家 fp4 GEMV 实测 58.5 GB/s |
| MiniMax-M3.1 | 1.48 tok/s | 约 5 GiB 未命中 | 同 shard 读天花板；缓存超过 30 GiB 会喂给压缩器 |

限制是算术，不是少了一个优化：一步要搬 20–30 GB，这台机器最高读 123 GB/s，而引擎一次权重读取只产出一个 token（完整推导在 [docs/PERF.zh-CN.md](docs/PERF.zh-CN.md)）。同一条闭环已经拿到的是 prefill 这一侧——MiniMax 上 550 token 95.5 s（5.8 tok/s），一个三问决策共享前缀后 102 s 而不是 214 s——以及一份清楚的剩余方向：每步字节更少，或一次权重读取产出多个 token。

## 验证

Beta，且变化很快。每个改动都先在这台机器上量过才留下，输掉的那些写在 [docs/DECISIONS.zh-CN.md](docs/DECISIONS.zh-CN.md) 里——包括那些看起来显而易见的。

```sh
make test        # 651 项断言：引擎、采样、前缀缓存、服务端、web UI、应用
make bench       # 不加载 checkpoint 也能测内核
```

正确性是锚定的，不靠肉眼：固定 prompt 必须产出固定的 top-1 token id 与 logit；每个内核改动都与参考实现互检到 1e-5（fp8 分块 GEMM 实测 8.3e-6，FP4 分组核与单行核逐位相同）。`cwick doctor` 会读一遍每个常驻字节加一段专家库抽样，所以坏分片在请求之前就被发现；`/v1/metrics` 发布的就是日志里打的同一份状态。

## 详细文档

深度文档都在 `docs/` 下，每一篇的第一行都可以在
[English](docs/ARCHITECTURE.md) 与[简体中文](docs/ARCHITECTURE.zh-CN.md)之间切换：

- [运行时架构与组件地图](docs/ARCHITECTURE.zh-CN.md)
- [有界内存流式与存储分层](docs/STORAGE.zh-CN.md)
- [逐相实测性能](docs/PERF.zh-CN.md)
- [决策记录：试过什么、否决了什么、为什么](docs/DECISIONS.zh-CN.md)
- [OpenAI 兼容服务端参考](docs/SERVER.zh-CN.md)——API、认证、应用、存储
- [DeepSeek 后端](docs/DEEPSEEK-PORT.zh-CN.md)
- [精度与各 checkpoint 的量化](docs/PRECISION.zh-CN.md)
- [验证套件](tests/test_cwick.c)

## FAQ

- **头几个 token 慢，日志里命中率很低。** 新进程的专家 arena 是空的；`--warmup 32` 可以在开端口前先把它付掉（本机构建上 1013 → 590 ms/step，17% → 68% 命中）。
- **每轮都在重新 prefill 同一段 prompt。** 说明没复用上：确认 `--prefix-cache-dir` 开着，并且 prompt 真的从同样的 token 开始（系统提示或时间戳变了，前缀就断了）。
- **`swap` 一直涨、机器发烫。** 有别的东西占了内存：把 `--mem-fraction` 调小，或设 `--max-memory`。不要在台 Mac 上跑两个 cwick 进程，即使 `CWICK_ALLOW_MULTI=1`——它们会互相把页挤出去，两边都慢 2–3 倍。
- **请求返回 503 带 `Retry-After`。** 正在换模型或正在关闭。按 header 里的时间重试；排队不是错误，最终会被服务。
- **Metal 打了一行警告，输出一样但更慢。** 没找到可用的 GPU，cwick 宁可回退到 CPU GEMV，也不让一次生成失败。`--backend cpu` 是同样的事，只是故意的。
- **`doctor` 要跑好几分钟。** 它会读完整段常驻 spine 并抽样专家库；`--deep` 则是全读而不是抽样。

## 目录结构

```text
cwick/
├── cwick              构建出的二进制
├── src/               引擎本体——C11，无第三方代码
│   ├── main.c         CLI：setup | run | serve | doctor，以及参数解析
│   ├── server.c       HTTP、SSE、队列、鉴权、指标、应用存储路由
│   ├── engine.c       请求 → 渲染 prompt → prefill/decode → 流式输出
│   ├── spine.c        常驻权重、内存计划、专家 pin 缓存
│   ├── stream.c       F_NOCACHE SSD 流式、淘汰、预取
│   ├── attention.c    MLA / GQA / DeltaNet 内核（NEON）
│   ├── moe.c          路由与专家 GEMV/GEMM
│   ├── ds_*.c         DeepSeek-V4.1 后端：FP4 专家、FP8 attention
│   ├── mm_*.c         MiniMax-M3.1 后端：NVFP4 专家、bf16 GQA
│   ├── metal*.c/.m    Metal 内核与零拷贝 arena
│   ├── pfx.c          磁盘上的前缀缓存快照
│   ├── jev.c          决策端点：三种 prompt 语法、字母读出、请求内共享前缀
│   ├── hub.c          模型库的下载任务与目录检索
│   ├── store.c        每个应用一个 sqlite
│   └── webui*.c       控制台，构建时嵌入
├── web/               控制台源码：index.html、app.js、app.css、login.html
├── apps/              两个 H5 应用与 manifest.json
├── models/            在线下载的权重（.gitignore）
├── docs/              架构、存储、性能、决策、服务端、精度
├── tests/             test_cwick.c——651 项检查
├── tools/             独立的 C 探针：内核、分层测试、模型运行器
├── start.sh           推荐入口（前缀缓存 + web + 局域网 + 密码）
└── download_model.sh  可断点续传的 checkpoint 下载
```

## 许可证

MIT，见 [LICENSE](LICENSE)。
