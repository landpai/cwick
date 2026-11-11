# CWICK — 开发交接说明

<p align="center"><a href="AGENTS.md">English</a> · <a href="AGENTS.zh-CN.md">简体中文</a></p>

本文档供 AI Agent 或开发者在**新设备上 clone 本仓库后**快速理解现状并继续开发。
面向使用者的说明见 [README.md](./README.md)（English）与 [README.zh-CN.md](./README.zh-CN.md)。
深度文档在 `docs/`：每篇都有英文 `X.md` 与中文 `X.zh-CN.md`，第 3 行互相切换。

## 1. 项目是什么

在一台 **Apple Silicon Mac（128 GiB 为设计点）** 上跑**完整、未剪枝**的
GLM-5.3-flash（≈306 GiB）/ DeepSeek-V4.1-Flash-1017（≈475 GiB）/
MiniMax-M3.1-Flash（≈233 GiB）：一个 C 二进制，按发布原样读 checkpoint
（不转换、不剪枝、top-k 不降），内置 OpenAI 兼容服务端 + 单页控制台 + 模型库 + 2 个 H5 应用。
权重留在 SSD 上流式读，每个 token 真正过总线的只有几 GB —— 这就是项目名"灯芯"的意思。

产品由两件事定义，而且这两件都是硬约束而不是愿景：checkpoint **整份跑**（不剪枝、
不降 top-k、不降精度）在一台 128 GiB 的机器上；常驻模型**有两种用法**——OpenAI 兼容
服务端，以及 `POST /v1/systemone`，它从一次 prefill 的分布里读出类型化决策，于是同一份
权重不必重训、不必换调用栈就成了一个基于概率的决策服务。

本机实测 decode 是 3.63 / 2.78 / 1.48 tok/s（`--backend metal`，走完整 API 请求）。
拉开差距的是每步字节数对上 123.5 GB/s 的内存读上限，不是少了一个优化——数字与推导都在
`docs/PERF.zh-CN.md`。

## 2. 硬约束（动手前先读，这些不是建议）

- **只用 C**。C11 + 一个 Objective-C 文件（`src/metal_objc.m`，Metal 桥）。链接的
  只有 macOS 系统框架（Accelerate / Metal / CoreImage / ImageIO / CoreGraphics）
  与系统 `sqlite3`（`Makefile:15-16`）。不引入第三方库、不用 Python、没有权重转换步骤。
  一次性验证工具也写纯 C（`tools/*.c`），shell 只负责驱动编译与运行。
- **只有三条模型路径，不要造第四条**：
  - GLM-5.3-flash：`spine.c`（常驻权重/内存计划）、`engine.c`（请求→prefill/decode）、
    `attention.c`（MLA/GQA/DeltaNet）、`moe.c`（路由 + 专家 GEMV/GEMM）；
  - DeepSeek-V4.1：`ds_model.c` / `ds_engine.c` / `ds_ops.c` / `ds_quant.c` / `ds_prompt.c`；
  - MiniMax-M3.1：`mm_model.c` / `mm_engine.c` / `mm_prompt.c`（+ `mm.h`）。
  接一个新模型 = 新加载器 + 新反量化 + 新 prompt 模板 + 新内核，**不是改一个 config**。
- **质量铁律**：top-k 取 checkpoint 自己的 `num_experts_per_token`，专家不剪、
  embedding 不共享、缺分片 fail closed（宁可起不来，也不悄悄跑小一号的模型）。
- **文档口径**：注释与文档只写机制和实测数字，**不写参考过哪些工程**。
  `goal.md`「文档规范」一节把这当作硬约束，代码注释、`docs/`、`README` 三处都算。
- **文档双语**：`docs/X.md`（英文）+ `docs/X.zh-CN.md`（中文），第 3 行是同一行切换器；
  `README.md` / `README.zh-CN.md` 同理。新增文档要么同时给两版，要么别加。

## 3. 当前实现状态

### ✅ 已落地

- 三个 checkpoint 端到端：加载 + `/v1/models` + 生成 + 切换，逐位验证过
  （DeepSeek 全链路 0 误差；GLM 走 `doctor` 结构校验，见 `docs/PRECISION*.md`）。
- OpenAI 兼容 API：`/v1/models`、`/v1/models/<id>`、`/v1/completions`、
  `/v1/chat/completions`（SSE、`reasoning_content` 与 `content` 分离、
  `tool_calls` 按 checkpoint 自己的协议生成再解析回 OpenAI 形状）。
- Responses API（`POST /v1/responses`）：`input` items（message / function_call /
  function_call_output、扁平的 `function` tools）先翻译成服务端早就在校验的 chat 请求，
  再按 `response` 对象或 `response.*` 事件回答。是一个适配器而不是第二个引擎——prompt、
  队列、前缀缓存与思考预算完全复用。没有持久化存储：`previous_response_id` /
  `conversation` 一律按名拒绝。代码在 `server.c`（`responses_parse`、`responses_body`、
  `resp_stream_*`），测试钉住信封、item 转换与拒绝路径。
- Anthropic Messages API（`POST /v1/messages`）：`system`（字符串或 block 数组）、
  content blocks（`text` / `thinking` / `tool_use` / `tool_result`）、带 `input_schema`
  的扁平工具、`thinking.budget_tokens` → 思考强度，流是
  `message_start → content_block_* → message_delta → message_stop`，没有 `[DONE]`。
  同样的适配器规则：`anthropic_parse` 重建 chat body 后交给 `validate_body`，
  `messages_body` / `msg_stream_*` 只写信封。错误用 `{"type":"error","error":{…}}`；
  `x-api-key` 与 `Authorization: Bearer` 等效。按名拒绝：`stop_sequences`、
  图像/文档 block、`tool_choice.any`。
- 决策接口（`--jev`）：`POST /v1/systemone`（也可用 `/inference/v1/systemone`）一次
  prefill 读选项字母的分布，不生成 token；两条路径的 `GET` 报告实时适配器（家族 /
  prompt 语法 / 可读字母数）。适配器在 tokenizer 加载与每次 `POST /v1/models/switch`
  后重新解析，所以切模型同时切接口；请求里写过期模型名会被 400 点名拒绝。
  `make jevprobe` 只读 tokenizer 打印字母表。报文格式按发布协议：`state` 可以是字符串
  或任意 JSON 值，`noul` 的 `criteria` 指定陈述两侧（真的一侧永远在前），`instructions`
  必填，`stream`/`reasoning_effort`/`thinking`/`images` 一律点名拒绝，状态码区分
  422（根本不是决策请求）与 400（合法但不提供）。十个问题与 26 个选项是有意保留的差异
  （`docs/DECISIONS.zh-CN.md` §48）。
- 决策 prefill（`jev.c` + `engine.c`，`docs/DECISIONS.zh-CN.md` §44）：一个请求里的问题
  先全部编码，取 token 级最长公共前缀只 prefill 一次，每个问题从那里续算。MiniMax：
  输入 1145 token → 实际 prefill 449，**213.8 → 102.0 s（2.10×）**，每个概率逐位相同；
  `usage` 报 `input_tokens` / `cached_input_tokens` / `prefilled_tokens` / `reused_tokens`。
  **复用只在 MiniMax 上打开**——在另两个后端强开，答案会变（§44 的表）。
  真正见效的是抬高块上限（550 token：109.3 → 95.5 s）；把 prompt 均匀切两半反而慢 4%，
  那个变体留在 `CWICK_JEV_CHUNK_EVEN=1` 后面。
- 上下文：`--context` 是 KV 容量，`serve` 默认 131072，`start.sh` 对 GLM/DeepSeek
  用 500000、MiniMax 用 32768（它的 KV 是 2.3 KiB/位置/层，500K 要 64 GiB）。
- 思考强度：关/低/中/高/超高/极致 → 0/512/2048/8192/16384/32768，
  **上限由引擎自己执行**（到顶时把闭合标记写进输出和上下文）。
- 前缀缓存（`pfx.c`）：每个 prompt 结束把推理状态快照到 `--prefix-cache-dir`，
  下轮同前缀直接回放。403 token 的 prompt：冷 169.4 s → 热 9.95 s。
- 内存计划（`spine.c` + `stream.c`）：上限默认 80% RAM，`--mem-fraction` /
  `--max-memory` 可改；spine 与专家 arena 默认 **wired**（`--allow-swap` 退出）；
  逼近上限时按冷却淘汰，不每步 trim。
- 服务端（`server.c`）：单槽 FIFO 队列 + 其余 endpoint 全并发；换模型/关闭时
  503 + `Retry-After`；排队中断连会被移出队列（不白烧一次 prefill）。
- 日志：stdout 一事件一行，带墙上时间与请求号；prefill 按层报进度（块大小
  `MM_CHUNK` / `DS_CHUNK` = 1024，典型决策 prompt 就是**一块**，满块在 60 层上要
  140–200 s，所以必须按层打，否则单块 prompt 几分钟 0 行），decode 分
  `decode(thinking)` / `decode(content)` / `decode(tools)`。
- `/v1/metrics`：vLLM 命名，槽位忙时照样能读。
- 控制台（`webui.c` + `src/webui_embed.c`）：对话 / 应用 / 模型 / 日志 / 设置五个 tab；
  对话页**三种方言都支持**（`/v1/chat/completions`、`/v1/responses`、`/v1/messages`，
  由面板上的 `接口` 下拉逐轮选择），模型页驱动 `/v1/hub/*`。三种请求/响应的映射都是
  `web/app.js` 里的纯函数（`respInput` / `respEventToChat` / `respFullToChat`、
  `msgMessages` / `msgEventToChat` / `msgFullToChat`），收敛在 `postChat`：
  三条路径交给下面控件的事件形状相同，**不要再加第四套渲染**。设置 = 原控制台合并进来：
  实时指标 + 四张 canvas 趋势图（解码速度/常驻内存/两级缓存命中率/TTFT）、
  Prometheus 指标浏览器（HELP/TYPE、名称筛选、任选 4 个序列各自成图、原始值或每秒增量）、
  `cwick:model_info` 全标签、生成参数、只读运行配置、本地模型切换。全部同源，
  **不能**把控制台指向第三方 OpenAI 地址；`--lan` 给每个应用二维码。
- 应用市场：`apps/` 下井字棋（人先手，问「哪格赢的概率最大」，右侧栏画出各格读数）与
  3D 卡丁车（莫比乌斯环，实时洗牌的 12 选 1 决策以难度最大为目标，右侧栏是路障与车身统计），
  各自一个 sqlite（`store.c`，`--store DIR`，默认 `~/.cwick/apps`），两者都通过
  `/v1/systemone` 决策；状态在服务端而不是 localStorage。
- 模型库（`hub.c`）：权重落 `./models/<owner>/<name>/`，`.incomplete` + Range 断点续传，
  后台任务表 + 目录只读检索；五个 `/v1/hub/*` 端点都在生成槽位之前应答。
  `DELETE /v1/hub/models/<name>` 删除已下载 checkpoint（递归、realpath 限制在库根内、
  拒绝常驻模型与正在下载的目录），同一请求内把 catalog 条目一并移除。
  `CWICK_HUB_ENDPOINT` / `HF_ENDPOINT` 覆盖目录服务地址；启动时扫描 `./models` 进 catalog。
- 鉴权：PBKDF2 单用户登录 + Bearer + HttpOnly cookie；`--api-key NAME` 给
  Agent 平台签发具名 key（只存哈希），`POST|GET|DELETE /v1/auth/keys` 管理。
- 热切换：`POST /v1/models/switch` 先做只读的模型契约 dry-run，通过后才释放旧模型。未知架构或
  config 不合法返回 400 且常驻模型不动；生成或排队中返回 409。`/v1/models` 与
  `/v1/hub/models` 都报 `supported`（发现目录时算好缓存，轮询不会在生成锁内读文件），
  `cwick:model_loaded` 是实时 1/0 指标。
- `make test`：**651 项断言，当前全 PASS**（`./tests/test_cwick` 每项一行 `ok:`）。
- `cwick doctor`、`download_model.sh`（断点续传、>100 GiB、一行进度）。

### ⚠️ 已知但还算能用

- **decode 在三个模型上都是带宽受限**：GLM 3.63 / DeepSeek 2.78 / MiniMax 1.48 tok/s
  （最后一个是 `docs/DECISIONS.zh-CN.md` §27 那轮改动之后的数字）。挡路的是每步字节数
  对上同 shard 读天花板，不是缓存没做好 —— 动之前先读那一节和 `docs/PERF*.md` 的 roofline。
- `cwick doctor` 的张量清单按 `model.language_model.*` 命名写死（KDA / DSA / MLA
  三个分支都有）。换成别的前缀命名（MiniMax 就是），会打一串 `FAIL missing` ——
  那是**清单不匹配，不是权重坏了**；判定以 `serve` 能否加载 + `make test` 为准。
- 队列没有深度上限（`slot_acquire()` 不拒绝），`429` 只在关闭等极端路径可达。
- `/v1/metrics` 里没有 swap / 压缩器指标（vLLM 的 `num_requests_swapped` 是请求
  抢占，跟 OS swap 无关，别照抄）。
- MiniMax 走 CPU 线程池，不是 Metal —— GPU 在这些形状下实测更慢，别再"顺手接上"。
- **GLM 与 DeepSeek 不能原地续算 prefill。** 强开会让决策概率偏移 0.41（GLM）/
  0.56（DeepSeek），三个问题里变 1–2 个答案：压缩 KV 与未填满的 DSA 池不是只按位置
  寻址的；GLM 走快照回放好一些，也还有 `|dp| = 0.11`。所以 `/v1/systemone` 只在
  MiniMax 上共享前缀，另两个后端每个问题付一次完整 prefill。让它们变精确是下一项
  工程工作，不是一个开关。

### ⬜ 没做 / 明确不做

- 在这台硬件上把 decode 再往上抬：失败的杠杆都在 §7，剩下的是架构性的——每步字节更少，
  或一次权重读取产出多个 token。
- DeepSeek 的专家缓存 **不要** wire：实测更慢（wire 之后命中率不变、步时变长）。
- MiniMax 上 GPU（要先把 GPU 侧的访存模式改掉，不是换个 backend 开关）。

## 4. 目录速览

```text
src/
  main.c           CLI：setup | run | serve | doctor，参数解析，内置单实例锁
  server.c         HTTP/SSE、队列、鉴权、metrics、日志流、应用存储路由
  engine.c         请求 → 渲染 prompt → prefill/decode → 采样 → 流式
  spine.c          常驻权重、内存计划、专家 pin 缓存
  stream.c         F_NOCACHE SSD 流式、淘汰、预取、冷却阀门（steer_memory）
  attention.c      MLA / GQA / DeltaNet 内核（NEON）
  moe.c            路由 + 专家 GEMV/GEMM
  sample.c         采样（t<=0 argmax；top_k → top_p → 抽样，HF 顺序）
  pfx.c            前缀缓存快照（磁盘 KV checkpoint）
  models.c         模型发现 + 对外别名（glm-5.3-flash / deepSeek-v4.1-flash-1017 /
                   MiniMax-M3.1-Flash）
  tokenizer.c      BPE；log.c 日志；logcap.c 环形缓冲（/v1/logs）
  mem.c / pool.c   内存与 NEON 线程池；prof.c 分相计时（CWICK_PROFILE=1）
  metal*.c/.m      Metal 内核与零拷贝 arena
  ds_*.c           DeepSeek 后端（FP4 专家、FP8 attention、prompt 模板）
  mm_*.c           MiniMax 后端（NVFP4 专家、bf16 GQA、<mm:think> 标记）
  jev.c            决策端点：按家族选 prompt 语法、字母读出、请求内共享前缀
  hub.c            模型库：可续传下载（libcurl）+ 目录检索
  metrics.c        /v1/metrics 背后的 vLLM 命名计数器
  store.c          每应用一个 sqlite；auth.c API key/会话；qr.m 二维码
  webui.c          控制台后端；webui_embed.c 由 Makefile 生成（勿手改）
web/              控制台源码：index.html / app.js / app.css / login.html
apps/             两个 H5 应用 + manifest.json
models/           在线下载的权重（gitignore，启动时扫描）
docs/             架构、存储、性能、决策、服务端、精度、DeepSeek 移植（中英各一份）
tests/test_cwick.c 651 项断言
tools/            独立 C 探针：内核基准、分层测试、模型运行器
start.sh          推荐入口：前缀缓存 + web + 局域网 + 三个模型目录 + 密码
```

## 5. 关键约定（Agent 必须遵守）

- **判内核行为前先 `make cwick`**。被 stale 二进制骗过不止一次。
- 改 `web/` 或 `apps/` 之后必须 `make src/webui_embed.c && make cwick`，
  否则二进制里还是旧的静态资源（表现是 404 或页面没变）。
- **一次只改一件事**，放在默认关闭的开关后面，让对照臂就是同一个二进制的另一侧。
- **"更快"必须先过锚点**：固定 prompt 必须产出相同的 top-1 token id 与 logit。
  没有锚点，一个算错的内核也快 3 倍。
- **A/B 必须同场跑**。基线会漂 ~20%，间隔十分钟就可能差 2.3×；<30 步量到的是冷缓存。
- **本仓库的 bash `grep` 经常整条返回空**（实测多次），要搜文本用 `rg`/Grep 工具；
  `sed -i ''` 会报错，用 `sed -i.bak`。
- 模型目录在进程生命周期内被 `.cwick.lock` 持有：常驻服务在跑时 CLI 起不来。
  `CWICK_ALLOW_MULTI=1` 能绕过，**但别用** —— 两个进程互相把对方的页挤出去。
- 每请求状态必须显式推给 DS 的独立循环（`ds_engine_open` 里初始化；`eng->ds` 没被
  memset）。历史事故：think budget 只在 GLM 循环生效，DS 空转生成 16568 token。
- 改完跑 `make test`；**没被测试覆盖的修复不算完成**。
- 提交信息沿用仓库风格：`feat:` / `fix:` / `docs:` / `refactor:` + 中英描述；推送走人工。

## 6. 新设备快速启动

```sh
git clone <repo> cwick && cd cwick
make && make test                      # 约 12 s，651 项

./download_model.sh deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    -o ./deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    --include '*.safetensors' '*.json' -j 2

sh ./start.sh deepseek                 # 0.0.0.0:8001，密码 cwick-local
```

取 token 和第一条 curl 请求见 `README.md` 的「Usage → Chat over HTTP」。
没有整套权重时也可以先跑 `./cwick setup --root /tmp/synth` 生成合成模型，
把 `make test` 和服务端链路走通。

## 7. 已知坑与"别再试"

**块验证（`MM_SPEC`）与单 token decode 不是逐位一致的。** 机制在，相对块自身的算术
精确，配 `MM_CHUNK_CAP=1` 时能逐字节复现普通 greedy 输出——但 n=8 时投影走 chunk
GEMM，近似并列的 argmax 会翻转（`short` -> `shorter`）。默认关闭。经济账在
`docs/DECISIONS.zh-CN.md` §28：8 个位置的块要约 2 倍于单 token 一步的专家字节，
所以草稿器要在普通文本上做到每前向 > 2.5 个 token 才划算。

**读路径（已经没有可做的了）**

- `dd` 测到的 6.7 GB/s 是页缓存；stream fd 带 `F_NOCACHE`，**3.4–4.0 GB/s 才是设备限**。
- 把专家读**按文件偏移排序**反而更慢（3.65 vs 3.98 GB/s）——顺序已经不是瓶颈。
- 预取默认关：实测亏，根因是字节不是核。`CWICK_PREFETCH` 只用于对照。

**内存**

- macOS 上 `madvise(MADV_DONTNEED)` 对**文件映射无效**，只有 `munmap` 真回收
  （热切换释放就靠它：DS→GLM RSS 99.7→17.6 GiB）。
- 每步 trim + `cwick_stream_limit` 会雪崩（90 GiB 的 cap 两秒归零，日志全是
  "released 0.00 GiB"）⇒ 必须 12 s 冷却 + 多裁 1 GiB + 下限 `max(cap/4, 256 MiB)`。
- 把稠密权重拷进匿名内存净亏：footprint 82→92 GiB，吃掉专家页缓存。

**内核（都实测否决过，别重开）**

- fp4 内核换 bf16 / fp16：指令数只从 12 降到 10（发射受限，不是 MAC 受限）；
  单线程 f32 16.8 / bf16 23.1 / fp16 38.5 GMAC/s 是不同口径，别直接换算。
- fp8 稠密走原生 e4m3：次正规 / NaN，已证伪。
- 8 激活一组的 fp4 分组 GEMM 比 4 激活慢（16 个累加器溢出寄存器）。
- Metal 单矩阵只有 ~80 GB/s，出路是**成批提交 + 延后等待**（`CWICK_DS_GPU` 默认 0）。
- 采样开销 0.10 ms/step，不是瓶颈；`t<=0` 走 argmax（要让输出可复现就靠这个）。

**决策 prefill（`/v1/systemone`，`docs/DECISIONS.zh-CN.md` §44）**

- prefill 的成本是**按块算的专家读取**，所以不要"均分" prompt：550 token 切成
  275+275 实测 113.6 s，而 512+38 是 109.3 s。该抬的是块上限（512 → 1024 换来 1.14×），
  代价是激活缓冲多 ~1.6 GiB。
- 共享前缀续算**只在 MiniMax 上精确**。GLM 与 DeepSeek 上会改答案（`|dp|` 0.41 / 0.56），
  GLM 走快照回放也还剩 0.11。**绝不能**为了让决策端点跑分好看就打开它——
  一个偶尔改变决策的 2× 是 bug，不是优化。
- 开关按家族写在 `jev_decide` 里（`CWICK_JEV_REUSE_ALL=1` 量已修好的后端，
  `CWICK_JEV_NO_PREFIX_REUSE=1` 强制走旧路径）。

**模型细节**

- MiniMax 的 think 标记是 **200059 / 200060 `<mm:think>`**，不是 200050
  （那个也在词表里，会踩）。
- `DS_CHUNK` 只在 `ds_model.h` 定义；在 `.c` 里再 `#define` 一次会触发
  `-Wmacro-redefined`，两个值随后互相打架。
- `ds_e4m3_to_f32()` 是带 `build_luts` 分支的函数调用，**热循环里必须换成本地表**
  （实测 2.6×）。
- 前缀缓存 off-by-one：`pfx_save_now` 需要另存 `nids-1` 的快照，
  `CWICK_PFX_MIN_TOKENS` 32；排错时把两轮渲染后的 prompt 逐字节 `cmp`，比读代码快。
- 输出看着不对，先找**可信基线**再怀疑内核：曾把"批式 prefill 乱码"查了半天，
  实际是贪心解码在重复 30 次的 prompt 上复读（`mechanical, mechanical,,`）。
