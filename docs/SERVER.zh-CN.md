# OpenAI 兼容服务端参考

<p align="center"><a href="SERVER.md">English</a> · <a href="SERVER.zh-CN.md">简体中文</a></p>

`cwick serve` 实现的是 OpenAI API 中一个刻意做小、严格校验的子集。

## Endpoint

- `GET /v1/models`（以及 `/models`）—— 本实例知道的每一个本地模型，常驻的排在前面。每个条目带有 `cwick.root`、`cwick.loaded` 和 `cwick.supported`（为 false 时还有 `cwick.unsupported_reason`），因此客户端可以区分*正在应答的是什么*、*可能应答的是什么*，以及*这个构建根本无法装载的是什么*。
- `GET /v1/models/<id>`（以及 `/models/<id>`）—— 该形状的单个条目，否则 404 `code:"model_not_found"`。这是 OpenAI 的 retrieve-model 调用；agent 框架会用自己配置里的名字探测它一次，所以应答它可以避免一个 `/v1/chat/completions` 本来不会抛出的 "model not found"。id 会被百分号解码，因此带空格的 checkpoint 目录也能用。
- `POST /v1/completions`（以及 `/completions`）—— 原始文本续写。
- `POST /v1/chat/completions`（以及 `/chat/completions`）—— 对话，使用 CWICK chat 模板（`<|begin|><|system|>…<|end|><|user|>…<|end|><|assistant|>`）。
- `POST /v1/responses`（以及 `/responses`）—— Responses 方言：输入 `input` items，输出 `response` 对象（或 `response.*` 事件）。prompt、队列、前缀缓存与 chat 完全同一套；见下面 **Responses API**。
- `POST /v1/messages`（以及 `/messages`）—— Anthropic 方言：输入 `system` 与 content blocks，输出 `message` 对象（或 `message_start … message_stop` 事件）。同一个引擎、同一条适配器规则；见下面 **Messages API**。
- `GET /v1/metrics`（以及 `/metrics`）—— Prometheus 文本，采用 vLLM 的命名。`cwick:model_loaded`（1/0）是实时的常驻状态位；`cwick:model_info` 保留上一次成功装载的 checkpoint 标签。
- `POST /v1/auth/login`、`GET /v1/auth/status`、`POST /v1/auth/logout` —— 单用户登录 endpoint。登录接受 `{"password":"..."}`，设置 HttpOnly 的 `cwick_session` cookie 并返回一个 Bearer token。配置了密码之后，其他每一个 API/metrics/logs endpoint 在没有它时都返回 401；`/`、应用 HTML 和管理页面则重定向到 `/login`。
- `GET /v1/config`（以及 `/config`）—— 本进程正在运行的选项。只读：其中的一切都是装载期的选择（见下文）。
- `POST /v1/models/switch` —— 把常驻模型换成目录中的另一个条目，无需重启。启动之后才下载完的 checkpoint 会在这里按需发现，不需要额外的刷新调用。本构建没有对应内核的架构、或契约校验不通过的 config，会在**释放任何东西之前**以 400 拒绝（见下文 **切换常驻模型**）。
- `GET /v1/hub/models`、`GET /v1/hub/downloads`、`GET /v1/hub/search?q=…`、
  `POST /v1/hub/download`、`POST /v1/hub/download/<id>/<pause|resume|retry|cancel>`
  —— 本地模型库与下载队列（见下文 **模型库**）。
- `POST /v1/systemone`（也可以是 `/inference/v1/systemone`）—— 一次 prefill 读出类型化决策（需要 `--jev`）；两条路径的 `GET` 都报告当前常驻 checkpoint 正在使用的适配器（见下文 **决策**）。
- `POST|GET|DELETE /v1/auth/keys` —— 供非浏览器客户端使用的 API key；需要登录会话（见下文 **鉴权**）。
- `GET /v1/logs`（以及 `/logs`）—— 本进程自己的 stdout/stderr，来自捕获环形缓冲。`?since=<seq>&limit=<n>`。
- `GET /v1/logs/stream` —— 同一个环形缓冲的实时 SSE 跟随输出。
- `GET /`、`/web/app.css`、`/web/app.js` —— 内置控制台（chat、apps、models、logs、settings；指标与趋势图在 settings 里），以及 `GET /apps/manifest.json` + `/apps/<id>/...` 提供应用包，这两者仅在 `serve` 以 `--web` 启动时才有。它们被编译进二进制；应答时不占用生成槽位。其他所有路径都返回 404。
- `GET /v1/share` —— 局域网 URL，以及手机可以打开的 tictactoe 入口 URL；`GET /v1/share/qr?text=<url>` —— 同一地址，以原生 CoreImage PNG 返回。在默认的 loopback 绑定下 `shareable` 为 false；`--lan` 绑定 `0.0.0.0`，用于可信的同一 Wi-Fi 网络（单用户密码登录；无 TLS）。
- `GET /v1/store`、`/v1/apps/<id>/{state,events,games}` —— 应用自己的数据，存放在**每个应用一个** sqlite 数据库里（`--store DIR`、`$CWICK_STORE`，否则 `~/.cwick/apps`，每个应用拥有 `<id>.db`，而服务端的 key 放在它们旁边的 `auth.db` 里）。`--store FILE` —— 一个本身就是文件的路径，或带 `.db`/`.sqlite` 后缀的路径 —— 保留旧的单一共享数据库。`state` 是 key → JSON，`games` 是每场比赛一行，`events` 是逐手回放日志。值与 payload 都按到达时的确切 JSON 文本存储（进来时解析只是为了证明它*确实*是 JSON），每个上限 1 MiB；`<id>` 为 `[A-Za-z0-9_-]{1,64}`，不合规时是被拒绝而不是被转义。与模型处于同一登录之后，且在生成槽位之前应答。

上述所有只读 endpoint 都在生成槽位**之前**应答，原因与 `/metrics` 相同：观察一次运行绝不能排在它后面。`POST /v1/models/switch` 也在那里应答，但有东西正在生成时会拒绝（409）。

## 请求可以包含什么

- 每个请求一个生成的答案（`n` 和 `best_of` 若出现，必须为 `1`）。
- `max_tokens` 或 `max_completion_tokens` 用于给答案设上限。
- 通过 server-sent events 流式返回，包括 `stream_options.include_usage`。
- `model` —— 若出现，必须是本服务端对外宣称的模型（配置里的 `model_type`）。
- `tools`（`{"type":"function","function":{name, description, parameters}}` 的数组，最多 64 个）和 `tool_choice`（`"auto"`、`"none"`，或 `{"type":"function","function":{"name":…}}` 指名一个已声明的函数）。
- `temperature` / `top_p` / `top_k` 与 `presence_penalty` / `frequency_penalty` 都真的生效（先做范围检查，再交给 sampler）。请求没给的字段保留 checkpoint 自己 `generation_config.json` 里的值，所以"只发 top_p"不会悄悄等于温度 0。
- `developer` 消息就是指令轮，与 `system` 等价（checkpoint 的模板只有一个指令角色；把它当 user 轮渲染，等于把指令变成了请求）。
- 无害的默认值（`store: false`、`response_format: {"type":"text"}`、空的 `stop`）被接受，因为它们什么都不改变。

## 会被拒绝的内容（fail-closed）

以下 OpenAI 特性没有实现，会在进入模型之前返回 OpenAI 形状的 HTTP 400：非默认的 `stop`、logit bias、logprobs、seed、suffix/echo、结构化的 `response_format`、多模态内容、`/v1/completions` 上的 tools、没有 `tools` 的 `tool_choice`、`tool_choice:"required"`（没有任何东西强制模型去调用函数）、指名未声明函数的 `tool_choice`、非 `function` 的 tool 类型，以及任何未知字段。

决策端点同样按名拒绝：`stream`、非 `"none"` 的 `reasoning_effort`、未关闭的 `thinking`
块、`images`，一律返回点名该字段的 400——决策就是一个不生成的 JSON 对象。超过请求体积
上限的 body 在所有端点上都是 **413**；body 根本不是该端点要的文档（JSON 损坏、缺必填
字段、问题非法）则为 **422** 而不是 400，详见下文 **决策**。

服务端一次只生成一个答案，但并发的生成请求不会被拒绝：它领一个号，按到达顺序排队，每个排队的请求都会跑到完成。等待期间 `/v1/models`、`/v1/metrics` 等轻量端点照常应答；排队中断连的客户端会在烧掉一次 prefill 之前被移出队列。503 + `Retry-After` 表示正在换模型或关闭服务；429 只在关闭时仍有永远轮不到的积压时才可能出现。

超过 KV 窗口的 prompt 会被拒绝，而不是被静默截断：引擎返回它被给定的容量，请求以 400 失败，且同时点出这两个数字。在启动时用 `--context N` 设定窗口大小
`--password P`/`CWICK_PASSWORD` 启用单用户登录；`--lan` 总是启用它（若未给出密码则生成并打印一个）。会话保存在内存中，12 h TTL，`HttpOnly; SameSite=Strict`；API 客户端使用登录 endpoint 返回的 Bearer token。没有用户数据库，也没有多租户模型。HTTP 没有 TLS。

（`serve` 默认 131072，即 >100K；`run` 默认 4096，或在 FP8 checkpoint 上是 DSA indexer 的 top-k）。窗口在装载时即固定 —— KV cache 只分配一次 —— 因此无法按请求更改。

## 响应形状

非流式 chat：

```json
{"id":"chatcmpl-…","object":"chat.completion","created":…,"model":"…",
 "choices":[{"index":0,"message":{"role":"assistant","content":"…",
 "reasoning_content":null},"finish_reason":"stop"|"length"|"tool_calls"}],
 "usage":{"prompt_tokens":…,"completion_tokens":…,"total_tokens":…,
          "completion_tokens_details":{"reasoning_tokens":…}}}
```

`reasoning_content` 是本服务端自己的字段（checkpoint 的思考块）；
`completion_tokens_details.reasoning_tokens` 说明答案里有多少是思考，与 Responses 的
`output_tokens_details` 是同一个数。`service_tier` 与 `system_fingerprint` 不发——两者在
schema 里都是可选的，而这里只有一个档位、一个进程一个模型。

流式 chat（`text/event-stream`）：起始 chunk（`delta.role=assistant`）、content delta（`delta.content`）、结束帧（`finish_reason`）、可选的 usage 帧，然后是 `data: [DONE]`。原始 completions 使用带 `delta.text` 的 `text_completion` chunk。

## Responses API

`POST /v1/responses` 是同一个引擎上的第二种方言，不是第二个引擎。请求会被翻译成服务端其余部分早就
在校验的对话，所以 prompt 渲染、单槽队列、前缀缓存和思考预算的行为与 `/v1/chat/completions` 完全一致。

**请求字段**（其余字段返回 400 并点名）：

| 字段 | 行为 |
|---|---|
| `input` | 字符串，或 item 数组：`message`（角色 `user`/`assistant`/`system`/`developer`，内容为字符串或文本 part）、`function_call`、`function_call_output`、`reasoning`（丢弃——隐藏思维链不是可回放的输入） |
| `instructions` | 作为 system 轮插到 `input` 之前 |
| `tools` | Responses 的扁平声明（`{type:"function",name,description,parameters}`），翻译成 chat 的嵌套形状；这里只有 `function` 工具 |
| `tool_choice` | `auto`、`none`，或 `{"type":"function","name":…}`；`required` 需要约束解码，直接拒绝 |
| `max_output_tokens` | 输出预算（也接受 `max_tokens` 别名） |
| `temperature`、`top_p`、`stream` | 同 chat |
| `reasoning` | `.effort` 映射到五档思考强度（`minimal`/`low`/`medium`/`high`/`xhigh`/`none`）；`.summary` 把思考块作为 summary 发布 |

接受但无作用：`metadata`、`user`、`include`、`service_tier`、`prompt_cache_key`、
`safety_identifier`。按名字拒绝：`previous_response_id` / `conversation`（没有持久 response
store——请回放完整 `input` items）、`store:true`、`background:true`、`truncation:"auto"`、
`top_logprobs`、`max_tool_calls`，以及图像/文件内容。`parallel_tool_calls` 与 `text` 会回显
进 response 对象：前者不强制，后者只有纯文本。

**响应**（非流式）：

```json
{"id":"resp-…","object":"response","created_at":…,"status":"completed","model":"…",
 "output":[{"id":"resp-…-rs","type":"reasoning","status":"completed","summary":[]},
           {"id":"resp-…-msg","type":"message","status":"completed","role":"assistant",
            "content":[{"type":"output_text","text":"…","annotations":[]}]},
           {"id":"resp-…-fc0","type":"function_call","status":"completed","call_id":"…",
            "name":"…","arguments":"{…}"}],
 "usage":{"input_tokens":…,"input_tokens_details":{"cached_tokens":…},
          "output_tokens":…,"output_tokens_details":{"reasoning_tokens":…},
          "total_tokens":…},
 "error":null,"incomplete_details":null,
 "parallel_tool_calls":true,"tool_choice":"auto","tools":[],"instructions":null,
 "max_output_tokens":…,"temperature":null,"top_p":null,"truncation":"disabled",
 "text":{"format":{"type":"text"}},"reasoning":null,"metadata":null,
 "store":false,"background":false,"previous_response_id":null}
```

后半段是请求的回显。`tools`、`tool_choice`、`parallel_tool_calls` 在 schema 里不是可选项，
其余字段是客户端拿来对照自己发了什么的；同一段同时拼进 `response.created`，所以流式信封与
最终 body 说的是同一件事。

item id 由 response id 派生（`…-rs`、`…-msg`、`…-fc<i>`）：`output_item.added` 公布的 item
与终止帧里的 item 是同一个 item，客户端靠 id 对应。

回合产生过思考就一定有 reasoning item（流式在第一个思考 token 到达时就公布了它），但 `summary`
只在请求了 `reasoning.summary` 时填充，否则是 `[]` —— 既不泄露思维链，也不改变 output 列表的
形状。没有工具调用的回合一定带 message item，即使模型把预算都花在思考上、正文为空。

预算用尽时 `status` 是 `incomplete`，`incomplete_details.reason` 为 `"max_output_tokens"`；
这是这个方言唯一能表达"被截断"的地方（Anthropic 那边是 `stop_reason:"max_tokens"`）。

**流式**使用具名 SSE 事件，没有 `data: [DONE]`。每个事件都带单调递增的 `sequence_number`，
终止事件是 `response.completed`、`response.incomplete`（预算）或 `response.failed`（响应已
建立但没能完成，此时 response 对象是 `status:"failed"` 且 `error` 有值）：

```
response.created → response.in_progress
→ response.output_item.added → response.content_part.added
→ response.output_text.delta …→ response.output_text.done
→ response.content_part.done → response.output_item.done
→（每个 function_call 依次为 added → arguments.delta → arguments.done → done）
→ response.completed | response.incomplete | response.failed
```

思考在请求了 summary 时走 `response.reasoning_summary_text.delta`，否则走
`response.reasoning_text.delta`。`response.output_text.delta` 与 `.done` 都带 `logprobs: []`
——schema 里这个字段没有默认值，而本服务端不算对数概率，所以永远是空数组。

工具调用先由 `output_item.added` 公布（`arguments` 为空），然后一个
`response.function_call_arguments.delta` 带上完整参数字符串，接着
`response.function_call_arguments.done`，最后 `output_item.done`。调用是从模型自己的协议里
解析出来的，只有写完才存在，所以参数没法逐 token 挤出来——这些事件存在是因为客户端就在那里
找载荷，而"一个 delta 装完整串"是诚实的形状。

**与参考文档的偏差。** 需要持久化能力的部分一律不实现，且按名拒绝而不是半服务：
`previous_response_id`、`conversation`、`store:true`、`background:true`、
`truncation:"auto"`、`top_logprobs`、`max_tool_calls`、非 `function` 工具、
图像/文件输入。`parallel_tool_calls` 会回显但不强制（发几次调用由 checkpoint 决定），
`text.format` 只有纯文本，`instructions` 只接受字符串（schema 也允许 item 数组）。
`metadata`、`user`、`include`、`service_tier`、`prompt_cache_key`、`safety_identifier`
接受但无作用。create 请求的其余字段都已实现。

## Messages API

`POST /v1/messages` 是 Anthropic 的方言 —— Claude Code 与 Anthropic 的 SDK 发的就是它。
和 Responses 一样是适配器：`anthropic_parse` 把 blocks 折成服务端早就在校验的 chat 会话，
`messages_body` / `msg_stream_*` 只写回复。prompt 渲染、单槽队列、前缀缓存与思考预算
都是同一份代码。

**请求字段**（其余字段一律 400 并点名）：

| 字段 | 行为 |
|---|---|
| `messages` | `{role: user\|assistant, content}`；content 是字符串或 block 数组：`text`、`thinking` / `redacted_thinking`（丢弃——隐藏思维链不是可回放的输入）、`tool_use`（assistant）、`tool_result`（user） |
| `system` | 字符串或 `text` blocks；成为 system 轮，且排在 `messages` 之前 |
| `max_tokens` | **必填**，这里没有默认值 |
| `tools` | 扁平的客户端工具：`{name, description, input_schema}`。`type` 可缺省或为 `"custom"`；服务端工具（`web_search` 等）按名拒绝 |
| `tool_choice` | `{"type":"auto"}`、`{"type":"none"}`、`{"type":"tool","name":…}`。`{"type":"any"}` 需要约束解码，拒绝 |
| `temperature`、`top_p`、`top_k` | 与 chat 相同 |
| `stream` | 具名事件的 SSE |
| `thinking` | `.type:"disabled"` 不思考直接答；`"enabled"` 把 `.budget_tokens` 映射到五档思考强度（≤512 低、≤2048 中、≤8192 高、≤16384 超高、以上极致）；`"adaptive"` 映射到高 |

接受但无作用：`metadata`、`service_tier`。按名拒绝：非空的 `stop_sequences`（引擎在
自己的分隔符上停）、`image` / `document` / `audio` / `video` block（纯文本服务端）、
以及 assistant 轮里的 `tool_result` 或 user 轮里的 `tool_use`（配对错了，猜一个就会
送出错的 prompt）。

**内容映射。** 一个 assistant 轮变成一条 chat 消息：`content` 加 `tool_calls`（`tool_use`
的 `input` 对象重新序列化成 chat 层要的 arguments 字符串）。user 轮里的 `tool_result`
block 变成 `tool` 消息——同一轮里的多个结果保持顺序——剩下的文本成为它们之后的 user 消息。

**响应**（缓冲）：

```json
{"id":"msg-…","type":"message","role":"assistant","model":"…",
 "content":[{"type":"thinking","thinking":"…","signature":""},
            {"type":"text","text":"…"},
            {"type":"tool_use","id":"…","name":"…","input":{"path":"a.c"}}],
 "stop_reason":"end_turn","stop_sequence":null,
 "usage":{"input_tokens":…,"output_tokens":…}}
```

`signature` 是空的：这个引擎没有签名密钥，也从不校验回放的思维链——这也是请求里的
`thinking` block 被丢弃的原因。模型调了工具时 `stop_reason` 是 `tool_use`，输出预算用尽
是 `max_tokens`，其余是 `end_turn`。前缀缓存覆盖了部分 prompt 时，
`usage.cache_read_input_tokens` 报告覆盖了多少。

**流式**用具名事件，以 `message_stop` 结束：

```
message_start → (content_block_start → content_block_delta* → content_block_stop)*
→ message_delta → message_stop
```

`content_block_delta` 按当前打开的 block 携带 `text_delta`、`thinking_delta` 或
`input_json_delta`。`tool_use` block 以空 input 打开，参数以 `input_json_delta` 分片到达，
所以客户端能看到调用正在成形——这个方言**确实**流式发参数，与 Responses 不同。
`message_delta` 是流式回合报告 `stop_reason` 的唯一位置，它同时重复 `output_tokens` **和**
`input_tokens`（前缀命中时还有 `cache_read_input_tokens`），所以只读收尾事件的客户端也能拿到
完整的 usage。响应头发出后的失败是具名 `error` 事件，载荷为
`{"type":"error","error":{"type":"api_error","message":…}}`——与缓冲失败返回的形状相同。

**与参考文档的偏差。** 没有 `POST /v1/messages/count_tokens`（引擎没有"只数 token、不渲染
prompt"的入口）。按名拒绝、绝不静默降级：非空的 `stop_sequences`、`image` / `document` /
`audio` / `video` block、`{"type":"any"}`。`thinking.budget_tokens` 允许低于文档写的 1024
下限（映射成低），`enabled` 不带预算时映射成中；不发 `ping` 事件；`thinking.signature` 为空，
因为这里既不签名也不校验。`metadata` 与 `service_tier` 接受但无作用，
`tool_result.is_error` 被当作普通结果文本透传。

**鉴权。** Anthropic 客户端用 `x-api-key` 而不是 `Authorization: Bearer` 送 key；本服务端
两者都收（头部解析本来就是如此，对所有 endpoint 有效）。`anthropic-version` 读了就忽略：
这里只有一份实现，且不按协议版本分叉。

## Tool 调用

`tools` 以 checkpoint 自己的协议告知模型，直接取自 `zai-org/GLM-5.3-Flash/chat_template.jinja` 而不是自己发明：

```
<tools>
{"name": "get_weather", "description": "...", "parameters": {...}}
</tools>

For each function call, output the function name and arguments within the
following XML format:
<tool_call>{function-name}<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>
```

然后再由 `src/tools.c` 从原始 completion 里读回来。包含格式良好的 `<tool_call>` 块的答案会以 `message.tool_calls[]` 返回，并带 `finish_reason: "tool_calls"`（流式客户端拿到的是 `delta.tool_calls`）；第一个块之前的文字仍留在 `content` 里。

没有任何东西约束 decode，所以失败模式是被处理掉的，而不是靠祈祷它不出现：

- 格式错误或未闭合的 `<tool_call>` → HTTP 500，绝不会把一个只解析了一半的调用交给客户端；
- `tool_choice` 指名了一个模型并未调用的函数 → HTTP 500，并给出指名该函数的消息。

## stdout 上的日志

每一条诊断都写到 **stdout**（`src/log.c`），按行原子、带时间戳：

| 级别 | 开关 | 内容 |
| --- | --- | --- |
| info | （总是） | 启动、内存预算、每个请求一行、每次生成一行、关闭 |
| verbose | `--verbose` | 每请求细节、带 RSS 的 prefill/decode 进度、完整的预算表、cache 与 SSD 计数器 |
| trace | `--trace` | 模型看到和产出的内容：system/user 消息、声明的 tools、渲染后的 prompt、输出，以及每一次 tool 调用 |

```
cwick: [info   +1.34s] [init] OpenAI-compatible server listening on http://127.0.0.1:8000/v1 (model glm-5.3-flash)
cwick: [info  +29.99s] [req 1] POST /v1/chat/completions (249 bytes body)
cwick: [info  +80.23s] [gen] prefill 8/147 — rss 53.75 GiB of 102.40 GiB ceiling, expert cache 36.27 GiB pinned
cwick: [info +366.28s] [req 1] POST /v1/chat/completions -> 503 in 336470.8ms interrupted: shutdown
```

`CWICK_SERVER_LOG=0` 让请求/生成那几行安静下来，`CWICK_SERVER_LOG=verbose` 在不用 `--verbose` 的情况下提高细节级别。

那份 stdout 也可以通过 HTTP 读到。`src/logcap.c` 把 fd 1 和 fd 2 tee 进两个管道；一个 pump 线程把每个字节回显给保存下来的描述符，并把每个完成的行推入一个有界环形缓冲（2000 行 / 4 MiB）。因为 tee 坐在描述符层面而不是 `log.c` 内部，Metal 诊断、Objective-C 的 `NSLog` 输出和零散的 `fprintf(stderr, …)` 也会落进去 —— 而 GUI 最需要的那几行，恰恰正是 `log.c` 从没见过的。

```
GET /v1/logs?since=0&limit=3
{"capture":true,"seq":18,"oldest":1,"dropped":0,"retained":17,
 "lines":[{"n":1,"src":"out","t":1792299405400,"msg":"cwick: … [init] log capture on: …"},
          {"n":2,"src":"out","t":1792299405401,"msg":"cwick: …"},
          {"n":3,"src":"out","t":1792299405402,"msg":"cwick: …"}]}
```

`src` 是 `out` 或 `err`；`t` 是捕获时的墙上时钟，自 epoch 起的毫秒数；`n` 是要续读的游标（`since=last_n+1`）。已经被淘汰的游标会被夹到 `oldest`，而不是返回一个空尾巴；`dropped` 说明有多少行先于其时地消失了，这样客户端在历史不完整时可以如实相告，而不是默默展示一份。

`/v1/logs/stream` 发送 `{"type":"hello","seq":…,"capture":…,"dropped":…}`，然后每行一个 `{"type":"line",…}` 帧，由环形缓冲的条件变量唤醒而不是靠轮询。它在客户端挂断或进程停止时关闭（停止标志在每次等待之间检查，所以即便开着一个 tail，Ctrl+C 也会在 400 ms 内被响应 —— 实测约 230 ms），并且每 15 s 发一个 `: idle` 注释，好让它前面的任何环节都不要把这个连接回收掉。

带上 `--trace`，prompt 和输出就在那个环形缓冲里，因此也就在这些 endpoint 上。这个开关就是为此而设；除非 `CWICK_ALLOW_PUBLIC=1`，监听器只在 loopback 上。

## 鉴权

两种凭据，因为「谁在调用」有两个非常不同的答案：

| | 会话 | API key |
| --- | --- | --- |
| 用于 | 浏览器控制台 | 第三方 agent / 脚本 |
| 获取方式 | 用密码调用 `POST /v1/auth/login` | `POST /v1/auth/keys`（需要会话），或 `--api-key` / `CWICK_API_KEY` |
| 发送方式 | `cwick_session` cookie，或 `Authorization: Bearer` | `Authorization: Bearer <key>` 或 `X-API-Key: <key>` |
| 生命周期 | 12 h，仅内存，重启即消失 | 直到被吊销，以哈希形式持久化 |
| 能否管理 key | 能 | **不能**（401） |

`--password P` / `CWICK_PASSWORD` 启用校验；`--lan` 强制启用；`--api-key` 也强制启用，因为一个不做校验的 key 什么也保护不了。没有给密码时，服务端会生成一个，并在 `auth:` 启动行上打印出来。密码以 PBKDF2-HMAC-SHA256（120k 次迭代、随机盐）持有，从不写入磁盘；五次错误尝试会把该 endpoint 锁定 60 s（`Retry-After`）。

### API key

```bash
# 从控制台会话铸造一个
curl -s -b cookies.txt -X POST localhost:8000/v1/auth/keys \
  -H 'Content-Type: application/json' -d '{"name":"cursor"}'
{"id":"mUDhRy6Ywlyu","name":"cursor","prefix":"cwick_6My8BTo2",
 "created_at":1792342101,"secret":"cwick_6My8BTo2Tso5J5vCmV91cMqlGaluDx6s",
 "note":"shown once; the server keeps only its hash"}

# 列出 / 吊销
curl -s -b cookies.txt localhost:8000/v1/auth/keys
{"keys":[{"id":"mUDhRy6Ywlyu","name":"cursor","prefix":"cwick_6My8BTo2",
          "created_at":1792342101,"last_used_at":1792342104,"enabled":true}]}
curl -s -b cookies.txt -X DELETE 'localhost:8000/v1/auth/keys?id=mUDhRy6Ywlyu'
{"status":"revoked"}
```

这些性质是刻意选定的：

- **160 个随机比特**（`cwick_` + 32 个 base62 字符），以*单个 SHA-256* 存储。PBKDF2 是给低熵的人类密码用的；拉伸一个 160 位的随机秘密什么也换不来，还会给每个请求加税。
- **明文从不持久化**，`auth_keys` 表里只有 `id`、`name`、`prefix` 和哈希。`prefix` 是展示形式（`cwick_6My8BTo2`），这样列表可读，访问日志也可以写 `auth=key:cwick_6My8BTo2` 而从不携带密钥本身。
- **key 能活过重启**；会话不能。agent 只配置一次。
- **key 不能铸造 key。** `POST /v1/auth/keys` 需要会话，对 key 返回 401，所以一个泄漏进日志或备份的秘密无法被用来种下更多。
- key 在会话查找之后才检查，而*会话 token 不是有效的 key*（反之亦然）：两者是分开的，这样泄漏了哪一个可以分辨。
- 畸形的哈希在装载时被拒绝，而不是被静默忽略；一个持久化失败的 key 会从内存里回滚掉，所以内存与磁盘对「什么被接受」永远不会各说各话。

## 切换常驻模型

`--models-dir DIR`（可重复，最多 8 个）让其他本地 checkpoint *可被列出*；在切换请求它之前什么都不会装载。一个目录合格的条件是它含有 `config.json` 以及 `resident/tensors`、`model.safetensors.index.json`、`model.safetensors` 之一；对外宣称的 id 是 `model_type`，其次 `architectures[0]`，再次目录名，重复的以 `@2`、`@3`、… 去重。常驻条目从不改名，所以 `data[0]` 保持稳定。

**「下载好了」不等于「能装载」。** `/v1/models` 与 `/v1/hub/models` 的每个条目都带 `supported`，为 false 时还带 `unsupported_reason`。模型库会下载任意仓库，但只有本构建实现了内核的架构才能变成常驻模型。判定读的是 `config.json` 的 `model_type`（有 `text_config` 时取内层，与装载器读法一致），并且在目录被发现时就算好缓存下来 —— 所以被轮询的模型列表不会在持有生成锁的时候解析配置文件。Web 控制台会禁用不支持条目的切换按钮。本构建实现了 GLM-5.3、DeepSeek-V4.1、MiniMax-M3.1 三套内核，以及 `cwick setup` 生成的小 checkpoint；要支持更多架构，就是加内核并在类型表里加一行，这份清单并不表示永远只能有这些模型。

发布版 checkpoint 还会额外宣称一个由目录 basename 导出的*服务名*，因为 `model_type` 是训练运行的产物，没有客户端会猜到（`glm5_next`、`deepseek_v41_text`）：`zai-org/GLM-5.3-Flash` 以 **`glm-5.3-flash`** 提供服务，`deepseek-ai/DeepSeek-V4.1-Flash-1017` 以 **`deepSeek-v4.1-flash-1017`** 提供服务。`/v1/models` 列出的就是这个名，completions 在 `model` 里回显的也是这个名，`/v1/models/switch` 接受的还是这个名 —— checkpoint 自己的 `model_type` 在各处也照旧可用，所以学过旧拼写的东西都不会坏掉。其他 checkpoint 不受影响，保留它们 config.json 里的名字。

```bash
curl -s localhost:8000/v1/models/switch \
  -H 'Content-Type: application/json' -d '{"model":"glm-5.3-flash"}'
```

| 状态 | 响应体 `status` | 何时 |
| --- | --- | --- |
| 200 | `switched` | 新模型已常驻并在服务 |
| 200 | `already_resident` | 空操作：那个 root 已装载 |
| 404 | — | 没有该 id 的目录条目 |
| 400 | `error` | 目标架构在本构建里没有内核，或它的 config 未通过模型契约校验；**常驻模型完全不动** |
| 409 | `error` | 某个槽位忙、有请求在排队、tokenizer 被占用，或正在换模型 |
| 500 | `error` | 旧模型已释放之后，替换引擎或其 tokenizer 装载失败 |

**校验在拆解之前。** 切换会先重读目标的 `config.json` 并跑一遍完整的模型契约检查（dry-run），只有通过了才释放旧引擎、tokenizer、Metal arena 和前缀缓存。未知的 `model_type` 会被 400 点名拒绝，而不会再掉进默认几何解析器、最后报一个无关的字段缺失（嵌套 config 里 `text_config.pad_token_id` 为 null 时，明明外层有值却报「missing pad_token_id」就是这么来的）。这两种 400 都不释放任何东西：常驻模型继续服务，`cwick:model_loaded` 保持 1，日志里也不会出现 `switching to` / `released`。

换模型进行中时，生成请求会被应答 503 `code:"model_switch"` 而不是排队：它们被校验时所依据的那个模型正在被拆掉。如果**拆解之后**替换才失败（权重损坏、缺张量、tokenizer 装载不了），服务端会带着「没有模型」继续运行，并应答 503 `code:"no_model"`，直到另一次切换成功 —— 它绝不会退回一个半初始化的引擎。此时 `cwick:model_loaded` 为 `0`。`cwick:model_info` 有意保留上一次的标签（指标名和标签在一个进程生命周期里必须稳定），所以监控面板要用这个 gauge 来区分「在服务」与「没有模型」。

`GET /v1/config` 报告本进程正在以什么运行：

```json
{"model":"…","model_root":"…","host":"127.0.0.1","port":8000,"slots":1,
 "max_tokens":1000000,
 "max_request_bytes":8388608,"context_window":512,"catalog_size":2,
 "web":true,"verbose":true,"trace":false,
 "prefix_cache":{"enabled":false,"block_tokens":4096},
 "log_capture":true,"log_retained":300,"models_dirs":["/models"]}
```

这里刻意没有写 endpoint：上下文窗口、内存比例、槽位数量和字节上限都在装载期决定，所以通过 HTTP「更改」它们会是一句谎话。`0` 表示没有限制。

## 决策

`--jev`（或 `CWICK_JEV=1`）开启决策端点。它用**已经常驻**的模型作答：不加载第二个
checkpoint、不做微调、不起第二个进程。对外发布的路径是 `/inference/v1/systemone`，所以
把客户端指向 `http://host:port/inference` 就能打到它；`/v1/systemone` 与 `/systemone`
是短写法。三条路径的 `GET` 与 `POST` 行为一致。

`POST /v1/systemone`

```json
{"model": "glm-5.3-flash",
 "state": {"board": ["X", null, "O"], "turn": "O"},
 "questions": {
   "move": {"type": "choice", "instructions": "哪一格让 O 的胜率最高？",
            "criteria": {"c0": "左上", "c4": "正中"}},
   "risk": {"type": "score", "instructions": "O 离输有多近？",
            "criteria": ["稳", "吃紧", "已输"]},
   "accept": {"type": "noul", "instructions": "O 是否已经三连？",
              "criteria": {"true": "O 已三连", "false": "还没有"}}
 }}
```

- `state` 必填且非空。字符串原样使用；其它 JSON 值（对象或数组）会被序列化后放进
  `<state>`，所以把状态存成 JSON 的调用方，看到的就是自己的字段。
- `questions` 是「调用方自己的 id → 问题」的映射，这些 id 会作为 `answers` 的键回来。
  每个问题都要有 `type`（`choice` / `score` / `noul`）与 `instructions`（用自然语言写的
  问题本身）。这里没有兜底文案：用一个错问题去决策，比拒绝更糟。
- `choice` 的 `criteria` 是对象（key → 描述）或名字数组；返回 `choice`、`probabilities`
  与 `confidence`。
- `score` 的 `criteria` 必须是 2 到 10 个档位描述、从低到高排列；返回概率加权索引、
  `legend` 与 `probabilities`。
- `noul` 的 `criteria` 描述陈述的两侧——`{"true": …, "false": …}` 或二元数组（真在前），
  返回 `noul` = P(陈述成立)。两侧在内部是固定顺序的，所以即使 body 先写 `false`，
  报出来的仍是 P(true)；不写 criteria 就是 yes/no。
- 一次请求最多十个问题、每个问题最多 26 个选项；每个问题一次 prefill，同一问题的所有
  选项共用那一次前向。
- `model` 可选；写了就必须是常驻 checkpoint（服务别名或 `model_type`，大小写不敏感）
  ——其它一律 400 并点名常驻模型，因为一个挂在过期 id 下的决策，与一个正确的决策在
  调用方看来无法区分。
- `temperature` 只缩放读数分布（默认 1.0）；这里不采样，所以同一个请求在同一个
  checkpoint 上是确定性的。
- `stream`、非 `"none"` 的 `reasoning_effort`、未关闭的 `thinking` 块，以及 `images`
  一律点名拒绝：决策是一个 JSON 对象，不生成 token、不做推理轮，接受这些字段等于回答
  一个调用方没问的问题。

响应：

```json
{"object": "systemone.decision", "model": "glm-5.3-flash",
 "adapter": {"family": "glm", "prompt": "glm-chat-closed", "readout": "letter",
             "letters": 26, "letters_total": 26, "first_missing": null,
             "space_lead": true},
 "answers": {"move": {"type": "choice", "choice": "c4",
                      "probabilities": {"c0": 0.12, "c4": 0.88},
                      "confidence": 0.76},
             "risk": {"type": "score", "score": 1.7,
                      "legend": {"0": "稳", "1": "吃紧", "2": "已输"},
                      "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
                      "confidence": 0.4},
             "accept": {"type": "noul", "noul": 0.02}},
 "usage": {"input_tokens": 118, "cached_input_tokens": 0, "output_tokens": 0,
           "prefilled_tokens": 118, "reused_tokens": 0}}
```

`answers` 里每种题型只给该题型承诺的东西：`noul` 是一个概率，`choice` 是最可能的 key
加整体分布，`score` 是期望档位加逐档分布与把下标映射回描述的 `legend`。

`confidence` 是 `(K·Σp² − 1)/(K − 1)`：均匀分布为 0，点质量为 1。它说的是模型有多集中，
不是它有多可能答对。

`usage.cached_input_tokens` 是参考协议里「从缓存提供的 state token 数」这一字段；
`prefilled_tokens` 与 `reused_tokens` 并列给出，因为日志行报的就是这两个数，调用方
能直接看到节省，而不用推算。

`GET /v1/systemone` 返回同样的 `model` 与 `adapter`，外加各项上限（`questions_max`、
`options_max`、`state_max_bytes`、`instructions_max_bytes`、`levels_min`、
`levels_max`）。适配器由常驻 checkpoint 解析而来：GLM 与 DeepSeek
用"思考块已关闭"的对话轮，MiniMax 用普通对话轮，选项字母则对已加载的 tokenizer 探测。
每次 `POST /v1/models/switch` 之后都会重新解析，所以切换同时换掉权重和接口。
`make jevprobe` 只读 tokenizer、不加载权重，打印字母表。

错误码沿用参考协议的分类，因为调用方对此能作出反应：body 根本不是决策请求（JSON 损坏、
缺必填字段、问题不合法）→ **422**；请求本身合法但本服务不提供（模型名过期、超上限、
不支持的字段、分词器无法把某个选项拼成单 token）→ **400**；body 超过请求体积上限 →
**413**；没有模型或队列已满 → **503**（该端点占用生成槽位，所以决策与对话轮不能重叠）。

**与参考文档的差异（有意为之，写在这里免得调用方自己踩）。**

| 参考协议 | 本实现 |
|---|---|
| 一次最多 256 个问题 | 10 个（goal：一个 state 配一小把问题） |
| `choice` 最多 208 个选项 | 26 个——读数就是每个选项一个字母 token |
| `model` 是 `accounts/x/models/y` 这样的路径 | 常驻模型的服务别名或 `model_type` |
| 没有 `temperature` | 有，只缩放读数（扩展） |
| 响应只有 `model`/`answers`/`usage` | 另加 `object` 与 `adapter`（扩展） |
| 另有 OpenAI 形状的 `/v1/decisions` | 未实现 |

`state`（4 MiB）、题面（64 KiB）、`score` 档位（2 到 10）与请求体积上限都与参考一致。


**日志。** 所有行都写到 stdout，带系统时间与请求编号 `[req N]`。一次
`POST /v1/systemone` 依次打印：请求（`decision: N question(s) ... stream=false
thinking=off`）、每个 prefill 一行 `question i/N`、引擎的
`prefill ... decision readout` 开始行、逐 token / 逐层进度心跳、
`prefill done: N tokens in X ms (Y tok/s)`、每个问题的读数
（`-> key p=... (runner-up ...), confidence=...`），以及
`decision done: N question(s), T input tokens, P prefilled (R reused), W ms wall,
R tok/s prefill`。
被拒时打印 `systemone: decision refused — <原因>` 和原有的 4xx 访问日志。
`--trace` 额外打印每个问题渲染后的 prompt（system + user）与完整决策对象。
决策运行期间 `cwick:prefill_*` 指标同步变化，所以 `/v1/metrics` 与控制台也能
看到这次 prefill。进度行只对在线 HTTP 请求输出：CLI 数值校验工具调用
`cwick_engine_prefill_logits` 时没有绑定请求标签，保持静默，机器可读输出不变。

**共享前缀。** 一个请求里的每个问题都以同一段 `system + <state>` 开头，所以端点先编码全部
问题、取最长的公共 **token** 前缀、只 prefill 它一次，之后每个问题只付自己的尾巴
（`question i/N: prefill S tokens, P reused from the shared prefix`）。`usage` 里同时给出
`input_tokens`、`cached_input_tokens`、`prefilled_tokens` 与 `reused_tokens`，节省是
看得见的，不用推算。

这条路径只在「跨 token 状态按位置寻址、且能被精确续写」的后端上启用。MiniMax（GQA + KV）
与不中断的 prefill 逐位相同——实测三问请求 `|dp| = 0.0`、选择完全一致。DeepSeek（压缩 KV +
indexer pool）与 GLM（DeltaNet/DSA pools）的状态是累积的，在任意位置重写并不能撤销：同一个
请求在它们上面会改变答案（`|dp|` 0.56 与 0.41），因此两者保留 N 次完整 prefill。
`CWICK_JEV_NO_PREFIX_REUSE=1` 可强制关闭，`CWICK_JEV_REUSE_ALL=1` 可在其它后端上开启用于
测量。完整推理与全部实测数据见 `docs/DECISIONS.md` §44。

## 应用存储

`--store FILE`（或 `$CWICK_STORE`，否则 `~/.cwick/apps.db`）指定那个 sqlite 文件，它保存内置应用自己的数据。它以 WAL 和 `synchronous=NORMAL` 打开，包在一个 mutex 里（每个连接有自己的线程，而一步棋不能排在一次 decode 后面），并用 `IF NOT EXISTS` 创建：

```sql
app_kv     (app_id, k, v, updated_at)                PRIMARY KEY (app_id, k)
app_games  (app_id, game_id, started_at, ended_at, outcome, moves)
app_events (id INTEGER PK, app_id, game_id, seq, ts, kind, payload)
```

```bash
curl -s localhost:8000/v1/store
{"enabled":true,"mode":"per-app","dir":"/Users/me/.cwick/apps",
 "primary":{"path":"/Users/me/.cwick/apps/auth.db"},
 "apps":[{"app":"tictactoe","path":"/Users/me/.cwick/apps/tictactoe.db",
          "games":1,"events":6,"kv_rows":1}],
 "path":"/Users/me/.cwick/apps/auth.db","games":0,"events":0,"kv_rows":0,
 "bytes":4096}

curl -s 'localhost:8000/v1/store?app=tictactoe'   # 单个应用自己的数字
{"enabled":true,"mode":"per-app","app":"tictactoe",
 "path":"/Users/me/.cwick/apps/tictactoe.db",
 "games":1,"events":6,"kv_rows":1,"bytes":20480}

curl -s -X PUT localhost:8000/v1/apps/tictactoe/state \
  -H 'Content-Type: application/json' \
  -d '{"key":"score","value":{"wins":2,"losses":1,"draws":0,"games":3}}'
{"status":"ok","written":1}

curl -s -X POST localhost:8000/v1/apps/tictactoe/events \
  -H 'Content-Type: application/json' \
  -d '{"game_id":"g1","seq":0,"kind":"human","payload":{"cell":4,"board":["","","","","X","","","",""]}}'
{"status":"ok"}

curl -s 'localhost:8000/v1/apps/tictactoe/events?game_id=g1'
{"app":"tictactoe","events":[{"id":1,"seq":0,"ts":1792337681,"game_id":"g1",
  "kind":"human","payload":{"cell":4,"board":["","","","","X","","","",""]}}]}
```

要紧的规则：

- 每个值和 payload 都必须是 JSON；它们**按到达时的确切文本存储**并原样返回，所以应用的字段能活过一次往返（包括数字格式 —— `0.42` 仍是 `0.42`）；
- 每个值/payload 1 MiB，每次读取 5000 行，`id` 为 `[A-Za-z0-9_-]{1,64}`；
- 坏的应用 id（`../etc`、引号、空格）是 400，绝不是被转义后的查询；
- `DELETE /v1/apps/<id>/games?game_id=G` 在一个事务里删掉比赛**及其**事件，因为一个被删了一半的比赛会留下孤立的步骤，而回放视图会把这些步骤显示成没有父级；
- 打不开的 store 会在启动时记录日志并应答 503；服务端继续服务模型，而井字棋应用退回到 `localStorage`（它本来就一直往那里镜像）；
- 每个应用的数据库是一个在首次使用时打开的独立 sqlite 句柄，并在一个 mutex 下缓存，所以一个浏览器标签页和局域网里的一部手机看到的是同一个文件，而无需每个请求都重跑一遍 schema。在代码检出目录里用 `--store ./apps` 会把它们放在应用源码旁边；`tools/embed_web.c` 在嵌入 `apps/` 时会跳过每一个数据库后缀，因此运行时数据从不进入二进制。

这些 endpoint 与模型处在同一个单用户登录之后：`/v1/store` 不在公共 API 允许列表里，所以未鉴权的请求是 401，而应用页面的导航是 302 到 `/login?next=…`。

## 停止

`Ctrl+C`（SIGINT/SIGTERM）会停掉服务端。空闲时到达的信号通过一个 self-pipe 打断 `poll()` 循环（macOS 在 handler 返回后会重启 `accept()`，这就是为什么旧的循环要等到下一个请求到来才注意到 Ctrl+C）。在 decode 中途到达的信号会在下一个 token 边界中止生成 —— 在途请求收到 OpenAI 形状的 503 `server is shutting down (interrupted)` —— 然后监听器关闭，进程退出。第二次按 Ctrl+C 会立即退出（状态码 130）。

## 用法

```bash
./cwick serve --root <model-root> --host 127.0.0.1 --port 8000 --verbose
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm_5_3_flash_synthetic","stream":true,"messages":[{"role":"user","content":"Hi"}]}'
```

除非你自己加上鉴权和网络边界，否则把服务端留在 loopback 上。

## 模型库

五个 endpoint，都在生成槽位**之前**应答：目录检索慢、或者正在下载几百 GiB 权重，都不能把
一次对话排在后面。它们由同一个进程、同一个登录会话提供；浏览器不直连远端，因此也不存在
CORS 问题。

| 方法 | 路径 | 应答 |
|---|---|---|
| GET | `/v1/hub/search?q=<text>&limit=<n>` | `{"models":[…],"count":N,"endpoint":…,"error":…}` —— 远端目录的精简投影：id、下载量、点赞、tags、pipeline tag、是否 gated/private。`limit` 为 1–50（默认 12）。 |
| GET | `/v1/hub/models` | `{"root":"./models","models":[…]}` —— 库根下的每个模型目录，含字节数、文件数、未完成字节、`complete`、`resumable`、`supported`，以及为 false 时的 `unsupported_reason`。 |
| DELETE | `/v1/hub/models/<name>` | 删除已下载的 checkpoint。200 `{"deleted":true,"path":…,"bytes":N,"files":N}`；名字未知或不安全 404；常驻模型、或有下载线程正在写入该目录 409。 |
| GET | `/v1/hub/downloads` | `{"jobs":[…]}` —— 实时任务表：状态、进度、字节、分片、当前文件、速率、ETA 与错误。 |
| POST | `/v1/hub/download` | body `{"repo":"owner/name"}` → 一个任务对象。对已在下载的仓库是幂等的。 |
| POST | `/v1/hub/download/<id>/<动作>` | `pause` / `resume` / `retry` / `cancel` → 更新后的任务对象。 |

状态流转：`queued` → `listing` → `running` → `done`，出口为 `paused`、`error`、
`canceled`。页面轮询 `/v1/hub/downloads`，没有推送流，任务表是唯一共享状态。

**文件落在哪。** `./models/owner/name/<path>`，并写一个 `.cwick-repo` 标记保存完整仓库 id。
启动扫描同时识别扁平的 `./models/<name>` 与 `owner/name` 两层布局并把它们登记进模型目录，
所以上一轮下载的 checkpoint 开机即可切换；本轮进程内下载完的，则由
`POST /v1/models/switch` 按需发现。只有存在 `config.json` 且没有任何 `.incomplete` 残留时，
一个目录才算「完整」。

**删除。** `DELETE /v1/hub/models/<name>` 接受 `owner/name`、裸目录名，或扁平的
`./models/<name>` 条目。名字会先做百分号解码；含 `..`、反斜杠、绝对路径、空分量或 `.`
分量的名字直接拒绝；解析出的目录在 realpath 之后必须严格位于库根之内，库根本身永远不可删。
删除是递归的，且不跟随符号链接。常驻模型返回 409（先切走）——删掉正在运行引擎已映射的权重会
把它弄坏；仍有下载线程在写入的目录也返回 409（先取消任务），否则一次可续传的传输会变成损坏的
一份。删除成功后，同一请求内会把 catalog 条目一并移除，所以 `/v1/models` 与磁盘保持一致；
指向该目录的已结束任务会被标为 `canceled`，不再继续宣称 `done`。

**断点续传。** 传输中的文件写入 `<目标>.incomplete`，大小与目录清单一致时才 rename。暂停、
出错、进程被杀、重启，留下的是同一种状态；下一次发 `Range: bytes=<size>-` 接着追加。所有
尺寸都是 64 位（`curl_off_t`），单个文件超过 100 GiB、整个仓库几百 GiB 都不会撞上 2 GiB /
4 GiB 的假设。若远端忽略 `Range` 而返回 `200`，会先截断部分文件再写，不会把两份拼在一起。
`LOW_SPEED_TIME` 会把卡死的传输变成可重试的 `error` 任务，而不是一根永远停在 93% 的进度条。

**目录服务地址。** 默认 `https://hf-mirror.com`；`CWICK_HUB_ENDPOINT` 优先，其次
`HF_ENDPOINT`（与 `download_model.sh` 读的是同一个变量）。`GET /v1/config` 里的
`model_library` 是库根、`model_catalog` 是目录服务地址。

这些 endpoint 刻意**不做**的事：加载权重、更换常驻模型、剪枝或量化 checkpoint。允许下载一个
引擎尚未实现其架构的仓库——它会在清单里标为 `supported:false`，对它的切换请求会在**常驻
checkpoint 被卸载之前**以 400 点名架构拒绝。
