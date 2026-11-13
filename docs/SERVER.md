# OpenAI-compatible server reference

<p align="center"><a href="SERVER.md">English</a> · <a href="SERVER.zh-CN.md">简体中文</a></p>

`cwick serve` implements a deliberately small, strictly-checked subset of the
OpenAI API.

## Endpoints

- `GET /v1/models` (and `/models`) — every local model this instance knows about,
  resident first. Each entry carries `cwick.root`, `cwick.loaded` and
  `cwick.supported` (plus `cwick.unsupported_reason` when false), so a client
  can tell *what is answering* from *what could be* and from *what can never
  be loaded by this build*.
- `GET /v1/models/<id>` (and `/models/<id>`) — one entry in that shape, or 404
  `code:"model_not_found"`. This is OpenAI's retrieve-model call; agent
  frameworks probe it once with the name from their config, so answering it
  avoids a "model not found" that `/v1/chat/completions` would not have raised.
  The id is percent-decoded, so a checkpoint directory with a space works.
- `POST /v1/completions` (and `/completions`) — raw text continuation.
- `POST /v1/chat/completions` (and `/chat/completions`) — chat, using the
  CWICK chat template (`<|begin|><|system|>…<|end|><|user|>…<|end|><|assistant|>`).
- `POST /v1/responses` (and `/responses`) — the Responses dialect: `input`
  items in, a `response` object (or `response.*` events) out. Same prompt,
  same queue, same prefix cache as chat; see **Responses API** below.
- `POST /v1/messages` (and `/messages`) — the Anthropic dialect: `system` plus
  content blocks in, a `message` object (or `message_start … message_stop`
  events) out. Same engine and the same adapter rule; see **Messages API**
  below.
  Paths are matched after the query string is stripped, so the `?beta=true`
  the Anthropic client appends to every call is harmless, and a repeated
  `/v1/v1/` prefix is collapsed — a base URL that already ends in `/v1` is
  served instead of answered with `404 unknown endpoint`. The access log and
  the `--trace` line still print the path exactly as it arrived.
- `GET /v1/metrics` (and `/metrics`) — Prometheus text, vLLM naming.
  `cwick:model_loaded` (1/0) is the live residency bit; `cwick:model_info`
  keeps the labels of the last successfully loaded checkpoint.
- `POST /v1/auth/login`, `GET /v1/auth/status`, `POST /v1/auth/logout` — the
  single-user login endpoints. Login accepts `{"password":"..."}`, sets the
  HttpOnly `cwick_session` cookie and returns a Bearer token. When a password is
  configured, every other API/metrics/logs endpoint answers 401 without it; `/`,
  app HTML and the admin pages redirect to `/login`.
- `GET /v1/config` (and `/config`) — the options this process is running with.
  Read-only: everything in it is a lifetime choice (see below).
- `POST /v1/models/switch` — replace the resident model with another catalog
  entry, without a restart. A checkpoint downloaded after startup is discovered
  on demand here, so no refresh call is needed. An architecture this build has
  no kernels for, or a config that fails the model contract, is refused with a
  400 *before* anything is released (see **Switching the resident model**).
- `GET /v1/hub/models`, `GET /v1/hub/downloads`, `GET /v1/hub/search?q=…`,
  `POST /v1/hub/download`,
  `POST /v1/hub/download/<id>/<pause|resume|retry|cancel>` — the local model
  library and its download queue (see **Model library** below).
- `POST /v1/systemone` (also `/inference/v1/systemone`) — typed decisions read
  from one prefill (requires `--jev`); `GET` on either path reports the adapter
  the resident checkpoint is using right now (see **Decisions** below).
- `POST|GET|DELETE /v1/auth/keys` — API keys for non-browser clients; requires a
  login session (see **Authentication** below).
- `GET /v1/logs` (and `/logs`) — the process's own stdout/stderr, from the
  capture ring. `?since=<seq>&limit=<n>`.
- `GET /v1/logs/stream` — the same ring as a live SSE tail.
- `GET /`, `/web/app.css`, `/web/app.js` — the built-in console (chat, apps,
  models, logs, settings; metrics and trends live in settings), and `GET /apps/manifest.json` + `/apps/<id>/...` for
  the app bundles, only when `serve` was started with `--web`. Compiled into the
  binary; answered without taking the generation slot. Every other path 404s.
- `GET /v1/share` — LAN URL(s) and the tictactoe entry URL a phone can open;
  `GET /v1/share/qr?text=<url>` — the same address as a native CoreImage PNG.
  With the default loopback bind `shareable` is false; `--lan` binds `0.0.0.0`
  for a trusted same-Wi-Fi network (single-user password login; no TLS).
- `GET /v1/store`, `/v1/apps/<id>/{state,events,games}` — the apps' own data, in
  one sqlite database **per app** (`--store DIR`, `$CWICK_STORE`, else
  `~/.cwick/apps`, each app owning `<id>.db` and the server's keys living in
  `auth.db` beside them). `--store FILE` — a path that is a file or has a
  `.db`/`.sqlite` suffix — keeps the older single shared database.
  `state` is key → JSON, `games` is one row per match, `events` is the
  move-by-move replay log. Values and payloads are stored as the exact JSON
  text that arrived (parsed on the way in only to prove it *is* JSON), capped at
  1 MiB each; `<id>` is `[A-Za-z0-9_-]{1,64}` and is refused rather than escaped.
  Behind the same login as the model, and answered before the generation slot.

All of the read-only endpoints above are answered **before** the generation
slot, for the same reason `/metrics` is: watching a run must never queue behind
it. `POST /v1/models/switch` is answered there too, but refuses (409) when
anything is generating.

## What requests can include

- One generated answer per request (`n` and `best_of`, if present, must be `1`).
- `max_tokens` or `max_completion_tokens` to cap the answer.
- If none of `reasoning_effort` / `thinking` / `reasoning` / `enable_thinking`
  is present, the level defaults to **关** — the same as an explicit
  `reasoning_effort:"none"`. An absent field used to mean 极致 (32768 tokens
  of reasoning); see DECISIONS §55. Any explicit spelling still wins.
- Streaming via server-sent events, including `stream_options.include_usage`.
- `model` — when present, must be the model this server advertises (the
  config's `model_type`).
- `tools` (array of `{"type":"function","function":{name, description,
  parameters}}`, at most 64) and `tool_choice` (`"auto"`, `"none"`, or
  `{"type":"function","function":{"name":…}}` naming a declared function).
- `temperature` / `top_p` / `top_k` and `presence_penalty` /
  `frequency_penalty` are honoured (range-checked, then applied by the sampler).
  Fields the request omits keep the checkpoint's own `generation_config.json`
  values, so "top_p only" does not silently mean temperature 0.
- A `developer` message is the instruction turn, exactly like `system` (the
  checkpoint's template has one such role; rendering it as a user turn would
  turn an instruction into a request).
- Harmless defaults (`store: false`, `response_format: {"type":"text"}`,
  empty `stop`) are accepted because they change nothing.

## What gets refused (fail-closed)

These OpenAI features are not implemented and return an OpenAI-shaped HTTP 400
before the model is entered: non-default `stop`, logit bias,
logprobs, seeds, suffix/echo, structured `response_format`, multimodal content,
tools on `/v1/completions`, `tool_choice` without `tools`, `tool_choice:
"required"` (nothing forces the model to call a function), a `tool_choice`
naming an undeclared function, non-`function` tool types, and any unknown
field.

The decision endpoint refuses the same way: `stream`, `reasoning_effort` other
than `"none"`, a `thinking` block that is not off, and `images` are 400s that
name the field, because a decision is one JSON object produced with no decode.
A body over the request cap is a 413 on every endpoint, and a body that is not
the document an endpoint serves (malformed JSON, a missing required field, an
invalid question) is a 422 rather than a 400 — see **Decisions** below.

The server generates one answer at a time, but a concurrent generation request is
not refused: it takes a ticket and waits in arrival order, and every queued
request is served to completion. While it waits, `/v1/models`, `/v1/metrics` and
the other cheap endpoints keep responding. A queued client that disconnected is
dropped before it costs a prefill. 503 + `Retry-After` means a model switch or a
shutdown is in progress; 429 is only reachable when the server is shutting down
with a backlog it will never serve.

A prompt longer than the KV window is refused rather than silently truncated:
the engine returns the capacity it was given and the request fails with a 400
that names both numbers. Size the window at startup with `--context N`
`--password P`/`CWICK_PASSWORD` enables the single-user login; `--lan` always
enables it (generating and printing a password if none was given). Sessions are
in-memory, 12 h TTL, `HttpOnly; SameSite=Strict`; API clients use the Bearer
token returned by the login endpoint. There is no user database or multi-tenant
model. HTTP has no TLS.

(default 131072 for `serve`, i.e. >100K; `run` defaults to 4096, or the DSA
indexer's top-k on an FP8 checkpoint). The window is
fixed at load time — the KV cache is allocated once — so it cannot be changed
per request.

## Response shapes

Non-streaming chat:

```json
{"id":"chatcmpl-…","object":"chat.completion","created":…,"model":"…",
 "choices":[{"index":0,"message":{"role":"assistant","content":"…",
 "reasoning_content":null},"finish_reason":"stop"|"length"|"tool_calls"}],
 "usage":{"prompt_tokens":…,"completion_tokens":…,"total_tokens":…,
          "completion_tokens_details":{"reasoning_tokens":…}}}
```

`reasoning_content` is this server's own field (the checkpoint's thinking block);
`completion_tokens_details.reasoning_tokens` says how much of the answer was thinking,
the same number Responses reports in `output_tokens_details`. `service_tier` and
`system_fingerprint` are not emitted — both are optional in the schema and this server
has one tier and one process per model.

Streaming chat (`text/event-stream`): begin chunk (`delta.role=assistant`),
content delta (`delta.content`), finish frame (`finish_reason`), optional usage
frame, then `data: [DONE]`. Raw completions use `text_completion` chunks with
`delta.text`.

## Responses API

`POST /v1/responses` is a second dialect over the same engine, not a second engine. The
request is translated into the chat conversation the rest of the server already
validates, so the prompt renderer, the single-slot queue, the prefix cache and the
thinking budget behave exactly as they do for `/v1/chat/completions`.

**Request fields** (anything else is a 400 that names the field):

| field | behaviour |
|---|---|
| `input` | a string, or an array of items: `message` (roles `user`/`assistant`/`system`/`developer`, content string or text parts), `function_call`, `function_call_output`, `reasoning` (dropped — hidden chain is not replayable input) |
| `instructions` | prepended as the system turn, ahead of `input` |
| `tools` | flat Responses declarations (`{type:"function",name,description,parameters}`); translated to the nested chat form. Only `function` tools exist here |
| `tool_choice` | `auto`, `none`, or `{"type":"function","name":…}`; `required` needs constrained decoding and is refused |
| `max_output_tokens` | the output budget (`max_tokens` accepted as an alias) |
| `temperature`, `top_p`, `stream` | as in chat |
| `reasoning` | `.effort` maps onto the five intensity levels (`minimal`/`low`/`medium`/`high`/`xhigh`/`none`); `.summary` publishes the reasoning block as a summary |

Accepted but inert: `metadata`, `user`, `include`, `service_tier`,
`prompt_cache_key`, `safety_identifier`. Refused by name: `previous_response_id` /
`conversation` (no durable response store — replay the full `input` items), `store:true`,
`background:true`, `truncation:"auto"`, `top_logprobs`, `max_tool_calls`, and image/file
content. `parallel_tool_calls` and `text` are echoed into the response object; the first is
not enforced, the second is text-only.

**Response** (buffered):

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

The second half of the object echoes the request. `tools`, `tool_choice` and
`parallel_tool_calls` are not optional in the schema, and the rest is what a client
compares its request against; the same fragment is appended to `response.created`, so the
streamed envelope and the final body agree.

Item ids are derived from the response id (`…-rs`, `…-msg`, `…-fc<i>`), because the item
announced by `output_item.added` and the item inside the terminal frame are the same item
and a client correlates them by id.

A reasoning item exists whenever the turn reasoned — the stream announces it as soon as
the first thinking token lands — but its `summary` is filled in only when
`reasoning.summary` asked for it. Otherwise it is `[]`, which keeps the chain private
without changing the shape of the output list. A turn with no tool call always carries a
message item, even when the model spent its budget thinking and the text is empty.

`status` is `incomplete` with `incomplete_details.reason = "max_output_tokens"` when the
turn stopped because the budget ran out; that is the only place this dialect can say
"truncated" (Anthropic says `stop_reason:"max_tokens"` instead).

**Streaming** uses named SSE events and has no `data: [DONE]` sentinel. Every event
carries a monotonically increasing `sequence_number`, and the terminal event is
`response.completed`, `response.incomplete` (budget) or `response.failed` (the response
exists but could not finish; the response object then has `status:"failed"` and `error`
filled in):

```
response.created → response.in_progress
→ response.output_item.added → response.content_part.added
→ response.output_text.delta …→ response.output_text.done
→ response.content_part.done → response.output_item.done
→ (one added → arguments.delta → arguments.done → done per function_call)
→ response.completed | response.incomplete | response.failed
```

Reasoning streams as `response.reasoning_summary_text.delta` when a summary was asked
for, otherwise as `response.reasoning_text.delta`. `response.output_text.delta` and
`response.output_text.done` carry `logprobs: []` — the field has no default in the
schema; this server does not compute log probabilities, so the list is always empty.

Function calls are announced by `output_item.added` with empty `arguments`, then one
`response.function_call_arguments.delta` carrying the whole argument string,
`response.function_call_arguments.done`, and finally `output_item.done`. The call is parsed
out of the model's own protocol only once it is complete, so the payload cannot be
dribbled out token by token — the events exist because that is where a client looks for
it, and one delta is the honest shape.

**Where this deviates from the reference.** The durable parts of the API are absent, and
requests that need them are refused rather than half-served: `previous_response_id`,
`conversation`, `store:true`, `background:true`, `truncation:"auto"`, `top_logprobs`,
`max_tool_calls`, non-`function` tools and image/file input. `parallel_tool_calls`
is echoed but not enforced (the checkpoint decides how many calls it emits), `text.format`
is text-only, and `instructions` must be a string (the schema also allows an item array).
`metadata`, `user`, `include`, `service_tier`, `prompt_cache_key` and `safety_identifier`
are accepted and ignored. Everything else in the create request is implemented.

## Messages API

`POST /v1/messages` is Anthropic's dialect — what Claude Code and the Anthropic
SDKs send. It is an adapter, exactly like Responses: `anthropic_parse` folds the
blocks into the chat conversation the rest of the server already validates, and
`messages_body` / `msg_stream_*` write the reply. Prompt rendering, the
single-slot queue, the prefix cache and the thinking budget are the same code.

**Request fields** (anything else is a 400 that names the field):

| field | behaviour |
|---|---|
| `messages` | `{role: user\|assistant, content}`; content is a string or an array of blocks: `text`, `thinking` / `redacted_thinking` (dropped — a hidden chain is not replayable), `tool_use` (assistant), `tool_result` (user) |
| `system` | a string or `text` blocks; it becomes the system turn and precedes `messages` |
| `max_tokens` | **required**; there is no default here |
| `tools` | flat client tools: `{name, description, input_schema}`. `type` may be absent or `"custom"`; server tools (`web_search`, …) are refused by name |
| `tool_choice` | `{"type":"auto"}`, `{"type":"none"}`, `{"type":"tool","name":…}`. `{"type":"any"}` needs constrained decoding and is refused |
| `temperature`, `top_p`, `top_k` | as in chat |
| `stream` | SSE with named events |
| `thinking` | `.type: "disabled"` answers without thinking; `"enabled"` maps `.budget_tokens` onto the five 思考强度 levels (≤512 低, ≤2048 中, ≤8192 高, ≤16384 超高, above 极致); `"adaptive"` maps to 高 |

Accepted but inert: `metadata`, `service_tier`. Refused by name: non-default
`stop_sequences` (the engine stops on its own delimiters), `image` / `document`
/ `audio` / `video` blocks (text-only server), and a `tool_result` in an
assistant turn or a `tool_use` in a user turn (the pairing is wrong, and
guessing would send the wrong prompt).

**Content mapping.** An assistant turn becomes one chat message with `content`
plus `tool_calls` (the `tool_use` `input` object, re-serialised as the
arguments string the chat layer expects). A user turn's `tool_result` blocks
become `tool` messages — several results in one turn stay in order — and the
remaining text becomes the user message after them.

**Response** (buffered):

```json
{"id":"msg-…","type":"message","role":"assistant","model":"…",
 "content":[{"type":"thinking","thinking":"…","signature":""},
            {"type":"text","text":"…"},
            {"type":"tool_use","id":"…","name":"…","input":{"path":"a.c"}}],
 "stop_reason":"end_turn","stop_sequence":null,
 "usage":{"input_tokens":…,"output_tokens":…}}
```

`signature` is empty: this engine has no signing key, and it never verifies a
replayed chain — which is also why `thinking` blocks in the request are dropped.
`stop_reason` is `tool_use` when the model called something, `max_tokens` when
the output budget ran out, `end_turn` otherwise. When the prefix cache covered
part of the prompt, `usage.cache_read_input_tokens` reports how much.

**Streaming** uses named events and ends at `message_stop`:

```
message_start → (content_block_start → content_block_delta* → content_block_stop)*
→ message_delta → message_stop
```

`content_block_delta` carries `text_delta`, `thinking_delta` or
`input_json_delta` depending on the open block. A `tool_use` block opens with
empty input and its arguments arrive as `input_json_delta` fragments, so a
client can show the call forming — this dialect *does* stream arguments, unlike
Responses. `message_delta` is the only place a streamed turn reports its
`stop_reason`, and it repeats `output_tokens` **and** `input_tokens` (plus
`cache_read_input_tokens` when a prefix hit covered part of the prompt), so a
client that only reads the closing events still has complete usage. A failure
after the head is on the wire is a named `error` event whose payload is
`{"type":"error","error":{"type":"api_error","message":…}}` — the same shape a
buffered failure returns.

**Where this deviates from the reference.** There is no
`POST /v1/messages/count_tokens` (the engine has no token-count entry point that
does not also render a prompt). Refused by name, never silently downgraded:
non-empty `stop_sequences`, `image` / `document` / `audio` / `video` blocks, and
`{"type":"any"}`. `thinking.budget_tokens` is accepted below the documented 1024
minimum (it maps to 低) and `enabled` without a budget maps to 中; `ping` events
are not sent; `thinking.signature` is empty because nothing is signed or
verified. `metadata` and `service_tier` are accepted and ignored, and
`tool_result.is_error` is passed through as ordinary result text.

**Authentication.** Anthropic clients send the key as `x-api-key` instead of
`Authorization: Bearer`; this server accepts either (the header parse already
did, for every endpoint). `anthropic-version` is read and ignored: there is one
implementation and it is not versioned by the protocol.

## Tool calling

`tools` are advertised to the model in the checkpoint's own protocol, copied
from `zai-org/GLM-5.3-Flash/chat_template.jinja` rather than invented:

```
<tools>
{"name": "get_weather", "description": "...", "parameters": {...}}
</tools>

For each function call, output the function name and arguments within the
following XML format:
<tool_call>{function-name}<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>
```

A second template nests `<function=NAME>` / `<parameter=KEY>VALUE</parameter>`
inside the same wrapper; `src/tools.c` rewrites it into the flat grammar above,
so one parser covers every checkpoint. An answer containing well-formed
`<tool_call>` blocks is returned as `message.tool_calls[]` with
`finish_reason: "tool_calls"`; the prose before the first block stays in
`content`.

**Streaming tool calls (DECISIONS §58).** A streaming turn no longer stays
silent while the call is being written. As soon as the model writes the
function name, each dialect opens the call with an empty `arguments` and then
appends argument shards as parameters close; concatenating them yields exactly
the object in the final payload.

| dialect | opening frame | argument shards |
|---|---|---|
| chat | `delta.tool_calls[0]` with `id` / `type` / `function.name` | `delta.tool_calls[0].function.arguments` |
| responses | `response.output_item.added` (`function_call`) | `response.function_call_arguments.delta` |
| messages | `content_block_start` (`tool_use`) | `content_block_delta` (`input_json_delta`) |

`delta.cwick_tool` is still emitted for the bundled console. Shards are
produced by feeding a closed prefix of the in-flight block to the same
`cwick_tools_extract()` the buffered path uses, so the wire bytes and the final
object cannot disagree; at the finish the two are reconciled. Only the first
call is streamed (further calls arrive whole), and a block in another dialect
falls back to the end-of-turn form unchanged.

Shards land **inside** a parameter value too (DECISIONS §59): an agent's tool
call is usually one long parameter — a shell script, a file body — and sharding
only at `</parameter>` left that entire value arriving in the last frame. A
value is streamed once it is certain to be a JSON string (not `{ [ " -` or a
digit, and not still a prefix of `true`/`false`/`null`), in steps of 12 bytes,
with trailing whitespace, an unfinished `</parameter>` and half a UTF-8
character held back. Numeric, boolean and object values therefore still arrive
whole, and every intermediate prefix remains a byte-prefix of the final
`arguments`, so a concatenating client cannot disagree with a buffered one.

Nothing constrains decoding, so the failure modes are handled rather than
hoped away:

- a malformed or unterminated `<tool_call>` → HTTP 500, never a half-parsed
  call handed to the client;
- `tool_choice` naming a function the model did not call → HTTP 500 with a
  message naming the function.

## Logging on stdout

Every diagnostic goes to **stdout** (`src/log.c`), line-atomic and
timestamped:

| level | flag | content |
| --- | --- | --- |
| info | (always) | startup, the memory budget, one line per request, one line per generation, shutdown |
| verbose | `--verbose` | per-request detail, prefill/decode progress with RSS, the full budget table, cache and SSD counters |
| trace | `--trace` | what the model saw and produced: system/user messages, declared tools, the rendered prompt, the output, and every tool call |

```
cwick: [info   +1.34s] [init] OpenAI-compatible server listening on http://127.0.0.1:8000/v1 (model glm-5.3-flash)
cwick: [info  +29.99s] [req 1] POST /v1/chat/completions (249 bytes body)
cwick: [info  +80.23s] [gen] prefill 8/147 — rss 53.75 GiB of 102.40 GiB ceiling, expert cache 36.27 GiB pinned
cwick: [info +366.28s] [req 1] POST /v1/chat/completions -> 503 in 336470.8ms interrupted: shutdown
```

`CWICK_SERVER_LOG=0` silences the request/generation lines,
`CWICK_SERVER_LOG=verbose` raises the detail level without `--verbose`.

That stdout is also readable over HTTP. `src/logcap.c` tees fd 1 and fd 2 into
two pipes; a pump thread echoes every byte back to the saved descriptors and
pushes each completed line into a bounded ring (2000 lines / 4 MiB). Because the
tee sits at the descriptor level rather than inside `log.c`, Metal diagnostics,
Objective-C `NSLog` output and stray `fprintf(stderr, …)` land in it too — the
lines a GUI most needs are exactly the ones `log.c` never saw.

```
GET /v1/logs?since=0&limit=3
{"capture":true,"seq":18,"oldest":1,"dropped":0,"retained":17,
 "lines":[{"n":1,"src":"out","t":1792299405400,"msg":"cwick: … [init] log capture on: …"},
          {"n":2,"src":"out","t":1792299405401,"msg":"cwick: …"},
          {"n":3,"src":"out","t":1792299405402,"msg":"cwick: …"}]}
```

`src` is `out` or `err`; `t` is the wall clock at capture, in ms since the
epoch; `n` is the cursor to resume from (`since=last_n+1`). A cursor that has
already been evicted is clamped to `oldest` rather than returning an empty tail,
and `dropped` says how many lines went before their time so a client can be
honest about an incomplete history instead of silently showing one.

`/v1/logs/stream` sends `{"type":"hello","seq":…,"capture":…,"dropped":…}` and
then one `{"type":"line",…}` frame per line, woken by the ring's condition
variable rather than by polling. It closes on client hangup or process stop (the
stop flag is checked between waits, so a Ctrl+C is honoured within 400 ms even
with a tail open — measured at ~230 ms), and emits a `: idle` comment every 15 s
so nothing in front of it reaps the connection.

With `--trace`, prompts and outputs are in that ring and therefore on these
endpoints. That is what the flag is for; the listener is loopback-only unless
`CWICK_ALLOW_PUBLIC=1`.

## Authentication

Two credentials, because "who is calling" has two very different answers:

| | session | API key |
| --- | --- | --- |
| for | the browser console | third-party agents / scripts |
| obtained by | `POST /v1/auth/login` with the password | `POST /v1/auth/keys` (needs a session), or `--api-key` / `CWICK_API_KEY` |
| sent as | `cwick_session` cookie, or `Authorization: Bearer` | `Authorization: Bearer <key>` or `X-API-Key: <key>` |
| lifetime | 12 h, memory only, gone on restart | until revoked, persisted as a hash |
| can manage keys | yes | **no** (401) |

`--password P` / `CWICK_PASSWORD` enables checking; `--lan` forces it; so does
`--api-key`, since an unchecked key protects nothing. Without a password the
server generates one and prints it on the `auth:` startup line. Passwords are
held as PBKDF2-HMAC-SHA256 (120k iterations, random salt) and never written to
disk; five wrong attempts lock the endpoint for 60 s (`Retry-After`).

### API keys

```bash
# mint one from the console session
curl -s -b cookies.txt -X POST localhost:8000/v1/auth/keys \
  -H 'Content-Type: application/json' -d '{"name":"cursor"}'
{"id":"mUDhRy6Ywlyu","name":"cursor","prefix":"cwick_6My8BTo2",
 "created_at":1792342101,"secret":"cwick_6My8BTo2Tso5J5vCmV91cMqlGaluDx6s",
 "note":"shown once; the server keeps only its hash"}

# list / revoke
curl -s -b cookies.txt localhost:8000/v1/auth/keys
{"keys":[{"id":"mUDhRy6Ywlyu","name":"cursor","prefix":"cwick_6My8BTo2",
          "created_at":1792342101,"last_used_at":1792342104,"enabled":true}]}
curl -s -b cookies.txt -X DELETE 'localhost:8000/v1/auth/keys?id=mUDhRy6Ywlyu'
{"status":"revoked"}
```

Properties that were chosen deliberately:

- **160 random bits** (`cwick_` + 32 base62 chars), stored as a *single SHA-256*.
  PBKDF2 is for low-entropy human passwords; stretching a 160-bit random secret
  buys nothing and would tax every request.
- **The plaintext is never persisted**, only `id`, `name`, `prefix` and the hash
  in the `auth_keys` table. `prefix` is the display form (`cwick_6My8BTo2`) so a
  list is readable and an access log can say `auth=key:cwick_6My8BTo2` without
  ever carrying the key.
- **Keys survive restarts**; sessions do not. An agent is configured once.
- **A key cannot mint keys.** `POST /v1/auth/keys` requires a session and answers
  401 to a key, so a secret that leaked into a log or a backup cannot be used to
  plant more.
- A key is checked after the session lookup, and a *session token is not a valid
  key* (nor the reverse): the two are separate so a leaked one is identifiable.
- Malformed hashes are refused on load rather than silently ignored; a key that
  fails to persist is rolled back out of memory, so memory and disk never
  disagree about what is accepted.

## Switching the resident model

`--models-dir DIR` (repeatable, max 8) makes other local checkpoints *listable*;
nothing is loaded until a switch asks for it. A directory qualifies if it holds
`config.json` and one of `resident/tensors`, `model.safetensors.index.json`,
`model.safetensors`; the advertised id is `model_type`, then
`architectures[0]`, then the directory name, de-duplicated as `@2`, `@3`, … The
resident entry is never renamed, so `data[0]` stays stable.

**"Downloaded" is not "loadable".** Every entry in `/v1/models` and
`/v1/hub/models` carries `supported`, plus `unsupported_reason` when it is
false. The library fetches any repository, but only architectures with kernels
in this build can become resident. The verdict comes from `config.json`'s
`model_type` (the inner `text_config` one when present, exactly as the loader
reads it) and is computed once when the directory is discovered — a polled
model list therefore never parses config files while holding the generation
lock. The Web console disables the switch button for an unsupported entry.
This build implements the GLM-5.3, DeepSeek-V4.1 and MiniMax-M3.1 kernels plus
the small `cwick setup` checkpoints; registering a further architecture means
adding its kernels and one line to the type table, not a claim that the list
is closed forever.

Release checkpoints additionally advertise a *served name* derived from their
directory basename, because `model_type` is a training-run artifact no client
would guess (`glm5_next`, `deepseek_v41_text`): `zai-org/GLM-5.3-Flash` serves
as **`glm-5.3-flash`** and `deepseek-ai/DeepSeek-V4.1-Flash-1017` as
**`deepSeek-v4.1-flash-1017`**. That name is what `/v1/models` lists, what
completions echo back in `model`, and what `/v1/models/switch` accepts — the
checkpoint's own `model_type` keeps working everywhere too, so nothing that
learned the old spelling breaks. Other checkpoints are unaffected and keep
their config.json name.

```bash
curl -s localhost:8000/v1/models/switch \
  -H 'Content-Type: application/json' -d '{"model":"glm-5.3-flash"}'
```

| status | body `status` | when |
| --- | --- | --- |
| 200 | `switched` | the new model is resident and serving |
| 200 | `already_resident` | a no-op: that root is already loaded |
| 404 | — | no catalog entry with that id |
| 400 | `error` | the target architecture has no kernels here, or its config fails the model contract; **the resident model is untouched** |
| 409 | `error` | a slot is busy, a request is queued, the tokenizer is held, or a swap is running |
| 500 | `error` | the replacement engine or its tokenizer failed to load after teardown |

**Validation precedes teardown.** A switch first re-reads the target's
`config.json` and runs the full model-contract check as a dry run; only after
that passes does it release the old engine, tokenizer, Metal arena and prefix
cache. An unknown `model_type` is refused with a 400 that names it instead of
falling through to the default geometry parser and failing on an unrelated
field (a nested config with a null `text_config.pad_token_id` used to report
"missing pad_token_id" even when the outer object had one). In both 400 cases
nothing is released: the resident model keeps serving, `cwick:model_loaded`
stays 1, and the log has no `switching to` / `released` lines.

While a swap is in progress, generation requests are answered 503
`code:"model_switch"` rather than queued: the model they were validated against
is being torn down. If the replacement fails *after* teardown (corrupt weights,
a missing tensor, a tokenizer that cannot load), the server keeps running with
no model and answers 503 `code:"no_model"` until another switch succeeds — it
never falls back to a half-initialised engine. `cwick:model_loaded` is `0` in
that state. `cwick:model_info` deliberately keeps its last labels because
metric names and labels must stay stable across a process lifetime, so a
dashboard must use the gauge to distinguish "serving" from "no model".

`GET /v1/config` reports what the process is running with:

```json
{"model":"…","model_root":"…","host":"127.0.0.1","port":8000,"slots":1,
 "max_tokens":1000000,
 "max_request_bytes":8388608,"context_window":512,"catalog_size":2,
 "web":true,"verbose":true,"trace":false,
 "prefix_cache":{"enabled":false,"block_tokens":4096},
 "log_capture":true,"log_retained":300,"models_dirs":["/models"]}
```

There is deliberately no write endpoint: context window, memory fraction, slot
count and the byte caps are decided at load time, so "changing" them over HTTP
would be a lie. `0` means no limit.

## Decisions

`--jev` (or `CWICK_JEV=1`) enables the decision endpoint. It answers from the
model already resident: no second checkpoint, no fine-tune, no second process.
The published path is `/inference/v1/systemone`, so a client pointed at
`http://host:port/inference` lands on it; `/v1/systemone` and `/systemone` are
short forms, and all three answer `GET` and `POST` alike.

`POST /v1/systemone`

```json
{"model": "glm-5.3-flash",
 "state": {"board": ["X", null, "O"], "turn": "O"},
 "questions": {
   "move": {"type": "choice", "instructions": "Which cell gives O the best win rate?",
            "criteria": {"c0": "top-left", "c4": "centre"}},
   "risk": {"type": "score", "instructions": "How close is O to losing?",
            "criteria": ["safe", "loose", "lost"]},
   "accept": {"type": "noul", "instructions": "Does O already have three in a row?",
              "criteria": {"true": "O has three in a row", "false": "it does not"}}
 }}
```

- `state` is required and non-empty. A string is used as-is; any other JSON
  value (an object or array) is serialized and placed inside `<state>`, so a
  caller that keeps its state as JSON gets its own fields back in the prompt.
- `questions` is a map from the caller's own id to a question; the ids come
  back as the keys of `answers`. Every question requires `type`
  (`choice` / `score` / `noul`) and `instructions` — the question in plain
  language. There is no fallback text: a decision asked with the wrong question
  is worse than a refusal.
- `choice` needs `criteria` as an object (key → description) or an array of
  names; the response carries `choice`, `probabilities` and `confidence`.
- `score` needs an ordered array of 2 to 10 level descriptions, lowest first,
  and returns the probability-weighted index, a `legend` and `probabilities`.
- `noul` takes `criteria` naming its two sides — `{"true": …, "false": …}` or a
  two-item array, true side first — and returns `noul` = P(the statement
  holds). The two sides have a fixed order internally, so a request that writes
  `false` before `true` still reports P(true); the default is yes/no.
- Up to ten questions per request and 26 options per question; each question is
  one prefill, and every option of a question is read from that same pass.
- `model` is optional; when present it must name the resident checkpoint (served
  alias or `model_type`, case-insensitive) — anything else is a 400 naming the
  resident model, because a decision returned under a stale id is
  indistinguishable from a correct one.
- `temperature` (default 1.0) only rescales the readout distribution; nothing is
  sampled, so the same request on the same checkpoint is deterministic.
- `stream`, `reasoning_effort` other than `"none"`, a `thinking` block that is
  not off, and `images` are refused by name: a decision is one JSON object
  produced with no decode and no reasoning pass, so accepting any of them would
  answer a question the caller did not ask.

Response:

```json
{"object": "systemone.decision", "model": "glm-5.3-flash",
 "adapter": {"family": "glm", "prompt": "glm-chat-closed", "readout": "letter",
             "letters": 26, "letters_total": 26, "first_missing": null,
             "space_lead": true},
 "answers": {"move": {"type": "choice", "choice": "c4",
                      "probabilities": {"c0": 0.12, "c4": 0.88},
                      "confidence": 0.76},
             "risk": {"type": "score", "score": 1.7,
                      "legend": {"0": "safe", "1": "loose", "2": "lost"},
                      "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
                      "confidence": 0.4},
             "accept": {"type": "noul", "noul": 0.02}},
 "usage": {"input_tokens": 118, "cached_input_tokens": 0, "output_tokens": 0,
           "prefilled_tokens": 118, "reused_tokens": 0}}
```

`answers` holds exactly what each type promises: `noul` is one probability,
`choice` the winning key with its distribution, `score` the expected level with
a per-level distribution and the legend that maps indices back to labels.

`confidence` is `(K·Σp² − 1)/(K − 1)`: 0 for a uniform distribution, 1 for a
point mass. It says how concentrated the model is, not how likely it is to be
right.

`usage.cached_input_tokens` is the reference field for "state tokens served
from the cache"; `prefilled_tokens` and `reused_tokens` sit next to it because
they are the two numbers the log line reports, so a caller can see the saving
rather than infer it.

`GET /v1/systemone` returns the same `model` and `adapter` objects plus the
limits (`questions_max`, `options_max`, `state_max_bytes`,
`instructions_max_bytes`, `levels_min`, `levels_max`). The adapter is
resolved from the resident checkpoint: GLM and DeepSeek use the chat turn with
the thinking block already closed, MiniMax its plain chat turn, and the option
letters are probed against the loaded tokenizer. It is re-resolved after every
`POST /v1/models/switch`, so a switch changes the interface as well as the
weights. `make jevprobe` prints the letter table for a tokenizer without
loading any weights.

Errors carry the reference protocol's split, because a client can act on it:
422 for a body that is not a decision request (malformed JSON, a missing
required field, an invalid question), 400 for a well-formed request this server
will not serve (a stale model id, a limit, an unsupported field, a tokenizer
that cannot spell an option as one token), 413 for a body over the request-size
cap, and 503 when no model is loaded or the queue is full (the endpoint takes a
generation slot, so a decision and a chat turn cannot overlap).

**Conformance.** The wire format follows the published decision protocol; these
are the deliberate differences, so a client is not left to discover them.

| Reference | Here |
|---|---|
| up to 256 questions per request | 10 (`goal`: one state, a handful of questions) |
| up to 208 options per `choice` | 26 — the readout is one letter token per option |
| `model` is a path like `accounts/x/models/y` | the resident model's served alias or `model_type` |
| no `temperature` | `temperature` rescales the readout (extension) |
| response is `model`/`answers`/`usage` | plus `object` and `adapter` (extension) |
| also served as an OpenAI-shaped `/v1/decisions` | not implemented |

`state` (4 MiB), question text (64 KiB), score levels (2–10) and the body cap
follow the reference.


**Logging.** Every line goes to stdout with the wall-clock time and the request
id (`[req N]`). A `POST /v1/systemone` prints, in order: the request
(`decision: N question(s) ... stream=false thinking=off`), one
`question i/N` line per prefill, the engine's `prefill ... decision readout`
start, the token/layer progress heartbeats, `prefill done: N tokens in X ms
(Y tok/s)`, the readout for each question (`-> key p=... (runner-up ...),
confidence=...`), and `decision done: N question(s), T input tokens, P
prefilled (R reused), W ms wall, R tok/s prefill`. A refusal prints
`systemone: decision refused — <reason>` plus the usual 4xx access-log line. `--trace` additionally prints
each question's rendered prompt (system + user) and the completed decision
object. The `cwick:prefill_*` gauges move while the decision runs, so
`/v1/metrics` and the console show that prefill too. Progress lines are
emitted only for live HTTP requests: CLI numeric verifiers calling
`cwick_engine_prefill_logits` without a bound request tag stay silent, so
machine-readable output is unchanged.

**Shared prefix.** Every question in a request begins with the same
`system + <state>` turn, so the endpoint encodes all of them first, takes the
longest common *token* prefix, prefills that once, and lets each question pay
only for its own tail (`question i/N: prefill S tokens, P reused from the
shared prefix`). `usage` reports `input_tokens`, `cached_input_tokens`,
`prefilled_tokens` and `reused_tokens`, so the saving is visible instead of
inferred.

This is enabled only on backends whose cross-token state is addressed by
position and can be continued exactly. MiniMax (GQA + KV) reproduces the
uninterrupted readout bit for bit — measured `|dp| = 0.0` and identical choices
on a three-question request. DeepSeek (compressed KV + indexer pool) and GLM
(DeltaNet/DSA pools) accumulate state that a rewrite at an arbitrary position
does not undo: the same request changed answers there (`|dp|` 0.56 and 0.41),
so both keep the N-full-prefills path. `CWICK_JEV_NO_PREFIX_REUSE=1` forces it
for any backend, `CWICK_JEV_REUSE_ALL=1` enables it on the others for
measurement. The rationale and every measurement are in
`docs/DECISIONS.md` §44.

## App storage

`--store FILE` (or `$CWICK_STORE`, else `~/.cwick/apps.db`) names the sqlite file
that holds the built-in apps' own data. It is opened with WAL and
`synchronous=NORMAL`, wrapped in one mutex (every connection has its own thread,
and a move must not wait behind a decode), and created with `IF NOT EXISTS`:

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

curl -s 'localhost:8000/v1/store?app=tictactoe'   # one app's own numbers
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

Rules that matter:

- every value and payload must be JSON; it is **stored as the exact text that
  arrived** and returned verbatim, so an app's fields survive a round trip
  (including number formatting — `0.42` stays `0.42`);
- 1 MiB per value/payload, 5000 rows per read, `id` = `[A-Za-z0-9_-]{1,64}`;
- a bad app id (`../etc`, quotes, spaces) is a 400, never an escaped query;
- `DELETE /v1/apps/<id>/games?game_id=G` removes the game **and** its events in
  one transaction, because a half-deleted game leaves orphan steps the replay
  view would show without a parent;
- a store that cannot be opened is logged at startup and answered 503; the
  server keeps serving the model, and the tic-tac-toe app falls back to
  `localStorage` (which it always mirrors into anyway);
- each app's database is a separate sqlite handle opened on first use and cached
  under a mutex, so a browser tab and a phone on the LAN see the same file
  without re-running the schema per request. `--store ./apps` from a checkout
  puts them next to the app sources; `tools/embed_web.c` skips every database
  suffix when it embeds `apps/`, so runtime data never enters the binary.

The endpoints sit behind the same single-user login as the model: `/v1/store`
is not in the public-API allowlist, so an unauthenticated request is 401 and an
app page navigation is a 302 to `/login?next=…`.

## Stopping

`Ctrl+C` (SIGINT/SIGTERM) stops the server. A signal arriving while idle breaks
the `poll()` loop through a self-pipe (macOS restarts `accept()` after a
handler returns, which is why the old loop only noticed Ctrl+C once the next
request arrived). A signal arriving mid-decode aborts the generation at the
next token boundary — the in-flight request gets an OpenAI-shaped 503
`server is shutting down (interrupted)` — then the listener closes and the
process exits. Pressing Ctrl+C a second time exits immediately (status 130).

## Usage

```bash
./cwick serve --root <model-root> --host 127.0.0.1 --port 8000 --verbose
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm_5_3_flash_synthetic","stream":true,"messages":[{"role":"user","content":"Hi"}]}'
```

Keep the server on loopback unless you add your own authentication and network
boundary.

## Model library

Five endpoints, all answered before the generation slot is taken, so a slow
catalog request or a multi-hundred-GiB download cannot delay a chat turn. They
are served by the same process and the same login session; the browser never
talks to the remote host itself, which also avoids CORS entirely.

| method | path | answer |
|---|---|---|
| GET | `/v1/hub/search?q=<text>&limit=<n>` | `{"models":[…],"count":N,"endpoint":…,"error":…}` — a small projection of the remote catalog: id, downloads, likes, tags, pipeline tag, gated/private. `limit` is 1–50 (default 12). |
| GET | `/v1/hub/models` | `{"root":"./models","models":[…]}` — every directory under the library root, with bytes, file count, partial bytes, `complete`, `resumable`, `supported` and (when false) `unsupported_reason`. |
| DELETE | `/v1/hub/models/<name>` | remove a downloaded checkpoint. 200 `{"deleted":true,"path":…,"bytes":N,"files":N}`; 404 for an unknown or unsafe name; 409 for the resident model or a directory an active download is still writing. |
| GET | `/v1/hub/downloads` | `{"jobs":[…]}` — the live job table: state, progress, bytes, files, current file, rate, ETA and error. |
| POST | `/v1/hub/download` | body `{"repo":"owner/name"}` → one job object. Idempotent for an already-active repository. |
| POST | `/v1/hub/download/<id>/<action>` | `pause`, `resume`, `retry` or `cancel` → the updated job object. |

Job states: `queued` → `listing` → `running` → `done`, with `paused`, `error`
and `canceled` as exits. The UI polls `/v1/hub/downloads`; there is no push
stream, so the job table is the only shared state.

**Where files go.** `./models/owner/name/<path>`, with a `.cwick-repo` marker
holding the full repository id. The startup scan registers both a flat
`./models/<name>` layout and this owner/name layout in the model catalog, so a
download finished earlier is switchable at startup, and one finished during
this process is discovered by `POST /v1/models/switch` on demand. A directory
is only "complete" when `config.json` exists and no `.incomplete` file remains.

**Deleting.** `DELETE /v1/hub/models/<name>` accepts `owner/name`, a bare
directory name, or a flat `./models/<name>` entry. The name is percent-decoded
and refused if it contains `..`, a backslash, an absolute path, or an empty or
`.` component; the resolved directory must sit strictly inside the library root
after symlinks are resolved, and the root itself is never deletable. Removal is
recursive and does not follow symlinks. The resident checkpoint is refused with
409 (switch away first) — deleting weights a running engine has mapped would
break it — and a directory a download worker is still writing is refused with
409 (cancel the job) so a resumable transfer never becomes a corrupt one. On
success the catalog entry is dropped in the same request, so `/v1/models` and
the disk agree, and finished jobs pointing at the removed directory are marked
`canceled` instead of continuing to claim `done`.

**Resume.** Each in-flight file is written to `<target>.incomplete` and renamed
only when its size matches the listing. A pause, an error, a killed process and
a reboot therefore leave exactly the same state; the next run sends
`Range: bytes=<size>-` and appends. All sizes are 64-bit (`curl_off_t`), so
single files above 100 GiB and repositories of a few hundred GiB are handled
without a 2 GiB or 4 GiB assumption. If the remote ignores `Range` and answers
`200`, the partial file is truncated instead of receiving a second copy.
`LOW_SPEED_TIME` turns a wedged transfer into an `error` job that can be
retried, rather than a progress bar frozen at 93%.

**Catalog host.** `https://hf-mirror.com` by default; `CWICK_HUB_ENDPOINT`
overrides it, then `HF_ENDPOINT` (the same variable `download_model.sh` reads).
The library root is reported by `GET /v1/config` as `model_library`, and the
host as `model_catalog`.

What these endpoints deliberately do *not* do: load weights, change the
resident model, or prune/quantize a checkpoint. Downloading a repository whose
architecture the engine does not implement is allowed — it stays visible in the
inventory with `supported:false`, and a switch to it is refused with a 400
that names the architecture *before* the resident checkpoint is unloaded.
