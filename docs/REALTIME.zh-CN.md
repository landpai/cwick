[English](REALTIME.md)

# Realtime 语音：Sophon-Realtime-32B-Tag

`--rt <checkpoint 目录>` 注册全双工语音管线。它和其它 checkpoint 一样是
**同一时刻只驻留一个模型**：加载期间文本接口返回 503，切回文本 id 会释放它。
WebSocket 端点为 `/v1/realtime?model=Sophon-Realtime-32B-Tag`，认证复用 HTTP
的登录会话 / Bearer token。

```sh
./cwick serve --root ./models/xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic \
  --port 8002 --web --backend metal \
  --rt ./models/adventists-ai/Sophon-Realtime-32B-Tag \
  --password cwick-local --verbose --trace
# 另一个终端：
curl --noproxy '*' -X POST http://127.0.0.1:8002/v1/auth/login \
  -H 'Content-Type: application/json' -d '{"password":"cwick-local"}'
curl --noproxy '*' -X POST http://127.0.0.1:8002/v1/models/switch \
  -H 'Authorization: Bearer <token>' -H 'Content-Type: application/json' \
  -d '{"model":"Sophon-Realtime-32B-Tag"}'
```

`tools/rtws` 是报文级验证工具：把 WAV 流进去、逐条打印带客户端时间戳的事件，
并把回复写回 WAV。

```sh
./tools/rtws --port 8002 --burst \
  --in models/adventists-ai/Sophon-Realtime-32B-Tag/voices/brand_zh.wav \
  --out /tmp/reply.wav --token <token>
```

## 管线

| 阶段 | 实现 | 实测 |
|---|---|---|
| 16 kHz PCM → 128 维 log-mel → 18 层音频塔 → projector | `rt_audio.c`，100 帧卷积块、104 帧注意力块 | 8.73 s 音频耗时 0.17 s（约 52 倍实时），产出 55 个向量 |
| 64 层 thinker，H=5120，FP8 E4M3 每个输出行一个 BF16 scale | `rt_thinker.c`，零拷贝 Metal row-scale GEMV/GEMM；CPU 回退数值一致 | 186 个 prompt 位置耗时 8.26 s（22.5 tok/s） |
| 解码，按句边界流式交给 talker | `rt_realtime.c`，句子一完整就送合成 | 每轮 18–40 个回复 token |
| talker（24 层 / 896 宽）→ flow → CausalHiFT | `rt_talker.c` / `rt_flow.c` / `rt_hift.c`，24 kHz 输出 | 每轮 4.1–10.4 s 语音 |
| WebSocket | `ws.c` + `rt_realtime.c`，每 200 ms 一段 PCM16 | 每轮 21–53 个 `response.audio.delta` |

权重全部就地读取：talker / flow / HiFT 的 `.pt` 张量经 `rt_torch.c` 里的小型
栈式解释器读取，thinker 分片以 mmap 映射，没有格式转换步骤。

## 事件

客户端 → 服务端：`session.update`、`input_audio_buffer.append`（base64 PCM16，
16 kHz）、`input_audio_buffer.commit`、`response.create`、
`conversation.item.create`，以及由服务端 VAD 驱动的插话路径。

服务端 → 客户端：`session.created`、`session.updated`、`conversation.created`、
`input_audio_buffer.speech_started` / `speech_stopped` / `committed`、
`response.created`、`response.output_item.added`、
`response.content_part.added`、`response.audio_transcript.delta`、
`response.audio.delta`、`response.audio.done`、`response.done`、`error`。

每个 `response.done` 都带 `cwick` 对象，给出全链路拆分（`ear_s`、
`prefill_s`、`decode_s`、`talker_s`、`turn_s`、`first_audio_s`、`audio_in_s`、
`audio_out_s`，音频/prompt/回复/语音 token 数和 `pcm_chunks`）。同样的数字由
stdout 的 `turn completed` 行带上请求编号打印一遍，日志与报文一致。

`/v1/metrics`（也可 `/metrics`）把同一轮的数据发布为
`cwick:realtime_sessions_total`、`cwick:realtime_turns_total`、
`cwick:realtime_cancelled_total`、
`cwick:realtime_stage_seconds_total{stage="ear|prefill|decode|talker|turn"}`、
`cwick:realtime_{audio_in,audio_out}_seconds`、
`cwick:realtime_pcm_samples_total` 和各 token 计数器，以及最近一轮的
`cwick:realtime_last_{turn,first_audio,audio_out}_seconds` 与
`cwick:realtime_last_rtf`（语音秒数 / 墙钟秒数）。阶段计数器是累计值，看采集
窗口的增量而不是绝对值；`decode` 与 `talker` 有重叠是设计使然。

## 实测全链路（M3 Max，12P+4E，40 核 GPU，`--backend metal`）

输入 `voices/brand_zh.wav`，8.730 s / 16 kHz → 55 个音频 token。第 2 轮复用
第 1 轮写下的 KV。

| | 第 A 次 | 第 1 轮 | 热第 2 轮 |
|---|---|---|---|
| prompt token（音频） | 186（55） | 186（55） | 63（55） |
| ear | 0.166 s | 0.172 s | 0.172 s |
| prefill | 8.257 s | 8.164 s | 2.827 s |
| decode（含合成） | 23.004 s | 9.402 s | 9.458 s |
| talker | 13.114 s | 4.902 s | 4.898 s |
| commit 到首音频 | 12.982 s | 12.960 s | 7.666 s |
| 整轮 | 32.162 s | 18.540 s | 13.216 s |
| 回复 / 语音 token | 40 / 261 | 18 / 102 | 18 / 102 |
| 语音时长 | 10.440 s | 4.080 s | 4.080 s |
| 音频 delta / PCM 块 | 53 / 12 | 21 / 5 | 21 / 5 |

decode 计时与合成重叠：到达句尾时同步调用 talker，因此减掉 `talker_s` 才是
纯解码（40 token 约 9.9 s，≈4 tok/s）。prefill 约 22 tok/s；热轮跳过静态系统
提示，省下 5.3 s。返回的音频已验证不是静音（A 次 RMS 0.087、峰值 0.84），
而不是「有事件就算成功」。

## TeachMe（应用 4）

`apps/teachme` 托管在 `/apps/teachme/`，与应用独立的 sqlite 共用存储。麦克风
按 16 kHz PCM16 采集，base64 `append` 上报，免提模式用服务端 VAD 断句；返回的
24 kHz 缓冲区排在同一条 `AudioContext` 时间轴上，界面显示每轮与累计耗时，
每轮回答都有重播按钮。浏览器要求用户手势后才允许播放，应用首次交互时会
resume 音频上下文。

## 两个值得先想到的故障模式

- **回答显示已完成却没有声音**：这是 prompt 问题，不是模型沉默。assistant
  回合必须在 `<|im_start|>assistant\n` 处结束，不能追加 think 块或多余换行；
  那里多出的文本会让首个 token 直接是结束符，于是 `response.done` 有转写、
  音频 chunk 数为零。见 `docs/DECISIONS.zh-CN.md` §82。
- **摘要为 cancelled 且 decode 为 0**：说明用户在 prefill 期间又开口了。
  40 秒音频的 ear 很快，但 prefill 要 8–12 s，第二段语音会在第一段回答前
  被提交。先区分日志里的 `turn cancelled` 与 `turn completed`，再查 talker。

## 断句：迟滞与短片段门限

输入 VAD 放在 `src/rt_vad.c`，这样同一段决策能被独立程序两次驱动
（`tools/rtvad.c`，`make rtvad`）。它在 10 ms 的 RMS 帧上工作，比单阈值多两条规则：

- **释放迟滞**。起点用客户端 `threshold`（0.5 对应 −40 dBFS），释放点在同一
  0..1 刻度上低 `0.15`（0.5 时 −44.5 dBFS）。两档之间的回落是保持区，不增加
  断句计时；词内一个轻读音节因此不会把一句话切成两句。
- **短片段门限**。一段声音必须连续高于起点 100 ms 才上报 speech。咔哒声或
  键盘敲击到不了这个门限，会被丢弃，采样也不会留在提交前缓冲里累积。

`CWICK_RT_VAD_LEGACY=1` 恢复原来的单阈值行为供 A/B；harness 在同一进程里
对同一信号跑两种规则。相同合成输入（threshold 0.5、prefix 300 ms、silence
200 ms；`make rtvad`）：

| 场景 | legacy | 增强 |
|------|--------|------|
| 40 ms 咔哒声、无语音 | 1 起点 / 1 终点（伪回合） | 0 起点 / 1 丢弃 |
| 0.5 s 语音、260 ms −38 dB 回落、0.5 s 语音 | 2 起点 / 2 终点（被切开） | 1 起点 / 1 终点 |
| 1 s 语音、400 ms 真停顿（超过 200 ms 计时） | 2 起点 | 2 起点 — 不变 |
| 1 s 语音、500 ms 静音尾 | 1 起点 / 1 终点 | 1 起点 / 1 终点 |

门限相对首个超阈样本多约一帧的决策延迟；由于预卷是从待定片段往回量的，
上报的 `audio_start_ms` 仍指向真实起点。五个场景都已写进 `make test`
（`ok: rt vad ...`）。

harness 注意：`tools/rtws --vad` 会播完一段成品 WAV 再补约 0.8 s 静音。
若录音里的停顿超过 `silence_duration_ms`，它被有意切成多个回合，后续音频
会被当成对第一段回答的插话。`CWICK_RT_VAD_LEGACY=1` 下行为完全一致；要测
成品音频的单轮性能请用 `--burst`（手动提交）。
