[简体中文](REALTIME.zh-CN.md)

# Realtime speech: Sophon-Realtime-32B-Tag

`--rt <checkpoint-dir>` registers the full-duplex speech pipeline. It is a
**single resident model**, like every other checkpoint in cwick: while it is
loaded the text endpoints answer 503, and switching to a text id unloads it.
The WebSocket endpoint is `/v1/realtime?model=Sophon-Realtime-32B-Tag` and uses
the same password session / Bearer token as the HTTP API.

```sh
./cwick serve --root ./models/xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic \
  --port 8002 --web --backend metal \
  --rt ./models/adventists-ai/Sophon-Realtime-32B-Tag \
  --password cwick-local --verbose --trace
# then, from another terminal:
curl --noproxy '*' -X POST http://127.0.0.1:8002/v1/auth/login \
  -H 'Content-Type: application/json' -d '{"password":"cwick-local"}'
curl --noproxy '*' -X POST http://127.0.0.1:8002/v1/models/switch \
  -H 'Authorization: Bearer <token>' -H 'Content-Type: application/json' \
  -d '{"model":"Sophon-Realtime-32B-Tag"}'
```

`tools/rtws` is the wire-level harness: it streams a WAV in, prints every event
with a client timestamp, and writes the spoken reply to a WAV:

```sh
./tools/rtws --port 8002 --burst \
  --in models/adventists-ai/Sophon-Realtime-32B-Tag/voices/brand_zh.wav \
  --out /tmp/reply.wav --token <token>
```

## Pipeline

| stage | implementation | measured |
|---|---|---|
| 16 kHz PCM → 128-bin log-mel → 18-layer audio tower → projector | `rt_audio.c`, 100-frame conv chunks, 104-frame attention blocks | 8.73 s audio in 0.17 s (≈52× realtime), 55 embeddings |
| 64-layer thinker, H=5120, FP8 E4M3 with one BF16 scale per output row | `rt_thinker.c`, zero-copy Metal row-scale GEMV/GEMM, CPU fallback with the same numbers | 186 prompt positions in 8.26 s (22.5 tok/s) |
| decode, sentence-boundary streaming | `rt_realtime.c` sends each finished sentence to the talker as soon as it is complete | 18–40 reply tokens per turn |
| talker (24 layers, 896 wide) → flow → CausalHiFT | `rt_talker.c` / `rt_flow.c` / `rt_hift.c`, 24 kHz output | 4.1–10.4 s of speech per turn |
| WebSocket | `ws.c` + `rt_realtime.c`, 200 ms PCM16 deltas | 21–53 `response.audio.delta` events per turn |

Weights are read from the checkpoint in place: the `.pt` tensors used by the
talker, flow and HiFT go through the small stack-machine reader in `rt_torch.c`,
and the thinker's shards are memory-mapped. No format conversion step exists.

## Events

Client → server: `session.update`, `input_audio_buffer.append` (base64 PCM16,
16 kHz), `input_audio_buffer.commit`, `response.create`,
`conversation.item.create`, and the barge-in path driven by server VAD.

Server → client: `session.created`, `session.updated`, `conversation.created`,
`input_audio_buffer.speech_started` / `speech_stopped` / `committed`,
`response.created`, `response.output_item.added`,
`response.content_part.added`, `response.audio_transcript.delta`,
`response.audio.delta`, `response.audio.done`, `response.done`, `error`.

Every `response.done` carries a `cwick` object with the full-chain breakdown
(`ear_s`, `prefill_s`, `decode_s`, `talker_s`, `turn_s`, `first_audio_s`,
`audio_in_s`, `audio_out_s`, audio/prompt/reply/speech token counts and
`pcm_chunks`). The same numbers are printed to stdout on the `turn completed`
line with the shared request id, so the logs and the wire agree.

`/v1/metrics` (also `/metrics`) publishes the same turn data as
`cwick:realtime_sessions_total`, `cwick:realtime_turns_total`,
`cwick:realtime_cancelled_total`,
`cwick:realtime_stage_seconds_total{stage="ear|prefill|decode|talker|turn"}`,
`cwick:realtime_{audio_in,audio_out}_seconds`,
`cwick:realtime_pcm_samples_total`, token counters, and last-turn gauges
`cwick:realtime_last_{turn,first_audio,audio_out}_seconds` and
`cwick:realtime_last_rtf` (spoken seconds / wall seconds). The stage counters
are cumulative, so sample the delta over an interval rather than the absolute
value; `decode` overlaps `talker` by design.

## Measured full chain (M3 Max, 12P+4E, 40-core GPU, `--backend metal`)

Input: `voices/brand_zh.wav`, 8.730 s, 16 kHz → 55 audio tokens. Warm turn 2
reuses the KV written by turn 1.

| | run A | turn 1 | warm turn 2 |
|---|---|---|---|
| prompt tokens (audio) | 186 (55) | 186 (55) | 63 (55) |
| ear | 0.166 s | 0.172 s | 0.172 s |
| prefill | 8.257 s | 8.164 s | 2.827 s |
| decode (includes synthesis) | 23.004 s | 9.402 s | 9.458 s |
| talker | 13.114 s | 4.902 s | 4.898 s |
| first audio after commit | 12.982 s | 12.960 s | 7.666 s |
| whole turn | 32.162 s | 18.540 s | 13.216 s |
| reply / speech tokens | 40 / 261 | 18 / 102 | 18 / 102 |
| spoken audio | 10.440 s | 4.080 s | 4.080 s |
| audio deltas / PCM chunks | 53 / 12 | 21 / 5 | 21 / 5 |

The decode timing overlaps talker synthesis: the talker is called synchronously
when a sentence boundary is reached, so subtracting `talker_s` gives roughly
4 tok/s of pure decoding (40 tokens in ≈9.9 s). Prefill is about 22 tok/s;
a warm turn skips the static system prompt and saves 5.3 s. The returned audio
is verified non-silent (RMS 0.087, peak 0.84 on run A), not merely an event
with a base64 field.

## TeachMe (app 4)

`apps/teachme` is hosted under `/apps/teachme/` and shares the app sqlite.
It captures the microphone as 16 kHz PCM16, sends base64 `append` events, uses
server VAD for hands-free turns, schedules the returned 24 kHz buffers on one
`AudioContext` timeline, shows per-turn and aggregate timings, and offers a
replay button for every answer. A user gesture is required before the browser
allows audio playback; the app resumes the context on the first interaction.

## Two failure modes worth knowing

- **An empty but "completed" answer with no audio** is a prompt problem, not a
  quiet model. The assistant turn must end at `<|im_start|>assistant\n` with no
  added think block or stray newlines: extra text there makes the first decoded
  token end-of-turn, so `response.done` arrives with a transcript and zero
  audio chunks. See `docs/DECISIONS.md` §82.
- **A response.cancelled summary with zero decoded tokens** means the user
  spoke during the prefill. On a 40 s clip the ear is fast but prefill takes
  8–12 s, so the VAD will commit the next utterance before the first answer
  starts. Watch `turn cancelled` versus `turn completed` in the log before
  blaming the talker.

## Turn detection: hysteresis and a fragment floor

The input VAD lives in `src/rt_vad.c` so the decision can be driven by a
standalone harness (`tools/rtvad.c`, `make rtvad`). It works on 10 ms RMS
frames and keeps two rules that a bare energy threshold does not have:

- **Release hysteresis.** Onset uses the client's `threshold` (0.5 maps to
  −40 dBFS); the release level sits `0.15` below it on the same 0..1 scale
  (−44.5 dBFS at 0.5). A dip between the two levels is a hold region: it does
  not advance the end-of-turn timer. This stops a quiet syllable inside a word
  from splitting one utterance into two.
- **Short-fragment floor.** A burst must stay above onset for 100 ms before it
  is reported as speech. A click or a keyboard tap never reaches that floor,
  is dropped, and its samples do not accumulate in the pre-commit buffer.

`CWICK_RT_VAD_LEGACY=1` restores the original single-threshold behaviour for
an A/B; the harness exercises both rules on the same signal in one process.
On identical synthetic input (threshold 0.5, prefix 300 ms, silence 200 ms;
`make rtvad`):

| case | legacy | enhanced |
|------|--------|----------|
| 40 ms click, no speech | 1 start / 1 end — a spurious turn | 0 starts / 1 drop |
| 0.5 s speech, 260 ms dip at −38 dB, 0.5 s speech | 2 starts / 2 ends — split | 1 start / 1 end |
| 1 s speech, 400 ms real pause (over the 200 ms timer) | 2 starts | 2 starts — unchanged |
| 1 s speech, 500 ms quiet tail | 1 start / 1 end | 1 start / 1 end |

The floor adds about one frame of decision latency relative to the first
above-threshold sample; the reported `audio_start_ms` still points at the true
onset because the pre-roll is measured back from the pending burst. All five
cases are asserted in `make test` (`ok: rt vad ...`).

Note on the harness: `tools/rtws --vad` replays a finished WAV and then streams
~0.8 s of silence. A recording with pauses longer than `silence_duration_ms`
is deliberately split into several turns, and the remaining audio is treated
as a barge-in against the answer to the first one. This happens identically
with `CWICK_RT_VAD_LEGACY=1`; use `--burst` (manual commit) for a single-turn
performance measurement of a pre-recorded clip.
