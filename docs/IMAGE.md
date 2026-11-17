# Image endpoints reference

<p align="center"><a href="IMAGE.md">English</a> · <a href="IMAGE.zh-CN.md">简体中文</a></p>

This build exposes two OpenAI-shaped image endpoints backed by `MiniMax-H3-Image`.

## Endpoints

- `POST /v1/images/generations` (and `/images/generations`) — JSON in, `{"created":…,"data":[{"b64_json":"…"}]}` out.
- `POST /v1/images/edits` (and `/images/edits`) — `multipart/form-data` with repeatable `image` parts; same response shape.

Both add a `cwick` timing object to the response: `total_ms`, `tokenize_ms`, `text_encode_ms`, `pack_ms`, `dit_ms`, `vae_ms`, `png_ms`, `text_tokens`, `image_tokens`, `canvas_w/h`, `output_w/h`.

Parameter constraints: `size` must be `WIDTHxHEIGHT`; `num_inference_steps` may be omitted or 6; `guidance_scale` may be omitted or 1; `negative_prompt` is rejected with 400; `response_format` must be `b64_json`.

## stdout progress log

Both endpoints print their render progress to stdout as it happens, so a
multi-minute request is never a silent block of time. Every line is stamped
with the wall-clock time, the elapsed time since the server started, and the
request id, and carries the `[image]` tag — the same conventions as the text
engine's logs. There is no flag to enable it; it is on at the `info` level.

A generation looks like this (a 256x256 request, with the debug area floor
lowered so the run is short):

```
cwick: 2026-11-18 00:23:02.389 [info  +19.54s] [req 1] [image] generate request: 256x256, seed 7, 0 reference(s)
cwick: 2026-11-18 00:23:02.389 [info  +19.54s] [req 1] [image] phase 1/6 tokenize: 7 tokens in 0.0 ms
cwick: 2026-11-18 00:24:01.038 [info  +78.19s] [req 1] [image] phase 2/6 text-encode: 58649.5 ms, 0 vision token(s)
cwick: 2026-11-18 00:24:01.038 [info  +78.19s] [req 1] [image] phase 3/6 pack: 64 image rows on 256x256 canvas (0 reference blocks), 0.1 ms
cwick: 2026-11-18 00:24:01.038 [info  +78.19s] [req 1] [image] phase 4/6 dit: 6 steps on 64 packed rows (seq 71), canvas 256x256
cwick: 2026-11-18 00:24:25.940 [info +103.10s] [req 1] [image]   dit step 1/6: 24902.0 ms (elapsed 24902.0 ms)
...
cwick: 2026-11-18 00:24:51.711 [info +128.87s] [req 1] [image]   dit step 6/6: 3786.8 ms (elapsed 50673.0 ms)
cwick: 2026-11-18 00:24:51.711 [info +128.87s] [req 1] [image] phase 4/6 dit done: 50673.0 ms total
cwick: 2026-11-18 00:24:59.657 [info +136.81s] [req 1] [image] phase 5/6 vae: 256x256 decoded in 7945.6 ms
cwick: 2026-11-18 00:24:59.673 [info +136.83s] [req 1] [image] phase 6/6 png: 256x256 output in 16.0 ms
cwick: 2026-11-18 00:24:59.673 [info +136.83s] [req 1] [image] generate complete: 256x256 -> 256x256, 117284 ms (tokenize 0, text-encode 58649, pack 0, dit 50673, vae 7946, png 16)
```

The six phases are tokenize, text-encode (the Qwen3-VL conditioner), pack
(latent noise and reference packing), dit (six sampling steps, one line each
plus a total), vae (decoder) and png. The DiT phase is normally the longest,
so its per-step lines are the heartbeat that shows the render is advancing
rather than hung. `edit` requests print the same sequence with the mode set to
`edit` and a `reference(s)` count on the arrival line; a failed render prints
`generation failed: …` with the same request id. These lines also land in the
log ring served by `GET /v1/logs` and `GET /v1/logs/stream`.

## Loading the model

Pass `--img DIR` to `cwick serve`, with `DIR` pointing at the checkpoint
(for example `./models/MiniMaxAI/MiniMax-H3-Image`). `CWICK_IMAGE_MODEL` sets
the same value for launchers that cannot pass flags; when both are present,
`--img` wins. Without either, both endpoints return 503 until a switch
discovers the pipeline in the model library.

One model runs at a time. Switching to the image pipeline releases the text
engine first, and switching back releases the image pipeline; the two weight
banks are never resident together. When `--img` is present at startup, the
text engine created to boot the server is released as soon as initialization
finishes, so only the image model stays loaded:

```bash
curl -s localhost:8000/v1/models/switch \
  -H 'Content-Type: application/json' -d '{"model":"MiniMax-H3-Image"}'
```

A switch is refused with 409 while a text generation or an image render is in
flight, so no request ever uses a pipeline that is being freed. Image requests
themselves return 503 during a switch.

`GET /v1/models` advertises `MiniMax-H3-Image` with
`cwick.kind:"image"` whenever a directory is configured;
`GET /v1/models/MiniMax-H3-Image` returns the same entry.
`GET /v1/config` reports `image_model.dir` and `image_model.loaded`, which the
built-in console's **设置** tab uses to show the current state and enable the
**加载 / 切换到图像模型** button. A switch returns `switched`,
`already_resident`, or an error with `detail` explaining why. Once the image
model is resident, text endpoints answer 503 until the operator switches back
to a text id; the same applies in reverse to the image endpoints.

The **模型** tab reads `/v1/hub/models`, which detects an image pipeline by its
`model_index.json` rather than by `config.json`. A finished pipeline is shown
as `complete`, `supported` and `kind:"image"`; its switch button posts
`served_id` (`MiniMax-H3-Image`) and hot-loads the pipeline, discovering the
directory from the library when the process was not started with `--img`.
An unfinished download keeps the `未下载完` / `Incomplete` state and cannot be
switched to.

`images/edits` runs the same pipeline and additionally reads the reference images (see *Editing: the reference path* below). The request must be multipart with at least one `image` part; an edit with no `image` is refused with 400 instead of being treated as a plain generation that quietly drops the references.

## Canvas: the model only works above about 4 MP

The checkpoint's released resolutions start at 4096×4096 and thin out from there. Below roughly 4 MP the sampled result is textured noise — and the released pipeline returns the *same* textured noise for the same request, so this is a property of the weights, not of this port.

`h3_canvas_size()` therefore scales a request up to `min_generation_area` (default 4194304, 2048²), keeping the aspect ratio, rounding both sides to a multiple of 32, and Lanczos-3 resizes the decoded image back to the requested size. `CWICK_H3_MIN_AREA` overrides the floor (1024…16777216) for experiments; lowering it is what produces the noise, so it is a debugging switch only.

A 512×512 request therefore costs exactly what a 2048×2048 request costs, and returns the same picture downsampled.

## Numerical alignment

Every stage is diffed against a reference implementation on a fixed seed, with `CWICK_H3_DUMP=<dir>` writing the intermediates and `CWICK_H3_LATENT=<file>` feeding a reference latent in to exercise the decoder on its own.

| Tensor | rel_rms vs reference | cos |
|---|---:|---:|
| conditioner output, 50 layers (`c_hs50`) | 1.14e-2 | 0.999985 |
| timestep embedding (`c_temb`) | 3.0e-6 | 1.000000 |
| token refiner output (`c_refiner`) | 1.26e-2 | 0.999921 |
| packed sequence (`c_packed`) | 4.4e-3 | 0.999991 |
| DiT block 0 / block 49 | 6.4e-3 / 4.2e-2 | 0.999987 / 0.999125 |
| final velocity (`c_vel`) | 2.9e-2 | 0.999591 |
| decoded latent (`c_latent`) | 5.4e-2 | 0.998514 |
| decoded image, 2048² (`c_rgb`) | **2.8e-3** | 0.99999857 |

The last row is the one that matters and it is the tightest, because the decoder runs on a reference latent: 79.2 dB PSNR against the reference image, and 99.96% of pixels are identical **after the 8-bit truncation**, which is the only thing the released image contains. The residual sits in a few hundred specular pixels above 0.98. Feeding the reference conditioner output in instead of this port's moves the latent by 5.45e-2 to 5.57e-2, which places the residual in the transformer's accumulation order rather than in conditioning.

The conditioner residual itself is floating-point noise, not a structural error: 99.4% of the error energy sits on the 1% of channels with the largest activations, but the error is uniform across tokens and channels (correlation −0.096 with row index, −0.114 with magnitude). Deliberate bf16 rounding of the residual stream or of individual conditioner operators made it *worse* (1.14e-2 → 2.4e-2 and 1.9e-1), so both were reverted.

## VAE decoder: spatial tiling is part of the recipe

`MiniMaxH3ImageViTDecoder2d` ships with `use_tiling = True`: 256 px tiles, a 64 px minimum overlap, linear cross-fade, and the released images are the blended-tile ones. Decoding a whole 128×128 latent in one pass puts the ViT far outside the 16×16 context it was released for and returns a per-cell modulation that is plainly visible as a grid.

`h3_vae_decode()` reproduces the reference geometry (`_split_tiles`: count tiles until their overlaps can span the length, then hand the slack out one cell at a time so every boundary stays cell-aligned), the `1 - i/e` / `i/e` weights, and the stitch order (cross-fade against the raw tile above, then against the raw tile to its left, then drop the far edges).

Two consequences beyond correctness:

- Work: attention is 16×16 per tile instead of 128×128, and the weights are resolved once per image and aliased out of the mapping instead of copied per tile — the tile loop used to re-copy 1.6 GiB of fp32 weights 121 times. VAE time at 2048² fell from 114028 ms to 26469 ms, and peak RSS from 112 GiB to 48 GiB, which is what stopped the process being killed on a 128 GiB machine.
- Fidelity: `c_rgb` rel_rms at 2048² went from 8.2e-2 (single-pass decode) to 2.8e-3.

One stitch detail cost a second pass: the cross-fade weight belongs to the offset *inside* the band, so the horizontal blend has to carry `i` down to the pixel. Blending the whole `e`-wide strip once per offset leaves only the last offset's weight applied on the whole strip, and since the recursion `cur <- a*(1-i/e) + cur*(i/e)` is idempotent once `cur == a`, the band silently became a hard cut on the left tile's side. The tell was that the *vertical* bands were fine and the horizontal ones were not: the vertical branch writes one row per offset, the horizontal branch wrote the same strip every time.

## Measured: full pipeline

Apple M3 Max (12P+4E, 40-core GPU, 128 GiB), `--backend cpu`, canvas 2048×2048, seed 42. The text encoder and the DiT both run the BF16 GEMM path on the CPU pool; the Metal path is measured separately under *Measured: the Metal path*.

| Phase | 512×512 request (canvas 2048²) |
|---|---:|
| Text encode (Qwen3-VL, 64 layers, 10 tokens) | 6800 ms |
| DiT (50 layers × 6 steps, 4096 image tokens) | 875968 ms |
| VAE (24-layer ViT decoder, 121 tiles) | 26653 ms |
| Total (seed 42) | 909693 ms |

The DiT dominates: 146 s per step, of which 64 s is GEMM and 64 s is attention.
The text-encoder figure is for a process whose page cache already holds the
62 GiB it reads; a first request after a restart measured 49611 ms for the same
3 tokens, so the number to plan with is closer to 50 s. Its 4096-token attention is already two `cblas_sgemm` calls per 256-row band (a hand-written loop measured 4.2 GFLOPS against 2.4 TFLOPS for the same work), so the remaining cost is the CPU's share of the ~62 GiB it reads six times.

## Editing: the reference path

One `images/edits` request feeds each reference image to **two** consumers, and the two size rules disagree:

| Consumer | Short-edge target | Note |
|---|---|---|
| VAE encoder | `[1024, 2048]` | kept as-is when the short edge is already inside the band, otherwise scaled to the nearer bound; both sides rounded to 32 |
| vision tower | `1024` fixed | a separate copy, scaled from the **original** rather than from the VAE copy |

A reference is never scaled to the generated size. When the short edge is inside the band and a side is not a multiple of 32, this port rounds it to the nearest 32: the released pipeline raises there, and a camera image hits that case almost always, so the cost of rounding is at most 16 px on that side.

How the four stages relate to the forward pass:

1. **VAE encode** → the reference latent (`h3_vae_encode()`). Tiled the same way (256 px tiles / 64 px overlap, stitched in latent space), because tiling changes the output here too: the reference reports the `moments` of a single tile differing from a 5×5 grid by 1.20e-1.
2. **Vision tower** → 1024 merged tokens per reference (a 1024² reference), plus three deepstack features.
3. **Conditioner**: each reference is preceded by a `<Picture i>: ` label (**no angle brackets**, unlike the `<Picture i>` the prompt may refer to) and by `<|vision_start|>` + `<|image_pad|>×N` + `<|vision_end|>`, then the prompt text follows verbatim. There are two different tag vectors: the conditioner's own `mm_token_type_ids` is 1 only on `<|image_pad|>` (it decides how the 3-axis mrope is segmented), while the transformer's token tag covers the **whole** vision block including both delimiters (it selects the AdaLN modality row).
4. **DiT**: the packed sequence becomes `[text | reference | generated]`, each reference takes one rope time slot (the first at `t = num_text_tokens`), the reference rows are noised to `t = 0.999` and held there for the whole loop, AdaLN consumes two timestep tables (`unique(sorted=True)` is what makes the generated rows index 0 and the reference rows index 1), and the scheduler updates only the generated rows.

### Numerical alignment, edit path

Diffed against the reference implementation tensor by tensor (1024² reference, 1036 text rows):

| Tensor | rel_rms |
|---|---:|
| VAE encoder moments (`c_enc_mom`) | 1.75e-6 |
| reference latent (`c_enc_latent`) | 2.71e-5 (98.4% bit-identical) |
| patch embed / `x0` / block 0 | 3.8e-7 / 3.5e-7 / 2.6e-6 |
| vision tower last layer / merger / three deepstacks | 8.5e-5 / 2.8e-5 / 7e-6…2.6e-5 |
| conditioner token layout (ids / types / tags) | zero mismatches |
| 3-axis mrope positions | zero mismatches |
| `hidden_states[1…3]` (the three deepstack injection points) | 2.6e-5 / 2.1e-5 / 1.8e-5 |
| conditioner output, 50 layers (`c_hs50`) | 4.5e-5 |
| reference rows, packed and noised | 4.4e-8 |
| step-0 velocity (`c_vel`, 1088 image rows) | 6.5e-3 |

The last two rows separate the two kinds of error: packing the reference rows and mixing them at `t=0.999` is exact (4.4e-8), so the residual is inside the transformer and of the same magnitude as the text-to-image path.

## Measured: edit

Same M3 Max, `--backend cpu`, 1024×1024 reference, 256×256 canvas (chosen only to hold the DiT at 64 generated rows, `CWICK_H3_MIN_AREA=1024`). The rows below are the `cwick` timings of one real HTTP request (`POST /v1/images/edits`, one reference):

| Phase | 256×256 request with 1 reference |
|---|---:|
| Reference handling (PNG decode, both resizes, VAE encode, vision tower) `tokenize_ms` | 12402 ms |
| Conditioner (1044 tokens) `text_encode_ms` | 86224 ms |
| Packed sequence build `pack_ms` | 15914 ms |
| DiT (50 layers × 6 steps, 2124 rows) | 398415 ms |
| VAE decode | 2469 ms |
| Total | 515430 ms |

Rows went from 4096 to 2124 while the per-step time fell from 146 s to 63 s, so the cost tracks the row count; a 2048² canvas with a 1024² reference is about 17408 rows, which puts a step near 500 s. The earlier CLI run with `CWICK_H3_REF_LATENT` (which skips the reference encoder) totalled 478 s, so the VAE encode and the vision tower together are about 28 s.

That 256² output is textured noise with visible blocky structure: `CWICK_H3_MIN_AREA=1024` put the canvas below 4 MP, which is the behaviour described under *Canvas*, not a stitching or layout error.

## Measured: the Metal path

`--backend metal` gives the transformer's projections and its attention to the
GPU (`CWICK_H3_GPU=0` keeps them on the CPU). The projections read the
checkpoint's BF16 bytes where they lie, so no weight is dequantised on the way
in; the attention is a streaming kernel that never materialises a score matrix.

Same machine, same request (`256×256` → 2048² canvas, 7 text + 4096 image rows,
6 steps), one row per sampling step:

| Phase, per step | `--backend cpu` | `--backend metal` |
|---|---:|---:|
| Projections (66.3 GB of weights, 1346.6 GMAC) | 63.7 s | 54.3 s |
| Attention | 64.5 s | 9.0 s |
| SwiGLU | 3.9 s | 3.1 s |
| AdaLN modulation | 0.25 s | 0.5 s |
| Remainder (norms, rope, residual adds, output heads) | 4.2 s | 7.2 s |
| **Step** | **136.6 s** | **74.3 s** |

| Whole request | cpu | metal |
|---|---:|---:|
| Text encoder | 46.1 s | 63.5 s |
| DiT, 6 steps | 874.3 s | 445.7 s |
| VAE decode | 28.3 s | 26.1 s |
| **Total** | **949.0 s** | **535.6 s** |

Attention is the term that matters. It is quadratic in the row count, so at the
17408 rows of a 2048² canvas carrying three references it decides whether a
request is minutes or hours; the 7× at 4096 rows is the floor, not the ceiling.
The projection win is smaller on purpose: the CPU path is already at 1.24 TFLOPS
at this size (a BF16 tile dequantised into an f32 arena and handed to
`cblas_sgemm`), and the kernel measures 1.51 TFLOPS. The arithmetic was not what
was missing there.

## Measured: one request with three references

`POST /v1/images/edits`, three reference images, `size=1536x2048`, seed 7, six
steps, same machine and `--backend metal`, measured through the HTTP path. The
three references are 1600×2016, 1376×2048 and 1024×1376 after the short-edge
rule, which is 3150 + 2752 + 1376 = 7278 reference rows in front of the 4070
generated rows — a packed sequence of **11348 rows**, against 4070 for the same
canvas without references.

| Phase | ms | Note |
|---|---:|---|
| Tokenize | 57 492 | prompt + vision placeholders, 4319 tokens |
| Text encoder | 222 850 | the conditioner over three references |
| Pack (VAE encode of the references) | 91 169 | three references, tiled |
| DiT, 6 steps | 2 022 601 | 337 s per step |
| VAE decode | 21 781 | output tiles only |
| PNG | 140 | |
| **Total** | **2 416 421** | 40.3 min |

Row count explains almost all of the gap against a reference-free request:
2.79× the rows means 2.79× the weight traffic and 7.8× the attention, and the
step went 74.3 → 337.0 s. The reference rows are also why the conditioner is no
longer noise: 222.9 s + 91.2 s against 63.5 s total for a text-only prompt.

Reproducibility: the same request run twice (`CWICK_H3_GPU` default) measured
2 364 432 ms and 2 416 421 ms, 2.2% apart, with `dit` 1 995 410 / 2 022 601 ms.
The second run was made under heavy VM pressure (45 GiB of swap in use, ~154 MiB/s
being written out), and it cost 2.2% — the process working set fits, so this is
not a memory-bound stage.

The three references each land where they should. The face reference fixes the
person (hair, features), the outfit reference transfers item by item (jacket,
dress, tights, socks, boots, bag), and the pose reference fixes the camera
(front-on, arms down, one hand forward). The output is 1536×2048, full body
inside the frame, uniform grey studio background, soft shadow, no text and no
watermark.

The two paths are not bit-identical. Per block the difference starts at 1.2e-3 —
BF16 rounding, because the GPU stages activations into BF16 tiles while the CPU
keeps them f32 — and reaches 8.6e-2 in the decoded RGB over 50 blocks × 6 steps.
The scene, the composition and the lighting match; fine detail does not. That is
the same tolerance the checkpoint's own pipeline runs at end to end.

### Residency: `CWICK_H3_WIRE=1`

The two checkpoint trees are 62 GiB each. Wiring both would ask for 124 GiB on a
128 GiB machine, so the switch wires **only the transformer**: it is the tree
re-read in full on every sampling step, while the text encoder is read once per
request and is deliberately left reclaimable. With the switch on, model open
reports `61.74 GiB faulted in and wired` and takes 26.7 s (2.5 GB/s) instead of
3.5 s. That is a one-time cost per process, not per request.

Same machine, same request as the top row of the previous table (1536×2048,
seed 7, `--backend metal`, 15 text + 4070 image rows, seq 4085), A first and B
second so the unwired run does not inherit a warm-up from the wired one:

| | `CWICK_H3_WIRE=0` | `CWICK_H3_WIRE=1` |
|---|---:|---:|
| Model open | 3.5 s | 26.9 s |
| Projections, mean of 6 steps | 52.57 s | **44.93 s** |
| Step, mean | 69.94 s | **60.37 s** |
| Step, range across 6 steps | 62.7 … 79.9 s | **60.2 … 60.7 s** |
| Step, standard deviation | 6.29 s | **0.17 s** |
| DiT, 6 steps | 419.7 s | **362.2 s** |
| Text encode / VAE | 63.4 / 23.2 s | 63.5 / 23.3 s |
| Request total | 506.8 s | **449.6 s** |
| CLI wall including model open | 511 s | **478 s** |

The gain is 13.7% on DiT and 11.3% on the request; for a one-shot CLI process
the 23.5 s wiring cost leaves 33 s net. The more useful signal is the variance:
unwired projections wander over a 15 s span from step to step because weight
pages are being reclaimed and refaulted in the middle of the loop, while wired
projections sit at 44.6–45.1 s. This was a residency problem, not an arithmetic
one. The two PNGs are byte-identical (same SHA-256) — wiring changes where a
page lives, not what it contains.

## Verification

The output of that run is a clean 512×512 photorealistic image with no grid, no noise and no seam at the 121-tile boundaries.

`make test` covers the PNG encode/decode round trip (dimensions and at most one byte of error). The numerical alignment above is checked against a reference implementation with the environment switches described in *Numerical alignment*, not from the test suite. The HTTP path of `images/edits` was exercised on a live server as well: no `image` and a corrupt `image` both return 400 with a specific reason, and a normal request returns 200 with `b64_json`.
