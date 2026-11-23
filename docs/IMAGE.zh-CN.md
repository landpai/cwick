# 图像接口参考

<p align="center"><a href="IMAGE.md">English</a> · <a href="IMAGE.zh-CN.md">简体中文</a></p>

本构建提供两个 OpenAI 风格的图像 endpoint 与一个进度 endpoint，模型为 `MiniMax-H3-Image`。

## Endpoint

- `POST /v1/images/generations`（以及 `/images/generations`）—— JSON 请求，返回 `{"created":…,"data":[{"b64_json":"…"}]}`。
- `POST /v1/images/edits`（以及 `/images/edits`）—— `multipart/form-data`，`image` 字段可重复；返回同一个响应形态。
- `GET /v1/images/progress`（以及 `/images/progress`）—— 当前渲染的实时阶段。

两个 POST endpoint 都会按需装载图像模型：若当前没有常驻管线，首个请求会关闭交换门、
释放当前的文本/实时模型并映射 `MiniMax-H3-Image`，与 `POST /v1/models/switch` 完全一致，
无需手动切换。若此时并发进行模型切换，返回 503 而不是双重装载。

`GET /v1/images/progress` 在生成准入之前应答，因此轮询不会排在被观察的那个长达数分钟的
任务后面。返回
`{"object":"image_progress","phase":P,"phase_name":"…","step":N,"steps":M,"idle":bool,"age_ms":…}`。
阶段为 0 idle、1 tokenize、2 text_encode、3 pack、4 dit（`step`/`steps` 是 DiT 计数）、
5 vae、6 png、7 done；静止或快照超过 10 秒时 `idle` 为 true。

两个 POST endpoint 都在响应中额外返回一个 `cwick` 计时对象，字段为 `total_ms`、`tokenize_ms`、`text_encode_ms`、`pack_ms`、`dit_ms`、`vae_ms`、`png_ms`、`text_tokens`、`image_tokens`、`canvas_w/h`、`output_w/h`。

参数约束：`size` 必须是 `WIDTHxHEIGHT`；`num_inference_steps` 只能省略或 6；`guidance_scale` 只能省略或 1；`negative_prompt` 不支持，会被 400 拒绝；`response_format` 只接受 `b64_json`。

## stdout 阶段进度日志

两个 endpoint 都会在渲染过程中把进度打印到 stdout，因此一个耗时数分钟的请求不会是一段沉默。每行都带系统时间、服务启动后的经过时间、请求编号，并带 `[image]` 标签 —— 与文本引擎日志的约定一致。无需开关，`info` 级别默认开启。

下面是一次生成（256x256 请求，调低了调试面积下限以缩短运行）：

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

六个阶段依次为 tokenize、text-encode（Qwen3-VL 条件编码器）、pack（latent 噪声与参考图打包）、dit（六次采样，每步一行并附合计）、vae（解码器）和 png。DiT 通常是耗时最长的阶段，因此它的逐步日志是显示渲染仍在推进、而不是卡住的心跳。`edit` 请求打印同样的序列，模式显示为 `edit`，到达行带 `reference(s)` 计数；渲染失败则以同一请求编号打印 `generation failed: …`。这些行同样进入 `GET /v1/logs` 与 `GET /v1/logs/stream` 的日志环。

## 装载模型

给 `cwick serve` 传 `--img DIR`，`DIR` 指向 checkpoint 目录（例如 `./models/MiniMaxAI/MiniMax-H3-Image`）。不方便传参数的启动器可以设置 `CWICK_IMAGE_MODEL`，两者都给出时 `--img` 优先。两者都没有时，两个 endpoint 返回 503，直到一次切换从模型库里找到该管线。

同一时刻只运行一个模型。切换到图像管线会先释放文本引擎，切回时再释放图像管线；两套权重不会同时驻留。启动时若带 `--img`，用于引导服务的文本引擎会在初始化完成后立即释放，只留下图像模型：

```bash
curl -s localhost:8000/v1/models/switch \
  -H 'Content-Type: application/json' -d '{"model":"MiniMax-H3-Image"}'
```

文本生成或图像渲染进行中时，切换返回 409，因此不会有请求用到正在被释放的管线；切换期间图像请求本身返回 503。

只要配置了目录，`GET /v1/models` 就会公布带 `cwick.kind:"image"` 的
`MiniMax-H3-Image`；`GET /v1/models/MiniMax-H3-Image` 返回同一条目。
`GET /v1/config` 报告 `image_model.dir` 与 `image_model.loaded`，内置控制台的
**设置**标签页用它显示当前状态，并决定是否启用**加载 / 切换到图像模型**按钮。
切换返回 `switched`、`already_resident`，或带 `detail` 的错误说明。图像模型驻留后，文本接口返回 503，直到操作员切回某个文本 id；反过来图像接口也一样。

**模型**tab 读的是 `/v1/hub/models`，它用 `model_index.json` 而不是 `config.json` 识别图像管线。已下载完成的管线会标为 `complete`、`supported`、`kind:"image"`；切换按钮提交 `served_id`（`MiniMax-H3-Image`）热装载管线，即使进程启动时没有 `--img`，也会从模型库中找到目录。未下载完的仍显示「未下载完」，且不允许切换。

`images/edits` 走与 `generations` 相同的管线，另外读入参考图（见下文「编辑：参考图路径」）。请求必须是 multipart，至少含一个 `image` 字段；没有 `image` 的编辑请求返回 400 而不是当成普通生成静默丢掉参考图。

## 画布：模型只在约 4 MP 以上有效

该 checkpoint 公布的可用分辨率从 4096×4096 起。低于约 4 MP 时采样结果是带纹理的噪声——参考实现在同一个请求上返回**同样的**噪声，所以这是权重的性质，不是本移植的缺陷。

因此 `h3_canvas_size()` 会把请求按同宽高比放大到 `min_generation_area`（默认 4194304，即 2048²），两轴取整到 32 的倍数，解码后再用 Lanczos-3 缩回请求尺寸。`CWICK_H3_MIN_AREA` 可覆盖该下限（1024…16777216）供实验用；调低它正是产生噪声的入口，因此只是调试开关。

也就是说 512×512 请求的开销与 2048×2048 完全相同，返回的是同一张图缩小后的结果。

## 数值对齐

每一阶段都在固定 seed 下与参考实现逐张量对比：`CWICK_H3_DUMP=<dir>` 导出中间结果，`CWICK_H3_LATENT=<file>` 注入参考 latent 以便单独检验解码器。

| 张量 | rel_rms | cos |
|---|---:|---:|
| 条件编码器输出 50 层（`c_hs50`） | 1.14e-2 | 0.999985 |
| 时间步嵌入（`c_temb`） | 3.0e-6 | 1.000000 |
| token refiner 输出（`c_refiner`） | 1.26e-2 | 0.999921 |
| packed 序列（`c_packed`） | 4.4e-3 | 0.999991 |
| DiT 第 0 层 / 第 49 层 | 6.4e-3 / 4.2e-2 | 0.999987 / 0.999125 |
| 最终速度场（`c_vel`） | 2.9e-2 | 0.999591 |
| 解码后 latent（`c_latent`） | 5.4e-2 | 0.998514 |
| 解码图像 2048²（`c_rgb`） | **2.8e-3** | 0.99999857 |

最后一行才是决定性的，而且它最紧——因为解码器吃的是参考 latent：对参考图 79.2 dB PSNR，且**按 8 位截断后 99.96% 的像素完全一致**，而公布出来的图像里只有这 8 位。残差集中在几百个幅值 0.98 以上的高光像素上。把参考条件编码器的输出注入进来替代本移植的结果，latent 误差只从 5.45e-2 动到 5.57e-2，说明残差在 transformer 的累加顺序里而不在条件化里。

条件编码器本身的残差是浮点噪声而非结构性错误：99.4% 的误差能量落在激活幅值最大的 1% 通道上，但误差在 token 与通道两维上均匀（与行号相关 −0.096，与幅值相关 −0.114）。主动把残差流或条件编码器的逐算子降到 bf16 反而更差（1.14e-2 → 2.4e-2 与 1.9e-1），两者均已回滚。

## VAE 解码器：空间分块是配方的一部分

`MiniMaxH3ImageViTDecoder2d` 的 `use_tiling = True` 是发布默认值：256 px 分块、最小 64 px 重叠、线性 cross-fade，且**公布出来的图像就是分块混合的那一版**。一次性解码整张 128×128 latent，会把 ViT 推出它被训练时的 16×16 上下文，产生肉眼可见的逐 cell 网格。

`h3_vae_decode()` 复现了参考实现的几何（`_split_tiles`：先数出块数直到重叠能覆盖全长，再把余量按 cell 轮流摊到各边界上，保证每个边界都落在 cell 边界）、`1 - i/e` / `i/e` 的权重，以及 stitch 顺序（先与上方 raw 块垂直混合，再与左侧 raw 块水平混合，最后裁掉远端边缘）。

除正确性外还有两个后果：

- 工作量：块内 attention 是 16×16 而不是 128×128；同时权重每张图只解析一次、直接别名到映射上，不再逐块拷贝——分块循环原先会把 1.6 GiB 的 fp32 权重重复拷贝 121 次。2048² 下 VAE 从 114028 ms 降到 26469 ms，峰值 RSS 从 112 GiB 降到 48 GiB，后者正是进程不再在 128 GiB 机器上被杀掉的原因。
- 保真度：2048² 下 `c_rgb` 的 rel_rms 从 8.2e-2（单块解码）降到 2.8e-3。

还多花了一轮才修对的一处 stitch 细节：cross-fade 的权重属于**带内偏移**，所以水平方向必须把 `i` 落到像素上。若每个偏移都把整条 `e` 宽的带重新混一遍，最后只有最后一个偏移的权重留下；而递推 `cur <- a*(1-i/e) + cur*(i/e)` 在 `cur == a` 之后是幂等的，于是那条带静默变成了「左边一块直接覆盖」。线索是**垂直带全对、水平带全错**：垂直分支每个偏移写一行，水平分支每次都写同一条带。

## 实测：全链路

Apple M3 Max（12P+4E，40 核 GPU，128 GiB），`--backend cpu`，画布 2048×2048，seed 42。文本编码与 DiT 都跑 CPU 池上的 BF16 GEMM；Metal 路径单独在「实测：Metal 路径」一节里给。

| 阶段 | 512×512 请求（画布 2048²） |
|---|---:|
| 文本编码（Qwen3-VL 64 层，10 个 token） | 6800 ms |
| DiT（50 层 × 6 步，4096 个图像 token） | 875968 ms |
| VAE（24 层 ViT decoder，121 块） | 26653 ms |
| 总计（seed 42） | 909693 ms |

DiT 是瓶颈：每步 146 s，其中 GEMM 64 s、attention 64 s。文本编码那一行是页缓存里
已经有它要读的 62 GiB 时的数字；重启后第一个请求同样 3 个 token 实测 49611 ms，
所以按 50 s 估算更实在。它 4096 token 的 attention 已经是每 256 行一次 `cblas_sgemm`（手写循环实测 4.2 GFLOPS，同样的工作量用 sgemm 是 2.4 TFLOPS），剩下的开销就是 CPU 分摊那约 62 GiB、要读完六遍的权重。

## 编辑：参考图路径

一次 `images/edits` 请求把每张参考图**同时**送进两个消费者，两处尺寸规则不同：

| 消费者 | 短边目标 | 说明 |
|---|---|---|
| VAE 编码器 | `[1024, 2048]` | 短边已在区间内则保留原尺寸，否则等比缩到较近的界；两边取整到 32 |
| vision tower | 固定 `1024` | 单独一份副本，从**原图**而不是从 VAE 副本再缩一次 |

参考图从不缩放到生成图的尺寸。区间内保留原尺寸时，若某边不是 32 的倍数，本实现把它按最近的 32 取整：发布实现会在这里直接抛错，而手机照片几乎必然踩到，取整的代价最多是该边 16 像素。

四段与前向的关系：

1. **VAE 编码** → 参考 latent（`h3_vae_encode()`）。同样分块（256 px 块 / 64 px，在 latent 空间 stitch），因为分块在这里也改变输出：参考实现单块与 5×5 分块的 `moments` 相差 1.20e-1。
2. **vision tower** → 每张参考一组 1024 个 merge token（1024² 参考），带三个 deepstack 特征。
3. **条件编码器**：每张参考前插 `<Picture i>: ` 标签（**无尖括号**，与 prompt 里引用的 `<Picture i>` 不同）和 `<|vision_start|>` + `<|image_pad|>×N` + `<|vision_end|>`，然后接 prompt 原文。tag 向量有两个且不同：条件编码器自己的 `mm_token_type_ids` 只在 `<|image_pad|>` 上为 1（决定 3 轴 mrope 的分段），transformer 的 token tag 覆盖**整个** vision block 含两个分隔符（决定 AdaLN 的模态行）。
4. **DiT**：packed 序列变成 `[text | reference | generated]`，每个参考占一个 rope 时间槽（第一个在 `t = num_text_tokens`），参考行被加噪到 `t = 0.999` 并在每一步保持不变，AdaLN 用两张时间步表（`unique(sorted=True)` 保证生成行索引 0、参考行索引 1），调度器只更新生成行。

### 编辑阶段的数值对齐

与参考实现逐张量对比（1024² 参考 + 1036 行文本）：

| 张量 | rel_rms |
|---|---:|
| VAE 编码器 moments（`c_enc_mom`） | 1.75e-6 |
| 参考 latent（`c_enc_latent`） | 2.71e-5（98.4% 逐位相同） |
| patch embed / `x0` / block 0 | 3.8e-7 / 3.5e-7 / 2.6e-6 |
| vision tower 末层 / merger / 三个 deepstack | 8.5e-5 / 2.8e-5 / 7e-6…2.6e-5 |
| 条件编码器 token 布局（ids / types / tags） | 零 mismatch |
| 3 轴 mrope 位置 | 零 mismatch |
| `hidden_states[1…3]`（三个 deepstack 注入点） | 2.6e-5 / 2.1e-5 / 1.8e-5 |
| 条件编码器输出 50 层（`c_hs50`） | 4.5e-5 |
| 参考行 packed + 加噪 | 4.4e-8 |
| 第 0 步速度场（`c_vel`，1088 图像行） | 6.5e-3 |

最后两行把两种错误分开了：参考行的 packing 与 `t=0.999` 混合是精确的（4.4e-8），残差只出现在 transformer 内部，量级与文生图一致。

## 实测：编辑

同一个 M3 Max，`--backend cpu`，参考图 1024×1024，画布 256×256（仅为把 DiT 行数压到 64 行而设，`CWICK_H3_MIN_AREA=1024`）。下面是一次真实 HTTP 请求（`POST /v1/images/edits`，单张参考）的 `cwick` 计时：

| 阶段 | 请求 256×256 + 1 张参考 |
|---|---:|
| 参考处理（PNG 解码、两套缩放、VAE 编码、vision tower）`tokenize_ms` | 12402 ms |
| 条件编码（1044 token）`text_encode_ms` | 86224 ms |
| packed 序列构造 `pack_ms` | 15914 ms |
| DiT（50 层 × 6 步，2124 行） | 398415 ms |
| VAE 解码 | 2469 ms |
| 总计 | 515430 ms |

DiT 行数从 4096 涨到 2124 但步内时间从 146 s/步降到 63 s/步，说明成本主要随行数走；2048² 画布加 1024² 参考约 17408 行，按此估算每步约 500 s。上一次 CLI 实测（用 `CWICK_H3_REF_LATENT` 跳过参考编码）总计 478 s，所以参考的 VAE 编码与 vision tower 一起约 28 s。

这张 256² 输出是带纹理的噪声并带可见的块状结构：`CWICK_H3_MIN_AREA=1024` 把画布压到了 4 MP 以下，正是「画布」一节描述的行为，不是拼接或布局错误。

## 实测：Metal 路径

`--backend metal` 把 transformer 的投影与注意力都交给 GPU（`CWICK_H3_GPU=0`
可强制留在 CPU）。投影直接就地读 checkpoint 的 BF16 字节，路上不做任何反量化；
注意力是流式内核，从不真的生成分数矩阵。

同一台机器、同一个请求（`256×256` → 2048² 画布，7 text + 4096 图像行，6 步），
每行是一个采样步：

| 步内阶段 | `--backend cpu` | `--backend metal` |
|---|---:|---:|
| 投影（66.3 GB 权重、1346.6 GMAC） | 63.7 s | 54.3 s |
| 注意力 | 64.5 s | 9.0 s |
| SwiGLU | 3.9 s | 3.1 s |
| AdaLN 调制 | 0.25 s | 0.5 s |
| 其余（归一化、rope、残差、输出头） | 4.2 s | 7.2 s |
| **单步** | **136.6 s** | **74.3 s** |

| 整个请求 | cpu | metal |
|---|---:|---:|
| 文本编码 | 46.1 s | 63.5 s |
| DiT（6 步） | 874.3 s | 445.7 s |
| VAE 解码 | 28.3 s | 26.1 s |
| **总计** | **949.0 s** | **535.6 s** |

真正起作用的是注意力：它随行数二次增长，所以在「2048² 画布 + 三张参考图」
的 17408 行上，它决定这一次请求是几分钟还是几小时；4096 行处的 7 倍是下限，
不是上限。投影的收益刻意做小了：这个规模下 CPU 路径已经有 1.24 TFLOPS
（BF16 块反量化进 f32 arena 再交给 `cblas_sgemm`），内核实测 1.51 TFLOPS ——
缺的并不是那部分算力。

两条路径不是逐字节相同。单块起差 1.2e-3 —— 这是 BF16 舍入，因为 GPU 把激活
暂存成 BF16 块而 CPU 保留 f32 —— 经 50 块 × 6 步放大到解码 RGB 上的 8.6e-2。
场景、构图、光线一致，细节不一致；这与该 checkpoint 自身端到端跑 BF16 的容差相同。

### 实测：一次三参考图请求

`POST /v1/images/edits`，三张参考图，`size=1536x2048`，seed 7，6 步，同机
`--backend metal`，走的是真实 HTTP 路径。三张参考图按短边规则落到
1600×2016、1376×2048、1024×1376，即 3150 + 2752 + 1376 = 7278 参考行，加上
4070 生成行 —— **packed 序列 11348 行**，而不带参考的同画布是 4070 行。

| 阶段 | ms | 说明 |
|---|---:|---|
| 分词 | 57 492 | prompt + 视觉占位，4319 token |
| 文本编码 | 222 850 | 三张参考的 conditioner |
| pack（参考图 VAE 编码） | 91 169 | 三张参考，分块 |
| DiT，6 步 | 2 022 601 | 337 s/步 |
| VAE 解码 | 21 781 | 只解输出 |
| PNG | 140 | |
| **总计** | **2 416 421** | 40.3 min |

差距几乎全部由行数解释：2.79 倍的行数意味着 2.79 倍的权重搬运和 7.8 倍的注意力，
单步 74.3 → 337.0 s。参考行也是 conditioner 不再是噪声的原因：222.9 s + 91.2 s，
而纯文本 prompt 全部只有 63.5 s。

重复性：同一请求跑两次（`CWICK_H3_GPU` 默认开）分别是 2 364 432 ms 与
2 416 421 ms，相差 2.2%，`dit` 1 995 410 / 2 022 601 ms。第二次是在整机 VM 压力
下跑的（swap 已用 45 GiB、约 154 MiB/s 正在写出），代价 2.2% —— 工作集装得下，
所以这一步不是内存受限的。

三张参考各自到位：脸图定人（发型、五官），服装图逐件迁移（外套、连衣裙、丝袜、
长袜、靴子、包），姿势图定机位（正面、双臂下垂、一只手略前）。产物 1536×2048、
全身在框内、均匀灰底影棚光、无文字无水印。

### 驻留：`CWICK_H3_WIRE=1`

两棵 checkpoint 各有 62 GiB。把两份都 wire 起来要 124 GiB，128 GiB 的机器装不下，
所以开关**只 wire transformer** —— 它是每一步都要整份重读的那棵；文本编码器每个请求
只读一次，刻意保持可回收。打开开关后，模型打开时会报 `61.74 GiB faulted in and wired`，
耗时 26.7 s（2.5 GB/s），而不是 3.5 s。这是每个进程一次性的开销，不是每请求。

同机、同一请求（1536×2048、seed 7、`--backend metal`、15 text + 4070 图像行、
seq 4085），先跑 A 再跑 B，这样不驻留那组不会继承到 wire 组的预热：

| | `CWICK_H3_WIRE=0` | `CWICK_H3_WIRE=1` |
|---|---:|---:|
| 模型打开 | 3.5 s | 26.9 s |
| 投影，6 步均值 | 52.57 s | **44.93 s** |
| 单步，均值 | 69.94 s | **60.37 s** |
| 单步，6 步范围 | 62.7 … 79.9 s | **60.2 … 60.7 s** |
| 单步，标准差 | 6.29 s | **0.17 s** |
| DiT，6 步 | 419.7 s | **362.2 s** |
| 文本编码 / VAE | 63.4 / 23.2 s | 63.5 / 23.3 s |
| 请求总计 | 506.8 s | **449.6 s** |
| CLI 墙钟（含模型打开） | 511 s | **478 s** |

DiT 收益 13.7%，请求 11.3%；一次性 CLI 扣掉 23.5 s 的 wire 开销仍净省 33 s。更有
说服力的是方差：不驻留时投影在 15 s 的范围内逐游走，因为权重页在循环中被回收又重新
换入；驻留后稳定落在 44.6–45.1 s。这是驻留问题，不是算术问题。两张 PNG 逐字节相同
（SHA-256 一致）—— wire 改变的是页放在哪里，不是页里的内容。

## 验证

上面那次全链路的产物是一张干净的 512×512 写实图像：无网格、无噪点，121 块边界上没有接缝。

`make test` 覆盖 PNG 编解码往返（尺寸、逐字节误差 ≤1）。上表的数值对齐是通过「数值对齐」一节里的环境开关对参考实现测的，不在测试套件里。`images/edits` 的 HTTP 路径另做了实机验证：无 `image` 与坏 `image` 均返回 400 并带明确原因，正常请求返回 200 与 `b64_json`。

