# Makefile — CWICK: full-model C runtime + OpenAI-compatible server.
#
# Targets:
#   make            build the cwick binary
#   make test       build and run the end-to-end verification suite
#   make probe      build tools/pf8probe (GLM FP8 dequantization semantics)
#   make clean

# The generated-asset rule below comes first, which would otherwise make
# `make` build only that file: keep the binary as the default goal.
.DEFAULT_GOAL := cwick

CC      ?= clang
CFLAGS   = -O2 -Wall -Wextra -fobjc-arc -I src -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64
LDFLAGS  = -framework Accelerate -lpthread -framework Metal -framework Foundation \
           -framework CoreImage -framework ImageIO -framework CoreGraphics -lsqlite3 \
           -lcurl

SRCS = src/util.c src/json.c src/mem.c src/log.c src/logcap.c src/auth.c src/model.c \
       src/safetensors.c \
       src/ds_tables.c src/ds_quant.c src/ds_ops.c src/ds_model.c src/ds_engine.c \
       src/ds_prompt.c src/mm_model.c src/mm_engine.c src/mm_prompt.c \
       src/q35_model.c src/q35_engine.c src/q35_prompt.c src/q35_gpu.c src/q35_shader.c \
       src/spine.c src/tensor.c src/pool.c \
       src/attention.c src/moe.c src/engine.c src/sample.c src/tokenizer.c src/tools.c \
       src/setup.c src/stream.c src/prof.c src/metrics.c src/pfx.c src/metal.c \
       src/metal_shader.c src/metal_objc.m src/qr.m src/q35_gpu.m src/webui.c src/webui_embed.c \
       src/models.c src/store.c src/server.c src/jev.c src/hub.c src/main.c
HDRS = src/cwick.h src/util.h src/json.h src/log.h src/logcap.h src/auth.h src/mem.h \
       src/metrics.h src/models.h src/store.h src/hub.h \
       src/tools.h src/pfx.h src/webui.h src/qr.h src/jev.h \
       src/ds.h src/ds_ops.h src/ds_model.h src/ds_engine.h src/ds_prompt.h src/sample.h \
       src/mm.h src/mm_engine.h src/mm_prompt.h \
       src/q35.h src/q35_engine.h src/q35_prompt.h src/q35_gpu.h

# The chat UI and apps are compiled into the binary (see src/webui.h).
# web/ and apps/ are the source of truth; src/webui_embed.c is generated and
# committed, so a plain checkout builds without Python. The generator is C
# (tools/embed_web.c) and is built on demand; the rule only fires after an edit.
WEB_SRC = web/index.html web/app.css web/app.js web/login.html
APP_SRC = $(shell find apps -type f 2>/dev/null)
EMBED   = tools/embed_web

$(EMBED): tools/embed_web.c
	$(CC) -O2 -Wall -Wextra -o $@ tools/embed_web.c

src/webui_embed.c: $(WEB_SRC) $(APP_SRC) $(EMBED) tools/embed_web.c
	./$(EMBED)

cwick: $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

TEST_SRCS = src/util.c src/json.c src/mem.c src/log.c src/logcap.c src/auth.c src/model.c \
            src/safetensors.c \
            src/ds_tables.c src/ds_quant.c src/ds_ops.c src/ds_model.c src/ds_engine.c \
       src/ds_prompt.c src/mm_model.c src/mm_engine.c src/mm_prompt.c \
       src/q35_model.c src/q35_engine.c src/q35_prompt.c src/q35_gpu.c src/q35_shader.c \
            src/spine.c src/tensor.c src/pool.c \
            src/attention.c src/moe.c src/engine.c src/sample.c src/tokenizer.c src/tools.c \
            src/setup.c src/stream.c src/prof.c src/metrics.c src/pfx.c src/metal.c \
            src/metal_shader.c src/metal_objc.m src/qr.m src/q35_gpu.m src/webui.c src/webui_embed.c \
            src/models.c src/store.c src/server.c src/jev.c src/hub.c tests/test_cwick.c

test: $(TEST_SRCS) $(HDRS)
	$(CC) $(CFLAGS) -o tests/test_cwick $(TEST_SRCS) $(LDFLAGS)
	./tests/test_cwick

# tools/pf8probe settles the GLM FP8 E4M3 dequantization direction from
# checkpoint bytes (see docs/PRECISION.md). It links the same objects as the
# test binary because the safetensors loader reaches into mem/metrics.
probe: tools/pf8probe.c $(filter-out tests/test_cwick.c,$(TEST_SRCS)) $(HDRS)
	$(CC) $(CFLAGS) -o tools/pf8probe tools/pf8probe.c \
		$(filter-out tests/test_cwick.c,$(TEST_SRCS)) $(LDFLAGS)

# The GPU-vs-CPU decode-step prototype for the dense checkpoint: same shape
# mix, same number of commit+waits a real layer would make (see DECISIONS §61).
gemvstep: tools/q35step.c $(filter-out tests/test_cwick.c,$(TEST_SRCS)) $(HDRS)
	$(CC) $(CFLAGS) -o tools/q35step tools/q35step.c \
		$(filter-out tests/test_cwick.c,$(TEST_SRCS)) $(LDFLAGS)

# Read the MiniMax-M3.1-Flash quantizer constants out of the shards and print
# what a decoded weight looks like (see the header comment in the source).
mmprobe: tools/mm_probe.c $(filter-out src/main.c,$(SRCS)) $(HDRS)
	$(CC) $(CFLAGS) -o tools/mm_probe tools/mm_probe.c \
		$(filter-out src/main.c,$(SRCS)) $(LDFLAGS)

# tools/ds_gemvbench measures the DeepSeek quantized GEMVs on their own, so a
# decode step's cost can be split into "bytes per second" and "MACs per
# second" without a 237 GiB checkpoint in the way.
bench: tools/ds_gemvbench.c $(filter-out src/main.c,$(SRCS)) $(HDRS)
	$(CC) $(CFLAGS) -o tools/ds_gemvbench tools/ds_gemvbench.c \
		$(filter-out src/main.c,$(SRCS)) $(LDFLAGS)

# tools/ds_run and tools/ds_layertest recompute whole DeepSeek blocks from the
# shard bytes and compare them against the engine's own trace, which is the only
# check that catches a wiring mistake between the primitives (a GEMV can be
# right on its own and still be fed the wrong row range). ds_layertest drives
# ds_run as a child process, so both have to exist and be current together.
layercheck: tools/ds_run tools/ds_layertest

tools/ds_run: tools/ds_run.c $(filter-out src/main.c,$(SRCS)) $(HDRS)
	$(CC) $(CFLAGS) -o $@ tools/ds_run.c $(filter-out src/main.c,$(SRCS)) $(LDFLAGS)

tools/ds_layertest: tools/ds_layertest.c $(filter-out src/main.c,$(SRCS)) $(HDRS)
	$(CC) $(CFLAGS) -o $@ tools/ds_layertest.c \
		$(filter-out src/main.c,$(SRCS)) $(LDFLAGS)

# tools/mm_decodeprobe measures the two ceilings a decode step is scored
# against: resident-memory read bandwidth, and the shards' pread bandwidth with
# every thread on one file vs spread across files. No model and no kernels, so
# it builds on its own.
mmdecodeprobe: tools/mm_decodeprobe.c
	$(CC) -O2 -Wall -Wextra -o tools/mm_decodeprobe tools/mm_decodeprobe.c

# tools/jevprobe answers "can this checkpoint answer a decision?" from its
# tokenizer alone: the endpoint reads one token per option letter, and whether
# a letter is one token is a property of the tokenizer. No weights load.
jevprobe: tools/jevprobe.c $(filter-out src/main.c,$(SRCS)) $(HDRS)
	$(CC) $(CFLAGS) -o tools/jevprobe tools/jevprobe.c \
		$(filter-out src/main.c,$(SRCS)) $(LDFLAGS)

clean:
	rm -f cwick tests/test_cwick $(EMBED) tools/pf8probe tools/ds_gemvbench
	rm -f tools/ds_run tools/ds_layertest tools/mm_decodeprobe tools/jevprobe
	rm -rf /tmp/cwick-test-*

.PHONY: test clean probe bench layercheck mmdecodeprobe jevprobe

# Drive the MiniMax-M3.1-Flash port on its own (see the header in the source).
# bf16/fp16 dot-product throughput, one thread. Needs -march because bf16 is an
# optional feature the default target does not enable; fp16 is not, which is
# itself part of the answer this tool exists to give.
tools/bfdotbench: tools/bfdotbench.c
	$(CC) -O2 -Wall -march=armv8.2-a+bf16 -o $@ tools/bfdotbench.c

# Attention kernels at the shape qwen3.5 runs: the score pass and the output
# pass, hand-written against cblas, same MAC count. This is the tool that
# decided the rewrite -- it prints GMAC/s for each, one thread.
tools/attnbench: tools/attnbench.c
	$(CC) -O2 -Wall -Wno-deprecated-declarations -o $@ tools/attnbench.c \
		-framework Accelerate

tools/mm_run: tools/mm_run.c $(filter-out src/main.c,$(SRCS)) $(HDRS)
	$(CC) $(CFLAGS) -o $@ tools/mm_run.c \
		$(filter-out src/main.c,$(SRCS)) $(LDFLAGS)
