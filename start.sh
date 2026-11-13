#!/bin/sh
# Serve a checkpoint on 0.0.0.0:8001 with the prefix cache on.
#
# Usage: ./start.sh [glm-5.3-flash | deepSeek-v4.1-flash-1017 | MiniMax-M3.1-Flash
#                    | qwen3.5-9b]
# (bare "glm" / "deepseek" / "minimax" / "qwen" work too; default is
# deepSeek-v4.1-flash-1017 — switching models later needs no restart:
# POST /v1/models/switch.)
#
# The prefix cache is the reason to use this script rather than invoking cwick
# directly: an agent client resends the same long system prompt every turn,
# and a cache hit turns that turn's TTFT from "re-prefill the whole thing"
# (~350 s measured on a 1018-token prompt) into a sequential read of one
# snapshot (~6 s). Drop --prefix-cache-dir to disable it.
#
# Snapshots are ~217 MiB per 1000 prompt tokens at this checkpoint, so the
# 8 GiB default holds roughly 35 of them; raise --prefix-cache-gib for a long
# session, or watch cwick:prefix_tokens_reused_total to see when eviction
# starts costing replays.
#
# Memory: the planner fills the expert cache to whatever the 80% ceiling
# leaves over (see docs/STORAGE.md) — there is no CWICK_EXPERT_CACHE_GIB here
# on purpose. Set one only if another process on this machine needs the RAM.
#
# A request that omits max_tokens inherits --max-tokens (8192 here), which at
# ~1.3 tok/s is still ~1.7 h of decoding and always ends as a client-side abort
# -- observed in a long access log as three requests whose finish_reason is
# "abort" and none "stop". There is no server-side clamp for that any more:
# --max-output-tokens was removed because it only ever applied to requests that
# omitted max_tokens, and every real client (the built-in web UI included) sets
# the field, so the clamp was dead weight that also made /v1/config and the
# settings panel report a limit nothing was held to. Bound the turn with
# max_tokens in the request instead.
#
# --lan binds 0.0.0.0 and enables the web UI so a phone on the same Wi-Fi can
# scan the app's QR. It forces single-user password login: set CWICK_PASSWORD
# (or --password) before starting, otherwise cwick generates one and prints it
# once in the auth: startup line. HTTP only; use a trusted Wi-Fi or a TLS proxy.
cd "$(dirname "$0")" || exit 1

case "$1" in
    glm|glm-5.3-flash)
        ROOT=./zai-org/GLM-5.3-Flash
        CTX=500000 ;;
    qwen|qwen3.5|qwen3.5-9b|DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic)
        # Dense BF16, 18 GiB, no experts: nothing streams, so the whole
        # checkpoint stays resident -- faulted in and wired at load (~2.6 s),
        # not faulted in piecemeal during the first prefill. The token rate is
        # then the memory bus. The KV window is 64 KiB per position: 8
        # attention layers of 4x256 keys and values, held in f32 because both
        # attention passes are GEMMs (the snapshot on disk stays bf16, half the
        # bytes). Add a fixed 50 MiB of DeltaNet state and one 64 MiB score
        # tile, and 128K positions cost 8.1 GiB against 18 GiB of wired
        # weights -- affordable on a machine this size, and the model itself
        # allows 262144.
        #
        # 128K and not 32K: an agent client sends a 30K-token system prompt
        # with its tool list, and at 32K that left 1078 output tokens and made
        # the *next* turn fail outright. Snapshots for this checkpoint are
        # ~33 MiB per 1000 prompt tokens (KV only; the DeltaNet state is a
        # fixed 50 MiB), so the prefix cache holds far more of them than the
        # GLM size quoted above suggests.
        ROOT=./models/xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic
        CTX=131072 ;;
    mm|minimax|MiniMax-M3.1-Flash)
        # 32K positions, not 500K: this checkpoint's KV is bf16 GQA at 2.3 KiB
        # per position per layer, so 500K would be 64 GiB of cache -- taken out
        # of the expert cache, which is what the token rate actually runs on.
        ROOT=./MiniMaxAI/MiniMax-M3.1-Flash
        CTX=32768 ;;
    ""|ds|deepseek|deepSeek-v4.1-flash-1017)
        ROOT=./deepseek-ai/DeepSeek-V4.1-Flash-1017
        CTX=500000 ;;
    *)
        echo "unknown model '$1' (want glm-5.3-flash, deepSeek-v4.1-flash-1017, MiniMax-M3.1-Flash or qwen3.5-9b)" >&2
        exit 2 ;;
esac

CWICK_SEED=42 ./cwick serve --root "$ROOT" --host 0.0.0.0 --lan --web --jev --port 8001 \
    --verbose --backend metal --trace \
    --context "$CTX"  --max-tokens 8192 \
    --prefix-cache-dir ./pfxcache --password "${CWICK_PASSWORD:-cwick-local}" \
    --store ./dbs/ --models-dir ./zai-org/GLM-5.3-Flash \
    --models-dir ./deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    --models-dir ./MiniMaxAI/MiniMax-M3.1-Flash \
    --models-dir ./models/xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic \
    --prefix-cache-gib 80 \
    | tee "${CWICK_LOG:-./logs/cwick-serve.log}"
