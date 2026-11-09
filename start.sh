#!/bin/sh
# Serve a checkpoint on 0.0.0.0:8001 with the prefix cache on.
#
# Usage: ./start.sh [glm-5.3-flash | deepSeek-v4.1-flash-1017 | MiniMax-M3.1-Flash]
# (bare "glm" / "deepseek" / "minimax" work too; default is
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
        echo "unknown model '$1' (want glm-5.3-flash, deepSeek-v4.1-flash-1017 or MiniMax-M3.1-Flash)" >&2
        exit 2 ;;
esac

CWICK_SEED=42 ./cwick serve --root "$ROOT" --host 0.0.0.0 --lan --web --jev --port 8001 \
    --verbose --backend metal --trace \
    --context "$CTX"  --max-tokens 8192 \
    --prefix-cache-dir ./pfxcache --password "${CWICK_PASSWORD:-cwick-local}" \
    --store ./dbs/ --models-dir ./zai-org/GLM-5.3-Flash \
    --models-dir ./deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    --models-dir ./MiniMaxAI/MiniMax-M3.1-Flash --prefix-cache-gib 80 \
    | tee "${CWICK_LOG:-./logs/cwick-serve.log}"
