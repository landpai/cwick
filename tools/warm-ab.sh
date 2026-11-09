#!/bin/bash
# A/B: first request after a cold serve start, with and without --warmup.
# Same machine, same prompt, same expert-cache budget, a fresh prefix cache per
# arm so both sides prefill in full.
#
# Two lessons are baked in:
#   - readiness comes from THIS server's own "listening" log line, not from
#     probing the port. A port probe can be answered by the previous run's
#     server, which may not have released the listener yet.
#   - the response is validated to contain "choices" before the server is
#     stopped, so a refused connection cannot be mistaken for a fast answer.
#
# Usage: tools/warm-ab.sh [port]   (default 8002)
set -u

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || exit 1
cd "$ROOT" || exit 1

MODEL=${CWICK_ROOT:-./zai-org/GLM-5.3-Flash}
PORT=${1:-8002}
PROMPT="$(python3 -c 'print("You are a careful assistant. "*40)')"

export CWICK_PROFILE=1
export CWICK_EXPERT_CACHE_GIB=${CWICK_EXPERT_CACHE_GIB:-83}

python3 - "$PROMPT" <<'PY' > /tmp/warm-req.json
import json, sys
print(json.dumps({"model": "glm-5.3-flash",
                  "messages": [{"role": "user", "content": sys.argv[1]}],
                  "max_tokens": 10, "stream": False}))
PY

run_one() {
    name=$1; shift
    rm -rf "/tmp/pfx-$name"
    ./cwick serve --root "$MODEL" --host 127.0.0.1 --port "$PORT" \
        --backend metal --verbose --trace --context 131072 \
        --prefix-cache-dir "/tmp/pfx-$name" "$@" > "/tmp/warm-$name.log" 2>&1 &
    SRV=$!

    for _ in $(seq 1 300); do
        grep -q "listening on http://127.0.0.1:$PORT" "/tmp/warm-$name.log" && break
        kill -0 "$SRV" 2>/dev/null || { echo "[$name] server died during load"; return 1; }
        sleep 2
    done
    echo "[$name] listening at $(date +%T), sending request"

    curl -s --max-time 3000 "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' -d @/tmp/warm-req.json \
        > "/tmp/warm-$name-resp.json"
    echo "[$name] request done at $(date +%T)"

    if ! grep -q '"choices"' "/tmp/warm-$name-resp.json"; then
        echo "[$name] FAILED: response has no choices: $(head -c 200 "/tmp/warm-$name-resp.json")"
    fi

    kill -INT "$SRV" 2>/dev/null
    wait "$SRV" 2>/dev/null
    echo "[$name] server stopped"
}

run_one cold
sleep 3
run_one warm --warmup 32

echo "=== per-request numbers ==="
for name in cold warm; do
    printf '%-5s ' "$name"
    grep -E "profile:" "/tmp/warm-$name.log" | tail -1
done
echo "=== DONE ==="
