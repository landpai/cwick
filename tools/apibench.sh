#!/bin/sh
# tools/apibench.sh — full-chain timings for the four dialects of one resident
# checkpoint. Drives a *running* server over HTTP, the same way a client would,
# and prints one row per request plus a mean per endpoint.
#
# "Full chain" here means what a caller waits for: from the moment the request
# is written to the moment the last byte of the answer arrives. TTFT comes from
# curl's `time_starttransfer` (the first byte of an SSE stream, i.e. the
# prefill's end), and the token counts come from the server's own summary line
# (`prompt=N gen=M prefill=... decode=...`), so the prefill and decode rates in
# the output are the engine's numbers, not a client-side re-derivation.
#
# Usage:
#   sh tools/apibench.sh [--host http://127.0.0.1:8001] [--password PW]
#                        [--model ID] [--tokens N] [--repeat K] [--log PATH]
#
# --password is the server's single-user login password (start.sh defaults to
# cwick-local); the script exchanges it for a Bearer token with
# POST /v1/auth/login. --log is the server's stdout capture (`tee` file); when
# it is omitted the token counts are reported as "-" and only the timings are
# shown.
set -u

HOST="http://127.0.0.1:8001"
PASSWORD="${CWICK_PASSWORD:-cwick-local}"
MODEL=""
TOKENS=64
REPEAT=3
LOG=""

while [ $# -gt 0 ]; do
    case "$1" in
        --host) HOST=$2; shift 2 ;;
        --password) PASSWORD=$2; shift 2 ;;
        --model) MODEL=$2; shift 2 ;;
        --tokens) TOKENS=$2; shift 2 ;;
        --repeat) REPEAT=$2; shift 2 ;;
        --log) LOG=$2; shift 2 ;;
        *) echo "apibench: unknown option '$1'" >&2; exit 2 ;;
    esac
done

# A paragraph long enough that prefill is a real number rather than a warm-up,
# and short enough that a 1.5 tok/s checkpoint still finishes a request.
PROMPT='Memory bandwidth is the rate at which a processor can move data to and from its main memory. Latency is the time one access takes. They are independent: a machine can have short latency and narrow bandwidth, or the reverse. Streaming kernels are limited by bandwidth, pointer-chasing kernels by latency. Summarize the distinction in one sentence.'

command -v curl >/dev/null 2>&1 || { echo "apibench: curl is required" >&2; exit 2; }

# The server refuses anonymous requests once --lan/--web is on, so trade the
# password for a session token first.
TOKEN=$(curl -s --noproxy '*' -m 10 -X POST "$HOST/v1/auth/login" \
    -H 'Content-Type: application/json' \
    -d "{\"password\":\"$PASSWORD\"}" |
    sed -n 's/.*"token":"\([^"]*\)".*/\1/p')
if [ -z "$TOKEN" ]; then
    echo "apibench: login failed (POST $HOST/v1/auth/login)" >&2
    exit 1
fi

if [ -z "$MODEL" ]; then
    MODEL=$(curl -s --noproxy '*' -m 10 "$HOST/v1/models" \
        -H "Authorization: Bearer $TOKEN" |
        sed -n 's/.*"id":"\([^"]*\)".*/\1/p' | head -1)
fi
[ -n "$MODEL" ] || { echo "apibench: cannot determine the resident model" >&2; exit 1; }

echo "apibench: $HOST model=$MODEL tokens=$TOKENS repeat=$REPEAT"
printf '%-10s %-3s %9s %9s %5s %7s %5s %8s %9s\n' \
    endpoint run first_ms total_ms http prompt gen prefill_tps decode_tps

# One request. $1 = endpoint, $2 = JSON body, $3 = run number.
run_one() {
    ep=$1; body=$2; run=$3
    case "$ep" in
        chat)      path=/v1/chat/completions ;;
        responses) path=/v1/responses ;;
        messages)  path=/v1/messages ;;
        systemone) path=/v1/systemone ;;
    esac
    log_from=0
    if [ -n "$LOG" ] && [ -f "$LOG" ]; then
        log_from=$(wc -c < "$LOG" | tr -d ' ')
    fi
    out=$(curl -s --noproxy '*' -m 900 -o /dev/null \
        -w '%{time_starttransfer} %{time_total} %{http_code} %{size_download}' \
        -X POST "$HOST$path" \
        -H 'Content-Type: application/json' \
        -H "Authorization: Bearer $TOKEN" \
        -d "$body")
    ttft=$(echo "$out" | awk '{printf "%.0f", $1 * 1000}')
    total=$(echo "$out" | awk '{printf "%.0f", $2 * 1000}')
    code=$(echo "$out" | awk '{print $3}')
    bytes=$(echo "$out" | awk '{printf "%.0f", $4}')

    # Server-side summary line for this request, when a log was supplied.
    prompt="-"; gen="-"; pf="-"; dc="-"
    if [ -n "$LOG" ] && [ -f "$LOG" ]; then
        line=$(tail -c +"$((log_from + 1))" "$LOG" |
            grep 'steps .* tok/s' | tail -1)
        if [ -n "$line" ]; then
            prompt=$(echo "$line" | sed -n 's/.*prompt=\([0-9]*\).*/\1/p')
            gen=$(echo "$line" | sed -n 's/.*gen=\([0-9]*\).*/\1/p')
            pf=$(echo "$line" | sed -n 's#.*prefill=[0-9]*ms/\([0-9.a-z]*\).*#\1#p')
            dc=$(echo "$line" | sed -n 's#.*steps \([0-9.a-z]*\) tok/s.*#\1#p')
        fi
    fi
    printf '%-10s %-3s %8s %9s %5s %8s %8s %8s %9s\n' \
        "$ep" "$run" "$ttft" "$total" "$code" "$prompt" "$gen" "$pf" "$dc"
    [ "$code" = "200" ] || echo "apibench: $ep returned HTTP $code ($bytes bytes)" >&2
}

chat_body() {
    printf '{"model":"%s","messages":[{"role":"user","content":%s}],"max_tokens":%s,"stream":true,"reasoning_effort":"none"}' \
        "$MODEL" "$(printf '%s' "$PROMPT" | sed 's/"/\\"/g; s/.*/"&"/')" "$TOKENS"
}
resp_body() {
    printf '{"model":"%s","input":%s,"max_output_tokens":%s,"stream":true,"reasoning":{"effort":"none"}}' \
        "$MODEL" "$(printf '%s' "$PROMPT" | sed 's/"/\\"/g; s/.*/"&"/')" "$TOKENS"
}
msg_body() {
    printf '{"model":"%s","max_tokens":%s,"stream":true,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":%s}]}' \
        "$MODEL" "$TOKENS" "$(printf '%s' "$PROMPT" | sed 's/"/\\"/g; s/.*/"&"/')"
}
# One decision: a state plus one question of each type. No decode, no stream --
# a decision is a single JSON object, and `stream` is refused by name.
jev_body() {
    printf '{"model":"%s","state":%s,"questions":{"move":{"type":"choice","instructions":"Which option best describes the paragraph?","criteria":{"bw":"it is about bandwidth","lat":"it is about latency"}},"clarity":{"type":"score","instructions":"How clearly is the distinction made?","criteria":["unclear","partial","clear"]},"correct":{"type":"noul","instructions":"Does the paragraph say the two are independent?","criteria":{"true":"it does","false":"it does not"}}}}' \
        "$MODEL" "$(printf '%s' "$PROMPT" | sed 's/"/\\"/g; s/.*/"&"/')"
}

i=1
while [ "$i" -le "$REPEAT" ]; do
    run_one chat "$(chat_body)" "$i"
    run_one responses "$(resp_body)" "$i"
    run_one messages "$(msg_body)" "$i"
    run_one systemone "$(jev_body)" "$i"
    i=$((i + 1))
done
