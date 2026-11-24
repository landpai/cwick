#!/bin/sh
# tools/apibench.sh — client-visible full-chain timings for one resident model.
# Streaming TTFT is the first non-empty content/reasoning/tool delta, not the
# arrival of HTTP headers or the initial role-only SSE frame.
set -u

HOST="http://127.0.0.1:8001"
PASSWORD="${CWICK_PASSWORD:-}"
MODEL=""
TOKENS=64
REPEAT=3
LOG=""
IMAGE=""
TEMPERATURE=0
STREAM=1

while [ $# -gt 0 ]; do
    case "$1" in
        --host) HOST=$2; shift 2 ;;
        --password) PASSWORD=$2; shift 2 ;;
        --model) MODEL=$2; shift 2 ;;
        --tokens) TOKENS=$2; shift 2 ;;
        --repeat) REPEAT=$2; shift 2 ;;
        --log) LOG=$2; shift 2 ;;
        --image) IMAGE=$2; shift 2 ;;
        --temperature) TEMPERATURE=$2; shift 2 ;;
        --no-stream) STREAM=0; shift ;;
        *) echo "apibench: unknown option '$1'" >&2; exit 2 ;;
    esac
done

PROMPT='Memory bandwidth is the rate at which a processor can move data to and from its main memory. Latency is the time one access takes. They are independent: a machine can have short latency and narrow bandwidth, or the reverse. Streaming kernels are limited by bandwidth, pointer-chasing kernels by latency. Summarize the distinction in one sentence.'
if [ -n "$IMAGE" ]; then
    PROMPT='Describe the main subject and colors in this image in one sentence.'
fi

command -v curl >/dev/null 2>&1 || { echo "apibench: curl is required" >&2; exit 2; }
TMP=$(mktemp -d "${TMPDIR:-/tmp}/cwick-apibench.XXXXXX") || exit 1
trap 'rm -rf "$TMP"' EXIT HUP INT TERM

AUTH=""
if [ -n "$PASSWORD" ]; then
    TOKEN=$(curl -s --noproxy '*' -m 10 -X POST "$HOST/v1/auth/login" \
        -H 'Content-Type: application/json' \
        -d "{\"password\":\"$PASSWORD\"}" |
        sed -n 's/.*"token":"\([^"]*\)".*/\1/p')
    if [ -n "$TOKEN" ]; then
        AUTH="Authorization: Bearer $TOKEN"
    else
        echo "apibench: login failed (POST $HOST/v1/auth/login)" >&2
        exit 1
    fi
fi

if [ -z "$MODEL" ]; then
    MODEL=$(curl -s --noproxy '*' -m 10 ${AUTH:+-H "$AUTH"} "$HOST/v1/models" |
        sed -n 's/.*"id":"\([^"]*\)".*/\1/p' | head -1)
fi
[ -n "$MODEL" ] || { echo "apibench: cannot determine the resident model" >&2; exit 1; }

json_quote() {
    printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g; s/.*/"&"/'
}

PROMPT_JSON=$(json_quote "$PROMPT")
CONTENT_JSON=$PROMPT_JSON
if [ -n "$IMAGE" ]; then
    [ -f "$IMAGE" ] || { echo "apibench: image not found: $IMAGE" >&2; exit 2; }
    B64=$(base64 -i "$IMAGE" | tr -d '\r\n')
    case "$IMAGE" in
        *.png|*.PNG) MIME=image/png ;;
        *.jpg|*.jpeg|*.JPG|*.JPEG) MIME=image/jpeg ;;
        *.webp|*.WEBP) MIME=image/webp ;;
        *) echo "apibench: unsupported image suffix: $IMAGE" >&2; exit 2 ;;
    esac
    IMAGE_JSON="{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:$MIME;base64,$B64\"}}"
    IMAGE_DATA_URL="data:$MIME;base64,$B64"
    CONTENT_JSON="[{\"type\":\"text\",\"text\":$PROMPT_JSON},$IMAGE_JSON]"
    ANTHROPIC_IMAGE="{\"type\":\"image\",\"source\":{\"type\":\"base64\",\"media_type\":\"$MIME\",\"data\":\"$B64\"}}"
    ANTHROPIC_CONTENT="[{\"type\":\"text\",\"text\":$PROMPT_JSON},$ANTHROPIC_IMAGE]"
else
    IMAGE_DATA_URL=""
    ANTHROPIC_CONTENT=$PROMPT_JSON
fi
STREAM_JSON=true
[ "$STREAM" = 1 ] || STREAM_JSON=false

chat_body() {
    printf '{"model":"%s","messages":[{"role":"user","content":%s}],"max_tokens":%s,"temperature":%s,"stream":%s,"reasoning_effort":"none"}' \
        "$MODEL" "$CONTENT_JSON" "$TOKENS" "$TEMPERATURE" "$STREAM_JSON"
}
resp_body() {
    printf '{"model":"%s","input":[{"role":"user","content":%s}],"max_output_tokens":%s,"temperature":%s,"stream":%s,"reasoning":{"effort":"none"}}' \
        "$MODEL" "$CONTENT_JSON" "$TOKENS" "$TEMPERATURE" "$STREAM_JSON"
}
msg_body() {
    printf '{"model":"%s","max_tokens":%s,"temperature":%s,"stream":%s,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":%s}]}' \
        "$MODEL" "$TOKENS" "$TEMPERATURE" "$STREAM_JSON" "$ANTHROPIC_CONTENT"
}
jev_body() {
    if [ -n "$IMAGE" ]; then
        printf '{"model":"%s","state":%s,"images":["%s"],"questions":{"subject":{"type":"choice","instructions":"What is the main subject of the image?","criteria":{"carrot":"carrot","corn":"corn","other":"something else"}}}}' \
            "$MODEL" "$PROMPT_JSON" "$IMAGE_DATA_URL"
    else
        printf '{"model":"%s","state":%s,"questions":{"move":{"type":"choice","instructions":"Which option best describes the paragraph?","criteria":{"bw":"it is about bandwidth","lat":"it is about latency"}},"clarity":{"type":"score","instructions":"How clearly is the distinction made?","criteria":["unclear","partial","clear"]},"correct":{"type":"noul","instructions":"Does the paragraph say the two are independent?","criteria":{"true":"it does","false":"it does not"}}}}' \
            "$MODEL" "$PROMPT_JSON"
    fi
}

trace_first_delta_ms() {
    awk '
        /=> Send data/ {
            if (!started) {
                split($1, t, ":");
                started = (((t[1] * 60) + t[2]) * 60) * 1000 + t[3] * 1000;
            }
            next;
        }
        /<= Recv data/ {
            recv = 1;
            split($1, t, ":");
            stamp = (((t[1] * 60) + t[2]) * 60) * 1000 + t[3] * 1000;
            next;
        }
        /=> / { recv = 0; next; }
        recv && /("content":"[^"]|"text":"[^"]|"reasoning":"[^"]|"reasoning_content":"[^"]|"tool_calls"|content_block_delta|response\.reasoning_text\.delta|response\.reasoning_summary_text\.delta|output_text\.delta|function_call_arguments\.delta)/ {
            delta = stamp - started;
            if (delta < 0) delta += 86400000;
            printf "%.0f", delta;
            exit
        }
    ' "$1"
}

printf 'apibench: %s model=%s tokens=%s repeat=%s stream=%s image=%s temperature=%s\n' \
    "$HOST" "$MODEL" "$TOKENS" "$REPEAT" "$STREAM" "${IMAGE:-none}" "$TEMPERATURE"
printf '%-10s %-3s %9s %9s %5s %8s %5s %9s %9s\n' \
    endpoint run first_ms total_ms http prompt gen prefill_tps decode_tps
: > "$TMP/rows"

run_one() {
    ep=$1; body=$2; run=$3
    case "$ep" in
        chat) path=/v1/chat/completions ;;
        responses) path=/v1/responses ;;
        messages) path=/v1/messages ;;
        systemone) path=/v1/systemone ;;
    esac
    log_from=0
    if [ -n "$LOG" ] && [ -f "$LOG" ]; then
        log_from=$(wc -c < "$LOG" | tr -d ' ')
    fi
    out=$(curl -sS --noproxy '*' -m 900 -o "$TMP/body" \
        --trace-ascii "$TMP/trace" --trace-time \
        -w '%{time_starttransfer} %{time_total} %{http_code} %{size_download}' \
        ${AUTH:+-H "$AUTH"} -X POST "$HOST$path" \
        -H 'Content-Type: application/json' -d "$body")
    header_ms=$(echo "$out" | awk '{printf "%.0f", $1 * 1000}')
    total=$(echo "$out" | awk '{printf "%.0f", $2 * 1000}')
    code=$(echo "$out" | awk '{print $3}')
    bytes=$(echo "$out" | awk '{printf "%.0f", $4}')
    first=$header_ms
    if [ "$ep" != systemone ] && [ "$STREAM" = 1 ]; then
        delta=$(trace_first_delta_ms "$TMP/trace")
        [ -n "$delta" ] && first=$delta
    fi

    prompt="-"; gen="-"; pf="-"; dc="-"
    if [ -n "$LOG" ] && [ -f "$LOG" ]; then
        line=$(tail -c +"$((log_from + 1))" "$LOG" |
            grep -E 'tokens in .*prefill' | tail -1)
        prompt_line=$(tail -c +"$((log_from + 1))" "$LOG" |
            grep -E 'prompt encoded: [0-9]+ tokens' | tail -1)
        if [ -n "$line" ]; then
            prompt=$(echo "$prompt_line" | sed -n 's/.*prompt encoded: \([0-9]*\) tokens.*/\1/p')
            gen=$(echo "$line" | sed -n 's/.*— \([0-9]*\) generated.*/\1/p')
            pf=$(echo "$line" | sed -n 's/.*prefill \([0-9]*\) ms (\([0-9.]*\) tok\/s).*/\2/p')
            dc=$(echo "$line" | sed -n 's/.*tokens in \([0-9]*\) ms (\([0-9.]* tok\/s\).*/\2/p')
        elif [ "$ep" = systemone ]; then
            line=$(tail -c +"$((log_from + 1))" "$LOG" |
                grep -E '\[systemone\] decision done:' | tail -1)
            prompt=$(echo "$line" | sed -n 's/.*decision done: .* \([0-9][0-9]*\) input tokens.*/\1/p')
            gen=0
            pf=$(echo "$line" | sed -n 's/.*wall, \([0-9.]*\) tok\/s prefill.*/\1/p')
        fi
    fi
    printf '%-10s %-3s %8s %9s %5s %8s %8s %8s %9s\n' \
        "$ep" "$run" "$first" "$total" "$code" "$prompt" "${gen:--}" "${pf:--}" "${dc:--}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$ep" "$first" "$total" "$code" "$prompt" "$gen" "$pf" "$dc" >> "$TMP/rows"
    [ "$code" = "200" ] || echo "apibench: $ep returned HTTP $code ($bytes bytes): $(tr '\n' ' ' < "$TMP/body")" >&2
}

i=1
while [ "$i" -le "$REPEAT" ]; do
    run_one chat "$(chat_body)" "$i"
    run_one responses "$(resp_body)" "$i"
    run_one messages "$(msg_body)" "$i"
    run_one systemone "$(jev_body)" "$i"
    i=$((i + 1))
done

awk -F '\t' '
    NF >= 3 { n[$1]++; first[$1] += $2; total[$1] += $3; if ($4 == 200) ok[$1]++ }
    END {
        print "mean        runs      first_ms      total_ms  http_ok";
        split("chat responses messages systemone", order, " ");
        for (i = 1; i <= 4; i++) {
            e = order[i];
            if (n[e]) printf "%-10s %5d %13.0f %13.0f %8d\n", e, n[e], first[e]/n[e], total[e]/n[e], ok[e];
        }
    }
' "$TMP/rows"
