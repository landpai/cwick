#!/bin/sh
# Verify the thinking budget (docs/DECISIONS.md §14) against a live server.
#
#   tools/thinkbudget-check.sh [port] [log]
#
# Sends the same question twice — once at 低 (cap 512) and once at 极致
# (cap 32768) — and then prints what the access log recorded for each. The
# 低 turn must show `think_cap=512(hit)` and a non-zero `content=`, i.e. the
# engine closed the think block and the answer followed. The 极致 turn runs
# under a cap its own thought will not reach, so it must show `think_cap=32768`
# without `(hit)` — the mechanism has to be invisible when it is not needed.
#
# Everything the check needs is already in the log line, so nothing is parsed
# here: `think=` / `content=` are the token split, `think_cap=` is the ceiling
# and whether it fired.
set -eu

PORT=${1:-8001}
LOG=${2:-logs/cwick-serve.log}
PASS=${CWICK_PASSWORD:-cwick-local}
HOST="http://127.0.0.1:$PORT"

TOK=$(curl -s -X POST "$HOST/v1/auth/login" -H 'Content-Type: application/json' \
      -d "{\"password\":\"$PASS\"}" -D - -o /dev/null \
      | sed -n 's/^[Ss]et-[Cc]ookie: *[^=]*=\([^;]*\).*/\1/p' | head -1)
if [ -z "${TOK:-}" ]; then
    echo "login failed on $HOST (password: \$CWICK_PASSWORD)" >&2
    exit 1
fi

ask() {   # $1 = thinking level, $2 = prompt
    curl -s --max-time 1200 "$HOST/v1/chat/completions" \
        -H "Authorization: Bearer $TOK" -H 'Content-Type: application/json' \
        -d "{\"model\":\"glm-5.3-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"$2\"}],\
\"thinking\":\"$1\",\"max_tokens\":1500,\"temperature\":0}" -o /dev/null
}

PROMPT='请用三句话解释什么是递归，并各举一个例子'

echo "== 低 (cap 512) =="
ask 低 "$PROMPT"
echo "== 极致 (cap 32768) =="
ask 极致 "$PROMPT"

echo
echo "== access log =="
tail -200 "$LOG" | grep 'POST /v1/chat/completions' | tail -2
echo
echo "expected: the first line carries think_cap=512(hit) with content>0;"
echo "          the second carries think_cap=32768 and no (hit)."
