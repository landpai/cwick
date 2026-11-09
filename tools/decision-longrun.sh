#!/bin/sh
# Long-run regression for POST /v1/systemone.
#
# Why this exists: a retention bug in the decode loop (docs/DECISIONS.md 37) was
# only visible after ~32 decoded tokens. The decision endpoint does not take that
# code path, but "it cannot be there" is a claim worth one measurement rather
# than an argument. This script is that measurement, and it is meant to be
# re-run after any change to prefill.
#
# What it does: starts its own server (one model, no switch), sends `rounds`
# requests, each carrying `q` questions over one state, and samples
# cwick:process_footprint_bytes immediately before and after each request. The
# state grows every round and the question set changes, so no round can be
# answered from the previous round's prefix cache.
#
# What to look for: the first request pays once for mapping and pools. After
# that, the per-round delta should be noise (tens of MiB) and should not grow
# with the round number. A delta of GiB per round is the failure mode this is
# built to catch -- that is what the decode leak looked like.
#
#   sh tools/decision-longrun.sh glm-5.3-flash zai-org/GLM-5.3-Flash 3 3
#
set -e

LABEL=${1:?usage: decision-longrun.sh <label> <root> [rounds] [questions] [port]}
ROOT=${2:?usage: decision-longrun.sh <label> <root> [rounds] [questions] [port]}
ROUNDS=${3:-3}
NQ=${4:-3}
PORT=${5:-8001}

HERE=$(cd "$(dirname "$0")/.." && pwd)
TMP=/tmp/cwick-longrun-$LABEL
rm -rf "$TMP"
mkdir -p "$TMP"

"$HERE/cwick" serve --root "$ROOT" --host 127.0.0.1 --jev --port "$PORT" \
    --backend metal --context 32768 --max-tokens 8192 --prefix-cache-gib 1 \
    --password cwick-local --store "$TMP/dbs" > "$TMP/serve.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT INT TERM

ready=0
i=0
while [ $i -lt 150 ]; do
    code=$(curl -s --noproxy '*' -m 2 -o /dev/null -w '%{http_code}' \
           "http://127.0.0.1:$PORT/v1/models" || true)
    if [ "$code" = "401" ] || [ "$code" = "200" ]; then ready=1; break; fi
    sleep 2
    i=$((i + 1))
done
if [ "$ready" != "1" ]; then
    echo "$LABEL: server did not start; last lines of $TMP/serve.log" >&2
    tail -20 "$TMP/serve.log" >&2
    exit 1
fi

TOK=$(curl -s --noproxy '*' -X POST "http://127.0.0.1:$PORT/v1/auth/login" \
      -H 'Content-Type: application/json' -d '{"password":"cwick-local"}' \
      | python3 -c 'import sys,json;print(json.load(sys.stdin).get("token",""))')
MODEL=$(curl -s --noproxy '*' "http://127.0.0.1:$PORT/v1/systemone" \
        -H "Authorization: Bearer $TOK" \
        | python3 -c 'import sys,json;print(json.load(sys.stdin).get("model",""))')
echo "$LABEL: model=$MODEL pid=$PID rounds=$ROUNDS questions=$NQ"

python3 - "$TMP" "$MODEL" "$ROUNDS" "$NQ" <<'PY'
import json, sys
tmp, model, rounds, nq = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

para = ("The support queue has been growing for three days. Half the tickets are "
        "the same question about a password reset link that expires before the "
        "mail arrives, the other half are refunds that were approved but never "
        "paid. Two agents are on leave next week and the team lead needs one "
        "decision rather than a list. ")
opts = [("refund", "give the money back and close the case"),
        ("replace", "send a replacement unit today"),
        ("credit", "offer store credit and keep the order"),
        ("repair", "pay for a repair"),
        ("explain", "explain the policy and change nothing"),
        ("escalate", "hand the case to a human owner"),
        ("wait", "wait one more day for more information"),
        ("decline", "decline the request")]
asks = ["What is the single most useful next action?",
        "Which option is safest if the facts are wrong?",
        "If only one thing is done, what is it?",
        "Which option would a careful manager refuse?",
        "What does the customer most want to hear?"]
for r in range(rounds):
    state = "Case %d. " % (r + 1) + para * (r + 1)
    qs = {}
    for k in range(min(nq, len(asks))):
        qs["q%02d" % (k + 1)] = {"type": "choice", "instructions": asks[k],
                                 "criteria": dict(opts)}
    json.dump({"model": model, "state": state, "temperature": 1.0,
               "questions": qs},
              open(tmp + "/req%d.json" % (r + 1), "w"), ensure_ascii=False)
PY

snap() {
    curl -s --noproxy '*' "http://127.0.0.1:$PORT/v1/metrics" \
         -H "Authorization: Bearer $TOK" \
         | python3 -c '
import sys
fp = 0
for line in sys.stdin:
    if line.startswith("cwick:process_footprint_bytes"):
        fp = int(float(line.split()[1]))
print("%.2f" % (fp / 2**30))'
}

echo "round prompt_tok/q questions input_tok wall_s fp_before_GiB fp_after_GiB delta_GiB q01"
R=1
while [ $R -le "$ROUNDS" ]; do
    S=$(snap)
    W=$(curl -s --noproxy '*' -X POST "http://127.0.0.1:$PORT/v1/systemone" \
        -H "Authorization: Bearer $TOK" -H 'Content-Type: application/json' \
        -d @"$TMP/req$R.json" -o "$TMP/out$R.json" -w '%{time_total}')
    E=$(snap)
    python3 - "$TMP" "$R" "$S" "$E" "$W" <<'PY'
import json, sys
tmp, r, s, e, w = sys.argv[1:6]
try:
    d = json.load(open(tmp + "/out" + r + ".json"))
except Exception:
    d = {}
u = d.get("usage") or {}
a = (d.get("answers") or {}).get("q01") or {}
nq = len(d.get("answers") or {}) or 1
tok = int(u.get("input_tokens") or 0)
# The first request also maps the weights, so its delta is the model's working
# set, not retention. Say so in the output rather than in a footnote.
note = "  <- first touch: includes weight mapping" if r == "1" else ""
print("%s %d %d %d %s %s %s %+.2f %s%s" % (
    r, tok // nq, nq, tok, w, s, e, float(e) - float(s), a.get("choice"), note))
PY
    R=$((R + 1))
done

echo "log: $TMP/serve.log"
