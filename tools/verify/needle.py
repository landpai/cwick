#!/usr/bin/env python3
"""Long-context retrieval check through the OpenAI-compatible API.

Builds a filler document long enough to push the prompt past the DSA indexer's
index_topk (2048), hides one fact ("the needle") at a known depth, then asks
the model to recall it. If the sparse selection kept the right positions the
fact comes back; if it pruned them away it does not.

This is the functional counterpart to CWICK_INDEXER_TRACE=1, which shows the
selection ratio but not whether the selection is *useful*.

Usage: needle.py PORT [filler_repeats] [max_tokens]
"""
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8311
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 40
BASE = f"http://127.0.0.1:{PORT}/v1"

NEEDLE = ("One specific line in this document records a secret project code: "
          "the code is ZEPHYR-7749. Remember it.")

PARAS = [
    "The maintenance log for the northern pumping station covers the months of "
    "March through June. Each entry records the date, the technician on duty, "
    "the measured pressure in kilopascals and any parts that were replaced.",
    "According to the regional water authority, consumption during the survey "
    "period rose by roughly four percent compared with the same months of the "
    "previous year, which the report attributes mostly to a drier spring.",
    "The archive also holds a set of handwritten field notes that predate the "
    "computerised log by about a decade. They are legible but the ink has "
    "faded unevenly, and several pages are watermarked from a roof leak.",
    "Routine calibration of the flow meters is carried out every eleven weeks. "
    "The procedure requires two people, a certified reference gauge and a "
    "signed checklist that is filed with the operations supervisor.",
]

filler = []
for i in range(REPEATS):
    filler.append(PARAS[i % len(PARAS)])
# Depth ~40%: the needle has to survive a selection that keeps only ~2048 of
# whatever the prompt turns out to be.
cut = len(filler) * 2 // 5
doc = "\n\n".join(filler[:cut] + [NEEDLE] + filler[cut:])

prompt = doc + "\n\nQuestion: what is the secret project code mentioned above?\nAnswer with just the code."

body = {
    "model": "glm-5.3-flash",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": MAXTOK,
    "temperature": 0.0,
}

req = urllib.request.Request(
    BASE + "/chat/completions", data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"})
t0 = time.time()
r = urllib.request.urlopen(req, timeout=None)
resp = json.loads(r.read().decode())
dt = time.time() - t0

msg = resp["choices"][0]["message"]
content = msg.get("content") or ""
reasoning = msg.get("reasoning_content") or ""
text = content + "\n" + reasoning
usage = resp.get("usage", {})
pt = usage.get("prompt_tokens", -1)

print(f"filler paragraphs : {REPEATS}")
print(f"prompt chars      : {len(prompt)}")
print(f"prompt tokens     : {pt}  ({'PAST index_topk' if pt > 2048 else 'INSIDE index_topk (control run)'})")
print(f"finish_reason     : {resp['choices'][0].get('finish_reason')}")
print(f"wall              : {dt/60.0:.1f} min")
print(f"content           : {content.strip()!r}")
print(f"reasoning (tail)  : ...{reasoning.strip()[-160:]!r}")
print("VERDICT:", "PASS -- needle retrieved"
      if "ZEPHYR-7749" in text else "FAIL -- needle lost")
if resp["choices"][0].get("finish_reason") == "length":
    print("NOTE: hit max_tokens before finishing; raise MAXTOK and re-run.")
