#!/usr/bin/env bash
# Smoke test against a running server:  ./scripts/smoke.sh [base_url]
set -euo pipefail
BASE="${1:-http://localhost:8000}"

echo "== GET /health"
curl -fsS "$BASE/health"; echo

echo "== POST /analyze (Korean demo)"
python3 - "$BASE" <<'EOF'
import json, sys, urllib.request
base = sys.argv[1]
text = json.load(urllib.request.urlopen(f"{base}/examples/ko_demo"))["text"]
req = urllib.request.Request(f"{base}/analyze", data=json.dumps({"lang": "ko", "text": text}).encode(),
                             headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req))
for line in r["lines"]:
    spans = ", ".join(f'{s["surface"]}[{s["type"]}|{s["status"]}]' for s in line["spans"]) or "-"
    print(f'{line["line_id"]:>2}  {line["text"]}\n    {spans}')
print("meta:", json.dumps(r["meta"], ensure_ascii=False))
assert len(r["lines"]) == 10 and r["meta"]["n_flagged"] > 0, "unexpected response"
print("OK")
EOF
