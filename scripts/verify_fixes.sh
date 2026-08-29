#!/usr/bin/env bash
# Verify the four bug fixes: chunked-id serialization (1), citation resolution (2),
# query-rewrite model (4), and .env loading (8).
#
# Usage:
#   ./scripts/verify_fixes.sh            # everything
#   ./scripts/verify_fixes.sh --no-api   # skip checks that call the LLM provider
#   PORT=8071 ./scripts/verify_fixes.sh  # use a different port for the server check
#
# Never writes to config.toml: the checks that need non-default settings build an
# in-process config instead, so your file is left exactly as it is.

set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
PORT="${PORT:-8071}"
PY="$ROOT/.venv/bin/python"
NO_API=0
[[ "${1:-}" == "--no-api" ]] && NO_API=1

PASS=0
FAIL=0
SKIP=0
SERVER_PID=""
LOG="$(mktemp -t rag-verify-XXXXXX.log)"

cleanup() {
  [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null
  wait "$SERVER_PID" 2>/dev/null
  rm -f "$LOG"
}
trap cleanup EXIT

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$*"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; FAIL=$((FAIL+1)); }
skip() { printf '  \033[33mSKIP\033[0m  %s\n' "$*"; SKIP=$((SKIP+1)); }
note() { printf '        %s\n' "$*"; }

# ---------------------------------------------------------------- preflight
say "Preflight"
[[ -x "$PY" ]] || { echo "  no venv at .venv — run ./setup/setup.sh first"; exit 2; }
ok "virtualenv present"

if [[ -f "$ROOT/.env" || -f "$ROOT/setup/.env" ]]; then
  ok ".env file present"
else
  bad "no .env at project root or setup/ — bug 8 and every API check need one"
fi

MODEL="$("$PY" - <<'EOF'
import sys; sys.path.insert(0, '.')
from src.pipeline.rag import _load_config
print(_load_config()["llm"]["model"])
EOF
)"
note "configured model: $MODEL"

if [[ $NO_API -eq 0 ]]; then
  if "$PY" - <<'EOF'
import os, sys; sys.path.insert(0, '.')
import httpx
from src.config.env import load_env
from src.pipeline.rag import _load_config
load_env()
key = os.getenv("GROQ_API_KEY")
if not key:
    sys.exit("  GROQ_API_KEY not resolvable")
r = httpx.get("https://api.groq.com/openai/v1/models",
              headers={"Authorization": f"Bearer {key}"}, timeout=30)
r.raise_for_status()
ids = {d["id"] for d in r.json()["data"]}
want = _load_config()["llm"]["model"]
sys.exit(0 if want in ids else f"  model '{want}' is not available on this key")
EOF
  then
    ok "configured model is reachable with the resolved key"
  else
    bad "configured model unreachable — the API checks below will fail"
    note "set [llm] model in config.toml to one your key can serve"
  fi
fi

# ------------------------------------------------------- 1. automated suite
say "Automated suite (all four fixes have regression tests)"
if "$PY" -m pytest -q >"$LOG" 2>&1; then
  ok "pytest: $(grep -oE '[0-9]+ passed.*' "$LOG" | tail -1)"
else
  bad "pytest failed"; tail -20 "$LOG" | sed 's/^/        /'
fi

if "$PY" -m pytest -q -k bugfix >"$LOG" 2>&1; then
  ok "bug-fix regressions: $(grep -oE '[0-9]+ passed.*' "$LOG" | tail -1)"
else
  bad "bug-fix regressions failed"; tail -20 "$LOG" | sed 's/^/        /'
fi

# ------------------------------------------------------------- 8. .env load
say "Bug 8 — load_env() reads .env instead of being a no-op stub"
if "$PY" - <<'EOF'
import os, sys, subprocess
sys.path.insert(0, '.')
# A child process with the keys scrubbed from its environment must still find
# them, which is only possible if the .env file is actually read.
code = (
    "import sys; sys.path.insert(0, '.');"
    "from src.config.env import load_env; load_env();"
    "import os; sys.exit(0 if os.getenv('GROQ_API_KEY') else 1)"
)
env = {k: v for k, v in os.environ.items() if k not in ("GROQ_API_KEY", "OPENAI_API_KEY")}
sys.exit(subprocess.call([sys.executable, "-c", code], env=env))
EOF
then
  ok "GROQ_API_KEY resolves from .env with nothing exported"
else
  bad "GROQ_API_KEY did not resolve from .env"
fi

if "$PY" - <<'EOF'
import os, sys, subprocess
sys.path.insert(0, '.')
code = (
    "import sys; sys.path.insert(0, '.');"
    "from src.config.env import load_env; load_env();"
    "import os; print(os.getenv('GROQ_API_KEY'))"
)
env = dict(os.environ, GROQ_API_KEY="exported-must-win")
out = subprocess.run([sys.executable, "-c", code], env=env,
                     capture_output=True, text=True).stdout.strip()
sys.exit(0 if out == "exported-must-win" else 1)
EOF
then
  ok "an exported variable still beats the file (override=False)"
else
  bad "the .env file clobbered an exported variable"
fi

# ------------------------------------------- 2. citation ordinal resolution
say "Bug 2 — [Passage N] resolves by position, not by corpus id"
if "$PY" - <<'EOF'
import sys; sys.path.insert(0, '.')
from src.generation.parser import extract_and_validate_citations
from src.prompts.templates import build_rag_prompt

# Realistic ids: a large PubMed int and a chunked string, neither equal to its ordinal.
ordered = [23179372, 19270706, "9797::1"]
cleaned, cites = extract_and_validate_citations(
    "Alpha [Passage 1]. Beta [Passage 3]. Gamma [Passage 99].", ordered)
assert cites == [("[Passage 1]", 23179372), ("[Passage 3]", "9797::1")], cites
assert "[Passage 99]" not in cleaned, "out-of-range marker survived"
assert "  " not in cleaned, "whitespace scar left behind"

# The enforce flag must reach the prompt, or nothing will ever be cited.
assert "citation" in build_rag_prompt("[Passage 1]\nx", "q?", require_citations=True).lower()
assert "Support every factual claim" not in build_rag_prompt("[Passage 1]\nx", "q?")
EOF
then
  ok "ordinals resolve to real ids; out-of-range stripped; enforce reaches the prompt"
else
  bad "citation resolution is wrong"
fi

# ------------------------------------------------------ 4. rewrite model id
say "Bug 4 — query rewriting sends a real model id, never the literal 'default'"
if "$PY" - <<'EOF'
import sys; sys.path.insert(0, '.')
from unittest.mock import MagicMock
from src.query.rewrite import rewrite_query

client = MagicMock()
client.chat.completions.create.return_value.choices[0].message.content = (
    "Hypothesis: h\nExpansions: a, b")
rewrite_query("What is X?", client, model="llama-3.3-70b-versatile")
sent = client.chat.completions.create.call_args.kwargs["model"]
assert sent == "llama-3.3-70b-versatile", sent

client.reset_mock()
client.chat.completions.create.return_value.choices[0].message.content = (
    "Hypothesis: h\nExpansions: a, b")
rewrite_query("What is X?", client)
assert client.chat.completions.create.call_args.kwargs["model"] != "default"
EOF
then
  ok "the configured model is forwarded; 'default' is never sent"
else
  bad "rewrite is still sending a bad model id"
fi

if [[ $NO_API -eq 0 ]]; then
  if "$PY" - >"$LOG" 2>&1 <<'EOF'
import logging, sys; sys.path.insert(0, '.')
logging.basicConfig(level=logging.WARNING)
import src.pipeline.rag as rag

cfg = rag._load_config()
cfg["rewrite"]["enabled"] = True
cfg["retrieval"]["top_k"] = 3
rag.rag_query_full("What causes glioblastoma?", config=cfg)
EOF
  then
    if grep -q "Query rewriting failed" "$LOG"; then
      bad "HyDE still failing against the live provider"
      grep "Query rewriting failed" "$LOG" | sed 's/^/        /'
    else
      ok "HyDE ran against the live provider with no fallback warning"
    fi
  else
    bad "live rewrite check errored"; tail -8 "$LOG" | sed 's/^/        /'
  fi
else
  skip "live HyDE check (--no-api)"
fi

# ------------------------------------------------ 3/5/6/9/11/12 static checks
say "Bugs 3, 5, 6, 9, 11, 12 — static invariants"
if "$PY" - <<'EOF'
import logging, sys; sys.path.insert(0, '.')
logging.disable(logging.WARNING)
from unittest.mock import MagicMock

# Bug 3: the groundedness judge must see passage text, not just markers.
from src.generation.groundedness import score_groundedness
c = MagicMock()
c.chat.completions.create.return_value.choices[0].message.content = "0.8"
score_groundedness("claim", [{"marker": "[Passage 1]", "passage_id": 7,
                              "text": "Distinctive source sentence."}], c)
prompt = c.chat.completions.create.call_args.kwargs["messages"][0]["content"]
assert "Distinctive source sentence." in prompt, "judge prompt has no passage text"
assert score_groundedness("claim", [], c) == 0.0, "no evidence must score 0"

# Bug 6: payload shape, endpoint and response parsing must agree.
import src.retrieval.rerank as rr
assert rr.OPENROUTER_RERANK_URL.endswith("/chat/completions"), rr.OPENROUTER_RERANK_URL

# Bug 12: a malformed 200 costs the reranking step, not the query.
cl = MagicMock()
cl.post.return_value.raise_for_status.return_value = None
cl.post.return_value.json.return_value = {"unexpected": "shape"}
out = rr.rerank("q", [(1, "a", 0.9), (2, "b", 0.5)], top_n=2, api_key="k", client=cl)
assert [p for p, _, _ in out] == [1, 2], out

# Bug 9: the *_at_k helpers must truncate to k.
from src.eval.metrics import recall_at_k, max_recall_at_k
assert recall_at_k([9, 9, 9, 9, 9, 2], [2], k=5) == 0.0
assert max_recall_at_k(list(range(8)), k=5) == 5 / 8

# Bug 11: no chunk may exceed the threshold or duplicate a sibling.
from src.data.chunking import chunk_passages
text = " ".join(["word"] * 60) + ". " + " ".join(["tail"] * 30) + "."
chunks = chunk_passages([{"id": 1, "text": text}], threshold=20)
assert all(len(c["text"].split()) <= 20 for c in chunks), "oversized chunk"
assert len(chunks) < 15, f"chunk count blew up: {len(chunks)}"

# Context budget must actually be respected.
from src.prompts.formatter import format_passages_as_context
ctx = format_passages_as_context([(1, "A" * 100, .9), (2, "B" * 5900, .8), (3, "C" * 9000, .7)],
                                 max_chars=6000)
assert len(ctx) <= 6000, f"context is {len(ctx)} chars"
EOF
then
  ok "judge sees evidence; rerank endpoint/parse agree; k enforced; chunks bounded"
else
  bad "one or more static invariants broken"
fi

if "$PY" - <<'EOF'
import logging, sys; sys.path.insert(0, '.')
logging.disable(logging.WARNING)  # the mocked throttle logs on every batch
from unittest.mock import MagicMock
import src.retrieval.dense as dense

# Bug 5: a slow opening minute used to disable the guard permanently, leaving a
# later burst unprotected. Per-batch token volume really does vary this much
# across the corpus's passage-size long tail.
clock = [0.0]
sent, throttles, state = [], [], {"n": 0}

def post(*a, **k):
    tokens = 400 if state["n"] < 9 else 2000
    sent.append((clock[0], tokens))
    state["n"] += 1
    clock[0] += 8.0
    r = MagicMock()
    r.raise_for_status.return_value = None
    r.json.return_value = {"data": [{"embedding": [0.1]}] * 10,
                           "usage": {"prompt_tokens": tokens}}
    return r

def sleep(s):
    if s > 2:
        throttles.append(clock[0])
    clock[0] += s

c = MagicMock(); c.post.side_effect = post
dense.time.monotonic = lambda: clock[0]
dense.time.sleep = sleep
dense.embed_texts(["t"] * 300, client=c)

burst_start = sent[9][0]
during_burst = [t for t in throttles if t >= burst_start]
assert during_burst, "guard stayed disabled after the first minute elapsed"
EOF
then
  ok "embedding TPM throttle survives a minute boundary"
else
  bad "TPM throttle is still disabled after the first minute"
fi

# --------------------------------------------- provider errors are not 500s
say "Provider errors map to real status codes, not a bare 500"
if "$PY" - <<'EOF'
import sys; sys.path.insert(0, '.')
import httpx
from openai import APIStatusError
from fastapi.testclient import TestClient
import scripts.run_server as srv

def raiser(status):
    req = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    res = httpx.Response(status, request=req, json={"error": {"message": "x"}})
    err = APIStatusError("boom", response=res, body={"error": {"message": "x"}})
    def _f(_q):
        raise err
    return _f

client = TestClient(srv.app)
for provider_status, expected in ((413, 413), (429, 429), (500, 502)):
    srv.rag_query_full = raiser(provider_status)
    got = client.post("/query", json={"question": "q?"}).status_code
    assert got == expected, f"provider {provider_status} -> {got}, expected {expected}"
EOF
then
  ok "413 -> 413, 429 -> 429, other provider errors -> 502"
else
  bad "provider errors still surface as internal errors"
fi

# --------------------------------------------- 1. chunked ids over real HTTP
say "Bug 1 — /query returns 200 for a question that retrieves a chunked passage"
if [[ $NO_API -eq 1 ]]; then
  if "$PY" - <<'EOF'
import sys; sys.path.insert(0, '.')
from scripts.run_server import PassageOut, CitationOut
assert PassageOut(passage_id="8559285::0", passage="t", score=0.5).passage_id == "8559285::0"
assert PassageOut(passage_id=30485523, passage="t", score=0.5).passage_id == 30485523
assert CitationOut(marker="[Passage 1]", passage_id="8559285::0", text="t", score=0.5)
EOF
  then
    ok "response models accept chunk ids (schema only; --no-api)"
  else
    bad "response models still reject chunk ids"
  fi
  skip "live /query check (--no-api)"
else
  QUESTION="$("$PY" - <<'EOF'
import sys; sys.path.insert(0, '.')
from src.data.loaders import load_corpus
chunked = [x for x in load_corpus() if isinstance(x["id"], str)]
if not chunked:
    sys.exit("no chunked passages in the corpus")
print(" ".join(chunked[0]["passage"].split()[30:45]))
EOF
)"
  if [[ -z "$QUESTION" ]]; then
    bad "could not build a question that hits a chunked passage"
  else
    note "question: ${QUESTION:0:70}..."
    "$PY" -m uvicorn scripts.run_server:app --port "$PORT" --log-level warning \
      >"$LOG" 2>&1 &
    SERVER_PID=$!

    UP=0
    for _ in $(seq 1 60); do
      curl -sf -o /dev/null -m 2 "http://localhost:$PORT/docs" && { UP=1; break; }
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 2
    done

    if [[ $UP -eq 0 ]]; then
      bad "server did not come up on port $PORT"
      tail -15 "$LOG" | sed 's/^/        /'
    else
      ok "server up on port $PORT"
      RESP="$(curl -s -m 180 -w '\n%{http_code}' -X POST \
        "http://localhost:$PORT/query" -H 'Content-Type: application/json' \
        --data "$("$PY" -c 'import json,sys; print(json.dumps({"question": sys.argv[1]}))' "$QUESTION")")"
      CODE="$(printf '%s' "$RESP" | tail -1)"
      BODY="$(printf '%s' "$RESP" | sed '$d')"

      if [[ "$CODE" == "200" ]]; then
        ok "HTTP 200 (this returned 500 before the fix)"
        # The body goes via a file, not a pipe: a heredoc would occupy the
        # interpreter's stdin and the piped JSON would never arrive.
        BODY_FILE="$(mktemp -t rag-verify-body-XXXXXX.json)"
        printf '%s' "$BODY" >"$BODY_FILE"
        if "$PY" -c '
import json, sys
d = json.load(open(sys.argv[1]))
ids = [p["passage_id"] for p in d["passages"]]
print("        passage_ids:", ids)
chunked = [i for i in ids if isinstance(i, str) and "::" in i]
if not chunked:
    sys.exit("        no chunked id in this response - check inconclusive")
print("        chunked id served:", chunked)
' "$BODY_FILE"
        then
          ok "a string chunk id survived JSON serialization"
        else
          bad "no chunked id in the response (inconclusive, not necessarily a regression)"
        fi
        rm -f "$BODY_FILE"
      else
        bad "HTTP $CODE"
        printf '%s' "$BODY" | head -c 300 | sed 's/^/        /'
        grep -E "Error|error" "$LOG" | tail -5 | sed 's/^/        /'
      fi
    fi
  fi
fi

# -------------------------------------------------------------------- summary
say "Summary"
printf '  %d passed, %d failed, %d skipped\n' "$PASS" "$FAIL" "$SKIP"
if [[ $FAIL -eq 0 ]]; then
  printf '  \033[32mall checks green\033[0m\n'
  exit 0
fi
exit 1
