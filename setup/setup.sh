#!/usr/bin/env bash
# Run from project root: ./setup/setup.sh
# Copies config to project root and setup/.env from template if missing, then installs deps.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f config.toml ]]; then
  cp setup/config.toml config.toml
fi

if [[ ! -f setup/.env ]]; then
  cp setup/.env.example setup/.env
  echo "Edit setup/.env and set OPENAI_API_KEY (or LLM_API_KEY)."
fi

pip install -r setup/requirements.txt
echo "Done. Run: python scripts/run_server.py"
