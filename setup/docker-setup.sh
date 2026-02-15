#!/usr/bin/env bash
# Run from project root: ./setup/docker-setup.sh
# Creates setup/.env and config.toml in project root if missing, then runs docker compose.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f setup/.env ]]; then
  echo "Creating setup/.env from setup/.env.example"
  cp setup/.env.example setup/.env
  echo "Edit setup/.env and set OPENAI_API_KEY (or LLM_API_KEY), then run again."
fi

if [[ ! -f config.toml ]]; then
  cp setup/config.toml config.toml
fi

exec docker compose -f setup/docker-compose.yaml up --build "$@"
