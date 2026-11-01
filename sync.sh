#!/usr/bin/env bash
# Build unforget/.venv from local source: 10xgraph and 10xgraph-api come from the
# sibling folders (never PyPI), plus this project itself.
# Usage: ./sync.sh [--clean]
set -euo pipefail

# Run from the script's own directory so it works from anywhere.
cd "$(dirname "${BASH_SOURCE[0]}")"
HERE="$PWD"

usage() {
  echo "Usage: $0 [--clean]" >&2
  echo "  --clean  delete .venv first and rebuild it" >&2
  exit 1
}

CLEAN=0
if [ "$#" -gt 1 ]; then usage; fi
if [ "$#" -eq 1 ]; then
  if [ "$1" = "--clean" ]; then CLEAN=1; else usage; fi
fi

# Resolve local package folders and check prerequisites.
CORE="$(cd "$HERE/../agentflow" 2>/dev/null && pwd)" || { echo "Missing folder: $HERE/../agentflow" >&2; exit 1; }
API="$(cd "$HERE/../agentflow-api" 2>/dev/null && pwd)" || { echo "Missing folder: $HERE/../agentflow-api" >&2; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "uv not found on PATH" >&2; exit 1; }

if [ "$CLEAN" -eq 1 ]; then
  echo "Removing .venv"
  rm -rf .venv
fi

if [ ! -d .venv ]; then
  uv venv --python 3.13 .venv
fi

# One command so uv resolves 10xgraph and 10xgraph-api from the local folders, not PyPI.
# Core extras: anthropic (Claude models) and sqlite_checkpoint (aiosqlite for unforget.db).
# editable_mode=compat writes plain path .pth files instead of setuptools import hooks,
# which Pylance/VS Code cannot follow (imports would show as unresolved).
uv pip install --python .venv/bin/python \
  -C editable_mode=compat \
  --reinstall-package 10xgraph --reinstall-package 10xgraph-api --reinstall-package unforget \
  -e "$CORE[anthropic,sqlite_checkpoint]" -e "$API" -e .

echo
echo "Installed:"
uv pip show --python .venv/bin/python 10xgraph 10xgraph-api unforget \
  | grep -E '^(Name|Version|Location|Editable project location):'
