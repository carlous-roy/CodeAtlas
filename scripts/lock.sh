#!/usr/bin/env bash
# Rebuild requirements.lock from pyproject.toml in a fresh virtual environment.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp="$(mktemp -d)"
python3 -m venv "$tmp/venv"
"$tmp/venv/bin/pip" install --quiet --upgrade pip
"$tmp/venv/bin/pip" install --quiet --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.14.0+cpu"
"$tmp/venv/bin/pip" install --quiet -e ".[dev]"
{
  echo "# Exact versions that produced results/. Regenerate with scripts/lock.sh."
  echo "# torch comes from the CPU wheel index so that a CPU-only install stays small."
  echo "--extra-index-url https://download.pytorch.org/whl/cpu"
  "$tmp/venv/bin/pip" freeze --exclude-editable
} > requirements.lock
rm -rf "$tmp"
echo "wrote requirements.lock"
