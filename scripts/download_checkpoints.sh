#!/usr/bin/env bash
# Wrapper: run the Python downloader from the repository root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$ROOT/scripts/download_checkpoints.py" "$@"
