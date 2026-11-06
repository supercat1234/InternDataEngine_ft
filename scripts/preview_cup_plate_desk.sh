#!/usr/bin/env bash
# Scene and camera inspection without planning or dataset recording.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
exec bash "$REPO_DIR/scripts/isaac_python.sh" "$REPO_DIR/scripts/preview_scene.py" "$@"
