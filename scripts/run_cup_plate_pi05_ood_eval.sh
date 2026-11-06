#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
profile="${OOD_PROFILE:-combined}"
case "$profile" in
  positions|objects|table|floor|combined) ;;
  *) echo "Unknown OOD_PROFILE: $profile (positions|objects|table|floor|combined)" >&2; exit 2 ;;
esac
exec bash scripts/run_cup_plate_pi05_eval.sh \
  --config "local_configs/cup_plate_pi05_ood/${profile}.yaml" "$@"
