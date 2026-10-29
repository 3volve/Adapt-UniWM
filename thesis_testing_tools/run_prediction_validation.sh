#!/usr/bin/env bash
# Activate the project's GPU/Habitat environment first; set checkpoint/data paths
# in cfg/prediction_validation_habitat.yaml for this machine.
# Usage: bash thesis_testing_tools/run_prediction_validation.sh [OUTPUT_DIR] [--prepare-only]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUTPUT="${1:-output/prediction_validation_habitat_$(date +%Y%m%d_%H%M%S)}"
if (( $# > 0 )); then shift; fi
LOG="${OUTPUT}.console.log"
if [[ -e "$OUTPUT" || -e "$LOG" ]]; then
    echo "Choose a new output path; refusing to overwrite $OUTPUT or $LOG" >&2
    exit 1
fi
mkdir -p "$(dirname "$LOG")"
export PYTHONUNBUFFERED=1

# pipefail preserves Python/worker failures while tee keeps a live console copy.
python -u -m thesis_testing_tools.run_prediction_validation \
    --config cfg/prediction_validation_habitat.yaml \
    --episode 827 --episode 352 \
    --steps 25 --seed 100 --turn-bins 20 \
    --metrics mae ssim lpips \
    --output "$OUTPUT" "$@" 2>&1 | tee "$LOG"
