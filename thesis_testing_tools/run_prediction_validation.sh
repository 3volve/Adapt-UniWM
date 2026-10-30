#!/usr/bin/env bash
# Activate the project's GPU/Habitat environment first; set checkpoint/data paths
# in cfg/prediction_validation_habitat.yaml for this machine.
# Usage: bash thesis_testing_tools/run_prediction_validation.sh [OUTPUT_ROOT] [--run-count N] [tester options...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUTPUT="output/prediction_validation_habitat_$(date +%Y%m%d_%H%M%S)"
RUN_COUNT=1
BASE_CONFIG=cfg/prediction_validation_habitat.yaml
EXTRA_ARGS=()
if (( $# > 0 )) && [[ "$1" != --* ]]; then OUTPUT="$1"; shift; fi
while (( $# > 0 )); do
    case "$1" in
        --run-count|--config|--output)
            if (( $# < 2 )); then echo "Missing value for $1" >&2; exit 1; fi
            case "$1" in
                --run-count) RUN_COUNT="$2" ;;
                --config) BASE_CONFIG="$2" ;;
                --output) OUTPUT="$2" ;;
            esac
            shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done
if [[ ! "$RUN_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "--run-count must be a positive integer" >&2
    exit 1
fi
if [[ -e "$OUTPUT" ]]; then
    echo "Choose a new output root: $OUTPUT already exists" >&2
    exit 1
fi
mkdir -p "$OUTPUT/configs"
export PYTHONUNBUFFERED=1

# Only prediction/training seeds vary; initialization and action generation stay fixed.
for (( INDEX=0; INDEX<RUN_COUNT; INDEX++ )); do
    SEED=$((100 + INDEX))
    CONFIG="$OUTPUT/configs/seed_${SEED}.yaml"
    python - "$BASE_CONFIG" "$SEED" "$CONFIG" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as handle:
    config = yaml.safe_load(handle)
config["wrapper"]["seed"] = int(sys.argv[2])
with open(sys.argv[3], "w") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY
    # pipefail preserves worker failures while tee keeps a live console copy.
    python -u -m thesis_testing_tools.run_prediction_validation \
        --config "$CONFIG" \
        --episode 827 --episode 352 \
        --steps 25 --seed 100 --turn-bins 20 \
        --metrics mae ssim lpips \
        --output "$OUTPUT/seed_${SEED}" "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$OUTPUT/seed_${SEED}.console.log"
done
