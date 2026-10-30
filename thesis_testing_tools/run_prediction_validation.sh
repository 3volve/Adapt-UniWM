#!/usr/bin/env bash
# Activate the project's GPU/Habitat environment first; set checkpoint/data paths
# in cfg/prediction_validation_habitat.yaml and cfg/replay_uniwm_cfg.yaml.
# Usage: bash thesis_testing_tools/run_prediction_validation.sh [OUTPUT_ROOT] [--run-count N] [--probe-steps 0 5 12 24]
# Every seed runs Habitat's four conditions and source's frozen memory-off/on pair.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUTPUT="output/prediction_validation_$(date +%Y%m%d_%H%M%S)"
RUN_COUNT=1
EXTRA_ARGS=()
if (( $# > 0 )) && [[ "$1" != --* ]]; then OUTPUT="$1"; shift; fi
while (( $# > 0 )); do
    case "$1" in
        --run-count|--output)
            if (( $# < 2 )); then echo "Missing value for $1" >&2; exit 1; fi
            case "$1" in
                --run-count) RUN_COUNT="$2" ;;
                --output) OUTPUT="$2" ;;
            esac
            shift 2 ;;
        --steps|--turn-bins)
            if (( $# < 2 )); then echo "Missing value for $1" >&2; exit 1; fi
            EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --probe-steps|--metrics)
            OPTION="$1"
            EXTRA_ARGS+=("$1"); shift
            if (( $# == 0 )) || [[ "$1" == --* ]]; then
                echo "Missing values for $OPTION" >&2; exit 1
            fi
            while (( $# > 0 )) && [[ "$1" != --* ]]; do
                EXTRA_ARGS+=("$1"); shift
            done ;;
        --prepare-only) EXTRA_ARGS+=("$1"); shift ;;
        *) echo "Unsupported launcher option: $1 (domains, episodes and conditions are fixed)" >&2; exit 1 ;;
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
    mkdir -p "$OUTPUT/seed_${SEED}"
    for DOMAIN in habitat source; do
        if [[ "$DOMAIN" == habitat ]]; then
            BASE_CONFIG=cfg/prediction_validation_habitat.yaml
            EPISODES=(--episode 827 --episode 352)
            CONDITIONS=(frozen_off frozen_on adapted_off adapted_on)
        else
            BASE_CONFIG=cfg/replay_uniwm_cfg.yaml
            EPISODES=(
                --episode go_stanford/no7vc_22_1_0
                --episode recon/jackal_2020-01-03-14-24-30_12_r00_0
                --episode scand/random_mdps_A_Spot_AHG_GDC_Tue_Nov_9_41_10_m0-1
            )
            CONDITIONS=(frozen_off frozen_on)
        fi
        CONFIG="$OUTPUT/configs/seed_${SEED}_${DOMAIN}.yaml"
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
            --config "$CONFIG" "${EPISODES[@]}" \
            --conditions "${CONDITIONS[@]}" \
            --steps 25 --seed 100 --turn-bins 20 \
            --metrics mae ssim lpips \
            --output "$OUTPUT/seed_${SEED}/$DOMAIN" "${EXTRA_ARGS[@]}" \
            2>&1 | tee "$OUTPUT/seed_${SEED}/${DOMAIN}.console.log"
    done
done
