#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Run from the repository root with the uniwm environment activated.
# Full-controller development mean from
# thesis_pipeline_20260812_235656/modulated_learning_1e-4, rounded to 3 s.f.
DEV_FIXED_MEAN_LR=6.98e-5

python -m thesis_testing_tools.run_sequential \
    --source-cfg replay_uniwm_cfg.yaml \
    --habitat-base-cfg habitat_uniwm_cfg.yaml \
    --seed 100 \
    --fixed-mean-lr "$DEV_FIXED_MEAN_LR" \
    --initial-checkpoint checkpoints/base_ckpt \
    --source-manifest cfg/eval_dataset_manifest.json \
    --schedule-shuffle-seed 20260827 \
    --output-root output/sequential_runs \
    "$@"

# Add explicit episode/step caps and --smoke-test for a smaller debug run.
