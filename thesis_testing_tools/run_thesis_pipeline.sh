#!/usr/bin/env bash

# Run from the repository root with the uniwm environment activated.
# Full-controller development mean from
# thesis_pipeline_20260812_235656/modulated_learning_1e-4, rounded to 3 s.f.
DEV_FIXED_MEAN_LR=6.98e-5

CUDA_VISIBLE_DEVICES=0 python thesis_testing_tools/run_thesis_pipeline.py \
    --all-conditions \
    --source-cfg replay_uniwm_cfg.yaml \
    --habitat-base-cfg habitat_uniwm_cfg.yaml \
    --smoke-test \
    --seed 100 \
    --fixed-mean-lr "$DEV_FIXED_MEAN_LR" \
    --initial-checkpoint checkpoints/base_ckpt \
    --source-manifest cfg/eval_dataset_manifest.json \
    --source-episodes 2 \
    --source-max-episode-steps 10 \
    --habitat-episodes 4 \
    --habitat-max-episode-steps 25 \
    --max-route-steps 5 \
    --schedule-shuffle-seed 20260827 \
    --habitat-port 20001 \
    --output-root output/debug_runs
