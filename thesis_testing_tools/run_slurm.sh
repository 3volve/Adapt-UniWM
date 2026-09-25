#!/usr/bin/env bash
set -euo pipefail

# Run with the thesis environment activated. All paths must be shared by nodes.
cd "$(dirname "$0")/.."
DEV_FIXED_MEAN_LR=6.98e-5

# Set SLURM_PARTITION / SLURM_ACCOUNT for your cluster; unset uses cluster defaults.
cluster_args=()
[[ -z "${SLURM_PARTITION:-}" ]] || cluster_args+=(--partition "$SLURM_PARTITION")
[[ -z "${SLURM_ACCOUNT:-}" ]] || cluster_args+=(--account "$SLURM_ACCOUNT")
[[ -z "${SLURM_FINALIZE_PARTITION:-}" ]] || cluster_args+=(--finalize-partition "$SLURM_FINALIZE_PARTITION")

python -m thesis_testing_tools.run_slurm submit \
    --seed 100 \
    --fixed-mean-lr "$DEV_FIXED_MEAN_LR" \
    --source-cfg replay_uniwm_cfg.yaml \
    --habitat-base-cfg habitat_uniwm_cfg.yaml \
    --source-manifest cfg/eval_dataset_manifest.json \
    --initial-checkpoint checkpoints/base_ckpt \
    --gres gpu:1 --cpus 4 --memory 32G --time 12:00:00 \
    --output-root output/slurm_runs \
    "${cluster_args[@]}" "$@"

# Add --dry-run to inspect the graph without writing files or submitting jobs.
# For debugging, add explicit --source-episodes / --habitat-episodes and step caps.
