#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Supply a prepared stage config containing the exact selected workload.
CFG_PATH="${1:?Provide the prepared stage YAML path}"
shift
python -m torch.distributed.run --standalone --nproc-per-node=1 uniwm_episode_runner.py \
  --config_path "$CFG_PATH" --data_id habitat "$@"
