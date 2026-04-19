#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/envs/habitat23-core.yml"
ENV_NAME="${ENV_NAME:-hab23-core}"
HABITAT_REF="${HABITAT_REF:-v0.2.3}"
HABITAT_LAB_DIR="${HABITAT_LAB_DIR:-${ROOT_DIR}/habitat-lab}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found on PATH" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

ENV_TOOL="conda"
if command -v mamba >/dev/null 2>&1; then
  ENV_TOOL="mamba"
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  "${ENV_TOOL}" env update --name "${ENV_NAME}" --file "${ENV_FILE}" --prune
else
  "${ENV_TOOL}" env create --name "${ENV_NAME}" --file "${ENV_FILE}"
fi

if [ ! -d "${HABITAT_LAB_DIR}" ]; then
  mkdir -p "$(dirname "${HABITAT_LAB_DIR}")"
  git clone --branch "${HABITAT_REF}" --depth 1 https://github.com/facebookresearch/habitat-lab.git "${HABITAT_LAB_DIR}"
fi

if [ ! -f "${HABITAT_LAB_DIR}/habitat-lab/setup.py" ]; then
  echo "Expected Habitat-Lab core package at ${HABITAT_LAB_DIR}/habitat-lab/setup.py" >&2
  echo "Override HABITAT_LAB_DIR if your checkout lives elsewhere." >&2
  exit 1
fi

conda activate "${ENV_NAME}"
python -m pip install --editable "${HABITAT_LAB_DIR}/habitat-lab"

python - <<'PY'
import inspect

import habitat
import habitat_sim

print("habitat", getattr(habitat, "__version__", "unknown"), inspect.getfile(habitat))
print("habitat_sim", getattr(habitat_sim, "__version__", "unknown"), inspect.getfile(habitat_sim))
PY

cat <<EOF

Environment ready:
  env name: ${ENV_NAME}
  Habitat-Lab source: ${HABITAT_LAB_DIR}
  Habitat-Lab ref target: ${HABITAT_REF}

Intentionally omitted:
  - habitat-baselines
  - datasets and benchmark assets
  - PPO training setup

This baseline is intended for custom non-PPO runners that need to:
  - import habitat and habitat_sim
  - load InstanceImageNav-compatible task code later
  - send actions and receive observations after env stepping
The default Habitat-Lab checkout lives at repo root and is gitignored on purpose.

EOF
