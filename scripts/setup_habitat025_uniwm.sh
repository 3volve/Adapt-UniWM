#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/envs/uniwm-habitat025-addons.yml"
PIP_FILE="${ROOT_DIR}/envs/uniwm-habitat025-pip.txt"
ENV_NAME="${ENV_NAME:-uniwm}"
HABITAT_REF="${HABITAT_REF:-v0.2.5}"
HABITAT_SIM_DIR="${HABITAT_SIM_DIR:-${ROOT_DIR}/habitat-sim}"
HABITAT_LAB_DIR="${HABITAT_LAB_DIR:-${ROOT_DIR}/habitat-lab}"
HABITAT_SIM_BUILD_DIR="${HABITAT_SIM_DIR}/build"
HABITAT_SIM_INSTALL_PREFIX="${HABITAT_SIM_BUILD_DIR}/prefix"

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
  "${ENV_TOOL}" env update --name "${ENV_NAME}" --file "${ENV_FILE}"
else
  "${ENV_TOOL}" env create --name "${ENV_NAME}" --file "${ENV_FILE}"
fi

clone_or_checkout() {
  local repo_url="$1"
  local repo_dir="$2"
  local ref="$3"

  if [ ! -d "${repo_dir}" ]; then
    mkdir -p "$(dirname "${repo_dir}")"
    git clone --branch "${ref}" --depth 1 "${repo_url}" "${repo_dir}"
    return
  fi

  if [ ! -d "${repo_dir}/.git" ]; then
    echo "Expected git checkout at ${repo_dir}" >&2
    exit 1
  fi

  git -C "${repo_dir}" checkout "${ref}"
}

clone_or_checkout "https://github.com/facebookresearch/habitat-sim.git" "${HABITAT_SIM_DIR}" "${HABITAT_REF}"
clone_or_checkout "https://github.com/facebookresearch/habitat-lab.git" "${HABITAT_LAB_DIR}" "${HABITAT_REF}"
git -C "${HABITAT_SIM_DIR}" submodule update --init --recursive

conda activate "${ENV_NAME}"

python -m pip install -r "${PIP_FILE}"

CONDA_PREFIX_VALUE="${CONDA_PREFIX}"
if [ ! -L "${CONDA_PREFIX_VALUE}/lib/libz.so" ] && [ ! -f "${CONDA_PREFIX_VALUE}/lib/libz.so" ]; then
  echo "Expected ${CONDA_PREFIX_VALUE}/lib/libz.so from the zlib package" >&2
  exit 1
fi

rm -rf "${HABITAT_SIM_BUILD_DIR}"

export HEADLESS=True
export WITH_BULLET=False
export WITH_CUDA=False
export CMAKE_ARGS="
-DCMAKE_PREFIX_PATH=${CONDA_PREFIX_VALUE}
-DCMAKE_LIBRARY_PATH=${CONDA_PREFIX_VALUE}/lib
-DCMAKE_INCLUDE_PATH=${CONDA_PREFIX_VALUE}/include
-DEGL_LIBRARY=${CONDA_PREFIX_VALUE}/lib/libEGL.so
-DOpenGL_GL_PREFERENCE=GLVND
-DCMAKE_POLICY_VERSION_MINIMUM=3.5
-DCMAKE_INSTALL_PREFIX=${HABITAT_SIM_INSTALL_PREFIX}
-DOPENEXR_FORCE_INTERNAL_ZLIB=ON
-DZLIB_LIBRARY=${CONDA_PREFIX_VALUE}/lib/libz.so
-DZLIB_INCLUDE_DIR=${CONDA_PREFIX_VALUE}/include
"

(
  cd "${HABITAT_SIM_DIR}"
  python setup.py install --cmake --headless --no-bullet --no-update-submodules
)

python -m pip install --no-deps -e "${HABITAT_LAB_DIR}/habitat-lab"

python - <<'PY'
import inspect
import numpy

import habitat
import habitat_sim

print("habitat", getattr(habitat, "__version__", "unknown"), inspect.getfile(habitat))
print("habitat_sim", getattr(habitat_sim, "__version__", "unknown"), inspect.getfile(habitat_sim))
print("numpy", numpy.__version__)

major, minor, *_ = [int(part) for part in numpy.__version__.split(".")]
if (major, minor) >= (1, 24):
    print("warning: habitat-sim 0.2.5 metadata declares numpy<1.24.0; imports currently succeed in this environment")
PY

cat <<EOF

Environment ready:
  env name: ${ENV_NAME}
  Habitat ref: ${HABITAT_REF}
  Habitat-Sim source: ${HABITAT_SIM_DIR}
  Habitat-Lab source: ${HABITAT_LAB_DIR}

Intentionally omitted:
  - habitat-baselines
  - datasets and benchmark assets
  - GUI viewers
  - Bullet physics
  - CUDA build path

This setup targets a custom non-PPO runner that needs Habitat task code,
headless simulator stepping, actions in, and observations out.
The default source checkouts live at repo root and are gitignored on purpose.

EOF
