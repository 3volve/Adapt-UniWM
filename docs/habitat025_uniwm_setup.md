# Habitat 0.2.5 In `uniwm`

This document captures the working Habitat setup inside the existing `uniwm` conda environment. The target is not PPO training and not a full benchmark runtime. The target is the lightest setup that can later support a custom runner which:

- imports `habitat` and `habitat_sim`
- reaches `InstanceImageNav` task code in Habitat-Lab
- sends actions into the environment and receives observations back after stepping

## Pinned Baseline

- Python: `3.10`
- Habitat-Sim source ref: `v0.2.5`
- Habitat-Lab source ref: `v0.2.5`
- Build mode: headless source build
- GUI viewers: disabled
- Bullet: disabled
- CUDA: disabled
- No `habitat-baselines`
- No datasets, benchmark assets, or test scenes

`InstanceImageNav` is present in Habitat-Lab `v0.2.5`, and this version is a better fit for Python 3.10 than the old `0.2.3` prebuilt-package path.

## Why Source Build

The `aihabitat` prebuilt simulator packages for the older Habitat 0.2.x line do not cover the Python 3.10 shape of `uniwm`. For this workstation, the stable path is:

- install the Linux graphics and build prerequisites into the conda env
- build `habitat-sim` from source
- install `habitat-lab` core only in editable mode

That keeps the setup lightweight and repo-trackable while staying compatible with the current UniWM Python version.

## Reproducible Files

- [envs/uniwm-habitat025-addons.yml](../envs/uniwm-habitat025-addons.yml)
- [envs/uniwm-habitat025-pip.txt](../envs/uniwm-habitat025-pip.txt)
- [scripts/setup_habitat025_uniwm.sh](../scripts/setup_habitat025_uniwm.sh)
- [docs/habitat025_uniwm_setup.md](habitat025_uniwm_setup.md)

If we later add task assets, a custom runner, or stricter dependency pins, update these files together.

## Install Flow

From the repo root:

```bash
bash scripts/setup_habitat025_uniwm.sh
```

That script will:

1. Update the existing `uniwm` env with the required conda-side graphics and build packages.
2. Install the small Python dependency delta needed by Habitat-Lab core.
3. Clone or checkout `habitat-sim` and `habitat-lab` at `v0.2.5` into repo-root local source directories.
4. Sync Habitat-Sim submodules.
5. Build Habitat-Sim from source in headless mode with Bullet and CUDA disabled.
6. Install Habitat-Lab core only with `--no-deps`.
7. Verify `import habitat_sim` and `import habitat`.

Those repo-root `habitat-sim/` and `habitat-lab/` directories are intentionally gitignored in this fork. They are treated as reproducible local upstream checkouts, not vendored project contents.

## Exact Build Shape

The working Habitat-Sim source install used:

- `python setup.py install --cmake --headless --no-bullet --no-update-submodules`
- `HEADLESS=True`
- `WITH_BULLET=False`
- `WITH_CUDA=False`
- explicit CMake hints into `${CONDA_PREFIX}/include` and `${CONDA_PREFIX}/lib`
- `OPENEXR_FORCE_INTERNAL_ZLIB=ON`
- `CMAKE_POLICY_VERSION_MINIMUM=3.5`

The `OPENEXR_FORCE_INTERNAL_ZLIB=ON` and `zlib` conda package matter on this workstation. Without them, the build can fall into a bad install path for zlib during the vendored OpenEXR step.

## Current Conflict Note

One dependency conflict remains at the metadata level:

- `habitat-sim==0.2.5` declares `numpy<1.24.0`
- `uniwm` currently has `numpy==1.24.3`

Current status on this machine:

- `import habitat_sim` succeeds
- `import habitat` succeeds
- the install completed without downgrading UniWM's NumPy

Assessment:

- likely resolvable by pinning NumPy to `1.23.5` if a runtime issue appears later
- not changed yet to avoid unnecessary churn in the existing UniWM environment
- nightly Habitat-Sim is not the preferred fallback right now because the pinned `0.2.5` source build already works and is more reproducible against Habitat-Lab `0.2.5`

## Verified Local Result

On April 17, 2026 in the local `uniwm` environment:

- `import habitat_sim` succeeded from the conda env site-packages install
- `import habitat` succeeded from the editable repo checkout
- both reported version `0.2.5`

## Historical Note

The older Habitat 0.2.3 material remains in this repo as a historical guide from the separate `hab23` environment:

- [docs/habitat23_core_setup.md](habitat23_core_setup.md)
- [envs/habitat23-core.yml](../envs/habitat23-core.yml)
- [scripts/setup_habitat23_core.sh](../scripts/setup_habitat23_core.sh)

That is no longer the primary target for UniWM.

## Intentionally Out Of Scope

- `habitat-baselines`
- PPO or DD-PPO training
- benchmark and test asset downloads
- InstanceImageNav datasets and scenes
- middleware code
- custom runner implementation

Those should be layered on top later, and when they are, this document and the setup script should be extended instead of creating a separate undocumented install path.
