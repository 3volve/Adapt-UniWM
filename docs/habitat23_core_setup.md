# Habitat 0.2.3 Core Setup

This document captures the reproducible Habitat install baseline derived from the working parts of the `hab23` environment. The target is not PPO training and not full benchmark setup. The target is a minimal, reproducible environment that can later support a custom runner which:

- imports `habitat` and `habitat_sim`
- interacts with `InstanceImageNav`
- sends actions and receives observations from environment steps

## Why This Shape

The original `hab23` environment worked because it used the prebuilt `aihabitat` simulator package instead of compiling `habitat-sim` from source on this workstation. That matters because the source build path was blocked by missing host EGL/OpenGL development libraries.

The broken part of `hab23` was not the simulator. It was a stale editable `habitat-lab` install whose source directory had been moved. This baseline avoids that by making the `habitat-lab` source path explicit and repo-relative by default, with an override for relocated checkouts.

## Pinned Baseline

- Python: `3.8.19`
- CMake: `3.14.0`
- Simulator package: `habitat-sim-challenge-2023=0.2.3`
- Habitat-Lab core source ref: `v0.2.3`
- No `habitat-baselines`
- No datasets or benchmark assets

`InstanceImageNav` is available in the Habitat 0.2.3 generation. Earlier `v0.2.0`, `v0.2.1`, and `v0.2.2` releases are too early for this target.

## Files To Keep Updated

- [envs/habitat23-core.yml](../envs/habitat23-core.yml)
- [scripts/setup_habitat23_core.sh](../scripts/setup_habitat23_core.sh)
- [docs/habitat23_core_setup.md](habitat23_core_setup.md)

If future work changes the Habitat version, required Python packages, or source layout, update these three files together. That is the durable record for reproducing the setup on a new workstation.

## Recommended Install Flow

From the repo root:

```bash
bash scripts/setup_habitat23_core.sh
```

That script will:

1. Create or update the conda environment from the pinned environment file.
2. Use `habitat-lab/` at the repo root by default.
3. Clone Habitat-Lab at `v0.2.3` if the checkout does not already exist.
4. Install only `habitat-lab` core in editable mode.
5. Verify `import habitat` and `import habitat_sim`.

That repo-root `habitat-lab/` checkout is intentionally gitignored in this fork. It is treated as a reproducible local upstream checkout, not vendored project source.

## Relocated Habitat-Lab Source Trees

If the Habitat-Lab checkout lives outside this repo, override the source path:

```bash
HABITAT_LAB_DIR="/path/to/habitat-lab" bash scripts/setup_habitat23_core.sh
```

This is the intended replacement for brittle editable installs tied to old absolute paths such as `/home/evolve/src/habitat-lab/...`.

## Current Local Context

During inspection, the moved Habitat-Lab checkout was found at:

`/media/evolve/b9f43586-ef86-4d01-aff3-b549f34e2aae/Neuromod Pilot/src/habitat-lab`

That checkout also contains local modifications and untracked files, so it is not a clean reproducible baseline by itself. This setup guide deliberately pins an official reproducible baseline instead of encoding those local changes.

## Intentionally Out Of Scope

- `habitat-baselines`
- PPO or DD-PPO training
- benchmark/test asset downloads
- HM3D or InstanceImageNav dataset download steps
- custom runner logic

Those can be layered on top later. When that happens, extend this document and the environment/install files rather than creating a separate undocumented setup path.
