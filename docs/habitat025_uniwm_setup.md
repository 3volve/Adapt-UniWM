# Habitat setup

The adaptation pipeline uses Habitat-Lab and Habitat-Sim `v0.2.5` in the
Python 3.10 `uniwm` environment. Complete the [base setup](../README.md#setup)
first. Run the following commands from the repository root on Linux.

## Install the simulator and task code

```bash
conda activate uniwm
bash scripts/setup_habitat025_uniwm.sh
```

The script installs the [graphics dependencies](../envs/uniwm-habitat025-addons.yml)
and [Python additions](../envs/uniwm-habitat025-pip.txt), checks out both Habitat
repositories at `v0.2.5`, builds Habitat-Sim, and installs Habitat-Lab core in
editable mode. The ignored `habitat-sim/` and `habitat-lab/` checkouts stay at the
repository root. CMake is included in the Habitat Conda additions; a C/C++ build
toolchain must be available on the machine.

The simulator build is headless with Bullet and simulator CUDA support disabled;
the world model still uses CUDA. Build flags and library paths are maintained in
the [setup script](../scripts/setup_habitat025_uniwm.sh), including its conda zlib
and OpenEXR settings. Rerunning it rebuilds the simulator's build directory.

The script finishes by importing both packages and printing their versions and
locations. It also reports the NumPy metadata conflict: the base environment pins
NumPy 1.24.3, while Habitat-Sim 0.2.5 declares NumPy below 1.24. Successful imports
alone do not verify scene loading; perform the check below with your assets.

## Install scenes and episodes

The setup script installs software only. Acquire these assets separately:

- **HM3D-Semantics v0.2 validation scenes**, including semantic assets and scene
  configuration files. Follow the access and download instructions in
  [Habitat-Sim's dataset guide](https://github.com/facebookresearch/habitat-sim/blob/v0.2.5/DATASETS.md).
- **InstanceImageNav v3 episodes**, linked by the
  [Habitat Challenge dataset instructions](https://github.com/facebookresearch/habitat-challenge#objectnavimagenav-baselines-and-dd-ppo-training-starter-code):
  [episode archive](https://dl.fbaipublicfiles.com/habitat/data/datasets/imagenav/hm3d/v3/instance_imagenav_hm3d_v3.zip).
  Keep the validation directory and any companion content files together.

Match the asset locations to `runner.adapter_params` in
[`cfg/habitat_uniwm_cfg.yaml`](../cfg/habitat_uniwm_cfg.yaml):

| Setting | Configured value |
| --- | --- |
| `config_path` | `benchmark/nav/instance_imagenav/instance_imagenav_hm3d_v2.yaml` |
| `split` | `val` |
| `data_path` | `data/datasets/instance_imagenav/hm3d/instance_imagenav_hm3d_v3/val/val.json.gz` |
| `scenes_dir` | `data/scene_datasets` |

`config_path` selects task configuration from the installed Habitat-Lab package.
The episode archive is v3; its location is supplied separately by `data_path`.
Place or link the extracted assets at these locations, or edit the path settings
to match your storage. Run from the repository root so relative paths resolve
consistently. Preserve scene subdirectories referenced by the episode data.

The experiment manifest's `habitat.test` list selects episode IDs from this
validation dataset. It does not select a different Habitat dataset split.

## Verify an episode

This check constructs the actual adapter, loads the manifest's selected episode
IDs, and resets the first episode without loading UniWM model weights:

```bash
python - <<'PY'
import json
from pathlib import Path
import yaml
from source_tools.habitat_source_tools import HabitatEpisodeAdapter

config = yaml.safe_load(Path('cfg/habitat_uniwm_cfg.yaml').read_text())
manifest = json.loads(Path('cfg/eval_dataset_manifest.json').read_text())
params = dict(config['runner']['adapter_params'])
params['episode_ids'] = manifest['habitat']['test']
params['fixed_action_files_dir'] = None
adapter = HabitatEpisodeAdapter(**params)
try:
    adapter.next_episode()
    print('Loaded episode:', adapter.current_episode.episode_id)
    print('Scene:', adapter.current_episode.scene_id)
finally:
    adapter.close()
PY
```

Use the manifest intended for your run in this check. A missing episode ID means
the selected episode dataset does not contain that ID; a missing scene or asset
requires checking the extracted layout and configured paths. This checks the
first selected scene. The [small pipeline run](running_experiments.md#configuration-and-workload)
then exercises action generation, model execution, and reporting together.
