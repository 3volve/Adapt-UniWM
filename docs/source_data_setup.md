# Source-domain data setup

The thesis pipeline evaluates source retention on four trajectory datasets from
the UniWM data release:

| Adapt-UniWM ID | Original dataset | Use in the thesis pipeline |
| --- | --- | --- |
| `go_stanford` | Go Stanford / GONet navigation data | Source-pre and source-post evaluation |
| `recon` | RECON navigation data | Source-pre and source-post evaluation |
| `sacson` | SACSoN social-navigation data | Source-pre and source-post evaluation |
| `scand` | SCAND social-navigation data | Source-pre and source-post evaluation |

UniWM republishes the converted trajectory archives in
[`fly1113/UniWM_Dataset`](https://huggingface.co/datasets/fly1113/UniWM_Dataset),
whose dataset card labels the consolidated release MIT and credits the original
datasets. Cite the original dataset papers as appropriate in the thesis.

## Automatic setup

From the repository root, run:

```bash
bash download_eval_dataset.sh
```

The script downloads the four archives at the pinned Hugging Face revision
`6585488b03fb5b60be1aba222999fa2a10c4e5b5`, extracts only trajectories named by
`cfg/eval_dataset_manifest.json`, rejects unsafe archive paths, and constructs
the ignored `eval_data/` tree used by `cfg/replay_uniwm_cfg.yaml`.

The complete upstream archives total several gigabytes even though only the
manifest-selected trajectories are retained. Interrupted downloads resume from
`output/dataset_downloads/`.

## Expected layout

```text
eval_data/
├── go_stanford/
│   └── <trajectory_id>/
│       ├── 0.jpg
│       ├── 1.jpg
│       ├── ...
│       └── traj_data.pkl
├── recon/
├── sacson/
├── scand/
└── bootstrap/
```

Each `traj_data.pkl` stores the per-step trajectory metadata; numbered image
files store the corresponding egocentric observations. The manifest fixes the
train, validation, test, and bootstrap trajectory identities used by this
repository.

The Habitat adaptation stage does not use these source archives. It separately
requires HM3D scenes and InstanceImageNav episodes installed as described in
`docs/habitat025_uniwm_setup.md`.
