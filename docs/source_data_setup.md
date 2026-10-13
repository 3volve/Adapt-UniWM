# Source-domain data setup

Source-pre and source-post evaluate `go_stanford`, `sacson`, `scand`, and `recon`
trajectories from the [UniWM dataset release](https://huggingface.co/datasets/fly1113/UniWM_Dataset).
The Habitat stage uses [separate scenes and episodes](habitat025_uniwm_setup.md).

## Download and prepare

From the repository root, with the `uniwm` environment active and `wget` available:

```bash
bash download_eval_dataset.sh
```

This uses `cfg/eval_dataset_manifest.json` and creates `eval_data/`. To prepare
the held-out selection in a separate directory:

```bash
bash download_eval_dataset.sh cfg/thesis_heldout_manifest.json eval_data_heldout
```

The optional arguments are the experiment manifest and output directory. Relative
arguments resolve from the current working directory. Use the same manifest for
preparation and the pipeline's `--source-manifest`, and set
`runner.adapter_params.data_root` in
[`cfg/replay_uniwm_cfg.yaml`](../cfg/replay_uniwm_cfg.yaml) to the prepared directory.

The script downloads four archives from pinned revision
[`6585488b03fb5b60be1aba222999fa2a10c4e5b5`](https://huggingface.co/datasets/fly1113/UniWM_Dataset/tree/6585488b03fb5b60be1aba222999fa2a10c4e5b5),
extracts selected trajectories into `data/`, and builds the subset. It includes
source `train`, `validation`, and `test` entries plus qualified bootstrap
references. Omitted splits are empty; Habitat and action-vocabulary entries are
excluded from source-data preparation.

Downloads require several gigabytes even when the selected subset is small.
Interrupted archives resume in `output/dataset_downloads/`; each archive is
removed after extraction. Extracted trajectories stay in `data/`.

The output directory must be new. Preparation creates **absolute symlinks** to
trajectories in `data/`, so keep those targets available. Missing trajectories
cause a nonzero exit and can leave a partial output directory.

## Use existing source data

If the archives are already extracted, run the subset builder directly:

```bash
python scripts/create_manifest_dataset.py \
  --manifest cfg/eval_dataset_manifest.json \
  --source-data-dir data \
  --output-data-dir eval_data
```

It searches beneath each source dataset directory for named trajectories
containing `traj_data.pkl`. Add `--copy` for a self-contained subset instead of
symlinks; this requires additional disk space.

## Expected layout and verification

```text
eval_data/
  go_stanford/<trajectory_id>/
    traj_data.pkl
    0.jpg
    1.jpg
    ...
  sacson/<trajectory_id>/
  scand/<trajectory_id>/
  recon/<trajectory_id>/
  bootstrap/<source>/<trajectory_id>/
```

The bootstrap directory is present when selected by the manifest. The builder's
`[OK] All requested episodes were found.` confirms the selected trajectory
directories contain `traj_data.pkl`; it does not validate every image or
trajectory record. Use the [small pipeline workload](running_experiments.md#configuration-and-workload)
after preparing source and Habitat data to check runtime loading.

The pipeline evaluates each source dataset's ordered `test` entries; see
[running experiments](running_experiments.md) for workload selection.
