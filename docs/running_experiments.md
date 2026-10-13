# Running experiments

The pipeline runs one C0-C5 batch per seed. Complete the
[repository setup](../README.md#setup) before launching it. Run the commands below
from the repository root with the `uniwm` environment active.

## Experiment protocol

Preparation generates one seeded action sequence per selected Habitat episode.
All six conditions use the same episodes, actions, and initial checkpoint.

| Condition / directory | Habitat learning behavior |
| --- | --- |
| C0 `c0_frozen` | No parameter updates; records the controller schedule and a within-episode shuffled copy. |
| C1 `c1_fixed_base` | Fixed learning rate from the Habitat template's `initial_lr`. |
| C2 `c2_fixed_mean` | Fixed learning rate supplied with `--fixed-mean-lr`. |
| C3 `c3_aligned_replay` | Replays C0's controller schedule in its original order. |
| C4 `c4_shuffled_replay` | Replays C0's within-episode shuffled schedule. |
| C5 `c5_full` | Online updates controlled by the live learning controller. |

The sequential pipeline evaluates **source-pre** once, then executes C0-C5 in
order. Each learning condition runs its entire **Habitat** sequence, saves a final
adapter, and evaluates that adapter in **source-post** on the same source
trajectories. C0 reuses the source-pre result for source-post. Source evaluation
covers `go_stanford`, `sacson`, `scand`, and `recon`.

## Launch and calibration

The editable launcher provides the experiment defaults:

```bash
bash thesis_testing_tools/run_sequential.sh
```

It selects seed 100, `cfg/eval_dataset_manifest.json`,
`checkpoints/base_ckpt`, and a C2 rate of `6.98e-5`. That rate is the rounded
Full-controller development mean identified in the launcher's comment. Trailing
arguments override launcher defaults, for example:

```bash
bash thesis_testing_tools/run_sequential.sh \
  --seed 101 \
  --source-manifest cfg/thesis_heldout_manifest.json
```

Despite its name, `--source-manifest` selects both source trajectories and Habitat
episode IDs. Choose the manifest for the intended development or held-out
workload, and ensure its trajectories and scenes are available locally.

C2's rate must be calibrated from valid Full-controller update opportunities on
development data before running the comparison. When using your own calibrated
rate, pass `--fixed-mean-lr VALUE`. To preserve its supporting record, also pass
`--fixed-mean-calibration path/to/calibration.json` (a CSV or another single file
is also accepted). The pipeline copies and hashes that record; it uses the rate
you supply without deriving it from the evaluation stream.

For cluster execution, follow the [Slurm guide](slurm_pipeline.md). Its launcher
accepts the same experiment options and supports `--dry-run` to inspect the job
graph without submitting work.

## Configuration and workload

| Option | Controls |
| --- | --- |
| `--source-cfg` | Source-pre/post template; defaults to `replay_uniwm_cfg.yaml`. |
| `--habitat-base-cfg` | Shared Habitat template; defaults to `habitat_uniwm_cfg.yaml`. |
| `--initial-checkpoint` | Common starting adapter; defaults to `checkpoints/base_ckpt`. |
| `--source-manifest` | Ordered `test` entries for each source dataset and Habitat. |
| `--schedule-shuffle-seed` | C0's within-episode schedule shuffle; defaults to `20260827`. |
| `--output-root` | Parent of the timestamped seed directory; keep it inside the repository. |

Template filenames resolve under `cfg/`; repository-relative and absolute
configuration paths also work. The pipeline saves the templates and generated
stage configurations. It sets condition-specific training, controller and
schedule behavior, C2's rate, selected episodes, checkpoint paths, and requested
step limits. Other template settings carry through. Edit the shared templates
for changes to the model, controller parameters, or environment settings.

By default, every `test` entry in the selected manifest is used, preserving its
order. `--debug-source-episodes N` takes the first N entries **per source dataset**;
`--debug-habitat-episodes N` takes the first N Habitat entries. Smaller selections
run in full.

Per-episode step limits resolve in this order: stage-specific
`--source-max-episode-steps` / `--habitat-max-episode-steps`, shared
`--max-episode-steps`, then the respective template value. `--max-route-steps`
overrides the wrapper's route limit in both stages. The runner's
`max_total_episodes` configuration is an additional safety ceiling; ensure it
accommodates the selected workload, including all source datasets in one worker.

For a small execution check:

```bash
bash thesis_testing_tools/run_sequential.sh \
  --debug-source-episodes 2 \
  --debug-habitat-episodes 2 \
  --max-episode-steps 5 \
  --smoke-test
```

This still runs all six conditions and loads the model. `--smoke-test` labels the
run; the explicit limits determine its size.

`--seed` is passed to every model worker to seed Python, NumPy, and PyTorch before
model initialization. Training uses dropout, and strict deterministic GPU
algorithms are not forced. Use the recorded settings and evidence when comparing
runs. Use Slurm for concurrent batches; sequential workers use fixed rendezvous
ports.

## Outputs and evidence

The sequential shell launcher writes to
`output/sequential_runs/thesis_seed_<seed>_<timestamp>/`; the Slurm launcher uses
`output/slurm_runs/`. Within a seed directory:

| Path | Purpose |
| --- | --- |
| `run_manifest.json` | Entry point for run settings, snapshots, stage status, timing, and evidence references. |
| `seed_manifest.json` / `seed_summary.json` | Planned condition queue and completed-batch summary. |
| `source_pre/` | Shared source evaluation. |
| `<condition>/habitat/` | Adaptation events, images, schedules for C0, and `final_ckpt/` for C1-C5. |
| `<condition>/source_post/` | Source-retention evaluation, or C0's reuse declaration. |
| `habitat_action_sequences/` | Shared generated actions. |
| `provenance/` / `provenance.json` | Input/code snapshots, environment records, artifact index, and checkpoint fingerprints. |
| `reconciliation.json` | Workload and execution checks. |
| `run_report.html` / `run_report_metrics/` | Offline result report and its metric tables and provenance. |

Each model worker writes `events.jsonl`, including startup metadata, attempted
transitions, update outcomes, and a terminal summary. The coordinator has its own
event file. Snapshots preserve the inputs and implementation used; external data
and full model downloads must be retained separately.

After execution evidence is finalized, the pipeline invokes reconciliation and
report generation, including after handled execution failures. Inspect console
errors if a derived report is missing, and read report notices if it is partial.
A completed execution alone does not establish valid coverage or measurements.
Abrupt process or node loss can leave incomplete evidence.

Use the [run report guide](run_report.md) to regenerate or inspect results,
[reconciliation guide](reconciliation.md) for execution checks, and
[metric guide](metric_generation.md) for measurement definitions and downloaded
path mappings. After adding analysis artifacts inside a run, refresh its index:

```bash
python -m thesis_testing_tools.run_manifest output/sequential_runs/thesis_seed_100_TIMESTAMP
```

Replace the example path with the actual seed directory.
