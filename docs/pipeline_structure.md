# Pipeline structure

For commands, conditions, and workload settings, see
[running experiments](running_experiments.md). This guide maps the code used to
prepare, execute, and record a seed batch.

## Responsibilities

| Component | Responsibility |
| --- | --- |
| [`pipeline.py`](../thesis_testing_tools/pipeline.py) | Shared experiment arguments, workload selection, action generation, stage configurations, commands, and summaries. |
| [`run_sequential.py`](../thesis_testing_tools/run_sequential.py) | Sequential stage ordering and execution-manifest ownership. |
| [`run_slurm.py`](../thesis_testing_tools/run_slurm.py) | Submission, dependencies, worker status, and consolidated finalization. |
| [`uniwm_episode_runner.py`](../uniwm_episode_runner.py) | Execute a selected workload, advance events, and save final model state. |
| [Source adapters](../source_tools/) | Load and traverse the selected episodes; translate source-specific actions and observations. |
| [`run_manifest.py`](../thesis_testing_tools/run_manifest.py) | Capture input/code snapshots, provenance, and execution lifecycle. |
| [`event_logger.py`](../runtime_scripts/event_logger.py) | Collect component evidence into each process's event file. |

## Preparation and execution

`resolve_seed_run()` validates options and episode selections and returns a
`SeedRun` without writing files or launching jobs. `prepare_seed_run()` captures
inputs, generates shared actions, writes stage configurations and the seed plan,
and returns a `PreparedSeedRun`. These objects hold data; the coordinator controls
execution.

`seed_stage()` selects a prepared command and `execute_seed_stage()` runs it.
Dependencies are the coordinator's responsibility. Sequential execution supplies
a `RunManifest` to record each stage. Slurm workers execute saved configurations
and write their own status files; they do not update the shared manifest.
Checkpoint and schedule fingerprints are collected for coordinator finalization.

`finalize_seed_run()` writes the batch summary. After execution evidence closes,
`generate_run_outputs()` invokes reconciliation and reporting. The sequential
coordinator also takes this reporting path after handled execution failures;
the Slurm finalizer assembles worker and scheduler outcomes first. See
[Slurm execution](slurm_pipeline.md) for its record layout.

## Episode traversal

Preparation checks unique episode IDs within each dataset and applies requested
debug caps. It snapshots the original manifest and a selected source manifest;
source-pre/post use the selected snapshot, while Habitat configurations contain
the ordered episode IDs. Replay data paths are resolved to absolute paths.

Adapters implement `next_episode()`, returning initial observations with dataset
and episode identity, or `None` at exhaustion. Replay advances between source
datasets internally. Habitat selects each episode before resetting the simulator.
Loading errors propagate to the runner.

The runner executes episodes until exhaustion. `runner.max_total_episodes` is a
safety ceiling: another episode beyond that count raises an error. Per-episode
step limits control individual episodes. When checkpoint saving is enabled, the
runner saves the final checkpoint after workload exhaustion. Reconciliation
compares execution evidence with the saved selection.

For direct stage execution, the `thesis_testing_tools/eval_online_*.sh` helpers
accept a prepared stage YAML as their first argument. Start from the preparation
functions above when arranging a custom execution flow.
