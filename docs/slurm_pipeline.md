# Slurm seed batches

The Slurm launcher submits one C0-C5 seed batch. Condition definitions,
calibration, templates, and workload selection are shared with sequential
execution; see [running experiments](running_experiments.md).

## Prepare and submit

Activate `uniwm` on the submission host and run from the repository root.
The repository, output directory, environment, datasets, and checkpoints must be
accessible at identical paths on every node. Keep the output directory inside
the repository and leave datasets and checkpoints unchanged during execution.

Inspect the selected episodes and submission commands first:

```bash
bash thesis_testing_tools/run_slurm.sh --dry-run
```

Dry-run reads inputs without creating files, submitting jobs, or loading the
model or simulator. Set resource options for your cluster, for example:

```bash
export SLURM_PARTITION=gpu
export SLURM_FINALIZE_PARTITION=cpu
export SLURM_ACCOUNT=my_account
bash thesis_testing_tools/run_slurm.sh \
  --debug-source-episodes 2 --debug-habitat-episodes 2 --max-episode-steps 5
```

Replace the example partition/account names with your site's values. The shell
launcher supplies `--gres gpu:1 --cpus 4 --memory 32G --time 12:00:00`;
trailing arguments override defaults. Finalization requests no GPU and has a
separate `--finalize-time` limit (default `02:00:00`). Use
`--finalize-partition` when it needs a CPU partition. Without that option,
finalization uses the scheduler's default partition.

Submission copies project code into `slurm/code` and configuration/manifest
inputs into `slurm/inputs`, recording hashes. Jobs execute the copied code using
the original repository as their working directory and the submission Python
interpreter. Model workers inherit Slurm GPU visibility and launch one process
with `torchrun --standalone`.

## Job dependencies

The batch contains fourteen jobs:

| Job | Prerequisite |
| --- | --- |
| Preparation | Generates shared actions and stage configurations. |
| Source-pre and C0/C1/C2/C5 Habitat | Successful preparation. |
| C3/C4 Habitat | Successful C0 Habitat, which supplies their schedules. |
| C1-C5 source-post | Successful Habitat job for that condition. |
| Finalization | All preceding jobs have terminated, including failed jobs. |

Each Habitat job executes its entire episode sequence. Source-pre can overlap
adaptation. The submitter uses `afterok` for execution dependencies and `afterany`
for finalization, with `--kill-on-invalid-dep=yes` on dependent jobs.

Preparation stays held until submission records all job IDs. If submission
fails, inspect `submission.json` and the recorded jobs before resubmitting;
already-submitted jobs are not automatically cancelled. Jobs use `--no-requeue`,
and workers reject an existing stage attempt.

## Inspect execution

The submitter prints the seed directory and an `squeue` command for its job IDs.
Use these files within that directory:

| File | Evidence |
| --- | --- |
| `submission.json` | Job IDs, dependencies, resource requests, and frozen input paths. |
| `slurm/<stage>-<job-id>.out` / `.err` | Scheduler stdout/stderr; condition/stage separators become hyphens. |
| `<stage>/stage_status.json` | Worker status, timing, job identity, GPU visibility, and caught failures. |
| `<stage>/events.jsonl` | Model-worker execution evidence. |
| `run_manifest.json` | Preparation evidence followed by the finalizer's consolidated stage outcomes. |

The root `events.jsonl` describes preparation. Each model worker owns its own
records. Finalization reads their status files and `sacct` outcomes, writes the
manifest and summaries, then invokes reconciliation and HTML reporting.
Unavailable accounting is recorded explicitly, and missing worker results cannot
count as successful execution. Inspect [report notices](run_report.md) and
[reconciliation findings](reconciliation.md) alongside scheduler errors.

Automated Slurm tests use mocked scheduling and model execution. Verify resource
policy, environment access, GPU execution, and dependency cancellation with a
small batch on the target cluster before a full experiment.
