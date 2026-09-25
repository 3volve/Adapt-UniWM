# Slurm seed batches

`thesis_testing_tools/run_slurm.py` schedules the same shared operations in
`pipeline.py` that `run_sequential.py` uses. Each invocation submits one seed and
all six C0-C5 conditions. It preserves the existing output names and does not
modify saved runs. The Bash entry points match their Python names:
`run_sequential.sh` and `run_slurm.sh`.

## Launch

Activate the experiment environment on the cluster submission host, then inspect:

```bash
bash thesis_testing_tools/run_slurm.sh --dry-run
```

Dry-run reads configuration and manifest inputs, lists selected episodes and
prints all submission commands. It creates no files, submits no jobs, and loads
no model or simulator. Set cluster resources before submitting, for example:

```bash
export SLURM_PARTITION=gpu
export SLURM_FINALIZE_PARTITION=cpu
export SLURM_ACCOUNT=my_account
bash thesis_testing_tools/run_slurm.sh \
  --source-episodes 2 --habitat-episodes 2 --max-episode-steps 5
```

The shell launcher contains editable experiment defaults. Trailing CLI arguments
override them. Omitted episode limits mean all entries in each dataset's `test`
split, preserving order and unequal dataset sizes. `--source-episodes N` caps
each source dataset individually; `--habitat-episodes N` caps Habitat. No balancing
or repetition is introduced. Sequential execution uses these same selection rules and CLI definitions.

The repository, output directory, Python environment, datasets and checkpoints
must be accessible at identical paths on every node. Use an output directory
inside the repository, as required by the existing action generator. Checkpoints
and datasets remain external inputs and must stay unchanged during the run.
The submission process copies project sources/configurations into `slurm/code`
and inputs into `slurm/inputs`, recording hashes. Jobs execute that code snapshot
with the original repository as their working directory for resource paths.

## Dependencies

There are fourteen jobs:

- Preparation generates shared action artifacts and configurations.
- Source-pre and C0/C1/C2/C5 Habitat depend on preparation succeeding.
- C3/C4 Habitat depend on C0 succeeding, so its schedules are available.
- Each learning condition's source-post depends on its own Habitat job succeeding.
- Finalization waits for every preceding job to terminate, including failures.

Each Habitat job runs its entire episode sequence in one process. Conditions have
independent state; source-pre is shared and can overlap adaptation. Slurm manages
dependencies using `afterok` and `afterany`; the Python submitter does not poll for
completion. See the [sbatch dependency documentation](https://slurm.schedmd.com/sbatch.html).

Preparation is held until all submissions and job IDs are recorded. Failed
submission leaves the held graph and its recorded IDs for manual inspection;
there are no automatic retries, requeues or cancellation of the submitted graph.
Dependent jobs use `--kill-on-invalid-dep=yes` so failed prerequisites can terminate
blocked jobs and allow finalization. Existing stage attempts are never overwritten.

Workers inherit Slurm GPU visibility. Each model stage uses `torchrun --standalone`
for its own rendezvous; no worker forces GPU 0 or shares the sequential fixed port.
See [PyTorch distributed launch](https://docs.pytorch.org/docs/main/elastic/run.html).
Preparation and model stages request one GPU by default; finalization requests no
GPU. Set `--finalize-partition` if your cluster requires a separate CPU partition.

## Evidence and finalization

`submission.json` records job IDs, dependencies, resources and frozen input paths.
Preparation owns the initial `run_manifest.json`; its root `events.jsonl` describes
preparation only. Each model worker owns its usual `events.jsonl` and a small
`stage_status.json` containing execution status, timing, job identity, device
visibility and failure details. Workers never write the shared run manifest.
Slurm stdout/stderr go to `slurm/<stage>-<job-id>.out` and `.err`.

The finalizer reads worker status and `sacct` outcomes, writes the consolidated
manifest and existing summaries, then invokes reconciliation and HTML reporting.
Missing accounting information is recorded explicitly; missing worker results
cannot count as success. Failed or cancelled jobs remain visible in the manifest.
Reconciliation retains its existing behavior for invalid or incomplete inputs.
The dedicated failure snapshot section in HTML is not implemented in this change.

Local tests mock Slurm and model execution. A real cluster smoke run is still
needed to verify site resource policy, environment availability, cancellation
propagation and GPU execution.
