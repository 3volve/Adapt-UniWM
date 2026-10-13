# Pipeline reconciliation

Reconciliation compares a seed batch's planned workload with its recorded
execution. Sequential execution and Slurm finalization invoke it before HTML
reporting, including after handled execution failures. Its result is separate
from the experiment's execution status.

## Run and interpret

Select the seed directory containing `run_manifest.json`:

```bash
python -m thesis_testing_tools.reconcile_run path/to/seed_run
```

The command creates `reconciliation.json` and prints failed or unknown checks.
It requires PyYAML, NumPy, and Pillow; it does not load the world model or simulator.

| Exit code | Meaning |
| --- | --- |
| 0 | All applicable checks passed. |
| 1 | At least one check failed. |
| 2 | Evidence is unknown with no failed checks, or CLI usage is invalid. Read the console output to distinguish these cases. |
| 3 | Invalid or unsupported input prevented reporting; no report was written. |

Each check includes its status (`pass`, `fail`, `unknown`, or `not_applicable`),
expected/observed evidence, and stage/event/episode location where available.
The report records inspected-file fingerprints, checker version, implementation
hashes, and path mappings.

Existing reports are protected. For another check, choose a new output path:

```bash
python -m thesis_testing_tools.reconcile_run path/to/seed_run \
  --output path/to/seed_run/reconciliation_review2.json
```

## What is checked

- Planned workers and stage completion; event structure, consecutive IDs,
  terminal summaries, and C0 source-pre reuse with training disabled.
- Selected episode identity/order, setup/end coverage, attempt indices, and
  reported episode/attempt/completion totals.
- Step ceilings and termination reasons. Source exhaustion and authorized stops
  are valid; fixed-action execution must match its planned sequence and count.
- No-ops, executed primitives, permitted early termination on collision or
  environment completion, update eligibility, and optimizer outcomes.
- Recorded/replayed schedule keys, actions, eligibility, skip reasons, scalars,
  base rate, and applied replay rate. Shuffling is reproduced from the saved seed.
- Snapshot hashes, image/checkpoint existence, final-checkpoint count and path,
  and source/schedule lifecycle outcomes.
- Planned checkpoint versus configured and startup-recorded adapter/processor
  paths and initialization mode.

A collision alone is not a failure. An absent training section means training
was not reached; a null optimizer outcome remains unresolved. Missing selection
or loading evidence is reported as unknown. Checkpoint checks compare recorded
paths and directory existence, without inspecting in-memory weights. Images are
hashed here; [metric generation](metric_generation.md) decodes and scores them.

## Downloaded runs and missing evidence

Map saved paths to the local copy explicitly:

```bash
python -m thesis_testing_tools.reconcile_run ./downloaded_seed \
  --path-map /cluster/project/results/seed_100=./downloaded_seed
```

Mappings replace prefixes in supplied order; first match wins. Repeat the option
for external artifacts. Relative manifest references resolve against the seed
directory; relative image references resolve against their worker directory.
Repository-relative external inputs may need additional mappings.

Missing worker files or artifacts in a readable run are reportable failures.
Malformed JSON/YAML or unsupported structures prevent reporting. An unwritten
in-memory event cannot be recovered; missing summaries or episode ends expose
incomplete evidence. A manifest's report reference can therefore point to an
absent file; inspect the console for the reason.

Reconciliation preserves worker logs and execution manifests. Its report is
referenced separately because it is generated after the execution artifact
index. The [HTML report](run_report.md) displays available findings. Passing
these checks establishes execution consistency, not scientific validity or
model quality.
