# Pipeline reconciliation

`run_thesis_pipeline.py` runs reconciliation automatically after finalizing each
run's execution evidence, on both success and handled execution failure. It prints
the result and records the intended report path in the manifest's references.
Reconciliation failures do not replace the original experiment status/exception.
The report is generated after the execution artifact index and is referenced
directly by the manifest rather than included in that earlier index.

To run it independently (choose a new output name if a report already exists):

```bash
python -m thesis_testing_tools.reconcile_run path/to/seed_run
```

Select the directory containing `run_manifest.json`, normally one seed batch.
The command creates `reconciliation.json`, prints failed/unknown checks, and exits
with **0** for pass, **1** for failed checks, or **2** for unknown evidence without
failed checks. Invalid or unsupported input returns **3**, prints a console note,
and writes no report. CLI usage errors also return 2. It requires PyYAML and the metric
reader's NumPy/Pillow dependencies, but does not initialize Habitat, PyTorch or
image metrics.

For downloaded runs, map the original absolute paths to the local copy:

```bash
python -m thesis_testing_tools.reconcile_run ./downloaded_seed \
  --path-map /cluster/project/results/seed_100=./downloaded_seed
```

Mappings are explicit prefix substitutions, first match wins; repeat the option
for external artifacts. Relative manifest references resolve against the run
directory; relative image references resolve against their worker directory.
Missing artifacts fail their checks; the tool never guesses a substitute file.
Configurations with repository-relative external inputs require mappings for
those inputs as well.

## Checks and evidence

- Planned versus launched workers, completed stages, startup/event structure,
  consecutive event IDs, final worker summary and outcomes. Both seed plans and
  the older pipeline stage plans are supported. C0 source-post reuse is checked
  against its reference and the condition's frozen configuration.
- Selected episode IDs/order per dataset, setup/end coverage, global episode
  indices, consecutive attempt indices, reported attempt/completion totals and
  summary episode count. Seed-batch metadata supplies source selections; older
  runs without that selection evidence report unknown coverage.
- Configured step ceilings and recorded termination reasons. Early source
  exhaustion or an authorized stop is valid. Fixed action sequences, in contrast,
  must match the entire planned sequence and target count.
- No-op attempts, executed primitive identities/counts, permitted early primitive
  termination on collision/environment completion, update eligibility and
  optimizer outcomes. A collision is not itself a failure. An absent training
  section means training was not reached; a null optimizer outcome is unresolved.
- Recorded/replayed schedule transition keys, actions, collision/eligibility/skip
  fields, scalars, base learning rate and actual applied replay learning rate.
  Within-episode shuffling is reproduced from the saved seed and aligned schedule.
- Snapshot hashes, referenced image/checkpoint existence, checkpoint coverage
  when saving is enabled, and source/schedule lifecycle outcomes.

The JSON contains individual `pass`, `fail`, `unknown`, or `not_applicable` checks
with expected/observed values and stage/event/episode locations where applicable.
It fingerprints inspected files and records the checker version, code hash and
path mappings. Checkpoint directories are checked for existence, not loaded or
declared numerically valid. Images are hashed but not decoded or scored here.
This is accounting evidence, not a claim of scientific validity or model quality.

Reconciliation is an offline derived artifact. It never appends to worker logs,
changes the pipeline manifest, repairs data, or affects execution. Keeping the
report separate preserves the original evidence and permits revised checks to be
rerun. Existing reports are protected; use a new output name for a repeat:

```bash
python -m thesis_testing_tools.reconcile_run path/to/seed_run \
  --output path/to/seed_run/reconciliation_review2.json
```

An unwritten in-memory event cannot be reconstructed. Missing summaries/end
records expose that gap; absent optional evidence remains unknown. Missing worker
files or artifacts in an otherwise readable run are reportable failures, whereas
malformed JSON/YAML or unsupported event/manifest structure prevents reporting.
The manifest's intended report reference may therefore have no corresponding file;
the console explains why. The compact HTML debug report remains a separate task.
