# Reusable seed-batch operations

`run_thesis_pipeline.py` keeps `run_seed_batch()` as the sequential coordinator.
Its experiment policy is unchanged: one shared source-pre, C0-C5 in order, each
condition's Habitat episodes in one runner invocation, and source-post after each
learning condition. C0 records source-pre reuse instead of launching source-post.

The reusable functions remain in the same module:

| Function | Responsibility |
| --- | --- |
| `resolve_seed_run(...)` | Read and validate options/manifest, resolve existing episode and step limits, and return a `SeedRun`. No files or model jobs are created. |
| `generate_seed_actions(...)` | Generate the shared standalone action files and snapshot them through the preparation owner's manifest. |
| `prepare_seed_configs(...)` | Write source and condition configurations and construct their existing command descriptions. No model stages run. |
| `prepare_seed_run(...)` | Capture provenance/snapshots, call action/config preparation, save the existing seed manifest, and return a `PreparedSeedRun`. |
| `seed_stage(prepared, stage_id)` | Select a source-pre, condition Habitat, or learning-condition source-post command. C0 source-post is not executable. |
| `execute_seed_stage(stage, ...)` | Run one command and return its outcome/checkpoint references. Exceptions propagate to the caller. |
| `collect_habitat_artifacts(stage)` | Return checkpoint or controller-schedule fingerprint information without modifying shared provenance. |
| `record_habitat_artifacts(...)` | Apply that information to the sequential coordinator's provenance and schedule snapshot. |
| `record_source_pre_reuse(...)` | Write the existing C0 reuse declaration. |
| `write_condition_summary(...)` | Write a condition summary from the supplied execution records. |
| `finalize_seed_run(...)` | Write the completed seed summary from collected condition results. |
| `generate_run_outputs(root)` | Invoke reconciliation and HTML reporting after the execution manifest is closed. |

`SeedRun` and `PreparedSeedRun` are plain data containers. They do not schedule,
execute, serialize themselves, or control lifecycle. Preparation still requires
one owner of the existing `RunManifest`; it does not launch model stages.

Individual stages can be called independently:

```python
stage = seed_stage(prepared, "c3_aligned_replay/habitat")
result = execute_seed_stage(stage)
artifacts = collect_habitat_artifacts(stage)
```

Without `run_manifest`, execution does not open or update a shared manifest. A
future Slurm worker wrapper can persist its result or catch/report its exception.
The sequential coordinator passes `run_manifest=run_record`, retaining the existing
snapshot-backed configuration execution and failure-recording context manager.
An independent caller must supply the intended saved configuration and arrange
prerequisites before executing a stage; these functions do not schedule dependencies.

`finalize_seed_run()` handles the existing completed-batch summary. On failure,
the sequential path still lets `RunManifest` record/close the failed execution,
then invokes `generate_run_outputs()`. A future distributed finalizer must collect
worker outcomes and close its consolidated manifest before calling reporting; this
refactor does not introduce distributed status files or scheduler handling.

The CLI, output layout, manifest fields, fixed ports, current episode-count
defaults, C2 development mean, and checkpoint/schedule policy are retained.
Manifest-driven unlimited selection, Slurm launch settings and the failure-focused
HTML section remain separate changes. Legacy disconnected metric helpers were
not removed. The legacy `run_pipeline()` path retains its flow and shares the
extracted reporting entry point.
