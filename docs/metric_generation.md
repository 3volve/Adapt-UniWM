# Reproducible thesis metric generation

Method version **1.1.0** is implemented in
[`thesis_testing_tools/generate_metrics.py`](../thesis_testing_tools/generate_metrics.py).
It reads EventLogger schema-1 **worker** logs and the images they reference. It
does not rerun the world model, change checkpoints, read legacy episode logs, or influence
experiment execution. The pipeline's general run report invokes this generator
automatically; it can also be run independently after downloading or completing
an experiment. Version 1.1.0 adds clipping flags and within-channel spatial
variation diagnostics; image metric formulas and aggregation are unchanged.

## Generate reports

From the repository root:

```bash
python -m thesis_testing_tools.generate_metrics \
  --events output/my_run/habitat/events.jsonl \
  --output output/my_run/analysis_v1
```

Supply several paths after `--events` to process several workers. Results remain
separate by worker and source dataset. Do not supply the coordinator's event file.
The output directory must be new; the tool never overwrites an earlier report.

For a source-retention comparison, provide the shared source-pre worker and one
or more source-post workers. These files are automatically included in analysis:

```bash
python -m thesis_testing_tools.generate_metrics \
  --source-pre output/my_run/source_pre/events.jsonl \
  --source-post output/my_run/c1_fixed_base/source_post/events.jsonl \
  --source-post output/my_run/c5_full/source_post/events.jsonl \
  --output output/my_run/retention_v1
```

A frozen condition that reuses source-pre does not have a new source-post worker;
do not fabricate a second measurement. Its reuse manifest remains the evidence
for that reuse. This tool currently requires explicit worker paths rather than
interpreting pipeline reuse manifests automatically.

Downloaded logs may contain absolute paths from another machine:

```bash
python -m thesis_testing_tools.generate_metrics \
  --events downloaded/run/habitat/events.jsonl \
  --path-map /cluster/project/output/run=downloaded/run \
  --output downloaded/run/analysis_v1
```

Mappings replace complete path prefixes, in supplied order. Windows separators
are normalized. Relative image references without a mapping are resolved against
the worker log's directory. Paths are not guessed from suffixes or route IDs.

Default metrics are `mae ssim lpips`. `--metrics mae` provides a lightweight,
explicit MAE-only report using NumPy and Pillow. SSIM also requires PyTorch and
`pytorch-msssim`; LPIPS requires `lpips` and its pretrained AlexNet weights (the
library may download weights on first use). The tool records installed versions
and the loaded LPIPS state hash. It never silently substitutes a different metric.

## Measurement definitions

Both images are decoded to RGB at their saved resolution, converted to float32
in `[0,1]`, and required to have identical shapes. No resizing is performed.

- **MAE:** arithmetic mean absolute difference over all RGB pixels, using float64
  accumulation. Lower is better. This avoids accumulating the mean in float32 as
  the older pipeline did; tiny rounding differences are therefore possible.
- **SSIM:** `pytorch_msssim.ssim`, data range 1, Gaussian window size 11 and sigma
  1.5, `K=(0.01,0.03)`, `size_average=True`, `nonnegative_ssim=False`. Higher is
  better. Dimensions below 11 are excluded rather than scored with a smaller window.
- **LPIPS:** pretrained AlexNet, LPIPS version 0.1, non-spatial score, images
  transformed to `[-1,1]`. Lower is better. Dimensions below 64 are excluded by
  this method's conservative minimum-size policy. CPU inference uses one Torch
  thread and deterministic algorithms.

Implementation references: [pytorch-msssim's SSIM source](https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py)
and [the LPIPS authors' implementation](https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py).
These links describe the APIs; the report's installed package versions identify
the versions actually used for a measurement.

Habitat uses `wrapper.predicted_obs_path`; source replay uses
`wrapper.evaluation.predicted_obs_path`. Both use `wrapper.real_obs_path` as the
target. Planned and forced-action predictions are not substituted for each other.

Only `transition.outcome == "completed"` is scored. A later checkpoint failure
may mark the overall event failed while its transition remains eligible. No-ops,
unfinished transitions, missing references, missing/unreadable image files,
mismatched shapes and undersized images have explicit exclusion statuses.
Missing measurements are empty CSV cells, never zeros. A collision does not by
itself exclude an otherwise completed and scorable transition.

Episode means weight scored transitions equally. Stage rows report two different
quantities: an equal-weight mean of completed episode means, and a pooled mean of
their scored transitions. Only episodes with `episode_end` evidence enter these
stage means. Incomplete episodes still have diagnostic rows and any available
episode-level scores. No stage mean pools different source datasets or workers.

Source-retention pairing uses `(data_id, episode_id)`. Both episodes must be
complete, with nonempty identical scored step-index sets and matching target-file
SHA-256 hashes. Deltas are post minus pre. Positive MAE/LPIPS deltas and negative
SSIM deltas indicate deterioration. Coverage mismatches and unmatched/incomplete
episodes are listed without a delta. Retention summaries weight eligible episode
deltas equally within each dataset and source-post worker.

## Outputs and provenance

| File | Contents |
| --- | --- |
| `step_metrics.csv` | One row per written attempt, IDs, outcomes, diagnostic evidence, resolved image references, hashes, exclusions and scores. |
| `episode_metrics.csv` | Coverage/outcome counts, episode means, finite diagnostic means and their sample counts, final observed navigation metrics. |
| `stage_metrics.csv` | Per-worker/per-dataset episode and transition means, and completed/scored episode counts. |
| `source_comparison.csv` | Explicit pairing status and per-episode retention deltas. |
| `source_retention.csv` | Mean eligible episode deltas and paired/excluded counts. |
| `metric_definitions.json` | Machine-readable versioned metric and selection definitions. |
| `analysis_manifest.json` | Settings, command, code hash, Git revision, package versions, input log/image hashes, backend settings, exclusions and output hashes. |
| `generate_metrics.py` | Exact source snapshot used to generate the report. |

Final navigation values are from the last written attempt, not an earlier
successful observation substituted after a failure. Optimizer evidence separates
not reached, not applied, uncertain and completed. Primitive collision counts
remain distinct from wrapper-level collision-transition counts.

The report preserves all input logs. Missing terminal run summaries are rejected
unless `--allow-incomplete` is explicitly set and recorded. Malformed/truncated
JSON lines remain errors even with that flag. Unwritten in-memory events cannot
be recovered. These checks define input suitability for analysis; they are not
the deferred schedule/manifest reconciliation system.

To reproduce a report, retain the logs and image artifacts, its source snapshot,
software versions and pretrained weights. Re-run the recorded settings, using
path mappings when relocating files, and compare numeric tables and input hashes.
The manifest records whether the generator has uncommitted changes; the exact
source snapshot and its hash remain authoritative in that case. Dates and
absolute paths in the manifest will differ across machines. Matching
hashes establish the inputs/code; they do not promise bitwise-identical learned
metrics across arbitrary library versions or CPU implementations.

## Citing the implementation

In the thesis methods section, cite the repository commit containing this script,
the path `thesis_testing_tools/generate_metrics.py`, and method version `1.1.0`.
For a particular result table, also identify its `analysis_manifest.json` and
`script_sha256`. This is an implementation citation, not a substitute for citing
the original SSIM and LPIPS methods in the thesis bibliography.

Suggested wording:

> Metrics were regenerated from saved EventLogger v1 worker records and image
> artifacts using Adapt-UniWM's offline metric generator, method version 1.1.0
> (`thesis_testing_tools/generate_metrics.py`, repository revision [commit]).
> Input hashes, preprocessing, exclusion counts and software versions accompany
> each result in its analysis manifest.

Validation uses synthetic image pairs with analytically known MAE, unequal episode
lengths, missing artifacts, moved paths, interrupted logs, late save failures and
paired/unpaired source coverage:

```bash
python -m unittest thesis_testing_tools.test_generate_metrics
```
