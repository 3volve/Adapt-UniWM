# Run reports

Open `run_report.html` in the seed directory to inspect results. Plots, controls,
and selected thumbnails are embedded and work offline; evidence links require
the adjacent files. The pipeline invokes report generation after reconciliation,
including after handled execution failures.

## Read the results

Start with execution status, reconciliation findings, and missing-evidence
notices. A report can be partial even when HTML generation succeeds.

| View | Interpretation |
| --- | --- |
| Prediction results | Per-worker/dataset MAE, SSIM, and LPIPS, using equal-weight completed-episode means. |
| Habitat comparison | Condition minus C0 for matched episodes. Negative MAE/LPIPS or positive SSIM means better than frozen; zero is the reference. Unmatched episodes remain gaps. |
| Raw Habitat scores | Includes available partial-episode scores. Variation in C0 scores reflects episode differences, not parameter updates. |
| Source retention | Post minus pre on matched episodes. Positive MAE/LPIPS or negative SSIM indicates deterioration. C0 reuse is labelled separately. |
| Coverage | Planned/completed episodes, attempts, transitions, no-ops, collisions, and incomplete or failed events. |
| Adaptation diagnostics | Confirmed/uncertain updates, applied learning rates, pre-clipping gradient norms, and clipping flags. C0's unapplied controller scalars are separate. |
| Prediction examples | Deterministically selected typical-MAE, highest-MAE, and lowest-spatial-variation examples per Habitat worker. Categories can share examples. |

Pairing requires completed episodes with identical scored step indices and target
image hashes. See [metric generation](metric_generation.md#measurement-definitions)
for exact scoring, exclusions, and aggregation. Spatial variation measures
within-channel RGB variation; low variation alone does not establish collapse.
Missing measurements stay unavailable. The report does not perform significance
tests or combine seeds into a statistical conclusion.

## Generate or reuse a report

```bash
python -m thesis_testing_tools.generate_run_report path/to/seed_run
```

This creates `run_report.html` and `run_report_metrics/` using the official
metric generator. Defaults are MAE, SSIM, and LPIPS, with the dependencies and
pretrained weights described in the [metric guide](metric_generation.md).
Use `--metrics mae` for an MAE-only report.

If metric calculation fails, the HTML records the reason and retains available
execution diagnostics. Missing or unreadable workers are listed and excluded
from calculation. Inspect notices before using the results.

To reuse existing metric outputs without recalculation:

```bash
python -m thesis_testing_tools.generate_run_report path/to/seed_run \
  --analysis-dir path/to/seed_run/run_report_metrics \
  --output path/to/seed_run/run_report_review2.html
```

Existing HTML and analysis directories are protected. A successful HTML write
returns **0**, including partial reports. Invalid top-level input or an output
write failure returns **1**; CLI usage errors return **2**. The exit code does
not certify the experiment's validity.

For downloaded runs, add repeatable `--path-map OLD=NEW` prefix mappings, such as
`--path-map /cluster/output/seed_run=./seed_run`. External image/source-pre paths
may need additional mappings. Reuse verifies analysis CSV hashes and event-input
hashes/coverage; gallery images must match recorded hashes.

The HTML records report/template and input-manifest hashes. Calculation
provenance remains in `run_report_metrics/analysis_manifest.json`. Report files
are referenced by the run manifest and are produced after its execution artifact
index; see [running experiments](running_experiments.md#outputs-and-evidence)
for refreshing that index.
