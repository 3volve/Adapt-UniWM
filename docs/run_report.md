# General run report

After finalizing execution evidence, the thesis pipeline runs reconciliation and
then generates `run_report.html` and `run_report_metrics/`. This also happens after
execution failure, without replacing the original experiment status or exception.
These derived outputs are referenced by the manifest, after its execution artifact
index was written. Open the HTML directly in a browser: plots, controls and selected
thumbnails are embedded and work offline. Evidence links need the adjacent files.

## Contents and interpretation

- Reconciliation findings and missing-evidence notices.
- Official per-worker/dataset MAE, SSIM, LPIPS and matched source-retention deltas.
  Stage results use completed-episode means. Habitat episode curves can include
  partial episodes; see completion coverage alongside them.
- Completed/planned episodes, attempts, completed transitions, no-ops, collisions,
  unfinished attempts and failed/interrupted step events.
- Confirmed/uncertain updates, applied LR median/range, pre-clipping gradient
  median/range, and clipped/observed flags. Attempt curves mark episode boundaries.
  C0's unapplied controller scalars are separate from applied learning rates.
- Deterministic typical-MAE, highest-MAE and lowest-spatial-variation examples per
  Habitat worker. Variation is mean within-channel spatial standard deviation in
  RGB [0,1], not a collapse diagnosis. Examples may repeat across categories.
- Expandable exclusions, settings, provenance hashes and evidence links.

C0 source-post reuse is labelled, not presented as an independently measured zero
delta. Missing values remain unavailable. There are no significance tests or
multi-seed conclusions. Metric calculations and pairing belong exclusively to
`generate_metrics.py`; the report reads its outputs.

## Commands

```bash
python -m thesis_testing_tools.generate_run_report path/to/seed_run
```

The default requests MAE, SSIM and LPIPS in the thesis environment, including the
official calculator's normal pretrained-weight requirements. Use `--metrics mae`
for an explicitly MAE-only report. If calculation fails, the HTML prominently
states why and retains diagnostics extracted through the existing generator's
functions. It never silently substitutes another image metric.

To reuse existing official metrics without recalculation:

```bash
python -m thesis_testing_tools.generate_run_report path/to/seed_run \
  --analysis-dir path/to/seed_run/run_report_metrics \
  --output path/to/seed_run/run_report_review2.html
```

Existing HTML/analysis outputs are not overwritten. A successful HTML write
returns 0, including partial reports; inspect notices and reconciliation for
experiment validity. Invalid top-level input or output-write failure returns 1.
Missing/unreadable workers are listed and excluded from metric generation.

For downloaded runs, add repeatable `--path-map OLD=NEW` prefix mappings, e.g.
`--path-map /cluster/output/seed_run=./seed_run`. External source-pre/image paths
may need separate mappings. Analysis CSV hashes and event-input hashes/coverage
are verified before reuse. Gallery files must match the recorded image hashes.
The HTML includes report/template and input-manifest hashes; the metric analysis
manifest retains its full calculation provenance.

Implementation: one Python module for evidence loading, summaries and selection,
and one HTML template with native SVG/JavaScript controls. No web server, external
chart library or separate debug metric implementation is used.
