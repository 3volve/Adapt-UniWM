"""Build an offline, single-file run report using the official metric generator."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
from datetime import datetime, timezone
import html
import io
import json
import math
import os
from pathlib import Path
import statistics
from urllib.parse import quote

from PIL import Image, UnidentifiedImageError

from thesis_testing_tools import generate_metrics as gm


def number(value):
    if value in (None, ""):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def truth(value):
    return value is True or value == "True"


def display(value):
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def table(headers, rows):
    escape = lambda x: html.escape(display(x))
    return '<div class="table-scroll"><table><thead><tr>' + ''.join(f'<th>{escape(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{escape(cell)}</td>' for cell in row) + '</tr>' for row in rows
    ) + '</tbody></table></div>'


def range_text(values):
    values = [n for v in values if (n := number(v)) is not None]
    return f"{display(statistics.median(values))} [{display(min(values))}, {display(max(values))}]" if values else "—"


def load_tables(directory):
    manifest = json.loads((directory / "analysis_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["outputs"]:
        path = directory / item["path"]
        if gm.sha256(path.read_bytes()) != item["sha256"]:
            raise ValueError(f"Analysis output hash mismatch: {path}")
    tables = {}
    for name in ("step_metrics", "episode_metrics", "stage_metrics", "source_retention"):
        with (directory / f"{name}.csv").open(newline="", encoding="utf-8") as stream:
            tables[name] = list(csv.DictReader(stream))
    return tables, manifest


def thumbnail(path, expected_hash):
    raw = path.read_bytes()
    if gm.sha256(raw) != expected_hash:
        raise ValueError(f"Image changed since metric generation: {path}")
    with Image.open(io.BytesIO(raw)) as image:
        image = image.convert("RGB")
        image.thumbnail((240, 240))
        output = io.BytesIO()
        image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def build_report(root, *, output=None, analysis_dir=None, metrics=("mae", "ssim", "lpips"), path_maps=()):
    root = Path(root).resolve()
    output = Path(output).resolve() if output else root / "run_report.html"
    if output.exists():
        raise FileExistsError(f"Report already exists: {output}; choose a new --output")
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("stages"), list):
        raise ValueError("Expected a version-1 pipeline run_manifest.json with stages")
    resolve = lambda value, base=root: gm.resolve_image(value, base / "_anchor", path_maps)
    notices, workers, records_by_path, fingerprints = [], [], {}, []
    def read_worker(path):
        if str(path) in records_by_path:
            return
        try:
            records, evidence = gm.read_events(path, allow_incomplete=True)
            records_by_path[str(path)] = records
            fingerprints.append(evidence)
        except (OSError, ValueError) as error:
            notices.append(f"Worker unavailable: {error}")
    for stage in manifest["stages"]:
        path = resolve(stage["events"])
        workers.append({**stage, "path": str(path)})
        read_worker(path)
    source_pre = resolve(manifest.get("references", {}).get("source_pre"))
    pre_path = source_pre / "events.jsonl" if source_pre else None
    if pre_path is not None:
        read_worker(pre_path)
    valid_pre = pre_path if pre_path is not None and str(pre_path) in records_by_path else None
    posts = [Path(w["path"]) for w in workers if w["name"] == "source_post" and w["path"] in records_by_path]
    analysis = Path(analysis_dir).resolve() if analysis_dir else root / "run_report_metrics"
    tables = {name: [] for name in ("step_metrics", "episode_metrics", "stage_metrics", "source_retention")}
    analysis_manifest = None
    try:
        if analysis_dir is None:
            if not records_by_path:
                raise ValueError("No readable worker logs; only execution status is available")
            print("Generating report metrics...", flush=True)
            gm.generate(list(records_by_path), analysis, metrics=metrics, path_maps=path_maps,
                        allow_incomplete=True, source_pre=valid_pre, source_posts=posts if valid_pre else ())
        tables, analysis_manifest = load_tables(analysis)
        # Match by mapped full path; never by basename or condition-name guesses.
        evidence_paths = set()
        for item in analysis_manifest["event_inputs"]:
            path = resolve(item["path"])
            if not path.is_file() or gm.sha256(path.read_bytes()) != item["sha256"]:
                raise ValueError(f"Analysis event input differs from this run: {path}")
            evidence_paths.add(str(path))
        if evidence_paths != set(records_by_path):
            raise ValueError("Analysis worker coverage differs from this run")
        for rows in tables.values():
            for row in rows:
                for key in ("event_file", "source_pre", "source_post", "prediction_path", "target_path"):
                    if row.get(key):
                        row[key] = str(resolve(row[key]))
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        notices.append(f"Image metrics unavailable: {error}. Diagnostics below do not substitute other metrics.")
        analysis_manifest = None
        tables = {name: [] for name in tables}
        # Use the same extraction code, with no metric backends or replacement scores.
        for path, records in records_by_path.items():
            steps = [gm.score_step(r, Path(path), gm.ImageCalculator(()), path_maps, {})
                     for r in records if r["_event"]["kind"] == "step"]
            tables["step_metrics"].extend(steps)
            tables["episode_metrics"].extend(gm.summarize_episodes(records, steps, Path(path), ()))
    metrics_used = analysis_manifest["settings"]["metrics"] if analysis_manifest else []
    reconciliation_path = resolve(manifest.get("references", {}).get("reconciliation", "reconciliation.json"))
    reconciliation = None
    try:
        reconciliation = json.loads(reconciliation_path.read_text(encoding="utf-8"))
        current_hashes = {item["path"]: item["sha256"] for item in fingerprints}
        current_hashes[str(manifest_path)] = gm.sha256(manifest_path.read_bytes())
        for item in reconciliation.get("inputs", []):
            path = resolve(item["path"])
            if str(path) in current_hashes and current_hashes[str(path)] != item["sha256"]:
                raise ValueError(f"Reconciliation used different execution evidence: {path}")
    except (OSError, ValueError) as error:
        notices.append(f"Reconciliation unavailable: {error}")
        reconciliation = None
    issues = [c for c in (reconciliation or {}).get("checks", []) if c["status"] in ("fail", "unknown")]
    metadata = manifest.get("metadata", {})
    condition_rows, coverage_rows, diagnostics_rows, chart_workers, gallery = [], [], [], [], []
    for worker in workers:
        name, path = worker["stage_id"], worker["path"]
        steps = [s for s in tables["step_metrics"] if s["event_file"] == path]
        episodes = [e for e in tables["episode_metrics"] if e["event_file"] == path]
        planned = len(metadata.get("habitat_episode_order", [])) if worker["name"] == "habitat" else sum(len(ids) for ids in metadata.get("source_episode_order", {}).values())
        completed = sum(truth(e["episode_completed"]) for e in episodes)
        evidence = next((e for e in fingerprints if e["path"] == path), {})
        coverage_rows.append([name, worker.get("status"), evidence.get("outcome", "unavailable"), f"{completed} / {planned or '?'}",
                              len(steps), sum(s["transition_outcome"] == "completed" for s in steps),
                              sum(s["transition_outcome"] == "no_op" for s in steps), sum(truth(s["collision"]) for s in steps),
                              sum(s["transition_outcome"] == "unfinished" for s in steps),
                              sum(s["event_outcome"] in ("failed", "interrupted") for s in steps)])
        applied = [s for s in steps if truth(s["optimizer_step"])]
        clipping = [s for s in steps if s.get("gradient_clipped") not in (None, "")]
        diagnostics_rows.append([name, len(applied), sum(s["optimizer_state"] == "uncertain" for s in steps),
                                 range_text([s["applied_learning_rate"] for s in applied]),
                                 range_text([s["grad_norm_before_clip"] for s in steps]),
                                 f"{sum(truth(s['gradient_clipped']) for s in clipping)} / {len(clipping)}" if clipping else None])
        for row in tables["stage_metrics"]:
            if row["event_file"] == path:
                condition_rows.append([name, row["data_id"], row["scored_episode_count"],
                                       *[number(row.get(m + "_episode_mean")) for m in ("mae", "ssim", "lpips")]])
        if worker["name"] == "habitat":
            ordered = sorted(episodes, key=lambda e: int(e["episode_index"]))
            chart_workers.append(dict(name=name, episodes=[dict(x=int(e["episode_index"]), label=e["episode_id"],
                **{m: number(e.get(m)) for m in ("mae", "ssim", "lpips")}) for e in ordered],
                steps=[dict(x=i, label=f"{s['episode_id']} / {s['step_idx']}", episode=s["episode_id"],
                            lr=number(s["applied_learning_rate"]) if truth(s["optimizer_step"]) else None,
                            grad=number(s["grad_norm_before_clip"]), scalar=number(s.get("update_weight"))) for i, s in enumerate(steps)]))
            candidates = [s for s in steps if s["pair_status"] == "scored" and number(s.get("mae")) is not None]
            selections = []
            if candidates:
                ordered = sorted(candidates, key=lambda s: (number(s["mae"]), int(s["event_id"])))
                median = statistics.median(number(s["mae"]) for s in ordered)
                selections.extend([("Typical MAE", min(ordered, key=lambda s: (abs(number(s["mae"]) - median), int(s["event_id"])))), ("Highest MAE", ordered[-1])])
            variance = [s for s in steps if number(s.get("prediction_spatial_std")) is not None]
            if variance:
                selections.append(("Lowest spatial variation", min(variance, key=lambda s: (number(s["prediction_spatial_std"]), int(s["event_id"])))))
            for label, row in selections:
                try:
                    images = [thumbnail(Path(row[k + "_path"]), row[k + "_sha256"]) for k in ("prediction", "target")]
                    gallery.append(dict(condition=name, selection=label, episode=row["episode_id"], step=row["step_idx"], images=images,
                                        metrics={m: number(row.get(m)) for m in ("mae", "ssim", "lpips")},
                                        variation=[number(row.get(k + "_spatial_std")) for k in ("prediction", "target")]))
                except (OSError, ValueError, UnidentifiedImageError) as error:
                    notices.append(f"Example omitted: {name}: {error}")
    retention_rows, retention_data = [], []
    names = {w["path"]: w["stage_id"] for w in workers}
    for row in tables["source_retention"]:
        name = names.get(row["source_post"], row["source_post"])
        values = {m: number(row.get(m + "_post_minus_pre_episode_mean")) for m in ("mae", "ssim", "lpips")}
        retention_rows.append([name, row["data_id"], row["paired_episode_count"], row["excluded_episode_count"], *values.values()])
        retention_data.append(dict(name=name, dataset=row["data_id"], **values))
    # Reuse is an execution declaration, never a newly measured zero delta.
    seed_plan = manifest.get("references", {}).get("seed_plan")
    if seed_plan:
        try:
            plan = json.loads(resolve(seed_plan).read_text(encoding="utf-8"))
            for condition in plan["conditions"]:
                if condition.get("source_post_reused_from_source_pre"):
                    notices.append(f"{condition['condition_id']}: source-post reuses source-pre; no separate post measurement or retention delta is reported.")
        except (OSError, ValueError, KeyError) as error:
            notices.append(f"Reuse metadata unavailable: {error}")
    exclusion_rows = []
    for worker in workers:
        counts = Counter(s["pair_status"] for s in tables["step_metrics"] if s["event_file"] == worker["path"])
        exclusion_rows.extend([worker["stage_id"], status, count] for status, count in sorted(counts.items()) if status != "scored")
    def link(path, label):
        relative = os.path.relpath(path, output.parent).replace("\\", "/")
        return f'<a href="{html.escape(quote(relative, safe="/:"), quote=True)}">{html.escape(label)}</a>'
    links = [link(manifest_path, "Run manifest")]
    if reconciliation is not None:
        links.append(link(reconciliation_path, "Reconciliation"))
    if analysis_manifest:
        links += [link(analysis / filename, filename) for filename in ("analysis_manifest.json", "metric_definitions.json", "step_metrics.csv", "episode_metrics.csv", "source_retention.csv")]
    provenance = dict(report_version="1.0.0", generated_at=datetime.now(timezone.utc).isoformat(),
                      report_script_sha256=gm.sha256(Path(__file__).read_bytes()),
                      template_sha256=gm.sha256(Path(__file__).with_suffix(".html").read_bytes()),
                      run_manifest_sha256=gm.sha256(manifest_path.read_bytes()), event_inputs=fingerprints,
                      analysis_manifest_sha256=gm.sha256((analysis / "analysis_manifest.json").read_bytes()) if analysis_manifest else None,
                      reconciliation_sha256=gm.sha256(reconciliation_path.read_bytes()) if reconciliation else None,
                      path_maps=path_maps, requested_metrics=list(metrics), available_metrics=metrics_used)
    def issue_row(check):
        detail = check.get("reason") or f"expected {check.get('expected')!r}; observed {check.get('observed')!r}"
        return [check["status"], check.get("stage"), check["check"], str(check.get("location") or ""), detail]
    replacements = {
        "TITLE": html.escape(manifest.get("run_id", root.name)),
        "STATUS": html.escape(f"Execution: {manifest.get('status', 'unknown')} · Reconciliation: {(reconciliation or {}).get('status', 'unavailable')}"),
        "META": html.escape(f"Seed {metadata.get('seed', '?')} · {len(workers)} launched workers · metrics: {', '.join(metrics_used) or 'unavailable'}"),
        "NOTICES": ''.join(f'<p class="notice">{html.escape(n)}</p>' for n in notices),
        "ISSUES": table(["Status", "Worker", "Check", "Location", "Evidence"], [issue_row(c) for c in issues[:8]]) if issues else '<p>No reported failures or unknowns.</p>' if reconciliation else '<p>Reconciliation evidence is unavailable.</p>',
        "ALL_ISSUES": table(["Status", "Worker", "Check", "Location", "Evidence"], [issue_row(c) for c in issues]),
        "RESULTS": table(["Worker", "Dataset", "Scored episodes", "MAE ↓", "SSIM ↑", "LPIPS ↓"], condition_rows),
        "RETENTION": table(["Worker", "Dataset", "Paired episodes", "Excluded episodes", "Δ MAE", "Δ SSIM", "Δ LPIPS"], retention_rows),
        "COVERAGE": table(["Worker", "Stage", "Worker outcome", "Episodes done/planned", "Attempts", "Completed", "No-ops", "Collisions", "Unfinished", "Failed/interrupted events"], coverage_rows),
        "DIAGNOSTICS": table(["Worker", "Updates", "Uncertain", "Applied LR median [min, max]", "Gradient norm median [min, max]", "Clipped / observed"], diagnostics_rows),
        "EXCLUSIONS": table(["Worker", "Exclusion", "Attempts"], exclusion_rows),
        "LINKS": ' · '.join(links),
        "PROVENANCE": html.escape(json.dumps(provenance, indent=2)),
        "SETTINGS": html.escape(json.dumps(metadata, indent=2)),
        "DATA": json.dumps(dict(workers=chart_workers, retention=retention_data, gallery=gallery), allow_nan=False).replace("<", "\\u003c"),
    }
    document = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")
    # One substitution pass: input text cannot introduce template directives.
    import re
    document = re.sub(r"@@([A-Z_]+)@@", lambda match: replacements[match[1]], document)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(document)
    print(f"Run report: {output}", flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--analysis-dir", type=Path, help="Reuse and verify existing official metric outputs")
    parser.add_argument("--metrics", nargs="+", choices=("mae", "ssim", "lpips"), default=["mae", "ssim", "lpips"])
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    args = parser.parse_args(argv)
    mappings = []
    for value in args.path_map:
        old, sep, new = value.partition("=")
        if not sep or not old or not new:
            parser.error("--path-map requires OLD=NEW")
        mappings.append((old, new))
    try:
        build_report(args.run_dir, output=args.output, analysis_dir=args.analysis_dir,
                     metrics=args.metrics, path_maps=mappings)
    except (OSError, ValueError) as error:
        print(f"Run report not generated: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
