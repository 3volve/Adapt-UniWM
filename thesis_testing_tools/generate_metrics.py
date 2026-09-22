"""Reproduce thesis metrics from EventLogger v1 worker logs and saved images.

Run with python -m thesis_testing_tools.generate_metrics --help.
The experiment runner never imports this offline analysis module.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from PIL import Image, UnidentifiedImageError

METHOD_VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS = {
    "method_version": METHOD_VERSION,
    "event_schema": 1,
    "images": "Decode to RGB, native resolution, float32 in [0,1]; no resizing. Shapes must match.",
    "mae": "Mean absolute error across all RGB pixels in [0,1], float32 differences accumulated with float64 mean; lower is better.",
    "ssim": {"implementation": "pytorch_msssim.ssim", "data_range": 1.0,
             "size_average": True, "win_size": 11, "win_sigma": 1.5,
             "K": [0.01, 0.03], "nonnegative_ssim": False},
    "lpips": {"implementation": "lpips.LPIPS", "net": "alex", "version": "0.1",
              "input_range": [-1, 1], "spatial": False, "device": "cpu",
              "pretrained": True, "pnet_rand": False, "pnet_tune": False, "eval_mode": True},
    "selection": "Score completed transition outcomes only, even if later checkpoint saving failed. Missing/corrupt/mismatched images are excluded with reasons; never zero-filled.",
    "prediction": "habitat: wrapper.predicted_obs_path; replay: wrapper.evaluation.predicted_obs_path. Target: wrapper.real_obs_path.",
    "aggregation": "Episode: arithmetic mean of scored transitions. Stage/data_id: equal-weight mean of completed episodes with scores, plus separately labelled pooled transition means. No pooling across source datasets or workers.",
    "retention": "Post minus pre episode means, only completed episodes with identical scored step indices and target-image hashes; positive MAE/LPIPS and negative SSIM indicate deterioration.",
    "incomplete": "No reconstruction of unwritten events. Missing terminal run_summary is rejected unless explicitly allowed. Episode completion requires episode_end evidence.",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def finite(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def mean(values):
    present = [v for v in values if v is not None]
    return math.fsum(present) / len(present) if present else None


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, empty_fields=()):
    keys = list(dict.fromkeys(key for row in rows for key in row)) or list(empty_fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_events(path, allow_incomplete=False):
    """Validate the file boundary once; consumers below use this representation."""
    raw = path.read_bytes()
    records = []
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        try:
            event = json.loads(line)
            meta = event["_event"]
            if type(meta["schema_version"]) is not int or meta["schema_version"] != 1 or type(meta["id"]) is not int or meta["id"] != len(records):
                raise ValueError("expected schema_version=1 and consecutive event IDs starting at 0")
            if meta["kind"] not in ("startup", "step", "run_summary") or not isinstance(event["outcome"], str):
                raise ValueError("invalid event kind or outcome")
            if records and meta["kind"] == "startup":
                raise ValueError("startup must occur only once")
            if records and records[-1]["_event"]["kind"] == "run_summary":
                raise ValueError("records follow the run summary")
            if meta["kind"] == "step":
                for key in ("data_id", "episode_id", "source_mode"):
                    if not isinstance(event[key], str):
                        raise ValueError(f"{key} must be a string")
                for key in ("episode_index", "step_idx"):
                    if type(event[key]) is not int or event[key] < 0:
                        raise ValueError(f"{key} must be a nonnegative integer")
                if event["source_mode"] not in ("habitat", "replay"):
                    raise ValueError("unsupported source_mode")
                if not isinstance(event["transition"]["outcome"], str):
                    raise ValueError("transition.outcome must be a string")
            for key in ("wrapper", "training", "environment", "adapter", "prediction", "episode_setup", "episode_end"):
                if key in event and not isinstance(event[key], dict):
                    raise ValueError(f"{key} must be a dictionary")
            for section, key in (("wrapper", "evaluation"), ("wrapper", "controller_state"),
                                 ("environment", "metrics"), ("adapter", "primitives")):
                if key in event.get(section, {}) and not isinstance(event[section][key], dict):
                    raise ValueError(f"{section}.{key} must be a dictionary")
            for section, key in (("wrapper", "collision"), ("wrapper", "replanned"), ("training", "optimizer_step")):
                value = event.get(section, {}).get(key)
                if value is not None and type(value) is not bool:
                    raise ValueError(f"{section}.{key} must be boolean or null")
            for key, primitive in event.get("adapter", {}).get("primitives", {}).items():
                if not isinstance(primitive, dict):
                    raise ValueError(f"adapter.primitives.{key} must be a dictionary")
            for section in ("episode_setup", "episode_end"):
                for key, details in event.get(section, {}).items():
                    data_id, index = key.rsplit("/", 1)
                    if not data_id or not index.isdecimal() or not isinstance(details, dict):
                        raise ValueError(f"{section}.{key} requires data_id/nonnegative_index and a dictionary")
                    if details.get("episode_id") is not None and not isinstance(details["episode_id"], str):
                        raise ValueError(f"{section}.{key}.episode_id must be a string")
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path}:{line_no}: invalid worker event: {error}") from error
        records.append(event)
    if not records or records[0]["_event"]["kind"] != "startup":
        raise ValueError(f"{path}: expected a startup record")
    if "pipeline" in records[0]:
        raise ValueError(f"{path}: coordinator log supplied; select worker logs instead")
    finalized = records[-1]["_event"]["kind"] == "run_summary"
    if not finalized and not allow_incomplete:
        raise ValueError(f"{path}: missing run_summary; use --allow-incomplete for partial analysis")
    return records, {"path": str(path), "sha256": sha256(raw), "bytes": len(raw),
                     "finalized": finalized, "outcome": records[-1]["outcome"] if finalized else "unfinished"}


class ImageCalculator:
    """CPU metrics with explicit settings; learned weights are fingerprinted."""

    def __init__(self, metrics):
        self.metrics = tuple(metrics)
        self.provenance = {"device": "cpu", "metrics": list(metrics)}
        if "ssim" in metrics or "lpips" in metrics:
            try:
                import torch
            except ModuleNotFoundError as error:
                raise RuntimeError("SSIM/LPIPS require the thesis PyTorch environment; use --metrics mae for a NumPy/Pillow-only report") from error
            self.torch = torch
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            self.provenance.update(torch_threads=1, deterministic_algorithms=True)
        if "ssim" in metrics:
            from pytorch_msssim import ssim
            self.ssim = ssim
        if "lpips" in metrics:
            import lpips
            self.lpips = lpips.LPIPS(net="alex", version="0.1", spatial=False,
                pretrained=True, pnet_rand=False, pnet_tune=False, eval_mode=True).cpu().eval()
            digest = hashlib.sha256()
            for name, tensor in sorted(self.lpips.state_dict().items()):
                digest.update(name.encode())
                digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            self.provenance["lpips_state_sha256"] = digest.hexdigest()

    def __call__(self, predicted, target):
        scores = {}
        if "mae" in self.metrics:
            scores["mae"] = float(np.abs(predicted - target).mean(dtype=np.float64))
        if "ssim" in self.metrics or "lpips" in self.metrics:
            p = self.torch.from_numpy(predicted).permute(2, 0, 1).unsqueeze(0)
            t = self.torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0)
            with self.torch.inference_mode():
                if "ssim" in self.metrics:
                    scores["ssim"] = self.ssim(p, t, data_range=1.0, size_average=True,
                        win_size=11, win_sigma=1.5, K=(0.01, 0.03), nonnegative_ssim=False).item()
                if "lpips" in self.metrics:
                    scores["lpips"] = self.lpips(p * 2 - 1, t * 2 - 1).item()
        if any(finite(value) is None for value in scores.values()):
            raise ValueError("Metric backend returned a non-finite score")
        return scores


def resolve_image(value, event_file, path_maps):
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("\\", "/")
    for old, new in path_maps:
        old = old.replace("\\", "/").rstrip("/")
        if normalized == old or normalized.startswith(old + "/"):
            return (Path(new) / normalized[len(old):].lstrip("/")).resolve()
    path = Path(normalized)
    return path.resolve() if path.is_absolute() else (event_file.parent / path).resolve()


def load_image(path, inventory):
    raw = path.read_bytes()
    digest = sha256(raw)
    key = str(path)
    if key in inventory and inventory[key]["sha256"] != digest:
        raise ValueError(f"Image changed during analysis: {path}")
    inventory[key] = {"path": key, "sha256": digest, "bytes": len(raw)}
    with Image.open(BytesIO(raw)) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return array, digest


def score_step(event, path, calculator, path_maps, inventory):
    wrapper, training = event.get("wrapper", {}), event.get("training", {})
    environment = event.get("environment", {}).get("metrics", {})
    row = {"event_file": str(path), "event_id": event["_event"]["id"],
           **{key: event[key] for key in ("data_id", "episode_id", "episode_index", "step_idx", "source_mode")},
           "event_outcome": event["outcome"], "transition_outcome": event["transition"]["outcome"],
           "collision": wrapper.get("collision"), "optimizer_step": training.get("optimizer_step"),
           "optimizer_state": ("not_reached" if "optimizer_step" not in training else
                               "uncertain" if training["optimizer_step"] is None else
                               "completed" if training["optimizer_step"] else "not_applied"),
           "replanned": wrapper.get("replanned"), "update_skip_reason": wrapper.get("update_skip_reason"),
           "primitive_count": sum(p.get("outcome") == "completed" for p in event.get("adapter", {}).get("primitives", {}).values()),
           "primitive_collision_count": sum(p.get("collision") is True for p in event.get("adapter", {}).get("primitives", {}).values()),
           "visual_loss": finite(training.get("visual_loss")),
           "effective_learning_rate": finite(wrapper.get("effective_learning_rate")),
           "applied_learning_rate": finite(training.get("applied_learning_rate")),
           "update_weight": finite(wrapper.get("update_weight")),
           "grad_norm_before_clip": finite(training.get("grad_norm_before_clip")),
           "grad_norm_after_clip": finite(training.get("grad_norm_after_clip")),
           "action_entropy": finite(event.get("prediction", {}).get("act_entropy")),
           "visualization_entropy": finite(event.get("prediction", {}).get("viz_entropy")),
           "forced_eval_visualization_entropy": finite(wrapper.get("evaluation", {}).get("viz_entropy"))}
    for key in ("gl_ach", "gl_ne_visual", "st_ach", "st_ne_visual"):
        row[key] = finite(wrapper.get("controller_state", {}).get(key))
    for key in ("success", "spl", "soft_spl", "distance_to_goal", "num_steps"):
        row[key] = finite(environment.get(key))
    predicted_ref = (wrapper.get("evaluation", {}).get("predicted_obs_path") if event["source_mode"] == "replay"
                     else wrapper.get("predicted_obs_path"))
    predicted = resolve_image(predicted_ref, path, path_maps)
    target = resolve_image(wrapper.get("real_obs_path"), path, path_maps)
    row.update(prediction_path=str(predicted) if predicted else None, target_path=str(target) if target else None,
               prediction_sha256=None, target_sha256=None, pair_status="scored")
    row.update({metric: None for metric in calculator.metrics})
    if row["transition_outcome"] != "completed":
        row["pair_status"] = "transition_not_completed"
    elif predicted is None or target is None:
        row["pair_status"] = "image_reference_missing"
    elif not predicted.is_file() or not target.is_file():
        row["pair_status"] = "image_file_missing"
    else:
        try:
            p, row["prediction_sha256"] = load_image(predicted, inventory)
            t, row["target_sha256"] = load_image(target, inventory)
        except (UnidentifiedImageError, OSError):
            row["pair_status"] = "image_unreadable"
            return row
        if p.shape != t.shape:
            row["pair_status"] = "image_shape_mismatch"
        elif ("ssim" in calculator.metrics and min(p.shape[:2]) < 11) or ("lpips" in calculator.metrics and min(p.shape[:2]) < 64):
            row["pair_status"] = "image_too_small"
        else:
            row.update(calculator(p, t))
    return row


def summarize_episodes(records, steps, path, metrics):
    groups, endings = {}, {}
    for event in records:
        for key, setup in event.get("episode_setup", {}).items():
            data_id, index = key.rsplit("/", 1)
            groups.setdefault((data_id, int(index)), {"episode_id": setup.get("episode_id"), "steps": []})
        for key, ending in event.get("episode_end", {}).items():
            data_id, index = key.rsplit("/", 1)
            endings[(data_id, int(index))] = ending
    for step in steps:
        key = (step["data_id"], step["episode_index"])
        group = groups.setdefault(key, {"episode_id": step["episode_id"], "steps": []})
        if group["episode_id"] not in (None, step["episode_id"]):
            raise ValueError(f"{path}: conflicting episode IDs at {key}")
        group["episode_id"] = step["episode_id"]
        if any(s["step_idx"] == step["step_idx"] for s in group["steps"]):
            raise ValueError(f"{path}: duplicate step index in {key}")
        group["steps"].append(step)
    rows = []
    diagnostic_names = ("visual_loss", "effective_learning_rate", "applied_learning_rate", "update_weight",
                        "grad_norm_before_clip", "grad_norm_after_clip", "action_entropy", "visualization_entropy",
                        "forced_eval_visualization_entropy", "gl_ach", "gl_ne_visual", "st_ach", "st_ne_visual")
    for (data_id, index), group in groups.items():
        items = group["steps"]
        ending = endings.get((data_id, index))
        row = {"event_file": str(path), "data_id": data_id, "episode_index": index,
               "episode_id": group["episode_id"], "episode_completed": ending is not None,
               "termination_reason": ending.get("termination_reason") if ending else None,
               "attempt_count": len(items), "completed_transition_count": sum(s["transition_outcome"] == "completed" for s in items),
               "no_op_count": sum(s["transition_outcome"] == "no_op" for s in items),
               "unfinished_transition_count": sum(s["transition_outcome"] == "unfinished" for s in items),
               "failed_event_count": sum(s["event_outcome"] == "failed" for s in items),
               "interrupted_event_count": sum(s["event_outcome"] == "interrupted" for s in items),
               "collision_count": sum(s["collision"] is True for s in items),
               "collision_observed_count": sum(s["collision"] is not None for s in items),
               "optimizer_step_count": sum(s["optimizer_step"] is True for s in items),
               "uncertain_optimizer_count": sum(s["optimizer_state"] == "uncertain" for s in items),
               "primitive_count": sum(s["primitive_count"] for s in items),
               "primitive_collision_count": sum(s["primitive_collision_count"] for s in items),
               "replan_count": sum(s["replanned"] is True for s in items),
               "scored_pair_count": sum(s["pair_status"] == "scored" for s in items)}
        row["excluded_pair_count"] = len(items) - row["scored_pair_count"]
        for metric in metrics:
            row[metric] = mean([s[metric] for s in items])
        for name in diagnostic_names:
            values = [s[name] for s in items if s[name] is not None]
            row[name + "_mean"] = mean(values)
            row[name + "_count"] = len(values)
        # The final observation may be absent on failure; never substitute an earlier success.
        for name in ("success", "spl", "soft_spl", "distance_to_goal", "num_steps"):
            row["final_" + name] = items[-1][name] if items else None
        rows.append(row)
    return rows


def compare_source(pre, post, episodes, steps, metrics):
    def index(path):
        result = {}
        for ep in episodes:
            if ep["event_file"] != str(path):
                continue
            key = (ep["data_id"], ep["episode_id"])
            if key in result:
                raise ValueError(f"{path}: source comparison requires unique data_id/episode_id pairs")
            result[key] = ep
        return result
    before, after = index(pre), index(post)
    rows = []
    for key in sorted(before.keys() | after.keys(), key=str):
        a, b = before.get(key), after.get(key)
        row = {"source_pre": str(pre), "source_post": str(post), "data_id": key[0], "episode_id": key[1], "status": "paired"}
        def coverage(ep):
            return {s["step_idx"]: s["target_sha256"] for s in steps if ep and s["event_file"] == ep["event_file"]
                    and s["data_id"] == ep["data_id"] and s["episode_index"] == ep["episode_index"] and s["pair_status"] == "scored"}
        if a is None or b is None:
            row["status"] = "unmatched_episode"
        elif not a["episode_completed"] or not b["episode_completed"]:
            row["status"] = "incomplete_episode"
        elif not coverage(a) or coverage(a) != coverage(b):
            row["status"] = "image_coverage_mismatch"
        for metric in metrics:
            row[metric + "_pre"] = a[metric] if a else None
            row[metric + "_post"] = b[metric] if b else None
            row[metric + "_post_minus_pre"] = b[metric] - a[metric] if row["status"] == "paired" else None
        rows.append(row)
    return rows


def generate(event_files, output, *, metrics=("mae", "ssim", "lpips"), path_maps=(),
             allow_incomplete=False, source_pre=None, source_posts=()):
    event_files = list(dict.fromkeys(Path(p).resolve() for p in event_files))
    if source_pre is not None:
        source_pre = Path(source_pre).resolve()
        source_posts = list(dict.fromkeys(Path(p).resolve() for p in source_posts))
        event_files = list(dict.fromkeys([*event_files, source_pre, *source_posts]))
    if not event_files or not metrics or set(metrics) - {"mae", "ssim", "lpips"}:
        raise ValueError("Supply worker event files and at least one of mae, ssim, lpips")
    if source_posts and source_pre is None:
        raise ValueError("--source-post requires --source-pre")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Report directory already exists: {output}; use a new directory")
    inputs = [(path, *read_events(path, allow_incomplete)) for path in event_files]
    metrics = tuple(dict.fromkeys(metrics))
    calculator = ImageCalculator(metrics)
    inventory, steps, episodes = {}, [], []
    for path, records, _ in inputs:
        stage_steps = [score_step(e, path, calculator, path_maps, inventory) for e in records if e["_event"]["kind"] == "step"]
        steps.extend(stage_steps)
        episodes.extend(summarize_episodes(records, stage_steps, path, metrics))
    stages = []
    for filename, data_id in sorted({(e["event_file"], e["data_id"]) for e in episodes}):
        selected = [e for e in episodes if e["event_file"] == filename and e["data_id"] == data_id]
        completed = [e for e in selected if e["episode_completed"]]
        indices = {e["episode_index"] for e in completed}
        pooled = [s for s in steps if s["event_file"] == filename and s["data_id"] == data_id and s["episode_index"] in indices]
        row = {"event_file": filename, "data_id": data_id, "episode_count": len(selected), "completed_episode_count": len(completed),
               "scored_episode_count": sum(e["scored_pair_count"] > 0 for e in completed)}
        for metric in metrics:
            row[metric + "_episode_mean"] = mean([e[metric] for e in completed])
            row[metric + "_transition_mean"] = mean([s[metric] for s in pooled])
        stages.append(row)
    comparisons = []
    for post in source_posts:
        if any(s["source_mode"] != "replay" for s in steps if s["event_file"] in (str(source_pre), str(post))):
            raise ValueError("Source retention comparisons require replay workers")
        comparisons.extend(compare_source(source_pre, post, episodes, steps, metrics))
    retention = []
    for post, data_id in sorted({(r["source_post"], r["data_id"]) for r in comparisons}):
        selected = [r for r in comparisons if r["source_post"] == post and r["data_id"] == data_id]
        paired = [r for r in selected if r["status"] == "paired"]
        retention.append({"source_pre": str(source_pre), "source_post": post, "data_id": data_id,
            "paired_episode_count": len(paired), "excluded_episode_count": len(selected) - len(paired),
            **{metric + "_post_minus_pre_episode_mean": mean([r[metric + "_post_minus_pre"] for r in paired]) for metric in metrics}})
    packages = {}
    for name in ("numpy", "Pillow", "torch", "torchvision", "lpips", "pytorch-msssim"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    code_status = subprocess.run(["git", "status", "--porcelain", "--", str(Path(__file__).resolve())],
                                 cwd=ROOT, capture_output=True, text=True)
    script = Path(__file__).read_bytes()
    manifest = {"method_version": METHOD_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "script_sha256": sha256(script), "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "script_git_status": code_status.stdout.strip() if code_status.returncode == 0 else None,
        "python": sys.version, "platform": platform.platform(), "packages": packages,
        "settings": {"metrics": list(metrics), "path_maps": list(path_maps), "allow_incomplete": allow_incomplete,
                     "source_pre": str(source_pre) if source_pre else None, "source_posts": [str(p) for p in source_posts]},
        "calculator": calculator.provenance, "event_inputs": [item[2] for item in inputs],
        "image_inputs": list(inventory.values()), "pair_status_counts": {status: sum(s["pair_status"] == status for s in steps) for status in sorted({s["pair_status"] for s in steps})}}
    output.mkdir(parents=True)
    write_csv(output / "step_metrics.csv", steps, ("event_file", "event_id", "pair_status", *metrics))
    write_csv(output / "episode_metrics.csv", episodes, ("event_file", "data_id", "episode_id", "episode_completed", *metrics))
    write_csv(output / "stage_metrics.csv", stages, ("event_file", "data_id", "episode_count"))
    write_csv(output / "source_comparison.csv", comparisons, ("source_pre", "source_post", "data_id", "episode_id", "status"))
    write_csv(output / "source_retention.csv", retention, ("source_pre", "source_post", "data_id", "paired_episode_count"))
    write_json(output / "metric_definitions.json", DEFINITIONS)
    (output / "generate_metrics.py").write_bytes(script)
    manifest["outputs"] = [{"path": p.name, "sha256": sha256(p.read_bytes())} for p in sorted(output.iterdir())]
    write_json(output / "analysis_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, nargs="+", default=[], help="Explicit worker events.jsonl files; never the coordinator log")
    parser.add_argument("--output", type=Path, required=True, help="New report directory; existing directories are not overwritten")
    parser.add_argument("--metrics", nargs="+", choices=("mae", "ssim", "lpips"), default=["mae", "ssim", "lpips"])
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW", help="Explicit image-path prefix mapping for downloaded runs; first matching prefix wins")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--source-pre", type=Path)
    parser.add_argument("--source-post", type=Path, action="append", default=[])
    args = parser.parse_args()
    maps = []
    for value in args.path_map:
        old, separator, new = value.partition("=")
        if not separator or not old or not new:
            parser.error("--path-map must have nonempty OLD=NEW prefixes")
        maps.append((old, new))
    generate(args.events, args.output, metrics=args.metrics, path_maps=maps,
             allow_incomplete=args.allow_incomplete, source_pre=args.source_pre, source_posts=args.source_post)
    print(f"Metrics written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
