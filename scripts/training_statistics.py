from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping


STATISTICS_FILENAME = "training_statistics.json"
SUMMARY_FILENAME = "training_summary.json"

EVALUATION_PLOT_METRICS = (
    "eval_loss",
    "eval_navigation_simulation_task_acc",
    "eval_action_valid_rate",
    "eval_action_exact_match_acc",
    "eval_simulation_visualization_ssim",
    "eval_simulation_visualization_lpips",
    "eval_visual_ssim_gain_over_copy",
    "eval_visual_lpips_gain_over_copy",
    "eval_visual_copy_beat_rate",
    "eval_simulation_visualization_fid",
)

_NON_METRIC_KEYS = {
    "epoch",
    "step",
    "total_flos",
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
    "eval_runtime",
    "eval_samples_per_second",
    "eval_steps_per_second",
}


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _metric_direction(name: str) -> str | None:
    lower_is_better = ("loss", "lpips", "dreamsim", "fid", "mae", "mse", "error")
    higher_is_better = ("accuracy", "_acc", "ssim", "psnr", "success", "spl", "gain")
    lowered = name.lower()
    if "gain" in lowered or "beat_rate" in lowered or "valid_rate" in lowered:
        return "higher"
    if any(part in lowered for part in lower_is_better):
        return "lower"
    if any(part in lowered for part in higher_is_better):
        return "higher"
    return None


def metric_series(log_history: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, float]]]:
    series: dict[str, list[dict[str, float]]] = {}
    for index, entry in enumerate(log_history):
        step = _finite_float(entry.get("step"))
        if step is None:
            step = float(index)
        epoch = _finite_float(entry.get("epoch"))
        for name, raw_value in entry.items():
            if name in _NON_METRIC_KEYS:
                continue
            value = _finite_float(raw_value)
            if value is None:
                continue
            point = {"step": step, "value": value}
            if epoch is not None:
                point["epoch"] = epoch
            series.setdefault(name, []).append(point)
    return series


def _linear_slope(points: list[dict[str, float]]) -> float | None:
    if len(points) < 2:
        return None
    xs = [point["step"] for point in points]
    ys = [point["value"] for point in points]
    x_mean, y_mean = fmean(xs), fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


def summarize_metric(name: str, points: list[dict[str, float]]) -> dict[str, Any]:
    values = [point["value"] for point in points]
    direction = _metric_direction(name)
    best_index = min(range(len(values)), key=values.__getitem__)
    if direction == "higher":
        best_index = max(range(len(values)), key=values.__getitem__)
    first, last = values[0], values[-1]
    change = last - first
    relative_change = None if first == 0 else change / abs(first)
    improved = None
    if direction == "lower":
        improved = last < first
    elif direction == "higher":
        improved = last > first

    return {
        "count": len(points),
        "direction": direction,
        "first": first,
        "first_step": int(points[0]["step"]),
        "last": last,
        "last_step": int(points[-1]["step"]),
        "minimum": min(values),
        "maximum": max(values),
        "mean": fmean(values),
        "best": values[best_index],
        "best_step": int(points[best_index]["step"]),
        "change": change,
        "relative_change": relative_change,
        "slope_per_step": _linear_slope(points),
        "improved": improved,
    }


def build_statistics(state: Mapping[str, Any], location: str | Path) -> dict[str, Any]:
    history = list(state.get("log_history", []))
    series = metric_series(history)
    summaries = {
        name: summarize_metric(name, points)
        for name, points in sorted(series.items())
        if points
    }
    latest_metrics = {name: points[-1]["value"] for name, points in sorted(series.items())}
    global_step = int(state.get("global_step", 0) or 0)
    max_steps = int(state.get("max_steps", 0) or 0)
    progress = None if max_steps <= 0 else global_step / max_steps

    warnings: list[str] = []
    eval_metrics = [name for name in summaries if name.startswith("eval_")]
    if not eval_metrics:
        warnings.append(
            "No held-out evaluation metrics were logged. Loss trends show optimization, not model efficacy."
        )
    elif max(summaries[name]["count"] for name in eval_metrics) < 2:
        warnings.append("Only one held-out evaluation point is available; improvement cannot be measured.")

    return {
        "schema_version": 1,
        "location": str(Path(location)),
        "global_step": global_step,
        "epoch": _finite_float(state.get("epoch")),
        "max_steps": max_steps,
        "progress": progress,
        "best_metric": _finite_float(state.get("best_metric")),
        "best_model_checkpoint": state.get("best_model_checkpoint"),
        "latest_metrics": latest_metrics,
        "metric_summaries": summaries,
        "warnings": warnings,
    }


def _state_to_mapping(state: Any) -> dict[str, Any]:
    if isinstance(state, Mapping):
        return dict(state)
    fields = (
        "global_step",
        "epoch",
        "max_steps",
        "best_metric",
        "best_model_checkpoint",
        "log_history",
    )
    return {field: getattr(state, field, None) for field in fields}


def write_training_statistics(output_dir: str | Path, state: Any) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    statistics = build_statistics(_state_to_mapping(state), output_dir)
    output_path = output_dir / STATISTICS_FILENAME
    output_path.write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    return output_path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def _checkpoint_statistics(run_dir: Path) -> list[dict[str, Any]]:
    checkpoints = sorted(
        (path for path in run_dir.glob("checkpoint-*") if path.is_dir()),
        key=_checkpoint_step,
    )
    results: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        statistics_path = checkpoint / STATISTICS_FILENAME
        state_path = checkpoint / "trainer_state.json"
        if statistics_path.exists():
            statistics = _load_json(statistics_path)
        elif state_path.exists():
            statistics = build_statistics(_load_json(state_path), checkpoint)
        else:
            continue
        statistics["checkpoint"] = checkpoint.name
        results.append(statistics)
    return results


def _latest_state(run_dir: Path, checkpoint_statistics: list[dict[str, Any]]) -> dict[str, Any] | None:
    root_state = run_dir / "trainer_state.json"
    if root_state.exists():
        return _load_json(root_state)
    if checkpoint_statistics:
        checkpoint = run_dir / checkpoint_statistics[-1]["checkpoint"] / "trainer_state.json"
        if checkpoint.exists():
            return _load_json(checkpoint)
    return None


def _write_checkpoint_csv(path: Path, checkpoints: list[dict[str, Any]]) -> None:
    metric_names = sorted({
        name
        for checkpoint in checkpoints
        for name in checkpoint.get("latest_metrics", {})
    })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["checkpoint", "global_step", "epoch", "progress", *metric_names],
        )
        writer.writeheader()
        for checkpoint in checkpoints:
            writer.writerow({
                "checkpoint": checkpoint.get("checkpoint"),
                "global_step": checkpoint.get("global_step"),
                "epoch": checkpoint.get("epoch"),
                "progress": checkpoint.get("progress"),
                **checkpoint.get("latest_metrics", {}),
            })


def _write_history_csv(path: Path, history: list[Mapping[str, Any]]) -> None:
    fieldnames = sorted({key for entry in history for key in entry})
    preferred = [key for key in ("step", "epoch") if key in fieldnames]
    fieldnames = preferred + [key for key in fieldnames if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)


def _plot_series(
    output_path: Path,
    series: Mapping[str, list[dict[str, float]]],
    names: list[str],
    title: str,
) -> bool:
    if not names:
        return False
    try:
        matplotlib_config = output_path.parent / ".matplotlib"
        matplotlib_config.mkdir(exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    columns = min(2, len(names))
    rows = (len(names) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(13, max(4, 3.2 * rows)), squeeze=False)
    for axis, name in zip(axes.flat, names):
        points = series[name]
        axis.plot([point["step"] for point in points], [point["value"] for point in points], linewidth=1.6)
        axis.set_title(name)
        axis.set_xlabel("training step")
        axis.grid(alpha=0.25)
    for axis in list(axes.flat)[len(names):]:
        axis.set_visible(False)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return True


def _interpret(metric_summaries: Mapping[str, Mapping[str, Any]]) -> list[str]:
    interpretation: list[str] = []
    loss = metric_summaries.get("loss")
    eval_loss = metric_summaries.get("eval_loss")
    if loss and loss["count"] >= 2:
        interpretation.append(
            f"Training loss changed from {loss['first']:.4g} to {loss['last']:.4g} "
            f"({loss['relative_change']:+.1%})."
        )
    if eval_loss and eval_loss["count"] >= 2:
        interpretation.append(
            f"Held-out loss changed from {eval_loss['first']:.4g} to {eval_loss['last']:.4g} "
            f"({eval_loss['relative_change']:+.1%})."
        )
        if loss and loss["last"] < loss["first"] and eval_loss["last"] > eval_loss["first"]:
            interpretation.append("Training loss improved while held-out loss worsened, which is evidence of overfitting.")

    efficacy_metrics = [
        (name, summary)
        for name, summary in metric_summaries.items()
        if name in EVALUATION_PLOT_METRICS
        and name != "eval_loss"
        and summary.get("direction") is not None
        and summary.get("count", 0) >= 2
    ]
    for name, summary in efficacy_metrics:
        verdict = "improved" if summary["improved"] else "worsened or remained flat"
        interpretation.append(
            f"{name} {verdict}: {summary['first']:.4g} -> {summary['last']:.4g}."
        )
    if not efficacy_metrics:
        interpretation.append(
            "There are not yet two comparable held-out evaluation points, so training efficacy is undetermined."
        )
    return interpretation


def summarize_training_run(run_path: str | Path, make_plots: bool = True) -> dict[str, Any]:
    requested_path = Path(run_path).resolve()
    run_dir = requested_path.parent if requested_path.name.startswith("checkpoint-") else requested_path
    checkpoints = _checkpoint_statistics(run_dir)
    latest_state = _latest_state(run_dir, checkpoints)
    if latest_state is None:
        direct_state = requested_path / "trainer_state.json"
        if direct_state.exists():
            latest_state = _load_json(direct_state)
        else:
            raise FileNotFoundError(f"No trainer_state.json found under {requested_path}")

    final_statistics = build_statistics(latest_state, run_dir)
    history = list(latest_state.get("log_history", []))
    series = metric_series(history)
    output_dir = run_dir / "training_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_csv = output_dir / "checkpoint_metrics.csv"
    history_csv = output_dir / "training_history.csv"
    _write_checkpoint_csv(checkpoint_csv, checkpoints)
    _write_history_csv(history_csv, history)

    plot_paths: list[str] = []
    if make_plots:
        loss_names = [
            name for name in ("loss", "bc_loss", "discrepancy_loss", "eval_loss")
            if name in series
        ]
        loss_plot = output_dir / "training_loss_curves.png"
        if _plot_series(loss_plot, series, loss_names, "Training and held-out losses"):
            plot_paths.append(str(loss_plot))

        eval_names = [name for name in EVALUATION_PLOT_METRICS if name in series]
        eval_plot = output_dir / "evaluation_metric_curves.png"
        if _plot_series(eval_plot, series, eval_names, "Held-out evaluation metrics"):
            plot_paths.append(str(eval_plot))

    summary = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "final_statistics": final_statistics,
        "interpretation": _interpret(final_statistics["metric_summaries"]),
        "artifacts": {
            "checkpoint_csv": str(checkpoint_csv),
            "history_csv": str(history_csv),
            "plots": plot_paths,
        },
    }
    summary_path = output_dir / SUMMARY_FILENAME
    summary["artifacts"]["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def format_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        f"Training run: {summary['run_dir']}",
        f"Checkpoints summarized: {summary['checkpoint_count']}",
    ]
    lines.extend(f"- {line}" for line in summary.get("interpretation", []))
    lines.append(f"Summary: {summary['artifacts']['summary']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Hugging Face trainer history without loading model weights."
    )
    parser.add_argument("run_path", type=Path, help="Training run directory or checkpoint directory.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG graph generation.")
    args = parser.parse_args()
    summary = summarize_training_run(args.run_path, make_plots=not args.no_plots)
    print(format_summary(summary))


if __name__ == "__main__":
    main()
