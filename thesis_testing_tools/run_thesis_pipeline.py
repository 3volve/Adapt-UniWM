"""Run the fixed source -> Habitat adaptation -> source thesis pilot.

This is deliberately a personal, hard-coded experiment driver.  It mirrors the
two online evaluation shell scripts, but keeps each torchrun in the foreground
and writes the cross-stage thesis aggregates after the individual runner
artifacts have been persisted.  An existing pipeline run can also supply the
source-pre reference for an additional Habitat -> source-post follow-up.
"""

from __future__ import annotations

import argparse, csv, json, os, subprocess, time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping


REPO_ROOT = Path(__file__).resolve().parent.parent
REPLAY_CONFIG = REPO_ROOT / "cfg" / "replay_uniwm_cfg.yaml"
HABITAT_CONFIG = REPO_ROOT / "cfg" / "habitat_uniwm_cfg.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output"
BASE_CHECKPOINT = REPO_ROOT / "checkpoints" / "base_ckpt"

SOURCE_DATA_IDS = "go_stanford,sacson,scand,recon"
HABITAT_DATA_ID = "habitat"
SOURCE_EPISODES = 2
HABITAT_EPISODES = 12
REPLAY_PORT = 20002
HABITAT_PORT = 20001
ENV_OVERRIDES = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "NCCL_P2P_DISABLE": "1",
}


MetricCalculator = Callable[[Path, Path], Mapping[str, float]]
SubprocessRunner = Callable[..., Any]


class AlexNetVisualMetricCalculator:
    """Lazy CPU calculator so importing this helper does not require ML deps."""

    def __init__(self) -> None:
        self._torch: Any = None
        self._ssim: Any = None
        self._lpips_model: Any = None

    def _load(self) -> None:
        if self._lpips_model is not None:
            return

        import lpips
        import torch
        from pytorch_msssim import ssim

        self._torch = torch
        self._ssim = ssim
        self._lpips_model = lpips.LPIPS(net="alex").to("cpu").eval()

    def __call__(self, prediction_path: Path, real_path: Path) -> Mapping[str, float]:
        self._load()

        import numpy as np
        from PIL import Image

        torch = self._torch

        def load_image(path: Path) -> Any:
            image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
            return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0) / 255.0

        prediction = load_image(prediction_path)
        real = load_image(real_path)

        with torch.inference_mode():
            mae = torch.mean(torch.abs(prediction - real)).item()
            ssim_value = self._ssim(
                prediction,
                real,
                data_range=1.0,
                size_average=True,
            ).item()
            lpips_value = self._lpips_model(
                prediction * 2.0 - 1.0,
                real * 2.0 - 1.0,
            ).item()

        return {
            "mae": float(mae),
            "ssim": float(ssim_value),
            "lpips": float(lpips_value),
        }


def build_torchrun_command(
    config_path: Path,
    data_id: str,
    run_dir: Path,
    num_episodes: int,
    master_port: int,
) -> list[str]:
    return [
        "torchrun",
        "--nproc_per_node=1",
        f"--master_port={master_port}",
        "./uniwm_episode_runner.py",
        "--config_path",
        str(config_path),
        "--data_id",
        data_id,
        "--run_dir",
        str(run_dir),
        "--num_episodes",
        str(num_episodes),
    ]


def stage_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(ENV_OVERRIDES)
    return environment


def run_stage(
    name: str,
    command: list[str],
    run_dir: Path,
    config_path: Path,
    data_id: str,
    subprocess_runner: SubprocessRunner | None = None,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    runner = subprocess.run if subprocess_runner is None else subprocess_runner
    started_at = datetime.now().astimezone()
    started = time.perf_counter()

    runner(
        command,
        check=True,
        cwd=repo_root,
        env=stage_environment(),
    )

    finished_at = datetime.now().astimezone()
    return {
        "name": name,
        "status": "completed",
        "data_id": data_id,
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "command": list(command),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": time.perf_counter() - started,
    }


def write_post_replay_config(
    source_config: Path,
    destination: Path,
    checkpoint: Path,
) -> None:
    lines = source_config.read_text(encoding="utf-8").splitlines(keepends=True)
    replacement = json.dumps(str(checkpoint))
    model_checkpoint_line = next(
        index
        for index, line in enumerate(lines)
        if line.lstrip().startswith("model_ckpt:")
    )
    indentation = lines[model_checkpoint_line][
        : len(lines[model_checkpoint_line]) - len(lines[model_checkpoint_line].lstrip())
    ]
    lines[model_checkpoint_line] = f"{indentation}model_ckpt: {replacement}\n"
    destination.write_text("".join(lines), encoding="utf-8")


def write_habitat_fixed_action_config(
    source_config: Path,
    destination: Path,
    action_run_dir: Path,
) -> None:
    lines = source_config.read_text(encoding="utf-8").splitlines(keepends=True)
    replacement = json.dumps(str(action_run_dir.resolve()))
    fixed_action_line = next(
        index
        for index, line in enumerate(lines)
        if line.lstrip().startswith("fixed_action_run_dir:")
    )
    indentation = lines[fixed_action_line][
        : len(lines[fixed_action_line]) - len(lines[fixed_action_line].lstrip())
    ]
    lines[fixed_action_line] = (
        f"{indentation}fixed_action_run_dir: {replacement}\n"
    )

    destination.write_text("".join(lines), encoding="utf-8")


def _nested(value: Mapping[str, Any] | None, *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _mean(values: list[Any]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _last(values: list[Any]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return present[-1] if present else None


def _step_values(steps: list[dict[str, Any]], *keys: str) -> list[Any]:
    return [_nested(step, *keys) for step in steps]


def _episode_visual_metrics(
    stage_dir: Path,
    episode_log: dict[str, Any],
    prediction_suffix: str,
    metric_calculator: MetricCalculator,
) -> dict[str, Any]:
    metric_values: dict[str, list[float]] = {
        "mae": [],
        "ssim": [],
        "lpips": [],
    }
    steps = episode_log["steps"]
    available_count = 0
    missing_count = 0

    for step in steps:
        image_stem = (
            stage_dir
            / f"episode_{episode_log['episode_id']}"
            / str(step["route_id"])
            / str(step["route_idx"])
        )
        prediction_path = image_stem.with_name(
            f"{image_stem.name}_{prediction_suffix}.png"
        )
        real_path = image_stem.with_name(f"{image_stem.name}_real.png")

        if not prediction_path.is_file():
            missing_count += 1
            continue

        scores = metric_calculator(prediction_path, real_path)
        available_count += 1
        for metric_name in metric_values:
            metric_values[metric_name].append(float(scores[metric_name]))

    return {
        "prediction_total_count": len(steps),
        "prediction_available_count": available_count,
        "prediction_missing_count": missing_count,
        "mae": _mean(metric_values["mae"]),
        "ssim": _mean(metric_values["ssim"]),
        "lpips": _mean(metric_values["lpips"]),
    }


def _episode_diagnostic_metrics(episode_log: dict[str, Any]) -> dict[str, Any]:
    steps: list[dict[str, Any]] = episode_log["steps"]
    final_metrics = (
        _nested(steps[-1], "env_info", "metrics")
        if steps
        else {}
    )
    if not isinstance(final_metrics, Mapping):
        final_metrics = {}

    collisions = [
        step.get("collision", _nested(step, "env_info", "collision"))
        for step in steps
    ]

    global_ach = _step_values(steps, "modulator_state", "gl_ach")
    global_ne = _step_values(steps, "modulator_state", "gl_ne_visual")
    step_ach = _step_values(steps, "modulator_state", "st_ach")
    step_ne = _step_values(steps, "modulator_state", "st_ne_visual")

    return {
        "success": final_metrics.get("success"),
        "spl": final_metrics.get("spl"),
        "soft_spl": final_metrics.get("soft_spl"),
        "final_distance_to_goal": final_metrics.get("distance_to_goal"),
        "runner_transition_count": len(steps),
        "habitat_primitive_step_count": final_metrics.get("num_steps"),
        "collision_transition_count": sum(bool(value) for value in collisions),
        "replan_count": sum(bool(step["replanned"]) for step in steps),
        "context_familiarity_mean": _mean(
            [step.get("context_familiarity") for step in steps]
        ),
        "context_stability_mean": _mean(
            [step.get("context_stability") for step in steps]
        ),
        "global_ach_mean": _mean(global_ach),
        "global_ach_last": _last(global_ach),
        "global_ne_visual_mean": _mean(global_ne),
        "global_ne_visual_last": _last(global_ne),
        "step_ach_mean": _mean(step_ach),
        "step_ne_visual_mean": _mean(step_ne),
        "update_weight_mean": _mean(
            _step_values(
                steps,
                "modulator_state",
                "step_events",
                "update_weight",
            )
        ),
        "effective_learning_rate_mean": _mean(
            _step_values(steps, "training_logs", "final_lr")
        ),
        "optimizer_learning_rate_mean": _mean(
            _step_values(steps, "training_logs", "optimizer_lr")
        ),
        "training_loss_mean": _mean(
            _step_values(steps, "training_logs", "base_loss")
        ),
        "action_entropy_mean": _mean(
            _step_values(steps, "step_info", "act_entropy")
        ),
        "visualization_entropy_mean": _mean(
            _step_values(steps, "step_info", "viz_entropy")
        ),
        "forced_eval_visualization_entropy_mean": _mean(
            _step_values(steps, "eval_logs", "viz_entropy")
        ),
    }


def collect_stage_episode_metrics(
    stage_dir: Path,
    data_id: str,
    prediction_suffix: str,
    metric_calculator: MetricCalculator,
) -> list[dict[str, Any]]:
    with (stage_dir / "episode_logs.json").open("r", encoding="utf-8") as handle:
        episode_logs: list[dict[str, Any]] = json.load(handle)

    rows: list[dict[str, Any]] = []
    for episode_log in episode_logs:
        row: dict[str, Any] = {
            "data_id": episode_log.get("data_id", data_id),
            "episode_index": episode_log["episode_index"],
            "episode_id": episode_log["episode_id"],
            "adapter_source_mode": episode_log["adapter_source_mode"],
            "termination_reason": episode_log["termination_reason"],
        }
        row.update(_episode_diagnostic_metrics(episode_log))
        row.update(
            _episode_visual_metrics(
                stage_dir,
                episode_log,
                prediction_suffix,
                metric_calculator,
            )
        )
        rows.append(row)

    _write_csv(stage_dir / "thesis_episode_metrics.csv", rows)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def load_source_episode_metrics(path: Path) -> list[dict[str, Any]]:
    """Load only the existing source-pre fields used for retention comparison."""
    count_fields = (
        "prediction_total_count",
        "prediction_available_count",
        "prediction_missing_count",
    )
    metric_fields = ("mae", "ssim", "lpips")

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for saved_row in csv.DictReader(handle):
            row: dict[str, Any] = {
                "data_id": saved_row["data_id"],
                "episode_id": saved_row["episode_id"],
            }
            row.update(
                {field: int(saved_row[field]) for field in count_fields}
            )
            row.update(
                {
                    field: (
                        None
                        if saved_row[field] == ""
                        else float(saved_row[field])
                    )
                    for field in metric_fields
                }
            )
            rows.append(row)

    return rows


def load_episode_metrics(path: Path) -> list[dict[str, Any]]:
    """Load an existing stage-local thesis metrics CSV for run summaries."""
    string_fields = {
        "data_id",
        "episode_id",
        "adapter_source_mode",
        "termination_reason",
    }

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for saved_row in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for field, value in saved_row.items():
                if field in string_fields:
                    row[field] = value
                elif value == "":
                    row[field] = None
                else:
                    try:
                        row[field] = float(value)
                    except ValueError:
                        row[field] = value
            rows.append(row)

    return rows


def _difference(after: Any, before: Any) -> float | None:
    if before is None or after is None:
        return None
    return float(after) - float(before)


def compare_source_episodes(
    source_pre_rows: list[dict[str, Any]],
    source_post_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pre_by_episode = {
        (row["data_id"], row["episode_id"]): row for row in source_pre_rows
    }
    post_by_episode = {
        (row["data_id"], row["episode_id"]): row for row in source_post_rows
    }

    comparisons: list[dict[str, Any]] = []
    for key in pre_by_episode.keys() & post_by_episode.keys():
        pre = pre_by_episode[key]
        post = post_by_episode[key]
        row: dict[str, Any] = {
            "data_id": key[0],
            "episode_id": key[1],
        }

        for count_name in (
            "prediction_total_count",
            "prediction_available_count",
            "prediction_missing_count",
        ):
            row[f"pre_{count_name}"] = pre[count_name]
            row[f"post_{count_name}"] = post[count_name]
            row[f"{count_name}_post_minus_pre"] = (
                int(post[count_name]) - int(pre[count_name])
            )

        for metric_name in ("mae", "ssim", "lpips"):
            pre_value = pre[metric_name]
            post_value = post[metric_name]
            post_minus_pre = _difference(post_value, pre_value)
            row[f"pre_{metric_name}"] = pre_value
            row[f"post_{metric_name}"] = post_value
            row[f"{metric_name}_post_minus_pre"] = post_minus_pre
            row[f"{metric_name}_retention_loss"] = (
                None
                if post_minus_pre is None
                else (
                    -post_minus_pre
                    if metric_name == "ssim"
                    else post_minus_pre
                )
            )

        comparisons.append(row)

    comparisons.sort(key=lambda row: (str(row["data_id"]), str(row["episode_id"])))
    return comparisons


def summarize_source_retention(
    comparison_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "matched_episode_count": len(comparison_rows),
    }

    for metric_name in ("mae", "ssim", "lpips"):
        summary[f"pre_{metric_name}_episode_mean"] = _mean(
            [row[f"pre_{metric_name}"] for row in comparison_rows]
        )
        summary[f"post_{metric_name}_episode_mean"] = _mean(
            [row[f"post_{metric_name}"] for row in comparison_rows]
        )
        summary[f"{metric_name}_post_minus_pre_episode_mean"] = _mean(
            [row[f"{metric_name}_post_minus_pre"] for row in comparison_rows]
        )
        summary[f"{metric_name}_retention_loss_episode_mean"] = _mean(
            [row[f"{metric_name}_retention_loss"] for row in comparison_rows]
        )

    for phase in ("pre", "post"):
        for count_name in (
            "prediction_total_count",
            "prediction_available_count",
            "prediction_missing_count",
        ):
            summary[f"{phase}_{count_name}_total"] = sum(
                int(row[f"{phase}_{count_name}"]) for row in comparison_rows
            )

    return summary


def summarize_habitat(habitat_rows: list[dict[str, Any]]) -> dict[str, Any]:
    mean_fields = (
        "success",
        "spl",
        "soft_spl",
        "final_distance_to_goal",
        "runner_transition_count",
        "habitat_primitive_step_count",
        "collision_transition_count",
        "replan_count",
        "mae",
        "ssim",
        "lpips",
        "context_familiarity_mean",
        "context_stability_mean",
        "global_ach_mean",
        "global_ach_last",
        "global_ne_visual_mean",
        "global_ne_visual_last",
        "step_ach_mean",
        "step_ne_visual_mean",
        "update_weight_mean",
        "effective_learning_rate_mean",
        "optimizer_learning_rate_mean",
        "training_loss_mean",
        "action_entropy_mean",
        "visualization_entropy_mean",
    )
    total_fields = (
        "runner_transition_count",
        "habitat_primitive_step_count",
        "collision_transition_count",
        "replan_count",
        "prediction_total_count",
        "prediction_available_count",
        "prediction_missing_count",
    )

    return {
        "episode_count": len(habitat_rows),
        "episode_metrics": habitat_rows,
        "episode_first_means": {
            field: _mean([row[field] for row in habitat_rows])
            for field in mean_fields
        },
        "totals": {
            field: sum(
                float(row[field])
                for row in habitat_rows
                if row[field] is not None
            )
            for field in total_fields
        },
    }


def build_pipeline_summary(
    pipeline_dir: Path,
    result_dir: Path,
    source_pre_dir: Path,
    stage_records: list[dict[str, Any]],
    source_pre_rows: list[dict[str, Any]],
    habitat_rows: list[dict[str, Any]],
    source_post_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    duration_seconds: float,
    run_type: str | None = None,
) -> dict[str, Any]:
    return {
        "status": "completed",
        "run_type": run_type or (
            "full_pipeline"
            if pipeline_dir == result_dir
            else "habitat_source_post_followup"
        ),
        "pipeline_dir": str(pipeline_dir),
        "result_dir": str(result_dir),
        "source_pre_reference": str(source_pre_dir),
        "completed_at": datetime.now().astimezone().isoformat(),
        "duration_seconds": duration_seconds,
        "stages": stage_records,
        "artifacts": {
            "source_pre_metrics": str(
                source_pre_dir / "thesis_episode_metrics.csv"
            ),
            "habitat_metrics": str(
                result_dir / "habitat" / "thesis_episode_metrics.csv"
            ),
            "source_post_metrics": str(
                result_dir / "source_post" / "thesis_episode_metrics.csv"
            ),
            "source_episode_comparison": str(
                result_dir / "source_episode_comparison.csv"
            ),
            "post_replay_config": str(
                result_dir / "source_post_config.yaml"
            ),
            "habitat_checkpoint": str(
                result_dir / "habitat" / "final_ckpt"
            ),
        },
        "stage_episode_counts": {
            "source_pre": len(source_pre_rows),
            "habitat": len(habitat_rows),
            "source_post": len(source_post_rows),
        },
        "habitat": summarize_habitat(habitat_rows),
        "source_retention": summarize_source_retention(comparison_rows),
    }


def _planned_stage(
    name: str,
    data_id: str,
    config_path: Path,
    run_dir: Path,
    command: list[str],
    input_checkpoint: Path,
    output_checkpoint: Path | None = None,
) -> dict[str, Any]:
    stage = {
        "name": name,
        "data_id": data_id,
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "command": command,
        "input_checkpoint_path": str(input_checkpoint),
    }
    if output_checkpoint is not None:
        stage["output_checkpoint_path"] = str(output_checkpoint)
    return stage


def run_pipeline(
    *,
    existing_run: Path | None = None,
    source_post_checkpoint: Path | None = None,
    habitat_action_run: Path | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    timestamp: str | None = None,
    subprocess_runner: SubprocessRunner | None = None,
    metric_calculator: MetricCalculator | None = None,
) -> Path:
    timestamp = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        if timestamp is None
        else timestamp
    )
    post_only = source_post_checkpoint is not None
    if post_only:
        final_checkpoint = Path(source_post_checkpoint).resolve()
        habitat_dir = final_checkpoint.parent
        result_dir = habitat_dir.parent
        pipeline_dir = result_dir.parent.parent
    elif existing_run is None:
        pipeline_dir = Path(output_root) / f"thesis_pipeline_{timestamp}"
        result_dir = pipeline_dir
    else:
        pipeline_dir = Path(existing_run).resolve()
        result_dir = (
            pipeline_dir
            / "followups"
            / f"followup_{timestamp}"
        )

    source_pre_dir = pipeline_dir / "source_pre"
    habitat_dir = result_dir / "habitat"
    source_post_dir = result_dir / "source_post"
    post_config = result_dir / "source_post_config.yaml"
    habitat_config = HABITAT_CONFIG
    if not post_only:
        final_checkpoint = habitat_dir / "final_ckpt"
        result_dir.mkdir(parents=True, exist_ok=False)

        if habitat_action_run is not None:
            habitat_config = result_dir / "habitat_fixed_actions_config.yaml"
            write_habitat_fixed_action_config(
                HABITAT_CONFIG,
                habitat_config,
                habitat_action_run,
            )

    source_pre_command = (
        build_torchrun_command(
            REPLAY_CONFIG,
            SOURCE_DATA_IDS,
            source_pre_dir,
            SOURCE_EPISODES,
            REPLAY_PORT,
        )
        if existing_run is None and not post_only
        else None
    )
    habitat_command = (
        None
        if post_only
        else build_torchrun_command(
            habitat_config,
            HABITAT_DATA_ID,
            habitat_dir,
            HABITAT_EPISODES,
            HABITAT_PORT,
        )
    )
    source_post_command = build_torchrun_command(
        post_config,
        SOURCE_DATA_IDS,
        source_post_dir,
        SOURCE_EPISODES,
        REPLAY_PORT,
    )

    planned_stages = []
    if source_pre_command is not None:
        planned_stages.append(
            _planned_stage(
                "source_pre",
                SOURCE_DATA_IDS,
                REPLAY_CONFIG,
                source_pre_dir,
                source_pre_command,
                BASE_CHECKPOINT,
            )
        )
    if habitat_command is not None:
        planned_stages.append(
            _planned_stage(
                "habitat",
                HABITAT_DATA_ID,
                habitat_config,
                habitat_dir,
                habitat_command,
                BASE_CHECKPOINT,
                final_checkpoint,
            )
        )
    planned_stages.append(
        _planned_stage(
            "source_post",
            SOURCE_DATA_IDS,
            post_config,
            source_post_dir,
            source_post_command,
            final_checkpoint,
        )
    )

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "run_type": (
            "source_post_from_existing_checkpoint"
            if post_only
            else (
                "full_pipeline"
                if existing_run is None
                else "habitat_source_post_followup"
            )
        ),
        "pipeline_dir": str(pipeline_dir),
        "result_dir": str(result_dir),
        "source_pre_reference": str(source_pre_dir),
        "source_data_ids": SOURCE_DATA_IDS.split(","),
        "source_episodes_per_data_id": SOURCE_EPISODES,
        "source_episode_count": (
            len(SOURCE_DATA_IDS.split(",")) * SOURCE_EPISODES
        ),
        "environment_overrides": ENV_OVERRIDES,
        "stages": planned_stages,
        "checkpoint_flow": {
            "base_checkpoint": str(BASE_CHECKPOINT),
            "habitat_output_checkpoint": str(final_checkpoint),
            "source_post_input_checkpoint": str(final_checkpoint),
        },
        "source_post_config": str(post_config),
        "habitat_action_reference_run": (
            None
            if habitat_action_run is None
            else str(habitat_action_run.resolve())
        ),
    }
    manifest_path = result_dir / (
        "source_post_manifest.json"
        if post_only
        else "pipeline_manifest.json"
    )
    with manifest_path.open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    calculate_metrics = (
        AlexNetVisualMetricCalculator()
        if metric_calculator is None
        else metric_calculator
    )
    pipeline_started = time.perf_counter()
    stage_records: list[dict[str, Any]] = []

    if source_pre_command is None:
        source_pre_metrics_path = (
            source_pre_dir / "thesis_episode_metrics.csv"
        )
        if source_pre_metrics_path.is_file():
            source_pre_rows = load_source_episode_metrics(
                source_pre_metrics_path
            )
        else:
            source_pre_rows = collect_stage_episode_metrics(
                source_pre_dir,
                SOURCE_DATA_IDS,
                "eval",
                calculate_metrics,
            )
    else:
        source_pre_record = run_stage(
            "source_pre",
            source_pre_command,
            source_pre_dir,
            REPLAY_CONFIG,
            SOURCE_DATA_IDS,
            subprocess_runner,
        )
        source_pre_record["input_checkpoint_path"] = str(BASE_CHECKPOINT)
        stage_records.append(source_pre_record)
        source_pre_rows = collect_stage_episode_metrics(
            source_pre_dir,
            SOURCE_DATA_IDS,
            "eval",
            calculate_metrics,
        )
        source_pre_record["thesis_episode_metrics"] = str(
            source_pre_dir / "thesis_episode_metrics.csv"
        )

    if habitat_command is None:
        habitat_metrics_path = habitat_dir / "thesis_episode_metrics.csv"
        if habitat_metrics_path.is_file():
            habitat_rows = load_episode_metrics(habitat_metrics_path)
        else:
            habitat_rows = collect_stage_episode_metrics(
                habitat_dir,
                HABITAT_DATA_ID,
                "pred",
                calculate_metrics,
            )
    else:
        habitat_record = run_stage(
            "habitat",
            habitat_command,
            habitat_dir,
            habitat_config,
            HABITAT_DATA_ID,
            subprocess_runner,
        )
        habitat_record["input_checkpoint_path"] = str(BASE_CHECKPOINT)
        habitat_record["output_checkpoint_path"] = str(final_checkpoint)
        stage_records.append(habitat_record)
        habitat_rows = collect_stage_episode_metrics(
            habitat_dir,
            HABITAT_DATA_ID,
            "pred",
            calculate_metrics,
        )
        habitat_record["thesis_episode_metrics"] = str(
            habitat_dir / "thesis_episode_metrics.csv"
        )

    write_post_replay_config(REPLAY_CONFIG, post_config, final_checkpoint)

    source_post_record = run_stage(
        "source_post",
        source_post_command,
        source_post_dir,
        post_config,
        SOURCE_DATA_IDS,
        subprocess_runner,
    )
    source_post_record["input_checkpoint_path"] = str(final_checkpoint)
    stage_records.append(source_post_record)
    source_post_rows = collect_stage_episode_metrics(
        source_post_dir,
        SOURCE_DATA_IDS,
        "eval",
        calculate_metrics,
    )
    source_post_record["thesis_episode_metrics"] = str(
        source_post_dir / "thesis_episode_metrics.csv"
    )

    comparison_rows = compare_source_episodes(
        source_pre_rows,
        source_post_rows,
    )
    _write_csv(
        result_dir / "source_episode_comparison.csv",
        comparison_rows,
    )

    summary = build_pipeline_summary(
        pipeline_dir,
        result_dir,
        source_pre_dir,
        stage_records,
        source_pre_rows,
        habitat_rows,
        source_post_rows,
        comparison_rows,
        time.perf_counter() - pipeline_started,
        (
            "source_post_from_existing_checkpoint"
            if post_only
            else None
        ),
    )
    with (result_dir / "pipeline_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    return result_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--existing-run",
        type=Path,
        help="Reuse source_pre from an existing thesis pipeline run.",
    )
    parser.add_argument(
        "--source-post-checkpoint",
        type=Path,
        help="Run only source-post using an existing Habitat final_ckpt.",
    )
    parser.add_argument(
        "--habitat-cfg",
        type=Path,
        help="Specify a specific config to use for the run",
    )
    parser.add_argument(
        "--habitat-port",
        type=int,
    )
    parser.add_argument(
        "--habitat-action-run",
        type=Path,
        help=(
            "Force Habitat actions from episode_logs.json in an existing "
            "Habitat run directory."
        ),
    )
    args = parser.parse_args()
    
    if args.habitat_cfg is not None:
        global HABITAT_CONFIG 
        HABITAT_CONFIG = REPO_ROOT / "cfg" / args.habitat_cfg
    
    if args.habitat_port is not None:
        global HABITAT_PORT, REPLAY_PORT
        HABITAT_PORT = args.habitat_port
        REPLAY_PORT = args.habitat_port + 1

    result_dir = run_pipeline(
        existing_run=args.existing_run,
        source_post_checkpoint=args.source_post_checkpoint,
        habitat_action_run=args.habitat_action_run,
    )
    print(f"[THESIS PIPELINE] Finished: {result_dir}")


if __name__ == "__main__":
    main()
