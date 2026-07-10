from __future__ import annotations

import csv
import json
import numpy as np
import torch
from pathlib import Path
from typing import Any
from PIL import Image

# This isn't intended as the final metric collection, this is only a temporary demonstration file to start seeing some outputs.

def save_runner_logs(
    logs: list[dict[str, Any]],
    output_dir: Path
):
    def convert(value: Any, path_parts: tuple[str | int, ...]) -> Any:
        # Skip / summarize any potentially large objects instead of serializing them:
        if isinstance(value, Image.Image):
            return {
                "__omitted__": "PIL.Image",
                "mode": value.mode,
                "size": value.size,
            }

        if isinstance(value, np.ndarray):
            return {
                "__omitted__": "np.ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }

        if isinstance(value, np.generic):
            return value.item()

        if torch.is_tensor(value):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return {
                "__omitted__": "torch.Tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }

        if isinstance(value, bytes):
            return {
                "__omitted__": "bytes",
                "length": len(value),
            }
        
        # Now start adding the normal objects
        if isinstance(value, dict):
            return {
                key: convert(child, (*path_parts, key))
                for key, child in value.items()
            }

        if isinstance(value, list):
            return [
                convert(child, (*path_parts, index))
                for index, child in enumerate(value)
            ]
        
        return value

    serialized_logs = convert(logs, ("logs",))

    with (output_dir / "episode_logs.json").open("w", encoding="utf-8") as handle:
        json.dump(serialized_logs, handle, indent=2)

    _write_csv(output_dir / "episode_metrics.csv", _build_episode_metric_rows(logs))
    _write_csv(output_dir / "step_metrics.csv", _build_step_metric_rows(logs))


def _build_episode_metric_rows(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for episode_log in logs:
        row: dict[str, Any] = {
            "episode_index": episode_log["episode_index"],
            "episode_id": episode_log["episode_id"],
            "adapter_source_mode": episode_log["adapter_source_mode"],
            "termination_reason": episode_log["termination_reason"],
        }

        row.update(_flatten_numeric_fields(episode_log["reset_info"], "reset_info"))
        row.update(_flatten_numeric_fields(episode_log["wrapper_reset_state"], "wrapper_reset_state"))
        row.update(_flatten_numeric_fields(episode_log["final_wrapper_state"], "final_wrapper_state"))

        steps = episode_log["steps"]
        divergences = [step["divergence"] for step in steps]
        replans = [int(step["replanned"]) for step in steps]

        row["step_count"] = len(steps)
        row["replanned_count"] = sum(replans)
        row["divergence_mean"] = sum(divergences) / len(divergences) if divergences else 0.0
        row["divergence_min"] = min(divergences) if divergences else 0.0
        row["divergence_max"] = max(divergences) if divergences else 0.0
        row["divergence_last"] = divergences[-1] if divergences else 0.0

        final_env_info = steps[-1]["env_info"] if steps else {}
        final_metrics = final_env_info.get("metrics", {})
        row.update(_flatten_numeric_fields(final_env_info, "final_env_info"))
        row.update(_flatten_numeric_fields(final_metrics, "final_metrics"))

        row["route_count"] = len(episode_log["routes"])
        row["transition_count"] = len(steps)

        rows.append(row)

    return rows


def _build_step_metric_rows(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for episode_log in logs:
        for step in episode_log["steps"]:
            row: dict[str, Any] = {
                "episode_index": episode_log["episode_index"],
                "episode_id": episode_log["episode_id"],
                "adapter_source_mode": episode_log["adapter_source_mode"],
                "termination_reason": episode_log["termination_reason"],
                "step_idx": step["step_idx"],
                "route_id": step["route_id"],
                "route_idx": step["route_idx"],
                "action": step["action"],
                "context_familiarity": step["context_familiarity"],
                "context_stability": step["context_stability"],
                "viz_used_memory": int(step["viz_used_memory"]),
                "divergence": step["divergence"],
                "replanned": int(step["replanned"]),
                "replan_reason": step["replan_reason"],
                "wrapper_requested_stop": int(step["wrapper_requested_stop"]),
            }

            row.update(_flatten_numeric_fields(step["modulator_state"], "modulator"))
            row.update(_flatten_numeric_fields(step["training_logs"], "training"))
            row.update(_flatten_numeric_fields(step["eval_logs"], "eval_logs"))
            row.update(_flatten_numeric_fields(step["env_info"], "env_info"))
            row.update(_flatten_numeric_fields(step["env_info"].get("metrics", {}), "metrics"))

            rows.append(row)

    return rows


def _flatten_numeric_fields(value: Any, prefix: str) -> dict[str, int | float]:
    if isinstance(value, dict):
        flattened: dict[str, int | float] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_numeric_fields(child, child_prefix))
        return flattened

    if isinstance(value, list):
        flattened: dict[str, int | float] = {}
        for index, child in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            flattened.update(_flatten_numeric_fields(child, child_prefix))
        return flattened

    if isinstance(value, bool):
        return {prefix: int(value)}

    if isinstance(value, np.generic):
        return {prefix: value.item()}

    if torch.is_tensor(value) and value.numel() == 1:
        return {prefix: value.detach().cpu().item()}

    if isinstance(value, int | float):
        return {prefix: value}

    return {}


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
