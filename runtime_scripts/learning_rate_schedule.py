from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ALIGNED_SCHEDULE_FILENAME = "learning_rate_schedule.json"
SHUFFLED_SCHEDULE_FILENAME = "learning_rate_schedule_shuffled.json"
SCHEDULE_SCHEMA_VERSION = 1


class LearningRateSchedule:
    """Record or replay transition-aligned learning-rate scalars."""

    def __init__(
        self,
        config: dict[str, Any] | bool | None,
        *,
        output_dir: str | Path,
        initial_lr: float | None,
    ) -> None:
        self.mode: str | None = None
        self._output_dir = Path(output_dir)
        self._initial_lr = initial_lr
        self._shuffled_seed: int | None = None
        self._entries: list[dict[str, Any]] = []
        self._entries_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
        self._consumed_keys: set[tuple[str, str, int]] = set()

        if config is None or config is False:
            return
        if not isinstance(config, dict):
            raise ValueError(
                "wrapper.learning_rate_schedule must be false or a mapping"
            )

        mode = config.get("mode")
        if mode not in ("record", "replay"):
            raise ValueError(
                "wrapper.learning_rate_schedule.mode must be 'record' or 'replay'"
            )
        self.mode = str(mode)
        self._initial_lr = self._validate_learning_rate(
            initial_lr,
            "engine.training.hyper_params.initial_lr",
        )

        if self.is_recording:
            shuffled_seed = config.get("shuffled_seed")
            if not isinstance(shuffled_seed, int) or isinstance(shuffled_seed, bool):
                raise ValueError(
                    "record mode requires integer learning_rate_schedule.shuffled_seed"
                )
            self._shuffled_seed = shuffled_seed
            return

        input_path = config.get("input_path")
        if not isinstance(input_path, str) or not input_path.strip():
            raise ValueError(
                "replay mode requires non-empty learning_rate_schedule.input_path"
            )
        shuffled = config.get("shuffled", False)
        if not isinstance(shuffled, bool):
            raise ValueError("learning_rate_schedule.shuffled must be a boolean")

        input_dir = Path(input_path)
        if not input_dir.is_absolute():
            input_dir = Path(__file__).resolve().parent.parent / input_dir
        filename = (
            SHUFFLED_SCHEDULE_FILENAME if shuffled else ALIGNED_SCHEDULE_FILENAME
        )
        self._load(input_dir / filename, expected_shuffled=shuffled)

    @property
    def is_recording(self) -> bool:
        return self.mode == "record"

    @property
    def is_replaying(self) -> bool:
        return self.mode == "replay"

    def record_transition(
        self,
        *,
        data_id: str,
        episode_id: str,
        step_idx: int,
        action: str,
        collision: bool,
        update_eligible: bool,
        skip_reason: str | None,
        lr_scalar: float | None,
    ) -> None:
        if not self.is_recording:
            return

        key = self._key(data_id, episode_id, step_idx)
        if key in self._entries_by_key:
            raise ValueError(f"Duplicate recorded schedule transition {key}")

        if update_eligible:
            scalar = self._validate_scalar(lr_scalar, key)
            effective_lr = self._initial_lr * scalar
            if skip_reason is not None:
                raise ValueError(
                    f"Eligible schedule transition {key} cannot have skip_reason={skip_reason!r}"
                )
        else:
            if lr_scalar is not None:
                raise ValueError(
                    f"Ineligible schedule transition {key} cannot have an LR scalar"
                )
            scalar = None
            effective_lr = None
            if skip_reason is None:
                raise ValueError(
                    f"Ineligible schedule transition {key} requires a skip reason"
                )

        entry = {
            "data_id": str(data_id),
            "episode_id": str(episode_id),
            "step_idx": int(step_idx),
            "action": str(action),
            "collision": bool(collision),
            "update_eligible": bool(update_eligible),
            "skip_reason": skip_reason,
            "lr_scalar": scalar,
            "effective_learning_rate": effective_lr,
        }
        self._entries.append(entry)
        self._entries_by_key[key] = entry

    def replay_transition(
        self,
        *,
        data_id: str,
        episode_id: str,
        step_idx: int,
        action: str,
        collision: bool,
        update_eligible: bool,
        skip_reason: str | None,
    ) -> float | None:
        if not self.is_replaying:
            return None

        key = self._key(data_id, episode_id, step_idx)
        if key in self._consumed_keys:
            raise ValueError(f"Schedule transition {key} was consumed more than once")
        if key not in self._entries_by_key:
            raise ValueError(f"No schedule entry found for transition {key}")

        entry = self._entries_by_key[key]
        expected = {
            "action": str(action),
            "collision": bool(collision),
            "update_eligible": bool(update_eligible),
            "skip_reason": skip_reason,
        }
        for field, actual_value in expected.items():
            if entry[field] != actual_value:
                raise ValueError(
                    f"Schedule mismatch for transition {key}: {field} is "
                    f"{actual_value!r}, expected {entry[field]!r}"
                )

        self._consumed_keys.add(key)
        scalar = entry["lr_scalar"]
        if update_eligible:
            return float(scalar)
        if scalar is not None:
            raise AssertionError(
                f"Validated ineligible schedule transition {key} has an LR scalar"
            )
        return None

    def save(self) -> None:
        if not self.is_recording:
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._write(
            self._output_dir / ALIGNED_SCHEDULE_FILENAME,
            entries=self._entries,
            shuffled=False,
            shuffled_seed=None,
        )
        self._write(
            self._output_dir / SHUFFLED_SCHEDULE_FILENAME,
            entries=self._within_episode_shuffled_entries(),
            shuffled=True,
            shuffled_seed=self._shuffled_seed,
        )

    def assert_fully_consumed(self) -> None:
        if not self.is_replaying:
            return
        unconsumed = set(self._entries_by_key) - self._consumed_keys
        if unconsumed:
            examples = sorted(unconsumed)[:5]
            raise ValueError(
                f"Replay finished with {len(unconsumed)} unconsumed schedule entries; "
                f"first entries: {examples}"
            )

    def _load(self, path: Path, *, expected_shuffled: bool) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Learning-rate schedule not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            artifact = json.load(handle)
        if not isinstance(artifact, dict):
            raise ValueError(f"Learning-rate schedule {path} must contain a mapping")
        if artifact.get("schema_version") != SCHEDULE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schedule schema_version in {path}: "
                f"{artifact.get('schema_version')!r}"
            )
        if artifact.get("shuffled") is not expected_shuffled:
            raise ValueError(
                f"Schedule {path} has shuffled={artifact.get('shuffled')!r}, "
                f"expected {expected_shuffled}"
            )

        artifact_lr = self._validate_learning_rate(
            artifact.get("base_learning_rate"),
            f"{path}.base_learning_rate",
        )
        if artifact_lr != self._initial_lr:
            raise ValueError(
                f"Schedule base learning rate {artifact_lr!r} does not match "
                f"configured initial_lr {self._initial_lr!r}"
            )

        entries = artifact.get("entries")
        if not isinstance(entries, list):
            raise ValueError(f"Schedule {path}.entries must be a list")
        for index, raw_entry in enumerate(entries):
            entry = self._validate_loaded_entry(raw_entry, path, index)
            key = self._key(
                entry["data_id"], entry["episode_id"], entry["step_idx"]
            )
            if key in self._entries_by_key:
                raise ValueError(f"Schedule {path} contains duplicate transition {key}")
            self._entries.append(entry)
            self._entries_by_key[key] = entry

    def _validate_loaded_entry(
        self,
        raw_entry: Any,
        path: Path,
        index: int,
    ) -> dict[str, Any]:
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Schedule {path}.entries[{index}] must be a mapping")
        required = {
            "data_id",
            "episode_id",
            "step_idx",
            "action",
            "collision",
            "update_eligible",
            "skip_reason",
            "lr_scalar",
            "effective_learning_rate",
        }
        missing = required - raw_entry.keys()
        if missing:
            raise ValueError(
                f"Schedule {path}.entries[{index}] is missing fields {sorted(missing)}"
            )

        data_id = raw_entry["data_id"]
        episode_id = raw_entry["episode_id"]
        step_idx = raw_entry["step_idx"]
        action = raw_entry["action"]
        collision = raw_entry["collision"]
        eligible = raw_entry["update_eligible"]
        skip_reason = raw_entry["skip_reason"]
        if not isinstance(data_id, str) or not data_id:
            raise ValueError(f"Schedule entry {index} has invalid data_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError(f"Schedule entry {index} has invalid episode_id")
        if not isinstance(step_idx, int) or isinstance(step_idx, bool) or step_idx < 0:
            raise ValueError(f"Schedule entry {index} has invalid step_idx")
        if not isinstance(action, str):
            raise ValueError(f"Schedule entry {index} has invalid action")
        if not isinstance(collision, bool) or not isinstance(eligible, bool):
            raise ValueError(f"Schedule entry {index} has invalid boolean fields")
        if skip_reason is not None and not isinstance(skip_reason, str):
            raise ValueError(f"Schedule entry {index} has invalid skip_reason")

        scalar = raw_entry["lr_scalar"]
        effective_lr = raw_entry["effective_learning_rate"]
        if eligible:
            scalar = self._validate_scalar(
                scalar, self._key(data_id, episode_id, step_idx)
            )
            self._validate_effective_learning_rate(
                effective_lr,
                f"Schedule entry {index}.effective_learning_rate",
            )
            if skip_reason is not None:
                raise ValueError(f"Eligible schedule entry {index} has a skip reason")
        else:
            if scalar is not None or effective_lr is not None:
                raise ValueError(
                    f"Ineligible schedule entry {index} must have null LR values"
                )
            if skip_reason is None:
                raise ValueError(f"Ineligible schedule entry {index} needs a skip reason")

        return {
            "data_id": data_id,
            "episode_id": episode_id,
            "step_idx": step_idx,
            "action": action,
            "collision": collision,
            "update_eligible": eligible,
            "skip_reason": skip_reason,
            "lr_scalar": scalar,
            "effective_learning_rate": effective_lr,
        }

    def _within_episode_shuffled_entries(self) -> list[dict[str, Any]]:
        shuffled_entries = [dict(entry) for entry in self._entries]
        eligible_indices_by_episode: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, entry in enumerate(shuffled_entries):
            if entry["update_eligible"]:
                eligible_indices_by_episode[
                    (entry["data_id"], entry["episode_id"])
                ].append(index)

        rng = random.Random(self._shuffled_seed)
        for indices in eligible_indices_by_episode.values():
            lr_pairs = [
                (
                    shuffled_entries[index]["lr_scalar"],
                    shuffled_entries[index]["effective_learning_rate"],
                )
                for index in indices
            ]
            rng.shuffle(lr_pairs)
            for index, (scalar, effective_lr) in zip(indices, lr_pairs):
                shuffled_entries[index]["lr_scalar"] = scalar
                shuffled_entries[index]["effective_learning_rate"] = effective_lr

        return shuffled_entries

    def _write(
        self,
        path: Path,
        *,
        entries: list[dict[str, Any]],
        shuffled: bool,
        shuffled_seed: int | None,
    ) -> None:
        artifact = {
            "schema_version": SCHEDULE_SCHEMA_VERSION,
            "base_learning_rate": self._initial_lr,
            "shuffled": shuffled,
            "shuffled_seed": shuffled_seed,
            "entries": entries,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2)

    @staticmethod
    def _key(data_id: str, episode_id: str, step_idx: int) -> tuple[str, str, int]:
        return str(data_id), str(episode_id), int(step_idx)

    @staticmethod
    def _validate_learning_rate(value: Any, field: str) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{field} must be a positive finite number")
        learning_rate = float(value)
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError(f"{field} must be a positive finite number")
        return learning_rate

    @staticmethod
    def _validate_scalar(
        value: Any,
        key: tuple[str, str, int],
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"Schedule transition {key} has an invalid LR scalar")
        scalar = float(value)
        if not math.isfinite(scalar) or scalar < 0.0:
            raise ValueError(f"Schedule transition {key} has an invalid LR scalar")
        return scalar

    @staticmethod
    def _validate_effective_learning_rate(value: Any, field: str) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{field} must be a non-negative finite number")
        learning_rate = float(value)
        if not math.isfinite(learning_rate) or learning_rate < 0.0:
            raise ValueError(f"{field} must be a non-negative finite number")
        return learning_rate
