from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from runtime_scripts.learning_rate_schedule import (
    ALIGNED_SCHEDULE_FILENAME,
    SHUFFLED_SCHEDULE_FILENAME,
    LearningRateSchedule,
)


class LearningRateScheduleTest(unittest.TestCase):
    def _record_schedule(self, output_dir: Path) -> None:
        schedule = LearningRateSchedule(
            {"mode": "record", "shuffled_seed": 17},
            output_dir=output_dir,
            initial_lr=0.1,
        )
        transitions = [
            ("ep-a", 0, "a0", False, 0.1),
            ("ep-a", 1, "a1", True, None),
            ("ep-a", 2, "a2", False, 0.2),
            ("ep-a", 3, "a3", False, 0.3),
            ("ep-b", 0, "b0", False, 1.1),
            ("ep-b", 1, "b1", False, 1.2),
            ("ep-b", 2, "b2", False, 1.3),
        ]
        for episode_id, step_idx, action, collision, scalar in transitions:
            schedule.record_transition(
                data_id="habitat",
                episode_id=episode_id,
                step_idx=step_idx,
                action=action,
                collision=collision,
                update_eligible=not collision,
                skip_reason="collision" if collision else None,
                lr_scalar=scalar,
            )
        schedule.save()

    def test_record_writes_aligned_and_within_episode_shuffled_files(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            output_dir = Path(temp_dir)
            self._record_schedule(output_dir)

            aligned_path = output_dir / ALIGNED_SCHEDULE_FILENAME
            shuffled_path = output_dir / SHUFFLED_SCHEDULE_FILENAME
            self.assertTrue(aligned_path.is_file())
            self.assertTrue(shuffled_path.is_file())

            aligned = json.loads(aligned_path.read_text(encoding="utf-8"))
            shuffled = json.loads(shuffled_path.read_text(encoding="utf-8"))
            self.assertFalse(aligned["shuffled"])
            self.assertTrue(shuffled["shuffled"])
            self.assertEqual(shuffled["shuffled_seed"], 17)

            def episode_scalars(artifact: dict, episode_id: str) -> list[float]:
                return sorted(
                    entry["lr_scalar"]
                    for entry in artifact["entries"]
                    if entry["episode_id"] == episode_id
                    and entry["update_eligible"]
                )

            self.assertEqual(
                episode_scalars(aligned, "ep-a"),
                episode_scalars(shuffled, "ep-a"),
            )
            self.assertEqual(
                episode_scalars(aligned, "ep-b"),
                episode_scalars(shuffled, "ep-b"),
            )
            collision_entry = shuffled["entries"][1]
            self.assertTrue(collision_entry["collision"])
            self.assertIsNone(collision_entry["lr_scalar"])
            self.assertIsNone(collision_entry["effective_learning_rate"])

    def test_replay_returns_saved_scalars_and_requires_all_entries(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            output_dir = Path(temp_dir)
            self._record_schedule(output_dir)
            artifact = json.loads(
                (output_dir / ALIGNED_SCHEDULE_FILENAME).read_text(encoding="utf-8")
            )
            replay = LearningRateSchedule(
                {"mode": "replay", "input_path": str(output_dir)},
                output_dir=output_dir / "replay-output",
                initial_lr=0.1,
            )

            for entry in artifact["entries"]:
                scalar = replay.replay_transition(
                    data_id=entry["data_id"],
                    episode_id=entry["episode_id"],
                    step_idx=entry["step_idx"],
                    action=entry["action"],
                    collision=entry["collision"],
                    update_eligible=entry["update_eligible"],
                    skip_reason=entry["skip_reason"],
                )
                self.assertEqual(scalar, entry["lr_scalar"])

            replay.assert_fully_consumed()

    def test_replay_validates_base_lr_once_when_loading(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            output_dir = Path(temp_dir)
            self._record_schedule(output_dir)
            with self.assertRaisesRegex(ValueError, "does not match configured initial_lr"):
                LearningRateSchedule(
                    {"mode": "replay", "input_path": str(output_dir)},
                    output_dir=output_dir / "replay-output",
                    initial_lr=0.2,
                )

    def test_replay_selects_shuffled_schedule_from_same_input_directory(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            output_dir = Path(temp_dir)
            self._record_schedule(output_dir)
            shuffled = json.loads(
                (output_dir / SHUFFLED_SCHEDULE_FILENAME).read_text(
                    encoding="utf-8"
                )
            )
            replay = LearningRateSchedule(
                {
                    "mode": "replay",
                    "input_path": str(output_dir),
                    "shuffled": True,
                },
                output_dir=output_dir / "replay-output",
                initial_lr=0.1,
            )

            first = shuffled["entries"][0]
            scalar = replay.replay_transition(
                data_id=first["data_id"],
                episode_id=first["episode_id"],
                step_idx=first["step_idx"],
                action=first["action"],
                collision=first["collision"],
                update_eligible=first["update_eligible"],
                skip_reason=first["skip_reason"],
            )
            self.assertEqual(scalar, first["lr_scalar"])

    def test_replay_rejects_unconsumed_entries(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            output_dir = Path(temp_dir)
            self._record_schedule(output_dir)
            replay = LearningRateSchedule(
                {"mode": "replay", "input_path": str(output_dir)},
                output_dir=output_dir / "replay-output",
                initial_lr=0.1,
            )
            with self.assertRaisesRegex(ValueError, "unconsumed schedule entries"):
                replay.assert_fully_consumed()

    def test_replay_fails_on_transition_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            output_dir = Path(temp_dir)
            self._record_schedule(output_dir)
            replay = LearningRateSchedule(
                {"mode": "replay", "input_path": str(output_dir)},
                output_dir=output_dir / "replay-output",
                initial_lr=0.1,
            )
            with self.assertRaisesRegex(ValueError, "action"):
                replay.replay_transition(
                    data_id="habitat",
                    episode_id="ep-a",
                    step_idx=0,
                    action="wrong-action",
                    collision=False,
                    update_eligible=True,
                    skip_reason=None,
                )

    def test_false_configuration_disables_schedule(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            schedule = LearningRateSchedule(
                False,
                output_dir=temp_dir,
                initial_lr=None,
            )
            self.assertFalse(schedule.is_recording)
            self.assertFalse(schedule.is_replaying)
            schedule.save()
            schedule.assert_fully_consumed()


if __name__ == "__main__":
    unittest.main()
