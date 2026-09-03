import json
import tempfile
import unittest
from pathlib import Path

from thesis_testing_tools.run_thesis_pipeline import (
    CORE_CONDITIONS,
    HABITAT_DATA_ID,
    SOURCE_DATA_IDS,
    run_seed_batch,
)


class SeedBatchPipelineTests(unittest.TestCase):
    def test_runs_one_source_pre_and_condition_specific_followups(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            temporary_path = Path(temporary_directory)
            output_root = temporary_path / "output"
            initial_checkpoint = temporary_path / "starting_ckpt"
            calls: list[list[str]] = []

            def fake_runner(command, **kwargs) -> None:
                self.assertTrue(kwargs["check"])
                calls.append(list(command))
                run_dir = Path(command[command.index("--run_dir") + 1])
                data_id = command[command.index("--data_id") + 1]
                run_dir.mkdir(parents=True)
                episode_log = [
                    {
                        "data_id": data_id,
                        "episode_index": 0,
                        "episode_id": "episode-0",
                        "adapter_source_mode": "test",
                        "termination_reason": "test-complete",
                        "steps": [],
                    }
                ]
                (run_dir / "episode_logs.json").write_text(
                    json.dumps(episode_log),
                    encoding="utf-8",
                )

            def unused_metric_calculator(
                prediction_path: Path,
                real_path: Path,
            ) -> dict[str, float]:
                raise AssertionError("Empty test episodes have no images")

            seed_dir = run_seed_batch(
                seed=321,
                fixed_mean_lr=6.25e-5,
                initial_checkpoint=initial_checkpoint,
                schedule_shuffle_seed=77,
                output_root=output_root,
                timestamp="test",
                subprocess_runner=fake_runner,
                metric_calculator=unused_metric_calculator,
            )

            invoked_data_ids = [
                command[command.index("--data_id") + 1]
                for command in calls
            ]
            self.assertEqual(
                invoked_data_ids,
                [
                    SOURCE_DATA_IDS,
                    HABITAT_DATA_ID,
                    HABITAT_DATA_ID,
                    SOURCE_DATA_IDS,
                    HABITAT_DATA_ID,
                    SOURCE_DATA_IDS,
                    HABITAT_DATA_ID,
                    SOURCE_DATA_IDS,
                    HABITAT_DATA_ID,
                    SOURCE_DATA_IDS,
                    HABITAT_DATA_ID,
                    SOURCE_DATA_IDS,
                ],
            )

            source_pre_dirs = {
                Path(command[command.index("--run_dir") + 1])
                for command in calls
                if command[command.index("--data_id") + 1] == SOURCE_DATA_IDS
            }
            self.assertIn(seed_dir / "source_pre", source_pre_dirs)
            self.assertNotIn(
                seed_dir / "c0_frozen" / "source_post",
                source_pre_dirs,
            )
            for condition_id, _ in CORE_CONDITIONS[1:]:
                self.assertIn(
                    seed_dir / condition_id / "source_post",
                    source_pre_dirs,
                )

            c0_config = (
                seed_dir / "c0_frozen" / "habitat_config.yaml"
            ).read_text(encoding="utf-8")
            self.assertIn("training_enabled: false", c0_config)
            self.assertIn("mode: record", c0_config)
            self.assertIn("shuffled_seed: 77", c0_config)
            self.assertIn("save_model_weights: false", c0_config)
            self.assertIn("seed: 321", c0_config)

            c2_config = (
                seed_dir / "c2_fixed_mean" / "habitat_config.yaml"
            ).read_text(encoding="utf-8")
            self.assertIn("initial_lr: 6.25e-05", c2_config)
            self.assertIn("learning_rate_schedule: false", c2_config)

            schedule_dir = seed_dir / "c0_frozen" / "habitat"
            c3_config = (
                seed_dir / "c3_aligned_replay" / "habitat_config.yaml"
            ).read_text(encoding="utf-8")
            self.assertIn(f"input_path: {json.dumps(str(schedule_dir))}", c3_config)
            self.assertIn("shuffled: false", c3_config)

            c4_config = (
                seed_dir / "c4_shuffled_replay" / "habitat_config.yaml"
            ).read_text(encoding="utf-8")
            self.assertIn(f"input_path: {json.dumps(str(schedule_dir))}", c4_config)
            self.assertIn("shuffled: true", c4_config)

            for condition_id, _ in CORE_CONDITIONS[1:]:
                post_config = (
                    seed_dir / condition_id / "source_post_config.yaml"
                ).read_text(encoding="utf-8")
                final_checkpoint = (
                    seed_dir / condition_id / "habitat" / "final_ckpt"
                )
                self.assertIn(
                    f"model_ckpt: {json.dumps(str(final_checkpoint))}",
                    post_config,
                )

            c0_summary = json.loads(
                (seed_dir / "c0_frozen" / "pipeline_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(c0_summary["source_post_reused_from_source_pre"])
            self.assertIsNone(c0_summary["artifacts"]["habitat_checkpoint"])
            self.assertIsNone(c0_summary["artifacts"]["post_replay_config"])
            self.assertEqual(
                c0_summary["source_retention"]["matched_episode_count"],
                1,
            )
            self.assertEqual(
                c0_summary["source_retention"][
                    "mae_post_minus_pre_episode_mean"
                ],
                None,
            )

            manifest = json.loads(
                (seed_dir / "seed_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["conditions"]), 6)
            self.assertTrue(
                manifest["conditions"][0]["source_post_reused_from_source_pre"]
            )
            self.assertTrue((seed_dir / "seed_summary.json").is_file())

    def test_rejects_invalid_fixed_mean_learning_rate(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive finite"):
            run_seed_batch(seed=1, fixed_mean_lr=0.0)


if __name__ == "__main__":
    unittest.main()
