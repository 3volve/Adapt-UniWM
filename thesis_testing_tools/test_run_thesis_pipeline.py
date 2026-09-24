from runtime_scripts.event_logger import EventLogger
import json
import io
import hashlib
import argparse
import ast
import types
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from thesis_testing_tools.run_thesis_pipeline import (
    CORE_CONDITIONS,
    HABITAT_DATA_ID,
    SOURCE_DATA_IDS,
    run_seed_batch,
    reconciled_run,
    smoke_test_result,
    _episode_diagnostic_metrics,
)


class ReconciliationIntegrationTests(unittest.TestCase):
    def test_runs_after_finalization_on_success_and_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "run"
                def reconcile(command, **kwargs):
                    manifest = json.loads((root / "run_manifest.json").read_text())
                    self.assertEqual(manifest["status"], "failed" if fail else "completed")
                    self.assertEqual(manifest["references"]["reconciliation"], "reconciliation.json")
                    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
                    self.assertEqual(events[-1]["_event"]["kind"], "run_summary")
                    self.assertIn("thesis_testing_tools.reconcile_run", command)
                    self.assertFalse(kwargs["check"])
                    return types.SimpleNamespace(returncode=1)
                with patch("thesis_testing_tools.run_thesis_pipeline.subprocess.run", side_effect=reconcile) as run:
                    def execute():
                        with reconciled_run(root=root, repo_root=root.parent, run_type="test", metadata={}, capture=lambda command: {}):
                            if fail:
                                raise RuntimeError("original experiment failure")
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "original experiment failure"):
                            execute()
                    else:
                        execute()
                    run.assert_called_once()

    def test_launch_failure_does_not_replace_experiment_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            with patch("thesis_testing_tools.run_thesis_pipeline.subprocess.run", side_effect=OSError("launch failed")):
                with self.assertRaisesRegex(RuntimeError, "experiment failed"):
                    with reconciled_run(root=root, repo_root=root.parent, run_type="test", metadata={}, capture=lambda command: {}):
                        raise RuntimeError("experiment failed")


class SeedBatchPipelineTests(unittest.TestCase):
    def test_manifest_count_checked_before_creating_run(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"habitat": {"test": ["352", "827"]}}))
            with patch("thesis_testing_tools.run_thesis_pipeline.generate_action_sequence") as generate:
                with self.assertRaisesRegex(ValueError, "lists only 2"):
                    run_seed_batch(seed=1, fixed_mean_lr=6.25e-5, source_manifest=manifest, habitat_episodes=3,
                                   output_root=root / "output")
                generate.assert_not_called()
                self.assertFalse((root / "output").exists())

    def test_generation_order_and_failure_precede_model_stages(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"habitat": {"test": ["352", "827", "817"]}}))
            with patch("thesis_testing_tools.run_thesis_pipeline._captured_command", return_value={}), patch("thesis_testing_tools.run_thesis_pipeline.generate_action_sequence",
                       side_effect=["first.json", RuntimeError("generation failed")]) as generate:
                with patch("thesis_testing_tools.run_thesis_pipeline.run_stage") as stage:
                    with self.assertRaisesRegex(RuntimeError, "generation failed"):
                        run_seed_batch(seed=1, fixed_mean_lr=6.25e-5, source_manifest=manifest, habitat_episodes=2,
                                       output_root=root / "output")
                    self.assertEqual([call.args[0] for call in generate.call_args_list], ["352", "827"])
                    stage.assert_not_called()
            record_path = next((root / "output").glob("*/run_manifest.json"))
            record = json.loads(record_path.read_text())
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["failure"]["phase"], "preparation")
            self.assertIn("generation failed", record["failure"]["message"])

    def test_past_run_action_option_is_retired(self):
        with self.assertRaisesRegex(TypeError, "habitat_action_run"):
            run_seed_batch(seed=1, fixed_mean_lr=6.25e-5, habitat_action_run=Path("old"))

    def test_sequence_directory_loader(self):
        # Exercise the actual loader without importing Habitat's native dependencies.
        path = Path("source_tools/habitat_source_tools.py")
        module = ast.parse(path.read_text())
        adapter = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "HabitatEpisodeAdapter")
        loader = next(node for node in adapter.body if isinstance(node, ast.FunctionDef) and node.name == "_load_fixed_actions_from_files")
        loader.decorator_list = []
        namespace = {"Path": Path, "json": json, "__file__": str(path.resolve())}
        exec(compile(ast.Module(body=[loader], type_ignores=[]), str(path), "exec"), namespace)
        load = namespace[loader.name]
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            payload = {"schema_version": 1, "episode_id": "827", "target_steps": 2, "actions": ["first", "second"]}
            (root / "827.json").write_text(json.dumps(payload))
            self.assertEqual(load(directory), (["827"], {"827": ["first", "second"]}))
            (root / "duplicate.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "duplicate episode_id"):
                load(directory)
            payload["episode_id"] = "352"
            payload["target_steps"] = 3
            (root / "duplicate.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "matching target_steps"):
                load(directory)

    @patch("thesis_testing_tools.run_thesis_pipeline.generate_action_sequence")
    @patch(
        "thesis_testing_tools.run_thesis_pipeline._captured_command",
        return_value={"available": False, "error": "not queried in unit tests"},
    )
    def test_runs_one_source_pre_and_condition_specific_followups(self, _capture, generate) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            temporary_path = Path(temporary_directory)
            output_root = temporary_path / "output"
            initial_checkpoint = temporary_path / "starting_ckpt"
            calibration = temporary_path / "development_calibration.json"
            calibration.write_text(json.dumps({"mean_lr": 6.25e-5, "split": "development"}))
            calls: list[list[str]] = []
            initial_c2_config = None

            def fake_generate(episode_id, target_steps, run_dir, **kwargs):
                self.assertEqual(calls, [])
                self.assertEqual((episode_id, target_steps, kwargs["seed"]), ("827", 8, 321))
                self.assertEqual(kwargs["checkpoint"], initial_checkpoint)
                path = run_dir / "habitat_action_sequences" / "827.json"
                path.parent.mkdir()
                path.write_text(json.dumps({"episode_id": episode_id, "actions": ["action"] * target_steps}))
                return path.relative_to(Path.cwd()).as_posix()

            generate.side_effect = fake_generate

            def fake_runner(command, **kwargs) -> None:
                nonlocal initial_c2_config
                self.assertTrue(kwargs["check"])
                calls.append(list(command))
                run_dir = Path(command[command.index("--run_dir") + 1])
                data_id = command[command.index("--data_id") + 1]
                run_dir.mkdir(parents=True)
                if run_dir.name == "source_pre":
                    # C2 and its provenance are finalized before any model stage.
                    c2_path = run_dir.parent / "c2_fixed_mean" / "habitat_config.yaml"
                    initial_c2_config = c2_path.read_bytes()
                    self.assertIn(b"initial_lr: 6.25e-05", initial_c2_config)
                    initial_provenance = json.loads((run_dir.parent / "provenance.json").read_text())
                    self.assertEqual(initial_provenance["workload"]["fixed_mean_learning_rate"], 6.25e-5)
                    fingerprint = next(item for item in initial_provenance["inputs"]
                                       if item["path"] == str(c2_path))
                    self.assertEqual(fingerprint["sha256"], hashlib.sha256(initial_c2_config).hexdigest())
                if run_dir.parent.name == "c0_frozen":
                    (run_dir / "learning_rate_schedule.json").write_text(json.dumps({
                        "entries": [
                            {"data_id": "habitat", "episode_id": "episode-0",
                             "update_eligible": rate is not None,
                             "effective_learning_rate": rate}
                            # Deliberately differs from the supplied development mean.
                            for rate in (1e-5, None, 3e-5)
                        ]
                    }))
                if run_dir.parent.name == "c2_fixed_mean" and data_id == HABITAT_DATA_ID:
                    config = Path(command[command.index("--config_path") + 1])
                    self.assertIn("initial_lr: 6.25e-05", config.read_text())
                with EventLogger(run_dir / "events.jsonl") as log:
                    log.feed({"outcome": "completed"})

            def unused_metric_calculator(
                prediction_path: Path,
                real_path: Path,
            ) -> dict[str, float]:
                raise AssertionError("Empty test episodes have no images")

            seed_dir = run_seed_batch(
                seed=321,
                fixed_mean_lr=6.25e-5,
                fixed_mean_calibration=calibration,
                initial_checkpoint=initial_checkpoint,
                schedule_shuffle_seed=77,
                source_episodes=1,
                habitat_episodes=1,
                max_episode_steps=2,
                habitat_max_episode_steps=8,
                max_route_steps=1,
                smoke_test=True,
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

            for command in calls:
                self.assertEqual(command[command.index("--seed") + 1], "321")
                config = Path(command[command.index("--config_path") + 1]).read_text()
                steps = 8 if command[command.index("--data_id") + 1] == HABITAT_DATA_ID else 2
                self.assertIn(f"max_episode_steps: {steps}", config)
                self.assertEqual(
                    command[command.index("--num_episodes") + 1],
                    "1",
                )

            source_pre_config = (seed_dir / "source_pre_config.yaml").read_text(
                encoding="utf-8"
            )
            self.assertIn("max_episode_steps: 2", source_pre_config)
            self.assertIn("max_route_steps: 1", source_pre_config)
            self.assertIn("eval_dataset_manifest.json", source_pre_config)

            c0_config = (
                seed_dir / "c0_frozen" / "habitat_config.yaml"
            ).read_text(encoding="utf-8")
            self.assertIn("training_enabled: false", c0_config)
            self.assertIn("mode: record", c0_config)
            self.assertIn("shuffled_seed: 77", c0_config)
            self.assertIn("save_model_weights: false", c0_config)
            self.assertIn("seed: 321", c0_config)
            self.assertIn("max_episode_steps: 8", c0_config)
            self.assertIn("max_route_steps: 1", c0_config)
            self.assertNotIn("fixed_action_run_dir", c0_config)

            c1_config = (
                seed_dir / "c1_fixed_base" / "habitat_config.yaml"
            ).read_text(encoding="utf-8")
            self.assertIn(
                "fixed_action_files_dir: "
                + json.dumps(str(seed_dir / "habitat_action_sequences")),
                c1_config,
            )

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
            self.assertEqual(c0_summary["analysis"]["status"], "disabled")
            self.assertNotIn("source_retention", c0_summary)
            self.assertEqual(list(seed_dir.rglob("thesis_episode_metrics.csv")), [])
            self.assertEqual(list(seed_dir.rglob("episode_logs.json")), [])

            manifest = json.loads(
                (seed_dir / "seed_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["conditions"]), 6)
            self.assertTrue(manifest["workload"]["smoke_test"])
            self.assertEqual(manifest["workload"]["source_max_episode_steps"], 2)
            self.assertEqual(manifest["workload"]["habitat_max_episode_steps"], 8)
            self.assertEqual(manifest["workload"]["model_random_seed"], 321)
            self.assertEqual(manifest["workload"]["habitat_episodes"], 1)
            self.assertTrue(
                manifest["conditions"][0]["source_post_reused_from_source_pre"]
            )
            self.assertNotIn("habitat_action_reference_run", manifest["conditions"][0])
            generate.assert_called_once()
            for condition in manifest["conditions"]:
                self.assertNotIn("habitat_action_reference_run", condition)
                config = Path(condition["habitat_config"]).read_text()
                self.assertIn('episode_ids: ["827"]', config)
                self.assertIn("fixed_action_files_dir: " + json.dumps(str(seed_dir / "habitat_action_sequences")), config)
            self.assertEqual(manifest["workload"]["habitat_episode_ids"], ["827"])
            sequence_files = manifest["workload"]["habitat_action_sequences"]["files"]
            self.assertEqual(len(sequence_files), 1)
            provenance = json.loads((seed_dir / "provenance.json").read_text())
            self.assertIn(str((Path.cwd() / sequence_files[0]).resolve()), [item["path"] for item in provenance["inputs"]])
            self.assertTrue((seed_dir / "seed_summary.json").is_file())
            summary = json.loads((seed_dir / "seed_summary.json").read_text())
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["smoke_test_result"]["status"], "not_evaluated")
            provenance = json.loads(
                (seed_dir / "provenance.json").read_text(encoding="utf-8")
            )
            self.assertTrue(provenance["workload"]["smoke_test"])
            self.assertIn("git", provenance)
            self.assertIn("initial_checkpoint", provenance)
            self.assertEqual(len(provenance["output_checkpoints"]), 5)
            self.assertEqual(manifest["fixed_mean_learning_rate"], 6.25e-5)
            self.assertEqual(summary["fixed_mean_learning_rate"], 6.25e-5)
            self.assertEqual(provenance["workload"]["fixed_mean_learning_rate"], 6.25e-5)
            self.assertEqual(provenance["workload"]["fixed_mean_learning_rate_source"], "provided_fixed_mean_lr")
            c2_path = seed_dir / "c2_fixed_mean" / "habitat_config.yaml"
            self.assertEqual(c2_path.read_bytes(), initial_c2_config)
            fingerprint = next(item for item in provenance["inputs"] if item["path"] == str(c2_path))
            self.assertEqual(fingerprint["sha256"], hashlib.sha256(c2_path.read_bytes()).hexdigest())
            self.assertTrue(any(item["path"] == str(schedule_dir / "learning_rate_schedule.json")
                                for item in provenance["inputs"]))
            record = json.loads((seed_dir / "run_manifest.json").read_text())
            self.assertEqual(record["status"], "completed")
            self.assertTrue(all(stage["events_available"] for stage in record["stages"]))
            self.assertEqual(len(record["stages"]), len(calls))
            self.assertEqual(len({stage["stage_id"] for stage in record["stages"]}), len(calls))
            self.assertTrue(all(stage["status"] == "completed" for stage in record["stages"]))
            self.assertEqual((seed_dir / record["metadata"]["fixed_mean_calibration"]).read_bytes(),
                             calibration.read_bytes())
            for stage, command in zip(record["stages"], calls):
                self.assertEqual(stage["executed_command"], command)
                self.assertEqual(Path(command[command.index("--config_path") + 1]),
                                 seed_dir / stage["config_snapshot"])
            artifacts = json.loads((seed_dir / record["artifact_index"]).read_text())["files"]
            self.assertIn("seed_summary.json", [item["path"] for item in artifacts])

    def test_rejects_invalid_fixed_mean_before_creating_run(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            output_root = Path(directory) / "output"
            for rate in (None, 0, -1, float("nan"), float("inf"), True):
                with self.subTest(rate=rate):
                    with self.assertRaisesRegex(ValueError, "fixed_mean_lr must be a positive finite"):
                        run_seed_batch(seed=1, fixed_mean_lr=rate, output_root=output_root)
                    self.assertFalse(output_root.exists())

    def test_cli_requires_development_mean_for_all_conditions(self) -> None:
        from thesis_testing_tools.run_thesis_pipeline import main
        for arguments, message in (
            (["--all-conditions", "--seed", "100"], "requires --fixed-mean-lr"),
            (["--fixed-mean-lr", "6.25e-5"], "--fixed-mean-lr require --all-conditions"),
        ):
            with self.subTest(arguments=arguments), patch("sys.argv", ["pipeline", *arguments]), patch(
                "sys.stderr", new_callable=io.StringIO
            ) as stderr, patch("thesis_testing_tools.run_thesis_pipeline.run_seed_batch") as run:
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
                self.assertIn(message, stderr.getvalue())
                run.assert_not_called()

    def test_rejects_invalid_smoke_workload(self) -> None:
        with self.assertRaisesRegex(ValueError, "source_episodes"):
            run_seed_batch(seed=1, fixed_mean_lr=6.25e-5, source_episodes=0)
        for name in ("source_max_episode_steps", "habitat_max_episode_steps"):
            with self.assertRaisesRegex(ValueError, name):
                run_seed_batch(seed=1, fixed_mean_lr=6.25e-5, **{name: 0})
        for seed in (-1, 2**32, True):
            with self.assertRaisesRegex(ValueError, "seed"):
                run_seed_batch(seed=seed, fixed_mean_lr=6.25e-5)

    def test_cli_forwards_stage_limits(self) -> None:
        from thesis_testing_tools.run_thesis_pipeline import main
        with patch("sys.argv", ["pipeline", "--all-conditions", "--seed", "100",
                                "--fixed-mean-lr", "6.25e-5",
                                "--smoke-test", "--source-max-episode-steps", "3",
                                "--habitat-max-episode-steps", "8"]), patch(
            "thesis_testing_tools.run_thesis_pipeline.run_seed_batch"
        ) as run:
            main()
        self.assertEqual(run.call_args.kwargs["source_max_episode_steps"], 3)
        self.assertEqual(run.call_args.kwargs["habitat_max_episode_steps"], 8)
        self.assertEqual(run.call_args.kwargs["max_episode_steps"], 2)
        self.assertEqual(run.call_args.kwargs["fixed_mean_lr"], 6.25e-5)

    def test_runner_seeds_before_model_initialization(self) -> None:
        # Execute the actual CLI block without importing the unavailable ML stack.
        path = Path(__file__).resolve().parent.parent / "uniwm_episode_runner.py"
        tree = ast.parse(path.read_text())
        block = ast.Module(body=tree.body[-1].body, type_ignores=[])
        events = []
        fake_numpy = types.SimpleNamespace(random=types.SimpleNamespace(
            seed=lambda seed: events.append(("numpy", seed))))
        fake_torch = types.SimpleNamespace(manual_seed=lambda seed: events.append(("torch", seed)))
        def runner(*args, **kwargs):
            events.append(("model", None))
            return types.SimpleNamespace(run_episodes=lambda *a: None, episode_index=1,
                                         wrapper=types.SimpleNamespace(engine=object()))
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory, patch(
            "sys.argv", ["runner", "--config_path", "unused.yaml", "--run_dir", directory,
                         "--seed", "321"]
        ), patch.dict("sys.modules", {"numpy": fake_numpy, "torch": fake_torch}), patch(
            "random.seed", side_effect=lambda seed: events.append(("python", seed))
        ):
            exec(compile(block, str(path), "exec"), {
                "argparse": argparse, "Path": Path, "UniWMEpisodeRunner": runner,
                "EventLogger": EventLogger, "event_values": lambda x: x,
                "runtime_metadata": lambda *a, **k: {},
            })
        self.assertEqual(events, [("python", 321), ("numpy", 321), ("torch", 321), ("model", None)])

    def test_optimizer_count_uses_explicit_update_flag(self) -> None:
        rows = _episode_diagnostic_metrics({"steps": [
            {"replanned": False, "training_logs": {
                "optimizer_step": updated, "final_lr": 1e-4,
            }}
            for updated in (False, True, False)
        ]})
        self.assertEqual(rows["optimizer_step_count"], 1)

    def test_smoke_coverage(self) -> None:
        def metric_row(data_id):
            return {
                "data_id": data_id, "optimizer_step_count": 1,
                "prediction_available_count": 1,
                "mae": 0.2, "ssim": 0.8, "lpips": 0.3,
            }

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            schedule_path = Path(directory) / "schedule.json"
            for case in (
                "passed", "no_update", "missing_source", "invalid_metric",
                "constant_rates", "different_episodes", "missing_schedule",
            ):
                with self.subTest(case=case):
                    stages = {"source_pre": [
                        metric_row(data_id) for data_id in SOURCE_DATA_IDS.split(",")
                    ]}
                    for condition_id, _ in CORE_CONDITIONS:
                        stages[f"{condition_id}/habitat"] = [metric_row(HABITAT_DATA_ID)]
                        if condition_id != "c0_frozen":
                            stages[f"{condition_id}/source_post"] = [
                                metric_row(data_id) for data_id in SOURCE_DATA_IDS.split(",")
                            ]
                    entries = [
                        {"data_id": "habitat", "episode_id": "827",
                         "update_eligible": True, "effective_learning_rate": rate}
                        for rate in (1e-4, 2e-4)
                    ]
                    if case == "no_update":
                        stages["c1_fixed_base/habitat"][0]["optimizer_step_count"] = 0
                    elif case == "missing_source":
                        stages["c2_fixed_mean/source_post"].pop()
                    elif case == "invalid_metric":
                        stages["source_pre"][0]["lpips"] = float("nan")
                    elif case == "constant_rates":
                        entries[1]["effective_learning_rate"] = 1e-4
                    elif case == "different_episodes":
                        entries[1]["episode_id"] = "352"
                    schedule_path.write_text(json.dumps({"entries": entries}))
                    if case == "missing_schedule":
                        schedule_path.unlink()
                    result = smoke_test_result(stages, schedule_path)
                    self.assertEqual(
                        result["status"], "passed" if case == "passed" else "inconclusive"
                    )
                    self.assertEqual(len(result["missing_coverage"]), 0 if case == "passed" else 1)


if __name__ == "__main__":
    unittest.main()
