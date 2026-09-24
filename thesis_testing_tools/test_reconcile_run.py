import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
import io

import yaml

from runtime_scripts.event_logger import EventLogger
from thesis_testing_tools.reconcile_run import InvalidRunData, Reconciliation, main


class ReconcileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.worker = self.root / "habitat"
        self.worker.mkdir()
        self.config = {"runner": {"max_episode_steps": 10, "stop_on_wrapper_done": True,
                                   "adapter_params": {"episode_ids": ["827"]}},
                       "wrapper": {"training_enabled": False, "learning_rate_schedule": False}}
        self.config["engine"] = {"training": {"hyper_params": {"initial_lr": 0.0001}}}
        self.stage = dict(name="habitat", stage_id="habitat", data_id="habitat", status="completed",
                          events="habitat/events.jsonl", run_dir=str(self.worker), config_snapshot="config.yaml")
        self.manifest = dict(schema_version=1, status="completed", metadata={}, stages=[self.stage],
                             references={"pipeline_plan": "pipeline_manifest.json"})
        self.plan = {"stages": [{"run_dir": str(self.worker), "command": ["python"]}]}
        self.save_inputs()
        log = EventLogger(self.worker / "events.jsonl", {"outcome": "completed", "episode_setup": {
            "habitat/0": {"outcome": "completed", "episode_id": "827"}}})
        log.next_step(dict(outcome="completed", data_id="habitat", episode_id="827", episode_index=0,
                           step_idx=0, source_mode="habitat", transition={"outcome": "completed"},
                           action={"requested": "forward", "stop": False},
                           formatter={"converted_actions": ["move_forward"]},
                           adapter={"outcome": "completed", "primitives": {"0": {
                               "action": "move_forward", "outcome": "completed", "collision": False, "source_done": True}}},
                           environment={"primitive_actions_executed": ["move_forward"], "primitive_action_count": 1},
                           wrapper={"phase": "completed", "source_done": True, "collision": False,
                                    "stop_action": False, "update_eligible": True, "eligibility_skip_reason": None},
                           episode_end={"habitat/0": {"episode_id": "827", "attempts": 1, "steps_executed": 1,
                                                       "termination_reason": "adapter_done"}}))
        log.finish({"outcome": "completed", "episodes": 1})

    def save_inputs(self):
        (self.root / "config.yaml").write_text(yaml.safe_dump(self.config), encoding="utf-8")
        (self.root / "run_manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        (self.root / "pipeline_manifest.json").write_text(json.dumps(self.plan), encoding="utf-8")

    def mutate(self, function):
        path = self.worker / "events.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        function(records)
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    def run_report(self):
        return Reconciliation(self.root).run()

    def failures(self):
        return {c["check"] for c in self.run_report()["checks"] if c["status"] == "fail"}

    def test_valid_early_end_and_read_only_cli(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        report = self.run_report()
        self.assertEqual(report["status"], "pass", report["checks"])
        self.assertEqual(main([str(self.root)]), 0)
        for path, data in before.items():
            self.assertEqual(data, path.read_bytes())
        with self.assertRaises(SystemExit):
            main([str(self.root)])

    def test_missing_planned_worker(self):
        self.plan["stages"].append({"run_dir": str(self.root / "missing"), "command": ["python"]})
        self.save_inputs()
        self.assertIn("planned_worker_coverage", self.failures())

    def test_missing_summary_and_bad_json(self):
        self.mutate(lambda records: records.pop())
        self.assertIn("run_summary", self.failures())
        with (self.worker / "events.jsonl").open("a") as handle:
            handle.write('{"_event":')
        with self.assertRaises(InvalidRunData):
            self.run_report()
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main([str(self.root)]), 3)
        self.assertIn("No report written", output.getvalue())
        self.assertFalse((self.root / "reconciliation.json").exists())

    def test_missing_episode_end_and_wrong_counts(self):
        self.mutate(lambda records: records[1]["episode_end"]["habitat/0"].update(attempts=8))
        self.assertIn("episode_attempts", self.failures())
        self.mutate(lambda records: records[1].pop("episode_end"))
        self.assertIn("episode_end_coverage", self.failures())

    def test_episode_order_and_attempt_indices(self):
        self.config["runner"]["adapter_params"]["episode_ids"] = ["999", "827"]
        self.save_inputs()
        self.mutate(lambda records: records[1].update(step_idx=1))
        self.assertTrue({"selected_episode_order", "attempt_indices"} <= self.failures())

    def test_collision_partial_primitives_is_valid(self):
        def collision(records):
            record = records[1]
            record["formatter"]["converted_actions"].append("turn_left")
            record["adapter"]["primitives"]["0"]["collision"] = True
            record["wrapper"].update(collision=True, update_eligible=False, eligibility_skip_reason="collision")
        self.mutate(collision)
        self.assertEqual(self.run_report()["status"], "pass")

    def test_no_op_and_frozen_update(self):
        def noop(records):
            record = records[1]
            record.update(outcome="no_op", transition={"outcome": "no_op"}, formatter={"converted_actions": []})
            for key in ("adapter", "wrapper", "environment"):
                record.pop(key)
            record["episode_end"]["habitat/0"].update(steps_executed=0, termination_reason="max_episode_steps")
        self.config["runner"]["max_episode_steps"] = 1
        self.save_inputs()
        self.mutate(noop)
        self.assertEqual(self.run_report()["status"], "pass")
        self.mutate(lambda records: records[1].update(training={"optimizer_step": True}))
        self.assertTrue({"no_op_not_executed", "optimizer_not_applied"} <= self.failures())

    def test_unknown_optimizer_and_absent_plan(self):
        self.mutate(lambda records: records[1].update(training={"optimizer_step": None}))
        self.assertIn("optimizer_outcome_known", self.failures())
        self.manifest["references"] = {}
        self.save_inputs()
        self.assertTrue(any(c["check"] == "planned_worker_coverage" and c["status"] == "unknown" for c in self.run_report()["checks"]))

    def test_action_sequence_and_replay_schedule(self):
        action_dir = self.root / "actions"
        action_dir.mkdir()
        (action_dir / "827.json").write_text(json.dumps(dict(episode_id="827", target_steps=1, actions=["forward"])))
        self.config["runner"]["adapter_params"]["fixed_action_files_dir"] = str(action_dir)
        self.config["wrapper"].update(training_enabled=True, learning_rate_schedule={"mode": "replay", "input_path": str(self.worker), "shuffled": True})
        self.save_inputs()
        schedule = dict(schema_version=1, shuffled=True, shuffled_seed=100, base_learning_rate=0.0001, entries=[dict(data_id="habitat", episode_id="827", step_idx=0,
                        action="forward", collision=False, update_eligible=True, skip_reason=None,
                        lr_scalar=0.5, effective_learning_rate=0.00005)])
        schedule_path = self.worker / "learning_rate_schedule_shuffled.json"
        schedule_path.write_text(json.dumps(schedule))
        (self.worker / "learning_rate_schedule.json").write_text(json.dumps({**schedule, "shuffled": False}))
        def train(records):
            records[1]["wrapper"]["update_weight"] = 0.5
            records[1]["training"] = {"optimizer_step": True, "applied_learning_rate": 0.00005}
        self.mutate(train)
        self.assertEqual(self.run_report()["status"], "pass")
        schedule["entries"][0]["effective_learning_rate"] = 0.1
        schedule_path.write_text(json.dumps(schedule))
        self.assertIn("schedule_applied_lr", self.failures())
        self.mutate(lambda records: records[1]["action"].update(requested="wrong"))
        self.assertIn("action_sequence", self.failures())

    def test_c0_reuse_and_relocated_plan(self):
        pre = self.root / "source_pre"
        pre.mkdir()
        (pre / "events.jsonl").write_bytes((self.worker / "events.jsonl").read_bytes())
        self.manifest["stages"].insert(0, {**self.stage, "name": "source_pre", "stage_id": "source_pre",
                                         "events": "source_pre/events.jsonl", "run_dir": str(pre)})
        self.manifest["metadata"] = {"source_episode_order": {"habitat": ["827"]}}
        self.plan = {"source_pre": {"run_dir": "/original/source_pre"}, "conditions": [
            {"condition_dir": "/original", "source_post_reused_from_source_pre": True}]}
        post = self.root / "source_post"
        post.mkdir()
        reuse_path = post / "reuse_manifest.json"
        reuse_path.write_text(json.dumps({"status": "reused", "source_pre_reference": "/original/source_pre"}))
        self.save_inputs()
        report = Reconciliation(self.root, [("/original", str(self.root))]).run()
        self.assertEqual(report["status"], "pass", report["checks"])
        reuse_path.write_text(json.dumps({"status": "reused", "source_pre_reference": "/original/wrong"}))
        report = Reconciliation(self.root, [("/original", str(self.root))]).run()
        self.assertTrue(any(c["check"] == "source_post_reuse_target" and c["status"] == "fail" for c in report["checks"]))

    def test_shuffle_preserves_episode_membership(self):
        from runtime_scripts.learning_rate_schedule import LearningRateSchedule
        schedule = LearningRateSchedule({"mode": "record", "shuffled_seed": 17}, output_dir=self.worker, initial_lr=0.001)
        for episode in ("a", "b"):
            for step in range(4):
                schedule.record_transition(data_id="habitat", episode_id=episode, step_idx=step,
                                           action="forward", collision=False, update_eligible=True,
                                           skip_reason=None, lr_scalar=step + (1 if episode == "a" else 10))
        schedule.save()
        check = Reconciliation(self.root)
        check.shuffle(self.worker, "habitat")
        self.assertEqual(check.report()["status"], "pass")
        path = self.worker / "learning_rate_schedule_shuffled.json"
        data = json.loads(path.read_text())
        data["entries"][0]["lr_scalar"] = 999
        path.write_text(json.dumps(data))
        check = Reconciliation(self.root)
        check.shuffle(self.worker, "habitat")
        self.assertEqual(check.report()["status"], "fail")

    def test_snapshot_hash_and_unsealed_event(self):
        self.manifest["inputs"] = [{"snapshot": "config.yaml", "sha256": "incorrect"}]
        self.save_inputs()
        self.mutate(lambda records: records[1]["_event"].update(finished_at=None))
        self.assertTrue({"snapshot_hash", "events_sealed"} <= self.failures())

    def test_missing_artifact_and_malformed_config(self):
        self.mutate(lambda records: records[1]["wrapper"].update(real_obs_path="missing.png"))
        self.assertIn("artifact_exists", self.failures())
        (self.root / "config.yaml").write_text("runner: [")
        with self.assertRaises(InvalidRunData):
            self.run_report()

    def test_invalid_manifest_writes_no_report(self):
        path = self.root / "run_manifest.json"
        for value in ("{", "[]", '{"schema_version": 99}', '{"schema_version": 1, "stages": [null]}'):
            with self.subTest(value=value):
                path.write_text(value)
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main([str(self.root)]), 3)
                self.assertFalse((self.root / "reconciliation.json").exists())

    def test_disabled_engine_training_is_valid(self):
        self.config["engine"]["training"] = False
        self.save_inputs()
        self.assertEqual(self.run_report()["status"], "pass")


if __name__ == "__main__":
    unittest.main()
