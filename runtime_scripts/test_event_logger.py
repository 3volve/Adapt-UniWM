import json
from pathlib import Path
import tempfile
import unittest

from runtime_scripts.event_logger import EventLogger


class EventLoggerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "worker" / "events.jsonl"

    def records(self):
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]

    def test_step_boundaries_and_final_run_details(self):
        logger = EventLogger(self.path, {"run_id": "trial", "seed": 100})
        self.assertEqual(self.records(), [])
        logger.feed({"outcome": "completed"})
        logger.next_step({"episode_id": "ep-1", "step_idx": 0})
        self.assertEqual([r["_event"]["kind"] for r in self.records()], ["startup"])
        logger.feed({"adapter": {"collision": True}, "outcome": "completed"})
        logger.next_step({"episode_id": "ep-1", "step_idx": 1})
        logger.feed({"action": {"converted": []}, "outcome": "no_op"})
        logger.finish({"outcome": "completed", "checkpoint": "final_ckpt"})
        records = self.records()
        self.assertEqual([r["_event"]["kind"] for r in records], ["startup", "step", "step", "run_summary"])
        self.assertEqual([r["_event"]["id"] for r in records], [0, 1, 2, 3])
        self.assertEqual([r["outcome"] for r in records], ["completed", "completed", "no_op", "completed"])
        self.assertTrue(records[1]["adapter"]["collision"])
        self.assertNotIn("adapter", records[2])
        self.assertNotIn("checkpoint", records[2])
        self.assertEqual(records[3]["checkpoint"], "final_ckpt")
        for record in records:
            self.assertEqual(record["_event"]["schema_version"], 1)
            self.assertLessEqual(record["_event"]["started_at"], record["_event"]["finished_at"])
        self.assertEqual(records[0]["_event"]["finished_at"], records[1]["_event"]["started_at"])

    def test_unfinished_outcomes_are_never_promoted_implicitly(self):
        with EventLogger(self.path) as logger:
            logger.next_step()
            logger.next_step()
        self.assertEqual([r["outcome"] for r in self.records()], ["unfinished"] * 4)

    def test_nested_merge_overwrite_and_contribution_copies(self):
        logger = EventLogger(self.path)
        initial = {"training": {"state": "entered", "measurements": {"loss": 0.72},
                                "groups": [{"lr": 0.1}], "evidence": [{"value": 2}]}, "optional": None}
        logger.feed(initial)
        initial["training"]["measurements"]["loss"] = 999
        initial["training"]["groups"][0]["lr"] = 999
        initial["training"]["evidence"][0]["value"] = 999
        logger.feed({"training": {"state": "completed", "measurements": {"gradient_norm": 2.0}},
                     "optional": "now known"})
        logger.feed({"training": {"groups": ["replacement"]}})
        logger.finish()
        record = self.records()[0]
        self.assertEqual(record["training"], {"state": "completed", "measurements": {"loss": 0.72, "gradient_norm": 2.0},
                                              "groups": ["replacement"], "evidence": [{"value": 2}]})
        self.assertEqual(record["optional"], "now known")

    def test_conflicting_merge_does_not_partially_modify_current_event(self):
        logger = EventLogger(self.path)
        logger.feed({"training": {"loss": 0.72}})
        with self.assertRaisesRegex(AssertionError, "training.loss"):
            logger.feed({"new_field": 1, "training": {"loss": {"value": 0.72}}})
        with self.assertRaisesRegex(AssertionError, "training"):
            logger.feed({"training": None})
        logger.finish()
        self.assertNotIn("new_field", self.records()[0])
        self.assertEqual(self.records()[0]["training"], {"loss": 0.72})

    def test_frozen_records_do_not_change_with_later_feeds(self):
        logger = EventLogger(self.path)
        logger.feed({"shared": {"value": "startup"}})
        logger.next_step()
        written = self.path.read_bytes()
        logger.feed({"shared": {"value": "step"}, "outcome": "completed"})
        self.assertEqual(self.path.read_bytes(), written)
        logger.finish()
        self.assertTrue(self.path.read_bytes().startswith(written))
        self.assertEqual(self.records()[0]["shared"]["value"], "startup")
        self.assertEqual(self.records()[1]["shared"]["value"], "step")

    def test_reserved_metadata_cannot_be_overwritten(self):
        with self.assertRaisesRegex(AssertionError, "reserved"):
            EventLogger(self.path, {"_event": {"id": 99}})
        self.assertFalse(self.path.exists())
        logger = EventLogger(self.path)
        for operation in (logger.feed, logger.next_step, logger.finish):
            with self.subTest(operation=operation.__name__), self.assertRaisesRegex(AssertionError, "reserved"):
                operation({"_event": {"id": 99}})
            self.assertEqual(self.records(), [])
        logger.finish()

    def test_exception_and_interrupt_finish_then_propagate(self):
        for error in (RuntimeError("update failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                path = self.path.with_name(type(error).__name__ + ".jsonl")
                with self.assertRaises(type(error)):
                    with EventLogger(path) as logger:
                        logger.next_step()
                        logger.feed({"training": {"state": "optimizer_started"}})
                        raise error
                records = [json.loads(line) for line in path.read_text().splitlines()]
                outcome = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                self.assertEqual(records[1]["outcome"], outcome)
                self.assertEqual(records[2]["outcome"], outcome)
                self.assertEqual(records[1]["training"]["state"], "optimizer_started")
                self.assertEqual(records[1]["_event"]["exception"]["type"], type(error).__name__)

    def test_explicit_finish_inside_context_does_not_write_twice(self):
        with EventLogger(self.path) as logger:
            logger.finish({"outcome": "completed", "details": "done"})
        self.assertEqual(len(self.records()), 2)
        self.assertEqual(self.records()[-1]["details"], "done")
        for operation in (lambda: logger.feed({}), logger.next_step, logger.finish):
            with self.assertRaisesRegex(RuntimeError, "finished"):
                operation()
        self.assertEqual(len(self.records()), 2)

    def test_existing_file_is_preserved(self):
        EventLogger(self.path).finish()
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            EventLogger(self.path)
        self.assertEqual(self.path.read_bytes(), before)

    def test_serialization_error_writes_no_partial_record(self):
        logger = EventLogger(self.path)
        with self.assertRaises(TypeError):
            logger.finish({"unsupported": object()})
        self.assertEqual(self.records(), [])
        logger.feed({"outcome": "completed"})
        logger.finish({"outcome": "completed"})
        self.assertEqual(len(self.records()), 2)

    def test_workers_have_independent_current_events(self):
        other = self.path.with_name("other.jsonl")
        with EventLogger(self.path) as first, EventLogger(other) as second:
            first.next_step({"worker": "first"})
            second.next_step({"worker": "second"})
            first.feed({"measurement": 10})
            second.feed({"measurement": 20})
        self.assertEqual(self.records()[1]["measurement"], 10)
        other_records = [json.loads(line) for line in other.read_text().splitlines()]
        self.assertEqual(other_records[1]["measurement"], 20)


if __name__ == "__main__":
    unittest.main()
