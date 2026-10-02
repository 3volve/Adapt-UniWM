"""Test producer serialization without importing the model stack."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from runtime_scripts.event_logger import EventLogger


def load_event_values():
    path = Path(__file__).with_name("runtime_utils.py")
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "event_values")
    scope = {"np": np, "Path": Path}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    return scope["event_values"]


class EventValuesTests(unittest.TestCase):
    def test_peft_sets_survive_startup_flush_and_exception_cleanup(self):
        event_values = load_event_values()
        target_modules = {"v_proj", "q_proj"}
        metadata = {"model": {"peft_config": {"default": {
            "target_modules": target_modules, "modules_to_save": frozenset({"head"})}}}}
        converted = event_values(metadata)
        self.assertEqual(converted["model"]["peft_config"]["default"]["target_modules"], ["q_proj", "v_proj"])
        self.assertIsInstance(target_modules, set)
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "events.jsonl"
                def run():
                    with EventLogger(path) as log:
                        log.feed({"runtime": {"after_model": converted}})
                        log.next_step({"episode_id": "first"})
                        if fail:
                            raise RuntimeError("original failure")
                        log.finish({"outcome": "completed"})
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "original failure"):
                        run()
                else:
                    run()
                records = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(records[0]["runtime"]["after_model"], converted)
                self.assertEqual(records[-1]["outcome"], "failed" if fail else "completed")


if __name__ == "__main__":
    unittest.main()
