"""CPU checks for validation selection, condition matching, and offline evidence."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from thesis_testing_tools.run_prediction_validation import condition_config, select_workload, report


class ValidationTests(unittest.TestCase):
    def base(self):
        return {"engine": {"training": {"hyper_params": {"initial_lr": 0.001}},
                "generation": {"visualization": {}}, "load_model_args": {"model_ckpt": "base"}},
                "wrapper": {"enable_modulators": True},
                "runner": {"source_type": "habitat", "adapter_params": {}}}

    def test_conditions_match_except_memory_and_update_switches(self):
        base = self.base()
        before = deepcopy(base)
        frozen = condition_config(base, "frozen_on", 8, {"habitat": ["7"]}, Path("manifest"), Path("actions"))
        adapted = condition_config(base, "adapted_on", 8, {"habitat": ["7"]}, Path("manifest"), Path("actions"))
        self.assertEqual(base, before)
        frozen["wrapper"]["training_enabled"] = True
        self.assertEqual(frozen, adapted)
        self.assertTrue(frozen["wrapper"]["add_global_memories"])
        self.assertEqual(frozen["wrapper"]["max_route_steps"], 1)
        self.assertEqual(frozen["runner"]["max_total_episodes"], 1)

    def test_normal_generator_receives_selected_episodes_and_shared_seed(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from thesis_testing_tools.run_prediction_validation import generate_validation_actions
        generate = Mock()
        with patch.dict(sys.modules, {"scripts.generate_habitat_action_sequence":
                                     SimpleNamespace(generate_action_sequence=generate)}):
            path = generate_validation_actions(self.base(), Path("tester.yaml"),
                ["827", "352"], Path("output/test"), 8, 100)
        self.assertEqual(path, Path("output/test/habitat_action_sequences"))
        self.assertEqual(generate.call_count, 2)
        for call, episode in zip(generate.call_args_list, ["827", "352"]):
            self.assertEqual(call.args, (episode, 8, Path("output/test")))
            self.assertEqual(call.kwargs, {"seed": 100, "config_path": Path("tester.yaml"), "checkpoint": "base"})

    def test_selection_rejects_heldout_and_unknown(self):
        cfg = self.base()
        development = {"habitat": {"test": ["7", "8"]}}
        heldout = {"habitat": {"test": ["8"]}}
        self.assertEqual(select_workload(cfg, ["7"], development, heldout), {"habitat": ["7"]})
        for episodes in (["8"], ["9"], ["7", "7"]):
            with self.assertRaises(ValueError):
                select_workload(cfg, episodes, development, heldout)

    def test_replay_requires_explicit_training_settings(self):
        cfg = self.base()
        cfg["engine"]["training"] = False
        with self.assertRaisesRegex(ValueError, "training mapping"):
            condition_config(cfg, "adapted_off", 1, {"habitat": ["7"]}, Path("m"), None)

    def test_action_probes_preserve_rng_and_do_not_replace_route(self):
        from dataclasses import dataclass
        import random
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        from thesis_testing_tools.run_prediction_validation import make_probe_engine
        @dataclass(frozen=True)
        class Bundle:
            action_text: str = "original"
            input_source: str = "input.png"
        route = SimpleNamespace(steps=[SimpleNamespace(visualization="image", action_text="original",
                                logging_info={"generation_id": 1})])
        class Base:
            def __init__(self):
                self.action_vocabulary = SimpleNamespace(bin_step=0.01, tokens_by_axis={"dyaw": ["<dyaw_pos_bin_20>"]})
                self.event_logger = SimpleNamespace(feed=lambda value: None)
                self.calls = []
            def reset_episode(self, *args): pass
            def predict_route(self, *args): return route
            def eval_step(self, bundle):
                self.calls.append((bundle.action_text, random.random()))
                return SimpleNamespace(visualization=None, logging_info={"generation_id": len(self.calls) + 1})
        modules = {
            "runtime_scripts.runtime_utils": SimpleNamespace(save_img=lambda *args: None),
            "thesis_testing_tools.evaluate_memory_refresh": SimpleNamespace(rng_state=random.getstate, restore_rng=random.setstate),
        }
        with patch.dict(sys.modules, modules):
            engine = make_probe_engine(Base, Path("unused"), 20)()
        engine.reset_episode("7", Bundle())
        state = random.getstate()
        self.assertIs(engine.predict_route(Bundle(), 1), route)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(len(engine.calls), 3)
        self.assertEqual(len({value for action, value in engine.calls}), 1)
        engine.predict_route(Bundle(), 1)
        self.assertEqual(len(engine.calls), 3)

    def test_report_uses_actual_next_frame_and_copy_baseline(self):
        from PIL import Image
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name, color in (("input", 0), ("pred", 255), ("target", 255)):
                Image.new("RGB", (12, 12), (color, color, color)).save(root / f"{name}.png")
            condition = root / "frozen_off"
            condition.mkdir()
            event = {"transition": {"outcome": "completed"}, "source_mode": "habitat", "data_id": "habitat",
                     "episode_id": "7", "step_idx": 0, "prediction": {"generation_id": 1, "action": "turn"},
                     "wrapper": {"predicted_obs_path": str(root / "pred.png"), "real_obs_path": str(root / "target.png")},
                     "diagnostics": {"generations": {"1": {"input_source": str(root / "input.png"), "completed_updates": 0}}}}
            (condition / "events.jsonl").write_text(json.dumps(event) + "\n")
            report(root, ["mae"])
            summary = json.loads((root / "summary.json").read_text())["conditions"]["frozen_off"]
            self.assertEqual(summary["mae"], 0)
            self.assertEqual(summary["copy_mae"], 1)
            self.assertTrue((root / "report.html").exists())
            other = root / "frozen_on"
            other.mkdir()
            event["prediction"]["action"] = "different"
            (other / "events.jsonl").write_text(json.dumps(event) + "\n")
            with self.assertRaisesRegex(ValueError, "Unmatched"):
                report(root, ["mae"])


if __name__ == "__main__":
    unittest.main()
