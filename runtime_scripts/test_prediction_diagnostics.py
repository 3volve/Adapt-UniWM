"""Exercise diagnostic recording without loading the GPU model stack."""
import ast
from contextlib import nullcontext
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from runtime_scripts.event_logger import EventLogger


class Tensor:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values[0]))

    def detach(self): return self
    def cpu(self): return self
    def clone(self): return self
    def mean(self): return self
    def __float__(self): return 0.5
    def tolist(self): return self.values


def load_method(name, scope, filename="runtime_engine.py", class_name="UniWMEngine"):
    path = Path(__file__).with_name(filename)
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[]))
    exec(compile(module, str(path), "exec"), scope)
    return scope[name]


class DiagnosticTests(unittest.TestCase):
    def test_generation_records_survive_merge_and_do_not_change_decoding(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), TemporaryDirectory() as directory:
                path = Path(directory) / "events.jsonl"
                decoded = []
                tokens = Tensor([[71, 72]])
                scope = {
                    "torch": SimpleNamespace(FloatTensor=Tensor, no_grad=nullcontext, autocast=lambda **kw: nullcontext()),
                    "cast": lambda kind, value: value,
                    "GenerateDecoderOnlyOutput": object,
                    "extract_generated_tokens": lambda outputs, length, count: tokens,
                    "decode_image": lambda model, processor, value: decoded.append(value.tolist()) or "image",
                }
                predict = load_method("_predict_visualization", scope)
                with EventLogger(path) as logger:
                    engine = SimpleNamespace(
                        _generation_id=0, _diagnostic_update_count=2,
                        event_logger=logger, memory_count=4, processor=None, _current_input_source="real.png",
                        config={"generation": {"visualization": {}, "diagnostics": enabled}},
                        model=SimpleNamespace(dtype=None, model=SimpleNamespace(bpe_indices=[]),
                            generate=lambda **kw: SimpleNamespace(scores=(1, 2))),
                        _get_viz_kwargs=lambda kw: ({"use_memory_bank": False, "use_global_memory_bank": False}, True),
                        _calculate_entropy=lambda *a, **kw: Tensor([[0]]),
                    )
                    for _ in range(2):
                        self.assertEqual(predict(engine, {"input_ids": Tensor([[1, 2, 3]])}), ("image", 0.5, True))
                event = json.loads(path.read_text().splitlines()[0])
                self.assertEqual(decoded, [[[71, 72]], [[71, 72]]])
                self.assertEqual(engine._generation_id, 2)
                if enabled:
                    generations = event["diagnostics"]["generations"]
                    self.assertEqual(set(generations), {"1", "2"})
                    self.assertEqual(generations["1"]["completed_updates"], 2)
                    self.assertEqual(generations["1"]["decoded_token_ids"], [[71, 72]])
                    self.assertEqual(generations["1"]["prompt_length"], 3)
                    self.assertEqual(generations["1"]["input_source"], "real.png")
                    self.assertFalse(generations["1"]["use_memory_bank"])
                else:
                    self.assertNotIn("diagnostics", event)

    def test_input_source_restores_with_tokens_after_imagined_updates(self):
        store = load_method("_store_state", {})
        restore = load_method("_restore_state", {})
        engine = SimpleNamespace(
            _memory_manager=SimpleNamespace(cache_step_state=lambda: None, load_cached_state=lambda: None),
            _current_tok_obs="real tokens", _current_input_source="real.png",
            _context_familiarity=0.2, _context_stability=0.4, memory_count=3, prior_actions=["turn"],
        )
        state = store(engine)
        engine._current_tok_obs = "imagined tokens"
        engine._current_input_source = 17
        engine.prior_actions.append("forward")
        restore(engine, *state)
        self.assertEqual(engine._current_tok_obs, "real tokens")
        self.assertEqual(engine._current_input_source, "real.png")
        self.assertEqual(engine.prior_actions, ["turn"])

    def test_wrapper_initial_source_reuses_saved_start(self):
        from dataclasses import dataclass, replace
        @dataclass(frozen=True)
        class Bundle:
            start_observation: object
            goal_observation: object
            current_observation: object
            input_source: object = None
        saved, received = [], []
        scope = {"replace": replace,
                 "build_img_paths": lambda root, episode, names: [name + ".png" for name in names],
                 "save_img": lambda image, path: saved.append(path)}
        reset = load_method("reset_episode", scope, "uniwm_wrapper.py", "UniWMWrapper")
        wrapper = SimpleNamespace(output_path="output", _reset_wrapper_state=lambda: None,
            engine=SimpleNamespace(reset_episode=lambda episode, bundle: received.append(bundle)),
            _plan_route=lambda bundle, reason: received.append(bundle), get_state_snapshot=lambda: {})
        start = object()
        reset(wrapper, Bundle(start, object(), start), "episode")
        self.assertEqual([bundle.input_source for bundle in received], ["start.png", "start.png"])
        self.assertEqual(saved, ["start.png", "goal.png"])

    def test_training_batch_logged_without_altering_supervision(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            batch = {"input_ids": Tensor([[1, 2, 71]]), "labels": Tensor([[-100, -100, 71]]),
                     "attention_mask": Tensor([[1, 1, 1]])}
            scope = {"torch": SimpleNamespace(autocast=lambda **kw: nullcontext()),
                     "build_viz_prompt": lambda **kw: "prompt",
                     "compute_supervised_uniwm_loss": lambda *a, **kw: kw["batch"]}
            loss = load_method("_compute_viz_step_loss", scope)
            class Model:
                dtype = None
                def __call__(self, **kw): return None
            with EventLogger(path) as logger:
                engine = SimpleNamespace(
                    event_logger=logger, _diagnostic_update_count=0, _current_input_source=7,
                    config={"training": {"diagnostics": True, "visualization": {}, "loss": {"ignore_index": -100}}},
                    _processor_inputs_from_prompt=lambda **kw: {"input_ids": Tensor([[1, 2]])},
                    _image_to_vq_bpe_tokens=lambda image: Tensor([[71]]),
                    _build_image_batch=lambda **kw: batch,
                    _get_viz_kwargs=lambda *a: ({"use_memory_bank": False}, False),
                    model=Model(), action_vocabulary=None, processor=None,
                )
                prediction = SimpleNamespace(real_input_obs="input", real_next_obs="target", action_text="turn",
                    input_bundle=SimpleNamespace(start_pose_str="pose"), viz_used_memory=False,
                    logging_info={"generation_id": 1})
                self.assertIs(loss(engine, prediction), batch)
            record = json.loads(path.read_text().splitlines()[0])["diagnostics"]["teacher_forced"]
            self.assertEqual(record["batch"]["labels"], [[-100, -100, 71]])
            self.assertEqual(record["prompt_length"], 2)
            self.assertEqual(record["generation_id"], 1)
            self.assertEqual(record["input_source"], 7)


if __name__ == "__main__":
    unittest.main()
