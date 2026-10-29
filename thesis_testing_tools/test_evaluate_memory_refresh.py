"""CPU tests of the standalone evaluator; no claim of real-model validation."""
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from thesis_testing_tools.evaluate_memory_refresh import compare_memory, next_forced_action, sampling_steps, summarize


class FakeEngine:
    def __init__(self, fail=False):
        self.model = SimpleNamespace(training=False)
        self.attention = SimpleNamespace(stored_keys=99, stored_values=199,
            global_stored_keys=[1, 2], global_stored_values=[3, 4], memory_top_k=3,
            compute_similarity_and_select_topk=lambda keys, k: ([0, 1], [0.5, 0.4]))
        self.config = {"generation": {"visualization": {"use_memory_bank": True, "use_global_memory_bank": True}}}
        self._memory_manager = SimpleNamespace(_cached_step_state=None)
        self.prior_actions = ["previous"]
        self.observed, self.fail = [], fail

    def attentions(self):
        return [self.attention]

    def _store_state(self):
        self._memory_manager._cached_step_state = SimpleNamespace(
            stored_keys=[self.attention.stored_keys], stored_values=[self.attention.stored_values])
        return (self.attention.stored_keys, self.attention.stored_values, list(self.prior_actions))

    def _restore_state(self, keys, values, actions):
        self.attention.stored_keys, self.attention.stored_values = keys, values
        self.prior_actions = actions
        self._memory_manager._cached_step_state = None

    def refresh_history(self):
        self.attention.global_stored_keys = [11, 12]
        self.attention.global_stored_values = [13, 14]
        self.attention.stored_keys = -1
        self.attention.stored_values = -1

    def _predict_step(self, bundle):
        self.observed.append((self.attention.stored_keys, list(self.attention.global_stored_keys),
            random.random(), np.random.random(), self.config["generation"]["visualization"]["use_memory_bank"]))
        self.prior_actions.append(bundle.action_text)
        if self.fail and len(self.observed) == 3:
            raise RuntimeError("synthetic inference failure")
        return SimpleNamespace(visualization=None)


class EvaluationTest(unittest.TestCase):
    def check_isolation(self, fail):
        engine = FakeEngine(fail)
        fake_torch = SimpleNamespace(get_rng_state=lambda: 0, set_rng_state=lambda value: None,
            cuda=SimpleNamespace(get_rng_state_all=lambda: [], set_rng_state_all=lambda value: None))
        before_python, before_numpy = random.getstate(), np.random.get_state()
        with patch.dict("sys.modules", {"torch": fake_torch}):
            if fail:
                with self.assertRaisesRegex(RuntimeError, "synthetic"):
                    compare_memory(engine, SimpleNamespace(action_text="fixed"))
            else:
                predictions, metadata = compare_memory(engine, SimpleNamespace(action_text="fixed"))
                self.assertEqual(set(predictions), {"bypass", "cached", "refreshed"})
                self.assertFalse(metadata["cached"]["available"])
        self.assertEqual(before_python, random.getstate())
        np.testing.assert_equal(before_numpy, np.random.get_state())
        self.assertEqual([x[0] for x in engine.observed], [99, 99, 99])
        self.assertEqual([x[1] for x in engine.observed], [[1, 2], [1, 2], [11, 12]])
        self.assertEqual([x[4] for x in engine.observed], [False, True, True])
        self.assertEqual(len({(x[2], x[3]) for x in engine.observed}), 1)
        self.assertEqual(engine.attention.global_stored_keys, [1, 2])
        self.assertEqual(engine.attention.global_stored_values, [3, 4])
        self.assertEqual(engine.prior_actions, ["previous"])
        self.assertIsNone(engine._memory_manager._cached_step_state)
        self.assertTrue(engine.config["generation"]["visualization"]["use_memory_bank"])

    def test_success_isolation(self):
        self.check_isolation(False)

    def test_error_isolation(self):
        self.check_isolation(True)

    def test_forced_action_alignment(self):
        self.assertEqual(next_forced_action(SimpleNamespace(action_text=["previous", "next"])), "next")
        self.assertEqual(next_forced_action(SimpleNamespace(action_text="next")), "next")
        self.assertIsNone(next_forced_action(SimpleNamespace(action_text=["previous"])))
        with self.assertRaisesRegex(ValueError, "fixed actions"):
            next_forced_action(SimpleNamespace(action_text=None))

    def test_pair_coverage_and_episode_weighting(self):
        def row(ep, delta, bypass):
            return {"episode_id": ep, "conditions": {"cached": {"scores": {"mae": 1}},
                "refreshed": {"scores": {"mae": 1 + delta}}, "bypass": {"scores": bypass}}}
        result = summarize([row("a", -0.5, None), row("a", -0.5, None), row("b", 0, {"mae": 1})], ["mae"])
        pair = result["refreshed_minus_cached"]["mae"]
        self.assertAlmostEqual(pair["mean_delta"], -1 / 3)
        self.assertEqual(pair["equal_episode_mean_delta"], -0.25)
        self.assertEqual(result["cached_minus_bypass"]["mae"]["paired_contexts"], 1)

    def test_contexts_span_memory_ages(self):
        slots = sampling_steps(25, 3, 5)
        self.assertEqual(slots[0], 3)
        self.assertEqual(slots[-1], 24)
        self.assertEqual(len(set(slots)), 5)
        self.assertEqual(sampling_steps(3, 3, 5), [])
        self.assertEqual(sampling_steps(4, 3, 5), [3])


if __name__ == "__main__":
    unittest.main()
