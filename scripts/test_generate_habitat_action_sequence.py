import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from scripts.generate_habitat_action_sequence import (
    REPO_ROOT, collect_sequence, extend_action, generate_action_sequence,
)


class GenerationTests(unittest.TestCase):
    def adapter(self, collision_at=None):
        position = [0., 0., 0.]
        adapter = Mock()
        adapter.moves = []
        adapter.sim.get_agent_state.side_effect = lambda: NS(position=position.copy())
        adapter.sim.pathfinder.get_random_navigable_point.return_value = [100., 0., 0.]
        adapter.sim.geodesic_distance.side_effect = math.dist
        def step(actions):
            adapter.moves.extend(actions)
            if actions[0] == "move_forward":
                position[0] += .25
            return [NS(is_collision=len(adapter.moves) == collision_at, done=False)]
        adapter.step.side_effect = step
        return adapter

    def collect(self, adapter, factory, target, max_waypoints=4):
        return collect_sequence(adapter, factory, {1: "move_forward", 2: "turn_left", 3: "turn_right"},
                                0, LookupError, target, 60, 208, 10, max_waypoints)

    def test_exact_budget_no_extra_movement(self):
        adapter = self.adapter()
        actions, collisions, _, _ = self.collect(adapter, lambda: NS(get_next_action=lambda goal: 1), 3)
        self.assertEqual(len(actions), 3)
        self.assertEqual(adapter.moves, ["move_forward"] * 6)
        self.assertTrue(all("dx_pos_bin_50" in text for text in actions))
        self.assertEqual(collisions, [False] * 3)

    def test_collision_counts_and_ends_group(self):
        adapter = self.adapter(collision_at=1)
        actions, collisions, _, _ = self.collect(adapter, lambda: NS(get_next_action=lambda goal: 1), 1)
        self.assertEqual(len(actions), 1)
        self.assertIn("dx_pos_bin_25", actions[0])
        self.assertEqual(collisions, [True])
        self.assertEqual(len(adapter.moves), 1)

    def test_turn_order_and_waypoint_flush(self):
        choices = iter([1, 2, 3, 0])
        actions, _, _, _ = self.collect(self.adapter(), lambda: NS(get_next_action=lambda goal: next(choices)), 2)
        self.assertIn("dx_pos_bin_25", actions[0])
        self.assertIn("dyaw_pos_bin_17", actions[0])
        self.assertIn("dx_pos_bin_00", actions[1])
        self.assertIn("dyaw_neg_bin_17", actions[1])
        self.assertIsNone(extend_action((25, 17), "move_forward", 60, 208))

    def test_follower_failures_are_bounded(self):
        follower = NS(get_next_action=Mock(side_effect=LookupError))
        with self.assertRaisesRegex(RuntimeError, "0/3 transitions.*2 follower failures"):
            self.collect(self.adapter(), lambda: follower, 3, max_waypoints=2)

    def test_unreachable_candidates_are_bounded(self):
        adapter = self.adapter()
        adapter.sim.geodesic_distance.side_effect = None
        adapter.sim.geodesic_distance.return_value = float("inf")
        factory = Mock()
        with self.assertRaisesRegex(RuntimeError, "0/3 transitions"):
            self.collect(adapter, factory, 3, max_waypoints=2)
        factory.assert_not_called()

    def test_rejects_invalid_inputs_before_loading_habitat(self):
        for episode, count, directory in (("../827", 25, "output"), ("827", 0, "output"),
                                           ("827", 25, REPO_ROOT.parent)):
            with self.subTest(episode=episode, count=count, directory=directory), self.assertRaises(ValueError):
                generate_action_sequence(episode, count, directory)

    def test_public_function_writes_complete_file_and_returns_relative_path(self):
        config = {"runner": {"adapter_params": {"bin_step": .01, "data_path": "data/val.json.gz",
                                                 "scenes_dir": "data/scenes"}}}
        vocab = NS(bin_step=.01, axes={"dx": {"max_bin": 60},
                                      "dyaw": {"max_bin": 208, "allow_negative": True}})
        adapter = self.adapter()
        adapter.current_episode = NS(scene_id="test-scene")
        adapter.sim.habitat_config = NS(default_agent_id=0)
        space = {i: NS(actuation=NS(amount=0)) for i in (1, 2, 3)}
        adapter.sim.sim_config = NS(agents=[NS(action_space=space)])
        adapter.sim.get_agent.return_value = NS(agent_config=NS(action_space=space))
        modules = {
            "omegaconf": NS(OmegaConf=NS(load=lambda path: config, to_container=lambda x, **kw: x)),
            "source_tools.habitat_source_tools": NS(HabitatEpisodeAdapter=lambda **kw: adapter),
            "habitat.tasks.nav.shortest_path_follower": NS(ShortestPathFollower=Mock()),
            "habitat.sims.habitat_simulator.actions": NS(HabitatSimActions=NS(stop=0, move_forward=1, turn_left=2, turn_right=3)),
            "habitat_sim": NS(errors=NS(GreedyFollowerError=LookupError)),
        }
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as directory, patch.dict("sys.modules", modules), patch(
            "scripts.action_utils.ActionTokenVocabulary.from_checkpoint", return_value=vocab
        ), patch("scripts.generate_habitat_action_sequence.collect_sequence", return_value=(["test-action"], [False], [], 0)):
            result = generate_action_sequence("827", 1, directory)
            self.assertFalse(Path(result).is_absolute())
            self.assertNotIn("\\", result)
            path = REPO_ROOT / result
            self.assertTrue(path.is_file())
            self.assertTrue(path.is_relative_to(Path(directory)))
            self.assertEqual(json.loads(path.read_text())["actions"], ["test-action"])
            adapter.close.assert_called_once()
            with patch("scripts.generate_habitat_action_sequence.collect_sequence",
                       side_effect=RuntimeError("generation budget exhausted")):
                with self.assertRaisesRegex(RuntimeError, "generation budget exhausted"):
                    generate_action_sequence("827", 2, directory)
            self.assertFalse((path.parent / "episode_827_seed_100_steps_2.json").exists())
            self.assertEqual(adapter.close.call_count, 2)


if __name__ == "__main__":
    unittest.main()
