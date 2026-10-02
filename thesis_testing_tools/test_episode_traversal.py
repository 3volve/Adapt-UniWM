"""Exercise runtime control flow without importing the model/simulator stack."""
import ast
import itertools
import pickle
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


def method(path, name, **scope):
    tree = ast.parse(Path(path).read_text())
    node = next(n for c in tree.body if isinstance(c, ast.ClassDef)
                for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), path, 'exec'), scope)
    return scope[name]


class EpisodeTraversalTests(unittest.TestCase):
    def test_replay_preserves_dataset_and_episode_order_and_exhausts(self):
        advance = method('source_tools/replay_source_tools.py', 'next_episode', pickle=pickle)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = {'b': {'test': ['2', '1']}, 'a': {'test': ['7']}}
            for dataset, split in selection.items():
                for episode in split['test']:
                    path = root / dataset / episode
                    path.mkdir(parents=True)
                    (path / 'traj_data.pkl').write_bytes(pickle.dumps({}))
            adapter = SimpleNamespace(data_root=root, datasets=iter(selection.items()),
                episodes=iter(()), manifest_split='test', _make_states_xy_yaw=lambda: [],
                _make_actions=lambda: [])
            adapter._pack_step = lambda *args: (adapter.data_id, adapter.current_episode_id)
            self.assertEqual([advance(adapter)[0] for _ in range(3)], [('b', '2'), ('b', '1'), ('a', '7')])
            self.assertIsNone(advance(adapter))
            self.assertIsNone(advance(adapter))
            adapter.datasets = iter({'missing': {'test': ['bad']}}.items())
            with self.assertRaises(FileNotFoundError):
                advance(adapter)

    def test_habitat_uses_selected_episode_and_propagates_reset_failure(self):
        advance = method('source_tools/habitat_source_tools.py', 'next_episode')
        env = SimpleNamespace(episode_over=False)
        calls = []
        def reset():
            calls.append(env.current_episode.episode_id)
            return {}
        env.reset = reset
        adapter = SimpleNamespace(selected_episodes=iter([SimpleNamespace(episode_id=e) for e in ['7', '2']]),
            env=env, fixed_actions_by_episode=None, _update_action_specs=lambda: None,
            _pack_step=lambda *args, **kwargs: env.current_episode.episode_id)
        self.assertEqual(advance(adapter), ['7'])
        self.assertEqual(advance(adapter), ['2'])
        self.assertIsNone(advance(adapter))
        self.assertEqual(calls, ['7', '2'])
        adapter.selected_episodes = iter([SimpleNamespace(episode_id='broken')])
        env.reset = Mock(side_effect=RuntimeError('simulator failed'))
        with self.assertRaisesRegex(RuntimeError, 'simulator failed'):
            advance(adapter)

    def test_runner_exhaustion_exact_ceiling_and_infinite_adapter(self):
        run = method('uniwm_episode_runner.py', 'run_episodes')
        for count in (0, 2, 3, None):
            with self.subTest(count=count):
                initial = [SimpleNamespace(data_id='a', episode_id='1')]
                episodes = itertools.repeat(initial) if count is None else iter([initial] * count)
                runner = SimpleNamespace(episode_index=0, config={'max_total_episodes': 2, 'save_model_weights': True},
                    full_output_path=Path('output'), event_logger=Mock(),
                    adapter=SimpleNamespace(next_episode=lambda: next(episodes, None)), wrapper=Mock())
                def episode(value):
                    runner.episode_index += 1
                runner.run_episode = episode
                if count is None or count > 2:
                    with self.assertRaisesRegex(RuntimeError, 'max_total_episodes'):
                        run(runner)
                    runner.wrapper.engine.save_online_training_state.assert_not_called()
                else:
                    run(runner)
                    runner.wrapper.engine.save_online_training_state.assert_called_once_with(Path('output/final_ckpt'))
                    runner.wrapper.finalize_learning_rate_schedule.assert_called_once()
                self.assertEqual(runner.episode_index, min(count, 2) if count is not None else 2)


if __name__ == '__main__':
    unittest.main()
