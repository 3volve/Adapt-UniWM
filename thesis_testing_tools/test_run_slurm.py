import contextlib
import ast
from dataclasses import asdict
import io
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

from thesis_testing_tools import pipeline, run_slurm as slurm
from thesis_testing_tools.run_manifest import write_json, now


class SlurmTests(unittest.TestCase):
    def test_runner_uses_each_dataset_count(self):
        tree = ast.parse(Path('uniwm_episode_runner.py').read_text())
        runner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and any(
            isinstance(m, ast.FunctionDef) and m.name == 'run_episodes' for m in n.body))
        method = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == 'run_episodes')
        scope = {'Path': Path}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<runner>', 'exec'), scope)
        calls = []
        def episode(dataset):
            calls.append(dataset)
            fake.episode_index += 1
        fake = types.SimpleNamespace(config={'save_model_weights': False}, episode_index=0,
            event_logger=types.SimpleNamespace(feed=lambda value: None),
            adapter=types.SimpleNamespace(reset_src=lambda dataset: None),
            wrapper=types.SimpleNamespace(save_learning_rate_schedule=lambda: None,
                                          finalize_learning_rate_schedule=lambda: None), run_episode=episode)
        scope['run_episodes'](fake, 99, 'a,b', Path('.'), {'a': 3, 'b': 1})
        self.assertEqual(calls, ['a', 'a', 'a', 'b'])

    def test_cli_manifest_selection_and_caps(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            run, _ = self.fixture(Path(temp))
            manifest = slurm.read_json(run.source_manifest)
            first = pipeline.SOURCE_DATA_IDS.split(',')[0]
            manifest[first]['test'].append('c')
            write_json(run.source_manifest, manifest)
            args = ['submit', '--seed', '7', '--fixed-mean-lr', '0.00005',
                    '--source-manifest', str(run.source_manifest), '--initial-checkpoint', temp,
                    '--output-root', str(Path(temp) / 'out'), '--dry-run']
            with patch.object(slurm, 'submit') as submit:
                slurm.main(args)
                selected, counts, _ = submit.call_args.args
                self.assertEqual(counts[first], 3)
                self.assertEqual(selected.habitat_ids, ['352'])
                self.assertEqual(selected.metadata['source_episode_order'][first], ['a', 'b', 'c'])
                slurm.main(args + ['--source-episodes', '1', '--habitat-episodes', '10'])
                self.assertEqual(set(submit.call_args.args[1].values()), {1})

    def fixture(self, root):
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps({'habitat': {'test': ['352']},
            **{d: {'test': ['a', 'b']} for d in pipeline.SOURCE_DATA_IDS.split(',')}}))
        run = pipeline.resolve_seed_run(seed=7, fixed_mean_lr=0.00005,
            source_manifest=manifest, habitat_episodes=1, source_episodes=2,
            max_episode_steps=2, output_root=root / 'output', timestamp='test')
        resources = dict(repository_root=str(Path.cwd()), cpus=4, memory='32G',
                         time='12:00:00', finalize_time='02:00:00', gres='gpu:1')
        return run, resources

    def test_graph_and_submission_flags(self):
        graph = slurm.job_graph()
        self.assertEqual(len(graph), 14)
        self.assertEqual(graph['source_pre'], ['prepare'])
        for cid in ('c3_aligned_replay', 'c4_shuffled_replay'):
            self.assertEqual(graph[cid + '/habitat'], ['c0_frozen/habitat'])
            self.assertEqual(graph[cid + '/source_post'], [cid + '/habitat'])
        self.assertEqual(graph['finalize'], list(graph)[:-1])
        with tempfile.TemporaryDirectory() as temp:
            run, resources = self.fixture(Path(temp))
            ids = {name: str(i) for i, name in enumerate(graph, 1)}
            prepare = slurm.sbatch_command('prepare', [], ids, run.seed_dir, Path(temp), resources)
            self.assertIn('--hold', prepare)
            final = slurm.sbatch_command('finalize', graph['finalize'], ids, run.seed_dir, Path(temp), resources)
            self.assertTrue(any(v.startswith('--dependency=afterany:') for v in final))
            self.assertFalse(any(v.startswith('--gres') for v in final))
            self.assertIn('--no-requeue', final)
            self.assertIn('--kill-on-invalid-dep=yes', final)

    def test_dry_run_and_partial_submission(self):
        with tempfile.TemporaryDirectory() as temp:
            run, resources = self.fixture(Path(temp))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                slurm.submit(run, {'a': 2}, resources, dry_run=True,
                             runner=lambda *a, **k: self.fail('dry run submitted'))
            self.assertEqual(len(json.loads(output.getvalue())['jobs']), 14)
            self.assertFalse(run.seed_dir.exists())
            calls = []
            def submit(command, **kwargs):
                calls.append(command)
                if len(calls) == 3:
                    raise subprocess.CalledProcessError(1, command)
                return types.SimpleNamespace(stdout=str(100 + len(calls)))
            with patch.object(slurm, 'snapshot_code', return_value=Path(temp)), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(subprocess.CalledProcessError):
                    slurm.submit(run, {'a': 2}, resources, runner=submit)
            registry = slurm.read_json(run.seed_dir / 'submission.json')
            self.assertEqual(registry['status'], 'submission_failed')
            self.assertEqual(len(registry['jobs']), 2)
            self.assertTrue(all(c[0] == 'sbatch' for c in calls))
            self.assertNotEqual(registry['run']['source_manifest'], str(run.source_manifest))

    def test_submission_releases_only_after_registry_is_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            run, resources = self.fixture(Path(temp))
            calls = []
            def submit(command, **kwargs):
                calls.append(command)
                if command[0] == 'scontrol':
                    registry = slurm.read_json(run.seed_dir / 'submission.json')
                    self.assertEqual(registry['status'], 'submitted')
                    self.assertEqual(len(registry['jobs']), 14)
                    self.assertEqual(command, ['scontrol', 'release', '101'])
                return types.SimpleNamespace(stdout=str(100 + len(calls)) + ';cluster\n')
            with patch.object(slurm, 'snapshot_code', return_value=Path(temp)), contextlib.redirect_stdout(io.StringIO()):
                slurm.submit(run, {'a': 2}, resources, runner=submit)
            self.assertEqual(len(calls), 15)

    def test_preparation_worker_ownership_and_finalization(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as temp:
                run, resources = self.fixture(Path(temp))
                root = run.seed_dir
                registry = dict(run=asdict(run), resources=resources, started_at=now(),
                    source_counts=run.source_episode_counts,
                    jobs={name: {'job_id': str(i)} for i, name in enumerate(slurm.job_graph(), 1)})
                def actions(episode_id, count, seed_dir, **kwargs):
                    path = seed_dir / 'habitat_action_sequences' / '352.json'
                    path.parent.mkdir()
                    write_json(path, dict(episode_id=episode_id, target_steps=count, actions=['forward'] * count))
                    return str(path)
                with patch.object(pipeline, 'generate_action_sequence', side_effect=actions), patch.object(
                    pipeline, '_captured_command', return_value={}):
                    slurm.prepare(root, registry)
                before = (root / 'run_manifest.json').read_bytes()
                plan = slurm.read_json(root / 'seed_manifest.json')
                self.assertEqual(plan['workload']['source_episode_counts'], registry['source_counts'])
                source = plan['stages']['source_pre']['command']
                self.assertIn('--standalone', source)
                self.assertEqual(json.loads(source[source.index('--source-episode-counts') + 1]), registry['source_counts'])
                def execute(stage, **kwargs):
                    if fail and stage['stage_id'] == 'source_pre':
                        raise subprocess.CalledProcessError(9, stage['command'])
                    return dict(stage, status='completed')
                with patch.object(pipeline, 'execute_seed_stage', side_effect=execute), patch.object(
                    pipeline, 'collect_habitat_artifacts', return_value={}):
                    for name in plan['stages']:
                        if fail and name == 'source_pre':
                            with self.assertRaises(subprocess.CalledProcessError):
                                slurm.execute(root, name, registry)
                        else:
                            slurm.execute(root, name, registry)
                self.assertEqual(before, (root / 'run_manifest.json').read_bytes())
                with self.assertRaises(FileExistsError):
                    slurm.execute(root, 'source_pre', registry)
                with patch.object(pipeline, 'generate_run_outputs') as report:
                    slurm.finalize(root, registry, outcomes={})
                    report.assert_called_once_with(root)
                final = slurm.read_json(root / 'run_manifest.json')
                self.assertEqual(final['status'], 'failed' if fail else 'completed')
                self.assertEqual(len(final['stages']), 12)
                summary = slurm.read_json(root / 'seed_summary.json')
                self.assertEqual(summary['status'], final['status'])
                if fail:
                    self.assertEqual(final['stages'][0]['failure']['returncode'], 9)
                with self.assertRaises(FileExistsError):
                    slurm.finalize(root, registry, outcomes={})

    def test_finalization_without_preparation(self):
        with tempfile.TemporaryDirectory() as temp:
            run, resources = self.fixture(Path(temp))
            run.seed_dir.mkdir(parents=True)
            registry = dict(run=asdict(run), jobs={name: {'job_id': str(i)}
                            for i, name in enumerate(slurm.job_graph(), 1)})
            with patch.object(pipeline, 'generate_run_outputs'):
                slurm.finalize(run.seed_dir, registry, outcomes={})
            manifest = slurm.read_json(run.seed_dir / 'run_manifest.json')
            self.assertEqual(manifest['status'], 'failed')
            self.assertTrue(all(s['status'] == 'not_started' for s in manifest['stages']))


if __name__ == '__main__':
    unittest.main()
