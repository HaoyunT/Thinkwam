"""CPU-only protocol fixtures; no policy inference, simulator, or real checkpoint."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.robotwin_eval import aggregate, make_plan, manifest


class RoboTwinProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name).resolve()
        self.root = self.directory / 'evaluation'
        self.checkpoint = self.directory / 'checkpoint.pt'
        self.checkpoint.write_bytes(b'synthetic checkpoint A')
        self.simulator = self.directory / 'simulator'
        self.simulator.mkdir()
        self.stats = self.directory / 'stats.json'
        self.stats.write_text('{}', encoding='utf-8')
        self.frames = self.directory / 'frames.json'
        self.frames.write_text('{"cameras": []}', encoding='utf-8')
        assets = {key: dict(path=str(path), sha256=manifest.sha256_file(path))
                  for key, path in [('stats', self.stats), ('frame_manifest', self.frames)]}
        self.plan = manifest.build_plan(dict(
            checkpoint=str(self.checkpoint), checkpoint_sha256=manifest.sha256_file(self.checkpoint),
            checkpoint_config_sha256='c' * 64), self.simulator, assets=assets)
        manifest.ensure_plan(self.root, self.plan)

    def row(self, condition, task, episode, success=None):
        if success is None:
            success = episode % 2 == 0
        return dict(**manifest.identity(self.plan), record_type='formal_episode',
                    condition=condition, task=task, episode=episode,
                    success=success, scene_seed=100000 + episode, instruction='Do the task.',
                    steps=25 if success else 100, step_limit=100, seconds=1.0,
                    completed_at=1000.0 + episode, initial_rgb_sha256='b' * 64)

    def write_rows(self, rows):
        streams = {}
        try:
            for rank in range(self.plan['world_size']):
                directory = self.root / 'parts' / f'rank{rank}'
                directory.mkdir(parents=True, exist_ok=True)
                streams[rank] = (directory / 'episodes.jsonl').open('w', encoding='utf-8')
            for row in rows:
                job = [row['condition'], row['task']]
                rank = self.plan['jobs'].index(job) % self.plan['world_size'] if job in self.plan['jobs'] else 0
                streams[rank].write(json.dumps(row) + '\n')
        finally:
            for stream in streams.values():
                stream.close()

    def full_rows(self):
        return [self.row(*key) for key in manifest.expected_cases()]

    def test_complete_success_and_failure_records_have_fixed_denominators(self):
        self.write_rows(self.full_rows())
        summary = aggregate.aggregate(self.root)
        for value in summary['conditions'].values():
            self.assertEqual(value, dict(successes=1000, episodes=2000, success_rate=0.5))
        self.assertEqual(summary['clean_random_average'], 0.5)
        self.assertEqual(summary['checkpoint_sha256'], self.plan['checkpoint_sha256'])
        rows = manifest.read_records(self.root, self.plan, complete=True)
        self.assertEqual(sum(not row['success'] for row in rows.values()), 2000)
        self.assertEqual(len(summary['tasks']), 20)
        for counts in summary['tasks'].values():
            self.assertTrue(all(value['episodes'] == 100 for value in counts.values()))

    def test_missing_case_cannot_report_a_score(self):
        self.write_rows(self.full_rows()[:-1])
        with self.assertRaisesRegex(ValueError, 'missing=1'):
            aggregate.aggregate(self.root)
        self.assertFalse((self.root / 'summary.json').exists())

    def test_extra_case_cannot_report_a_score(self):
        rows = self.full_rows()
        extra = copy.deepcopy(rows[-1])
        extra['episode'] = 100
        self.write_rows(rows + [extra])
        with self.assertRaisesRegex(ValueError, 'episode index'):
            aggregate.aggregate(self.root)

    def test_duplicate_failure_then_success_is_rejected_not_selected(self):
        rows = self.full_rows()
        retry = copy.deepcopy(rows[1])
        retry['success'] = True
        retry['steps'] = 25
        self.write_rows(rows + [retry])
        with self.assertRaisesRegex(ValueError, 'duplicate episode'):
            aggregate.aggregate(self.root)

    def test_4000_records_with_wrong_condition_task_distribution_rejected(self):
        rows = [self.row('demo_clean', self.plan['tasks'][0], i) for i in range(3999)]
        rows.append(self.row('demo_randomized', self.plan['tasks'][0], 0))
        self.assertEqual(len(rows), 4000)
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError, 'episode index'):
            aggregate.aggregate(self.root)

    def test_wrong_metadata_and_non_boolean_success_rejected(self):
        original = self.row(*manifest.expected_cases()[0])
        for field, value in [('checkpoint_sha256', 'd' * 64), ('checkpoint_config_sha256', 'e' * 64),
                             ('plan_sha256', 'f' * 64), ('case_manifest_sha256', 'a' * 64),
                             ('checkpoint_step', 75000), ('flow_steps', 10), ('action_horizon', 16),
                             ('replan_steps', 16), ('schema_version', True), ('success', 1),
                             ('success', 'false'), ('record_type', 'preflight'),
                             ('condition', 'unknown'), ('task', 'unknown'), ('episode', True)]:
            with self.subTest(field=field, value=value):
                row = dict(original, **{field: value})
                self.write_rows([row])
                with self.assertRaises(ValueError):
                    manifest.read_records(self.root, self.plan)

    def test_legacy_rows_and_legacy_plan_are_rejected(self):
        row = self.row(*manifest.expected_cases()[0])
        del row['checkpoint_sha256']
        self.write_rows([row])
        with self.assertRaisesRegex(ValueError, 'checkpoint_sha256'):
            manifest.ensure_plan(self.root, self.plan)
        legacy = dict(self.plan)
        del legacy['plan_sha256']
        manifest.write_json(self.root / 'plan.json', legacy)
        with self.assertRaisesRegex(ValueError, 'legacy'):
            manifest.load_plan(self.root)

    def test_consistent_partial_resume_preserves_success_and_failure(self):
        rows = [self.row(*key) for key in manifest.expected_cases()[:17]]
        self.write_rows(rows)
        log = self.root / 'parts/rank0/episodes.jsonl'
        before = log.read_bytes()
        self.assertFalse(manifest.ensure_plan(self.root, self.plan))
        actual = manifest.read_records(self.root, self.plan, rank=0)
        self.assertEqual(len(actual), 17)
        self.assertEqual(sum(not row['success'] for row in actual.values()), 8)
        self.assertEqual(before, log.read_bytes())

    def test_changed_plan_cannot_relabel_existing_results(self):
        self.write_rows([self.row(*manifest.expected_cases()[0])])
        before = (self.root / 'plan.json').read_bytes()
        changed = dict(self.plan)
        changed['checkpoint_sha256'] = 'f' * 64
        changed.pop('plan_sha256')
        changed['plan_sha256'] = manifest.digest(changed)
        with self.assertRaisesRegex(ValueError, 'different plan/checkpoint'):
            manifest.ensure_plan(self.root, changed)
        self.assertEqual(before, (self.root / 'plan.json').read_bytes())
        # Even a manually replaced, internally valid plan cannot relabel rows.
        manifest.write_json(self.root / 'plan.json', changed)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            aggregate.aggregate(self.root)

    def test_checkpoint_content_and_asset_change_invalidate_launch(self):
        receipt = self.root / 'receipt.json'
        manifest.verify_checkpoint_once(self.root, receipt)
        manifest.validate_receipt(receipt, self.plan)
        self.checkpoint.write_bytes(b'synthetic checkpoint B')
        with self.assertRaisesRegex(ValueError, 'changed since'):
            manifest.validate_receipt(receipt, self.plan)
        with self.assertRaisesRegex(ValueError, 'content differs'):
            manifest.verify_checkpoint_once(self.root, receipt)
        self.checkpoint.write_bytes(b'synthetic checkpoint A')
        self.stats.write_text('{"changed": true}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'stats content differs'):
            manifest.verify_checkpoint_once(self.root, receipt)

    def test_manually_rehashed_nonpaper_plan_rejected(self):
        for field, value in [('episodes_per_task', 10), ('replan_steps', 16),
                             ('checkpoint_step', 75000), ('action_horizon', 16), ('flow_steps', 10),
                             ('tasks', list(reversed(self.plan['tasks'])))]:
            with self.subTest(field=field):
                changed = dict(self.plan, **{field: value})
                changed.pop('plan_sha256')
                changed['plan_sha256'] = manifest.digest(changed)
                with self.assertRaisesRegex(ValueError, 'protocol constants'):
                    manifest.validate_plan(changed)

    def test_cli_rejects_nonpaper_episodes_replanning_or_step(self):
        for flag, value in [('--episodes', '1'), ('--replan-steps', '16'), ('--checkpoint-step', '75000')]:
            result = subprocess.run([
                sys.executable, str(ROOT / 'scripts/robotwin_eval/make_plan.py'),
                '--root', str(self.root), '--checkpoint', str(self.checkpoint),
                '--simulator', str(self.simulator), flag, value],
                capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn('invalid choice', result.stderr)

    def test_relative_root_resolves_before_reading_or_writing(self):
        self.write_rows(self.full_rows())
        with mock.patch('sys.argv', ['aggregate', '--root', 'evaluation']):
            # An actual subprocess exercises the public CLI from another cwd.
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/robotwin_eval/aggregate.py'),
                                     '--root', 'evaluation'], cwd=self.directory,
                                    capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = manifest.read_json(self.root / 'summary.json')
        self.assertEqual(summary['checkpoint'], str(self.checkpoint))

    def test_gap_or_reused_scene_seed_cannot_resume(self):
        first, second = [self.row(*key) for key in manifest.expected_cases()[:2]]
        self.write_rows([second])
        with self.assertRaisesRegex(ValueError, 'non-contiguous'):
            manifest.read_records(self.root, self.plan)
        second['scene_seed'] = first['scene_seed']
        self.write_rows([first, second])
        with self.assertRaisesRegex(ValueError, 'reused or reordered'):
            manifest.read_records(self.root, self.plan)

    def test_stale_readiness_pid_is_rejected(self):
        ready = self.root / 'ready.json'
        manifest.write_json(ready, dict(**manifest.identity(self.plan), pid=123, loaded_strictly=True))
        command = [sys.executable, str(ROOT / 'scripts/robotwin_eval/manifest.py'), 'ready',
                   '--root', str(self.root), '--file', str(ready), '--pid']
        self.assertEqual(subprocess.run(command + ['124'], capture_output=True).returncode, 1)
        self.assertEqual(subprocess.run(command + ['123'], capture_output=True).returncode, 0)

    def test_synthetic_saved_checkpoint_requires_current_schema_and_recipe(self):
        try:
            import torch
            import yaml
            from dataclasses import asdict
            from thinkwam_p1.config import P1Config
        except ImportError as exc:
            self.skipTest(f'CPU torch/yaml unavailable: {exc}')
        recipe = yaml.safe_load((ROOT / 'configs/paper/robotwin2.yaml').read_text(encoding='utf-8'))
        for name in ('jepa_horizons', 'jepa_token_counts'):
            recipe[name] = tuple(recipe[name])
        values = asdict(P1Config(**recipe))
        self.assertEqual(values['video_residual_loss_weight'], 0.2)
        values.update(robotwin_stats_path=str(self.stats), robotwin_frame_cache=str(self.directory),
                      robotwin_manifest='manifest_3view.json')
        (self.directory / 'manifest_3view.json').write_bytes(self.frames.read_bytes())
        checkpoint = self.directory / 'torch_checkpoint.pt'
        state = dict(step=90000, config=values, model={'synthetic_weight': torch.ones(1)})
        torch.save(state, checkpoint)
        actual, loaded_values = manifest.inspect_checkpoint(checkpoint)
        self.assertEqual(actual['checkpoint_sha256'], manifest.sha256_file(checkpoint))
        self.assertEqual(actual['checkpoint_config_sha256'], manifest.digest(values))
        self.assertEqual(loaded_values, values)
        for field, value in [('flow_steps', 10), ('max_steps', 150000),
                             ('video_residual_loss_weight', 0.8), ('video_residual_loss_weight', 1.0),
                             ('unknown_sampling_bias', True), ('phase_enabled', True)]:
            with self.subTest(field=field, value=value):
                changed = dict(values, **{field: value})
                with self.assertRaises(ValueError):
                    manifest.paper_checkpoint_config(changed)
        state['step'] = 75000
        torch.save(state, checkpoint)
        with self.assertRaisesRegex(ValueError, 'step-90000'):
            manifest.inspect_checkpoint(checkpoint)


if __name__ == '__main__':
    unittest.main()
