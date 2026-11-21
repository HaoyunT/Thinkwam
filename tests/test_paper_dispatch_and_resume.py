"""CPU interface tests: no torchrun, model download, dataset or GPU is started."""
import copy
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import torch

from scripts.train_p1 import (
    cosine_warmup_lambda,
    load_config,
    restore_p1_training_state,
    validate_p1_resume,
)

ROOT = Path(__file__).resolve().parents[1]
BASH = os.environ.get('TEST_BASH') or shutil.which('bash')


@unittest.skipUnless(BASH, 'Bash is required for the dispatcher interface tests')
class PaperDispatcherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='paper-dispatch-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.capture = self.root / 'calls.txt'
        self.python = self.root / 'capture_python.sh'
        self.python.write_text('#!/usr/bin/env bash\nprintf "CALL\\n" >> "$CAPTURE"\n'
                               'printf "%s\\n" "$@" >> "$CAPTURE"\n')
        self.python.chmod(0o755)

    def run_dispatch(self, *args, processes='4'):
        self.capture.unlink(missing_ok=True)
        env = dict(os.environ, PYTHON=self.python.as_posix(), CAPTURE=self.capture.as_posix(),
                   NPROC_PER_NODE=processes)
        result = subprocess.run([BASH, (ROOT / 'scripts/train_paper.sh').as_posix(), *args],
                                env=env, capture_output=True, text=True, encoding='utf-8', timeout=15)
        calls = self.capture.read_text(encoding='utf-8').split('CALL\n')[1:] if self.capture.exists() else []
        return result, [call.splitlines() for call in calls]

    def test_exactly_five_recipes_select_the_expected_engine_and_config(self):
        recipes = {
            'libero-p0': ('train_p0.py', 'libero_p0.yaml'),
            'libero-p1': ('train_p1.py', 'libero_p1.yaml'),
            'robotwin': ('train_p1.py', 'robotwin2.yaml'),
            'robocasa24-p0': ('train_p0.py', 'robocasa24_p0.yaml'),
            'robocasa24-p1': ('train_p1.py', 'robocasa24_p1.yaml'),
        }
        for recipe, (engine, config) in recipes.items():
            with self.subTest(recipe=recipe):
                result, calls = self.run_dispatch(recipe)
                self.assertEqual(result.returncode, 0, result.stderr)
                call = calls[-1]
                self.assertEqual(call[:4], ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4'])
                self.assertEqual(Path(call[4]).name, engine)
                self.assertEqual(call.count('--config'), 1)
                self.assertEqual(Path(call[call.index('--config') + 1]).name, config)
                self.assertNotIn('--no-closed-loop', call)
                self.assertEqual(call[call.index('--backend') + 1], 'qwen')
                self.assertEqual(len(calls), 2 if recipe.startswith('robocasa24-') else 1)

    def test_runtime_options_preserve_spaces_and_the_selected_recipe(self):
        for recipe in ('libero-p0', 'libero-p1', 'robotwin', 'robocasa24-p0', 'robocasa24-p1'):
            with self.subTest(recipe=recipe):
                result, calls = self.run_dispatch(recipe, '--resume', '/tmp/stage checkpoint.pt',
                                                 '--output-dir=/tmp/output directory', '--save-every', '1000')
                self.assertEqual(result.returncode, 0, result.stderr)
                call = calls[-1]
                self.assertEqual(call[call.index('--resume') + 1], '/tmp/stage checkpoint.pt')
                self.assertEqual(call[call.index('--output-dir') + 1], '/tmp/output directory')

    def test_protocol_overrides_and_nonpaper_selectors_fail_before_launch(self):
        bad = [
            ('focused',), ('robotwin-unified',),
            ('robotwin', '--config', '/tmp/other.yaml'), ('robotwin', '--config=/tmp/other.yaml'),
            ('robotwin', '--conf', '/tmp/other.yaml'), ('robotwin', '--max-steps', '150000'),
            ('robotwin', '--backend', 'tiny'), ('robotwin', '--smoke'),
            ('robotwin', '--allow-missing-affordance'), ('robotwin', '--resume'),
            ('robotwin', '--save-every', '0'),
            ('robotwin', '--resume', 'one.pt', '--resume', 'two.pt'),
        ]
        for arguments in bad:
            with self.subTest(arguments=arguments):
                result, calls = self.run_dispatch(*arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(calls, [])
        result, calls = self.run_dispatch('robotwin', processes='8')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, [])

    def test_warm_start_is_only_for_fresh_two_stage_p1_recipes(self):
        for recipe in ('libero-p1', 'robocasa24-p1'):
            result, calls = self.run_dispatch(recipe, '--p0-checkpoint', '/tmp/p0.pt')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--p0-checkpoint', calls[-1])
            result, calls = self.run_dispatch(recipe, '--resume', 'p1.pt', '--p0-checkpoint', 'p0.pt')
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(calls, [])
        for recipe in ('robotwin', 'libero-p0', 'robocasa24-p0'):
            result, calls = self.run_dispatch(recipe, '--p0-checkpoint', '/tmp/p0.pt')
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(calls, [])


class P1ResumeTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(dataset='libero', max_steps=20, batch_size=24,
                                      learning_rate=0.01, min_learning_rate=0.001,
                                      warmup_ratio=0.1, p1_stage='staged', first_action_loss_weight=1.2,
                                      video_residual_loss_weight=0.2,
                                      jepa_horizons=(7, 15, 31), output_dir='/old', p0_checkpoint='/old/p0.pt')

    def components(self):
        model = torch.nn.Linear(1, 1, bias=False, dtype=torch.double)
        optimizer = torch.optim.AdamW([{'name': 'new_modules', 'params': list(model.parameters())}],
                                      lr=self.config.learning_rate, betas=(0.9, 0.95), weight_decay=0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: cosine_warmup_lambda(s, self.config))
        return model, optimizer, scheduler

    @staticmethod
    def step(model, optimizer, scheduler):
        optimizer.zero_grad(set_to_none=True)
        (model(torch.ones((1, 1), dtype=torch.double)) - 2).square().sum().backward()
        optimizer.step()
        scheduler.step()

    def checkpoint(self):
        model, optimizer, scheduler = self.components()
        for _ in range(4):
            self.step(model, optimizer, scheduler)
        state = copy.deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                   scheduler=scheduler.state_dict(), step=4, config=vars(self.config)))
        return (model, optimizer, scheduler), state

    def test_resume_matches_the_next_uninterrupted_optimizer_update(self):
        original, state = self.checkpoint()
        resumed = self.components()
        validate_p1_resume(state, self.config)
        resumed[0].load_state_dict(state['model'], strict=True)
        self.assertEqual(restore_p1_training_state(resumed[1], resumed[2], state), 4)
        self.assertEqual(original[1].param_groups[0]['lr'], resumed[1].param_groups[0]['lr'])
        for components in (original, resumed):
            self.step(*components)
        self.assertTrue(torch.equal(original[0].weight, resumed[0].weight))
        self.assertEqual(original[2].state_dict(), resumed[2].state_dict())

    def test_resume_rejects_other_stages_or_changed_training_protocol(self):
        _, state = self.checkpoint()
        for field, value in (('dataset', 'robotwin2.0'), ('max_steps', 30), ('learning_rate', 0.02),
                             ('video_residual_loss_weight', 0.8), ('video_residual_loss_weight', 1.0)):
            altered = copy.deepcopy(state)
            altered['config'][field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, 'same stage'):
                validate_p1_resume(altered, self.config)
        del state['config']['first_action_loss_weight']
        with self.assertRaisesRegex(ValueError, 'same stage'):
            validate_p1_resume(state, self.config)

    def test_resume_accepts_output_relocation_and_does_not_require_old_p0_file(self):
        _, state = self.checkpoint()
        state['config'].update(output_dir='/moved', p0_checkpoint='/other/p0.pt')
        state['config']['jepa_horizons'] = [7, 15, 31]
        validate_p1_resume(state, self.config)

    def test_incomplete_or_incompatible_optimizer_resume_fails_instead_of_resetting(self):
        _, state = self.checkpoint()
        missing = copy.deepcopy(state)
        del missing['optimizer']
        with self.assertRaisesRegex(ValueError, 'requires'):
            validate_p1_resume(missing, self.config)
        altered = copy.deepcopy(state)
        altered['optimizer']['param_groups'][0]['name'] = 'different_group'
        _, optimizer, scheduler = self.components()
        with self.assertRaisesRegex(ValueError, 'groups differ'):
            restore_p1_training_state(optimizer, scheduler, altered)
        state['scheduler']['last_epoch'] = 3
        with self.assertRaisesRegex(ValueError, 'position'):
            restore_p1_training_state(optimizer, scheduler, state)

    def test_archived_event_sampling_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text('dataset: robocasa24\nrobocasa_event_sampling: true\n')
            with self.assertRaisesRegex(TypeError, 'robocasa_event_sampling'):
                load_config(str(path))


if __name__ == '__main__':
    unittest.main()
