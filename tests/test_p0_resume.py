"""Small CPU-only tests for resuming one P0 stage without a new LR clock."""

import copy
import io
import unittest

import torch
from torch import nn

from thinkwam_p0.training.checkpoint import (
    restore_p0_training_state,
    validate_p0_resume_checkpoint,
)


class ToyP0(nn.Module):
    def __init__(self):
        super().__init__()
        self.vjepa = nn.Linear(2, 2)
        with torch.no_grad():
            self.vjepa.weight.copy_(torch.eye(2))
            self.vjepa.bias.zero_()
        self.vjepa.requires_grad_(False)
        self.policy = nn.Linear(2, 1)

    def forward(self, inputs):
        return self.policy(self.vjepa(inputs))


class P0ResumeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = {
            "dataset": "libero", "max_steps": 6, "learning_rate": 0.05,
            "min_learning_rate": 0.005, "warmup_ratio": 0.2,
            "batch_size": 2, "gradient_accumulation_steps": 1,
            "jepa_horizons": (7, 15, 31), "output_dir": "/first/run",
        }

    def components(self):
        model = ToyP0().cpu()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=self.config["learning_rate"], betas=(0.9, 0.95), weight_decay=0.0,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 1.0 - 0.5 * step / self.config["max_steps"]
        )
        return model, optimizer, scheduler

    @staticmethod
    def update(model, optimizer, scheduler):
        optimizer.zero_grad(set_to_none=True)
        inputs = torch.tensor([[0.1, 0.2], [0.7, -0.4]])
        targets = torch.tensor([[0.3], [-0.1]])
        loss = (model(inputs) - targets).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

    def checkpoint(self, model, optimizer, scheduler, step, best_metric=0.25):
        # Exercise the existing on-disk contract without writing files or
        # retaining aliases to the live optimizer's mutable moment tensors.
        payload = {
            "model": {name: value for name, value in model.state_dict().items()
                      if not name.startswith("vjepa.")},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": copy.deepcopy(self.config),
            "step": step, "best_metric": best_metric,
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu", weights_only=False)

    def interrupted(self):
        model, optimizer, scheduler = self.components()
        for _ in range(2):
            self.update(model, optimizer, scheduler)
        return model, optimizer, scheduler, self.checkpoint(model, optimizer, scheduler, 2)

    def test_next_updates_match_uninterrupted_training_and_stop_at_original_budget(self):
        model, optimizer, scheduler, state = self.interrupted()
        resumed, resumed_optimizer, resumed_scheduler = self.components()
        step, best = restore_p0_training_state(
            resumed, resumed_optimizer, resumed_scheduler, state, self.config
        )
        self.assertEqual((step, best), (2, 0.25))
        self.assertEqual(resumed_optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["lr"])
        self.assertTrue(resumed_optimizer.state)
        for _ in range(step, self.config["max_steps"]):
            self.update(model, optimizer, scheduler)
            self.update(resumed, resumed_optimizer, resumed_scheduler)
        self.assertEqual(scheduler.last_epoch, self.config["max_steps"])
        self.assertEqual(resumed_scheduler.last_epoch, self.config["max_steps"])
        self.assertEqual(resumed_scheduler.get_last_lr(), scheduler.get_last_lr())
        for name, expected in model.state_dict().items():
            torch.testing.assert_close(resumed.state_dict()[name], expected, rtol=0, atol=0)

    def test_completed_stage_is_not_reset_to_zero(self):
        model, optimizer, scheduler = self.components()
        for _ in range(self.config["max_steps"]):
            self.update(model, optimizer, scheduler)
        state = self.checkpoint(model, optimizer, scheduler, self.config["max_steps"])
        restored = self.components()
        step, _ = restore_p0_training_state(*restored, state, self.config)
        self.assertEqual(step, self.config["max_steps"])
        self.assertEqual(list(range(step, self.config["max_steps"])), [])

    def test_rejects_changed_recipe_or_missing_training_state(self):
        *_, state = self.interrupted()
        for field, value in (("max_steps", 8), ("learning_rate", 0.1), ("dataset", "robocasa24")):
            with self.subTest(field=field):
                changed = dict(self.config, **{field: value})
                with self.assertRaisesRegex(ValueError, field):
                    validate_p0_resume_checkpoint(state, changed)
        missing_optimizer = dict(state)
        del missing_optimizer["optimizer"]
        with self.assertRaisesRegex(ValueError, "optimizer"):
            validate_p0_resume_checkpoint(missing_optimizer, self.config)

    def test_runtime_relocation_and_list_tuple_horizons_are_compatible(self):
        *_, state = self.interrupted()
        state["config"]["jepa_horizons"] = [7, 15, 31]
        config = dict(self.config, output_dir="/new/run", num_workers=0)
        validate_p0_resume_checkpoint(state, config)

    def test_only_frozen_encoder_state_may_be_missing(self):
        *_, state = self.interrupted()
        del state["model"]["policy.weight"]
        with self.assertRaisesRegex(RuntimeError, "policy.weight"):
            restore_p0_training_state(*self.components(), state, self.config)

    def test_scheduler_position_and_learning_rate_are_checked(self):
        *_, state = self.interrupted()
        wrong_position = copy.deepcopy(state)
        wrong_position["scheduler"]["last_epoch"] = 0
        with self.assertRaisesRegex(ValueError, "position"):
            validate_p0_resume_checkpoint(wrong_position, self.config)
        wrong_lr = copy.deepcopy(state)
        wrong_lr["optimizer"]["param_groups"][0]["lr"] = 0.0
        with self.assertRaisesRegex(ValueError, "inconsistent learning rates"):
            restore_p0_training_state(*self.components(), wrong_lr, self.config)

    def test_null_best_metric_from_old_checkpoints_means_no_best_yet(self):
        *_, state = self.interrupted()
        state["best_metric"] = None
        _, best = restore_p0_training_state(*self.components(), state, self.config)
        self.assertEqual(best, float("inf"))


if __name__ == "__main__":
    unittest.main()
