# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Loss diagnostics and throughput must not change the optimization objective."""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from transformers.modeling_outputs import CausalLMOutput

from hyper_parallel.components.losses.model_output import ModelOutputLoss
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.callbacks.environ_meter_callback import EnvironMeterCallback
from hyper_parallel.trainer.callbacks.logging_callback import LoggingCallback
from hyper_parallel.trainer.runtime.flops import estimate_deepseek_flops
from hyper_parallel.trainer.state import TrainerState
from tests.ut.trainer.test_flops import deepseek_config


METER = "hyper_parallel.trainer.callbacks.environ_meter_callback"


class _ScalarModel(torch.nn.Module):
    """Return an objective and deliberately different logging-only scalar."""

    def __init__(self) -> None:
        """Use one parameter so an accidental diagnostic gradient is visible."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> SimpleNamespace:
        """Return a differentiable diagnostic to exercise the detach boundary."""
        del use_cache
        return SimpleNamespace(loss=self.weight.square() * input_ids.mean(),
                               loss_metrics={"mtp_1_loss": self.weight * 100})


def metric_trainer(**overrides: object) -> SimpleNamespace:
    """Build the real callbacks' small owner contract."""
    values = {
        "mesh": SimpleNamespace(dp_cp_mesh=None, dp_size=1, cp_size=1, tp_size=1, pp_size=1),
        "lr_scheduler": None, "optimizer": SimpleNamespace(param_groups=[{"lr": 0.001}]),
        "global_rank": 0, "config": SimpleNamespace(training=SimpleNamespace(logging_steps=2)),
    }
    return SimpleNamespace(**(values | overrides))


class TestLossLogging(unittest.TestCase):
    """Verify detached collection, timing, units and complete log lines."""

    def setUp(self) -> None:
        """Keep logic tests independent of accelerator and process-group state."""
        self.enterContext(patch(f"{METER}.get_device_type", return_value="cpu"))
        self.enterContext(patch(f"{METER}.get_world_size_safe", return_value=1))

    def test_diagnostics_mean_reset_and_logging_cadence(self) -> None:
        """Two micro-step means are logged once and never overwrite total loss."""
        trainer = metric_trainer()
        meter = EnvironMeterCallback(trainer)
        logger = LoggingCallback(trainer)
        state = TrainerState(global_step=1)
        with patch(f"{METER}.time.perf_counter", side_effect=(1.0, 3.0)):
            meter.on_step_begin(state)
            for value in (2.0, 4.0):
                diagnostic = torch.tensor(value, requires_grad=True)
                meter.record_loss_metrics(SimpleNamespace(loss_metrics={"mtp_1_loss": diagnostic},
                                                          indexer_loss=diagnostic * 2))
            self.assertTrue(all(not value.requires_grad for value in meter._loss_metrics.values()))
            meter.on_step_end(state, loss=5.0, loss_dict={"foundation_loss": 5.0}, grad_norm=0.2)
        self.assertEqual(trainer.step_train_metrics["training/mtp_1_loss"], 3.0)
        self.assertEqual(trainer.step_train_metrics["training/indexer_loss"], 6.0)
        self.assertEqual(trainer.step_train_metrics["training/total_loss"], 5.0)
        self.assertFalse(meter._loss_metrics)
        with patch.object(logger, "_write") as write:
            logger.on_step_end(state, loss=5.0, loss_dict={}, grad_norm=0.2)
            self.assertFalse(write.called)
            state.global_step = 2
            for _ in range(2):
                logger.on_step_end(state, loss=5.0, loss_dict={}, grad_norm=0.2)
            write.assert_called_once()
            self.assertIn("training/mtp_1_loss=3", write.call_args.args[0])
        meter.on_step_begin(state)
        meter.on_step_end(state, loss=1.0, loss_dict=None, grad_norm=0.0)
        self.assertNotIn("training/mtp_1_loss", trainer.step_train_metrics)

    def test_throughput_counts_input_work_and_waits_for_device(self) -> None:
        """Masked labels change valid token/s, not padded model FLOPs."""
        trainer = metric_trainer(model_config=deepseek_config())
        meter = EnvironMeterCallback(trainer)
        device = Mock()
        with patch(f"{METER}.get_device_type", return_value="npu"), \
                patch(f"{METER}.get_torch_device", return_value=device), \
                patch.object(meter, "_memory_metrics", return_value={}), \
                patch(f"{METER}.time.perf_counter", side_effect=(10.0, 12.0)):
            state = TrainerState(global_step=1)
            meter.on_step_begin(state, micro_batches=[{"input_ids": torch.zeros(2, 8), "token_count": 3}])
            meter.on_step_end(state, loss=1.0, loss_dict=None, grad_norm=0.0)
        metrics = trainer.step_env_metrics
        self.assertEqual(metrics["performance/tokens_per_second"], 1.5)
        self.assertEqual(metrics["performance/input_tokens_per_second"], 8.0)
        self.assertEqual(metrics["performance/samples_per_second"], 1.0)
        expected = estimate_deepseek_flops(trainer.model_config, 2, 8) / 2e12
        self.assertEqual(metrics["performance/throughput_tflops_per_device"], expected)
        self.assertEqual(device.synchronize.call_count, 2)

    def test_invalid_metrics_do_not_silently_replace_objectives(self) -> None:
        """Reject vectors, unstable per-step keys and reserved-name collisions."""
        meter = EnvironMeterCallback(metric_trainer())
        state = TrainerState(global_step=1)
        meter.on_step_begin(state)
        with self.assertRaisesRegex(ValueError, "scalar"):
            meter.record_loss_metrics({"loss_metrics": {"mtp_1_loss": torch.ones(2)}})
        meter.record_loss_metrics({"loss_metrics": {"mtp_1_loss": torch.tensor(1.0)}})
        with self.assertRaisesRegex(ValueError, "stable"):
            meter.record_loss_metrics({"loss_metrics": {}})
        meter.on_step_begin(state)
        meter.record_loss_metrics({"loss_metrics": {"total_loss": torch.tensor(99.0)}})
        with self.assertRaisesRegex(ValueError, "collides"):
            meter.on_step_end(state, loss=1.0, loss_dict=None, grad_norm=0.0)

    def test_model_output_accepts_attached_diagnostics(self) -> None:
        """HF ModelOutput can carry a new attribute outside its mapping keys."""
        trainer = metric_trainer()
        meter = EnvironMeterCallback(trainer)
        state = TrainerState(global_step=1)
        output = CausalLMOutput(loss=torch.tensor(2.0))
        output.loss_metrics = {"mtp_1_loss": torch.tensor(3.0)}
        meter.on_step_begin(state)
        meter.record_loss_metrics(output)
        meter.on_step_end(state, loss=2.0, loss_dict=None, grad_norm=0.0)
        self.assertEqual(trainer.step_train_metrics["training/mtp_1_loss"], 3.0)

    def test_real_backward_ignores_diagnostics(self) -> None:
        """Run BaseTrainer forward/backward; an extra diagnostic must not add its gradient."""
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.mesh = SimpleNamespace(dp_cp_mesh=None, dp_size=1, cp_size=1, sequence_parallel=False)
        trainer.model = _ScalarModel()
        trainer.loss_fn = ModelOutputLoss()
        trainer.config = SimpleNamespace(training=SimpleNamespace(empty_cache_before_backward=False))
        trainer.model_fwd_context = nullcontext()
        trainer.model_bwd_context = nullcontext()
        trainer.model_integration = SimpleNamespace(record_batch=lambda *args: None)
        trainer.preforward = lambda batch: batch
        trainer.current_token_counts = {"foundation_tokens": torch.tensor(1.0)}
        trainer.step_token_counts = trainer.current_token_counts
        trainer.environ_meter_callback = EnvironMeterCallback(trainer)
        trainer.environ_meter_callback.on_step_begin(TrainerState(global_step=1))
        with patch("hyper_parallel.trainer.runtime.metrics.all_reduce", side_effect=lambda value, **kwargs: value):
            loss, _ = trainer.forward_backward_step({"input_ids": torch.ones(1, 2)})
        self.assertEqual(loss.item(), 4.0)
        self.assertEqual(trainer.model.weight.grad.item(), 4.0)
        self.assertEqual(trainer.environ_meter_callback._loss_metrics["mtp_1_loss"].item(), 200.0)
