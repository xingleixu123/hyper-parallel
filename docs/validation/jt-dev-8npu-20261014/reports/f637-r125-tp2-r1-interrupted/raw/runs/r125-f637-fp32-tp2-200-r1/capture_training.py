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
"""Experiment-only observation around the unmodified text Trainer."""

import hashlib
import json
import math
import os
from pathlib import Path

import torch

from hyper_parallel.models.jt_deepseek_v3.adapter import jt_builder
from hyper_parallel.tools.model_integration.tensor_probes import InputIdentityRecorder
from hyper_parallel.trainer.callbacks.base import Callback
from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.text_trainer import TextTrainer
from module_profile import ModuleProfileCallback, install_target_ablation
from expert_probe import install_expert_probes
from precision_experiments import install_fp32_newton_schulz, install_fp32_ns_pipeline
from fp32_wgrad_experiment import install_fp32_expert_wgrad


OUTPUT = Path(os.environ["JT_CASE_OUTPUT"])
RANK = int(os.environ.get("RANK", "0"))
NATIVE_INITIALIZE = jt_builder._initialize_model_weights


def write_json(name: str, value: object) -> None:
    """Persist a compact observation in this run's isolated directory."""
    (OUTPUT / name).write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def initialize_and_hash(model: torch.nn.Module) -> None:
    """Prove complete pre-sharding initial-state identity without retaining copies."""
    NATIVE_INITIALIZE(model)
    records = {}
    for name, tensor in model.state_dict().items():
        array = tensor.detach().contiguous().numpy()
        records[name] = {
            "shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest(),
        }
    write_json(f"initial-state-rank{RANK}.json", records)


class EvidenceCallback(Callback):
    """Save existing scalar metrics plus canonical input identity for each step."""

    def __init__(self, trainer: object, recorder: InputIdentityRecorder) -> None:
        super().__init__(trainer)
        self.recorder = recorder

    def on_train_begin(self, state: object, **kwargs: object) -> None:
        write_json(f"start-rank{RANK}.json", {
            "global_step": state.global_step, "epoch": state.epoch,
            "train_iters": self.trainer.train_iters,
            "micro_batches": self.trainer.num_micro_batches,
            "model_parameters": [
                {"name": name, "dtype": str(param.dtype), "shape": list(param.shape),
                 "main_dtype": str(getattr(getattr(param, "main_param", None), "dtype", param.dtype))}
                for name, param in self.trainer.model.named_parameters()
            ],
        })

    def on_step_begin(self, state: object, **kwargs: object) -> None:
        self.recorder.begin_step()

    def on_step_end(self, state: object, **kwargs: object) -> None:
        identity = self.recorder.finish_step()
        metrics = dict(self.trainer.step_env_metrics)
        if not all(math.isfinite(float(value)) for value in metrics.values()):
            raise ValueError(f"Non-finite training metrics at step {state.global_step}")
        row = {"step": state.global_step, "rank": RANK, **metrics, "input_identity": identity}
        with (OUTPUT / f"metrics-rank{RANK}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def on_train_end(self, state: object, **kwargs: object) -> None:
        write_json(f"complete-rank{RANK}.json", {"global_step": state.global_step, "epoch": state.epoch})


def main() -> None:
    """Run the normal Trainer with non-mutating experiment observations."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    jt_builder._initialize_model_weights = initialize_and_hash
    if os.environ.get("JT_EXPERT_PROBES") == "1":
        install_expert_probes(OUTPUT, RANK)
    if os.environ.get("JT_NS_FP32") == "1":
        install_fp32_newton_schulz()
        write_json(f"precision-experiment-rank{RANK}.json", {
            "kind": "diagnostic_only_optimizer_precision_ablation",
            "ns_compute": "float32", "ns_input_and_output": "original_bfloat16",
            "original_recipe": False,
            "npu_matmul_allow_hf32": getattr(getattr(torch.npu, "matmul", None), "allow_hf32", None),
        })
    elif os.environ.get("JT_NS_FP32") == "2":
        changed_methods = install_fp32_ns_pipeline()
        write_json(f"precision-experiment-rank{RANK}.json", {
            "kind": "diagnostic_only_optimizer_precision_ablation",
            "ns_compute": "float32", "ns_input_and_output": "float32",
            "ns_update_relay": "float32", "original_recipe": False,
            "modified_methods": changed_methods,
            "npu_matmul_allow_hf32": getattr(getattr(torch.npu, "matmul", None), "allow_hf32", None),
        })
    if os.environ.get("JT_WGRAD_FP32") == "1":
        install_fp32_expert_wgrad()
        write_json(f"wgrad-experiment-rank{RANK}.json", {
            "kind": "diagnostic_only_expert_weight_gradient_precision",
            "forward_and_dgrad": "original_bfloat16", "wgrad_and_reduce": "float32",
            "fsdp_delivery": "existing_unsharded_accumulation_buffer",
        })
    if os.environ.get("JT_RECOMPUTE_TARGETS"):
        install_target_ablation(os.environ["JT_RECOMPUTE_TARGETS"].split(","), OUTPUT, RANK)
    config = parse_training_args()
    trainer = TextTrainer(config)
    base = trainer.base
    if RANK == 0:
        write_json("config.resolved.json", config.to_dict())
    recorder = InputIdentityRecorder(base.mesh, cp_replicated_forward_fields=("attention_metadata",))
    original_get_batch = base.get_batch

    def observe_batch(iterator: object) -> object:
        model_inputs, loss_inputs = original_get_batch(iterator)
        recorder.record(model_inputs, loss_inputs)
        return model_inputs, loss_inputs

    if callable(getattr(original_get_batch, "prepare_step", None)):

        def observe_prepared_step(iterator: object, micro_steps: int) -> object:
            supervision, batches = original_get_batch.prepare_step(iterator, micro_steps)

            def observe_inputs(inputs: object) -> object:
                recorder.record(*inputs)
                return inputs

            return supervision, map(observe_inputs, batches)

        observe_batch.prepare_step = observe_prepared_step

    base.get_batch = observe_batch
    # Register last so restore has completed before the start-state observation.
    base._callbacks.append(EvidenceCallback(base, recorder))
    if os.environ.get("JT_MODULE_PROFILE") == "1":
        base._callbacks.append(ModuleProfileCallback(base, OUTPUT, RANK))
    trainer.train()


if __name__ == "__main__":
    main()
