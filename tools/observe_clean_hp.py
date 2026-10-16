# Copyright 2026 Huawei Technologies Co., Ltd.
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

"""Observe the native HP continuation path; never inject or repair its state."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch_npu
import hyper_parallel

from hyper_parallel.trainer.callbacks.base import Callback
from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.text_trainer import TextTrainer


def leaves(optimizer: Any) -> list[Any]:
    """Find leaf optimizers without mutating their states."""
    if hasattr(optimizer, "optimizers_dict"):
        return [leaf for item in optimizer.optimizers_dict.values() for leaf in leaves(item)]
    if hasattr(optimizer, "chained_optimizers"):
        return [leaf for item in optimizer.chained_optimizers for leaf in leaves(item)]
    if hasattr(optimizer, "optimizer"):
        return leaves(optimizer.optimizer)
    return [optimizer]


def numpy_value(value: Any) -> np.ndarray:
    """Read a local tensor as NumPy without changing its dtype."""
    if hasattr(value, "to_local"):
        value = value.to_local()
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def exact_record(name: str, actual: Any, expected: np.ndarray) -> dict:
    """Fail on any shape, dtype or value mismatch at the resume boundary."""
    array = numpy_value(actual)
    if array.shape != expected.shape or array.dtype != expected.dtype or not np.array_equal(array, expected):
        raise ValueError(f"Restored tensor differs: {name}; actual={array.shape}/{array.dtype}, "
                         f"expected={expected.shape}/{expected.dtype}")
    return {"name": name, "shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(array)).hexdigest()}


def cursors(tree: Any) -> list[int]:
    """Find every restored sampler cursor in the nested loader state."""
    found = []
    if isinstance(tree, dict):
        if "consumed_samples" in tree:
            found.append(int(tree["consumed_samples"]))
        for value in tree.values():
            found.extend(cursors(value))
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            found.extend(cursors(value))
    return found


class Observer(Callback):
    """Audit native restoration and record only effective inputs and scalars."""

    def __init__(self, trainer: Any, output: Path, offline: Path, offset: int) -> None:
        super().__init__(trainer)
        self.output, self.offline, self.offset = output, offline, offset
        (output / "batches").mkdir()

    def on_train_begin(self, state: Any, **kwargs: Any) -> None:
        """Run after the native checkpointer and independently check its output."""
        del kwargs
        if state.global_step != self.offset:
            raise ValueError(f"Native resume step {state.global_step} != {self.offset}")
        report = {"step": state.global_step, "model": [], "optimizer": []}
        model_state = self.trainer.model.state_dict()
        with np.load(self.offline / "model.npz", allow_pickle=False) as archive:
            for name in archive.files:
                report["model"].append(exact_record(name, model_state[name], archive[name]))
            missing = set(dict(self.trainer.model.named_parameters())) - set(archive.files)
            if missing:
                raise ValueError(f"Converted model omitted parameters: {sorted(missing)}")
        names = {id(parameter): name for name, parameter in self.trainer.model.named_parameters()}
        used = set()
        with np.load(self.offline / "optimizer.npz", allow_pickle=False) as archive:
            for leaf in leaves(self.trainer.optimizer):
                is_muon = "muon" in type(leaf).__name__.lower()
                fields = [("muon_m", "momentum_buffer")] if is_muon else [
                    ("adam_m", "exp_avg"), ("adam_v", "exp_avg_sq")]
                for group in leaf.param_groups:
                    for parameter in group["params"]:
                        name = names[id(parameter)]
                        for prefix, field in fields:
                            key = prefix + "." + name
                            record = exact_record(key, leaf.state[parameter][field], archive[key])
                            report["optimizer"].append(record)
                            used.add(key)
                        restored_step = leaf.state[parameter].get("step", group.get("step"))
                        if restored_step is None or int(numpy_value(restored_step).item()) != self.offset:
                            raise ValueError(f"Optimizer step differs: {name}")
            unused_placeholders = []
            for key in archive.files:
                if key in used:
                    continue
                value = archive[key]
                # PR 166 concatenates the two unused Q/KV placeholders along with their weights.
                expected_shape = (2,) if key.endswith(".self_attn.linear_qkv.weight") else (1,)
                if value.shape != expected_shape or np.any(value != 0):
                    raise ValueError(f"Unused non-placeholder optimizer state: {key}, shape={value.shape}")
                unused_placeholders.append({"name": key, "shape": list(value.shape)})
            report["unused_converted_placeholders"] = unused_placeholders
        loader_state = self.trainer.train_dataloader.state_dict()
        report["consumed_samples"] = cursors(loader_state)
        if not report["consumed_samples"] or set(report["consumed_samples"]) != {self.offset}:
            raise ValueError(f"Native data cursor differs: {report['consumed_samples']}")
        report["lr_scheduler"] = str(self.trainer.lr_scheduler.state_dict())
        report["all_exact"] = True
        (self.output / "RESTORE.json").write_text(json.dumps(report, indent=2) + "\n")
        print("[P12-RESTORE] " + json.dumps({"step": self.offset, "model": len(report["model"]),
                                            "optimizer": len(report["optimizer"]), "all_exact": True}), flush=True)

    def on_micro_step_begin(self, state: Any, micro_batch: dict, **kwargs: Any) -> None:
        """Record what the native training loop actually presents to the model."""
        del kwargs
        arrays = {key: numpy_value(value) for key, value in micro_batch.items()
                  if key in ("input_ids", "shift_labels", "position_ids", "actual_seq_len", "loss_mask")
                  and value is not None}
        np.savez(self.output / "batches" / f"{state.global_step:06d}.npz", **arrays)

    def on_step_end(self, state: Any, *, loss: float, loss_dict: dict, grad_norm: float, **kwargs: Any) -> None:
        """Persist unrounded native losses without changing the training path."""
        del kwargs
        record = {"step": state.global_step, "loss": float(loss),
                  "components": {name: float(value) for name, value in loss_dict.items()},
                  "grad_norm": float(grad_norm)}
        with (self.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print("[P12-HP] " + json.dumps(record), flush=True)


def main() -> None:
    """Use the unchanged YAML, trainer and native checkpoint restore callback."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=200)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.source_root.resolve()
    if not Path(hyper_parallel.__file__).resolve().is_relative_to(source):
        raise ValueError("Editable HyperParallel import must resolve to the clean checkout")
    cann = Path("/usr/local/Ascend/cann/opp/version.info").read_text()
    if "Version=9.2." not in cann:
        raise ValueError("The clean HP continuation requires CANN 9.2.X")
    if any(key.startswith("P12_") for key in os.environ):
        raise ValueError("Experimental P12 controls must not enter a clean product run")
    trainer = TextTrainer(parse_training_args([args.config]))
    runtime = {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
               "hyper_parallel_path": hyper_parallel.__file__, "cann": cann,
               "deterministic": torch.are_deterministic_algorithms_enabled(),
               "environment": {key: os.environ.get(key) for key in (
                   "CUSTOM_MATMUL_SHUFFLE", "CLOSE_MATMUL_K_SHIFT", "HCCL_DETERMINISTIC", "LCCL_DETERMINISTIC")}}
    (args.output / "RUNTIME.json").write_text(json.dumps(runtime, indent=2) + "\n")
    trainer.base._callbacks.append(Observer(trainer.base, args.output, args.offline, args.offset))
    trainer.train()
    modules = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
               if getattr(module, "__file__", None) and name.startswith("hyper_parallel")}
    if any(not Path(path).is_relative_to(source) for path in modules.values()):
        raise ValueError("A HyperParallel module escaped the clean checkout")
    if any("/phase12/tools" in str(getattr(module, "__file__", "")) for module in sys.modules.values()):
        raise ValueError("A historical experimental module was imported")
    (args.output / "IMPORTED_MODULES.json").write_text(json.dumps(modules, indent=2) + "\n")
    steps = [json.loads(line)["step"] for line in (args.output / "metrics.jsonl").read_text().splitlines()]
    expected = list(range(args.offset + 1, trainer.base.train_iters + 1))
    if steps != expected:
        raise ValueError(f"Incomplete continuation: {steps}")
    (args.output / "SUCCESS.json").write_text(json.dumps({"steps": len(steps), "first": steps[0], "last": steps[-1]}))


if __name__ == "__main__":
    main()
