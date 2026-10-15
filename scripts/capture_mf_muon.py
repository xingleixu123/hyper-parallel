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
"""Observe native MF training and export full single-card continuation state."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import time

import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore.train.callback import Callback
from mindformers.trainer.base_trainer import BaseTrainer
from mindformers.utils.parameter_register import parameter_register

from mf_input_tap import attach_input_tap
from mf_activation_taps import attach_taps
from mf_routing_tap import attach_routes


def values(value):
    """Convert observed callback scalars without rounding."""
    if isinstance(value, (tuple, list)):
        return [values(item) for item in value]
    if hasattr(value, "asnumpy"):
        value = value.asnumpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_array(directory, filename, parameter):
    """Save one native tensor with byte-level provenance."""
    array = parameter.asnumpy()
    np.save(directory / filename, array, allow_pickle=False)
    return {"file": filename, "shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(array)).hexdigest()}


def export_state(directory, network, optimizer, step):
    """Export model, Muon/AdamW state and step after all native callbacks."""
    directory.mkdir(parents=True, exist_ok=False)
    model_dir, optimizer_dir = directory / "model", directory / "optimizer"
    model_dir.mkdir()
    optimizer_dir.mkdir()
    manifest = []
    for index, (name, parameter) in enumerate(network.parameters_and_names()):
        entry = write_array(model_dir, f"{index:04d}.npy", parameter)
        manifest.append({"name": name, "parameter_name": parameter.name,
                         "requires_grad": parameter.requires_grad, **entry})
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    states = []
    for index, parameter in enumerate(optimizer.parameters):
        use_muon = bool(optimizer.use_muon[index])
        tensors = {"muon_m": optimizer.muon_m[index]} if use_muon else {
            "exp_avg": optimizer.moments1[index], "exp_avg_sq": optimizer.moments2[index]}
        states.append({"parameter_name": parameter.name, "use_muon": use_muon,
                       "state": {name: write_array(optimizer_dir, f"{index:04d}-{name}.npy", tensor)
                                 for name, tensor in tensors.items()}})
    global_step = int(optimizer.global_step.asnumpy().reshape(-1)[0])
    if global_step != step:
        raise ValueError(f"Native optimizer step {global_step} differs from observed step {step}")
    (optimizer_dir / "manifest.json").write_text(json.dumps(states, indent=2) + "\n")
    (directory / "STATE.json").write_text(json.dumps({
        "completed_steps": step, "optimizer_global_step": global_step,
        "checkpoint_timing": "after native optimizer and TopkBiasBalanceCallback",
        "model_tensors": len(manifest), "optimizer_parameters": len(states),
        "optimizer_type": type(optimizer).__name__, "format_version": 1,
    }, indent=2) + "\n")
    print(f"[CONTINUATION-CAPTURE] saved step {step} at {directory}", flush=True)


class ObserveMetrics(Callback):
    """Capture the actual graph inputs and unrounded native outputs."""

    def __init__(self, output, state):
        super().__init__()
        self.output, self.state = output, state
        self.seen = set()

    def on_train_step_end(self, run_context):
        params = run_context.original_args()
        step = int(params.cur_step_num) + self.state.get("offset", 0)
        record = {"step": step, "outputs": values(params.net_outputs), "wall_time": time.time(),
                  "components": {name: values(parameter_register.get(name))
                                 for name in ("lm_loss", "mtp_loss", "aux_loss")}}
        arrays = {}
        columns = ("input_ids", "labels", "loss_mask", "position_ids", "actual_seq_len")
        for name in columns:
            paths = set((self.output / "input_dumps").glob(name + "_*.npy")) - self.seen
            if len(paths) != 1:
                raise ValueError(f"Step {step}: expected one actual {name} input, got {len(paths)}")
            path = paths.pop()
            arrays[name] = np.load(path, allow_pickle=False)
            self.seen.add(path)
            path.unlink()
        np.savez(self.output / "batches" / f"{step - 1:06d}.npz", **arrays)
        record["input_sha256"] = {name: hashlib.sha256(array.tobytes()).hexdigest()
                                  for name, array in arrays.items()}
        if self.state.get('routing_names'):
            route_arrays = {}
            for name in self.state['routing_names']:
                for kind in ('indices', 'probabilities'):
                    paths = list((self.output / 'routing_dumps').glob(name + '_' + kind + '_*.npy'))
                    if len(paths) != 1:
                        raise ValueError(f'Expected one routing observation: {name}:{kind}, got {len(paths)}')
                    route_arrays[name + '.' + kind] = np.load(paths[0], allow_pickle=False)
                    paths[0].unlink()
            np.savez(self.output / 'routes' / f'{step - 1:06d}.npz', **route_arrays)
        self.state["losses"].append(float(np.asarray(record["outputs"][0]).reshape(-1)[0]))
        with (self.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print("[CONTINUATION-METRICS] " + json.dumps(record), flush=True)


class SaveContinuation(Callback):
    """Keep the previous 100-step anchor until the target band is reached."""

    def __init__(self, output, state, checkpoint_root, interval):
        super().__init__()
        self.output, self.state = output, state
        self.checkpoint_root, self.interval = checkpoint_root, interval

    def on_train_step_end(self, run_context):
        step = int(run_context.original_args().cur_step_num)
        if step == self.interval:
            export_state(self.checkpoint_root / f"step-{step:06d}", self.state["network"],
                         self.state["optimizer"], step)
            run_context.request_stop()


def main():
    """Use observation callbacks around the unchanged MF training entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--initial-weights", type=Path)
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--dump-activations", action="store_true")
    parser.add_argument("--deep-trace", action="store_true")
    parser.add_argument("--routing-only", action="store_true")
    parser.add_argument("--gradient-output", type=Path)
    parser.add_argument("--gradient-hash-output", type=Path)
    parser.add_argument("--gradient-hash-keep-step", type=int, default=206)
    parser.add_argument("--gradient-max-elements", type=int)
    parser.add_argument("--router-step-trace-output", type=Path)
    parser.add_argument("--save-state-step", type=int)
    parser.add_argument("--forward-trace-step", type=int)
    parser.add_argument('--attention-trace-layer', type=int)
    args = parser.parse_args()
    if os.environ.get("MS_ROLE") == "MS_SCHED":
        sys.argv = ["/home/xxl/repos/mindformers/run_mindformer.py", "--config", args.config]
        runpy.run_path(sys.argv[0], run_name="__main__")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "metrics.jsonl").exists():
        raise ValueError("Refusing to append to an existing training run")
    (args.output / "batches").mkdir()
    args.checkpoint_root.mkdir(parents=True, exist_ok=False)
    state = {"losses": [], "offset": 0}
    if args.resume_state:
        state["offset"] = json.loads((args.resume_state / "STATE.json").read_text())["completed_steps"]
        if args.initial_weights.resolve() != (args.resume_state / "model").resolve():
            raise ValueError("Resume model and optimizer must come from the same checkpoint")
    native_network = BaseTrainer.create_network
    native_optimizer = BaseTrainer.create_optimizer_scheduler
    native_callbacks = BaseTrainer.sort_callbacks

    def create_network(trainer, *positional, **keywords):
        network = native_network(trainer, *positional, **keywords)
        (args.output / "RUNTIME.json").write_text(json.dumps({
            "mindspore": ms.__version__, "deterministic": ms.get_context("deterministic"),
            "environment": {name: os.environ.get(name) for name in (
                "HCCL_DETERMINISTIC", "LCCL_DETERMINISTIC", "CUSTOM_MATMUL_SHUFFLE",
                "CLOSE_MATMUL_K_SHIFT", "ASCEND_LAUNCH_BLOCKING", "TE_PARALLEL_COMPILER")},
        }, indent=2) + "\n")
        network.init_parameters_data()
        if args.initial_weights is not None:
            manifest = json.loads((args.initial_weights / "manifest.json").read_text())
            source = {item["name"]: item for item in manifest}
            for name, parameter in network.parameters_and_names():
                item = source.pop(name)
                array = np.load(args.initial_weights / item["file"], allow_pickle=False)
                if tuple(parameter.shape) != array.shape:
                    raise ValueError(f"Initial tensor shape mismatch: {name}")
                parameter.set_data(Tensor(array))
                if not np.array_equal(parameter.asnumpy(), array):
                    raise ValueError(f"Initial tensor did not load exactly: {name}")
            if source:
                raise ValueError(f"Unused initial weights: {list(source)}")
        attach_input_tap(network, args.output / "input_dumps")
        if args.routing_only:
            state['routing_names'] = attach_routes(network, args.output / 'routing_dumps')
            (args.output / 'routes').mkdir()
        if args.deep_trace:
            attach_taps(network, args.checkpoint_root / "deep")
        elif args.dump_activations:
            attach_taps(network, args.output / "activations")
        if args.forward_trace_step:
            from forward_step_trace import attach_mf
            attach_mf(network, args.checkpoint_root / "forward")
        if args.attention_trace_layer is not None:
            from attention_step_trace import attach_mf
            attach_mf(network, args.checkpoint_root / 'attention', args.attention_trace_layer)
        state["network"] = network
        if args.router_step_trace_output:
            from router_step_trace import attach_mf
            attach_mf(network, args.router_step_trace_output)
        return network

    def create_optimizer(trainer, *positional, **keywords):
        optimizer = native_optimizer(trainer, *positional, **keywords)
        if args.resume_state:
            entries = {item["parameter_name"]: item for item in json.loads(
                (args.resume_state / "optimizer/manifest.json").read_text())}
            for index, parameter in enumerate(optimizer.parameters):
                entry = entries.pop(parameter.name)
                if entry["use_muon"] != bool(optimizer.use_muon[index]):
                    raise ValueError(f"Optimizer family differs: {parameter.name}")
                slots = {"muon_m": optimizer.muon_m[index]} if entry["use_muon"] else {
                    "exp_avg": optimizer.moments1[index], "exp_avg_sq": optimizer.moments2[index]}
                for name, slot in slots.items():
                    item = entry["state"][name]
                    array = np.load(args.resume_state / "optimizer" / item["file"], allow_pickle=False)
                    if hashlib.sha256(np.ascontiguousarray(array)).hexdigest() != item["sha256"]:
                        raise ValueError(f"Optimizer source bytes differ: {parameter.name}:{name}")
                    slot.set_data(Tensor(array))
                    if not np.array_equal(slot.asnumpy(), array):
                        raise ValueError(f"Optimizer restoration differs: {parameter.name}:{name}")
            if entries:
                raise ValueError("Unconsumed MF optimizer states")
            optimizer.global_step.set_data(Tensor(
                np.full(optimizer.global_step.shape, state["offset"], dtype=np.int32),
                dtype=optimizer.global_step.dtype))
            (args.output / "RESTORE.json").write_text(json.dumps({
                "completed_steps": state["offset"], "all_state_arrays_exact": True}))
        state["optimizer"] = optimizer
        if args.gradient_output or args.gradient_hash_output:
            from mf_gradient_tap import attach
            destination = args.gradient_output or (args.checkpoint_root / 'gradients')
            attach(optimizer, destination, max_elements=args.gradient_max_elements)
        return optimizer

    class SaveProbe(Callback):
        def on_train_step_end(self, run_context):
            step = int(run_context.original_args().cur_step_num) + state['offset']
            if args.save_state_step is None or step == args.save_state_step:
                export_state(args.checkpoint_root / f'step-{step:06d}', state['network'], state['optimizer'], step)

    class PruneForwardTrace(Callback):
        def on_train_step_end(self, run_context):
            step = int(run_context.original_args().cur_step_num) + state['offset']
            if step == args.forward_trace_step:
                return
            destination = args.checkpoint_root / 'forward'
            paths = list(destination.glob('*.npy'))
            record = {'step': step, 'files': len(paths), 'bytes': sum(p.stat().st_size for p in paths)}
            for path in paths:
                if path.resolve().parent != destination.resolve():
                    raise ValueError('Unexpected forward trace cleanup path')
                path.unlink()
            with (destination / 'early-step-cleanup.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')

    def sort_callbacks(callbacks):
        result = [ObserveMetrics(args.output, state), *native_callbacks(callbacks)]
        if args.gradient_output or args.save_state_step is not None:
            result.append(SaveProbe())
        if args.gradient_hash_output:
            from gradient_hash_stream import native_callback
            result.append(native_callback(args.checkpoint_root / 'gradients', args.gradient_hash_output,
                                          state['offset'], args.gradient_hash_keep_step))
        if args.forward_trace_step:
            result.append(PruneForwardTrace())
        if args.resume_state is None:
            result.append(SaveContinuation(args.output, state, args.checkpoint_root, 200))
        return result

    BaseTrainer.create_network = create_network
    BaseTrainer.create_optimizer_scheduler = create_optimizer
    BaseTrainer.sort_callbacks = staticmethod(sort_callbacks)
    sys.argv = ["/home/xxl/repos/mindformers/run_mindformer.py", "--config", args.config]
    runpy.run_path(sys.argv[0], run_name="__main__")
    (args.output / "WORKER_SUCCESS.json").write_text(json.dumps({
        "completed_steps": len(state["losses"]), "offset": state["offset"]}))


if __name__ == "__main__":
    main()
