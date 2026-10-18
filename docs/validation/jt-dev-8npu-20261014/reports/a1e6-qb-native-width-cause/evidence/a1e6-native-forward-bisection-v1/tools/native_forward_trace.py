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
"""Bounded native forward observation; no collectives, replay, or tensor replacement.

Full bytes are hashed in canonical 512-token by 128-feature blocks. Only hashes
and a small numerical grid leave the hook. This is a boundary diagnostic, not a
proof of complete initial parameter/state identity or a performance measurement.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


TOKEN_BLOCK = 512
FEATURE_BLOCK = 128
WATCHED_CALLS = {
    "model.embed_tokens": 2,
    "model.layers.0": 1,
    "model.layers.1": 1,
    "model.norm": 1,
    "mtp.layers.0.enorm": 1,
    "mtp.layers.0.hnorm": 1,
    "mtp.layers.0.eh_proj": 1,
    "mtp.layers.0.transformer_layer": 1,
    "mtp.layers.0.final_layernorm": 1,
}


@dataclass(frozen=True)
class Topology:
    """Describe the only supported mapping; hidden size may shrink in CPU tests."""

    rank: int
    dp_rank: int
    tp_rank: int
    tp_size: int
    dp_size: int
    sequence_parallel: bool
    hidden_size: int
    cp_size: int = 1
    pp_size: int = 1
    ep_size: int = 1
    micro_batch_size: int = 1
    sequence_length: int = 4096

    def __post_init__(self) -> None:
        """Reject unsupported topology before attaching observers."""
        if self.cp_size != 1 or self.pp_size != 1 or self.ep_size != 1:
            raise ValueError("Forward observer supports CP1/PP1/EP1 only")
        if self.tp_size not in (1, 2) or self.dp_size * self.tp_size != 8:
            raise ValueError("Forward observer requires DP8/TP1 or DP4/TP2")
        if not 0 <= self.rank < 8 or not 0 <= self.dp_rank < self.dp_size:
            raise ValueError("Invalid global or DP coordinate")
        if not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("Invalid TP coordinate")
        if self.micro_batch_size != 1 or self.sequence_length != 4096:
            raise ValueError("Forward observer requires MBS1 and sequence 4096")
        if self.hidden_size < FEATURE_BLOCK or self.hidden_size % FEATURE_BLOCK:
            raise ValueError("Hidden size must be a positive multiple of 128")


def _layout_metadata(value: Any) -> tuple[Any, tuple[Any, ...]]:
    layout = getattr(value, "layout", None)
    is_partial = getattr(layout, "is_partial", None)
    if callable(is_partial) and is_partial():
        raise ValueError("Partial hidden tensors are not complete hidden states")
    placements = getattr(value, "placements", None)
    if placements is None:
        placements = getattr(layout, "placements", None)
    return getattr(value, "device_mesh", None), tuple(placements or ())


def _placement_kind(placement: Any) -> str:
    for name in ("is_partial", "is_ragged_shard"):
        predicate = getattr(placement, name, None)
        if callable(predicate) and predicate():
            raise ValueError("Partial and ragged placements are unsupported")
    if hasattr(placement, "split_factor"):
        raise ValueError("Strided shard layouts are unsupported")
    if callable(getattr(placement, "is_shard", None)) and placement.is_shard():
        if placement.dim != 1:
            raise ValueError("Feature/batch sharding cannot represent full-width hidden states")
        return "sequence"
    if callable(getattr(placement, "is_replicate", None)) and placement.is_replicate():
        return "replicate"
    raise ValueError(f"Unknown placement type: {type(placement).__name__}")


def local_hidden(value: Any, topology: Topology, sequence_offset: int | None = None) -> tuple[torch.Tensor, int, str]:
    """Validate full-width hidden layout and return a detached local view.

    Plain local tensors use the explicitly selected JT boundary contract and
    actual TP coordinate. DTensors must additionally expose compatible mesh and
    placement metadata. This function never redistributes or gathers a tensor.
    """
    mesh, placements = _layout_metadata(value)
    distributed = callable(getattr(value, "to_local", None))
    if distributed and not placements:
        raise ValueError("Distributed tensor has no interpretable placement metadata")
    if not callable(getattr(value, "detach", None)):
        raise TypeError("Expected one tensor at the selected output boundary")
    detached = value.detach()
    local = detached.to_local() if distributed else detached
    if not isinstance(local, torch.Tensor) or local.ndim != 3:
        raise ValueError("Hidden output must be a [1, sequence, full_hidden] tensor")
    if local.shape[0] != 1 or local.shape[2] != topology.hidden_size or not local.is_floating_point():
        raise ValueError(f"Unknown hidden shape/dtype: {tuple(local.shape)}, {local.dtype}")
    sequence = local.shape[1]
    if sequence == topology.sequence_length:
        offset, mapping = 0, "full_sequence"
    elif topology.sequence_parallel and topology.tp_size == 2 and sequence == 2048:
        offset, mapping = topology.tp_rank * sequence, "contiguous_tp_sequence_shard"
    else:
        raise ValueError(f"Unknown sequence shape: {tuple(local.shape)}")
    if sequence_offset is not None and sequence_offset != offset:
        raise ValueError(f"Wrong sequence offset: expected {offset}, received {sequence_offset}")
    if placements:
        kinds = [_placement_kind(placement) for placement in placements]
        shard_axes = [axis for axis, kind in enumerate(kinds) if kind == "sequence"]
        if len(shard_axes) > 1 or bool(shard_axes) != (mapping == "contiguous_tp_sequence_shard"):
            raise ValueError("Placement metadata contradicts local sequence coverage")
        if tuple(value.shape) != (1, topology.sequence_length, topology.hidden_size):
            raise ValueError("Distributed global shape differs from the declared hidden state")
        if shard_axes:
            axis = shard_axes[0]
            names = getattr(mesh, "mesh_dim_names", None)
            coordinate_fn = getattr(mesh, "get_coordinate", None)
            mesh_shape = getattr(mesh, "shape", None)
            if mesh_shape is None:
                mesh_shape = getattr(mesh, "mesh_shape", None)
            coordinate = coordinate_fn() if callable(coordinate_fn) else None
            if (names is None or mesh_shape is None or coordinate is None
                    or len(names) != len(placements) or len(mesh_shape) != len(placements)
                    or len(coordinate) != len(placements) or names[axis] != "tp"
                    or mesh_shape[axis] != 2 or coordinate[axis] != topology.tp_rank):
                raise ValueError("Sequence shard mesh does not match the actual TP coordinate")
    return local, offset, mapping


def summarize_hidden(value: Any, topology: Topology, sequence_offset: int | None = None) -> dict[str, Any]:
    """Hash every canonical block and retain only a small CPU numerical grid."""
    local, offset, mapping = local_hidden(value, topology, sequence_offset)
    cpu = local.cpu().contiguous()
    if not bool(torch.isfinite(cpu).all()):
        raise ValueError("Non-finite hidden output")
    blocks = []
    for token in range(0, cpu.shape[1], TOKEN_BLOCK):
        for feature in range(0, topology.hidden_size, FEATURE_BLOCK):
            block = cpu[:, token:token + TOKEN_BLOCK, feature:feature + FEATURE_BLOCK].contiguous()
            blocks.append({
                "tokens": [offset + token, offset + token + TOKEN_BLOCK],
                "features": [feature, feature + FEATURE_BLOCK],
                "sha256": hashlib.sha256(block.view(torch.uint8).numpy().tobytes()).hexdigest(),
            })
    positions = sorted(set(range(0, 4096, TOKEN_BLOCK)) | {4095})
    positions = [position for position in positions if offset <= position < offset + cpu.shape[1]]
    stride = max(FEATURE_BLOCK, (topology.hidden_size // 8 // FEATURE_BLOCK) * FEATURE_BLOCK)
    features = sorted(set(range(0, topology.hidden_size, stride)) | {topology.hidden_size - 1})
    grid = cpu[0, [position - offset for position in positions]][:, features].float()
    return {
        "shape": list(local.shape), "global_shape": [1, 4096, topology.hidden_size],
        "dtype": str(local.dtype), "sequence_offset": offset, "mapping": mapping,
        "blocks": blocks, "grid": {"tokens": positions, "features": features, "values": grid.tolist()},
    }


def batch_identity(model_inputs: Mapping[str, Any], loss_inputs: Mapping[str, Any]) -> str:
    """Hash actual IDs and shifted labels without persisting their token contents."""
    hasher = hashlib.sha256()
    fields = (("input_ids", model_inputs.get("input_ids")), ("shift_labels", loss_inputs.get("shift_labels")))
    for name, value in fields:
        if (not isinstance(value, torch.Tensor) or callable(getattr(value, "to_local", None))
                or tuple(value.shape) != (1, 4096) or value.dtype not in (torch.int32, torch.int64)):
            raise ValueError(f"Expected complete integer [1,4096] {name} at the CP1 batch boundary")
        cpu = value.detach().cpu().contiguous()
        hasher.update(f"{name}:{tuple(cpu.shape)}:{cpu.dtype}:".encode())
        hasher.update(cpu.view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def wrap_get_batch(original: Any, trace: NativeForwardTrace) -> Any:
    """Preserve eager and deferred batch contracts, recording at consumption time."""
    def observe(iterator: Any) -> Any:
        inputs = original(iterator)
        trace.queue_batch(*inputs)
        return inputs

    prepare_step = getattr(original, "prepare_step", None)
    if callable(prepare_step):
        def observe_prepared(iterator: Any, micro_steps: int) -> Any:
            supervision, batches = prepare_step(iterator, micro_steps)

            def observe_inputs(inputs: Any) -> Any:
                trace.queue_batch(*inputs)
                return inputs

            return supervision, map(observe_inputs, batches)

        observe.prepare_step = observe_prepared
    return observe


class NativeForwardTrace:
    """Observe exact selected module outputs during optimizer step one only."""

    def __init__(self, model: torch.nn.Module, output: Path, topology: Topology) -> None:
        self.topology = topology
        self.path = output / f"native-forward-rank{topology.rank}.jsonl"
        self.enabled = False
        self.pending = deque()
        self.batch: str | None = None
        self.micro = -1
        self.calls: Counter[str] = Counter()
        self.seen_batches: set[str] = set()
        self.handles = []
        modules = dict(model.named_modules())
        missing = sorted(set(WATCHED_CALLS) - set(modules))
        if missing:
            raise ValueError(f"Required native boundaries are missing: {missing}")
        if len({id(modules[name]) for name in WATCHED_CALLS}) != len(WATCHED_CALLS):
            raise ValueError("Selected boundaries contain module aliases")
        metadata = {
            "kind": "metadata", "schema": "native-forward-v1", "topology": asdict(topology),
            "watched_calls_per_micro": WATCHED_CALLS,
            "scope": "First-step outputs only; no complete NPU parameter identity; no performance claim",
            "token_block": TOKEN_BLOCK, "feature_block": FEATURE_BLOCK,
        }
        with self.path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(metadata, sort_keys=True) + "\n")
        for name in WATCHED_CALLS:
            self.handles.append(modules[name].register_forward_hook(self._hook(name)))

    def _event(self, kind: str, **fields: Any) -> None:
        row = {"kind": kind, "rank": self.topology.rank, **fields}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    def start_step(self, completed_steps: int) -> None:
        """Arm only the first optimizer step, never a resumed step."""
        self.enabled = completed_steps == 0
        if self.enabled:
            self.pending.clear()
            self.batch = None
            self.micro = -1
            self.calls.clear()
            self.seen_batches.clear()

    def queue_batch(self, model_inputs: Mapping[str, Any], loss_inputs: Mapping[str, Any]) -> None:
        """Keep only the identity until the matching microbatch actually executes."""
        if self.enabled:
            self.pending.append(batch_identity(model_inputs, loss_inputs))

    def begin_microbatch(self, fields: Mapping[str, Any]) -> None:
        """Bind output hooks to the batch consumed by this actual forward call."""
        if not self.enabled:
            return
        self._check_previous_calls()
        identity = batch_identity(fields, fields)
        if not self.pending or self.pending.popleft() != identity:
            raise ValueError("Forward inputs do not match the queued batch identity")
        if identity in self.seen_batches:
            raise ValueError("Ambiguous repeated batch SHA within this rank's optimizer step")
        self.seen_batches.add(identity)
        self.batch = identity
        self.micro += 1
        self.calls.clear()
        self._event("batch", step=1, micro=self.micro, dp=self.topology.dp_rank,
                    tp=self.topology.tp_rank, batch_sha256=identity)

    def _check_previous_calls(self) -> None:
        if self.batch is not None and dict(self.calls) != WATCHED_CALLS:
            raise ValueError(f"Missing or repeated forward boundaries: {dict(self.calls)}")

    def _hook(self, name: str) -> Any:
        def observe(_module: Any, _args: Any, value: Any) -> None:
            if not self.enabled:
                return
            if self.batch is None:
                raise ValueError("A selected module executed before batch binding")
            call = self.calls[name]
            self.calls[name] += 1
            if self.calls[name] > WATCHED_CALLS[name]:
                raise ValueError(f"Unexpected replay or repeated call at {name}")
            try:
                summary = summarize_hidden(value, self.topology)
            except (TypeError, ValueError) as error:
                self._event("error", step=1, micro=self.micro, module=name, message=str(error))
                raise
            self._event("hidden", step=1, micro=self.micro, dp=self.topology.dp_rank,
                        tp=self.topology.tp_rank, batch_sha256=self.batch, module=name, call=call,
                        stage="output", **summary)
            # Returning None preserves the exact original Tensor object and graph.
        return observe

    def finish_step(self, completed_steps: int) -> None:
        """Check first-step completeness, then remove every observation hook."""
        if self.enabled:
            self._check_previous_calls()
            if completed_steps != 1 or self.pending or self.micro + 1 != self.topology.tp_size:
                raise ValueError("Incomplete first-step batch coverage")
            self._event("step_complete", step=1, micro_batches=self.micro + 1,
                        unique_batches=len(self.seen_batches))
            self.enabled = False
            self.close()

    def close(self) -> None:
        """Release hooks without retaining any training tensor."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def finish_run(self, completed_steps: int) -> None:
        """Mark the three-step observation run complete."""
        self.close()
        if completed_steps != 3:
            raise ValueError("The observer requires exactly three completed optimizer steps")
        self._event("run_complete", completed_steps=completed_steps)


def canonicalize_run(rank_records: Mapping[int, Iterable[dict[str, Any]]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Validate all eight files and merge by input identity, never by rank number.

    Repeated SHA groups across distinct DP/micro pairs are ambiguous and rejected.
    Replica overlap must agree; every token/feature block must be present. This
    read-only offline pass performs no distributed operation.
    """
    if set(rank_records) != set(range(8)):
        raise ValueError("Exactly eight rank files are required")
    batches: dict[tuple[int, int], dict[int, str]] = {}
    hidden: dict[tuple[Any, ...], dict[str, Any]] = {}
    coordinate_set = set()
    reference: Topology | None = None
    for rank, records_iter in rank_records.items():
        records = list(records_iter)
        metas = [row for row in records if row["kind"] == "metadata"]
        if len(metas) != 1:
            raise ValueError("Missing or duplicate rank metadata")
        topology = Topology(**metas[0]["topology"])
        if topology.rank != rank or metas[0]["watched_calls_per_micro"] != WATCHED_CALLS:
            raise ValueError("Rank or boundary contract mismatch")
        if reference is None:
            reference = topology
        if (topology.tp_size, topology.hidden_size, topology.sequence_parallel) != (
                reference.tp_size, reference.hidden_size, reference.sequence_parallel):
            raise ValueError("Mixed topology or hidden contracts")
        coordinate = (topology.dp_rank, topology.tp_rank)
        if coordinate in coordinate_set:
            raise ValueError("Duplicate DP/TP coordinates")
        coordinate_set.add(coordinate)
        completed = [row for row in records if row["kind"] == "step_complete"]
        run_completed = [row for row in records if row["kind"] == "run_complete"]
        if (len(completed) != 1 or completed[0]["step"] != 1
                or completed[0]["micro_batches"] != topology.tp_size
                or len(run_completed) != 1 or run_completed[0]["completed_steps"] != 3):
            raise ValueError("Incomplete three-step run or first-step observation")
        batch_rows = [row for row in records if row["kind"] == "batch"]
        if len(batch_rows) != topology.tp_size or {row["micro"] for row in batch_rows} != set(range(topology.tp_size)):
            raise ValueError("Missing or duplicate microbatch identity")
        rank_batches = {row["micro"]: row["batch_sha256"] for row in batch_rows}
        counts: Counter[tuple[Any, ...]] = Counter()
        for row in records:
            if row["kind"] in ("metadata", "step_complete", "run_complete"):
                continue
            if row["kind"] == "error":
                raise ValueError("Observer recorded a layout error")
            if (row["step"] != 1 or row["rank"] != rank or row["dp"] != topology.dp_rank
                    or row["tp"] != topology.tp_rank or row["batch_sha256"] != rank_batches.get(row["micro"])):
                raise ValueError("Event identity does not match its physical batch")
            if row["kind"] == "batch":
                batches.setdefault((row["dp"], row["micro"]), {})[row["tp"]] = row["batch_sha256"]
                continue
            if row["kind"] != "hidden" or row["module"] not in WATCHED_CALLS or row["stage"] != "output":
                raise ValueError("Unknown boundary event")
            event_key = (row["micro"], row["module"], row["call"])
            counts[event_key] += 1
            key = (row["batch_sha256"], row["module"], row["call"])
            merged = hidden.setdefault(key, {"dtype": row["dtype"], "blocks": {}, "grid": {}})
            if row["dtype"] != merged["dtype"] or row["global_shape"] != [1, 4096, topology.hidden_size]:
                raise ValueError("Mixed dtype/global shape for a logical hidden state")
            if row["shape"] == [1, 4096, topology.hidden_size]:
                begin, end, mapping = 0, 4096, "full_sequence"
            elif (topology.sequence_parallel and topology.tp_size == 2
                  and row["shape"] == [1, 2048, topology.hidden_size]):
                begin, end = topology.tp_rank * 2048, (topology.tp_rank + 1) * 2048
                mapping = "contiguous_tp_sequence_shard"
            else:
                raise ValueError("Unknown local shape in saved hidden evidence")
            if row["sequence_offset"] != begin or row["mapping"] != mapping:
                raise ValueError("Saved sequence offset disagrees with the actual TP coordinate")
            local_keys = [(*block["tokens"], *block["features"]) for block in row["blocks"]]
            expected_local = {(token, token + TOKEN_BLOCK, feature, feature + FEATURE_BLOCK)
                              for token in range(begin, end, TOKEN_BLOCK)
                              for feature in range(0, topology.hidden_size, FEATURE_BLOCK)}
            if len(local_keys) != len(expected_local) or set(local_keys) != expected_local:
                raise ValueError("Incomplete local canonical token/feature blocks")
            for block in row["blocks"]:
                block_key = (*block["tokens"], *block["features"])
                previous = merged["blocks"].setdefault(block_key, block["sha256"])
                if previous != block["sha256"]:
                    raise ValueError(f"Replica/overlap disagreement at {key}, {block_key}")
            grid = row["grid"]
            for token, values in zip(grid["tokens"], grid["values"], strict=True):
                for feature, value in zip(grid["features"], values, strict=True):
                    previous = merged["grid"].setdefault((token, feature), value)
                    if previous != value:
                        raise ValueError("Replica numerical-grid disagreement")
        expected_events = {(micro, module, call) for micro in range(topology.tp_size)
                           for module, calls in WATCHED_CALLS.items() for call in range(calls)}
        if set(counts) != expected_events or any(count != 1 for count in counts.values()):
            raise ValueError("Missing or duplicate hidden boundary events")
    batch_shas = []
    for group in batches.values():
        if set(group) != set(range(reference.tp_size)) or len(set(group.values())) != 1:
            raise ValueError("TP peers disagree on the logical input batch")
        batch_shas.append(next(iter(group.values())))
    if len(batch_shas) != 8 or len(set(batch_shas)) != 8:
        raise ValueError("Ambiguous repeated batch SHA across logical DP/micro groups")
    expected_blocks = {(token, token + TOKEN_BLOCK, feature, feature + FEATURE_BLOCK)
                       for token in range(0, 4096, TOKEN_BLOCK)
                       for feature in range(0, reference.hidden_size, FEATURE_BLOCK)}
    for value in hidden.values():
        if set(value["blocks"]) != expected_blocks:
            raise ValueError("Missing or out-of-range canonical token/feature blocks")
    return hidden
