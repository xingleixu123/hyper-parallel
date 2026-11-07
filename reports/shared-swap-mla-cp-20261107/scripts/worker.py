"""Bounded real-NPU checks for the public swap and MLA callback changes."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import traceback

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint
import torch_npu

import hyper_parallel
from hyper_parallel import init_device_mesh
from hyper_parallel.components.functional.npu_fusion_attention import npu_fusion_attention_forward
from hyper_parallel.core.optimizer import SwapOptimizerConfig, swap_optimizer
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.apply import apply_sharding_plan
from hyper_parallel.distributed.context_parallel import mla_context_parallel as mla_runtime
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.deepseek_v3.adapter.distributed.context_parallel import mla_cp_wrapper
from hyper_parallel.trainer.config import Target
from tests.common.mla_cp_utils import MLADimensions, attention_options, make_mla, position_embeddings

CALLS = []
ROPE_CALLS = []


def rope_function():
    public = getattr(mla_runtime, "apply_mla_rope", None)
    return public if public is not None else mla_runtime._apply_mla_rope


def inverse_rope(tensor, positions, *, interleaved):
    ROPE_CALLS.append({"shape": list(tensor.shape), "interleaved": interleaved})
    changed = None if positions is None else (positions[0], -positions[1])
    return rope_function()(tensor, changed, interleaved=interleaved)


def head_scaled_attention(module, query, key, value, mask=None, **options):
    offset = options.pop("mla_head_offset", 0)
    local_heads = options.pop("mla_local_heads", query.shape[1])
    gains = module.probe_head_gain
    assert gains.numel() == local_heads
    assert 0 <= offset and offset + query.shape[1] <= local_heads
    CALLS.append({"role": module.probe_role, "offset": offset, "local_heads": local_heads,
                  "computed_heads": query.shape[1], "q_shape": list(query.shape),
                  "k_shape": list(key.shape)})
    query = query * gains[offset:offset + query.shape[1]].view(1, -1, 1, 1)
    return npu_fusion_attention_forward(module, query, key, value, mask, **options)


def compare(actual, expected, dtype, label):
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    atol, rtol = {torch.float16: (5e-4, 0.012), torch.bfloat16: (4e-3, 0.06)}[dtype]
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=lambda s: label + ": " + s)
    return float((actual - expected).abs().max())


class Model(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.self_attn = attention
        self.config = attention.config
        self.config.model_type = "deepseek_v3"
        self.config.architectures = ["DeepseekV3ForCausalLM"]

    def forward(self, inputs, **options):
        return self.self_attn(inputs, **options)[0]


def sharded(tensor, name, degree, rank):
    if ".q_b_proj." in name or ".kv_b_proj." in name:
        return tensor.chunk(degree, 0)[rank]
    if name.endswith("o_proj.weight"):
        return tensor.chunk(degree, 1)[rank]
    return tensor


def build(mesh, tp, cp, device, strategy, overlap, ulysses, custom):
    dtype = torch.bfloat16
    dims = MLADimensions(8, 32, 32, 128, 64, 128)
    torch.manual_seed(409)
    reference = Model(make_mla(dims, dtype=dtype)).to(device)
    parallel = deepcopy(reference)
    callbacks = {"rotary_fn": inverse_rope, "attention_interface": head_scaled_attention} if custom else {}
    specification = ModuleShardingSpec(
        inner_target="self", region_dispatch=False,
        inner_wrapper=Target(
            mla_cp_wrapper,
            target_path="hyper_parallel.models.deepseek_v3.adapter.distributed.context_parallel.mla_cp_wrapper",
            strategy=strategy, overlap=overlap, ulysses_degree=ulysses, **callbacks,
        ),
    )
    plan = ShardingPlanner(plan_overrides={"self_attn": specification}).plan(
        parallel, mesh, tp_size=tp, cp_size=cp, sequence_parallel=False,
    )
    parallel, source_info = apply_sharding_plan(parallel, plan, mesh)
    tp_rank = mesh["tp"].get_local_rank() if tp > 1 else 0
    if tp > 1:
        assert source_info
    for name, parameter in parallel.named_parameters():
        compare(parameter, sharded(reference.get_parameter(name), name, tp, tp_rank), dtype, "initial " + name)
    runtime = parallel.self_attn.mla_cp_runtime
    assert runtime.plan.compute_heads == dims.heads // (tp * ulysses)
    if custom:
        assert runtime.rotary_fn is inverse_rope and runtime.attention_interface is head_scaled_attention
        gains = (0.75 + torch.arange(dims.heads, dtype=torch.float32, device=device) / 16).to(dtype)
        reference.self_attn.register_buffer("probe_head_gain", gains)
        parallel.self_attn.register_buffer("probe_head_gain", gains.chunk(tp)[tp_rank].contiguous())
        reference.self_attn.probe_role = "reference"
        parallel.self_attn.probe_role = "parallel"
        reference.self_attn.attention_interface = head_scaled_attention
    return reference, parallel, dims


def distributed_case(mesh, tp, cp, device, strategy, overlap, ulysses, custom, packed):
    CALLS.clear()
    ROPE_CALLS.clear()
    reference, parallel, dims = build(mesh, tp, cp, device, strategy, overlap, ulysses, custom)
    dtype = torch.bfloat16
    tp_mesh, cp_mesh = (mesh["tp"] if tp > 1 else None), mesh["cp"]
    tp_rank = tp_mesh.get_local_rank() if tp_mesh is not None else 0
    cp_rank = cp_mesh.get_local_rank()
    options = attention_options((3, 31, 64) if packed else None)
    positions = position_embeddings(64, dims.rope_dim, options, dtype=dtype, device=device)
    expected_positions = (positions[0], -positions[1]) if custom else positions
    begin, end = cp_rank * 64 // cp, (cp_rank + 1) * 64 // cp
    local_positions = tuple(x[:, begin:end].contiguous() for x in positions)
    metrics = {"output_max_abs": 0.0, "input_grad_max_abs": 0.0, "loss_max_abs": 0.0}
    for micro in range(2):
        torch.manual_seed(1000 + micro)
        batch = 1 if packed or tp > 1 else 2
        hidden = torch.randn(batch, 64, 64, dtype=dtype).to(device).requires_grad_()
        local_hidden = hidden.detach()[:, begin:end].contiguous().requires_grad_()
        target = torch.randn_like(hidden)
        expected = reference(hidden, position_embeddings=expected_positions, **options)

        def forward(inputs):
            return parallel(inputs, position_embeddings=local_positions, **options)

        actual = checkpoint(forward, local_hidden, use_reentrant=False)
        metrics["output_max_abs"] = max(metrics["output_max_abs"], compare(
            actual, expected[:, begin:end], dtype, "parallel output"))
        expected_loss = (expected.float() - target.float()).square().mean() / 2
        loss = (actual.float() - target[:, begin:end].float()).square().sum() / (hidden.numel() * tp * 2)
        global_loss = loss.detach().clone()
        dist.all_reduce(global_loss)
        metrics["loss_max_abs"] = max(metrics["loss_max_abs"], compare(global_loss, expected_loss, dtype, "loss"))
        expected_loss.backward()
        loss.backward()
        gradient = local_hidden.grad.clone()
        if tp_mesh is not None:
            dist.all_reduce(gradient, group=tp_mesh.get_group())
        metrics["input_grad_max_abs"] = max(metrics["input_grad_max_abs"], compare(
            gradient, hidden.grad[:, begin:end], dtype, "logical input gradient"))
    actual_gradients, expected_gradients = [], []
    for name, parameter in parallel.named_parameters():
        assert parameter.grad is not None
        dist.all_reduce(parameter.grad, group=cp_mesh.get_group())
        replicated = not (".q_b_proj." in name or ".kv_b_proj." in name or name.endswith("o_proj.weight"))
        if tp_mesh is not None and replicated:
            dist.all_reduce(parameter.grad, group=tp_mesh.get_group())
        target_gradient = sharded(reference.get_parameter(name).grad, name, tp, tp_rank)
        compare(parameter.grad, target_gradient, dtype, "parameter " + name)
        actual_gradients.append(parameter.grad.detach().cpu().double().flatten())
        expected_gradients.append(target_gradient.detach().cpu().double().flatten())
    actual_gradient, target_gradient = torch.cat(actual_gradients), torch.cat(expected_gradients)
    relative = float((actual_gradient - target_gradient).norm() / target_gradient.norm().clamp_min(1e-12))
    assert relative < 0.025, relative
    parallel_calls = [row for row in CALLS if row["role"] == "parallel"]
    if custom:
        assert len(parallel_calls) >= 2 and len(ROPE_CALLS) >= 4
        offset = (cp_rank % ulysses) * (dims.heads // (tp * ulysses))
        assert all(row["offset"] == offset and row["local_heads"] == dims.heads // tp for row in parallel_calls)
        assert {row["shape"][2] for row in ROPE_CALLS} == {1, dims.heads // tp}
    return {"tp": tp, "cp": cp, "ulysses": ulysses, "strategy": strategy, "overlap": overlap,
            "custom_callbacks": custom, "packed": packed, "microbatches": 2, "checkpoint": "non-reentrant",
            "gradient_relative_l2": relative, **metrics, "attention_calls": CALLS.copy(),
            "rotary_calls": ROPE_CALLS.copy()}


def callbacks(device):
    assert int(os.environ["WORLD_SIZE"]) == 4
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    result = []
    try:
        for tp, cp, ulysses, strategy, overlap, custom, packed in (
            (2, 2, 2, "expanded_ulysses", False, False, False),
            (2, 2, 2, "expanded_ulysses", False, True, False),
            (2, 2, 2, "expanded_ulysses", True, True, False),
            (2, 2, 2, "latent_kv_head", True, True, True),
            (1, 4, 2, "latent_kv_head", True, True, False),
        ):
            mesh = init_device_mesh(device.type, (cp, tp), mesh_dim_names=("cp", "tp"))
            result.append(distributed_case(mesh, tp, cp, device, strategy, overlap, ulysses, custom, packed))
        return result
    finally:
        dist.destroy_process_group()


def single(device):
    result = []
    torch.manual_seed(811)
    positions = position_embeddings(14, 64, attention_options(), dtype=torch.float32)
    positions = tuple(x.reshape(2, 7, 64).to(device) for x in positions)
    for dtype in (torch.bfloat16, torch.float16):
        for interleaved in (False, True):
            for heads in (4, 1):
                source = torch.randn(2, 7, heads, 64, dtype=dtype).to(device).requires_grad_()
                reference = source.detach().cpu().double().requires_grad_()
                cos, sin = (x.to(dtype).cpu().double().unsqueeze(2) for x in positions)
                ordered = torch.cat((reference[..., 0::2], reference[..., 1::2]), -1) if interleaved else reference
                first, second = ordered.chunk(2, -1)
                expected = ordered * cos + torch.cat((-second, first), -1) * sin
                actual = rope_function()(source, positions, interleaved=interleaved)
                output_error = compare(actual, expected, dtype, "shared RoPE")
                target = torch.randn_like(expected)
                actual.backward(target.to(device=device, dtype=dtype))
                expected.backward(target.to(dtype).double())
                grad_error = compare(source.grad, reference.grad, dtype, "shared RoPE gradient")
                result.append({"kind": "rope", "dtype": str(dtype), "interleaved": interleaved,
                               "heads": heads, "output_max_abs": output_error, "grad_max_abs": grad_error})
        for packed, explicit in ((False, False), (True, False), (False, True)):
            query = torch.randn(1, 4, 32, 192, dtype=dtype).to(device).requires_grad_()
            key = torch.randn_like(query).requires_grad_()
            value = torch.randn(1, 4, 32, 128, dtype=dtype).to(device).requires_grad_()
            refs = [x.detach().cpu().float().requires_grad_() for x in (query, key, value)]
            allowed = torch.ones(32, 32, dtype=torch.bool).tril()
            if packed:
                ids = torch.cat((torch.zeros(3), torch.ones(14), torch.full((15,), 2)))
                allowed &= ids[:, None] == ids[None, :]
            if explicit:
                allowed[3] = False
            valid = allowed.any(-1, keepdim=True)
            scale = 0.73 / 192**0.5
            scores = (refs[0] @ refs[1].transpose(-2, -1)) * scale
            scores = torch.where(valid, scores.masked_fill(~allowed, -torch.inf), 0.0)
            expected = (scores.softmax(-1).masked_fill(~valid, 0.0) @ refs[2]).transpose(1, 2)
            actual = npu_fusion_attention_forward(
                nn.Module(), query, key, value, allowed[None, None].to(device) if explicit else None,
                scaling=scale, is_causal=not explicit,
                actual_seq_qlen=(3, 17, 32) if packed else None,
                actual_seq_kvlen=(3, 17, 32) if packed else None,
            )[0]
            output_error = compare(actual, expected, dtype, "default attention")
            target = torch.randn_like(expected)
            (actual.float() * target.to(device)).mean().backward()
            (expected * target).mean().backward()
            errors = [compare(x.grad, y.grad, dtype, "attention gradient") for x, y in zip((query, key, value), refs)]
            result.append({"kind": "attention", "dtype": str(dtype), "packed": packed,
                           "explicit_4d_mask": explicit, "output_max_abs": output_error, "grad_max_abs": max(errors)})
    return result


def tensor_record(value):
    if not isinstance(value, torch.Tensor):
        return {"value": value}
    raw = value.detach().contiguous().cpu()
    return {"shape": list(raw.shape), "dtype": str(raw.dtype),
            "sha256": hashlib.sha256(raw.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def swap(device, expected_guard):
    parameter = nn.Parameter(torch.linspace(-1, 1, 256, device=device).reshape(16, 16))
    optimizer = torch.optim.AdamW([parameter], lr=0.001, foreach=False)
    count = {"pre": 0, "post": 0}

    def pre_hook(_optimizer, _args, _kwargs):
        count["pre"] += 1

    def post_hook(_optimizer, _args, _kwargs):
        count["post"] += 1

    before = tensor_record(parameter)
    handles = [optimizer.register_step_pre_hook(pre_hook), optimizer.register_step_post_hook(post_hook)]
    rejected = False
    try:
        wrapped = swap_optimizer(optimizer, SwapOptimizerConfig(swap_times=2, min_numel=1, packed_swap=False))
    except ValueError as exc:
        assert "step hooks" in str(exc)
        rejected = True
    if expected_guard:
        assert rejected and count == {"pre": 0, "post": 0} and not optimizer.state
        assert tensor_record(parameter) == before
    else:
        assert not rejected
        parameter.grad = torch.ones_like(parameter)
        wrapped.step()
        torch.npu.synchronize()
        assert count == {"pre": 0, "post": 0}
    for handle in handles:
        handle.remove()
    parameter = nn.Parameter(torch.linspace(-1, 1, 256, device=device).reshape(16, 16))
    optimizer = torch.optim.AdamW([parameter], lr=0.001, foreach=False)
    wrapped = swap_optimizer(optimizer, SwapOptimizerConfig(swap_times=2, min_numel=1, packed_swap=False))
    for step in range(3):
        parameter.grad = torch.full_like(parameter, (step + 1) / 16)
        wrapped.step()
    torch.npu.synchronize()
    state = wrapped.state_dict()
    slots = tuple(wrapped.adapter.tracked_slots())
    assert slots and any(slot.cpu_tensor is not None for slot in slots)
    return {"hook_guard_expected": expected_guard, "rejected": rejected, "hook_counts": count,
            "no_hook_steps": 3, "parameter": tensor_record(parameter),
            "optimizer_state": {str(k): {name: tensor_record(v) for name, v in values.items()}
                                for k, values in state["state"].items()},
            "tracked_slots": len(slots), "cpu_mirrors": sum(slot.cpu_tensor is not None for slot in slots),
            "copy_stream_created": wrapped.runtime.copy_stream is not None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("single", "callbacks", "swap"), required=True)
    parser.add_argument("--expect-guard", choices=("yes", "no"), default="yes")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.output.mkdir(parents=True, exist_ok=True)
    record = {"mode": args.mode, "rank": rank, "local_rank": local_rank,
              "python": platform.python_version(), "torch": torch.__version__,
              "torch_npu": torch_npu.__version__, "import_path": hyper_parallel.__file__}
    try:
        torch.npu.set_device(local_rank)
        device = torch.device("npu", local_rank)
        record["device_count"] = torch.npu.device_count()
        record["device_name"] = torch.npu.get_device_name(local_rank)
        record["result"] = (callbacks(device) if args.mode == "callbacks"
                            else single(device) if args.mode == "single"
                            else swap(device, args.expect_guard == "yes"))
        torch.npu.synchronize()
        record["peak_allocated_bytes"] = torch.npu.max_memory_allocated()
        record["status"] = "passed"
    except Exception as exc:
        record.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        (args.output / ("worker-rank-" + str(rank) + ".json")).write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf8")
        print(json.dumps({"mode": args.mode, "rank": rank, "status": record.get("status")}), flush=True)


if __name__ == "__main__":
    main()
