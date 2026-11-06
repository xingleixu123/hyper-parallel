"""External PR1477 planner/boundary/router diagnostic worker.

Default mode derives and compiles plans on CPU without accelerator use.
--execute is explicit and is intended for a parent-reviewed torchrun command.
This is not a MoE execution, expert dispatch, FSDP or model acceptance test.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
from types import ModuleType, SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

sys.dont_write_bytecode = True

CASES = (("tp2_ep1_sp", 1, True), ("tp2_ep2_sp", 2, True), ("tp2_ep2_nosp", 2, False))
FQN = "model.layers.0.mlp"
TP_SIZE, HIDDEN, SEQUENCE, EXPERTS, TOP_K = 2, 3584, 4096, 64, 4


def sha_tensor(tensor):
    tensor = tensor.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def source_identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def cpu_npu_import_stub():
    """Permit metadata-only imports; any actual accelerator function fails."""
    if "torch_npu" in sys.modules:
        raise RuntimeError("CPU plan mode refuses an already imported torch_npu")
    module = ModuleType("torch_npu")
    module.__spec__ = importlib.util.spec_from_loader("torch_npu", loader=None)

    def blocked(*args, **kwargs):
        raise RuntimeError("Metadata-only probe reached an NPU operation")

    for name in ("npu_swiglu", "npu_rotary_mul", "npu_fusion_attention", "npu_rms_norm"):
        setattr(module, name, blocked)
    sys.modules["torch_npu"] = module


def load_runtime(repo, execute):
    """Bind this process explicitly to the selected source, then verify every HP module."""
    repo = repo.resolve()
    if any(name == "hyper_parallel" or name.startswith("hyper_parallel.") for name in sys.modules):
        raise RuntimeError("hyper_parallel was already imported before source binding")
    editable = None
    if execute:
        distribution = importlib.metadata.distribution("hyper_parallel")
        editable = json.loads(distribution.read_text("direct_url.json"))
        if editable.get("url") != repo.as_uri() or editable.get("dir_info", {}).get("editable") is not True:
            raise RuntimeError("Execution requires editable installation bound to --repo")
    else:
        cpu_npu_import_stub()
    os.environ.setdefault("USE_TF", "0")
    spec = importlib.util.spec_from_file_location("hyper_parallel", repo / "hyper_parallel/__init__.py",
        submodule_search_locations=[str(repo / "hyper_parallel")])
    package = importlib.util.module_from_spec(spec)
    sys.modules["hyper_parallel"] = package
    spec.loader.exec_module(package)
    model = importlib.import_module("hyper_parallel.models.xing_4.modeling_xing4_0")
    importlib.import_module("hyper_parallel.models.xing_4.adapter.registration")
    from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
    from hyper_parallel.core.dtensor.dtensor import distribute_tensor
    from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
    from hyper_parallel.distributed._builder.forward_rewriter import _bind_input_indices, _wrap_local_region_forward
    from hyper_parallel.distributed._builder.planner import ShardingPlanner
    from hyper_parallel.distributed._builder.precompiled_boundary import PrecompiledBoundary
    from hyper_parallel.distributed._builder.rule_resolver import _resolve_local_compute_fn
    from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec, local_compute, resolve_placements
    from hyper_parallel.components.functional.aux_loss import aux_loss_scale_context, bind_aux_loss_scale
    escaped = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
        if (name == "hyper_parallel" or name.startswith("hyper_parallel.")) and getattr(module, "__file__", None)
        and not Path(module.__file__).resolve().is_relative_to(repo)}
    if escaped:
        raise RuntimeError("Mixed HP source imports: " + repr(escaped))
    runtime = dict(locals())
    runtime["factory"] = local_compute(router_only_factory)
    runtime["source_receipts"] = [source_identity(path) for path in (
        model.__file__, sys.modules[ShardingPlanner.__module__].__file__,
        sys.modules[PrecompiledBoundary.__module__].__file__,
        sys.modules[_wrap_local_region_forward.__module__].__file__,
        sys.modules[_resolve_local_compute_fn.__module__].__file__)]
    return runtime


def config(runtime):
    return runtime["model"].create_config(dict(
        vocab_size=131072, hidden_size=HIDDEN, intermediate_size=9216, moe_intermediate_size=1024,
        n_routed_experts=EXPERTS, n_shared_experts=1, num_experts_per_tok=TOP_K,
        num_hidden_layers=1, first_k_dense_replace=0, num_nextn_predict_layers=1,
        n_group=8, topk_group=4, moe_router_group_score_topk=2,
        use_pad_tokens=False, moe_aux_loss_coeff=.0005, moe_input_jitter_eps=0.0,
        moe_router_enable_expert_bias=True, moe_router_bias_update_rate=.001,
        moe_n_hash_layers=0, moe_router_force_expert_balance=False, compute_dtype="bfloat16"))


def meta_model(runtime, cfg):
    """Use the actual Xing4MoE parameter tree, with zero-allocation expert tensors."""
    with torch.device("meta"):
        source = runtime["model"].Xing4_0MoE(cfg)
        moe = runtime["model"].Xing4MoE(cfg, source, layer_idx=0)
        layer = nn.Module()
        layer.mlp = moe
        trunk = nn.Module()
        trunk.layers = nn.ModuleList([layer])
        root = nn.Module()
        root.model = trunk
        root.config = cfg
    return root


def build_plan(runtime, cfg, mesh, ep_size, sp):
    """Only the EP1 packed-expert parameter/output opt-out is explicit; no input I/O override."""
    replica = runtime["Replicate"]
    spec_type = runtime["ModuleShardingSpec"]
    fields = dict(local_compute_fn=runtime["factory"], region_dispatch=False)
    if ep_size == 1:
        # Packed SwiGLU cannot be naively TP-split. This is the documented
        # EP1 diagnostic opt-out, matching complete expert outputs. It leaves
        # in_src/in_dst unset, so the actual planner derives both input views.
        fields.update(params={name: {"tp": replica()} for name in
            ("gate.weight", "experts.gate_up_proj", "experts.down_proj")},
            out_src={"output": {"tp": replica()}})
    planner = runtime["ShardingPlanner"](plan_overrides={FQN: spec_type(**fields)})
    model = meta_model(runtime, cfg)
    plan = planner.plan(model, mesh, tp_size=TP_SIZE, cp_size=1, ep_size=ep_size,
        sequence_parallel=sp, loss_parallel=False)
    if FQN not in plan.modules:
        raise RuntimeError("Actual planner did not derive the expected Xing4 MoE boundary")
    spec = plan.modules[FQN]
    if spec.in_src is None or spec.in_dst is None or len(spec.in_src) != 1 or len(spec.in_dst) != 1:
        raise RuntimeError("Expected one authoritative hidden-state input contract")
    source = next(iter(spec.in_src.values()))
    destination = next(iter(spec.in_dst.values()))
    src = tuple(runtime["resolve_placements"](source, plan.mesh_dim_names))
    dst = tuple(runtime["resolve_placements"](destination, plan.mesh_dim_names))
    tp_index = tuple(plan.mesh_dim_names).index("tp")
    tp_dst = dst[tp_index]
    need_tp = isinstance(tp_dst, runtime["Shard"]) and tp_dst.dim % 3 == 1
    if not need_tp and not isinstance(tp_dst, replica):
        raise RuntimeError("Unsupported TP router input placement: " + repr(tp_dst))
    return plan, spec, src, dst, need_tp


def plan_receipt(runtime, plan, spec, src, dst, need_tp, mesh):
    boundary = runtime["PrecompiledBoundary"](spec, mesh, plan.mesh_dim_names)
    return {"boundary_fqn": FQN, "mesh_dim_names": list(plan.mesh_dim_names),
        "in_src": {k: {axis: repr(p) for axis,p in values.items()} for k,values in spec.in_src.items()},
        "in_dst": {k: {axis: repr(p) for axis,p in values.items()} for k,values in spec.in_dst.items()},
        "resolved_src": list(map(repr, src)), "resolved_dst": list(map(repr, dst)),
        "tp_sequence_partition_from_resolved_destination": need_tp,
        "ep_size_metadata": getattr(spec, "_ep_size", None),
        "compiled_input_collectives": [op.collective_type for op in boundary.in_plan],
        "compiled_output_collectives": [op.collective_type for op in boundary.out_plan],
        "parameter_placements": {k: {axis:repr(p) for axis,p in value.items()} for k,value in spec.params.items()},
        "diagnostic_override_scope": "EP1 only: packed expert/gate TP replicate + complete-output out_src; input layouts always planner-derived"}


def assets(args):
    if args.assets:
        from safetensors.torch import load_file
        stored = load_file(str(args.assets), device="cpu")
        x, weight, bias = stored["hidden_states"], stored["router_weight"], stored["router_bias"]
        provenance = {"kind": "external_safetensors", "file": source_identity(args.assets)}
    else:
        generator = torch.Generator(device="cpu").manual_seed(147720261106)
        x = (torch.randn(args.dp_size, 1, SEQUENCE, HIDDEN, generator=generator) * .025).bfloat16()
        weight = torch.randn(EXPERTS, HIDDEN, generator=generator) * .0005
        # A stable partition-sensitive decision, with dense nonzero features.
        # No learned-router decision is forced or replaced in the worker.
        for dp in range(args.dp_size):
            positive = SEQUENCE // (2 + 2*dp)
            x[dp, 0, :, 0] = torch.cat((torch.ones(positive), -torch.ones(SEQUENCE-positive))).roll(dp*317)
            x[dp, 0, :, 1] = 1.0
        weight[:, 0] = torch.cat((torch.full((32,), 4.0), torch.full((32,), -4.0)))
        weight[:, 1] = torch.arange(EXPERTS).remainder(32).float() * .02
        bias = torch.linspace(-.01, .01, EXPERTS)
        provenance = {"kind": "deterministic_synthetic_activations", "seed": 147720261106,
            "not_MF_training_samples": True}
    if tuple(x.shape) != (args.dp_size, 1, SEQUENCE, HIDDEN) or tuple(weight.shape) != (EXPERTS, HIDDEN) or tuple(bias.shape) != (EXPERTS,):
        raise ValueError("Assets must be DP×1×4096×3584, 64×3584 and 64 respectively")
    return x, weight.float(), bias.float(), provenance


def independent_reference(x, weight, bias, cfg, external_scale):
    """Independent FP64 per-sequence equation and explicit task objective; no HP aux primitive."""
    w = weight.detach().cpu().double().requires_grad_()
    states = x.detach().cpu().double()
    scores = F.linear(states.reshape(-1, HIDDEN), w).sigmoid()
    unbiased = scores.topk(TOP_K, dim=-1, sorted=False).indices
    counts = torch.zeros(EXPERTS, dtype=torch.float64).scatter_add_(
        0, unbiased.flatten(), torch.ones_like(unbiased.flatten(), dtype=torch.float64))
    affinity = (scores/(scores.sum(-1, keepdim=True)+1e-20)).mean(0)
    raw_aux = EXPERTS * (affinity * (counts / unbiased.numel())).sum()
    (raw_aux * cfg.moe_aux_loss_coeff * external_scale).backward()
    biased = scores.detach() + bias.detach().cpu().double()
    groups = biased.unflatten(-1, (cfg.n_group, -1))
    group_scores = groups.topk(cfg.moe_router_group_score_topk, dim=-1).values.sum(-1)
    keep = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(
        1, group_scores.topk(cfg.topk_group, dim=-1, sorted=False).indices, True)
    selected = groups.masked_fill(~keep.unsqueeze(-1), float("-inf")).flatten(-2).topk(
        TOP_K, dim=-1, sorted=False).indices
    actual_counts = torch.zeros(EXPERTS, dtype=torch.int64).scatter_add_(
        0, selected.flatten(), torch.ones_like(selected.flatten(), dtype=torch.int64))
    return float(raw_aux.detach()), w.grad.detach().float(), actual_counts


class RouterCarrier(nn.Module):
    """Own the real Xing4Router only; do not allocate or execute any expert weight."""
    def __init__(self, router, observed):
        super().__init__()
        self.gate, self.observed = router, observed

    def forward(self, hidden_states):
        return self.route_only(hidden_states)

    def route_only(self, hidden_states):
        logits, weights, indices = self.gate(hidden_states)
        self.observed["router_input"] = hidden_states.detach()
        self.observed["logits"] = logits.detach()
        self.observed["selected_indices"] = indices.detach()
        b,s,h = hidden_states.shape
        return weights.sum(-1).reshape(b,s,1).expand(b,s,h).contiguous()


def router_only_factory(*, module, mesh, tp_mesh, cp_mesh, ep_mesh):
    """A declared diagnostic factory resolved by the real framework machinery."""
    del mesh, tp_mesh, cp_mesh, ep_mesh

    def compute(module, hidden_states):
        return module.route_only(hidden_states)

    return compute


def error(actual, expected):
    delta = actual.detach().cpu().float() - expected.detach().cpu().float()
    return {"max_abs": float(delta.abs().max()), "l2": float(torch.linalg.vector_norm(delta)),
        "reference_l2": float(torch.linalg.vector_norm(expected.float())),
        "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected.float()).clamp_min(1e-30))}


def execute_case(args, runtime, cfg, case, mesh, x_cpu, weight, bias, out):
    name, ep_size, sp = case
    plan, spec, src, dst, need_tp = build_plan(runtime, cfg, mesh, ep_size, sp)
    working_mesh = mesh[tuple(plan.mesh_dim_names)]
    tp_mesh, dp_rank = mesh["tp"], mesh.get_local_rank("dp")
    tp_group = tp_mesh.get_group()
    receipt = plan_receipt(runtime, plan, spec, src, dst, need_tp, working_mesh)
    full = x_cpu[dp_rank].to(args.device)
    full_ids = torch.arange(SEQUENCE, dtype=torch.int64, device=args.device).view(1, SEQUENCE, 1)
    oracle_aux, oracle_grad, oracle_counts = independent_reference(x_cpu[dp_rank], weight, bias, cfg, args.external_scale)
    records = []
    gradients = {}
    for binding in ("original_1477", "resolved_token_partition_candidate"):
        source = runtime["model"].Xing4_0TopkRouter(cfg).to(args.device)
        router = runtime["model"].Xing4Router(cfg, source, layer_idx=0).to(args.device).train()
        with torch.no_grad():
            router.weight.copy_(weight.to(args.device))
            router.e_score_correction_bias.copy_(bias.to(args.device))
            router._token_counts.zero_()
        router.aux_loss_cp_group = tp_group if binding.endswith("candidate") and need_tp else None
        observed = {}
        carrier = RouterCarrier(router, observed)
        boundary = runtime["PrecompiledBoundary"](spec, working_mesh, plan.mesh_dim_names)
        runtime["_bind_input_indices"](boundary, carrier)
        compute = runtime["_resolve_local_compute_fn"](carrier, spec, working_mesh, plan.mesh_dim_names, expert_mesh=None)
        runtime["_wrap_local_region_forward"](carrier, boundary, spec, working_mesh, plan.mesh_dim_names,
            validate_mode=False, compute_fn=compute)
        # Source view comes from the actual resolved in_src through real HP
        # distribute_tensor. There is no manual chunking based on the SP flag.
        source_view = runtime["distribute_tensor"](full, working_mesh, src).to_local()
        source_ids = runtime["distribute_tensor"](full_ids, working_mesh, src).to_local()
        before_ids = source_ids.flatten().cpu().tolist()
        with runtime["aux_loss_scale_context"]():
            output = carrier(source_view)
            task = output.float().sum() * 0.0
            runtime["bind_aux_loss_scale"](task)
            (task * args.external_scale).backward()
        if args.device == "npu":
            torch.npu.synchronize()
        router_input = observed["router_input"]
        # Position metadata traverses the same real input RedistOp sequence.
        # Verify the actual activation bytes against those positions; this
        # works for actual captures with repeated/identical activation rows.
        id_args, _ = boundary.redistribute_inputs((source_ids,), {}, as_dtensor=False)
        actual_ids = id_args[0].flatten().cpu().tolist()
        expected_rows = x_cpu[dp_rank].index_select(1, torch.tensor(actual_ids, dtype=torch.long))
        if not torch.equal(router_input.detach().cpu(), expected_rows):
            raise RuntimeError("Actual router input bytes do not match positions traversing the same input boundary")
        local_grad = router.weight.grad.detach().clone()
        mean_grad = local_grad.clone()
        dist.all_reduce(mean_grad, group=tp_group)
        mean_grad.div_(TP_SIZE)
        actual_counts = router._token_counts.detach().clone()
        combined_counts = actual_counts.clone()
        dist.all_reduce(combined_counts, group=tp_group)
        if not need_tp:
            # Diagnostic logical count deduplication, not an expert-bias update.
            combined_counts.div_(TP_SIZE, rounding_mode="floor")
        raw_aux = float(router.last_aux_loss.detach().cpu())
        local_aux_ref, local_grad_ref, local_count_ref = independent_reference(
            router_input, weight, bias, cfg, args.external_scale)
        record = {"binding": binding, "router_type": type(router).__name__,
            "actual_router_dimensions": {"experts": router.num_experts, "hidden": router.hidden_dim, "top_k": router.top_k},
            "before_boundary_shape": list(source_view.shape), "before_boundary_token_ids": before_ids,
            "at_router_shape": list(router_input.shape), "at_router_token_ids": actual_ids,
            "token_position_evidence": "position IDs through same actual input RedistOps; actual activation bytes exactly verified against full input at those IDs",
            "at_router_sha256": sha_tensor(router_input), "full_sequence_sha256": sha_tensor(full),
            "actual_statistics_group": dist.get_process_group_ranks(router.aux_loss_cp_group) if router.aux_loss_cp_group is not None else [],
            "statistics_group_reason": "resolved in_dst TP Shard(sequence)" if binding.endswith("candidate") and need_tp else "none",
            "raw_aux": raw_aux, "global_sequence_raw_aux_reference": oracle_aux,
            "raw_aux_global_reference_delta": raw_aux-oracle_aux, "local_view_raw_aux_reference": local_aux_ref,
            "bias_counts_raw": actual_counts.cpu().tolist(), "bias_counts_global_deduplicated": combined_counts.cpu().tolist(),
            "bias_counts_global_reference": oracle_counts.tolist(),
            "bias_counts_local_reference_equal": torch.equal(actual_counts.cpu(), local_count_ref),
            "bias_counts_global_reference_equal": torch.equal(combined_counts.cpu(), oracle_counts),
            "tp_mean_router_gradient_vs_global_reference": error(mean_grad.cpu(), oracle_grad),
            "local_gradient_vs_local_view_reference": error(local_grad.cpu(), local_grad_ref),
            "gradient_reduction_policy": "Explicit diagnostic TP replica mean; no actual FSDP or optimizer is run",
            "gradient_reference": "Independent FP64 sigmoid/sequence formula with explicit coeff and external multiplier; routing top-k is stable by construction",
            "external_scale": args.external_scale, "aux_coefficient": cfg.moe_aux_loss_coeff}
        records.append(record)
        gradients[binding+"_local"] = local_grad.cpu()
        gradients[binding+"_tp_mean"] = mean_grad.cpu()
        del carrier, router, source, source_view, source_ids, output, task, observed, local_grad, mean_grad
    gradients["independent_global_reference"] = oracle_grad
    from safetensors.torch import save_file
    grad_path = out / (name + "-gradients.safetensors")
    save_file(gradients, str(grad_path))
    return {"case": name, "requested_ep": ep_size, "requested_sp": sp, "rank": dist.get_rank(),
        "dp_rank": dp_rank, "tp_rank": tp_mesh.get_local_rank(), "plan": receipt,
        "records": records, "gradient_file": source_identity(grad_path),
        "scope": "Actual planner + actual PrecompiledBoundary + actual forward_rewriter + actual Xing4Router; router-only carrier, no experts, A2A, full applier parameter sharding, FSDP, optimizer or model acceptance"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--external-scale", type=float, default=.5)
    parser.add_argument("--assets", type=Path)
    args = parser.parse_args()
    if not args.execute and args.device != "cpu":
        parser.error("metadata-only mode always uses CPU; --device npu requires explicit --execute")
    if args.dp_size < 1 or not 0 < args.external_scale <= 1:
        parser.error("dp-size must be positive and external-scale must be in (0, 1]")
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    out = args.output / ("rank-"+str(rank))
    out.mkdir(exist_ok=True)
    result = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "execute": args.execute,
        "device": args.device, "training_updates": 0, "status": "preparing", "cases": []}
    temporary = None
    try:
        if args.execute and args.device == "npu":
            import torch_npu
            torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
            torch.npu.matmul.allow_hf32 = False
            result["runtime"] = {"torch":torch.__version__, "torch_npu":torch_npu.__version__,
                "local_rank":int(os.environ["LOCAL_RANK"]), "matmul_allow_hf32":torch.npu.matmul.allow_hf32}
        runtime = load_runtime(args.repo, args.execute)
        result["sources"] = runtime["source_receipts"]
        result["editable"] = runtime["editable"]
        cfg = config(runtime)
        if args.execute:
            if int(os.environ.get("WORLD_SIZE", "0")) != args.dp_size*TP_SIZE:
                raise RuntimeError("torchrun world size must equal dp_size * 2")
            dist.init_process_group(backend="hccl" if args.device == "npu" else "gloo", timeout=datetime.timedelta(seconds=120))
            mesh = runtime["init_device_mesh"](args.device, (args.dp_size, TP_SIZE), mesh_dim_names=("dp", "tp"))
            inputs, weight, bias, provenance = assets(args)
            result["assets"] = {"provenance": provenance, "input_sha256": sha_tensor(inputs),
                "weight_sha256": sha_tensor(weight), "bias_sha256": sha_tensor(bias),
                "input_shape": list(inputs.shape), "weight_shape": list(weight.shape)}
            for case in CASES:
                result["cases"].append(execute_case(args, runtime, cfg, case, mesh, inputs, weight, bias, out))
                write_json(out/"result.json", result)
                dist.barrier()
            result["status"] = "diagnostic_execution_completed_no_acceptance_claim"
        else:
            temporary = tempfile.TemporaryDirectory(prefix="pr1477-router-plan-")
            dist.init_process_group("gloo", init_method="file://"+str(Path(temporary.name)/"pg"), rank=0, world_size=1)
            mesh = runtime["init_device_mesh"]("cpu", (args.dp_size, TP_SIZE), mesh_dim_names=("dp", "tp"),
                rank_list=tuple(range(args.dp_size*TP_SIZE)), init_backend=False)
            for name, ep_size, sp in CASES:
                plan, spec, src, dst, need_tp = build_plan(runtime, cfg, mesh, ep_size, sp)
                result["cases"].append({"case": name, "requested_ep": ep_size, "requested_sp": sp,
                    "plan": plan_receipt(runtime, plan, spec, src, dst, need_tp, mesh[tuple(plan.mesh_dim_names)])})
            result["status"] = "metadata_plan_only_no_boundary_execution"
        result["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(out/"result.json", result)
        print(json.dumps({"rank": rank, "status": result["status"], "output": str(out/"result.json")}, ensure_ascii=False))
    except BaseException as exc:
        result.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        write_json(out/"result.json", result)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
