"""Read-only semantic probes against explicitly selected PR1477 source files.

No hyper_parallel package is imported: the existing venv's editable install points
to another checkout. Complete leaf functions/classes are loaded from their stated
paths, with only environment/distributed plumbing stubbed for the single-CPU case.
"""

from __future__ import annotations

import argparse
import ast
import copy
from contextlib import nullcontext
from dataclasses import dataclass, fields
from functools import cached_property
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Optional

import torch
from torch import nn

# Loading a source file must not leave import caches in the audited checkout.
sys.dont_write_bytecode = True


def source_identity(path):
    payload = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}


def extract(path, selectors, namespace):
    tree = ast.parse(path.read_text())
    selected = []
    for selector in selectors:
        if isinstance(selector, tuple):
            owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == selector[0])
            node = next(node for node in owner.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name == selector[1])
        else:
            node = next(node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                        and node.name == selector)
        selected.append(copy.deepcopy(node))
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def load_leaf(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def probe_aux(repo):
    # The exact PR1477 auxiliary autograd function needs only these two DTensor
    # symbols. No tensors here are DTensors and no distributed collective runs.
    dtensor_stub = ModuleType("hyper_parallel.core.dtensor.dtensor")
    dtensor_stub.DTensor = type("UnusedDTensor", (), {})
    dtensor_stub.SkipDTensorDispatch = nullcontext
    sys.modules[dtensor_stub.__name__] = dtensor_stub
    aux_path = repo / "hyper_parallel/components/functional/aux_loss.py"
    aux = load_leaf(aux_path, "pr1477_exact_aux_loss")
    metrics_path = repo / "hyper_parallel/trainer/runtime/metrics.py"
    metrics = extract(metrics_path, ["_reduce_weighted_loss", "mean_global_loss"], {
        "torch": torch, "dist": torch.distributed, "Union": Any,
        "_is_tracing_or_compiling": lambda: False,
        "all_reduce": lambda value, op, group: value,
    })
    base_path = repo / "hyper_parallel/trainer/base.py"
    base = extract(base_path, [("BaseTrainer", "postforward")], {
        "torch": torch, "ModelOutput": Any, "Optional": Optional, "Dict": Dict, "Any": Any,
        "bind_aux_loss_scale": aux.bind_aux_loss_scale,
        "mean_global_loss": metrics["mean_global_loss"],
    })
    actual_postforward = base["postforward"]

    def bind_all_foundation(losses):
        for name, loss in losses.items():
            if name.split("_loss", maxsplit=1)[0] == "foundation":
                aux.bind_aux_loss_scale(loss)

    mesh = SimpleNamespace(dp_size=1, cp_size=1, dp_cp_mesh=None, sequence_parallel=False)
    trainer = SimpleNamespace(loss_fn=lambda *, model_output, labels: model_output.loss, mesh=mesh)
    cases = []
    for ga in (1, 2, 4):
        for order in ("lm_then_mtp", "mtp_then_lm"):
            for binding in ("actual_pr1477", "all_foundation_candidate"):
                h = nn.Parameter(torch.tensor(2.0))
                a_back = nn.Parameter(torch.tensor(3.0))
                a_mtp = nn.Parameter(torch.tensor(5.0))
                observed = []
                with aux.aux_loss_scale_context():
                    backbone_output = aux.aux_loss_auto_scale(h * 2.0, a_back * 0.0005)
                    lm = backbone_output.square()
                    mtp_output = aux.aux_loss_auto_scale(backbone_output * 3.0, a_mtp * 0.0005)
                    mtp = 0.1 * mtp_output.square()
                    ordered = [("foundation_loss", lm), ("foundation_loss/mtp", mtp)]
                    if order == "mtp_then_lm":
                        ordered.reverse()
                    losses = dict(ordered)
                    for name, loss in losses.items():
                        loss.register_hook(lambda grad, name=name: observed.append({
                            "event": name, "incoming_multiplier": float(grad)}))
                    a_mtp.register_hook(lambda grad: observed.append({
                        "event": "mtp_aux", "incoming_gradient": float(grad)}))
                    if binding == "actual_pr1477":
                        result, logged = actual_postforward(
                            trainer, SimpleNamespace(loss=losses), None,
                            current_token_counts={"foundation_tokens": torch.tensor(93)},
                            step_token_counts={"foundation_tokens": torch.tensor(93 * ga)})
                    else:
                        bind_all_foundation(losses)
                        logged = metrics["mean_global_loss"](
                            losses, {"foundation_tokens": torch.tensor(93)},
                            {"foundation_tokens": torch.tensor(93 * ga)}, mesh)
                        result = torch.stack(list(logged.values())).sum()
                    result.backward()
                expected = 0.0005 / ga
                cases.append({
                    "gradient_accumulation": ga, "loss_mapping_order": order, "binding": binding,
                    "backbone_aux_gradient": float(a_back.grad), "mtp_aux_gradient": float(a_mtp.grad),
                    "expected_independent_aux_gradient": expected,
                    "backbone_aux_ratio_to_expected": float(a_back.grad) / expected,
                    "mtp_aux_ratio_to_expected": float(a_mtp.grad) / expected,
                    "main_parameter_gradient": float(h.grad), "backward_events": observed,
                    "logged_lm": float(logged["foundation_loss"].detach()),
                    "logged_weighted_mtp": float(logged["foundation_loss/mtp"].detach()),
                })
    return {"sources": [source_identity(p) for p in (aux_path, metrics_path, base_path)],
            "single_rank_plumbing": "all_reduce identity; no DTensors, FSDP, NPU or distributed claim",
            "objective": "Independent backbone/MTP aux each get external 1/GA, never internal MTP factor 0.1",
            "cases": cases}


def probe_mhc(repo):
    post_path = repo / "hyper_parallel/components/functional/mhc_post.py"
    post = load_leaf(post_path, "pr1477_exact_mhc_post")
    model_path = repo / "hyper_parallel/models/xing_4/modeling_xing4_0.py"
    model = extract(model_path, [("Xing4DecoderLayer", "_post")], {"torch": torch, "mhc_post": post.mhc_post})
    # Complete function includes @staticmethod; resolve the descriptor explicitly.
    actual_post = model["_post"].__func__
    torch.manual_seed(1967)
    values = [torch.randn(shape, dtype=torch.float64) for shape in (
        (1, 2, 4), (1, 2, 3, 4), (1, 2, 3), (1, 2, 3, 3))]
    cotangent = torch.randn((1, 2, 3, 4), dtype=torch.float64)

    def evaluate(kind):
        x, residual, hp, mix = [v.clone().requires_grad_() for v in values]
        if kind == "actual_pr1477":
            output = actual_post(x, residual, hp, mix)
        elif kind == "destination_by_source_reference":
            output = hp.unsqueeze(-1) * x.unsqueeze(2) + torch.einsum("bsij,bsjh->bsih", mix, residual)
        else:
            output = post.mhc_post(x, residual.flatten(2), hp, mix, 3).unflatten(-1, (3, 4))
        gradients = torch.autograd.grad((output * cotangent).sum(), (x, residual, hp, mix))
        return output, gradients

    candidate, cg = evaluate("actual_pr1477")
    reference, rg = evaluate("destination_by_source_reference")
    unadapted, ug = evaluate("unadapted_shared_post")
    return {"sources": [source_identity(post_path), source_identity(model_path)],
            "scope": "CPU post boundary only; NPU pre and post kernels require independent hardware evidence",
            "matrix_asymmetry_max": float((values[3] - values[3].transpose(-1, -2)).abs().max()),
            "candidate_vs_Hx": {"output_max_abs": float((candidate - reference).abs().max().detach()),
                "gradient_max_abs": [float((a-b).abs().max()) for a,b in zip(cg, rg)]},
            "without_transpose_vs_Hx": {"output_max_abs": float((unadapted - reference).abs().max().detach()),
                "gradient_max_abs": [float((a-b).abs().max()) for a,b in zip(ug, rg)]}}


@dataclass
class NSInputTransform:
    tensors: list
    restore: Any


def probe_muon(repo):
    path = repo / "hyper_parallel/models/xing_4/adapter/optim/muon_layout.py"
    loaded = extract(path, ["Xing4MuonLayout"], {"dataclass": dataclass, "fields": fields,
        "cached_property": cached_property, "Any": Any, "torch": torch, "NSInputTransform": NSInputTransform})
    layout = loaded["Xing4MuonLayout"](q_lora_rank=2, kv_lora_rank=3, qk_nope_head_dim=2,
        qk_rope_head_dim=2, v_head_dim=3, num_attention_heads=2, hidden_size=4,
        intermediate_size=6, moe_intermediate_size=3, n_shared_experts=1)
    entries = [
        ("model.layers.1.self_attn.linear_qkv.weight", (7, 4)),
        ("mtp.layers.0.mtp_block.self_attn.linear_qkv.weight", (7, 4)),
        ("model.layers.0.mlp.linear_fc1.weight", (12, 4)),
        ("model.layers.1.mlp.shared_experts.linear_fc1.weight", (6, 4)),
        ("model.layers.1.mlp.experts.gate_up_proj", (3, 4, 6)),
        ("mtp.layers.0.mtp_block.mlp.experts.gate_up_proj", (3, 4, 6)),
        ("model.layers.1.self_attn.q_b_proj.weight", (8, 2)),
        ("model.layers.1.self_attn.kv_b_proj.weight", (10, 3)),
    ]
    cases = []
    for name, shape in entries:
        tensor = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
        transform = layout.ns_transform_fn(name, tensor)
        if transform is None:
            views = layout.reshape_fn(name, tensor)
            restore_error = None
            sharing = all(v.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr() for v in views)
        else:
            views = transform.tensors
            output = torch.full_like(tensor, float("nan"))
            transform.restore([v.clone() for v in views], output)
            restore_error = float((output-tensor).abs().max())
            sharing = None
        cases.append({"name": name, "shape": shape, "logical_shapes": [list(t.shape) for t in views],
            "reshape_shares_storage": sharing, "identity_restore_max_abs": restore_error,
            "logical_matrix_values": [v.tolist() for v in views]})
    return {"sources": [source_identity(path)],
        "scope": "Logical boundary and reversible layout probe, not a Newton-Schulz/optimizer numerical parity test",
        "cases": cases}


def probe_legacy_weighting(repo):
    path = repo / "hyper_parallel/trainer/runtime/metrics.py"
    metrics = extract(path, ["_reduce_weighted_loss", "mean_global_loss"], {
        "torch": torch, "dist": torch.distributed, "Union": Any,
        "_is_tracing_or_compiling": lambda: False,
        "all_reduce": lambda value, op, group: value,
    })
    mesh = SimpleNamespace(dp_size=1, cp_size=1, dp_cp_mesh=None, sequence_parallel=False)
    theta = nn.Parameter(torch.tensor(1.0))
    total = torch.zeros(())
    # TextTrainer sets step_token_counts anew as the CURRENT microbatch's
    # count * GA; it does not have the actual sum of all step counts.
    for count, slope in ((93, 1.0), (740, 9.0)):
        total = total + metrics["mean_global_loss"]({"foundation_loss": theta * slope},
            {"foundation_tokens": torch.tensor(count)},
            {"foundation_tokens": torch.tensor(count * 2)}, mesh)["foundation_loss"]
    total.backward()
    return {"sources": [source_identity(path),
                source_identity(repo / "hyper_parallel/trainer/text_trainer.py")],
            "microbatch_counts": [93, 740], "microbatch_mean_loss_slopes": [1.0, 9.0],
            "actual_reported_loss": float(total.detach()), "actual_gradient": float(theta.grad),
            "microbatch_equal_mean_reference": 5.0,
            "whole_step_token_reference": (93.0 + 740.0 * 9.0) / (93.0 + 740.0)}


def probe_router_domains(repo):
    path = repo / "hyper_parallel/models/xing_4/modeling_xing4_0.py"
    aux = sys.modules["pr1477_exact_aux_loss"]
    loaded = extract(path, [("Xing4Router", "_attach_balance_loss")], {
        "torch": torch, "dist": torch.distributed, "aux_loss_auto_scale": aux.aux_loss_auto_scale})
    attach = loaded["_attach_balance_loss"]
    scores = torch.tensor([[.9, .1], [.8, .2], [.7, .3], [.1, .9]])

    def evaluate(raw_scores, batch):
        tensor = raw_scores.clone().requires_grad_()
        indices = tensor.topk(1, dim=-1, sorted=False).indices
        router = SimpleNamespace(force_expert_balance=False, num_experts=2, top_k=1,
            aux_loss_cp_group=None, aux_loss_coeff=0.0005)
        routed = attach(router, tensor.gather(1, indices), tensor, indices, batch)
        # Zero task gradient isolates the independent auxiliary autograd edge.
        (routed.sum() * 0.0).backward()
        return {"raw_aux": float(router.last_aux_loss),
            "normalized_aux_gradient": (tensor.grad / router.aux_loss_coeff).tolist()}

    full = evaluate(scores, 1)
    shards = [evaluate(chunk, 1) for chunk in scores.chunk(2)]
    dp = evaluate(scores, 2)
    return {"sources": [source_identity(path), source_identity(
                repo / "hyper_parallel/models/xing_4/adapter/distributed/context_parallel.py")],
            "scope": "Real router auxiliary formula on CPU; CP collective execution is not simulated",
            "one_global_sequence": full,
            "two_tp_replicas": [full["raw_aux"], full["raw_aux"]],
            "two_sequence_shards_without_tp_pooling": shards,
            "mean_of_separate_sequence_shards": sum(x["raw_aux"] for x in shards) / 2,
            "two_independent_dp_sequences": dp,
            "interpretation": "TP-replicated full sequences need no TP pooling. TP-sharded sequence chunks require pooling; DP sequences remain separate sequence objectives."}


def probe_config_options(repo):
    model_path = repo / "hyper_parallel/models/xing_4/modeling_xing4_0.py"
    config_path = repo / "hyper_parallel/models/xing_4/transformer/configuration_xing4_0.py"
    config_module = load_leaf(config_path, "pr1477_exact_configuration")
    tree = ast.parse(model_path.read_text())
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_REQUIRED_OPTIONS" for t in n.targets))
    required = ast.literal_eval(assignment.value)
    ns = extract(model_path, ["create_config"], {"Xing4_0Config": config_module.Xing4_0Config,
        "_REQUIRED_OPTIONS": required, "Any": Any})
    requested = {"swiglu_limit": 1.0, "swiglu_compute_dtype": "float32"}
    config = ns["create_config"](requested)
    return {"sources": [source_identity(model_path), source_identity(config_path)],
        "requested": requested, "accepted": True,
        "retained_config_fields": {k:getattr(config,k) for k in requested},
        "note": "The actual create_config accepts both fields; current model/GroupedExperts have no policy consumer. This is outside the MF baseline where both fields are None."}


def probe_fp32_buffer_validation(repo):
    helper_path = repo / "hyper_parallel/core/multicore/modules/_precision.py"
    helper = load_leaf(helper_path, "pr1477_exact_precision_helper")
    builder_path = repo / "hyper_parallel/models/_transformers/model_builder.py"
    ns = extract(builder_path, ["_model_fp32_parameter_ids", "_validate_model_state_dtype"], {
        "nn": nn, "torch": torch, "get_model_external_state": lambda model: None})

    class BufferModule(nn.Module):
        fp32_buffer_names = ("e_score_correction_bias",)

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))
            self.register_buffer("e_score_correction_bias", torch.tensor([0.1234567]))

        def _apply(self, fn, recurse=True):
            return super()._apply(helper.preserve_fp32_parameters(self, fn, self.fp32_buffer_names), recurse=recurse)

    model = BufferModule().to(dtype=torch.bfloat16)
    error = None
    try:
        ns["_validate_model_state_dtype"](model, torch.bfloat16)
    except RuntimeError as exc:
        error = str(exc)
    model.to(dtype=torch.float32)
    ns["_validate_model_state_dtype"](model, torch.float32)
    return {"sources": [source_identity(helper_path), source_identity(builder_path)],
        "bfloat16_target_error": error, "protected_buffer_dtype": str(model.e_score_correction_bias.dtype),
        "float32_target_passed": True,
        "scope": "Demonstrates BF16 initialization/declared-FP32-buffer validation boundary; required MF continuation uses FP32 initialization and is unaffected."}


def probe_hf_init_guard(repo):
    from transformers.initialization import guard_torch_init_functions
    import transformers.initialization as initialization
    path = repo / "hyper_parallel/models/xing_4/modeling_xing4_0.py"

    class HyperConnection(nn.Module):
        def __init__(self):
            super().__init__()
            self.hc_fn = nn.Parameter(torch.full((3, 4), 0.12345))
            self.hc_scale = nn.Parameter(torch.full((3,), 0.23456))
            self.hc_base = nn.Parameter(torch.full((3,), 0.34567))

    ns = extract(path, ["_init_training_weights"], {"nn": nn, "torch": torch,
        "Xing4_0HyperConnection": HyperConnection, "Xing4_0PreTrainedModel": type("UnusedOwner", (), {}),
        "RMSNorm": type("UnusedNorm", (), {}), "GroupedExperts": type("UnusedExperts", (), {}),
        "Xing4Router": type("UnusedRouter", (), {})})
    model = HyperConnection()
    before = {k:v.detach().clone() for k,v in model.named_parameters()}
    for value in model.parameters():
        value._is_hf_initialized = True
    with guard_torch_init_functions():
        ns["_init_training_weights"](SimpleNamespace(config=SimpleNamespace(mhc_init_gating_factor=.2)), model)
    after = {k:v.detach().clone() for k,v in model.named_parameters()}
    direct = torch.tensor([.34567])
    direct._is_hf_initialized = True
    direct.zero_()
    return {"sources": [source_identity(path), source_identity(Path(initialization.__file__))],
        "loaded_hc_parameter_differences": {k:float((before[k]-after[k]).abs().max()) for k in before},
        "direct_tensor_zero_bypasses_guard": float(direct) == 0.0,
        "scope": "Exact training initializer on HC-shaped parameters inside actual installed Transformers guard; does not claim official HF full-load support."}


def probe_swap_failfast(repo):
    path = repo / "hyper_parallel/core/optimizer/swap_optimizer_base.py"
    ns = extract(path, [("SwapOptimizer", "__init__")], {"Any": Any, "AdapterFactory": Any})
    parameter = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=.1)
    optimizer.register_step_post_hook(lambda optimizer, args, kwargs: None)
    error = None
    try:
        ns["__init__"](SimpleNamespace(), optimizer, SimpleNamespace(), None)
    except ValueError as exc:
        error = str(exc)
    return {"sources": [source_identity(path)], "preexisting_leaf_post_hook_error": error,
        "scope": "Exact wrapper check rejects before constructing swap runtime or offloading; no swap execution claim."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = {"torch_version": torch.__version__, "python": sys.executable,
        "explicit_repo": str(args.repo.resolve()), "package_import_policy": "No hyper_parallel package imported",
        "aux": probe_aux(args.repo), "mhc": probe_mhc(args.repo), "muon": probe_muon(args.repo),
        "legacy_weighting": probe_legacy_weighting(args.repo), "router_domains": probe_router_domains(args.repo),
        "config_options": probe_config_options(args.repo), "fp32_buffer_validation": probe_fp32_buffer_validation(args.repo),
        "hf_init_guard": probe_hf_init_guard(args.repo), "swap_failfast": probe_swap_failfast(args.repo)}
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps({"output": str(args.output), "aux_cases": len(output["aux"]["cases"]),
        "aux_summary": [{k:v for k,v in case.items() if k in ("gradient_accumulation", "loss_mapping_order", "binding",
        "mtp_aux_ratio_to_expected", "backbone_aux_ratio_to_expected")} for case in output["aux"]["cases"]],
        "mhc": {k:v for k,v in output["mhc"].items() if k not in ("sources",)},
        "muon_cases": len(output["muon"]["cases"]), "legacy_weighting": output["legacy_weighting"],
        "router_domain_summary": {"global_sequence_aux": output["router_domains"]["one_global_sequence"]["raw_aux"],
            "unpooled_shards_mean_aux": output["router_domains"]["mean_of_separate_sequence_shards"],
            "independent_dp_sequences_aux": output["router_domains"]["two_independent_dp_sequences"]["raw_aux"]},
        "config_options": output["config_options"], "fp32_buffer_validation": output["fp32_buffer_validation"],
        "hf_init_guard": output["hf_init_guard"], "swap_failfast": output["swap_failfast"]},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
