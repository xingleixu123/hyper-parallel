"""Prepare the existing router fixture and an independent FP64 GA4 target."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'router-boundary-probe'
torch.set_num_threads(2)
spec = importlib.util.spec_from_file_location('old_router_fixture', OLD / 'router_boundary_worker.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
inputs, weight, bias, provenance = common.assets(SimpleNamespace(dp_size=2, assets=None))
positions = torch.arange(4096).reshape(1, -1)
cp_counts = ((93, 740), (111, 311), (222, 500), (444, 1234))
local_positions = torch.arange(2048).reshape(1, -1)
masks = torch.stack([torch.cat([
    ((local_positions * 1237 + micro * 173 + cp * 317) % 2048 < cp_counts[micro][cp]).long()
    for cp in range(2)], dim=1) for micro in range(4)])
labels = torch.where(masks.bool(), torch.zeros_like(masks), torch.full_like(masks, -100))
counts = masks.sum((1, 2)).tolist()
denominator = sum(counts)
w = weight.double().requires_grad_()
raw = []
for sample in range(2):
    scores = F.linear(inputs[sample].double().reshape(-1, 3584), w).sigmoid()
    indices = scores.topk(4, dim=-1, sorted=False).indices
    assignments = torch.zeros(64, dtype=torch.float64).scatter_add_(
        0, indices.reshape(-1), torch.ones(indices.numel(), dtype=torch.float64))
    probabilities = scores / (scores.sum(-1, keepdim=True) + 1e-20)
    raw.append(64 * (probabilities.mean(0) * (assignments / indices.numel())).sum())
objective = .0005 * sum(counts[micro] / denominator * raw[micro % 2] for micro in range(4))
objective.backward()
asset = HERE / 'aux-native-fixture.safetensors'
save_file({'hidden_states': inputs, 'router_weight': weight, 'router_bias': bias,
           'shift_labels': labels, 'loss_mask': masks, 'input_ids': positions,
           'independent_fp64_global_gradient': w.grad.detach()}, str(asset))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
reference = {
    'status': 'CPU_independent_FP64_fixture_not_device_acceptance',
    'assets_sha256': sha(asset), 'assets_bytes': asset.stat().st_size,
    'source_manifest_sha256': '4967881062f63d8144bd8c4214b60c16f9ca7495b7f241a6701d04a630e08d34',
    'old_fixture_helper_sha256': sha(OLD / 'router_boundary_worker.py'),
    'input_sha256': common.sha_tensor(inputs), 'weight_sha256': common.sha_tensor(weight),
    'bias_sha256': common.sha_tensor(bias), 'provenance': provenance,
    'materialized_by_torch': torch.__version__,
    'old_12case_fixture_recipe_reused_not_old_materialized_bytes': True,
    'not_MF_training_samples': True,
    'dimensions': {'experts': 64, 'hidden': 3584, 'top_k': 4, 'sequence': 4096,
                   'DP': 1, 'CP': 2, 'TP': 2, 'GA': 4, 'safe_padding': 64},
    'micro_sample_indices': [0, 1, 0, 1],
    'foundation_counts_global': counts, 'step_foundation_denominator': denominator,
    'foundation_counts_CP': [[int(mask[:, :2048].sum()), int(mask[:, 2048:].sum())] for mask in masks],
    'mask_scope': 'Stored actual nonprefix masks with deliberately unequal CP counts; no valid_count-to-prefix inference',
    'raw_aux_fp64_by_fixture_sample': [float(value.detach()) for value in raw],
    'aux_coefficient': .0005, 'independent_weighted_objective': float(objective.detach()),
    'reference_gradient_dtype': str(w.grad.dtype),
    'reference_gradient_l2': float(w.grad.norm()),
    'reference_gradient_sha256': common.sha_tensor(w.grad.detach()),
    'reference_equation': 'coeff * sum_micro(n_global_micro / N_step * E*sum_e(f_e*mean_token(P_e)))',
    'hypotheses': {'LP_false': '1x target if native source SUM+DP*CP*TP division matches the declared replica objective',
                   'LP_true': 'Compare both 1x and 2x; hypothesis2x concerns independent AUX injection, not main LP CE'},
    'predeclared_reference_relative_l2_tolerance': 5e-4,
    'scope': 'No HP balance helper; direct FP64 sigmoid, unbiased topk, per-sequence counts/probabilities and autograd. Router counts all real tokens; foundation masks only set Trainer weights.',
}
(HERE / 'aux-native-reference.json').write_text(json.dumps(reference, indent=2) + '\n')
print(json.dumps({'fixture_sha256': sha(asset), 'fixture_bytes': asset.stat().st_size,
                  'reference_sha256': sha(HERE / 'aux-native-reference.json'), 'counts': counts,
                  'CP_counts': reference['foundation_counts_CP'], 'grad_l2': float(w.grad.norm())}))
