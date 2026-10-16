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

"""Test native reduction scheduling on the captured MF input; this is not training acceptance."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch_npu

from mindspore272_stat_hash import native_dump_sha1
from muon_q_norm_probe import half_tree

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
EPSILON = 1.0011717677116394e-7


def digest(path: Path) -> str:
    """Hash bounded source or metadata files."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(value: torch.Tensor) -> dict:
    """Hash all values with native MindSpore statistic semantics."""
    value = value.detach().contiguous().cpu()
    native = value.view(torch.uint16).numpy() if value.dtype == torch.bfloat16 else value.numpy()
    return {'shape': list(value.shape), 'dtype': str(value.dtype), 'native_dump_sha1': native_dump_sha1(native)}


def load_verified(directory: Path, row: dict) -> np.ndarray:
    """Use the exact selected actual training arrays."""
    value = np.load(directory / row['file'])
    if value.dtype != np.float32 or value.shape != (5120, 3072):
        raise ValueError('Unexpected actual fixture layout')
    if hashlib.sha256(value.tobytes()).hexdigest() != row['sha256']:
        raise ValueError('Actual fixture changed')
    return value


def main() -> None:
    """Require native update fidelity, then separate reduction effects on one actual input."""
    proof = json.loads((ROOT / 'evidence/frontier396-mf-builtin-stat.json').read_text())
    if not proof['complete'] or not all(proof['state_observer_fidelity'].values()):
        raise ValueError('MF observation fidelity is incomplete')
    if not all(row['exact_actual_training'] for row in proof['native_state_fidelity'].values()):
        raise ValueError('Native update did not reproduce training')
    if (ROOT / 'runs/frontier396-mf-builtin-stat.exit').read_text().strip() != '0':
        raise ValueError('Native replay is not complete')
    report_path = ROOT / 'evidence/frontier396-normalization-fixture.json'
    if report_path.exists():
        raise ValueError('Fixture report already exists')
    csv_path = STATE / 'frontier396-mf-builtin-stat/rank_0/native_muon396/0/0/statistic.csv'
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    native = {(row['Op Name'], row['IO'], int(row['Slot'])): row for row in rows}
    reduction = 'Default/GraphKernel_Cast_Mul_ReduceSum_split-op0'
    normalization = 'Default/GraphKernel_Sqrt_Add_Div_Cast_split-op0'
    gradient_meta = json.loads((ROOT / 'runs/mf-frontier396-state/FRONTIER_GRADIENTS-396.json').read_text())
    state_meta = json.loads((ROOT / 'runs/mf-frontier396-state/FRONTIER_STATE-396.json').read_text())
    gradient = load_verified(STATE / 'mf-frontier396-state/gradients', gradient_meta[0])
    momentum = load_verified(STATE / 'mf-frontier396-state/state-396',
                             next(row for row in state_meta['records'] if row['kind'] == 'muon_m'))
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    current = (torch.from_numpy(gradient).npu() + torch.from_numpy(momentum).npu() * .95).bfloat16().mT.float()
    original = describe(current)
    if original['native_dump_sha1'] != native[reduction, 'output', 0]['SHA1']:
        raise ValueError('Reconstructed Nesterov input is not identical to the native graph input')
    tile_count = (current.numel() + 9775) // 9776
    if tile_count != 1609:
        raise ValueError('Unexpected native reduction tile count')
    padded = current.new_zeros(tile_count * 9776)
    padded[:current.numel()] = current.reshape(-1).square()
    lanes = half_tree(padded.reshape(tile_count, 611, 16).transpose(-1, -2))
    while lanes.shape[-1] > 1:
        lanes = lanes[..., ::2] + lanes[..., 1::2]
    partials = lanes[..., 0].cpu().numpy()
    accumulators = np.zeros(40, dtype=np.float32)
    for index, partial in enumerate(partials):
        core = index % 40
        accumulators[core] = np.float32(accumulators[core] + partial)
    total = np.float32(0)
    for partial in accumulators:
        total = np.float32(total + partial)
    scheduled_norm = torch.tensor(total, device=current.device).sqrt()
    scheduled = (current / (scheduled_norm + EPSILON)).bfloat16()
    usual_norm = current.norm()
    usual = (current / (usual_norm + EPSILON)).bfloat16()
    raw = current.cpu().numpy()
    norm64 = float(np.sqrt(np.sum(np.square(raw, dtype=np.float64))))
    ideal64 = raw.astype(np.float64) / (norm64 + EPSILON)
    ideal = torch.from_numpy(ideal64.astype(np.float32)).bfloat16()
    scheduled_cpu, usual_cpu = scheduled.cpu(), usual.cpu()
    scheduled_error = np.abs(scheduled_cpu.float().numpy().astype(np.float64) - ideal64)
    usual_error = np.abs(usual_cpu.float().numpy().astype(np.float64) - ideal64)
    report = {
        'complete': True, 'diagnostic_only': True, 'fixture': 'MF actual396 postclip gradient and updated momentum',
        'independent_hp_training_control_pending': True, 'source_sha256': digest(Path(__file__)),
        'native_replay_proof_sha256': digest(ROOT / 'evidence/frontier396-mf-builtin-stat.json'),
        'native_statistics_sha256': digest(csv_path), 'input_exact_native': True, 'input': original,
        'schedule': {'tile_elements': 9776, 'tile_rows': 611, 'lanes': 16,
                     'tiles': 1609, 'tail_elements': 8832, 'vector_cores': 40},
        'scheduled_square_sum': float(total),
        'scheduled_square_sum_exact_native': hashlib.sha1(total.tobytes()).hexdigest()
            == native[reduction, 'output', 1]['SHA1'],
        'scheduled_norm': float(scheduled_norm.cpu()), 'usual_norm': float(usual_norm.cpu()), 'norm_fp64': norm64,
        'scheduled_normalized': describe(scheduled), 'usual_normalized': describe(usual),
        'scheduled_normalized_exact_native': describe(scheduled)['native_dump_sha1']
            == native[normalization, 'output', 0]['SHA1'],
        'usual_normalized_exact_native': describe(usual)['native_dump_sha1']
            == native[normalization, 'output', 0]['SHA1'],
        'normalized_different_elements': int((scheduled_cpu != usual_cpu).sum()),
        'scheduled_different_from_fp64_then_fp32_then_bf16_reference': int((scheduled_cpu != ideal).sum()),
        'usual_different_from_fp64_then_fp32_then_bf16_reference': int((usual_cpu != ideal).sum()),
        'scheduled_closer_to_fp64_normalized': int(np.count_nonzero(scheduled_error < usual_error)),
        'usual_closer_to_fp64_normalized': int(np.count_nonzero(usual_error < scheduled_error)),
        'scope': 'Isolated normalization fixture only; no claim about HP actual full update or full200 acceptance.',
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
