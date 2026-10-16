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

"""Validate the proposed standalone diagnostic against actual native training evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import torch_npu

import compare_shared_down_normalization as proposed
from hp_shared_down396_factor import normalized_shared_down
from mindspore272_stat_hash import native_dump_sha1

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')


def main():
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    origin = STATE / 'hp-jte213-expertnorm328-state396-r2'
    gradient = torch.from_numpy(np.load(origin / 'gradient-396/main1_shared_down-postclip.npy')).npu()
    momentum = torch.from_numpy(np.load(origin / 'state-396/main1_shared_down-muon_m.npy')).npu()
    value = (gradient + momentum * .95).bfloat16().float()
    original = value.clone()
    pre_hash = native_dump_sha1(value.mT.contiguous().cpu().numpy())
    if pre_hash != '4ff79a33e5b1cbb3ef83fad0c3ba7b55b4c4eb56':
        raise ValueError('Actual HP input does not match native MF')
    input_path = STATE / 'frontier396-pr-normalization-input.npy'
    output_path = ROOT / 'evidence/frontier396-pr-diagnostic-output.json'
    if input_path.exists() or output_path.exists():
        raise ValueError('Use fresh validation outputs')
    np.save(input_path, value.cpu().numpy())
    expected, scalar = normalized_shared_down(value)
    actual, actual_norm, square_sum = proposed._archived_normalization(value)
    checks = {
        'complete_input_native_hash': pre_hash == '4ff79a33e5b1cbb3ef83fad0c3ba7b55b4c4eb56',
        'candidate_exact_executed_training_factor': torch.equal(actual, expected),
        'candidate_exact_mf_native_hash': native_dump_sha1(actual.cpu().view(torch.uint16).numpy())
            == 'a4ceaba50f6f60dc871d43d10fe2e8710b5bebfd',
        'square_sum_exact': square_sum == scalar['square_sum'],
        'norm_exact': float(actual_norm.cpu()) == scalar['norm'],
        'input_unchanged': torch.equal(value, original),
    }
    script = Path(proposed.__file__)
    result = subprocess.run([sys.executable, str(script), '--input', str(input_path), '--output', str(output_path)],
                            capture_output=True, text=True, check=True)
    report = json.loads(output_path.read_text())
    fixture = json.loads((ROOT / 'evidence/frontier396-normalization-fixture.json').read_text())
    checks.update({
        'cli_exit_zero': result.returncode == 0,
        'cli_source_sha': report['source_sha256'] == hashlib.sha256(script.read_bytes()).hexdigest(),
        'cli_candidate_hash': report['archived_bf16_sha256'] == hashlib.sha256(
            expected.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest(),
        'cli_actual_norms': report['ordinary_norm'] == fixture['usual_norm']
            and report['archived_norm'] == fixture['scheduled_norm'],
        'fp64_same_actual_reference': report['precision']['norm_fp64'] == fixture['norm_fp64'],
        'bf16_difference_count': report['precision']['bf16_different_elements'] == 92955,
        'fp64_direct_distance': report['precision']['ordinary_closer_to_fp64'] == 92955
            and report['precision']['archived_closer_to_fp64'] == 0,
    })
    for label, invalid in [('shape', np.zeros((1, 1), dtype=np.float32)),
                           ('dtype', np.zeros((5120, 3072), dtype=np.int8))]:
        try:
            proposed.compare_normalization(invalid)
        except ValueError:
            checks['reject_' + label] = True
        else:
            checks['reject_' + label] = False
    duplicate = subprocess.run([sys.executable, str(script), '--input', str(input_path), '--output', str(output_path)],
                               capture_output=True, text=True)
    checks['existing_output_not_overwritten'] = duplicate.returncode != 0 and json.loads(output_path.read_text()) == report
    if not all(checks.values()):
        raise ValueError(checks)
    cleanup = {'path': str(input_path), 'bytes': input_path.stat().st_size,
               'sha256': hashlib.sha256(input_path.read_bytes()).hexdigest()}
    if input_path.is_symlink() or input_path.resolve().parent != STATE.resolve():
        raise ValueError('Unsafe diagnostic input cleanup')
    input_path.unlink()
    cleanup['deleted'] = not input_path.exists()
    result = {'complete': True, 'checks': checks, 'source_sha256': report['source_sha256'],
              'validation_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'temporary_input_cleanup': cleanup, 'result': report}
    (ROOT / 'evidence/frontier396-pr-diagnostic-validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
