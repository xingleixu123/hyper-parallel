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

"""Test whether the four observed gradient rows explain every226 head dinput difference."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch_npu

from hyper_parallel.trainer.runtime.random import enable_full_determinism
from compare_preclip212 import compare_arrays, bf16_bits

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
RUN = 'hp-jtf637-qsqrt222-head226-r3'
OUTPUT = STATE / 'head226-gradient-cause-replay'


def load_bits(record):
    """Reject any changed full native operand before NPU replay."""
    array = np.load(STATE / RUN / 'head' / record['file'], mmap_mode='r')
    if list(array.shape) != record['shape'] or array.dtype != np.uint16:
        raise ValueError('Unexpected BF16 bit storage')
    if hashlib.sha256(array.tobytes()).hexdigest() != record['sha256']:
        raise ValueError('Saved operand SHA differs')
    return array


def to_fp32(bits):
    """Expand BF16 storage without numerical conversion."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


def main():
    """Keep shapes, determinism and input bits fixed; require original mm fidelity."""
    audit = json.loads((ROOT / 'evidence/head226-selected-comparison.json').read_text())
    if not audit['complete'] or not all(all(v.values()) for v in audit['fidelity'].values()):
        raise ValueError('Head observation fidelity missing')
    OUTPUT.mkdir(exist_ok=False)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    enable_full_determinism(42, close_matmul_k_shift=False)
    os.environ['CLOSE_MATMUL_K_SHIFT'] = '0'
    if torch.npu.matmul.allow_hf32:
        raise ValueError('FP32 diagnostic requires full FP32 operands')
    manifest = json.loads((ROOT / 'runs' / RUN / 'HEAD226_OPERANDS.json').read_text())
    records = {r['name']: r for r in manifest['full_records']}
    if len(records) != 5:
        raise ValueError('Incomplete full operands')
    weight_bits = load_bits(records['shared_weight'])
    weight = torch.from_numpy(np.array(weight_bits)).view(torch.bfloat16).npu()
    mf_rows = json.loads((ROOT / 'runs/mf-backward226/BACKWARD226_BOUNDARIES.jsonl').read_text().splitlines()[-1])
    mapping = audit['mapping_from_actual_input']
    results = []
    for module_name, call in mapping.items():
        branch = 'mtp' if '.mtp.' in module_name else 'main'
        gradient_bits = load_bits(records[f'call{call}_dy'])
        actual_bits = load_bits(records[f'call{call}_dx'])
        gradient = torch.from_numpy(np.array(gradient_bits)).view(torch.bfloat16).npu()
        target_name = 'model.mtp.layers.0.final_layernorm' if branch == 'mtp' else 'model.decoder.final_layernorm'
        target_record = next(r for r in mf_rows['records'] if r['name'] == target_name and r['kind'] == 'dy')
        target = np.load(STATE / 'mf-backward226/deep' / target_record['file'], mmap_mode='r').reshape(4096, 5120)
        if hashlib.sha256(target.tobytes()).hexdigest() != target_record['sha256']:
            raise ValueError('Actual MF dx target changed')
        selected_records = json.loads((ROOT / 'runs/mf-head226/HEAD226_OPERANDS.jsonl').read_text().splitlines()[-1])['records']
        selected_record = next(r for r in selected_records if r['name'] == module_name and r['kind'] == 'dy_matmul')
        selected = np.load(STATE / 'mf-head226/deep' / selected_record['file'])
        if hashlib.sha256(selected.tobytes()).hexdigest() != selected_record['sha256']:
            raise ValueError('MF observed gradient rows changed')
        indices = torch.tensor(audit['selection']['token_rows'], device=gradient.device)
        replacement = torch.from_numpy(selected).to(device=gradient.device, dtype=torch.bfloat16)
        for mode in ['native_mm', 'diagnostic_mf_dy_rows']:
            metadata = None
            if mode == 'native_mm':
                input_gradient = gradient
            else:
                input_gradient = gradient.clone()
                input_gradient.index_copy_(0, indices, replacement)
            output = torch.mm(input_gradient, weight)
            torch.npu.synchronize()
            bits = output.contiguous().view(torch.uint16).cpu().numpy()
            row = {'branch': branch, 'call': call, 'mode': mode,
                   'exact_hp_native': bool(np.array_equal(bits, actual_bits)),
                   'different_hp_native': int(np.count_nonzero(bits != actual_bits)),
                   'mf_comparison': compare_arrays(to_fp32(bits), target),
                   'shape': list(bits.shape), 'output_sha256': hashlib.sha256(bits.tobytes()).hexdigest(),
                   'storage_metadata': metadata}
            if mode == 'native_mm' and not row['exact_hp_native']:
                raise ValueError('Original dinput replay does not match actual training')
            np.save(OUTPUT / (branch + '-' + mode + '-bf16_bits.npy'), bits)
            results.append(row)
            print(json.dumps(row), flush=True)
            del output, bits, input_gradient
        del gradient
    report = {'complete': True, 'diagnostic_only': True, 'results': results,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'teacher_values_used': True, 'not_an_independent_training_candidate': True,
              'scope': 'Original full shapes and actual HP BF16 operands; replace only four observed MF dy rows in a diagnostic replay; compare every20971520 output; both original controls must exactly reproduce HP.',
              'mf_operand_scope': 'Four full vocabulary rows audited separately; MF full dy not observed, so unmatched inputs outside those rows are not excluded.'}
    (ROOT / 'evidence/head226-gradient-cause-replay.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
