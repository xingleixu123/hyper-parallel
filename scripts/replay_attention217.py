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

"""Replay full native attention operands without feeding them into training."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch_npu

from compare_preclip212 import compare_arrays
from compare_vocab_optimizer_trace import read_rows

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')


def main() -> None:
    """Separate upstream RoPE input differences from CANN and boundary representation."""
    os.environ['CLOSE_MATMUL_K_SHIFT'] = '0'
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.use_deterministic_algorithms(True)
    hp = json.loads((ROOT / 'runs/hp-jtf637-weighted-dvm-attention217/ATTENTION217.json').read_text())
    mf = read_rows(ROOT / 'runs/mf-attention217/ATTENTION217.jsonl')[-1]
    metadata = json.loads((STATE / 'mf-attention217/deep/manifest.json').read_text())
    sources = {'hp': {row['name']: row for row in hp['records']},
               'mf': {row['kind']: row for row in mf['records']}}

    def load(side: str, name: str) -> np.ndarray:
        row = sources[side][name]
        directory = 'hp-jtf637-weighted-dvm-attention217/attention' if side == 'hp' else 'mf-attention217/deep'
        array = np.load(STATE / directory / row['file'], mmap_mode='r')
        if hashlib.sha256(array.tobytes()).hexdigest() != row['sha256']:
            raise ValueError('Native operand changed')
        return array

    compact = hp['options']['actual_seq_qlen']
    padded = load('mf', 'qlen').astype(np.int64).tolist()
    mask = torch.from_numpy(load('hp', 'atten_mask').copy()).npu().bool()
    results = []
    for side in ['hp', 'mf']:
        tensors = {name: torch.from_numpy(load(side, name).copy()).npu().bfloat16()
                   for name in ['query', 'key', 'value']}
        native = load(side, 'output')
        for padding, lengths in [('compact', compact), ('mf_fixed', padded)]:
            for window in ([1048576, 2147483647] if side == 'mf' and padding == 'mf_fixed' else [1048576]):
                outputs = torch_npu.npu_fusion_attention(
                    tensors['query'], tensors['key'], tensors['value'], 128, 'TND',
                    pse=None, padding_mask=None, atten_mask=mask,
                    scale=float(metadata['options']['scale_value']), pre_tockens=window,
                    next_tockens=0, keep_prob=1.0, inner_precise=0, sparse_mode=3,
                    actual_seq_qlen=lengths, actual_seq_kvlen=lengths, softmax_layout='TND')
                actual = outputs[0].float().cpu().numpy()
                row = {'operand_source': side, 'boundary_representation': padding, 'pre_tokens': window,
                       'output_sha256': hashlib.sha256(actual.tobytes()).hexdigest(),
                       'vs_native_same_operands': compare_arrays(actual, native)}
                results.append(row)
                print(json.dumps(row), flush=True)
                del outputs, actual
        del tensors, native
    libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                        if '/cann-' in line and '.so' in line})
    if not libraries or any('/cann-9.2.0-beta.2/' not in path for path in libraries):
        raise ValueError('Unexpected CANN libraries')
    report = {'complete': True, 'results': results, 'cann_libraries': libraries,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Full attention forward replay only; no training intervention or backward correctness claim.'}
    (ROOT / 'evidence/attention217-same-input-replay.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
