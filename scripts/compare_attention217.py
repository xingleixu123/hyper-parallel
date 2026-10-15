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

"""Compare complete native layer0 attention inputs and localize packed-position differences."""
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_preclip212 import compare_arrays
from compare_head206_interventions import source_sha
from compare_vocab_optimizer_trace import fidelity, read_rows

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
HP = 'hp-jtf637-weighted-dvm-attention217'
MF = 'mf-attention217'
BASE = 'hp-jtf637-weighted-dvm-main226'


def main() -> None:
    """Require exact native trajectories before comparing all actual kernel operands."""
    checks = {'hp': fidelity(HP, BASE, 17), 'mf': fidelity(MF, 'mf-resume', 17)}
    for side, run in [('hp', HP), ('mf', MF)]:
        checks[side]['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
    checks['hp']['same_recipe'] = (ROOT / 'configs' / (BASE + '.yaml')).read_text().replace(
        BASE, HP).replace('train_iters: 226', 'train_iters: 217') == (ROOT / 'configs' / (HP + '.yaml')).read_text()
    checks['mf']['same_recipe'] = (ROOT / 'configs/mf-preclip217.yaml').read_text().replace(
        'mf-preclip217', MF) == (ROOT / 'configs' / (MF + '.yaml')).read_text()
    h = json.loads((ROOT / 'runs' / HP / 'ATTENTION217.json').read_text())
    m = read_rows(ROOT / 'runs' / MF / 'ATTENTION217.jsonl')
    meta = json.loads((STATE / MF / 'deep/manifest.json').read_text())
    checks['hp']['observer_source'] = h['source_sha256'] == source_sha('hp_attention217_observer.py')
    checks['mf']['observer_source'] = meta['source_sha256'] == hashlib.sha256(
        (ROOT / 'tools/mf_attention217_tap.py').read_bytes()).hexdigest()
    checks['mf']['all17_observed_steps'] = [r['step'] for r in m] == list(range(201,218))
    checks['hp']['all5_operands'] = len(h['records']) == 5
    checks['mf']['all7_operands'] = len(m[-1]['records']) == 7
    for name in ['SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json', 'HEAD_GEMM_ROUTE.json',
                 'SHARED_UP_FP32_FACTOR.json', 'ADAMW_ARCHIVE_ARITHMETIC.json', 'Q_NS_ADJACENT_FACTOR.json',
                 'ROUTER_NS_NPU_SQRT_FACTOR.json', 'WEIGHTED_DVM_FACTOR.json']:
        checks['hp']['unchanged_' + name] = (ROOT / 'runs' / HP / name).read_bytes() == (ROOT / 'runs' / BASE / name).read_bytes()
    if not all(all(v.values()) for v in checks.values()):
        raise ValueError(checks)
    hp = {r['name']: r for r in h['records']}
    mf = {r['kind']: r for r in m[-1]['records']}

    def load(side, key):
        row = (hp if side == 'hp' else mf)[key]
        path = STATE / (HP if side == 'hp' else MF) / ('attention' if side == 'hp' else 'deep') / row['file']
        array = np.load(path, mmap_mode='r')
        if list(array.shape) != row['shape'] or hashlib.sha256(array.tobytes()).hexdigest() != row['sha256']:
            raise ValueError('Actual attention operand changed')
        return array

    qlen, kvlen = load('mf','qlen'), load('mf','kvlen')
    # MF supplies a leading zero and repeated terminal offsets in a fixed-size vector.
    # Compare nonempty partitions, retaining the exact representation as an unresolved route factor.
    def partition(values):
        integers = values.astype(np.int64)
        if not np.array_equal(values, integers) or np.any(np.diff(integers) < 0):
            raise ValueError('Malformed cumulative boundary vector')
        return np.unique(integers[integers > 0]).tolist()

    q = partition(qlen)
    k = partition(kvlen)
    options = {'nonempty_query_partition_exact': q == h['options']['actual_seq_qlen'],
               'nonempty_key_partition_exact': k == h['options']['actual_seq_kvlen'],
               'scale_fp32_exact': bool(np.float32(h['options']['scale']) == np.float32(meta['options']['scale_value'])),
               'head_num_exact': h['options']['head_num'] == meta['options']['head_num'],
               'sparse_mode_exact': h['options']['sparse_mode'] == meta['options']['sparse_mode'],
               'pre_tokens_both_cover_sequence': min(h['options']['pre_tockens'],meta['options']['pre_tokens']) >= 4096}
    if not all(options.values()) or len(q) != 2:
        raise ValueError(options)
    observations = {}
    for name, other in [('query','query'),('key','key'),('value','value'),('output','output'),('atten_mask','mask')]:
        left, right = load('hp',name), load('mf',other)
        record = {'full': compare_arrays(left,right)}
        if name != 'atten_mask':
            record['first_document'] = compare_arrays(left[:q[0]],right[:q[0]])
            record['second_document'] = compare_arrays(left[q[0]:],right[q[0]:])
        if name in ['query','key']:
            record['non_rotary_channels'] = compare_arrays(np.ascontiguousarray(left[...,:128]),np.ascontiguousarray(right[...,:128]))
            record['rotary_channels'] = compare_arrays(np.ascontiguousarray(left[...,128:]),np.ascontiguousarray(right[...,128:]))
        observations[name] = record
        del left,right
    # Verify observed attention outputs are exactly the earlier full projection-input snapshots.
    old_hp = json.loads((ROOT/'runs/hp-jtf637-weighted-dvm-forward217/FORWARD217.json').read_text())['records']
    old_mf = read_rows(ROOT/'runs/mf-forward217/FORWARD217.jsonl')[-1]['records']
    links = {'hp_output_to_projection_input': hp['output']['sha256'] == next(r['sha256'] for r in old_hp if r['name']=='model.layers.0.self_attn.o_proj' and r['kind']=='input'),
             'mf_output_to_projection_input': mf['output']['sha256'] == next(r['sha256'] for r in old_mf if r['name']=='model.decoder.layers.0.self_attention.linear_proj' and r['kind']=='input')}
    if not all(links.values()):
        raise ValueError(links)
    report = {'complete':True,'fidelity':checks,'options_checks':options,'options':{'hp':h['options'],'mf':meta['options']},
              'mf_raw_qlen': qlen.tolist(), 'mf_raw_kvlen': kvlen.tolist(),
              'boundary_representation_identical': False,
              'boundary_links':links,'observations':observations,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope':'Complete actual inputs; query/key differences precede kernel. Nonempty partitions agree, but MF boundary padding differs. No same-input kernel-accuracy claim.'}
    (ROOT/'evidence/attention217-comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'fidelity':True,'options':options,'links':links,'differences':{name:{scope:r['fp32_bitwise_different'] for scope,r in records.items()} for name,records in observations.items()}}))


if __name__ == '__main__':
    main()
