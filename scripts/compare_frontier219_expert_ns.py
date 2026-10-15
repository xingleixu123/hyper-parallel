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

"""Compare complete valid expert219 NS replay boundaries; exclude invalid dense replay."""
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_preclip217 import compare_arrays, bf16_bits
ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')


def main():
    hp_report = json.loads((ROOT / 'evidence/frontier219-hp-muon-replay.json').read_text())
    mf_report = json.loads((ROOT / 'evidence/frontier219-mf-muon-replay.json').read_text())
    row = next(r for r in mf_report['records'] if r['label'] == 'expert')
    assert all(row[k + '_weight_exact_training'] and row[k + '_momentum_exact_training']
               for k in ['native', 'observed'])
    assert all(r['weight_exact_training'] and r['momentum_exact_training'] for r in hp_report['checks'])
    hp_dir, mf_dir = STATE / 'hp-frontier219-muon-replay', STATE / 'mf-frontier219-muon-replay'
    files = []

    def native(prefix):
        paths = list(mf_dir.glob(prefix + '_*.npy'))
        assert len(paths) == 1, (prefix, paths)
        value = np.load(paths[0], mmap_mode='r')
        assert value.dtype == np.float32 and paths[0].stat().st_size == value.offset + value.nbytes
        files.append({'file': paths[0].name, 'shape': list(value.shape),
                      'sha256': hashlib.sha256(value.tobytes()).hexdigest()})
        return value

    pre = native('expert-pre_ns').reshape(8, 5120, 3072).transpose(0, 2, 1)
    comparisons = []
    for piece in [0, 1]:
        prefix = 'expert-' + str(piece)
        hp_row = next(r for r in hp_report['records'] if r['label'] == 'expert' and r['piece'] == piece)
        hp_arrays = {}
        for record in hp_row['arrays']:
            value = np.load(hp_dir / prefix / (record['name'] + '.npy'), mmap_mode='r')
            assert hashlib.sha256(value.tobytes()).hexdigest() == record['sha256']
            hp_arrays[record['name']] = value
        rounded = (bf16_bits(np.ascontiguousarray(pre[:, piece * 1536:(piece + 1) * 1536]))
                   .astype(np.uint32) << 16).view(np.float32)
        item = {'piece': piece, 'pre_ns_rounded': compare_arrays(hp_arrays['pre_ns'], rounded),
                'norm_hp': hp_row['scalar']['norm'], 'norm_mf': native(prefix + '-norm').tolist(),
                'denominator_hp': hp_row['scalar']['denominator'],
                'denominator_mf': native(prefix + '-denominator').tolist()}
        for kind in ['normalized', 'ns_output']:
            other = native(prefix + '-' + kind)
            if kind == 'ns_output':
                other = np.ascontiguousarray(other.transpose(0, 2, 1))
            item[kind] = compare_arrays(hp_arrays[kind], other)
            item[kind + '_by_expert'] = [compare_arrays(hp_arrays[kind][i], other[i]) for i in range(8)]
        comparisons.append(item)
    report = {'complete': True, 'expert_replays_exact_training': True,
              'dense_replay_excluded': True, 'comparisons': comparisons, 'mf_array_manifest': files,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / 'evidence/frontier219-expert-ns-comparison.json').write_text(json.dumps(report, indent=2))
    for row in comparisons:
        print(json.dumps({k:v for k,v in row.items() if not k.endswith('_by_expert')}), flush=True)


if __name__ == '__main__':
    main()
