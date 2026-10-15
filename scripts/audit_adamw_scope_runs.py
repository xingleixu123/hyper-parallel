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

"""Audit three independent AdamW scope ablations before promoting any full run."""
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml
from compare_loss_components import read_metrics
from compare_vocab_optimizer_trace import stats

ROOT = Path('/alignment/phase12')
BASE = 'hp-jtf637-denseadj-short206'


def main():
    mf = read_metrics(ROOT/'runs/mf-resume/metrics.jsonl', 'mf', .0001, .3)
    anchors = {(r['step'], r['name']): r['state'] for r in json.loads((ROOT/'evidence/vocab-native-coordinate-anchors.json').read_text())}
    result = []
    for scope in ['embedding', 'head', 'all']:
        name = 'hp-jtf637-adamw-'+scope+'-archive206'
        directory = ROOT/'runs'/name
        if (ROOT/'runs'/(name+'.exit')).read_text().strip() != '0':
            raise ValueError('Unsuccessful run '+name)
        checks = {}
        for filename in ['RESTORE.json', 'SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json']:
            checks[filename] = json.loads((directory/filename).read_text()) == json.loads((ROOT/'runs'/BASE/filename).read_text())
        checks['configuration'] = yaml.safe_load((ROOT/'configs'/(name+'.yaml')).read_text()) == yaml.safe_load((ROOT/'configs'/(BASE+'.yaml')).read_text())
        policy = json.loads((directory/'ADAMW_ARCHIVE_ARITHMETIC.json').read_text())
        checks['factor_source'] = policy['source_sha256'] == hashlib.sha256((ROOT/'tools-f637/adamw_archive_scope_factor.py').read_bytes()).hexdigest()
        checks['no_teacher'] = policy['teacher_values_used'] is False
        for step in range(201, 207):
            filename = f'{step-1:06d}.npz'
            with np.load(directory/'batches'/filename) as actual, np.load(ROOT/'runs'/BASE/'batches'/filename) as control:
                checks['input_'+str(step)] = set(actual.files) == set(control.files) and all(np.array_equal(actual[k], control[k]) for k in actual.files)
        actual = read_metrics(directory/'metrics.jsonl', 'hp', .0001, .3)
        checks['six_steps'] = sorted(actual) == list(range(201, 207))
        if not all(checks.values()):
            raise ValueError(checks)
        traces = [json.loads(line) for line in (directory/'VOCAB_OPTIMIZER_TRACE.jsonl').read_text().splitlines()]
        coordinates = []
        for row in traces:
            if row['step'] == 204:
                native = anchors[(204, row['name'])]
                coordinates.append({'name': row['name'], **{field: stats(value, native[field]) for field, value in row['after'].items()}})
        result.append({'run': name, 'scope': scope, 'checks': checks,
                       'deltas': {step: {key: actual[step][key]-mf[step][key] for key in mf[step]} for step in actual},
                       'native204_coordinate_comparison': coordinates})
    report = {'complete': True, 'records': result, 'scope': 'Six-step independent arithmetic ablations; not200-step acceptance.'}
    (ROOT/'evidence/adamw-three-scope-six-step-audit.json').write_text(json.dumps(report, indent=2))
    for row in result:
        print(row['scope'], {step: row['deltas'][step] for step in [205, 206]})
        print('coordinate204', {v['name']: {key: value['equal_count'] for key, value in v.items() if key != 'name'} for v in row['native204_coordinate_comparison']})


if __name__ == '__main__':
    main()
