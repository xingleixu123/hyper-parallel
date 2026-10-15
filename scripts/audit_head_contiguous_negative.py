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

"""Verify executed head layout ablations preserve the observed six-step control."""
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml
ROOT = Path('/alignment/phase12')


def main():
    base = 'hp-jtf637-adamw-all-archive206'
    records = []
    for version in ['r2', 'r3']:
        name = 'hp-jtf637-adamw-all-headcontig206-'+version
        directory = ROOT/'runs'/name
        checks = {'exit_zero': (ROOT/'runs'/(name+'.exit')).read_text().strip() == '0'}
        for file in ['metrics.jsonl', 'VOCAB_OPTIMIZER_TRACE.jsonl']:
            checks[file+'_exact'] = (directory/file).read_text() == (ROOT/'runs'/base/file).read_text()
        for file in ['RESTORE.json', 'SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json']:
            checks[file+'_exact'] = json.loads((directory/file).read_text()) == json.loads((ROOT/'runs'/base/file).read_text())
        checks['configuration_exact'] = yaml.safe_load((ROOT/'configs'/(name+'.yaml')).read_text()) == yaml.safe_load((ROOT/'configs'/(base+'.yaml')).read_text())
        for step in range(201, 207):
            filename = f'{step-1:06d}.npz'
            with np.load(directory/'batches'/filename) as actual, np.load(ROOT/'runs'/base/'batches'/filename) as control:
                checks['input_'+str(step)] = set(actual.files) == set(control.files) and all(np.array_equal(actual[k], control[k]) for k in actual.files)
        policy = json.loads((directory/'HEAD_WEIGHT_GRADIENT.json').read_text())
        checks['source_hash'] = policy['source_sha256'] == hashlib.sha256((ROOT/'tools-f637'/('head_weight_contiguous_factor_'+version+'.py')).read_bytes()).hexdigest()
        if version == 'r3':
            calls = [json.loads(line) for line in (directory/'HEAD_BACKWARD_COORDINATES.jsonl').read_text().splitlines()]
            checks['twelve_backward_calls_executed'] = [r['call'] for r in calls] == list(range(1, 13))
            checks['all_gemm_shapes'] = all(r['dy_shape'] == [4096, 152064] for r in calls)
        if not all(checks.values()):
            raise ValueError(checks)
        records.append({'run': name, 'checks': checks})
    report = {'complete': True, 'control': base, 'records': records,
              'conclusion': 'Output-head contiguous transpose has no observed six-step metric or sampled optimizer-gradient/state effect; actual12backward calls verified. Deprioritize this factor.'}
    (ROOT/'evidence/head-dweight-contiguous-six-step-negative.json').write_text(json.dumps(report, indent=2))
    print('head negative fidelity complete')


if __name__ == '__main__':
    main()
