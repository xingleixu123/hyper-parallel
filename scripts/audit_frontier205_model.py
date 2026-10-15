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

"""Verify the step205 model interventions against independent controls."""
import json
from pathlib import Path
import numpy as np
from compare_loss_components import read_metrics

ROOT = Path('/alignment/phase12')
CONTROL = 'hp-cann92-jtad1-pr180-192-routercontig-denseadj-castall'


def rows(name):
    return {row['step']: row for row in map(json.loads, (ROOT/'runs'/name/'metrics.jsonl').read_text().splitlines())}


def exact_inputs(left, right, steps):
    for step in steps:
        with np.load(ROOT/'runs'/left/'batches'/f'{step-1:06d}.npz') as a, np.load(ROOT/'runs'/right/'batches'/f'{step-1:06d}.npz') as b:
            if set(a.files) != set(b.files) or not all(np.array_equal(a[k], b[k]) for k in a.files):
                raise ValueError(('input mismatch', left, right, step))
    return True


def main():
    baseline = rows(CONTROL)
    mf = read_metrics(ROOT/'runs/mf-resume/metrics.jsonl', 'mf', 1e-4, .3)
    probe = read_metrics(ROOT/'runs/mf-grad205-state/metrics.jsonl', 'mf', 1e-4, .3)
    if sorted(probe) != list(range(201, 207)) or not all(probe[s] == mf[s] for s in probe):
        raise ValueError('MF205 observer changed metrics')
    exact_inputs('mf-grad205-state', 'mf-resume', range(201, 207))
    short = rows('hp-routercontig-denseadj-grad205')
    if not all(short[s] == baseline[s] for s in short):
        raise ValueError('Independent HP observer changed metrics')
    records = []
    for factor, count in {'head': 1, 'embedding': 1, 'norms': 3, 'remaining': 5}.items():
        name = 'hp-denseadj-mf'+factor+'204-diag206'
        folder = ROOT/'runs'/name
        observed = rows(name)
        if sorted(observed) != list(range(201,207)):
            raise ValueError('incomplete run')
        checks = {f: json.loads((folder/f).read_text()) == json.loads((ROOT/'runs'/CONTROL/f).read_text())
                  for f in ('RESTORE.json', 'SOURCE_REVISION.json')}
        checks['all_inputs_exact'] = exact_inputs(name, CONTROL, range(201,207))
        checks['pre_intervention_metrics_exact'] = all(observed[s] == baseline[s] for s in range(201,205))
        marker = json.loads((folder/'DIAGNOSTIC_ONLY.json').read_text())
        intervention = json.loads((folder/'MODEL204_INTERVENTION.json').read_text())
        checks['intervention_verified'] = (marker['diagnostic_only'] and not marker['eligible_for_acceptance']
            and intervention['complete'] and intervention['optimizer_untouched']
            and len(intervention['records']) == count
            and all(r['all_elements_copied_and_verified'] for r in intervention['records']))
        if not all(checks.values()):raise ValueError((name,checks))
        hp = read_metrics(folder/'metrics.jsonl','hp',1e-4,.3)
        grad = json.loads((ROOT/'evidence'/(name+'-comparison.json')).read_text())['records']
        records.append({'run':name,'factor':factor,'checks':checks,'diagnostic_only':True,
            'gradient_count':len(grad),'gradient_exact':sum(r['unequal']==0 for r in grad),
            'worst_gradient_relative_l2':max(r['relative_l2'] for r in grad),
            'loss_errors':{str(s):{k:hp[s][k]-mf[s][k] for k in ('total_loss','lm_loss','mtp_raw','aux_raw')}
                           for s in (205,206)},'grad_norm':{str(s):observed[s]['grad_norm'] for s in (205,206)}})
    report={'complete':True,'mf_observer_metrics_and_inputs_exact':True,
            'hp_independent_observer_metrics_exact':True,'records':records,
            'scope':'Causal diagnosis only; no teacher intervention is an acceptance run.'}
    (ROOT/'evidence/frontier205-model-causal-audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
