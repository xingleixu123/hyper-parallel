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

"""Verify the new JT revision preserves the controlled six-step numerical window."""
import json
from pathlib import Path
import numpy as np
import yaml
ROOT=Path('/alignment/phase12')
OLD='hp-cann92-jtad1-pr180-192-routercontig-denseadj-castall'
NEW='hp-jtf637-denseadj-short206'


def main():
    old_dir=ROOT/'runs'/OLD
    new_dir=ROOT/'runs'/NEW
    old={r['step']:r for r in map(json.loads,(old_dir/'metrics.jsonl').read_text().splitlines())}
    new={r['step']:r for r in map(json.loads,(new_dir/'metrics.jsonl').read_text().splitlines())}
    checks={'six_steps':sorted(new)==list(range(201,207)),
            'all_metrics_and_grad_norm_identical':all(r==old[s] for s,r in new.items()),
            'restoration_identical':json.loads((old_dir/'RESTORE.json').read_text())==json.loads((new_dir/'RESTORE.json').read_text())}
    for s in new:
        with np.load(old_dir/'batches'/f'{s-1:06d}.npz') as a, np.load(new_dir/'batches'/f'{s-1:06d}.npz') as b:
            checks['input_'+str(s)]=set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
    old_cfg=yaml.safe_load((ROOT/'configs'/(OLD+'.yaml')).read_text())
    new_cfg=yaml.safe_load((ROOT/'configs'/(NEW+'.yaml')).read_text())
    old_cfg['training']['train_iters']=206
    chunk=old_cfg['model']['config'].pop('loss_chunk_size')
    old_cfg['model']['config']['enable_chunk_loss']=chunk>0
    old_cfg['model']['config']['chunk_loss_size']=chunk
    checks['configuration_semantics_preserved']=old_cfg==new_cfg
    revision=json.loads((new_dir/'SOURCE_REVISION.json').read_text())
    checks['latest_integration']=revision['commit']=='9213a97147020fb75e7a98a40f7e7b636f98d349'
    runtime=json.loads((new_dir/'CANN_RUNTIME.json').read_text())
    checks['actual_latest_import']='/hp-jt-f637-pr192/' in runtime['hyper_parallel_path']
    if not all(checks.values()):raise ValueError(checks)
    report={'complete':True,'checks':checks,'old_run':OLD,'new_run':NEW,
            'old_upstream':'521d3a8a9aa62353327093ab1258ca2c22006e8b',
            'new_upstream':'f637ef7ee4d4b416800716f1ebaaae7709b8a06a',
            'scope':'Controlled compatibility trajectory; six-step parity does not establish200-step acceptance.'}
    (ROOT/'evidence/jt-f637-six-step-transition.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':
    main()
