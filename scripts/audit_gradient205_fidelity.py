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

"""Reject the205 gradient audit unless all six observed steps preserve the control."""
import hashlib
import json
from pathlib import Path

from compare_vocab_optimizer_trace import fidelity

ROOT = Path('/alignment/phase12')
RUN = 'hp-jtf637-adamw-all-grad205audit206'


def main():
    checks = {'exit_zero': (ROOT/'runs'/(RUN+'.exit')).read_text().strip() == '0'}
    if not checks['exit_zero']:
        raise ValueError('Gradient audit did not complete')
    checks.update(fidelity(RUN, 'hp-jtf637-adamw-all-archive206', 6))
    checks['selected_optimizer_trace_exact'] = (ROOT/'runs'/RUN/'VOCAB_OPTIMIZER_TRACE.jsonl').read_bytes() == (
        ROOT/'runs/hp-jtf637-adamw-all-archive206/VOCAB_OPTIMIZER_TRACE.jsonl').read_bytes()
    report = json.loads((ROOT/'evidence'/(RUN+'-grad205.json')).read_text())
    checks['source_hash_exact'] = report['source_sha256'] == hashlib.sha256(
        (ROOT/'tools-f637/hp_gradient205_audit.py').read_bytes()).hexdigest()
    if not all(checks.values()):
        raise ValueError(checks)
    records = report['records']
    out = {'complete': True, 'fidelity': checks, 'parameter_count': len(records),
           'actual_postclip_exact': sum(r['actual_postclip']['different']==0 for r in records),
           'readonly_mf_coefficient_replay_exact': sum(r['preclip_replay_mf_coefficient']['different']==0 for r in records),
           'remaining': [{'name':r['name'], **{k:v for k,v in r['preclip_replay_mf_coefficient'].items()
                           if k not in ('examples','actual_unscaled_sha256')}}
                         for r in records if r['preclip_replay_mf_coefficient']['different']],
           'scope': 'Complete44 non-vocabulary parameter gradients; coefficient replay is diagnostic only.'}
    (ROOT/'evidence/gradient205-alladamw-fidelity.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out))


if __name__ == '__main__':
    main()
