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

"""Locate the first complete model divergence after packed compatibility factors."""
import hashlib
import json
from pathlib import Path

from compare_head206_interventions import source_sha
from compare_modelhash226 import compare_hashes, hashes, verify_sources
from compare_vocab_optimizer_trace import fidelity

ROOT = Path('/alignment/phase12')
RUN = 'hp-jtf637-packed217-both-hash226'
BASE = 'hp-jtf637-packed217-both226'
MF = 'mf-modelhash226'


def main() -> None:
    """Require exact loss/norm trajectory before trusting full-model hash observations."""
    checks = {'hp': fidelity(RUN, BASE, 26), 'mf': fidelity(MF, 'mf-resume', 26)}
    checks['mf'].update(verify_sources(MF, 'mf'))
    for label, run in [('hp', RUN), ('mf', MF)]:
        checks[label]['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
    directory = ROOT / 'runs' / RUN
    checks['hp']['same_recipe'] = (ROOT / 'configs' / (BASE + '.yaml')).read_text().replace(
        BASE, RUN) == (ROOT / 'configs' / (RUN + '.yaml')).read_text()
    for name in ['SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json', 'HEAD_GEMM_ROUTE.json',
                 'SHARED_UP_FP32_FACTOR.json', 'ADAMW_ARCHIVE_ARITHMETIC.json',
                 'Q_NS_ADJACENT_FACTOR.json', 'ROUTER_NS_NPU_SQRT_FACTOR.json',
                 'WEIGHTED_DVM_FACTOR.json', 'PACKED_SEMANTICS_FACTOR.json',
                 'HEAD_GEMM_ROUTE_CALLS.jsonl', 'Q_NS_ADJACENT_CALLS.jsonl',
                 'ROUTER_NS_NPU_SQRT_CALLS.jsonl', 'WEIGHTED_DVM_CALLS.jsonl',
                 'PACKED_SEMANTICS_CALLS.jsonl']:
        checks['hp']['unchanged_' + name] = (directory / name).read_bytes() == (ROOT / 'runs' / BASE / name).read_bytes()
    frozen = json.loads((directory / 'PROBE_SOURCE_MANIFEST.json').read_text())
    checks['hp']['49_frozen_sources'] = len(frozen['files']) == 49 and all(
        row['sha256'] == source_sha(row['name']) for row in frozen['files'])
    observer = json.loads((directory / 'MODEL_HASH_OBSERVER.json').read_text())
    checks['hp']['model_observer_source'] = observer['source_sha256'] == source_sha('hp_weighted212_model_hash.py')
    checks['hp']['model_hash_helper_source'] = observer['helper_sha256'] == source_sha('model_hash_stream.py')
    preclip = {}
    for step in [216, 217]:
        current = json.loads((directory / ('PRECLIP-' + str(step) + '.json')).read_text())
        reference = json.loads((ROOT / 'runs/mf-preclip217' / ('PRECLIP-' + str(step) + '.json')).read_text())
        expected = {row['name']: row for row in reference['records']}
        checks['hp']['preclip_source_' + str(step)] = current['source_sha256'] == source_sha(
            'hp_packed217_preclip_hashes.py')
        checks['hp']['46_preclip_' + str(step)] = len(current['records']) == len(expected) == 46 and {
            row['name'] for row in current['records']} == set(expected)
        different = [row['name'] for row in current['records'] if row['sha256'] != expected[row['name']]['sha256']]
        preclip[step] = {'exact': 46 - len(different), 'different': different, 'norm': current['norm']}
    if not all(all(check.values()) for check in checks.values()):
        raise ValueError(checks)
    models = compare_hashes(hashes(RUN), hashes(MF))
    first = next((row for row in models if any(row['different'].values())), None)
    report = {'complete': True, 'fidelity': checks, 'preclip': preclip, 'models': models,
              'first_model_difference': first,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'All48 postupdate model states201-226 and46 preclip gradients216/217; no moment equality claim.'}
    (ROOT / 'evidence/packed217-model-frontier-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'fidelity': True, 'preclip': preclip, 'first_model_difference': first}))


if __name__ == '__main__':
    main()
