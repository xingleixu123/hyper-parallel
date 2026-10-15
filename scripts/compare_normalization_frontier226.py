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

"""Verify a read-only full-model observer on the independent Q/router trajectory."""
import hashlib
import json
from pathlib import Path

import yaml

from compare_head206_interventions import model_hashes, source_sha
from compare_vocab_optimizer_trace import fidelity, read_rows

ROOT = Path('/alignment/phase12')
RUN = 'hp-jtf637-norm208-q-router-modelhash226'
BASE = 'hp-jtf637-norm208-q-router-independent226'


def main() -> None:
    """Require exact loss, norm, inputs, restore and real calls before locating differences."""
    directory = ROOT / 'runs' / RUN
    baseline = ROOT / 'runs' / BASE
    checks = fidelity(RUN, BASE, 26)
    checks['exit_zero'] = (ROOT / 'runs' / (RUN + '.exit')).read_text().strip() == '0'
    for name in ['SOURCE_REVISION.json', 'PROBE_SOURCE_MANIFEST.json', 'RESTORE.json',
                 'HEAD_GEMM_ROUTE.json', 'HEAD_GEMM_ROUTE_CALLS.jsonl', 'SHARED_UP_FP32_FACTOR.json',
                 'ADAMW_ARCHIVE_ARITHMETIC.json', 'Q_NS_ADJACENT_FACTOR.json', 'Q_NS_ADJACENT_CALLS.jsonl',
                 'ROUTER_NS_NPU_SQRT_FACTOR.json', 'ROUTER_NS_NPU_SQRT_CALLS.jsonl']:
        checks['unchanged_' + name] = (directory / name).read_bytes() == (baseline / name).read_bytes()
    sources = json.loads((directory / 'PROBE_SOURCE_MANIFEST.json').read_text())
    checks['49_frozen_sources'] = len(sources['files']) == 49 and all(
        row['sha256'] == source_sha(row['name']) for row in sources['files'])
    observer = json.loads((directory / 'MODEL_HASH_OBSERVER.json').read_text())
    checks['observer_source'] = observer['source_sha256'] == source_sha('hp_normalization_frontier_model_hash.py')
    checks['observer_helper'] = observer['helper_sha256'] == source_sha('model_hash_stream.py')
    config = yaml.safe_load((ROOT / 'configs' / (RUN + '.yaml')).read_text())
    reference_config = yaml.safe_load((ROOT / 'configs' / (BASE + '.yaml')).read_text())
    config['checkpoint']['checkpoint_dir'] = reference_config['checkpoint']['checkpoint_dir']
    checks['same_recipe'] = config == reference_config
    checks['all26_steps'] = [row['step'] for row in read_rows(directory / 'metrics.jsonl')] == list(range(201, 227))
    checks['no_teacher'] = not (directory / 'DIAGNOSTIC_ONLY.json').exists()
    actual, reference = model_hashes(RUN), model_hashes('mf-modelhash226')
    checks['all26_hash_steps'] = list(actual) == list(range(201, 227))
    checks['all48_model_names'] = all(len(actual[step]) == 48 and set(actual[step]) == set(reference[step])
                                        for step in actual)
    if not all(checks.values()):
        raise ValueError(checks)
    rows = [{'step': step, 'different_vs_mf': {key: [
        name for name, value in actual[step].items() if value[key] != reference[step][name][key]]
        for key in ['fp32_sha256', 'bf16_as_fp32_sha256']}} for step in actual]
    report = {'complete': True, 'readonly': True, 'run': RUN, 'control': BASE, 'fidelity': checks,
              'first_different_model_step': next((row['step'] for row in rows
                                                  if row['different_vs_mf']['fp32_sha256']), None),
              'rows': rows, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Model hashes only; no claim that all optimizer slots are equal after step200. '
                       'Same independent26 trajectory and full200 prefix; no new independent full200 run.'}
    (ROOT / 'evidence/normalization-frontier226-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'complete': True, 'first': report['first_different_model_step'],
                      'counts': [(row['step'], len(row['different_vs_mf']['fp32_sha256'])) for row in rows]}))


if __name__ == '__main__':
    main()
