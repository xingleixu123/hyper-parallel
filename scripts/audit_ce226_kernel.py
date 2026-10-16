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

"""Verify graph-kernel dumping preserves all observed native training boundaries."""
import hashlib
import json
from pathlib import Path
from compare_vocab_optimizer_trace import fidelity, read_rows

root = Path('/alignment/phase12')
state = Path('/state/jt-recipe-20261014')
run, baseline = 'mf-ce226-kernel', 'mf-ce226'
checks = fidelity(run, baseline, 26)
checks['exit_zero'] = (root / 'runs' / (run + '.exit')).read_text().strip() == '0'
checks['config_exact_except_path'] = (root / 'configs' / (baseline + '.yaml')).read_text().replace(baseline, run) == (root / 'configs' / (run + '.yaml')).read_text()
actual = read_rows(root / 'runs' / run / 'CE226_OPERANDS.jsonl')
expected = read_rows(root / 'runs' / baseline / 'CE226_OPERANDS.jsonl')
def normalized(row):
    return row['step'], sorted((r['name'], r['kind'], str(r['shape']), r['sha256']) for r in row['records'])
checks['all26_operand_hashes_exact'] = list(map(normalized, actual)) == list(map(normalized, expected))
report = {'complete': all(checks.values()), 'checks': checks, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(root / 'evidence/ce226-kernel-fidelity.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
if not report['complete']:
    raise ValueError('Kernel dump changed native training')
