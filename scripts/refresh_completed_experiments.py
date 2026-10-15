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

"""Rebuild the complete numerical inventory from verified completed run artifacts."""
import csv
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    references = {}
    for path in (root / 'evidence').glob('*components/component-comparison.json'):
        report = json.loads(path.read_text(encoding='utf-8'))
        if len(report.get('steps', [])) == 200:
            references.setdefault(report['source_sha256']['hp'], []).append((path, report))
    records, excluded = [], []
    for success in sorted(set((root / 'runs').glob('hp-cann92*/SUCCESS.json')) | set((root / 'runs').glob('hp-jtf637*/SUCCESS.json'))):
        directory = success.parent
        info = json.loads(success.read_text(encoding='utf-8'))
        if info.get('steps') != 200:
            continue
        if (directory / 'DIAGNOSTIC_ONLY.json').exists():
            excluded.append({'run': directory.name, 'reason': 'diagnostic intervention'})
            continue
        metrics = directory / 'metrics.jsonl'
        lines = metrics.read_text(encoding='utf-8').splitlines()
        if [json.loads(line)['step'] for line in lines] != list(range(201, 401)):
            raise ValueError('Incomplete/duplicate window: ' + directory.name)
        digest = hashlib.sha256(metrics.read_bytes()).hexdigest()
        candidates = references.get(digest, [])
        candidates = [(path, row) for path, row in candidates if row.get('eligible_for_acceptance', True)]
        if not candidates:
            raise ValueError('Completed metrics have no matching existing comparison: ' + directory.name)
        if any(row['losses'] != candidates[0][1]['losses'] for _, row in candidates):
            raise ValueError('Comparison disagreement: ' + directory.name)
        path, comparison = min(candidates, key=lambda pair: (directory.name not in pair[0].parent.name, len(str(pair[0]))))
        revision_file = directory / 'SOURCE_REVISION.json'
        revision = json.loads(revision_file.read_text()) if revision_file.exists() else None
        records.append({'run': directory.name, 'source_revision': revision,
                        'metrics_sha256': digest, 'comparison': str(path.relative_to(root)),
                        'losses': comparison['losses'], 'numerical_passed': all(item['passed'] for item in comparison['losses'].values())})
    report = {'complete_runs': len(records), 'excluded': excluded,
              'scope': 'Numerical inventory of completed CANN9.2 independent trajectories; causal priority must use controlled pairs',
              'records': records}
    output = root / 'evidence/completed-cann92-inventory.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    fields = ['run', 'commit', 'total_first', 'total_max', 'total_max_step', 'total_mae', 'total_bad_steps',
              'mtp_raw_max', 'aux_raw_max', 'numerical_passed', 'comparison']
    with (root / 'evidence/completed-cann92-inventory.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            total = row['losses']['total_loss']
            writer.writerow({'run': row['run'], 'commit': (row['source_revision'] or {}).get('commit', 'historical source audit'),
                             'total_first': total['first'], 'total_max': total['max'], 'total_max_step': total['max_step'],
                             'total_mae': total['mean'], 'total_bad_steps': total['over_window_tolerance'],
                             'mtp_raw_max': row['losses']['mtp_raw']['max'], 'aux_raw_max': row['losses']['aux_raw']['max'],
                             'numerical_passed': row['numerical_passed'], 'comparison': row['comparison']})
    print(json.dumps({'complete': len(records), 'excluded': excluded,
                      'all_six_components_passed': sum(row['numerical_passed'] for row in records)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
