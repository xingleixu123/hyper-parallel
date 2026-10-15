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

"""Stage only bounded task evidence; preserve raw arrays on the private host."""
import hashlib
import json
from pathlib import Path
import shutil

from sync_frontier217_evidence import RUNS

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / '公开证据'
REMOVE = {'worst_index', 'hp_at_worst', 'mf_at_worst'}


def sanitize(value):
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items() if key not in REMOVE}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def copy(source, destination):
    if not source.is_file():
        return
    if source.suffix.lower() in ('.npy', '.npz') or source.stat().st_size > 20_000_000:
        raise ValueError('Unexpected large or raw artifact')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def main():
    for filename in ['frontier219-source-manifest.json', 'frontier222-source-manifest.json', 'frontier226-source-manifest.json', 'head226-source-manifest.json']:
        manifest = json.loads((ROOT / 'evidence' / filename).read_text(encoding='utf-8'))
        for row in manifest['files']:
            source = ROOT / row['name']
            if hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Source manifest mismatch: ' + row['name'])
            folder = 'scripts/latest-jt-f637' if row['folder'] == 'tools-f637' else 'scripts'
            copy(source, PUBLIC / folder / row['name'])
    for name in ['sync_frontier217_evidence.py', 'plot_qsqrt222_independent200.py', 'stage_frontier226_public.py']:
        copy(ROOT / name, PUBLIC / 'scripts' / name)
    for run in RUNS:
        for suffix in ['.yaml', '.json']:
            copy(ROOT / 'configs' / (run + suffix), PUBLIC / 'configs' / (run + suffix))
        copy(ROOT / ('run-' + run + '.sh'), PUBLIC / 'launchers' / ('run-' + run + '.sh'))
        copy(ROOT / 'evidence' / (run + '-launch.json'), PUBLIC / 'evidence' / (run + '-launch.json'))
        terminal = ROOT / 'runs' / (run + '.exit')
        if not terminal.exists():
            continue
        for suffix in ['.exit', '.log', '.dispatch.log']:
            copy(ROOT / 'runs' / (run + suffix), PUBLIC / 'runs' / (run + suffix))
        for path in (ROOT / 'runs' / run).glob('*'):
            if path.suffix in ('.json', '.jsonl'):
                copy(path, PUBLIC / 'runs' / run / path.name)
        for suffix in ['.yaml', '.json']:
            copy(ROOT / 'configs' / (run + suffix), PUBLIC / 'configs' / (run + suffix))
        copy(ROOT / ('run-' + run + '.sh'), PUBLIC / 'launchers' / ('run-' + run + '.sh'))
        copy(ROOT / 'evidence' / (run + '-launch.json'), PUBLIC / 'evidence' / (run + '-launch.json'))
    for prefix in ['frontier222-', 'qsqrt222-', 'preclip226-', 'frontier226-', 'forward226-', 'backward226-', 'head226-']:
        for path in (ROOT / 'evidence').glob(prefix + '*'):
            if path.suffix not in ('.json', '.csv', '.png', '.svg', '.log'):
                continue
            if path.name in ['frontier219-parent-before.json', 'backward226-source-manifest.json', 'frontier226-parent-before.json']:
                continue
            destination = PUBLIC / 'evidence' / path.name
            if path.name == 'frontier219-issue-pr-link-audit.json':
                audit = json.loads(path.read_text(encoding='utf-8'))
                for row in audit['rows']:
                    body = row.pop('body', '')
                    row['body_sha256'] = hashlib.sha256(body.encode()).hexdigest()
                    row['links_parent217'] = '#217' in body or '/issues/217' in body
                audit['private_source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                destination.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                continue
            if path.suffix == '.json':
                original = json.loads(path.read_text(encoding='utf-8'))
                redacted = sanitize(original)
                if redacted != original:
                    redacted['public_redaction'] = {'removed_fields': sorted(REMOVE),
                                                    'private_source_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                    destination.write_text(json.dumps(redacted, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                    continue
            copy(path, destination)
    copy(ROOT / '阶段记录-226前反向首差与消融.md', PUBLIC / '阶段记录-226前反向首差与消融.md')
    copy(ROOT / '阶段附录-206输出头矩与207梯度因果完整记录.md', PUBLIC / '阶段附录-206输出头矩与207梯度因果完整记录.md')
    for path in (ROOT / 'industry').glob('frontier219-*.py'):
        copy(path, PUBLIC / 'industry' / path.name)
    for name in ['frontier219-state-lossless-archive.json', 'frontier219-state-lossless-archive.log', 'frontier219-parent-publication.json', 'frontier219-source-manifest.json']:
        copy(ROOT / 'evidence' / name, PUBLIC / 'evidence' / name)
    for suffix in ['json', 'csv']:
        copy(ROOT / ('evidence/completed-cann92-inventory.' + suffix), PUBLIC / ('evidence/completed-cann92-inventory.' + suffix))
    print('Staged bounded source/config/log/hash evidence; no tensor arrays copied.')


if __name__ == '__main__':
    main()
