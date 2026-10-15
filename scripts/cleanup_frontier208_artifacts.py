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

"""Remove only proven duplicate owned observations and hardlink identical retained state files."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
DESTINATION = ROOT / 'evidence/frontier208-cleanup.json'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def checked(path):
    if path.is_symlink() or not path.resolve().is_relative_to(STATE.resolve()) or path.suffix != '.npy':
        raise ValueError('Unsafe owned artifact path')
    return path


def main():
    if DESTINATION.exists():
        raise ValueError('Cleanup receipt already exists; refusing duplicate accounting')
    targets = ['mf-frontier208-ns-train', 'mf-frontier208-native-norm', 'mf-frontier208-native-normalized']
    for run in targets + ['mf-frontier208-muon-replay', 'hp-frontier208-muon-replay', 'frontier208-normalization-replay-r3']:
        if (ROOT / 'runs' / (run + '.exit')).read_text().strip() != '0':
            raise ValueError('Relevant observation or replay has not terminated successfully')
    open_paths = set()
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            open_paths.add((proc / 'cwd').resolve())
            open_paths.update(fd.resolve() for fd in (proc / 'fd').iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            pass
    native_rows = {row['prefix']: row for row in json.loads((ROOT / 'runs/mf-frontier208-ns-train/NS_BOUNDARIES-208.json').read_text())}
    removals = []
    for label in ['q', 'router']:
        for path in sorted((STATE / 'mf-frontier208-muon-replay' / label).glob('*.npy')):
            checked(path)
            name = path.name.split('_float32_')[0]
            canonical = (STATE / 'hp-frontier208-muon-replay' / label / (name + '.npy') if label == 'q'
                         else STATE / 'mf-frontier208-ns-operators' / native_rows[label + '-' + name]['file'])
            checked(canonical)
            left, right = np.load(path), np.load(canonical)
            if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
                raise ValueError('Isolated observation is not an exact duplicate')
            if path.resolve() in open_paths:
                raise ValueError('Duplicate is still in use')
            record = {'path': str(path), 'canonical': str(canonical), 'file_sha256': digest(path),
                      'canonical_file_sha256': digest(canonical), 'array_sha256': hashlib.sha256(left.tobytes()).hexdigest(),
                      'shape': list(left.shape), 'dtype': str(left.dtype), 'logical_bytes': path.stat().st_size,
                      'physical_bytes_released': path.stat().st_blocks * 512 if path.stat().st_nlink == 1 else 0,
                      'reconstruction': 'Load canonical NPY; array shape, dtype and SHA are exact. Original NPY header bytes need not match.'}
            removals.append((path, record))
    canonical_files = {}
    for path in (STATE / 'mf-frontier208-state').rglob('*.npy'):
        canonical_files[digest(checked(path))] = path
    links = []
    for run in targets:
        for path in sorted((STATE / run).rglob('*.npy')):
            checked(path)
            sha = digest(path)
            source = canonical_files.get(sha)
            if source is None:
                raise ValueError('Retained state or gradient no longer reproduces its canonical file')
            if path.resolve() in open_paths:
                raise ValueError('Retained state is still open')
            if path.stat().st_ino == source.stat().st_ino:
                continue
            links.append((path, source, {'path': str(path), 'canonical': str(source), 'sha256': sha,
                                         'physical_bytes_released': path.stat().st_blocks * 512 if path.stat().st_nlink == 1 else 0}))
    # All scope, ownership, fidelity and active-file checks precede mutation.
    for path, _ in removals:
        path.unlink()
    for path, source, record in links:
        temporary = path.with_suffix('.p12-dedup-tmp')
        if temporary.exists():
            raise ValueError('Unexpected dedup temporary path')
        os.link(source, temporary)
        os.replace(temporary, path)
        if digest(path) != record['sha256'] or path.stat().st_ino != source.stat().st_ino:
            raise ValueError('Post-dedup verification failed')
    temporary_logical = sum(row['bytes'] for step in range(201, 208) for row in json.loads(
        (ROOT / 'runs/mf-frontier208-ns-train' / ('NS_BOUNDARIES-' + str(step) + '.json')).read_text()))
    report = {'complete': True, 'deleted_duplicates': [row for _, row in removals],
              'hardlinked_retained_files': [row for _, _, row in links],
              'deleted_physical_bytes': sum(row['physical_bytes_released'] for _, row in removals),
              'hardlink_savings_bytes': sum(row['physical_bytes_released'] for _, _, row in links),
              'deleted_logical_bytes': sum(row['logical_bytes'] for _, row in removals),
              'transient201_207_ns_logical_bytes_already_pruned_not_added_to_physical_total': temporary_logical,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    DESTINATION.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ['deleted_duplicates', 'hardlinked_retained_files']}))


if __name__ == '__main__':
    main()
