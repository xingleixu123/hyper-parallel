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

"""Verify compact207 gradient reconstruction, optionally restore the redundant HP array for replay."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')


def main() -> None:
    """Hash full MF and reconstructed HP gradients before any optional restoration."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore-hp', action='store_true')
    args = parser.parse_args()
    report = json.loads((ROOT / 'evidence/head-moments207-gradient-comparison.json').read_text())
    row = report['gradient_comparison']
    patch = Path(row['patch_file'])
    checks = {'patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest() == row['patch_sha256']}
    with np.load(patch) as archive:
        indices, hp, mf = (archive[key].copy() for key in ['indices', 'hp', 'mf'])
    original = np.load(report['native_kernels']['input.0']['path'], mmap_mode='r')
    flat = original.reshape(-1)
    checks['shape_dtype'] = original.shape == (152064, 5120) and original.dtype == np.float32
    checks['indices'] = bool(indices.size == row['different'] and np.all(indices[1:] > indices[:-1])
                            and indices.min() >= 0 and indices.max() < flat.size)
    checks['native_values'] = bool(np.array_equal(flat[indices], mf) and np.all(mf != hp))
    if not all(checks.values()):
        raise ValueError(checks)
    hashes = {'mf': hashlib.sha256(), 'hp': hashlib.sha256()}
    for start in range(0, flat.size, 1_000_000):
        stop = min(start + 1_000_000, flat.size)
        values = np.array(flat[start:stop], copy=True)
        hashes['mf'].update(memoryview(values).cast('B'))
        low, high = np.searchsorted(indices, [start, stop])
        values[indices[low:high] - start] = hp[low:high]
        hashes['hp'].update(memoryview(values).cast('B'))
    actual = {side: sha.hexdigest() for side, sha in hashes.items()}
    checks['mf_full_hash'] = actual['mf'] == report['native_kernels']['input.0']['sha256']
    checks['hp_full_hash'] = actual['hp'] == report['hp_gradient']['sha256']
    if not all(checks.values()):
        raise ValueError(checks)
    if args.restore_hp:
        target = Path(report['hp_gradient']['file'])
        if target.exists() or target.is_symlink() or not target.resolve().is_relative_to(STATE.resolve()):
            raise ValueError('Restore requires an absent target inside this task state root')
        restored = np.lib.format.open_memmap(target, mode='w+', dtype=np.float32, shape=original.shape)
        restored[:] = original
        restored.reshape(-1)[indices] = hp
        restored.flush()
        if hashlib.sha256(memoryview(restored).cast('B')).hexdigest() != actual['hp']:
            raise ValueError('Restored full array hash failed')
    result = {'complete': True, 'checks': checks, 'hashes': actual, 'full_hp_array_restored': args.restore_hp,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / 'evidence/head-moments207-compact-reconstruction.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
