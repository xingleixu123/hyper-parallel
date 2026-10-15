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

"""Keep only step205 gradients while preserving the archive's native arithmetic."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import capture_mf_muon

SEEN = set()

ORIGINAL_END = capture_mf_muon.ObserveMetrics.on_train_step_end


def observed_end(self, run_context):
    """Hash and prune consumed step201 gradients before the next native step."""
    ORIGINAL_END(self, run_context)
    step = int(run_context.original_args().cur_step_num) + self.state.get('offset', 0)
    directory = Path('/state/jt-recipe-20261014/mf-grad205-state/gradients')
    if step not in (201, 202, 203, 204, 205, 206):
        raise ValueError('Only the six-step observation is supported')
    records = []
    for row in json.loads((directory / 'manifest.json').read_text()):
        deadline = time.monotonic() + 60
        while True:
            paths = [item for item in directory.glob(row['prefix'] + '_*.npy') if item not in SEEN]
            if len(paths) > 1:
                raise ValueError('More than one unconsumed native gradient: ' + row['name'])
            if paths:
                path = paths[0]
                try:
                    value = np.load(path, mmap_mode='r')
                    if list(value.shape) != row['shape']:
                        raise ValueError('Gradient shape differs')
                    if path.stat().st_size == value.offset + value.nbytes:
                        break
                    del value
                except (ValueError, EOFError, OSError):
                    pass
            if time.monotonic() >= deadline:
                raise TimeoutError('Native asynchronous TensorDump did not finish: ' + row['name'])
            time.sleep(.05)
        SEEN.add(path)
        records.append({'name': row['name'], 'shape': row['shape'], 'file': path.name,
                        'sha256': hashlib.sha256(np.ascontiguousarray(value)).hexdigest(),
                        'bytes': path.stat().st_size, 'deleted_after_hash': step != 205})
        del value
        if step != 205:
            if path.is_symlink() or path.resolve().parent != directory.resolve():
                raise ValueError('Unsafe diagnostic cleanup path')
            path.unlink()
    (self.output / ('GRADIENTS-' + str(step) + '.json')).write_text(json.dumps(records, indent=2))
    print('[P12-GRAD205]', step, len(records), 'native gradients audited', flush=True)


capture_mf_muon.ObserveMetrics.on_train_step_end = observed_end

if __name__ == '__main__':
    capture_mf_muon.main()
