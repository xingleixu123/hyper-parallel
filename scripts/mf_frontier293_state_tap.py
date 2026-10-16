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

"""Observe the native main-layer0 packed gate/up gradient and Muon state without changing arithmetic."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import capture_mf_muon

TARGETS = {'main0_fc1': 'decoder.layers.0.mlp.linear_fc1.weight'}
SHAPES = {'main0_fc1': (24576, 5120)}
RUN = 'mf-frontier293-state'
SEEN = set()


def record_array(directory, label, name, tensor):
    """Preserve a small actual state vector with its original dtype and hash."""
    array = tensor.asnumpy()
    if tuple(array.shape) != SHAPES[label]:
        raise ValueError('Unexpected target Muon matrix')
    filename = label + '-' + name + '.npy'
    np.save(directory / filename, array, allow_pickle=False)
    return {'label': label, 'kind': name, 'file': filename, 'shape': list(array.shape), 'dtype': str(array.dtype),
            'sha256': hashlib.sha256(array.tobytes()).hexdigest()}


def install():
    original_end = capture_mf_muon.ObserveMetrics.on_train_step_end

    def end(self, run_context):
        original_end(self, run_context)
        step = int(run_context.original_args().cur_step_num) + self.state.get('offset', 0)
        root = Path('/state/jt-recipe-20261014') / RUN
        directory = root / 'gradients'
        records = []
        for row in json.loads((directory / 'manifest.json').read_text()):
            deadline = time.monotonic() + 45
            while True:
                paths = [path for path in directory.glob(row['prefix'] + '_*.npy') if path not in SEEN]
                if len(paths) > 1:
                    raise ValueError('Multiple unconsumed gradient files')
                if paths:
                    path = paths[0]
                    try:
                        value = np.load(path, mmap_mode='r')
                        if list(value.shape) == row['shape'] and path.stat().st_size == value.offset + value.nbytes:
                            break
                        del value
                    except (ValueError, EOFError, OSError):
                        pass
                if time.monotonic() > deadline:
                    raise TimeoutError('Native selected gradient dump did not finish')
                time.sleep(.05)
            SEEN.add(path)
            keep = row['name'] in TARGETS.values() and step == 293
            records.append({'name': row['name'], 'shape': row['shape'], 'file': path.name,
                            'sha256': hashlib.sha256(value.tobytes()).hexdigest(), 'retained': keep})
            del value
            if not keep:
                if path.is_symlink() or path.resolve().parent != directory.resolve():
                    raise ValueError('Unsafe observation cleanup path')
                path.unlink()
        (self.output / ('FRONTIER_GRADIENTS-' + str(step) + '.json')).write_text(json.dumps(records, indent=2))
        if step in (292, 293):
            optimizer = self.state['optimizer']
            directory = root / ('state-' + str(step))
            directory.mkdir(exist_ok=False)
            state_rows = []
            for label, target in TARGETS.items():
                matches = [(index, parameter) for index, parameter in enumerate(optimizer.parameters)
                           if parameter.name == target]
                if len(matches) != 1:
                    raise ValueError('Native target parameter not unique')
                index, parameter = matches[0]
                if not optimizer.use_muon[index]:
                    raise ValueError('Expected Muon target state')
                state_rows.extend(record_array(directory, label, kind, value) for kind, value in [
                    ('weight', parameter), ('muon_m', optimizer.muon_m[index])])
            report = {'step': step, 'targets': TARGETS, 'records': state_rows,
                      'optimizer_step': int(optimizer.global_step.asnumpy().reshape(-1)[0]),
                      'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'read_only': True, 'teacher_values_used': False}
            (self.output / ('FRONTIER_STATE-' + str(step) + '.json')).write_text(json.dumps(report, indent=2))
        print('[P12-FRONTIER293-STATE]', step, 'read-only observation complete', flush=True)

    capture_mf_muon.ObserveMetrics.on_train_step_end = end
