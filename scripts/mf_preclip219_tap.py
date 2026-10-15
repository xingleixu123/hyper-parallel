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

"""Read native preclip inputs without editing the archived MF implementation."""
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from mindspore import ops
from mindformers.core.clip_grad import ClipGradNorm
import mindformers.wrapper.wrapper as wrapper_module

import capture_mf_muon
from frontier212_array_io import describe

CONTEXT: dict[str, Any] = {}
SEEN: set[Path] = set()
RETAIN_RAW = False


class ObservedClip(ClipGradNorm):
    """Dump unchanged inputs and call the original clipping arithmetic."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.preclip_probe_paths = tuple(str(CONTEXT['directory'] / row['prefix']) for row in CONTEXT['rows'])

    def construct(self, values: Any) -> Any:
        for index in range(len(self.preclip_probe_paths)):
            ops.TensorDump()(self.preclip_probe_paths[index], values[index])
        return super().construct(values)


def attach(optimizer: Any, destination: Any, max_elements: Any = None) -> None:
    """Validate all46 native names and shapes before instrumenting the wrapper."""
    if max_elements is not None:
        raise ValueError('This observer requires all46 native gradients')
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=False)
    order_path = Path('/alignment/phase12/configs/mf-gradient-layout-order.json')
    order = json.loads(order_path.read_text())
    parameters = list(optimizer.parameters)
    actual = [{'name': value.name, 'shape': list(value.shape)} for value in parameters]
    if actual != order or len(actual) != 46:
        raise ValueError('Native optimizer order differs from the verified MF layout')
    rows = [{**row, 'prefix': f'{index:04d}'} for index, row in enumerate(actual)]
    CONTEXT.update(directory=directory, rows=rows)
    (directory / 'manifest.json').write_text(json.dumps(rows, indent=2))
    wrapper_module.ClipGradNorm = ObservedClip


def install() -> None:
    """Consume each step completely and retain only218/219 full hashes, removing raw step copies."""
    original_end = capture_mf_muon.ObserveMetrics.on_train_step_end

    def end(self: Any, run_context: Any) -> None:
        original_end(self, run_context)
        step = int(run_context.original_args().cur_step_num) + self.state.get('offset', 0)
        records, removed_bytes = [], 0
        directory = CONTEXT['directory']
        for row in CONTEXT['rows']:
            deadline = time.monotonic() + 120
            while True:
                matches = [path for path in directory.glob(row['prefix'] + '_*.npy') if path not in SEEN]
                if len(matches) > 1:
                    raise ValueError('Multiple unconsumed preclip arrays')
                if matches:
                    path = matches[0]
                    try:
                        array = np.load(path, mmap_mode='r')
                        complete = list(array.shape) == row['shape'] and path.stat().st_size == array.offset + array.nbytes
                        if complete:
                            break
                        del array
                    except (ValueError, EOFError, OSError):
                        pass
                if time.monotonic() > deadline:
                    raise TimeoutError('Native preclip dump did not finish: ' + row['name'])
                time.sleep(.1)
            SEEN.add(path)
            if step in (218, 219):
                records.append({'name': row['name'], 'file': path.name, 'retained': RETAIN_RAW, **describe(array)})
            del array
            if not RETAIN_RAW:
                if path.is_symlink() or path.resolve().parent != directory.resolve():
                    raise ValueError('Unsafe task-owned temporary array path')
                removed_bytes += path.stat().st_size
                path.unlink()
        report = {'step': step, 'complete': True, 'arrays': 46, 'records': records,
                  'deleted_logical_bytes': removed_bytes, 'retained': RETAIN_RAW,
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'helper_sha256': hashlib.sha256((Path(__file__).parent / 'frontier212_array_io.py').read_bytes()).hexdigest(),
                  'readonly': True}
        (self.output / ('PRECLIP-' + str(step) + '.json')).write_text(json.dumps(report, indent=2))
        print('[P12-PRECLIP219]', step, '46 native arrays consumed', flush=True)

    capture_mf_muon.ObserveMetrics.on_train_step_end = end
