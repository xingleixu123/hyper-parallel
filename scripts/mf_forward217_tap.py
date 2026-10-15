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

"""Observe complete native forward boundaries and prune previous-step arrays."""
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import mindspore as ms
from mindspore import ops

import capture_mf_muon
import mf_activation_taps as taps

CONTEXT = {}
SEEN = set()


def save_full(path: str, value: Any) -> None:
    """Preserve native values while observing their FP32 representation."""
    ops.TensorDump()(path, value.astype(ms.float32))


def attach(network: Any, directory: Path) -> None:
    """Select norms, attention output projections and two MoE outputs only."""
    directory.mkdir(parents=True, exist_ok=False)
    taps.save_sample = save_full
    replacements = {taps.FusedRMSNorm: taps.ObservedFusedRMSNorm,
                    taps.RMSNorm: taps.ObservedRMSNorm, taps.MoELayer: taps.ObservedMoE}
    records = []
    for name, module in network.cells_and_names():
        original = type(module)
        replacement = replacements.get(original)
        if name.endswith('self_attention.linear_proj'):
            replacement = {taps.RowParallelLinear: taps.ObservedRow,
                           taps.SequenceParallelLinear: taps.ObservedSequenceLinear}.get(original)
        if replacement is None:
            continue
        module.__class__ = replacement
        prefix = name.replace('.', '_')
        module.probe_path = str(directory / prefix)
        kinds = ['input', 'output'] if original is not taps.MoELayer else ['output']
        records.append({'name': name, 'prefix': prefix, 'kinds': kinds, 'original': original.__name__})
    if len(records) != 21:
        raise ValueError(('Expected21 complete boundaries', records))
    CONTEXT.update(directory=directory, modules=records)
    (directory / 'manifest.json').write_text(json.dumps({'modules': records,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'helper_sha256': hashlib.sha256(Path(taps.__file__).read_bytes()).hexdigest()}, indent=2))


ORIGINAL_END = capture_mf_muon.ObserveMetrics.on_train_step_end


def observed_end(self: Any, run_context: Any) -> None:
    """Hash each executed observation and promptly remove unneeded full arrays."""
    ORIGINAL_END(self, run_context)
    step = int(run_context.original_args().cur_step_num) + self.state.get('offset', 0)
    records = []
    for module in CONTEXT['modules']:
        for kind in module.get('kinds', ('x', 'y', 'dy', 'dx')):
            deadline = time.monotonic() + 30
            while True:
                suffix = '_input' if kind == 'input' else ''
                paths = sorted(set(CONTEXT['directory'].glob(module['prefix'] + suffix + '_float32_*.npy')) - SEEN)
                try:
                    arrays = [np.load(path, mmap_mode='r') for path in paths]
                    ready = paths and all(path.stat().st_size == array.offset + array.nbytes
                                          for path, array in zip(paths, arrays))
                except (ValueError, EOFError, OSError):
                    ready = False
                if ready:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(module['name'] + ':' + kind)
                time.sleep(.05)
            hashes = {hashlib.sha256(array.tobytes()).hexdigest() for array in arrays}
            if len(hashes) != 1:
                raise ValueError('Recomputed norm values differ')
            for path, array in zip(paths, arrays):
                records.append({'name': module['name'], 'kind': kind, 'file': path.name,
                                'shape': list(array.shape), 'sha256': next(iter(hashes)), 'retained': step == 217,
                                'logical_bytes': path.stat().st_size, 'physical_bytes': path.stat().st_blocks * 512})
                SEEN.add(path)
            del arrays, array
            if step != 217:
                for path in paths:
                    if path.is_symlink() or path.resolve().parent != CONTEXT['directory'].resolve():
                        raise ValueError('Unsafe temporary branch observation path')
                    path.unlink()
    with (self.output / 'FORWARD217.jsonl').open('a') as stream:
        stream.write(json.dumps({'step': step, 'records': records}) + '\n')


capture_mf_muon.attach_taps = attach
capture_mf_muon.ObserveMetrics.on_train_step_end = observed_end
