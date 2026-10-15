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

"""Read layer0 native attention inputs, packed boundaries, mask and output."""
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import mindspore as ms
from mindspore import ops
from mindformers.parallel_core.training_graph.transformer.flash_attention import FlashAttention

import capture_mf_muon

CONTEXT = {}
SEEN = set()


class ObservedAttention(FlashAttention):
    """Preserve the native FlashAttention construct and original tensor arguments."""
    def construct(self, query, key, value, attention_mask, attn_mask_type=None,
                  attention_bias=None, packed_seq_params=None, alibi_mask=None,
                  prefix=None, padding_mask=None, actual_seq_qlen=None, actual_seq_kvlen=None):
        ops.TensorDump()(self.probe_path + '_query', query.astype(ms.float32))
        ops.TensorDump()(self.probe_path + '_key', key.astype(ms.float32))
        ops.TensorDump()(self.probe_path + '_value', value.astype(ms.float32))
        ops.TensorDump()(self.probe_path + '_mask', attention_mask.astype(ms.float32))
        ops.TensorDump()(self.probe_path + '_qlen', actual_seq_qlen.astype(ms.float32))
        ops.TensorDump()(self.probe_path + '_kvlen', actual_seq_kvlen.astype(ms.float32))
        result = super().construct(query, key, value, attention_mask, attn_mask_type,
                                   attention_bias, packed_seq_params, alibi_mask, prefix,
                                   padding_mask, actual_seq_qlen, actual_seq_kvlen)
        ops.TensorDump()(self.probe_path + '_output', result.astype(ms.float32))
        return result


def attach(network: Any, directory: Path) -> None:
    """Limit full dumps to the first attention layer and retain only217."""
    directory.mkdir(parents=True, exist_ok=False)
    selected = [(name, cell) for name, cell in network.cells_and_names()
                if type(cell) is FlashAttention and name.startswith('model.decoder.layers.0.')]
    if len(selected) != 1:
        raise ValueError('Expected one native attention module')
    name, module = selected[0]
    if module.input_layout != 'TND' or module.lower_triangle_mask is not None:
        raise ValueError('Unexpected actual native attention route')
    options = {key: module.flash_attention.attrs.get(key) for key in [
        'head_num', 'input_layout', 'sparse_mode', 'keep_prob', 'scale_value', 'pre_tokens', 'next_tokens', 'inner_precise']}
    module.__class__ = ObservedAttention
    module.probe_path = str(directory / 'attention')
    kinds = ['query', 'key', 'value', 'mask', 'qlen', 'kvlen', 'output']
    CONTEXT.update(directory=directory, modules=[{'name': name, 'prefix': 'attention', 'kinds': kinds}])
    (directory / 'manifest.json').write_text(json.dumps({'modules': CONTEXT['modules'], 'options': options,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))


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
                suffix = '_' + kind
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
    with (self.output / 'ATTENTION217.jsonl').open('a') as stream:
        stream.write(json.dumps({'step': step, 'records': records}) + '\n')


capture_mf_muon.attach_taps = attach
capture_mf_muon.ObserveMetrics.on_train_step_end = observed_end
