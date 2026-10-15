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

"""Read-only full-training NS boundaries with bounded retention and explicit fidelity requirement."""
import hashlib
import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import mindspore as ms
from mindspore import ops
from mindformers.core.optim import muon

import capture_mf_muon

ROOT = Path('/alignment/phase12')
OUTPUT = Path('/state/jt-recipe-20261014/mf-frontier208-ns-operators')
Q_NAME = 'decoder.layers.0.self_attention.linear_q_down_proj.weight'
ROUTER_NAME = 'decoder.layers.1.mlp.router.weight'
RUN = 'mf-frontier208-ns-train'
SEEN = set()


def traced_ns(value, dim_a, dim_b, param_name):
    """Keep archived expression order, add only TensorDump observations."""
    prefix = '/state/jt-recipe-20261014/mf-frontier208-ns-operators/'
    prefix = prefix + ('q-' if param_name == Q_NAME else 'router-')
    if dim_a > dim_b:
        value = value.T
    norm = value.norm()
    denominator = norm + 1e-7
    current = value / denominator
    ops.TensorDump()(prefix + 'norm', norm.astype(ms.float32))
    ops.TensorDump()(prefix + 'denominator', denominator.astype(ms.float32))
    ops.TensorDump()(prefix + 'normalized', current.astype(ms.float32))
    for index in range(5):
        gram = current @ current.T
        polynomial = -4.7750 * gram + 2.0315 * gram @ gram
        product = polynomial @ current
        current = 3.4445 * current + product
        ops.TensorDump()(prefix + 'gram-' + str(index + 1), gram.astype(ms.float32))
        ops.TensorDump()(prefix + 'poly-' + str(index + 1), polynomial.astype(ms.float32))
        ops.TensorDump()(prefix + 'product-' + str(index + 1), product.astype(ms.float32))
        ops.TensorDump()(prefix + 'stage-' + str(index + 1), current.astype(ms.float32))
    if dim_a > dim_b:
        current = current.T
    return current


def install():
    """Generate an isolated function copy; leave the archived repository file untouched."""
    OUTPUT.mkdir(exist_ok=False)
    original = inspect.getsource(muon._apply_muon_update)
    needle = 'x = zeropower_via_newtonschulz5_2d(ns_inputs_item, dim_a, dim_b)'
    if original.count(needle) != 1:
        raise ValueError('Archived update function changed')
    replacement = ('if param_name in (Q_NAME, ROUTER_NAME):\n'
                   '                x = traced_ns(ns_inputs_item, dim_a, dim_b, param_name)\n'
                   '            else:\n'
                   '                ' + needle)
    altered = original.replace(needle, replacement)
    needle = 'ns_inputs = op_cast(gradient_fp32, mstype.bfloat16)'
    if altered.count(needle) != 1:
        raise ValueError('Archived pre-NS boundary changed')
    replacement = ('if param_name == Q_NAME:\n'
                   '        P.TensorDump()("/state/jt-recipe-20261014/mf-frontier208-ns-operators/q-pre_ns", gradient_fp32)\n'
                   '    elif param_name == ROUTER_NAME:\n'
                   '        P.TensorDump()("/state/jt-recipe-20261014/mf-frontier208-ns-operators/router-pre_ns", gradient_fp32)\n'
                   '    ' + needle)
    altered = altered.replace(needle, replacement)
    header = ('from mindformers.core.optim.muon import (P, F, mstype, _perform_allgather_op, '
              '_slice_tensor_to_shards, zeropower_via_newtonschulz5_2d, zeropower_via_newtonschulz5_3d)\n'
              'from mf_frontier208_ns_tap import traced_ns, Q_NAME, ROUTER_NAME\n\n')
    generated = OUTPUT / 'observed_native_update.py'
    generated.write_text(header + altered)
    spec = importlib.util.spec_from_file_location('observed_frontier208_native_update', generated)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    muon._apply_muon_update = module._apply_muon_update
    receipt = {'archive_function_sha256': hashlib.sha256(original.encode()).hexdigest(),
               'generated_source_sha256': hashlib.sha256(generated.read_bytes()).hexdigest(),
               'tap_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'archive_file_unchanged': True, 'read_only': True, 'teacher_values_used': False}
    (ROOT / 'evidence/frontier208-mf-ns-source.json').write_text(json.dumps(receipt, indent=2))
    original_end = capture_mf_muon.ObserveMetrics.on_train_step_end

    def end(self, context):
        original_end(self, context)
        step = int(context.original_args().cur_step_num) + self.state.get('offset', 0)
        prefixes = [label + '-' + kind for label in ['q', 'router'] for kind in (
            ['pre_ns', 'norm', 'denominator', 'normalized'] +
            [kind + '-' + str(index) for index in range(1, 6) for kind in ['gram', 'poly', 'product', 'stage']])]
        records = []
        for prefix in prefixes:
            deadline = time.monotonic() + 45
            while True:
                files = [path for path in OUTPUT.glob(prefix + '_*.npy') if path not in SEEN]
                if len(files) > 1:
                    raise ValueError('Multiple unconsumed NS observations')
                if files:
                    path = files[0]
                    try:
                        value = np.load(path, mmap_mode='r')
                        if path.stat().st_size == value.offset + value.nbytes:
                            break
                        del value
                    except (ValueError, EOFError, OSError):
                        pass
                if time.monotonic() > deadline:
                    raise TimeoutError('NS dump did not complete: ' + prefix)
                time.sleep(.05)
            SEEN.add(path)
            records.append({'prefix': prefix, 'file': path.name, 'shape': list(value.shape),
                            'sha256': hashlib.sha256(value.tobytes()).hexdigest(), 'retained': step == 208,
                            'bytes': path.stat().st_size})
            del value
            if step != 208:
                if path.is_symlink() or path.resolve().parent != OUTPUT.resolve():
                    raise ValueError('Unsafe temporary dump path')
                path.unlink()
        (self.output / ('NS_BOUNDARIES-' + str(step) + '.json')).write_text(json.dumps(records, indent=2))
        print('[P12-NS-BOUNDARIES]', step, len(records), flush=True)

    capture_mf_muon.ObserveMetrics.on_train_step_end = end
