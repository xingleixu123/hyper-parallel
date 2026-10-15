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

"""Replay the archived native Muon update using real207 states and208 gradients."""
import hashlib
import json
from pathlib import Path

import numpy as np
import mindspore as ms
from mindspore import nn, ops
from mindformers.core.optim.muon import _apply_muon_update

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
OUTPUT = STATE / 'mf-frontier208-muon-replay'
NAMES = {'q': 'decoder.layers.0.self_attention.linear_q_down_proj.weight',
         'router': 'decoder.layers.1.mlp.router.weight'}


def split_identity(name, value):
    return [value]


def merge_identity(name, values):
    return values[0]


class NativeUpdate(nn.Cell):
    """Call the original archived function without modifying its implementation."""
    def __init__(self, label, weight, momentum):
        super().__init__()
        self.param_name = NAMES[label]
        self.weight = ms.Parameter(ms.Tensor(weight), name='actual_weight')
        self.momentum = ms.Parameter(ms.Tensor(momentum), name='actual_momentum')

    def construct(self, gradient, lr, weight_decay, beta, rms):
        return _apply_muon_update(gradient, self.momentum, beta, True, self.weight, lr, weight_decay,
                                  rms, split_identity, merge_identity, self.param_name,
                                  1, 1, -1, 0, '', '')


class TracedUpdate(nn.Cell):
    """Expose native source arithmetic boundaries; require final training fidelity before use."""
    def __init__(self, directory):
        super().__init__()
        self.pre_path = str(directory / 'pre_ns')
        self.norm_path = str(directory / 'normalized')
        self.stage_paths = tuple(str(directory / ('stage-' + str(i))) for i in range(1, 6))
        self.gram_paths = tuple(str(directory / ('gram-' + str(i))) for i in range(1, 6))
        self.poly_paths = tuple(str(directory / ('poly-' + str(i))) for i in range(1, 6))
        self.product_paths = tuple(str(directory / ('product-' + str(i))) for i in range(1, 6))

    def construct(self, gradient, momentum, parameter, lr, weight_decay, beta, rms):
        next_m = momentum * beta + gradient
        pre_ns = gradient + next_m * beta
        ops.TensorDump()(self.pre_path, pre_ns)
        current = pre_ns.astype(ms.bfloat16)
        current = current / (current.norm() + 1e-7)
        ops.TensorDump()(self.norm_path, current.astype(ms.float32))
        for index in range(5):
            gram = current @ current.T
            polynomial = -4.7750 * gram + 2.0315 * gram @ gram
            product = polynomial @ current
            current = 3.4445 * current + product
            ops.TensorDump()(self.gram_paths[index], gram.astype(ms.float32))
            ops.TensorDump()(self.poly_paths[index], polynomial.astype(ms.float32))
            ops.TensorDump()(self.product_paths[index], product.astype(ms.float32))
            ops.TensorDump()(self.stage_paths[index], current.astype(ms.float32))
        adjusted_ratio = ops.sqrt(ms.Tensor(5120, ms.float32)) * rms
        adjusted_lr = lr * adjusted_ratio
        result = parameter * (1 - lr * weight_decay) - adjusted_lr * current
        return result, next_m


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    OUTPUT.mkdir(exist_ok=False)
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', device_id=0, deterministic='ON',
                   jit_config={'jit_level': 'O1'}, ascend_config={'parallel_speed_up_json_path':
                   '/workspace/mindformers/configs/qwen3/parallel_speed_up.json'})
    state = STATE / 'mf-frontier208-state'
    gradient_rows = json.loads((ROOT / 'runs/mf-frontier208-state/FRONTIER_GRADIENTS-208.json').read_text())
    records = []
    scalars = [ms.Tensor([value], ms.float32) for value in [1e-4, .1, .95, .2]]
    for label, name in NAMES.items():
        weight = np.load(state / 'state-207' / (label + '-weight.npy'))
        momentum = np.load(state / 'state-207' / (label + '-muon_m.npy'))
        after_weight = np.load(state / 'state-208' / (label + '-weight.npy'))
        after_momentum = np.load(state / 'state-208' / (label + '-muon_m.npy'))
        row = next(row for row in gradient_rows if row['name'] == name)
        gradient = np.load(state / 'gradients' / row['file'])
        if digest(gradient) != row['sha256']:
            raise ValueError('Actual gradient changed')
        directory = OUTPUT / label
        directory.mkdir()
        native = NativeUpdate(label, weight, momentum)
        native(ms.Tensor(gradient), *scalars).asnumpy()
        actual_weight, actual_momentum = native.weight.asnumpy(), native.momentum.asnumpy()
        record = {'label': label, 'native_weight_exact_training': np.array_equal(actual_weight, after_weight),
                  'native_momentum_exact_training': np.array_equal(actual_momentum, after_momentum),
                  'native_weight_sha256': digest(actual_weight), 'expected_weight_sha256': digest(after_weight)}
        traced = TracedUpdate(directory)
        result = traced(ms.Tensor(gradient), ms.Tensor(momentum), ms.Tensor(weight), *scalars)
        observed_weight, observed_momentum = [item.asnumpy() for item in result]
        record.update(traced_weight_exact_training=np.array_equal(observed_weight, after_weight),
                      traced_momentum_exact_training=np.array_equal(observed_momentum, after_momentum),
                      traced_weight_sha256=digest(observed_weight))
        records.append(record)
        print(json.dumps(record), flush=True)
    report = {'complete': True, 'diagnostic_only': True, 'records': records,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'warning': 'Only traces with full native and instrumented output fidelity are evidence of actual training.'}
    (ROOT / 'evidence/frontier208-mf-muon-replay.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
