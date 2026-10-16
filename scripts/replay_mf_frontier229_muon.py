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

"""Replay actual229 archived Muon updates; require complete native/traced state fidelity."""
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import mindspore as ms
from mindspore import nn
from mindformers.core.optim import muon
from mindformers.core.optim.muon_utils import make_muon_fns

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
OUTPUT = STATE / 'mf-frontier229-muon-replay'
NAMES = {'mtp_q_up': 'mtp.layers.0.transformer_layer.self_attention.linear_q_up_proj.weight'}
SPLIT, MERGE = make_muon_fns([{'patterns': ['*self_attention.linear_q_up_proj.weight'],
                             'kind': 'periodic', 'parts': (128, 64, 128)}])
OBSERVED = None


def prepare_observer():
    """Copy original expressions and insert dumps into an isolated generated module."""
    functions = []
    for dimension in [2, 3]:
        function = getattr(muon, 'zeropower_via_newtonschulz5_' + str(dimension) + 'd')
        source = inspect.getsource(function)
        source = source.replace('(x, dim_a, dim_b):', '(x, dim_a, dim_b, tag):')
        if dimension == 2:
            original = 'x = x / (x.norm() + 1e-7)'
            replaced = ('P.TensorDump()(tag + "pre_norm", P.Cast()(x, mstype.float32))\n'
                        '    norm = x.norm()\n    denominator = norm + 1e-7\n    x = x / denominator')
        else:
            original = 'x = x / P.ExpandDims()(P.ExpandDims()((x.norm(dim=(1, 2)) + 1e-7), 1), 1)'
            replaced = ('norm = x.norm(dim=(1, 2))\n    denominator = norm + 1e-7\n'
                        '    x = x / P.ExpandDims()(P.ExpandDims()(denominator, 1), 1)')
        assert source.count(original) == 1
        replaced += ('\n    P.TensorDump()(tag + "norm", P.Cast()(norm, mstype.float32))'
                     '\n    P.TensorDump()(tag + "denominator", P.Cast()(denominator, mstype.float32))'
                     '\n    P.TensorDump()(tag + "normalized", P.Cast()(x, mstype.float32))')
        source = source.replace(original, replaced)
        source = source.replace('    return x', '    P.TensorDump()(tag + "ns_output", P.Cast()(x, mstype.float32))\n    return x')
        functions.append(source)
    source = inspect.getsource(muon._apply_muon_update)
    source = source.replace('    ns_inputs = op_cast(gradient_fp32, mstype.bfloat16)',
                            '    label = "mtp_q_up"\n'
                            '    prefix = "' + str(OUTPUT) + '/" + label\n'
                            '    P.TensorDump()(prefix + "-pre_ns", gradient_fp32)\n'
                            '    ns_inputs = op_cast(gradient_fp32, mstype.bfloat16)')
    source = source.replace('for ns_inputs_item in ns_inputs_list:',
                            'for piece, ns_inputs_item in enumerate(ns_inputs_list):\n'
                            '        tag = prefix + "-" + str(piece) + "-"')
    for dimension in [2, 3]:
        source = source.replace('zeropower_via_newtonschulz5_' + str(dimension) + 'd(ns_inputs_item, dim_a, dim_b)',
                                'zeropower_via_newtonschulz5_' + str(dimension) + 'd(ns_inputs_item, dim_a, dim_b, tag)')
    functions.append(source)
    generated = OUTPUT / 'observed_native_update.py'
    generated.write_text('from mindformers.core.optim.muon import (P, F, mstype, _perform_allgather_op, _slice_tensor_to_shards)\n\n' + '\n\n'.join(functions))
    spec = importlib.util.spec_from_file_location('frontier229_observed_native', generated)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._apply_muon_update


class NativeUpdate(nn.Cell):
    """Run original or observed original source with identical actual states."""
    def __init__(self, label, weight, momentum, observed):
        super().__init__()
        self.param_name = NAMES[label]
        self.weight = ms.Parameter(ms.Tensor(weight), name='actual_weight')
        self.momentum = ms.Parameter(ms.Tensor(momentum), name='actual_momentum')
        self.observed = observed

    def construct(self, gradient, lr, weight_decay, beta, rms):
        if self.observed:
            return OBSERVED(gradient, self.momentum, beta, True, self.weight, lr, weight_decay,
                            rms, SPLIT, MERGE, self.param_name, 1, 1, -1, 0, '', '')
        return muon._apply_muon_update(gradient, self.momentum, beta, True, self.weight, lr, weight_decay,
                                      rms, SPLIT, MERGE, self.param_name, 1, 1, -1, 0, '', '')


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    global OBSERVED
    OUTPUT.mkdir(exist_ok=False)
    OBSERVED = prepare_observer()
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', device_id=0, deterministic='ON',
                   jit_config={'jit_level': 'O1'}, ascend_config={'parallel_speed_up_json_path':
                   '/workspace/mindformers/configs/qwen3/parallel_speed_up.json'})
    state = STATE / 'mf-frontier229-state'
    gradient_rows = json.loads((ROOT / 'runs/mf-frontier229-state/FRONTIER_GRADIENTS-229.json').read_text())
    records = []
    scalars = [ms.Tensor([value], ms.float32) for value in [1e-4, .1, .95, .2]]
    for label, name in NAMES.items():
        weight = np.load(state / 'state-228' / (label + '-weight.npy'))
        momentum = np.load(state / 'state-228' / (label + '-muon_m.npy'))
        after_weight = np.load(state / 'state-229' / (label + '-weight.npy'))
        after_momentum = np.load(state / 'state-229' / (label + '-muon_m.npy'))
        row = next(row for row in gradient_rows if row['name'] == name)
        gradient = np.load(state / 'gradients' / row['file'])
        if digest(gradient) != row['sha256']:
            raise ValueError('Actual gradient changed')
        record = {'label': label, 'expected_weight_sha256': digest(after_weight)}
        for observed in [False, True]:
            native = NativeUpdate(label, weight, momentum, observed)
            native(ms.Tensor(gradient), *scalars).asnumpy()
            actual_weight, actual_momentum = native.weight.asnumpy(), native.momentum.asnumpy()
            key = 'observed' if observed else 'native'
            record[key + '_weight_exact_training'] = np.array_equal(actual_weight, after_weight)
            record[key + '_momentum_exact_training'] = np.array_equal(actual_momentum, after_momentum)
            record[key + '_weight_sha256'] = digest(actual_weight)
        records.append(record)
        print(json.dumps(record), flush=True)
    report = {'complete': True, 'diagnostic_only': True, 'records': records,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'archive_file_sha256': hashlib.sha256(Path(muon.__file__).read_bytes()).hexdigest(),
              'generated_sha256': hashlib.sha256((OUTPUT / 'observed_native_update.py').read_bytes()).hexdigest(),
              'warning': 'Only full native and observed final state fidelity permits attribution to training.'}
    (ROOT / 'evidence/frontier229-mf-muon-replay.json').write_text(json.dumps(report, indent=2))
    assert all(row[key + '_weight_exact_training'] and row[key + '_momentum_exact_training']
               for row in records for key in ['native', 'observed'])


if __name__ == '__main__':
    main()
