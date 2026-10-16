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

"""Run the archived native update with framework Dump and verify actual state fidelity."""
import hashlib
import json
import os
from pathlib import Path

import mindspore as ms
import numpy as np

from mindspore import nn
from mindformers.core.optim import muon
from mindformers.core.optim.muon_utils import make_muon_fns
from compare_vocab_optimizer_trace import fidelity

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
NAMES = {'main1_shared_down': 'decoder.layers.1.mlp.shared_experts.linear_fc2.weight'}
SPLIT, MERGE = make_muon_fns([])


def digest(value: np.ndarray) -> str:
    """Hash the complete array in its logical row-major order."""
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


class NativeUpdate(nn.Cell):
    """Preserve the archived shared-expert down update as one complete dense matrix."""

    def __init__(self, label, weight, momentum, observed):
        super().__init__()
        if observed:
            raise ValueError('Only unmodified native source is allowed')
        self.param_name = NAMES[label]
        self.weight = ms.Parameter(ms.Tensor(weight), name='actual_weight')
        self.momentum = ms.Parameter(ms.Tensor(momentum), name='actual_momentum')

    def construct(self, gradient, lr, weight_decay, beta, rms):
        return muon._apply_muon_update(
            gradient, self.momentum, beta, True, self.weight, lr, weight_decay,
            rms, SPLIT, MERGE, self.param_name, 1, 1, -1, 0, '', '')



def main() -> None:
    """Observe one replay graph without source TensorDump or new training checkpoints."""
    checks = fidelity('mf-frontier396-state', 'mf-resume', 196)
    checks['exit_zero'] = (ROOT / 'runs/mf-frontier396-state.exit').read_text().strip() == '0'
    for step in [395, 396]:
        actual = next(row for row in map(json.loads, (ROOT / 'runs/mf-frontier396-state/MODEL_HASHES.jsonl').read_text().splitlines()) if row['step'] == step)
        expected = next(row for row in map(json.loads, (ROOT / 'runs/mf-native-stat397/MODEL_HASHES.jsonl').read_text().splitlines()) if row['step'] == step)
        checks['independent_model_' + str(step)] = actual == expected
    if not all(checks.values()):
        raise ValueError({'state_observer_fidelity': checks})
    config_path = Path(os.environ['MINDSPORE_DUMP_CONFIG'])
    config = json.loads(config_path.read_text())
    report_path = Path(os.environ['P12_DUMP_REPORT'])
    if report_path.exists():
        raise ValueError('The dump report destination must be new')
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', device_id=0, deterministic='ON',
                   jit_config={'jit_level': 'O1'}, ascend_config={'parallel_speed_up_json_path':
                   '/workspace/mindformers/configs/qwen3/parallel_speed_up.json'})
    state = STATE / 'mf-frontier396-state'
    rows = json.loads((ROOT / 'runs/mf-frontier396-state/FRONTIER_GRADIENTS-396.json').read_text())
    label = 'main1_shared_down'
    row = next(item for item in rows if item['name'] == NAMES[label])
    gradient = np.load(state / 'gradients' / row['file'])
    if digest(gradient) != row['sha256']:
        raise ValueError('Actual gradient changed')
    weight = np.load(state / 'state-395' / (label + '-weight.npy'))
    momentum = np.load(state / 'state-395' / (label + '-muon_m.npy'))
    native = NativeUpdate(label, weight, momentum, False)
    scalars = [ms.Tensor([value], ms.float32) for value in [1e-4, .1, .95, .2]]
    native(ms.Tensor(gradient), *scalars).asnumpy()
    result = {}
    for key, actual in [('weight', native.weight.asnumpy()), ('muon_m', native.momentum.asnumpy())]:
        expected = np.load(state / 'state-396' / (label + '-' + key + '.npy'))
        result[key] = {'exact_actual_training': digest(actual) == digest(expected),
                       'actual_sha256': digest(actual), 'expected_sha256': digest(expected)}
    report = {
        'complete': True, 'diagnostic_only': True, 'mindspore_version': ms.__version__,
        'source_tensor_dump_inserted': False, 'training_step_replayed': 396,
        'replay_iteration': 0, 'native_state_fidelity': result, 'dump_config': config,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'state_observer_fidelity': checks,
        'native_split': 'identity, one5120x3072 dense matrix',
        'archive_source_sha256': hashlib.sha256(Path(muon.__file__).read_bytes()).hexdigest(),
        'dump_config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not all(item['exact_actual_training'] for item in result.values()):
        raise ValueError('Framework Dump changed native state; do not attribute dump values to training')


if __name__ == '__main__':
    main()
