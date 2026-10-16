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

"""Observe actual full-shape CE arithmetic, retaining four rows only at step226."""
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

import numpy as np
import torch

from capture_hp_native import Observer
import ce_sum_tree_factor as factor

CONTEXT = {'step': 201, 'calls': 0, 'records': []}


def save(kind: str, tensor: torch.Tensor) -> None:
    """Sample already-computed tensors without substituting reduced-shape arithmetic."""
    if CONTEXT['step'] != 226:
        return
    selected = tensor.detach()
    if selected.ndim == 2 and selected.shape[0] == 4096:
        selected = selected.index_select(0, CONTEXT['rows'].to(selected.device))
    array = selected.float().cpu().numpy()
    name = f"call{CONTEXT['calls']}_{kind}.npy"
    path = CONTEXT['directory'] / name
    if path.exists():
        raise ValueError('Duplicate CE observation: ' + name)
    np.save(path, array)
    CONTEXT['records'].append({'call': CONTEXT['calls'], 'kind': kind, 'file': name,
                               'shape': list(array.shape), 'dtype': str(tensor.dtype),
                               'sha256': hashlib.sha256(array.tobytes()).hexdigest()})


def start_call(inputs: torch.Tensor, targets: torch.Tensor) -> None:
    """Associate one actual CE backward with its inputs; branch is inferred afterward."""
    CONTEXT['calls'] += 1
    save('logits', inputs)
    save('targets', targets)


def replace_once(source: str, old: str, new: str) -> str:
    """Fail closed when the frozen arithmetic source no longer matches the observer."""
    if source.count(old) != 1:
        raise ValueError('CE observer anchor is absent or nonunique: ' + old)
    return source.replace(old, new)


def install() -> None:
    """Insert read-only observations into the existing factor without editing its file."""
    forward_source = inspect.getsource(factor.log_probabilities)
    backward_source = textwrap.dedent(inspect.getsource(factor.TreeBackwardCE.backward))
    forward = replace_once(forward_source, '    total = tiled_sum(shifted.exp())',
                           '    exponent = shifted.exp()\n    total = tiled_sum(exponent)')
    forward = replace_once(forward, '    return shifted - total.log()',
                           "    log_total = total.log()\n    result = shifted - log_total\n"
                           "    ce226_save('shifted', shifted)\n    ce226_save('exponent', exponent)\n"
                           "    ce226_save('sum', total)\n    ce226_save('log_sum', log_total)\n"
                           "    ce226_save('log_probability', result)\n    return result")
    backward = backward_source.replace('@staticmethod\n', '', 1)
    backward = replace_once(backward, '    inputs, targets = ctx.saved_tensors',
                            '    inputs, targets = ctx.saved_tensors\n    ce226_start(inputs, targets)')
    backward = replace_once(backward, '    valid = targets != ctx.ignore_index',
                            "    ce226_save('probability', result)\n    valid = targets != ctx.ignore_index")
    backward = replace_once(backward, '    result = (result - hot) * (valid[:, None] * gradient)',
                            "    scaled = valid[:, None] * gradient\n    result = (result - hot) * scaled\n"
                            "    ce226_save('one_hot', hot)\n    ce226_save('mask', valid)\n"
                            "    ce226_save('dout', scaled)\n    ce226_save('gradient', gradient)\n"
                            "    ce226_save('dx_fp32', result)\n    ce226_save('dx_bf16', result.bfloat16())")
    namespace = factor.__dict__
    namespace.update(ce226_save=save, ce226_start=start_call)
    exec(compile(forward, __file__ + ':log_probabilities', 'exec'), namespace)
    exec(compile(backward, __file__ + ':backward', 'exec'), namespace)
    factor.TreeBackwardCE.backward = staticmethod(namespace['backward'])
    original_begin, original_end = Observer.on_train_begin, Observer.on_step_end

    def begin(self, state, **kwargs):
        original_begin(self, state, **kwargs)
        source = Path('/alignment/phase12/evidence/head226-selected-coordinates.json')
        selection = json.loads(source.read_text())
        directory = Path('/state/jt-recipe-20261014') / self.output.name / 'ce'
        directory.mkdir(parents=True, exist_ok=False)
        CONTEXT.update(step=int(state.global_step) + 1, calls=0, records=[], directory=directory,
                       rows=torch.tensor(selection['token_rows']))
        (self.output / 'CE226_OBSERVER.json').write_text(json.dumps({
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'factor_sha256': hashlib.sha256(Path(factor.__file__).read_bytes()).hexdigest(),
            'selection': selection, 'teacher_values_used': False,
            'original_log_probabilities': forward_source, 'original_backward': backward_source,
            'instrumented_log_probabilities': forward, 'instrumented_backward': backward}, indent=2))

    def end(self, state, **kwargs):
        original_end(self, state, **kwargs)
        if CONTEXT['calls'] != 2:
            raise ValueError('Expected two actual CE backwards')
        with (self.output / 'CE226_CALLS.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': int(state.global_step), 'calls': CONTEXT['calls']}) + '\n')
        if int(state.global_step) == 226:
            (self.output / 'CE226_OPERANDS.json').write_text(json.dumps({
                'step': 226, 'records': CONTEXT['records']}, indent=2))
        CONTEXT.update(step=int(state.global_step) + 1, calls=0)

    Observer.on_train_begin, Observer.on_step_end = begin, end
