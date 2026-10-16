# Copyright 2026 Huawei Technologies Co., Ltd
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

"""CPU contracts for the bounded 200-step diagnostic observer and real configs."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

# The experiment directly tests Torch derivatives, as required by the current native-Torch convention.
import torch  # pylint: disable=forbidden-backend-import
from torch.nn import functional  # pylint: disable=forbidden-backend-import
import yaml

from linear_dgrad_long_observer import LongLinearDgrad

from hyper_parallel.trainer.config.parser import parse_training_args


ROOT = Path('/work')


def derivative_case(flags: tuple[bool, bool, bool], dtype: torch.dtype) -> dict:
    """Use an FP64 input-gradient oracle and native weight/bias derivatives."""
    torch.manual_seed(20261016)
    inputs = torch.randn(2, 7, 23, dtype=dtype, requires_grad=flags[0])
    weight = torch.randn(31, 23, dtype=dtype, requires_grad=flags[1])
    bias = torch.randn(31, dtype=dtype, requires_grad=flags[2])
    upstream = torch.randn(2, 7, 31, dtype=dtype)
    targets = [value for value in (inputs, weight, bias) if value.requires_grad]
    native = functional.linear(inputs, weight, bias)
    expected = torch.autograd.grad(native, targets, upstream) if targets else ()
    original = functional.linear
    observer = LongLinearDgrad()
    try:
        actual = functional.linear(inputs, weight, bias)
        gradients = torch.autograd.grad(actual, targets, upstream) if targets else ()
        coverage = observer.finish_step()
        empty = observer.finish_step()
    finally:
        observer.close()
    if functional.linear is not original or not torch.equal(actual, native):
        raise AssertionError('Forward or patch restoration changed')
    for target, gradient, reference in zip(targets, gradients, expected, strict=True):
        if target is inputs and dtype == torch.bfloat16:
            reference = (upstream.double() @ weight.detach().double()).to(dtype)
        torch.testing.assert_close(gradient, reference, rtol=0, atol=0)
    if empty != {'counts': {}, 'shapes': []}:
        raise AssertionError('Step metadata was retained after it was consumed')
    if dtype == torch.bfloat16:
        counts = coverage['counts']
        if counts.get('actual_backward', 0) != int(bool(targets)):
            raise AssertionError('Actual derivative coverage differs from the autograd branches')
        for name, needed in zip(('fp32_dgrad', 'native_wgrad', 'native_bias_grad'), flags, strict=True):
            if counts.get(name, 0) != int(needed):
                raise AssertionError('Frozen-input derivative coverage is incorrect')
    elif coverage['counts'] != {'bypassed_forward': 1}:
        raise AssertionError('Non-BF16 operands did not bypass the diagnostic')
    return {'flags': list(flags), 'dtype': str(dtype), 'exact_oracle_match': True,
            'step_metadata_cleared': True, 'coverage': coverage['counts']}


def main() -> None:
    """Validate math, bounded counters and both 200-step configs without device initialization."""
    rows = [derivative_case(flags, torch.bfloat16) for flags in itertools.product((False, True), repeat=3)]
    rows.append(derivative_case((True, True, True), torch.float32))
    configs = []
    for layout, reference in (('fsdp8', 'r129-f637-fp32-fsdp8-0-200-r4'),
                              ('cp2', 'r129-f637-fp32-cp2-200')):
        case = f'r129-f637-dgrad-{layout}-fp32-200-v1'
        path = ROOT / 'configs' / (case + '.yaml')
        raw = yaml.safe_load(path.read_bytes())
        if raw['training']['train_iters'] != 200:
            raise AssertionError('The candidate no longer covers 200 consecutive optimizer steps')
        actual = parse_training_args([str(path)]).to_dict()
        expected = json.loads((ROOT / 'runs' / reference / 'config.resolved.json').read_bytes())
        expected['checkpoint']['save_ckpt'] = False
        expected['checkpoint']['checkpoint_dir'] = '/work/checkpoints/' + case
        if actual != expected:
            raise AssertionError('Candidate altered more than the disabled checkpoint destination: ' + case)
        configs.append({'case': case, 'status': 'PASS', 'reference': reference,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    initialized = hasattr(torch, 'npu') and torch.npu.is_initialized()
    if initialized or torch.cuda.is_initialized():
        raise RuntimeError('Device-free validation initialized an accelerator')
    names = ['check_dgrad_long_cpu.py', 'linear_dgrad_long_observer.py', 'capture_dgrad_200_training.py',
             'linear_dgrad_experiment.py', 'linear_dgrad_trace_experiment.py']
    result = {'status': 'CPU_ONLY_PASS', 'derivative_cases': rows, 'resolved_configs': configs,
              'npu_initialized': initialized, 'torch': torch.__version__,
              'source_sha256': {name: hashlib.sha256((ROOT / 'tools' / name).read_bytes()).hexdigest()
                                for name in names},
              'scope': 'Small CPU derivative and metadata contracts; no NPU or 200-step training acceptance'}
    path = ROOT / 'evidence/f637-dgrad-long-cpu-v1.json'
    with path.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
