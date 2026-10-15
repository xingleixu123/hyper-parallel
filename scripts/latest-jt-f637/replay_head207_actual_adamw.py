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

"""Replay the actual installed archive AdamW factor with fixed206 state and two real207 gradients."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch_npu

import adamw_archive_scope_factor
import capture_hp_native
import hyper_parallel.core.optimizer.adamw as adamw_module
from head_moments206_observation import digest
from hyper_parallel.trainer.runtime.random import enable_full_determinism

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
RUN = 'head207-actual-adamw-replay'
SHAPE = (152064, 5120)


def state_hashes(run: str, native: bool) -> dict:
    """Load independently observed complete postupdate state identities."""
    report = json.loads((ROOT / 'runs' / run / 'HEAD_MOMENTS207_HASH.json').read_text())
    result = {row['kind']: row['sha256'] for row in report['records']}
    if native:
        result['weight'] = report['model_fp32_sha256']
    else:
        rows = [json.loads(line) for line in (ROOT / 'runs' / run / 'MODEL_HASHES.jsonl').read_text().splitlines()]
        result['weight'] = next(item['fp32_sha256'] for row in rows if row['step'] == 207
                                for item in row['records'] if item['name'] == 'lm_head.weight')
    return result


def copy_to_device(path: Path, destination: torch.Tensor, expected: str) -> None:
    """Verify every complete source before loading the actual NPU update buffers."""
    array = np.load(path, mmap_mode='r')
    if digest(array) != expected:
        raise ValueError('Replay input changed: ' + str(path))
    tensor = torch.from_numpy(np.array(array, copy=True)).to(destination.device)
    destination.copy_(tensor)


@torch.no_grad()
def main() -> None:
    """First reproduce all HP207 state hashes, then change only the gradient to native MF."""
    comparison = json.loads((ROOT / 'evidence/head-moments207-gradient-comparison.json').read_text())
    if not comparison['complete'] or not all(all(row.values()) for row in comparison['fidelity'].values()):
        raise ValueError('Actual gradient observations are not audited')
    torch.npu.set_device(0)
    enable_full_determinism(42, close_matmul_k_shift=False)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    output = ROOT / 'runs' / RUN
    output.mkdir(exist_ok=False)
    model = torch.nn.Module()
    model.register_parameter('head', torch.nn.Parameter(torch.empty(SHAPE, device='npu', dtype=torch.float32),
                                                      requires_grad=False))
    model.named_parameters = lambda: iter([('lm_head.weight', model.head)])
    current = {'weight': model.head, 'exp_avg': torch.empty_like(model.head),
               'exp_avg_sq': torch.empty_like(model.head)}
    gradient = torch.empty_like(model.head)
    capture_hp_native.Observer.on_train_begin = lambda *args, **kwargs: None
    os.environ['P12_ADAMW_ARCHIVE_SCOPE'] = 'all'
    adamw_archive_scope_factor.install()
    observer = SimpleNamespace(output=output, trainer=SimpleNamespace(model=model))
    capture_hp_native.Observer.on_train_begin(observer, SimpleNamespace(global_step=206))
    factor_sha = hashlib.sha256(Path(adamw_archive_scope_factor.__file__).read_bytes()).hexdigest()
    expected_factor = json.loads((ROOT / 'runs' / comparison['hp_run'] / 'ADAMW_ARCHIVE_ARITHMETIC.json').read_text())
    if factor_sha != expected_factor['source_sha256']:
        raise ValueError('Replay factor differs from actual training source')
    moment_report = json.loads((ROOT / 'evidence/head-moments206-comparison.json').read_text())
    model_report = json.loads((ROOT / 'evidence/head206-model-comparison.json').read_text())
    initial = {'weight': (STATE / 'mf-head206-state-r2/head206.npy',
                          model_report['model_records']['mf']['fp32_sha256'])}
    initial.update({row['kind']: (STATE / moment_report['runs']['mf'] / row['file'], row['sha256'])
                    for row in moment_report['metadata']['mf']['records']})
    expected = {'hp': state_hashes(comparison['hp_run'], False), 'mf': state_hashes(comparison['mf_run'], True)}
    gradients = {'hp': (Path(comparison['hp_gradient']['file']), comparison['hp_gradient']['sha256']),
                 'mf': (Path(comparison['native_kernels']['input.0']['path']),
                        comparison['native_kernels']['input.0']['sha256'])}
    group = moment_report['metadata']['hp']['extra']['group']
    results = {}
    for family in ['hp', 'mf']:
        for kind, (path, sha) in initial.items():
            copy_to_device(path, current[kind], sha)
        copy_to_device(gradients[family][0], gradient, gradients[family][1])
        adamw_module.adamw([model.head], [gradient], [current['exp_avg']], [current['exp_avg_sq']], [], 207,
                          amsgrad=group['amsgrad'], maximize=False, beta1=group['betas'][0], beta2=group['betas'][1],
                          lr=group['lr'], weight_decay=group['weight_decay'], eps=group['eps'])
        actual = {kind: digest(capture_hp_native.numpy_value(value)) for kind, value in current.items()}
        results[family] = {'actual_sha256': actual,
                           'matches_hp_actual': {kind: sha == expected['hp'][kind] for kind, sha in actual.items()},
                           'matches_mf_native': {kind: sha == expected['mf'][kind] for kind, sha in actual.items()}}
        print('[P12-HEAD207-REPLAY]', family, json.dumps(results[family]), flush=True)
        if family == 'hp' and not all(results[family]['matches_hp_actual'].values()):
            (output / 'FAILED_FIDELITY.json').write_text(json.dumps(results, indent=2))
            raise ValueError('Same-input replay did not reproduce actual HP update')
    report = {'complete': True, 'actual_hp_replay_exact': True, 'results': results,
              'initial_state': {kind: {'path': str(path), 'sha256': sha} for kind, (path, sha) in initial.items()},
              'gradient_sources': {family: {'path': str(path), 'sha256': sha}
                                   for family, (path, sha) in gradients.items()},
              'optimizer_group': group, 'optimizer_step': 207, 'factor_source_sha256': factor_sha,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
              'scope': 'Exact installed factor, full778567680-element state; one gradient substitution only. '
                       'Not a new independent training trajectory or universal AdamW correctness claim.'}
    (ROOT / 'evidence/head-moments207-actual-adamw-replay.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
