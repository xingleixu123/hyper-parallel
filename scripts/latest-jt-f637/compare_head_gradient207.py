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

"""Audit native head207 optimizer input against the state-aligned HP postclip gradient."""
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_head_moments206 import compare
from compare_head206_interventions import model_hashes
from compare_vocab_optimizer_trace import fidelity
from head_moments206_observation import digest

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
MF = 'mf-head207-nativegradient'
HP = 'hp-jtf637-headroutermoments-gradient207'
HP_REFERENCE = 'hp-jtf637-sharedup-headroutermoments206diag226'


def read(run: str, name: str) -> dict:
    """Read a small immutable observation record."""
    return json.loads((ROOT / 'runs' / run / name).read_text())


def main() -> None:
    """Require unchanged native207 states and reproduce the prior HP diagnostic before comparison."""
    checks = {'mf': fidelity(MF, 'mf-resume', 7), 'hp': fidelity(HP, HP_REFERENCE, 7)}
    for family, run, reference in [('mf', MF, 'mf-head207-momenthash'), ('hp', HP, HP_REFERENCE)]:
        checks[family]['exit_zero'] = (ROOT / 'runs' / (run + '.exit')).read_text().strip() == '0'
        checks[family]['moments207_exact_reference'] = read(run, 'HEAD_MOMENTS207_HASH.json') == read(
            reference, 'HEAD_MOMENTS207_HASH.json')
    checks['hp']['models206207_exact_reference'] = model_hashes(HP) == model_hashes(HP_REFERENCE)
    for name in ['DIAGNOSTIC_ONLY.json', 'HEAD_MOMENTS206_INTERVENTION.json', 'MODEL206_INTERVENTION.json']:
        checks['hp'][name + '_exact_reference'] = read(HP, name) == read(HP_REFERENCE, name)
    manifest = read(HP, 'PROBE_SOURCE_MANIFEST.json')
    checks['hp']['49_sources'] = len(manifest['files']) == 49 and all(
        row['sha256'] == hashlib.sha256((ROOT / 'tools-f637' / row['name']).read_bytes()).hexdigest()
        for row in manifest['files'])
    observed = read(HP, 'HEAD_GRADIENT207.json')
    checks['hp']['observer_source'] = observed['source_sha256'] == hashlib.sha256(
        (ROOT / 'tools-f637/hp_head_gradient207_probe.py').read_bytes()).hexdigest()
    checks['hp']['observer_helper'] = observed['helper_sha256'] == hashlib.sha256(
        (ROOT / 'tools-f637/head_moments206_observation.py').read_bytes()).hexdigest()
    checks['hp']['boundary'] = observed['step'] == 207 and observed['read_only_observation']
    hp = np.load(observed['file'], mmap_mode='r')
    checks['hp']['full_gradient_hash'] = digest(hp) == observed['sha256']
    directory = STATE / (MF + '-kernels')
    files = sorted(directory.rglob('*.npy'))
    checks['mf']['native_dump_exact_four_arrays'] = len(files) == 4
    kernels = {}
    for path in files:
        keys = [key for key in ['input.0', 'input.1', 'input.2', 'output.0'] if '.' + key + '.' in path.name]
        if len(keys) != 1 or keys[0] in kernels or path.parent.name != '6':
            raise ValueError('Unexpected kernel input/output or iteration')
        array = np.load(path, mmap_mode='r')
        kernels[keys[0]] = {'path': str(path), 'sha256': digest(array), 'shape': list(array.shape),
                            'dtype': str(array.dtype), 'bytes': path.stat().st_size}
    checks['mf']['native_dump_slots'] = set(kernels) == {'input.0', 'input.1', 'input.2', 'output.0'}
    expected_m = next(row['sha256'] for row in read(MF, 'HEAD_MOMENTS207_HASH.json')['records']
                      if row['kind'] == 'exp_avg')
    checks['mf']['kernel_output_exact_native207_moment'] = kernels['output.0']['sha256'] == expected_m
    graph = directory / 'rank_0/graphs/ms_output_trace_code_graph_3.ir'
    lines = graph.read_text().splitlines()
    hits = [index for index, line in enumerate(lines)
            if '# Fullname with scope: (Default/GraphKernel_Mul_Add_Assign_fusion-op1)' in line]
    checks['mf']['unique_head_kernel'] = len(hits) == 1
    snippet = '\n'.join(lines[hits[0] - 4:hits[0] + 1])
    checks['mf']['first_input_native_clip_x'] = '(%para93_clip_x,' in snippet
    checks['mf']['head_parameter_moment'] = 'ref_key=adam_m.output_layer.weight' in snippet
    if not all(all(row.values()) for row in checks.values()):
        raise ValueError(checks)
    mf = np.load(kernels['input.0']['path'], mmap_mode='r')
    patch_directory = STATE / 'head-gradient207-difference-patch'
    patch_directory.mkdir(exist_ok=False)
    comparison = compare(mf, hp, patch_directory / 'gradient.npz')
    report = {'complete': True, 'eligible_for_acceptance': False, 'fidelity': checks,
              'mf_run': MF, 'hp_run': HP, 'hp_reference': HP_REFERENCE, 'native_kernels': kernels,
              'hp_gradient': observed, 'gradient_comparison': comparison,
              'graph': {'path': str(graph), 'sha256': hashlib.sha256(graph.read_bytes()).hexdigest(),
                        'snippet': snippet, 'line': hits[0] + 1},
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Actual207 postclip gradient after206 model/head-moment synchronization. '
                       'Separates newly generated gradients from inherited head moment differences; '
                       'does not yet establish optimizer arithmetic correctness.'}
    (ROOT / 'evidence/head-moments207-gradient-comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'complete': True, 'fidelity': checks, 'gradient_comparison': comparison}))


if __name__ == '__main__':
    main()
