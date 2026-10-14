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
"""Exercise performance exclusions and resume boundaries with independent fixtures."""

import json
from pathlib import Path
import tempfile

from compare_runs import audit, measurement_context


def main():
    with tempfile.TemporaryDirectory() as temp:
        reference = Path(temp) / 'reference'
        candidate = Path(temp) / 'candidate'
        for directory in (reference, candidate):
            directory.mkdir()
            (directory / 'config.resolved.json').write_text(json.dumps({'model_integration': {'mode': 'build'}}))
        eligible = 'eligible_for_matching_build_mode_performance_comparison'
        assert measurement_context(reference, candidate)[eligible]
        markers = ['expert-samples.jsonl', 'routing-rank0.jsonl', 'ns-batches-rank0.jsonl',
                   'ns-single-matrix-rank0.jsonl', 'activations-rank0.jsonl',
                   'projection-digests-rank0.jsonl', 'loss-gradients-rank0.jsonl',
                   'linear-calls-rank0.jsonl', 'stable-clip-rank0.jsonl',
                   'stage-profile-rank0.jsonl', 'module-profile-rank0.jsonl']
        nested = candidate / 'nested'
        nested.mkdir()
        for marker in markers:
            path = nested / marker
            path.write_text('{}\n')
            result = measurement_context(reference, candidate)
            assert not result[eligible], marker
            assert result['excluded_reasons'], marker
            path.unlink()
        for config in [{'model_integration': {'mode': 'runtime'}}, {},
                       {'model_integration': {'mode': 'build'}, 'profiler': {'enabled': True}},
                       {'model_integration': {'mode': 'build'}, 'memory': {'enable': True}}]:
            (candidate / 'config.resolved.json').write_text(json.dumps(config))
            assert not measurement_context(reference, candidate)[eligible], config
        (candidate / 'config.resolved.json').write_text(json.dumps({'model_integration': {'mode': 'build'}}))
        for directory in (reference, candidate):
            (directory / 'EXIT_CODE').write_text('0\n')
            for rank in range(8):
                (directory / f'initial-state-rank{rank}.json').write_text('{"seed": 42}')
                (directory / f'start-rank{rank}.json').write_text('{"global_step": 0}')
                (directory / f'complete-rank{rank}.json').write_text('{"global_step": 2}')
                rows = []
                for step in (1, 2):
                    rows.append({'step': step, 'input_identity': {'global_sha256': str(step)},
                                 'training/total_loss': 4., 'training/foundation_loss/lm': 3.,
                                 'training/foundation_loss/mtp': .9, 'training/foundation_loss/aux': .1,
                                 'training/grad_norm': 1., 'training/lr': .0001,
                                 'memory/device_max_allocated_gb': 2., 'memory/device_max_reserved_gb': 3.})
                (directory / f'metrics-rank{rank}.jsonl').write_text('\n'.join(map(json.dumps, rows)) + '\n')
        assert audit(reference, candidate, 1, 2)['status'] == 'PASS'
        (candidate / 'start-rank7.json').write_text('{"global_step": 1}')
        failed = audit(reference, candidate, 1, 2)
        assert failed['status'] == 'FAIL'
        assert any('rank 7' in finding and 'starts at 1' in finding for finding in failed['findings'])
        print(json.dumps({'status': 'PASS', 'diagnostic_markers_excluded': len(markers),
                          'configurations_excluded': 4, 'matching_trace_passed': True,
                          'wrong_start_rank7_rejected': True}))


if __name__ == '__main__':
    main()
