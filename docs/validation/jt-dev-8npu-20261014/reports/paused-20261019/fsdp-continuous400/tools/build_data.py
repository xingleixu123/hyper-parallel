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

"""Export small public projections of completed local evidence; never execute a model."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


SOURCE = '59b25cdd15505a4f6fb1990484d4e890e3d3724c'
CASES = {'continuous400': 'r125-a1e6-muon-no-cross-param-fsdp8-continuous400-v1',
         'reference200': 'r129-a1e6-muon-no-cross-param-fsdp8-200-v2'}
CONT = 'evidence/a1e6-125-muon-no-cross-param-continuous400-control-v1/collections/continuous400-v1-c1/'
REF = 'evidence/a1e6-muon-no-cross-param-200-control-v1/collections/fsdp8-v2-c1/'
PAIR = 'evidence/a1e6-muon-no-cross-param-200-control-v1/collections/'
HISTORY = 'evidence/a1e6-fsdp-resume400-issue-publication-v1/'
PINS = {
    CONT + 'manifest.json': 'e1737f301416c5f4a8850b416efcb650f423fc94d0c79101238742129e5725cd',
    CONT + 'actual-scientific-independent-review-v1.json':
        '9c4521a661edcc060fd6eceafb347aa1fcbcbf4a20293fe6c84b543af09ff99b',
    REF + 'manifest.json': '727e8d08bf740dbbbfc65b629ca32a5ff8ab8ee227e748fab94d716024cae0da',
    REF + 'actual-functional-independent-review-v1.json':
        'c32806a519fd94874722aec16076eea267f9d937ef50ca1837a896ab180d5df0',
    PAIR + 'f2-vs-ep4-mainlm-independent-review-v1.json':
        '14cb3d229f05169e3b0e912ef202c2b311538359ef7559c765486d65864767d0',
    HISTORY + 'issue208-body.md': 'abb3504078a2cbc010a62b4c47e7ecfea4f03ccef28be4d61cf60eed0ca979ca',
    HISTORY + 'publication-receipt.json': '78db06c36f5f5814347c164370004b2379c8b37219d81307964df9820082ef02',
}
FIELDS = ('training/total_loss', 'training/foundation_loss/lm', 'training/foundation_loss/mtp',
          'training/foundation_loss/aux', 'training/grad_norm', 'training/lr', 'data/step_tokens',
          'data/consumed_tokens', 'data/step_samples', 'data/consumed_samples')
LM = 'training/foundation_loss/lm'


def require(condition: bool, message: str) -> None:
    """Reject an incomplete or changed input instead of making a partial figure."""
    if not condition:
        raise ValueError(message)


def digest(raw: bytes) -> str:
    """Return the SHA256 of exact bytes."""
    return hashlib.sha256(raw).hexdigest()


def encode_json(value: object) -> bytes:
    """Encode deterministic UTF8 JSON with no nonfinite extensions."""
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def encode_csv(columns: list[str], rows: list[dict]) -> bytes:
    """Preserve Python float round trips and actual integer axes in a small CSV."""
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def difference_paths(left: object, right: object, prefix: str = '') -> list[str]:
    """List structural JSON recipe differences without discarding keys."""
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(left.keys() | right.keys()):
            name = prefix + '/' + key
            result.extend([name] if key not in left or key not in right else
                          difference_paths(left[key], right[key], name))
        return result
    return [] if left == right else [prefix]


def build(source_root: Path, output: Path) -> None:
    """Verify original metric bindings, then write sanitized derived evidence exclusively."""
    source_root = source_root.resolve()
    inputs = {}

    def read(relative: str, expected: dict | None = None) -> bytes:
        path = (source_root / relative).resolve()
        require(path.is_relative_to(source_root) and path.is_file(), '来源文件缺失或越界：' + relative)
        raw = path.read_bytes()
        descriptor = {'path': relative, 'bytes': len(raw), 'sha256': digest(raw)}
        if expected is not None:
            require(descriptor['bytes'] == expected['bytes'] and descriptor['sha256'] == expected['sha256'],
                    '来源大小/SHA改变：' + relative)
        if relative in PINS:
            require(descriptor['sha256'] == PINS[relative], '固定来源SHA改变：' + relative)
        inputs[relative] = descriptor
        return raw

    metadata = {name: read(name) for name in PINS}
    review = json.loads(metadata[CONT + 'actual-scientific-independent-review-v1.json'])
    require(review['verdict'] == 'PASS_WITH_BOUNDED_SCOPE' and review['blocking_findings'] == [], '独审状态改变')
    require(review['resume201_to400_from_checkpoint_validated'] is False, '不得扩为恢复资格')
    exports, datasets, manifests, recipes = [], {}, {}, {}
    for role, case in CASES.items():
        steps = 400 if role == 'continuous400' else 200
        prefix = CONT if role == 'continuous400' else REF
        manifest = json.loads(metadata[prefix + 'manifest.json'])
        manifests[role] = manifest
        require(manifest['case'] == case and manifest['source'] == SOURCE, '运行身份或源码不符')
        require(manifest['exit_code'] == 0 and manifest['released'] is True, '运行未成功释放')
        file_map = {row['path']: row for row in manifest['files']}
        base = 'runs/' + case + '/'

        def run_read(name: str) -> bytes:
            relative = base + name
            return read(relative, file_map[relative])

        require(run_read('source-commit.txt').decode().strip() == SOURCE, '源码提交不符')
        require(run_read('source-status.txt') == b'' and run_read('source.patch') == b'', '源码并非clean快照')
        recipes[role] = json.loads(run_read('config.resolved.json'))
        per_rank = []
        for rank in range(8):
            rows = [json.loads(line) for line in run_read(f'metrics-rank{rank}.jsonl').splitlines()]
            require(len(rows) == steps, '指标行数不完整')
            require([row['step'] for row in rows] == list(range(1, steps + 1)), '步序不完整')
            require(all(type(row['rank']) is int and row['rank'] == rank for row in rows), 'rank不符')
            require(all(type(row[field]) in (int, float) and math.isfinite(row[field])
                        for row in rows for field in FIELDS), '十字段存在缺失或非有限数')
            start = json.loads(run_read(f'start-rank{rank}.json'))
            complete = json.loads(run_read(f'complete-rank{rank}.json'))
            require(start['global_step'] == 0 and start['train_iters'] == steps, '起点不符')
            require(complete == {'global_step': steps, 'epoch': 1}, '完成标记不符')
            initial = json.loads(run_read(f'initial-state-rank{rank}.json'))
            runtime = json.loads(run_read(f'runtime-flags-rank{rank}.json'))
            require(len(initial) == 48 and runtime['training']['npu_matmul']['allow_hf32'] is False,
                    '初态或HF32不符')
            if role == 'reference200':
                prior = datasets['continuous400'][rank]
                require(initial == prior['initial'] and runtime == prior['runtime'], '跨机初态/runtime不等')
                require(all(row['input_identity'] == prior['rows'][index]['input_identity']
                            for index, row in enumerate(rows)), '共同窗口输入身份不等')
            per_rank.append({'rows': rows, 'initial': initial, 'runtime': runtime})
            exports.extend({'role': role, 'rank': rank, 'step': row['step'], 'main_lm': row[LM]} for row in rows)
        require(all(per_rank[rank]['rows'][index][LM] == per_rank[0]['rows'][index][LM]
                    for rank in range(8) for index in range(steps)), '全局主LM各rank并非同值')
        datasets[role] = per_rank
    recipe_differences = difference_paths(recipes['continuous400'], recipes['reference200'])
    require(recipe_differences == ['/checkpoint/checkpoint_dir', '/checkpoint/save_ckpt', '/training/train_iters'],
            '出现未列出的recipe差异')
    field_checks = {}
    for field in FIELDS:
        errors = [abs(datasets['continuous400'][rank]['rows'][index][field] -
                      datasets['reference200'][rank]['rows'][index][field])
                  for rank in range(8) for index in range(200)]
        field_checks[field] = {'points': len(errors), 'max_abs': max(errors), 'nonzero': sum(x != 0 for x in errors)}
    require(all(item['max_abs'] == 0 for item in field_checks.values()), '共同窗口十字段非零差')
    curves = []
    for index in range(400):
        curves.append({'step': index + 1, 'continuous400_rank0_lm': datasets['continuous400'][0]['rows'][index][LM],
                       'reference200_rank0_lm': datasets['reference200'][0]['rows'][index][LM] if index < 200 else '',
                       'common_maxabs': 0.0 if index < 200 else ''})
    allrank_csv = encode_csv(['role', 'rank', 'step', 'main_lm'], exports)
    curves_csv = encode_csv(['step', 'continuous400_rank0_lm', 'reference200_rank0_lm', 'common_maxabs'], curves)
    ep = json.loads(metadata[PAIR + 'f2-vs-ep4-mainlm-independent-review-v1.json'])['main_LM_gate']
    require(ep['pass'] is False and ep['maximum_abs_difference'] == 0.010309219360351562, 'EP历史结论改变')
    summary = {
        'schema': 'paused-goal-continuous400-public-summary-v1', 'status': 'DRAFT_NOT_PUBLISHED',
        'goal_state': 'PAUSED', 'source_commit': SOURCE, 'source_parent': '544526f7db8342c607244fd41009b2a74adb84c8',
        'cann': '9.2.0-beta.2', 'cases': CASES, 'ranks': 8,
        'continuous400': {'steps': 400, 'rank_steps': 3200, 'start_step': 0, 'complete_step': 400,
                          'finite_points_per_field': 3200, 'main_lm_rank0_first': curves[0]['continuous400_rank0_lm'],
                          'main_lm_rank0_step200': curves[199]['continuous400_rank0_lm'],
                          'main_lm_rank0_last': curves[-1]['continuous400_rank0_lm'], 'save_ckpt': False},
        'common_first200': {'steps': 200, 'rank_steps': 1600, 'fields': field_checks,
                            'main_lm_threshold_abs': 0.005, 'main_lm_within_threshold': True,
                            'initial_entries_equal_per_rank': 48, 'runtime_equal_all8': True,
                            'input_identity_equal_all1600': True, 'recipe_difference_paths': recipe_differences},
        'policy': {'ns_dtype': 'float32', 'ns_batch_across_params': False, 'allow_hf32': False,
                   'seed': 42, 'lr_scheduler_iters': 1000},
        'old125_warning200': {'common_first200_ten_fields_max_abs': 0,
                              'basis': '9c4521a661edcc060fd6eceafb347aa1fcbcbf4a20293fe6c84b543af09ff99b',
                              'execution_gate': 'OBSERVATIONAL_ONLY_EXECUTION_GATE_UNVERIFIED',
                              'original_reason': 'owned-process-observation-unavailable',
                              'allow_incomplete': True, 'complete_collection': False},
        'current59b_ep200': {'pass': False, 'maximum_abs_difference': ep['maximum_abs_difference'],
                             'maximum_step': 146, 'first_exceed_step': ep['first_exceed_step'],
                             'exceed_steps': ep['exceed_steps'], 'exceed_rank_step_points': ep['exceed_point_count']},
        'resume201_to400_validated': False, 'performance_analyzed': False,
        'ranks_are_independent_repeats': False,
        'csv': {name: {'bytes': len(raw), 'sha256': digest(raw)} for name, raw in
                [('main-lm-allranks.csv', allrank_csv), ('curves.csv', curves_csv)]},
    }
    provenance = {
        'schema': 'paused-goal-public-provenance-projection-v1', 'status': 'DRAFT_NOT_PUBLISHED',
        'projection_only': True, 'private_absolute_paths_credentials_fullCID_removed': True,
        'continuous400_review': {'sha256': PINS[CONT + 'actual-scientific-independent-review-v1.json'],
                                 'status': review['status'], 'verdict': review['verdict'],
                                 'blocking_findings': [], 'scope': '连续400功能与共同前200观察，不含resume'},
        'continuous400_receipt': {'exit_code': 0, 'reason': None, 'attempted': True, 'released': True,
                                  'container_stopped': True, 'remaining_owned_pids': [],
                                  'canonical_sha256': manifests['continuous400']['canonical_receipt_sha256'],
                                  'remote_sha256': manifests['continuous400']['remote_receipt_sha256'],
                                  'canonical_remote_JSON_equal_bytes_distinct': True},
        'continuous400_collection': {'sha256': PINS[CONT + 'manifest.json'], 'files': 94,
                                     'raw_bytes': 3066452, 'batches': 19, 'two_inventories_stable': True,
                                     'fullCID_absence_verified': True, 'complete_collection': True,
                                     'allow_incomplete': False},
        'historical544_resume': {
            'source': '544526f7db8342c607244fd41009b2a74adb84c8', 'status': 'HISTORICAL_REPORTED_PASS',
            'window': [201, 400], 'not_recomputed_here': True,
            'documentation_commit': '3dfd73fb84ed69a93f883478cf055d38dae46808',
            'published_report_sha256': '16d344416f64b55107c6f5480acb3aea54ff74a6f434e566639810010e6c209a',
            'scope': '固定544历史恢复窗口对独立连续参考，不迁移为59b恢复资格'},
        'source_descriptors': sorted(inputs.values(), key=lambda item: item['path']),
        'not_included': ['原始大checkpoint', '专家供体大tensor', '私有宿主机绝对路径与凭证', '性能与显存字段'],
    }
    outputs = {'data/main-lm-allranks.csv': allrank_csv, 'data/curves.csv': curves_csv,
               'data/summary.json': encode_json(summary), 'evidence/provenance.json': encode_json(provenance)}
    require(not any((output / name).exists() for name in outputs), '输出已存在，拒绝覆盖')
    for name, raw in outputs.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
    print(json.dumps({'status': 'EXPORTED', 'main_lm_rows': len(exports), 'curve_rows': len(curves),
                      'original_inputs_hashed': len(inputs)}, ensure_ascii=False))


def main() -> None:
    """Read local raw evidence and write to one explicit, fresh output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    build(args.source_root, args.output_dir.resolve())


if __name__ == '__main__':
    main()
