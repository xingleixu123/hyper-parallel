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

"""从已收回的三臂小型日志生成主LM图数据；不执行训练或外部命令。"""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


REPORT = Path(__file__).resolve().parents[1]
TASK_ROOT = REPORT.parents[1]
SOURCE = '544526f7db8342c607244fd41009b2a74adb84c8'
READOUT = 'evidence/a1e6-muon-update-trace-root-review-v1/selected-unbatch-v2-actual-metrics-readout-v1.json'
READOUT_SHA = '4d5219fafdd19bc158e192ee5ba4d1b7466e60c0da4161b9090cc8019ccf8b09'
CASES = {
    'F': 'r129-a1e6-muon-update-fsdp8-3-v1',
    'B': 'r129-a1e6-muon-update-ep4-both-3-v1',
    'S': 'r129-a1e6-muon-selected-ns-unbatch-both-3-v2',
}
COLLECTIONS = {
    'F': ('evidence/a1e6-muon-update-trace-control-v1/collections/fsdp-manifest-v1.json',
          '923fede3296e6a2970e7950004940bf3b7a8a439ef854159ef0fd4d73c2e60bb'),
    'B': ('evidence/a1e6-muon-update-trace-control-v1/collections/both-manifest-v1.json',
          '1eb39f5279da26ce43ff700da4d338879917e64631ba29425f9eba31d1f71272'),
    'S': ('evidence/a1e6-muon-selected-ns-unbatch-control-v2/collections/both-manifest-v1.json',
          '7fee65ff001568b28711b1c1e2bbbde6eddeb7410ec0d4ca06c4012b0fdb32d9'),
}
SCALARS = ('training/total_loss', 'training/foundation_loss/lm', 'training/foundation_loss/mtp',
           'training/foundation_loss/aux', 'training/grad_norm', 'training/lr')
COUNTERS = ('data/step_tokens', 'data/consumed_tokens', 'data/step_samples', 'data/consumed_samples')
LM = 'training/foundation_loss/lm'


def require(condition: bool, message: str) -> None:
    """拒绝缺失、混合或不完整的输入。"""
    if not condition:
        raise ValueError(message)


def sha256(raw: bytes) -> str:
    """返回原始字节的SHA256。"""
    return hashlib.sha256(raw).hexdigest()


def read_bound(root: Path, relative: str, expected: str) -> tuple[bytes, dict]:
    """只读任务根内、由完整SHA绑定的小文件。"""
    path = root / relative
    require(path.resolve() == path and path.is_file(), '原件路径不是固定普通文件：' + relative)
    raw = path.read_bytes()
    require(len(raw) <= 4 * 1024 ** 2 and sha256(raw) == expected, '原件大小或SHA不同：' + relative)
    return raw, {'path': relative, 'bytes': len(raw), 'sha256': expected}


def csv_bytes(rows: list[dict], fields: list[str]) -> bytes:
    """将小表写成确定性的UTF-8 CSV。"""
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def write_new(path: Path, raw: bytes) -> None:
    """只创建新产物，保留已有输出。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)


def build(root: Path, output: Path) -> dict:
    """核对三份收集manifest与24原日志，并派生72行及三步主LM表。

    Args:
        root: 已收集原文的本机任务根目录。
        output: 尚不存在的图数据目录。

    Returns:
        限定为日志数值复算的摘要；不签发科学、执行或CPU资格。
    """
    require(not output.exists(), '数据输出目录必须是新的')
    raw_readout, readout_proof = read_bound(root, READOUT, READOUT_SHA)
    readout = json.loads(raw_readout)
    require(readout['cases'] == CASES, '实际metrics读数的三臂身份不同')
    bound = {row['path']: row for row in readout['bindings']}
    require(len(bound) == len(readout['bindings']) == 24, '日志SHA闭包不是24个唯一文件')
    documents, projections, proofs, manifests = {}, [], [], []
    for arm, case in CASES.items():
        relative, expected = COLLECTIONS[arm]
        raw_manifest, descriptor = read_bound(root, relative, expected)
        manifest = json.loads(raw_manifest)
        require(manifest['case'] == case and manifest['source'] == SOURCE and manifest['released'] is True
                and manifest['allow_incomplete'] is False, '收集身份、source或完整收集状态不同')
        files = {row['path']: row for row in manifest['files']}
        require(len(files) == len(manifest['files']), '收集manifest有重复路径')
        manifests.append({**descriptor, 'arm': arm, 'files': len(files), 'raw_bytes': manifest['raw_bytes']})
        for rank in range(8):
            path = 'runs/' + case + '/metrics-rank' + str(rank) + '.jsonl'
            require(path in bound and files.get(path) == bound[path], '原日志与收集manifest绑定不同')
            raw, proof = read_bound(root, path, bound[path]['sha256'])
            require(len(raw) == bound[path]['bytes'], '原日志字节数不同')
            rows = [json.loads(line) for line in raw.splitlines()]
            require([row['step'] for row in rows] == [1, 2, 3], '每rank必须恰有三步')
            proofs.append(proof)
            for row in rows:
                require(type(row['rank']) is int and row['rank'] == rank, 'rank身份不同')
                require(all(type(row[field]) in (int, float) and math.isfinite(row[field])
                            for field in SCALARS + COUNTERS), '日志含非有限或非数值科学字段')
                key = (arm, rank, row['step'])
                documents[key] = row
                projections.append({'arm': arm, 'rank': rank, 'step': row['step'],
                                    **{field: row[field] for field in SCALARS + COUNTERS},
                                    'global_input_sha256': row['input_identity']['global_sha256']})
    require(len(documents) == 72, '三臂×8rank×3步覆盖不完整')
    step_rows, checks = [], []
    for step in (1, 2, 3):
        step_row = {'step': step, 'ranks_per_arm': 8}
        for arm in CASES:
            values = [documents[arm, rank, step][LM] for rank in range(8)]
            require(len(set(values)) == 1, '同臂同一步各rank的主LM不是同一全局标量')
            step_row[arm + '_main_LM'] = values[0]
        for arm in ('B', 'S'):
            values = []
            for field in SCALARS:
                maximum = max(abs(documents[arm, rank, step][field] - documents['F', rank, step][field])
                              for rank in range(8))
                reference = readout['comparisons'][arm + '_vs_F'][step - 1]
                require(reference['step'] == step and reference['ranks'] == 8
                        and maximum == reference['max_abs'][field], '复算与绑定的root读数不同')
                if field == LM:
                    values.append(maximum)
            require(len(values) == 1, '主LM字段不唯一')
            step_row[arm + '_vs_F_all8rank_max_absdiff'] = values[0]
        input_equal = all(documents[arm, rank, step]['input_identity']
                          == documents['F', rank, step]['input_identity'] for arm in CASES for rank in range(8))
        counters_equal = all(documents[arm, rank, step][field] == documents['F', rank, step][field]
                             for arm in CASES for rank in range(8) for field in COUNTERS + ('training/lr',))
        require(input_equal and counters_equal, '输入无序指纹、LR或计数器不同')
        checks.append({'step': step, 'all8rank_global_LM_equal_within_each_arm': True,
                       'input_component_fingerprint_equal': input_equal, 'LR_and_four_counters_equal': counters_equal})
        step_rows.append(step_row)
    csv_payloads = {
        'metrics-72rows.csv': csv_bytes(projections, list(projections[0])),
        'main-lm-3steps.csv': csv_bytes(step_rows, list(step_rows[0])),
    }
    summary = {
        'schema': 'selected-ns-unbatch-report-metrics-v1', 'status': 'NOT_PUBLISHED', 'source': SOURCE,
        'cases': CASES, 'steps': [1, 2, 3], 'ranks': 8, 'metrics_rows': 72,
        'LM_comparison': step_rows, 'checks': checks,
        'csv_sha256': {name: sha256(raw) for name, raw in csv_payloads.items()},
        'parsed_fields_only': list(SCALARS + COUNTERS) + ['step', 'rank', 'input_identity'],
        'rank_scope': '八rank是同一运行的归约标量观测，不是八次独立重复。',
        'input_scope': 'global_sha是输入组件的无序指纹，不是逐token UID或物理rank token对应证明。',
        'parameter_and_NS_science': 'NOT_ASSESSED_BY_METRICS_BUILDER',
        'not_claimed': ['200步验收', '产品修复', '唯一数学或kernel根因', '性能或显存收益'],
    }
    provenance = {'schema': 'selected-unbatch-report-small-source-provenance-v1', 'status': 'NOT_PUBLISHED',
                  'root_metrics_readout': readout_proof, 'collection_manifests': manifests,
                  'metric_files': proofs, 'large_tensor_payload_copied': False,
                  'scope': '只重算24份metrics及原manifest中的相同SHA条目，不替代完整执行链或参数NS科学独审。'}
    for name, raw in csv_payloads.items():
        write_new(output / name, raw)
    for name, value in (('summary.json', summary), ('provenance.json', provenance)):
        write_new(output / name, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    return {'status': 'NOT_PUBLISHED_METRICS_ONLY', 'rows': 72, 'LM_comparison': step_rows}


def main() -> None:
    """创建一个新的限定输出目录，不读取tensor或执行训练。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-root', type=Path, default=TASK_ROOT)
    parser.add_argument('--output-dir', type=Path, default=REPORT / 'data')
    args = parser.parse_args()
    print(json.dumps(build(args.task_root.resolve(), args.output_dir.resolve()), ensure_ascii=False))


if __name__ == '__main__':
    main()
