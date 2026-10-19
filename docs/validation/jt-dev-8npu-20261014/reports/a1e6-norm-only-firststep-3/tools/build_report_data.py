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

"""从已绑定的实际小证据生成私有诊断图数据；不运行分析器、框架或远端程序。"""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import struct


REPORT = Path(__file__).resolve().parents[1]
READER = 'evidence/a1e6-no-cross-param-firststep-norm-only-core-v2/outputreader-v1/'
ANALYSIS_SHA = '4d8e23470ca368d990dc78d133c246e0685a71f24dd2258788b7968345c1db89'
BINDINGS_SHA = 'ff83789682e9a3f14c26290c751763a4be84dc832b8fc0ab7fc8021b282ac361'
HISTORICAL = ('evidence/a1e6-muon-no-cross-param-200-control-v1/collections/'
              'f2-vs-ep4-mainlm-independent-review-v1.json')
HISTORICAL_SHA = '14cb3d229f05169e3b0e912ef202c2b311538359ef7559c765486d65864767d0'
LM = 'training/foundation_loss/lm'
PHASES = ('P0', 'G0', 'G1', 'P1')
ARMS = ('F', 'EP', 'N')


def csv_bytes(rows: list[dict], fields: list[str]) -> bytes:
    """以可逆十进制浮点数和固定行顺序输出便携CSV。"""
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def build(root: Path) -> dict[str, bytes]:
    """重读24份真实metrics；阶段计数仅投影冻结comparison，明确非独立原生解码。"""
    origins = {}

    def read(relative: str, expected: str, size: int | None = None) -> bytes:
        path = root / relative
        if path.resolve() != path or path.is_symlink() or not path.is_file():
            raise ValueError('来源路径重定向或缺失')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected or (size is not None and len(raw) != size):
            raise ValueError('来源SHA或字节数不符：' + relative)
        origins[relative] = {'path': relative, 'bytes': len(raw), 'sha256': expected}
        return raw

    def descriptor(row: dict) -> dict:
        return json.loads(read(row['path'], row['sha256'], row['bytes']))

    analysis = json.loads(read(READER + 'actual-comparison-v1.json', ANALYSIS_SHA))
    bindings = json.loads(read(READER + 'actual-bindings-v1.json', BINDINGS_SHA))
    if analysis['bindings_sha256'] != BINDINGS_SHA or analysis['pure_observation'] is not False:
        raise ValueError('必须绑定实际非纯N干预结果')
    pure = descriptor(bindings['pure_bindings'])
    proofs = {'F': pure['runs']['fsdp'], 'EP': pure['runs']['ep4'], 'N': bindings['norm']}
    reports, recipes, metric_rows, values = {}, {}, [], {}
    actual_scalars = {(row['rank'], row['step']): row['values'][LM] for row in analysis['three_step_scalars']}
    for arm, proof in proofs.items():
        manifest = descriptor(proof['collection'])
        case = manifest['case']
        if manifest['complete_collection'] is not True or manifest['allow_incomplete'] is not False:
            raise ValueError('图数据只接受完整实际collection')
        reports[arm] = {'case': case, 'files': len(manifest['files']), 'collection': proof['collection']}
        table = {row['path']: row for row in manifest['files']}
        config = descriptor(table['runs/' + case + '/config.resolved.json'])
        muon = config['optimizer']['muon_config']
        if muon['ns_dtype'] != 'float32' or muon['ns_batch_across_params'] is not False:
            raise ValueError('本轮实际配方必须是FP32 NS且不跨参数批处理')
        recipes[arm] = {'ns_dtype': muon['ns_dtype'], 'ns_batch_across_params': muon['ns_batch_across_params'],
                        'train_iters': config['training']['train_iters'], 'seed': config['training']['seed']}
        for rank in range(8):
            row = table['runs/' + case + '/metrics-rank%d.jsonl' % rank]
            raw = read(row['path'], row['sha256'], row['bytes'])
            metrics = [json.loads(line) for line in raw.splitlines()]
            if len(metrics) != 3 or [entry['step'] for entry in metrics] != [1, 2, 3]:
                raise ValueError('必须有每臂3步×8rank原始metrics')
            for entry in metrics:
                value, step = entry[LM], entry['step']
                if entry['rank'] != rank or type(value) not in (float, int) or not math.isfinite(value):
                    raise ValueError('主LM标签或有限性不符')
                key = {'F': 'fsdp', 'EP': 'ep4', 'N': 'norm_only'}[arm]
                if value != actual_scalars[(rank, step)][key]:
                    raise ValueError('原metrics与绑定comparison不同')
                values[(arm, rank, step)] = value
                metric_rows.append({'arm': arm, 'rank': rank, 'step': step, 'main_lm': value})
    curves = []
    for step in range(1, 4):
        for arm in ARMS:
            if len({values[(arm, rank, step)] for rank in range(8)}) != 1:
                raise ValueError('不能将非同值rank标量标成全局共识')
        curves.append({'step': step, 'F': values[('F', 0, step)], 'EP': values[('EP', 0, step)],
                       'N': values[('N', 0, step)],
                       'EP_vs_F_maxabs': max(abs(values[('EP', r, step)] - values[('F', r, step)]) for r in range(8)),
                       'N_vs_F_maxabs': max(abs(values[('N', r, step)] - values[('F', r, step)]) for r in range(8))})
    stages = []
    for reference, label in (('fsdp', 'F'), ('ep4', 'EP')):
        for phase in PHASES:
            source = analysis['responses'][reference]['stages'][phase]
            differences = source['differences']
            identities = {(row['name'], tuple(row['coordinate'])) for row in differences}
            if len(identities) != len(differences):
                raise ValueError('阶段差异列表含重复逻辑片')
            for group, total, fqns in (('nonexpert', 336, 42), ('expert', 32, 4)):
                rows = [row for row in differences if (row['family'] == 'Muon-expert') == (group == 'expert')]
                record = source[group]
                if (record['total'] != total or record['equal'] != total - len(rows)
                        or record['complete_fqns_equal'] != fqns - len({row['name'] for row in rows})):
                    raise ValueError('阶段计数与冻结difference数组不符')
                stages.append({'reference': label, 'phase': phase, 'group': group, 'equal': record['equal'],
                               'total': total, 'equal_fqns': record['complete_fqns_equal'], 'total_fqns': fqns})
    coefficients = []
    if [row['rank'] for row in analysis['actual_clip']] != list(range(8)):
        raise ValueError('裁剪原标量必须完整8rank')
    for row in analysis['actual_clip']:
        for arm, measured, used, key in (('F', row['used_hex'], row['used_hex'], 'fsdp'),
                                         ('EP', row['measured_hex'], row['measured_hex'], 'ep4'),
                                         ('N', row['measured_hex'], row['used_hex'], 'norm_only')):
            clip = row[key]
            if clip['original_code_verified'] is not True:
                raise ValueError('原clip CodeType未绑定')
            coefficients.append({'arm': arm, 'rank': row['rank'], 'measured_fp32_hex': measured,
                                 'used_fp32_hex': used, 'clip_coefficient_fp32_hex': clip['clip_coef_clamped_hex'],
                                 'measured_float32': struct.unpack('<f', bytes.fromhex(measured))[0],
                                 'used_float32': struct.unpack('<f', bytes.fromhex(used))[0]})
    historical = json.loads(read(HISTORICAL, HISTORICAL_SHA))
    history = historical['main_LM_gate']
    files = {'main-lm-rank-step.csv': csv_bytes(metric_rows, ['arm', 'rank', 'step', 'main_lm']),
             'curves.csv': csv_bytes(curves, list(curves[0])),
             'stage-equality.csv': csv_bytes(stages, list(stages[0])),
             'norm-coefficients.csv': csv_bytes(coefficients, list(coefficients[0]))}
    summary = {'schema': 'norm-only-private-diagnostic-report-data-v1', 'status': 'PRIVATE_NOT_PUBLISHED',
               'source': analysis['source'], 'arms': reports, 'recipes': recipes, 'metrics_rows': len(metric_rows),
               'rank_values_equal_within_each_arm_step': True, 'curves': curves, 'stage_counts': stages,
               'clip_first_rank': [row for row in coefficients if row['rank'] == 0],
               'prefix': analysis['prefix'],
               'preintervention_counts': analysis['preintervention']['native_before_clip'],
               'historical_200_main_LM_gate': {'pass': history['pass'], 'threshold': history['threshold_abs'],
                                             'maximum_abs': history['maximum_abs_difference'],
                                             'first_exceed_step': history['first_exceed_step'],
                                             'exceed_step_count': history['exceed_step_count']},
               'limits_zh': ['阶段数是原生SHA编码相同数，不是数值误差幅度；阶段计数投影冻结分析而非独立张量解码。',
                             '8rank是归约后同值全局标量，不是8次独立重复；只有首步干预，三步不是200步验收。',
                             'global输入SHA是内容多重集合指纹，不是另存逐DP-slot/micro/token顺序的实证。',
                             '没有新独立norm产品bug或double-count证据；本图不作唯一根因或全模型通过结论。'],
               'csv': {name: {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
                       for name, raw in files.items()}}
    files['summary.json'] = (json.dumps(summary, ensure_ascii=False, indent=2) + '\n').encode()
    provenance = {'schema': 'norm-only-private-report-provenance-v1', 'status': 'PRIVATE_NOT_PUBLISHED',
                  'analysis': {'path': READER + 'actual-comparison-v1.json', 'sha256': ANALYSIS_SHA},
                  'bindings': {'path': READER + 'actual-bindings-v1.json', 'sha256': BINDINGS_SHA},
                  'source_files': list(origins.values()),
                  'scope_zh': '仅重新读取24份metrics/3份resolvedconfig及冻结结果，未运行原分析器、远端或性能分析。'}
    files['provenance.json'] = (json.dumps(provenance, ensure_ascii=False, indent=2) + '\n').encode()
    return files


def main() -> None:
    """从真实本机证据重建到全新目录；不覆盖既有报告数据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-root', type=Path, default=REPORT.parents[2])
    parser.add_argument('--output-dir', type=Path, default=REPORT / 'data')
    args = parser.parse_args()
    files = build(args.task_root.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any((args.output_dir / name).exists() for name in files):
        raise ValueError('只允许全新数据目标')
    for name, raw in files.items():
        with (args.output_dir / name).open('xb') as stream:
            stream.write(raw)
    print(json.dumps({'status': 'PRIVATE_DERIVED_DATA_ONLY', 'files': len(files), 'metrics_rows': 72}))


if __name__ == '__main__':
    main()
