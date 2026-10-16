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

"""Collect native replay statistics and eight scalar values with explicit storage-dtype validation."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import shutil
import struct

ROOT = Path('/home/xxl/jt-adamw-align-20261013/phase12')
STATE = Path('/dev/shm/jt-adamw-align-20261013/state/jt-recipe-20261014')


def main() -> None:
    """Preserve only small graphs, statistics and scalar JSON, after native fidelity passes."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, required=True)
    step = parser.parse_args().step
    if not 201 <= step <= 400:
        raise ValueError('Step outside the selected continuation interval')
    prefix = 'frontier' + str(step)
    for suffix in ['-mf-builtin-stat', '-mf-normalization-scalars']:
        report = json.loads((ROOT / 'evidence' / (prefix + suffix + '.json')).read_text())
        if not report['complete'] or not all(row['exact_actual_training']
                                              for row in report['native_state_fidelity'].values()):
            raise ValueError('Native replay fidelity failed')
    statistics = STATE / (prefix + '-mf-builtin-stat')
    csv_path = statistics / ('rank_0/native_muon' + str(step) + '/0/0/statistic.csv')
    graph = statistics / 'rank_0/graphs/ms_output_trace_code_graph_0.ir'
    dvm = list((STATE / ('work-' + prefix + '-mf-builtin-stat/graph_kernel_dump')).glob('dvm_kernel_*.txt'))
    if len(dvm) != 1:
        raise ValueError('Expected one native DVM compilation trace')
    for original, name in [(csv_path, '-native-update-stat.csv'), (graph, '-native-update-graph.ir'),
                           (dvm[0], '-native-dvm.txt')]:
        destination = ROOT / 'evidence' / (prefix + name)
        if destination.exists() and destination.read_bytes() != original.read_bytes():
            raise ValueError('Existing native evidence differs')
        shutil.copyfile(original, destination)
    rows = list(csv.DictReader(csv_path.open()))
    files = sorted((STATE / (prefix + '-mf-normalization-scalars')).rglob('*.npy'))
    if len(files) != 2:
        raise ValueError('Expected only one eight-element input and output')
    records = []
    for path in files:
        raw = path.read_bytes()
        if raw[:8] != b'\x93NUMPY\x01\x00':
            raise ValueError('Unexpected scalar NPY header version')
        size = int.from_bytes(raw[8:10], 'little')
        offset = 10 + size
        header = ast.literal_eval(raw[10:offset].decode('latin1'))
        payload = raw[offset:]
        if header != {'descr': '<f4', 'fortran_order': False, 'shape': (8,)} or len(payload) != 32:
            raise ValueError('Expected eight scalar values stored in FP32')
        values = list(struct.unpack('<8f', payload))
        io = 'input' if '.input.0.' in path.name else 'output'
        native_payload = payload
        if io == 'output':
            bits = struct.unpack('<8I', payload)
            if any(value & 0xffff for value in bits):
                raise ValueError('Native BF16 denominator is not exactly represented')
            native_payload = struct.pack('<8H', *(value >> 16 for value in bits))
        selected = [row for row in rows if row['Op Name'] == 'Default/GraphKernel_Sqrt_Add_Cast_split-op0'
                    and row['IO'] == io and row['Slot'] == '0']
        if len(selected) != 1 or hashlib.sha1(native_payload).hexdigest() != selected[0]['SHA1']:
            raise ValueError('Scalar values do not match the independent native statistics')
        records.append({'file': path.name, 'header': header, 'values': values,
                        'file_sha256': hashlib.sha256(raw).hexdigest(), 'payload_bytes': len(payload),
                        'sha1': hashlib.sha1(payload).hexdigest(),
                        'native_sha1': hashlib.sha1(native_payload).hexdigest()})
    result = {'complete': True, 'records': records, 'native_statistics_exact': True,
              'logical_native_payload_bytes': 48, 'stored_scalar_payload_bytes': 64,
              'storage_note': 'MindSpore Dump stores the eight BF16 output scalars as eight FP32 values.',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    target = ROOT / 'evidence' / (prefix + '-native-normalization-scalars-values.json')
    if target.exists():
        raise ValueError('Scalar evidence already exists; preserve original provenance')
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
