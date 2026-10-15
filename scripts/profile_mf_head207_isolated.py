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

"""Replay both actual207 output-head operands through isolated MF graph and eager MatMul."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import mindspore as ms
from mindspore import nn, ops
import numpy as np

ROOT = Path('/alignment/phase12')
STATE = Path('/state/jt-recipe-20261014')
SOURCE = STATE / 'mf-head207-nativebranches-kernels/rank_0/HeadBranches207/1/6'
FILES = {2: {'dy': '1680306536.npy', 'x': '1244596951.npy', 'dw': '964996192.npy'},
         3: {'dy': '1421125496.npy', 'x': '1222844361.npy', 'dw': '252264425.npy'}}


class HeadGradient(nn.Cell):
    """Keep exactly the native transpose flags for the isolated product."""

    def construct(self, gradient, inputs):
        return ops.MatMul(transpose_a=True, transpose_b=False)(gradient, inputs)


def bf16_hash(values: np.ndarray) -> str:
    """Hash the complete BF16 payload represented by the native dump's FP32 values."""
    digest = hashlib.sha256()
    flat = values.reshape(-1)
    for start in range(0, flat.size, 4_000_000):
        chunk = np.asarray(flat[start:start + 4_000_000], dtype=np.float32)
        words = chunk.view(np.uint32)
        if np.any(words & 65535):
            raise ValueError('Expected exact BF16-representable source values')
        digest.update((words >> 16).astype(np.uint16).tobytes())
    return digest.hexdigest()


def main() -> None:
    """Preserve full native input and output hashes; write only a small replay report."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['graph', 'eager'], required=True)
    args = parser.parse_args()
    ms.set_context(mode=ms.GRAPH_MODE if args.mode == 'graph' else ms.PYNATIVE_MODE,
                   device_target='Ascend', device_id=0, deterministic='ON', max_device_memory='48GB',
                   memory_optimize_level='O0', jit_config={'jit_level': 'O1'},
                   ascend_config={'parallel_speed_up_json_path':
                                  '/workspace/mindformers/configs/qwen3/parallel_speed_up.json'})
    records = []
    cell = HeadGradient()
    for kernel, files in FILES.items():
        source = {name: np.load(SOURCE / filename, mmap_mode='r') for name, filename in files.items()}
        expected = json.loads((ROOT / 'evidence' / f'head207-gemm-screen-op{kernel}.json').read_text())
        inputs = {}
        input_hashes = {}
        for name in ['dy', 'x']:
            original_hash = bf16_hash(source[name])
            if original_hash != expected['input_hashes'][name]:
                raise ValueError('Native input differs from audited actual207 operand')
            inputs[name] = ms.Tensor(np.array(source[name]), ms.bfloat16)
            actual = inputs[name].astype(ms.float32).asnumpy()
            if bf16_hash(actual) != original_hash:
                raise ValueError('Actual device operand changed during loading')
            input_hashes[name] = original_hash
            del actual
        result = cell(inputs['dy'], inputs['x'])
        if result.dtype != ms.bfloat16 or tuple(result.shape) != (152064, 5120):
            raise ValueError('Unexpected output shape or dtype')
        output = result.astype(ms.float32).asnumpy()
        sha = bf16_hash(output)
        different = 0
        maximum = 0.0
        sum_square = 0.0
        reference_square = 0.0
        for start in range(0, output.shape[0], 512):
            ref = np.asarray(source['dw'][start:start + 512], dtype=np.float64)
            delta = output[start:start + 512].astype(np.float64) - ref
            different += int(np.count_nonzero(delta))
            maximum = max(maximum, float(np.abs(delta).max()))
            sum_square += float(np.sum(delta * delta))
            reference_square += float(np.sum(ref * ref))
        modes = {row['mode']: row['bf16_sha256'] for row in expected['results']}
        row = {'kernel': kernel, 'input_hashes': input_hashes, 'bf16_sha256': sha,
               'native_mf_bf16_sha256': bf16_hash(source['dw']), 'different_vs_native_mf': different,
               'max_abs_vs_native_mf': maximum, 'relative_l2_vs_native_mf': (sum_square / reference_square) ** .5,
               'equals_hp_screen_modes': [name for name, value in modes.items() if value == sha]}
        records.append(row)
        print(json.dumps(row), flush=True)
        del output, result, inputs, source
        gc.collect()
    report = {'complete': True, 'mode': args.mode, 'mindspore': ms.__version__,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'records': records,
              'loaded_libraries': sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                          if any(name in line for name in ['libopapi.so', 'libascendcl.so', 'libnnopbase.so'])}),
              'scope': 'Isolated MatMul using actual207 operands; not an independent training result.'}
    (ROOT / 'evidence' / ('head207-gemm-mf-profile-' + args.mode + '.json')).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
