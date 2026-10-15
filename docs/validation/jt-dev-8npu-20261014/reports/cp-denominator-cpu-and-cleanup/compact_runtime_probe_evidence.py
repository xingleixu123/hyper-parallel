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

"""Losslessly compact completed, task-owned runtime probes with byte verification."""

import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path('/home/xxl/jt-dev-8npu-20261014')
CASES = (
    'fpgmm-nsall32-cp2-200',
    'fpgmm-nsall32-ep2-200',
    'fpgmm-nsall32-tp2ep2-200',
    'reviewed-tp2-replicated-experts-200',
    'fpgmm-nsall32-fsdp8-200',
)
CHUNK_BYTES = 1024 * 1024


def _signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _assert_not_open(path: Path) -> None:
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            descriptors = list((directory / 'fd').iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for descriptor in descriptors:
            try:
                if os.readlink(descriptor) == str(path):
                    raise RuntimeError('An active file descriptor still uses the task probe')
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue


def _digest_stream(handle) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    while block := handle.read(CHUNK_BYTES):
        digest.update(block)
        size += len(block)
    return size, digest.hexdigest()


def _compact(case: str) -> dict:
    directory = ROOT / 'runs' / case
    source = directory / 'integration/parameter_probes/rank0.jsonl'
    if source.resolve() != source or not source.is_file() or not source.is_relative_to(ROOT / 'runs'):
        raise RuntimeError('Probe path is missing, aliased, or outside the owned run directory')
    for rank in range(8):
        complete = json.loads((directory / f'complete-rank{rank}.json').read_bytes())
        if complete.get('global_step') != 200:
            raise RuntimeError('Only completed 200-step probes may be compacted')
    _assert_not_open(source)
    before = _signature(source)
    target = source.with_suffix('.jsonl.gz')
    partial = source.with_suffix('.jsonl.gz.task-partial')
    if target.exists() or partial.exists():
        raise RuntimeError('Refusing to replace an existing compressed artifact')
    digest = hashlib.sha256()
    size = 0
    with source.open('rb') as original, partial.open('xb') as output:
        with gzip.GzipFile(filename='', mode='wb', fileobj=output, compresslevel=6, mtime=0) as compressed:
            while block := original.read(CHUNK_BYTES):
                digest.update(block)
                size += len(block)
                compressed.write(block)
        output.flush()
        os.fsync(output.fileno())
    with gzip.open(partial, 'rb') as restored:
        restored_size, restored_sha = _digest_stream(restored)
    if (restored_size, restored_sha) != (size, digest.hexdigest()) or _signature(source) != before:
        raise RuntimeError('Probe changed or compressed bytes did not reconstruct exactly; original preserved')
    with partial.open('rb') as compressed:
        compressed_size, compressed_sha = _digest_stream(compressed)
    _assert_not_open(source)
    if _signature(source) != before:
        raise RuntimeError('Probe changed before replacement; original preserved')
    # Hard-link creation is exclusive, unlike replace(), so an unexpected artifact is never overwritten.
    os.link(partial, target)
    partial.unlink()
    source.unlink()
    return {
        'case': case, 'original_path': str(source), 'compressed_path': str(target),
        'original_bytes': size, 'original_sha256': digest.hexdigest(),
        'compressed_bytes': compressed_size, 'compressed_sha256': compressed_sha,
        'saved_bytes': size - compressed_size, 'lossless_readback_verified': True,
        'completed_steps_each_rank': 200, 'active_source_descriptors': 0,
        'scope_zh': '仅把已完成的本任务诊断JSONL换为逐字节可还原的gzip；其余日志、checkpoint和用户文件不变。',
    }


def main() -> None:
    """Compact the explicit completed probe list and retain a bounded audit receipt."""
    os.nice(19)
    if shutil.which('ionice'):
        subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True, capture_output=True)
    receipt_path = ROOT / 'evidence/runtime-probes-lossless-compaction-20261016.json'
    receipt = {'started_wall': time.time(), 'results': [], 'status': 'running', 'npu_devices_used': []}
    with receipt_path.open('x', encoding='utf-8') as handle:
        json.dump(receipt, handle)
    try:
        for case in CASES:
            record = _compact(case)
            receipt['results'].append(record)
            receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            print(json.dumps(record, ensure_ascii=False), flush=True)
        receipt['status'] = 'PASS'
    except Exception as error:
        receipt.update(status='incomplete', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt['finished_wall'] = time.time()
        receipt['saved_bytes'] = sum(item['saved_bytes'] for item in receipt['results'])
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
