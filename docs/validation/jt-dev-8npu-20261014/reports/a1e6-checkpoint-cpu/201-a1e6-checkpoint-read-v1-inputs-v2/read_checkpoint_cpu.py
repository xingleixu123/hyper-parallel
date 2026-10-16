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

"""Read the pinned step-200 checkpoint on CPU without a model or process group.

--headers-only uses the collected safetensors headers and small local files; it
does not claim tensor-value or extra-state validation. Full mode rehashes every
file, loads each small extra_state on CPU and checks tensor finiteness in bounded
slices. Neither mode proves that distributed/NPU resume restores these objects.
"""

import argparse
import ast
from collections import Counter, defaultdict
import dataclasses
import datetime
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import pickle
import random
import struct
import sys
import time
import types
from typing import Any, Optional


SOURCE_COMMIT = '544526f7db8342c607244fd41009b2a74adb84c8'
SOURCE_PINS = {
    'hyper_parallel/core/distributed_checkpoint/metadata.py':
        '4a263925e584f5dfd2172adb521d887a3d13333a4b52708aa8691ac9c59ed9c4',
    'hyper_parallel/core/distributed_checkpoint/storage.py':
        'c00ccedd6d24981c296ce5a3f05e6b71663239d395f2c010ea2d0c9fea76b61b',
    'hyper_parallel/core/distributed_checkpoint/filesystem_storage.py':
        '79370c1a945e98a7dc091c96e01c42e18646e3d7bcf84934f7948603b076d692',
    'hyper_parallel/core/distributed_checkpoint/standard_planner.py':
        'b6b8e59cc98e46cfd6e843e192164d5b7f609b7753e8d14a6cdfd0383a0b518d',
    'hyper_parallel/components/checkpoint/dcp_checkpointer.py':
        '94d15f64ad0f3a7877430ea83b6f817ba62afe0933aef6979107987091d62fd3',
    'hyper_parallel/trainer/callbacks/checkpoint_callback.py':
        'd45182bd39beaebd8fb4c4f785ae8f16e456289e75f1e9e0f6264891cb2840bf',
    'hyper_parallel/trainer/callbacks/environ_meter_callback.py':
        'f83e3fa5c5dabe37af299adb016045c2e6430abac010bef4cadc8a6189b9517d',
    'hyper_parallel/core/optimizer/lr_scheduler.py':
        '465b8e70ad160877ab7adab5519f5c93ac4ed2e16fe643f99e85d8d8647a8908',
}
DTYPES = {'torch.float32': ('F32', 4), 'torch.float64': ('F64', 8),
          'torch.float16': ('F16', 2), 'torch.bfloat16': ('BF16', 2),
          'torch.int64': ('I64', 8), 'torch.int32': ('I32', 4),
          'torch.int16': ('I16', 2), 'torch.int8': ('I8', 1),
          'torch.uint8': ('U8', 1), 'torch.bool': ('BOOL', 1)}
METADATA_MODULE = 'hyper_parallel.core.distributed_checkpoint.metadata'
STORAGE_MODULE = 'hyper_parallel.core.distributed_checkpoint.storage'


def require(condition: bool, message: str) -> None:
    """Fail explicitly when a checkpoint invariant is not established."""
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    """Hash a file with an 8 MiB read buffer."""
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            result.update(block)
    return result.hexdigest()


def inside(root: Path, relative: str) -> Path:
    """Resolve only a normal file name below the supplied checkpoint."""
    name = PurePosixPath(relative)
    require(not name.is_absolute() and '..' not in name.parts and '\\' not in relative,
            'Invalid relative checkpoint path: ' + relative)
    path = root.joinpath(*name.parts)
    require(path.resolve() == path and root in path.parents, 'Checkpoint path redirects: ' + relative)
    return path


def inventory(checkpoint: Path, expected: dict, headers_only: bool) -> dict:
    """Require the exact file set and capture identity/size/mtime for the read window."""
    names = {name for name in expected if not headers_only or not name.endswith('.safetensors')}
    actual = set()
    snapshot = {}
    for directory, subdirectories, filenames in os.walk(checkpoint, followlinks=False):
        for name in subdirectories + filenames:
            path = Path(directory) / name
            require(not path.is_symlink(), 'Checkpoint contains a symbolic link: ' + str(path))
        for name in filenames:
            path = Path(directory) / name
            relative = path.relative_to(checkpoint).as_posix()
            stat = path.stat()
            require(path.is_file(), 'Checkpoint contains a non-regular file: ' + relative)
            actual.add(relative)
            snapshot[relative] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    require(actual == names, 'Checkpoint file set differs: missing=' + str(sorted(names - actual))
            + ', unexpected=' + str(sorted(actual - names)))
    return snapshot


def finite_primitives(value: Any) -> None:
    """Reject non-finite primitive optimizer scalars as well as unexpected value classes."""
    if isinstance(value, dict):
        for key, item in value.items():
            finite_primitives(key)
            finite_primitives(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite_primitives(item)
    elif type(value) is float:
        require(math.isfinite(value), 'Non-finite BYTE_IO scalar')
    else:
        require(value is None or type(value) in (str, int, bool), 'Unexpected BYTE_IO value type')


def load_types(source: Path) -> tuple[Any, Any]:
    """Load the actual inert dataclasses, avoiding the hyper_parallel package initializer."""
    for relative, expected in SOURCE_PINS.items():
        require(digest(source / relative) == expected, 'Source changed: ' + relative)
    metadata_path = source / 'hyper_parallel/core/distributed_checkpoint/metadata.py'
    spec = importlib.util.spec_from_file_location('cpu_checkpoint_metadata_types', metadata_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    storage_module_name = 'cpu_checkpoint_storage_types'
    storage = types.ModuleType(storage_module_name)
    sys.modules[storage_module_name] = storage
    storage.__dict__.update(dataclass=dataclasses.dataclass, Optional=Optional)
    tree = ast.parse((source / 'hyper_parallel/core/distributed_checkpoint/storage.py').read_text())
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == 'StorageInfo')
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<pinned StorageInfo>', 'exec'), storage.__dict__)
    return module, storage


class MetadataReader(pickle.Unpickler):
    """Resolve only the exact persisted DCP classes and tuple-like torch.Size."""

    def __init__(self, stream: Any, metadata_types: Any, storage_types: Any) -> None:
        super().__init__(stream)
        self.metadata_types = metadata_types
        self.storage_types = storage_types

    def find_class(self, module: str, name: str) -> Any:
        """Reject imports or reducers outside the inspected metadata schema."""
        names = {'Metadata', 'MetadataIndex', 'ChunkStorageMetadata', 'ChunkInfo',
                 'TensorProperties', 'TensorStorageMetadata', 'BytesStorageMetadata'}
        if module == METADATA_MODULE and name in names:
            return getattr(self.metadata_types, name)
        if module == STORAGE_MODULE and name == 'StorageInfo':
            return self.storage_types.StorageInfo
        if (module, name) == ('torch', 'Size'):
            return tuple
        raise ValueError('Unexpected metadata global: ' + module + '.' + name)


class PrimitiveReader(pickle.Unpickler):
    """Read the actual primitive optimizer BYTE_IO entries without importing reducers."""

    def find_class(self, module: str, name: str) -> Any:
        """This checkpoint's optimizer bytes require no global class references."""
        raise ValueError('Unexpected BYTE_IO global: ' + module + '.' + name)


def contiguous(ranges: list, extent: int, label: str) -> None:
    """Require nonoverlapping intervals to cover exactly the saved byte extent."""
    cursor = 0
    for start, end in sorted(ranges):
        require(type(start) is int and type(end) is int and start == cursor and end >= start,
                'Gap, overlap or invalid byte extent: ' + label)
        cursor = end
    require(cursor == extent, 'Byte coverage differs from file extent: ' + label)


def geometry(shape: tuple, chunks: list, fqn: str) -> None:
    """Prove rectangular chunk coverage by bounds, disjointness and total element count."""
    require(all(type(value) is int and value >= 0 for value in shape), 'Invalid global shape: ' + fqn)
    boxes = []
    for chunk in chunks:
        offsets, sizes = tuple(chunk.offsets), tuple(chunk.sizes)
        require(len(offsets) == len(sizes) == len(shape), 'Chunk rank differs: ' + fqn)
        require(all(type(off) is int and type(size) is int and off >= 0 and size >= 0 and off + size <= dim
                    for off, size, dim in zip(offsets, sizes, shape)), 'Chunk outside global shape: ' + fqn)
        if math.prod(sizes):
            box = tuple((off, off + size) for off, size in zip(offsets, sizes))
            require(not any(all(max(a[0], b[0]) < min(a[1], b[1]) for a, b in zip(box, prior))
                            for prior in boxes), 'Overlapping logical chunks: ' + fqn)
            boxes.append(box)
    require(chunks and sum(math.prod(chunk.sizes) for chunk in chunks) == math.prod(shape),
            'Logical chunk coverage incomplete: ' + fqn)


def read_header(path: Path, expected: dict, headers_only: bool) -> dict:
    """Validate actual/captured safetensors data offsets independently of StorageInfo sentinels."""
    if headers_only:
        header, length = expected['safetensors_header'], expected['header_bytes']
    else:
        with path.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            require(0 < length <= 16 * 1024 ** 2, 'Unexpected safetensors header size')
            header = json.loads(stream.read(length))
        require(header == expected['safetensors_header'] and length == expected['header_bytes'],
                'Safetensors header differs from collected evidence: ' + path.name)
    tensors = {name: value for name, value in header.items() if name != '__metadata__'}
    ranges = []
    element_sizes = dict(DTYPES.values())
    for key, value in tensors.items():
        require(value['dtype'] in element_sizes, 'Unsupported dtype: ' + key)
        require(all(type(dim) is int and dim >= 0 for dim in value['shape']), 'Invalid stored shape: ' + key)
        start, end = value['data_offsets']
        require(end - start == math.prod(value['shape']) * element_sizes[value['dtype']],
                'Tensor byte count differs: ' + key)
        ranges.append((start, end))
    contiguous(ranges, expected['bytes'] - 8 - length, path.name)
    require(expected['data_bytes'] == expected['bytes'] - 8 - length, 'Captured data extent differs')
    return tensors


def inspect_structure(checkpoint: Path, source: Path, manifest: dict, headers_only: bool) -> tuple[dict, dict]:
    """Validate all logical/physical mappings, headers, primitive bytes and step counters."""
    metadata_types, storage_types = load_types(source)
    expected = {row['path']: row for row in manifest['files']}
    require(len(expected) == len(manifest['files']) == 25, 'Expected the exact 25-file checkpoint inventory')
    checked = []
    for relative, row in expected.items():
        path = inside(checkpoint, relative)
        if headers_only and relative.endswith('.safetensors'):
            continue
        require(path.stat().st_size == row['bytes'] and digest(path) == row['sha256'],
                'Checkpoint size or SHA differs: ' + relative)
        checked.append(relative)
    with inside(checkpoint, '.metadata').open('rb') as stream:
        metadata = MetadataReader(stream, metadata_types, storage_types).load()
        require(stream.read(1) == b'', 'Trailing bytes after metadata')
    require(isinstance(metadata, metadata_types.Metadata) and metadata.version == '1.0', 'Unexpected metadata version')
    state, storage = metadata.state_dict_metadata, metadata.storage_data
    require(set(metadata.planner_data) == set(state), 'Planner/state key sets differ')
    for fqn, path in metadata.planner_data.items():
        require(path and '.'.join(map(str, path)) == fqn, 'Planner FQN path differs: ' + fqn)
    headers = {name: read_header(inside(checkpoint, name), row, headers_only)
               for name, row in expected.items() if name.endswith('.safetensors')}
    mapping, byte_ranges, byte_values, seen = defaultdict(dict), defaultdict(list), {}, set()
    for index, location in storage.items():
        require(isinstance(index, metadata_types.MetadataIndex)
                and isinstance(location, getattr(storage_types, 'StorageInfo')),
                'Unexpected storage/index class')
        require(index.fqn in state and location.relative_path in expected, 'Storage names an unknown item/file')
        logical = (index.fqn, tuple(index.offset))
        require(logical not in seen, 'Duplicate logical storage index')
        seen.add(logical)
        entry = state[index.fqn]
        if isinstance(entry, metadata_types.TensorStorageMetadata):
            require(location.offset == 0 and location.length == -1, 'Tensor StorageInfo sentinels differ')
            chunks = [chunk for chunk in entry.chunks if tuple(chunk.offsets) == tuple(index.offset)]
            require(len(chunks) == 1, 'StorageIndex does not identify exactly one chunk: ' + index.fqn)
            if index.index is not None:
                require(type(index.index) is int and 0 <= index.index < len(entry.chunks)
                        and entry.chunks[index.index] == chunks[0], 'Incorrect optional chunk ordinal')
            key = location.tensor_key or index.fqn
            require(location.relative_path in headers and key in headers[location.relative_path], 'Missing tensor key')
            require(key not in mapping[location.relative_path], 'Physical tensor key reused')
            header = headers[location.relative_path][key]
            require(entry.properties.dtype in DTYPES and DTYPES[entry.properties.dtype][0] == header['dtype'],
                    'Tensor dtype differs: ' + index.fqn)
            require(tuple(header['shape']) == tuple(chunks[0].sizes), 'Tensor chunk shape differs: ' + index.fqn)
            mapping[location.relative_path][key] = {'fqn': index.fqn, 'shape': header['shape'],
                                                  'dtype': entry.properties.dtype}
        else:
            require(isinstance(entry, metadata_types.BytesStorageMetadata) and not index.offset
                    and location.relative_path.endswith('.bytes') and location.tensor_key is None,
                    'Unexpected BYTE_IO mapping')
            require(type(location.offset) is int and type(location.length) is int and location.offset >= 0
                    and location.length > 0, 'Invalid BYTE_IO interval')
            with inside(checkpoint, location.relative_path).open('rb') as stream:
                stream.seek(location.offset)
                raw = stream.read(location.length)
            require(len(raw) == location.length, 'Truncated BYTE_IO entry')
            stream = io.BytesIO(raw)
            value = PrimitiveReader(stream).load()
            require(stream.read(1) == b'', 'Trailing bytes after BYTE_IO entry')
            finite_primitives(value)
            byte_values[index.fqn] = value
            byte_ranges[location.relative_path].append((location.offset, location.offset + location.length))
    for fqn, entry in state.items():
        if isinstance(entry, metadata_types.TensorStorageMetadata):
            geometry(tuple(entry.size), entry.chunks, fqn)
            require(all((fqn, tuple(chunk.offsets)) in seen for chunk in entry.chunks), 'Chunk lacks StorageInfo')
        else:
            require(fqn in byte_values, 'BYTE_IO entry lacks StorageInfo')
    for name, header in headers.items():
        require(set(mapping[name]) == set(header), 'Unreferenced tensor keys: ' + name)
    for name, row in expected.items():
        if name.endswith('.bytes'):
            contiguous(byte_ranges[name], row['bytes'], name)
    steps = {key: value for key, value in byte_values.items() if key.endswith('.step')}
    require(steps and all(type(value) is int and value == 200 for value in steps.values()),
            'Optimizer step entries do not all equal 200')
    summary = {'logical_items': len(state), 'tensor_items': sum(isinstance(v, metadata_types.TensorStorageMetadata)
                                                              for v in state.values()),
               'byte_items': len(byte_values), 'physical_storage_items': len(storage),
               'physical_tensor_chunks': sum(len(v) for v in mapping.values()),
               'tensor_dtypes': dict(Counter(v.properties.dtype for v in state.values()
                                            if isinstance(v, metadata_types.TensorStorageMetadata))),
               'optimizer_step_entries': len(steps), 'optimizer_step': 200,
               'model_tensor_items': sum(name.startswith('model.') and isinstance(v, metadata_types.TensorStorageMetadata)
                                         for name, v in state.items()),
               'files_rehashed': checked, 'all_25_files_rehashed': len(checked) == 25,
               'header_source': 'collected manifest only' if headers_only else 'actual files compared to manifest'}
    return summary, dict(mapping)


def cpu_runtime() -> tuple[Any, Any]:
    """Load optional CPU libraries only for full validation, without backend autoload."""
    os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch_module = importlib.import_module('torch')
    safetensors = importlib.import_module('safetensors')
    require('torch_npu' not in sys.modules and not torch_module.distributed.is_initialized(),
            'Expected a fresh CPU process without accelerator/distributed initialization')
    torch_module.set_num_threads(4)
    return torch_module, safetensors.safe_open


def tensor_tiles(shape: list, item_bytes: int, limit: int) -> Any:
    """Split large rectangular chunks along any dimension until one tile fits the byte limit."""
    pending = [tuple((0, dim) for dim in shape)]
    while pending:
        box = pending.pop()
        sizes = [end - start for start, end in box]
        if math.prod(sizes) * item_bytes <= limit:
            yield tuple(slice(start, end) for start, end in box), sizes
            continue
        axis = max(range(len(sizes)), key=sizes.__getitem__)
        require(sizes[axis] > 1, 'Slice limit smaller than one element')
        start, end = box[axis]
        middle = (start + end) // 2
        pending.extend([box[:axis] + ((start, middle),) + box[axis + 1:],
                        box[:axis] + ((middle, end),) + box[axis + 1:]])


def read_tensors(checkpoint: Path, mapping: dict, limit: int, torch_module: Any, safe_open: Any) -> dict:
    """Read and discard each bounded CPU tile, closing the mmap between tiles."""
    total_bytes, tiles, peak = 0, 0, 0
    files = []
    for relative, tensors in sorted(mapping.items()):
        for key, entry in sorted(tensors.items()):
            for slices, sizes in tensor_tiles(entry['shape'], DTYPES[entry['dtype']][1], limit):
                with safe_open(str(inside(checkpoint, relative)), framework='pt', device='cpu') as reader:
                    value = reader.get_slice(key)[slices] if slices else reader.get_tensor(key)
                    require(list(value.shape) == sizes and str(value.dtype) == entry['dtype']
                            and value.device.type == 'cpu', 'Read-back tensor shape/dtype/device differs')
                    size = value.numel() * value.element_size()
                    require(size <= limit and bool(torch_module.isfinite(value).all()),
                            'Oversized or non-finite tensor tile: ' + key)
                    total_bytes += size
                    peak = max(peak, size)
                    tiles += 1
                    del value
        files.append(relative)
        print(json.dumps({'phase': 'cpu-tensors-read', 'file': relative, 'cumulative_bytes': total_bytes}), flush=True)
    return {'all_tensor_values_finite': True, 'files': files, 'tiles': tiles,
            'tensor_payload_bytes_read': total_bytes, 'maximum_materialized_tensor_tile_bytes': peak,
            'limit_bytes': limit, 'limit_scope': 'one tensor tile; excludes framework overhead, finite mask and OS cache'}


def scalar_counters(value: Any, path: str = '') -> dict:
    """Report only named numeric cursors, excluding data samples or RNG contents."""
    result = {}
    if isinstance(value, dict):
        for name, item in value.items():
            result.update(scalar_counters(item, path + '.' + str(name)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result.update(scalar_counters(item, path + '.' + str(index)))
    elif type(value) is int and any(word in path for word in ('consumed', 'yielded', 'world_size', 'batch_size', 'epoch')):
        require(value >= 0, 'Negative dataloader cursor: ' + path)
        result[path] = value
    return result


def read_extra_state(checkpoint: Path, metrics_dir: Path, torch_module: Any) -> list:
    """Read every rank's small CPU state and compare saved progress with its final metric record."""
    results = []
    scheduler_reference = None
    for rank in range(8):
        path = inside(checkpoint, f'extra_state/extra_state_rank_{rank}.pt')
        extra = torch_module.load(path, map_location='cpu', weights_only=True)
        require(type(extra.get('global_step')) is int and extra['global_step'] == 200, 'Wrong extra_state step')
        require(type(extra.get('epoch')) is int and extra['epoch'] >= 0, 'Invalid extra_state epoch')
        metric_path = metrics_dir / f'metrics-rank{rank}.jsonl'
        with metric_path.open(encoding='utf-8') as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        require(len(records) == 200 and [item['step'] for item in records] == list(range(1, 201))
                and all(item['rank'] == rank for item in records), 'Incomplete metric evidence')
        last = records[-1]
        meter = extra['environ_meter']
        for name in ('consumed_samples', 'consumed_tokens'):
            require(type(meter[name]) is int and meter[name] == last['data/' + name], 'Saved meter differs: ' + name)
        require(meter['consumed_samples'] == 1600, 'Unexpected global sample count')
        schedulers = extra['lr_scheduler']
        require(isinstance(schedulers, dict) and schedulers, 'Missing scheduler state')
        for state in schedulers.values():
            require(state['last_epoch'] == 200 and state['_step_count'] == 201, 'Wrong scheduler cursor')
            require(state['_last_lr'] and all(value == last['training/lr'] for value in state['_last_lr']),
                    'Scheduler LR differs from final metric')
        require(scheduler_reference is None or schedulers == scheduler_reference, 'Schedulers differ across ranks')
        scheduler_reference = schedulers
        rng = extra['rng_state']
        rng_summary = {}
        for name in ('torch_cpu', 'torch_device'):
            value = rng[name]
            require(isinstance(value, torch_module.Tensor) and value.device.type == 'cpu'
                    and value.dtype == torch_module.uint8 and value.ndim == 1 and value.numel() > 0,
                    'Missing or malformed RNG state: ' + name)
            rng_summary[name] = {'bytes': value.numel(), 'sha256': hashlib.sha256(value.numpy().tobytes()).hexdigest()}
        generator = torch_module.Generator(device='cpu')
        generator.set_state(rng['torch_cpu'])
        require(torch_module.equal(generator.get_state(), rng['torch_cpu']), 'CPU RNG roundtrip differs')
        python_rng = random.Random()
        python_rng.setstate(rng['python'])
        require(python_rng.getstate() == rng['python'], 'Python RNG roundtrip differs')
        rng_summary['python'] = {'sha256': hashlib.sha256(pickle.dumps(rng['python'], protocol=4)).hexdigest(),
                                 'local_rng_roundtrip': True}
        dataloader = extra['train_dataloader']
        require(isinstance(dataloader, dict) and dataloader, 'Missing dataloader state')
        counters = scalar_counters(dataloader)
        require(counters, 'No recognized dataloader counters')
        results.append({'rank': rank, 'global_step': 200, 'epoch': extra['epoch'], 'environ_meter': meter,
                        'scheduler_names': sorted(schedulers), 'scheduler_cursors': {
                            name: {'last_epoch': state['last_epoch'], 'step_count': state['_step_count']}
                            for name, state in schedulers.items()},
                        'rng': rng_summary, 'dataloader_counter_structure': counters,
                        'metric_sha256': digest(metric_path), 'device_rng_applied': False})
        del extra
    return results


def main() -> None:
    """Write an auditable CPU-only result, keeping partial inspection distinct from full read-back."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--expected-hashes', type=Path, required=True)
    parser.add_argument('--metrics-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-slice-mib', type=int, default=32)
    parser.add_argument('--headers-only', action='store_true')
    args = parser.parse_args()
    checkpoint, source, output = args.checkpoint.resolve(), args.source.resolve(), args.output.resolve()
    require(checkpoint not in output.parents and source not in output.parents, 'Output must be outside readonly inputs')
    require(not output.exists() and not args.output.is_symlink(), 'Audit output already exists or is a symbolic link')
    require(1 <= args.max_slice_mib <= 32, 'Slice limit must be between 1 and 32 MiB')
    manifest = json.loads(args.expected_hashes.read_bytes())
    require(manifest['source'] == SOURCE_COMMIT, 'Manifest source does not match the reviewed integration')
    started = time.monotonic()
    report = {'status': 'RUNNING', 'source': SOURCE_COMMIT, 'checkpoint': str(checkpoint),
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'manifest_sha256': digest(args.expected_hashes), 'reader_sha256': digest(Path(__file__)),
              'mode': 'headers-only' if args.headers_only else 'full-cpu', 'npu_used': False,
              'distributed_resume_verified': False, 'migration_admitted': False,
              'limits': ['Tensor hashes were first collected after save; not a comparison with save-time tensor hashes.',
                         'CPU mapping/finite/state checks do not execute DCP restore into a model or NPU continuation.',
                         'Device RNG is checked as saved CPU bytes; never applied to an accelerator.',
                         'Dataloader counters establish persisted structure, not proof that future batches replay exactly.']}
    try:
        expected = {row['path']: row for row in manifest['files']}
        before = inventory(checkpoint, expected, args.headers_only)
        report['structure'], mapping = inspect_structure(checkpoint, source, manifest, args.headers_only)
        if args.headers_only:
            report.update(status='PASS_METADATA_BYTES_AND_CAPTURED_HEADERS_ONLY',
                          tensor_values_verified=False, extra_state_unpacked=False)
        else:
            torch_module, safe_open = cpu_runtime()
            report['extra_state'] = read_extra_state(checkpoint, args.metrics_dir, torch_module)
            report['tensor_readback'] = read_tensors(checkpoint, mapping, args.max_slice_mib * 1024 ** 2,
                                                     torch_module, safe_open)
            require(report['tensor_readback']['tensor_payload_bytes_read'] == sum(
                row.get('data_bytes', 0) for row in manifest['files']), 'Read tensor byte total differs')
            report.update(status='PASS_CPU_CHECKPOINT_READBACK_ONLY', tensor_values_verified=True,
                          extra_state_unpacked=True, torch_version=torch_module.__version__)
        require(inventory(checkpoint, expected, args.headers_only) == before,
                'Checkpoint file identity, size or modification time changed during read-back')
        report['input_inventory_stable_during_readback'] = True
        report['input_files_observed'] = len(before)
    except Exception as error:
        report.update(status='FAIL', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'status': report['status'], 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
