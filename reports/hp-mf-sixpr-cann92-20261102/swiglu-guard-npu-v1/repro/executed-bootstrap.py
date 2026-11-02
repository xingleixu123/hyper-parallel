"""Native four-card HP training and checkpoint observations, no arithmetic hooks."""
import argparse
import fnmatch
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import sys
import time

import numpy as np
import torch
import torch_npu
import torch.distributed as dist

ROOT = Path('/work/output')
RANK = int(os.environ['RANK'])
LOCAL = int(os.environ['LOCAL_RANK'])
OUT = ROOT / ('rank-' + str(RANK))
OUT.mkdir(exist_ok=False)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with path.open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def local(t):
    return t.to_local() if hasattr(t, 'to_local') else t


def tensor_record(t):
    v = local(t).detach()
    h = hashlib.sha256()
    # Model and native optimizer storage is FP32; empty local shards are valid.
    flat = v.contiguous().reshape(-1)
    for i in range(0, flat.numel(), 2 ** 20):
        h.update(flat[i:i + 2 ** 20].cpu().numpy().tobytes())
    mesh = getattr(t, 'device_mesh', None)
    return {'global_shape': list(t.shape), 'local_shape': list(v.shape), 'dtype': str(v.dtype),
            'sha256': h.hexdigest(), 'placements': str(getattr(t, 'placements', None)),
            'mesh': None if mesh is None else {'shape': [int(x) for x in mesh.shape],
                'ranks': [int(x) for x in mesh.rank_list], 'names': list(mesh.mesh_dim_names or []),
                'coordinate': [int(x) for x in mesh.get_coordinate()]}}


def normalize(v):
    if torch.is_tensor(v):
        return tensor_record(v)
    if isinstance(v, dict):
        return {str(k): normalize(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [normalize(x) for x in v]
    if isinstance(v, (torch.dtype, torch.device)):
        return str(v)
    return v


def digest_object(value):
    return hashlib.sha256(json.dumps(normalize(value), sort_keys=True).encode()).hexdigest()


def rng():
    from hyper_parallel.trainer.runtime.device import get_device_rng_state
    n = np.random.get_state()
    return {'torch_cpu': digest_object(torch.get_rng_state()),
            'torch_device': digest_object(get_device_rng_state()),
            'python': digest_object(random.getstate()),
            'numpy': digest_object([n[0], n[1].tolist(), *n[2:]])}


def fields(batch):
    result = {}
    for name, value in batch.items():
        a = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        a = np.ascontiguousarray(a)
        result[name] = {'shape': list(a.shape), 'dtype': str(a.dtype),
                        'sha256_c_order': hashlib.sha256(a.tobytes()).hexdigest()}
    return result


def mapped_libraries(stage):
    files = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) != 6 or '.so' not in parts[5] or not re.search(
                r'ascend|cann|acl|hccl|libtorch_npu|omni_training', parts[5], re.I):
            continue
        require(not parts[5].endswith(' (deleted)'), 'deleted loaded shared library')
        path = Path(parts[5]).resolve(strict=True)
        require(not re.search(r'9[._-]1(?:[._/-]|$)', str(path)), 'unexpected CANN9.1 library')
        if '/ascend-toolkit/' in str(path) or '/cann920/' in str(path):
            require(str(path).startswith('/work/runtime/cann920/cann-9.2.0/'), 'unexpected toolkit path')
        files[str(path)] = sha(path)
    write(OUT / ('libraries-' + stage + '.json'), files)
    return files


def handshake():
    require(int(os.environ['WORLD_SIZE']) == 4 and RANK == LOCAL and 0 <= RANK < 4, 'rank environment')
    require(torch.__version__ == '2.12.0+cpu' and importlib.metadata.version('torch-npu') == '2.12.0.post3',
            'runtime versions')
    require(sys.version_info[:3] == (3, 12, 14) and np.__version__ == '1.26.4' and
            importlib.metadata.version('transformers') == '5.21.0', 'Python/numpy/transformers versions')
    for module in (torch, torch_npu):
        require(Path(module.__file__).resolve().is_relative_to(Path('/work/env')), 'unexpected framework origin')
    assets = {'/work/pyext/omni_training_custom_ops/custom_ops_lib.cpython-312-aarch64-linux-gnu.so':
              'e222d9b73fae8fb9212b9f83c5fcccb6583897ec3709042a46c21838b26e8ee7',
              '/work/task/vendor-opp/vendors/omni_training_custom_transformer/op_api/lib/libcust_opapi.so':
              '546929376c5695fc2fc55752152ec4057548b79c79861a79e88cc65188ddc0a7'}
    for path, digest in assets.items():
        require(sha(path) == digest, 'fused CANN9.2 asset identity mismatch')
    require(torch.npu.device_count() == 4, 'expected four visible devices')
    torch.npu.set_device(LOCAL)
    x = torch.ones(1, device='npu:' + str(LOCAL))
    require(x.item() == 1, 'device allocation failed')
    stat = Path('/proc/self/stat').read_text().split(') ', 1)[1].split()
    ready = {'rank': RANK, 'local_rank': LOCAL, 'world_size': 4, 'pid': os.getpid(), 'starttick': int(stat[19]),
             'tag': os.environ['HP_FOURCARD_TAG'], 'visible': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
             'namespaces': {k: os.readlink('/proc/self/ns/' + k) for k in ('mnt', 'pid', 'ipc', 'net', 'uts')},
             'cgroup': Path('/proc/self/cgroup').read_text(),
             'status': [s for s in Path('/proc/self/status').read_text().splitlines()
                        if s.startswith(('Uid:', 'Gid:', 'CapEff:', 'Seccomp:', 'NoNewPrivs:'))]}
    write(OUT / 'device-ready.json', ready)
    deadline = time.monotonic() + 180
    while not (ROOT / 'physical-mapping-release.json').exists():
        require(time.monotonic() < deadline, 'device release timeout')
        time.sleep(.2)
    release = json.loads((ROOT / 'physical-mapping-release.json').read_text())
    require(release['tag'] == ready['tag'] and release['run_id'] == os.environ['HP_FOURCARD_RUN_ID'],
            'physical release identity')


