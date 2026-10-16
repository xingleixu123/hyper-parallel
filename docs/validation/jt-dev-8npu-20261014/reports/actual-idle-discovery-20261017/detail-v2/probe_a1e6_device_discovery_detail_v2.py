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

"""Retain production device discovery, then separately record bounded self-only diagnostics."""

from ctypes import CDLL, byref, c_int
import importlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Callable

import torch

from hyper_parallel.models.jt_deepseek_v3.adapter import jt_builder
from hyper_parallel.models import build_options


def _observe(operation: Callable[[], object]) -> dict:
    """Retain a diagnostic exception as data without replacing the prior production result."""
    try:
        value = operation()
        return {'status': 'RETURNED', 'value': value}
    except Exception as error:
        return {'status': 'EXCEPTION', 'error_type': type(error).__name__, 'error': str(error)[:4096]}


def _raw_hal_device_count() -> dict:
    """Use the installed PTA convention without replacing either the raw return code or out value."""
    library = CDLL('libascend_hal.so')
    count = c_int(-1)
    return_code = library.drvGetDevNum(byref(count))
    return {'loader_request': 'libascend_hal.so', 'initial_out_value': -1,
            'return_code': return_code, 'device_count_out': count.value}


def _c_backend_device_count() -> dict:
    """Call the actual C fallback only after the original production JSON has been flushed."""
    module = importlib.import_module('torch_npu')
    backend = getattr(module, '_C')
    count = getattr(backend, '_npu_getDeviceCount')()
    return {'return_value': count, 'return_type': type(count).__name__,
            'torch_npu_file': getattr(module, '__file__', None)}


def _npu_state() -> dict:
    """Observe the two cache flags without changing initialization or selecting a device."""
    return {name: {'present': hasattr(torch.npu, name), 'value': getattr(torch.npu, name, None)}
            for name in ('_cached_device_count', '_initialized')}


def _loaded_driver_libraries() -> list:
    """Record actual mappings of relevant libraries from this process only."""
    with Path('/proc/self/maps').open('r', encoding='utf-8') as stream:
        text = stream.read(1024 * 1024 + 1)
    if len(text) > 1024 * 1024:
        raise ValueError('Self maps exceeds the one-MiB diagnostic limit')
    paths = set()
    for line in text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and ('libascend_hal' in fields[5] or 'libdrvdsmi' in fields[5]):
            paths.add(fields[5])
    return sorted(paths)


def _self_namespace_identity() -> dict:
    """Read this process namespace links without enumerating other processes or host mappings."""
    result = {'pid': os.getpid(), 'namespaces': {}}
    for name in ('pid', 'mnt', 'user', 'net', 'ipc', 'uts', 'cgroup'):
        try:
            result['namespaces'][name] = {'value': os.readlink('/proc/self/ns/' + name)}
        except OSError as error:
            result['namespaces'][name] = {'error_type': type(error).__name__, 'error': str(error)[:4096]}
    return result


def _self_status_fields() -> dict:
    """Keep only the requested capability and sandbox fields from this process status."""
    allowed = ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb', 'NoNewPrivs', 'Seccomp')
    with Path('/proc/self/status').open('r', encoding='utf-8') as stream:
        text = stream.read(65537)
    if len(text) > 65536:
        raise ValueError('Self status exceeds the 64-KiB diagnostic limit')
    values = {}
    for line in text.splitlines():
        name, separator, value = line.partition(':')
        if separator and name in allowed:
            values[name] = value.strip()
    return {'fields': values, 'missing_fields': [name for name in allowed if name not in values]}


def _visible_device_nodes() -> dict:
    """Stat only visible Ascend device nodes, without following links or opening the devices."""
    directory = Path('/dev')
    fixed = {directory / name for name in ('davinci_manager', 'devmm_svm', 'hisi_hdc')}
    paths = fixed | {path for path in directory.glob('davinci[0-9]*')
                     if re.fullmatch(r'davinci[0-9]+', path.name)}
    if len(paths) > 128:
        raise ValueError('Visible Ascend device node count exceeds the diagnostic limit')
    result = {'nodes': [], 'not_present': [], 'errors': []}
    for path in sorted(paths):
        try:
            info = path.lstat()
            result['nodes'].append({'path': str(path), 'mode': oct(info.st_mode),
                                    'is_character_device': stat.S_ISCHR(info.st_mode), 'rdev': info.st_rdev,
                                    'rdev_major': getattr(os, 'major')(info.st_rdev),
                                    'rdev_minor': getattr(os, 'minor')(info.st_rdev)})
        except FileNotFoundError:
            result['not_present'].append(str(path))
        except OSError as error:
            result['errors'].append({'path': str(path), 'error_type': type(error).__name__,
                                     'error': str(error)[:4096]})
    return result


def _own_device_cgroup_rules() -> dict:
    """Read only the fixed container-visible devices controller file, without searching other cgroups."""
    path = Path('/sys/fs/cgroup/devices/devices.list')
    with path.open('r', encoding='utf-8') as stream:
        text = stream.read(65537)
    if len(text) > 65536:
        raise ValueError('Container device rules exceed the 64-KiB diagnostic limit')
    return {'path': str(path), 'lines': text.splitlines()}


def post_production_diagnostics(production_status: str) -> dict:
    """Query diagnostic boundaries independently while retaining their order and original status."""
    result = {
        'record_type': 'post_production_diagnostics',
        'production_status': production_status,
        'scope': 'Post-production diagnostic queries only; no tensors, set_device or training',
        'environment': {name: os.environ.get(name) for name in (
            'ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES', 'ASCEND_HOME_PATH',
            'LD_LIBRARY_PATH', 'TORCH_DEVICE_BACKEND_AUTOLOAD', 'CLOSE_MATMUL_K_SHIFT')},
        'npu_state_before': _observe(_npu_state),
    }
    result['hal_drvGetDevNum'] = _observe(_raw_hal_device_count)
    result['torch_npu_C_getDeviceCount'] = _observe(_c_backend_device_count)
    result['npu_state_after'] = _observe(_npu_state)
    result['loaded_driver_libraries'] = _observe(_loaded_driver_libraries)
    result['self_namespace_identity'] = _observe(_self_namespace_identity)
    result['uid'] = _observe(lambda: getattr(os, 'getuid')())
    result['euid'] = _observe(lambda: getattr(os, 'geteuid')())
    result['self_status'] = _observe(_self_status_fields)
    result['visible_device_nodes'] = _observe(_visible_device_nodes)
    result['own_device_cgroup_rules'] = _observe(_own_device_cgroup_rules)
    return result


def _emit_post_production_diagnostics(production_status: str) -> None:
    """Keep the production status even if an unexpected diagnostic or serialization error occurs."""
    try:
        result = post_production_diagnostics(production_status)
        text = json.dumps(result, ensure_ascii=False)
    except Exception as error:
        text = json.dumps({'record_type': 'post_production_diagnostics',
                           'production_status': production_status,
                           'diagnostic_error': {'error_type': type(error).__name__, 'error': str(error)[:4096]}},
                          ensure_ascii=False)
    print(text, flush=True)


def main() -> None:
    """Record discovery facts and fail on the observed noninteger NPU index symptom."""
    result = {
        'source_file': build_options.__file__,
        'torch_file': torch.__file__,
        'torch_version': torch.__version__,
        'cached_npu_available': build_options.IS_NPU_AVAILABLE,
        'cached_cuda_available': build_options.IS_CUDA_AVAILABLE,
        'device_type': build_options.get_device_type(),
        'npu_device_count': torch.npu.device_count(),
        'npu_available_now': torch.npu.is_available(),
        'environment': {name: os.environ.get(name) for name in (
            'ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES', 'ASCEND_HOME_PATH',
            'LD_LIBRARY_PATH', 'TORCH_DEVICE_BACKEND_AUTOLOAD', 'CLOSE_MATMUL_K_SHIFT')},
        'scope': 'Device discovery and torch.device construction only; no model, tensors or training',
    }
    try:
        index = jt_builder.get_device_id()
        result.update(device_index=repr(index), device_index_type=type(index).__name__)
        device = torch.device('npu', index)
        result['constructed_device'] = str(device)
        result['status'] = 'PASS' if build_options.IS_NPU_AVAILABLE and isinstance(index, int) else 'FAIL'
    except (RuntimeError, TypeError, ValueError) as error:
        result.update(status='FAIL', error_type=type(error).__name__, error=str(error))
    result['loaded_driver_libraries'] = sorted({line.split()[-1] for line in
        Path('/proc/self/maps').read_text().splitlines() if 'libascend_hal' in line or 'libdrvdsmi' in line})
    print(json.dumps(result, ensure_ascii=False), flush=True)
    _emit_post_production_diagnostics(result['status'])
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
