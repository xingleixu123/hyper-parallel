#!/usr/bin/env bash
set -euo pipefail
source /workspace/env.sh
export PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1
test ! -e /work/evidence/r125-f637-runtime-preflight.json
test ! -e /work/venvs/validation
python -m venv --system-site-packages /work/venvs/validation
printf '%s\n' /workspace/venv/lib/python3.12/site-packages > /work/venvs/validation/lib/python3.12/site-packages/workspace-base.pth
export PATH="/work/venvs/validation/bin:$PATH"
python -m pip install --no-deps --no-build-isolation --use-pep517 -e /workspace/hyper-parallel
cd /work
python - <<'PY'
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import datasets
import hyper_parallel
from hyper_parallel.trainer.config.parser import parse_training_args

root = Path('/work')
source = Path(hyper_parallel.__file__).resolve()
expected = Path('/workspace/hyper-parallel/hyper_parallel')
if expected not in source.parents:
    raise RuntimeError(f'Incorrect editable import: {source}')
config = parse_training_args(['/work/configs/r125-f637-fp32-fsdp8-0-200.yaml'])
data_root = root / 'data/sft_data_4k'
dataset = datasets.load_from_disk(str(data_root))
if len(dataset) != 4710 or config.training.train_iters != 200:
    raise RuntimeError('Unexpected dataset or step count')
data_files = {}
for path in sorted(data_root.iterdir()):
    if path.is_file():
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        data_files[path.name] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
result = {
    'status': 'PASS', 'scope': 'CPU only; no NPU devices',
    'python': sys.version, 'import_path': str(source),
    'source': subprocess.check_output(['git', '-C', '/workspace/hyper-parallel', 'rev-parse', 'HEAD'], text=True).strip(),
    'tree': subprocess.check_output(['git', '-C', '/workspace/hyper-parallel', 'rev-parse', 'HEAD^{tree}'], text=True).strip(),
    'versions': {name: importlib.metadata.version(name) for name in ('torch', 'torch-npu', 'datasets', 'pyarrow', 'transformers')},
    'ascend_home': os.environ.get('ASCEND_HOME_PATH'),
    'train_iters': config.training.train_iters, 'dataset_rows': len(dataset),
    'dataset_columns': dataset.column_names,
    'sample_lengths': [{key: len(value) if isinstance(value, list) else None for key, value in dataset[index].items()} for index in (0, len(dataset) - 1)],
    'data_files': data_files,
}
(root / 'evidence/r125-f637-runtime-preflight.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
PY
