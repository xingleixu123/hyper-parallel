#!/usr/bin/env bash
set -euo pipefail
source /workspace/env.sh
export PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 HYPER_PARALLEL_TEST_DEVICE_TYPE=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 USE_TF=0
python -m venv --system-site-packages /work/venv
printf '%s\n' /workspace/venv/lib/python3.12/site-packages > /work/venv/lib/python3.12/site-packages/workspace-base.pth
export PATH=/work/venv/bin:$PATH
python -m pip install --no-index --find-links=/work/wheels pytest==9.1.1
python -m pip install --no-deps --no-build-isolation --use-pep517 -e /work/src
cd /work/src
python - <<'PY'
from pathlib import Path
import hyper_parallel,hashlib,json,sys,importlib.metadata as metadata
root=Path('/work/src');source=Path(hyper_parallel.__file__).resolve();assert source.is_relative_to(root)
m=json.loads(Path('/work/red-input-manifest.json').read_text())
checks={v['path']:hashlib.sha256((root/v['path']).read_bytes()).hexdigest()==v['sha256'] for v in m['files']}
assert all(checks.values());print(source,checks)
Path('/work/environment.json').write_text(json.dumps({'python':sys.version,'packages':{k:metadata.version(k) for k in ['torch','torch_npu','transformers','pytest']},'import_path':str(source),'source_hash_checks':checks},indent=2))
PY
set +e
python -m pytest -p no:cacheprovider -q tests/ut/models/jt_deepseek_v3/test_modeling_jt_deepseek_v3.py tests/ut/models/jt_deepseek_v3/test_jt_builder.py -k 'mla_head_chunk_options or head_chunks_require_complete' > /work/red.log 2>&1
status=$?
set -e
printf '%s\n' "$status" > /work/red.exit
test "$status" -eq 1
