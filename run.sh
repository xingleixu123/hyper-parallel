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
cd /work
python - <<'PY'
from pathlib import Path
import hyper_parallel,hashlib,json
root=Path('/work/src');source=Path(hyper_parallel.__file__).resolve()
assert source.is_relative_to(root)
m=json.loads(Path('/work/manifest.json').read_text())
checks={v['path']:hashlib.sha256((root/v['path']).read_bytes()).hexdigest()==v['sha256'] for v in m['files']}
assert all(checks.values());print(source,checks)
PY
cd /work/src
validator=/work/src/hyper_parallel/tools/model_integration/integration_validator.py
trap 'cp /work/candidate-validator.py "$validator"' EXIT
cp /work/base-validator.py "$validator"
set +e
python -m pytest -p no:cacheprovider -q tests/ut/model_integration/test_model_integration.py -k activation_checkpoint_inventory_handles_fsdp_wrapper > /work/red.log 2>&1
status=$?
set -e
printf '%s\n' "$status" > /work/red.exit
test "$status" -eq 1
cp /work/candidate-validator.py "$validator"
set +e
python -m pytest -p no:cacheprovider -q tests/ut/model_integration/test_model_integration.py > /work/green.log 2>&1
status=$?
set -e
printf '%s\n' "$status" > /work/green.exit
test "$status" -eq 0
