#!/usr/bin/env bash
set -euo pipefail
source /workspace/env.sh
export PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HYPER_PARALLEL_TEST_DEVICE_TYPE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 USE_TF=0
export PATH=/work/venv/bin:$PATH
cd /work
python - <<'PY'
from pathlib import Path
import hashlib,json,hyper_parallel,torch,pytest,sys
root=Path('/work/src')
source=Path(hyper_parallel.__file__).resolve()
if not source.is_relative_to(root):raise ValueError('Stale editable source: '+str(source))
manifest=json.loads(Path('/work/input-manifest-v3.json').read_text())
checks={row['path']:hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256'] for row in manifest['files']}
if not all(checks.values()):raise ValueError(checks)
info={'import_path':str(source),'torch':torch.__version__,'pytest':pytest.__version__,'python':sys.version,'source_checks':checks,'accelerator_initialized':torch.cuda.is_initialized(),'distributed_initialized':torch.distributed.is_initialized()}
Path('/work/environment-v3.json').write_text(json.dumps(info,indent=2));print(json.dumps(info))
PY
cd /work/src
validator=/work/src/hyper_parallel/tools/model_integration/integration_validator.py
trap 'cp /work/candidate-validator.py "$validator"' EXIT
cp /work/base-validator.py "$validator"
set +e
python -m pytest -p no:cacheprovider -q tests/ut/model_integration/test_integration_validator.py > /work/base-ut-v3.log 2>&1
status=$?
set -e
printf '%s\n' "$status" > /work/base-ut-v3.exit
test "$status" -eq 1
cp /work/mro-validator.py "$validator"
set +e
python -m pytest -p no:cacheprovider -q tests/ut/model_integration/test_integration_validator.py > /work/mro-ut-v3.log 2>&1
status=$?
set -e
printf '%s\n' "$status" > /work/mro-ut-v3.exit
test "$status" -eq 1
cp /work/candidate-validator.py "$validator"
set +e
python -m pytest -p no:cacheprovider -q tests/ut/model_integration > /work/candidate-ut-v3.log 2>&1
status=$?
set -e
printf '%s\n' "$status" > /work/candidate-ut-v3.exit
test "$status" -eq 0
cd /work
python /work/import_smoke.py > /work/import-smoke-v3.log 2>&1
python -m pylint --output-format=json /work/src/hyper_parallel/tools/model_integration/integration_validator.py /work/src/tests/ut/model_integration/test_integration_validator.py > /work/pylint-v3.json
