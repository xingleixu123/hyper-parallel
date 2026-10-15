#!/usr/bin/env bash
set -euo pipefail
source /workspace/env.sh
export PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HYPER_PARALLEL_TEST_DEVICE_TYPE=cpu OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
test_root=/work/fixes/trainer-npu-hf32-cpu
test_venv=/work/venvs/npu-precision-tests
test ! -e /work/evidence/npu-precision-ut-before.log
test ! -e /work/evidence/npu-precision-ut-after.log
python -m venv --system-site-packages "$test_venv"
printf '%s\n' /workspace/venv/lib/python3.12/site-packages > "$test_venv/lib/python3.12/site-packages/workspace-base.pth"
export PATH="$test_venv/bin:$PATH"
python -m pip install --no-index --find-links=/work/tmp/issue211-test-wheels pytest==9.1.1
python -m pip install --no-deps --no-build-isolation --use-pep517 -e "$test_root"
cd /work
python - <<'PY'
from pathlib import Path
import hyper_parallel
source = Path(hyper_parallel.__file__).resolve()
print('EDITABLE_IMPORT', source)
assert source.is_relative_to(Path('/work/fixes/trainer-npu-hf32-cpu'))
PY
cd "$test_root"
set +e
python -m pytest -p no:cacheprovider -q tests/ut/trainer/runtime/test_random.py > /work/evidence/npu-precision-ut-before.log 2>&1
red_status=$?
set -e
printf '%s\n' "$red_status" > /work/evidence/npu-precision-ut-before.exit
test "$red_status" -eq 1
cp /work/tmp/npu-hf32-random-fixed.py "$test_root/hyper_parallel/trainer/runtime/random.py"
set +e
python -m pytest -p no:cacheprovider -q tests/ut/trainer/runtime/test_random.py > /work/evidence/npu-precision-ut-after.log 2>&1
green_status=$?
set -e
printf '%s\n' "$green_status" > /work/evidence/npu-precision-ut-after.exit
tail -n 18 /work/evidence/npu-precision-ut-before.log
cat /work/evidence/npu-precision-ut-after.log
test "$green_status" -eq 0
python /work/tools/probe_npu_precision_options.py > /work/evidence/npu-precision-option-after.json
python - <<'PY'
import json
from pathlib import Path
data = json.loads(Path('/work/evidence/npu-precision-option-after.json').read_text())
assert data['before_helper'] is True
assert data['after_helper'] is False
assert data['final_npu_initialized'] is False
assert Path(data['source']).is_relative_to(Path('/work/fixes/trainer-npu-hf32-cpu'))
print(json.dumps(data, ensure_ascii=False, indent=2))
PY
