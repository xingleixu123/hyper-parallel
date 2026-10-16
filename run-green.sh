#!/usr/bin/env bash
set -euo pipefail
source /workspace/env.sh
export PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 HYPER_PARALLEL_TEST_DEVICE_TYPE=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 USE_TF=0
export PATH=/work/venv/bin:$PATH
cd /work/src
python - <<'PY'
from pathlib import Path
import hyper_parallel,hashlib,json
root=Path('/work/src');source=Path(hyper_parallel.__file__).resolve();assert source.is_relative_to(root)
m=json.loads(Path('/work/green-input-manifest.json').read_text());checks={v['path']:hashlib.sha256((root/v['path']).read_bytes()).hexdigest()==v['sha256'] for v in m['files']};assert all(checks.values());print(source,checks)
PY
set +e
python -m pytest -p no:cacheprovider -q tests/ut/models/jt_deepseek_v3/test_modeling_jt_deepseek_v3.py tests/ut/models/jt_deepseek_v3/test_context_parallel.py -k 'mla_head or fa_observation' > /work/green-model.log 2>&1
model_status=$?
printf '%s\n' "$model_status" > /work/green-model.exit
python -m pytest -p no:cacheprovider -q tests/ut/models/jt_deepseek_v3/test_jt_builder.py > /work/green-builder.log 2>&1
builder_status=$?
printf '%s\n' "$builder_status" > /work/green-builder.exit
python -m pytest -p no:cacheprovider -q tests/ut/core/activation_memory/test_sac.py tests/ut/core/activation_memory/test_api.py tests/ut/core/activation_memory/test_checkpoint.py tests/ut/core/activation_memory/test_checkpoint_exclude.py tests/ut/core/activation_memory/test_wrapper.py > /work/green-core.log 2>&1
core_status=$?
printf '%s\n' "$core_status" > /work/green-core.exit
python -m pytest -p no:cacheprovider -q tests/ut/auto_models/distributed/test_activation_checkpoint.py > /work/green-trainer.log 2>&1
trainer_status=$?
printf '%s\n' "$trainer_status" > /work/green-trainer.exit
set -e
test "$model_status" -eq 0 && test "$builder_status" -eq 0 && test "$core_status" -eq 0 && test "$trainer_status" -eq 0
