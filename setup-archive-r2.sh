set -euo pipefail
source /workspace/env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/workspace:${PYTHONPATH:-}
ROOT=/alignment/phase13
STATE=/state/jt-clean-archive-r2-20261017
mkdir -p "$STATE"
exec > "$ROOT/runs/archive-r2-setup.log" 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase13/runs/archive-r2-setup.exit' EXIT
/workspace/venv/bin/python -m venv --without-pip --system-site-packages "$STATE/venv"
/workspace/venv/bin/python - <<'PY'
from pathlib import Path
import sys
root = Path('/state/jt-clean-archive-r2-20261017/venv')
version = f'python{sys.version_info.major}.{sys.version_info.minor}'
directory = root / 'lib' / version / 'site-packages'
(directory / 'base-runtime.pth').write_text(
    f'/workspace/venv/lib/{version}/site-packages\n'
    f'/state/jt-recipe-20261014/pr-kshift-venv/lib/{version}/site-packages\n')
PY
"$STATE/venv/bin/python" -m pip install --use-pep517 --no-deps --no-build-isolation -e "$ROOT/archive-r2-source"
cd "$ROOT/archive-r2-source"
"$STATE/venv/bin/python" - <<'PY'
import hashlib
import json
from pathlib import Path
import hyper_parallel
root = Path('/alignment/phase13/archive-r2-source')
if not Path(hyper_parallel.__file__).is_relative_to(root):
    raise ValueError('Editable import escaped the clean source checkout')
print('Verified clean editable source:', hyper_parallel.__file__)

PY
export ASCEND_RT_VISIBLE_DEVICES=4
"$STATE/venv/bin/python" -m pytest -q tests/ut/model_integration/test_jt_mf_archive.py tests/ut/model_integration/test_jt_muon_fused_matrices.py tests/ut/core/optimizer/test_muon.py tests/ut/trainer/test_text_trainer.py
