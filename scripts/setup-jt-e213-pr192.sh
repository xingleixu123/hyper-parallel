set -euo pipefail
source /workspace/env.sh
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
exec > /alignment/phase12/runs/jt-e213-pr192-setup.log 2>&1
trap 'code=$?; echo "$code" > /alignment/phase12/runs/jt-e213-pr192-setup.exit' EXIT
/workspace/venv/bin/python -m venv --without-pip --system-site-packages /state/jt-recipe-20261014/jt-e213-pr192-venv
/workspace/venv/bin/python - <<'PY'
from pathlib import Path
import sys
root = Path('/state/jt-recipe-20261014/jt-e213-pr192-venv')
version = f'python{sys.version_info.major}.{sys.version_info.minor}'
directory = root / 'lib' / version / 'site-packages'
(directory / 'base-runtime.pth').write_text(
    f'/workspace/venv/lib/{version}/site-packages\n'
    f'/state/jt-recipe-20261014/pr-kshift-venv/lib/{version}/site-packages\n')
PY
/state/jt-recipe-20261014/jt-e213-pr192-venv/bin/python -m pip install --use-pep517 --no-deps --no-build-isolation -e /alignment/phase12/hp-jt-e213-pr192
cd /alignment/phase12/hp-jt-e213-pr192
/state/jt-recipe-20261014/jt-e213-pr192-venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path
import hyper_parallel
root = Path('/alignment/phase12/hp-jt-e213-pr192')
if not Path(hyper_parallel.__file__).is_relative_to(root):
    raise ValueError('Editable import resolved to another tree')
manifest = json.loads(Path('/alignment/phase12/evidence/jt-e213-pr192-source-manifest.json').read_text())
for record in manifest['files']:
    if hashlib.sha256((root / record['path']).read_bytes()).hexdigest() != record['sha256']:
        raise ValueError(record['path'])
print('Verified editable import and all source hashes:', hyper_parallel.__file__)
PY
export ASCEND_RT_VISIBLE_DEVICES=4
/state/jt-recipe-20261014/jt-e213-pr192-venv/bin/python -m pytest -q tests/ut/model_integration/test_model_integration.py
