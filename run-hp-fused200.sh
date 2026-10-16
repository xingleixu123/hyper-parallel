set -euo pipefail
source /workspace/env.sh
export PYTHONPATH=/workspace OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
ROOT=/alignment/phase13
STATE=/state/jt-clean-fused-20261017
exec > "$ROOT/runs/hp-fused200.log" 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase13/runs/hp-fused200.exit' EXIT
mkdir -p "$STATE/work"
/workspace/venv/bin/python -m venv --without-pip --system-site-packages "$STATE/venv"
/workspace/venv/bin/python - <<'SETUP'
from pathlib import Path
import sys,json,hashlib
p=Path('/state/jt-clean-fused-20261017/venv/lib')/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages'
(p/'base-runtime.pth').write_text('/workspace/venv/lib/python3.12/site-packages\n/state/jt-recipe-20261014/pr-kshift-venv/lib/python3.12/site-packages\n')
root=Path('/alignment/phase13/fused-source')
manifest=json.loads(Path('/alignment/phase13/evidence/source-fused-manifest.json').read_text())
for name,expected in manifest['files'].items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise ValueError(name)
print('Source manifest verified',manifest['commit'],len(manifest['files']))
SETUP
"$STATE/venv/bin/python" -m pip install --use-pep517 --no-deps --no-build-isolation -e "$ROOT/fused-source"
export ASCEND_RT_VISIBLE_DEVICES=4 CLOSE_MATMUL_K_SHIFT=0
cd "$STATE/work"
"$STATE/venv/bin/python" -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=31819 "$ROOT/tools/observe_clean_hp.py" --config "$ROOT/configs/hp-fused200.yaml" --output "$ROOT/runs/hp-fused200" --offline /state/jt-recipe-20261014/hp-offline-step200 --source-root "$ROOT/fused-source"
