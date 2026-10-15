set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH=/alignment/phase12/tools-f637:/workspace:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /state/jt-recipe-20261014/work-head207-gemm-aclop-nz-layout-smoke
exec > /alignment/phase12/runs/head207-gemm-aclop-nz-layout-smoke.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/head207-gemm-aclop-nz-layout-smoke.exit' EXIT
/state/jt-recipe-20261014/jt-f637-pr192-venv/bin/python /alignment/phase12/tools-f637/smoke_head207_aclop_nz_layout.py
