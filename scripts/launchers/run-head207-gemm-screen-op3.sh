set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=1
export PYTHONPATH=/alignment/phase12/tools-f637:/workspace:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd /state/jt-recipe-20261014/work-head207-gemm-screen-op3
exec > /alignment/phase12/runs/head207-gemm-screen-op3.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/head207-gemm-screen-op3.exit' EXIT
/state/jt-recipe-20261014/jt-f637-pr192-venv/bin/python /alignment/phase12/tools-f637/screen_head207_gemm_routes.py --kernel 3
