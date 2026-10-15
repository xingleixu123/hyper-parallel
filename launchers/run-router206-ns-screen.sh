set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=4
export PYTHONPATH=/alignment/phase12/tools-f637:/workspace:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CLOSE_MATMUL_K_SHIFT=0
exec > /alignment/phase12/runs/router206-ns-screen.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/router206-ns-screen.exit' EXIT
/state/jt-recipe-20261014/jt-f637-pr192-venv/bin/python /alignment/phase12/tools-f637/screen_router206_ns.py
