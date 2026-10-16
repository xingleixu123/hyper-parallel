set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=6
export PYTHONPATH=/alignment/phase12/tools-f637:/workspace:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CLOSE_MATMUL_K_SHIFT=0
cd /state/jt-recipe-20261014/work-frontier235-global-sqrt-replay
exec > /alignment/phase12/runs/frontier235-global-sqrt-replay.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/frontier235-global-sqrt-replay.exit' EXIT
/state/jt-recipe-20261014/jt-f637-pr192-venv/bin/python /alignment/phase12/tools-f637/replay_frontier235_global_sqrt.py
