set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=6
export PYTHONPATH=/alignment/phase12/tools-f637:/workspace:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CLOSE_MATMUL_K_SHIFT=0 P12_NS_EAGER_SCOPE=all
cd /state/jt-recipe-20261014/work-frontier258-norm-expression-screen
exec > /alignment/phase12/runs/frontier258-norm-expression-screen.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/frontier258-norm-expression-screen.exit' EXIT
/state/jt-recipe-20261014/jt-f637-pr192-venv/bin/python /alignment/phase12/tools-f637/replay_hp_frontier258_norm_screen.py
