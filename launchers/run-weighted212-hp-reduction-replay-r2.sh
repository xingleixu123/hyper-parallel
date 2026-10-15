set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=2
export PYTHONPATH=/alignment/phase12/tools:/alignment/phase12/tools-f637:/workspace:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CLOSE_MATMUL_K_SHIFT=0
exec > /alignment/phase12/runs/weighted212-hp-reduction-replay-r2.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/weighted212-hp-reduction-replay-r2.exit' EXIT
/state/jt-recipe-20261014/jt-f637-pr192-venv/bin/python /alignment/phase12/tools/replay_weighted212_hp_r2.py
