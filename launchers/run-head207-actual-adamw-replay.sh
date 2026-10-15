set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=2
export PYTHONPATH=/alignment/phase12/tools-f637:/workspace:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
exec > /alignment/phase12/runs/head207-actual-adamw-replay.log 2>&1
trap 'code=$?; printf "%s
" "$code" > /alignment/phase12/runs/head207-actual-adamw-replay.exit' EXIT
/state/jt-recipe-20261014/jt-f637-pr192-venv/bin/python /alignment/phase12/tools-f637/replay_head207_actual_adamw.py
