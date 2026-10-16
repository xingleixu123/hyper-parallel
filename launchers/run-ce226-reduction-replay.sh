set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=2
export PYTHONPATH=/alignment/phase12/tools:/alignment/phase12/tools-f637
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd /state/jt-recipe-20261014/work-ce226-reduction-replay
exec > /alignment/phase12/runs/ce226-reduction-replay.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/ce226-reduction-replay.exit' EXIT
python /alignment/phase12/tools/replay_ce226_reduction.py
