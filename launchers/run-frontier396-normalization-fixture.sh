set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH=/alignment/phase12/tools-f637:/workspace:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd /state/jt-recipe-20261014/work-frontier396-normalization-fixture
exec > /alignment/phase12/runs/frontier396-normalization-fixture.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/frontier396-normalization-fixture.exit' EXIT
/state/jt-recipe-20261014/jt-e213-pr192-venv/bin/python /alignment/phase12/tools-f637/replay_frontier396_normalization.py
