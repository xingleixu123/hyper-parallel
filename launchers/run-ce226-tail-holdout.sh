set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=2
export MS_DEV_GRAPH_KERNEL_FLAGS=--dump_as_text
export PYTHONPATH=/alignment/phase12/tools:/alignment/phase12/tools-f637:${PYTHONPATH:-}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd /state/jt-recipe-20261014/work-ce226-tail-holdout
exec > /alignment/phase12/runs/ce226-tail-holdout.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/ce226-tail-holdout.exit' EXIT
python /alignment/phase12/tools/replay_ce226_tail_holdout.py
