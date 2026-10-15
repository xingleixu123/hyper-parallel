set -o pipefail
source /workspace/env.sh
set -u
export ASCEND_RT_VISIBLE_DEVICES=0 CLOSE_MATMUL_K_SHIFT=0 PYTHONPATH=/alignment/phase12/tools:/alignment/phase12/tools-f637 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/state/jt-recipe-20261014/jt-f637-pr192-venv/bin/python /alignment/phase12/tools/replay_head226_gradient_cause.py > /alignment/phase12/runs/head226-gradient-cause-replay.log 2>&1
code=$?
printf '%s
' "$code" > /alignment/phase12/runs/head226-gradient-cause-replay.exit
exit "$code"
