set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH=/alignment/phase12/tools:/alignment/tools:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd /state/jt-recipe-20261014/work-weighted212-mf-reduction-graph-r3
exec > /alignment/phase12/runs/weighted212-mf-reduction-graph-r3.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/weighted212-mf-reduction-graph-r3.exit' EXIT
msprof --output=/state/jt-recipe-20261014/weighted212-mf-reduction-graph-r3/profiler --application='python /alignment/phase12/tools/replay_weighted212_mf_r3.py --mode graph' --ascendcl=on --ge-api=l1 --runtime-api=on --task-time=l0 --ai-core=off --storage-limit=500MB
