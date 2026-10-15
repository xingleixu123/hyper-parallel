set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=1
export PYTHONPATH=/alignment/phase12/tools:/alignment/tools:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd /state/jt-recipe-20261014/work-weighted212-mf-reduction-eager
exec > /alignment/phase12/runs/weighted212-mf-reduction-eager.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/weighted212-mf-reduction-eager.exit' EXIT
msprof --output=/state/jt-recipe-20261014/weighted212-mf-reduction-eager/profiler --application='python /alignment/phase12/tools/replay_weighted212_mf.py --mode eager' --ascendcl=on --ge-api=l1 --runtime-api=on --task-time=l0 --ai-core=off --storage-limit=500MB
