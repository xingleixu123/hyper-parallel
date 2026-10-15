set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=2
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd /state/jt-recipe-20261014/work-head207-gemm-mf-isolated-eager
exec > /alignment/phase12/runs/head207-gemm-mf-isolated-eager.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/head207-gemm-mf-isolated-eager.exit' EXIT
python /alignment/phase12/tools/replay_mf_head207_isolated.py --mode eager
