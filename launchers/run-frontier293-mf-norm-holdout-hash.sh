set -euo pipefail
source /workspace/env.sh
export MS_DEV_GRAPH_KERNEL_FLAGS=--dump_as_text
export ASCEND_RT_VISIBLE_DEVICES=5
export PYTHONPATH=/alignment/phase12/tools:/workspace:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CLOSE_MATMUL_K_SHIFT=0 P12_NS_EAGER_SCOPE=all
cd /state/jt-recipe-20261014/work-frontier293-mf-norm-holdout-hash
exec > /alignment/phase12/runs/frontier293-mf-norm-holdout-hash.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/frontier293-mf-norm-holdout-hash.exit' EXIT
export MINDSPORE_DUMP_CONFIG=/alignment/phase12/configs/frontier293-mf-norm-holdout-hash.json
export P12_DUMP_REPORT=/alignment/phase12/evidence/frontier293-mf-norm-holdout-hash.json
python /alignment/phase12/tools/replay_mf_frontier293_norm_holdout_hash.py
