set -euo pipefail
source /workspace/env.sh
export MS_DEV_GRAPH_KERNEL_FLAGS=--dump_as_text
export ASCEND_RT_VISIBLE_DEVICES=2
export PYTHONPATH=/alignment/phase12/tools:/workspace:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CLOSE_MATMUL_K_SHIFT=0 P12_NS_EAGER_SCOPE=all
cd /state/jt-recipe-20261014/work-frontier396-mf-builtin-stat
exec > /alignment/phase12/runs/frontier396-mf-builtin-stat.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/frontier396-mf-builtin-stat.exit' EXIT
export MINDSPORE_DUMP_CONFIG=/alignment/phase12/configs/frontier396-mf-builtin-stat.json
export P12_DUMP_REPORT=/alignment/phase12/evidence/frontier396-mf-builtin-stat.json
python /alignment/phase12/tools/replay_mf_frontier396_builtin_dump.py
