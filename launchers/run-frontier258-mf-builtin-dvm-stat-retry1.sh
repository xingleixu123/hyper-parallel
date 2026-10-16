set -euo pipefail
source /workspace/env.sh
export MS_DEV_GRAPH_KERNEL_FLAGS=--dump_as_text
export ASCEND_RT_VISIBLE_DEVICES=5
export PYTHONPATH=/alignment/phase12/tools:/workspace:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CLOSE_MATMUL_K_SHIFT=0 P12_NS_EAGER_SCOPE=all
cd /state/jt-recipe-20261014/work-frontier258-mf-builtin-dvm-stat-retry1
exec > /alignment/phase12/runs/frontier258-mf-builtin-dvm-stat-retry1.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/frontier258-mf-builtin-dvm-stat-retry1.exit' EXIT
export MINDSPORE_DUMP_CONFIG=/alignment/phase12/configs/frontier258-mf-builtin-dvm-stat-retry1.json
export P12_DUMP_REPORT=/alignment/phase12/evidence/frontier258-mf-builtin-dvm-stat-retry1.json
python /alignment/phase12/tools/replay_mf_frontier258_builtin_dump.py
