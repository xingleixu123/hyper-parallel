set -euo pipefail
source /workspace/env.sh
export MS_DEV_GRAPH_KERNEL_FLAGS=--dump_as_text
export ASCEND_RT_VISIBLE_DEVICES=2
export PYTHONPATH=/alignment/phase12/tools:/workspace:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CLOSE_MATMUL_K_SHIFT=0 P12_NS_EAGER_SCOPE=all
cd /state/jt-recipe-20261014/work-frontier328-mf-normalization-scalars
exec > /alignment/phase12/runs/frontier328-mf-normalization-scalars.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/frontier328-mf-normalization-scalars.exit' EXIT
export MINDSPORE_DUMP_CONFIG=/alignment/phase12/configs/frontier328-mf-normalization-scalars.json
export P12_DUMP_REPORT=/alignment/phase12/evidence/frontier328-mf-normalization-scalars.json
python /alignment/phase12/tools/replay_mf_frontier328_builtin_dump.py
