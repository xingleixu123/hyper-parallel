set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=1
export PYTHONPATH=/alignment/phase12/tools-f637:/workspace:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd /state/jt-recipe-20261014/work-head207-gemm-hp-profile
exec > /alignment/phase12/runs/head207-gemm-hp-profile.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/head207-gemm-hp-profile.exit' EXIT
msprof --output=/state/jt-recipe-20261014/head207-gemm-hp-profile/profiler --application='/state/jt-recipe-20261014/jt-f637-pr192-venv/bin/python /alignment/phase12/tools-f637/profile_hp_head207_routes.py' --ascendcl=on --ge-api=l1 --runtime-api=on --task-time=l0 --ai-core=off --storage-limit=500MB --aicore-shape=on
