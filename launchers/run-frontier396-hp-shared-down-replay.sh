set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH=/alignment/phase12/tools-e213:/alignment/phase12/tools-f637:/workspace:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CLOSE_MATMUL_K_SHIFT=0
cd /state/jt-recipe-20261014/work-frontier396-hp-shared-down-replay
exec > /alignment/phase12/runs/frontier396-hp-shared-down-replay.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/frontier396-hp-shared-down-replay.exit' EXIT
/state/jt-recipe-20261014/jt-e213-pr192-venv/bin/python -m torch.distributed.run --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=31719 /alignment/phase12/tools-f637/replay_hp_frontier396_shared_down.py
