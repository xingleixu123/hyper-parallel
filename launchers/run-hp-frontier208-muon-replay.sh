set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=4
export PYTHONPATH=/alignment/phase12/tools-f637:/workspace:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CLOSE_MATMUL_K_SHIFT=0
cd /state/jt-recipe-20261014/work-hp-frontier208-muon-replay
exec > /alignment/phase12/runs/hp-frontier208-muon-replay.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/hp-frontier208-muon-replay.exit' EXIT
/state/jt-recipe-20261014/jt-f637-pr192-venv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=31556 /alignment/phase12/tools-f637/replay_hp_frontier208_muon.py
