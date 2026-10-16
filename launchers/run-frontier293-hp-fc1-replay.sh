set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=6
export PYTHONPATH=/alignment/phase12/tools-f637:/workspace:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CLOSE_MATMUL_K_SHIFT=0 P12_NS_EAGER_SCOPE=all
cd /state/jt-recipe-20261014/work-frontier293-hp-fc1-replay
exec > /alignment/phase12/runs/frontier293-hp-fc1-replay.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/frontier293-hp-fc1-replay.exit' EXIT
/state/jt-recipe-20261014/jt-f637-pr192-venv/bin/python -m torch.distributed.run --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=31674 /alignment/phase12/tools-f637/replay_hp_frontier293_fc1.py
