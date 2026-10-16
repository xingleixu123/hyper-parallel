set -euo pipefail
source /workspace/env.sh
export PYTHONPATH=/workspace:${PYTHONPATH:-} OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export ASCEND_RT_VISIBLE_DEVICES=4 CLOSE_MATMUL_K_SHIFT=0
mkdir -p /state/jt-clean-archive-r2-20261017/train-work
cd /state/jt-clean-archive-r2-20261017/train-work
exec > /alignment/phase13/runs/hp-archive200-r2.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase13/runs/hp-archive200-r2.exit' EXIT
/state/jt-clean-archive-r2-20261017/venv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=31822 /alignment/phase13/tools/observe_archived_hp.py --config /alignment/phase13/configs/hp-archive200-r2.yaml --output /alignment/phase13/runs/hp-archive200-r2 --offline /state/jt-recipe-20261014/hp-offline-step200 --source-root /alignment/phase13/archive-r2-source
