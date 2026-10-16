set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=4
export PYTHONPATH=/workspace
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CLOSE_MATMUL_K_SHIFT=0
cd /state/jt-clean-rerun-20261017/work-hp-product200
exec > /alignment/phase13/runs/hp-product200.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase13/runs/hp-product200.exit' EXIT
/state/jt-clean-rerun-20261017/venv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=31817 /alignment/phase13/tools/observe_clean_hp.py --config /alignment/phase13/configs/hp-product200.yaml --output /alignment/phase13/runs/hp-product200 --offline /state/jt-recipe-20261014/hp-offline-step200 --source-root /alignment/phase13/hyper-parallel
