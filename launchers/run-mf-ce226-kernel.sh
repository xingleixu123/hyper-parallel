set -euo pipefail
source /workspace/env.sh
export MS_DEV_GRAPH_KERNEL_FLAGS=--dump_as_text
export ASCEND_RT_VISIBLE_DEVICES=3
export PYTHONPATH=/alignment/phase12/tools:/alignment/tools:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /state/jt-recipe-20261014/work-mf-ce226-kernel
exec > /alignment/phase12/runs/mf-ce226-kernel.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/mf-ce226-kernel.exit' EXIT
msrun --worker_num=1 --local_worker_num=1 --master_addr=127.0.0.1 --master_port=31627 --join=True --log_dir=/alignment/phase12/runs/mf-ce226-kernel-msrun python /alignment/phase12/tools/capture_mf_ce226.py --config /alignment/phase12/configs/mf-ce226-kernel.yaml --output /alignment/phase12/runs/mf-ce226-kernel --checkpoint-root /state/jt-recipe-20261014/mf-ce226-kernel --initial-weights /state/jt-recipe-20261014/mf-warmup/step-000200/model --resume-state /state/jt-recipe-20261014/mf-warmup/step-000200 --deep-trace --save-state-step -1
