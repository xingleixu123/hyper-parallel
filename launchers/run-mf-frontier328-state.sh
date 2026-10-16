set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=5
export PYTHONPATH=/alignment/phase12/tools:/alignment/tools:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /state/jt-recipe-20261014/work-mf-frontier328-state
exec > /alignment/phase12/runs/mf-frontier328-state.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/mf-frontier328-state.exit' EXIT
msrun --worker_num=1 --local_worker_num=1 --master_addr=127.0.0.1 --master_port=31694 --join=True --log_dir=/alignment/phase12/runs/mf-frontier328-state-msrun python /alignment/phase12/tools/capture_mf_frontier328_state.py --config /alignment/phase12/configs/mf-frontier328-state.yaml --output /alignment/phase12/runs/mf-frontier328-state --checkpoint-root /state/jt-recipe-20261014/mf-frontier328-state --initial-weights /state/jt-recipe-20261014/mf-warmup/step-000200/model --resume-state /state/jt-recipe-20261014/mf-warmup/step-000200 --gradient-output /state/jt-recipe-20261014/mf-frontier328-state/gradients --save-state-step -1
