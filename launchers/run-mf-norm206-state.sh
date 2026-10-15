set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH=/alignment/phase12/tools:/alignment/tools:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1
cd /state/jt-recipe-20261014/work-mf-norm206-state
exec > /alignment/phase12/runs/mf-norm206-state.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/mf-norm206-state.exit' EXIT
msrun --worker_num=1 --local_worker_num=1 --master_addr=127.0.0.1 --master_port=31516 --join=True --log_dir=/alignment/phase12/runs/mf-norm206-state-msrun python /alignment/phase12/tools/capture_mf_norm206_state.py --config /alignment/phase12/configs/mf-norm206-state.yaml --output /alignment/phase12/runs/mf-norm206-state --checkpoint-root /state/jt-recipe-20261014/mf-norm206-state --initial-weights /state/jt-recipe-20261014/mf-warmup/step-000200/model --resume-state /state/jt-recipe-20261014/mf-warmup/step-000200 --gradient-output /state/jt-recipe-20261014/mf-norm206-state/gradients --gradient-max-elements 5120 --save-state-step -1
