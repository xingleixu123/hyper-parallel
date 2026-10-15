set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=4
export PYTHONPATH=/alignment/phase12/tools:/alignment/tools:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1
cd /state/jt-recipe-20261014/work-mf-head206-state-r2
exec > /alignment/phase12/runs/mf-head206-state-r2.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/mf-head206-state-r2.exit' EXIT
msrun --worker_num=1 --local_worker_num=1 --master_addr=127.0.0.1 --master_port=31528 --join=True --log_dir=/alignment/phase12/runs/mf-head206-state-r2-msrun python /alignment/phase12/tools/capture_mf_head206_state_v2.py --config /alignment/phase12/configs/mf-head206-state-r2.yaml --output /alignment/phase12/runs/mf-head206-state-r2 --checkpoint-root /state/jt-recipe-20261014/mf-head206-state-r2 --initial-weights /state/jt-recipe-20261014/mf-warmup/step-000200/model --resume-state /state/jt-recipe-20261014/mf-warmup/step-000200 --save-state-step -1
