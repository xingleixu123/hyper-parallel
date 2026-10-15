set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH=/alignment/phase12/tools:/alignment/tools:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1
cd /state/jt-recipe-20261014/work-mf-head207-momenthash
exec > /alignment/phase12/runs/mf-head207-momenthash.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/mf-head207-momenthash.exit' EXIT
msrun --worker_num=1 --local_worker_num=1 --master_addr=127.0.0.1 --master_port=31539 --join=True --log_dir=/alignment/phase12/runs/mf-head207-momenthash-msrun python /alignment/phase12/tools/capture_mf_head_moments207_hash.py --config /alignment/phase12/configs/mf-head207-momenthash.yaml --output /alignment/phase12/runs/mf-head207-momenthash --checkpoint-root /state/jt-recipe-20261014/mf-head207-momenthash --initial-weights /state/jt-recipe-20261014/mf-warmup/step-000200/model --resume-state /state/jt-recipe-20261014/mf-warmup/step-000200 --save-state-step -1
