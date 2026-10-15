set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH=/alignment/phase12/tools:/alignment/tools:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1
cd /state/jt-recipe-20261014/work-mf-finalnorm-boundaries206
exec > /alignment/phase12/runs/mf-finalnorm-boundaries206.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/mf-finalnorm-boundaries206.exit' EXIT
msrun --worker_num=1 --local_worker_num=1 --master_addr=127.0.0.1 --master_port=31496 --join=True --log_dir=/alignment/phase12/runs/mf-finalnorm-boundaries206-msrun python /alignment/phase12/tools/capture_mf_final_norm_boundaries.py --config /alignment/phase12/configs/mf-finalnorm-boundaries206.yaml --output /alignment/phase12/runs/mf-finalnorm-boundaries206 --checkpoint-root /state/jt-recipe-20261014/mf-finalnorm-boundaries206 --initial-weights /state/jt-recipe-20261014/mf-warmup/step-000200/model --resume-state /state/jt-recipe-20261014/mf-warmup/step-000200 --deep-trace --save-state-step 999999
