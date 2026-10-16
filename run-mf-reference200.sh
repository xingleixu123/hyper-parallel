set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=5
export PYTHONPATH=/alignment/phase12/tools:/alignment/tools:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1
cd /state/jt-clean-rerun-20261017/work-mf-reference200
exec > /alignment/phase13/runs/mf-reference200.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase13/runs/mf-reference200.exit' EXIT
msrun --worker_num=1 --local_worker_num=1 --master_addr=127.0.0.1 --master_port=31818 --join=True --log_dir=/alignment/phase13/runs/mf-reference200-msrun python /alignment/phase12/tools/capture_mf_muon.py --config /alignment/phase13/configs/mf-reference200.yaml --output /alignment/phase13/runs/mf-reference200 --checkpoint-root /state/jt-clean-rerun-20261017/mf-reference200 --initial-weights /state/jt-recipe-20261014/mf-warmup/step-000200/model --resume-state /state/jt-recipe-20261014/mf-warmup/step-000200
