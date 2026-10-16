set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=5
export PYTHONPATH=/alignment/phase12/tools:/alignment/tools:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1
cd /state/jt-recipe-20261014/work-mf-native-stat259
exec > /alignment/phase12/runs/mf-native-stat259.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/mf-native-stat259.exit' EXIT
export MINDSPORE_DUMP_CONFIG=/alignment/phase12/configs/mf-native-stat259.json
msrun --worker_num=1 --local_worker_num=1 --master_addr=127.0.0.1 --master_port=31658 --join=True --log_dir=/alignment/phase12/runs/mf-native-stat259-msrun python /alignment/phase12/tools/capture_mf_model_hash259.py --config /alignment/phase12/configs/mf-native-stat259.yaml --output /alignment/phase12/runs/mf-native-stat259 --checkpoint-root /state/jt-recipe-20261014/mf-native-stat259 --initial-weights /state/jt-recipe-20261014/mf-warmup/step-000200/model --resume-state /state/jt-recipe-20261014/mf-warmup/step-000200 --save-state-step -1
