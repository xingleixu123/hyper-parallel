set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=1
export PYTHONPATH=/alignment/phase12/tools:/alignment/tools:${PYTHONPATH:-}
export MINDSPORE_DUMP_CONFIG=/alignment/phase12/configs/mf-native-router-norm206.json
export MS_DEV_GRAPH_KERNEL_FLAGS=--dump_as_text
export PYTHONDONTWRITEBYTECODE=1
mkdir -p /state/jt-recipe-20261014/work-mf-native-router-norm206
cd /state/jt-recipe-20261014/work-mf-native-router-norm206
exec > /alignment/phase12/runs/mf-native-router-norm206.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/mf-native-router-norm206.exit' EXIT
msrun --worker_num=1 --local_worker_num=1 --master_addr=127.0.0.1 --master_port=31525 --join=True --log_dir=/alignment/phase12/runs/mf-native-router-norm206-msrun python /alignment/phase12/tools/capture_mf_muon.py --config /alignment/phase12/configs/mf-native-router-norm206.yaml --output /alignment/phase12/runs/mf-native-router-norm206 --checkpoint-root /state/jt-recipe-20261014/mf-native-router-norm206 --initial-weights /state/jt-recipe-20261014/mf-warmup/step-000200/model --resume-state /state/jt-recipe-20261014/mf-warmup/step-000200 --save-state-step 999
