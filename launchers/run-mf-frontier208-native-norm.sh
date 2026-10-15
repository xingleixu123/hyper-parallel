set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=4
export PYTHONPATH=/alignment/phase12/tools:/alignment/tools:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1
export MS_DEV_GRAPH_KERNEL_FLAGS=--dump_as_text
export MINDSPORE_DUMP_CONFIG=/alignment/phase12/configs/mf-frontier208-native-norm.json
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /state/jt-recipe-20261014/work-mf-frontier208-native-norm
exec > /alignment/phase12/runs/mf-frontier208-native-norm.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/mf-frontier208-native-norm.exit' EXIT
msrun --worker_num=1 --local_worker_num=1 --master_addr=127.0.0.1 --master_port=31558 --join=True --log_dir=/alignment/phase12/runs/mf-frontier208-native-norm-msrun python /alignment/phase12/tools/capture_mf_frontier208_native_norm.py --config /alignment/phase12/configs/mf-frontier208-native-norm.yaml --output /alignment/phase12/runs/mf-frontier208-native-norm --checkpoint-root /state/jt-recipe-20261014/mf-frontier208-native-norm --initial-weights /state/jt-recipe-20261014/mf-warmup/step-000200/model --resume-state /state/jt-recipe-20261014/mf-warmup/step-000200 --gradient-output /state/jt-recipe-20261014/mf-frontier208-native-norm/gradients --save-state-step -1
