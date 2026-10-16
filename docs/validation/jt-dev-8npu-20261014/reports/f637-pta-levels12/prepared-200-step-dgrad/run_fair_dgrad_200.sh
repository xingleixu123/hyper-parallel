#!/usr/bin/env bash
set -eo pipefail
source /workspace/env.sh
export PATH="/work/venvs/validation/bin:$PATH"
test "$#" = 3
case_name=$1
port=$2
card_count=$3
case "$case_name:$port" in
  r129-f637-dgrad-fsdp8-fp32-200-v1:30378|r129-f637-dgrad-cp2-fp32-200-v1:30379) ;;
  *) exit 2 ;;
esac
test "$card_count" = 8
export JT_MODULE_PROFILE=0 JT_EXPERT_PROBES=0 JT_NS_FP32=0 JT_WGRAD_FP32=0
export JT_RECOMPUTE_TARGETS=""
export PYTHONHASHSEED=42 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=180 HCCL_EXEC_TIMEOUT=600
export JT_CASE_OUTPUT="/work/runs/$case_name"
export HYPER_PARALLEL_MODEL_INTEGRATION_OUTPUT_DIR="$JT_CASE_OUTPUT/integration"
test ! -e "$JT_CASE_OUTPUT"
mkdir -p "$JT_CASE_OUTPUT"
for helper in capture_training.py capture_precision_training.py ns_batch_trace.py module_profile.py expert_probe.py \
  precision_experiments.py fp32_wgrad_experiment.py capture_dgrad_200_training.py linear_dgrad_long_observer.py \
  linear_dgrad_trace_experiment.py linear_dgrad_experiment.py; do
  cp "/work/tools/$helper" "$JT_CASE_OUTPUT/$helper"
done
cp "/work/configs/$case_name.yaml" "$JT_CASE_OUTPUT/config.yaml"
cd /workspace/hyper-parallel
git rev-parse HEAD > "$JT_CASE_OUTPUT/source-commit.txt"
git status --porcelain=v1 > "$JT_CASE_OUTPUT/source-status.txt"
git diff --binary HEAD > "$JT_CASE_OUTPUT/source.patch"
python -c 'import hyper_parallel; print(hyper_parallel.__file__); assert hyper_parallel.__file__.startswith("/workspace/hyper-parallel/")' > "$JT_CASE_OUTPUT/import-path.txt"
set +e
timeout --signal=TERM --kill-after=15s 840s \
  python -m torch.distributed.run --nnodes=1 --nproc_per_node="$card_count" --master_addr=127.0.0.1 --master_port="$port" \
  --log-dir="$JT_CASE_OUTPUT/worker-logs" --redirects=3 --tee=0 \
  "$JT_CASE_OUTPUT/capture_dgrad_200_training.py" "$JT_CASE_OUTPUT/config.yaml" \
  > "$JT_CASE_OUTPUT/launcher.log" 2>&1
result=$?
printf '%s\n' "$result" > "$JT_CASE_OUTPUT/EXIT_CODE"
exit "$result"
