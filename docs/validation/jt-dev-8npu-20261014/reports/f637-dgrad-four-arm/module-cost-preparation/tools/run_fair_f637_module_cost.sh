#!/usr/bin/env bash
# 模块成本专用入口。仅由后续审阅后的 fair guard 调度；本文件不创建容器。
set -eo pipefail
source /workspace/env.sh
test "$#" = 3
case_name=$1
port=$2
card_count=$3
case "$case_name" in
  r129-f637-module-off-40-v1) arm=off; expected_port=30371 ;;
  r129-f637-module-norm6-40-v1) arm=norm6; expected_port=30372 ;;
  r129-f637-module-dense1-40-v1) arm=dense1; expected_port=30373 ;;
  r129-f637-module-moe2-40-v1) arm=moe2; expected_port=30374 ;;
  *) exit 2 ;;
esac
test "$port" = "$expected_port"
test "$card_count" = 8
python_bin=/work/venvs/validation/bin/python
test -x "$python_bin"
export PATH="/work/venvs/validation/bin:$PATH"
export JT_MODULE_COST_CASE="$case_name" JT_MODULE_COST_ARM="$arm"
export JT_MODULE_PROFILE=0 JT_EXPERT_PROBES=0 JT_NS_FP32=0 JT_WGRAD_FP32=0 JT_HEAD_TRACE=0
export JT_RECOMPUTE_TARGETS=""
export PYTHONHASHSEED=42 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=180 HCCL_EXEC_TIMEOUT=600
export JT_CASE_OUTPUT="/work/runs/$case_name"
export HYPER_PARALLEL_MODEL_INTEGRATION_OUTPUT_DIR="$JT_CASE_OUTPUT/integration"
test ! -e "$JT_CASE_OUTPUT"
mkdir "$JT_CASE_OUTPUT"
record_exit() {
  result=$?
  printf '%s\n' "$result" > "$JT_CASE_OUTPUT/EXIT_CODE"
}
trap record_exit EXIT
cd /workspace/hyper-parallel
"$python_bin" /work/tools/preflight_f637_module_cost.py --case "$case_name" \
  --output "$JT_CASE_OUTPUT/module-cost-preflight.json" > "$JT_CASE_OUTPUT/preflight.log" 2>&1
for helper in capture_training.py capture_precision_training.py ns_batch_trace.py module_profile.py expert_probe.py \
  precision_experiments.py fp32_wgrad_experiment.py f637_module_cost_spec.py f637_module_cost_experiment.py \
  capture_f637_module_cost_training.py; do
  cp "/work/tools/$helper" "$JT_CASE_OUTPUT/$helper"
done
cp "/work/configs/$case_name.yaml" "$JT_CASE_OUTPUT/config.yaml"
cp /work/scheduling/f637-module-cost-v1.manifest.json "$JT_CASE_OUTPUT/module-cost-manifest.json"
/usr/bin/git rev-parse HEAD > "$JT_CASE_OUTPUT/source-commit.txt"
/usr/bin/git status --porcelain=v1 > "$JT_CASE_OUTPUT/source-status.txt"
/usr/bin/git diff --binary HEAD > "$JT_CASE_OUTPUT/source.patch"
"$python_bin" -c 'import hyper_parallel; print(hyper_parallel.__file__)' > "$JT_CASE_OUTPUT/import-path.txt"
set +e
timeout --signal=TERM --kill-after=15s 840s \
  "$python_bin" -m torch.distributed.run --nnodes=1 --nproc_per_node="$card_count" \
  --master_addr=127.0.0.1 --master_port="$port" \
  --log-dir="$JT_CASE_OUTPUT/worker-logs" --redirects=3 --tee=0 \
  "$JT_CASE_OUTPUT/capture_f637_module_cost_training.py" "$JT_CASE_OUTPUT/config.yaml" \
  > "$JT_CASE_OUTPUT/launcher.log" 2>&1
result=$?
exit "$result"
