#!/usr/bin/env bash
set -euo pipefail
source /workspace/env.sh
export PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export OMP_NUM_THREADS=1 HYPER_PARALLEL_TEST_DEVICE_TYPE=cpu GLOO_SOCKET_IFNAME=lo
test_root=/work/tmp/pr226-gloo-source
test_venv=/work/venvs/pr226-gloo-tests
output=/work/evidence/pr226-gloo-8cpu
test ! -e "$test_venv"
test ! -e "$output"
mkdir "$output"
python -m venv --system-site-packages "$test_venv"
printf '%s\n' /workspace/venv/lib/python3.12/site-packages > "$test_venv/lib/python3.12/site-packages/workspace-base.pth"
export PATH="$test_venv/bin:$PATH"
cd /work
for mode in base head; do
    python -m pip install --no-deps --no-build-isolation --use-pep517 -e "$test_root/$mode" > "$output/$mode-editable.log" 2>&1
    timeout --signal=TERM --kill-after=10s 100s python -m torch.distributed.run \
        --nnodes=1 --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=30426 \
        /work/tools/check_resource_metrics_gloo.py "$mode" "$test_root/$mode" "$output" \
        > "$output/$mode-launcher.log" 2>&1
    printf '%s\n' "$?" > "$output/$mode-exit.txt"
done
