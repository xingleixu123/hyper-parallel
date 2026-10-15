#!/usr/bin/env bash
set -eo pipefail
source /workspace/env.sh
case_name=$1
port=$2
card_count=$3
test "$card_count" -eq 4
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1
export HYPER_PARALLEL_TEST_DEVICE_TYPE=npu HCCL_CONNECT_TIMEOUT=180 HCCL_EXEC_TIMEOUT=300
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PATH="/work/venvs/issue211-tests/bin:$PATH"
test -x /work/venvs/issue211-tests/bin/python
output="/work/runs/$case_name"
test ! -e "$output"
mkdir -p "$output"
cp /work/tools/run_fair_pr214_st.sh "$output/runner.sh"
cd /workspace/hyper-parallel
git rev-parse HEAD > "$output/source-commit.txt"
git status --porcelain=v1 > "$output/source-status.txt"
git diff --binary HEAD > "$output/source.patch"
python -m pip install --no-deps --no-build-isolation --use-pep517 -e /workspace/hyper-parallel > "$output/editable-install.log" 2>&1
cd "$output"
python - <<'PY' > "$output/import-path.txt"
from pathlib import Path
import hyper_parallel

source = Path(hyper_parallel.__file__).resolve()
print(source)
if Path('/workspace/hyper-parallel/hyper_parallel') not in source.parents:
    raise RuntimeError(f'Unexpected editable source: {source}')
PY
# The tests package is imported from the checkout, outside the installed library.
cd /workspace/hyper-parallel
set +e
# Launch workers directly so nested helpers cannot replace the guard's physical device selection.
timeout --signal=TERM --kill-after=15s 400s \
  python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port="$port" \
  --log-dir="$output/two-rank-logs" --redirects=3 --tee=0 \
  -m pytest -p no:cacheprovider -s -q /workspace/hyper-parallel/tests/torch/shard/redistribute.py::test_uneven_shard_reconstruction_rejected \
  > "$output/two-rank-launcher.log" 2>&1
result=$?
printf '%s\n' "$result" > "$output/TWO_RANK_EXIT_CODE"
if [ "$result" -eq 0 ]; then
  timeout --signal=TERM --kill-after=15s 400s \
    python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port="$((port + 1))" \
    --log-dir="$output/four-rank-logs" --redirects=3 --tee=0 \
    -m pytest -p no:cacheprovider -s -q /workspace/hyper-parallel/tests/torch/shard/redistribute.py::test_uneven_partial_reduction_rejected \
    > "$output/four-rank-launcher.log" 2>&1
  result=$?
  printf '%s\n' "$result" > "$output/FOUR_RANK_EXIT_CODE"
fi
printf '%s\n' "$result" > "$output/EXIT_CODE"
exit "$result"
