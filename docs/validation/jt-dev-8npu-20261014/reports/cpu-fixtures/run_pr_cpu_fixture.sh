#!/usr/bin/env bash
set -euo pipefail
source /workspace/env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 HYPER_PARALLEL_TEST_DEVICE_TYPE=cpu OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
number=$1
evidence_prefix=${2:-pr$number-cpu-fixture}
[[ "$evidence_prefix" =~ ^[a-z0-9-]+$ ]]
case "$number" in
  195) tests=(tests/ut/auto_models/distributed/test_activation_checkpoint.py tests/ut/models/jt_deepseek_v3/test_modeling_jt_deepseek_v3.py::TestCompleteModel::test_public_recompute_covers_mtp_and_preserves_aux_gradients) ;;
  196) tests=(tests/ut/models/jt_deepseek_v3/test_modeling_jt_deepseek_v3.py::TestCompleteModel::test_auxiliary_loss_pop_preserves_router_and_input_gradients tests/ut/models/jt_deepseek_v3/test_modeling_jt_deepseek_v3.py::TestCompleteModel::test_checkpoint_auxiliary_loss_consumption_matches_eager_gradients) ;;
  200) tests=(tests/ut/components/functional/test_npu_grouped_swiglu.py tests/ut/core/fully_shard/test_fused_grad.py) ;;
  *) exit 2 ;;
esac
test_root="/work/fixes/pr$number-cpu-fixture"
test_venv="/work/venvs/pr$number-cpu-fixture"
test ! -e "/work/evidence/$evidence_prefix-ut.log"
if [[ ! -e "$test_venv/bin/python" ]]; then
  python -m venv --system-site-packages "$test_venv"
  printf '%s\n' /workspace/venv/lib/python3.12/site-packages > "$test_venv/lib/python3.12/site-packages/workspace-base.pth"
fi
export PATH="$test_venv/bin:$PATH"
python -m pip install --no-index --find-links=/work/tmp/issue211-test-wheels pytest==9.1.1
python -m pip install --no-deps --no-build-isolation --use-pep517 -e "$test_root"
cd /work
python -c 'import sys,hyper_parallel; from pathlib import Path; p=Path(hyper_parallel.__file__).resolve(); print("EDITABLE_IMPORT", p); assert p.is_relative_to(Path(sys.argv[1]))' "$test_root"
cd "$test_root"
set +e
timeout --signal=TERM --kill-after=10s 150s python -m pytest -q "${tests[@]}" > "/work/evidence/$evidence_prefix-ut.log" 2>&1
status=$?
printf '%s\n' "$status" > "/work/evidence/$evidence_prefix-ut.exit"
tail -n 10 "/work/evidence/$evidence_prefix-ut.log"
exit "$status"
