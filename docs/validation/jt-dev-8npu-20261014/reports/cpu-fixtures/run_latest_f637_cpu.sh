#!/usr/bin/env bash
set -euo pipefail
source /workspace/env.sh
export PYTHONDONTWRITEBYTECODE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HYPER_PARALLEL_TEST_DEVICE_TYPE=cpu
export OMP_NUM_THREADS=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
test_root=/work/fixes/latest-jt-validation-f637
test_venv=/work/venvs/latest-f637-tests
evidence_prefix=${1:-latest-f637-cpu}
[[ "$evidence_prefix" =~ ^[a-z0-9-]+$ ]]
test ! -e "/work/evidence/$evidence_prefix-ut.log"
if [[ ! -e "$test_venv/bin/python" ]]; then
    python -m venv --system-site-packages "$test_venv"
    printf '%s\n' /workspace/venv/lib/python3.12/site-packages > "$test_venv/lib/python3.12/site-packages/workspace-base.pth"
fi
export PATH="$test_venv/bin:$PATH"
python -m pip install --no-index --find-links=/work/tmp/issue211-test-wheels pytest==9.1.1
python -m pip install --no-deps --no-build-isolation --use-pep517 -e "$test_root"
cd /work
python -c 'import hyper_parallel; from pathlib import Path; p=Path(hyper_parallel.__file__).resolve(); print("EDITABLE_IMPORT", p); assert p.is_relative_to(Path("/work/fixes/latest-jt-validation-f637"))'
cd "$test_root"
python -c 'import sys; import tests.torch.context_parallel.test_jt_mla_cp; assert "torch" not in sys.modules and "hyper_parallel" not in sys.modules; print("LIGHTWEIGHT_LAUNCHER_PASS")'
set +e
timeout --signal=TERM --kill-after=10s 180s python -m pytest -q tests/ut/data/test_get_batch.py tests/ut/trainer/test_text_trainer.py tests/ut/trainer/test_checkpoint_callback.py tests/ut/auto_models/distributed/test_activation_checkpoint.py tests/ut/models/jt_deepseek_v3/test_modeling_jt_deepseek_v3.py::TestCompleteModel::test_checkpoint_auxiliary_loss_consumption_matches_eager_gradients tests/ut/models/jt_deepseek_v3/test_modeling_jt_deepseek_v3.py::TestCompleteModel::test_public_recompute_covers_mtp_and_preserves_aux_gradients tests/ut/models/jt_deepseek_v3/test_context_parallel.py tests/ut/models/jt_deepseek_v3/test_jt_packed.py tests/ut/auto_models/test_projected_cross_entropy.py tests/ut/core/fully_shard/test_hsdp_scheduler.py tests/ut/core/fully_shard/test_fused_grad.py tests/ut/components/functional/test_npu_grouped_swiglu.py tests/ut/core/optimizer/test_muon_ns_precision.py tests/ut/trainer/test_muon_precision_checkpoint.py > "/work/evidence/$evidence_prefix-ut.log" 2>&1
ut_status=$?
printf '%s\n' "$ut_status" > "/work/evidence/$evidence_prefix-ut.exit"
tail -n 40 "/work/evidence/$evidence_prefix-ut.log"
exit "$ut_status"
