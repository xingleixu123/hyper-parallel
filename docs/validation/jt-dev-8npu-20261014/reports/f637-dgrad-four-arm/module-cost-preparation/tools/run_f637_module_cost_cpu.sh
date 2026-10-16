#!/usr/bin/env bash
# 仅在另行批准的无设备容器内调用；不自行创建容器或改 editable。
set -eo pipefail
source /workspace/env.sh
test "$#" = 2
python_bin=/work/venvs/validation/bin/python
test -x "$python_bin"
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd /workspace/hyper-parallel
bash -n /work/tools/run_fair_f637_module_cost.sh
"$python_bin" /work/tools/preflight_f637_module_cost.py --output "$1"
"$python_bin" /work/tools/check_f637_module_cost_cpu.py --output "$2"
