set -euo pipefail
source /workspace/env.sh
export PYTHONPATH=/alignment/phase12/tools-f637:/workspace:${PYTHONPATH:-}
set +e
/state/jt-recipe-20261014/jt-f637-pr192-venv/bin/python /alignment/phase12/tools-f637/replay_head207_aclop_v2_nz_r3.py
code=$?
printf "%s\n" "$code" > /alignment/phase12/runs/head207-gemm-aclop-v2-nz-replay-r3.app.exit
exit "$code"
