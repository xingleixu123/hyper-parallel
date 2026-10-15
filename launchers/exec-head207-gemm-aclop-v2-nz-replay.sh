set -euo pipefail
source /workspace/env.sh
export PYTHONPATH=/alignment/phase12/tools-f637:/workspace:${PYTHONPATH:-}
exec /state/jt-recipe-20261014/jt-f637-pr192-venv/bin/python /alignment/phase12/tools-f637/replay_head207_aclop_v2_nz.py
