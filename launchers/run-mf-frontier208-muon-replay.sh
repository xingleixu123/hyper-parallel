set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=3
export PYTHONPATH=/alignment/phase12/tools:/alignment/tools:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /state/jt-recipe-20261014/work-mf-frontier208-muon-replay
exec > /alignment/phase12/runs/mf-frontier208-muon-replay.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/mf-frontier208-muon-replay.exit' EXIT
python /alignment/phase12/tools/replay_mf_frontier208_muon.py
