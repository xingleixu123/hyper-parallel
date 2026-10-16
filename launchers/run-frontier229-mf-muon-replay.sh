set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=1
export PYTHONPATH=/alignment/phase12/tools:/workspace:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CLOSE_MATMUL_K_SHIFT=0 P12_NS_EAGER_SCOPE=all
cd /state/jt-recipe-20261014/work-frontier229-mf-muon-replay
exec > /alignment/phase12/runs/frontier229-mf-muon-replay.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/frontier229-mf-muon-replay.exit' EXIT
python /alignment/phase12/tools/replay_mf_frontier229_muon.py
