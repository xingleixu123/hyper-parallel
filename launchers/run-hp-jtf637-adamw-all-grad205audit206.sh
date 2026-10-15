set -euo pipefail
source /workspace/env.sh
export ASCEND_RT_VISIBLE_DEVICES=1
export PYTHONPATH=/alignment/phase12/tools-f637:/workspace:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1
export CLOSE_MATMUL_K_SHIFT=0
export P12_NS_EAGER_SCOPE=all
export P12_ADAMW_ARCHIVE_SCOPE=all
export P12_OPTIMIZER_FACTOR=dense-expert-eager-view
export P12_FACTOR_CONTROL=hp-cann92-jtad1-pr180-192-nativeclip-expert-eager-view-castall
export P12_MAPPED_REFERENCE=1
export P12_VOCABULARY_OBSERVE=0
export P12_ROUTER_MODE=joint-normalization
export P12_CE_TREE_SILU_GATE=1
export P12_SILU_GATE_SCOPE=all
cd /state/jt-recipe-20261014/work-hp-jtf637-adamw-all-grad205audit206
exec > /alignment/phase12/runs/hp-jtf637-adamw-all-grad205audit206.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/hp-jtf637-adamw-all-grad205audit206.exit' EXIT
/state/jt-recipe-20261014/jt-f637-pr192-venv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=31501 /alignment/phase12/tools-f637/capture_hp_gradient205_audit.py --config /alignment/phase12/configs/hp-jtf637-adamw-all-grad205audit206.yaml --output /alignment/phase12/runs/hp-jtf637-adamw-all-grad205audit206 --offline /state/jt-recipe-20261014/hp-offline-step200
