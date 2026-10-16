set -euo pipefail
source /workspace/env.sh
export P12_QUP258_SCOPE=both
export ASCEND_RT_VISIBLE_DEVICES=7
export PYTHONPATH=/alignment/phase12/tools-e213:/alignment/phase12/tools-f637:/workspace:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CLOSE_MATMUL_K_SHIFT=0
export P12_NS_EAGER_SCOPE=all
export P12_ADAMW_ARCHIVE_SCOPE=all
export P12_GRAD205_AUDIT=0
export P12_HEAD_DWEIGHT_ROUTE=legacy_nz
export P12_FRONTIER208_NORM_FACTOR=q_router
export P12_WEIGHTED_DVM_SCOPE=both
export P12_PACKED_SEMANTICS_FACTOR=both
export P12_NORM219_FACTOR=both
export P12_NORM219_MODEL_HASH=0
export P12_OPTIMIZER_FACTOR=dense-expert-eager-view
export P12_FACTOR_CONTROL=hp-cann92-jtad1-pr180-192-nativeclip-expert-eager-view-castall
export P12_MAPPED_REFERENCE=1
export P12_VOCABULARY_OBSERVE=0
export P12_ROUTER_MODE=joint-normalization
export P12_CE_TREE_SILU_GATE=1
export P12_SILU_GATE_SCOPE=all
cd /state/jt-recipe-20261014/work-hp-jte213-fc1norm-qupboth400
exec > /alignment/phase12/runs/hp-jte213-fc1norm-qupboth400.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > /alignment/phase12/runs/hp-jte213-fc1norm-qupboth400.exit' EXIT
/state/jt-recipe-20261014/jt-e213-pr192-venv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=31680 /alignment/phase12/tools-e213/capture_hp_fc1_norm293.py --config /alignment/phase12/configs/hp-jte213-fc1norm-qupboth400.yaml --output /alignment/phase12/runs/hp-jte213-fc1norm-qupboth400 --offline /state/jt-recipe-20261014/hp-offline-step200
