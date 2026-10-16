# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""固定四组模块实验的配置与证据边界；本模块不导入训练框架。"""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE = "18602c4524dc664b81b2bad9a7b624455ce1494f"
IMAGE = "sha256:5f9e677f439d11ab4e7c888f54b142335a27eee7411cbd2ce894a4798a70146d"
REFERENCE = "runs/r129-f637-fp32-fsdp8-0-200-r4/config.resolved.json"
MANIFEST = "scheduling/f637-module-cost-v1.manifest.json"
TARGETS = {
    "off": (),
    "norm6": (
        "model.layers.0.input_layernorm", "model.layers.0.post_attention_layernorm",
        "model.layers.1.input_layernorm", "model.layers.1.post_attention_layernorm",
        "mtp.layers.0.transformer_layer.input_layernorm",
        "mtp.layers.0.transformer_layer.post_attention_layernorm",
    ),
    "dense1": ("model.layers.0.mlp",),
    "moe2": ("model.layers.1.mlp", "mtp.layers.0.transformer_layer.mlp"),
}
CASES = {f"r129-f637-module-{arm}-40-v1": arm for arm in TARGETS}
PORTS = {case: 30371 + index for index, case in enumerate(CASES)}
OBSERVED_TARGETS = tuple(name for targets in TARGETS.values() for name in targets)
DISABLED_OVERRIDES = (
    "JT_MODULE_PROFILE", "JT_EXPERT_PROBES", "JT_NS_FP32", "JT_WGRAD_FP32", "JT_HEAD_TRACE",
)


def expected_config(reference: dict[str, Any], case: str) -> dict[str, Any]:
    """生成唯一允许的 resolved 配方差异，不修改输入字典。"""
    if case not in CASES:
        raise ValueError(f"Unknown module-cost case: {case}")
    expected = copy.deepcopy(reference)
    expected["training"]["train_iters"] = 40
    expected["checkpoint"].update(save_ckpt=False, restore_from=None,
                                  checkpoint_dir=f"/work/checkpoints/{case}")
    expected["activation_checkpoint"]["mode"] = "off" if CASES[case] == "off" else "full"
    return expected


def write_new_json(path: Path, value: Any) -> None:
    """排他创建证据，拒绝覆盖先前记录。"""
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def verify_bundle(root: Path) -> dict[str, Any]:
    """校验冻结 manifest 中的所有文件，路径必须留在任务根内。"""
    root = root.resolve()
    payload = (root / MANIFEST).read_bytes()
    manifest = json.loads(payload)
    if manifest["source"] != SOURCE or manifest["image"] != IMAGE:
        raise ValueError("Module-cost source/image pin differs")
    for relative, digest in manifest["files"].items():
        path = (root / relative).resolve()
        if root not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Pinned dependency changed or escaped task root: {relative}")
    return {"manifest_sha256": hashlib.sha256(payload).hexdigest(), "files": manifest["files"]}
