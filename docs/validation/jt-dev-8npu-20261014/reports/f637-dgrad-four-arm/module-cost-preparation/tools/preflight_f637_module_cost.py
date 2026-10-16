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

"""无设备 preflight：真实 parser、数值类型、源码导入与冻结依赖核对。"""

# 当前规范要求 Torch 顶层导入；冻结 lint 插件仍包含已退役 Platform 规则。
# pylint: disable=forbidden-backend-import

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch
import torch_npu
import yaml

from f637_module_cost_spec import CASES, IMAGE, PORTS, REFERENCE, SOURCE, expected_config, verify_bundle, write_new_json

import hyper_parallel
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import JTDeepseekV3Config
from hyper_parallel.trainer.config.parser import parse_training_args


def verify_source(repo: Path) -> dict:
    """拒绝错误 editable、不同提交和工作区修改，不初始化设备。"""
    repo = repo.resolve()
    imported = Path(hyper_parallel.__file__).resolve()
    if imported != repo / "hyper_parallel/__init__.py":
        raise ValueError(f"Editable import differs: {imported}")
    commit = subprocess.check_output(["/usr/bin/git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["/usr/bin/git", "-C", str(repo), "status", "--porcelain=v1"], text=True)
    patch = subprocess.check_output(["/usr/bin/git", "-C", str(repo), "diff", "--binary", "HEAD"])
    if commit != SOURCE or status or patch:
        raise ValueError("Source pin changed or checkout is not clean")
    return {"commit": commit, "patch_sha256": hashlib.sha256(patch).hexdigest(), "import_path": str(imported)}


def main() -> None:
    """实际解析四份 YAML，只允许计划中的精确差异；输出排他创建。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/work"))
    parser.add_argument("--repo", type=Path, default=Path("/workspace/hyper-parallel"))
    parser.add_argument("--case", choices=tuple(CASES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if root not in output.parents or output.exists():
        raise ValueError("Preflight output must be a new file under this task root")
    if torch.npu.is_initialized() or torch.cuda.is_initialized():
        raise RuntimeError("Preflight must start without initialized accelerators")
    bundle = verify_bundle(root)
    source = verify_source(args.repo)
    reference = json.loads((root / REFERENCE).read_bytes())
    rows = []
    for case in ([args.case] if args.case else CASES):
        path = root / "configs" / f"{case}.yaml"
        payload = path.read_bytes()
        raw = yaml.safe_load(payload)
        numeric_fields = {
            "rms_norm_eps": raw["model"]["config"]["rms_norm_eps"],
            "ns_epsilon": raw["optimizer"]["muon_config"]["ns_epsilon"],
            "adamw_eps": raw["optimizer"]["adamw_config"]["adamw_eps"],
        }
        if any(not isinstance(value, float) for value in numeric_fields.values()):
            raise ValueError(f"YAML numerical fields must parse as floats: {numeric_fields}")
        JTDeepseekV3Config(**raw["model"]["config"])
        actual = parse_training_args([str(path)]).to_dict()
        if actual != expected_config(reference, case):
            raise ValueError(f"Unexpected resolved recipe difference: {case}")
        rows.append({"case": case, "config_sha256": hashlib.sha256(payload).hexdigest(),
                     "port": PORTS[case], "status": "PASS", "mode": actual["activation_checkpoint"]["mode"],
                     "numeric_fields": numeric_fields})
    if torch.npu.is_initialized() or torch.cuda.is_initialized():
        raise RuntimeError("Preflight unexpectedly initialized an accelerator")
    write_new_json(output, {"status": "PASS", "cases": rows, "source": source, "bundle": bundle,
                            "expected_image": IMAGE, "accelerators_initialized": False,
                            "versions": {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
                                         "yaml": yaml.__version__}})


if __name__ == "__main__":
    main()
