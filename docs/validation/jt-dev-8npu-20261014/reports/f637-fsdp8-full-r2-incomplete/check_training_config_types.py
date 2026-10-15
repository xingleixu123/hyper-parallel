"""Validate saved experiment configurations with the actual image's parsers."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_npu
import yaml
from huggingface_hub.errors import StrictDataclassFieldValidationError

from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import JTDeepseekV3Config
from hyper_parallel.trainer.config.parser import parse_training_args


def differences(left, right, path=""):
    result = []
    if type(left) is not type(right):
        result.append({"path": path, "json_type": type(left).__name__, "yaml_type": type(right).__name__,
                       "json_value": left, "yaml_value": right})
    elif isinstance(left, dict):
        for key, value in left.items():
            result.extend(differences(value, right[key], path + "/" + key))
    elif isinstance(left, list):
        for index, value in enumerate(left):
            result.extend(differences(value, right[index], path + "/" + str(index)))
    elif left != right:
        result.append({"path": path, "json_value": left, "yaml_value": right})
    return result


parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
if torch.npu.is_initialized():
    raise RuntimeError("Configuration preflight must not initialize an NPU")
rows = []
for file_path in sorted((args.root / "configs").glob("r129-f637-fp32-*.yaml")):
    data = file_path.read_bytes()
    row = {"path": str(file_path.relative_to(args.root)), "sha256": hashlib.sha256(data).hexdigest()}
    raw = yaml.safe_load(data.decode())
    try:
        as_json = json.loads(data)
    except json.JSONDecodeError:
        row.update(format="yaml", json_yaml_comparison="not_applicable")
    else:
        row.update(format="json_in_yaml_file", json_yaml_differences=differences(as_json, raw))
    row["scalar_types"] = {}
    for key in ("model/config/rms_norm_eps", "optimizer/muon_config/ns_epsilon", "optimizer/adamw_config/adamw_eps"):
        value = raw
        for part in key.split("/"):
            value = value[part]
        row["scalar_types"][key] = {"type": type(value).__name__, "value": value}
    try:
        parse_training_args([str(file_path)])
        JTDeepseekV3Config(**raw["model"]["config"])
    except (TypeError, ValueError, StrictDataclassFieldValidationError) as error:
        row.update(status="FAIL", error_type=type(error).__name__, error=str(error))
    else:
        row["status"] = "PASS"
    rows.append(row)
result = {"yaml_version": yaml.__version__, "torch_version": torch.__version__,
          "torch_npu_version": torch_npu.__version__, "npu_initialized": torch.npu.is_initialized(),
          "checks": ["yaml.safe_load", "parse_training_args", "JTDeepseekV3Config construction"],
          "files": rows}
with args.output.open("x", encoding="utf-8") as handle:
    handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
failures = [row["path"] for row in rows if row["status"] != "PASS"]
expected = ["configs/r129-f637-fp32-fsdp8-full-200-r2.yaml"]
if failures != expected or result["npu_initialized"]:
    raise RuntimeError(f"Unexpected configuration preflight result: {failures}")
retry = [row for row in rows if row["path"].endswith("full-200-r3.yaml")]
if len(retry) != 1 or retry[0]["status"] != "PASS" or retry[0].get("json_yaml_differences"):
    raise RuntimeError("The next full-recompute retry did not pass preflight")
print(json.dumps({"checked": len(rows), "passed": len(rows) - len(failures),
                  "known_invalid_control": failures, "retry_r3": "PASS", "npu_initialized": False}))
