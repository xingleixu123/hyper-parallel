"""Check the installed TorchNPU precision option without allocating a device."""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
import torch_npu

import hyper_parallel
from hyper_parallel.trainer.runtime import random as controls


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path)
parser.add_argument("--expect-disabled", action="store_true")
args = parser.parse_args()
source = Path(controls.__file__).resolve()
if torch.npu.is_initialized():
    raise RuntimeError("The device-free probe must not initialize the NPU runtime")
initial = torch.npu.matmul.allow_hf32
result = {
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "editable_import": str(Path(hyper_parallel.__file__).resolve()),
    "source": str(source),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "initial_allow_hf32": initial,
    "initial_npu_initialized": torch.npu.is_initialized(),
    "note": "Force only the hardware-availability branch; use the actual installed backend option.",
}
try:
    torch.npu.matmul.allow_hf32 = True
    result["before_helper"] = torch.npu.matmul.allow_hf32
    with patch.object(controls, "IS_NPU_AVAILABLE", True), patch.object(controls, "IS_CUDA_AVAILABLE", False):
        controls.enable_high_precision_for_bf16()
    result["after_helper"] = torch.npu.matmul.allow_hf32
    result["cuda_style_tf32_readback"] = torch.npu.matmul.allow_tf32
    result["cuda_style_bf16_reduction_readback"] = torch.npu.matmul.allow_bf16_reduced_precision_reduction
    torch.npu.matmul.allow_hf32 = False
    result["after_supported_api"] = torch.npu.matmul.allow_hf32
finally:
    torch.npu.matmul.allow_hf32 = initial
    result["restored_allow_hf32"] = torch.npu.matmul.allow_hf32
    result["final_npu_initialized"] = torch.npu.is_initialized()
print(json.dumps(result, ensure_ascii=False, indent=2))
if args.output:
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
if result["final_npu_initialized"] or result["after_supported_api"] is not False:
    raise RuntimeError("The device-free precision-option probe failed")
if args.expect_disabled and result["after_helper"] is not False:
    raise RuntimeError("The Trainer helper did not disable the actual HF32 option")
