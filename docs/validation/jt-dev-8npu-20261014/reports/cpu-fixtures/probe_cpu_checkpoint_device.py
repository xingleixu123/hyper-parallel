"""Diagnose checkpoint device inference without exposing any NPU device."""

import json

import torch
import torch_npu
from torch.utils.checkpoint import DefaultDeviceType, checkpoint as native_checkpoint

from hyper_parallel.core.activation_memory import checkpoint as hyper_checkpoint


def run_case(label, operation):
    value = torch.tensor([0.1, 0.4], requires_grad=True)
    result = {"implementation": label, "default_device": DefaultDeviceType.get_device_type()}
    try:
        operation(torch.sin, value, use_reentrant=False).sum().backward()
        torch.testing.assert_close(value.grad, value.detach().cos(), rtol=0, atol=0)
        result["passed"] = True
    except RuntimeError as error:
        result.update(passed=False, error=str(error).splitlines()[0])
    print(json.dumps(result), flush=True)
    return result


original = DefaultDeviceType.get_device_type()
results = []
try:
    for device in (original, "cpu"):
        DefaultDeviceType.set_device_type(device)
        for name, function in (("native", native_checkpoint), ("hyper", hyper_checkpoint)):
            results.append(run_case(name, function))
finally:
    DefaultDeviceType.set_device_type(original)
print(json.dumps({"original_device": original, "restored_device": DefaultDeviceType.get_device_type()}))
assert all(result["passed"] for result in results if result["default_device"] == "cpu")
