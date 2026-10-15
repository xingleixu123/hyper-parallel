"""Record import provenance without initializing an accelerator."""

from dataclasses import fields
import hashlib
import inspect
import json
from pathlib import Path
import sys

import torch
import torch_npu
import hyper_parallel
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import JTDeepseekV3Config
from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.config.trainer import TrainerConfig


sources = {}
for name, path in {
    "package": hyper_parallel.__file__,
    "trainer_config": inspect.getfile(TrainerConfig),
    "parser": inspect.getfile(parse_training_args),
    "model_config": inspect.getfile(JTDeepseekV3Config),
}.items():
    file_path = Path(path).resolve()
    sources[name] = {"path": str(file_path), "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest()}
result = {"python": sys.executable, "sys_path": sys.path, "cwd": str(Path.cwd()), "sources": sources,
          "trainer_fields": [field.name for field in fields(TrainerConfig)],
          "npu_initialized": torch.npu.is_initialized(), "torch_npu": torch_npu.__version__}
with Path(sys.argv[1]).open("x", encoding="utf-8") as handle:
    handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
