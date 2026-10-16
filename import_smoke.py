"""Check both initial import orders against the candidate editable checkout."""
import json
import subprocess
import sys
from pathlib import Path

rows=[]
for order in ['validator-first','wrapper-first']:
    first='from hyper_parallel.tools.model_integration.integration_validator import collect_activation_checkpoint_wrappers'
    second='from hyper_parallel.core.activation_memory.wrapper import CheckpointWrapper'
    lines=[first,second] if order=='validator-first' else [second,first]
    lines += ['from hyper_parallel.core.activation_memory.api import checkpoint_wrapper',
              'from hyper_parallel.core.fully_shard import api as fsdp', 'from torch import nn',
              'import torch,json', 'from unittest import mock',
              'model=checkpoint_wrapper(nn.Linear(3,3))',
              'with mock.patch.dict(fsdp.origin_class_to_extend_class,clear=True):',
              '    fsdp._extend_module_with_hsdp_interface(model)',
              '    assert isinstance(model,CheckpointWrapper)',
              '    assert collect_activation_checkpoint_wrappers(model)["count"]==1',
              'assert not torch.distributed.is_initialized()', 'assert not torch.cuda.is_initialized()',
              'print(json.dumps({"order":'+repr(order)+',"dynamic_wrapper":type(model).__name__,"passed":True}))']
    result=subprocess.run([sys.executable,'-c',chr(10).join(lines)],check=True,capture_output=True,text=True,timeout=60)
    rows.append(json.loads(result.stdout.strip().splitlines()[-1]))
Path('/work/import-smoke.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows))
