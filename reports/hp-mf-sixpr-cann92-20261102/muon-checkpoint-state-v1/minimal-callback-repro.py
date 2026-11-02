# Copyright 2026 The Hyper-Parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU lifecycle reproduction; replica state stands in for a completed HSDP export."""
import json
from types import SimpleNamespace
import torch
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.core.optimizer.sharding_category import ParamRecord, HSDPGroupAssignment
from hyper_parallel.components.checkpoint.config import CheckpointingConfig
from hyper_parallel.trainer.callbacks.checkpoint_callback import CheckpointerCallback
from hyper_parallel.trainer.state import TrainerState

model = torch.nn.Module()
model.register_parameter("owner", torch.nn.Parameter(torch.ones(2, 2)))
model.register_parameter("replica", torch.nn.Parameter(torch.ones(2, 2)))
optimizer = Muon(model.parameters())
records = [ParamRecord(0, model.owner), ParamRecord(1, model.replica)]
assignment = HSDPGroupAssignment(owned_records=records[:1], all_records=records,
    owner_by_index={0: (0,), 1: (1,)}, replicate_group_ranks=(0,), replicate_sizes=(2,))
optimizer._hsdp_assignment_batches = {0: {"batch_groups": [{"sub_batches": [assignment]}]}}
# Both ranks receive this complete state during ordinary HSDP checkpoint export/load.
# No process group is built in this CPU example; real NPU evidence is separate.
for parameter in model.parameters():
    optimizer.state[parameter]["momentum_buffer"] = torch.ones_like(parameter)
config = CheckpointingConfig(save_optimizer=True, save_train_state=False, is_async=False)
trainer = SimpleNamespace(config=SimpleNamespace(checkpoint=config, model_init_dtype=None),
    mesh=None, model=model, optimizer=optimizer, state=TrainerState(global_step=1),
    model_integration=None, lr_scheduler=None, train_dataloader=None, data_iterator=None, train_steps=1)
callback = CheckpointerCallback(trainer)
writer_observation = []
def save(_path, _payload, **_kwargs):
    writer_observation.append(model.replica in optimizer.state)
# Synchronous writer return is the successful persistence boundary in this unit reproduction.
callback.checkpointer = SimpleNamespace(save=save, maybe_wait_for_async_save=lambda: None)
callback._save_checkpoint(trainer.state)
print(json.dumps({"writer_saw_replica": writer_observation, "owner_preserved": model.owner in optimizer.state,
    "replica_after_sync_save": model.replica in optimizer.state, "state_entries": len(optimizer.state)}))
