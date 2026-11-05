"""CPU-only synthetic HF finalizer and HP DCP compatibility probe."""
import tempfile
import torch
from torch import nn
import torch.distributed as dist
import hyper_parallel.core.distributed_checkpoint as dcp
from hyper_parallel.components.checkpoint.dcp_checkpointer import _ModelStrictLoadPlanner
from hyper_parallel.components.checkpoint.load_groups import LoadReport, build_load_targets
from hyper_parallel.models._transformers.checkpoint_loader import _finalize_model_loading
from hyper_parallel.models.materialization import register_rebuildable_buffer

class Toy(nn.Module):
    def __init__(self, persistent):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([2.]))
        self.register_buffer("_token_counts", torch.tensor([13], dtype=torch.int64), persistent=persistent)
        if not persistent:
            register_rebuildable_buffer(self, "_token_counts", value=torch.zeros(1, dtype=torch.int64))

old = Toy(True)
keys = tuple(build_load_targets(old))
print("HF persistent targets:", keys)
try:
    _finalize_model_loading(old, LoadReport(("weight",), ("_token_counts",), ()), strict=True)
except RuntimeError as error:
    print("HF strict missing persistent:", error)

new = Toy(False)
print("HF nonpersistent targets:", tuple(build_load_targets(new)))
report = _finalize_model_loading(new, LoadReport(("weight",), (), ()), strict=True)
print("HF nonpersistent strict:", report.missing_keys, "count before rebuild:", new._token_counts.tolist())

with tempfile.TemporaryDirectory(prefix="xing4-cold-dcp-") as path:
    dist.init_process_group("gloo", init_method="file://" + path + "/r", rank=0, world_size=1)
    try:
        dcp.save({"model": old.state_dict()}, checkpoint_id=path + "/old")
        dcp.save({"model": new.state_dict()}, checkpoint_id=path + "/new")
        target = {"model": Toy(False).state_dict()}
        dcp.load(target, checkpoint_id=path + "/old", planner=_ModelStrictLoadPlanner())
        print("old DCP into nonpersistent skeleton keys:", list(target["model"]), "old nonzero count not requested")
        target = {"model": Toy(True).state_dict()}
        try:
            dcp.load(target, checkpoint_id=path + "/new", planner=_ModelStrictLoadPlanner())
        except Exception as error:
            print("new DCP into persistent strict skeleton:", type(error).__name__, str(error)[:300])
    finally:
        dist.destroy_process_group()
