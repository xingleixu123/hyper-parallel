"""Four-rank Xing4 functional worker. Requires controller physical-device handshake."""
import argparse
import hashlib
import json
import math
import importlib.metadata
import os
from pathlib import Path
import time
import torch
import numpy as np
import torch.distributed as dist
import torch_npu  # registers NPU backend

ROOT = Path("/work/output")
RANK = int(os.environ["RANK"])
LOCAL = int(os.environ["LOCAL_RANK"])
OUT = ROOT / ("rank-" + str(RANK))

def write(name, data):
    with (OUT / name).open("x", encoding="utf8") as f:
        json.dump(data, f, indent=2, allow_nan=False, default=str)
        f.write("\n")

def tensor_record(t):
    local = t.to_local() if hasattr(t, "to_local") else t
    raw = local.detach().contiguous().reshape(-1).view(torch.uint8)
    h = hashlib.sha256()
    for offset in range(0, raw.numel(), 4 * 1024 * 1024):
        h.update(raw[offset:offset + 4 * 1024 * 1024].cpu().numpy().tobytes())
    mesh = getattr(t, "device_mesh", None)
    return {"shape": list(t.shape), "local_shape": list(local.shape), "dtype": str(local.dtype),
            "sha256": h.hexdigest(), "placements": str(getattr(t, "placements", None)),
            "mesh": None if mesh is None else {"shape": list(mesh.shape), "ranks": list(mesh.rank_list),
                    "names": list(mesh.mesh_dim_names or []), "coordinate": mesh.get_coordinate()}}

def normalize_state(value):
    if torch.is_tensor(value):
        return tensor_record(value)
    if isinstance(value, dict):
        return {str(k): normalize_state(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_state(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("unsupported extra-state type: " + str(type(value)))

def live_state(base):
    params = dict(base.model.named_parameters())
    names = {id(getattr(p, "main_param", p)): n for n, p in params.items()}
    model = {n: tensor_record(v) for n, v in base.model.state_dict().items()}
    slots, groups, aliases = {}, [], {}
    for name, p in params.items():
        main = getattr(p, "main_param", None)
        if main is None:
            raise RuntimeError("missing main parameter: " + name)
        aliases[name] = {"same_object": main is p,
                         "same_storage": (main.to_local() if hasattr(main, "to_local") else main).data_ptr() == (p.to_local() if hasattr(p, "to_local") else p).data_ptr(),
                         "main": tensor_record(main)}
    for family, leaf in base.optimizer.optimizers_dict.items():
        for param, values in leaf.state.items():
            if id(param) not in names:
                raise RuntimeError("unknown optimizer parameter")
            slots[names[id(param)]] = {k: tensor_record(v) if torch.is_tensor(v) else v for k,v in values.items()}
        for group in leaf.param_groups:
            groups.append({"family": family, "parameters": [names[id(p)] for p in group["params"]],
                           "options": {k:v for k,v in group.items() if k != "params"}})
    return {"model": model, "slots": slots, "groups": groups, "main_aliases": aliases}

def handshake():
    if int(os.environ["WORLD_SIZE"]) != 4 or LOCAL != RANK or not 0 <= RANK < 4:
        raise RuntimeError("unexpected rank configuration")
    for module in (torch, torch_npu):
        if not Path(module.__file__).resolve().is_relative_to(Path("/work/env")):
            raise RuntimeError("unexpected torch or torch_npu origin")
    assets = {
        "/work/pyext/omni_training_custom_ops/custom_ops_lib.cpython-312-aarch64-linux-gnu.so":
            "e222d9b73fae8fb9212b9f83c5fcccb6583897ec3709042a46c21838b26e8ee7",
        "/work/task/vendor-opp/vendors/omni_training_custom_transformer/op_api/lib/libcust_opapi.so":
            "546929376c5695fc2fc55752152ec4057548b79c79861a79e88cc65188ddc0a7",
    }
    for path, expected in assets.items():
        with open(path, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        if actual != expected:
            raise RuntimeError("mHC asset identity mismatch: " + path)
    if torch.npu.device_count() != 4:
        raise RuntimeError("expected four visible NPUs")
    torch.npu.set_device(LOCAL)
    if torch.ones(1, device="npu:" + str(LOCAL)).item() != 1:
        raise RuntimeError("device allocation failed")
    stat = Path("/proc/self/stat").read_text().split(") ", 1)[1].split()
    tag = os.environ["HP_FOURCARD_TAG"]
    ready = dict(rank=RANK, local_rank=LOCAL, world_size=4, pid=os.getpid(),
                 starttick=int(stat[19]), tag=tag, visible=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                 namespaces={k: os.readlink("/proc/self/ns/" + k) for k in ("mnt", "pid", "ipc", "net", "uts")},
                 cgroup=Path("/proc/self/cgroup").read_text())
    write("device-ready.json", ready)
    deadline = time.monotonic() + 180
    release = ROOT / "physical-mapping-release.json"
    while not release.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("device mapping release")
        time.sleep(0.2)
    r = json.loads(release.read_text())
    if r["tag"] != tag or r["run_id"] != os.environ["HP_FOURCARD_RUN_ID"]:
        raise RuntimeError("invalid physical mapping release")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["resume"], required=True)
    ap.add_argument("--stop", type=int, choices=[105], required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=False)
    handshake()
    from hyper_parallel.trainer.config.parser import load_training_config
    from hyper_parallel.trainer.text_trainer import TextTrainer
    from hyper_parallel.trainer.callbacks.base import Callback
    from hyper_parallel.trainer.callbacks.checkpoint_callback import CheckpointerCallback

    class PlannedStop(Exception):
        pass

    class Recorder(Callback):
        def __init__(self, base, checkpointer):
            super().__init__(base)
            self.rows, self.micro, self.checkpointer = [], [], checkpointer
        def on_train_begin(self, state, **kwargs):
            if state.global_step != 5:
                raise RuntimeError("native restore did not load step5")
            extra = torch.load(Path("/work/checkpoint/extra_state") / ("extra_state_rank_" + str(RANK) + ".pt"), map_location="cpu", weights_only=False)
            observed = self.checkpointer._collect_extra_state(state)
            for key in ("global_step", "epoch", "lr_scheduler", "rng_state"):
                if normalize_state(extra[key]) != normalize_state(observed[key]):
                    raise RuntimeError("restored extra state differs: " + key)
            # Native HSDP checkpoint path broadcasts owner Muon state to replica peers.
            # Capture both native owned state and the expanded logical view, then restore
            # exactly the native ownership state before any forward/update.
            from hyper_parallel.components.checkpoint.dcp_checkpointer import cleanup_synced_optimizer_state
            before = json.loads(json.dumps(live_state(self.trainer), default=str))
            for leaf in self.trainer.optimizer.chained_optimizers:
                broadcast = getattr(leaf, "_broadcast_state_fused_for_ckpt", None)
                if callable(broadcast):
                    broadcast()
            actual = json.loads(json.dumps(live_state(self.trainer), default=str))
            cleanup_synced_optimizer_state(self.trainer.optimizer)
            after = json.loads(json.dumps(live_state(self.trainer), default=str))
            if before != after:
                raise RuntimeError("native optimizer ownership changed during verification")
            write("owner-state-before.json", before)
            write("restored-state.json", actual)
            dist.barrier()
            deadline = time.monotonic() + 7200
            release = ROOT / "restore-release.json"
            while not release.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("independent full DCP verification release")
                time.sleep(0.5)
            result = json.loads(release.read_text())
            own = hashlib.sha256((OUT / "restored-state.json").read_bytes()).hexdigest()
            if result.get("status") != "full_state_verified" or result.get("run_id") != os.environ["HP_FOURCARD_RUN_ID"] or result.get("tag") != os.environ["HP_FOURCARD_TAG"] or result.get("rank_proof_sha256", {}).get(str(RANK)) != own:
                raise RuntimeError("unverified restore release")
            write("restore-verification.json", {"rank": RANK, "step": 5, "full_dcp_verified": True,
                  "scheduler_rng_match_checkpoint": True, "release_sha256": hashlib.sha256(release.read_bytes()).hexdigest()})
            dist.barrier()
        def on_step_begin(self, state, **kwargs):
            torch.npu.synchronize(); torch.npu.reset_peak_memory_stats(); self.timed_start = time.perf_counter()
            if state.global_step == 5:
                sampler = self.trainer.train_dataloader.batch_sampler.state_dict() if self.trainer.train_dataloader is not None else None
                if self.trainer.mesh.tp_rank == 0 and sampler != {"consumed_samples":20,"epoch":0,"global_batch_size":4,"dp_world_size":self.trainer.mesh.dp_size}:
                    raise RuntimeError("restored sampler mismatch: " + repr(sampler))
            if self.trainer.mesh.dp_size * self.trainer.num_micro_batches != 4 or self.trainer.mesh.cp_size != 2:
                raise RuntimeError("unexpected CP layout or batch")
            self.micro = []
            self.lr_before = [{"family": name, "lr": float(group["lr"]), "step": int(group.get("step", 0))}
                for name, leaf in self.trainer.optimizer.optimizers_dict.items() for group in leaf.param_groups]
        def on_micro_step_begin(self, state, micro_batch, **kwargs):
            pos = state.global_step * 4 + len(self.micro) * self.trainer.mesh.dp_size + self.trainer.mesh.dp_rank
            row = {"logical_position": pos}
            for key in ("input_ids", "labels", "shift_labels", "loss_mask", "position_ids"):
                v = micro_batch.get(key)
                if torch.is_tensor(v):
                    t = v.detach().contiguous().cpu()
                    row[key] = dict(shape=list(t.shape), dtype=str(t.dtype), sha256=hashlib.sha256(t.numpy().tobytes()).hexdigest())
            from ordered_factory import OrderedFourFieldDataset
            if not hasattr(self, "evidence_dataset"):
                self.evidence_dataset = OrderedFourFieldDataset(["/datasets/mmap_deepseekv3_datasets_text_document"])
            sample = self.evidence_dataset[pos]
            tokens, labels, mask, positions = (torch.from_numpy(np.asarray(x)).reshape(1, -1) for x in sample)
            cp_rank = self.trainer.mesh.cp_rank
            cp_size = self.trainer.mesh.cp_size
            if cp_size != 2 or tokens.shape[1] % cp_size:
                raise RuntimeError("invalid CP split")
            width = tokens.shape[1] // cp_size
            start = cp_rank * width
            tokens, labels, mask, positions = (v[:, start:start + width] for v in (tokens, labels, mask, positions))
            row.update(cp_rank=cp_rank, tp_rank=self.trainer.mesh.tp_rank,
                       dp_rank=self.trainer.mesh.dp_rank, global_length=width * cp_size,
                       cp_start=start, cp_end=start + width)
            actual_labels = micro_batch["shift_labels"].detach().cpu()
            actual_mask = micro_batch["loss_mask"].detach().cpu()
            checks = ((micro_batch["input_ids"].detach().cpu(), tokens),
                      (actual_mask, mask), (micro_batch["position_ids"].detach().cpu(), positions),
                      (torch.where(actual_mask.bool(), actual_labels, 0), labels))
            if not all(torch.equal(a.to(torch.int64), b.to(torch.int64)) for a,b in checks):
                raise RuntimeError("actual batch differs from ordered dataset position")
            row["dataset_fields_verified"] = True
            self.micro.append(row)
        def on_step_end(self, state, loss, loss_dict, grad_norm, **kwargs):
            torch.npu.synchronize(); elapsed = time.perf_counter() - self.timed_start
            peak_allocated = torch.npu.max_memory_allocated(); peak_reserved = torch.npu.max_memory_reserved()
            values = {k: float(v) for k, v in loss_dict.items()}
            if "foundation_loss/mtp" not in values or "foundation_loss" not in values:
                raise RuntimeError("named foundation and MTP losses required")
            if not all(math.isfinite(v) for v in [float(loss), float(grad_norm), *values.values()]):
                raise RuntimeError("non-finite training metric")
            self.rows.append(dict(step=state.global_step, loss=float(loss), losses=values,
                                  mtp_raw=values["foundation_loss/mtp"] / 0.1,
                                  grad_norm=float(grad_norm), lr_before=self.lr_before,
                                  microbatches=self.micro, sync_seconds=elapsed, peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved))
            if state.global_step == 5:
                before = live_state(self.trainer)
                self.checkpointer._save_checkpoint(state, force_sync=True)
                self.checkpointer.wait_for_pending_save()
                dist.barrier()
                ckpt = ROOT / "checkpoints/global_step_5"
                if not (ckpt / ".metadata").is_file() or not (ckpt / "extra_state" / ("extra_state_rank_" + str(RANK) + ".pt")).is_file():
                    raise RuntimeError("native checkpoint incomplete")
                after = live_state(self.trainer)
                if before != after:
                    raise RuntimeError("checkpoint save changed model/optimizer state")
                write("checkpoint-5-live-state.json", before)
                sampler = self.trainer.train_dataloader.batch_sampler.state_dict()
                if sampler != {"consumed_samples": 20, "epoch": 0, "global_batch_size": 4, "dp_world_size": 4}:
                    raise RuntimeError("unexpected checkpoint cursor: " + repr(sampler))
                write("checkpoint-5.json", {"rank": RANK, "step": 5, "sampler": sampler,
                      "note": "native synchronous save; independent global DCP readback remains required"})
                dist.barrier()
            if state.global_step == 105:
                write("steps.json", self.rows)
                raise PlannedStop()

    config = load_training_config(args.config)
    if (config.training.train_iters != 2500 or config.training.lr_scheduler_iters != 7500
        or config.training.global_batch_size != 4 or config.training.micro_batch_size != 1):
        raise ValueError("unexpected training schedule or batch")
    import hyper_parallel
    if not Path(hyper_parallel.__file__).resolve().is_relative_to(Path("/work/sources/hyper-parallel")):
        raise RuntimeError("unexpected hyper_parallel import origin")
    config.checkpoint.save_ckpt = True
    config.checkpoint.checkpoint_dir = "/work/output/checkpoints"
    config.checkpoint.save_steps = 2500
    config.checkpoint.is_async = False
    config.checkpoint.restore_from = "/work/checkpoint"
    trainer = TextTrainer(config)
    base = trainer.base
    if base.world_size != 4 or base.mesh.ep_size != 1 or base.mesh.cp_size != 2 or (base.mesh.dp_size, base.mesh.tp_size, base.num_micro_batches) not in ((2,1,2),(1,2,4)):
        raise ValueError("world size or effective batch mismatch")
    callbacks = [x for x in base._callbacks if isinstance(x, CheckpointerCallback)]
    if len(callbacks) != 1:
        raise RuntimeError("native checkpoint callback missing")
    recorder = Recorder(base, callbacks[0])
    base._callbacks.append(recorder)
    evidence = {
        "rank": RANK, "mesh": str(base.mesh), "microbatches": base.num_micro_batches,
        "mesh_sizes": {k: getattr(base.mesh, k, None) for k in ("dp_size","dp_rank","tp_size","tp_rank","cp_size","cp_rank","ep_size")},
        "runtime": {"torch": torch.__version__, "torch_npu": importlib.metadata.version("torch-npu"),
                    "torch_file": torch.__file__, "torch_npu_file": torch_npu.__file__,
                    "hyper_parallel_file": hyper_parallel.__file__,
                    "ascend_home": os.environ.get("ASCEND_HOME_PATH")},
        "modules": [{"name": n, "type": type(m).__module__ + "." + type(m).__qualname__}
                    for n,m in base.model.named_modules()],
        "parameters": [{"name": n, "shape": list(p.shape), "placements": str(getattr(p, "placements", None))}
                       for n,p in base.model.named_parameters()],
        "config": config.to_dict(),
    }
    write("inventory.json", evidence)
    try:
        trainer.train()
    except PlannedStop:
        if base.state.global_step != 105 or len(recorder.rows) != 100:
            raise RuntimeError("invalid planned stop")
    else:
        raise RuntimeError("training ended before planned stop")
    dist.barrier()
    write("complete.json", dict(rank=RANK, run_id=os.environ["HP_FOURCARD_RUN_ID"],
                                status="100_updates_complete", updates=100, horizon=7500))
    dist.barrier()

if __name__ == "__main__":
    main()


