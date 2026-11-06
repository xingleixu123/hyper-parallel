"""Native CP2/TP2/GA4 AUX autograd diagnostic, one LP setting per process.

Real Xing4Router, resolved statistics binding, FSDP2Manager/source layout,
root backward scheduler, native CP mean/Trainer token weighting and final
loss_normalize. Task zero isolates the independent AUX edge. No optimizer,
checkpoint, manual gradient SUM/AVG, expert/GMM/attention or full-model claim.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
import torch
import torch.distributed as dist
from torch import nn
from safetensors.torch import load_file, save_file

SOURCE = Path('/work/source')
SOURCE_SHA = '3292a5b93c7b62d4f97c3edf42572aeaca91a475057bb3540ca6bcf1b9eae245'
ASSET_SHA = 'a035d1bbb9afe561dc4a62f93586ae20653aa9a2f12d05d8760d7d4758e35d54'
REFERENCE_SHA = 'ccda55fcad2ca45a64b5f07af98dfc5e8f232ff4ff25f2af945d25c52e14056f'
RANK = int(os.environ['RANK'])
LOCAL = int(os.environ['LOCAL_RANK'])
ROOT = Path('/work/output')
OUT = ROOT / ('rank-' + str(RANK))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, data):
    with (OUT / name).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False, default=str)
        stream.write('\n')


def handshake():
    if int(os.environ["WORLD_SIZE"]) != 4 or LOCAL != RANK or not 0 <= RANK < 4:
        raise RuntimeError("unexpected rank configuration")
    for module in (torch, torch_npu):
        if not Path(module.__file__).resolve().is_relative_to(Path("/work/base-env")):
            raise RuntimeError("unexpected torch or torch_npu origin")
    require(torch.__version__ == '2.12.0+cpu' and torch_npu.__version__ == '2.12.0.post3', 'runtime versions differ')
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


class RouterUnit(nn.Module):
    """Only real router compute; safe padding follows the Xing4MoE facade."""
    def __init__(self, gate):
        super().__init__()
        self.gate = gate

    def forward(self, hidden_states):
        batch, sequence, hidden = hidden_states.shape
        padded = torch.cat((hidden_states.new_zeros(batch, self.gate.padding, hidden), hidden_states), dim=1)
        _, weights, _ = self.gate(padded)
        real = weights.reshape(batch, sequence + self.gate.padding, self.gate.top_k)[:, self.gate.padding:]
        return real.sum(-1, keepdim=True).expand(batch, sequence, hidden)


class AuxModel(nn.Module):
    def __init__(self, unit, config):
        super().__init__()
        self.unit = unit
        self.config = config
        self.mtp = None

    def forward(self, hidden_states):
        return self.unit(hidden_states)


def error(actual, target):
    a, b = actual.double(), target.double()
    difference = a - b
    return {'max_abs': float(difference.abs().max()),
            'relative_l2': float(difference.norm() / b.norm().clamp_min(1e-300)),
            'least_squares_multiplier': float((a * b).sum() / (b * b).sum().clamp_min(1e-300))}


def main():
    global torch_npu
    import torch_npu
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss-parallel', required=True, choices=('false', 'true'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execute-reviewed', action='store_true')
    args = parser.parse_args()
    require(args.execute_reviewed and os.environ.get('HP_ROOT_REVIEWED') == 'PR1477_NATIVE_AUX_LP_REDUCTION', 'ROOT-reviewed native AUX diagnostic required')
    require(args.output == ROOT and int(os.environ['WORLD_SIZE']) == 4, 'fixed four-rank output contract differs')
    OUT.mkdir(parents=True, exist_ok=False)
    handshake()
    torch.set_num_threads(2)
    torch.npu.matmul.allow_hf32 = False
    require(sha('/work/scripts/source-manifest.json') == SOURCE_SHA, 'source manifest differs')
    require(sha('/work/scripts/aux-native-fixture.safetensors') == ASSET_SHA, 'fixture differs')
    require(sha('/work/scripts/aux-native-reference.json') == REFERENCE_SHA, 'reference differs')
    direct = json.loads(importlib.metadata.distribution('hyper_parallel').read_text('direct_url.json'))
    require(direct.get('url') == SOURCE.as_uri() and direct.get('dir_info', {}).get('editable') is True, 'editable does not bind actual source')
    require(not any(name == 'hyper_parallel' or name.startswith('hyper_parallel.') for name in sys.modules), 'HP imported before source binding')
    sys.path.insert(0, str(SOURCE))
    import hyper_parallel
    from hyper_parallel.models.xing_4 import modeling_xing4_0 as xing
    from hyper_parallel.models.xing_4.adapter import registration  # noqa: F401
    from hyper_parallel.models.xing_4.adapter.distributed.context_parallel import xing4_cp_mean_loss
    from hyper_parallel.distributed import ShardingPlanner
    from hyper_parallel.distributed.apply import apply_sharding_plan
    from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
    from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
    from hyper_parallel.distributed.mesh import MeshContext
    from hyper_parallel.models.build_options import FSDP2Config, FSDP2MixedPrecisionConfig
    from hyper_parallel.trainer.runtime.distributed import _populate_mesh_ranks
    from hyper_parallel.trainer.runtime.metrics import mean_global_loss
    from hyper_parallel.trainer.runtime.fsdp import configure_fsdp_gradient_sync
    from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token
    from hyper_parallel.trainer.base import BaseTrainer
    from hyper_parallel.components.functional import aux_loss as aux
    from hyper_parallel.data.parallel.dataloader_parallel import create_dataloader_parallel_context
    from hyper_parallel.data.parallel.batch_parallel import CPBatchSharder
    from hyper_parallel.core.distributed_checkpoint.utils import create_chunk_list_for_tensor
    from hyper_parallel import Replicate, Shard
    manifest = json.loads(Path('/work/scripts/source-manifest.json').read_text())
    imported = {}
    for name, module in tuple(sys.modules.items()):
        if (name == 'hyper_parallel' or name.startswith('hyper_parallel.')) and getattr(module, '__file__', None):
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(SOURCE), 'mixed HP source: ' + name)
            relative = path.relative_to(SOURCE).as_posix()
            require(relative in manifest['files'] and sha(path) == manifest['files'][relative]['sha256'], 'imported source differs: ' + relative)
            imported[relative] = sha(path)
    reference = json.loads(Path('/work/scripts/aux-native-reference.json').read_text())
    assets = load_file('/work/scripts/aux-native-fixture.safetensors', device='cpu')
    lp = args.loss_parallel == 'true'
    dist.init_process_group('hccl', timeout=datetime.timedelta(seconds=180))
    meshes = MeshContext(dp_size=1, dp_replicate_size=1, dp_shard_size=2,
                         tp_size=2, cp_size=2, ep_size=1, sequence_parallel=False, loss_parallel=lp)
    meshes.build_meshs('npu', 4)
    _populate_mesh_ranks(meshes, 2)
    device = torch.device('npu:' + str(LOCAL))
    config = xing.create_config(dict(vocab_size=131072, hidden_size=3584,
        intermediate_size=9216, moe_intermediate_size=1024, n_routed_experts=64,
        n_shared_experts=1, num_experts_per_tok=4, num_hidden_layers=1,
        first_k_dense_replace=0, num_nextn_predict_layers=1, n_group=0, topk_group=0,
        moe_router_group_score_topk=2, use_pad_tokens=True, moe_aux_loss_coeff=.0005,
        moe_input_jitter_eps=0., moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=.0001, moe_n_hash_layers=0,
        moe_router_force_expert_balance=False, compute_dtype='bfloat16'))
    source = xing.Xing4_0TopkRouter(config)
    with torch.no_grad():
        source.weight.copy_(assets['router_weight'])
        source.e_score_correction_bias.copy_(assets['router_bias'])
    gate = xing.Xing4Router(config, source, layer_idx=0)
    net = AuxModel(RouterUnit(gate), config).to(device)
    activation = {'tp': Replicate(), 'cp': Shard(1)}
    spec = ModuleShardingSpec(params={'gate.weight': {'tp': Replicate(), 'cp': Replicate()}},
        in_src={'hidden_states': activation.copy()}, in_dst={'hidden_states': activation.copy()},
        out_src={'output': activation.copy()}, out_dst={'output': activation.copy()},
        inner_wrapper=xing.xing4_router_statistics, inner_target='self', region_dispatch=False)
    plan = ShardingPlanner(plan_overrides={'unit': spec}).plan(net, meshes.device_mesh,
        tp_size=2, cp_size=2, ep_size=1, sequence_parallel=False, loss_parallel=lp)
    net, source_info = apply_sharding_plan(net, plan, meshes)
    require(source_info['unit.gate.weight'][0] == (Replicate(),), 'source TP Replicate was not retained')
    require(source_info['unit.gate.weight'][1].mesh_dim_names == ('tp',), 'source parameter mesh includes a non-TP axis')
    fsdp = FSDP2Config(dp_shard_size=2, wrapper_class=['RouterUnit'],
        backward_prefetch_depth=0, forward_prefetch_depth=0, comm_fusion=False,
        mix_precision=FSDP2MixedPrecisionConfig(param_dtype='bfloat16', reduce_dtype='float32',
            cast_forward_inputs=False, custom_params={'unit.gate.weight': {'param_dtype': 'float32', 'reduce_dtype': 'float32'}}))
    net = FSDP2Manager(fsdp, meshes, fp32_main_params=True).parallelize(net, source_info)
    mean_cp = xing4_cp_mean_loss(module=net, method_fqn='model.mean_loss',
        original_method=xing.Xing4Model.mean_loss, context={'mesh_context': meshes})
    sharder = CPBatchSharder(create_dataloader_parallel_context(meshes),
        token_fields=frozenset(('input_ids', 'shift_labels', 'loss_mask')))
    # Reuse the actual Trainer normalizer without constructing a full Trainer.
    # All five consumed fields are real model/mesh/count state; no arithmetic is reimplemented.
    normalizer = BaseTrainer.__new__(BaseTrainer)
    normalizer.model, normalizer.mesh, normalizer.device = net, meshes, device
    normalizer.loss_names, normalizer.step_token_counts = ('foundation_loss',), {}
    micros, total_loss, total_named = [], 0., {'foundation_loss': 0.}
    for micro in range(4):
        batch = sharder.shard({'input_ids': assets['input_ids'],
            'shift_labels': assets['shift_labels'][micro], 'loss_mask': assets['loss_mask'][micro]})
        ids = batch['input_ids'].reshape(-1)
        hidden = assets['hidden_states'][micro % 2].index_select(1, ids).to(device)
        batch = {key: value.to(device) for key, value in batch.items()}
        counts = count_loss_token(batch)
        count = counts['foundation_tokens']
        normalizer.step_token_counts['foundation_tokens'] = normalizer.step_token_counts.get('foundation_tokens', 0) + count.detach()
        require(int(count) == reference['foundation_counts_CP'][micro][meshes.cp_rank], 'actual CP mask/count differs')
        configure_fsdp_gradient_sync([net], fsdp, meshes.dp_replicate_size, micro, 4)
        observed = {}
        with aux.aux_loss_scale_context(loss_parallel=meshes.loss_parallel, tp_size=meshes.tp_size):
            output = net(hidden)
            zero_sum = output.float().sum() * 0.
            foundation = mean_cp(zero_sum, count)
            aux.bind_aux_loss_scale(foundation)
            state_object = aux._AUX_LOSS_STATE.get()
            def scalar_hook(gradient):
                observed['bound_foundation_multiplier'] = float(gradient.detach())
                observed['actual_aux_context_scale'] = float(state_object.scale)
            def sum_hook(gradient):
                observed['native_CP_loss_sum_multiplier'] = float(gradient.detach())
            handles = [foundation.register_hook(scalar_hook), zero_sum.register_hook(sum_hook)]
            named = mean_global_loss({'foundation_loss': foundation}, counts, meshes)
            task = torch.stack(list(named.values())).sum()
            task.backward()
            for handle in handles:
                handle.remove()
        torch.npu.synchronize()
        require(all(value == 0. for value in (float(foundation), float(task))), 'zero task isolation failed')
        total_loss += float(task)
        total_named['foundation_loss'] += float(named['foundation_loss'])
        observed.update(micro=micro, actual_local_S=hidden.shape[1], router_padded_S=hidden.shape[1]+gate.padding,
            local_foundation_count=int(count), raw_aux=float(gate.last_aux_loss),
            expected_global_count=reference['foundation_counts_global'][micro])
        micros.append(observed)
    params = list(net.unit.hsdp_scheduler.hsdp_state.hsdp_params)
    require(len(params) == 1, 'probe manages unexpected parameters')
    hp = params[0]
    param = dict(net.named_parameters())['unit.gate.weight']
    require(param is hp.sharded_param and getattr(param, 'main_grad', None) is not None, 'native FP32 main_grad not exposed on managed parameter')
    grad = param.main_grad
    before = grad.to_local().detach().cpu().clone()
    normalizer.loss_normalize(total_loss, total_named)
    torch.npu.synchronize()
    actual = param.main_grad.to_local().detach().cpu().clone()
    chunks = list(create_chunk_list_for_tensor(param))
    require(len(chunks) == 1 and list(chunks[0].sizes) == list(actual.shape), 'native gradient target chunk is not one matching rectangle')
    chunk = chunks[0]
    index = tuple(slice(offset, offset + size) for offset, size in zip(chunk.offsets, chunk.sizes))
    target = assets['independent_fp64_global_gradient'][index]
    require(torch.isfinite(actual).all() and target.norm() > 0, 'nonfinite or empty causal gradient')
    cp_members = dist.get_process_group_ranks(meshes.cp_mesh.get_group())
    tp_members = dist.get_process_group_ranks(meshes.device_mesh['tp'].get_group())
    groups = [dist.get_process_group_ranks(group) for group in gate.aux_loss_groups]
    require(groups == [cp_members] and cp_members != tp_members, 'actual AUX group is not CP only')
    one, two = error(actual, target), error(actual, 2 * target)
    tolerance = reference['predeclared_reference_relative_l2_tolerance']
    classification = 'matches_1x' if one['relative_l2'] < tolerance else ('matches_2x' if two['relative_l2'] < tolerance else 'neither_1x_nor_2x')
    save_file({'native_before_step_token_normalize': before, 'native_after_step_token_normalize': actual}, str(OUT / 'native-gradients.safetensors'))
    write('result.json', {'status': 'native_aux_probe_complete', 'rank': RANK,
        'run_id': os.environ['HP_FOURCARD_RUN_ID'], 'tag': os.environ['HP_FOURCARD_TAG'], 'loss_parallel': lp,
        'source_manifest_sha256': SOURCE_SHA, 'imported_source_sha256': imported, 'editable': direct,
        'fixture_sha256': ASSET_SHA, 'reference_sha256': REFERENCE_SHA, 'micros': micros,
        'actual_CP_group': cp_members, 'actual_TP_group': tp_members, 'actual_aux_groups': groups,
        'native_source_mesh': list(hp.source_shard_info.mesh.mesh_dim_names),
        'native_source_placements': [repr(value) for value in hp.source_shard_info.placements],
        'native_storage_source_mesh': list(hp._storage_source_layout[0].mesh_dim_names),
        'native_storage_source_placements': [repr(value) for value in hp._storage_source_layout[1]],
        'native_gradient_scaling_factor': hp.gradient_scaling_factor,
        'native_reduce_op': str(net.unit.hsdp_scheduler.hsdp_state.reduce_op_type),
        'native_chunk': {'offsets': list(chunk.offsets), 'sizes': list(chunk.sizes)},
        'actual_grad_mesh': list(param.main_grad.device_mesh.mesh_dim_names),
        'actual_grad_placements': repr(param.main_grad.placements),
        'step_foundation_denominator_reference': reference['step_foundation_denominator'],
        'native_normalize_before_after_max_abs': float((before / reference['step_foundation_denominator'] - actual).abs().max()),
        'vs_independent_1x': one, 'vs_independent_2x': two, 'classification': classification,
        'gradient_file_sha256': sha(OUT / 'native-gradients.safetensors'),
        'updates': 0, 'manual_gradient_collectives_or_compensation': False,
        'scope': 'AUX-only real autograd/native FSDP scheduling diagnostic, not MoE/CE/backbone/optimizer/full training acceptance'})
    dist.barrier()
    write('complete.json', {'rank': RANK, 'run_id': os.environ['HP_FOURCARD_RUN_ID'],
        'status': 'native_aux_probe_complete', 'loss_parallel': lp, 'result_sha256': sha(OUT / 'result.json')})


if __name__ == '__main__':
    try:
        main()
    except BaseException as error_value:
        if OUT.is_dir():
            write('failure.json', {'rank': RANK, 'error': str(error_value), 'traceback': traceback.format_exc(), 'updates': 0})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
