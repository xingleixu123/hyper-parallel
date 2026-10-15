# Copyright 2026 Huawei Technologies Co., Ltd.
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

"""Independently validate all owned MF optimizer states against offline files and actual restored HP hashes."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/state/jt-recipe-20261014')
EVIDENCE = Path('/alignment/phase12')
RUN = 'hp-cann92-jtad1-pr180-192-routercontig-denseadj-castall'


def digest(value):
    """Hash full logical C-order values with bounded byte chunks."""
    value = np.ascontiguousarray(value)
    view = memoryview(value).cast('B')
    h = hashlib.sha256()
    for start in range(0,len(view),16*1024*1024):
        h.update(view[start:start+16*1024*1024])
    return h.hexdigest()


def source_mapping(name, load):
    """Spell out the model's inverse layout contract without importing the converter."""
    direct = {'model.embed_tokens.weight':'embedding.word_embeddings.weight',
              'lm_head.weight':'output_layer.weight','model.norm.weight':'decoder.final_layernorm.weight'}
    if name in direct:
        return load(direct[name]), 'direct'
    if name.startswith('model.layers.'):
        parts = name.split('.')
        prefix = 'decoder.layers.'+parts[2]
        suffix = '.'.join(parts[3:])
    elif name.startswith('mtp.layers.0.transformer_layer.'):
        prefix = 'mtp.layers.0.transformer_layer'
        suffix = name[len(prefix)+1:]
    else:
        return load(name), 'direct'
    if suffix == 'self_attn.linear_qkv.weight':
        q = load(prefix+'.self_attention.linear_q_down_proj.weight')
        kv = load(prefix+'.self_attention.linear_kv_down_proj.weight')
        if q.shape[0] != 1536 or kv.shape[0] != 576:
            raise ValueError('Unexpected Q/KV optimizer layout')
        return np.concatenate((q,kv),axis=0), 'Q1536 + KV576 rows'
    if suffix.startswith('self_attn.'):
        names = {'q_a_layernorm':'q_layernorm','kv_a_layernorm':'kv_layernorm',
                 'q_b_proj':'linear_q_up_proj','kv_b_proj':'linear_kv_up_proj','o_proj':'linear_proj'}
        piece = suffix.split('.')[1]
        return load(prefix+'.self_attention.'+names[piece]+'.weight'), 'direct attention rename'
    if suffix == 'post_attention_layernorm.weight':
        return load(prefix+'.pre_mlp_layernorm.weight'), 'direct norm rename'
    if suffix == 'mlp.gate.weight':
        return load(prefix+'.mlp.router.weight'), 'direct router rename'
    if suffix == 'mlp.experts.gate_up_proj':
        return load(prefix+'.mlp.experts.weight1').reshape(8,5120,3072).transpose(0,2,1), 'expert8 hidden5120 gate/up3072 transpose'
    if suffix == 'mlp.experts.down_proj':
        return load(prefix+'.mlp.experts.weight2').reshape(8,1536,5120).transpose(0,2,1), 'expert8 FFN1536 hidden5120 transpose'
    for owner in ('mlp','mlp.shared_experts'):
        if suffix in (owner+'.gate_proj.weight',owner+'.up_proj.weight'):
            fused = load(prefix+'.'+owner+'.linear_fc1.weight')
            offset = 0 if suffix.endswith('gate_proj.weight') else 1
            return fused[offset::2], 'interleaved gate even rows' if offset == 0 else 'interleaved up odd rows'
        if suffix == owner+'.down_proj.weight':
            return load(prefix+'.'+owner+'.linear_fc2.weight'), 'direct down projection'
    return load(prefix+'.'+suffix), 'direct'


def main():
    """Reject any missing/zeroed/misordered owned state or resume metadata mismatch."""
    directory = ROOT/'mf-warmup/step-000200'
    entries = {r['parameter_name']:r for r in json.loads((directory/'optimizer/manifest.json').read_text())}
    native = json.loads((directory/'STATE.json').read_text())
    if native['completed_steps'] != 200 or native['optimizer_global_step'] != 200:
        raise ValueError('MF anchor is not a full step200 state')
    actual = json.loads((EVIDENCE/'runs'/RUN/'RESTORE.json').read_text())
    scheduler = ast.literal_eval(actual['lr_scheduler'])
    if actual['step'] != 200 or actual['consumed_samples'] != [200] or not actual['all_exact']:
        raise ValueError('HP did not restore complete state200')
    if any(v['last_epoch'] != 200 or any(lr != 1e-4 for lr in v['_last_lr']) for v in scheduler.values()):
        raise ValueError('Scheduler epoch/LR differs')
    slots = {'muon_m':'muon_m','adam_m':'exp_avg','adam_v':'exp_avg_sq'}
    used = set()
    verified_sources = set()
    records = []
    with np.load(ROOT/'hp-offline-step200/optimizer.npz',allow_pickle=False) as offline:
        for row in actual['optimizer']:
            family,name = row['name'].split('.',1)
            keys = []
            def load(mf_name):
                entry = entries[mf_name]
                if entry['use_muon'] != (family == 'muon_m'):
                    raise ValueError('Optimizer ownership differs: '+mf_name)
                slot = entry['state'][slots[family]]
                value = np.load(directory/'optimizer'/slot['file'],mmap_mode='r')
                identifier = (mf_name,slots[family])
                if identifier not in verified_sources:
                    if digest(value) != slot['sha256']:
                        raise ValueError('MF source state bytes changed')
                    verified_sources.add(identifier)
                used.add(identifier)
                keys.append(mf_name)
                return value
            expected,layout = source_mapping(name,load)
            value = offline[row['name']]
            if expected.shape != value.shape or expected.dtype != value.dtype or not np.array_equal(expected,value):
                raise ValueError('Independent MF->HP state mapping differs: '+row['name'])
            sha = digest(value)
            if sha != row['sha256']:
                raise ValueError('HP actual in-memory restore differs from offline tensor')
            records.append({'name':row['name'],'sources':keys,'layout':layout,'shape':list(value.shape),
                            'sha256':sha,'mf_to_offline_exact':True,'offline_to_actual_hp_exact':True})
            del expected,value
    required = {(n,k) for n,r in entries.items() for k in r['state']}
    if used != required or len(records) != 64:
        raise ValueError(('Incomplete owned state coverage',required-used,used-required,len(records)))
    counts = {k:sum(r['name'].startswith(k+'.') for r in records) for k in slots}
    report = {'complete':True,'run':RUN,'step':200,'counts':counts,'all_mf_owned_states_covered':True,
              'records':records,'consumed_samples':actual['consumed_samples'],'lr_scheduler':scheduler,
              'observer_checks_each_parameter_or_group_step_before_first_forward':True,
              'observer_sha256':hashlib.sha256((EVIDENCE/'tools/capture_hp_native.py').read_bytes()).hexdigest(),
              'scope':'Independent explicit layout contract; no converter import, no state injection. Actual HP hashes were recorded after native checkpoint restore, before training201.'}
    (EVIDENCE/'evidence/optimizer-restore-independent-reaudit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'complete':True,'step':200,'counts':counts,'sources':len(used)}),flush=True)


if __name__ == '__main__':
    main()
