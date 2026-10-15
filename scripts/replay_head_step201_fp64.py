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

"""Replay selected first-step head gradients with full-token FP64 dot references."""
import json
from pathlib import Path
import numpy as np
from compare_head_weight_boundaries import rounded_bf16
from compare_head_selected_operands import stats
from compare_vocab_optimizer_trace import read_rows
ROOT=Path('/alignment/phase12')
HP='hp-jtf637-adamw-all-headcontig206-r4'


def main():
    hp=next(r for r in read_rows(ROOT/'runs'/HP/'VOCAB_OPTIMIZER_TRACE.jsonl') if r['step']==201 and r['name']=='lm_head.weight')
    mf=next(r for r in read_rows(ROOT/'runs/mf-vocab-optimizer-coords-r2/VOCAB_OPTIMIZER_TRACE.jsonl') if r['step']==201 and r['name']=='lm_head.weight')
    indices=np.asarray(hp['indices']);dw=[];ideal=[];details=[]
    calls=read_rows(ROOT/'runs'/HP/'HEAD_BACKWARD_COORDINATES.jsonl')
    for call in [1,2]:
        actual=np.asarray(calls[call-1]['dw_at_coordinates'],dtype=np.float32)
        with np.load(ROOT/'runs'/HP/f'head-operands-call{call}.npz') as operands:
            row=np.searchsorted(operands['vocabulary_rows'],indices//5120)
            col=np.searchsorted(operands['hidden_columns'],indices%5120)
            ref=np.sum(operands['dy'][:,row].astype(np.float64)*operands['x'][:,col].astype(np.float64),axis=0)
        rounded=rounded_bf16(ref.astype(np.float32));dw.append(actual);ideal.append(rounded)
        details.append({'call':call,'actual_vs_fp64_rounded':stats(actual,rounded),
                        'actual_error_max':float(np.max(np.abs(actual-ref)))})
    norm=np.float32(read_rows(ROOT/'runs'/HP/'metrics.jsonl')[0]['grad_norm'])
    coefficient=np.float32(1)/np.float32(norm+np.float32(1e-6))
    actual_clip=(dw[0]+dw[1])*coefficient
    ideal_clip=(ideal[0]+ideal[1])*coefficient
    hp_g=np.asarray(hp['gradient'],dtype=np.float32);mf_g=np.asarray(mf['gradient'],dtype=np.float32)
    report={'step':201,'count':len(indices),'branches':details,'norm':float(norm),'coefficient':float(coefficient),
            'reconstructed_hp_vs_actual_hp':stats(actual_clip,hp_g),'reconstructed_hp_vs_actual_mf':stats(actual_clip,mf_g),
            'ideal_bf16_branches_vs_mf':stats(ideal_clip,mf_g),'ideal_bf16_branches_vs_hp':stats(ideal_clip,hp_g),
            'scope':'Selected coordinates; framework comparison uses audited identical201actualx/dy. CPUFP32clip is independently checked against actualHPgradient.'}
    (ROOT/'evidence/head-step201-fp64-reference-replay.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':main()
