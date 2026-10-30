#!/usr/bin/env python3
"""Recheck five scalar warnings with private local JSON; emit aggregates only.

No input tensor or sample is included with this code. Inputs are read locally;
individual parameter, gradient and optimizer-state values are never printed or
written. The FP32 calculation is a diagnostic trajectory, not a device oracle.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

COMPARISON_SHA='42d66cc870378959224362060532853b8e790b6164a94d5c60196790eccf4480'
DELTA_SHA='76469ba4198dafcbbaeeba6372bb178be0af323575f7115ba7d784636b7032ff'
NAMES=(
    'model.layers.0.attn_hc_pre.branch_alpha',
    'model.layers.0.ffn_hc_pre.branch_alpha',
    'model.layers.1.attn_hc_pre.branch_alpha',
    'model.layers.1.ffn_hc_pre.branch_alpha',
    'model.layers.2.attn_hc_pre.branch_alpha',
)


def require(value,message):
    if not value:raise ValueError(message)


def read_pinned(path,expected):
    raw=Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==expected,'input SHA mismatch')
    return json.loads(raw)


def f32(x):
    return struct.unpack('<f',struct.pack('<f',float(x)))[0]


def bits(x):
    return struct.pack('<f',float(x))


def float_sha(x):
    return hashlib.sha256(bits(x)).hexdigest()


def reconstruct_abs(initial_l2,initial_sha):
    require(isinstance(initial_l2,(int,float)) and math.isfinite(initial_l2) and initial_l2>=0,'invalid scalar L2')
    positive=f32(initial_l2)
    candidates=[math.copysign(positive,1.0),math.copysign(positive,-1.0)]
    matches=[x for x in candidates if float_sha(x)==initial_sha]
    require(len(matches)==1,'initial scalar sign not uniquely identified by SHA')
    return matches[0]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--comparison',type=Path,required=True)
    ap.add_argument('--native-delta',type=Path,required=True)
    for arm in ('sync','deferred'):
        for rank in (2,3):ap.add_argument(f'--{arm}-rank{rank}',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    comp=read_pinned(args.comparison,COMPARISON_SHA)
    delta=read_pinned(args.native_delta,DELTA_SHA)
    require(comp.get('status')=='compared' and delta.get('status')=='compared','input status')
    rows={(x['arm'],x['rank'],x['kind'],x['name']):x for x in delta['per_role_rank_arm']}
    require(len(rows)==len(delta['per_role_rank_arm'])==1152,'native row coverage')
    reports={}
    for arm in ('sync','deferred'):
        for rank in (2,3):
            path=getattr(args,f'{arm}_rank{rank}')
            reports[(arm,rank)]=read_pinned(path,comp['rank_reports'][arm][rank])
    reconstructed=slot_matches=slot_total=unchanged=model_matches=0
    bounds=[];subnormal_state_count=0
    for arm in ('sync','deferred'):
        for rank in (2,3):
            report=reports[(arm,rank)]
            for name in NAMES:
                group=next(g for g in report['groups'] if name in g['members'])
                opts=group['options'];require(group['optimizer_class'].endswith('.adamw.AdamW')
                    and opts['betas']==[0.95,0.95] and opts['step']==1501
                    and opts['lr']==1e-5 and opts['eps']==1e-8 and opts['weight_decay']==0
                    and opts['amsgrad'] is False and opts['maximize'] is False,'unexpected optimizer group')
                pre=report['preclip']['gradients'][name]
                g=report['preclip']['clip']['postclip_samples'][name]
                require(pre['present'] is True and pre['sample_positions']==[0]
                        and isinstance(g,list) and len(g)==1 and math.isfinite(g[0]),'gradient sample unavailable')
                gradient=f32(g[0]); values={}
                for kind,field in (('model',name),('slots',name+'.exp_avg'),('slots',name+'.exp_avg_sq')):
                    record=rows[(arm,rank,kind,field)]
                    require(record['elements']==1 and record['nonfinite_initial']==0 and record['nonfinite_post']==0,
                            'not a finite scalar')
                    initial=reconstruct_abs(record['initial_l2'],record['initial_sha256'])
                    postrec=report['post'][kind][field]
                    require(postrec['sample_positions']==[0] and len(postrec['sample_values'])==1,
                            'post scalar sample coverage')
                    post=f32(postrec['sample_values'][0])
                    require(float_sha(post)==record['post_sha256']==postrec['local_sha256'],'post SHA mismatch')
                    values[field]=(initial,post)
                    reconstructed+=1
                initial_x,post_x=values[name]
                initial_m,post_m=values[name+'.exp_avg']
                initial_v,post_v=values[name+'.exp_avg_sq']
                require(initial_v>=0 and post_v>=0 and initial_x!=0,'invalid AdamW inputs')
                if 0<initial_v<2**-126:subnormal_state_count+=1
                b=f32(opts['betas'][0]);a=f32(1.0-b)
                mdiag=f32(b*initial_m+a*gradient)
                vdiag=f32(f32(b*initial_v)+f32(a*f32(gradient*gradient)))
                slot_total+=2
                slot_matches+=int(bits(mdiag)==bits(post_m))+int(bits(vdiag)==bits(post_v))
                unchanged+=int(bits(initial_x)==bits(post_x))
                beta1,beta2=opts['betas'];step=opts['step']
                bias1=1.0-beta1**step;bias2=1.0-beta2**step
                ideal_delta=opts['lr']*(post_m/bias1)/(math.sqrt(post_v/bias2)+opts['eps'])
                predicted_x=f32(initial_x-ideal_delta)
                model_matches+=int(bits(predicted_x)==bits(post_x))
                _,exp=math.frexp(abs(initial_x));half_ulp=2.0**(exp-25)
                bounds.append((opts['lr']*abs(post_m)/(bias1*opts['eps']))/half_ulp)
    require(reconstructed==60 and slot_total==40 and len(bounds)==20,'scalar coverage')
    result={'schema':'tp-single-step-scalar-diagnostic-public-v1',
            'status':'diagnostic_only_no_acceptance',
            'instances':20,'sha_constrained_initial_scalar_values':reconstructed,
            'fp32_trajectory_slot_matches':slot_matches,'fp32_trajectory_slot_total':slot_total,
            'initial_equals_post_parameters':unchanged,'fp32_parameter_prediction_matches':model_matches,
            'conservative_bounds_below_half_ulp':sum(x<1 for x in bounds),
            'max_conservative_half_ulp_ratio':max(bounds),
            'has_subnormal_initial_state':subnormal_state_count>0,
            'comparison_sha256':COMPARISON_SHA,'native_delta_sha256':DELTA_SHA,
            'scope':'private scalar inputs checked in memory; this code emits no sample values',
            'limitations':['one possible CPU FP32 rounding trajectory, not an NPU execution trace',
                           'not an independent full-model oracle or long-run acceptance']}
    raw=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    with args.out.open('xb') as f:f.write(raw)
    print(json.dumps({'status':result['status'],'output_sha256':hashlib.sha256(raw).hexdigest()},sort_keys=True))


if __name__=='__main__':main()
