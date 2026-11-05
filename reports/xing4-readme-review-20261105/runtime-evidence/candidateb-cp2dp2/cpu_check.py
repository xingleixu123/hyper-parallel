import importlib.util,json,tempfile,sys
from pathlib import Path
sys.path.insert(0,"/home/xxl/.cache/codex/xing4-deep-reuse-20261104/after")
from hyper_parallel.data.parallel.batch_sampler import build_dataset_batch_sampler
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('controller',root/'crosslaunch/ep_controller.py');controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
source={'consumed_samples':20,'epoch':0,'global_batch_size':4,'dp_world_size':4}
for dp in (1,2):
 for r in range(dp):
  s=build_dataset_batch_sampler(total_samples=500,micro_batch_size=1,global_batch_size=4,dp_rank=r,dp_world_size=dp,consumed_samples=0,drop_last=True,sampler_type='single')
  s.load_state_dict(source); assert s.state_dict()=={**source,'dp_world_size':dp}
  it=iter(s);assert [next(it)[0] for _ in range(4//dp)]==[20+m*dp+r for m in range(4//dp)]
  assert s.state_dict()['consumed_samples']==24
 try:
  s.load_state_dict({**source,'consumed_samples':21})
  raise AssertionError('unaligned cursor accepted')
 except ValueError:pass
try:
 s.load_state_dict({**source,'global_batch_size':8});raise AssertionError('changed global batch accepted')
except ValueError:pass
for conf,(dp,cp,tp,ga) in {'cp2dp2.yaml':(2,2,1,2),'tp2cp2.yaml':(1,2,2,4)}.items():
 with tempfile.TemporaryDirectory() as tmp:
  path=Path(tmp);rank=0
  for d in range(dp):
   for c in range(cp):
    for t in range(tp):
     p=path/f'rank-{rank}';p.mkdir();m={'dp_size':dp,'cp_size':cp,'tp_size':tp,'ep_size':1,'dp_rank':d,'cp_rank':c,'tp_rank':t}
     (p/'inventory.json').write_text(json.dumps({'mesh_sizes':m,'microbatches':ga}));(p/'complete.json').write_text(json.dumps({'rank':rank,'run_id':'test','status':'100_updates_complete','updates':100,'horizon':7500}))
     rows=[]
     for step in range(6,106):
      mb=[]
      for k in range(ga):
       pos=(step-1)*4+k*dp+d;z={'dataset_fields_verified':True,'logical_position':pos,'dp_rank':d,'cp_rank':c,'tp_rank':t,'cp_start':c*2048,'cp_end':(c+1)*2048,'global_length':4096}
       z.update({name:{'sha256':f'{name}-{pos}-{c}'} for name in ('input_ids','shift_labels','loss_mask','position_ids')});mb.append(z)
      rows.append({'step':step,'microbatches':mb})
     (p/'steps.json').write_text(json.dumps(rows))
     for n in ('restore-verification.json','restored-state.json'):(p/n).write_text('{}')
     rank+=1
  assert controller.perf_completion(path,{'config':conf,'run_id':'test'})['global_sha256']=='all-ranks-complete'
  p=path/'rank-0/steps.json';rows=json.loads(p.read_text());rows[0]['microbatches'][0]['cp_start']=99;p.write_text(json.dumps(rows))
  try:controller.perf_completion(path,{'config':conf,'run_id':'test'});raise AssertionError('corrupt slice accepted')
  except RuntimeError:pass
for cards in ((0,1,2,3),(4,5,6,7)):
 controller.candidate.CARDS=cards
 proof=[{'rank':i,'host_pid':100+i} for i in range(4)]
 samples={card:f'Process id: {100+i} Process name: python Process memory(MB): 1' for i,card in enumerate(cards)}
 assert controller.candidate.require_bijection(proof,samples)['physical_by_rank']==list(cards)
 samples[cards[0]]='No process in device.'
 try:controller.candidate.require_bijection(proof,samples);raise AssertionError('bad physical mapping accepted')
 except ValueError:pass
print('CPU checks passed: elastic sampler DP4->DP2/1; cursor alignment/GBS guard; both layouts 100-step topology/GA/CP/TP coverage; corrupt slice rejected')

