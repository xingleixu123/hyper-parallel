import sys,hashlib
from pathlib import Path
sys.path.insert(0,'/home/xxl/.cache/codex/xing4-deep-reuse-20261104/after')
sys.path.insert(0,str(Path(__file__).resolve().parent/'crosslaunch'))
import torch
from local_hash import expected_local
from hyper_parallel.core.dtensor.layout import Layout,infer_slice_area_by_layout
from hyper_parallel.core.dtensor.placement_types import Shard,StridedShard
x=torch.arange(48,dtype=torch.float32).reshape(16,3)
layout=Layout((2,2),('fsdp','tp'),init_backend=False)([StridedShard(0,2),Shard(0)]);layout.placement_to_tensor_map(2)
for rank in range(4):
 area=infer_slice_area_by_layout(layout,rank,x.shape);y=x[tuple(slice(a,b) for a,b in area)].contiguous();h=hashlib.sha256(y.view(torch.uint8).numpy().tobytes()).hexdigest()
 rec={'global_shape':list(x.shape),'dtype':str(x.dtype),'local_shape':list(y.shape),'sha256':h,'placements':'(StridedShard(dim=0, split_factor=2), Shard(dim=0))','mesh':{'shape':[2,2],'ranks':[0,1,2,3],'coordinate':[rank//2,rank%2]}}
 assert expected_local(x,rec,rank)['slices']==[list(a) for a in area]
 rec['placements']='(StridedShard(dim=0, split_factor=3), Shard(dim=0))'
 try:expected_local(x,rec,rank);raise AssertionError('bad split accepted')
 except ValueError:pass
print('PASS: 4 CPU ranks, HP Layout native placement/slice equals gate digest; bad split rejected')

