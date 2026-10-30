#!/usr/bin/env python3
"""Recompute archived green/red gradients and losses with exact rational arithmetic."""
import hashlib,json
from fractions import Fraction as F
from pathlib import Path
import ast

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
GREEN=ROOT/"results"/"green"
RED=ROOT/"results"/"red"


def read(path): return json.loads(path.read_bytes())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def ensure(ok, message):
    if not ok: raise RuntimeError(message)
def matrix(rows): return [[F(str(x)) for x in row] for row in rows]
def scalar(x): return F(str(x))
def encoded(mat): return [[float(x) for x in row] for row in mat]
def repetition(v): return [[F(x)]*4 for x in v]
def record_map(rows):
    d={r["fqn"]:r for r in rows}
    ensure(set(d)=={"common.weight","early.weight"} and len(rows)==2,"FQN coverage")
    return d
def value(rec,shape):
    ensure(rec is not None and rec["dtype"]=="torch.float32" and rec["shape"]==shape,"missing or wrong FP32 shape")
    return matrix(rec["values"])
def expected_input(rank,micro): return [rank+1,micro+1,rank+micro+1,2]
def loss_value(x,common,early,use_early):
    return sum(F(x[i])*(common[i][j]+(early[i][j] if use_early else 0)) for i in range(4) for j in range(4))
def gradient_sum(xs):
    return [sum(F(row[i]) for row in xs) for i in range(4)]
def file_for(root,rank,fusion,label):
    return root/("rank-%d-fusion-%s-%s.json"%(rank,str(fusion).lower(),label))

# Recompute from public inputs only; no device, network, private paths or process records.
manifest_path=ROOT/"source"/"source-manifest.json"
ensure(digest(manifest_path)=="a06da217316eff5f307898e8822c6f98b21ac7e1117a5b326e38b60876460df6","manifest pin")
manifest=read(manifest_path);scheduler="hyper_parallel/core/fully_shard/hsdp_scheduler.py"
patch="51abc1d50805960b50c334c8f5a5b5c4a5b2a5e0d7984686bae3d2e553499fe2"
ensure(len(manifest["files"])==2659 and manifest["head"]=="5cde7c54ce3da8be8689d0783cf9b70d5306117a" and manifest["files"][scheduler]["sha256"]=="2d311ed2ee490370118100d457dfbfe4f773c7d2ba5997f180238a10a1ad4cdd","source identity")
v1=(ROOT/"workers"/"worker-red-v1.py").read_bytes();v2=(ROOT/"workers"/"worker-green-v2.py").read_bytes()
ensure(hashlib.sha256(v1).hexdigest()=="5f1bdca3bb94c025d751c40a549bda0af87fae9ec5a8e47b1b71886a9ff06ed1" and hashlib.sha256(v2).hexdigest()=="0268c21fb43c47ffeada364e90682e9793fdf4ea4bfb30002db9ed08bbc741b7","worker pins")
a=b"deadline = time.monotonic() + 180";b=b"deadline = time.monotonic() + 600"
ensure(v1.count(a)==1 and v1.replace(a,b)==v2,"worker bytes differ beyond TTL")
ensure(ast.dump(ast.parse(v1.replace(a,b).decode()),include_attributes=False)==ast.dump(ast.parse(v2.decode()),include_attributes=False),"normalized AST differs")
names=["both_true","both_false","micro0_only_true","micro0_only_false"]
xs={(r,m):expected_input(r,m) for r in range(2) for m in range(2)}
first=gradient_sum([xs[(0,0)],xs[(1,0)]])
second=gradient_sum([xs[(0,1)],xs[(1,1)]])
both=[a+b for a,b in zip(first,second)]
ensure(first==[3,2,3,4] and second==[3,4,5,4] and both==[6,6,8,8],"sum formula")
tables=[]
results=[]
red_counts={"passed":0,"failed":0}
green_counts={"passed":0,"failed":0}
for fusion in (False,True):
 for label in names:
  branch_both=label.startswith("both_")
  sync_each=label.endswith("_true")
  docs=[read(file_for(GREEN,r,fusion,label)) for r in range(2)]
  red_docs=[read(file_for(RED,r,fusion,label)) for r in range(2)]
  reference={}
  for name,offset in (("common.weight",1),("early.weight",3)):
   reconstructed=[]
   for d in docs:reconstructed+=matrix(record_map(d["initial"])[name]["parameter"]["values"])
   independent=[[F(4*i+j+offset,16) for j in range(4)] for i in range(4)]
   ensure(reconstructed==independent,"initial logical weights "+name)
   reference[name]=reconstructed
  for rank,(doc,red) in enumerate(zip(docs,red_docs)):
   expected_label=(rank,fusion,label)
   ensure((doc["rank"],doc["fusion"],doc["case"])==expected_label and (red["rank"],red["fusion"],red["case"])==expected_label,"case identity")
   ensure(doc["status"]=="passed" and not doc["failures"],"green status")
   green_counts[doc["status"]]+=1
   red_counts[red["status"]]+=1
   ensure(doc["arm"]=="green" and red["arm"]=="red" and doc["mesh_shape"]==red["mesh_shape"]==[2] and doc["reduction"]==red["reduction"]=="SUM","case topology")
   ensure(doc["source_pins"][scheduler]==patch and red["source_pins"][scheduler]==manifest["files"][scheduler]["sha256"],"actual scheduler pins")
   ensure({k:v for k,v in doc["source_pins"].items() if k!=scheduler}=={k:v for k,v in red["source_pins"].items() if k!=scheduler},"other keyed sources differ")
   ensure(doc["initial"]==red["initial"] and doc["phases"][0]["observation"]==red["phases"][0]["observation"],"red/green initial or first-micro observations differ")
   init=record_map(doc["initial"])
   for model_name,info in doc["model_types"].items():
    ensure(info["comm_fusion"] is fusion and info["zero_copy"] is False and info["device"]=="npu:%d"%rank,"actual wrapper/fusion metadata")
   for name,row in init.items():
    expected=reference[name][2*rank:2*rank+2]
    ensure(value(row["parameter"],[2,4])==expected and row["main_grad"] is None and row["grad"] is None,"initial parameter or gradient")
    ensure(row["pending_unsharded"] is None and row["pending_reduce_partial"] is None and not row["pending_group_buckets"],"initial pending")
    ensure(row["shard_rank"]==rank and row["shard_world_size"]==row["shard_group_size"]==2,"shard identity")
   ensure(len(doc["phases"])==len(red["phases"])==2,"microbatch count")
   measured_losses=[]
   for m in range(2):
    gp,rp=doc["phases"][m],red["phases"][m]
    x=xs[(rank,m)]
    use_early=branch_both or m==0
    expect_loss=loss_value(x,reference["common.weight"],reference["early.weight"],use_early)
    for ph in (gp,rp):
     ensure(ph["micro"]==m and ph["input"]==[x] and ph["use_early"] is use_early and scalar(ph["loss"])==expect_loss,"input/use/loss differs")
     for name,row in record_map(ph["observation"]).items():
      ensure(value(row["parameter"],[2,4])==reference[name][2*rank:2*rank+2],"weight changed")
    measured_losses.append(float(expect_loss))
   phase0=record_map(doc["phases"][0]["observation"])
   phase1=record_map(doc["phases"][1]["observation"])
   for name in ("common.weight","early.weight"):
    a,b=phase0[name],phase1[name]
    ensure(a["pending_reduce_partial"] is None and not a["pending_group_buckets"],"first partial/bucket")
    if sync_each:
     ensure(a["pending_unsharded"] is None and value(a["main_grad"],[2,4])==repetition(first[2*rank:2*rank+2]),"sync micro0 gradient")
    else:
     ensure(a["main_grad"] is None and value(a["pending_unsharded"],[4,4])==repetition(xs[(rank,0)]),"no-sync pending")
    expected=both if name=="common.weight" or branch_both else first
    ensure(value(b["main_grad"],[2,4])==repetition(expected[2*rank:2*rank+2]),"green final gradient")
    ensure(b["pending_unsharded"] is None and b["pending_reduce_partial"] is None and not b["pending_group_buckets"],"green pending not drained")
    ensure(b["reduce_grads"] is True,"final sync flag")
   red_final=record_map(red["phases"][1]["observation"])
   ensure(value(red_final["common.weight"]["main_grad"],[2,4])==repetition(both[2*rank:2*rank+2]),"red common gradient")
   missing=(not branch_both and not sync_each)
   expected_early=both if branch_both else first
   red_early=red_final["early.weight"]
   if missing:
    ensure(red["status"]=="failed" and red_early["main_grad"] is None and value(red_early["pending_unsharded"],[4,4])==repetition(xs[(rank,0)]),"red missing-branch observation")
   else:
    ensure(red["status"]=="passed" and value(red_early["main_grad"],[2,4])==repetition(expected_early[2*rank:2*rank+2]),"red control")
    ensure(red_early["pending_unsharded"] is None and red_early["pending_reduce_partial"] is None and not red_early["pending_group_buckets"],"red control pending")
   results.append({"rank":rank,"fusion":fusion,"case":label,"green_sha256":digest(file_for(GREEN,rank,fusion,label)),"red_sha256":digest(file_for(RED,rank,fusion,label)),"losses":measured_losses,"expected_common":encoded(repetition(both[2*rank:2*rank+2])),"expected_early":encoded(repetition(expected_early[2*rank:2*rank+2])),"green_early_main_grad":phase1["early.weight"]["main_grad"]["values"],"green_early_pending":phase1["early.weight"]["pending_unsharded"],"red_early_main_grad":None if red_early["main_grad"] is None else red_early["main_grad"]["values"],"red_early_pending":None if red_early["pending_unsharded"] is None else red_early["pending_unsharded"]["values"],"green_status":doc["status"],"red_status":red["status"]})
ensure(green_counts=={"passed":16,"failed":0} and red_counts=={"passed":12,"failed":4},"aggregate case counts")
for root,arm in ((RED,"red"),(GREEN,"green")):
 for rank in range(2):
  t=read(root/("terminal-rank-%d.json"%rank))
  ensure(t["rank"]==rank and t["failed_any_case"] is (arm=="red"),"rank terminal")
  ensure(t["source_pins_after"][scheduler]==(patch if arm=="green" else manifest["files"][scheduler]["sha256"]),"post-run scheduler pin")
print(json.dumps({"status":"passed", "scope":"local exact-rational recomputation of archived toy values; no runtime hardware proof", "red":red_counts, "green":green_counts, "sum_rows":{"first":list(map(int,first)),"second":list(map(int,second)),"both":list(map(int,both))}, "comparisons":results},sort_keys=True,indent=2))
