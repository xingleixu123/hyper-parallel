from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent.parent
roles=("single","master_swap","candidate_swap")
v={k:json.loads((r/"NPU"/k/"worker-rank-0.json").read_text()) for k in roles}
for x in v.values():
 assert x["status"]=="passed"
 assert x["torch"]=="2.12.0+cpu" and x["torch_npu"]=="2.12.0.post3"
a,b=v["master_swap"]["result"],v["candidate_swap"]["result"]
assert not a["rejected"] and b["rejected"]
assert a["hook_counts"]==b["hook_counts"]=={"pre":0,"post":0}
for k in ("parameter","optimizer_state","tracked_slots","cpu_mirrors","copy_stream_created","no_hook_steps"):
 assert a[k]==b[k],k
assert a["no_hook_steps"]==3 and a["cpu_mirrors"]>0 and a["copy_stream_created"]
assert len(v["single"]["result"])==14
assert sum(x["kind"]=="rope" for x in v["single"]["result"])==8
manifest=json.loads((r/"manifest.json").read_text())
for rel,spec in manifest["files"].items():
 raw=(r/rel).read_bytes()
 assert len(raw)==spec["bytes"] and hashlib.sha256(raw).hexdigest()==spec["sha256"],rel
print("Exported file hashes, 14 single-card checks and no-hook swap parameter/state equality: passed")

expected=[(2,2,2,"expanded_ulysses",False,False,False),(2,2,2,"expanded_ulysses",False,True,False),(2,2,2,"expanded_ulysses",True,True,False),(2,2,2,"latent_kv_head",True,True,True),(1,4,2,"latent_kv_head",True,True,False)]
maximum=0.0
for rank in range(4):
 record=json.loads((r/"NPU/callbacks"/("worker-rank-"+str(rank)+".json")).read_text())
 assert record["rank"]==rank and record["status"]=="passed" and len(record["result"])==5
 for case,sig in zip(record["result"],expected):
  assert tuple(case[k] for k in ("tp","cp","ulysses","strategy","overlap","custom_callbacks","packed"))==sig
  assert case["microbatches"]==2 and case["checkpoint"]=="non-reentrant"
  assert case["gradient_relative_l2"]<0.025
  maximum=max(maximum,case["gradient_relative_l2"])
  if case["custom_callbacks"]:
   offset=((rank//case["tp"])%case["ulysses"])*(8//(case["tp"]*case["ulysses"]))
   calls=[x for x in case["attention_calls"] if x["role"]=="parallel"]
   assert len(calls)>=2 and len(case["rotary_calls"])>=4
   assert all(x["offset"]==offset and x["local_heads"]==8//case["tp"] and x["computed_heads"]==8//(case["tp"]*case["ulysses"]) for x in calls)
print("Five four-card cases, all ranks, head metadata and gradient relative-L2: passed",maximum)
