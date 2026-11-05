
import csv,hashlib,json,math,statistics
from pathlib import Path
d=Path(__file__).resolve().parent
base=d/"regression/performance"
names=("perfold2-fsdp4","perfnew-fsdp4","perfold2-cp2dp2","perfnew2-cp2dp2")
available=[p.name for p in base.iterdir() if p.is_dir()]
if names[-1] not in available:
 alternatives=[x for x in available if x.startswith("perfnew") and x.endswith("cp2dp2")]
 assert len(alternatives)==1,available
 names=(*names[:3],alternatives[0])
out={};table=[];raw={}
for name in names:
 paths=[base/name/f"rank-{r}/steps.json" for r in range(4)]
 rows=[json.loads(p.read_text()) for p in paths]
 assert all([r["step"] for r in rank]==list(range(6,26)) for rank in rows)
 raw[name]=rows
 secs=[max(rows[r][i]["sync_seconds"] for r in range(4)) for i in range(5,20)]
 percard=[{"rank":rank,"peak_allocated_bytes":max(r["peak_allocated_bytes"] for r in data[5:]),
           "peak_reserved_bytes":max(r["peak_reserved_bytes"] for r in data[5:])} for rank,data in enumerate(rows)]
 mu=statistics.mean(secs);sd=statistics.stdev(secs)
 out[name]={"mean_seconds":mu,"median_seconds":statistics.median(secs),"std_seconds":sd,
            "min_seconds":min(secs),"max_seconds":max(secs),"cv_percent":sd/mu*100,
            "sequences_per_second":4/mu,"input_tokens_per_second":4*4096/mu,
            "per_card_memory":percard,"raw_sha256":[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]}
 for i,sec in enumerate(secs):
  table.append({"run":name,"step":i+11,"max_rank_seconds":sec,"input_tokens_per_second":4*4096/sec})
for old,new in ((names[0],names[1]),(names[2],names[3])):
 for rank in range(4):
  for a,b in zip(raw[old][rank],raw[new][rank]):
   assert a["lr_before"]==b["lr_before"]
   assert a["microbatches"]==b["microbatches"]
 out[new]["change_vs_old_percent"]=(out[new]["mean_seconds"]/out[old]["mean_seconds"]-1)*100
with (d/"performance-steps.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
result={"scope":"Serial isolated short regression; 5 warmup +15 measured steps, synchronized max-rank time including sample hashes; not intrinsic model benchmark",
"old_source":"frozen d780 composite (not immediate f96 parent)","new_source":"e2a128bcd273 composite with tested current model and six 1caf Git blobs","GBS":4,"LBS":1,"sequence":4096,"results":out}
(d/"root-performance-verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
