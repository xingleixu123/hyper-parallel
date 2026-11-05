import csv,json,math,statistics
from pathlib import Path
d=Path(__file__).parent
old=d.parent/"xing4-cp-integration-20261105/regression/results"
out={"window":[86,105],"excluded_initial_updates":80,
 "timing":"per-step NPU synchronize before/after; max over four ranks; includes sample hashing",
 "scope":"paired historical runs, not a randomized isolated benchmark; BF16 probe and aborted EP excluded",
 "throughput":"nominal input tokens/s = 4 * 4096 / mean(max-rank seconds)",
 "memory":"Torch NPU allocator peaks per optimization step; allocated/reserved, not full driver HBM",
 "configurations":{}}
records=[]
for mode in ("fsdp4","cp2dp2","tp2cp2"):
 results={}
 for label,root in (("previous_1caf",old),("candidate",d/"results")):
  ranks=[[x for x in json.loads((root/mode/f"rank-{r}"/"steps.json").read_bytes()) if 86<=x["step"]<=105] for r in range(4)]
  assert all([x["step"] for x in rows]==list(range(86,106)) for rows in ranks)
  times=[max(ranks[r][i]["sync_seconds"] for r in range(4)) for i in range(20)]
  assert all(t>0 and math.isfinite(t) for t in times)
  mean=statistics.mean(times)
  results[label]={"mean_seconds":mean,"median_seconds":statistics.median(times),
   "p95_seconds":sorted(times)[18],"min_seconds":min(times),"max_seconds":max(times),
   "stdev_seconds":statistics.pstdev(times),"nominal_tokens_per_second":16384/mean,
   "per_rank_peak_allocated_gib":[max(x["peak_allocated_bytes"] for x in rows)/2**30 for rows in ranks],
   "per_rank_peak_reserved_gib":[max(x["peak_reserved_bytes"] for x in rows)/2**30 for rows in ranks]}
  for rank,rows in enumerate(ranks):
   for row in rows:records.append({"configuration":mode,"version":label,"rank":rank,"step":row["step"],
      **{k:row[k] for k in ("sync_seconds","peak_allocated_bytes","peak_reserved_bytes")}})
 results["mean_time_change_percent"]=(results["candidate"]["mean_seconds"]/results["previous_1caf"]["mean_seconds"]-1)*100
 out["configurations"][mode]=results
(d/"performance-diagnostic.json").write_text(json.dumps(out,indent=2))
with (d/"performance-raw.csv").open("w",newline="") as f:
 writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
print(json.dumps(out,indent=2))
