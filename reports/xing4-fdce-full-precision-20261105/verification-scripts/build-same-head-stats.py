"""Compare current-head configurations to the current-head FSDP replay."""
import csv,hashlib,json,math
from pathlib import Path
d=Path(__file__).parent;src=d/"root-verification-all/fourcard-versus-fsdp.csv"
rows=list(csv.DictReader(src.open(newline="",encoding="utf8")))
cases={}
for r in rows:
 name=r["configuration"];step=int(r["step"]);assert step not in cases.setdefault(name,{})
 cases[name][step]={k:float(v) for k,v in r.items() if k not in ("configuration","step")}
assert set(cases)=={"fsdp_full","fsdp_backbone","ep2","cp2dp2","tp2cp2"}
expected=list(range(1501,2501));ref=cases["fsdp_full"]
assert all(sorted(v)==expected for v in cases.values())
stats={};out=[]
for name in ("fsdp_backbone","ep2","cp2dp2","tp2cp2"):
 rr=[]
 for step in expected:
  c,b=cases[name][step],ref[step]
  rr.append({"configuration":name,"step":step,"loss":c["loss"],"fsdp_loss":b["loss"],"mtp_raw":c["mtp_raw"],"fsdp_mtp_raw":b["mtp_raw"],"loss_delta":c["loss"]-b["loss"],"mtp_delta":c["mtp_raw"]-b["mtp_raw"]})
 result={}
 for label,key in (("LOSS","loss_delta"),("MTPLOSS","mtp_delta")):
  vals=[r[key] for r in rr];mean=math.fsum(vals)/1000;ix=max(range(1000),key=lambda i:abs(vals[i]))
  result[label]={"steps":1000,"signed_mean":mean,"abs_signed_mean":abs(mean),"positive":sum(x>0 for x in vals),"negative":sum(x<0 for x in vals),"zero":sum(x==0 for x in vals),"mae":math.fsum(map(abs,vals))/1000,"max_abs":abs(vals[ix]),"max_abs_step":expected[ix],"mean_threshold":.005,"mean_pass":abs(mean)<=.005,"identical":all(x==0 for x in vals)}
  for i,r in enumerate(rr):r[key+"_cumulative_mean"]=math.fsum(vals[:i+1])/(i+1)
 stats[name]=result;out+=rr
dst=d/"root-verification-all"
with (dst/"fourcard-same-head.csv").open("w",encoding="utf8",newline="") as f:
 writer=csv.DictWriter(f,fieldnames=out[0]);writer.writeheader();writer.writerows(out)
summary={"comparison":"current configuration - current fdce FSDP replay","window":[1501,2500],"model_head":"fdce1a9a2285f1ab76f4c73d4e68b620b134eaea","source_manifest_sha256":"7dd9613c60843e646e15f6bb9879f477c201d63824793859e5386bd8059d8f4f","source_csv_sha256":hashlib.sha256(src.read_bytes()).hexdigest(),"reference_scope":"FSDP independent 1501-2500 replay from the same validated continuous-baseline checkpoint1500; this replay is not claimed as a new 0-2500 continuous run.","metrics":stats}
(dst/"same-head-statistics.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
print(json.dumps(stats,indent=2))
