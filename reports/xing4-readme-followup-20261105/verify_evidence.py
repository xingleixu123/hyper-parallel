import hashlib,json,subprocess
from pathlib import Path
d=Path(__file__).parent
r=Path(r"\\wsl.localhost\Ubuntu\home\xxl\codes-github\hp-mf-sixpr-run-20261102")
hashes=json.loads((d/"npu-evidence-sha256.json").read_bytes())
for name,v in hashes.items():
 b=(d/name).read_bytes();assert len(b)==v["size"] and hashlib.sha256(b).hexdigest()==v["sha256"],name
stage=json.loads((d/"stage.json").read_bytes())
binding=json.loads((d/"xing-commit-binding.json").read_bytes())
assert set(binding["changes"])==set(stage["changes"])
for name,h in binding["changes"].items():
 assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h==stage["changes"][name]["after"]["sha256"]
runs=[]
for attempt,expected in (("attempt2","22 passed"),("attempta4","1 passed")):
 v=json.loads((d/attempt/"controller.json").read_bytes())
 p=json.loads((d/attempt/"physical-mapping-release.json").read_bytes())
 results=json.loads((d/attempt/"probe-result.json").read_bytes())
 assert v["returncode"]==0 and results["pytest_returncode"]==0
 assert v["physical_card"]==p["physical_card"]==5 and v["device_cgroup_verified"] and v["container_removed"]
 assert p["worker_cgroups"]["devices"]==p["cgroups"]["devices"]
 assert str(p["worker"]["host_pid"]) in p["npu_smi"]
 assert expected in (d/attempt/"stdout.log").read_text(encoding="utf-8")
 runs.append({"attempt":attempt,"physical_card":5,"host_pid":p["worker"]["host_pid"],"exit_code":0,"result":expected})
rows=[json.loads(x) for x in (d/"attempta4/stdout.log").read_text().splitlines() if x.startswith("{")]
assert rows[0]["count"]==[71,71] and rows[1]["count"]==[0,0]
assert all(x["device"]=="npu:0" and x["loaded_weight_bias"]=="unchanged" for x in rows)
cpu=[]
for f in ("hf-guard-candidate.log","hf-guard-hf-init-stage.log"):
 lines=(d/f).read_text().splitlines()
 cpu.append(json.loads(next(x.removeprefix("RESULT_JSON ") for x in lines if x.startswith("RESULT_JSON "))))
assert cpu[0]["random_hash"]==cpu[1]["random_hash"]
assert cpu[0]["cases"]["count"]["bias"]==0 and cpu[1]["cases"]["count"]["bias"]==3
assert cpu[1]["cases"]["hc_fn"]["scale"]==2 and cpu[1]["cases"]["hc_fn"]["base"]==4
v={"source_binding":binding,"npu_file_hashes_verified":len(hashes),"runs":runs,"cpu_random_init_exact":True,"cpu_partial_load_preservation_verified":True,"loader_npu_before_after_verified":True,
"controller_display_hashes_not_used":["controller_sha256","worker_sha256","candidate_sha256","review_sha256"],
"scope":"module/loading regression only; no new complete-training acceptance"}
(d/"root-evidence-verification.json").write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"file_hashes_verified":len(hashes),"runs":runs,"random_init_exact":True},ensure_ascii=False))
