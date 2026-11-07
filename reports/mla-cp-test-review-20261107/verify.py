from pathlib import Path
import hashlib,json,re
r=Path(__file__).resolve().parent
manifest=json.loads((r/"manifest.json").read_text())
for name,spec in manifest["files"].items():
 raw=(r/name).read_bytes()
 assert len(raw)==spec["bytes"] and hashlib.sha256(raw).hexdigest()==spec["sha256"],name
controller=json.loads((r/"NPU/controller-summary.json").read_text())
assert controller["status"]=="failed" and controller["returncode"]==0
records=[]
for rank in range(4):
 text=(r/("NPU/rank-"+str(rank)+".log")).read_text()
 assert "1 passed" in text and not re.search(r"[1-9]\d* (failed|skipped)",text)
 for line in text.splitlines():
  if '{"tp":' in line:records.append(json.loads(line[line.index("{"):]))
assert len(records)==3
assert {(x["strategy"],x["overlap"]) for x in records}=={("expanded_ulysses",False),("expanded_ulysses",True),("latent_kv_head",True)}
assert all(x["tp"]==x["cp"]==2 and x["backward_passes"]==2 and x["gradient_relative_l2"]<0.025 for x in records)
assert "1 passed" in (r/"NPU/launcher.log").read_text()
print("File identities and native pytest outcomes verified; the controller remains failed.")
