import hashlib,json
from pathlib import Path
base=Path('/home/xxl/hp-mf-master-sixpr-20261102');dst=base/'runtime/xing4-ep2-87dac1da-5-to105-a2';source=base/'fourcard-canonical-100/xing4-fsdp4-87dac1da-5-to105-a1';audit=base/'fourcard-crosslaunch-audit';c=json.loads((base/'runtime/xing4-fsdp4-87dac1da-5-to105-a1/spec.json').read_text());v=json.loads((audit/'verify-spec.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
c.update(run_id='xing4-ep2-87dac1da-5-to105-a2',mode='resume',stop=105,master_port=32101,hccl_start=32120,config='ep2.yaml',entry_sha256=sha(dst/'crosslaunch/ep-fourcard/entry.py'),config_sha256=sha(dst/'ep2.yaml'),source_manifest=str(dst/'source-manifest.json'),checkpoint_path=str(source/'checkpoints/global_step_5'),reference_path=str(source),baseline_output=str(source),checkpoint_sha256=v['checkpoint_sha256'],reference_sha256={str(i):sha(source/('rank-'+str(i))/'checkpoint-5-live-state.json') for i in range(4)},crosslaunch_path=str(dst/'crosslaunch'),source_verify_spec=str(audit/'verify-spec.json'),source_verify_spec_sha256=sha(audit/'verify-spec.json'),source_verify_receipt=str(audit/'source-verify.json'),source_verify_receipt_sha256=sha(audit/'source-verify.json'),image='sha256:5f9e677f439d11ab4e7c888f54b142335a27eee7411cbd2ce894a4798a70146d',gate_asset_sha256={n:sha(dst/'crosslaunch'/n) for n in ['ep_release.py','verify_dcp.py','local_hash.py']});c['paths']['scripts']=str(dst)
out=dst/'spec.json'
with out.open('x') as f:json.dump(c,f,indent=2);f.write('\n')
print(json.dumps({'spec':str(out),'sha256':sha(out),'entry_sha256':c['entry_sha256'],'gate':c['gate_asset_sha256']}))
