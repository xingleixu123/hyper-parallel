"""Build a hash-pinned CP launch spec from a verified local baseline spec and anchor."""
import argparse,hashlib,json
from pathlib import Path
import candidate

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cards',choices=['0,1,2,3','4,5,6,7'],required=True);ap.add_argument('--source',required=True);ap.add_argument('--source-head',required=True);ap.add_argument('--source-tree',required=True);ap.add_argument('--base-spec',required=True);ap.add_argument('--scripts',required=True);ap.add_argument('--anchor',required=True);ap.add_argument('--output-root',required=True);ap.add_argument('--output-spec',required=True);ap.add_argument('--layout',choices=['cp2dp2','tp2cp2'],required=True);ap.add_argument('--run-id',required=True);ap.add_argument('--master-port',type=int,required=True);ap.add_argument('--hccl-start',type=int,required=True);a=ap.parse_args()
 scripts=Path(a.scripts).resolve(strict=True);anchor=Path(a.anchor).resolve(strict=True);base=anchor/'baseline';audit=anchor/'audit';verify=audit/'verify-spec.json';receipt=audit/'source-verify.json';manifest=scripts/'source-manifest.json'
 c=json.loads(Path(a.base_spec).read_text());v=json.loads(verify.read_text());r=json.loads(receipt.read_text());m=json.loads(manifest.read_text(encoding='utf-8-sig'))
 if r.get('status')!='passed' or r.get('source_step')!=5 or r.get('spec_sha256')!=sha(verify):raise ValueError('source verification receipt invalid')
 if (m['commit'],m['tree'])!=(a.source_head,a.source_tree):raise ValueError('wrong runtime manifest')
 source=str(Path(a.source).resolve(strict=True)); config=a.layout+'.yaml';gate=scripts/'crosslaunch'
 c.update(image=candidate.IMAGE,cards=[int(x) for x in a.cards.split(',')],run_id=a.run_id,mode='resume',stop=105,master_port=a.master_port,hccl_start=a.hccl_start,config=config,entry_sha256=sha(gate/'cp-fourcard/entry.py'),config_sha256=sha(scripts/config),source_manifest=str(manifest),source_manifest_sha256=sha(manifest),source_head=m['commit'],source_tree=m['tree'],checkpoint_path=str(base/'checkpoints/global_step_5'),reference_path=str(base),baseline_output=str(base),checkpoint_sha256=v['checkpoint_sha256'],reference_sha256={str(i):sha(base/f'rank-{i}'/'checkpoint-5-live-state.json') for i in range(4)},crosslaunch_path=str(gate),source_verify_spec=str(verify),source_verify_spec_sha256=sha(verify),source_verify_receipt=str(receipt),source_verify_receipt_sha256=sha(receipt),gate_asset_sha256={n:sha(gate/n) for n in ['ep_release.py','verify_dcp.py','local_hash.py']},output_root=str(Path(a.output_root).resolve(strict=True)))
 c['paths']['scripts']=str(scripts);c['paths']['source']=source;c['asset_sha256']={n:sha(scripts/n) for n in ('issue12_hp_data.py','ordered_factory.py')}
 out=Path(a.output_spec)
 with out.open('x') as f:json.dump(c,f,indent=2);f.write('\n')
 print(json.dumps({'path':str(out),'sha256':sha(out),'layout':a.layout}))
if __name__=='__main__':main()



