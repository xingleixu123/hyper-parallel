import ast,hashlib,json,re
from pathlib import Path
root=Path(__file__).resolve().parent; original=(root/'repro/executed-entry.py').read_text(); repro=(root/'repro/repro.py').read_text()
for p in (root/'repro').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
anchor='    from hyper_parallel.core.dtensor.device_mesh import init_device_mesh'
assert original[original.index(anchor):]==repro[repro.index(anchor):]
for case in ('default','replicate'):
 for rank in range(4):
  d=root/'reports'/'swiglu-tp'/case/('rank-'+str(rank)); raw=(d/'micro-result.json').read_bytes(); result=json.loads(raw); done=json.loads((d/'complete.json').read_bytes())
  assert result['rank']==rank and result['case']==case and done['report_sha256']==hashlib.sha256(raw).hexdigest()
  assert result['classification']==('exception' if case=='default' else 'compared')
patterns={'private_key':r'-----BEGIN [A-Z ]*PRIVATE KEY-----','credential':r'(?i)(?:password|api[_-]?key|secret|bearer)\s*[:=]\s*[^\s,]+','config_leak':r'(?i)\.ya?ml|data_prefix|lr_scheduler|weight_decay','host_private_path':r'/home/xxl/|/nfs/staging/'}
leaks={k:[str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='check_bundle.py' and p.suffix in ('.json','.log','.py') and re.search(v,p.read_text(errors='replace'))] for k,v in patterns.items()}
assert not any(leaks.values()),leaks
print(json.dumps({'status':'cpu_static_pass','measured_compute_body_exact':True,'cases':2,'rank_reports':8,'complete_report_hashes_verified':True,'scan_findings':leaks,'standalone_npu_run':False},indent=2))
