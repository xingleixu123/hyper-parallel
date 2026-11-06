from pathlib import Path
import hashlib,json,subprocess,sys
R=Path(__file__).parent
m=json.loads((R/'manifest.json').read_text(encoding='utf8'))
actual={p.relative_to(R).as_posix() for p in R.rglob('*') if p.is_file()}
assert actual==set(m['files'])|{'manifest.json','SHA256SUMS'}
for n,v in m['files'].items():
 p=R/n
 assert not p.is_symlink() and p.stat().st_size==v['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256']
 assert p.suffix.lower() not in ('.yaml','.yml','.bin','.idx','.pt','.pth','.ckpt','.safetensors')
sums=dict(row.split('  ',1)[::-1] for row in (R/'SHA256SUMS').read_text().splitlines())
assert sums=={**{n:v['sha256'] for n,v in m['files'].items()},'manifest.json':hashlib.sha256((R/'manifest.json').read_bytes()).hexdigest()}
subprocess.run([sys.executable,'-B',str(R/'numerical/verify_final_package.py')],check=True)
a=json.loads((R/'website/analysis-public.json').read_text(encoding='utf8'))
assert a['status']=='complete' and a['render']['site_mode']=='online' and a['data_points']==88000 and not a['visible_interactive_verified']
assert len(list((R/'website/charts').glob('*.png')))==36 and len(list((R/'website/website-csv').glob('*.csv')))==36
assert len(list((R/'figures').glob('*.png')))==7 and len(list((R/'figures').glob('*.svg')))==7
print('VERIFIED: seven full precision/state windows; 88,000 online-site points; 36 original PNG/CSV pairs; seven unsmoothed summary charts. Visible interactive delivery remains blocked; external CI not claimed green.')
