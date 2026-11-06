"""Check a throwaway public checkout; optionally export the exact source file subset."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--checkout',type=Path,required=True)
ap.add_argument('--candidate',action='store_true');ap.add_argument('--export',type=Path);a=ap.parse_args()
manifest_path=(ROOT/'repro/candidate-source-manifest.json' if a.candidate else
               ROOT/'model-audit/router-aux-native-reduction-probe/source-manifest.json')
m=json.loads(manifest_path.read_bytes());verified=[]
for name,want in m['files'].items():
    relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts
    p=a.checkout/relative;raw=p.read_bytes()
    assert len(raw)==want['size'] and hashlib.sha256(raw).hexdigest()==want['sha256'],name
    verified.append((p,relative))
if a.export:
    assert not a.export.exists(),'export must be a new empty destination'
    a.export.mkdir(parents=True)
    for p,relative in verified:
        target=a.export/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
print(json.dumps({'files':len(verified),'all_bytes_match':True,'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  'candidate':a.candidate,'exported_to':str(a.export) if a.export else None}))
