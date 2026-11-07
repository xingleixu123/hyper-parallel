import hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parent; files={p.relative_to(r).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in r.rglob('*') if p.is_file() and p!=r/'manifest.json'}; (r/'manifest.json').write_text(json.dumps({'algorithm':'sha256','files':dict(sorted(files.items()))},indent=2)+'\n',encoding='utf8')
