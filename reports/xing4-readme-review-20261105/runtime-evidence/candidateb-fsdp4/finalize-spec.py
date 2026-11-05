import hashlib,json
from pathlib import Path
p=Path('/home/xxl/hp-mf-master-sixpr-20261102/runtime/xing4-fsdp4-1453-resume-5-to105-a1');c=json.loads((p/'spec.initial.json').read_text());m=json.loads((p/'source-manifest.json').read_text())
assert m['commit']=='304b8df54a2731039652c54579a1c7708b7e990d' and m['tree']=='df794fafa7135441b035255867694251ee68f3da' and len(m['files'])==1514
c['source_head']=m['commit'];c['source_tree']=m['tree'];c['paths']['source']='/home/xxl/hp-mf-master-sixpr-20261102/runtime/xing4-master-normalization-304b8df54a2731039652c54579a1c7708b7e990d/source'
out=p/'spec.json'
with out.open('x') as f:json.dump(c,f,indent=2);f.write('\n')
print(hashlib.sha256(out.read_bytes()).hexdigest())
