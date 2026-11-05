"""Rebuild exact reviewed composite from archived overlay; never launches."""
import argparse,hashlib,json,pathlib,shlex,subprocess
q=pathlib.Path(__file__).parent;p=argparse.ArgumentParser();p.add_argument('--kind',choices=['single','four'],required=True);p.add_argument('--label',required=True);a=p.parse_args();assert a.label.isalnum();raw=(q/(a.kind+'-manifest.json')).read_bytes();manifest=json.loads(raw);overlay=(q/(a.kind+'-overlay.tar')).read_bytes();digest=hashlib.sha256(raw).hexdigest();base='/home/xxl/hp-mf-master-sixpr-20261102';root=base+'/xing4-review-readme-20261105/restage-'+a.label;src=root+'/source';orig=base+'/runtime/xing4-reuse-20261105-'+('single-7607542586b646a4' if a.kind=='single' else 'four-993fa3626388ff11')+'/source'
def run(c,data=None):
 r=subprocess.run(['ssh','-o','BatchMode=yes','129-root',c],input=data,capture_output=True,timeout=180)
 if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
 return r.stdout
run('set -eu; umask 077; test ! -e '+shlex.quote(root)+'; mkdir '+shlex.quote(root)+'; cp -a '+shlex.quote(orig)+' '+shlex.quote(src)+'; tar -xf - -C '+shlex.quote(src),overlay)
code="""import json,sys,pathlib,hashlib
p=pathlib.Path(sys.argv[1]);f=json.load(sys.stdin)['files'];assert {x.relative_to(p).as_posix() for x in p.rglob('*') if x.is_file()}==set(f)
for n,v in f.items():
 x=p/n;assert not x.is_symlink() and x.stat().st_size==v['size'] and hashlib.sha256(x.read_bytes()).hexdigest()==v['sha256'],n
print(len(f))
"""
assert int(run('python3 -B -c '+shlex.quote(code)+' '+shlex.quote(src),raw))==len(manifest['files']);print(json.dumps({'source':src,'manifest_sha256':digest,'files':len(manifest['files'])}))
