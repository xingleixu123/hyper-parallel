"""Collect bounded run evidence over SSH, verify every file, then remove duplicate archives."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/home/xxl/jt-dev-8npu-20261014"
REMOTE_SCRIPT = r'''
import hashlib,json,sys,tarfile
from pathlib import Path
root=Path('/home/xxl/jt-dev-8npu-20261014').resolve()
case,tag,allow_incomplete=sys.argv[1:]
run=root/'runs'/case
assert run.resolve().parent==root/'runs' and not run.is_symlink()
if allow_incomplete!='1':
 assert (run/'EXIT_CODE').read_text().strip()=='0'
 assert len(list(run.glob('complete-rank*.json')))==8
paths=[p for p in run.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
receipt=root/'scheduling'/(case+'.receipt.json')
receipt_data=json.loads(receipt.read_text())
assert receipt_data['released']
paths.append(receipt)
if receipt_data.get('ownership_event'):
 event=root/receipt_data['ownership_event']
 if event.resolve()!=root/'scheduling'/(case+'.ownership-event.json') or event.is_symlink():
  raise RuntimeError('Unexpected ownership evidence path')
 paths.append(event)
files=[]
for p in sorted(paths):
 assert not p.is_symlink() and root in p.resolve().parents
 assert p.stat().st_size<64*1024*1024, str(p)
 files.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
assert sum(item['bytes'] for item in files)<128*1024*1024
archive=root/'tmp'/(tag+'-evidence.tgz')
manifest_path=root/'evidence'/(tag+'-evidence-manifest.json')
if archive.exists():
 prior=json.loads(manifest_path.read_text())
 assert prior['files']==files and prior['archive_sha256']==hashlib.sha256(archive.read_bytes()).hexdigest()
 print(json.dumps(prior))
 sys.exit(0)
with tarfile.open(archive,'w:gz') as out:
 for p in sorted(paths): out.add(p,arcname=str(p.relative_to(root)),recursive=False)
manifest={'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':files}
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("tag")
    parser.add_argument("--host", choices=("125-root", "129-root"), default="129-root")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if not all(re.fullmatch(r"[a-z0-9-]+", value) for value in (args.case, args.tag)):
        raise ValueError("Case and tag must be simple names")
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", args.host,
                             "python3", "-", args.case, args.tag, "1" if args.allow_incomplete else "0"], input=REMOTE_SCRIPT,
                            encoding="utf-8", capture_output=True, check=True, timeout=90)
    manifest = json.loads(result.stdout)
    archive = ROOT / "tmp" / f"{args.tag}-evidence.tgz"
    subprocess.run(["scp", "-q", f"{args.host}:{REMOTE_ROOT}/tmp/{archive.name}", str(archive)],
                   check=True, timeout=90)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["archive_sha256"]:
        raise RuntimeError("Archive hash mismatch")
    expected = {item["path"]: item for item in manifest["files"]}
    with tarfile.open(archive) as source:
        members = source.getmembers()
        if len(members) != len(expected) or {item.name for item in members} != set(expected):
            raise RuntimeError("Archive inventory mismatch")
        for member in members:
            target = (ROOT / member.name).resolve()
            if not member.isfile() or ROOT.resolve() not in target.parents:
                raise RuntimeError("Unsafe archive member")
            data = source.extractfile(member).read()
            if len(data) != expected[member.name]["bytes"] or hashlib.sha256(data).hexdigest() != expected[member.name]["sha256"]:
                raise RuntimeError(f"Member hash mismatch: {member.name}")
            if target.exists() and target.read_bytes() != data:
                same_receipt = (target.name.endswith(".receipt.json")
                                and json.loads(target.read_bytes()) == json.loads(data))
                if not same_receipt:
                    raise RuntimeError(f"Refusing to replace different evidence: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    for name, entry in expected.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Extracted evidence mismatch: {name}")
    (ROOT / "evidence" / f"{args.tag}-evidence-manifest.json").write_bytes(
        (json.dumps(manifest, indent=2) + "\n").encode())
    cleanup = r'''
import hashlib,sys
from pathlib import Path
root=Path('/home/xxl/jt-dev-8npu-20261014').resolve()
p=root/'tmp'/sys.argv[1]
assert p.resolve().parent==root/'tmp' and not p.is_symlink()
assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2]
p.unlink()
'''
    subprocess.run(["ssh", "-o", "BatchMode=yes", args.host, "python3", "-", archive.name,
                    manifest["archive_sha256"]], input=cleanup, encoding="utf-8", check=True, timeout=30)
    if archive.resolve().parent != (ROOT / "tmp").resolve():
        raise RuntimeError("Local archive path escaped task tmp")
    archive.unlink()
    receipt = {"verified_files": len(expected), "all_hashes_match": True,
               "archive_sha256": manifest["archive_sha256"], "duplicate_archives_removed": ["local", args.host]}
    (ROOT / "evidence" / f"{args.tag}-local-verification.json").write_bytes(
        (json.dumps(receipt, indent=2) + "\n").encode())
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
