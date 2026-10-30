#!/usr/bin/env python3
"""Bounded mechanical scan of this publication candidate; no network/device access."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATTERNS = {
    'private_key_header': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'github_token_format': re.compile(rb'(?i)(?:gh[pousr]_[A-Za-z0-9_]{24,}|github_pat_[A-Za-z0-9_]{24,})'),
    'aws_access_key_format': re.compile(rb'(?<![A-Z0-9])AKIA[A-Z0-9]{16}(?![A-Z0-9])'),
    'bearer_value': re.compile(rb'(?i)\bBearer\s+[A-Za-z0-9._~+/-]{24,}'),
    'windows_user_profile': re.compile(rb'(?i)[A-Z]:[\\/]Users[\\/][A-Za-z0-9._-]+[\\/]'),
    'credential_assignment': re.compile(rb'''(?i)(?:access_token|client_secret|api_key|password)[\"']?\s*[:=]\s*[\"'][A-Za-z0-9._~+/-]{16,}'''),
    'long_base64_payload': re.compile(rb'(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{256,}={0,2}(?![A-Za-z0-9+/])'),
}
FORBIDDEN_SUFFIXES={'.yaml','.yml','.npy','.npz','.safetensors','.bin','.idx','.pt','.pth','.ckpt','.env'}
files=sorted((p for p in ROOT.rglob('*') if p.is_file()),key=lambda p:p.relative_to(ROOT).as_posix())
violations=[]
for p in files:
    name=p.relative_to(ROOT).as_posix()
    if p.is_symlink(): violations.append([name,'symlink'])
    if p.suffix.lower() in FORBIDDEN_SUFFIXES: violations.append([name,'forbidden suffix'])
    raw=p.read_bytes()
    try: raw.decode('utf-8')
    except UnicodeDecodeError: violations.append([name,'non-UTF8/binary'])
    for label,pattern in PATTERNS.items():
        matches=pattern.findall(raw)
        if label=='long_base64_payload': matches=[v for v in matches if not re.fullmatch(rb'[0-9a-f]+',v)]
        if matches: violations.append([name,label])
run=ROOT/'run'; cases=list((run/'output').glob('rank-*-fusion-*.json')); terms=list((run/'output').glob('terminal-rank-*.json'))
if len(cases)!=16 or len(terms)!=2: violations.append(['run/output','case/terminal count'])
proof=json.loads((run/'output/physical-mapping-release.json').read_bytes())['rank_proof']
for action in ('observe','release'):
    for stage,tag in (('preimport','pre-obs-root-r1'),('mapping','map-obs-root-r1')):
        for rank in range(2):
            card=rank+1; p=run/('smi-%s-%s-%s/card-%d.stdout'%(action,stage,tag,card)); line=p.read_text(encoding='utf-8')
            if stage=='preimport': okay=line.strip()=='No process in device.'
            else:
                m=re.fullmatch(r'\s*Process id:(\d+)\s+Process name:python\s+Process memory\(MB\):\d+\s*',line)
                okay=bool(m) and int(m.group(1))==proof[rank]['host_pid']
            if not okay: violations.append([p.relative_to(ROOT).as_posix(),'unexpected external SMI entry'])
result={'schema':'public-candidate-scan-v1','status':'passed' if not violations else 'failed','files_scanned':len(files),'bytes_scanned':sum(p.stat().st_size for p in files),'case_count':len(cases),'terminal_count':len(terms),'checks':list(PATTERNS)+['forbidden_file_suffixes','binary_files','symlinks','SMI_contains_only_own_pid_or_idle'],'violations':violations,'limitations':'Mechanical patterns and file selection cannot prove the absence of every possible secret or unrelated sensitive information; human review is still required. The public source manifest names official paths but is not a YAML/data file.'}
print(json.dumps(result,sort_keys=True,ensure_ascii=False))
if violations: raise SystemExit(1)