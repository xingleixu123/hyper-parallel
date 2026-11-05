#!/usr/bin/env python3
"""CPU-only guard component for review; no launcher action is enabled."""
import fcntl,json,os,re,stat,time
from pathlib import Path
CARDS=(4,5,6,7)
LOCK_ROOT=Path('/tmp/hp-mf-cann92-pr301-20261027')

def fail(s): raise RuntimeError(s)
def tick(pid):
 s=Path('/proc/%d/stat'%pid).read_text();return int(s[s.rfind(')')+2:].split()[19])
def acquire_locks():
 if LOCK_ROOT.is_symlink() or LOCK_ROOT.stat().st_uid!=0:fail('lock root identity')
 held=[]
 try:
  for card in CARDS:
   p=LOCK_ROOT/('module-card%d.lock'%card)
   fd=os.open(p,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW|os.O_CLOEXEC,0o600);held.append(fd);s=os.fstat(fd)
   if not stat.S_ISREG(s.st_mode) or s.st_uid!=0 or s.st_nlink!=1 or stat.S_IMODE(s.st_mode)&0o077:fail('lock identity '+str(p))
   fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  return held
 except BaseException:
  for fd in reversed(held):os.close(fd)
  raise

def groups(pid):
 out={}
 for line in Path('/proc/%d/cgroup'%pid).read_text().splitlines():
  _,names,path=line.split(':',2)
  for name in names.split(','):out[name]=path
 return out

def pid_ns_ids(pid):
 text=Path('/proc/%d/status'%pid).read_text()
 match=re.search(r'^NSpid:\s+([\d\s]+)$',text,re.M)
 if not match:fail('NSpid unavailable')
 return [int(v) for v in match.group(1).split()]

def verify_device_cgroup(path):
 base=Path('/sys/fs/cgroup/devices').resolve();folder=(base/path.lstrip('/')).resolve()
 if not folder.is_relative_to(base):fail('cgroup path escapes')
 want={(os.major(os.stat('/dev/davinci%d'%n).st_rdev),os.minor(os.stat('/dev/davinci%d'%n).st_rdev)) for n in CARDS}
 if len(want)!=4 or len({x[0] for x in want})!=1:fail('unexpected accelerator node numbers')
 major=next(iter(want))[0];seen=set()
 for rule in (folder/'devices.list').read_text().splitlines():
  kind,dev,perms=rule.split();a,b=dev.split(':')
  if kind not in ('c','a') or not set(perms)&set('rw'):continue
  if a=='*' or a==str(major):
   if a=='*' or b=='*' or not b.isdigit() or (major,int(b)) not in want:fail('devices cgroup allows other NPU')
   if set('rw').issubset(perms):seen.add((major,int(b)))
 if seen!=want:fail('missing card device permissions')
 return sorted(seen)

def verify_ranks(rank_receipts,target_pid,tag,expected_groups,run_id):
 """Map namespace PIDs to live host processes; return identity evidence, never initialize NPU."""
 ns=os.readlink('/proc/%d/ns/pid'%target_pid); matches={}
 for path in Path('/proc').iterdir():
  if not path.name.isdigit():continue
  pid=int(path.name)
  try:
   if os.readlink(path/'ns/pid')!=ns:continue
   g=groups(pid)
   if any(g.get(k)!=expected_groups[k] for k in ('devices','memory')):continue
   env=(path/'environ').read_bytes().split(b'\0')
   if ('HP_PR310_ID='+tag).encode() not in env:continue
   ids=pid_ns_ids(pid);matches[ids[-1]]=(pid,tick(pid))
  except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 out=[]
 if len(rank_receipts)!=4:fail('exactly 4 ready receipts required')
 for rank in range(4):
  r=rank_receipts[rank]
  if r.get('rank')!=rank or r.get('local_rank')!=rank or r.get('world_size')!=4 or r.get('run_id')!=run_id:fail('rank identity differs')
  p=r.get('pid');host=matches.get(p)
  if not host or host[1]!=r.get('starttick'):fail('rank process identity or cgroup differs')
  out.append({'rank':rank,'namespace_pid':p,'host_pid':host[0],'starttick':host[1],'pid_namespace':ns,'devices_cgroup':expected_groups['devices'],'memory_cgroup':expected_groups['memory']})
 if len({x['host_pid'] for x in out})!=4:fail('duplicate host rank pid')
 return out

def main():
 fail('review-only guard component: no four-card launcher action implemented')
if __name__=='__main__':main()
