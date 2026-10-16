# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Launch isolated native and observed Muon replays on separate idle cards."""
from pathlib import Path
import subprocess,json,hashlib,shutil,datetime
root=Path('/home/xxl/jt-adamw-align-20261013/phase12')
state=Path('/dev/shm/jt-adamw-align-20261013/state/jt-recipe-20261014')
status=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True).stdout
assert all('No running processes found in NPU '+str(i) in status for i in [6])
assert shutil.disk_usage(state).free>1_000_000_000
ports=subprocess.run(['ss','-ltn'],capture_output=True,text=True,check=True).stdout
if ':31679 ' in ports:
 raise ValueError('Rendezvous port is occupied')
audit=json.loads((root/'evidence/frontier293-state-comparison.json').read_text())
if not audit['complete'] or not all(all(v.values()) for v in audit['fidelity'].values()):
 raise ValueError('State observation fidelity failed')
jobs=[]
for family,card in [('hp',6)]:
 run='frontier293-normalization-precision'
 entry='replay_hp_frontier293_precision.py'
 tools='tools-f637' if family=='hp' else 'tools'
 assert not (root/'runs'/(run+'.log')).exists()
 launcher=root/('run-'+run+'.sh')
 cmd=('/state/jt-recipe-20261014/jt-f637-pr192-venv/bin/python -m torch.distributed.run --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=31679' if family=='hp' else 'python')+' /alignment/phase12/'+tools+'/'+entry
 script='set -euo pipefail\nsource /workspace/env.sh\n'
 script+='export ASCEND_RT_VISIBLE_DEVICES='+str(card)+'\nexport PYTHONPATH=/alignment/phase12/'+tools+':/workspace:${PYTHONPATH:-}\n'
 script+='export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CLOSE_MATMUL_K_SHIFT=0 P12_NS_EAGER_SCOPE=all\n'
 (state/('work-'+run)).mkdir(exist_ok=False)
 script+='cd /state/jt-recipe-20261014/work-'+run+'\nexec > /alignment/phase12/runs/'+run+'.log 2>&1\n'
 script+="trap 'code=$?; printf \"%s\\n\" \"$code\" > /alignment/phase12/runs/"+run+".exit' EXIT\n"
 script+=cmd+'\n'
 launcher.write_text(script)
 socket='hp-pr-two-commits-20261008' if family=='hp' else 'mf-image-verify-20261008'
 command=['docker','-H','unix:///run/'+socket+'/docker.sock','exec','jt-adamw-'+family+'-20261013','bash','/alignment/phase12/'+launcher.name]
 with (root/'runs'/(run+'.dispatch.log')).open('wb') as stream:
  process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 row={'run':run,'pid':process.pid,'card':card,'started_at':datetime.datetime.now().astimezone().isoformat(),'command':command,'source_sha256':hashlib.sha256((root/tools/entry).read_bytes()).hexdigest()}
 (root/'evidence'/(run+'-launch.json')).write_text(json.dumps(row,indent=2));jobs.append(row)
print(json.dumps(jobs,indent=2))
