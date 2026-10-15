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

from pathlib import Path
import subprocess,json,datetime,hashlib,re,shutil
root=Path('/home/xxl/jt-adamw-align-20261013/phase12');state=Path('/dev/shm/jt-adamw-align-20261013/state/jt-recipe-20261014')
status=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True).stdout
assert all('No running processes found in NPU '+str(i) in status for i in [2,3])
assert shutil.disk_usage(state).free>5_000_000_000
mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
assert int(mem['MemAvailable'].split()[0])*1024>350_000_000_000
checks={}
for mode in ['both','expert']:
 directory=root/'runs'/('hp-jtf637-qsqrt222-'+mode+'226')
 baseline=root/'runs'/('hp-jtf637-norm219-'+mode+'226')
 rows=[json.loads(v) for v in (directory/'metrics.jsonl').read_text().splitlines()][:5]
 reference=[json.loads(v) for v in (baseline/'metrics.jsonl').read_text().splitlines()][:5]
 models=[json.loads(v) for v in (directory/'MODEL_HASHES.jsonl').read_text().splitlines()][:5]
 model_reference=[json.loads(v) for v in (baseline/'MODEL_HASHES.jsonl').read_text().splitlines()][:5]
 assert len(rows)==5 and len(models)==5
 clean=lambda r:{k:v for k,v in r.items() if k!='wall_time'}
 assert list(map(clean,rows))==list(map(clean,reference)),mode
 assert models==model_reference,mode
 factor=json.loads((directory/'Q_NPU_SQRT222_FACTOR.json').read_text())
 assert factor['source_sha256']==hashlib.sha256((root/'tools-f637/hp_qsqrt222_factor.py').read_bytes()).hexdigest()
 checks[mode]={'first5_metrics_norm_exact':True,'first5_all48_models_exact':True,'factor_source_exact':True}
(root/'evidence/qsqrt222-early-short-fidelity.json').write_text(json.dumps(checks,indent=2))

jobs=[]
for mode,card,port in [('both',2,31613),('expert',3,31614)]:
 base='hp-jtf637-norm219-'+mode+'400';run='hp-jtf637-qsqrt222-'+mode+'400'
 assert not (root/'runs'/(run+'.log')).exists()
 config=root/'configs'/(run+'.yaml');config.write_text((root/'configs'/(base+'.yaml')).read_text().replace(base,run))
 script=(root/('run-'+base+'.sh')).read_text().replace(base,run).replace('capture_hp_norm219.py','capture_hp_qsqrt222.py')
 script=re.sub(r'ASCEND_RT_VISIBLE_DEVICES=\d+', 'ASCEND_RT_VISIBLE_DEVICES='+str(card),script)
 script=re.sub(r'--master_port=\d+', '--master_port='+str(port),script)
 assert 'export P12_NORM219_FACTOR='+mode in script
 launcher=root/('run-'+run+'.sh');launcher.write_text(script);(state/('work-'+run)).mkdir()
 command=['docker','-H','unix:///run/hp-pr-two-commits-20261008/docker.sock','exec','jt-adamw-hp-20261013','bash','/alignment/phase12/'+launcher.name]
 with (root/'runs'/(run+'.dispatch.log')).open('wb') as stream:
  process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 record={'run':run,'pid':process.pid,'card':card,'port':port,'mode':mode,'started_at':datetime.datetime.now().astimezone().isoformat(),'command':command,'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),'launcher_sha256':hashlib.sha256(launcher.read_bytes()).hexdigest(),'factor_sha256':hashlib.sha256((root/'tools-f637/hp_qsqrt222_factor.py').read_bytes()).hexdigest()}
 (root/'evidence'/(run+'-launch.json')).write_text(json.dumps(record,indent=2));jobs.append(record)
print(json.dumps(jobs,indent=2))
