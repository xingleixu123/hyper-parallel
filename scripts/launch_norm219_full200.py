from pathlib import Path
import subprocess,json,datetime,hashlib,re,shutil
root=Path('/home/xxl/jt-adamw-align-20261013/phase12');state=Path('/dev/shm/jt-adamw-align-20261013/state/jt-recipe-20261014')
status=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True).stdout
assert all('No running processes found in NPU '+str(i) in status for i in [4,5,6])
assert shutil.disk_usage(state).free>5_000_000_000
mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
assert int(mem['MemAvailable'].split()[0])*1024>350_000_000_000
reference=[json.loads(s) for s in (root/'runs/hp-jtf637-packed217-both226/metrics.jsonl').read_text().splitlines()]
model_reference=[json.loads(s) for s in (root/'runs/mf-modelhash226/MODEL_HASHES.jsonl').read_text().splitlines()]
checks={}
for mode in ['dense','expert','both']:
 directory=root/'runs'/('hp-jtf637-norm219-'+mode+'226')
 rows=[json.loads(s) for s in (directory/'metrics.jsonl').read_text().splitlines()][:18]
 models=[json.loads(s) for s in (directory/'MODEL_HASHES.jsonl').read_text().splitlines()][:18]
 assert len(rows)>=5 and len(models)>=5
 clean=lambda r:{k:v for k,v in r.items() if k!='wall_time'}
 assert list(map(clean,rows))==list(map(clean,reference[:len(rows)])),mode
 assert models==model_reference[:len(models)],mode
 checks[mode]={'early_metrics_norm_exact_steps':len(rows),'early_all48_model_hashes_exact_steps':len(models)}
(root/'evidence/norm219-early-short-fidelity.json').write_text(json.dumps(checks,indent=2))

jobs=[]
for mode,card,port in [('dense',4,31604),('expert',5,31605),('both',6,31606)]:
 base='hp-jtf637-packed217-both400';run='hp-jtf637-norm219-'+mode+'400'
 assert not (root/'runs'/(run+'.log')).exists()
 config=root/'configs'/(run+'.yaml');config.write_text((root/'configs'/(base+'.yaml')).read_text().replace(base,run))
 script=(root/('run-'+base+'.sh')).read_text().replace(base,run).replace('capture_hp_packed217.py','capture_hp_norm219.py')
 script=re.sub(r'ASCEND_RT_VISIBLE_DEVICES=\d+', 'ASCEND_RT_VISIBLE_DEVICES='+str(card),script)
 script=re.sub(r'--master_port=\d+', '--master_port='+str(port),script)
 script=script.replace('export P12_PACKED_SEMANTICS_FACTOR=both','export P12_PACKED_SEMANTICS_FACTOR=both\nexport P12_NORM219_FACTOR='+mode+'\nexport P12_NORM219_MODEL_HASH=0')
 assert 'export P12_NORM219_FACTOR='+mode in script
 launcher=root/('run-'+run+'.sh');launcher.write_text(script);(state/('work-'+run)).mkdir()
 command=['docker','-H','unix:///run/hp-pr-two-commits-20261008/docker.sock','exec','jt-adamw-hp-20261013','bash','/alignment/phase12/'+launcher.name]
 with (root/'runs'/(run+'.dispatch.log')).open('wb') as stream:
  process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 record={'run':run,'pid':process.pid,'card':card,'port':port,'mode':mode,'started_at':datetime.datetime.now().astimezone().isoformat(),'command':command,'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),'launcher_sha256':hashlib.sha256(launcher.read_bytes()).hexdigest(),'factor_sha256':hashlib.sha256((root/'tools-f637/hp_norm219_factor.py').read_bytes()).hexdigest()}
 (root/'evidence'/(run+'-launch.json')).write_text(json.dumps(record,indent=2));jobs.append(record)
print(json.dumps(jobs,indent=2))
