from pathlib import Path
import subprocess,json,hashlib,datetime,re,shutil
root=Path('/home/xxl/jt-adamw-align-20261013/phase12'); state=Path('/dev/shm/jt-adamw-align-20261013/state/jt-recipe-20261014')
status=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True).stdout
assert 'No running processes found in NPU 1' in status
assert shutil.disk_usage(state).free>8_000_000_000
base='mf-frontier219-state';run='mf-frontier219-ns-train'
assert not (root/'runs'/(run+'.log')).exists()
config=root/'configs'/(run+'.yaml'); config.write_text((root/'configs'/(base+'.yaml')).read_text().replace(base,run))
s=(root/('run-'+base+'.sh')).read_text().replace(base,run).replace('--master_port=31596','--master_port=31598').replace('capture_mf_frontier219_state.py','capture_mf_frontier219_ns.py')
s=re.sub(r' --gradient-output \S+', '', s)
launcher=root/('run-'+run+'.sh');launcher.write_text(s)
(state/('work-'+run)).mkdir()
command=['docker','-H','unix:///run/mf-image-verify-20261008/docker.sock','exec','jt-adamw-mf-20261013','bash','/alignment/phase12/'+launcher.name]
with (root/'runs'/(run+'.dispatch.log')).open('wb') as stream:
 process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
record={'run':run,'pid':process.pid,'card':1,'port':31598,'started_at':datetime.datetime.now().astimezone().isoformat(),'command':command,'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),'launcher_sha256':hashlib.sha256(launcher.read_bytes()).hexdigest()}
(root/'evidence'/(run+'-launch.json')).write_text(json.dumps(record,indent=2));print(json.dumps(record))
