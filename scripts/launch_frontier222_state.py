from pathlib import Path
import subprocess,json,hashlib,datetime,re,shutil
root=Path('/home/xxl/jt-adamw-align-20261013/phase12');state=Path('/dev/shm/jt-adamw-align-20261013/state/jt-recipe-20261014')
status=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True).stdout
hp_card=next((i for i in [0,7] if 'No running processes found in NPU '+str(i) in status),None)
assert hp_card is not None and 'No running processes found in NPU 1' in status
assert shutil.disk_usage(state).free>4_000_000_000
jobs=[]
for family,card,port in [('hp',hp_card,31607),('mf',1,31608)]:
 base='hp-jtf637-norm219-both226' if family=='hp' else 'mf-frontier219-state'
 run='hp-jtf637-norm219-state222' if family=='hp' else 'mf-frontier222-state'
 assert not (root/'runs'/(run+'.log')).exists()
 config_text=(root/'configs'/(base+'.yaml')).read_text().replace(base,run)
 config_text=config_text.replace('train_iters: 226','train_iters: 222') if family=='hp' else config_text.replace('  stop_step: 19','  stop_step: 22')
 config=root/'configs'/(run+'.yaml');config.write_text(config_text)
 script=(root/('run-'+base+'.sh')).read_text().replace(base,run)
 script=re.sub(r'ASCEND_RT_VISIBLE_DEVICES=\d+', 'ASCEND_RT_VISIBLE_DEVICES='+str(card),script)
 script=re.sub(r'--master_port=\d+', '--master_port='+str(port),script)
 script=script.replace('capture_hp_norm219.py','capture_hp_frontier222_state.py').replace('capture_mf_frontier219_state.py','capture_mf_frontier222_state.py').replace('P12_NORM219_MODEL_HASH=1','P12_NORM219_MODEL_HASH=0')
 launcher=root/('run-'+run+'.sh');launcher.write_text(script);(state/('work-'+run)).mkdir()
 socket='hp-pr-two-commits-20261008' if family=='hp' else 'mf-image-verify-20261008'
 command=['docker','-H','unix:///run/'+socket+'/docker.sock','exec','jt-adamw-'+family+'-20261013','bash','/alignment/phase12/'+launcher.name]
 with (root/'runs'/(run+'.dispatch.log')).open('wb') as stream:
  process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 record={'run':run,'pid':process.pid,'card':card,'port':port,'started_at':datetime.datetime.now().astimezone().isoformat(),'command':command,'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),'launcher_sha256':hashlib.sha256(launcher.read_bytes()).hexdigest()}
 (root/'evidence'/(run+'-launch.json')).write_text(json.dumps(record,indent=2));jobs.append(record)
print(json.dumps(jobs,indent=2))
