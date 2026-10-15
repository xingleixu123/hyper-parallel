from pathlib import Path
import subprocess,json,datetime,hashlib,re
root=Path('/home/xxl/jt-adamw-align-20261013/phase12');status=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True).stdout
assert all('No running processes found in NPU '+str(i) in status for i in [0,1])
assert json.loads((root/'evidence/frontier222-state-comparison.json').read_text())['complete']
for variant,card,port in [('control',0,31609),('npu',1,31610)]:
 run='frontier222-q-'+variant+'-replay';assert not (root/'runs'/(run+'.log')).exists()
 script=(root/'run-frontier219-hp-muon-replay.sh').read_text().replace('frontier219-hp-muon-replay',run).replace('replay_hp_frontier219_muon.py','replay_hp_frontier222_qsqrt.py')
 script=script.replace('ASCEND_RT_VISIBLE_DEVICES=0','ASCEND_RT_VISIBLE_DEVICES='+str(card)+'\nexport P12_Q222_VARIANT='+variant).replace('31597',str(port))
 launcher=root/('run-'+run+'.sh');launcher.write_text(script)
 command=['docker','-H','unix:///run/hp-pr-two-commits-20261008/docker.sock','exec','jt-adamw-hp-20261013','bash','/alignment/phase12/'+launcher.name]
 with (root/'runs'/(run+'.dispatch.log')).open('wb') as stream:
  proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 record={'run':run,'pid':proc.pid,'card':card,'port':port,'started_at':datetime.datetime.now().astimezone().isoformat(),'source_sha256':hashlib.sha256((root/'tools-f637/replay_hp_frontier222_qsqrt.py').read_bytes()).hexdigest()}
 (root/'evidence'/(run+'-launch.json')).write_text(json.dumps(record,indent=2));print(json.dumps(record))
