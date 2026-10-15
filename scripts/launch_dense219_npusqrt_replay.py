from pathlib import Path
import subprocess,json,datetime,hashlib
root=Path('/home/xxl/jt-adamw-align-20261013/phase12');run='frontier219-dense-npusqrt-replay'
status=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True).stdout
assert 'No running processes found in NPU 2' in status
assert not (root/'runs'/(run+'.log')).exists()
script=(root/'run-frontier219-hp-muon-replay.sh').read_text().replace('frontier219-hp-muon-replay',run).replace('replay_hp_frontier219_muon.py','replay_hp_frontier219_dense_npusqrt.py').replace('31597','31600').replace('ASCEND_RT_VISIBLE_DEVICES=0','ASCEND_RT_VISIBLE_DEVICES=2')
launcher=root/('run-'+run+'.sh');launcher.write_text(script)
command=['docker','-H','unix:///run/hp-pr-two-commits-20261008/docker.sock','exec','jt-adamw-hp-20261013','bash','/alignment/phase12/'+launcher.name]
with (root/'runs'/(run+'.dispatch.log')).open('wb') as stream:
 process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
record={'run':run,'pid':process.pid,'card':2,'started_at':datetime.datetime.now().astimezone().isoformat(),'source_sha256':hashlib.sha256((root/'tools-f637/replay_hp_frontier219_dense_npusqrt.py').read_bytes()).hexdigest()}
(root/'evidence'/(run+'-launch.json')).write_text(json.dumps(record,indent=2));print(json.dumps(record))
