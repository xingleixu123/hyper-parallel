import concurrent.futures,subprocess,os,pathlib,json,hashlib,sys
p=pathlib.Path('/mnt/d/Obsidian Vault D/hyperllal/HP-MF-Master-SixPR-20261102/xing4-readme-followup-20261105');stage=pathlib.Path('/tmp/xing4-loader-regression-20261105'); python='/home/xxl/.cache/codex/hp-mf-master-sixpr-20261102-venv/bin/python'
env=os.environ.copy();env['PYTHONPATH']=str(stage);env['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
file=stage/'hyper_parallel/models/_transformers/checkpoint_loader.py';candidate=p/'checkpoint_loader-after.py';file.write_bytes(candidate.read_bytes())
assert hashlib.sha256(file.read_bytes()).hexdigest()==json.loads((p/'loader-candidate-binding.json').read_text())['after']
cases={'before':[python,str(p/'loader-regression-probe.py'),'--version','before'],'after':[python,str(p/'loader-regression-probe.py'),'--version','after'],'existing':[python,'-m','pytest','-q','tests/ut/components/checkpoint/test_huggingface_load_planner.py','tests/ut/components/checkpoint/test_hf_external_targets.py','tests/ut/auto_models/_transformers/test_deferred_pretrained.py']}
def run(item):
 name,cmd=item;r=subprocess.run(cmd,cwd=stage,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);output=r.stdout+f'\nexit_code={r.returncode}\n'.encode();(p/f'loader-regression-{name}.log').write_bytes(output);print(name,'exit',r.returncode,'bytes',len(output),flush=True);return name,r.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool: result=dict(pool.map(run,cases.items()))
(p/'loader-regression-exit.json').write_text(json.dumps(result,indent=2)+'\n');sys.exit(any(result.values()))
