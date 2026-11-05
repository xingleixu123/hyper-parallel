import json,os,time,torch,torch_npu,pytest
from pathlib import Path
out=Path('/work/output');torch.npu.set_device(0);torch.empty(1,device='npu:0');torch.npu.synchronize();(out/'device-ready.json').write_text(json.dumps({'pid':os.getpid(),'tag':os.environ['HP_PR310_ID'],'logical_device':0}))
deadline=time.monotonic()+90
while not (out/'physical-mapping-release.json').is_file():
 if time.monotonic()>deadline:raise TimeoutError('physical admission')
 time.sleep(.2)
r=json.loads((out/'physical-mapping-release.json').read_text());assert r['tag']==os.environ['HP_PR310_ID'] and r['physical_card']==int(os.environ['PHYSICAL_CARD'])
os.environ['TEST_DEVICE']='npu:0';rc=pytest.main(['-q','-s','-p','no:cacheprovider','/work/scripts/hp-inventory/test_loader_npu.py']);(out/'probe-result.json').write_text(json.dumps({'pytest_returncode':int(rc),'torch':torch.__version__,'torch_npu':torch_npu.__version__,'logical_device':'npu:0'})+'\n');raise SystemExit(int(rc))
