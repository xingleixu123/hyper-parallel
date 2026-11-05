"""CPU-only four-card command/handshake candidate. Does not launch devices."""
import argparse,json,re
from pathlib import PurePosixPath
CARDS=(4,5,6,7)
IMAGE='sha256:5f9e677f439d11ab4e7c888f54b142335a27eee7411cbd2ce894a4798a70146d'

def physical_pids(raw):
    lines=[x.strip() for x in raw.splitlines() if x.strip()]
    if lines==['No process in device.']: return set()
    matches=[re.fullmatch(r'Process id:\s*(\d+)\s+Process name:\s*\S+\s+Process memory\(MB\):\s*\d+',x) for x in lines]
    if not matches or not all(matches): raise ValueError('unknown npu-smi format')
    pids=[int(m.group(1)) for m in matches]
    if len(pids)!=len(set(pids)): raise ValueError('duplicate process entries')
    return set(pids)

def require_bijection(rank_proof,observations):
    if len(rank_proof)!=4 or set(observations)!=set(CARDS): raise ValueError('incomplete ranks/cards')
    if [x['rank'] for x in rank_proof]!=list(range(4)): raise ValueError('rank ordering')
    pids=[x['host_pid'] for x in rank_proof]
    if len(set(pids))!=4: raise ValueError('duplicate rank pid')
    actual={card:physical_pids(observations[card]) for card in CARDS}
    for rank,card in enumerate(CARDS):
        if actual[card]!={pids[rank]}: raise ValueError('card is not exclusively owned by expected rank')
    return {'physical_by_rank':list(CARDS),'rank_proof':rank_proof}

def holder_argv(run_id, paths):
    if not re.fullmatch('[a-z0-9][a-z0-9_-]{0,63}',run_id): raise ValueError('unsafe run id')
    required={'source','env','python','cann','driver','vendor','pyext','data_bin','data_idx','cache','worker','output'}
    if set(paths)!=required: raise ValueError('incomplete mount map')
    if any(not PurePosixPath(v).is_absolute() or '..' in PurePosixPath(v).parts or ',' in v for v in paths.values()): raise ValueError('unsafe mount')
    dest={'source':'/work/sources/hyper-parallel','env':'/work/env','python':'/usr/local/python3.12.14','cann':'/work/runtime/cann920/cann-9.2.0','driver':'/usr/local/Ascend/driver','vendor':'/work/vendor-opp','pyext':'/work/pyext','data_bin':'/datasets/data.bin','data_idx':'/datasets/data.idx','cache':'/work/cache','worker':'/work/worker','output':'/work/output'}
    args=['docker','run','-d','--rm','--name','sixpr-fourcard-'+run_id,'--label','codex.task=sixpr-fourcard-'+run_id,'--read-only','--network','host','--cpus','16','--memory','192g','--tmpfs','/tmp:rw,nosuid,nodev,size=4g','--workdir','/work/output']
    for d in ['/dev/davinci'+str(c) for c in CARDS]+['/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']: args+=['--device',d+':'+d]
    for name in sorted(required): args+=['--mount','type=bind,src='+paths[name]+',dst='+dest[name]+('' if name=='output' else ',readonly')]
    return args+[IMAGE,'sleep','infinity']

def worker_argv(phase,config,master_port,hccl_start,run_id):
    if phase not in ('start5probe','formal2500','resume1000'): raise ValueError('unknown phase')
    if not 1024<=master_port<=65535 or not 1024<=hccl_start<=65520 or hccl_start<=master_port<hccl_start+16: raise ValueError('invalid/overlapping ports')
    if not config.startswith('/work/worker/') or '..' in PurePosixPath(config).parts: raise ValueError('config path')
    return ['/work/env/bin/python','-B','-m','torch.distributed.run','--nnodes=1','--node-rank=0','--nproc-per-node=4','--master-addr=127.0.0.1','--master-port='+str(master_port),'/work/worker/worker.py','--phase',phase,'--config',config,'--run-id',run_id,'--master-port',str(master_port),'--hccl-start',str(hccl_start),'--output','/work/output']

if __name__=='__main__':
    raise SystemExit('Candidate library only: ROOT worker, live guards and namespace runner not yet connected; no launch permitted.')
