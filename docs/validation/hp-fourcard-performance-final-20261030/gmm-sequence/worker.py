"""Candidate only: one-card sequence-normalization and SAC semantics; no training benchmark."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback
from contextlib import contextmanager

RUN_ID = 'hp-gmm-sequence-fix-129-v1'
ORIGINAL_SHA = 'ad46b219aed99282ea002c2b710b856f65be44208b42e575090250dafe84e637'
FIXED_SHA = '34ab7dd53763101d1f08e72e87a92195aed78a0fa5cb3dfb8b5610aa36a0b677'
OUT = Path('/work/run')
OVERLAY = Path('/work/overlay/policy.py')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def atomic_new(path, data):
    tmp = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    with tmp.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.link(tmp, path)
    tmp.unlink()

def write_json(name, value):
    atomic_new(OUT / name, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode())

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def main(state):
    require(os.environ.get('OUTPUT_DIR') == str(OUT) and OUT.is_dir(), 'OUTPUT_DIR must be existing /work/run')
    source = Path(os.environ['HP_SOURCE']).resolve(strict=True)
    cid = os.environ['HP_DIAG_CID']
    require(len(cid) == 64 and all(c in '0123456789abcdef' for c in cid), 'invalid full CID')
    require(bool(os.environ.get('ASCEND_RT_VISIBLE_DEVICES')), 'visible-device environment missing')
    state['visible_devices_env'] = os.environ['ASCEND_RT_VISIBLE_DEVICES']
    policy_file = source / 'hyper_parallel/distributed/activation_checkpoint.py'
    require(sha(policy_file.read_bytes()) == ORIGINAL_SHA, 'original policy SHA mismatch')
    require(sha(OVERLAY.read_bytes()) == FIXED_SHA, 'fixed overlay SHA mismatch')
    gmm_file = source / 'hyper_parallel/components/functional/grouped_matmul.py'
    gmm_sha = sha(gmm_file.read_bytes())
    require(gmm_sha == '9ac03602f50857c28b33812c44fb1ef2fd9b14ad2fa4369964069a24c66da0c8', 'patched grouped_matmul SHA mismatch')
    state['source_identity'] = {'base': '1a39572fef23819a2d2651b4bd53047a05a4892f manifest b18fef99 (host verified)', 'overlay_path': 'hyper_parallel/components/functional/grouped_matmul.py', 'overlay_sha256': gmm_sha}
    sys.path.insert(0, str(source))
    os.environ.setdefault('HYPER_PARALLEL_PLATFORM', 'torch')
    import numpy as np
    import torch
    import torch_npu
    from torch.utils.checkpoint import checkpoint
    from torch.utils._python_dispatch import TorchDispatchMode
    import hyper_parallel
    require(Path(hyper_parallel.__file__).resolve().is_relative_to(source), 'wrong HP import root')
    from hyper_parallel.components.functional import grouped_matmul
    import hyper_parallel.distributed.activation_checkpoint as original
    require(Path(original.__file__).resolve() == policy_file.resolve(), 'wrong original policy module')
    spec = importlib.util.spec_from_file_location('sac_fixed_overlay', OVERLAY)
    fixed = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixed
    spec.loader.exec_module(fixed)
    state['versions'] = {'torch': torch.__version__, 'torch_npu': getattr(torch_npu, '__version__', 'unknown')}
    state['device_count'] = torch.npu.device_count()
    require(state['device_count'] == 1, 'expected exactly one visible NPU')
    torch.npu.set_device(0)
    require(int(torch.npu.current_device()) == 0, 'logical device is not 0')
    probe = torch.empty(1, dtype=torch.float32, device='npu:0')
    torch.npu.synchronize()
    del probe
    stat = Path('/proc/self/stat').read_text()
    ready = {'run_id': RUN_ID, 'pid': os.getpid(), 'starttick': int(stat[stat.rfind(')')+2:].split()[19]),
             'pid_namespace': os.readlink('/proc/self/ns/pid'), 'worker_sha256': sha(Path(__file__).read_bytes()),
             'expected_physical_card': 4, 'logical_device': int(torch.npu.current_device()),
             'device_name': str(torch.npu.get_device_name()), 'cid': cid}
    write_json('device-ready.json', ready)
    state['ready'] = ready
    deadline = time.monotonic()+180
    release_path = OUT / 'device-release.json'
    while not release_path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError('device release timed out')
        time.sleep(.2)
    raw = release_path.read_bytes()
    release = json.loads(raw)
    require(release.get('kind') == 'root-verified-physical-device-map' and release.get('run_id') == RUN_ID
            and release.get('cid') == cid and release.get('worker_sha256') == ready['worker_sha256']
            and type(release.get('physical_card')) is int and release['physical_card'] == 4, 'invalid device release')
    state['device_release_sha256'] = sha(raw)
    # Everything below this point may execute module kernels.
    class Trace(TorchDispatchMode):
        def __init__(self):
            super().__init__(); self.phase='outside'; self.events=[]
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if str(func).startswith('npu.npu_grouped_matmul'):
                self.events.append({'phase':self.phase, 'op':str(func)})
            return func(*args, **(kwargs or {}))
        @contextmanager
        def phase_context(self, phase, inner=None):
            prev=self.phase; self.phase=phase
            try:
                if inner is None:
                    yield
                else:
                    with inner:
                        yield
            finally:
                self.phase=prev
        def context_fn(self, factory):
            def wrapped():
                forward, recompute = factory()
                return self.phase_context('forward',forward), self.phase_context('recompute',recompute)
            return wrapped
    def tensor_hash(t):
        return sha(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    def save_tensor(name, t):
        finite=bool(torch.isfinite(t).all())
        value=t.detach().cpu().float().contiguous().numpy()
        buf=io.BytesIO(); np.save(buf, value, allow_pickle=False)
        data=buf.getvalue(); atomic_new(OUT / name, data)
        return {'file':name, 'file_sha256':sha(data), 'original_dtype':str(t.dtype), 'stored_dtype':'float32',
                'shape':list(t.shape), 'original_bytes_sha256':tensor_hash(t), 'finite':finite}
    generator=torch.Generator(device='cpu').manual_seed(72184)
    def make(shape, scale=1.0):
        return torch.randn(shape, generator=generator, dtype=torch.bfloat16) * scale
    inputs={'x':make((128,64)), 'w1':make((2,64,128),.02), 'w2':make((2,128,64),.02), 'grad':make((128,64),.01)}
    hashes={name:tensor_hash(t) for name,t in inputs.items()}
    expected_hashes = {'grad':'d0a038211b3ff29b6ac5d29db7abb187664789e05fdb3ac158761e7baee86c53', 'w1':'c71b0a1995eaa5edc573cf44d0844e5c37086d76af666eb78ce5e9daca45fe38', 'w2':'245d09c1ba4df9b05602c61db8d4a636f902094fd19a1c70f206d56aa76f6fac', 'x':'b4a80c4a16a606b2c58ace47d2bb6f71ecdb18f880c442fc9dc448fc89191068'}
    state['input_hashes']=hashes
    require(hashes == expected_hashes, 'input bytes differ from failed List probe')
    state['input_files']={name:save_tensor('input-'+name+'.npy', t) for name,t in inputs.items()}
    state['runs']={}
    values={}
    for name, policy, kind in [('eager-1',None,'list'),('eager-2',None,'list'),('tuple-eager-1',None,'tuple'),('tuple-eager-2',None,'tuple'),('original-1',original,'list'),('original-2',original,'list'),('fixed-1',fixed,'list'),('fixed-2',fixed,'list')]:
        state['current_run']=name
        v={k:t.to('npu:0').detach().clone().requires_grad_(k != 'grad') for k,t in inputs.items()}
        for k,t in v.items():
            require(tensor_hash(t)==hashes[k], 'cloned input hash mismatch '+k)
        ends=[32,128] if kind == 'list' else (32,128)
        state['current_group_list_kind'] = type(ends).__name__
        def fn(x,w1,w2):
            z=grouped_matmul(x,w1,bias=None,group_list=ends,group_type=0,group_list_type=0,gemm_fusion=False)
            return grouped_matmul(torch.nn.functional.silu(z),w2,bias=None,group_list=ends,group_type=0,group_list_type=0,gemm_fusion=False)
        trace=Trace()
        state['current_events'] = trace.events
        with trace:
            if policy is None:
                with trace.phase_context('forward'):
                    output=fn(v['x'],v['w1'],v['w2'])
            else:
                factory=policy.make_selective_checkpoint_context_fn(allow_cache_entry_mutation=False)
                output=checkpoint(fn,v['x'],v['w1'],v['w2'],use_reentrant=False,context_fn=trace.context_fn(factory))
            with trace.phase_context('backward'):
                torch.autograd.backward(output,v['grad'])
            torch.npu.synchronize()
        require(tuple(ends) == (32,128) and type(ends) is (list if kind == 'list' else tuple) and trace.phase == 'outside', 'group list or phase mutated')
        tensors={'output':output.detach()}
        for k in ('x','w1','w2'):
            require(v[k].grad is not None, 'missing grad '+k)
            tensors[k+'_grad']=v[k].grad.detach()
        saved={k:save_tensor(name+'-'+k+'.npy', t) for k,t in tensors.items()}
        values[name]={k:t.cpu().clone() for k,t in tensors.items()}
        counts={}
        for e in trace.events:
            key=e['phase']+'|'+e['op']; counts[key]=counts.get(key,0)+1
        record={'group_list_kind':kind, 'group_list_unchanged':True, 'input_hashes':hashes,'clones_verified':True,'events':trace.events,'counts':counts,'tensors':saved,'cache_entry_mutation_allowed':False if policy else None}
        state['runs'][name]=record
        write_json(name+'.json', record)
        require(counts.get('forward|npu.npu_grouped_matmul.default',0)>0, 'default forward absent')
        require(counts.get('backward|npu.npu_grouped_matmul.default',0)>0, 'default backward absent')
        require(not any(e['op']=='npu.npu_grouped_matmul.List' for e in trace.events), 'unexpected List overload dispatch')
        del v,output,tensors
    comparison={}
    ref=values['eager-1']
    for name, result in values.items():
        comparison[name]={}
        for k,t in result.items():
            a=t.float(); b=ref[k].float()
            finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            comparison[name][k]={'bitwise_equal':tensor_hash(t)==tensor_hash(ref[k]), 'max_abs':float((a-b).abs().max()) if finite else None,
                                  'finite':finite}
    stable=all(x['bitwise_equal'] for x in comparison['eager-2'].values())
    all_equal=all(x['bitwise_equal'] for n in ('tuple-eager-1','tuple-eager-2','original-1','original-2','fixed-1','fixed-2') for x in comparison[n].values())
    state['comparison']=comparison
    all_finite=all(x['finite'] for r in comparison.values() for x in r.values())
    state['numerical_conclusion']='bitwise_equal_in_this_fixture' if stable and all_equal and all_finite else 'inconclusive'
    state['status']='complete'
    state.pop('current_run',None)
    state.pop('current_events',None)
    state.pop('current_group_list_kind',None)
    write_json('result.json',state)

if __name__ == '__main__':
    state={'run_id':RUN_ID,'status':'error','scope':'BF16 sequence-normalized toy module; default overload SAC dispatch, not List overload or kernel counts'}
    try:
        main(state)
    except BaseException:
        state['traceback']=traceback.format_exc()
        if OUT.is_dir():
            write_json('error.json',state)
        raise



