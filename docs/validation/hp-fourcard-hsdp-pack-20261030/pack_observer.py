"""Per-rank, opt-in CPU metadata observer. Never retains tensor objects."""
import contextlib
import hashlib
import json
from pathlib import Path

_active = False
_step = None
_rank = None
_sequence = 0
_events = []
_tensors = {}
_buffers = {}
_record_function = None
_LIMIT = 100000

def begin_step(step, rank, record_function):
    global _active, _step, _rank, _record_function
    if _active or _buffers or _tensors: raise RuntimeError('observer state not clean')
    if _rank is not None and _rank != rank: raise RuntimeError('rank changed')
    if step not in range(1511, 1515): raise ValueError('step outside diagnostic range')
    _step, _rank, _record_function, _active = step, rank, record_function, True

def _add(event):
    if len(_events) >= _LIMIT: raise RuntimeError('observer event limit exceeded')
    event['step'] = _step
    _events.append(event)

def begin_flush():
    if _active:
        if _tensors: raise RuntimeError('nested flush')

def end_flush():
    if _active: _tensors.clear()

def _pgs(pgs, dist):
    return [None if pg is None else list(dist.get_process_group_ranks(pg)) for pg in pgs]

def record_assignment(assignment, group, group_idx, batch_idx, to_local):
    if not _active: return
    owned=[]
    for record in assignment.owned_records:
        name=getattr(record.param, 'model_name', None)
        if not isinstance(name, str) or not name: raise RuntimeError('missing owned FQN')
        local=to_local(record.param.data)
        owned.append(dict(fqn=name, record_index=record.index, local_shape=list(local.shape),
                          global_shape=list(record.param.shape)))
        del local
    opts={str(k): v if isinstance(v, (str, int, float, bool)) or v is None else str(v)
          for k, v in group.items() if str(k).startswith('ns_')}
    _add(dict(kind='assignment', assignment_id=id(assignment), group_idx=group_idx,
              batch_idx=batch_idx, is_shard=assignment.is_shard, owned=owned, ns_options=opts))

def register_tensor(tensor, record, assignment, owner, dist):
    if not _active: return
    ident=id(tensor)
    if ident in _tensors: raise RuntimeError('ambiguous tensor identity in flush')
    name=getattr(record.param,'model_name',None)
    if not isinstance(name,str) or not name: raise RuntimeError('missing FQN')
    if any(v['fqn']==name for v in _tensors.values()): raise RuntimeError('duplicate FQN in flush')
    if tuple(assignment.owner_by_index.get(record.index,())) != tuple(owner): raise RuntimeError('owner mapping mismatch')
    placements=getattr(record.param,'placements',None)
    _tensors[ident]={'tensor_id':ident,'fqn':name,'record_index':record.index,
                     'owner_coord':list(owner),'global_shape':list(record.param.shape),
                     'local_shape':list(tensor.shape), 'placements':None if placements is None else [str(p) for p in placements],
                     'assignment_id':id(assignment),'replicate_pgs':_pgs(assignment.replicate_pgs,dist),
                     'shard_pgs':_pgs(assignment.shard_pgs,dist)}

def register_pack(buffer, offsets, total, alignment, owner, local, pgs, dist, async_op, max_broadcast_bytes=None):
    global _sequence
    if not _active: return
    ident=id(buffer)
    if ident in _buffers: raise RuntimeError('buffer identity reused before release')
    entries=[]; end=0
    for tensor, offset, actual, padded in offsets:
        if id(tensor) not in _tensors: raise RuntimeError('unregistered tensor in actual offsets')
        if offset != end or actual != tensor.numel() or padded < actual or padded % alignment: raise RuntimeError('invalid actual offset layout')
        entry=dict(_tensors[id(tensor)]); entry.update(offset=offset,actual_numel=actual,padded_numel=padded)
        entries.append(entry); end=offset+padded
    if end != total or total != buffer.numel(): raise RuntimeError('pack coverage mismatch')
    pack='r%d-s%d-p%d'%(_rank,_step,_sequence); _sequence+=1
    _buffers[ident]=pack
    _add(dict(kind='pack',pack=pack,buffer_id=ident,entries=entries,total=total,alignment_elements=alignment,
              dtype=str(buffer.dtype),owner_coord=list(owner),local_coord=list(local),pgs=_pgs(pgs,dist),async_op=async_op, max_broadcast_bytes=max_broadcast_bytes))

def collective(tensor, op, ranks, src, async_op, comm_label):
    if not _active or tensor is None: return None
    pack=_buffers.get(id(tensor))
    if pack is not None:
        if op != 'broadcast': raise RuntimeError('pack buffer used by unexpected collective')
        _add(dict(kind='collective',pack=pack,buffer_id=id(tensor),pg=list(ranks),src=src,async_op=async_op,comm_label=comm_label))
    return pack

def wait_scope(buffer, kind):
    if not _active: return contextlib.nullcontext()
    pack=_buffers.get(id(buffer))
    if pack is None: return contextlib.nullcontext()  # unrelated non-replica paths
    return _record_function('hp_pack_wait|pack=%s|kind=%s'%(pack,kind))

def unpack_scope(buffer):
    if not _active: return contextlib.nullcontext()
    pack=_buffers.get(id(buffer))
    if pack is None: raise RuntimeError('unpack without registered pack')
    return _record_function('hp_pack_unpack|pack='+pack)


def release(buffer):
    if _active: _buffers.pop(id(buffer),None)

def end_step():
    global _active, _step
    if not _active: raise RuntimeError('observer not active')
    if _buffers or _tensors: raise RuntimeError('unfinished pack or flush at step end')
    _active=False
    _step=None

def flush_json(path):
    """Call only after profiler/timed region; emits one step then clears CPU metadata."""
    global _step, _rank, _record_function
    if _active or _rank is None: raise RuntimeError('end step first')
    raw=json.dumps({'schema':'hp-pack-observer-v1','rank':_rank,'events':_events},ensure_ascii=False,sort_keys=True).encode('utf-8')
    with Path(path).open('xb') as stream: stream.write(raw)
    digest=hashlib.sha256(raw).hexdigest()
    _events.clear(); _step=None; _rank=None; _record_function=None
    return digest
