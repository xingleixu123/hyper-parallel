"""Compute standard DTensor rank-local contiguous-byte digest from global tensor.

Conservative: handles Replicate and ordinary Shard including nested same-dim
chunking in mesh-axis order; rejects partial, undocumented plain and custom layouts.
"""
import hashlib,math,re

def _placements(s):
    if not isinstance(s,str):raise ValueError('placement must be native repr')
    pattern=r'(?:StridedShard\(dim=-?\d+, split_factor=\d+(?:, uneven_shard=True)?\)|Shard\(dim=-?\d+(?:, uneven_shard=True)?\)|Replicate\(\))'
    pieces=re.findall(pattern,s)
    if not pieces or '('+(', '.join(pieces))+(',' if len(pieces)==1 else '')+')'!=s:
        raise ValueError('unsupported/unparsed placement '+s)
    return [None if x=='Replicate()' else (int(re.search(r'dim=(-?\d+)',x).group(1)), x.startswith('StridedShard'), int(re.search(r'split_factor=(\d+)',x).group(1)) if x.startswith('StridedShard') else 1, 'uneven_shard=True' in x) for x in pieces]

def expected_local(t,record,rank,*,plain_full=False):
    if list(t.shape)!=record['global_shape'] or str(t.dtype)!=record['dtype']:raise ValueError('global shape or dtype mismatch')
    m=record['mesh']; slices=[[0,int(n)] for n in t.shape]
    if m is None:
        if not plain_full:raise ValueError('plain tensor requires reviewed full-replica declaration')
    else:
        shape=m['shape'];coord=m['coordinate'];ranks=m['ranks'];places=_placements(record['placements'])
        if len(shape)!=len(coord) or len(shape)!=len(places) or math.prod(shape)!=len(ranks):raise ValueError('mesh dimension mismatch')
        linear=0
        for width,c in zip(shape,coord):
            if not 0<=c<width:raise ValueError('bad coordinate')
            linear=linear*width+c
        if ranks[linear]!=rank or len(set(ranks))!=len(ranks):raise ValueError('rank mapping mismatch')
        for axis,(width,c,placement) in enumerate(zip(shape,coord,places)):
            if placement is None:continue
            dim,strided,split,uneven=placement
            if strided:
                if width != 1:raise ValueError('strided shard with non-unit mesh axis is unsupported')
                right=math.prod(shape[i] for i,p in enumerate(places) if i>axis and p is not None and p[0]==dim)
                if right!=split:raise ValueError('strided split factor mismatch')
            if dim<0:dim+=len(slices)
            if dim<0 or dim>=len(slices):raise ValueError('invalid shard dim')
            lo,hi=slices[dim];size=hi-lo
            if uneven:
                chunk=(size+width-1)//width;start=min(lo+c*chunk,hi);end=min(start+chunk,hi)
            else:
                chunk,remainder=divmod(size,width);start=lo+c*chunk+min(c,remainder);end=start+chunk+int(c<remainder)
            slices[dim]=[start,end]
    result=t[tuple(slice(a,b) for a,b in slices)]
    if list(result.shape)!=record['local_shape']:raise ValueError('local shape mismatch')
    data=result.detach().contiguous().reshape(-1).view(__import__('torch').uint8).cpu().numpy()
    h=hashlib.sha256(memoryview(data).cast('B')).hexdigest()
    if h!=record['sha256']:raise ValueError('local hash mismatch rank '+str(rank))
    return {'rank':rank,'slices':slices,'sha256':h}
