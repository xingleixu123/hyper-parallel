"""Pure CPU geometry and role validation for a source-only region cache."""
import hashlib

def require(ok, message):
    if not ok:
        raise ValueError(message)

def regions(shape):
    require(all(type(n) is int and n >= 0 for n in shape), 'invalid shape')
    full = ((0,) * len(shape), tuple(shape))
    if not shape:
        return [('full', *full)]
    width = (shape[0] + 1) // 2
    result = [('full', *full)]
    for rank in range(2):
        start = min(rank * width, shape[0])
        extent = min(width, shape[0] - start)
        result.append(('candidate_shard_dim0_degree2', (start,) + (0,) * (len(shape)-1),
                       (extent,) + tuple(shape[1:])))
    return result

def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()

def verify_digest(data, expected):
    require(digest_bytes(data) == expected, 'source region hash mismatch')

def alias_roles(section, name, proofs):
    if section != 'model_local_states' or name not in proofs[0]['model_local_parameters']:
        return []
    for rank in range(4):
        require(proofs[rank]['model_local_states'][name] == proofs[rank]['model_local_parameters'][name],
                'model state/parameter alias differs')
    return [('model_local_parameters', name)]

def classify_metadata(metadata, tensor_type, bytes_type):
    """Reject every unsupported entry before partitioning the full metadata map."""
    require(all(isinstance(value, (tensor_type, bytes_type)) for value in metadata.values()),
            'unsupported checkpoint metadata type')
    return ({key: value for key, value in metadata.items() if isinstance(value, tensor_type)},
            {key: value for key, value in metadata.items() if isinstance(value, bytes_type)})

def preflight_metadata(tensors, mapping, proofs, audit, describe, max_tensor_bytes):
    """Inspect metadata and source proof geometry; never load checkpoint payloads."""
    items=[]; allroles=[]; total=0; largest=0
    for key,item in sorted(tensors.items()):
        shape,dtype,size=describe(item)
        shape=tuple(shape)
        require(type(size) is int and size>=0 and size<=max_tensor_bytes,'max tensor bytes: '+key)
        total+=size;largest=max(largest,size);roles=[];sources=[]
        for row in (x for x in mapping if x['dcp_key']==key):
            section,name=row['section'],row['name'];records={r:proofs[r][section][name] for r in range(4)}
            audit.partition(records,section,name)
            for rank,record in records.items():
                gs,dt,offset,extent=audit.region(record,rank,section,name)
                require(gs==shape and dt==dtype,'metadata/source shape or dtype mismatch: '+section+':'+name)
                sources.append({'section':section,'name':name,'rank':rank,'offset':offset,'shape':extent,
                                'source_live_sha256':record['local_sha256']})
            roles.append({'section':section,'name':name}); allroles.append((section,name))
            for s,n in alias_roles(section,name,proofs):
                roles.append({'section':s,'name':n,'alias_of':{'section':section,'name':name}});allroles.append((s,n))
        items.append({'dcp_key':key,'global_shape':shape,'dtype':dtype,'logical_tensor_bytes':size,
                      'roles':roles,'source_regions':sources})
    require(len(allroles)==258 and len(set(allroles))==258, 'preflight role coverage')
    return {'tensors':items,'logical_tensor_bytes':total,'max_single_tensor_bytes':largest,'roles':len(allroles)}
