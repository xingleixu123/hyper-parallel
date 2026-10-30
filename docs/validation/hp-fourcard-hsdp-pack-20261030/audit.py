import hashlib,json,math,re
from pathlib import Path
from collections import defaultdict
out=Path(__file__).resolve().parent; data=out
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def check(x,msg):
 if not x: raise ValueError(msg)
expected={0: '0259f75fae4b87010ac0cd530bb58eb4fd8c065251b375ad3d2ecb9d7376160d', 1: '02714f8043d25ee404f85a5b39555f142bdb2f7e6bf289ad2c0fd06bf1fa206a', 2: '6233c138081f46950dd45098372c109b3e6ebff56c60508b81eaa611a44fd94e', 3: 'b707cf456dc6a8b418e3b677577d04a357c2ce88e61b384329f8d17ed1a9840c'}
records=[]; assignments=[]; streams=defaultdict(list); evidence=[]
for rank in range(4):
 sp=data/f'rank-{rank}-pack-observer.json'; check(digest(sp)==expected[rank], 'sidecar SHA mismatch'); d=json.loads(sp.read_bytes()); check(d['rank']==rank and d['schema']=='hp-pack-observer-v1', 'rank/schema mismatch'); evidence.append({'rank':rank,'sidecar_sha256':digest(sp)})
 events=d['events']; check({x['step'] for x in events}==set(range(1511,1515)),'steps')
 ids=set(); comm_ids=set()
 for step in range(1511,1515):
  es=[e for e in events if e['step']==step]; packs={e['pack']:e for e in es if e['kind']=='pack'}; cs=[e for e in es if e['kind']=='collective']; aa=[e for e in es if e['kind']=='assignment']
  check(len(packs)==4 and len(cs)==4 and len(aa)==2 and set(e['pack'] for e in cs)==set(packs),'wrong event count')
  assignments.extend(dict(rank=rank,**a) for a in aa)
  for c in cs:
   p=packs[c['pack']]; pid=p['pack']; check(pid not in ids and re.fullmatch(rf'r{rank}-s{step}-p\d+',pid),'duplicate pack id'); ids.add(pid)
   check(p['dtype']=='torch.float32' and p['async_op'] is True and c['async_op'] is True and c['buffer_id']==p['buffer_id'],'dtype/async/buffer')
   cursor=0; names=set()
   for x in p['entries']:
    check(x['fqn'] not in names and x['offset']==cursor and x['actual_numel']==math.prod(x['local_shape']) and x['padded_numel']>=x['actual_numel'] and x['padded_numel']%p['alignment_elements']==0,'offset/shape'); names.add(x['fqn']); cursor+=x['padded_numel']
    check(x['owner_coord']==p['owner_coord'] and x['replicate_pgs']==p['pgs'],'owner/group mismatch')
   check(cursor==p['total'],'coverage'); pg=c['pg']; dims=[i for i,x in enumerate(p['pgs']) if x==pg]; check(len(dims)==1,'PG dimension'); i=dims[0]; owner=p['owner_coord'][i]; check(pg[owner]==c['src'] and pg[p['local_coord'][i]]==rank,'rank/source map')
   parts=c['comm_label'].split('|'); kv=dict(z.split('=',1) for z in parts[1:]); check(parts[0]=='hp_comm' and kv['op']=='broadcast' and kv['pg']==','.join(map(str,pg)) and kv['input_numel']==str(p['total']) and kv['dtype']==p['dtype'] and kv['async']=='true' and kv['src']==str(c['src']),'label mismatch')
   check(kv['id'] not in comm_ids,'comm label duplicate'); comm_ids.add(kv['id']);
   signature={'src':c['src'],'dtype':p['dtype'],'total':p['total'],'owner_coord':p['owner_coord'],'members':[{k:x[k] for k in ('fqn','record_index','offset','actual_numel','padded_numel','global_shape','local_shape')} for x in p['entries']]}
   streams[(step,tuple(pg),rank)].append((pid,signature)); records.append(dict(rank=rank,step=step,pack=pid,pg=pg,src=c['src'],total_numel=p['total'],bytes=p['total']*4,logical_bytes=sum(x['actual_numel'] for x in p['entries'])*4,member_count=len(p['entries']),members=signature['members'],label=c['comm_label'],cap=p['max_broadcast_bytes']))
checks=[]
for step,pg in sorted({(s,g) for s,g,_ in streams}):
 rows=[streams[(step,pg,r)] for r in pg]; check(len(rows)==2 and all(len(x)==4 for x in rows),'PG sequence missing')
 for n,(a,b) in enumerate(zip(*rows)):
  check(a[1]==b[1],f'PG sequence differs {step} {pg} {n}'); checks.append({'step':step,'pg':pg,'ordinal':n,'packs':[a[0],b[0]],'equal':True})
check(len(records)==64 and len(checks)==32,'counts')
result={'inputs':evidence,'verified_packs':64,'pairings':checks,'records':records,'assignments':assignments,'audit_script_sha256':digest(Path(__file__))}
(out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
