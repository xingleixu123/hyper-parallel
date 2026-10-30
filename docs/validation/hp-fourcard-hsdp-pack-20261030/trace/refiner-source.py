"""Refine captured flow evidence without rereading or rerunning device traces."""
from collections import Counter, defaultdict
from decimal import Decimal as D
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
SRC=HERE/'trace-parser-reviewed-v1'
OUT=HERE/'trace-refined-v1'
PINS={0:'f8d5e7f38ee8752bf8a3c90e39804f64a6072b16b4d356fe1ebfa0eea005e907',
      2:'0f7c982ce08dcb3b6728af226dfa6f7539b8deedc230643ece9f40e089f024af'}


def digest(raw):return hashlib.sha256(raw).hexdigest()
def endpoint(e):return (e['pid'],e['tid'],D(e['ts']))
def direct(link):
    return link['start_count']==link['finish_count']==link['endpoint_count']==1 and link.get('device') is not None
def label_values(name):return dict(p.split('=',1) for p in name.split('|')[1:])


def refine(row,rank):
    out={'scope':row['scope'],'pack':row.get('pack'),'parent_status':row['status'],
         'status':'unknown','evidence':[]}
    if row.get('sidecar_pack_count')!=1 or row.get('sidecar_collective_count')!=1:return out
    p=row['sidecar_pack'];c=row['sidecar_collective'];name=row['scope']['name'];links=row['flow_links']
    assert p['pack']==c['pack']==row['pack'] and p['step'] in (1512,1513,1514)
    assert rank in c['pg'] and c['pg'] in p['pgs']
    if not links or not all(direct(x) for x in links):return out
    if len({endpoint(x['device']) for x in links})!=len(links):return out
    if name.startswith('hp_comm|'):
        lab=label_values(name)
        assert name==c['comm_label']+'|pack='+p['pack']
        assert lab['pg']==','.join(map(str,c['pg'])) and int(lab['src'])==c['src']
        assert lab['dtype']==p['dtype']=='torch.float32' and int(lab['input_numel'])==p['total']
        assert lab['async']=='true' and c['async_op'] is p['async_op'] is True
        targets=[x for x in links if x['device']['name'].startswith('hcom_')]
        if len(targets)!=1:return out
        chosen=targets[0];dev=chosen['device'];csv=chosen.get('csv_row')
        if not (dev['name'].startswith('hcom_broadcast') and chosen['label_container_count']==1 and
                chosen['csv_match_count']==1 and csv and csv['Accelerator Core']=='COMMUNICATION'):return out
        assert dev['args']['count']==p['total'] and dev['args']['data_type']=='FP32'
        assert csv['Name']==dev['name'] and D(csv['Start Time(us)'].strip())==D(dev['ts'])
        assert int(csv['Step Id'])==p['step']-1511
        assert all(x is chosen or x['device']['name'] in ('EVENT RECORD','EVENT WAIT') for x in links)
        out.update(status='assigned_unique_broadcast_flow_and_csv',evidence=[chosen],
                   ancillary_event_flows=[x for x in links if x is not chosen],members=p['entries'],
                   step=p['step'],logical_buffer_bytes=p['total']*4,pg=c['pg'],src=c['src'])
    elif name.startswith('hp_pack_wait|'):
        if len(links)==1 and links[0]['device']['name']=='EVENT WAIT':
            out.update(status='assigned_unique_wait_flow_trace_only',evidence=links,
                       step=p['step'],csv_coverage='not present in kernel_details.csv')
    elif name.startswith('hp_pack_unpack|'):
        copies=[x for x in links if 'InplaceCopy' in x['device']['name'] and x['csv_match_count']==1]
        if len(copies)==len(p['entries']) and len(copies)==len(links):
            out.update(status='assigned_unique_unpack_flows_and_csv',evidence=copies,step=p['step'],
                       member_count=len(p['entries']),member_to_copy='Not assigned individually; enclosing actual pack consumer loop is observed.')
    return out


def union_length(intervals):
    ordered=sorted(intervals);total=D(0);end=None
    for a,b in ordered:
        assert b>=a
        if end is None or a>=end:total+=b-a
        elif b>end:total+=b-end
        end=b if end is None else max(end,b)
    return total


def main():
    OUT.mkdir(exist_ok=False)
    all_rows=[];sources=[];status={};waits=[]
    for rank,pin in PINS.items():
        raw=(SRC/f'rank-{rank}.json').read_bytes();assert digest(raw)==pin
        v=json.loads(raw);assert v['rank']==rank
        rows=[refine(x,rank) for x in v['scope_results']]
        # Real regression controls: extra unrelated task flows are not a second
        # broadcast. A second broadcast target or duplicated endpoint is rejected.
        first=next(x for x in v['scope_results'] if x['scope']['name'].startswith('hp_comm|'))
        assert refine(first,rank)['status']=='assigned_unique_broadcast_flow_and_csv'
        bad=json.loads(json.dumps(first));bad['flow_links'].append(bad['flow_links'][0])
        assert refine(bad,rank)['status']=='unknown'
        assert sum(x['status']=='assigned_unique_broadcast_flow_and_csv' for x in rows)==12
        status[str(rank)]=dict(Counter(x['status'] for x in rows))
        by_step=defaultdict(list)
        for row in rows:
            if row['status']=='assigned_unique_wait_flow_trace_only':by_step[row['step']].append(row)
        for step,group in sorted(by_step.items()):
            by_stream=defaultdict(list)
            for row in group:
                e=row['evidence'][0]['device'];a=D(e['ts']);by_stream[(e['pid'],e['tid'])].append((a,a+D(e['dur'])))
            waits.append({'rank':rank,'step':step,'pack_waits':len(group),
                          'union_us_by_device_stream':{str(k):str(union_length(x)) for k,x in by_stream.items()},
                          'scope':'Trace EVENT WAIT interval unions on each stream, from the original Work.wait sites; no step-level critical-path or cross-stream/cross-rank sum.'})
        all_rows.extend(dict(rank=rank,**x) for x in rows)
        sources.append({'rank':rank,'parent_sha256':pin,'raw_trace_and_csv':v['source']})
    result={'schema':'hp-pack-trace-refinement-v1','refiner_sha256':digest(Path(__file__).read_bytes()),
            'sources':sources,'statuses':status,'rows':all_rows,'device_wait_intervals':waits,
            'limits':['Broadcast source-call/unique-device/CSV/actual-pack membership association is proved for24 calls on ranks0/2, not all4 ranks device traces.',
                      'Work.wait has direct device EVENT WAIT flow evidence; CSV lacks these event rows. No Event ID edge to an EVENT RECORD is reconstructed.',
                      'Unpack scopes observe actual pack consumer loops; individual copy to FQN is not inferred from time, size or loop order.',
                      'Host annotation time is not device waiting time. Stream interval unions are not additive contributions to the HSDP/FSDP step difference.',
                      'Profiler run is diagnostic, not a new speed/memory ranking. No communication schedule or arithmetic was changed.']}
    with (OUT/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,ensure_ascii=False)
    with (OUT/'summary.json').open('x',encoding='utf8') as f:json.dump({'statuses':status,'waits':waits,'limits':result['limits']},f,indent=2,ensure_ascii=False)
    print(json.dumps({'statuses':status,'waits':waits},ensure_ascii=False))


if __name__=='__main__':main()
