from pathlib import Path
import csv,json,math
R=Path(__file__).resolve().parent
rows=list(csv.DictReader((R/'data/paired.csv').open(encoding='utf8')));s=json.loads((R/'data/stats.json').read_text())
assert len(rows)==1000 and [int(x['global_step']) for x in rows]==list(range(1501,2501))
d=[]
for x in rows:
 value=float(x['hp_loss'])-float(x['mf_loss']);assert value==float(x['hp_minus_mf']);d.append(value)
 assert float(x['cumulative_signed_mean'])==math.fsum(d)/len(d)
mean=math.fsum(d)/len(d)
assert mean==s['signed_mean_hp_minus_mf'] and abs(mean)==s['absolute_signed_mean']
assert math.fsum(abs(x) for x in d)/1000==s['mae_diagnostic'] and max(map(abs,d))==s['max_absolute_diagnostic']
assert [sum(x>0 for x in d),sum(x<0 for x in d),sum(x==0 for x in d)]==[s['positive'],s['negative'],s['zero']]
print(json.dumps({'count':len(d),'signed_mean':mean,'abs_mean':abs(mean),'threshold':.005,'passed':abs(mean)<=.005}))
