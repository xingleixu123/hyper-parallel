"""Recompute the public scalar comparison; Python 3 standard library only."""
import csv, hashlib, json, math
from pathlib import Path
P = Path(__file__).resolve().parent
PIN = {
    'paired-loss1000.csv': '6dc75d68e9efc38246c21ef1b3edcafc0e738dd63981183c08e97e3f96b2641d',
    'statistics.json': 'db4dd0e490e0ec03fa52e4b0f8c1a7310b38728b82c00f56cd2fd820785f5396',
    'independent-verification.json': 'd2188e9cbc872dd275a980bed4d564fbff4c14a2388b9cd741aa4ae6d3a24f3b',
}
for name, digest in PIN.items():
    if hashlib.sha256((P / name).read_bytes()).hexdigest() != digest:
        raise ValueError('source SHA mismatch: ' + name)
with (P / 'paired-loss1000.csv').open(encoding='utf-8', newline='') as f:
    rows = list(csv.DictReader(f))
if [int(x['step']) for x in rows] != list(range(1501, 2501)):
    raise ValueError('steps must be exactly 1501–2500')
values = []
for r in rows:
    hp, mf, delta = (float(r[k]) for k in ('hp_loss', 'mf_loss', 'delta_hp_minus_mf'))
    if not all(map(math.isfinite, (hp, mf, delta))) or hp - mf != delta:
        raise ValueError('non-finite or inconsistent row: ' + r['step'])
    values.append(delta)
def calc(values, first_step):
    maximum = max(range(len(values)), key=lambda i: abs(values[i]))
    mean = math.fsum(values) / len(values)
    return {'count': len(values), 'signed_mean': mean, 'abs_signed_mean': abs(mean),
            'mae': math.fsum(map(abs, values)) / len(values), 'max_abs': abs(values[maximum]),
            'max_step': first_step + maximum, 'positive': sum(x > 0 for x in values),
            'negative': sum(x < 0 for x in values), 'zero': sum(x == 0 for x in values)}
full = calc(values, 1501)
segments = [calc(values[i:i+100], 1501+i) for i in range(0, 1000, 100)]
reference = json.loads((P / 'statistics.json').read_bytes())
independent = json.loads((P / 'independent-verification.json').read_bytes())
for key, value in full.items():
    if value != reference['full_1000'][key] or value != independent['loss'][key]:
        raise ValueError('full-window statistic differs: ' + key)
for i, segment in enumerate(segments):
    for key, value in segment.items():
        if value != reference['segments_100'][i][key] or value != independent['segments100'][i][key]:
            raise ValueError('segment statistic differs: ' + str(i) + '/' + key)
if full['abs_signed_mean'] > 0.005:
    raise ValueError('full-window mean exceeds the specified threshold')
print(json.dumps({'status': 'passed', 'full_window': full, 'segments_100': segments,
                  'first_500_mae': calc(values[:500], 1501)['mae'],
                  'last_500_mae': calc(values[500:], 2001)['mae']}, indent=2))
