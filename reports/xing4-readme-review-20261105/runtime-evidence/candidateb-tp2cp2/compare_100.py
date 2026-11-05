"""Compare logged step metrics without asserting unverified numerical tolerances."""
import argparse,json,math,statistics
from pathlib import Path

def read(root):
 per=[]
 for rank in range(4):
  p=Path(root)/f'rank-{rank}'/'steps.json';rows=json.loads(p.read_text());d={x['step']:x for x in rows}
  if not set(range(6,106)).issubset(d):raise ValueError('missing required baseline or target steps')
  per.append(d)
 return per

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--baseline',required=True);ap.add_argument('--target',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 b=read(a.baseline);t=read(a.target);rows=[]
 for step in range(6,106):
  record={'step':step}
  for metric in ('foundation_loss','foundation_loss/mtp'):
   lhs=[float(x[step]['losses'][metric]) for x in b];rhs=[float(x[step]['losses'][metric]) for x in t]
   if not all(map(math.isfinite,lhs+rhs)):raise ValueError('nonfinite loss')
   record[metric]={'baseline_ranks':lhs,'target_ranks':rhs,'baseline_mean':statistics.mean(lhs),'target_mean':statistics.mean(rhs),'delta_of_rank_means':statistics.mean(rhs)-statistics.mean(lhs),'baseline_span':max(lhs)-min(lhs),'target_span':max(rhs)-min(rhs)}
  rows.append(record)
 out={'status':'descriptive_only','warning':'Rank-mean differences require confirmation of logging/reduction semantics; not an accuracy acceptance.', 'steps':rows}
 p=Path(a.output)
 with p.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
if __name__=='__main__':main()
