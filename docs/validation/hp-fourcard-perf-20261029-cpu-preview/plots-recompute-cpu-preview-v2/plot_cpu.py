#!/usr/bin/env python3
"""Rebuild CPU previews from completed aggregate.csv inputs using ReportLab charts."""
import csv,hashlib,json,math,statistics
from pathlib import Path
from reportlab.graphics.shapes import Drawing, String, Rect, Line, Circle
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderPDF
from reportlab.lib import colors
import pypdfium2 as pdfium

O=Path(__file__).resolve().parent; ROOT=O.parent
TAGS=('fsdp4-full','fsdp4-selective','fsdp4-off')
LABEL={'fsdp4-full':'Full','fsdp4-selective':'Selective','fsdp4-off':'Off'}
COL={'fsdp4-full':colors.HexColor('#225A92'),'fsdp4-selective':colors.HexColor('#BA3C33'),'fsdp4-off':colors.HexColor('#278063')}
FIELDS=('Time_s','Tokens_per_s','Peak_allocated_GiB','Peak_reserved_GiB')
runs={}; source=[]
for tag in TAGS:
    f=ROOT/'measurement-v2'/tag/'run'/'aggregate'/'aggregate.csv'
    raw=f.read_bytes(); rows=list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
    assert len(rows)==30 and [int(x['step']) for x in rows]==list(range(1511,1541))
    for x in rows:
        for k in FIELDS: assert math.isfinite(float(x[k]))
    runs[tag]=rows
    source.append({'tag':tag,'path':str(f),'sha256':hashlib.sha256(raw).hexdigest(),'rows':len(rows)})
summary={'inputs':source,'method':'Each point is one completed run; max rank local time per step, nominal 16384 tokens/update; error bars show min–max of 30 steps within same run, no paired repeats or confidence interval.', 'runs':{}}
for tag in TAGS:
    x=[float(r['Tokens_per_s']) for r in runs[tag]]; t=[float(r['Time_s']) for r in runs[tag]]; y=[float(r['Peak_allocated_GiB']) for r in runs[tag]]; z=[float(r['Peak_reserved_GiB']) for r in runs[tag]]
    summary['runs'][tag]={'time_mean_s':statistics.fmean(t),'tokens_mean_step_s':statistics.fmean(x),'tokens_window_s':30*16384/math.fsum(t),'tokens_step_min':min(x),'tokens_step_max':max(x),'allocated_peak_max_gib':max(y),'allocated_step_min_gib':min(y),'allocated_step_max_gib':max(y),'reserved_peak_max_gib':max(z)}

def label(d,x,y,text,size=9,colour=colors.HexColor('#283747')):
    d.add(String(x,y,text,fontName='Helvetica',fontSize=size,fillColor=colour))
def header(d,title,sub):
    label(d,36,690,title,18,colors.HexColor('#152A3B'))
    label(d,36,670,sub,9)
    label(d,36,655,'Host 125  |  physical cards 4-7  |  source 1a39572f',9)
    label(d,36,642,'GBS 4  |  LBS 1  |  sequence 4096  |  10 warmup + 30 timed steps',9)
    for i,tag in enumerate(TAGS):
        x=39+i*190; d.add(Line(x,625,x+20,625,strokeColor=COL[tag],strokeWidth=2.5));label(d,x+26,622,LABEL[tag]+' (mean %.4f s)' % summary['runs'][tag]['time_mean_s'],9)

def linechart(d,x,y,w,h,field,title):
    label(d,x,y+h+15,title,11)
    chart=LinePlot(); chart.x=x;chart.y=y;chart.width=w;chart.height=h
    chart.data=[[(int(v['step']),float(v[field])) for v in runs[tag]] for tag in TAGS]
    chart.xValueAxis.valueMin=1511;chart.xValueAxis.valueMax=1540;chart.xValueAxis.valueStep=5
    vals=[float(v[field]) for tag in TAGS for v in runs[tag]]; low=min(vals); high=max(vals); pad=max((high-low)*.12,abs(high)*.002,1e-6)
    chart.yValueAxis.valueMin=max(0,low-pad);chart.yValueAxis.valueMax=high+pad
    chart.xValueAxis.labelTextFormat='%d'; chart.yValueAxis.labelTextFormat='%.2f' if field!='Tokens_per_s' else '%.0f'
    chart.xValueAxis.labels.fontSize=8;chart.yValueAxis.labels.fontSize=8
    chart.xValueAxis.visibleGrid=True;chart.yValueAxis.visibleGrid=True
    chart.xValueAxis.gridStrokeColor=colors.HexColor('#E3E8ED');chart.yValueAxis.gridStrokeColor=colors.HexColor('#E3E8ED')
    for i,tag in enumerate(TAGS): chart.lines[i].strokeColor=COL[tag];chart.lines[i].strokeWidth=1.5
    d.add(chart)

D=Drawing(1000,720)
header(D,'Recomputation: per-update observations','Single run per strategy; raw 30 points, no smoothing or downsampling. Step time = max rank local duration.')
linechart(D,90,407,365,180,'Time_s','Step time (s)')
linechart(D,585,407,365,180,'Tokens_per_s','Global nominal tokens / s')
linechart(D,90,140,365,180,'Peak_allocated_GiB','Max-rank peak allocated (GiB)')
linechart(D,585,140,365,180,'Peak_reserved_GiB','Max-rank peak reserved (GiB)')
label(D,430,105,'Original global step (1511-1540)',10)
label(D,36,52,'Allocator peak reset each step. Values are framework allocator metrics, not physical HBM.',9)
label(D,36,36,'Mean refers to arithmetic mean across these 30 max-rank step observations; not a repeated-run estimate.',9)
label(D,36,22,'CPU preview; website export pending.',9)
label(D,36,9,'Endpoint occupancy only; continuous exclusivity not recorded.',9)
renderPDF.drawToFile(D,str(O/'01-per-step.pdf'))

D=Drawing(1000,720)
header(D,'Recomputation: throughput vs allocator peak','One point per strategy: window throughput = 30 x 16384 / sum(max-rank step time).')
label(D,36,605,'Bars = min-max across the 30 steps within this run; not confidence intervals. No paired reruns yet.',10)
# Use ReportLab LinePlot for the scatter axes and the crossbars. X = throughput, Y = peak allocated.
chart=LinePlot();chart.x=100;chart.y=135;chart.width=760;chart.height=410
series=[]
for tag in TAGS:
    s=summary['runs'][tag]; xx=s['tokens_window_s']; yy=s['allocated_peak_max_gib']
    series.extend([[(s['tokens_step_min'],yy),(s['tokens_step_max'],yy)],[(xx,s['allocated_step_min_gib']),(xx,s['allocated_step_max_gib'])]])
chart.data=series
xs=[float(r['Tokens_per_s']) for tag in TAGS for r in runs[tag]];ys=[float(r['Peak_allocated_GiB']) for tag in TAGS for r in runs[tag]]
chart.xValueAxis.valueMin=min(xs)*.95;chart.xValueAxis.valueMax=max(xs)*1.05
chart.yValueAxis.valueMin=min(ys)*.96;chart.yValueAxis.valueMax=max(ys)*1.04
chart.xValueAxis.labelTextFormat='%.0f';chart.yValueAxis.labelTextFormat='%.2f';chart.xValueAxis.visibleGrid=True;chart.yValueAxis.visibleGrid=True
chart.xValueAxis.gridStrokeColor=colors.HexColor('#E3E8ED');chart.yValueAxis.gridStrokeColor=colors.HexColor('#E3E8ED')
for i,tag in enumerate(TAGS):
    for j in (2*i,2*i+1): chart.lines[j].strokeColor=COL[tag];chart.lines[j].strokeWidth=2
D.add(chart)
# Map the point positions with the same axis limits as LinePlot.
for tag in TAGS:
    s=summary['runs'][tag];xx=s['tokens_window_s'];yy=s['allocated_peak_max_gib']
    px=chart.x+chart.width*(xx-chart.xValueAxis.valueMin)/(chart.xValueAxis.valueMax-chart.xValueAxis.valueMin)
    py=chart.y+chart.height*(yy-chart.yValueAxis.valueMin)/(chart.yValueAxis.valueMax-chart.yValueAxis.valueMin)
    D.add(Circle(px,py,5,fillColor=COL[tag],strokeColor=colors.white));label(D,px+8,py+8,LABEL[tag],10,COL[tag])
label(D,360,92,'Window nominal tokens / s (higher is better)',11)
label(D,100,565,'Max of per-step max-rank peak allocated (GiB)',11)
label(D,36,48,'Bars show run-internal step ranges; overlapping ranges do not establish statistical significance.',9)
label(D,36,32,'Elapsed time is max rank local duration per update, not cross-rank wall-clock makespan.',9)
label(D,36,20,'CPU preview; website export pending.',9)
label(D,36,8,'Endpoint occupancy only; continuous exclusivity not recorded.',9)
renderPDF.drawToFile(D,str(O/'02-tradeoff.pdf'))
for stem in ('01-per-step','02-tradeoff'):
    doc=pdfium.PdfDocument(str(O/(stem+'.pdf'))); page=doc[0]; bitmap=page.render(scale=2); bitmap.to_pil().save(O/(stem+'.png')); bitmap.close();page.close();doc.close()
print(json.dumps(summary['runs'],indent=2))
