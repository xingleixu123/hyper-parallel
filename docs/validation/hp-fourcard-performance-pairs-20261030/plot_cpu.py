"""Local CPU scientific preview. Run with ReportLab and pypdfium2; no site export."""
from pathlib import Path
import csv,statistics
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
import pypdfium2 as pdfium
O=Path(__file__).resolve().parent;D=O/'data';C=[HexColor(x) for x in ('#16628e','#c1473e','#6da8b9','#e3a456')]; dark=HexColor('#243747');grid=HexColor('#dbe3e8')
def read(n):return list(csv.DictReader((D/n).open(encoding='utf8',newline='')))
def text(c,x,y,s,size=9):c.setFillColor(dark);c.setFont('Helvetica',size);c.drawString(x,y,s)
def line(c,x1,y1,x2,y2,col=grid,width=.7):c.setStrokeColor(col);c.setLineWidth(width);c.line(x1,y1,x2,y2)
def series(c,x,y,w,h,title,data,fields,lo,hi,labels):
 text(c,x,y+h+24,title,12);text(c,x,y+h+10,'seconds / update (max rank)',8)
 for j in range(4):
  v=lo+(hi-lo)*j/3;yy=y+j*h/3;line(c,x,yy,x+w,yy);text(c,x-35,yy-3,f'{v:.2f}',8)
 for i,key in enumerate(fields):
  path=c.beginPath()
  for j,row in enumerate(data):
   xx=x+w*j/29;yy=y+h*(float(row[key])-lo)/(hi-lo)
   if j:path.lineTo(xx,yy)
   else:path.moveTo(xx,yy)
  c.setStrokeColor(C[i]);c.setLineWidth(1.3);c.drawPath(path)
  line(c,x+i*120,y-37,x+i*120+15,y-37,C[i],2);text(c,x+i*120+18,y-40,labels[i],8)
 text(c,x,y-17,'1511',8);text(c,x+w-22,y-17,'1540',8)
def bars(c,x,y,w,h,title,categories,values,maxv,labels):
 text(c,x,y+h+23,title,12)
 for j in range(4):
  v=maxv*j/3;yy=y+h*j/3;line(c,x,yy,x+w,yy);text(c,x-30,yy-3,f'{v:.0f}',8)
 group=w/len(categories);bw=min(22,group/4)
 for i,cat in enumerate(categories):
  for k,vals in enumerate(values):
   val=vals[i];c.setFillColor(C[k]);c.rect(x+i*group+k*bw+group*.25,y,bw-2,h*val/maxv,stroke=0,fill=1)
  text(c,x+i*group+group*.27,y-15,cat,8)
 for k,label in enumerate(labels):
  c.setFillColor(C[k]);c.rect(x+k*210,y-39,10,10,stroke=0,fill=1);text(c,x+k*210+15,y-38,label,8)
W,H=1100,900;c=canvas.Canvas(str(O/'performance-preview.pdf'),pagesize=(W,H));text(c,45,862,'129 performance evidence: short samples and separate traces',20);text(c,45,842,'CPU preview; not website export | H/F + SAC original source 1a | SAC fixed source 03ea | producer 33096 | cards 4-7',9)
series(c,65,618,440,145,'FSDP / HSDP two adjacent pairs',read('hf-steps.csv'),['H1_seconds','F1_seconds','H2_seconds','F2_seconds'],.72,1.50,['H1','F1','H2','F2'])
series(c,610,618,440,145,'SAC original / fixed two pairs',read('sac-steps.csv'),['original1','fixed1','original2','fixed2'],.78,.87,['orig 1','fixed 1','orig 2','fixed 2'])
hf=read('hf-steps.csv'); means={k:statistics.mean(float(r[k]) for r in hf) for k in ('H1_seconds','F1_seconds','H2_seconds','F2_seconds')}
text(c,65,557,f"Mean H/F: {means['H1_seconds']/means['F1_seconds']:.4f} (pair 1), {means['H2_seconds']/means['F2_seconds']:.4f} (pair 2)",9)
sac=read('sac-steps.csv'); sm={k:statistics.mean(float(r[k]) for r in sac) for k in ('original1','fixed1','original2','fixed2')}
text(c,610,557,f"Mean fixed - original: {sm['fixed1']-sm['original1']:+.6f}s (1), {sm['fixed2']-sm['original2']:+.6f}s (2)",9)
m=read('memory.csv');cats=['F1','H1','F2','H2'];peak=[max(float(r['peak_allocated_gib']) for r in m if r['run']==k) for k in cats];storage=[max(float(r['final_unique_storage_gib']) for r in m if r['run']==k) for k in cats];bars(c,65,316,440,165,'Allocator peak vs persistent unique storage (GiB)',cats,[peak,storage],25,['Peak allocated','Unique persistent storage'])
p=read('profile-steps.csv');cats=['FSDP','HSDP'];comput=[statistics.mean(float(r[k+'_Computing_us']) for r in p)/1000 for k in ('fsdp','hsdp')];comm=[statistics.mean(float(r[k+'_Communication(Not Overlapped)_us']) for r in p)/1000 for k in ('fsdp','hsdp')];bars(c,610,316,440,165,'Profiler: computing / non-overlapped comm (ms)',cats,[comput,comm],800,['Computing','Non-overlapped communication'])
text(c,45,226,'GBS4 / LBS1 / seq4096; 10 warmup + 30 timed. All points retained; two pairs, no CI.',10);text(c,45,207,'Profiler: 4 ranks x 3 active steps, separately instrumented runs; categories not causal and may overlap.',10);text(c,45,188,'Blue/red memory = peak allocated / final unique storage; reserved retained in CSV.',10);text(c,45,169,'Discrete occupancy samples do not prove continuous exclusivity; SAC r1 fixed supervisor failed after child exit 0.',10)
c.showPage();c.save();doc=pdfium.PdfDocument(str(O/'performance-preview.pdf'));page=doc[0];page.render(scale=1.5).to_pil().save(O/'performance-preview.png');page.close();doc.close();(O/'performance-preview.pdf').unlink()

