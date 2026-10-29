"""Rebuild previews offline from data/perf-steps.csv + perf-summary.json."""
from pathlib import Path
import json,csv,math
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
P=Path(__file__).resolve().parent;rows=list(csv.DictReader((P/'data/perf-steps.csv').open(encoding='utf8')));data=json.loads((P/'data/perf-summary.json').read_text(encoding='utf8'));runs={x['run']:x for x in data['runs']};memory=data['memory_by_rank']
if Path('C:/Windows/Fonts/msyh.ttc').exists():plt.rcParams['font.family']=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name()
plt.rcParams['axes.unicode_minus']=False
panels=[('125 RAF 反向配对｜1a / full',['raf-no','raf-full']),('129 HSDP/FSDP 两对｜1a / full',['hsdp-1','fsdp-1','hsdp-2','fsdp-2']),('129 EP2 与 FSDP 参考｜1a / full',['ep2-2','fsdp-1','fsdp-2']),('125 TP2 与 FSDP｜1a / full',['tp2','tp-fsdp']),('129 CP2 与 FSDP｜03ea / selective',['cp2-fixed','fsdp-fixed']),('125 重算策略｜1a / FSDP4',['recompute-full','recompute-off','recompute-selective'])]
for run in runs:
 seq=[x for x in rows if x['run']==run];assert [int(x['step']) for x in seq]==list(range(1511,1541));assert all(math.isclose(float(x['max_rank_seconds']),max(float(x[f'rank{r}_seconds']) for r in range(4)),rel_tol=0,abs_tol=1e-12) for x in seq)
(P/'figures').mkdir(exist_ok=True)
fig,axes=plt.subplots(3,2,figsize=(16,13),layout='constrained');fig.patch.set_facecolor('#f7f8fa')
for ax,(title,ids) in zip(axes.flat,panels):
 for name in ids:
  seq=[x for x in rows if x['run']==name];ax.plot([int(x['step']) for x in seq],[float(x['max_rank_seconds']) for x in seq],lw=1.2,label=runs[name]['label'])
 ax.set_title(title,loc='left',fontweight='bold',fontsize=12);ax.set_xlabel('step');ax.set_ylabel('秒 / 更新');ax.grid(alpha=.22);ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8,frameon=False,ncol=2)
fig.get_layout_engine().set(rect=(0,.055,1,.90));fig.suptitle('四卡计时｜每步四 rank 本地耗时最大值',fontsize=18,fontweight='bold');fig.text(.5,.016,'GBS4 / LBS1 / S4096｜1501–1510 预热，1511–1540 原始 30 点｜CANN 9.2.0 / PTA 2.12.0.post3\n数据：data/perf-steps.csv 及 data/perf-summary.json',ha='center',fontsize=10);fig.savefig(P/'figures/timing.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(3,2,figsize=(16,13),layout='constrained');fig.patch.set_facecolor('#f7f8fa')
for ax,(title,ids) in zip(axes.flat,panels):
 for off,key,color,label in [(-.25,'peak_allocated_gib','#2f68aa','allocated 峰'),(0,'peak_reserved_gib','#87b4df','reserved 峰'),(.25,'warm_deduplicated_gib','#dda14a','warm 去重 storage')]:
  vals=[max(x[key] for x in memory if x['run']==run) for run in ids];ax.bar([i+off for i in range(len(ids))],vals,.25,color=color,label=label)
 ax.set_title(title,loc='left',fontweight='bold',fontsize=12);ax.set_xticks(range(len(ids)),[runs[x]['label'] for x in ids],fontsize=8);ax.set_ylabel('GiB / 卡（四卡最大）');ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8,frameon=False,ncol=3)
fig.get_layout_engine().set(rect=(0,.065,1,.90));fig.suptitle('显存口径分列｜allocator 峰与 warm 唯一 storage',fontsize=18,fontweight='bold');fig.text(.5,.012,'GBS4 / LBS1 / S4096｜1501–1510 预热，1511–1540 原始 30 点｜CANN 9.2.0 / PTA 2.12.0.post3\nallocator 各卡峰取四卡最大；reserved 不是 HBM 总占用；warm 为去重快照｜data/perf-steps.csv / perf-summary.json',ha='center',fontsize=10);fig.savefig(P/'figures/memory.png',dpi=160);plt.close(fig)

# Separate SAC1000 regression from the included raw loss data.
import csv,json,hashlib,pathlib,sys,statistics,math
O=pathlib.Path(__file__).resolve().parent/'figures'; S=pathlib.Path(__file__).resolve().parent/'data'
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt;from matplotlib import font_manager
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=list(csv.DictReader((S/'sac1000-paired.csv').open(encoding='utf-8',newline='')));a=json.loads((S/'sac1000-audit.json').read_text(encoding='utf-8'));assert len(rows)==1000 and [int(r['step']) for r in rows]==list(range(1501,2501));steps=[int(r['step']) for r in rows]
col=lambda x:[float(r[x]) for r in rows]
x=col('original_global_loss');y=col('fixed_global_loss');d=col('fixed_minus_original');cum=col('cumulative_signed_mean');gx=col('original_grad_norm');gy=col('fixed_grad_norm'); assert all(math.isfinite(v) for v in x+y+d+cum+gx+gy);stats=a['loss']; assert all(math.isclose(y[i]-x[i],d[i],abs_tol=1e-14) for i in range(1000));assert math.isclose(statistics.mean(d),stats['signed_mean'],abs_tol=1e-14) and math.isclose(statistics.mean(map(abs,d)),stats['mae'],abs_tol=1e-14);assert sum(z>0 for z in d)==stats['positive'] and sum(z<0 for z in d)==stats['negative'] and sum(z==0 for z in d)==stats['zero']; im=max(range(1000),key=lambda i:abs(d[i]));assert steps[im]==1804 and math.isclose(abs(d[im]),stats['max_absolute'],abs_tol=1e-14)
font=pathlib.Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams['axes.unicode_minus']=False
fig,axs=plt.subplots(4,1,figsize=(15,14),sharex=True);fig.subplots_adjust(left=.10,right=.97,top=.87,bottom=.13,hspace=.42)
axs[0].plot(steps,x,lw=.8,label='原注册 1a39572f');axs[0].plot(steps,y,lw=.8,label='修复注册 03ea3194',alpha=.8);axs[0].set_ylabel('global mean loss');axs[0].legend(loc='upper right',ncol=2,fontsize=9)
axs[1].plot(steps,d,lw=.8,color='#1764ab');axs[1].axhline(0,c='#555',lw=.75);axs[1].set_ylabel('loss 差值\n修复 − 原注册');axs[1].annotate('step 1804\n最大绝对差 %.6f'%abs(d[im]),xy=(steps[im],d[im]),xytext=(1860,d[im]*.72),arrowprops={'arrowstyle':'->','color':'#a33'},fontsize=9,color='#a33');axs[1].set_ylim(min(d)-.008,max(d)+.008)
axs[2].plot(steps,cum,lw=.85,color='#217a5c');axs[2].axhline(0,c='#555',lw=.75);axs[2].set_ylabel('累计有符号\n均值');axs[2].set_title('前缀均值轨迹；±0.005 仅用于完整千步最终均值',loc='left',fontsize=9)
axs[3].plot(steps,gx,lw=.8,label='原注册');axs[3].plot(steps,gy,lw=.8,alpha=.8,label='修复注册');axs[3].set_ylabel('global grad norm');axs[3].set_xlabel('训练 step');axs[3].legend(loc='upper right',ncol=2,fontsize=9)
for ax in axs:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False);ax.set_xlim(1501,2500)
fig.text(.5,.973,'HP SAC 注册修复｜1501–2500 千步回归',ha='center',va='top',fontsize=17,fontweight='bold')
fig.text(.5,.94,'原注册 source 1a39572f ｜修复 source 03ea3194 ｜GBS4 / LBS1 / S4096 ｜CANN 9.2.0 ｜PTA 2.12.0.post3',ha='center',va='top',fontsize=10)
fig.text(.5,.078,'最终 absolute mean 9.07969e-5 ｜signed mean −9.07969e-5 ｜正 480 / 负 519 / 零 1',ha='center',fontsize=10)
fig.text(.5,.051,'MAE 0.000372414 ｜max absolute 0.076935（step 1804）｜MTP: N/A',ha='center',fontsize=10)
fig.text(.5,.025,'门槛：完整 1000 步 |signed mean| ≤ 0.005；每步差值保留全部极值，未经平滑',ha='center',fontsize=9,color='#505050')
fig.savefig(O/'sac1000-loss.png',dpi=150);plt.close(fig)
(O/'source-sha256.json').write_text(json.dumps({'paired.csv':sha(S/'sac1000-paired.csv'),'audit.json':sha(S/'sac1000-audit.json'),'scope':'rank0 trainer global mean; 1000 paired updates','max_abs_step':steps[im]},indent=2)+'\n',encoding='utf-8');print('saved plot and SHA')
