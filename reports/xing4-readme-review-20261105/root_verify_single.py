import csv,hashlib,json,math,statistics
from pathlib import Path
d=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
hpfile=d/"single-run/evidence/step-observations.jsonl"
mffile=d/"reference/mf-formal-1501-1505.jsonl"
oldfile=d/"reference/previous-hp-1501-1505.jsonl"
load=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
hp=load(hpfile);mf={r["step"]:r for r in load(mffile)};old={r["step"]:r for r in load(oldfile)}
assert [r["step"] for r in hp]==list(range(1501,1506))
rows=[];fields=0
for h in hp:
    m=mf[h["step"]];o=old[h["step"]]
    assert len(h["microbatches"])==len(m["microbatches"])==len(o["microbatches"])==2
    for hn,mn,on in zip(h["microbatches"],m["microbatches"],o["microbatches"]):
        for f in ("input_ids","labels","loss_mask","position_ids"):
            subset=lambda p:{k:p[f][k] for k in ("shape","dtype","sha256_c_order")}
            assert subset(hn)==subset(mn)==subset(on)
            assert hn[f]["shape"]==[1,4096]
            fields+=1
    assert math.isfinite(h["grad_norm"]) and h["grad_norm"]>0
    assert abs(h["loss"]+0.1*h["mtp_raw"]-h["total_loss_diagnostic"])<1e-6
    row={"step":h["step"],"hp_loss":h["loss"],"mf_loss":m["callback_metrics"]["loss"],
         "hp_mtp_raw":h["mtp_raw"],"mf_mtp_raw":m["callback_metrics"]["mtp_loss_values"]["mtp-1-loss"]}
    row["loss_delta"]=row["hp_loss"]-row["mf_loss"];row["mtp_delta"]=row["hp_mtp_raw"]-row["mf_mtp_raw"]
    rows.append(row)
def stats(key):
    x=[r[key] for r in rows]
    return {"count":len(x),"signed_mean":statistics.mean(x),"abs_signed_mean":abs(statistics.mean(x)),
            "positive":sum(v>0 for v in x),"negative":sum(v<0 for v in x),"zero":sum(v==0 for v in x),
            "mae":statistics.mean(map(abs,x)),"max_abs":max(map(abs,x)),
            "max_abs_step":rows[max(range(len(x)),key=lambda i:abs(x[i]))]["step"]}
rv=json.loads((d/"single-run/evidence/restore-verification.json").read_text())
assert rv["restore_components_verified"] and rv["restored_state_verified"]
assert (rv["model_tensors_verified"],rv["optimizer_slots_verified"],rv["fp32_alias_verified"])==(65,91,61)
assert rv["global_step"]==1500 and rv["sampler_state"]["consumed_samples"]==3000
assert fields==40
with (d/"single-mf-hp.csv").open("w",encoding="utf-8",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
out={"scope":"MF1500->HP1501-1505 integration regression; not a new 1000-step acceptance",
     "fields_verified":fields,"metrics":{"LOSS":stats("loss_delta"),"MTPLOSS":stats("mtp_delta")},
     "restore":{"model":65,"optimizer_slots":91,"fp32_alias":61,"step":1500,"consumed_samples":3000},
     "source_sha256":{str(p):sha(p) for p in (hpfile,mffile,oldfile)}}
(d/"root-single-verification.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(out,ensure_ascii=False,indent=2))
