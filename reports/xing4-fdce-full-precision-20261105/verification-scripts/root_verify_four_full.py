"""Audit actual four-card optimizer-step records without smoothing or per-step loss thresholds."""
import argparse
import csv
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

FIELDS=("input_ids","labels","shift_labels","loss_mask","position_ids")
LAYOUTS={"fsdp10":(4,1,1,1,1),"fsdp_backbone":(4,1,1,1,1),"ep2":(4,1,1,2,1),
         "cp2dp2":(2,1,2,1,2),"tp2cp2":(1,2,2,1,4),"baseline":(4,1,1,1,1)}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def file_records(p):
    rows=read(p)
    assert len({x["step"] for x in rows})==len(rows),("duplicate",str(p))
    return rows

def audit(name,folder):
    dp,tp,cp,ep,ga=LAYOUTS[name]
    stop=1510 if name=="fsdp10" else 2500
    expected=list(range(1 if name=="baseline" else 1501,stop+1))
    ranks=[];source={};restored=[];coordinates=[]
    for rank in range(4):
        rd=folder/f"rank-{rank}"
        rows=file_records(rd/"steps.json")
        assert [r["step"] for r in rows]==expected,(name,rank,"step-window")
        assert [r["step"] for r in map(json.loads,(rd/"steps.jsonl").read_text().splitlines())]==expected
        assert [json.loads(x) for x in (rd/"steps.jsonl").read_text().splitlines()]==rows
        done=read(rd/"complete.json")
        assert done["rank"]==rank and done["updates"]==len(expected) and done["horizon"]==7500
        inv=read(rd/"inventory.json")
        assert inv["rank"]==rank and inv["microbatches"]==ga
        assert {k:inv["mesh_sizes"][k] for k in ("dp_size","tp_size","cp_size","ep_size")}==dict(
            dp_size=dp,tp_size=tp,cp_size=cp,ep_size=ep)
        if cp>1:
            coordinates.append(tuple(inv["mesh_sizes"][k] for k in ("dp_rank","cp_rank","tp_rank")))
        cfg=inv["config"];train=cfg["training"]
        assert train["global_batch_size"]==4 and train["micro_batch_size"]==1
        assert train["train_iters"]==2500 and train["lr_scheduler_iters"]==7500
        if name=="baseline":
            init=read(rd/"initial-state.json")
            assert init["step"]==0 and init["sampler"]["consumed_samples"]==0
        else:
            proof=read(rd/"restore-verification.json")
            assert proof["rank"]==rank and proof["step"]==1500 and proof["scheduler_rng_match_checkpoint"]
            assert proof["local_state_matches_producer"] if name.startswith("fsdp") else proof["full_dcp_verified"]
            restored.append({"rank":rank,"sha256":sha(rd/"restore-verification.json"),"proof":proof,
                             "state_sha256":sha(rd/"restored-state.json")})
        for f in ("steps.json","steps.jsonl","complete.json","inventory.json"):
            source[str(rank)+"/"+f]=sha(rd/f)
        ranks.append({r["step"]:r for r in rows})
    if cp>1:
        assert set(coordinates)=={(d,c,t) for d in range(dp) for c in range(cp) for t in range(tp)}
        assert len(coordinates)==len(set(coordinates))==4
    output={};samples={};fields=0;duplicate_views=0
    for step in expected:
        rs=[r[step] for r in ranks]
        for key in ("foundation_loss","foundation_loss/mtp"):
            assert len({r["losses"][key] for r in rs})==1,(name,step,"rank-loss",key)
        assert len({r["mtp_raw"] for r in rs})==1
        assert all(r["lr_before"]==rs[0]["lr_before"] for r in rs),(name,step,"rank-lr")
        bysample={};view_counts=Counter()
        for rank,r in enumerate(rs):
            assert len(r["microbatches"])==ga,(name,step,rank,"ga")
            assert math.isfinite(r["grad_norm"]) and math.isfinite(r["loss"])
            assert math.isclose(r["mtp_raw"],r["losses"]["foundation_loss/mtp"]/0.1,rel_tol=1e-12)
            assert abs(r["loss"]-r["losses"]["foundation_loss"]-r["losses"]["foundation_loss/mtp"])<1e-5
            for micro in r["microbatches"]:
                assert micro["dataset_fields_verified"] is True
                start,end=(0,4096) if cp==1 else (micro["cp_start"],micro["cp_end"])
                if cp!=1:
                    assert micro["global_length"]==4096 and end-start==2048
                    assert micro["cp_rank"] in range(cp) and start==micro["cp_rank"]*2048
                    assert micro["tp_rank"] in range(tp) and micro["dp_rank"] in range(dp)
                    assert (micro["dp_rank"],micro["cp_rank"],micro["tp_rank"])==coordinates[rank]
                part={k:micro[k] for k in FIELDS}
                assert part["labels"]==part["shift_labels"],(name,step,rank,"labels-alias-content")
                view_counts[(micro["logical_position"],start,end)]+=1
                for value in part.values():
                    assert value["shape"]==[1,end-start] and value["dtype"]=="torch.int64"
                    assert len(value["sha256"])==64
                seen=bysample.setdefault(micro["logical_position"],{})
                if (start,end) in seen:
                    assert seen[(start,end)]==part,(name,step,"replicated-input",rank)
                    duplicate_views+=1
                else:seen[(start,end)]=part
                fields+=len(FIELDS)
        assert sorted(bysample)==list(range((step-1)*4,step*4)),(name,step,"global-samples")
        assert set(view_counts.values())=={tp},(name,step,"missing-or-duplicate-TP-view")
        for parts in bysample.values():
            cursor=0
            for start,end in sorted(parts):
                assert start==cursor
                cursor=end
            assert cursor==4096
        samples[step]=bysample
        output[step]={"step":step,"loss":rs[0]["losses"]["foundation_loss"],"mtp_raw":rs[0]["mtp_raw"],
                      "grad_norm_min":min(r["grad_norm"] for r in rs),"grad_norm_max":max(r["grad_norm"] for r in rs),
                      "lr_before":rs[0]["lr_before"]}
    return output,samples,{"window":[expected[0],stop],"ranks":4,"steps":len(expected),
         "unique_global_samples":len(expected)*4,"recorded_field_hashes":fields,"verified_semantic_fields_per_micro":4,"matching_duplicate_views":duplicate_views,
         "batch":{"GBS":4,"LBS":1,"gradient_accumulation":ga},"layout":{"dp":dp,"tp":tp,"cp":cp,"ep":ep},
         "raw_sha256":source,"restore_proofs":restored}

def metrics(values,steps):
    mean=math.fsum(values)/len(values);idx=max(range(len(values)),key=lambda i:abs(values[i]))
    return dict(count=len(values),signed_mean=mean,abs_signed_mean=abs(mean),
        positive=sum(v>0 for v in values),negative=sum(v<0 for v in values),zero=sum(v==0 for v in values),
        mae=math.fsum(map(abs,values))/len(values),max_abs=abs(values[idx]),max_abs_step=steps[idx],
        mean_threshold=0.005,mean_pass=abs(mean)<=0.005,
        signs_present=any(v>0 for v in values) and any(v<0 for v in values),identical=all(v==0 for v in values))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--baseline",required=True,type=Path)
    ap.add_argument("--case",action="append",default=[],help="name=directory")
    ap.add_argument("--output",required=True,type=Path)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    reference,ref_samples,refproof=audit("baseline",args.baseline)
    results={};allrows=[]
    for item in args.case:
        name,path=item.split("=",1)
        assert name!="baseline" and name not in results
        values,samples,proof=audit(name,Path(path))
        steps=sorted(values);rows=[];same_shape_fields=0
        for step in steps:
            v,ref=values[step],reference[step]
            assert v["lr_before"]==ref["lr_before"],(name,step,"baseline-lr")
            if LAYOUTS[name][2]==1:
                assert samples[step]==ref_samples[step],(name,step,"baseline-full-input")
                same_shape_fields+=4*len(FIELDS)
            row={"configuration":name,"step":step,"loss":v["loss"],"fsdp_loss":ref["loss"],
                 "mtp_raw":v["mtp_raw"],"fsdp_mtp_raw":ref["mtp_raw"],
                 "loss_delta":v["loss"]-ref["loss"],"mtp_delta":v["mtp_raw"]-ref["mtp_raw"],
                 "grad_norm_min":v["grad_norm_min"],"grad_norm_max":v["grad_norm_max"],
                 "fsdp_grad_norm_min":ref["grad_norm_min"],"fsdp_grad_norm_max":ref["grad_norm_max"]}
            rows.append(row)
        stats={}
        for key,metric in (("loss_delta","LOSS"),("mtp_delta","MTPLOSS")):
            vals=[r[key] for r in rows];stats[metric]=metrics(vals,steps)
            for i,row in enumerate(rows):row[key+"_cumulative_mean"]=math.fsum(vals[:i+1])/(i+1)
        proof["direct_full_shape_input_fields_equal"]=same_shape_fields
        proof["cp_input_evidence"]="Real input_ids, loss_mask, position_ids and valid shift_labels checked against the frozen dataset; masked-off label values are ignored by the model and not claimed byte-identical to the raw dataset. labels and shift_labels hashes must match the frozen preprocessing alias. All rank coordinates and exactly TP copies per sample/slice are verified. SHA256 is not concatenated to invent a full input hash."
        results[name]={"full_window":steps==list(range(1501,2501)),"metrics":stats,"proof":proof}
        allrows.extend(rows)
    if allrows:
        with (args.output/"fourcard-versus-fsdp.csv").open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=allrows[0]);w.writeheader();w.writerows(allrows)
    out={"direction":"candidate-pure-FSDP-continuous","baseline":refproof,"results":results,
        "full_required_configurations_present":all(k in results and results[k]["full_window"] for k in
            ("fsdp_backbone","ep2","cp2dp2","tp2cp2")),
        "scope":"Step metrics, actual-input coverage and restore receipt audit. Independent DCP source/target readback, physical mapping and runtime identities are bound separately. Timings include instrumentation and do not establish performance comparisons."}
    (args.output/"root-four-verification.json").write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:{"full_window":v["full_window"],"metrics":v["metrics"]} for k,v in results.items()},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
