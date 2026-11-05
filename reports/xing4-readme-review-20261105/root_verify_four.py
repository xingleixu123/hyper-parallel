import csv, hashlib, json, math, statistics
from pathlib import Path
d=Path(__file__).resolve().parent
base=d/"results"
names=("fsdp4","ep2","cp2dp2","tp2cp2")
proof={};values={};sourcehash={}
for name in names:
    paths=[base/name/f"rank-{r}"/"steps.json" for r in range(4)]
    ranks=[json.loads(p.read_text()) for p in paths]
    for rank,rows in enumerate(ranks):
        assert [x["step"] for x in rows]==list(range(6,106)),(name,rank)
        restored=json.loads((base/name/f"rank-{rank}"/"restore-verification.json").read_text())
        assert restored["step"]==5
        if name == "fsdp4":
            assert restored["local_state_matches_producer"]
        else:
            assert restored["full_dcp_verified"]
        assert restored["scheduler_rng_match_checkpoint"]
        sourcehash[str(paths[rank].relative_to(d))]=hashlib.sha256(paths[rank].read_bytes()).hexdigest()
    samples=fields=0;result=[]
    for i in range(100):
        step=6+i;records=[r[i] for r in ranks]
        for metric in ("foundation_loss","foundation_loss/mtp"):
            assert len({x["losses"][metric] for x in records})==1,(name,step,metric)
        assert len({x["mtp_raw"] for x in records})==1
        global_samples={}
        for row in records:
            assert math.isfinite(row["grad_norm"])
            assert math.isclose(row["mtp_raw"],row["losses"]["foundation_loss/mtp"]/0.1,rel_tol=1e-12)
            for micro in row["microbatches"]:
                assert micro["dataset_fields_verified"]
                if name in ("fsdp4", "ep2"):
                    assert micro["input_ids"]["shape"] == [1,4096]
                else:
                    assert micro["global_length"] == 4096
                position=micro["logical_position"]
                start,end=(0,4096) if name in ("fsdp4", "ep2") else (micro["cp_start"],micro["cp_end"])
                assert 0<=start<end<=4096
                key=(start,end)
                subset={f:micro[f] for f in ("input_ids","labels","shift_labels","loss_mask","position_ids")}
                for field in subset.values():assert field["shape"]==[1,end-start]
                seen=global_samples.setdefault(position,{})
                if key in seen:assert seen[key]==subset,(name,step,position,key)
                else:seen[key]=subset
                fields+=len(subset)
        assert sorted(global_samples)==list(range((step-1)*4,step*4)),(name,step,sorted(global_samples))
        for shards in global_samples.values():
            cursor=0
            for start,end in sorted(shards):
                assert start==cursor,(name,step)
                cursor=end
            assert cursor==4096
            samples+=1
        result.append({"step":step,"loss":records[0]["losses"]["foundation_loss"],"mtp_raw":records[0]["mtp_raw"]})
    proof[name]={"ranks":4,"steps":100,"global_samples":samples,"verified_field_records":fields,"rank_metrics_equal":True}
    values[name]=result
    proof[name]["restore_scope"]="independent full DCP equality"
    if name=="fsdp4":
        full=d/"fsdp-global-verify"
        global_result=json.loads((full/"result.json").read_text())
        global_spec=json.loads((full/"spec.json").read_text())
        assert global_result["status"]=="passed" and len(global_result["rows"])==624
        assert global_result["spec_sha256"]==hashlib.sha256((full/"spec.json").read_bytes()).hexdigest()
        for rank in range(4):
            local=base/name/f"rank-{rank}"/"restored-state.json"
            assert global_spec["target_rank_proofs"][str(rank)]["sha256"]==hashlib.sha256(local.read_bytes()).hexdigest()
        proof[name]["global_dcp_bindings"]=624

def stats(delta,rows):
    return {"count":len(delta),"signed_mean":statistics.mean(delta),"abs_signed_mean":abs(statistics.mean(delta)),
            "positive":sum(x>0 for x in delta),"negative":sum(x<0 for x in delta),"zero":sum(x==0 for x in delta),
            "mae":statistics.mean(map(abs,delta)),"max_abs":max(map(abs,delta)),
            "max_abs_step":rows[max(range(len(delta)),key=lambda i:abs(delta[i]))]["step"]}
comparisons={}
allrows=[]
for name in names:
    reference=values["fsdp4"]
    rows=[]
    for current,ref in zip(values[name],reference):
        row={"configuration":name,"step":current["step"],"loss":current["loss"],"fsdp_loss":ref["loss"],
             "mtp_raw":current["mtp_raw"],"fsdp_mtp_raw":ref["mtp_raw"],
             "loss_delta":current["loss"]-ref["loss"],"mtp_delta":current["mtp_raw"]-ref["mtp_raw"]}
        rows.append(row);allrows.append(row)
    comparisons[name]={metric:stats([r[key] for r in rows],rows) for metric,key in (("LOSS","loss_delta"),("MTPLOSS","mtp_delta"))}
with (d/"fourcard-versus-fsdp.csv").open("w",encoding="utf-8",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
out={"scope":"Four-card native step5->6-105 regression, 100 updates; not 1500->2500 full acceptance",
     "batch":{"GBS":4,"LBS":1},"comparisons":comparisons,"proof":proof,"raw_sha256":sourcehash}
(d/"root-four-verification.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in out.items() if k!="raw_sha256"},ensure_ascii=False,indent=2))
