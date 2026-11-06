from pathlib import Path
import array,hashlib,json,math,struct,sys
r=Path(__file__).parent
def load_tensor(path,key):
 raw=path.read_bytes();n=struct.unpack("<Q",raw[:8])[0];h=json.loads(raw[8:8+n]);m=h[key]
 assert m["dtype"] in ("F32","F64")
 a=array.array("f" if m["dtype"]=="F32" else "d")
 start,end=m["data_offsets"];a.frombytes(raw[8+n+start:8+n+end])
 if sys.byteorder!="little":a.byteswap()
 assert len(a)==math.prod(m["shape"])
 return m,a
fixture=r/"model-audit/router-aux-native-reduction-probe/aux-native-fixture.safetensors"
assert hashlib.sha256(fixture.read_bytes()).hexdigest()=="a035d1bbb9afe561dc4a62f93586ae20653aa9a2f12d05d8760d7d4758e35d54"
meta,reference=load_tensor(fixture,"independent_fp64_global_gradient")
assert meta["shape"]==[64,3584]
stats=[]
for rank in range(4):
 vals=[]
 for setting,terminal in [("false","f702a06e5171e2d528f600d85935d487fe84f2ef780505c42b1ef6950ff71914"),("true","5ed2894d5bb811ff13883845a98d61897b5eaec2561df0c8b24fdbea4de9b90c")]:
  d=r/f"fourcard-runtime-prep/native-aux-lp{setting}-a2-evidence"
  controller=json.loads((d/"controller.json").read_bytes())
  assert hashlib.sha256((d/"controller.json").read_bytes()).hexdigest()==terminal
  assert controller["status"]=="finished_process_exit_0" and controller["returncode"]==0 and controller["holder_removed"]
  assert controller["source_after"]=={"files":1544,"all_bytes_match":True}
  row=json.loads((d/f"rank-{rank}/result.json").read_bytes())
  path=d/f"rank-{rank}/native-gradients.safetensors"
  assert hashlib.sha256(path.read_bytes()).hexdigest()==row["gradient_file_sha256"]
  assert row["updates"]==0 and row["manual_gradient_collectives_or_compensation"] is False
  assert row["native_gradient_scaling_factor"]==(.25 if setting=="false" else .5)
  assert row["source_manifest_sha256"]=="4967881062f63d8144bd8c4214b60c16f9ca7495b7f241a6701d04a630e08d34"
  for micro in row["micros"]:
   assert micro["bound_foundation_multiplier"]==micro["actual_aux_context_scale"]==2*micro["local_foundation_count"]
   assert micro["native_CP_loss_sum_multiplier"]==2
  shape,a=load_tensor(path,"native_after_step_token_normalize")
  chunk=row["native_chunk"];assert chunk["sizes"]==shape["shape"] and chunk["offsets"][1]==0 and chunk["sizes"][1]==3584
  start=chunk["offsets"][0]*3584;b=reference[start:start+len(a)]
  assert len(a)==len(b) and all(map(math.isfinite,a))
  norm=math.fsum(x*x for x in b);assert norm>0
  ratio=math.fsum(x*y for x,y in zip(a,b))/norm
  rel1=math.sqrt(math.fsum((x-y)**2 for x,y in zip(a,b))/norm)
  rel2=math.sqrt(math.fsum((x-2*y)**2 for x,y in zip(a,b))/(4*norm))
  assert (rel1 if setting=="false" else rel2)<1e-5
  stats.append({"rank":rank,"LP":setting=="true","LS_multiplier":ratio,"relL2_vs_1x":rel1,"relL2_vs_2x":rel2})
  vals.append(a)
 assert all(t==2*f for f,t in zip(*vals))
out={"status":"root_actual_NPU_gradients_stdlib_independent_readback_passed","source":"4967881062f63d8144bd8c4214b60c16f9ca7495b7f241a6701d04a630e08d34","cases":stats,"every_rank_true_equals_two_times_false_elementwise_exact":True,"oracle":"independent FP64 global AUX mean gradient; no checkpoint or optimizer","scope":"Confirmed TP2CP2/SPfalse/GA4/DP1 native AUX-only LP replica multiplier defect; no full-model precision claim"}
p=r/"model-audit/router-aux-native-reduction-probe/root-native-pair-verified.json"
p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"report_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),**out},ensure_ascii=False))
