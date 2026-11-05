"""Independently audit every paired MF/HP optimizer step."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--mf", type=Path, required=True)
p.add_argument("--hp", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
def records(path):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    steps = [r["step"] for r in rows]
    assert len(set(steps)) == len(steps), ("duplicate steps", path)
    return rows, hashlib.sha256(raw).hexdigest()
mf_rows, mf_hash = records(a.mf)
hp_rows, hp_hash = records(a.hp)
assert [r["step"] for r in hp_rows] == list(range(1501, 2501))
mf = {r["step"]: r for r in mf_rows}
assert all(s in mf for s in range(1501, 2501))
proof = json.loads((a.hp.parent / "restore-verification.json").read_bytes())
assert proof["restored_state_verified"] and proof["restore_components_verified"]
assert (proof["model_tensors_verified"], proof["optimizer_slots_verified"], proof["fp32_alias_verified"]) == (65, 91, 61)
assert proof["global_step"] == 1500 and proof["sampler_state"] == {
    "consumed_samples": 3000, "global_batch_size": 2, "dp_world_size": 1, "epoch": 0}
native_types = {}
rows = []
checked = 0
for h in hp_rows:
    m = mf[h["step"]]
    assert len(h["microbatches"]) == len(m["microbatches"]) == len(h["native_microbatches"]) == 2
    for i in range(2):
        for key in ("input_ids", "labels", "loss_mask", "position_ids"):
            hv, mv = h["microbatches"][i][key], m["microbatches"][i][key]
            assert {k: hv[k] for k in ("shape", "dtype", "sha256_c_order")} == {
                k: mv[k] for k in ("shape", "dtype", "sha256_c_order")}, (h["step"], i, key)
            assert hv["shape"] == [1, 4096] and hv["dtype"] == "int32"
            nv = h["native_microbatches"][i][key]
            assert nv["shape"] == [1, 4096] and nv["integral"] and nv["int32_range"]
            assert -(2**31) <= nv["min"] <= nv["max"] < 2**31
            if key == "loss_mask":
                assert nv["binary"] and 0 <= nv["min"] <= nv["max"] <= 1
            native_types.setdefault(key, set()).add(nv["dtype"])
            checked += 1
    assert all(math.isfinite(float(h[k])) for k in ("loss", "mtp_raw", "total_loss_diagnostic", "grad_norm"))
    assert abs(h["loss"] + 0.1 * h["mtp_raw"] - h["total_loss_diagnostic"]) < 1e-5
    assert all(v["last_epoch"] == h["step"] for v in h["scheduler"].values())
    cm = m["callback_metrics"]
    row = {"step": h["step"], "hp_loss": h["loss"], "mf_loss": cm["loss"],
           "hp_mtp_raw": h["mtp_raw"], "mf_mtp_raw": cm["mtp_loss_values"]["mtp-1-loss"],
           "hp_grad_norm": h["grad_norm"]}
    assert all(math.isfinite(float(v)) for v in row.values())
    row["loss_delta"] = row["hp_loss"] - row["mf_loss"]
    row["mtp_delta"] = row["hp_mtp_raw"] - row["mf_mtp_raw"]
    rows.append(row)
assert checked == 8000
summary = {}
for name, key in (("LOSS", "loss_delta"), ("MTPLOSS", "mtp_delta")):
    values = [r[key] for r in rows]
    mean = math.fsum(values) / len(values)
    maximum = max(range(len(values)), key=lambda i: abs(values[i]))
    summary[name] = {"steps": len(values), "signed_mean": mean, "abs_signed_mean": abs(mean),
        "positive": sum(v > 0 for v in values), "negative": sum(v < 0 for v in values),
        "zero": sum(v == 0 for v in values), "mae": math.fsum(map(abs, values)) / len(values),
        "max_abs": abs(values[maximum]), "max_abs_step": rows[maximum]["step"],
        "mean_threshold": 0.005, "mean_pass": abs(mean) <= 0.005,
        "signs_present": any(v > 0 for v in values) and any(v < 0 for v in values)}
    for i, row in enumerate(rows):
        row[key + "_cumulative_mean"] = math.fsum(values[:i + 1]) / (i + 1)
with (a.output / "single-mf-hp.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
out = {"window": [1501, 2500], "paired_steps": 1000, "direction": "HP-MF",
       "metrics": summary, "actual_sample_fields_verified": checked,
       "sample_hash_semantics": "canonical int32 exact values; native HP dtype/range separately verified",
       "native_hp_dtypes": {k: sorted(v) for k, v in native_types.items()},
       "raw_sha256": {"mf": mf_hash, "hp": hp_hash},
       "restore_proof_sha256": hashlib.sha256((a.hp.parent / "restore-verification.json").read_bytes()).hexdigest(),
       "scope": "Full 1000-step metrics; checkpoint completion and source/resource evidence audited separately"}
(a.output / "root-single-verification.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
