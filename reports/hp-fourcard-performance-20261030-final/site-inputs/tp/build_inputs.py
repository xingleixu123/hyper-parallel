#!/usr/bin/env python3
"""Build or verify source-bound website input from sealed local scalar evidence."""
import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
G = HERE.parent
P = G.parent
CANDIDATE = G / "tp-perf-final-public-candidate-v1"
MEMORY_AUDIT = G / "tp-perf-memory-attribution-max-v1" / "analysis.json"
BASELINE_LOG = P / "sac-strict-1000-adopt-v2" / "original-selective" / "run" / "rank-0-perf.jsonl"
DEFERRED_RUN = G / "tp-perf-acceptance-closed-v1" / "deferred" / "run"
SYNC_RUN = G / "tp-perf-acceptance-closed-v1" / "sync" / "run"
EXPECTED_SUMS_SHA = "060ffe710cfbf5129d7131d852500fbff1e74880b204addfd9f7cdb685c8af0c"
EXPECTED_MEMORY_AUDIT_SHA = "2a0affbbd2d98d60b55c403e4c9a613376169f835ec5ba98012572cec99fa988"


def require(condition, detail):
    if not condition:
        raise RuntimeError(detail)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked(path, expected):
    data = path.read_bytes()
    require(sha(data) == expected, f"SHA mismatch: {path}")
    return data


def csv_rows(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline="")))


def json_rows(data):
    lines = data.decode("utf-8").splitlines()
    require(all(line.strip() for line in lines), "empty raw JSONL line")
    return [json.loads(line) for line in lines]


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def jsonl_bytes(rows):
    return ("\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                      for row in rows) + "\n").encode("utf-8")


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-14, abs_tol=1e-14)


def build_expected():
    # Verify the frozen public candidate and all its named files before use.
    sums = checked(CANDIDATE / "SHA256SUMS", EXPECTED_SUMS_SHA)
    pins = {}
    for line in sums.decode("utf-8").splitlines():
        digest, sep, name = line.partition("  ")
        require(sep and len(digest) == 64 and name and "/" not in name and "\\" not in name and name not in pins,
                "invalid candidate SHA256SUMS")
        pins[name] = digest
    required = {"README.md", "build_public.py", "first40.csv", "loss-pairs-1000.csv", "memory.csv",
                "metrics.json", "source-lineage.json", "validation.json"}
    require(set(pins) == required, "candidate manifest files differ")
    source = {name: checked(CANDIDATE / name, digest) for name, digest in pins.items()}
    upstream = json.loads(source["source-lineage.json"])
    summary = json.loads(source["metrics.json"])
    mem_audit = json.loads(checked(MEMORY_AUDIT, EXPECTED_MEMORY_AUDIT_SHA))
    raw_pins = {}
    raw_records = {}
    for arm, directory in (("sync", SYNC_RUN), ("deferred", DEFERRED_RUN)):
        for rank in range(4):
            path = directory / f"rank-{rank}-perf.jsonl"
            expected = mem_audit["inputs_sha256"][arm][str(rank)]["perf_sha256"]
            data = checked(path, expected)
            raw_pins[f"{arm}_rank{rank}"] = {"path_from_P": path.relative_to(P).as_posix(), "sha256": expected}
            records = json_rows(data)
            require(len(records) == 40, f"{arm} rank{rank}: expected 40 rows")
            for i, rec in enumerate(records):
                require(rec.get("step") == 1501 + i and rec.get("rank") == rank
                        and rec.get("phase") == ("warmup" if i < 10 else "measured"),
                        f"{arm} rank{rank}: step/rank/phase mismatch")
                duration = rec.get("duration_seconds")
                require(isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 0,
                        f"{arm} rank{rank}: invalid duration")
                for metric in ("peak_allocated_bytes", "peak_reserved_bytes"):
                    v = rec["memory_after"][metric]
                    require(type(v) is int and v >= 0, f"{arm} rank{rank}: invalid {metric}")
            raw_records[arm, rank] = records

    raw_lineage = upstream["rank0_raw_lineage"]
    baseline_sha = raw_lineage["reference_perf_jsonl"]["sha256"]
    base_records = json_rows(checked(BASELINE_LOG, baseline_sha))
    require(len(base_records) == 1000, "expected 1000 reference rows")
    raw_pins["historical_reference_rank0"] = {"path_from_P": BASELINE_LOG.relative_to(P).as_posix(), "sha256": baseline_sha}
    require(raw_pins["deferred_rank0"]["sha256"] == raw_lineage["candidate_perf_jsonl"]["sha256"],
            "candidate rank0 perf lineage mismatch")
    continuation_path = DEFERRED_RUN / "rank-0-continuation.jsonl"
    continuation_sha = raw_lineage["candidate_continuation_jsonl"]["sha256"]
    continuation = json_rows(checked(continuation_path, continuation_sha))
    require(len(continuation) == 960, "expected 960 continuation rows")
    raw_pins["tp_continuation_rank0"] = {"path_from_P": continuation_path.relative_to(P).as_posix(), "sha256": continuation_sha}
    tp_records = raw_records["deferred", 0] + continuation

    loss_rows = csv_rows(source["loss-pairs-1000.csv"])
    require(len(loss_rows) == 1000, "expected 1000 loss CSV rows")
    loss_ref, loss_tp, loss_zero, loss_delta, loss_map, deltas = [], [], [], [], [], []
    for index, (row, ref, tp) in enumerate(zip(loss_rows, base_records, tp_records)):
        step = 1501 + index
        require(int(row["step"]) == step and ref.get("step") == step and tp.get("step") == step
                and ref.get("rank") == 0 and tp.get("rank") == 0, f"loss step/rank mismatch: {step}")
        a, b, delta, prefix = (float(row[key]) for key in
                               ("reference_loss", "tp_deferred_loss", "delta", "cumulative_signed_mean"))
        require(all(math.isfinite(v) for v in (a, b, delta, prefix)) and a == ref["loss"] and b == tp["loss"]
                and delta == b - a, f"loss source/delta mismatch: {step}")
        deltas.append(delta)
        require(prefix == math.fsum(deltas) / len(deltas), f"prefix mismatch: {step}")
        loss_ref.append({"step": step, "loss": a})
        loss_tp.append({"step": step, "loss": b})
        loss_zero.append({"step": step, "signed_delta": 0, "cumulative_signed_mean": 0})
        loss_delta.append({"step": step, "signed_delta": delta, "cumulative_signed_mean": prefix})
        loss_map.append({"step": step, "derived_line": index + 1, "public_loss_csv_line": index + 2,
                         "reference_source": "historical_reference_rank0", "reference_line": index + 1,
                         "tp_source": "deferred_rank0" if step <= 1540 else "tp_continuation_rank0",
                         "tp_line": step - 1500 if step <= 1540 else step - 1540})
    loss_expected = summary["loss_1000"]
    mean = math.fsum(deltas) / len(deltas)
    magnitude = max(abs(d) for d in deltas)
    max_step = 1501 + next(i for i, d in enumerate(deltas) if abs(d) == magnitude)
    counts = {"positive": sum(x > 0 for x in deltas), "negative": sum(x < 0 for x in deltas),
              "zero": sum(x == 0 for x in deltas)}
    require(loss_expected["n"] == 1000 and all(counts[k] == loss_expected[k] for k in counts)
            and mean == loss_expected["signed_mean_delta"] and abs(mean) == loss_expected["absolute_signed_mean_delta"]
            and magnitude == loss_expected["max_absolute_delta"] and max_step == loss_expected["max_absolute_delta_step"]
            and math.fsum(map(abs, deltas)) / 1000 == loss_expected["mean_absolute_error"], "loss summary mismatch")

    perf_rows = csv_rows(source["first40.csv"])
    require(len(perf_rows) == 40, "expected 40 performance rows")
    time_out = {"sync": [], "deferred": []}
    memory_out = {"sync": [], "deferred": []}
    times = {"sync": [], "deferred": []}
    peak = {arm: {rank: {kind: [] for kind in ("allocated", "reserved")} for rank in range(4)}
            for arm in ("sync", "deferred")}
    perf_map = []
    for index, row in enumerate(perf_rows):
        step = 1501 + index
        require(int(row["step"]) == step and row["phase"] == ("warmup" if index < 10 else "measured"),
                f"first40 step/phase mismatch: {step}")
        pair_max = {}
        winners = {}
        for arm in ("sync", "deferred"):
            values = [raw_records[arm, rank][index]["duration_seconds"] for rank in range(4)]
            maximum = max(values)
            require(maximum == float(row[f"{arm}_max_seconds"]), f"4-rank maximum mismatch: {arm} {step}")
            pair_max[arm] = maximum
            winners[arm] = [rank for rank, value in enumerate(values) if value == maximum]
        require(float(row["deferred_minus_sync_seconds"]) == pair_max["deferred"] - pair_max["sync"]
                and float(row["deferred_over_sync"]) == pair_max["deferred"] / pair_max["sync"],
                f"first40 comparison mismatch: {step}")
        if index < 10:
            continue
        for arm in ("sync", "deferred"):
            maximum = pair_max[arm]
            times[arm].append(maximum)
            time_out[arm].append({"step": step, "step_seconds_max_rank": maximum,
                                  "nominal_tokens_per_second": 16384 / maximum})
            memory = {"step": step}
            for rank in range(4):
                after = raw_records[arm, rank][index]["memory_after"]
                for kind in ("allocated", "reserved"):
                    v = after[f"peak_{kind}_bytes"]
                    peak[arm][rank][kind].append(v)
                    memory[f"peak_{kind}_rank{rank}_gib"] = v / (2 ** 30)
            memory_out[arm].append(memory)
        perf_map.append({"step": step, "derived_line": index - 9, "public_first40_csv_line": index + 2,
                         "all_rank_perf_raw_line": index + 1, "max_rank_winners": winners})
    require(len(perf_map) == 30 and perf_map[0]["step"] == 1511 and perf_map[-1]["step"] == 1540,
            "measured step window mismatch")
    for arm in ("sync", "deferred"):
        metrics = summary["performance_30_steps"][arm]
        average = math.fsum(times[arm]) / 30
        require(metrics["n"] == 30 and average == metrics["mean_seconds"]
                and 16384 / average == metrics["nominal_tokens_per_second"]
                and min(times[arm]) == metrics["min_seconds"] and max(times[arm]) == metrics["max_seconds"]
                and close(statistics.pstdev(times[arm]), metrics["population_std_seconds"]),
                f"performance summary mismatch: {arm}")
    mem_rows = csv_rows(source["memory.csv"])
    require(len(mem_rows) == 4, "expected four memory summary rows")
    devices = {}
    for rank, row in enumerate(mem_rows):
        require(int(row["rank"]) == rank and int(row["physical_device"]) == rank + 4,
                "memory rank/physical mapping mismatch")
        devices[str(rank)] = int(row["physical_device"])
        for arm in ("sync", "deferred"):
            for kind in ("allocated", "reserved"):
                require(max(peak[arm][rank][kind]) == int(row[f"{arm}_measured_peak_{kind}_bytes"]),
                        f"memory peak summary mismatch: {arm} rank{rank} {kind}")

    artifacts = {
        "data/01-fsdp-loss.jsonl": jsonl_bytes(loss_ref),
        "data/02-tp-loss.jsonl": jsonl_bytes(loss_tp),
        "data/03-zero-guides.jsonl": jsonl_bytes(loss_zero),
        "data/04-tp-delta.jsonl": jsonl_bytes(loss_delta),
        "data/05-tp-sync-perf.jsonl": jsonl_bytes(time_out["sync"]),
        "data/06-tp-deferred-perf.jsonl": jsonl_bytes(time_out["deferred"]),
        "data/07-tp-sync-allocator.jsonl": jsonl_bytes(memory_out["sync"]),
        "data/08-tp-deferred-allocator.jsonl": jsonl_bytes(memory_out["deferred"]),
    }
    lineage = {
        "schema": "tp-website-input-lineage-v1",
        "public_candidate_manifest_sha256": EXPECTED_SUMS_SHA,
        "public_candidate_files_sha256": pins,
        "memory_attribution_analysis_sha256": EXPECTED_MEMORY_AUDIT_SHA,
        "raw_files": raw_pins,
        "derived_sha256": {name: sha(data) for name, data in artifacts.items()},
        "loss_points": loss_map,
        "measured_points": perf_map,
        "rank_to_physical_device": devices,
        "zero_guides": "synthetic mathematical zero at each real step, not a training observation",
        "loss_fields": {"loss": "trainer-reported rank0 value", "signed_delta": "TP deferred - historical reference",
                        "cumulative_signed_mean": "fsum of signed deltas 1501 through step / (step-1500)"},
        "performance_fields": {"step_seconds_max_rank": "max four synchronized rank durations",
                               "nominal_tokens_per_second": "16384 / that step max; nominal, per step"},
        "allocator_fields": "memory_after peak allocated/reserved bytes in each corresponding raw rank-step divided by 2**30; GiB",
        "warm_storage": "not plotted: it is an independent one-time snapshot",
    }
    artifacts["lineage.json"] = json_bytes(lineage)
    validation = {
        "schema": "tp-website-input-validation-v1", "status": "cpu_source_comparison_passed_not_plotted",
        "source_candidate_manifest_sha256": EXPECTED_SUMS_SHA, "memory_audit_sha256": EXPECTED_MEMORY_AUDIT_SHA,
        "builder_sha256": sha(Path(__file__).read_bytes()),
        "loss_1000": {"steps": [1501, 2500], "rows": 1000, "signed_mean": mean, "counts": counts,
                      "max_abs": magnitude, "max_abs_step": max_step, "nonfinite": 0},
        "performance_30": {arm: {"steps": [1511, 1540], "rows": 30,
                                   "mean_seconds": math.fsum(times[arm]) / 30,
                                   "nominal_tokens_per_second_from_mean": 16384 / (math.fsum(times[arm]) / 30)}
                           for arm in ("sync", "deferred")},
        "allocator": "2 arms x 4 ranks x 30 actual rank-step records, allocated and reserved peak in GiB",
        "data_sha256": {name: sha(data) for name, data in artifacts.items()},
        "limitations": ["CPU data preparation only; no website, browser, NPU, or publication",
                        "1000-step reference differs in TP/recompute/sync policy; not single-factor causality",
                        "0.005 is a full-window absolute signed-mean condition, not a per-step line",
                        "step throughput is nominal; ratio to mean seconds differs from mean of per-step throughput",
                        "reserved can include allocator cache; warm storage not plotted"],
    }
    artifacts["validation.json"] = json_bytes(validation)
    # Confirm both sealed inputs and each underlying raw source did not change while preparing.
    checked(CANDIDATE / "SHA256SUMS", EXPECTED_SUMS_SHA)
    checked(MEMORY_AUDIT, EXPECTED_MEMORY_AUDIT_SHA)
    for name, pin in pins.items():
        checked(CANDIDATE / name, pin)
    for item in raw_pins.values():
        checked(P / item["path_from_P"], item["sha256"])
    return artifacts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    artifacts = build_expected()
    if args.build:
        require(not (HERE / "data").exists() and not (HERE / "lineage.json").exists()
                and not (HERE / "validation.json").exists(), "output already exists; no overwrite")
        (HERE / "data").mkdir()
        for name, data in artifacts.items():
            with (HERE / name).open("xb") as stream:
                stream.write(data)
        print(f"BUILT {len(artifacts)} source-verified files under {HERE}")
    else:
        for name, data in artifacts.items():
            require((HERE / name).read_bytes() == data, f"derived output differs: {name}")
        print(f"PASS: {len(artifacts)} outputs byte-identical; all source pins and 1000/30 points verified")


if __name__ == "__main__":
    main()
