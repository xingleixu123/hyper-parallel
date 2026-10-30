"""CPU-only comparison of sealed TP2 one-update raw shards; no tolerance-based approval."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import stat
import sys

import numpy as np

DTYPES = {"torch.float32": np.dtype("<f4"), "torch.float64": np.dtype("<f8"),
          "torch.float16": np.dtype("<f2"), "torch.bfloat16": np.dtype("<u2"),
          "torch.int64": np.dtype("<i8"), "torch.int32": np.dtype("<i4"),
          "torch.bool": np.dtype("u1")}
COUNTS = {"model_local_states": 58, "model_local_parameters": 57,
          "optimizer_local_main_parameters": 57, "optimizer_local_slots": 86}
CHUNK_BYTES = 4 * 1024 * 1024


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    path = Path(path)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), "JSON is not a regular file: " + str(path))
    raw = path.read_bytes()
    after = path.lstat()
    require((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "JSON changed while reading")
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def layout(record, rank, kind, name):
    shape, local, mesh = record["global_shape"], record["shape"], record["mesh"]
    require(all(type(x) is int and x >= 0 for x in shape + local), "invalid shape")
    offsets, sizes = [0] * len(shape), list(shape)
    if mesh is None:
        require(kind == "buffers" and name == "model.layers.2.mlp.gate.e_score_correction_bias"
                and shape == local and record["placements"] == "None", "unsupported plain layout")
    else:
        widths, coords, placements, ranks = (mesh[x] for x in ("shape", "coordinate", "placements", "rank_list"))
        require(len(widths) == len(coords) == len(placements) and math.prod(widths) == len(ranks)
                and len(ranks) == len(set(ranks)), "invalid mesh")
        linear = 0
        for width, coord in zip(widths, coords):
            require(type(width) is int and type(coord) is int and width > 0 and 0 <= coord < width,
                    "invalid mesh coordinate")
            linear = linear * width + coord
        require(ranks[linear] == rank, "mesh does not map current rank")
        kinds = [p["class"].rsplit(".", 1)[-1] for p in placements]
        if "StridedShard" in kinds:
            require(len(widths) == 2 and kinds == ["StridedShard", "Shard"]
                    and placements[0].get("dim") == placements[1].get("dim") == 0
                    and placements[0].get("split_factor") == widths[1]
                    and shape and shape[0] % math.prod(widths) == 0, "unsupported strided layout")
            chunk = shape[0] // math.prod(widths)
            offsets[0] = (coords[1] * widths[0] + coords[0]) * chunk
            sizes[0] = chunk
        else:
            used = set()
            for width, coord, placement, kind in zip(widths, coords, placements, kinds):
                if kind == "Replicate":
                    continue
                dim = placement.get("dim")
                require(kind == "Shard" and type(dim) is int and 0 <= dim < len(shape) and dim not in used,
                        "unsupported placement")
                used.add(dim)
                require(shape[dim] % width == 0 or placement.get("uneven_shard") is True,
                        "unknown uneven shard")
                chunk = (shape[dim] + width - 1) // width
                offsets[dim] = min(coord * chunk, shape[dim])
                sizes[dim] = min(chunk, shape[dim] - offsets[dim])
    require(local == sizes, "local shape differs from shard geometry")
    return tuple(shape), tuple(offsets), tuple(sizes)


def coverage(records, kind, name):
    require(set(records) == set(range(4)), "rank coverage incomplete")
    boxes = {}
    shape = None
    for rank, rec in records.items():
        global_shape, offset, extent = layout(rec, rank, kind, name)
        require(shape is None or shape == global_shape, "global shape differs across ranks")
        shape = global_shape
        key = (offset, extent)
        if key in boxes:
            require(boxes[key]["sha"] == rec["local_sha256"], "replicated region bytes differ")
            boxes[key]["ranks"].append(rank)
        else:
            boxes[key] = {"sha": rec["local_sha256"], "ranks": [rank]}
    require(sum(math.prod(extent) for _, extent in boxes) == math.prod(shape), "logical coverage incomplete")
    for (a, b) in itertools.combinations(boxes, 2):
        overlap = math.prod(max(0, min(x + n, y + m) - max(x, y))
                            for x, n, y, m in zip(a[0], a[1], b[0], b[1]))
        require(overlap == 0, "logical regions overlap")
    return {"global_shape": list(shape), "unique_regions": len(boxes),
            "logical_elements": math.prod(shape), "replicated_region_count": sum(len(v["ranks"]) > 1 for v in boxes.values())}


def safe_raw(run, rank, rec):
    relative = Path(rec["path"])
    require(not relative.is_absolute() and ".." not in relative.parts and relative.suffix == ".bin",
            "unsafe raw path")
    root = Path(run) / ("probe-rank-%d" % rank)
    target = root / relative
    require(root.is_dir() and not root.is_symlink() and not target.is_symlink(), "unsafe raw location")
    for ancestor in target.relative_to(root).parents:
        check = root / ancestor
        require(not check.is_symlink(), "raw path contains symlink")
    before = target.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_size == rec["size"], "raw size differs")
    dtype = DTYPES.get(rec["dtype"])
    require(dtype is not None and before.st_size == math.prod(rec["shape"]) * dtype.itemsize,
            "unsupported dtype or element count differs")
    require(re.fullmatch(r"[a-f0-9]{64}", rec["local_sha256"]) is not None, "invalid raw hash")
    return target, before, dtype


def verify_empty_raw(path, before, expected):
    """Verify an empty regular file using the same opened object and path identity."""
    require(before.st_size == 0, "empty raw has nonzero size")
    identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        opened = os.fstat(fd)
        # Windows path stat and descriptor stat may expose different ctime semantics.
        # Compare shared identity across APIs, then each API's full identity over time.
        common = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
        same_open = (common(opened) == common(before) if os.name == "nt"
                     else identity(opened) == identity(before))
        require(stat.S_ISREG(opened.st_mode) and same_open, "empty raw changed before open")
        raw = os.read(fd, 1)
        require(not raw and identity(os.fstat(fd)) == identity(opened)
                and identity(path.lstat()) == identity(before), "empty raw changed during comparison")
    finally:
        os.close(fd)
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == expected, "empty raw hash differs")
    return digest


def as_float(block, dtype_name):
    if dtype_name == "torch.bfloat16":
        return (block.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    return block.astype(np.float64)


def compare_raw(run_a, run_b, rank, left, right):
    pa, before_a, dtype_a = safe_raw(run_a, rank, left)
    pb, before_b, dtype_b = safe_raw(run_b, rank, right)
    require(dtype_a == dtype_b and left["dtype"] == right["dtype"]
            and left["shape"] == right["shape"], "raw dtype/shape mismatch")
    if math.prod(left["shape"]) == 0:
        for key in ("global_shape", "placements", "mesh"):
            require(key in left and key in right and left[key] == right[key], "empty raw layout mismatch")
        layout(left, rank, "model", "empty")
        layout(right, rank, "model", "empty")
        sha_a = verify_empty_raw(pa, before_a, left["local_sha256"])
        sha_b = verify_empty_raw(pb, before_b, right["local_sha256"])
        return {"elements": 0, "different_elements": 0,
                "nonfinite_sync": 0, "nonfinite_deferred": 0, "nonzero_sync": 0, "nonzero_deferred": 0,
                "max_abs": 0.0, "difference_l2": 0.0, "reference_l2": 0.0, "relative_l2": None,
                "sync_sha256": sha_a, "deferred_sha256": sha_b, "empty_shard": True}
    a, b = np.memmap(pa, mode="r", dtype="u1"), np.memmap(pb, mode="r", dtype="u1")
    sha_a, sha_b = hashlib.sha256(), hashlib.sha256()
    count = math.prod(left["shape"])
    nonfinite_a = nonfinite_b = different = nonzero_a = nonzero_b = 0
    max_abs, sum_diff_sq, sum_ref_sq = 0.0, 0.0, 0.0
    integer = left["dtype"] in ("torch.int64", "torch.int32", "torch.bool")
    step = max(dtype_a.itemsize, CHUNK_BYTES - CHUNK_BYTES % dtype_a.itemsize)
    for off in range(0, len(a), step):
        ba, bb = a[off:off + step], b[off:off + step]
        sha_a.update(memoryview(ba))
        sha_b.update(memoryview(bb))
        xa, xb = ba.view(dtype_a), bb.view(dtype_b)
        different += int(np.count_nonzero(xa != xb))
        if integer:
            nonzero_a += int(np.count_nonzero(xa))
            nonzero_b += int(np.count_nonzero(xb))
            continue
        xa, xb = as_float(xa, left["dtype"]), as_float(xb, right["dtype"])
        fa, fb = np.isfinite(xa), np.isfinite(xb)
        nonfinite_a += int(np.count_nonzero(~fa))
        nonfinite_b += int(np.count_nonzero(~fb))
        nonzero_a += int(np.count_nonzero(fa & (xa != 0)))
        nonzero_b += int(np.count_nonzero(fb & (xb != 0)))
        valid = fa & fb
        if np.any(valid):
            x, y = xa[valid], xb[valid]
            delta = x - y
            max_abs = max(max_abs, float(np.max(np.abs(delta))))
            sum_diff_sq += float(np.dot(delta, delta))
            sum_ref_sq += float(np.dot(x, x))
    del a, b
    for path, before in ((pa, before_a), (pb, before_b)):
        after = path.lstat()
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                "raw changed during comparison")
    require(sha_a.hexdigest() == left["local_sha256"] and sha_b.hexdigest() == right["local_sha256"],
            "raw hash differs")
    require(math.isfinite(sum_diff_sq) and math.isfinite(sum_ref_sq), "numeric norm overflow")
    return {"elements": count, "different_elements": different,
            "nonfinite_sync": nonfinite_a, "nonfinite_deferred": nonfinite_b,
            "nonzero_sync": nonzero_a, "nonzero_deferred": nonzero_b,
            "max_abs": None if integer else max_abs,
            "difference_l2": None if integer else math.sqrt(sum_diff_sq),
            "reference_l2": None if integer else math.sqrt(sum_ref_sq),
            "relative_l2": None if integer or sum_ref_sq == 0 else math.sqrt(sum_diff_sq / sum_ref_sq),
            "sync_sha256": sha_a.hexdigest(), "deferred_sha256": sha_b.hexdigest()}


def collect(run, expected_policy):
    run = Path(run)
    audit, audit_sha = read_json(run / "root-external-audit-report.json")
    flags = ("source_live_compared", "target_live_compared", "groups_compared",
             "extra_compared", "checkpoint_identity_stable")
    require(audit.get("status") == "passed" and audit.get("source_step") == 1500
            and audit.get("topology") == "tp2-dp2"
            and audit.get("source_contract_sha256") ==
            "1fd4ac373a20f3edd8620cb0ac0593cf8f0ca5876a0d7f7227089358ef5cb633"
            and audit.get("global_readback_sha256") ==
            "f8a11dbae22b44c1c9ce0edf7f3edb05899b36fe87e63343135ebe8e0cf76466"
            and audit.get("role_counts") == COUNTS and len(audit.get("comparisons", [])) == 258
            and all(audit.get(k) is True for k in flags), "native1500 audit incomplete")
    rows = {}
    for rank in range(4):
        report, report_sha = read_json(run / ("probe-rank-%d.json" % rank))
        target, target_sha = read_json(run / ("target-live-rank-%d.json" % rank))
        require(report.get("schema") == "tp-nosync-one-update-rank-v1"
                and report.get("status") == "complete" and report.get("rank") == rank
                and report.get("source_step") == 1500
                and report.get("global_step") == 1501 and report.get("training_updates") == 1
                and report.get("requires_grad_sync_config") is expected_policy
                and report.get("audit_report_sha256") == audit_sha
                and report.get("target_sha256") == target_sha
                and audit.get("target_rank_sha256", {}).get(str(rank)) == target_sha
                and report.get("run_id") == target.get("run_id") == audit.get("run_id"),
                "rank/audit/target identity differs")
        require(report["topology"] == "tp2-dp2" and not report["preclip"]["pending_full_gradients"],
                "topology or pending gradients differ")
        trace = report["sync_trace"]
        require([x["micro_step"] for x in trace] == [0, 1] and
                all(x["num_micro_steps"] == 2 and x["schedulers"] for x in trace),
                "actual synchronization trace incomplete")
        for micro, item in enumerate(trace):
            expected = expected_policy or micro == 1
            require(all(x["reduce_grads"] is expected for x in item["schedulers"]),
                    "actual synchronization state differs")
            roots = [x for x in item["schedulers"] if x["is_part_root"]]
            require(roots and all(x["is_last_backward"] is (micro == 1) for x in roots),
                    "actual root last-backward differs")
        target_tensors = target["tensors"]
        require({k: len(v) for k, v in target_tensors.items()} == COUNTS, "initial role coverage differs")
        require(set(report["preclip"]["gradients"]) == set(target_tensors["model_local_parameters"])
                and set(report["post"]["model"]) == set(target_tensors["model_local_parameters"])
                and set(report["post"]["slots"]) == set(target_tensors["optimizer_local_slots"])
                and set(report["post"]["buffers"]) == {"model.layers.2.mlp.gate.e_score_correction_bias"},
                "post/gradient inventory differs")
        require(all(x.get("main_is_model") is True and x.get("state_model_same_view") is True
                    for x in report["post"]["aliases"].values())
                and set(report["post"]["aliases"]) == set(report["post"]["model"]),
                "model/main alias proof incomplete")
        require(len(report["verified_batches"]) == 2
                and all(x.get("prepared_four_fields_checked") is True for x in report["verified_batches"]),
                "sample proof incomplete")
        rows[rank] = {"report": report, "report_sha256": report_sha,
                      "target": target, "target_sha256": target_sha}
    return {"audit": audit, "audit_sha256": audit_sha, "rows": rows}


def clip_evidence(rank, sync_report, deferred_report):
    evidence = []
    ca, cb = sync_report["preclip"]["clip"], deferred_report["preclip"]["clip"]
    require(ca["path"] == cb["path"] == "native_clip_grad_norm_"
            and ca["formula"] == cb["formula"] and ca["max_norm"] == cb["max_norm"]
            and ca["returned_norm_dtype"] == cb["returned_norm_dtype"]
            and ca["scale_dtype"] == cb["scale_dtype"], "clipping contract differs")
    for report, clip in ((sync_report, ca), (deferred_report, cb)):
        require(all(isinstance(clip[k], (int, float)) and math.isfinite(clip[k])
                    for k in ("max_norm", "returned_norm", "recomputed_scale")),
                "nonfinite clipping scalar")
        require(clip["max_norm"] > 0 and 0 <= clip["recomputed_scale"] <= 1
                and report["grad_norm"] == clip["returned_norm"],
                "clipping scalar or returned training norm differs")
    for report, clip in ((sync_report, ca), (deferred_report, cb)):
        expected = {name for name, rec in report["preclip"]["gradients"].items() if rec.get("present")}
        require(set(clip["postclip_samples"]) == expected, "postclip sample inventory differs")
    for name in sorted(set(ca["postclip_samples"]) | set(cb["postclip_samples"])):
        per_arm = []
        for report, clip in ((sync_report, ca), (deferred_report, cb)):
            rec = report["preclip"]["gradients"].get(name)
            if not rec or not rec.get("present"):
                per_arm.append(None)
                continue
            positions, before, after = rec["sample_positions"], rec["sample_values"], clip["postclip_samples"].get(name)
            require(isinstance(after, list) and len(positions) == len(before) == len(after)
                    and len(set(positions)) == len(positions)
                    and all(type(p) is int and 0 <= p < math.prod(rec["shape"]) for p in positions)
                    and all(isinstance(v, (int, float)) and math.isfinite(v) for v in before + after),
                    "postclip sample incomplete or nonfinite")
            approximate = [abs(float(y) - float(x) * clip["recomputed_scale"]) for x, y in zip(before, after)]
            per_arm.append({"dtype": rec["dtype"], "positions": positions,
                            "selection": rec["sample_selection"], "pre": before, "post": after,
                            "max_abs_from_uncast_scale_product": max(approximate, default=0.0)})
        evidence.append({"rank": rank, "name": name, "sync": per_arm[0], "deferred": per_arm[1]})
    return {"rank": rank, "sync": {k: v for k, v in ca.items() if k != "postclip_samples"},
            "deferred": {k: v for k, v in cb.items() if k != "postclip_samples"},
            "norm_abs_difference": abs(ca["returned_norm"] - cb["returned_norm"]),
            "scale_abs_difference": abs(ca["recomputed_scale"] - cb["recomputed_scale"]),
            "samples": evidence,
            "sample_bound": "product is approximate, ignores dtype cast/rounding and is not an acceptance criterion"}


def compare_runs(run_sync, run_deferred):
    sync = collect(run_sync, True)
    deferred = collect(run_deferred, False)
    results, unresolved, warnings, clip, steps = [], [], [], [], []
    for rank in range(4):
        a, b = sync["rows"][rank], deferred["rows"][rank]
        require(a["target"]["tensors"] == b["target"]["tensors"], "initial state/layout/hash differs across arms")
        for key in ("groups", "extra", "loader", "initial_sampler", "mesh", "source_contract_sha256"):
            require(a["target"][key] == b["target"][key], "initial metadata differs: " + key)
        batch_a, batch_b = a["report"]["verified_batches"], b["report"]["verified_batches"]
        require(batch_a == batch_b, "actual batch records differ")
        require(a["report"]["groups"] == b["report"]["groups"] and
                a["report"]["final_loader"] == b["report"]["final_loader"], "post-step metadata differs")
        clip.append(clip_evidence(rank, a["report"], b["report"]))
        require(all(math.isfinite(float(row["report"][key])) for row in (a, b)
                    for key in ("loss", "grad_norm")), "nonfinite loss or norm")
        steps.append({"rank": rank, "sync_loss": a["report"]["loss"],
                      "deferred_loss": b["report"]["loss"], "sync_grad_norm": a["report"]["grad_norm"],
                      "deferred_grad_norm": b["report"]["grad_norm"]})
        for kind in ("gradients", "model", "buffers", "slots"):
            aa = a["report"]["preclip"]["gradients"] if kind == "gradients" else a["report"]["post"][kind]
            bb = b["report"]["preclip"]["gradients"] if kind == "gradients" else b["report"]["post"][kind]
            require(set(aa) == set(bb), "role names differ")
            for name in sorted(aa):
                left, right = aa[name], bb[name]
                if kind == "gradients":
                    require(left.get("trainable") == right.get("trainable"), "trainability differs")
                    if not left["trainable"]:
                        continue
                    if not left.get("present") or not right.get("present"):
                        unresolved.append({"rank": rank, "name": name, "sync_present": left.get("present", False),
                                           "deferred_present": right.get("present", False)})
                        continue
                    require(left["kind"] == right["kind"] and left["layout_source"] == right["layout_source"],
                            "gradient storage kind or layout provenance differs")
                    for rec, row in ((left, a), (right, b)):
                        if rec["layout_source"] == "parameter_spec_fallback_for_plain_local_grad":
                            physical = rec["actual_gradient_layout"]
                            parameter = row["target"]["tensors"]["model_local_parameters"][name]
                            require(physical["mesh"] is None and physical["shape"] == parameter["shape"]
                                    and all(rec[k] == parameter[k] for k in ("global_shape", "placements", "mesh")),
                                    "gradient parameter-spec inference differs")
                        else:
                            require(rec["layout_source"] == "gradient_spec"
                                    and all(rec[k] == rec["actual_gradient_layout"][k]
                                            for k in ("shape", "global_shape", "placements", "mesh")),
                                    "gradient spec metadata differs")
                category = {"gradients": "preclip-grad", "model": "post-model",
                            "buffers": "post-buffer", "slots": "post-slot"}[kind]
                expected_path = category + "/" + hashlib.sha256(name.encode("utf-8")).hexdigest() + ".bin"
                require(left["path"] == right["path"] == expected_path, "raw role/path mismatch")
                for key in ("shape", "dtype", "global_shape", "placements", "mesh"):
                    require(left[key] == right[key], "layout differs: " + kind + ":" + name + ":" + key)
                metrics = compare_raw(run_sync, run_deferred, rank, left, right)
                results.append({"rank": rank, "kind": kind, "name": name, **metrics})
                if kind == "model":
                    orig = a["target"]["tensors"]["model_local_parameters"][name]["local_sha256"]
                    ga = a["report"]["preclip"]["gradients"][name]
                    gb = b["report"]["preclip"]["gradients"][name]
                    if left["size"] and (left["local_sha256"] == orig or right["local_sha256"] == orig):
                        warnings.append({"rank": rank, "name": name,
                                         "sync_unchanged": left["local_sha256"] == orig,
                                         "deferred_unchanged": right["local_sha256"] == orig,
                                         "sync_grad_present": ga.get("present", False),
                                         "deferred_grad_present": gb.get("present", False),
                                         "interpretation": "requires gradient, optimizer and rounding review"})
    cover = []
    for kind in ("gradients", "model", "buffers", "slots"):
        first = sync["rows"][0]["report"]["preclip"]["gradients"] if kind == "gradients" else sync["rows"][0]["report"]["post"][kind]
        for name in sorted(first):
            records_by_arm = []
            for data in (sync, deferred):
                records = {rank: (data["rows"][rank]["report"]["preclip"]["gradients"][name]
                                  if kind == "gradients" else data["rows"][rank]["report"]["post"][kind][name])
                           for rank in range(4)}
                if kind == "gradients" and not all(rec.get("present", False) for rec in records.values()):
                    records_by_arm.append(None)
                else:
                    records_by_arm.append(coverage(records, kind, name))
            cover.append({"kind": kind, "name": name, "sync": records_by_arm[0], "deferred": records_by_arm[1]})
    nonfinite = sum(row["nonfinite_sync"] + row["nonfinite_deferred"] for row in results)
    return {"schema": "tp-nosync-one-update-comparison-v1",
            "status": "invalid_nonfinite" if nonfinite else "incomplete" if unresolved else "compared",
            "scope": "same-layout one-step relative comparison; no automatic tolerance or independent full-model oracle",
            "rank_reports": {"sync": [sync["rows"][i]["report_sha256"] for i in range(4)],
                             "deferred": [deferred["rows"][i]["report_sha256"] for i in range(4)]},
            "audit_sha256": {"sync": sync["audit_sha256"], "deferred": deferred["audit_sha256"]},
            "role_comparisons": results, "coverage": cover, "unresolved_gradients": unresolved,
            "clip_evidence": clip, "steps": steps,
            "unchanged_model_warnings": warnings, "nonfinite_elements": nonfinite}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync", type=Path, required=True)
    parser.add_argument("--deferred", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(sys.byteorder == "little", "little-endian CPU required")
    sync, deferred = args.sync.resolve(strict=True), args.deferred.resolve(strict=True)
    output = args.out.resolve()
    require(sync != deferred and not output.is_relative_to(sync) and not output.is_relative_to(deferred),
            "output or runs overlap")
    result = compare_runs(sync, deferred)
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": result["status"], "sha256": hashlib.sha256(raw).hexdigest(),
                      "nonfinite_elements": result["nonfinite_elements"],
                      "unresolved_gradients": len(result["unresolved_gradients"])}))


if __name__ == "__main__":
    main()
