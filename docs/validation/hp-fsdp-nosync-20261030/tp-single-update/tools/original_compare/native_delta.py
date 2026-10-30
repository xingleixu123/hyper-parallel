"""CPU-only native1500 value to post-update comparison for the fixed TP2 model."""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import compare_step as cmp

MAX_CHUNK_ELEMENTS = 1024 * 1024
BUFFER = "model.layers.2.mlp.gate.e_score_correction_bias"


def chunks(array):
    """Yield C-order chunks of at most 4 MiB without materializing a full slice."""
    if array.size <= MAX_CHUNK_ELEMENTS:
        yield np.ascontiguousarray(array).reshape(-1)
        return
    if array.ndim <= 1:
        for off in range(0, array.size, MAX_CHUNK_ELEMENTS):
            yield np.ascontiguousarray(array[off:off + MAX_CHUNK_ELEMENTS]).reshape(-1)
        return
    row = math.prod(array.shape[1:])
    if row <= MAX_CHUNK_ELEMENTS:
        step = max(1, MAX_CHUNK_ELEMENTS // row)
        for off in range(0, array.shape[0], step):
            yield np.ascontiguousarray(array[off:off + step]).reshape(-1)
    else:
        for off in range(array.shape[0]):
            yield from chunks(array[off])


def compare_initial_post(initial, run, rank, rec, initial_sha):
    """Compare a CPU numpy logical region and pinned post bytes, in logical order."""
    cmp.require(initial.dtype == np.dtype("<f4") and rec["dtype"] == "torch.float32"
                and tuple(rec["shape"]) == initial.shape,
                "source delta supports the observed float32 tensors only")
    path, before, dtype = cmp.safe_raw(run, rank, rec)
    cmp.require(dtype == np.dtype("<f4"), "post dtype differs")
    if initial.size == 0:
        cmp.layout(rec, rank, "model", "empty")
        initial_raw = initial.tobytes(order="C")
        cmp.require(not initial_raw and hashlib.sha256(initial_raw).hexdigest() == initial_sha,
                    "native initial empty region hash differs")
        post_sha = cmp.verify_empty_raw(path, before, rec["local_sha256"])
        return {"elements": 0, "different_elements": 0, "unchanged_elements": 0,
                "all_values_unchanged": None, "nonzero_initial": 0, "nonzero_post": 0,
                "nonfinite_initial": 0, "nonfinite_post": 0, "max_abs_update": 0.0,
                "update_l2": 0.0, "initial_l2": 0.0, "relative_update_l2": None,
                "initial_sha256": initial_sha, "post_sha256": post_sha, "empty_shard": True}
    post = np.memmap(path, mode="r", dtype="<f4")
    source_sha, post_sha = hashlib.sha256(), hashlib.sha256()
    cursor = different = nonzero_source = nonzero_post = nonfinite_source = nonfinite_post = 0
    max_abs = diff_sq = source_sq = 0.0
    for block in chunks(initial):
        cmp.require(block.size <= MAX_CHUNK_ELEMENTS and block.dtype == np.dtype("<f4"), "source chunk differs")
        other = post[cursor:cursor + block.size]
        cmp.require(len(other) == block.size, "post tensor truncated")
        source_sha.update(memoryview(block).cast("B")); post_sha.update(memoryview(other).cast("B"))
        different += int(np.count_nonzero(block != other))
        fa, fb = np.isfinite(block), np.isfinite(other)
        nonfinite_source += int(np.count_nonzero(~fa)); nonfinite_post += int(np.count_nonzero(~fb))
        nonzero_source += int(np.count_nonzero(fa & (block != 0)))
        nonzero_post += int(np.count_nonzero(fb & (other != 0)))
        valid = fa & fb
        if np.any(valid):
            a, b = block[valid].astype(np.float64), other[valid].astype(np.float64)
            diff = b - a
            max_abs = max(max_abs, float(np.max(np.abs(diff))))
            diff_sq += float(np.dot(diff, diff)); source_sq += float(np.dot(a, a))
        cursor += block.size
    del post
    after = path.lstat()
    cmp.require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                "post raw changed during source comparison")
    cmp.require(cursor == initial.size and source_sha.hexdigest() == initial_sha
                and post_sha.hexdigest() == rec["local_sha256"], "native initial/post region hash differs")
    cmp.require(math.isfinite(diff_sq) and math.isfinite(source_sq), "source delta norm overflow")
    return {"elements": cursor, "different_elements": different, "unchanged_elements": cursor - different,
            "all_values_unchanged": different == 0, "nonzero_initial": nonzero_source,
            "nonzero_post": nonzero_post, "nonfinite_initial": nonfinite_source,
            "nonfinite_post": nonfinite_post, "max_abs_update": max_abs,
            "update_l2": math.sqrt(diff_sq), "initial_l2": math.sqrt(source_sq),
            "relative_update_l2": None if source_sq == 0 else math.sqrt(diff_sq / source_sq),
            "initial_sha256": source_sha.hexdigest(), "post_sha256": post_sha.hexdigest()}


class DeltaObserver:
    def __init__(self, sync, deferred, progress_path=None):
        self.runs = {"sync": Path(sync).resolve(strict=True), "deferred": Path(deferred).resolve(strict=True)}
        cmp.require(self.runs["sync"] != self.runs["deferred"], "arms share an output root")
        self.data = {"sync": cmp.collect(self.runs["sync"], True),
                     "deferred": cmp.collect(self.runs["deferred"], False)}
        for rank in range(4):
            x = self.data["sync"]["rows"][rank]["target"]
            y = self.data["deferred"]["rows"][rank]["target"]
            for key in ("tensors", "groups", "extra", "loader", "initial_sampler", "mesh", "source_contract_sha256"):
                cmp.require(x[key] == y[key], "initial states differ across arms: " + key)
        self.result = []
        self.roles = set()
        self.progress_path = Path(progress_path) if progress_path is not None else None

    def observe(self, dcp_key, value, mapping):
        cmp.require(str(value.dtype) == "torch.float32" and str(value.device) == "cpu",
                    "unexpected native tensor dtype/device")
        global_array = value.detach().numpy()
        for row in mapping:
            if row["dcp_key"] != dcp_key:
                continue
            section, name = row["section"], row["name"]
            if section == "optimizer_local_main_parameters":
                continue  # alias proven on the fresh target; model stored once
            cmp.require(section in ("model_local_states", "optimizer_local_slots"), "unexpected mapped role")
            kind = ("buffers" if name == BUFFER else "model") if section == "model_local_states" else "slots"
            role = (kind, name)
            cmp.require(role not in self.roles, "duplicate post role")
            self.roles.add(role)
            for arm in ("sync", "deferred"):
                data = self.data[arm]
                for rank in range(4):
                    target = data["rows"][rank]["target"]["tensors"][section][name]
                    rec = data["rows"][rank]["report"]["post"][kind][name]
                    for k in ("shape", "dtype", "global_shape", "placements", "mesh"):
                        cmp.require(target[k] == rec[k], "initial/post layout differs: " + name)
                    category = {"model": "post-model", "buffers": "post-buffer", "slots": "post-slot"}[kind]
                    expected_path = category + "/" + hashlib.sha256(name.encode()).hexdigest() + ".bin"
                    cmp.require(rec["path"] == expected_path, "post role/path mismatch")
                    global_shape, offset, extent = cmp.layout(rec, rank, kind, name)
                    cmp.require(global_shape == tuple(global_array.shape), "native DCP global shape differs")
                    region = global_array[tuple(slice(start, start + length) for start, length in zip(offset, extent))]
                    metrics = compare_initial_post(region, self.runs[arm], rank, rec, target["local_sha256"])
                    row={"arm": arm, "rank": rank, "kind": kind, "name": name,
                         "dcp_key": dcp_key, **metrics}
                    self.result.append(row)
                    if self.progress_path is not None:
                        raw=(json.dumps({"phase":"delta_complete",**row},sort_keys=True,
                                        separators=(",",":"),allow_nan=False)+"\n").encode()
                        with self.progress_path.open("ab") as stream:
                            stream.write(raw);stream.flush();os.fsync(stream.fileno())

    def finish(self, path, *, source_validation_sha256, reader_sha256):
        cmp.require(all(len(x) == 64 and all(c in "0123456789abcdef" for c in x)
                        for x in (source_validation_sha256, reader_sha256)), "source validation pin missing")
        for arm, data in self.data.items():
            run = self.runs[arm]
            _, audit_sha = cmp.read_json(run / "root-external-audit-report.json")
            cmp.require(audit_sha == data["audit_sha256"], "audit changed during native read")
            for rank in range(4):
                _, report_sha = cmp.read_json(run / ("probe-rank-%d.json" % rank))
                _, target_sha = cmp.read_json(run / ("target-live-rank-%d.json" % rank))
                cmp.require(report_sha == data["rows"][rank]["report_sha256"]
                            and target_sha == data["rows"][rank]["target_sha256"],
                            "target/report changed during native read")
        expected = {"model": 57, "buffers": 1, "slots": 86}
        cmp.require({kind: sum(k == kind for k, _ in self.roles) for kind in expected} == expected,
                    "native comparison role coverage differs")
        cmp.require(len(self.result) == 2 * 4 * sum(expected.values()), "native comparison record count differs")
        invalid = sum(x["nonfinite_initial"] + x["nonfinite_post"] for x in self.result)
        doc = {"schema": "tp-one-update-native-delta-v1", "status": "invalid_nonfinite" if invalid else "compared",
               "scope": "native1500 actual values versus each post shard; no acceptance threshold or oracle",
               "audit_sha256": {arm: data["audit_sha256"] for arm, data in self.data.items()},
               "source_validation_sha256": source_validation_sha256, "reader_sha256": reader_sha256,
               "role_counts_unique": expected, "comparison_records": len(self.result),
               "nonfinite_elements": invalid, "per_role_rank_arm": self.result,
               "limitations": ["zero deltas require gradient, weight decay, optimizer and rounding interpretation",
                               "float32 only; any unreviewed layout or dtype fails closed",
                               "empty shards contain no update elements and do not establish an update"]}
        raw = (json.dumps(doc, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        with Path(path).open("xb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        folder_fd = os.open(Path(path).parent, os.O_RDONLY)
        try:
            os.fsync(folder_fd)
        finally:
            os.close(folder_fd)
        return {"sha256": hashlib.sha256(raw).hexdigest(), "status": doc["status"],
                "comparison_records": len(self.result), "nonfinite_elements": invalid}
