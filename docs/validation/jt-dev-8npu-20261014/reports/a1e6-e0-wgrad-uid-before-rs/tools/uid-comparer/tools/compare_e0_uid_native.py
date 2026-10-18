# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#
# Experiment-only offline evidence validation; not a training implementation.

"""Compare bounded E0 sidecars by actual source-DP UIDs, without importing Torch."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import types
from typing import Any


CONTRACT_SHA = "d2fd3aecbb23b51c4bee147ab20d71c7e84c7738968ca201f24f121b2b311eeb"
CONTRACT_PATH = Path(__file__).resolve().parents[2] / "e0_wgrad_contract.py"
IDENTITY_SCHEMA = "e0-wgrad-runtime-identity-v1"
COMPLETE = "LOCAL_CAPTURE_PENDING_GLOBAL_UID_VERIFICATION"
CASES = {"fsdp8": "r129-a1e6-e0-wgrad-fsdp8-3-v3", "ep4": "r129-a1e6-e0-wgrad-ep4-3-v3"}
IDENTITY_FIELDS = {
    "schema", "case", "source", "mode", "rank", "dp", "ep", "rank_count", "step", "micro",
    "target", "coord", "pins", "complete", "steps", "raw", "metadata",
}
EXPECTED_PINS: dict[str, str] = {
    "contract": "d2fd3aecbb23b51c4bee147ab20d71c7e84c7738968ca201f24f121b2b311eeb",
    "native_parent": "6aac4bc8ac19df4f222154a4784cf652b0d0dc1e73cf37845454a92c369d3929",
    "native_training": "650004d255678696d6cd48353f1e9283769e8edb0405497ccd03557938da185d",
    "capture": "aef5a45ef770da810f07a59a3d97e3363368e7bb7076525d8e95f05ac9c6d717",
    "runtime": "e00039cd516ce01975d068318fa613b0f1d893304899f2a78562af6a2c36a1df",
    "profile": "b01dcf73a23ec967b04b5b5358f2f8cdae028227cb87e28ac465c0bb7e722c9c",
    "expert_probe": "461fb5c6aeca377d1157f5f6a6a021650629d9d9decccef80d4442d030a6257f",
    "precision": "50acc479aaa8ca4603243cfb9ce3dd0b8907e35a9fe430b816671e6d7bdf9659",
    "fp32_wgrad": "c9e3448d9f818f4c0ef3257f31e8bc4f24c7b9300109e85f9db086fd1bb7f670",
    "trainer": "79426f3fdd5f91f8b74791b45b727558637552eadc702176761ac96ce6bf3c87",
    "base": "eb0293f1c87c09895398e848cee62b39a2500484ab2a4f35ce9cfd7085b8a0c7",
    "swiglu": "b477b1bb81279c7e31c40f9011133d6417e7e376308d19f84860b3960ace852e",
    "modeling": "2ddd28910688710ea777f1724fb8aeb8821c6045a01f18f21eea4e92507fd906",
    "experts": "475c80c77fae3345c2f3a53075aa1858f8dddf4bd7a30b7dd8ef5c9feea41fee",
    "collectives": "831258bffcffe9fccb71c6e580f8efc8e1198485b3dda23431a13cacd66cc7c0",
    "recorder": "65fd9112101e6d978ecb5e52bfbcd9173752d95754d441d49da1cc55007a9b46",
    "hsdp": "b910d5e4e79accee631fe7ac7f0f00dd6436f00b6aabf78f02691ee398fd9ec0",
    "hsdp_utils": "d75122dea756fa141ad7ae67ff16e991ca0f9d19b1cda84b41d1cc30349c2491",
    "c10d": "c03488a330d66d9b20fb9480f6c3b1bbfba60e9aeadf18b3eca1c5ca93f88728",
    "logger": "b7b5866e4d3c3847e24f3fe488f187cbd90ac94955a24a98dc10f0d96c99b649",
    "torch_version": "f2bf53f312426647423216eeb78ad168e11cea511e6309d15aacaef05ef19a53",
    "checker_source": "503ad96653850eee76c03beb8fd986aed74342e9043189d54b71aafdc71e0e8f",
    "cpu_tests": "4ad878deb8411d28063b83ce7eac3d18508808e297c1e458953ff6500c11b379",
    "cpu_receipt": "b0e8ad85fe8f90cccecb1977fc61bfc9b450d167c9fbfc3d0b0c71454de85722",
    "cpu_execution": "b08391adf30561c17793d10d2ec9ba2718872b9dd7a0abd0d634c643402c0034",
    "cpu_review": "c6a4f1ef0d5a174ed18bdd47416e0e74ce6570973d72074ae3ff0dd010030649",
    "checker_review": "33e7969c83184699e5cf0bc91b85168ecd49011971c821992f638177a7085cf4",
    "ownership": "31ed29f45d5e847d372ca31ef553717ee260158b73e0d1d70128f20dfff262d4",
    "cpu8_native": "732a7c1c3d64003e8010e761019fe1a7b3d0bd881636c31e290ad2141116ed7d",
    "cpu8_checker": "6435921fb597d4aa7f8660354fe3203475686f84a5d88610e799ad0e4ede19f5",
    "cpu8_tests": "0c78528e23327deced2b5a0d08817c591ab6fdbd9d6eaa6733b29d46cc7a7385",
    "cpu8_receipt": "126a353d45d64205fe7214bd268063474532ecfc7606a37cf602d960cecb382a",
    "cpu8_execution": "6561c79df7a4baea827ac97494222006a46346a290383cf7b7b2fe1a61ed8efb",
    "cpu8_review": "51b523ffc0ac16f0a3ee90d6a3b3efc322e65fb59d60ce8e292c8c516150022f",
    "adapter": "3ad7f86ffd6b32c97d42a4698704d99417fbd7708bba74e5067bba964df97a50"
}


def require(condition: bool, message: str) -> None:
    """Reject incomplete or ambiguous evidence."""
    if not condition:
        raise ValueError(message)


def digest(raw: bytes) -> str:
    """Hash already-read bytes."""
    return hashlib.sha256(raw).hexdigest()


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON members before dictionary construction."""
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON member")
        result[key] = value
    return result


def decode_json(raw: bytes) -> Any:
    """Parse bounded strict JSON, rejecting nonfinite numeric syntax."""
    def invalid_constant(_: str) -> None:
        raise ValueError("Nonfinite JSON constant")

    try:
        return json.loads(raw, object_pairs_hook=strict_object, parse_constant=invalid_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Invalid bounded JSON") from error


def read_bounded(path: Path, limit: int) -> bytes:
    """Read one regular file into a bounded snapshot; never follow a final symlink."""
    require(path.is_file() and not path.is_symlink(), "Missing regular evidence file")
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    require(len(raw) <= limit, "Evidence file exceeds its byte quota")
    return raw


def load_contract() -> types.ModuleType:
    """Execute the exact verified contract snapshot using only the standard library."""
    raw = read_bounded(CONTRACT_PATH, 65536)
    require(digest(raw) == CONTRACT_SHA, "Frozen contract SHA differs")
    module = types.ModuleType("_e0_uid_contract")
    module.__file__ = str(CONTRACT_PATH)
    exec(compile(raw, str(CONTRACT_PATH), "exec", dont_inherit=True), module.__dict__)
    return module


def identity_check(value: Any, rank: int, contract: types.ModuleType) -> None:
    """Validate the fixed external identity schema and frozen execution-input pins."""
    require(bool(EXPECTED_PINS), "Peer execution pins are not frozen; candidate is not runnable")
    require(type(value) is dict and set(value) == IDENTITY_FIELDS, "Identity keyset differs")
    require(value["schema"] == IDENTITY_SCHEMA and value["source"] == contract.SOURCE, "Identity source differs")
    for name, expected in (("rank", rank), ("rank_count", 8), ("step", 1), ("micro", 0), ("steps", 3)):
        require(type(value[name]) is int and value[name] == expected, "Identity integer or step differs")
    contract.integer(value["dp"], 0, 7)
    require(type(value["ep"]) is int and value["ep"] in (1, 4), "Unsupported EP size")
    require(value["mode"] == ("fsdp8" if value["ep"] == 1 else "ep4"), "Identity topology differs")
    require(value["case"] == CASES[value["mode"]], "Unsupported exact case identity")
    require(value["target"] == contract.TARGET and value["coord"] == {"expert": 0, "row": 306, "column": 58},
            "Selected global coordinate differs")
    require(all(type(value["coord"][key]) is int for key in value["coord"]), "Coordinate has noninteger fields")
    require(value["complete"] == COMPLETE, "A complete three-step local capture is required")
    require(type(value["pins"]) is dict and value["pins"] == EXPECTED_PINS, "Runtime input pins differ")


def descriptor_check(value: Any, filename: str, raw: bytes) -> None:
    """Bind a fixed same-directory file to its complete raw bytes."""
    require(type(value) is dict and set(value) == {"file", "bytes", "sha256"}, "File descriptor keyset differs")
    require(value["file"] == filename and type(value["bytes"]) is int and value["bytes"] == len(raw),
            "File descriptor name or size differs")
    require(value["sha256"] == digest(raw), "Complete file SHA differs")


def references(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Enumerate references only from known event fields."""
    kind = event["kind"]
    if kind in ("route", "wgrad"):
        require(type(event["refs"]) is dict, "Native refs must be a dictionary")
    result = list(event["refs"].values()) if kind in ("route", "wgrad") else []
    if kind in ("wgrad", "scaling"):
        result.extend(event[name] for name in ("pre", "post") if event[name] is not None)
    return result


@dataclass
class RankEvidence:
    """Hold bounded immutable sidecar snapshots and strict event records."""

    identity: dict[str, Any]
    events: dict[str, dict[str, Any]]
    raw: bytes
    identity_sha256: str

    def block(self, reference: dict[str, Any]) -> bytes:
        """Return a previously validated slice."""
        start = reference["offset"]
        return self.raw[start:start + reference["bytes"]]


def read_rank(run: Path, rank: int, contract: types.ModuleType) -> RankEvidence:
    """Validate all descriptors, event schemas, quotas and the entire raw-reference union."""
    identity_raw = read_bounded(run / f"e0-identity-rank{rank}.json", 4096)
    identity = decode_json(identity_raw)
    identity_check(identity, rank, contract)
    raw_name, meta_name = f"e0-wgrad-rank{rank}.bin", f"e0-wgrad-rank{rank}.jsonl"
    raw = read_bounded(run / raw_name, contract.RANK_RAW_LIMIT)
    metadata = read_bounded(run / meta_name, contract.RANK_META_LIMIT)
    descriptor_check(identity["raw"], raw_name, raw)
    descriptor_check(identity["metadata"], meta_name, metadata)
    require(metadata.endswith(b"\n"), "Incomplete metadata tail")
    lines = metadata.splitlines()
    require(0 < len(lines) <= 16, "Wrong metadata event count")
    events = {}
    validator = contract.Sink(io.BytesIO(), io.BytesIO(), rank)
    validator.raw_bytes = len(raw)
    order = []
    for line in lines:
        require(len(line) + 1 <= 16384, "Metadata event exceeds bound")
        event = decode_json(line)
        require(type(event) is dict and event.get("kind") in contract.FIELDS, "Unknown event kind")
        kind = event["kind"]
        require(kind != "failure" and kind not in events, "Failure or repeated event cannot pass")
        require(set(event) == contract.FIELDS[kind] | {"schema", "rank", "kind"}, "Event keyset differs")
        require(event["schema"] == "e0-wgrad-coordinate-v1" and type(event["rank"]) is int and event["rank"] == rank,
                "Event schema or rank differs")
        for reference in references(event):
            require(type(reference) is dict and set(reference) == {"offset", "bytes", "sha256"},
                    "Native reference keyset differs")
            start = contract.integer(reference["offset"], 0, len(raw))
            size = contract.integer(reference["bytes"], 1, contract.CHUNK)
            require(start + size <= len(raw), "Native reference exceeds file")
            require(reference["sha256"] == digest(raw[start:start + size]), "Native block SHA differs")
            prior = validator.references.get(start)
            require(prior is None or prior == reference, "Aliased offset has conflicting references")
            validator.references[start] = reference
        require(len(validator.references) <= 32, "Too many raw blocks")
        validator.validate(kind, {key: event[key] for key in contract.FIELDS[kind]})
        events[kind] = event
        order.append(kind)
    require("route" in events and "batch" in events, "Missing route or batch")
    route, batch = events["route"], events["batch"]
    batch_value = {key: batch[key] for key in contract.FIELDS["batch"]}
    require(route["batch"] == batch_value and batch["dp"] == identity["dp"], "Source DP/batch binding differs")
    require(route["mode"] == identity["mode"], "Mode differs from external identity")
    if route["mode"] == "ep4":
        require(all(route["transport"][key] is True for key in
                    ("blocking", "slot_views", "received_squeeze_alias")), "Transport flags are not genuine booleans")
    owns_e0 = identity["mode"] == "fsdp8" or route["ep_rank"] == 0
    expected = ["batch", "route", "wgrad", "scaling", "complete"] if owns_e0 else ["batch", "route", "complete"]
    require(order == expected, "Selected E0 call counts or event order differ")
    require(type(events["complete"]["raw_bytes"]) is int, "Completion byte count is not an integer")
    position = 0
    for start, reference in sorted(validator.references.items()):
        require(start == position, "Raw file has a gap, overlap or unreferenced bytes")
        position += reference["bytes"]
    require(position == len(raw), "Raw file has an unreferenced suffix")
    if owns_e0:
        scaling = events["scaling"]
        expected_scaling = ("AVG", 8, None) if identity["mode"] == "fsdp8" else ("SUM", 2, 0.125)
        require((scaling["op"], len(scaling["group"]), scaling["factor"]) == expected_scaling,
                "Actual scaling contract does not match mode")
    return RankEvidence(identity, events, raw, digest(identity_raw))


def integers(evidence: RankEvidence, name: str, count: int) -> tuple[int, ...]:
    """Decode an exact little-endian int64 routing vector."""
    raw = evidence.block(evidence.events["route"]["refs"][name])
    require(len(raw) == count * 8, "Original routing vector length differs")
    return tuple(value[0] for value in struct.iter_unpack("<q", raw))


def read_run(run: Path, contract: types.ModuleType) -> dict[int, RankEvidence]:
    """Require all eight rank sidecars and consistent source/topology/DP identities."""
    require(run.is_dir() and not run.is_symlink(), "Run directory is not regular")
    expected_names = {f"e0-identity-rank{rank}.json" for rank in range(8)} | {
        f"e0-wgrad-rank{rank}.{suffix}" for rank in range(8) for suffix in ("bin", "jsonl")}
    observed_names = {path.name for pattern in ("e0-identity-*", "e0-wgrad-*") for path in run.glob(pattern)}
    require(observed_names == expected_names, "Missing rank or unknown sidecar filename")
    ranks = {rank: read_rank(run, rank, contract) for rank in range(8)}
    require({e.identity["dp"] for e in ranks.values()} == set(range(8)), "DP slot coverage is not bijective")
    for name in ("case", "source", "mode", "ep", "pins"):
        require(all(e.identity[name] == ranks[0].identity[name] for e in ranks.values()),
                "Run identity is inconsistent")
    require(sum(len(e.raw) for e in ranks.values()) <= contract.RUN_RAW_LIMIT, "Run native budget exceeded")
    return ranks


def native_bf16(raw: bytes) -> tuple[int, float]:
    """Decode a native BF16 element without converting any device tensor."""
    require(len(raw) == 2, "BF16 element size differs")
    bits = struct.unpack("<H", raw)[0]
    value = struct.unpack("<f", struct.pack("<I", bits << 16))[0]
    require(math.isfinite(value), "Nonfinite selected BF16 element")
    return bits, value


def reconstruct(ranks: dict[int, RankEvidence], contract: types.ModuleType) -> tuple[dict[tuple, tuple], list[dict]]:
    """Rebuild actual row order, then bind E0 UIDs to original native X and dY bytes."""
    rows_by_rank = {}
    if ranks[0].identity["mode"] == "fsdp8":
        for rank, evidence in ranks.items():
            route = evidence.events["route"]
            counts = integers(evidence, "counts", 8)
            require(counts == tuple(route["counts"]), "Metadata counts differ from original count bytes")
            rows_by_rank[rank] = contract.fsdp_map(
                route["batch"], integers(evidence, "expert_ids", contract.LOCAL_ROWS),
                integers(evidence, "order", contract.LOCAL_ROWS), integers(evidence, "sources", contract.LOCAL_ROWS),
                counts)
    else:
        groups = {tuple(e.events["route"]["group"]) for e in ranks.values()}
        require(len(groups) == 2 and set.union(*(set(group) for group in groups)) == set(range(8)),
                "Two complete EP groups must partition all ranks")
        for group in groups:
            require(all(tuple(ranks[rank].events["route"]["group"]) == group for rank in group),
                    "EP members disagree on original group order")
            senders = {}
            for rank in group:
                evidence = ranks[rank]
                senders[rank] = contract.sender_map(
                    evidence.events["route"]["batch"], integers(evidence, "send_sources", contract.LOCAL_ROWS),
                    integers(evidence, "send_order", contract.LOCAL_ROWS),
                    integers(evidence, "send_experts", contract.LOCAL_ROWS),
                    integers(evidence, "send_counts", 4), group)
            for rank in group:
                evidence = ranks[rank]
                received = sum(integers(evidence, "recv_counts", 4))
                counts = integers(evidence, "local_counts", 2)
                require(counts == tuple(evidence.events["route"]["counts"]),
                        "Metadata counts differ from original count bytes")
                rows_by_rank[rank] = contract.receiver_map(
                    senders, group, rank, integers(evidence, "recv_counts", 4),
                    integers(evidence, "received_experts", received), integers(evidence, "token_order", received),
                    counts)
    selected = {}
    local_scalars = []
    owners = {rank for rank, evidence in ranks.items() if "wgrad" in evidence.events}
    scaling_groups = {tuple(ranks[rank].events["scaling"]["group"]) for rank in owners}
    require(len(scaling_groups) == 1 and set(next(iter(scaling_groups))) == owners,
            "Scaling group is not the complete actual E0 owner group")
    for rank, rows in rows_by_rank.items():
        evidence = ranks[rank]
        uids = contract.e0_rows(rows)
        if "wgrad" not in evidence.events:
            require(not uids, "E0 native values are missing on an actual owner")
            continue
        wgrad, scaling = evidence.events["wgrad"], evidence.events["scaling"]
        require(len(uids) == wgrad["rows"] == evidence.events["route"]["counts"][0], "E0 row coverage differs")
        inputs = evidence.block(wgrad["refs"]["inputs"])
        gradients = evidence.block(wgrad["refs"]["grad_output"])
        products = []
        for index, uid in enumerate(uids):
            require(uid not in selected, "Duplicate global E0 UID")
            x_raw, dy_raw = inputs[index * 2:index * 2 + 2], gradients[index * 2:index * 2 + 2]
            x_bits, x = native_bf16(x_raw)
            dy_bits, dy = native_bf16(dy_raw)
            selected[uid] = (x_bits, dy_bits, x, dy)
            products.append(x * dy)
        observed = {}
        for prefix, event in (("wgrad", wgrad), ("pre_rs_scaling", scaling)):
            for edge in ("pre", "post"):
                reference = event[edge]
                value = None if reference is None else contract.scalar_f32(evidence.block(reference))
                require(value is None or math.isfinite(value), "Nonfinite captured scalar")
                observed[prefix + "_" + edge] = value
        local_scalars.append({"rank": rank, "source_dp_slots": sorted({uid[1] for uid in uids}),
                              "rows": len(uids), "scaling_group": scaling["group"], "op": scaling["op"],
                              "factor": scaling["factor"], "observed_native_f32": observed,
                              "offline_products_fsum_f64": math.fsum(products)})
    require(bool(selected), "No complete E0 native rows")
    return selected, local_scalars


def compare_runs(left: Path, right: Path) -> dict[str, Any]:
    """Compare one FSDP8 and one EP4 run after strict global UID/native binding."""
    contract = load_contract()
    first, second = read_run(left, contract), read_run(right, contract)
    require(first[0].identity["mode"] == "fsdp8" and second[0].identity["mode"] == "ep4", "Expected FSDP8 then EP4")
    batches = [{e.identity["dp"]: e.events["route"]["batch"]["sha256"] for e in ranks.values()}
               for ranks in (first, second)]
    require(batches[0] == batches[1], "Per-source-DP batch SHA differs")
    lhs, left_scalars = reconstruct(first, contract)
    rhs, right_scalars = reconstruct(second, contract)
    require(set(lhs) == set(rhs), "Global expert0 UID coverage or routing differs")
    summaries = {}
    for name, bit_index, value_index in (("X_column58", 0, 2), ("dY_row306", 1, 3)):
        nonzero = byte_difference = 0
        maximum = 0.0
        squares = []
        maximum_uid = None
        hashes = [hashlib.sha256(), hashlib.sha256()]
        for uid in sorted(lhs):
            a, b = lhs[uid], rhs[uid]
            difference = abs(a[value_index] - b[value_index])
            nonzero += int(difference != 0)
            byte_difference += int(a[bit_index] != b[bit_index])
            squares.append(difference * difference)
            if difference > maximum:
                maximum, maximum_uid = difference, uid
            for hasher, row in zip(hashes, (a, b)):
                hasher.update(json.dumps(uid, separators=(",", ":")).encode())
                hasher.update(struct.pack("<H", row[bit_index]))
        sse = math.fsum(squares)
        summaries[name] = {"elements": len(lhs), "native_byte_differences": byte_difference,
                           "numeric_nonzero": nonzero, "max_abs": maximum, "max_uid": maximum_uid,
                           "squared_error_sum_f64": sse, "rms_f64": math.sqrt(sse / len(lhs)),
                           "canonical_uid_native_sha256": [hasher.hexdigest() for hasher in hashes]}
    return {"schema": "e0-eight-rank-uid-native-comparison-v1", "status": "SELECTED_UID_NATIVE_COMPARISON_COMPLETED",
            "source": contract.SOURCE, "cases": [first[0].identity["case"], second[0].identity["case"]],
            "scope": {"expert": 0, "row": 306, "column": 58, "step": 1, "micro": 0, "training_steps": 3},
            "global_uid_rows": len(lhs), "source_dp_slots": sorted({uid[1] for uid in lhs}),
            "padding_rows": sum(uid[5] == "pad" for uid in lhs), "summaries": summaries,
            "identity_sha256_by_arm_rank": [{rank: e.identity_sha256 for rank, e in ranks.items()}
                                             for ranks in (first, second)],
            "local_scalars_by_arm": [left_scalars, right_scalars],
            "native_X_dY_exact": all(value["native_byte_differences"] == 0 for value in summaries.values()),
            "limits": ["Local scalar scopes differ; no cross-topology scalar equality assertion.",
                       "Offline FP64 products/fsum do not reproduce NPU GMM or collective addition order.",
                       "This does not validate full training-prefix transparency, all gradients or 200 steps.",
                       "SHA-bound identities do not replace independent execution/receipt provenance review."],
            "actual_Torch_NPU_or_collectives_executed": False}


def main() -> None:
    """Read two local runs and exclusively create a compact comparison report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare_runs(args.left, args.right)
    raw = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    require(len(raw) <= 131072, "Comparison report exceeds bound")
    with args.output.open("xb") as handle:
        written = handle.write(raw)
        require(written == len(raw), "Short comparison write")
    print(json.dumps({"status": result["status"], "bytes": len(raw), "sha256": digest(raw)}))


if __name__ == "__main__":
    main()
