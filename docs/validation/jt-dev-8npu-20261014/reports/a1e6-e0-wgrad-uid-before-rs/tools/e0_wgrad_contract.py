# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
#
# ============================================================================
#
# Experiment-only, fixed-model observation; not a product implementation.

"""Pure identities and bounded storage for the first E0 wgrad coordinate."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import math
import struct
from typing import Any, BinaryIO


SOURCE = "544526f7db8342c607244fd41009b2a74adb84c8"
TARGET = "model.layers.1.mlp.experts.down_proj"
ROW, COLUMN = 306, 58
RANKS, TOKENS, PAD, TOPK = 8, 4096, 8, 2
EXPERTS, HIDDEN, INTERMEDIATE = 8, 5120, 1536
LOCAL_ROWS = (TOKENS + PAD) * TOPK
RUN_RAW_LIMIT = 8 * 1024**2
RUN_META_LIMIT = 1024**2
RANK_RAW_LIMIT = RUN_RAW_LIMIT // RANKS
RANK_META_LIMIT = RUN_META_LIMIT // RANKS
CHUNK = 1024**2


def require(condition: bool, message: str) -> None:
    """Reject an incomplete observation rather than inventing an identity."""
    if not condition:
        raise ValueError(message)


def integer(value: Any, low: int, high: int) -> int:
    """Check a genuine bounded integer, excluding bool and coercion."""
    require(type(value) is int and low <= value <= high, "Noninteger or out-of-range identity")
    return value


def ints(values: Sequence[int], size: int, high: int) -> tuple[int, ...]:
    """Validate one small original integer vector without sorting it again."""
    require(len(values) == size, "Original vector length differs")
    return tuple(integer(value, 0, high) for value in values)


def batch_identity(value: dict[str, Any]) -> tuple[int, str, int]:
    """Use source DP coordinate and a full per-batch SHA, never global SHA alone."""
    require(set(value) == {"dp", "sha256", "micro"}, "Batch identity keyset differs")
    dp = integer(value["dp"], 0, 7)
    digest = value["sha256"]
    require(type(digest) is str and len(digest) == 64 and
            all(char in "0123456789abcdef" for char in digest), "Missing per-DP batch SHA")
    require(value["micro"] == 0 and type(value["micro"]) is int, "Only micro0 is supported")
    return dp, digest, 0


def uid(batch: dict[str, Any], route: int) -> tuple[Any, ...]:
    """Encode real/pad positions and repeated top-k slots distinctly."""
    dp, digest, micro = batch_identity(batch)
    route = integer(route, 0, LOCAL_ROWS - 1)
    token, slot = divmod(route, TOPK)
    return (dp, digest, micro, 0, "pad" if token < PAD else "real",
            token if token < PAD else token - PAD, slot)


def fsdp_map(batch: dict[str, Any], expert_ids: Sequence[int], order: Sequence[int],
             sources: Sequence[int], counts: Sequence[int]) -> list[tuple[Any, ...]]:
    """Validate the original local sort and return all rows in its actual order."""
    ids = ints(expert_ids, LOCAL_ROWS, 7)
    permutation = ints(order, LOCAL_ROWS, LOCAL_ROWS - 1)
    source = ints(sources, LOCAL_ROWS, TOKENS + PAD - 1)
    require(len(set(permutation)) == LOCAL_ROWS, "Original FSDP order is not a permutation")
    counts = ints(counts, EXPERTS, LOCAL_ROWS)
    require(sum(counts) == LOCAL_ROWS, "FSDP expert counts do not cover routes")
    expected_experts = tuple(expert for expert, count in enumerate(counts) for _ in range(count))
    require(tuple(ids[index] for index in permutation) == expected_experts, "FSDP expert grouping differs")
    require(source == tuple(index // TOPK for index in permutation), "FSDP token sources differ")
    return [(ids[index], *uid(batch, index)) for index in permutation]


def sender_map(batch: dict[str, Any], sources: Sequence[int], order: Sequence[int],
               dispatched_experts: Sequence[int], send_counts: Sequence[int],
               group: Sequence[int]) -> list[list[tuple[Any, ...]]]:
    """Split actual sender routes using the recorded destination counts."""
    group = ints(group, 4, 7)
    require(len(set(group)) == 4, "EP group rank list is not unique")
    source = ints(sources, LOCAL_ROWS, TOKENS + PAD - 1)
    permutation = ints(order, LOCAL_ROWS, LOCAL_ROWS - 1)
    experts = ints(dispatched_experts, LOCAL_ROWS, 7)
    counts = ints(send_counts, 4, LOCAL_ROWS)
    require(sum(counts) == LOCAL_ROWS and len(set(permutation)) == LOCAL_ROWS, "EP sender coverage differs")
    require(source == tuple(index // TOPK for index in range(LOCAL_ROWS)), "EP source vector differs")
    rows = [(expert, *uid(batch, index)) for expert, index in zip(experts, permutation)]
    result, start = [], 0
    for destination, count in enumerate(counts):
        chunk = rows[start:start + count]
        require(all(row[0] // 2 == destination for row in chunk), "EP destination/expert split differs")
        result.append(chunk)
        start += count
    return result


def receiver_map(sender_chunks: dict[int, list[list[tuple[Any, ...]]]], group: Sequence[int],
                 receiver: int, recv_counts: Sequence[int], received_experts: Sequence[int],
                 token_order: Sequence[int], local_counts: Sequence[int]) -> list[tuple[Any, ...]]:
    """Apply the explicit all_to_all list-slot contract and the original receiver permutation.

    This returns structural UIDs only. A caller must separately bind actual installed
    API code, original runtime frames, batch hashes and native storage before calling
    any result scientifically exact.
    """
    group = ints(group, 4, 7)
    require(len(set(group)) == 4 and set(sender_chunks) == set(group), "EP sender group is incomplete")
    position = group.index(integer(receiver, 0, 7))
    recv = ints(recv_counts, 4, 4 * LOCAL_ROWS)
    rows = []
    for sender, expected in zip(group, recv):
        chunks = sender_chunks[sender]
        require(len(chunks) == 4 and len(chunks[position]) == expected, "EP receive split differs")
        rows.extend(chunks[position])
    experts = ints(received_experts, len(rows), 7)
    require(tuple(row[0] for row in rows) == experts, "Received original expert IDs disagree")
    permutation = ints(token_order, len(rows), max(0, len(rows) - 1))
    require(len(set(permutation)) == len(rows), "Receiver order is not a permutation")
    counts = ints(local_counts, 2, 4 * LOCAL_ROWS)
    require(sum(counts) == len(rows), "Local counts do not cover received rows")
    ordered = [rows[index] for index in permutation]
    expected = tuple(2 * position + local for local, count in enumerate(counts) for _ in range(count))
    require(tuple(row[0] for row in ordered) == expected, "Actual receiver expert grouping differs")
    return ordered


def e0_rows(rows: Sequence[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """Select expert0 without dropping artificial padding or duplicate route slots."""
    result = [row for row in rows if row[0] == 0]
    require(len(result) == len(set(result)), "Duplicate E0 UID")
    return result


def scalar_f32(raw: bytes) -> float:
    """Decode only an explicitly captured native scalar; do not infer other values."""
    require(type(raw) is bytes and len(raw) == 4, "Scalar must be four native FP32 bytes")
    return struct.unpack("<f", raw)[0]


FIELDS = {
    "batch": {"dp", "sha256", "micro"},
    "route": {"mode", "group", "ep_rank", "batch", "refs", "counts", "transport"},
    "wgrad": {"row", "column", "expert", "rows", "refs", "pre", "post", "versions"},
    "scaling": {"op", "group", "factor", "pre", "post"},
    "failure": {"site", "error_type"},
    "complete": {"status", "raw_bytes", "backend_workspace"},
}


class Sink:
    """Write a fixed per-rank share of the whole-run budget without new collectives."""

    def __init__(self, raw: BinaryIO, metadata: BinaryIO, rank: int) -> None:
        self.raw, self.metadata = raw, metadata
        self.rank = integer(rank, 0, 7)
        self.raw_bytes = self.meta_bytes = 0
        self.closed = False
        self.references: dict[int, dict[str, Any]] = {}
        self.events = 0

    def block(self, data: bytes) -> dict[str, Any]:
        """Persist exactly one bounded native vector/scalar block."""
        require(not self.closed and type(data) is bytes, "Closed sink or nonbytes native payload")
        require(len(self.references) < 32 and 0 < len(data) <= CHUNK and
                self.raw_bytes + len(data) <= RANK_RAW_LIMIT,
                "Native whole-run per-rank quota exceeded")
        offset = self.raw_bytes
        try:
            written = self.raw.write(data)
            require(written == len(data), "Short native write")
        except BaseException:
            self.closed = True
            raise
        self.raw_bytes += len(data)
        reference = {"offset": offset, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        self.references[offset] = reference.copy()
        return reference

    def reference(self, value: Any, size: int | None = None) -> None:
        """Reject forged, nested-extra or unwritten native references."""
        require(type(value) is dict and set(value) == {"offset", "bytes", "sha256"}, "Reference keyset differs")
        offset = integer(value["offset"], 0, RANK_RAW_LIMIT)
        integer(value["bytes"], 1, CHUNK)
        require(value == self.references.get(offset), "Reference does not identify a written block")
        require(size is None or value["bytes"] == size, "Reference byte length differs")

    def validate(self, kind: str, payload: dict[str, Any]) -> None:
        """Validate every nested descriptor before JSON serialization can see it."""
        if kind == "batch":
            batch_identity(payload)
        elif kind == "route":
            mode = payload["mode"]
            require(mode in ("fsdp8", "ep4"), "Unknown routing mode")
            group = ints(payload["group"], 8 if mode == "fsdp8" else 4, 7)
            require(len(set(group)) == len(group) and self.rank in group, "Invalid routing group")
            require(payload["ep_rank"] == (None if mode == "fsdp8" else group.index(self.rank)),
                    "Actual EP rank differs")
            require(mode == "fsdp8" or type(payload["ep_rank"]) is int, "Invalid EP rank type")
            batch_identity(payload["batch"])
            counts = ints(payload["counts"], 8 if mode == "fsdp8" else 2, 4 * LOCAL_ROWS)
            require(sum(counts) <= (LOCAL_ROWS if mode == "fsdp8" else 4 * LOCAL_ROWS), "Row count exceeds bound")
            require(mode != "fsdp8" or sum(counts) == LOCAL_ROWS, "FSDP routing is incomplete")
            names = {"expert_ids", "order", "sources", "counts"} if mode == "fsdp8" else {
                "send_sources", "send_order", "send_experts", "send_counts", "recv_counts", "received_experts",
                "token_order", "local_counts"}
            require(type(payload["refs"]) is dict and set(payload["refs"]) == names, "Route refs differ")
            for reference in payload["refs"].values():
                self.reference(reference)
            proof = payload["transport"]
            require(proof is None if mode == "fsdp8" else type(proof) is dict and proof == {
                "api": "all_to_all", "blocking": True, "slot_views": True, "received_squeeze_alias": True,
                "c10d_sha": "c03488a330d66d9b20fb9480f6c3b1bbfba60e9aeadf18b3eca1c5ca93f88728",
                "logger_sha": "b7b5866e4d3c3847e24f3fe488f187cbd90ac94955a24a98dc10f0d96c99b649"},
                "Actual transport proof descriptor differs")
        elif kind == "wgrad":
            require((payload["row"], payload["column"], payload["expert"]) == (ROW, COLUMN, 0),
                    "Selected wgrad coordinate differs")
            for name in ("row", "column", "expert"):
                integer(payload[name], 0, HIDDEN)
            size = integer(payload["rows"], 1, 4 * LOCAL_ROWS) * 2
            require(type(payload["refs"]) is dict and set(payload["refs"]) == {"inputs", "grad_output"},
                    "Selected vector refs differ")
            for reference in payload["refs"].values():
                self.reference(reference, size)
            if payload["pre"] is not None:
                self.reference(payload["pre"], 4)
            self.reference(payload["post"], 4)
            versions = payload["versions"]
            require(type(versions) is dict and set(versions) ==
                    {"weight", "input", "grad_output", "main_pre", "main_post"}, "Version keyset differs")
            for name, value in versions.items():
                if name != "main_pre" or value is not None:
                    integer(value, 0, 2**63 - 1)
        elif kind == "scaling":
            require(payload["op"] in ("AVG", "SUM"), "Unknown reduction operation")
            require(type(payload["group"]) in (list, tuple) and len(payload["group"]) in (2, 8),
                    "Invalid scaling group descriptor")
            group = tuple(integer(rank, 0, 7) for rank in payload["group"])
            require(len(group) in (2, 8) and len(set(group)) == len(group) and self.rank in group,
                    "Invalid scaling group")
            factor = payload["factor"]
            require(factor is None or type(factor) is float and math.isfinite(factor) and factor == 0.125,
                    "Unexpected gradient scaling factor")
            require((payload["op"], len(group), factor) in (("AVG", 8, None), ("SUM", 2, 0.125)),
                    "Unsupported pre-reduce-scatter scaling contract")
            self.reference(payload["pre"], 4)
            self.reference(payload["post"], 4)
        elif kind == "failure":
            require(payload["site"] in ("getter", "accumulate", "record", "exchange", "scaling", "finish", "scope"),
                    "Unknown failure site")
            name = payload["error_type"]
            require(type(name) is str and 0 < len(name) <= 80 and name.isidentifier(), "Unsafe exception descriptor")
        else:
            require(payload["status"] == "LOCAL_CAPTURE_PENDING_GLOBAL_UID_VERIFICATION" and
                    payload["raw_bytes"] == self.raw_bytes and payload["backend_workspace"] == "UNKNOWN",
                    "Unsupported completion descriptor")

    def emit(self, kind: str, payload: dict[str, Any]) -> None:
        """Accept only a fixed schema; unknown descriptor extras never reach disk."""
        require(not self.closed and self.events < 16 and kind in FIELDS and
                set(payload) == FIELDS[kind], "Metadata keyset or event bound differs")
        self.validate(kind, payload)
        body = {"schema": "e0-wgrad-coordinate-v1", "rank": self.rank, "kind": kind, **payload}
        data = (json.dumps(body, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()
        require(len(data) <= 16384 and self.meta_bytes + len(data) <= RANK_META_LIMIT,
                "Metadata whole-run per-rank quota exceeded")
        try:
            written = self.metadata.write(data)
            require(written == len(data), "Short metadata write")
        except BaseException:
            self.closed = True
            raise
        self.meta_bytes += len(data)
        self.events += 1

    def close(self) -> None:
        """Flush caller-owned streams without closing or deleting foreign storage."""
        if not self.closed:
            self.raw.flush()
            self.metadata.flush()
            self.closed = True


class Journal:
    """Restore only our own hooks and reject a replacement without clobbering it."""

    def __init__(self) -> None:
        self.entries: list[tuple[Any, str, Any, Any]] = []
        self.closed = False

    def patch(self, owner: Any, name: str, original: Any, replacement: Any) -> None:
        """Install a single exact attribute after proving its original identity."""
        require(not self.closed and getattr(owner, name, None) is original, "Unexpected hook owner/identity")
        require(not any(item[0] is owner and item[1] == name for item in self.entries), "Duplicate patch")
        setattr(owner, name, replacement)
        self.entries.append((owner, name, original, replacement))

    def check(self) -> None:
        """Verify every installed hook before an observed action."""
        require(not self.closed, "Closed hook journal")
        require(all(getattr(owner, name, None) is replacement for owner, name, _, replacement in self.entries),
                "A live hook was replaced")

    def close(self) -> None:
        """Uninstall all still-owned hooks even when one attribute was replaced."""
        if self.closed:
            return
        changed = False
        for owner, name, original, replacement in reversed(self.entries):
            if getattr(owner, name, None) is replacement:
                setattr(owner, name, original)
            else:
                changed = True
        self.entries.clear()
        self.closed = True
        require(not changed, "A replaced foreign hook was preserved; this capture is invalid")
