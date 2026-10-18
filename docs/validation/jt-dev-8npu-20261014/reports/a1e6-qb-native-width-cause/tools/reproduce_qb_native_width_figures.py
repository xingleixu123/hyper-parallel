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
"""Optional RAM figure replay, retaining frozen sources and explicit SVG formatting."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
import re
from types import ModuleType
from unittest.mock import patch
import xml.etree.ElementTree as ET

from PIL import Image
import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402


def normalize_svg_line_endings(raw: bytes) -> bytes:
    """Remove only line-final spaces/tabs; preserve original line ending bytes."""
    rows = []
    for row in raw.splitlines(keepends=True):
        ending = b"\r\n" if row.endswith(b"\r\n") else b"\n" if row.endswith(b"\n") else b""
        body = row[:-len(ending)] if ending else row
        rows.append(body.rstrip(b" \t") + ending)
    return b"".join(rows)


def _svg_tree(raw: bytes) -> tuple:
    root = ET.fromstring(raw)
    ids = {node.attrib["id"]: f"id{index}" for index, node in enumerate(root.iter()) if "id" in node.attrib}

    def attribute(value: str) -> str:
        if value.startswith("#") and value[1:] in ids:
            return "#" + ids[value[1:]]
        return re.sub(r"url\(#([^)]+)\)", lambda match: "url(#" + ids.get(match[1], match[1]) + ")", value)

    def node(element: ET.Element) -> tuple:
        attributes = tuple(sorted((key, ids[value] if key == "id" else attribute(value))
                                  for key, value in element.attrib.items()))
        text = "" if element.tag.endswith("}date") else element.text
        return element.tag, attributes, text, element.tail, tuple(node(child) for child in element)

    return node(root)


def reproduce_figures(root: Path) -> list[dict]:
    """Execute byte-identical plot scripts with output I/O redirected to RAM only."""
    original_save, original_exists = Figure.savefig, Path.exists
    original_read = Path.read_bytes
    results = []
    specs = (
        ("a1e6-attention0-forward-bisection-v1", "plot_attention0_boundaries_v1.py",
         "attention0-projection-boundaries-v1"),
        ("a1e6-qb-shadow-bisection-v1", "plot_qb_native_width_v1.py", "qb-native-width-shadow-v1"),
    )
    for stage_name, script, stem in specs:
        stage = root / "evidence" / stage_name
        source = stage / "tools" / script
        saved = {str(stage / "figures" / f"{stem}.{extension}"):
                 (stage / "figures" / f"{stem}.{extension}").read_bytes() for extension in ("png", "svg")}
        generated = {}

        def save(figure: Figure, path: Path, *args, **kwargs) -> None:
            key = str(Path(path))
            if key not in saved:
                raise ValueError("Frozen plot attempted an unexpected output")
            buffer = io.BytesIO()
            kwargs["format"] = Path(path).suffix[1:]
            original_save(figure, buffer, *args, **kwargs)
            generated[key] = buffer.getvalue()

        def exists(path: Path) -> bool:
            return False if str(path) in saved else original_exists(path)

        def read(path: Path) -> bytes:
            return generated[str(path)] if str(path) in generated else original_read(path)

        def mkdir(path: Path, *_args, **_kwargs) -> None:
            if path != stage / "figures":
                raise ValueError("Frozen plot attempted an unexpected directory")

        module = ModuleType("public_plot_" + stage_name)
        module.__file__ = str(source)
        exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
        with (patch.object(Figure, "savefig", save), patch.object(Path, "exists", exists),
              patch.object(Path, "read_bytes", read), patch.object(Path, "mkdir", mkdir),
              contextlib.redirect_stdout(io.StringIO())):
            module.main()
        png, svg = (str(stage / "figures" / f"{stem}.{extension}") for extension in ("png", "svg"))
        if generated[png] != saved[png]:
            raise ValueError("PNG byte replay differs; use the recorded plotting environment/font")
        first = Image.open(io.BytesIO(generated[png])).convert("RGBA")
        second = Image.open(io.BytesIO(saved[png])).convert("RGBA")
        if first.size != second.size or first.tobytes() != second.tobytes():
            raise ValueError("PNG pixels differ")
        formatted = normalize_svg_line_endings(generated[svg])
        if normalize_svg_line_endings(saved[svg]) != saved[svg] or _svg_tree(formatted) != _svg_tree(saved[svg]):
            raise ValueError("SVG tree differs after declared formatting/date/automatic-ID normalization")
        results.append({"stage": stage_name, "PNG_bytes_equal": True, "PNG_RGBA_pixels_equal": True,
                        "PNG_dimensions": list(first.size), "SVG_line_rstrip_applied": True,
                        "SVG_normalized_full_tree_equal": True, "outputs_written": False})
    return results
