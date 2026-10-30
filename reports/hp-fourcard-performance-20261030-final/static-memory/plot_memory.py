#!/usr/bin/env python3
"""Offline static Matplotlib summary of sealed per-card allocator peaks."""
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "all-strategy-website-inputs-candidate-v1"
PINS = {
    "SHA256SUMS": "3f9c674ab8c7ce20f4f14d048689da1b8c38d8f11c4f8a345da16ecb5f5db7c6",
    "manifest.json": "be2491f3eeafbe79efc2a8a061a36295f0416dc47f00f842fbddf4b57d807146",
    "groups.json": "41588a6c90e22e22b8919e7ec58f7c1116eb1eaac2d6208ffb3d02f985b092db",
    "memory-peaks.csv": "8b487ff35b571a74e4a31dc56dfd9a85f3e924b9369f82cef7d3eb797e8495eb",
}
CACHE = Path.home() / "AppData/Local/pip/Cache/http-v2"
WHEELS = [
    ("matplotlib", "3.11.2", "4/a/5/8/4/4a584060c042ab187f426fab8cf4667cef0869b76cd3873d7590bb85.body", "ecea603dd2fbf8242fd31a305a8b12a4ece2de28096870c65fdd0d1e35b8d9a6"),
    ("contourpy", "1.4.0", "a/5/2/1/9/a52198591b4b8a098edb3ba3b84047627396ab26a4e4c1f6a41a04de.body", "181bea01bc742734ae672fa00c717d855dd35e1f029536406538c52e7b0cd73d"),
    ("fonttools", "4.67.0", "8/e/2/e/e/8e2ee3d47e5b0e73d8d57ca5e11e0a18c1e453ced2f0d48c51486238.body", "4e2c1586b5b6588a47d02e2588170eefdc996b708f2659c44dbe169bd6fcacb5"),
    ("kiwisolver", "1.5.1", "e/8/9/0/f/e890fe33c7111e7b31991bcbc5a8b1b1b0e0a88679ae30ca12a9b66b.body", "f4167e87b397f273dc2356fcf1eaf50a6bac51e6105f45103ef7129c8efb0255"),
    ("cycler", "0.12.1", "6/c/6/e/e/6c6eeaf6757edbde690577822daacaba826c2b12ce67b57b33e8021d.body", "85cef7cff222d8644161529808465972e51340599459b8ac3ccbac5a854e0d30"),
    ("pyparsing", "3.3.3", "e/b/1/9/1/eb191316105c4d39809788cbc595af2313b33a8242498096126657d4.body", "ece8c00a69cf01b45d0b1dedabb469c90d8caf996d4fda40f147627a122849a4"),
    ("python-dateutil", "2.9.0.post0", "7/d/0/6/d/7d06ddc2f8070ce09a5a2290e2020c460e8ab32b6a2c6b69fc06e979.body", "a8b2bc7bffae282281c8140a97d3aa9c14da0b136dfe83f850eea9a5f7470427"),
    ("six", "1.17.0", "4/0/0/d/b/400db3c9a550719d76958c9c597bfbff6eb98533f684c02cc870de92.body", "4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274"),
]
TITLES = {
    "g01_raf_reverse_125": ("01  RAF / reverse order", "Host 125  |  1a  |  full recompute  |  paired"),
    "g02a_hsdp_fsdp_129_pair1": ("02  HSDP vs FSDP - pair 1", "Host 129  |  1a  |  full recompute  |  paired"),
    "g02b_hsdp_fsdp_129_pair2": ("03  HSDP vs FSDP - pair 2", "Host 129  |  1a  |  full recompute  |  paired"),
    "g03_tp_fsdp_125": ("04  TP2 vs FSDP4", "Host 125  |  1a  |  full recompute  |  paired"),
    "g04_cp_fixed_129": ("05  CP2 and FSDP4 references", "Host 129  |  03ea  |  selective  |  nonconcurrent"),
    "g05_ep2_129": ("06  EP2 and FSDP4 references", "Host 129  |  1a  |  full recompute  |  nonconcurrent"),
    "g06_recompute_125": ("07  Activation checkpoint policies", "Host 125  |  1a  |  independent policy runs"),
    "g07_hsdp_replica_125": ("08  HSDP Muon replica count", "Host 125  |  1a + Muon a3e5  |  paired"),
}
LABELS = {"no_raf": "No RAF", "full_raf": "Full RAF", "hsdp": "HSDP", "fsdp": "FSDP",
          "tp2": "TP2", "fsdp4": "FSDP4", "cp2": "CP2", "fsdp4_reference": "FSDP4 ref",
          "ep2_retest": "EP2 retest", "fsdp_reference1": "FSDP4 ref 1", "fsdp_reference2": "FSDP4 ref 2",
          "full": "Full", "off": "Off", "selective": "Selective", "default": "Default", "explicit1": "Explicit 1"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_data():
    raw = {}
    for name, wanted in PINS.items():
        raw[name] = (SRC / name).read_bytes()
        check(digest(raw[name]) == wanted, "source SHA differs: " + name)
    manifest = json.loads(raw["manifest.json"])
    listed = {x["path"]: x["sha256"] for x in manifest["files_excluding_manifest_and_sums"]}
    check(listed.get("memory-peaks.csv") == PINS["memory-peaks.csv"] and listed.get("groups.json") == PINS["groups.json"],
          "source manifest differs")
    groups = json.loads(raw["groups.json"])["groups"]
    with (SRC / "memory-peaks.csv").open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    check(len(groups) == 8 and len(rows) == 72 and sum(len(g["series"]) for g in groups) == 18,
          "expected 8 groups, 18 series, 72 card records")
    check([g["group"] for g in groups] == list(TITLES), "group order differs")
    result = []
    for group in groups:
        gr = {"group": group["group"], "kind": group["kind"], "runs": []}
        for series in group["series"]:
            matching = [r for r in rows if r["group"] == group["group"] and r["series"] == series["alias"]]
            check(len(matching) == 4, "run does not contain 4 card rows")
            check([r["physical_card"] for r in matching] == ["4", "5", "6", "7"], "card order differs")
            check(all(r["source_run"] == series["source_run"] and r["host"] == series["host"] for r in matching),
                  "group identity mismatch")
            cards = []
            for row in matching:
                alloc = float(row["peak_allocated_gib"])
                reserve = float(row["peak_reserved_gib"])
                check(math.isfinite(alloc) and math.isfinite(reserve) and 0 <= alloc <= reserve,
                      "invalid allocation/reservation data")
                for key, value in (("peak_allocated_bytes", alloc), ("peak_reserved_bytes", reserve)):
                    if row[key]:
                        check(int(row[key]) / 2 ** 30 == value, "byte/GiB values differ")
                cards.append({"physical_card": int(row["physical_card"]), "allocated_gib": alloc,
                              "reserved_gib": reserve, "source_file": row["source_file"],
                              "source_sha256": row["source_sha256"], "source_json_pointer": row["source_json_pointer"]})
            gr["runs"].append({"alias": series["alias"], "run_id": series["run_id"], "host": series["host"],
                               "source_head": series["source_head"], "source_manifest_sha256": series["source_manifest_sha256"],
                               "allocated_bar_gib": max(card["allocated_gib"] for card in cards),
                               "reserved_bar_gib": max(card["reserved_gib"] for card in cards), "cards": cards})
        result.append(gr)
    check(sum(len(g["runs"]) for g in result) == 18, "unexpected run count")
    return result


def extract_wheels(target):
    check(sys.version_info[:2] == (3, 14), "cached binary wheels require Python 3.14")
    check(target.is_dir(), "retained task-owned extraction is missing")
    runtime = []
    for name, version, rel, wanted in WHEELS:
        wheel = CACHE / rel
        data = wheel.read_bytes()
        check(digest(data) == wanted, "cached wheel SHA differs: " + name)
        with zipfile.ZipFile(wheel) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                resolved = (target / info.filename).resolve()
                check(target.resolve() in resolved.parents, "unsafe cached wheel path")
                content = z.read(info)
                if resolved.exists():
                    check(resolved.is_file() and digest(resolved.read_bytes()) == digest(content),
                          "existing wheel extraction differs: " + name + "/" + info.filename)
                else:
                    # Earlier cleanup stopped on a locked extension after deleting
                    # some files. Restore only missing, SHA-pinned cached files.
                    resolved.parent.mkdir(parents=True, exist_ok=True)
                    with resolved.open("xb") as f:
                        f.write(content)
        runtime.append({"package": name, "version": version, "wheel_sha256": wanted, "cache_relative_path": rel})
    return runtime


def render_child(groups, runtime):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import numpy
    import PIL

    check(matplotlib.__version__ == "3.11.2", "unexpected matplotlib version")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.0, "axes.titlesize": 11.5,
                         "axes.labelsize": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "all-strategy-memory-v1"})
    fig, axes = plt.subplots(4, 2, figsize=(15.2, 15.6), sharey=True)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.085, right=0.975, top=0.825, bottom=0.105, hspace=0.61, wspace=0.22)
    fig.text(0.085, 0.970, "Measured NPU allocator peaks by strategy", fontsize=21, fontweight="bold", color="#13273b")
    fig.text(0.085, 0.946, "8 isolated groups  |  18 displayed series (some references reused)  |  72 card records",
             fontsize=11.5, color="#405467")
    fig.text(0.085, 0.925, "GBS 4  /  LBS 1  /  sequence 4096    •    measured steps 1511–1540    •    static Matplotlib summary",
             fontsize=10.7, color="#405467")
    allocated, reserved = "#356D9A", "#C6D8E5"
    shapes = ["o", "s", "^", "D"]
    offsets = [-0.075, -0.025, 0.025, 0.075]
    legend = [Patch(facecolor=allocated, label="Allocated peak"),
              Patch(facecolor=reserved, edgecolor="#7897AD", label="Reserved peak")]
    legend.extend(Line2D([0], [0], marker=shape, color="none", markerfacecolor="white",
                         markeredgecolor="#203749", markersize=5.5, label=f"Card {rank + 4}")
                  for rank, shape in enumerate(shapes))
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.082, 0.895), ncol=6,
               frameon=False, fontsize=9.3, handletextpad=0.45, columnspacing=1.25)
    for ax, group in zip(axes.flat, groups):
        title, subtitle = TITLES[group["group"]]
        ax.set_title(title, loc="left", color="#183247", fontweight="bold", pad=22)
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.6, color="#5A6D7D")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E6ECF0", linewidth=0.8)
        ax.set_ylim(0, 25)
        ax.set_yticks([0, 5, 10, 15, 20, 25])
        ax.tick_params(axis="both", length=0, labelsize=9.0, colors="#34495b")
        x = list(range(len(group["runs"])))
        ax.set_xlim(-0.6, len(x) - 0.4)
        ax.set_xticks(x, [LABELS[run["alias"]] for run in group["runs"]])
        width = 0.31
        for i, run in enumerate(group["runs"]):
            for dx, key, color, edge in ((-width / 2, "allocated", allocated, allocated),
                                          (width / 2, "reserved", reserved, "#7897AD")):
                height = run[key + "_bar_gib"]
                xc = i + dx
                ax.bar(xc, height, width=width * 0.94, color=color, edgecolor=edge, linewidth=0.7, zorder=2)
                for offset, marker, card in zip(offsets, shapes, run["cards"]):
                    ax.plot(xc + offset, card[key + "_gib"], marker=marker, markersize=4.0,
                            markerfacecolor="white", markeredgecolor="#203749", markeredgewidth=0.75,
                            linestyle="None", zorder=4)
                ax.text(xc, height + 0.38, f"{height:.2f}", fontsize=8.2, color="#31495C",
                        ha="center", va="bottom")
        if ax in axes[:, 0]:
            ax.set_ylabel("Peak (GiB)")
    fig.text(0.085, 0.073, "Bars = maximum of the four per-card measured peaks; symbols = each card's recorded peak.",
             fontsize=9.2, color="#34495b")
    fig.text(0.085, 0.055, "Reserved includes allocator cache. Groups are isolated; nonconcurrent references do not establish causal comparisons.",
             fontsize=9.2, color="#34495b")
    fig.text(0.085, 0.037, "Source: memory-peaks.csv SHA256 8b487ff35b571a74…; group / run identities and full hashes in accompanying files.",
             fontsize=8.3, color="#637586")
    fig.savefig(HERE / "allocator-peaks.png", dpi=180, facecolor="white", metadata={"Software": "Matplotlib 3.11.2"})
    fig.savefig(HERE / "allocator-peaks.svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)
    env = {"schema": "local-offline-matplotlib-environment-v1", "python": sys.version.split()[0],
           "matplotlib": matplotlib.__version__, "numpy": numpy.__version__, "pillow": PIL.__version__,
           "renderer": "Agg PNG and Matplotlib SVG", "font": "Matplotlib-bundled DejaVu Sans",
           "offline_cached_wheels": runtime,
           "scope": "only cached wheels; extraction retained due to prior automatic cleanup policy rejection; no downloads"}
    (HERE / "environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")


def render(groups):
    # Reuse the one task-owned extraction after the cleanup policy rejection;
    # never remove it or a parent as a workaround.
    tmp = HERE / ".mpl-offline-mc04sz3o"
    check(tmp.is_dir() and tmp.resolve().parent == HERE.resolve(), "retained runtime path differs")
    runtime = extract_wheels(tmp / "packages")
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp / "mpl-config")
    env["PYTHONPATH"] = str(tmp / "packages")
    env["CODEX_MPL_RUNTIME"] = json.dumps(runtime)
    subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--render-child"],
                   env=env, check=True)


def main():
    if sys.argv[1:] != ["--replace"]:
        for name in ("allocator-peaks.png", "allocator-peaks.svg", "figure-data.json", "environment.json"):
            check(not (HERE / name).exists(), "refusing to overwrite " + name)
    groups = load_data()
    render(groups)
    data = {"schema": "all-strategy-static-memory-figure-v1", "source_sha256": PINS,
            "scope": "8 isolated groups, 18 displayed series, 72 card entries; repeated references are not new runs",
            "bar": "max of four per-card peaks for that run; no mean, stack, subtraction or time-series inference",
            "step_window": [1511, 1540], "unit": "GiB", "groups": groups}
    (HERE / "figure-data.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    load_data()  # assert immutable source inputs remained pinned through rendering
    print("PASS: 8 panels, 18 series and 72 recorded cards; static PNG/SVG created")


if __name__ == "__main__":
    if sys.argv[1:] == ["--render-child"]:
        render_child(load_data(), json.loads(os.environ["CODEX_MPL_RUNTIME"]))
    else:
        main()
