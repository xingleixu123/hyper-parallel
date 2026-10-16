# Copyright 2026 Huawei Technologies Co., Ltd.
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

"""Publish completed396/397 boundary evidence while keeping incomplete training out of results."""
import ast
import hashlib
import json
from pathlib import Path

from stage_frontier258_public import copy

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "公开证据"
RUNS = ["hp-jte213-expertnorm328-preclip397", "mf-native-stat397"]


def main() -> None:
    """Verify exact source and trajectory receipts before copying bounded artifacts."""
    audit = json.loads((ROOT / "evidence/frontier397-native-statistics-comparison.json").read_text())
    if not audit["complete"] or not all(all(row.values()) for row in audit["fidelity"].values()):
        raise ValueError("Boundary fidelity incomplete")
    manifest = json.loads((ROOT / "evidence/frontier397-source-manifest.json").read_text())
    if not all(manifest[key] for key in ["complete", "ast_valid", "local_remote_exact"]):
        raise ValueError("Executed source manifest incomplete")
    for row in manifest["files"]:
        source = ROOT / row["local_path"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Executed source changed")
        ast.parse(source.read_text(encoding="utf-8"))
        remote = row["remote_relative"]
        if remote.startswith("tools-f637/"):
            dest = "scripts/latest-jt-f637/" + source.name
        elif remote.startswith("tools-e213/"):
            dest = "scripts/tools-e213/" + source.name
        else:
            dest = "scripts/" + source.name
        copy(source, PUBLIC / dest)
    for run in RUNS:
        if (ROOT / "runs" / (run + ".exit")).read_text().strip() != "0":
            raise ValueError("Run not completed")
        for suffix in [".yaml", ".json"]:
            copy(ROOT / "configs" / (run + suffix), PUBLIC / "configs" / (run + suffix))
        copy(ROOT / ("run-" + run + ".sh"), PUBLIC / "launchers" / ("run-" + run + ".sh"))
        copy(ROOT / "evidence" / (run + "-launch.json"), PUBLIC / "evidence" / (run + "-launch.json"))
        for suffix in [".exit", ".log", ".dispatch.log"]:
            copy(ROOT / "runs" / (run + suffix), PUBLIC / "runs" / (run + suffix))
        for source in (ROOT / "runs" / run).glob("*"):
            if source.suffix in [".json", ".jsonl"]:
                copy(source, PUBLIC / "runs" / run / source.name)
    for source in (ROOT / "evidence").glob("frontier397-*"):
        if source.is_file() and source.suffix in [".json", ".log", ".ir", ".csv"]:
            copy(source, PUBLIC / "evidence" / source.name)
    copy(ROOT / "evidence/mtp-aux-baseline99-acceptance.json", PUBLIC / "evidence/mtp-aux-baseline99-acceptance.json")
    copy(ROOT / "阶段记录-MTPAux397边界定位.md", PUBLIC / "阶段记录-MTPAux397边界定位.md")
    for name in ["stage_frontier397_public.py", "check_loss_acceptance.py"]:
        copy(ROOT / name, PUBLIC / "scripts" / name)
    readme = PUBLIC / "README.md"
    previous = PUBLIC / "README-历史总loss阶段.md"
    if not previous.exists():
        previous.write_bytes(readme.read_bytes())
    readme.write_text("""# JT单卡200→400：MTP/Aux继续对齐中

当前总loss和原始Aux均已达标；原始MTP仅398步误差0.009670257568359375超限。
总loss首步保留≤0.0005；MTP/Aux首步各≤0.001；三项200步每一步都必须≤0.005。
系数、单卡、输入、完整优化器恢复与第99组训练因素保持，未以改变系数或参考值替代计算。

- [当前边界定位记录](阶段记录-MTPAux397边界定位.md)：395全模型精确，396仅主干共享专家down权重不同。
- [原生完整审计](evidence/frontier397-native-statistics-comparison.json)：两端197步保真通过，396全部46项梯度、平方和及范数精确。
- [三项完整200步基线验收](evidence/mtp-aux-baseline99-acceptance.json)：总loss、Aux通过，MTP398失败。
- [上一阶段详细总报告](总问题-总loss达标.md)及[原README](README-历史总loss阶段.md)：早期问题、99组消融与独立产品PR。

正在另行抓取395/396目标状态，尚未确认396更新内部根因，也没有新增产品修复结论。
本提交中的两组统计观测已退出0；阶段记录提到的定点状态运行不是已完成结果。
不含用户数据、模型或优化器全量值；原生IR/统计和完整标量/hash保留原始字节。
""", encoding="utf-8", newline="\n")
    print(json.dumps({"completed_runs": RUNS, "verified_sources": len(manifest["files"])}))


if __name__ == "__main__":
    main()
