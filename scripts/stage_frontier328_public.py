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

"""Collect completed expert-normalization evidence in the existing history checkout."""
import ast
import hashlib
import json
from pathlib import Path

from stage_frontier258_public import copy
from sync_frontier217_evidence import RUNS

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "公开证据"


def main() -> None:
    """Reject incomplete runs, changed sources and missing acceptance evidence."""
    audit = json.loads((ROOT / "evidence/frontier328-expert-independent200-audit.json").read_text())
    if not audit["acceptance_complete"] or not all(audit["fidelity"].values()):
        raise ValueError("Independent acceptance has not passed")
    manifest_path = ROOT / "evidence/frontier328-final-source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not all(manifest[key] for key in ["complete", "local_remote_exact", "ast_valid"]):
        raise ValueError("Source manifest is incomplete")
    for row in manifest["files"]:
        source = ROOT / row["local_path"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Source changed: " + row["local_path"])
        ast.parse(source.read_text(encoding="utf-8"))
        copy(source, PUBLIC / row["public_path"])
    recent_runs = RUNS[RUNS.index("hp-jte213-fc1norm-qupboth-preclip330"):]
    for run in recent_runs:
        if (ROOT / "runs" / (run + ".exit")).read_text().strip() != "0":
            raise ValueError("Run is incomplete: " + run)
        for suffix in [".yaml", ".json"]:
            copy(ROOT / "configs" / (run + suffix), PUBLIC / "configs" / (run + suffix))
        launcher = "run-" + run + ".sh"
        copy(ROOT / launcher, PUBLIC / "launchers" / launcher)
        launch_record = run + "-launch.json"
        copy(ROOT / "evidence" / launch_record, PUBLIC / "evidence" / launch_record)
        for suffix in [".exit", ".log", ".dispatch.log"]:
            copy(ROOT / "runs" / (run + suffix), PUBLIC / "runs" / (run + suffix))
        for source in (ROOT / "runs" / run).glob("*"):
            if source.suffix in [".json", ".jsonl"]:
                copy(source, PUBLIC / "runs" / run / source.name)
    for prefix in ["frontier328-", "frontier329-", "frontier330-"]:
        for source in (ROOT / "evidence").glob(prefix + "*"):
            if source.is_file() and source.suffix in [".json", ".log", ".txt", ".ir", ".csv", ".png", ".svg", ".md"]:
                copy(source, PUBLIC / "evidence" / source.name)
            elif source.is_dir() and "cleanup" in source.name:
                for metadata in source.rglob("*.json"):
                    copy(metadata, PUBLIC / metadata.relative_to(ROOT))
    for name in ["completed-cann92-inventory.json", "completed-cann92-inventory.csv"]:
        copy(ROOT / "evidence" / name, PUBLIC / "evidence" / name)
    for name in ["阶段记录-328专家归一化及分阶段定位.md", "阶段记录-329专家更新首差.md", "总问题-306阶段原文归档.md"]:
        copy(ROOT / name, PUBLIC / name)
    phase = PUBLIC / "阶段记录-328专家归一化及分阶段定位.md"
    phase.write_text(phase.read_text(encoding="utf-8").replace(
        "(hp_expert_norm328_factor.py)", "(scripts/hp_expert_norm328_factor.py)"),
        encoding="utf-8", newline="\n")
    previous_phase = PUBLIC / "阶段记录-329专家更新首差.md"
    previous_text = previous_phase.read_text(encoding="utf-8")
    for name in ["compare_frontier329_state.py", "replay_hp_frontier329_muon.py",
                 "replay_mf_frontier329_builtin_dump.py"]:
        previous_text = previous_text.replace("(" + name + ")", "(scripts/" + name + ")")
    previous_phase.write_text(previous_text, encoding="utf-8", newline="\n")
    for name in ["stage_frontier328_public.py", "prepare_frontier328_publication.py", "sync_frontier217_evidence.py"]:
        ast.parse((ROOT / name).read_text(encoding="utf-8"))
        copy(ROOT / name, PUBLIC / "scripts" / name)
    for relative in ["industry/dvm-r27-source/src/vm_aiv.cce", "industry/dvm-r27-source/src/isa.h",
                     "industry/frontier328-mf-current-pynative-muon.py"]:
        copy(ROOT / relative, PUBLIC / relative)
    table = (ROOT / "evidence/frontier328-final-loss-table.md").read_text(encoding="utf-8").strip()
    body = (ROOT / "Issue-总问题-328最终模板.md").read_text(encoding="utf-8").replace("FINAL_LOSS_TABLE", table)
    for prefix in ["https://github.com/xingleixu123/hyper-parallel/blob/EVIDENCE_COMMIT/",
                   "https://github.com/xingleixu123/hyper-parallel/tree/EVIDENCE_COMMIT/",
                   "https://raw.githubusercontent.com/xingleixu123/hyper-parallel/EVIDENCE_COMMIT/"]:
        body = body.replace(prefix, "")
    if "EVIDENCE_COMMIT" in body:
        raise ValueError("Unresolved public evidence link")
    (PUBLIC / "总问题-总loss达标.md").write_text(body, encoding="utf-8", newline="\n")
    readme = PUBLIC / "README.md"
    old_readme = PUBLIC / "README-历史截至306阶段.md"
    if not old_readme.exists():
        old_readme.write_bytes(readme.read_bytes())
    readme.write_text("""# JT单卡200→400：总loss阶段已达标

第99组独立200步已完成：首步绝对差9.5367431640625e-7，全部200步最大0.00276947021484375，
平均0.000021147727966308594，0步超过0.005。当前门槛为总loss首步≤0.001、每一步≤0.005。
按用户约定已停止进一步训练，MTP/Aux专项待决定；原始MTP仍有1步差0.009670257568359375。

- [完整中文总报告](总问题-总loss达标.md)：配置、完整状态恢复、消融优先级、正确性、独立issue/PR、复现和限制。
- [第328步专家归约因果及完整复验](阶段记录-328专家归一化及分阶段定位.md)。
- [99组完整实验清单](evidence/completed-cann92-inventory.csv)：另3组完整teacher干预不计独立验收。
- [第99组200步逐步审计](evidence/frontier328-expert-independent200-audit.json)。
- [329/330原生统计及48项模型](evidence/frontier328-expert-native-statistics-comparison.json)。
- [原总issue完整归档](总问题-306阶段原文归档.md)与[历史README](README-历史截至306阶段.md)：保留早期结论、负面消融和固定证据链接。

![200步全部分项误差](evidence/frontier328-expert-independent200-loss-errors.png)

最后四组使用JT_dev e21302f9＋PR192集成181fab5f，包含PR180；实际Ascend910B3单卡，
HP CANN9.2.0-beta.2、MF CANN8.5.0，MF用户原包未改。所有运行均从同次MF200完整模型和
28份Muon动量、18＋18份AdamW矩恢复。最后一个因素只改变实际专家down输入的归约顺序，
无teacher数据、步号或坐标特判。它是旧归档兼容因素，不是已证明的HP通用数学错误。

目录保留源码、配置、未截断标量指标、日志、原生统计/hash、失败记录和清理回执；
不含用户训练数据、模型或优化器大数组。环境路径需按复现机器映射，初始状态需来自同次MF200。
已清理的328/329边界数组需用对应入口重新生成，不能直接运行依赖这些数组的分析。

`文件校验.json`按实际Git blob字节列出hash。`scripts/latest-jt-f637`对应冻结的旧工具层，
`scripts/tools-e213`对应最新入口；本阶段源路径映射见`evidence/frontier328-final-source-manifest.json`。
本目录为追加式实验历史，不重写旧提交或permalink；数值通过不等于产品PR已合入或CI已绿。
""", encoding="utf-8", newline="\n")
    print(json.dumps({"completed_runs": len(recent_runs), "verified_sources": len(manifest["files"]),
                      "report_characters": len(body)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
