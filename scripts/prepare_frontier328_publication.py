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

"""Finalize completed numerical evidence without rerunning or changing experiments."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"


def read_json(name: str) -> dict:
    """Read a local evidence record."""
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def write_text(path: Path, value: str) -> None:
    """Preserve stable UTF-8 and LF for new publication artifacts."""
    path.write_text(value, encoding="utf-8", newline="\n")


def main() -> None:
    """Require the complete acceptance audit before updating stage status."""
    audit = read_json("frontier328-expert-independent200-audit.json")
    if not (audit["complete"] and audit["acceptance_complete"] and audit["total_loss_passed"]):
        raise ValueError("Complete independent acceptance is required")
    if not all(audit["fidelity"].values()) or audit["steps"] != 200:
        raise ValueError("Experiment fidelity is incomplete")
    stats = read_json("frontier328-expert-native-statistics-comparison.json")
    if not stats["complete"]:
        raise ValueError("Native boundary evidence is incomplete")
    old_body = (ROOT / "Issue-总问题-发布稿.md").read_text(encoding="utf-8")
    expected = "7fe8a10285d97c895cd54be68f5bde396efcade21b2c2a68179aa9b95732c093"
    if hashlib.sha256(old_body.encode()).hexdigest() != expected:
        raise ValueError("Parent body changed before archiving")
    archive = ROOT / "总问题-306阶段原文归档.md"
    if archive.exists() and archive.read_text(encoding="utf-8") != old_body:
        raise ValueError("Existing archive is different")
    write_text(archive, old_body)

    summary = audit["summary"]["expert_down"]
    control = audit["summary"]["control"]["total_loss"]
    total = summary["total_loss"]
    max_improvement = 100 * (1 - total["max"] / control["max"])
    mean_improvement = 100 * (1 - total["mean"] / control["mean"])
    table = ["| 指标 | 首步绝对差 | 200步最大绝对差 | 最大差所在步 | 平均绝对差 | 第400步绝对差 |",
             "|---|---:|---:|---:|---:|---:|"]
    names = {"total_loss": "总loss（当前验收）", "lm_loss": "LM", "mtp_weighted": "MTP加权",
             "mtp_raw": "MTP原始", "aux_weighted": "Aux加权", "aux_raw": "Aux原始"}
    for key, label in names.items():
        row = summary[key]
        table.append(f"| {label} | {row['first']:.12g} | {row['max']:.12g} | {row['max_step']} | "
                     f"{row['mean']:.12g} | {row['last']:.12g} |")
    result_table = "\n".join(table)
    write_text(EVIDENCE / "frontier328-final-loss-table.md", result_table + "\n")

    phase = ROOT / "阶段记录-328专家归一化及分阶段定位.md"
    content = phase.read_text(encoding="utf-8")
    content = content.replace(
        "本阶段的新200步实验正在执行，单次重放成功不作为完整验收。",
        "第99组独立200步现已完成并通过当前总loss验收；不是只凭单次重放宣告成功。"
        "按用户约定停止进一步训练，MTP/Aux专项等待交互后决定。")
    begin = content.index("## 独立200步与优先级")
    end = content.index("## Dump、失败记录和磁盘", begin)
    section = f"""## 独立200步与优先级

第99组`hp-jte213-expertnorm328400`在125卡4、端口31699完成200步并退出0；独立观测组
`hp-jte213-expertnorm328-preclip330`在卡5、端口31700完成130步并退出0。两组均从MF200完整状态恢复。
48项完整轨迹保真检查均通过：实际输入、完整恢复、配置、49冻结来源、既有因素和实际调用保持一致；
只新增实时专家down归约，epsilon仍为1e-7。观测组前130步所有指标精确复现完整组。

{result_table}

总loss首步9.5367431640625e-7≤0.001；全部201–400步均≤0.005，最大0.00276947021484375@398，
0步超限。对照第98组最大0.022772789001464844、平均0.0006631982326507569、3步超限；
本次最大和平均误差分别下降{max_improvement:.2f}%、{mean_improvement:.2f}%。这是同背景单因素的实测收益，
不能与不同背景历史因素线性相加，也不证明该因素单独应用于原始HP就足够。

独立原生统计复核：329、330两步各46项裁剪前梯度、46项平方和、总平方和、全局范数，
以及48项FP32/BF16模型哈希全部精确。此前观察到的328权重首差与330传播差消失。
完整组第一次梯度范数差现在出现在397，但总loss已满足阶段门槛，按约定没有继续追查或启动MTP/Aux专项。
原始MTP最大差0.009670257568359375仍高于0.005，原始Aux最大0.001382586546242237；
两者是当前记录的诊断结果，不扩大本阶段验收，也不隐去原始MTP残差。

优先级结论：专家down归约是第98→99组背景下直接使总loss过线的首要因素；epsilon和普通sum/sqrt候选
在同328完整更新上没有效果，降级；更早fc1和Q-up等因素保留其条件性收益及负面项。没有再增加因素。

[200步逐步及48项保真审计](evidence/frontier328-expert-independent200-audit.json) ·
[329/330原生统计与全模型](evidence/frontier328-expert-native-statistics-comparison.json) ·
[99组完整实验清单](evidence/completed-cann92-inventory.csv)

![200步完整误差曲线](evidence/frontier328-expert-independent200-loss-errors.png)

"""
    content = content[:begin] + section + content[end:]
    content = content.replace(
        "328/329输入、MF200起点和原始共享数据保留。",
        "随后在因果、高精度、邻近输入和完整200步审计完成后，清理328/329的22个临时数组，"
        "逻辑字节5,536,484,096、分配字节5,536,571,392；原元数据和完整哈希保留。"
        "这三批共清理逻辑字节12,230,596,480。MF200起点、离线HP200和原始共享数据保留；"
        "已清理的边界数组需按入口重新生成，不能直接重跑依赖这些数组的离线分析。"
        "[328/329清理回执](evidence/frontier328-completed-boundary-state-cleanup.json)。")
    write_text(phase, content)

    previous = ROOT / "阶段记录-329专家更新首差.md"
    note = ("\n\n## 阶段最终结果\n\n"
            "第99组独立200步总loss已通过：最大绝对差0.00276947021484375，0步超限。"
            "真实首差是328，329继承该差，完整因果、全窗消融、精度判断和清理记录见"
            "[328阶段最终记录](阶段记录-328专家归一化及分阶段定位.md)。"
            "本文件保留定位过程，不把早期运行中状态当作当前状态。\n")
    if "## 阶段最终结果" not in previous.read_text(encoding="utf-8"):
        write_text(previous, previous.read_text(encoding="utf-8") + note)

    state_path = ROOT / "继续运行状态.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    cleanup = read_json("frontier328-completed-boundary-state-cleanup.json")
    if not cleanup["complete"] or cleanup["active_users"]:
        raise ValueError("Cleanup receipt is incomplete")
    if not state.get("frontier328_boundary_cleanup_accounted"):
        for key in ["cleanup_deleted_bytes", "cleanup_deleted_logical_bytes",
                    "cleanup_unique_total_with_hardlinks_bytes"]:
            state[key] += cleanup["deleted_bytes"]
        state["frontier328_boundary_cleanup_accounted"] = True
    state["frontier328_boundary_cleanup"] = {
        "receipt": "evidence/frontier328-completed-boundary-state-cleanup.json",
        "logical_bytes": cleanup["deleted_bytes"], "allocated_bytes": cleanup["deleted_allocated_bytes"]}
    for key in ["active_runs", "active_runs_current", "active_jobs"]:
        state[key] = []
    state["completed_native_full_runs"] = 99
    state["latest_stage"] = "总loss独立200步已达标；停止进一步训练，完成发布后与用户交互"
    state["latest_expert328_full200"] = {
        "run": "hp-jte213-expertnorm328400", "total_loss_passed": True,
        "losses": summary, "all_fidelity": True, "independent_short130_exact": True,
        "native329330_all46_gradients_and48_models_exact": True,
        "audit": "evidence/frontier328-expert-independent200-audit.json"}
    state["next_actions"] = ["完成证据与总issue发布", "与用户交互后再决定MTP/Aux专项，不自动继续训练"]
    state["next_steps"] = state["next_actions"]
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_text(state_path, json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"total_loss_passed": True, "max_improvement_percent": max_improvement,
                      "mean_improvement_percent": mean_improvement,
                      "cleanup_aggregate": state["cleanup_unique_total_with_hardlinks_bytes"]}))


if __name__ == "__main__":
    main()
