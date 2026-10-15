# 当前固定配方 TP2：完整200步、精度失败与资源成本

本轮独立运行 `r129-f637-fp32-tp2-200-r4` 在129完成200步×8rank，退出0，517.637332秒后释放全部设备，无本任务设备PID残留。没有资源冲突中断。**证据有效，训练精度未通过。**

## 固定输入与实际比较对象

- 最新上游复核仍为 `f637ef7ee4d4b416800716f1ebaaae7709b8a06a`，包含PR180；本轮集成源 `18602c4524dc664b81b2bad9a7b624455ce1494f`，clean patch。基线为独立 `r129-f637-fp32-fsdp8-0-200-r4`。
- 两侧镜像 `sha256:5f9e677f439d11ab4e7c888f54b142335a27eee7411cbd2ce894a4798a70146d`，CANN9.2.0-beta.2、Torch/TorchNPU2.12.0、8×910B3、同一Arrow数据。2个主干层+1个MTP、8专家/top2、seq4096、vocab152064、hidden5120、GBS8/MBS1、seed42、dataloader seed1234、LR1e-4恒定。
- FSDP参数BF16、归约FP32；FP32专家权重梯度与Muon NS、FP32主参数保持一致；chunk4096、MTP系数0.3、aux系数0.0001。不是整模型FP32训练。
- TP2实际为DP4×TP2，SP与loss_parallel开启、每优化器步2个微批，另有两条MoE布局规则并保持路由/专家权重TP复制。基线DP8、1个微批。EP/CP/PP均1、重算off。不是仅改一个tp_size字段，也不按微批耗时比较。
- 本轮不写checkpoint；基线step200 checkpoint继续保留。两侧200步输入全局身份、逐步/累计样本和监督token、48项初态摘要以及数值配方检查通过。共同recipe SHA为 `c51ed8ff345759c96f9dce13c47f2531b4a5cdb0c487bc5164d01ba926919af5`。

## 全部rank的逐步误差

| 分量 | 最大绝对差 | 最大误差步 | 首次超过0.005 | 超过门槛的步数 |
|---|---:|---:|---:|---:|
| LM | 0.021448135375976562 | 188 | 111 | 11 |
| MTP，日志加权值 | 0.007400989532470703 | 98 | 45 | 2 |
| MTP，按0.3还原 | 0.02466996510823568 | 98 | 45 | 14 |
| aux，日志加权值 | 0.000015583820641040802 | 178 | 未超过 | 0 |
| aux，按0.0001还原 | 0.15583820641040802 | 178 | 19 | 145 |
| 总loss | 0.021886825561523438 | 188 | 91 | 17 |
| 梯度范数，诊断项 | 0.629734992980957 | 148 | 13 | 171 |

最大值覆盖8×200个观测；“步数”按各步全rank最大差计数。还原分量是日志差值除以固定系数，并非额外保存的加权前张量。梯度结论仅为范数，不代表完整梯度张量比较。LM、MTP、aux首步均已有非零差；日志aux虽然很小，还原后的aux不能判为达标。

![TP2完整200步误差](../../figures/f637-tp2-r4-200-raw-zh.png)

独立规范轴检查20项、174份输入指纹；独立需求轴验证两侧152份原始文件并重新遍历每个rank和step。两轴均确认本报告精度FAIL。比较器完整性findings为空，不是缺输入或缺步导致的失败。

旧第三轮中断在69步，其与本轮共同1–69步的LM、加权及还原MTP/aux、total、grad_norm逐项差0。该检查仅说明标量前缀重复性；没有拼接旧前缀，也不将旧冲突轮次用于性能统计。

## 资源观测与解释

| 口径相同的单次观测 | FSDP8 off | TP2本轮 | 相对变化 |
|---|---:|---:|---:|
| 21–200步，逐步8rank MAX中位秒/步 | 1.155002654 | 1.857553244 | +60.8268% |
| 全200步累计allocated峰值GiB | 22.032196045 | 17.294672012 | 节省4.737524GiB，21.5027% |
| 全200步累计reserved峰值GiB | 27.250000000 | 18.638671875 | 节省8.611328GiB |
| 监督token/s，按窗口token总数/耗时总和 | 16008.406569 | 9957.676842 | -37.7972% |

TP2窗口总耗时333.968660831秒、监督token3325552，仅计一次逻辑数据。rank0在全200步中的51步、计时窗口内46步低于其他TP lane；最大漏报1.213431358毫秒（step5，rank1/3/5/7所在lane）。本轮rank0的allocated和reserved峰值没有低于全rank最大值，不能沿用旧69步轮次的reserved漏报数。资源汇总已用离线全八rankMAX修正。

显存下降方向符合词表/矩阵分片和SP预期；模型中复制的专家与共享模块不具有同等TP切分收益。完整step需两个微批，通信与较小局部矩阵可能抵消计算收益；本轮明确未获得吞吐提升。此解释是结合源码的推断，没有算子profile证明各成本份额。测量包含输入摘要等固定观察开销，回调host计时不含全部日志/保存开销，不是端到端或无插桩吞吐；一次运行不提供重复实验置信区间。精度未达标，资源数字只用于诊断，不推荐该配置为已验收方案。

## 释放、清理与复算

76份文件逐文件SHA匹配，传输包SHA `5ee8c5854c3764a57a040d257d8d6f291e30d41a5f26f07423e7d377ff294f20`；校验后已删除本机与129重复包，保留原始日志、源码、配置及当前checkpoint200。没有停止他人任务。协调器记录的下一次最早启动为2026-10-16 **04:26:53.124037+08**，同时满足monotonic 2972406.953，并须再确认空卡。

在原归档根目录（tools、runs、evidence同级）可用本目录reproduce中的同字节工具复算：

```text
python tools/strict_compare_runs.py runs/r129-f637-fp32-fsdp8-0-200-r4 runs/r129-f637-fp32-tp2-200-r4 1 200 --output evidence/tp2-recheck.json --reference-manifest evidence/f637-fsdp8-r4-complete-evidence-manifest.json --candidate-manifest evidence/f637-tp2-r4-complete-evidence-manifest.json
python tools/all_rank_resources.py runs/r129-f637-fp32-tp2-200-r4 1 200 --output evidence/tp2-resource-recheck.json
```

精度命令预期exit1、status FAIL；资源命令只复算观测，不决定精度。下载复现时按manifest中的原始相对路径恢复数据；基线公开证据位于相邻f637-fsdp8-r4-complete目录。不要覆盖已有证据输出。

[严格比较](f637-tp2-r4-strict-comparison.json) · [完整分析](f637-tp2-r4-complete-analysis.json) · [全rank资源](f637-tp2-r4-all-rank-resources.json) · [需求轴独立审计](f637-tp2-r4-precision-independent-review.json) · [规范/资源轴审计](f637-tp2-r4-resource-independent-review.json) · [原始文件SHA清单](f637-tp2-r4-complete-evidence-manifest.json) · [实际配置](raw/runs/r129-f637-fp32-tp2-200-r4/config.yaml) · [释放凭据](raw/scheduling/r129-f637-fp32-tp2-200-r4.receipt.json)。

该结果补充总问题单#202的Top1和资源漏报问题#225/PR#226。后者仍为原单提交b58fed7，未加入本训练源；其CPU/Gloo修复验证与本轮修复前NPU观测各自独立。跨并行数值根因尚未确认，不据此创建没有修复证据的新PR。
