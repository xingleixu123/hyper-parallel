# PR 1369 `388ec460`：2-rank NPU 生命周期与梯度回归

本包是一次真实、已结束的 NPU 模块验证证据，**不是整网训练或性能结论**。两 rank 使用公共 `fully_shard`、HCCL SUM、FP32、fusion false/true、zero-copy false。独立直接导数复核实际 **28 个 rank-case、72 条 rank backward-phase 观测、4 个首窗清零前快照和 2 个 rank 终态**，全部一致；运行级 child exit 0。72 是跨 rank 的观察条数，不是 72 个独立的双卡训练步。

- 16 个原始两 micro rank-case：两 micro 均参加或 early 仅在 micro0 参加，分别覆盖首 micro sync / no-sync 和两种 fusion。
- 8 个 child-only rank-case：root no-grad forward 初始化后，仅 child 输出触发 backward，input `requires_grad` true/false 和两种 fusion。
- 4 个两窗口 rank-case：每窗口 3 micro，前两次 no-sync、末次 sync，early 仅在首 micro 参与。先记录并核对首窗末的真实状态，再调用实际混合精度 optimizer 的正常 `zero_grad(set_to_none=True)`，进入下一窗口；未执行 optimizer step。

每次 backward 后记录的 root/common/early scheduler state 为 `None`、共享 root flag 为 false；应保留的 no-sync pending 与最终同步梯度逐元素符合计算。**没有观测 callback 实际执行次数**。

## 源与运行身份

- [PR 1369](https://github.com/mindspore-ai/hyper-parallel/pull/1369)：当次完整 PR head [`388ec460a163ae06e18b9189c490cf37da460f8a`](https://github.com/xingleixu123/hyper-parallel/commit/388ec460a163ae06e18b9189c490cf37da460f8a)、base `e0cdf62c9d66f1f9ec48c6f15df87dcf4ce80a5f`，tree `1cc99a8cf4d2bc0615b75beebeefce8467f1294e`。不是旧 `51` overlay；[Git tree 核验](source/git-verification.json) 记录相对 base 六处文件变化，其中生产变化只有 scheduler。
- 实际完整源码是 **2679 项**；[源清单](inputs/source-manifest.json) SHA256 `dd4c52de67f7617e416f67b5ea81de3d405580c1756a8d294a45aae9c5d35256`，scheduler SHA256 `219ce9346d528948dbe0755b50a261db025c87d1a6f207b893c22b8e9de0524d`，真实 [worker](inputs/worker.py) SHA256 `e5dfb59bb7702bb1896620fc878e88251a096db2de79c71b7c4e463d30060324`。原始闭合档案中的 preflight 和 source-after 核验一致；本包不复制完整源码树。
- 当次设备 Ascend 910B3 物理 1、2 对应 logical/rank 0、1；Python 3.12.14、Torch `2.12.0+cpu`、PTA `2.12.0.post3`、CANN 9.2.0。CANN 版本来自运行入口 `ascend_home` 目录名；[派生身份摘要](derived/run-provenance.json) 另记录环境指纹和 `cann_env` 哈希。
- 这是 host-mount + image-root 的 nsenter 执行：实际 worker UID 0、`Seccomp: 0`、有效 capabilities `000001ffffffffff`；`devices` / `memory` 进入任务容器 cgroup，其他控制器观察到宿主分组。见有来源哈希的 [派生安全摘录](derived/security-context.json)，不能将 Docker init 的限制冒充实际 worker 的安全上下文。
- [映射时物理卡1原始SMI](evidence/smi-mapping-card-1.txt) / [物理卡2原始SMI](evidence/smi-mapping-card-2.txt) 分别为本次 rank0/rank1 PID；[放行证明](output/physical-mapping-release.json) 带具体 PID 与设备映射，[导入前卡1](evidence/smi-preimport-card-1.txt) / [卡2](evidence/smi-preimport-card-2.txt) 查询均无进程。这里只保留本任务两卡的短文本。
- 原始 [运行级终态](evidence/execute-terminal.json) 为 exited/0。任务 CID `41ecd76ae759970fdbf4519d011967ab06a5846fced7fde5efeb291dab61fabf` 的 [stop/remove 完成回执](evidence/stop-remove-complete.json) 与 [控制结果](evidence/stop-remove-result.json) 绑定闭合清单；这是本次精确容器的清理记录。

## 证据与离线复核

[output/](output/) 的 40 份 JSON 为原始字节：28 个 rank-case、4 个清零前边界、2 个 rank terminal、2 个 preimport、2 个 device-ready 和 2 个 release。它们包含的是此模块构造的人工输入与 4×4 人工参数/梯度，用于数学验证；不含真实训练样本、正式模型 checkpoint、模型权重或私有 YAML。

[原审计器](audit_results.py) 的相对依赖为同目录 `inputs/worker.py` 与 `inputs/source-manifest.json`。它读取 40 个输出，独立计算直接导数、核验 source key 和 release；本包的 CPU 复跑结果与 [原 ROOT 审计结果](analysis/root-auditor-original.json) 字节相同。

[独立 Fraction 报告](analysis/REPORT.md) 及 [原结果](analysis/independent-math-original.json) 使用精确有理算式独立计算实际输入、初始参数、loss、FP32 pending 和 rank SUM。`analysis/independent_math-original.py` 是封存的原始脚本；它依赖未公开的原始 107 文件目录结构，**不能从本精简包独立运行**。`analysis/independent_math_public.py` 是公开目录适配版，数学和 28 项状态检查的主体保持相同，重新核验本包 40 文件；它明确不声称重新核验未打包的 107 条。两份本地 CPU 复算输出在 `derived/`，属于派生结果。

在包目录下可用系统自带的 Python 3.12 和标准库离线复算，以下两个输出必须是不存在的新路径：

```sh
sha256sum -c SHA256SUMS
python -B audit_results.py --input-dir output --output /tmp/pr1369-audit-new.json
python -B analysis/independent_math_public.py /tmp/pr1369-fraction-new.json
```

这只复算已录制结果，不运行 NPU。若重做 NPU 实验，还须准备与清单逐项一致的完整 `388ec` 源树、匹配环境、隔离双卡、完整的 preimport 与 physical-mapping 门控和安全上下文；[选取的真实 worker argv](derived/run-provenance.json) 是入口记录的结构化摘录，目录中没有可直接重启整个实验的 host 控制器，也不能把它当成无需门控的单条重跑命令。原始 worker 固定 `/work/input/worker.py`、`/work/input/source-manifest.json` 和 `/work/sources/hyper-parallel`；平台、设备、身份和 release 的约束见 worker 源码。

原始闭合清单 SHA256 `ec55fff0d26439135ac5f42d2fe8aac9f7d2d850e829b57e4db4578017aee84e`，私有完整回读当时逐字节核对 **107/107** 项；本包仅发布并可重算其中 40 个 output，另带本任务短 SMI 和终态/清理回执。完整清单和进程观察未随包公开，相关结论是标注来源的历史核验记录，并非由精简包再次完成 107 项全量核验。

## 适用边界

本次没有覆盖激活重计算、grouped、zero-copy true、HSDP partial-only、完整模型训练或原生 checkpoint 全状态保存/恢复，也不报告性能；未统计 callback 次数。完整 `388ec` 树基于 `e0`，虽然 scheduler 与后续 `1a+219` 共用同一字节，其他源码不同，因此不能把这里的模块通过写成 `1a+219` 整网/性能通过。PR 的状态以 PR 页面为准。
