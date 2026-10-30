# Muon 显式 replica count：有界验证证据（草稿）

本包面向 [问题 #1364](https://github.com/mindspore-ai/hyper-parallel/issues/1364) 和 [PR #1366](https://github.com/mindspore-ai/hyper-parallel/pull/1366)，记录一次 CPU 回归和四卡 NPU toy 验证。PR 保持草稿，本目录供独立复核。各包内文件的哈希见 [SHA256SUMS](SHA256SUMS)。

## 问题与修复

Muon 的 `hsdp_replica_count` 显式设为 1 或 2 后，原实现重建参数缓存时仍执行自动拓扑选择；在自动推断为 `None` 的情况下，显式设置会被覆盖。显式 0、−1 也可能在进入既有正数校验之前被覆盖。PR 在构造时记下是否显式传值；只在未显式指定时执行原有自动分支。显式值继续使用既有的正数检查及子组处理。补丁不引入新的 NS 数学、通信次序或动态 owner 迁移。

## CPU red/green

CPU 试验基于官方源码 `93572190d0f07ced830ee12e430459437df19851`，Python 3.12.14、PyTorch 2.12.0+cpu，在未暴露 NPU 的容器中运行 `tests/ut/core/optimizer/test_muon.py`，并核对被测树的实际 import 来源。测试参数的自动拓扑推断使用 mock，Muon 本身真实构造。

| 测试文件 | 原版源码 | 修复源码 |
| --- | --- | --- |
| 初版 `626a407e…fe5` | 4 failed、9 passed | 13 passed |
| 最终版 `4984f002…648a` | 4 failed、9 passed | 13 passed |

初版和最终版各自使用同一测试文件完成 red/green。两次 red 的四个失败均是显式 1、2 未保留和 0、−1 未触发原有校验。最终版还增加 `state_dict` 加载后 reset 的覆盖并调整样式；其 red 在构造处先失败，不能据此声称执行到了后面的 state_dict 检查。机器可读摘要和原始 receipt 的哈希见 [cpu/summary.json](cpu/summary.json)，两个测试版本及生产差异均在 [cpu/](cpu/)。四份 CPU 输出文件是从对应原始 receipt 的 stdout 字段按原文提取，外层 receipt 中的本机路径未随包复制。

如需重新执行，可在一次性检出官方 `93572190...` 并安装匹配的测试依赖后，把 `cpu/test-early.py.txt` 作为目标测试文件运行；对同一检出应用 `cpu/production.diff` 后重跑，再换成 `cpu/test-final.py.txt` 运行。原试验使用 `HYPER_PARALLEL_PLATFORM=torch`、`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 和 `python -B -m pytest -q -p no:cacheprovider --import-mode=importlib tests/ut/core/optimizer/test_muon.py`。应确认实际导入来自被测检出。此说明提供复现条件，不保证其他环境会得到相同结果。

## 实际四卡 NPU toy 验证

环境固定为 Ascend 910B3、Python 3.12.14、PyTorch 2.12.0+cpu、PTA/torch_npu 2.12.0.post3、NumPy 1.26.4、CANN 9.2.0（V100R001C12SPC001B237）。两阶段使用相同镜像与挂载的软件；完整实际版本和 CANN 文件指纹见 [control 预检](npu/control/cpu-preflight.json)、[DCP 预检](npu/dcp/cpu-preflight.json)。预检本身没有导入 torch_npu，真实设备初始化及更新另由四 rank 记录证明。

control 和 DCP 使用同一固定配置、同一冻结源码及同一单文件 overlay：base commit [`1a39572f`](https://github.com/xingleixu123/hyper-parallel/commit/1a39572fef23819a2d2651b4bd53047a05a4892f)，原始完整清单 SHA-256 `b18fef99d3d55649c439d865718db90cfc1d0b1f6742bad1f7c87fd7d9a9a8cb`，仅 `hyper_parallel/core/optimizer/muon.py` 一处更改后的实际完整清单 SHA-256 `f829c3146ec8981b0bc91ba87cc9f6c1cbcd3a3e04a563e7ab72a94b0ce4d4a1`。原文件 SHA-256 `27e81dc93bcbbc4c3a288977ff3cab81f8e27c4f537d466d41c2830566aa4f10`，overlay 文件 SHA-256 `a3e5d62341c61309d8b7fb1261f2b4bb15022448e3430ea514ec5731c86ee265`；清单各有 2661 文件，路径集合一致，只有该文件的记录不同。[原始 pins](npu/actual-pins.json)

这是与上述 CPU 的官方 `93572190...` 不同的源码树。两个树应用的是同一显式配置保留语义，不是同一个 Muon 文件的字节；不能把 NPU 结果称为官方 `935...` 树的直接设备回归。实验使用 row-major `[[0,1],[2,3]]` 的 2×2 mesh、四个 FP32 参数、固定梯度和 10 步；model 与 FP32 master 为同一对象，NS 计算 dtype 为 BF16。具体参数和组配置记录在各 rank JSON 中。

- **Control：** 运行 `None`、显式 1、显式 2 三个配置。独立 CPU 审计报告 PASS，校验 960 个逐步参数/动量数组，最大相对更新差异为 0、最大 momentum 绝对差异为 0。显式 1 的 owner 和 replica 通信与 `None/2` 不同，shard 通信仍存在。显式 2 在此例等于原 replica 组大小 2，未覆盖从更大的原组切分为 2。[原始 audit](npu/control/audit.json) · [四 rank 记录](npu/control/)
- **DCP：** 以 default 第 5 步保存，经原生加载分别恢复到 default 和显式 1，再运行到第 10 步。独立 CPU 审计报告 PASS，校验 704 个逐步参数/动量数组，最大相对更新差异和最大 momentum 绝对差异均为 0。第 5 步恢复值和后续轨迹的检查详见 [原始 audit](npu/dcp/audit.json) 和 [四 rank 记录](npu/dcp/)。

DCP 的 raw momentum 在原生加载完成后、脚本 **额外的 checkpoint state 取证广播之前**被读取和比较；脚本没有用参考值手工修补恢复状态。原生加载阶段自身存在 collectives，不能说此检查发生在任何广播之前。自恢复分支每 rank 记录 4 次 `native_dcp_load` broadcast，显式 1 分支每 rank 记录 2 次；该标签跨多个调用，不能仅凭标签断言每条调用栈。

两阶段均含逐步取证 state 广播/清理；非纯衰减检查要求每个参数每步至少存在一个偏差大于 4 FP32 ULP 的元素。它有助于排除共同纯衰减假通过，但不是独立 NS 数学 oracle，也不能保证检出任意单个小元素的缺失更新。实验未覆盖任意 rank 排列、BF16 分离 master、完整模型训练、吞吐或显存收益。

## 文件及完整性

`npu/worker.py.txt` 和 `npu/audit.py.txt` 保留实际运行/审计脚本的原始字节，SHA-256 分别为 `b8403f7a9492c6510db36d7e7fe37c0affefbb40ab7eab2cfc151adec5e2721b` 和 `8811f77a929e21f6ffefba0c7dbe03ef0fed3888a4744bad38a23bce8ef86d9b`。两个 phase 的 4 份原始 rank JSON、terminal、device-release、原始 audit JSON 与完整回读 inventory 均按字节复制。回读清单记录 control 1161 文件、7,362,643 字节，DCP 947 文件、5,812,152 字节。

完整回读 inventory 列出了实验目录全部文件的文件字节哈希，但本包只提供上述选定文件：**NPY 数值数组、toy 第 5 步 checkpoint、日志及完整源码均未放入本包**，原始数值和 checkpoint 保留在本地与远端归档中。因而仅凭本包不能重跑完整数组审计或确认每个未发布文件的内容；需要原始归档或重新运行。rank JSON 中数组描述符的 SHA-256 是连续数组内容字节的哈希，回读 inventory 的 NPY SHA-256 是整个 `.npy` 文件字节的哈希，二者不可直接互换。

仓内 `reviews/` 的两个 `.txt` 是已完成独立审查的原始字节，保留原报告指向本机封存目录的历史相对链接，部分链接目标不随本包提供；其中 v2 设计审查时间早于实际 NPU 运行。`SHA256SUMS` 列出本包其他每个文件的哈希，不列自身。

## 设备初始化观察

一次失败的旧 DCP 容器中，CPU-only `device_count()` 探测在已闭合的本任务 control 容器移除前返回 0，移除后在同一旧 DCP 容器返回 4；随后移除旧 DCP 容器并启动独立的新 DCP 容器，完成上述验证。未更改设备权限。该前后变化仅是一次相关观察，不能据此确定驱动原因或承诺可复现；旧失败不属于新的 DCP 数值通过结果。

## 源码与入口复现

两份源清单随包提供：[baseline](npu/source-manifest-baseline.json)、[唯一 overlay 后的 actual](npu/actual-source-manifest.json)；它们仅含公开源树的相对文件名、大小和哈希，不包含训练 YAML 内容。实际 [Muon overlay 原字节](npu/muon-overlay.py.txt) 也已保留。复现时应从上述公开 base commit 导出源码，只替换该文件，核对全部 2661 项；不要把 CPU 的 935 源或带 profiler 标记的源树冒充该源。

将原字节脚本分别复制成 worker.py、audit.py，worker 固定读取只读 /work/sources/hyper-parallel、/work/input/source-manifest.json（baseline b18）和 /work/input/worker.py，输出 /work/run。通过匹配软件的隔离四卡环境，以 python -B -m torch.distributed.run --nproc-per-node=4 --no-python python -B /work/input/worker.py 启动；必须明确设置 HP_TASK_ROOT=/work、HP_SOURCE=/work/sources/hyper-parallel、OUTPUT_DIR=/work/run、HP_PHASE=control 或 dcp、HP_DIAG_CID=本次真实容器ID、HP_PHYSICAL_CARDS=本次四卡、PYTHONHASHSEED=0、PYTHONDONTWRITEBYTECODE=1，并选择独立 rendezvous/HCCL 端口。运行前及阶段切换重新核验可用设备与进程，不复用旧 release。

worker 在 HCCL 前发布四份 device-ready，操作者须独立核验宿主 PID/starttick/namespace/cgroup 与真实物理卡映射，再按脚本定义发布本次 device-release；仅通过名称或同 seed 不能放行。control 完成并以 python -B audit.py control OUTPUT 审核后，回收已闭合容器，再在新容器、新输出目录执行 dcp，并审核 audit.py dcp OUTPUT。脚本含固定初值/梯度生成过程和精确 options；此模块不读取 mmap 训练数据，不能替代正式模型的数据和 checkpoint 验收。公共包未提供 NPU 软件 wheel 或完整数值数组，完整复现仍需满足软件、设备与源依赖。

## 上游复查

本轮 CPU 创建基线是 93572190。随后上游 master 更新到 [5cde7c54](https://github.com/mindspore-ai/hyper-parallel/commit/5cde7c54ce3da8be8689d0783cf9b70d5306117a)：相对 935 的 2 个提交仅涉及 11 个 compile/VPP pipeline 文件，optimizer、fully_shard、activation_checkpoint、components/optim、checkpoint 及 trainer/runtime 路径均无差异。本轮 PP=1、graph compiler 关闭，继续保留已有冻结试验身份；不把后来的上游版本称为本轮实跑版本。
