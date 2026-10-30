# 上库测试：5c 两 rank CPU/Gloo 红绿与 launcher

production scheduler 与已测 patch `51abc1d50805960b50c334c8f5a5b5c4a5b2a5e0d7984686bae3d2e553499fe2` 字节一致。新 worker SHA `8849b9273fd0fddec9431d704ca1f000fbb19092b423eae9fc85469b2326a895`，launcher SHA `2f4305de72b8090e51b9c0848f29f057e2c9c14335eec477f90702ae00200924`。

| 独立运行 | 结果 |
| --- | --- |
| 5c 原 scheduler + 新 worker | 两 rank 均在 `fusion=False, branch_both=False, sync_each=False` 断言 early.main_grad 缺失；exit 1。pytest 遇首个失败退出，未声称此 worker 的 red fusion=True 已跑。 |
| 5c + 只读 scheduler overlay + 新 worker | 两 rank 各 `1 passed`，完整执行 worker 的 2 fusion × 4 组 controls；exit 0。 |
| green + 原版 `_test_fully_shard_hook_state_machine.py` | 两 rank 各 `1 passed`；normal、non_reentrant、reentrant 三种原有轨迹断言全部通过；exit 0。 |
| green + 新 launcher | 独立 import smoke 返回 `launcher_import_heavy_modules=[]`；现成 `torchrun_case` 两 rank 工作进程日志均 `1 passed`，外层 pytest `1 passed`；exit 0。 |

新测试使用初始 sharded 参数比较、独立普通 PyTorch 图遍历所有 rank/micro 的 SUM oracle、FP32 main_grad 精确对比及 pending 检查；CPU scheduler seam 与现有 state-machine 测试一致。此回归使用 `set_requires_gradient_sync` 直接覆盖待修 setter，Trainer 的策略调用由先前 [四组诊断](../cpu-actual-v1/report.md)覆盖。它不验证四 rank HSDP 的 replica-only partial，也不验证 rank 条件分支不一致。

launcher 顶层没有导入 Torch/HP；`cpu_linux`/`level0`/`allcards`/`essential` 标记与仓库现有 CPU 测试一致。隔离挂载使独立 pytest 未发现仓库配置，显示四条 unknown marker warning；不影响两 rank 运行和断言。helper 子进程沿用仓库实现，通过 `PYTHONDONTWRITEBYTECODE=1` 禁止字节码。

四组均真实运行 CPU 容器，官方源运行前后各验证 2659 文件，容器 inspect 确认 readonly 源与新测试、green 唯一 scheduler overlay、无设备授权和 network=none，四容器退出后均已 `--rm`。未运行 NPU、未修改生产源。每组原始 stdout/stderr/argv/receipt 已按字节回读；[collection-summary.json](collection-summary.json) 索引各组逐文件 hash。launcher 两 rank 的 stdout/stderr 使用 `flat-logs/` 保存，`flat-logs-map.json` 记录原深路径和 hash。

独立的同镜像、只读挂载和环境 [import probe](repo-green-origin-r1/origin-summary.json) 实际记录：`hyper_parallel` 和 `tests.torch.utils` 均来自 `/work/sources/hyper-parallel`，scheduler 同路径且 SHA 为 `51abc1d50805960b50c334c8f5a5b5c4a5b2a5e0d7984686bae3d2e553499fe2`；editable `direct_url.json` 与 `.pth` 均指向此源码路径，`.pth` SHA 为 `1789fa1f2d88f64d1d618aa6a39b62e139c8ee9812c89e00355d176247d3f200`。Python 3.12.14、Torch 2.12.0+cpu、PTA distribution metadata 2.12.0.post3；实际 `source` CANN 后 PYTHONPATH 前置 CANN 目录。probe 退出 0、无设备授权、容器已 `--rm`，源码 2659 项前后相同；其观察是独立进程证据，不是原 ST 工作进程内的 origin 记录。原始证据逐项 SHA 见 [artifact-hashes.json](repo-green-origin-r1/artifact-hashes.json)。

Windows MAX_PATH 令 launcher 的本地日志第一次提取失败，后续从已保存的远端 receipt 按 hash 校验并扁平回读成功，没有重跑。清理这一失败留下的空目录时，自动审批策略拒绝 `Remove-Item`（仅返回 blocked by policy）；空目录保留，未绕过该限制。