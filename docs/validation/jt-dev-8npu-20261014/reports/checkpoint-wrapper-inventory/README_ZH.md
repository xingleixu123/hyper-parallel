# FSDP 动态派生重算 wrapper 被集成报告漏计

总问题单：[#202](https://github.com/hf1s/hyper-parallel/issues/202)。修复基点为 `f637ef7ee4d4b416800716f1ebaaae7709b8a06a`，独立单提交为 `b54005154e8d9ae3b2e1728ba106d2d628e6053e`。

## 实际问题

八卡完全重算尝试 `r129-f637-fp32-fsdp8-full-200-r3` 的八份日志均记录安装3个区域，最终模块 inventory 包含两个 `abc.HSDPCheckpointWrapper` 和一个普通 HP `CheckpointWrapper`；原集成报告只计1个。报告函数只比较实例类名，遗漏FSDP动态派生类。该轮因其他任务进入4号卡而释放本任务容器，0个完整优化器步、exit137、115.91秒；这些事实不能用于200步精度或性能结论。

[真实报告、模块类型、八rank日志指纹及归属摘要](checkpoint-inventory-real-run-observation.json)；[中断原始记录](../f637-fsdp8-full-r3-interrupted/public-manifest.json)。其他任务的详细进程归属只保留在本地。

## 独立修复

使用完整模块名和类名检查MRO，识别HP wrapper及其动态子类，排除同名无关类。报告结构、FQN、顺序和直接包裹对象类型保持；函数不新增后端导入，不改FSDP、重算或loss数学。它不修复现有CP2/EP2的200步数值偏离，也没有集成到固定训练源18602c4。

## 复现及回归

同一 `jt-clean:hp-910b-20261009` 镜像，Python3.12.14、Torch2.12.0+cpu、TorchNPU2.12.0、CANN9.2.0-beta.2。容器无设备映射，4CPU、8GiB、network none。单独的editable checkout由基点Git archive展开，测试期间替换的仅为生产修复文件；实际导入路径及最终两文件SHA已核对。容器receipt中的source_mount记录公共固定训练源，实际UT导入路径以日志中 `/work/fixes/checkpoint-wrapper-inventory-cpu` 为准。

- 新增用例在基点：pytest报告 `3 failed, 2 passed, 1 subtests passed`，其中失败为2个测试和1个子测试，exit1。
- 修复后运行完整 `python -m pytest -p no:cacheprovider -q tests/ut/model_integration`：`62 passed, 2 subtests passed`，exit0。
- 定向覆盖FSDP在wrapper外/内的组合、根wrapper、普通模块、同名无关类及报告函数执行期间不新增后端导入。
- 首版测试误把整个包的既有导入也要求为无Torch，子进程超时，整批exit1；修正为函数调用边界后进行第二轮完整红绿对照。初版日志及测试源码仍保留，不能把第一次整批说成成功。
- `pylint`两文件返回0、诊断[]；`git diff --check`通过。规范轴和需求轴分别PASS；CI状态另行按PR当前head核实。

[最终红测](checkpoint-inventory-v2-ut-before.log) · [最终完整绿测](checkpoint-inventory-v2-ut-after.log) · [容器释放凭据](checkpoint-inventory-red-green-v2.receipt.json) · [初版失败](checkpoint-inventory-ut-after.log) · [规范检视](checkpoint-inventory-standards-review.json) · [需求检视](checkpoint-inventory-spec-review.json) · [部署/证据SHA核对](checkpoint-inventory-evidence-verification.json) · [最终lint](checkpoint-inventory-lint-final.json) · [完整文件指纹](manifest.json)。
