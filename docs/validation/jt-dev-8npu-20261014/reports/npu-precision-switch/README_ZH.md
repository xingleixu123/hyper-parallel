# NPU HF32 高精度开关：独立缺陷与验证证据

关联总单：[#202](https://github.com/hf1s/hyper-parallel/issues/202)。修复提交 `453fe5723429b0e45ee6f0236d6e262cbe58b04f`，基点 `f637ef7ee4d4b416800716f1ebaaae7709b8a06a`，只改生产文件和对应 CPU UT。

旧代码向 TorchNPU matmul 配置对象写入 CUDA 风格的两个无效属性。实际 TorchNPU 2.12.0 对它们静默不处理；HF32 预先开启时，高精度函数调用后仍开启。修复改用 `allow_hf32=False`，并明确该选项不控制 NPU BF16 累加。CUDA 和 CPU 行为保留。

| 验证 | 修复前 | 修复后 |
|---|---|---|
| 实际安装 TorchNPU 的 HF32，先设为 true 再调用函数 | true | false |
| 默认 HF32 读回 | false | false |
| NPU 运行时已初始化 | false | false |
| 同一 CPU UT | 两项 NPU 子测试失败 | 3 测试、2 子测试通过 |

测试使用无设备容器，4 CPU、8 GiB、network none；只强制选择硬件可用性分支，实际 TorchNPU 选项对象未被替换。pytest 的 CUDA/NPU/CPU 单元用例则使用硬件无关替身验证分支策略，两种证据不能混淆。

初版批次的测试退出码分别为 1（旧源码）、0（修复源码），但尾部 JSON 读取因驱动往 stdout 输出诊断文本而失败，整批退出 1。保留原始批次和原 stdout；后续独立 JSON 文件输出的 probe 返回 0、容器已移除，真实开关读回成功。没有伪报初版整批成功。

- [修复前真实选项](npu-precision-option-before.container.log) / [无设备容器凭据](npu-precision-option-before.receipt.json)
- [修复前 UT](npu-precision-ut-before.log) / [修复后 UT](npu-precision-ut-after.log)
- [初版批次原日志](npu-precision-red-green.container.log) / [原执行脚本](npu-precision-red-green-as-run.sh)
- [修复后真实选项 JSON](npu-precision-option-after-verified.json) / [退出0与移除凭据](npu-precision-option-after-v2.receipt.json)
- [规范轴检视](npu-precision-standards-review.json) / [需求轴检视](npu-precision-spec-review.json)
- [基点/修改后 lint 对照](npu-precision-lint-comparison.json)：修改后4项均为继承诊断，新增文件无诊断，不称完整 lint 全绿。
- [远端原始证据与最终源码大小/SHA核验](npu-precision-remote-hash-verification.json)
- [本包所有文件大小与SHA](manifest.json)

源码与测试的副本仅用于复核，独立 PR 保持单提交。当前固定八卡实验源码没有混入此修复。默认 HF32 已关闭，**本缺陷未证明是 EP2/CP2 的200步精度失败根因**；没有本修复的实际设备 matmul、完整训练或性能验收结论。软件版本为 Python3.12.14、Torch2.12.0+cpu、TorchNPU2.12.0，镜像 CANN9.2.0-beta.2。
