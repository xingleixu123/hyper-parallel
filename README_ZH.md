# PR #230：采用真实基类识别重算 wrapper 的回归记录

当前产品提交：`81f54d2670e7fdca2755ceadd0077cd32af31f11`；基点：`f637ef7ee4d4b416800716f1ebaaae7709b8a06a`。关联 [PR #230](https://github.com/hf1s/hyper-parallel/pull/230)、[issue #229](https://github.com/hf1s/hyper-parallel/issues/229)，总问题为 #202。

## 结论

模块顶部导入真实 `CheckpointWrapper`，通过 `isinstance` 覆盖 FSDP 动态派生类。无需比较 MRO 字符串或增加运行时导入。返回结构、FQN、直接包裹类型及原遍历行为保持。

同一最终回归对基点、上一版 MRO 实现、新实现进行对照。基点定向结果为 `4 failed, 3 passed, 1 subtests passed`，MRO 方案为 `1 failed, 4 passed, 3 subtests passed`，其唯一失败是同模块同名但无继承关系的伪装对象被误计。新实现完整 `tests/ut/model_integration` 为 **62 passed, 4 subtests passed**，退出0；两种初始导入顺序均通过，真实动态类为 `HSDPCheckpointWrapper`；仓库根目录按 `.pylintrc` 运行 pylint，退出0、诊断 `[]`。

生产及测试两个文件的实际 editable 路径和 SHA 均核对；提交前后的 Git tree 一致，PR 相对基点只有一个提交。单提交仅包含生产识别和 CPU 回归两个文件。当前证据没有 NPU collective、训练 loss、显存或吞吐的新验收结论。

## 环境与复现

125 上相同固定镜像 `sha256:5f9e677f439d11ab4e7c888f54b142335a27eee7411cbd2ce894a4798a70146d`，Python3.12.14、Torch2.12.0+cpu、pytest9.1.1、pylint3.3.7。独立容器无设备映射、network none、4CPU/8GiB；所有包使用本任务专用离线 wheel 目录。导入时确认进程组和 CUDA 均未初始化。镜像仍安装 TorchNPU，其无可见设备告警保留于原始日志。

从产品提交 editable 安装后运行：

```bash
python -m pytest -p no:cacheprovider -q tests/ut/model_integration
python -m pylint --output-format=json hyper_parallel/tools/model_integration/integration_validator.py tests/ut/model_integration/test_integration_validator.py
```

两个命令都从仓库根目录执行。完整分阶段步骤见 [run-tests-v3.sh](run-tests-v3.sh)；[输入哈希](input-manifest-v3.json)、[环境](environment-v3.json)、[基点红测](base-ut-v3.log)、[MRO红测](mro-ut-v3.log)、[完整绿测](candidate-ut-v3.log)、[导入顺序](import-smoke.json)、[最终lint](pylint-final.json)、[提交验证](commit-verification.json)。

## 失败及边界记录

首轮挂载的共享离线 wheel 目录为空，安装 pytest 时退出1，未执行用例；改为本任务独立保存的依赖。随后附加的“同一个真实 wrapper 必须返回两个别名”断言失败：底层 `ActivationWrapper.named_modules` 已按 memo 处理共享 wrapper，collector 不应改变这个既有遍历行为。这项额外断言超出本修复边界，已收回；失败日志和测试源码保留，不将该轮算作通过。

最终回归通过后，从仓库外执行的首轮 pylint 未加载项目配置，产生既有复杂度、行宽、受保护访问诊断。回到仓库根目录按原 `.pylintrc` 运行后为 `[]`，生产代码及最终测试源码未再改变；没有为此增加屏蔽项、跳过测试或放宽规则。

旧版 CPU62/2子测试及八卡构建漏计的历史证据仍在原固定快照。本次62/4子测试、导入检查和lint分别以此目录为准，历史独立评审不自动视为当前提交评审。PR 保留草稿，issue 保持开放，等待 Maintainer 与当前 head 的 CI。
