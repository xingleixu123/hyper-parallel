# PR #230：精简后的必要回归

当前产品提交：`85b091b9100141ebc161a68a5c4bed5932d44509`，基点：`f637ef7ee4d4b416800716f1ebaaae7709b8a06a`。PR 保持一个提交，只修改 validator 和已有 `tests/ut/model_integration/test_model_integration.py` 两个文件；不新增测试文件。

生产修复按评审建议使用真实 `CheckpointWrapper` 的 `isinstance`。只新增一个回归用例：在同一模型中放置经真实 FSDP 类扩展的 wrapper、普通 wrapper、普通层，以及同模块同名但无继承关系的对象；断言完整结果中的数量、FQN 和直接包裹类型。没有新增根节点变体或导入行为测试。

| 验证 | 结果 |
|---|---|
| 基点生产函数 + 当前唯一新增用例 | `1 failed, 58 deselected`，exit 1 |
| 修复后完整已有测试模块 | `59 passed`，exit 0：原有 58 个 + 新增 1 个 |
| 两个修改文件按项目 `.pylintrc` 执行 pylint | 退出 0，诊断 `[]` |
| `git diff --check` | 通过 |

基点的计数碰巧也是 2，但漏掉动态 wrapper、误计无关对象，完整结果断言会失败。当前测试同时约束这两类错误，避免仅检查数量掩盖缺陷。

CPU 红绿验证在 125 的固定镜像 `sha256:5f9e677f439d11ab4e7c888f54b142335a27eee7411cbd2ce894a4798a70146d` 中执行：Python 3.12.14、Torch 2.12.0+cpu、pytest 9.1.1，4 CPU、8 GiB、network none、无 NPU 设备映射。源码以 editable 方式安装，导入路径和两文件 SHA 已核对。容器最终退出 0。lint 在本机 Python 3.14.2 / pylint 3.3.7 / astroid 3.3.11、项目根目录使用原 `.pylintrc` 执行。

原始输出见 `red.log`、`green.log`、`pylint.json`；`manifest.json` 和 `commit.json` 绑定最终源码 SHA 与提交。`run.sh` 是实际离线执行记录，引用实验挂载目录。常规复现需在可导入本 checkout 的 CPU 环境从项目根目录执行：

```bash
python -m pytest -p no:cacheprovider -q tests/ut/model_integration/test_model_integration.py
python -m pylint --output-format=json hyper_parallel/tools/model_integration/integration_validator.py tests/ut/model_integration/test_model_integration.py
```

早前 62 个测试 / 4 个子测试的版本及导入顺序验证保留在[上一轮不可变快照](https://github.com/xingleixu123/hyper-parallel/blob/0137ab4462bc1190d45ce71ce1063a0398903d20/README_ZH.md)，不是当前精简版测试数量。初次 editable 安装缺少显式 PEP517 选项，在测试前失败；修正安装命令后复用了同一隔离容器，未跳过测试或屏蔽 lint 规则。

本变更仅修复重算覆盖报告识别，不改变训练计算；这些 CPU 结果不构成 NPU 训练、200 步精度、性能或显存收益结论，也不代表 GitHub CI 已通过。
