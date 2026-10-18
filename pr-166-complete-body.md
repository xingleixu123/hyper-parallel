**What type of PR is this?**

/kind feature

**What does this PR do / why do we need it**:

仅从 MF 转换模型权重无法恢复同一次续训。本 PR 为固定旧 MF safetensors 与两层 JT 配方提供模型、AdamW 一二阶矩、Muon 动量及恢复计数的两阶段转换工具，不改变损失或优化器算法。

当前提交 `c9134cb852010bb179686ceea163ac100963f2f3`，目标 JT_dev，原 base `06df6f9986de6508c90050ff0579874710fc3148`，1 commit / 3 files。本次仅补全关联和验证边界，没有修改代码 head。

- `convert_mf_ckpt_to_hp.py`：按参数名称及布局映射生成 model.npz、optimizer.npz、state.json、CONVERSION.json；模型和 optimizer 使用一致的拼接、去交错、专家转置映射。
- `materialize_hp_native_checkpoint.py`：构建指定 Trainer、注入 optimizer state 和数据游标、推进常数 LR 配方的计数，调用原生 CheckpointerCallback 写 DCP；不执行训练。
- README 记录固定形状和 dtype 条件。源层0为 dense、层1为 MoE、一个 MTP；materializer 要求 global_batch_size=1，仅按当前常数 LR、顺序数据游标配方恢复。不宣称任意 MF master、随机 sampler、非恒定 LR 或多卡拓扑通用可用。

源模型/optimizer 的既定契约为 FP32；转换保留数组 dtype，BF16 safetensors 不支持时拒绝。模型/状态保存精度与 forward compute precision 是两个概念。输出目录需预先存在，原生 DCP 实际落盘路径以 callback 为准。

### 使用方法

```bash
mkdir -p /your/hp-offline-state /your/hp-native-checkpoint
python scripts/checkpoint_conversion/convert_mf_ckpt_to_hp.py \
  --ckpt /your/mf-step200.safetensors --out-dir /your/hp-offline-state
python scripts/checkpoint_conversion/materialize_hp_native_checkpoint.py \
  /your/train-4k.yaml --state-dir /your/hp-offline-state \
  --output /your/hp-native-checkpoint --step 200
```

训练 YAML 必须构建同一个 JT 模型，并把 reference_weights 指向转换目录。只运行第一步或只加载 model.npz 不能称为完整续训。

### 合入前尚未完成的保护/验证

当前 load_optimizer_npz 对缺失 key 会 continue，expected 也只统计已找到项，故 assigned==expected 不足以证明全部必需槽存在。需要缺失槽、真实标量/占位、错误 shape/dtype、未消费非占位状态的失败路径；同时明确非恒定 LR/随机游标的拒绝范围和准确源身份元数据。当前真实锚点另经独立 64 数组审计确认完整，不代表工具对任意缺失输入都受保护。

**Which issue(s) this PR fixes**:

Fixes #247

总问题 #217；当前是已提交的工具能力与待完善验收，不随 PR 存在就宣布全部实现/测试已完成。

**Test Plan and Test result：What scenarios were tested, and what were the verification results（Function, performance, reliability, etc.）**：

本次完成固定源码与提交覆盖核对，没有重新运行当前 head 的转换工具。该 PR 没有提交 UT/ST 文件，未勾选测试完成。

后续单卡续训第一次前向前独立验证了 48 个模型数组、28 个 Muon 动量、18 个 AdamW 一阶矩和 18 个二阶矩，shape/dtype/值精确，step/scheduler/数据游标均200。它证明该次锚点恢复完整，不能冒记为本 PR 当前 head 全部测试通过。

[实际恢复记录](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/runs/hp-archive200-r2/RESTORE.json) · [独立审计](https://github.com/xingleixu123/hyper-parallel/blob/3f1afbb1ff6b8a4d4c36cd1880a64d1e0b982410/evidence/clean-final-validation.json)。

合入前应对该固定 head 做转换→物化→重新加载的独立回归及缺失/错误输入反例。CI、Maintainer 方案评审与官网文档状态没有完成证据，保留未勾选；包内 README 已有，不能代替官网资料评审。

**Self-checklist**:（**请自检，在[ ]内打上x，我们将检视你的完成情况，否则会导致pr无法合入**）

- [ ] **设计**：PR对应的方案是否已经经过Maintainer评审，方案检视意见是否均已答复并完成方案修改
- [ ] **测试**：PR中的代码是否已有UT/ST测试用例进行充分的覆盖，新增测试用例是否随本PR一并上库或已经上库
- [ ] **验证**：PR描述信息中是否已包含对该PR对应的Feature、Refactor、Bugfix的预期目标达成情况的详细验证结果描述
- [ ] **接口**：是否涉及对外接口变更，相应变更已得到接口评审组织的通过，API对应的注释信息已经刷新正确
- [ ] **文档**：是否涉及官网文档修改，如果涉及请及时提交资料到Doc仓

