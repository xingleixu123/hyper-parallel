# 最终六文件只读复核

基于 `candidate-validation.json`：父提交 `1caf044aa35f4e69ade9938039f587281f14a3be`，候选六文件及各 SHA 见该文件；实查 training SHA `c969d9029c059bce931c0912b212fc62e0cd1e6c09b78cbda3ecc3e47cf1257f`。仅静态审阅和 lint 对照，不修改产品。`git diff --check` 通过。

- `_precision.py`：保留旧 `preserve_fp32_parameters` 入口别名，扩展至控制 buffer； `_apply` 目标 dtype 非 FP32 时用原张量直接转到目标 device+FP32，避免先舍入再回转。meta 与设备路径必须依据相应已归档测试，不以静态阅读宣称任意设备通过。
- `model_builder.py`：按每个模块局部 `fp32_buffer_names` 收集对象身份，dtype 检查支持别名；非保护浮点 buffer 保留原规则。受保护 buffer 的构造模块负责 `_apply` 真正保持 dtype，不是 validator 自动转换。
- `configuration.py`：MTP/路由字段按架构是否出现而非一律强制，布尔型和有限数值按训练配置校验；`chunk_loss_size` 在实际使用 full-vocab 分支前校验。此处是当前训练契约，不是对任意 HF 推理构造入口保证兼容。
- `module_replacement.py`：局部 NPU import 不增加 CPU fallback；明确拒绝 sparse indices 与非零 dropout；非空 SwiGLU 的 partial 关键词校验能识别已锁定 #356：当前公共 GroupedExperts 实际绑定公共 swiglu、dim=-1。该检查没有判断函数身份和 dim，因此它只保障已核实版本；未来替换公共实现时须重新核验，不能把它宣称为通用能力探测。保留的 `Xing4GroupedExperts.forward` 是共享 builder 所需的显式签名边界；不能仅因函数体委派删除。
- `moe.py`：load 使用 int64，只有启用 bias 时才累计，重算期间跳过；整数 `sum-count*E` 的 sign 避免浮点平均舍入；bias 通过 `_apply` 保 FP32 并由 builder 验证。维持 buffer 持久化以保留现有 checkpoint 契约。实际 HCCL int64 及 FSDP buffer 行为以测试日志为准。
- `training.py`：epsilon 说明为文档，chunk size 校验只影响 full-vocab 分支；TP loss-parallel 分支仍走原接口。已知 HF MTP 权重缺口、连续 CP 分片、WORLD QKClip 与指定 #1441/#356 依赖继续按既有范围声明。

静态复核未发现当前锁定组合中可独立确证的新缺陷；不把未测变体视为通过。lint 对照结果另见 `lint-final-comparison.json` 和对应 stdout/stderr；它不是全仓 CI 或 lint 全绿声明。

## 同配置 lint 对照

父版和候选分别为 21 条诊断，按路径、对象、symbol 和 message（不计行号移动）对照：**新增 0、消失 0**；两者 exit code 均为 22。既有项为 `unsupported-binary-operation` 7、`forbidden-backend-import` 6、`non-parent-init-called` 4、`missing-public-docstring` 3、`useless-parent-delegation` 1。部分源于 Python union 类型分析或 Torch 后端规则；逐条原文见 `lint-before-final.json` / `lint-after-final.json`。两个版本使用同一 `.pylintrc`、相同 repo/scripts plugin 解析和相同导入环境；基线/候选文件独立快照，候选六文件 SHA 逐一校验。详情和逐项差异见 `lint-final-comparison.json`。这不是 lint 清零或全仓 CI 通过。初次缺 plugin 的无效试运行已覆盖，不参与比较。
