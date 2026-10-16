# PR #178 配置与重算复核

复核日期：2026-10-16。PR：[hf1s/hyper-parallel #178](https://github.com/hf1s/hyper-parallel/pull/178)。基点 `521d3a8a9aa62353327093ab1258ca2c22006e8b`，最终单提交 `0b11a299f182841591151f223481e9d54fd3b73a`。此前 head `27b4d9d3d0eef5a64f22b4312aae4faac68f8e59`。

## Standards

初查发现1项 P2 配置边界问题：模型入口按正值配置关闭 HF checkpoint，但缺少 MLA 替换时配置静默无效。违反项目 code-style 的“Validate inputs at boundaries”。复核整改后无剩余阻塞：模型和 MLA 共用参数校验；builder 在 meta 阶段核对所有原始 trunk/MTP attention FQN；手写普通 attention 遇到正值明确报错。默认省略/0 不强制替换、不因本功能关闭 HF checkpoint。未发现新增分布式硬规则违规或需要额外抽象的代码味道。

## Spec

初查发现1项 P2，与 Standards 指向同一实际问题。#173 要求“整数范围 0..num_heads”“builder 与模型边界拒绝不支持的配置”，此前直接模型可接受非法值，无/不完整替换规则也会静默失效。整改后对应契约均满足；没有发现新的 Spec 问题。测试增量充分必要，无需新文件或额外组合矩阵。

## 全部评审意见的判断

读取了6份 review、7条行内评论、4个讨论线程及全部 issue comments（0条）。4个讨论线程在读取时都尚未标为解决；此处判断不代表 Maintainer 批准。

| 原始评论 | 判断与当前处理 |
|---|---|
| [TP/CP限制太强](https://github.com/hf1s/hyper-parallel/pull/178#discussion_r4278233065) | 担忧合理，但功能定位是 TP=CP=1 的容量路径，直接删除 guard 不正确。配置默认0；启用后同时校验完整替换、dropout和执行条件。 |
| [头并行降低可能损失性能](https://github.com/hf1s/hyper-parallel/pull/178#discussion_r4278342360) | 成立。分块增加投影/attention调用，重算还增加重放；这是容量与吞吐的权衡，不宣传普遍加速。优先选择满足容量要求的较大块。 |
| [硬编码内部重算与外层嵌套](https://github.com/hf1s/hyper-parallel/pull/178#discussion_r4278600211) | 此前实现不够清晰。现在依据实际 Hyper checkpoint 上下文建立内层边界；off且无手写checkpoint时用普通autograd。 |
| [整个attention再包重算是否合理](https://github.com/hf1s/hyper-parallel/pull/178#discussion_r4279409237) | full/selective组合有数值与梯度回归；外层SAC不再缓存内层投影。外层完整重放仍有额外成本，不能称为无代价或通用最优方案。 |

## 实现与影响范围

- 只有 `attention_head_chunk_size` 的正整数才切换按头路径；0/省略保持普通attention。不支持bool、负数、小数、超过总头数或非零dropout。
- 开启必须先把全部主干/MTP attention替换成JT MLA；builder在物化权重前拒绝遗漏，并报告具体FQN。普通attention的直接Python调用也拒绝配置静默失效。
- 要求TP=CP=1、eager、原生Linear、无KV cache且提供RoPE；不扩大TP/CP或compile支持。
- 分块保留Parameter身份和子模块调用、按完整头坐标记录QK最大值，拼接后只执行一次完整o_proj。
- 外层full/selective真正包含attention时，内部采用完整头块重算；selective仍保留外层SAC。无外层checkpoint或no_grad不建立块内checkpoint。
- SAC共享修正适用于所有嵌套Hyper checkpoint：每个区域只管理自己的缓存。非嵌套区域策略不变。该修正不以JT配置开关为门控，不能把“默认关闭”泛化为整个SAC完全没有变化。

## 验证

| 组别 | 当前结果 |
|---|---|
| 旧生产实现＋本轮配置回归 | exit1；非法模型配置、无替换和遗漏MTP未报错，见red.log |
| 模型定向配置/数值/梯度/统计 | 4 passed，58 subtests passed |
| 完整已有builder模块 | 5 passed，9 subtests passed |
| checkpoint/SAC/exclude/wrapper核心回归 | 149 passed，17 subtests passed |
| Trainer checkpoint选择回归 | 18 passed，12 subtests passed |
| 项目Pylint与精确基点比较 | 当前27条，基点27条，无新增；未加suppression |
| git diff --check | 通过 |

四组绿测各用独立Python进程，日志保留全部warning；未新增skip。本轮仅新增一个已有builder文件中的测试方法，扩展原配置方法；整个PR共新增5个测试方法，修改4个已有测试文件（其中统计扩展原用例），没有新测试文件、YAML或运行脚本进入产品PR。58个模型subtests主要来自全部参数及输入/RoPE梯度对照，不是新增58个测试方法。

运行在125固定镜像 `sha256:5f9e677f439d11ab4e7c888f54b142335a27eee7411cbd2ce894a4798a70146d` 的隔离CPU容器：Python3.12.14、Torch2.12.0+cpu、TorchNPU2.12.0、Transformers5.13.1、pytest9.1.1；4CPU、8GiB、network none、无NPU设备映射。导入为 `/work/src/hyper_parallel/__init__.py`。CPU attention使用独立SDPA；builder原有CPU测试只替换NPU grouped kernel边界，不伪造分布式FSDP运行。

lint在本机项目根目录按原 `.pylintrc` 执行，Python3.14.2、pylint3.3.7。基线已有诊断为sac.py 22条、test_sac.py 3条、模型测试2条，其余修改文件无诊断。比较忽略随新增代码移动的行号，保留path/obj/symbol/message及出现次数。

`green-input-manifest.json`绑定实际测试的原始字节；`commit-verification.json`同时记录实测Windows CRLF副本与仓库LF提交的SHA，逐文件核对仅换行规范化且AST一致。提交范围为7个已有Python文件：3个业务文件、4个测试文件，347行增加、11行删除。

本轮没有重新进行NPU/FSDP真实反向、256K峰值或吞吐测量。历史多补丁组合的NPU结果不作为当前提交的验证结果；CPU正确性不能替代这些硬件验收。PR保持草稿，等待Maintainer对方案和当前head的CI复核。
