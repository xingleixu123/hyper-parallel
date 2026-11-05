# #1439 最新检视意见答复建议与候选复核

以下是本地建议，尚未发送或 resolve；依据 1439 最新快照和 `f96dd5b` 上的六文件工作区候选。原有 reviews、评论和线程状态保留。最终提交后须重新核对 SHA 与实际门禁。

## 建议逐条答复

**4475845264 — Muon 原生 hook**  
感谢建议。已改为向框架 Muon 叶优化器注册原生 `register_step_post_hook`，不再使用 ChainedOptimizer 扩展；本 builder 按 AdamW、Muon 顺序执行，Q/K 仅归 Muon，混精 wrapper 在 chain 返回后回写 FP32 主权重。#1454 / #1455 不再是本 PR 前置；它们涉及通用 chain callback，状态维持原样，不因此关闭。CPU 已验证真实 builder、三步重绑及首步独立缩放公式；尚不以此宣称 NPU 或多卡已过。

**4475905678 — swap 兼容**  
接受风险。Trainer 安装 swap 前本模型已注册 Muon hook；通用 swap 构造器在进行 offload 前检查既有 leaf step pre/post hook 并明确拒绝，避免此组合静默训练。通用 late registration、global hook 及完整 swap hook 语义另行跟踪，当前不声称全支持。CPU 覆盖 dispatcher 与 Muon/Adam 直接入口、已有 hook 拒绝及无 hook 正常 step；真实 NPU swap 尚未验证。

**4475925241 — 参数归属**  
接受建议。启用 QKClip 时，在构造阶段逐个检查所有 attention 的 q_b_proj / kv_b_proj 可训练且其参数身份属于 Muon；无 attention、冻结或归 AdamW 时显式报错。FP32 main 重绑发生在此后，hook 缩放 main copy，CPU 回归检查重绑及回写。该契约要求所有这些投影都由 Muon 管理，不能宣称支持任意混合归属。

**4476138012 — MLA 展开抽取**  
抽出 `_expand_key_value` 是为了让公共 MLA 与 Xing 的 RoPE 定制路径共享 KV 投影、reshape 和拆分，Xing 仅覆写必要的旋转精度与拼接。公共原 forward 同样调用此方法；该边界避免复制实现，未把 Xing 策略写入公共方法。欢迎对扩展点位置继续讨论。

**4476142089 — CP 输入职责**  
当前 master 的 TextParallelBatch 已负责 CP 分片并提供当前分片的 shift_labels；它没有提供 MTP 每个深度的跨片 next-token。这里的 method replacement 只补分片边界，并在全局序列末尾补 pad。可以把这类位移上移为通用 batch 能力，但现有接口直接删除此交换会丢每片边界；需要多深度及片长小于深度的等价验证。

**4476144198 — CP loss 职责**  
Trainer 已负责 loss_fn 和 microbatch/token 权重。这里每层 MTP 的有效 token 数不同，因此先分别跨 CP 求 numerator/count 并归一化，再交 Trainer 现有后续处理；master 现有单 `loss_sum`/`valid_token_count` 协议不能直接表达多个深度的不同分母。若转入 Trainer，应先提供多深度 loss 协议并验证 CP/DP 梯度归一化，不应叠加重复归约。

## 六文件差异复核

- `core/optimizer/swap_optimizer_base.py`：构造首部拒绝已有 pre/post hooks，发生在 runtime 创建和 offload 之前；四入口 CPU 验证覆盖。仅构造期检查，不能覆盖之后新注册或全局 hooks；多 leaf Trainer 逐一包装可能在较晚 leaf 拒绝时已包装先前 leaf，不能声称完全无副作用。
- `distributed/context_parallel/mla_context_parallel.py`：沿正常和 overlap 的 Q/K 路径透传自定义 rotary，attention 回调收到局部 head offset/总 head 数。公共默认 attention 接口接受 **kwargs；实际 NPU 及其他 family 数值仍须由集成回归证明。
- `models/deepseek_v3/adapter/distributed/context_parallel.py`：复用原 forward 包装，扩展可选 rotary/attention 参数，默认仍用公共原接口；保持 cache 拒绝、shape、o_proj 和 companion 重写契约。
- `models/xing_4/adapter/conversion/module_replacement.py`：显式拒绝非 causal attention，避免只用 causal kernel 时静默误算。
- `models/xing_4/adapter/distributed/context_parallel.py`：保留现有 CP model 边界和 MTP shift/loss，透传 strategy、overlap、ulysses_degree、kv_exchange 给公共 plan；不得把接口增加当作所有组合已验证。
- `models/xing_4/adapter/optim/muon.py`：改为 Muon leaf 原生 hook，并以训练状态及身份验证两个投影；不依赖 chain hook。以上静态审阅不替代并行、性能与长程验证。`git diff --check` 通过。

## 证据与限制

[完整 CPU 输出](regression/hook_probe.stdout) `exit_code=0`，stdout SHA256 `e4edf4a4c5633f87cc95a6d7ff8a21b23997ac56b18b53c3a5c539e765eb9021`，Torch `2.12.0+cpu`。两个 hook 源文件的 SHA256 已在日志中输出；其他四文件的候选 SHA256 如下：

```
1006e044794f085cd9e0f484cae25021019613ac4c30f13271db400f2ccf1fc5  distributed/context_parallel/mla_context_parallel.py
e7b0f5aa9f108cca72999c095ba51fa5c75ee5d59ca65108464dbe8ef89e5698  models/deepseek_v3/adapter/distributed/context_parallel.py
faa3ad2759e815aa58682044389ab5894f6a45ee1f417d132c0f4ecfcd18b496  models/xing_4/adapter/conversion/module_replacement.py
1f0bc66e792422b887c2c50e7f4888a7fd7fb57ea414a87fbf04568129b2d743  models/xing_4/adapter/distributed/context_parallel.py
```

本地 fake-adapter 实验用 PyTorch 私有 `_patch_step_function` 能触发 wrapper hook，但 callback 接收 wrapper、类级 step 被修改；未经真实 swap/HSDP/NPU 验证，不纳入产品修复。

`upstream-reuse.json` 的 hook 查询报告 108 条，仅返回前 100；其他 cp、townwish、jt 查询分别返回 42、0、92。已返回结果及当前 master 中未发现可直接解决 swap 绕过 leaf hook 的现成修复；这不是对未返回结果或全 GitHub 的穷尽证明。#1454 / #1455 解决的是 chain hook，当前 Muon leaf 路径不再依赖；其状态均保持 open，不执行关闭。

真实 CPU swap 重现见 [原版对照日志](regression/swap_original_repro.stdout)：master a9fa 与 c75 的 swap 文件 SHA256 均为 `752d19acd1a84f52fa10c93d3878bc8136b5146bd880d8d6b5ad6f89e24de3bb`。AdamW 和 Muon 各自的裸 leaf pre/post 调用数为 1/1；原 swap 实际更新参数但调用数为 0/0；候选构造器对已有 hooks 明确拒绝。该证据仅针对已有 hook，不涵盖 late/global hook、NPU 或 HSDP。
