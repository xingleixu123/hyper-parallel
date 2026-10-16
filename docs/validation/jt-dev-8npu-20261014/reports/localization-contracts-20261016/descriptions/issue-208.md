总问题单：[#202](https://github.com/hf1s/hyper-parallel/issues/202)。本单只跟踪一个独立问题；对应 PR 和验证边界见下文。

## 检查清单

- [x] 已检索 hf1s/hyper-parallel 的全部现有 issue，并核对相关问题的范围。
- [x] 已阅读本问题涉及的源码及回归接口。
- [x] 已保留可执行复现/回归用例和实际结果。

## 问题描述与影响

优先级：本轮 Top7。Trainer恢复optimizer step与数据位置后，环境meter仍是新初始化状态。checkpoint200续训的step201记录累计sample=8、监督token=6304，而独立连续参考应为1608/3720095。训练loss和逐步计数正确，错误发生在累计进度上报。

## 预期行为与验收标准

完整训练恢复应保存/加载meter state，step201–400累计sample/token与连续400参考逐步一致。兼容无meter旧checkpoint及没有meter的Trainer；weights-only恢复不应用旧监控状态。旧checkpoint缺失历史计数时不能靠step数捏造token历史。

## 复现步骤、证据与修复边界

复现流程：完整训练0–200→保存checkpoint→新进程恢复201–400；对照独立连续400步的累计与逐步计数。PR定向CPU入口：

```bash
pytest -q tests/ut/trainer/test_checkpoint_callback.py
```

定向CPU8 passed、6 subtests，真实文件save/load，CPU测试mock rank/barrier/device RNG边界，不冒充多卡checkpoint。真实8卡续训中，旧实现2种累计计数×200步×8rank共3200项不一致；修复后全部一致。基线与续训的LM/MTP/aux/total loss、grad norm均最大误差0，输入/LR/覆盖检查通过。

commit `bd18261a14fc0608d0728defef756fd0028f80aa`。

![累计计数对照](https://raw.githubusercontent.com/xingleixu123/hyper-parallel/ee310ea9e02d4d97e30037450247d8092f540805/docs/validation/jt-dev-8npu-20261014/figures/resume-counters-zh.png)

独立 PR：[#197](https://github.com/hf1s/hyper-parallel/pull/197)。每个 PR 一个 commit；本 issue 不使用自动关闭关键字。

[完整记录及按影响排序的报告](https://github.com/xingleixu123/hyper-parallel/blob/66464cdd569d864562578eb45e9aa234d81770f4/docs/validation/jt-dev-8npu-20261014/REPORT_ZH.md)；[配置、曲线与原始标量索引](https://github.com/xingleixu123/hyper-parallel/blob/66464cdd569d864562578eb45e9aa234d81770f4/docs/validation/jt-dev-8npu-20261014/README.md)。

实验已获准在 125/129 空闲时继续，每轮设备释放后至少间隔 20 分钟，不停止其他人的实验。问题单保持开启；草稿 PR 不等于已合入或完整 CI 通过。

## 18602c4历史配方：基线与续训逐步对齐独立参考（2026-10-15）

本轮基点为包含#180及#186的JT_dev `f637ef7ee4d4b416800716f1ebaaae7709b8a06a`，固定集成源码`18602c4524dc664b81b2bad9a7b624455ce1494f`，CANN9.2.0-beta.2。按FSDP8 0–200→保存checkpoint200→新进程200–400的顺序运行，再以第三个独立容器完整连续0–400作为参考。三次运行全部退出0并释放设备。

两个窗口分别覆盖200步×8rank，各1600个观测：LM、日志MTP/aux、按实际系数0.3/0.0001还原的未加权MTP/aux、总loss、梯度范数的最大绝对差均为0，全部满足≤0.005。还原值来自日志标量和实际配置系数，不冒充直接采集加权前张量。逐步及累计样本/token、LR、输入全局哈希和初态检查全部一致；step201累计1608样本/3720095监督token，step400累计3200样本/7467738监督token。

严格比较另外绑定完整rank/step、起止标记、源码/补丁/镜像/数值配方、两个不同容器身份以及逐文件大小/SHA清单。规范轴和验收轴独立复算通过。多项修复共同参与本轮集成，本结果证明该固定配方的恢复和计数一致性，不把整轮结果全部归因于本PR，也不替代仍待运行的TP/CP/EP和重算矩阵。

[基线严格比较](https://github.com/xingleixu123/hyper-parallel/blob/f272a0fbd1ce105160b37077a8977d59bbdca634/docs/validation/jt-dev-8npu-20261014/reports/f637-fsdp8-continuous400-r3-complete/f637-fsdp8-baseline-independent-comparison.json) · [续训严格比较](https://github.com/xingleixu123/hyper-parallel/blob/f272a0fbd1ce105160b37077a8977d59bbdca634/docs/validation/jt-dev-8npu-20261014/reports/f637-fsdp8-continuous400-r3-complete/f637-fsdp8-resume-independent-comparison.json) · [原始8rank证据目录](https://github.com/xingleixu123/hyper-parallel/blob/f272a0fbd1ce105160b37077a8977d59bbdca634/docs/validation/jt-dev-8npu-20261014/reports/f637-fsdp8-continuous400-r3-complete/) · [两轴检视](https://github.com/xingleixu123/hyper-parallel/blob/f272a0fbd1ce105160b37077a8977d59bbdca634/docs/validation/jt-dev-8npu-20261014/reports/f637-fsdp8-continuous400-r3-complete/f637-fsdp-independent-two-axis-review.json) · [完整中文报告](https://github.com/xingleixu123/hyper-parallel/blob/f272a0fbd1ce105160b37077a8977d59bbdca634/docs/validation/jt-dev-8npu-20261014/REPORT_ZH.md)。

![最新固定配方续训201–400误差对齐；MTP和aux按系数还原](https://raw.githubusercontent.com/xingleixu123/hyper-parallel/f272a0fbd1ce105160b37077a8977d59bbdca634/docs/validation/jt-dev-8npu-20261014/figures/f637-fsdp8-resume-independent-zh.png)

同窗口去掉前20步，基线/独立参考中位步时1.155003/1.157494秒，续训/独立参考1.160107/1.157457秒。约0.2%的单次差异不足以认定性能变化。allocated峰值均22.032196 GiB；reserved基线27.25、续训25.316406、连续参考27.25 GiB，但峰值从各进程启动累计，不能据此认定续训节省内存。回调区间步时包含相同输入身份观测，未涵盖全部落盘和checkpoint I/O，不等于端到端吞吐。

PR仍保持单提交草稿、问题保持开启；未宣称维护者评审、完整CI或合入通过。资源继续遵守每轮释放后全局至少20分钟间隔，仅在125/129空闲设备上运行，不停止他人实验。

## 本轮环境与复现配置

8×Ascend 910B3 64GiB；Python3.12.14，Torch/TorchNPU2.12.0，transformers5.13.1。两层主干＋一层MTP、8路由专家/topk2，序列4096、全局batch8、微批1、seed42、常数LR0.0001，专家权重梯度FP32及Muon NS float32，chunk loss4096，关闭重算。本轮环境与旧`ec2e16b`定向修复前后对照分开保存，不改写旧证据。

## 具体定位、触发条件与两侧正确性

固定本PR提交 `bd18261a14fc0608d0728defef756fd0028f80aa`。相关位置：[hyper_parallel/trainer/callbacks/checkpoint_callback.py::_collect_extra_state:197](https://github.com/hf1s/hyper-parallel/blob/bd18261a14fc0608d0728defef756fd0028f80aa/hyper_parallel/trainer/callbacks/checkpoint_callback.py#L197)；[hyper_parallel/trainer/callbacks/checkpoint_callback.py::_restore_meter_state:477](https://github.com/hf1s/hyper-parallel/blob/bd18261a14fc0608d0728defef756fd0028f80aa/hyper_parallel/trainer/callbacks/checkpoint_callback.py#L477)；[hyper_parallel/trainer/callbacks/checkpoint_callback.py::_apply_extra_state:408](https://github.com/hf1s/hyper-parallel/blob/bd18261a14fc0608d0728defef756fd0028f80aa/hyper_parallel/trainer/callbacks/checkpoint_callback.py#L408)。

触发条件：checkpoint 恢复后 global_step=200，但 meter 累计计数从零起算；step201 出现 8/6304 而不是 1608/3720095。

正确性参考：独立连续 400 步计数轨迹为 oracle；旧 checkpoint 字段缺失和 weights-only 的兼容行为必须单独验证，不能由 step 推测 token 数。

修复侧或并行侧核验：CPU 8 passed、6 subtests 覆盖实际 callback save/load 边界；历史红测累计计数 3200 处不符，绿测 0。

覆盖边界：18602c4 baseline 1–200 与 resume 201–400 分别对独立连续参考，8 rank×200 步全部记录量零差；最新 544526f7 不能继承此恢复结论。

### 参考适用范围

本项的正确性由上述所有权、设备/API、布局、累计状态或统计契约及其独立回归证明；没有强行套用MF/Megatron的不同实现，也未用该修复解释尚未定位的TP/CP/EP长步漂移。数值、重算和Muon的最新官方源码对照统一关联总问题单[#202](https://github.com/hf1s/hyper-parallel/issues/202)。

## 当前整体验收边界

本问题的定向回归与上文按版本标注的历史集成证据分别保留。最新训练源544526f7目前只有201的单次FSDP8完整200步及checkpoint CPU读回，共享机同源参考、跨机精度、最新续训和重算/并行矩阵尚未验收。当前先完成全200步、8rank主LM loss最大绝对差≤0.005；主loss全矩阵通过后先汇报，由用户决定是否继续MTP/aux专项。最新进度、曲线和资源约束见总问题单[#202](https://github.com/hf1s/hyper-parallel/issues/202)。本次仅补充说明，代码提交未改变，现有CI空结果不构成通过。
