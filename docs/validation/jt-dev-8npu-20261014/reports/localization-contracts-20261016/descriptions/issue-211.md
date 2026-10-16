## Ascend 定向回归结果（2026-10-15）

本PR原独立提交`453618aaad72c3a2f9fb7bf367cd31ca45437d17`在129、CANN9.2.0-beta.2同镜像上完成复跑：两卡rank0/1各`2 passed`（FP32/BF16），四卡rank0–3各`1 passed`（Partial归约）。按2、1、3、4行交替验证合法等长、空分片、一般非整除、重复调用及缓存路径；非法源/目标分片和直接Partial入口一致拒绝，合法重建或归约后重建对独立参考零误差。

运行`r129-pr214-uneven-st4-r2`的两阶段和总退出码均为0，80.32秒后释放所用0–3号卡，无本任务设备PID残留。源码状态干净、patch为空、editable导入指向本checkout；24份原始文件大小/SHA全部吻合，规范/验收两轴独立复核通过，两端重复归档已清理。**这是Ascend/HCCL定向ST通过，不是八卡200步loss验收，也没有性能或显存收益声明。** PR代码和单提交SHA没有改变，问题保持开启、PR保持草稿，当前完整CI仍需单独核对。

[复跑分析](https://github.com/xingleixu123/hyper-parallel/blob/8a2ceac79e4436c54a9f7905f8507866e15bd6b8/docs/validation/jt-dev-8npu-20261014/reports/pr214-uneven-npu-st-r2/pr214-uneven-npu-st-r2-analysis.json) · [24份采集清单](https://github.com/xingleixu123/hyper-parallel/blob/8a2ceac79e4436c54a9f7905f8507866e15bd6b8/docs/validation/jt-dev-8npu-20261014/reports/pr214-uneven-npu-st-r2/pr214-uneven-npu-st-r2-evidence-manifest.json) · [原始日志及执行脚本](https://github.com/xingleixu123/hyper-parallel/blob/8a2ceac79e4436c54a9f7905f8507866e15bd6b8/docs/validation/jt-dev-8npu-20261014/reports/pr214-uneven-npu-st-r2/raw/runs/r129-pr214-uneven-st4-r2) · [释放凭据](https://github.com/xingleixu123/hyper-parallel/blob/8a2ceac79e4436c54a9f7905f8507866e15bd6b8/docs/validation/jt-dev-8npu-20261014/reports/pr214-uneven-npu-st-r2/raw/scheduling/r129-pr214-uneven-st4-r2.receipt.json) · [两轴检视](https://github.com/xingleixu123/hyper-parallel/blob/8a2ceac79e4436c54a9f7905f8507866e15bd6b8/docs/validation/jt-dev-8npu-20261014/reports/pr214-uneven-npu-st-r2/pr214-uneven-npu-st-r2-two-axis-review.json)。

实际测试入口如下；执行前由守卫检查空闲卡、保留上层`ASCEND_RT_VISIBLE_DEVICES`，两阶段使用独立日志目录和端口，每阶段400秒上限、容器总900秒上限：

```bash
cd /workspace/hyper-parallel
export HYPER_PARALLEL_TEST_DEVICE_TYPE=npu
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m torch.distributed.run --nproc_per_node=2 --master_port=30359 -m pytest -p no:cacheprovider -s -q tests/torch/shard/redistribute.py::test_uneven_shard_reconstruction_rejected
python -m torch.distributed.run --nproc_per_node=4 --master_port=30360 -m pytest -p no:cacheprovider -s -q tests/torch/shard/redistribute.py::test_uneven_partial_reduction_rejected
```

首轮从日志目录启动导致`ModuleNotFoundError: No module named tests`、0用例，退出1并释放；已修正操作脚本的工作目录并关闭pytest缓存，PR源码未改。首次CPU收集预检还出现Windows stdin换行问题，按bytes传输后3节点收集成功；直到上述实际Ascend复跑完成才记为NPU通过。[首轮及入口修复原记录](https://github.com/xingleixu123/hyper-parallel/blob/d13b610940b77284beb02f007c4ead9c6e56a024/docs/validation/jt-dev-8npu-20261014/reports/pr214-uneven-npu-st)。

总问题单：[#202](https://github.com/hf1s/hyper-parallel/issues/202)。独立单提交修复：[#214](https://github.com/hf1s/hyper-parallel/pull/214)，提交 `453618aaad72c3a2f9fb7bf367cd31ca45437d17`。

## 问题描述与影响

普通 `Shard(0)` 对全局 `[1,64]` 做两路切分时，本地分别为 `[1,64]` 和 `[0,64]`。旧 `full_tensor()` 仍使用等长分片重建路径，在实际 NPU 观察中 rank0 返回 `[2,64]`、rank1 返回 `[0,64]`，均违反全局形状契约。独立 CPU/Gloo 进一步复现一般 `[3,64]` 的通信长度不匹配，可能导致某 rank 报错、其余 rank 等待或连接关闭。

这是 DTensor 重分布的布局检查缺口，独立于 Muon 更新。最初 #201 的优化器 ST 改用本地分片与独立参考零容差比较后更新通过，并不表示重建已正确。

## 独立复现与根因

在 JT_dev `521d3a8` 上构造双进程 mesh，两 rank 使用相同完整张量并调用 `distribute_tensor(..., [Shard(0)])`，再重复 `full_tensor()`。先测 `[2,64]`：两次重建均零误差；随后分别运行 `[1,64]`、`[3,64]`：原版出现形状异常或 Gloo 通信失败。复现不初始化优化器，不涉及 loss、NS 精度或训练数据。

现有 `layout.has_uneven_shard` 检查只覆盖显式不均匀描述，未拦截普通 Shard 的全局维度不能整除 mesh 分片数的情况。直接 `reduce_partial()` 还可先归约并重建布局，改变后续观察到的形状；仅在 `redistribution()` 检查不完整。检视中新增了四进程 `(Shard(0), Partial())` 回归，撤掉直接入口检查后所有 rank 都出现 `DID NOT RAISE`。

历史真实 NPU 原始证据：[原始rank0日志](https://github.com/xingleixu123/hyper-parallel/blob/66464cdd569d864562578eb45e9aa234d81770f4/docs/validation/jt-dev-8npu-20261014/reports/dtensor-full-tensor-uneven-rank0.log)；[rank1日志](https://github.com/xingleixu123/hyper-parallel/blob/66464cdd569d864562578eb45e9aa234d81770f4/docs/validation/jt-dev-8npu-20261014/reports/dtensor-full-tensor-uneven-rank1.log)；[96行原始worker](https://github.com/xingleixu123/hyper-parallel/blob/66464cdd569d864562578eb45e9aa234d81770f4/docs/validation/jt-dev-8npu-20261014/reports/uneven-full-tensor-observed-worker.txt)。原失败运行 `production-muon-precision-st-v1` 的观察问题与最终优化器用例的本地零容差比较均保留，未修改原日志。

## 修复方案及兼容边界

#214 选择**在通信前一致拒绝当前尚不支持的普通非整除分片**：

1. 重分布入口同时检查源和目标的全局形状，按每维所有相关 mesh 轴的累计分片数判断整除性；检查早于 Partial 归约、RaggedShard 分流和变换缓存。
2. 直接 `DTensor.reduce_partial()` 同样检查，避免先通信、后才暴露错误。
3. 所有 rank 依据相同全局元数据抛出同文 `NotImplementedError`，包含形状、维度、长度和分片数。检查只读取主机元数据，不新增 collective、设备同步或张量分配。
4. 保留合法等长分片、全局零长度、现有 RaggedShard 路径与 FSDP 私有填充通信边界。未实现普通非整除 Shard 的收集算法；直接访问本地分片仍可用于相关优化器验证。

## 回归命令与实际结果

```bash
export HYPER_PARALLEL_TEST_DEVICE_TYPE=cpu
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export OMP_NUM_THREADS=1
python -m pytest -q -rs tests/ut/core/dtensor
python -m pytest -q tests/torch/shard/test_redistribute.py::test_uneven_shard_reconstruction tests/torch/shard/test_redistribute.py::test_uneven_partial_reduction
```

- DTensor UT：457项测试、41项子测试通过；96项为基点已有的整模块跳过，未新增跳过。
- 分布式 ST：两个启动用例通过，实际分别启动2/4个 Gloo 进程。FP32/BF16 均按2、1、3、4行交替执行，覆盖空分片、一般非整除、重复调用和缓存变换；非法布局所有 rank 同文异常，合法重建相对独立参考要求零误差。
- 四进程合法 Partial 归约后重建要求等于 `2 × reference`，零容差；非整除时重复直接归约也必须在通信前拒绝。撤掉入口保护的对照证明确实能抓到缺陷。
- 首版 launcher 因工作目录相关路径解析失败，修正为相对于测试文件的绝对 worker 路径后重跑通过；保留首次失败。薄 launcher 未导入 Torch/HyperParallel。
- 规范及验收条件两轴复审完成；生产代码 pylint 基点32项、补丁30项，无新增；本地缺 Torch，静态检查不包含 import-error/no-member，不能称全量 lint 无告警。

上述CPU回归使用隔离容器、Python3.12.14、Torch2.12.0+cpu、pytest9.1.1，editable导入已核验，无NPU挂载。实际Ascend/HCCL结果见开头新增证据：TorchNPU2.12.0、CANN9.2.0-beta.2下两卡及四卡均通过；没有从CPU结果推导NPU通过，也没有吞吐或显存收益声明。

[原始 UT 日志](https://github.com/xingleixu123/hyper-parallel/blob/2f5ba55838338dcb5c37c9cd44a958586da2e89a/docs/validation/jt-dev-8npu-20261014/reports/issue211/issue211-cpu-v3-ut.log)；[原始 ST 日志](https://github.com/xingleixu123/hyper-parallel/blob/2f5ba55838338dcb5c37c9cd44a958586da2e89a/docs/validation/jt-dev-8npu-20261014/reports/issue211/issue211-cpu-v3-st.log)；[源码指纹与检视记录](https://github.com/xingleixu123/hyper-parallel/blob/2f5ba55838338dcb5c37c9cd44a958586da2e89a/docs/validation/jt-dev-8npu-20261014/reports/issue211/issue211-review.json)；[全 rank 日志索引](https://github.com/xingleixu123/hyper-parallel/blob/2f5ba55838338dcb5c37c9cd44a958586da2e89a/docs/validation/jt-dev-8npu-20261014/reports/issue211/worker-log-index.json)。

## 验收状态

- [x] 保留独立于优化器的2进程旧版失败证据。
- [x] 普通非整除源/目标布局在重分布和直接 Partial 入口一致拒绝，覆盖多 mesh 轴累计分片。
- [x] 空分片、一般非整除、重复调用、缓存与合法路径的 CPU 分布式回归通过。
- [x] 原独立提交Ascend/HCCL两卡FP32/BF16及四卡Partial补丁回归完成。
- [ ] 维护者兼容性评审、当前提交完整 CI 与合入确认。

问题保持开启，PR保持草稿。NPU 回归遵守总单的空闲设备与全局20分钟间隔约束，不停止他人实验。该单只处理 DTensor 边界，不将其他 PR 或完整200步训练结果混作本修复的证据。

## 具体定位、触发条件与两侧正确性

固定本PR提交 `453618aaad72c3a2f9fb7bf367cd31ca45437d17`。相关位置：[hyper_parallel/core/dtensor/tensor_redistribution.py::_check_equal_shard_sizes:294](https://github.com/hf1s/hyper-parallel/blob/453618aaad72c3a2f9fb7bf367cd31ca45437d17/hyper_parallel/core/dtensor/tensor_redistribution.py#L294)；[hyper_parallel/core/dtensor/tensor_redistribution.py::reduce_partial:497](https://github.com/hf1s/hyper-parallel/blob/453618aaad72c3a2f9fb7bf367cd31ca45437d17/hyper_parallel/core/dtensor/tensor_redistribution.py#L497)；[hyper_parallel/core/dtensor/tensor_redistribution.py::redistribution:270](https://github.com/hf1s/hyper-parallel/blob/453618aaad72c3a2f9fb7bf367cd31ca45437d17/hyper_parallel/core/dtensor/tensor_redistribution.py#L270)。

触发条件：逻辑 [1,64]/[3,64] 普通 Shard(0) 分给两个 rank，进入等长重分布；空分片/一般非整除可返回错形状或通信失败，直接 Partial 归约也需入口拒绝。

正确性参考：合法等长重建对原逻辑 tensor 零差；Partial 归约后对独立 2×reference 零差；不支持输入各 rank 根据同一全局元数据一致拒绝。

修复侧或并行侧核验：CPU DTensor 457 passed、41 subtests；2/4 进程 Gloo 通过；真实 Ascend/HCCL 2/4 卡定向 ST 通过，24 原始文件核验。

覆盖边界：选择明确拒绝未支持普通非整除分片，不是实现 ragged gather；不改变 RaggedShard/FSDP 私有 padding。无 200 步、性能或显存收益声明。

### 参考适用范围

本项的正确性由上述所有权、设备/API、布局、累计状态或统计契约及其独立回归证明；没有强行套用MF/Megatron的不同实现，也未用该修复解释尚未定位的TP/CP/EP长步漂移。数值、重算和Muon的最新官方源码对照统一关联总问题单[#202](https://github.com/hf1s/hyper-parallel/issues/202)。

## 当前整体验收边界

本问题的定向回归与上文按版本标注的历史集成证据分别保留。最新训练源544526f7目前只有201的单次FSDP8完整200步及checkpoint CPU读回，共享机同源参考、跨机精度、最新续训和重算/并行矩阵尚未验收。当前先完成全200步、8rank主LM loss最大绝对差≤0.005；主loss全矩阵通过后先汇报，由用户决定是否继续MTP/aux专项。最新进度、曲线和资源约束见总问题单[#202](https://github.com/hf1s/hyper-parallel/issues/202)。本次仅补充说明，代码提交未改变，现有CI空结果不构成通过。
