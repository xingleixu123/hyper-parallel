# CP2 × DP2 token 梯度 CPU 核验

四进程 Gloo，正式 fdce composite source manifest `7dd9613c60843e646e15f6bb9879f477c201d63824793859e5386bd8059d8f4f`；从 source-relay 解出的 1513 文件逐 SHA 比对全部一致。仅为让 CPU 环境导入 CP 模块，用 shim 代替未执行的 NPU attention replacement 模块；真实 CP mean、shift、autograd all-reduce、StepLossNormalization、aux context/bind 均直接来自该源码。没有模拟通信。

两 microstep、每个 DP 一个不同序列，LM 有效数分别为 3、5、4、6；首序列第二 CP shard 无有效 token。MTP 使用真实跨 shard shift 和其自身 global valid count；aux 独立参数非零，经真实 aux 注入。实际 rank loss 报告只包含 LM+0.1 MTP，aux 不改变 forward 数值；独立参考对 aux 额外加入目标以核对梯度。外部以四进程 all_reduce/4 模拟 FSDP 平均，不是实际 NPU FSDP 实现。

真实 CP mean 的可微 SUM 在反向按 CP 聚合上游权重；各 shard 上游权重是各自的 foundation local token count，因此合起来是该序列的全局 LM 有效数。模拟 FSDP 平均后，finish 乘 DP×CP / 全局 token 数，抵消该平均；MTP 使用各自的 CP 预测计数作为内部 mean 分母，同时保持外层 LM token 权重。aux 的独立分支也按公共 context 的外层权重计算。该例不验证其他 aux 定义或真实 NPU FSDP。

总 LM token 18；四 rank 的 visible loss 为 4.584926181369358，对照 4.584925925925925；三个参数梯度分别为 [13.622222323715684, 0.970740747973323, 0.260000001937151]，对照 [13.62222222222222, 0.9707407407407408, 0.26]。rtol/atol 均 2e-6；源码归一化内有 float32 转换。日志包含 PyTorch 分布式 API 的 FutureWarning。

脚本 SHA256 `d3b0bfb89edaac08171eaa5504191ef390be4758e4c282325dff472a48d6c4b7`，stdout/stderr SHA256 `0eb438fd9a904c010a6186c5a3ac442a0daddf5c020d1fa49bb9ab556b05b813`。复现：在已核 7dd 源根，将其加入 PYTHONPATH，使用 CPU torch 2.12 环境运行 `OMP_NUM_THREADS=1 python -m torch.distributed.run --standalone --nproc-per-node=4 <cp-dp4-token-probe.py>`。
