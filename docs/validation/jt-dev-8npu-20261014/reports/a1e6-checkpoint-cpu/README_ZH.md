**201 第 200 步 checkpoint：CPU 完整读回通过，尚未验收 NPU 续训**

2026-10-16，固定源码 `544526f7db8342c607244fd41009b2a74adb84c8`、原验证镜像及 Torch `2.12.0+cpu`。
检查对象为 `r201-a1e6-fsdp8-0-200-v1/global_step_200`。本次检查不使用 NPU，也不改变 201 的跨机精度准入状态。

25 个文件共 **27,973,262,840 字节**完成哈希复核。实际元数据包含 648 个逻辑项：112 个 F32 tensor、536 个 BYTE_IO 项；882 个物理 tensor 分片的形状、dtype、文件范围和全局覆盖检查通过，46 个优化器 step 记录全部为 200。

全部 **27,972,632,640 字节 tensor 数据**通过 safetensors 在 CPU 上分块读回，没有发现非有限值。共读取 1,682 块，实际最大单块为 30 MiB，配置上限 32 MiB。这是单块 tensor 大小，不能作为进程或主机内存峰值。检查容器限制为 4 CPU、8 GiB、600 秒，使用 runc、隔离网络及只读 checkpoint/源码挂载，没有映射加速设备。读回检查耗时 90.93 秒，容器退出码 0、未发生 OOM，容器已移除并再次核实。

8 个 rank 的额外状态均能通过 `torch.load(map_location="cpu", weights_only=True)` 解析，并完成以下核对：

- `global_step=200`、`epoch=0`；AdamW 和 Muon scheduler 均为 `last_epoch=200`、`step_count=201`，保存的学习率与第 200 步记录一致。
- 累计样本数 1,600、累计 token 数 3,713,791，与各 rank 最后一条训练记录一致。
- dataloader 保存的全局 batch 和 DP 大小均为 8，采样/迭代器计数为 200，已消费样本数为 1,600。
- CPU/Python RNG 可以在独立本地 generator 中往返恢复；设备 RNG 只检查保存字节的格式，没有在 NPU 上应用。

本次 tensor SHA 是训练结束后建立的指纹，不能当作与保存瞬间内存内容的比较。这里也没有把 checkpoint 恢复到分布式模型、执行优化器更新或验证未来 batch 重放。因此，**CPU 读回通过不代表 200–400 步续训精度通过，也不代表 201 与 125/129 精度相同**。

201 checkpoint 继续保留在 `/opt/xxl-jt-dev-8npu-20261014/checkpoints/r201-a1e6-fsdp8-0-200-v1/global_step_200`。125/129 继续等待设备自然释放，先完成匹配单卡探针及同源 200 步参考。只有全部 200 步、8 rank 主 loss 差不超过 0.005 且数据/初始状态等条件一致，后续 NPU 矩阵才迁到 201；随后使用该 checkpoint 做 200–400 步续训，并与独立连续 400 步参考比较。

证据与复核：

- [完整 CPU 读回结果](201-a1e6-checkpoint-read-v1/checkpoint-cpu-read.json)、[容器回执](201-a1e6-checkpoint-read-v1/receipt.json)、[采集清单](201-a1e6-checkpoint-cpu-collection-v1.json)。
- [25 文件哈希和 header](201-a1e6-step200-checkpoint-sha256-v1.json)、[root 独立证据复核](201-a1e6-checkpoint-cpu-root-review-v1.json)。后者核对 49 项输入、退出状态、覆盖和计数条件，没有重复执行远端 tensor 扫描。
- [最终运行器独立审阅](201-a1e6-checkpoint-cpu-runner-review-v3.json)、[reader 验证记录](201-a1e6-checkpoint-cpu-draft/independent-review-v2.json)。
- [最终冻结输入](201-a1e6-checkpoint-read-v1-inputs-v2/input-manifest.json)、[部署读回](201-a1e6-checkpoint-cpu-deployment-v2.json)。
- [清理回执](201-a1e6-checkpoint-cpu-cleanup-v1.json)：远端未执行的旧版暂存 5 文件、308,126 字节已删除；本机历史证据、最终结果和续训 checkpoint 保留。没有复制约 28 GB tensor 到本机。

归档说明：JSON 内的主机路径记录原执行位置；本目录按原文件字节保留证据。冻结输入目录保留其三个校验对象；[reader](201-a1e6-checkpoint-read-v1-inputs-v2/read_checkpoint_cpu.py) 和 [CPU 运行器](run_201_checkpoint_cpu_v2.py) 用于复核当时实际读取及隔离方式，运行器中的路径需要显式适配后才能在其他机器复现。输入数据和 checkpoint tensor 未上传。
