# Xing4 router 统计域修复：NPU 独立诊断

结论：12/12 配对记录通过真实 planner/boundary、实际 router 与独立 FP64 公式核验。TP2/EP2/SP=true 的原实现按半序列求 AUX；候选按最终输入布局合并同一序列分片后恢复完整序列目标。两个完整序列场景的 raw AUX 与梯度逐值不变。

环境为 Ascend910B3 ×4、DP2×TP2、CP1，PyTorch 2.12.0+cpu / torch_npu 2.12.0.post3，FP32 matmul HF32 关闭。基线源码为 PR1477 `eb1121470917789ce641e3f12b4b6bad65b6ba54`。两个实现使用相同 W/input 字节；两个 DP 样本独立。

fixture 为固定 seed 的合成因果输入，不是 MF 训练样本：E64/H3584/K4/S4096、n_group=topk_group=0、bias rate=.0001、AUX coeff=.0005、真实 facade prepend/remove 64 safe tokens。独立参考显式计算 FP64 sigmoid、unbiased top-k、每序列归一化 affinity/counts，并乘 .0005×.5 求 W 梯度。

| Rank | 场景 | Router tokens | 原始 raw AUX | 候选 raw AUX | FP64 全序列 AUX | 原始梯度 relL2 | 候选梯度 relL2 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | EP1/SP | 4096 | 1.0105425119 | 1.0105425119 | 1.0105424562 | 6.280227e-07 | 6.280227e-07 |
| 0 | EP2/SP | 2048 | 1.9578151703 | 1.0105423927 | 1.0105424562 | 1.230985 | 5.034898e-07 |
| 0 | EP2/noSP | 4096 | 1.0105425119 | 1.0105425119 | 1.0105424562 | 6.280227e-07 | 6.280227e-07 |
| 1 | EP1/SP | 4096 | 1.0105425119 | 1.0105425119 | 1.0105424562 | 6.280227e-07 | 6.280227e-07 |
| 1 | EP2/SP | 2048 | 1.9578151703 | 1.0105423927 | 1.0105424562 | 1.230985 | 5.034898e-07 |
| 1 | EP2/noSP | 4096 | 1.0105425119 | 1.0105425119 | 1.0105424562 | 6.280227e-07 | 6.280227e-07 |
| 2 | EP1/SP | 4096 | 1.2473604679 | 1.2473604679 | 1.2473605671 | 8.126542e-07 | 8.126542e-07 |
| 2 | EP2/SP | 2048 | 1.0105421543 | 1.2473607063 | 1.2473605671 | 0.33522 | 7.104523e-07 |
| 2 | EP2/noSP | 4096 | 1.2473604679 | 1.2473604679 | 1.2473605671 | 8.126542e-07 | 8.126542e-07 |
| 3 | EP1/SP | 4096 | 1.2473604679 | 1.2473604679 | 1.2473605671 | 8.126542e-07 | 8.126542e-07 |
| 3 | EP2/SP | 2048 | 1.9578155279 | 1.2473607063 | 1.2473605671 | 0.33522 | 7.104523e-07 |
| 3 | EP2/noSP | 4096 | 1.2473604679 | 1.2473604679 | 1.2473605671 | 8.126542e-07 | 8.126542e-07 |

候选最大 raw AUX 绝对误差为 1.392e-07，最大梯度 relative L2 为 8.127e-07，均通过预定阈值 2e-5 / 5e-4。EP1/SP 与 EP2/noSP 共 8 个配对记录的 AUX 和 TP-mean 梯度逐值 exact；全部 12 个记录的本地 bias counts、去重后完整序列 counts 均与独立参考精确相同，候选不改 counts。

| 场景 | 实际 planner 输入 TP 布局 | Source → router → output 的 local token 数 | 样本覆盖与候选统计域 |
|---|---|---|---|
| EP1/SP | Shard(1) → Replicate，真实 all_gather | 2048 → 4096 → 2048 | 每个 router 副本均覆盖 0..4095；不合并 TP |
| EP2/SP | Shard(1) → Shard(1)，identity | 2048 → 2048 → 2048 | 同一 DP 内两 rank 无交集且并集 0..4095；仅 TP [0,1] 或 [2,3] |
| EP2/noSP | Replicate → Replicate，identity | 4096 → 4096 → 4096 | 每个 router 副本均覆盖 0..4095；不合并 TP |

位置 ID 经过同一真实 boundary，gate 入口激活逐字节核对到这些位置；不根据 SP 布尔手工切片。输出 shape 与原始 source view 一致，safe64 的零权重与专家覆盖均验证通过。两个 DP 样本始终没有被混入同一序列统计。

这是 router/boundary 的修复证据：调用真实模型 facade、router、resolver 与 PrecompiledBoundary；GMM、专家 A2A 和 shared MLP 明确截断。梯度只做显式诊断 TP 平均，未运行 FSDP、optimizer、完整模型或 1000 步续训；CP-only/CP×TP 也没有本轮 NPU 验收声明。

控制器边界单独保留：baseline probe 进程 rc=0；candidate 四个 rank 均完成三场景并保存结果后，退出阶段的资源监控把已归属 PID 的 R/D 状态与根 cgroup 观测判为 foreign，停止了 elastic wrapper，candidate rc=-15、controller 状态为 failed。原报告没有保存 stat flags，因此不能事后确证 PF_EXITING；不把该 wrapper 结束状态记成正常 controller 退出。四张设备最终均空闲，全部 holder 已移除，设备 FD 持有者为空。

完整原值、token ID、源码 manifest、worker/独立 FP64 参考与控制器原报告已归档。24 个原梯度文件的 72 个张量按内容无损去重为 16 个张量（14,681,264 bytes），逐文件验证可精确重建原始 safetensors 字节。

证据身份：

- stage SHA256 `69dffbaea48243dd77894e49da390d341a77e015618068e5b436b6cd96c9ce6b`
- controller SHA256 `9754ea511b25165a90797dc6c934094cd3d94f5bc63aaa98913868c26b27f4ca`
- baseline source manifest SHA256 `f83fa2b7b4e7fc6edcef2e71cbf80021b51baead940123c36610f1fcf5afd1f6`
- candidate source manifest SHA256 `4eb1a71aa1ca1f187542036419171980dff14796b0b6438087360b9de76a9cd0`
- worker SHA256 `46539704da9d486c197d741d1d64a566eacf230036750a226612a605b4875f47`
- gradient bundle SHA256 `70a6679d517a3e590f32b3a90a2d4e8bd0e489bc4941b316074b7ab25ce99898`
