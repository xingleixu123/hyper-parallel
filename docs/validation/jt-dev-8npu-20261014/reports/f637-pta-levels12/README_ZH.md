# 固定 PTA level 1/2 单卡测量：所测算子未观察到跨行布局差异

两种级别请求下，合成 BF16 RMSNorm 和 q_b_proj 形状 Linear 的重复、二分行划分、行置换及置换后二分，前向和输入梯度全部比较统计为零。两级别的参考输出/dX 原始字节 SHA256 相同。**本结果不证明训练漂移已修复，不支持据此修改训练确定性级别。**

| 直接使用的实现与形状 | level 1 四种比较：前向/dX 最大差 | level 2 四种比较：前向/dX 最大差 |
|---|---:|---:|
| JT 实际导入的 HF `DeepseekV32RMSNorm`，`[1,4096,5120]` | 0 / 0 | 0 / 0 |
| 原生 `torch.nn.Linear`，`[1,4096,1536] → [1,4096,24576]`，只计算 dX | 0 / 0 | 0 / 0 |

每个级别、每种算子执行 7 次原生调用：一次参考，加上四种比较所需的 1/2/1/2 次调用。按全局行号对齐；二分覆盖 2×2048 行，所有行恰好出现一次。16 组前向/dX 比较汇总共记录 1,191,182,336 个坐标。文件中保存全量统计与参考字节 hash，没有保存完整输出 tensor；本次自审核对这些已保存统计，不能冒称第二套独立算子实现重算。

## 固定来源与有效范围

源码 `18602c4524dc664b81b2bad9a7b624455ce1494f`，JT_dev `f637ef7ee4d4b416800716f1ebaaae7709b8a06a`（含 PR180/186）；固定镜像 `sha256:5f9e677f439d11ab4e7c888f54b142335a27eee7411cbd2ce894a4798a70146d`，CANN 9.2.0-beta.2/V100R001C12B086、Torch 2.12.0+cpu、TorchNPU 2.12.0 / commit `fa0f83fe49d309dcbc31e264e9e6ed6e5dc49d2d`、Transformers 5.13.1、Ascend910B3。

两级别在独立 Python 子进程中使用同一 CPU 固定种子输入，检查输入、权重及上游梯度 SHA；公开 level setter 每进程调用一次，不再调用普通 deterministic setter，也不调用具有副作用的私有 getter。HF32 关闭，`CLOSE_MATMUL_K_SHIFT=1`，请求 `jit_compile=False`。setter 返回成功和普通确定性开关开启都不等于有公开只读接口确认实际 CANN runtime level。真实 Torch dispatch 元数据也不能读取 CANN 内部算法。

这是两个合成算子的 forward/dX 探针，未覆盖真实训练输入、LM head 大矩阵、权重梯度、通信或优化器。不能由这两个形状零差排除其他形状或运算引入误差。跨级别只比较原始字节 hash 及保存的样本，不提供未保存全量数组的逐值差。

## 执行、让卡和资源

- v2：`r129-f637-pta-levels12-1npu-v2`，只用 129 的 0 号卡，共 50.154 秒。level 1 进程完整结束，level 2 在运行中出现另一容器的 NPU 进程；守卫核实 cgroup/PID namespace 不同，只停止自己的容器。外部 PID 在释放后仍存在。v2 的 level 2 不完整，原样保留。
- v3：`r129-f637-pta-level2-1npu-v3`，从 v2 确认释放后等待完整 1200 秒，再检查空卡。仅重跑 level 2，单卡 30.814 秒，exit 0、所有自有 PID 已释放。v3 没有使用 v2 的中断前缀。
- 两级别 allocator 累计峰值均为 629,160,960 bytes，reserved 峰值均为 694,157,312 bytes。守卫物理 HBM 采样最大值 v2/v3 为 3569/3523（npu-smi 标示 MB）；这是整卡采样，冲突期间可能包含其他进程，不能作为本进程或瞬时全程峰值，也不是训练显存成本。
- 每次单卡预约硬限 600 秒；子进程限 110 秒，显存 allocator 配额 8 GiB，物理 HBM 采样阈值 16 GiB，CPU 容器上限 16 GiB/4 CPU。结束后继续执行至少 20 分钟间隔。两端重复传输归档已核验后清理。

## 原始证据及恢复

[完成比较与来源](fixed-pta-levels12-completed-comparison.json)、[v2 中断自审](fixed-pta-levels12-v2-partial-audit.json)、[v3 部署与边界](fixed-pta-level2-retry-v3-deployment.json)。这里的审阅者是主代理自审，未标为独立审计。

[无损包](probe-inputs.tar.gz) 包含两轮 22 份原始文件、采集清单、请求及各自冻结的源码依赖；[完整大小/SHA 清单](probe-inputs-manifest.json) 记录每个成员。不同守卫版本分开保存，未用后续修改替换执行时输入。

```bash
python restore_inputs.py /path/to/new-directory
```

目标必须不存在。恢复器逐文件核对大小、SHA 和路径，不覆盖旧文件。完整 tensor 未落盘，不在包内。

## 下一步：主 loss 的 200 步配对候选，尚未运行

三步 C/D 中 FP32 Linear dgrad 的主 LM 差减小，只能支持进一步实验。准备了同一新数值配方的 FSDP8、CP2 两个独立 200 步实验，关闭新 checkpoint 输出，复用已验证的导数实现，仅增加每步有界实际分支计数。CPU 9 个导数用例、实际 parser 2 份配置及守卫 25 项边界测试通过，NPU 200 步仍未执行。

CPU 首次配置检查拒绝了一次诊断配置序列化错误：JSON 的无小数点指数被 YAML 解析为字符串。修复保留原始科学计数法格式后，第二次 CPU 容器通过；两次容器均无 NPU 映射并已移除。此为实验工具问题，不新建产品 issue/PR。新结果文件名中的 v1 指 helper 版本，成功执行凭据为 v2，不能混淆。

[设计与判定](prepared-200-step-dgrad/f637-dgrad-200-plan.md)、[实际 CPU 原始结果](prepared-200-step-dgrad/f637-dgrad-long-cpu-v1.json)、[执行凭据核验](prepared-200-step-dgrad/f637-dgrad-long-cpu-verification.json)、[前置自审](prepared-200-step-dgrad/f637-dgrad-long-prelaunch-review.json)。两组需同配方比较，不同旧原生基线混比；主 loss 门槛保持 0.005，MTP/aux 记录但不改变其权重。普通 wgrad 的 autograd replay 带来诊断额外计算，不能宣传为生产性能收益。
