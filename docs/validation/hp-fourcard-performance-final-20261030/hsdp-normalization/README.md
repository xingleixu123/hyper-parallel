# [草稿] 四卡 NPU HSDP 归一化独立验证

2026-10-30，129 的物理卡4–7，未插桩冻结1a源、CANN9.2.0 / PTA2.12.0.post3。这里关闭的是“Python trace 的 AVG 标签是否真的得到正确归一化”的证据缺口，不是性能优化或真实模型训练验收。实际运行完整结束、child exit0、四rank complete/pass；17份闭合证据逐字节核验。原始记录与设备身份细节保留在任务目录，公开JSON只投影数学与版本字段。

## 假设与区分性结果

mesh为replicate2×shard2；shard组[0,1]/[2,3]，replica组[0,2]/[1,3]。各rank的输入向量是 `(rank+1)*[1..8]`。若RS AVG正确，两个shard组系数分别为1.5、3.5；随后replica SUM再除2应得到系数2.5。四rank的实际RS输出及最终输出均逐元素与这些独立期望完全相等。数据刻意不同，不能由复制相同输入掩盖缩放错误。

模块复用固定源码现有测试的两层hidden64 MLP，FP32、local batch4。CPU oracle从四个不同的实际输入分别累积 `local sum-loss / 4` 的梯度；三次NPU初始化的全部8参数字节hash均与CPU相同，各rank实际输入hash也相同。比较第一层fc1/fc2的weight/bias，检查实际[Replicate,Shard(0)]、组、形状与offset。共4rank×3分支×4参数=48项：

| 路径 | 比较目标 | 最大绝对误差 |
|---|---|---:|
| no_fusion_avg | CPU四rank平均梯度 | 1.7881393432617188e-7 |
| fusion_avg | CPU四rank平均梯度 | 1.7881393432617188e-7 |
| no_fusion_sum | CPU平均梯度的4倍 | 7.152557373046875e-7 |

均满足既有测试容差rtol=1e-4、atol=1e-5；SUM对照同时排除把错误倍数当均值。没有clip或optimizer。这里的loss是toy的输出求和，不是训练token-mask loss；不能将本结果扩大为4096序列混合精度整网梯度等价或所有版本/形状的证明。完整原始梯度数组未导出，离线审阅可重算直接通信算术和记录统计，不能第二次从原tensor重算神经网络误差。

## 固定版本与复核

完整版本、每参数误差、直接通信原值、权重/输入hash见[results.json](results.json)。`preflight_versions.torch_npu_imported=false`仅表示CPU版本探针未导入PTA；实际worker明确导入torch_npu并通过HCCL在四卡运行。八个关键源码文件pin见[source-pins.json](source-pins.json)。实际manifest是b18fef…，参考profile manifest740502…含插桩，二者已区分；本次实际源没有profile插桩。

[worker.py](worker.py)保留实际执行字节及SHA732ff95b…，直接加载冻结源中的测试；[pinned_test.py](pinned_test.py)是相同SHA的阅读副本，保留原许可证。`python -B verify_results.py`仅需Python标准库，可离线复核公开数据投影。它不能替代设备执行。

实际启动使用隔离容器、只映射物理卡4–7，readonly挂载固定Python/venv、HP源、CANN和driver，另设全新的run/private目录。核心argv为：

```bash
source /work/runtime/cann920/cann/set_env.sh
/work/envs/hp-pta212/bin/python -m torch.distributed.run \
  --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=56940 \
  /work/scripts/worker.py
```

干净环境需要`HYPER_PARALLEL_PLATFORM=torch`、`HP_SOURCE=/work/sources/hyper-parallel`、`OUTPUT_DIR=/work/run`、`HP_DIAG_CID=本次新容器ID`、`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`、`HCCL_NPU_SOCKET_PORT_RANGE=57360-57375`，HOME/TMPDIR为隔离的`/work/private`，以及固定venv/Python的PATH/LD_LIBRARY_PATH。复现前重新核验空闲卡、端口和只读源SHA；不要复用已完成输出目录或未经审核的卡映射。

worker先只分配小tensor，原子写四份ready文件，在初始化HCCL前等待最多180秒。负责人必须用容器PID namespace/cgroup、进程命令和npu-smi实测，核验逻辑rank0–3实际对应物理4–7；随后在新run目录原子发布`device-release.json`，包含`kind=root-verified-physical-device-map`、本worker SHA、`run_id=hp-hsdp-normalization-129-v1`、本次CID和四项rank/physical_card映射。不能仅凭visible_devices或手填映射释放。完整创建、版本探针、实测映射、释放、fetch及清理命令/收据在本机同名任务的`npu-hsdp-normalization-v1`目录；运行采用这些受审脚本，未读取checkpoint或训练数据。

这次结果支持当前FP32 reducer语义，不能解释HSDP比FSDP慢、SAC attention梯度本底波动或第1804步异常，也不改变既有千步结果的范围。诊断结束后仅移除已确认无worker的本任务容器及空生成缓存目录，保留17份证据、source、共享CANN和checkpoint；本阶段累计清理27个结束容器。
