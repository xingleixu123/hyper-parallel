# CANN 9.2 单卡 1501–2500 数值对照

基于本轮 MF1500 转换状态，HP 原生恢复后完成 1501–2500 连续 1000 次更新；HP controller finished、returncode 0、容器移除，worker passed。恢复收据记录模型、master、优化器槽、参数组及 native extra state 精确一致。MF 原始训练调用返回，但原 runner 的终态 maps guard 错将已 unlink 共享内存识别为库导致 exit1；条件性收据不能证明终态 inode/权限连续性。原失败保留，不能称 MF 正常退出。MS 2.10 / CANN 9.2 是用户选择保留的实验栈，未找到该组合的官方配套声明。

比较逐优化步 HP `loss_native`（两 micro 返回值之和）减 MF float32 callback 原值，不做平滑。1000 步 signed mean −0.00251027512550354、abs(mean) 0.00251027512550354，满足既定 ≤ 0.005 数值门槛；103 正、897 负、0 零。MAE 0.0027149217128753663，最大绝对差 0.02550530433654785，发生在 step 1778（差为负）；MAE 和最大差为诊断，无附加阈值。MTP 按配置关闭，MTP LOSS 为 N/A。数值达标不证明 Muon 或跨框架算法语义等价，且受 MF 终态缺口限制。

实际输入比较共 8000 项（1000 步×2 micro×4 字段）全部相同；每 micro 有效 token 4096、每 step 8192；1000 步 HP stdout 中 step loss 与原始 ledger 一致，ledger 两个 native micro 返回值之和等于该值；HP 默认 RNG 每步前后摘要相同，并非声称框架间 RNG 状态等价。逐步主臂原值和差值保留在 paired-loss.csv，完整统计见 statistics.json。独立计算与原统计工具一致。

| 对 MF callback，同 1501–1800 的 300 步 | signed mean | abs(mean) | MAE | max abs（step） |
|---|---:|---:|---:|---:|
| HP 主臂 asym5 | −0.0006307220458984375 | 0.0006307220458984375 | 0.0011129124959309896 | 0.02550530433654785（1778） |
| HP legacy 消融 | +0.0003791729609171549 | 0.0003791729609171549 | 0.0005170313517252604 | 0.004960298538208008（1703） |

消融经恢复后仅改变 native Muon ns_variant，两个 run 均 finished/rc0，legacy 300 步 actual 2400 字段、token 数及默认 RNG 前后摘要核对通过；上述仅比较相同窗口，不以 300 步推断 1000 步或证明其他因素因果。消融主臂的 300 步与整体 1000 步不可互换。

原始证据已逐文件核对采集前后文件身份与 SHA；本公开包仅包含原始配对 loss 值、统计与版本摘要；不含原始样本、私有 YAML、RNG bytes 或 checkpoint。MF ledger SHA256 为 f7b40a1d3e9167de74757eeacf2b9209449939e3a34ed3c3c3f223277262ea45，HP 主臂 ledger SHA256 为 384cb8b6bc22152631156f4b299745aca2d7797f684e89a5371646b4b00b42e1。四卡验收尚待。

## HP1500 原生 checkpoint 的独立 CPU 读回

使用只读、无 NPU 的 world1 CPU 容器，通过 HP native DCP 全局逐 tensor 读取：58 个 model state 和 86 个优化器槽逐项与保存前活体 SHA 一致；596 个参数组非 Tensor 字段一致。此前活体 inventory 中 57 个参数的 main 和 model 为同一对象且共享存储，并经转换 binder 复核；不虚构额外的 DCP main 张量。checkpoint 四文件前后流式 SHA 与原闭合收据一致、文件身份不变，未产生完整 tensor dump。world1 不证明四卡分片读取或跨 mesh 恢复。

## 复核命令与复现边界

在本目录用 Python 标准库复算 CSV 的 1000 步统计：

```bash
python -c "import csv,math; r=list(csv.DictReader(open('paired-loss.csv',encoding='utf8'))); assert [int(x['step']) for x in r]==list(range(1501,2501)); d=[float(x['main_loss'])-float(x['mf_callback']) for x in r]; print('count',len(d),'mean',math.fsum(d)/len(d),'absmean',abs(math.fsum(d)/len(d)),'mae',math.fsum(map(abs,d))/len(d),'max',max(map(abs,d)))"
```

训练需持有同一数据、私有配置、受控设备及 MF1500 闭合源状态；公开包不含其原件。原试验最终容器内入口为 `source /work/output/cann92-set-env.sh && source /work/task/vendor-opp/vendors/omni_training_custom_transformer/bin/set_env.bash && exec /work/env/bin/python -B /work/output/worker.py`，依赖原 launcher 的只读挂载、设备隔离、审签和物理卡握手；这条内层命令本身不是可独立复现整个实验的一键命令。版本、数据和配置 SHA 见 versions.json。
