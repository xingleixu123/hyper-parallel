# 四卡 HSDP Muon replica pack 与实际设备 flow（草稿证据）

本目录归档一次独立、保持原算术的四卡诊断。step 1511–1514、每 rank 每步观察到四个 FP32 replica 广播 pack：专家 `gate_up_proj` 单成员 939,524,096 bytes，`down_proj` 单成员 469,762,048 bytes；另外两包包含 15 和 11 个矩阵，分别为 296,763,392 和 296,779,776 bytes。每 rank 每步成员实际元素共 2,002,829,312 bytes。pack 在 replica PG [0,2] 和 [1,3] 的对应调用、成员和 offset 可由 sidecar 与 `audit.py` 复核。这里的 bytes 是元素逻辑大小，不是网络传输量。

新增设备 trace 关联：rank 0、2 在 1512–1514 共 24 个 replica 广播，实际 pack→Python collective 标签→唯一设备 flow 端点→同名同时间 CSV 行全部匹配；24 个原有 Work.wait 的主机范围有唯一 flow 到设备 EVENT WAIT。18 个 unpack 范围的实际拷贝 flow 与 CSV 完整匹配，另外 6 个范围保留 unknown。完整关键路径仍未证明。replica owner 与 shard NS compute owner 是不同概念。专家参数为本地 expert batch NS，不能称全部 64 expert 做 NS all-gather。观测工具只记录 CPU metadata，不读取 tensor 值；插桩可能扰动执行，本实验不是无 profiler 性能排名。

环境：129 cards 4–7，Ascend 910B3 / HCCS；Python 3.12.14、PyTorch 2.12.0+cpu、PTA 2.12.0.post3、CANN toolkit 9.2.0（内部版 V100R001C12SPC001B237）。基础性能源码 HyperParallel `1a39572fef23819a2d2651b4bd53047a05a4892f`，本次为其冻结 source tree 加观察 patch，不能将基础 commit 当成本次完整部署内容。模型 HSDP mesh 2×2（replicate、shard），GBS 4、LBS 1、sequence length 4096，3 层（2 Dense + 1 MoE），MTP off。

从 native step 1500 恢复后，外部审计对 258 项参数、状态、optimizer 角色及组和 extra 状态进行核验；每 rank 的 native RNG bytes 恢复比对通过，不宣称跨 mesh 随机流等价。运行 1501–1510 warmup、1511 profiler warmup、1512–1514 profiler active。实际加载样本的检查包含 1501 和 1502 的四个准备字段及对应 sample index/切片；这里只保留范围说明，不公开样本值或完整报告。此核验不代表逐字段核验了全部 14 步。

四份原始 pack sidecar 的 SHA-256 和字节数见 [allowlist.json](allowlist.json)。采集侧 report 与 sidecar SHA 的绑定、27 文件回读及完整恢复审计留在私有证据中，本公开包无法独立复现这些私有链路。训练 YAML 不在本包；已存档于[既有私有归档的固定 blob](https://github.com/xingleixu123/mindformers-private/blob/e3f6f182e05bd6dde80f635c75c1455d20b329a3/private_review/archives/performance-final-20261030/blobs/9c5ae59905917bf43dce0bf04ff0537d6471f5088cbe9a5afb2cf25481bcc8bc.yaml)，其访问受该私仓权限限制。

离线校验（Python 3，当前目录为本目录）：

```text
python -B audit.py
```

该命令校验四份 sidecar 原字节 SHA、64 包的 offset/shape/标签/source 和 32 次 replica PG 顺序配对，并重写 `audit.json`。`pack_observer.py` 与两个 patch 是观察机制的可审查材料，不构成可直接执行的完整部署。粗扫描和 allowlist 仅供人工复核，不能保证没有敏感信息或授权问题。


## 设备 trace 的证据等级

`trace/raw-rank-0.json`、`raw-rank-2.json` 保留第一次保守解析的所有原始关联和 unknown；`trace/refined.json` 是可逐项复核的本地精化，没有重新采集设备数据。一个广播调用产生 1 个 HCOM 以及数个 EVENT RECORD/WAIT 的不同设备任务，不等于出现多个相互冲突的广播。精化只接受唯一 HCOM 目标、唯一 source/finish/端点和 label 容器、唯一 CSV，以及与原始 sidecar 完全相同的 dtype/count/PG/source。其他关联仍保留。CSV Step Id 在本次数据中做一致性检查，训练 step 的依据是实际 pack ID 和 sidecar。

Work.wait 关联只证明该原有调用发出了可直接对应的 EVENT WAIT；没有从时间或 stream 邻近关系重建 EVENT RECORD→WAIT 的 Event ID 依赖。对应 trace lane 上的四处 EVENT WAIT 区间并集，rank 0 三步分别为 78.425、78.422、78.417 ms，rank 2 为 127.614、131.069、126.165 ms。这是各 rank、各步、各 trace lane 的描述统计，不能跨 rank 相加，不能直接当作 HSDP/FSDP 整步差的可加分账或完整关键路径。实际 CPU Work.wait 范围耗时也不代表设备等待耗时。未将每个 unpack copy 按顺序或大小分配到单个 FQN。

两条路径需要区分：这里直接观察的是更新后参数的 replica 广播；last2d-sharded NS update 的 compute owner、内部 pack 和设备计算负载还需另证。本次 metadata 已证明本地专家 batch 的 ownership：rank 0/1 负责 gate/up，rank 2/3 负责 down；这本身不证明算量或时间失衡。没有据此提交算术或通信调度修复。

## 本轮固定标识与恢复核验

- 运行标识：`perf-hsdp-pack-profile-r1`。完整 source manifest SHA256：`12d81630ed5292cf4b12ed6ee9b70c401a9f5484f2d3817dc9d9290cb0f82aa2`；其母 profile manifest 为 `74050203e4db77d71979e006f2b0a19ebc346836ffaf24ce61baf97911950e78`，原性能 manifest 为 `b18fef99d3d55649c439d865718db90cfc1d0b1f6742bad1f7c87fd7d9a9a8cb`。两份 optimizer 源 AST 去除观察调用后与冻结源相同；增加的 helper/注解仍会扰动 timing。
- 私有 YAML SHA256：`9c5ae59905917bf43dce0bf04ff0537d6471f5088cbe9a5afb2cf25481bcc8bc`；没有公开 YAML，也没有修改既有 YAML blob。数据前缀 `mmap_deepseekv3_datasets_text_document`，bin SHA256 `6b925f342504c3eae80c5bde3163ff5924ad36b262bc6a276839a6f3f817ea3c`、idx `d535e502d62368e7a195b27a9a717543534ee1eaa9b4f9ce45f4d08d746cb5cd`。
- 原 checkpoint producer `33096a1030012777a5f71502dee009a7dd409bf9`；第 1500 步完整状态独立核验 258 项角色及组/extra 通过，外部报告 SHA256 `9f09a02970dbc1fc308ad2a4410ba21a313f28a469c5e930fdec9089258e20b1`。14 步实际读取的全局 sample index 恰为 6000–6055，前两步逐字段 hash 核验；原始文档长度不一，不称所有 idx 项均为 4096。
- 四个 rank 完成 14 次更新、finite loss/grad norm；新 HCCL 范围 `57260–57275` 启动前逐端口检查，本轮实际通信成功。旧 health receipt 只验证其原端口范围，不混称本轮健康实测。最后一次 read-only topo 显示所用卡经 HCCS 互连；离散资源观察不证明全程独占。
- 原始闭合 profiler 产物 527 文件、684,890,333 bytes，留在任务远端；完整 inventory SHA256 `fd3b73366a443e5421cb89bcff8e246829ce001f8f72a31385f481dc7678f759`。本包不复制全部 trace，只保留可追溯源 SHA 和必要的抽取结果。新诊断不是 MF/HP 千步验收，也不是无 profiler 性能比较。

本轮继续遵循用户“先不画图”的指示，未新画图或操作网站，既有页面、输入数据、配置和预览保留。
