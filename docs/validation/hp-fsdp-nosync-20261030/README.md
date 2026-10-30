# FSDP 梯度同步回归：CPU 证据（草稿）

本目录归档已完成的有界 CPU 回归与复现证据。所有数值输入和梯度均来自合成小模型。证据覆盖两 rank CPU/Gloo SUM 的特定嵌套 scheduler 测试，不代表完整模型、NPU/HCCL、性能或四 rank HSDP replica-only partial 场景。

## 固定源与修复

CPU 源固定于 HyperParallel `5cde7c54ce3da8be8689d0783cf9b70d5306117a`，其 2659 项文件散列见 [source-manifest.json](source-manifest.json)。[pins.json](pins.json) 记录源清单及原始/修复 scheduler 的哈希。red 使用固定源；green 使用唯一的只读 scheduler 单文件 overlay。

修复将同步开关从 false 变为 true、且处于 BACKWARD 并存在真实 pending 梯度时的收尾限定到该转换。原文件为 [original_scheduler.py.txt](original_scheduler.py.txt)，差异为 [hsdp-scheduler.patch](hsdp-scheduler.patch)；三个待入库文件按原字节存放在 [repo-patch](repo-patch/)，以 `.py.txt` 结尾。诊断 worker 原字节位于 [diagnostic/worker-14e249.py.txt](diagnostic/worker-14e249.py.txt)，同目录保留 red/green 的源和配置描述。

## 两 rank 诊断

四次独立运行分别是 red/green × fusion false/true，每次包含 2 rank × 4 case 的八份 JSON；因此总共 **32 份 rank/case 记录**，不是 32 次独立训练。

| 运行 | fusion | rank/case 结果 | 退出码 |
| --- | --- | --- | --- |
| red | false | 6/8 通过，两个 rank 的 `micro0_only_false` 失败 | 1 |
| green | false | 8/8 通过 | 0 |
| red | true | 6/8 通过，两个 rank 的 `micro0_only_false` 失败 | 1 |
| green | true | 8/8 通过 | 0 |

red 中，提前 microbatch 使用、最后 microbatch 跳过的 `early.weight` 未取得最终主梯度，未分片梯度仍 pending；green 对该合成例与独立 SUM oracle 一致。四组输入及对照范围见 [诊断报告](cpu-actual-v1/report.md)、[集合摘要](cpu-actual-v1/collection-summary.json) 和各组 `output/` 原始 JSON。[参考语义与公式复核](reference-semantics-max-v1.md) 给出手算和外部源码版本边界。

## 仓库回归与来源探针

[仓库回归报告](cpu-repo-exact-v1/report.md) 和 [集合摘要](cpu-repo-exact-v1/collection-summary.json) 保留四次实际运行：原 scheduler 的新 worker 两 rank 在 fusion=false 的目标案例报缺失主梯度；green worker 两 rank 各通过，覆盖两种 fusion 与四组 controls；原 hook state-machine 和新 launcher 的 green 运行通过。red worker 在首个失败处退出，并未证明其 fusion=true 分支已运行。launcher 的两个 rank 日志在 `repo-green-launcher-r1/flat-logs/`。

[独立 import 探针](cpu-repo-exact-v1/repo-green-origin-r1/origin-summary.json) 记录同镜像、挂载与环境中源码导入位置及版本。探针是单独进程，不是原仓库测试工作进程内的 import 记录；其原始 stdout/stderr、argv 和 inspect 提取件另行保留。

## 可核验性

`SHA256SUMS` 列出包内除自身外每个文件的 SHA-256，可在支持该格式的环境中运行 `sha256sum -c SHA256SUMS`。`copy-audit.json` 记录逐个原样复制文件的来源、大小和哈希，以及派生文件说明。四组诊断、四组仓库回归和独立探针的九份原始 `artifact-hashes.json` 均完整保留；封装前对其中所有原始条目的大小和哈希做过核验。32 份诊断 JSON 和 launcher 四份 rank 日志也与原 receipt 中的嵌入副本逐字节核对。

各 `receipt-extract.json` 保留原 receipt 的 inspect、源校验、退出、运行标识及嵌入文件的大小/哈希，并标明原 receipt SHA；它是提取件，不是原始 receipt。各组的独立 stdout、stderr、argv、summary 和实际结果以原字节保留。原始 receipt、SSH 返回的 Base64 数据封套、传输输入和收集封套有意不复制；原始 archive 仍在本地。原始 `artifact-hashes.json` 因此还索引了未装入本包的文件，不能要求其所有路径都在本包中存在。原始报告中对 archive 内容的描述也应按这一差异理解。

包内不包含 YAML 文件、实际训练数据、凭据、整套源码或大型 trace。