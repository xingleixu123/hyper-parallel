# CPU归一化舍入探针与旧诊断文件清理

这组证据不增加已确认缺陷或PR，不代表CP训练验收。

## CPU单变量观察

实际调用固定源`18602c4`的`projected_cross_entropy`和Transformers CE。参考与两个分区保持每块16位置、相同初值及标签，覆盖五组不均衡监督数（含零监督分区）、8个seed、FP32/BF16两种dtype，共80例。局部均值再乘local/global与直接使用global分母具有相同数学目标。

| 原有局部均值再缩放，对匹配全局分母参考 | FP32，40例 | BF16，40例 |
|---|---:|---:|
| loss非零差例数/最大绝对差 | 6 / 4.76837158203125e-7 | 6 / 4.76837158203125e-7 |
| hidden梯度非零差例数/最大绝对差 | 8 / 1.862645149230957e-9 | 0 / 0 |
| weight梯度非零差例数/最大绝对差 | 8 / 1.4901161193847656e-8 | 0 / 0 |

通过loss回调直接使用全局分母的候选在80例中全部loss及两类返回梯度差0；没有修改生产源码。独立复核14项检查通过，结论仅是该CPU路径存在小量FP32舍入。BF16日志明确记录aarch64 MKLDNN不支持而回退BLAS；本组BF16返回梯度未差，不能据此确认NPU CP200步漂移根因。没有CP通信、Trainer权重或实际模型训练；参考也不是FP64 oracle。镜像/源码及运行凭据已固定，但本次没有额外保存Transformers依赖文件哈希，溯源边界保留。

CPU容器没有设备映射，未初始化NPU；exit0后已移除。详见[80例原始结果](f637-cp-denominator-rounding-cpu.json)、[实际日志](f637-cp-denominator-rounding-cpu.container.log)、[独立复核](f637-cp-denominator-rounding-independent-review.json)和[脚本](check_cp_denominator_rounding.py)。

## 125旧参数探针的无损压缩

五份已完成200步的旧运行各有约2GB参数探针JSONL。本任务以低优先级、1MiB流式缓冲压缩到gzip，逐份解压读回并核对原始字节数及SHA；再核对原文件身份、时间及没有打开的进程描述符后，才移除未压缩文本。五份全部通过，释放**6,136,618,679字节**。数据仍完整保留在gzip；本轮step200 checkpoint、其余运行证据、共享数据和用户镜像不在清理范围内。

[五份原始/压缩SHA与字节数凭据](runtime-probes-lossless-compaction-20261016.json) · [执行脚本](compact_runtime_probe_evidence.py) · [传回本机SHA核验](cleanup-and-cp-rounding-evidence-local-verification.json)。压缩前后校验由执行脚本记录，未把已移除原文本冒称为发布后重新核验的文件。
