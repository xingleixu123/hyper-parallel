# PR #1439：1105 README 检视增量及回归

PR仅追加 [7f4a3fd7](https://github.com/mindspore-ai/hyper-parallel/commit/7f4a3fd7cac6da3c465743b57d832042bcc4ae29)，保留原4个提交，总计5个；不amend、rebase或强推。6个逻辑文件，+76/−13，官方HF三Python文件的原名和原始字节不变，未添加UT/ST或配方。

[逐项检视决定](review-decisions.md) · [六文件复核](final-six-file-audit.md) · [最新master/依赖核查](upstream-audit.md) · [MF/HP YAML私有归档](https://github.com/xingleixu123/mindformers-private/blob/97e3e5caa0d5e824226e1b4d328fe11af6c36180/archives/xing4-ce2dbe84-87dac1da-singlecard/README.md)

## 实际修改及边界

修复 [路由精度 #1464](https://github.com/mindspore-ai/hyper-parallel/issues/1464)：计数保持同名persistent int64，偏置保留FP32 buffer，复用框架精度转换helper并补充局部buffer声明校验。整数比较保留负载方向，不改变MoE辅助目标和优化器数学。

[专家替换 #1456](https://github.com/mindspore-ai/hyper-parallel/issues/1456)继续复用公共GroupedExperts/#356：显式SwiGLU策略缺消费能力时拒绝构建，不重新实现激活。补充条件配置、chunk size、MLA sparse/dropout输入检查。可选torch_npu延迟到kernel处，不增加CPU算法。

删除透明forward的检视建议被真实构建反例否决：公共替换器要求保留ModuleList调用签名。失败发生在恢复前，原始日志保留，最终恢复该包装并验证签名正负例。没有为了减行数删除必要契约。

## 单卡和并行短程回归

| 对照 | 指标 | mean差（保留符号） | abs(mean差) | 正/负/零 | MAE | max abs |
|---|---|---:|---:|---:|---:|---:|
| HP−MF，1501–1505 | LOSS | -0.000511884689331 | 0.000511884689331 | 1/4/0 | 0.000522947311401 | 0.00107884407043 |
| HP−MF，1501–1505 | MTPLOSS | -0.000293707847595 | 0.000293707847595 | 1/4/0 | 0.000544810295105 | 0.000697195529938 |
| EP2−FSDP，6–105 | LOSS | -1.08641043189e-05 | 1.08641043189e-05 | 45/55/0 | 0.000340142938175 | 0.00102043151855 |
| EP2−FSDP，6–105 | MTPLOSS | -2.0559696966e-05 | 2.0559696966e-05 | 48/51/1 | 0.00044953476293 | 0.00156172583657 |
| CP2×DP2−FSDP，6–105 | LOSS | +5.70883669137e-05 | 5.70883669137e-05 | 57/43/0 | 0.0003198667064 | 0.00112295150757 |
| CP2×DP2−FSDP，6–105 | MTPLOSS | +5.37919951143e-05 | 5.37919951143e-05 | 56/44/0 | 0.000421954895357 | 0.00134375854876 |
| TP2×CP2−FSDP，6–105 | LOSS | +7.31548303816e-05 | 7.31548303816e-05 | 56/44/0 | 0.000352057047353 | 0.00118184089661 |
| TP2×CP2−FSDP，6–105 | MTPLOSS | -3.0973489035e-05 | 3.0973489035e-05 | 52/48/0 | 0.00044274228625 | 0.00132184683236 |


LOSS为主LM，MTPLOSS为未乘0.1的原始MTP；差值为HP−MF或待测布局−同版FSDP。每个对照两种指标均自然出现正负。0.005约束完整验收窗口的abs(mean)，不是单步阈值。**本次5/100步是增量回归，不等于新的完整1000步验收。** 旧571d完整1000步仍保留原版本归属，四卡1500→2500长期验收未在本次完成。

原值：[单卡CSV](single-mf-hp.csv)、[四卡CSV](fourcard-versus-fsdp.csv)、[单卡独立复核](root-single-verification.json)、[四卡独立复核](root-four-verification.json)。没有平滑、删步或取绝对值后再平均。

单卡从真正MF1500完整状态恢复：65模型项、91优化器槽、61 FP32主权重别名、scheduler1500、数据游标3000；两条旧float32计数仅在可精确往返的整数值条件下转换。其余状态保持严格校验。10微批×4个实际数据字段与MF/旧HP共40项一致。

四卡从同一原生FSDP step5 checkpoint独立恢复，四布局各100步/400全局样本。各rank指标一致，CP片段覆盖4096且TP副本相同。FSDP另以CPU独立读回全局DCP，624项绑定（260模型、364优化器槽）通过，并绑定本次各rank恢复文件hash。重分片/恢复门和统计见原始JSON。

## 环境、配置和源码身份

129卡5单卡，物理0–3四卡。HP Python3.12.14、Torch2.12.0+cpu、PTA2.12.0.post3、Transformers5.21.0、CANN9.2 B237、910B3；MF formal52f614ee / HP-MS r1.0.0 c69f8f70 / MS2.10 / CANN9.1 B243。4096 MLA、1Dense+1MoE+1MTP、融合mHC，clamp/CMHC关闭。数据前缀mmap_deepseekv3_datasets_text_document，实际样本和字段hash见逐rank记录。

单卡GBS2/LBS1；四卡GBS4/LBS1，FSDP4/EP2/CP2×DP2/TP2×CP2的GA分别1/1/2/4，TP使用loss parallel。FSDP重计算主干与MTP，其余重计算主干。train_iters2500，LR horizon7500、warmup1500不压缩。

每个实际组合1514文件锁定；[提交与源码绑定](commit-binding.json)证明六个Git blob和两套实测源逐字节一致。单卡manifest 0b2f30d2501c5cf8b701015200bed5b4ed46e9a95153befd4575370643aa0b2c；四卡manifest 2b08a0f01a77cc078730275463a9072b65e8f6b095bb7b21a5952a228f55d0ae。完整运行时包含单卡#1441或四卡#1453等依赖，不能冒称裸PR一个提交。最新master审查到7b1ea07e，不改写已锁定实验的版本身份。

## 模块和公共回归

真实NPU [20项专项](probe-final.log)通过，含BF16计数/小量bias更新、padding/重计算、旧DCP恢复、非空SwiGLU前反向及缺能力拒绝、签名正负例。另[四逻辑rank BF16/FSDP](fsdp-bf16-probe5.log)验证物化、DCP恢复、前反向、SGD与公共MoEMonitor更新/清零；脚本SHA 300d0b0c37027afcf4eb2689d869c273b1099cc9c4054d81a317ca8961e454f4。

补充BF16探针的物理映射未经过原控制器验证，并与一次EP恢复阶段产生设备竞争，EP被admission gate中止、未训练。该结果只用作功能断言，不用于性能；最终EP在全部空闲后用原控制器重跑。旧probe4原脚本未保留，不能用最终probe5源码冒认为其身份。

CPU已有24项/14子测试通过；外置12项通过、8项因PTA依赖跳过并在真实NPU补齐。DSv41一项因缺Pillow收集失败，不计通过。同配置pylint父/新版各21项，新增0，退出码均22，[诊断对照](lint-final-comparison.json)不能称lint绿。外部CI另查当前head状态。

## 性能与显存风险观察

| 配置 | 前版/本版平均step秒 | 变化 | 本版中位数 / p95秒 | 本版名义tokens/s | 本版各rank峰值allocated GiB | 本版各rank峰值reserved GiB |
|---|---:|---:|---:|---:|---|---|
| fsdp4 | 0.831288 / 0.830290 | -0.120% | 0.830016 / 0.832581 | 19732.9 | 18.355569, 18.355569, 18.355569, 18.355563 | 20.777344, 20.777344, 20.779297, 20.779297 |
| cp2dp2 | 1.227967 / 1.225337 | -0.214% | 1.225782 / 1.226895 | 13371.0 | 20.596020, 20.598277, 20.596020, 20.598271 | 21.710938, 21.789062, 21.710938, 21.789062 |
| tp2cp2 | 3.530196 / 3.529201 | -0.028% | 3.530364 / 3.532825 | 4642.4 | 21.349693, 21.354910, 21.350632, 21.353937 | 23.320312, 23.144531, 22.761719, 22.767578 |


统一取已完成100步的86–105步，前80次更新排除；每步同步、取四rank最大耗时，计时包含样本hash，吞吐为GBS×4096/平均step。显存是Torch NPU allocator的逐步allocated/reserved峰值，非整卡driver HBM。[全部时序/显存原值](performance-raw.csv)、[波动与统计](performance-diagnostic.json)。

这是与前轮1caf同配置、同起点、同样本窗口的历史配对观察，未随机化重复，不宣称稳定提速或完全独占benchmark。独立CPU DCP读回与CP前半段重叠，所列末20步不在该读回窗口；BF16探针和失败EP均排除。

EP原控制器没有同口径的逐步同步耗时/allocator字段，本轮只报告其数值回归，不由进程总时长估算性能。TP×CP的rank0峰值reserved从23.138672增至23.320312GiB（+186MiB），allocated峰值基本相同；这是实际缓存池占用变化，未做因果归因，不能宣称所有显存完全不变。

两个64专家路由器的计数常驻存储从FP32改int64，理论净增512字节；BF16初始化时再加偏置保留FP32，相对旧BF16状态合计净增1024字节。临时整数scatter和allocator对齐另计，不把理论常驻差当实测峰值收益。

## 复现与归档

统计复算：python root_verify_single.py、python root_verify_four.py；性能计算：python verify_performance.py。专项：锁定源码PYTHONPATH下TEST_DEVICE=npu:0 python -m pytest -q -p no:cacheprovider test_review_probe.py。完整训练控制器及版本/配置/数据身份由现有私有归档和本地运行目录保存；source manifest、overlay和旧MF1500/原生FSDP step5关键checkpoint保留。

清理仅针对已失败、归档且可重建的本任务旧stage/重复产物；最终源、关键checkpoint和所有有效/失败审计日志不删除。总跟踪更新#353，#304不变。

本地single是准备时误复制的旧证据副本，最终结果仅引用single-run。删除该冗余目录的命令在创建进程时被自动审批以blocked by policy拒绝，未执行、未重试绕道；该副本仍保留，不作为本轮证据。
