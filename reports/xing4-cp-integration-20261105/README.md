# #1439 整合 #1425：检视与 NPU 回归

本轮在现有分支追加 [1caf044a](https://github.com/mindspore-ai/hyper-parallel/commit/1caf044aa35f4e69ade9938039f587281f14a3be)，原有三个提交不改写；检视评论与线程状态将在发布后逐项核验。当前候选仅 6 个文件，122 行新增、60 行删除；HF transformer 中的三份官方 Python 文件及测试/配方均未修改。

## 设计与依赖

- 将 #1425 两处公共 MLA CP 回调实现带入 #1439；现有 Xing4 adapter 已承接旧 Xing40 的模型语义，不复制旧目录或引入 #1413。
- 暴露已有 strategy、overlap、ulysses_degree、kv_exchange；默认仍为 Ulysses。通信调度与计划校验复用公共实现。
- QKClip 注册到 Muon 原生 post hook，要求两种上投影均为 Muon 管理的可训练参数；在 FP32 主权重 copy-back 之前执行，不再依赖 #1455。
- swap 构造期拒绝已有 per-instance step hooks，完整支持独立跟踪 [#1459](https://github.com/mindspore-ai/hyper-parallel/issues/1459)。没有声称支持后续注册、全局 hooks 或事务性多叶包装。
- MTP 跨片 token 位移和各深度分母仍保留模型 adapter 边界，Trainer 负责原有微批/整步权重。最新 master 尚未提供可以直接替代这些模型语义的接口。
- 最新 master c75f5f9 相对原 base a9fa47d6 仅改 CODEOWNERS；#1453 从 eead893c 变为 ac4aea64，仅测试文件有变，四个训练生产文件与冻结 d780 相同。

详细审查：[CP 范围](cp-audit.md)、[最新意见逐项答复](review-response.md)、[最终代码复核](final-code-review.md)、[#1453 依赖复核](dependency-1453-audit.md)。

## 已锁定的实验

两端 4096 MLA、1 Dense + 1 MoE + 1 MTP，融合 mHC、clamp/CMHC 关闭。单卡 GBS2/LBS1，四卡 GBS4/LBS1。MF 基线 formal 52f614ee / HP-MS r1.0.0 c69f8f70 / MS2.10.0 / CANN9.1.0 B243；HP Python3.12.14 / Torch2.12.0+cpu / PTA2.12.0.post3 / CANN9.2.0 B237。学习率 horizon7500、warmup1500不变，短跑只提前停止。

这是固定依赖的组合源码，不能将任意单一 Git commit 当作整个运行时。单卡和四卡分别从保留的571d、d780组合加显式文件 overlay 构建；1514文件逐一核验。源 manifest、实际复现命令和卡号在 regression/ 保存。公共 CP 两文件原本已存在于冻结运行时，因此本轮既是消除产品 PR 缺失依赖，也有 Muon hook/ownership 等实际增量。

## 单卡恢复回归

从真实 MF1500 更新后的完整 checkpoint，HP独立运行1501–1505。65模型项、91优化器槽、61FP32主权重别名、调度器1500、sampler consumed3000核验；10微批×4字段共40项与原连续MF及上一轮HP的 dtype/shape/hash 相同。主代理读取原始两端 ledger 独立重算：

| 指标 | mean(HP−MF) | abs(mean) | 正/负/零 | MAE | max abs（step） |
|---|---:|---:|---|---:|---|
| LOSS | -0.0005126953125 | 0.0005126953125 | 1/4/0 | 0.000657272338867 | 0.00157189369202（1503） |
| MTPLOSS（原始未加权） | -0.0000241160392761 | 0.0000241160392761 | 3/2/0 | 0.000454652309418 | 0.00111365318298（1503） |

[逐步原值](single-mf-hp.csv)、[独立核验](root-single-verification.json)、[核验脚本](root_verify_single.py)。这5步是集成回归，不替代完整1000步验收。旧571d的完整1000步结果保留原版本归属。

## 四卡与性能

四种布局各完成6–105共100步，从同一FSDP原生step5恢复。所有rank指标逐步一致；每配置400个全局样本、CP全长切片及TP副本字段均核对。四种布局均有独立全局DCP读回证据。FSDP在rank-local producer相等之外，新增完整checkpoint文件SHA及metadata覆盖核验，逐tensor核对四rank的260模型片段、364优化器槽、optimizer bytes和FP32 main alias；原始现场gate的范围没有改写。

| 对照（待测−FSDP4） | 指标 | mean | abs(mean) | 正/负/零 | MAE | max abs（step） |
|---|---|---:|---:|---|---:|---|
| ep2 | LOSS | +5.77339908434e-05 | 5.77339908434e-05 | 54/45/1 | 0.000311141544104 | 0.00100443018637（39） |
| ep2 | MTPLOSS | -8.23393602514e-05 | 8.23393602514e-05 | 42/58/0 | 0.000397852809632 | 0.0013090777988（90） |
| cp2dp2 | LOSS | +5.70461456556e-05 | 5.70461456556e-05 | 60/40/0 | 0.000353332133248 | 0.00119597826968（105） |
| cp2dp2 | MTPLOSS | -8.43603727806e-05 | 8.43603727806e-05 | 45/55/0 | 0.000442811794698 | 0.00169098377228（37） |
| tp2cp2 | LOSS | +1.98904280331e-05 | 1.98904280331e-05 | 53/47/0 | 0.000343661245191 | 0.00109749532553（94） |
| tp2cp2 | MTPLOSS | -1.86270310278e-05 | 1.86270310278e-05 | 52/48/0 | 0.000408929421025 | 0.00135411737544（12） |

GBS4/LBS1：FSDP4/EP2/CP2×DP2/TP2×CP2 的梯度累积分别1/1/2/4。FSDP重计算主干和MTP，其他布局重计算主干；TP使用loss parallel。不同重计算范围已记录，不能把跨布局耗时直接归因为单一并行策略。

[四卡逐步CSV](fourcard-versus-fsdp.csv) · [统计及样本核验](root-four-verification.json)。这是100步回归，四卡1500→2500长程仍未完成。

在129卡0–3串行独占执行四次短窗口，GBS4/LBS1/4096一致。每次从同一step5恢复，6–10预热、11–25统计；step_begin先同步并reset峰值，step_end同步后读耗时/显存，每步取四rank最大耗时。时间包含实际样本拷贝和哈希等验证开销。旧控制为冻结d780组合，**不是紧邻f96父提交**；因此这是组合更新的风险检查，不是本次六文件单变量基准。

| 布局/版本 | 秒/步 mean ± sample std | 输入token/s | 峰值allocated GiB（四rank最大） | reserved GiB | 新−旧耗时 |
|---|---:|---:|---:|---:|---:|
| perfold2-fsdp4 | 0.832080 ± 0.002330 | 19690.41 | 18.355568 | 20.779297 | — |
| perfnew-fsdp4 | 0.832285 ± 0.002035 | 19685.58 | 18.355568 | 20.779297 | +0.025% |
| perfold2-cp2dp2 | 1.231914 ± 0.003144 | 13299.63 | 20.598791 | 21.789062 | — |
| perfnew-cp2dp2 | 1.227035 ± 0.003868 | 13352.51 | 20.598276 | 21.789062 | -0.396% |

四rank的逐卡allocated/reserved、min/max/median/CV、实际逐步值与SHA见 [独立性能统计](root-performance-verification.json) 和 [逐步CSV](performance-steps.csv)。FSDP峰值allocated不变，CP最大allocated只差约0.53MiB，最大reserved相同。该单次短窗口未发现明显性能/显存回归，但不能据0.4%宣称稳定提速，也不能直接以不同GA/重计算的两个布局评价策略优劣。


## 已有检查与边界

- CPU复用247项既有用例：246项及79子测试通过；1项因CPU环境缺少torch_npu在导入阶段失败。真实PTA/CANN容器补跑相关CP用例及7项外部callback检查，共15 passed、16 subtests passed，包含该失败项。
- 原生Muon hook的CPU和真实NPU探针通过：FP32重绑、独立缩放公式、三步回写、冻结/AdamW归属拒绝、已有hook的swap构造拒绝及无hook的小矩阵step。
- 独立Xing4 callback CP2、CP4真实NPU探针检查输出、QKV梯度、逐头最大值及跨rank MAX；这与完整训练矩阵是不同证据。
- 公共MLA原ST的二维mask用例在旧/新源码均失败；已确证契约冲突。其修复独立为 [issue #1460](https://github.com/mindspore-ai/hyper-parallel/issues/1460) / [Draft PR #1461](https://github.com/mindspore-ai/hyper-parallel/pull/1461)，仅三行；只将reference调整为4D而CP保持2D时，旧版失败、修复变体两rank通过。原始失败不会记为通过或用放宽阈值掩盖。
- 六文件lint无新增诊断，保留module_replacement两项既有warning；不宣称整体lint或CI已绿。
- 本轮使用 GPT-6.1 Terra、medium 委派检视和证据整理；API无独立Fast设置，未宣称开启。

YAML继续保留在已有私有归档；不上传到公开证据分支。任务结束时仅清理本轮可重建临时源、重复输出与未再需要的wrapper，保留checkpoint、命令、日志和源manifest。

## 复核入口及版本边界

~~~sh
python root_verify_single.py
python root_verify_four.py
~~~

单卡核验所需MF及上一版HP的1501–1505行按原文摘取并一并提供；[摘取来源及SHA](single-reference-extraction.json)。四卡脚本直接读取四种布局各四rank的逐步记录。restore-verification和全局DCP报告提供恢复证据，原始训练和核验脚本/命令继续本地归档。重新运行统计无需NPU；重新训练仍需要授权的数据、完整checkpoint和[私有YAML归档](https://github.com/xingleixu123/mindformers-private/blob/97e3e5caa0d5e824226e1b4d328fe11af6c36180/archives/xing4-ce2dbe84-87dac1da-singlecard/README.md)。

[本次commit与实测源绑定](commit-binding.json)。manifest里的commit/tree是冻结组合的起点，不代表所有overlay；完整文件SHA与provenance才是实际运行身份。原有571d完整单卡1000步[报告](https://github.com/xingleixu123/hyper-parallel/blob/dd24a3d85b620af36ae60d6ea036e76bd444f7c4/reports/xing4-reuse-precision-20261105/README.md)保持原归属。
