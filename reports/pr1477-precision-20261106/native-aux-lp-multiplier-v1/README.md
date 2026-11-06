# Synthetic AUX：原生LP复制倍率的修前／修后证据

这是独立、可复核的公开证据包，不覆盖旧精度报告。只包含synthetic fixture配方、真实原生诊断梯度与必要回执；没有私有训练YAML、真实MF权重/样本、checkpoint或凭据。原始stdout/stderr及完整controller均留在本地，包内controller-closure明确是字段投影并保留原SHA，不冒称原件。

**一条CPU核验命令：** `python -B verify_evidence.py`。只需Python标准库，不导入Torch/HP、不联网、不使用NPU，也无需生成61.8MB fixture。它读取原始梯度blob与从原fixture精确提取的小型FP64 oracle，复算16份rank结果。

## 结果与直接证据

四个case均为CP2×TP2、DP1、SP=false、GA4，独立新进程/PG/run、0参数更新；只切换LP布尔值及后续修复source，fixture、AUX系数和FP64目标不变。

| source | LP | 原生FSDP SUM factor | 对独立1×目标的最小二乘倍率（rank0/1；rank2/3） | 与匹配目标的relative L2 |
|---|---|---:|---|---|
| 原496 | false | .25 | .99999992548；.99999986223 | 2.97e-7；5.70e-7（1×） |
| 原496 | true | .5 | 1.99999985096；1.99999972446 | 2.97e-7；5.70e-7（2×） |
| 候选3292／f8ec增量 | false | .25 | .99999992548；.99999986223 | 2.97e-7；5.70e-7（1×） |
| 候选3292／f8ec增量 | true | .5 | .99999992548；.99999986223 | 2.97e-7；5.70e-7（1×） |

原source的LP=true最终梯度逐元素exact等于2×LP=false。候选两种LP设置的完整梯度文件均与原LP=false逐字节相同。真实before/after-token-normalization F32梯度仍保留；TP peers和修前后相同文件按SHA去重为4个blob，**每rank原result/complete及gradient-blob映射均保留**，没有合成或拟合NPU梯度。

原独立pair JSON为 `17bf7c48d1c7ed98d702a844f32ab61867037e3117c9f5c7632663b1b7f39c5b`；ROOT标准库独立结果为 `435552c3235dbe763361d0c42303c2b91110fcd5fff01b0a0ed6e88e8da4f4e1`；候选独立pair为 `414316a040906619520a761c9af8ec79b43b81447ccb0329eb6204365833f7bc`。它们的原字节均在model-audit/router-aux-native-reduction-probe对应目录；verification-result.json是本包新增的标准库复核，身份分开记录。

## 数学目标与真实路径

E64、H3584、top-k4、全序列4096、每CP真实2048＋safe padding64。fixture中首维2表示两个合成样本池，不是本探针的DP大小；4micro使用样本0/1/0/1。非前缀mask的CP有效数为[93,740]、[111,311]、[222,500]、[444,1234]，global数833/422/722/1678，整步N=3655。router统计全部真实token；这些foundation masks只确定外部微批权重。

独立FP64目标直接用CPU sigmoid、unbiased top-k计数与概率均值、autograd计算：

`0.0005 × sum_micro(n_global_micro / 3655 × E × sum_e(f_e × mean_token(P_e)))`。

目标objective为0.0005733037907340564，reference梯度L2为6.767933266957011e-6。参考不调用HP balance helper，也不使用未来训练梯度。原始reference JSON保留公式、tensor SHA、维度与预置容差；本包标准库复核使用更紧的1e-5 relative-L2区分1×/2×。

实际worker运行真实Xing4Router、planner/resolver得到的统计组、apply_sharding_plan/source_shard_info、FSDP2Manager及原生root backward scheduler。主输出乘0隔离AUX，再经原生CP mean、bind_aux_loss_scale、mean_global_loss和BaseTrainer.loss_normalize；不运行expert/GMM、attention、完整MoE、主CE、optimizer或checkpoint，不手动all_reduce/AVG/补偿梯度。每rank记录实际CP/TP组、TP Replicate源、native chunk和缩放因子；最终比较按actual chunk的全局offset/size切FP64目标。

两原case实际CP loss_sum反向倍率均为2、foundation绑定倍率为2×n_local。原LP=true改变native SUM factor后，独立且TP复制的AUX边产生2×；候选通过明确的replicated-router mean opt-in处理此边，并向context传真实LP/TP flags。没有盲目对DSA或所有loss除TP，也没有改系数使数值接近。

## 必须保留的边界

- 这是AUX-only原生因果证据，**不等于完整MoE、主CE、optimizer、SP或1501–2500整网精度通过**，没有同条件性能比较。
- 旧12-case router/boundary探针使用LP=false、外部倍率.5，并显式TP SUM后除TP2；它验证统计域及副本均值比较，不能代替本次LP=true原生最终梯度。原片段/文件SHA见provenance/old12case-boundary-excerpts.json，原结论保持该范围。
- 首次a1的ContextVar取值失败发生在task只读标量observer跨autograd线程取context，0更新、无可比较原梯度；属于观测脚本问题。a2只在forward context捕获同一state对象供hook读取，未改产品梯度算术。窄diff和CPU线程合同随包保存；不能把这个a1失败当产品bug或有效倍率结果。
- controller-closure是从本地原controller提取的闭合字段，含原SHA和clean rc0/removed/source前后一致；完整机器日志/控制器原件没有公开。历史ROOT脚本原件也保存，依赖原目录/完整fixture及原controller；本包可直接运行的入口是verify_evidence.py，不冒称已原样重跑ROOT历史脚本。

## fixture的准确生成与oracle提取

61,768,528B完整fixture不入包；SHA为 `a035d1bbb9afe561dc4a62f93586ae20653aa9a2f12d05d8760d7d4758e35d54`，reference JSON为 `ccda55fcad2ca45a64b5f07af98dfc5e8f232ff4ff25f2af945d25c52e14056f`。原生成脚本SHA `e6dfa7943b6aee503d1d779782719f360fbf99f5fcceddff3d28234a41e1ca05`，依赖sibling helper SHA `29fc81bfaca7d77483d308589b0a882f24ad582390455fd04e4d7a5eb8da2c0c`，seed147720261106、torch.set_num_threads(2)。

本次在独立临时目录实际复跑：Python3.12.3／x86_64／Torch2.12.0+cpu／safetensors0.8.0，**完整fixture和reference都再现原SHA，所有tensor dtype/shape/bytes exact**，见provenance/fixture-repro.json。临时61.8MB已清理，原文件未覆盖。这是该已测依赖环境的事实，不保证任意Torch、CPU架构或序列化版本都同字节。

如需生成，请先复制本包到可丢弃replay目录，勿覆盖原证据，再执行：

```bash
python -B model-audit/router-aux-native-reduction-probe/prepare_fixture_cpu.py
sha256sum model-audit/router-aux-native-reduction-probe/aux-native-fixture.safetensors
sha256sum model-audit/router-aux-native-reduction-probe/aux-native-reference.json
```

生成函数只使用Torch／safetensors／标准库，不import HP，不需MF数据。小oracle由**原a035文件**精确提取FP64 tensor payload，原tensor bytes SHA为 `c8303f90a3723134953f1249255770b023ace929b3332526cc440ed37497f501`；新单tensor容器SHA `f13b52c2423c0e1880b110f4e64409924680bdcfac2361b177468222ef7e1b8f`。只重写容器header，未重算/修改payload，见oracle/extraction-receipt.json。

## 固定源码与原生重放命令

provenance/source-set.json及原始source manifest记录原496：1477 `eb1121470917789ce641e3f12b4b6bad65b6ba54`＋1420 `bab14d2c57b269f3571fd39f55088d1fe80926f0`＋1470 `60a2dc219489f39f7aefa199bd6413ef5fcc775a`＋1453 `9b8ee660ce0a927d6a87072b6e4db18f00756401`＋JT364 `3b76915e6c87f4d1eb7d3383b3558b0439146a90`。1453测试hunk因1477上下文排除tests/**，原生产hunk原样应用/反向核验；不以whole-file替换覆盖1477生命周期增量。源码manifest仅公开HP源码路径/size/hash，无私有MF源码或配置。

候选为原496叠本地修复commit `f8ec544bef230383d62af5c365d60b6f723d761d` 的3生产文件增量，实际manifest `3292a5b93c7b62d4f97c3edf42572aeaca91a475057bb3540ca6bcf1b9eae245`。Git commit与交付manifest不是同一种身份；runtime-production.patch及两个manifest已保留。修复commit已推送[用户fork](https://github.com/xingleixu123/hyper-parallel/commit/f8ec544bef230383d62af5c365d60b6f723d761d)，尚未合入master。

在独立公共checkout按以下顺序重建（本包准备过程未执行Git/API/发布）：

```bash
git checkout --detach eb1121470917789ce641e3f12b4b6bad65b6ba54
git apply /path/to/evidence/repro/source-patches/dependency-1420.patch
git apply /path/to/evidence/repro/source-patches/dependency-1470.patch
git apply --exclude='tests/**' /path/to/evidence/repro/source-patches/dependency-1453.patch
git apply /path/to/evidence/repro/source-patches/dependency-364.patch
python /path/to/evidence/repro/check_source_subset.py --checkout . --export /new/source496
# 候选在同一throwaway checkout再加3文件增量
git apply /path/to/evidence/repro/runtime-production.patch
python /path/to/evidence/repro/check_source_subset.py --checkout . --candidate --export /new/source3292
```

实际NPU环境为Torch2.12.0+cpu／torch_npu2.12.0.post3／Python3.12.14／CANN9.2.0 B237，4×Ascend910B3。原worker481与候选worker只改source SHA及context flags，差异见repro/candidate-worker-from481.diff。它们保留实际环境、二进制SHA和physical-mapping-release握手；不能移除准入后声称原样重放。包不提供自动占卡或绕过握手的启动脚本，实际容器内计算命令是：

```bash
HP_ROOT_REVIEWED=PR1477_NATIVE_AUX_LP_REDUCTION \
torchrun --nproc_per_node=4 --master_port=<approved-port> native_aux_lp_worker.py \
  --loss-parallel false --output /work/output --execute-reviewed
# 第二个fresh run只改LP=true；候选另用native_aux_lp_worker_candidate.py和source3292
```

source按对应manifest只读挂到/work/source、生成fixture/reference/source-manifest只读挂到/work/scripts；必须绑定匹配的editable安装，并由独立资源控制器设置run/tag/设备/通信端口与mapping release。原握手还校mHC扩展二进制身份，但本AUX探针不执行mHC。无需真实MF权重、训练数据或私有YAML。公开原值核验不要求重跑NPU。

manifest.json及SHA256SUMS封印本包；provenance/original-file-index.json区分原件、按SHA引用的blob和只保留本地的controller。只准备文件，发布由ROOT另行执行。

此处1e-5只用于独立数学梯度复核，不是完整训练AUX诊断的新验收阈值。训练LOSS／MTPLOSS仍按完整1000步abs(mean(有符号差值))≤0.005；本包不宣称整网窗口通过。
