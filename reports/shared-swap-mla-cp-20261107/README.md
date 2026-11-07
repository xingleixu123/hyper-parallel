# 公共 optimizer swap 与 MLA CP 拆分回归

源基准：mindspore-ai/hyper-parallel master c086152596226c99f5c06597491ee94b4c26d26c。两份候选分别只修改 sources/file-manifest.json 所列文件，互不依赖，不包含 models/xing_4。此报告保留合成输入的逐rank结果、CPU输出、源SHA和实际执行worker，不包含模型权重、训练YAML或真实训练数据。

## 实际结果

| 范围 | 结果 |
| --- | --- |
| swap 现有 CPU 全文件 | 130 tests、26 subtests 通过；真实 AdamW/Muon pre/post hooks 在副作用前被拒绝，移除后可包装。 |
| MLA CP 现有 CPU 全文件 | 10 tests、18 subtests 通过；wrapper透传、真实CP1投影/attention前后向、非callable及默认晚绑定。 |
| 单卡真实 NPU | 14项通过：8项RoPE（BF16/FP16、两布局、多头Q/单头K），6项attention（dense/packed/4D mask）前后向。 |
| swap NPU 新旧 | master静默跳过已有hooks；候选构造拒绝；无hook的3步真实state swap参数/step/动量槽SHA完全相同，确有2个CPU mirror及copy stream。 |
| 四卡真实 NPU | 下列5配置，每项4rank均通过；2微批＋非重入checkpoint，核对输出、逻辑输入梯度、参数梯度及TP-local头信息。 |

| TP | CP | Ulysses | 路径 | overlap | 回调 | 输入 |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 2 | 2 | expanded_ulysses | 否 | 默认 | dense |
| 2 | 2 | 2 | expanded_ulysses | 否 | 自定义 | dense |
| 2 | 2 | 2 | expanded_ulysses | 是 | 自定义 | dense |
| 2 | 2 | 2 | latent_kv_head | 是 | 自定义 | packed |
| 1 | 4 | 2 | latent_kv_head | 是 | 自定义 | dense |

回调使用反向旋转位置及非均匀逐头Q倍率，未分片参考按相同数学定义计算，核验回调实际调用及正确的TP-local切片，不复制通信/规约轨迹。四卡最大参数梯度relative L2为0.003976614784545177，小于事先固定的0.025；最大输出绝对差0.00390625，逻辑输入梯度绝对差1.9073486328125e-6，合成目标loss差1.0371208190917969e-5。逐元素BF16 atol0.004/rtol0.06，FP16为0.0005/0.012；未调整容差。

这些是公共模块回归，不是整网2500步或未来私仓组合的1000步验收；不将0.005完整窗口均值指标用于模块逐元素比较。不外推HSDP、未测拓扑或吞吐/峰值显存收益，NPU JSON中峰值仅为诊断。

## 环境与执行边界

- CPU：Python3.12.3、Torch2.12.0+cpu、pytest9.2.0；候选各有独立editable源。
- NPU：Python3.12.14、Torch2.12.0+cpu、torch_npu2.12.0.post3；CANN toolkit/910B ops 9.2.0、内部版本V100R001C12SPC001B237，Ascend910B3。
- driver26.0.rc1、内部版本V100R001C10SPC001B231、ascendhal7.35.23；只读核验此次实际挂载，未升级共享环境。
- 四卡129物理4–7→容器逻辑0–3；单卡算子物理4，swap新旧顺序使用物理6。最终任务串行、逐次准入、独立进程/输出/端口。
- 镜像、source archive、worker/controller SHA和真实退出码见runtime-summary.json。必要字段导出，原命令/monitor/receipt本地保留。
- 入口修正仅涉及tests导入作用域、容器逻辑编号。0–3四卡probe device_count0，未反复试跑。
- 4–7原probe初始化全部成功，但旧controller将退出己方PID的npu-smi残留误判foreign，原failed回执保留。监控器经PID身份/复用/gone/不可读分类自检后重新准入，最终callbacks controller clean0、四rank通过、14监测无foreign/unknown；没有回写旧失败或忽略活体外来进程。

## 复现

把目标候选安装为editable，从其源码根执行，先核对 hyper_parallel.__file__ 与源SHA：

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q tests/ut/core/optimizer/test_swap_optimizer_base.py
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q tests/ut/distributed/context_parallel/test_mla_context_parallel.py

NPU需先配置匹配的CANN/PTA，独占物理卡并使用容器逻辑编号。将scripts/worker.py置于实际绝对路径，从候选源码根执行：

    python -B -c "import runpy; runpy.run_path('/absolute/worker.py', run_name='__main__')" --mode single --output /absolute/single-output
    python -B -m torch.distributed.run --master-addr=127.0.0.1 --nnodes=1 --nproc-per-node=4 --master-port=29579 --run-path /absolute/worker.py --mode callbacks --output /absolute/four-output
    python -B -c "import runpy; runpy.run_path('/absolute/worker.py', run_name='__main__')" --mode swap --expect-guard no --output /absolute/master-swap-output
    python -B -c "import runpy; runpy.run_path('/absolute/worker.py', run_name='__main__')" --mode swap --expect-guard yes --output /absolute/candidate-swap-output

single/callbacks使用CP候选；guard no使用原master，yes使用swap候选。新旧swap顺序运行、独立进程/输出。worker只用仓库tests/common的合成MLA，不加载私有模型。以上需要真实NPU，报告离线校验不能代替设备执行。

在报告目录执行 python scripts/verify_public_evidence.py，可复核导出文件SHA、无hook参数/槽一致性、四卡实际头分片/调用次数及梯度门限，不重新执行设备计算。manifest.json记录除自身外全部文件SHA。原始完整证据在本任务本地归档保留。
