# packed SwiGLU TP guard：修复后四 rank NPU 回归

源码 [8400a764](https://github.com/xingleixu123/hyper-parallel/commit/8400a7648c4d41e9429789130660ff7961590879)，[Draft PR hf1s#351](https://github.com/hf1s/hyper-parallel/pull/351)，[问题 #1429](https://github.com/mindspore-ai/hyper-parallel/issues/1429)。此提交堆叠在 #1420，源 tree ab31e7aaf621b815ec1fbd65bc1b4fe5c445d39c。2721 文件部署清单 SHA 为2de52c8aae061e3c79d080d7b51cbad1b158a0b8b4e8032266872a2a93926dae。

CANN9.2.0 B237，Python3.12.14，PyTorch2.12.0+cpu / torch_npu2.12.0.post3，Ascend910B3，四卡0–3，HCCL，DP2×TP2。独立 BF16 小模块 H16/I32/B2/S8，权重与输入均为合成数据，未使用真实训练数据、生产 YAML 或 checkpoint。

| case | 四 rank 结果 | fused 调用/每rank | 输出、输入梯度、两权重梯度 max_abs |
|---|---|---:|---|
| default packed 输出轴分片 | plan 阶段明确 ValueError，含 gate/up 配对说明 | 0 | 未执行，不填0 |
| 完整 Replicate 参数与输入/输出契约 | 前向、反向完成 | 1 | 每rank四项均0，rtol=.008、atol=0 |

两个 case 控制器都 rc0 且 holder_removed；default rc0表示预期异常被测试记录，不是 forward 成功。结果明细和完整 traceback 在 reports/case/rank-N/micro-result.json。summary.json 记录实际输入/共享权重 hash，与[原缺陷测试](../swiglu-tp-evidence-v1/README.md)逐rank相同。原始default已进入fused后在FC2报K mismatch；修复没有实现 section-aware packed TP，仅使不支持布局提前失败、合法完整复制保持可用。不能将小模块结果外推为完整训练或性能验收。

repro/executed-entry.py 与 executed-bootstrap.py 是实际执行文件的原样拷贝，entry SHA926fab5aead74fc767a249e03bb31f9aaee3168e3c043a8a4051046d214ebccf；其 bootstrap 使用本任务隔离运行环境与握手。repro/repro.py 将握手简化为常规 torchrun，并对齐新的 functional.swiglu 计数位置；计算测试相同，但这个简化入口未另行 NPU 重跑，不能冒称其命令本身已验证。

在已安装上述兼容栈并加载相应 CANN 环境、当前目录可import修复后hyper_parallel时，简化复现入口为：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 repro/repro.py --case default --output guard-results
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 repro/repro.py --case replicate --output guard-results
```

输出目录必须是各case尚不存在的新目录。这里只给独立模块用法，不重新运行或公开完整训练配置。SHA256SUMS覆盖公开附件；完整日志与源部署收据另保存在任务归档。
