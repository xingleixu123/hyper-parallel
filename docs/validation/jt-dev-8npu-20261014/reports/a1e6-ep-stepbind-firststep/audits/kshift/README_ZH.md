# CLOSE_MATMUL_K_SHIFT：官方出处与解释边界

可作为单变量诊断候选；尚无证据把它称为本镜像 `npu_grouped_matmul_add_` 已生效的控制。

有效官方依据是 [MindStudio 8.0.RC1《大模型训练精度问题定位案例》](https://www.hiascend.com/doc_center/source/zh/mindstudio/80RC1/practicalcases/LargeModelTrainingAccuracy/MindStudio%208.0.RC1%20%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E7%B2%BE%E5%BA%A6%E9%97%AE%E9%A2%98%E5%AE%9A%E4%BD%8D%E6%A1%88%E4%BE%8B%2001.pdf)，文档01，实际发布日期2025-06-13，PDF第10页／印刷第6页§2.5：在疑难Matmul排查中，可尝试设置 `CLOSE_MATMUL_K_SHIFT=1` 关闭错峰策略。它说明排查动机，没有枚举 CANN9.2.0-beta.2、910B3、GroupedMatmulAdd 或某个 tiling 的支持范围。本次取得原PDF正文，另以1MiB流式读回其SHA和字节数，[回执](official-document-readback-v1.json)不把搜索摘要或导航页冒充有效9.2算子规格。未取得该规格不等于该开关不支持。

HP544中，[普通 grouped_matmul.backward L98–115](sources/hyper_parallel/components/functional/grouped_matmul.py) 在gemm_fusion且main_grad存在时调用GMMAdd；[fused swiglu L115–130](sources/hyper_parallel/components/functional/npu_grouped_swiglu.py) 也有FP32 main_grad累加入口。这里的FP32 buffer合同不自动确定后端K轴分块、累加顺序或开关消费位置。[random.py L47–65](sources/hyper_parallel/trainer/runtime/random.py) 保留调用者K_SHIFT值，既不赋0也不清空；源码作者的硬件说明不是本审新增实验。完整原文SHA在[source binding](source-binding-v1.json)。

建议两侧新进程只增加此一个env值，记录trainer前后实际值，保持HCCL、确定性开关、数据／初态、FP32 expert wgrad、Muon NS和全部recipe不变；继续观测同125完整E0 G0及后续边界。开关可能影响前向、dgrad或其他matmul；即使E0 G0变成exact，也只能证明整个开关干预足够改变该边界，不能把效果唯一定位到GMMAdd。若不变，也不能仅凭env存在声称GMMAdd已消费该开关并已被排除。

因此无需在首次短pair中猜测内部tiling。若要把结论推进到具体算子，后续才需实际调用／kernel证据，或固定同一真实输入、同shape、同原生算子路径的独立局部probe；当前没有此类新执行。正常125无observer完整200基线及最终完整验收仍各自必要。本目录仅新文档与固定源快照，没有修改冻结范数包、产品或运行控制面。
