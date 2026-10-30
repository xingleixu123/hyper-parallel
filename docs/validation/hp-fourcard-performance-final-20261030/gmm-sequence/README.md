# [草稿] grouped_matmul 整数序列参数的语义兼容修复验证

这个问题在收尾检查SAC的List覆盖范围时独立发现：HP公开接口接受整数list/tuple作为group_list，但本环境下Python list被派发到PTA旧List overload，未保留HP所需的分组参数。它发生在无checkpoint的第一次前向，独立于SAC注册修复，也不影响已测真实模型的Tensor group_list基线。

## 区分实验与根因证据

- 固定PTA2.12.0.post3、CANN9.2.0、Ascend910B3，BF16 x=(128,64)、w1=(2,64,128)、w2=(2,128,64)，非均匀累计边界[32,128]，group_type=0/group_list_type=0，HP split_item=3。
- 原始List输入在eager首个前向报aclnnGroupedMatmul 161002：后端按groupType=-1/no split校验并拒绝splitItem3。完整[错误栈](original-eager-error.txt)保留。没有执行成功的List SAC结果。
- 输入、两权重和上游梯度的实际字节完全相同，仅将边界改成device int64 Tensor，两个eager前反向均完成；输出和三份梯度重复逐位一致。一次前置尝试曾因PTA设备枚举为0终止，未算作模型失败；旧结束容器清理后实测枚举为1，但不主张已证明驱动根因。
- wheel记录PTA git6b88d20…，其[op-plugin submodule](https://github.com/Ascend/pytorch/tree/6b88d20d143ef15243c44f3582f37c480e786f2e/third_party/op-plugin)为79a427…。该[精确源List分支](https://github.com/Ascend/op-plugin/blob/79a42714ebf7662b5a3f7361f2b86ebb3dc4a1e3/op_plugin/ops/opapi/GroupedMatmulKernelNpuOpApi.cpp#L359-L432)接收group_type/group_list_type，但调用旧aclnnGroupedMatmul时不传两者；Tensor V4/V5分支会传递。与实际错误栈行429一致。源码绑定仍不证明wheel编译产物没有额外补丁。

## 修复与验证

在HP公共函数边界把list/tuple转换成x.device上的int64 Tensor，再交给现有autograd桥；既有Tensor输入保持原对象。保持分组边界、分组轴和输出/梯度语义，不修改矩阵数学、规约顺序、模型YAML或训练日程。两个[生产/测试文件](grouped_matmul.py)与[test](test_grouped_matmul.py)在本目录保存，最终独立Draft issue/PR见总issue304置顶。

CPU复用现有indexed-matmul/Einsum前向和梯度oracle，新增list/tuple的空专家、零总token及Tensor identity覆盖：未改源码4个sequence子用例失败，补丁后4个测试方法全通过（含旧用例）。CPU不是NPU kernel的替代证据。

129物理卡4实际NPU八条件：patched list eager×2、tuple eager×2、list original SAC×2、list fixed SAC×2；mutation=False。所有完整输出、x/W1/W2梯度共32项，root从36份NPY无损还原原BF16字节，独立对照未改Tensor路径，全部逐位相同、finite、max_abs=0。每条件actual default forward2/backward4；SAC原/新注册的recompute dispatch为2/1。这是below-SAC dispatcher事件数，不是kernel计数或耗时结论；规范化后的调用走default，不能称原生List overload的SAC已验证。

完整版本、每条件误差、事件和每个小tensor文件的shape/dtype/SHA见[results.json](results.json)，实际执行[worker](worker.py)也保留。57份闭合文件和36份完整NPY、CPU红绿日志、实际argv/只读挂载/gate/fetch/清理收据保留在本机任务`gmm-sequence-fix-v1`；公开JSON是投影，未冒称原tensor已全部公开。实际源码是冻结1a基础加这一个显式GMM文件overlay；该GMM原始文件与当前官方master96143的字节相同。

复现：固定所列Python/PTA/CANN环境和源，先运行`python -B -m unittest discover -s tests/ut/components/functional -p test_grouped_matmul.py -v`；NPU用隔离新run目录、仅映射空闲物理卡4，按本任务已受审的host脚本启动worker，核验真实PID/namespace/cgroup及npu-smi映射后才释放模块gate。worker必须核patch9ac036…、原/新policy和实际输入SHA，不读训练数据或checkpoint。完整命令保留在本机，不能绕过设备映射门直接伪造release。

性能/显存边界：Tensor调用没有新增边界tensor分配或复制（identity用例已验），多一个Python类型判断。整数序列调用新增device int64边界tensor及主机到设备拷贝；本fixture每次GMM两个边界的逻辑payload为16bytes，allocator粒度/临时buffer另计，未声称16bytes就是实测峰值。高频调用可预先传device Tensor。原List路径无法完成，不能给出有效before/after速度比；未宣称端到端加速、完整训练HBM差或所有shape/编译模式通过。没有改既有性能数据、YAML或精度门限。
