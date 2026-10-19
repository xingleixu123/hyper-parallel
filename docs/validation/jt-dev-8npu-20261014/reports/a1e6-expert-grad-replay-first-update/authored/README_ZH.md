# 专家梯度与裁剪范数的首更新五臂消融

在这次固定初态和数据下，**B臂同时使用F的指定专家梯度与裁剪范数后，首更新的32个专家目标参数完整原生FP32哈希全部匹配F；S/N/G均为0/32**。B的第2、3步主LM仍有`3.4332275390625e-5`、`3.814697265625e-5`残差。因此这是指定首更新的局部联合干预证据，不是200步修复，也不能推广为Muon全正确、唯一产品数学错误或唯一内核归因。

## 五臂及干预位置

五臂均在129、clean源`544526f7db8342c607244fd41009b2a74adb84c8`、同一固定image5f9e上各执行一次3步×8rank。干预仅在step1的原backward/RS及缩放完成后、原clip计算前发生；自有包装在第2步前卸载，随后两步按原生配方继续。GBS、seed、数据、LR、MTP/aux系数、FP32 expert wgrad及Muon NS FP32保持原配方。

| 臂 | 拓扑及作用 | 首步专家梯度 | clip实际used norm |
|---|---|---|---|
| F | FSDP8 capture参照 | 原生F | 原生F |
| S | EP4自重放透明控制 | 自身原生值原位复制回自身 | 自身原生 |
| N | EP4仅范数 | 保留EP原生值 | 替为F的原4字节范数 |
| G | EP4仅专家梯度 | 重放F的32目标 | 原函数对当前混合梯度实测值 |
| B | EP4两者同时 | 重放F的32目标 | 替为F的原4字节范数 |

F/S分别与各自原200步控制的前3步六标量、计数、输入、初态及runtime严格零差，构成新捕获/自重放链的透明性控制。干预臂不能套用“对自身原轨迹透明”的要求。32目标按**全局专家键**匹配，不按相同物理rank配对；它们是decoder第1层和MTP第0层的`gate_up_proj/down_proj`四个routed-expert族×8global experts，不包括shared expert、router、attention及其余参数。

## 三步主LM响应

![相对F的五臂三步主LM差](figures/main-lm-three-step.png)

图中是每步8rank最大绝对差，使用线性轴并保留真实0；没有画200步阈值或给出三步“达标”判定。8rank为归约后的全局标量日志，**不是8次独立重复**。原数据可从[losses.csv](data/losses.csv)查看；[SVG](figures/main-lm-three-step.svg)包含字形轮廓。

| 臂 | step1主LM绝对差 | step2主LM绝对差 | step3主LM绝对差 |
|---|---:|---:|---:|
| F（参照自身） | 0 | 0 | 0 |
| S | 0 | 0.0001430511474609375 | 0.0000896453857421875 |
| N | 0 | 0.00003147125244140625 | 0.000274658203125 |
| G | 0 | 0.00014495849609375 | 0.0004444122314453125 |
| B | 0 | 0.000034332275390625 | 0.00003814697265625 |

首步前向四项loss均保持原值，因为干预发生在其后的梯度路径。不同单干预的后续响应并非统一减小；不报告贡献百分比。CSV同时保留total、grad norm、LR及MTP/aux的**日志带系数值**（MTP乘0.3、aux乘0.0001），没有把它们升级为独立专项矩阵或原始未加权张量验收。

## 整专家哈希与范数原字节

![parameter_after哈希匹配数与measured/used norm](figures/parameter-hashes-and-norm.png)

B的after_clip和parameter_after均32/32匹配F。这里比较的是每个选定整专家Tensor的原生FP32 SHA256；**哈希不等不能换算成不等元素数、max/RMS参数误差**。完整参数after的160条记录见[parameter-hashes.csv](data/parameter-hashes.csv)，[第二图SVG](figures/parameter-hashes-and-norm.svg)可直接放大。

范数在每臂8rank上各自一致。下表hex是**小端原4字节**，相应位模式分别为`0x41b3c99e`和`0x41b3c99f`；两值差1ULP，即`1.9073486328125e-6`。

| 臂 | measured FP32 | measured hex | used FP32 | used hex |
|---|---:|---|---:|---|
| F | 22.473445892333984 | `9ec9b341` | 22.473445892333984 | `9ec9b341` |
| S | 22.473447799682617 | `9fc9b341` | 22.473447799682617 | `9fc9b341` |
| N | 22.473447799682617 | `9fc9b341` | 22.473445892333984 | `9ec9b341` |
| G | 22.473447799682617 | `9fc9b341` | 22.473447799682617 | `9fc9b341` |
| B | 22.473447799682617 | `9fc9b341` | 22.473445892333984 | `9ec9b341` |

原norm/clip调用各一次，原`max_norm=1`、`epsilon=1e-6`及clamp保留。N/B只替换clip所用分母，返回和日志仍保留真实measured norm，不能把日志伪造为F；因此这两臂基类据返回值推算的postclip norm不再代表替换分母下的实际值。G使用替换梯度后原函数的实际测量，本次仍落在EP的同一个FP32值，不预设其必须等于任一控制。详细标志见[norms.csv](data/norms.csv)。1ULP归约舍入差本身不被定性为独立产品bug。

## 来源、独审与可移植范围

原实际比较SHA为`300152f2995e1af14327470706be70a774908e8e23c5dbf344f2edfd383e0310`；外部五臂绑定SHA为`e66b4f58ffff9a96d2d120a04ad7ec931ea7a1fb875be23fec568550e860633f`，root实际reader执行SHA为`ffbd981444182e101d4846c788d872cd14062fbf3311d96efe238a4d995b55d6`。构建器仅读取有SHA约束的小型JSON/JSONL；40份metrics重算LM，40份native journals解码范数4B，32目标哈希匹配数从冻结实际结果重算。[机器摘要](data/summary.json)与[来源链](evidence/source-provenance.json)保存精确来源指纹；个人路径、完整容器ID与原生命周期文本不随报告复制。

A2实际科学独审SHA为`4aa049a4cdecaab3d6a96e813a90747177a826cf0bd076196b7e28a3e1dbddd3`，核验596份原件、所有目标记录，并独立按FP32重解码32份donor的有限性；其[安全独审绑定](evidence/actual-scientific-independent-review-binding-v1.json)已与本包数值互核。该独审针对原五臂科学证据，**不冒充本次新图文报告的独审**；报告独审/发布状态见[review-status](evidence/review-status.json)。

**本包不是完整donor复现包。** 32个donor共`1,509,949,440 B = 1.40625 GiB`；donor、checkpoint、权重、训练文本均未复制。完整raw与donor留在原任务证据位置。原参数Tensor的完整数值未保留在此包中，不能从SHA反推出数值误差；本报告工具也不会打开或复制donor。它支持小型数据/图的复建，不提供重新执行原NPU干预的完整输入。

只用本包的小型CSV与summary即可重绘，需Python、Matplotlib及可用中文字体：

```powershell
python -B -X utf8 tools/plot_replay.py --output-dir <全新绘图目录>
```

若拥有本地保留的小型原始证据，可用[build_data.py](tools/build_data.py)重新提取全部表数据，再用[plot_replay.py](tools/plot_replay.py)重绘：

```powershell
python -B -X utf8 tools/build_data.py --task-root <任务根> --output-dir <全新数据目录>
python -B -X utf8 tools/plot_replay.py --data-dir <新目录>/data --output-dir <新目录>/figures
```

工具拒绝覆盖已有输出；不调用SSH、容器、NPU、训练、Git或API，也不生成资格/gate。作者已在新限定目录实际复建并重绘，九个数据/图输出逐字节一致，两张PNG像素一致且已实际查看；只清理本次九份经SHA核实的重复产物。[作者QA](evidence/author-validation-v1.json)记录范围。选定32专家首更新相同并没有消除B后续主LM残差，剩余来源仍未定位；本报告不推断性能收益、不构造产品PR，不扩大为全模型、全部梯度、全部optimizer state或200步验收。
