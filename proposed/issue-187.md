<!-- correctness-review-20261018 -->
## 2026-10-18 正确性复核：K-shift：当前由调用者控制，PR #232 已处理环境接管

**当前状态**：最新 JT_dev `a1e6c222eab097c9609d6e744a3c5f98aeed6bdc` 已合入 PR #232（`d790be97`），`enable_full_determinism` 不再写 `CLOSE_MATMUL_K_SHIFT`。旧正文的无条件设置只描述旧基线，不能作为当前默认结论。

**PR 判断**：PR #192 已为 Draft，默认 true 的方案若原样合入当前 JT，会重新覆盖调用者环境，与 #232 已采用的环境所有权策略相冲突。本次保留历史提案但标明已被另一方案取代，不纳入当前必要修复组合。

**正确性依据**：确定性设置不应暗改独立 matmul 算术策略；调用者应可控制该变量。这一结论来自配置/环境接口契约，不依赖 MF 采用哪种 tiling。K-shift 开/关本身均不能仅凭与 MF 的距离判为数学上对或错。

**收窄历史断言**：下方“唯一来源”“换 CANN/宽度均不变”“代价为 0”只能指当时记录的具体 shape、权重、版本和有限复跑；没有跨所有实现的证明。清除变量后某个样本逐位一致，不等于普遍无性能或稳定性代价。旧建议的 10–30 步不能替代用户要求的完整 200 步。

**验收**：当前代码调用确定性设置前后，变量 unset/0/1 均应保持调用者值；不借此承诺任意 shape/kernel 逐位复现。本轮未重新运行旧 129 实验，也未接触 zth 目录；原始现场记录保留作历史定位。


本次固定参考：MF GitCode master `409019aac0c6579bb4fb8a9b04a75c259d98a96e`（只核查 `mindformers/pynative/`，未用旧 training_graph 冒充），Megatron main `1effc75a46b26c6b93478355ff9fc3bb8ac1fafc`，当前 JT_dev `a1e6c222eab097c9609d6e744a3c5f98aeed6bdc`。源码路径/行号/SHA-256 见[完整正确性复核](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/AUDIT.md)与[来源清单](https://github.com/xingleixu123/hyper-parallel/blob/xxl-dev/jt-correctness-review-20261018/source-catalog.json)。本轮不新增训练结果，不以 MF loss 距离代替独立正确性判据。


<details><summary>原问题与历史实验记录（适用旧基线；当前判断以上方复核为准）</summary>

相关：#185（第 301 步逐位排查全过程，CANN 版本已被该过程排除）、#157（Muon 线总 issue）、#179（AdamW 线 29 项消融）。

### 本轮关联与适用边界

- 本轮总问题：[JT 单卡 MF200→MF/HP400 完整排查 #217](https://github.com/hf1s/hyper-parallel/issues/217)。
- 本问题独立 PR：[支持配置 NPU 矩阵乘 K-shift 选项 #192](https://github.com/hf1s/hyper-parallel/pull/192)。默认行为不变，显式配置后选择对应算术路径。
- 下文保留原 300→600 步实验背景和首步证据。它证明该实验中指定投影差异的来源，不能扩展为所有 CANN 版本、形状、并行拓扑均无影响，也不能由有限次数复跑宣称普遍零代价。
- 本轮使用 125 上的单卡、4K Arrow、MF200 完整权重和优化器状态；HP CANN9.2、MF CANN8.5。PR192 已在真实 Trainer 的 true/false 首步对照中验证；CPU 相关回归为24项通过，详细结果见 PR。当前整体200步验收仍未通过，沿用总问题规定的完整窗口标准，不采用下文历史建议替代验收。
- 每项问题保持独立追踪，本单和 PR192 不承接其他算子兼容差异，也不自动关闭总问题。

### 检查清单

- [x] 1. 已检索现有问题。
- [x] 2. 已阅读相关文档。
- [x] 3. 本单给出可直接复现的证据，不是 traceback 类 bug。

### 🐛 结论

1. `hyper_parallel/trainer/runtime/random.py:60`（`enable_full_determinism()` 的 `IS_NPU_AVAILABLE` 分支）无条件设置 `os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"`，关闭 CANN matmul 的 K-shift tiling，改变 matmul 的 K 轴累加顺序。MF 不设该变量，走 CANN 默认 tiling（K-shift 开）。
2. 这就是 #185 定位到的、layer0 `kv_a` 投影 311/2,097,152 个元素差异（303×1 ULP、5×4 ULP、3×12 ULP）的**唯一来源**。不是 CANN 版本问题（#185 已用 CANN 8.5 实验排除），也不是调用形态/宽度/布局问题（#185 的 9 种写法、本单新增的 3D/bmm/预转置等变体都测过，全部同一结果）。
3. 该开关在 CANN 内核层面把 tiling 方式钉死，所以 Python 层无论怎么换调用方式、换 CANN 版本、换矩阵宽度，结果都不变——这解释了 #185 里"9 种写法全同 sha""换 CANN 8.5 一位不变""512 宽 latent-only 仍不匹配"这三个此前无法解释的现象。
4. 清除该变量后，第 301 步前向 **anchors a–f 6/6 逐位等于 MF**。

### 决定性证据（同一训练进程 / 独立进程，同一 `linear_qkv` 融合 576 宽投影）

| 条件 | 576 宽融合投影输出 sha256 | 与 MF |
|---|---|---|
| `CLOSE_MATMUL_K_SHIFT=1`（当前默认行为） | `add35e55682ab4ce…` | diff 311 元素，max 12 ULP |
| 清除该变量 | `9be24f4903712f9e…` | **diff 0，逐位相同** |
| 清除 + 真实训练进程内模型自身输出（连跑两次） | 同上 | diff 0（两次均成立） |

清除后 anchors：

| 锚点 | HP（清除开关） | MF | 一致 |
|---|---|---|---|
| a embedding 输出 | `2239faa0` | `2239faa0` | ✅ |
| b layer0 input_layernorm | `14105f3d` | `14105f3d` | ✅ |
| c layer0 attention 输出 | `869bccde` | `869bccde` | ✅ |
| d layer0 MLP 输出 | `f04cdd73` | `f04cdd73` | ✅ |
| e final norm 输出 | `8f6a8a0c` | `8f6a8a0c` | ✅ |
| f lm_head logits | `5325f893` | `5325f893` | ✅ |

MF 侧交叉验证：`ops.MatMul(transpose_b=True)` 用 `[576,5120]` 权重重算 = MF 自己的 `p2`，MS 的 3D batched N=512 形式与 HP 内核输出逐字节相同——两框架实际共用同一套 CANN 内核，只有这一个开关把两边分开。

### 数值影响（step 301，Muon 线，B0 + #179 的 29 项 overlay）

| | 开关=1（修复前） | 清除开关（修复后） | MF |
|---|---|---|---|
| loss | 8.142436027526855（Δ +3.24e-4） | 8.142110824584961（Δ −9.5e-7，另见 loss 相加顺序那单的 0 ULP 结果） | 8.142111778259277 |
| grad_norm（fp64） | 10.289084888924563（Δ +4.66e-3） | 10.284422100404774（Δ −6.6e-7） | 10.284422760440519 |
| anchors c/d/e/f | 全部 ≠ MF | **全部 = MF** | — |

修复前的下游放大（作为诊断方法保留）：逐 token LM CE 只有 14/3872 逐位相同；layer1 路由 top-1 有 4/4096 个近乎相等的 top-2 互换；46 个梯度 sha 全不同，`mlp.router.weight` 相对差达 2.26e-2。

### 操作要点

1. `enable_full_determinism()` 在 **trainer 构建期间**写这个变量，会覆盖 shell 里提前 `export CLOSE_MATMUL_K_SHIFT=0`（无效）。必须在**trainer 构建之后、首次 matmul 之前**清除。
2. 该变量只在**每个 shape 首次 matmul 编译时**生效；进程中途改回 `"1"` 不会撤销已编译的 shape。
3. 全框架只有这一处代码设置该变量（`random.py:60`）。125 的 219 个对齐配置里 **168 个**显式 `enable_full_determinism: true`，均会命中这个开关。

### 代价评估

MF 侧已证明"K-shift 开（默认）"状态下同参数连跑 4 组逐位可复现。HP 侧清除开关后同参数连跑两次同样逐位可复现（metrics、46/46 梯度指纹、step301 参数哈希、anchors、grad_norm 全同）。**结论：清除该开关不影响 HP 自身的 run-to-run 可复现性，代价为 0。**

### 建议

- **对齐实验（短期）**：在实验入口里，trainer 构建后 `os.environ.pop("CLOSE_MATMUL_K_SHIFT", None)`，不改框架代码。
- **产品默认（长期，需 framework owner 确认）**：把 `random.py:60` 改成可配置项（例如 `training.close_matmul_k_shift: bool`，默认维持现状 `true`），这样"与 MF 逐位对齐"的跑法不必依赖入口 hack，同时不改变现有默认行为。
- **验收口径**：这个结果让 #185 末尾"same-topology 不过、cross-topology 过"的争议基本失去意义——根因清楚后单步已经逐位对齐，建议验收改用"单步 + 前 10–30 步 + 模块级逐位"，300 步整体曲线只在存在残留数值差时才有区分度。

### 环境

同 #185：Ascend 910B3 单卡，batch 1，seq 4096，`hp-910b-arrow` / `mf-910b-arrow` 容器，HP `torch_npu 2.12.0 + CANN 9.2.0-beta.2`，MF `MindSpore 2.7.2 + CANN 8.5.0`，同一 restore 点（step 300）、同一 batch、同一权重。

<details><summary>复现与产物（129 机器）</summary>

- HP 运行：worktree `/workspace/worktree/align-l2-f` @ `57c35de0`，入口加一行 `os.environ.pop("CLOSE_MATMUL_K_SHIFT", None)`（trainer 构建后，首次 matmul 前）。
- 产物：`/workspace/align-work/{HPAccumOrder,HPFormMatrix,MFLayoutProbe}`（HP 容器）、`/home/zth/tmp/align-cmp/`（对比脚本与结果）。
- 本机：`~/project/hp-mf-sync-20261014-kshift/k-shift-fix/`（`accorder_summary.json` 25 种写法 sweep、`accorder_report*.json`、`accorder_landing.md` 最小落地写法、`entry_kshift_pop.py` + `RESULTS.md`）。

</details>


</details>
