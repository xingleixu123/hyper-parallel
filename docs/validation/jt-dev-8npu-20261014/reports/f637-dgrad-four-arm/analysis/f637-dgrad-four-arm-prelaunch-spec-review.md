# FSDP/CP 四臂三步诊断启动前 Spec 复核

**PASS：本轮未发现新的启动阻断项，四臂配置、运行入口与守卫增量支持匹配对照。** 本结论只覆盖下列固定字节的启动准备；不表示四臂已经运行，也不表示 NPU forward/wgrad 等价、精度根因成立或200步验收通过。

固定生产源：`fixes/latest-jt-validation-f637` HEAD `18602c4524dc664b81b2bad9a7b624455ce1494f`，审核时本地工作树干净。源补丁SHA为 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。镜像固定为 `sha256:5f9e677f439d11ab4e7c888f54b142335a27eee7411cbd2ce894a4798a70146d`。规格为父任务的四臂要求及 `precision-localization-plan-v7.md`；沿用已启用 code-review 的 Spec 轴。

**对照配置与算术边界。**

| 臂 | case | 配置来源 | 模式 | 端口 |
|---|---|---|---|---:|
| A | r129-f637-dgrad-fsdp8-native-3 | FSDP8 off baseline r4 | native | 30361 |
| B | r129-f637-dgrad-cp2-native-3 | CP2完整200步 | native | 30362 |
| C | r129-f637-dgrad-fsdp8-fp32-3 | 与A相同 | fp32 dgrad | 30363 |
| D | r129-f637-dgrad-cp2-fp32-3 | 与B相同 | fp32 dgrad | 30364 |

四臂均为8卡、从初始化执行3步、无恢复/不保存checkpoint，TP/EP/PP=1且SP=false。每个native/fp32配置对除checkpoint输出目录外逐项相同；相对各自200步参考只允许train_iters=3、save_ckpt=false与checkpoint_dir变化。CP仍为expanded_ulysses/overlap=false，chunk4096、GBS8/MBS1、种子、数据、Muon FP32 NS、专家FP32 wgrad、BF16参数/FP32归约、LR与clip均保持参考配方。

已独立解析原始四份配置：CP为完整JSON形式的YAML，FSDP为省略默认项的YAML。使用本机PyYAML 6.0.3逐字段核对全部显式值及类型；未把原始YAML直接等同于填充默认值后的resolved config。默认项的完整一致性依据固定镜像内实际 `parse_training_args` 的4/4预检：`preflight_fair_dgrad.py:45–68` 构造参考仅改三项，再严格比较整个to_dict。预检结果、日志中的完整JSON、配置/参考SHA与receipt均吻合，devices=[]、npu_initialized=false、exit=0、removed=true。预检只解析配置并构造Config，未构建真实模型或执行分布式训练。

`run_fair_dgrad_case.sh:8–14` 只从验证过的case名提取native/fp32模式，并清除旧NS/wgrad/recompute/profile干预环境。`capture_fair_dgrad_training.py:29–50` 在正常Trainer构建后安装同一观察器，限制TP1/CP1或2/3步；`linear_dgrad_trace_experiment.py:140–149` 在native分支直接调用原始F.linear，只注册返回None的观察hook，只有eligible且changed时才使用已审Function。没有顺便修改CP分母、LR、参数/梯度缩放或后端精度开关；两臂遇HF32已开启都明确拒绝。旧数学实现SHA保持不变。原生算术透明性最终仍须由A/B分别精确匹配原参考前三步来确认，不能用静态检查代替实际大矩阵NPU结果。

当前新观察层记录FQN、call_id、step、首步批次身份、实际dtype和干预反向是否执行；避免旧形状去重掩盖同形状层。首步边界的prepare_step/lazy map与非首步loss记录跳过修复保持原字节。DTensor/子类与非BF16继续旁路，4D Linear可被干预但3D边界采样不涵盖其全部输入；这些是明确的诊断范围。22项真实CPU autograd/冻结组合/checkpoint/异常清理结果、source SHA、部署、日志和退出回执全部对上；它们没有覆盖此Trainer工厂的真实8卡运行。

**输入固定与运行隔离。**

四份request各固定16项：13个runner复制的Python helper、runner、guard、该臂配置。已从复制列表及所有Python AST imports检查本任务工具依赖闭包，没有遗漏运行时本地helper。每项SHA均与现文件及部署回读一致；共核对22个部署after/readback项，125/129两处guard变更前副本也与计划before SHA相同。未变化文件没有独立before备份不影响after固定；本复核没有据此编造历史副本。

`fair_resource_guard.py:149–159` 在设备观察和创建容器前核对repo commit、binary diff与全部请求文件SHA；runner把helper/config复制到全新独立run目录，再保存source commit/status/patch及导入路径（runner20–32）。各臂端口唯一。840秒内层timeout加15秒kill-after，外层guard仍为请求900秒上限；torchrun用8个进程，所有rank日志进入该臂独立worker-logs，launcher日志与实际退出码也分开保存（runner34–41）。runner没有设置或扩大ASCEND_RT_VISIBLE_DEVICES，使用guard分配的设备集合。缺文件、导入错误或初始化失败不能记为完成；后续必须同时核对每rank start0/complete3、EXIT_CODE及released receipt。

**守卫边界未削弱。** 固定旧guard SHA `83d461a2aba65cc0ad08b48bee8141ac45aaa4cd48eaea8a21eb129897cb9ae4`；新SHA `90fd776c32aff270582ac087ad51deb25d2d6a4521c2e8a615ae6feb2f29f81e`。精确diff仅新增runner→repo映射（40）及dgrad请求必须8卡、明确layout/mode/三步case的校验（140–143）。独立AST比对显示：删除该新增分支后remote_job与旧版相同；其余全部9个函数AST相同。

因此原有两次空闲检查（163–174）、1200秒全局wall/monotonic保守冷却及全局锁（46–53、291–338）、900秒限制、容器owner+精确ID双重验证（199–207）、冲突时仅停止本任务容器、最终释放/清理边界（220–287）保持不变。既有训练/PR199/PR214入口未改动。已有PID观测时序或检查后输入被修改的通用风险没有在本次扩展中新增；本报告不把模拟测试包装为实际设备仲裁证明。

本审计在内存执行原remote_job AST，合法请求在首次文件系统观察前用受控异常停止，未调用subprocess/SSH/容器/设备，也未写调度状态。16个入口边界通过：四个现有arm允许；4卡、200步名、未知mode、TP布局名、路径case、901秒、未验证image、host126均拒绝；r2重试及既有training/PR199/PR214合法请求仍通过前置验证。另外3个冷却检查确认未满1200秒、到期、墙钟回退。脚本和这些输出仅在本会话内存执行，未另建测试文件；此处保留方法及被测SHA。

另核验父任务持久证据 `f637-dgrad-guard-boundary-tests.json/.log`：13例成功，guard/test source/log SHA与当前 `check_pr214_guard.py` 一致；其中包括源码或runner变化先拒绝、仅7空卡不启动、二次空闲检查见外部任务不创建容器。该13例是隔离外部命令的CPU边界测试，不是NPU实验。

**后续结果的验收边界。** A/B必须逐rank精确匹配对应参考1–3步的输入、初态、计数、LR、LM/加权及还原MTP/aux/total/grad_norm；不匹配应先处理观察或执行差异。四臂必须独立完整，采样/导数/实际Linear调用不能空；C−A、D−B解释干预，D−C解释新导数方案跨布局差。CP before_cp_sum与FSDP after_model_forward按同样本匹配；不能按相同rank匹配不同样本，也不能固定乘2替代真实监督权重。首步采样相等不是全张量相等。不得把3步或此审核PASS改称200步通过；现有严格200步比较器保持不变。

诊断的FP32大矩阵临时量及BF16 wgrad重放仍有额外开销；每rank事件有20,000上限且不保留Tensor，但不据此推断显存一定足够。运行后检查真实覆盖、有限值与显存；只在证据验收后清理本任务临时产物。该流程使用独立进程，避免跨臂复用全局patch/hooks。

本轮只读源/请求/配置/已有证据，执行了本地无设备AST与解析检查；没有启动容器/NPU、连接设备或修改源码/原始证据/GitHub。唯一新文件为本报告。以下SHA为审核时字节指纹。

| 文件（相对任务根） | SHA256 |
|---|---|
| `configs/r129-f637-dgrad-cp2-fp32-3.yaml` | `bd6407aebac6a494d1709c81df34cabd43256dbb6c75ca62e1c9a55fa11d6821` |
| `configs/r129-f637-dgrad-cp2-native-3.yaml` | `4c91ed587dc08b62354a25c92048d12412dc064f0cebb885eb8afe1da42be64d` |
| `configs/r129-f637-dgrad-fsdp8-fp32-3.yaml` | `1f522a34e95aef6a97fe9e508ded292f16c0877d1c175092d177e5ce621347c1` |
| `configs/r129-f637-dgrad-fsdp8-native-3.yaml` | `e42f9569758436a901d64fd879e8ae76ee714e66b163b5bc33a1dc571ff20a4c` |
| `evidence/dgrad-prelaunch-source-before/125/tools/fair_resource_guard.py` | `83d461a2aba65cc0ad08b48bee8141ac45aaa4cd48eaea8a21eb129897cb9ae4` |
| `evidence/dgrad-prelaunch-source-before/129/tools/fair_resource_guard.py` | `83d461a2aba65cc0ad08b48bee8141ac45aaa4cd48eaea8a21eb129897cb9ae4` |
| `evidence/f637-dgrad-deployment-plan.json` | `d660ea928b95d5978e69c6893a472ee135e0d6c333c7abb80b504ead38cccce0` |
| `evidence/f637-dgrad-deployment-readback.json` | `b0678914fe9ee5b76fadf28abdccb6bfa9e416f561c300bcccb992dbf2881410` |
| `evidence/f637-dgrad-four-arm-preflight.container.log` | `8bbfc370112b2af35f81a29b299c3b8dd7c305063821d0e3b9b1e0cafeef46c0` |
| `evidence/f637-dgrad-four-arm-preflight.json` | `02a9656056037e391da4cc183fea6045d1944cf196d30b9cd08e01fc7659ff82` |
| `evidence/f637-dgrad-four-arm-preflight.receipt.json` | `b6a8d6251988781c9a1ebdaa30b719fdc7d4adfcfc23fbeb6289b6f7dc789654` |
| `evidence/f637-dgrad-guard-boundary-tests.json` | `f8eba9b073a09fe0d13b17722a93457be7f44a9a9bbada7144d81cd3a9a78d1b` |
| `evidence/f637-dgrad-guard-boundary-tests.log` | `5673ddaed2749538934d845a296ebc8e0c6d30ec8fe703901eaa8a147d4d8807` |
| `evidence/f637-linear-dgrad-trace-cpu-v2-deployment.json` | `ddfba2ea26e2e747ab5169c59786a21f4d160ca4062a20ace54e3c6e0dd4d9b7` |
| `evidence/f637-linear-dgrad-trace-cpu-v2.container.log` | `84ef807447322f8cc0e48e8857e4190ffbb855ab831f69e826f02edaad505a4a` |
| `evidence/f637-linear-dgrad-trace-cpu-v2.json` | `b15a2fe57500c5f5857f3bf71540808968a2f1b1242270839cbf86352fcdcce2` |
| `evidence/f637-linear-dgrad-trace-cpu-v2.receipt.json` | `b2c326c79d8925dd6312844f4e74d07972b486f36c5f94001443f4088270b95d` |
| `evidence/f637-linear-dgrad-trace-v2-standards-review.md` | `3300f18ef6659ee03819c76422a44c50ea46adfb271ca849bae89a271ef91b80` |
| `evidence/precision-localization-plan-v7.md` | `f6a01255cdb575266a87c69ca2756e8293622b443768f053ce1ee41794ce2948` |
| `runs/r129-f637-fp32-cp2-200/config.resolved.json` | `27048090089da4b657bd5873fd7d20bdb8a4cba643599eca420de5e0a4271bee` |
| `runs/r129-f637-fp32-fsdp8-0-200-r4/config.resolved.json` | `dfb276d6b985e0d6197888d4480c340611bc2ab42a3d58c89121a1319457f782` |
| `scheduling/r129-f637-dgrad-cp2-fp32-3.request.json` | `173de3f73eb6f1512ff5f59cb0c537f80ed5cc16721010e7e2cdc25e35d9d0db` |
| `scheduling/r129-f637-dgrad-cp2-native-3.request.json` | `5831727e17d93a53e322cabeb61f78654949904d440a2df9a5ac2eb792117fdf` |
| `scheduling/r129-f637-dgrad-fsdp8-fp32-3.request.json` | `a4baf9eba20a330df6b0132d95015448111c7dcb81515fe0669d50279626021a` |
| `scheduling/r129-f637-dgrad-fsdp8-native-3.request.json` | `bcc6defc97bcea6b7e0ab6b47f9fb934226482f9cce62680ff851579d54722ce` |
| `tools/capture_cp_loss_training.py` | `47bfe15b93fba3dc1fba732eac143748a800ebc0b3e31aee1cf8ece169d7381e` |
| `tools/capture_fair_dgrad_training.py` | `0ee23ecae60977dda9dd7efee91c34de1e809acb428e10da82850dc8b306f97b` |
| `tools/capture_precision_training.py` | `dad38bef57ef4b1f9b3ed81eafffff2735ad229b1b29dbed4a13fc7a339f72f6` |
| `tools/capture_training.py` | `aef5a45ef770da810f07a59a3d97e3363368e7bb7076525d8e95f05ac9c6d717` |
| `tools/check_pr214_guard.py` | `bf13ad48787b7ddbc5c7a8804edd6e83c1a43fb48e59aa2df5fae04b55e0b081` |
| `tools/cp_backward_trace.py` | `b682fcbfe3067988245d16dcecaf1a32f6232d574ccee923a250d95a7471da32` |
| `tools/expert_probe.py` | `461fb5c6aeca377d1157f5f6a6a021650629d9d9decccef80d4442d030a6257f` |
| `tools/fair_resource_guard.py` | `90fd776c32aff270582ac087ad51deb25d2d6a4521c2e8a615ae6feb2f29f81e` |
| `tools/fp32_wgrad_experiment.py` | `c9e3448d9f818f4c0ef3257f31e8bc4f24c7b9300109e85f9db086fd1bb7f670` |
| `tools/linear_dgrad_experiment.py` | `373b5ed016604a92dbfec9afeb2c76edc8b456cf427ade592dfab2aa8098f925` |
| `tools/linear_dgrad_trace_experiment.py` | `62a18e143b1a8af826cc8ebdaf201e7be442c8179362caa102e02a73bea8ab87` |
| `tools/loss_gradient_trace.py` | `9b57516f6a5800197ca0fddf19a0878e509fb84da556138a0665e66683ef1f60` |
| `tools/module_profile.py` | `b01dcf73a23ec967b04b5b5358f2f8cdae028227cb87e28ac465c0bb7e722c9c` |
| `tools/ns_batch_trace.py` | `bcb698b5b26bbb2ef3df29c23b75c1a0b7d2995d47a25174972035be9a73de3e` |
| `tools/precision_experiments.py` | `50acc479aaa8ca4603243cfb9ce3dd0b8907e35a9fe430b816671e6d7bdf9659` |
| `tools/preflight_fair_dgrad.py` | `c585865228a130638d7fc79d9979b4638e9c19274b9a912f27fa1be40b417838` |
| `tools/run_fair_dgrad_case.sh` | `c2d5a521a91190eaaa7bbfbce10affd5687fde5f8e821e3c3c4735d7bdc99c7c` |
| `tools/run_fair_dgrad_preflight.sh` | `b4da5f8c272927aff4242e91faaf6a2188e2a3bbe936398ddf40756a7706a8fc` |

审核时间（UTC）：2026-10-15T20:45:03.780101+00:00
