# fdce1a9a 全部原始检视处置

29 个原始 thread 逐一对应；代码覆盖不等同线程已 resolve。未完成的全窗训练不作为已通过结果。

| 评论 | 要求 | 结论 | 当前代码与理由 |
|---|---|---|---|
| 4462484211 | 为什么需要这个文件，没有已有的实现吗？或者已有的实现能少量修改吗 | 已覆盖 | adapter/conversion/module_replacement.py:130,232,292：独立 grouped_experts.py 已移除；复用公共 GroupedExperts、通用权重转换；保留原专家的校验与权重可逆映射。 |
| 4462497711 | 你重计算为什么需要一个文件。你看下库上dsv41咋做的相关的，是不是少量修改，尽可能复用库上能力 | 已覆盖 | adapter/registration.py:27：RecomputePolicy 指向 model.layers.* 和 mtp.layers.*，不再有独立策略文件。 |
| 4462525951 | 类似这样的 为什么没有导入trans的xing，然后一些embedding类似继承的是xing4XXX | 已覆盖 | adapter/training.py:39,215,271,319：适配类继承捆绑 HF Xing4 类并重用其 backbone；用户要求 HF 三文件不改。 |
| 4462531999 | 类似不应该有reference的前缀？ 然后你看下是不是有比较多的能继承 | 已覆盖 | adapter/training.py:215,271,319：训练类无 reference 前缀并继承官方实现。 |
| 4466118410 | 我们也有啊 为啥要写这么多呢 到底改什么了。为啥不能用我们的 或者提供的modeling 4那个文件，为啥要修改，修改也应该很少啊 | 已覆盖 | adapter/training.py:215,271; adapter/training.py:140,177：旧 adapter/moe/layers.py 已移除；保留训练 router/MTP 接线及共享 GroupedExperts；仍可整理文件边界。 |
| 4466123082 | 为啥还有这种写法， | 已覆盖 | adapter/training.py:248; adapter/conversion/module_replacement.py:317：保留 mHC post 转置和 pre 输出 residual 的梯度链；旧模型专用兼容路径已移除。 |
| 4466124923 | 没必要的删掉啊 | 已覆盖 | git tree e401: hyper_parallel/models/xing_4：旧 .gitattributes 不在当前 Git tree。 |
| 4466130247 | 没必要的删掉啊 | 已覆盖 | git diff a9fa47d6...e401 -- setup.py：setup.py 不在现 PR 改动；没有为模型单独改打包。 |
| 4466230866 | 这里不需要修改，后续非融合计划删除 | 已覆盖 | git diff a9fa47d6...e401 -- hyper_parallel/components/functional/mhc_post.py：公共 mhc_post.py 不在当前 PR 改动。 |
| 4466419147 | 这个别加进代码仓里，这是具体的模型规格，应该是类似ckpt那样放在外面供hyper-parallel去读取的 | 已覆盖 | git tree e401: hyper_parallel/models/xing_4：不纳入模型 config.json/tokenizer 资产；模型规格由外部配置提供。 |
| 4475698652 | 这个文件似乎可以和builder放在一起吧 ；都是把yaml里面的model.config变成Xing4_0Config然后给builder。感觉完全可以和builder文件merge起来，也不需要那么多校验 | 本次整改 | adapter/builder.py:32,106：配置创建已合并 builder，继续保留 alias 冲突、clamp off、动态必填和训练语义校验；旧配置文件删除。 |
| 4475844486 | 这个def没地方用到吧  | 有依据保留 | adapter/registration.py:63,94,97：_load_replacements 被 ModelAdapterSpec.replacements 引用并实际调用；真实 provider CPU 检验通过。 |
| 4475845264 | 感觉没必要在#1455为chained optimizer新增register_step_post_hook并在此调用。chained optimizer里内含muon和adamw两个优化器，均是torch.optim.Optimizer的子类，直接调用muon.register_step_post_hook就好 也更符合“qk clip用来解决muon优化过程中的logit爆炸问题”的语义 | 已覆盖 | adapter/optim/muon.py:69,74,81：已经直接对 Muon leaf 注册 native post hook；不依赖 #1455 chain hook。 |
| 4475905775 | 这个文件用不用参考modeling_deepseek_v41.py 放进training.py里面？ 也就这一个需要调用的地方。   | 本次整改 | adapter/training.py:44,140,177：Router、MLP、MoE 已并入 training.py，转换与 CP 导入指向新位置；官方 HF 文件未改。 |
| 4475925241 | 这种实现会不考虑其是否实际属于muon管理的参数 \|\| required_grad == True就直接进行缩放。或许可以结合上面说的注册到muon的post hook，然后调用时候会判断参数是否是muon管理的参数，是的话再进行缩放。不过在该模型场景下直接这样做应该没啥问题，看你要不要考虑泛化了 | 已覆盖 | adapter/optim/muon.py:71,75,79：阈值启用时两投影 requires_grad 且在 Muon param_groups，缺 attention 或不满足即拒绝；后注册的 swap 通用支持另属 #1459。 |
| 4476024276 | 是不是改个名字 | 有依据保留 | adapter/training.py:1,319; adapter/registration.py:88：保持 training.py 及注册架构名，符合用户要求并避免无意义重命名。 |
| 4476138012 | 这个需要单独抽取出来吗 | 有依据保留 | components/modules/mla_attention.py:241; adapter/conversion/module_replacement.py:122：抽取 KV 展开作为可覆写边界供 Xing RoPE/分头实现复用；这是公共小接口改动，验原 MLA/DS 路径与梯度。 |
| 4476142089 | 这个要在这里做还是trainer提供？ | 有依据保留 | adapter/distributed/context_parallel.py:116; adapter/training.py:294：shift 是模型各深度 MTP 目标边界；Trainer 既有输入切分不能代替跨 CP 邻片交换和末尾填充。可未来公共化，现保证深度/梯度回归。 |
| 4476144198 | loss的处理是trainer负责的吧 | 有依据保留 | adapter/distributed/context_parallel.py:140; adapter/training.py:387,407：每 MTP 深度 shifted targets 的分母不同；该函数先实现模型序列目标，Trainer 再做外层 micro/token 权重；不能机械替换通用单分母。 |
| 4476285672 | 这个融合算子需要吗，在这里构造一次，然后被module_replacement替换了 | 本次整改 | adapter/training.py:215,228; adapter/conversion/module_replacement.py:318：删中间 Xing4HyperConnection 包装，直接保留 HF HC 及 norm epsilon，由 plan 替换真实 fused；需要注意 seed 冷初始化随机消费改变，已独立留证。 |
| 4476404636 | 这个校验是不是有点奇怪。 | 有依据保留 | adapter/builder.py:115,118：删除重复模型类型检查，保留每个原始 HC 必须替换为 fused 的全覆盖保护；漏 plan 明确拒绝。 |
| 4476484034 | ref里面好像是用的clamp版的算子，这里拒绝了； 我们目前框架的mhc应该还不支持 | 有依据保留 | adapter/builder.py:65,71：训练配置继续明确禁用 clamp，非空拒绝；不能把公共 fused mHC 误称支持参考 clamp。 |
| 4476511406 | 这几行都可以靠plan_overrides替换，但是def _post不行 这个似乎是开源代码部分class Xing4_0HyperConnection 有forward可以直接替换，但是里面没有post得在这里写一份  如果能修改开源代码 似乎就方便很多 | 有依据保留 | adapter/training.py:236,248; adapter/conversion/module_replacement.py:318：官方 HF 文件不改；训练 decoder 将 pre 交 plan，保留 post 混合转置和 residual 梯度边界。 |
| 4476673821 | 稍微有点怪 但是不知道有什么好的写法 | 有依据保留 | adapter/training.py:257：guarded nn.init 保存已加载 tensor；真实 HF 部分加载覆盖 HC 及 router 状态，避免恢复直接 zero_/fill_。 |
| 4477449370 | 这里isinstance(module.experts, GroupedExperts)和写死use_grouped_gemm=True会导致moe部分强制走grouped_gemm分支。 如果期望仅保留该分支，则直接在modeling里面实现grouped-gemm版本，并删除这里的配置；如果希望保留for-loop专家计算版本做精度比对基线，则删除这里的校验，use_grouped_gemm改为可配置参数  | 有依据保留 | adapter/distributed/expert_parallel.py:31,40,46; adapter/training.py:192,202：非 EP 原始 ModuleList loop 保留；EP 保留已验 grouped executor，类型检查改为 expert-major 能力检查。公共 stacked loop 是不同布局/加载契约，未新增未验配置；#1407 仍为 Draft。 |
| 4479302597 | 这里是否有必要添加一个大函数做校验和weight打包？ | 有依据保留 | adapter/conversion/module_replacement.py:135,201,232,292：集中打包仍检查来源类型、hooks/buffer/别名、shape、策略与 checkpoint 映射；公共 GroupedExperts 执行，#1407 不提供等价转换。 |
| 4479308908 | 这里为什么不是直接从weight的shape里面读取hidden_size, intermediate_size | 已覆盖 | adapter/conversion/module_replacement.py:201,208,212：已从 first_gate.shape 得到维度，再校验 config 和其余专家权重。 |
| 4479501581 | 这个文件不应该改 | 本次整改 | core/multicore/modules/_precision.py:17：公共文件相对原 base 不再有差异；功能 helper 仍被复用。 |
| 4479507617 | 这个文件应该放在training.py | 本次整改 | adapter/training.py:44,140,177：训练 Router、MLP、MoE 合并进 training.py，保留 HF 官方三文件原字节。 |

补充 discussion：权重转换集中于 module_replacement；训练 MoE 合并 training；用户明确官方三个 HF 文件原字节不改，故不直接改官方 modeling；config/tokenizer 私有资产不入仓。

证据范围：`cpu-tests/evidence-manifest.json` 绑定原 patch c034c716；CPU replacement/provider、HF 部分加载、MH​​C shared-state/grad 桩、CP callback 和 Muon hook 已核；`ep10-result.json` 仅 EP2 十步短测。完整单卡、全窗并行及当前 head CI 未据此宣称通过。相同 seed 冷初始化因删除四次 wrapper 重复初始化而不同；MH​​C 参数身份和真实部分加载值保护单独验证。
