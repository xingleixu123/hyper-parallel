# README 后续四文件 NPU 验证

以 commit `7f4a3fd7cac6da3c465743b57d832042bcc4ae29` 上四个未提交文件快照，叠加上轮已验证单卡组合，不表示完整 Git tree。`stage.json` 给四文件前后SHA；最终 `source-manifest.json` SHA `ea0bb3500e6b4276fddaa97475918c56501efe2c309e320f56030a88c4d3f233`，1514文件执行前与执行后逐项核验相同。远端独立stage仍保留；源tar与脚本本机归档。

129物理卡5实际NPU运行：复用单卡设备cgroup、PID namespace与 `npu-smi` 准入readback，启动前两次为空、准入时设备PID与worker对应，结束容器移除且卡5空闲。设备 910B3；torch `2.12.0+cpu` 与 `torch_npu 2.12.0.post3`，CANN toolkit `9.2.0 / V100R001C12SPC001B237`。`attempt2/stdout.log` 22 passed，controller returncode 0：原20项适配保留功能断言，另加固定seed随机初始化等价；另加真实HF `Xing4ForCausalLM.from_pretrained` 部分加载，漏count且保留权重/bias：加载过程在CPU，之后移动到NPU再次检查count int64为零、已加载权重与bias不变；它不是NPU上执行HF初始化。`attempt1` 是新增fixture直接使用未补默认字段的HF config导致构建失败（21 passed/1 failed），随后改成真实 `create_config` 路径，非生产失败；原失败日志与按其hash重建的fixture均保留。

独立公共loader补充：`attempta4/stdout.log` 为真实NPU tensor上使用HF `PreTrainedModel.initialize_weights` 和 guarded nn.init，动态加载 before `1306f7754b9c7835b2278c1f97477fb1df9fe107114f387efb7853214bda0fb2`、after `8f623e58927b1923fecf92330fc3a0bc414e1a5e8d37d9111051b18a8f463385`；strict ignored missing下 before count保留sentinel `[71,71]`，after为 `[0,0]`，两者已加载NPU weight/bias不变，测试1 passed/return0。`attempta3` 同一断言首次通过但pytest捕获了成功打印，a4只为留直接数值而加 `-s`；无产品代码变更。

审计边界：控制器由旧模板派生，`controller.json` 中 controller_sha256、worker_sha256、candidate_sha256、review_sha256 为未更新的旧展示常量，不用于本次身份结论。控制器实际检查本次source manifest全量、worker/test hash及设备准入；controller自身并非自校验。a2实际controller `6d75a155c9c001f4e5ba615d5413cf1c0559c402598c09227439e0d4ed7e7d3a`，worker `4761c3a31dd36cf093596f6415f587fe626303a97b711bad6a25efdf72d38e1c`，test `5f687b34b76557f26b9e0f66a1c6df428b3b91631f203be67c23567831bca6c0`；a4 controller `0074729946e81157d3083905647c148db153024ad39860c0febc0a6e8bf08a3e`，worker `b6d7ed07c8e6f8e9785cd27e3f48951288c21c652be0b844b21e287f35b048b8`，test `afa308a567e4ec275974841f402640ce03ee5ae6a9e55b71509c89d8fbd55f8b`，均事后本机与远端一致。物理映射和实际返回码见各attempt的 `controller.json` / `physical-mapping-release.json` / `probe-result.json`。

无本轮完整训练和四卡100步，不将前轮结论归属于此快照。未触碰上轮因自动审批拒绝清理的本机single副本；本轮stage、bundle及小型证据保留以复现，无自建大临时文件需删除。
