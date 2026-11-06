# log.javazero.top 实际在线后台导出

本目录的36张PNG、36份CSV与website-report.png由真实在线网站、Edge后台会话原生导出，所有88000个点均与12份导入文件逐点核对。analysis-public.json记录complete/online与匹配规则；该complete只指后台解析/导出。普通Edge的可见交互核验因工具无法可靠确认当前URL而停止，未取得经核验的交互会话或可复现分享链接；不以普通网站地址代替会话分享。

logs/是由七组封印CSV生成的标量JSONL，不是原生NPU日志全量。原始LOSS/rawMTP/rawAUX保留round-trip数值，delta为有符号目标−参考，cumulative为从1501到当前step的有符号算术均值；没有平滑、插值或裁剪。原生网站默认横轴从0起，1501以前为空轴段；所有导入点仍严格1501–2500，汇总图只把显示轴收紧到该范围。

重载时按upload-plan.json的01–12顺序上传12份logs，按照globalParsingConfig.json的18项name/keyword配置解析，使用记录step。provenance-manifest.json保留准备阶段的19份源CSV精确SHA和逐行映射；其中prepared状态是输入准备的历史快照，当前实际网站执行结果见analysis-public.json。数据来源相对于../numerical；只读取统计标量，不含YAML、tokens、RNG或权重。

常数Zero仅为signed delta/cumulative零线。±0.005仅显示LOSS/rawMTP的cumulative参考，**只在2500判断完整1000步均值，不是单步或早期prefix门禁**。AUX没有阈值；四卡AUX只对新FSDP完整global raw参考，不能对旧rank0末micro AUX。网站对零基线的相对误差会显示0，该百分比不具验收意义，保留的有符号原值才用于统计。

源身份：单卡MF52f/HP49bf，四卡主参考旧连续a852，四个TP1目标496、两TP/LP目标3292。GBS单卡2/四卡4，LBS1，1Dense+1MoE+1MTP、MLA4096、fused mHC、clamp/CMHC关闭。完整版本见数值子包。后台导出未测统一稳态性能，不作速度/显存收益结论。

本机可复现命令为既有训练日志助手：node plot.cjs --headless --out <新空目录> logs/01_Zero_reference.log ... logs/12_SP3292.log。普通可见页面须通过可用的Edge控制工具或手动重载；本报告不会借后台导出冒称可见页已经验证。
