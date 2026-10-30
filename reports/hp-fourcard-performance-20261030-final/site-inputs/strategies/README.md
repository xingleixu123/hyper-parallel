# 全策略网站输入候选（本地，未绘图）

本包只整理已经实际训练的 1511–1540 共30个 step，**不连接网站、不绘图、不发布**。18 份 JSONL 各30行，字段为真实 step、四 rank 各自同步计时的最大值及 `16384/该步时长` 的名义 tokens/s；不平滑、不删除点、不将最大 rank 时长改称全局 makespan，也不将名义 tokens 当作有效 mask tokens。

8组配置见 [groups.json](groups.json)：125 noRAF/full反向配对；129 HSDP/FSDP两对分别隔离；125旧TP/FSDP配对；129 CP fixed 与FSDP fixed非同期参考；129 EP2复测及两个非同期 FSDP参考；125 原1a full/off/selective；125 修复后 HSDP default/explicit1配对。配对组内才共享指标名；CP、EP参考和独立重算配置均各用不同指标名，避免网站自动计算跨条件差值。不同组的指标名全部不同；不能将它们汇总为无条件速度排名。各run为离散监测或监测不完整等实际状态，不证明计时窗口连续独占。EP的两个FSDP参考有意从已公布的两次记录重复引用，并没有新实验。

[summary.json](summary.json) 给每组每run的mean/median/CV、极值和两种吞吐统计：官网报表对应 `16384 / mean(seconds)`，不等于对逐步 `16384 / seconds` 求均值。[source-index.csv](source-index.csv) 将每个输出点绑定到原 CSV 文件 SHA、精确原行号及每行原字节 SHA；HSDP 每个点绑定该 step 的四个 rank 行，并逐点与已封存 max 曲线核对。

[memory-peaks.csv](memory-peaks.csv) 是每个run每张物理卡的**静态** timed peak allocated、peak reserved 与 warm 唯一 storage，保留原始汇总的单位和JSON指针；仅部分来源有原始 bytes。它不是逐步内存曲线，缺失 bytes 不得反算冒充原始精度。reserved 不是物理HBM，peak差或reserved−allocated不能当作 activation、碎片或泄漏。

每组真实 source、host、配置指纹、run ID、metric 及适用限制均在 groups.json；文件完整哈希见 [SHA256SUMS](SHA256SUMS)。在新空同级目录复制 `build_candidate.py` 并以 `python -B build_candidate.py` 可从任务内固定原始表重建；parser-check.json 仅为离线 helper 解析验证，不是网站解析或可交互页面验收。
