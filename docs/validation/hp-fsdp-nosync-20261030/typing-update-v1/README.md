# CPU launcher 类型注解更新：追加验证证据

本包补充一次 CPU/Gloo launcher 验证，不替换[历史证据快照 `cf5f915d`](https://github.com/xingleixu123/hyper-parallel/tree/cf5f915d9440993e8e21601aa279cc013909c20f/docs/validation/hp-fsdp-nosync-20261030)。

## 版本关系

| 内容 | SHA-256 |
| --- | --- |
| 历史快照中的 launcher | `2f4305de72b8090e51b9c0848f29f057e2c9c14335eec477f90702ae00200924` |
| 本次验证的 launcher | `391848fdfa9daa10473a9033e051c490e1c482da3ebf42ad9f961dfae6b1f688` |
| 两 rank worker | `8849b9273fd0fddec9431d704ca1f000fbb19092b423eae9fc85469b2326a895` |
| 已验证的 scheduler 修复 | `51abc1d50805960b50c334c8f5a5b5c4a5b2a5e0d7984686bae3d2e553499fe2` |

比较历史快照中的旧 launcher 与本次文件，字节变化仅为增加 `from pytest import MonkeyPatch`，以及给测试函数的参数和返回值添加类型注解；见 [launcher-typing.patch](launcher-typing.patch) 与 [intent.json](intent.json)。两版本的测试函数体未改。本包只保留小差异，未复制整套源码。

历史快照里的 **32 份诊断 JSON** 由独立诊断 worker 直接执行产生，不是本次 `391848…` launcher 的运行结果。历史快照中的仓库 launcher 回归使用的是旧版 `2f4305…`；本包记录的是更新后的单独 green 运行。历史快照及其结论保留原样。

## 本次实际 CPU 验证

固定的官方源为 `5cde7c54ce3da8be8689d0783cf9b70d5306117a`，green 使用 `51abc1…` scheduler 的只读单文件 overlay。原始 [validation.json](cpu-validation/validation.json) 记录了三文件编译、公开测试签名和 pytest `MonkeyPatch` 导入检查，以及容器退出码为 0、2659 项源文件前后校验、容器移除。

[外层原始 stdout](cpu-validation/stdout) 记录 launcher 导入时 Torch、torch_npu 和 HyperParallel 重型模块列表为空，并记录外层 pytest `1 passed`。两个 rank 的原始 stdout 分别记录 `1 passed`：[rank 0](cpu-validation/flat-logs/rank-0-stdout.log)、[rank 1](cpu-validation/flat-logs/rank-1-stdout.log)；对应 stderr 一并保留。隔离目录中的外层 pytest 有四条未注册 mark 警告，rank 日志记录只读目录下的 pytest cache 警告。

本次只验证两 rank CPU/Gloo 合成测试及上述 launcher 行为，不验证 NPU/HCCL、完整模型、性能或任意动态分支。

## 核验与范围

[原始 artifact-hashes.json](cpu-validation/artifact-hashes.json) 的 21 项原始文件在封装前均按 size 和 SHA-256 核对。四份 rank 原始日志也与原 receipt 的嵌入副本逐字节核对；[flat-logs-map.json](cpu-validation/flat-logs-map.json) 保留日志映射。[copy-audit.json](copy-audit.json) 记录复制和版本比对。

[receipt-extract.json](cpu-validation/receipt-extract.json) 是标明原 receipt SHA 的派生提取件：保留运行结果、版本 pin、源文件校验、设备授权和容器 inspect 等字段，移除 Base64、重复 stdout/stderr、传输 argv 和宿主机挂载来源路径。原 receipt、原始传输封套和完整源码仍在本地档案，不在本包中；因此原 artifact 清单中的部分路径有意不随包提供。

[sensitive-audit.json](sensitive-audit.json) 记录内容和常见凭据格式初筛；包内没有 YAML 文件或实际训练数据。`SHA256SUMS` 覆盖包内除其自身之外的全部文件，可使用 `sha256sum -c SHA256SUMS` 校验。