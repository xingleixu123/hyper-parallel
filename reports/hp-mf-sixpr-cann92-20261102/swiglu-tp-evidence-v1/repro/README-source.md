# 重建实测源码

实测 local HEAD 为 `2dd33e02f46fc5d6810e806b9d7d4717cf10a3e4`。以下公开 PR 原始提交与模型侧 CE 补丁可重建相同 tracked tree；重建过程的提交 ID 可因时间信息不同而不同。所有命令应使用已授权的代理访问 GitHub。实验 YAML 不在此归档中。

```bash
git clone https://github.com/mindspore-ai/hyper-parallel.git hp-repro
cd hp-repro
git switch --detach 4144bb242f1a696308e75bd609cb61c9143b3d8f
git fetch origin pull/1171/head:refs/remotes/repro/pr-1171
git fetch origin pull/1413/head:refs/remotes/repro/pr-1413
git fetch origin pull/1420/head:refs/remotes/repro/pr-1420
git fetch origin pull/1425/head:refs/remotes/repro/pr-1425
git fetch origin pull/1358/head:refs/remotes/repro/pr-1358
git fetch origin pull/1362/head:refs/remotes/repro/pr-1362
git cherry-pick -x f7ed3c0d2c2111816e5218ac9d08280cacc3d39c
git cherry-pick -x 4463d5a30440aedc1e9d455332a328b816c9bde9
git cherry-pick -x bab14d2c57b269f3571fd39f55088d1fe80926f0
git cherry-pick -x 8f4244be32d9684d6774aee43c3fc06b9d8d6a48
git cherry-pick -x aa56714b58abaaf10d63fd58d2e46e333068a31e
git cherry-pick -x 2fde96553707b4a01b2d6a471650d98f2bc19f4c
git cherry-pick -x 8104a78acf1b8c6a0dc03e2670cd5842bb2a6fdb
git cherry-pick -x 6066caa77d6102aff240fd5f0dd313cd1e22be79
# 将本归档中的 ce-model-only.patch 放到当前目录后应用。
git apply ce-model-only.patch
git add hyper_parallel/models/xing40/modeling_xing40.py
git write-tree
# 上一条命令必须输出：
# 3d4a21c306e69bfb2c0d17ddc3d8519af0c6ef76
```

独立微模块不读取模型训练数据、权重或生产 YAML；按主 README 配置兼容 NPU 环境后运行。
