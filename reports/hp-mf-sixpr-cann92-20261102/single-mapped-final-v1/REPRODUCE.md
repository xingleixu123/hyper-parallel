# 复算与复现实验

本目录只包含公开 scalar 和元数据；不包含训练 checkpoint、样本、RNG 字节或原始配置。以下命令使用 Python 3 标准库，在本目录执行，可从 CSV 重新计算全窗与每 100 步统计，并与两份已封存统计逐字段比较：

```sh
python3 -B verify_paired_loss.py
```

验证也会核对 CSV、`statistics.json` 和 `independent-verification.json` 的 SHA。该命令验证公开 scalar 文件及统计，不单独重做原始样本、恢复状态或 NPU 训练的检查。

实测集成源码可按以下命令获取并核对 commit 与 tree：

```sh
git clone https://github.com/xingleixu123/hyper-parallel.git
cd hyper-parallel
git fetch origin refs/heads/codex/hp-mf-sixpr-logical-c419-20261102
git checkout --detach c419d1356af9eccec8377e636a6d9e5a2b7a285b
test "$(git rev-parse HEAD)" = c419d1356af9eccec8377e636a6d9e5a2b7a285b
test "$(git rev-parse HEAD^{tree})" = 82e641d37ea1073eb8690a4345cbc7f6fee23767
```

原实验的续训入口使用了如下 **参数模板**；尖括号需替换为已审核私有运行材料和本地可用资源，不能当作一条可直接执行的公共命令：

```sh
python3 -B <reviewed-private-controller.py> \
  --phase resume --stop 2500 \
  --runid mf1500-logical-mapped-resume2500-v3 \
  --resume-output <mapped-native-1500-checkpoint-root> \
  --execute-reviewed --operator ROOT
```

实测集成源码已公开，获取前请按当前环境配置 GitHub 代理；公开功能提交 909 与实测源码仅已核验两个生产 adapter 文件相同。完整复现还需要同一 MF 来源 checkpoint 及其映射后的完整训练状态、原数据与采样次序、经审核运行器与私有配置、兼容的软件和单卡环境。公开版本与 SHA 见 `versions.json`；私有 v6 配置归档固定于 [commit `565682bf00c3e74ec80ad43113cae51c0c7e50e8`](https://github.com/xingleixu123/mindformers-private/tree/565682bf00c3e74ec80ad43113cae51c0c7e50e8/private_review/hp-mf-sixpr-cann92-20261102-single-mapped-v6)，需要相应访问权限。源码、输入或配置不同时，不应把新运行冒充同一次实验。
