# 5c 两 rank CPU/Gloo：no-sync setter 红绿对照

仅验证最小合成模型和 Gloo SUM 路径，不是 NPU/HCCL 性能或四 rank HSDP 测试。官方 5c 源是经 manifest 固定的 2659 文件导出；green 只在容器对 scheduler 做只读单文件 overlay。两臂的 worker SHA 均为 `14e249a3565d17232d68384200abc00121f69f53f7802f818cf031466520e8e2`；原 scheduler SHA `2d311ed2ee490370118100d457dfbfe4f773c7d2ba5997f180238a10a1ad4cdd`，patch SHA `51abc1d50805960b50c334c8f5a5b5c4a5b2a5e0d7984686bae3d2e553499fe2`。

| 组别 | fusion | 结果 | 进程退出 |
| --- | --- | --- | --- |
| red | false | 6/8 通过；两 rank micro0_only_false 失败 | 1 |
| green | false | 8/8 通过 | 0 |
| red | true | 6/8 通过；两 rank micro0_only_false 失败 | 1 |
| green | true | 8/8 通过 | 0 |

两种 fusion 下，red 的失败均为 `early.weight` 在末微批没有出现时 `main_grad=None` 且未分片累积梯度仍 pending；green 的 rank 0 两行全为 3、2，rank 1 两行全为 3、4，精确等于独立完整 world SUM oracle，pending 已清。两臂同 rank/case 的初始参数、两个输入、两个 loss 和 oracle 预期全部一致。both_false 等其余对照通过，未出现该模型中的重复归约。

每组单独串行：宿主运行前后各逐项验证 2659 文件，worker 自验 15 个文件 hash、import 源位置和测试 fixture，运行中 Docker inspect 验固定镜像、只读 source/diagnostic/PY/VENV/CANN/driver、green 唯一只读 overlay、网络 none、无 device/DeviceRequests/device cgroup grants 且非 privileged。四次容器均 `--rm` 并在退出后确认不存在；child 两级均 `python -B`，使用 loopback `127.0.0.1:62580` 和 Gloo CPU。日志包含现有 torch_npu storage warning；证据不声称 torch_npu 未加载。

每组目录保留 8 个结果 JSON、原始 stdout/stderr、Docker/SSH argv、实际远端 receipt 与输入代码；`artifact-hashes.json` 给出逐文件长度和 SHA，[collection-summary.json](collection-summary.json) 给出各组清单的 hash。原始远端目录保留于 `/home/xxl/hp-mf-cann92-pr310-20261028/runtime/tp-nosync-fix-cpu-v1/runs/`，没有对生产源作写入。