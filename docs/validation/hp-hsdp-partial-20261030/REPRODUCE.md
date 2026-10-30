# 4 rank CPU 复现步骤

这是精确 worker 参数和相同容器挂载策略的复现配方；需要 Linux Docker、相同镜像 digest、驱动库/CANN/Python/VENV 目录及官方固定源。镜像和这些依赖不在包内，也未验证可从公共镜像仓库获取；其他依赖版本或设备环境只能算新实验。先将当前目录切换到本候选包根目录，并由操作者设置下列五个已存在的绝对路径环境变量：

- SOURCE：commit 5cde7c54ce3da8be8689d0783cf9b70d5306117a 的完整只读 HyperParallel 源树，包含 tests。
- PYTHON_ROOT：Python 3.12.14 根目录。
- VENV_ROOT：实际 torch 2.12.0+cpu、torch_npu 2.12.0.post3 的环境目录。
- CANN_ROOT：包含 cann/set_env.sh 的 CANN 9.2.0 目录。
- DRIVER_ROOT：驱动库根目录。

确保源文件与 red/green 15 项 pins 一致。下面的步骤在新建临时目录中只为 green 准备单文件 overlay，不修改 SOURCE；四个 output 必须独立且为空。运行不需要 NPU device 映射。四组均预期返回 1，因为每组至少一个目标 case 失败。

~~~bash
set -eu
P=$(pwd -P)
: "$SOURCE" "$PYTHON_ROOT" "$VENV_ROOT" "$CANN_ROOT" "$DRIVER_ROOT"
SRC=$(realpath "$SOURCE")
PYROOT=$(realpath "$PYTHON_ROOT")
VENV=$(realpath "$VENV_ROOT")
CANN=$(realpath "$CANN_ROOT")
DRIVER=$(realpath "$DRIVER_ROOT")
test "$(git -C "$SRC" rev-parse HEAD)" = 5cde7c54ce3da8be8689d0783cf9b70d5306117a
RUN_ROOT=$(mktemp -d)
mkdir -p "$RUN_ROOT/patch/hyper_parallel/core/fully_shard" "$RUN_ROOT/red" "$RUN_ROOT/green"
cp "$SRC/hyper_parallel/core/fully_shard/hsdp_scheduler.py" "$RUN_ROOT/patch/hyper_parallel/core/fully_shard/hsdp_scheduler.py"
printf '%s  %s\n' 2d311ed2ee490370118100d457dfbfe4f773c7d2ba5997f180238a10a1ad4cdd "$RUN_ROOT/patch/hyper_parallel/core/fully_shard/hsdp_scheduler.py" | sha256sum -c -
(cd "$RUN_ROOT/patch" && patch -p1 < "$P/hsdp-scheduler.patch")
PATCHED="$RUN_ROOT/patch/hyper_parallel/core/fully_shard/hsdp_scheduler.py"
printf '%s  %s\n' 51abc1d50805960b50c334c8f5a5b5c4a5b2a5e0d7984686bae3d2e553499fe2 "$PATCHED" | sha256sum -c -
for ARM in red green; do
  cp "$P/diagnostic/worker-14e249.py" "$RUN_ROOT/$ARM/cpu_gloo_mesh_fusion.py"
  cp "$P/diagnostic/$ARM/source-and-config-pins.json" "$RUN_ROOT/$ARM/source-and-config-pins.json"
done
printf '%s  %s\n' 14e249a3565d17232d68384200abc00121f69f53f7802f818cf031466520e8e2 "$RUN_ROOT/red/cpu_gloo_mesh_fusion.py" | sha256sum -c -

for ARM in red green; do
  for FUSION in false true; do
    RUN_DIR="$RUN_ROOT/$ARM-$FUSION"
    mkdir -p "$RUN_DIR/output"
    set --
    if [ "$ARM" = green ]; then
      set -- --mount "type=bind,src=$PATCHED,dst=/work/sources/hyper-parallel/hyper_parallel/core/fully_shard/hsdp_scheduler.py,readonly"
    fi
    INNER="source /work/runtime/cann920/cann/set_env.sh && exec /work/envs/hp-pta212/bin/python -B -m torch.distributed.run --nnodes=1 --master-addr=127.0.0.1 --master-port=62580 --nproc-per-node=4 --no-python /work/envs/hp-pta212/bin/python -B /work/diagnostic/cpu_gloo_mesh_fusion.py --mesh-shape 2,2 --comm-fusion $FUSION --output-dir /work/output"
    set +e
    docker run --rm --read-only --network none --cpus 4 --memory 8g --shm-size 1g \
      --tmpfs /tmp:rw,nosuid,nodev,size=1g \
      --mount "type=bind,src=$PYROOT,dst=/usr/local/python3.12.14,readonly" \
      --mount "type=bind,src=$VENV,dst=/work/envs/hp-pta212,readonly" \
      --mount "type=bind,src=$SRC,dst=/work/sources/hyper-parallel,readonly" \
      --mount "type=bind,src=$CANN,dst=/work/runtime/cann920,readonly" \
      --mount "type=bind,src=$DRIVER,dst=/usr/local/Ascend/driver,readonly" \
      --mount "type=bind,src=$RUN_ROOT/$ARM,dst=/work/diagnostic,readonly" \
      --mount "type=bind,src=$RUN_DIR/output,dst=/work/output" \
      "$@" --entrypoint /usr/bin/env \
      sha256:5f9e677f439d11ab4e7c888f54b142335a27eee7411cbd2ce894a4798a70146d \
      -i HOME=/tmp TMPDIR=/tmp GLOO_SOCKET_IFNAME=lo HYPER_PARALLEL_PLATFORM=torch \
      HYPER_PARALLEL_TEST_DEVICE_TYPE=cpu HP_TEST_SOURCE_ROOT=/work/sources/hyper-parallel \
      PYTHONPATH=/work/sources/hyper-parallel TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
      PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PATH=/work/envs/hp-pta212/bin:/usr/local/python3.12.14/bin:/usr/bin:/bin \
      LD_LIBRARY_PATH=/usr/local/python3.12.14/lib \
      bash --noprofile --norc -c "$INNER" \
      > "$RUN_DIR/stdout" 2> "$RUN_DIR/stderr"
    STATUS=$?
    set -e
    echo "$ARM $FUSION exit=$STATUS output=$RUN_DIR/output"
    test "$STATUS" -eq 1
    test "$(find "$RUN_DIR/output" -maxdepth 1 -name 'rank-*.json' | wc -l)" -eq 16
  done
done
~~~

原始任务还使用独立容器名、cidfile、任务标签和外部回读守卫；上述配方省略仅用于原任务身份追踪的参数，保留实际容器安全配置、设备排除、源码挂载、环境和四 rank worker 命令。请比较各 JSON 的 status、main_grad 与 pending，不能只看退出码。原 2659 文件 source manifest 本包不含，仅 15 项 worker 检查可由本包直接重做；新的执行结果不自动等同于原始回执。

