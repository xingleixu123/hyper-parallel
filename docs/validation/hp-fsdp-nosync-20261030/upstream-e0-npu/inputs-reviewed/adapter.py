#!/usr/bin/env python3
"""Review candidate for a two-rank historical-context no-sync HCCL diagnostic.

No action is performed on import. Root must review each actual phase before use.
"""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import socket
import stat
import subprocess
import sys
import time

R = Path("/home/xxl/hp-mf-cann92-pr310-20261028")
BASE = R / "runtime/hp-upstream-nosync-e0-125-v2"
INPUTS = BASE / "inputs"
SOURCE = BASE / "source/upstream"
SOURCE_MANIFEST = INPUTS / "source-manifest.json"
OVERLAY = INPUTS / "patched_scheduler.py"
SOURCE_MANIFEST_SHA = "2ee3981b6643d6ffa27c3e670bf1c0fca7500379d436874af0a761e86d3b242a"
SCHEDULER_REL = "hyper_parallel/core/fully_shard/hsdp_scheduler.py"
BASE_SCHEDULER_SHA = "2d311ed2ee490370118100d457dfbfe4f773c7d2ba5997f180238a10a1ad4cdd"
PATCH_SCHEDULER_SHA = "51abc1d50805960b50c334c8f5a5b5c4a5b2a5e0d7984686bae3d2e553499fe2"
WORKER_SHA = "5bea6387d6ce46bad195fd6d13d1338d6877fe489c1ac54770865250d04fb00e"
HEAD = "e0cdf62c9d66f1f9ec48c6f15df87dcf4ce80a5f"
LIBRARY = R / "runtime/perf-procscan-cgroupfirst-v1/hp33096_129_nfs_data_library.py"
LIBRARY_SHA = "6c53721550fc94d0f25f3d688b23290506d4f08821cc591bbb9c3f79caee8354"
PY = R / "runtime/python3.12.14"
VENV = R / "envs/hp-pta212-hf521-cpu"
CANN = Path("/home/xxl/hp-mf-cann92-pr301-20261027/runtime/cann920")
DRIVER = Path("/usr/local/Ascend/driver")
IMAGE = "sha256:5f9e677f439d11ab4e7c888f54b142335a27eee7411cbd2ce894a4798a70146d"
ENV_SHA = "e14f2704e4d568d79ef9df15546f3ed82e0613455ccb512f8c9e8c8b14b948cc"
TASK = "hp-upstream-nosync-e0-125-v2"
CARDS = (1, 2)
DEVICES = [f"/dev/davinci{i}" for i in CARDS] + ["/dev/davinci_manager", "/dev/devmm_svm", "/dev/hisi_hdc"]
TIMEOUT_SECONDS = 2340
ARM_CONFIG = {"green": (63620, "63640-63655")}
ARM = RUN = PRIVATE = OUTPUT = RUN_ID = NAME = TAG = MASTER_PORT = HCCL_RANGE = None

def set_arm(arm):
    global ARM, RUN, PRIVATE, OUTPUT, RUN_ID, NAME, TAG, MASTER_PORT, HCCL_RANGE
    require(arm in ARM_CONFIG, "arm differs")
    ARM = arm
    MASTER_PORT, HCCL_RANGE = ARM_CONFIG[arm]
    RUN = BASE / "runs" / arm
    PRIVATE = RUN / "private"
    OUTPUT = RUN / "output"
    RUN_ID = "hp-upstream-nosync-e0-125-" + arm + "-v2"
    NAME = RUN_ID
    TAG = RUN_ID + ":execute"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_arg(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA-256")
    return value


def read_pinned(path, expected):
    path = Path(path)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), "not a regular input: " + str(path))
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        key = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        require(key(before) == key(opened), "input changed before read")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read()
        require(key(opened) == key(os.fstat(fd)) == key(path.lstat()), "input changed during read")
        require(digest(raw) == hash_arg(expected), "input hash differs: " + str(path))
        return raw
    finally:
        os.close(fd)


def write_new(path, value):
    path = Path(path)
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return digest(raw)


def write_raw_new(path, raw):
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def fresh_idle(stage):
    """Record two cards' raw npu-smi output before evaluating strict idleness."""
    require(stage in ("create", "execute"), "invalid idle stage")
    folder = RUN / (stage + "-idle-raw")
    folder.mkdir(mode=0o700, exist_ok=False)
    rows = []
    try:
        for card in CARDS:
            for what in ("proc-mem", "usages"):
                argv = ["npu-smi", "info", "-t", what, "-i", str(card), "-c", "0"]
                try:
                    result = subprocess.run(argv, capture_output=True, timeout=15, check=False)
                    out, err, rc, timed_out = result.stdout, result.stderr, result.returncode, False
                except subprocess.TimeoutExpired as exc:
                    out, err, rc, timed_out = exc.stdout or b"", exc.stderr or b"", None, True
                if isinstance(out, str):
                    out = out.encode()
                if isinstance(err, str):
                    err = err.encode()
                prefix = f"card-{card}-{what}"
                out_path = folder / (prefix + ".stdout")
                err_path = folder / (prefix + ".stderr")
                write_raw_new(out_path, out)
                write_raw_new(err_path, err)
                rows.append({"card": card, "query": what, "argv": argv, "returncode": rc,
                             "timed_out": timed_out, "stdout": str(out_path.relative_to(RUN)),
                             "stdout_sha256": digest(out), "stderr": str(err_path.relative_to(RUN)),
                             "stderr_sha256": digest(err)})
                require(not timed_out and rc == 0 and not err.strip(), "npu-smi query failed or timed out")
                text = out.decode("utf-8", errors="strict")
                if what == "proc-mem":
                    require(text.strip() == "No process in device.", "card has an occupant")
                else:
                    ai = re.findall(r"Aicore Usage Rate\(%\)\s*:\s*(\d+)", text)
                    hbm = re.findall(r"HBM Usage Rate\(%\)\s*:\s*(\d+)", text)
                    require(len(ai) == len(hbm) == 1 and int(ai[0]) == 0 and int(hbm[0]) <= 8,
                            "card utilization differs")
    except Exception as exc:
        write_new(RUN / (stage + "-idle.json"), {"stage": stage, "status": "failed",
                                                    "checks": rows, "error": repr(exc)})
        raise
    return write_new(RUN / (stage + "-idle.json"), {"stage": stage, "status": "passed", "checks": rows})


def command(argv, *, timeout=60):
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    require(completed.returncode == 0, "command failed: " + argv[0] + ": " + completed.stderr[-2000:])
    return completed.stdout


def check_host(pins_sha):
    require(os.geteuid() == 0 and socket.gethostname() == "ascend125", "requires root on ascend125")
    require(not BASE.is_symlink() and BASE.is_dir(), "unsafe candidate base")
    require(Path(__file__).resolve(strict=True) == (BASE / "adapter.py").resolve(strict=True), "adapter path differs")
    doc = json.loads(read_pinned(BASE / "pins.json", pins_sha))
    require(doc.get("schema") == "nosync-upstreammodule-e0-pins-v2", "pins schema differs")
    for component in ("adapter.py", "observe.py", "release.py"):
        require(doc.get(component.removesuffix(".py") + "_sha256") == sha_file(BASE / component), "component hash differs")
    fixed = {"worker_sha256": WORKER_SHA, "source_manifest_sha256": SOURCE_MANIFEST_SHA,
             "original_scheduler_sha256": BASE_SCHEDULER_SHA, "patch_scheduler_sha256": PATCH_SCHEDULER_SHA,
             "baseline_library_sha256": LIBRARY_SHA, "image": IMAGE, "cards": list(CARDS), "head": HEAD}
    require(all(doc.get(k) == v for k, v in fixed.items()), "fixed pin differs")
    read_pinned(INPUTS / "worker.py", WORKER_SHA)
    read_pinned(INPUTS / "source-manifest.json", SOURCE_MANIFEST_SHA)
    read_pinned(SOURCE_MANIFEST, SOURCE_MANIFEST_SHA)
    read_pinned(OVERLAY, PATCH_SCHEDULER_SHA)
    read_pinned(LIBRARY, LIBRARY_SHA)
    return runpy.run_path(str(LIBRARY), run_name="historical_library")


def mounts():
    result = [(PY, "/usr/local/python3.12.14", True), (VENV, "/work/envs/hp-pta212", True),
              (SOURCE, "/work/sources/hyper-parallel", True), (CANN, "/work/runtime/cann920", True),
              (DRIVER, "/usr/local/Ascend/driver", True), (INPUTS, "/work/input", True),
              (PRIVATE, "/work/private", False), (OUTPUT, "/work/run", False)]
    if ARM == "green":
        result.append((OVERLAY, "/work/sources/hyper-parallel/" + SCHEDULER_REL, True))
    return result


def source_and_environment(library):
    require(SOURCE.is_dir() and not SOURCE.is_symlink() and SOURCE.resolve(strict=True) == SOURCE
            and not SOURCE.parent.is_symlink(), "source directory differs")
    doc = json.loads(read_pinned(SOURCE_MANIFEST, SOURCE_MANIFEST_SHA))
    require(doc.get("schema") == "official-cpu-source-export-v1" and doc.get("head") == HEAD,
            "source identity differs")
    rows = doc.get("files")
    require(isinstance(rows, dict) and len(rows) == 2676 and rows[SCHEDULER_REL]["sha256"] == BASE_SCHEDULER_SHA,
            "source manifest coverage differs")
    seen = set()
    def walk_error(exc):
        raise RuntimeError("source walk failed") from exc
    for current, dirs, names in os.walk(SOURCE, followlinks=False, onerror=walk_error):
        parent = Path(current)
        pst = parent.lstat()
        require(stat.S_ISDIR(pst.st_mode) and not (pst.st_mode & 0o222), "source directory writable or unsafe")
        for name in dirs:
            d = (parent / name).lstat()
            require(stat.S_ISDIR(d.st_mode) and not (d.st_mode & 0o222), "source symlink or writable directory")
        for name in names:
            f = parent / name; rel = f.relative_to(SOURCE).as_posix()
            fst = f.lstat()
            require(rel in rows and stat.S_ISREG(fst.st_mode) and fst.st_nlink == 1 and
                    not (fst.st_mode & 0o222), "unexpected/writable source file: " + rel)
            raw = read_pinned(f, rows[rel]["sha256"])
            require(len(raw) == rows[rel]["size"], "source size differs: " + rel)
            seen.add(rel)
    require(seen == set(rows), "source files missing")
    read_pinned(OVERLAY, PATCH_SCHEDULER_SHA)
    fp = library["env_fingerprint"]()
    require(fp["sha256"] == ENV_SHA, "historical environment fingerprint differs")
    keys = ("hyper_parallel/__init__.py", "hyper_parallel/core/fully_shard/api.py", SCHEDULER_REL,
            "hyper_parallel/core/fully_shard/hsdp_state.py", "hyper_parallel/core/fully_shard/hsdp_param.py",
            "hyper_parallel/core/fully_shard/param_group.py", "hyper_parallel/core/fully_shard/hsdp_utils.py",
            "hyper_parallel/core/fully_shard/utils.py", "hyper_parallel/core/fully_shard/hook_function.py",
            "hyper_parallel/core/dtensor/device_mesh.py")
    files = {"/work/sources/hyper-parallel/" + x: (PATCH_SCHEDULER_SHA if ARM == "green" and x == SCHEDULER_REL else rows[x]["sha256"]) for x in keys}
    files.update({"/work/input/worker.py": WORKER_SHA, "/work/input/source-manifest.json": SOURCE_MANIFEST_SHA})
    return {"source_head": HEAD, "source_manifest_sha256": SOURCE_MANIFEST_SHA, "source_file_count": len(seen),
            "scheduler_sha256": PATCH_SCHEDULER_SHA if ARM == "green" else BASE_SCHEDULER_SHA,
            "files": files, "environment": fp}


def inspect(cid):
    require(re.fullmatch(r"[a-f0-9]{64}", cid or ""), "container id is not full")
    rows = json.loads(command(["docker", "inspect", cid]))
    require(len(rows) == 1 and rows[0]["Id"] == cid, "container identity differs")
    return rows[0]


def cgroup_folder(controller, path, cid):
    require(path == "/docker/" + cid or path == "/system.slice/docker-" + cid + ".scope",
            "target is not this container leaf")
    root = Path("/sys/fs/cgroup") / controller
    folder = root / path.lstrip("/")
    require(folder.resolve(strict=True).is_relative_to(root.resolve(strict=True)), "cgroup escaped root")
    require(folder.is_dir() and not folder.is_symlink(), "unsafe cgroup leaf")
    return folder


def check_acl(folder):
    nodes = {}
    for name in DEVICES:
        item = os.stat(name)
        require(stat.S_ISCHR(item.st_mode), "accelerator node is not a character device")
        nodes[name] = [os.major(item.st_rdev), os.minor(item.st_rdev)]
    expected = {tuple(value) for value in nodes.values()}
    require(len(expected) == 5 and len({nodes[f"/dev/davinci{i}"][0] for i in CARDS}) == 1,
            "accelerator device numbers differ")
    target_majors = {major for major, _ in expected}
    raw = (folder / "devices.list").read_text()
    seen = set()
    for line in raw.splitlines():
        kind, number, permissions = line.split()
        major, minor = number.split(":")
        if kind not in ("a", "c") or not set(permissions) & set("rw"):
            continue
        require(major != "*", "wildcard device major permits broader access")
        if major.isdigit() and int(major) in target_majors:
            require(minor.isdigit() and (int(major), int(minor)) in expected,
                    "device cgroup grants other accelerator/control nodes")
            if set("rw").issubset(permissions):
                seen.add((int(major), int(minor)))
    require(seen == expected, "device cgroup lacks expected nodes")
    return {"nodes": nodes, "devices_list": raw, "devices_list_sha256": digest(raw.encode())}


def snapshot(library, cid, expected_pid=None, expected_tick=None):
    obj = inspect(cid)
    require(obj["State"]["Running"] and obj["Image"] == IMAGE, "container not running or image differs")
    pid = obj["State"]["Pid"]
    require(type(pid) is int and pid > 1, "invalid init pid")
    tick = library["starttick"](pid)
    require(expected_pid is None or (pid == expected_pid and tick == expected_tick), "init identity changed")
    host = obj["HostConfig"]
    require(not host["Privileged"] and host["ReadonlyRootfs"] and host["NetworkMode"] == "host",
            "container boundary differs")
    require(not host.get("DeviceCgroupRules") and not host.get("DeviceRequests")
            and not host.get("CapAdd") and not host.get("SecurityOpt"), "unexpected device/security grants")
    actual_devices = {(v["PathOnHost"], v["PathInContainer"], v["CgroupPermissions"])
                      for v in host.get("Devices", [])}
    require(len(host.get("Devices", [])) == 5
            and actual_devices == {(v, v, "rwm") for v in DEVICES}, "container device grants differ")
    labels = obj["Config"].get("Labels") or {}
    require(labels.get("codex.task") == TASK and labels.get("codex.run") == RUN_ID
            and obj["Config"]["Entrypoint"] == ["/bin/sleep"]
            and obj["Config"]["Cmd"] == ["infinity"], "container labels or entrypoint differ")
    binds = [v for v in obj["Mounts"] if v["Type"] == "bind"]
    actual_mounts = {(os.path.realpath(v["Source"]), v["Destination"], v["RW"], v["Type"])
                     for v in binds}
    expected_mounts = {(os.path.realpath(src), dest, not ro, "bind") for src, dest, ro in mounts()}
    require(len(actual_mounts) == len(binds) == len(expected_mounts)
            and actual_mounts == expected_mounts, "container bind mounts differ")
    destinations = [v["Destination"] for v in obj["Mounts"]]
    require(len(destinations) == len(set(destinations)), "duplicate mount destination")
    other_mounts = sorted((v for v in obj["Mounts"] if v["Type"] != "bind"),
                          key=lambda v: v["Destination"])
    groups = library["cgroup_paths"](pid)
    device_dir = cgroup_folder("devices", groups["devices"], cid)
    memory_dir = cgroup_folder("memory", groups["memory"], cid)
    acl = check_acl(device_dir)
    for directory in (device_dir, memory_dir):
        members = {int(v) for v in (directory / "cgroup.procs").read_text().split()}
        require(members == {pid}, "container cgroup is not idle with only its init")
    ns = {n: os.readlink(f"/proc/{pid}/ns/{n}") for n in ("pid", "uts", "ipc", "net", "user")}
    ns["mnt"] = os.readlink("/proc/self/ns/mnt")
    return {"cid": cid, "init_pid": pid, "init_starttick": tick, "target_cgroups": groups,
            "expected_entry_namespaces": ns, "device_acl": acl,
            "bind_mounts": sorted([list(v) for v in actual_mounts]), "other_mounts": other_mounts,
            "source_and_environment": source_and_environment(library)}


def create(library, pins_sha):
    require(not RUN.exists() and not RUN.is_symlink(), "run already exists")
    require(not command(["docker", "ps", "-a", "--filter", "name=^/" + NAME + "$", "--format", "{{.ID}}"]).strip(),
            "candidate container name already exists")
    require((BASE / "runs").is_dir() and not (BASE / "runs").is_symlink(), "runs directory missing or unsafe")
    RUN.mkdir(mode=0o700)
    PRIVATE.mkdir(mode=0o700)
    OUTPUT.mkdir(mode=0o700)
    for name in ("home", "tmp", "cache", "ascend-log"):
        (PRIVATE / name).mkdir(mode=0o700)
    write_new(RUN / "create-start.json", {"run_id": RUN_ID, "pins_sha256": pins_sha})
    idle_sha = fresh_idle("create")
    argv = ["docker", "run", "-d", "--name", NAME, "--read-only", "--network", "host",
            "--shm-size", "16g", "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g",
            "--tmpfs", "/var/tmp:rw,nosuid,nodev,size=1g",
            "--label", "codex.task=" + TASK, "--label", "codex.run=" + RUN_ID]
    for name in DEVICES:
        argv += ["--device", name]
    for source, dest, ro in mounts():
        require(source.exists(), "mount source missing: " + str(source))
        argv += ["--mount", f"type=bind,src={source},dst={dest}" + (",readonly" if ro else "")]
    argv += ["--entrypoint", "/bin/sleep", IMAGE, "infinity"]
    cid = command(argv).strip()
    require(re.fullmatch(r"[a-f0-9]{64}", cid), "docker did not return a full id")
    write_new(RUN / "create-cid.json", {"cid": cid, "argv": argv})
    info = snapshot(library, cid)
    receipt = {"schema": "nosync-nsenter-create-v1", "run_id": RUN_ID, "pins_sha256": pins_sha,
               "fresh_idle_sha256": idle_sha, "argv": argv, "container": info}
    value = write_new(RUN / "create-receipt.json", receipt)
    print(json.dumps({"cid": cid, "create_sha256": value}, sort_keys=True))


def preflight(library, pins_sha, create_sha):
    receipt = json.loads(read_pinned(RUN / "create-receipt.json", create_sha))
    require(receipt["run_id"] == RUN_ID and receipt["pins_sha256"] == pins_sha, "create identity differs")
    first = receipt["container"]
    current = snapshot(library, first["cid"], first["init_pid"], first["init_starttick"])
    require(current == first, "preflight identity or full device ACL changed")
    doc = {"schema": "nosync-nsenter-preflight-v1", "run_id": RUN_ID, "pins_sha256": pins_sha,
           "create_sha256": create_sha, "snapshot": current}
    value = write_new(RUN / "preflight.json", doc)
    print(json.dumps({"preflight_sha256": value, "cid": first["cid"]}, sort_keys=True))


def check_review(pins_sha, preflight_sha, review_sha):
    pre = json.loads(read_pinned(RUN / "preflight.json", preflight_sha))
    rev = json.loads(read_pinned(RUN / "root-review.json", review_sha))
    snap = pre["snapshot"]
    wanted = {"schema": "nosync-nsenter-root-review-v1", "arm": ARM, "run_id": RUN_ID, "approved": True,
              "pins_sha256": pins_sha, "preflight_sha256": preflight_sha, "cid": snap["cid"],
              "init_pid": snap["init_pid"], "init_starttick": snap["init_starttick"],
              "devices_list_sha256": snap["device_acl"]["devices_list_sha256"], "physical_cards": list(CARDS),
              "complete_device_acl_reviewed": True, "task_container_ownership_verified": True,
              "source_and_overlay_reviewed": True, "actual_entry_security_unproven_until_observed": True}
    require(pre["run_id"] == RUN_ID and pre["pins_sha256"] == pins_sha, "preflight identity differs")
    require(all(rev.get(k) == v and (v is not True or rev.get(k) is True) for k, v in wanted.items()),
            "root review differs")
    return pre


def descendant(pid, ancestor):
    for _ in range(64):
        if pid == ancestor:
            return True
        if pid <= 1:
            return False
        fields = Path(f"/proc/{pid}/status").read_text().splitlines()
        parent = next(row.split()[1] for row in fields if row.startswith("PPid:"))
        pid = int(parent)
    return False


def execute(library, pins_sha, preflight_sha, review_sha):
    pre = check_review(pins_sha, preflight_sha, review_sha)
    prior = pre["snapshot"]
    now = snapshot(library, prior["cid"], prior["init_pid"], prior["init_starttick"])
    require(now == prior, "snapshot changed after root review")
    require(not any(PRIVATE.iterdir()) or {p.name for p in PRIVATE.iterdir()} == {"home", "tmp", "cache", "ascend-log"},
            "private directory has prior output")
    write_new(RUN / "execute-once.json", {"run_id": RUN_ID, "review_sha256": review_sha,
                                           "policy": "unknown outcome must not be automatically retried"})
    idle_sha = fresh_idle("execute")
    spec = {"namespaces": now["expected_entry_namespaces"], "cgroups": now["target_cgroups"],
            "ro": ["/"] + [dest for _, dest, ro in mounts() if ro],
            "rw": [dest for _, dest, ro in mounts() if not ro],
            "files": now["source_and_environment"]["files"], "env_fingerprint": ENV_SHA,
            "evidence": "/work/private/entry-evidence.json", "ack": "/work/private/entry-ack.json"}
    encoded = base64.b64encode(json.dumps(spec, sort_keys=True).encode()).decode()
    env = ["HOME=/work/private/home", "TMPDIR=/work/private/tmp", "XDG_CACHE_HOME=/work/private/cache",
           "ASCEND_PROCESS_LOG_PATH=/work/private/ascend-log", "HYPER_PARALLEL_PLATFORM=torch",
           "HYPER_PARALLEL_TEST_DEVICE_TYPE=npu", "TORCH_DEVICE_BACKEND_AUTOLOAD=0",
           "PYTHONPATH=/work/sources/hyper-parallel", "PATH=/work/envs/hp-pta212/bin:/usr/local/python3.12.14/bin:/usr/bin:/bin",
           "LD_LIBRARY_PATH=/usr/local/python3.12.14/lib", "PYTHONDONTWRITEBYTECODE=1", "PYTHONUNBUFFERED=1",
           "ASCEND_RT_VISIBLE_DEVICES=0,1", "HP_DIAG_CID=" + now["cid"], "HP_PHYSICAL_CARDS=1,2",
           "HCCL_NPU_SOCKET_PORT_RANGE=" + HCCL_RANGE, "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "HP_PR310_ID=" + TAG]
    worker = ["/work/envs/hp-pta212/bin/python", "-B", "-m", "torch.distributed.run", "--nnodes=1",
              "--master-addr=127.0.0.1", "--master-port=" + str(MASTER_PORT), "--nproc-per-node=2", "--no-python",
              "/work/envs/hp-pta212/bin/python", "-B", "/work/input/worker.py", "--arm", ARM,
              "--run-id", RUN_ID, "--output-dir", "/work/run"]
    argv = ["nsenter", "-t", str(now["init_pid"]), "--root", "--pid", "--uts", "--ipc", "--net",
            "--wd", "--", "/usr/bin/env", "-i", *env, "bash", "--noprofile", "--norc", "-c",
            'source /work/runtime/cann920/cann/set_env.sh && cd "$1" && shift && exec "$@"',
            "nosync", "/work/private", "/work/envs/hp-pta212/bin/python", "-B", "-I", "-c",
            library["ENTRY_CODE"], encoded, *worker]
    start, end = (int(v) for v in HCCL_RANGE.split("-"))
    tested_ports = {str(port): library["port"](port) for port in [MASTER_PORT, *range(start, end + 1)]}
    write_new(RUN / "execute-start.json", {"tested_ports": tested_ports, "run_id": RUN_ID, "cid": now["cid"],
                                             "preflight_sha256": preflight_sha,
                                             "review_sha256": review_sha, "fresh_idle_sha256": idle_sha,
                                             "argv": argv})
    terminal = {"schema": "nosync-nsenter-terminal-v1", "run_id": RUN_ID, "cid": now["cid"],
                "init_pid": now["init_pid"], "init_starttick": now["init_starttick"],
                "preflight_sha256": preflight_sha, "review_sha256": review_sha,
                "state": "unknown", "automatic_retry": False, "automatic_cleanup": False}
    try:
        with (RUN / "execute.stdout").open("xb") as out, (RUN / "execute.stderr").open("xb") as err:
            child = subprocess.Popen(argv, stdout=out, stderr=err, start_new_session=True,
                                     preexec_fn=lambda: library["cg_join"](now["target_cgroups"]))
            terminal.update({"entry_pid": child.pid, "entry_starttick": library["starttick"](child.pid)})
            write_new(RUN / "execute-process.json", {"run_id": RUN_ID, "cid": now["cid"],
                                                      "entry_pid": child.pid, "entry_starttick": terminal["entry_starttick"]})
            deadline = time.monotonic() + TIMEOUT_SECONDS
            while child.poll() is None and time.monotonic() < deadline:
                entry = PRIVATE / "entry-evidence.json"
                ack = PRIVATE / "entry-ack.json"
                if entry.is_file() and not ack.exists():
                    found = library["scan_workers"](now["expected_entry_namespaces"]["pid"],
                                                     now["target_cgroups"], TAG)
                    require(len(found) <= 1, "ambiguous entry process")
                    if found:
                        require(descendant(found[0]["host_pid"], child.pid), "entry is not the new child")
                        require(library["starttick"](found[0]["host_pid"]) == found[0]["starttick"],
                                "entry identity changed")
                        write_new(ack, found[0])
                        terminal["entry_proof"] = found[0]
                time.sleep(0.1)
            code = child.poll()
            if code is not None:
                terminal.update({"state": "exited", "child_exit_code": child.wait()})
    except Exception as exc:
        terminal["exception"] = repr(exc)
    value = write_new(RUN / "execute-terminal.json", terminal)
    print(json.dumps({"terminal": terminal, "terminal_sha256": value}, sort_keys=True))
    require(terminal["state"] == "exited", "execution outcome unknown; do not retry or remove")


def alive(library, pid, tick):
    try:
        return library["starttick"](pid) == tick
    except (FileNotFoundError, ProcessLookupError):
        return False


def closed_readback(library, terminal_sha):
    terminal = json.loads(read_pinned(RUN / "execute-terminal.json", terminal_sha))
    require(terminal.get("run_id") == RUN_ID and terminal.get("state") == "exited",
            "execution not known closed")
    require(not alive(library, terminal["entry_pid"], terminal["entry_starttick"]), "entry process still alive")
    current = snapshot(library, terminal["cid"], terminal["init_pid"], terminal["init_starttick"])
    groups = current["target_cgroups"]
    pidns = current["expected_entry_namespaces"]["pid"]
    require(not library["scan_workers"](pidns, groups, TAG), "diagnostic process still alive")
    names = ["create-receipt.json", "create-idle.json", "preflight.json", "root-review.json",
             "execute-once.json", "execute-idle.json", "execute-start.json", "execute-terminal.json",
             "execute.stdout", "execute.stderr", "execute-process.json", "private/entry-evidence.json",
             "private/entry-ack.json"]
    names.extend(p.relative_to(RUN).as_posix() for p in sorted(OUTPUT.iterdir()) if p.is_file() and not p.is_symlink())
    for stage in ("create", "execute"):
        folder = RUN / (stage + "-idle-raw")
        if folder.is_dir() and not folder.is_symlink():
            names.extend(path.relative_to(RUN).as_posix() for path in sorted(folder.iterdir()))
    # Include every local review/observation JSON, including explicitly named versions.
    names.extend(p.relative_to(RUN).as_posix() for p in sorted(RUN.glob("*.json")))
    # Preserve raw SMI from successful and failed observe/release attempts.
    for pattern in ("smi-observe-*", "smi-release-*"):
        for folder in sorted(RUN.glob(pattern)):
            require(folder.is_dir() and not folder.is_symlink(), "unsafe SMI capture directory")
            names.extend(p.relative_to(RUN).as_posix() for p in sorted(folder.iterdir()))
    names = list(dict.fromkeys(names))
    entries = []
    for name in names:
        path = RUN / name
        if path.is_file() and not path.is_symlink():
            entries.append({"path": name, "size": path.stat().st_size, "sha256": sha_file(path)})
        else:
            entries.append({"path": name, "missing": True})
    report = {"schema": "nosync-nsenter-closed-readback-v1", "run_id": RUN_ID, "cid": terminal["cid"],
              "terminal_sha256": terminal_sha, "child_exit_code": terminal["child_exit_code"],
              "files": entries, "scope": "process closed; result content still requires independent review"}
    value = write_new(RUN / "closed-readback.json", report)
    print(json.dumps({"readback_sha256": value, "report": report}, sort_keys=True))


def stop_remove(library, readback_sha):
    readback = json.loads(read_pinned(RUN / "closed-readback.json", readback_sha))
    require(readback.get("run_id") == RUN_ID, "readback identity differs")
    terminal = json.loads(read_pinned(RUN / "execute-terminal.json", readback["terminal_sha256"]))
    require(terminal["state"] == "exited" and not alive(library, terminal["entry_pid"], terminal["entry_starttick"]),
            "entry not closed")
    entries = {entry["path"]: entry for entry in readback["files"]}
    require("sha256" in entries.get("create-receipt.json", {}), "create receipt not in closed readback")
    create_receipt = json.loads(read_pinned(RUN / "create-receipt.json", entries["create-receipt.json"]["sha256"]))
    prior = create_receipt["container"]
    require(prior["cid"] == readback["cid"] == terminal["cid"], "container id differs")
    current = snapshot(library, prior["cid"], prior["init_pid"], prior["init_starttick"])
    require(current["cid"] == prior["cid"], "container changed")
    require(not library["scan_workers"](current["expected_entry_namespaces"]["pid"],
                                         current["target_cgroups"], TAG), "diagnostic process still alive")
    write_new(RUN / "stop-remove-attempt.json", {"run_id": RUN_ID, "cid": prior["cid"],
                                                  "readback_sha256": readback_sha})
    command(["docker", "stop", prior["cid"]])
    command(["docker", "rm", prior["cid"]])
    value = write_new(RUN / "stop-remove-complete.json", {"run_id": RUN_ID, "cid": prior["cid"],
                                                           "readback_sha256": readback_sha})
    print(json.dumps({"cleanup_sha256": value, "cid": prior["cid"]}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("create", "preflight", "execute", "closed-readback", "stop-remove"))
    parser.add_argument("--arm", choices=("green",), required=True)
    parser.add_argument("--pins-sha256", required=True)
    parser.add_argument("--create-sha256")
    parser.add_argument("--preflight-sha256")
    parser.add_argument("--review-sha256")
    parser.add_argument("--terminal-sha256")
    parser.add_argument("--readback-sha256")
    args = parser.parse_args()
    set_arm(args.arm)
    pins_sha = hash_arg(args.pins_sha256)
    library = check_host(pins_sha)
    if args.action == "create":
        create(library, pins_sha)
    elif args.action == "preflight":
        preflight(library, pins_sha, hash_arg(args.create_sha256))
    elif args.action == "execute":
        with (BASE / "execute.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            other = BASE / "runs" / ("green" if ARM == "red" else "red")
            if (other / "execute-once.json").exists():
                require((other / "closed-readback.json").is_file(), "other arm not known closed")
                other_readback = json.loads((other / "closed-readback.json").read_text())
                other_terminal = json.loads(read_pinned(other / "execute-terminal.json", other_readback["terminal_sha256"]))
                require(other_terminal.get("state") == "exited"
                        and not alive(library, other_terminal["entry_pid"], other_terminal["entry_starttick"]),
                        "other arm still active or unknown")
                other_create = json.loads((other / "create-receipt.json").read_text())
                c = other_create["container"]
                removed_path = other / "stop-remove-complete.json"
                if removed_path.is_file():
                    removed = json.loads(removed_path.read_text())
                    require(removed.get("cid") == c["cid"] and not command(
                        ["docker", "ps", "-a", "--no-trunc", "--filter", "id=" + c["cid"], "--format", "{{.ID}}"]
                    ).strip(), "other arm cleanup not confirmed")
                else:
                    for ctrl in ("devices", "memory"):
                        folder = cgroup_folder(ctrl, c["target_cgroups"][ctrl], c["cid"])
                        require({int(v) for v in (folder / "cgroup.procs").read_text().split()} == {c["init_pid"]},
                                "other arm cgroup not idle")
            execute(library, pins_sha, hash_arg(args.preflight_sha256), hash_arg(args.review_sha256))
    elif args.action == "closed-readback":
        closed_readback(library, hash_arg(args.terminal_sha256))
    else:
        stop_remove(library, hash_arg(args.readback_sha256))


if __name__ == "__main__":
    main()