# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Read the retained checkpoint in one bounded CPU-only container on 201."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Optional


ROOT = Path("/opt/xxl-jt-dev-8npu-20261014")
OWNER = "01a13aa4-a553-7e22-a1c1-ebd16b083bb9"
IMAGE = "sha256:5f9e677f439d11ab4e7c888f54b142335a27eee7411cbd2ce894a4798a70146d"
SOURCE = "544526f7db8342c607244fd41009b2a74adb84c8"
CASE = "201-a1e6-checkpoint-read-v1"
TRAINING_CASE = "r201-a1e6-fsdp8-0-200-v1"
INPUTS = ROOT / "evidence" / (CASE + "-inputs-v2")
OUTPUT = ROOT / "evidence" / CASE
TIMEOUT_SECONDS = 600


def fingerprint(path: Path) -> str:
    """Return a bounded-read SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_owned_file(path: Path, digest: str, size: Optional[int] = None) -> None:
    """Reject redirected or changed files inside this task's root."""
    if ROOT not in path.parents or path.resolve() != path or not path.is_file():
        raise ValueError("Unexpected input path: " + str(path))
    if size is not None and path.stat().st_size != size:
        raise ValueError("Input size changed: " + str(path))
    if fingerprint(path) != digest:
        raise ValueError("Input hash changed: " + str(path))


def preflight() -> dict:
    """Verify ownership, immutable inputs, source and the completed training run."""
    if ROOT.resolve() != ROOT or Path(__file__).resolve() != ROOT / "tools" / Path(__file__).name:
        raise ValueError("Unexpected runner path")
    if json.loads((ROOT / "owner.json").read_bytes())["owner"] != OWNER:
        raise ValueError("Task owner changed")
    if OUTPUT.exists():
        raise ValueError("Refusing to replace checkpoint read evidence")
    if find_owned_container() is not None:
        raise ValueError("The exact diagnostic container already exists")
    inputs = json.loads((INPUTS / "input-manifest.json").read_bytes())
    if set(inputs["files"]) != {"read_checkpoint_cpu.py", "expected-hashes.json", "run-manifest.json"}:
        raise ValueError("Unexpected checkpoint audit input set")
    for name, item in inputs["files"].items():
        verify_owned_file(INPUTS / name, item["sha256"], item["bytes"])
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / "repo", text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT / "repo")
    if source != SOURCE or dirty:
        raise ValueError("Source is not the clean reviewed checkout")
    manifest = json.loads((INPUTS / "run-manifest.json").read_bytes())
    for item in manifest["files"]:
        verify_owned_file(ROOT / item["path"], item["sha256"], item["bytes"])
    receipt = json.loads((ROOT / "scheduling" / (TRAINING_CASE + ".receipt.json")).read_bytes())
    if receipt["source"] != SOURCE or receipt.get("released") is not True or receipt.get("exit_code") != 0:
        raise ValueError("Checkpoint source run is not completed and released")
    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes < 1024**3:
        raise ValueError("Less than 1 GiB free for bounded diagnostic output")
    return {"source": source, "inputs": inputs, "run_files_verified": len(manifest["files"]),
            "runner_sha256": fingerprint(Path(__file__)), "free_bytes_before": free_bytes,
            "scope": "CPU checkpoint readability only; no training or cross-host precision acceptance"}


def inspect_owned(container_id: str) -> dict:
    """Inspect the exact owned CPU diagnostic container before cleanup."""
    info = json.loads(subprocess.check_output(["docker", "inspect", container_id], timeout=20))[0]
    labels = info["Config"]["Labels"]
    if (info["Id"] != container_id or labels.get("codex.owner") != OWNER
            or labels.get("codex.case") != CASE or info["Name"] != "/jt-cpu-" + CASE
            or info["Image"] != IMAGE):
        raise ValueError("Container identity or ownership changed")
    return info


def find_owned_container() -> Optional[str]:
    """Reconcile an uncertain create response using the exact name and all ownership fields."""
    result = subprocess.check_output(["docker", "ps", "--all", "--quiet", "--no-trunc",
                                      "--filter", "name=^/jt-cpu-" + CASE + "$"], text=True, timeout=20).split()
    if not result:
        return None
    if len(result) != 1:
        raise ValueError("Ambiguous exact diagnostic container name")
    inspect_owned(result[0])
    return result[0]


def cleanup_container(container_id: Optional[str], receipt: dict) -> None:
    """Remove only a verified owned container, retaining uncertainty when creation cannot be resolved."""
    if container_id is None:
        container_id = find_owned_container()
        if container_id is None:
            receipt.update(removed_verified=None, creation_outcome_unknown=True,
                           cleanup_observation="No matching container observed after failed create; recheck required")
            return
        receipt.update(container_id=container_id, reconciled_after_create_error=True)
    info = inspect_owned(container_id)
    if info["State"]["Running"]:
        subprocess.run(["docker", "kill", "--signal", "KILL", container_id], check=True,
                       stdout=subprocess.DEVNULL, timeout=15)
    receipt["final_container_state"] = inspect_owned(container_id)["State"]
    subprocess.run(["docker", "rm", container_id], check=True, stdout=subprocess.DEVNULL, timeout=30)
    remaining = subprocess.check_output(["docker", "ps", "--all", "--quiet", "--no-trunc",
                                         "--filter", "id=" + container_id], text=True, timeout=20).strip()
    receipt["removed_verified"] = not remaining


def validate_container(info: dict) -> dict:
    """Verify CPU isolation and that only the output bind mount is writable."""
    host = info["HostConfig"]
    exact = {"Runtime": "runc", "Privileged": False, "NetworkMode": "none", "Memory": 8 * 1024**3,
             "NanoCpus": 4 * 1000**3, "PidsLimit": 256, "ReadonlyRootfs": True, "PidMode": ""}
    for key, value in exact.items():
        if host.get(key) != value:
            raise ValueError("Unexpected container setting: " + key)
    for key in ("Devices", "DeviceRequests", "DeviceCgroupRules"):
        if host.get(key):
            raise ValueError("Unexpected device exposure: " + key)
    if info["Image"] != IMAGE:
        raise ValueError("Image mismatch")
    expected = {"/work": (str(ROOT), False),
                "/workspace/hyper-parallel": (str(ROOT / "repo"), False),
                "/result": (str(OUTPUT), True)}
    binds = {item["Destination"]: (item["Source"], item["RW"]) for item in info["Mounts"]
             if item["Type"] == "bind"}
    if binds != expected or any(item["Type"] not in ("bind", "tmpfs") for item in info["Mounts"]):
        raise ValueError("Unexpected container mount")
    if host.get("Tmpfs") != {"/tmp": "rw,nosuid,nodev,size=1g"}:
        raise ValueError("Unexpected temporary filesystem")
    return {"host_config": {key: host.get(key) for key in list(exact) +
                            ["Devices", "DeviceRequests", "DeviceCgroupRules", "Tmpfs"]},
            "mounts": info["Mounts"]}


def main() -> None:
    """Run once with a strict deadline, retaining diagnostics and removing its container."""
    receipt = {"case": CASE, "owner": OWNER, "started_epoch": time.time(), "preflight": preflight(),
               "timeout_seconds": TIMEOUT_SECONDS, "status": "STARTING", "npu_started": False,
               "migration_admitted": False}
    OUTPUT.mkdir(mode=0o700)
    in_container = "/work/evidence/" + INPUTS.name
    command_text = (
        "source /workspace/env.sh && export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && "
        "nice -n 10 /work/venvs/validation/bin/python " + in_container + "/read_checkpoint_cpu.py "
        "--checkpoint /work/checkpoints/" + TRAINING_CASE + "/global_step_200 "
        "--source /workspace/hyper-parallel --expected-hashes " + in_container + "/expected-hashes.json "
        "--metrics-dir /work/runs/" + TRAINING_CASE + " --output /result/checkpoint-cpu-read.json "
        "--max-slice-mib 32"
    )
    command = ["docker", "create", "--name", "jt-cpu-" + CASE,
               "--label", "codex.owner=" + OWNER, "--label", "codex.case=" + CASE,
               "--runtime", "runc", "--network", "none", "--memory", "8g", "--cpus", "4",
               "--pids-limit", "256", "--read-only", "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g",
               "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "TORCH_DEVICE_BACKEND_AUTOLOAD=0",
               "--workdir", "/workspace/hyper-parallel", "--entrypoint", "/bin/bash",
               "--mount", "type=bind,source=" + str(ROOT) + ",target=/work,readonly",
               "--mount", "type=bind,source=" + str(ROOT / "repo") + ",target=/workspace/hyper-parallel,readonly",
               "--mount", "type=bind,source=" + str(OUTPUT) + ",target=/result",
               IMAGE, "-lc", command_text]
    container_id = None
    receipt.update(docker_create_command=command, phase="create")
    try:
        container_id = subprocess.check_output(command, text=True, timeout=30).strip()
        receipt.update(container_id=container_id, phase="validate")
        receipt["verified_container"] = validate_container(inspect_owned(container_id))
        receipt["phase"] = "start-and-read"
        with (OUTPUT / "stdout.log").open("xb") as stdout, (OUTPUT / "stderr.log").open("xb") as stderr:
            result = subprocess.run(["docker", "start", "--attach", container_id], stdout=stdout, stderr=stderr,
                                    timeout=TIMEOUT_SECONDS, check=False)
        info = inspect_owned(container_id)
        receipt.update(attach_exit_code=result.returncode, container_state=info["State"])
        receipt["status"] = "PASS" if result.returncode == 0 and info["State"]["ExitCode"] == 0 else "FAIL"
        result_path = OUTPUT / "checkpoint-cpu-read.json"
        if result_path.is_file():
            receipt["result_sha256"] = fingerprint(result_path)
        else:
            receipt["status"] = "FAIL_NO_RESULT"
    except subprocess.TimeoutExpired:
        receipt.update(status="TIMEOUT", attach_exit_code=124)
    except BaseException as error:
        receipt.update(status="ERROR", error=str(error))
    finally:
        try:
            cleanup_container(container_id, receipt)
        except BaseException as error:
            receipt.update(removed_verified=False, cleanup_error=str(error), status="CLEANUP_UNCONFIRMED")
        receipt["finished_epoch"] = time.time()
        with (OUTPUT / "receipt.json").open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps({key: receipt[key] for key in ["case", "status", "removed_verified", "finished_epoch"]}),
          flush=True)
    if receipt["status"] != "PASS" or not receipt["removed_verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
