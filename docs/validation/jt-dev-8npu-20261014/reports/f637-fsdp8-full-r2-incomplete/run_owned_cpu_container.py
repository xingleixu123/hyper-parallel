"""Run a bounded, device-free diagnostic container and remove only that container."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import time


ROOT = Path("/home/xxl/jt-dev-8npu-20261014")
OWNER = "01a13aa4-a553-7e22-a1c1-ebd16b083bb9"
IMAGE = "sha256:5f9e677f439d11ab4e7c888f54b142335a27eee7411cbd2ce894a4798a70146d"


parser = argparse.ArgumentParser()
parser.add_argument("case")
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
if not re.fullmatch(r"[a-z0-9-]{3,64}", args.case) or not args.command:
    raise ValueError("Expected a unique case name and a bounded CPU command")
log_path = ROOT / "evidence" / (args.case + ".container.log")
receipt_path = ROOT / "evidence" / (args.case + ".receipt.json")
if log_path.exists() or receipt_path.exists():
    raise ValueError("Refusing to replace existing diagnostic evidence")
command = ["docker", "create", "--name", "jt-cpu-" + args.case,
           "--label", "codex.owner=" + OWNER, "--label", "codex.case=" + args.case,
           "--network", "none", "--memory", "8g", "--cpus", "4", "--pids-limit", "256",
           "--entrypoint", "/bin/bash", "-w", "/work", "-v", str(ROOT) + ":/work",
           "-v", str(ROOT / "repo") + ":/workspace/hyper-parallel",
           "-v", "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro", IMAGE, *args.command]
container_id = subprocess.check_output(command, text=True).strip()
receipt = {"case": args.case, "owner": OWNER, "container_id": container_id,
           "devices": [], "image": IMAGE, "command": args.command, "started": time.time()}
try:
    inspect = json.loads(subprocess.check_output(["docker", "inspect", container_id]))[0]
    assert not inspect["HostConfig"]["Devices"]
    assert inspect["Config"]["Labels"]["codex.owner"] == OWNER
    source_mount = next(mount for mount in inspect["Mounts"] if mount["Destination"] == "/workspace/hyper-parallel")
    if Path(source_mount["Source"]).resolve() != (ROOT / "repo").resolve():
        raise RuntimeError("CPU diagnostic source bind does not match the task checkout")
    receipt["source_mount"] = source_mount
    receipt["source_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT / "repo", text=True).strip()
    with log_path.open("wb") as handle:
        result = subprocess.run(["docker", "start", "-a", container_id], stdout=handle,
                                stderr=subprocess.STDOUT, timeout=240, check=False)
    receipt["exit"] = result.returncode
except subprocess.TimeoutExpired:
    receipt["exit"] = 124
    receipt["reason"] = "cpu-container-timeout"
finally:
    inspect = json.loads(subprocess.check_output(["docker", "inspect", container_id]))[0]
    if inspect["Id"] != container_id or inspect["Config"]["Labels"].get("codex.owner") != OWNER:
        raise RuntimeError("Container ownership changed; refusing cleanup")
    if inspect["State"]["Running"]:
        subprocess.run(["docker", "stop", "--time", "10", container_id], check=True,
                       stdout=subprocess.DEVNULL, timeout=30)
    subprocess.run(["docker", "rm", container_id], check=True, stdout=subprocess.DEVNULL, timeout=30)
    receipt.update(ended=time.time(), removed=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
print(log_path.read_text(errors="replace")[-6000:])
raise SystemExit(receipt.get("exit", 1))
