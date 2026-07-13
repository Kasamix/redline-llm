#!/usr/bin/env python3
"""Create a RunPod GPU pod for benchmark runs - optionally provision it end to end.

REST API: https://rest.runpod.io/v1 (POST /pods, GET /pods/{id}). The API key
is read from $REDLINE_SECRETS_DIR (runpod_api_key.txt, falling back to
runpod.key) or $RUNPOD_API_KEY - keys are never hardcoded and never live in
this repository. The SSH public key is injected through the pod's PUBLIC_KEY
environment variable as a fallback for an account-level RunPod SSH key.

Usage:
    REDLINE_SECRETS_DIR=/path/to/.secrets python scripts/runpod/create_pod.py \\
        --gpu RTX4090 --wait --setup

    --wait   poll until the pod is RUNNING and its SSH port answers
    --setup  rsync this repository to /workspace/redline-llm and run
             scripts/runpod/pod_setup.sh (deps, model, pod-release build,
             ctest) - the last line printed on success is READY
    --dry-run  print the create request body and exit (no API call)

Everything is stdlib-only so the script runs on a bare python3.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API_BASE = "https://rest.runpod.io/v1"
KEY_FILENAMES = ("runpod_api_key.txt", "runpod.key")

# CUDA 12.9 / torch 2.9.1 / Ubuntu 24.04 base image (pinned; matches the
# toolchain the sm_89 build and benchmarks are validated against).
DEFAULT_IMAGE = "runpod/pytorch:1.0.7-rc.138-cu1290-torch291-ubuntu2404"

GPU_ALIASES = {
    "RTX4090": "NVIDIA GeForce RTX 4090",
    "RTX3090": "NVIDIA GeForce RTX 3090",
    "A5000": "NVIDIA RTX A5000",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
REMOTE_REPO = "/workspace/redline-llm"
RSYNC_EXCLUDES = (".git", "models", "build", "__pycache__", ".pytest_cache")


def load_api_key() -> str:
    """Read the RunPod API key from $RUNPOD_API_KEY or $REDLINE_SECRETS_DIR."""
    env_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if env_key:
        return env_key
    secrets_dir = os.environ.get("REDLINE_SECRETS_DIR")
    if not secrets_dir:
        raise SystemExit(
            "REDLINE_SECRETS_DIR is not set (and no RUNPOD_API_KEY); refusing to guess a key location."
        )
    tried = []
    for name in KEY_FILENAMES:
        key_path = os.path.join(secrets_dir, name)
        tried.append(key_path)
        try:
            with open(key_path, encoding="utf-8") as f:
                key = f.read().strip()
        except FileNotFoundError:
            continue
        if not key:
            raise SystemExit(f"empty API key file: {key_path}")
        return key
    raise SystemExit("missing API key file; tried: " + ", ".join(tried))


def api(method: str, path: str, api_key: str, body: dict | None = None,
        retries: int = 3) -> dict | list:
    """One REST call with small retry on transient failures."""
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    last_err: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                text = resp.read().decode() or "{}"
                return json.loads(text)
        except urllib.error.HTTPError as e:  # 4xx: no point retrying
            detail = e.read().decode(errors="replace")
            if 400 <= e.code < 500:
                raise SystemExit(f"{method} {path} -> HTTP {e.code}: {detail}")
            last_err = RuntimeError(f"HTTP {e.code}: {detail}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
        time.sleep(2 * (attempt + 1))
    raise SystemExit(f"{method} {path} failed after {retries} attempts: {last_err}")


def resolve_pubkey(args: argparse.Namespace) -> str | None:
    """Public key text for the PUBLIC_KEY env fallback (None -> rely on account key)."""
    candidates = []
    if args.pubkey:
        candidates.append(Path(args.pubkey))
    else:
        if args.ssh_key:
            candidates.append(Path(args.ssh_key + ".pub"))
        secrets_dir = os.environ.get("REDLINE_SECRETS_DIR")
        if secrets_dir:
            candidates.append(Path(secrets_dir) / "id_ed25519.pub")
        candidates.append(Path.home() / ".ssh" / "id_ed25519.pub")
    for path in candidates:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    return None


def build_request_body(args: argparse.Namespace, pubkey: str | None) -> dict:
    body = {
        "name": args.name,
        "imageName": args.image,
        "cloudType": args.cloud_type,
        "gpuTypeIds": [GPU_ALIASES.get(args.gpu, args.gpu)],
        "gpuCount": 1,
        "containerDiskInGb": args.disk_gb,
        "volumeInGb": args.volume_gb,
        "ports": ["22/tcp"],
    }
    if pubkey:
        body["env"] = {"PUBLIC_KEY": pubkey}
    return body


def ssh_endpoint(pod: dict) -> tuple[str, int] | None:
    """(ip, port) once the pod exposes its mapped SSH port, else None."""
    ip = pod.get("publicIp")
    mappings = pod.get("portMappings") or {}
    port = mappings.get("22")
    if ip and port:
        return ip, int(port)
    return None


def wait_for_ssh(api_key: str, pod_id: str, timeout_s: int) -> tuple[dict, str, int]:
    """Poll the pod until RUNNING and the SSH TCP port accepts connections."""
    deadline = time.time() + timeout_s
    pod: dict = {}
    while time.time() < deadline:
        pod = api("GET", f"/pods/{pod_id}", api_key)  # type: ignore[assignment]
        status = pod.get("desiredStatus")
        endpoint = ssh_endpoint(pod)
        print(f"  pod {pod_id}: status={status} ssh={endpoint or 'pending'}", flush=True)
        if status == "RUNNING" and endpoint:
            ip, port = endpoint
            try:
                with socket.create_connection((ip, port), timeout=10):
                    print(f"  ssh port open at {ip}:{port}")
                    return pod, ip, port
            except OSError:
                pass  # port mapped but sshd not up yet
        time.sleep(10)
    raise SystemExit(f"timed out after {timeout_s}s waiting for pod {pod_id} SSH")


def ssh_base(args: argparse.Namespace, port: int) -> list[str]:
    cmd = ["ssh", "-p", str(port), "-o", "BatchMode=yes",
           "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15"]
    if args.ssh_key:
        cmd += ["-i", args.ssh_key]
    return cmd


def run_setup(args: argparse.Namespace, ip: str, port: int) -> None:
    """rsync the repo, then run pod_setup.sh remotely (streams its output)."""
    ssh_cmd = " ".join(ssh_base(args, port))
    rsync = ["rsync", "-az", "--delete", "-e", ssh_cmd]
    for pattern in RSYNC_EXCLUDES:
        rsync += ["--exclude", pattern]
    rsync += [f"{REPO_ROOT}{os.sep}".replace(os.sep, "/"), f"root@{ip}:{REMOTE_REPO}/"]
    print(f"+ {' '.join(shlex.quote(c) for c in rsync)}", flush=True)
    subprocess.run(rsync, check=True)

    remote = f"bash {REMOTE_REPO}/scripts/runpod/pod_setup.sh"
    ssh_full = ssh_base(args, port) + [f"root@{ip}", remote]
    print(f"+ {' '.join(shlex.quote(c) for c in ssh_full)}", flush=True)
    proc = subprocess.run(ssh_full)
    if proc.returncode != 0:
        raise SystemExit(f"pod_setup.sh failed (exit {proc.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", default="redline-bench")
    parser.add_argument("--gpu", "--gpu-type", dest="gpu", default="RTX4090",
                        help="alias (RTX4090) or full RunPod gpuTypeId")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--cloud-type", default="SECURE", choices=["SECURE", "COMMUNITY"])
    parser.add_argument("--disk-gb", type=int, default=40, help="container disk GiB")
    parser.add_argument("--volume-gb", type=int, default=0, help="persistent volume GiB (0 = none)")
    parser.add_argument("--ssh-key", default=None,
                        help="private key for --setup ssh/rsync (default: "
                             "$REDLINE_SECRETS_DIR/id_ed25519 if present, else ssh defaults)")
    parser.add_argument("--pubkey", default=None,
                        help="public key file injected as PUBLIC_KEY env (fallback "
                             "for an account-level RunPod SSH key)")
    parser.add_argument("--wait", action="store_true", help="wait until RUNNING + SSH reachable")
    parser.add_argument("--setup", action="store_true",
                        help="after --wait: rsync repo + run pod_setup.sh (implies --wait)")
    parser.add_argument("--wait-timeout", type=int, default=900, help="seconds for --wait")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the create request body and exit without calling the API")
    args = parser.parse_args()
    if args.setup:
        args.wait = True

    if args.ssh_key is None:
        secrets_dir = os.environ.get("REDLINE_SECRETS_DIR")
        if secrets_dir and (Path(secrets_dir) / "id_ed25519").is_file():
            args.ssh_key = str(Path(secrets_dir) / "id_ed25519")

    pubkey = resolve_pubkey(args)
    body = build_request_body(args, pubkey)

    if args.dry_run:
        print("DRY RUN - POST /pods request body:")
        print(json.dumps(body, indent=2))
        print("DRY RUN - no API call made.")
        return

    api_key = load_api_key()
    if not pubkey:
        print("warning: no SSH public key found for PUBLIC_KEY injection; "
              "relying on the account-level RunPod SSH key.", file=sys.stderr)

    pod = api("POST", "/pods", api_key, body)  # type: ignore[assignment]
    pod_id = pod["id"]
    cost = pod.get("costPerHr", "?")
    print(f"created pod {pod_id} ({body['gpuTypeIds'][0]}, {args.cloud_type}, ${cost}/hr)")

    if not args.wait:
        print("hint:  re-run with --wait --setup for unattended provisioning")
        print(f"stop:  python scripts/runpod/stop_pod.py {pod_id}")
        return

    pod, ip, port = wait_for_ssh(api_key, pod_id, args.wait_timeout)
    key_flag = f" -i {args.ssh_key}" if args.ssh_key else ""
    print(f"ssh command: ssh{key_flag} -p {port} root@{ip}")

    if args.setup:
        run_setup(args, ip, port)

    print(f"pod {pod_id} at ${cost}/hr - stop it when idle "
          f"(python scripts/runpod/stop_pod.py {pod_id}) and record the "
          f"pod-hours + cost in your ops spend log.")
    print("READY")


if __name__ == "__main__":
    main()
