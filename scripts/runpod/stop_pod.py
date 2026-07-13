#!/usr/bin/env python3
"""Stop - or with --terminate, delete - RunPod pods, printing their cost.

REST API: https://rest.runpod.io/v1 (GET /pods, POST /pods/{id}/stop,
DELETE /pods/{id}). The API key is read from $REDLINE_SECRETS_DIR
(runpod_api_key.txt, falling back to runpod.key) or $RUNPOD_API_KEY - never
hardcoded (same convention as create_pod.py; the loader is duplicated so each
script stays standalone).

Usage:
    REDLINE_SECRETS_DIR=/path/to/.secrets python scripts/runpod/stop_pod.py <pod_id> [--terminate]
    REDLINE_SECRETS_DIR=/path/to/.secrets python scripts/runpod/stop_pod.py --all [--terminate]
    ... --dry-run     # show what would be stopped + estimated cost, change nothing

A stopped pod keeps billing for disk until terminated; --terminate deletes the
pod so billing fully stops. The printed cost is estimated from the current
boot's uptime (lastStartedAt) x costPerHr - record the actual pod-hours and
dollars in your ops spend log after every session.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
import urllib.error
import urllib.request

API_BASE = "https://rest.runpod.io/v1"
KEY_FILENAMES = ("runpod_api_key.txt", "runpod.key")


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


def api(method: str, path: str, api_key: str, retries: int = 3):
    """One REST call; raises RuntimeError on failure (callers decide fatality)."""
    last_err: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(f"{API_BASE}{path}", method=method)
        req.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode() or "{}")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            if 400 <= e.code < 500:
                raise RuntimeError(f"{method} {path} -> HTTP {e.code}: {detail}") from None
            last_err = RuntimeError(f"HTTP {e.code}: {detail}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{method} {path} failed after {retries} attempts: {last_err}")


def parse_runpod_time(text: str | None) -> dt.datetime | None:
    """Parse '2026-07-09 21:53:08.572 +0000 UTC' (fractional part optional)."""
    if not text:
        return None
    cleaned = text.removesuffix(" UTC").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f %z", "%Y-%m-%d %H:%M:%S %z"):
        try:
            return dt.datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def uptime_hours(pod: dict) -> float | None:
    started = parse_runpod_time(pod.get("lastStartedAt"))
    if started is None:
        return None
    return max(0.0, (dt.datetime.now(dt.timezone.utc) - started).total_seconds() / 3600.0)


def describe(pod: dict) -> tuple[str, float | None]:
    """One status line + estimated cost for this boot (None when unknown)."""
    cost_hr = pod.get("costPerHr")
    hours = uptime_hours(pod)
    est = None
    if isinstance(cost_hr, (int, float)) and hours is not None:
        est = cost_hr * hours
    line = (f"pod {pod.get('id')} ({pod.get('name', '?')}): "
            f"{pod.get('desiredStatus', '?')}, ${cost_hr}/hr, "
            f"up {hours:.2f} h this boot" if hours is not None else
            f"pod {pod.get('id')} ({pod.get('name', '?')}): "
            f"{pod.get('desiredStatus', '?')}, ${cost_hr}/hr, uptime unknown")
    if est is not None:
        line += f" -> est ${est:.2f}"
    return line, est


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pod_id", nargs="?", help="pod to stop (omit with --all)")
    parser.add_argument("--all", action="store_true", help="stop every pod on the account")
    parser.add_argument(
        "--terminate",
        action="store_true",
        help="delete the pod (and its disk) after stopping so billing fully stops",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be stopped and the estimated cost; change nothing")
    args = parser.parse_args()

    if bool(args.pod_id) == args.all:
        parser.error("give exactly one of <pod_id> or --all")

    api_key = load_api_key()
    pods = api("GET", "/pods", api_key)
    if not isinstance(pods, list):
        raise SystemExit(f"unexpected GET /pods response: {pods!r}")

    if args.all:
        targets = pods
    else:
        targets = [p for p in pods if p.get("id") == args.pod_id]
        if not targets:
            print(f"pod {args.pod_id} not found; account has "
                  f"{len(pods)} pod(s): {[p.get('id') for p in pods]}")
            return 1

    if not targets:
        print("no pods on the account - nothing to stop.")
        return 0

    verb = "terminate" if args.terminate else "stop"
    total_est = 0.0
    any_est = False
    failures = 0
    for pod in targets:
        line, est = describe(pod)
        print(line)
        if est is not None:
            total_est += est
            any_est = True
        if args.dry_run:
            print(f"  DRY RUN - would {verb}")
            continue
        try:
            if pod.get("desiredStatus") == "RUNNING":
                api("POST", f"/pods/{pod['id']}/stop", api_key)
                print("  stopped")
            else:
                print(f"  not running (status {pod.get('desiredStatus')}) - skip stop")
            if args.terminate:
                api("DELETE", f"/pods/{pod['id']}", api_key)
                print("  terminated (deleted)")
        except RuntimeError as e:
            failures += 1
            print(f"  FAILED: {e}")

    if any_est:
        print(f"total estimated cost this boot: ${total_est:.2f} "
              f"({len(targets)} pod{'s' if len(targets) != 1 else ''})")
    if not args.terminate and not args.dry_run:
        print("note: stopped pods keep billing for disk - use --terminate to delete.")
    print("record the session's actual pod-hours and dollars in your ops spend log.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
