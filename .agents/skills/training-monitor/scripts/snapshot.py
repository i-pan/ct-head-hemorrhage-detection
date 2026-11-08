#!/usr/bin/env python3
"""Read-only Linux training snapshot. No ML dependencies, model loads, or job changes."""

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import time


def command(args, timeout=8):
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
        if result.returncode:
            return (
                None,
                result.stderr.strip()
                or result.stdout.strip()
                or f"exit {result.returncode}",
            )
        return result.stdout, None
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, str(error)


def kb_fields(path):
    fields = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition(":")
        bits = value.split()
        if len(bits) == 2 and bits[1] == "kB":
            fields[key] = int(bits[0]) * 1024
    return fields


def system_memory():
    try:
        m = kb_fields(Path("/proc/meminfo"))
        total, available = m["MemTotal"], m.get("MemAvailable")
        return dict(
            total_bytes=total,
            available_bytes=available,
            used_bytes=total - available if available is not None else None,
            swap_total_bytes=m.get("SwapTotal"),
            swap_used_bytes=m["SwapTotal"] - m["SwapFree"]
            if "SwapTotal" in m and "SwapFree" in m
            else None,
            used_definition="MemTotal - MemAvailable; includes all host workloads",
        )
    except (OSError, KeyError, ValueError) as error:
        return dict(unavailable=str(error))


def process_tree(roots):
    entries = {}
    for path in Path("/proc").glob("[0-9]*"):
        try:
            stat = (path / "stat").read_text()
            end = stat.rindex(")")
            tail = stat[end + 2 :].split()
            entries[int(path.name)] = dict(
                pid=int(path.name),
                ppid=int(tail[1]),
                name=stat[stat.index("(") + 1 : end],
                state=tail[0],
            )
        except (OSError, ValueError, IndexError):
            continue
    selected = set(roots) & entries.keys()
    while True:
        descendants = {pid for pid, row in entries.items() if row["ppid"] in selected}
        enlarged = selected | descendants
        if enlarged == selected:
            break
        selected = enlarged
    rows = []
    for pid in sorted(selected):
        row = entries[pid].copy()
        try:
            row["rss_bytes"] = kb_fields(Path(f"/proc/{pid}/status")).get("VmRSS")
        except (OSError, ValueError):
            row["rss_bytes"] = None
        try:
            row["pss_bytes"] = kb_fields(Path(f"/proc/{pid}/smaps_rollup")).get("Pss")
        except (OSError, ValueError):
            row["pss_bytes"] = None
        rows.append(row)
    summary = dict(
        requested_root_pids=roots,
        missing_root_pids=sorted(set(roots) - entries.keys()),
        processes=rows,
        process_count=len(rows),
    )
    for field in ["pss_bytes", "rss_bytes"]:
        known = [r[field] for r in rows if r[field] is not None]
        summary[f"summed_{field}"] = sum(known) if known else None
        summary[f"{field}_coverage"] = f"{len(known)}/{len(rows)} processes"
    summary["memory_note"] = (
        "PSS apportions shared pages; RSS sums double-count shared pages. Totals include only readable live processes."
    )
    return summary, selected


def number(value):
    try:
        return float(value.strip())
    except ValueError:
        return None


def gpu_snapshot(job_pids, fast=False):
    fields = ["index", "uuid", "name", "memory.used", "memory.total", "utilization.gpu"]
    output, error = command(
        [
            "nvidia-smi",
            "--query-gpu=" + ",".join(fields),
            "--format=csv,noheader,nounits",
        ],
        timeout=3 if fast else 8,
    )
    if error:
        return dict(unavailable=error)
    devices = []
    for row in csv.reader(io.StringIO(output)):
        if len(row) != len(fields):
            continue
        devices.append(
            dict(
                index=int(row[0]),
                uuid=row[1].strip(),
                name=row[2].strip(),
                used_mib=number(row[3]),
                total_mib=number(row[4]),
                utilization_percent=number(row[5]),
            )
        )
    if fast:
        return dict(
            devices=devices,
            process_attribution="omitted in fast mode",
            note="Device-wide current allocations/utilization; may include other jobs. Not peak usage.",
        )
    output, error = command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    processes = []
    if not error:
        for row in csv.reader(io.StringIO(output)):
            if len(row) == 3 and row[1].strip().isdigit():
                pid = int(row[1])
                processes.append(
                    dict(
                        gpu_uuid=row[0].strip(),
                        pid=pid,
                        used_mib=number(row[2]),
                        belongs_to_target_tree=pid in job_pids,
                    )
                )
    return dict(
        devices=devices,
        compute_processes=processes,
        process_query_error=error,
        note="Current driver allocations, not peak memory or tensor-only memory; device totals may include other jobs and graphics.",
    )


def log_tail(path, lines, max_bytes):
    try:
        path = Path(path)
        # A bounded tail avoids loading multi-GB training logs.
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            start = max(0, size - max_bytes)
            stream.seek(start)
            data = stream.read(max_bytes)
        normalized = (
            data.decode("utf-8", errors="replace").replace("\r", "\n").splitlines()
        )
        if start and normalized:
            normalized = normalized[1:]  # initial record can be partial
        normalized = [line for line in normalized if line.strip()]
        stat = path.stat()
        return dict(
            path=str(path.resolve()),
            size_bytes=size,
            modified_utc=datetime.fromtimestamp(
                stat.st_mtime, timezone.utc
            ).isoformat(),
            age_seconds=max(0, time.time() - stat.st_mtime),
            tail="\n".join(normalized[-lines:]),
            tail_truncated=start > 0 or len(normalized) > lines,
        )
    except OSError as error:
        return dict(path=str(path), unavailable=str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip process-tree/PSS and GPU process scans; default to a short bounded log tail",
    )
    parser.add_argument("--pid", type=int, action="append", default=[])
    parser.add_argument(
        "--service",
        help="systemd user service whose MainPID anchors the job process tree",
    )
    parser.add_argument("--log", action="append", default=[])
    parser.add_argument("--tail-lines", type=int, default=None)
    parser.add_argument("--max-log-bytes", type=int, default=None)
    args = parser.parse_args()
    args.tail_lines = (
        args.tail_lines if args.tail_lines is not None else (8 if args.fast else 100)
    )
    args.max_log_bytes = (
        args.max_log_bytes
        if args.max_log_bytes is not None
        else (16384 if args.fast else 262144)
    )
    if args.tail_lines <= 0 or args.max_log_bytes <= 0 or any(p <= 0 for p in args.pid):
        parser.error("PID, tail-lines, and max-log-bytes must be positive")
    roots = list(args.pid)
    result = dict(
        snapshot_utc=datetime.now(timezone.utc).isoformat(),
        mode="fast" if args.fast else "full",
    )
    if args.service:
        output, error = command(
            [
                "systemctl",
                "--user",
                "show",
                args.service,
                "-p",
                "MainPID",
                "-p",
                "ActiveState",
                "-p",
                "SubState",
                "-p",
                "ExecMainStartTimestamp",
            ],
            timeout=3 if args.fast else 8,
        )
        service = dict(name=args.service, error=error)
        if output:
            service.update(
                line.split("=", 1) for line in output.splitlines() if "=" in line
            )
            if service.get("MainPID", "0").isdigit() and int(service["MainPID"]) > 0:
                roots.append(int(service["MainPID"]))
        result["service"] = service
    result["system_ram"] = system_memory()
    if args.fast:
        pids = set(roots)
        result["job"] = dict(
            requested_root_pids=sorted(pids),
            root_pid_exists={str(p): Path(f"/proc/{p}").exists() for p in sorted(pids)},
            memory_status="omitted in fast mode; no process-tree or PSS scan",
        )
    else:
        result["job"], pids = process_tree(sorted(set(roots)))
    result["gpu"] = gpu_snapshot(pids, fast=args.fast)
    result["logs"] = [
        log_tail(p, args.tail_lines, args.max_log_bytes) for p in args.log
    ]
    result["snapshot_finished_utc"] = datetime.now(timezone.utc).isoformat()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
