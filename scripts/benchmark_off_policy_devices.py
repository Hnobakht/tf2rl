"""Measure all CPU/GPU simulation and learner combinations with equal SAC work."""

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def device_info():
    """Record the runtime and hardware used by the child training processes."""
    versions = {}
    for package in ("tensorflow", "gymnasium", "mujoco", "mujoco-warp", "warp-lang", "cpprb"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "gpu": query.stdout.strip(),
        "versions": versions,
        "python": sys.version,
    }


def summarize(metrics, warmup_windows):
    """Use completed work and synchronized timings, excluding initial report windows."""
    windows = metrics["iterations"][warmup_windows:]
    steps = sum(row["transitions"] for row in windows)
    updates = sum(row["updates"] for row in windows)
    collection = sum(row["collection_seconds"] for row in windows)
    training = sum(row["update_seconds"] for row in windows)
    seconds = sum(row["seconds"] for row in windows)
    return {
        "measured_transitions": steps,
        "measured_updates": updates,
        "collection_seconds": collection,
        "training_seconds": training,
        "total_seconds": seconds,
        "collection_samples_per_second": steps / collection,
        "updates_per_second": updates / training,
        "training_ms_per_update": 1000 * training / updates,
        "total_samples_per_second": steps / seconds,
        "training_fraction": training / seconds,
        "window_samples_per_second": [row["steps_per_second"] for row in windows],
        "window_throughput_cv": statistics.stdev(row["steps_per_second"] for row in windows)
        / statistics.mean(row["steps_per_second"] for row in windows)
        if len(windows) > 1
        else 0,
        "model_devices": metrics["model_devices"],
    }


def save_report(output, report):
    """Save incremental results so every completed combination remains reviewable."""
    rows = report["results"]
    for row in rows:
        baseline = next(
            (
                r
                for r in rows
                if r["num_envs"] == row["num_envs"]
                and r["simulation"] == "cpu"
                and r["learner"] == "cpu"
            ),
            None,
        )
        if baseline is not None:
            row["speedup_vs_cpu_cpu"] = (
                row["total_samples_per_second"] / baseline["total_samples_per_second"]
            )
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = [
        "num_envs",
        "simulation",
        "learner",
        "measured_transitions",
        "measured_updates",
        "collection_samples_per_second",
        "training_ms_per_update",
        "total_samples_per_second",
        "speedup_vs_cpu_cpu",
        "total_seconds",
    ]
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# SAC CPU/GPU device benchmark",
        "",
        json.dumps(report["configuration"]),
        "",
        (
            "Collection includes inference, transfers, simulation and replay insertion. "
            "Training includes replay sampling, transfers and optimizer updates. "
            "Startup, warmup windows, evaluation and checkpoint I/O are excluded."
        ),
        "",
        (
            "CPU simulation uses serial Gymnasium SyncVectorEnv; GPU uses MJWarp. "
            "These are throughput measurements, not convergence comparisons."
        ),
        "",
        (
            "| Worlds | Simulation | Learner | Collection samples/s | Train ms/update | "
            "Total samples/s | Speedup vs CPU/CPU |"
        ),
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['num_envs']} | {row['simulation']} | {row['learner']} | "
            f"{row['collection_samples_per_second']:,.0f} | "
            f"{row['training_ms_per_update']:.2f} | "
            f"{row['total_samples_per_second']:,.0f} | "
            f"{row.get('speedup_vs_cpu_cpu', 1):.2f}x |"
        )
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", default="HalfCheetah-v5")
    parser.add_argument("--num-envs", nargs="+", type=int, default=[64, 256, 1024])
    parser.add_argument("--window-transitions", type=int, default=10240)
    parser.add_argument("--warmup-windows", type=int, default=2)
    parser.add_argument("--measure-windows", type=int, default=3)
    parser.add_argument("--updates-per-transition", type=float, default=1 / 64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--units", nargs=2, type=int, default=[256, 256])
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--cpu-inter-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--absorbing", action="store_true")
    parser.add_argument("--xla", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(
            *args.num_envs,
            args.window_transitions,
            args.warmup_windows,
            args.measure_windows,
            args.batch_size,
            args.cpu_threads,
            args.cpu_inter_threads,
            *args.units,
        )
        < 1
    ):
        parser.error("Counts must be positive")
    if any(args.window_transitions % count for count in args.num_envs):
        parser.error("window-transitions must be divisible by every environment count")
    if args.window_transitions < args.batch_size or args.updates_per_transition <= 0:
        parser.error("A window must supply at least a minibatch and positive update credit")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "comparison.json").exists():
        parser.error("Choose a new output directory to preserve previous results")
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": device_info(),
        "configuration": config,
        "results": [],
    }
    total = args.window_transitions * (args.warmup_windows + args.measure_windows)
    # Isolated sequential processes avoid GPU allocator leftovers and concurrent contention.
    for count in args.num_envs:
        for backend, gpu in (("gymnasium", -1), ("gymnasium", 0), ("mjwarp", -1), ("mjwarp", 0)):
            simulation, learner = (
                ("gpu" if backend == "mjwarp" else "cpu"),
                ("cpu" if gpu < 0 else "gpu"),
            )
            label = f"n{count}_env_{simulation}_train_{learner}"
            run_output = args.output / label
            command = [
                sys.executable,
                "-m",
                "tf2rl.experiments.off_policy",
                "--algorithm",
                "sac",
                "--env-name",
                args.env_name,
                "--env-backend",
                backend,
                "--gpu",
                str(gpu),
                "--num-envs",
                str(count),
                "--env-gpu",
                "0",
                "--cpu-threads",
                str(args.cpu_threads),
                "--cpu-inter-threads",
                str(args.cpu_inter_threads),
                "--seed",
                str(args.seed),
                "--max-steps",
                str(total),
                "--collection-steps",
                str(args.window_transitions // count),
                "--batch-size",
                str(args.batch_size),
                "--units",
                *map(str, args.units),
                "--memory-capacity",
                str(max(65536, total)),
                "--n-warmup",
                "0",
                "--updates-per-transition",
                str(args.updates_per_transition),
                "--eval-episodes",
                "0",
                "--checkpoint-interval",
                str(total + 1),
                "--output",
                str(run_output),
            ]
            if args.absorbing:
                command.append("--absorbing")
            if args.xla:
                command.append("--xla")
            print(f"RUN {label}", flush=True)
            with (args.output / f"{label}.log").open("w") as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            if process.returncode:
                raise RuntimeError(f"Failed: {label}; see {args.output / (label + '.log')}")
            metrics = json.loads((run_output / "metrics.json").read_text())
            result = {
                "num_envs": count,
                "simulation": simulation,
                "learner": learner,
                "command": command,
                **summarize(metrics, args.warmup_windows),
            }
            expected_updates = (
                args.window_transitions * args.measure_windows * (args.updates_per_transition)
            )
            if result["measured_transitions"] != args.window_transitions * args.measure_windows:
                raise RuntimeError("Measured sample counts differ between combinations")
            if abs(result["measured_updates"] - expected_updates) > 1:
                raise RuntimeError("Measured optimizer work differs between combinations")
            expected_device = "GPU:0" if gpu >= 0 else "CPU:0"
            if any(
                not devices or any(expected_device not in device for device in devices)
                for devices in result["model_devices"].values()
            ):
                raise RuntimeError(f"Learner placement mismatch: {result['model_devices']}")
            report["results"].append(result)
            save_report(args.output, report)
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in result.items()
                        if key not in ("command", "model_devices")
                    }
                ),
                flush=True,
            )
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save_report(args.output, report)


if __name__ == "__main__":
    main()
