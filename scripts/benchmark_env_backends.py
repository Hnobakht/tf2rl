"""Compare vector backends using the same selected on-policy algorithm and update count."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--algorithm", choices=("ppo", "vpg"), default="ppo")
    parser.add_argument("--num-envs", nargs="+", type=int, default=[1, 64, 256, 1024])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/mjwarp/benchmark"))
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 4 if args.algorithm == "ppo" else 1
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for count in args.num_envs:
        horizon = count * args.rollout_steps
        batch = min(1024, horizon)
        if horizon % batch:
            parser.error("num-envs * rollout-steps must be divisible by min(1024, horizon)")
        results = {}
        for backend in ("gymnasium", "mjwarp"):
            output = args.output / f"{backend}_{count}"
            command = [
                sys.executable,
                "-m",
                "tf2rl.experiments.on_policy",
                "--algorithm",
                args.algorithm,
                "--env-name",
                args.env_name,
                "--env-backend",
                backend,
                "--gpu",
                str(args.gpu),
                "--num-envs",
                str(count),
                "--rollout-steps",
                str(args.rollout_steps),
                "--batch-size",
                str(batch),
                "--epochs",
                str(args.epochs),
                "--iterations",
                str(args.iterations),
                "--eval-episodes",
                "0",
                "--output",
                str(output),
            ]
            if args.algorithm == "ppo":
                command.extend(["--target-kl", "1000000000"])
            print(f"Running {backend}, {count} environments", flush=True)
            log_path = args.output / f"{backend}_{count}.log"
            with log_path.open("w") as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            if process.returncode:
                raise RuntimeError(f"Benchmark failed; see {log_path}")
            results[backend] = json.loads((output / "metrics.json").read_text())
        row = {
            "num_envs": count,
            "cpu_training_sps": results["gymnasium"]["training_steps_per_second"],
            "mjwarp_training_sps": results["mjwarp"]["training_steps_per_second"],
            "cpu_rollout_sps": results["gymnasium"]["rollout_steps_per_second"],
            "mjwarp_rollout_sps": results["mjwarp"]["rollout_steps_per_second"],
        }
        row["training_speedup"] = row["mjwarp_training_sps"] / row["cpu_training_sps"]
        row["rollout_speedup"] = row["mjwarp_rollout_sps"] / row["cpu_rollout_sps"]
        rows.append(row)
        (args.output / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
