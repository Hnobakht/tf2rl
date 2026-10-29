"""Run SAC demonstrations, dense GAIL, and EQN-GAIL sequentially on Pendulum.

All stages use the same training budget; their complete args, checkpoints and
TensorBoard records are retained. This is a single-seed integration experiment.
"""
import argparse
import math
import os
import subprocess
import sys
from pathlib import Path


def main():
    """Run the configured training and validation pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--interval', type=int, default=5000)
    parser.add_argument('--units', default='64,64')
    parser.add_argument('--output', type=Path, default=Path('results/validated_pipeline'))
    args = parser.parse_args()
    if not 0 < args.warmup < args.steps or args.interval <= 0 or args.steps % args.interval:
        parser.error('Require 0 < warmup < steps and an interval that divides steps')
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve()
    if not output.is_relative_to(root):
        parser.error('Output must be inside this checkout')
    environment = os.environ.copy()
    for key, relative in [('TMPDIR', '.tmp'), ('MPLCONFIGDIR', '.cache/matplotlib'),
                          ('KERAS_HOME', '.cache/keras')]:
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        environment[key] = str(directory)
    environment.setdefault('TF_NUM_INTRAOP_THREADS', '2')
    environment.setdefault('TF_NUM_INTEROP_THREADS', '2')
    common = ['--env-name', 'Pendulum-v1', '--max-steps', str(args.steps),
              '--episode-max-steps', '200', '--n-warmup', str(args.warmup),
              '--batch-size', '128', '--batch-size-ail', '32',
              '--memory-capacity', str(max(50000, args.steps)), '--units', args.units,
              '--units-ail', args.units, '--test-interval', str(args.interval),
              '--save-model-interval', str(args.interval),
              '--save-summary-interval', str(math.gcd(100, args.interval))]
    expert = None
    for algorithm, seed in [('sac', 7), ('gail', 11), ('eqn-gail', 13)]:
        directory = output / algorithm
        directory.mkdir(parents=True, exist_ok=True)
        before = set(directory.iterdir())
        command = [sys.executable, '-m', 'tf2rl.experiments.train', '--runner', 'imitation', '--algorithm', algorithm,
                   '--seed', str(seed), '--logdir', str(directory),
                   '--dir-suffix', f'pendulum_seed{seed}', *common]
        if expert is None:
            command += ['--save-test-path', '--test-episodes', '20']
        else:
            command += ['--expert-path-dir', str(expert), '--n-expert-trajectories', '20',
                        '--test-episodes', '10']
        print(f'Starting {algorithm}', flush=True)
        subprocess.run(command, cwd=root, env=environment, check=True)
        created = set(directory.iterdir()) - before
        if len(created) != 1:
            raise RuntimeError(f'Expected one new experiment under {directory}, got {created}')
        if expert is None:
            expert = created.pop()
    subprocess.run([sys.executable, str(root / 'scripts/inspect_training_records.py'), str(output)],
                   cwd=root, env=environment, check=True)


if __name__ == '__main__':
    main()
