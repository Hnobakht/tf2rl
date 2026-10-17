"""Pendulum SAC -> original EQN-GAIL -> staged EQN-GAIL integration experiment.

Creates demonstrations unless --expert-dir points to an existing Pendulum SAC run.
Outputs include TensorBoard, checkpoints and standalone exported equations.
"""
import argparse
import math
import os
import subprocess
import sys
from pathlib import Path


def main():
    """Run the configured equation-learning experiments and collect their outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/equation-learning'))
    parser.add_argument('--expert-dir', type=Path)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--interval', type=int, default=5000)
    parser.add_argument('--l1', type=float, default=.001)
    parser.add_argument('--stages', default='1000,4000')
    parser.add_argument('--threshold', type=float, default=.01)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve()
    if not output.is_relative_to(root):
        parser.error('Output must remain inside this repository')
    if args.steps <= 1000 or args.interval < 1 or args.steps % args.interval:
        parser.error('Require steps > 1000 and interval dividing steps')
    try:
        stages = [int(value) for value in args.stages.split(',')]
    except ValueError:
        parser.error('Stages must be comma-separated integer update counts')
    if not math.isfinite(args.l1) or args.l1 <= 0 or not math.isfinite(args.threshold) or args.threshold < 0:
        parser.error('Require a finite positive L1 coefficient and nonnegative threshold')
    updates = args.steps // 3 - 999 // 3
    if len(stages) != 2 or stages[0] < 0 or stages[1] < 1 or sum(stages) >= updates:
        parser.error('Choose warmup,sparse update counts that leave time for refitting')
    environment = os.environ.copy()
    for key, relative in [('TMPDIR', '.tmp'), ('MPLCONFIGDIR', '.cache/matplotlib'),
                          ('KERAS_HOME', '.cache/keras')]:
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        environment[key] = str(path)
    environment.update(PYTHONDONTWRITEBYTECODE='1')
    environment.setdefault('TF_NUM_INTRAOP_THREADS', '2')
    environment.setdefault('TF_NUM_INTEROP_THREADS', '2')
    common = ['--env-name', 'Pendulum-v1', '--max-steps', str(args.steps),
              '--n-warmup', '1000', '--batch-size', '128', '--batch-size-ail', '32',
              '--memory-capacity', '50000', '--units', '64,64', '--test-interval', str(args.interval),
              '--save-model-interval', str(args.interval),
              '--save-summary-interval', str(math.gcd(100, args.interval))]
    expert = args.expert_dir.resolve() if args.expert_dir else None
    variants = ['baseline', 'sparse'] if expert else ['sac', 'baseline', 'sparse']
    for variant in variants:
        directory = output / variant
        directory.mkdir(parents=True, exist_ok=True)
        before = set(directory.iterdir())
        algorithm = 'sac' if variant == 'sac' else 'eqn-gail'
        command = [sys.executable, '-m', 'tf2rl.experiments.cli', '--algorithm', algorithm,
                   '--seed', '7' if variant == 'sac' else '13', '--logdir', str(directory), *common]
        if variant == 'sac':
            command += ['--test-episodes', '20', '--save-test-path']
        else:
            command += ['--expert-path-dir', str(expert), '--test-episodes', '10', '--export-equation']
        if variant == 'sparse':
            command += ['--eqn-l1', str(args.l1), '--eqn-stages', args.stages,
                        '--eqn-prune-threshold', str(args.threshold)]
        print(f'Starting {variant}: {" ".join(command)}', flush=True)
        subprocess.run(command, cwd=root, env=environment, check=True)
        created = set(directory.iterdir()) - before
        if len(created) != 1:
            raise RuntimeError(f'Expected one run under {directory}, got {created}')
        if variant == 'sac':
            expert = created.pop()
        else:
            subprocess.run([sys.executable, 'scripts/inspect_equation.py', str(created.pop())],
                           cwd=root, env=environment, check=True)
    subprocess.run([sys.executable, 'scripts/inspect_training_records.py', str(output)],
                   cwd=root, env=environment, check=True)


if __name__ == '__main__':
    main()
