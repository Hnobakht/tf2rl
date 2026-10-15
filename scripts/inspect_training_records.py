"""Validate completed CLI runs and summarize all TensorBoard scalar records.

Run with .venv-modern/bin/python scripts/inspect_training_records.py LOGDIR.
Writes validation_summary.json inside LOGDIR. Missing metrics, incomplete runs,
non-finite values, or missing checkpoints cause a nonzero exit.
"""
import argparse
import json
import sys
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def inspect_run(path):
    args = json.loads((path / 'args.txt').read_text())
    algorithm = args['algorithm']
    events = EventAccumulator(str(path), size_guidance={'tensors': 0}).Reload()
    required = {'SAC/actor_loss', 'SAC/critic_Q1_loss', 'Common/average_test_return'}
    if algorithm != 'sac':
        discriminator = {'gail': 'GAIL', 'eqn-gail': 'GAIL_EQL',
                         'airl': 'AIRL', 'eqn-airl': 'AIRL_EQL'}[algorithm]
        required.add(discriminator + ('/Loss' if 'airl' in algorithm else '/DiscriminatorLoss'))
        required.add('DAC/absorbing_batch_fraction')
    tags = set(events.Tags()['tensors'])
    if missing := required - tags:
        raise ValueError(f'{path}: missing TensorBoard metrics: {sorted(missing)}')
    metrics = {}
    for tag in sorted(tags):
        records = events.Tensors(tag)
        values = [float(tf.make_ndarray(record.tensor_proto)) for record in records]
        if not np.isfinite(values).all():
            raise ValueError(f'{path}: non-finite metric {tag}')
        metrics[tag] = {'count': len(records), 'final_step': records[-1].step,
                        'final_value': values[-1]}
        if tag == 'Common/average_test_return':
            metrics[tag]['curve'] = [[record.step, value] for record, value in zip(records, values)]
    if metrics['SAC/actor_loss']['final_step'] != args['max_steps']:
        raise ValueError(f'{path}: final actor summary is missing; use a summary interval dividing max_steps')
    checkpoint = tf.train.latest_checkpoint(str(path))
    if checkpoint is None:
        raise ValueError(f'{path}: no policy checkpoint')
    variables = [name for name, _ in tf.train.list_variables(checkpoint)]
    for critic in ('qf1_target', 'qf2_target'):
        if not any(f'policy/{critic}/hidden_0/' in name for name in variables):
            raise ValueError(f'{path}: target critic hidden weights missing from checkpoint')
    if algorithm != 'sac':
        irl_checkpoint = tf.train.latest_checkpoint(str(path / 'irl'))
        if irl_checkpoint is None:
            raise ValueError(f'{path}: no discriminator checkpoint')
        if algorithm == 'eqn-gail':
            variables = [name for name, _ in tf.train.list_variables(irl_checkpoint)]
            if not any('irl/disc/equation_0/' in name for name in variables):
                raise ValueError(f'{path}: equation weights missing from checkpoint')
    result = {'directory': str(path), 'algorithm': algorithm, 'seed': args['seed'],
            'environment': args['env_name'], 'training_steps': args['max_steps'],
            'checkpoint': checkpoint, 'metrics': metrics}
    if algorithm == 'sac' and args['save_test_path']:
        files = sorted(path.glob(f"step_{args['max_steps']:08d}_epi_*.pkl"))
        if len(files) != args['test_episodes']:
            raise ValueError(f'{path}: missing final demonstration episodes')
        returns, transitions = [], 0
        for filename in files:
            data = joblib.load(filename)
            length = len(data['obs'])
            for key in ('obs', 'next_obs', 'act', 'rew', 'done', 'terminated', 'truncated', 'logp'):
                if len(data[key]) != length or not np.isfinite(data[key]).all():
                    raise ValueError(f'{filename}: malformed {key}')
            if not np.allclose(data['next_obs'][:-1], data['obs'][1:]):
                raise ValueError(f'{filename}: trajectory is not ordered')
            if np.any(data['done'][:-1]) or not data['done'][-1]:
                raise ValueError(f'{filename}: not one complete episode')
            if not np.array_equal(data['done'], data['terminated'] | data['truncated']):
                raise ValueError(f'{filename}: inconsistent boundary flags')
            returns.append(float(np.sum(data['rew'])))
            transitions += length
        np.testing.assert_allclose(np.mean(returns),
                                   metrics['Common/average_test_return']['final_value'], rtol=1e-5)
        result['demonstrations'] = {'final_episodes': len(files), 'transitions': transitions,
                                    'mean_return': float(np.mean(returns))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logdir', type=Path)
    args = parser.parse_args()
    runs = sorted(path.parent for path in args.logdir.glob('*/*/args.txt'))
    if not runs:
        parser.error('No experiment run directories found')
    report = [inspect_run(path) for path in runs]
    output = args.logdir / 'validation_summary.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    packages = {distribution.metadata['Name']: distribution.version
                for distribution in metadata.distributions()}
    (args.logdir / 'environment_packages.json').write_text(
        json.dumps({'python': sys.version, 'packages': packages}, indent=2) + '\n')
    for run in report:
        metric = run['metrics']['Common/average_test_return']
        print(f"{run['algorithm']}: {run['training_steps']} steps; "
              f"final mean evaluation return {metric['final_value']:.3f}; "
              f"{len(run['metrics'])} finite metric series")
    print(f'Report: {output}')


if __name__ == '__main__':
    main()
