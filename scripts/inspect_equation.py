"""Verify a saved EQN run against its exported NumPy equation on Pendulum inputs."""
import argparse
import importlib.util
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from tf2rl.algos.airl_eql import AIRL_EQL
from tf2rl.algos.gail_eql import GAIL_EQL
from tf2rl.envs.absorbing_wrapper import AbsorbingWrapper
from tf2rl.networks.equation_export import evaluate_equation


def inspect_equation(path):
    args = json.loads((path / 'args.txt').read_text())
    if args['env_name'] != 'Pendulum-v1':
        raise ValueError('This diagnostic samples Pendulum observations and actions')
    config = {'state_shape': (4,), 'action_dim': 1, 'v': [int(v) for v in args['v'].split(',')],
                  'num_layers': len(args['v'].split(',')), 'drop_out': [float(p) for p in args['dropout'].split(',')],
                  'exclude': [v.split('-') if v else [] for v in args['exclude'].split(',')] if args['exclude'] else None,
                  'lmbda': args['eqn_l1'], 'training_stages': [int(v) for v in args['eqn_stages'].split(',')]
                  if args['eqn_stages'] else None, 'prune_threshold': args['eqn_prune_threshold']}
    if args['algorithm'] == 'eqn-gail':
        irl = GAIL_EQL(**config)
    elif args['algorithm'] == 'eqn-airl':
        irl = AIRL_EQL(**config, is_state_only=args['state_only'],
                       units=[int(v) for v in args['units_ail'].split(',')])
    else:
        raise ValueError('Requires an EQN algorithm')
    checkpoint = tf.train.latest_checkpoint(str(path / 'irl'))
    if checkpoint is None:
        raise ValueError('No discriminator checkpoint')
    tf.train.Checkpoint(irl=irl).restore(checkpoint).assert_existing_objects_matched()
    model = irl.equation_model
    filename = path / 'equations' / f"step_{args['max_steps']:08d}.json"
    graph = json.loads(filename.read_text())
    states, actions = [], []
    env = AbsorbingWrapper(gym.make('Pendulum-v1'), args['absorbing_indicator'])
    try:
        obs, _ = env.reset(seed=17)
        env.action_space.seed(17)
        for _ in range(1000):
            act = env.action_space.sample()
            states.append(obs)
            actions.append(act)
            obs, _, terminated, truncated, _ = env.step(act)
            if terminated or truncated:
                obs, _ = env.reset()
        states.append(env.get_absorbing_state())
        actions.append(np.zeros(1, np.float32))
    finally:
        env.close()
    states, actions = np.asarray(states), np.asarray(actions)
    inputs = states if getattr(irl, '_state_only', False) else np.concatenate((states, actions), axis=1)
    results = evaluate_equation(graph, inputs)
    expected = irl.inference(states, actions).numpy()
    np.testing.assert_allclose(results['reward'], expected, atol=2e-5, rtol=3e-5)
    spec = importlib.util.spec_from_file_location('exported', filename.with_suffix('.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    np.testing.assert_array_equal(module.predict(inputs)['reward'], results['reward'])
    metrics = {name: float(np.asarray(value)) for name, value in model.sparsity_metrics().items()}
    np.testing.assert_allclose(list(metrics.values()), list(graph['metadata']['sparsity'].values()))
    phases = []
    if model.sparse:
        for layer in model.parameter_layers:
            for weight, mask in ((layer.w, layer.w_mask), (layer.b, layer.b_mask)):
                assert np.all(weight.numpy()[mask.numpy() == 0] == 0)
        if not args['irl_model_dir']:
            updates = (args['max_steps'] // args['update_delay']
                       - (max(1, args['n_warmup']) - 1) // args['update_delay']) * args['n_training']
            assert metrics['discriminator_updates'] == updates
        events = EventAccumulator(str(path), size_guidance={'tensors': 0}).Reload()
        for tag in ('EQN/phase', 'EQN/pruned_fraction', 'EQN/l1_norm', 'EQN/l1_coefficient'):
            assert tag in events.Tags()['tensors'], f'Missing metric {tag}'
        phases = sorted({float(tf.make_ndarray(event.tensor_proto)) for event in events.Tensors('EQN/phase')})
    return {'run': str(path), 'checkpoint': checkpoint, 'samples': len(inputs),
            'max_reward_export_error': float(np.max(np.abs(expected - results['reward']))),
            'reachable_expression_nodes': len(graph['nodes']), 'recorded_phases': phases,
            'sparsity': metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error('Run must be inside this repository')
    report = inspect_equation(run)
    (run / 'equation_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
