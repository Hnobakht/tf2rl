"""Compare saved Pendulum EQN policies on the same independently seeded starts.

Writes returns and equation complexity alongside a uniform-random policy baseline.
Uses the latest saved policy from each run; does not train or alter checkpoints.
"""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import tensorflow as tf

from tf2rl.algos.sac_irl import SAC_IRL
from tf2rl.envs.absorbing_wrapper import AbsorbingWrapper
from tf2rl.networks.equation_export import expression_statistics


def evaluate(policy, seeds, indicator=.1):
    env = gym.vector.SyncVectorEnv([
        lambda: AbsorbingWrapper(gym.make('Pendulum-v1'), indicator) for _ in seeds
    ])
    try:
        observations, _ = env.reset(seed=seeds)
        rng = np.random.default_rng(seeds[0])
        returns = np.zeros(len(seeds))
        active = np.ones(len(seeds), dtype=bool)
        for _ in range(env.envs[0].spec.max_episode_steps):
            if policy is None:
                actions = rng.uniform(env.single_action_space.low, env.single_action_space.high,
                                      size=(len(seeds),) + env.single_action_space.shape).astype(np.float32)
            else:
                actions = np.asarray(policy.get_action(observations, test=True)[0])
            observations, rewards, terminated, truncated, _ = env.step(actions)
            returns += rewards * active
            active &= ~(terminated | truncated)
            if not active.any():
                break
        return returns
    finally:
        env.close()


def summarize(returns):
    return {'mean': float(np.mean(returns)), 'std': float(np.std(returns, ddof=1)),
            'min': float(np.min(returns)), 'max': float(np.max(returns)),
            'returns': returns.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--first-seed', type=int, default=1000)
    parser.add_argument('--output', type=Path, default=Path('results/compact-equations/comparison.json'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(root):
        parser.error('Output must remain inside this repository')
    if args.episodes < 2 or args.first_seed < 0:
        parser.error('Require at least two episodes and a nonnegative seed')
    seeds = list(range(args.first_seed, args.first_seed + args.episodes))
    report = {'environment': 'Pendulum-v1', 'seeds': seeds, 'runs': [],
              'selection_note': 'Shared evaluation starts; one trained seed per candidate, not an independent benchmark.'}
    random_returns = evaluate(None, seeds)
    report['random_policy'] = summarize(random_returns)
    for path in args.runs:
        config = json.loads((path / 'args.txt').read_text())
        if config['env_name'] != 'Pendulum-v1' or config['algorithm'] != 'eqn-gail':
            parser.error(f'{path}: this comparison supports Pendulum EQN-GAIL runs')
        if config['normalize_obs'] or config['pixels']:
            parser.error('Observation normalization/pixels need a corresponding evaluation transform')
        units = [int(n) for n in config['units'].split(',')]
        tf.keras.utils.set_random_seed(config['seed'])
        policy = SAC_IRL(state_shape=(4,), action_dim=1, max_action=2., gpu=-1,
                         actor_units=units, critic_units=units, auto_alpha=config['auto_alpha_disable'],
                         alpha=config['alpha'], is_absorbed_state=config['absorbed_state_disable'])
        checkpoint = tf.train.latest_checkpoint(str(path))
        if checkpoint is None:
            parser.error(f'No policy checkpoint in {path}')
        tf.train.Checkpoint(policy=policy).restore(checkpoint).assert_existing_objects_matched()
        returns = evaluate(policy, seeds, config['absorbing_indicator'])
        graph_path = path / 'equations' / f"step_{config['max_steps']:08d}.json"
        graph = json.loads(graph_path.read_text())
        result = {'run': str(path), 'checkpoint': checkpoint,
                  'evaluation': summarize(returns), 'raw_expression': expression_statistics(graph),
                  'sparsity': graph['metadata']['sparsity'],
                  'paired_difference_from_random': summarize(returns - random_returns)}
        report['runs'].append(result)
        print(f"{path.parent.name}: mean {np.mean(returns):.3f}; "
              f"raw expression {result['raw_expression']}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Random-policy mean: {np.mean(random_returns):.3f}; report: {output}')


if __name__ == '__main__':
    main()
