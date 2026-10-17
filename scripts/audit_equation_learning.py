"""Numerical diagnostic for equation learning; writes its evidence inside the repo.

Exercises actual optimizers, checkpoints, independent equation evaluation and DAC
boundaries. This is an executable diagnostic, not a unit-test suite.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import tensorflow as tf

from tf2rl.algos.airl_eql import AIRL_EQL
from tf2rl.algos.gail_eql import GAIL_EQL
from tf2rl.envs.absorbing_wrapper import AbsorbingWrapper
from tf2rl.misc.dac_ops import dac_transitions
from tf2rl.networks.eqn import EquationLayer, EquationNetwork
from tf2rl.networks.equation_export import evaluate_equation, extract_equation, save_equation


def arrays(model):
    """Copy the model's trainable parameters into NumPy arrays."""
    return [value.numpy().copy() for value in model.trainable_variables]


def masked_zeros(model):
    """Assert that masked parameters remain zero and return their count."""
    pruned = 0
    for layer in model.parameter_layers:
        for weight, mask in ((layer.w, layer.w_mask), (layer.b, layer.b_mask)):
            zeros = mask.numpy() == 0
            assert np.all(weight.numpy()[zeros] == 0), 'Pruned parameters regrew'
            pruned += int(zeros.sum())
    assert pruned > 0, 'Diagnostic failed to exercise pruning'
    return pruned


def export_error(model, inputs, kind, path):
    """Export the model and report the maximum NumPy inference error and graph size."""
    graph = extract_equation(model, reward_kind=kind)
    save_equation(graph, path)
    expected = model(inputs, training=False).numpy()
    results = evaluate_equation(graph, inputs)
    key = 'probability' if kind == 'gail' else 'reward'
    np.testing.assert_allclose(results[key], expected, atol=3e-6, rtol=3e-5)
    if kind == 'gail':
        np.testing.assert_allclose(results['reward'], model.compute_reward(inputs).numpy(), atol=1e-5, rtol=3e-5)
    spec = importlib.util.spec_from_file_location('exported', path.with_suffix('.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, value in results.items():
        np.testing.assert_array_equal(module.predict(inputs)[name], value)
    return {'max_output_error': float(np.max(np.abs(results[key] - expected))),
            'nodes': len(graph['nodes'])}


def main():
    """Run equation fitting, pruning, checkpoint, and export consistency diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('.work/equation-learning-audit'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(root):
        parser.error('Output must be inside this repository')
    output.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(21)
    rng = np.random.default_rng(21)
    x = rng.normal(size=(32, 5)).astype(np.float32)
    batch = {'agent_states': x[:, :4], 'agent_acts': x[:, 4:],
                 'expert_states': -x[:, :4], 'expert_acts': -x[:, 4:]}
    common = {'state_shape': (4,), 'action_dim': 1, 'v': (2, 1), 'drop_out': (0., 0.), 'update_delay': 3}
    baseline = GAIL_EQL(**common)
    sparse = GAIL_EQL(**common, lmbda=.01, training_stages=(2, 3), prune_threshold=.03)
    for target, source in zip(sparse.disc.trainable_variables, baseline.disc.trainable_variables):
        target.assign(source)
    # Before sparsification, identical discriminator parameters give identical updates.
    for step in range(1, 7):
        a = baseline.train(**batch, itr=step)
        b = sparse.train(**batch, itr=step)
        if a is not None:
            np.testing.assert_allclose(a, b, atol=1e-7, rtol=1e-6)
    for a, b in zip(arrays(baseline.disc), arrays(sparse.disc)):
        np.testing.assert_array_equal(a, b)
    assert int(sparse.disc.sparsity_updates.numpy()) == 2
    sparse.disc.prepare_update()
    with tf.GradientTape() as tape:
        sparse.disc(x, training=False)
        loss = sparse.disc.losses[0]  # Network L1 is separate from singularity penalties.
        expected_l1 = .01 * sparse.disc.l1_norm()
    gradients = tape.gradient(expected_l1, sparse.disc.trainable_variables)
    assert all(g is not None and np.any(g.numpy() != 0) for g in gradients)
    np.testing.assert_allclose(loss, expected_l1)
    baseline_loss = baseline.train(**batch, itr=9)[0]
    regularized_loss = sparse.train(**batch, itr=9)[0]
    np.testing.assert_allclose(regularized_loss - baseline_loss, expected_l1, atol=1e-6)
    for step in range(10, 31):
        sparse.train(**batch, itr=step)
    assert int(sparse.disc.sparsity_phase.numpy()) == 2
    assert float(sparse.disc.l1_coefficient.numpy()) == 0
    count = masked_zeros(sparse.disc)
    checkpoint = tf.train.Checkpoint(irl=sparse)
    path = checkpoint.save(str(output / 'resume'))
    restored = GAIL_EQL(**common, lmbda=.5, training_stages=(99, 100), prune_threshold=.9)
    restored_checkpoint = tf.train.Checkpoint(irl=restored)
    restored_checkpoint.restore(path).assert_consumed()
    for step in range(31, 43):
        left = sparse.train(**batch, itr=step)
        right = restored.train(**batch, itr=step)
        if left is not None:
            np.testing.assert_allclose(left, right, atol=1e-7, rtol=1e-6)
    for a, b in zip(arrays(sparse.disc), arrays(restored.disc)):
        np.testing.assert_array_equal(a, b)
    assert masked_zeros(restored.disc) == count
    report = {'warmup_matches_default': True, 'all_layers_receive_l1_gradients': True,
              'checkpoint_resume_matches_uninterrupted': True, 'pruned_parameters': count,
              'gail_export': export_error(sparse.disc, x, 'gail', output / 'gail')}
    # Exercise the GP's extra discriminator call and both AIRL reward input modes.
    gp = GAIL_EQL(**common, lmbda=.01, training_stages=(0, 1), grad_penalty_coeff=1.)
    for step in (3, 6):
        assert np.isfinite(np.asarray(gp.train(**batch, itr=step))).all()
    report['gail_gradient_penalty_finite'] = True
    for state_only in (False, True):
        airl = AIRL_EQL(**common, lmbda=.01, training_stages=(0, 2),
                        prune_threshold=.03, is_state_only=state_only)
        for step in (3, 6, 9, 12):
            values = airl.train(**batch, agent_next_states=x[:, :4], expert_next_states=-x[:, :4],
                                agent_logps=np.zeros((32, 1), np.float32),
                                expert_logps=np.zeros((32, 1), np.float32), itr=step)
            assert np.isfinite(np.asarray(values)).all()
        masked_zeros(airl.rew_net)
        key = 'airl_state' if state_only else 'airl_state_action'
        report[key] = export_error(airl.rew_net, x[:, :4] if state_only else x, 'airl', output / key)
    # Standalone hidden layers also honor lambda.
    layer = EquationLayer(lmbda=.01)
    layer(x)
    assert len(layer.losses) == 2
    # Pendulum truncates, so explicitly exercise true termination as well.
    env = AbsorbingWrapper(gym.make('Pendulum-v1'), .1)
    try:
        obs, _ = env.reset(seed=21)
        action = np.zeros(1, np.float32)
        next_obs, reward, _, _, _ = env.step(action)
        for terminated, truncated in ((False, False), (False, True), (True, False), (True, True)):
            rows = dac_transitions(env, obs, action, next_obs, reward, terminated, truncated)
            assert [row['done'] for row in rows] == ([0., -1.] if terminated else [1.])
            if terminated:
                np.testing.assert_array_equal(rows[0]['next_obs'], env.get_absorbing_state())
                np.testing.assert_array_equal(rows[1]['obs'], rows[1]['next_obs'])
                assert not np.any(rows[1]['act'])
        report['dac_termination_and_truncation'] = True
    finally:
        env.close()
    # A fully masked network must simplify to input-independent constants.
    constant = EquationNetwork((5,), lmbda=.01, v=(1, 1), output_activation='sigmoid')
    for layer in constant.parameter_layers:
        layer.prune(100.)
    graph = extract_equation(constant, reward_kind='gail')
    assert all(node['op'] == 'constant' for node in graph['nodes'])
    np.testing.assert_allclose(evaluate_equation(graph, x)['probability'], constant(x, training=False))
    report['constant_graph_nodes'] = len(graph['nodes'])
    # Known polynomial diagnostic: deliberately restricted vocabulary, not a claim
    # that Pendulum imitation recovers its physical reward or a unique equation.
    tf.keras.utils.set_random_seed(29)
    polynomial = EquationNetwork((1,), num_layers=1, v=(1,),
        exclude=[['sin', 'cos', 'sig', 'log', 'div']],
        lmbda=.001, training_stages=(300, 300), prune_threshold=.02)
    optimizer = tf.keras.optimizers.Adam(.02)
    optimizer.build(polynomial.trainable_variables)
    train_x = np.linspace(-1, 1, 128, dtype=np.float32)[:, None]
    train_y = 1.5 * train_x ** 2 - .7 * train_x + .2

    @tf.function
    def fit_polynomial():
        """Apply one update to the polynomial fitting objective and regularization."""
        with tf.GradientTape() as tape:
            predictions = polynomial(train_x, training=False)
            objective = tf.reduce_mean(tf.square(predictions - train_y)) + tf.add_n(polynomial.losses)
        optimizer.apply_gradients(zip(tape.gradient(objective, polynomial.trainable_variables),
                                      polynomial.trainable_variables))
        polynomial.finish_update()

    for _ in range(2000):
        polynomial.prepare_update()
        fit_polynomial()
    holdout = np.linspace(-2, 2, 201, dtype=np.float32)[:, None]
    predictions = polynomial(holdout, training=False).numpy()
    expected = 1.5 * holdout ** 2 - .7 * holdout + .2
    np.testing.assert_allclose(predictions, expected, atol=1e-4, rtol=1e-4)
    report['restricted_polynomial'] = {
        'target_coefficients_ascending': [.2, -.7, 1.5],
        'fitted_coefficients_ascending': np.polynomial.polynomial.polyfit(
            holdout[:, 0], predictions[:, 0], 2).tolist(),
        'max_holdout_error': float(np.max(np.abs(expected - predictions))),
        'export': export_error(polynomial, holdout, 'airl', output / 'polynomial'),
        'sparsity': {name: float(np.asarray(value)) for name, value in polynomial.sparsity_metrics().items()}}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
