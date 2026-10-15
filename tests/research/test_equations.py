import numpy as np
import pytest
import tensorflow as tf

from tf2rl.networks.eqn import EquationLayer, EquationOutput
from tf2rl.algos.irl_losses import airl_loss, gail_loss, agent_gradient_penalty
from tf2rl.algos.sac_irl import dac_target, dac_actor_loss


def test_equation_operators_and_penalty_match_analytic_reference():
    layer = EquationLayer(v=1)
    x = tf.Variable([[1., 2.], [-1., 3.]], dtype=tf.float32)
    layer(x)
    weights = np.arange(18, dtype=np.float32).reshape(2, 9) / 30 - .3
    bias = np.array([.1, -.2, .3, -.4, -.5, .6, -.7, .8, -.9], np.float32)
    layer.w.assign(weights)
    layer.b.assign(bias)
    z = x.numpy() @ weights + bias
    expected = np.column_stack((z[:, 0], np.sin(z[:, 1]), np.cos(z[:, 2]),
                                 1 / (1 + np.exp(-z[:, 3])), np.log(np.maximum(z[:, 4], 0) + .1),
                                 (z[:, 5] + z[:, 6]) ** 2,
                                 (z[:, 7] + .1) / (np.maximum(z[:, 8], 0) + .1)))
    with tf.GradientTape() as tape:
        result = layer(x)
        objective = tf.reduce_sum(result) + tf.add_n(layer.losses)
    gradient = tape.gradient(objective, x).numpy()
    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)
    penalty = (np.maximum(.001 - z[:, 4], 0).sum() + np.maximum(.001 - z[:, 8], 0).sum()) / len(z)
    np.testing.assert_allclose(layer.losses[0], penalty, rtol=1e-6)
    # Independent finite difference verifies gradients through operators and penalty.
    def reference(value):
        zz = value @ weights + bias
        vals = np.column_stack((zz[:, 0], np.sin(zz[:, 1]), np.cos(zz[:, 2]),
                               1/(1+np.exp(-zz[:, 3])), np.log(np.maximum(zz[:, 4], 0)+.1),
                               (zz[:, 5]+zz[:, 6])**2, (zz[:, 7]+.1)/(np.maximum(zz[:, 8], 0)+.1)))
        return vals.sum() + (np.maximum(.001-zz[:, 4], 0).sum()+np.maximum(.001-zz[:, 8], 0).sum())/len(zz)
    numerical = np.zeros_like(x.numpy())
    for index in np.ndindex(numerical.shape):
        plus, minus = x.numpy().copy(), x.numpy().copy()
        plus[index] += 1e-3
        minus[index] -= 1e-3
        numerical[index] = (reference(plus)-reference(minus)) / .002
    np.testing.assert_allclose(gradient, numerical, rtol=.01, atol=.002)


def test_exclusions_advance_binary_operator_inputs_correctly():
    layer = EquationLayer(v=2, exclude=['sin', 'cos', 'sig', 'log'])
    layer(tf.zeros((1, 1)))
    layer.w.assign(tf.zeros_like(layer.w))
    layer.b.assign(tf.range(10, dtype=tf.float32))
    np.testing.assert_allclose(layer(tf.zeros((1, 1))), [[0, 9, 3.1/4.1, 5, 169, 8.1/9.1]])
    assert layer.losses[0] == 0


def test_output_l1_value_and_gradient_inside_tf_function():
    layer = EquationOutput(lmbda=.25, regularize=True)
    layer(tf.zeros((1, 2)))
    layer.w.assign([[2.], [-3.]])
    layer.b.assign([4.])

    @tf.function
    def loss_and_gradients():
        with tf.GradientTape() as tape:
            layer(tf.zeros((2, 2)))
            loss = tf.add_n(layer.losses)
        return loss, tape.gradient(loss, layer.trainable_variables)

    loss, gradients = loss_and_gradients()
    np.testing.assert_allclose(loss, .25 * 9)
    np.testing.assert_allclose(gradients[0], [[.25], [-.25]])
    np.testing.assert_allclose(gradients[1], [.25])


def test_airl_loss_values_and_gradients():
    real = tf.Variable([[2.], [-1.]])
    fake = tf.Variable([[.5], [3.]])
    expert_logp = np.array([[-.2], [-.7]], np.float32)
    agent_logp = np.array([[-.6], [-.1]], np.float32)
    with tf.GradientTape() as tape:
        loss = airl_loss(real, fake, expert_logp, agent_logp)
    gradients = tape.gradient(loss, [real, fake])
    expected = -np.mean(real.numpy() - np.logaddexp(real.numpy(), expert_logp))
    expected -= np.mean(agent_logp - np.logaddexp(fake.numpy(), agent_logp))
    np.testing.assert_allclose(loss, expected)
    np.testing.assert_allclose(gradients[0], -(1 - 1/(1+np.exp(expert_logp-real.numpy())))/2, atol=1e-7)
    np.testing.assert_allclose(gradients[1], 1/(1+np.exp(agent_logp-fake.numpy()))/2, atol=1e-7)
    np.testing.assert_allclose(airl_loss(real, fake, expert_logp.ravel(), agent_logp.ravel()), loss)


def test_gail_loss_and_agent_only_gradient_penalty():
    real, fake = tf.constant([[.8], [.6]]), tf.constant([[.2], [.4]])
    expected = -(np.log([.8+1e-8, .6+1e-8]).mean()+np.log([.8+1e-8, .6+1e-8]).mean())
    np.testing.assert_allclose(gail_loss(real, fake), expected, rtol=1e-6)
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, use_bias=False)])
    model(tf.zeros((2, 3)))
    model.layers[0].kernel.assign([[1.], [2.], [3.]])
    np.testing.assert_allclose(agent_gradient_penalty(model, tf.ones((2, 3))), 14.)


@pytest.mark.parametrize('absorbing', [False, True])
def test_dac_targets_and_actor_masks(absorbing):
    masks = tf.constant([1., 0., -1.])
    values = dac_target(tf.constant([2., 3., 4.]), masks, tf.constant([5., 6., 7.]),
                         tf.constant([-.5, -.6, -.7]), .2, .9, absorbing)
    expected = [2 + .9*(5 + .1), 3 + .9*6, 4 + .9*7] if absorbing else [2+.9*(5+.1), 3, 4]
    np.testing.assert_allclose(values, expected, rtol=1e-6)
    loss = dac_actor_loss(tf.constant([1., 2., 100.]), tf.constant([3., 4., -100.]), masks, .2, absorbing)
    expected = (-2.8-3.6)/2 if absorbing else (-2.8-3.6+120)/3
    np.testing.assert_allclose(loss, expected, rtol=1e-6)
    if absorbing:
        assert dac_actor_loss(tf.ones(3), tf.ones(3), -tf.ones(3), .2, True) == 0
