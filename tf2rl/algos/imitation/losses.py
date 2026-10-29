"""Adversarial objectives with this project's expert-positive label convention.

References: Ho & Ermon (2016), https://arxiv.org/abs/1606.03476;
Fu et al. (2018), https://arxiv.org/abs/1710.11248. These functions preserve
the executable research baseline; callers determine the sampling distribution.
"""
import tensorflow as tf

EPSILON = 1e-8


def airl_loss(real_f, fake_f, expert_logps, agent_logps):
    """Compute ``-E_expert[log D] - E_agent[log(1-D)]`` for AIRL.

    For shaped score ``f = r + gamma * V(s_next) - V(s)``, the discriminator
    is ``D = exp(f) / (exp(f) + exp(logp))``. Scores have shape ``(batch, 1)``;
    each class mean is taken separately, so class sizes need not match.
    Log-sum-exp avoids explicitly exponentiating potentially large scores.

    Canonical AIRL evaluates the current generator density on both classes.
    The repository trainer instead supplies stored behavior-policy log densities,
    including expert densities; this historical approximation is preserved here.
    """
    expert_logps = tf.reshape(tf.cast(expert_logps, real_f.dtype), (-1, 1))
    agent_logps = tf.reshape(tf.cast(agent_logps, fake_f.dtype), (-1, 1))
    real_normalizer = tf.reduce_logsumexp(tf.stack((real_f, expert_logps)), axis=0)
    fake_normalizer = tf.reduce_logsumexp(tf.stack((fake_f, agent_logps)), axis=0)
    return -(tf.reduce_mean(real_f - real_normalizer)
             + tf.reduce_mean(agent_logps - fake_normalizer))


def gail_loss(real_probabilities, fake_probabilities):
    """Return ``-E_expert[log D] - E_agent[log(1-D)]`` with expert label one.

    Inputs are probabilities, not logits. EPSILON stabilizes saturated sigmoid
    outputs. The policy reward is chosen separately: dense GAIL uses ``log D``
    and EQN-GAIL uses stabilized log odds, so their learning signals differ.
    """
    return -(tf.reduce_mean(tf.math.log(real_probabilities + EPSILON))
             + tf.reduce_mean(tf.math.log(1. - fake_probabilities + EPSILON)))


def singular_penalty(model):
    """Sum the model's auxiliary losses, returning scalar zero if none exist."""
    return tf.add_n(model.losses) if model.losses else tf.zeros((), tf.float32)


def agent_gradient_penalty(model, inputs):
    """Return ``E_agent[||grad_x D(x)||_2**2]`` for scalar probability outputs.

    This is a zero-centered penalty on agent samples, without interpolated
    inputs or a unit-norm target. The caller's outer tape differentiates this
    quantity with respect to model parameters. Summing outputs in the inner
    gradient gives per-row derivatives only for networks without batch coupling.
    """
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(inputs)
        output = model(inputs)
    gradient = tape.gradient(output, inputs)
    return tf.reduce_mean(tf.reduce_sum(tf.square(gradient), axis=-1))


def js_divergence(fake, real):
    # Historical diagnostic, not a full Bernoulli Jensen-Shannon divergence.
    """Return the historical output-based diagnostic, rather than a full Bernoulli JS divergence."""
    mean = (fake + real) / 2.
    return tf.reduce_mean((fake * tf.math.log(fake / mean + EPSILON)
                           + real * tf.math.log(real / mean + EPSILON)) / 2.)
