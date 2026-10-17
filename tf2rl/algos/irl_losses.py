"""The original research objectives, kept independently testable."""
import tensorflow as tf

EPSILON = 1e-8


def airl_loss(real_f, fake_f, expert_logps, agent_logps):
    """Return the AIRL discriminator loss normalized by expert and agent policy log densities."""
    expert_logps = tf.reshape(tf.cast(expert_logps, real_f.dtype), (-1, 1))
    agent_logps = tf.reshape(tf.cast(agent_logps, fake_f.dtype), (-1, 1))
    real_normalizer = tf.reduce_logsumexp(tf.stack((real_f, expert_logps)), axis=0)
    fake_normalizer = tf.reduce_logsumexp(tf.stack((fake_f, agent_logps)), axis=0)
    return -(tf.reduce_mean(real_f - real_normalizer)
             + tf.reduce_mean(agent_logps - fake_normalizer))


def gail_loss(real_probabilities, fake_probabilities):
    """
    Return binary cross-entropy for expert probabilities labeled one and agent probabilities zero.
    """
    return -(tf.reduce_mean(tf.math.log(real_probabilities + EPSILON))
             + tf.reduce_mean(tf.math.log(1. - fake_probabilities + EPSILON)))


def singular_penalty(model):
    """Sum the model's auxiliary losses, returning scalar zero if none exist."""
    return tf.add_n(model.losses) if model.losses else tf.zeros((), tf.float32)


def agent_gradient_penalty(model, inputs):
    """Return the mean squared input-gradient norm on agent samples."""
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
