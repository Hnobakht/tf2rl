# Copyright 2018 The trfl Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or    implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Tensorflow ops for updating target networks.

Tensorflow ops that are used to update a target network from a source network.
This is used in agents such as DQN or DPG, which use a target network that
changes more slowly than the online network, in order to improve stability.
"""

import tensorflow as tf


def update_target_variables(
    target_variables, source_variables, tau=1.0, use_locking=False, name="update_target_variables"
):
    """
    Returns an op to update a list of target variables from source variables.

    The update rule is:
    `target_variable = (1 - tau) * target_variable + tau * source_variable`.

    :param target_variables: a list of the variables to be updated.
    :param source_variables: a list of the variables used for the update.
    :param tau: weight used to gate the update. The permitted range is 0 < tau <= 1,
        with small tau representing an incremental update, and tau == 1
        representing a full update (that is, a straight copy).
    :param use_locking: use `tf.Variable.assign`'s locking option when assigning
        source variable values to target variables.
    :param name: sets the `name_scope` for this op.
    :raise TypeError: when tau is not a Python float
    :raise ValueError: when tau is out of range, or the source and target variables
        have different numbers or shapes.
    :return: An op that executes all the variable updates.
    """
    if not isinstance(tau, float):
        raise TypeError("Tau has wrong type (should be float) {}".format(tau))
    if not 0.0 < tau <= 1.0:
        raise ValueError("Invalid parameter tau {}".format(tau))
    if len(target_variables) != len(source_variables):
        raise ValueError(
            "Number of target variables {} is not the same as number of source variables {}".format(
                len(target_variables), len(source_variables)
            )
        )

    same_shape = all(trg.shape == src.shape for trg, src in zip(target_variables, source_variables))
    if not same_shape:
        raise ValueError("Target variables don't have the same shape as source variables.")

    def update_op(target_variable, source_variable, tau):
        """Assign source parameters to the target, interpolating when tau is below one."""
        if tau == 1.0:
            return target_variable.assign(source_variable)
        else:
            return target_variable.assign(tau * source_variable + (1.0 - tau) * target_variable)

    # with tf.name_scope(name):
    update_ops = [
        update_op(target_var, source_var, tau)
        for target_var, source_var in zip(target_variables, source_variables)
    ]
    return tf.group(name="update_all_variables", *update_ops)


def periodic_target_update(
    target_variables,
    source_variables,
    update_period,
    tau=1.0,
    use_locking=False,
    name="periodic_target_update",
):
    """
    Returns an op to periodically update a list of target variables.

    The `update_target_variables` op is executed every `update_period`
    executions of the `periodic_target_update` op.

    The update rule is:
    `target_variable = (1 - tau) * target_variable + tau * source_variable`.

    :param target_variables: a list of the variables to be updated.
    :param source_variables: a list of the variables used for the update.
    :param update_period: inverse frequency with which to apply the update.
    :param tau: weight used to gate the update. The permitted range is 0 < tau <= 1,
        with small tau representing an incremental update, and tau == 1
        representing a full update (that is, a straight copy).
    :param use_locking: use `tf.variable.Assign`'s locking option when assigning
        source variable values to target variables.
    :param name: sets the `name_scope` for this op.
    :return: An op that periodically updates `target_variables` with `source_variables`.
    """

    def update_op():
        """Apply the configured target-network update."""
        return update_target_variables(target_variables, source_variables, tau, use_locking)

    with tf.name_scope(name):
        return periodically(update_op, update_period)


def huber_loss(x, delta=1.0):
    """

    Args:
        x: np.ndarray or tf.Tensor
            Values to compute the huber loss.
        delta: float
            Positive floating point value. Represents the
            maximum possible gradient magnitude.

    Returns: tf.Tensor
        The huber loss.
    """
    delta = tf.ones_like(x) * delta
    less_than_max = 0.5 * tf.square(x)
    greater_than_max = delta * (tf.abs(x) - 0.5 * delta)
    return tf.where(tf.abs(x) <= delta, x=less_than_max, y=greater_than_max)


def configure_tensorflow(seed, cpu_threads, cpu_inter_threads=1):
    """Configure a fresh vector-training process before constructing any tensors."""
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.threading.set_intra_op_parallelism_threads(cpu_threads)
    tf.config.threading.set_inter_op_parallelism_threads(cpu_inter_threads)
    tf.keras.utils.set_random_seed(seed)


# Copyright 2018 The trfl Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or    implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Periodic execution ops.

It is very common in Reinforcement Learning for certain ops to only need to be
executed periodically, for example: once every N agent steps. The ops below
support this common use-case by wrapping a subgraph as a periodic op that only
actually executes the underlying computation once every N evaluations of the op,
behaving as a no-op in all other calls.
"""


def periodically(body, period, name="periodically"):
    """
    Periodically performs a tensorflow op.

    The body tensorflow op will be executed every `period` times the periodically
    op is executed. More specifically, with `n` the number of times the op has
    been executed, the body will be executed when `n` is a non zero positive
    multiple of `period` (i.e. there exist an integer `k > 0` such that
    `k * period == n`).

    If `period` is 0 or `None`, it would not perform any op and would return a
    `tf.no_op()`.

    :param body (callable): callable that returns the tensorflow op to be performed every time
        an internal counter is divisible by the period. The op must have no
        output (for example, a tf.group()).
    :param period (int): inverse frequency with which to perform the op.
    :param name (str): name of the variable_scope.
    :raise TypeError: if body is not a callable.
    :raise ValueError: if period is negative.
    :return: An op that periodically performs the specified op.
    """
    if not callable(body):
        raise TypeError("body must be callable.")

    if period is None or period == 0:
        return tf.no_op()

    if period < 0:
        raise ValueError("period cannot be less than 0.")

    if period == 1:
        return body()

    with tf.variable_scope(None, default_name=name):
        counter = tf.get_variable(
            "counter",
            shape=[],
            dtype=tf.int64,
            trainable=False,
            initializer=tf.constant_initializer(period, dtype=tf.int64),
        )

        def _wrapped_body():
            """Run the periodic operation and reset its invocation counter."""
            with tf.control_dependencies([body()]):
                # Done the deed, resets the counter.
                return counter.assign(1)

        update = tf.cond(tf.equal(counter, period), _wrapped_body, lambda: counter.assign_add(1))

    return update
