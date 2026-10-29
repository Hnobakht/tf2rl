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
    """Returns an op to update a list of target variables from source variables.

    The update rule is:
    `target_variable = (1 - tau) * target_variable + tau * source_variable`.

    Args:
        target_variables (sequence[tf.Variable]): Destination variables updated from corresponding
            source variables.
        source_variables (sequence[tf.Variable]): Online/source variables paired with
            target_variables.
        tau (float): Target-network interpolation coefficient: target <- (1-tau)*target +
            tau*online; 1 copies all weights.
        use_locking (bool): Historical assignment-locking option retained by the target-update API.
            This implementation does not read this argument.
        name (str): Name assigned to the model, operation, or registered component. This
            implementation does not read this argument.

    Returns:
        tf.Operation or None: Grouped assignment operation in graph mode; eager execution applies
            the assignments immediately and returns None.

    Raises:
        ValueError: Target variables don't have the same shape as source variables.
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
        """Assign source parameters to the target, interpolating when tau is below one.

        Args:
            target_variable (tf.Variable): Destination variable receiving the interpolated assignment.
            source_variable (tf.Variable): Source variable supplying the new parameter values.
            tau (float): Target-network interpolation coefficient: target <- (1-tau)*target +
                tau*online; 1 copies all weights.

        Returns:
            tf.Tensor: Assigned target-variable value, with the same shape and dtype as the target.
        """
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
    """Returns an op to periodically update a list of target variables.

    The `update_target_variables` op is executed every `update_period`
    executions of the `periodic_target_update` op.

    The update rule is:
    `target_variable = (1 - tau) * target_variable + tau * source_variable`.

    Args:
        target_variables (sequence[tf.Variable]): Destination variables updated from corresponding
            source variables.
        source_variables (sequence[tf.Variable]): Online/source variables paired with
            target_variables.
        update_period (int): Number of calls between target-network updates.
        tau (float): Target-network interpolation coefficient: target <- (1-tau)*target +
            tau*online; 1 copies all weights.
        use_locking (bool): Historical assignment-locking option retained by the target-update API.
        name (str): Name assigned to the model, operation, or registered component.

    Returns:
        tf.Operation or tf.Tensor or None: Scheduled target-update operation; eager execution
            applies the selected assignments immediately.
    """

    def update_op():
        """Apply the configured target-network update.

        Returns:
            tf.Operation or None: Result of assigning the target variables from their paired sources.
        """
        return update_target_variables(target_variables, source_variables, tau, use_locking)

    with tf.name_scope(name):
        return periodically(update_op, update_period)


def huber_loss(x, delta=1.0):
    """Compute the elementwise quadratic/linear Huber penalty for prediction errors.

    Args:
        x (tf.Tensor): Elementwise prediction residuals to penalize.
        delta (float): Huber-loss threshold between quadratic and linear error penalties.

    Returns:
        tf.Tensor: Elementwise Huber losses with the same shape as x; no batch reduction is applied.
    """
    delta = tf.ones_like(x) * delta
    less_than_max = 0.5 * tf.square(x)
    greater_than_max = delta * (tf.abs(x) - 0.5 * delta)
    return tf.where(tf.abs(x) <= delta, x=less_than_max, y=greater_than_max)


def configure_tensorflow(seed, cpu_threads, cpu_inter_threads=1):
    """Configure a fresh vector-training process before constructing any tensors.

    Args:
        seed (int or None): Random seed; None leaves initialization to the underlying random
            generator.
        cpu_threads (int): TensorFlow intra-operation CPU thread count.
        cpu_inter_threads (int): TensorFlow inter-operation CPU thread count.

    Returns:
        None.
    """
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
    """Periodically performs a tensorflow op.

    The body tensorflow op will be executed every `period` times the periodically
    op is executed. More specifically, with `n` the number of times the op has
    been executed, the body will be executed when `n` is a non zero positive
    multiple of `period` (i.e. there exist an integer `k > 0` such that
    `k * period == n`).

    If `period` is 0 or `None`, it would not perform any op and would return a
    `tf.no_op()`.

    Args:
        body (callable): Zero-argument TensorFlow operation factory executed at the requested
            period.
        period (int or None): Number of calls between executions; None or zero disables the
            operation.
        name (str): Name assigned to the model, operation, or registered component.

    Returns:
        tf.Operation or tf.Tensor or None: No-op when disabled, body result for period one, or the
            counter-update operation for longer periods.

    Raises:
        TypeError: body must be callable.
        ValueError: period cannot be less than 0.
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
            """Run the periodic operation and reset its invocation counter.

            Returns:
                tf.Tensor: Counter value after executing the body and resetting the counter to one.
            """
            with tf.control_dependencies([body()]):
                # Done the deed, resets the counter.
                return counter.assign(1)

        update = tf.cond(tf.equal(counter, period), _wrapped_body, lambda: counter.assign_add(1))

    return update
